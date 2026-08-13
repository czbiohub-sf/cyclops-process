import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path
import re
from tqdm import tqdm
import argparse
from matplotlib.ticker import ScalarFormatter
import yaml

from ops_utils.data.experiment import OpsDataset
from iohub.ngff import open_ome_zarr
from cyclops_process.metrics.plate_stats.match_reads import (
    _get_effective_iss_rounds,
    match_reads,
)


def _select_rounds(barcode: str, rounds: list[int]) -> str:
    """Reduce a barcode to just the positions in *rounds*.

    Mirrors the position selection ``match_reads`` applies to both the reads and
    the codebook, so barcodes are keyed consistently when non-contiguous rounds
    are in play.
    """
    return "".join([barcode[i] for i in rounds if i < len(barcode)])


def get_top_gene_cell_count(
    experiment: str,
    method: str = "mine",
    iss_rounds: list[int] | None = None,
    confidence_threshold: float = 0.95,
    failed_rounds_by_well: dict[str, list[int]] | None = None,
) -> tuple[str, int, dict]:
    """
    Get the top gene by cell count for a given experiment.

    Args:
        iss_rounds: List of ISS round indices to use. If None, defaults to first 10 rounds.
        failed_rounds_by_well: Dictionary mapping wells to lists of failed round indices
            to exclude.

    Returns:
        tuple: (gene_name, total_cell_count, well_counts_dict)
        where well_counts_dict maps well names to cell counts for that gene
    """
    if iss_rounds is None:
        iss_rounds = list(range(10))

    dataset = OpsDataset(experiment, method=method)

    try:
        codebook_db = dataset.load_codebook()
        gene_index_db = dataset.load_gene_index()
    except Exception as e:
        return None, 0, {}

    try:
        seg_store = open_ome_zarr(dataset.store_paths["iss_segmentation"], mode="r")
        position_list = [a[0] for a in seg_store.positions()]
    except Exception as e:
        return None, 0, {}

    if not {"barcode", "gene_name"}.issubset(gene_index_db.columns):
        return None, 0, {}

    all_good_reads = []
    well_read_map = {}  # Track which reads came from which well

    for pos in position_list:
        # Per-well rounds, so wells with dropped rounds are matched on the
        # positions they actually have (same resolution the other ISS metrics use).
        well_iss_rounds = _get_effective_iss_rounds(
            iss_rounds, pos, failed_rounds_by_well
        )

        try:
            read_db = pd.read_csv(dataset.append_well("reads", pos))
            if method == "probabilistic":
                good_reads = read_db[read_db["confidence"] >= confidence_threshold]
            else:  # 'mine'
                good_reads = match_reads(
                    read_db,
                    codebook_db,
                    iss_rounds=well_iss_rounds,
                    well_name=pos,
                    failed_rounds_by_well=failed_rounds_by_well,
                )

            if good_reads.empty:
                continue

            # Key both sides of the gene lookup on this well's rounds.
            gene_lookup = {
                _select_rounds(str(bc), well_iss_rounds): gene
                for bc, gene in zip(
                    gene_index_db["barcode"], gene_index_db["gene_name"]
                )
            }
            good_reads = good_reads.copy()
            good_reads["gene_name"] = (
                good_reads["barcode"]
                .astype(str)
                .map(lambda bc: gene_lookup.get(_select_rounds(bc, well_iss_rounds)))
            )
            good_reads.dropna(subset=["gene_name"], inplace=True)

            if not good_reads.empty:
                all_good_reads.append(good_reads)
                well_read_map[pos] = good_reads
        except FileNotFoundError:
            continue

    if not all_good_reads:
        return None, 0, {}

    # Pool all reads
    pooled_reads = pd.concat(all_good_reads, ignore_index=True)

    if pooled_reads.empty:
        return None, 0, {}

    # Get top gene by cell count
    cells_per_gene = pooled_reads.groupby("gene_name")["cell"].nunique()
    if cells_per_gene.empty:
        return None, 0, {}

    top_gene = cells_per_gene.idxmax()
    total_count = cells_per_gene.max()

    # Get per-well counts for the top gene ("gene_name" already resolved above,
    # using each well's own rounds)
    well_counts = {}
    for well, reads_df in well_read_map.items():
        top_gene_reads = reads_df[reads_df["gene_name"] == top_gene]
        if not top_gene_reads.empty:
            well_counts[well] = top_gene_reads["cell"].nunique()

    return top_gene, int(total_count), well_counts


def plot_top_gene_over_time(
    experiment: str,
    method: str = "mine",
    iss_rounds: list[int] | None = None,
    confidence_threshold: float = 0.95,
    compare: bool = False,
    verbose: bool = False,
    failed_rounds_by_well: dict[str, list[int]] | None = None,
):
    """
    Plot the cell count of the top gene (by cell count) across all experiments over time.

    Similar to metrics_over_time.py but specifically for tracking the best-performing gene.
    """
    if iss_rounds is None:
        iss_rounds = list(range(10))

    primary_method = method
    if primary_method not in {"probabilistic", "mine"}:
        print(
            f"Unsupported method '{primary_method}'. Falling back to 'mine'."
        )
        primary_method = "mine"
    compare_method = "mine" if primary_method == "probabilistic" else "probabilistic"

    dataset = OpsDataset(experiment, method=primary_method)

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    config_dir = dataset.config_paths["exp_config_dir"]
    config_files = sorted(list(config_dir.glob("ops*.yaml")))

    # Exclude list
    exp_to_exclude = [
        "ops0080_20250904",
        "ops0079_rnd10_cardboard",
        "ops0079_black2pt0",
        "ops0079_black_oracal",
        "ops0079_blk_anodized",
        "ops0079_delrin",
        "ops0079_milled_acry",
        "ops0079_orig_cap",
        "ops0079_PETG",
        "ops0079_PLA",
    ]
    if exp_to_exclude:
        excluded_prefixes = {f"{exp}_" for exp in exp_to_exclude}
        config_files = [
            cf
            for cf in config_files
            if not any(cf.name.startswith(pref) for pref in excluded_prefixes)
        ]

    _log(f"Found {len(config_files)} experiment configs. Processing top genes...")

    primary_data = {}  # {exp_name: (gene_name, total_count, well_counts)}
    for config_file in tqdm(
        config_files, desc=f"Processing {primary_method} experiments"
    ):
        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)
            exp_name = config_data.get("experiment_name")

        if not exp_name:
            continue

        try:
            gene_name, total_count, well_counts = get_top_gene_cell_count(
                exp_name,
                method=primary_method,
                iss_rounds=iss_rounds,
                confidence_threshold=confidence_threshold,
                failed_rounds_by_well=failed_rounds_by_well,
            )
            if gene_name:
                primary_data[exp_name] = (gene_name, total_count, well_counts)
        except Exception as e:
            _log(f"Error processing {exp_name}: {e}")

    compare_data = {}
    if compare:
        _log(f"Comparison mode enabled: Loading data for '{compare_method}' analysis.")
        for config_file in tqdm(
            config_files, desc=f"Processing {compare_method} experiments"
        ):
            with open(config_file, "r") as f:
                config_data = yaml.safe_load(f)
                exp_name = config_data.get("experiment_name")

            if not exp_name:
                continue

            try:
                gene_name, total_count, well_counts = get_top_gene_cell_count(
                    exp_name,
                    method=compare_method,
                    iss_rounds=iss_rounds,
                    confidence_threshold=confidence_threshold,
                    failed_rounds_by_well=failed_rounds_by_well,
                )
                if gene_name:
                    compare_data[exp_name] = (gene_name, total_count, well_counts)
            except Exception as e:
                _log(f"Error processing {exp_name} with {compare_method}: {e}")

    if not primary_data:
        print(f"No valid top gene data found for {primary_method} method.")
        return

    # Set up colors for wells
    all_wells = set()
    for _, _, well_counts in primary_data.values():
        all_wells.update(well_counts.keys())
    sorted_wells = sorted(list(all_wells))
    color_map = plt.colormaps.get("Set1")
    well_color_map = {
        well: color_map(i % color_map.N) for i, well in enumerate(sorted_wells)
    }

    output_dir = dataset.metrics_paths["over_time"]
    output_dir.mkdir(exist_ok=True, parents=True)

    experiments_sorted = sorted(primary_data.keys())

    # Create the plot
    fig, (main_ax, violin_ax) = plt.subplots(
        1,
        2,
        figsize=(20, 10),
        sharey=True,
        gridspec_kw={"width_ratios": [20, 1], "wspace": 0.02},
        constrained_layout=True,
    )

    primary_means = []
    primary_plot_data = []
    compare_means = []
    compare_plot_data = []
    x_ticks_labels = []
    gene_labels = []

    for exp_name in experiments_sorted:
        gene_name, total_count, well_counts = primary_data[exp_name]

        # Prepare replicate data
        well_names = list(well_counts.keys())
        well_values = list(well_counts.values())

        primary_plot_data.append((well_values, well_names))
        primary_means.append(np.mean(well_values))
        x_ticks_labels.append(exp_name)
        gene_labels.append(gene_name)

        if compare and exp_name in compare_data:
            gene_name_cmp, total_count_cmp, well_counts_cmp = compare_data[exp_name]
            well_names_cmp = list(well_counts_cmp.keys())
            well_values_cmp = list(well_counts_cmp.values())
            compare_plot_data.append((well_values_cmp, well_names_cmp))
            compare_means.append(np.mean(well_values_cmp))
        else:
            compare_plot_data.append(([], []))
            compare_means.append(np.nan)

    if not primary_plot_data:
        print("No data to plot.")
        plt.close()
        return

    current_exp_index = (
        x_ticks_labels.index(experiment) if experiment in x_ticks_labels else None
    )

    # Plot primary method replicates
    for i, (replicates, wells) in enumerate(primary_plot_data):
        for j, replicate in enumerate(replicates):
            color = well_color_map.get(wells[j], "grey")
            main_ax.plot(i, replicate, "o", color=color, alpha=0.8, markersize=7)

    # Plot compare method replicates
    if compare:
        for i, (replicates, wells) in enumerate(compare_plot_data):
            if not any(replicates):
                continue
            for j, replicate in enumerate(replicates):
                color = well_color_map.get(wells[j], "grey")
                main_ax.plot(i, replicate, "x", color=color, alpha=0.8, markersize=7)

    # Plot mean lines
    x_positions = range(len(primary_plot_data))
    main_ax.hlines(
        y=primary_means,
        xmin=[i - 0.4 for i in x_positions],
        xmax=[i + 0.4 for i in x_positions],
        color="dodgerblue",
        linewidth=2,
        alpha=0.7,
    )
    mean_line_handle = main_ax.plot(
        x_positions,
        primary_means,
        "-",
        color="dodgerblue",
        linewidth=2,
        alpha=0.7,
        label=f"Mean of replicates ({primary_method})",
    )

    if compare:
        main_ax.hlines(
            y=compare_means,
            xmin=[i - 0.4 for i in x_positions],
            xmax=[i + 0.4 for i in x_positions],
            color="green",
            linewidth=2,
            alpha=0.7,
        )
        mean_line_handle_cmp = main_ax.plot(
            x_positions,
            compare_means,
            "--",
            color="green",
            linewidth=2,
            alpha=0.7,
            label=f"Mean of replicates ({compare_method})",
        )

    # Mark current experiment
    if current_exp_index is not None:
        main_ax.axvline(
            x=current_exp_index, color="red", linestyle="--", linewidth=2, alpha=0.2
        )

    main_ax.set_title(
        f"Top Gene Cell Count Over Time\nMethod: {primary_method}", fontsize=24
    )
    main_ax.set_xlabel("Experiment", fontsize=18)
    main_ax.set_ylabel("Number of Cells (Top Gene)", fontsize=18)
    main_ax.set_xticks(range(len(x_ticks_labels)))

    # Create x-tick labels with gene names on same line
    xtick_labels_with_genes = [
        f"{exp} ({gene})" for exp, gene in zip(x_ticks_labels, gene_labels)
    ]
    main_ax.set_xticklabels(
        xtick_labels_with_genes, rotation=90, ha="center", fontsize=8
    )
    main_ax.tick_params(axis="both", which="major", labelsize=14)
    main_ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Add more y-axis tick marks
    main_ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=15, integer=True))

    # Bold current experiment
    if current_exp_index is not None:
        labels = main_ax.get_xticklabels()
        labels[current_exp_index].set_fontweight("bold")
        labels[current_exp_index].set_fontsize(12)

    # Legend
    well_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=well,
            markerfacecolor=color,
            markersize=8,
        )
        for well, color in well_color_map.items()
    ]
    all_handles = well_handles + mean_line_handle
    if compare:
        all_handles.extend(mean_line_handle_cmp)
    if current_exp_index is not None:
        all_handles.append(
            Line2D(
                [],
                [],
                color="red",
                linestyle="--",
                linewidth=2,
                label="Current Experiment",
                alpha=0.2,
            )
        )

    main_ax.legend(
        handles=all_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=min(len(all_handles), 6),
        fancybox=True,
        shadow=True,
        fontsize=10,
    )

    # Violin plot
    if primary_means:
        parts = violin_ax.violinplot(
            primary_means,
            vert=True,
            widths=0.8,
            showmeans=False,
            showmedians=False,
            positions=[1],
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("gray")
            pc.set_edgecolor("gray")
            pc.set_alpha(0.2)
            pc.set_linewidth(1)

        jitter = np.random.normal(0, 0.04, size=len(primary_means))
        violin_ax.plot(
            1 + jitter, primary_means, "o", color="black", alpha=0.4, markersize=2
        )
        overall_mean = np.mean(primary_means)
        violin_ax.axhline(overall_mean, color="green", linestyle=":", linewidth=1.5)
        violin_ax.text(
            1.1,
            overall_mean,
            f" {overall_mean:.0f}",
            verticalalignment="bottom",
            color="green",
            fontweight="bold",
            fontsize=18,
        )

        if current_exp_index is not None:
            current_exp_mean = primary_means[current_exp_index]
            violin_ax.axhline(
                current_exp_mean, color="red", linestyle=":", linewidth=1.5
            )
            violin_ax.text(
                0.9,
                current_exp_mean,
                f"{current_exp_mean:.0f} ",
                verticalalignment="bottom",
                horizontalalignment="right",
                color="red",
                fontweight="bold",
                fontsize=18,
            )

    if compare and any(~np.isnan(compare_means)):
        compare_means_clean = np.array(compare_means)[~np.isnan(compare_means)]
        violin_ax.violinplot(
            compare_means_clean,
            vert=True,
            positions=[1],
            widths=0.8,
            showmeans=False,
            showmedians=False,
        )
        jitter_cmp = np.random.normal(0, 0.04, size=len(compare_means_clean))
        violin_ax.plot(
            1 + jitter_cmp,
            compare_means_clean,
            "x",
            color="green",
            alpha=0.4,
            markersize=2,
        )
        overall_mean_cmp = np.mean(compare_means_clean)
        violin_ax.axhline(
            overall_mean_cmp, color="dodgerblue", linestyle=":", linewidth=1.5
        )
        violin_ax.text(
            1.1,
            overall_mean_cmp,
            f" {overall_mean_cmp:.0f}",
            verticalalignment="top",
            color="dodgerblue",
            fontweight="bold",
            fontsize=18,
        )

    violin_ax.set_title("All Exp.\\nMeans", fontsize=15)
    violin_ax.tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    violin_ax.spines["top"].set_visible(False)
    violin_ax.spines["right"].set_visible(False)
    violin_ax.spines["bottom"].set_visible(False)

    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    main_ax.yaxis.set_major_formatter(formatter)

    # Save plot
    plot_filename = output_dir / "top_gene_cell_count_over_time.png"
    fig.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(
        f"[top_gene_over_time] method={primary_method}, compare={compare} "
        f"experiments={len(primary_data)} plot saved to {plot_filename}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot top gene cell count over time for all experiments."
    )
    parser.add_argument(
        "experiment",
        type=str,
        help="The experiment name to use for determining the output directory.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="mine",
        choices=["probabilistic", "mine"],
        help="Primary method to plot (default: mine).",
    )
    parser.add_argument(
        "--iss-rounds",
        type=int,
        nargs="+",
        default=None,
        help="ISS round indices to use (default: first 10 rounds).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.95,
        help="Confidence threshold for probabilistic method (default: 0.95).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also plot the other method for comparison.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress.",
    )
    args = parser.parse_args()

    plot_top_gene_over_time(
        args.experiment,
        method=args.method,
        iss_rounds=args.iss_rounds,
        confidence_threshold=args.confidence_threshold,
        compare=args.compare,
        verbose=args.verbose,
    )

# Usage: python -m cyclops_process.metrics.top_gene_over_time ops0061_20250728 --method probabilistic --compare
