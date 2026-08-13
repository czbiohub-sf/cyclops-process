import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for parallel plotting
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path
import re
from tqdm import tqdm
import argparse
from matplotlib.ticker import ScalarFormatter
import yaml
from joblib import Parallel, delayed

from ops_utils.data.experiment import OpsDataset
from ops_utils.hpc.resource_manager import get_optimal_workers


def get_metric_category(metric_name: str) -> str:
    """
    Determine the category subdirectory for a metric based on its prefix.

    Args:
        metric_name: Name of the metric

    Returns:
        Category name for subdirectory organization
    """
    metric_lower = metric_name.lower()

    # Reconstruction metrics (z-offset, zenith, azimuth, focus, n_subtiles) - check BEFORE track_ prefix
    if "z_offset" in metric_lower or "focus" in metric_lower or "zenith" in metric_lower or "azimuth" in metric_lower or "recon_n_subtiles" in metric_lower:
        return "reconstruction"

    # Stitch metrics - check BEFORE track_ prefix
    if "stitch" in metric_lower:
        return "stitch"

    # SNR/signal quality metrics
    if metric_lower.startswith("snr_") or metric_lower.startswith("mean_snr") or metric_lower.startswith("median_snr"):
        return "snr"

    # Tracking metrics
    if metric_lower.startswith("track_") or metric_lower.startswith("post_track_"):
        return "track"

    # Registration metrics (ISS cycle drift and auto-registration)
    if metric_lower.startswith("iss_cycle_") or metric_lower.startswith("iss_round_") or metric_lower.startswith("iss_to_track") or metric_lower.startswith("pheno_to_track"):
        return "registration"

    # Link pipeline metrics
    if metric_lower.startswith("link_"):
        return "link"

    # Growth effect metrics
    if "growth_effect" in metric_lower:
        return "growth_effect"

    # Entropy metrics (entropy, top_guide_ratio, correlation to reference)
    if "entropy" in metric_lower or "top_guide_ratio" in metric_lower or "correlation" in metric_lower:
        return "entropy"

    # Spatial coherence metrics
    if metric_lower.startswith("sc_"):
        return "spatial_coherence"

    # Flatfield correction QC metrics
    if metric_lower.startswith("flatfield_"):
        return "flatfield"

    # Cell segmentation size/shape metrics
    if metric_lower.startswith("cell_seg_"):
        return "cell_size"

    # ISS quality metrics (including cell/read counts - all ISS-related)
    # This catches: iss_*, num_cells, num_reads, cells_with_*, percent_*, etc.
    if metric_lower.startswith("iss_"):
        return "iss"

    # Default: everything else goes to ISS (cell counts, read stats, etc.)
    return "iss"


def _format_sig_no_sci(value: float, sig: int = 2) -> str:
    """Format a number with 'sig' significant digits, never using scientific notation."""
    try:
        if value == 0 or np.isclose(value, 0.0):
            return "0"
        import math
        decimals = max(0, sig - int(math.floor(math.log10(abs(value)))) - 1)
        return f"{value:.{decimals}f}"
    except Exception:
        return f"{value:.2f}"


def _generate_single_plot(
    metric: str,
    stats: dict,
    experiments_sorted: list,
    well_color_map: dict,
    output_dir: Path,
    experiment: str,
    method: str,
) -> tuple:
    """
    Generate a single metric plot. Returns (metric, success, skip_reason).
    Designed to be called in parallel via joblib.
    """
    # Gather data for this metric
    means, plot_data = [], []
    x_ticks_labels = []

    for exp_name in experiments_sorted:
        if exp_name in stats and metric in stats[exp_name].index:
            metric_series = stats[exp_name].loc[metric]
            plot_data.append((metric_series.values, metric_series.index.tolist()))
            means.append(np.nanmean(metric_series.values))
            x_ticks_labels.append(exp_name)

    if not plot_data:
        return (metric, False, "no_plot_data")

    valid_means_check = [m for m in means if not pd.isna(m)]
    if not valid_means_check:
        return (metric, False, "all_nan_means")

    # Create figure
    fig, (main_ax, violin_ax) = plt.subplots(
        1, 2, figsize=(30, 10), sharey=True,
        gridspec_kw={"width_ratios": [20, 1], "wspace": 0.02},
        constrained_layout=True,
    )

    current_exp_index = (
        x_ticks_labels.index(experiment) if experiment in x_ticks_labels else None
    )

    # Plot individual replicates
    for i, (replicates, wells) in enumerate(plot_data):
        for j, replicate in enumerate(replicates):
            if pd.isna(replicate):
                continue
            color = well_color_map.get(wells[j], "grey")
            main_ax.plot(i, replicate, "o", color=color, alpha=0.8, markersize=7)

    x_positions = list(range(len(plot_data)))

    # Filter out NaN means for hlines
    valid_indices = [i for i, m in enumerate(means) if not pd.isna(m)]
    valid_means_for_hlines = [means[i] for i in valid_indices]
    valid_xmin = [i - 0.4 for i in valid_indices]
    valid_xmax = [i + 0.4 for i in valid_indices]

    if valid_means_for_hlines:
        main_ax.hlines(
            y=valid_means_for_hlines, xmin=valid_xmin, xmax=valid_xmax,
            color="dodgerblue", linewidth=2, alpha=0.7,
        )

    means_array = np.array(means)
    mean_line_handle = main_ax.plot(
        x_positions, means_array, "-", color="dodgerblue",
        linewidth=2, alpha=0.7, label=f"Mean of replicates ({method})",
    )

    if current_exp_index is not None:
        main_ax.axvline(x=current_exp_index, color="red", linestyle="--", linewidth=2, alpha=0.2)

    main_ax.set_title(f"Trend for: {metric}", fontsize=24)
    main_ax.set_xlabel("Experiment", fontsize=18)
    main_ax.set_ylabel(metric, fontsize=18)
    main_ax.set_xticks(range(len(x_ticks_labels)))
    main_ax.set_xticklabels(x_ticks_labels, rotation=45, ha="right")
    main_ax.tick_params(axis="both", which="major", labelsize=14)
    main_ax.grid(axis="y", linestyle="--", alpha=0.7)

    if metric == "post_track_cell_loss_pct":
        main_ax.set_ylim(bottom=0)

    if current_exp_index is not None:
        labels = main_ax.get_xticklabels()
        labels[current_exp_index].set_fontweight("bold")
        labels[current_exp_index].set_fontsize(14)

    # Legend
    well_handles = [
        Line2D([0], [0], marker="o", color="w", label=well,
               markerfacecolor=color, markersize=8)
        for well, color in well_color_map.items()
    ]
    all_handles = well_handles + mean_line_handle
    if current_exp_index is not None:
        all_handles.append(
            Line2D([], [], color="red", linestyle="--", linewidth=2,
                   label="Current Experiment", alpha=0.2)
        )

    main_ax.legend(
        handles=all_handles, loc="upper center", bbox_to_anchor=(0.5, 1.15),
        ncol=len(all_handles), fancybox=True, shadow=True, fontsize=14,
    )

    # Violin plot
    valid_means = [m for m in means if not pd.isna(m)]
    if valid_means:
        parts = violin_ax.violinplot(
            valid_means, vert=True, widths=0.8,
            showmeans=False, showmedians=False, positions=[1],
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("gray")
            pc.set_edgecolor("gray")
            pc.set_alpha(0.2)
            pc.set_linewidth(1)

        jitter = np.random.normal(0, 0.04, size=len(valid_means))
        violin_ax.plot(1 + jitter, valid_means, "o", color="black", alpha=0.4, markersize=2)
        overall_mean = np.mean(valid_means)
        violin_ax.axhline(overall_mean, color="green", linestyle=":", linewidth=1.5)
        violin_ax.text(
            1.1, overall_mean, f" {_format_sig_no_sci(overall_mean, 2)}",
            verticalalignment="bottom", color="green", fontweight="bold", fontsize=18,
        )

        if current_exp_index is not None:
            current_exp_mean = means[current_exp_index]
            if not pd.isna(current_exp_mean):
                violin_ax.axhline(current_exp_mean, color="red", linestyle=":", linewidth=1.5)
                violin_ax.text(
                    0.9, current_exp_mean, f"{_format_sig_no_sci(current_exp_mean, 2)} ",
                    verticalalignment="bottom", horizontalalignment="right",
                    color="red", fontweight="bold", fontsize=18,
                )

    violin_ax.set_title("All Exp.\nMeans", fontsize=15)
    violin_ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    violin_ax.spines["top"].set_visible(False)
    violin_ax.spines["right"].set_visible(False)
    violin_ax.spines["bottom"].set_visible(False)

    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    main_ax.yaxis.set_major_formatter(formatter)

    # Save plot
    sanitized_metric_name = re.sub(r'[\\/:"*?<>|%]+', "_", metric)
    category = get_metric_category(metric)
    category_dir = output_dir / category
    category_dir.mkdir(exist_ok=True, parents=True)

    plot_filename = category_dir / f"{sanitized_metric_name}.png"
    fig.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return (metric, True, None)


def plot_metrics_over_time(
    experiment: str,
    method: str = "mine",
    verbose: bool = False,
    sort_by: str = "date",
):
    """
    Looks over all experiments in the ops folder, finds plate_stats.csv for each,
    and plots the evolution of each metric over time.

    Behavior:
    - Uses the provided 'method' ("probabilistic" or "mine") as the dataset.
    - Experiments are sorted by 'sort_by' parameter: "date" (default, uses date suffix from experiment name) or "name" (alphabetical by ops number).

    For each metric, it generates a line plot where the x-axis represents
    different experiments sorted chronologically. Each experiment has its
    replicate values (from different wells) plotted as individual points,
    and a line connects the mean of these replicates across experiments.
    """
    if method not in {"probabilistic", "mine"}:
        print(
            f"Unsupported method '{method}'. Falling back to 'mine'."
        )
        method = "mine"

    dataset = OpsDataset(experiment, method=method)

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    config_dir = dataset.config_paths["exp_config_dir"]
    config_files = sorted(list(config_dir.glob("ops*.yaml")))

    # list of experiments to exclude
    exp_to_exclude = [
        "ops0094_lala",
        "ops0048_20250617_rnd11_20",
        "ops0080_20250904",
        "ops0073_20250818",
        "ops0079_rnd10_cardboard",
        "ops0079_black2pt0",
        "ops0079_black_oracal",
        "ops0079_blk_anodized",
        "ops0079_delrin",
        "ops0079_milled_acry",
        "ops0079_orig_cap",
        "ops0079_PETG",
        "ops0079_PLA",
        "ops0079_11to20_20251103",
        "ops0083_20251124",
        "ops0092_11to20_20251106",
        "ops0074_20250825_mark"
    ]
    # need to match the exclude name to the config path stem
    # Exclude any config whose filename starts with "<experiment>_"
    if exp_to_exclude:
        excluded_prefixes = {f"{exp}_" for exp in exp_to_exclude}
        config_files = [
            cf
            for cf in config_files
            if not any(cf.name.startswith(pref) for pref in excluded_prefixes)
        ]

    _log(
        f"Found {len(config_files)} experiment configs. Checking for corresponding stats files..."
    )

    stats = {}
    for config_file in tqdm(
        config_files, desc=f"Processing {method} experiments"
    ):
        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)
            exp_name = config_data.get("experiment_name")

        if not exp_name:
            _log(
                f"--> Skipping config file {config_file.name}: does not contain 'experiment_name'."
            )
            continue

        exp_dataset = OpsDataset(exp_name, method=method)
        stats_file = exp_dataset.metrics_paths["statistics"]
        if stats_file.exists():
            try:
                stats_df = pd.read_csv(stats_file, index_col=0)
                # Use dropna(how='all') to keep metrics that have some valid values
                # (don't drop rows just because one well is missing data)
                stats_df = stats_df.apply(pd.to_numeric, errors="coerce").dropna(how='all')
                if not stats_df.empty:
                    stats[exp_name] = stats_df
            except Exception as e:
                _log(f"Could not process {stats_file} for experiment {exp_name}: {e}")
        else:
            _log(f"--> No stats file found for configured experiment '{exp_name}'.")

    if not stats:
        if verbose:
            print(
                "No valid 'plate_stats.csv' files found for any configured experiments."
            )
        # succinct summary when not verbose
        summary = (
            f"[metrics_over_time] method={method}"
            f" configs={len(config_files)} exps=0"
            f" plots=0 out={dataset.metrics_paths['over_time']}"
        )
        if not verbose:
            print(summary)
        return

    if experiment in stats:
        metric_names = stats[experiment].index.tolist()
        _log(f"Using metrics from current experiment '{experiment}' as the reference.")
    else:
        latest_exp_with_stats = sorted(stats.keys())[-1]
        metric_names = stats[latest_exp_with_stats].index.tolist()
        _log(f"Warning: Current experiment '{experiment}' lacks a stats file.")
        _log(
            f"Using metrics from the most recent experiment with stats ('{latest_exp_with_stats}') as the reference."
        )

    _log(f"METRIC NAMES: {metric_names}")

    all_wells = set(well for exp in stats.values() for well in exp.columns)
    sorted_wells = sorted(list(all_wells))
    color_map = plt.colormaps.get("viridis")
    well_color_map = {
        well: color_map(i / max(len(sorted_wells) - 1, 1)) for i, well in enumerate(sorted_wells)
    }

    output_dir = dataset.metrics_paths["over_time"]
    output_dir.mkdir(exist_ok=True, parents=True)

    # Sort experiments based on sort_by parameter
    def extract_date(exp_name: str) -> str:
        """Extract date suffix from experiment name (e.g., 'ops0061_20250728' -> '20250728')"""
        match = re.search(r"_(\d{8})$", exp_name)
        return match.group(1) if match else exp_name

    if sort_by == "date":
        experiments_sorted = sorted(stats.keys(), key=extract_date)
    else:
        experiments_sorted = sorted(stats.keys())

    # Determine optimal number of workers for CPU-bound plotting
    n_workers = get_optimal_workers(use_gpu=False, model_ram_gb=0.5, data_ram_gb=0.5, verbose=verbose)

    # Generate plots in parallel
    results = Parallel(n_jobs=n_workers, backend="loky")(
        delayed(_generate_single_plot)(
            metric, stats, experiments_sorted, well_color_map, output_dir, experiment, method
        )
        for metric in tqdm(metric_names, desc="Generating plots")
    )

    # Process results
    plots_generated = sum(1 for _, success, _ in results if success)
    skipped_metrics = [(m, reason, None) for m, success, reason in results if not success]

    # Final succinct summary when not verbose
    summary = (
        f"[metrics_over_time] method={method}"
        f" configs={len(config_files)} exps={len(stats)}"
        f" plots={plots_generated} skipped={len(skipped_metrics)} out={output_dir}"
    )
    if verbose:
        print(f"\nAll plots saved in {output_dir}")
        print(summary)
    else:
        print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot metrics over time for all experiments."
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
        help="Method to plot (default: mine).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress; otherwise prints a succinct summary at the end.",
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        default="date",
        choices=["date", "name"],
        help="Sort experiments by date (from experiment name suffix) or name (alphabetical by ops number). Default: date.",
    )
    args = parser.parse_args()

    plot_metrics_over_time(
        args.experiment,
        method=args.method,
        verbose=args.verbose,
        sort_by=args.sort_by,
    )

    # to run: python -m cyclops_process.metrics.metrics_over_time ops0061_20250728 --method probabilistic
