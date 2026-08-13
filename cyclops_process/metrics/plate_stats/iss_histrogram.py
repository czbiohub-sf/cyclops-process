import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from cyclops_utils.data.experiment import OpsDataset
from iohub.ngff import open_ome_zarr
from cyclops_process.metrics.plate_stats.match_reads import (
    _get_effective_iss_rounds,
    match_reads,
)
from cyclops_process.paths import BASE_PATH

def save_filtered_frequency_table(
    experiment: str,
    method: str = "mine",
    confidence_threshold: float = 0.95,
    iss_rounds: list[int] | None = None,
    failed_rounds_by_well: dict[str, list[int]] | None = None,
) -> None:
    """
    Save frequency_table_filtered.csv with only gene-matched reads.

    Same format as frequency_table.csv but restricted to reads that mapped
    to a gene in the codebook (i.e., what actually ends up in the cells-per-gene
    histogram). Output: barcode, gene_id, count (one row per unique barcode+gene).
    """
    if iss_rounds is None:
        iss_rounds = list(range(10))

    dataset = OpsDataset(experiment, method=method)
    codebook_db = dataset.load_codebook()

    if "gene_id" not in codebook_db.columns:
        print(f"[{experiment}] No 'gene_id' in codebook. Skipping.")
        return

    # Auto-load failed_rounds from config if not provided — critical so we
    # trim barcodes to the same effective rounds the linking pipeline uses.
    if failed_rounds_by_well is None:
        try:
            import yaml
            from pathlib import Path
            if Path(dataset.failed_rounds).exists():
                with open(dataset.failed_rounds) as f:
                    cfg = yaml.safe_load(f) or {}
                ops_key = experiment.split("_")[0]
                exp_cfg = cfg.get(ops_key) or cfg.get(experiment)
                # Config may be wrapped in {"failed_rounds_by_well": {...}}
                if isinstance(exp_cfg, dict) and "failed_rounds_by_well" in exp_cfg:
                    failed_rounds_by_well = exp_cfg["failed_rounds_by_well"]
                else:
                    failed_rounds_by_well = exp_cfg
                if failed_rounds_by_well:
                    print(f"[{experiment}] Loaded failed_rounds: {failed_rounds_by_well}")
        except Exception as e:
            print(f"[{experiment}] Could not load failed_rounds config: {e}")

    try:
        seg_store = open_ome_zarr(dataset.store_paths["iss_segmentation"], mode="r")
        position_list = [a[0] for a in seg_store.positions()]
    except Exception as e:
        print(f"[{experiment}] Could not load segmentation: {e}. Skipping.")
        return

    all_reads = []
    for pos in position_list:
        well_iss_rounds = _get_effective_iss_rounds(iss_rounds, pos, failed_rounds_by_well)

        try:
            read_db = pd.read_csv(dataset.append_well("reads", pos))
            if method == "probabilistic":
                good_reads = read_db[read_db["confidence"] >= confidence_threshold].copy()
            else:
                matched = match_reads(read_db, codebook_db, iss_rounds=well_iss_rounds, well_name=pos, failed_rounds_by_well=failed_rounds_by_well, debug=False)
                good_reads = matched.copy() if not matched.empty else matched

            if good_reads.empty or "barcode" not in good_reads.columns:
                continue

            good_reads["barcode"] = good_reads["barcode"].apply(
                lambda bc: "".join([bc[i] for i in well_iss_rounds if i < len(bc)])
            )
            # Build globally-unique cell ID (cell IDs are position-specific)
            good_reads["unique_cell_id"] = pos + "_" + good_reads["cell"].astype(str)

            # Map to gene_id via well-specific codebook trimming
            cb = codebook_db.copy()
            cb["trim_sgRNA"] = cb["sgRNA"].apply(lambda a: "".join([a[i] for i in well_iss_rounds]))
            cb_unique = cb.drop_duplicates(subset=["trim_sgRNA"], keep="first")
            sgRNA_to_gene_map = dict(zip(cb_unique["trim_sgRNA"], cb_unique["gene_id"]))
            good_reads["gene_id"] = good_reads["barcode"].map(sgRNA_to_gene_map)
            all_reads.append(good_reads)
        except FileNotFoundError:
            continue

    if not all_reads:
        print(f"[{experiment}] No reads found. Skipping.")
        return

    pooled = pd.concat(all_reads, ignore_index=True)
    pooled.dropna(subset=["gene_id"], inplace=True)

    if pooled.empty:
        print(f"[{experiment}] No gene-matched reads. Skipping.")
        return

    # Map codebook gene_id -> perturbation (and optionally a secondary id column).
    # For regular geneKO:    perturbation=gene_name, gene_id=numeric NCBI_ID (kept as-is)
    # For custom-perturbation libs: perturbation=<configured column>, gene_id=gene_target
    pooled["perturbation"] = pooled["gene_id"]
    secondary_map = None
    try:
        gene_index_db = dataset.load_gene_index()
        gene_id_col = _get_col(gene_index_db, ["gene_id", "NCBI_ID"])
        pert_col = _get_col(
            gene_index_db,
            ([dataset.gene_name_output_column] if dataset.gene_name_output_column else [])
            + ["Gene name", "dep_map_gene_name", "gene_name"],
        )
        if gene_id_col and pert_col:
            gene_id_to_pert = dict(zip(gene_index_db[gene_id_col], gene_index_db[pert_col]))
            pooled["perturbation"] = pooled["gene_id"].map(gene_id_to_pert).fillna(pooled["gene_id"])
        # For libraries with a configured secondary column (e.g. gene_target),
        # replace the numeric gene_id with that secondary identifier
        if dataset.iss_secondary_gene_column and gene_id_col:
            sec_col = _get_col(gene_index_db, [dataset.iss_secondary_gene_column])
            if sec_col:
                secondary_map = dict(zip(gene_index_db[gene_id_col], gene_index_db[sec_col]))
                pooled["gene_id"] = pooled["gene_id"].map(secondary_map).fillna(pooled["gene_id"])
    except Exception as e:
        print(f"[{experiment}] Could not load gene_index: {e}")

    # Deduplicate: assign each cell to a single (barcode, gene_id) by majority vote
    # (the combo with the most reads in that cell). This makes sum(count) == unique cells.
    pooled["_n"] = 1
    per_cell_best = (
        pooled.groupby(["unique_cell_id", "barcode", "perturbation", "gene_id"], as_index=False)["_n"]
        .sum()
        .sort_values(["unique_cell_id", "_n"], ascending=[True, False])
        .drop_duplicates(subset="unique_cell_id", keep="first")
    )
    filtered_freq = (
        per_cell_best.groupby(["barcode", "perturbation", "gene_id"], as_index=False)["unique_cell_id"]
        .nunique()
        .rename(columns={"unique_cell_id": "count"})
        .sort_values("count", ascending=False)
    )
    filtered_freq = filtered_freq[["barcode", "perturbation", "gene_id", "count"]]

    out_path = dataset.metrics_paths["frequency_table"].parent / "frequency_table_filtered.csv"
    if not out_path.parent.exists():
        print(f"[{experiment}] Output dir {out_path.parent} does not exist "
              f"(ISS never run with method={method}?). Skipping.")
        return
    filtered_freq.to_csv(out_path, index=False)
    print(f"[{experiment}] Saved {len(filtered_freq)} rows to {out_path}")


def _get_col(df, candidates: list[str]) -> str | None:
    """Find first matching column name from candidates list."""
    if df is None:
        return None
    for col in candidates:
        if col in df.columns:
            return col
    return None



def plot_cells_per_gene_histogram(
    experiment: str,
    method: str = "mine",
    confidence_threshold: float = 0.95,
    iss_rounds: list[int] | None = None,
    failed_rounds_by_well: dict[str, list[int]] | None = None,
) -> None:
    """
    Generates and saves a histogram of the number of cells per gene, pooled across all wells.

    Args:
        iss_rounds: List of ISS round indices to use. If None, defaults to first 10 rounds.
        method: Base calling method ('mine' or 'probabilistic').
        confidence_threshold: Confidence threshold for probabilistic method.
        failed_rounds_by_well: Dictionary mapping wells to lists of failed round indices to exclude
    """
    # Backward compatibility
    if iss_rounds is None:
        iss_rounds = list(range(10))

    print("--- Generating pooled cells-per-gene histogram ---")
    dataset = OpsDataset(experiment, method=method)
    codebook_db = dataset.load_codebook()

    try:
        seg_store = open_ome_zarr(dataset.store_paths["iss_segmentation"], mode="r")
        position_list = [a[0] for a in seg_store.positions()]
    except Exception as e:
        print(
            f"Could not load segmentation data to get wells: {e}. Aborting histogram generation."
        )
        return

    all_good_reads = []
    for pos in position_list:
        # Filter out failed rounds for this well
        well_iss_rounds = _get_effective_iss_rounds(
            iss_rounds, pos, failed_rounds_by_well
        )

        try:
            read_db = pd.read_csv(dataset.append_well("reads", pos))
            if method == "probabilistic":
                good_reads = read_db[read_db["confidence"] >= confidence_threshold].copy()
            else:  # 'mine'
                matched = match_reads(read_db, codebook_db, iss_rounds=well_iss_rounds, well_name=pos, failed_rounds_by_well=failed_rounds_by_well, debug=False)
                good_reads = matched.copy() if not matched.empty else matched

            # Add position identifier for unique cell tracking across wells
            good_reads["position"] = pos
            good_reads["unique_cell_id"] = good_reads["position"].astype(str) + "_" + good_reads["cell"].astype(str)

            # Filter barcodes to effective positions and add gene mapping for this well
            if not good_reads.empty and "barcode" in good_reads.columns:
                # Filter barcodes
                good_reads["barcode"] = good_reads["barcode"].apply(
                    lambda barcode: "".join([barcode[i] for i in well_iss_rounds if i < len(barcode)])
                )

                # Create well-specific gene mapping
                cb = codebook_db.copy()
                cb["trim_sgRNA"] = cb["sgRNA"].apply(lambda a: "".join([a[i] for i in well_iss_rounds]))
                cb_unique = cb.drop_duplicates(subset=["trim_sgRNA"], keep="first")
                sgRNA_to_gene_map = dict(zip(cb_unique["trim_sgRNA"], cb_unique["gene_id"]))

                # Map barcodes to genes
                good_reads["gene_id"] = good_reads["barcode"].map(sgRNA_to_gene_map)

            all_good_reads.append(good_reads)
        except FileNotFoundError:
            print(f"Reads file for well {pos} not found. Skipping.")
            continue

    if not all_good_reads:
        print("No good reads found across any wells. Aborting histogram generation.")
        return

    # --- Data Aggregation ---
    pooled_reads = pd.concat(all_good_reads, ignore_index=True)
    # Count unique cells across all wells (cell IDs are position-specific)
    total_cells_with_reads = pooled_reads["unique_cell_id"].nunique()

    if "gene_id" not in codebook_db.columns:
        print("Could not find 'gene_id' in codebook. Aborting histogram generation.")
        return
    pooled_reads.dropna(subset=["gene_id"], inplace=True)

    if pooled_reads.empty:
        print("No reads could be matched to a gene. Aborting histogram generation.")
        return

    # Count unique cells per gene (using globally unique cell IDs)
    cells_per_gene = pooled_reads.groupby("gene_id")["unique_cell_id"].nunique()

    # Save filtered frequency table (barcode, perturbation, gene_id, count).
    # For libraries with a secondary identifier, gene_id holds the gene_target identifier.
    pooled_reads["perturbation"] = pooled_reads["gene_id"]
    try:
        gene_index_db = dataset.load_gene_index()
        gene_id_col = _get_col(gene_index_db, ["gene_id", "NCBI_ID"])
        pert_col = _get_col(
            gene_index_db,
            ([dataset.gene_name_output_column] if dataset.gene_name_output_column else [])
            + ["Gene name", "dep_map_gene_name", "gene_name"],
        )
        if gene_id_col and pert_col:
            gene_id_to_pert = dict(zip(gene_index_db[gene_id_col], gene_index_db[pert_col]))
            pooled_reads["perturbation"] = pooled_reads["gene_id"].map(gene_id_to_pert).fillna(pooled_reads["gene_id"])
        if dataset.iss_secondary_gene_column and gene_id_col:
            sec_col = _get_col(gene_index_db, [dataset.iss_secondary_gene_column])
            if sec_col:
                secondary_map = dict(zip(gene_index_db[gene_id_col], gene_index_db[sec_col]))
                pooled_reads["gene_id"] = pooled_reads["gene_id"].map(secondary_map).fillna(pooled_reads["gene_id"])
    except Exception as e:
        print(f"Could not load gene_index: {e}")

    # Deduplicate: assign each cell to a single (barcode, gene_id) by majority vote
    # so that sum(count) equals unique cells with reads (matching the histogram legend).
    pooled_reads["_n"] = 1
    per_cell_best = (
        pooled_reads.groupby(["unique_cell_id", "barcode", "perturbation", "gene_id"], as_index=False)["_n"]
        .sum()
        .sort_values(["unique_cell_id", "_n"], ascending=[True, False])
        .drop_duplicates(subset="unique_cell_id", keep="first")
    )
    filtered_freq = (
        per_cell_best.groupby(["barcode", "perturbation", "gene_id"], as_index=False)["unique_cell_id"]
        .nunique()
        .rename(columns={"unique_cell_id": "count"})
        .sort_values("count", ascending=False)
    )
    filtered_freq = filtered_freq[["barcode", "perturbation", "gene_id", "count"]]
    filtered_freq_path = dataset.metrics_paths["frequency_table"].parent / "frequency_table_filtered.csv"
    filtered_freq.to_csv(filtered_freq_path, index=False)
    print(f"Saved filtered frequency table to {filtered_freq_path}")

    # --- Statistics Calculation ---
    mean_cpg = cells_per_gene.mean()
    median_cpg = cells_per_gene.median()
    std_cpg = cells_per_gene.std()

    # Define a dynamic x-limit to exclude extreme outliers and make the plot readable
    xlim_upper = cells_per_gene.quantile(0.95)

    # --- Font sizes ---
    TITLE_SIZE = 20
    LABEL_SIZE = 16
    TICK_SIZE = 12
    LEGEND_SIZE = 12

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        cells_per_gene,
        bins=100,
        range=(0, xlim_upper),
        alpha=0.7,
        label="Cells per Gene Distribution",
    )

    # Adding vertical lines for stats
    ax.axvline(
        mean_cpg,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Mean: {mean_cpg:.2f}",
    )
    ax.axvline(
        median_cpg,
        color="green",
        linestyle="dashed",
        linewidth=2,
        label=f"Median: {median_cpg:.2f}",
    )

    fig.suptitle("Histogram of Cells per Gene (All Wells Pooled)", fontsize=TITLE_SIZE)
    ax.set_title(experiment, fontsize=LABEL_SIZE - 4)
    ax.set_xlabel("Number of Cells", fontsize=LABEL_SIZE)
    ax.set_ylabel("Number of Genes", fontsize=LABEL_SIZE)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim(0, xlim_upper)
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)

    # --- Legend ---
    legend_text = (
        f"Total Cells w/ Reads: {total_cells_with_reads}\n"
        f"Mean Cells/Gene: {mean_cpg:.2f}\n"
        f"Median Cells/Gene: {median_cpg:.2f}\n"
        f"Std Dev: {std_cpg:.2f}"
    )
    # Place legend text box
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.95,
        0.95,
        legend_text,
        transform=ax.transAxes,
        fontsize=LEGEND_SIZE,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
    )

    ax.legend(loc="upper left", fontsize=LEGEND_SIZE)

    plt.tight_layout()
    plt.savefig(dataset.metrics_paths["cells_per_gene_histogram"], dpi=300)
    print(
        f"Saved cells-per-gene histogram to {dataset.metrics_paths['cells_per_gene_histogram']}"
    )





def plot_top_genes_by_cell_count(
    experiment: str,
    iss_rounds: list[int] | None = None,
    method: str = "mine",
    confidence_threshold: float = 0.95,
    top_n: int = 50,
    failed_rounds_by_well: dict[str, list[int]] | None = None,
) -> None:
    """
    Generates and saves a bar plot of the top N genes by cell count, pooled across all wells.
    Y-axis: number of cells, X-axis: genes ordered by descending cell count.

    Args:
        iss_rounds: List of ISS round indices to use. If None, defaults to first 10 rounds.
        method: Base calling method ('mine' or 'probabilistic').
        confidence_threshold: Confidence threshold for probabilistic method.
        top_n: Number of top genes to display.
        failed_rounds_by_well: Dictionary mapping wells to lists of failed round indices to exclude
    """
    # Backward compatibility
    if iss_rounds is None:
        iss_rounds = list(range(10))

    print(f"--- Generating top {top_n} genes by cell count bar plot ---")
    dataset = OpsDataset(experiment, method=method)
    codebook_db = dataset.load_codebook()

    # Load gene index to get gene names
    try:
        gene_index_db = dataset.load_gene_index()
    except Exception as e:
        print(f"Could not load gene index: {e}. Using gene_id instead of gene_name.")
        gene_index_db = None

    try:
        seg_store = open_ome_zarr(dataset.store_paths["iss_segmentation"], mode="r")
        position_list = [a[0] for a in seg_store.positions()]
    except Exception as e:
        print(
            f"Could not load segmentation data to get wells: {e}. Aborting top genes plot generation."
        )
        return

    all_good_reads = []
    for pos in position_list:
        # Filter out failed rounds for this well
        well_iss_rounds = _get_effective_iss_rounds(
            iss_rounds, pos, failed_rounds_by_well
        )

        try:
            read_db = pd.read_csv(dataset.append_well("reads", pos))
            if method == "probabilistic":
                good_reads = read_db[read_db["confidence"] >= confidence_threshold].copy()
            else:  # 'mine'
                good_reads = match_reads(read_db, codebook_db, iss_rounds=well_iss_rounds, well_name=pos, failed_rounds_by_well=failed_rounds_by_well, debug=False)
                if good_reads is not None:
                    good_reads = good_reads.copy()

            # Filter barcodes to effective positions
            if good_reads is not None and not good_reads.empty and "barcode" in good_reads.columns:
                good_reads["barcode"] = good_reads["barcode"].apply(
                    lambda barcode: "".join([barcode[i] for i in well_iss_rounds if i < len(barcode)])
                )

                # Map barcodes to gene_name and gene_id for this well
                gene_name_col = _get_col(gene_index_db, ["Gene name", "dep_map_gene_name", "gene_name"])
                gene_id_col = _get_col(gene_index_db, ["gene_id", "NCBI_ID"])

                if gene_index_db is not None and "barcode" in gene_index_db.columns and gene_name_col:
                    gi = gene_index_db.copy()
                    offset = dataset.codebook_round_offset
                    if offset:
                        raw_cb = pd.read_csv(dataset.codebook)
                        bc_map = dict(zip(raw_cb["sgRNA"].str[:10], raw_cb["sgRNA"].str[offset:offset + 10]))
                        gi["barcode"] = gi["barcode"].map(bc_map).apply(
                            lambda bc: "".join([bc[i] for i in well_iss_rounds if i < len(bc)]) if pd.notna(bc) else ""
                        )
                    else:
                        gi["barcode"] = gi["barcode"].apply(
                            lambda bc: "".join([bc[i] for i in well_iss_rounds if i < len(bc)])
                        )
                    good_reads["gene_name"] = good_reads["barcode"].map(dict(zip(gi["barcode"], gi[gene_name_col])))
                    if gene_id_col:
                        good_reads["gene_id"] = good_reads["barcode"].map(dict(zip(gi["barcode"], gi[gene_id_col])))

            if good_reads is not None:
                all_good_reads.append(good_reads)
        except FileNotFoundError:
            print(f"Reads file for well {pos} not found. Skipping.")
            continue

    if not all_good_reads:
        print(
            "No good reads found across any wells. Aborting top genes plot generation."
        )
        return

    # --- Data Aggregation ---
    pooled_reads = pd.concat(all_good_reads, ignore_index=True)

    # Add NCBI_ID if available from gene_index to identify NTCs
    # Note: barcodes in pooled_reads are already filtered to well-specific ISS rounds,
    # but gene_index_db needs matching. Since wells may have different ISS rounds,
    # we check if NCBI_ID was already mapped during the per-well loop.
    # Alternatively, we map gene_id to NCBI_ID if both are available.
    if gene_index_db is not None and "NCBI_ID" in gene_index_db.columns:
        # If gene_id is available, map via gene_id instead of barcode
        if "gene_id" in pooled_reads.columns and "gene_id" in gene_index_db.columns:
            gene_id_to_ncbi = dict(zip(gene_index_db["gene_id"], gene_index_db["NCBI_ID"]))
            pooled_reads["NCBI_ID"] = pooled_reads["gene_id"].map(gene_id_to_ncbi)

    # Check which gene identifier we have and separate NTCs
    if "gene_name" in pooled_reads.columns:
        pooled_reads.dropna(subset=["gene_name"], inplace=True)
        if pooled_reads.empty:
            print(
                "No reads could be matched to a gene name. Aborting top genes plot generation."
            )
            return

        # Separate NTCs from targeting genes
        if "NCBI_ID" in pooled_reads.columns:
            ntc_reads = pooled_reads[pooled_reads["NCBI_ID"] == -1]
            targeting_reads = pooled_reads[pooled_reads["NCBI_ID"] != -1]
        else:
            # If no NCBI_ID, assume all are targeting
            ntc_reads = pd.DataFrame()
            targeting_reads = pooled_reads

        cells_per_gene = targeting_reads.groupby("gene_name")["cell"].nunique()
        gene_labels = (
            cells_per_gene.nlargest(top_n).sort_values(ascending=False).index.tolist()
        )
    elif "gene_id" in pooled_reads.columns:
        pooled_reads.dropna(subset=["gene_id"], inplace=True)

        if pooled_reads.empty:
            print(
                "No reads could be matched to a gene. Aborting top genes plot generation."
            )
            return

        # Separate NTCs (gene_id == -1) from targeting genes
        ntc_reads = pooled_reads[pooled_reads["gene_id"] == -1]
        targeting_reads = pooled_reads[pooled_reads["gene_id"] != -1]

        cells_per_gene = targeting_reads.groupby("gene_id")["cell"].nunique()
        top_gene_ids = cells_per_gene.nlargest(top_n).sort_values(ascending=False).index

        # Map gene_id to gene_name for labels if available
        gene_id_col = _get_col(gene_index_db, ["gene_id", "NCBI_ID"])
        gene_name_col = _get_col(gene_index_db, ["Gene name", "dep_map_gene_name", "gene_name"])
        if gene_id_col and gene_name_col:
            gene_id_to_name = dict(zip(gene_index_db[gene_id_col], gene_index_db[gene_name_col]))
            gene_labels = [gene_id_to_name.get(gid, str(gid)) for gid in top_gene_ids]
        else:
            gene_labels = [str(gid) for gid in top_gene_ids]
    else:
        print("No gene identifier column found. Aborting top genes plot generation.")
        return

    top_genes = cells_per_gene.nlargest(top_n).sort_values(ascending=False)

    if top_genes.empty:
        print("No genes with cell counts found. Aborting top genes plot generation.")
        return

    # --- Plotting Targeting Genes ---
    fig, ax = plt.subplots(figsize=(12, 18))
    ax.barh(range(len(top_genes)), top_genes.values, color="steelblue", alpha=0.8)
    ax.set_yticks(range(len(top_genes)))
    ax.set_yticklabels(gene_labels, fontsize=14)
    ax.set_ylabel("Gene", fontsize=20)
    ax.set_xlabel("Number of Cells", fontsize=20)
    ax.set_title(
        f"Top {top_n} Targeting Genes by Cell Count (All Wells Pooled)\n{experiment} - Method: {method}",
        fontsize=22,
    )
    ax.grid(True, axis="x", linestyle="--", alpha=0.6)
    ax.invert_yaxis()  # Highest count at the top
    ax.set_ylim(len(top_genes) - 0.5, -0.5)  # Small margin at top and bottom
    ax.tick_params(axis="x", labelsize=16)  # Larger x-axis tick labels

    plt.tight_layout()

    # Save the plot
    plt.savefig(dataset.metrics_paths["top_genes_by_cell_count"], dpi=300)
    print(f"Saved top genes plot to {dataset.metrics_paths['top_genes_by_cell_count']}")
    plt.close()

    # --- Plotting NTC Guides (if available) ---
    if not ntc_reads.empty:
        print(f"--- Generating top {top_n} NTC guides by cell count bar plot ---")

        # Count cells per NTC guide (using barcode as identifier)
        if "barcode" in ntc_reads.columns and "cell" in ntc_reads.columns:
            cells_per_ntc_guide = ntc_reads.groupby("barcode")["cell"].nunique()
            # Take only top N NTC guides
            ntc_guides_sorted = cells_per_ntc_guide.nlargest(top_n).sort_values(ascending=False)

            if not ntc_guides_sorted.empty:
                # Create labels with gene names if available
                ntc_labels = []
                if "gene_name" in ntc_reads.columns:
                    barcode_to_gene_name = dict(zip(ntc_reads["barcode"], ntc_reads["gene_name"]))
                    for barcode in ntc_guides_sorted.index:
                        gene_name = barcode_to_gene_name.get(barcode, "Unknown")
                        ntc_labels.append(f"{barcode}\n({gene_name})")
                else:
                    ntc_labels = ntc_guides_sorted.index.tolist()

                # Calculate figure height based on number of NTC guides
                num_ntcs = len(ntc_guides_sorted)
                fig_height = max(8, num_ntcs * 0.4)  # At least 8 inches, scale with number of guides

                fig_ntc, ax_ntc = plt.subplots(figsize=(12, fig_height))
                ax_ntc.barh(range(len(ntc_guides_sorted)), ntc_guides_sorted.values, color="orange", alpha=0.8)
                ax_ntc.set_yticks(range(len(ntc_guides_sorted)))
                ax_ntc.set_yticklabels(ntc_labels, fontsize=12)
                ax_ntc.set_ylabel("NTC Guide (Gene Name)", fontsize=18)
                ax_ntc.set_xlabel("Number of Cells", fontsize=18)
                ax_ntc.set_title(
                    f"Top {top_n} NTC Guides by Cell Count (All Wells Pooled)\n{experiment} - Method: {method}",
                    fontsize=20,
                )
                ax_ntc.grid(True, axis="x", linestyle="--", alpha=0.6)
                ax_ntc.invert_yaxis()  # Highest count at the top
                ax_ntc.set_ylim(len(ntc_guides_sorted) - 0.5, -0.5)
                ax_ntc.tick_params(axis="x", labelsize=14)

                plt.tight_layout()

                # Save the NTC plot
                ntc_plot_path = dataset.metrics_paths["top_genes_by_cell_count"].parent / f"ntc_guides_by_cell_count_{method}.png"
                plt.savefig(ntc_plot_path, dpi=300)
                print(f"Saved NTC guides plot to {ntc_plot_path}")
                plt.close()
            else:
                print("No NTC guides with cell counts found.")
        else:
            print("Required columns for NTC plotting not found.")
    else:
        print("No NTC reads found to plot separately.")


def plot_top_guides_by_cell_count(
    experiment: str,
    iss_rounds: list[int] | None = None,
    method: str = "mine",
    confidence_threshold: float = 0.95,
    top_n: int = 50,
    failed_rounds_by_well: dict[str, list[int]] | None = None,
) -> None:
    """
    Generates and saves a bar plot of the top N guides (sgRNAs) by cell count, pooled across all wells.
    Unlike plot_top_genes_by_cell_count which aggregates all guides for a gene, this shows individual guide performance.
    Y-axis: number of cells, X-axis: guides ordered by descending cell count.

    Args:
        iss_rounds: List of ISS round indices to use. If None, defaults to first 10 rounds.
        method: Base calling method ('mine' or 'probabilistic').
        confidence_threshold: Confidence threshold for probabilistic method.
        top_n: Number of top guides to display.
    """
    # Backward compatibility
    if iss_rounds is None:
        iss_rounds = list(range(10))

    print(f"--- Generating top {top_n} guides by cell count bar plot ---")
    dataset = OpsDataset(experiment, method=method)
    codebook_db = dataset.load_codebook()

    # Load gene index to get gene names for labeling
    try:
        gene_index_db = dataset.load_gene_index()
    except Exception as e:
        print(f"Could not load gene index: {e}. Using barcode only.")
        gene_index_db = None

    try:
        seg_store = open_ome_zarr(dataset.store_paths["iss_segmentation"], mode="r")
        position_list = [a[0] for a in seg_store.positions()]
    except Exception as e:
        print(
            f"Could not load segmentation data to get wells: {e}. Aborting top guides plot generation."
        )
        return

    all_good_reads = []
    for pos in position_list:
        # Filter out failed rounds for this well
        well_iss_rounds = _get_effective_iss_rounds(
            iss_rounds, pos, failed_rounds_by_well
        )

        try:
            read_db = pd.read_csv(dataset.append_well("reads", pos))
            if method == "probabilistic":
                good_reads = read_db[read_db["confidence"] >= confidence_threshold].copy()
            else:  # 'mine'
                good_reads = match_reads(read_db, codebook_db, iss_rounds=well_iss_rounds, well_name=pos, failed_rounds_by_well=failed_rounds_by_well, debug=False)
                if good_reads is not None:
                    good_reads = good_reads.copy()

            # Filter barcodes to effective positions for this well
            if good_reads is not None and not good_reads.empty and "barcode" in good_reads.columns:
                good_reads["guide"] = good_reads["barcode"].apply(
                    lambda barcode: "".join([barcode[i] for i in well_iss_rounds if i < len(barcode)])
                )
                # Map guide to gene_name using filtered barcodes
                gene_name_col = _get_col(gene_index_db, ["Gene name", "dep_map_gene_name", "gene_name"])
                if gene_index_db is not None and "barcode" in gene_index_db.columns and gene_name_col:
                    gi = gene_index_db.copy()
                    offset = dataset.codebook_round_offset
                    if offset:
                        raw_cb = pd.read_csv(dataset.codebook)
                        bc_map = dict(zip(raw_cb["sgRNA"].str[:10], raw_cb["sgRNA"].str[offset:offset + 10]))
                        gi["guide"] = gi["barcode"].map(bc_map).apply(
                            lambda bc: "".join([bc[i] for i in well_iss_rounds if i < len(bc)]) if pd.notna(bc) else ""
                        )
                    else:
                        gi["guide"] = gi["barcode"].apply(
                            lambda bc: "".join([bc[i] for i in well_iss_rounds if i < len(bc)])
                        )
                    good_reads["gene_name"] = good_reads["guide"].map(dict(zip(gi["guide"], gi[gene_name_col])))

            if good_reads is not None:
                all_good_reads.append(good_reads)
        except FileNotFoundError:
            print(f"Reads file for well {pos} not found. Skipping.")
            continue

    if not all_good_reads:
        print(
            "No good reads found across any wells. Aborting top guides plot generation."
        )
        return

    # --- Data Aggregation ---
    pooled_reads = pd.concat(all_good_reads, ignore_index=True)

    # Count unique cells per guide
    cells_per_guide = pooled_reads.groupby("guide")["cell"].nunique()
    top_guides = cells_per_guide.nlargest(top_n).sort_values(ascending=False)

    if top_guides.empty:
        print("No guides with cell counts found. Aborting top guides plot generation.")
        return

    # Create labels with gene names (mapped during per-well loop)
    if "gene_name" in pooled_reads.columns:
        guide_to_gene = pooled_reads.drop_duplicates("guide").set_index("guide")["gene_name"].to_dict()
        guide_labels = [f"{g} ({guide_to_gene.get(g, 'Unknown')})" for g in top_guides.index]
    else:
        guide_labels = top_guides.index.tolist()

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 18))
    ax.barh(range(len(top_guides)), top_guides.values, color="teal", alpha=0.8)
    ax.set_yticks(range(len(top_guides)))
    ax.set_yticklabels(guide_labels, fontsize=14)
    ax.set_ylabel("Guide (Gene)", fontsize=20)
    ax.set_xlabel("Number of Cells", fontsize=20)
    ax.set_title(
        f"Top {top_n} Guides by Cell Count (All Wells Pooled)\n{experiment} - Method: {method}",
        fontsize=22,
    )
    ax.grid(True, axis="x", linestyle="--", alpha=0.6)
    ax.invert_yaxis()  # Highest count at the top
    ax.set_ylim(len(top_guides) - 0.5, -0.5)  # Small margin at top and bottom
    ax.tick_params(axis="x", labelsize=16)  # Larger x-axis tick labels

    plt.tight_layout()

    # Save the plot
    plt.savefig(dataset.metrics_paths["top_guides_by_cell_count"], dpi=300)
    print(
        f"Saved top guides plot to {dataset.metrics_paths['top_guides_by_cell_count']}"
    )
    plt.close()





def plot_guide_entropy_vs_cell_count(
    experiment: str,
    iss_rounds: list[int] | None = None,
    method: str = "mine",
    confidence_threshold: float = 0.95,
    failed_rounds_by_well: dict[str, list[int]] | None = None,
) -> dict:
    """
    Diagnostic plot: Guide sequence entropy vs cell count.

    This plot helps identify if the base-calling method systematically over-calls
    low-complexity (low entropy) guides. A strong negative correlation indicates bias.

    Args:
        iss_rounds: List of ISS round indices to use. If None, defaults to first 10 rounds.
        method: Base calling method ('mine' or 'probabilistic').
        confidence_threshold: Confidence threshold for probabilistic method.

    Returns:
        dict: Regression statistics (slope, R², p-value)
    """
    from scipy import stats as scipy_stats

    # Backward compatibility
    if iss_rounds is None:
        iss_rounds = list(range(10))

    print(f"--- Generating guide entropy vs cell count diagnostic plot ({method}) ---")
    dataset = OpsDataset(experiment, method=method)
    codebook_db = dataset.load_codebook()

    # Load gene index to identify NTC guides
    try:
        gene_index_db = dataset.load_gene_index()
    except Exception as e:
        print(f"Could not load gene index: {e}. Will not distinguish NTC guides.")
        gene_index_db = None

    try:
        seg_store = open_ome_zarr(dataset.store_paths["iss_segmentation"], mode="r")
        position_list = [a[0] for a in seg_store.positions()]
    except Exception as e:
        print(f"Could not load segmentation data: {e}. Aborting.")
        return {}

    # Collect all good reads across wells
    all_good_reads = []
    for pos in position_list:
        # Filter out failed rounds for this well
        well_iss_rounds = _get_effective_iss_rounds(
            iss_rounds, pos, failed_rounds_by_well
        )

        try:
            read_db = pd.read_csv(dataset.append_well("reads", pos))
            if method == "probabilistic":
                good_reads = read_db[read_db["confidence"] >= confidence_threshold].copy()
            else:  # 'mine'
                matched = match_reads(read_db, codebook_db, iss_rounds=well_iss_rounds, well_name=pos, failed_rounds_by_well=failed_rounds_by_well, debug=False)
                good_reads = matched.copy() if not matched.empty else matched

            # Filter barcodes to effective positions for this well
            if not good_reads.empty and "barcode" in good_reads.columns:
                good_reads["guide"] = good_reads["barcode"].apply(
                    lambda barcode: "".join([barcode[i] for i in well_iss_rounds if i < len(barcode)])
                )
                # Map guide to gene_name using filtered barcodes
                gene_name_col = _get_col(gene_index_db, ["Gene name", "dep_map_gene_name", "gene_name"])
                if gene_index_db is not None and "barcode" in gene_index_db.columns and gene_name_col:
                    gi = gene_index_db.copy()
                    offset = dataset.codebook_round_offset
                    if offset:
                        raw_cb = pd.read_csv(dataset.codebook)
                        bc_map = dict(zip(raw_cb["sgRNA"].str[:10], raw_cb["sgRNA"].str[offset:offset + 10]))
                        gi["guide"] = gi["barcode"].map(bc_map).apply(
                            lambda bc: "".join([bc[i] for i in well_iss_rounds if i < len(bc)]) if pd.notna(bc) else ""
                        )
                    else:
                        gi["guide"] = gi["barcode"].apply(
                            lambda bc: "".join([bc[i] for i in well_iss_rounds if i < len(bc)])
                        )
                    good_reads["gene_name"] = good_reads["guide"].map(dict(zip(gi["guide"], gi[gene_name_col])))

            all_good_reads.append(good_reads)
        except FileNotFoundError:
            print(f"Reads file for well {pos} not found. Skipping.")
            continue

    if not all_good_reads:
        print("No good reads found. Aborting.")
        return {}

    # Pool reads across wells
    pooled_reads = pd.concat(all_good_reads, ignore_index=True)

    # Build guide-to-gene lookup from pooled data
    guide_to_gene = {}
    if "gene_name" in pooled_reads.columns:
        guide_to_gene = pooled_reads.drop_duplicates("guide").set_index("guide")["gene_name"].to_dict()

    # Count unique cells per guide
    cells_per_guide = pooled_reads.groupby("guide")["cell"].nunique()

    if cells_per_guide.empty:
        print("No guides with cell counts. Aborting.")
        return {}

    # Calculate entropy for each guide
    def calculate_entropy(seq):
        from collections import Counter

        counts = Counter(seq)
        probs = np.array([c / len(seq) for c in counts.values()])
        return -np.sum(probs * np.log2(probs + 1e-9))

    guide_entropies = {
        guide: calculate_entropy(guide) for guide in cells_per_guide.index
    }

    # Create dataframe for analysis
    plot_df = pd.DataFrame(
        {
            "guide": cells_per_guide.index,
            "cell_count": cells_per_guide.values,
            "entropy": [guide_entropies[g] for g in cells_per_guide.index],
        }
    )

    # Identify NTC guides if possible
    plot_df["is_ntc"] = False
    if (
        gene_index_db is not None
        and "barcode" in gene_index_db.columns
        and "NCBI_ID" in gene_index_db.columns
    ):
        ntc_barcodes = set(
            gene_index_db[gene_index_db["NCBI_ID"] == -1]["barcode"].values
        )
        plot_df["is_ntc"] = plot_df["guide"].isin(ntc_barcodes)

    # Separate targeting and NTC
    targeting = plot_df[~plot_df["is_ntc"]]
    ntc = plot_df[plot_df["is_ntc"]]

    # Linear regression on targeting guides
    X = targeting["entropy"].values.reshape(-1, 1)
    y = targeting["cell_count"].values

    if len(X) < 3:
        print("Not enough data points for regression. Aborting.")
        return {}

    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = r2_score(y, y_pred)

    # Calculate p-value
    _, _, _, p_value, _ = scipy_stats.linregress(X.flatten(), y)

    # Calculate confidence interval for regression line
    from scipy import stats as scipy_stats

    n = len(X)
    dof = n - 2
    t_val = scipy_stats.t.ppf(0.975, dof)
    residuals = y - y_pred
    s_err = np.sqrt(np.sum(residuals**2) / dof)
    x_mean = np.mean(X)
    se_line = s_err * np.sqrt(1 / n + (X - x_mean) ** 2 / np.sum((X - x_mean) ** 2))
    conf_interval = t_val * se_line.flatten()

    # Binned medians for overlay
    n_bins = 10
    entropy_bins = np.linspace(
        targeting["entropy"].min(), targeting["entropy"].max(), n_bins
    )
    bin_centers = (entropy_bins[:-1] + entropy_bins[1:]) / 2
    plot_df_with_bins = targeting.copy()
    plot_df_with_bins["entropy_bin"] = pd.cut(
        plot_df_with_bins["entropy"], bins=entropy_bins
    )
    binned_medians = plot_df_with_bins.groupby("entropy_bin", observed=False)["cell_count"].median()

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 8))

    # Scatter plot
    ax.scatter(
        targeting["entropy"],
        targeting["cell_count"],
        alpha=0.4,
        s=30,
        c="steelblue",
        label="Targeting Guides",
    )

    if not ntc.empty:
        ax.scatter(
            ntc["entropy"],
            ntc["cell_count"],
            alpha=0.6,
            s=30,
            c="orange",
            label="NTC Guides",
            marker="x",
        )

    # Regression line
    x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(x_line)
    ax.plot(x_line, y_line, "r-", linewidth=2, label="Linear Regression")

    # Confidence interval (simplified - using first and last points)
    x_flat = X.flatten()
    sort_idx = np.argsort(x_flat)
    ax.fill_between(
        x_flat[sort_idx],
        (y_pred - conf_interval)[sort_idx],
        (y_pred + conf_interval)[sort_idx],
        alpha=0.2,
        color="red",
        label="95% CI",
    )

    # Binned medians
    ax.plot(
        bin_centers,
        binned_medians,
        "ko-",
        linewidth=2,
        markersize=8,
        label="Binned Median",
        zorder=5,
    )

    # Label top guide with highest cell count
    top_guide = targeting.nlargest(1, "cell_count").iloc[0]
    top_barcode = top_guide["guide"]
    top_entropy = top_guide["entropy"]
    top_count = top_guide["cell_count"]

    # Get gene name from guide_to_gene lookup (built from pooled data)
    top_gene_name = guide_to_gene.get(top_barcode, "Unknown")

    # Add annotation for top guide
    ax.annotate(
        f"{top_gene_name}\n{top_barcode}",
        xy=(top_entropy, top_count),
        xytext=(top_entropy + 0.1, top_count + top_count * 0.1),
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
        arrowprops=dict(
            arrowstyle="->", connectionstyle="arc3,rad=0.3", lw=2, color="black"
        ),
    )

    # Labels and title
    ax.set_xlabel("Guide Sequence Entropy (bits)", fontsize=16)
    ax.set_ylabel("Number of Cells (log scale)", fontsize=16)
    ax.set_yscale("log")
    ax.set_title(
        f"Guide Entropy vs Cell Count\n{experiment} - Method: {method}", fontsize=18
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(labelsize=12)

    # Legend
    ax.legend(loc="upper right", fontsize=11)

    # Statistics annotation - positioned below legend in top right
    stats_text = (
        f"Slope: {slope:.1f} cells/bit\n"
        f"R²: {r2:.3f}\n"
        f"p-value: {p_value:.2e}\n"
        f"n={len(targeting)}"
    )
    ax.text(
        0.98,
        0.75,
        stats_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Interpretation helper - bottom right
    if abs(slope) > 100 and p_value < 0.001:
        bias_text = "⚠️ STRONG BIAS DETECTED"
        bias_color = "red"
    elif abs(slope) > 50 and p_value < 0.01:
        bias_text = "⚠️ MODERATE BIAS"
        bias_color = "orange"
    else:
        bias_text = "✓ NO SIGNIFICANT BIAS"
        bias_color = "green"

    ax.text(
        0.98,
        0.02,
        bias_text,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color=bias_color,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.9,
            edgecolor=bias_color,
            linewidth=2,
        ),
    )
    plt.tight_layout()

    # Save plot
    plt.savefig(dataset.metrics_paths["guide_entropy_vs_cell_count"], dpi=300)
    print(
        f"Saved entropy diagnostic plot to {dataset.metrics_paths['guide_entropy_vs_cell_count']}"
    )
    plt.close()

    # Check for outlier guides/genes (>1.5x median)
    guide_max = cells_per_guide.max()
    guide_median = cells_per_guide.median()
    guide_ratio = guide_max / guide_median if guide_median > 0 else 0
    top_guide_barcode = cells_per_guide.idxmax()
    top_guide_gene = guide_to_gene.get(top_guide_barcode, "Unknown")
    
    # Calculate cells per gene if gene_name available
    gene_max, gene_median, gene_ratio, top_gene = 0, 0, 0, None
    if "gene_name" in pooled_reads.columns:
        cells_per_gene = pooled_reads.groupby("gene_name")["cell"].nunique()
        cells_per_gene = cells_per_gene[cells_per_gene.index.notna()]  # drop NaN genes
        if not cells_per_gene.empty:
            gene_max = cells_per_gene.max()
            gene_median = cells_per_gene.median()
            gene_ratio = gene_max / gene_median if gene_median > 0 else 0
            top_gene = cells_per_gene.idxmax()

    # Return statistics
    stats_dict = {
        "entropy_bias_slope": round(slope, 2),
        "entropy_bias_r2": round(r2, 3),
        "entropy_bias_pvalue": p_value,
        "entropy_bias_interpretation": bias_text,
        "top_guide_ratio": round(guide_ratio, 2),
        "top_gene_ratio": round(gene_ratio, 2),
    }

    print(f"\n{'='*60}")
    print(f"ENTROPY BIAS ANALYSIS RESULTS ({method})")
    print(f"{'='*60}")
    print(f"Slope: {slope:.1f} cells per entropy bit")
    print(f"R²: {r2:.3f}")
    print(f"P-value: {p_value:.2e}")
    print(f"Interpretation: {bias_text}")
    
    # Top guide/gene info (with warning if exceeds threshold)
    warn_guide = "⚠️ " if guide_ratio > 10 else ""
    print(f"{warn_guide}Top guide: '{top_guide_gene}' [{top_guide_barcode}] has {guide_max} cells ({guide_ratio:.1f}x median of {guide_median:.0f})")
    if top_gene:
        warn_gene = "⚠️ " if gene_ratio > 2 else ""
        print(f"{warn_gene}Top gene: '{top_gene}' has {gene_max} cells ({gene_ratio:.1f}x median of {gene_median:.0f})")
    
    print(f"{'='*60}\n")

    return stats_dict


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from cyclops_utils.data.filesystem import resolve_experiment_name

    parser = argparse.ArgumentParser(
        description="Generate frequency_table_filtered.csv for one or more experiments"
    )
    parser.add_argument(
        "experiments", nargs="*",
        help="Experiment names or shorthand (e.g. 94 ops0094_20251217)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process every experiment in /path/to/ops_data/",
    )
    parser.add_argument(
        "--method", default="mine",
        choices=["probabilistic", "mine"],
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.95)
    parser.add_argument(
        "--slurm", action="store_true",
        help="Submit one SLURM job per experiment (parallel)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="With --slurm: print plan without submitting",
    )
    args = parser.parse_args()

    if not args.experiments and not args.all:
        parser.error("Provide experiments or use --all")

    if args.all:
        ops_dir = Path(f"{BASE_PATH}")
        experiments = sorted(
            d.name for d in ops_dir.iterdir()
            if d.is_dir() and d.name.startswith("ops0")  # exclude utility dirs
        )
        print(f"Found {len(experiments)} experiments under {ops_dir}\n")
    else:
        experiments = [resolve_experiment_name(e) for e in args.experiments]

    if args.slurm:
        from cyclops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

        jobs = [
            {
                "name": f"filtered_freq_{exp}",
                "func": save_filtered_frequency_table,
                "kwargs": {
                    "experiment": exp,
                    "method": args.method,
                    "confidence_threshold": args.confidence_threshold,
                },
                "metadata": {"type": "filtered_freq", "experiment": exp},
            }
            for exp in experiments
        ]

        submit_parallel_jobs(
            jobs_to_submit=jobs,
            experiment="batch",
            slurm_params={
                "timeout_min": 3,
                "mem": "32GB",
                "cpus_per_task": 4,
                "slurm_partition": "cpu",
            },
            log_dir="filtered_freq_table",
            manifest_prefix="filtered_freq",
            dry_run=args.dry_run,
        )
    else:
        for i, exp in enumerate(experiments, 1):
            print(f"\n[{i}/{len(experiments)}] {exp}")
            try:
                save_filtered_frequency_table(
                    experiment=exp,
                    method=args.method,
                    confidence_threshold=args.confidence_threshold,
                )
            except Exception as e:
                print(f"  ERROR: {e}")
