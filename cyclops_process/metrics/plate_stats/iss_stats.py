import pandas as pd
import numpy as np
import dask.array as da
from iohub.ngff import open_ome_zarr
from cyclops_utils.data.experiment import OpsDataset
from cyclops_process.metrics.plate_stats.match_reads import (
    _get_effective_iss_rounds,
    match_reads,
)
from cyclops_process.metrics.plate_stats.iss_entropy import (
    compute_iss_quality_stats,
    load_reference_distribution,
)


from cyclops_process.metrics.plate_stats.iss_register_metrics import (
    get_auto_register_stats_for_well,
    get_iss_drift_stats_for_well,
)
from cyclops_process.metrics.plate_stats.plate_stats_track_metrics import (
    get_post_tracking_stats_for_well,
    get_link_stats_for_well,
    get_tracking_graph_stats_for_well,
)
from cyclops_process.metrics.plate_stats.plate_stats_stitch_metrics import (
    get_stitch_confidence_for_experiment,
)
from cyclops_process.metrics.plate_stats.plate_stats_reconstruction_metrics import (
    get_z_offset_stats_for_experiment,
)
from cyclops_process.metrics.plate_stats.iss_spatial_coherence import (
    get_spatial_coherence_stats_for_well,
)
from cyclops_process.metrics.plate_stats.iss_cell_size import (
    get_cell_size_stats_for_well,
)
from cyclops_process.metrics.plate_stats.iss_flatfield_metrics import (
    get_flatfield_stats_for_well,
)



def statistics(
    experiment: str,
    iss_rounds: list[int] | None = None,
    prob_stats: dict = None,
    growth_effect_stats: dict = None,
    method: str = "mine",
    confidence_threshold: float = 0.95,
    failed_rounds_by_well: dict[str, list[int]] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Calculate cell numbers for each step of the process, adapting to the base calling method.

    Args:
        iss_rounds: List of ISS round indices to use. If None, defaults to first 10 rounds.
        prob_stats: Dictionary containing probabilistic statistics.
        growth_effect_stats: Dictionary containing growth effect statistics.
        method: Base calling method ('mine' or 'probabilistic').
        confidence_threshold: Confidence threshold for probabilistic method.
        failed_rounds_by_well: Dictionary mapping wells to lists of failed round indices to exclude
    """
    if iss_rounds is None:
        iss_rounds = list(range(10))

    dataset = OpsDataset(experiment, method=method)
    codebook_db = dataset.load_codebook()

    # Load and merge codebooks to get gene names
    # gene_index_db = pd.read_csv(dataset.gene_index)
    # codebook_db = pd.merge(codebook_db_base, gene_index_db[['sgRNA', 'gene_name']], on='sgRNA', how='left')

    seg_store = open_ome_zarr(dataset.store_paths["iss_segmentation"], mode="r")
    position_list = [a[0] for a in seg_store.positions()]

    if prob_stats is None:
        prob_stats = {}
    if growth_effect_stats is None:
        growth_effect_stats = {}

    # Load reference distribution once for correlation stats (if available)
    reference_freq = load_reference_distribution()

    pos_dict = {}
    freq_dict = {}
    for pos in position_list:
        # Filter out failed rounds for this well
        well_iss_rounds = _get_effective_iss_rounds(
            iss_rounds, pos, failed_rounds_by_well
        )

        out = {}
        try:
            read_db = pd.read_csv(dataset.append_well("reads", pos))
        except FileNotFoundError:
            read_db = pd.DataFrame()

        has_barcode = "barcode" in read_db.columns
        has_cell = "cell" in read_db.columns

        # Define "good" reads based on the method
        if method == "probabilistic":
            if "confidence" in read_db.columns:
                good_reads = read_db[
                    read_db["confidence"] >= confidence_threshold
                ].copy()
            else:
                good_reads = read_db.iloc[0:0].copy()
            good_reads_key = "high_confidence"
        else:  # 'mine'
            if has_barcode and not read_db.empty:
                good_reads = match_reads(read_db, codebook_db, iss_rounds=well_iss_rounds, well_name=pos, failed_rounds_by_well=failed_rounds_by_well, debug=False)
            else:
                good_reads = read_db.iloc[0:0].copy()
            good_reads_key = "matched"

        # Filter barcodes to effective positions (same as in create_freq_tables)
        if not good_reads.empty and "barcode" in good_reads.columns:
            good_reads["barcode"] = good_reads["barcode"].apply(
                lambda barcode: "".join([barcode[i] for i in well_iss_rounds if i < len(barcode)])
            )

        # load freq table
        try:
            freq_table = pd.read_csv(
                dataset.append_well("frequency_table", pos), index_col=0
            )
        except FileNotFoundError:
            freq_table = pd.DataFrame(
                columns=["barcode", "count"]
            )  # empty safe default
        freq_dict[pos] = freq_table

        # Load points (spots) safely
        try:
            points = np.load(dataset.append_well("spots", pos))
            num_spots = len(points)
        except Exception:
            num_spots = 0

        seg_da_array = da.from_array(seg_store[pos].data)
        num_cells = int(len(da.unique(seg_da_array).compute()) - 1)
        if num_cells < 0:
            num_cells = 0
        out["num_cells"] = num_cells

        cells_with_reads = (
            int(
                read_db["cell"][read_db.get("cell", pd.Series(dtype=int)) > 0].nunique()
            )
            if has_cell and not read_db.empty
            else 0
        )
        good_cells_with_reads = (
            int(
                good_reads["cell"][
                    good_reads.get("cell", pd.Series(dtype=int)) > 0
                ].nunique()
            )
            if (not good_reads.empty and "cell" in good_reads.columns)
            else 0
        )
        out["cells_with_reads"] = cells_with_reads
        out[f"cells_with_{good_reads_key}_reads"] = good_cells_with_reads

        out["percent_cells_with_reads"] = (
            float(np.round(100 * (cells_with_reads / num_cells), 1))
            if num_cells > 0
            else 0.0
        )
        out[f"percent_cells_with_{good_reads_key}_reads"] = (
            float(np.round(100 * (good_cells_with_reads / num_cells), 1))
            if num_cells > 0
            else 0.0
        )

        out[f"percent_{good_reads_key}_cells_of_cells_with_reads"] = (
            float(np.round(100 * (good_cells_with_reads / cells_with_reads), 1))
            if cells_with_reads > 0
            else 0.0
        )

        out["avg_guide_coverage"] = (
            float(np.round(freq_table["count"].mean()))
            if ("count" in freq_table.columns and not freq_table.empty)
            else 0
        )

        # add avg guide coverage including dropouts
        out["avg_guide_coverage_inc_dropouts"] = float(
            np.round(
                (freq_table["count"].sum() / len(codebook_db))
                if len(codebook_db) > 0
                else 0
            )
        )

        out["num_spots"] = num_spots
        out["num_reads"] = 0 if read_db is None else len(read_db)
        out[f"num_reads_{good_reads_key}"] = len(good_reads)
        out["num_unique_reads"] = (
            int(len(np.unique(
                good_reads["barcode"].astype(str).apply(
                    lambda barcode: "".join([barcode[i] for i in well_iss_rounds if i < len(barcode)])
                )
            )))
            if (
                not good_reads.empty
                and "barcode" in good_reads.columns
            )
            else 0
        )
        out["num_codes"] = len(codebook_db)
        out["percent_codes_found"] = (
            float(np.round(100 * (out["num_unique_reads"] / out["num_codes"]), 3))
            if out["num_codes"] > 0
            else 0.0
        )
        out[f"percent_reads_{good_reads_key}"] = (
            float(
                np.round(
                    100 * (out[f"num_reads_{good_reads_key}"] / out["num_reads"]), 1
                )
            )
            if out["num_reads"] > 0
            else 0.0
        )

        # estimate number of doublets
        if not good_reads.empty and "cell" in good_reads.columns:
            grouped = good_reads.groupby("cell")
            filtered = grouped.filter(lambda x: len(x) == 2)
            result = (
                filtered.groupby("cell")["barcode"]
                .nunique()
                .reset_index(name="unique_B_count")
            )

            # Values where B is the same in both rows
            cells_with_2_diff_barcodes = result[result["unique_B_count"] == 2]
        else:
            cells_with_2_diff_barcodes = pd.DataFrame()

        out["fraction_doublets"] = (
            float(np.round(len(cells_with_2_diff_barcodes) / num_cells, 3))
            if num_cells > 0
            else 0.0
        )

        # --- NEW: Calculate cells per gene stats ---
        if (
            "gene_id" in codebook_db.columns
            and "sgRNA" in codebook_db.columns
            and not good_reads.empty
        ):
            # Create a duplicate-safe mapping from trimmed sgRNA to gene_id
            cb = codebook_db.copy()
            cb["trim_sgRNA"] = cb["sgRNA"].apply(
                lambda a: "".join([a[i] for i in well_iss_rounds])
            )
            cb_unique = cb.drop_duplicates(subset=["trim_sgRNA"], keep="first")
            sgRNA_to_gene_map = dict(zip(cb_unique["trim_sgRNA"], cb_unique["gene_id"]))

            # Use the map to add the 'gene_id' column to our reads dataframe
            reads_with_gene = good_reads.copy()
            if "barcode" in reads_with_gene.columns:
                # Barcodes are already filtered to well_iss_rounds positions, just map them
                reads_with_gene["gene_id"] = (
                    reads_with_gene["barcode"]
                    .astype(str)
                    .map(sgRNA_to_gene_map)
                )
                # Filter out reads that didn't match a gene in the codebook
                reads_with_gene.dropna(subset=["gene_id"], inplace=True)

                if not reads_with_gene.empty and "cell" in reads_with_gene.columns:
                    # Count unique cells for each gene
                    cells_per_gene = reads_with_gene.groupby("gene_id")[
                        "cell"
                    ].nunique()

                    if not cells_per_gene.empty:
                        out["iss_mean_cells_per_gene"] = float(
                            np.round(cells_per_gene.mean(), 2)
                        )
                        out["iss_median_cells_per_gene"] = float(
                            np.round(cells_per_gene.median(), 2)
                        )
                        out["iss_max_cells_per_gene"] = int(cells_per_gene.max())
                        out["iss_min_cells_per_gene"] = int(cells_per_gene.min())
                    else:
                        out["iss_mean_cells_per_gene"] = 0
                        out["iss_median_cells_per_gene"] = 0
                        out["iss_max_cells_per_gene"] = 0
                        out["iss_min_cells_per_gene"] = 0
                else:
                    out["iss_mean_cells_per_gene"] = 0
                    out["iss_median_cells_per_gene"] = 0
                    out["iss_max_cells_per_gene"] = 0
                    out["iss_min_cells_per_gene"] = 0
        else:
            print(
                "Skipping cells per gene stats: missing required columns or empty reads."
            )

        # --- Compute ISS quality stats (entropy and correlation) ---
        if not good_reads.empty and "barcode" in good_reads.columns:
            # Need to use the original good_reads with full barcodes for entropy calculation
            # Re-match reads to get unfiltered barcodes for entropy analysis
            if method == "probabilistic":
                good_reads_for_entropy = read_db[
                    read_db["confidence"] >= confidence_threshold
                ].copy() if "confidence" in read_db.columns else read_db.iloc[0:0].copy()
            else:
                good_reads_for_entropy = match_reads(
                    read_db, codebook_db, iss_rounds=well_iss_rounds, well_name=pos, failed_rounds_by_well=failed_rounds_by_well, debug=False
                ) if has_barcode and not read_db.empty else read_db.iloc[0:0].copy()

            iss_quality_stats = compute_iss_quality_stats(
                good_reads_for_entropy,
                well_iss_rounds,
                reference_freq=reference_freq,
            )
            out.update(iss_quality_stats)

        # --- Add probabilistic stats to the output ---
        if pos in prob_stats:
            out.update(prob_stats[pos])

        # --- Add growth effect stats to the output ---
        if pos in growth_effect_stats:
            stats_to_add = growth_effect_stats[pos]
            # print(f'stats from growth efffect: {stats_to_add}')
            out["growth_effect_slope"] = stats_to_add.get("growth_effect_slope")
            out["growth_effect_r2"] = stats_to_add.get("growth_effect_r2")
            out["growth_effect_fc"] = stats_to_add.get("growth_effect_fold_change")

        # --- Add stitch confidence stats (per-well) ---
        try:
            stitch_conf_stats = get_stitch_confidence_for_experiment(dataset, well=pos)
            out.update(stitch_conf_stats)
        except Exception as e:
            print(f"Warning: Could not load stitch confidence stats for {pos}: {e}")

        # --- Add auto-register stats (per-well) ---
        try:
            # Pass the full row/col unit so row B doesn't read row A's data.
            auto_reg_stats = get_auto_register_stats_for_well(dataset, pos)
            out.update(auto_reg_stats)
        except Exception as e:
            print(f"Warning: Could not load auto-register stats for {pos}: {e}")

        # --- Add post-tracking cells per gene stats (per-well) ---
        try:
            post_track_stats = get_post_tracking_stats_for_well(dataset, pos)
            out.update(post_track_stats)

            # Compute cell loss percentage from ISS to tracking
            # Uses cells_with_matched_reads (mine) or cells_with_high_confidence_reads (probabilistic)
            cells_before_key = f"cells_with_{good_reads_key}_reads"
            cells_before = out.get(cells_before_key, 0)
            cells_after = post_track_stats.get("post_track_total_tracked_cells", 0)
            if cells_before and cells_before > 0 and cells_after is not None:
                loss_pct = float(np.round(100 * (1 - cells_after / cells_before), 1))
                out["post_track_cell_loss_pct"] = loss_pct
            else:
                out["post_track_cell_loss_pct"] = None
        except Exception as e:
            print(f"Warning: Could not load post-tracking stats for {pos}: {e}")

        # --- Add z-offset stats (per-well) ---
        try:
            z_offset_stats = get_z_offset_stats_for_experiment(dataset, well=pos)
            out.update(z_offset_stats)
        except Exception as e:
            print(f"Warning: Could not load z-offset stats for {pos}: {e}")

        # --- Add ISS drift stats (per-well) ---
        try:
            drift_stats = get_iss_drift_stats_for_well(dataset, pos)
            out.update(drift_stats)
        except Exception as e:
            print(f"Warning: Could not load ISS drift stats for {pos}: {e}")

        # --- Add link stats (per-well) ---
        try:
            link_stats = get_link_stats_for_well(dataset, pos)
            out.update(link_stats)
        except Exception as e:
            print(f"Warning: Could not load link stats for {pos}: {e}")

        # --- Add tracking graph/doublet stats (per-well) ---
        try:
            track_graph_stats = get_tracking_graph_stats_for_well(dataset, pos)
            out.update(track_graph_stats)
        except Exception as e:
            print(f"Warning: Could not load tracking graph stats for {pos}: {e}")

        # --- Add spatial coherence stats (per-well) ---
        try:
            sc_stats = get_spatial_coherence_stats_for_well(dataset, pos, force=force)
            out.update(sc_stats)
        except Exception as e:
            print(f"Warning: Could not load spatial coherence stats for {pos}: {e}")

        # --- Add cell size/shape stats (per-well) ---
        try:
            cell_size_well_stats = get_cell_size_stats_for_well(dataset, pos, force=force)
            out.update(cell_size_well_stats)
        except Exception as e:
            print(f"Warning: Could not load cell size stats for {pos}: {e}")

        # --- Add flatfield correction stats (per-well) ---
        try:
            flatfield_stats = get_flatfield_stats_for_well(dataset, pos)
            out.update(flatfield_stats)
        except Exception as e:
            print(f"Warning: Could not load flatfield stats for {pos}: {e}")

        pos_dict[pos] = out

    pos_df = pd.DataFrame.from_dict(pos_dict, orient="index")

    # Convert integer-like float columns to proper integers (using nullable Int64)
    # This ensures values like 496995.0 are saved as 496995 in the CSV
    for col in pos_df.columns:
        try:
            # Check if all non-null values are integer-like
            non_null = pos_df[col].dropna()
            if non_null.empty:
                continue
            if all(isinstance(v, (int, float)) and (pd.isna(v) or float(v) == int(float(v))) for v in pos_df[col]):
                pos_df[col] = pos_df[col].astype("Int64")  # Nullable integer
        except (ValueError, TypeError):
            pass  # Keep as-is if conversion fails

    pos_df.T.to_csv(dataset.metrics_paths["statistics"])

    # merged_freq = pd.concat(freq_dict.values())
    # exp_freq = merged_freq.groupby("barcode", as_index=False)["count"].sum()
    # exp_freq.to_csv(dataset.metrics_paths["frequency_table"])

    return pos_df.T