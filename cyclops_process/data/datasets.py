from tqdm import tqdm
from typing import List, Tuple
from collections import defaultdict
from prettytable import PrettyTable
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import yaml
from numpy.typing import ArrayLike
from iohub.ngff import open_ome_zarr
from skimage.measure import regionprops

from ops_utils.data.experiment import OpsDataset
from stitch.connect import read_shifts_biahub



def match_cell(
    points: List[Tuple[int, int]],
    segmentation: ArrayLike,
):
    """
    Given a point and a segmentation, read the cell ID of that point
    """
    # Assume points are passed in (Y,X) format
    y_points = points[:, 0].astype(int)
    x_points = points[:, 1].astype(int)
    segmentation = np.asarray(segmentation)
    cell_ids = segmentation[0, 0, 0, np.array(y_points), np.array(x_points)]

    return cell_ids


def get_inv_transform(
    transform_path: str,
):
    """
    Get the inverse affine transform from the yaml file
    """

    with open(transform_path, "r") as file:
        raw_settings = yaml.safe_load(file)
    # Biahub directly saves the inverse transform
    transform = np.asarray(raw_settings["affine_transform_zyx"])

    return transform


def apply_inv_transform(
    points: List[Tuple[int, int]],
    inv_transform: np.ndarray,
) -> np.ndarray:
    """
    Apply the inverse affine transform to the points
    """

    # TODO: visually confim this in napari using bc_segmentation

    # trasnform expects things to be in the order yx
    points = np.fliplr(np.asarray(points))
    points = np.concatenate(
        [np.zeros((points.shape[0], 1)), points, np.ones((points.shape[0], 1))], axis=1
    )
    points = inv_transform @ points.T
    points = points[1:3, :]

    return points.T


def link_metrics_summary(all_metrics: List[dict], experiment: str, dataset) -> None:
    """
    Print and save summary metrics for the linking pipeline.

    Args:
        all_metrics: List of metrics dictionaries, one per well
        experiment: Experiment name for CSV filename
        dataset: OpsDataset instance for saving metrics CSV
    """
    # Print per-well summary tables
    for metrics in all_metrics:
        well = metrics["well"]
        print(f"\n=== Summary for {well} ===")
        table = PrettyTable()
        table.field_names = ["Stage", "Unique Tracks", "Million"]
        table.align["Stage"] = "l"
        table.align["Unique Tracks"] = "r"
        table.align["Million"] = "r"

        table.add_row(
            [
                "Full length tracks",
                f"{metrics['1_full_length_tracks']:,}",
                f"{metrics['1_full_length_tracks']/1e6:.2f}",
            ]
        )
        table.add_row(
            [
                "ISS cells linked",
                f"{metrics['2_iss_linked']:,}",
                f"{metrics['2_iss_linked']/1e6:.2f}",
            ]
        )
        table.add_row(
            [
                "After ISS linking",
                f"{metrics['3_iss_linked_with_barcode']:,}",
                f"{metrics['3_iss_linked_with_barcode']/1e6:.2f}",
            ]
        )
        table.add_row(
            [
                "After pheno linking",
                f"{metrics['4_pheno_linked']:,}",
                f"{metrics['4_pheno_linked']/1e6:.2f}",
            ]
        )
        table.add_row(
            [
                "After ISS-pheno merge",
                f"{metrics['5_tracks_with_gene_matched']:,}",
                f"{metrics['5_tracks_with_gene_matched']/1e6:.2f}",
            ]
        )
        table.add_row(
            [
                "After gene merge",
                f"{metrics['6_after_bbox']:,}",
                f"{metrics['6_after_bbox']/1e6:.2f}",
            ]
        )
        table.add_row(
            [
                "  └─ Valid segmentation",
                f"{metrics['6_valid_segmentation']:,}",
                f"{metrics['6_valid_segmentation']/1e6:.2f}",
            ]
        )
        table.add_row(
            [
                "  └─ Fallback 200px bbox",
                f"{metrics['6_fallback_bbox']:,}",
                f"{metrics['6_fallback_bbox']/1e6:.2f}",
            ]
        )
        if "7_after_dedupe" in metrics:
            table.add_row(
                [
                    "After dedupe + NaN-seg drop",
                    f"{metrics['7_after_dedupe']:,}",
                    f"{metrics['7_after_dedupe']/1e6:.2f}",
                ]
            )
            table.add_row(
                [
                    "  └─ Multiplets dropped",
                    f"{metrics['7_multiplets_dropped']:,}",
                    f"{metrics['7_multiplets_dropped']/1e6:.2f}",
                ]
            )
            if "7_nan_seg_dropped" in metrics:
                table.add_row(
                    [
                        "  └─ NaN-seg fallback-bbox dropped",
                        f"{metrics['7_nan_seg_dropped']:,}",
                        f"{metrics['7_nan_seg_dropped']/1e6:.2f}",
                    ]
                )

        print(table)

        # Print detailed merge breakdown
        if "merge_breakdown" in metrics:
            mb = metrics["merge_breakdown"]
            print(f"\n=== Merge Breakdown for {well} ===")
            merge_table = PrettyTable()
            merge_table.field_names = ["Category", "Count", "Million", "% of Full Tracks"]
            merge_table.align["Category"] = "l"
            merge_table.align["Count"] = "r"
            merge_table.align["Million"] = "r"
            merge_table.align["% of Full Tracks"] = "r"

            total_tracks = metrics['1_full_length_tracks']

            merge_table.add_row(
                [
                    "ISS-Pheno matched",
                    f"{mb['iss_pheno_matched']:,}",
                    f"{mb['iss_pheno_matched']/1e6:.2f}",
                    f"{100*mb['iss_pheno_matched']/total_tracks:.1f}%",
                ]
            )
            merge_table.add_row(
                [
                    "  └─ With barcode=0",
                    f"{mb['zero_barcode']:,}",
                    f"{mb['zero_barcode']/1e6:.2f}",
                    f"{100*mb['zero_barcode']/total_tracks:.1f}%",
                ]
            )
            merge_table.add_row(
                [
                    "  └─ With nonzero barcode",
                    f"{mb['nonzero_barcode']:,}",
                    f"{mb['nonzero_barcode']/1e6:.2f}",
                    f"{100*mb['nonzero_barcode']/total_tracks:.1f}%",
                ]
            )
            merge_table.add_row(
                [
                    "      └─ Gene matched",
                    f"{mb['after_gene_merge']:,}",
                    f"{mb['after_gene_merge']/1e6:.2f}",
                    f"{100*mb['after_gene_merge']/total_tracks:.1f}%",
                ]
            )
            merge_table.add_row(
                [
                    "      └─ Gene unmatched",
                    f"{mb['unmatched_genes']:,}",
                    f"{mb['unmatched_genes']/1e6:.2f}",
                    f"{100*mb['unmatched_genes']/total_tracks:.1f}%",
                ]
            )
            merge_table.add_row(
                [
                    "ISS only (no pheno match)",
                    f"{mb['iss_only']:,}",
                    f"{mb['iss_only']/1e6:.2f}",
                    f"{100*mb['iss_only']/total_tracks:.1f}%",
                ]
            )
            merge_table.add_row(
                [
                    "Pheno only (no ISS match)",
                    f"{mb['pheno_only']:,}",
                    f"{mb['pheno_only']/1e6:.2f}",
                    f"{100*mb['pheno_only']/total_tracks:.1f}%",
                ]
            )

            print(merge_table)

    # Save metrics to CSV (flatten merge_breakdown for CSV export)
    metrics_for_csv = []
    for m in all_metrics:
        row = {k: v for k, v in m.items() if k != "merge_breakdown"}
        if "merge_breakdown" in m:
            # Flatten merge breakdown into individual columns
            for key, value in m["merge_breakdown"].items():
                row[f"merge_{key}"] = value
        metrics_for_csv.append(row)

    metrics_df = pd.DataFrame(metrics_for_csv)
    metrics_path = dataset.result_paths["link_metrics"]
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n=== Metrics saved to: {metrics_path} ===")

    # Print final summary table for all wells
    if len(all_metrics) > 1:
        print(f"\n=== Final Summary (All Wells) ===")
        summary_table = PrettyTable()
        summary_table.field_names = [
            "Well",
            "Initial",
            "Full-Length",
            "ISS Linked",
            "Pheno Linked",
            "Final Output",
        ]
        summary_table.align = "r"
        summary_table.align["Well"] = "l"

        for m in all_metrics:
            summary_table.add_row(
                [
                    m["well"],
                    f"{m['1_full_length_tracks']/1e6:.2f}M",
                    f"{m['2_iss_linked']/1e6:.2f}M",
                    f"{m['3_iss_linked_with_barcode']/1e6:.2f}M",
                    f"{m['5_tracks_with_gene_matched']/1e6:.2f}M",
                    f"{m['6_after_bbox']/1e6:.2f}M",
                ]
            )

        print(summary_table)


def read_tracks_geff(dataset, well, skip_track: bool = False) -> pd.DataFrame:
    import tracksdata as td
    import rustworkx as rx

    # Handle both old (2 values) and new (3+ values) tracksdata API
    result = td.graph.InMemoryGraph().from_geff(
        dataset.append_well("tracking_geff", well)
    )
    if isinstance(result, tuple):
        graph, metadata = result[0], result[1]
    else:
        graph, metadata = result, {}

    # Determine ops number for track length logic
    ops_num = int(dataset.experiment.split("_")[0].replace("ops", ""))
    well_num = int(well.split("/")[1])

    # First, detect actual number of timepoints in the graph
    node_ids = graph.node_ids()
    actual_timepoints = set()
    for node_id in node_ids:
        node_data = graph.rx_graph[node_id]
        actual_timepoints.add(node_data["t"])

    actual_num_timepoints = len(actual_timepoints)
    print(f"Detected {actual_num_timepoints} timepoints in graph: {sorted(actual_timepoints)}")

    # Determine expected track length based on experiment number and well
    if skip_track:
        len_full_tracks = 2
        print(f"skip_track mode: expecting 2 timepoints (pheno + ISS only)")
    elif ops_num < 69:
        if well == "A/1/0":
            len_full_tracks = 5  # pheno + track t1,t2,t3 + ISS
        elif well == "A/2/0":
            len_full_tracks = 4  # pheno + track t2,t3 + ISS
        elif well == "A/3/0":
            len_full_tracks = 3  # pheno + track t3 + ISS
        else:
            raise ValueError(f"Well {well} not recognized")
    else:
            # New logic (ops >= 69): Well 1 uses tracking t0+t1, well 2 uses t1 only, well 3 uses phenotyping only
            # A/1/0: pheno(t=0), tracking(t=0), tracking(t=1), ISS(t=0) -> 4 frames
            # A/2/0: pheno(t=0), tracking(t=1), ISS(t=0) -> 3 frames (but may only have 2 if ISS missing)
            # A/3/0: pheno(t=0), ISS(t=1) -> 2 frames
            if well == "A/1/0":
                # New graphs have 4 frames (pheno, track_t0, track_t1, ISS)
                # Old graphs have 3 frames (pheno, track_t1, ISS) - use actual for backwards compat
                len_full_tracks = actual_num_timepoints
                if actual_num_timepoints not in (4,):
                    print(f"Warning: {well} (ops>=69) unexpected {actual_num_timepoints} timepoints")
            elif well == "A/2/0":
                # For A/2/0, use actual number of timepoints if less than expected
                len_full_tracks = min(3, actual_num_timepoints)
                if actual_num_timepoints < 3:
                    print(f"Warning: {well} expected 3 timepoints but found {actual_num_timepoints}, adjusting track length")
            elif well == "A/3/0":
                len_full_tracks = 2
            else:
                raise ValueError(f"Well {well} not recognized")

    tracks_dict = {}
    for i in node_ids:
        a = rx.ancestors(graph.rx_graph, i)
        if len(a) == len_full_tracks - 1:
            tracks_dict[i] = a
    print(f"Found {len(tracks_dict)} full length tracks, {len_full_tracks} frames")

    track_info = defaultdict(list)

    # Determine if we're missing the tracking timepoint
    # This happens in two cases:
    # 1. skip_track mode (all wells have only pheno + ISS)
    # 2. A/2/0 with only 2 timepoints (tracking data missing)
    missing_tracking = skip_track or (well == "A/2/0" and actual_num_timepoints == 2)

    for k, v in tracks_dict.items():
        v = tracks_dict[k].copy()
        v.add(k)
        for i in v:
            node_data = graph.rx_graph[i]

            if node_data["t"] == 0:
                track_info["y_pheno"].append(node_data["y"])
                track_info["x_pheno"].append(node_data["x"])
            elif missing_tracking and node_data["t"] == 1:
                # When tracking is skipped, t=1 is ISS
                track_info["y_iss"].append(node_data["y"])
                track_info["x_iss"].append(node_data["x"])
            elif node_data["t"] == len_full_tracks - 1:
                track_info["y_iss"].append(node_data["y"])
                track_info["x_iss"].append(node_data["x"])
            else:
                track_info[f'y_tracking_t{node_data["t"]}'].append(node_data["y"])
                track_info[f'x_tracking_t{node_data["t"]}'].append(node_data["x"])

    return pd.DataFrame(track_info)


def link_calls_tracks(
    experiment: str,
    wells: List[str] = ["A/1/0", "A/2/0", "A/3/0"],
    method: str = "mine",
    confidence_threshold: float = 0.95,
    iss_rounds: List[int] = None,
    failed_rounds_by_well: dict = None,
    n_jobs: int = None,
    skip_track: bool = False,
):
    # Lazy imports to speed up module load time
    from cyclops_process.metrics.plate_stats.match_reads import (
        _get_effective_iss_rounds,
        get_read_codebook_positions,
    )
    from ops_utils.hpc.resource_manager import get_optimal_workers

    method = "mine"
    # loop over wells
    print(f"Linking base calling results to tracking results for {experiment}")
    print(f"Using base calling method: {method}")
    if skip_track:
        print("⚠️  WARNING: skip_track=True - linking will use ISS and pheno only (no tracking timepoints)")
    dataset = OpsDataset(experiment, method=method)
    gene_df = dataset.load_gene_index()

    # Default to first 10 rounds if not specified
    if iss_rounds is None:
        print("Warning: ISS ROUNDS NOT DECALRED, using defaault 0-9")
        iss_rounds = list(range(10))

    print(f"Requested ISS rounds: {iss_rounds}")
    if failed_rounds_by_well:
        print(f"Failed rounds by well: {failed_rounds_by_well}")

    # Note: gene_df barcode filtering will be done per-well in _process_well
    # to account for per-well failed rounds

    # Determine number of workers
    if n_jobs is None:
        n_jobs = get_optimal_workers(use_gpu=False, verbose=False)
        n_jobs = max(1, n_jobs)

    print(f"Processing {len(wells)} wells in parallel with {n_jobs} workers")

    def _process_well(well):
        # Initialize metrics dictionary for this well
        metrics = {"well": well}

        # Read↔gene position mapping for this well (handles dropout + offset/shift).
        # read_positions index the read barcode; codebook_positions index the
        # gene-index barcode. For plain dropout/none these are equal; for an offset
        # (e.g. no-incorporation round 0) they differ so read round c+N matches
        # gene position c.
        _code_len = len(str(gene_df["barcode"].iloc[0])) if len(gene_df) else max(iss_rounds) + 1
        read_positions, codebook_positions = get_read_codebook_positions(
            iss_rounds, well, failed_rounds_by_well, _code_len
        )
        print(f"[{well}] read positions {read_positions} -> gene positions {codebook_positions}")

        # Filter gene_df barcodes to the codebook positions
        well_gene_df = gene_df.copy()
        well_gene_df["barcode"] = well_gene_df["barcode"].apply(
            lambda x: "".join([x[i] for i in codebook_positions if i < len(x)])
        )

        track_results = read_tracks_geff(dataset, well, skip_track=skip_track)
        metrics["1_full_length_tracks"] = len(track_results)

        # ISS linking
        print(f"[{well}] ISS linking {len(track_results)} tracks...")
        iss_points = track_results[["y_iss", "x_iss"]].to_numpy().astype(np.uint16)
        iss_linked = link_tracking_iss(
            iss_points, dataset, well, method, confidence_threshold
        )
        metrics["2_iss_linked"] = len(iss_linked)
        metrics["3_iss_linked_with_barcode"] = int(np.sum(iss_linked.barcode != 0))
        print(f"[{well}] ISS linked: {len(iss_linked)} tracks, {metrics['3_iss_linked_with_barcode']} with barcode")

        # Pheno linking
        print(f"[{well}] Pheno linking...")
        pheno_points = track_results[["y_pheno", "x_pheno"]].to_numpy()
        pheno_linked = link_tracking_phenotyping(pheno_points, dataset, well, skip_track=skip_track)
        metrics["4_pheno_linked"] = len(pheno_linked)
        print(f"[{well}] Pheno linked: {len(pheno_linked)} tracks")

        # Tracking linking
        tracking_df = (
            track_results.copy()
            .drop(columns=["y_iss", "x_iss", "y_pheno", "x_pheno"])
            .astype(np.uint16)
        )
        tracking_df["og_index"] = np.arange(len(tracking_df))

        # Detailed merge statistics
        merge_stats = {}

        # Step 1: ISS + Pheno merge (outer to see what's lost)
        iss_pheno_merge = pd.merge(iss_linked, pheno_linked, on="og_index", how="outer", indicator=True)
        merge_stats["iss_only"] = int((iss_pheno_merge["_merge"] == "left_only").sum())
        merge_stats["pheno_only"] = int((iss_pheno_merge["_merge"] == "right_only").sum())
        merge_stats["iss_pheno_matched"] = int((iss_pheno_merge["_merge"] == "both").sum())

        # Step 2: Inner join for actual processing
        merged_temp = pd.merge(iss_linked, pheno_linked, on="og_index", how="inner")
        merged_temp = pd.merge(merged_temp, tracking_df, on="og_index", how="inner")

        # Extract read positions from barcodes (paired with codebook_positions above)
        merged_temp["barcode"] = merged_temp["barcode"].apply(
            lambda x: "".join([str(x)[i] for i in read_positions if i < len(str(x))]) if x != 0 else "0"
        )

        # Step 3: Barcode breakdown before gene merge
        merge_stats["zero_barcode"] = int((merged_temp["barcode"] == "0").sum())
        merge_stats["nonzero_barcode"] = int((merged_temp["barcode"] != "0").sum())

        # Step 4: Gene merge (removes zero barcodes and unmatched genes)
        merged = pd.merge(merged_temp, well_gene_df, on="barcode", how="inner")
        merge_stats["after_gene_merge"] = len(merged)

        # Calculate cells lost to unmatched genes (nonzero barcodes that didn't match gene_df)
        merge_stats["unmatched_genes"] = merge_stats["nonzero_barcode"] - merge_stats["after_gene_merge"]

        metrics["5_tracks_with_gene_matched"] = len(merged)
        metrics["merge_breakdown"] = merge_stats
        print(f"[{well}] Gene merge: {len(merged)} tracks matched")

        out = merged.loc[:, ~merged.columns.str.startswith("Unnamed")]

        # Rename gene identifier column "Gene name" → "gene_name". The custom
        # rename for custom-perturbation libraries (gene_name → the configured output column)
        # happens AFTER dedupe by convention — dedupe_linked_csv auto-detects
        # both `gene_name` and the configured output column so order doesn't
        # affect correctness, but keeping the canonical name through dedupe
        # makes the metrics log easier to read.
        if "Gene name" in out.columns:
            out = out.rename(columns={"Gene name": "gene_name"})

        # Non-targeting controls have an empty "Gene name" in the gene_index
        # (NCBI_ID=-1), so they land here as NaN gene_name. Label them "NTC".
        if "gene_name" in out.columns:
            out["gene_name"] = out["gene_name"].fillna("NTC")

        print(f"[{well}] Computing bounding boxes for {len(out)} cells...")
        out = cell_bounding_boxes(out, dataset, well)
        metrics["6_after_bbox"] = len(out)
        metrics["6_fallback_bbox"] = int(out["segmentation_id"].isna().sum())
        metrics["6_valid_segmentation"] = int(out["segmentation_id"].notna().sum())
        print(f"[{well}] Bboxes done: {metrics['6_valid_segmentation']} segmented, {metrics['6_fallback_bbox']} fallback")

        # Link tracking 5x coordinates
        print(f"[{well}] Linking 5x tracking coordinates...")
        track_5x_linked = link_tracking_5x(tracking_df, dataset, well)

        # Merge tracking coordinates with output NOTE: this will not remove any cells that are not in the 5x space
        if not track_5x_linked.empty and "og_index" in track_5x_linked.columns:
            out = pd.merge(out, track_5x_linked, on="og_index", how="left")

        # Dedupe: drop multiplet cells (≥2 distinct barcodes at the same
        # (well, tile_pheno, segmentation_id)) AND fallback-bbox cells
        # (segmentation_id is NaN — no valid cell boundary, just a centroid
        # window from cell_bounding_boxes). `dedupe_linked_csv` keys on
        # `barcode` for both CRISPR (1:1 with sgRNA) and custom-perturbation
        # (sgRNA-less) experiments. The custom rename below
        # (gene_name → the configured output column) still happens after
        # dedupe by convention; the gene-tie-break col is auto-detected.
        from cyclops_process.utils.dedupe_linked_pheno_iss import dedupe_linked_csv
        n_before_dedupe = len(out)
        n_nan_seg_dropped = int(out["segmentation_id"].isna().sum())
        well_token = well.replace("/", "")
        # Canonicalize to "A1"-style (e.g. "A/1/0" → "A1") to match the well
        # value derived from the CSV filename downstream.
        out["well"] = well_token[:2] if len(well_token) >= 2 else well_token
        out = dedupe_linked_csv(out)
        out = out.drop(columns=["well"])
        metrics["7_after_dedupe"] = len(out)
        metrics["7_nan_seg_dropped"] = n_nan_seg_dropped
        metrics["7_multiplets_dropped"] = (
            n_before_dedupe - n_nan_seg_dropped - len(out)
        )
        print(f"[{well}] Dedup: {n_before_dedupe} → {len(out)} "
              f"({metrics['7_multiplets_dropped']} multiplets + "
              f"{n_nan_seg_dropped} fallback-bbox cells dropped)")

        # For experiments with a custom gene name output column (e.g. a label-free perturbation-name column)
        if dataset.gene_name_output_column and "gene_name" in out.columns:
            out = out.rename(columns={"gene_name": dataset.gene_name_output_column})

        # Reorder columns to group tile/local coordinates together after segmentation_id
        # Move tile_pheno, y_local_pheno, x_local_pheno to after segmentation_id
        if 'tile_pheno' in out.columns and 'segmentation_id' in out.columns:
            cols = out.columns.tolist()
            # Remove pheno tile columns from their current position
            pheno_tile_cols = ['tile_pheno', 'y_local_pheno', 'x_local_pheno']
            other_cols = [c for c in cols if c not in pheno_tile_cols]

            # Find position of segmentation_id and insert pheno tile columns after it
            seg_id_idx = other_cols.index('segmentation_id')
            reordered_cols = other_cols[:seg_id_idx+1] + pheno_tile_cols + other_cols[seg_id_idx+1:]
            out = out[reordered_cols]

        # Save output
        out_path = dataset.append_well("linked_results", well)
        print(f"[{well}] Saving {len(out)} linked results...")
        out.to_csv(out_path, index=False)

        # Guard assertion: re-read and confirm zero remaining multiplets.
        # Cheap (one groupby/nunique) and catches dedupe regressions immediately.
        chk = pd.read_csv(out_path, low_memory=False)
        chk["_well"] = well_token[:2] if len(well_token) >= 2 else well_token
        chk["_k"] = (
            chk["_well"].astype(str) + "|"
            + chk["tile_pheno"].astype(str) + "|"
            + chk["segmentation_id"].astype(str)
        )
        guide_col = "barcode" if "barcode" in chk.columns else None
        n_multi_remaining = 0 if guide_col is None else int(
            (chk.dropna(subset=[guide_col]).groupby("_k")[guide_col].nunique() > 1).sum()
        )
        if n_multi_remaining != 0:
            raise RuntimeError(
                f"[{well}] dedupe regression: {n_multi_remaining} multiplet cells "
                f"in {out_path} after dedupe step"
            )
        print(f"[{well}] Done!")

        return metrics

    # Process wells in parallel
    all_metrics = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_process_well)(well) for well in wells
    )

    # Print summary tables and save metrics CSV
    link_metrics_summary(all_metrics, experiment, dataset)

    return


def _load_cell_seg_v3(dataset: OpsDataset, well: str) -> tuple:
    """
    Try to load cell_seg from pheno_assembled_v3 (zarr v3 store).

    Returns:
        Tuple of (segmentation_array, success, message)
        - segmentation_array: 2D numpy array (loaded into memory)
        - success: bool
        - message: str describing source or error
    """
    import zarr

    v3_path = dataset.store_paths.get("pheno_assembled_v3")
    if v3_path is None or not v3_path.exists():
        return None, False, "pheno_assembled_v3 store not found"

    try:
        store = zarr.open(str(v3_path), mode="r")

        # Check if cell_seg label exists for this well
        labels_path = f"{well}/labels/cell_seg/0"
        if labels_path not in store:
            return None, False, f"cell_seg label not found at {labels_path}"

        # Load full array into memory
        lazy_seg_array = store[labels_path]
        segmentation_array = np.asarray(lazy_seg_array[0, 0, 0, :, :])

        return segmentation_array, True, "cell_seg (v3)"

    except Exception as e:
        return None, False, f"Error loading cell_seg from v3: {e}"


PHENO_NUCLEAR_SEG_5X_LEVEL = 2  # native-20x pyramid: level 0 = 20x, level 2 = 5x


def load_pheno_nuclear_seg_v3(
    dataset: OpsDataset, well: str, pyramid_level: int = PHENO_NUCLEAR_SEG_5X_LEVEL
):
    """Lazy zarr array for the phenotyping_v3 ``nuclear_seg`` label of a well.

    Nuclei are segmented at native 20x; level 2 is the 5x view tracking and
    registration operate in. Returns None if the store or label is missing.
    """
    import zarr

    v3_path = dataset.store_paths.get("pheno_assembled_v3")
    if v3_path is None or not v3_path.exists():
        return None
    store = zarr.open(str(v3_path), mode="r")
    label_path = f"{well}/labels/nuclear_seg/{pyramid_level}"
    return store[label_path] if label_path in store else None


def _load_seg_v2(dataset: OpsDataset, well: str) -> tuple:
    """
    Load seg from pheno_assembled (zarr v2 store) - legacy fallback.

    Returns:
        Tuple of (segmentation_array, success, message)
        - segmentation_array: 2D numpy array (loaded into memory)
        - success: bool
        - message: str describing source or error
    """
    import zarr

    v2_path = dataset.store_paths.get("pheno_assembled")
    if v2_path is None:
        return None, False, "pheno_assembled not in store_paths"
    if not v2_path.exists():
        return None, False, f"pheno_assembled store not found at {v2_path}"

    try:
        # seg/0 is often a symlink to the actual segmentation array
        # Use raw zarr instead of iohub since the symlink target may not be OME-NGFF
        seg_path = v2_path / well / "seg" / "0"
        if not seg_path.exists():
            return None, False, f"seg path not found: {seg_path}"

        # Resolve symlink and open with raw zarr (not iohub)
        resolved_path = seg_path.resolve()
        lazy_seg_array = zarr.open(str(resolved_path), mode="r")

        # Load full array and squeeze to 2D if needed
        if lazy_seg_array.ndim > 2:
            segmentation_array = np.asarray(lazy_seg_array[0, 0, 0, :, :])
        else:
            segmentation_array = np.asarray(lazy_seg_array[:, :])

        return segmentation_array, True, "seg (v2 legacy)"

    except Exception as e:
        return None, False, f"Error loading seg from v2: {e}"


def cell_bounding_boxes(
    results: pd.DataFrame, dataset: OpsDataset, well: str
) -> pd.DataFrame:
    """
    Add bounding boxes for the cell segmentation.

    Priority order:
    1. cell_seg from pheno_assembled_v3 (zarr v3 store) - new tiled IoU-stitched cell segmentation
    2. seg from pheno_assembled (zarr v2 store) - legacy stitched cell segmentation

    Loads full array into memory for fast regionprops computation.
    """
    bbox_list = []
    seg_id_list = []
    results_dict = None
    segmentation_array = None
    seg_source = None

    # Try v3 store first, then fall back to v2
    segmentation_array, success, message = _load_cell_seg_v3(dataset, well)
    if success:
        seg_source = message
        print(f"[bbox] ✓ Using {seg_source} for {well}")
    else:
        print(f"[bbox] ⚠ Could not load cell_seg from v3 store: {message}")
        print(f"[bbox]   Falling back to legacy cell segmentation (v2)...")

        segmentation_array, success, message = _load_seg_v2(dataset, well)
        if success:
            seg_source = message
            print(f"[bbox] ✓ Using {seg_source} for {well}")
        else:
            print(f"[bbox] ✗ ERROR: Could not load segmentation for well {well}: {message}")
            results["segmentation_id"] = np.nan
            results["bbox"] = None
            return results

    # Compute bounding boxes once
    label_props = regionprops(segmentation_array, cache=False)
    results_dict = {obj.label: {"bbox": obj.bbox} for obj in label_props}

    print(
        f"[bbox] Loaded segmentation: shape={segmentation_array.shape}, {len(results_dict)} cells"
    )

    # Vectorized bbox computation (replaces slow iterrows loop)
    fallback_bbox_size = 200
    half_size = fallback_bbox_size // 2
    h, w = segmentation_array.shape

    y_coords = results["y_pheno"].values.astype(int)
    x_coords = results["x_pheno"].values.astype(int)

    # Check bounds
    in_bounds = (y_coords >= 0) & (x_coords >= 0) & (y_coords < h) & (x_coords < w)

    # Lookup seg IDs for in-bounds points (clip to avoid index errors on out-of-bounds)
    y_safe = np.clip(y_coords, 0, h - 1)
    x_safe = np.clip(x_coords, 0, w - 1)
    seg_ids = segmentation_array[y_safe, x_safe].astype(int)
    seg_ids[~in_bounds] = 0

    # Build bbox lookup array from regionprops (label -> bbox tuple)
    valid_seg = (seg_ids > 0) & np.array([sid in results_dict for sid in seg_ids])

    # Build results
    seg_id_out = np.full(len(results), np.nan)
    seg_id_out[valid_seg] = seg_ids[valid_seg]

    bbox_out = [None] * len(results)
    # Valid segmentation bboxes
    for idx in np.where(valid_seg)[0]:
        bbox_out[idx] = results_dict[seg_ids[idx]]["bbox"]
    # Fallback bboxes for invalid/missing segmentation
    fallback_mask = ~valid_seg
    for idx in np.where(fallback_mask)[0]:
        y, x = y_coords[idx], x_coords[idx]
        bbox_out[idx] = (max(0, y - half_size), max(0, x - half_size),
                         min(h, y + half_size), min(w, x + half_size))

    results["bbox"] = bbox_out
    results["segmentation_id"] = seg_id_out

    # No longer dropping cells without valid segmentation - using fallback bboxes instead
    # results = results.dropna(subset=["segmentation_id"]) # Drop rows where spot does not match to a segmented cell

    return results


def link_tracking_iss(
    track_iss_points: pd.DataFrame,
    dataset: OpsDataset,
    well: str,
    method: str = "mine",
    confidence_threshold: float = 0.95,
) -> pd.DataFrame:
    """
    track_iss_points: numpy array of shape (N, 2) with (Y, X) points at the ISS timepoint
    """
    import dask.array as da  # Lazy import to speed up module load time

    iss_reads = pd.read_csv(dataset.append_well("reads", well))
    if method == "probabilistic":
        iss_reads = iss_reads[iss_reads["confidence"] >= confidence_threshold]
        # print(f"# of iss reads after confidence threshold: {len(iss_reads)}")

    iss_transform = get_inv_transform(dataset.append_well("auto_iss_register", well)) # manual: iss_seg_register
    iss_affine_input = track_iss_points[:, ::-1]  # apply_inv_transform expects (X,Y)
    iss_points_unreg = apply_inv_transform(
        iss_affine_input, iss_transform
    )  # output is (Y,X))

    iss_seg_store = open_ome_zarr(dataset.store_paths["iss_segmentation"] / well, layout="fov", mode="r+")
    iss_seg = da.from_array(iss_seg_store["1"])
    iss_cell_ids = match_cell(iss_points_unreg, iss_seg).astype(int)

    iss_reads = iss_reads[iss_reads["cell"] != 0]
    cell_barcode_lookup = iss_reads.set_index("cell")["barcode"].to_dict()

    barcodes = [cell_barcode_lookup.get(cell_id, 0) for cell_id in iss_cell_ids]

    iss_results = pd.DataFrame()
    iss_results["og_index"] = np.arange(len(track_iss_points))
    iss_results["y"] = iss_points_unreg[:, 0]
    iss_results["x"] = iss_points_unreg[:, 1]

    iss_results = iss_results.rename(
        columns={col: f"{col}_iss" for col in iss_results.columns if col != "og_index"}
    )
    iss_results["barcode"] = barcodes

    return iss_results


def link_tracking_phenotyping(
    track_pheno_points: pd.DataFrame,
    dataset: OpsDataset,
    well: str,
    skip_track: bool = False,
) -> pd.DataFrame:
    """
    Link phenotyping coordinates to tile locations.

    Returns DataFrame with columns: og_index, y_pheno, x_pheno (global registered coords),
    tile_pheno, y_local_pheno, x_local_pheno
    """
    # Determine if tracking data exists based on config and experiment number
    has_tracking = not skip_track

    # Check ops number for well-specific logic (ops>=69 A/3/0 has no tracking)
    if has_tracking:
        try:
            ops_num = int(dataset.experiment.split("_")[0].replace("ops", ""))
            well_token = well.replace("/", "")
            well_token_short = well_token[0] + well_token[1] if len(well_token) >= 2 else well_token
            if ops_num >= 69 and well_token_short == "A3":
                has_tracking = False
        except (ValueError, IndexError):
            pass

    if not has_tracking:
        # In skip_track mode or ops>=69 A/3/0, pheno is the reference frame
        # The tracking graph t=0 coords are already in pheno space (at pyramid level 2)
        # No registration transform needed - just use identity (4x upscale applied separately)
        pheno_transform = np.identity(4)
    else:
        pheno_transform = get_inv_transform(
            dataset.append_well("auto_pheno_register", well) # manual: lc_20x_seg_register
        )
    upscale_diag = [1, 4, 4, 1]
    upscale_affine = np.diag(upscale_diag)
    pt_scale = upscale_affine @ pheno_transform
    pheno_affine_input = track_pheno_points[
        :, ::-1
    ]  # apply_inv_transform expects (X,Y)
    pheno_points_unreg = apply_inv_transform(pheno_affine_input, pt_scale)

    # Create basic results with global coordinates
    pheno_results = pd.DataFrame()
    pheno_results["og_index"] = np.arange(len(track_pheno_points))
    pheno_results["y_pheno"] = pheno_points_unreg[:, 0]
    pheno_results["x_pheno"] = pheno_points_unreg[:, 1]

    # Get tile location information
    pheno_shifts = read_shifts(dataset, "lc_20x_stitch", well)

    tile_results = get_local_tile(
        pheno_points_unreg,
        shifts=pheno_shifts,
        window_size=0,
    )

    # # Validation: check that global coordinates from get_local_tile match the pheno_results coordinates
    # if len(tile_results) > 0:
    #     # Get first matching cell that has tile info
    #     first_og_idx = tile_results["og_index"].iloc[0]
    #     if first_og_idx < len(pheno_results):
    #         original_y = pheno_results.loc[first_og_idx, "y_pheno"]
    #         original_x = pheno_results.loc[first_og_idx, "x_pheno"]
    #         tile_y = tile_results[tile_results["og_index"] == first_og_idx]["y_global"].iloc[0]
    #         tile_x = tile_results[tile_results["og_index"] == first_og_idx]["x_global"].iloc[0]

    #         # Allow small floating point differences
    #         if not (np.isclose(original_y, tile_y, atol=1.0) and np.isclose(original_x, tile_x, atol=1.0)):
    #             print(f"Warning: PHENO coordinate mismatch for {well}!")
    #             print(f"  Original y_pheno/x_pheno: y={original_y}, x={original_x}")
    #             print(f"  Tile y_global/x_global: y={tile_y}, x={tile_x}")
    #             print(f"  Difference: dy={abs(original_y - tile_y):.1f}, dx={abs(original_x - tile_x):.1f}")

    # Merge tile info (keep only tile and local coordinates, not global since we already have y_pheno/x_pheno)
    tile_results = tile_results[["og_index", "tile", "y_local", "x_local"]].rename(
        columns={
            "tile": "tile_pheno",
            "y_local": "y_local_pheno",
            "x_local": "x_local_pheno"
        }
    )

    # Merge with left join to preserve all cells
    pheno_results = pd.merge(pheno_results, tile_results, on="og_index", how="left")

    return pheno_results


def link_tracking_5x(
    tracking_df: pd.DataFrame,
    dataset: OpsDataset,
    well: str,
) -> pd.DataFrame:
    """
    Link tracking coordinates to the 5x zarr tile locations.

    Args:
        tracking_df: DataFrame with columns like y_tracking_t1, x_tracking_t1, etc.
        dataset: OpsDataset instance
        well: Well identifier

    Returns:
        DataFrame with og_index and tracking coordinate columns (y, x, y_local, x_local, tile)
        for each timepoint (e.g., y_tracking_t1, x_tracking_t1, y_local_tracking_t1, x_local_tracking_t1, tile_tracking_t1)
    """
    # Find all tracking timepoint columns
    tracking_cols = [col for col in tracking_df.columns if col.startswith('y_tracking_t')]

    if len(tracking_cols) == 0:
        # No tracking timepoints to process
        return pd.DataFrame({'og_index': tracking_df['og_index']})

    # Extract timepoints from column names (e.g., 'y_tracking_t1' -> 1)
    timepoints = sorted([int(col.split('y_tracking_t')[1]) for col in tracking_cols])

    track_shifts = read_shifts(dataset, "lc_5x_stitch", well)

    track_results_list = []

    for t in timepoints:
        # Extract tracking points for this timepoint
        y_col = f'y_tracking_t{t}'
        x_col = f'x_tracking_t{t}'

        if y_col not in tracking_df.columns or x_col not in tracking_df.columns:
            continue

        track_t_points = tracking_df[[y_col, x_col]].to_numpy().astype(np.float64)

        # Apply inverse transformations to global coordinates
        # During stitching, 5x tracking uses: fliplr=True, rot90=1
        # The shifts are for the transformed space, but we need coordinates in original tile space
        # To reverse: first undo rot90, then undo fliplr
        # For the stitched global space, we need to work backwards

        # The stitched coordinates are in the transformed space
        # We need to find which tile they came from in the ORIGINAL (untransformed) space
        # So we DON'T transform the global coordinates - they're already correct for finding the tile
        # But the shifts account for the transformation, so we can use them directly

        # Get local tile information
        tile_results = get_local_tile(
            track_t_points,
            shifts=track_shifts,
            window_size=0,
        )

        # Now apply inverse transformations to LOCAL coordinates
        # The local coordinates from get_local_tile are in the transformed tile space
        # We need to convert them back to original tile space
        if len(tile_results) > 0:
            tile_size = 2048
            y_local = tile_results["y_local"].values.copy()
            x_local = tile_results["x_local"].values.copy()

            # Undo rot90=1 (was rotated 90° clockwise) by rotating 90° counter-clockwise
            # For rot90 counter-clockwise: (y, x) -> (x, tile_size - 1 - y)
            y_temp = x_local
            x_temp = tile_size - 1 - y_local

            # Undo fliplr (was flipped horizontally): x -> tile_size - 1 - x
            x_original = tile_size - 1 - x_temp
            y_original = y_temp

            tile_results["y_local"] = y_original.astype(np.uint16)
            tile_results["x_local"] = x_original.astype(np.uint16)

        # Keep only tile and local coordinates (not global since tracking_df already has y_tracking_tX/x_tracking_tX)
        tile_results = tile_results[["og_index", "tile", "y_local", "x_local"]].rename(
            columns={
                "tile": f"tile_tracking_t{t}",
                "y_local": f"y_local_tracking_t{t}",
                "x_local": f"x_local_tracking_t{t}"
            }
        )

        track_results_list.append(tile_results)

    # Start with all og_index values from tracking_df to ensure no cells are dropped
    # (get_local_tile only returns cells that fall within tile bounds)
    all_indices = pd.DataFrame({'og_index': tracking_df['og_index'].values})

    # Merge all timepoints with all_indices using left join to preserve all cells
    if len(track_results_list) == 0:
        return all_indices

    merged_track_results = all_indices
    for tile_results in track_results_list:
        merged_track_results = pd.merge(merged_track_results, tile_results, on="og_index", how="left")

    return merged_track_results


def get_local_tile(
    points,
    shifts: dict[str, tuple[int, int]],
    window_size: int = 0,
    tile_size: tuple[int, int] = (2048, 2048),
) -> pd.DataFrame:

    tile_x, tile_y = tile_size
    tile_shift_array = np.asarray([v for v in shifts.values()])
    tile_name_array = np.asarray([k for k in shifts.keys()])
    shifts_y = np.expand_dims(tile_shift_array[:, 0], axis=1)
    shifts_x = np.expand_dims(tile_shift_array[:, 1], axis=1)
    py = np.expand_dims(points[:, 0], axis=0)
    px = np.expand_dims(points[:, 1], axis=0)

    contains = (
        (shifts_x <= px)
        & (px < shifts_x + tile_x)
        & (shifts_y <= py)
        & (py < shifts_y + tile_y)
    )
    matching_tiles = np.where(contains)
    tile_shift_pos = tile_shift_array[matching_tiles[0]]
    local_points = points[matching_tiles[1]] - tile_shift_pos
    local_points = local_points.astype(np.uint16)

    mask_1 = local_points - window_size > 0
    mask_2 = local_points + window_size < tile_x

    combined_mask = np.all(mask_1 & mask_2, axis=1)

    valid_tile_indices = matching_tiles[0][combined_mask]
    valid_points = matching_tiles[1][combined_mask]
    valid_local_points = local_points[combined_mask]

    _, unique_indices = np.unique(valid_points, return_index=True)

    tile_names_filtered = tile_name_array[valid_tile_indices[unique_indices]]
    local_points_filtered = valid_local_points[unique_indices]
    points_filtered = points[valid_points[unique_indices]]
    out = pd.DataFrame()
    out["og_index"] = valid_points[unique_indices]
    out["tile"] = tile_names_filtered
    out["y_global"] = points_filtered[:, 0]
    out["x_global"] = points_filtered[:, 1]
    out["y_local"] = local_points_filtered[:, 0]
    out["x_local"] = local_points_filtered[:, 1]

    return out

from ops_utils.data.shifts import read_shifts