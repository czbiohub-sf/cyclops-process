import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from iohub.ngff import open_ome_zarr
from cyclops_utils.data.experiment import OpsDataset
from cyclops_process.metrics.plate_stats.match_reads import (
    _get_effective_iss_rounds,
    match_reads,
)
from cyclops_utils.io.tiling import split_into_tiles
from typing import Dict

import dask.array as da



def read_accuracy(
    experiment: str,
    iss_rounds: list[int] | None = None,
    method: str = "mine",
    failed_rounds_by_well: dict[str, list[int]] | None = None,
    force: bool = False,
) -> Dict:
    """
    Heatmap to visualize spatial effects on read accuracy (mine) or confidence (probabilistic).

    Args:
        iss_rounds: List of ISS round indices to use. If None, defaults to first 10 rounds.
        method: Base calling method ('mine' or 'probabilistic').
        failed_rounds_by_well: Dictionary mapping wells to lists of failed round indices to exclude
        force: If True, regenerate even if output file exists. Default False.
    """
    if iss_rounds is None:
        iss_rounds = list(range(10))

    dataset = OpsDataset(experiment, method=method)

    # Check if output already exists
    output_path = dataset.metrics_paths["read_accuracy_heatmap"]
    if output_path.exists() and not force:
        print(f"Read accuracy heatmap already exists at: {output_path}")
        print("Skipping. Use --force or force=True to regenerate.")
        return None, None
    # get positions
    iss_seg_store = open_ome_zarr(dataset.store_paths["iss_segmentation"])
    pos_list = [a[0] for a in iss_seg_store.positions()]

    if method == "probabilistic":
        metric_label = "Mean Confidence"
        plot_title = "Read Confidence Heatmap"
    else:  # 'mine'
        metric_label = "Read Accuracy"
        plot_title = "Read Accuracy Heatmap"
        codebook_db = dataset.load_codebook()

    plate_metric = {}
    for pos in pos_list:
        # Filter out failed rounds for this well
        well_iss_rounds = _get_effective_iss_rounds(
            iss_rounds, pos, failed_rounds_by_well
        )

        # Debug: Show what rounds are being used for this well
        print(f"\n--- Processing well {pos} for read_accuracy ---")
        print(f"Original iss_rounds: {iss_rounds}")
        print(f"Well-specific iss_rounds after failed_rounds_by_well: {well_iss_rounds}")
        if failed_rounds_by_well and pos in failed_rounds_by_well:
            print(f"Failed rounds config for {pos}: {failed_rounds_by_well[pos]}")

        iss_seg_data = da.array(iss_seg_store[pos].data[0, 0, 0, :, :])
        shape = iss_seg_data.shape[-2:]
        try:
            reads = pd.read_csv(dataset.append_well("reads", pos))
        except FileNotFoundError:
            print(f"Reads file for {pos} not found, skipping heatmap.")
            continue

        if method == "probabilistic" and "confidence" not in reads.columns:
            print(
                f"'confidence' column not found for well {pos}. Cannot generate confidence heatmap. Skipping."
            )
            continue

        tile_list, indx = split_into_tiles(shape, 30, 0)

        if method == "mine":
            matched_reads = match_reads(reads, codebook_db, iss_rounds=well_iss_rounds, well_name=pos, failed_rounds_by_well=failed_rounds_by_well)

        well_metric = []
        for tile in tile_list:
            x_min, x_max, y_min, y_max = tile

            reads_in_tile = reads[
                (reads["i_global"] >= x_min)
                & (reads["i_global"] < x_max)
                & (reads["j_global"] >= y_min)
                & (reads["j_global"] < y_max)
            ]

            if len(reads_in_tile) == 0:
                well_metric.append(0)
            else:
                if method == "probabilistic":
                    mean_confidence = reads_in_tile["confidence"].mean()
                    well_metric.append(mean_confidence)
                else:  # 'mine'
                    matched_reads_in_tile = matched_reads[
                        (matched_reads["i_global"] >= x_min)
                        & (matched_reads["i_global"] < x_max)
                        & (matched_reads["j_global"] >= y_min)
                        & (matched_reads["j_global"] < y_max)
                    ]
                    well_metric.append(len(matched_reads_in_tile) / len(reads_in_tile))
        plate_metric[pos] = well_metric

    if not plate_metric:
        print("No data to plot for heatmap.")
        return None, None

    positions = list(plate_metric.keys())
    num_wells = len(positions)
    # NOTE: `indx` is taken from the last well; assumes all wells have the same tiling grid.
    indx_i = [a[0] for a in indx]
    indx_j = [a[1] for a in indx]
    n = int(np.sqrt(len(indx_i)))  # grid size
    fig, ax = plt.subplots(1, num_wells, figsize=(15, 5))
    if num_wells == 1:  # Ensure ax is always iterable
        ax = [ax]

    for i in range(num_wells):
        out = np.zeros((n, n))
        out[indx_i, indx_j] = plate_metric[positions[i]]
        im = ax[i].imshow(out, vmin=0, vmax=1)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.012)
    cbar.set_label(metric_label)
    fig.suptitle(f"{plot_title}\n{experiment}", fontsize=20)
    plt.savefig(dataset.metrics_paths["read_accuracy_heatmap"], dpi=300)

    return plate_metric, indx


def cells_with_reads_heatmaps(
    experiment: str,
    iss_rounds: list[int] | None = None,
    method: str = "mine",
    confidence_threshold: float = 0.95,
    failed_rounds_by_well: dict[str, list[int]] | None = None,
    force: bool = False,
) -> Dict:
    """
    Generate BOTH heatmaps in one pass per tile using the same underlying counts:
      1) Percent cells with reads per tile = 100 * (#cells-with-reads / #cells)
      2) Normalized (cells-with-reads per cell) per tile = (#cells-with-reads / #cells)

    This matches the statistics metric for percent, and reuses the same numerator
    and denominator for the normalized view (no recompute).

    Args:
        iss_rounds: List of ISS round indices to use. If None, defaults to first 10 rounds.
        method: Base calling method ('mine' or 'probabilistic').
        confidence_threshold: Confidence threshold for probabilistic method.
        failed_rounds_by_well: Dictionary mapping wells to lists of failed round indices to exclude
        force: If True, regenerate even if output file exists. Default False.
    """
    if iss_rounds is None:
        iss_rounds = list(range(10))

    dataset = OpsDataset(experiment, method=method)

    # Check if output already exists
    output_path = dataset.metrics_paths["percent_cells_with_reads_heatmap"]
    if output_path.exists() and not force:
        print(f"Cells-with-reads heatmap already exists at: {output_path}")
        print("Skipping. Use --force or force=True to regenerate.")
        return None, None
    iss_seg_store = open_ome_zarr(dataset.store_paths["iss_segmentation"])
    pos_list = [a[0] for a in iss_seg_store.positions()]

    percent_map = {}
    normalized_map = {}
    good_reads_count_map = {}

    codebook_db = None
    if method != "probabilistic":
        try:
            codebook_db = dataset.load_codebook()
        except Exception as e:
            print(f"Failed to read codebook: {e}")

    for pos in pos_list:
        # Filter out failed rounds for this well
        well_iss_rounds = _get_effective_iss_rounds(
            iss_rounds, pos, failed_rounds_by_well
        )

        iss_seg_data = da.array(iss_seg_store[pos].data[0, 0, 0, :, :])
        shape = iss_seg_data.shape[-2:]

        try:
            reads_df = pd.read_csv(dataset.append_well("reads", pos))
        except FileNotFoundError:
            print(
                f"Reads file for {pos} not found, skipping cells-with-reads heatmaps."
            )
            continue

        required_cols = {"i_global", "j_global", "cell"}
        if not required_cols.issubset(set(reads_df.columns)):
            print(
                f"Required columns {required_cols} not all present for {pos}. Skipping."
            )
            continue

        tile_list, indx = split_into_tiles(shape, 30, 0)

        well_percent = []
        well_norm = []
        well_good_counts = []

        # Prepare good reads according to method for the third heatmap
        good_reads = reads_df
        if method == "probabilistic":
            if "confidence" in reads_df.columns:
                good_reads = reads_df[
                    reads_df["confidence"] >= confidence_threshold
                ].copy()
            else:
                print(
                    f"'confidence' column missing for {pos}; good-reads count will be zero."
                )
                good_reads = reads_df.iloc[0:0].copy()
        else:
            # Use match_reads with iss_rounds
            if codebook_db is None:
                print(
                    f"Could not load codebook for {pos}; good-reads count will be zero."
                )
                good_reads = reads_df.iloc[0:0].copy()
            else:
                good_reads = match_reads(reads_df, codebook_db, iss_rounds=well_iss_rounds, well_name=pos, failed_rounds_by_well=failed_rounds_by_well, debug=False)
        for tile in tile_list:
            x_min, x_max, y_min, y_max = tile

            tile_labels = iss_seg_data[x_min:x_max, y_min:y_max]
            num_cells_tile = int(len(da.unique(tile_labels).compute()) - 1)

            if num_cells_tile <= 0:
                well_percent.append(0.0)
                well_norm.append(0.0)
                well_good_counts.append(0)
                continue

            reads_in_tile = reads_df[
                (reads_df["i_global"] >= x_min)
                & (reads_df["i_global"] < x_max)
                & (reads_df["j_global"] >= y_min)
                & (reads_df["j_global"] < y_max)
            ]

            cells_with_reads = (
                reads_in_tile["cell"][reads_in_tile["cell"] > 0].nunique()
                if "cell" in reads_in_tile.columns
                else 0
            )

            frac = float(cells_with_reads) / float(num_cells_tile)
            well_percent.append(max(0.0, min(100.0, 100.0 * frac)))
            well_norm.append(max(0.0, min(1.0, frac)))

            # Good reads count per tile (not normalized)
            good_reads_in_tile = good_reads[
                (good_reads["i_global"] >= x_min)
                & (good_reads["i_global"] < x_max)
                & (good_reads["j_global"] >= y_min)
                & (good_reads["j_global"] < y_max)
            ]
            well_good_counts.append(int(len(good_reads_in_tile)))

        percent_map[pos] = well_percent
        normalized_map[pos] = well_norm
        good_reads_count_map[pos] = well_good_counts

    if not percent_map:
        print("No data to plot for cells-with-reads heatmaps.")
        return None, None

    positions = list(percent_map.keys())
    num_wells = len(positions)
    indx_i = [a[0] for a in indx]
    indx_j = [a[1] for a in indx]
    n = int(np.sqrt(len(indx_i)))

    # Plot Percent heatmap
    fig1, ax1 = plt.subplots(1, num_wells, figsize=(15, 5))
    if num_wells == 1:
        ax1 = [ax1]
    for i in range(num_wells):
        out = np.zeros((n, n), dtype=float)
        out[indx_i, indx_j] = percent_map[positions[i]]
        im1 = ax1[i].imshow(out, vmin=0, vmax=100)
        ax1[i].set_xticks([])
        ax1[i].set_yticks([])
        ax1[i].set_title(positions[i])
    cbar1 = fig1.colorbar(im1, ax=ax1, fraction=0.012)
    cbar1.set_label("Percent cells with reads")
    fig1.suptitle(f"Percent Cells with Reads per Tile\n{experiment}", fontsize=18)
    try:
        plt.savefig(dataset.metrics_paths["percent_cells_with_reads_heatmap"], dpi=300)
    except Exception:
        out_path = (
            dataset.results_iss / f"percent_cells_with_reads_heatmap_{method}.png"
        )
        plt.savefig(out_path, dpi=300)

    # # Plot Normalized (cells-with-reads per cell) heatmap
    # fig2, ax2 = plt.subplots(1, num_wells, figsize=(15, 5))
    # if num_wells == 1:
    #     ax2 = [ax2]
    # for i in range(num_wells):
    #     out = np.zeros((n, n), dtype=float)
    #     out[indx_i, indx_j] = normalized_map[positions[i]]
    #     im2 = ax2[i].imshow(out, vmin=0, vmax=1)
    #     ax2[i].set_xticks([])
    #     ax2[i].set_yticks([])
    #     ax2[i].set_title(positions[i])
    # cbar2 = fig2.colorbar(im2, ax=ax2, fraction=0.012)
    # cbar2.set_label("Cells-with-reads per cell")
    # fig2.suptitle(f"Cells-with-Reads per Cell per Tile\n{experiment}", fontsize=18)
    # try:
    #     plt.savefig(dataset.metrics_paths.get("cells_with_reads_normalized_heatmap", dataset.results_iss / f"cells_with_reads_normalized_heatmap_{method}.png"), dpi=300)
    # except Exception:
    #     out_path = dataset.results_iss / f"cells_with_reads_normalized_heatmap_{method}.png"
    #     plt.savefig(out_path, dpi=300)

    # # Plot Good reads count heatmap (matched or high-confidence)
    # fig3, ax3 = plt.subplots(1, num_wells, figsize=(15, 5))
    # if num_wells == 1:
    #     ax3 = [ax3]
    # vmax_counts = 0
    # for pos in positions:
    #     vmax_counts = max(vmax_counts, max(good_reads_count_map[pos]) if good_reads_count_map[pos] else 0)
    # if vmax_counts <= 0:
    #     vmax_counts = 1
    # for i in range(num_wells):
    #     out = np.zeros((n, n), dtype=float)
    #     out[indx_i, indx_j] = good_reads_count_map[positions[i]]
    #     im3 = ax3[i].imshow(out, vmin=0, vmax=vmax_counts)
    #     ax3[i].set_xticks([])
    #     ax3[i].set_yticks([])
    #     ax3[i].set_title(positions[i])
    # cbar3 = fig3.colorbar(im3, ax=ax3, fraction=0.012)
    # cbar3.set_label("Good reads per tile")
    # fig3.suptitle(f"Good Reads per Tile\n{experiment}", fontsize=18)
    # try:
    #     plt.savefig(dataset.metrics_paths.get("good_reads_per_tile_heatmap", dataset.results_iss / f"good_reads_per_tile_heatmap_{method}.png"), dpi=300)
    # except Exception:
    #     out_path = dataset.results_iss / f"good_reads_per_tile_heatmap_{method}.png"
    #     plt.savefig(out_path, dpi=300)

    return {"percent": percent_map}, indx

