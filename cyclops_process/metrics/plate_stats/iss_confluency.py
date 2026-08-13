"""
ISS Confluency Metrics

This module provides functions to compute and visualize cell confluency
(cell density) across wells in an ISS experiment.
"""

from typing import Dict, Tuple, List
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
from iohub.ngff import open_ome_zarr
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from ops_utils.io.tiling import split_into_tiles
from ops_utils.data.experiment import OpsDataset
from ops_utils.hpc.resource_manager import get_optimal_workers






def _count_cells_in_tile(
    tile: Tuple[int, int, int, int],
    iss_seg_data: da.Array,
) -> int:
    """
    Count unique cells in a single tile.

    Args:
        tile: Tuple of (x_min, x_max, y_min, y_max) defining tile boundaries.
        iss_seg_data: Dask array of segmentation data.

    Returns:
        Number of unique cells in the tile (excluding background label 0).
    """
    x_min, x_max, y_min, y_max = tile
    tile_data = iss_seg_data[x_min:x_max, y_min:y_max]
    return len(da.unique(tile_data).compute()) - 1


def _process_tiles_parallel(
    tile_list: List[Tuple[int, int, int, int]],
    iss_seg_data: da.Array,
    num_workers: int,
) -> List[int]:
    """
    Process tiles in parallel to count cells.

    Args:
        tile_list: List of tile boundaries.
        iss_seg_data: Dask array of segmentation data.
        num_workers: Number of parallel workers.

    Returns:
        List of cell counts per tile (in same order as tile_list).
    """
    well_num_cells = [0] * len(tile_list)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks with their index
        future_to_idx = {
            executor.submit(_count_cells_in_tile, tile, iss_seg_data): idx
            for idx, tile in enumerate(tile_list)
        }

        # Collect results with progress bar
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(tile_list),
            desc="Counting cells per tile",
        ):
            idx = future_to_idx[future]
            try:
                well_num_cells[idx] = future.result()
            except Exception as e:
                print(f"Error processing tile {idx}: {e}")
                well_num_cells[idx] = 0

    return well_num_cells


def confluency(
    experiment: str,
    num_workers: int = None,
    force: bool = False,
) -> Tuple[Dict[str, List[int]], List[Tuple[int, int]]]:
    """
    Heatmap of cell density across the plate.

    Args:
        experiment: Experiment name/identifier.
        num_workers: Number of parallel workers. If None, automatically
            determined using get_optimal_workers().
        force: If True, regenerate even if output file exists. Default False.

    Returns:
        Tuple of:
            - plate_num_cells: Dict mapping position to list of cell counts per tile
            - indx: List of (i, j) tile indices for reconstruction
    """
    # TODO: convert cells/tile to cells / um^2

    dataset = OpsDataset(experiment, method=None)

    # Check if output already exists
    output_path = dataset.metrics_paths["confluency"]
    if output_path.exists() and not force:
        print(f"Confluency heatmap already exists at: {output_path}")
        print("Skipping. Use --force or force=True to regenerate.")
        return {}, []

    # Determine optimal workers if not specified
    if num_workers is None:
        num_workers = get_optimal_workers(
            use_gpu=False,
            model_ram_gb=0.5,
            data_ram_gb=1.0,
            verbose=False,
        )
    print(f"Using {num_workers} workers for confluency computation")

    iss_seg_store = open_ome_zarr(dataset.store_paths["iss_segmentation"])
    pos_list = [a[0] for a in iss_seg_store.positions()]

    plate_num_cells = {}
    indx = None

    for pos in pos_list:
        print(f"Processing well {pos}...")
        iss_seg_data = da.array(iss_seg_store[pos].data[0, 0, 0, :, :])
        shape = iss_seg_data.shape[-2:]
        tile_list, indx = split_into_tiles(shape, 30, 0)

        # Process tiles in parallel
        well_num_cells = _process_tiles_parallel(
            tile_list, iss_seg_data, num_workers
        )
        plate_num_cells[pos] = well_num_cells

    # --- Plotting ---
    positions = list(plate_num_cells.keys())
    num_wells = len(positions)
    indx_i = [a[0] for a in indx]
    indx_j = [a[1] for a in indx]

    fig, ax = plt.subplots(1, num_wells, figsize=(15, 5))
    if num_wells == 1:
        ax = [ax]

    for i in range(num_wells):
        out = np.zeros((30, 30))
        out[indx_i, indx_j] = plate_num_cells[positions[i]]
        im = ax[i].imshow(out)
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    cbar = fig.colorbar(im, ax=ax, fraction=0.045)
    cbar.set_label("Cells / tile")
    fig.suptitle(f"Cell Confluency\n{experiment}", fontsize=18)
    plt.savefig(dataset.metrics_paths["confluency"], dpi=300)

    return plate_num_cells, indx
