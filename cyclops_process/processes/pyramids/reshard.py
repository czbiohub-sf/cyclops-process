"""Resharding helpers for pyramid/overlay arrays (leaf module)."""
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict, List
import dask.array as da
import numpy as np
import json
import pandas as pd
import logging
from tqdm import tqdm
from iohub import open_ome_zarr

from cyclops_process.napari.dask.dask_utils import (
    _load_total_translations,
    _synthesize_edges_from_rects,
    _candidate_pos_prefixes,
)
from cyclops_process.napari.dask.channel_clims import (
    match_profile,
    compute_position_clims,
)
from ops_utils.io.zarr_utils import (
    _iter_position_paths,
    write_component_attrs,
    write_zarr_slice_direct,
    list_numeric_levels,
    get_level0_shape,
    get_channel_dim,
    ensure_pyramid_levels,
    enumerate_units,
    add_missing_zarr_metadata,
    detect_zarr_format,
    has_zarr_array_metadata,
    create_zarr_array,
)
import zarr
from ops_utils.data.experiment import OpsDataset
from ops_utils.data.filesystem import (
    decide_overwrite_resume_skip,
    prompt_overwrite_resume_skip,
)

from joblib import Parallel, delayed
from ops_utils.hpc.resource_manager import get_optimal_workers



def _reshard_single_level(task: dict) -> str | None:
    """
    Reshard a single overlay level array. Worker function for parallel resharding.

    Returns error message on failure, None on success.
    """
    from ops_utils.io.zarr_utils import reshard_zarr_array

    level_dir = task["level_dir"]
    tile_size = task["tile_size"]
    shards_ratio = task["shards_ratio"]

    try:
        reshard_zarr_array(
            source_path=level_dir,
            dest_path=None,  # In-place resharding
            chunks=None,  # Keep existing chunks
            shards_ratio=shards_ratio,
            tile_size=tile_size,
            show_progress=False,  # Quiet mode
        )
        return None
    except Exception as e:
        return f"Warning: Could not reshard {level_dir}: {e}"


def _reshard_overlay_arrays(
    source_store: Path,
    positions: Sequence[str],
    overlay_names: Sequence[str],
    zarr_format: int,
    label: str = "overlay",
    tile_size: int = 4096,
) -> None:
    """
    Reshard overlay arrays after parallel tile writes are complete.

    This converts unsharded arrays (written safely in parallel) to
    sharded arrays for efficient storage and access. Uses tile-by-tile
    copying to avoid loading entire arrays into memory.

    Resharding is parallelized across all (position, overlay, level) combinations.

    Parameters
    ----------
    source_store : Path
        Path to zarr store
    positions : Sequence[str]
        List of position paths to reshard
    overlay_names : Sequence[str]
        Names of overlays to reshard (e.g., ["iss_gene_image", "iss_guide_image", "grid_overlay"])
    zarr_format : int
        Zarr format version (only reshards v3)
    label : str
        Label for progress messages (e.g., "ISS", "grid")
    tile_size : int
        Tile size for chunked copying (default: 4096)
    """
    from joblib import Parallel, delayed
    from ops_utils.hpc.resource_manager import get_optimal_workers

    if zarr_format != 3:
        return  # Only v3 supports sharding

    # Collect all reshard tasks
    tasks = []
    for pos in positions:
        pos_dir = source_store / pos

        for overlay_name in overlay_names:
            overlay_dir = pos_dir / "labels" / overlay_name

            if not overlay_dir.exists():
                continue

            # Find all numeric level directories
            level_dirs = [d for d in overlay_dir.iterdir() if d.is_dir() and d.name.isdigit()]

            for level_dir in level_dirs:
                tasks.append({
                    "level_dir": level_dir,
                    "tile_size": tile_size,
                    "shards_ratio": (32, 32, 1),  # 3D: (Y, X, C)
                })

    if not tasks:
        return

    # Get optimal worker count for CPU-bound I/O task
    n_workers = get_optimal_workers(use_gpu=False, model_ram_gb=0.5, data_ram_gb=0.5)
    print(f"\n[{label}] Resharding {len(tasks)} arrays in parallel ({n_workers} workers)...")

    # Process all tasks in parallel
    results = Parallel(n_jobs=n_workers, verbose=0)(
        delayed(_reshard_single_level)(task) for task in tasks
    )

    # Report any errors
    errors = [r for r in results if r is not None]
    for err in errors:
        print(f"  {err}")

    print(f"[{label}] Resharding complete ({len(tasks) - len(errors)}/{len(tasks)} succeeded)")


