"""Pyramid engine: image/seg/organelle/base pyramid building + clims.

Overlay rendering (ISS/grid) lives in .overlays; resharding in .reshard.
The two overlay entrypoints are re-exported here so existing
`from ...pyramids.build_dask import build_{iss,grid}_overlay_in_place` keeps working.
"""
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

from cyclops_process.processes.pyramids.overlays import (
    build_iss_overlay_in_place,
    build_grid_overlay_in_place,
)
from cyclops_process.processes.pyramids.clims import build_clims_in_place



def _init_image_levels(
    source_store: Path, pos_paths: Sequence[str], levels: int
) -> None:
    print("Initializing image pyramid levels")
    for pos in pos_paths:
        ensure_pyramid_levels(source_store, str(pos), levels)


def _get_seg_component_path(pos: str, seg_name: str, level: int | str, in_labels_group: bool = False) -> str:
    """
    Build the zarr component path for a segmentation array.

    Args:
        pos: Position path like "A/1/0"
        seg_name: Segmentation name (e.g., "seg", "nuclear_seg", "mitoc_tomm20_seg")
        level: Pyramid level (0, 1, 2, etc.)
        in_labels_group: If True, path is pos/labels/seg_name/level; else pos/seg_name/level

    Returns:
        Component path string like "A/1/0/seg/0" or "A/1/0/labels/mitoc_tomm20_seg/0"
    """
    if in_labels_group:
        return str(Path(str(pos)) / "labels" / seg_name / str(level))
    else:
        return str(Path(str(pos)) / seg_name / str(level))


def _get_seg_dir_path(source_store: Path, pos: str, seg_name: str, level: int | str, in_labels_group: bool = False) -> Path:
    """Get the filesystem path to a segmentation level directory."""
    if in_labels_group:
        return source_store / str(pos) / "labels" / seg_name / str(level)
    else:
        return source_store / str(pos) / seg_name / str(level)


def _init_seg_levels(
    source_store: Path,
    pos_paths: Sequence[str],
    levels: int,
    seg_name: str = "seg",
    in_labels_group: bool = False,
    preserve_dtype: bool = False,
    shards_ratio: tuple = (1, 1, 1, 32, 32),
    chunks: tuple = None,
) -> None:
    """
    Initialize pyramid levels for segmentation arrays.

    Supports both zarr v2 and v3 stores, and both top-level seg (pos/seg/0) and
    labels group seg (pos/labels/seg_name/0) structures.

    Args:
        source_store: Path to zarr store
        pos_paths: List of position paths to process
        levels: Number of pyramid levels to create
        seg_name: Name of the segmentation (e.g., "seg", "nuclear_seg", "mitoc_tomm20_seg")
        in_labels_group: If True, look for arrays under pos/labels/seg_name/0
        preserve_dtype: If True, preserve original dtype; else use int32
        shards_ratio: Sharding ratio for v3 zarr format (default: (1, 1, 1, 32, 32) for single-channel labels)
        chunks: Base chunk size tuple. If None, reads from level 0; if level 0 has no chunks, uses (1, 1, 1, 512, 512)
    """
    path_desc = f"labels/{seg_name}" if in_labels_group else seg_name
    print(f"Initializing {path_desc} pyramid levels")

    # Detect zarr format once for the store
    zarr_format = detect_zarr_format(source_store)
    if in_labels_group:
        print(f"  Detected zarr format: v{zarr_format}")

    for pos in pos_paths:
        seg0_path = _get_seg_dir_path(source_store, pos, seg_name, 0, in_labels_group)
        if not seg0_path.exists():
            continue

        # Check for valid metadata (v2: .zarray, v3: zarr.json)
        if not has_zarr_array_metadata(seg0_path):
            if not in_labels_group:
                print(f"WARNING: {pos}/{seg_name}/0 missing metadata - attempting to reconstruct")
                success = add_missing_zarr_metadata(source_store, str(pos), seg_name, level=0)
                if not success:
                    print(f"WARNING: Skipping {pos}/{seg_name}/0 - could not reconstruct metadata")
                    continue
            else:
                print(f"WARNING: {pos}/labels/{seg_name}/0 missing metadata - skipping")
                continue

        # For top-level seg, use image pyramid levels; for labels, create levels 1 to levels-1
        if in_labels_group:
            lvl_range = range(1, int(levels))
        else:
            lvl_names = list_numeric_levels(source_store, str(pos))
            if not lvl_names:
                continue
            lvl_range = [l for l in sorted(map(int, lvl_names)) if l > 0 and l < int(levels)]
            # If image store has only level 0 (no image pyramids), still build seg pyramids
            if not lvl_range:
                lvl_range = range(1, int(levels))

        # Read base seg shape, dtype, and chunks
        component_path = _get_seg_component_path(pos, seg_name, 0, in_labels_group)
        try:
            base_da = da.from_zarr(str(source_store), component=component_path)
            seg_shape = tuple(int(s) for s in base_da.shape)
            base_dtype = base_da.dtype if preserve_dtype else np.int32
            # Get base chunks from level 0 if not provided
            if chunks is None:
                base_chunks = base_da.chunksize
            else:
                base_chunks = chunks
        except Exception as e:
            print(f"WARNING: Could not read {component_path}: {e}")
            continue

        for lvl in lvl_range:
            # Compute target shape
            # ceil division matches numpy stride arr[::factor] output size
            factor = 2**lvl
            yl = max(1, -(-seg_shape[-2] // factor))
            xl = max(1, -(-seg_shape[-1] // factor))

            # Skip if level exists with correct spatial shape
            lvl_dir = _get_seg_dir_path(source_store, pos, seg_name, lvl, in_labels_group)
            if lvl_dir.exists():
                try:
                    lvl_component = _get_seg_component_path(pos, seg_name, lvl, in_labels_group)
                    existing = da.from_zarr(str(source_store), component=lvl_component)
                    expected_shape = seg_shape[:-2] + (yl, xl)
                    if existing.shape == expected_shape:
                        continue
                    print(f"  Recreating {seg_name} level {lvl}: shape {existing.shape[-2:]} != expected ({yl}, {xl})")
                except Exception:
                    pass

            # Build seg shape matching base dimensions but with target spatial dims
            # Use base_chunks for spatial dimensions, clamped to array size
            if len(seg_shape) >= 5:
                shape = (seg_shape[0], seg_shape[1], seg_shape[2], yl, xl)
                chunk_shape = (1, 1, 1, min(base_chunks[-2], yl), min(base_chunks[-1], xl))
            elif len(seg_shape) == 4:
                shape = (seg_shape[0], seg_shape[1], yl, xl)
                chunk_shape = (1, 1, min(base_chunks[-2], yl), min(base_chunks[-1], xl))
            else:
                shape = (yl, xl)
                chunk_shape = (min(base_chunks[-2], yl), min(base_chunks[-1], xl))

            # Use helper function for array creation with sharding
            lvl_component = _get_seg_component_path(pos, seg_name, lvl, in_labels_group)
            lvl_path = source_store / lvl_component

            create_zarr_array(
                path=str(lvl_path),
                shape=shape,
                chunks=chunk_shape,
                dtype=base_dtype,
                zarr_format=zarr_format,
                shards_ratio=shards_ratio,
                fill_value=0,
                overwrite=True,
            )


def _is_zero_like_component(
    source_store: Path, component_path: str,
    t: int | None = None, c: int | None = None,
) -> bool:
    try:
        arr = da.from_zarr(str(source_store), component=component_path)
        # Probe a tiny window in the first chunk to avoid full-plane reads
        try:
            y_chunk = int(getattr(arr, "chunks", ((), (), (), (256,), (256,)))[-2][0])
            x_chunk = int(getattr(arr, "chunks", ((), (), (), (256,), (256,)))[-1][0])
        except Exception:
            y_chunk, x_chunk = 256, 256
        cy = max(1, min(64, int(arr.shape[-2]), y_chunk))
        cx = max(1, min(64, int(arr.shape[-1]), x_chunk))

        # Sample multiple locations to check if ANY data exists
        # IMPORTANT: Sample from MIDDLE of array, not corner, as stitched images often have empty padding at edges
        mid_y = arr.shape[-2] // 2
        mid_x = arr.shape[-1] // 2

        samples_to_check = []

        if arr.ndim >= 5:
            # 5D: (t, c, z, y, x) — when t/c known, check only that slice to avoid
            # a populated channel masking an empty one (e.g. Phase2D hiding Focus3D)
            t_idx = t if t is not None else 0
            c_idx = c if c is not None else 0
            t_idx = min(t_idx, arr.shape[0] - 1)
            c_idx = min(c_idx, arr.shape[1] - 1)
            samples_to_check = [
                arr[t_idx, c_idx, 0, mid_y:mid_y+cy, mid_x:mid_x+cx],
            ]
        elif arr.ndim == 4:
            # 4D: (t/c, z/c, y, x) or similar
            dim0_mid = min(1, arr.shape[0] - 1) if arr.shape[0] > 1 else 0
            dim1_mid = min(1, arr.shape[1] - 1) if arr.shape[1] > 1 else 0
            samples_to_check = [
                arr[0, 0, mid_y:mid_y+cy, mid_x:mid_x+cx],
                arr[dim0_mid, dim1_mid, mid_y:mid_y+cy, mid_x:mid_x+cx],
            ]
        elif arr.ndim == 3:
            samples_to_check = [arr[0, mid_y:mid_y+cy, mid_x:mid_x+cx]]
        else:
            samples_to_check = [arr[mid_y:mid_y+cy, mid_x:mid_x+cx]]

        # If ANY sample has non-zero data, the array is not zero-like
        for sample in samples_to_check:
            computed = sample.compute()
            if np.count_nonzero(computed) > 0:
                return False  # Has data, not zero-like

        # All samples were zero
        return True
    except Exception as e:
        # If we cannot read the array, treat as zero-like (needs rebuilding)
        # This handles missing/corrupt arrays that should be regenerated
        print(f"Warning: Cannot read {component_path} for zero check ({e}), will rebuild")
        return True


def _process_pos_t_unit(
    source_store: Path, pos_path: str, t: int,
    c_jobs: list[tuple[int, list[int]]], factor: int,
) -> None:
    """Build all channels of one (pos, t) — write each level once with all c.

    Shards pack all C (e.g. (1,6,1,Y,X)); writing per-c rewrites each shard
    n_channels times. Batching collapses to one write per shard per level.
    """
    import time
    base_da = da.from_zarr(str(source_store), component=f"{pos_path}/0")
    n_c = int(base_da.shape[1])
    dtype = base_da.dtype
    # Union of all levels any c needs.
    all_targets = sorted(set(l for _, t_lvls in c_jobs for l in t_lvls))
    if not all_targets:
        return
    # Pre-allocate level arrays sized for all channels.
    level_arrays: dict[int, np.ndarray] = {}
    for lvl in all_targets:
        f = factor ** lvl
        h_l = -(-base_da.shape[-2] // f)
        w_l = -(-base_da.shape[-1] // f)
        level_arrays[lvl] = np.zeros((1, n_c, 1, h_l, w_l), dtype=dtype)

    # Load+downsample channels in parallel via cv2 INTER_AREA (block-mean
    # equivalent for integer factors, SIMD/multithreaded → ~5-10× faster
    # than skimage.downscale_local_mean, same dtype in/out).
    from concurrent.futures import ThreadPoolExecutor
    import cv2
    t_load = time.time()
    def _load_channel(args):
        i, (c, c_targets) = args
        t_c = time.time()
        base_np = np.asarray(base_da[t, c].compute())
        if base_np.ndim == 3 and base_np.shape[0] == 1:
            base_np = base_np[0]
        t_loaded = time.time()
        # cv2 doesn't support float16; upcast to float32 for the resize.
        cv_input = base_np.astype(np.float32, copy=False) if base_np.dtype == np.float16 else base_np
        for lvl in c_targets:
            f = factor ** lvl
            h_l, w_l = base_np.shape[0] // f, base_np.shape[1] // f
            ds = cv2.resize(cv_input, (w_l, h_l), interpolation=cv2.INTER_AREA)
            if np.issubdtype(dtype, np.integer):
                ds = np.rint(ds).astype(dtype, copy=False)
            else:
                ds = ds.astype(dtype, copy=False)
            level_arrays[lvl][0, c, 0, :ds.shape[0], :ds.shape[1]] = ds
        del base_np, cv_input
        print(f"  [{i}/{len(c_jobs)}] c={c}: load {t_loaded - t_c:.1f}s + downscale {time.time() - t_loaded:.1f}s", flush=True)
    n_parallel = min(6, len(c_jobs))  # all 6 channels concurrently (peak ≈ 350GB)
    with ThreadPoolExecutor(max_workers=n_parallel) as pool:
        list(pool.map(_load_channel, enumerate(c_jobs, 1)))
    print(f"  Total load + downscale: {time.time() - t_load:.1f}s ({n_parallel} parallel)", flush=True)

    # Write each level via parallel shard-row strips (cell_seg pattern):
    # each thread writes one shard-row → parallel lz4 compression + NFS I/O.
    import zarr, json
    from concurrent.futures import ThreadPoolExecutor
    t_write = time.time()
    for lvl in all_targets:
        t0 = time.time()
        arr_path = f"{source_store}/{pos_path}/{lvl}"
        # Read outer chunk_shape (shard) Y dim from zarr.json.
        zj = json.load(open(f"{arr_path}/zarr.json"))
        shard_y = int(zj["chunk_grid"]["configuration"]["chunk_shape"][-2])
        H = level_arrays[lvl].shape[-2]
        n_strips = max(1, (H + shard_y - 1) // shard_y)
        data = level_arrays[lvl]
        def _write_strip(s):
            a = zarr.open(str(source_store), mode="r+")[f"{pos_path}/{lvl}"]
            y0 = s * shard_y
            y1 = min(H, (s + 1) * shard_y)
            a[t:t+1, :, :, y0:y1, :] = data[:, :, :, y0:y1, :]
        with ThreadPoolExecutor(max_workers=n_strips) as pool:
            list(pool.map(_write_strip, range(n_strips)))
        mb = data.nbytes / 1024 / 1024
        print(f"    Level {lvl}: {time.time() - t0:.1f}s ({mb:.1f} MB, {n_strips} strips)", flush=True)
    print(f"  Write took {time.time() - t_write:.1f}s", flush=True)

    # Warn-only verify, drops Z=1.
    t_verify = time.time()
    for lvl in all_targets:
        verify_arr = da.from_zarr(str(source_store), component=f"{pos_path}/{lvl}")
        sample_c = c_jobs[0][0]
        verify_slice = verify_arr[t, sample_c, 0]
        h, w = verify_slice.shape[-2], verify_slice.shape[-1]
        ps = min(256, h, w)
        had_data = False
        for fy in (0.15, 0.5, 0.85):
            for fx in (0.15, 0.5, 0.85):
                y0 = max(0, int(h * fy) - ps // 2)
                x0 = max(0, int(w * fx) - ps // 2)
                patch = verify_slice[y0:y0 + ps, x0:x0 + ps].compute()
                if patch.size and ((patch != 0).any() or np.isnan(patch).any()):
                    had_data = True
                    break
            if had_data:
                break
        print(f"    {'✓' if had_data else '⚠'} Level {lvl}{'' if had_data else ' (sparse?)'}")
    print(f"  Verify took {time.time() - t_verify:.1f}s")


def _process_pyramid_unit(
    source_store: Path, pos_path: str, t: int, c: int, targets: list[int], factor: int
) -> None:
    """Process one (position, t, c) unit through all pyramid levels.

    Loads the entire level-0 (t, c) slice into memory in a single bulk read,
    then downsamples sequentially through pyramid levels.  This avoids tens of
    thousands of individual chunk I/O operations which are extremely slow on
    sharded zarr v3 stores on parallel filesystems.
    """
    import time
    from skimage.transform import downscale_local_mean as cpu_downscale

    # Load level 0 as dask array
    base_da = da.from_zarr(str(source_store), component=f"{pos_path}/0")
    orig_dtype = base_da.dtype
    base_tc = base_da[t, c]

    # --- Bulk load the entire (t, c) slice into memory ----------------------
    t_load = time.time()
    shape_str = " x ".join(str(s) for s in base_tc.shape)
    nbytes_gb = base_tc.nbytes / (1024**3)
    print(f"  Loading level 0 ({shape_str}, {orig_dtype}, {nbytes_gb:.2f} GB) ...")
    base_np = np.asarray(base_tc.compute())
    # Squeeze out Z=1 dimension if present (3D → 2D)
    if base_np.ndim == 3 and base_np.shape[0] == 1:
        base_np = base_np[0]
    print(f"  Load took {time.time() - t_load:.1f}s")

    # --- Zero check (fast on in-memory array) --------------------------------
    # Sample a 5x5 grid across the image to detect sparse data (e.g. fluorescence
    # channels where signal is concentrated in specific regions).
    # Only skip if ALL 25 patches are zero — avoids false positives on sparse channels.
    ps = min(256, base_np.shape[-2], base_np.shape[-1])
    h, w = base_np.shape[-2], base_np.shape[-1]
    _all_zero = True
    for fy in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for fx in [0.1, 0.3, 0.5, 0.7, 0.9]:
            sy = max(0, int(h * fy) - ps // 2)
            sx = max(0, int(w * fx) - ps // 2)
            if np.count_nonzero(base_np[sy:sy+ps, sx:sx+ps]) > 0:
                _all_zero = False
                break
        if not _all_zero:
            break

    if _all_zero:
        print(
            f"  ⚠️  Source data is all zeros (25 patches sampled) - skipping pyramid build for this unit"
        )
        for lvl in targets:
            arr_lvl_da = da.from_zarr(
                str(source_store), component=f"{pos_path}/{lvl}"
            )
            if arr_lvl_da.ndim >= 2:
                shape = arr_lvl_da[t, c].shape
            else:
                shape = arr_lvl_da.shape
            zeros = np.zeros(shape, dtype=orig_dtype)
            write_zarr_slice_direct(
                store_path=source_store,
                component_path=f"{pos_path}/{lvl}",
                data=zeros,
                t=t,
                c=c,
            )
        return

    # Block-mean downsample via cv2 INTER_AREA — SIMD/multithreaded, ~5-10×
    # faster than skimage.downscale_local_mean with identical output for
    # integer factors. Stride is reserved for label and overlay arrays where
    # averaging would corrupt IDs / RGBA bytes.
    # cv2 doesn't support float16 (track store dtype), so upcast to float32
    # for the resize and cast back.
    import cv2
    t_ds = time.time()
    output_arrays: dict[int, np.ndarray] = {}
    cv_input = base_np.astype(np.float32, copy=False) if base_np.dtype == np.float16 else base_np
    for lvl in targets:
        f = factor ** lvl
        h_l, w_l = base_np.shape[0] // f, base_np.shape[1] // f
        ds = cv2.resize(cv_input, (w_l, h_l), interpolation=cv2.INTER_AREA)
        output_arrays[lvl] = (
            np.rint(ds).astype(orig_dtype, copy=False)
            if np.issubdtype(orig_dtype, np.integer)
            else ds.astype(orig_dtype, copy=False)
        )
    # Free level-0 memory early
    del base_np, cv_input
    print(f"  Downsample ({len(targets)} levels) took {time.time() - t_ds:.1f}s")

    # Write all levels for this unit (use fast zarr utilities)
    t_write = time.time()
    print(f"  Writing {len(targets)} levels to zarr in parallel...")

    from ops_utils.io.zarr_utils import write_zarr_slices_parallel

    # Prepare write jobs
    writes = [(f"{pos_path}/{lvl}", output_arrays[lvl], t, c) for lvl in targets]

    # Execute parallel writes (3-4x faster than iohub due to direct zarr access)
    write_results = write_zarr_slices_parallel(source_store, writes, max_workers=4)

    # Report results
    for component_path, elapsed, mb_size in write_results:
        lvl = int(component_path.split("/")[-1])
        print(
            f"    Level {lvl}: {elapsed:.1f}s ({mb_size:.1f} MB, {mb_size/max(0.01, elapsed):.1f} MB/s)"
        )

    print(f"  Write took {time.time() - t_write:.1f}s")

    # Verify each level — warn-only. Drop the Z=1 dim so [y0:y0+ps, x0:x0+ps] indexes Y,X.
    t_verify = time.time()
    for lvl in targets:
        verify_arr = da.from_zarr(str(source_store), component=f"{pos_path}/{lvl}")
        verify_slice = verify_arr[t, c, 0]  # 2D (Y, X)
        h, w = verify_slice.shape[-2], verify_slice.shape[-1]
        ps = min(256, h, w)
        had_data = False
        for fy in (0.15, 0.5, 0.85):
            for fx in (0.15, 0.5, 0.85):
                y0 = max(0, int(h * fy) - ps // 2)
                x0 = max(0, int(w * fx) - ps // 2)
                patch = verify_slice[y0:y0 + ps, x0:x0 + ps].compute()
                if patch.size and ((patch != 0).any() or np.isnan(patch).any()):
                    had_data = True
                    break
            if had_data:
                break
        marker = "✓" if had_data else "⚠"
        note = "" if had_data else " (sparse channel?)"
        print(f"    {marker} Level {lvl}{note}")
    print(f"  Verify took {time.time() - t_verify:.1f}s")


def _build_image_pyramid(
    source_store: Path,
    pos_paths: Sequence[str],
    levels: int,
    factor: int,
    resume: bool,
    t_indices: Optional[Sequence[int]] = None,
) -> None:
    """Build image pyramid by downsampling level 0 through multiple levels.

    `t_indices` (optional) restricts which timepoints to build — used when
    dispatching per-(pos, t) SLURM jobs in parallel (shards are (1, C, 1, Y, X)
    so different t's don't share shard files).
    """
    print("Building image pyramid")
    from cyclops_process.napari.dask.dask_utils import determine_target_levels
    import time

    # Plan jobs: determine which (pos, t, c, levels) need building
    jobs: list[tuple[str, int, int, list[int]]] = []
    units = enumerate_units(source_store, list(pos_paths), t_indices=t_indices)

    for pos_path, t, c in units:
        targets = determine_target_levels(source_store, pos_path, levels, resume, t=t, c=c)
        if targets:
            jobs.append((pos_path, t, c, targets))

    if not jobs:
        print("No pyramid levels to build")
        return

    # Batch all c's of each (pos, t) into one write per level so shards
    # (which pack all C) are rewritten once instead of n_channels times.
    by_pos_t: dict[tuple[str, int], list[tuple[int, list[int]]]] = {}
    for pos_path, t, c, targets in jobs:
        by_pos_t.setdefault((pos_path, t), []).append((c, targets))

    print(f"Processing {len(by_pos_t)} (pos, t) units...")
    for i, ((pos_path, t), c_jobs) in enumerate(by_pos_t.items()):
        t_unit_start = time.time()
        print(f"\nUnit {i+1}/{len(by_pos_t)}: {pos_path} t={t} channels={[c for c,_ in c_jobs]}")
        _process_pos_t_unit(source_store, pos_path, t, c_jobs, factor)
        print(f"  Unit completed in {time.time() - t_unit_start:.1f}s")


# -------------------------------------------------------------------------
# Parallel zarr level-0 loader — worker + dispatcher.
#
# Replaces ``np.asarray(base_tc.compute())`` which has been observed to drive
# only ~138% CPU and ~300 MB/s NFS read across a 40+ GB sharded zarr. Default
# dask scheduler collapses the graph into per-shard tasks that effectively
# serialize inside a single-threaded decompress loop.
#
# Design:
#   * Parent allocates an ``mp.shared_memory`` block sized for the full
#     (H, W) output array (int32 = 40 GB for a 100k² int32 label map).
#   * N worker processes (default 8 — env ``OPS_PYRAMID_LOAD_WORKERS``)
#     each attach to the shm by name, open their own raw zarr handle,
#     read a non-overlapping (y, x) tile into the shm view, and exit.
#   * Parent joins workers, then uses the shm-backed numpy array directly
#     for downsampling (no copy).
#
# The 2D-squeeze rule that the caller previously handled is done here so the
# returned array is always shape (H, W).
# -------------------------------------------------------------------------

def _pyramid_load_worker(
    shm_name: str,
    shm_shape: tuple,
    shm_dtype: str,
    source_store: str,
    base_component: str,
    t: int,
    c: int,
    tiles: list,  # list of (y0, x0, y1, x1)
) -> None:
    """Child process: open zarr, attach to shm, read each tile → shm."""
    import numpy as _np
    import zarr as _zarr
    from multiprocessing import shared_memory as _shm

    shm_handle = _shm.SharedMemory(name=shm_name)
    try:
        out = _np.ndarray(shm_shape, dtype=_np.dtype(shm_dtype), buffer=shm_handle.buf)
        arr = _zarr.open(source_store, mode="r")[base_component]
        # zarr shape is (T, C, Z, Y, X); we want the (t, c, 0) Y×X slice.
        for y0, x0, y1, x1 in tiles:
            tile = _np.asarray(arr[t, c, 0, y0:y1, x0:x1])
            if tile.dtype.str != out.dtype.str:
                tile = tile.astype(out.dtype, copy=False)
            out[y0:y1, x0:x1] = tile
    finally:
        # Drop the numpy view first so close() doesn't warn about active refs.
        out = None  # noqa: F841
        shm_handle.close()


def _parallel_load_zarr_2d(
    source_store: str,
    base_component: str,
    base_shape: tuple,  # zarr shape of base_tc (may include Z=1 leading dim)
    t: int,
    c: int,
    dtype,
):
    """Bulk-load the 2D (Y, X) slice at (t, c, 0) using a process pool.

    Returns ``(base_np, shm_block)`` — ``base_np`` is a numpy view onto the
    shm block (shape ``(H, W)``). The caller must release the shm when
    done with the array (typically: ``del base_np; shm_block.close();
    shm_block.unlink()``). If shm setup fails, falls back to the dask
    single-process path and returns ``(base_np, None)``.
    """
    import multiprocessing as _mp
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import shared_memory as _shm

    # Normalize shape to (H, W) — zarr returns 5D arrays; base_tc has sliced
    # out t and c, so leading Z dim is what remains.
    if len(base_shape) == 3 and base_shape[0] == 1:
        H, W = int(base_shape[1]), int(base_shape[2])
    elif len(base_shape) == 2:
        H, W = int(base_shape[0]), int(base_shape[1])
    else:
        import dask.array as _da
        arr_np = np.asarray(
            _da.from_zarr(source_store, component=base_component)[t, c].compute()
        )
        if arr_np.ndim == 3 and arr_np.shape[0] == 1:
            arr_np = arr_np[0]
        return arr_np, None

    np_dtype = np.dtype(dtype)
    nbytes = int(H * W * np_dtype.itemsize)

    n_workers = int(os.environ.get("OPS_PYRAMID_LOAD_WORKERS", "8"))
    tile_pixels = int(os.environ.get("OPS_PYRAMID_LOAD_TILE", "8192"))

    # Build tile list (y0, x0, y1, x1) covering the full (H, W) grid.
    tiles = []
    for y0 in range(0, H, tile_pixels):
        y1 = min(y0 + tile_pixels, H)
        for x0 in range(0, W, tile_pixels):
            x1 = min(x0 + tile_pixels, W)
            tiles.append((y0, x0, y1, x1))

    partitions = [tiles[i::n_workers] for i in range(n_workers)]

    shm_block = _shm.SharedMemory(create=True, size=nbytes)
    try:
        base_np = np.ndarray((H, W), dtype=np_dtype, buffer=shm_block.buf)

        ctx = _mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            futures = [
                pool.submit(
                    _pyramid_load_worker,
                    shm_block.name, (H, W), np_dtype.str,
                    source_store, base_component, int(t), int(c), part,
                )
                for part in partitions
            ]
            for f in futures:
                f.result()

        return base_np, shm_block
    except Exception:
        shm_block.close()
        shm_block.unlink()
        raise


def _process_seg_unit(
    source_store: Path,
    pos_path: str,
    t: int,
    c: int,
    targets: list[int],
    seg_name: str = "seg",
    in_labels_group: bool = False,
    preserve_dtype: bool = False,
    prebuilt_level0: "np.ndarray | None" = None,
) -> None:
    """Process one segmentation unit using appropriate downsampling.

    Loads the entire level-0 (t, c) slice into memory in a single bulk read,
    then downsamples through pyramid levels.  This avoids tens of thousands of
    individual chunk I/O operations which are extremely slow on sharded zarr v3
    stores on parallel filesystems.

    Supports both top-level segmentation (pos/seg/0) and labels group segmentation
    (pos/labels/seg_name/0). Uses stride downsampling for integer labels (preserves IDs)
    and averaging for continuous data (like vesselness maps).

    Args:
        source_store: Path to zarr store
        pos_path: Position path like "A/1/0"
        t: Time index
        c: Channel index
        targets: List of pyramid levels to build
        seg_name: Segmentation name (e.g., "seg", "nuclear_seg", "mitoc_tomm20_seg")
        in_labels_group: If True, look for arrays under pos/labels/seg_name/0
        preserve_dtype: If True, preserve original dtype and use appropriate downsampling;
                       else use int32 with stride downsampling
    """
    import time
    from ops_utils.io.zarr_utils import write_zarr_slice_direct
    from skimage.transform import downscale_local_mean as cpu_downscale

    # Build component path using helper
    base_component = _get_seg_component_path(pos_path, seg_name, 0, in_labels_group)
    path_desc = f"labels/{seg_name}" if in_labels_group else seg_name

    # Load base segmentation data as dask array (lazy)
    base_da = da.from_zarr(str(source_store), component=base_component)
    base_tc = base_da[t, c]
    base_dtype = base_da.dtype if preserve_dtype else np.int32

    # Determine if this is continuous data (float) or discrete labels (int)
    is_continuous = preserve_dtype and np.issubdtype(base_da.dtype, np.floating)
    if is_continuous:
        print(f"  Detected continuous data (dtype={base_da.dtype}), using averaging downsample")

    # --- Bulk load the entire (t, c) slice into memory ----------------------
    t_load = time.time()
    shape_str = " x ".join(str(s) for s in base_tc.shape)
    nbytes_gb = base_tc.nbytes / (1024**3)
    if prebuilt_level0 is not None:
        # Caller already has the assembled (H, W) level-0 in host RAM —
        # skip the ~110s NFS read of what was just written. Shape must
        # match the (H, W) tail of the zarr (T, C, Z, H, W).
        expected_hw = tuple(int(s) for s in base_tc.shape[-2:])
        if prebuilt_level0.shape != expected_hw:
            raise ValueError(
                f"prebuilt_level0 shape {prebuilt_level0.shape} does not match "
                f"expected (H, W) {expected_hw}"
            )
        base_np = prebuilt_level0
        _shm_block = None
        print(f"  [skip-load] using prebuilt {path_desc} level 0 "
              f"({shape_str}, {base_np.dtype}, {nbytes_gb:.2f} GB)")
    else:
        print(f"  Loading {path_desc} level 0 ({shape_str}, {base_da.dtype}, {nbytes_gb:.2f} GB) ...")
        base_np, _shm_block = _parallel_load_zarr_2d(
            str(source_store), base_component, base_tc.shape, t, c, base_da.dtype,
        )
        print(f"  Load took {time.time() - t_load:.1f}s")

    # --- Downsample through pyramid levels — parallel across row bands ------
    # The stride-subsample + astype on a 40 GB int array is memory-bandwidth
    # bound on one core. Split the image into N contiguous row bands and
    # downsample each band in a thread; threads release the GIL during
    # numpy's C-level astype so they overlap well.
    t_ds = time.time()
    output_arrays: dict[int, np.ndarray] = {}
    n_downsample_threads = int(os.environ.get("OPS_PYRAMID_DOWNSAMPLE_THREADS", "8"))

    from concurrent.futures import ThreadPoolExecutor as _TPE

    def _downsample_band(band_np, fy, fx, out_dtype, continuous):
        if continuous:
            ds = cpu_downscale(band_np.astype(np.float64), (fy, fx))
            return ds.astype(out_dtype)
        return band_np[::fy, ::fx].astype(out_dtype)

    H = base_np.shape[0]
    for lvl in targets:
        fy = 2 ** lvl
        fx = 2 ** lvl
        # For labels we stride — bands must be aligned to `fy` so row
        # indexing stays consistent across bands. Ensure band size is a
        # multiple of fy (except possibly the last band).
        band_size = max(fy, (H // n_downsample_threads // fy) * fy)
        if band_size == 0:
            band_size = fy
        bands = []
        y = 0
        while y < H:
            y_end = min(y + band_size, H)
            bands.append((y, y_end))
            y = y_end

        with _TPE(max_workers=n_downsample_threads, thread_name_prefix="pyr_ds") as pool:
            futs = [
                pool.submit(
                    _downsample_band,
                    base_np[y0:y1],
                    fy, fx, base_dtype, is_continuous,
                )
                for (y0, y1) in bands
            ]
            parts = [f.result() for f in futs]
        output_arrays[lvl] = np.concatenate(parts, axis=0)

    # Free level-0 memory early
    del base_np
    if _shm_block is not None:
        try:
            _shm_block.close()
            _shm_block.unlink()
        except Exception as _e:
            print(f"  [warn] parallel-load shm cleanup: {_e}")
    print(f"  Downsample ({len(targets)} levels, {n_downsample_threads}-way) took {time.time() - t_ds:.1f}s")

    # Write all levels concurrently — levels are independent zarr arrays,
    # no cross-level dependency. Uses write_zarr_slices_parallel which
    # the continuous-data pyramid path already uses.
    t_write = time.time()
    print(f"  Writing {len(targets)} {path_desc} levels to zarr (parallel)...")

    from ops_utils.io.zarr_utils import write_zarr_slices_parallel
    writes = [
        (_get_seg_component_path(pos_path, seg_name, lvl, in_labels_group),
         output_arrays[lvl], t, c)
        for lvl in targets
    ]
    n_write_workers = int(os.environ.get("OPS_PYRAMID_WRITE_WORKERS", "4"))
    write_results = write_zarr_slices_parallel(
        source_store, writes, max_workers=n_write_workers,
    )
    for component_path, elapsed, mb_size in write_results:
        lvl = int(component_path.split("/")[-1])
        print(
            f"    {path_desc} level {lvl}: {elapsed:.1f}s ({mb_size:.1f} MB, {mb_size/max(0.01, elapsed):.1f} MB/s)"
        )
    print(f"  {path_desc} write took {time.time() - t_write:.1f}s")


def _build_seg_pyramid(
    source_store: Path,
    pos_paths: Sequence[str],
    levels: int,
    resume: bool,
    seg_name: str = "seg",
    in_labels_group: bool = False,
    preserve_dtype: bool = False,
    prebuilt_level0: "np.ndarray | None" = None,
) -> None:
    """Build segmentation pyramid using appropriate downsampling.

    Supports both top-level segmentation (pos/seg/0) and labels group segmentation
    (pos/labels/seg_name/0).

    Args:
        source_store: Path to zarr store
        pos_paths: List of position paths to process
        levels: Number of pyramid levels
        resume: If True, skip already-built levels; if False, rebuild all
        seg_name: Segmentation name (e.g., "seg", "nuclear_seg", "mitoc_tomm20_seg")
        in_labels_group: If True, look for arrays under pos/labels/seg_name/0
        preserve_dtype: If True, preserve original dtype and use appropriate downsampling
    """
    path_desc = f"labels/{seg_name}" if in_labels_group else seg_name
    print(f"Building {path_desc} pyramid")
    import time

    # Plan jobs: determine which (pos, t, c, levels) need building
    seg_jobs: list[tuple[str, int, int, list[int]]] = []

    for pos in pos_paths:
        pos_path = str(pos)
        base_path = _get_seg_dir_path(source_store, pos_path, seg_name, 0, in_labels_group)
        if not base_path.exists():
            continue

        # Get base segmentation array metadata
        base_component = _get_seg_component_path(pos_path, seg_name, 0, in_labels_group)
        try:
            base_da = da.from_zarr(str(source_store), component=base_component)
        except Exception as e:
            print(f"Warning: Could not read {base_component}: {e}")
            continue

        if base_da.ndim >= 5:
            T, C = int(base_da.shape[0]), int(base_da.shape[1])
        elif base_da.ndim == 4:
            T, C = int(base_da.shape[0]), int(base_da.shape[1])
        else:
            T, C = 1, 1

        # Determine target levels for segmentation
        targets = []
        desired_levels = set(range(1, int(levels)))
        for lvl in sorted(desired_levels):
            lvl_path = _get_seg_dir_path(source_store, pos_path, seg_name, lvl, in_labels_group)
            lvl_component = _get_seg_component_path(pos_path, seg_name, lvl, in_labels_group)
            exists = lvl_path.exists()
            if not resume:
                targets.append(lvl)
            else:
                if (not exists) or _is_zero_like_component(source_store, lvl_component):
                    targets.append(lvl)

        # Create jobs for each (pos, t, c) combination
        if targets:
            for t in range(T):
                for c in range(C):
                    seg_jobs.append((pos_path, t, c, targets))

    if not seg_jobs:
        print(f"No {path_desc} levels to build")
        return

    print(f"Processing {len(seg_jobs)} {path_desc} units...")

    # Process units sequentially (each unit is fast, parallelizing causes OOM)
    # Unlike image pyramids, seg data is loaded fully into memory per unit
    # When prebuilt_level0 is provided, route it to the unique (pos, t=0, c=0)
    # job — organelle invocations always have T=C=1 with a single position
    # per call, so this is unambiguous. Multi-job calls must keep the
    # disk-load path.
    use_prebuilt = (
        prebuilt_level0 is not None
        and len(seg_jobs) == 1
        and seg_jobs[0][1] == 0 and seg_jobs[0][2] == 0
    )
    if prebuilt_level0 is not None and not use_prebuilt:
        print(f"  [pyr-prebuilt] cannot use prebuilt level-0: "
              f"{len(seg_jobs)} jobs (need 1) — falling back to disk load")
    for i, (pos_path, t, c, targets) in enumerate(
        tqdm(seg_jobs, desc=f"{path_desc} units"), 1
    ):
        _process_seg_unit(
            source_store, pos_path, t, c, targets, seg_name,
            in_labels_group=in_labels_group, preserve_dtype=preserve_dtype,
            prebuilt_level0=prebuilt_level0 if use_prebuilt else None,
        )

    print(f"{path_desc} pyramid build complete")


def build_pyramid_in_place(
    source_store: str | Path,
    levels: int = 5,
    factor: int = 2,
    positions: Optional[Sequence[str]] = None,
) -> Path:
    """
    Create multiscale levels inside the existing OME-Zarr using iohub, per-position.
    CPU-only implementation using skimage.downscale_local_mean.
    Downsamples all spatial trailing dimensions (e.g., Z, Y, X) by `factor`.
    Preserves original dtype (rounds for integer dtypes).
    """
    source_store = Path(source_store)

    from skimage.transform import downscale_local_mean as cpu_downscale  # type: ignore

    vprintf(
        "Starting in-place pyramid build: store=%s, levels=%d, factor=%d, backend=%s",
        str(source_store),
        levels,
        factor,
        "cpu,gpu",
    )

    # Prompt for overwrite/resume/skip
    try:
        print(f"Deciding overwrite/resume/skip for {source_store}")
        choice = decide_overwrite_resume_skip(source_store, is_debug=False)
    except Exception:
        choice = "resume"
    if choice == "skip":
        print("Skipping pyramid build (user choice).")
        return source_store
    resume_mode = choice == "resume"

    # Discover positions
    pos_paths = positions or _iter_position_paths(source_store)
    vprintf("Found %d positions to process", len(pos_paths))

    # Detect zarr format to determine correct seg location
    # v2: pos/seg/0, v3: pos/labels/seg/0
    zarr_format = detect_zarr_format(source_store)
    in_labels_group = (zarr_format == 3)
    vprintf("Detected zarr format: v%d (seg location: %s)", zarr_format, 'labels/' if in_labels_group else 'top-level')

    # Initialize levels (image + seg + nuclear_seg)
    _init_image_levels(source_store, pos_paths, levels)

    # Check which segmentation types exist and initialize them
    seg_types = []
    for seg_type in ["seg", "nuclear_seg"]:
        # Check appropriate structure based on zarr version
        if in_labels_group:
            has_seg = any(
                (source_store / str(pos) / "labels" / seg_type / "0").exists() for pos in pos_paths
            )
        else:
            has_seg = any(
                (source_store / str(pos) / seg_type / "0").exists() for pos in pos_paths
            )
        if has_seg:
            seg_types.append(seg_type)
            _init_seg_levels(source_store, pos_paths, levels, seg_type, in_labels_group=in_labels_group)

    if not seg_types:
        print("No segmentation data found (checked 'seg' and 'nuclear_seg')")

    # Build pyramids (image + all available seg types)
    _build_image_pyramid(source_store, pos_paths, levels, factor, resume_mode)

    for seg_type in seg_types:
        _build_seg_pyramid(source_store, pos_paths, levels, resume_mode, seg_type, in_labels_group=in_labels_group)

    return source_store


def build_seg_pyramid_only(
    source_store: str | Path,
    levels: int = 5,
    positions: Optional[Sequence[str]] = None,
    resume: bool = True,
    seg_types: Optional[Sequence[str]] = None,
    shards_ratio: tuple = None,
    chunks: tuple = None,
) -> Path:
    """
    Rebuild ONLY the segmentation pyramid levels inside an existing OME-Zarr store.

    This is useful when segmentation data has been updated but image pyramids are fine.
    Uses stride-based downsampling to preserve label IDs.

    Parameters
    ----------
    source_store : str | Path
        Path to the OME-Zarr store
    levels : int
        Number of pyramid levels (default: 5)
    positions : Optional[Sequence[str]]
        Specific positions to rebuild (default: all positions)
    resume : bool
        If True, skip already-built levels; if False, rebuild all (default: True)
    seg_types : Optional[Sequence[str]]
        Segmentation types to build (default: ["seg", "nuclear_seg"] - will auto-detect which exist)
    shards_ratio : tuple
        Sharding ratio for v3 zarr format (default: None, uses (1, 1, 1, 32, 32) for labels)
    chunks : tuple
        Base chunk size tuple (default: None, reads from level 0)

    Returns
    -------
    Path
        Path to the updated store
    """
    source_store = Path(source_store)

    vprintf(
        "Starting segmentation-only pyramid rebuild: store=%s, levels=%d",
        str(source_store),
        levels,
    )

    # Detect zarr format to determine correct seg location
    # v2: pos/seg/0, v3: pos/labels/seg/0
    zarr_format = detect_zarr_format(source_store)
    in_labels_group = (zarr_format == 3)
    print(f"Detected zarr format: v{zarr_format} (seg location: {'labels/' if in_labels_group else 'top-level'})")

    # Discover positions
    pos_paths = positions or _iter_position_paths(source_store)
    vprintf("Found %d positions to process", len(pos_paths))

    # Determine which segmentation types to process
    if seg_types is None:
        seg_types_to_check = ["seg", "nuclear_seg"]
    else:
        seg_types_to_check = list(seg_types)

    # Check which segmentation types exist based on zarr version
    available_seg_types = []
    for seg_type in seg_types_to_check:
        # Check appropriate structure based on zarr version:
        # v2: pos/seg_type/0 or pos/0/seg_type/0
        # v3: pos/labels/seg_type/0
        if in_labels_group:
            has_seg = any(
                (source_store / str(pos) / "labels" / seg_type / "0").exists()
                for pos in pos_paths
            )
        else:
            has_seg = any(
                (source_store / str(pos) / seg_type / "0").exists()
                or (source_store / str(pos) / "0" / seg_type / "0").exists()
                for pos in pos_paths
            )
        if has_seg:
            available_seg_types.append(seg_type)
            print(f"Found {seg_type} data")
        else:
            print(f"No {seg_type} data found, skipping")

    if not available_seg_types:
        print(f"No segmentation data found (checked: {seg_types_to_check})")
        return source_store

    # Default shards_ratio for labels (single-channel)
    if shards_ratio is None:
        shards_ratio = (1, 1, 1, 32, 32)

    # Initialize and build segmentation levels for each type
    for seg_type in available_seg_types:
        print(f"\nInitializing {seg_type} pyramid levels...")
        _init_seg_levels(source_store, pos_paths, levels, seg_type, in_labels_group=in_labels_group,
                        shards_ratio=shards_ratio, chunks=chunks)

        print(f"Building {seg_type} pyramids...")
        _build_seg_pyramid(source_store, pos_paths, levels, resume, seg_type, in_labels_group=in_labels_group)

    print(f"\n✓ Segmentation pyramid rebuild complete for {source_store}")
    print(f"  Built pyramids for: {', '.join(available_seg_types)}")
    return source_store


def _discover_organelle_labels(source_store: Path, pos_paths: Sequence[str]) -> list[str]:
    """
    Discover organelle segmentation labels in the labels/ group.

    These are stored under pos/labels/{label_name}/0 and are distinct from
    the top-level seg/nuclear_seg structures.

    Returns list of label names found (e.g., ['nucle_vs_seg', 'mitoc_tomm20_seg']).
    """
    label_names = set()

    for pos in pos_paths:
        labels_dir = source_store / str(pos) / "labels"
        if not labels_dir.exists():
            continue

        # List subdirectories in labels/
        try:
            for subdir in labels_dir.iterdir():
                if not subdir.is_dir():
                    continue
                # Skip seg and nuclear_seg - these are handled separately
                if subdir.name in ["seg", "nuclear_seg"]:
                    continue
                # Check if this label has a "0" array (level 0 data)
                level0_path = subdir / "0"
                if level0_path.exists():
                    label_names.add(subdir.name)
        except Exception as e:
            print(f"Warning: Could not scan labels dir for {pos}: {e}")
            continue

    return sorted(label_names)


def build_organelle_seg_pyramids(
    source_store: str | Path,
    levels: int = 5,
    positions: Optional[Sequence[str]] = None,
    resume: bool = True,
    label_names: Optional[Sequence[str]] = None,
    prebuilt_level0: "np.ndarray | None" = None,
) -> Path:
    """
    Build pyramid levels for organelle segmentation labels.

    Organelle labels are stored under pos/labels/{label_name}/0 and include
    segmentations like nucleoli, mitochondria, ER, etc.

    Parameters
    ----------
    source_store : str | Path
        Path to the OME-Zarr store (e.g., phenotyping_v3.zarr)
    levels : int
        Number of pyramid levels to create (default: 5)
    positions : Optional[Sequence[str]]
        Specific positions to process (default: all positions)
    resume : bool
        If True, skip already-built levels; if False, rebuild all (default: True)
    label_names : Optional[Sequence[str]]
        Specific label names to build (default: auto-discover all labels in labels/ group)

    Returns
    -------
    Path
        Path to the updated store

    Example
    -------
    >>> from cyclops_process.processes.pyramids.build_dask import build_organelle_seg_pyramids
    >>> build_organelle_seg_pyramids(
    ...     "/path/to/phenotyping_v3.zarr",
    ...     levels=5,
    ...     label_names=["nucle_vs_seg", "mitoc_tomm20_seg"],
    ... )
    """
    source_store = Path(source_store)

    # Detect zarr format
    zarr_format = detect_zarr_format(source_store)
    vprintf(
        "Starting organelle segmentation pyramid build: store=%s, levels=%d, zarr_format=v%d",
        str(source_store),
        levels,
        zarr_format,
    )

    # Discover positions
    pos_paths = positions or _iter_position_paths(source_store)
    vprintf("Found %d positions to process", len(pos_paths))

    # Discover or use provided label names
    if label_names is None:
        discovered_labels = _discover_organelle_labels(source_store, pos_paths)
        if not discovered_labels:
            print("No organelle segmentation labels found in labels/ group")
            return source_store
        print(f"Discovered {len(discovered_labels)} organelle labels: {discovered_labels}")
        labels_to_build = discovered_labels
    else:
        labels_to_build = list(label_names)
        print(f"Building pyramids for specified labels: {labels_to_build}")

    # Build pyramids for each label using the unified functions
    for label_name in labels_to_build:
        print(f"\n{'='*60}")
        print(f"Processing labels/{label_name}")
        print(f"{'='*60}")

        # Initialize pyramid levels (using unified function with in_labels_group=True)
        _init_seg_levels(
            source_store, pos_paths, levels, label_name,
            in_labels_group=True, preserve_dtype=True
        )

        # Build pyramid (using unified function with in_labels_group=True)
        _build_seg_pyramid(
            source_store, pos_paths, levels, resume, label_name,
            in_labels_group=True, preserve_dtype=True,
            prebuilt_level0=prebuilt_level0,
        )

    print(f"\n✓ Organelle segmentation pyramid build complete for {source_store}")
    print(f"  Built pyramids for: {', '.join(labels_to_build)}")
    return source_store


def build_base_image_pyramids(
    source_store: str | Path,
    levels: int = 5,
    factor: int = 2,
    positions: Optional[Sequence[str]] = None,
    resume: bool = True,
    t_indices: Optional[Sequence[int]] = None,
) -> Path:
    """
    Build pyramid levels for base image channels only (not segmentation labels).

    This function builds/rebuilds the image pyramid levels (1, 2, 3, ...) for
    the main image data in the zarr store. It does NOT touch segmentation labels.

    Parameters
    ----------
    source_store : str | Path
        Path to the OME-Zarr store
    levels : int
        Number of pyramid levels to create (default: 5)
    factor : int
        Downsampling factor between levels (default: 2)
    positions : Optional[Sequence[str]]
        Specific positions to process (default: all positions)
    resume : bool
        If True, skip already-built levels; if False, rebuild all (default: True)

    Returns
    -------
    Path
        Path to the updated store

    Example
    -------
    >>> from cyclops_process.processes.pyramids.build_dask import build_base_image_pyramids
    >>> build_base_image_pyramids(
    ...     "/path/to/phenotyping.zarr",
    ...     levels=5,
    ...     resume=False,  # Force rebuild
    ... )
    """
    source_store = Path(source_store)

    # Detect zarr format
    zarr_format = detect_zarr_format(source_store)
    print(f"Building base image pyramids: store={source_store.name}, levels={levels}, factor={factor}, zarr_format=v{zarr_format}")

    # Discover positions
    pos_paths = list(positions) if positions else list(_iter_position_paths(source_store))
    if not pos_paths:
        print("No positions found in store")
        return source_store

    print(f"Found {len(pos_paths)} positions to process")

    # Initialize pyramid levels for images
    _init_image_levels(source_store, pos_paths, levels)

    # Build pyramids
    _build_image_pyramid(source_store, pos_paths, levels, factor, resume, t_indices=t_indices)

    print(f"\n✓ Base image pyramid build complete for {source_store}")
    return source_store
