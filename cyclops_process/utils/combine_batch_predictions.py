#!/usr/bin/env python
"""
Combine batch prediction outputs into a proper HCS OME-Zarr store.

This script combines the output from viscy_batch_inference.py, which writes
raw batch tensors (batch_NNNNN), into a properly structured HCS OME-Zarr
store with the same FOV layout as the input.

The batch inference saves data as:
    intermediate_dir/predictions_{start}_{end}.zarr/batch_NNNNN/

This script reconstructs the position ordering and writes to:
    output.zarr/Row/Well/Position/0  (proper HCS OME-Zarr)

Usage:
    python combine_batch_predictions.py \
        --intermediate-dir /path/to/intermediate \
        --input-store /path/to/input.zarr \
        --output-store /path/to/output.zarr \
        [--channel-names nuclei membrane] \
        [--num-workers 8]

The script uses the same position ordering logic as viscy_batch_inference.py
(sorted rows -> sorted wells -> sorted positions) to map batch indices back
to their original FOV locations.
"""

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import zarr
from iohub.ngff import open_ome_zarr
from joblib import Parallel, delayed
from tqdm import tqdm

from ops_utils.io.zarr_precreate import create_hcs_store_fast
from ops_utils.data.filesystem import async_delete_path
from ops_utils.hpc.resource_manager import get_optimal_workers
from ops_utils.io.zarr_utils import _validate_output_images


def _reconcile_scaffold_version(output_store: Path) -> None:
    """Fast-mv copies a v2 (.zattrs) plate scaffold but moves in v3 (zarr.json)
    positions; iohub 0.3.x can't read that mixed store. If the plate is v2 and
    its positions are v3, rewrite plate/row/well metadata as v3 (keeping extra
    keys like `normalization`) and drop the stale v2 files."""
    root = output_store / ".zattrs"
    if not root.exists():
        return  # plate scaffold already v3 — nothing to reconcile
    w0 = (json.loads(root.read_text()).get("plate", {}).get("wells") or [{}])[0].get("path")
    imgs = json.loads((output_store / w0 / ".zattrs").read_text())["well"]["images"] if w0 else []
    if not (imgs and (output_store / w0 / imgs[0]["path"] / "zarr.json").exists()):
        return  # positions aren't v3 — the store is already consistent

    def to_v3(gdir: Path, ome_key: str | None) -> None:
        a = json.loads((gdir / ".zattrs").read_text()) if (gdir / ".zattrs").exists() else {}
        attrs = {k: v for k, v in a.items() if k != ome_key}
        if ome_key in a:
            attrs["ome"] = {ome_key: {**a[ome_key], "version": "0.5"}, "version": "0.5"}
        (gdir / "zarr.json").write_text(json.dumps(
            {"attributes": attrs, "zarr_format": 3, "node_type": "group"}))
        (gdir / ".zattrs").unlink(missing_ok=True)
        (gdir / ".zgroup").unlink(missing_ok=True)

    to_v3(output_store, "plate")
    for row in output_store.iterdir():
        if row.is_dir() and not row.name.startswith("."):
            to_v3(row, None)
            for well in row.iterdir():
                if well.is_dir() and not well.name.startswith("."):
                    to_v3(well, "well")
    print("  [scaffold] converted v2 plate/well metadata -> v3 to match v3 positions")


def _pad_shape(shape: tuple[int, ...], target: int = 5) -> tuple[int, ...]:
    """Pad shape tuple to a target length (from viscy)."""
    pad = target - len(shape)
    return (1,) * pad + shape


def collect_all_positions(input_store: Path) -> list[dict]:
    """
    Collect all positions from input store in sorted order.

    This MUST match the ordering used by viscy_batch_inference.py's
    create_subset_store function to correctly map batch indices to positions.

    Returns list of dicts with 'row', 'well', 'position', 'path' keys.
    """
    positions = []

    for row_dir in sorted(input_store.iterdir()):
        if not row_dir.is_dir() or row_dir.name.startswith('.'):
            continue
        row_name = row_dir.name

        for well_dir in sorted(row_dir.iterdir()):
            if not well_dir.is_dir() or well_dir.name.startswith('.'):
                continue
            well_name = well_dir.name

            for pos_dir in sorted(well_dir.iterdir()):
                if not pos_dir.is_dir() or pos_dir.name.startswith('.'):
                    continue
                pos_name = pos_dir.name

                positions.append({
                    'row': row_name,
                    'well': well_name,
                    'position': pos_name,
                    'path': pos_dir,
                    'index': len(positions),
                })

    return positions


def discover_prediction_stores(intermediate_dir: Path) -> list[dict]:
    """
    Discover all prediction stores in the intermediate directory.

    Supports two formats:
    1. Batch format: predictions_{start}_{end}.zarr with batch_NNNNN groups
    2. HCS format: gpu{N}.zarr with Row/Well/Position structure

    Returns sorted list of dicts with 'path', 'start', 'end', 'format' keys.
    """
    stores = []

    for item in intermediate_dir.iterdir():
        if not item.is_dir():
            continue

        # Format 1: predictions_{start}_{end}.zarr (batch format)
        if item.name.startswith('predictions_') and item.name.endswith('.zarr'):
            # Parse range from name: predictions_{start}_{end}.zarr
            parts = item.name.replace('.zarr', '').split('_')
            if len(parts) >= 3:
                try:
                    start = int(parts[1])
                    end = int(parts[2])
                    stores.append({
                        'path': item,
                        'start': start,
                        'end': end,
                        'format': 'batch',
                    })
                except ValueError:
                    print(f"Warning: Could not parse range from {item.name}")

        # Format 2: gpu{N}.zarr (HCS format from viscy_multigpu_inference.py)
        elif item.name.startswith('gpu') and item.name.endswith('.zarr'):
            # gpu{N}.zarr - these are full HCS stores, we'll handle position mapping later
            stores.append({
                'path': item,
                'start': None,  # Will be determined by position ordering
                'end': None,
                'format': 'hcs',
            })

    # Sort by start position for batch format, by name for HCS format
    stores.sort(key=lambda x: (x['start'] if x['start'] is not None else 0, x['path'].name))
    return stores


def get_batch_info(prediction_store: Path) -> list[dict]:
    """
    Get information about batches in a prediction store.

    Returns list of dicts with 'name', 'path', 'index' (batch number).
    """
    batches = []

    store = zarr.open(str(prediction_store), mode='r')

    for name in sorted(store.keys()):
        if name.startswith('batch_'):
            try:
                idx = int(name.replace('batch_', ''))
                batches.append({
                    'name': name,
                    'path': prediction_store / name,
                    'index': idx,
                })
            except ValueError:
                pass

    return batches


def move_position_from_hcs(
    src_store_path: Path,
    dst_store_path: Path,
    row: str,
    well: str,
    position: str,
) -> bool:
    """
    Move position directory from source HCS store to destination (fast).

    Uses filesystem move (mv) which is instant because it only updates
    inode pointers - no data is actually copied. This is ~100x faster
    than reading/writing the array data.

    Args:
        src_store_path: Source zarr store (e.g., gpu0.zarr)
        dst_store_path: Destination zarr store
        row: Row name (e.g., 'A')
        well: Well name (e.g., '1')
        position: Position name (e.g., '000001')

    Returns:
        True if successful, False otherwise
    """
    try:
        src_pos_path = src_store_path / row / well / position
        dst_pos_path = dst_store_path / row / well / position

        # Check if source exists
        if not src_pos_path.exists():
            return False

        # Ensure destination parent (well) directory exists with metadata
        dst_well_path = dst_store_path / row / well
        dst_well_path.mkdir(parents=True, exist_ok=True)

        # Remove destination if it exists (for overwrite case)
        if dst_pos_path.exists():
            shutil.rmtree(dst_pos_path)

        # Move the entire position directory (instant - just updates inode)
        shutil.move(str(src_pos_path), str(dst_pos_path))

        return True

    except Exception as e:
        print(f"Error moving position {row}/{well}/{position}: {e}")
        return False


def collect_hcs_positions(hcs_store: Path) -> list[dict]:
    """
    Collect all positions from an HCS zarr store.

    Returns list of dicts with 'row', 'well', 'position' keys.
    """
    positions = []

    for row_dir in sorted(hcs_store.iterdir()):
        if not row_dir.is_dir() or row_dir.name.startswith('.'):
            continue
        row_name = row_dir.name

        for well_dir in sorted(row_dir.iterdir()):
            if not well_dir.is_dir() or well_dir.name.startswith('.'):
                continue
            well_name = well_dir.name

            for pos_dir in sorted(well_dir.iterdir()):
                if not pos_dir.is_dir() or pos_dir.name.startswith('.'):
                    continue
                pos_name = pos_dir.name

                positions.append({
                    'row': row_name,
                    'well': well_name,
                    'position': pos_name,
                })

    return positions


def get_scale_from_input(input_store: Path) -> Optional[list]:
    """Get scale metadata from input store."""
    try:
        with open_ome_zarr(input_store, mode='r') as plate:
            for _, pos in plate.positions():
                return pos.scale
    except Exception:
        pass
    return None


def get_volume_shape_from_input(input_store: Path) -> tuple[int, ...]:
    """
    Get the Z, Y, X dimensions from the input store.

    For 2.5D inference, predictions are written with Z=1 per batch sample,
    but the output store needs the full Z dimension from the input.

    Returns:
        Tuple of (Z, Y, X) dimensions
    """
    try:
        with open_ome_zarr(input_store, mode='r') as plate:
            for _, pos in plate.positions():
                # Shape is (T, C, Z, Y, X)
                shape = pos['0'].shape
                return shape[2:]  # Return (Z, Y, X)
    except Exception as e:
        raise ValueError(f"Could not read volume shape from input store: {e}")


def create_output_store(
    output_path: Path,
    channel_names: list[str],
    positions: list[dict],
    sample_shape: tuple,
    sample_dtype: np.dtype,
    scale: Optional[list] = None,
) -> None:
    """Create output HCS store with all positions pre-allocated.

    Uses fast_zarr_precreate for O(1) scaling metadata creation instead of
    iohub's O(n) overhead from repeated read-modify-write cycles.
    """
    prediction_channels = [ch + "_prediction" for ch in channel_names]

    # Convert positions to "row/well/position" format for fast_zarr_precreate
    position_paths = [
        f"{p['row']}/{p['well']}/{p['position']}"
        for p in positions
    ]

    # Shape: (T=1, C=num_channels, Z, Y, X)
    # sample_shape is (C, Z, Y, X)
    full_shape = (1,) + sample_shape

    # Chunks: (1, 1, 1, Y, X) - chunk per 2D slice
    chunks = _pad_shape(tuple(full_shape[-2:]), 5)

    # Scale: default to (1, 1, z_scale, y_scale, x_scale) if provided
    if scale is not None:
        # scale from input is typically [T, C, Z, Y, X] or [Z, Y, X]
        if len(scale) == 5:
            scale_tuple = tuple(scale)
        elif len(scale) == 3:
            # [Z, Y, X] -> [T=1, C=1, Z, Y, X]
            scale_tuple = (1.0, 1.0) + tuple(scale)
        else:
            scale_tuple = (1.0, 1.0, 1.0, 1.0, 1.0)
    else:
        scale_tuple = (1.0, 1.0, 1.0, 1.0, 1.0)

    print(f"  Using fast_zarr_precreate for {len(positions)} positions...")
    create_hcs_store_fast(
        store_path=output_path,
        positions=position_paths,
        shape=full_shape,
        chunks=chunks,
        dtype=sample_dtype,
        scale=scale_tuple,
        channel_names=prediction_channels,
    )


def write_position_data(
    output_store_path: Path,
    row: str,
    well: str,
    position: str,
    data: np.ndarray,
) -> bool:
    """
    Write data to a pre-existing position in the output store.

    This uses direct zarr array access (thread-safe) instead of iohub
    to avoid race conditions with concurrent writes.

    Args:
        output_store_path: Path to output zarr store
        row: Row name (e.g., 'A')
        well: Well name (e.g., '1')
        position: Position name (e.g., '002026')
        data: Prediction data (C, Z, Y, X)

    Returns:
        True if successful
    """
    try:
        # Direct zarr access is thread-safe for array writes
        store = zarr.open(str(output_store_path), mode="r+")
        arr_path = f"{row}/{well}/{position}/0"

        # Ensure data is 4D (C, Z, Y, X)
        if data.ndim == 5:
            data = data[0]  # Remove batch dim

        # Write directly to zarr array (thread-safe)
        # Array is pre-allocated with shape (1, C, Z, Y, X)
        store[arr_path][0, :, :, :, :] = data

        return True

    except Exception as e:
        print(f"Error writing position {row}/{well}/{position}: {e}")
        return False


def _write_position_slice(
    output_store_path: Path,
    row: str,
    well: str,
    position: str,
    z: int,
    slice_data: np.ndarray,
) -> bool:
    """Write a single Z slice (C, Y, X) into a pre-allocated FOV at depth z.

    Thread-safe (per-slice chunk). Used by the 2.5D combine path where a
    prediction batch may straddle FOV boundaries.
    """
    try:
        store = zarr.open(str(output_store_path), mode="r+")
        arr = store[f"{row}/{well}/{position}/0"]  # (1, C, Z, Y, X)
        arr[0, :, z, :, :] = slice_data
        return True
    except Exception as e:
        print(f"Error writing slice z={z} of {row}/{well}/{position}: {e}")
        return False


def load_and_write_batch(
    batch_info: dict,
    pred_zarr_path: str,
    positions_in_range: list,
    ps_start: int,
    batch_size: int,
    output_store_path: Path,
    volume_z: Optional[int] = None,
) -> int:
    """
    Load a single batch and immediately write to output store (streaming).

    This avoids accumulating data in memory by writing each slice immediately
    after loading it. Memory usage is O(1 batch) instead of O(all batches).

    For 2.5D inference (z_window_size=1) the predictor flattens (FOV, Z) into a
    single sample stream and writes it in fixed chunks of ``batch_size``. A
    batch therefore corresponds to one FOV ONLY when ``batch_size == volume_z``;
    otherwise batches straddle FOV boundaries. We map each sample to its
    (FOV, z) via its global sample index so any Z-depth assembles correctly.

    Returns:
        Number of Z slices successfully written.
    """
    pred_zarr = zarr.open(pred_zarr_path, mode='r')
    batch_name = batch_info['name']
    batch_idx = batch_info['index']
    written = 0

    # Load batch data - shape: (num_samples, C, 1, Y, X) for 2.5D inference
    batch_data = pred_zarr[batch_name][:]

    if batch_data.shape[2] == 1:
        # 2.5D mode: each sample is one Z slice of some FOV.
        batch_data = batch_data.squeeze(axis=2)  # (B, C, Y, X)
        B = batch_data.shape[0]
        # Legacy fallback: if Z is unknown assume one batch == one FOV
        # (correct only when batch_size == Z, the original assumption).
        z_per_fov = volume_z if volume_z else B
        for k in range(B):
            j = batch_idx * batch_size + k  # global sample index within this store
            pos_local, z = divmod(j, z_per_fov)
            if pos_local >= len(positions_in_range):
                break
            pos_info = positions_in_range[pos_local]
            if _write_position_slice(
                output_store_path,
                pos_info['row'],
                pos_info['well'],
                pos_info['position'],
                z,
                batch_data[k],  # (C, Y, X)
            ):
                written += 1
    else:
        # 3D mode or other: each sample is a separate position
        for sample_idx in range(batch_data.shape[0]):
            pos_idx_in_store = batch_idx * batch_size + sample_idx

            if pos_idx_in_store >= len(positions_in_range):
                break

            pos_info = positions_in_range[pos_idx_in_store]
            sample_data = batch_data[sample_idx]  # (C, Z, Y, X)

            success = write_position_data(
                output_store_path,
                pos_info['row'],
                pos_info['well'],
                pos_info['position'],
                sample_data,
            )
            if success:
                written += 1

    # Explicitly delete batch_data to free memory immediately
    del batch_data
    return written


def combine_predictions(
    intermediate_dir: Path,
    input_store: Path,
    output_store: Path,
    channel_names: list[str],
    batch_size: int = 7,
    num_workers: int = None,
) -> None:
    """
    Combine batch predictions into HCS OME-Zarr store.

    Supports two intermediate formats:
    1. Batch format: predictions_{start}_{end}.zarr with batch_NNNNN groups
    2. HCS format: gpu{N}.zarr with Row/Well/Position structure (from viscy_multigpu_inference.py)

    Args:
        intermediate_dir: Directory containing prediction stores
        input_store: Original input zarr store (for position ordering)
        output_store: Output HCS OME-Zarr store path
        channel_names: Output channel names (e.g., ['nuclei', 'membrane'])
        batch_size: Batch size used during inference (only used for batch format)
        num_workers: Number of parallel workers (auto-detected if None)
    """
    # Auto-detect optimal workers if not specified
    if num_workers is None:
        num_workers = get_optimal_workers(use_gpu=False, model_ram_gb=1.0, data_ram_gb=2.0)

    print("=" * 70)
    print("Combining Batch Predictions into HCS OME-Zarr")
    print("=" * 70)
    print(f"  Intermediate: {intermediate_dir}")
    print(f"  Input store:  {input_store}")
    print(f"  Output store: {output_store}")
    print(f"  Channels:     {channel_names}")
    print(f"  Batch size:   {batch_size}")
    print(f"  Workers:      {num_workers}")
    print("=" * 70)

    start_time = time.perf_counter()

    # Collect all positions from input store
    print("\nCollecting positions from input store...")
    all_positions = collect_all_positions(input_store)
    print(f"  Found {len(all_positions)} positions")

    # Discover prediction stores
    print("\nDiscovering prediction stores...")
    pred_stores = discover_prediction_stores(intermediate_dir)

    if not pred_stores:
        raise ValueError(f"No prediction stores found in {intermediate_dir}")

    # Determine format - check first store
    store_format = pred_stores[0].get('format', 'batch')
    print(f"  Found {len(pred_stores)} prediction stores (format: {store_format}):")
    for ps in pred_stores:
        if ps.get('format') == 'hcs':
            print(f"    {ps['path'].name}: HCS format")
        else:
            print(f"    {ps['path'].name}: positions {ps['start']}-{ps['end']}")

    # Get scale from input
    scale = get_scale_from_input(input_store)
    if scale:
        print(f"\nScale metadata: {scale}")

    # Get sample shape from predictions AND input store
    # For 2.5D inference (batch format), each batch sample has Z=1, but output needs full Z from input
    print("\nGetting sample shape...")
    sample_shape = None
    sample_dtype = None

    # Get the full volume shape (Z, Y, X) from the input store
    input_zyx_shape = get_volume_shape_from_input(input_store)
    print(f"  Input volume shape (Z, Y, X): {input_zyx_shape}")

    for ps_info in pred_stores:
        if ps_info.get('format') == 'hcs':
            # HCS format - read from first position (already has correct Z)
            hcs_positions = collect_hcs_positions(ps_info['path'])
            if hcs_positions:
                pos = hcs_positions[0]
                pred_zarr = zarr.open(str(ps_info['path']), mode='r')
                arr_path = f"{pos['row']}/{pos['well']}/{pos['position']}/0"
                if arr_path in pred_zarr:
                    arr = pred_zarr[arr_path]
                    # Shape is (T, C, Z, Y, X), we want (C, Z, Y, X)
                    sample_shape = arr.shape[1:]
                    sample_dtype = arr.dtype
                    print(f"  HCS prediction shape (C, Z, Y, X): {sample_shape}")
                    break
        else:
            # Batch format - Z=1 per sample, need to use input's Z dimension
            batches = get_batch_info(ps_info['path'])
            if batches:
                pred_zarr = zarr.open(str(ps_info['path']), mode='r')
                batch_data = pred_zarr[batches[0]['name']]
                # batch_data shape is (B, C, 1, Y, X) for 2.5D inference
                # We need (C, Z_from_input, Y, X) for the output
                num_channels = batch_data.shape[1]
                sample_dtype = batch_data.dtype
                # Use Z from input, Y/X from predictions (should match)
                sample_shape = (num_channels,) + input_zyx_shape
                print(f"  Batch prediction shape per sample: {batch_data.shape[1:]}")
                print(f"  Output shape (C, Z, Y, X): {sample_shape}")
                break

    if sample_shape is None:
        raise ValueError("No prediction data found in intermediate stores")

    # Rebuild from scratch: every position is rewritten below.
    async_delete_path(output_store)
    # Process based on format
    total_positions_written = 0
    total_batches_processed = 0

    # Separate stores by format
    hcs_stores = [ps for ps in pred_stores if ps.get('format') == 'hcs']
    batch_stores = [ps for ps in pred_stores if ps.get('format') != 'hcs']

    # Determine which creation method to use based on store types
    # HCS stores use fast mv (metadata-only output structure)
    # Batch stores need pre-allocated arrays for writing
    use_fast_mv = len(hcs_stores) > 0 and len(batch_stores) == 0

    print(f"\nCreating output store: {output_store}")
    if use_fast_mv:
        # Fast path: create metadata-only structure, positions will be moved in
        print("  Using fast mv mode (metadata only)...")
        output_store.mkdir(parents=True, exist_ok=True)

        # Copy plate-level metadata from input store
        for meta_file in [".zattrs", ".zgroup"]:
            src_meta = input_store / meta_file
            if src_meta.exists():
                shutil.copy(src_meta, output_store / meta_file)

        # Create row directories with metadata
        for row_dir in sorted(input_store.iterdir()):
            if not row_dir.is_dir() or row_dir.name.startswith('.'):
                continue
            row_out = output_store / row_dir.name
            row_out.mkdir(exist_ok=True)
            row_zgroup = row_dir / ".zgroup"
            if row_zgroup.exists():
                shutil.copy(row_zgroup, row_out / ".zgroup")

            # Create well directories with metadata
            for well_dir in sorted(row_dir.iterdir()):
                if not well_dir.is_dir() or well_dir.name.startswith('.'):
                    continue
                well_out = row_out / well_dir.name
                well_out.mkdir(exist_ok=True)
                for meta_file in [".zattrs", ".zgroup"]:
                    src_meta = well_dir / meta_file
                    if src_meta.exists():
                        shutil.copy(src_meta, well_out / meta_file)
    else:
        # Slow path: pre-allocate arrays for batch format stores
        print("  Using pre-allocated arrays mode...")
        create_output_store(output_store, channel_names, all_positions, sample_shape, sample_dtype, scale)

    # Process HCS format stores (gpu*.zarr) using fast mv
    if hcs_stores:
        print(f"\nProcessing {len(hcs_stores)} HCS format store(s) (fast mv mode)...")

        for ps_info in tqdm(hcs_stores, desc="HCS stores"):
            ps_path = ps_info['path']

            # Collect positions from this HCS store
            hcs_positions = collect_hcs_positions(ps_path)
            print(f"  {ps_path.name}: {len(hcs_positions)} positions (moving...)")

            # Move positions - this is instant (just updates inodes)
            for pos in tqdm(hcs_positions, desc=f"  Moving {ps_path.name}", leave=False):
                success = move_position_from_hcs(
                    ps_path,
                    output_store,
                    pos['row'],
                    pos['well'],
                    pos['position'],
                )
                if success:
                    total_positions_written += 1

    # Process batch format stores (predictions_*.zarr)
    # STREAMING MODE: Process batches sequentially to avoid OOM
    # Each batch is loaded, written, and freed before loading the next
    if batch_stores:
        print(f"\nProcessing {len(batch_stores)} batch format store(s) (streaming mode)...")

        for ps_info in tqdm(batch_stores, desc="Prediction stores"):
            ps_path = ps_info['path']
            ps_start = ps_info['start']
            ps_end = ps_info['end']

            # Get batches in this store
            batches = get_batch_info(ps_path)

            if not batches:
                continue

            # The positions for this store are [ps_start, ps_end)
            positions_in_range = all_positions[ps_start:ps_end]

            # Stream batches: load and write each batch immediately
            # Using limited parallelism to control memory usage
            # Each worker loads ONE batch at a time, writes it, then moves on
            max_concurrent_batches = min(num_workers, 8)  # Cap concurrent loads to limit memory

            results = Parallel(n_jobs=max_concurrent_batches, prefer="threads")(
                delayed(load_and_write_batch)(
                    batch_info,
                    str(ps_path),
                    positions_in_range,
                    ps_start,
                    batch_size,
                    output_store,
                    input_zyx_shape[0],  # Z slices per FOV
                )
                for batch_info in tqdm(batches, desc=f"  Streaming {ps_path.name}", leave=False)
            )

            total_batches_processed += len(batches)
            total_positions_written += sum(results)

    elapsed = time.perf_counter() - start_time

    # The fast-mv scaffold is copied from the (v2) input while positions are
    # moved in from v3 shards — reconcile so the store isn't a mixed v2/v3
    # plate that iohub 0.3.x refuses to read (blocks segment_and_stitch etc.).
    _reconcile_scaffold_version(output_store)

    # Validate output images are not blank/zeros
    # Sample positions across all wells to catch blank outputs
    _validate_output_images(output_store, n_samples=10, raise_on_blank=True)

    print("\n" + "=" * 70)
    print("COMBINE COMPLETE")
    print("=" * 70)
    print(f"  Positions written: {total_positions_written}")
    if total_batches_processed > 0:
        print(f"  Batches processed: {total_batches_processed}")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Output: {output_store}")
    print("=" * 70)


def create_combine_output_store(
    intermediate_dir: Path,
    input_store: Path,
    output_store: Path,
    channel_names: list[str],
    batch_size: int = 7,
) -> None:
    """
    Phase 1 of parallel combine: create the pre-allocated output store.

    Discovers prediction stores, collects metadata, and creates the output
    HCS OME-Zarr store with all positions pre-allocated. This must complete
    before parallel per-store streaming jobs begin.

    Args:
        intermediate_dir: Directory containing prediction stores
        input_store: Original input zarr store (for position ordering)
        output_store: Output HCS OME-Zarr store path to create
        channel_names: Output channel names (e.g., ['nuclei', 'membrane'])
        batch_size: Batch size used during inference
    """
    start_time = time.perf_counter()

    # WARNING: all prints in this function go to stdout. The caller
    # (virtual_staining_combine_setup) appends the n_jobs count as the LAST
    # line of stdout so Nextflow can parse it via .readLines().last().toInteger().
    # Do NOT add prints after the caller's final print(), and do NOT redirect
    # these prints to suppress them — Nextflow captures the full stdout and
    # relies on the last line being the numeric count.
    print("=" * 70)
    print("Phase 1: Creating Output Store for Parallel Combine")
    print("=" * 70)
    print(f"  Intermediate: {intermediate_dir}")
    print(f"  Input store:  {input_store}")
    print(f"  Output store: {output_store}")
    print("=" * 70)

    # Collect all positions from input store
    print("\nCollecting positions from input store...")
    all_positions = collect_all_positions(input_store)
    print(f"  Found {len(all_positions)} positions")

    # Discover prediction stores
    print("\nDiscovering prediction stores...")
    pred_stores = discover_prediction_stores(intermediate_dir)
    if not pred_stores:
        raise ValueError(f"No prediction stores found in {intermediate_dir}")

    batch_stores = [ps for ps in pred_stores if ps.get('format') != 'hcs']
    print(f"  Found {len(batch_stores)} batch-format prediction stores")

    # Get scale from input
    scale = get_scale_from_input(input_store)
    if scale:
        print(f"\nScale metadata: {scale}")

    # Get sample shape from first batch store
    print("\nGetting sample shape...")
    input_zyx_shape = get_volume_shape_from_input(input_store)
    print(f"  Input volume shape (Z, Y, X): {input_zyx_shape}")

    sample_shape = None
    sample_dtype = None
    for ps_info in batch_stores:
        batches = get_batch_info(ps_info['path'])
        if batches:
            pred_zarr = zarr.open(str(ps_info['path']), mode='r')
            batch_data = pred_zarr[batches[0]['name']]
            num_channels = batch_data.shape[1]
            sample_dtype = batch_data.dtype
            sample_shape = (num_channels,) + input_zyx_shape
            print(f"  Batch prediction shape per sample: {batch_data.shape[1:]}")
            print(f"  Output shape (C, Z, Y, X): {sample_shape}")
            break

    if sample_shape is None:
        raise ValueError("No prediction data found in intermediate stores")

    # Rebuild from scratch.
    async_delete_path(output_store)
    print(f"\nCreating output store: {output_store}")
    print("  Using pre-allocated arrays mode...")
    create_output_store(output_store, channel_names, all_positions, sample_shape, sample_dtype, scale)

    elapsed = time.perf_counter() - start_time
    print(f"\nStore creation complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")


def combine_single_batch_store(
    intermediate_dir: Path,
    input_store: Path,
    output_store: Path,
    store_index: int,
    batch_size: int = 7,
    num_workers: int = 8,
) -> int:
    """
    Phase 2 of parallel combine: stream one prediction store into the output.

    Each prediction store writes to non-overlapping positions in the output,
    so multiple instances of this function can safely run in parallel.

    Args:
        intermediate_dir: Directory containing prediction stores
        input_store: Original input zarr store (for position ordering)
        output_store: Pre-existing output HCS OME-Zarr store
        store_index: Index into the sorted list of batch-format prediction stores
        batch_size: Batch size used during inference
        num_workers: Number of parallel batch loaders within this store

    Returns:
        Number of positions successfully written.
    """
    start_time = time.perf_counter()

    # Discover stores and positions (fast - just directory listing)
    all_positions = collect_all_positions(input_store)
    pred_stores = discover_prediction_stores(intermediate_dir)
    batch_stores = [ps for ps in pred_stores if ps.get('format') != 'hcs']

    if store_index >= len(batch_stores):
        raise IndexError(f"store_index {store_index} >= {len(batch_stores)} batch stores")

    ps_info = batch_stores[store_index]
    ps_path = ps_info['path']
    ps_start = ps_info['start']
    ps_end = ps_info['end']

    print(f"Processing store {store_index}: {ps_path.name} (positions {ps_start}-{ps_end})")

    # Get batches in this store
    batches = get_batch_info(ps_path)
    if not batches:
        raise ValueError(f"No batches found in {ps_path}")

    positions_in_range = all_positions[ps_start:ps_end]
    volume_z = get_volume_shape_from_input(input_store)[0]  # Z slices per FOV

    # Stream batches with limited parallelism
    max_concurrent_batches = min(num_workers, 8)

    results = Parallel(n_jobs=max_concurrent_batches, prefer="threads")(
        delayed(load_and_write_batch)(
            batch_info,
            str(ps_path),
            positions_in_range,
            ps_start,
            batch_size,
            output_store,
            volume_z,
        )
        for batch_info in tqdm(batches, desc=f"  Streaming {ps_path.name}")
    )

    total_written = sum(results)
    elapsed = time.perf_counter() - start_time
    print(f"  Store {store_index} complete: {total_written} positions in {elapsed:.1f}s")

    return total_written


def validate_combine_output(
    output_store: Path,
    n_samples: int = 10,
) -> None:
    """
    Phase 3 of parallel combine: validate the combined output store.

    Checks that sampled positions have non-blank images.

    Args:
        output_store: Path to the combined output zarr store
        n_samples: Number of positions to sample for validation
    """
    print(f"Validating output store: {output_store}")
    _validate_output_images(output_store, n_samples=n_samples, raise_on_blank=True)
    print("Validation passed: output images are non-blank")


def main():
    parser = argparse.ArgumentParser(
        description='Combine batch predictions into HCS OME-Zarr'
    )
    parser.add_argument(
        '--intermediate-dir', '-i',
        type=Path,
        required=True,
        help='Directory containing predictions_X_Y.zarr stores'
    )
    parser.add_argument(
        '--input-store', '-s',
        type=Path,
        required=True,
        help='Original input zarr store (for position ordering and metadata)'
    )
    parser.add_argument(
        '--output-store', '-o',
        type=Path,
        required=True,
        help='Output HCS OME-Zarr store path'
    )
    parser.add_argument(
        '--channel-names', '-c',
        nargs='+',
        default=['nuclei', 'membrane'],
        help='Output channel names (default: nuclei membrane)'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=7,
        help='Batch size used during inference (default: 7)'
    )
    parser.add_argument(
        '--num-workers', '-w',
        type=int,
        default=None,
        help='Number of parallel workers (auto-detected if not specified)'
    )

    args = parser.parse_args()

    combine_predictions(
        intermediate_dir=args.intermediate_dir,
        input_store=args.input_store,
        output_store=args.output_store,
        channel_names=args.channel_names,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == '__main__':
    main()
