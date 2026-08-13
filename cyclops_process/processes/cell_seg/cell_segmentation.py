"""
Cell Segmentation with Hybrid IoU-based Stitching
=================================================

Segments cells from phenotyping_v3.zarr using Cellpose-SAM and writes
to 'cell_seg' label group.

Uses hybrid stitching:
- Tiles image with configurable overlap
- Segments each tile independently
- Merges labels across boundaries only if IoU > threshold
- Avoids both edge-cell loss (segment.py) and over-merging (organelle_seg)

Usage:
    # Preview mode (2x2 small tiles, in-memory debug, no zarr write)
    python -m cyclops_process.processes.cell_seg.cell_segmentation \\
        --experiment ops0033_20250429 --position A/1/0 --preview

    # Preview-full mode (2x2 production tiles, runs FULL pipeline)
    # Use this to validate end-to-end before running on full position
    python -m cyclops_process.processes.cell_seg.cell_segmentation \\
        --experiment ops0033_20250429 --position A/1/0 --preview-full

    # Full position segmentation
    python -m cyclops_process.processes.cell_seg.cell_segmentation \\
        --experiment ops0033_20250429 --position A/1/0
"""

import argparse
import time
from pathlib import Path

import numpy as np
try:
    import torch
except ModuleNotFoundError:  # optional GPU dep; used only in GPU runtime paths
    torch = None
from iohub import open_ome_zarr
from tqdm import tqdm
from dask.distributed import as_completed

import sys
import os
from ops_utils.profiling.decorators import versioned_function

sys.path.insert(0, os.getcwd())



# Verify cellpose >= 4.0.5 (SAM/Transformer architecture required). Skip the check
# where cellpose isn't installed (e.g. torch-less CI that only imports this module),
# so the module stays importable; the actual cell-seg run requires cellpose anyway.
from importlib.metadata import version as _pkg_version, PackageNotFoundError as _PkgNotFound
from packaging.version import Version as _Version
try:
    _cellpose_ver = _Version(_pkg_version("cellpose"))
    if _cellpose_ver < _Version("4.0.5"):
        raise RuntimeError(
            f"cellpose >= 4.0.5 (with SAM architecture) is required, "
            f"but found {_cellpose_ver}. Run: uv sync"
        )
except _PkgNotFound:
    pass

# Reuse from existing modules
from cyclops_process.processes.segment import preprocess_pheno_cells, _create_cellpose_model
from cyclops_process.utils.segmentation_utils import (
    torch_fastremap,
    torch_sparse_onehot,
    fast_sparse_dual_iou,
)
from ops_utils.data.experiment import OpsDataset
from ops_utils.hpc.gpu_utils import _setup_gpu_environment
from ops_utils.hpc.parallel_utils import MultiGPUCluster, _cleanup_worker_memory
from ops_utils.hpc.resource_manager import _measure_vram
from ops_utils.hpc.resource_manager import get_optimal_workers

# Zarr writing and metadata
from ops_utils.io.zarr_labels import (
    _init_organelle_label_array,
    _update_labels_metadata,
)
from cyclops_process.convert.v3_metadata import (
    build_label_metadata,
    write_label_metadata_to_store,
)

# Pyramid building
from cyclops_process.processes.pyramids.build_dask import build_organelle_seg_pyramids

# Resharding utility
from ops_utils.io.zarr_utils import reshard_zarr_array


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_TILE_SIZE = 4096  # Match organelle_seg for better GPU utilization
DEFAULT_OVERLAP = 512  # ~12.5% overlap (256 was too small for 4096 tiles)
DEFAULT_DIAMETER = 100
DEFAULT_FLOW_THRESHOLD = 0.7
DEFAULT_IOU_THRESHOLD = 0.1  # Higher than segment.py's 0.1 to be conservative

# Channel names for preprocessing (CLAHE takes max of nucleus + membrane)
DEFAULT_NUCLEI_CHANNEL = "nuclei_prediction"
DEFAULT_MEMBRANE_CHANNEL = "membrane_prediction"

# Cell painting mode: uses actin-only for CLAHE preprocessing (best results from sweep testing)
CP_NUCLEI_CHANNEL = "CP1_f_actin_Phalloidin"  # Actin channel (same as membrane for actin-only mode)
CP_MEMBRANE_CHANNEL = "CP1_f_actin_Phalloidin"  # Actin channel for cell boundaries
CP_OUTPUT_LABEL = "cp_cell_seg"

# Cell painting sweep configurations for testing different channel combinations
# Each config: (name, nuclei_channel, membrane_channel)
CP_SWEEP_CONFIGS = [
    ("actin_only", "CP1_f_actin_Phalloidin", "CP1_f_actin_Phalloidin"),
    ("actin_nuclei", "CP1_nuclei_Hoechst", "CP1_f_actin_Phalloidin"),
    ("actin_WGA", "CP1_plasma_membrane_WGA", "CP1_f_actin_Phalloidin"),
    ("actin_tubulin", "CP2_microtubules_Tubulin", "CP1_f_actin_Phalloidin"),
    ("actin_conA", "CP2_ER_ConA", "CP1_f_actin_Phalloidin"),
]

# 4i sweep configurations: focus on membrane-like antibody markers
# (RSP6, pS6, p53, p21 show cytoplasmic/membrane-ish staining patterns).
# Previous sweep identified p21 as the best single membrane channel — now try
# pairings of p21 as nuclei-input with other membranous channels as membrane.
# Each config: (name, nuclei_channel, membrane_channel)
FOUR_I_SWEEP_CONFIGS = [
    # Membranous channels paired with DAPI as baseline
    ("dapi_p21",       "4i_R1_nuclei_DAPI", "4i_R4_p21"),
    ("dapi_p53",       "4i_R1_nuclei_DAPI", "4i_R1_p53"),
    ("dapi_RSP6",      "4i_R1_nuclei_DAPI", "4i_R2_RSP6"),
    ("dapi_pS6",       "4i_R1_nuclei_DAPI", "4i_R5_pS6"),
    # p21 as nuclei-input, paired with other membranous channels
    ("p21_p53",        "4i_R4_p21",         "4i_R1_p53"),
    ("p21_RSP6",       "4i_R4_p21",         "4i_R2_RSP6"),
    ("p21_pS6",        "4i_R4_p21",         "4i_R5_pS6"),
    ("p21_bcatenin",   "4i_R4_p21",         "4i_R4_b-catenin"),
    # Other membranous nuclei-input pairings for completeness
    ("p53_p21",        "4i_R1_p53",         "4i_R4_p21"),
    ("RSP6_p21",       "4i_R2_RSP6",        "4i_R4_p21"),
    ("pS6_p21",        "4i_R5_pS6",         "4i_R4_p21"),
]

# Preview mode settings
PREVIEW_N_TILES = 2  # 2x2 grid
PREVIEW_TILE_SIZE = 1024
PREVIEW_OVERLAP = 128

# Preview-full mode: run entire pipeline on 2x2 full-size tiles
PREVIEW_FULL_N_TILES = 2  # 2x2 grid of full-size tiles
PREVIEW_FULL_LABEL_NAME = "cell_seg_preview"  # Separate label to avoid affecting real data


# ============================================================================
# DASK CLUSTER UTILITIES
# ============================================================================


# ============================================================================
# CHANNEL DETECTION
# ============================================================================

def _find_channel_index(
    channel_names: list[str],
    target_name: str,
) -> int:
    """
    Find the index of a channel by name (case-insensitive substring match).

    Args:
        channel_names: List of channel names from zarr metadata
        target_name: Target channel name to find

    Returns:
        Channel index, or -1 if not found
    """
    target = target_name.lower()
    for i, name in enumerate(channel_names):
        if target in str(name).lower():
            return i
    return -1


def _get_channel_indices(
    store_path: Path,
    nuclei_channel: str = DEFAULT_NUCLEI_CHANNEL,
    membrane_channel: str = DEFAULT_MEMBRANE_CHANNEL,
) -> tuple[list[str], int, int]:
    """
    Get channel indices for nuclei and membrane from zarr metadata.

    Args:
        store_path: Path to the zarr store
        nuclei_channel: Name of nuclei channel (default: "nuclei_prediction")
        membrane_channel: Name of membrane channel (default: "membrane_prediction")

    Returns:
        Tuple of (channel_names, nuclei_index, membrane_index)
        Raises ValueError if membrane channel not found.
    """
    with open_ome_zarr(str(store_path), mode="r") as ds:
        channel_names = list(ds.channel_names)

    nuclei_idx = _find_channel_index(channel_names, nuclei_channel)
    membrane_idx = _find_channel_index(channel_names, membrane_channel)

    if membrane_idx < 0:
        raise ValueError(
            f"Membrane channel '{membrane_channel}' not found in zarr metadata.\n"
            f"Available channels: {channel_names}"
        )

    if nuclei_idx < 0:
        raise ValueError(
            f"Nuclei channel '{nuclei_channel}' not found in zarr metadata.\n"
            f"Available channels: {channel_names}"
        )

    return channel_names, nuclei_idx, membrane_idx


# ============================================================================
# TILE GRID CALCULATION
# ============================================================================

def _calculate_tile_grid(
    height: int,
    width: int,
    tile_size: int,
    overlap: int,
) -> list[dict]:
    """
    Calculate tile coordinates for tiled processing.

    Returns list of tile info dicts with:
    - ty, tx: tile indices
    - y_start, y_end, x_start, x_end: global coordinates for full tile (with overlap)
    - core_y_start, core_y_end, core_x_start, core_x_end: global coordinates for core region

    The step size is (tile_size - overlap), ensuring tiles overlap by 'overlap' pixels.
    """
    step = tile_size - overlap
    n_tiles_y = max(1, (height - overlap + step - 1) // step)
    n_tiles_x = max(1, (width - overlap + step - 1) // step)

    tiles = []
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            # Full tile coordinates (includes overlap)
            y_start = ty * step
            x_start = tx * step
            y_end = min(y_start + tile_size, height)
            x_end = min(x_start + tile_size, width)

            # Core region coordinates (non-overlapping)
            core_y_start = y_start
            core_x_start = x_start
            core_y_end = min((ty + 1) * step, height)
            core_x_end = min((tx + 1) * step, width)

            tiles.append({
                "ty": ty,
                "tx": tx,
                "y_start": y_start,
                "y_end": y_end,
                "x_start": x_start,
                "x_end": x_end,
                "core_y_start": core_y_start,
                "core_y_end": core_y_end,
                "core_x_start": core_x_start,
                "core_x_end": core_x_end,
            })

    return tiles, n_tiles_y, n_tiles_x


# ============================================================================
# SINGLE TILE SEGMENTATION
# ============================================================================



def _get_cached_model(model_type: str = "cyto3", gpu: bool = True):
    """
    Get or create a cached Cellpose model for the current Dask worker.
    """
    from dask.distributed import get_worker

    try:
        worker = get_worker()
        cache_attr = f"_cellpose_model_{model_type}_{gpu}"
        if not hasattr(worker, cache_attr):
            setattr(worker, cache_attr, _create_cellpose_model(model_type, gpu))
        return getattr(worker, cache_attr)
    except ValueError:
        return _create_cellpose_model(model_type, gpu)


def _segment_tile_worker(
    tile_idx: int,
    tile_info: dict,
    source_path: str,
    position: str,
    y_offset: int,
    x_offset: int,
    diameter: float,
    flow_threshold: float,
    nuclei_channel: int,
    membrane_channel: int,
    use_clahe: bool,
    clahe_params: dict,
    model_type: str = "cyto3",
    # New parameters for memory-efficient zarr writing
    temp_label_name: str = None,
    tile_overlap: int = 512,
    n_tiles_y: int = 1,
    n_tiles_x: int = 1,
    label_offset: int = 0,
) -> dict:
    """
    Worker function for parallel tile segmentation with zarr writing.

    Memory-efficient approach:
    - Writes CORE region (non-overlapping) directly to temp zarr
    - Returns only overlap edges for IoU computation (~10GB vs ~93GB total)

    Args:
        tile_idx: Index of tile in processing order
        tile_info: Dict with tile coordinates (y_start, y_end, x_start, x_end, etc.)
        source_path: Path to source zarr store
        position: Position path (e.g., "A/1/0")
        y_offset, x_offset: Offset for preview mode cropping
        diameter, flow_threshold: Cellpose parameters
        nuclei_channel, membrane_channel: Channel indices
        use_clahe, clahe_params: Preprocessing parameters
        model_type: Cellpose model type
        temp_label_name: Name of temp label array in zarr (if None, returns full labels)
        tile_overlap: Overlap size in pixels
        n_tiles_y, n_tiles_x: Grid dimensions (to determine if tile has neighbors)
        label_offset: Offset to add to labels for global uniqueness

    Returns:
        Dict with tile info and overlap edges (not full labels) for IoU computation
    """
    import zarr

    source_path = Path(source_path)

    # Get cached model (GPU already assigned by _GPUAssignPlugin at worker startup)
    model = _get_cached_model(model_type, gpu=True)

    ty, tx = tile_info["ty"], tile_info["tx"]

    try:
        import time as _time

        # Load tile data (direct zarr, skip NGFF metadata scan)
        t0 = _time.monotonic()
        import zarr as _zarr
        source_arr = _zarr.open(str(source_path / position / "0"), mode="r")
        y0 = y_offset + tile_info["y_start"]
        y1 = y_offset + tile_info["y_end"]
        x0 = x_offset + tile_info["x_start"]
        x1 = x_offset + tile_info["x_end"]

        tile_data = np.asarray(source_arr[0, :, 0, y0:y1, x0:x1])
        t_read = _time.monotonic() - t0

        # Segment tile (with sub-step timing)
        t1 = _time.monotonic()
        labels, preprocessed = _segment_tile(
            tile_data,
            model,
            diameter=diameter,
            flow_threshold=flow_threshold,
            nuclei_channel=nuclei_channel,
            membrane_channel=membrane_channel,
            use_clahe=use_clahe,
            clahe_params=clahe_params,
        )
        t_seg = _time.monotonic() - t1

        n_labels = int(labels.max())

        # Log per-tile timing (first 10 tiles per worker)
        if tile_idx < 10:
            t_pre = getattr(_segment_tile, '_last_preprocess', 0)
            t_mod = getattr(_segment_tile, '_last_model', 0)
            print(f"  [Tile {tile_idx}] read={t_read:.2f}s preproc={t_pre:.2f}s "
                  f"model={t_mod:.2f}s total={t_read+t_seg:.2f}s labels={n_labels}")

        # If no temp_label_name provided (debug mode), return full labels
        if temp_label_name is None:
            return {
                "tile_idx": tile_idx,
                "ty": ty,
                "tx": tx,
                "labels": labels,
                "preprocessed": preprocessed,
                "raw_membrane": tile_data[membrane_channel] if tile_data.ndim == 3 else tile_data,
                "n_labels": n_labels,
            }

        # =========================================================================
        # In-memory mode: Return full tile labels for canvas writing
        # =========================================================================
        # Instead of writing cores to zarr, we return full tile labels.
        # The main function will write them to an in-memory canvas sequentially.
        # This ensures each cell's pixels come from ONE Cellpose run (consistent).

        # Cast to int32 and apply offset for global uniqueness
        labels_with_offset = labels.astype(np.int32)
        if labels_with_offset.max() > 0:
            labels_with_offset[labels_with_offset > 0] += label_offset

        # Extract overlap edges for IoU computation BEFORE canvas writing
        # We need all 4 edges because IoU compares A's right with B's left, etc.
        # Memory: ~29 GB for 900 tiles (4 edges × 512×4096 × 4 bytes × 900)
        right_overlap = None
        if tx + 1 < n_tiles_x and labels.shape[1] >= tile_overlap:
            right_overlap = labels_with_offset[:, -tile_overlap:].copy()

        bottom_overlap = None
        if ty + 1 < n_tiles_y and labels.shape[0] >= tile_overlap:
            bottom_overlap = labels_with_offset[-tile_overlap:, :].copy()

        left_overlap = None
        if tx > 0 and labels.shape[1] >= tile_overlap:
            left_overlap = labels_with_offset[:, :tile_overlap].copy()

        top_overlap = None
        if ty > 0 and labels.shape[0] >= tile_overlap:
            top_overlap = labels_with_offset[:tile_overlap, :].copy()

        return {
            "tile_idx": tile_idx,
            "ty": ty,
            "tx": tx,
            "n_labels": n_labels,
            # Full tile labels for canvas writing
            "labels": labels_with_offset,
            # Overlap edges for IoU computation
            "right_overlap": right_overlap,
            "bottom_overlap": bottom_overlap,
            "left_overlap": left_overlap,
            "top_overlap": top_overlap,
        }

    finally:
        # Release VRAM after each 4096×4096 tile (~20 GB per inference).
        # Without this, VRAM creeps up and eventually OOMs.
        torch.cuda.empty_cache()


def _segment_tile(
    tile_data: np.ndarray,
    model,
    diameter: float,
    flow_threshold: float,
    nuclei_channel: int,
    membrane_channel: int,
    use_clahe: bool = True,
    clahe_params: dict = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Segment a single tile using Cellpose.

    Args:
        tile_data: Raw tile data, shape (C, Y, X)
        model: Cellpose model
        diameter: Cell diameter
        flow_threshold: Flow threshold
        nuclei_channel: Channel index for nuclei (used in CLAHE max projection)
        membrane_channel: Channel index for membrane (used in CLAHE max projection)
        use_clahe: Whether to apply CLAHE preprocessing
        clahe_params: CLAHE parameters

    Returns:
        Tuple of (segmentation mask, preprocessed image) - both shape (Y, X)
    """
    import time as _time

    if clahe_params is None:
        clahe_params = {"clip_limit": 0.01}

    # Preprocess if needed
    t_pre = _time.monotonic()
    if use_clahe and tile_data.ndim == 3:
        # Add T and Z dims: (C, Y, X) -> (T, C, Z, Y, X)
        tile_5d = tile_data[np.newaxis, :, np.newaxis, :, :]

        preprocessed = preprocess_pheno_cells(
            tile_5d,
            channel_0=nuclei_channel,
            channel_1=membrane_channel,
            clahe_params=clahe_params,
        )
    else:
        # No preprocessing - just use membrane channel with percentile normalization
        if tile_data.ndim == 3:
            preprocessed = tile_data[membrane_channel].astype(np.float32)
        else:
            preprocessed = tile_data.astype(np.float32)

        # Normalize to [0, 1]
        p_min, p_max = np.percentile(preprocessed, [1, 99])
        preprocessed = np.clip((preprocessed - p_min) / (p_max - p_min + 1e-8), 0, 1)

    # Ensure preprocessed is float32 in [0, 1] for Cellpose
    if preprocessed.max() > 1.0:
        preprocessed = preprocessed.astype(np.float32)
        p_min, p_max = np.percentile(preprocessed, [1, 99])
        preprocessed = np.clip((preprocessed - p_min) / (p_max - p_min + 1e-8), 0, 1)
    t_preprocess = _time.monotonic() - t_pre

    # Run Cellpose
    t_eval = _time.monotonic()
    result = model.eval(
        preprocessed,
        diameter=diameter,
        flow_threshold=flow_threshold,
    )
    t_model = _time.monotonic() - t_eval
    masks = result[0]

    # Store timing on function for caller to access
    _segment_tile._last_preprocess = t_preprocess
    _segment_tile._last_model = t_model

    return masks, preprocessed


def _segment_tiles_batch_worker(
    tile_batch: list,
    source_path: str,
    position: str,
    y_offset: int,
    x_offset: int,
    diameter: float,
    flow_threshold: float,
    nuclei_channel: int,
    membrane_channel: int,
    use_clahe: bool,
    clahe_params: dict,
    model_type: str = "cyto3",
    temp_label_name: str = None,
    tile_overlap: int = 512,
    n_tiles_y: int = 1,
    n_tiles_x: int = 1,
    canvas_shm_name: str = None,
    canvas_height: int = 0,
    canvas_width: int = 0,
) -> list:
    """Batch worker: process multiple tiles with prefetch pipeline.

    Reads and preprocesses the next tile on a background thread while the
    GPU runs model.eval() on the current tile. This overlaps CPU work
    (zarr read + CLAHE) with GPU inference, eliminating the 5-6s idle gap.

    Labels are written directly to shared memory canvas (if provided),
    and only overlap strips are returned through Dask — reducing
    serialization from ~9.5GB to ~3GB per batch.
    """
    import zarr as _zarr
    import time as _time
    from concurrent.futures import ThreadPoolExecutor
    from collections import deque

    source_path = Path(source_path)

    # Limit torch intra-op thread pool: 6 workers share 64 cores, ~10 each.
    import torch as _torch_init
    _torch_init.set_num_threads(8)
    import os as _os_threads
    _os_threads.environ["OMP_NUM_THREADS"] = "8"
    _os_threads.environ["MKL_NUM_THREADS"] = "8"

    # Attach to shared memory canvas (created by parent process)
    _worker_canvas = None
    _worker_shm = None
    if canvas_shm_name:
        import multiprocessing.shared_memory as _shm_mod
        _worker_shm = _shm_mod.SharedMemory(name=canvas_shm_name)
        _worker_canvas = np.ndarray((canvas_height, canvas_width),
                                     dtype=np.int32, buffer=_worker_shm.buf)

    model = _get_cached_model(model_type, gpu=True)

    PREFETCH_DEPTH = 4  # tiles buffered ahead of GPU

    # Rescaling factor: model.eval() rescales to 30px/diameter
    _image_scaling = 30.0 / diameter  # 0.3 for diameter=100
    _niter_scaled = int(200 / _image_scaling)  # 666 for diameter=100

    def _read_and_preprocess(tile_entry):
        """Read tile, CLAHE, convert, normalize, rescale to 0.3x for run_net."""
        tile_info = tile_entry["tile_info"]
        source_arr = _zarr.open(str(source_path / position / "0"), mode="r")
        y0 = y_offset + tile_info["y_start"]
        y1 = y_offset + tile_info["y_end"]
        x0 = x_offset + tile_info["x_start"]
        x1 = x_offset + tile_info["x_end"]
        tile_data = np.asarray(source_arr[0, :, 0, y0:y1, x0:x1])

        Ly_0 = y1 - y0  # original tile height
        Lx_0 = x1 - x0  # original tile width

        # Preprocess (CLAHE or percentile normalization)
        if clahe_params is None:
            _clahe = {"clip_limit": 0.01}
        else:
            _clahe = clahe_params

        if use_clahe and tile_data.ndim == 3:
            tile_5d = tile_data[np.newaxis, :, np.newaxis, :, :]
            preprocessed = preprocess_pheno_cells(
                tile_5d,
                channel_0=nuclei_channel,
                channel_1=membrane_channel,
                clahe_params=_clahe,
            )
        else:
            if tile_data.ndim == 3:
                preprocessed = tile_data[membrane_channel].astype(np.float32)
            else:
                preprocessed = tile_data.astype(np.float32)
            p_min, p_max = np.percentile(preprocessed, [1, 99])
            preprocessed = np.clip(
                (preprocessed - p_min) / (p_max - p_min + 1e-8), 0, 1)

        if preprocessed.max() > 1.0:
            preprocessed = preprocessed.astype(np.float32)
            p_min, p_max = np.percentile(preprocessed, [1, 99])
            preprocessed = np.clip(
                (preprocessed - p_min) / (p_max - p_min + 1e-8), 0, 1)

        # Cellpose channel conversion + normalization
        img_conv = cp_transforms.convert_image(preprocessed)
        img_norm = cp_transforms.normalize_img(img_conv, normalize=True, norm3D=False)

        # Rescale to 0.3x for run_net (same as model.eval())
        Ly_new = int(Ly_0 * _image_scaling)
        Lx_new = int(Lx_0 * _image_scaling)
        # img_norm is (H, W, nchan) — resize_image expects (Y, X, nchan)
        img_rescaled = cp_transforms.resize_image(
            img_norm, Ly=Ly_new, Lx=Lx_new, no_channels=False
        )

        return (img_rescaled, Ly_0, Lx_0)  # return original dims for resize-back

    from cellpose import transforms as cp_transforms
    from cellpose import dynamics as cp_dynamics
    from cellpose import core as cp_core

    io_pool = ThreadPoolExecutor(max_workers=PREFETCH_DEPTH + 1,
                                  thread_name_prefix="cseg_io")

    def _run_gpu(preprocess_result):
        """Run GPU forward pass on rescaled image, resize outputs back.

        Replicates model.eval() pipeline:
        1. run_net on 0.3x image (fast — 11x fewer pixels)
        2. Resize dP and cellprob back to original resolution
        """
        img_rescaled, Ly_0, Lx_0 = preprocess_result

        stacked = img_rescaled[np.newaxis, ...]  # (1, H_small, W_small, nchan)
        yf, styles = cp_core.run_net(
            model.net, stacked,
            batch_size=64,
            bsize=256,
        )

        # Extract dP and cellprob at rescaled resolution
        # yf shape: (1, H_small, W_small, 3) — last dim is (dY, dX, cellprob)
        dP_small = yf[0, :, :, :2].transpose(2, 0, 1)  # (2, H_small, W_small)
        cellprob_small = yf[0, :, :, 2]  # (H_small, W_small)

        # Resize back to original resolution (same as model.eval() lines 196-197)
        # model.eval() does: dP.transpose(1,2,3,0) -> resize -> .transpose(3,0,1,2)
        # For 2D: dP (2, H, W) -> (H, W, 2) -> resize -> (H, W, 2) -> (2, H, W)
        dP_for_resize = dP_small.transpose(1, 2, 0)  # (H_small, W_small, 2)
        dP_resized = cp_transforms.resize_image(
            dP_for_resize, Ly=Ly_0, Lx=Lx_0, no_channels=False
        )  # (Ly_0, Lx_0, 2)
        dP_full = dP_resized.transpose(2, 0, 1)  # (2, Ly_0, Lx_0)

        cellprob_full = cp_transforms.resize_image(
            cellprob_small, Ly=Ly_0, Lx=Lx_0, no_channels=True
        )

        # Repack into yf format for _compute_masks: (H, W, 3)
        yf_full = np.zeros((Ly_0, Lx_0, 3), dtype=np.float32)
        yf_full[:, :, :2] = dP_full.transpose(1, 2, 0)
        yf_full[:, :, 2] = cellprob_full

        return yf_full

    def _fast_get_masks_torch(pt, inds, shape0, rpad=20, max_size_fraction=0.4):
        """Vectorized get_masks_torch: unfold + batch scatter replaces Python loops.

        Same as segment.py's optimized version — the biggest single win
        from the segment_and_stitch optimization (47% mask time reduction).
        """
        import torch as _torch
        import torch.nn.functional as _F
        import fastremap

        device = pt.device
        shape0_arr = np.array(shape0)
        pt = pt + rpad
        pt = _torch.clamp(pt, min=0)
        for i in range(len(pt)):
            pt[i] = _torch.clamp(pt[i], max=shape0_arr[i] + rpad - 1)
        shape = tuple(shape0_arr + 2 * rpad)

        coo = _torch.sparse_coo_tensor(
            pt, _torch.ones(pt.shape[1], device=device, dtype=_torch.int), shape)
        h1 = coo.to_dense()
        del coo

        h1_4d = h1.unsqueeze(0).unsqueeze(0).float()
        hmax1 = _F.max_pool2d(h1_4d, kernel_size=5, stride=1, padding=2)
        hmax1 = hmax1.squeeze().int()
        seeds1 = _torch.nonzero((h1 - hmax1 > -1e-6) * (h1 > 10))
        del hmax1, h1_4d

        if len(seeds1) == 0:
            return np.zeros(shape0, dtype="uint16")

        npts = h1[tuple(seeds1.T)]
        seeds1 = seeds1[npts.argsort()]
        n_seeds = len(seeds1)

        # Vectorized 11×11 patch extraction via unfold
        patches = h1.unfold(0, 11, 1).unfold(1, 11, 1)
        h_slc = patches[seeds1[:, 0] - 5, seeds1[:, 1] - 5]
        del h1

        seed_masks = _torch.zeros((n_seeds, 11, 11), device=device)
        seed_masks[:, 5, 5] = 1
        for _ in range(5):
            sm_4d = seed_masks.unsqueeze(1)
            sm_4d = _F.max_pool2d(sm_4d, kernel_size=3, stride=1, padding=1)
            seed_masks = sm_4d.squeeze(1)
            seed_masks *= (h_slc > 2)
        del h_slc

        # Vectorized label assignment
        batch_idx, row_idx, col_idx = _torch.nonzero(seed_masks, as_tuple=True)
        global_row = row_idx + seeds1[batch_idx, 0] - 5
        global_col = col_idx + seeds1[batch_idx, 1] - 5
        del seed_masks

        dtype = _torch.int32 if n_seeds < 2**16 else _torch.int64
        M1 = _torch.zeros(shape, dtype=dtype, device=device)
        M1[global_row, global_col] = (1 + batch_idx).to(dtype)
        M1 = M1[tuple(pt)].cpu().numpy()

        dtype = "uint16" if n_seeds < 2**16 else "uint32"
        M0 = np.zeros(shape0, dtype=dtype)
        M0[inds] = M1

        uniq, counts = fastremap.unique(M0, return_counts=True)
        big = np.prod(shape0) * max_size_fraction
        bigc = uniq[counts > big]
        if len(bigc) > 0 and (len(bigc) > 1 or bigc[0] != 0):
            M0 = fastremap.mask(M0, bigc)
        fastremap.renumber(M0, in_place=True)
        return M0.reshape(tuple(shape0))

    # niter for follow_flows: 200 is the Cellpose default for native-resolution images.
    # model.eval() scales niter by 1/image_scaling when it rescales the image (e.g., 666
    # for 0.3x), but we run run_net on full 4096px without rescaling, so niter=200 is correct.
    _niter = 200

    def _sor_extend_centers_gpu(neighbors, meds, isneighbor, shape, n_iter=200,
                                device=torch.device("cpu"), omega=1.3):
        """SOR-accelerated diffusion — drop-in replacement for _extend_centers_gpu.

        Successive Over-Relaxation converges 3-5x faster than Jacobi iteration
        for the Laplace equation. Uses the same neighbor structure and produces
        the same converged flow field, just reaches it in fewer iterations.

        omega=1.7 is near-optimal for typical mask geometries (theoretical
        optimum for a square grid is 2/(1+sin(pi/N)) ≈ 1.7-1.9 for N~50-200).
        """
        if torch.prod(torch.tensor(shape)) > 4e7 or device.type == "mps":
            T = torch.zeros(shape, dtype=torch.float, device=device)
        else:
            T = torch.zeros(shape, dtype=torch.double, device=device)

        for i in range(n_iter):
            T[tuple(meds.T)] += 1
            Tneigh = T[tuple(neighbors)]
            Tneigh *= isneighbor
            T_avg = Tneigh.mean(axis=0)
            # SOR update: blend current value with neighbor average
            T_old = T[tuple(neighbors[:, 0])]
            T[tuple(neighbors[:, 0])] = (1.0 - omega) * T_old + omega * T_avg
        del meds, isneighbor, Tneigh, T_avg, T_old

        if T.ndim == 2:
            grads = T[neighbors[0, [2, 1, 4, 3]], neighbors[1, [2, 1, 4, 3]]]
            del neighbors
            dy = grads[0] - grads[1]
            dx = grads[2] - grads[3]
            del grads
            mu_torch = np.stack((dy.cpu().squeeze(0), dx.cpu().squeeze(0)), axis=-2)
        else:
            grads = T[tuple(neighbors[:, 1:])]
            del neighbors
            dz = grads[0] - grads[1]
            dy = grads[2] - grads[3]
            dx = grads[4] - grads[5]
            del grads
            mu_torch = np.stack(
                (dz.cpu().squeeze(0), dy.cpu().squeeze(0), dx.cpu().squeeze(0)), axis=-2)
        return mu_torch

    def _sor_masks_to_flows_gpu(masks, device=None, niter=None, omega=1.3):
        """SOR-accelerated masks_to_flows_gpu — same algorithm, faster convergence.

        Drop-in replacement for cellpose.dynamics.masks_to_flows_gpu that uses
        SOR instead of Jacobi iteration. Produces the same flow field with
        3-5x fewer iterations needed for convergence.
        """
        import torch.nn.functional as F
        from scipy.ndimage import find_objects
        from cellpose.dynamics import get_centers

        if device is None:
            device = model.device

        if masks.max() > 0:
            Ly0, Lx0 = masks.shape
            Ly, Lx = Ly0 + 2, Lx0 + 2

            masks_padded = torch.from_numpy(masks.astype("int64")).to(device)
            masks_padded = F.pad(masks_padded, (1, 1, 1, 1))
            shape = masks_padded.shape

            y, x = torch.nonzero(masks_padded, as_tuple=True)
            y = y.int()
            x = x.int()
            neighbors = torch.zeros((2, 9, y.shape[0]), dtype=torch.int, device=device)
            yxi = [[0, -1, 1, 0, 0, -1, -1, 1, 1], [0, 0, 0, -1, 1, -1, 1, -1, 1]]
            for i in range(9):
                neighbors[0, i] = y + yxi[0][i]
                neighbors[1, i] = x + yxi[1][i]
            isneighbor = torch.ones((9, y.shape[0]), dtype=torch.bool, device=device)
            m0 = masks_padded[neighbors[0, 0], neighbors[1, 0]]
            for i in range(1, 9):
                isneighbor[i] = masks_padded[neighbors[0, i], neighbors[1, i]] == m0
            del m0, masks_padded

            slices = find_objects(masks)
            centers, ext = get_centers(masks, slices)
            meds_p = torch.from_numpy(centers).to(device).long()
            meds_p += 1

            # SOR with fewer iterations: omega=1.7 converges ~3-4x faster
            n_iter = 2 * ext.max() if niter is None else niter
            # Reduce iterations by convergence speedup factor
            sor_niter = max(30, n_iter // 3)

            mu = _sor_extend_centers_gpu(neighbors, meds_p, isneighbor, shape,
                                         n_iter=sor_niter, device=device, omega=omega)
            mu = mu.astype("float64")
            mu /= (1e-60 + (mu**2).sum(axis=0)**0.5)

            mu0 = np.zeros((2, Ly0, Lx0))
            mu0[:, y.cpu().numpy() - 1, x.cpu().numpy() - 1] = mu
        else:
            mu0 = np.zeros((2, masks.shape[0], masks.shape[1]))
        return mu0

    def _fast_remove_bad_flow_masks(masks, dP_net, threshold=0.4):
        """SOR-accelerated flow QC: same algorithm as Cellpose, 3-4x faster.

        Uses Successive Over-Relaxation (omega=1.7) to solve the same Laplace
        diffusion equation as masks_to_flows_gpu, but converges in 1/3 the
        iterations. Produces the same flow field and same MSE values, so the
        same threshold filters the same masks.
        """
        from scipy.ndimage import mean as _mean

        if masks.max() == 0:
            return masks

        # Compute flows from masks using SOR-accelerated diffusion
        dP_masks = _sor_masks_to_flows_gpu(masks, device=model.device)

        # Compute per-mask flow error (same as Cellpose flow_error)
        flow_errors = np.zeros(masks.max())
        for i in range(dP_masks.shape[0]):
            flow_errors += _mean((dP_masks[i] - dP_net[i] / 5.) ** 2, masks,
                                 index=np.arange(1, masks.max() + 1))

        # Remove bad masks (same as Cellpose remove_bad_flow_masks)
        badi = 1 + (flow_errors > threshold).nonzero()[0]
        if len(badi) > 0:
            masks[np.isin(masks, badi)] = 0

        return masks

    def _compute_masks(yf):
        """Compute masks replicating model.eval() pipeline exactly.

        dP and cellprob are already at original resolution (resized back from
        0.3x by _run_gpu). This matches model.eval() which also runs
        compute_masks at original resolution.

        - follow_flows with niter=666 (scaled: 200/0.3)
        - get_masks (vectorized)
        - SOR-accelerated flow QC (same diffusion, fewer iterations)
        - fill_holes_and_remove_small_masks(min_size=15)
        """
        import torch as _torch
        from cellpose import utils as cp_utils

        dP = yf[:, :, :2].transpose(2, 0, 1)
        cellprob = yf[:, :, 2]

        if (cellprob > 0.0).sum() == 0:
            return np.zeros(cellprob.shape, "uint16")
        inds = np.nonzero(cellprob > 0.0)
        if len(inds[0]) == 0:
            return np.zeros(cellprob.shape, "uint16")

        p_final = cp_dynamics.follow_flows(
            dP * (cellprob > 0.0) / 5.,
            inds=inds, niter=_niter_scaled, device=model.device)
        if not _torch.is_tensor(p_final):
            p_final = _torch.from_numpy(p_final).to(model.device, dtype=_torch.int)
        else:
            p_final = p_final.int()

        mask = _fast_get_masks_torch(
            p_final, inds, dP.shape[1:], max_size_fraction=0.4)
        del p_final

        # SOR-accelerated flow QC (same algorithm as Cellpose, faster convergence)
        if mask.max() > 0 and flow_threshold is not None and flow_threshold > 0:
            mask = _fast_remove_bad_flow_masks(mask, dP, threshold=flow_threshold)

        if mask.max() < 2**16 and mask.dtype != "uint16":
            mask = mask.astype("uint16")

        mask = cp_utils.fill_holes_and_remove_small_masks(mask, min_size=15)
        return mask

    def _build_tile_result(entry, masks, results_list):
        """Write labels to shared memory canvas, return only overlaps through Dask.

        This reduces Dask serialization from ~9.5GB to ~3GB per batch by
        writing the 64MB labels array directly to shared memory instead of
        returning it through pickle.
        """
        tile_info = entry["tile_info"]
        tile_idx = entry["tile_idx"]
        label_offset = entry["label_offset"]
        ty, tx = tile_info["ty"], tile_info["tx"]
        n_labels = int(masks.max())

        labels_with_offset = masks.astype(np.int32)
        if labels_with_offset.max() > 0:
            labels_with_offset[labels_with_offset > 0] += label_offset

        # Write labels directly to shared memory canvas (zero-copy, no Dask)
        if _worker_canvas is not None and labels_with_offset.max() > 0:
            y0 = tile_info["y_start"]
            y1 = tile_info["y_end"]
            x0 = tile_info["x_start"]
            x1 = tile_info["x_end"]
            fg = labels_with_offset > 0
            _worker_canvas[y0:y1, x0:x1][fg] = labels_with_offset[fg]

        # Extract overlaps (returned through Dask for Pass 2 IoU)
        right_overlap = None
        if tx + 1 < n_tiles_x and masks.shape[1] >= tile_overlap:
            right_overlap = labels_with_offset[:, -tile_overlap:].copy()
        bottom_overlap = None
        if ty + 1 < n_tiles_y and masks.shape[0] >= tile_overlap:
            bottom_overlap = labels_with_offset[-tile_overlap:, :].copy()
        left_overlap = None
        if tx > 0 and masks.shape[1] >= tile_overlap:
            left_overlap = labels_with_offset[:, :tile_overlap].copy()
        top_overlap = None
        if ty > 0 and masks.shape[0] >= tile_overlap:
            top_overlap = labels_with_offset[:tile_overlap, :].copy()

        results_list.append({
            "tile_idx": tile_idx,
            "ty": ty,
            "tx": tx,
            "n_labels": n_labels,
            # No "labels" — written to shared memory canvas directly
            "right_overlap": right_overlap,
            "bottom_overlap": bottom_overlap,
            "left_overlap": left_overlap,
            "top_overlap": top_overlap,
        })

    # Prime prefetch queue
    prefetch_q = deque()
    prime_count = min(PREFETCH_DEPTH, len(tile_batch))
    for k in range(prime_count):
        prefetch_q.append(
            (tile_batch[k], io_pool.submit(_read_and_preprocess, tile_batch[k])))
    next_idx = prime_count

    results = []
    t_read_total = 0.0
    t_gpu_total = 0.0
    t_masks_total = 0.0
    t_post_total = 0.0

    # Pipeline: overlap GPU run_net (tile N+1) with compute_masks (tile N)
    prev_yf = None
    prev_entry = None
    gpu_future = None

    try:
        for i in range(len(tile_batch)):
            # Get preprocessed tile from prefetch queue
            t0 = _time.monotonic()
            entry, read_future = prefetch_q.popleft()
            preprocessed = read_future.result()
            t_read_total += _time.monotonic() - t0

            # Refill prefetch queue
            if next_idx < len(tile_batch):
                prefetch_q.append(
                    (tile_batch[next_idx],
                     io_pool.submit(_read_and_preprocess, tile_batch[next_idx])))
                next_idx += 1

            # Submit GPU forward pass for this tile in a thread
            gpu_future = io_pool.submit(_run_gpu, preprocessed)

            # While GPU runs, compute masks for PREVIOUS tile
            if prev_yf is not None:
                t_m = _time.monotonic()
                masks = _compute_masks(prev_yf)
                t_masks_total += _time.monotonic() - t_m

                # Build result for this previous tile
                t2 = _time.monotonic()
                _build_tile_result(prev_entry, masks, results)
                t_post_total += _time.monotonic() - t2

            # Wait for current GPU to finish
            t_g = _time.monotonic()
            yf = gpu_future.result()
            t_gpu_total += _time.monotonic() - t_g

            prev_yf = yf
            prev_entry = entry

        # Process final tile's masks (no next GPU to overlap with)
        if prev_yf is not None:
            t_m = _time.monotonic()
            masks = _compute_masks(prev_yf)
            t_masks_total += _time.monotonic() - t_m
            t2 = _time.monotonic()
            _build_tile_result(prev_entry, masks, results)
            t_post_total += _time.monotonic() - t2

        print(f"  [Batch worker] {len(tile_batch)} tiles: "
              f"read={t_read_total:.1f}s gpu={t_gpu_total:.1f}s "
              f"masks={t_masks_total:.1f}s post={t_post_total:.1f}s")

    finally:
        # Drain prefetch futures
        for _, fut in prefetch_q:
            try:
                fut.result()
            except Exception:
                pass
        io_pool.shutdown(wait=False)
        torch.cuda.empty_cache()

    # Close shared memory handle (don't unlink — parent owns it)
    if _worker_shm is not None:
        _worker_shm.close()

    return results


# ============================================================================
# IoU-BASED MERGE PAIR DETECTION
# ============================================================================

def _compute_iou_merge_pairs(
    tile_labels: dict,
    tiles: list[dict],
    n_tiles_y: int,
    n_tiles_x: int,
    overlap: int,
    iou_threshold: float,
    label_offsets: dict,
) -> list[tuple[int, int]]:
    """
    Compute merge pairs based on IoU in overlap zones.

    For each pair of adjacent tiles, extract the overlap region and compute
    IoU between all label pairs. Add pairs with IoU > threshold to merge list.

    Args:
        tile_labels: Dict mapping (ty, tx) to label array for that tile
        tiles: List of tile info dicts
        n_tiles_y: Number of tiles in Y
        n_tiles_x: Number of tiles in X
        overlap: Overlap size in pixels
        iou_threshold: IoU threshold for merging
        label_offsets: Dict mapping (ty, tx) to label offset for that tile

    Returns:
        List of (label_a, label_b) pairs to merge
    """
    merge_pairs = []

    # Build tile lookup
    tile_lookup = {(t["ty"], t["tx"]): t for t in tiles}

    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            tile_info = tile_lookup.get((ty, tx))
            if tile_info is None:
                continue

            labels_a = tile_labels.get((ty, tx))
            if labels_a is None or labels_a.max() == 0:
                continue

            offset_a = label_offsets.get((ty, tx), 0)

            # Check RIGHT neighbor
            if tx + 1 < n_tiles_x:
                labels_b = tile_labels.get((ty, tx + 1))
                if labels_b is not None and labels_b.max() > 0:
                    offset_b = label_offsets.get((ty, tx + 1), 0)

                    # Extract overlap regions
                    # Tile A: rightmost 'overlap' columns
                    # Tile B: leftmost 'overlap' columns
                    overlap_a = labels_a[:, -overlap:]
                    overlap_b = labels_b[:, :overlap]

                    # Find IoU-based merge pairs
                    pairs = _find_iou_pairs(overlap_a, overlap_b, offset_a, offset_b, iou_threshold)
                    merge_pairs.extend(pairs)

            # Check BOTTOM neighbor
            if ty + 1 < n_tiles_y:
                labels_b = tile_labels.get((ty + 1, tx))
                if labels_b is not None and labels_b.max() > 0:
                    offset_b = label_offsets.get((ty + 1, tx), 0)

                    # Extract overlap regions
                    # Tile A: bottom 'overlap' rows
                    # Tile B: top 'overlap' rows
                    overlap_a = labels_a[-overlap:, :]
                    overlap_b = labels_b[:overlap, :]

                    # Find IoU-based merge pairs
                    pairs = _find_iou_pairs(overlap_a, overlap_b, offset_a, offset_b, iou_threshold)
                    merge_pairs.extend(pairs)

    return merge_pairs


def _compute_iou_merge_pairs_from_cache(
    overlap_cache: dict,
    n_tiles_y: int,
    n_tiles_x: int,
    iou_threshold: float,
) -> list[tuple[int, int]]:
    """
    Compute merge pairs based on IoU using cached overlap edges.

    Memory-efficient version that uses pre-extracted overlap edges
    instead of full tile arrays. Parallelized across CPU threads —
    each boundary IoU is independent and torch.sparse.mm releases the GIL.

    Args:
        overlap_cache: Dict mapping (ty, tx) to {"right", "bottom", "left", "top"} overlap arrays
        n_tiles_y: Number of tiles in Y
        n_tiles_x: Number of tiles in X
        iou_threshold: IoU threshold for merging

    Returns:
        List of (label_a, label_b) pairs to merge
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Collect all boundary work items: (overlap_a, overlap_b) pairs
    boundary_tasks = []
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            cache_a = overlap_cache.get((ty, tx))
            if cache_a is None:
                continue

            # Right neighbor
            if tx + 1 < n_tiles_x:
                cache_b = overlap_cache.get((ty, tx + 1))
                if cache_b is not None:
                    overlap_a = cache_a.get("right")
                    overlap_b = cache_b.get("left")
                    if overlap_a is not None and overlap_b is not None:
                        if overlap_a.max() > 0 and overlap_b.max() > 0:
                            boundary_tasks.append((overlap_a, overlap_b))

            # Bottom neighbor
            if ty + 1 < n_tiles_y:
                cache_b = overlap_cache.get((ty + 1, tx))
                if cache_b is not None:
                    overlap_a = cache_a.get("bottom")
                    overlap_b = cache_b.get("top")
                    if overlap_a is not None and overlap_b is not None:
                        if overlap_a.max() > 0 and overlap_b.max() > 0:
                            boundary_tasks.append((overlap_a, overlap_b))

    n_boundaries = len(boundary_tasks)
    n_workers = min(32, n_boundaries)
    print(f"    {n_boundaries} boundary pairs, {n_workers} threads")

    # Parallel IoU computation — torch.sparse.mm releases GIL
    merge_pairs = []
    completed = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(_find_iou_pairs_with_offsets, ov_a, ov_b, iou_threshold)
            for ov_a, ov_b in boundary_tasks
        ]
        for future in as_completed(futures):
            pairs = future.result()
            if pairs:
                merge_pairs.extend(pairs)
            completed += 1
            if completed % 100 == 0 or completed == n_boundaries:
                print(f"    IoU progress: {completed}/{n_boundaries}", flush=True)

    return merge_pairs

def _find_iou_pairs_with_offsets(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    iou_threshold: float,
) -> list[tuple[int, int]]:
    """
    Find label pairs with IoU > threshold between two overlap regions.

    Uses scipy.sparse instead of torch — scipy's sparse matmul calls into
    C/BLAS and releases the GIL for the entire computation, enabling true
    parallelism with ThreadPoolExecutor. The torch version held the GIL
    for most operations (tensor creation, unique, nonzero, etc.), limiting
    32-thread parallelism to only ~4x speedup.
    """
    from scipy import sparse as sp

    # Skip if either region is empty
    if labels_a.max() == 0 or labels_b.max() == 0:
        return []

    flat_a = labels_a.ravel()
    flat_b = labels_b.ravel()
    n_pixels = len(flat_a)

    # Find non-background pixels (both must be non-zero for intersection)
    mask_a = flat_a > 0
    mask_b = flat_b > 0

    # Get unique non-zero labels and remap to 0-based indices
    unique_a = np.unique(flat_a[mask_a])
    unique_b = np.unique(flat_b[mask_b])
    if len(unique_a) == 0 or len(unique_b) == 0:
        return []

    # Build label → index mappings
    idx_map_a = np.zeros(unique_a.max() + 1, dtype=np.int32)
    idx_map_a[unique_a] = np.arange(len(unique_a))
    idx_map_b = np.zeros(unique_b.max() + 1, dtype=np.int32)
    idx_map_b[unique_b] = np.arange(len(unique_b))

    # Build sparse one-hot matrices (CSR format — fast for row slicing and matmul)
    # onehot_a: (n_labels_a, n_pixels), onehot_b: (n_labels_b, n_pixels)
    pixels_a = np.where(mask_a)[0]
    rows_a = idx_map_a[flat_a[pixels_a]]
    onehot_a = sp.csr_matrix(
        (np.ones(len(pixels_a), dtype=np.float32), (rows_a, pixels_a)),
        shape=(len(unique_a), n_pixels),
    )

    pixels_b = np.where(mask_b)[0]
    rows_b = idx_map_b[flat_b[pixels_b]]
    onehot_b = sp.csc_matrix(
        (np.ones(len(pixels_b), dtype=np.float32), (rows_b, pixels_b)),
        shape=(len(unique_b), n_pixels),
    )

    # Intersection matrix: (n_labels_a, n_labels_b) — sparse matmul releases GIL
    intersection = (onehot_a @ onehot_b.T).toarray()

    # Area of each label
    area_a = np.array(onehot_a.sum(axis=1)).ravel()
    area_b = np.array(onehot_b.sum(axis=1)).ravel()

    # IoU = intersection / (area_a + area_b - intersection)
    union = area_a[:, None] + area_b[None, :] - intersection
    # Avoid division by zero
    union = np.maximum(union, 1e-8)
    iou_matrix = intersection / union

    # Find pairs above threshold
    rows, cols = np.where(iou_matrix > iou_threshold)

    pairs = []
    for r, c in zip(rows, cols):
        label_a = int(unique_a[r])
        label_b = int(unique_b[c])
        if label_a != label_b:
            pairs.append((label_a, label_b))

    return pairs


def _find_iou_pairs(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    offset_a: int,
    offset_b: int,
    iou_threshold: float,
) -> list[tuple[int, int]]:
    """
    Find label pairs with IoU > threshold between two overlap regions.

    Uses fast sparse IoU computation from segmentation_utils.
    """
    pairs = []

    # Skip if either region is empty
    if labels_a.max() == 0 or labels_b.max() == 0:
        return pairs

    # Convert to torch tensors
    tensor_a = torch.tensor(labels_a.astype(np.int64))
    tensor_b = torch.tensor(labels_b.astype(np.int64))

    # Get sparse one-hot encodings
    onehot_a, unique_a = torch_sparse_onehot(tensor_a, flatten=True)
    onehot_b, unique_b = torch_sparse_onehot(tensor_b, flatten=True)

    # Skip background (label 0)
    if unique_a.min() == 0:
        unique_a = unique_a[unique_a > 0]
    if unique_b.min() == 0:
        unique_b = unique_b[unique_b > 0]

    if len(unique_a) == 0 or len(unique_b) == 0:
        return pairs

    # Compute IoU matrix
    iou_matrix = fast_sparse_dual_iou(onehot_a, onehot_b)

    # Find pairs above threshold
    matches = torch.nonzero(iou_matrix > iou_threshold)

    for match in matches:
        idx_a, idx_b = match[0].item(), match[1].item()
        # Convert back to global labels with offsets
        label_a = int(unique_a[idx_a].item()) + offset_a
        label_b = int(unique_b[idx_b].item()) + offset_b

        if label_a != label_b:
            pairs.append((label_a, label_b))

    return pairs


# ============================================================================
# UNION-FIND MERGING
# ============================================================================

def _build_pyramids_from_canvas(
    canvas: np.ndarray,
    source_path: Path,
    position: str,
    label_name: str,
    n_levels: int = 5,
    chunks: tuple = (1, 1, 1, 512, 512),
    shards_ratio: tuple = (1, 1, 1, 32, 32),
) -> None:
    """Build pyramid levels directly from in-memory canvas, skipping zarr re-load.

    For discrete label data, downsampling is stride-based (preserves label IDs).
    Writes each pyramid level to zarr with proper sharding.
    """
    import time as _time
    from ops_utils.io.zarr_utils import (
        create_zarr_array,
        detect_zarr_format,
        write_zarr_slice_direct,
    )
    from concurrent.futures import ThreadPoolExecutor

    zarr_format = detect_zarr_format(source_path)
    height, width = canvas.shape

    # Build all downsampled levels in memory
    t_ds = _time.time()
    levels_data = {}
    for lvl in range(1, n_levels):
        factor = 2 ** lvl
        yl = max(1, height // factor)
        xl = max(1, width // factor)
        levels_data[lvl] = canvas[::factor, ::factor].astype(np.int32)
    print(f"    Downsample ({n_levels - 1} levels) took {_time.time() - t_ds:.1f}s")

    # Create zarr arrays for each level and write in parallel
    t_write = _time.time()
    for lvl, data in levels_data.items():
        yl, xl = data.shape
        lvl_shape = (1, 1, 1, yl, xl)
        lvl_chunks = (1, 1, 1, min(chunks[-2], yl), min(chunks[-1], xl))
        lvl_path = source_path / position / "labels" / label_name / str(lvl)
        create_zarr_array(
            path=str(lvl_path),
            shape=lvl_shape,
            chunks=lvl_chunks,
            dtype=np.int32,
            zarr_format=zarr_format,
            shards_ratio=shards_ratio,
            fill_value=0,
            overwrite=True,
        )

    # Write all levels in parallel threads
    def _write_level(lvl):
        t0 = _time.time()
        component = f"{position}/labels/{label_name}/{lvl}"
        # Add Z=1 dim: (Y, X) -> (1, Y, X) so write_zarr_slice_direct
        # adds t,c to get (1, 1, 1, Y, X) matching the 5D zarr array
        data_3d = levels_data[lvl][np.newaxis, :, :]
        write_zarr_slice_direct(
            store_path=source_path,
            component_path=component,
            data=data_3d,
            t=0,
            c=0,
        )
        mb = levels_data[lvl].nbytes / 1024 / 1024
        elapsed = _time.time() - t0
        print(f"    Level {lvl}: {elapsed:.1f}s ({mb:.1f} MB)")

    with ThreadPoolExecutor(max_workers=n_levels - 1) as pool:
        list(pool.map(_write_level, range(1, n_levels)))
    print(f"    Pyramid write took {_time.time() - t_write:.1f}s")


def _apply_union_find_merges(
    canvas: np.ndarray,
    merge_pairs: list[tuple[int, int]],
    max_label: int,
) -> np.ndarray:
    """
    Apply Union-Find merging to relabel the canvas.

    Vectorized: uses numpy arrays for parent/LUT instead of Python loops.
    At 44.9M labels, Python loops took ~1 min; vectorized takes ~2-3s.
    """
    import time as _time

    if not merge_pairs or max_label == 0:
        return canvas

    t0 = _time.time()

    # Union-Find with path compression on numpy array
    parent = np.arange(max_label + 1, dtype=np.int64)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        # Path compression
        while parent[x] != root:
            next_x = parent[x]
            parent[x] = root
            x = next_x
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    # Process merge pairs (311K pairs — fast even in Python)
    for a, b in merge_pairs:
        if a <= max_label and b <= max_label:
            union(a, b)
    t_union = _time.time() - t0

    # Vectorized root resolution: chase parent pointers until convergence
    # Instead of calling find() 44M times in Python, iterate the entire array
    t1 = _time.time()
    lut = parent.copy()
    while True:
        next_lut = lut[lut]
        if np.array_equal(next_lut, lut):
            break
        lut = next_lut
    t_lut = _time.time() - t1

    # Compact remap over labels ACTUALLY PRESENT in the canvas (not the full
    # 1..max_label parent array). The per-tile label-block offsets
    # (idx*MAX_LABELS_PER_TILE) leave the 1..max_label ID space ~99% empty, so
    # remapping over np.unique(lut[1:]) scattered the real labels across ~44M
    # and left max_label ~44M despite only ~n_unique real nuclei. Composing the
    # union-find root LUT with a compaction over present roots yields contiguous
    # ids 1..N (max ≈ nucleus count) — clearer, and avoids the 44M-range
    # float32/find_objects hazards downstream.
    t2 = _time.time()
    present_labels = np.unique(canvas)                    # label ids used by any pixel
    present_roots = np.unique(lut[present_labels])        # their union-find roots (incl. 0)
    n_unique = int((present_roots > 0).sum())
    print(f"  Union-Find: {max_label} labels merged to {n_unique} unique labels (compacted 1..{n_unique})")

    # compact[root] -> contiguous id; present_roots is sorted so 0 (background) -> 0
    compact = np.zeros(max_label + 1, dtype=np.int32)
    compact[present_roots] = np.arange(len(present_roots), dtype=np.int32)
    composed = compact[lut]                               # original label -> compact id
    t_remap = _time.time() - t2

    # Apply composed LUT to canvas (single vectorized pass)
    t3 = _time.time()
    result = composed[canvas]
    t_apply = _time.time() - t3

    print(f"    UF timing: union={t_union:.1f}s lut={t_lut:.1f}s remap={t_remap:.1f}s apply={t_apply:.1f}s")

    return result


# ============================================================================
# METADATA BUILDING
# ============================================================================

def _build_cell_seg_metadata(
    label_name: str,
    channel_names: list,
    experiment: str,
    n_cells: int,
    tile_size: int,
    tile_overlap: int,
    diameter: float,
    flow_threshold: float,
    iou_threshold: float,
) -> dict:
    """
    Build comprehensive metadata for cell segmentation labels.

    Matches the format used by convert_v3_metadata.py for consistency
    across all segmentation labels in the zarr stores.

    Args:
        label_name: Label name (e.g., "cell_seg")
        channel_names: List of channel names from the zarr store
        experiment: Experiment name
        n_cells: Number of cells segmented
        tile_size: Tile size used for processing
        tile_overlap: Overlap between tiles
        diameter: Cellpose diameter parameter
        flow_threshold: Cellpose flow threshold
        iou_threshold: IoU threshold for merging

    Returns:
        Dictionary with comprehensive segmentation metadata
    """
    # Determine source channel (membrane_prediction for cell seg)
    source_channel = "membrane_prediction"
    channel_index = -1
    if source_channel in channel_names:
        channel_index = channel_names.index(source_channel)

    metadata = {
        # Core identification
        "label_name": label_name,
        "annotation_type": "cell_segmentation",
        "is_ome_label": True,

        # Source channel information
        "source_channel": {
            "name": source_channel,
            "index": channel_index,
            "type": "virtual_stain",
            "all_channels": channel_names,
        },

        # Biological annotation
        "biological_annotation": {
            "organelle": "cell_membrane",
            "marker": "virtual stain",
            "marker_type": "virtual_stain",
            "full_label": "cell membrane, virtual stain",
        },

        # Segmentation method
        "segmentation": {
            "method": "cellpose-sam",
            "version": "cell_seg-v1",
            "stitching": "hybrid_iou",
            "parameters": {
                "diameter": diameter,
                "flow_threshold": flow_threshold,
                "iou_threshold": iou_threshold,
                "tile_size": tile_size,
                "tile_overlap": tile_overlap,
            },
        },

        # Statistics
        "statistics": {
            "n_cells": n_cells,
        },

        # Human-readable description
        "description": (
            f"Cell segmentation from membrane virtual stain using Cellpose-SAM "
            f"with hybrid IoU-based stitching (IoU > {iou_threshold})"
        ),
    }

    return metadata


# ============================================================================
# MAIN SEGMENTATION FUNCTION
# ============================================================================

@versioned_function("v1.0")
def segment_single_position(
    experiment: str,
    position: str,
    tile_size: int = DEFAULT_TILE_SIZE,
    tile_overlap: int = DEFAULT_OVERLAP,
    diameter: float = DEFAULT_DIAMETER,
    flow_threshold: float = DEFAULT_FLOW_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    nuclei_channel_name: str = DEFAULT_NUCLEI_CHANNEL,
    membrane_channel_name: str = DEFAULT_MEMBRANE_CHANNEL,
    use_clahe: bool = True,
    clahe_params: dict = None,
    debug_only: bool = False,
    preview_full: bool = False,
    output_label_name: str = "cell_seg",
    use_parallel: bool = True,
    store_override: str = None,
) -> dict:
    """
    Segment a single position using hybrid IoU-based stitching.

    Args:
        experiment: Experiment name (e.g., "ops0033_20250429")
        position: Position path (e.g., "A/1/0")
        tile_size: Size of each tile (default: 2048)
        tile_overlap: Overlap between tiles (default: 256)
        diameter: Cellpose diameter (default: 100)
        flow_threshold: Cellpose flow threshold (default: 0.7)
        iou_threshold: IoU threshold for merging (default: 0.1)
        nuclei_channel_name: Name of nuclei channel for CLAHE preprocessing
            (default: "nuclei_prediction"). Used with membrane for max projection.
        membrane_channel_name: Name of membrane channel for CLAHE preprocessing
            (default: "membrane_prediction"). Primary channel for segmentation.
        use_clahe: Whether to apply CLAHE preprocessing
        clahe_params: CLAHE parameters
        debug_only: If True, run preview mode without writing to zarr (in-memory only)
        preview_full: If True, run FULL pipeline (zarr write, reshard, pyramids) on
            2x2 center tiles using production tile_size. Writes to 'cell_seg_preview'
            label to avoid affecting real data. Use this to validate end-to-end
            pipeline before running on full position.
        output_label_name: Name for output label group (default: "cell_seg")
        use_parallel: If True, use Dask parallelization with GPU workers.
            If False, process tiles sequentially (useful for debugging).

    Returns:
        Dict with results including:
        - success: bool
        - n_cells: int
        - elapsed_time: float
        - debug_images: dict (if debug_only=True)
    """
    start_time = time.time()
    result = {
        "success": False,
        "n_cells": 0,
        "elapsed_time": 0,
        "experiment": experiment,
        "position": position,
    }

    if clahe_params is None:
        clahe_params = {"clip_limit": 0.01}

    # Resolve paths
    dataset = OpsDataset(experiment)
    if store_override:
        source_path = Path(store_override)
    else:
        source_path = dataset.store_paths.get("pheno_assembled_v3")

    if source_path is None or not source_path.exists():
        result["error"] = f"v3 store not found for {experiment}: {source_path}"
        return result

    # Get channel indices from metadata (plate-level, can't use layout="fov")
    try:
        channel_names, nuclei_idx, membrane_idx = _get_channel_indices(
            source_path,
            nuclei_channel=nuclei_channel_name,
            membrane_channel=membrane_channel_name,
        )
        print(f"Channel mapping:")
        print(f"  Available: {channel_names}")
        print(f"  Nuclei: '{nuclei_channel_name}' -> index {nuclei_idx}")
        print(f"  Membrane: '{membrane_channel_name}' -> index {membrane_idx}")
    except ValueError as e:
        result["error"] = str(e)
        return result

    # Handle preview_full mode: use default preview label name if not customized
    if preview_full:
        # Only override if using the default "cell_seg" label name
        # This allows --preview-full-cp-sweep to use custom label names
        if output_label_name == "cell_seg":
            output_label_name = PREVIEW_FULL_LABEL_NAME
        print(f"\n{'='*60}")
        print(f"PREVIEW-FULL MODE: End-to-end pipeline validation")
        print(f"{'='*60}")
        print(f"  This runs the FULL pipeline (zarr write, reshard, pyramids)")
        print(f"  on a 2x2 grid of center tiles using production settings.")
        print(f"  Output label: '{output_label_name}' (separate from real data)")

    # Delete existing label groups if they exist (ensures fresh start)
    # This is important when re-running with --force to avoid stale data
    # Skip deletion in debug_only mode (no zarr writes)
    if not debug_only:
        import zarr
        store = zarr.open(str(source_path), mode="r+")
        labels_group = store.get(f"{position}/labels", None)

        # Delete both the final label and any leftover temp/unstitched label
        labels_to_delete = [
            output_label_name,
            f"{output_label_name}_unstitched",  # Temp label from interrupted runs
        ]

        for label_name in labels_to_delete:
            if labels_group is not None and label_name in labels_group:
                print(f"\n*** Found existing label group: {position}/labels/{label_name}")
                try:
                    del store[f"{position}/labels/{label_name}"]
                    print(f"*** Deleted existing label group - starting fresh ***\n")
                except KeyError:
                    print(f"*** Warning: Could not delete label group (may not exist) ***\n")

    print(f"\n{'='*60}")
    print(f"Cell Segmentation: {experiment} / {position}")
    print(f"{'='*60}")
    print(f"  Source: {source_path}")
    print(f"  CLAHE preprocessing: {use_clahe}")
    print(f"  Tile size: {tile_size}, Overlap: {tile_overlap}")
    print(f"  Cellpose: d={diameter}, ft={flow_threshold}")
    print(f"  IoU threshold: {iou_threshold}")
    if debug_only:
        print(f"  Mode: preview (in-memory only, no zarr write)")
    elif preview_full:
        print(f"  Mode: preview-full (full pipeline on 2x2 center tiles)")
    else:
        print(f"  Mode: full (all tiles)")

    # Setup GPU parallelization
    # Preview mode always uses sequential processing for debugging
    # Preview-full mode CAN use parallel processing to test the real pipeline
    if debug_only:
        use_parallel = False

    available_gpus = []
    num_workers = 1
    if use_parallel:
        available_gpus = _setup_gpu_environment()

        # VRAM-adaptive worker scaling:
        #   2 workers/GPU for ≤100GB (A40, A100-40, A100-80, H100-80)
        #   3 workers/GPU for >100GB (H200-140GB)
        # Each worker uses ~25GB VRAM (model + activations for 4096×4096 tiles).
        # 3 workers on H200: 75GB / 140GB = 54% VRAM, plenty of headroom.
        # 3 workers was 5.8% SLOWER on H100-80GB (context contention at 80GB),
        # but H200 has 75% more VRAM + 43% more memory bandwidth (4.8 vs 3.35 TB/s).
        # IMPORTANT: Use nvidia-smi instead of torch.cuda to avoid initializing
        # CUDA in the parent process. Parent CUDA init poisons child processes —
        # all Dask workers end up on GPU 0 regardless of CUDA_VISIBLE_DEVICES.
        import subprocess as _sp
        _nvsmi = _sp.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        total_vram_gb = int(_nvsmi.stdout.strip().split("\n")[0]) / 1024
        # Auto-scale: 3 workers on GPUs with ≥48GB VRAM (H200/H100/A100-80/A40).
        # Each worker uses ~12-13GB VRAM with batch_size=64 → 3 workers = ~37GB.
        # Benchmarked: 3w on H200 (140GB) = 99% SM; 3w on H100 (80GB) = no regression vs 2w.
        # GPUs <48GB (A100-40): 2 workers (3 would use ~37GB, too tight with 40GB).
        # SEG_WORKERS_PER_GPU env var overrides for A/B testing.
        env_workers = int(os.environ.get("SEG_WORKERS_PER_GPU", 0))
        if env_workers > 0:
            workers_per_gpu = env_workers
        elif total_vram_gb >= 48:
            workers_per_gpu = 3
        else:
            workers_per_gpu = 2
        num_workers = workers_per_gpu * len(available_gpus)
        print(f"  Parallel mode: {num_workers} total workers "
              f"({workers_per_gpu}/GPU, VRAM={total_vram_gb:.0f}GB), "
              f"GPUs: {available_gpus}")
    else:
        print(f"  Sequential mode (single GPU)")

    # Load model only if not using parallel mode (workers load their own)
    model = None
    if not use_parallel:
        model = _create_cellpose_model(model_type="cyto3", gpu=True)

    # Open source zarr and get image dimensions
    pos_path = source_path / position
    if not pos_path.exists():
        result["error"] = f"Position {position} not found in store"
        return result

    with open_ome_zarr(pos_path, layout="fov", mode="r") as ds:
        data_shape = ds["0"].shape  # (T, C, Z, Y, X)
        height, width = data_shape[-2], data_shape[-1]

    print(f"  Image size: {height} x {width}")

    # Use smaller tiles for preview mode (in-memory debug)
    if debug_only:
        tile_size = PREVIEW_TILE_SIZE
        tile_overlap = PREVIEW_OVERLAP
        # Center crop for preview
        crop_size = PREVIEW_N_TILES * (tile_size - tile_overlap) + tile_overlap
        y_start = max(0, (height - crop_size) // 2)
        x_start = max(0, (width - crop_size) // 2)
        height = min(crop_size, height - y_start)
        width = min(crop_size, width - x_start)
        print(f"  Preview: {PREVIEW_N_TILES}x{PREVIEW_N_TILES} tiles, center crop at ({y_start}, {x_start})")
    # Preview-full mode: use production tile size but only 2x2 tiles
    # Use ~25% offset from origin (not center) to better test internal boundaries
    elif preview_full:
        # Calculate crop for 2x2 tiles at production size
        n_preview_tiles = PREVIEW_FULL_N_TILES
        crop_size = n_preview_tiles * (tile_size - tile_overlap) + tile_overlap
        # Use 25% offset instead of center - ensures we test real internal boundaries
        y_start = max(0, int(height * 0.25))
        x_start = max(0, int(width * 0.25))
        # Ensure we don't go past the image bounds
        y_start = min(y_start, height - crop_size) if height > crop_size else 0
        x_start = min(x_start, width - crop_size) if width > crop_size else 0
        height = min(crop_size, height - y_start)
        width = min(crop_size, width - x_start)
        print(f"  Preview-full: {n_preview_tiles}x{n_preview_tiles} tiles @ {tile_size}px, 25% offset at ({y_start}, {x_start})")
        print(f"  Crop region: {height} x {width} pixels")
    else:
        y_start, x_start = 0, 0

    # Calculate tile grid
    tiles, n_tiles_y, n_tiles_x = _calculate_tile_grid(height, width, tile_size, tile_overlap)
    print(f"  Tile grid: {n_tiles_y} x {n_tiles_x} = {len(tiles)} tiles")

    # Cap num_workers to number of tiles (no point having more workers than tiles)
    if use_parallel and num_workers > len(tiles):
        print(f"  Capping workers from {num_workers} to {len(tiles)} (num tiles)")
        num_workers = len(tiles)

    # =========================================================================
    # Initialize temp zarr for memory-efficient processing
    # =========================================================================
    # Get full image shape for label array
    with open_ome_zarr(source_path / position, layout="fov", mode="r") as ds:
        full_shape = ds["0"].shape  # (T, C, Z, Y, X)

    label_shape = (1, 1, full_shape[2], full_shape[3], full_shape[4])
    temp_label_name = f"{output_label_name}_unstitched"

    if not debug_only:
        print(f"\n  Initializing temp label array: {temp_label_name}")
        # Write directly to final sharded format — the canvas is written as a single
        # sequential operation after Union-Find, so shard lock contention is not an issue.
        # This eliminates the resharding step (~5 min at full scale).
        _init_organelle_label_array(
            zarr_path=source_path,
            pos_path=position,
            organelle_name=temp_label_name,
            shape=label_shape,
            dtype=np.int32,
            chunks=(1, 1, 1, 512, 512),
            shards_ratio=(1, 1, 1, 32, 32),  # ~16K×16K shards, final format
        )

    # =========================================================================
    # CACHE LOAD: Try to load Pass 1 results if caching is enabled
    # =========================================================================
    # =========================================================================
    # PASS 1: Segment each tile
    # =========================================================================
    # Memory-efficient approach:
    # - Stream tiles to in-memory canvas as they complete (saves ~14 GB)
    # - Cache overlap edges for IoU computation in Pass 2 (~14 GB)
    print(f"\n  Pass 1: Segmenting {len(tiles)} tiles...")
    _t_pass1_start = time.time()

    label_offsets = {}  # (ty, tx) -> offset for global label IDs
    running_offset = 0
    debug_images = {"raw_tiles": [], "preprocessed_tiles": [], "seg_tiles": []}
    # Only keep tile_labels for debug mode (small number of tiles)
    tile_labels = {} if debug_only else None
    # Cache overlap edges for IoU computation (all 4 edges per tile)
    # Memory: ~14 GB for 900 tiles (4 edges × 512×4096 × 4 bytes × ~900)
    overlap_cache = {}  # (ty, tx) -> {"right", "bottom", "left", "top"}

    # Create canvas in shared memory so Dask workers can write labels directly,
    # eliminating ~56GB of Dask serialization (64MB × 900 tiles).
    # Workers return only overlap strips (~32MB/tile) through Dask.
    _shm = None
    _shm_name = None
    _t_canvas = time.time()
    canvas_gb = height * width * 4 / 1e9
    if not debug_only:
        import multiprocessing.shared_memory as _shm_mod
        canvas_nbytes = height * width * 4
        _shm = _shm_mod.SharedMemory(create=True, size=canvas_nbytes)
        _shm_name = _shm.name
        canvas = np.ndarray((height, width), dtype=np.int32, buffer=_shm.buf)
        canvas[:] = 0
        print(f"  Canvas allocated in shared memory: {height}×{width} int32 "
              f"({canvas_gb:.1f} GB, shm={_shm_name}) in {time.time()-_t_canvas:.1f}s")
    else:
        canvas = None

    # Build tile lookup for streaming writes
    tiles_by_coord = {(t["ty"], t["tx"]): t for t in tiles}

    if use_parallel and len(tiles) > 1:
        # =================================================================
        # PARALLEL TILE SEGMENTATION (Dask + GPU workers)
        # Memory-efficient: writes cores to zarr, returns only overlap edges
        # =================================================================
        # Pre-estimate label offsets for each tile
        # We estimate based on typical label density, will be corrected after segmentation
        # For now, use a large offset per tile to ensure uniqueness
        MAX_LABELS_PER_TILE = 50000  # Conservative estimate for 4096x4096 tiles
        for idx, tile_info in enumerate(tiles):
            ty, tx = tile_info["ty"], tile_info["tx"]
            label_offsets[(ty, tx)] = idx * MAX_LABELS_PER_TILE

        tile_results = []

        # Save and clear parent CUDA_VISIBLE_DEVICES (MultiGPUCluster sets per-cluster)
        parent_cuda_devices = os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        # Create one LocalCluster per GPU with CUDA_VISIBLE_DEVICES set before
        # worker spawn (see MultiGPUCluster docstring for why this is needed).
        with MultiGPUCluster(available_gpus, workers_per_gpu) as multi_cluster:
            print(f"  Dashboard: {multi_cluster.dashboard_link}")

            # Build tile entries with metadata for batch workers
            tile_entries = []
            for idx, tile_info in enumerate(tiles):
                ty, tx = tile_info["ty"], tile_info["tx"]
                tile_entries.append({
                    "tile_idx": idx,
                    "tile_info": tile_info,
                    "label_offset": label_offsets[(ty, tx)],
                })

            # Shuffle tiles before batching to balance cell density across workers.
            # Without shuffle, tiles are in row-major order so edge workers get
            # sparse tiles while interior workers get dense tiles — up to 3.7 min
            # gpu spread + 3.4 min masks spread on 2×H200 with 900 tiles.
            # Deterministic seed for reproducibility.
            import random as _random
            _rng = _random.Random(42)
            _rng.shuffle(tile_entries)

            # Split tiles into exactly n_total_workers batches (no runt batch).
            # Each worker runs a prefetch pipeline internally,
            # overlapping read+CLAHE with GPU inference.
            n_total_workers = workers_per_gpu * len(available_gpus)
            n_batches = min(n_total_workers, len(tile_entries))
            batches = []
            for b in range(n_batches):
                start = b * len(tile_entries) // n_batches
                end = (b + 1) * len(tile_entries) // n_batches
                batches.append(tile_entries[start:end])
            batch_size = len(batches[0]) if batches else 0
            print(f"  Submitting {len(batches)} batches "
                  f"({batch_size} tiles/batch, {len(tiles)} total)")

            # Submit batch workers (round-robin across GPUs)
            futures = []
            for batch in batches:
                future = multi_cluster.submit(
                    _segment_tiles_batch_worker,
                    tile_batch=batch,
                    source_path=str(source_path),
                    position=position,
                    y_offset=y_start,
                    x_offset=x_start,
                    diameter=diameter,
                    flow_threshold=flow_threshold,
                    nuclei_channel=nuclei_idx,
                    membrane_channel=membrane_idx,
                    use_clahe=use_clahe,
                    clahe_params=clahe_params,
                    temp_label_name=temp_label_name,
                    tile_overlap=tile_overlap,
                    n_tiles_y=n_tiles_y,
                    n_tiles_x=n_tiles_x,
                    canvas_shm_name=_shm_name,
                    canvas_height=height,
                    canvas_width=width,
                )
                futures.append(future)

            # Collect results: deserialize all 6 batches in parallel threads,
            # then process sequentially. Each batch is ~9.5GB — sequential
            # deserialization takes ~47s × 6 = 280s. Parallel cuts to ~50s.
            from concurrent.futures import ThreadPoolExecutor as _GatherPool

            _t_gather_start = time.time()

            # Wait for all Dask futures to complete (workers are still computing)
            print("  Waiting for workers to complete...")
            from distributed import wait as _dask_wait
            _dask_wait(futures)
            _t_workers_done = time.time()
            print(f"  Workers done in {_t_workers_done - _t_gather_start:.1f}s")

            # Deserialize all batches in parallel threads
            print("  Deserializing results (parallel)...")
            _t_deser_start = time.time()
            def _fetch_result(future):
                return future.result()
            with _GatherPool(max_workers=len(futures)) as gather_pool:
                all_batch_results = list(gather_pool.map(_fetch_result, futures))
            _t_deser = time.time() - _t_deser_start
            print(f"  Deserialized {len(all_batch_results)} batches in {_t_deser:.1f}s")

            # Process results: labels already in shared memory canvas,
            # only collect overlap cache from Dask results.
            _t_proc_start = time.time()
            n_tiles_done = 0
            for batch_results in all_batch_results:
                for tile_result in batch_results:
                    ty, tx = tile_result["ty"], tile_result["tx"]
                    n_labels = tile_result["n_labels"]

                    running_offset = max(running_offset, label_offsets[(ty, tx)] + n_labels)

                    overlap_cache[(ty, tx)] = {
                        "right": tile_result.get("right_overlap"),
                        "bottom": tile_result.get("bottom_overlap"),
                        "left": tile_result.get("left_overlap"),
                        "top": tile_result.get("top_overlap"),
                    }
                    n_tiles_done += 1
            _t_proc = time.time() - _t_proc_start
            _t_gather_total = time.time() - _t_gather_start
            print(f"  Canvas writes: {_t_proc:.1f}s")
            print(f"  Result gathering total: {_t_gather_total:.1f}s "
                  f"(workers={_t_workers_done - _t_gather_start:.1f}s, "
                  f"deser={_t_deser:.1f}s, process={_t_proc:.1f}s)")
            print(f"  Result gathering: {_t_gather_total:.1f}s total")
            _t_pre_shutdown = time.time()
        # Dask cluster has shut down (MultiGPUCluster context manager exited)
        print(f"  Dask cluster shutdown in {time.time() - _t_pre_shutdown:.1f}s")

        # Copy canvas from shared memory and clean up
        if _shm is not None:
            _t_shm_cleanup = time.time()
            canvas_copy = canvas.copy()
            _shm.close()
            _shm.unlink()
            canvas = canvas_copy
            print(f"  Shared memory cleanup: {time.time() - _t_shm_cleanup:.1f}s")

        # Restore parent CUDA_VISIBLE_DEVICES
        if parent_cuda_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = parent_cuda_devices

    else:
        # =================================================================
        # SEQUENTIAL TILE SEGMENTATION (for preview/debug or single tile)
        # In debug mode, keep full tiles in memory for visualization
        # =================================================================
        with open_ome_zarr(source_path / position, layout="fov", mode="r") as ds:
            for tile_info in tqdm(tiles, desc="  Segmenting tiles"):
                ty, tx = tile_info["ty"], tile_info["tx"]

                # Load tile data (with offset for preview mode)
                y0 = y_start + tile_info["y_start"]
                y1 = y_start + tile_info["y_end"]
                x0 = x_start + tile_info["x_start"]
                x1 = x_start + tile_info["x_end"]

                tile_data = np.asarray(ds["0"][0, :, 0, y0:y1, x0:x1])

                # Segment tile (returns both labels and preprocessed image)
                labels, preprocessed = _segment_tile(
                    tile_data,
                    model,
                    diameter=diameter,
                    flow_threshold=flow_threshold,
                    nuclei_channel=nuclei_idx,
                    membrane_channel=membrane_idx,
                    use_clahe=use_clahe,
                    clahe_params=clahe_params,
                )

                # Store with offset
                label_offsets[(ty, tx)] = running_offset
                n_labels = int(labels.max())
                labels_int32 = labels.astype(np.int32)

                if debug_only:
                    # In debug mode, keep full tiles for visualization
                    tile_labels[(ty, tx)] = labels_int32
                else:
                    # In non-debug mode, write directly to canvas (memory-efficient)
                    if n_labels > 0:
                        labels_with_global_offset = labels_int32.copy()
                        labels_with_global_offset[labels_with_global_offset > 0] += running_offset
                        mask = labels_with_global_offset > 0
                        tile_y0, tile_y1 = tile_info["y_start"], tile_info["y_end"]
                        tile_x0, tile_x1 = tile_info["x_start"], tile_info["x_end"]
                        canvas[tile_y0:tile_y1, tile_x0:tile_x1][mask] = labels_with_global_offset[mask]

                if n_labels > 0:
                    running_offset += n_labels

                # Build overlap_cache for IoU computation (non-debug mode)
                # In debug mode, we use the old _compute_iou_merge_pairs function with tile_labels
                if not debug_only:
                    labels_with_offset = labels_int32.copy()
                    if labels_with_offset.max() > 0:
                        labels_with_offset[labels_with_offset > 0] += label_offsets[(ty, tx)]

                    overlap_cache[(ty, tx)] = {
                        "right": labels_with_offset[:, -tile_overlap:].copy() if tx + 1 < n_tiles_x else None,
                        "bottom": labels_with_offset[-tile_overlap:, :].copy() if ty + 1 < n_tiles_y else None,
                        "left": labels_with_offset[:, :tile_overlap].copy() if tx > 0 else None,
                        "top": labels_with_offset[:tile_overlap, :].copy() if ty > 0 else None,
                    }

                # Store debug images (only in debug mode)
                if debug_only:
                    debug_images["raw_tiles"].append({
                        "ty": ty, "tx": tx,
                        "data": tile_data[membrane_idx] if tile_data.ndim == 3 else tile_data,
                    })
                    debug_images["preprocessed_tiles"].append({
                        "ty": ty, "tx": tx,
                        "data": preprocessed,
                    })
                    debug_images["seg_tiles"].append({
                        "ty": ty, "tx": tx,
                        "labels": labels.astype(np.int32),
                        "offset": label_offsets[(ty, tx)],
                    })

    max_label = running_offset
    _t_pass1 = time.time() - _t_pass1_start
    print(f"  Pass 1 complete: {max_label} total labels before merging ({_t_pass1:.1f}s)")

    # =========================================================================
    # PASS 2: Compute IoU-based merge pairs
    # =========================================================================
    _t_pass2_start = time.time()
    print(f"\n  Pass 2: Computing IoU-based merge pairs...")

    if debug_only:
        # In debug mode, use the old function with full tile_labels
        merge_pairs = _compute_iou_merge_pairs(
            tile_labels=tile_labels,
            tiles=tiles,
            n_tiles_y=n_tiles_y,
            n_tiles_x=n_tiles_x,
            overlap=tile_overlap,
            iou_threshold=iou_threshold,
            label_offsets=label_offsets,
        )
    else:
        # Use cached overlap edges (extracted BEFORE canvas writing)
        # This is critical: canvas has "first-writer wins" so we need original edges
        merge_pairs = _compute_iou_merge_pairs_from_cache(
            overlap_cache=overlap_cache,
            n_tiles_y=n_tiles_y,
            n_tiles_x=n_tiles_x,
            iou_threshold=iou_threshold,
        )

    _t_pass2 = time.time() - _t_pass2_start
    print(f"  Found {len(merge_pairs)} merge pairs (IoU > {iou_threshold}) ({_t_pass2:.1f}s)")

    # =========================================================================
    # PASS 3: Apply Union-Find merges + write + reshard + pyramids
    # =========================================================================
    _t_pass3_start = time.time()
    print(f"\n  Pass 3: Applying Union-Find merges...")

    if debug_only:
        # DEBUG MODE: Use canvas in memory for visualization
        canvas = np.zeros((height, width), dtype=np.int32)

        # Write tiles to canvas (with global offsets)
        for (ty, tx), labels in tile_labels.items():
            if labels.max() == 0:
                continue

            tile_info = next(t for t in tiles if t["ty"] == ty and t["tx"] == tx)
            offset = label_offsets[(ty, tx)]

            # Write to canvas (full tile region, will be overwritten by neighbors)
            y0 = tile_info["y_start"]
            y1 = tile_info["y_end"]
            x0 = tile_info["x_start"]
            x1 = tile_info["x_end"]

            # Only write non-zero pixels (don't overwrite existing labels)
            mask = labels > 0
            canvas[y0:y1, x0:x1][mask] = labels[mask] + offset

        # Apply Union-Find merges
        canvas = _apply_union_find_merges(canvas, merge_pairs, max_label)

        n_cells = int(canvas.max())
        print(f"  Final: {n_cells} cells after merging (debug mode)")

    else:
        # IN-MEMORY MODE: Canvas was already populated during tile streaming
        # This is memory-efficient: tiles were written as they completed
        # Apply Union-Find merges to canvas
        canvas = _apply_union_find_merges(canvas, merge_pairs, max_label)

        n_cells = int(canvas.max())
        print(f"  Final: {n_cells} cells after merging")

        # Write canvas to zarr in parallel shard-row strips.
        # With shards=(1,1,1,16384,16384), a ~20K×20K canvas has ~4 shards.
        # Writing each shard-row in a separate thread parallelizes lz4 compression
        # and NFS I/O across shard files, avoiding single-thread bottleneck.
        print(f"  Writing canvas to zarr (parallel shard-rows)...")
        _t_write_start = time.time()
        import zarr
        from concurrent.futures import ThreadPoolExecutor

        shard_y = 512 * 32  # 16384 — shard height from chunks(512) * shards_ratio(32)
        n_strips = max(1, (height + shard_y - 1) // shard_y)
        print(f"    {n_strips} shard-row strips of {shard_y}px each")

        def _write_strip(strip_idx):
            """Write one shard-row strip to zarr (each thread opens its own store)."""
            import zarr as _zarr
            _store = _zarr.open(str(source_path), mode="r+")
            _arr = _store[position]["labels"][temp_label_name]["0"]
            row_start = strip_idx * shard_y
            row_end = min(height, (strip_idx + 1) * shard_y)
            _arr[0, 0, 0,
                 y_start + row_start:y_start + row_end,
                 x_start:x_start + width] = canvas[row_start:row_end, :]

        with ThreadPoolExecutor(max_workers=n_strips) as pool:
            list(pool.map(_write_strip, range(n_strips)))
        _t_write = time.time() - _t_write_start
        print(f"  Canvas write took {_t_write:.1f}s ({n_strips} parallel threads)")

    # =========================================================================
    # Rename temp zarr to final
    # =========================================================================
    if not debug_only:
        import zarr
        import shutil

        print(f"\n  Renaming {temp_label_name} -> {output_label_name}...")

        # Open store and get labels group
        store = zarr.open(str(source_path), mode="r+")
        labels_group = store[position]["labels"]

        # Remove existing output label if it exists
        if output_label_name in labels_group:
            print(f"  Removing existing {output_label_name}...")
            del labels_group[output_label_name]

        # Rename by moving the group
        # Zarr v3 groups are stored as directories, so we can use filesystem rename
        temp_path = source_path / position / "labels" / temp_label_name
        final_path = source_path / position / "labels" / output_label_name

        if temp_path.exists():
            shutil.move(str(temp_path), str(final_path))
            print(f"  Renamed {temp_label_name} -> {output_label_name}")
        else:
            print(f"  WARNING: Temp path {temp_path} does not exist!")

        # Re-open store to see the renamed group
        store = zarr.open(str(source_path), mode="r+")
        labels_group = store[position]["labels"]

        # Build and write metadata
        print(f"  Writing metadata...")
        metadata = _build_cell_seg_metadata(
            label_name=output_label_name,
            channel_names=channel_names,
            experiment=experiment,
            n_cells=n_cells,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            diameter=diameter,
            flow_threshold=flow_threshold,
            iou_threshold=iou_threshold,
        )

        # Update labels group metadata (skip in preview_full mode since we'll delete it)
        if not preview_full:
            _update_labels_metadata(
                zarr_path=source_path,
                pos_path=position,
                new_label_name=output_label_name,
                metadata=metadata,
            )
            print(f"  Metadata written to labels/{output_label_name}")

            # Also remove temp label name from metadata if it was added
            existing_attrs = dict(labels_group.attrs)
            existing_labels = existing_attrs.get("labels", [])
            if temp_label_name in existing_labels:
                existing_labels.remove(temp_label_name)
                labels_group.attrs["labels"] = existing_labels
                print(f"  Removed {temp_label_name} from labels metadata")
        else:
            print(f"  Skipping metadata update in preview-full mode (label will be deleted)")

        # Build pyramids from in-memory canvas (skip 94s zarr re-load)
        if not preview_full:
            print(f"\n  Building pyramids for {output_label_name} (from memory)...")
            _t_pyr = time.time()
            _build_pyramids_from_canvas(
                canvas=canvas,
                source_path=source_path,
                position=position,
                label_name=output_label_name,
                n_levels=5,
                chunks=(1, 1, 1, 512, 512),
                shards_ratio=(1, 1, 1, 32, 32),
            )
            print(f"  Pyramids built in {time.time() - _t_pyr:.1f}s")
        else:
            print(f"\n  Skipping pyramids in preview-full mode (would process mostly zeros)")

    # Finalize result
    _t_pass3 = time.time() - _t_pass3_start
    result["success"] = True
    result["n_cells"] = n_cells
    result["elapsed_time"] = time.time() - start_time
    result["n_tiles"] = len(tiles) if 'tiles' in dir() else 0
    result["pass1_time"] = _t_pass1 if '_t_pass1' in dir() else 0
    result["pass2_time"] = _t_pass2 if '_t_pass2' in dir() else 0
    result["pass3_time"] = _t_pass3
    result["merge_pairs"] = len(merge_pairs)
    result["filled_pixels"] = filled_pixels if 'filled_pixels' in dir() else 0

    if debug_only:
        result["debug_images"] = debug_images
        result["canvas"] = canvas

        # Save preview images
        _save_preview_images(
            experiment=experiment,
            position=position,
            debug_images=debug_images,
            canvas=canvas,
            tiles=tiles,
            tile_overlap=tile_overlap,
            source_path=source_path,
        )

    # Save preview-full summary image (read back from zarr to verify write worked)
    if preview_full:
        _save_preview_full_image(
            experiment=experiment,
            position=position,
            source_path=source_path,
            output_label_name=output_label_name,
            y_start=y_start,
            x_start=x_start,
            crop_height=height,
            crop_width=width,
            nuclei_idx=nuclei_idx,
            membrane_idx=membrane_idx,
            n_cells=n_cells,
            elapsed_time=result["elapsed_time"],
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            n_tiles_y=n_tiles_y,
            n_tiles_x=n_tiles_x,
            filled_pixels=result.get("filled_pixels", 0),
        )

        # Delete the preview_seg label from zarr after saving PNG
        # This prevents it from polluting the metadata and being hard to delete later
        import shutil
        preview_label_path = source_path / position / "labels" / output_label_name
        if preview_label_path.exists():
            print(f"\n  Cleaning up preview label: {output_label_name}")
            shutil.rmtree(preview_label_path)
            print(f"  Deleted {preview_label_path}")

    print(f"\n  Completed in {result['elapsed_time']:.1f}s")
    print(f"  Phase timing: Pass1={result.get('pass1_time', 0):.1f}s, Pass2={result.get('pass2_time', 0):.1f}s, Pass3+={result.get('pass3_time', 0):.1f}s")
    print(f"{'='*60}\n")

    return result


# ============================================================================
# PREVIEW MODE VISUALIZATION
# ============================================================================

def _save_preview_images(
    experiment: str,
    position: str,
    debug_images: dict,
    canvas: np.ndarray,
    tiles: list[dict],
    tile_overlap: int,
    source_path: Path,
):
    """Save preview mode debug images."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import random

    # Create output directory
    assembly_dir = source_path.parent
    debug_dir = assembly_dir / "cell_seg_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    pos_safe = position.replace("/", "_")

    # Create random colormap for labels
    n_colors = max(256, int(canvas.max()) + 1)
    colors = [(0, 0, 0)]  # Background is black
    for _ in range(n_colors - 1):
        colors.append((random.random(), random.random(), random.random()))
    cmap = ListedColormap(colors)

    # Create comparison figure (2 rows x 4 columns)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(
        f"Cell Segmentation Preview: {experiment} / {position}\n"
        f"Hybrid IoU-based stitching (IoU > {DEFAULT_IOU_THRESHOLD})",
        fontsize=14,
        fontweight="bold",
    )

    # Row 0: Raw membrane, CLAHE preprocessed, Per-tile seg, Stitched result
    # Row 1: Overlay, Boundary viz, Zoom center, Zoom boundary

    # Assemble raw image from tiles (membrane channel only)
    raw_canvas = np.zeros_like(canvas, dtype=np.float32)
    for tile_dict in debug_images["raw_tiles"]:
        ty, tx = tile_dict["ty"], tile_dict["tx"]
        tile_info = next(t for t in tiles if t["ty"] == ty and t["tx"] == tx)
        y0, y1 = tile_info["y_start"], tile_info["y_end"]
        x0, x1 = tile_info["x_start"], tile_info["x_end"]
        raw_canvas[y0:y1, x0:x1] = tile_dict["data"][:y1-y0, :x1-x0]

    # Assemble preprocessed (CLAHE) image from tiles
    preprocessed_canvas = np.zeros_like(canvas, dtype=np.float32)
    for tile_dict in debug_images["preprocessed_tiles"]:
        ty, tx = tile_dict["ty"], tile_dict["tx"]
        tile_info = next(t for t in tiles if t["ty"] == ty and t["tx"] == tx)
        y0, y1 = tile_info["y_start"], tile_info["y_end"]
        x0, x1 = tile_info["x_start"], tile_info["x_end"]
        preprocessed_canvas[y0:y1, x0:x1] = tile_dict["data"][:y1-y0, :x1-x0]

    # Assemble per-tile segmentation (before merging)
    unmerged_canvas = np.zeros_like(canvas, dtype=np.int32)
    for tile_dict in debug_images["seg_tiles"]:
        ty, tx = tile_dict["ty"], tile_dict["tx"]
        tile_info = next(t for t in tiles if t["ty"] == ty and t["tx"] == tx)
        y0, y1 = tile_info["y_start"], tile_info["y_end"]
        x0, x1 = tile_info["x_start"], tile_info["x_end"]
        labels = tile_dict["labels"]
        offset = tile_dict["offset"]
        mask = labels > 0
        unmerged_canvas[y0:y1, x0:x1][mask] = labels[mask] + offset

    # Normalize raw canvas for display
    raw_min, raw_max = np.percentile(raw_canvas, [1, 99])
    raw_canvas_norm = np.clip((raw_canvas - raw_min) / (raw_max - raw_min + 1e-8), 0, 1)

    # Plot raw membrane image
    axes[0, 0].imshow(raw_canvas_norm, cmap="gray")
    axes[0, 0].set_title("Raw membrane channel")
    axes[0, 0].axis("off")

    # Plot CLAHE preprocessed image
    axes[0, 1].imshow(preprocessed_canvas, cmap="gray")
    axes[0, 1].set_title("CLAHE preprocessed\n(nucleus + membrane max)")
    axes[0, 1].axis("off")

    # Plot per-tile segmentation (before merging)
    axes[0, 2].imshow(unmerged_canvas, cmap=cmap, interpolation="nearest")
    axes[0, 2].set_title(f"Per-tile segmentation\n({int(unmerged_canvas.max())} labels)")
    axes[0, 2].axis("off")

    # Plot final stitched result
    axes[0, 3].imshow(canvas, cmap=cmap, interpolation="nearest")
    axes[0, 3].set_title(f"After IoU merging\n({int(canvas.max())} cells)")
    axes[0, 3].axis("off")

    # Plot overlay of preprocessed + segmentation
    axes[1, 0].imshow(preprocessed_canvas, cmap="gray")
    mask = canvas > 0
    overlay = np.zeros((*canvas.shape, 4))
    for label_id in range(1, int(canvas.max()) + 1):
        cell_mask = canvas == label_id
        color = colors[label_id % len(colors)]
        overlay[cell_mask] = (*color, 0.4)
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("Overlay on preprocessed")
    axes[1, 0].axis("off")

    # Plot tile boundaries
    axes[1, 1].imshow(canvas, cmap=cmap, interpolation="nearest")
    for tile_info in tiles:
        # Draw core region boundaries
        y0 = tile_info["core_y_start"]
        y1 = tile_info["core_y_end"]
        x0 = tile_info["core_x_start"]
        x1 = tile_info["core_x_end"]
        axes[1, 1].axhline(y=y0, color="red", linewidth=0.5, alpha=0.7)
        axes[1, 1].axhline(y=y1, color="red", linewidth=0.5, alpha=0.7)
        axes[1, 1].axvline(x=x0, color="red", linewidth=0.5, alpha=0.7)
        axes[1, 1].axvline(x=x1, color="red", linewidth=0.5, alpha=0.7)
    axes[1, 1].set_title("Tile boundaries (red)")
    axes[1, 1].axis("off")

    # Zoom on center region
    h, w = canvas.shape
    center_y, center_x = h // 2, w // 2
    zoom_size = 300
    y0 = max(0, center_y - zoom_size // 2)
    y1 = min(h, center_y + zoom_size // 2)
    x0 = max(0, center_x - zoom_size // 2)
    x1 = min(w, center_x + zoom_size // 2)
    axes[1, 2].imshow(canvas[y0:y1, x0:x1], cmap=cmap, interpolation="nearest")
    axes[1, 2].set_title(f"Zoom on center ({zoom_size}x{zoom_size})")
    axes[1, 2].axis("off")

    # Zoom showing CLAHE with segmentation overlay
    axes[1, 3].imshow(preprocessed_canvas[y0:y1, x0:x1], cmap="gray")
    # Add segmentation overlay on the zoomed region
    zoom_canvas = canvas[y0:y1, x0:x1]
    zoom_overlay = np.zeros((*zoom_canvas.shape, 4))
    for label_id in range(1, int(zoom_canvas.max()) + 1):
        cell_mask = zoom_canvas == label_id
        color = colors[label_id % len(colors)]
        zoom_overlay[cell_mask] = (*color, 0.4)
    axes[1, 3].imshow(zoom_overlay)
    axes[1, 3].set_title(f"CLAHE + overlay zoom ({zoom_size}x{zoom_size})")
    axes[1, 3].axis("off")

    plt.tight_layout()

    output_path = debug_dir / f"preview_{pos_safe}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Preview saved to: {output_path}")


def _save_preview_full_image(
    experiment: str,
    position: str,
    source_path: Path,
    output_label_name: str,
    y_start: int,
    x_start: int,
    crop_height: int,
    crop_width: int,
    nuclei_idx: int,
    membrane_idx: int,
    n_cells: int,
    elapsed_time: float,
    tile_size: int = 4096,
    tile_overlap: int = 512,
    n_tiles_y: int = 2,
    n_tiles_x: int = 2,
    filled_pixels: int = 0,
):
    """
    Save preview-full mode summary image with tile boundary visualization.

    Reads back from zarr to verify the write worked, then creates a figure
    showing the CLAHE-preprocessed input, segmentation overlay, and zoomed
    views of tile boundaries to showcase the boundary gap filling algorithm.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from skimage.exposure import equalize_adapthist
    import random

    print(f"\n  Saving preview-full summary image...")

    # Create output directory
    assembly_dir = source_path.parent
    debug_dir = assembly_dir / "cell_seg_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    pos_safe = position.replace("/", "_")

    # Read back the segmentation from zarr to verify it was written correctly
    import zarr

    # Read image channels using iohub
    with open_ome_zarr(source_path / position, layout="fov", mode="r") as ds:
        # Read BOTH nuclei and membrane channels for the cropped region
        nuclei = np.asarray(ds["0"][0, nuclei_idx, 0, y_start:y_start+crop_height, x_start:x_start+crop_width])
        membrane = np.asarray(ds["0"][0, membrane_idx, 0, y_start:y_start+crop_height, x_start:x_start+crop_width])

    # Read segmentation labels using direct zarr access
    store = zarr.open(str(source_path), mode="r")
    labels_path = f"{position}/labels/{output_label_name}/0"
    if labels_path not in store:
        print(f"  Warning: {output_label_name} not found in labels, skipping image save")
        return

    seg_arr = store[labels_path]
    segmentation = np.asarray(seg_arr[0, 0, 0, y_start:y_start+crop_height, x_start:x_start+crop_width])

    # Recreate the CLAHE preprocessing
    p1_n, p99_n = np.percentile(nuclei, [1, 99])
    nuclei_norm = np.clip((nuclei.astype(np.float32) - p1_n) / (p99_n - p1_n + 1e-8), 0, 1)

    p1_m, p99_m = np.percentile(membrane, [1, 99])
    membrane_norm = np.clip((membrane.astype(np.float32) - p1_m) / (p99_m - p1_m + 1e-8), 0, 1)

    max_proj = np.maximum(nuclei_norm, membrane_norm)
    clahe_input = equalize_adapthist(max_proj, clip_limit=0.01).astype(np.float32)

    # Create random colormap for labels
    n_labels = int(segmentation.max()) + 1
    n_colors = max(256, n_labels)
    colors = [(0, 0, 0)]  # Background is black
    random.seed(42)
    for _ in range(n_colors - 1):
        colors.append((random.random(), random.random(), random.random()))
    cmap = ListedColormap(colors)

    # Build color LUT for overlays
    color_lut = np.zeros((n_colors, 4), dtype=np.float32)
    for i, c in enumerate(colors):
        if i == 0:
            color_lut[i] = (0, 0, 0, 0)
        else:
            color_lut[i] = (*c, 0.4)

    # Calculate tile boundaries (in local crop coordinates)
    step = tile_size - tile_overlap
    h_boundaries = [ty * step for ty in range(1, n_tiles_y)]  # Horizontal boundaries
    v_boundaries = [tx * step for tx in range(1, n_tiles_x)]  # Vertical boundaries

    # Create figure (3x3 grid) - more panels to showcase boundary fix
    fig, axes = plt.subplots(3, 3, figsize=(24, 24))
    fig.suptitle(
        f"Preview-Full: {experiment} / {position}\n"
        f"{n_cells} cells | {elapsed_time:.1f}s | Boundary fill: {filled_pixels} pixels | "
        f"Tiles: {n_tiles_y}x{n_tiles_x} @ {tile_size}px",
        fontsize=14,
        fontweight="bold",
    )

    # Row 1: Input channels
    axes[0, 0].imshow(nuclei_norm, cmap="gray")
    axes[0, 0].set_title("Nuclei channel")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(membrane_norm, cmap="gray")
    axes[0, 1].set_title("Membrane channel")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(clahe_input, cmap="gray")
    axes[0, 2].set_title("CLAHE(max(nuclei, membrane))")
    axes[0, 2].axis("off")

    # Row 2: Full segmentation with tile boundaries
    # Left: Labels only
    axes[1, 0].imshow(segmentation, cmap=cmap, interpolation="nearest")
    axes[1, 0].set_title(f"Segmentation ({n_cells} cells)")
    # Draw tile boundaries
    for y_bound in h_boundaries:
        axes[1, 0].axhline(y=y_bound, color="red", linewidth=2, linestyle="--", alpha=0.8)
    for x_bound in v_boundaries:
        axes[1, 0].axvline(x=x_bound, color="red", linewidth=2, linestyle="--", alpha=0.8)
    axes[1, 0].axis("off")

    # Middle: CLAHE + overlay with tile boundaries
    axes[1, 1].imshow(clahe_input, cmap="gray")
    label_indices = (segmentation.astype(np.int64) % n_colors).astype(np.int32)
    label_indices[segmentation == 0] = 0
    overlay = color_lut[label_indices]
    axes[1, 1].imshow(overlay)
    for y_bound in h_boundaries:
        axes[1, 1].axhline(y=y_bound, color="red", linewidth=2, linestyle="--", alpha=0.8)
    for x_bound in v_boundaries:
        axes[1, 1].axvline(x=x_bound, color="red", linewidth=2, linestyle="--", alpha=0.8)
    axes[1, 1].set_title("Overlay + tile boundaries (red dashed)")
    axes[1, 1].axis("off")

    # Right: Center zoom
    h, w = segmentation.shape
    center_y, center_x = h // 2, w // 2
    zoom_size = min(1024, h // 2, w // 2)
    y0 = max(0, center_y - zoom_size // 2)
    y1 = min(h, center_y + zoom_size // 2)
    x0 = max(0, center_x - zoom_size // 2)
    x1 = min(w, center_x + zoom_size // 2)

    axes[1, 2].imshow(clahe_input[y0:y1, x0:x1], cmap="gray")
    zoom_seg = segmentation[y0:y1, x0:x1]
    zoom_indices = (zoom_seg.astype(np.int64) % n_colors).astype(np.int32)
    zoom_indices[zoom_seg == 0] = 0
    axes[1, 2].imshow(color_lut[zoom_indices])
    # Draw boundaries if they fall within zoom region
    for y_bound in h_boundaries:
        if y0 < y_bound < y1:
            axes[1, 2].axhline(y=y_bound - y0, color="red", linewidth=3, linestyle="--")
    for x_bound in v_boundaries:
        if x0 < x_bound < x1:
            axes[1, 2].axvline(x=x_bound - x0, color="red", linewidth=3, linestyle="--")
    axes[1, 2].set_title(f"Center zoom ({zoom_size}px)")
    axes[1, 2].axis("off")

    # Row 3: Boundary close-ups to showcase the fix
    # Show zoomed views at each tile boundary intersection

    boundary_zoom_size = 256  # Small zoom to see boundary details

    # Panel 3,0: Horizontal boundary zoom (if exists)
    if h_boundaries:
        h_bound = h_boundaries[0]  # First horizontal boundary
        # Center the zoom on the boundary
        by0 = max(0, h_bound - boundary_zoom_size // 2)
        by1 = min(h, h_bound + boundary_zoom_size // 2)
        bx0 = max(0, w // 2 - boundary_zoom_size // 2)
        bx1 = min(w, w // 2 + boundary_zoom_size // 2)

        axes[2, 0].imshow(clahe_input[by0:by1, bx0:bx1], cmap="gray")
        bound_seg = segmentation[by0:by1, bx0:bx1]
        bound_indices = (bound_seg.astype(np.int64) % n_colors).astype(np.int32)
        bound_indices[bound_seg == 0] = 0
        axes[2, 0].imshow(color_lut[bound_indices])
        # Draw the exact boundary line
        axes[2, 0].axhline(y=h_bound - by0, color="yellow", linewidth=2, label="Tile boundary")
        axes[2, 0].set_title(f"H-Boundary @ y={h_bound}\n(step={step}px)")
    else:
        axes[2, 0].text(0.5, 0.5, "No horizontal\nboundaries", ha="center", va="center", fontsize=12)
    axes[2, 0].axis("off")

    # Panel 3,1: Vertical boundary zoom (if exists)
    if v_boundaries:
        v_bound = v_boundaries[0]  # First vertical boundary
        by0 = max(0, h // 2 - boundary_zoom_size // 2)
        by1 = min(h, h // 2 + boundary_zoom_size // 2)
        bx0 = max(0, v_bound - boundary_zoom_size // 2)
        bx1 = min(w, v_bound + boundary_zoom_size // 2)

        axes[2, 1].imshow(clahe_input[by0:by1, bx0:bx1], cmap="gray")
        bound_seg = segmentation[by0:by1, bx0:bx1]
        bound_indices = (bound_seg.astype(np.int64) % n_colors).astype(np.int32)
        bound_indices[bound_seg == 0] = 0
        axes[2, 1].imshow(color_lut[bound_indices])
        axes[2, 1].axvline(x=v_bound - bx0, color="yellow", linewidth=2, label="Tile boundary")
        axes[2, 1].set_title(f"V-Boundary @ x={v_bound}\n(step={step}px)")
    else:
        axes[2, 1].text(0.5, 0.5, "No vertical\nboundaries", ha="center", va="center", fontsize=12)
    axes[2, 1].axis("off")

    # Panel 3,2: Boundary intersection (corner between 4 tiles)
    if h_boundaries and v_boundaries:
        h_bound = h_boundaries[0]
        v_bound = v_boundaries[0]
        by0 = max(0, h_bound - boundary_zoom_size // 2)
        by1 = min(h, h_bound + boundary_zoom_size // 2)
        bx0 = max(0, v_bound - boundary_zoom_size // 2)
        bx1 = min(w, v_bound + boundary_zoom_size // 2)

        axes[2, 2].imshow(clahe_input[by0:by1, bx0:bx1], cmap="gray")
        bound_seg = segmentation[by0:by1, bx0:bx1]
        bound_indices = (bound_seg.astype(np.int64) % n_colors).astype(np.int32)
        bound_indices[bound_seg == 0] = 0
        axes[2, 2].imshow(color_lut[bound_indices])
        axes[2, 2].axhline(y=h_bound - by0, color="yellow", linewidth=2)
        axes[2, 2].axvline(x=v_bound - bx0, color="yellow", linewidth=2)
        axes[2, 2].set_title(f"Boundary intersection\n({filled_pixels} pixels filled)")
    else:
        axes[2, 2].text(0.5, 0.5, "No boundary\nintersection", ha="center", va="center", fontsize=12)
    axes[2, 2].axis("off")

    plt.tight_layout()

    output_path = debug_dir / f"preview_full_{output_label_name}_{pos_safe}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Preview-full image saved to: {output_path}")
    print(f"  Zarr label path: {source_path}/{position}/labels/{output_label_name}")
    print(f"  Tile boundaries at: H={h_boundaries}, V={v_boundaries}")


# ============================================================================
# CELL PAINTING SWEEP
# ============================================================================

def run_cp_sweep_preview(
    experiment: str,
    position: str,
    tile_size: int = DEFAULT_TILE_SIZE,
    tile_overlap: int = DEFAULT_OVERLAP,
    diameter: float = DEFAULT_DIAMETER,
    flow_threshold: float = DEFAULT_FLOW_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    use_clahe: bool = True,
    use_parallel: bool = True,
    configs: list = None,
    label_prefix: str = "cp_sweep",
    sweep_name: str = "CELL PAINTING CHANNEL SWEEP",
    store_override: str = None,
) -> dict:
    """
    Run preview-full segmentation with multiple channel combinations.

    Iterates through the given sweep configs (defaults to CP_SWEEP_CONFIGS) and runs
    segment_single_position for each, saving results to separate label groups and
    generating a comparison summary.

    Returns:
        dict with keys: success, results (list of per-config results), summary_path
    """
    import matplotlib.pyplot as plt
    from ops_utils.data.experiment import OpsDataset

    if configs is None:
        configs = CP_SWEEP_CONFIGS

    results = []

    print("=" * 70)
    print(sweep_name)
    print("=" * 70)
    print(f"Experiment: {experiment}")
    print(f"Position: {position}")
    print(f"Configurations to test: {len(configs)}")
    for name, nuc, mem in configs:
        print(f"  - {name}: nuclei={nuc}, membrane={mem}")
    print("=" * 70)

    for i, (config_name, nuclei_ch, membrane_ch) in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Running: {config_name}")
        print(f"  Nuclei channel: {nuclei_ch}")
        print(f"  Membrane channel: {membrane_ch}")

        # Use unique label name for each config
        output_label = f"{label_prefix}_{config_name}"

        result = segment_single_position(
            experiment=experiment,
            position=position,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            diameter=diameter,
            flow_threshold=flow_threshold,
            iou_threshold=iou_threshold,
            nuclei_channel_name=nuclei_ch,
            membrane_channel_name=membrane_ch,
            use_clahe=use_clahe,
            debug_only=False,
            preview_full=True,
            output_label_name=output_label,
            use_parallel=use_parallel,
            store_override=store_override,
        )

        result["config_name"] = config_name
        result["nuclei_channel"] = nuclei_ch
        result["membrane_channel"] = membrane_ch
        results.append(result)

        if result["success"]:
            print(f"  ✓ {config_name}: {result['n_cells']} cells in {result['elapsed_time']:.1f}s")
        else:
            print(f"  ✗ {config_name}: FAILED - {result.get('error', 'Unknown error')}")

    # Generate summary comparison
    print("\n" + "=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)

    # Find debug output directory
    dataset = OpsDataset(experiment)
    debug_dir = dataset.experiment_path / "3-assembly" / "cell_seg_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Create summary table
    print(f"\n{'Config':<20} {'Cells':>8} {'Time (s)':>10} {'Status':<10}")
    print("-" * 50)
    for r in results:
        status = "OK" if r["success"] else "FAILED"
        n_cells = r.get("n_cells", 0) if r["success"] else "-"
        elapsed = f"{r.get('elapsed_time', 0):.1f}" if r["success"] else "-"
        print(f"{r['config_name']:<20} {str(n_cells):>8} {elapsed:>10} {status:<10}")

    # Create comparison figure: grid of per-config preview PNGs
    successful = [r for r in results if r["success"]]
    if len(successful) >= 1:
        from PIL import Image
        import math

        pos_safe = position.replace("/", "_")
        panels = []
        for r in successful:
            cfg = r["config_name"]
            preview_png = debug_dir / f"preview_full_{label_prefix}_{cfg}_{pos_safe}.png"
            if not preview_png.exists():
                # Fallback name (without label_prefix in path) used by older runs
                preview_png = debug_dir / f"preview_full_{cfg}_{pos_safe}.png"
            if preview_png.exists():
                panels.append((cfg, r, Image.open(preview_png)))
            else:
                print(f"  WARNING: preview PNG not found for {cfg}: {preview_png}")

        if panels:
            ncols = min(3, len(panels))
            nrows = math.ceil(len(panels) / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
            if nrows == 1 and ncols == 1:
                axes = np.array([[axes]])
            elif nrows == 1 or ncols == 1:
                axes = np.array(axes).reshape(nrows, ncols)

            for i, (cfg, r, img) in enumerate(panels):
                ax = axes[i // ncols, i % ncols]
                ax.imshow(img)
                ax.set_title(f"{cfg}\n{r['n_cells']} cells, {r.get('elapsed_time', 0):.1f}s",
                             fontsize=11)
                ax.axis("off")

            # Hide unused axes
            for i in range(len(panels), nrows * ncols):
                axes[i // ncols, i % ncols].axis("off")

            plt.suptitle(f"{sweep_name}: {experiment} / {position}", fontsize=14)
            plt.tight_layout()

            summary_path = debug_dir / f"{label_prefix}_summary_{position.replace('/', '_')}.png"
            plt.savefig(summary_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"\nSummary figure saved to: {summary_path}")
        else:
            summary_path = None
            print("\nNo preview PNGs found for summary figure.")
    else:
        summary_path = None
        print("\nNo successful runs for summary figure.")

    return {
        "success": any(r["success"] for r in results),
        "results": results,
        "summary_path": summary_path,
    }


# ============================================================================
# CLI
# ============================================================================

def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Cell segmentation with hybrid IoU-based stitching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--experiment", "-e",
        type=str,
        required=True,
        help="Experiment name (e.g., ops0033_20250429)",
    )

    parser.add_argument(
        "--position", "-p",
        type=str,
        default="A/1/0",
        help="Position path (default: A/1/0)",
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help="Run preview mode (2x2 small tiles, in-memory only, saves debug images)",
    )

    parser.add_argument(
        "--preview-full",
        action="store_true",
        help="Run FULL pipeline on 2x2 center tiles at production size. "
             "Tests end-to-end: zarr write, resharding, pyramids. "
             "Writes to 'cell_seg_preview' label (separate from real data).",
    )

    parser.add_argument(
        "--preview-full-cp-sweep",
        action="store_true",
        help="Run --preview-full multiple times with different cell painting channel combinations: "
             "actin_only, actin+nuclei, actin+WGA, actin+tubulin, actin+conA. "
             "Saves comparison images and summary to debug_output folder.",
    )

    parser.add_argument(
        "--preview-full-4i-sweep",
        action="store_true",
        help="Run --preview-full multiple times with different 4i channel combinations "
             "(DAPI-only + every antibody channel across rounds 1-5). "
             "Saves comparison images and summary to debug_output folder.",
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        help=f"Tile size in pixels (default: {DEFAULT_TILE_SIZE})",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help=f"Tile overlap in pixels (default: {DEFAULT_OVERLAP})",
    )

    parser.add_argument(
        "--diameter",
        type=float,
        default=DEFAULT_DIAMETER,
        help=f"Cellpose diameter (default: {DEFAULT_DIAMETER})",
    )

    parser.add_argument(
        "--flow-threshold",
        type=float,
        default=DEFAULT_FLOW_THRESHOLD,
        help=f"Cellpose flow threshold (default: {DEFAULT_FLOW_THRESHOLD})",
    )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help=f"IoU threshold for merging (default: {DEFAULT_IOU_THRESHOLD})",
    )

    parser.add_argument(
        "--nuclei-channel",
        type=str,
        default=DEFAULT_NUCLEI_CHANNEL,
        help=f"Nuclei channel name for CLAHE preprocessing (default: {DEFAULT_NUCLEI_CHANNEL}). "
             "Combined with membrane via max projection before CLAHE.",
    )

    parser.add_argument(
        "--membrane-channel",
        type=str,
        default=DEFAULT_MEMBRANE_CHANNEL,
        help=f"Membrane channel name for CLAHE preprocessing (default: {DEFAULT_MEMBRANE_CHANNEL}). "
             "Primary channel for cell boundary detection.",
    )

    parser.add_argument(
        "--no-clahe",
        action="store_true",
        help="Disable CLAHE preprocessing (use percentile normalization only)",
    )

    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Disable parallel processing (process tiles sequentially on single GPU)",
    )

    # Cell painting mode
    parser.add_argument(
        "--cell-paint",
        action="store_true",
        help=(
            "Cell painting mode: segment cells using actin channel (CP1_f_actin_Phalloidin) "
            "instead of virtual staining membrane_prediction. Output stored to 'cp_cell_seg' label."
        ),
    )

    parser.add_argument(
        "--output-label",
        type=str,
        default=None,
        help=(
            "Custom output label name (default: 'cell_seg', or 'cp_cell_seg' with --cell-paint)."
        ),
    )

    # ------------------------------------------------------------------
    # Native-20x nuclei pass — runs nuclei_pass.segment_nuclei_single_position
    # AFTER cell seg completes. Reuses the same source store and tile grid;
    # replaces the legacy 5x segment_and_stitch_pheno step.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--also-segment-nuclei",
        action="store_true",
        help=(
            "After cell seg completes, run native-20x nuclei segmentation on the "
            "nuclei_prediction channel and write in place to the canonical "
            "'nuclear_seg' label. Replaces the legacy 5x segment_and_stitch_pheno step."
        ),
    )
    parser.add_argument(
        "--nuclei-diameter",
        type=float,
        default=150.0,
        help="Cellpose diameter for the 20x nuclei pass (default 150 — H100 bench sweet spot).",
    )
    parser.add_argument(
        "--nuclei-label-name",
        type=str,
        default="nuclear_seg",
        help="Output label group name for the 20x nuclei pass.",
    )

    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Validate mutually exclusive options
    preview_modes = [args.preview, args.preview_full, args.preview_full_cp_sweep, args.preview_full_4i_sweep]
    if sum(preview_modes) > 1:
        print("Error: --preview, --preview-full, --preview-full-cp-sweep, and --preview-full-4i-sweep are mutually exclusive.")
        print("  --preview: In-memory debug mode (2x2 small tiles)")
        print("  --preview-full: Full pipeline test (2x2 production tiles)")
        print("  --preview-full-cp-sweep: Test multiple cell painting channel combinations")
        print("  --preview-full-4i-sweep: Test multiple 4i channel combinations")
        return 1

    # Resolve experiment name
    from ops_utils.data.filesystem import resolve_experiment_name
    experiment = resolve_experiment_name(args.experiment, allow_interactive=True)
    if experiment is None:
        print("No experiment found. Exiting.")
        return 1

    # Handle cell painting sweep mode
    if args.preview_full_cp_sweep:
        result = run_cp_sweep_preview(
            experiment=experiment,
            position=args.position,
            tile_size=args.tile_size,
            tile_overlap=args.overlap,
            diameter=args.diameter,
            flow_threshold=args.flow_threshold,
            iou_threshold=args.iou_threshold,
            use_clahe=not args.no_clahe,
            use_parallel=not args.sequential,
        )
        return 0 if result["success"] else 1

    # Handle 4i sweep mode
    if args.preview_full_4i_sweep:
        result = run_cp_sweep_preview(
            experiment=experiment,
            position=args.position,
            tile_size=args.tile_size,
            tile_overlap=args.overlap,
            diameter=args.diameter,
            flow_threshold=args.flow_threshold,
            iou_threshold=args.iou_threshold,
            use_clahe=not args.no_clahe,
            use_parallel=not args.sequential,
            configs=FOUR_I_SWEEP_CONFIGS,
            label_prefix="4i_sweep",
            sweep_name="4i CHANNEL SWEEP",
        )
        return 0 if result["success"] else 1

    # Handle cell painting mode
    if args.cell_paint:
        nuclei_channel = CP_NUCLEI_CHANNEL
        membrane_channel = CP_MEMBRANE_CHANNEL
        output_label = args.output_label if args.output_label else CP_OUTPUT_LABEL
        print(f"\nCell Painting Mode:")
        print(f"  Channel: {membrane_channel} (actin)")
        print(f"  Output label: {output_label}\n")
    else:
        nuclei_channel = args.nuclei_channel
        membrane_channel = args.membrane_channel
        output_label = args.output_label if args.output_label else "cell_seg"

    result = segment_single_position(
        experiment=experiment,
        position=args.position,
        tile_size=args.tile_size,
        tile_overlap=args.overlap,
        diameter=args.diameter,
        flow_threshold=args.flow_threshold,
        iou_threshold=args.iou_threshold,
        nuclei_channel_name=nuclei_channel,
        membrane_channel_name=membrane_channel,
        use_clahe=not args.no_clahe,
        debug_only=args.preview,
        preview_full=args.preview_full,
        use_parallel=not args.sequential,
        output_label_name=output_label,
    )

    if result["success"]:
        print(f"Success: {result['n_cells']} cells in {result['elapsed_time']:.1f}s")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
        return 1

    # Optional fused 20x nuclei pass — sequential after cell seg.
    if getattr(args, "also_segment_nuclei", False):
        from cyclops_process.processes.cell_seg.nuclei_pass import (
            segment_nuclei_single_position,
        )
        nuc_result = segment_nuclei_single_position(
            experiment=experiment,
            position=args.position,
            tile_size=args.tile_size,
            tile_overlap=args.overlap,
            diameter=args.nuclei_diameter,
            flow_threshold=args.flow_threshold,
            iou_threshold=args.iou_threshold,
            nuclei_channel_name=args.nuclei_channel,
            output_label_name=args.nuclei_label_name,
            use_parallel=not args.sequential,
            preview_full=args.preview_full,
        )
        if nuc_result["success"]:
            print(
                f"Nuclei pass: {nuc_result['n_nuclei']} nuclei in "
                f"{nuc_result['elapsed_time']:.1f}s"
            )
        else:
            print(f"Nuclei pass error: {nuc_result.get('error', 'Unknown')}")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
