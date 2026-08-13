"""
Native-20x Nuclei Segmentation
==============================

Sibling of cell_segmentation.py — segments the `nuclei_prediction` channel
of phenotyping_v3.zarr at native 20x using Cellpose-SAM with diameter=150
and the same hybrid IoU stitching used for cells.

Motivation
----------
The legacy `segment_and_stitch_pheno` step ran nuclei segmentation on a
4×-downsampled version of the 20x VS (effective "5x"). The decision to
downsample dated to when segmentation was substantially slower; modern
cell-seg infrastructure (PRs #80/#82/#87) makes native 20x cheaper per
physical area than the 5x pass:

  * +33-50% boundary fidelity (mean |∇nuclei_prediction| at the mask edge,
    measured across ops0094 and ops0042; 92% of paired nuclei have a
    higher-fidelity boundary in NEW)
  * ~70% lower Pass-1 GPU cost per 20x-tile-area (8.7 s → 2.6 s on H100)
  * Topology agreement vs PROD 5x: ±1.1% count, 99% one-to-one centroid
    pairing both ways, ~2% over-seg, ~2-3% under-seg (balanced)
  * Native 20x mask = no 4× nearest-neighbor staircase

Reuses cell_segmentation helpers (tile grid, IoU merge, union-find,
pyramid build, metadata writer, shared-memory canvas) via lazy import.
This file is the per-position entry point; `nuclei_segmentation_orchestrator.py`
fans it out across positions.

Usage
-----
    # Single position (debug or preview):
    python -m cyclops_process.processes.cell_seg.nuclei_pass \\
        --experiment ops0094_20251217 --position A/1/0 --preview-full

    # Batch via orchestrator (DAG runner):
    python run.py --slurm-steps --dag --rerun submit_nuclei_segmentation_jobs

    # Direct batch CLI:
    python -m cyclops_process.processes.cell_seg.nuclei_segmentation_orchestrator \\
        --experiment ops0094_20251217
"""

from __future__ import annotations
from ops_utils.profiling.decorators import versioned_function

# ---------------------------------------------------------------------------
# CRITICAL (main-process only): hide GPUs from the parent BEFORE any cupy /
# torch import. Even bare `import cupy` calls `cudaGetDeviceCount()` which
# initializes the CUDA driver context — once initialized, the context is
# bound to a specific GPU and inherited via fork() by every Dask Nanny
# worker. All workers then end up on the same physical GPU regardless of
# the per-Nanny CUDA_VISIBLE_DEVICES env override. By popping CVD at module
# top, the parent's cupy import sees zero GPUs → driver stays uninitialized
# → workers fork from a clean parent and bind to their assigned GPU when
# they first touch CUDA in their own process.
#
# We MUST gate this to the main process. Dask Nanny workers also import
# this module at spawn time, and (a) they're daemonic so they can't spawn
# children for the forkserver pre-warm, and (b) we don't want to pop THEIR
# CVD — they need it to bind to their assigned GPU.
# See diagnostic 34395297 — without this gate, workers crashed with
# `AssertionError: daemonic processes are not allowed to have children`.
import multiprocessing as _mp
import os as _os_early


def _forkserver_noop():
    return None


_PARENT_CUDA_VISIBLE_DEVICES = None
if _mp.current_process().name == "MainProcess":
    _PARENT_CUDA_VISIBLE_DEVICES = _os_early.environ.pop("CUDA_VISIBLE_DEVICES", None)
    try:
        _mp.set_start_method("forkserver", force=True)
        _p = _mp.Process(target=_forkserver_noop)
        _p.start()
        _p.join()
    except (RuntimeError, AssertionError):
        # RuntimeError: start method already set. AssertionError: extra safety
        # if some unexpected daemon-context case reaches here.
        pass

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch  # lazy — does not init CUDA on bare import
from iohub import open_ome_zarr

# IMPORTANT: do NOT import from cell_segmentation at module top.
# cell_segmentation → segment.py → cupy/stitch.tile → CUDA init in PARENT.
# Once the parent's CUDA context is bound to GPU 0, Dask Nanny workers
# fork()'d from it inherit that binding and ignore CUDA_VISIBLE_DEVICES.
# Imports happen lazily inside segment_nuclei_single_position() AFTER the
# MultiGPUCluster has spawned its workers (so workers fork from a clean
# parent and each binds to its assigned GPU).
# See diagnostic 34394059 — every worker had a different
# CUDA_VISIBLE_DEVICES but the same physical GPU UUID via nvidia-smi.

from ops_utils.data.experiment import OpsDataset
from ops_utils.hpc.gpu_utils import _setup_gpu_environment
from ops_utils.hpc.parallel_utils import MultiGPUCluster
from ops_utils.io.zarr_labels import (
    _init_organelle_label_array,
    _update_labels_metadata,
)


# Constants — duplicated here rather than imported from cell_segmentation
# (which triggers the cupy/stitch.tile chain — see note above).
DEFAULT_TILE_SIZE = 4096
DEFAULT_OVERLAP = 512
DEFAULT_FLOW_THRESHOLD = 0.7
DEFAULT_IOU_THRESHOLD = 0.1
DEFAULT_NUCLEI_CHANNEL = "nuclei_prediction"

# Sweet spot from H100 sweep (scratch/nuclear_seg/bench_nuclei_pass.py).
# Wall plateaus 150-240; mask count starts drifting past 180.
DEFAULT_NUCLEI_DIAMETER = 150.0
# Written in place; pyramid level 0 = 20x, level 2 = 5x.
DEFAULT_NUCLEI_LABEL = "nuclear_seg"


# =============================================================================
# Inlined helpers (copied from cell_segmentation.py) — avoid importing
# cell_segmentation in the parent process before MultiGPUCluster.fork().
# =============================================================================

def _find_channel_index(channel_names: list[str], target_name: str) -> int:
    target = target_name.lower()
    for i, name in enumerate(channel_names):
        if target in str(name).lower():
            return i
    return -1


def _get_channel_indices_local(
    store_path: Path,
    nuclei_channel: str = DEFAULT_NUCLEI_CHANNEL,
) -> tuple[list[str], int]:
    """Inlined channel index lookup — only needs the nuclei channel."""
    with open_ome_zarr(str(store_path), mode="r") as ds:
        channel_names = list(ds.channel_names)
    nuc_idx = _find_channel_index(channel_names, nuclei_channel)
    if nuc_idx < 0:
        raise ValueError(
            f"Nuclei channel '{nuclei_channel}' not found. Available: {channel_names}"
        )
    return channel_names, nuc_idx


def _calculate_tile_grid_local(
    height: int,
    width: int,
    tile_size: int,
    overlap: int,
):
    """Inlined tile-grid calculator — same algorithm as cell_segmentation."""
    step = tile_size - overlap
    n_tiles_y = max(1, (height - overlap + step - 1) // step)
    n_tiles_x = max(1, (width - overlap + step - 1) // step)
    tiles = []
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            y_start = ty * step
            x_start = tx * step
            y_end = min(y_start + tile_size, height)
            x_end = min(x_start + tile_size, width)
            tiles.append({
                "ty": ty, "tx": tx,
                "y_start": y_start, "y_end": y_end,
                "x_start": x_start, "x_end": x_end,
                "core_y_start": y_start, "core_x_start": x_start,
                "core_y_end": min((ty + 1) * step, height),
                "core_x_end": min((tx + 1) * step, width),
            })
    return tiles, n_tiles_y, n_tiles_x


# =============================================================================
# Tile worker (single-channel Cellpose nuclei, no CLAHE)
# =============================================================================

def _segment_nuclei_tiles_batch_worker(
    tile_batch: list,
    source_path: str,
    position: str,
    y_offset: int,
    x_offset: int,
    diameter: float,
    flow_threshold: float,
    nuclei_channel: int,
    tile_overlap: int,
    n_tiles_y: int,
    n_tiles_x: int,
    canvas_shm_name: str,
    canvas_height: int,
    canvas_width: int,
) -> list:
    """Per-tile nuclei segmentation worker.

    Strategy mirrors `_segment_tiles_batch_worker` in cell_segmentation.py but
    is simpler: single-channel raw nuclei_prediction, no CLAHE, plain
    `model.eval(diameter=...)` call (no hand-rolled run_net split — the
    benchmark showed it isn't needed at the diameters we use).

    Returns per-tile overlap edges for Pass 2 IoU merge. Labels are written
    directly to the shared-memory canvas (zero-copy, first-writer-wins).
    """
    import zarr as _zarr
    import time as _time
    from concurrent.futures import ThreadPoolExecutor
    from collections import deque

    source_path = Path(source_path)

    # Match cell-seg worker thread caps
    torch.set_num_threads(8)
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"

    # Attach to shared memory canvas (parent owns it)
    import multiprocessing.shared_memory as _shm_mod
    _worker_shm = _shm_mod.SharedMemory(name=canvas_shm_name)
    _worker_canvas = np.ndarray(
        (canvas_height, canvas_width), dtype=np.int32, buffer=_worker_shm.buf)

    # Lazy import — done inside the worker, AFTER the worker's
    # CUDA_VISIBLE_DEVICES is set, so any CUDA touch during cell_segmentation
    # module load (cupy import, stitch.tile's xp.array probe, etc.) binds to
    # this worker's assigned GPU, not the parent's.
    from cyclops_process.processes.cell_seg.cell_segmentation import (
        _get_cached_model as _local_get_cached_model,
    )

    # Log which physical GPU each worker actually bound to. Distinct UUIDs
    # across workers confirm the multi-GPU plumbing is working (see header
    # comment about CUDA_VISIBLE_DEVICES handling). nvidia-smi --query-gpu
    # ignores CVD (uses NVML), so we use torch.cuda which respects CVD.
    try:
        _name = torch.cuda.get_device_name(0)
        _dev = torch.cuda.current_device()
        try:
            _uuid = str(torch.cuda.get_device_properties(0).uuid)
        except Exception:
            _uuid = "<no uuid attr>"
    except Exception as e:
        _name, _dev, _uuid = f"<err {e}>", -1, "<err>"
    print(
        f"  [nuclei worker] PID={os.getpid()}  "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '?')}  "
        f"torch.cuda: current={_dev}  name='{_name}'  uuid={_uuid}"
    )

    model = _local_get_cached_model("nuclei", gpu=True)

    PREFETCH_DEPTH = 2  # nuclei pass is light — small prefetch is sufficient
    io_pool = ThreadPoolExecutor(
        max_workers=PREFETCH_DEPTH + 1, thread_name_prefix="nuc_io"
    )

    def _read_tile(tile_entry):
        """Read raw nuclei channel for one tile."""
        tile_info = tile_entry["tile_info"]
        source_arr = _zarr.open(str(source_path / position / "0"), mode="r")
        y0 = y_offset + tile_info["y_start"]
        y1 = y_offset + tile_info["y_end"]
        x0 = x_offset + tile_info["x_start"]
        x1 = x_offset + tile_info["x_end"]
        nuc = np.asarray(source_arr[0, nuclei_channel, 0, y0:y1, x0:x1]).astype(
            np.float32, copy=False
        )
        return nuc

    def _build_result(entry, masks, results):
        tile_info = entry["tile_info"]
        ty, tx = tile_info["ty"], tile_info["tx"]
        label_offset = entry["label_offset"]
        n_labels = int(masks.max())

        labels_offset = masks.astype(np.int32)
        if labels_offset.max() > 0:
            labels_offset[labels_offset > 0] += label_offset

        if labels_offset.max() > 0:
            y0 = tile_info["y_start"]
            y1 = tile_info["y_end"]
            x0 = tile_info["x_start"]
            x1 = tile_info["x_end"]
            fg = labels_offset > 0
            _worker_canvas[y0:y1, x0:x1][fg] = labels_offset[fg]

        right_overlap = bottom_overlap = left_overlap = top_overlap = None
        if tx + 1 < n_tiles_x and masks.shape[1] >= tile_overlap:
            right_overlap = labels_offset[:, -tile_overlap:].copy()
        if ty + 1 < n_tiles_y and masks.shape[0] >= tile_overlap:
            bottom_overlap = labels_offset[-tile_overlap:, :].copy()
        if tx > 0 and masks.shape[1] >= tile_overlap:
            left_overlap = labels_offset[:, :tile_overlap].copy()
        if ty > 0 and masks.shape[0] >= tile_overlap:
            top_overlap = labels_offset[:tile_overlap, :].copy()

        results.append({
            "tile_idx": entry["tile_idx"],
            "ty": ty,
            "tx": tx,
            "n_labels": n_labels,
            "right_overlap": right_overlap,
            "bottom_overlap": bottom_overlap,
            "left_overlap": left_overlap,
            "top_overlap": top_overlap,
        })

    # Prime prefetch
    prefetch_q = deque()
    prime_count = min(PREFETCH_DEPTH, len(tile_batch))
    for k in range(prime_count):
        prefetch_q.append((tile_batch[k], io_pool.submit(_read_tile, tile_batch[k])))
    next_idx = prime_count

    results = []
    t_io = 0.0
    t_gpu = 0.0
    try:
        for i in range(len(tile_batch)):
            t0 = _time.monotonic()
            entry, fut = prefetch_q.popleft()
            nuc = fut.result()
            t_io += _time.monotonic() - t0

            if next_idx < len(tile_batch):
                prefetch_q.append(
                    (tile_batch[next_idx], io_pool.submit(_read_tile, tile_batch[next_idx]))
                )
                next_idx += 1

            t1 = _time.monotonic()
            masks, _, _ = model.eval(nuc, diameter=diameter, flow_threshold=flow_threshold)
            t_gpu += _time.monotonic() - t1

            _build_result(entry, masks, results)
        print(
            f"  [Nuclei batch worker] {len(tile_batch)} tiles: "
            f"io={t_io:.1f}s gpu={t_gpu:.1f}s"
        )
    finally:
        for _, fut in prefetch_q:
            try:
                fut.result()
            except Exception:
                pass
        io_pool.shutdown(wait=False)
        torch.cuda.empty_cache()
        _worker_shm.close()

    return results


# =============================================================================
# Metadata
# =============================================================================

def _build_nuclei_seg_metadata(
    label_name: str,
    channel_names: list,
    experiment: str,
    n_nuclei: int,
    tile_size: int,
    tile_overlap: int,
    diameter: float,
    flow_threshold: float,
    iou_threshold: float,
) -> dict:
    """Mirror of _build_cell_seg_metadata for nuclear masks."""
    source_channel = "nuclei_prediction"
    channel_index = channel_names.index(source_channel) if source_channel in channel_names else -1
    return {
        "label_name": label_name,
        "annotation_type": "nuclei_segmentation",
        "is_ome_label": True,
        "source_channel": {
            "name": source_channel,
            "index": channel_index,
            "type": "virtual_stain",
            "all_channels": channel_names,
        },
        "biological_annotation": {
            "organelle": "nucleus",
            "marker": "virtual stain",
            "marker_type": "virtual_stain",
            "full_label": "nucleus, virtual stain",
        },
        "segmentation": {
            "method": "cellpose-sam",
            "version": "nuclear_seg-v1-20x",
            "stitching": "hybrid_iou",
            "parameters": {
                "diameter": diameter,
                "flow_threshold": flow_threshold,
                "iou_threshold": iou_threshold,
                "tile_size": tile_size,
                "tile_overlap": tile_overlap,
            },
        },
        "statistics": {"n_nuclei": n_nuclei},
        "description": (
            f"Nuclei segmentation from nuclei_prediction virtual stain at 20x "
            f"using Cellpose-SAM with hybrid IoU stitching (IoU > {iou_threshold})."
        ),
    }


# =============================================================================
# Orchestrator
# =============================================================================

@versioned_function("v1.0")
def segment_nuclei_single_position(
    experiment: str,
    position: str,
    tile_size: int = DEFAULT_TILE_SIZE,
    tile_overlap: int = DEFAULT_OVERLAP,
    diameter: float = DEFAULT_NUCLEI_DIAMETER,
    flow_threshold: float = DEFAULT_FLOW_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    nuclei_channel_name: str = DEFAULT_NUCLEI_CHANNEL,
    output_label_name: str = DEFAULT_NUCLEI_LABEL,
    use_parallel: bool = True,
    store_override: str = None,
    preview_full: bool = False,
    save_canvas_to: str = None,
) -> dict:
    """Native-20x nuclei segmentation for one position.

    Companion to `cell_segmentation.segment_single_position`.

    Args:
        experiment, position: usual identifiers
        tile_size, tile_overlap: shared with cell seg
        diameter: Cellpose nuclei diameter at 20x (default 150 — see bench notes)
        flow_threshold, iou_threshold: shared with cell seg
        nuclei_channel_name: source channel (default "nuclei_prediction")
        output_label_name: label group to write into (default "nuclear_seg")
        use_parallel: Dask multi-GPU mode (default on)
        store_override: optional source zarr path override
        preview_full: 2x2 production-tile preview mode (avoids touching real labels)
    """
    start = time.time()
    result = {
        "success": False,
        "n_nuclei": 0,
        "elapsed_time": 0,
        "experiment": experiment,
        "position": position,
    }

    # ---------------- Resolve store + channel index ----------------
    dataset = OpsDataset(experiment)
    source_path = Path(store_override) if store_override else dataset.store_paths.get(
        "pheno_assembled_v3"
    )
    if source_path is None or not source_path.exists():
        result["error"] = f"v3 store not found for {experiment}: {source_path}"
        return result

    try:
        channel_names, nuc_idx = _get_channel_indices_local(
            source_path, nuclei_channel=nuclei_channel_name,
        )
    except ValueError as e:
        result["error"] = str(e)
        return result

    print(f"\n{'='*60}")
    print(f"Nuclei Segmentation (20x): {experiment} / {position}")
    print(f"{'='*60}")
    print(f"  Source: {source_path}")
    print(f"  Channel: '{nuclei_channel_name}' -> index {nuc_idx}")
    print(f"  Tile size: {tile_size}, Overlap: {tile_overlap}")
    print(f"  Cellpose: d={diameter}, ft={flow_threshold}, iou={iou_threshold}")
    print(f"  Output label: '{output_label_name}'")
    if preview_full:
        # Use a separate label so we don't clobber real nuclei_seg data
        if output_label_name == DEFAULT_NUCLEI_LABEL:
            output_label_name = f"{DEFAULT_NUCLEI_LABEL}_preview"
        print(f"  PREVIEW-FULL mode → writing to '{output_label_name}' (2x2 tiles)")

    # ---------------- Delete stale label groups ----------------
    import zarr as _zarr
    _store_root = _zarr.open(str(source_path), mode="r+")
    _labels_group = _store_root.get(f"{position}/labels", None)
    for name in (output_label_name, f"{output_label_name}_unstitched"):
        if _labels_group is not None and name in _labels_group:
            print(f"  *** Removing stale label group '{name}' ***")
            try:
                del _store_root[f"{position}/labels/{name}"]
            except KeyError:
                pass

    # ---------------- Source shape + tile grid ----------------
    with open_ome_zarr(source_path / position, layout="fov", mode="r") as ds:
        full_shape = ds["0"].shape  # (T, C, Z, Y, X)
        height_full, width_full = full_shape[-2], full_shape[-1]
    print(f"  Image size: {height_full} x {width_full}")

    if preview_full:
        n = 2
        crop = n * (tile_size - tile_overlap) + tile_overlap
        y_start = min(int(height_full * 0.25), max(0, height_full - crop))
        x_start = min(int(width_full * 0.25), max(0, width_full - crop))
        height = min(crop, height_full - y_start)
        width = min(crop, width_full - x_start)
        print(f"  Preview-full crop: ({y_start},{x_start}) {height}x{width}")
    else:
        y_start, x_start = 0, 0
        height, width = height_full, width_full

    tiles, n_tiles_y, n_tiles_x = _calculate_tile_grid_local(
        height, width, tile_size, tile_overlap,
    )
    print(f"  Tile grid: {n_tiles_y} x {n_tiles_x} = {len(tiles)} tiles")

    # ---------------- Init temp label array ----------------
    label_shape = (1, 1, full_shape[2], full_shape[3], full_shape[4])
    temp_label_name = f"{output_label_name}_unstitched"
    print(f"\n  Initializing temp label array: {temp_label_name}")
    _init_organelle_label_array(
        zarr_path=source_path,
        pos_path=position,
        organelle_name=temp_label_name,
        shape=label_shape,
        dtype=np.int32,
        chunks=(1, 1, 1, 512, 512),
        shards_ratio=(1, 1, 1, 32, 32),
    )

    # ---------------- Shared-mem canvas ----------------
    import multiprocessing.shared_memory as _shm_mod
    canvas_nbytes = height * width * 4
    _shm = _shm_mod.SharedMemory(create=True, size=canvas_nbytes)
    canvas = np.ndarray((height, width), dtype=np.int32, buffer=_shm.buf)
    canvas[:] = 0
    print(f"  Canvas: {height}x{width} int32 ({canvas_nbytes/1e9:.1f} GB, shm={_shm.name})")

    # ---------------- Pass 1: parallel tile segmentation ----------------
    print(f"\n  Pass 1: Segmenting {len(tiles)} tiles at 20x ...")
    _t1 = time.time()

    MAX_LABELS_PER_TILE = 50000
    label_offsets = {}
    tile_entries = []
    for idx, t in enumerate(tiles):
        label_offsets[(t["ty"], t["tx"])] = idx * MAX_LABELS_PER_TILE
        tile_entries.append({
            "tile_idx": idx,
            "tile_info": t,
            "label_offset": idx * MAX_LABELS_PER_TILE,
        })

    overlap_cache = {}
    running_offset = 0

    if use_parallel and len(tiles) > 1:
        available_gpus = _setup_gpu_environment()
        # VRAM-adaptive worker count — nuclei pass uses ~2 GB, so we can pack more
        # than cell seg (which uses ~12-13 GB). Stay conservative at 3/GPU for now;
        # can tune if Pass 1 dominates.
        env_workers = int(os.environ.get("NUC_SEG_WORKERS_PER_GPU", 0))
        if env_workers > 0:
            workers_per_gpu = env_workers
        else:
            workers_per_gpu = 3
        num_workers = workers_per_gpu * len(available_gpus)
        num_workers = min(num_workers, len(tile_entries))
        print(f"  Parallel: {num_workers} workers ({workers_per_gpu}/GPU), GPUs: {available_gpus}")

        # Shuffle for load balance (same trick as cell seg)
        import random as _random
        _random.Random(42).shuffle(tile_entries)

        # Even-split batches
        n_batches = min(num_workers, len(tile_entries))
        batches = []
        for b in range(n_batches):
            s = b * len(tile_entries) // n_batches
            e = (b + 1) * len(tile_entries) // n_batches
            batches.append(tile_entries[s:e])

        # Tell Dask to spawn workers via spawn (fresh interpreter) instead of
        # the platform default (fork on Linux). Defends against the case where
        # something has already initialized CUDA in the parent — spawn never
        # inherits parent state. Set BEFORE MultiGPUCluster constructs Nannies.
        import dask as _dask
        _dask.config.set(
            {"distributed.worker.multiprocessing-method": "spawn"}
        )

        # Restore CUDA_VISIBLE_DEVICES (we popped it at module top to keep the
        # parent's cupy import from initializing CUDA). MultiGPUCluster will
        # pop it again internally and feed per-worker env={CVD=...} to Nannies.
        if _PARENT_CUDA_VISIBLE_DEVICES is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = _PARENT_CUDA_VISIBLE_DEVICES

        parent_cuda_devices = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        with MultiGPUCluster(available_gpus, workers_per_gpu) as multi_cluster:
            print(f"  Dashboard: {multi_cluster.dashboard_link}")
            futures = []
            for batch in batches:
                fut = multi_cluster.submit(
                    _segment_nuclei_tiles_batch_worker,
                    tile_batch=batch,
                    source_path=str(source_path),
                    position=position,
                    y_offset=y_start,
                    x_offset=x_start,
                    diameter=diameter,
                    flow_threshold=flow_threshold,
                    nuclei_channel=nuc_idx,
                    tile_overlap=tile_overlap,
                    n_tiles_y=n_tiles_y,
                    n_tiles_x=n_tiles_x,
                    canvas_shm_name=_shm.name,
                    canvas_height=height,
                    canvas_width=width,
                )
                futures.append(fut)

            from distributed import wait as _dask_wait
            _dask_wait(futures)
            from concurrent.futures import ThreadPoolExecutor as _GP
            with _GP(max_workers=len(futures)) as gp:
                all_results = list(gp.map(lambda f: f.result(), futures))
            for br in all_results:
                for tr in br:
                    ty, tx = tr["ty"], tr["tx"]
                    overlap_cache[(ty, tx)] = {
                        "right": tr.get("right_overlap"),
                        "bottom": tr.get("bottom_overlap"),
                        "left": tr.get("left_overlap"),
                        "top": tr.get("top_overlap"),
                    }
                    running_offset = max(
                        running_offset, label_offsets[(ty, tx)] + tr["n_labels"]
                    )

        if parent_cuda_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = parent_cuda_devices

    else:
        # Sequential fallback (preview / debug / single-tile)
        print("  Sequential mode (single GPU)")
        # Lazy import — only this branch needs it
        from cyclops_process.processes.cell_seg.cell_segmentation import (
            _create_cellpose_model,
        )
        model = _create_cellpose_model("nuclei", gpu=True)
        with open_ome_zarr(source_path / position, layout="fov", mode="r") as ds:
            for entry in tile_entries:
                t = entry["tile_info"]
                ty, tx = t["ty"], t["tx"]
                y0 = y_start + t["y_start"]
                y1 = y_start + t["y_end"]
                x0 = x_start + t["x_start"]
                x1 = x_start + t["x_end"]
                nuc = np.asarray(ds["0"][0, nuc_idx, 0, y0:y1, x0:x1]).astype(np.float32)
                masks, _, _ = model.eval(nuc, diameter=diameter, flow_threshold=flow_threshold)
                masks_off = masks.astype(np.int32)
                if masks_off.max() > 0:
                    masks_off[masks_off > 0] += entry["label_offset"]
                    fg = masks_off > 0
                    canvas[t["y_start"]:t["y_end"], t["x_start"]:t["x_end"]][fg] = masks_off[fg]
                running_offset = max(running_offset, entry["label_offset"] + int(masks.max()))
                overlap_cache[(ty, tx)] = {
                    "right": masks_off[:, -tile_overlap:].copy() if tx + 1 < n_tiles_x else None,
                    "bottom": masks_off[-tile_overlap:, :].copy() if ty + 1 < n_tiles_y else None,
                    "left": masks_off[:, :tile_overlap].copy() if tx > 0 else None,
                    "top": masks_off[:tile_overlap, :].copy() if ty > 0 else None,
                }

    max_label = running_offset
    _t1_total = time.time() - _t1
    print(f"  Pass 1 done: {max_label} provisional labels ({_t1_total:.1f}s)")

    # Lazy-import Pass 2/3 helpers from cell_segmentation NOW that the GPU
    # workers have finished (Dask cluster shut down above). The chain
    # cell_segmentation → segment → cupy/stitch.tile initializes CUDA in the
    # parent here, but that no longer matters: no more workers will fork.
    from cyclops_process.processes.cell_seg.cell_segmentation import (
        _compute_iou_merge_pairs_from_cache,
        _apply_union_find_merges,
        _build_pyramids_from_canvas,
    )

    # ---------------- Pass 2: IoU merge ----------------
    _t2 = time.time()
    print(f"\n  Pass 2: IoU merge ...")
    merge_pairs = _compute_iou_merge_pairs_from_cache(
        overlap_cache=overlap_cache,
        n_tiles_y=n_tiles_y,
        n_tiles_x=n_tiles_x,
        iou_threshold=iou_threshold,
    )
    _t2_total = time.time() - _t2
    print(f"  {len(merge_pairs)} merge pairs ({_t2_total:.1f}s)")

    # ---------------- Pass 3: union-find + write ----------------
    _t3 = time.time()
    print(f"\n  Pass 3: Union-find + write ...")
    canvas[:] = _apply_union_find_merges(canvas.copy(), merge_pairs, max_label)
    n_nuclei = int(canvas.max())
    print(f"  {n_nuclei} nuclei after merging")

    # Parallel shard-row write (same as cell seg)
    print(f"  Writing canvas → zarr ...")
    _t_w = time.time()
    from concurrent.futures import ThreadPoolExecutor
    shard_y = 512 * 32  # 16384
    n_strips = max(1, (height + shard_y - 1) // shard_y)

    def _write_strip(strip_idx):
        _store = _zarr.open(str(source_path), mode="r+")
        _arr = _store[position]["labels"][temp_label_name]["0"]
        row_s = strip_idx * shard_y
        row_e = min(height, (strip_idx + 1) * shard_y)
        _arr[0, 0, 0,
             y_start + row_s:y_start + row_e,
             x_start:x_start + width] = canvas[row_s:row_e, :]

    with ThreadPoolExecutor(max_workers=n_strips) as pool:
        list(pool.map(_write_strip, range(n_strips)))
    print(f"  Wrote {n_strips} strips in {time.time() - _t_w:.1f}s")

    # Snapshot canvas + release shm before rename
    canvas_snapshot = canvas.copy()
    _shm.close()
    _shm.unlink()

    # Save canvas + crop coords for A/B comparison (preview mode is the common case)
    if save_canvas_to:
        save_path = Path(save_canvas_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_path,
            canvas=canvas_snapshot,
            y_start=y_start,
            x_start=x_start,
            height=height,
            width=width,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            diameter=diameter,
            experiment=experiment,
            position=position,
        )
        print(f"  Saved canvas → {save_path}")

    # ---------------- Rename + metadata ----------------
    import shutil
    print(f"\n  Rename {temp_label_name} -> {output_label_name}")
    _store_root = _zarr.open(str(source_path), mode="r+")
    _labels_group = _store_root[position]["labels"]
    if output_label_name in _labels_group:
        del _labels_group[output_label_name]
    temp_path = source_path / position / "labels" / temp_label_name
    final_path = source_path / position / "labels" / output_label_name
    if temp_path.exists():
        shutil.move(str(temp_path), str(final_path))

    if not preview_full:
        meta = _build_nuclei_seg_metadata(
            label_name=output_label_name,
            channel_names=channel_names,
            experiment=experiment,
            n_nuclei=n_nuclei,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            diameter=diameter,
            flow_threshold=flow_threshold,
            iou_threshold=iou_threshold,
        )
        _update_labels_metadata(
            zarr_path=source_path,
            pos_path=position,
            new_label_name=output_label_name,
            metadata=meta,
        )
        # Clean stale temp entry in labels list
        _store_root = _zarr.open(str(source_path), mode="r+")
        _lg = _store_root[position]["labels"]
        existing = list(_lg.attrs.get("labels", []))
        if temp_label_name in existing:
            existing.remove(temp_label_name)
            _lg.attrs["labels"] = existing
        print(f"  Metadata written.")

    # ---------------- Pyramids ----------------
    if not preview_full:
        print(f"\n  Building pyramids for {output_label_name} ...")
        _t_p = time.time()
        _build_pyramids_from_canvas(
            canvas=canvas_snapshot,
            source_path=source_path,
            position=position,
            label_name=output_label_name,
            n_levels=5,
            chunks=(1, 1, 1, 512, 512),
            shards_ratio=(1, 1, 1, 32, 32),
        )
        print(f"  Pyramids built in {time.time() - _t_p:.1f}s")
    else:
        print(f"  Preview-full: skipping pyramid build (will delete preview label)")
        # Clean up preview label
        if final_path.exists():
            shutil.rmtree(final_path)
            print(f"  Deleted preview label {final_path}")

    _t3_total = time.time() - _t3
    result["success"] = True
    result["n_nuclei"] = n_nuclei
    result["elapsed_time"] = time.time() - start
    result["pass1_time"] = _t1_total
    result["pass2_time"] = _t2_total
    result["pass3_time"] = _t3_total
    result["merge_pairs"] = len(merge_pairs)
    print(f"\n  Completed in {result['elapsed_time']:.1f}s")
    print(f"  Pass1={_t1_total:.1f}s Pass2={_t2_total:.1f}s Pass3={_t3_total:.1f}s")
    print(f"{'='*60}\n")
    return result


# =============================================================================
# CLI
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Native-20x nuclei segmentation")
    p.add_argument("--experiment", required=True)
    p.add_argument("--position", required=True, help="e.g. A/1/0")
    p.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    p.add_argument("--tile-overlap", type=int, default=DEFAULT_OVERLAP)
    p.add_argument("--diameter", type=float, default=DEFAULT_NUCLEI_DIAMETER)
    p.add_argument("--flow-threshold", type=float, default=DEFAULT_FLOW_THRESHOLD)
    p.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    p.add_argument("--nuclei-channel", default=DEFAULT_NUCLEI_CHANNEL)
    p.add_argument("--output-label", default=DEFAULT_NUCLEI_LABEL)
    p.add_argument("--store", default=None, help="Override source zarr path")
    p.add_argument("--no-parallel", action="store_true")
    p.add_argument("--preview-full", action="store_true")
    p.add_argument(
        "--save-canvas-to",
        default=None,
        help="Save final 20x mask canvas + crop coords to this .npz path (for A/B compare).",
    )
    return p


def main():
    args = _build_arg_parser().parse_args()
    res = segment_nuclei_single_position(
        experiment=args.experiment,
        position=args.position,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        diameter=args.diameter,
        flow_threshold=args.flow_threshold,
        iou_threshold=args.iou_threshold,
        nuclei_channel_name=args.nuclei_channel,
        output_label_name=args.output_label,
        use_parallel=not args.no_parallel,
        store_override=args.store,
        preview_full=args.preview_full,
        save_canvas_to=args.save_canvas_to,
    )
    if not res.get("success"):
        print(f"FAILED: {res.get('error', 'unknown error')}")
        raise SystemExit(1)
    print(f"\nDONE. n_nuclei={res['n_nuclei']}  wall={res['elapsed_time']:.1f}s")


if __name__ == "__main__":
    main()
