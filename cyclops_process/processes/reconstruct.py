import os
import shutil
import sys
import json
import math
import argparse
import random
import time
import uuid

import numpy as np
import psutil
import yaml
from tqdm import tqdm
from pathlib import Path
from typing import Tuple
from scipy.ndimage import map_coordinates as map_coordinates_cpu
from joblib import Parallel, delayed

# GPU-accelerated map_coordinates via CuPy (fallback to CPU if unavailable)
try:
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates as map_coordinates_gpu
    cp.array([1.0])  # Test GPU access
    _HAS_CUPY_GPU = True
    print("[reconstruct.py] Using CuPy (GPU) for distortion correction")
except Exception:
    _HAS_CUPY_GPU = False

from contextlib import redirect_stdout, redirect_stderr

# Dask for GPU-safe parallel processing
from dask.distributed import LocalCluster, Client

# Configure waveorder package location via WAVEORDER_PATH environment variable
# IMPORTANT: Must be set BEFORE any waveorder imports so the loky forkserver
# (forked from parent) inherits the correct modules.  Workers inherit from
# forkserver, avoiding 15 cold-imports on the network filesystem.
if "WAVEORDER_PATH" in os.environ:
    waveorder_path = os.environ["WAVEORDER_PATH"]
    sys.path = [p for p in sys.path if 'waveorder' not in p or 'site-packages' in p]
    sys.path.insert(0, waveorder_path)

from iohub import open_ome_zarr
from iohub.ngff import TransformationMeta

# Pre-import waveorder modules at module level so the loky forkserver inherits them.
# Without this, each of 15 loky workers cold-imports torch+waveorder from NFS,
# causing ~9 min startup overhead (was ~90s with module-level imports).
try:
    from waveorder.cli.compute_transfer_function import compute_transfer_function_cli
    from waveorder.cli.apply_inverse_transfer_function import (
        apply_inverse_transfer_function_single_position,
    )
except ModuleNotFoundError:  # waveorder (-> torch) optional; used only in GPU recon paths
    compute_transfer_function_cli = None
    apply_inverse_transfer_function_single_position = None

sys.path.insert(0, os.getcwd())
from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.profiling.decorators import versioned_function
from cyclops_utils.hpc.resource_manager import get_optimal_workers
from cyclops_utils.hpc.gpu_utils import _setup_gpu_environment
from cyclops_utils.data.filesystem import (
    ensure_output_path,
    async_delete_path,
)

from cyclops_process.utils.waveorder_utils import _normalize_subpixel_options
from cyclops_utils.hpc.parallel_utils import call_in_spawned_process
from cyclops_process.metrics.metrics_reconstruction import (
    generate_subtile_heatmaps_from_csv,
)
from cyclops_process.utils.recon_utils import (
    _setup_auto2d_paths_and_config,
    _create_output_stores,
    save_subtile_metadata,
    generate_subtile_report,
)
from cyclops_utils.io.zarr_utils import (
    _validate_output_images,
    _maybe_sample_positions,
    _resolve_output_path_for_debug,
)
from cyclops_utils.io.zarr_precreate import create_hcs_store_fast

from cyclops_process.processes.reconstruct_subtile import reconstruct_subtile_autofocus


def _apply_poly_transform(mat, height, width, indices, order, mode, use_gpu=False):
    """Apply 2D distortion correction to each (T,C,Z) slice independently.

    Expects mat of shape (T, C, Z, H, W) and returns the same shape with corrected H,W.
    """
    # function adapted from discorpy
    # https://github.com/DiamondLightSource/discorpy
    t, c, z = mat.shape[0], mat.shape[1], mat.shape[2]
    # Flatten over T, C, Z to a stack of 2D images
    mat_flat = np.reshape(mat, (t * c * z, height, width))

    if use_gpu and _HAS_CUPY_GPU:
        # GPU path: transfer indices once, process all slices on GPU
        # CuPy expects coordinates as a single (ndim, ...) array, not a tuple
        # indices is a tuple of two (H*W, 1) arrays — stack into (2, H*W, 1)
        gpu_indices = cp.asarray(np.stack([indices[0], indices[1]], axis=0))
        corrected_list = []
        for slice2d in mat_flat:
            gpu_slice = cp.asarray(slice2d)
            gpu_result = map_coordinates_gpu(
                gpu_slice, gpu_indices, order=order, mode=mode
            )
            corrected_list.append(cp.asnumpy(gpu_result))
        corrected_flat = np.array(corrected_list)
    else:
        # CPU path
        corrected_flat = np.array(
            [
                map_coordinates_cpu(slice2d, indices, order=order, mode=mode)
                for slice2d in mat_flat
            ]
        )

    # Restore original leading dimensions (T, C, Z, H, W)
    corrected = corrected_flat.reshape((t, c, z, height, width))
    return corrected


def _correct_distortion_worker(
    pos, source_path, corrected_path, channel_slice, height, width, indices, use_gpu,
):
    """Worker function for correct_distortion (module-level for Dask pickling)."""
    import zarr as zarr_mod

    src_arr = zarr_mod.open(str(Path(source_path) / pos / "0"), mode="r")
    data = np.asarray(src_arr)[:, channel_slice, :, :, :]
    corrected_data = _apply_poly_transform(
        data, height, width, indices, order=3, mode="constant", use_gpu=use_gpu,
    )
    dst_arr = zarr_mod.open(str(Path(corrected_path) / pos / "0"), mode="r+")
    dst_arr[:] = corrected_data


def _reconstruct_chunk_with_streams_worker(
    chunk, input_store_path, transfer_function_path, config_path,
    output_store_path, output_channel_names, num_timepoints,
    ngff_version="0.4",
):
    """GPU reconstruction worker with direct zarr I/O and double prefetch.

    Bypasses iohub entirely — uses zarr.open() for direct array access.
    Double prefetch: 2 positions loaded ahead while GPU processes current.
    Async writes overlap with next position's compute.
    """
    import torch
    import zarr as zarr_mod
    import time as _time
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor, Future
    from waveorder.io import utils as wo_utils
    from waveorder.cli.settings import ReconstructionSettings
    from waveorder.models import phase_thick_3d

    pid = os.getpid()
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    t_init = _time.time()

    # Parse config ONCE
    settings = wo_utils.yaml_to_model(config_path, ReconstructionSettings)
    tf_settings = settings.phase.transfer_function
    recon_settings = settings.phase.apply_inverse

    # Resolve input channel index from config
    # Read channel names from first position's metadata (v3: zarr.json, v2: .zattrs)
    import json
    first_pos           = Path(input_store_path) / chunk[0]
    first_pos_zarr_json = first_pos / "zarr.json"
    first_pos_zattrs    = first_pos / ".zattrs"
    if first_pos_zarr_json.exists():
        if first_pos_zattrs.exists():
            print(f"[WARN] Both .zattrs and zarr.json found at {first_pos} — reading as zarr v3")
        with open(first_pos_zarr_json) as f:
            raw = json.load(f)
        pos_meta = raw.get("attributes", raw)
    elif first_pos_zattrs.exists():
        with open(first_pos_zattrs) as f:
            pos_meta = json.load(f)
    else:
        pos_meta = {}
    store_channels = [
        ch.get("label", f"ch{i}")
        for i, ch in enumerate(
            pos_meta.get("omero", {}).get("channels", [])
        )
    ]
    if settings.input_channel_names and store_channels:
        input_channel_idx = store_channels.index(settings.input_channel_names[0])
    else:
        input_channel_idx = 0

    # Load transfer functions to CPU pinned memory (complex64)
    # Use iohub for TF store (small, complex nested structure)
    tf_ds = open_ome_zarr(str(transfer_function_path), mode="r")
    real_tf_cpu = torch.tensor(
        tf_ds["real_potential_transfer_function"][0, 0]
    ).pin_memory()
    imag_tf_cpu = torch.tensor(
        tf_ds["imaginary_potential_transfer_function"][0, 0]
    ).pin_memory()
    tf_ds.close()

    print(f"[Worker {pid}] Ready: {len(chunk)} positions, device={device_str}, "
          f"channel={input_channel_idx}, init={_time.time()-t_init:.1f}s", flush=True)

    # Thread pool for I/O: 2 prefetch + 1 write
    io_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="io")

    def _load_position(pos_name, t_idx):
        """Load one position via direct zarr (no iohub)."""
        arr = zarr_mod.open(str(Path(input_store_path) / pos_name / "0"), mode="r")
        data = np.array(arr[t_idx, input_channel_idx])  # (Z, Y, X)
        if np.all(data == 0):
            return None
        return np.int32(data)

    def _write_position(pos_name, result_np, t_idx):
        """Write result via iohub fov (fast single-position open)."""
        with open_ome_zarr(
            str(Path(output_store_path) / pos_name), layout="fov", mode="r+",
            version=ngff_version,
        ) as pos_ds:
            pos_ds["0"].oindex[t_idx, 0] = result_np[0]  # Single output channel

    processed = 0
    skipped = 0
    errors = 0
    t_start = _time.time()
    prev_write_future = None

    for t_idx in range(num_timepoints):
        # Prime double prefetch
        prefetch_q: deque[Future] = deque()
        for k in range(min(2, len(chunk))):
            prefetch_q.append(io_pool.submit(_load_position, chunk[k], t_idx))

        for i, pos in enumerate(chunk):
            try:
                # Get current position's data (already prefetched)
                zyx_int32 = prefetch_q.popleft().result()

                # Submit prefetch for position i+2 (double-buffer)
                if i + 2 < len(chunk):
                    prefetch_q.append(
                        io_pool.submit(_load_position, chunk[i + 2], t_idx)
                    )

                if zyx_int32 is None:
                    skipped += 1
                    continue

                # H2D transfer (data + TFs)
                zyx_gpu = torch.tensor(zyx_int32, dtype=torch.float32, device=device)
                real_tf = real_tf_cpu.to(device, non_blocking=True)
                imag_tf = imag_tf_cpu.to(device, non_blocking=True)

                # GPU compute
                result_gpu = phase_thick_3d.apply_inverse_transfer_function(
                    zyx_data=zyx_gpu,
                    real_potential_transfer_function=real_tf,
                    imaginary_potential_transfer_function=imag_tf,
                    z_padding=tf_settings.z_padding,
                    reconstruction_algorithm=recon_settings.reconstruction_algorithm,
                    regularization_strength=recon_settings.regularization_strength,
                    TV_rho_strength=recon_settings.TV_rho_strength,
                    TV_iterations=recon_settings.TV_iterations,
                )

                # EXPLICIT SYNC — ensure compute is done before D2H
                torch.cuda.synchronize()
                del zyx_gpu, real_tf, imag_tf

                # D2H + expand dims
                while result_gpu.ndim < 4:
                    result_gpu = result_gpu.unsqueeze(0)
                result_np = result_gpu.cpu().numpy()
                del result_gpu

                # Wait for previous write before submitting new one
                if prev_write_future is not None:
                    prev_write_future.result()

                # Async write
                prev_write_future = io_pool.submit(
                    _write_position, pos, result_np, t_idx
                )
                processed += 1

                if processed % 50 == 0:
                    elapsed = _time.time() - t_start
                    rate = processed / elapsed if elapsed > 0 else 0
                    print(f"[Worker {pid}] {processed}/{len(chunk)*num_timepoints} done "
                          f"({rate:.1f} pos/s, {skipped} skipped, {errors} errors)",
                          flush=True)

            except Exception as e:
                errors += 1
                print(f"[Worker {pid}] [{pos}] Error: {e}", file=sys.stderr, flush=True)
                import traceback
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()

    # Wait for final write and drain prefetch queue
    if prev_write_future is not None:
        prev_write_future.result()
    for f in prefetch_q:
        f.result()

    io_pool.shutdown(wait=True)

    elapsed = _time.time() - t_start
    print(f"[Worker {pid}] Complete: {processed} processed, {skipped} skipped, "
          f"{errors} errors in {elapsed:.1f}s ({processed/elapsed:.1f} pos/s)",
          flush=True)
    return len(chunk)


def _parse_tile_row_col(tile_str: str) -> tuple[int, int]:
    digits = "".join(ch for ch in tile_str if ch.isdigit())
    if not digits:
        return 0, 0
    # Heuristic: many datasets use 6 digits (RRRCCC). Fallback to even split if possible.
    if len(digits) >= 4 and len(digits) % 2 == 0:
        half = len(digits) // 2
        return int(digits[:half]), int(digits[half:])
    # Fallback: treat as a square index (row=col=index)
    val = int(digits)
    return val, val


@versioned_function("v1.0")
def correct_distortion(
    experiment: str,
    process: str = "lc_5x",
    num_workers: int = None,
    debug_n_positions: int | None = None,
    debug_output_suffix: str = "_debug",
    ngff_version: str = "0.4",
):
    """
    Correct the pin-cusion distortion based on a pre-existing calibration, created by imaging a grid.

    Supports two modes via `process`:
      - process="lc_5x" (default): writes corrected 5x BF images (BF channel only) to a new output store
      - process="iss": writes corrected multi-channel ISS images to a dedicated corrected store
    """
    if num_workers is None:
        if _HAS_CUPY_GPU:
            # GPU workers share a single GPU — each position's map_coordinates
            # is fast on GPU (~ms), so scale workers to available VRAM.
            num_workers = get_optimal_workers(
                use_gpu=True, model_vram_gb=0.2, data_vram_gb=0.3, verbose=False
            )
        else:
            num_workers = get_optimal_workers(use_gpu=False, verbose=False)

    print(
        f"Correcting distortion for {experiment} using {num_workers} workers (process={process})"
    )

    dataset = OpsDataset(experiment)

    # Helper to build distortion indices from params and shape
    def _build_indices(height, width, coeffs):
        xcenter, ycenter, *list_fact = coeffs
        xu_list = np.arange(width) - xcenter
        yu_list = np.arange(height) - ycenter
        xu_mat, yu_mat = np.meshgrid(xu_list, yu_list)
        ru_mat = np.sqrt(xu_mat**2 + yu_mat**2)
        fact_mat = np.sum(
            np.asarray([factor * ru_mat**i for i, factor in enumerate(list_fact)]),
            axis=0,
        )
        xd_mat = np.float32(np.clip(xcenter + fact_mat * xu_mat, 0, width - 1))
        yd_mat = np.float32(np.clip(ycenter + fact_mat * yu_mat, 0, height - 1))
        return (np.reshape(yd_mat, (-1, 1)), np.reshape(xd_mat, (-1, 1)))

    # ISS mode: write to a dedicated corrected output store and process all channels
    if process == "iss":
        source_path = dataset.store_paths["iss_drift_corrected"]
        corrected_path = dataset.store_paths["iss_distortion_corrected"]
        params_path = dataset.config_paths["distoration_corr_params_iss"]

    elif process == "lc_5x":
        # Default (5x tracking BF) mode below
        source_path = dataset.store_paths["lc_5x"]
        corrected_path = dataset.store_paths["lc_5x_bf_corrected"]
        params_path = dataset.config_paths["distortion_corr_params"]

    # Get metadata from source store (positions, channels, shapes)
    with open_ome_zarr(source_path, mode="r", version=ngff_version) as ds:
        position_list = [a[0] for a in ds.positions()]
        src_channel_names = ds.channel_names
        output_scale = ds[position_list[0]].scale
        src_shape = ds[position_list[0]].data.shape
        dtype = ds[position_list[0]].data.dtype
        src_chunks = ds[position_list[0]].data.chunks  # Preserve source chunking
        output_store_transform = TransformationMeta(type="scale", scale=output_scale)

    # Mode-specific channel selection and output shape
    if process == "iss":
        channel_slice = slice(0, len(src_channel_names))
        out_channel_names = list(src_channel_names)
        out_shape = src_shape
    else:  # lc_5x (BF-only)
        if len(src_channel_names) > 1:
            if "BF" in src_channel_names:
                bf_channel = src_channel_names.index("BF")
            else:
                bf_channel = 0
            channel_slice = slice(bf_channel, bf_channel + 1)
            out_channel_names = ["BF"]
            out_shape = src_shape[:1] + (1,) + src_shape[2:]
        else:
            channel_slice = slice(0, 1)
            out_channel_names = list(src_channel_names)
            out_shape = src_shape

    # Prepare distortion correction indices
    with open(params_path, "r") as file:
        coeffs = [float(line.strip().rsplit(" ", -1)[-1]) for line in file]
    (height, width) = out_shape[-2:]
    indices = _build_indices(height, width, coeffs)

    # Pre-allocate the output zarr store sequentially to prevent race conditions
    # Debug mode: allow sampling and write to a dedicated debug store
    position_list = _maybe_sample_positions(position_list, debug_n_positions)
    corrected_path = _resolve_output_path_for_debug(
        corrected_path, debug_n_positions, debug_output_suffix
    )

    print(f"Initializing output store at {corrected_path}")
    # Rebuild from scratch: every position is recorrected below.
    async_delete_path(corrected_path)
    # Use fast precreation method (38x faster, O(1) scaling)
    print(f"Pre-creating {len(position_list)} positions using fast method...")
    create_hcs_store_fast(
        store_path=corrected_path,
        positions=position_list,
        shape=out_shape,
        chunks=src_chunks,  # Preserve source chunking
        dtype=dtype,
        scale=output_store_transform.scale,
        channel_names=out_channel_names,
        version=ngff_version,
    )

    # Detect GPU and set worker count
    use_gpu = _HAS_CUPY_GPU
    if use_gpu:
        print("GPU detected — using CuPy-accelerated map_coordinates")
    else:
        print("No GPU — using CPU scipy.ndimage.map_coordinates")

    # Run correction in parallel using Dask LocalCluster (GPU-safe, no fork())
    from functools import partial

    process_func = partial(
        _correct_distortion_worker,
        source_path=str(source_path),
        corrected_path=str(corrected_path),
        channel_slice=channel_slice,
        height=height,
        width=width,
        indices=indices,
        use_gpu=use_gpu,
    )

    print(f"Applying distortion correction with {num_workers} workers...")
    cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1)
    client = Client(cluster)
    try:
        futures = client.map(process_func, position_list)
        for future in tqdm(futures, desc="Correcting positions", total=len(position_list)):
            future.result()
    finally:
        try:
            client.close()
            cluster.close(timeout=120)
        except TimeoutError:
            pass
    print("Distortion correction complete.")
    return


def _ensure_transfer_function(
    transfer_function_path: Path,
    input_store_path: Path,
    positions: list[str],
    config2d_path: Path,
    verbose: bool,
):
    """Ensure transfer function exists, computing it if necessary."""
    if transfer_function_path.exists():
        if verbose:
            print(
                f"[Auto2D] Using existing transfer function at {transfer_function_path}"
            )
    else:
        if verbose:
            print(
                f"[Auto2D] Computing transfer function (2D config) → {transfer_function_path.name} (example FOV: {positions[0]})"
            )
        # Suppress all Waveorder output during TF computation
        with open(os.devnull, "w") as devnull, redirect_stdout(
            devnull
        ), redirect_stderr(devnull):
            compute_transfer_function_cli(
                input_position_dirpath=input_store_path / positions[0],
                config_filepath=config2d_path,
                output_dirpath=transfer_function_path,
            )


def _validate_channel_names_match(input_position_path: Path, config_path: Path) -> None:
    """Validate that channel names in config match the input dataset.

    Raises a clear error message if there's a mismatch, preventing confusing
    downstream errors from waveorder.
    """
    # Load config to get expected channel names
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config_channels = config.get("input_channel_names", [])
    if not config_channels:
        return  # No channel names specified in config

    # Load dataset channel names (v3: zarr.json, v2: .zattrs)
    zarr_json_path = input_position_path / "zarr.json"
    zattrs_path    = input_position_path / ".zattrs"
    if zarr_json_path.exists():
        if zattrs_path.exists():
            print(f"[WARN] Both .zattrs and zarr.json found at {input_position_path} — reading as zarr v3")
        with open(zarr_json_path, "r") as f:
            raw = json.load(f)
        meta = raw.get("attributes", raw)
    elif zattrs_path.exists():
        with open(zattrs_path, "r") as f:
            meta = json.load(f)
    else:
        return  # Can't validate without metadata

    # Extract channel labels from omero metadata
    dataset_channels = []
    omero = meta.get("omero", {})
    for ch in omero.get("channels", []):
        label = ch.get("label", "")
        if label:
            dataset_channels.append(label)

    if not dataset_channels:
        return  # No channel info in dataset

    # Check for mismatches
    missing_channels = [ch for ch in config_channels if ch not in dataset_channels]
    if missing_channels:
        raise ValueError(
            f"Channel name mismatch!\n"
            f"  Config '{config_path.name}' expects channels: {config_channels}\n"
            f"  Dataset '{input_position_path}' contains channels: {dataset_channels}\n"
            f"  Missing channels: {missing_channels}\n"
            f"  Please update the config file to use the correct channel names."
        )


def reconstruct_3d(
    dataset: OpsDataset,
    process: str,
    example_fov: str,
    input_path: Path,
    config_path: Path,
    output_path: Path,
    debug_n_positions: int | None,
    debug_output_suffix: str = "_debug",
    verbose: bool = False,
    ngff_version: str = "0.4",
):
    """Reconstruct 3D using Waveorder (single-channel Phase3D output)."""

    transfer_function_path = output_path.parent / Path(
        "transfer_function_" + config_path.stem + ".zarr"
    )

    # Validate channel names match before attempting transfer function creation
    _validate_channel_names_match(input_path / example_fov, config_path)

    # Ensure/confirm transfer function output path behavior
    # Reuse existing TF if present — it only depends on config params, not data.
    # Saves ~2 minutes of serial TF computation on re-runs.
    tf_choice = "skip" if transfer_function_path.exists() else "create"
    if tf_choice in ("skip", "resume"):
        if not transfer_function_path.exists():
            print(f"Transfer function does not exist at {transfer_function_path} - creating it...")
            print(f"  Input: {input_path / example_fov}")
            print(f"  Config: {config_path}")
            print(f"  Output: {transfer_function_path}")
            # Call directly to see output instead of spawning
            compute_transfer_function_cli(
                input_position_dirpath=input_path / example_fov,
                config_filepath=config_path,
                output_dirpath=transfer_function_path,
            )
            # Verify transfer function was actually created
            if not transfer_function_path.exists():
                raise FileNotFoundError(
                    f"Transfer function creation appeared to succeed but file not found at {transfer_function_path}"
                )
            print(f"Transfer function successfully created at {transfer_function_path}")
        else:
            print(f"Using existing transfer function at {transfer_function_path}")
    elif tf_choice in ("create", "overwrite"):
        if tf_choice == "overwrite" and transfer_function_path.exists():
            shutil.rmtree(transfer_function_path)
        print(f"Creating transfer function at {transfer_function_path}...")
        print(f"  Input: {input_path / example_fov}")
        print(f"  Config: {config_path}")
        print(f"  Output: {transfer_function_path}")
        # Call directly to see output instead of spawning
        compute_transfer_function_cli(
            input_position_dirpath=input_path / example_fov,
            config_filepath=config_path,
            output_dirpath=transfer_function_path,
        )
        # Verify transfer function was actually created
        if not transfer_function_path.exists():
            raise FileNotFoundError(
                f"Transfer function creation appeared to succeed but file not found at {transfer_function_path}"
            )
        print(f"Transfer function successfully created at {transfer_function_path}")

    input_store = open_ome_zarr(input_path, mode="r+", version=ngff_version)
    store_channels = input_store.channel_names
    from waveorder.io import utils as wo_utils
    from waveorder.cli.settings import ReconstructionSettings
    config_settings = wo_utils.yaml_to_model(config_path, ReconstructionSettings)
    config_channels = config_settings.input_channel_names
    if config_channels and len(config_channels) == 1 and config_channels[0] in store_channels:
        selected_idx = store_channels.index(config_channels[0])
    else:
        selected_idx = 0
    print(f"Channel selection: config={config_channels}, store={list(store_channels)}, using index {selected_idx} ('{store_channels[selected_idx]}')")
    positions = [a[0] for a in input_store.positions()]
    positions = _maybe_sample_positions(positions, debug_n_positions)
    output_path = _resolve_output_path_for_debug(
        output_path, debug_n_positions, debug_output_suffix
    )

    # Pre-create output store if missing; otherwise prompt/ reuse existing
    temp_fov = input_store[positions[0]]
    # Force single-channel output while preserving Z, Y, X from input
    output_shape = temp_fov.data.shape[:1] + (1,) + temp_fov.data.shape[2:]
    # Phase output must be floating point; force float32 to avoid integer casts
    output_dtype = np.float32
    # Preserve source chunking but adapt for single channel
    src_chunks = temp_fov.data.chunks
    output_chunks = src_chunks[:1] + (1,) + src_chunks[2:]  # (T, 1, Z, Y, X)

    # Rebuild from scratch: every position is reconstructed below.
    async_delete_path(output_path)
    # Detect GPU for CUDA streams pipelining
    import torch
    import threading
    import time

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        # Scale workers based on available VRAM
        # Each worker uses ~5GB GPU memory (observed on H200 with phenotyping data)
        # Reserve 10GB headroom for compute allocations (~544MB per FFT)
        gpu_props = torch.cuda.get_device_properties(0)
        vram_gb = gpu_props.total_memory / (1024 ** 3)
        # TFs transferred per-position then freed (~1.1GB transient, not resident)
        # Multiple workers may have TFs + data + intermediates on GPU simultaneously
        mem_per_worker_gb = 6.0
        usable_vram_gb = vram_gb - 7.0  # headroom for concurrent TF copies
        num_workers_vram = max(1, int(usable_vram_gb / mem_per_worker_gb))

        # Also cap by system RAM: each Dask worker process needs ~5GB
        # (Python/PyTorch imports ~1.5GB + TF pinned ~0.2GB + 2x position data + overhead)
        total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        num_workers_ram = max(1, int(total_ram_gb / mem_per_worker_gb))

        num_workers = min(num_workers_vram, num_workers_ram)
        print(f"GPU detected: {gpu_props.name} ({vram_gb:.0f}GB VRAM, {total_ram_gb:.0f}GB RAM) - "
              f"using {num_workers} workers (VRAM limit: {num_workers_vram}, RAM limit: {num_workers_ram})")
    else:
        num_workers = get_optimal_workers(use_gpu=False)

    num_timepoints = output_shape[0]
    print(
        f"Applying inverse transfer function locally across {len(positions)} positions x {num_timepoints} timepoints with {num_workers} workers..."
    )

    # Run store precreation in parallel with worker initialization
    precreation_thread = None
    precreation_exception = None

    def _precreate_store():
        nonlocal precreation_exception
        try:
            create_hcs_store_fast(
                store_path=output_path,
                positions=positions,
                shape=output_shape,
                chunks=output_chunks,
                dtype=output_dtype,
                scale=temp_fov.scale,
                channel_names=["Phase3D"],
                version=ngff_version,
            )
        except Exception as e:
            precreation_exception = e

    print(f"Pre-creating {len(positions)} positions in background while initializing workers...")
    precreation_thread = threading.Thread(target=_precreate_store, daemon=False)
    precreation_start = time.time()
    precreation_thread.start()

    if use_gpu:
        # GPU path: Use Dask LocalCluster (not joblib) — fork() breaks CUDA contexts
        # Start cluster in parallel with store precreation to save ~16s
        from functools import partial

        # Partition positions into chunks (one per worker)
        chunk_size = (len(positions) + num_workers - 1) // num_workers
        position_chunks = [
            positions[i:i + chunk_size]
            for i in range(0, len(positions), chunk_size)
        ]

        process_func = partial(
            _reconstruct_chunk_with_streams_worker,
            input_store_path=input_path,
            transfer_function_path=transfer_function_path,
            config_path=config_path,
            output_store_path=output_path,
            output_channel_names=["Phase3D"],
            num_timepoints=num_timepoints,
            ngff_version=ngff_version,
        )

        # Start cluster while precreation runs in background
        t_cluster = time.time()
        cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1)
        client = Client(cluster)
        print(f"Dask cluster started in {time.time()-t_cluster:.1f}s", flush=True)

        # Now wait for precreation before submitting work
        if precreation_thread is not None:
            precreation_thread.join()
            precreation_elapsed = time.time() - precreation_start
            print(f"Store precreation completed in {precreation_elapsed:.1f}s")
            if precreation_exception is not None:
                raise precreation_exception
        try:
            futures = client.map(process_func, position_chunks)
            results = []
            for future in tqdm(futures, desc="Worker chunks", total=len(position_chunks)):
                results.append(future.result())
        finally:
            try:
                client.close()
                cluster.close(timeout=120)
            except TimeoutError:
                pass
        print(f"Processed {sum(results)} positions total with {num_workers} GPU-pipelined workers")

    else:
        # Wait for precreation before CPU fallback
        if precreation_thread is not None:
            precreation_thread.join()
            precreation_elapsed = time.time() - precreation_start
            print(f"Store precreation completed in {precreation_elapsed:.1f}s")
            if precreation_exception is not None:
                raise precreation_exception
        # CPU fallback: process positions one at a time
        def _reconstruct_single_position(pos: str):
            with open(os.devnull, "w") as devnull, redirect_stdout(
                devnull
            ), redirect_stderr(devnull):
                apply_inverse_transfer_function_single_position(
                    input_position_dirpath=input_path / pos,
                    transfer_function_dirpath=transfer_function_path,
                    config_filepath=config_path,
                    output_position_dirpath=output_path / pos,
                    num_processes=1,
                    output_channel_names=["Phase3D"],
                )

        Parallel(n_jobs=num_workers)(
            delayed(_reconstruct_single_position)(pos)
            for pos in tqdm(positions, desc="Reconstructing positions")
        )

    _validate_output_images(output_path, raise_on_blank=True)


def _process_subtile_position_wrapper(
    pos,
    phase3d_store_path,
    raw_store_path,
    phase2d_store_path,
    dataset,
    cfg2d_base_model,
    optical_params,
    T, Y, X,
    position_scale,
    n_subtiles,
    blend_pixels,
    enable_subpixel_precision,
    polynomial_fit_order,
    midband_fractions,
    device="cpu",
):
    """Module-level wrapper for reconstruct_subtile_autofocus (needed for Dask pickling)."""
    from cyclops_process.processes.reconstruct_subtile import reconstruct_subtile_autofocus

    return pos, reconstruct_subtile_autofocus(
        pos,
        phase3d_store_path=phase3d_store_path,
        raw_store_path=raw_store_path,
        phase2d_store_path=phase2d_store_path,
        dataset=dataset,
        cfg2d_base_model=cfg2d_base_model,
        optical_params=optical_params,
        T=T, Y=Y, X=X,
        position_scale=position_scale,
        n_subtiles=n_subtiles,
        blend_pixels=blend_pixels,
        verbose=False,
        enable_subpixel_precision=enable_subpixel_precision,
        polynomial_fit_order=polynomial_fit_order,
        midband_fractions=midband_fractions,
        device=device,
    )


def reconstruct_2d_autofocus(
    experiment: str,
    process: str = "pheno-2d",
    n_subtiles: int = 256,
    blend_pixels: int = 25,
    enable_subpixel_precision: bool = True,
    polynomial_fit_order: int = 3,
    midband_fractions: Tuple[float, float] = (0.125, 0.25),
    verbose: bool = False,
    debug_n_positions: int | None = None,
    debug_output_suffix: str = "_debug",
    ngff_version: str = "0.4",
):
    """
    Auto-focus 2D reconstruction using an existing 3D phase stack to pick focus, for either 20x phenotyping
    (process="pheno-2d") or 5x tracking (process="track-2d"). Writes a unified 2-channel output store
    (channel 0=Phase2D, channel 1=Focus3D).

    Args:
        process: "pheno-2d" or "track-2d" to route to correct stores/configs
        n_subtiles: Number of subtiles (must be perfect square: 4, 9, 16, 25, etc.)
        blend_pixels: Pixels to blend at subtile borders for seamless stitching
    """
    # print the parameters selected
    print(f"[Auto2D] Reconstructing {experiment} with parameters:")
    print(f"  n_subtiles: {n_subtiles}")
    print(f"  blend_pixels: {blend_pixels}")
    print(f"  enable_subpixel_precision: {enable_subpixel_precision}")
    print(f"  polynomial_fit_order: {polynomial_fit_order}")
    print(f"  midband_fractions: {midband_fractions}")

    import torch
    import os
    print(f"\n[GPU Setup] CUDA_VISIBLE_DEVICES in main process: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    if torch.cuda.is_available():
        print(f"[GPU Setup] Detected {torch.cuda.device_count()} GPU(s), available indices: {list(range(torch.cuda.device_count()))}")

    # Setup paths, configuration, and positions (process-aware) # TODO: move to utils
    setup_data = _setup_auto2d_paths_and_config(
        experiment, process, debug_n_positions, debug_output_suffix, verbose,
        ngff_version=ngff_version,
    )
    dataset = setup_data["dataset"]
    raw_store_path = setup_data["raw_store_path"]
    phase3d_store_path = setup_data["phase3d_store_path"]
    phase2d_store_path = setup_data["phase2d_store_path"]
    config2d_path = setup_data["config2d_path"]
    cfg2d_base_model = setup_data["cfg2d_base_model"]
    optical_params = setup_data["optical_params"]
    transfer_function_path = setup_data["transfer_function_path"]
    positions = setup_data["positions"]

    # Determine effective sub-pixel capability and enforce subtile cap if disabled
    eff_enabled, eff_order, _ = _normalize_subpixel_options(  # TODO: waveorder utils
        enable_subpixel_precision, polynomial_fit_order
    )
    if not eff_enabled and n_subtiles > 25:
        print(
            "[SubtileRecon][WARN] Sub-pixel precision disabled; limiting n_subtiles to 25 (5x5), final subtiles: {n_subtiles}"
        )
        n_subtiles = 25

    # Ensure transfer function exists - TODO: don't need this
    _ensure_transfer_function(
        transfer_function_path, raw_store_path, positions, config2d_path, verbose
    )

    # Ensure output paths are ready (prompt user to overwrite/resume/skip if exists)
    is_debug = debug_n_positions is not None and debug_n_positions > 0
    # Rebuild from scratch: shapes come from _create_output_stores, not from
    # whatever an existing output store happened to declare.
    async_delete_path(phase2d_store_path)
    T, Y, X, position_scale, out_dtype = _create_output_stores(
        phase3d_store_path,
        phase2d_store_path,
        positions,
        dataset,
        cfg2d_base_model,
        verbose,
        ngff_version=ngff_version,
    )

    # Get the raw data path, which is needed for the actual reconstruction input
    raw_store_path = setup_data["raw_store_path"]

    print(
        f"[SubtileRecon] Using subtile-based autofocus: {n_subtiles} subtiles ({int(np.sqrt(n_subtiles))}x{int(np.sqrt(n_subtiles))}), blend_pixels={blend_pixels}"
    )

    # Setup GPU environment for Dask workers
    available_gpus = _setup_gpu_environment()
    use_gpu = torch.cuda.is_available() and len(available_gpus) > 0
    device = "cuda" if use_gpu else "cpu"

    # Scale workers based on GPU availability.
    # Each worker loads one position (~150MB RAM) plus Dask overhead (~0.5GB).
    # GPU usage is brief autofocus (~50-100ms), not the bottleneck.
    # Cap at 16 to avoid overwhelming system RAM and NFS.
    if use_gpu:
        gpu_props = torch.cuda.get_device_properties(0)
        vram_gb = gpu_props.total_memory / (1024 ** 3)
        num_workers = min(16, max(4, int(vram_gb // 5)))
        print(f"[SubtileRecon] GPU detected: {gpu_props.name} ({vram_gb:.0f}GB VRAM), device={device}, {num_workers} workers")
    else:
        num_workers = get_optimal_workers(use_gpu=False)

    print(
        f"[SubtileRecon] Reconstructing {len(positions)} positions with Dask ({num_workers} workers, device={device})..."
    )

    # Extract additional config parameters for reporting
    tf_params = cfg2d_base_model.phase.transfer_function
    apply_inv_params = cfg2d_base_model.phase.apply_inverse

    # Print the final recon config parameters
    print(f"[SubtileRecon] Config path: {config2d_path}")
    print(f"[SubtileRecon] Final recon config parameters:")
    print(f"  - NA_detection: {tf_params.numerical_aperture_detection}")
    print(f"  - NA_illumination: {tf_params.numerical_aperture_illumination}")
    print(f"  - wavelength_illumination: {tf_params.wavelength_illumination} µm")
    print(f"  - pixel_size (xy): {tf_params.yx_pixel_size} µm")
    print(f"  - pixel_size (z): {tf_params.z_pixel_size} µm")
    print(f"  - index_of_refraction: {tf_params.index_of_refraction_media}")
    print(f"  - invert_phase_contrast: {tf_params.invert_phase_contrast}")
    print(f"  - reconstruction_algorithm: {apply_inv_params.reconstruction_algorithm}")
    print(f"  - regularization_strength: {apply_inv_params.regularization_strength}")
    print(f"  - TV_rho_strength: {apply_inv_params.TV_rho_strength}")
    print(f"  - TV_iterations: {apply_inv_params.TV_iterations}")
    print(f"  - midband_fractions (focus detection): {midband_fractions}")

    timing_log = phase2d_store_path.parent / "reconstruction_timing.log"
    timing_log.unlink(missing_ok=True)
    print(f"\n{'='*80}")
    print(f"REAL-TIME TIMING LOG: {timing_log}")
    print(f"Monitor with: tail -f {timing_log}")
    print(f"{'='*80}\n")

    with open(timing_log, "a") as f:
        f.write(f"[TIMING] Starting subtile reconstruction at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"[TIMING] Total positions: {len(positions)}, Workers: {num_workers}, "
                f"Subtiles per position: {n_subtiles}\n\n")
        f.flush()

    recon_start = time.time()

    print(f"[GPU Config] Available GPUs: {available_gpus}")
    print(f"[GPU Config] Using device for autofocus and reconstruction: {device}")

    # Use Dask LocalCluster for GPU-safe parallel processing
    # Workers inherit CUDA_VISIBLE_DEVICES from parent (no plugin needed)
    from functools import partial
    process_func = partial(
        _process_subtile_position_wrapper,
        phase3d_store_path=phase3d_store_path,
        raw_store_path=raw_store_path,
        phase2d_store_path=phase2d_store_path,
        dataset=dataset,
        cfg2d_base_model=cfg2d_base_model,
        optical_params=optical_params,
        T=T, Y=Y, X=X,
        position_scale=position_scale,
        n_subtiles=n_subtiles,
        blend_pixels=blend_pixels,
        enable_subpixel_precision=enable_subpixel_precision,
        polynomial_fit_order=polynomial_fit_order,
        midband_fractions=midband_fractions,
        device=device,
    )

    cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1, memory_limit="8GiB")
    client = Client(cluster)
    try:
        print(f"[Dask] Dashboard: {client.dashboard_link}")
        futures = client.map(process_func, positions)
        results = []
        for future in tqdm(futures, desc="Subtile reconstruction", total=len(positions)):
            results.append(future.result())
    finally:
        try:
            client.close()
            cluster.close(timeout=120)
        except TimeoutError:
            pass  # Acceptable if work is complete

    subtile_results = dict(results)

    recon_elapsed = time.time() - recon_start

    with open(timing_log, "a") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"[TIMING] FINAL SUMMARY\n")
        f.write(f"[TIMING] Total reconstruction time: {recon_elapsed:.2f}s ({recon_elapsed/60:.1f} min)\n")
        f.write(f"[TIMING] Average time per position: {recon_elapsed/len(positions):.2f}s\n")
        f.write(f"[TIMING] Throughput: {len(positions)/recon_elapsed:.2f} positions/second\n")
        f.write(f"{'='*80}\n")
        f.flush()

    print(f"[TIMING] Total reconstruction time: {recon_elapsed:.2f}s ({recon_elapsed/60:.1f} min)")
    print(f"[TIMING] Average time per position: {recon_elapsed/len(positions):.2f}s")
    print(f"[TIMING] Throughput: {len(positions)/recon_elapsed:.2f} positions/second")

    # Save subtile metadata (use debug-specific tag to avoid overwriting production CSV)
    csv_dir = phase2d_store_path.parent
    csv_tag = f"{process}_debug" if is_debug else process
    metadata_csv = save_subtile_metadata(
        subtile_results, csv_dir, experiment, tag=csv_tag
    )

    if is_debug:
        print(f"[SubtileRecon] Skipping heatmaps and reports in debug mode to protect production data")
    else:
        # Generate subtile report
        if verbose:
            print(f"[SubtileRecon] Generating subtile report...")
            generate_subtile_report(subtile_results, experiment)

        #  generate heatmaps from the saved CSV (decoupled, reproducible)
        try:
            print(f"[SubtileRecon] Generating heatmaps from CSV...")
            # Tag figures with process (pheno-2d or track-2d) and save into results/phase_recon
            generate_subtile_heatmaps_from_csv(
                metadata_csv, dataset, experiment, tag=process
            )
        except Exception as e_csv:
            print(f"[SubtileRecon][WARN] Failed to generate heatmaps from CSV: {e_csv}")

    # No need to build reconstruction_results in subtile mode; reporting is handled above

    _validate_output_images(phase2d_store_path, raise_on_blank=True)

    print("Auto-focus subtiling 2D reconstruction complete.")

    return


@versioned_function("v1.0")
def reconstruct(
    experiment: str = None,
    process: str = None,
    input_path: str = None,
    config_path: str = None,
    output_path: str = None,
    example_fov: str = None,
    n_subtiles: int | None = None,  # make optional
    blend_pixels: int | None = None,  # make optional
    polynomial_fit_order: int | None = None,  # keep this argument
    midband_fractions: Tuple[float, float] = (0.125, 0.25),
    verbose: bool = True,
    debug_n_positions: int | None = None,
    debug_output_suffix: str = "_debug",
    ngff_version: str = "0.4",
):
    """
    Wrapper around wave0rder CLI functions

    Reconstruct quantitative phase images from live-cell brightfield z-stacks

    Note:
        - Replace the 3d reconstruction with a 2d reconstruction for tracking dataset
    """
    print(f"Reconstructing {experiment} for {process}")

    if experiment is None:
        assert (
            input_path is not None
        ), "Input path must be provided if experiment is None"
        input_path = Path(input_path)
        assert (
            config_path is not None
        ), "Config path must be provided if experiment is None"
        config_path = Path(config_path)
        assert (
            output_path is not None
        ), "Output path must be provided if experiment is None"
        output_path = Path(output_path)
        assert (
            example_fov is not None
        ), "Example FOV must be provided if experiment is None"

    else:
        dataset = OpsDataset(experiment)

        # Route pheno-2d to autofocus flow by default (configurable via `autofocus`)
        if process == "pheno-2d":
            # set parameters
            n_subtiles = 256
            blend_pixels = 25
            enable_subpixel_precision = True
            polynomial_fit_order = 3
            enable_subpixel_precision = True

        elif process == "track-2d":
            n_subtiles = 16
            blend_pixels = 25
            enable_subpixel_precision = True
            polynomial_fit_order = 3

        if process == "pheno-2d" or process == "track-2d":
            reconstruct_2d_autofocus(
                experiment=experiment,
                process=process,
                verbose=verbose,
                debug_n_positions=debug_n_positions,
                debug_output_suffix=debug_output_suffix,
                n_subtiles=n_subtiles,
                blend_pixels=blend_pixels,
                enable_subpixel_precision=enable_subpixel_precision,
                polynomial_fit_order=polynomial_fit_order,
                midband_fractions=midband_fractions,
                ngff_version=ngff_version,
            )
            return

        if process == "track":
            input_path = dataset.store_paths["lc_5x_bf_corrected"]
            config_path = dataset.config_paths["lc_5x_phase_recon"]
            output_path = dataset.store_paths["lc_5x_phase"]
            example_fov = "A/1/006006"
        if process == "pheno":
            input_path = dataset.store_paths["lc_20x"]
            config_path = dataset.config_paths["lc_20x_phase_recon"]
            output_path = dataset.store_paths["lc_20x_phase"]
            example_fov = "A/1/020020"
        if process == "20x_beads":
            input_path = dataset.store_paths["lc_20x_beads"]
            config_path = dataset.config_paths["lc_20x_phase_recon"]
            output_path = dataset.store_paths["lc_20x_beads_phase"]
            example_fov = "0/0/0"

    reconstruct_3d(
        dataset=dataset,
        process=process,
        example_fov=example_fov,
        input_path=input_path,
        config_path=config_path,
        output_path=output_path,
        debug_n_positions=debug_n_positions,
        debug_output_suffix=debug_output_suffix,
        verbose=verbose,
        ngff_version=ngff_version,
    )
    return


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run reconstruction utilities directly (debug-friendly)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # reconstruct subcommand
    r = subparsers.add_parser(
        "reconstruct", help="Run phase reconstruction for track/track-2d/pheno/pheno-2d"
    )
    r.add_argument(
        "--experiment",
        type=str,
        help="Experiment name (e.g., ops0042_20250520). If omitted, direct-path mode is used.",
    )
    r.add_argument(
        "--process",
        type=str,
        choices=["track", "track-2d", "pheno", "pheno-2d", "20x_beads"],
        help="Process type for dataset-scoped paths.",
    )
    r.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Verbose logging for autofocus and reconstruction steps (default).",
    )
    r.add_argument(
        "--quiet",
        dest="verbose",
        action="store_false",
        help="Reduce logging verbosity.",
    )
    r.set_defaults(verbose=True)

    r.add_argument(
        "--n-subtiles",
        dest="n_subtiles",
        type=int,
        default=25,
        help="Number of subtiles (must be perfect square: 4, 9, 16, 25, etc.). Default: 9",
    )
    r.add_argument(
        "--blend-pixels",
        dest="blend_pixels",
        type=int,
        default=25,
        help="Number of pixels to blend at subtile borders for seamless stitching. Default: 10",
    )

    r.set_defaults(enable_subpixel_precision=False)
    r.add_argument(
        "--polynomial-fit-order",
        dest="polynomial_fit_order",
        type=int,
        default=2,
        help="Polynomial order for sub-pixel focus fitting (default: 2).",
    )
    r.add_argument(
        "--midband-fractions",
        dest="midband_fractions",
        type=float,
        nargs=2,
        default=[0.125, 0.25],
        help="Low and high fractions of max frequency for bandpass (default: 0.125 0.25).",
    )

    # Debug options
    _add_common_debug_args(r)

    # Direct-path mode (no experiment)
    r.add_argument("--input-path", type=str, help="Direct path to input zarr")
    r.add_argument(
        "--config-path", type=str, help="Direct path to Waveorder config YAML"
    )
    r.add_argument("--output-path", type=str, help="Direct path to output zarr")
    r.add_argument(
        "--example-fov",
        type=str,
        help="Example position path (e.g. A/1/020020) for transfer function compute.",
    )

    # correct_distortion subcommand (experiment-scoped)
    cd = subparsers.add_parser(
        "correct_distortion",
        help="Run distortion correction for 5x brightfield (experiment-scoped)",
    )
    cd.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment name (e.g., ops0042_20250520)",
    )
    cd.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override auto-detected CPU workers.",
    )
    _add_common_debug_args(cd)

    return parser


def _add_common_debug_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--debug-n-positions",
        type=int,
        default=None,
        help="Debug: run only the first N positions and write to a debug output store.",
    )
    p.add_argument(
        "--debug-output-suffix",
        type=str,
        default="_debug",
        help="Suffix for the debug output store name.",
    )


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.command == "reconstruct":
        if args.experiment:
            from cyclops_utils.data.filesystem import resolve_experiment_name
            args.experiment = resolve_experiment_name(args.experiment, autoselect=True)
            reconstruct(
                experiment=args.experiment,
                process=args.process,
                verbose=args.verbose,
                debug_n_positions=args.debug_n_positions,
                debug_output_suffix=args.debug_output_suffix,
                n_subtiles=args.n_subtiles,
                blend_pixels=args.blend_pixels,
                polynomial_fit_order=args.polynomial_fit_order,
                midband_fractions=tuple(args.midband_fractions),
            )
        else:
            # Direct-path mode requires all four paths
            if not (
                args.input_path
                and args.config_path
                and args.output_path
                and args.example_fov
            ):
                parser.error(
                    "Direct-path mode requires --input-path, --config-path, --output-path, and --example-fov."
                )
            reconstruct(
                experiment=None,
                input_path=Path(args.input_path),
                config_path=Path(args.config_path),
                output_path=Path(args.output_path),
                example_fov=args.example_fov,
                verbose=args.verbose,
                debug_n_positions=args.debug_n_positions,
                debug_output_suffix=args.debug_output_suffix,
                n_subtiles=args.n_subtiles,
                blend_pixels=args.blend_pixels,
                polynomial_fit_order=args.polynomial_fit_order,
                midband_fractions=tuple(args.midband_fractions),
            )
        return
    if args.command == "correct_distortion":
        from cyclops_utils.data.filesystem import resolve_experiment_name
        args.experiment = resolve_experiment_name(args.experiment, autoselect=True)
        correct_distortion(
            experiment=args.experiment,
            num_workers=args.num_workers,
            debug_n_positions=args.debug_n_positions,
            debug_output_suffix=args.debug_output_suffix,
        )
        return


if __name__ == "__main__":
    main()

# Example CLI usages:
# - Standard dataset-scoped reconstruction:
#  python -m cyclops_process.processes.reconstruct reconstruct --experiment ops0060_20250724 --process pheno-2d

# - Debug mode with subtiles and sub-pixel precision focus detection
#  python -m cyclops_process.processes.reconstruct reconstruct --experiment ops0060_20250724 --process pheno-2d --debug-n-positions 5
# python -m cyclops_process.processes.reconstruct reconstruct --experiment ops0020_20250306 --process pheno-2d --debug-n-positions 5
# python -m cyclops_process.processes.reconstruct reconstruct --experiment ops0063_20250731 --process pheno-2d --midband-fractions 0.125 0.99 --debug-n-positions 50