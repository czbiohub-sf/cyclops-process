import os

# CRITICAL: Set BLAS threading limits BEFORE importing numpy/scipy
# This prevents over-subscription when using ThreadPoolExecutor
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import time
import click
import warnings
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, List
from joblib import Parallel, delayed
import re

import scipy
import numpy as np
import pandas as pd
import dask.array as da
import tensorstore as ts
from iohub.ngff import open_ome_zarr
import yaml

from cyclops_process.bc import extract
from ops_utils.data.experiment import OpsDataset
from ops_utils.profiling.decorators import versioned_function
from ops_utils.hpc.resource_manager import get_optimal_workers


warnings.filterwarnings("ignore")

# Experiments with only 9 physical ISS rounds (indices 0-8) instead of the default 10
EXPERIMENTS_WITH_9_ROUNDS = [42]


def max_filter(data: np.array, width: int = 3, n_threads: int = 8) -> np.array:
    """Apply a maximum filter in a window of `width`.

    Args:
        data: Input array with shape (rounds, channels, y, x)
        width: Filter window size
        n_threads: Number of threads for parallel processing

    Returns:
        Filtered array with same shape as input
    """
    from concurrent.futures import ThreadPoolExecutor

    T, C, Y, X = data.shape

    # Pre-allocate output array
    maxed = np.empty_like(data)

    def process_slice(t, c):
        """Process a single (round, channel) 2D slice."""
        slice_2d = data[t, c, :, :]
        filtered_2d = scipy.ndimage.filters.maximum_filter(slice_2d, size=(width, width))
        return t, c, filtered_2d

    # Process all (round, channel) combinations in parallel
    # BLAS threading controlled at SLURM job level via environment variables
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        import threading
        futures = []
        for t in range(T):
            for c in range(C):
                futures.append(executor.submit(process_slice, t, c))

        # Check thread count while executor is active
        active_threads = threading.active_count()
        if active_threads > 1:
            print(f"[DEBUG max_filter ACTIVE] {active_threads} threads while processing")

        # Collect results
        for future in futures:
            t, c, filtered_2d = future.result()
            maxed[t, c, :, :] = filtered_2d

    return maxed


from ops_utils.io.tiling import split_into_tiles


def split_into_fixed_tiles(arr_shape: Tuple, tile_size: int = 4096) -> List[Tuple[int]]:
    """Split array into fixed-size tiles that align with zarr chunk boundaries.

    Args:
        arr_shape: (height, width) of the array
        tile_size: Size of each tile (default 4096 to match zarr chunks)

    Returns:
        List of (row_start, row_stop, col_start, col_stop) tuples
    """
    tiles = []
    height, width = arr_shape

    row_starts = list(range(0, height, tile_size))
    col_starts = list(range(0, width, tile_size))

    for row_start in row_starts:
        row_stop = min(row_start + tile_size, height)
        for col_start in col_starts:
            col_stop = min(col_start + tile_size, width)
            tiles.append((row_start, row_stop, col_start, col_stop))

    return tiles


def filter_points(tile: Tuple, points: np.array) -> np.array:
    x_min, x_max, y_min, y_max = tile
    points_in_tile = points[
        (points[:, 0] >= x_min)
        & (points[:, 0] < x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] < y_max)
    ]
    return points_in_tile - np.array([x_min, y_min])


WELL = "well"
TILE = "tile"
CELL = "cell"
READ = "read"
BARCODE = "barcode"
CHANNEL = "channel"
CYCLE = "cycle"
INTENSITY = "intensity"


def bf_normalize_bases(df):
    """Normalize the channel intensities by the median brightness of each channel in all spots.

    Args:
        df (pandas.DataFrame): DataFrame containing spot intensity data.

    Returns:
        pandas.DataFrame: DataFrame with normalized intensity values.
    """
    # Calculate median brightness of each channel
    df_medians = df.groupby("channel").intensity.median()

    # Vectorized normalization - map channel to median, then divide
    df_out = df.copy()
    df_out.intensity = df_out.intensity / df_out.channel.map(df_medians)

    return df_out


def get_iss_rounds_for_well(
    well_name: str,
    default_rounds: List[int],
    failed_rounds_by_well: dict | None = None,
) -> List[int]:
    """Get the effective ISS rounds for a well, handling failed rounds.

    Args:
        well_name: Well identifier (e.g., "A/1/0")
        default_rounds: Default list of ISS rounds to use
        failed_rounds_by_well: Optional dict mapping well names to failed round specs
            Supports: simple list [0, 3] or dict {"dropout": [0], "offset": [5]}

    Returns:
        List of ISS round indices to use for this well
    """
    from cyclops_process.metrics.plate_stats.match_reads import _get_effective_iss_rounds

    rounds = _get_effective_iss_rounds(default_rounds, well_name, failed_rounds_by_well)

    if rounds != default_rounds:
        print(
            f"Well {well_name}: iss_rounds={rounds} (from default={default_rounds})"
        )

    return rounds


def _base_calling_params_from_config(experiment: str) -> tuple:
    """Read ``base_calling_params`` (iss_rounds, failed_rounds_by_well) from the
    generated experiment config. Returns (None, None) when unavailable."""
    import yaml

    try:
        cfg_path = OpsDataset(experiment).config_paths["exp_config"]
        if not cfg_path.exists():
            return None, None
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"WARNING: could not read base_calling_params from config: {e}")
        return None, None

    params = cfg.get("base_calling_params") or {}
    return params.get("iss_rounds"), params.get("failed_rounds_by_well")


def _assert_rounds_available(
    iss_rounds: List[int], n_available: int, experiment: str, source_desc: str
) -> None:
    """Fail fast when requested rounds exceed what the source array holds —
    otherwise it surfaces as an IndexError inside a tile worker."""
    missing = [r for r in iss_rounds if r >= n_available or r < -n_available]
    if not missing:
        return
    raise ValueError(
        f"{experiment}: base calling requested iss_rounds={list(iss_rounds)} but "
        f"{source_desc} holds only {n_available} rounds (missing {missing}). "
        f"Either the store is short a round (check stack_symlinks "
        f"pre_nuclei_round/skip_pre_dapi_round and that correct_cycle_drift + "
        f"stitch were rebuilt after it) or iss_rounds in the experiment config "
        f"is wrong."
    )


@versioned_function("v1.0")
def base_calling(
    experiment: str,
    method: str = "mine",
    debug: bool = False,
    debug_num_spots: int = None,
    iss_rounds: List[int] | None = None,
    failed_rounds_by_well: dict | None = None,
    n_rounds: int = 9,
    read_length: int | None = None,  # DEPRECATED: use iss_rounds instead
    *,
    well_pos: str | None = None,
    registered_shm_name: str | None = None,
    registered_shm_shape: tuple | None = None,
) -> None:
    """
    Assign nucleotides based on intensity values

    Args:
        experiment (str):
            Experiment name i.e. ops{num}_{YYYMMDD}
        method (str):
            the normalization method to use. very important becuase the raw
            intensities from each channel are not directly comparable
        debug (bool):
            If True, prints debugging information.
        debug_num_spots (int, optional):
            If provided, runs analysis on only the first N spots for debugging. Defaults to None.
        iss_rounds (List[int], optional):
            List of ISS round indices to use (e.g., [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] for rounds 1-10).
            Allows skipping failed rounds or processing non-contiguous rounds.
        failed_rounds_by_well (dict, optional):
            Dict mapping well names to lists of failed round indices to exclude.
            Example: {"A/1/0": [3, 7], "A/2/0": [9]} excludes rounds 3,7 from well A1 and round 9 from A2.
        read_length (int, optional):
            DEPRECATED: Use iss_rounds instead. Number of sequencing rounds to analyze.
        well_pos (str, optional):
            If set (e.g. "A/1/0"), only process tiles from that single well. Used by
            the per-well ISS merge step.
        registered_shm_name (str, optional):
            Name of a caller-allocated ``multiprocessing.shared_memory`` block backing
            a float32 ndarray of shape ``registered_shm_shape``, holding the
            registered (T, C, Z, Y, X) data for the well in ``well_pos``. When set,
            tile workers read their data from shm instead of opening
            ``bc_stitched_registered.zarr`` via TensorStore — used by the merge
            step to avoid re-reading what's already in host RAM.
        registered_shm_shape (tuple, optional):
            Shape of the shared-memory ndarray. Required when ``registered_shm_name``
            is set.
    """
    print(f"Running base calling using {method} method")

    # Limit BLAS/MKL threading to prevent over-subscription
    # We control parallelism via ThreadPoolExecutor, so BLAS should be single-threaded
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

    # Handle backward compatibility with deprecated parameters
    if iss_rounds is None:
        if read_length is not None:
            print("WARNING: read_length is deprecated. Use iss_rounds instead.")
            iss_rounds = list(range(read_length))
            print(f"Converting read_length={read_length} to iss_rounds={iss_rounds}")
        else:
            # Config wins over the hardcoded default — the merge step doesn't
            # thread base_calling_params through.
            cfg_rounds, cfg_failed = _base_calling_params_from_config(experiment)
            if cfg_rounds is not None:
                iss_rounds = list(cfg_rounds)
                print(f"Using iss_rounds={iss_rounds} from experiment config")
            else:
                # No config entry: rounds 0-9, or 0-8 for the 9-round experiments
                exp_match = re.search(r'ops(\d+)', experiment)
                exp_number = int(exp_match.group(1)) if exp_match else None
                n_rounds = 9 if exp_number in EXPERIMENTS_WITH_9_ROUNDS else 10
                iss_rounds = list(range(n_rounds))
                # iss_rounds = list(range(n_rounds + 1)) # (aliddell): beta/nextflow change, keeping for review
            if failed_rounds_by_well is None and cfg_failed:
                failed_rounds_by_well = cfg_failed
                print(f"Using failed_rounds_by_well from experiment config")

    print(f"Using iss_rounds={iss_rounds}")
    if failed_rounds_by_well:
        print(f"Failed rounds by well specified: {failed_rounds_by_well}")

    # With 48 CPUs allocated:
    # - TensorStore: 15 workers × 6 threads = 90 threads
    # - max_filter: 15 workers × 8 threads = 120 threads (oversubscribed but sequential with TensorStore)
    # - Method section: 15 workers × 3 BLAS threads = 45 threads
    num_workers = 15
    num_tensorstore_threads = 6  # TensorStore I/O threads (zarr mode only)
    num_max_filter_threads = 2 if registered_shm_name is not None else 8
    num_blas_threads = 3  # Method section BLAS threads
    print(f"Starting base calling for {experiment} using {num_workers} workers with {num_tensorstore_threads} I/O threads, {num_max_filter_threads} compute threads, and {num_blas_threads} BLAS threads")

    dataset = OpsDataset(experiment)
    codebook_df = dataset.load_codebook()
    stitched_path = dataset.store_paths["iss_stitch_registered_v3"]
    segmentation_path = dataset.store_paths["iss_segmentation"]

    # Validate shm args
    if registered_shm_name is not None:
        if registered_shm_shape is None:
            raise ValueError("registered_shm_shape is required when registered_shm_name is set")
        if well_pos is None:
            raise ValueError(
                "registered_shm_name requires well_pos — the in-memory mode is per-well"
            )

    # 1. Create a master list of all tiles to process. When well_pos is given,
    #    only that well's tiles are queued.
    all_tiles_to_process = []
    stitch_position_list = []
    if well_pos is not None:
        # Use the registered zarr shape (or shm shape) to determine tile bounds.
        # If we're in shm mode, prefer the shm shape; otherwise read from the zarr.
        if registered_shm_shape is not None:
            shape_yx = (registered_shm_shape[-2], registered_shm_shape[-1])
            _assert_rounds_available(
                iss_rounds, int(registered_shm_shape[0]), experiment,
                f"the registered array for {well_pos} (shape {tuple(registered_shm_shape)})",
            )
        else:
            with open_ome_zarr(stitched_path / well_pos, layout="fov", mode="r") as fov_ds:
                shape_yx = fov_ds.data.shape[-2:]
                _assert_rounds_available(
                    iss_rounds, int(fov_ds.data.shape[0]), experiment,
                    f"{stitched_path / well_pos} (shape {tuple(fov_ds.data.shape)})",
                )
        tile_list, _ = split_into_tiles(shape_yx, 5, 0)
        for tile in tile_list:
            all_tiles_to_process.append((well_pos, tile))
        stitch_position_list = [well_pos]
    else:
        stitch_position_list = []
        with open_ome_zarr(stitched_path, mode="r") as stitch_ds:
            stitch_position_list = [a[0] for a in stitch_ds.positions()]
            for pos in stitch_position_list:
                _assert_rounds_available(
                    iss_rounds, int(stitch_ds[pos].data.shape[0]), experiment,
                    f"{stitched_path / pos} (shape {tuple(stitch_ds[pos].data.shape)})",
                )
                shape = stitch_ds[pos].data.shape[-2:]
                tile_list, _ = split_into_tiles(shape, 5, 0)
                for tile in tile_list:
                    all_tiles_to_process.append((pos, tile))

    def _process_tile(pos_tile_tuple):
        """Worker function to process a single tile from a single position."""
        import time as perf_time
        import os
        import threading
        t_tile_start = perf_time.time()

        pos, tile = pos_tile_tuple

        # DEBUG: Print environment and thread info at worker startup (once per worker)
        if not hasattr(_process_tile, '_debug_printed'):
            _process_tile._debug_printed = True
            print(f"[DEBUG WORKER {os.getpid()}] Environment variables:")
            print(f"  OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'NOT SET')}")
            print(f"  MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS', 'NOT SET')}")
            print(f"  OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS', 'NOT SET')}")
            print(f"  NUMEXPR_NUM_THREADS={os.environ.get('NUMEXPR_NUM_THREADS', 'NOT SET')}")
            print(f"[DEBUG WORKER {os.getpid()}] Active thread count: {threading.active_count()}")

        # Always call bases for ALL requested rounds — failed round handling
        # is a matching concern (link_calls_tracks / match_reads), not a
        # calling concern. Skipping rounds here produces shorter barcodes
        # that break downstream assumptions.
        effective_rounds = iss_rounds

        # DEBUG: Thread count before opening stores
        threads_start = threading.active_count()
        print(f"[THREAD DEBUG {pos}] Start: {threads_start} threads")

        # Open segmentation store at position level with layout="fov" to skip plate metadata parsing.
        # stitch_ds is not needed here - TensorStore handles stitched data reads directly.
        t_io_start = perf_time.time()
        segmentation_ds = open_ome_zarr(segmentation_path / pos, layout="fov", mode="r")
        t_io_open = perf_time.time() - t_io_start
        threads_after_open = threading.active_count()
        if threads_after_open != threads_start:
            print(f"[THREAD DEBUG {pos}] After open zarr: {threads_start} -> {threads_after_open} threads")

        t_load_spots_start = perf_time.time()
        points = np.load(dataset.append_well("spots", pos))
        t_load_spots = perf_time.time() - t_load_spots_start
        threads_after_load = threading.active_count()
        if threads_after_load != threads_after_open:
            print(f"[THREAD DEBUG {pos}] After load spots: {threads_after_open} -> {threads_after_load} threads")

        if debug_num_spots is not None and debug_num_spots > 0:
            print(
                f"--- DEBUG MODE: Analyzing only the first {debug_num_spots} spots for position {pos}. ---"
            )
            points = points[:debug_num_spots]

        tile_points = filter_points(tile, points)

        if tile_points.shape[0] == 0:
            return None

        row_start, row_stop, col_start, col_stop = tile

        print(f"[START] Tile {pos} {tile} - {len(tile_points)} spots")

        # Read image data — either from the caller's in-memory shm (merge step)
        # or via TensorStore from the stitched-registered zarr (legacy path).
        t_read_data_start = perf_time.time()
        if registered_shm_name is not None:
            # In-memory: attach the shared-memory block, slice, copy out.
            from multiprocessing import shared_memory as _shm

            shm_handle = _shm.SharedMemory(name=registered_shm_name)
            try:
                arr = np.ndarray(
                    registered_shm_shape, dtype=np.float32, buffer=shm_handle.buf,
                )
                data = np.ascontiguousarray(
                    arr[effective_rounds, :, 0, row_start:row_stop, col_start:col_stop],
                    dtype=np.float32,
                )
            finally:
                # Drop the numpy view so close() doesn't warn about active refs.
                arr = None  # noqa: F841
                shm_handle.close()
        else:
            threads_before_tensorstore = threading.active_count()
            print(f"[THREAD DEBUG {pos}] Before TensorStore: {threads_before_tensorstore} threads")

            # Import TensorStore locally to avoid global thread pool inheritance
            import tensorstore as ts

            # TensorStore path: zarr_root/position/0 (the "0" is the OME-Zarr array name)
            zarr_array_path = f"{stitched_path}/{pos}/0"

            # Pick the driver by store format: OME-NGFF 0.5 is zarr v3 (zarr.json),
            # legacy 0.4 is zarr v2 (.zarray). The registered store is now written
            # v3-native, but auto-detecting keeps this read working on legacy v2 stores.
            ts_driver = "zarr3" if Path(zarr_array_path, "zarr.json").exists() else "zarr"

            # Open and read with TensorStore using multiple threads for parallel decompression
            store = ts.open({
                'driver': ts_driver,
                'kvstore': {
                    'driver': 'file',
                    'path': zarr_array_path,
                },
                'context': {
                    'cache_pool': {'total_bytes_limit': 100000000},
                    'data_copy_concurrency': {'limit': num_tensorstore_threads}  # Threads for I/O, sequential with max_filter
                }
            }).result()

            # Read with slicing - TensorStore handles decompression and I/O efficiently
            data = store[effective_rounds, :, 0, row_start:row_stop, col_start:col_stop].read().result()
            data = np.asarray(data).astype("float32")

            threads_after_tensorstore = threading.active_count()
            print(f"[THREAD DEBUG {pos}] After TensorStore read: {threads_before_tensorstore} -> {threads_after_tensorstore} threads")

            # Force close TensorStore to clean up threads
            store = None

        t_read_data = perf_time.time() - t_read_data_start
        print(f"  → Tile {pos} {tile} - Read data: {t_read_data:.1f}s "
              f"({'shm' if registered_shm_name is not None else 'zarr'})")

        threads_before_max_filter = threading.active_count()
        print(f"[THREAD DEBUG {pos}] Before max_filter: {threads_before_max_filter} threads")
        t_max_filter_start = perf_time.time()
        fov_max = max_filter(data, width=10, n_threads=num_max_filter_threads)
        t_max_filter = perf_time.time() - t_max_filter_start
        threads_after_max_filter = threading.active_count()
        print(f"[THREAD DEBUG {pos}] After max_filter: {threads_after_max_filter} threads")
        print(f"  → Tile {pos} {tile} - Max filter: {t_max_filter:.1f}s")

        t_read_nuclei_start = perf_time.time()
        nuclei_tile = segmentation_ds["1"][
            0, 0, 0, row_start:row_stop, col_start:col_stop
        ]
        t_read_nuclei = perf_time.time() - t_read_nuclei_start
        threads_after_nuclei = threading.active_count()
        if threads_after_nuclei != threads_after_max_filter:
            print(f"[THREAD DEBUG {pos}] After read nuclei: {threads_after_max_filter} -> {threads_after_nuclei} threads")

        t_extract_start = perf_time.time()
        bases_df = extract.extract_reads(fov_max, tile_points, nuclei_tile)
        t_extract = perf_time.time() - t_extract_start
        threads_after_extract = threading.active_count()
        if threads_after_extract != threads_after_nuclei:
            print(f"[THREAD DEBUG {pos}] After extract_reads: {threads_after_nuclei} -> {threads_after_extract} threads")

        # Allow BLAS to use multiple threads during Method section.
        # In shm mode every per-tile threadpool_limits call invokes
        # dl_iterate_phdr under a process-wide loader lock; on contended
        # nodes that lock serializes all 4 outer threads and stalls
        # base_calling. Env already pins OMP/MKL/OPENBLAS=1, so the
        # inner BLAS limit is redundant — use a nullcontext instead.
        from threadpoolctl import threadpool_limits
        import contextlib
        if registered_shm_name is not None:
            def _blas_ctx():
                return contextlib.nullcontext()
        else:
            def _blas_ctx():
                return threadpool_limits(limits=num_blas_threads, user_api='blas')

        if method == "mine":
            t_normalize_start = perf_time.time()
            with _blas_ctx():
                bases_df = bf_normalize_bases(bases_df)
            t_normalize = perf_time.time() - t_normalize_start
            threads_after_normalize = threading.active_count()
            if threads_after_normalize != threads_after_extract:
                print(f"[THREAD DEBUG {pos}] After bf_normalize_bases: {threads_after_extract} -> {threads_after_normalize} threads")

            t_call_reads_start = perf_time.time()
            with _blas_ctx():
                reads_df = extract.call_reads(bases_df)
            t_call_reads = perf_time.time() - t_call_reads_start
            threads_after_call_reads = threading.active_count()
            if threads_after_call_reads != threads_after_normalize:
                print(f"[THREAD DEBUG {pos}] After call_reads: {threads_after_normalize} -> {threads_after_call_reads} threads")

            t_method = t_normalize + t_call_reads

        else:
            reads_df = None
            t_method = 0

        if reads_df is None or reads_df.empty:
            return None

        reads_df["i_global"] = reads_df["i"] + row_start
        reads_df["j_global"] = reads_df["j"] + col_start

        t_tile_total = perf_time.time() - t_tile_start
        print(f"[DONE] Tile {pos} {tile} - Total: {t_tile_total:.1f}s, {len(reads_df)} reads")

        # Print timing breakdown for every 10th tile or first 5 tiles
        tile_index = all_tiles_to_process.index((pos, tile))
        if tile_index < 5 or tile_index % 10 == 0:
            print(f"[Tile {pos} {tile}] Total: {t_tile_total:.3f}s | "
                  f"I/O open: {t_io_open:.3f}s | Load spots: {t_load_spots:.3f}s | "
                  f"Read data: {t_read_data:.3f}s | Max filter: {t_max_filter:.3f}s | "
                  f"Read nuclei: {t_read_nuclei:.3f}s | Extract: {t_extract:.3f}s | "
                  f"Method ({method}): {t_method:.3f}s | {len(reads_df)} reads")

        return pos, reads_df

    # 2. Run base calling in parallel on all tiles
    print(
        f"Applying base calling in parallel across {len(all_tiles_to_process)} tiles..."
    )
    # Force each worker to use max 4 threads by wrapping Parallel call with threadpool_limits
    from threadpoolctl import threadpool_limits

    if registered_shm_name is not None:
        # shm mode (merge step): use ThreadPoolExecutor instead of joblib processes.
        # All threads share the parent's single mapping of the shm-backed array,
        # so the kernel rmap has one entry — no TLB-shootdown / lock-contention
        # amplification that 15 separate processes mapping the same VMA produces
        # on contended nodes. Per-tile work releases the GIL during max_filter
        # (C), numpy/BLAS ops, and pandas — so 8 threads scale well in practice.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        # 4 outer threads × 2 max_filter inner threads = 8 worker threads max,
        # safely under our 32-CPU allocation. Higher counts caused thread
        # oversubscription on contended nodes (load avg 137 + 64 internal
        # threads from 8x8 = scheduler thrash and slow user-space progress).
        n_thread_workers = min(4, len(all_tiles_to_process))
        print(f"  shm mode: using ThreadPoolExecutor with {n_thread_workers} threads "
              f"(× {num_max_filter_threads}-thread max_filter = "
              f"{n_thread_workers * num_max_filter_threads} compute threads)")
        results = []
        with threadpool_limits(limits=4):
            with ThreadPoolExecutor(max_workers=n_thread_workers) as ex:
                futures = {ex.submit(_process_tile, pt): pt
                           for pt in all_tiles_to_process}
                for fut in tqdm(as_completed(futures),
                                total=len(futures),
                                desc="Processing all tiles"):
                    results.append(fut.result())
    else:
        # Production zarr-read path: keep joblib processes (TensorStore inside
        # each worker has its own thread pool, so process isolation matters).
        with threadpool_limits(limits=4):
            results = Parallel(n_jobs=num_workers)(
                delayed(_process_tile)(pos_tile)
                for pos_tile in tqdm(all_tiles_to_process, desc="Processing all tiles")
            )
    print("Parallel processing complete. Aggregating results...")

    # 3. Aggregate results
    pos_dfs = {pos: [] for pos in stitch_position_list}
    for result in results:
        if result is not None:
            pos, reads_df = result
            pos_dfs[pos].append(reads_df)

    # 6. Save aggregated results for each position
    for pos, df_list in pos_dfs.items():
        if not df_list:
            print(f"No reads found for position {pos}, creating empty file.")
            empty_cols = [
                "i",
                "j",
                "cell",
                "Q_min",
                "barcode",
                "gene_id",
                "i_global",
                "j_global",
            ]
            pos_df = pd.DataFrame(columns=empty_cols)
        else:
            pos_df = pd.concat(df_list, ignore_index=True)

        # Per-well reads path from OpsDataset.result_paths["reads"] (already under
        # base_calling/mine/ for the mine pipeline; do not insert another mine/).
        if method == "mine":
            output_path = dataset.append_well("reads", pos)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Other, experimental methods get a unique filename to avoid overwriting.
            reads_dir = Path(dataset.result_paths["reads"]).parent
            well_prefix = f"{pos[0]}{pos[2]}_"
            output_filename = f"reads_{method}_{well_prefix}.csv"
            output_path = reads_dir / output_filename

        pos_df.to_csv(output_path, index=False)
        print(f"Saved {len(pos_df)} reads for position {pos} to {output_path}")

    print("Base calling complete.")
    return
