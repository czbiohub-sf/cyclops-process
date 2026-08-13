import os
import time

import numpy as np
from tqdm import tqdm
from typing import List
from pathlib import Path
from collections import defaultdict, Counter
from joblib import Parallel, delayed


import dask.array as da
import matplotlib.pyplot as plt
from iohub.ngff import open_ome_zarr, TransformationMeta
# dexp (-> cupy/GPU) is imported lazily at its use sites (_compute_tile_shifts,
# tile_sabilization_model) so this module imports cleanly in dexp-less envs (CI).

from ops_utils.data.experiment import OpsDataset
from ops_utils.profiling.decorators import versioned_function
from stitch.registration.register import read_transform_biahub, register
from ops_utils.data.filesystem import (
    ensure_output_path,
    async_delete_path,
)
from ops_utils.hpc.resource_manager import get_optimal_workers
from ops_utils.io.zarr_utils import (
    _validate_output_images,
    _resolve_output_path_for_debug,
    _discover_positions_fast_balanced,
    _discover_positions,
    _ensure_store_position,
)
from ops_utils.hpc.gpu_utils import _setup_gpu_environment
from ops_utils.io.zarr_precreate import create_hcs_store_fast

# Import CuPy backend from stitch package (already has CuPy properly configured)
from stitch.stitch.tile import xp, cundi, _USING_CUPY


def _validated_chunks(chunks, Y: int, X: int, store_path) -> tuple:
    """Validate chunk YX against array YX before creating a zarr store.

    Why: dataset.store_props["chunk_size"] used to be globally overridden by the
    ISS tile_size (e.g. 2304 on bioe.ops2), which leaked into non-ISS stores
    written at 2048×2048. The mismatch produced .zarray files where chunks > shape,
    and zarr later failed with `cannot reshape array of size N into shape (...)`.
    The override was removed in experiment.py, but this guard fails loudly if
    any caller passes mismatched chunks again.
    """
    if len(chunks) < 2 or chunks[-2] > Y or chunks[-1] > X:
        raise ValueError(
            f"chunks YX ({chunks[-2]}, {chunks[-1]}) exceed array YX ({Y}, {X}) "
            f"when creating {store_path}. This would produce an unreadable zarr "
            f"(chunk size > array size). Likely cause: a non-ISS chunk_size was "
            f"set from an ISS-only tile_size override. Use chunks derived from "
            f"the actual array shape."
        )
    return tuple(chunks)


# --- Module-level worker function for Dask (must be picklable) ---

def _compute_tile_shifts(store_path, pos):
    """Compute drift shift vectors for a single tile position.

    Runs in a Dask worker process. Returns numpy array of shape (n_rounds-1, 2)
    containing (x, y) shift vectors.
    """
    import numpy as np
    import dask.array as da
    from iohub.ngff import open_ome_zarr
    from dexp.processing.registration.sequence import image_stabilisation
    from dexp.utils.backends.best_backend import BestBackend

    store = open_ome_zarr(store_path, mode="r")
    data = da.from_array(store[pos].data)
    summed_data = data[:, 1:, 0, :, :].sum(axis=1).compute()

    with BestBackend():
        sequence_model = image_stabilisation(
            image=summed_data[:, :, :].astype(float), axis=0, max_range=7
        )
    store.close()

    # Extract shift vectors as plain numpy (picklable)
    shifts = np.array([
        [m.shift_vector[0], m.shift_vector[1]]
        for m in sequence_model.model_list
    ])
    return shifts


def _process_position_chunk(chunk, input_path, output_path, well_drifts_np,
                            pad, well_paddings_py, order):
    """Process a chunk of positions with pipelined read/GPU/write.

    Within one Dask worker process, threads overlap I/O with GPU:
      [read1][GPU1+read2][write1+GPU2+read3][write2+GPU3+read4]...
    zarr I/O and CuPy both release the GIL, so they genuinely overlap.
    """
    import time
    import numpy as np
    import cupy as cp
    import cupyx.scipy.ndimage as cundi_local
    from iohub.ngff import open_ome_zarr
    from concurrent.futures import ThreadPoolExecutor
    import numcodecs
    numcodecs.blosc.set_nthreads(4)

    # Open stores once for the whole chunk
    fov_store = open_ome_zarr(input_path, mode="r")
    output_store = open_ome_zarr(output_path, layout="hcs", mode="r+")

    def _read(pos):
        return np.array(fov_store[pos].data)

    def _write(pos, data):
        output_store[pos]["0"][:] = data

    def _skip_check(pos):
        try:
            existing = output_store[pos]
            if "0" in existing.array_keys():
                test = existing["0"][0, 0, 0, :10, :10]
                if test.max() > 0:
                    return True
        except (KeyError, ValueError):
            pass
        return False

    def _gpu_shift(fov_np, well):
        mean_drift_np = well_drifts_np[well]
        mean_drift = cp.asarray(mean_drift_np)
        shifted_image_list = []
        batch_size = 5
        for batch_start in range(0, mean_drift.shape[0], batch_size):
            batch_end = min(batch_start + batch_size, mean_drift.shape[0])
            batch_rounds = list(range(batch_start, batch_end))
            batch_images = [cp.asarray(fov_np[r]) for r in batch_rounds]
            if pad:
                padding_xy = well_paddings_py.get(well)
                if padding_xy is not None:
                    px, py = padding_xy
                    batch_images = [
                        cp.pad(img, pad_width=((0, 0), (0, 0), px, py))
                        for img in batch_images
                    ]
            for idx, r in enumerate(batch_rounds):
                shift = (0, 0, mean_drift[r, 0], mean_drift[r, 1])
                shifted = cp.expand_dims(
                    cundi_local.shift(batch_images[idx], shift=shift, order=order), axis=0
                )
                shifted_image_list.append(shifted)
        return cp.concatenate(shifted_image_list, axis=0).get()

    read_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="read")
    write_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="write")

    all_timings = []
    skipped = 0
    write_future = None  # Track the outstanding write

    # Pre-submit first read
    first_pos, first_well = chunk[0]
    pending_read = read_pool.submit(_read, first_pos)

    for i, (pos, well) in enumerate(chunk):
        t_total_start = time.time()

        # Skip check
        if _skip_check(pos):
            skipped += 1
            # Still need to consume the pending read and start next
            pending_read.result()
            if i + 1 < len(chunk):
                pending_read = read_pool.submit(_read, chunk[i + 1][0])
            all_timings.append((pos, well, {"skipped": True}))
            continue

        # Wait for pre-read to complete
        t0 = time.time()
        fov_np = pending_read.result()
        t_read = time.time() - t0

        # Start pre-reading next position (overlaps with GPU + write)
        if i + 1 < len(chunk):
            pending_read = read_pool.submit(_read, chunk[i + 1][0])

        # GPU processing (main thread)
        t0 = time.time()
        out = _gpu_shift(fov_np, well)
        t_gpu = time.time() - t0
        del fov_np

        # Wait for previous write to finish before starting new one
        t0 = time.time()
        if write_future is not None:
            write_future.result()
        t_write_wait = time.time() - t0

        # Submit async write (overlaps with next read + GPU)
        t_write_start = time.time()
        write_future = write_pool.submit(_write, pos, out)

        all_timings.append((pos, well, {
            "read": t_read, "gpu_total": t_gpu,
            "write_wait": t_write_wait,
            "total": time.time() - t_total_start,
        }))

    # Wait for final write
    if write_future is not None:
        write_future.result()

    read_pool.shutdown(wait=True)
    write_pool.shutdown(wait=True)
    fov_store.close()
    output_store.close()

    return all_timings


@versioned_function("v1.0")
def correct_cycle_drift(
    experiment: str,
    pad: bool = False,
    fast: bool = True,
    overwrite: bool | None = None,
) -> None:
    """
    Args:
        experiment (str):
            Experiment name i.e. ops{num}_{YYYMMDD}
        padding (bool):
            Default False
        fast (bool):
            Default True
            If True, shifts to nearest pixel (fast)
            If False, interpolates sub-pixel shifts (slow)
        overwrite (bool | None):
            Override overwrite behavior. If None (default), uses interactive prompts.
            If True, force overwrite. If False, skip if output exists.

    Notes:
    - could be improved in the future by caclulating drfit for each tile, rather than applying the mean
    """
    dataset = OpsDataset(experiment)
    fov_store = open_ome_zarr(dataset.store_paths["iss"], mode="r")
    position_list = [a[0] for a in fov_store.positions()]

    grouped_positions = defaultdict(list)
    for p in position_list:
        group = p[:3]
        grouped_positions[group].append(p)

    output_path = dataset.store_paths["iss_drift_corrected"]

    if overwrite is False and output_path.exists():
        print(f"Skipping cycle drift correction (overwrite=False for {output_path}).")
        return

    # Always rebuild — resume kept the old store's metadata, so a store written
    # before a round was added stayed short a round. Async delete: rename is
    # instant, the rm -rf runs detached.
    if output_path.exists():
        print(f"Removing existing drift-corrected store (async): {output_path}")
        async_delete_path(output_path)
    mode = "w"

    if fast:
        order = 0
    else:
        order = 1

    def fprint(*args, **kwargs):
        """Print with immediate flush."""
        print(*args, **kwargs, flush=True)

    # Start Dask cluster — used for both drift computation and position processing.
    from dask.distributed import LocalCluster, Client

    NUM_WORKERS = 30

    fprint(f"Starting Dask cluster ({NUM_WORKERS} workers)...")
    t_cluster_start = time.time()
    cluster = LocalCluster(n_workers=NUM_WORKERS, threads_per_worker=1)
    client = Client(cluster)
    fprint(f"  Cluster ready in {time.time() - t_cluster_start:.1f}s. "
           f"Dashboard: {client.dashboard_link}")

    # Compute drift for all wells+tiles in parallel via Dask workers.
    # 5 tiles per well × 3 wells = 15 tiles, all submitted at once.
    # Each worker opens its own zarr store (true multiprocessing parallelism).
    t_drift_start = time.time()
    input_path_str = str(dataset.store_paths["iss"])
    wells = list(grouped_positions.keys())
    tile_offsets = ["002007", "007002", "007007", "007012", "012007"]
    fprint(f"Computing drift for {len(wells)} wells ({len(tile_offsets)} tiles each) in parallel...")

    drift_futures = []
    for g in wells:
        for tile in tile_offsets:
            pos = f"{g}/{tile}"
            f = client.submit(_compute_tile_shifts, input_path_str, pos)
            drift_futures.append((f, g, pos))

    # Collect results, group by well, compute mean drift + plot
    well_shift_lists = defaultdict(list)
    for f, g, pos in drift_futures:
        shifts = f.result(timeout=300)  # shape (n_rounds-1, 2)
        well_shift_lists[g].append(shifts)

    well_drifts = {}
    well_padding = {}
    for g in wells:
        shift_arrays = well_shift_lists[g]  # list of (n, 2) arrays
        # Plot and compute mean drift (same logic as plot_drift)
        fig, ax = plt.subplots()
        all_list = []
        for i, shifts in enumerate(shift_arrays):
            ax.plot(shifts[:, 0], shifts[:, 1], "o-", label=f"tile {i}")
            all_list.append(xp.expand_dims(xp.asarray(shifts), 0))
        temp = xp.concatenate(all_list, axis=0)
        mean_drift = xp.mean(temp, axis=0)
        ax.plot(mean_drift[:, 0].get(), mean_drift[:, 1].get(),
                "o--", label="mean drift", color="red")
        ax.legend()
        ax.set_xlabel("X Shift (pixels)")
        ax.set_ylabel("Y Shift (pixels)")
        plt.savefig(dataset.append_well("iss_cycle_drift", g), dpi=300)
        plt.close(fig)

        max_drift = xp.max(xp.abs(xp.cumsum(mean_drift, axis=0)))
        fprint(f"  Well {g}: max drift = {max_drift:.1f} px")
        well_drifts[g] = mean_drift

        if pad:
            well_padding[g] = (
                (
                    xp.abs(xp.clip(xp.floor(xp.min(mean_drift[:, 0])).astype(int), None, 0)),
                    xp.clip(xp.ceil(xp.max(mean_drift[:, 0])).astype(int), 0, None),
                ),
                (
                    xp.abs(xp.clip(xp.floor(xp.min(mean_drift[:, 1])).astype(int), None, 0)),
                    xp.clip(xp.ceil(xp.max(mean_drift[:, 1])).astype(int), 0, None),
                ),
            )

    t_drift_elapsed = time.time() - t_drift_start
    fprint(f"Drift computation: {t_drift_elapsed:.1f}s ({t_drift_elapsed/60:.1f}m)")

    # Pre-create position structures for ALL wells
    all_positions = [(pos, g) for g, positions in grouped_positions.items() for pos in positions]
    position_names = [pos for pos, _ in all_positions]

    # Get shape from first position (all positions have same shape before padding)
    source_shape = fov_store[position_names[0]].data.shape
    source_dtype = fov_store[position_names[0]].data.dtype

    t_precreate_start = time.time()
    if pad:
        # With padding, shapes differ per well — create per-well
        fprint(f"Pre-creating {len(all_positions)} positions (padded) across {len(grouped_positions)} wells...")
        for i, (g, positions) in enumerate(grouped_positions.items()):
            padding_x, padding_y = well_padding[g]
            padded_shape = (
                source_shape[0],
                source_shape[1],
                source_shape[2],
                source_shape[3] + padding_x[0] + padding_x[1],
                source_shape[4] + padding_y[0] + padding_y[1]
            )
            create_hcs_store_fast(
                store_path=output_path,
                positions=positions,
                shape=padded_shape,
                chunks=dataset.store_props["chunk_size"],
                dtype=source_dtype,
                scale=dataset.store_props["5x_scale"],
                channel_names=fov_store.channel_names,
                version="0.5",
                mode=mode if i == 0 else "a",
            )
    else:
        fprint(f"Pre-creating {len(all_positions)} positions across {len(grouped_positions)} wells...")
        create_hcs_store_fast(
            store_path=output_path,
            positions=position_names,
            shape=source_shape,
            chunks=dataset.store_props["chunk_size"],
            dtype=source_dtype,
            scale=dataset.store_props["5x_scale"],
            channel_names=fov_store.channel_names,
            version="0.5",
        )
    t_precreate_elapsed = time.time() - t_precreate_start
    fprint(f"Pre-created structures in {t_precreate_elapsed:.2f}s")

    # Setup GPU environment
    available_gpus = _setup_gpu_environment()

    # --- Dask Multiprocessing Architecture (v3) ---
    # Each Dask worker is a separate process with its own GIL.
    # Memory bounded: each worker holds 1 position (~460MB) at a time.

    # Convert drift arrays to numpy for pickling across processes
    well_drifts_np = {}
    for g, drift in well_drifts.items():
        well_drifts_np[g] = drift.get() if hasattr(drift, 'get') else np.array(drift)

    # Convert padding tuples to plain Python types for pickling
    well_padding_py = {}
    if pad:
        for g, (px, py) in well_padding.items():
            well_padding_py[g] = (
                (int(px[0]), int(px[1])),
                (int(py[0]), int(py[1])),
            )

    output_path_str = str(output_path)

    # Split positions into chunks — one chunk per worker with internal pipelining
    chunk_size = (len(all_positions) + NUM_WORKERS - 1) // NUM_WORKERS
    chunks = [all_positions[i:i + chunk_size] for i in range(0, len(all_positions), chunk_size)]

    fprint(f"Processing {len(all_positions)} positions across {len(grouped_positions)} wells "
           f"with {NUM_WORKERS} pipelined Dask workers ({len(chunks)} chunks of ~{chunk_size})")

    t_parallel_start = time.time()

    try:
        # Submit one chunk per worker — each worker pipelines read/GPU/write internally
        futures = []
        for chunk in chunks:
            f = client.submit(
                _process_position_chunk,
                chunk, input_path_str, output_path_str,
                well_drifts_np, pad, well_padding_py, order,
            )
            futures.append(f)

        # Collect results from all workers
        all_timings = defaultdict(list)
        well_processed = Counter()
        well_skipped = Counter()
        well_total = Counter()
        completed = 0
        skipped = 0

        for f in futures:
            chunk_timings = f.result(timeout=600)
            for pos, well, timings in chunk_timings:
                well_total[well] += 1
                if timings.get("skipped"):
                    skipped += 1
                    well_skipped[well] += 1
                else:
                    for phase, t in timings.items():
                        all_timings[phase].append(t)
                    well_processed[well] += 1
                    completed += 1

        fprint(f"  All done: {completed} processed, {skipped} skipped")

    finally:
        try:
            client.close()
            cluster.close(timeout=120)
        except Exception:
            pass

    t_parallel_elapsed = time.time() - t_parallel_start

    # Report results with per-phase timing breakdown
    n = max(1, len(all_timings.get("total", [1])))
    fprint(f"\n=== PROFILING: Per-position timing breakdown (avg over {n} positions) ===")
    for phase in ["read", "gpu_total", "write_wait", "total"]:
        vals = all_timings.get(phase, [])
        if vals:
            avg = sum(vals) / len(vals)
            total = sum(vals)
            pct = (total / sum(all_timings.get("total", [1]))) * 100 if phase != "total" else 100
            fprint(f"  {phase:>12s}: avg={avg:.3f}s  total={total:.1f}s  ({pct:.1f}%)")

    for g in grouped_positions.keys():
        fprint(f"Well {g}: processed {well_processed[g]}/{well_total[g]} positions "
               f"(skipped {well_skipped.get(g, 0)})")
    fprint(f"Total: {sum(well_processed.values())} positions in {t_parallel_elapsed:.1f}s ({t_parallel_elapsed/60:.1f}m)")

    return


def tile_sabilization_model(
    pos: str,
    store,
):
    from dexp.processing.registration.sequence import image_stabilisation

    data = da.from_array(store[pos].data)
    # sum all channles so that all spots appear in each frame
    summed_data = data[:, 1:, 0, :, :].sum(axis=1).compute()

    # Use dexp's GPU backend for pairwise registration computation
    with BestBackend():
        sequence_model = image_stabilisation(
            image=summed_data[:, :, :].astype(float), axis=0, max_range=7
        )

    return sequence_model

def plot_drift(sequence_models: list, dataset, well: str) -> None:
    all_list = []
    fig, ax = plt.subplots()
    for i in range(len(sequence_models)):
        x_list = []
        y_list = []
        seq_model_list = sequence_models[i].model_list
        for j in range(len(seq_model_list)):
            x_list.append(seq_model_list[j].shift_vector[0])
            y_list.append(seq_model_list[j].shift_vector[1])
        ax.plot(x_list, y_list, "o-", label=f"tile {i}")

        all_list.append(xp.expand_dims(xp.asarray([x_list, y_list]).T, 0))

    temp = xp.concatenate(all_list, axis=0)
    # mean_drift = xp.mean(temp, axis=0).get()
    mean_drift = xp.mean(temp, axis=0)
    ax.plot(
        mean_drift[:, 0].get(),
        mean_drift[:, 1].get(),
        "o--",
        label="mean drift",
        color="red",
    )
    ax.legend()
    ax.set_xlabel("X Shift (pixels)")
    ax.set_ylabel("Y Shift (pixels)")
    plt.savefig(dataset.append_well("iss_cycle_drift", well), dpi=300)

    return mean_drift


def compare_tiles(store, dataset, well: str) -> list:
    """Sample 5 tiles from each of the 4 quadrents of the welll and the center"""

    # this is speficic to iss, will fail if applied to either lc dataset
    pos_list = [
        f"{well}/002007",
        f"{well}/007002",
        f"{well}/007007",
        f"{well}/007012",
        f"{well}/012007",
    ]

    # Parallelize tile stabilization computation (was sequential bottleneck)
    model_list = Parallel(n_jobs=5, backend="threading")(
        delayed(tile_sabilization_model)(pos=pos, store=store)
        for pos in pos_list
    )

    mean_drifts = plot_drift(model_list, dataset, well)

    return mean_drifts


@versioned_function("v2.0")
def register_fluor_2d_tiles(
    experiment: str = None,
    bf_path: str = None,
    phase_path: str = None,
    cell_seg_path: str = None,
    output_path: str = None,
    gfp_config_path: str = None,
    mcherry_config_path: str = None,
    cy5_config_path: str = None,
    normalize: bool = True,
    debug_n_positions: int | None = None,
    debug_output_suffix: str = "_debug",
) -> None:
    """
    Register fluorescent channels to the phase contrast image
    Args:
        experiment (str):
            Experiment name i.e. ops{num}_{YYYMMDD}

    """
    print(
        f"Registering fluorescent channels to the phase contrast image for {experiment}"
    )
    if experiment is None:
        if bf_path is None or phase_path is None or cell_seg_path is None:
            raise ValueError(
                "Either experiment or bf/phase/cell_seg paths must be provided."
            )
        bf_fov_path = Path(bf_path)
        output_path = Path(output_path)
        gfp_config_path = Path(gfp_config_path) if gfp_config_path is not None else None
        mcherry_config_path = (
            Path(mcherry_config_path) if mcherry_config_path is not None else None
        )
        cy5_config_path = (
            Path(cy5_config_path) if cy5_config_path is not None else None
        )
        output_path = _resolve_output_path_for_debug(
            output_path, debug_n_positions, debug_output_suffix
        )
        dataset = OpsDataset("")

    else:
        dataset = OpsDataset(experiment)
        # Use flatfield-corrected 2D tiles as input (already max-projected + corrected)
        flatfield_path = dataset.store_paths["lc_20x_fluor_2d_flatfield"]
        if flatfield_path.exists():
            bf_fov_path = flatfield_path
            print(f"  Using flatfield-corrected fluor tiles: {bf_fov_path}")
        else:
            bf_fov_path = dataset.store_paths["lc_20x"]
            print(f"  WARNING: Flatfield-corrected store not found, falling back to raw 3D: {bf_fov_path}")
        gfp_config_path = dataset.config_paths["lc_GFP_register"]
        mcherry_config_path = dataset.config_paths["lc_mCherry_register"]
        cy5_config_path = dataset.config_paths["lc_Cy5_register"]

        output_path = dataset.store_paths["lc_20x_fluor_2d_registered"]
        output_path = _resolve_output_path_for_debug(
            output_path, debug_n_positions, debug_output_suffix
        )

    bf_fov_store = open_ome_zarr(bf_fov_path, mode="r")
    channel_names = bf_fov_store.channel_names
    channel_names = ["Phase" if s == "BF" else s for s in channel_names]

    gfp_affine = None
    mcherry_affine = None
    cy5_affine = None

    fluor_channels_present = []
    if "GFP" in channel_names:
        fluor_channels_present.append("GFP")
        if gfp_config_path is not None and gfp_config_path.exists():
            gfp_affine = read_transform_biahub(gfp_config_path)
        else:
            raise FileNotFoundError(
                f"GFP channel found but registration transform missing at {gfp_config_path}. "
                f"Run the submit_channel_registration_jobs step to generate fluor "
                f"registration transforms."
            )
    if "mCherry" in channel_names:
        fluor_channels_present.append("mCherry")
        if mcherry_config_path is not None and mcherry_config_path.exists():
            mcherry_affine = read_transform_biahub(mcherry_config_path)
        else:
            raise FileNotFoundError(
                f"mCherry channel found but registration transform missing at {mcherry_config_path}. "
                f"Run the submit_channel_registration_jobs step to generate fluor "
                f"registration transforms."
            )
    if "Cy5" in channel_names:
        fluor_channels_present.append("Cy5")
        if cy5_config_path is not None and cy5_config_path.exists():
            cy5_affine = read_transform_biahub(cy5_config_path)
        else:
            raise FileNotFoundError(
                f"Cy5 channel found but registration transform missing at {cy5_config_path}. "
                f"Run the submit_channel_registration_jobs step to generate fluor "
                f"registration transforms."
            )

    # Determine output fluor channel names to match actual data written
    out_channel_names = []
    if gfp_affine is not None:
        out_channel_names.append("GFP")
    if mcherry_affine is not None:
        out_channel_names.append("mCherry")
    if cy5_affine is not None:
        out_channel_names.append("Cy5")
    if len(out_channel_names) == 0:
        if fluor_channels_present:
            raise RuntimeError(
                f"Fluorescent channels {fluor_channels_present} found but no transforms loaded. "
                f"This should not happen — check registration config paths."
            )
        return None

    # Discover positions (debug-aware) without walking entire store
    if debug_n_positions is not None and int(debug_n_positions) > 0:
        position_list = _discover_positions_fast_balanced(
            Path(bf_fov_path), int(debug_n_positions)
        )
        print(f"Debug mode: {len(position_list)} positions")
    else:
        position_list = _discover_positions(Path(bf_fov_path))
        print(f"Full mode: {len(position_list)} positions")

    # Rebuild from scratch: every position is re-registered below.
    async_delete_path(output_path)
    # Get metadata from first position
    first_pos = position_list[0]
    fov = bf_fov_store[first_pos]
    ref = fov["0"] if "0" in fov.array_keys() else fov.data
    Y0, X0 = int(ref.shape[-2]), int(ref.shape[-1])

    chunks = _validated_chunks(dataset.store_props["chunk_size"], Y0, X0, output_path)
    # Precreate all positions at once
    create_hcs_store_fast(
        store_path=output_path,
        positions=position_list,
        shape=(1, len(out_channel_names), 1, Y0, X0),
        chunks=chunks,
        dtype=xp.float32,
        scale=dataset.store_props["20x_scale"],
        channel_names=out_channel_names,
        # v2 (0.4): prepare_unified_pheno_tiles symlinks this store's v2
        # nested chunks (0/t/c/z) into the unified tiles. A v3 store's
        # chunks live under 0/c/... (sharded) and can't be symlinked per
        # channel, so the fluor channels would come through blank.
        version="0.4",
    )

    # Parallel worker to process a single position
    def _process_pos(pos: str):
        bf_fov = bf_fov_store[pos].data
        channel_stack_list = []
        if gfp_affine is not None:
            gfp_ind = channel_names.index("GFP")
            gfp_fov_max_proj = xp.max(bf_fov[0, gfp_ind, :, :, :], axis=0)
            gfp_array = xp.expand_dims(xp.asarray(gfp_fov_max_proj), axis=0)
            gfp_assembled = cundi.affine_transform(
                gfp_array.astype(xp.float32, copy=False),
                xp.asarray(gfp_affine, dtype=xp.float32),
                order=3,
                mode="constant",
                cval=0.0,
                output_shape=gfp_array.shape,
            )
            channel_stack_list.append(gfp_assembled)
        if mcherry_affine is not None:
            mcherry_ind = channel_names.index("mCherry")
            mcherry_fov_max_proj = xp.max(bf_fov[0, mcherry_ind, :, :, :], axis=0)
            mcherry_array = xp.expand_dims(xp.asarray(mcherry_fov_max_proj), axis=0)
            mcherry_assembled = cundi.affine_transform(
                mcherry_array.astype(xp.float32, copy=False),
                xp.asarray(mcherry_affine, dtype=xp.float32),
                order=0,
                mode="constant",
                cval=0.0,
                output_shape=mcherry_array.shape,
            )
            channel_stack_list.append(mcherry_assembled)
        if cy5_affine is not None:
            cy5_ind = channel_names.index("Cy5")
            cy5_fov_max_proj = xp.max(bf_fov[0, cy5_ind, :, :, :], axis=0)
            cy5_array = xp.expand_dims(xp.asarray(cy5_fov_max_proj), axis=0)
            cy5_assembled = cundi.affine_transform(
                cy5_array.astype(xp.float32, copy=False),
                xp.asarray(cy5_affine, dtype=xp.float32),
                order=0,
                mode="constant",
                cval=0.0,
                output_shape=cy5_array.shape,
            )
            channel_stack_list.append(cy5_assembled)
        out = xp.expand_dims(xp.stack(channel_stack_list, axis=0), axis=(0)).get()
        # Write at position level (skip plate metadata parsing)
        with open_ome_zarr(output_path / pos, layout="fov", mode="r+") as dst_ds:
            dst_ds.data[:] = out

    # Determine workers and run in parallel using threads (keeps CuPy context in-process)
    n_jobs = get_optimal_workers(use_gpu=False, verbose=False)
    Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_process_pos)(pos)
        for pos in tqdm(position_list, desc="Registering positions")
    )

    return None


def build_unified_pheno_tiles_symlink(
    experiment: str,
    fluor_src_store_path: Path | None = None,
    debug_n_positions: int | None = None,
    debug_output_suffix: str = "_debug",
) -> Path:
    """CLI: assemble unified Phase2D+Fluor2D+VS tiles by symlinking, WITHOUT
    re-registering fluor (optionally from an override fluor store). Thin
    wrapper over the shared builder prepare_unified_pheno_tiles."""
    prepare_unified_pheno_tiles(
        experiment=experiment,
        debug_n_positions=debug_n_positions,
        debug_output_suffix=debug_output_suffix,
        register_fluor=False,
        fluor_src_store_path=fluor_src_store_path,
    )
    up = OpsDataset(experiment).store_paths["pheno_tiles_unified"]
    return _resolve_output_path_for_debug(up, debug_n_positions, debug_output_suffix)


@versioned_function("v1.0")
def prepare_unified_pheno_tiles(
    experiment: str,
    debug_n_positions: int | None = None,
    debug_output_suffix: str = "_debug",
    register_fluor: bool = True,
    fluor_src_store_path: Path | None = None,
) -> None:
    """Build the unified phase+fluor+VS tiles by symlinking source chunks.

    Single shared builder for both the orchestrator DAG step and the
    ``build_unified_pheno_tiles_symlink`` CLI wrapper.

    register_fluor: when True (orchestrator default) (re)register fluor tiles
        first; set False for the CLI "symlink-only" mode.
    fluor_src_store_path: optional override for the registered-fluor source
        (CLI); defaults to the dataset's lc_20x_fluor_2d_registered.
    """
    print(f"Preparing unified phenotyping tiles for {experiment}")
    # 1) Register fluor tiles if present; otherwise skip gracefully
    dataset = OpsDataset(experiment)
    bf_store = open_ome_zarr(dataset.store_paths["lc_20x"], mode="r")
    bf_channel_names = list(bf_store.channel_names)
    has_fluor_input = ("GFP" in bf_channel_names) or ("mCherry" in bf_channel_names) or ("Cy5" in bf_channel_names)
    if register_fluor and has_fluor_input:
        register_fluor_2d_tiles(
            experiment=experiment,
            debug_n_positions=debug_n_positions,
            debug_output_suffix=debug_output_suffix,
        )
    elif not register_fluor:
        print("Skipping fluor registration (register_fluor=False).")
    else:
        print("No fluorescent channels detected; skipping fluor registration.")
    # 2) Build unified tiles sequentially (initialize, then link/write per-pos)
    dataset = OpsDataset(experiment)
    phase_store_path = dataset.store_paths["lc_20x_phase_2d_optimized"]
    vs_store_path = dataset.store_paths["lc_20x_vs_max_proj"]
    fluor_store_path = fluor_src_store_path or dataset.store_paths["lc_20x_fluor_2d_registered"]
    has_fluor_store = fluor_store_path.exists()
    if not has_fluor_store:
        print(f"No registered fluor store found at {fluor_store_path}; proceeding without fluorescent channels.")
    # Align fluor store to debug-suffixed output when in debug mode (default
    # store only; an explicit override is used as-is).
    if fluor_src_store_path is None and has_fluor_store and debug_n_positions is not None and int(debug_n_positions) > 0:
        fluor_store_path = _resolve_output_path_for_debug(
            fluor_store_path, debug_n_positions, debug_output_suffix
        )
    # cell_seg_store_path = dataset.store_paths["lc_20x_segmentation_cells"]

    phase_store = open_ome_zarr(phase_store_path)
    vs_store = open_ome_zarr(vs_store_path)
    # cell_seg_store = open_ome_zarr(cell_seg_store_path)
    phase_channels = list(phase_store.channel_names)
    try:
        fluor_store = open_ome_zarr(fluor_store_path)
        fluor_channels = list(fluor_store.channel_names)
        has_fluor = True
        print(f"Fluor channels: {fluor_channels}")
    except Exception:
        fluor_channels = []
        has_fluor = False
    vs_channels = list(vs_store.channel_names)

    # Discover positions (debug-aware) - must do this before validation
    if debug_n_positions is not None and int(debug_n_positions) > 0:
        try:
            positions = _discover_positions_fast_balanced(
                Path(phase_store_path), int(debug_n_positions)
            )
        except Exception:
            positions = _discover_positions(Path(phase_store_path))[
                : int(debug_n_positions)
            ]
    else:
        positions = _discover_positions(Path(phase_store_path))

    unified_tiles_path = dataset.store_paths["pheno_tiles_unified"]
    unified_tiles_path = _resolve_output_path_for_debug(
        unified_tiles_path, debug_n_positions, debug_output_suffix
    )
    # Always rebuild from scratch — no resume. This step only precreates
    # metadata and symlinks chunk dirs (cheap, idempotent), so there's no
    # expensive work to resume. The all-zeros "resume" heuristic actively
    # harms here: a freshly-precreated store looks identical to a blank or
    # wrong-version store, so resume would keep a broken scaffold in place
    # instead of recreating it correctly.
    # Async delete so we don't block on a slow NFS unlink of the
    # multi-thousand-position symlink tree.
    async_delete_path(unified_tiles_path)

    first_pos = next(phase_store.positions())[0]
    T, _, _, Y, X = phase_store[first_pos]["0"].shape
    scale = phase_store[first_pos].scale

    total_channels = (
        len(phase_channels)
        + (len(fluor_channels) if has_fluor else 0)
        + len(vs_channels)
    )
    print(f"Total channels: {total_channels}")

    # Precreate all positions (metadata + empty arrays) in one write session.
    create_hcs_store_fast(
        store_path=unified_tiles_path,
        positions=positions,
        shape=(T, total_channels, 1, Y, X),
        chunks=dataset.store_props["chunk_size"],
        dtype=xp.float32,
        scale=scale,
        channel_names=phase_channels + fluor_channels + vs_channels,
        # v2 (0.4): _link_source symlinks v2 nested chunks (0/t/c/z) from the
        # v2 source stores (phase/fluor/VS). A v3 (0.5) array reads chunks
        # under a different key scheme and would see the symlinked data as
        # all-zero. Keep this store's version matched to the symlink layout.
        version="0.4",
    )

    # Write/Link per position sequentially (no further metadata opens)
    phase_do_aug = False
    print(
        f"[UnifiedTiles][Config] phase_aug={phase_do_aug}; has_fluor={has_fluor} ({fluor_channels})"
    )
    print(
        "[UnifiedTiles][Note] assemble.py and create_morphology_dataset apply VS augment (fliplr=True, rot90=1). "
        + (
            "We will write augmented VS (phase_aug path)."
            if phase_do_aug
            else "Currently VS is symlinked unchanged."
        )
    )
    # Link Phase (if not augmented), Fluor, VS symlinks
    import os as _os
    from pathlib import Path as _P

    for pos in tqdm(positions, desc="Linking unified tiles"):

        def _link_source(store_path: Path, c_src: int, c_dst: int):
            for t_idx in range(int(T)):
                src_dir = (
                    _P(str(store_path))
                    / _P(pos)
                    / "0"
                    / str(int(t_idx))
                    / str(int(c_src))
                    / "0"
                )
                dst_dir = (
                    _P(str(unified_tiles_path))
                    / _P(pos)
                    / "0"
                    / str(int(t_idx))
                    / str(int(c_dst))
                    / "0"
                )
                try:
                    dst_dir.parent.mkdir(parents=True, exist_ok=True)
                    if dst_dir.exists() or dst_dir.is_symlink():
                        try:
                            if dst_dir.is_symlink():
                                dst_dir.unlink()
                            else:
                                import shutil as _sh

                                _sh.rmtree(dst_dir)
                        except Exception:
                            pass
                    _os.symlink(src_dir, dst_dir, target_is_directory=True)
                except Exception as e:
                    print(
                        f"[UnifiedTiles] Failed to link {pos} t{t_idx} c{c_src}→{c_dst}: {e}"
                    )

        # Phase symlinks only when not augmented
        if not phase_do_aug:
            for i in range(len(phase_channels)):
                _link_source(phase_store_path, i, i)
        # Fluor
        base = len(phase_channels)
        if has_fluor and fluor_channels:

            for i in range(len(fluor_channels)):
                _link_source(Path(fluor_store_path), i, base + i)
        # VS symlinks only when not augmented
        base2 = len(phase_channels) + (len(fluor_channels) if has_fluor else 0)
        if not phase_do_aug:
            for i in range(len(vs_channels)):
                _link_source(vs_store_path, i, base2 + i)

        # seg image is not present in unified tiles

    return


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified phenotyping tiles builder (phase+fluor+VS)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # prepare_unified_pheno_tiles subcommand
    ppt = subparsers.add_parser(
        "prepare_unified_pheno_tiles",
        help="End-to-end: register fluor (if present) and assemble unified phase+fluor+VS tiles",
    )
    ppt.add_argument("--experiment", type=str, required=True, help="Experiment name")
    ppt.add_argument(
        "--debug-n-positions",
        type=int,
        default=None,
        help="Limit to first N positions and write to *_debug store",
    )
    ppt.add_argument(
        "--debug-output-suffix",
        type=str,
        default="_debug",
        help="Suffix for debug output store",
    )

    bu = subparsers.add_parser(
        "build_unified_pheno_tiles_symlink",
        help="Assemble unified Phase2D+Fluor2D+VS tiles by symlinking (no fluor registration)",
    )
    bu.add_argument("--experiment", type=str, required=True, help="Experiment name")
    bu.add_argument(
        "--fluor-src-store-path",
        type=str,
        default=None,
        help="Override fluor_2d_registered store path",
    )
    bu.add_argument(
        "--phase-flipud",
        action="store_true",
        help="Flip phase vertically before linking (not supported; will error)",
    )
    bu.add_argument(
        "--phase-fliplr",
        action="store_true",
        help="Flip phase horizontally before linking (not supported; will error)",
    )
    bu.add_argument(
        "--phase-rot90",
        type=int,
        default=0,
        help="Rotate phase by k*90 before linking (not supported; will error)",
    )
    bu.add_argument(
        "--debug-n-positions",
        type=int,
        default=None,
        help="Limit to first N positions and write to *_debug store",
    )
    bu.add_argument(
        "--debug-output-suffix",
        type=str,
        default="_debug",
        help="Suffix for debug output store",
    )

    up = subparsers.add_parser(
        "build_phase_family_tiles_symlink",
        help="Build unified tiles from Phase2D + Fluor2D + VS",
    )
    up.add_argument("--experiment", type=str, required=True, help="Experiment name")
    # Orientation options
    up.add_argument(
        "--invert-affine",
        action="store_true",
        help="Invert the estimated affine before applying (swap source/target)",
    )
    # Debug options
    up.add_argument(
        "--debug-n-positions",
        type=int,
        default=None,
        help="Limit to first N positions and write to *_debug store",
    )
    up.add_argument(
        "--debug-output-suffix",
        type=str,
        default="_debug",
        help="Suffix for debug output store",
    )

    # register_stitched_fluor_to_phase subcommand
    rs = subparsers.add_parser(
        "register_stitched_fluor_to_phase",
        help="Register stitched fluorescence mosaic into stitched phase canvas (supports debug limiting)",
    )
    rs.add_argument("--experiment", type=str, required=True, help="Experiment name")
    rs.add_argument(
        "--phase-store-path",
        type=str,
        default=None,
        help="Override phase store path (assembled phase)",
    )
    rs.add_argument(
        "--fluor-store-path",
        type=str,
        default=None,
        help="Override fluor store path (assembled fluorescence)",
    )
    rs.add_argument(
        "--output-store-path",
        type=str,
        default=None,
        help="Override output assembled store path",
    )
    rs.add_argument(
        "--gfp-config-path",
        type=str,
        default=None,
        help="Path to GFP affine YAML (per well)",
    )
    rs.add_argument(
        "--mcherry-config-path",
        type=str,
        default=None,
        help="Path to mCherry affine YAML (per well)",
    )
    rs.add_argument(
        "--invert-affine",
        action="store_true",
        help="Invert the 4x4 affine before applying (try if alignment is flipped)",
    )
    rs.add_argument(
        "--swap-xy",
        action="store_true",
        help="Swap X and Y axes before/after transform (try if axes are flipped)",
    )
    # Manual affine debug options
    rs.add_argument(
        "--manual-dy", type=float, default=None, help="Manual translation in Y (pixels)"
    )
    rs.add_argument(
        "--manual-dx", type=float, default=None, help="Manual translation in X (pixels)"
    )
    rs.add_argument(
        "--manual-zoom",
        type=float,
        default=None,
        help="Manual uniform zoom/scale factor",
    )
    rs.add_argument(
        "--manual-rot-deg",
        type=float,
        default=None,
        help="Manual in-plane rotation in degrees",
    )
    rs.add_argument(
        "--debug-n-positions",
        type=int,
        default=None,
        help="Limit to first N positions and write to *_debug store",
    )
    rs.add_argument(
        "--debug-output-suffix",
        type=str,
        default="_debug",
        help="Suffix for debug output store",
    )
    rs.add_argument(
        "--debug-n-chunks",
        type=int,
        default=None,
        help="Limit number of YX chunks processed per (t,z) for quick debugging",
    )

    # register_fluor_2d_tiles subcommand
    rf = subparsers.add_parser(
        "register_fluor_2d_tiles",
        help="Register 2D fluor tiles into Phase2D canvas (dataset or direct paths)",
    )
    rf.add_argument("--experiment", type=str, help="Experiment name")
    # Direct-path mode (no experiment)
    rf.add_argument(
        "--bf-path", type=str, default=None, help="Direct BF tiles store path (20x)"
    )
    rf.add_argument(
        "--phase-path", type=str, default=None, help="Direct Phase2D store path (20x)"
    )
    rf.add_argument(
        "--cell-seg-path",
        type=str,
        default=None,
        help="Direct pheno_cells segmentation store path",
    )
    rf.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Direct output path for fluor_2d_registered",
    )
    rf.add_argument(
        "--gfp-config-path", type=str, default=None, help="Path to GFP affine YAML"
    )
    rf.add_argument(
        "--mcherry-config-path",
        type=str,
        default=None,
        help="Path to mCherry affine YAML",
    )
    # Options
    rf.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable post-registration normalization",
    )
    rf.add_argument(
        "--debug-n-positions",
        type=int,
        default=None,
        help="Limit to first N positions and write to *_debug store",
    )
    rf.add_argument(
        "--debug-output-suffix",
        type=str,
        default="_debug",
        help="Suffix for debug output store",
    )

    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.command == "prepare_unified_pheno_tiles":
        prepare_unified_pheno_tiles(
            experiment=args.experiment,
            debug_n_positions=args.debug_n_positions,
            debug_output_suffix=args.debug_output_suffix,
        )
        return

    if args.command == "build_unified_pheno_tiles_symlink":
        build_unified_pheno_tiles_symlink(
            experiment=args.experiment,
            fluor_src_store_path=(
                Path(args.fluor_src_store_path) if args.fluor_src_store_path else None
            ),
            debug_n_positions=args.debug_n_positions,
            debug_output_suffix=args.debug_output_suffix,
        )
        return

    if args.command == "register_fluor_2d_tiles":
        normalize = not bool(getattr(args, "no_normalize", False))
        if args.experiment:
            register_fluor_2d_tiles(
                experiment=args.experiment,
                normalize=normalize,
                debug_n_positions=getattr(args, "debug_n_positions", None),
                debug_output_suffix=getattr(args, "debug_output_suffix", "_debug"),
            )
        else:
            # Direct-path mode requires BF/phase/cell_seg/output
            if not (
                args.bf_path
                and args.phase_path
                and args.cell_seg_path
                and args.output_path
            ):
                parser.error(
                    "Direct-path mode requires --bf-path, --phase-path, --cell-seg-path, and --output-path."
                )
            register_fluor_2d_tiles(
                experiment=None,
                bf_path=Path(args.bf_path),
                phase_path=Path(args.phase_path),
                cell_seg_path=Path(args.cell_seg_path),
                output_path=Path(args.output_path),
                gfp_config_path=(
                    Path(args.gfp_config_path) if args.gfp_config_path else None
                ),
                mcherry_config_path=(
                    Path(args.mcherry_config_path) if args.mcherry_config_path else None
                ),
                normalize=normalize,
                debug_n_positions=getattr(args, "debug_n_positions", None),
                debug_output_suffix=getattr(args, "debug_output_suffix", "_debug"),
            )
        return


if __name__ == "__main__":
    main()
    # usaage: python -m cyclops_process.processes.register prepare_unified_pheno_tiles --experiment ops0033_20250429 --debug-n-positions 16 --debug-output-suffix _debug
    # usage: python -m cyclops_process.processes.register register_stitched_fluor_to_phase --experiment ops0033_20250429 --debug-n-positions 1 --debug-output-suffix _debug --debug-n-chunks 16
    # usage debug stitched zarrrs:
    # python -m cyclops_process.processes.register register_stitched_fluor_to_phase --experiment ops0033_20250429 \
    #      --phase-store-path /path/to/data/ops0033_20250429/1-preprocess/live_imaging/stitch/pheno_phase_stitched_debug.zarr \
    #      --fluor-store-path /path/to/data/ops0033_20250429/1-preprocess/live_imaging/stitch/pheno_fluor_stitched_debug.zarr \
    #      --output-store-path /path/to/data/ops0033_20250429/3-assembly/phenotyping_debug.zarr \
    #      --debug-n-positions 1 --debug-output-suffix _debug --debug-n-chunks 16

    # usage: python -m cyclops_process.processes.register build_phase_family_tiles_symlink --experiment ops0033_20250429 --debug-n-positions 10 --debug-output-suffix _debug
    # usage: python -m cyclops_process.processes.register register_fluor_2d_tiles --experiment ops0033_20250429 --debug-n-positions 10 --debug-output-suffix _debug
    # usage: python -m cyclops_process.processes.register prepare_unified_pheno_tiles --experiment ops0033_20250429 --debug-n-positions 10 --debug-output-suffix _debug
