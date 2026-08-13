from __future__ import annotations

import numpy as np
try:
    import torch
except ModuleNotFoundError:  # optional GPU dep; used only in GPU runtime paths
    torch = None
from iohub.ngff import open_ome_zarr
from scipy.spatial import KDTree
from pathlib import Path
import random
from cyclops_utils.data.experiment import OpsDataset
from typing import Callable, TYPE_CHECKING
from cyclops_utils.profiling.decorators import versioned_function
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

if TYPE_CHECKING:
    from spotiflow.model import Spotiflow


def normalize(x, mi, ma, eps: float = 1e-20):
    return ((x - mi) / (ma - mi + eps)).astype(np.float32)


def get_min_max(experiment, anchor_round: int = 0):
    dataset = OpsDataset(experiment)
    source_path = dataset.store_paths["iss"]

    print(f"Estimating min and max values for normalization (anchor round {anchor_round})")
    pos_list_list = []
    lanes = []
    with open_ome_zarr(source_path, mode="r+") as ds:
        # Discover wells (row/col) from positions, row-agnostic — not just row A.
        well_to_positions = {}
        for pos_path, _ in ds.positions():
            parts = pos_path.split("/")
            if len(parts) >= 2:
                well_to_positions.setdefault(f"{parts[0]}/{parts[1]}", []).append(pos_path)
        for well in sorted(well_to_positions):
            row, col = well.split("/")[:2]
            pos_list_list.append(well_to_positions[well])
            lanes.append(f"{row}{col}")

    keys = lanes
    p1_dict = {}
    p998_dict = {}
    for i, pos_list in enumerate(pos_list_list):
        if len(pos_list) == 0:
            continue
        num_samples = min(11, len(pos_list))
        sample = random.sample(pos_list, num_samples)
        p1_list = []
        p998_list = []
        for pos in sample:
            fov = ds[pos].data[anchor_round, 1:, 0, :, :]
            p1 = np.percentile(fov, 1, axis=(1, 2))
            p998 = np.percentile(fov, 99.8, axis=(1, 2))
            p1_list.append(p1)
            p998_list.append(p998)
        p1_array = np.array(p1_list)
        p998_array = np.array(p998_list)
        p1_median = np.median(p1_array, axis=0)
        p998_median = np.median(p998_array, axis=0)
        p1_dict[keys[i]] = p1_median
        p998_dict[keys[i]] = p998_median

    return p1_dict, p998_dict


def detect_points(
    fov: np.array,
    model: Spotiflow,
    min_distance: int = 5,
    prob_threshold: float = None,
    tiles: tuple = (10, 10),
    normalizer: Callable = None,
    p1: dict = None,
    p998: dict = None,
    verbose: bool = False,
) -> np.array:
    """
    helper function for detecting spots in a single fov
    """
    t_start = time.time()

    if p1 is not None:
        gp1, tp1, ap1, cp1 = p1
    if p998 is not None:
        gp998, tp998, ap998, cp998 = p998

    yxc_fov = fov

    # Process channels sequentially with detailed timing
    t_gpu_start = time.time()
    channel_names = ['G', 'T', 'A', 'C']
    channel_times = []

    if verbose:
        print(f"    Processing 4 channels sequentially...")

    t_ch_start = time.time()
    g_points, details = model.predict(
        yxc_fov[:, :, 0],
        min_distance=min_distance,
        prob_thresh=prob_threshold,
        n_tiles=tiles,
        normalizer=lambda x: normalizer(x, gp1, gp998),
        device=torch.device("cuda"),
        verbose=False,
    )
    t_ch_elapsed = time.time() - t_ch_start
    channel_times.append(t_ch_elapsed)
    if verbose:
        print(f"      Channel G: {t_ch_elapsed:.2f}s ({len(g_points)} spots)")

    t_ch_start = time.time()
    t_points, details = model.predict(
        yxc_fov[:, :, 1],
        min_distance=min_distance,
        prob_thresh=prob_threshold,
        n_tiles=tiles,
        normalizer=lambda x: normalizer(x, tp1, tp998),
        device=torch.device("cuda"),
        verbose=False,
    )
    t_ch_elapsed = time.time() - t_ch_start
    channel_times.append(t_ch_elapsed)
    if verbose:
        print(f"      Channel T: {t_ch_elapsed:.2f}s ({len(t_points)} spots)")

    t_ch_start = time.time()
    a_points, details = model.predict(
        yxc_fov[:, :, 2],
        min_distance=min_distance,
        prob_thresh=prob_threshold,
        n_tiles=tiles,
        normalizer=lambda x: normalizer(x, ap1, ap998),
        device=torch.device("cuda"),
        verbose=False,
    )
    t_ch_elapsed = time.time() - t_ch_start
    channel_times.append(t_ch_elapsed)
    if verbose:
        print(f"      Channel A: {t_ch_elapsed:.2f}s ({len(a_points)} spots)")

    t_ch_start = time.time()
    c_points, details = model.predict(
        yxc_fov[:, :, 3],
        min_distance=min_distance,
        prob_thresh=prob_threshold,
        n_tiles=tiles,
        normalizer=lambda x: normalizer(x, cp1, cp998),
        device=torch.device("cuda"),
        verbose=False,
    )
    t_ch_elapsed = time.time() - t_ch_start
    channel_times.append(t_ch_elapsed)
    if verbose:
        print(f"      Channel C: {t_ch_elapsed:.2f}s ({len(c_points)} spots)")

    t_gpu_elapsed = time.time() - t_gpu_start

    if verbose:
        print(f"    GPU total (4 channels sequential): {t_gpu_elapsed:.2f}s")

    # Combine and deduplicate points
    t_postprocess_start = time.time()

    t_vstack_start = time.time()
    all_points = np.vstack([g_points, t_points, a_points, c_points])
    t_vstack_elapsed = time.time() - t_vstack_start
    if verbose:
        print(f"    Vstack points: {t_vstack_elapsed:.3f}s ({len(all_points)} total points)")

    t_kdtree_start = time.time()
    point_tree = KDTree(all_points)
    t_kdtree_elapsed = time.time() - t_kdtree_start
    if verbose:
        print(f"    Build KDTree: {t_kdtree_elapsed:.3f}s")

    t_query_start = time.time()
    pairs = point_tree.query_pairs(min_distance, output_type="ndarray")
    t_query_elapsed = time.time() - t_query_start
    if verbose:
        print(f"    Query pairs: {t_query_elapsed:.3f}s ({len(pairs)} pairs)")

    t_dedup_start = time.time()
    index_to_remove = np.unique(pairs[:, 1])
    trimmed_points = np.delete(all_points, index_to_remove, axis=0)
    t_dedup_elapsed = time.time() - t_dedup_start
    if verbose:
        print(f"    Deduplicate: {t_dedup_elapsed:.3f}s (removed {len(index_to_remove)} points)")

    t_postprocess_elapsed = time.time() - t_postprocess_start
    t_total_elapsed = time.time() - t_start

    if verbose:
        print(f"    Postprocessing total: {t_postprocess_elapsed:.2f}s")
        print(f"    detect_points total: {t_total_elapsed:.2f}s")

    return trimmed_points


def detect_spots_in_image(
    image_yxc: np.ndarray,
    p1_vals: np.ndarray,
    p998_vals: np.ndarray,
    peak_model: "Spotiflow",
    *,
    prob_thresh: float = 0.7,
    n_tiles: tuple = (30, 30),
    min_distance: int = 5,
    verbose: bool = True,
) -> np.ndarray:
    """Detect ISS spots in a single 4-channel image and de-duplicate.

    The compute core extracted from ``detect_spots`` so the per-well
    merge step can hand in an in-memory ndarray (skipping the
    bc_stitched_registered.zarr read).

    Parameters
    ----------
    image_yxc : ndarray
        Shape ``(Y, X, 4)`` — round-0 channels 1-4, channels-last layout
        (spotiflow's expected layout).
    p1_vals, p998_vals : ndarray
        Per-channel normalization parameters, shape ``(4,)``. Typically
        produced by ``get_min_max(experiment)`` and looked up by position.
    peak_model : Spotiflow
        Pre-loaded Spotiflow model. The caller owns lifecycle so the
        model can be reused across positions/wells.
    prob_thresh, n_tiles, min_distance : passed through to
        ``peak_model.predict`` and KDTree dedup.
    verbose : bool
        Print per-channel timings.

    Returns
    -------
    points : ndarray
        Deduplicated ``(N, 2)`` spot coordinates.
    """
    channel_names = ['G', 'T', 'A', 'C']

    def process_single_channel(ch_idx, ch_name):
        t_ch_start = time.time()
        ch_data = image_yxc[:, :, ch_idx:ch_idx + 1]
        p1_ch = p1_vals[ch_idx]
        p998_ch = p998_vals[ch_idx]
        ch_points, _ = peak_model.predict(
            ch_data[:, :, 0],
            prob_thresh=prob_thresh,
            n_tiles=n_tiles,
            min_distance=min_distance,
            normalizer=lambda x: normalize(x, p1_ch, p998_ch),
            device=torch.device("cuda"),
            verbose=False,
        )
        return ch_name, ch_points, time.time() - t_ch_start

    # 4 threads — spotiflow/PyTorch release GIL during GPU work
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(process_single_channel, ch_idx, ch_name): ch_idx
            for ch_idx, ch_name in enumerate(channel_names)
        }
        channel_results = [None] * 4
        for future in as_completed(futures):
            ch_idx = futures[future]
            ch_name, ch_points, ch_time = future.result()
            channel_results[ch_idx] = ch_points
            if verbose:
                print(f"      Channel {ch_name}: {ch_time:.2f}s ({len(ch_points)} spots)")

    # Combine + dedup via KDTree
    all_points = np.vstack(channel_results)
    point_tree = KDTree(all_points)
    pairs = point_tree.query_pairs(min_distance, output_type="ndarray")
    if len(pairs) > 0:
        index_to_remove = np.unique(pairs[:, 1])
        points = np.delete(all_points, index_to_remove, axis=0)
    else:
        points = all_points
    if verbose:
        print(f"    Deduplicated: {len(all_points) - len(points)} duplicates removed, "
              f"{len(points)} points remaining")
    return points


def detect_spots_for_well_in_memory(
    registered_array: np.ndarray,
    well_pos: str,
    dataset: "OpsDataset",
    peak_model: "Spotiflow",
    p1_vals: np.ndarray,
    p998_vals: np.ndarray,
    *,
    anchor_round: int = 0,
    save: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """Run ISS spot detection for one well from an in-memory registered array.

    Per-well entry point used by the merge orchestrator. Slices the anchor
    round's channels 1-4 out of the (T, C, Z, Y, X) array (no zarr read), runs
    ``detect_spots_in_image``, optionally saves the points .npy.

    ``anchor_round`` (default 0) is the round spots are detected on — set >0 when
    round 0 has no spots (e.g. a no-incorporation cycle).

    Parameters
    ----------
    registered_array : ndarray
        Shape ``(T, C, Z, Y, X)`` for ONE well. Typically the output of
        ``apply_iss_transforms`` with shm-backed output.
    well_pos : str
        Position string (e.g. ``"A/1/0"``) — used to look up the save path.
    dataset, peak_model, p1_vals, p998_vals : as ``detect_spots_in_image``,
        plus the dataset for save-path resolution.
    save : bool
        Write the spots .npy to ``dataset.append_well("spots", well_pos)``.
    """
    if registered_array.ndim != 5:
        raise ValueError(
            f"registered_array must be 5D (T,C,Z,Y,X), got {registered_array.shape}"
        )

    t_pos_start = time.time()
    if verbose:
        print(f"\n  detect_spots ({well_pos}) — in-memory")

    # Slice the anchor round, channels 1-4, Z=0; spotiflow wants (Y, X, C) so move axis
    image_yxc = np.moveaxis(
        np.asarray(registered_array[anchor_round, 1:, 0, :, :]), 0, -1
    )
    if verbose:
        print(f"    in-memory slice shape: {image_yxc.shape}")

    points = detect_spots_in_image(
        image_yxc, p1_vals, p998_vals, peak_model, verbose=verbose,
    )

    if save:
        save_path = dataset.append_well("spots", well_pos)
        np.save(save_path, points)
        if verbose:
            print(f"    saved to {save_path}")

    if verbose:
        print(f"    well total: {time.time() - t_pos_start:.2f}s "
              f"({len(points)} spots)")
    return points


@versioned_function("v1.0")
def detect_spots(experiment: str, reproducible: bool = None):
    """
    Use Spotiflow to detect ISS spots in each channel for the anchor round of ISS
    (round 0 by default; a later round when round 0 has no spots — see the
    top-level ``anchor_round`` config key).

    Reads the registered zarr per-position. For an in-memory variant used by
    the per-well merge step, see ``detect_spots_for_well_in_memory``.

    Args:
        experiment: Experiment name
        reproducible: If True, set random seed for deterministic results.
                     If None, check experiment config for 'reproducible_detect_spots' setting.

    Note:
        - Could maybe get a better idea of which spots are real and which are noise by detecting
        spots in all rounds, then aligning
    """
    from spotiflow.model import Spotiflow

    dataset = OpsDataset(experiment)

    import yaml
    config_path = dataset.config_paths['exp_config']
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    # Check if reproducible mode should be enabled
    if reproducible is None:
        reproducible = config.get('detect_spots_params', {}).get('reproducible', False)

    # Anchor spots round — round 0 may have no spots (no-incorporation cycle);
    # use the same anchor as ISS cycle registration.
    anchor_round = int(config.get('anchor_round', 0) or 0)
    if anchor_round:
        print(f"Detecting spots on anchor round {anchor_round} (round 0 skipped)")

    if reproducible:
        random.seed(42)
        np.random.seed(42)
        print("Using reproducible mode with seed=42 for detect_spots")

    print("\n" + "=" * 80)
    print(f"Detect Spots: {experiment}")
    print("=" * 80)

    t_total_start = time.time()
    t_load_model_start = time.time()
    input_path = dataset.store_paths["iss_stitch_registered_v3"]
    peak_model = Spotiflow.from_pretrained("general")
    print(f"\nLoaded Spotiflow model ({time.time() - t_load_model_start:.2f}s)")

    t_minmax_start = time.time()
    p1_dict, p998_dict = get_min_max(experiment, anchor_round=anchor_round)
    print(f"Computed min/max normalization values ({time.time() - t_minmax_start:.2f}s)")

    with open_ome_zarr(input_path, mode="r+") as ds:
        position_list = [a[0] for a in ds.positions()]
        n_positions = len(position_list)

        print(f"\nProcessing {n_positions} positions sequentially...")

        for i, pos in enumerate(position_list):
            pos_name = pos[0] + pos[2]
            t_pos_start = time.time()
            print(f"\n  Position {i+1}/{n_positions} ({pos_name}):")

            t_load_start = time.time()
            # spotiflow requires channels to be the last dim; use the anchor round
            data = np.moveaxis(np.asarray(ds[pos].data[anchor_round, 1:, 0, :, :]), 0, -1)
            print(f"    I/O (load data): {time.time() - t_load_start:.2f}s "
                  f"(shape: {data.shape})")

            t_detect_start = time.time()
            points = detect_spots_in_image(
                data, p1_dict[pos_name], p998_dict[pos_name], peak_model,
                verbose=True,
            )
            print(f"    Detection wall: {time.time() - t_detect_start:.2f}s")

            t_save_start = time.time()
            save_path = dataset.append_well("spots", pos)
            np.save(save_path, points)

            print(f"    Save: {time.time() - t_save_start:.2f}s")
            print(f"    Position total: {time.time() - t_pos_start:.2f}s "
                  f"({len(points)} spots detected)")

    print("\n" + "=" * 80)
    print(f"Detect Spots Complete - Total time: "
          f"{time.time() - t_total_start:.2f}s ({(time.time() - t_total_start) / 60:.1f}m)")
    print("=" * 80)
    return
