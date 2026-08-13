from pathlib import Path
from tqdm import tqdm
import os
import multiprocessing as _mp


def _forkserver_noop():
    return None


# Set forkserver as the multiprocessing start method BEFORE any CUDA-using
# import so the forkserver process is CUDA-clean. Workers fork()'d from it
# inherit no CUDA state, which matters because the per-timepoint loop here
# runs cupy in the parent while regionprops fans out to a ProcessPoolExecutor.
# See memory pattern `pattern_forkserver_multi_gpu`.
try:
    _mp.set_start_method("forkserver", force=True)
    _p = _mp.Process(target=_forkserver_noop)
    _p.start()
    _p.join()
except RuntimeError:
    pass

from concurrent.futures import ProcessPoolExecutor

try:
    import torch
except ModuleNotFoundError:  # optional GPU dep; used only in GPU runtime paths
    torch = None
import numpy as np
import polars as pl

from iohub import open_ome_zarr

from stitch.registration.register import read_transform_biahub
from cyclops_utils.profiling.decorators import versioned_function
from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.data.filesystem import (
    ensure_output_path,
    decide_overwrite_resume_skip,
    parse_well,
)

try:
    import cupy as cp
except ModuleNotFoundError:  # optional GPU dep; used only in GPU runtime paths
    cp = None
try:
    import cupyx.scipy.ndimage as ndi
except ModuleNotFoundError:  # optional GPU dep; used only in GPU runtime paths
    ndi = None


# Lazy, per-process worker pool. Reused across timepoints and wells so the
# forkserver fan-out overhead is paid once per process.
_REGIONPROPS_POOL: dict = {"pool": None, "workers": None}


def _regionprops_worker_init():
    """Pre-import heavy modules in each worker (first batch is otherwise slow)."""
    import skimage.measure._regionprops  # noqa: F401
    import scipy.ndimage  # noqa: F401
    import tracksdata.constants  # noqa: F401
    import tracksdata.nodes._mask  # noqa: F401


def _regionprops_worker(args):
    """
    Compute regionprops attrs for a batch of labels.

    Attaches to the labels array via shared memory (no pickling of the 2.7 GB
    array), constructs skimage RegionProperties per label using the
    pre-computed slice, returns a list of dicts in the same shape as
    tracksdata.RegionPropsNodes._nodes_per_time.
    """
    shm_name, shape, dtype_str, slices, label_ids, extra_properties, t = args
    from multiprocessing import shared_memory
    from skimage.measure._regionprops import RegionProperties
    from tracksdata.constants import DEFAULT_ATTR_KEYS
    from tracksdata.nodes._mask import Mask

    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        labels = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
        out: list[dict] = []
        for sl, lbl in zip(slices, label_ids):
            obj = RegionProperties(
                slice=sl,
                label=lbl,
                label_image=labels,
                intensity_image=None,
                cache_active=True,
                spacing=None,
            )
            cz, cy, cx = obj.centroid
            attrs: dict = {
                DEFAULT_ATTR_KEYS.Z: cz,
                DEFAULT_ATTR_KEYS.Y: cy,
                DEFAULT_ATTR_KEYS.X: cx,
            }
            for prop in extra_properties:
                attrs[prop] = getattr(obj, prop)
            attrs[DEFAULT_ATTR_KEYS.MASK] = Mask(obj.image, obj.bbox)
            attrs[DEFAULT_ATTR_KEYS.BBOX] = np.asarray(obj.bbox, dtype=int)
            attrs[DEFAULT_ATTR_KEYS.T] = t
            obj._cache.clear()
            out.append(attrs)
    finally:
        shm.close()
    return out


def _get_regionprops_pool(max_workers: int) -> ProcessPoolExecutor:
    if (
        _REGIONPROPS_POOL["pool"] is None
        or _REGIONPROPS_POOL["workers"] != max_workers
    ):
        if _REGIONPROPS_POOL["pool"] is not None:
            _REGIONPROPS_POOL["pool"].shutdown(wait=True)
        ctx = _mp.get_context("forkserver")
        _REGIONPROPS_POOL["pool"] = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=_regionprops_worker_init,
        )
        _REGIONPROPS_POOL["workers"] = max_workers
    return _REGIONPROPS_POOL["pool"]


def _parallel_regionprops_nodes_data(
    labels_4d: np.ndarray,
    t: int,
    extra_properties: list[str],
    max_workers: int,
) -> list[dict]:
    """
    Parallel regionprops via find_objects + process pool with shared memory.

    Replaces tracksdata.RegionPropsNodes._nodes_per_time's serial Python loop.
    Calls scipy.ndimage.find_objects once on the full label volume — a single
    O(pixels) C scan — to get the bbox slice for every label. The label array
    is exposed to workers through multiprocessing.shared_memory so the 2.7 GB
    payload is not pickled. Workers fan out per-label property computation,
    each constructing skimage RegionProperties from its precomputed slice and
    returning a list of attrs dicts. Workers run in real OS processes, so the
    per-cell Python work parallelizes without GIL contention.

    Returns nodes_data in the same shape as RegionPropsNodes._nodes_per_time.
    """
    from scipy.ndimage import find_objects as _find_objects
    from multiprocessing import shared_memory

    labels_3d = np.ascontiguousarray(labels_4d[0])  # (1, H, W)

    all_slices = _find_objects(labels_3d)
    pairs = [(sl, i + 1) for i, sl in enumerate(all_slices) if sl is not None]
    if not pairs:
        return []

    shm = shared_memory.SharedMemory(create=True, size=labels_3d.nbytes)
    try:
        shm_arr = np.ndarray(labels_3d.shape, dtype=labels_3d.dtype, buffer=shm.buf)
        shm_arr[:] = labels_3d

        # ~4 batches per worker for load balance. Per-cell work varies by
        # bbox size; small batches let fast workers steal extra batches.
        n = len(pairs)
        n_batches = max(1, max_workers * 4)
        batch_size = max(1, (n + n_batches - 1) // n_batches)
        batches = []
        for i in range(0, n, batch_size):
            chunk = pairs[i : i + batch_size]
            slices = [p[0] for p in chunk]
            label_ids = [p[1] for p in chunk]
            batches.append(
                (
                    shm.name,
                    labels_3d.shape,
                    str(labels_3d.dtype),
                    slices,
                    label_ids,
                    extra_properties,
                    t,
                )
            )

        pool = _get_regionprops_pool(max_workers)
        results = list(pool.map(_regionprops_worker, batches))

        nodes_data: list[dict] = []
        for r in results:
            nodes_data.extend(r)
        return nodes_data
    finally:
        shm.close()
        shm.unlink()


def _check_timepoint_quality(
    tracking_seg_store,
    well: str,
    drop_threshold: float = 0.15,
    crop_half: int = 512,
) -> tuple[list[int], list[int]]:
    """
    Determine which tracking timepoints are usable by checking for label
    loss. A timepoint is degraded when its label count drops by more than
    drop_threshold relative to the running maximum up to that point.

    Only penalizes label *loss* (cell disappearance), not natural growth
    where earlier timepoints have fewer labels than later ones.

    Returns (valid_timepoint_indices, label_counts_per_timepoint).
    """
    arr = tracking_seg_store[well]["0"]
    T = arr.shape[0]
    h, w = arr.shape[-2], arr.shape[-1]

    # 5 regions at different ring distances from center (15%-75% of half-extent)
    cy, cx = h // 2, w // 2
    regions = []
    for frac in [0.15, 0.30, 0.45, 0.60, 0.75]:
        ry = int(cy + frac * (h // 2))
        rx = int(cx + frac * (w // 2))
        regions.append((ry, rx))

    label_counts = []
    for t in range(T):
        all_labels: set[int] = set()
        for cy, cx in regions:
            y0 = max(0, cy - crop_half)
            y1 = min(h, cy + crop_half)
            x0 = max(0, cx - crop_half)
            x1 = min(w, cx + crop_half)
            crop = np.asarray(arr[t, 0, 0, y0:y1, x0:x1])
            all_labels.update(np.unique(crop).tolist())
        all_labels.discard(0)
        label_counts.append(len(all_labels))

    # Flag timepoints where labels drop relative to the running max (prior peak).
    # t0 is always valid since there's no prior timepoint to compare against.
    valid = [0]
    running_max = label_counts[0]
    for t in range(1, len(label_counts)):
        drop_frac = (running_max - label_counts[t]) / running_max if running_max > 0 else 0.0
        if drop_frac <= drop_threshold:
            valid.append(t)
        running_max = max(running_max, label_counts[t])

    return valid, label_counts


def _resolve_track_backend() -> str:
    """Resolve OPS_TRACK_BACKEND='auto' to a concrete 'ilpy' or 'cuopt'.

    In "auto" mode we choose cuopt when a CUDA GPU is visible, else ilpy.
    Returns the literal env value when explicitly set to ilpy or cuopt.
    """
    # Default to ilpy (Gurobi). cuOpt is opt-in only — the installed ILP solver
    # is Gurobi-licensed and hoct's ILPSolverConfig has no cuOpt/backend support.
    requested = os.environ.get("OPS_TRACK_BACKEND", "ilpy")
    if requested in ("ilpy", "cuopt"):
        return requested
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuopt"
    except Exception:
        pass
    return "ilpy"


def create_track_graph(
    config: dict,
    debug: bool = False,
    debug_tile_size: int = 1024,
    dataset: OpsDataset = None,
    crop_coords: tuple[int, int] = None,
):
    """ """
    import tracksdata as td
    from hoct.features import add_delta_t

    # Per-phase timing for tracking-step wall-clock profiling. Mirror of
    # the instrumentation already in hoct._api.predict() and
    # hoct.inference._predict.model_predict(); together these
    # account for every meaningful slice of the per-well wall.
    import time as _phase_time
    _t_ctg_start = _phase_time.monotonic()
    _t_last = _t_ctg_start
    def _phase(name):
        nonlocal _t_last
        now = _phase_time.monotonic()
        print(
            f"[PHASE-TIMING] create_track_graph::{name}: "
            f"{now - _t_last:.2f}s (cumulative {now - _t_ctg_start:.2f}s)",
            flush=True,
        )
        _t_last = now

    if debug:
        y_slice = slice(crop_coords[0], crop_coords[0] + debug_tile_size)
        x_slice = slice(crop_coords[1], crop_coords[1] + debug_tile_size)
    else:
        y_slice = slice(None)
        x_slice = slice(None)

    graph = None
    _extra_properties = ["equivalent_diameter_area", "inertia_tensor"]
    node_operator = td.nodes.RegionPropsNodes(extra_properties=_extra_properties)

    # Parallel regionprops via find_objects + per-label fan-out. Default ON.
    # Disable with OPS_TRACK_PARALLEL_REGIONPROPS=0 to fall back to tracksdata serial.
    _parallel_enabled = os.environ.get("OPS_TRACK_PARALLEL_REGIONPROPS", "1") != "0"
    _parallel_workers = int(os.environ.get("OPS_TRACK_REGIONPROPS_WORKERS", "16"))

    T = len(config["labels_list"])
    _phase("setup (RegionPropsNodes ctor + slice config)")
    if debug:
        # save the transformed labels for reference
        debug_subdir = dataset.tracking / "debug"
        debug_subdir.mkdir(exist_ok=True, parents=True)
        output_store = open_ome_zarr(
            debug_subdir / f"debug_images.zarr",
            layout="hcs",
            mode="w",
            channel_names=["segmentation"],
        )
        output_pos = output_store.create_position(*Path(config["well"]).parts)
        output_pos.create_zeros(
            name="0",
            shape=(T, 1, 1, debug_tile_size, debug_tile_size),
            dtype=np.float32,
        )

    for t in tqdm(range(T), desc=f"Creating graph for well {config['well']}"):
        _t_iter0 = _phase_time.monotonic()
        tmp_graph = td.graph.InMemoryGraph()

        # apply affine transform on the fly — move labels into "tracking" space
        cp_labels_crop = cp.asarray(
            config["labels_list"][t][config["well"]][0][
                config["time_point"][t], 0, 0, :, :
            ]
        )
        cp_affine_matrix = cp.asarray(config["affine_list"][t])
        labels_transformed = ndi.affine_transform(
            cp_labels_crop, cp_affine_matrix, order=0
        )
        l_transformed = labels_transformed.get()
        _t_iter1 = _phase_time.monotonic()

        if debug:
            output_store[config["well"]]["0"][t, 0, 0, :, :] = l_transformed[
                y_slice, x_slice
            ]
        _labels_4d = np.expand_dims(
            l_transformed[y_slice, x_slice], axis=(0, 1)
        )  # (T=1, Z=1, Y, X) — what add_nodes expects
        if _parallel_enabled:
            _nodes_data = _parallel_regionprops_nodes_data(
                labels_4d=_labels_4d,
                t=0,
                extra_properties=_extra_properties,
                max_workers=_parallel_workers,
            )
            if _nodes_data:
                node_operator._init_node_attrs(tmp_graph, _nodes_data[0])
                tmp_graph.bulk_add_nodes(_nodes_data)
        else:
            node_operator.add_nodes(graph=tmp_graph, labels=_labels_4d)
        _t_iter2 = _phase_time.monotonic()
        _n_nodes_this_t = tmp_graph.num_nodes()

        if t == 0:
            graph = tmp_graph
        else:
            tmp_attrs = tmp_graph.node_attrs().drop("node_id", "t").with_columns(t=t)
            graph.bulk_add_nodes(tmp_attrs.to_dicts())
        _t_iter3 = _phase_time.monotonic()
        print(
            f"[PHASE-TIMING] create_track_graph::loop t={t}: "
            f"zarr_read+cupy_affine={_t_iter1 - _t_iter0:.2f}s, "
            f"add_nodes(regionprops, {_n_nodes_this_t} nodes)={_t_iter2 - _t_iter1:.2f}s, "
            f"merge={_t_iter3 - _t_iter2:.2f}s",
            flush=True,
        )

    _phase(f"per-timepoint loop (T={T}: affine_transform + RegionPropsNodes + bulk_add)")

    for empty_column in ["border_dist", "intensity_mean", "intensity_min", "intensity_max", "intensity_std"]:
        graph.add_node_attr_key(empty_column, pl.Float32, 0.0)
    _phase("add empty node attr keys")

    # Edge generation knobs — these define the candidate-edge graph the
    # ILP solver searches over. The 25/5 default produces 9-15 M-var
    # ILPs; Gurobi handles them fine but cuOpt's branch-and-bound can
    # OOM on the largest wells (15 M-var crash on ops0042 A/1). Cells
    # in 5x tracking move well under 18 px between adjacent frames, so
    # on the cuOpt path we drop the distance threshold to 18 — that
    # shrinks the ILP by ~30% and let all three ops0042 wells (8-13 M
    # vars) solve to Optimal in ~15 min each.
    #
    # Default selection is backend-aware so Gurobi-licensed runs stay
    # bit-equivalent to the pre-cuOpt era while cuOpt runs get a
    # graph that actually fits. Override either default with the env
    # vars below.
    _dist_default = 18.0 if _resolve_track_backend() == "cuopt" else 25.0
    _dist = float(os.environ.get("OPS_TRACK_DISTANCE_THRESHOLD", _dist_default))
    _knn = int(os.environ.get("OPS_TRACK_N_NEIGHBORS", 5))
    td.edges.DistanceEdges(
        distance_threshold=_dist,
        delta_t=1,
        n_neighbors=_knn,
    ).add_edges(graph=graph)
    _phase(f"DistanceEdges.add_edges (n={graph.num_nodes()} → e={graph.num_edges()})")

    add_delta_t(graph)
    _phase("add_delta_t")

    print(
        f"[PHASE-TIMING-SUMMARY] create_track_graph total: "
        f"{_phase_time.monotonic() - _t_ctg_start:.2f}s "
        f"(nodes={graph.num_nodes()}, edges={graph.num_edges()})",
        flush=True,
    )

    return graph


def prepare_affine(path: str, ndim: int = 2) -> np.ndarray:
    """
    - If using matrix multiplication (@) need to invert the affine matrix
    - If using ndi.affine_transoform then do not invert
    """

    T_reg_embedded = np.identity(ndim + 1)
    T_reg = read_transform_biahub(path)
    T_2d = np.identity(3)
    T_2d[0:2, 0:2] = T_reg[1:3, 1:3]  # YX rotation/scale
    T_2d[0:2, 2] = T_reg[1:3, 3]  # YX translation
    T_reg_embedded[-(2 + 1) :, -(2 + 1) :] = T_2d
    # AT = np.linalg.inv(T_reg_embedded)

    return T_reg_embedded


def _check_tracking_qc(
    config_list: list[dict],
    tracking_seg_store,
    dataset: OpsDataset,
    drop_threshold: float = 0.15,
) -> None:
    """
    QC check on tracking timepoints. Raises an error if any tracking
    timepoint shows significant label loss (>drop_threshold), which
    indicates a reconstruction issue (e.g. missing tiles).

    Does not modify configs — only warns or errors.
    """
    for config in config_list:
        well = config["well"]
        tp = config["time_point"]
        labels = config["labels_list"]

        # Only check middle entries that use tracking_seg_store
        tracking_indices = [
            i for i in range(1, len(tp) - 1)
            if labels[i] is tracking_seg_store
        ]
        if not tracking_indices:
            print(f"  {well}: no tracking timepoints to check (direct mode)")
            continue

        try:
            valid_tp, label_counts = _check_timepoint_quality(
                tracking_seg_store, well, drop_threshold=drop_threshold,
            )
        except KeyError:
            print(f"  {well}: not found in tracking seg store, skipping QC")
            continue

        T_total = len(label_counts)
        degraded = []
        for i in tracking_indices:
            t_idx = tp[i]
            if t_idx >= T_total:
                degraded.append((t_idx, "MISSING", "timepoint does not exist in store"))
            elif t_idx not in valid_tp:
                running_max = max(label_counts[:t_idx + 1]) if t_idx > 0 else label_counts[0]
                prev_max = max(label_counts[:t_idx]) if t_idx > 0 else label_counts[0]
                drop_pct = (prev_max - label_counts[t_idx]) / prev_max * 100 if prev_max > 0 else 0
                degraded.append((t_idx, label_counts[t_idx], f"{drop_pct:.1f}% loss from prior peak {prev_max}"))

        if degraded:
            msg_lines = [
                f"\n{'='*70}",
                f"  QC ERROR: Low quality tracking timepoints for {well}",
                f"  Experiment: {dataset.experiment}",
                f"  Tracking seg store: {dataset.store_paths['lc_5x_segmentation']}",
                f"  Label counts per timepoint: {label_counts}",
                f"",
                f"  Degraded timepoints used by config:",
            ]
            for t_idx, count, reason in degraded:
                msg_lines.append(f"    t={t_idx}: labels={count} — {reason}")
            msg_lines.extend([
                f"",
                f"  This likely indicates missing tiles in the tracking reconstruction.",
                f"  Please inspect the tracking segmentation images for {well}.",
                f"{'='*70}",
            ])
            raise RuntimeError("\n".join(msg_lines))
        else:
            tracking_tp_used = [tp[i] for i in tracking_indices]
            print(f"  {well}: all tracking timepoints OK {tracking_tp_used} (counts={label_counts})")


def _resolve_track_wells(dataset: OpsDataset, well) -> list[str]:
    """Resolve a well arg to full row/col units; "all" derives from config/store."""
    if well != "all":
        row, col = parse_well(well)
        return [f"{row}/{col}/0"]

    # "all": prefer config wells_to_process, else infer from on-disk outputs.
    import yaml
    wells = []
    exp_config_path = dataset.config_paths["exp_config"]
    if exp_config_path.exists():
        with open(exp_config_path) as f:
            cfg = yaml.safe_load(f) or {}
        wells = cfg.get("wells_to_process") or []
    if not wells:
        wells = dataset.infer_wells()
    units = []
    for w in wells:
        row, col = parse_well(w)
        units.append(f"{row}/{col}/0")
    return units


def _build_track_config(
    dataset: OpsDataset,
    position: str,
    col: int,
    ops_num: int,
    skip_track: bool,
    pheno_seg_store,
    iss_seg_store,
    tracking_seg_store,
) -> dict:
    """Build one well's tracking config. Timepoint layout is keyed by column (1/2/3)."""
    if skip_track:
        # Skip_track: only ISS→pheno registration (pheno is reference).
        iss_pheno_affine = prepare_affine(
            dataset.append_well("auto_iss_register", position)
        )
        return {
            "affine_list": [np.identity(3), iss_pheno_affine],
            "well": position,
            "labels_list": [pheno_seg_store, iss_seg_store],
            "time_point": [0, 0],
        }

    if col == 1:
        pheno_track_affine = prepare_affine(
            dataset.append_well("auto_pheno_register", position)
        )
        iss_track_affine = prepare_affine(
            dataset.append_well("auto_iss_register", position)
        )
        if ops_num < 69:
            # ops_num < 69: tracking timepoints [0,1,2,3]
            return {
                "affine_list": [
                    pheno_track_affine,
                    np.identity(3),
                    np.identity(3),
                    np.identity(3),
                    iss_track_affine,
                ],
                "well": position,
                "labels_list": [
                    pheno_seg_store,
                    tracking_seg_store,
                    tracking_seg_store,
                    tracking_seg_store,
                    iss_seg_store,
                ],
                "time_point": [0, 1, 2, 3, 0],
            }
        # ops_num >= 69: only tracking timepoints 0 and 1 available
        return {
            "affine_list": [
                pheno_track_affine,
                np.identity(3),
                np.identity(3),
                iss_track_affine,
            ],
            "well": position,
            "labels_list": [
                pheno_seg_store,
                tracking_seg_store,
                tracking_seg_store,
                iss_seg_store,
            ],
            "time_point": [0, 0, 1, 0],
        }

    if col == 2:
        pheno_track_affine = prepare_affine(
            dataset.append_well("auto_pheno_register", position)
        )
        iss_track_affine = prepare_affine(
            dataset.append_well("auto_iss_register", position)
        )
        if ops_num < 69:
            # pheno→track t=2, ISS→track t=3
            return {
                "affine_list": [
                    pheno_track_affine,
                    np.identity(3),
                    np.identity(3),
                    iss_track_affine,
                ],
                "well": position,
                "labels_list": [
                    pheno_seg_store,
                    tracking_seg_store,
                    tracking_seg_store,
                    iss_seg_store,
                ],
                "time_point": [0, 2, 3, 0],
            }
        # ops_num >= 69: tracking timepoint 1
        return {
            "affine_list": [
                pheno_track_affine,
                np.identity(3),
                iss_track_affine,
            ],
            "well": position,
            "labels_list": [
                pheno_seg_store,
                tracking_seg_store,
                iss_seg_store,
            ],
            "time_point": [0, 1, 0],
        }

    if col == 3:
        if ops_num < 69:
            # Registration targets tracking t=3 for well 3
            pheno_track_affine = prepare_affine(
                dataset.append_well("auto_pheno_register", position)
            )
            iss_track_affine = prepare_affine(
                dataset.append_well("auto_iss_register", position)
            )
            return {
                "affine_list": [
                    pheno_track_affine,
                    np.identity(3),
                    iss_track_affine,
                ],
                "well": position,
                "labels_list": [
                    pheno_seg_store,
                    tracking_seg_store,
                    iss_seg_store,
                ],
                "time_point": [0, 3, 0],
            }
        # ops_num >= 69: no tracking data — direct ISS→Pheno from auto_iss_register
        iss_pheno_affine = prepare_affine(
            dataset.append_well("auto_iss_register", position)
        )
        return {
            "affine_list": [np.identity(3), iss_pheno_affine],
            "well": position,
            "labels_list": [pheno_seg_store, iss_seg_store],
            "time_point": [0, 0],
        }

    raise ValueError(f"Unsupported well column {col} (position {position}); expected 1, 2, or 3")


def _compact_seed_labels(seg):
    """Relabel a pheno seed segmentation to contiguous ids (background 0 kept).

    Tracking is insensitive to label *values* — only the partition matters — but
    native-20x nuclei stitching leaves ids scattered up to ~44M, and ids > 2**24
    alias under a float32 cast in the tracksdata/geff node-attr path, collapsing
    distinct cells into spurious dedupe multiplets (~half the cells culled).
    Compacting to contiguous int32 removes the hazard. Idempotent: a store born
    compact (post cell_segmentation.py union-find fix) is returned unchanged. The
    final assert doubles as a guard that the seed is safe to track on.
    """
    arr = np.asarray(seg)
    if int(arr.max()) < (1 << 24):
        return arr.astype(np.int32, copy=False)  # no aliasing hazard; leave as-is
    uniq = np.unique(arr)  # sorted -> background 0 stays 0
    lut = np.zeros(int(uniq[-1]) + 1, dtype=np.int32)
    lut[uniq] = np.arange(len(uniq), dtype=np.int32)
    compacted = lut[arr]
    assert compacted.max() < (1 << 24), (
        f"pheno seed has {len(uniq)} labels — exceeds 2**24 even after "
        "compaction; float32 id-aliasing hazard remains"
    )
    return compacted


def assemble_track_configs(
    experiment,
    well: str = "all",
    skip_track: bool = False,
):
    dataset = OpsDataset(experiment)
    config_list = []

    # Determine ops number for logic branching
    ops_num = int(experiment.split("_")[0].replace("ops", ""))

    # Load segmentation stores. Pheno nuclei come from the v3 nuclear_seg label
    # (5x level); {0: arr} mirrors the iohub position[0] access the config loop uses.
    from cyclops_process.data.datasets import load_pheno_nuclear_seg_v3

    # Resolve wells to full row/col units so rows A/B never collide.
    pheno_wells = _resolve_track_wells(dataset, well)
    pheno_seg_store = {
        w: {0: _compact_seed_labels(seg)}
        for w in pheno_wells
        if (seg := load_pheno_nuclear_seg_v3(dataset, w)) is not None
    }
    iss_seg_store = open_ome_zarr(dataset.store_paths["iss_segmentation"])

    # Only load tracking segmentation if not skipping tracking data
    tracking_seg_store = None
    if not skip_track:
        tracking_seg_store = open_ome_zarr(dataset.store_paths["lc_5x_segmentation"])

    # Build a config per well, branching on column for the timepoint layout.
    for position in pheno_wells:
        _, col = parse_well(position)
        config_list.append(
            _build_track_config(
                dataset, position, col, ops_num, skip_track,
                pheno_seg_store, iss_seg_store, tracking_seg_store,
            )
        )

    # QC check: warn about degraded tracking timepoints (reconstruction issues)
    if not skip_track:
        print("\n  Checking tracking timepoint quality...")
        _check_tracking_qc(
            config_list,
            tracking_seg_store,
            dataset,
        )

    return config_list


def _check_gurobi_license():
    """Verify Gurobi license is accessible before starting any heavy computation.

    If Gurobi is unavailable (no license, license server unreachable, or the
    `gurobipy` package missing) AND the cuOpt backend is installed, the error
    message points the caller at the open-license fallback via OPS_TRACK_BACKEND=cuopt.
    """
    def _suggest_cuopt(reason: str) -> str:
        try:
            import cuopt  # noqa: F401
            return (
                f"{reason}\n"
                "cuOpt is installed in this environment — set "
                "OPS_TRACK_BACKEND=cuopt to use NVIDIA's open-license GPU ILP "
                "solver instead. Note: hoct's ILPSolverConfig has no `backend` "
                "field, so this needs upstream support before it takes effect."
            )
        except ImportError:
            return reason

    try:
        import gurobipy
        m = gurobipy.Model()
        m.dispose()
    except ImportError:
        raise RuntimeError(
            _suggest_cuopt(
                "gurobipy is not installed. Cannot run ILP tracking solver."
            )
        )
    except gurobipy.GurobiError as e:
        raise RuntimeError(
            _suggest_cuopt(
                f"Gurobi license check failed: {e}\n"
                f"Expected license file at: ~/gurobi.lic (TOKENSERVER=license01.czbiohub.org)\n"
                "Ensure the file exists and the token server is reachable."
            )
        ) from e


@versioned_function("v1.3")
def track_wells(
    experiment: str,
    debug: bool = False,
    debug_tile_size: int = 1024,
    crop_coords: tuple[int, int] = (10000, 10000),
    debug_output_suffix: str = "_debug",
    well: str = "A/1/0",
    skip_track: bool = False,
    test_time_augs: int = 0,
):
    """
    Track cells in a single well using ultrack.

    Parameters
    ----------
    experiment : str
        Experiment name (e.g., "ops0033_20250429")
    debug : bool
        If True, crop ROI and save debug images
    debug_tile_size : int
        Size of debug ROI tile (default: 1024)
    crop_coords : tuple[int, int]
        Top-left coordinates for debug ROI (default: (10000, 10000))
    debug_output_suffix : str
        Suffix for debug output files (default: "_debug")
    well : str
        Single well to process (e.g., "A/1/0"). Use "all" to process all wells sequentially.
    skip_track : bool
        If True, track using only ISS and pheno timepoints without intermediate tracking data (default: False)
    test_time_augs : int
        Number of test-time augmentations (flip + rotation) to average over. 0 = disabled (default: 0)

    Returns
    -------
    solution
        Tracking solution for the well(s)
    """
    import tracksdata as td
    from hoct import predict
    from hoct.tracking import ILPSolverConfig

    # OPS_TRACK_BACKEND only decides whether the Gurobi license check runs
    # below. hoct's ILPSolverConfig exposes no `backend` field, so there is no
    # cuOpt path to select — the solver is always Gurobi via ilpy.
    _track_backend = os.environ.get("OPS_TRACK_BACKEND", "ilpy")
    if _track_backend == "ilpy":
        _check_gurobi_license()

    # Per-phase timing for the pre-create_track_graph window — the 119s
    # of "where did that go" Mark flagged.
    import time as _phase_time
    _t_wells_start = _phase_time.monotonic()
    _t_last = _t_wells_start
    def _wells_phase(name):
        nonlocal _t_last
        now = _phase_time.monotonic()
        print(
            f"[PHASE-TIMING] track_wells::{name}: "
            f"{now - _t_last:.2f}s (cumulative {now - _t_wells_start:.2f}s)",
            flush=True,
        )
        _t_last = now

    dataset = OpsDataset(experiment)
    _wells_phase("OpsDataset(experiment)")

    # Info message if skip_track mode
    if skip_track:
        print(f"ℹ️  Tracking in skip_track mode for {experiment}")
        print("  Only using ISS and pheno timepoints (no intermediate tracking data)")

    configs = assemble_track_configs(experiment, well=well, skip_track=skip_track)
    _wells_phase("assemble_track_configs (open_ome_zarr × 3 + prepare_affine × N + quality check)")
    print(f"Assembled {len(configs)} tracking configurations for well(s): {well}")

    # Load the model once (shared across wells if processing multiple)
    model = torch.jit.load(dataset.model_paths["track_model"], map_location="cpu")
    _wells_phase("torch.jit.load(track_model) [cpu]")
    if torch.cuda.is_available():
        model = model.to("cuda:0")
    model.eval()
    _wells_phase("model.to(cuda:0) + model.eval()")
    # solver_config = ILPSolverConfig(
    #     appearance_weight=0.25,
    #     disappearance_weight=0.5,
    #     division_weight=0.5,
    #     node_weight=-10,
    #     delta_t_weight=1.0,
    #     timeout=36_000,
    #     edge_bias=0.5,          # bias added to edge weights before solving (-p_ij + bias)
    #     tracklet_solver=False,  # two-pass solving (tracklets then linkage)
    # )
    # Resolve Gurobi thread count from the SLURM allocation, for the day
    # ILPSolverConfig grows a num_threads field (see the guard below — hoct
    # 0.1.0rc0 does not have it, so this value is currently unused and Gurobi
    # runs at its own default thread count).
    #   OPS_TRACK_NUM_THREADS — explicit override
    #   SLURM_CPUS_PER_TASK   — match the SLURM allocation (typical case)
    #   os.cpu_count() cap 32 — bare-metal fallback; Gurobi peaks well before
    #                           32 threads on the graphs we solve
    #   1                     — final fallback
    _num_threads_val = None
    for _var in ("OPS_TRACK_NUM_THREADS", "SLURM_CPUS_PER_TASK"):
        _v = os.environ.get(_var)
        if _v:
            try:
                _n = int(_v)
                if _n > 0:
                    _num_threads_val = _n
                    break
            except ValueError:
                pass
    if _num_threads_val is None:
        _num_threads_val = min(os.cpu_count() or 1, 32)

    solver_config = ILPSolverConfig(
        appearance_weight=0.5,
        disappearance_weight=0.5,
        division_weight=0.5,
        node_weight=-10,
        delta_t_weight=1.0,
        timeout=36_000,
        edge_bias=0.0,          # bias added to edge weights before solving (-p_ij + bias)
        tracklet_solver=False,  # two-pass solving (tracklets then linkage)
        # num_threads and backend are forward-compat guards, both currently
        # INACTIVE: hoct 0.1.0rc0's ILPSolverConfig has neither field (and sets
        # pydantic extra="forbid", so passing them unconditionally would raise
        # ValidationError). They stay so that an upstream hoct release adding
        # either field is picked up without a code change here.
        **({"num_threads": _num_threads_val}
           if "num_threads" in getattr(ILPSolverConfig, "model_fields", {}) else {}),
        **({"backend": _track_backend}
           if "backend" in getattr(ILPSolverConfig, "model_fields", {}) else {}),
    )



    print("Loaded model")

    # Process each well configuration
    # Note: When called with a single well, this will only loop once
    output_paths = []
    for config in configs:
        print(f"Processing well: {config['well']}")
        well_graph = create_track_graph(
            config, debug, debug_tile_size, dataset, crop_coords
        )
        cp.get_default_memory_pool().free_all_blocks()
        print(f"Created graph for well {config['well']}")
        well_graph.summary()
        tps = len(well_graph.time_points())
        print(f"Number of timepoints: {tps}")

        print(f"Running inference for well {config['well']}")
        solution = predict(
            model=model,
            graph=well_graph,
            solver_config=solver_config,
            tiling_scheme=td.functional.TilingScheme(
                tile_shape=(tps, 1, 512, 512),
                overlap_shape=(0, 0, 64, 64),
            ),
            test_time_augs=test_time_augs,
        )
        print(f"Found solution for well {config['well']}")

        # save results...
        output_path = Path(dataset.append_well("tracking_geff", config["well"]))
        if debug:
            output_path = output_path.with_name(
                f"{output_path.stem}_{debug_output_suffix}{output_path.suffix}"
            )
        if output_path.exists():
            import shutil
            print(f"Removing existing geff file: {output_path}")
            shutil.rmtree(output_path)
        solution.to_geff(output_path)
        print(f"Saved tracking solution to: {output_path}")

        # Save completion marker YAML
        # Format well name: "A/1/0" -> "A1"
        import pandas as pd
        import yaml
        well_name = config["well"].replace("/", "")
        completion_path = output_path.parent / f"{well_name}_tracking_complete.yaml"
        completion_data = {
            "experiment": experiment,
            "well": config["well"],
            "completed_at": pd.Timestamp.now().isoformat(),
            "output_file": str(output_path.name),
            "debug_mode": debug,
            "num_timepoints": tps,
            "num_nodes": len(solution.node_attrs()),
            "num_edges": len(solution.edge_attrs()),
        }
        with open(completion_path, "w") as f:
            yaml.dump(completion_data, f, default_flow_style=False, sort_keys=False)
        print(f"Saved completion marker to: {completion_path}")
        output_paths.append(str(output_path))

    # Return paths (picklable) instead of GraphView (contains Cython RTree, not picklable by submitit)
    return output_paths


# -----------------------------
# CLI support
# -----------------------------
def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ultrack tracking on wells with optional debug ROI cropping"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # track_well subcommand
    tw = subparsers.add_parser(
        "track",
        help="Track cells in a single well using ultrack",
    )
    tw.add_argument(
        "--experiment", type=str, required=True, help="Experiment name (OPS convention)"
    )
    tw.add_argument(
        "--well",
        type=str,
        default="A/1/0",
        help="Well position (default: A/1/0), all for all wells",
    )
    tw.add_argument(
        "--debug",
        action="store_true",
        help="Run tracking on a debug ROI instead of the full image",
    )
    tw.add_argument(
        "--debug-tile-size",
        type=int,
        default=2048,
        help="Size of roi if only running a subset for debugging (default: 1024)",
    )
    tw.add_argument(
        "--debug-output-suffix",
        type=str,
        default="_debug",
        help="Suffix for debug output files (default: _debug)",
    )
    tw.add_argument(
        "--crop-coords",
        type=int,
        nargs=2,
        default=(10000, 10000),
        help="Coordinates for cropping the debug ROI (default: 10000 10000)",
    )
    tw.add_argument(
        "--test-time-augs",
        type=int,
        default=0,
        help="Number of test-time augmentations to average over (default: 0, disabled)",
    )

    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.command == "track":
        print(
            f"[track][track_well] experiment={args.experiment}, well={args.well}, "
            f"debug_tile_size={args.debug_tile_size}, crop_coords={args.crop_coords}"
        )
        track_wells(
            experiment=args.experiment,
            well=args.well,
            debug=args.debug,
            debug_tile_size=args.debug_tile_size,
            crop_coords=tuple(args.crop_coords),
            debug_output_suffix=args.debug_output_suffix,
            test_time_augs=args.test_time_augs,
        )
        return


if __name__ == "__main__":
    main()
    # Usage examples:
    # python -m cyclops_process.processes.track track_well --experiment ops0033_20250429 --well A/1/0 --debug --tile-size 1024 --well A/1/0
    # python -m cyclops_process.processes.track track_well --experiment ops0033_20250429 --well A/1/0
    # python -m cyclops_process.processes.track track_well --experiment ops0033_20250429 --well all
