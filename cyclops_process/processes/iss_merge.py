"""Per-well merged ISS step + per-experiment fanout orchestrator.

Replaces the post-stitch sequence (apply_iss_transforms inside
job_finalize_registration → detect_spots → base_calling) with a single
function that runs all three on one well's data, holding the
registered (T, C, Z, Y, X) float32 array in ``multiprocessing.shared_memory``
across the steps. This eliminates the bc_stitched_registered.zarr
NFS round-trip that today happens between finalize and detect_spots,
and again between detect_spots and base_calling.

The merge writes the registered zarr exactly once (downstream consumers —
convert_v3, iss_metrics, napari — still need it on disk).

Designed to run as a per-well SLURM job, three jobs in parallel for a
3-well experiment. Same shape as today's wave-2 finalize jobs.

Resources expected: 1 GPU (spotiflow), 32 CPU (warp + base_calling),
~350 GB RAM (75 GB stitched f16 read transient + 150 GB registered f32
shm + per-tile working set).
"""
from __future__ import annotations
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import yaml

from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.data.filesystem import parse_well
from cyclops_utils.profiling.decorators import versioned_function


def _load_cumulative_transforms(transforms_dir: Path) -> dict:
    """Load *_cumulative.yaml files into the dict apply_iss_transforms expects."""
    affines: dict = {}

    def _affine_4x4_to_3x3(affine_4x4_inv: np.ndarray) -> np.ndarray:
        affine_4x4 = np.linalg.inv(affine_4x4_inv)
        affine_3x3 = np.eye(3)
        affine_3x3[:2, :2] = affine_4x4[1:3, 1:3]
        affine_3x3[:2, 2] = affine_4x4[1:3, 3]
        return affine_3x3

    for path in transforms_dir.glob("*_cumulative.yaml"):
        with open(path) as f:
            data = yaml.safe_load(f)
        affine_4x4_inv = np.array(data["affine_transform_zyx"])
        affine_3x3 = _affine_4x4_to_3x3(affine_4x4_inv)

        name = path.stem.replace("_cumulative", "")
        if name == "segmentation_to_round0":
            affines[-2] = affine_3x3
        elif name == "nucleus_to_round0":
            affines[-1] = affine_3x3
        elif name.startswith("round"):
            r = int(name.split("_")[0].replace("round", ""))
            affines[r] = affine_3x3
    return affines


@versioned_function("v1.0")
def merge_spots_base_calling_well(
    experiment: str,
    well,
    *,
    method: str = "mine",
    skip_registered_zarr_write: bool = False,
    skip_detect_spots: bool = False,
    skip_base_calling: bool = False,
    reproducible: bool | None = None,
    verbose: bool = True,
) -> dict:
    """Run apply_iss_transforms + detect_spots + base_calling for one well,
    keeping the registered (T, C, Z, Y, X) array in shared memory across
    all three steps.

    Parameters
    ----------
    experiment : str
        Experiment name (e.g. "ops0147_20260422").
    well
        Well identifier (row/col unit, e.g. "A/1/0" or "B2"; a bare int means row A).
    method : str
        Base calling method, passed through to ``base_calling``.
    skip_registered_zarr_write : bool
        If True, do NOT write bc_stitched_registered.zarr to disk. Default
        False because downstream consumers (convert_v3, iss_metrics, napari)
        still expect it.
    skip_detect_spots, skip_base_calling : bool
        For incremental testing — skip individual stages.
    reproducible : bool, optional
        Forwarded to detect_spots seed handling.
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict
        ``{"well": well, "wall_total_s": float, "wall_warp_s": ...,
           "wall_detect_s": ..., "wall_base_call_s": ..., "n_spots": int}``
    """
    from cyclops_process.processes.auto_register.iss_cycle_register import (
        apply_iss_transforms,
    )
    from cyclops_process.processes.spots import (
        detect_spots_for_well_in_memory,
        get_min_max,
    )
    from cyclops_process.processes.iss import base_calling

    t_total_start = time.monotonic()
    row, col = parse_well(well)
    well_token = f"{row}{col}"
    well_pos = f"{row}/{col}/0"

    if verbose:
        print(f"\n{'='*80}")
        print(f"merge_spots_base_calling_well: {experiment} well {well_token}")
        print(f"{'='*80}\n")

    dataset = OpsDataset(experiment)
    iss_zarr = dataset.store_paths["iss_stitch"]
    # Write the v2 registered store; convert_iss_to_v3 (after merge) produces the
    # proper v3 (pyramids/metadata). Writing v3 here would make convert skip.
    registered_zarr = dataset.store_paths["iss_stitch_registered"]
    register_root = dataset.preprocess_in_situ / "register"
    transforms_dir = register_root / f"transforms/{well_token}"

    # Anchor spots round — round 0 may have no spots (no-incorporation cycle).
    # Use the same anchor as ISS cycle registration so spot detection + the
    # normalization percentiles come from a round that actually has spots.
    _cfg_path = dataset.config_paths["exp_config"]
    if _cfg_path.exists():
        with open(_cfg_path) as f:
            _merge_cfg = yaml.safe_load(f) or {}
    else:
        _merge_cfg = {}
    anchor_round = int(_merge_cfg.get("anchor_round", 0) or 0)
    if verbose and anchor_round:
        print(f"anchor spots round = {anchor_round} (round 0 skipped for spot detection)")

    # === 1. Load cumulative transforms ======================================
    affines = _load_cumulative_transforms(transforms_dir)
    if verbose:
        print(f"loaded {len(affines)} cumulative transforms from {transforms_dir}")

    # === 2. Probe input zarr for shape ======================================
    import zarr as _zarr
    in_z = _zarr.open(f"{iss_zarr}/{well_pos}/0", mode="r")
    T, C, Z, Y, X = in_z.shape
    if verbose:
        print(f"input shape: T={T} C={C} Z={Z} Y={Y} X={X}")
        nbytes_gb = T * C * Z * Y * X * 4 / 1024**3
        print(f"will allocate registered shm: {nbytes_gb:.1f} GB float32")

    # === 3. Allocate shm for registered array ===============================
    nbytes = int(T * C * Z * Y * X * 4)
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    if verbose:
        print(f"allocated shm: {shm.name}")

    walls: dict = {}
    n_spots = 0

    try:
        # === 4. Warp: stitched zarr → shm (and optionally to registered zarr) ===
        out_zarr_arg: Path | None = None if skip_registered_zarr_write else registered_zarr
        if verbose:
            print(f"\n--- apply_iss_transforms ---")
        t = time.monotonic()
        apply_iss_transforms(
            input_zarr=iss_zarr,
            position=well_pos,
            affines_cumulative=affines,
            output_zarr=out_zarr_arg,
            output_shm_name=shm.name,
            output_shm_shape=(T, C, Z, Y, X),
            verbose=verbose,
        )
        walls["warp"] = time.monotonic() - t

        # === 5. Detect spots from shm-backed array ==========================
        if not skip_detect_spots:
            import random
            from spotiflow.model import Spotiflow

            # Resolve reproducible flag (mirrors detect_spots(experiment) logic):
            # explicit kwarg wins; else read experiment config; else default False.
            effective_reproducible = reproducible
            if effective_reproducible is None:
                import yaml as _yaml
                cfg_path = dataset.config_paths['exp_config']
                if cfg_path.exists():
                    with open(cfg_path, 'r') as f:
                        _cfg = _yaml.safe_load(f) or {}
                    effective_reproducible = _cfg.get(
                        'detect_spots_params', {}
                    ).get('reproducible', False)
                else:
                    effective_reproducible = False

            if effective_reproducible:
                # Seed before get_min_max — its random.sample() over positions
                # is the only source of run-to-run variance in detect_spots.
                random.seed(42)
                np.random.seed(42)
                if verbose:
                    print(f"  reproducible mode: seeded random/np.random with 42")

            if verbose:
                print(f"\n--- detect_spots_for_well_in_memory ---")
            t = time.monotonic()
            peak_model = Spotiflow.from_pretrained("general")
            p1_dict, p998_dict = get_min_max(experiment, anchor_round=anchor_round)
            pos_name = well_token  # e.g. "A1", "B2"

            registered_view = np.ndarray(
                (T, C, Z, Y, X), dtype=np.float32, buffer=shm.buf,
            )
            try:
                points = detect_spots_for_well_in_memory(
                    registered_view, well_pos, dataset, peak_model,
                    p1_dict[pos_name], p998_dict[pos_name],
                    anchor_round=anchor_round, save=True, verbose=verbose,
                )
                n_spots = int(len(points))
            finally:
                registered_view = None  # noqa: F841
            walls["detect"] = time.monotonic() - t

        # === 6. Base call from shm-backed array =============================
        if not skip_base_calling:
            if verbose:
                print(f"\n--- base_calling (well {well_token}, shm-backed) ---")
            t = time.monotonic()
            base_calling(
                experiment=experiment,
                method=method,
                well_pos=well_pos,
                registered_shm_name=shm.name,
                registered_shm_shape=(T, C, Z, Y, X),
            )
            walls["base_call"] = time.monotonic() - t

        walls["total"] = time.monotonic() - t_total_start

        if verbose:
            print(f"\n=== merge wall summary (well {well_token}) ===")
            for k in ("warp", "detect", "base_call", "total"):
                if k in walls:
                    print(f"  {k:>10}: {walls[k]:.1f}s")
            print(f"  n_spots: {n_spots:,}")

        return {
            "experiment": experiment,
            "well": well,
            "row": row,
            "n_spots": n_spots,
            **{f"wall_{k}_s": v for k, v in walls.items()},
        }

    finally:
        # Always release shm — caller is the merge step itself.
        try:
            shm.close()
        finally:
            shm.unlink()
        if verbose:
            print(f"released shm")


def _precreate_registered_zarr(experiment: str, well_specs: list, verbose: bool = True):
    """Pre-create bc_stitched_registered.zarr with one position per well, sized
    to that well's source stitched shape.

    Per-well stitched canvases can differ in Y/X by a few pixels, so each
    position is created at its own shape. If a stale position already exists
    with a mismatched shape, it is removed first so it gets recreated.
    """
    import iohub
    from cyclops_utils.io.zarr_utils import ensure_position_array
    import zarr as _zarr
    import shutil as _shutil

    dataset = OpsDataset(experiment)
    iss_zarr = dataset.store_paths["iss_stitch"]
    registered_zarr = dataset.store_paths["iss_stitch_registered"]  # v2; convert_iss_to_v3 makes v3

    # Keyed by full position so wells in different rows never collide.
    well_shapes: dict[str, tuple] = {}
    channel_names = None
    for row, col in well_specs:
        pos = f"{row}/{col}/0"
        in_z = _zarr.open(f"{iss_zarr}/{pos}/0", mode="r")
        well_shapes[pos] = tuple(in_z.shape)  # (T, C, Z, Y, X)
        if channel_names is None:
            with iohub.open_ome_zarr(iss_zarr / pos, layout="fov", mode="r") as st:
                channel_names = list(st.channel_names)

    if verbose:
        print(f"  precreating registered zarr: {registered_zarr}")
        for pos, sh in well_shapes.items():
            print(f"    {pos}: shape={sh}")

    # If the dst exists with any per-well shape mismatch, rebuild it from
    # scratch. Per-position surgery isn't reliable: removing just the
    # position dir leaves the well's .zattrs dangling, and iohub's
    # create_position then fails to recreate the well. The data is fully
    # derived from iss_stitch + transforms, so a full rebuild is safe.
    if registered_zarr.exists():
        rebuild = False
        for pos, expected_shape in well_shapes.items():
            pos_array = registered_zarr / pos / "0"
            if not pos_array.exists():
                rebuild = True
                if verbose:
                    print(f"    {pos} missing in existing store; will rebuild")
                break
            try:
                existing_shape = tuple(_zarr.open(str(pos_array), mode="r").shape)
            except Exception as e:
                rebuild = True
                if verbose:
                    print(f"    {pos} unreadable ({e}); will rebuild")
                break
            if existing_shape != expected_shape:
                rebuild = True
                if verbose:
                    print(
                        f"    shape mismatch for {pos}: existing={existing_shape}, "
                        f"expected={expected_shape}; will rebuild"
                    )
                break
        if rebuild:
            if verbose:
                print(f"  removing stale registered zarr to rebuild from scratch")
            _shutil.rmtree(registered_zarr)

    store_mode = "w" if not registered_zarr.exists() else "r+"
    with iohub.open_ome_zarr(
        registered_zarr, layout="hcs", mode=store_mode,
        channel_names=channel_names, version="0.4",  # v2; convert_iss_to_v3 makes v3
    ) as store_out:
        for pos, expected_shape in well_shapes.items():
            ensure_position_array(
                store_out, pos,
                shape=expected_shape,
                chunk_size=(1, 1, 1, 4096, 4096),
                dtype=np.float32,
                scale=[1, 1, 1, 1, 1],
            )


def merge_spots_base_calling(
    experiment: str,
    slurm_params: dict | None = None,
    wait_for_completion: bool = True,
    verbose: bool = True,
) -> dict:
    """Per-experiment orchestrator that fans out ``merge_spots_base_calling_well``
    jobs (one per well) in parallel.

    Replaces the wave-2 finalize ``apply_iss_transforms`` call AND the
    standalone ``detect_spots`` and ``base_calling`` steps for ISS. Run
    after ``register_iss_cycles(..., skip_apply_transforms=True)`` so the
    cumulative transform YAMLs exist but the registered zarr is left to
    this step to produce.

    Same head-node-orchestrator pattern as ``register_iss_cycles``.
    """
    import yaml
    from pathlib import Path
    from cyclops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    from cyclops_process.pipelinerunner.exceptions import PipelineHalted

    dataset = OpsDataset(experiment)
    exp_config_path = dataset.config_paths["exp_config"]

    if exp_config_path.exists():
        with open(exp_config_path, "r") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    wells_to_process = config.get("wells_to_process", ["A/1/0", "A/2/0", "A/3/0"])
    # Carry (row, col) so wells in different rows (A1 vs B1) don't collide.
    well_specs = []
    for w in wells_to_process:
        parts = str(w).split("/")
        if len(parts) >= 2:
            well_specs.append((parts[0], int(parts[1])))
        else:
            well_specs.append(("A", int(w)))

    if verbose:
        print(f"\n=== merge_spots_base_calling: {experiment} ===")
        print(f"Wells to process: {[f'{r}{c}' for r, c in well_specs]}")

    if slurm_params is None:
        cyclops_process_dir = str(Path(__file__).parents[2])
        slurm_params = {
            "timeout_min": 30,
            "mem": "400GB",
            "cpus_per_task": 32,
            "slurm_partition": "gpu",
            "slurm_gres": "gpu:1",
            "slurm_gpus_per_node": 1,  # one well-job per node — prevents two
                                       # wells contending for the same GPU
            "slurm_constraint": "[a100_80|h100|h200]",
            "slurm_srun_args": ["--cpu-bind=none"],
            "slurm_setup": [
                f"export PYTHONPATH={cyclops_process_dir}:$PYTHONPATH",
                "export OMP_NUM_THREADS=1",
                "export MKL_NUM_THREADS=1",
                "export OPENBLAS_NUM_THREADS=1",
                "export NUMEXPR_NUM_THREADS=1",
            ],
        }

    # Pre-create registered zarr structure once so per-well jobs don't race.
    _precreate_registered_zarr(experiment, well_specs, verbose=verbose)

    jobs_to_submit = [
        {
            "name": f"w{row}{col}_merge",
            "func": merge_spots_base_calling_well,
            "kwargs": {"experiment": experiment, "well": f"{row}/{col}/0"},
            "metadata": {"type": "merge", "well": f"{row}/{col}/0"},
        }
        for (row, col) in well_specs
    ]

    log_dir = f"slurm_step_logs/{experiment}"

    result = submit_parallel_jobs(
        jobs_to_submit=jobs_to_submit,
        experiment=experiment,
        slurm_params=slurm_params,
        log_dir=log_dir,
        manifest_prefix=f"merge_spots_base_calling_{experiment}",
        step_name='merge_spots_base_calling',
        wait_for_completion=wait_for_completion,
        verbose=verbose,
    )

    # submit_parallel_jobs returns success=True once the array is submitted, so
    # without this a failed well reads as a completed step.
    failed = result.get("failed") or []
    if failed:
        raise PipelineHalted(
            f"merge_spots_base_calling failed for {len(failed)}/"
            f"{len(jobs_to_submit)} wells: {failed}. "
            f"See slurm_logs/{log_dir}/<job_id>/*_log.err "
            f"(array {result.get('base_job_id')})."
        )

    return result
