"""
SLURM batch submission for pyramid building.

Supports two parallelization modes:
- Unit-level (default): Each (position, t, c) unit is a separate SLURM job (max parallelism)
- Position-level (--per-position): Each position is a separate SLURM job

Supports both zarr v2 and v3 stores via --zarr-version flag.

Usage:
------
# Submit pyramid build for v3 store (default)
python -m cyclops_process.processes.pyramids.launcher -e ops0033

# Submit pyramid build for v2 store
python -m cyclops_process.processes.pyramids.launcher -e ops0033 --zarr-version 2

# Force rebuild all pyramids (v3)
python -m cyclops_process.processes.pyramids.launcher -e ops0033 --force

# Force rebuild all pyramids (v2)
python -m cyclops_process.processes.pyramids.launcher -e ops0033 --zarr-version 2 --force

# Position-level parallelism - each position processes all (t, c) sequentially
python -m cyclops_process.processes.pyramids.launcher -e ops0033 --per-position

# Dry run (show what would be submitted)
python -m cyclops_process.processes.pyramids.launcher -e ops0033 --dry-run

# Process specific wells only
python -m cyclops_process.processes.pyramids.launcher -e ops0033 --wells A/1/0 A/2/0

# Skip overlays (grid, ISS, clims) - only build pyramids
python -m cyclops_process.processes.pyramids.launcher -e ops0033 --skip-overlays

# Submit and don't wait for completion
python -m cyclops_process.processes.pyramids.launcher -e ops0033 --no-wait

# Use a specific store key
python -m cyclops_process.processes.pyramids.launcher -e ops0033 --store pheno_assembled
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Sequence
from cyclops_utils.profiling.decorators import versioned_function

# Add project root to path
sys.path.insert(0, os.getcwd())

from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.hpc.slurm_batch_utils import (
    submit_parallel_jobs,
    wait_for_multiple_job_arrays,
    handle_single_experiment_cli,
)


# Native level-0 YX spacing (µm/px) by store type. Pheno (20x) is 0.325; track
# (5x) is 4x coarser = 1.3. Each pyramid level halves resolution, so level i is
# native * 2**i. The upstream optimized stores have historically mis-declared
# YX (e.g. 0.65 for both), so we stamp the canonical pyramid here at build time
# rather than trusting whatever level-0 happens to carry. ISS spacing comes from
# an independent path — left untouched (None).
_NATIVE_YX_BY_STORE = {"pheno": 0.325, "track": 1.3}


def _store_type_for_yx(store_key: str) -> Optional[str]:
    """Classify a store key for YX-scale stamping: 'pheno', 'track', or None
    (iss / unknown — leave its declared spacing alone)."""
    k = (store_key or "").lower()
    if "iss" in k:
        return None
    if "lc_5x" in k or "tracking" in k or "track" in k:
        return "track"
    if "pheno" in k or "lc_20x" in k:
        return "pheno"
    return None


def _stamp_canonical_yx_scale(
    store_path: Path,
    positions: Sequence[str],
    store_key: str,
    factor: int = 2,
) -> None:
    """Rewrite each position's multiscale ``datasets`` so level i has YX spacing
    ``native * factor**i`` (native from :data:`_NATIVE_YX_BY_STORE`). Idempotent;
    repairs both an over/under-declared level 0 and a malformed coarsest level
    (which a blind per-level halving would corrupt). No-op for ISS/unknown."""
    import json

    store_type = _store_type_for_yx(store_key)
    native = _NATIVE_YX_BY_STORE.get(store_type)
    if native is None:
        return
    fixed = 0
    for position in positions:
        pj = Path(store_path) / str(position) / "zarr.json"
        if not pj.exists():
            continue
        try:
            d = json.loads(pj.read_text())
            attrs = d.get("attributes", {})
            ome = attrs.get("ome") or attrs
            ms = ome.get("multiscales")
            if not ms:
                continue
            changed = False
            for i, ds in enumerate(ms[0].get("datasets", [])):
                level_yx = native * (factor ** i)
                for ct in ds.get("coordinateTransformations", []):
                    if ct.get("type") == "scale" and len(ct.get("scale", [])) >= 2:
                        if ct["scale"][-1] != level_yx or ct["scale"][-2] != level_yx:
                            ct["scale"][-1] = level_yx
                            ct["scale"][-2] = level_yx
                            changed = True
            if changed:
                pj.write_text(json.dumps(d, indent=2))
                fixed += 1
        except Exception as e:
            print(f"  [yx-scale] WARN {position}: {e}")
    if fixed:
        print(f"  [yx-scale] stamped canonical {store_type} YX scale "
              f"(level0={native}) on {fixed} position(s)")


POSITION_SLURM_PARAMS = {
    "timeout_min": 30,
    "mem": "200GB",
    "cpus_per_task": 32,
    "slurm_partition": "cpu",
}

# Per-(pos, t, c) seg jobs — work ~1-2 min; wall capped at 8min.
UNIT_SLURM_PARAMS = {
    "timeout_min": 8,
    "mem": "100GB",
    "cpus_per_task": 16,
    "slurm_partition": "cpu",
}

# Per-(pos, t, c) image — pheno L0 is huge (~104k²); load+downsample ran ~7min
# and timed out at 8, so cap at 12min.
IMAGE_SLURM_PARAMS = {
    "timeout_min": 12,
    "mem": "200GB",
    "cpus_per_task": 16,
    "slurm_partition": "cpu",
}

# ISS/track stores are tiny vs pheno. 8min wall.
SMALL_STORE_UNIT_SLURM_PARAMS = {
    "timeout_min": 8,
    "mem": "64GB",
    "cpus_per_task": 8,
    "slurm_partition": "cpu",
}

# Grid overlay holds full RGBA canvas in memory (~44GB pheno L0).
OVERLAY_SLURM_PARAMS = {
    "timeout_min": 8,
    "mem": "200GB",
    "cpus_per_task": 16,
    "slurm_partition": "cpu",
}

# Reshard L2-L4 — small (≤19GB raw); cheap node.
RESHARD_SLURM_PARAMS = {
    "timeout_min": 8,
    "mem": "64GB",
    "cpus_per_task": 8,
    "slurm_partition": "cpu",
}

# Reshard L1 — I/O-bound (16 shard-tile threads); CPUs above 16 don't help.
RESHARD_L1_SLURM_PARAMS = {
    "timeout_min": 8,
    "mem": "350GB",
    "cpus_per_task": 16,
    "slurm_partition": "cpu",
}


def _get_unit_slurm_params(store_key: str = None) -> dict:
    """Return SLURM params appropriate for the given store key."""
    if store_key and ("iss" in store_key or "lc_5x" in store_key or "tracking" in store_key):
        return SMALL_STORE_UNIT_SLURM_PARAMS
    return UNIT_SLURM_PARAMS


from cyclops_process.processes.pyramids.workers import (
    build_pyramid_unit_worker,
    build_seg_unit_worker,
    build_overlays_worker,
    reshard_level_worker,
    build_position_pyramids_worker,
)
def get_positions_for_experiment(
    experiment: str,
    store_key: str = None,
    wells: Optional[Sequence[str]] = None,
    zarr_version: int = 3,
) -> tuple[Path, list[str]]:
    """
    Get all positions in an experiment's zarr store.

    Parameters
    ----------
    experiment : str
        Experiment name
    store_key : str, optional
        Store key to use (overrides zarr_version). If None, uses pheno store for zarr_version.
    wells : Optional[Sequence[str]]
        Filter to specific wells (e.g., ["A/1/0", "A/2/0"])
    zarr_version : int
        Zarr format version (2 or 3, default: 3)

    Returns
    -------
    tuple[Path, list[str], str]
        (store_path, list of position paths, used store key)
    """
    from cyclops_utils.io.zarr_utils import _iter_position_paths

    ds = OpsDataset(experiment)

    # Determine store key based on zarr version if not explicitly provided
    if store_key:
        keys_to_try = [store_key]
    elif zarr_version == 3:
        keys_to_try = ["pheno_assembled_v3", "lc_5x_phase_2d_stitched_v3", "iss_stitch_registered_v3"]
    else:  # v2
        keys_to_try = ["pheno_assembled", "lc_5x_phase_2d_stitched", "iss_stitch_registered"]

    store_path = None
    used_key = None
    for key in keys_to_try:
        if key in ds.store_paths and ds.store_paths[key].exists():
            store_path = ds.store_paths[key]
            used_key = key
            break

    if store_path is None:
        raise ValueError(f"No valid v{zarr_version} store found for {experiment}. Tried: {keys_to_try}")

    print(f"Using store: {used_key} ({store_path})")

    # Get all positions
    all_positions = list(_iter_position_paths(store_path))

    # Filter by wells if specified
    if wells:
        positions = [
            p for p in all_positions
            if any(str(p) == str(w) or str(p).startswith(str(w)) for w in wells)
        ]
        print(f"Selected {len(positions)}/{len(all_positions)} positions matching wells: {wells}")
    else:
        positions = all_positions
        print(f"Found {len(positions)} positions")

    return store_path, positions, used_key


def get_units_for_positions(
    source_store: Path,
    positions: list[str],
) -> list[tuple[str, int, int]]:
    """
    Get all (position, t, c) units for the given positions.

    Parameters
    ----------
    source_store : Path
        Path to zarr store
    positions : list[str]
        List of position paths

    Returns
    -------
    list[tuple[str, int, int]]
        List of (position, t, c) tuples
    """
    from cyclops_utils.io.zarr_utils import enumerate_units
    return enumerate_units(source_store, positions)


def submit_pyramid_job(
    experiment: str,
    slurm_params: dict,
    args,
) -> dict:
    """
    Submit SLURM jobs for pyramid building.

    Supports two modes:
    - Unit-level (default): Each (pos, t, c) is a job (max parallelism)
    - Position-level (--per-position): Each position is a job

    Parameters
    ----------
    experiment : str
        Experiment name
    slurm_params : dict
        SLURM parameters (ignored, uses internal params)
    args : argparse.Namespace
        CLI arguments

    Returns
    -------
    dict
        Job submission results
    """
    # Parse wells from args
    wells = getattr(args, 'wells', None)
    per_position = getattr(args, 'per_position', False)
    zarr_version = getattr(args, 'zarr_version', 3)

    # Get store and positions
    try:
        store_path, positions, used_key = get_positions_for_experiment(
            experiment,
            store_key=getattr(args, 'store', None),
            wells=wells,
            zarr_version=zarr_version,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return {"success": False, "error": str(e)}

    if not positions:
        print(f"No positions found for {experiment}")
        return {"success": False, "error": "No positions found"}

    # Check if we should resume, overwrite, or skip
    resume = True
    if args.force:
        resume = False
        print("Force mode: will rebuild all pyramid levels")

    # Select SLURM params based on store type
    unit_params = _get_unit_slurm_params(used_key)

    if per_position:
        # Position-level mode: each position is a job (less parallelism)
        return submit_per_position_jobs(experiment, store_path, positions, args, resume)
    else:
        # Fine-grained mode (default): each (pos, t, c) unit is a separate job
        return submit_per_unit_jobs(experiment, store_path, positions, args, resume, unit_params=unit_params)


def submit_per_position_jobs(
    experiment: str,
    store_path: Path,
    positions: list[str],
    args,
    resume: bool,
) -> dict:
    """Submit one job per position."""

    jobs_to_submit = []
    for position in positions:
        job_name = f"pyramid_{experiment}_{position.replace('/', '_')}"

        job_spec = {
            "name": job_name,
            "func": build_position_pyramids_worker,
            "kwargs": {
                "experiment": experiment,
                "position": position,
                "source_store": str(store_path),
                "levels": getattr(args, 'levels', 5),
                "factor": getattr(args, 'factor', 2),
                "grid_line_width": getattr(args, 'grid_line_width', 1),
                "skip_overlays": getattr(args, 'skip_overlays', False),
                "resume": resume,
            },
            "metadata": {
                "experiment": experiment,
                "position": position,
                "store": str(store_path),
            },
        }
        jobs_to_submit.append(job_spec)

    print(f"\nSubmitting {len(jobs_to_submit)} position-level jobs")

    # Submit jobs
    result = submit_parallel_jobs(
        jobs_to_submit=jobs_to_submit,
        experiment=experiment,
        slurm_params=POSITION_SLURM_PARAMS,
        log_dir=f"slurm_pyramid_logs/{experiment}",
        manifest_prefix="pyramid",
        dry_run=args.dry_run,
        wait_for_completion=not args.no_wait,
        verbose=not args.quiet,
        post_completion_callback=None,
    )

    return result


def _seg_unit_needs_build(source_path, position, seg_name, levels, in_labels_group) -> bool:
    """True if any pyramid level for this seg label is missing or empty.
    Mirrors build_seg_unit_worker's resume check so submission can skip
    already-built units instead of launching a no-op SLURM job."""
    from cyclops_process.processes.pyramids.build_dask import (
        _get_seg_dir_path, _get_seg_component_path, _is_zero_like_component,
    )
    for lvl in range(1, int(levels)):
        lvl_path = _get_seg_dir_path(source_path, position, seg_name, lvl, in_labels_group)
        lvl_component = _get_seg_component_path(position, seg_name, lvl, in_labels_group)
        if (not lvl_path.exists()) or _is_zero_like_component(source_path, lvl_component):
            return True
    return False


def submit_per_unit_jobs(
    experiment: str,
    store_path: Path,
    positions: list[str],
    args,
    resume: bool,
    unit_params: dict = None,
) -> dict:
    """
    Submit jobs at the (pos, t, c) unit level for maximum parallelism.

    Submits three arrays:
    1. Image pyramid units
    2. Segmentation pyramid units
    3. Overlay jobs (after pyramids complete, if not skipped)
    """
    if unit_params is None:
        unit_params = UNIT_SLURM_PARAMS
    from cyclops_utils.io.zarr_utils import detect_zarr_format, get_channel_dim, ensure_pyramid_levels_unsharded, ensure_pyramid_levels
    from iohub import open_ome_zarr
    from tqdm import tqdm
    import dask.array as da

    # Detect zarr format
    zarr_format = detect_zarr_format(store_path)
    in_labels_group = (zarr_format == 3)
    print(f"Detected zarr format: v{zarr_format}")

    # Pre-initialize pyramid levels for ALL positions BEFORE submitting parallel jobs
    # This avoids race conditions where multiple workers try to initialize the same position
    levels = getattr(args, 'levels', 5)
    factor = getattr(args, 'factor', 2)
    force = not resume  # force=True when --force flag is used (resume=False)

    # Init UNSHARDED for v3 — per-(pos,t,c) parallel writes are race-free (chunk-per-file).
    # Reshard step at the end consolidates into the final all-C-packed shard layout.
    print(f"\nInitializing pyramid levels for {len(positions)} positions...")
    if zarr_format == 3:
        for position in tqdm(positions, desc="Initializing pyramids"):
            ensure_pyramid_levels_unsharded(store_path, position, levels, force=force, factor=factor)
    else:
        for position in tqdm(positions, desc="Initializing pyramids"):
            ensure_pyramid_levels(store_path, position, levels, force=force)

    # Stamp canonical native-derived YX spacing per level (pheno 0.325, track 1.3)
    # so the multiscale records the correct physical size regardless of upstream.
    _stamp_canonical_yx_scale(store_path, positions, str(store_path), factor=factor)

    # Build image pyramid unit jobs
    image_jobs = []
    seg_jobs = []

    print(f"\nEnumerating (position, t, c) units...")

    # Completion-aware submission: skip units whose pyramid levels are already
    # built so a re-run only submits the missing work (e.g. just the ISS store)
    # instead of relaunching every unit. Bypassed entirely under --force.
    from cyclops_process.napari.dask.dask_utils import determine_target_levels
    img_levels = getattr(args, 'levels', 5)
    n_img_skipped = 0
    n_seg_skipped = 0

    for position in positions:
        # Get dimensions for this position
        with open_ome_zarr(store_path, mode="r") as store:
            fov = store[position]
            t_dim = int(fov.data.shape[0]) if fov.data.ndim >= 1 else 1
            c_dim = int(get_channel_dim(store_path, position))

        # Per-(pos, t, c) image jobs — one channel each, race-free on unsharded init.
        for t in range(t_dim):
            for c in range(c_dim):
                if resume and not determine_target_levels(
                    store_path, position, img_levels, resume=True, t=t, c=c
                ):
                    n_img_skipped += 1
                    continue  # all image levels already built for this unit
                image_jobs.append({
                    "name": f"img_{experiment}_{position.replace('/', '_')}_t{t}_c{c}",
                    "func": build_pyramid_unit_worker,
                    "kwargs": {
                        "experiment": experiment, "position": position, "t": t, "c": c,
                        "source_store": str(store_path),
                        "levels": getattr(args, 'levels', 5),
                        "factor": getattr(args, 'factor', 2),
                        "resume": resume,
                    },
                    "metadata": {"experiment": experiment, "position": position, "t": t, "c": c, "type": "image"},
                })

        # Check for segmentation data and create seg jobs (unless --skip-seg)
        skip_seg = getattr(args, 'skip_seg', False)
        if not skip_seg:
            for seg_name in ["seg", "nuclear_seg"]:
                if in_labels_group:
                    seg_exists = (store_path / position / "labels" / seg_name / "0").exists()
                else:
                    seg_exists = (store_path / position / seg_name / "0").exists()

                if seg_exists:
                    # Get seg dimensions
                    if in_labels_group:
                        seg_component = f"{position}/labels/{seg_name}/0"
                    else:
                        seg_component = f"{position}/{seg_name}/0"

                    try:
                        seg_arr = da.from_zarr(str(store_path), component=seg_component)
                        if seg_arr.ndim >= 5:
                            seg_t, seg_c = int(seg_arr.shape[0]), int(seg_arr.shape[1])
                        elif seg_arr.ndim == 4:
                            seg_t, seg_c = int(seg_arr.shape[0]), int(seg_arr.shape[1])
                        else:
                            seg_t, seg_c = 1, 1

                        if resume and not _seg_unit_needs_build(
                            store_path, position, seg_name, img_levels, in_labels_group
                        ):
                            n_seg_skipped += seg_t * seg_c
                            continue  # all levels already built for this seg label

                        from cyclops_process.processes.pyramids.build_dask import _init_seg_levels
                        _init_seg_levels(store_path, [position], getattr(args, 'levels', 5), seg_name, in_labels_group=in_labels_group)
                        for t in range(seg_t):
                            for c in range(seg_c):
                                job_name = f"{seg_name}_{experiment}_{position.replace('/', '_')}_t{t}_c{c}"
                                seg_jobs.append({
                                    "name": job_name,
                                    "func": build_seg_unit_worker,
                                    "kwargs": {
                                        "experiment": experiment,
                                        "position": position,
                                        "t": t,
                                        "c": c,
                                        "seg_name": seg_name,
                                        "source_store": str(store_path),
                                        "levels": getattr(args, 'levels', 5),
                                        "in_labels_group": in_labels_group,
                                        "preserve_dtype": False,
                                        "resume": resume,
                                    },
                                    "metadata": {
                                        "experiment": experiment,
                                        "position": position,
                                        "t": t,
                                        "c": c,
                                        "seg_name": seg_name,
                                        "type": "seg",
                                    },
                                })
                    except Exception as e:
                        print(f"Warning: Could not read {seg_component}: {e}")

        # Check for organelle labels (unless --skip-seg)
        # Exclude ISS overlays (iss_gene_image, iss_guide_image, grid_overlay) - they're RGBA images, not TCZYX segmentations
        if not skip_seg:
            labels_dir = store_path / position / "labels"
            skip_labels = {"seg", "nuclear_seg", "iss_gene_image", "iss_guide_image", "grid_overlay"}
            if labels_dir.exists():
                for subdir in labels_dir.iterdir():
                    if subdir.is_dir() and subdir.name not in skip_labels:
                        if (subdir / "0").exists():
                            label_name = subdir.name
                            seg_component = f"{position}/labels/{label_name}/0"
                            try:
                                seg_arr = da.from_zarr(str(store_path), component=seg_component)
                                if seg_arr.ndim >= 5:
                                    seg_t, seg_c = int(seg_arr.shape[0]), int(seg_arr.shape[1])
                                elif seg_arr.ndim == 4:
                                    seg_t, seg_c = int(seg_arr.shape[0]), int(seg_arr.shape[1])
                                else:
                                    seg_t, seg_c = 1, 1

                                if resume and not _seg_unit_needs_build(
                                    store_path, position, label_name, img_levels, True
                                ):
                                    n_seg_skipped += seg_t * seg_c
                                    continue  # all levels already built for this label

                                from cyclops_process.processes.pyramids.build_dask import _init_seg_levels
                                _init_seg_levels(store_path, [position], getattr(args, 'levels', 5), label_name, in_labels_group=True, preserve_dtype=True)
                                for t in range(seg_t):
                                    for c in range(seg_c):
                                        job_name = f"{label_name}_{experiment}_{position.replace('/', '_')}_t{t}_c{c}"
                                        seg_jobs.append({
                                            "name": job_name,
                                            "func": build_seg_unit_worker,
                                            "kwargs": {
                                                "experiment": experiment,
                                                "position": position,
                                                "t": t,
                                                "c": c,
                                                "seg_name": label_name,
                                                "source_store": str(store_path),
                                                "levels": getattr(args, 'levels', 5),
                                                "in_labels_group": True,
                                                "preserve_dtype": True,
                                                "resume": resume,
                                            },
                                            "metadata": {
                                                "experiment": experiment,
                                                "position": position,
                                                "t": t,
                                                "c": c,
                                                "seg_name": label_name,
                                                "type": "organelle_seg",
                                            },
                                        })
                            except Exception as e:
                                print(f"Warning: Could not read {seg_component}: {e}")

    print(f"Found {len(image_jobs)} image units, {len(seg_jobs)} segmentation units")
    if n_img_skipped or n_seg_skipped:
        print(f"  (completion-aware: skipped {n_img_skipped} image + {n_seg_skipped} seg units already built; use --force to rebuild)")

    if args.dry_run:
        reshard_count = len(positions) * (levels - 1) if zarr_format == 3 else 0
        overlay_count = 0 if getattr(args, 'skip_overlays', False) else len(positions)
        print(f"\n{'='*60}")
        print(f"DRY RUN: Job Submission Plan")
        print(f"{'='*60}\n")
        print(f"Would submit {len(image_jobs)} image pyramid jobs")
        print(f"Would submit {len(seg_jobs)} segmentation pyramid jobs")
        if reshard_count:
            print(f"Would submit {reshard_count} resharding jobs ({len(positions)} positions x {levels - 1} levels)")
        if overlay_count:
            print(f"Would submit {overlay_count} overlay jobs (after resharding)")
        print(f"\nTotal: {len(image_jobs) + len(seg_jobs) + reshard_count + overlay_count} jobs")
        print(f"\nDRY RUN: No jobs submitted\n")
        return {"dry_run": True, "image_jobs": len(image_jobs), "seg_jobs": len(seg_jobs)}

    job_arrays_to_monitor = []
    total_submitted = 0

    # Submit image pyramid jobs
    if image_jobs:
        print(f"\nSubmitting {len(image_jobs)} image pyramid jobs...")
        image_result = submit_parallel_jobs(
            jobs_to_submit=image_jobs,
            experiment=f"{experiment}_image_pyramids",
            slurm_params=IMAGE_SLURM_PARAMS,
            log_dir=f"slurm_pyramid_logs/{experiment}",
            manifest_prefix="pyramid_img",
            step_name='build_pyramids_image',
            dry_run=False,
            wait_for_completion=False,
            verbose=not args.quiet,
            post_completion_callback=None,
        )

        if image_result.get("success"):
            total_submitted += len(image_jobs)
            job_arrays_to_monitor.append({
                "submitted_jobs": image_result["submitted_jobs"],
                "base_job_id": image_result["base_job_id"],
                "label": "image",
                "slurm_params": IMAGE_SLURM_PARAMS,
            })

    # Submit segmentation pyramid jobs
    if seg_jobs:
        print(f"\nSubmitting {len(seg_jobs)} segmentation pyramid jobs...")
        seg_result = submit_parallel_jobs(
            jobs_to_submit=seg_jobs,
            experiment=f"{experiment}_seg_pyramids",
            slurm_params=unit_params,
            log_dir=f"slurm_pyramid_logs/{experiment}",
            manifest_prefix="pyramid_seg",
            step_name='build_pyramids_seg',
            dry_run=False,
            wait_for_completion=False,
            verbose=not args.quiet,
            post_completion_callback=None,
        )

        if seg_result.get("success"):
            total_submitted += len(seg_jobs)
            job_arrays_to_monitor.append({
                "submitted_jobs": seg_result["submitted_jobs"],
                "base_job_id": seg_result["base_job_id"],
                "label": "seg",
                "slurm_params": unit_params,
            })

    # Submit overlay jobs alongside pyramids (overlays read base data, not pyramid levels)
    overlay_jobs = []
    if not getattr(args, 'skip_overlays', False):
        for position in positions:
            job_name = f"overlay_{experiment}_{position.replace('/', '_')}"
            overlay_jobs.append({
                "name": job_name,
                "func": build_overlays_worker,
                "kwargs": {
                    "experiment": experiment,
                    "position": position,
                    "source_store": str(store_path),
                    "factor": getattr(args, 'factor', 2),
                    "grid_line_width": getattr(args, 'grid_line_width', 1),
                },
                "metadata": {
                    "experiment": experiment,
                    "position": position,
                    "type": "overlay",
                },
            })

        print(f"\nSubmitting {len(overlay_jobs)} overlay jobs...")
        overlay_result = submit_parallel_jobs(
            jobs_to_submit=overlay_jobs,
            experiment=f"{experiment}_overlays",
            slurm_params=OVERLAY_SLURM_PARAMS,
            log_dir=f"slurm_pyramid_logs/{experiment}",
            manifest_prefix="pyramid_overlay",
            step_name='build_pyramids_overlay',
            dry_run=False,
            wait_for_completion=False,
            verbose=not args.quiet,
            post_completion_callback=None,
        )

        if overlay_result.get("success"):
            total_submitted += len(overlay_jobs)
            job_arrays_to_monitor.append({
                "submitted_jobs": overlay_result["submitted_jobs"],
                "base_job_id": overlay_result["base_job_id"],
                "label": "overlay",
                "slurm_params": OVERLAY_SLURM_PARAMS,
            })

    # Phase 1: Wait for pyramid + overlay jobs together
    total_failed = 0
    if not args.no_wait and job_arrays_to_monitor:
        print(f"\n{'='*60}")
        print(f"Waiting for {total_submitted} pyramid + overlay jobs to complete...")
        print(f"{'='*60}\n")

        wait_results = wait_for_multiple_job_arrays(
            job_arrays=job_arrays_to_monitor,
            experiment=experiment,
            verbose=not args.quiet,
        )

        # Check for failures
        if wait_results.get("array_results"):
            for array_label, array_result in wait_results["array_results"].items():
                total_failed += len(array_result.get("failed", []))

        if total_failed > 0:
            print(f"\n⚠️  {total_failed} pyramid/overlay jobs failed")

    # Reshard unsharded levels into final all-C-packed shard layout.
    reshard_failed = 0
    if not args.no_wait and total_failed == 0 and zarr_format == 3:
        l1_jobs, smaller_jobs = [], []
        for position in positions:
            for level in range(1, levels):
                if (store_path / position / str(level)).exists():
                    job = {
                        "name": f"reshard_{experiment}_{position.replace('/', '_')}_L{level}",
                        "func": reshard_level_worker,
                        "kwargs": {"experiment": experiment, "position": position, "level": level, "source_store": str(store_path)},
                        "metadata": {"experiment": experiment, "position": position, "level": level, "type": "reshard"},
                    }
                    (l1_jobs if level == 1 else smaller_jobs).append(job)
        reshard_arrays = []
        for jobs, params, suffix in ((l1_jobs, RESHARD_L1_SLURM_PARAMS, "L1"), (smaller_jobs, RESHARD_SLURM_PARAMS, "L2plus")):
            if not jobs:
                continue
            print(f"\nSubmitting {len(jobs)} reshard {suffix} jobs...")
            r = submit_parallel_jobs(
                jobs_to_submit=jobs,
                experiment=f"{experiment}_reshard_{suffix}",
                slurm_params=params,
                log_dir=f"slurm_pyramid_logs/{experiment}",
                manifest_prefix=f"pyramid_reshard_{suffix}",
                dry_run=False, wait_for_completion=False, verbose=not args.quiet,
            )
            if r.get("success") and r.get("submitted_jobs"):
                reshard_arrays.append({
                    "submitted_jobs": r["submitted_jobs"],
                    "base_job_id": r["base_job_id"],
                    "label": f"reshard {suffix}",
                    "slurm_params": params,
                })
        if reshard_arrays:
            wait_results = wait_for_multiple_job_arrays(job_arrays=reshard_arrays, experiment=experiment, verbose=not args.quiet)
            reshard_failed = sum(len(a.get("failed", [])) for a in wait_results.get("array_results", {}).values())
    return {
        "success": total_failed == 0 and reshard_failed == 0,
        "image_jobs": len(image_jobs),
        "seg_jobs": len(seg_jobs),
        "overlay_jobs": len(overlay_jobs),
        "total_submitted": total_submitted,
        "pyramid_failed": total_failed,
        "reshard_failed": reshard_failed,
    }


def main():
    """CLI entry point for SLURM batch pyramid building."""
    parser = argparse.ArgumentParser(
        description="Submit pyramid building jobs to SLURM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--experiment", "-e", type=str, required=True,
        help="Experiment name (e.g., ops0033 or ops0033_20250429)",
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Force rebuild even if pyramids exist",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be submitted without actually submitting",
    )
    parser.add_argument(
        "--per-position", action="store_true",
        help="Submit one job per position instead of per (position, t, c) unit (less parallelism)",
    )
    parser.add_argument(
        "--wells", "-w", nargs="+", type=str, default=None,
        help="Specific wells to process (e.g., A/1/0 A/2/0)",
    )
    parser.add_argument(
        "--store", type=str, default=None,
        help="Store key to use (overrides --zarr-version). If not specified, uses pheno store for the selected zarr version.",
    )
    parser.add_argument(
        "--zarr-version", type=int, choices=[2, 3], default=3,
        help="Zarr format version to build pyramids for (default: 3)",
    )
    parser.add_argument(
        "--levels", type=int, default=5,
        help="Number of pyramid levels (default: 5)",
    )
    parser.add_argument(
        "--factor", type=int, default=2,
        help="Downsampling factor between levels (default: 2)",
    )
    parser.add_argument(
        "--grid-line-width", type=int, default=1,
        help="Width of grid overlay lines in pixels (default: 1)",
    )
    parser.add_argument(
        "--skip-overlays", action="store_true",
        help="Skip building overlays (clims, grid, ISS) - only build pyramids",
    )
    parser.add_argument(
        "--skip-seg", action="store_true",
        help="Skip building segmentation pyramids - only build base image pyramids",
    )
    parser.add_argument(
        "--no-wait", action="store_true",
        help="Submit jobs and return immediately without waiting for completion",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Reduce verbosity",
    )
    parser.add_argument(
        "--reshard-only", action="store_true",
        help="Skip pyramid build; just submit reshard jobs against existing on-disk levels.",
    )

    args = parser.parse_args()

    if args.reshard_only:
        sys.exit(_run_reshard_only(args))

    # Use the standard single experiment CLI handler
    exit_code = handle_single_experiment_cli(
        submit_func=submit_pyramid_job,
        args=args,
        slurm_params=POSITION_SLURM_PARAMS,
    )
    sys.exit(exit_code)


def _run_reshard_only(args) -> int:
    """Submit ONLY the reshard jobs across all v3 stores for this experiment."""
    from cyclops_utils.data.filesystem import resolve_experiment_name
    from cyclops_utils.io.zarr_utils import _iter_position_paths, detect_zarr_format
    experiment = resolve_experiment_name(args.experiment, autoselect=True)
    ds = OpsDataset(experiment)
    candidates = ["pheno_assembled_v3", "lc_5x_phase_2d_stitched_v3", "iss_stitch_registered_v3"]
    if args.store:
        candidates = [args.store]
    existing = [(k, ds.store_paths[k]) for k in candidates
                if ds.store_paths.get(k) and ds.store_paths[k].exists()
                and detect_zarr_format(ds.store_paths[k]) == 3]
    if not existing:
        print("No v3 stores found.")
        return 1
    l1, l2plus = [], []
    for key, sp in existing:
        if args.wells:
            positions = [p for p in _iter_position_paths(sp)
                         if any(str(p) == str(w) or str(p).startswith(str(w)) for w in args.wells)]
        else:
            positions = list(_iter_position_paths(sp))
        for position in positions:
            for lvl in range(1, args.levels):
                if (sp / position / str(lvl)).exists():
                    job = {
                        "name": f"reshard_{key}_{position.replace('/', '_')}_L{lvl}",
                        "func": reshard_level_worker,
                        "kwargs": {"experiment": experiment, "position": position, "level": lvl, "source_store": str(sp)},
                        "metadata": {"experiment": experiment, "store_key": key, "position": position, "level": lvl},
                    }
                    (l1 if lvl == 1 else l2plus).append(job)
    # Submit both arrays without waiting, then monitor together (one watch).
    job_arrays_to_monitor = []
    for jobs, params, suffix in ((l1, RESHARD_L1_SLURM_PARAMS, "L1"), (l2plus, RESHARD_SLURM_PARAMS, "L2plus")):
        if not jobs:
            continue
        print(f"\nSubmitting {len(jobs)} reshard {suffix} jobs...")
        r = submit_parallel_jobs(
            jobs_to_submit=jobs, experiment=f"{experiment}_reshard_only_{suffix}",
            slurm_params=params,
            log_dir=f"slurm_pyramid_logs/{experiment}",
            manifest_prefix=f"reshard_only_{suffix}",
            dry_run=args.dry_run, wait_for_completion=False,
            verbose=not args.quiet,
        )
        if r.get("success") and r.get("submitted_jobs"):
            job_arrays_to_monitor.append({
                "submitted_jobs": r["submitted_jobs"],
                "base_job_id": r["base_job_id"],
                "label": f"reshard {suffix}",
                "slurm_params": params,
            })
    if args.no_wait or not job_arrays_to_monitor:
        return 0
    wait_results = wait_for_multiple_job_arrays(
        job_arrays=job_arrays_to_monitor, experiment=experiment, verbose=not args.quiet,
    )
    failed = sum(len(a.get("failed", [])) for a in wait_results.get("array_results", {}).values())
    return 0 if failed == 0 else 1


# =============================================================================
# Pipeline-compatible functions (called by orchestrator / Nextflow)
# =============================================================================

def build_pyramids(
    experiment: str,
    wells: list[str] | None = None,
    levels: int = 5,
    factor: int = 2,
    grid_line_width: int = 1,
    store_key: str | None = None,
    skip_overlays: bool = False,
    per_position: bool = False,
    force: bool = False,
    use_v3_stores: bool = False,
    build_all_stores: bool = True,
    **kwargs,
) -> dict:
    """
    Pipeline-compatible function for building pyramids via SLURM.

    This function is called by the orchestrator and submits SLURM jobs for
    parallel pyramid building across all relevant zarr stores.
    By default uses fine-grained (position, t, c) unit parallelization for
    maximum throughput.

    Parameters
    ----------
    experiment : str
        Name of the experiment
    wells : list[str] | None
        List of well positions to process (e.g., ["A/1/0", "A/2/0"]).
        If None, processes all available wells.
    levels : int
        Number of pyramid levels to create (default: 5)
    factor : int
        Downsampling factor between levels (default: 2)
    grid_line_width : int
        Width of grid overlay lines in pixels (default: 1)
    store_key : str | None
        Specific store key to build (overrides build_all_stores).
        If None, builds all stores based on use_v3_stores flag.
    skip_overlays : bool
        If True, skip building overlays (clims, grid, ISS)
    per_position : bool
        If True, use position-level parallelization instead of unit-level
    force : bool
        If True, rebuild even if pyramids already exist
    use_v3_stores : bool
        If True, build v3 stores. If False (default), build v2 stores.
    build_all_stores : bool
        If True (default), build all stores for the selected version (v2 or v3).
        If False and store_key is provided, only build that specific store.
    **kwargs
        Additional keyword arguments (ignored for compatibility)

    Returns
    -------
    dict
        Job submission results with keys: success, job_ids, etc.
    """
    from argparse import Namespace

    # Determine which stores to build
    if store_key:
        # Specific store requested
        stores_to_build = [store_key]
        print(f"Building pyramids for specified store: {store_key}")
    elif build_all_stores:
        # Build all stores for the selected version
        if use_v3_stores:
            stores_to_build = [
                "pheno_assembled_v3",
                "lc_5x_phase_2d_stitched_v3",
                "iss_stitch_registered_v3",
            ]
            print("Building pyramids for all v3 stores")
        else:
            stores_to_build = [
                "pheno_assembled",
                "lc_5x_phase_2d_stitched",
                "iss_stitch_registered",
            ]
            print("Building pyramids for all v2 stores")
    else:
        # Default to pheno_assembled
        stores_to_build = ["pheno_assembled_v3" if use_v3_stores else "pheno_assembled"]
        print(f"Building pyramids for default store: {stores_to_build[0]}")

    # Filter to only stores that exist
    ds = OpsDataset(experiment)
    existing_stores = []
    for key in stores_to_build:
        store_path = ds.store_paths.get(key)
        if store_path and store_path.exists():
            existing_stores.append((key, store_path))
            print(f"  ✓ Will build pyramids for: {key}")
        else:
            print(f"  ⊗ Skipping {key} (does not exist)")

    if not existing_stores:
        print("⚠️  No stores found to build pyramids for")
        return {"success": False, "error": "No stores found"}

    # If only one store, use the original approach
    if len(existing_stores) == 1:
        store_key, store_path = existing_stores[0]
        print(f"\nBuilding pyramids for single store: {store_key}\n")

        args = Namespace(
            experiment=experiment,
            wells=wells,
            levels=levels,
            factor=factor,
            grid_line_width=grid_line_width,
            store=store_key,
            skip_overlays=skip_overlays,
            per_position=per_position,
            force=force,
            dry_run=False,
            no_wait=False,
            quiet=False,
        )

        result = submit_pyramid_job(
            experiment=experiment,
            slurm_params=UNIT_SLURM_PARAMS if not per_position else POSITION_SLURM_PARAMS,
            args=args,
        )

        return {
            "success": result.get("success", False),
            "stores_processed": [store_key],
            "results": {store_key: result},
        }

    # Multiple stores - gather all jobs from all stores and submit together
    print(f"\n{'='*60}")
    print(f"Gathering jobs for {len(existing_stores)} stores to submit in parallel")
    print(f"{'='*60}\n")

    from cyclops_utils.io.zarr_utils import detect_zarr_format, get_channel_dim, _iter_position_paths, ensure_pyramid_levels
    from iohub import open_ome_zarr
    from tqdm import tqdm
    import dask.array as da

    all_image_jobs = []
    all_seg_jobs = []
    all_overlay_jobs = []
    store_job_counts = {}
    resume = not force

    # Gather all jobs from all stores
    for store_key, store_path in existing_stores:
        print(f"Enumerating jobs for {store_key}...")

        zarr_format = detect_zarr_format(store_path)
        in_labels_group = (zarr_format == 3)

        # Get positions
        if wells:
            all_positions = list(_iter_position_paths(store_path))
            positions = [
                p for p in all_positions
                if any(str(p) == str(w) or str(p).startswith(str(w)) for w in wells)
            ]
        else:
            positions = list(_iter_position_paths(store_path))

        # Init UNSHARDED for v3 — per-(pos,t,c) writes race-free; reshard at end.
        from cyclops_utils.io.zarr_utils import ensure_pyramid_levels_unsharded
        print(f"  Pre-initializing pyramid levels for {len(positions)} positions...")
        for position in tqdm(positions, desc=f"  {store_key} init", leave=False):
            if zarr_format == 3:
                ensure_pyramid_levels_unsharded(store_path, position, levels, force=force, factor=factor)
            else:
                ensure_pyramid_levels(store_path, position, levels, force=force)

        # Stamp canonical native-derived YX spacing per level (pheno 0.325,
        # track 1.3) so the multiscale records correct physical size.
        _stamp_canonical_yx_scale(store_path, positions, store_key, factor=factor)

        img_count, seg_count, overlay_count = 0, 0, 0

        for position in positions:
            # Get dimensions
            with open_ome_zarr(store_path, mode="r") as store:
                fov = store[position]
                t_dim = int(fov.data.shape[0]) if fov.data.ndim >= 1 else 1
                c_dim = int(get_channel_dim(store_path, position))

            # Per-(pos, t, c) image jobs — unsharded init, parallel writes, reshard at end.
            for t in range(t_dim):
                for c in range(c_dim):
                    all_image_jobs.append({
                        "name": f"img_{store_key}_{position.replace('/', '_')}_t{t}_c{c}",
                        "func": build_pyramid_unit_worker,
                        "kwargs": {
                            "experiment": experiment, "position": position, "t": t, "c": c,
                            "source_store": str(store_path), "levels": levels,
                            "factor": factor, "resume": resume,
                        },
                        "metadata": {"experiment": experiment, "store_key": store_key, "position": position, "t": t, "c": c, "type": "image"},
                    })
                    img_count += 1

            # Segmentation jobs (seg, nuclear_seg)
            from cyclops_process.processes.pyramids.build_dask import _init_seg_levels
            for seg_name in ["seg", "nuclear_seg"]:
                seg_exists = (store_path / position / ("labels/" + seg_name if in_labels_group else seg_name) / "0").exists()
                if seg_exists:
                    seg_component = f"{position}/{'labels/' if in_labels_group else ''}{seg_name}/0"
                    try:
                        seg_arr = da.from_zarr(str(store_path), component=seg_component)
                        seg_t = int(seg_arr.shape[0]) if seg_arr.ndim >= 4 else 1
                        seg_c = int(seg_arr.shape[1]) if seg_arr.ndim >= 5 else 1

                        # Pre-init seg levels once before dispatching workers (avoids race).
                        _init_seg_levels(store_path, [position], levels, seg_name, in_labels_group=in_labels_group)
                        for t in range(seg_t):
                            for c in range(seg_c):
                                all_seg_jobs.append({
                                    "name": f"{seg_name}_{store_key}_{position.replace('/', '_')}_t{t}_c{c}",
                                    "func": build_seg_unit_worker,
                                    "kwargs": {
                                        "experiment": experiment,
                                        "position": position,
                                        "t": t,
                                        "c": c,
                                        "seg_name": seg_name,
                                        "source_store": str(store_path),
                                        "levels": levels,
                                        "in_labels_group": in_labels_group,
                                        "preserve_dtype": False,
                                        "resume": resume,
                                    },
                                    "metadata": {"experiment": experiment, "store_key": store_key, "position": position, "t": t, "c": c, "seg_name": seg_name, "type": "seg"},
                                })
                                seg_count += 1
                    except Exception as e:
                        print(f"  Warning: {seg_component}: {e}")

            # Organelle labels. Exclude RGBA overlays (grid_overlay,
            # iss_gene_image, iss_guide_image) — they're (H, W, 4) uint8
            # images, not 5D TCZYX masks; the seg pyramid worker IndexErrors
            # on them. Their pyramids are built by their own dedicated
            # `build_grid_overlay_in_place` / `build_iss_overlay_in_place`
            # pipelines, not here. Match the per-store path's skip_labels.
            labels_dir = store_path / position / "labels"
            skip_labels = {"seg", "nuclear_seg", "iss_gene_image", "iss_guide_image", "grid_overlay"}
            if labels_dir.exists():
                for subdir in labels_dir.iterdir():
                    if subdir.is_dir() and subdir.name not in skip_labels and (subdir / "0").exists():
                        try:
                            seg_arr = da.from_zarr(str(store_path), component=f"{position}/labels/{subdir.name}/0")
                            seg_t = int(seg_arr.shape[0]) if seg_arr.ndim >= 4 else 1
                            seg_c = int(seg_arr.shape[1]) if seg_arr.ndim >= 5 else 1
                            _init_seg_levels(store_path, [position], levels, subdir.name, in_labels_group=True, preserve_dtype=True)
                            for t in range(seg_t):
                                for c in range(seg_c):
                                    all_seg_jobs.append({
                                        "name": f"{subdir.name}_{store_key}_{position.replace('/', '_')}_t{t}_c{c}",
                                        "func": build_seg_unit_worker,
                                        "kwargs": {
                                            "experiment": experiment,
                                            "position": position,
                                            "t": t,
                                            "c": c,
                                            "seg_name": subdir.name,
                                            "source_store": str(store_path),
                                            "levels": levels,
                                            "in_labels_group": True,
                                            "preserve_dtype": True,
                                            "resume": resume,
                                        },
                                        "metadata": {"experiment": experiment, "store_key": store_key, "position": position, "t": t, "c": c, "seg_name": subdir.name, "type": "organelle_seg"},
                                    })
                                    seg_count += 1
                        except Exception as e:
                            print(f"  Warning: {subdir.name}: {e}")

            # Overlay jobs
            if not skip_overlays:
                all_overlay_jobs.append({
                    "name": f"overlay_{store_key}_{position.replace('/', '_')}",
                    "func": build_overlays_worker,
                    "kwargs": {
                        "experiment": experiment,
                        "position": position,
                        "source_store": str(store_path),
                        "factor": factor,
                        "grid_line_width": grid_line_width,
                    },
                    "metadata": {"experiment": experiment, "store_key": store_key, "position": position, "type": "overlay"},
                })
                overlay_count += 1

        store_job_counts[store_key] = {"image": img_count, "seg": seg_count, "overlay": overlay_count}
        print(f"  {store_key}: {img_count} image, {seg_count} seg, {overlay_count} overlay jobs")

    print(f"\nTotal: {len(all_image_jobs)} image, {len(all_seg_jobs)} seg, {len(all_overlay_jobs)} overlay")

    # Submit all jobs
    job_arrays = []
    if all_image_jobs:
        print(f"\nSubmitting {len(all_image_jobs)} image pyramid jobs...")
        img_result = submit_parallel_jobs(
            jobs_to_submit=all_image_jobs,
            experiment=f"{experiment}_all_stores_img",
            slurm_params=IMAGE_SLURM_PARAMS,
            log_dir=f"slurm_pyramid_logs/{experiment}",
            manifest_prefix="pyramid_img_all",
            dry_run=False,
            wait_for_completion=False,
            verbose=True,
            post_completion_callback=None,
            print_resource_summary=False,  # Skip per-job resource stats for large batches
        )
        if img_result.get("success"):
            job_arrays.append({"submitted_jobs": img_result["submitted_jobs"], "base_job_id": img_result["base_job_id"], "label": "image", "slurm_params": IMAGE_SLURM_PARAMS})

    if all_seg_jobs:
        print(f"\nSubmitting {len(all_seg_jobs)} segmentation jobs...")
        seg_result = submit_parallel_jobs(
            jobs_to_submit=all_seg_jobs,
            experiment=f"{experiment}_all_stores_seg",
            slurm_params=UNIT_SLURM_PARAMS,
            log_dir=f"slurm_pyramid_logs/{experiment}",
            manifest_prefix="pyramid_seg_all",
            dry_run=False,
            wait_for_completion=False,
            verbose=True,
            post_completion_callback=None,
            print_resource_summary=False,  # Skip per-job resource stats for large batches
        )
        if seg_result.get("success"):
            job_arrays.append({"submitted_jobs": seg_result["submitted_jobs"], "base_job_id": seg_result["base_job_id"], "label": "seg", "slurm_params": UNIT_SLURM_PARAMS})

    # Submit overlays alongside pyramids (overlays read base data, not pyramid levels)
    if not skip_overlays and all_overlay_jobs:
        print(f"\nSubmitting {len(all_overlay_jobs)} overlay jobs...")
        overlay_result = submit_parallel_jobs(
            jobs_to_submit=all_overlay_jobs,
            experiment=f"{experiment}_all_stores_overlay",
            slurm_params=OVERLAY_SLURM_PARAMS,
            log_dir=f"slurm_pyramid_logs/{experiment}",
            manifest_prefix="pyramid_overlay_all",
            dry_run=False,
            wait_for_completion=False,
            verbose=True,
            post_completion_callback=None,
            print_resource_summary=False,
        )
        if overlay_result.get("success"):
            job_arrays.append({"submitted_jobs": overlay_result["submitted_jobs"], "base_job_id": overlay_result["base_job_id"], "label": "overlay", "slurm_params": OVERLAY_SLURM_PARAMS})

    # Phase 1: Wait for pyramid + overlay jobs together
    total_failed = 0
    overlay_failed = 0
    if job_arrays:
        total_jobs = len(all_image_jobs) + len(all_seg_jobs) + len(all_overlay_jobs)
        print(f"\nWaiting for {total_jobs} pyramid + overlay jobs...")
        wait_results = wait_for_multiple_job_arrays(job_arrays=job_arrays, experiment=experiment, verbose=True, print_resource_summary=False)
        if wait_results.get("array_results"):
            for label, result in wait_results["array_results"].items():
                n_fail = len(result.get("failed", []))
                if label == "overlay":
                    overlay_failed += n_fail
                else:
                    total_failed += n_fail
        if total_failed > 0:
            print(f"\n⚠️  {total_failed} pyramid jobs failed")
        if overlay_failed > 0:
            print(f"\n⚠️  {overlay_failed} overlay jobs failed")

    # Reshard unsharded levels into final all-C-packed shard layout.
    reshard_failed = 0
    if total_failed == 0:
        l1_jobs, smaller_jobs = [], []
        for store_key, store_path in existing_stores:
            zarr_format = detect_zarr_format(store_path)
            if zarr_format != 3:
                continue
            if wells:
                all_positions = list(_iter_position_paths(store_path))
                positions = [p for p in all_positions
                             if any(str(p) == str(w) or str(p).startswith(str(w)) for w in wells)]
            else:
                positions = list(_iter_position_paths(store_path))
            for position in positions:
                for level in range(1, levels):
                    if (store_path / position / str(level)).exists():
                        job = {
                            "name": f"reshard_{store_key}_{position.replace('/', '_')}_L{level}",
                            "func": reshard_level_worker,
                            "kwargs": {"experiment": experiment, "position": position, "level": level, "source_store": str(store_path)},
                            "metadata": {"experiment": experiment, "store_key": store_key, "position": position, "level": level, "type": "reshard"},
                        }
                        (l1_jobs if level == 1 else smaller_jobs).append(job)
        reshard_arrays = []
        for jobs, params, suffix in ((l1_jobs, RESHARD_L1_SLURM_PARAMS, "L1"), (smaller_jobs, RESHARD_SLURM_PARAMS, "L2plus")):
            if not jobs:
                continue
            print(f"\nSubmitting {len(jobs)} reshard {suffix} jobs...")
            r = submit_parallel_jobs(
                jobs_to_submit=jobs,
                experiment=f"{experiment}_all_stores_reshard_{suffix}",
                slurm_params=params,
                log_dir=f"slurm_pyramid_logs/{experiment}",
                manifest_prefix=f"pyramid_reshard_all_{suffix}",
                dry_run=False, wait_for_completion=False, verbose=True,
                print_resource_summary=False,
            )
            if r.get("success") and r.get("submitted_jobs"):
                reshard_arrays.append({
                    "submitted_jobs": r["submitted_jobs"],
                    "base_job_id": r["base_job_id"],
                    "label": f"reshard {suffix}",
                    "slurm_params": params,
                })
        if reshard_arrays:
            wait_results = wait_for_multiple_job_arrays(job_arrays=reshard_arrays, experiment=experiment, verbose=True, print_resource_summary=False)
            reshard_failed = sum(len(a.get("failed", [])) for a in wait_results.get("array_results", {}).values())
    all_failed = total_failed + reshard_failed + overlay_failed
    print(f"\n{'='*60}")
    print(f"Summary: {', '.join([k for k, _ in existing_stores])}")
    print(f"Pyramid failed: {total_failed}, Reshard failed: {reshard_failed}, Overlay failed: {overlay_failed}")
    print(f"Status: {'SUCCESS' if all_failed == 0 else 'FAILED'}")
    print(f"{'='*60}\n")

    return {
        "success": all_failed == 0,
        "stores_processed": [k for k, _ in existing_stores],
        "store_job_counts": store_job_counts,
        "pyramid_failed": total_failed,
        "reshard_failed": reshard_failed,
        "overlay_failed": overlay_failed,
    }


def build_pyramids_setup(
    experiment: str,
    use_v3_stores: bool = True,
    store_key: str | None = None,
    **kwargs,
) -> None:
    """
    Nextflow setup step: discovers stores and positions, prints '{store_key}:{position}' per line.
    Called before the build_pyramids_position_job fan-out.
    """
    stores_to_check = (
        [store_key]
        if store_key
        else (
            ["pheno_assembled_v3", "lc_5x_phase_2d_stitched_v3", "iss_stitch_registered_v3"]
            if use_v3_stores
            else ["pheno_assembled", "lc_5x_phase_2d_stitched", "iss_stitch_registered"]
        )
    )
    for key in stores_to_check:
        try:
            _, positions, _ = get_positions_for_experiment(experiment, store_key=key)
        except ValueError:
            continue
        for pos in positions:
            # Sentinel-tagged so the Nextflow fan-out can pick these out from any other
            # stdout noise ("Using store: ..." status lines, import logs).
            print(f"PYRAMID_UNIT {key}:{pos}")


def build_pyramids_position_job(
    experiment: str,
    store_key: str,
    position: str,
    levels: int = 5,
    factor: int = 2,
    grid_line_width: int = 1,
    skip_overlays: bool = False,
    force: bool = False,
    **kwargs,
) -> None:
    """
    Nextflow per-position job: builds all pyramids, seg, and overlays for one (store, position).
    Called in parallel for each line emitted by build_pyramids_setup.
    """
    ds = OpsDataset(experiment)
    store_path = ds.store_paths.get(store_key)
    if store_path is None or not store_path.exists():
        raise ValueError(f"Store {store_key!r} not found for experiment {experiment!r}")
    result = build_position_pyramids_worker(
        experiment=experiment,
        position=position,
        source_store=str(store_path),
        levels=levels,
        factor=factor,
        grid_line_width=grid_line_width,
        skip_overlays=skip_overlays,
        resume=not force,
    )
    if result.get("status") == "failed":
        raise RuntimeError(f"Pyramid build failed for {position}: {result.get('error')}")


if __name__ == "__main__":
    main()


# ── In-process builders (folded in from the former build_pyramids.py) ──────────
# build_pyramids_local: sequential in-process build (used by the Nextflow path,
# where the executor provides per-task parallelism). build_organelle_pyramids_only:
# the registered build_organelle_pyramids step.
from typing import List, Optional
from cyclops_process.processes.pyramids.build_dask import (
    build_pyramid_in_place,
    build_clims_in_place,
    build_grid_overlay_in_place,
    build_iss_overlay_in_place,
    build_organelle_seg_pyramids,
)
from cyclops_process.napari.dask.dask_utils import _resolve_stitch_config_path

def build_pyramids_local(
    experiment: str,
    wells: Optional[List[str]] = None,
    levels: int = 5,
    factor: int = 2,
    grid_line_width: int = 1,
    **kwargs
):
    """
    Build multiscale pyramids for all stitched stores (pheno, tracking, ISS).

    Parameters
    ----------
    experiment : str
        Name of the experiment
    wells : Optional[List[str]]
        List of well positions to process (e.g., ["A/1/0", "A/2/0"]).
        If None, processes all available wells.
    levels : int
        Number of pyramid levels to create (default: 5)
    factor : int
        Downsampling factor between levels (default: 2)
    grid_line_width : int
        Width of grid overlay lines in pixels (default: 1)
    """
    ds = OpsDataset(experiment)

    # Define the stores to build pyramids for, with their corresponding modes
    store_configs = [
        ("pheno_assembled", "pheno"),
        ("lc_5x_phase_2d_stitched", "track"),
        ("iss_stitch", "iss"),
    ]

    # Filter to only stores that exist in config AND on disk
    target_stores = [
        (ds.store_paths[k], mode)
        for k, mode in store_configs
        if k in ds.store_paths and ds.store_paths[k].exists()
    ]

    if not target_stores:
        print(
            f"WARNING: No stores found for pyramid building (checked: {[k for k, _ in store_configs]})"
        )
        return

    print(
        f"Building pyramids for {len(target_stores)} store(s): {[src.name for src, _ in target_stores]}"
    )

    for src, mode in target_stores:
        print(f"\nProcessing store: {src} (mode: {mode})")

        # Get all positions in this store
        from cyclops_process.napari.dask.view_dask import _iter_position_paths

        all_positions = list(_iter_position_paths(src))

        # Filter by requested wells if specified
        if wells:
            positions_to_process = [
                p
                for p in all_positions
                if any(str(p).startswith(str(w)) for w in wells)
            ]
            print(
                f"Selected {len(positions_to_process)}/{len(all_positions)} positions"
            )
        else:
            positions_to_process = all_positions
            print(f"Processing all {len(positions_to_process)} positions")

        if not positions_to_process:
            print(f"WARNING: No positions found in {src}")
            continue

        # Build pyramid
        print(f"Building {levels} pyramid levels with factor {factor}...")
        build_pyramid_in_place(
            source_store=src,
            levels=levels,
            factor=factor,
            positions=positions_to_process,
        )

        # Build contrast limits
        print("Building contrast limits...")
        build_clims_in_place(
            source_store=src,
            positions=positions_to_process,
            scale_factor=factor,
        )

        # Build grid overlay
        print("Building grid overlay...")
        stitch_config_path = _resolve_stitch_config_path(experiment, mode)
        build_grid_overlay_in_place(
            source_store=src,
            positions=positions_to_process,
            line_width_px=grid_line_width,
            stitch_config_path=stitch_config_path,
            dataset=ds,
        )

        # Build ISS overlays (iss_gene_image + iss_guide_image)
        # Only renders if linked_results CSV exists
        print("Building ISS overlays (gene names + guide sequences)...")
        build_iss_overlay_in_place(
            source_store=src,
            experiment=experiment,
            positions=positions_to_process,
        )

        # Build organelle segmentation pyramids (only for pheno store which has labels/)
        if mode == "pheno":
            print("Building organelle segmentation pyramids...")
            build_organelle_seg_pyramids(
                source_store=src,
                levels=levels,
                positions=positions_to_process,
                resume=True,
            )

        print(f"✓ Completed pyramid build for {src.name}")

    print(f"\n✓ All pyramids built successfully")


def build_organelle_pyramids_only(
    experiment: str,
    wells: Optional[List[str]] = None,
    levels: int = 5,
    label_names: Optional[List[str]] = None,
    resume: bool = True,
):
    """
    Build organelle segmentation pyramids for the phenotyping store.

    This is useful after running organelle segmentation without rebuilding
    all image pyramids.

    Parameters
    ----------
    experiment : str
        Name of the experiment
    wells : Optional[List[str]]
        List of well positions to process (e.g., ["A/1/0", "A/2/0"]).
        If None, processes all available wells.
    levels : int
        Number of pyramid levels to create (default: 5)
    label_names : Optional[List[str]]
        Specific label names to build pyramids for. If None, auto-discovers
        all labels in the labels/ group.
    resume : bool
        If True, skip already-built levels; if False, rebuild all (default: True)

    Example
    -------
    >>> from cyclops_process.processes.pyramids.launcher import build_organelle_pyramids_only
    >>> build_organelle_pyramids_only(
    ...     experiment="ops0049_20250626",
    ...     wells=["A/3/0"],
    ...     label_names=["nucle_vs_seg", "mitoc_tomm20_seg"],
    ... )
    """
    ds = OpsDataset(experiment)

    # Get the phenotyping v3 store (where organelle labels are stored)
    # Organelle segmentation writes to phenotyping_v3.zarr, not phenotyping.zarr
    pheno_store_key = "pheno_assembled_v3"
    if pheno_store_key not in ds.store_paths or not ds.store_paths[pheno_store_key].exists():
        print(f"ERROR: Phenotyping v3 store not found at {ds.store_paths.get(pheno_store_key, 'N/A')}")
        return

    src = ds.store_paths[pheno_store_key]
    print(f"Building organelle segmentation pyramids for: {src}")

    # Get all positions in this store
    from cyclops_process.napari.dask.view_dask import _iter_position_paths

    all_positions = list(_iter_position_paths(src))

    # Filter by requested wells if specified
    if wells:
        positions_to_process = [
            p
            for p in all_positions
            if any(str(p).startswith(str(w)) for w in wells)
        ]
        print(f"Selected {len(positions_to_process)}/{len(all_positions)} positions")
    else:
        positions_to_process = all_positions
        print(f"Processing all {len(positions_to_process)} positions")

    if not positions_to_process:
        print(f"WARNING: No positions found in {src}")
        return

    # Build organelle segmentation pyramids
    build_organelle_seg_pyramids(
        source_store=src,
        levels=levels,
        positions=positions_to_process,
        resume=resume,
        label_names=label_names,
    )

    print(f"\n✓ Organelle segmentation pyramids built successfully")
