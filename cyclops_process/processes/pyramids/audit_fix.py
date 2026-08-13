#!/usr/bin/env python
"""
Unified batch builder for OPS pyramid overlays and segmentation pyramids.

Handles:
- ISS overlays (iss_gene_image + iss_guide_image - both built by --iss flag)
- Segmentation pyramids (seg, nuclear_seg - v2 zarr)
- Organelle segmentation pyramids (labels in v3 zarr labels/ group)
- Base image pyramids (main image channels)

Supports both v2 and v3 zarr stores.

Usage examples:
    # Build ISS overlays (both gene names and guide sequences)
    python -m cyclops_process.processes.pyramids.audit_fix --iss-labels --all
    python -m cyclops_process.processes.pyramids.audit_fix --iss-labels -e 103 69

    # Build seg pyramids for v2 stores
    python -m cyclops_process.processes.pyramids.audit_fix --seg-pyramids --zarr-version 2 --all

    # Build for specific experiments (supports shorthand names)
    python -m cyclops_process.processes.pyramids.audit_fix --iss-labels -e 103 69
    python -m cyclops_process.processes.pyramids.audit_fix --iss-labels --experiments ops0069

    # Build organelle segmentation pyramids (v3 zarr labels/ group)
    python -m cyclops_process.processes.pyramids.audit_fix --organelle-pyramids -e 113
    python -m cyclops_process.processes.pyramids.audit_fix --organelle-pyramids -e 33 --no-resume
    python -m cyclops_process.processes.pyramids.audit_fix --organelle-pyramids --label-filter nuclo_phase_seg -e 33 --no-resume
    python -m cyclops_process.processes.pyramids.audit_fix --organelle-pyramids --all

    # Build cell painting channel pyramids (channels 4-11 in phenotyping_v3.zarr after adding cell painting)
    python -m cyclops_process.processes.pyramids.audit_fix --cell-painting-pyramids -e 94
    python -m cyclops_process.processes.pyramids.audit_fix --cell-painting-pyramids --channel-start 4 --channel-end 12 -e 94

    # Build base image pyramids (main channels, not segmentation)
    python -m cyclops_process.processes.pyramids.audit_fix --base-image -e 115
    python -m cyclops_process.processes.pyramids.audit_fix --base-image --zarr-version 2 -e 115 --no-resume

    # Dry run - scan and report status
    python -m cyclops_process.processes.pyramids.audit_fix --iss-labels --dry-run

    # Build grid overlay
    python -m cyclops_process.processes.pyramids.audit_fix --build-grid -e 94

    # Rebuild contrast limits for all stores (pheno, iss, track)
    python -m cyclops_process.processes.pyramids.audit_fix --clims -e 103
    python -m cyclops_process.processes.pyramids.audit_fix --clims --store pheno -e 103
    python -m cyclops_process.processes.pyramids.audit_fix --clims --all

    # Audit v3 stores - check what's missing (pyramids, labels, clims, grids)
    # Prints per-store status and copy-paste fix commands
    python -m cyclops_process.processes.pyramids.audit_fix --audit -e 141
    python -m cyclops_process.processes.pyramids.audit_fix --audit -e 103 69 94

    # Audit a specific zarr path (e.g. a test copy outside the standard location).
    # --store selects which store-type the path represents (default: pheno).
    # -e is still required for the wells-config filter.
    python -m cyclops_process.processes.pyramids.audit_fix --audit -e 42 \
        --store-path /path/to/ops_data/ops0042_pro6000_iss_test/ops0042_20250520/3-assembly/phenotyping_v3.zarr

    # Fix v3 stores - audit + submit SLURM jobs for all missing components
    # Prompts before submitting, monitors all jobs together
    python -m cyclops_process.processes.pyramids.audit_fix --fix -e 141
    python -m cyclops_process.processes.pyramids.audit_fix --fix --no-wait -e 141

Note: To RUN organelle segmentation (not just build pyramids), use the pipeline runner:
    python -m cyclops_process.processes.run -e <experiment> --rerun organelle_segmentation
"""
import argparse
import itertools
from pathlib import Path
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.getcwd())

from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.io.zarr_utils import (
    _iter_position_paths,
    _discover_last_position_per_well,
    detect_zarr_format,
    level_has_data,
)
from cyclops_utils.hpc.slurm_batch_utils import (
    detect_experiments_needing_processing,
    submit_parallel_jobs,
)
from cyclops_utils.data.filesystem import resolve_experiment_name

from cyclops_process.processes.pyramids.build_drivers import (
    _cleanup_top_level_seg_symlinks_for_experiment,
    _filter_to_configured_wells,
    _get_store_path,
    _build_clims,
    build_iss_overlay,
    _build_grid_overlay,
    _build_seg_pyramids,
    _discover_organelle_labels,
    _build_organelle_pyramids,
    _build_base_image_pyramids,
    _run_single_experiment_build,
    detect_experiments_for_build,
)

from cyclops_process.processes.pyramids.store_audit import (
    PHENO_NATIVE_YX,
    PHENO_BUGGY_YX,
    TRACK_NATIVE_YX,
    PYRAMID_DOWNSAMPLE,
    audit_v3_stores,
    _prompt_iss_rebuild,
)
from cyclops_process.paths import BASE_PATH













def _fix_yx_scale_one(
    store_path: Path,
    native_yx: float = PHENO_NATIVE_YX,
    buggy_yx: float = PHENO_BUGGY_YX,
) -> dict:
    """Rewrite per-position `zarr.json` files under `store_path` so every
    multiscale level's Y/X spacing follows the canonical pyramid
    `native_yx * PYRAMID_DOWNSAMPLE**level`. Idempotent — skips positions whose
    level-0 Y/X already equals `native_yx`. Sibling `.bak` saved per file before
    any write.

    Writing canonical absolute values (rather than halving in place) repairs
    track's malformed pyramids, where level-0 was halved to 0.65 but the coarsest
    level was left untouched — a blind /2 would corrupt the good levels.

    Returns {"fixed": int, "already": int, "skipped": int, "errors": [...]}.
    """
    import json
    out = {"fixed": 0, "already": 0, "skipped": 0, "errors": []}
    if not store_path or not store_path.exists():
        out["errors"].append(f"store missing: {store_path}")
        return out
    for pj in sorted(store_path.glob("*/*/*/zarr.json")):
        try:
            original = pj.read_text()
            d = json.loads(original)
            attrs = d.get("attributes", {})
            ome = attrs.get("ome") or attrs
            ms = ome.get("multiscales")
            if not ms:
                out["skipped"] += 1
                continue
            l0 = ms[0]["datasets"][0]["coordinateTransformations"][0]["scale"]
            if l0[-1] == native_yx and l0[-2] == native_yx:
                out["already"] += 1
                continue
            if l0[-1] != buggy_yx or l0[-2] != buggy_yx:
                out["errors"].append(f"{pj}: unexpected level-0 YX {(l0[-2], l0[-1])}")
                out["skipped"] += 1
                continue
            for i, ds in enumerate(ms[0]["datasets"]):
                s = ds["coordinateTransformations"][0]["scale"]
                level_yx = native_yx * (PYRAMID_DOWNSAMPLE ** i)
                s[-1] = level_yx
                s[-2] = level_yx
            bak = pj.with_name(pj.name + ".bak")
            if not bak.exists():
                bak.write_text(original)
            tmp = pj.with_name(pj.name + ".tmp")
            tmp.write_text(json.dumps(d, indent=2))
            tmp.replace(pj)
            out["fixed"] += 1
        except Exception as e:
            out["errors"].append(f"{pj}: {e}")
    return out






def _fix_normalization_one(store_path: Path) -> dict:
    """Mirror each position's top-level `normalization` into
    `custom_metadata.normalization` in its `zarr.json`, matching the v3 schema
    convert_v3.py produced. Idempotent — skips positions already mirrored, and
    writes a sibling `.bak` before any edit.

    Positions whose top-level `normalization` is absent cannot be mirrored and
    are reported under `no_source` (re-run the viscy_normalize step for those).

    Returns {"fixed": int, "already": int, "no_source": int, "errors": [...]}.
    """
    import json
    out = {"fixed": 0, "already": 0, "no_source": 0, "errors": []}
    if not store_path or not store_path.exists():
        out["errors"].append(f"store missing: {store_path}")
        return out
    for pj in sorted(store_path.glob("*/*/*/zarr.json")):
        try:
            original = pj.read_text()
            d = json.loads(original)
            attrs = d.setdefault("attributes", {})
            cm = attrs.get("custom_metadata", {})
            if cm.get("normalization"):
                out["already"] += 1
                continue
            norm = attrs.get("normalization")
            if not norm:
                out["no_source"] += 1
                continue
            cm = dict(cm)
            cm["normalization"] = norm
            attrs["custom_metadata"] = cm
            bak = pj.with_name(pj.name + ".bak")
            if not bak.exists():
                bak.write_text(original)
            tmp = pj.with_name(pj.name + ".tmp")
            tmp.write_text(json.dumps(d, indent=2))
            tmp.replace(pj)
            out["fixed"] += 1
        except Exception as e:
            out["errors"].append(f"{pj}: {e}")
    return out












def fix_v3_stores(
    experiment: str,
    no_wait: bool = False,
    quiet: bool = False,
    auto_fix: bool = False,
) -> dict:
    """Run audit and submit SLURM jobs to fix all missing components.

    Submits independent fix jobs in parallel (one per build type per store),
    then monitors them all together.

    Args:
        experiment: Experiment name or shorthand
        no_wait: If True, submit jobs and return without waiting
        quiet: Reduce output verbosity

    Returns:
        Dict with job submission results
    """
    from cyclops_utils.hpc.slurm_batch_utils import (
        submit_parallel_jobs,
        wait_for_multiple_job_arrays,
    )

    # Run audit first
    audit = audit_v3_stores(experiment, verbose=True)

    # Add fix commands for failed reshards (same path as missing levels)
    for store_type, store_result in audit.items():
        if not isinstance(store_result, dict):
            continue
        if store_result.get("failed_reshards"):
            cmd = (
                f"uv run python -m cyclops_process.processes.pyramids.audit_fix "
                f"--base-image --store {store_type} -e {experiment}"
            )
            fix_cmds = audit.get("fix_commands", [])
            if cmd not in fix_cmds:
                fix_cmds.append(cmd)
                audit["fix_commands"] = fix_cmds

    fix_cmds = audit.get("fix_commands", [])
    if not fix_cmds:
        print("Nothing to fix!")
        _prompt_iss_rebuild(experiment)
        return {"success": True, "jobs_submitted": 0}

    # Run symlink commands inline first (instant filesystem ops, not SLURM jobs).
    # Must complete before convert_v3 jobs that depend on the symlinked data.
    symlink_cmds = [c for c in fix_cmds if "batch_symlink_nuclear_seg" in c]
    if symlink_cmds:
        from cyclops_process.utils.batch.batch_symlink_nuclear_seg import (
            symlink_nuclear_seg,
        )
        for cmd in symlink_cmds:
            parts = cmd.split()
            target = "iss"
            zver = 3  # default to v3-native dest
            if "--symlink-target" in parts:
                idx = parts.index("--symlink-target")
                if idx + 1 < len(parts):
                    target = parts[idx + 1]
            if "--zarr-version" in parts:
                idx = parts.index("--zarr-version")
                if idx + 1 < len(parts):
                    zver = int(parts[idx + 1])
            print(f"\n  Running symlink inline: {experiment} -> {target} (v{zver})")
            symlink_nuclear_seg(experiment, symlink_target=target, zarr_version=zver)

    # Run fix steps inline before SLURM jobs
    import subprocess as _sp
    import sys as _sys

    def _rewrite_uv_to_venv(cmd: str) -> list[str]:
        """Replace the ``uv run python`` prefix with the current interpreter.

        The printed fix commands use ``uv run python`` so an operator can copy
        them into a fresh shell. When we auto-execute them via subprocess,
        ``uv run`` spawns a clean subprocess that does NOT inherit the venv's
        site-packages — so things like submitit go missing and the inner SLURM
        submit fails. Use ``sys.executable`` (the current venv's python) so
        the child sees the same packages this process has.
        """
        parts = cmd.split()
        # Strip a leading "uv run python" → ["uv","run","python"]
        if len(parts) >= 3 and parts[0] == "uv" and parts[1] == "run" and parts[2] == "python":
            return [_sys.executable] + parts[3:]
        # Fallback: leave alone
        return parts

    # YX-scale fixes are instant per-position metadata edits, not SLURM jobs —
    # run them inline (pheno native 0.325, track native 1.3).
    yx_scale_cmds = [c for c in fix_cmds if "--fix-yx-scale" in c]
    for cmd in yx_scale_cmds:
        parts = cmd.split()
        store_type = "pheno"
        if "--store" in parts:
            idx = parts.index("--store")
            if idx + 1 < len(parts):
                store_type = parts[idx + 1]
        native_yx = PHENO_NATIVE_YX if store_type == "pheno" else TRACK_NATIVE_YX
        store_path = _get_store_path(
            OpsDataset(experiment), zarr_version=3, store_type=store_type
        )
        print(f"\n  Fixing YX scale inline: {experiment} ({store_type}, native={native_yx})")
        r = _fix_yx_scale_one(store_path, native_yx=native_yx)
        print(f"    fixed={r['fixed']} already={r['already']} "
              f"skipped={r['skipped']} errors={len(r['errors'])}")
        for e in r["errors"][:3]:
            print(f"      ✗ {e}")

    # Re-run upscale_nuclear_segmentations via SLURM if needed
    upscale_cmds = [c for c in fix_cmds if "upscale_nuclear_segmentations" in c]
    if upscale_cmds:
        from cyclops_process.processes.segment import upscale_nuclear_segmentations
        print(f"\n  Re-running upscale_nuclear_segmentations via SLURM...")
        jobs = [{
            "name": f"upscale_nuclear_seg_{experiment}",
            "func": upscale_nuclear_segmentations,
            "kwargs": {"experiment": experiment, "overwrite": True},
        }]
        result = submit_parallel_jobs(
            jobs,
            experiment=f"{experiment}_upscale_seg",
            slurm_params={"timeout_min": 30, "mem": "128G", "cpus_per_task": 16, "slurm_partition": "cpu,gpu"},
            wait_for_completion=True,
        )
        if not result.get("success"):
            print(f"  WARNING: upscale_nuclear_segmentations failed")

    # Run v2 pyramid builds via SLURM before convert_v3 (so v2 has all levels to convert)
    v2_pyramid_cmds = [c for c in fix_cmds if "batch_build_pyramids" in c and "--zarr-version 2" in c]
    for cmd in v2_pyramid_cmds:
        slurm_cmd = cmd + " --slurm -y"
        print(f"\n  Running v2 pyramid build via SLURM: {slurm_cmd}")
        _sp.run(_rewrite_uv_to_venv(slurm_cmd), check=True)

    # Run convert_v3 inline — waits for inner SLURM jobs to complete
    convert_cmds = [c for c in fix_cmds if "convert_v3_slurm" in c]
    pyramid_cmds_from_convert = []
    for cmd in convert_cmds:
        print(f"\n  Running convert_v3 inline: {cmd}")
        _sp.run(_rewrite_uv_to_venv(cmd), check=True)
        # Extract --only-labels and --mode to queue pyramid build
        parts = cmd.split()
        _mode = "pheno"
        _label = None
        if "--mode" in parts:
            idx = parts.index("--mode")
            if idx + 1 < len(parts):
                _mode = parts[idx + 1]
        if "--only-labels" in parts:
            idx = parts.index("--only-labels")
            if idx + 1 < len(parts):
                _label = parts[idx + 1]
        if _label:
            # Only queue if fix_cmds doesn't already have a pyramid build for this label+store
            already_has = any(
                f"--seg-types {_label}" in c and f"--store {_mode}" in c
                for c in fix_cmds
            )
            if not already_has:
                pyramid_cmd = (
                    f"uv run python -m cyclops_process.processes.pyramids.audit_fix "
                    f"--seg-pyramids --seg-types {_label} "
                    f"--store {_mode} --no-resume -e {experiment}"
                )
                print(f"  Queuing pyramid build for converted label: {_label} ({_mode})")
                pyramid_cmds_from_convert.append(pyramid_cmd)

    # Map remaining fix commands + any pyramid builds needed after conversion
    # Exclude commands already run inline (symlinks, convert_v3)
    already_ran = set(symlink_cmds) | set(convert_cmds) | set(upscale_cmds) | set(v2_pyramid_cmds) | set(yx_scale_cmds)
    remaining_cmds = [c for c in fix_cmds if c not in already_ran] + pyramid_cmds_from_convert
    jobs_to_submit = []
    for cmd in remaining_cmds:
        job_specs = _fix_command_to_job_specs(experiment, cmd)
        jobs_to_submit.extend(job_specs)

    if not jobs_to_submit:
        if symlink_cmds or convert_cmds or pyramid_cmds_from_convert:
            print("All fixes completed inline. No additional SLURM jobs needed.")
            _cleanup_top_level_seg_symlinks_for_experiment(experiment)
            print(f"\n{'=' * 60}")
            print(f"  POST-FIX VERIFICATION AUDIT")
            print(f"{'=' * 60}\n")
            verify = audit_v3_stores(experiment, verbose=True)
            remaining = verify.get("fix_commands", [])
            if remaining:
                print(f"\n  ⚠ {len(remaining)} issues remain after fix")
            else:
                print(f"\n  ✓ All stores verified OK")
            # Prompt user to rebuild ISS overlays if link_calls_tracks was rerun
            _prompt_iss_rebuild(experiment)
            return {"success": not remaining, "jobs_submitted": 0}
        print("Could not create job specs from fix commands")
        return {"success": False, "jobs_submitted": 0}

    # Prompt user before launching
    print(f"\n{'=' * 60}")
    print(f"  Ready to submit {len(jobs_to_submit)} fix jobs:")
    print(f"{'=' * 60}")
    for j in jobs_to_submit:
        print(f"  - {j['name']}")
    print()

    if not auto_fix:
        try:
            resp = input(f"Submit {len(jobs_to_submit)} SLURM jobs? [y/N]: ").strip().lower()
        except EOFError:
            resp = "y"  # non-interactive: proceed
        if resp not in ("y", "yes"):
            print("Aborted.")
            return {"success": False, "jobs_submitted": 0, "aborted": True}

    # Group jobs by slurm params for efficient array submission.
    # clims jobs must run AFTER pyramid-building jobs (base_image, seg, grid,
    # iss, organelle) because compute_position_clims samples the coarsest
    # pyramid level — if levels 1-4 are still being built when clims fires,
    # the sample comes from empty data and the fluorescence profile falls
    # back to a default range (e.g. mCherry → [0, 2.356]). Split into two
    # waves to enforce that ordering.
    from collections import defaultdict

    CLIMS_LABEL = "clims"
    pyramid_jobs = [j for j in jobs_to_submit if j.get("label") != CLIMS_LABEL]
    clims_jobs = [j for j in jobs_to_submit if j.get("label") == CLIMS_LABEL]

    def _submit_wave(wave_jobs: list, wave_name: str) -> tuple[list, int]:
        """Submit a wave of jobs grouped by slurm_params; return (arrays, n_submitted)."""
        wave_arrays = []
        wave_total = 0
        param_groups_local = defaultdict(list)
        for job in wave_jobs:
            param_key = tuple(sorted(job["slurm_params"].items()))
            param_groups_local[param_key].append(job)
        for param_key, group_jobs in param_groups_local.items():
            slurm_params = dict(param_key)
            label = group_jobs[0].get("label", "fix")
            result = submit_parallel_jobs(
                jobs_to_submit=group_jobs,
                experiment=f"{experiment}_fix_{label}",
                slurm_params=slurm_params,
                log_dir=f"slurm_fix_logs/{experiment}",
                manifest_prefix=f"fix_{label}",
                dry_run=False,
                wait_for_completion=False,
                verbose=not quiet,
            )
            if result.get("success"):
                wave_total += len(group_jobs)
                wave_arrays.append({
                    "submitted_jobs": result["submitted_jobs"],
                    "base_job_id": result["base_job_id"],
                    "label": label,
                    "slurm_params": slurm_params,
                })
        return wave_arrays, wave_total

    # Wave 1: pyramid builders (base_image, seg, grid, iss, organelle)
    job_arrays_to_monitor, total_submitted = _submit_wave(pyramid_jobs, "pyramid")

    # Wait for wave 1 before launching clims so it samples a complete pyramid.
    if not no_wait and clims_jobs and job_arrays_to_monitor:
        print(f"\n{'=' * 60}")
        print(f"Waiting for {total_submitted} pyramid-build fix jobs before launching clims...")
        print(f"{'=' * 60}\n")
        wait_for_multiple_job_arrays(
            job_arrays=job_arrays_to_monitor,
            experiment=experiment,
            verbose=not quiet,
        )

    # Wave 2: clims (depends on a complete pyramid)
    clims_arrays, clims_submitted = _submit_wave(clims_jobs, "clims")
    job_arrays_to_monitor.extend(clims_arrays)
    total_submitted += clims_submitted

    if not no_wait and job_arrays_to_monitor:
        print(f"\n{'=' * 60}")
        print(f"Waiting for {total_submitted} fix jobs to complete...")
        print(f"{'=' * 60}\n")

        wait_results = wait_for_multiple_job_arrays(
            job_arrays=job_arrays_to_monitor,
            experiment=experiment,
            verbose=not quiet,
        )

        total_failed = 0
        failed_jobs_for_retry = []
        if wait_results.get("array_results"):
            for array_label, array_result in wait_results["array_results"].items():
                failed_names = set(array_result.get("failed", []))
                total_failed += len(failed_names)
                if failed_names:
                    # Collect original job specs for retry
                    for job_spec in jobs_to_submit:
                        if job_spec["name"] in failed_names:
                            failed_jobs_for_retry.append(job_spec)

        # Auto-retry failed jobs once with 2x timeout
        if failed_jobs_for_retry:
            print(f"\n  {total_failed} fix jobs failed — retrying {len(failed_jobs_for_retry)} with 2x timeout...")
            retry_arrays = []
            for label, group_jobs in itertools.groupby(
                sorted(failed_jobs_for_retry, key=lambda j: j.get("label", "fix")),
                key=lambda j: j.get("label", "fix"),
            ):
                group_jobs = list(group_jobs)
                slurm_params = dict(group_jobs[0].get("slurm_params", {"timeout_min": 15, "mem": "64GB", "cpus_per_task": 8, "slurm_partition": "cpu,gpu"}))
                slurm_params["timeout_min"] = slurm_params.get("timeout_min", 15) * 2
                result = submit_parallel_jobs(
                    jobs_to_submit=group_jobs,
                    experiment=f"{experiment}_retry_{label}",
                    slurm_params=slurm_params,
                    log_dir=f"slurm_fix_logs/{experiment}",
                    manifest_prefix=f"retry_{label}",
                    dry_run=False,
                    wait_for_completion=False,
                    verbose=not quiet,
                )
                if result.get("success"):
                    retry_arrays.append({
                        "submitted_jobs": result["submitted_jobs"],
                        "base_job_id": result["base_job_id"],
                        "label": f"retry_{label}",
                        "slurm_params": slurm_params,
                    })

            if retry_arrays:
                retry_results = wait_for_multiple_job_arrays(
                    job_arrays=retry_arrays, experiment=experiment, verbose=not quiet,
                )
                total_failed = 0
                if retry_results.get("array_results"):
                    for array_label, array_result in retry_results["array_results"].items():
                        total_failed += len(array_result.get("failed", []))
                if total_failed > 0:
                    print(f"\n  {total_failed} jobs still failed after retry")
                else:
                    print(f"\n  All retried jobs succeeded!")

        if total_failed > 0:
            print(f"\n  {total_failed} fix jobs failed")

        # Phase 2: Reshard pyramid levels if any base_image unit jobs succeeded (v3 zarr)
        reshard_failed = 0
        if total_failed == 0:
            reshard_failed = _run_post_fix_resharding(jobs_to_submit, experiment, quiet)

        # Re-audit to verify fixes
        if total_failed == 0 and reshard_failed == 0:
            _cleanup_top_level_seg_symlinks_for_experiment(experiment)
            print(f"\n{'=' * 60}")
            print(f"  POST-FIX VERIFICATION AUDIT")
            print(f"{'=' * 60}\n")
            verify = audit_v3_stores(experiment, verbose=True)
            remaining = verify.get("fix_commands", [])
            if remaining:
                print(f"\n  ⚠ {len(remaining)} issues remain after fix")
            else:
                print(f"\n  ✓ All stores verified OK")

            # Prompt user to rebuild ISS overlays if link_calls_tracks was rerun
            _prompt_iss_rebuild(experiment)

        return {
            "success": total_failed == 0 and reshard_failed == 0,
            "jobs_submitted": total_submitted,
            "jobs_failed": total_failed,
            "reshard_failed": reshard_failed,
        }

    return {"success": True, "jobs_submitted": total_submitted}


def _run_post_fix_resharding(jobs_to_submit: list, experiment: str, quiet: bool = False) -> int:
    """Submit resharding jobs for base_image fix units that need it (v3 zarr).

    Called after all pyramid unit jobs complete successfully. Discovers which
    positions/stores need resharding from job metadata, deduplicates, and
    submits reshard jobs via SLURM.

    Returns number of failed reshard jobs (0 = success).
    """
    from cyclops_process.processes.pyramids.workers import reshard_level_worker

    # Collect unique (store_path, position, num_levels) from completed jobs
    reshard_targets = {}
    for job in jobs_to_submit:
        meta = job.get("metadata", {})
        if not meta.get("needs_reshard"):
            continue
        store_path = meta["store_path"]
        pos = meta["position"]
        num_levels = meta["num_levels"]
        key = (store_path, pos)
        if key not in reshard_targets:
            reshard_targets[key] = num_levels

    if not reshard_targets:
        return 0

    reshard_slurm = {
        "timeout_min": 30,
        "mem": "300GB",
        "cpus_per_task": 8,
        "slurm_partition": "cpu,gpu",
    }

    reshard_jobs = []
    for (store_path, pos), num_levels in reshard_targets.items():
        pos_label = pos.replace("/", "_")
        for level in range(1, num_levels):
            reshard_jobs.append({
                "name": f"reshard_{pos_label}_L{level}_{experiment}",
                "func": reshard_level_worker,
                "kwargs": {
                    "experiment": experiment,
                    "position": pos,
                    "level": level,
                    "source_store": store_path,
                },
                "metadata": {"type": "reshard", "position": pos, "level": level},
            })

    print(f"\n{'='*60}")
    print(f"Submitting {len(reshard_jobs)} resharding jobs...")
    print(f"{'='*60}\n")

    result = submit_parallel_jobs(
        jobs_to_submit=reshard_jobs,
        experiment=f"{experiment}_fix_reshard",
        slurm_params=reshard_slurm,
        log_dir=f"slurm_fix_logs/{experiment}",
        manifest_prefix="fix_reshard",
        dry_run=False,
        wait_for_completion=True,
        verbose=not quiet,
    )

    failed = len(result.get("failed", []))
    if failed > 0:
        print(f"\n  {failed} resharding jobs failed")
    return failed


def _fix_command_to_job_specs(experiment: str, cmd: str) -> list[dict]:
    """Convert a fix CLI command string into one or more SLURM job specs.

    Seg pyramid jobs are split per-well for parallelism.
    Returns a list of job spec dicts (usually 1, but N for per-well seg).

    All fix jobs run with resume=False (equivalent to --no-resume) so that
    broken/incomplete levels are rebuilt instead of skipped.
    """
    # Default SLURM params for fix jobs
    default_params = {
        "timeout_min": 30,
        "mem": "128GB",
        "cpus_per_task": 32,
        "slurm_partition": "cpu,gpu",
    }
    # Base image pheno pyramids load full-res images (~41 GB float32) and need
    # extra headroom for the downsample pipeline.
    base_image_params = {
        "timeout_min": 60,
        "mem": "256GB",
        "cpus_per_task": 32,
        "slurm_partition": "cpu,gpu",
    }
    clims_params = {
        "timeout_min": 15,
        "mem": "64GB",
        "cpus_per_task": 16,
        "slurm_partition": "cpu,gpu",
    }
    grid_params = {
        "timeout_min": 20,
        "mem": "250GB",
        "cpus_per_task": 32,
        "slurm_partition": "cpu,gpu",
    }

    def _extract_flag(parts, flag, default):
        if flag in parts:
            idx = parts.index(flag)
            if idx + 1 < len(parts):
                return parts[idx + 1]
        return default

    parts = cmd.split()

    if "--clims" in cmd:
        store = _extract_flag(parts, "--store", "all")
        return [{
            "name": f"fix_clims_{store}_{experiment}",
            "func": _build_clims,
            "kwargs": {
                "experiment": experiment,
                "zarr_version": 3,
                "store": store,
            },
            "metadata": {"type": "clims", "store": store},
            "slurm_params": clims_params,
            "label": "clims",
        }]

    elif "--seg-pyramids" in cmd:
        store = _extract_flag(parts, "--store", "pheno")
        seg_type = _extract_flag(parts, "--seg-types", "nuclear_seg")

        # Discover positions to split per-well
        dataset = OpsDataset(experiment)
        store_path = _get_store_path(dataset, zarr_version=3, store_type=store)
        positions = []
        if store_path and store_path.exists():
            positions = _iter_position_paths(store_path)
            if not positions:
                try:
                    store_p = Path(store_path)
                    for row_dir in sorted(store_p.iterdir()):
                        if row_dir.is_dir() and row_dir.name.isalpha():
                            for col_dir in sorted(row_dir.iterdir()):
                                if col_dir.is_dir() and col_dir.name.isdigit():
                                    for fov_dir in sorted(col_dir.iterdir()):
                                        if fov_dir.is_dir() and fov_dir.name.isdigit():
                                            positions.append(
                                                f"{row_dir.name}/{col_dir.name}/{fov_dir.name}"
                                            )
                except Exception:
                    pass

        if not positions:
            # Fallback: single job for entire store
            return [{
                "name": f"fix_seg_{seg_type}_{store}_{experiment}",
                "func": _build_seg_pyramids,
                "kwargs": {
                    "experiment": experiment,
                    "zarr_version": 3,
                    "seg_types": [seg_type],
                    "store": store,
                    "resume": False,
                },
                "metadata": {"type": "seg_pyramids", "store": store, "seg_type": seg_type},
                "slurm_params": default_params,
                "label": "seg",
            }]

        # One job per well
        jobs = []
        for pos in positions:
            well_label = pos.replace("/", "_")
            jobs.append({
                "name": f"fix_seg_{seg_type}_{store}_{well_label}_{experiment}",
                "func": _build_seg_pyramids,
                "kwargs": {
                    "experiment": experiment,
                    "zarr_version": 3,
                    "seg_types": [seg_type],
                    "store": store,
                    "positions": [pos],
                    "resume": False,
                },
                "metadata": {"type": "seg_pyramids", "store": store, "seg_type": seg_type, "position": pos},
                "slurm_params": default_params,
                "label": "seg",
            })
        return jobs

    elif "--build-grid" in cmd:
        store = _extract_flag(parts, "--store", "pheno")
        return [{
            "name": f"fix_grid_{store}_{experiment}",
            "func": _build_grid_overlay,
            "kwargs": {
                "experiment": experiment,
                "zarr_version": 3,
                "store_type": store,
                "force": True,
            },
            "metadata": {"type": "grid", "store": store},
            "slurm_params": grid_params,
            "label": "grid",
        }]

    elif "--iss" in cmd:
        # gene and guide overlays are independent — fire them as TWO parallel
        # SLURM jobs to halve wall-clock time.
        _iss_slurm = {
            "timeout_min": 25,
            "mem": "350GB",
            "cpus_per_task": 32,
            "slurm_partition": "cpu,gpu",
        }
        return [
            {
                "name": f"fix_iss_gene_overlay_{experiment}",
                "func": build_iss_overlay,
                "kwargs": {
                    "experiment": experiment,
                    "zarr_version": 3,
                    "force": True,
                    "kinds": ("gene",),
                },
                "metadata": {"type": "iss_overlay", "kind": "gene"},
                "slurm_params": _iss_slurm,
                "label": "iss_gene",
            },
            {
                "name": f"fix_iss_guide_overlay_{experiment}",
                "func": build_iss_overlay,
                "kwargs": {
                    "experiment": experiment,
                    "zarr_version": 3,
                    "force": True,
                    "kinds": ("guide",),
                },
                "metadata": {"type": "iss_overlay", "kind": "guide"},
                "slurm_params": _iss_slurm,
                "label": "iss_guide",
            },
        ]

    elif "--base-image" in cmd:
        store = _extract_flag(parts, "--store", "pheno")

        # Discover positions
        dataset = OpsDataset(experiment)
        store_path = _get_store_path(dataset, zarr_version=3, store_type=store)
        positions = []
        if store_path and store_path.exists():
            positions = _iter_position_paths(store_path)
            if not positions:
                try:
                    store_p = Path(store_path)
                    for row_dir in sorted(store_p.iterdir()):
                        if row_dir.is_dir() and row_dir.name.isalpha():
                            for col_dir in sorted(row_dir.iterdir()):
                                if col_dir.is_dir() and col_dir.name.isdigit():
                                    for fov_dir in sorted(col_dir.iterdir()):
                                        if fov_dir.is_dir() and fov_dir.name.isdigit():
                                            positions.append(
                                                f"{row_dir.name}/{col_dir.name}/{fov_dir.name}"
                                            )
                except Exception:
                    pass

        if not positions:
            return [{
                "name": f"fix_base_image_{store}_{experiment}",
                "func": _build_base_image_pyramids,
                "kwargs": {
                    "experiment": experiment,
                    "zarr_version": 3,
                    "store": store,
                    "resume": False,
                },
                "metadata": {"type": "base_image", "store": store},
                "slurm_params": base_image_params,
                "label": "base_image",
            }]

        # Per-(pos, t, c) unit jobs for max parallelism
        return _generate_base_image_unit_jobs(experiment, store=store)

    elif "--organelle-pyramids" in cmd:
        label_filter = None
        if "--label-filter" in parts:
            idx = parts.index("--label-filter")
            if idx + 1 < len(parts):
                label_filter = [parts[idx + 1]]

        return [{
            "name": f"fix_organelle_{experiment}",
            "func": _build_organelle_pyramids,
            "kwargs": {
                "experiment": experiment,
                "label_filter": label_filter,
                "resume": False,
            },
            "metadata": {"type": "organelle"},
            "slurm_params": default_params,
            "label": "organelle",
        }]

    elif "batch_symlink_nuclear_seg" in cmd:
        # Symlinks are run inline in fix_v3_stores before SLURM submission,
        # not as SLURM jobs (they're instant filesystem ops).
        return []

    elif "convert_v3_slurm" in cmd:
        # convert_v3_slurm is itself a SLURM submission script (not a worker),
        # so it's run inline in fix_v3_stores, not wrapped in another SLURM job.
        return []

    return []


# =============================================================================
# Scanning and reporting
# =============================================================================


def scan_experiments(
    experiment_configs_dir: Path,
    zarr_version: int,
    check_iss: bool = False,
    check_seg_pyramids: bool = False,
) -> dict:
    """Scan experiments and report status."""
    results = {
        "has_store": [],
        "no_store": [],
        "iss_done": [],
        "iss_needed": [],
        "seg_done": [],
        "seg_needed": [],
    }

    config_files = sorted(experiment_configs_dir.glob("ops*_config.yaml"))
    print(f"Found {len(config_files)} experiment configs\n")

    for config_file in tqdm(config_files, desc="Scanning"):
        experiment = config_file.stem.replace("_config", "")

        try:
            dataset = OpsDataset(experiment)
            store_path = _get_store_path(dataset, zarr_version, "pheno")

            if not store_path or not store_path.exists():
                results["no_store"].append(experiment)
                continue

            results["has_store"].append(experiment)

            # Check ISS
            if check_iss:
                positions = _iter_position_paths(store_path)
                if positions:
                    iss_path = store_path / positions[0] / "iss_points"
                    if iss_path.exists():
                        results["iss_done"].append(experiment)
                    else:
                        results["iss_needed"].append(experiment)

            # Check seg pyramids
            if check_seg_pyramids:
                positions = _iter_position_paths(store_path)
                if positions:
                    seg_path = store_path / positions[0] / "0" / "seg" / "1"
                    if seg_path.exists():
                        results["seg_done"].append(experiment)
                    else:
                        results["seg_needed"].append(experiment)

        except Exception as e:
            print(f"  Error checking {experiment}: {e}")
            continue

    return results


def print_scan_results(results: dict, zarr_version: int):
    """Pretty print scan results."""
    print(f"\n{'='*80}")
    print(f"PYRAMID BUILD SCAN RESULTS (v{zarr_version} zarr)")
    print(f"{'='*80}\n")

    print(f"Store status:")
    print(f"  Has v{zarr_version} store: {len(results['has_store'])} experiments")
    print(f"  No v{zarr_version} store: {len(results['no_store'])} experiments")
    print()

    if results["iss_needed"] or results["iss_done"]:
        print(f"ISS overlays:")
        print(f"  Done: {len(results['iss_done'])}")
        print(f"  Needed: {len(results['iss_needed'])}")
        if results["iss_needed"][:5]:
            for exp in results["iss_needed"][:5]:
                print(f"    - {exp}")
            if len(results["iss_needed"]) > 5:
                print(f"    ... and {len(results['iss_needed']) - 5} more")
        print()

    if results["seg_needed"] or results["seg_done"]:
        print(f"Seg pyramids:")
        print(f"  Done: {len(results['seg_done'])}")
        print(f"  Needed: {len(results['seg_needed'])}")
        if results["seg_needed"][:5]:
            for exp in results["seg_needed"][:5]:
                print(f"    - {exp}")
            if len(results["seg_needed"]) > 5:
                print(f"    ... and {len(results['seg_needed']) - 5} more")
        print()

    print(f"{'='*80}\n")


# =============================================================================
# Main CLI
# =============================================================================


def _generate_base_image_unit_jobs(experiment: str, store: str = "pheno", num_levels: int = 5):
    """Generate per-(pos, t, c) unit jobs + reshard metadata for base image pyramids.

    Shared by both ``--fix --base-image`` and ``--base-image --slurm`` paths.
    Pre-initialises unsharded pyramid levels so parallel workers don't contend.

    Returns list of job dicts ready for ``submit_parallel_jobs``.
    """
    from cyclops_process.processes.pyramids.workers import build_pyramid_unit_worker
    from cyclops_utils.io.zarr_utils import (
        enumerate_units,
        detect_zarr_format,
        ensure_pyramid_levels_unsharded,
        ensure_pyramid_levels,
    )

    dataset = OpsDataset(experiment)
    store_path = _get_store_path(dataset, zarr_version=3, store_type=store)

    if not store_path or not store_path.exists():
        print(f"  {experiment}: store not found for {store}")
        return []

    positions = _iter_position_paths(store_path)
    if not positions:
        print(f"  {experiment}: no positions found")
        return []

    zarr_format = detect_zarr_format(store_path)

    print(f"  Pre-initializing pyramid levels for {len(positions)} positions...")
    for pos in positions:
        if zarr_format == 3:
            ensure_pyramid_levels_unsharded(store_path, pos, num_levels, force=True, factor=2)
        else:
            ensure_pyramid_levels(store_path, pos, num_levels, force=True)

    units = enumerate_units(store_path, positions)
    print(f"  Discovered {len(units)} (pos, t, c) units across {len(positions)} positions")

    # SLURM params: pheno = large, iss/track = small
    if store in ("iss", "track"):
        unit_slurm = {
            "timeout_min": 10,
            "mem": "32GB",
            "cpus_per_task": 8,
            "slurm_partition": "cpu,gpu",
        }
    else:
        unit_slurm = {
            "timeout_min": 35,
            "mem": "250GB",
            "cpus_per_task": 32,
            "slurm_partition": "cpu,gpu",
        }

    jobs = []
    for pos, t, c in units:
        pos_label = pos.replace("/", "_")
        job = {
            "name": f"base_image_{store}_{pos_label}_t{t}_c{c}_{experiment}",
            "func": build_pyramid_unit_worker,
            "kwargs": {
                "experiment": experiment,
                "position": pos,
                "t": t,
                "c": c,
                "source_store": str(store_path),
                "levels": num_levels,
                "factor": 2,
                "resume": False,
            },
            "metadata": {
                "type": "base_image",
                "store": store,
                "position": pos,
                "t": t,
                "c": c,
                "needs_reshard": zarr_format == 3,
                "store_path": str(store_path),
                "num_levels": num_levels,
            },
            "slurm_params": unit_slurm,
            "label": "base_image",
        }
        jobs.append(job)

    return jobs


def _submit_base_image_per_unit_slurm(experiments: list, args) -> int:
    """Submit per-(pos, t, c) base image pyramid jobs via SLURM.

    Used by ``--base-image --slurm`` CLI path.
    """
    all_jobs = []
    for experiment in experiments:
        print(f"\n  {experiment}:")
        jobs = _generate_base_image_unit_jobs(
            experiment, store=args.store, num_levels=args.num_levels,
        )
        all_jobs.extend(jobs)

    if not all_jobs:
        print("No jobs to submit.")
        return 0

    unit_slurm = all_jobs[0]["slurm_params"]

    print(f"\n{'='*60}")
    print(f"SLURM Job Submission Plan (per-unit)")
    print(f"{'='*60}\n")
    print(f"Total unit jobs: {len(all_jobs)}")
    print(f"SLURM Resources (per job):")
    print(f"  Timeout: {unit_slurm['timeout_min']} min")
    print(f"  Memory: {unit_slurm['mem']}")
    print(f"  CPUs: {unit_slurm['cpus_per_task']}")
    print(f"\n{'='*60}\n")

    if not args.yes:
        try:
            response = input(f"Submit {len(all_jobs)} jobs to SLURM? [y/N]: ").strip().lower()
            if response not in ("y", "yes"):
                print("\nCancelled.\n")
                return 0
        except (KeyboardInterrupt, EOFError):
            print("\n\nCancelled.\n")
            return 0

    experiment_label = experiments[0] if len(experiments) == 1 else f"batch_{len(experiments)}"
    log_dir = f"slurm_base_image_logs/{experiment_label}"

    result = submit_parallel_jobs(
        jobs_to_submit=all_jobs,
        experiment=f"base_image_{experiment_label}",
        slurm_params=unit_slurm,
        log_dir=log_dir,
        manifest_prefix="base_image",
        dry_run=False,
        wait_for_completion=not args.no_wait,
        verbose=True,
    )

    total_failed = len(result.get("failed", []) or [])
    if total_failed > 0:
        print(f"\n  {total_failed} unit jobs failed")

    # Reshard after all unit jobs complete
    if not args.no_wait and total_failed == 0:
        reshard_failed = _run_post_fix_resharding(all_jobs, experiment_label, quiet=False)
        if reshard_failed > 0:
            return 1

    return 1 if total_failed > 0 else 0


def main():
    parser = argparse.ArgumentParser(
        description="Unified batch builder for OPS pyramid overlays and segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build ISS overlays for v3 stores
  python -m cyclops_process.processes.pyramids.audit_fix --iss-labels --all

  # Build seg pyramids for v2 stores
  python -m cyclops_process.processes.pyramids.audit_fix --seg-pyramids --zarr-version 2 --all

  # Build for specific experiments (supports shorthand names like "103" or "ops103")
  python -m cyclops_process.processes.pyramids.audit_fix --iss-labels -e 103
  python -m cyclops_process.processes.pyramids.audit_fix --iss-labels --seg-pyramids -e ops0069 ops0103

  # Build organelle segmentation pyramids (v3 zarr labels/ group)
  python -m cyclops_process.processes.pyramids.audit_fix --organelle-pyramids -e 33
  python -m cyclops_process.processes.pyramids.audit_fix --organelle-pyramids --all

  # Dry run to see what needs building
  python -m cyclops_process.processes.pyramids.audit_fix --iss-labels --seg-pyramids --dry-run
        """,
    )

    # Build type selection (can combine)
    build_group = parser.add_argument_group("Build types (can combine)")
    build_group.add_argument(
        "--iss-labels",
        "--iss",  # backward-compat alias; --iss-labels is the canonical name
        dest="iss_labels",
        action="store_true",
        help="Build ISS label overlays (iss_gene_image with gene names + "
             "iss_guide_image with guide sequences). NOT to be confused with "
             "ISS store pyramids — these are rendered text overlays drawn from "
             "the linked_pheno_iss.csv data.",
    )
    build_group.add_argument(
        "--seg-pyramids",
        action="store_true",
        help="Build segmentation pyramids (use --seg-types to specify which labels)",
    )
    build_group.add_argument(
        "--organelle-pyramids",
        action="store_true",
        help="Build organelle segmentation pyramids (auto-discovers labels in v3 zarr labels/ group)",
    )
    build_group.add_argument(
        "--cell-painting-pyramids",
        action="store_true",
        help="Build pyramids for cell painting channels (added to phenotyping_v3.zarr via add_cell_painting_channels.py)",
    )
    build_group.add_argument(
        "--build-grid",
        action="store_true",
        help="Build grid overlay (tile boundaries + IDs) from stitch config",
    )
    build_group.add_argument(
        "--base-image",
        action="store_true",
        help="Build base image pyramids (main channels, not segmentation labels)",
    )
    build_group.add_argument(
        "--clims",
        action="store_true",
        help="Rebuild contrast limits for all stores (pheno, iss, track). Use --store to limit to one.",
    )
    build_group.add_argument(
        "--audit",
        action="store_true",
        help="Audit v3 stores and report missing components with fix commands. Use with -e.",
    )
    build_group.add_argument(
        "--store-path",
        type=Path,
        default=None,
        help="Audit a specific zarr path directly instead of resolving by experiment. "
             "Use with --audit; pair with --store to specify which store type the path "
             "represents (default pheno). -e is still required (used for the wells-config "
             "filter and fix-command generation). Only the named store is audited; the "
             "other two are skipped.",
    )
    build_group.add_argument(
        "--fix",
        action="store_true",
        help="Audit v3 stores and submit SLURM jobs to fix all missing components. Use with -e.",
    )
    build_group.add_argument(
        "--fix-yx-scale",
        action="store_true",
        help="Rewrite pheno_assembled_v3 zarr.json YX scale from the legacy "
             "0.65 µm/px to native 0.325 µm/px (idempotent, with .bak sidecars). "
             "Only touches the pheno store. Use with -e.",
    )
    build_group.add_argument(
        "--fix-normalization",
        action="store_true",
        help="Mirror the top-level `normalization` field into "
             "`custom_metadata.normalization` in pheno_assembled_v3 zarr.json "
             "(idempotent, with .bak sidecars), matching the convert_v3 schema. "
             "Only touches the pheno store. Use with -e.",
    )
    build_group.add_argument(
        "--label-filter",
        nargs="+",
        type=str,
        default=None,
        help="Filter organelle labels to build (e.g., --label-filter nuclo_phase_seg mcher_seg). Supports partial matching.",
    )
    build_group.add_argument(
        "--channel-start",
        type=int,
        default=4,
        help="Start channel index for cell painting pyramid build (default: 4, after Phase2D/Focus3D/nuclei/membrane)",
    )
    build_group.add_argument(
        "--channel-end",
        type=int,
        default=12,
        help="End channel index (exclusive) for cell painting pyramid build (default: 12 = 8 cell painting channels)",
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--all",
        action="store_true",
        help="Process all experiments needing builds (batch SLURM mode)",
    )
    mode_group.add_argument(
        "-e", "--experiments",
        nargs="+",
        type=str,
        help="Process only specific experiments (supports shorthand like '103' or 'ops103')",
    )
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Only scan and report status, don't build",
    )

    # Zarr version
    parser.add_argument(
        "--zarr-version",
        type=int,
        choices=[2, 3],
        default=3,
        help="Zarr format version to target (default: 3)",
    )

    # Position filter
    parser.add_argument(
        "-p", "--positions",
        nargs="+",
        type=str,
        default=None,
        help="Only process specific positions (e.g., A/1/0 A/1/1). If not specified, all positions are processed.",
    )

    # Seg pyramid options
    seg_group = parser.add_argument_group("Segmentation pyramid options")
    seg_group.add_argument(
        "--store",
        type=str,
        choices=["pheno", "iss", "track", "all", "bf"],
        default="pheno",
        help="Store(s) to build pyramids for (default: pheno; 'bf' = BF-slice titration v3 store)",
    )
    seg_group.add_argument(
        "--seg-types",
        nargs="+",
        type=str,
        default=["seg", "nuclear_seg"],
        help="Segmentation label arrays to build pyramids for (default: seg nuclear_seg). Can specify any label name, e.g., phase_2d_seg",
    )
    seg_group.add_argument(
        "--num-levels",
        type=int,
        default=5,
        help="Number of pyramid levels (default: 5)",
    )
    seg_group.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="Force rebuild all levels, even if they already exist",
    )

    # ISS options
    iss_group = parser.add_argument_group("ISS overlay options")
    iss_group.add_argument(
        "--font-size",
        type=int,
        default=24,
        help="Font size for gene labels (default: 24)",
    )
    iss_group.add_argument(
        "--point-radius",
        type=int,
        default=3,
        help="Radius of dots in pixels (default: 3)",
    )
    iss_group.add_argument(
        "--guide-text-offset",
        type=int,
        default=28,
        help="Vertical offset for guide text below gene name (default: 28)",
    )

    # Grid overlay options
    grid_group = parser.add_argument_group("Grid overlay options")
    grid_group.add_argument(
        "--grid-line-width",
        type=int,
        default=2,
        help="Line width for grid boundaries (default: 2)",
    )
    grid_group.add_argument(
        "--grid-font-size",
        type=int,
        default=120,
        help="Font size for tile ID labels (default: 120)",
    )

    # Common options
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild, overwriting existing data",
    )

    # SLURM parameters
    slurm_group = parser.add_argument_group("SLURM options")
    slurm_group.add_argument(
        "--slurm-memory",
        type=str,
        default="400GB",
        help="Memory per SLURM job (default: 400GB)",
    )
    slurm_group.add_argument(
        "--slurm-time",
        type=int,
        default=60,
        help="Time limit per SLURM job in minutes (default: 60)",
    )
    slurm_group.add_argument(
        "--slurm-cpus",
        type=int,
        default=64,
        help="CPUs per SLURM job (default: 64)",
    )

    # Batch control
    batch_group = parser.add_argument_group("Batch control")
    batch_group.add_argument(
        "--slurm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Submit as SLURM job(s) instead of running locally. Default ON for "
             "pyramid builds, OFF for --clims (use --no-slurm to force local).",
    )
    batch_group.add_argument(
        "--yes", "-y",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip confirmation prompt. Default ON for pyramid builds, OFF for "
             "--clims (use --no-yes to force the prompt).",
    )
    batch_group.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for SLURM jobs to complete",
    )
    batch_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity",
    )

    # Config paths
    parser.add_argument(
        "--experiment-configs-dir",
        type=Path,
        default=Path(
            f"{BASE_PATH}/configs/experiment_configs"
        ),
        help="Path to experiment_configs directory",
    )

    args = parser.parse_args()

    # --slurm and -y default ON for pyramid builds, OFF for clims (light, local)
    # and the inline metadata fixes (which don't touch SLURM at all). Explicit
    # --slurm/--no-slurm / -y/--no-yes always win.
    _clims_only = args.clims and not (
        args.base_image or args.seg_pyramids or args.organelle_pyramids
        or args.cell_painting_pyramids or args.build_grid or args.iss_labels
    )
    if args.slurm is None:
        args.slurm = not _clims_only
    if args.yes is None:
        args.yes = not _clims_only

    # Validate: at least one build type must be specified
    # Handle --audit early (doesn't need other build flags)
    if args.audit:
        if args.store_path is not None and not args.experiments:
            parser.error(
                "--store-path requires -e/--experiments (used for the wells-config filter "
                "and fix-command generation)"
            )
        if args.store_path is not None and args.store == "all":
            parser.error(
                "--store-path can only target a single store type — use --store pheno|iss|track "
                "(default pheno)"
            )
        if args.experiments:
            exps = [resolve_experiment_name(e, autoselect=True) for e in args.experiments]
        else:
            import os
            base = f"{BASE_PATH}"
            exps = sorted(
                d for d in os.listdir(base)
                if d.startswith("ops0") and (Path(base) / d / "3-assembly").exists()
            )
            print(f"Auditing all {len(exps)} experiments...\n")
        for exp in exps:
            try:
                audit_v3_stores(
                    exp,
                    verbose=True,
                    store_path_override=args.store_path,
                    store_type_override=args.store if args.store_path is not None else None,
                )
            except Exception as e:
                print(f"  [ERROR] {exp}: {e}\n")
        return 0

    # Handle --fix-yx-scale early (metadata-only edit). Repairs the level-0 YX
    # spacing regression on pheno (native 0.325) and/or track (native 1.3).
    # Use --store pheno|track|all (default pheno) to select which.
    if args.fix_yx_scale:
        if not args.experiments:
            parser.error("--fix-yx-scale requires -e/--experiments")
        store_types = (
            ["pheno", "track"] if args.store == "all" else [args.store]
        )
        store_types = [s for s in store_types if s in ("pheno", "track")]
        if not store_types:
            parser.error("--fix-yx-scale supports --store pheno|track|all only")
        total_fixed = 0
        total_err = 0
        for exp_input in args.experiments:
            exp = resolve_experiment_name(exp_input, autoselect=True)
            try:
                dataset = OpsDataset(exp)
            except Exception as e:
                print(f"  [ERROR] {exp}: {e}")
                total_err += 1
                continue
            for store_type in store_types:
                native_yx = (
                    PHENO_NATIVE_YX if store_type == "pheno" else TRACK_NATIVE_YX
                )
                store_path = _get_store_path(
                    dataset, zarr_version=3, store_type=store_type
                )
                if not store_path or not store_path.exists():
                    print(f"  [SKIP] {exp} ({store_type}): no v3 store")
                    continue
                r = _fix_yx_scale_one(store_path, native_yx=native_yx)
                status = ("fixed" if r["fixed"] else ("already_correct"
                          if r["already"] and not r["errors"] else "no_change"))
                print(f"  {status:<16s} {exp:<25s} [{store_type:<5s}] "
                      f"fixed={r['fixed']:>4d}  already={r['already']:>4d}  "
                      f"skipped={r['skipped']:>3d}  errors={len(r['errors'])}")
                for e in r["errors"][:3]:
                    print(f"      ✗ {e}")
                total_fixed += r["fixed"]
                total_err += len(r["errors"])
        print(f"\nTotal: fixed={total_fixed}, errors={total_err}")
        return 0 if total_err == 0 else 1

    # Handle --fix-normalization early (metadata-only edit on pheno store)
    if args.fix_normalization:
        if not args.experiments:
            parser.error("--fix-normalization requires -e/--experiments")
        total_fixed = 0
        total_err = 0
        for exp_input in args.experiments:
            exp = resolve_experiment_name(exp_input, autoselect=True)
            try:
                dataset = OpsDataset(exp)
                pheno = _get_store_path(dataset, zarr_version=3, store_type="pheno")
            except Exception as e:
                print(f"  [ERROR] {exp}: {e}")
                total_err += 1
                continue
            if not pheno or not pheno.exists():
                print(f"  [SKIP] {exp}: no pheno_assembled_v3 store")
                continue
            r = _fix_normalization_one(pheno)
            status = ("fixed" if r["fixed"] else ("already_correct"
                      if r["already"] and not r["errors"] and not r["no_source"]
                      else "no_change"))
            print(f"  {status:<16s} {exp:<25s} fixed={r['fixed']:>4d}  "
                  f"already={r['already']:>4d}  no_source={r['no_source']:>3d}  "
                  f"errors={len(r['errors'])}")
            for e in r["errors"][:3]:
                print(f"      ✗ {e}")
            total_fixed += r["fixed"]
            total_err += len(r["errors"])
        print(f"\nTotal: fixed={total_fixed}, errors={total_err}")
        return 0 if total_err == 0 else 1

    # Handle --fix early (audit + submit SLURM jobs for all missing components)
    if args.fix:
        if not args.experiments:
            parser.error("--fix requires -e/--experiments")
        for exp_input in args.experiments:
            exp = resolve_experiment_name(exp_input, autoselect=True)
            fix_v3_stores(
                exp,
                no_wait=getattr(args, 'no_wait', False),
                quiet=getattr(args, 'quiet', False),
            )
        return 0

    if not (args.iss_labels or args.seg_pyramids or args.organelle_pyramids or args.cell_painting_pyramids or args.build_grid or args.base_image or args.clims):
        parser.print_help()
        print("\nError: At least one build type must be specified (--iss, --seg-pyramids, --organelle-pyramids, --cell-painting-pyramids, --build-grid, --base-image, or --clims)")
        return 1

    # Default --store to "all" when --clims is used alone (process pheno + iss + track)
    if args.clims and args.store == "pheno" and not (args.seg_pyramids or args.base_image):
        args.store = "all"

    # Build SLURM parameters with build-type specific defaults
    # ISS builds benefit from more CPUs due to tile-level parallelization
    using_defaults = args.slurm_time == 60 and args.slurm_cpus == 64 and args.slurm_memory == "400GB"

    if args.iss_labels and using_defaults:
        # Use ISS-optimized defaults if user didn't override
        slurm_params = {
            "timeout_min": 25,
            "mem": "350GB",
            "cpus_per_task": 64,
            "slurm_partition": "cpu,gpu",
        }
        print("[ISS] Using optimized SLURM params: 25min, 64 CPUs, 350GB mem")
    elif (args.seg_pyramids or args.organelle_pyramids) and using_defaults:
        # Per-well pyramid jobs: I/O bound, 32 CPUs is plenty.
        # 128GB needed for full-assembly labels like cell_seg (~42GB int32 at 104k x 104k).
        slurm_params = {
            "timeout_min": 30,
            "mem": "128GB",
            "cpus_per_task": 32,
            "slurm_partition": "cpu,gpu",
        }
        print("[Seg Pyramids] Using optimized SLURM params: 30min, 32 CPUs, 128GB mem (per-well jobs)")
    elif args.build_grid and using_defaults:
        # Grid overlay is lightweight - uses ~2GB RAM and ~10min
        slurm_params = {
            "timeout_min": 15,
            "mem": "32GB",
            "cpus_per_task": 32,
            "slurm_partition": "cpu,gpu",
        }
        print("[Grid] Using optimized SLURM params: 15min, 32 CPUs, 32GB mem")
    elif args.clims and using_defaults:
        # Clims: light I/O, reads a few blocks per channel
        slurm_params = {
            "timeout_min": 2,
            "mem": "16GB",
            "cpus_per_task": 8,
            "slurm_partition": "cpu,gpu",
        }
        print("[Clims] Using optimized SLURM params: 2min, 8 CPUs, 16GB mem")
    else:
        slurm_params = {
            "timeout_min": args.slurm_time,
            "mem": args.slurm_memory,
            "cpus_per_task": args.slurm_cpus,
            "slurm_partition": "cpu,gpu",
        }

    # Determine build description
    build_types = []
    if args.iss_labels:
        build_types.append("iss_image")
    if args.seg_pyramids:
        build_types.append("seg_pyramids")
    if args.organelle_pyramids:
        build_types.append("organelle_pyramids")
    if args.cell_painting_pyramids:
        build_types.append("cell_painting_pyramids")
    if args.build_grid:
        build_types.append("grid")
    if args.base_image:
        build_types.append("base_image")
    if args.clims:
        build_types.append("clims")
    build_desc = "+".join(build_types)

    # --dry-run mode
    if args.dry_run:
        print(f"Scanning for experiments in: {args.experiment_configs_dir}")
        print(f"Zarr version: v{args.zarr_version}\n")

        scan_results = scan_experiments(
            experiment_configs_dir=args.experiment_configs_dir,
            zarr_version=args.zarr_version,
            check_iss=args.iss_labels,
            check_seg_pyramids=args.seg_pyramids,
        )
        print_scan_results(scan_results, args.zarr_version)
        print("Dry run - exiting without processing")
        return 0

    # --all mode: batch SLURM submission
    if args.all:
        # Use first build type for detection
        primary_build_type = build_types[0]
        experiments_to_process, experiments_completed = detect_experiments_for_build(
            build_type=primary_build_type,
            zarr_version=args.zarr_version,
            force=args.force,
            verbose=not args.quiet,
        )

        if not experiments_to_process:
            print(f"\nAll experiments are complete! No {build_desc} jobs needed.\n")
            return 0

        # Print summary
        print(f"\n{'='*60}")
        print(f"Batch {build_desc} Submission: {len(experiments_to_process)} experiments")
        print(f"Zarr version: v{args.zarr_version}")
        print(f"{'='*60}\n")

        for exp, n_done, n_total, _ in experiments_to_process:
            print(f"  {exp}")

        # Build job list
        all_jobs = []
        for experiment, _, _, _ in experiments_to_process:
            job = {
                "name": f"{build_desc}_{experiment}",
                "func": _run_single_experiment_build,
                "kwargs": {
                    "experiment": experiment,
                    "build_iss": args.iss_labels,
                    "build_seg_pyramids": args.seg_pyramids,
                    "build_organelle_pyramids": args.organelle_pyramids,
                    "build_cell_painting_pyramids": args.cell_painting_pyramids,
                    "build_grid": args.build_grid,
                    "build_base_image": args.base_image,
                    "build_clims_flag": args.clims,
                    "zarr_version": args.zarr_version,
                    "seg_types": args.seg_types,
                    "store": args.store,
                    "force": args.force,
                    "font_size": args.font_size,
                    "point_radius": args.point_radius,
                    "guide_text_offset": args.guide_text_offset,
                    "grid_line_width": args.grid_line_width,
                    "grid_font_size": args.grid_font_size,
                    "num_levels": args.num_levels,
                    "resume": not args.no_resume,
                    "label_filter": args.label_filter,
                    "channel_start": args.channel_start,
                    "channel_end": args.channel_end,
                },
                "metadata": {
                    "experiment": experiment,
                    "step": build_desc,
                },
            }
            all_jobs.append(job)

        # Show job plan
        print(f"\n{'='*60}")
        print(f"Job Submission Plan")
        print(f"{'='*60}\n")
        print(f"Total jobs: {len(all_jobs)}")
        print(f"Build types: {build_desc}")
        print(f"SLURM Resources (per job):")
        print(f"  Timeout: {slurm_params['timeout_min']} min")
        print(f"  Memory: {slurm_params['mem']}")
        print(f"  CPUs: {slurm_params['cpus_per_task']}")
        print(f"\n{'='*60}\n")

        # Confirmation
        if not args.yes:
            try:
                response = input(f"Submit {len(all_jobs)} jobs to SLURM? [y/N]: ").strip().lower()
                if response not in ['y', 'yes']:
                    print("\nCancelled. No jobs submitted.\n")
                    return 0
            except (KeyboardInterrupt, EOFError):
                print("\n\nCancelled. No jobs submitted.\n")
                return 0
            print()
        else:
            print("Proceeding with submission (--yes flag provided)...\n")

        # Submit jobs
        result = submit_parallel_jobs(
            jobs_to_submit=all_jobs,
            experiment=f"batch_{build_desc}_{len(experiments_to_process)}_experiments",
            slurm_params=slurm_params,
            log_dir=f"slurm_{build_desc}_logs/all/%j",
            manifest_prefix=f"{build_desc}_batch",
            dry_run=False,
            wait_for_completion=not args.no_wait,
            verbose=not args.quiet,
        )

        # Save experiment-to-job mapping in the all/ directory
        if result.get("success"):
            from collections import defaultdict
            import yaml

            base_job_id = result.get("base_job_id")
            jobs_list = result.get("jobs", [])

            # Build experiment -> job IDs mapping
            exp_to_jobs = defaultdict(list)
            for job in jobs_list:
                exp = job.get("experiment", "unknown")
                job_id = job.get("job_id", job.get("array_index", "?"))
                exp_to_jobs[exp].append(job_id)

            # Save mapping to all/ directory
            manifest_dir = Path(f"slurm_logs/slurm_{build_desc}_logs/all")
            manifest_dir.mkdir(parents=True, exist_ok=True)
            manifest_file = manifest_dir / f"experiment_mapping_{base_job_id}.yaml"

            mapping_data = {
                "slurm_array_id": base_job_id,
                "total_jobs": len(jobs_list),
                "experiments": {exp: {"job_ids": ids, "count": len(ids)} for exp, ids in sorted(exp_to_jobs.items())},
            }

            with open(manifest_file, "w") as f:
                yaml.dump(mapping_data, f, default_flow_style=False, sort_keys=False)

            print(f"\nExperiment mapping saved: {manifest_file}")

        if result.get("success"):
            return 0 if result.get("all_completed", True) else 1
        return 1

    # --experiments mode: local or SLURM processing
    if args.experiments:
        # Resolve experiment names (e.g., "103" -> "ops0103_20250612")
        resolved_experiments = []
        for exp_input in args.experiments:
            resolved = resolve_experiment_name(exp_input)
            resolved_experiments.append(resolved)

        print(f"Processing experiments: {resolved_experiments}")
        print(f"Build types: {build_desc}")
        print(f"Zarr version: v{args.zarr_version}")
        print(f"Mode: {'SLURM' if args.slurm else 'local'}\n")

        # For organelle pyramids, discover and show labels before building
        if args.organelle_pyramids:
            print(f"{'='*60}")
            print("Discovering organelle segmentation labels...")
            print(f"{'='*60}\n")

            all_labels_by_exp = {}
            filtered_labels_by_exp = {}
            for experiment in resolved_experiments:
                labels, store_path, positions = _discover_organelle_labels(experiment)
                all_labels_by_exp[experiment] = labels

                # Apply filter if provided
                if args.label_filter and labels:
                    filtered = []
                    for label in labels:
                        for pattern in args.label_filter:
                            if pattern in label or label in pattern:
                                filtered.append(label)
                                break
                    filtered_labels_by_exp[experiment] = filtered
                else:
                    filtered_labels_by_exp[experiment] = labels

                if labels:
                    print(f"  {experiment}:")
                    for label in labels:
                        # Mark which labels will be built
                        if label in filtered_labels_by_exp[experiment]:
                            print(f"    - {label} [WILL BUILD]")
                        else:
                            print(f"    - {label} (skipped by filter)")
                else:
                    print(f"  {experiment}: No organelle labels found (*_seg)")

            # Check if any labels will be built after filtering
            total_to_build = sum(len(labels) for labels in filtered_labels_by_exp.values())
            if total_to_build == 0:
                if args.label_filter:
                    print(f"\nNo labels match filter: {args.label_filter}")
                else:
                    print("\nNo organelle segmentation labels found.")
                print("Nothing to build.")
                return 0

            print(f"\n{'='*60}")
            if args.label_filter:
                print(f"Filter: {args.label_filter}")
            print(f"Will build: {total_to_build} labels across {len([e for e, l in filtered_labels_by_exp.items() if l])} experiments")
            print(f"{'='*60}\n")

            # Prompt for confirmation
            if not args.yes:
                try:
                    response = input("Build pyramids for these labels? [y/N]: ").strip().lower()
                    if response not in ['y', 'yes']:
                        print("\nCancelled. No pyramids built.\n")
                        return 0
                except (KeyboardInterrupt, EOFError):
                    print("\n\nCancelled. No pyramids built.\n")
                    return 0
                print()

        # SLURM mode: submit jobs for each experiment (split by well for seg/organelle pyramids)
        if args.slurm:
            # Base image only: use per-(pos, t, c) unit jobs for max parallelism
            if args.base_image and not (args.seg_pyramids or args.organelle_pyramids or args.iss_labels or args.build_grid or args.clims or args.cell_painting_pyramids):
                return _submit_base_image_per_unit_slurm(resolved_experiments, args)

            all_jobs = []
            split_by_well = args.seg_pyramids or args.organelle_pyramids or args.base_image
            for experiment in resolved_experiments:
                # Discover wells to split into per-well jobs
                well_positions = [None]  # Default: single job for all positions
                if split_by_well:
                    try:
                        dataset = OpsDataset(experiment)
                        store_path = _get_store_path(dataset, args.zarr_version, args.store)
                        if store_path and store_path.exists():
                            all_positions = _iter_position_paths(store_path)
                            if all_positions:
                                # Group positions by well (e.g., "A/1/0" -> "A/1")
                                from collections import defaultdict
                                wells = defaultdict(list)
                                for pos in all_positions:
                                    well_key = "/".join(pos.split("/")[:2])
                                    wells[well_key].append(pos)
                                well_positions = list(wells.values())
                                print(f"  {experiment}: splitting into {len(well_positions)} per-well jobs")
                    except Exception as e:
                        print(f"  {experiment}: could not discover wells ({e}), using single job")

                for well_pos_list in well_positions:
                    # Build a descriptive name
                    if well_pos_list is not None:
                        well_name = well_pos_list[0].replace("/", "_").rsplit("_", 1)[0]
                        job_name = f"{build_desc}_{experiment}_{well_name}"
                    else:
                        job_name = f"{build_desc}_{experiment}"

                    job = {
                        "name": job_name,
                        "func": _run_single_experiment_build,
                        "kwargs": {
                            "experiment": experiment,
                            "build_iss": args.iss_labels,
                            "build_seg_pyramids": args.seg_pyramids,
                            "build_organelle_pyramids": args.organelle_pyramids,
                            "build_cell_painting_pyramids": args.cell_painting_pyramids,
                            "build_grid": args.build_grid,
                            "build_base_image": args.base_image,
                    "build_clims_flag": args.clims,
                            "zarr_version": args.zarr_version,
                            "seg_types": args.seg_types,
                            "store": args.store,
                            "force": args.force,
                            "font_size": args.font_size,
                            "point_radius": args.point_radius,
                            "guide_text_offset": args.guide_text_offset,
                            "grid_line_width": args.grid_line_width,
                            "grid_font_size": args.grid_font_size,
                            "num_levels": args.num_levels,
                            "resume": not args.no_resume,
                            "label_filter": args.label_filter,
                            "channel_start": args.channel_start,
                            "channel_end": args.channel_end,
                            "positions": well_pos_list,
                        },
                        "metadata": {
                            "experiment": experiment,
                            "step": build_desc,
                        },
                    }
                    all_jobs.append(job)

            # Show job plan
            print(f"\n{'='*60}")
            print(f"SLURM Job Submission Plan")
            print(f"{'='*60}\n")
            print(f"Total jobs: {len(all_jobs)}")
            print(f"Build types: {build_desc}")
            print(f"SLURM Resources (per job):")
            print(f"  Timeout: {slurm_params['timeout_min']} min")
            print(f"  Memory: {slurm_params['mem']}")
            print(f"  CPUs: {slurm_params['cpus_per_task']}")
            print(f"\n{'='*60}\n")

            # Confirmation
            if not args.yes:
                try:
                    response = input(f"Submit {len(all_jobs)} jobs to SLURM? [y/N]: ").strip().lower()
                    if response not in ['y', 'yes']:
                        print("\nCancelled. No jobs submitted.\n")
                        return 0
                except (KeyboardInterrupt, EOFError):
                    print("\n\nCancelled. No jobs submitted.\n")
                    return 0
                print()
            else:
                print("Proceeding with submission (--yes flag provided)...\n")

            # Determine log directory based on number of experiments
            if len(resolved_experiments) == 1:
                # Single experiment: organize by experiment name
                log_dir = f"slurm_{build_desc}_logs/{resolved_experiments[0]}"
            else:
                # Multiple experiments: use all/ subdirectory
                log_dir = f"slurm_{build_desc}_logs/all/%j"

            # Submit jobs
            result = submit_parallel_jobs(
                jobs_to_submit=all_jobs,
                experiment=f"batch_{build_desc}_{len(resolved_experiments)}_experiments",
                slurm_params=slurm_params,
                log_dir=log_dir,
                manifest_prefix=f"{build_desc}_batch",
                dry_run=False,
                wait_for_completion=not args.no_wait,
                verbose=not args.quiet,
            )

            # Save experiment-to-job mapping for multi-experiment batches
            if result.get("success") and len(resolved_experiments) > 1:
                from collections import defaultdict
                import yaml

                base_job_id = result.get("base_job_id")
                jobs_list = result.get("jobs", [])

                # Build experiment -> job IDs mapping
                exp_to_jobs = defaultdict(list)
                for job in jobs_list:
                    exp = job.get("experiment", "unknown")
                    job_id = job.get("job_id", job.get("array_index", "?"))
                    exp_to_jobs[exp].append(job_id)

                # Save mapping to all/ directory
                manifest_dir = Path(f"slurm_logs/slurm_{build_desc}_logs/all")
                manifest_dir.mkdir(parents=True, exist_ok=True)
                manifest_file = manifest_dir / f"experiment_mapping_{base_job_id}.yaml"

                mapping_data = {
                    "slurm_array_id": base_job_id,
                    "total_jobs": len(jobs_list),
                    "experiments": {exp: {"job_ids": ids, "count": len(ids)} for exp, ids in sorted(exp_to_jobs.items())},
                }

                with open(manifest_file, "w") as f:
                    yaml.dump(mapping_data, f, default_flow_style=False, sort_keys=False)

                print(f"\nExperiment mapping saved: {manifest_file}")

            if result.get("success"):
                return 0 if result.get("all_completed", True) else 1
            return 1

        # Local mode: process each experiment sequentially
        for i, experiment in enumerate(resolved_experiments, 1):
            print(f"\n[{i}/{len(resolved_experiments)}] {experiment}")

            result = _run_single_experiment_build(
                experiment=experiment,
                build_iss=args.iss_labels,
                build_seg_pyramids=args.seg_pyramids,
                build_organelle_pyramids=args.organelle_pyramids,
                build_cell_painting_pyramids=args.cell_painting_pyramids,
                build_grid=args.build_grid,
                build_base_image=args.base_image,
                build_clims_flag=args.clims,
                zarr_version=args.zarr_version,
                seg_types=args.seg_types,
                store=args.store,
                force=args.force,
                font_size=args.font_size,
                point_radius=args.point_radius,
                guide_text_offset=args.guide_text_offset,
                grid_line_width=args.grid_line_width,
                grid_font_size=args.grid_font_size,
                num_levels=args.num_levels,
                resume=not args.no_resume and not args.force,
                label_filter=args.label_filter,
                channel_start=args.channel_start,
                channel_end=args.channel_end,
                positions=args.positions,
            )

            print(f"  {result}")

        print(f"\n{'='*80}")
        print(f"Batch processing complete!")
        print(f"{'='*80}")
        return 0

    # No mode specified
    parser.print_help()
    print("\nError: Specify --all, --experiments, or --dry-run")
    return 1


if __name__ == "__main__":
    sys.exit(main())
