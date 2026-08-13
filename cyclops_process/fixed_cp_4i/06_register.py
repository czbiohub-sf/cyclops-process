#!/usr/bin/env python3
"""Auto-register fixed-cell segmentations via SLURM (cell painting + 4i).

Step 06 in the unified pipeline. Reads the 5x nuclear-seg stores produced by
step 05 (``<unit_stem>_max_proj_flatfield_segmentation.zarr``) and writes chained
register YAMLs. Modality-aware via ``--modality {cp,4i}`` (default ``cp``); ``--cp``
is a backwards-compatible alias for ``--modality cp``.

Source-path resolution and register-YAML naming both come from ``modality_config``
so the two modalities share one code path; only the chaining topology differs:

CP mode (--modality cp):
  Chained: part1 → pheno, partN → part(N-1). Writes
  2-tracking/A{well}_cell_painting{part}_register.yml.

4i mode (--modality 4i):
  1. Cross-round: rounds 2-5 → round 1 (same-modality DAPI, PCC only)
  2. 4i→pheno:    round 1 → phenotyping segmentation (cross-modality, RANSAC)
  Writes 0-convert/4i/registration/A{well}_4i_round{n}_register.yml.

Usage:
    # CP (default): chained register for all wells, both parts
    python -m cyclops_process.fixed_cp_4i.06_register -e ops0094 --force

    # 4i: register all (cross-round + round1→pheno) for all wells
    python -m cyclops_process.fixed_cp_4i.06_register --modality 4i -e ops0144_20260406

    # 4i: specific wells / rounds / types
    python -m cyclops_process.fixed_cp_4i.06_register --modality 4i -e ops0144 --type cross-round
    python -m cyclops_process.fixed_cp_4i.06_register --modality 4i -e ops0144 --type to-pheno --wells 1 2

    # Check / refine existing YAMLs (works for either modality)
    python -m cyclops_process.fixed_cp_4i.06_register -e ops0094 --check-all
    python -m cyclops_process.fixed_cp_4i.06_register --modality 4i -e ops0144 --refine-all
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.getcwd())

from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.data.filesystem import resolve_experiment_name, parse_well
from cyclops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
from cyclops_process.processes.auto_register.auto_register import (
    auto_estimate_registration,
    DEFAULT_PARAMS,
)
from cyclops_process.fixed_cp_4i.configs.four_i_config import NUM_ROUNDS, EXPERIMENT
from cyclops_process.fixed_cp_4i.configs.modality_config import get_modality

SLURM_PARAMS = {
    "timeout_min": 30,
    "mem": "250GB",
    "cpus_per_task": 32,
    "slurm_partition": "cpu,gpu",
}


def register_pair(
    source_seg_path: str,
    target_seg_path: str,
    output_yaml: str,
    well: int,
    label: str,
    verbose: bool = True,
    skip_ransac: bool = False,
) -> dict:
    """Register one segmentation against a target. Called by SLURM.

    skip_ransac=True forces PCC-only alignment (no RANSAC refinement).
    Use for same-modality pairs (e.g. CP{N}→CP{N-1}, or 4i round{N}→round1)
    where the images are already well co-registered and RANSAC over-corrects.
    """
    source_seg_path = Path(source_seg_path)
    target_seg_path = Path(target_seg_path)
    output_yaml = Path(output_yaml)
    row, col = parse_well(well)
    position = f"{row}/{col}/0"

    output_yaml.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Auto-registering {label}")
        print(f"  Source: {source_seg_path}")
        print(f"  Target: {target_seg_path}")
        print(f"  Output: {output_yaml}")
        print(f"  skip_ransac: {skip_ransac}")
        print(f"{'='*60}")

    params = DEFAULT_PARAMS.copy()
    params["center_fraction"] = 1.0
    params["pcc_center_fraction"] = 1.0
    params["pcc_downsample_factor"] = 8
    params["skip_ransac"] = skip_ransac

    overlay_dir = output_yaml.parent / "auto_overlays" / output_yaml.stem

    results = auto_estimate_registration(
        source_seg_path=source_seg_path,
        target_seg_path=target_seg_path,
        position=position,
        output_yaml_path=output_yaml,
        t_idx_source=0,
        t_idx_target=0,
        params=params,
        create_overlays=True,
        overlay_output_dir=overlay_dir,
        verbose=verbose,
        use_cache=True,
    )

    return results


def check_yaml(
    source_seg_path: str,
    target_seg_path: str,
    yaml_path: str,
    well: int,
    label: str,
    verbose: bool = True,
) -> dict:
    """Check a registration YAML: compute overlap metrics and generate overlays.

    Uses the same create_final_alignment_overlays as auto_register for consistent visuals.
    """
    import numpy as np
    import yaml as _yaml
    from scipy.ndimage import affine_transform as _aff
    from cyclops_process.processes.auto_register.auto_register_visualization import (
        load_mask_2d,
        create_final_alignment_overlays,
        compute_crop_regions,
    )
    from cyclops_process.processes.auto_register.auto_register_pcc import (
        estimate_translation_pcc,
    )

    source_seg_path = Path(source_seg_path)
    target_seg_path = Path(target_seg_path)
    yaml_path = Path(yaml_path)
    row, col = parse_well(well)
    position = f"{row}/{col}/0"

    if not yaml_path.exists():
        return {"label": label, "status": "skipped"}

    # Load affine (stored as inverse — apply directly with scipy for forward warp)
    with open(yaml_path) as f:
        aff_4x4 = np.array(_yaml.safe_load(f)["affine_transform_zyx"])

    # Build 3x3 affine for 2D operations
    affine_3x3 = np.eye(3)
    affine_3x3[:2, :2] = aff_4x4[1:3, 1:3]
    affine_3x3[:2, 2] = aff_4x4[1:3, 3]

    # Load full-res masks
    mask_src_full = load_mask_2d(source_seg_path, position, 0)
    mask_tgt_full = load_mask_2d(target_seg_path, position, 0)

    # Compute overlap at multiple downsample levels
    results = {}
    for ds, ds_label in [(4, "4x"), (8, "8x"), (16, "16x")]:
        mask_src = (mask_src_full[::ds, ::ds] > 0).astype(np.uint8)
        mask_tgt = (mask_tgt_full[::ds, ::ds] > 0).astype(np.uint8)

        a3 = np.eye(3)
        a3[:2, :2] = affine_3x3[:2, :2]
        a3[:2, 2] = affine_3x3[:2, 2] / ds

        warped = _aff(mask_src.astype(float), a3[:2, :2], offset=a3[:2, 2],
                      output_shape=mask_tgt.shape, order=0, mode='constant')
        warped = (warped > 0.5).astype(np.uint8)

        inter = int(np.sum(warped & mask_tgt))
        union = int(np.sum(warped | mask_tgt))
        iou = inter / union if union > 0 else 0
        mask_pct = inter / np.sum(mask_tgt) * 100 if np.sum(mask_tgt) > 0 else 0

        results[f"iou_{ds_label}"] = round(iou, 3)
        results[f"mask%_{ds_label}"] = round(mask_pct, 1)

    # Generate overlays using the standard auto_register visualization
    overlay_dir = yaml_path.parent / "check_overlays" / yaml_path.stem
    try:
        pcc_shift, _ = estimate_translation_pcc(
            source_seg_path, target_seg_path,
            position, position, 0,
            downsample_factor=8, use_cache=True, center_fraction=1.0,
        )
        crop_regions = compute_crop_regions(mask_src_full.shape, crop_size=1024)
        create_final_alignment_overlays(
            mask_src=mask_src_full,
            mask_tgt=mask_tgt_full,
            affine_3x3=affine_3x3,
            crop_region=crop_regions,
            output_dir=overlay_dir,
            pcc_shift=pcc_shift,
            auto_yaml_path=yaml_path,
            verbose=verbose,
        )
    except Exception as e:
        print(f"  [{label}] Overlay error: {e}")

    if verbose:
        print(f"  [{label}] IoU: 4x={results['iou_4x']:.3f} 8x={results['iou_8x']:.3f} 16x={results['iou_16x']:.3f} | mask%: {results['mask%_8x']:.1f}%")

    results["label"] = label

    # Save per-registration metrics CSV
    import csv
    overlay_dir.mkdir(parents=True, exist_ok=True)
    csv_path = overlay_dir / "auto_register_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in results.items():
            writer.writerow([k, v])

    return results


def refine_yaml(
    source_seg_path: str,
    target_seg_path: str,
    yaml_path: str,
    well: int,
    label: str,
    verbose: bool = True,
    skip_ransac: bool = False,
) -> dict:
    """Refine an existing registration YAML by re-running seeded from current affine.

    Backs up the original, re-runs auto_estimate_registration with the existing
    affine as seed, compares mask IoU before/after, and keeps whichever is better.
    """
    import shutil
    import numpy as np
    import yaml as _yaml
    from scipy.ndimage import affine_transform as _aff
    from cyclops_process.processes.auto_register.auto_register_visualization import load_mask_2d

    source_seg_path = Path(source_seg_path)
    target_seg_path = Path(target_seg_path)
    yaml_path = Path(yaml_path)
    row, col = parse_well(well)
    position = f"{row}/{col}/0"

    if not yaml_path.exists():
        return {"label": label, "status": "skipped", "reason": "YAML not found"}

    # Backup original
    backup_dir = yaml_path.parent / "backup_affines"
    backup_dir.mkdir(exist_ok=True)
    backup_path = backup_dir / yaml_path.name
    if not backup_path.exists():
        shutil.copy2(yaml_path, backup_path)

    # Load masks at 16x downsample for mask overlap comparison
    _ds = 16
    mask_src = (load_mask_2d(source_seg_path, position, 0)[::_ds, ::_ds] > 0).astype(np.uint8)
    mask_tgt = (load_mask_2d(target_seg_path, position, 0)[::_ds, ::_ds] > 0).astype(np.uint8)
    tgt_sum = int(np.sum(mask_tgt))

    def _mask_pct_from_yaml(yp):
        """Return % of target mask covered by warped source (intersection / target)."""
        with open(yp) as f:
            aff_4x4 = np.array(_yaml.safe_load(f)["affine_transform_zyx"])
        a3 = np.eye(3)
        a3[:2, :2] = aff_4x4[1:3, 1:3]
        a3[:2, 2] = aff_4x4[1:3, 3] / _ds
        warped = _aff(mask_src.astype(float), a3[:2, :2], offset=a3[:2, 2],
                      output_shape=mask_tgt.shape, order=0, mode='constant')
        warped = (warped > 0.5).astype(np.uint8)
        inter = int(np.sum(warped & mask_tgt))
        return (inter / tgt_sum * 100) if tgt_sum > 0 else 0.0

    pct_before = _mask_pct_from_yaml(yaml_path)

    # Re-run registration seeded from current affine
    import tempfile
    refined_yaml = Path(tempfile.mktemp(suffix=".yml"))
    overlay_dir = yaml_path.parent / "auto_overlays" / yaml_path.stem

    params = DEFAULT_PARAMS.copy()
    params["skip_ransac"] = skip_ransac

    # When skip_ransac=True we want fresh PCC (not seeded). Seeded refine sets
    # skip_pcc=True, which combined with skip_ransac would make the run a no-op
    # that just rewrites the existing affine. For same-modality (CP→CP / round→round)
    # we explicitly want to discard the prior (RANSAC-corrupted) affine and
    # re-derive from PCC alone.
    seed_path = None if skip_ransac else yaml_path

    try:
        auto_estimate_registration(
            source_seg_path=source_seg_path,
            target_seg_path=target_seg_path,
            position=position,
            output_yaml_path=refined_yaml,
            t_idx_source=0,
            t_idx_target=0,
            params=params,
            create_overlays=True,
            overlay_output_dir=overlay_dir,
            verbose=verbose,
            seed_affine_path=seed_path,
        )
    except Exception as e:
        refined_yaml.unlink(missing_ok=True)
        print(f"  [{label}] Refinement failed: {e}")
        return {"label": label, "status": "failed", "mask_pct_before": pct_before}

    pct_after = _mask_pct_from_yaml(refined_yaml)

    if pct_after > pct_before:
        shutil.move(str(refined_yaml), str(yaml_path))
        status = "improved"
    else:
        refined_yaml.unlink(missing_ok=True)
        status = "kept_original"

    print(f"  [{label}] Mask overlap: {pct_before:.1f}% → {pct_after:.1f}% ({status})")

    # Persist result for the orchestrator to read after SLURM completes
    import json
    result = {
        "label": label, "status": status,
        "mask_pct_before": float(pct_before), "mask_pct_after": float(pct_after),
    }
    result_path = yaml_path.parent / "refine_results" / f"{yaml_path.stem}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f)

    return result


def _slug(label: str) -> str:
    """SLURM-safe job-name slug from a human label."""
    return label.replace(" ", "_").replace("→", "_to_").replace("%", "")


def _iter_pairs(m, experiment, wells, args, pheno_seg):
    """Yield registration pairs for the modality's chaining topology.

    Source paths and YAML names both resolve through ``modality_config`` so cp and
    4i share this generator; only the chain vs cross-round/to-pheno shape differs.

    Each pair: {source, target, yaml (Path), well, unit, label, skip_ransac}.
    ``target`` may be None (pheno store missing); callers must guard on it.
    """
    reg_dir = m.register_dir(experiment)

    if m.name == "cp":
        # Chained: unit1 → pheno, unitN → unit(N-1).
        units = sorted(args.parts if args.parts is not None else m.default_units)
        for unit in units:
            source = m.seg_store_path(experiment, unit)
            if unit == 1:
                target, tlabel, skip = pheno_seg, "Pheno", False
            else:
                target, tlabel, skip = m.seg_store_path(experiment, unit - 1), f"CP{unit-1}", True
            for well in wells:
                yield {
                    "source": source, "target": target,
                    "yaml": reg_dir / m.register_yaml_name(well, unit),
                    "well": well, "unit": unit,
                    "label": f"CP{unit}→{tlabel} W{well}",
                    "skip_ransac": skip,
                }
        return

    # 4i: cross-round (roundN→round1) + to-pheno (round1→pheno), gated by --type.
    target_r1 = m.seg_store_path(experiment, 1)
    if args.type in ("all", "cross-round"):
        rounds = args.rounds if args.rounds is not None else [u for u in m.default_units if u != 1]
        for rnd in rounds:
            source = m.seg_store_path(experiment, rnd)
            for well in wells:
                yield {
                    "source": source, "target": target_r1,
                    "yaml": reg_dir / m.register_yaml_name(well, rnd),
                    "well": well, "unit": rnd,
                    "label": f"R{rnd}→R1 W{well}",
                    "skip_ransac": True,  # same-modality DAPI — PCC alone suffices
                }
    if args.type in ("all", "to-pheno"):
        for well in wells:
            yield {
                "source": target_r1, "target": pheno_seg,
                "yaml": reg_dir / m.register_yaml_name(well, 1),
                "well": well, "unit": 1,
                "label": f"R1→Pheno W{well}",
                "skip_ransac": False,  # cross-modality — needs RANSAC
            }


def _existing_pairs(m, experiment, wells, args, pheno_seg):
    """Pairs whose source, target, and YAML all exist (for check/refine modes)."""
    out = []
    for p in _iter_pairs(m, experiment, wells, args, pheno_seg):
        if not Path(p["source"]).exists():
            continue
        if p["target"] is None or not Path(p["target"]).exists():
            continue
        if not Path(p["yaml"]).exists():
            continue
        out.append(p)
    return out


def _check_jobs(pairs):
    return [{
        "name": f"check_{_slug(p['label'])}",
        "func": check_yaml,
        "kwargs": {
            "source_seg_path": str(p["source"]),
            "target_seg_path": str(p["target"]),
            "yaml_path": str(p["yaml"]),
            "well": p["well"],
            "label": p["label"],
        },
        "metadata": {"label": p["label"]},
    } for p in pairs]


def _print_refine_summary(pairs):
    """Read refine_result JSONs written by workers and print a before/after table."""
    import json

    rows = []
    for p in pairs:
        yaml_path = Path(p["yaml"])
        result_path = yaml_path.parent / "refine_results" / f"{yaml_path.stem}.json"
        if result_path.exists():
            with open(result_path) as f:
                rows.append(json.load(f))
        else:
            rows.append({"label": p["label"], "status": "no_result",
                         "mask_pct_before": None, "mask_pct_after": None})

    if not rows:
        return

    print(f"\n{'='*72}")
    print(f"  REFINE-ALL SUMMARY (mask overlap %)")
    print(f"{'='*72}")
    print(f"  {'Registration':<22} {'Before %':>10} {'After %':>10} {'Delta':>10} {'Status':<14}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*14}")

    n_improved = 0
    n_kept = 0
    n_missing = 0
    for r in rows:
        label = r.get("label", "?")
        before = r.get("mask_pct_before")
        after = r.get("mask_pct_after")
        status = r.get("status", "?")
        if before is None or after is None:
            print(f"  {label:<22} {'N/A':>10} {'N/A':>10} {'—':>10} {status:<14}")
            n_missing += 1
            continue
        delta = after - before
        if status == "improved":
            n_improved += 1
        else:
            n_kept += 1
        print(f"  {label:<22} {before:>9.1f}% {after:>9.1f}% {delta:>+9.1f}% {status:<14}")

    print(f"\n  Improved: {n_improved} | Kept original: {n_kept} | Missing: {n_missing}")
    print(f"{'='*72}\n")


def _print_check_summary(check_dir: Path):
    """Read auto_register_metrics.csv from all check_overlay dirs and print a summary table."""
    import csv

    check_dir = Path(check_dir)
    if not check_dir.exists():
        print("No check results found.")
        return

    rows = []
    for d in sorted(check_dir.iterdir()):
        csv_path = d / "auto_register_metrics.csv"
        if not csv_path.exists():
            continue
        metrics = {}
        with open(csv_path) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for k, v in reader:
                metrics[k] = v
        rows.append(metrics)

    if not rows:
        print("No check metrics found.")
        return

    # Print table
    print(f"\n{'='*80}")
    print(f"  Registration Check Summary")
    print(f"{'='*80}")
    print(f"  {'Registration':<22} {'IoU@4x':>8} {'IoU@8x':>8} {'IoU@16x':>8} {'Mask%@8x':>10} {'Status':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")

    for r in rows:
        label = r.get("label", "?")
        iou4 = float(r.get("iou_4x", 0))
        iou8 = float(r.get("iou_8x", 0))
        iou16 = float(r.get("iou_16x", 0))
        mask8 = float(r.get("mask%_8x", 0))

        if iou8 >= 0.3:
            status = "GOOD"
        elif iou8 >= 0.15:
            status = "OK"
        else:
            status = "LOW"

        print(f"  {label:<22} {iou4:>8.3f} {iou8:>8.3f} {iou16:>8.3f} {mask8:>9.1f}% {status:>8}")

    print(f"{'='*80}")
    print(f"  Overlays saved to: {check_dir}/")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Auto-register fixed-cell segmentations via SLURM (cell painting + 4i)",
    )
    parser.add_argument("--modality", choices=["cp", "4i"], default="cp",
                        help="Imaging modality (default: cp)")
    parser.add_argument("--cp", action="store_true",
                        help="Alias for --modality cp (backwards compatible)")
    parser.add_argument("--experiment", "-e", default=EXPERIMENT)
    parser.add_argument("--wells", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--rounds", nargs="+", type=int, default=list(range(2, NUM_ROUNDS + 1)),
                        help="[4i] Rounds to register to round 1 (default: 2-5)")
    parser.add_argument("--type", choices=["all", "cross-round", "to-pheno"], default="all",
                        help="[4i] Registration type (default: all)")
    parser.add_argument("--parts", nargs="+", type=int, default=None,
                        help="[CP] Parts to chain-register (default: modality default_units)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing registrations")
    parser.add_argument("--refine-all", action="store_true",
                        help="Refine all existing registrations (re-run seeded from current affine)")
    parser.add_argument("--check-all", action="store_true",
                        help="Check all existing registrations (compute overlap + generate overlays)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local", action="store_true")

    args = parser.parse_args()

    modality = "cp" if args.cp else args.modality
    m = get_modality(modality)
    experiment = resolve_experiment_name(args.experiment, autoselect=True)
    dataset = OpsDataset(experiment)

    # Phenotyping segmentation target (nuclear_seg label in the v3 store;
    # registration helpers resolve it via resolve_seg_array_path).
    pheno_seg = dataset.store_paths.get("pheno_assembled_v3")

    check_dir = m.register_dir(experiment) / "check_overlays"

    # --check-all mode
    if args.check_all:
        pairs = _existing_pairs(m, experiment, args.wells, args, pheno_seg)
        if not pairs:
            print("No existing YAMLs found to check.")
            return

        print(f"\nCheck-all: {len(pairs)} registrations")

        if args.dry_run:
            for p in pairs:
                print(f"  {p['label']}: {Path(p['yaml']).name}")
            return

        jobs = _check_jobs(pairs)

        if args.local:
            for job in jobs:
                job["func"](**job["kwargs"])
        else:
            submit_parallel_jobs(
                jobs_to_submit=jobs,
                experiment=experiment,
                slurm_params=SLURM_PARAMS,
                log_dir=f"{m.name}/register",
                manifest_prefix=f"check_{m.name}",
                dry_run=False,
            )

        _print_check_summary(check_dir)
        return

    # --refine-all mode
    if args.refine_all:
        pairs = _existing_pairs(m, experiment, args.wells, args, pheno_seg)
        if not pairs:
            print("No existing YAMLs found to refine.")
            return

        print(f"\nRefine-all: {len(pairs)} registrations")
        for p in pairs:
            print(f"  {p['label']}: {Path(p['yaml']).name}")

        if args.dry_run:
            return

        jobs = [{
            "name": f"refine_{_slug(p['label'])}",
            "func": refine_yaml,
            "kwargs": {
                "source_seg_path": str(p["source"]),
                "target_seg_path": str(p["target"]),
                "yaml_path": str(p["yaml"]),
                "well": p["well"],
                "label": p["label"],
                "skip_ransac": p.get("skip_ransac", False),
            },
            "metadata": {"label": p["label"]},
        } for p in pairs]

        if args.local:
            for job in jobs:
                job["func"](**job["kwargs"])
        else:
            submit_parallel_jobs(
                jobs_to_submit=jobs,
                experiment=experiment,
                slurm_params=SLURM_PARAMS,
                log_dir=f"{m.name}/register",
                manifest_prefix=f"refine_{m.name}",
                dry_run=False,
            )

        _print_refine_summary(pairs)

        # Auto-run check-all to regenerate overlays + metrics with the refined affines
        print(f"\nRefine complete — running check-all to regenerate overlays and metrics...")
        check_jobs = _check_jobs(pairs)

        if args.local:
            for job in check_jobs:
                job["func"](**job["kwargs"])
        else:
            submit_parallel_jobs(
                jobs_to_submit=check_jobs,
                experiment=experiment,
                slurm_params=SLURM_PARAMS,
                log_dir=f"{m.name}/register",
                manifest_prefix=f"check_{m.name}",
                dry_run=False,
            )

        _print_check_summary(check_dir)
        return

    # --- Registration: one code path for both modalities via _iter_pairs. ---
    jobs = []
    for p in _iter_pairs(m, experiment, args.wells, args, pheno_seg):
        if not Path(p["source"]).exists():
            print(f"WARNING: source segmentation not found: {p['source']}")
            continue
        if p["target"] is None or not Path(p["target"]).exists():
            print(f"WARNING: target segmentation not found: {p['target']}")
            continue
        yaml_path = Path(p["yaml"])
        if yaml_path.exists() and not args.force:
            print(f"SKIP: {yaml_path.name} (exists, use --force to overwrite)")
            continue
        jobs.append({
            "name": _slug(p["label"]),
            "func": register_pair,
            "kwargs": {
                "source_seg_path": str(p["source"]),
                "target_seg_path": str(p["target"]),
                "output_yaml": str(p["yaml"]),
                "well": p["well"],
                "label": p["label"],
                "skip_ransac": p["skip_ransac"],
            },
            "metadata": {"modality": m.name, "unit": p["unit"], "well": p["well"]},
        })

    if not jobs:
        print("No registration jobs to submit.")
        return

    print(f"\n{len(jobs)} registration jobs to submit")

    if args.local:
        for job in jobs:
            print(f"\n{'='*60}")
            job["func"](**job["kwargs"])
    else:
        submit_parallel_jobs(
            jobs_to_submit=jobs,
            experiment=experiment,
            slurm_params=SLURM_PARAMS,
            log_dir=f"{m.name}/register",
            manifest_prefix=f"register_{m.name}",
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
