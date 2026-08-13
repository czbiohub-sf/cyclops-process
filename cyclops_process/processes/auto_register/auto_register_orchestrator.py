"""
SLURM batch submission for automatic registration.

Submits ISS→Track and Pheno→Track registration jobs for wells 1-3 as parallel SLURM jobs.
Each job runs independently with dedicated resources to maximize throughput.

Registration Types (--registration-type / -t):
-----------------------------------------------
  iss           ISS→Track registration only
  pheno         Pheno→Track registration only
  both          ISS→Track and Pheno→Track (default)
  cell_painting CellPainting→Pheno registration

Wells (--wells / -w):
---------------------
  Accepts one or more of: 1 2 3 (default: 1 2 3)

PCC Alignment (--skip-pcc / --use-pcc):
-----------------------------------------
  By default, registration uses PCC-based coarse alignment followed by RANSAC.
  Well 2 pheno→track automatically skips PCC (known to fail for that well).
  --skip-pcc   Skip PCC for all wells (RANSAC only)
  --use-pcc    Force PCC for all wells, including well 2 (overrides auto-skip rule)

Usage:
------
# Submit all 6 jobs (ISS→Track and Pheno→Track for wells 1-3) for a single experiment
python -m cyclops_process.processes.auto_register.auto_register_orchestrator --experiment ops0033_20250429

# Process ALL experiments that need registration (auto-detect missing outputs)
python -m cyclops_process.processes.auto_register.auto_register_orchestrator --all

# Dry run to see which experiments need registration
python -m cyclops_process.processes.auto_register.auto_register_orchestrator --all --dry-run

# Force re-run all experiments even if outputs exist
python -m cyclops_process.processes.auto_register.auto_register_orchestrator --all --force

# Submit only ISS→Track jobs
python -m cyclops_process.processes.auto_register.auto_register_orchestrator --experiment ops0033_20250429 --registration-type iss

# Submit only Pheno→Track jobs
python -m cyclops_process.processes.auto_register.auto_register_orchestrator --experiment 90 --registration-type pheno

# Submit specific wells only
python -m cyclops_process.processes.auto_register.auto_register_orchestrator --experiment ops0033_20250429 --wells 1 2

# Submit a single well
python -m cyclops_process.processes.auto_register.auto_register_orchestrator --experiment ops0033_20250429 --wells 1

# Skip PCC coarse alignment (RANSAC only)
python -m cyclops_process.processes.auto_register.auto_register_orchestrator --experiment ops0033_20250429 --skip-pcc

# Force PCC for all wells including well 2 (overrides auto-skip rule for well 2)
python -m cyclops_process.processes.auto_register.auto_register_orchestrator --experiment ops0033_20250429 --use-pcc

# Combine: specific wells + registration type + skip PCC
python -m cyclops_process.processes.auto_register.auto_register_orchestrator --experiment ops0033_20250429 --registration-type iss --wells 1 2 --skip-pcc

# Process all experiments needing only pheno registration
python -m cyclops_process.processes.auto_register.auto_register_orchestrator --all --registration-type pheno

# Re-run experiments with overlap below custom threshold (e.g., 15%)
python -m cyclops_process.processes.auto_register.auto_register_orchestrator --all --low-quality-threshold 15.0

# Batch mode: collect all jobs from all experiments and submit as one large batch
# This is much faster than sequential processing when handling many experiments
python -m cyclops_process.processes.auto_register.auto_register_orchestrator --all --batch-mode

# Batch mode with dry run to see total job count
python -m cyclops_process.processes.auto_register.auto_register_orchestrator --all --batch-mode --dry-run

# Cell Painting Registration (CellPainting→Pheno):
# Submit cell painting registration for all wells and both parts
python -m cyclops_process.processes.auto_register.auto_register_orchestrator --experiment ops0094_20251217 --registration-type cell_painting

# Submit cell painting registration for specific parts only
python -m cyclops_process.processes.auto_register.auto_register_orchestrator --experiment ops0094_20251217 --registration-type cell_painting --parts 1

# Submit cell painting registration for specific wells
python -m cyclops_process.processes.auto_register.auto_register_orchestrator --experiment ops0094_20251217 --registration-type cell_painting --wells 1 2 3 --parts 1 2

# Recompute overlap metrics from an existing affine YAML (no SLURM, no cached CSV)
python -m cyclops_process.processes.auto_register.auto_register_orchestrator \
    --check-yaml /path/to/A1_auto_register.yml \
    -e ops0033_20250429 -w 1 -t iss

# Recompute metrics + overlays for ALL existing registrations (all wells × iss/pheno)
python -m cyclops_process.processes.auto_register.auto_register_orchestrator --check-all -e ops0033_20250429

# Check only ISS registrations for wells 1 and 2
python -m cyclops_process.processes.auto_register.auto_register_orchestrator --check-all -e ops0033_20250429 -t iss -w 1 2

# Refine all existing affine YAMLs (one SLURM job per YAML, backs up originals)
python -m cyclops_process.processes.auto_register.auto_register_orchestrator --refine-all -e ops0128_20260225

# Refine only pheno registrations for wells 1 and 3
python -m cyclops_process.processes.auto_register.auto_register_orchestrator --refine-all -e ops0128_20260225 -t pheno -w 1 3

# Dry run to see which YAMLs would be refined
python -m cyclops_process.processes.auto_register.auto_register_orchestrator --refine-all -e ops0128_20260225 --dry-run
"""

import argparse
import re
import sys
import os
import time
import subprocess
from pathlib import Path
from datetime import datetime
import yaml
import submitit
import pandas as pd
from prettytable import PrettyTable

# Add project root to path
sys.path.insert(0, os.getcwd())

from cyclops_process.processes.auto_register.auto_register import (
    auto_register_iss_to_track,
    auto_register_pheno_to_track,
    auto_register_cell_painting_to_pheno,
    compose_registration,
    RETRY_STRATEGY_ORDER,
)
from cyclops_process.processes.auto_register.auto_register_utils import (
    print_overlap_quality_warning,
    AutoRegistrationError,
    validate_registration_results,
)
from ops_utils.hpc.slurm_utils import print_slurm_job_stats, format_time
from ops_utils.data.experiment import OpsDataset
from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
from ops_utils.data.filesystem import resolve_experiment_name, parse_well
from ops_utils.profiling.decorators import versioned_function
from cyclops_process.paths import BASE_PATH

# Minimum acceptable centroid overlap percentage for quality gating.
# Used in check-yaml verification, retry logic, and validation.
MIN_CENTROID_OVERLAP_THRESHOLD = 9.0

# Minimum mask overlap that overrides a low centroid score.
MIN_MASK_OVERLAP_OVERRIDE = 30.0


MIN_MASK_OVERLAP_THRESHOLD = 15.0


def registration_passed(centroid_pct: float, mask_pct: float = None) -> bool:
    """Check if a registration passes quality gate.

    Passes if:
      - centroid_pct >= 9%, OR
      - mask_pct >= 30% (mask well aligned regardless of centroid)
    Fails if:
      - mask_pct < 15% (insufficient mask alignment)
    """
    if mask_pct is not None and mask_pct < MIN_MASK_OVERLAP_THRESHOLD:
        return False
    if centroid_pct >= MIN_CENTROID_OVERLAP_THRESHOLD:
        return True
    if mask_pct is not None and mask_pct >= MIN_MASK_OVERLAP_OVERRIDE:
        return True
    return False


# ---------------------------------------------------------------------------
# Refine-all: per-YAML refinement worker and orchestration
# ---------------------------------------------------------------------------

@versioned_function("v1.0")
def _refine_single_yaml(
    experiment: str,
    well,
    reg_type: str,
    skip_track: bool = False,
) -> dict:
    """SLURM worker: refine a single registration YAML by mask IoU grid search.

    Loads the existing affine, runs coarse-to-fine translation+rotation grid
    search on downsampled binary masks, and overwrites the YAML only if
    mask IoU improves. Backs up the original YAML first.

    Returns a dict with before/after mask IoU stats.
    """
    import shutil
    import numpy as np
    import yaml as _yaml

    from cyclops_process.processes.auto_register.auto_register import (
        refine_affine_by_mask_iou,
        resolve_registration_paths,
    )
    from cyclops_process.processes.auto_register.auto_register_utils import (
        affine_3x3_to_4x4_zyx,
        save_affine_to_yaml,
    )
    from cyclops_process.processes.auto_register.auto_register_visualization import load_mask_2d

    dataset = OpsDataset(experiment)
    row, col = parse_well(well)
    position = f"{row}/{col}/0"
    label = f"{experiment}_{reg_type}_w{row}{col}"

    # Resolve paths (same source/target/timepoint as registration used)
    paths = resolve_registration_paths(dataset, experiment, well, reg_type, skip_track=skip_track)

    # Determine YAML path
    if reg_type == "iss":
        yaml_path = Path(dataset.append_well("auto_iss_register", position))
    else:
        yaml_path = Path(dataset.append_well("auto_pheno_register", position))

    if not yaml_path.exists():
        return {"label": label, "status": "skipped", "reason": "YAML not found"}

    # Load existing affine (stored as inverse, convert to forward)
    with open(yaml_path) as f:
        stored = np.array(_yaml.safe_load(f)["affine_transform_zyx"])
    fwd_4x4 = np.linalg.inv(stored)
    fwd_3x3 = np.eye(3)
    fwd_3x3[:2, :2] = fwd_4x4[1:3, 1:3]
    fwd_3x3[:2, 2] = fwd_4x4[1:3, 3]

    # Re-run registration seeded from current affine (includes iterative RANSAC refinement)
    from cyclops_process.processes.auto_register.auto_register import auto_estimate_registration
    import tempfile
    refined_yaml = Path(tempfile.mktemp(suffix=".yml"))
    try:
        auto_estimate_registration(
            source_seg_path=paths["source_seg_path"],
            target_seg_path=paths["target_seg_path"],
            position=paths["position"],
            output_yaml_path=refined_yaml,
            t_idx_source=paths["t_idx_source"],
            t_idx_target=paths["t_idx_target"],
            create_overlays=False,
            verbose=True,
            seed_affine_path=yaml_path,
        )
    except Exception as e:
        refined_yaml.unlink(missing_ok=True)
        return {"label": label, "status": "failed", "error": str(e)}

    # Compare mask IoU: original vs refined
    _ds = 16
    mask_src = (load_mask_2d(paths["source_seg_path"], paths["position"], paths["t_idx_source"])[::_ds, ::_ds] > 0).astype(np.uint8)
    mask_tgt = (load_mask_2d(paths["target_seg_path"], paths["position"], paths["t_idx_target"])[::_ds, ::_ds] > 0).astype(np.uint8)

    from scipy.ndimage import affine_transform as _aff
    def _iou(yaml_p):
        with open(yaml_p) as f:
            stored = np.array(_yaml.safe_load(f)["affine_transform_zyx"])
        fwd_4 = np.linalg.inv(stored)
        A = np.eye(3)
        A[:2, :2] = fwd_4[1:3, 1:3]
        A[:2, 2] = fwd_4[1:3, 3] / _ds
        inv_A = np.linalg.inv(A)
        warped = _aff(mask_src, inv_A[:2, :2], offset=inv_A[:2, 2], order=0, output_shape=mask_tgt.shape)
        inter = (warped & mask_tgt).sum()
        union = (warped | mask_tgt).sum()
        return float(inter / union) if union > 0 else 0.0

    original_iou = _iou(str(yaml_path))
    refined_iou = _iou(str(refined_yaml))

    if refined_iou > original_iou:
        # Backup and save
        backup_dir = dataset.tracking / "backup_affines"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / yaml_path.name
        shutil.copyfile(yaml_path, backup_path)
        shutil.copyfile(str(refined_yaml), str(yaml_path))
        refined_yaml.unlink(missing_ok=True)

        return {
            "label": label, "status": "improved",
            "before_iou": original_iou, "after_iou": refined_iou,
            "delta_iou": refined_iou - original_iou,
            "backup": str(backup_path),
        }

    refined_yaml.unlink(missing_ok=True)
    return {
        "label": label, "status": "unchanged",
        "before_iou": original_iou, "after_iou": refined_iou,
        "delta_iou": 0.0,
    }


def _collect_overlaps(dataset, wells, reg_types, skip_track):
    """Read current overlap % from metrics CSVs for each well/type combo.

    Returns dict: "iss_w1" -> overlap_percent (or None).
    """
    from cyclops_process.processes.auto_register.auto_register import _read_existing_overlap
    overlaps = {}
    for well in wells:
        row, col = parse_well(well)
        well_token = f"{row}{col}"
        for reg_type in reg_types:
            if reg_type == "iss":
                target = "pheno" if skip_track else "track"
                overlay_name = f"{well_token}_iss_to_{target}"
            else:
                overlay_name = f"{well_token}_pheno_to_track"
            # Read from check_yaml/ (refreshed by the gate check-all pre-refine and by
            # the post-refine check-all), NOT auto_overlays/ (a registration-time snapshot
            # that refine never updates — which made the before/after summary always
            # show "unchanged").
            overlay_dir = dataset.tracking / "check_yaml" / overlay_name
            key = f"{reg_type}_w{well_token}"
            overlaps[key] = _read_existing_overlap(overlay_dir)
    return overlaps


def _submit_refine_all_auto(
    experiment: str,
    wells: list,
    reg_types: list,
    skip_track: bool,
    slurm_params: dict,
    verbose: bool = True,
) -> None:
    """Auto-submit refine-all after all registrations pass threshold.

    Called automatically at the end of fresh registration or check-all
    to squeeze out the last few percent of overlap improvement.
    """
    dataset = OpsDataset(experiment)

    # Snapshot before-overlaps. _collect_overlaps reads from check_yaml/ which
    # is populated by check-all; run a pre-refine check-all so the "before"
    # column reflects the actual current state instead of N/A. Skip if the
    # caller already populated check_yaml/ (idempotent — check-all just rewrites
    # the CSVs from the current YAMLs).
    _run_check_all_blocking(experiment, wells, reg_types, skip_track, verbose, prefix="prerefine")
    before_overlaps = _collect_overlaps(dataset, wells, reg_types, skip_track)

    # Build one refine job per existing YAML
    jobs = []
    for well in wells:
        row, col = parse_well(well)
        position = f"{row}/{col}/0"
        for reg_type in reg_types:
            if reg_type == "iss":
                yaml_path = Path(dataset.append_well("auto_iss_register", position))
            elif reg_type == "pheno":
                if skip_track:
                    continue
                yaml_path = Path(dataset.append_well("auto_pheno_register", position))
            else:
                continue
            if not yaml_path.exists():
                continue
            jobs.append({
                "name": f"refine_{experiment}_{reg_type}_w{row}{col}",
                "func": _refine_single_yaml,
                "kwargs": {
                    "experiment": experiment,
                    "well": well,
                    "reg_type": reg_type,
                    "skip_track": skip_track,
                },
                "metadata": {"type": reg_type, "well": well},
            })

    if not jobs:
        return

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Auto-refine: attempting to improve {len(jobs)} registrations")
        print(f"{'='*60}")

    submit_result = submit_parallel_jobs(
        jobs_to_submit=jobs,
        experiment=experiment,
        slurm_params=slurm_params,
        log_dir="auto_register",
        step_name="submit_registration_jobs",
        manifest_prefix="refine_all",
        wait_for_completion=True,
    )

    if submit_result.get("success"):
        # Refresh metrics against the refined affines so the after-snapshot and
        # the persisted auto_register_metrics_combined.csv reflect the new YAMLs.
        _run_check_all_blocking(experiment, wells, reg_types, skip_track, verbose, prefix="postrefine")
        after_overlaps = _collect_overlaps(dataset, wells, reg_types, skip_track)
        _print_refine_summary(before_overlaps, after_overlaps, experiment)


def _run_check_all_blocking(
    experiment: str,
    wells: list,
    reg_types: list,
    skip_track: bool,
    verbose: bool,
    prefix: str,
) -> None:
    """Submit a single check-all SLURM job and block until it finishes. Writes
    metrics CSVs into check_yaml/ — used to populate both the before- and
    after-refine snapshots."""
    from cyclops_process.processes.auto_register.auto_register import check_all_yaml_registrations
    if verbose:
        print(f"\nRunning check-all ({prefix}) to refresh registration metrics...")
    submit_parallel_jobs(
        jobs_to_submit=[{
            "name": f"check_all_{prefix}_{experiment}",
            "func": check_all_yaml_registrations,
            "kwargs": {
                "experiment": experiment,
                "wells": wells,
                "registration_types": reg_types,
                "verbose": verbose,
                "skip_track": skip_track,
            },
        }],
        experiment=experiment,
        slurm_params={
            "timeout_min": 15, "mem": "250GB", "cpus_per_task": 32,
            "slurm_partition": "cpu,gpu",
        },
        log_dir="auto_register",
        manifest_prefix=f"check_all_{prefix}",
        wait_for_completion=True,
        print_resource_summary=False,
    )


def _print_refine_summary(before_overlaps: dict, after_overlaps: dict, experiment: str) -> None:
    """Compare before/after overlaps and print summary table."""
    print(f"\n{'='*70}")
    print(f"  REFINE-ALL SUMMARY — {experiment}")
    print(f"{'='*70}")
    print(f"  {'Registration':<30} {'Before':>10} {'After':>10} {'Delta':>10} {'Status':<12}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")

    n_improved = 0
    n_unchanged = 0
    n_missing = 0

    for key in sorted(before_overlaps.keys()):
        before = before_overlaps.get(key)
        after = after_overlaps.get(key)

        if before is None and after is None:
            continue

        before_str = f"{before:.1f}%" if before is not None else "N/A"
        after_str = f"{after:.1f}%" if after is not None else "N/A"

        if before is not None and after is not None:
            delta = after - before
            delta_str = f"{delta:+.1f}%"
            if delta > 0.1:
                status = "IMPROVED"
                n_improved += 1
            else:
                status = "unchanged"
                n_unchanged += 1
        else:
            delta_str = "—"
            status = "missing"
            n_missing += 1

        print(f"  {key:<30} {before_str:>10} {after_str:>10} {delta_str:>10} {status:<12}")

    print(f"\n  Improved: {n_improved} | Unchanged: {n_unchanged} | Missing: {n_missing}")
    print(f"{'='*70}\n")


def _combine_metrics_csvs(experiment: str, submitted_jobs: list = None, verbose: bool = True) -> Path:
    """
    Combine individual auto_register_metrics.csv files into a single summary CSV.

    Scans ALL wells in the tracking directory (not just submitted jobs) to create
    a comprehensive summary CSV that includes all completed registrations.

    Args:
        experiment: Experiment name
        submitted_jobs: Optional list of submitted job info dicts (for legacy compatibility)
        verbose: Print progress

    Returns:
        Path to combined metrics CSV
    """
    dataset = OpsDataset(experiment)
    tracking_dir = dataset.tracking
    auto_overlays_dir = tracking_dir / "auto_overlays"

    if not auto_overlays_dir.exists():
        if verbose:
            print(f"  ⚠️  No auto_overlays directory found: {auto_overlays_dir}")
        return None

    # Collect all metrics CSVs by scanning the auto_overlays directory
    all_metrics = {}

    # Scan for all overlay subdirectories matching pattern: {row}{col}_{type}_to_{target}
    # Matches both normal mode (to_track) and skip_track mode (to_pheno/to_iss)
    pattern = re.compile(r"([A-Za-z]+\d+)_(iss|pheno)_to_(track|pheno|iss)")

    for overlay_subdir in auto_overlays_dir.iterdir():
        if not overlay_subdir.is_dir():
            continue

        match = pattern.match(overlay_subdir.name)
        if not match:
            continue

        well = match.group(1)  # full row/col token, e.g. "A1" or "B1"
        source_type = match.group(2)  # "iss" or "pheno"
        target_type = match.group(3)  # "track", "pheno", or "iss"
        metrics_csv = overlay_subdir / "auto_register_metrics.csv"

        if not metrics_csv.exists():
            if verbose:
                print(f"  ⚠️  Metrics CSV not found: {overlay_subdir.name}")
            continue

        # Read metrics (CSV has columns: metric, value)
        try:
            df = pd.read_csv(metrics_csv)

            # Create column name based on source and target
            if target_type == "track":
                col_name = f"{source_type}_w{well}"  # e.g., "iss_wA1", "pheno_wB2"
            else:
                col_name = f"{source_type}_to_{target_type}_w{well}"  # e.g., "iss_to_pheno_wA1"

            # Convert to dict with metric as key (overlap metrics are already flattened)
            metrics_dict = dict(zip(df["metric"], df["value"]))

            all_metrics[col_name] = metrics_dict

            if verbose:
                print(f"  ✓ Loaded metrics for {col_name} ({len(metrics_dict)} metrics)")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  Error reading {overlay_subdir.name}: {e}")
            continue

    if not all_metrics:
        if verbose:
            print(f"  ⚠️  No metrics found, skipping combined CSV")
        return None

    # Create combined DataFrame
    combined_df = pd.DataFrame(all_metrics)

    # Sort columns: row-A normal mode first (iss_wA1, ...), then skip_track mode.
    # Tokens now carry the row (A1/B1) so any extra rows fall through to the tail.
    desired_order = []
    # Normal mode: iss→track and pheno→track
    desired_order.extend([f"{t}_wA{w}" for t in ["iss", "pheno"] for w in [1, 2, 3]])
    # Skip_track mode: iss→pheno and pheno→iss
    desired_order.extend([f"iss_to_pheno_wA{w}" for w in [1, 2, 3]])
    desired_order.extend([f"pheno_to_iss_wA{w}" for w in [1, 2, 3]])

    columns_sorted = [c for c in desired_order if c in combined_df.columns]
    columns_sorted += [c for c in combined_df.columns if c not in columns_sorted]
    combined_df = combined_df[columns_sorted]

    # Save to tracking directory
    output_path = tracking_dir / "auto_register_metrics_combined.csv"
    combined_df.to_csv(output_path, index=True)

    if verbose:
        print(f"  ✓ Combined metrics saved to: {output_path}")
        print(f"    Shape: {combined_df.shape[0]} metrics × {combined_df.shape[1]} jobs")

    return output_path


def find_experiments_needing_registration(
    experiment_configs_dir: Path,
    wells: list = [1, 2, 3],
    registration_types: list[str] = ["iss", "pheno"],
    force: bool = False,
    low_quality_threshold: float = 10.0,
) -> tuple[list[tuple[str, int, int, dict]], list[tuple[str, int, int, dict]]]:
    """
    Find all experiments that need automatic registration.

    Args:
        experiment_configs_dir: Path to experiment configs directory
        wells: Wells to check for (default: [1, 2, 3])
        registration_types: Types to check ("iss", "pheno", or both)
        force: If True, include all experiments even if outputs exist
        low_quality_threshold: Centroid overlap threshold below which registrations are considered low quality (default: 10.0%)

    Returns:
        Tuple of (experiments_to_process, experiments_completed)
        Each item is: (experiment_name, n_completed, n_expected, metrics_dict)
    """
    experiments_to_process = []
    experiments_completed = []

    # Find all experiment config files
    config_files = list(experiment_configs_dir.glob("ops*_config.yaml"))
    print(f"Found {len(config_files)} experiment configs")
    print(f"Scanning for experiments needing registration...")

    for config_file in config_files:
        experiment = config_file.stem.replace("_config", "")

        try:
            # Load experiment config to check skip_track setting
            with open(config_file, "r") as f:
                exp_config = yaml.safe_load(f)
            
            # Check if this experiment uses skip_track mode
            skip_track = exp_config.get("auto_register_params", {}).get("skip_track", False)
            
            dataset = OpsDataset(experiment)

            # Check if tracking directory exists
            if not dataset.tracking.exists():
                continue

            # Check for required target segmentation (tracking 5x - always required)
            target_seg = dataset.store_paths["lc_5x_segmentation"]
            if not target_seg.exists():
                continue

            # Check if experiment has required source modalities based on registration types
            has_iss = True
            has_pheno = True

            if "iss" in registration_types:
                iss_seg = dataset.store_paths["iss_segmentation"]
                has_iss = iss_seg.exists()

            # In skip_track mode, pheno registration is computed from ISS (inverse), so don't check for it
            if "pheno" in registration_types and not skip_track:
                pheno_seg = dataset.store_paths["pheno_assembled_v3"]
                has_pheno = pheno_seg.exists()

            # Skip if missing required source segmentations for requested types
            # In skip_track mode, pheno YAMLs aren't needed (derived from ISS)
            required_check = ("iss" in registration_types and not has_iss)
            if not skip_track:
                required_check = required_check or ("pheno" in registration_types and not has_pheno)
            
            if required_check:
                continue

            # Count expected outputs and existing outputs
            expected_outputs = []
            existing_outputs = []

            # Build list of expected outputs based on valid inputs
            for well in wells:
                row, col = parse_well(well)
                position = f"{row}/{col}/0"
                for reg_type in registration_types:
                    # Skip if source segmentation doesn't exist for this registration type
                    if reg_type == "iss" and not has_iss:
                        continue
                    if reg_type == "pheno" and not has_pheno:
                        continue

                    # Skip pheno YAMLs entirely in skip_track mode (computed from ISS inverse)
                    if reg_type == "pheno" and skip_track:
                        continue

                    # Use dataset.append_well to get correct path (matches auto_register.py)
                    if reg_type == "iss":
                        yaml_file = dataset.append_well("auto_iss_register", position)
                    else:  # pheno
                        yaml_file = dataset.append_well("auto_pheno_register", position)

                    expected_outputs.append(yaml_file)
                    if yaml_file.exists():
                        existing_outputs.append(yaml_file)

            # Load metrics from auto_register_metrics.csv files in overlay subdirectories
            metrics_dict = {}
            auto_overlays_dir = dataset.tracking / "auto_overlays"

            # First try to load from individual well-specific CSVs in auto_overlays
            if auto_overlays_dir.exists():
                try:
                    import pandas as pd
                    # Check each well and registration type for metrics
                    for well in wells:
                        row, col = parse_well(well)
                        well_token = f"{row}{col}"
                        for reg_type in registration_types:
                            # Skip if source segmentation doesn't exist for this registration type
                            if reg_type == "iss" and not has_iss:
                                continue
                            if reg_type == "pheno" and not has_pheno:
                                continue

                            # Construct overlay subdirectory path (check both normal and skip_track modes)
                            # Try normal mode first (to_track)
                            overlay_subdir = auto_overlays_dir / f"{well_token}_{reg_type}_to_track"
                            metrics_csv = overlay_subdir / "auto_register_metrics.csv"

                            # If not found, try skip_track mode paths
                            if not metrics_csv.exists():
                                if reg_type == "iss":
                                    overlay_subdir = auto_overlays_dir / f"{well_token}_iss_to_pheno"
                                else:  # pheno
                                    overlay_subdir = auto_overlays_dir / f"{well_token}_pheno_to_iss"
                                metrics_csv = overlay_subdir / "auto_register_metrics.csv"

                            if metrics_csv.exists():
                                df_metrics = pd.read_csv(metrics_csv, index_col=0)
                                job_name = f"{well_token}_{reg_type}_to_track"
                                job_metrics = {}
                                if "overlap_forward_overlap_percent" in df_metrics.index:
                                    job_metrics["overlap"] = df_metrics.loc["overlap_forward_overlap_percent", df_metrics.columns[0]]
                                if "ransac_inlier_ratio" in df_metrics.index:
                                    job_metrics["inlier_ratio"] = df_metrics.loc["ransac_inlier_ratio", df_metrics.columns[0]]
                                metrics_dict[job_name] = job_metrics
                except Exception:
                    pass

            # Fallback: for any expected YAML that exists but doesn't have metrics, add placeholder
            for well in wells:
                row, col = parse_well(well)
                well_token = f"{row}{col}"
                position = f"{row}/{col}/0"
                for reg_type in registration_types:
                    # Skip if source segmentation doesn't exist for this registration type
                    if reg_type == "iss" and not has_iss:
                        continue
                    if reg_type == "pheno" and not has_pheno:
                        continue

                    # Skip pheno YAMLs entirely in skip_track mode (computed from ISS inverse)
                    if reg_type == "pheno" and skip_track:
                        continue

                    # Check if YAML exists
                    if reg_type == "iss":
                        yaml_file = dataset.append_well("auto_iss_register", position)
                    else:
                        yaml_file = dataset.append_well("auto_pheno_register", position)

                    job_name = f"{well_token}_{reg_type}_to_track"

                    # If YAML exists but no metrics were loaded, add placeholder
                    if yaml_file.exists() and job_name not in metrics_dict:
                        metrics_dict[job_name] = {"overlap": None, "inlier_ratio": None}

            n_completed = len(existing_outputs)
            n_expected = len(expected_outputs)

            # Check for low-quality registrations (overlap < threshold)
            has_low_quality = False
            if metrics_dict:
                for job_name, job_metrics in metrics_dict.items():
                    overlap = job_metrics.get("overlap")
                    if overlap is not None and overlap < low_quality_threshold:
                        has_low_quality = True
                        break

            # In force mode, process all experiments with valid inputs (and expected outputs)
            if force and n_expected > 0:
                experiments_to_process.append((experiment, n_completed, n_expected, metrics_dict))
            # Process if some outputs are missing OR has low quality registrations
            elif n_completed < n_expected or has_low_quality:
                experiments_to_process.append((experiment, n_completed, n_expected, metrics_dict))
            # All outputs exist with good quality - mark as completed
            elif n_expected > 0 and n_completed == n_expected:
                experiments_completed.append((experiment, n_completed, n_expected, metrics_dict))

        except Exception as e:
            print(f"  ✗ Error checking {experiment}: {e}")
            continue

    # Sort experiments numerically by ops number
    def get_ops_number(item: tuple) -> int:
        """Extract numeric ops number from experiment tuple (experiment_name, ...) """
        exp_name = item[0]
        try:
            return int(exp_name.split("_")[0].replace("ops", ""))
        except (ValueError, IndexError):
            return 9999  # Put malformed names at end

    experiments_to_process.sort(key=get_ops_number)
    experiments_completed.sort(key=get_ops_number)

    return experiments_to_process, experiments_completed


def _get_wells_below_threshold(
    experiment: str,
    wells: list,
    reg_type: str,
    threshold: float,
    skip_track: bool = False,
) -> list:
    """
    Return wells that fail registration_passed() quality gate.
    """
    dataset = OpsDataset(experiment)
    auto_overlays_dir = dataset.tracking / "auto_overlays"
    below = []

    for well in wells:
        row, col = parse_well(well)
        well_token = f"{row}{col}"
        if reg_type == "iss":
            subdir_name = f"{well_token}_iss_to_{'pheno' if skip_track else 'track'}"
        else:
            subdir_name = f"{well_token}_pheno_to_{'iss' if skip_track else 'track'}"
        metrics_csv = auto_overlays_dir / subdir_name / "auto_register_metrics.csv"

        if not metrics_csv.exists():
            below.append(well)
            continue

        try:
            df = pd.read_csv(metrics_csv, index_col=0)
            centroid = float(df.loc["overlap_forward_overlap_percent", df.columns[0]]) if "overlap_forward_overlap_percent" in df.index else 0.0
            mask = float(df.loc["mask_overlap_forward_overlap_percent", df.columns[0]]) if "mask_overlap_forward_overlap_percent" in df.index else None
            if pd.isna(centroid):
                centroid = 0.0
            if mask is not None and pd.isna(mask):
                mask = None
            if not registration_passed(centroid, mask):
                below.append(well)
        except Exception:
            below.append(well)

    return below


def collect_registration_jobs(
    experiment: str,
    wells: list = [1, 2, 3],
    registration_types: list[str] = ["iss", "pheno"],
    verbose: bool = True,
    skip_track: bool = False,
    skip_pcc: bool = False,
    use_pcc: bool = False,
) -> tuple[list[dict], list[str]]:
    """
    Collect registration jobs for an experiment without submitting them.

    Returns
    -------
    tuple
        (jobs_to_submit, skipped_jobs)
    """
    dataset = OpsDataset(experiment)
    auto_overlays_dir = dataset.tracking / "auto_overlays"


    jobs_to_submit = []
    skipped_jobs = []

    for well in wells:
        row, col = parse_well(well)
        well_token = f"{row}{col}"
        position = f"{row}/{col}/0"

        if "iss" in registration_types:
            iss_target = "pheno" if skip_track else "track"
            yaml_file = dataset.append_well("auto_iss_register", position)
            # Check overlay dir matching current mode first, fallback to other
            overlay_subdir = auto_overlays_dir / f"{well_token}_iss_to_{iss_target}"
            metrics_csv = overlay_subdir / "auto_register_metrics.csv"

            if not metrics_csv.exists():
                fallback_target = "track" if skip_track else "pheno"
                overlay_subdir = auto_overlays_dir / f"{well_token}_iss_to_{fallback_target}"
                metrics_csv = overlay_subdir / "auto_register_metrics.csv"

            should_skip = False
            if yaml_file.exists() and metrics_csv.exists():
                try:
                    df_metrics = pd.read_csv(metrics_csv, index_col=0)
                    if "overlap_forward_overlap_percent" in df_metrics.index:
                        overlap = df_metrics.loc["overlap_forward_overlap_percent", df_metrics.columns[0]]
                        if overlap >= MIN_CENTROID_OVERLAP_THRESHOLD:
                            should_skip = True
                            skipped_jobs.append(f"{experiment}_iss_to_{iss_target}_w{well_token} (overlap={overlap:.1f}%)")
                except Exception:
                    pass

            if not should_skip:
                jobs_to_submit.append({
                    "name": f"{experiment}_iss_to_{iss_target}_w{well_token}",
                    "func": auto_register_iss_to_track,
                    "kwargs": {
                        "experiment": experiment,
                        "well": well,
                        "compare_with_manual": True,
                        "verbose": verbose,
                        "skip_track": skip_track,
                        "skip_pcc": skip_pcc,
                        "force_pcc": use_pcc,
                    },
                    "metadata": {
                        "experiment": experiment,
                        "type": "iss",
                        "well": well,
                    },
                })

        # Skip pheno jobs entirely when skip_track=True (ISS→pheno is sufficient, pheno is inverse)
        if "pheno" in registration_types and not skip_track:
            yaml_file = dataset.append_well("auto_pheno_register", position)
            # Check both normal mode (to_track) and skip_track mode (to_iss)
            overlay_subdir = auto_overlays_dir / f"{well_token}_pheno_to_track"
            metrics_csv = overlay_subdir / "auto_register_metrics.csv"

            if not metrics_csv.exists():
                overlay_subdir = auto_overlays_dir / f"{well_token}_pheno_to_iss"
                metrics_csv = overlay_subdir / "auto_register_metrics.csv"

            should_skip = False
            if yaml_file.exists() and metrics_csv.exists():
                try:
                    df_metrics = pd.read_csv(metrics_csv, index_col=0)
                    if "overlap_forward_overlap_percent" in df_metrics.index:
                        overlap = df_metrics.loc["overlap_forward_overlap_percent", df_metrics.columns[0]]
                        if overlap >= MIN_CENTROID_OVERLAP_THRESHOLD:
                            should_skip = True
                            skipped_jobs.append(f"{experiment}_pheno_to_track_w{well_token} (overlap={overlap:.1f}%)")
                except Exception:
                    pass

            if not should_skip:
                jobs_to_submit.append({
                    "name": f"{experiment}_pheno_to_track_w{well_token}",
                    "func": auto_register_pheno_to_track,
                    "kwargs": {
                        "experiment": experiment,
                        "well": well,
                        "compare_with_manual": True,
                        "verbose": verbose,
                        "skip_track": skip_track,
                        "force_pcc": use_pcc,
                    },
                    "metadata": {
                        "experiment": experiment,
                        "type": "pheno",
                        "well": well,
                    },
                })

    return jobs_to_submit, skipped_jobs


@versioned_function(version="1.0")
def submit_registration_jobs(
    experiment: str,
    wells: list = [1, 2, 3],
    registration_types: list[str] = ["iss", "pheno"],
    slurm_params: dict = None,
    dry_run: bool = False,
    wait_for_completion: bool = True,
    skip_prompt: bool = True,
    verbose: bool = True,
    skip_track: bool = False,
    skip_pcc: bool = False,
    use_pcc: bool = False,
    strategy: str = None,
    skip_check: bool = False,
    force: bool = False,
) -> dict:
    """
    Submit parallel SLURM jobs for automatic registration.

    Parameters
    ----------
    experiment : str
        Experiment name (e.g., "ops0031_20250424")
    wells : list[int]
        Wells to process (default: [1, 2, 3])
    registration_types : list[str]
        Registration types to run: "iss", "pheno", or both (default: both)
    slurm_params : dict
        SLURM parameters (timeout_min, mem, cpus_per_task, etc.)
        If None, uses default from slurm_task_config.yaml
    dry_run : bool
        If True, print what would be submitted without actually submitting
    wait_for_completion : bool
        If True, wait for all jobs to complete before returning (default: True)
    verbose : bool
        Print detailed progress
    skip_track : bool
        If True, register ISS and pheno directly without using tracking data (default: False)

    Returns
    -------
    dict
        Job submission results with job IDs, metadata, and completion status
    """
    # Default SLURM parameters
    if slurm_params is None:
        slurm_params = {
            "timeout_min": 30,
            "mem": "250GB",  # Increased from 200GB for larger datasets
            "cpus_per_task": 32,
            "slurm_partition": "cpu",
        }

    # Check existing outputs and their quality metrics
    dataset = OpsDataset(experiment)

    # Gate: don't launch registration unless EVERY requested type's assembled v3
    # store is built. If any (iss or pheno) is missing, skip the whole step —
    # don't run the others either. Protects --rerun (where DAG deps are bypassed)
    # from firing before the stores exist. force=True bypasses.
    _store_for_type = {
        "iss": dataset.store_paths.get("iss_stitch_registered_v3"),
        "pheno": dataset.store_paths.get("pheno_assembled_v3"),
        "cell_painting": dataset.store_paths.get("pheno_assembled_v3"),
    }
    if not force:
        _missing = [
            f"{rt} ({_store_for_type.get(rt)})"
            for rt in registration_types
            if not (_store_for_type.get(rt) and Path(_store_for_type[rt]).exists())
        ]
        if _missing:
            from cyclops_process.pipelinerunner.exceptions import PipelineHalted
            raise PipelineHalted(
                f"required store(s) not built yet: {', '.join(_missing)} — "
                f"registration can't run, so halting the pipeline. Resolve the "
                f"upstream step(s) and re-run."
            )

    # Check if all expected affine YAMLs already exist
    existing_yamls = {}
    for well in wells:
        row, col = parse_well(well)
        well_token = f"{row}{col}"
        position = f"{row}/{col}/0"
        if "iss" in registration_types:
            yaml_path = dataset.append_well("auto_iss_register", position)
            if yaml_path.exists():
                existing_yamls[f"iss_w{well_token}"] = {"well": well, "type": "iss", "yaml": yaml_path}
        if "pheno" in registration_types and not skip_track:
            yaml_path = dataset.append_well("auto_pheno_register", position)
            if yaml_path.exists():
                existing_yamls[f"pheno_w{well_token}"] = {"well": well, "type": "pheno", "yaml": yaml_path}

    # Count expected jobs
    expected_count = 0
    for well in wells:
        if "iss" in registration_types:
            expected_count += 1
        if "pheno" in registration_types and not skip_track:
            expected_count += 1

    # If all affine YAMLs exist, run check-all to get fresh mask overlap metrics
    jobs_to_submit = []
    skipped_jobs = []

    if skip_check and existing_yamls:
        if verbose:
            print(f"\n--skip-check: skipping quality verification, forcing re-registration of all {expected_count} wells...")
        for well in wells:
            row, col = parse_well(well)
            well_token = f"{row}{col}"
            position = f"{row}/{col}/0"
            if "iss" in registration_types:
                job_name = f"iss_to_track_w{well_token}"
                if strategy and strategy != "default":
                    job_name += f"_{strategy}"
                if strategy == "compose":
                    jobs_to_submit.append({
                        "name": job_name,
                        "func": compose_registration,
                        "kwargs": {
                            "experiment": experiment,
                            "well": well,
                            "reg_type": "iss",
                            "verbose": verbose,
                            "skip_track": skip_track,
                        },
                        "metadata": {"type": "iss", "well": well, "strategy": "compose"},
                    })
                else:
                    jobs_to_submit.append({
                        "name": job_name,
                        "func": auto_register_iss_to_track,
                        "kwargs": {
                            "experiment": experiment,
                            "well": well,
                            "compare_with_manual": True,
                            "verbose": verbose,
                            "skip_track": skip_track,
                            "skip_pcc": skip_pcc,
                            "force_pcc": use_pcc,
                            "strategy": strategy,
                        },
                        "metadata": {"type": "iss", "well": well},
                    })
            if "pheno" in registration_types and not skip_track:
                job_name = f"pheno_to_track_w{well_token}"
                if strategy and strategy != "default":
                    job_name += f"_{strategy}"
                if strategy == "compose":
                    jobs_to_submit.append({
                        "name": job_name,
                        "func": compose_registration,
                        "kwargs": {
                            "experiment": experiment,
                            "well": well,
                            "reg_type": "pheno",
                            "verbose": verbose,
                            "skip_track": skip_track,
                        },
                        "metadata": {"type": "pheno", "well": well, "strategy": "compose"},
                    })
                else:
                    jobs_to_submit.append({
                        "name": job_name,
                        "func": auto_register_pheno_to_track,
                        "kwargs": {
                            "experiment": experiment,
                            "well": well,
                            "compare_with_manual": True,
                            "verbose": verbose,
                            "skip_track": skip_track,
                            "force_pcc": use_pcc,
                            "strategy": strategy,
                        },
                        "metadata": {"type": "pheno", "well": well},
                    })

    elif len(existing_yamls) == expected_count and expected_count > 0 and not force:
        if verbose:
            print(f"\nAll {expected_count} affine YAMLs exist — submitting check-all SLURM job to verify quality...")

        from cyclops_process.processes.auto_register.auto_register import check_all_yaml_registrations

        check_reg_types = [t for t in registration_types if not (t == "pheno" and skip_track)]

        # Submit check-all as a SLURM job (loads full-res masks, shouldn't run on login node)
        check_slurm_result = submit_parallel_jobs(
            jobs_to_submit=[{
                "name": f"check_all_{experiment}",
                "func": check_all_yaml_registrations,
                "kwargs": {
                    "experiment": experiment,
                    "wells": wells,
                    "registration_types": check_reg_types,
                    "verbose": verbose,
                    "skip_track": skip_track,
                },
            }],
            experiment=experiment,
            slurm_params={
                "timeout_min": 15,
                "mem": "250GB",
                "cpus_per_task": 32,
                "slurm_partition": "cpu",
            },
            log_dir="auto_register",
            manifest_prefix="check_all",
            wait_for_completion=True,
        )

        # Read results from the combined CSV that check_all writes
        check_results = {}
        combined_csv = dataset.tracking / "check_yaml" / "auto_register_metrics_combined.csv"
        if combined_csv.exists():
            try:
                df = pd.read_csv(combined_csv)
                for _, row in df.iterrows():
                    label = row.get("registration", "")
                    centroid_pct = row.get("forward_overlap_percent", 0.0)
                    check_results[label] = {
                        "forward": {"overlap_percent": float(centroid_pct) if pd.notna(centroid_pct) else 0.0},
                    }
            except Exception as exc:
                print(f"  ⚠️  Could not read combined CSV: {exc}")
        elif check_slurm_result.get("all_completed"):
            print(f"  ⚠️  Check-all completed but no combined CSV found at {combined_csv}")

        if not check_results:
            print("  ⚠️  Check-all job failed or returned no results — resubmitting all registrations")
            # Fall through to submit all jobs
            for label, info in existing_yamls.items():
                well = info["well"]
                reg_type = info["type"]
                _r, _c = parse_well(well)
                job_name = f"{reg_type}_to_track_w{_r}{_c}"
                if strategy and strategy != "default":
                    job_name += f"_{strategy}"
                func = auto_register_iss_to_track if reg_type == "iss" else auto_register_pheno_to_track
                jobs_to_submit.append({
                    "name": job_name,
                    "func": func,
                    "kwargs": {
                        "experiment": experiment,
                        "well": well,
                        "compare_with_manual": True,
                        "verbose": verbose,
                        "skip_track": skip_track,
                        "skip_pcc": skip_pcc if reg_type == "iss" else False,
                        "force_pcc": use_pcc,
                        "strategy": strategy,
                    },
                    "metadata": {"type": reg_type, "well": well},
                })

        # Use centroid + mask overlap to decide which need re-registration
        if check_results:
            for label, info in existing_yamls.items():
                m = check_results.get(label, {})
                if "error" in m:
                    centroid_pct = 0.0
                    mask_pct = None
                else:
                    centroid_pct = m.get("forward", {}).get("overlap_percent", 0.0)
                    mask_pct = m.get("mask_forward", {}).get("overlap_percent", None)

                if registration_passed(centroid_pct, mask_pct):
                    mask_str = f", mask={mask_pct:.1f}%" if mask_pct is not None else ""
                    skipped_jobs.append(f"{label} (centroid={centroid_pct:.1f}%{mask_str})")
                    if verbose:
                        print(f"  ✓ {label}: centroid={centroid_pct:.1f}%{mask_str} — OK")
                else:
                    if verbose:
                        print(f"  ✗ {label}: centroid overlap {centroid_pct:.1f}% — needs re-registration")
                    well = info["well"]
                    reg_type = info["type"]
                    _r, _c = parse_well(well)
                    job_name = f"{reg_type}_to_track_w{_r}{_c}"
                    if strategy and strategy != "default":
                        job_name += f"_{strategy}"
                    if strategy == "compose":
                        jobs_to_submit.append({
                            "name": job_name,
                            "func": compose_registration,
                            "kwargs": {
                                "experiment": experiment,
                                "well": well,
                                "reg_type": reg_type,
                                "verbose": verbose,
                                "skip_track": skip_track,
                            },
                            "metadata": {"type": reg_type, "well": well, "strategy": "compose"},
                        })
                    else:
                        func = auto_register_iss_to_track if reg_type == "iss" else auto_register_pheno_to_track
                        jobs_to_submit.append({
                            "name": job_name,
                            "func": func,
                            "kwargs": {
                                "experiment": experiment,
                                "well": well,
                                "compare_with_manual": True,
                                "verbose": verbose,
                                "skip_track": skip_track,
                                "skip_pcc": skip_pcc if reg_type == "iss" else False,
                                "force_pcc": use_pcc,
                                "strategy": strategy,
                            },
                            "metadata": {
                                "type": reg_type,
                                "well": well,
                            },
                        })
    else:
        # Not all YAMLs exist — submit missing ones directly
        for well in wells:
            row, col = parse_well(well)
            well_token = f"{row}{col}"
            position = f"{row}/{col}/0"

            if "iss" in registration_types:
                label = f"iss_w{well_token}"
                if label in existing_yamls and not force:
                    skipped_jobs.append(f"{label} (YAML exists)")
                    if verbose:
                        print(f"  ℹ️  Skipping iss_to_track_w{well_token}: affine YAML exists")
                else:
                    job_name = f"iss_to_track_w{well_token}"
                    if strategy and strategy != "default":
                        job_name += f"_{strategy}"
                    jobs_to_submit.append({
                        "name": job_name,
                        "func": auto_register_iss_to_track,
                        "kwargs": {
                            "experiment": experiment,
                            "well": well,
                            "compare_with_manual": True,
                            "verbose": verbose,
                            "skip_track": skip_track,
                            "skip_pcc": skip_pcc,
                            "force_pcc": use_pcc,
                            "strategy": strategy,
                        },
                        "metadata": {"type": "iss", "well": well},
                    })

            if "pheno" in registration_types and not skip_track:
                label = f"pheno_w{well_token}"
                if label in existing_yamls and not force:
                    skipped_jobs.append(f"{label} (YAML exists)")
                    if verbose:
                        print(f"  ℹ️  Skipping pheno_to_track_w{well_token}: affine YAML exists")
                else:
                    job_name = f"pheno_to_track_w{well_token}"
                    if strategy and strategy != "default":
                        job_name += f"_{strategy}"
                    jobs_to_submit.append({
                        "name": job_name,
                        "func": auto_register_pheno_to_track,
                        "kwargs": {
                            "experiment": experiment,
                            "well": well,
                            "compare_with_manual": True,
                            "verbose": verbose,
                            "skip_track": skip_track,
                            "force_pcc": use_pcc,
                            "strategy": strategy,
                        },
                        "metadata": {"type": "pheno", "well": well},
                    })

    # Report skipped jobs
    if skipped_jobs and verbose:
        print(f"\nSkipping {len(skipped_jobs)} jobs with good existing results (centroid overlap ≥ {MIN_CENTROID_OVERLAP_THRESHOLD}%):")
        for job in skipped_jobs:
            print(f"  ✓ {job}")
        print()

    if not jobs_to_submit:
        if len(skipped_jobs) == expected_count:
            print(f"All {expected_count} registrations pass quality check — nothing to resubmit!")
            # Auto-refine to squeeze out last bits of improvement
            check_reg_types = [t for t in registration_types if not (t == "pheno" and skip_track)]
            _submit_refine_all_auto(
                experiment, wells, check_reg_types, skip_track, slurm_params, verbose,
            )
            return {"success": True, "all_completed": True, "skipped": skipped_jobs}
        print("No jobs to submit!")
        return {"success": False, "error": "No jobs to submit"}

    # Prompt user for confirmation before submitting (unless in dry_run mode)
    if not dry_run and not skip_prompt:
        print(f"\n{'='*60}")
        print(f"Ready to submit {len(jobs_to_submit)} jobs:")
        for i, job in enumerate(jobs_to_submit, 1):
            print(f"  {i}. {job['name']}")
        print(f"\nSLURM Resources (per job):")
        print(f"  Timeout: {slurm_params['timeout_min']} min")
        print(f"  Memory: {slurm_params.get('mem', slurm_params.get('slurm_mem', 'N/A'))}")
        print(f"  CPUs: {slurm_params['cpus_per_task']}")
        print(f"  Partition: {slurm_params['slurm_partition']}")
        print(f"{'='*60}\n")

        response = input("Proceed with submission? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("\nSubmission cancelled by user.")
            return {"success": False, "error": "Cancelled by user"}
        print()

    # Define post-completion callback for metrics aggregation
    def _post_completion_callback(submitted_jobs: list[dict], experiment: str):
        """Aggregate metrics CSVs and display quality assessment after jobs complete."""
        # Combine metrics CSVs into single summary file (scans ALL wells, not just submitted)
        print(f"\n{'='*60}")
        print(f"Combining Metrics CSVs")
        print(f"{'='*60}\n")
        combined_path = _combine_metrics_csvs(
            experiment=experiment,
            verbose=verbose,
        )
        if combined_path:
            print(f"\n✓ Combined metrics CSV: {combined_path.name}")
            print(f"  Location: {combined_path.parent}")

            # Display combined metrics as pretty table
            print(f"\n{'='*60}")
            print(f"Combined Metrics Summary")
            print(f"{'='*60}\n")

            df = pd.read_csv(combined_path, index_col=0)

            # Create PrettyTable
            table = PrettyTable()
            table.field_names = ["Metric"] + list(df.columns)
            table.align = "l"  # Left align all columns
            table.align["Metric"] = "l"

            # Add rows, formatting numeric values
            for metric_name, row in df.iterrows():
                row_values = [metric_name]
                for val in row:
                    # Format numbers to 2 decimal places, leave strings as-is
                    if isinstance(val, (int, float)) and not pd.isna(val):
                        row_values.append(f"{val:.2f}")
                    else:
                        row_values.append(str(val) if not pd.isna(val) else "N/A")
                table.add_row(row_values)

            print(table)
            print(f"\n{'='*60}\n")

            # Check for low overlap warnings — filter to relevant registrations only
            # skip_track mode: show only iss_to_pheno_w* columns
            # normal mode: show only iss_w* and pheno_w* columns (not iss_to_pheno)
            centroid_metric = "overlap_forward_overlap_percent"
            mask_metric = "mask_overlap_forward_overlap_percent"
            if centroid_metric in df.index:
                print(f"Registration Quality Assessment:")
                print(f"-" * 60)

                for col in df.columns:
                    is_cross_modal = "_to_" in col
                    if skip_track and not is_cross_modal:
                        continue
                    if not skip_track and is_cross_modal:
                        continue

                    c_val = df.loc[centroid_metric, col] if centroid_metric in df.index else None
                    m_val = df.loc[mask_metric, col] if mask_metric in df.index else None
                    c = float(c_val) if pd.notna(c_val) else 0.0
                    m = float(m_val) if pd.notna(m_val) else None

                    if registration_passed(c, m):
                        m_str = f", mask={m:.1f}%" if m is not None else ""
                        print(f"\n{col}:\n  ✓ Good (centroid={c:.1f}%{m_str}) - registration successful!")
                    else:
                        m_str = f", mask={m:.1f}%" if m is not None else ""
                        print(f"\n{col}:\n  🚨 FAILED (centroid={c:.1f}%{m_str})")

                print(f"\n{'='*60}\n")

    # Submit initial batch (all ISS + pheno jobs with "default" strategy)
    result = submit_parallel_jobs(
        jobs_to_submit=jobs_to_submit,
        experiment=experiment,
        slurm_params=slurm_params,
        log_dir="slurm_auto_register_logs/%j",
        manifest_prefix="auto_register",
        dry_run=dry_run,
        wait_for_completion=wait_for_completion,
        verbose=verbose,
        post_completion_callback=_post_completion_callback if wait_for_completion else None,
    )

    # SLURM-level retry for jobs with low overlap.
    # Each retry strategy is submitted as its own SLURM job, giving it a full
    # time allocation (default 30min). Stops early if all wells reach threshold.
    #
    # Strategy order (after "default" which ran above):
    #
    # 1. "toggle_pcc" — Try the OPPOSITE PCC setting from the default.
    #      Well 2: PCC is ON (auto-skipped by default, maybe it works)
    #      Wells 1,3: PCC is OFF (maybe PCC was hurting alignment)
    #
    # 2. "compose" — Compose from two other successful registrations.
    #      E.g. ISS→Track = Pheno→Track @ ISS→Pheno (run ISS→Pheno on the fly).
    if wait_for_completion and not dry_run and not strategy:
        # Map registration types to their submission functions
        reg_type_to_func = {
            "iss": auto_register_iss_to_track,
            "pheno": auto_register_pheno_to_track,
        }

        # Determine which registration types to retry
        # In skip_track mode, pheno is derived from ISS inverse
        retry_reg_types = [t for t in registration_types if t in reg_type_to_func]
        if skip_track and "pheno" in retry_reg_types:
            retry_reg_types.remove("pheno")

        # Strategies after "default" (which was already submitted above)
        retry_strategies = RETRY_STRATEGY_ORDER[1:]

        for retry_strategy in retry_strategies:
            # Collect all wells that need retry across all registration types
            retry_jobs = []

            for reg_type in retry_reg_types:
                wells_to_retry = _get_wells_below_threshold(
                    experiment, wells, reg_type, MIN_CENTROID_OVERLAP_THRESHOLD, skip_track,
                )

                if not wells_to_retry:
                    continue

                target = "pheno" if skip_track and reg_type == "iss" else "track"
                if reg_type == "pheno" and skip_track:
                    target = "iss"

                if verbose:
                    print(f"\n{'='*60}")
                    print(f"🔄 RETRY: Strategy '{retry_strategy}' for {len(wells_to_retry)} "
                          f"{reg_type} well(s) with overlap < {MIN_CENTROID_OVERLAP_THRESHOLD}%")
                    print(f"   Wells: {wells_to_retry}")
                    print(f"{'='*60}\n")

                for well in wells_to_retry:
                    _r, _c = parse_well(well)
                    well_token = f"{_r}{_c}"
                    if retry_strategy == "compose":
                        # Compose strategy: build affine from other successful registrations
                        retry_jobs.append({
                            "name": f"{reg_type}_to_{target}_w{well_token}_compose",
                            "func": compose_registration,
                            "kwargs": {
                                "experiment": experiment,
                                "well": well,
                                "reg_type": reg_type,
                                "verbose": verbose,
                                "skip_track": skip_track,
                            },
                            "metadata": {
                                "type": reg_type,
                                "well": well,
                                "strategy": "compose",
                            },
                        })
                    else:
                        func = reg_type_to_func[reg_type]
                        job_kwargs = {
                            "experiment": experiment,
                            "well": well,
                            "compare_with_manual": True,
                            "verbose": verbose,
                            "skip_track": skip_track,
                            "force_pcc": use_pcc,
                            "strategy": retry_strategy,
                        }
                        # ISS jobs also accept skip_pcc
                        if reg_type == "iss":
                            job_kwargs["skip_pcc"] = skip_pcc

                        retry_jobs.append({
                            "name": f"{reg_type}_to_{target}_w{well_token}_{retry_strategy}",
                            "func": func,
                            "kwargs": job_kwargs,
                            "metadata": {
                                "type": reg_type,
                                "well": well,
                                "strategy": retry_strategy,
                            },
                        })

            if not retry_jobs:
                if verbose:
                    print(f"\n✓ All wells have overlap ≥ {MIN_CENTROID_OVERLAP_THRESHOLD}%, "
                          f"no further retries needed.")
                # Auto-refine to squeeze out last bits of improvement
                refine_reg_types = [t for t in registration_types if not (t == "pheno" and skip_track)]
                _submit_refine_all_auto(
                    experiment, wells, refine_reg_types, skip_track, slurm_params, verbose,
                )
                break

            # Compose runs a full internal registration — give it double time
            retry_slurm_params = slurm_params.copy()
            if retry_strategy == "compose":
                retry_slurm_params["timeout_min"] = slurm_params.get("timeout_min", 30) * 2

            retry_result = submit_parallel_jobs(
                jobs_to_submit=retry_jobs,
                experiment=experiment,
                slurm_params=retry_slurm_params,
                log_dir="slurm_auto_register_logs/%j",
                manifest_prefix=f"auto_register_retry_{retry_strategy}",
                dry_run=False,
                wait_for_completion=True,
                verbose=verbose,
                post_completion_callback=_post_completion_callback,
            )

            # Merge retry failures into the main result
            result.setdefault("retry_results", {})[retry_strategy] = retry_result
            if retry_result.get("failed"):
                result["failed"] = result.get("failed", []) + retry_result["failed"]

    # Validate final results if we waited for completion
    # Skip validation when an explicit strategy was passed (user is manually testing)
    if wait_for_completion and not dry_run and not strategy:
        # Determine which registration types were actually processed
        # In skip_track mode, only ISS jobs are submitted (pheno is derived from ISS inverse)
        actual_registration_types = registration_types.copy()
        if skip_track and "pheno" in actual_registration_types:
            actual_registration_types.remove("pheno")

        is_valid, error_messages = validate_registration_results(
            experiment=experiment,
            result=result,
            wells=wells,
            registration_types=actual_registration_types,
            min_centroid_overlap_threshold=MIN_CENTROID_OVERLAP_THRESHOLD,
            verbose=verbose,
            skip_track=skip_track,
        )

        if not is_valid:
            raise AutoRegistrationError(
                f"Auto-registration validation failed for {experiment}:\n" +
                "\n".join(error_messages)
            )

    return result


def submit_cell_painting_registration_jobs(
    experiment: str,
    wells: list = [1, 2, 3],
    parts: list[int] = [1, 2],
    slurm_params: dict = None,
    dry_run: bool = False,
    wait_for_completion: bool = True,
    skip_prompt: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Submit parallel SLURM jobs for cell painting to phenotyping registration.

    Parameters
    ----------
    experiment : str
        Experiment name (e.g., "ops0094_20251217")
    wells : list[int]
        Wells to process (default: [1, 2, 3])
    parts : list[int]
        Cell painting parts to register (default: [1, 2])
    slurm_params : dict
        SLURM parameters (timeout_min, mem, cpus_per_task, etc.)
    dry_run : bool
        If True, print what would be submitted without actually submitting
    wait_for_completion : bool
        If True, wait for all jobs to complete before returning (default: True)
    skip_prompt : bool
        If True, skip confirmation prompt
    verbose : bool
        Print detailed progress

    Returns
    -------
    dict
        Job submission results with job IDs, metadata, and completion status
    """
    # Default SLURM parameters
    if slurm_params is None:
        slurm_params = {
            "timeout_min": 30,
            "mem": "250GB",
            "cpus_per_task": 32,
            "slurm_partition": "cpu",
        }

    dataset = OpsDataset(experiment)
    cell_painting_base = dataset.experiment_path / "0-convert" / "cell_painting"

    # Prepare job list
    jobs_to_submit = []
    skipped_jobs = []

    for part in parts:
        # Check if source segmentation exists
        source_seg_path = cell_painting_base / f"part{part}_max_proj_segmentation.zarr"
        if not source_seg_path.exists():
            if verbose:
                print(f"  ⚠️  Skipping part{part}: segmentation not found at {source_seg_path}")
            continue

        for well in wells:
            row, col = parse_well(well)
            well_token = f"{row}{col}"
            # Check if output YAML already exists
            output_yaml = dataset.tracking / f"{well_token}_cell_painting{part}_register.yml"

            if output_yaml.exists():
                skipped_jobs.append(f"cell_painting{part}_to_pheno_w{well_token} (exists)")
                if verbose:
                    print(f"  ℹ️  Skipping cell_painting{part}_to_pheno_w{well_token}: output exists")
                continue

            jobs_to_submit.append({
                "name": f"cell_painting{part}_to_pheno_w{well_token}",
                "func": auto_register_cell_painting_to_pheno,
                "kwargs": {
                    "experiment": experiment,
                    "well": well,
                    "part": part,
                    "compare_with_manual": False,
                    "verbose": verbose,
                },
                "metadata": {
                    "experiment": experiment,
                    "type": f"cell_painting{part}",
                    "well": well,
                },
            })

    # Report skipped jobs
    if skipped_jobs and verbose:
        print(f"\nSkipping {len(skipped_jobs)} jobs with existing results:")
        for job in skipped_jobs:
            print(f"  ✓ {job}")
        print()

    if not jobs_to_submit:
        print("No jobs to submit!")
        return {"success": False, "error": "No jobs to submit"}

    # Prompt user for confirmation before submitting (unless in dry_run mode)
    if not dry_run and not skip_prompt:
        print(f"\n{'='*60}")
        print(f"Ready to submit {len(jobs_to_submit)} cell painting registration jobs:")
        for i, job in enumerate(jobs_to_submit, 1):
            print(f"  {i}. {job['name']}")
        print(f"\nSLURM Resources (per job):")
        print(f"  Timeout: {slurm_params['timeout_min']} min")
        print(f"  Memory: {slurm_params.get('mem', slurm_params.get('slurm_mem', 'N/A'))}")
        print(f"  CPUs: {slurm_params['cpus_per_task']}")
        print(f"  Partition: {slurm_params['slurm_partition']}")
        print(f"{'='*60}\n")

        response = input("Proceed with submission? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("\nSubmission cancelled by user.")
            return {"success": False, "error": "Cancelled by user"}
        print()

    # Submit jobs using shared utility
    result = submit_parallel_jobs(
        jobs_to_submit=jobs_to_submit,
        experiment=experiment,
        slurm_params=slurm_params,
        log_dir="slurm_auto_register_logs/%j",
        manifest_prefix="cell_painting_register",
        dry_run=dry_run,
        wait_for_completion=wait_for_completion,
        verbose=verbose,
    )

    return result


def main():
    """CLI entry point for SLURM batch submission."""
    parser = argparse.ArgumentParser(
        description="Submit automatic registration jobs to SLURM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        help="Experiment name (e.g., ops0031_20250424). Use --all to process all experiments.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all experiments that need registration (auto-detect from configs)",
    )

    parser.add_argument(
        "--experiment-configs-dir",
        type=Path,
        default=Path(f"{BASE_PATH}/configs/experiment_configs"),
        help="Path to experiment configs directory (for --all mode)",
    )

    parser.add_argument(
        "--wells",
        "-w",
        type=str,
        nargs="+",
        default=[1, 2, 3],
        help="Wells to process as full units (default: 1 2 3; e.g. 1 2 B1 A/2/0)",
    )

    parser.add_argument(
        "--registration-type",
        "-t",
        type=str,
        choices=["iss", "pheno", "both", "cell_painting"],
        default="both",
        help="Registration type: iss (ISS→Track), pheno (Pheno→Track), both, or cell_painting (CellPainting→Pheno)",
    )

    parser.add_argument(
        "--parts",
        type=int,
        nargs="+",
        default=[1, 2],
        choices=[1, 2],
        help="Cell painting parts to register (default: 1 2, for --registration-type cell_painting)",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="SLURM timeout in minutes (default: 30)",
    )

    parser.add_argument(
        "--mem",
        type=str,
        default="250GB",
        help="SLURM memory allocation (default: 250GB)",
    )

    parser.add_argument(
        "--cpus",
        type=int,
        default=32,
        help="SLURM CPUs per task (default: 32)",
    )

    parser.add_argument(
        "--partition",
        type=str,
        default="cpu",
        help="SLURM partition (default: cpu)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be submitted without actually submitting",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Reduce verbosity (suppress job output)",
    )

    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit jobs and return immediately without waiting for completion",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if outputs exist (for --all mode)",
    )

    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip the check-all quality verification and force re-registration of all specified wells",
    )

    parser.add_argument(
        "--low-quality-threshold",
        type=float,
        default=9.0,
        help="Centroid overlap threshold below which registrations are re-run (default: 10.0%%)",
    )

    parser.add_argument(
        "--batch-mode",
        action="store_true",
        help="Collect all jobs from all experiments and submit as one large batch (for --all mode)",
    )

    parser.add_argument(
        "--skip-pcc",
        action="store_true",
        help="Skip PCC-based coarse alignment (use RANSAC only)",
    )

    parser.add_argument(
        "--use-pcc",
        action="store_true",
        help="Force PCC coarse alignment for all wells, including well 2 (overrides the auto-skip rule)",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        choices=["default", "toggle_pcc", "refine", "compose"],
        default=None,
        help="Run a specific retry strategy instead of the default. "
             "'refine' uses the existing affine as a seed and re-runs RANSAC on top. "
             "Useful for testing individual strategies on failing wells.",
    )

    parser.add_argument(
        "--check-yaml",
        type=Path,
        default=None,
        metavar="YAML_PATH",
        help="Recompute overlap metrics from an existing affine YAML (no SLURM submission). "
             "Requires --experiment, --wells (single), and --registration-type (iss or pheno).",
    )

    parser.add_argument(
        "--check-all",
        action="store_true",
        help="Recompute overlap metrics and overlays for all existing affine YAMLs "
             "(all wells × iss/pheno) in parallel. Requires --experiment. "
             "Respects --wells and --registration-type to filter.",
    )

    parser.add_argument(
        "--refine-all",
        action="store_true",
        help="Refine all existing affine YAMLs using RANSAC seeded from the current affine. "
             "Submits one SLURM job per YAML. Only overwrites if improvement found. "
             "Backs up all existing YAMLs to 2-tracking/backup_affines/. "
             "Prints before/after stats after completion.",
    )

    parser.add_argument(
        "--local",
        action="store_true",
        help="Run check-all locally instead of submitting to SLURM.",
    )

    args = parser.parse_args()

    # --check-yaml mode: recompute overlap metrics and generate overlays
    if args.check_yaml is not None:
        if not args.experiment:
            parser.error("--check-yaml requires --experiment")
        if len(args.wells) != 1:
            parser.error("--check-yaml requires exactly one well (e.g., --wells 2)")
        if args.registration_type not in ("iss", "pheno"):
            parser.error("--check-yaml requires --registration-type iss or pheno")

        yaml_path = Path(args.check_yaml)
        if not yaml_path.exists():
            print(f"Error: YAML not found: {yaml_path}")
            sys.exit(1)

        resolved_experiment = resolve_experiment_name(
            args.experiment, verbose=True, allow_interactive=True
        )
        if resolved_experiment is None:
            print("No experiment found. Exiting.")
            sys.exit(1)

        # Read skip_track from experiment config YAML
        _dataset = OpsDataset(resolved_experiment)
        _skip_track = False
        _config_file = _dataset.config_paths.get("exp_config")
        if _config_file and Path(_config_file).exists():
            with open(_config_file, "r") as f:
                _cfg = yaml.safe_load(f) or {}
            _skip_track = _cfg.get("auto_register_params", {}).get("skip_track", False)

        # In skip_track mode there is only ONE registration direction (iss→pheno);
        # the pheno→iss direction is just its inverse, so silently coerce
        # --registration-type pheno to iss instead of rendering a pheno_to_iss
        # overlay that doesn't correspond to anything in the pipeline.
        reg_type = args.registration_type
        if _skip_track:
            if reg_type == "pheno":
                print(f"  skip_track=True — coercing --registration-type pheno → iss "
                      f"(pheno_to_iss isn't a separate fit in skip_track mode)")
                reg_type = "iss"
            else:
                print(f"  skip_track=True (from experiment config) — checking iss→pheno")

        from cyclops_process.processes.auto_register.auto_register import check_yaml_registration
        check_yaml_registration(
            experiment=resolved_experiment,
            well=args.wells[0],
            registration_type=reg_type,
            yaml_path=yaml_path,
            skip_track=_skip_track,
        )
        sys.exit(0)

    # --check-all mode: recompute all existing YAMLs in parallel
    if args.check_all:
        if not args.experiment:
            parser.error("--check-all requires --experiment")

        resolved_experiment = resolve_experiment_name(
            args.experiment, verbose=True, allow_interactive=True
        )
        if resolved_experiment is None:
            print("No experiment found. Exiting.")
            sys.exit(1)

        reg_types = None
        if args.registration_type != "both":
            reg_types = [args.registration_type]

        # Read skip_track from experiment config YAML
        _dataset = OpsDataset(resolved_experiment)
        _skip_track = False
        _config_file = _dataset.config_paths.get("exp_config")
        if _config_file and Path(_config_file).exists():
            with open(_config_file, "r") as f:
                _cfg = yaml.safe_load(f) or {}
            _skip_track = _cfg.get("auto_register_params", {}).get("skip_track", False)
        if _skip_track:
            print(f"  skip_track=True (from experiment config)")

        from cyclops_process.processes.auto_register.auto_register import check_all_yaml_registrations

        if not args.local:
            check_result = submit_parallel_jobs(
                jobs_to_submit=[{
                    "name": f"check_all_{resolved_experiment}",
                    "func": check_all_yaml_registrations,
                    "kwargs": {
                        "experiment": resolved_experiment,
                        "wells": args.wells,
                        "registration_types": reg_types,
                        "skip_track": _skip_track,
                    },
                }],
                experiment=resolved_experiment,
                slurm_params={
                    "timeout_min": 15,
                    "mem": "250GB",
                    "cpus_per_task": 32,
                    "slurm_partition": "cpu",
                },
                log_dir="auto_register",
                manifest_prefix="check_all",
                wait_for_completion=True,
            )
        else:
            check_all_yaml_registrations(
                experiment=resolved_experiment,
                wells=args.wells,
                registration_types=reg_types,
                skip_track=_skip_track,
            )
        sys.exit(0)

    # --refine-all mode: refine all existing YAMLs with one SLURM job per YAML
    if args.refine_all:
        if not args.experiment:
            parser.error("--refine-all requires --experiment")

        resolved_experiment = resolve_experiment_name(
            args.experiment, verbose=True, allow_interactive=True
        )
        if resolved_experiment is None:
            print("No experiment found. Exiting.")
            sys.exit(1)

        if args.registration_type == "both":
            reg_types = ["iss", "pheno"]
        else:
            reg_types = [args.registration_type]

        dataset = OpsDataset(resolved_experiment)
        # Read skip_track from experiment config YAML
        skip_track = False
        config_file = dataset.config_paths.get("exp_config")
        if config_file and Path(config_file).exists():
            with open(config_file, "r") as f:
                _cfg = yaml.safe_load(f) or {}
            skip_track = _cfg.get("auto_register_params", {}).get("skip_track", False)
        if skip_track:
            print(f"  skip_track=True (from experiment config) — pheno will be skipped")
            reg_types = [r for r in reg_types if r != "pheno"]

        # Discover existing YAMLs and build one job per YAML
        jobs = []
        for well in args.wells:
            row, col = parse_well(well)
            well_token = f"{row}{col}"
            position = f"{row}/{col}/0"
            for reg_type in reg_types:
                if reg_type == "iss":
                    yaml_path = Path(dataset.append_well("auto_iss_register", position))
                elif reg_type == "pheno":
                    # skip_track=True means pheno is derived from ISS (inverse);
                    # no per-well pheno YAML to refine. Mirrors the same skip
                    # in _submit_refine_all_auto().
                    if skip_track:
                        continue
                    yaml_path = Path(dataset.append_well("auto_pheno_register", position))
                else:
                    continue

                if not yaml_path.exists():
                    print(f"  Skip {reg_type} w{well_token}: YAML not found ({yaml_path.name})")
                    continue

                jobs.append({
                    "name": f"refine_{resolved_experiment}_{reg_type}_w{well_token}",
                    "func": _refine_single_yaml,
                    "kwargs": {
                        "experiment": resolved_experiment,
                        "well": well,
                        "reg_type": reg_type,
                        "skip_track": skip_track,
                    },
                    "metadata": {"type": reg_type, "well": well},
                })

        if not jobs:
            print("No existing YAMLs found to refine.")
            sys.exit(0)

        # Snapshot before-overlaps from existing metrics CSVs
        before_overlaps = _collect_overlaps(dataset, args.wells, reg_types, skip_track)

        print(f"\n  Refine-all: {len(jobs)} registration YAMLs for {resolved_experiment}")
        print(f"  Wells: {args.wells} | Types: {reg_types}")
        print(f"  Backups will be saved to: {dataset.tracking / 'backup_affines'}")
        print(f"\n  Current overlaps:")
        for key, val in sorted(before_overlaps.items()):
            print(f"    {key}: {val:.1f}%" if val is not None else f"    {key}: N/A")

        if args.dry_run:
            for j in jobs:
                print(f"    [DRY RUN] {j['name']}")
            sys.exit(0)

        submit_result = submit_parallel_jobs(
            jobs_to_submit=jobs,
            experiment=resolved_experiment,
            slurm_params={
                "timeout_min": 30,
                "mem": "250GB",
                "cpus_per_task": 32,
                "slurm_partition": args.partition,
            },
            log_dir="auto_register",
            manifest_prefix="refine_all",
            wait_for_completion=not args.no_wait,
        )

        # After refinement, run check-all to regenerate overlays and metrics
        if submit_result.get("success") and not args.no_wait:
            # Snapshot current CSV before check-all overwrites it
            import pandas as pd
            check_yaml_dir = dataset.tracking / "check_yaml"
            combined_csv = check_yaml_dir / "auto_register_metrics_combined.csv"
            prev_df = pd.read_csv(combined_csv) if combined_csv.exists() else None

            print(f"\nRefinement complete. Running check-all for fresh overlays...")
            from cyclops_process.processes.auto_register.auto_register import check_all_yaml_registrations
            check_all_result = submit_parallel_jobs(
                jobs_to_submit=[{
                    "name": f"check_all_{resolved_experiment}",
                    "func": check_all_yaml_registrations,
                    "kwargs": {
                        "experiment": resolved_experiment,
                        "wells": args.wells,
                        "registration_types": reg_types,
                        "skip_track": skip_track,
                    },
                }],
                experiment=resolved_experiment,
                slurm_params={
                    "timeout_min": 15,
                    "mem": "250GB",
                    "cpus_per_task": 32,
                    "slurm_partition": "cpu",
                },
                log_dir="auto_register",
                manifest_prefix="check_all",
                wait_for_completion=True,
            )

            # Print summary comparing previous vs current
            if combined_csv.exists():
                df = pd.read_csv(combined_csv)

                print(f"\n{'='*80}")
                print(f"  REFINE-ALL RESULTS — {resolved_experiment}")
                print(f"{'='*80}")

                if prev_df is not None:
                    print(f"  {'Registration':<16} {'Centroid (prev→now)':>22} {'Mask (prev→now)':>22}")
                    print(f"  {'-'*16} {'-'*22} {'-'*22}")
                    for _, row in df.iterrows():
                        reg = row.get("registration", "?")
                        c_now = row.get("forward_overlap_percent", float("nan"))
                        m_now = row.get("mask_forward_overlap_percent", float("nan"))
                        prev_row = prev_df[prev_df["registration"] == reg]
                        if len(prev_row) > 0:
                            c_prev = prev_row.iloc[0].get("forward_overlap_percent", float("nan"))
                            m_prev = prev_row.iloc[0].get("mask_forward_overlap_percent", float("nan"))
                            c_str = f"{c_prev:.1f}→{c_now:.1f}%"
                            m_str = f"{m_prev:.1f}→{m_now:.1f}%"
                        else:
                            c_str = f"→{c_now:.1f}%"
                            m_str = f"→{m_now:.1f}%"
                        print(f"  {reg:<16} {c_str:>22} {m_str:>22}")
                else:
                    print(f"  {'Registration':<20} {'Centroid':>10} {'Mask':>10} {'Gap':>10}")
                    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
                    for _, row in df.iterrows():
                        c = row.get("forward_overlap_percent", float("nan"))
                        m = row.get("mask_forward_overlap_percent", float("nan"))
                        gap = c - m if pd.notna(c) and pd.notna(m) else float("nan")
                        print(f"  {row['registration']:<20} {c:>9.1f}% {m:>9.1f}% {gap:>+9.1f}pp")

                print(f"{'='*80}\n")

        sys.exit(0)

    # Validate arguments
    if not args.all and not args.experiment:
        parser.error("Either --experiment or --all must be specified")

    # Parse registration types
    if args.registration_type == "both":
        registration_types = ["iss", "pheno"]
    elif args.registration_type == "cell_painting":
        registration_types = ["cell_painting"]
    else:
        registration_types = [args.registration_type]

    # Build SLURM parameters
    slurm_params = {
        "timeout_min": args.timeout,
        "mem": args.mem,
        "cpus_per_task": args.cpus,
        "slurm_partition": args.partition,
    }

    # Handle cell_painting registration separately (single experiment only for now)
    if "cell_painting" in registration_types:
        if args.all:
            parser.error("--all mode is not yet supported for cell_painting registration. Use --experiment instead.")

        resolved_experiment = resolve_experiment_name(
            args.experiment,
            verbose=True,
            allow_interactive=True
        )

        result = submit_cell_painting_registration_jobs(
            experiment=resolved_experiment,
            wells=args.wells,
            parts=args.parts,
            slurm_params=slurm_params,
            dry_run=args.dry_run,
            wait_for_completion=not args.no_wait,
            skip_prompt=False,
            verbose=not args.quiet,
        )

        sys.exit(0 if result.get("success") else 1)

    # Determine which experiments to process
    if args.all:
        print(f"\n{'='*60}")
        print(f"Scanning for experiments needing registration...")
        print(f"{'='*60}\n")

        experiments_to_process, experiments_completed = find_experiments_needing_registration(
            experiment_configs_dir=args.experiment_configs_dir,
            wells=args.wells,
            registration_types=registration_types,
            force=args.force,
            low_quality_threshold=args.low_quality_threshold,
        )

        # Build output lines for both display and file
        from datetime import datetime
        output_lines = []
        output_lines.append("=" * 60)
        output_lines.append("Registration Status Summary")
        output_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output_lines.append("=" * 60)
        output_lines.append(f"\n✓ {len(experiments_completed)} experiments already completed:")

        if experiments_completed:
            for exp_name, n_completed, n_expected, metrics in experiments_completed:
                output_lines.append(f"  - {exp_name} ({n_completed}/{n_expected})")
                if metrics:
                    for job_name, job_metrics in sorted(metrics.items()):
                        overlap = job_metrics.get("overlap")
                        inlier = job_metrics.get("inlier_ratio")

                        # Determine quality indicator based on overlap threshold
                        if overlap is not None and overlap < args.low_quality_threshold:
                            indicator = "🔴"
                        elif overlap is not None:
                            indicator = "🟢"
                        else:
                            indicator = "⚪"

                        if overlap is not None:
                            overlap_str = f"{overlap:.1f}%"
                        else:
                            overlap_str = "N/A"
                        if inlier is not None:
                            inlier_str = f"{inlier*100:.1f}%"
                        else:
                            inlier_str = "N/A"
                        output_lines.append(f"      {indicator} {job_name}: overlap={overlap_str}, inliers={inlier_str}")
        else:
            output_lines.append("  (none)")

        output_lines.append(f"\n⚙ {len(experiments_to_process)} experiments needing registration:")
        if experiments_to_process:
            for exp_name, n_completed, n_expected, metrics in experiments_to_process:
                # Check if this experiment needs re-processing due to low quality
                has_low_quality = False
                if metrics:
                    for job_name, job_metrics in metrics.items():
                        overlap = job_metrics.get("overlap")
                        if overlap is not None and overlap < args.low_quality_threshold:
                            has_low_quality = True
                            break

                # Add indicator for why it needs processing
                reason = ""
                if n_completed == n_expected and has_low_quality:
                    reason = " [low quality]"
                elif n_completed == 0:
                    reason = " [not started]"
                elif n_completed < n_expected:
                    reason = " [incomplete]"

                output_lines.append(f"  - {exp_name} ({n_completed}/{n_expected}){reason}")
                if metrics:
                    for job_name, job_metrics in sorted(metrics.items()):
                        overlap = job_metrics.get("overlap")
                        inlier = job_metrics.get("inlier_ratio")

                        # Determine quality indicator based on overlap threshold
                        if overlap is not None and overlap < args.low_quality_threshold:
                            indicator = "🔴"
                        elif overlap is not None:
                            indicator = "🟢"
                        else:
                            indicator = "⚪"

                        if overlap is not None:
                            overlap_str = f"{overlap:.1f}%"
                        else:
                            overlap_str = "N/A"
                        if inlier is not None:
                            inlier_str = f"{inlier*100:.1f}%"
                        else:
                            inlier_str = "N/A"
                        output_lines.append(f"      {indicator} {job_name}: overlap={overlap_str}, inliers={inlier_str}")
        else:
            output_lines.append("  (none)")
        output_lines.append("\n" + "=" * 60)

        # Print to console
        print("\n" + "\n".join(output_lines) + "\n")

        # Save to file in local auto_register directory
        output_dir = Path(__file__).parent
        output_file = output_dir / "registration_status_summary.txt"

        with open(output_file, 'w') as f:
            f.write("\n".join(output_lines))

        print(f"✓ Saved registration status summary to: {output_file}\n")

        if args.dry_run:
            print("Dry run - exiting without processing")
            sys.exit(0)

        if not experiments_to_process:
            print("No experiments need processing!")
            sys.exit(0)
    else:
        # Resolve experiment name (supports partial matching and interactive selection)
        resolved_experiment = resolve_experiment_name(
            args.experiment,
            verbose=True,
            allow_interactive=True
        )
        experiments_to_process = [resolved_experiment]

    # Submit jobs - either in batch mode or sequentially
    all_results = []

    if args.batch_mode and len(experiments_to_process) > 1:
        # BATCH MODE: Collect all jobs from all experiments first, then submit as one batch
        print(f"\n{'='*60}")
        print(f"Batch Mode: Collecting jobs from {len(experiments_to_process)} experiments")
        print(f"{'='*60}\n")

        all_jobs = []
        all_skipped = []
        jobs_by_experiment = {}  # Track jobs per experiment for detailed display

        for i, exp_item in enumerate(experiments_to_process, 1):
            # Extract experiment name
            if isinstance(exp_item, tuple):
                experiment = exp_item[0]
            else:
                experiment = exp_item

            print(f"[{i}/{len(experiments_to_process)}] Collecting jobs for {experiment}...")

            # Read skip_track from experiment config YAML
            _dataset = OpsDataset(experiment)
            _skip_track = False
            _config_file = _dataset.config_paths.get("exp_config")
            if _config_file and Path(_config_file).exists():
                with open(_config_file, "r") as f:
                    _cfg = yaml.safe_load(f) or {}
                _skip_track = _cfg.get("auto_register_params", {}).get("skip_track", False)
            if _skip_track:
                print(f"  skip_track=True (from experiment config)")

            jobs, skipped = collect_registration_jobs(
                experiment=experiment,
                wells=args.wells,
                registration_types=registration_types,
                verbose=not args.quiet,
                skip_track=_skip_track,
                skip_pcc=args.skip_pcc,
                use_pcc=args.use_pcc,
            )

            all_jobs.extend(jobs)
            all_skipped.extend(skipped)
            jobs_by_experiment[experiment] = {
                "jobs": jobs,
                "skipped": skipped,
            }

            if jobs:
                print(f"  → {len(jobs)} jobs to submit")
            if skipped:
                print(f"  → {len(skipped)} jobs skipped (good overlap)")

        print(f"\n{'='*60}")
        print(f"Batch Summary")
        print(f"{'='*60}")
        print(f"Experiments to process: {len(experiments_to_process)}")
        print(f"Jobs to submit: {len(all_jobs)}")
        print(f"Jobs skipped (already good quality): {len(all_skipped)}")
        print(f"{'='*60}\n")

        if not all_jobs:
            print("No jobs to submit!")
            sys.exit(0)

        # Show detailed breakdown by experiment
        print("Jobs to submit by experiment:")
        for experiment, info in jobs_by_experiment.items():
            if info["jobs"]:
                job_names = [j["name"].replace(f"{experiment}_", "") for j in info["jobs"]]
                print(f"  {experiment}:")
                for job_name in job_names:
                    print(f"    - {job_name}")
        print()

        # Prompt user for confirmation
        if not args.dry_run:
            print(f"{'='*60}")
            print(f"Ready to submit {len(all_jobs)} jobs across {len(experiments_to_process)} experiments")
            print(f"\nSLURM Resources (per job):")
            print(f"  Timeout: {slurm_params['timeout_min']} min")
            print(f"  Memory: {slurm_params.get('mem', slurm_params.get('slurm_mem', 'N/A'))}")
            print(f"  CPUs: {slurm_params['cpus_per_task']}")
            print(f"  Partition: {slurm_params['slurm_partition']}")
            print(f"{'='*60}\n")

            response = input("Proceed with batch submission? [y/N]: ").strip().lower()
            if response not in ['y', 'yes']:
                print("\nBatch submission cancelled by user.")
                sys.exit(0)
            print()

        # Define batch post-completion callback
        def _batch_post_completion_callback(submitted_jobs: list[dict], _: str):
            """Create per-experiment combined metrics CSVs after batch completion."""
            # Get unique experiments from submitted jobs
            experiments = set()
            for job in submitted_jobs:
                exp = job["metadata"].get("experiment")
                if exp:
                    experiments.add(exp)

            if not experiments:
                print("\n⚠️  No experiment metadata found in jobs")
                return

            print(f"\n{'='*60}")
            print(f"Creating Per-Experiment Summary CSVs")
            print(f"{'='*60}\n")

            for exp in sorted(experiments):
                print(f"\n{exp}:")
                combined_path = _combine_metrics_csvs(
                    experiment=exp,
                    verbose=not args.quiet,
                )
                if combined_path:
                    print(f"  ✓ {combined_path.name}")

            print(f"\n{'='*60}")
            print(f"✓ Created summary CSVs for {len(experiments)} experiments")
            print(f"{'='*60}\n")

        # Submit all jobs as one batch with callback
        result = submit_parallel_jobs(
            jobs_to_submit=all_jobs,
            experiment="batch_auto_register",
            slurm_params=slurm_params,
            log_dir="slurm_auto_register_logs/%j",
            manifest_prefix="auto_register_batch",
            dry_run=args.dry_run,
            wait_for_completion=not args.no_wait,
            verbose=not args.quiet,
            post_completion_callback=_batch_post_completion_callback if not args.no_wait else None,
        )
        all_results.append(result)

    else:
        # SEQUENTIAL MODE: Process experiments one at a time
        for i, exp_item in enumerate(experiments_to_process, 1):
            # Extract experiment name (handle both tuple and string format)
            if isinstance(exp_item, tuple):
                experiment = exp_item[0]
            else:
                experiment = exp_item

            if len(experiments_to_process) > 1:
                print(f"\n{'='*60}")
                print(f"[{i}/{len(experiments_to_process)}] Processing {experiment}")
                print(f"{'='*60}\n")

            # Read skip_track from experiment config YAML
            _dataset = OpsDataset(experiment)
            _skip_track = False
            _config_file = _dataset.config_paths.get("exp_config")
            if _config_file and Path(_config_file).exists():
                with open(_config_file, "r") as f:
                    _cfg = yaml.safe_load(f) or {}
                _skip_track = _cfg.get("auto_register_params", {}).get("skip_track", False)
            if _skip_track:
                print(f"  skip_track=True (from experiment config)")

            result = submit_registration_jobs(
                experiment=experiment,
                wells=args.wells,
                registration_types=registration_types,
                slurm_params=slurm_params,
                dry_run=args.dry_run,
                wait_for_completion=not args.no_wait,
                verbose=not args.quiet,
                skip_track=_skip_track,
                skip_pcc=args.skip_pcc,
                use_pcc=args.use_pcc,
                strategy=args.strategy,
                skip_check=args.skip_check,
                force=args.force,
            )
            all_results.append(result)

            # Run check-all after registration to get fresh mask overlap metrics + overlays
            if not args.dry_run and not args.no_wait:
                print(f"\nRunning check-all for fresh overlays and mask metrics...")
                from cyclops_process.processes.auto_register.auto_register import check_all_yaml_registrations
                _check_reg_types = [t for t in registration_types if not (t == "pheno" and _skip_track)]
                check_result = submit_parallel_jobs(
                    jobs_to_submit=[{
                        "name": f"check_all_{experiment}",
                        "func": check_all_yaml_registrations,
                        "kwargs": {
                            "experiment": experiment,
                            "wells": [1, 2, 3],
                            "skip_track": _skip_track,
                        },
                    }],
                    experiment=experiment,
                    slurm_params={
                        "timeout_min": 15,
                        "mem": "250GB",
                        "cpus_per_task": 32,
                        "slurm_partition": "cpu",
                    },
                    log_dir="auto_register",
                    manifest_prefix="check_all",
                    wait_for_completion=True,
                )

    # Exit with appropriate code
    if any(r.get("dry_run") for r in all_results):
        sys.exit(0)
    elif all(r.get("success") for r in all_results):
        # If we waited for completion, check if all succeeded
        if all(r.get("all_completed") is not None for r in all_results):
            all_success = all(r.get("all_completed") for r in all_results)
            sys.exit(0 if all_success else 1)
        else:
            # Didn't wait, assume success
            sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
