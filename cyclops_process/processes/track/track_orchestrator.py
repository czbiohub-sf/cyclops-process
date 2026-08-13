"""
SLURM batch submission for cell tracking.

Submits tracking jobs for wells 1-3 as parallel SLURM jobs.
Each job runs independently with dedicated resources to maximize throughput.

Usage:
------
# Submit tracking for all wells in a single experiment
python -m cyclops_process.processes.track.track_orchestrator --experiment ops0033_20250429

# Submit tracking for specific wells
python -m cyclops_process.processes.track.track_orchestrator --experiment 126 --wells A/1/0 A/2/0

# Process ALL experiments that need tracking (batch mode)
python -m cyclops_process.processes.track.track_orchestrator --all

# Preview what --all mode would submit (dry run)
python -m cyclops_process.processes.track.track_orchestrator --all --dry-run

# Process all with interactive confirmation (default)
python -m cyclops_process.processes.track.track_orchestrator --all

# Process all with automatic confirmation (for scripts)
python -m cyclops_process.processes.track.track_orchestrator --all --yes

# Force reprocess all experiments even if outputs exist
python -m cyclops_process.processes.track.track_orchestrator --all --force

# Dry run for single experiment
python -m cyclops_process.processes.track.track_orchestrator --experiment ops0033_20250429 --dry-run

# Submit without waiting for completion
python -m cyclops_process.processes.track.track_orchestrator --all --no-wait

# Debug mode with ROI cropping
python -m cyclops_process.processes.track.track_orchestrator --experiment ops0033_20250429 --debug --debug-tile-size 2048
"""

import argparse
import re
import sys
from pathlib import Path
import os
import pandas as pd

from cyclops_utils.data.filesystem import parse_well


# Add project root to path
sys.path.insert(0, os.getcwd())

# Lazy import track_wells to avoid NumPy version conflicts in wave_env
# from cyclops_process.processes.track.track import track_wells  # Moved to function level
from cyclops_utils.hpc.slurm_batch_utils import (
    submit_parallel_jobs,
    detect_experiments_needing_processing,
)
from cyclops_utils.data.experiment import OpsDataset


def check_registration_quality(
    experiment: str,
    well,
    min_overlap: float = 9.0,
    verbose: bool = False,
) -> dict[str, bool | None | float]:
    """
    Check if registration quality is adequate for tracking.

    Parameters
    ----------
    experiment : str
        Experiment name
    well
        Well unit or number (e.g. "A/1/0", "B2", 1)
    min_overlap : float
        Minimum overlap percentage required (default: 9.0%)
    verbose : bool
        Print quality details

    Returns
    -------
    dict
        {
            "iss_ok": bool or None (None = no metrics available),
            "pheno_ok": bool or None (None = no metrics available),
            "both_ok": bool,
            "iss_overlap": float or None,
            "pheno_overlap": float or None,
        }
    """
    dataset = OpsDataset(experiment)
    tracking_dir = dataset.tracking
    auto_overlays_dir = tracking_dir / "auto_overlays"

    row, col = parse_well(well)
    well_token = f"{row}{col}"

    iss_ok = None
    pheno_ok = None
    iss_overlap = None
    pheno_overlap = None

    # Check ISS registration metrics (same approach as auto_register_orchestrator.py lines 232-244)
    iss_overlay_dir = auto_overlays_dir / f"{well_token}_iss_to_track"
    iss_metrics_csv = iss_overlay_dir / "auto_register_metrics.csv"

    if iss_metrics_csv.exists():
        try:
            df_metrics = pd.read_csv(iss_metrics_csv, index_col=0)
            if "overlap_forward_overlap_percent" in df_metrics.index:
                overlap_value = df_metrics.loc["overlap_forward_overlap_percent", df_metrics.columns[0]]
                if pd.notna(overlap_value):
                    iss_overlap = float(overlap_value)
                    iss_ok = iss_overlap >= min_overlap
                    if verbose:
                        print(f"  {experiment} ISS {well_token}: overlap={iss_overlap:.1f}% {'✓' if iss_ok else '✗'}")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  Error reading ISS metrics: {e}")

    # Check Pheno registration metrics
    pheno_overlay_dir = auto_overlays_dir / f"{well_token}_pheno_to_track"
    pheno_metrics_csv = pheno_overlay_dir / "auto_register_metrics.csv"

    if pheno_metrics_csv.exists():
        try:
            df_metrics = pd.read_csv(pheno_metrics_csv, index_col=0)
            if "overlap_forward_overlap_percent" in df_metrics.index:
                overlap_value = df_metrics.loc["overlap_forward_overlap_percent", df_metrics.columns[0]]
                if pd.notna(overlap_value):
                    pheno_overlap = float(overlap_value)
                    pheno_ok = pheno_overlap >= min_overlap
                    if verbose:
                        print(f"  {experiment} Pheno {well_token}: overlap={pheno_overlap:.1f}% {'✓' if pheno_ok else '✗'}")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  Error reading Pheno metrics: {e}")

    # both_ok is True if:
    # - Both have metrics AND both pass quality check
    # - OR at least one has no metrics (None) and the other passes (assume missing metrics are OK)
    if iss_ok is None and pheno_ok is None:
        both_ok = True  # No metrics available, proceed with warning
    elif iss_ok is None:
        both_ok = pheno_ok if pheno_ok is not None else True
    elif pheno_ok is None:
        both_ok = iss_ok if iss_ok is not None else True
    else:
        both_ok = iss_ok and pheno_ok

    return {
        "iss_ok": iss_ok,
        "pheno_ok": pheno_ok,
        "both_ok": both_ok,
        "iss_overlap": iss_overlap,
        "pheno_overlap": pheno_overlap,
    }


def detect_experiments_needing_tracking(
    wells: list = [1, 2, 3],
    force: bool = False,
    verbose: bool = True,
) -> tuple[list[tuple[str, int, int, dict]], list[tuple[str, int, int, dict]]]:
    """
    Scan /path/to/ops_data/ to find experiments that need tracking.

    Parameters
    ----------
    wells : list
        Wells to check as full units or numbers (default: [1, 2, 3])
    force : bool
        If True, include all experiments with valid inputs even if outputs exist
    verbose : bool
        Print progress during scan

    Returns
    -------
    tuple[list, list]
        (experiments_to_process, experiments_completed)
        Each list contains tuples of (experiment_name, n_completed, n_expected, extra_data)
    """
    # Define input checker for tracking
    def check_tracking_input(dataset):
        """Check if experiment has all required base inputs for tracking."""
        # Tracking requires multiple data sources
        required_stores = [
            "lc_5x_phase_2d_stitched_v3",  # Tracking phase data
            "lc_5x_segmentation",         # Tracking segmentation
            "pheno_assembled_v3",         # Pheno nuclear_seg label
            "iss_stitch",                 # ISS stitched data
            "iss_segmentation",           # ISS segmentation
        ]

        # Check all required stores exist
        for store_key in required_stores:
            try:
                if not dataset.store_paths[store_key].exists():
                    return False
            except (KeyError, AttributeError):
                return False

        # Also check that registration transforms exist for at least one well
        # (we'll check per-well in the output checker)
        has_any_registration = False
        for well in wells:
            row, col = parse_well(well)
            position = f"{row}/{col}/0"
            try:
                pheno_reg = dataset.append_well("auto_pheno_register", position)
                iss_reg = dataset.append_well("auto_iss_register", position)
                if pheno_reg.exists() and iss_reg.exists():
                    has_any_registration = True
                    break
            except Exception:
                continue

        return has_any_registration

    # Define output checker for tracking (checks per-well outputs)
    def get_tracking_outputs(dataset, wells_list):
        """Get expected tracking output paths for wells with valid registration."""
        outputs = []
        for well in wells_list:
            row, col = parse_well(well)
            position = f"{row}/{col}/0"

            # Check that registration transforms exist for this well
            try:
                pheno_reg = dataset.append_well("auto_pheno_register", position)
                iss_reg = dataset.append_well("auto_iss_register", position)

                # Only include wells that have both registration files
                if pheno_reg.exists() and iss_reg.exists():
                    tracking_output = dataset.append_well("tracking_geff", position)
                    outputs.append(tracking_output)
            except Exception:
                # Skip wells with path errors
                continue

        return outputs

    # Use shared detection utility
    return detect_experiments_needing_processing(
        input_checker=check_tracking_input,
        output_checker=get_tracking_outputs,
        wells=wells,
        force=force,
        verbose=verbose,
    )


def submit_tracking_jobs(
    experiment: str,
    wells: list[str] = None,
    slurm_params: dict = None,
    dry_run: bool = False,
    wait_for_completion: bool = True,
    verbose: bool = True,
    debug: bool = False,
    debug_tile_size: int = 1024,
    crop_coords: tuple[int, int] = (10000, 10000),
    debug_output_suffix: str = "_debug",
    skip_track: bool = False,
    test_time_augs: int = 0,
) -> dict:
    """
    Submit parallel SLURM jobs for cell tracking.

    Parameters
    ----------
    experiment : str
        Experiment name (e.g., "ops0031_20250424")
    wells : list[str]
        Wells to process (default: ["A/1/0", "A/2/0", "A/3/0"])
    slurm_params : dict
        SLURM parameters (timeout_min, mem, cpus_per_task, etc.)
        If None, uses defaults
    dry_run : bool
        If True, print what would be submitted without actually submitting
    wait_for_completion : bool
        If True, wait for all jobs to complete before returning (default: True)
    verbose : bool
        Print detailed progress
    debug : bool
        If True, run in debug mode with ROI cropping
    debug_tile_size : int
        Size of ROI for debug mode (default: 1024)
    crop_coords : tuple[int, int]
        Coordinates for debug ROI cropping (default: (10000, 10000))
    debug_output_suffix : str
        Suffix for debug output files (default: "_debug")
    skip_track : bool
        If True, track using only ISS and pheno timepoints without intermediate tracking data (default: False)

    Returns
    -------
    dict
        Job submission results with job IDs, metadata, and completion status
    """
    # Default wells if not specified
    if wells is None:
        wells = ["A/1/0", "A/2/0", "A/3/0"]

    # Default SLURM parameters
    if slurm_params is None:
        # Tracking needs a GPU because cupy in tile.py / utils.py for graph
        # build does NOT cleanly fall back to CPU when no GPU is present
        # (cudaErrorNoDevice during cupy._core.core._try_skip_h2d_copy).
        # But the GPU is only used for ~13 min of EET inference; the rest
        # is the SCIP solver on CPU + create_track_graph's parallel
        # regionprops (16 workers by default, see track.py OPS_TRACK_*).
        # Allocate enough CPUs to actually feed those workers — previous
        # config of cpus_per_task=1 made the 16-worker pool time-slice
        # on a single core, yielding effectively serial regionprops at
        # ~5-6 min/timepoint per well of ~520k cells. With 32 cores the
        # regionprops phase parallelizes cleanly.
        slurm_params = {
            "timeout_min": 600,
            "mem": "128GB",
            "cpus_per_task": 32,
            "gpus_per_node": 1,
            "slurm_partition": "gpu",
        }

    # Lazy import track_wells here to avoid NumPy conflicts
    from cyclops_process.processes.track.track import track_wells

    # Prepare job list
    jobs_to_submit = []

    for well in wells:
        row, col = parse_well(well)
        position = f"{row}/{col}/0"
        jobs_to_submit.append({
            "name": f"track_{row}{col}",
            "func": track_wells,
            "kwargs": {
                "experiment": experiment,
                "well": position,
                "debug": debug,
                "debug_tile_size": debug_tile_size,
                "crop_coords": crop_coords,
                "debug_output_suffix": debug_output_suffix,
                "skip_track": skip_track,
                "test_time_augs": test_time_augs,
            },
            "metadata": {
                "well": position,
            },
        })

    if not jobs_to_submit:
        print("No jobs to submit!")
        return {}

    # Submit jobs using shared utility
    result = submit_parallel_jobs(
        jobs_to_submit=jobs_to_submit,
        experiment=experiment,
        slurm_params=slurm_params,
        log_dir=f"slurm_tracking_logs/{experiment}",
        step_name="submit_tracking_jobs",
        manifest_prefix="tracking",
        dry_run=dry_run,
        wait_for_completion=wait_for_completion,
        verbose=verbose,
        post_completion_callback=None,  # Could add tracking result aggregation here
    )

    if wait_for_completion and result.get("failed"):
        failed_names = [
            f[0] if isinstance(f, tuple) else f for f in result["failed"]
        ]
        raise RuntimeError(
            f"Tracking failed for {len(failed_names)} job(s): {failed_names}"
        )

    return result


def main():
    """CLI entry point for SLURM batch tracking submission."""
    parser = argparse.ArgumentParser(
        description="Submit cell tracking jobs to SLURM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        help="Experiment name (e.g., ops0031_20250424). Required unless --all is used.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all experiments that need tracking (batch submission)",
    )

    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-run even if outputs exist (use with --all)",
    )

    parser.add_argument(
        "--wells",
        "-w",
        type=str,
        nargs="+",
        default=None,
        help="Wells to process (e.g., A/1/0 A/2/0). Default: all wells (A/1/0, A/2/0, A/3/0)",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="SLURM timeout in minutes (default: 300)",
    )

    parser.add_argument(
        "--mem",
        type=str,
        default="128GB",
        help="SLURM memory allocation (default: 64GB)",
    )

    parser.add_argument(
        "--cpus",
        type=int,
        default=32,
        help="SLURM CPUs per task (default: 32). Feeds both the parallel "
        "regionprops workers and Gurobi's multi-threaded ILP solver.",
    )

    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="SLURM GPUs per node (default: 1)",
    )

    parser.add_argument(
        "--partition",
        type=str,
        default="gpu",
        help="SLURM partition (default: gpu)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with ROI cropping",
    )

    parser.add_argument(
        "--debug-tile-size",
        type=int,
        default=2048,
        help="Size of ROI for debug mode (default: 2048)",
    )

    parser.add_argument(
        "--crop-coords",
        type=int,
        nargs=2,
        default=(10000, 10000),
        help="Coordinates for debug ROI cropping (default: 10000 10000)",
    )

    parser.add_argument(
        "--debug-output-suffix",
        type=str,
        default="_debug",
        help="Suffix for debug output files (default: _debug)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be submitted without actually submitting",
    )

    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt and submit immediately (use with --all)",
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
        "--test-time-augs",
        type=int,
        default=0,
        help="Number of test-time augmentations to average over (default: 0, disabled)",
    )

    args = parser.parse_args()

    # Validation
    if not args.all and not args.experiment:
        parser.error("--experiment is required unless --all is used")

    # Build SLURM parameters
    slurm_params = {
        "timeout_min": args.timeout,
        "mem": args.mem,
        "cpus_per_task": args.cpus,
        "gpus_per_node": args.gpus,
        "slurm_partition": args.partition,
    }

    # Single experiment mode - resolve experiment name with partial matching
    if args.experiment:
        from cyclops_utils.data.filesystem import resolve_experiment_name

        # Resolve partial experiment names with interactive selection
        resolved_name = resolve_experiment_name(
            args.experiment,
            allow_interactive=True
        )

        if resolved_name is None:
            print("No experiment selected or found. Exiting.")
            sys.exit(1)

        # Update args with resolved name
        args.experiment = resolved_name

    # Handle --all mode
    if args.all:
        # Detect experiments needing tracking. Carry full row/col units so
        # wells in different rows (A1 vs B1) never collide on the bare column.
        wells_to_check = [
            f"{parse_well(w)[0]}/{parse_well(w)[1]}/0"
            for w in (args.wells or ["A/1/0", "A/2/0", "A/3/0"])
        ]
        experiments_to_process, experiments_completed = detect_experiments_needing_tracking(
            wells=wells_to_check,
            force=args.force,
            verbose=not args.quiet,
        )

        if not experiments_to_process:
            print("\n✓ All experiments are complete! No tracking jobs needed.\n")
            if not args.quiet and experiments_completed:
                print(f"Completed experiments ({len(experiments_completed)}):")
                for exp, n_done, n_total, _ in experiments_completed:
                    print(f"  ✓ {exp}: {n_done}/{n_total} wells tracked")
            sys.exit(0)

        # Print summary
        print(f"\n{'='*60}")
        print(f"Batch Tracking Submission: {len(experiments_to_process)} experiments")
        print(f"{'='*60}\n")

        for exp, n_done, n_total, _ in experiments_to_process:
            status = f"{n_done}/{n_total} complete"
            print(f"  • {exp}: {status}")

        if experiments_completed and not args.quiet:
            print(f"\nAlready completed ({len(experiments_completed)}):")
            for exp, n_done, n_total, _ in experiments_completed:  # Show all
                print(f"  ✓ {exp}: {n_done}/{n_total} wells")

        print(f"\n{'='*60}")
        print("Note: Only experiments with all required inputs are shown:")
        print("  - Tracking phase & segmentation (5x)")
        print("  - Phenotyping data & segmentation (20x)")
        print("  - ISS stitched & segmentation")
        print("  - Registration transforms (pheno & ISS to tracking)")
        print(f"{'='*60}\n")

        # Build job list across all experiments and wells
        # First pass: check quality for all experiments

        # Lazy import track_wells here to avoid NumPy conflicts
        from cyclops_process.processes.track.track import track_wells

        all_jobs = []
        skipped_experiments = {}  # Track experiments skipped due to low overlap
        experiments_with_missing_metrics = {}  # Track experiments with missing metrics but still included

        for experiment, n_done, n_total, _ in experiments_to_process:
            dataset = OpsDataset(experiment)

            # Determine which wells to process (full row/col units).
            wells_list = [
                f"{parse_well(w)[0]}/{parse_well(w)[1]}/0"
                for w in (args.wells or ["A/1/0", "A/2/0", "A/3/0"])
            ]

            # Check registration quality for ALL wells in this experiment first
            experiment_has_low_quality = False
            low_quality_wells = []
            missing_metrics_wells = []

            for well in wells_list:
                # Check that registration transforms exist for this well
                try:
                    pheno_reg = dataset.append_well("auto_pheno_register", well)
                    iss_reg = dataset.append_well("auto_iss_register", well)

                    # Only check quality if both registration files exist
                    if not (pheno_reg.exists() and iss_reg.exists()):
                        continue

                except Exception:
                    # Skip wells with path errors
                    continue

                # Check registration quality (overlap >= 9%)
                quality = check_registration_quality(
                    experiment=experiment,
                    well=well,
                    min_overlap=9.0,
                    verbose=False,
                )

                # Distinguish between missing metrics and actual low quality
                has_missing_metrics = quality["iss_ok"] is None or quality["pheno_ok"] is None
                has_actual_low_quality = (quality["iss_ok"] is False) or (quality["pheno_ok"] is False)

                if has_actual_low_quality:
                    # Genuine low quality - skip this experiment
                    experiment_has_low_quality = True
                    low_quality_wells.append((well, quality))
                elif has_missing_metrics:
                    # Missing metrics but registration files exist - include with warning
                    missing_metrics_wells.append((well, quality))

            # Skip entire experiment if ANY well has actual low quality
            if experiment_has_low_quality:
                skipped_experiments[experiment] = low_quality_wells
                continue

            # Track experiments with missing metrics (but still proceeding)
            if missing_metrics_wells:
                experiments_with_missing_metrics[experiment] = missing_metrics_wells

            # All wells passed quality check - add jobs for this experiment
            for well in wells_list:
                # Check that registration transforms exist for this well before adding job
                try:
                    pheno_reg = dataset.append_well("auto_pheno_register", well)
                    iss_reg = dataset.append_well("auto_iss_register", well)

                    # Only add jobs for wells with both registration files
                    if not (pheno_reg.exists() and iss_reg.exists()):
                        continue

                    # Skip wells that are already complete (unless --force is used)
                    if not args.force:
                        tracking_output = dataset.append_well("tracking_geff", well)
                        if tracking_output.exists():
                            continue

                except Exception:
                    # Skip wells with path errors
                    continue

                row, col = parse_well(well)
                all_jobs.append({
                    "name": f"{experiment}_{row}{col}",
                    "func": track_wells,
                    "kwargs": {
                        "experiment": experiment,
                        "well": well,
                        "debug": args.debug,
                        "debug_tile_size": args.debug_tile_size,
                        "crop_coords": tuple(args.crop_coords),
                        "debug_output_suffix": args.debug_output_suffix,
                        "skip_track": False,  # Could be enhanced to read from config
                    },
                    "metadata": {
                        "experiment": experiment,
                        "well": well,
                    },
                })

        # Show warning for experiments with missing metrics (but still proceeding)
        if experiments_with_missing_metrics:
            print(f"{'='*60}")
            print(f"⚠️  WARNING: {len(experiments_with_missing_metrics)} experiments with missing overlap metrics")
            print(f"{'='*60}")
            print("These experiments will be INCLUDED (registration files exist):\n")

            for exp, well_list in sorted(experiments_with_missing_metrics.items()):
                print(f"  {exp}:")
                for well, quality in well_list:
                    # Use indicator format: 🟢 = pass, 🔴 = fail, ⚠️ = no metrics
                    def get_indicator(ok_status, overlap):
                        if ok_status is None:
                            return "⚠️  (no metrics)"
                        elif ok_status:
                            return f"🟢 ({overlap:.1f}%)"
                        else:
                            return f"🔴 ({overlap:.1f}%)"

                    iss_indicator = get_indicator(quality["iss_ok"], quality["iss_overlap"])
                    pheno_indicator = get_indicator(quality["pheno_ok"], quality["pheno_overlap"])
                    print(f"    {well}: ISS {iss_indicator}, Pheno {pheno_indicator}")
            print()

        # Show warning for skipped experiments with low quality
        if skipped_experiments:
            print(f"{'='*60}")
            print(f"⚠️  SKIPPED: {len(skipped_experiments)} experiments with low registration quality (<9.0% overlap)")
            print(f"{'='*60}\n")

            for exp, well_list in sorted(skipped_experiments.items()):
                print(f"  {exp}:")
                for well, quality in well_list:
                    # Use indicator format: 🟢 = pass, 🔴 = fail, ⚠️ = no metrics
                    def get_indicator(ok_status, overlap):
                        if ok_status is None:
                            return "⚠️  (no metrics)"
                        elif ok_status:
                            return f"🟢 ({overlap:.1f}%)"
                        else:
                            return f"🔴 ({overlap:.1f}%)"

                    iss_indicator = get_indicator(quality["iss_ok"], quality["iss_overlap"])
                    pheno_indicator = get_indicator(quality["pheno_ok"], quality["pheno_overlap"])
                    print(f"    {well}: ISS {iss_indicator}, Pheno {pheno_indicator}")
            print()

        # Show detailed job plan
        print(f"{'='*60}")
        if args.dry_run:
            print(f"DRY RUN: Job Submission Plan")
        else:
            print(f"Job Submission Plan")
        print(f"{'='*60}\n")
        print(f"Total jobs to submit: {len(all_jobs)}")
        print(f"SLURM Resources (per job):")
        print(f"  Timeout: {slurm_params['timeout_min']} min")
        print(f"  Memory: {slurm_params['mem']}")
        print(f"  CPUs: {slurm_params['cpus_per_task']}")
        print(f"  GPUs: {slurm_params['gpus_per_node']}")
        print(f"  Partition: {slurm_params['slurm_partition']}")
        print(f"\nJobs by experiment:")

        # Group jobs by experiment for cleaner display
        from collections import defaultdict
        jobs_by_exp = defaultdict(list)
        for job in all_jobs:
            exp = job['metadata']['experiment']
            well = job['metadata']['well']
            jobs_by_exp[exp].append(well)

        for exp, wells_in_exp in sorted(jobs_by_exp.items()):
            print(f"  {exp}: {', '.join(wells_in_exp)}")

        print(f"\n{'='*60}\n")

        # Exit if dry run
        if args.dry_run:
            print("DRY RUN: No jobs submitted\n")
            sys.exit(0)

        # Prompt user for confirmation before submitting (unless --yes flag is used)
        if not args.yes:
            try:
                response = input(f"Submit {len(all_jobs)} jobs to SLURM? [y/N]: ").strip().lower()
                if response not in ['y', 'yes']:
                    print("\nCancelled by user. No jobs submitted.\n")
                    sys.exit(0)
            except (KeyboardInterrupt, EOFError):
                print("\n\nCancelled by user. No jobs submitted.\n")
                sys.exit(0)
            print()  # Blank line before submission output
        else:
            print("Proceeding with submission (--yes flag provided)...\n")

        # Submit all jobs as one big batch
        result = submit_parallel_jobs(
            jobs_to_submit=all_jobs,
            experiment=f"batch_tracking_{len(experiments_to_process)}_experiments",
            slurm_params=slurm_params,
            log_dir="slurm_tracking_logs/all",
            manifest_prefix="tracking_batch",
            dry_run=args.dry_run,
            wait_for_completion=not args.no_wait,
            verbose=not args.quiet,
            post_completion_callback=None,
        )

        # Exit based on result
        if result.get("dry_run"):
            sys.exit(0)
        elif result.get("success"):
            if result.get("all_completed") is not None:
                sys.exit(0 if result.get("all_completed") else 1)
            else:
                sys.exit(0)
        else:
            sys.exit(1)

    # Single experiment mode
    else:
        result = submit_tracking_jobs(
            experiment=args.experiment,
            wells=args.wells,
            slurm_params=slurm_params,
            dry_run=args.dry_run,
            wait_for_completion=not args.no_wait,
            verbose=not args.quiet,
            debug=args.debug,
            debug_tile_size=args.debug_tile_size,
            crop_coords=tuple(args.crop_coords),
            debug_output_suffix=args.debug_output_suffix,
            test_time_augs=args.test_time_augs,
        )

        # Exit with appropriate code
        if result.get("dry_run"):
            sys.exit(0)
        elif result.get("success"):
            # If we waited for completion, check if all succeeded
            if result.get("all_completed") is not None:
                sys.exit(0 if result.get("all_completed") else 1)
            else:
                # Didn't wait, assume success
                sys.exit(0)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
