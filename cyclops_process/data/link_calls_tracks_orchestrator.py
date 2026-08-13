"""
SLURM batch submission for ISS-tracking linking (link_calls_tracks).

Submits the linking job via SLURM with proper resource allocation and monitoring.

Usage:
------
# Standard linking
python -m cyclops_process.data.link_calls_tracks_orchestrator --experiment ops0058_20250805

# Link using old (legacy) tracks — writes *_old.csv outputs
python -m cyclops_process.data.link_calls_tracks_orchestrator --experiment ops0058_20250805 --old-tracks

# Old tracks without intensity
python -m cyclops_process.data.link_calls_tracks_orchestrator --experiment ops0058_20250805 --old-tracks --no-intensity

# Specific wells, dry run
python -m cyclops_process.data.link_calls_tracks_orchestrator --experiment ops0058_20250805 --wells A/1/0 A/2/0 --dry-run
"""

import argparse
import sys
import os

sys.path.insert(0, os.getcwd())

from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
from ops_utils.data.experiment import OpsDataset


def submit_link_calls_tracks_job(
    experiment: str,
    wells: list[str] = None,
    slurm_params: dict = None,
    dry_run: bool = False,
    wait_for_completion: bool = True,
    verbose: bool = True,
    n_jobs: int = None,
    confidence_threshold: float = 0.95,
    old_tracks: bool = False,
    no_intensity: bool = False,
    skip_track: bool = False,
    iss_rounds: list[int] = None,
    failed_rounds_by_well: dict = None,
) -> dict:
    """
    Submit link_calls_tracks as a SLURM job.
    """
    if old_tracks or no_intensity:
        raise ValueError("--old-tracks/--no-intensity are no longer supported by link_calls_tracks")

    if wells is None:
        wells = ["A/1/0", "A/2/0", "A/3/0"]

    if slurm_params is None:
        slurm_params = {
            "timeout_min": 45,
            "mem": "250G",
            "cpus_per_task": 12,
            "slurm_partition": "cpu,gpu",
        }

    from cyclops_process.data.datasets import link_calls_tracks

    suffix = ""
    if old_tracks:
        suffix = "_old_no_intensity" if no_intensity else "_old"

    jobs_to_submit = [
        {
            "name": f"link_calls_tracks{suffix}_{experiment}",
            "func": link_calls_tracks,
            "kwargs": {
                "experiment": experiment,
                "wells": wells,
                "confidence_threshold": confidence_threshold,
                "n_jobs": n_jobs,
                "skip_track": skip_track,
                "iss_rounds": iss_rounds,
                "failed_rounds_by_well": failed_rounds_by_well,
            },
            "metadata": {
                "experiment": experiment,
                "wells": wells,
                "old_tracks": old_tracks,
            },
        }
    ]

    result = submit_parallel_jobs(
        jobs_to_submit=jobs_to_submit,
        experiment=experiment,
        slurm_params=slurm_params,
        log_dir=f"slurm_link_calls_tracks_logs/%j",
        manifest_prefix=f"link_calls_tracks{suffix}",
        dry_run=dry_run,
        wait_for_completion=wait_for_completion,
        verbose=verbose,
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Submit link_calls_tracks job to SLURM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--experiment", "-e",
        type=str,
        required=True,
        help="Experiment name (e.g., ops0058_20250805)",
    )
    parser.add_argument(
        "--wells", "-w",
        type=str,
        nargs="+",
        default=None,
        help="Wells to process (default: A/1/0 A/2/0 A/3/0)",
    )
    parser.add_argument(
        "--old-tracks",
        action="store_true",
        help="Use legacy tracking geffs (*_old.geff) and write *_old.csv outputs.",
    )
    parser.add_argument(
        "--no-intensity",
        action="store_true",
        help="With --old-tracks, use *_old_no_intensity.geff files.",
    )
    parser.add_argument(
        "--skip-track",
        action="store_true",
        help="Skip tracking timepoints (ISS + pheno only).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="ISS barcode confidence threshold (default: 0.95)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto-detect)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=45,
        help="SLURM timeout in minutes (default: 45)",
    )
    parser.add_argument(
        "--mem",
        type=str,
        default="250G",
        help="SLURM memory allocation (default: 250G)",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=12,
        help="SLURM CPUs per task (default: 12)",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="cpu,gpu",
        help="SLURM partition (default: cpu,gpu)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be submitted without actually submitting",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit and return immediately without waiting for completion",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce verbosity",
    )

    args = parser.parse_args()

    if args.no_intensity and not args.old_tracks:
        print("Error: --no-intensity requires --old-tracks")
        sys.exit(1)

    from ops_utils.data.filesystem import resolve_experiment_name

    resolved_name = resolve_experiment_name(args.experiment, allow_interactive=True)
    if resolved_name is None:
        print("No experiment selected or found. Exiting.")
        sys.exit(1)

    experiment = resolved_name
    wells = args.wells or ["A/1/0", "A/2/0", "A/3/0"]

    slurm_params = {
        "timeout_min": args.timeout,
        "mem": args.mem,
        "cpus_per_task": args.cpus,
        "slurm_partition": args.partition,
    }

    mode = "standard"
    if args.old_tracks:
        mode = "old (no-intensity)" if args.no_intensity else "old"

    print(f"\n{'='*60}")
    print(f"Link Calls Tracks SLURM Submission")
    print(f"{'='*60}")
    print(f"Experiment: {experiment}")
    print(f"Wells: {wells}")
    print(f"Tracking mode: {mode}")
    print(f"\nSLURM Resources:")
    print(f"  Timeout: {args.timeout} min")
    print(f"  Memory: {args.mem}")
    print(f"  CPUs: {args.cpus}")
    print(f"  Partition: {args.partition}")
    print(f"{'='*60}\n")

    result = submit_link_calls_tracks_job(
        experiment=experiment,
        wells=wells,
        slurm_params=slurm_params,
        dry_run=args.dry_run,
        wait_for_completion=not args.no_wait,
        verbose=not args.quiet,
        n_jobs=args.n_jobs,
        confidence_threshold=args.confidence,
        old_tracks=args.old_tracks,
        no_intensity=args.no_intensity,
        skip_track=args.skip_track,
    )

    if result.get("dry_run"):
        sys.exit(0)
    elif result.get("success"):
        if result.get("all_completed") is not None:
            sys.exit(0 if result.get("all_completed") else 1)
        else:
            sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
