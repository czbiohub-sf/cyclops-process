"""
SLURM batch submission for cell painting linking.

Submits linking jobs for wells 1-3 as parallel SLURM jobs.
Each job runs independently with dedicated resources to maximize throughput.

Usage:
------
# Submit linking for all wells in a single experiment
python -m cyclops_process.data.link_cell_painting_slurm --experiment ops0094_20251217

# Submit linking for specific wells
python -m cyclops_process.data.link_cell_painting_slurm --experiment ops0094 --wells A/1/0 A/2/0

# Dry run (preview what would be submitted)
python -m cyclops_process.data.link_cell_painting_slurm --experiment ops0094 --dry-run

# Submit without waiting for completion
python -m cyclops_process.data.link_cell_painting_slurm --experiment ops0094 --no-wait
"""

import argparse
import sys
from pathlib import Path
import os

# Add project root to path
sys.path.insert(0, os.getcwd())

from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
from ops_utils.data.experiment import OpsDataset


def check_cell_painting_inputs(dataset: OpsDataset, well: str, primary_label: str = "CP1_nuclear_seg") -> dict:
    """Check if linking inputs exist for a well (primary nuclear seg + pheno + ISS)."""
    v3_path = dataset.store_paths.get("pheno_assembled_v3")
    iss_seg_path = dataset.store_paths.get("iss_segmentation")

    primary_key = primary_label.lower()  # e.g. "cp1_nuclear_seg" or "4i_r1_nuclear_seg"
    status = {
        "pheno_assembled_v3": v3_path is not None and v3_path.exists(),
        "iss_segmentation": iss_seg_path is not None and iss_seg_path.exists(),
        primary_key: False,
        "pheno_nuclear_seg": False,
    }

    if status["pheno_assembled_v3"]:
        import zarr
        try:
            store = zarr.open(str(v3_path), mode="r")
            for level in [0, 2]:
                if f"{well}/labels/{primary_label}/{level}" in store:
                    status[primary_key] = True
                    break
            for label_name in ["nuclear_seg", "seg"]:
                for level in [0, 2]:
                    if f"{well}/labels/{label_name}/{level}" in store:
                        status["pheno_nuclear_seg"] = True
                        break
                if status["pheno_nuclear_seg"]:
                    break
        except Exception:
            pass

    return status


def check_centroids_cached(dataset: OpsDataset, well: str) -> bool:
    """
    Check if centroid caches exist for a well.

    If all centroid parquet files exist, the job can run much faster
    (just linking, no centroid extraction needed).

    Returns:
        True if all required centroid caches exist
    """
    well_safe = well.replace("/", "_")
    cache_dir = dataset.results_fast / "cp_links"

    if not cache_dir.exists():
        return False

    # Check for the main centroid caches (CP1 required, others optional)
    cp1_cache = cache_dir / f"centroids_cp1_{well_safe}.parquet"
    if not cp1_cache.exists():
        return False

    # Check for at least one other modality cache
    cp2_cache = cache_dir / f"centroids_cp2_{well_safe}.parquet"
    pheno_cache = cache_dir / f"centroids_pheno_{well_safe}.parquet"
    iss_cache = cache_dir / f"centroids_iss_{well_safe}.parquet"

    # If CP1 exists and at least one other exists, consider it cached
    has_other = cp2_cache.exists() or pheno_cache.exists() or iss_cache.exists()

    return has_other


def submit_cell_painting_linking_jobs(
    experiment: str,
    wells: list[str] = None,
    slurm_params: dict = None,
    dry_run: bool = False,
    wait_for_completion: bool = True,
    verbose: bool = True,
    n_jobs: int = None,
    confidence_threshold: float = 0.95,
    force: bool = False,
    mode: str = "cp",
) -> dict:
    """
    Submit parallel SLURM jobs for cell painting linking.

    Parameters
    ----------
    experiment : str
        Experiment name (e.g., "ops0094_20251217")
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
    n_jobs : int
        Number of parallel workers for chunk processing within each job
    confidence_threshold : float
        ISS barcode confidence threshold (default: 0.95)
    force : bool
        If True, overwrite existing outputs (default: False)

    Returns
    -------
    dict
        Job submission results with job IDs, metadata, and completion status
    """
    # Default wells if not specified
    if wells is None:
        wells = ["A/1/0", "A/2/0", "A/3/0"]

    # Import the linking function
    from cyclops_process.fixed_cp_4i.link import _process_single_well, MODE_CONFIG
    mcfg = MODE_CONFIG[mode]

    # Check inputs and prepare job list
    dataset = OpsDataset(experiment)
    jobs_to_submit = []
    skipped_wells = []

    # Check if all wells have cached centroids - if so, use reduced timeout
    all_cached = all(check_centroids_cached(dataset, well) for well in wells)
    if all_cached:
        default_timeout = 10  # 10 min is plenty when centroids are cached
        if verbose:
            print("  Centroids cached for all wells - using reduced timeout (10 min)")
    else:
        default_timeout = 60  # Full 60 min when extracting centroids

    # Default SLURM parameters - cell painting linking is CPU-intensive
    if slurm_params is None:
        slurm_params = {
            "timeout_min": default_timeout,
            "mem": "128GB",
            "cpus_per_task": 64,
            "slurm_partition": "cpu",
        }
    elif "timeout_min" not in slurm_params:
        slurm_params["timeout_min"] = default_timeout

    primary_label = mcfg["primary_label"]
    primary_key = primary_label.lower()
    for well in wells:
        # Check inputs
        input_status = check_cell_painting_inputs(dataset, well, primary_label=primary_label)

        if not input_status[primary_key]:
            skipped_wells.append((well, f"Missing {primary_label}"))
            continue

        # Check if output already exists
        _parts = well.split("/")
        output_path = dataset.results_fast / mcfg["output_template"].format(
            well_safe=well.replace("/", "_"),
            well_short=f"{_parts[0]}{_parts[1]}",
        )
        if output_path.exists() and not dry_run and not force:
            if verbose:
                print(f"  Skipping {well}: output already exists at {output_path}")
            continue

        jobs_to_submit.append({
            "name": f"{mode}_link_{well.replace('/', '_')}",
            "func": _process_single_well,
            "kwargs": {
                "experiment": experiment,
                "well": well,
                "verbose": True,
                "n_jobs": n_jobs,
                "mode": mode,
            },
            "metadata": {
                "well": well,
                "mode": mode,
            },
        })

    # Print skipped wells
    if skipped_wells and verbose:
        print(f"\nSkipped wells (missing inputs):")
        for well, reason in skipped_wells:
            print(f"  {well}: {reason}")

    if not jobs_to_submit:
        print("\nNo jobs to submit!")
        return {"success": True, "jobs": []}

    # Submit jobs using shared utility
    result = submit_parallel_jobs(
        jobs_to_submit=jobs_to_submit,
        experiment=experiment,
        slurm_params=slurm_params,
        log_dir="cell_painting",
        manifest_prefix="cell_painting_linking",
        dry_run=dry_run,
        wait_for_completion=wait_for_completion,
        verbose=verbose,
        post_completion_callback=None,
    )

    return result


def main():
    """CLI entry point for SLURM batch cell painting linking submission."""
    # submit_parallel_jobs writes logs to "slurm_logs/<log_dir>" relative to CWD.
    # Anchor to project root so the printed `View logs:` path is always correct
    # regardless of where the script was invoked from.
    project_root = Path(__file__).resolve().parents[3]
    os.chdir(project_root)

    parser = argparse.ArgumentParser(
        description="Submit cell painting linking jobs to SLURM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--experiment", "-e",
        type=str,
        required=True,
        help="Experiment name (e.g., ops0094_20251217 or ops0094)",
    )

    parser.add_argument(
        "--wells", "-w",
        type=str,
        nargs="+",
        default=None,
        help="Wells to process (e.g., A/1/0 A/2/0). Default: all wells (A/1/0, A/2/0, A/3/0)",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="SLURM timeout in minutes (default: auto - 10 min if cached, 60 min otherwise)",
    )

    parser.add_argument(
        "--mem",
        type=str,
        default="128GB",
        help="SLURM memory allocation (default: 128GB)",
    )

    parser.add_argument(
        "--cpus",
        type=int,
        default=64,
        help="SLURM CPUs per task (default: 64)",
    )

    parser.add_argument(
        "--partition",
        type=str,
        default="cpu",
        help="SLURM partition (default: cpu)",
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel workers for chunk processing within each job (default: auto-detect)",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="ISS barcode confidence threshold (default: 0.95)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be submitted without actually submitting",
    )

    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit jobs and return immediately without waiting for completion",
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce verbosity",
    )

    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-run even if outputs exist",
    )

    parser.add_argument(
        "--4i", dest="four_i", action="store_true",
        help="4i mode: link R1_nuclear_seg -> R2..R5_nuclear_seg + pheno + ISS, "
             "using 4i_cell_seg for bboxes. Output: four_i_linked_<well>.csv",
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Clear centroid caches before running (forces full recomputation)",
    )

    args = parser.parse_args()

    # Resolve experiment name with partial matching
    from ops_utils.data.filesystem import resolve_experiment_name

    resolved_name = resolve_experiment_name(
        args.experiment,
        allow_interactive=True
    )

    if resolved_name is None:
        print("No experiment selected or found. Exiting.")
        sys.exit(1)

    experiment = resolved_name

    # Print job plan
    wells = args.wells or ["A/1/0", "A/2/0", "A/3/0"]

    # Initialize dataset
    dataset = OpsDataset(experiment)

    # Clear cache if --no-cache is specified
    if args.no_cache:
        cache_dir = dataset.results_fast / "cp_links"
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            print(f"Cleared centroid cache: {cache_dir}")
        else:
            print("No centroid cache to clear")

    # Check cache status to determine timeout
    all_cached = all(check_centroids_cached(dataset, well) for well in wells)

    # Determine timeout: user-specified > auto-detect based on cache
    if args.timeout is not None:
        timeout_min = args.timeout
        timeout_source = "user-specified"
    elif all_cached:
        timeout_min = 10
        timeout_source = "auto (centroids cached)"
    else:
        timeout_min = 60
        timeout_source = "auto (need centroid extraction)"

    # Build SLURM parameters
    slurm_params = {
        "timeout_min": timeout_min,
        "mem": args.mem,
        "cpus_per_task": args.cpus,
        "slurm_partition": args.partition,
    }

    print(f"\n{'='*60}")
    print(f"Cell Painting Linking SLURM Submission")
    print(f"{'='*60}")
    print(f"Experiment: {experiment}")
    print(f"Wells: {wells}")
    print(f"\nSLURM Resources (per job):")
    print(f"  Timeout: {timeout_min} min ({timeout_source})")
    print(f"  Memory: {slurm_params['mem']}")
    print(f"  CPUs: {slurm_params['cpus_per_task']}")
    print(f"  Partition: {slurm_params['slurm_partition']}")
    print(f"\nOptions:")
    print(f"  Confidence threshold: {args.confidence}")
    print(f"  Chunk workers: {args.n_jobs or 'auto-detect'}")
    print(f"{'='*60}\n")

    # Check inputs for each well (dataset already loaded above)
    print("Checking inputs...")
    from cyclops_process.fixed_cp_4i.link import MODE_CONFIG as _MC_PRE
    _primary_label = _MC_PRE["4i" if args.four_i else "cp"]["primary_label"]
    for well in wells:
        status = check_cell_painting_inputs(dataset, well, primary_label=_primary_label)
        status_str = " ".join([
            f"{'✓' if v else '✗'} {k}"
            for k, v in status.items()
        ])
        print(f"  {well}: {status_str}")

        # Check output
        from cyclops_process.fixed_cp_4i.link import MODE_CONFIG as _MC
        _mode = "4i" if args.four_i else "cp"
        _parts = well.split("/")
        output_path = dataset.results_fast / _MC[_mode]["output_template"].format(
            well_safe=well.replace("/", "_"),
            well_short=f"{_parts[0]}{_parts[1]}",
        )
        if output_path.exists():
            if args.force:
                print(f"    Output exists (will overwrite with --force)")
            else:
                print(f"    Output exists: {output_path.name}")
    print()

    # Submit jobs
    mode = "4i" if args.four_i else "cp"
    result = submit_cell_painting_linking_jobs(
        experiment=experiment,
        wells=args.wells,
        slurm_params=slurm_params,
        dry_run=args.dry_run,
        wait_for_completion=not args.no_wait,
        verbose=not args.quiet,
        n_jobs=args.n_jobs,
        confidence_threshold=args.confidence,
        force=args.force,
        mode=mode,
    )

    # Exit with appropriate code
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
