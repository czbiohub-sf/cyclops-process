"""
SLURM batch submission for ISS failed rounds optimization.

Submits optimization jobs for all experiment-well combinations as parallel SLURM jobs.
Each job runs independently with dedicated resources to maximize throughput.

Usage:
------
# Process ALL experiments and wells (batch mode)
python -m cyclops_process.metrics.plate_stats.optimize_failed_rounds_orchestrator --all

# Force reprocess all experiments even if outputs exist
python -m cyclops_process.metrics.plate_stats.optimize_failed_rounds_orchestrator --all --force

# Preview what would be submitted (dry run)
python -m cyclops_process.metrics.plate_stats.optimize_failed_rounds_orchestrator --all --dry-run

# Submit without waiting for completion
python -m cyclops_process.metrics.plate_stats.optimize_failed_rounds_orchestrator --all --no-wait

# Single experiment mode
python -m cyclops_process.metrics.plate_stats.optimize_failed_rounds_orchestrator --experiment ops0108_20251209

# Single experiment, specific wells
python -m cyclops_process.metrics.plate_stats.optimize_failed_rounds_orchestrator --experiment ops0108_20251209 --wells "A/1/0" "A/2/0"

# aggregate results
python -m cyclops_process.metrics.plate_stats.optimize_failed_rounds_orchestrator --aggregate

# aggregate results for a specific job ID
python -m cyclops_process.metrics.plate_stats.optimize_failed_rounds_orchestrator --aggregate --job-id 123456

# aggregate results for a specific experiment
python -m cyclops_process.metrics.plate_stats.optimize_failed_rounds_orchestrator --aggregate --experiment ops0108_20251209

"""

import argparse
import sys
from pathlib import Path
import os

# Add project root to path
sys.path.insert(0, os.getcwd())

from cyclops_process.metrics.plate_stats.optimize_failed_rounds import (
    optimize_well,
    scan_ops_experiments,
    DEFAULT_OPS_DIR,
    CACHE_DIR,
)
from ops_utils.hpc.slurm_batch_utils import (
    submit_parallel_jobs,
)
from ops_utils.data.experiment import OpsDataset


def _has_non_default_library(experiment: str) -> bool:
    """Check if experiment uses a non-default codebook via ops_library_map.yaml.

    Experiments with custom (non-default) libraries should skip
    reference correlation since the global reference was built from the
    default library.
    """
    import re

    from ops_utils.data.bad_experiments import load_library_map

    lib_map = load_library_map()

    default_codebook = lib_map.get("default", {}).get("codebook", "")
    overrides = lib_map.get("overrides", {})

    # Extract ops key (e.g., "ops0149") from full experiment name
    match = re.search(r"ops(\d{4})", experiment, re.IGNORECASE)
    if not match:
        return False
    ops_key = f"ops{match.group(1)}".lower()

    exp_override = overrides.get(ops_key, {})
    codebook = exp_override.get("codebook", "")
    return codebook != "" and codebook != default_codebook


def get_experiment_wells(experiment: str) -> list[str]:
    """Get all wells for an experiment using OpsDataset.infer_wells()."""
    try:
        dataset = OpsDataset(experiment, method="mine")
        # infer_wells() returns wells in "A/1" format
        # Add default position "/0" to get "A/1/0" format expected by optimize_failed_rounds
        wells = dataset.infer_wells()
        return sorted([f"{w}/0" for w in wells])
    except Exception:
        return []


def optimize_single_well_job(
    experiment: str,
    well: str,
    max_dropouts: int = 3,
    max_shifts: int = 2,
    min_effective_rounds: int = 7,
    improvement_threshold: float = 1.0,
    use_ops_median: bool = True,
    rebuild_cache: bool = False,
    ops_dir: str = DEFAULT_OPS_DIR,
) -> dict:
    """
    Optimize a single experiment-well combination.

    This is the function that runs inside each SLURM job.
    Returns a dict with results that can be aggregated later.
    """
    import pandas as pd
    import yaml
    from cyclops_process.metrics.plate_stats.optimize_failed_rounds import (
        compute_entropy_stats,
        _get_shift_round_mapping,
        validate_matches,
        get_or_build_reference_cache,
        CACHE_DIR,
    )
    from cyclops_process.metrics.plate_stats.match_reads import match_reads

    try:
        dataset = OpsDataset(experiment, method="mine")
        codebook_db = dataset.load_codebook()

        # ISS reads at most 10 rounds even if codebook barcodes are longer.
        barcode_col = "sgRNA" if "sgRNA" in codebook_db.columns else "barcode"
        n_iss_rounds = min(10, int(codebook_db[barcode_col].str.len().max()))
        constant_rounds = []
        iss_rounds = list(range(n_iss_rounds))

        # Load gene_index for gene name lookups
        gene_index_db = None
        try:
            gene_index_db = dataset.load_gene_index()
        except Exception:
            pass

        reads_df = pd.read_csv(dataset.append_well("reads", well))

        # Run optimization
        results_df = optimize_well(
            experiment, well,
            max_dropouts=max_dropouts,
            max_shifts=max_shifts,
            min_effective_rounds=min_effective_rounds,
            improvement_threshold=improvement_threshold,
            verbose=True,  # Enable full logging to SLURM log files
            use_ops_median=use_ops_median,
            rebuild_cache=rebuild_cache,
            ops_dir=ops_dir,
        )

        # Get baseline and best config
        baseline_row = results_df[results_df["config_type"] == "baseline"]
        valid_df = results_df[results_df["is_valid"]]

        if len(baseline_row) == 0:
            return {
                "success": False,
                "experiment": experiment,
                "well": well,
                "error": "No baseline configuration found",
            }

        baseline = baseline_row.iloc[0]

        # If no valid configs, use baseline as best (this is expected when baseline is best)
        if len(valid_df) == 0:
            # Use baseline as the "best" config
            best = baseline
        else:
            best = valid_df.iloc[0]

        # Calculate stats
        baseline_rate = baseline["match_rate"]
        best_rate = best["match_rate"]
        absolute_improvement = best_rate - baseline_rate
        relative_improvement = best_rate / baseline_rate if baseline_rate > 0 else 0

        # Get current config from ops_failed_rounds.yaml
        current_yaml_path = Path(__file__).parent.parent.parent / "configs" / "ops_failed_rounds.yaml"
        current_config_str = "baseline"
        current_match_rate = baseline_rate
        current_correlation = baseline["correlation"]
        current_slope = baseline["entropy_slope"]
        current_pvalue = baseline["entropy_pvalue"]
        current_top_gene_vs_median = baseline["top_gene_vs_median"]
        current_top_guide_vs_median = baseline["top_guide_vs_median"]
        current_dropouts = []
        current_shifts = []

        if current_yaml_path.exists():
            with open(current_yaml_path) as f:
                current_configs = yaml.safe_load(f) or {}

            if experiment in current_configs:
                exp_config = current_configs[experiment]
                if "failed_rounds_by_well" in exp_config and well in exp_config["failed_rounds_by_well"]:
                    well_config = exp_config["failed_rounds_by_well"][well]

                    # Parse the config format
                    if isinstance(well_config, list):
                        current_dropouts = well_config
                    elif isinstance(well_config, dict):
                        current_dropouts = well_config.get("dropout", [])
                        current_shifts = well_config.get("shift", [])

                    # Build config string
                    if current_dropouts and current_shifts:
                        current_config_str = f"dropout {current_dropouts} + shift {current_shifts}"
                    elif current_shifts:
                        current_config_str = f"shift {current_shifts}"
                    elif current_dropouts:
                        current_config_str = f"dropout {current_dropouts}"

                    # Test current config to get match rate, correlation, and slope
                    if current_dropouts or current_shifts:
                        current_failed_config = {well: {}}
                        if current_dropouts:
                            current_failed_config[well]["dropout"] = current_dropouts
                        if current_shifts:
                            current_failed_config[well]["shift"] = current_shifts

                        current_matched = match_reads(
                            reads_df.copy(), codebook_db,
                            iss_rounds=iss_rounds, well_name=well,
                            failed_rounds_by_well=current_failed_config,
                        )
                        cells_with_reads = reads_df["cell"].nunique() if "cell" in reads_df.columns else len(reads_df)
                        cells_with_current_matched = current_matched["cell"][current_matched["cell"] > 0].nunique() if len(current_matched) > 0 and "cell" in current_matched.columns else 0
                        current_match_rate = (cells_with_current_matched / cells_with_reads * 100) if cells_with_reads > 0 else 0

                        # Get position mapping for current config
                        if current_shifts:
                            read_positions, codebook_positions = _get_shift_round_mapping(
                                iss_rounds, well, current_failed_config
                            )
                        else:
                            effective_rounds = [r for r in iss_rounds if r not in current_dropouts]
                            read_positions = effective_rounds
                            codebook_positions = effective_rounds

                        # Load reference cache for validation (only if using ops-median)
                        total_guides = len(codebook_db)
                        reference_freq = None
                        if use_ops_median:
                            reference_freq, _ = get_or_build_reference_cache(
                                ops_dir, rebuild=False
                            )

                        # Validate current config to get correlation, slope, pvalue, and gene/guide skew
                        (_, _, _, _, _, _, _, _, _, _, current_correlation, current_slope,
                         current_top_gene_vs_median, current_top_guide_vs_median, _, current_pvalue) = validate_matches(
                            current_matched, codebook_db, read_positions, codebook_positions, total_guides,
                            use_ops_median=use_ops_median,
                            reference_freq=reference_freq,
                            baseline_correlation=baseline["correlation"],
                            baseline_top_guide_vs_median=0.0,
                            baseline_entropy_slope=baseline["entropy_slope"],
                            baseline_entropy_pvalue=baseline["entropy_pvalue"],
                        )

        improvement_vs_current = best_rate - current_match_rate

        # Collect results
        return {
            "success": True,
            "experiment": experiment,
            "well": well,
            "baseline_match_rate": baseline_rate,
            "current_config_str": current_config_str,
            "current_match_rate": current_match_rate,
            "best_match_rate": best_rate,
            "absolute_improvement": absolute_improvement,
            "relative_improvement": relative_improvement,
            "improvement_vs_current": improvement_vs_current,
            "config_str": best["config_str"],
            "dropouts": sorted(set(best["dropouts"] + constant_rounds)),
            "shifts": best["shifts"],
            "current_dropouts": current_dropouts,
            "current_shifts": current_shifts,
            "baseline_correlation": baseline["correlation"],
            "current_correlation": current_correlation,
            "best_correlation": best["correlation"],
            "correlation_vs_baseline": best["correlation"] - baseline["correlation"],
            "correlation_vs_current": best["correlation"] - current_correlation,
            "baseline_entropy": baseline["avg_entropy"],
            "best_entropy": best["avg_entropy"],
            "entropy_improvement": best["avg_entropy"] - baseline["avg_entropy"],
            "baseline_slope": baseline["entropy_slope"],
            "current_slope": current_slope,
            "best_slope": best["entropy_slope"],
            "slope_vs_baseline": best["entropy_slope"] - baseline["entropy_slope"],
            "slope_vs_current": best["entropy_slope"] - current_slope,
            "baseline_pvalue": baseline["entropy_pvalue"],
            "current_pvalue": current_pvalue,
            "best_pvalue": best["entropy_pvalue"],
            "baseline_top_gene_vs_median": baseline["top_gene_vs_median"],
            "current_top_gene_vs_median": current_top_gene_vs_median,
            "best_top_gene_vs_median": best["top_gene_vs_median"],
            "top_gene_vs_median_vs_baseline": best["top_gene_vs_median"] - baseline["top_gene_vs_median"],
            "top_gene_vs_median_vs_current": best["top_gene_vs_median"] - current_top_gene_vs_median,
            "baseline_top_guide_vs_median": baseline["top_guide_vs_median"],
            "current_top_guide_vs_median": current_top_guide_vs_median,
            "best_top_guide_vs_median": best["top_guide_vs_median"],
            "top_guide_vs_median_vs_baseline": best["top_guide_vs_median"] - baseline["top_guide_vs_median"],
            "top_guide_vs_median_vs_current": best["top_guide_vs_median"] - current_top_guide_vs_median,
            "unique_genes": best["unique_genes"],
            "effective_rounds": best["effective_rounds"],
        }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "experiment": experiment,
            "well": well,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def submit_optimization_jobs(
    experiments: list[str] = None,
    wells: list[str] = None,
    max_dropouts: int = 2,
    max_shifts: int = 2,
    min_effective_rounds: int = 7,
    improvement_threshold: float = 1.0,
    rebuild_cache: bool = False,
    ops_dir: str = DEFAULT_OPS_DIR,
    slurm_params: dict = None,
    dry_run: bool = False,
    wait_for_completion: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Submit parallel SLURM jobs for failed rounds optimization.

    Parameters
    ----------
    experiments : list[str]
        Experiments to process (default: all experiments)
    wells : list[str]
        Wells to process (default: all wells per experiment)
    slurm_params : dict
        SLURM parameters (timeout_min, mem, cpus_per_task, etc.)
    dry_run : bool
        If True, print what would be submitted without actually submitting
    wait_for_completion : bool
        If True, wait for all jobs to complete before returning
    verbose : bool
        Print detailed progress

    Returns
    -------
    dict
        Job submission results with job IDs, metadata, and completion status
    """
    # Get experiments to process
    if experiments is None:
        experiments = scan_ops_experiments(ops_dir)

    if verbose:
        print(f"Found {len(experiments)} experiments to process")

    # Default SLURM parameters for CPU-bound optimization
    default_slurm_params = {
        "timeout_min": 5,  # ~2 min average, 5 min buffer
        "mem": "64GB",
        "cpus_per_task": 16,
        "gpus_per_node": 0,
        "slurm_partition": "cpu,gpu",
    }

    if slurm_params:
        default_slurm_params.update(slurm_params)

    # Build job list - one job per experiment-well combination
    jobs_to_submit = []
    any_need_reference = False

    for experiment in experiments:
        exp_wells = wells if wells else get_experiment_wells(experiment)
        skip_ref = _has_non_default_library(experiment)
        if skip_ref and verbose:
            print(f"  {experiment}: non-default library detected, skipping reference correlation")
        if not skip_ref:
            any_need_reference = True

        for well in exp_wells:
            well_safe = well.replace("/", "_")
            job_name = f"opt_{experiment}_{well_safe}"

            jobs_to_submit.append({
                "name": job_name,
                "func": optimize_single_well_job,
                "kwargs": {
                    "experiment": experiment,
                    "well": well,
                    "max_dropouts": max_dropouts,
                    "max_shifts": max_shifts,
                    "min_effective_rounds": min_effective_rounds,
                    "improvement_threshold": improvement_threshold,
                    "use_ops_median": not skip_ref,
                    "rebuild_cache": rebuild_cache,
                    "ops_dir": ops_dir,
                },
                "metadata": {
                    "experiment": experiment,
                    "well": well,
                },
                "slurm_params": default_slurm_params,
            })

    if not jobs_to_submit:
        print("No jobs to submit!")
        return {"success": False, "error": "No jobs to submit"}

    # Pre-warm the reference cache so SLURM jobs don't each rebuild it
    # (only needed if at least one experiment uses reference correlation)
    from cyclops_process.metrics.plate_stats.optimize_failed_rounds import get_or_build_reference_cache
    if any_need_reference:
        if verbose:
            print("Pre-warming reference cache (so SLURM jobs can load it instantly)...")
        get_or_build_reference_cache(
            ops_dir=ops_dir, rebuild=rebuild_cache, verbose=verbose
        )
    elif verbose:
        print("Skipping reference cache (all experiments use non-default libraries)")

    # Print summary
    print(f"\n{'='*60}")
    print(f"ISS Failed Rounds Optimization Batch Submission")
    print(f"{'='*60}")
    print(f"Experiments: {len(experiments)}")
    print(f"Total jobs: {len(jobs_to_submit)} (experiment-well combinations)")
    print(f"Resources: {default_slurm_params['mem']}, {default_slurm_params['cpus_per_task']} CPUs, CPU partition")
    print(f"Timeout: {default_slurm_params['timeout_min']} minutes per job")
    print(f"{'='*60}\n")

    # Submit all jobs
    print(f"Submitting {len(jobs_to_submit)} optimization jobs...")
    result = submit_parallel_jobs(
        jobs_to_submit=jobs_to_submit,
        experiment="iss_optimization_batch",
        slurm_params=default_slurm_params,
        log_dir="slurm_iss_optimize/all",
        manifest_prefix="iss_optimization",
        dry_run=dry_run,
        wait_for_completion=False,  # Don't wait yet
        verbose=verbose,
        post_completion_callback=None,
    )

    # If user wants to wait, wait for job array completion
    if wait_for_completion and not dry_run:
        from ops_utils.hpc.slurm_batch_utils import wait_for_multiple_job_arrays

        job_arrays = []
        if result.get("success") and "submitted_jobs" in result:
            job_arrays.append({
                "submitted_jobs": result["submitted_jobs"],
                "base_job_id": result["base_job_id"],
                "label": f"Optimization ({result['base_job_id']})",
                "slurm_params": default_slurm_params,
            })

        if job_arrays:
            wait_result = wait_for_multiple_job_arrays(
                job_arrays=job_arrays,
                experiment="iss_optimization_batch",
                verbose=verbose,
                print_resource_summary=False,  # Skip for large batch jobs
            )

            return {
                "success": True,
                "result": result,
                "completed": wait_result.get("completed", []),
                "failed": wait_result.get("failed", []),
                "all_completed": len(wait_result.get("failed", [])) == 0,
            }

    return {
        "success": result.get("success", False),
        "result": result,
        "dry_run": dry_run,
    }


def aggregate_results(
    log_dir: str = "slurm_logs/slurm_iss_optimize/all",
    job_id: str = None,
) -> None:
    """
    Aggregate results from completed SLURM jobs into summary CSV and YAML.

    Run this after jobs complete to generate the final output files.
    Reads submitit pickle result files from the job directories.

    Parameters
    ----------
    log_dir : str
        Directory containing SLURM job output directories
    job_id : str, optional
        If provided, only aggregate results from this specific job run.
        Filters to directories starting with this job_id prefix.
    """
    import pandas as pd
    import pickle
    from pathlib import Path

    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"Log directory not found: {log_dir}")
        return

    # Find all result pickle files from submitit
    # Pattern: {job_id}_{array_index}/{job_id}_{array_index}_0_result.pkl
    if job_id:
        # Only aggregate results from this specific job run
        result_files = list(log_path.glob(f"{job_id}_*/*_result.pkl"))
        print(f"Filtering to job_id: {job_id}")
    else:
        result_files = list(log_path.glob("**/*_result.pkl"))

    if not result_files:
        print(f"No result files found in {log_dir}")
        print(f"Expected pattern: {log_dir}/<job_id>_<idx>/<job_id>_<idx>_0_result.pkl")
        return

    print(f"Found {len(result_files)} result files to aggregate")

    all_stats = []
    failed_jobs = []  # Track failed jobs with details

    for result_file in result_files:
        # Extract job_id from path like: .../27517794_123/27517794_123_0_result.pkl
        job_dir = result_file.parent.name  # e.g., "27517794_123"
        slurm_job_id = job_dir

        try:
            with open(result_file, "rb") as f:
                # submitit stores results as tuple: ('success', result_dict) or ('error', exception)
                status, result = pickle.load(f)

            if status != "success" or not isinstance(result, dict):
                failed_jobs.append({
                    "slurm_job_id": slurm_job_id,
                    "experiment": "unknown",
                    "well": "unknown",
                    "error": f"status={status}" if status != "success" else "result not dict",
                })
                continue

            if not result.get("success"):
                error_msg = result.get("error", "unknown error")

                # Special case: "No valid configurations found" means baseline is best
                # This is not a real failure - just no improvement possible
                if "No valid configurations found" in error_msg:
                    # Create a "baseline is best" result entry
                    stats = {
                        "experiment": result.get("experiment"),
                        "well": result.get("well"),
                        "baseline_match_rate": 0.0,  # Unknown from old pickles
                        "current_config_str": "baseline",
                        "current_match_rate": 0.0,
                        "best_match_rate": 0.0,
                        "absolute_improvement": 0.0,
                        "relative_improvement": 0.0,
                        "improvement_vs_current": 0.0,
                        "config_str": "baseline",  # No dropouts/shifts - baseline is best
                        "dropouts": "[]",
                        "shifts": "[]",
                        "current_dropouts": "[]",
                        "current_shifts": "[]",
                        "baseline_correlation": 0.0,
                        "current_correlation": 0.0,
                        "best_correlation": 0.0,
                        "correlation_vs_baseline": 0.0,
                        "correlation_vs_current": 0.0,
                        "baseline_entropy": 0.0,
                        "best_entropy": 0.0,
                        "entropy_improvement": 0.0,
                        "baseline_slope": 0.0,
                        "current_slope": 0.0,
                        "best_slope": 0.0,
                        "slope_vs_baseline": 0.0,
                        "slope_vs_current": 0.0,
                        "baseline_pvalue": 1.0,
                        "current_pvalue": 1.0,
                        "best_pvalue": 1.0,
                        "baseline_top_gene_vs_median": 0.0,
                        "current_top_gene_vs_median": 0.0,
                        "best_top_gene_vs_median": 0.0,
                        "top_gene_vs_median_vs_baseline": 0.0,
                        "top_gene_vs_median_vs_current": 0.0,
                        "baseline_top_guide_vs_median": 0.0,
                        "current_top_guide_vs_median": 0.0,
                        "best_top_guide_vs_median": 0.0,
                        "top_guide_vs_median_vs_baseline": 0.0,
                        "top_guide_vs_median_vs_current": 0.0,
                        "unique_genes": 0,
                        "effective_rounds": 10,  # Full rounds used at baseline
                    }
                    all_stats.append(stats)
                    continue

                failed_jobs.append({
                    "slurm_job_id": slurm_job_id,
                    "experiment": result.get("experiment", "unknown"),
                    "well": result.get("well", "unknown"),
                    "error": error_msg[:100],
                })
                continue

            # Convert dropouts/shifts to string for CSV
            stats = {
                "experiment": result.get("experiment"),
                "well": result.get("well"),
                "baseline_match_rate": float(result.get("baseline_match_rate", 0)),
                "current_config_str": result.get("current_config_str"),
                "current_match_rate": float(result.get("current_match_rate", 0)),
                "best_match_rate": float(result.get("best_match_rate", 0)),
                "absolute_improvement": float(result.get("absolute_improvement", 0)),
                "relative_improvement": float(result.get("relative_improvement", 0)),
                "improvement_vs_current": float(result.get("improvement_vs_current", 0)),
                "config_str": result.get("config_str"),
                "dropouts": str(result.get("dropouts", [])),
                "shifts": str(result.get("shifts", [])),
                "current_dropouts": str(result.get("current_dropouts", [])),
                "current_shifts": str(result.get("current_shifts", [])),
                "baseline_correlation": float(result.get("baseline_correlation", 0)),
                "current_correlation": float(result.get("current_correlation", 0)),
                "best_correlation": float(result.get("best_correlation", 0)),
                "correlation_vs_baseline": float(result.get("correlation_vs_baseline", 0)),
                "correlation_vs_current": float(result.get("correlation_vs_current", 0)),
                "baseline_entropy": float(result.get("baseline_entropy", 0)),
                "best_entropy": float(result.get("best_entropy", 0)),
                "entropy_improvement": float(result.get("entropy_improvement", 0)),
                "baseline_slope": float(result.get("baseline_slope", 0)),
                "current_slope": float(result.get("current_slope", 0)),
                "best_slope": float(result.get("best_slope", 0)),
                "slope_vs_baseline": float(result.get("slope_vs_baseline", 0)),
                "slope_vs_current": float(result.get("slope_vs_current", 0)),
                "baseline_pvalue": float(result.get("baseline_pvalue", 1.0)),
                "current_pvalue": float(result.get("current_pvalue", 1.0)),
                "best_pvalue": float(result.get("best_pvalue", 1.0)),
                "baseline_top_gene_vs_median": float(result.get("baseline_top_gene_vs_median", 0)),
                "current_top_gene_vs_median": float(result.get("current_top_gene_vs_median", 0)),
                "best_top_gene_vs_median": float(result.get("best_top_gene_vs_median", 0)),
                "top_gene_vs_median_vs_baseline": float(result.get("top_gene_vs_median_vs_baseline", 0)),
                "top_gene_vs_median_vs_current": float(result.get("top_gene_vs_median_vs_current", 0)),
                "baseline_top_guide_vs_median": float(result.get("baseline_top_guide_vs_median", 0)),
                "current_top_guide_vs_median": float(result.get("current_top_guide_vs_median", 0)),
                "best_top_guide_vs_median": float(result.get("best_top_guide_vs_median", 0)),
                "top_guide_vs_median_vs_baseline": float(result.get("top_guide_vs_median_vs_baseline", 0)),
                "top_guide_vs_median_vs_current": float(result.get("top_guide_vs_median_vs_current", 0)),
                "unique_genes": int(result.get("unique_genes", 0)),
                "effective_rounds": int(result.get("effective_rounds", 0)),
            }
            all_stats.append(stats)
        except Exception as e:
            failed_jobs.append({
                "slurm_job_id": slurm_job_id,
                "experiment": "unknown",
                "well": "unknown",
                "error": f"pickle error: {str(e)[:80]}",
            })
            continue

    # Print failed jobs summary
    if failed_jobs:
        print(f"\n  FAILED JOBS ({len(failed_jobs)} total):")
        # Group by error type
        from collections import Counter
        error_counts = Counter(f["error"][:50] for f in failed_jobs)
        print(f"  Error summary:")
        for err, count in error_counts.most_common(5):
            print(f"    {count}x: {err}...")

        # Print details for first 20 failed jobs
        print(f"\n  Failed job details (showing first 20):")
        for f in failed_jobs[:20]:
            print(f"    [{f['slurm_job_id']}] {f['experiment']} / {f['well']}: {f['error'][:60]}")
        if len(failed_jobs) > 20:
            print(f"    ... and {len(failed_jobs) - 20} more")

    if not all_stats:
        print("\nNo successful results found to aggregate")
        return

    # Create summary DataFrame
    summary_df = pd.DataFrame(all_stats)

    # Wire in fully: merge the optimized rounds straight into the real
    # configs/ops_failed_rounds.yaml (no separate suggestion CSV/YAML/PNG), so
    # get_metrics picks them up on its runtime re-read.
    _update_ops_failed_rounds_config(summary_df, verbose=True)

    # Filter for summary statistics (exclude ISS-only experiments and early experiments)
    import re
    EXCLUDED_EXPERIMENTS = [6, 7, 11, 28, 29, 39, 40, 44, 60, 61, 73, 74, 80, 82, 88, 96]

    def get_exp_number(exp_name):
        match = re.match(r'ops(\d+)', exp_name)
        return int(match.group(1)) if match else -1

    filtered_df = summary_df[~summary_df["experiment"].apply(get_exp_number).isin(EXCLUDED_EXPERIMENTS)]
    n_filtered = len(summary_df) - len(filtered_df)

    # Print the decision summary, then save it as a report alongside the
    # experiment's ISS/mine results (failed_rounds/).
    import io, contextlib
    _buf = io.StringIO()
    with contextlib.redirect_stdout(_buf):
        _print_summary(summary_df, filtered_df, n_filtered)
    report_text = _buf.getvalue()
    print(report_text, end="")
    _save_failed_rounds_report(summary_df, report_text)


def _print_summary(summary_df, filtered_df, n_filtered):
    """Print the optimizer decision summary (match rates, QC metrics, config)."""
    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: Aggregated {len(summary_df)} wells across experiments")
    if n_filtered > 0:
        print(f"  (Filtered {n_filtered} wells from excluded experiments for summary)")
    print(f"{'='*70}")

    if len(filtered_df) > 0:
        # Configuration selection summary
        _print_config_summary(filtered_df)

        # Table 1: Match rates
        print(f"\n  MATCH RATES:")
        hdr = f"    {'Experiment':<28} {'Well':<8} {'Baseline':>8} {'Current':>8} {'Optimized':>9} {'vs Base':>8} {'vs Curr':>8}  Config"
        print(hdr)
        print(f"    {'─'*len(hdr)}")
        for _, row in filtered_df.sort_values(["experiment", "well"]).iterrows():
            vs_base = row["absolute_improvement"]
            vs_curr = row["improvement_vs_current"]
            flag = " ⚠️" if vs_curr < -1 else ""
            print(f"    {row['experiment']:<28} {row['well']:<8} {row['baseline_match_rate']:>7.1f}% {row['current_match_rate']:>7.1f}% {row['best_match_rate']:>8.1f}% {vs_base:>+7.1f}% {vs_curr:>+7.1f}%  {row['config_str']}{flag}")

        # Table 2: QC metrics - baseline vs best
        has_corr = filtered_df["best_correlation"].abs().sum() > 0
        corr_hdr = f"  {'Corr':>6}" if has_corr else ""
        hdr2 = f"    {'Experiment':<28} {'Well':<8} {'':>10}{corr_hdr} {'Slope':>8} {'P-value':>10} {'Top1/Med':>9} {'Genes':>6}"

        print(f"\n  QC METRICS:")
        print(hdr2)
        print(f"    {'─'*len(hdr2)}")
        SLOPE_WARN = -50
        PVALUE_WARN = 0.05
        warned_wells = []
        for _, row in filtered_df.sort_values(["experiment", "well"]).iterrows():
            b_slope = row.get("baseline_slope", 0)
            slope = row.get("best_slope", 0)
            b_pval = row.get("baseline_pvalue", 1.0)
            pval = row.get("best_pvalue", 1.0)
            b_guide = row.get("baseline_top_guide_vs_median", 0)
            guide_ratio = row.get("best_top_guide_vs_median", 0)
            genes = int(row.get("unique_genes", 0))

            b_corr_col = f"  {row.get('baseline_correlation', 0):>6.3f}" if has_corr else ""
            corr_col = f"  {row.get('best_correlation', 0):>6.3f}" if has_corr else ""

            warn = ""
            if slope < SLOPE_WARN or (slope < 0 and pval < PVALUE_WARN):
                warn = "  ⚠️"
                warned_wells.append((row["experiment"], row["well"], slope, pval, b_slope))

            print(f"    {row['experiment']:<28} {row['well']:<8} {'baseline':>10}{b_corr_col} {b_slope:>8.1f} {b_pval:>10.2e} {b_guide:>8.1f}x {genes:>6}")
            print(f"    {'':<28} {'':<8} {'best':>10}{corr_col} {slope:>8.1f} {pval:>10.2e} {guide_ratio:>8.1f}x {genes:>6}{warn}")

        if warned_wells:
            print(f"\n  ⚠️  {len(warned_wells)} well(s) have entropy warnings:")
            for exp, well, sl, pv, bsl in warned_wells:
                print(f"      {exp} {well}: slope={bsl:.1f}→{sl:.1f}, p={pv:.2e}")

        # Flag wells where optimizer found worse than current
        worse_vs_current = filtered_df[filtered_df["improvement_vs_current"] < -1]
        if len(worse_vs_current) > 0:
            print(f"\n  ⚠️  {len(worse_vs_current)} well(s) have optimized config worse than current.")
            print(f"      Current configs may use round combinations the optimizer rejected via QC.")
            print(f"      Review these wells manually before applying.")

        # Print YAML config preview
        import ast
        print(f"\n  RECOMMENDED CONFIG (ops_failed_rounds.yaml format):")
        for experiment in sorted(filtered_df["experiment"].unique()):
            exp_df = filtered_df[filtered_df["experiment"] == experiment]
            print(f"    {experiment}:")
            print(f"      failed_rounds_by_well:")
            for _, row in exp_df.sort_values("well").iterrows():
                well = row["well"]
                dropouts = ast.literal_eval(row["dropouts"]) if row["dropouts"] else []
                shifts = ast.literal_eval(row["shifts"]) if row["shifts"] else []
                if shifts and dropouts:
                    print(f'        "{well}": {{"dropout": {dropouts}, "shift": {shifts}}}')
                elif shifts:
                    print(f'        "{well}": {{"shift": {shifts}}}')
                elif dropouts:
                    print(f'        "{well}": {dropouts}')
                else:
                    print(f'        "{well}": []')


def _save_failed_rounds_report(summary_df, report_text: str) -> None:
    """Save the decision summary to each experiment's ISS/mine/failed_rounds/ dir."""
    for experiment in sorted(summary_df["experiment"].unique()):
        try:
            dataset = OpsDataset(experiment, method="mine")
            report_dir = dataset.results_iss / "failed_rounds"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / "optimization_report.txt"
            report_path.write_text(report_text)
            print(f"  Report saved: {report_path}")
        except Exception as e:
            print(f"  WARNING: could not save report for {experiment}: {e}")


def _print_config_summary(summary_df) -> None:
    """Print configuration selection summary to console."""
    import ast
    from collections import Counter

    total = len(summary_df)

    # Count config types
    config_counts = Counter()
    round_counts = Counter()  # Track which rounds are dropped

    for _, row in summary_df.iterrows():
        dropouts = ast.literal_eval(row["dropouts"]) if row["dropouts"] else []
        shifts = ast.literal_eval(row["shifts"]) if row["shifts"] else []

        n_dropouts = len(dropouts)
        n_shifts = len(shifts)

        # Count round usage
        for r in dropouts:
            round_counts[r] += 1

        # Categorize config type
        if n_dropouts == 0 and n_shifts == 0:
            config_counts["baseline"] += 1
        elif n_shifts > 0 and n_dropouts > 0:
            config_counts["dropout+shift"] += 1
        elif n_shifts > 0:
            config_counts["shift_only"] += 1
        elif n_dropouts == 1:
            config_counts["1_dropout"] += 1
        elif n_dropouts == 2:
            config_counts["2_dropout"] += 1
        else:
            config_counts["3+_dropout"] += 1

    # Print config type summary
    print(f"\n  CONFIGURATION SELECTION:")
    type_order = ["baseline", "1_dropout", "2_dropout", "3+_dropout", "shift_only", "dropout+shift"]
    type_labels = {
        "baseline": "Baseline (no change)",
        "1_dropout": "1-dropout",
        "2_dropout": "2-dropout",
        "3+_dropout": "3+-dropout",
        "shift_only": "Shift only",
        "dropout+shift": "Dropout + Shift",
    }
    for t in type_order:
        count = config_counts.get(t, 0)
        if count > 0:
            print(f"    {type_labels[t]:20s}: {count:4d} wells ({count/total*100:5.1f}%)")

    # Print most common dropout rounds
    if round_counts:
        print(f"\n  MOST COMMON DROPOUT ROUNDS:")
        for round_num, count in round_counts.most_common(5):
            print(f"    Round {round_num}: {count:4d} times ({count/total*100:5.1f}%)")


def _extract_date_from_experiment(exp_name: str) -> str:
    """Extract date string from experiment name like 'ops0006_20250121' -> '20250121'."""
    import re
    match = re.search(r'_(\d{8})', exp_name)
    return match.group(1) if match else "99999999"


def _extract_exp_number(exp_name: str) -> str:
    """Extract experiment number from name like 'ops0006_20250121' -> '0006'."""
    import re
    match = re.match(r'ops(\d+)', exp_name)
    return match.group(1) if match else exp_name[:6]


def _get_dynamic_threshold(n_rounds: int) -> float:
    """Get the minimum improvement threshold based on number of rounds removed.

    Conservative inclusion criterion (removing rounds must clear a rising bar):
    - 1 round:  must improve match rate by >3%
    - 2 rounds: must improve by >5%
    - 3+ rounds: never accepted (removal is capped at 2 rounds)
    """
    if n_rounds <= 1:
        return 3.0            # 1 round: >3%
    elif n_rounds == 2:
        return 5.0            # 2 rounds: >5%
    else:
        return float("inf")   # 3+ rounds: never accept (cap at 2)


# The real per-experiment config the pipeline reads (loaded by
# generate_config_files._load_ops_failed_rounds and re-read at metrics time).
OPS_FAILED_ROUNDS_CONFIG = (
    Path(__file__).parent.parent.parent / "configs" / "ops_failed_rounds.yaml"
)


def _compute_failed_rounds_by_well(summary_df) -> dict:
    """Map each experiment -> {well: failed-rounds} from the optimizer summary.

    Applies the conservative dynamic threshold (see _get_dynamic_threshold):
    a config only "wins" if its improvement clears the bar for the number of
    rounds it removes; otherwise the well falls back to baseline (no removal).
    """
    import ast

    result = {}
    for experiment in sorted(summary_df["experiment"].unique()):
        exp_df = summary_df[summary_df["experiment"] == experiment]
        failed_rounds_by_well = {}
        for _, row in exp_df.iterrows():
            well = row["well"]
            dropouts = ast.literal_eval(row["dropouts"]) if row["dropouts"] else []
            shifts = ast.literal_eval(row["shifts"]) if row["shifts"] else []

            n_rounds = len(dropouts) + len(shifts)
            improvement = row.get("improvement_vs_current", 0)
            if improvement <= _get_dynamic_threshold(n_rounds):
                failed_rounds_by_well[well] = []            # below bar -> baseline
            elif shifts and dropouts:
                failed_rounds_by_well[well] = {"dropout": dropouts, "shift": shifts}
            elif shifts:
                failed_rounds_by_well[well] = {"shift": shifts}
            else:
                failed_rounds_by_well[well] = dropouts
        result[experiment] = failed_rounds_by_well
    return result


def _render_ops_failed_rounds(all_entries: dict) -> str:
    """Render the full ops_failed_rounds.yaml text from {experiment: {well: cfg}}."""
    lines = [
        "# Optimized failed rounds configuration",
        "# Auto-updated by optimize_failed_rounds_orchestrator (per-experiment)",
        f"# Last updated: {__import__('pandas').Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "# Conservative thresholds: 1 round>3%, 2 rounds>5%; removal capped at 2 rounds",
        "",
    ]
    for experiment, failed_rounds in sorted(all_entries.items()):
        lines.append(f"{experiment}:")
        lines.append("  failed_rounds_by_well:")
        for well in sorted(failed_rounds.keys()):
            cfg = failed_rounds[well]
            if isinstance(cfg, dict):
                parts = []
                if cfg.get("dropout"):
                    parts.append(f'"dropout": {list(cfg["dropout"])}')
                if cfg.get("shift"):
                    parts.append(f'"shift": {list(cfg["shift"])}')
                lines.append(f'    "{well}": {{{", ".join(parts)}}}')
            else:
                lines.append(f'    "{well}": {list(cfg) if cfg else []}')
        lines.append("")
    return "\n".join(lines)


def _update_experiment_config(experiment: str, failed_rounds_by_well: dict, verbose: bool = True) -> None:
    """Write failed_rounds_by_well into the experiment's generated config.

    metrics (get_metrics) reads its rounds from the per-experiment config, so we
    update the same three param blocks generate_config_files writes
    (base_calling / metrics / link_calls_tracks) — keeping the experiment config
    the single source of truth rather than adding a second input to metrics.
    """
    import yaml
    from copy import deepcopy

    cfg_path = OpsDataset(experiment).config_paths["exp_config"]
    if not cfg_path.exists():
        if verbose:
            print(f"  experiment config not found, skipped: {cfg_path}")
        return
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    for section in ("base_calling_params", "metrics_params", "link_calls_tracks_params"):
        cfg.setdefault(section, {})
        cfg[section]["failed_rounds_by_well"] = deepcopy(failed_rounds_by_well)
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False, default_flow_style=False)
    if verbose:
        print(f"  Updated experiment config: {cfg_path}")


def _update_ops_failed_rounds_config(summary_df, verbose: bool = True) -> dict:
    """Persist the optimizer's results for the processed experiments to BOTH:

    1. the master ``configs/ops_failed_rounds.yaml`` (merged, preserving every
       other experiment's entry), and
    2. each experiment's generated config (the source metrics actually reads).

    Returns the per-experiment failed_rounds_by_well that was written.
    """
    import yaml

    computed = _compute_failed_rounds_by_well(summary_df)

    # 1. Master config: load existing, overwrite only the experiments we optimized.
    all_entries = {}
    if OPS_FAILED_ROUNDS_CONFIG.exists():
        with open(OPS_FAILED_ROUNDS_CONFIG) as f:
            existing = yaml.safe_load(f) or {}
        for exp, entry in existing.items():
            if isinstance(entry, dict):
                all_entries[exp] = entry.get("failed_rounds_by_well", {}) or {}
    for exp, frbw in computed.items():
        all_entries[exp] = frbw
    OPS_FAILED_ROUNDS_CONFIG.write_text(_render_ops_failed_rounds(all_entries))
    if verbose:
        print(f"Updated {len(computed)} experiment(s) in {OPS_FAILED_ROUNDS_CONFIG}")

    # 2. Per-experiment generated config (what metrics reads).
    for exp, frbw in computed.items():
        _update_experiment_config(exp, frbw, verbose=verbose)

    return computed


def optimize_failed_rounds(
    experiment,
    wells: list[str] = None,
    max_dropouts: int = 2,
    max_shifts: int = 2,
    min_effective_rounds: int = 7,
    no_shifts: bool = False,
    rebuild_cache: bool = False,
    ops_dir: str = DEFAULT_OPS_DIR,
    slurm_params: dict = None,
    dry_run: bool = False,
    **kwargs,
) -> dict:
    """Orchestrator step (runs before get_metrics): optimize ISS failed rounds for
    one experiment, write the result into configs/ops_failed_rounds.yaml, and save a
    decision report under its ISS/mine/failed_rounds/ dir.

    Fans per-well optimization jobs out to SLURM, waits, then aggregates. get_metrics
    re-reads ops_failed_rounds.yaml at runtime, so the fresh rounds flow through.
    """
    from ops_utils.data.filesystem import resolve_experiment_name

    resolved = resolve_experiment_name(experiment, allow_interactive=False) or experiment

    if slurm_params is None:
        slurm_params = {
            "timeout_min": 10,
            "mem": "64GB",
            "cpus_per_task": 16,
            "gpus_per_node": 0,
            "slurm_partition": "cpu,gpu",
        }

    result = submit_optimization_jobs(
        experiments=[resolved],
        wells=wells,
        max_dropouts=max_dropouts,
        max_shifts=0 if no_shifts else max_shifts,
        min_effective_rounds=min_effective_rounds,
        rebuild_cache=rebuild_cache,
        ops_dir=ops_dir,
        slurm_params=slurm_params,
        dry_run=dry_run,
        wait_for_completion=True,
        verbose=True,
    )

    if dry_run:
        return result

    if result.get("all_completed"):
        job_id = result.get("result", {}).get("base_job_id")
        aggregate_results(job_id=job_id)
    else:
        print("optimize_failed_rounds: not all jobs completed; ops_failed_rounds.yaml left unchanged.")
    return result


def main():
    """CLI entry point for SLURM batch optimization submission."""
    parser = argparse.ArgumentParser(
        description="Submit ISS failed rounds optimization jobs to SLURM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "experiment_pos",
        nargs="?",
        type=str,
        help="Experiment to process as a positional arg (e.g., 180 or ops0108_20251209). Equivalent to -e/--experiment.",
    )

    parser.add_argument(
        "max_dropouts_pos",
        nargs="?",
        type=int,
        help="Max dropout rounds as a positional arg (e.g., '180 3'). Equivalent to --max-dropouts.",
    )

    parser.add_argument(
        "--experiment", "-e",
        type=str,
        help="Single experiment to process (e.g., ops0108_20251209)",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Process ALL experiments (batch submission)",
    )

    parser.add_argument(
        "--wells", "-w",
        type=str,
        nargs="+",
        help="Wells to process (e.g., A/1/0 A/2/0). Default: all wells",
    )

    parser.add_argument(
        "--max-dropouts", "--max-drops",
        type=int,
        default=None,
        help="Maximum dropout rounds to test (default: 2, our conservative cap). Rounds with no informative base across the codebook don't count toward this limit.",
    )

    parser.add_argument(
        "--max-shifts",
        type=int,
        default=2,
        help="Maximum shift rounds to test (default: 2)",
    )

    parser.add_argument(
        "--no-shifts",
        action="store_true",
        help="Only search for dropouts, skip all shift testing",
    )

    parser.add_argument(
        "--min-effective-rounds",
        type=int,
        default=7,
        help="Minimum effective rounds required (default: 7)",
    )

    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Force rebuild of reference cache",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=5,
        help="SLURM timeout in minutes (default: 5)",
    )

    parser.add_argument(
        "--mem",
        type=str,
        default="64GB",
        help="SLURM memory allocation (default: 64GB)",
    )

    parser.add_argument(
        "--cpus",
        type=int,
        default=16,
        help="SLURM CPUs per task (default: 16)",
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
        "--aggregate",
        action="store_true",
        help="Aggregate results from completed jobs into summary CSV and YAML",
    )

    parser.add_argument(
        "--job-id",
        type=str,
        help="Filter aggregation to a specific SLURM job ID (use with --aggregate)",
    )

    parser.add_argument(
        "--ops-dir",
        default=DEFAULT_OPS_DIR,
        help=f"Base OPS directory (default: {DEFAULT_OPS_DIR})",
    )

    args = parser.parse_args()

    # Positional experiment is shorthand for -e/--experiment
    if args.experiment_pos:
        if args.experiment and args.experiment != args.experiment_pos:
            parser.error("give the experiment once: either positionally or via -e/--experiment, not both")
        args.experiment = args.experiment_pos

    # Positional max-dropouts is shorthand for --max-dropouts (e.g. "180 3")
    if args.max_dropouts_pos is not None:
        if args.max_dropouts is not None and args.max_dropouts != args.max_dropouts_pos:
            parser.error("give max-dropouts once: either positionally or via --max-dropouts, not both")
        args.max_dropouts = args.max_dropouts_pos
    if args.max_dropouts is None:
        args.max_dropouts = 2  # conservative cap (see _get_dynamic_threshold)

    # Handle --aggregate mode
    if args.aggregate:
        aggregate_results(job_id=args.job_id)
        sys.exit(0)

    # Validation
    if not args.all and not args.experiment:
        parser.error("--experiment or --all is required")

    # Build SLURM parameters
    slurm_params = {
        "timeout_min": args.timeout,
        "mem": args.mem,
        "cpus_per_task": args.cpus,
        "gpus_per_node": 0,
        "slurm_partition": "cpu,gpu",
    }

    # Get experiments to process
    if args.all:
        experiments = scan_ops_experiments(args.ops_dir)
    else:
        from ops_utils.data.filesystem import resolve_experiment_name
        resolved = resolve_experiment_name(args.experiment, allow_interactive=True)
        if resolved is None:
            print("No experiment found. Exiting.")
            sys.exit(1)
        experiments = [resolved]

    # Count total jobs
    total_jobs = 0
    for exp in experiments:
        exp_wells = args.wells if args.wells else get_experiment_wells(exp)
        total_jobs += len(exp_wells)

    # Print plan
    print(f"\n{'='*60}")
    print(f"ISS Failed Rounds Optimization - Job Submission Plan")
    print(f"{'='*60}")
    print(f"Experiments: {len(experiments)}")
    print(f"Total jobs: {total_jobs}")
    print(f"Resources per job: {args.mem}, {args.cpus} CPUs, {args.timeout} min timeout")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("DRY RUN: No jobs will be submitted\n")
        for exp in experiments[:10]:
            exp_wells = args.wells if args.wells else get_experiment_wells(exp)
            print(f"  {exp}: {len(exp_wells)} wells")
        if len(experiments) > 10:
            print(f"  ... and {len(experiments) - 10} more experiments")
        sys.exit(0)

    # Submit jobs
    result = submit_optimization_jobs(
        experiments=experiments,
        wells=args.wells,
        max_dropouts=args.max_dropouts,
        max_shifts=0 if args.no_shifts else args.max_shifts,
        min_effective_rounds=args.min_effective_rounds,
        rebuild_cache=args.rebuild_cache,
        ops_dir=args.ops_dir,
        slurm_params=slurm_params,
        dry_run=args.dry_run,
        wait_for_completion=not args.no_wait,
        verbose=not args.quiet,
    )

    # If jobs completed, aggregate results
    if result.get("all_completed") and not args.no_wait:
        print("\n" + "="*60)
        print("All jobs completed. Aggregating results...")
        print("="*60 + "\n")
        # Get job_id from the submission result to only aggregate current run
        job_id = result.get("result", {}).get("base_job_id")
        aggregate_results(job_id=job_id)

    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
