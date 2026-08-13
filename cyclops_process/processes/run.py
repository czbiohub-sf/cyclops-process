"""
Minimal entrypoint for running the OPS pipeline.

Examples:
  - Run a single experiment by name:
      python -m cyclops_process.processes.run --experiment ops0042_20250520

  - Run all experiments with configs:
      python -m cyclops_process.processes.run --all

  - Re-run specific steps for a single experiment:
      python -m cyclops_process.processes.run --experiment ops0042_20250520 --rerun base_calling get_metrics

  - Run with parallel DAG execution (independent branches run concurrently):
      python -m cyclops_process.processes.run --experiment ops0042_20250520 --dag

  - Submit steps to Slurm with step-specific resources:
      python -m cyclops_process.processes.run --all --slurm-steps
      python -m cyclops_process.processes.run --experiment ops0042_20250520 --slurm-steps

  - Run all experiments in parallel locally (requires --rerun):
      python -m cyclops_process.processes.run --all --local-parallel --rerun base_calling

  - Batch submit specific steps for all experiments (with live progress tracking):
      python -m cyclops_process.processes.run --all --slurm-batch --rerun build_pyramids
      python -m cyclops_process.processes.run --all --slurm-batch --rerun build_pyramids --force
      python -m cyclops_process.processes.run --all --slurm-batch --rerun build_pyramids --dry-run
      python -m cyclops_process.processes.run --all --slurm-batch --rerun build_pyramids --no-wait
"""

from pathlib import Path
import argparse
import atexit
import sys
import os

sys.path.insert(0, os.getcwd())


from cyclops_process.pipelinerunner.orchestrator import (
    run_experiment_from_config,
    run_all_experiments_from_configs,
    resolve_experiment_config,
)
from ops_utils.ops_mode import (
    mirror_experiment_logs,
    print_manual_copy_hint_if_any,
    run_timestamp,
    central_log_root,
)


def main():
    parser = argparse.ArgumentParser(description="Run the OPS pipeline.")
    parser.add_argument(
        "-e",
        "--exp",
        "--experiment",
        dest="experiment",
        type=str,
        help="Run a single experiment by name; partial names allowed (e.g., 'ops0042' or 'ops0042_20250520').",
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Run all experiments that have a `_config.yaml` file.",
    )
    parser.add_argument(
        "--rerun",
        nargs="+",
        help="List of specific steps to re-run (e.g., --rerun base_calling get_metrics).",
    )
    parser.add_argument(
        "-ss",
        "--slurm-steps",
        action="store_true",
        default=True,
        help="(default) Submit individual pipeline steps as Slurm jobs with step-specific resource allocation from slurm_task_config.yaml. Pass --local-steps to disable.",
    )
    parser.add_argument(
        "--local-steps",
        action="store_false",
        dest="slurm_steps",
        help="Run pipeline steps locally instead of submitting each as a SLURM job (opt out of the -ss default).",
    )
    parser.add_argument(
        "--slurm-batch",
        action="store_true",
        help="Batch submit specific steps to Slurm for all experiments without waiting. Requires --rerun to be specified. Fast parallel submission.",
    )
    parser.add_argument(
        "--local-parallel",
        action="store_true",
        help="Run all experiments in parallel locally. Requires --rerun to be specified.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if outputs exist (for --slurm-batch).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be submitted without submitting (for --slurm-batch).",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for jobs to complete, exit immediately after submission (for --slurm-batch).",
    )
    parser.add_argument(
        "--dag",
        action="store_true",
        help="Use the parallel DAG runner (steps fire as soon as deps complete). Default is sequential.",
    )
    parser.add_argument(
        "--no-preflight",
        action="store_true",
        help="Skip file-existence preflight; let each step's internal completion logic decide whether to run.",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-run every incomplete step in order and silently skip any complete step. No prompts.",
    )
    parser.add_argument(
        "--slurm-task-config",
        type=str,
        help="Path to custom slurm_task_config.yaml file. Overrides default config location.",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        help="Comma-separated list of experiment numbers to filter (e.g., '46,47,52'). Works with --all.",
    )
    parser.add_argument(
        "--slurm-tag",
        type=str,
        help="Slurm QoS tag to apply to all submitted jobs (e.g., 'icd.ops.processing').",
    )
    parser.add_argument(
        "--mode",
        choices=["research", "operational"],
        default="research",
        help=(
            "Run mode. 'research' (default) keeps logs local only and "
            "redirects OPS_OUTPUT_BASE_DIR / OPS_FAST_OUTPUT_BASE_DIR into a "
            "/rerun/ subdirectory so test runs cannot overwrite real data. "
            "'operational' additionally dual-writes logs to OPS_LOG_ROOTDIR "
            "(see OPS_LOG_ROOTDIR) and uses real output paths."
        ),
    )

    # Positional experiment name (first non-flag arg) fallback
    parser.add_argument(
        "experiment_positional",
        nargs="?",
        help="Experiment name (positional). Used if -e/--experiment is not provided.",
    )

    args = parser.parse_args()

    # If no -e/--experiment provided, but a first positional arg exists, use it
    if not args.experiment and getattr(args, "experiment_positional", None):
        args.experiment = args.experiment_positional

    # Set QoS as environment variable so all submission paths pick it up
    if args.slurm_tag:
        os.environ["OPS_SLURM_QOS"] = args.slurm_tag

    # Resolve effective OPS_MODE: explicit --mode flag wins, else pre-set env var,
    # else the argparse default ("research"). Then propagate via OPS_MODE so SLURM
    # children inherit it, and in research mode redirect output base dirs to /rerun/
    # to prevent test runs from overwriting production data.
    mode_explicit = any(
        a == "--mode" or a.startswith("--mode=") for a in sys.argv[1:]
    )
    if mode_explicit or "OPS_MODE" not in os.environ:
        os.environ["OPS_MODE"] = args.mode
    effective_mode = os.environ.get("OPS_MODE", "research").strip().lower()
    if effective_mode not in ("research", "operational"):
        effective_mode = "research"
    os.environ["OPS_MODE"] = effective_mode

    if effective_mode == "research":
        for var in ("OPS_OUTPUT_BASE_DIR", "OPS_FAST_OUTPUT_BASE_DIR"):
            base = os.environ.get(var)
            if not base:
                continue
            if Path(base).name == "rerun":
                continue
            redirected = str(Path(base) / "rerun")
            os.environ[var] = redirected
            Path(redirected).mkdir(parents=True, exist_ok=True)

    # Stable timestamp shared across all steps in this run; inherits into SLURM
    # children so every step writes under the same central subdirectory.
    run_ts = run_timestamp()

    print(f"==> OPS_MODE={effective_mode}")
    print(f"    OPS_OUTPUT_BASE_DIR={os.environ.get('OPS_OUTPUT_BASE_DIR', '(unset)')}")
    print(f"    OPS_FAST_OUTPUT_BASE_DIR={os.environ.get('OPS_FAST_OUTPUT_BASE_DIR', '(unset)')}")
    if effective_mode == "operational":
        print(f"    OPS_LOG_ROOTDIR={central_log_root()}")
        print(f"    OPS_RUN_TS={run_ts}")

    atexit.register(print_manual_copy_hint_if_any)

    if args.slurm_batch and not args.rerun:
        print("Error: --slurm-batch requires --rerun to be specified.")
        sys.exit(1)

    if args.local_parallel and not args.rerun:
        print("Error: --local-parallel requires --rerun to be specified.")
        sys.exit(1)

    if args.experiment:
        # Resolve experiment config (includes name resolution with interactive selection)
        config_path = resolve_experiment_config(args.experiment, allow_interactive=True)
        if config_path is None:
            print("No experiment selected or found. Exiting.")
            sys.exit(1)
        # Mirror the per-experiment aggregate yaml + logs/ dir at end of run.
        # No-op in research mode. Registered before the manual-copy hint so
        # any mirror failures get captured into MANUAL_COPY_NEEDED.txt before
        # the hint prints.
        atexit.register(mirror_experiment_logs, args.experiment)
        run_experiment_from_config(
            config_path,
            rerun_steps=args.rerun,
            use_slurm_steps=args.slurm_steps,
            slurm_task_config_path=args.slurm_task_config,
            no_preflight=args.no_preflight,
            use_dag=args.dag,
            slurm_qos=args.slurm_tag,
            auto_run=args.auto,
        )
        return

    if args.all:
        # Parse experiment filter if provided
        experiment_filter = None
        if args.experiments:
            experiment_filter = [x.strip() for x in args.experiments.split(",")]

        if args.slurm_batch:
            # Fast batch submission mode - submit specific steps for all experiments
            from cyclops_process.pipelinerunner.pipelinerunner import (
                batch_submit_steps_all_experiments,
            )

            batch_submit_steps_all_experiments(
                rerun_steps=args.rerun,
                force=args.force,
                dry_run=args.dry_run,
                wait=not args.no_wait,
                slurm_task_config_path=args.slurm_task_config,
                experiment_filter=experiment_filter,
                slurm_qos=args.slurm_tag,
            )
        else:
            run_all_experiments_from_configs(
                rerun_steps=args.rerun,
                use_slurm_steps=args.slurm_steps,
                use_local_parallel=args.local_parallel,
                slurm_task_config_path=args.slurm_task_config,
                experiment_filter=experiment_filter,
                use_dag=args.dag,
                slurm_qos=args.slurm_tag,
            )
        return

    print(
        "Please specify --experiment <name> or --all. Optionally use --rerun <step1> <step2> ..."
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
