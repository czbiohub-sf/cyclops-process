"""
PipelineRunner: file-based, idempotent pipeline executor for OPS

Overview
--------
PipelineRunner coordinates execution of the OPS processing pipeline step-by-step
using a strictly file-based completion policy. A step is considered complete iff
all of its expected output files (as defined by
`cyclops_process.data.experiment.OpsDataset.get_output_files_for_step`) exist on
disk. Legacy log-based completion has been retired; logs are written for
auditing and timing only.

Key concepts
------------
- Step identity (log_key):
  - Computed as `func.__qualname__.replace('.', '_')` and optionally suffixed
    with `_process` and/or `_<well>` where `well` is normalized to the same
    convention used elsewhere (e.g., `A/1` becomes `A_1`). This matches the
    keys used by the status reporter so both tools agree.

- Completion checks:
  - For each `log_key`, `OpsDataset.get_output_files_for_step(log_key, config)`
    returns a list of `pathlib.Path` outputs to validate.
  - If the list is non-empty, the step is complete only when all paths exist.
  - If the list is empty or `None`, the step has no definitive output files and
    will be run (never treated as already complete).

- Per-well steps:
  - Callers pass `well=...` when invoking per-well operations. The dataset
    helper maps that to per-well filenames and checks existence accordingly.
  - If the config omits `wells_to_process`, the dataset may infer wells from
    the filesystem, ensuring partial runs can still be resumed deterministically.

Execution modes and user decisions
----------------------------------
- Rerun specific steps (non-interactive):
  - If `rerun_steps` is provided to the constructor, only those exact `log_key`
    steps are executed; all others are silently skipped. Use this for precise,
    non-interactive re-execution.

- Normal interactive mode (default):
  - If a step is missing outputs (incomplete), it is executed automatically.
  - If a step appears complete (all outputs exist), you are prompted:
    - `[s]` skip to next incomplete: sets a flag to skip any further completed
      steps without additional prompts until the first incomplete step is found.
    - `[y]` yes: re-run the current step.
    - `[n]` no: skip the current step only; proceed to next step normally.
    - `[a]` all: re-run all subsequent steps unconditionally (disables
      completion skipping).

- Skip-to-incomplete behavior:
  - When `[s]` is chosen, the runner will suppress prompts for any already
    complete steps and fast-forward until the first incomplete step (where it
    runs). After encountering an incomplete step, the skip flag is cleared.

Slurm integration
-----------------
- `use_slurm=True` enables per-step Slurm submission when a matching spec is
  found in `cyclops_process/configs/slurm_task_config.yaml` under the computed
  `log_key`. The YAML should contain a `slurm_params` section compatible with
  `submitit.AutoExecutor.update_parameters` (e.g., `cpus_per_task`, `mem`,
  `timeout_min`, `slurm_partition`). Steps without a matching entry run locally.

Logging (auditing only)
-----------------------
- For non-versioned functions, the runner writes `function_call_log.yaml` with
  timestamps, elapsed time, and the current git commit. These logs are not used
  to determine completion and exist purely for provenance and diagnostics.

Design notes
------------
- This runner mirrors the status reporter's understanding of completion by using
  the same `log_key` generation and file-path registry in `OpsDataset`,
  guaranteeing that both tools agree on the pipeline state.
"""

import time
import os
import inspect
import sys
import yaml
from pathlib import Path

sys.path.insert(0, os.getcwd())

from cyclops_utils.data.experiment import OpsDataset
from cyclops_process.pipelinerunner.piperun_utils import (
    Deferred,
    _generate_log_key,
    _matches_selection,
    _audit_log_start,
    _audit_log_end,
    _format_timeout_display,
    _lookup_slurm_config_with_fallback,
    scale_timeout_for_wells,
)
from cyclops_process.pipelinerunner.slurm_executor import (
    SlurmExecutor,
    wait_for_outputs,
    wait_for_virtual_staining_jobs,
)
from cyclops_process.pipelinerunner.interactive_menu import InteractiveMenu
from cyclops_process.pipelinerunner.completion_checker import (
    CompletionChecker,
    get_dataset_for_kwargs,
)
from cyclops_utils.hpc.slurm_batch_utils import (
    detect_experiments_needing_processing,
    submit_parallel_jobs,
    check_step_dependencies_satisfied,
)
from cyclops_process.paths import BASE_PATH


def batch_submit_steps_all_experiments(
    rerun_steps: list,
    force: bool = False,
    dry_run: bool = False,
    wait: bool = True,
    slurm_task_config_path: str = None,
    experiment_filter: list = None,
    slurm_qos: str = None,
):
    """
    Fast batch SLURM submission: Submit specific steps for all experiments without orchestration.

    Bypasses full PipelineRunner to avoid interactive prompts and manual checkpoints.
    Directly submits jobs with proper resource allocation from slurm_task_config.yaml.

    Args:
        rerun_steps: List of step names to submit (e.g., ["build_pyramids"])
        force: If True, submit even if outputs exist (default: False)
        dry_run: If True, show what would be submitted without submitting (default: False)
        wait: If True, wait for all jobs to complete (default: True)
        slurm_task_config_path: Optional path to custom slurm_task_config.yaml (default: use standard location)
        experiment_filter: Optional list of experiment number substrings to filter (e.g., ["46", "47", "52"])
    """
    from cyclops_process.pipelinerunner.step_registry import (
        get_step_function,
        get_step_metadata,
        list_all_steps,
    )

    # Validate step names
    try:
        for step in rerun_steps:
            get_step_function(step)  # Will raise if invalid
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)

    dummy_dataset = OpsDataset("dummy")
    config_root_dir = dummy_dataset.config_paths["exp_config_dir"]

    if not config_root_dir.is_dir():
        print(f"Configuration directory not found at {config_root_dir}. Aborting.")
        sys.exit(1)

    # Load SLURM task config
    if slurm_task_config_path:
        slurm_config_path = Path(slurm_task_config_path)
    else:
        slurm_config_path = dummy_dataset.config_paths["slurm_task_config"]
    with open(slurm_config_path, "r") as f:
        slurm_task_config = yaml.safe_load(f) or {}

    # Detect experiments needing processing using shared utility
    def check_step_input(dataset: OpsDataset) -> bool:
        """Check if experiment config exists."""
        return dataset.config_paths["exp_config"].exists()

    def get_step_outputs(dataset: OpsDataset, _) -> list[Path]:
        """Get expected output files for the requested steps."""
        # Load config for this experiment
        try:
            with open(dataset.config_paths["exp_config"], "r") as f:
                config = yaml.safe_load(f) or {}
        except Exception:
            return []

        all_outputs = []
        for step_name in rerun_steps:
            outputs = dataset.get_output_files_for_step(step_name, config)
            if outputs:
                all_outputs.extend(outputs)
        return all_outputs

    # First pass: detect experiments using simple file existence
    experiments_to_process, experiments_completed = detect_experiments_needing_processing(
        input_checker=check_step_input,
        output_checker=get_step_outputs,
        wells=[],  # Not well-based
        force=force,
        verbose=True,
    )

    # Apply experiment filter if provided
    if experiment_filter:
        from cyclops_utils.data.filesystem import resolve_experiment_name

        # Resolve each filter to canonical experiment name
        resolved_experiments = set()
        for f in experiment_filter:
            resolved = resolve_experiment_name(f, autoselect=True)
            resolved_experiments.add(resolved)

        original_count = len(experiments_to_process)
        experiments_to_process = [e for e in experiments_to_process if e[0] in resolved_experiments]
        experiments_completed = [e for e in experiments_completed if e[0] in resolved_experiments]

        print(f"\n[Filter] Applied experiment filter: {experiment_filter}")
        print(f"[Filter] {original_count} -> {len(experiments_to_process)} experiments to process")

    # Second pass: for incomplete experiments, use CompletionChecker to get detailed status
    experiments_with_missing_files = {}
    for exp_tuple in experiments_to_process:
        experiment = exp_tuple[0]
        try:
            config_path = dummy_dataset.config_paths["exp_config_dir"] / f"{experiment}_config.yaml"
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            dataset = OpsDataset(experiment, config)
            checker = CompletionChecker(experiment, config, dataset)

            all_missing = []
            for step_name in rerun_steps:
                # Get the function for this step
                try:
                    step_func = get_step_function(step_name)
                except ValueError:
                    continue

                # Check completion with detailed reasons
                is_complete, missing_files = checker.is_step_complete(step_func, kwargs=None)
                if not is_complete and missing_files:
                    all_missing.extend(missing_files)

            if all_missing:
                experiments_with_missing_files[experiment] = all_missing
        except Exception:
            pass

    print(f"\n{'='*60}")
    print(f"Batch Submission: {', '.join(rerun_steps)}")
    print(f"{'='*60}\n")

    # Filter experiments by checking dependencies FIRST (before printing)
    experiments_ready = []
    experiments_not_ready = []

    for exp_tuple in experiments_to_process:
        experiment = exp_tuple[0]

        try:
            # Load experiment config
            config_path = dummy_dataset.config_paths["exp_config_dir"] / f"{experiment}_config.yaml"
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            # Check dependencies for each step
            dataset = OpsDataset(experiment, config)
            all_deps_satisfied = True
            failed_steps = []

            for step_name in rerun_steps:
                is_ready, reason = check_step_dependencies_satisfied(
                    dataset, step_name, config
                )
                if not is_ready:
                    all_deps_satisfied = False
                    failed_steps.append(f"{step_name} ({reason})")

            if all_deps_satisfied:
                experiments_ready.append((experiment, config))
            else:
                experiments_not_ready.append((experiment, failed_steps))

        except Exception as e:
            experiments_not_ready.append((experiment, [f"Config error: {e}"]))
            continue

    # Print results in order: 1) completed, 2) missing deps, 3) ready to submit

    # 1. Already completed experiments
    if experiments_completed:
        print(f"✓ Already completed ({len(experiments_completed)}):")
        for exp, _, _, _ in experiments_completed:
            print(f"  ✓ {exp}")
        print()

    # 2. Experiments with missing dependencies
    if experiments_not_ready:
        print(f"⚠️  Missing dependencies ({len(experiments_not_ready)}):")
        for exp, reasons in experiments_not_ready:
            print(f"  ✗ {exp}")
            for reason in reasons:
                print(f"      - {reason}")
        print()

    # 3. Experiments ready to submit
    if experiments_ready:
        print(f"→ Ready to submit ({len(experiments_ready)}):")
        for exp, config in experiments_ready:
            print(f"  → {exp}")
            # Show missing files if we have that info
            if exp in experiments_with_missing_files:
                missing_paths = experiments_with_missing_files[exp]
                print(f"      Missing files: {len(missing_paths)}")
                # Show first few paths as examples
                for p in missing_paths[:2]:
                    # Make path relative for readability
                    try:
                        rel_path = p.relative_to(Path(os.environ.get('OPS_OUTPUT_BASE_DIR', f'{BASE_PATH}')))
                        print(f"        - {rel_path}")
                    except:
                        print(f"        - {p}")
        print()
    else:
        print(f"❌ No experiments ready to submit.")
        if not experiments_completed and not experiments_not_ready:
            print("   All experiments are complete.")
        else:
            print("   All remaining experiments have missing dependencies.")
        sys.exit(0)

    # Prepare jobs for submission
    all_jobs = []

    for experiment, config in experiments_ready:
        for step_name in rerun_steps:
            # Get function and metadata from registry
            step_func = get_step_function(step_name)
            metadata = get_step_metadata(step_name)

            # Look up SLURM config
            task_config = _lookup_slurm_config_with_fallback(
                slurm_task_config, step_name
            )
            slurm_params = task_config.get("slurm_params")

            if not slurm_params:
                print(f"  ⚠️  No SLURM config for '{step_name}', skipping {experiment}")
                continue
            slurm_params = scale_timeout_for_wells(slurm_params, config, step_name)

            # Get step parameters from config
            step_params = config.get(f"{step_name}_params", {})

            # Build kwargs based on step metadata
            kwargs = {"experiment": experiment, **step_params}

            if metadata["needs_wells"]:
                # Extract wells from config
                wells = config.get("wells_to_process", [])
                if not wells:
                    print(f"  ⚠️  No wells configured for {experiment}, skipping {step_name}")
                    continue
                kwargs["wells"] = wells

            if metadata["needs_process"]:
                # Process from config (per-experiment override) or step registry default
                kwargs["process"] = step_params.get("process") or metadata.get("process")

            # Pass through extra params from step registry metadata (e.g., restitch_base_only)
            RESERVED_METADATA_KEYS = {"module", "function", "needs_wells", "needs_process", "process"}
            for key, value in metadata.items():
                if key not in RESERVED_METADATA_KEYS and key not in kwargs:
                    kwargs[key] = value

            all_jobs.append({
                "name": f"{experiment}_{step_name}",
                "func": step_func,
                "kwargs": kwargs,
                "metadata": {
                    "experiment": experiment,
                    "step": step_name,
                },
            })

    if not all_jobs:
        print("\n❌ No jobs to submit.")
        sys.exit(1)

    # Get SLURM params from first job (all should use same step params)
    first_step = rerun_steps[0]
    task_config = _lookup_slurm_config_with_fallback(slurm_task_config, first_step)
    slurm_params = task_config.get("slurm_params", {})
    if slurm_qos:
        slurm_params["slurm_qos"] = slurm_qos

    # Get step metadata to show all params being passed
    first_job_metadata = get_step_metadata(first_step)

    # Show job submission plan
    print(f"\n{'='*60}")
    if dry_run:
        print("DRY RUN: Job Submission Plan")
    else:
        print("Job Submission Plan")
    print(f"{'='*60}\n")
    print(f"Step: {first_step}")
    print(f"Total jobs: {len(all_jobs)}")
    print(f"\nSLURM Resources (per job):")
    print(f"  Timeout: {slurm_params.get('timeout_min')} min")
    print(f"  Memory: {slurm_params.get('mem')}")
    print(f"  CPUs: {slurm_params.get('cpus_per_task')}")
    if slurm_params.get('gpus_per_node'):
        print(f"  GPUs: {slurm_params.get('gpus_per_node')}")
    print(f"  Partition: {slurm_params.get('slurm_partition', 'cpu')}")

    # Show all step params from metadata
    print(f"\nStep Metadata:")
    for key, value in first_job_metadata.items():
        print(f"  {key}: {value}")

    print(f"\n{'='*60}\n")

    # Exit if dry run
    if dry_run:
        print("DRY RUN: No jobs submitted\n")
        sys.exit(0)

    # Always prompt for confirmation
    try:
        response = input(f"Submit {len(all_jobs)} jobs to SLURM? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("\nCancelled by user. No jobs submitted.\n")
            sys.exit(0)
    except (KeyboardInterrupt, EOFError):
        print("\n\nCancelled by user. No jobs submitted.\n")
        sys.exit(0)
    print()  # Blank line before submission

    # Submit using shared utility
    log_dir = "slurm_step_logs/all/%j"
    result = submit_parallel_jobs(
        jobs_to_submit=all_jobs,
        experiment=f"batch_{len(experiments_to_process)}_experiments",
        slurm_params=slurm_params,
        log_dir=log_dir,
        manifest_prefix=f"batch_{first_step}",
        dry_run=False,
        wait_for_completion=wait,
        verbose=True,
    )

    # Print full log path for user reference
    full_log_dir = f"slurm_logs/{log_dir}".replace("/%j", "/")
    print(f"\n📁 SLURM logs directory: {full_log_dir}")

    # Exit with appropriate code
    if result.get("success"):
        if result.get("all_completed") is not None:
            sys.exit(0 if result.get("all_completed") else 1)
        else:
            sys.exit(0)
    else:
        sys.exit(1)


class PipelineRunner:
    """
    Manages the execution of the OPS pipeline, handling step completion checks,
    user overrides, and logging for undecorated functions.
    """

    def __init__(
        self,
        experiment: str,
        config: dict,
        rerun_steps: list = None,
        use_slurm: bool = False,
        dataset: OpsDataset = None,
        slurm_qos: str = None,
        auto_run: bool = False,
    ):
        self.experiment = experiment
        self.config = config
        self.rerun_steps = rerun_steps
        self.use_slurm = use_slurm
        self.slurm_qos = slurm_qos
        self.auto_run = auto_run
        self.override_all = False
        self.skip_to_incomplete = (
            True  # Start by fast-forwarding through completed steps
        )

        # Core components - use provided dataset or create new one
        self.dataset = dataset if dataset is not None else OpsDataset(experiment)
        self.log_file_path = self.dataset.logfile
        self.completion_checker = CompletionChecker(experiment, config, self.dataset)
        self.slurm_executor = SlurmExecutor(log_dir=f"slurm_step_logs/{experiment}/%j")

        # Interactive menu state
        self._first_run_prompt_done = bool(auto_run)
        self._abort_all = False
        self._initial_skip_banner_shown = False
        self._last_completed_step_info: tuple | None = None
        self._completed_steps_history: list[tuple] = []
        self._history_index: int | None = None
        self._planned_steps: list[tuple] = []
        self._target_log_key_to_run_once: str | None = None

        # Interactive menu component
        self.menu = InteractiveMenu(
            experiment,
            config,
            self.dataset,
            self.completion_checker,
            lambda key: self._format_timeout_display_wrapper(key),
        )

        # Logs are kept for auditing only
        if self.log_file_path.exists():
            try:
                with open(self.log_file_path, "r") as f:
                    self.log_data = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                print(f"[warning] Corrupted log file {self.log_file_path}, starting fresh: {e}")
                self.log_data = {}
        else:
            self.log_data = {}

        # Load Slurm task configuration
        slurm_config_path = self.dataset.config_paths["slurm_task_config"]
        if slurm_config_path.exists():
            with open(slurm_config_path, "r") as f:
                self.slurm_task_config = yaml.safe_load(f) or {}
        else:
            self.slurm_task_config = {}

        # Menu next steps configuration
        try:
            cfg_val = None
            if isinstance(self.config, dict):
                cfg_val = self.config.get("menu_next_steps")
            env_val = os.getenv("OPS_MENU_NEXT_STEPS")
            if cfg_val is not None:
                self.menu_next_steps = max(1, int(cfg_val))
            elif env_val is not None:
                self.menu_next_steps = max(1, int(env_val))
            else:
                self.menu_next_steps = 8
        except Exception:
            self.menu_next_steps = 8

    # Bind imported helper functions as instance methods
    _generate_log_key = _generate_log_key
    _matches_selection = _matches_selection
    _audit_log_start = _audit_log_start
    _audit_log_end = _audit_log_end

    def _format_timeout_display_wrapper(self, step_key: str) -> str:
        """Wrapper to provide slurm_task_config to _format_timeout_display."""
        return _format_timeout_display(self, step_key)

    def _methods_for_step(self, step_key: str) -> list[str] | None:
        """
        Determine required method variants for a step from the experiment config.

        Convention:
        - Per-step params live under "<step_key>_params" in the config.
        - For steps beginning with "get_", params live under the part after "get_",
          e.g., step_key "get_metrics" uses config key "metrics_params".
        - Within that params dict, either "methods" (list) or "method" (str | list)
          defines the required method variants.

        Returns a list of method names or None if not specified.
        """
        if not isinstance(self.config, dict):
            return None

        # Derive the parameter block key from the step key
        if step_key.startswith("get_"):
            base_name = step_key[4:]
        else:
            base_name = step_key
        params_key = f"{base_name}_params"

        params = self.config.get(params_key)
        if not isinstance(params, dict):
            return None

        # Prefer an explicit list under "methods" if provided
        if "methods" in params:
            methods_value = params.get("methods")
            if isinstance(methods_value, list):
                return [str(m) for m in methods_value if m is not None]
            if isinstance(methods_value, str):
                return [methods_value]

        # Fallback to single or list under "method"
        if "method" in params:
            value = params.get("method")
            if isinstance(value, list):
                return [str(m) for m in value if m is not None]
            if isinstance(value, str):
                return [value]

        return None

    def _execute_step(self, func, kwargs):
        """Execute a pipeline function with logging and optional Slurm."""
        is_versioned = hasattr(func, "_version")
        log_key = self._generate_log_key(func, **kwargs)

        # For unversioned functions, log start time
        if not is_versioned:
            self._audit_log_start(log_key)

        # Look up Slurm configuration with fallback to base key (strip well/method tags)
        task_slurm_config = _lookup_slurm_config_with_fallback(
            self.slurm_task_config, log_key
        )
        slurm_params = task_slurm_config.get("slurm_params")
        slurm_params = scale_timeout_for_wells(slurm_params, self.config, log_key)
        if slurm_params and self.slurm_qos:
            slurm_params["slurm_qos"] = self.slurm_qos
        run_locally_override = bool(task_slurm_config.get("run_locally") is True)

        # Merge parameter overrides from slurm config into kwargs if specified
        # Reserved keys that should not be passed as function parameters
        RESERVED_KEYS = {"slurm_params", "dependencies", "run_locally", "wait_for_outputs",
                         "poll_interval_s", "timeout_min", "env", "total_phases"}

        # This needs to happen BEFORE slurm submission so it gets pickled with the job
        kwargs = dict(kwargs)  # Make a copy to avoid modifying caller's dict

        # Apply any parameter overrides from task config
        for key, value in task_slurm_config.items():
            if key not in RESERVED_KEYS and value is not None:
                kwargs[key] = value
                print(f"[Config Override] Setting {key}={value} for '{log_key}'")

        use_slurm_for_step = (
            self.use_slurm and slurm_params is not None and not run_locally_override
        )

        start_time = time.time()

        # Execute with Slurm or locally
        if use_slurm_for_step:
            self.slurm_executor.execute_with_slurm(
                func, self.experiment, kwargs, slurm_params, log_key
            )
        else:
            if inspect.ismethod(func):
                func(**kwargs)
            else:
                func(self.experiment, **kwargs)

        # Post-execution: wait for outputs if configured
        self._wait_for_step_outputs(func, kwargs, task_slurm_config, log_key)

        # Special handling for virtual staining steps
        self._wait_for_virtual_staining(log_key, task_slurm_config)

        end_time = time.time()

        # Log end time for unversioned functions
        if not is_versioned:
            elapsed_time = end_time - start_time
            self._audit_log_end(log_key, elapsed_time)

    def _wait_for_step_outputs(self, func, kwargs, task_config, log_key):
        """Wait for step outputs to materialize if configured."""
        wait_cfg = task_config.get("wait_for_outputs")
        if not wait_cfg:
            return

        # Normalize wait config (bool or dict)
        if isinstance(wait_cfg, bool):
            poll_interval_s = task_config.get("poll_interval_s", 10)
            timeout_min = task_config.get("timeout_min")
        else:
            poll_interval_s = int(wait_cfg.get("poll_interval_s", 10))
            timeout_min = wait_cfg.get("timeout_min")

        # Get output files
        ds = get_dataset_for_kwargs(self.experiment, self.config, self.dataset, kwargs)
        output_files = self.completion_checker._get_output_files_for_func(
            func, kwargs, ds
        )

        if output_files:
            wait_for_outputs(
                output_files,
                log_key,
                poll_interval_s,
                timeout_min,
                self.completion_checker.has_content,
            )

    def _wait_for_virtual_staining(self, log_key, task_config):
        """Wait for virtual staining Slurm jobs if applicable."""
        if not isinstance(log_key, str) or not log_key.startswith("virtual_staining_"):
            return

        # Determine metadata file path for job IDs
        jobs_meta_path = None
        if log_key.endswith(("_track", "_track_2d", "_track_3d")):
            jobs_meta_path = self.dataset.config_paths.get("vs_jobs_track")
        elif log_key.endswith(("_pheno", "_pheno_2d", "_pheno_3d")):
            jobs_meta_path = self.dataset.config_paths.get("vs_jobs_pheno")

        if not jobs_meta_path:
            return

        # Get wait configuration
        wait_cfg = task_config.get("wait_for_outputs")
        poll_interval_s = 60
        timeout_min = 48 * 60
        if isinstance(wait_cfg, dict):
            poll_interval_s = int(wait_cfg.get("poll_interval_s", poll_interval_s))
            if wait_cfg.get("timeout_min") is not None:
                timeout_min = int(wait_cfg.get("timeout_min"))

        wait_for_virtual_staining_jobs(
            log_key, jobs_meta_path, poll_interval_s, timeout_min
        )

    def plan_step(self, func, **kwargs) -> None:
        """Register an upcoming step for numbered menu display."""
        log_key = self._generate_log_key(func, **kwargs)
        self._planned_steps.append((func, kwargs, log_key))

    def _add_completed_history_entry(self, func, kwargs: dict, log_key: str) -> int:
        """Add a completed step to history and return its selection number."""
        self._completed_steps_history.append((func, kwargs, log_key))
        number = self.menu.add_selection(
            "history", len(self._completed_steps_history) - 1
        )
        self._last_completed_step_info = self._completed_steps_history[-1]
        return number

    def run(self, func, **kwargs):
        """
        Executes a pipeline function.

        If `rerun_steps` is provided during initialization, it will only run
        steps in that list. Otherwise, it operates in interactive mode.
        """
        # Respect global abort
        if self._abort_all or self.menu.is_aborted():
            return

        log_key = self._generate_log_key(func, **kwargs)

        # If user selected a specific future step to run once, skip until we reach it
        target_key = self.menu.get_target_log_key()
        if target_key is not None and not self._matches_selection(log_key, target_key):
            print(
                f"--- Skipping step '{log_key}' until selected target '{target_key}' ---"
            )
            return

        should_run = False

        # --- Mode 1: Rerun specific steps (non-interactive) ---
        if self.rerun_steps is not None:
            if any(self._matches_selection(log_key, step) for step in self.rerun_steps):
                # Show resource allocation summary for SLURM jobs
                if self.use_slurm:
                    task_slurm_config = _lookup_slurm_config_with_fallback(
                        self.slurm_task_config, log_key
                    )
                    slurm_params = task_slurm_config.get("slurm_params")
                    run_locally_override = bool(
                        task_slurm_config.get("run_locally") is True
                    )

                    if slurm_params and not run_locally_override:
                        timeout_display = self._format_timeout_display_wrapper(log_key)
                        print(
                            f"--- Force re-running specified step: '{log_key}'{timeout_display} ---"
                        )
                    else:
                        print(
                            f"--- Force re-running specified step: '{log_key}' (local execution) ---"
                        )
                else:
                    print(f"--- Force re-running specified step: '{log_key}' ---")
                should_run = True
                # NOTE: previous behavior unconditionally aborted after a slurm
                # rerun step ("if self.use_slurm: self._abort_all = True"),
                # which was correct for fire-and-forget --slurm-batch but
                # broke --slurm-steps (waited synchronously then bailed before
                # the next step). Removing the abort here lets the
                # topological iteration continue through every --rerun step.
                # If --slurm-batch behavior needs the early abort, gate it on
                # the actual no-wait flag instead.
            else:
                return

        # --- Mode 2: Interactive execution with completion checks ---
        else:

            should_run = self._check_and_prompt_for_step(func, kwargs, log_key)

        # --- Execute the function if required ---
        if should_run:
            # Check if user wants to proceed with this step
            if not self._handle_first_run_prompt(log_key):
                # User selected a different step to run - skip current step
                return

            # When explicitly re-running a step, inject force=True if the function accepts it
            if self.rerun_steps is not None:
                sig = inspect.signature(func)
                if "force" in sig.parameters:
                    kwargs = {**kwargs, "force": True}

            self._execute_step(func, kwargs)
            # Clear one-shot target after running
            if target_key and self._matches_selection(log_key, target_key):
                self.menu.set_target_log_key(None)

    def _check_and_prompt_for_step(self, func, kwargs, log_key) -> bool:
        """Check completion status and prompt user if needed. Returns True if step should run."""
        # Check completion using CompletionChecker
        # Don't pass methods here - let CompletionChecker handle method variants from kwargs
        is_complete, output_files = self.completion_checker.is_step_complete(
            func, kwargs, methods=None
        )

        if self.auto_run:
            if is_complete and output_files:
                self._add_completed_history_entry(func, kwargs, log_key)
                print(f"[auto] 🟢 skip '{log_key}'")
                return False
            print(f"[auto] ▶ run '{log_key}'")
            return True

        if not output_files:
            # No file-based check defined; default to running
            if self.skip_to_incomplete and not self.override_all:
                self.skip_to_incomplete = False
            return True

        target_key = self.menu.get_target_log_key()
        is_selected_target = target_key and self._matches_selection(log_key, target_key)

        if is_complete:
            # Always record completed steps so they can be re-run via full list
            self._add_completed_history_entry(func, kwargs, log_key)

            # If user explicitly selected this step, force execution
            if is_selected_target:
                return True

            # Skip completed steps in skip-to-incomplete mode
            if self.skip_to_incomplete and not self.override_all:
                print(f"[{len(self._completed_steps_history)}] 🟢 '{log_key}'")
                return False

            # Step is complete but not skipping - prompt user
            if self.override_all:
                return True

            if not is_selected_target:
                action = self.menu.prompt_completed_step(log_key)
                if action == "skip_to_incomplete":
                    self.skip_to_incomplete = True
                    return False
                elif action == "rerun":
                    return True
                elif action == "skip_step":
                    return False
                elif action == "rerun_all":
                    self.override_all = True
                    return True
                elif action == "skip_to_selected":
                    # If target is a previously completed step, re-run it
                    # directly and clear the target so the pipeline continues.
                    target_key = self.menu.get_target_log_key()
                    if target_key:
                        for prev_func, prev_kwargs, prev_key in self._completed_steps_history:
                            if self._matches_selection(prev_key, target_key):
                                print(f"--- Re-running selected previous step: '{prev_key}' ---")
                                self._execute_step(prev_func, prev_kwargs)
                                self.menu.set_target_log_key(None)
                                break
                    return False
                elif action == "quit":
                    print("Aborting pipeline run at user request.")
                    self._abort_all = True
                    self.menu.set_aborted(True)
                    return False
        else:
            # Clear skip mode once we hit first incomplete step
            if self.skip_to_incomplete:
                self.skip_to_incomplete = False
            return True

        return False

    def _handle_first_run_prompt(self, log_key) -> bool:
        """Handle the first-run confirmation prompt.

        Returns:
            True if should execute current step, False if should skip
        """
        if self._first_run_prompt_done or self.rerun_steps is not None:
            return True

        target_key = self.menu.get_target_log_key()
        if target_key and self._matches_selection(log_key, target_key):
            self._first_run_prompt_done = True
            return True

        # Show prompt for first executable step
        timeout_display = self._format_timeout_display_wrapper(log_key)
        action, self._history_index = self.menu.prompt_checkpoint(
            message=f"🟡 First executable step: '{log_key}'{timeout_display}",
            include_initial_banner=True,
            completed_steps_history=self._completed_steps_history,
            history_index=self._history_index,
            initial_skip_banner_shown=self._initial_skip_banner_shown,
            rerun_steps=self.rerun_steps,
        )

        if action == "quit":
            print("Aborting pipeline run at user request.")
            self._abort_all = True
            self.menu.set_aborted(True)
            return False

        if (
            action == "back"
            and self._completed_steps_history
            and self._history_index is not None
        ):
            prev_func, prev_kwargs, prev_key = self._completed_steps_history[
                self._history_index
            ]
            print(f"--- Re-running selected previous step: '{prev_key}' ---")
            self._execute_step(prev_func, prev_kwargs)
            self._first_run_prompt_done = True
            return True
        elif action == "proceed":
            self._first_run_prompt_done = True
            return True
        elif action == "skip":
            # Check if selected target is current step
            target_key = self.menu.get_target_log_key()
            if target_key and self._matches_selection(log_key, target_key):
                self._first_run_prompt_done = True
                return True
            else:
                # If target is a previously completed step, re-run it
                # directly and clear the target so the pipeline continues.
                if target_key:
                    for prev_func, prev_kwargs, prev_key in self._completed_steps_history:
                        if self._matches_selection(prev_key, target_key):
                            print(f"--- Re-running selected previous step: '{prev_key}' ---")
                            self._execute_step(prev_func, prev_kwargs)
                            self.menu.set_target_log_key(None)
                            self._first_run_prompt_done = True
                            return True
                # User selected a future step - skip current step
                return False

        return True

    def _prepare_step_execution(self, func, kwargs):
        """Prepare a step for execution: resolve config, env, slurm params.

        Returns a dict with all resolved execution context, or None if the
        step should be skipped (e.g., no slurm_params in SLURM mode).
        """
        log_key = self._generate_log_key(func, **kwargs)

        task_slurm_config = _lookup_slurm_config_with_fallback(
            self.slurm_task_config, log_key
        )
        slurm_params = task_slurm_config.get("slurm_params")
        slurm_params = scale_timeout_for_wells(slurm_params, self.config, log_key)
        if slurm_params and self.slurm_qos:
            slurm_params["slurm_qos"] = self.slurm_qos
        run_locally_override = bool(task_slurm_config.get("run_locally") is True)

        RESERVED_KEYS = {"slurm_params", "dependencies", "run_locally", "wait_for_outputs",
                         "poll_interval_s", "timeout_min", "env", "total_phases"}

        kwargs = dict(kwargs)
        # Resolve deferred params at dispatch time (after upstream steps ran), so a
        # step can read state those steps produced (e.g. fresh failed_rounds).
        for k, v in list(kwargs.items()):
            if isinstance(v, Deferred):
                kwargs[k] = v.resolve()
        for key, value in task_slurm_config.items():
            if key not in RESERVED_KEYS and value is not None:
                kwargs[key] = value
                print(f"[Config Override] Setting {key}={value} for '{log_key}'")

        use_slurm_for_step = (
            self.use_slurm and slurm_params is not None and not run_locally_override
        )

        return {
            "func": func,
            "kwargs": kwargs,
            "log_key": log_key,
            "task_slurm_config": task_slurm_config,
            "slurm_params": slurm_params,
            "run_locally_override": run_locally_override,
            "use_slurm_for_step": use_slurm_for_step,
            "is_versioned": hasattr(func, "_version"),
        }

    def _execute_step_locally(self, ctx: dict):
        """Execute a prepared step locally (no SLURM).

        Returns the function's return value (e.g. LauncherResult for launcher steps).
        """
        func = ctx["func"]
        kwargs = ctx["kwargs"]

        if inspect.ismethod(func):
            return func(**kwargs)
        else:
            return func(self.experiment, **kwargs)

    def run_group(self, steps: list[tuple]):
        """Execute multiple independent steps, potentially in parallel.

        Each element of *steps* is a ``(func, kwargs_dict)`` tuple, identical
        to what you would pass to ``runner.run(func, **kwargs)``.

        Parallelism behaviour:
        - ``OPS_PARALLEL_GROUPS=0`` (or ``self.parallel_groups is False``):
          falls back to sequential ``run()`` calls (fully backward-compatible).
        - SLURM mode: submits all steps simultaneously, then waits for all to
          finish via ``SlurmExecutor.wait_for_all_jobs()``.
        - Local mode: uses ``concurrent.futures.ThreadPoolExecutor`` so that
          steps that internally launch subprocesses still benefit.

        Args:
            steps: List of (func, kwargs_dict) tuples.  All steps in a group
                   must be independent (no mutual dependencies).
        """
        # Check parallel mode preference
        parallel_enabled = os.environ.get("OPS_PARALLEL_GROUPS", "1") != "0"

        # Sequential fallback — just delegate to run() one by one
        if not parallel_enabled:
            for func, kwargs in steps:
                self.run(func, **kwargs)
            return

        if self._abort_all or self.menu.is_aborted():
            return

        # ── Pre-flight: filter steps that need to run ──────────────────
        steps_to_run: list[tuple] = []  # (func, kwargs, log_key)

        for func, kwargs in steps:
            if self._abort_all or self.menu.is_aborted():
                return

            log_key = self._generate_log_key(func, **kwargs)

            # Honour target-key skip (user selected a specific step to jump to)
            target_key = self.menu.get_target_log_key()
            if target_key is not None and not self._matches_selection(log_key, target_key):
                print(f"--- Skipping step '{log_key}' until selected target '{target_key}' ---")
                continue

            should_run = False

            # Mode 1: rerun_steps filter
            if self.rerun_steps is not None:
                if any(self._matches_selection(log_key, step) for step in self.rerun_steps):
                    print(f"--- Force re-running specified step: '{log_key}' ---")
                    should_run = True
                else:
                    continue
            else:
                # Mode 2: interactive completion check
                should_run = self._check_and_prompt_for_step(func, kwargs, log_key)

            if should_run:
                steps_to_run.append((func, kwargs, log_key))

        if not steps_to_run:
            return

        # Handle first-run prompt for the group (show first step)
        first_log_key = steps_to_run[0][2]
        if not self._handle_first_run_prompt(first_log_key):
            return

        # ── Show group banner ──────────────────────────────────────────
        step_names = ", ".join(lk for _, _, lk in steps_to_run)
        print(f"\n{'='*60}")
        print(f"Parallel group ({len(steps_to_run)} steps): {step_names}")
        print(f"{'='*60}\n")

        # ── Prepare all steps ──────────────────────────────────────────
        prepared = []
        for func, kwargs, log_key in steps_to_run:
            ctx = self._prepare_step_execution(func, kwargs)
            prepared.append(ctx)

        # Audit log start
        for ctx in prepared:
            if not ctx["is_versioned"]:
                self._audit_log_start(ctx["log_key"])

        group_start = time.time()

        # ── Partition into SLURM-submitted vs locally-executed ──────
        slurm_jobs = []   # (submitit.Job, log_key, slurm_params, submission_time)
        local_ctxs = []   # contexts that run locally

        for ctx in prepared:
            if ctx["use_slurm_for_step"]:
                job, sub_time = self.slurm_executor.submit_step(
                    ctx["func"], self.experiment, ctx["kwargs"],
                    ctx["slurm_params"], ctx["log_key"],
                )
                slurm_jobs.append((job, ctx["log_key"], ctx["slurm_params"], sub_time))
            else:
                local_ctxs.append(ctx)

        # Run local steps in threads (they may internally launch subprocesses)
        local_errors: dict[str, Exception] = {}
        if local_ctxs:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def _run_local(ctx):
                self._execute_step_locally(ctx)
                return ctx["log_key"]

            with ThreadPoolExecutor(max_workers=len(local_ctxs)) as pool:
                futures = {pool.submit(_run_local, ctx): ctx["log_key"] for ctx in local_ctxs}
                for future in as_completed(futures):
                    lk = futures[future]
                    try:
                        future.result()
                        print(f"  ✓ '{lk}' completed locally")
                    except Exception as e:
                        local_errors[lk] = e
                        print(f"  ✗ '{lk}' FAILED locally: {e}")

        # Wait for all SLURM jobs
        if slurm_jobs:
            self.slurm_executor.wait_for_all_jobs(slurm_jobs, self.experiment)

        # Post-execution for each step
        for ctx in prepared:
            self._wait_for_step_outputs(
                ctx["func"], ctx["kwargs"],
                ctx["task_slurm_config"], ctx["log_key"]
            )
            self._wait_for_virtual_staining(
                ctx["log_key"], ctx["task_slurm_config"]
            )

        if local_errors:
            step_names_failed = ", ".join(local_errors.keys())
            errors = "; ".join(f"{k}: {v}" for k, v in local_errors.items())
            raise RuntimeError(
                f"Parallel group local failures ({len(local_errors)} steps): {step_names_failed}\n{errors}"
            )

        # Audit log end
        group_end = time.time()
        for ctx in prepared:
            if not ctx["is_versioned"]:
                self._audit_log_end(ctx["log_key"], group_end - group_start)

        # Clear one-shot target if any step matched
        target_key = self.menu.get_target_log_key()
        if target_key:
            for _, _, log_key in steps_to_run:
                if self._matches_selection(log_key, target_key):
                    self.menu.set_target_log_key(None)
                    break

    def is_aborted(self) -> bool:
        """Check if pipeline execution was aborted."""
        return bool(self._abort_all) or self.menu.is_aborted()
