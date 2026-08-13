"""
Slurm execution and monitoring for PipelineRunner.

This module handles:
- Slurm job submission
- Job status monitoring and completion detection
- Resource usage statistics display
"""

import os
import shutil
import time
import inspect
import subprocess
import sys
from pathlib import Path
from typing import Callable
from datetime import datetime
import yaml
import submitit

from cyclops_utils.hpc.slurm_utils import format_time, print_slurm_job_stats
from cyclops_utils.ops_mode import mirror_slurm_log_dir


def _detect_cuda_path() -> str | None:
    """Auto-detect CUDA toolkit path for CuPy kernel compilation.

    CuPy needs CUDA headers (e.g., cuda.h) to compile reduction kernels at runtime.
    The pip-installed nvidia packages provide libraries but not headers, so we need
    the system CUDA toolkit. Returns None if not found.
    """
    # 1. Check environment variable
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path and Path(cuda_path).is_dir():
        return cuda_path

    # 2. Try to find nvcc on PATH and derive CUDA root
    nvcc = shutil.which("nvcc")
    if nvcc:
        # nvcc is at <cuda_root>/bin/nvcc
        cuda_root = str(Path(nvcc).resolve().parent.parent)
        if Path(cuda_root).is_dir():
            return cuda_root

    # 3. Search toolkit install roots for a version matching PyTorch's CUDA.
    #    Set OPS_CUDA_SEARCH_DIRS (colon-separated) to add cluster-specific roots,
    #    e.g. OPS_CUDA_SEARCH_DIRS=/opt/apps/cuda
    try:
        import torch
        cuda_version = torch.version.cuda  # e.g., "12.6"
        if cuda_version:
            major_minor = cuda_version  # "12.6"
            search_dirs = [
                d for d in os.environ.get("OPS_CUDA_SEARCH_DIRS", "").split(":") if d
            ] + ["/usr/local"]
            for base in search_dirs:
                cuda_dirs = sorted(Path(base).glob(f"{major_minor}*"), reverse=True)
                for d in cuda_dirs:
                    if (d / "bin" / "nvcc").exists():
                        return str(d)
    except Exception:
        pass

    return None


class SlurmExecutor:
    """Handles Slurm job submission and monitoring for pipeline steps."""

    def __init__(self, log_dir: str = "slurm_step_logs/%j"):
        # Prepend slurm_logs/ to keep all SLURM logs in a central location
        log_dir = f"slurm_logs/{log_dir}"

        # Convert relative path to absolute to avoid working directory issues
        from pathlib import Path
        import os
        if not Path(log_dir).is_absolute():
            self.log_dir = str(Path(os.getcwd()) / log_dir)
        else:
            self.log_dir = log_dir

    def _build_setup_commands(self, requires_gpu=False):
        """Build SLURM setup commands for env vars and optional GPU/CPU monitoring."""
        setup_commands = []

        # Preserve environment variables that affect output paths, mode, and tool locations
        for env_var in ("STITCH_PATH", "STITCH_PROFILE",
                        "OPS_OUTPUT_BASE_DIR", "OPS_FAST_OUTPUT_BASE_DIR",
                        "OPS_INPUT_BASE_DIR", "OPS_OVERLAY_DEPTH",
                        "OPS_CONFIGS_DIR", "WAVEORDER_PATH", "PROFILE_PROJECTION",
                        "OPS_MODE", "OPS_LOG_ROOTDIR", "OPS_RUN_TS"):
            if env_var in os.environ:
                setup_commands.append(f"export {env_var}={os.environ[env_var]}")

        # Set CUDA_PATH for CuPy kernel compilation (uv .venv doesn't set this unlike conda)
        cuda_path = _detect_cuda_path()
        if cuda_path:
            setup_commands.append(f"export CUDA_PATH={cuda_path}")

        # Background monitors: collect PIDs and set one cleanup trap at the end
        monitor_pids = []
        log_dir = str(self.log_dir).replace("%j", "${SLURM_JOB_ID}")

        # GPU monitor (nvidia-smi) for GPU jobs
        if requires_gpu and os.environ.get("OPS_GPU_MONITOR", "1") != "0":
            interval = os.environ.get("OPS_GPU_MONITOR_INTERVAL", "5")
            setup_commands.extend([
                f'GPU_MON_LOG="{log_dir}/${{SLURM_JOB_ID}}_gpu_monitor.csv"',
                (
                    f'nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,'
                    f'memory.used,memory.total,temperature.gpu '
                    f'--format=csv -l {interval} > "$GPU_MON_LOG" 2>/dev/null &'
                ),
                f'GPU_MON_PID=$!',
            ])
            monitor_pids.append("$GPU_MON_PID")

        # CPU monitor (mpstat) for per-core utilization
        if os.environ.get("OPS_CPU_MONITOR", "1") != "0":
            interval = os.environ.get("OPS_CPU_MONITOR_INTERVAL", "5")
            setup_commands.extend([
                f'CPU_MON_LOG="{log_dir}/${{SLURM_JOB_ID}}_cpu_monitor.txt"',
                f'mpstat -P ALL {interval} > "$CPU_MON_LOG" 2>/dev/null &',
                f'CPU_MON_PID=$!',
            ])
            monitor_pids.append("$CPU_MON_PID")

        # Single trap to clean up all monitors
        if monitor_pids:
            pids_str = " ".join(monitor_pids)
            setup_commands.append(
                f"trap 'kill {pids_str} 2>/dev/null; wait {pids_str} 2>/dev/null' EXIT"
            )

        return setup_commands

    def _mirror_log_dir_safe(self, job_id: str, experiment: str, log_key: str) -> None:
        """Dual-write this job's log directory to the central store when operational.

        Never raises — central-mirror bookkeeping must not break the pipeline.
        No-op in research mode.
        """
        try:
            local_job_dir = str(self.log_dir).replace("%j", str(job_id))
            mirror_slurm_log_dir(local_job_dir, experiment, log_key, job_id=str(job_id))
        except Exception as e:
            print(f"[ops_mode] warn: central log mirror failed for {log_key}: {e}", file=sys.stderr)

    def submit_step(
        self,
        func: Callable,
        experiment: str,
        kwargs: dict,
        slurm_params: dict,
        log_key: str,
    ) -> tuple[submitit.Job, float]:
        """Submit a Slurm job for a pipeline step WITHOUT waiting.

        Returns the submitit.Job and submission timestamp so the caller can
        decide when/how to wait.  This is the building block for both
        single-step (execute_with_slurm) and multi-step (wait_for_all_jobs)
        execution.

        Args:
            func: Function to execute
            experiment: Experiment name (passed to non-method functions)
            kwargs: Keyword arguments for the function
            slurm_params: Slurm parameters dict (cpus_per_task, mem, timeout_min, etc.)
            log_key: Unique identifier for this step

        Returns:
            (submitit.Job, submission_time) tuple
        """
        # Check if this is a GPU job
        requires_gpu = slurm_params.get("gpus_per_node", 0) > 0 or "gpu" in str(
            slurm_params.get("slurm_gres", "")
        )

        # Check for any environment variables that might affect SLURM
        saved_env = {}

        # Check for toxic NoMachine LD_PRELOAD
        if "LD_PRELOAD" in os.environ:
            val = os.environ["LD_PRELOAD"]
            if "libnxegl.so" in val:
                print(f"DEBUG: Found toxic LD_PRELOAD={val}. Unsetting for submission.")
                saved_env["LD_PRELOAD"] = val
                del os.environ["LD_PRELOAD"]

        # If this is a CPU-only job, temporarily remove GPU env vars from the process
        # environment before calling submitit to prevent them from being inherited
        if not requires_gpu:
            gpu_env_vars = ['CUDA_VISIBLE_DEVICES', 'SLURM_GPUS', 'SLURM_GPUS_PER_NODE',
                            'SLURM_GPUS_PER_TASK', 'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS']
            for var in gpu_env_vars:
                if var in os.environ:
                    saved_env[var] = os.environ[var]
                    del os.environ[var]
            if saved_env:
                print(f"DEBUG: CPU-only job - temporarily unsetting env vars for submission: {list(saved_env.keys())}")

        try:
            executor = submitit.AutoExecutor(folder=self.log_dir)

            # Apply QoS if set via --slurm-tag or OPS_SLURM_QOS env var
            slurm_qos = os.environ.get("OPS_SLURM_QOS")
            if slurm_qos and "slurm_qos" not in slurm_params:
                slurm_params = dict(slurm_params)  # Don't mutate caller's dict
                slurm_params["slurm_qos"] = slurm_qos

            executor.update_parameters(**slurm_params)

            # Set descriptive job name so squeue shows experiment.step instead of "submitit"
            job_name = f"{experiment}.{log_key}"[:128]  # SLURM truncates at 128 chars
            executor.update_parameters(slurm_job_name=job_name)

            # Disable CPU binding to avoid srun binding failures on some nodes
            executor.update_parameters(slurm_srun_args=["--cpu-bind=none"])

            # Build setup commands (env vars, GPU/CPU monitoring)
            setup_commands = self._build_setup_commands(requires_gpu)
            if setup_commands:
                executor.update_parameters(slurm_setup=setup_commands)

            # Bound methods already carry their dataset/experiment on `self`;
            # plain step functions take the experiment as their first argument.
            if inspect.ismethod(func):
                job = executor.submit(func, **kwargs)
            else:
                job = executor.submit(func, experiment, **kwargs)
        finally:
            # Restore environment variables
            if saved_env:
                for var, val in saved_env.items():
                    os.environ[var] = val

        submission_time = time.time()
        self._display_submission_table(
            job.job_id, experiment, log_key, slurm_params, submission_time
        )
        return job, submission_time

    def execute_with_slurm(
        self,
        func: Callable,
        experiment: str,
        kwargs: dict,
        slurm_params: dict,
        log_key: str,
        run_locally_override: bool = False,
    ) -> None:
        """Submit and monitor a single Slurm job for a pipeline step.

        Backward-compatible wrapper: calls submit_step() then waits.

        Args:
            func: Function to execute
            experiment: Experiment name (passed to non-method functions)
            kwargs: Keyword arguments for the function
            slurm_params: Slurm parameters dict (cpus_per_task, mem, timeout_min, etc.)
            log_key: Unique identifier for this step
            run_locally_override: If True, skip Slurm and run locally
        """
        if run_locally_override:
            if inspect.ismethod(func):
                func(**kwargs)
            else:
                func(experiment, **kwargs)
            return

        job, submission_time = self.submit_step(
            func, experiment, kwargs, slurm_params, log_key
        )

        print(
            f"--- Step '{log_key}' submitted as Slurm job {job.job_id}. Waiting for completion... ---"
        )

        timeout_min = slurm_params.get("timeout_min")
        queue_time_sec = 0
        try:
            queue_time_sec = self._wait_for_job_with_timer(
                job, log_key, submission_time, timeout_min
            )
            print(
                f"\n--- Slurm job {job.job_id} for '{log_key}' completed successfully. ---"
            )
        finally:
            # Mirror logs whether the step succeeded or failed — post-mortems need the err log.
            self._mirror_log_dir_safe(job.job_id, experiment, log_key)

        print_slurm_job_stats(
            job.job_id, experiment, slurm_params, queue_time_sec,
        )

    def wait_for_all_jobs(
        self,
        jobs: list[tuple[submitit.Job, str, dict, float]],
        experiment: str,
    ) -> dict[str, int]:
        """Wait for multiple Slurm jobs to complete, showing combined progress.

        Args:
            jobs: List of (submitit.Job, log_key, slurm_params, submission_time) tuples
            experiment: Experiment name for display

        Returns:
            Dict mapping log_key -> queue_time_sec for each job

        Raises:
            RuntimeError if any job fails (after all jobs reach a terminal state)
        """
        if not jobs:
            return {}

        print(f"\n--- Waiting for {len(jobs)} parallel jobs to complete... ---")

        update_interval = 10
        last_update = 0

        # Per-job tracking
        running_start: dict[str, float] = {}
        queue_times: dict[str, int] = {}
        last_states: dict[str, str] = {}
        completed: set[str] = set()
        failed: dict[str, Exception] = {}

        job_map = {log_key: (job, slurm_params, sub_time) for job, log_key, slurm_params, sub_time in jobs}

        while len(completed) + len(failed) < len(jobs):
            current_time = time.time()

            # Check each job for completion
            for log_key, (job, slurm_params, sub_time) in job_map.items():
                if log_key in completed or log_key in failed:
                    continue
                try:
                    if job.done():
                        try:
                            job.result()
                            completed.add(log_key)
                            elapsed = int(current_time - sub_time)
                            print(f"\n  ✓ '{log_key}' completed (job {job.job_id}, {format_time(elapsed)})")
                            print_slurm_job_stats(
                                job.job_id, experiment, slurm_params,
                                queue_times.get(log_key, 0),
                            )
                        except Exception as e:
                            failed[log_key] = e
                            print(f"\n  ✗ '{log_key}' FAILED (job {job.job_id}): {e}")
                        # Mirror logs for every terminal job, regardless of outcome.
                        self._mirror_log_dir_safe(job.job_id, experiment, log_key)
                except Exception as e:
                    failed[log_key] = e
                    print(f"\n  ✗ '{log_key}' error checking status: {e}")

            # Combined progress display
            if current_time - last_update >= update_interval:
                n_done = len(completed)
                n_fail = len(failed)
                n_total = len(jobs)
                n_active = n_total - n_done - n_fail

                # Get states for active jobs
                state_counts: dict[str, int] = {}
                for log_key, (job, slurm_params, sub_time) in job_map.items():
                    if log_key in completed or log_key in failed:
                        continue
                    state = self._get_job_state(job.job_id)

                    # Track queue -> running transitions
                    if last_states.get(log_key) == "PENDING" and state == "RUNNING":
                        running_start[log_key] = current_time
                        queue_times[log_key] = int(current_time - sub_time)

                    last_states[log_key] = state
                    state_counts[state] = state_counts.get(state, 0) + 1

                parts = []
                if state_counts.get("RUNNING", 0):
                    parts.append(f"{state_counts['RUNNING']} running")
                if state_counts.get("PENDING", 0):
                    parts.append(f"{state_counts['PENDING']} queued")
                if n_done:
                    parts.append(f"{n_done} done")
                if n_fail:
                    parts.append(f"{n_fail} failed")

                earliest_sub = min(sub_time for _, _, _, sub_time in jobs)
                total_elapsed = int(current_time - earliest_sub)
                status_line = f"  [{', '.join(parts)}] | Total: {format_time(total_elapsed)}"
                sys.stdout.write(f"\r{status_line}   ")
                sys.stdout.flush()
                last_update = current_time

            time.sleep(1)

        sys.stdout.write("\n")
        sys.stdout.flush()

        if failed:
            step_names = ", ".join(failed.keys())
            errors = "; ".join(f"{k}: {v}" for k, v in failed.items())
            raise RuntimeError(
                f"Parallel group failed ({len(failed)}/{len(jobs)} steps): {step_names}\n{errors}"
            )

        print(f"--- All {len(jobs)} parallel jobs completed successfully. ---\n")
        return queue_times

    def _get_job_state(self, job_id: str) -> str:
        """Query squeue/sacct to get current job state.

        Returns:
            str: Job state (PENDING/RUNNING/COMPLETED/etc)
        """
        try:
            # First check if job is in queue
            cmd = ["squeue", "-j", str(job_id), "-h", "-o", "%T"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()

            # Job not in queue - check sacct for completion
            cmd = ["sacct", "-j", str(job_id), "--format=State", "-n", "-X"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                output = result.stdout.strip()
                # Extract state (first word)
                state = output.split()[0] if output.split() else "UNKNOWN"
                return state
        except Exception:
            pass
        return "UNKNOWN"

    def _wait_for_job_with_timer(
        self,
        job: submitit.Job,
        log_key: str,
        start_time: float,
        timeout_min: int = None,
        progress_callback=None,
    ) -> int:
        """Wait for job completion with live-updating elapsed time display.

        Args:
            job: Submitit job object
            log_key: Step identifier for logging
            start_time: Job submission timestamp
            timeout_min: Optional job timeout in minutes (for progress display)
            progress_callback: Optional callable(progress: float, slurm_state: str).
                When provided, suppresses stdout display and calls this instead.
                progress is 0.0-1.0 based on elapsed/timeout ratio.

        Returns:
            int: Queue time in seconds (0 if not captured)
        """
        update_interval = 2 if progress_callback else 10
        last_update = 0
        last_state = None
        running_start_time = None
        queue_time = None
        terminal_states = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"}

        while True:
            try:
                if job.done():
                    if progress_callback:
                        progress_callback(1.0, "COMPLETED")
                    else:
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                    job.result()
                    return queue_time if queue_time is not None else 0
            except Exception:
                if not progress_callback:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                raise

            current_time = time.time()
            if current_time - last_update >= update_interval:
                state = self._get_job_state(job.job_id)
                total_elapsed = int(current_time - start_time)

                # Detect transition from PENDING to RUNNING
                if last_state == "PENDING" and state == "RUNNING":
                    running_start_time = current_time
                    queue_time = int(current_time - start_time)
                elif state == "RUNNING" and running_start_time is None:
                    running_start_time = current_time

                # SLURM says terminal — job is done
                if state in terminal_states:
                    if progress_callback:
                        # DAG mode: trust sacct directly (like slurm_batch_utils)
                        # Don't wait for NFS-based job.done() — sacct is authoritative
                        progress_callback(1.0, state)
                        if state == "COMPLETED":
                            return queue_time if queue_time is not None else 0
                        else:
                            # FAILED/CANCELLED/TIMEOUT — raise with SLURM state
                            raise RuntimeError(
                                f"SLURM job {job.job_id} ended with state: {state}"
                            )
                    else:
                        # Original mode: wait for submitit NFS files for job.result()
                        for _ in range(10):
                            time.sleep(1)
                            if job.done():
                                sys.stdout.write("\n")
                                sys.stdout.flush()
                                try:
                                    job.result()
                                except Exception:
                                    sys.stdout.write("\n")
                                    sys.stdout.flush()
                                    raise
                                return queue_time if queue_time is not None else 0

                # Calculate progress
                progress = 0.0
                if running_start_time is not None and timeout_min:
                    timeout_sec = timeout_min * 60
                    running_elapsed = current_time - running_start_time
                    progress = min(running_elapsed / timeout_sec, 0.95)

                if progress_callback:
                    progress_callback(progress, state)
                else:
                    # Original stdout display
                    if state == "PENDING":
                        status = "⏳ Queued"
                        timer_str = (
                            f"{status} | Time elapsed: {format_time(total_elapsed)}"
                        )
                    elif state == "RUNNING" and running_start_time is not None:
                        status = "▶️  Running"
                        running_elapsed = int(current_time - running_start_time)
                        pct_str = ""
                        if timeout_min:
                            pct = (running_elapsed / (timeout_min * 60)) * 100
                            pct_str = f" ({pct:.1f}%)"
                        timer_str = (
                            f"{status} | "
                            f"Queue: {format_time(queue_time or 0)} | "
                            f"Running: {format_time(running_elapsed)}{pct_str} | "
                            f"Total: {format_time(total_elapsed)}"
                        )
                    elif state == "RUNNING":
                        status = "▶️  Running"
                        timer_str = (
                            f"{status} | Time elapsed: {format_time(total_elapsed)}"
                        )
                    else:
                        status = f"🔄 {state.title()}"
                        timer_str = (
                            f"{status} | Time elapsed: {format_time(total_elapsed)}"
                        )
                    sys.stdout.write(f"\r{timer_str}")
                    sys.stdout.flush()

                last_update = current_time
                last_state = state

            time.sleep(1)

    def _display_submission_table(
        self,
        job_id: str,
        experiment: str,
        log_key: str,
        slurm_params: dict,
        submission_time: float = None,
    ) -> None:
        """Display formatted table of Slurm job submission parameters."""
        print(f"\n--- Submitting step '{log_key}' to Slurm ---")
        print("┌─────────────┬──────────────────────────┐")
        print("│ Parameter   │ Value                    │")
        print("├─────────────┼──────────────────────────┤")
        print(f"│ Experiment  │ {experiment:<24} │")
        print(f"│ Step        │ {log_key:<24} │")
        print(f"│ Job ID      │ {job_id:<24} │")

        # Add submission timestamp if provided
        if submission_time:
            timestamp = datetime.fromtimestamp(submission_time).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            print(f"│ Submitted   │ {timestamp:<24} │")

        params_to_display = [
            ("timeout_min", "Timeout", lambda v: f"{v} min"),
            ("mem", "Memory", lambda v: v),
            ("cpus_per_task", "CPUs", lambda v: str(v)),
            ("gpus_per_node", "GPUs", lambda v: str(v)),
            ("slurm_partition", "Partition", lambda v: v),
            ("slurm_constraint", "Constraint", lambda v: v),
        ]

        for key, label, formatter in params_to_display:
            if key in slurm_params:
                value = formatter(slurm_params[key])
                print(f"│ {label:<11} │ {value:<24} │")

        print("└─────────────┴──────────────────────────┘")


def wait_for_outputs(
    output_files: list,
    log_key: str,
    poll_interval_s: int = 10,
    timeout_min: float = None,
    has_content_func: Callable = None,
) -> None:
    """Poll for output files until they exist or timeout.

    Args:
        output_files: List of pathlib.Path objects to wait for
        log_key: Step identifier for logging
        poll_interval_s: Seconds between polling checks
        timeout_min: Optional timeout in minutes
        has_content_func: Function to check if a path has content
    """
    if not output_files or has_content_func is None:
        return

    print(
        f"--- Waiting for outputs of '{log_key}' to materialize ({len(output_files)} paths) ---"
    )

    deadline = None
    if isinstance(timeout_min, (int, float)):
        deadline = time.time() + float(timeout_min) * 60.0

    while True:
        all_ready = all(has_content_func(p) for p in output_files)
        if all_ready:
            print(f"--- Detected all outputs for '{log_key}'. Continuing. ---")
            break
        if deadline is not None and time.time() > deadline:
            print(
                f"!!! Timeout while waiting for outputs of '{log_key}'. Continuing anyway."
            )
            break
        time.sleep(float(poll_interval_s))


def wait_for_virtual_staining_jobs(
    log_key: str, jobs_meta_path, poll_interval_s: int = 60, timeout_min: float = None
) -> None:
    """Wait for virtual staining Slurm jobs to complete.

    Args:
        log_key: Step identifier for logging
        jobs_meta_path: Path to YAML file containing job IDs
        poll_interval_s: Seconds between polling checks
        timeout_min: Optional timeout in minutes
    """
    if not jobs_meta_path or not jobs_meta_path.exists():
        return

    try:
        with open(jobs_meta_path, "r") as jf:
            meta = yaml.safe_load(jf) or {}
        array_jid = str(meta.get("array_job_id", "")).strip()
        combine_jid = str(meta.get("combine_job_id", "")).strip()

        print("--- Waiting for virtual staining Slurm jobs to finish ---")

        deadline = time.time() + timeout_min * 60.0 if timeout_min else None

        while True:
            array_done = _job_done(array_jid)
            combine_done = _job_done(combine_jid)
            if array_done and combine_done:
                print("--- Virtual staining Slurm jobs finished. Continuing. ---")
                break
            if deadline is not None and time.time() > deadline:
                print(
                    "!!! Timeout waiting for virtual staining Slurm jobs; continuing."
                )
                break
            time.sleep(float(poll_interval_s))
    except Exception:
        # Fail-soft: do not block pipeline if the readiness check itself fails
        pass


def _job_done(jid: str) -> bool:
    """Check if a Slurm job is complete.

    Args:
        jid: Job ID to check

    Returns:
        True if job is complete or not found, False if still running
    """
    if not jid:
        return True
    try:
        # sacct is faster and does not require interactive session
        cmd = ["sacct", "-j", jid, "--format=State", "-n", "-X"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        out = (res.stdout or "").strip().upper()
        if not out:
            # If sacct returns nothing (e.g., accounting delay), try squeue
            cmd2 = ["squeue", "-j", jid, "-h"]
            res2 = subprocess.run(cmd2, capture_output=True, text=True)
            # If not in queue, assume completed
            return (res2.stdout or "").strip() == ""
        # Consider these as terminal states
        terminal = (
            ("COMPLETED" in out)
            or ("FAILED" in out)
            or ("CANCELLED" in out)
            or ("TIMEOUT" in out)
        )
        return terminal
    except Exception:
        return False
