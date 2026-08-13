"""
Fully async DAG-based pipeline executor for OPS.

Each step fires as soon as its specific dependencies complete. Independent
branches (ISS, tracking, phenotyping) run concurrently without waiting for
each other.

Dependencies are loaded from slurm_task_config.yaml (single source of truth).
Steps are registered declaratively via dag.add(name, func, params).

Usage in orchestrator:
    dag = DAGRunner(runner)
    dag.add("convert_iss", convert.convert, {"process": "iss", ...})
    dag.add("link_tracking", convert.link_tracking, {...})
    ...
    dag.run()
"""

import os
import sys
import time
import traceback
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

from contextlib import nullcontext as _nullcontext

from cyclops_process.pipelinerunner.visualize_dag import parse_dag
from cyclops_process.pipelinerunner.piperun_utils import _lookup_slurm_config_with_fallback


class StepState(Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"      # Downstream of a FAILED step, awaiting retry/skip decision


@dataclass
class StepDef:
    """Definition of a pipeline step."""
    name: str
    func: Callable
    params: dict
    condition: bool = True


@dataclass
class CheckpointDef:
    """Definition of a manual checkpoint node."""
    name: str
    message: str
    after: list[str] = field(default_factory=list)
    before: list[str] = field(default_factory=list)
    condition: bool = True


class DAGRunner:
    """Fully async DAG-based pipeline executor.

    Steps fire as soon as their specific dependencies complete. Independent
    branches run concurrently without waiting for each other.

    Args:
        runner: PipelineRunner instance (provides execution mechanics,
                completion checking, SLURM executor, config).
    """

    def __init__(self, runner):
        self.runner = runner
        self.steps: dict[str, StepDef] = {}
        self.checkpoints: dict[str, CheckpointDef] = {}
        self.state: dict[str, StepState] = {}
        self.errors: dict[str, Exception] = {}
        self.timers: dict[str, dict] = {}  # name -> {"start": float, "end": float}

        # Load dependency graph from slurm_task_config.yaml
        self.yaml_deps: dict[str, list[str]] = {}
        self._load_deps_from_yaml()

        # Retry infrastructure
        self._failed_queue: list[str] = []  # Steps awaiting user retry/skip decision
        self._retry_counts: dict[str, int] = {}  # step -> number of retries

        # Threading coordination
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._display = None  # Set during run()

    def _load_deps_from_yaml(self):
        """Load dependency graph from slurm_task_config.yaml."""
        yaml_path = self.runner.dataset.config_paths.get("slurm_task_config")
        if yaml_path and Path(yaml_path).exists():
            self.yaml_deps = parse_dag(Path(yaml_path))
        else:
            # Fallback: try the standard configs location relative to the package
            fallback = Path(__file__).parent.parent / "configs" / "slurm_task_config.yaml"
            if fallback.exists():
                self.yaml_deps = parse_dag(fallback)
            else:
                self.yaml_deps = {}

    def add(self, name: str, func: Callable, params: dict, condition: bool = True):
        """Register a pipeline step.

        Dependencies are NOT specified here — they come from slurm_task_config.yaml.
        The step name must match the yaml key.

        Args:
            name: Step name (must match slurm_task_config.yaml key).
            func: Callable to execute.
            params: Keyword arguments dict for the function.
            condition: If False, step is immediately marked SKIPPED.
        """
        self.steps[name] = StepDef(name=name, func=func, params=params, condition=condition)

    def add_checkpoint(
        self,
        name: str,
        message: str,
        after: list[str] | None = None,
        before: list[str] | None = None,
        condition: bool = True,
    ):
        """Register a manual checkpoint node in the DAG.

        When all `after` dependencies are satisfied, the scheduler pauses and
        prompts the user. Once confirmed, steps listed in `before` are unblocked.

        Args:
            name: Checkpoint identifier.
            message: Prompt message shown to the user.
            after: Steps that must complete before the checkpoint triggers.
            before: Steps that are blocked until the checkpoint is confirmed.
            condition: If False, checkpoint is auto-confirmed (skipped).
        """
        self.checkpoints[name] = CheckpointDef(
            name=name,
            message=message,
            after=after or [],
            before=before or [],
            condition=condition,
        )

    # ── Dependency resolution ───────────────────────────────────────────

    def _get_deps(self, name: str) -> list[str]:
        """Get dependencies for a step, filtered to only registered steps.

        Unregistered deps are treated as satisfied (enables partial data).
        """
        raw_deps = self.yaml_deps.get(name, [])
        # Only keep deps that are registered steps or checkpoints
        registered = set(self.steps.keys()) | set(self.checkpoints.keys())
        return [d for d in raw_deps if d in registered]

    def _get_checkpoint_deps(self, step_name: str) -> list[str]:
        """Get checkpoint names that block this step (via checkpoint.before)."""
        return [
            cp.name for cp in self.checkpoints.values()
            if step_name in cp.before
        ]

    def _all_deps_satisfied(self, name: str) -> bool:
        """Check if all dependencies for a step are satisfied.

        DONE and SKIPPED both count as satisfied.
        """
        for dep in self._get_deps(name):
            if self.state.get(dep) not in (StepState.DONE, StepState.SKIPPED):
                return False
        # Also check checkpoint gates
        for cp_name in self._get_checkpoint_deps(name):
            if self.state.get(cp_name) not in (StepState.DONE, StepState.SKIPPED):
                return False
        return True

    def _get_children(self, name: str) -> list[str]:
        """Get all steps that depend on this step."""
        children = []
        for step_name in self.steps:
            if name in self._get_deps(step_name):
                children.append(step_name)
        return children

    def _get_downstream(self, name: str) -> set[str]:
        """Get all transitive downstream steps (BFS)."""
        visited = set()
        queue = [name]
        while queue:
            current = queue.pop(0)
            for child in self._get_children(current):
                if child not in visited:
                    visited.add(child)
                    queue.append(child)
        return visited

    # ── Topological ordering ────────────────────────────────────────────

    def _topological_order(self) -> list[str]:
        """Return steps in topological order for display and sequential execution.

        Accounts for both yaml deps and checkpoint gates.
        Uses orchestrator registration order as tiebreaker when multiple
        steps are ready at the same level (not alphabetical).
        """
        # Build registration-order index for stable tiebreaking
        reg_order = {name: i for i, name in enumerate(self.steps.keys())}

        in_degree: dict[str, int] = {}
        for name in self.steps:
            yaml_deps = self._get_deps(name)
            cp_deps = self._get_checkpoint_deps(name)
            in_degree[name] = len(yaml_deps) + len(cp_deps)

        # Also track checkpoint -> step edges for degree decrements
        # Checkpoints are synthetic nodes that become done when their
        # after deps are done. For topological ordering, treat them
        # as passing through: after deps -> checkpoint -> before deps.
        checkpoint_after_steps: dict[str, set[str]] = {}
        for cp_name, cp_def in self.checkpoints.items():
            checkpoint_after_steps[cp_name] = set(
                d for d in cp_def.after if d in self.steps
            )

        order = []
        ready = [n for n, d in in_degree.items() if d == 0]
        ready.sort(key=lambda n: reg_order.get(n, 999))

        resolved_checkpoints: set[str] = set()

        while ready:
            current = ready.pop(0)
            order.append(current)

            # Check if this step completion resolves any checkpoint
            for cp_name, after_set in checkpoint_after_steps.items():
                if cp_name in resolved_checkpoints:
                    continue
                after_set.discard(current)
                if not after_set:
                    # Checkpoint resolved — decrement in_degree for gated steps
                    resolved_checkpoints.add(cp_name)
                    cp_def = self.checkpoints[cp_name]
                    for gated_step in cp_def.before:
                        if gated_step in in_degree:
                            in_degree[gated_step] -= 1
                            if in_degree[gated_step] == 0:
                                ready.append(gated_step)

            # Decrement yaml deps
            for child in self._get_children(current):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    ready.append(child)

            ready.sort(key=lambda n: reg_order.get(n, 999))

        return order

    # ── Pre-flight scan ─────────────────────────────────────────────────

    def _preflight_scan(self) -> dict[str, StepState]:
        """Check completion status for all registered steps.

        Returns:
            Dict mapping step name -> initial state.
        """
        states = {}

        for name, step_def in self.steps.items():
            if not step_def.condition:
                states[name] = StepState.SKIPPED
                continue

            # Check file-based completion
            is_complete, output_files = self.runner.completion_checker.is_step_complete(
                step_def.func, step_def.params, methods=None
            )

            if is_complete and output_files:
                states[name] = StepState.DONE
            else:
                states[name] = StepState.PENDING

        # Handle checkpoints
        for name, cp_def in self.checkpoints.items():
            if not cp_def.condition:
                states[name] = StepState.SKIPPED
            else:
                states[name] = StepState.PENDING

        return states

    def _preflight_scan_skip(self) -> dict[str, StepState]:
        """Skip file checks — mark all steps as PENDING (condition=False → SKIPPED).

        Each step's function will use its own internal completion logic.
        """
        states = {}
        for name, step_def in self.steps.items():
            states[name] = StepState.SKIPPED if not step_def.condition else StepState.PENDING
        for name, cp_def in self.checkpoints.items():
            states[name] = StepState.SKIPPED if not cp_def.condition else StepState.PENDING
        return states

    # ── Step execution ──────────────────────────────────────────────────

    def _execute_step(self, name: str) -> None:
        """Execute a single step using PipelineRunner's execution mechanics.

        This runs in a worker thread. All stdout/stderr is redirected to a
        per-step log file so the terminal table is not disrupted.
        """
        step_def = self.steps[name]

        # Redirect output to log file
        redirect = self._display.redirect_step_output(name) if self._display else None

        try:
            ctx_manager = redirect if redirect else _nullcontext()
            with ctx_manager:
                # Use PipelineRunner's prepare + execute infrastructure
                ctx = self.runner._prepare_step_execution(step_def.func, step_def.params)

                if ctx["use_slurm_for_step"]:
                    # SLURM submission + wait
                    job, sub_time = self.runner.slurm_executor.submit_step(
                        ctx["func"], self.runner.experiment, ctx["kwargs"],
                        ctx["slurm_params"], ctx["log_key"],
                    )
                    # Update display to show SLURM job ID
                    if self._display:
                        self._display.update_step(name, StepState.RUNNING, job_id=job.job_id)

                    # Wait for SLURM job using existing infrastructure with progress callback
                    timeout_min = ctx["slurm_params"].get("timeout_min")

                    def _progress_cb(progress, slurm_state, _name=name):
                        if self._display:
                            self._display.update_step(
                                _name, StepState.RUNNING,
                                progress=progress, slurm_state=slurm_state,
                            )

                    self.runner.slurm_executor._wait_for_job_with_timer(
                        job, ctx["log_key"], time.time(), timeout_min,
                        progress_callback=_progress_cb,
                    )
                    # Skip print_slurm_job_stats — it does sacct + sleep(2) and
                    # stdout is redirected to log file anyway. Stats are available
                    # via sacct post-hoc.
                else:
                    # Local execution — set up PhaseTracker context so multi-phase
                    # launcher steps can delegate waiting to DAG-tracked polling.
                    from ops_utils.hpc.phase_tracker import PhaseTracker, _current_phase_tracker
                    total_phases = ctx.get("task_slurm_config", {}).get("total_phases", 1)
                    tracker = PhaseTracker(name, self._track_inner_jobs, total_phases=total_phases)
                    token = _current_phase_tracker.set(tracker)
                    try:
                        result = self.runner._execute_step_locally(ctx)
                    finally:
                        _current_phase_tracker.reset(token)

                    # If this is a single-phase launcher step that returned
                    # LauncherResult directly, track those jobs now.
                    from ops_utils.hpc.launcher_result import LauncherResult
                    if isinstance(result, LauncherResult):
                        n_failed, _, _ = self._track_inner_jobs(name, result)
                        if n_failed > 0:
                            raise RuntimeError(
                                f"{n_failed}/{result.total_jobs} inner jobs failed"
                            )

                # Post-execution: wait for outputs
                self.runner._wait_for_step_outputs(
                    ctx["func"], ctx["kwargs"],
                    ctx["task_slurm_config"], ctx["log_key"]
                )
                self.runner._wait_for_virtual_staining(
                    ctx["log_key"], ctx["task_slurm_config"]
                )

                # Audit log
                elapsed = time.time() - self.timers[name]["start"]
                if not ctx["is_versioned"]:
                    self.runner._audit_log_end(ctx["log_key"], elapsed)

        except Exception:
            raise

    def _track_inner_jobs(self, name: str, lr, phase_label: str = "") -> tuple:
        """Poll inner SLURM jobs submitted by a launcher step.

        Uses the same terminal-state detection and polling pattern as
        ``_wait_for_jobs`` / ``wait_for_multiple_job_arrays`` in
        ``slurm_batch_utils.py`` to correctly handle fast-completing jobs,
        CANCELLED+reason suffixes, PREEMPTED, etc.

        Supports multiple job arrays (e.g. convert_v3 submits base + seg per
        store). Queries sacct for each base_job_id and aggregates progress.

        Args:
            name: Step name (for display updates).
            lr: LauncherResult with job_arrays, submitted_jobs, total_jobs.
            phase_label: Optional label like "1/2 pyramids" for multi-phase steps.

        Returns:
            Tuple of (n_failed, completed_names_list, failed_ids_list).
        """
        import subprocess
        from ops_utils.hpc.slurm_batch_utils import _is_terminal_state

        if self._display:
            self._display.update_step(name, StepState.RUNNING, job_id=lr.base_job_id)

        completed_set: set[str] = set()  # tracks job_ids (not names — names can collide)
        failed_list: list[tuple[str, str, str]] = []  # (job_name, job_id, state)
        failed_ids: set[str] = set()  # tracks job_ids

        # Collect unique base IDs for sacct queries (strip array suffixes)
        base_ids: set[str] = set()
        for arr in lr.job_arrays:
            bid = arr.base_job_id.split("_")[0] if "_" in arr.base_job_id else arr.base_job_id
            base_ids.add(bid)

        submission_time = time.time()
        poll_interval = 2  # Match slurm_batch_utils polling cadence

        while len(completed_set) + len(failed_ids) < lr.total_jobs:
            # Batch query sacct for all job arrays
            # (same pattern as wait_for_multiple_job_arrays in slurm_batch_utils.py)
            job_states: dict[str, str] = {}
            for base_id in base_ids:
                try:
                    cmd = ["sacct", "-j", base_id, "--format=JobID,State", "-n", "-P"]
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    if r.returncode == 0:
                        for line in r.stdout.strip().split("\n"):
                            if "|" in line:
                                parts = line.split("|")
                                jid = parts[0].split(".")[0]  # Remove .batch suffix
                                job_states[jid] = parts[1]
                except Exception:
                    pass

            # Check each submitted job across all arrays
            for job_info in lr.submitted_jobs:
                job_name = job_info["name"]
                job_id = str(job_info["job"].job_id)
                if job_id in completed_set or job_id in failed_ids:
                    continue

                state = job_states.get(job_id, "")
                # Use the same _is_terminal_state that handles CANCELLED+reason,
                # PREEMPTED, BOOT_FAIL, DEADLINE, REVOKED, etc.
                if _is_terminal_state(state):
                    if state == "COMPLETED":
                        completed_set.add(job_id)
                        # Log to the step's redirected stdout
                        print(f"  ✓ {job_name} completed "
                              f"({len(completed_set)}/{lr.total_jobs}) "
                              f"[job: {job_id}]")
                    else:
                        failed_list.append((job_name, job_id, state))
                        failed_ids.add(job_id)
                        print(f"  ✗ {job_name} {state} "
                              f"({len(failed_ids)} failed) "
                              f"[job: {job_id}]")

            # Update display with fractional progress
            n_done = len(completed_set)
            n_fail = len(failed_ids)
            progress = (n_done + n_fail) / lr.total_jobs if lr.total_jobs > 0 else 0
            slurm_label = f"{n_done}/{lr.total_jobs}"
            if n_fail:
                slurm_label += f" ({n_fail}err)"
            if phase_label:
                slurm_label = f"{phase_label} {slurm_label}"

            if self._display:
                self._display.update_step(
                    name, StepState.RUNNING,
                    progress=progress, slurm_state=slurm_label,
                )

            if n_done + n_fail < lr.total_jobs:
                time.sleep(poll_interval)

        # Log final summary (goes to step log file via redirect)
        elapsed = time.time() - submission_time
        print(f"\n{'='*60}")
        print(f"Inner jobs finished in {elapsed/60:.1f} minutes")
        print(f"  Completed: {len(completed_set)}/{lr.total_jobs}")
        print(f"  Failed: {len(failed_ids)}/{lr.total_jobs}")
        print(f"{'='*60}\n")

        return (
            len(failed_list),
            list(completed_set),
            [name for name, _, _ in failed_list],
        )

    def _handle_checkpoint(self, name: str) -> bool:
        """Handle a manual checkpoint — prompt user and wait for confirmation.

        Returns True if confirmed, False if user quit.
        """
        cp_def = self.checkpoints[name]
        if not cp_def.condition:
            return True

        # Display checkpoint prompt below the progress table
        if self._display:
            self._display.show_checkpoint(name, cp_def.message)

        # Use PipelineRunner's interactive menu for the prompt
        action = self.runner.menu.prompt_checkpoint(message=cp_def.message)[0]

        # Resume the display refresh thread
        if self._display:
            self._display.resume_after_prompt()

        if action == "quit":
            return False

        return True

    # ── Main execution loop ─────────────────────────────────────────────

    def run(self, skip_preflight: bool = False):
        """Execute all registered steps respecting dependencies.

        Args:
            skip_preflight: If True, skip file-existence checks and mark all
                steps as PENDING. Each step's function will use its own
                internal completion logic to decide whether to actually run.

        Execution mode:
        - OPS_PARALLEL_GROUPS=0: sequential (topological order, one at a time)
        - Otherwise: fully async (steps fire as soon as deps complete)
        """
        if not self.steps:
            print("DAGRunner: No steps registered. Nothing to run.")
            return

        parallel_enabled = os.environ.get("OPS_PARALLEL_GROUPS", "1") != "0"

        # ── Phase 1: Pre-flight scan ────────────────────────────────────
        if skip_preflight:
            self.state = self._preflight_scan_skip()
        else:
            self.state = self._preflight_scan()

        # Handle rerun_steps: only the tagged steps run, everything else is skipped.
        # Rerun mode is always sequential — no parallelism, no dependency chasing.
        rerun_mode = self.runner.rerun_steps is not None
        if rerun_mode:
            rerun_names = []
            for name in self.steps:
                log_key = self.runner._generate_log_key(
                    self.steps[name].func, **self.steps[name].params
                )
                if any(
                    self.runner._matches_selection(log_key, s)
                    for s in self.runner.rerun_steps
                ):
                    self.state[name] = StepState.PENDING
                    rerun_names.append(name)
                else:
                    # Skip everything not explicitly tagged
                    self.state[name] = StepState.SKIPPED

        # Count states
        n_done = sum(1 for s in self.state.values() if s == StepState.DONE)
        n_pending = sum(1 for s in self.state.values() if s == StepState.PENDING)
        n_skipped = sum(1 for s in self.state.values() if s == StepState.SKIPPED)
        total = len(self.steps) + len(self.checkpoints)

        print(f"\nDAG Pipeline: {total} steps total — "
              f"{n_done} done, {n_pending} pending, {n_skipped} skipped")

        if n_pending == 0:
            print("All steps complete. Nothing to run.")
            return

        # ── Phase 2: Create display and confirm ─────────────────────────
        from cyclops_process.pipelinerunner.dag_display import DAGDisplay

        # Order steps for display (topological)
        topo_order = self._topological_order()
        log_dir = Path(f"slurm_logs/dag/{self.runner.experiment}/dag_run_{int(time.time())}")
        self._display = DAGDisplay(topo_order, self.state, log_dir)
        self._display.draw_initial()

        # Interactive confirmation (skip in rerun mode — steps already chosen)
        if not rerun_mode:
            # Pause refresh so the prompt isn't overwritten
            if self._display:
                self._display._refresh_paused = True
            action = self._prompt_dag_confirmation(topo_order)
            if action == "cancel":
                print("DAG execution cancelled.")
                if self._display:
                    self._display.finalize()
                return
            # Resume refresh for execution
            if self._display:
                self._display._refresh_paused = False
                with self._display._lock:
                    self._display._redraw()

        # ── Phase 3: Execute ────────────────────────────────────────────
        dag_start = time.time()

        try:
            if rerun_mode:
                # Rerun mode: run only tagged steps sequentially, no deps
                self._run_rerun(rerun_names)
            elif parallel_enabled:
                self._run_async()
            else:
                self._run_sequential()
        except KeyboardInterrupt:
            print("\n\nDAG execution interrupted by user.")
        except Exception as e:
            from cyclops_process.pipelinerunner.exceptions import PipelineHalted
            if isinstance(e, PipelineHalted):
                print(f"\n--- Pipeline halted: {e.reason} ---")
            else:
                raise
        finally:
            if self._display:
                self._display.finalize()

        # ── Phase 4: Summary ────────────────────────────────────────────
        dag_elapsed = time.time() - dag_start
        n_done = sum(1 for s in self.state.values() if s == StepState.DONE)
        n_failed = sum(1 for s in self.state.values() if s == StepState.FAILED)
        n_skipped = sum(1 for s in self.state.values() if s == StepState.SKIPPED)

        print(f"\nDAG Pipeline complete in {self._format_elapsed(dag_elapsed)}")
        print(f"  Done: {n_done}  Failed: {n_failed}  Skipped: {n_skipped}")

        if self.errors:
            print(f"\nFailed steps:")
            for name, err in self.errors.items():
                print(f"  {name}: {err}")

    def _print_step_list(self, topo_order: list[str]):
        """Print a compact numbered list of all steps with their current state."""
        n_done = 0
        n_pending = 0
        n_skipped = 0
        for i, name in enumerate(topo_order, 1):
            state = self.state.get(name, StepState.PENDING)
            if state == StepState.DONE:
                marker = "\u2714 done   "
                n_done += 1
            elif state == StepState.PENDING:
                marker = "\u25b6 RUN    "
                n_pending += 1
            elif state == StepState.SKIPPED:
                marker = "\u2013 skip   "
                n_skipped += 1
            else:
                marker = f"? {state.value:<7}"
            print(f"  {i:3d}) {marker}  {name}")
        print(f"\n  {n_pending} to run, {n_done} done, {n_skipped} skipped")

    def _prompt_dag_confirmation(self, topo_order: list[str]) -> str:
        """Show preflight plan and ask user to proceed, edit, or cancel.

        Prints a full numbered step list so the user can see exactly what
        will run, then prompts for confirmation. Edit mode lets the user
        toggle steps on/off by number.

        Returns:
            "proceed" or "cancel"
        """
        # Show which steps are ready to fire immediately (all deps satisfied)
        ready_now = [
            n for n in topo_order
            if self.state.get(n) == StepState.PENDING and self._all_deps_satisfied(n)
        ]
        pending_blocked = [
            n for n in topo_order
            if self.state.get(n) == StepState.PENDING and not self._all_deps_satisfied(n)
        ]

        if ready_now:
            print(f"\n  Will trigger immediately ({len(ready_now)}):")
            for name in ready_now:
                print(f"    \u25b6 {name}")
        if pending_blocked:
            print(f"  + {len(pending_blocked)} more pending (waiting on deps)")
        if not ready_now and not pending_blocked:
            print("\n  No steps to run.")

        while True:
            print(f"\n  [y] proceed  [e] edit step selection  [q] cancel")
            choice = input("  > ").strip().lower()

            if choice in ("p", "proceed", "y", "yes", ""):
                return "proceed"

            if choice in ("c", "cancel", "q", "quit"):
                return "cancel"

            if choice in ("e", "edit"):
                # Pause display refresh during editing
                if self._display:
                    self._display._refresh_paused = True

                print(f"\n  Toggle steps with numbers (e.g. '3', '1-5', '3,7,12'), 'all', or 'done' to finish:")
                print(f"  {'─' * 60}")
                self._print_step_list(topo_order)
                print(f"  {'─' * 60}")

                while True:
                    sel = input("  edit> ").strip().lower()
                    if sel in ("done", "d", ""):
                        break

                    if sel == "all":
                        # Toggle all: if any are pending, skip all; if all skipped, pend all
                        any_pending = any(
                            self.state.get(n) == StepState.PENDING for n in topo_order
                        )
                        for name in topo_order:
                            if self.state.get(name) in (StepState.PENDING, StepState.SKIPPED):
                                self.state[name] = StepState.SKIPPED if any_pending else StepState.PENDING
                    else:
                        # Parse comma-separated numbers and ranges
                        indices = set()
                        for part in sel.split(","):
                            part = part.strip()
                            if "-" in part:
                                try:
                                    a, b = part.split("-", 1)
                                    for j in range(int(a), int(b) + 1):
                                        indices.add(j)
                                except ValueError:
                                    pass
                            elif part.isdigit():
                                indices.add(int(part))

                        for idx in indices:
                            if 1 <= idx <= len(topo_order):
                                name = topo_order[idx - 1]
                                cur = self.state.get(name)
                                # Toggle between PENDING and SKIPPED (don't touch DONE)
                                if cur == StepState.PENDING:
                                    self.state[name] = StepState.SKIPPED
                                elif cur == StepState.SKIPPED:
                                    self.state[name] = StepState.PENDING

                    # Reprint after each edit
                    print(f"  {'─' * 60}")
                    self._print_step_list(topo_order)
                    print(f"  {'─' * 60}")

                # Update display with new states and resume refresh
                if self._display:
                    for name in topo_order:
                        self._display.states[name] = self.state[name]
                    self._display._refresh_paused = False
                    with self._display._lock:
                        self._display._redraw()
                continue

            print("  Invalid choice. Enter 'y', 'e', or 'q'.")

    def _run_sequential(self):
        """Execute steps in topological order, one at a time."""
        topo_order = self._topological_order()

        for name in topo_order:
            if self.runner._abort_all or self.runner.menu.is_aborted():
                break

            if self.state[name] != StepState.PENDING:
                continue

            # Check if deps are satisfied (some may have failed)
            if not self._all_deps_satisfied(name):
                self.state[name] = StepState.SKIPPED
                if self._display:
                    self._display.update_step(name, StepState.SKIPPED)
                continue

            # Handle checkpoints that gate this step
            for cp_name in self._get_checkpoint_deps(name):
                if self.state.get(cp_name) == StepState.PENDING:
                    # Check if checkpoint deps are satisfied
                    cp_def = self.checkpoints[cp_name]
                    cp_deps_ok = all(
                        self.state.get(d) in (StepState.DONE, StepState.SKIPPED)
                        for d in cp_def.after
                        if d in self.state
                    )
                    if cp_deps_ok:
                        self.state[cp_name] = StepState.RUNNING
                        if not self._handle_checkpoint(cp_name):
                            self.runner._abort_all = True
                            return
                        self.state[cp_name] = StepState.DONE

            # Re-check after checkpoint handling
            if not self._all_deps_satisfied(name):
                self.state[name] = StepState.SKIPPED
                if self._display:
                    self._display.update_step(name, StepState.SKIPPED)
                continue

            # Execute step (with retry loop)
            while True:
                self.state[name] = StepState.RUNNING
                self.timers[name] = {"start": time.time()}
                if self._display:
                    self._display.update_step(name, StepState.RUNNING)

                # Audit log start
                step_def = self.steps[name]
                log_key = self.runner._generate_log_key(step_def.func, **step_def.params)
                if not hasattr(step_def.func, "_version"):
                    self.runner._audit_log_start(log_key)

                try:
                    self._execute_step(name)
                    self.state[name] = StepState.DONE
                    self.timers[name]["end"] = time.time()
                    if self._display:
                        elapsed = self.timers[name]["end"] - self.timers[name]["start"]
                        self._display.update_step(name, StepState.DONE, elapsed=elapsed)
                    break  # Success — move to next step

                except Exception as e:
                    # PipelineHalted = step signalled clean shutdown (e.g.
                    # required upstream artifact missing). Don't prompt for
                    # retry; bubble it up to the orchestrator's outer loop.
                    from cyclops_process.pipelinerunner.exceptions import PipelineHalted
                    if isinstance(e, PipelineHalted):
                        raise

                    self._handle_failure(name, e)

                    # Interactive retry prompt (sequential — no threading concerns)
                    action = self._prompt_retry(name)
                    self._failed_queue.pop(0)
                    self._resolve_failure(name, action)

                    if action == "retry":
                        continue  # Re-execute the step
                    else:
                        break  # Skip — move to next step

    def _run_rerun(self, rerun_names: list[str]):
        """Run only the explicitly tagged steps, sequentially, ignoring deps.

        Used when --rerun is passed. No parallelism, no dependency resolution.
        Steps run in the order they were tagged.
        """
        for name in rerun_names:
            if self.runner._abort_all or self.runner.menu.is_aborted():
                break

            while True:
                self.state[name] = StepState.RUNNING
                self.timers[name] = {"start": time.time()}
                if self._display:
                    self._display.update_step(name, StepState.RUNNING)

                step_def = self.steps[name]
                log_key = self.runner._generate_log_key(step_def.func, **step_def.params)
                if not hasattr(step_def.func, "_version"):
                    self.runner._audit_log_start(log_key)

                try:
                    self._execute_step(name)
                    self.state[name] = StepState.DONE
                    self.timers[name]["end"] = time.time()
                    elapsed = self.timers[name]["end"] - self.timers[name]["start"]
                    if self._display:
                        self._display.update_step(name, StepState.DONE, elapsed=elapsed)
                    break

                except Exception as e:
                    from cyclops_process.pipelinerunner.exceptions import PipelineHalted
                    if isinstance(e, PipelineHalted):
                        raise
                    self._handle_failure(name, e)
                    action = self._prompt_retry(name)
                    self._failed_queue.pop(0)
                    self._resolve_failure(name, action)

                    if action == "retry":
                        continue
                    else:
                        break

    def _run_async(self):
        """Execute steps with full concurrency — fire as soon as deps complete."""
        max_workers = int(os.environ.get("OPS_DAG_MAX_WORKERS", "8"))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures: dict[str, Future] = {}  # step_name -> Future

            with self._lock:
                # Seed initial READY set
                self._promote_ready_steps()

                # Submit initial batch
                self._submit_ready(pool, futures)

            # Main loop: wait for completions and promote new steps
            while True:
                with self._lock:
                    # Check termination: no RUNNING, READY, or BLOCKED steps left
                    active = sum(
                        1 for s in self.state.values()
                        if s in (StepState.RUNNING, StepState.READY, StepState.BLOCKED)
                    )
                    if active == 0 and not self._failed_queue:
                        break

                    if self.runner._abort_all or self.runner.menu.is_aborted():
                        break

                # Wait a bit then check for completions
                # (We use polling rather than as_completed because we need
                #  to handle checkpoints and state transitions under the lock)
                time.sleep(1)

                with self._lock:
                    # Check completed futures
                    completed_names = []
                    for name, future in list(futures.items()):
                        if not future.done():
                            continue

                        completed_names.append(name)
                        try:
                            future.result()
                            self.state[name] = StepState.DONE
                            self.timers[name]["end"] = time.time()
                            elapsed = self.timers[name]["end"] - self.timers[name]["start"]
                            if self._display:
                                self._display.update_step(name, StepState.DONE, elapsed=elapsed)

                        except Exception as e:
                            from cyclops_process.pipelinerunner.exceptions import PipelineHalted
                            if isinstance(e, PipelineHalted):
                                raise
                            self._handle_failure(name, e)

                    # Remove completed from futures tracking
                    for name in completed_names:
                        del futures[name]

                    # Handle pending failure decisions (one at a time)
                    if self._failed_queue:
                        failed_name = self._failed_queue[0]
                        # Debug: log that we're about to prompt
                        if self._display:
                            _dbg = self._display.log_dir / "_retry_debug.log"
                            with open(_dbg, "a") as f:
                                f.write(f"[{time.strftime('%H:%M:%S')}] Main loop: about to prompt for {failed_name}\n")
                        # Release lock for interactive prompt
                        self._lock.release()
                        try:
                            action = self._prompt_retry(failed_name)
                        finally:
                            self._lock.acquire()

                        self._failed_queue.pop(0)
                        self._resolve_failure(failed_name, action)

                    # Handle checkpoints whose deps are now satisfied
                    self._handle_ready_checkpoints()

                    # Promote newly ready steps and submit
                    self._promote_ready_steps()
                    self._submit_ready(pool, futures)

    def _promote_ready_steps(self):
        """Move PENDING steps to READY if all deps are satisfied. Must hold lock."""
        for name in self.steps:
            if self.state[name] == StepState.PENDING and self._all_deps_satisfied(name):
                self.state[name] = StepState.READY

    def _submit_ready(self, pool: ThreadPoolExecutor, futures: dict[str, Future]):
        """Submit all READY steps to the thread pool. Must hold lock."""
        for name in list(self.steps.keys()):
            if self.state[name] != StepState.READY:
                continue

            self.state[name] = StepState.RUNNING
            self.timers[name] = {"start": time.time()}
            if self._display:
                self._display.update_step(name, StepState.RUNNING)

            # Audit log start
            step_def = self.steps[name]
            log_key = self.runner._generate_log_key(step_def.func, **step_def.params)
            if not hasattr(step_def.func, "_version"):
                self.runner._audit_log_start(log_key)

            futures[name] = pool.submit(self._execute_step, name)

    def _handle_ready_checkpoints(self):
        """Handle checkpoints whose `after` deps are all satisfied. Must hold lock."""
        for cp_name, cp_def in self.checkpoints.items():
            if self.state.get(cp_name) != StepState.PENDING:
                continue

            # Check if all `after` deps are satisfied
            all_after_done = all(
                self.state.get(d) in (StepState.DONE, StepState.SKIPPED)
                for d in cp_def.after
                if d in self.state
            )

            if not all_after_done:
                continue

            # Release lock for interactive prompt
            self._lock.release()
            try:
                confirmed = self._handle_checkpoint(cp_name)
            finally:
                self._lock.acquire()

            if confirmed:
                self.state[cp_name] = StepState.DONE
            else:
                self.state[cp_name] = StepState.FAILED
                self.runner._abort_all = True

    # ── Error & retry helpers ────────────────────────────────────────────

    @staticmethod
    def _extract_error_summary(e: Exception) -> str:
        """Extract a short error summary for display in the table.

        Format: 'ExceptionType: first line of message' truncated to 240 chars.
        """
        type_name = type(e).__name__
        msg = str(e).split("\n")[0].strip()
        summary = f"{type_name}: {msg}" if msg else type_name
        if len(summary) > 240:
            summary = summary[:237] + "..."
        return summary

    def _prompt_retry(self, name: str) -> str:
        """Prompt user to retry or skip a failed step.

        Called outside the lock so the async loop can continue.
        Returns 'retry' or 'skip'. Times out after 120s (defaults to skip).
        """
        import select

        error_msg = self._extract_error_summary(self.errors[name])
        log_path = self._display.get_step_log_path(name) if self._display else ""
        job_id = self._display.job_ids.get(name, "") if self._display else ""

        if self._display:
            self._display.show_failure_prompt(name, error_msg, str(log_path), job_id)

        # Write debug info to dag log directory
        _debug_path = Path(self._display.log_dir / "_retry_debug.log") if self._display else None
        if _debug_path:
            with open(_debug_path, "a") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] Prompting retry for {name}\n")
                f.write(f"  stdin type: {type(sys.__stdin__)}, fileno: ")
                try:
                    f.write(f"{sys.__stdin__.fileno()}\n")
                except Exception as ex:
                    f.write(f"ERROR: {ex}\n")
                f.write(f"  isatty: {sys.__stdin__.isatty() if hasattr(sys.__stdin__, 'isatty') else 'N/A'}\n")

        try:
            # Wait up to 120s for user input, then default to skip
            stdin_fd = sys.__stdin__
            ready, _, _ = select.select([stdin_fd], [], [], 120)
            if ready:
                response = stdin_fd.readline().strip().lower()
                if _debug_path:
                    with open(_debug_path, "a") as f:
                        f.write(f"  user response: {response!r}\n")
            else:
                response = "s"  # timeout → skip
                if _debug_path:
                    with open(_debug_path, "a") as f:
                        f.write(f"  TIMED OUT after 120s\n")
                if self._display:
                    self._display._terminal.write(
                        "\n  \033[2m(timed out after 120s \u2014 skipping)\033[0m\n"
                    )
                    self._display._terminal.flush()
        except (EOFError, OSError, ValueError) as ex:
            response = "s"
            if _debug_path:
                with open(_debug_path, "a") as f:
                    f.write(f"  EXCEPTION during select/readline: {type(ex).__name__}: {ex}\n")

        if _debug_path:
            with open(_debug_path, "a") as f:
                f.write(f"  final action: {'retry' if response in ('r', 'retry') else 'skip'}\n\n")

        # Resume the display refresh thread (clears prompt, redraws table)
        if self._display:
            self._display.resume_after_prompt()

        if response in ("r", "retry"):
            return "retry"
        return "skip"

    def _handle_failure(self, name: str, e: Exception):
        """Common failure handling: set state, update display, block downstream.

        Must be called while holding self._lock (for async mode) or from
        the main thread (sequential/rerun mode).
        """
        self.state[name] = StepState.FAILED
        self.errors[name] = e
        self.timers[name]["end"] = time.time()
        error_msg = self._extract_error_summary(e)

        if self._display:
            self._display.update_step(name, StepState.FAILED, error_msg=error_msg)

        # Mark downstream as BLOCKED (not SKIPPED — user may retry)
        for downstream in self._get_downstream(name):
            if self.state.get(downstream) in (StepState.PENDING, StepState.READY):
                self.state[downstream] = StepState.BLOCKED
                if self._display:
                    self._display.update_step(downstream, StepState.BLOCKED)

        self._failed_queue.append(name)

    def _resolve_failure(self, name: str, action: str):
        """Apply user's retry/skip decision for a failed step.

        Must be called while holding self._lock (for async mode) or from
        the main thread (sequential/rerun mode).
        """
        if action == "retry":
            self.state[name] = StepState.PENDING
            del self.errors[name]
            self._retry_counts[name] = self._retry_counts.get(name, 0) + 1
            if self._display:
                self._display.update_step(name, StepState.PENDING)
            # Unblock downstream back to PENDING
            for downstream in self._get_downstream(name):
                if self.state.get(downstream) == StepState.BLOCKED:
                    self.state[downstream] = StepState.PENDING
                    if self._display:
                        self._display.update_step(downstream, StepState.PENDING)
        else:
            # Skip: cascade SKIPPED to all blocked downstream
            self.state[name] = StepState.SKIPPED
            if self._display:
                self._display.update_step(name, StepState.SKIPPED)
            for downstream in self._get_downstream(name):
                if self.state.get(downstream) == StepState.BLOCKED:
                    self.state[downstream] = StepState.SKIPPED
                    if self._display:
                        self._display.update_step(downstream, StepState.SKIPPED)

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        """Format elapsed time as H:MM:SS."""
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"
