"""
Nextflow-style persistent ANSI progress table for DAG pipeline execution.

The table stays in the same terminal position and redraws in-place as step
states change. All step output is redirected to per-step log files so nothing
disrupts the table.

A background thread redraws the table every second to update spinners,
elapsed times, and the global clock.

Non-TTY fallback: prints one line per state change.
"""

import os
import sys
import time
import threading
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from io import TextIOWrapper
from pathlib import Path

from cyclops_process.pipelinerunner.dag_runner import StepState


# ── ANSI helpers ────────────────────────────────────────────────────────

def _detect_tty() -> bool:
    """Detect whether the terminal supports ANSI redrawing.

    VSCode's integrated terminal (including over SSH) supports ANSI escapes
    but reports isatty()=False. We detect it via VSCODE_IPC_HOOK_CLI.
    """
    if hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
        return True
    # VSCode integrated terminal over SSH: not a real PTY but supports ANSI
    if os.environ.get("VSCODE_IPC_HOOK_CLI"):
        return True
    return False

_IS_TTY = _detect_tty()

def _ansi(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _IS_TTY else text

def _dim(t: str) -> str:    return _ansi("2", t)
def _bold(t: str) -> str:   return _ansi("1", t)
def _green(t: str) -> str:  return _ansi("32", t)
def _red(t: str) -> str:    return _ansi("31", t)
def _yellow(t: str) -> str: return _ansi("33", t)
def _cyan(t: str) -> str:   return _ansi("36", t)
def _blue(t: str) -> str:   return _ansi("34", t)
def _magenta(t: str) -> str: return _ansi("35", t)


# ── State rendering ─────────────────────────────────────────────────────

_STATE_ICONS = {
    StepState.PENDING:  ("·", "pending",  _dim),
    StepState.READY:    ("\u23f3", "queued",   _yellow),
    StepState.RUNNING:  ("\u25b6", "running",  _cyan),
    StepState.DONE:     ("\u2714", "done",     _green),
    StepState.FAILED:   ("\u2717", "FAILED",   _red),
    StepState.SKIPPED:  ("\u2013", "skipped",  _dim),
    StepState.BLOCKED:  ("\u23f8", "blocked",  _yellow),
}

_SPINNER_CHARS = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"


class DAGDisplay:
    """Nextflow-style persistent ANSI progress table.

    Redraws the full table in-place using ANSI cursor control. All step
    stdout/stderr is redirected to per-step log files.

    A background daemon thread redraws every ~1 second to keep spinners,
    elapsed times, and the global clock up to date.

    Args:
        step_order: Steps in topological order (display order).
        initial_states: Initial state map from preflight scan.
        log_dir: Directory for per-step log files.
    """

    def __init__(
        self,
        step_order: list[str],
        initial_states: dict[str, StepState],
        log_dir: Path,
    ):
        self.step_order = step_order
        self.states: dict[str, StepState] = dict(initial_states)
        self.log_dir = log_dir
        self.elapsed: dict[str, float] = {}       # step -> final elapsed (set on completion)
        self.start_times: dict[str, float] = {}    # step -> wall-clock start (set on RUNNING)
        self.job_ids: dict[str, str] = {}          # step -> SLURM job ID
        self.progress: dict[str, float] = {}       # step -> 0.0-1.0
        self.error_msgs: dict[str, str] = {}      # step -> short error message
        self.slurm_states: dict[str, str] = {}    # step -> raw SLURM state (PENDING, RUNNING, etc.)
        self._start_time = time.time()
        self._lock = threading.Lock()
        self._table_lines = 0
        self._spinner_idx = 0
        self._log_handles: dict[str, TextIOWrapper] = {}
        self._finalized = False
        self._refresh_stop = threading.Event()
        self._refresh_paused = False  # Pause refresh during interactive prompts
        self._prompt_lines = 0  # Extra lines written by show_failure_prompt

        # Capture the real terminal stdout BEFORE any redirect_stdout calls.
        # This ensures the refresh thread always writes to the terminal,
        # even when sys.stdout is temporarily redirected to a log file.
        self._terminal = sys.__stdout__

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def draw_initial(self):
        """Draw the initial table and start the background refresh thread."""
        if _IS_TTY:
            # Save cursor position — all future redraws restore to here
            self._terminal.write("\033[s")
            self._terminal.flush()
            self._redraw()
            # Start background refresh thread (daemon so it dies with main)
            self._refresh_thread = threading.Thread(
                target=self._refresh_loop, daemon=True, name="dag-display-refresh"
            )
            self._refresh_thread.start()
        else:
            n_pending = sum(1 for s in self.states.values() if s == StepState.PENDING)
            n_done = sum(1 for s in self.states.values() if s == StepState.DONE)
            total = len(self.step_order)
            print(f"DAG Pipeline: {total} steps \u2014 {n_done} done, {n_pending} pending")

    def update_step(
        self,
        name: str,
        state: StepState,
        elapsed: float | None = None,
        job_id: str | None = None,
        progress: float | None = None,
        error_msg: str | None = None,
        slurm_state: str | None = None,
    ):
        """Update a step's state and redraw the table."""
        with self._lock:
            self.states[name] = state

            if state == StepState.RUNNING and name not in self.start_times:
                self.start_times[name] = time.time()

            if elapsed is not None:
                self.elapsed[name] = elapsed
            if job_id is not None:
                self.job_ids[name] = job_id
            if progress is not None:
                self.progress[name] = progress
            if error_msg is not None:
                self.error_msgs[name] = error_msg
            if slurm_state is not None:
                self.slurm_states[name] = slurm_state

            if _IS_TTY:
                self._redraw()
            else:
                self._print_state_change(name, state, elapsed, error_msg)

    def show_checkpoint(self, name: str, message: str):
        """Display a checkpoint prompt below the table.

        Pauses the refresh thread. Call resume_after_prompt() after the
        checkpoint is confirmed/dismissed.
        """
        with self._lock:
            self._refresh_paused = True
            if _IS_TTY:
                self._terminal.write(f"\n{_yellow('\u26a0 CHECKPOINT')}: {message}\n")
                self._terminal.flush()
            else:
                print(f"\nCHECKPOINT: {message}")

    def show_failure_prompt(self, name: str, error_msg: str, log_path: str, job_id: str):
        """Display a failure prompt below the table for retry/skip decision.

        Pauses the background refresh thread so it doesn't draw over the prompt.
        Call resume_after_prompt() after getting user input.
        """
        with self._lock:
            # Pause refresh thread so it doesn't draw over the prompt
            self._refresh_paused = True

            if _IS_TTY:
                lines = [
                    "",
                    f"  {_red('━' * 60)}",
                    f"  {_red('\u2717')} {_bold(name)} {_red('FAILED')}: {_red(error_msg)}",
                ]
                if job_id:
                    lines.append(f"    Job: {_dim(job_id)}")
                if log_path:
                    lines.append(f"    Log: {_dim(log_path)}")
                lines.append(f"  {_red('━' * 60)}")
                lines.append(f"    {_yellow('[r]etry')} / {_dim('[s]kip')}: ")
                self._terminal.write("\n".join(lines))
                self._terminal.flush()
                self._prompt_lines = len(lines)
            else:
                print(f"\n  FAILED {name}: {error_msg}")
                if log_path:
                    print(f"    Log: {log_path}")
                print("    [r]etry / [s]kip: ", end="", flush=True)

    def resume_after_prompt(self):
        """Clear the failure prompt area and resume the refresh thread.

        Call this after getting user input from show_failure_prompt().
        """
        with self._lock:
            self._refresh_paused = False
            self._prompt_lines = 0
            if _IS_TTY:
                self._redraw()

    def finalize(self):
        """Stop refresh thread, close log handles, print final table."""
        if self._finalized:
            return
        self._finalized = True

        # Stop the refresh thread
        self._refresh_stop.set()

        # Close all log file handles
        for handle in self._log_handles.values():
            try:
                handle.close()
            except Exception:
                pass

        if _IS_TTY:
            with self._lock:
                self._redraw()
                self._terminal.write("\n")
                self._terminal.flush()

    def get_step_log_path(self, name: str) -> Path:
        """Get the log file path for a step."""
        return self.log_dir / f"{name}.log"

    @contextmanager
    def redirect_step_output(self, name: str):
        """Context manager to redirect stdout/stderr to step log file."""
        log_path = self.get_step_log_path(name)
        log_handle = open(log_path, "w", buffering=1)
        self._log_handles[name] = log_handle

        try:
            with redirect_stdout(log_handle), redirect_stderr(log_handle):
                yield log_handle
        finally:
            log_handle.flush()

    # ── Background refresh ──────────────────────────────────────────────

    def _refresh_loop(self):
        """Background thread: redraw table every ~1s to update spinners and elapsed."""
        while not self._refresh_stop.is_set():
            self._refresh_stop.wait(timeout=1.0)
            if self._finalized:
                break
            with self._lock:
                if _IS_TTY and not self._finalized and not self._refresh_paused:
                    self._redraw()

    # ── Private rendering ───────────────────────────────────────────────

    @staticmethod
    def _visible_len(s: str) -> int:
        """Return the visible (non-ANSI) length of a string."""
        import re
        return len(re.sub(r"\033\[[0-9;]*m", "", s))

    def _physical_line_count(self, lines: list[str], term_width: int) -> int:
        """Count physical terminal lines, accounting for line wrapping."""
        if term_width <= 0:
            return len(lines)
        count = 0
        for line in lines:
            w = self._visible_len(line)
            # Each line takes at least 1 row; wrapping adds extra rows
            count += max(1, -(-w // term_width))  # ceil division
        return count

    def _redraw(self):
        """Clear and redraw the full table using ANSI cursor control.

        Must be called while holding self._lock.

        Uses save/restore cursor position (like Nextflow) instead of
        relative cursor-up math, which breaks on terminal resize.
        """
        lines = self._build_table()

        buf = []
        buf.append("\033[u")   # restore cursor to saved position (set in draw_initial)
        buf.append("\033[J")   # clear everything below
        buf.append("\n".join(lines))

        self._terminal.write("".join(buf))
        self._terminal.flush()

    def _build_table(self) -> list[str]:
        """Build the table lines.

        When the table would exceed terminal height, pending steps are
        collapsed into a single summary line to prevent ANSI cursor overflow.
        """
        lines = []
        now = time.time()
        total_elapsed = now - self._start_time
        self._spinner_idx = (self._spinner_idx + 1) % len(_SPINNER_CHARS)

        # Determine terminal height budget
        try:
            term_height = os.get_terminal_size().lines
        except (OSError, ValueError):
            term_height = 50
        # Header (3) + footer (4) + blank = 8 overhead lines
        max_step_rows = term_height - 8

        # Header
        timestamp = time.strftime("%H:%M:%S")
        lines.append("")
        lines.append(f"  {_bold('OPS Pipeline')}  {_dim(f'[{timestamp}]')}")
        lines.append(f"  {_dim('\u2500' * 72)}")

        # Step rows — if too many, hide pending steps
        name_width = max(len(n) for n in self.step_order) if self.step_order else 20
        total_steps = len(self.step_order)
        need_collapse = total_steps > max_step_rows

        # Separate active vs pending steps if we need to collapse
        if need_collapse:
            active_names = []
            pending_names = []
            for name in self.step_order:
                state = self.states.get(name, StepState.PENDING)
                if state == StepState.PENDING:
                    pending_names.append(name)
                else:
                    active_names.append(name)
            # Show all active, fill remaining slots with pending
            remaining = max(0, max_step_rows - len(active_names) - 1)  # -1 for collapse line
            visible_pending = pending_names[:remaining]
            hidden_pending = len(pending_names) - remaining
            visible_names = set(active_names) | set(visible_pending)
        else:
            visible_names = set(self.step_order)
            hidden_pending = 0

        for name in self.step_order:
            if name not in visible_names:
                continue
            state = self.states.get(name, StepState.PENDING)
            icon, label, color_fn = _STATE_ICONS.get(state, ("?", "unknown", _dim))

            # Progress bar
            prog = self.progress.get(name, None)
            if state == StepState.DONE:
                bar = _green("\u2588" * 20)
                pct = "100%"
            elif state == StepState.RUNNING:
                if prog is not None:
                    filled = max(1, int(prog * 20)) if prog > 0 else 0
                    bar = _cyan("\u2588" * filled + "\u2591" * (20 - filled))
                    pct = f"{int(prog * 100):>3d}%"
                else:
                    spinner = _SPINNER_CHARS[self._spinner_idx]
                    bar = _cyan(f"{spinner}{'\u2591' * 19}")
                    pct = "  \u2014 "
            elif state == StepState.FAILED:
                bar = _red("\u2588" * 20)
                pct = "ERR "
            else:
                bar = _dim("\u2591" * 20)
                pct = "  \u2014 "

            # Elapsed time — live for running steps, final for done
            step_elapsed = self.elapsed.get(name)
            if step_elapsed is not None:
                elapsed_str = self._format_elapsed(step_elapsed)
            elif state == StepState.RUNNING and name in self.start_times:
                live_elapsed = now - self.start_times[name]
                elapsed_str = self._format_elapsed(live_elapsed)
            else:
                elapsed_str = "  \u2014  "

            # Job ID suffix
            jid = self.job_ids.get(name)
            jid_str = _dim(f" [{jid}]") if jid else ""

            # Error message suffix for FAILED steps
            err_str = ""
            if state == StepState.FAILED and name in self.error_msgs:
                err_str = " " + _red(self.error_msgs[name])

            # Show SLURM state for running steps (e.g., "pending", "1/2 pyramids 12/45")
            if state == StepState.RUNNING and name in self.slurm_states:
                slurm_st = self.slurm_states[name].lower()
                # Truncate the label prefix but always keep the job count visible.
                # Format is typically "<label> <n/total>" or "<phase> <label> <n/total>"
                # Split off the count suffix (last token with "/") to preserve it.
                parts = slurm_st.rsplit(" ", 1)
                if len(parts) == 2 and "/" in parts[1]:
                    prefix, count = parts
                    max_prefix = 20
                    if len(prefix) > max_prefix:
                        prefix = prefix[:max_prefix - 1] + "…"
                    slurm_st = f"{prefix} {count}"
                elif len(slurm_st) > 25:
                    slurm_st = slurm_st[:22] + "..."
                display_label = f"{icon} {slurm_st}"
            else:
                display_label = f"{icon} {label:<10}"
            colored_label = color_fn(display_label)
            lines.append(
                f"  {name:<{name_width}}  [{bar}] {pct}  {colored_label} {elapsed_str}{jid_str}{err_str}"
            )

        # Collapsed pending summary
        if hidden_pending > 0:
            lines.append(f"  {_dim(f'  ... {hidden_pending} pending steps hidden (terminal too short)')}")

        # Footer
        lines.append(f"  {_dim('\u2500' * 72)}")

        n_running = sum(1 for s in self.states.values() if s == StepState.RUNNING)
        n_queued = sum(1 for s in self.states.values() if s == StepState.READY)
        n_done = sum(1 for s in self.states.values() if s == StepState.DONE)
        n_failed = sum(1 for s in self.states.values() if s == StepState.FAILED)
        n_blocked = sum(1 for s in self.states.values() if s == StepState.BLOCKED)
        total = len(self.step_order)

        parts = [
            f"Running: {_cyan(str(n_running))}",
            f"Queued: {_yellow(str(n_queued))}",
            f"Done: {_green(f'{n_done}/{total}')}",
        ]
        if n_failed:
            parts.append(f"Failed: {_red(str(n_failed))}")
        if n_blocked:
            parts.append(f"Blocked: {_yellow(str(n_blocked))}")

        elapsed_str = self._format_elapsed(total_elapsed)
        lines.append(f"  {' \u00b7 '.join(parts)}    Elapsed: {_bold(elapsed_str)}")

        lines.append(f"  {_dim(f'Logs: {self.log_dir}/')}")
        lines.append("")

        return lines

    def _print_state_change(self, name: str, state: StepState, elapsed: float | None, error_msg: str | None = None):
        """Print a single state change line (non-TTY mode)."""
        icon, label, _ = _STATE_ICONS.get(state, ("?", "unknown", _dim))
        elapsed_str = f" ({self._format_elapsed(elapsed)})" if elapsed else ""
        err_str = f" — {error_msg}" if error_msg else ""
        print(f"  {icon} {name}: {label}{elapsed_str}{err_str}")

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        """Format elapsed time as H:MM:SS or M:SS."""
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"
