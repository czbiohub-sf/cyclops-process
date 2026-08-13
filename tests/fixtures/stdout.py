"""Stdout / dry-run helpers for orchestrator-stage real-data tests.

Some pipeline stages are *stdout-driven*: they print a value Nextflow captures
from the job's stdout (e.g. ``get_n_positions`` prints a position count that
fans out into per-position jobs), or they are *dry-run orchestrators* that,
given ``dry_run=True``, report the set of jobs they WOULD submit instead of
submitting them. These stages have no single output artifact to diff, so the
``compare_*`` family does not apply; we assert on their printed/returned plan.

- ``run_and_capture`` submits a stage to SLURM (like ``submit_stage``) but
  returns the job's captured stdout text instead of its return value.
- ``assert_dry_run_plan`` asserts on a dry-run orchestrator's plan, accepting
  either captured stdout text or the orchestrator's structured return value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

from fixtures.slurm import make_executor, python_kwargs_for


def run_and_capture(
    stage: str,
    fn: Callable[..., Any],
    log_dir: Path,
    *,
    merge_yaml_kwargs: bool = True,
    **kwargs: Any,
) -> str:
    """Submit ``fn(**kwargs)`` to SLURM and return the stage's captured stdout.

    Mirrors ``fixtures.slurm.submit_stage`` (same resource resolution and yaml
    kwargs merge) but, after blocking on completion, returns the job's stdout
    text rather than its return value. Use for stdout-driven stages whose
    contract is "print X". Raises submitit's JobError-family exceptions if the
    job fails (the submitit folder retains stdout/stderr + pickled traceback).

    Two things make the captured text usable:

    - The job is run with ``PYTHONUNBUFFERED=1`` (exported to the SLURM job via
      submitit's --export=ALL). Without it, a bare ``print(...)`` in a non-TTY
      batch job is block-buffered and may never reach submitit's ``*_log.out``,
      so only submitit's own logging lines would be captured.
    - submitit's own log lines (``submitit INFO ...`` / ``submitit WARNING ...``)
      are stripped from the returned text, leaving just the stage's stdout.
    """
    import os

    merged: dict[str, Any] = (
        {**python_kwargs_for(stage), **kwargs} if merge_yaml_kwargs else dict(kwargs)
    )
    executor = make_executor(stage, log_dir)
    prev_unbuffered = os.environ.get("PYTHONUNBUFFERED")
    os.environ["PYTHONUNBUFFERED"] = "1"  # propagates to the job at submit time
    try:
        job = executor.submit(fn, **merged)
    finally:
        if prev_unbuffered is None:
            os.environ.pop("PYTHONUNBUFFERED", None)
        else:
            os.environ["PYTHONUNBUFFERED"] = prev_unbuffered
    job.result()  # block until done; propagate exceptions

    # job.result() returns once the result pickle is visible, but on a shared
    # (VAST) filesystem the captured *_log.out tail can lag behind by a few
    # seconds -- a fast job is usually synced, a slower one is not, so reading
    # stdout immediately can miss the stage's print(). submitit writes a
    # "Job completed successfully" line AFTER the function's stdout, so poll
    # until that marker is visible (bounded), guaranteeing the print() is too.
    import time

    raw = job.stdout() or ""
    deadline = 60  # seconds; generous for FS sync, never an infinite hang
    waited = 0
    while "completed successfully" not in raw and waited < deadline:
        time.sleep(2)
        waited += 2
        raw = job.stdout() or ""

    # Remove submitit's own log RECORDS wherever they appear -- not just at line
    # start. A stage that prints with end="" (e.g. "{n_jobs} {n_positions}")
    # leaves submitit's next log record concatenated on the same line, so a
    # line-prefix filter would keep it. Stripping the record pattern handles
    # both end="" and newline-terminated prints, leaving only the stage stdout.
    import re

    cleaned = re.sub(
        r"submitit (?:INFO|WARNING|ERROR|DEBUG|CRITICAL) \([^)]*\) - [^\n]*",
        "",
        raw,
    )
    return cleaned.strip()


def assert_dry_run_plan(
    result: Any,
    *,
    expect_jobs: int | None = None,
    contains: str | Iterable[str] | None = None,
) -> None:
    """Assert on the plan reported by a ``dry_run=True`` orchestrator.

    ``result`` may be either the orchestrator's structured return value (a list
    or dict of planned jobs) or its captured stdout text. The plan is reduced to
    a count and a text form so the same assertion works regardless of shape:

    - ``expect_jobs``: the exact number of planned jobs. For a list/dict result
      this is ``len(result)``; for stdout text it is the number of non-blank
      lines (override by passing text you have already filtered).
    - ``contains``: a substring or iterable of substrings that must each appear
      in the plan's text form.

    Raises AssertionError if a constraint is not met.
    """
    if isinstance(result, str):
        text = result
        n_jobs = len([ln for ln in result.splitlines() if ln.strip()])
    elif isinstance(result, dict):
        text = "\n".join(f"{k}: {v}" for k, v in result.items())
        n_jobs = len(result)
    elif isinstance(result, (list, tuple)):
        text = "\n".join(str(item) for item in result)
        n_jobs = len(result)
    else:
        text = str(result)
        n_jobs = None

    if expect_jobs is not None:
        if n_jobs is None:
            raise AssertionError(
                f"cannot count jobs in dry-run result of type {type(result).__name__}: "
                f"{result!r}"
            )
        if n_jobs != expect_jobs:
            raise AssertionError(
                f"dry-run plan job count mismatch: expected {expect_jobs}, got "
                f"{n_jobs}\n  plan:\n{text}"
            )

    if contains is not None:
        needles = [contains] if isinstance(contains, str) else list(contains)
        missing = [n for n in needles if n not in text]
        if missing:
            raise AssertionError(
                f"dry-run plan missing expected substring(s): {missing}\n"
                f"  plan:\n{text}"
            )
