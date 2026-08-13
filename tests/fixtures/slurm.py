"""SLURM submission helpers for real-data stage tests.

`submit_stage(stage, fn, log_dir, **kwargs)` packages a single pipeline-stage
invocation into a SLURM job whose resource request matches the corresponding
`slurm_params` block in `nextflow/nextflow_ops_args.yaml`. The job's env
is inherited from the calling process (submitit defaults to --export=ALL),
so OPS_OUTPUT_BASE_DIR / OPS_FAST_OUTPUT_BASE_DIR / OPS_CONFIGS_DIR set on
the test process via `monkeypatch.setenv` propagate to the job.

The function call blocks on `job.result()` until SLURM completes the job.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

import yaml
import submitit


NF_ARGS_YAML = (
    Path(__file__).resolve().parent.parent.parent
    / "nextflow" / "nextflow_ops_args.yaml"
)
# iss.config is the production source of truth for SLURM clusterOptions (gres,
# constraint). The yaml's per-stage slurm_params.clusterOptions is IGNORED by
# Nextflow for most stages; iss.config sets it in each `withName:` block (a
# literal, or a `params.processes.<X>.slurm_params.clusterOptions` reference).
# We replicate that here so test jobs request the same GPUs production does.
ISS_CONFIG = (
    Path(__file__).resolve().parent.parent.parent
    / "nextflow" / "iss.config"
)


def _parse_mem_gb(mem: str) -> int:
    """Convert memory strings like '400G', '16GB', '500MB' to integer GB."""
    m = re.match(r"^\s*(\d+)\s*([KMGT]B?)?\s*$", mem.strip(), re.IGNORECASE)
    if not m:
        raise ValueError(f"unparseable memory string: {mem!r}")
    n = int(m.group(1))
    unit = (m.group(2) or "G").upper().rstrip("B")
    factor = {"K": 1 / (1024 * 1024), "M": 1 / 1024, "G": 1, "T": 1024}[unit]
    return max(1, int(round(n * factor)))


def _parse_cluster_options(opts: str | None) -> dict[str, str]:
    """Parse Nextflow clusterOptions into submitit slurm_additional_parameters.

    Example: '--gres=gpu:2 --constraint=[h100|h200]' ->
        {'gres': 'gpu:2', 'constraint': '[h100|h200]'}
    """
    if not opts:
        return {}
    out: dict[str, str] = {}
    for token in opts.split():
        token = token.lstrip("-")
        key, _, value = token.partition("=")
        out[key] = value
    return out


def load_stage_config(stage: str) -> dict[str, Any]:
    with open(NF_ARGS_YAML) as f:
        cfg = yaml.safe_load(f)
    procs = cfg.get("processes") or {}
    if stage not in procs:
        raise KeyError(
            f"stage {stage!r} not found in {NF_ARGS_YAML} "
            f"(known: {sorted(procs.keys())[:5]}...)"
        )
    return procs[stage]


def slurm_params_for(stage: str) -> dict[str, Any]:
    return load_stage_config(stage).get("slurm_params") or {}


def _params_gpu_constraint() -> str | None:
    """Top-level `gpu_constraint` from the args yaml — the default architecture
    constraint the iss.config clusterOptions closure applies to any GPU step that
    doesn't set its own `ext.gpu_constraint`."""
    with open(NF_ARGS_YAML) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("gpu_constraint")


def cluster_options_for(stage: str) -> str | None:
    """Resolve a stage's effective SLURM clusterOptions the way production does.

    Reads iss.config's `withName: <stage>` block:
      - `clusterOptions = "<literal>"`  -> the literal (inner quotes normalized)
      - `clusterOptions = params.processes.<X>.slurm_params.clusterOptions`
                                        -> resolved from the yaml for <X>
      - no clusterOptions line          -> None (no gres requested)

    Falls back to the yaml's own slurm_params.clusterOptions if iss.config has
    no `withName` block for the stage (shouldn't happen for pipeline stages).
    """
    try:
        text = ISS_CONFIG.read_text()
    except OSError:
        return slurm_params_for(stage).get("clusterOptions")

    block = re.search(
        rf"withName:\s*{re.escape(stage)}\s*\{{(.*?)\n\s*\}}",
        text,
        re.DOTALL,
    )
    if not block:
        return slurm_params_for(stage).get("clusterOptions")

    body = block.group(1)
    m = re.search(r"clusterOptions\s*=\s*(.+)", body)
    if m:
        rhs = m.group(1).strip()
        ref = re.match(r"params\.processes\.(\w+)\.slurm_params\.clusterOptions", rhs)
        if ref:
            return (load_stage_config(ref.group(1)).get("slurm_params") or {}).get(
                "clusterOptions"
            )
        # Quoted literal (single or double outer quotes). Strip the outer quotes,
        # then drop inner double-quotes so `--constraint="[h100|h200]"` becomes the
        # bare `--constraint=[h100|h200]` form that _parse_cluster_options/sbatch
        # accept cleanly.
        lit = rhs.split("//", 1)[0].strip()  # tolerate trailing inline comments
        if len(lit) >= 2 and lit[0] in "\"'" and lit[-1] == lit[0]:
            lit = lit[1:-1]
        return lit.replace('"', "") or None

    # Current iss.config form: GPU stages set `ext.gpus = N` (+ optionally
    # `ext.gpu_constraint`); the process-scope clusterOptions closure turns that into
    # `--gres=gpu:N --constraint=<ext.gpu_constraint OR params.gpu_constraint>`. CPU
    # stages set neither -> no GPU request. Replicate that here.
    gm = re.search(r"ext\.gpus\s*=\s*(\d+)", body)
    if not gm:
        return None  # CPU stage (no gres requested)
    cm = re.search(r"""ext\.gpu_constraint\s*=\s*['"]([^'"]+)['"]""", body)
    constraint = cm.group(1) if cm else _params_gpu_constraint()
    opts = f"--gres=gpu:{gm.group(1)}"
    if constraint:
        opts += f" --constraint={constraint}"
    return opts


def python_kwargs_for(stage: str) -> dict[str, Any]:
    return load_stage_config(stage).get("python_kwargs") or {}


def make_executor(stage: str, log_dir: Path) -> submitit.AutoExecutor:
    sp = slurm_params_for(stage)
    mem = sp.get("mem") or sp.get("memory") or "16G"
    extra = _parse_cluster_options(cluster_options_for(stage))
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=str(log_dir))
    params: dict[str, Any] = dict(
        name=f"pytest-{stage}",
        timeout_min=int(sp.get("timeout_min", 60)),
        slurm_partition=sp.get("slurm_partition", "cpu"),
        cpus_per_task=int(sp.get("cpus", 4)),
        mem_gb=_parse_mem_gb(str(mem)),
    )
    if extra:
        params["slurm_additional_parameters"] = extra
    executor.update_parameters(**params)
    return executor


def submit_stage(
    stage: str,
    fn: Callable[..., Any],
    log_dir: Path,
    *,
    merge_yaml_kwargs: bool = True,
    **kwargs: Any,
) -> Any:
    """Submit `fn(**kwargs)` to SLURM with resource params from the yaml.

    If `merge_yaml_kwargs` (default), the stage's `python_kwargs` block from
    nextflow_ops_args.yaml is used as a base; explicit `kwargs` override
    yaml values key-for-key. Set False to bypass yaml kwargs entirely.

    Returns the function's return value. Raises submitit's JobError-family
    exceptions if the job fails; the submitit folder retains stdout/stderr
    plus the pickled traceback for debugging.
    """
    merged: dict[str, Any] = (
        {**python_kwargs_for(stage), **kwargs} if merge_yaml_kwargs else dict(kwargs)
    )
    executor = make_executor(stage, log_dir)
    job = executor.submit(fn, **merged)
    return job.result()
