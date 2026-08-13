"""Every GPU stage in iss.config must resolve a SLURM ``--constraint``.

Encodes A. Hillsley's GPU-scheduling consolidation (the single ``ext.gpus`` /
``ext.gpu_constraint`` + ``params.gpu_constraint`` allowlist closure in
``nextflow/iss.config``) together with the gpu-i-1 / Blackwell lesson: a GPU
job submitted without an architecture constraint can land on a node whose CuPy
build can't run it (silent CPU fallback at best, ``NO_BINARY_FOR_GPU`` crash at
worst). So any process that requests GPUs (``ext.gpus``) must carry a
``--constraint`` — either its own ``ext.gpu_constraint`` or the global
``params.gpu_constraint`` fallback.

These are unit tests (no SLURM, no GPU): they parse iss.config and replay the
same resolution the production clusterOptions closure performs, via the
harness's ``cluster_options_for``.
"""

import re


def _gpu_stages():
    """Process names in iss.config whose `withName` block sets `ext.gpus = N`."""
    from fixtures.slurm import ISS_CONFIG

    text = ISS_CONFIG.read_text()
    stages = []
    for m in re.finditer(r"withName:\s*(\w+)\s*\{(.*?)\n\s*\}", text, re.DOTALL):
        name, body = m.group(1), m.group(2)
        if re.search(r"ext\.gpus\s*=\s*\d+", body):
            stages.append(name)
    return stages


def test_gpu_stages_present():
    # Guard against the regex silently matching nothing (which would make the
    # per-stage check below vacuously pass).
    assert _gpu_stages(), "expected iss.config to define at least one ext.gpus (GPU) stage"


def test_global_gpu_constraint_allowlist_is_set():
    from fixtures.slurm import _params_gpu_constraint

    assert _params_gpu_constraint(), (
        "params.gpu_constraint (the global arch allowlist) is unset; any GPU stage "
        "without its own ext.gpu_constraint would request a bare --gres and could "
        "land on an incompatible GPU arch (cf. gpu-i-1 / Blackwell CuPy incompat)."
    )


def test_every_gpu_stage_resolves_a_constraint():
    from fixtures.slurm import cluster_options_for

    offenders = {}
    for stage in _gpu_stages():
        opts = cluster_options_for(stage) or ""
        if "--gres=gpu:" not in opts or "--constraint=" not in opts:
            offenders[stage] = opts
    assert not offenders, (
        "these GPU stages do not resolve to '--gres=gpu:N --constraint=...' — a bare "
        f"--gres can land on an incompatible GPU arch: {offenders}"
    )
