"""Shared pytest fixtures for cyclops_process tests.

The fixtures here support the *real-data* test mode used by tests marked
`@pytest.mark.real_data`. Those tests read a read-only reference cache under
`$OPS_REFERENCE_BASE` (see `reference_cache`), submit a single pipeline stage
to SLURM via submitit, and compare the stage's output against the cached
"known good" output.

The cache directory is never written to; tests symlink upstream stage
outputs from the cache into a writable `tmp_path/work` tree, and the stage
under test writes its own outputs into that tree.
"""

from __future__ import annotations

import getpass
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable

import pytest


# The candidate experiment name (drives the candidate workdir + seed_from_cache
# default). When OPS_REFERENCE_DIR points at an alternate reference experiment
# (e.g. .../ops0161_20260521), derive the name from its basename so the
# candidate runs under that experiment's identity/geometry; otherwise default
# to ops0094. (OPS_REFERENCE_DIR only redirects the reference PATH; this keeps
# the candidate experiment name in sync with it.)
_REFERENCE_DIR_OVERRIDE = os.environ.get("OPS_REFERENCE_DIR")
REFERENCE_EXPERIMENT = (
    Path(_REFERENCE_DIR_OVERRIDE).name if _REFERENCE_DIR_OVERRIDE else "ops0094_20251217"
)
# Reference cache root for real-data tests. No site default: set $OPS_REFERENCE_BASE
# (or $OPS_REFERENCE_DIR for a specific experiment dir) to enable those tests.
DEFAULT_REFERENCE_BASE = os.environ.get("OPS_BASE_PATH")


@pytest.fixture
def shared_tmp_path(request) -> Path:
    """tmp_path equivalent on a shared filesystem reachable from compute nodes.

    pytest's `tmp_path` lives under /tmp, which is per-node on many clusters:
    SLURM jobs running on compute nodes cannot write logs/outputs back to the
    submit node's /tmp. This fixture provides a writable directory under
    $OPS_TEST_TMP_BASE (required, e.g. a scratch dir visible to compute nodes)
    and removes it at test teardown.
    """
    tmp_base = os.environ.get("OPS_TEST_TMP_BASE")
    if not tmp_base:
        pytest.skip(
            "set $OPS_TEST_TMP_BASE to a directory visible to compute nodes "
            "(e.g. your scratch space) to run tests that submit SLURM jobs"
        )
    base = Path(tmp_base) / getpass.getuser()
    base.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(
        c if c.isalnum() or c in "._-" else "_" for c in request.node.name
    )
    workdir = base / f"{safe_name}_{uuid.uuid4().hex[:8]}"
    workdir.mkdir(parents=True)
    yield workdir
    # On failure, preserve the submitit logs (stdout/stderr/pickled traceback)
    # before deleting the (large) workdir -- otherwise the only record of WHY a
    # SLURM job failed is rmtree'd away. Set OPS_TEST_KEEP_WORKDIR=1 to keep the
    # whole tree.
    rep = getattr(request.node, "rep_call", None)
    failed = rep is not None and rep.failed
    if failed:
        logs_src = workdir / "work" / "submitit_logs"
        if logs_src.exists():
            dest = base / "_failed_logs" / safe_name
            shutil.rmtree(dest, ignore_errors=True)
            try:
                shutil.copytree(logs_src, dest)
            except OSError:
                pass
    if os.environ.get("OPS_TEST_KEEP_WORKDIR") and failed:
        return
    shutil.rmtree(workdir, ignore_errors=True)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Expose each phase's report on the item so fixtures can detect failure
    during teardown (used by shared_tmp_path to preserve submitit logs)."""
    outcome = yield
    setattr(item, "rep_" + outcome.get_result().when, outcome.get_result())


@pytest.fixture(scope="session")
def reference_cache() -> Path:
    """Path to the read-only reference experiment directory.

    Resolves the experiment subdir under `$OPS_REFERENCE_BASE`, falling back to
    `$OPS_BASE_PATH`. Skips the test if neither is set or the directory is not
    accessible (e.g., running off-cluster).
    """
    # OPS_REFERENCE_DIR lets us point at an alternate reference EXPERIMENT dir
    # (e.g. <base>/single_ISS) whose directory name need not match the experiment
    # name used for the candidate workdir/config.
    override = os.environ.get("OPS_REFERENCE_DIR")
    if override:
        cache = Path(override)
    else:
        base = os.environ.get("OPS_REFERENCE_BASE", DEFAULT_REFERENCE_BASE)
        if not base:
            pytest.skip(
                "set $OPS_REFERENCE_BASE (or $OPS_BASE_PATH) to the reference "
                "storage root to run real-data tests"
            )
        cache = Path(base) / REFERENCE_EXPERIMENT
    if not cache.exists():
        pytest.skip(f"reference cache not found at {cache}")
    if not cache.is_dir():
        pytest.skip(f"reference cache at {cache} is not a directory")
    return cache


@pytest.fixture(scope="session")
def reference_ngff_version(reference_cache) -> str:
    """Detect the NGFF version of the reference cache's zarr stores.

    Stages that internally read cached zarrs (e.g. stack_symlinks indexes
    chunk paths by version) must be invoked with the cache's version to
    resolve those paths correctly. Fresh writes can still use a different
    version; the comparator handles cross-version compares on the read side.
    """
    probe_dir = reference_cache / "0-convert" / "in_situ_sequencing"
    probes = sorted(probe_dir.glob("A*.zarr"))
    if not probes:
        pytest.skip(f"no zarr stores under {probe_dir} to probe NGFF version")
    probe = probes[0]
    if (probe / "zarr.json").exists():
        return "0.5"
    if (probe / ".zgroup").exists() or (probe / ".zarray").exists():
        return "0.4"
    pytest.skip(f"could not determine NGFF version for {probe}")


@pytest.fixture(scope="session")
def reference_tile_size(reference_cache) -> tuple[int, int]:
    """Detect the (Y, X) tile size of the reference cache's convert tiles.

    Stages that build output array shapes from a `tile_size` parameter
    (e.g. stack_symlinks) must be invoked with the reference's geometry —
    ops0094 used 2048x2048, ops0161 uses 2304x2304. Reads one per-round
    convert tile's level-0 array metadata (v3 zarr.json or v2 .zarray).
    """
    import json

    probe_dir = reference_cache / "0-convert" / "in_situ_sequencing"
    tiles = sorted(probe_dir.glob("A*.zarr"))
    if not tiles:
        pytest.skip(f"no convert tiles under {probe_dir} to probe tile size")
    # descend to a position's level-0 array
    for arr_dir in sorted(tiles[0].glob("*/*/*/0")):
        zj, za = arr_dir / "zarr.json", arr_dir / ".zarray"
        meta = None
        if zj.exists():
            meta = json.load(open(zj))
        elif za.exists():
            meta = json.load(open(za))
        if meta and "shape" in meta:
            shape = meta["shape"]
            return (int(shape[-2]), int(shape[-1]))
    pytest.skip(f"could not determine tile size from {tiles[0]}")


@pytest.fixture(scope="session")
def reference_n_rounds(reference_cache) -> int:
    """ISS sequencing-round count of the reference (ops0094=8, ops0161=9).

    `n_rounds` = ISS rounds EXCLUDING round 0 (the nuclear reference). The v3
    registered store's leading (T) dimension counts ALL rounds INCLUDING round 0,
    so n_rounds = T - 1 (e.g. ops0161 T=10 -> 9, giving reads.csv Q_0..Q_9). Passing
    T directly makes base_calling read one round past the end (OUT_OF_RANGE).
    """
    import json

    store = (
        reference_cache
        / "1-preprocess/in_situ_sequencing/register/bc_stitched_registered_v3.zarr"
    )
    for arr in sorted(store.glob("A/*/*/0")):
        for m in (arr / "zarr.json", arr / ".zarray"):
            if m.exists():
                return int(json.load(open(m))["shape"][0]) - 1
    pytest.skip(f"could not determine n_rounds from {store}")


@pytest.fixture
def real_data_workdir(shared_tmp_path, monkeypatch, reference_cache) -> Path:
    """Writable OPS_OUTPUT_BASE_DIR on a shared FS, scoped to a single test.

    The work tree mirrors the OpsDataset layout: `<shared_tmp>/work/<experiment>/`.
    OPS env vars are set so any `OpsDataset(REFERENCE_EXPERIMENT)` constructed
    inside the test (including inside SLURM jobs submitted from the test)
    points at this work tree.

    Use `seed_from_cache(work, reference_cache, [<subpaths>])` to populate
    upstream-stage outputs as read-only symlinks before invoking the stage.
    """
    work = shared_tmp_path / "work"
    (work / REFERENCE_EXPERIMENT / "logs").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OPS_OUTPUT_BASE_DIR", str(work))
    monkeypatch.setenv("OPS_FAST_OUTPUT_BASE_DIR", str(work))
    monkeypatch.setenv(
        "OPS_CONFIGS_DIR",
        os.environ.get("OPS_CONFIGS_DIR", f"{os.environ['OPS_BASE_PATH']}/configs"),
    )
    return work


def seed_from_cache(
    workdir: Path,
    cache: Path,
    paths: list[str],
    *,
    experiment: str = REFERENCE_EXPERIMENT,
) -> None:
    """Symlink files/dirs from `cache` into `workdir/<experiment>`.

    Each entry in `paths` is a subpath relative to the experiment root.

    - If the entry contains glob metacharacters (`*`, `?`, `[`), it is expanded
      against `cache` (NOT `cache/<experiment>`, so the entry can include the
      experiment-relative dir prefix). Each match is symlinked individually,
      with parent directories created as real dirs so the workdir remains
      writable around the symlinks.
    - For non-glob paths pointing at a directory (excluding `.zarr` stores),
      the workdir destination is created as a real directory and each direct
      child is symlinked. This prevents stages from writing through a
      directory-level symlink when input and output share a parent (e.g.
      stack_symlinks reads + writes under `0-convert/in_situ_sequencing`).
    - For non-glob paths pointing at a file or `.zarr` store, the target is
      symlinked as a single unit.

    Already-present destinations are skipped (idempotent across reseeds).
    """
    work_exp = workdir / experiment
    for entry in paths:
        if any(ch in entry for ch in "*?["):
            matches = sorted(cache.glob(entry))
            if not matches:
                raise FileNotFoundError(
                    f"cache glob {entry!r} matched nothing under {cache}"
                )
            for src in matches:
                rel = src.relative_to(cache)
                dst = work_exp / rel
                if dst.exists() or dst.is_symlink():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.symlink_to(src)
            continue

        src = cache / entry
        if not src.exists():
            raise FileNotFoundError(f"cache missing expected input: {src}")
        dst = work_exp / entry
        if dst.exists() or dst.is_symlink():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir() and not src.name.endswith(".zarr"):
            dst.mkdir()
            for child in src.iterdir():
                (dst / child.name).symlink_to(child)
        else:
            dst.symlink_to(src)


def seed_writable_store(
    workdir: Path,
    cache: Path,
    store_relpath: str,
    *,
    experiment: str = REFERENCE_EXPERIMENT,
) -> Path:
    """Materialize an OME-Zarr store with REAL dirs + REAL metadata files but
    SYMLINKED chunk data, so a stage can write metadata IN PLACE (e.g.
    viscy_normalize's per-position ``custom_metadata.normalization``) without
    copying the (huge) pixel data and without writing through to the read-only
    reference.

    v3-aware: chunk dirs (``c``) are symlinked wholesale; ``zarr.json`` / ``.z*``
    metadata files are copied (writable); any other leaf is symlinked. Returns the
    destination store path.
    """
    import os
    import shutil

    META = {"zarr.json", ".zattrs", ".zgroup", ".zarray", ".zmetadata"}
    src_root = cache / store_relpath
    if not src_root.exists():
        raise FileNotFoundError(f"cache missing store: {src_root}")
    dst_root = workdir / experiment / store_relpath
    for dirpath, dirnames, filenames in os.walk(src_root):
        rel = Path(dirpath).relative_to(src_root)
        (dst_root / rel).mkdir(parents=True, exist_ok=True)
        # Symlink chunk dirs ('c' = v3 chunk grid) wholesale; don't descend.
        for d in list(dirnames):
            if d == "c":
                (dst_root / rel / d).symlink_to(src_root / rel / d)
                dirnames.remove(d)
        for f in filenames:
            s, t = src_root / rel / f, dst_root / rel / f
            if t.exists() or t.is_symlink():
                continue
            if f in META:
                shutil.copy2(s, t)  # real + writable
            else:
                t.symlink_to(s)  # chunk file / other leaf
    return dst_root


def seed_dir_skeleton(
    workdir: Path,
    cache: Path,
    store_relpath: str,
    *,
    depth: int = 3,
    experiment: str = REFERENCE_EXPERIMENT,
) -> Path:
    """Mirror a reference store's DIRECTORY tree as REAL (empty) dirs, down to
    ``depth`` levels below the store root. No files, metadata, or chunk data are
    created -- only directories.

    Purpose: some fan-out *setup* stages enumerate positions purely by directory
    structure rather than by opening the zarr -- e.g.
    ``virtual_staining_inference_setup`` runs
    ``find <store> -mindepth 3 -maxdepth 3 -type d`` to count HCS position dirs
    (row/col/fov are at depth 3). A ``seed_from_cache`` symlink of the whole
    ``.zarr`` is a single symlink that ``find -type d`` (no ``-L``, no trailing
    slash, as the production stage runs it) will NOT traverse -> it counts 0
    positions. ``seed_writable_store`` is the wrong tool too: it only shortcuts
    v3 ``c`` chunk dirs, so on a v0.4 store (``.zarray``/``.zgroup``, chunks as
    bare files) it would recurse into every chunk. This helper materializes just
    the real dir skeleton to the depth the stage scans, so the count is correct
    without copying any pixel data.

    ``depth`` counts directory levels below the store root; dirs at exactly
    ``depth`` are created, but their contents are not (the stage's bounded
    ``find`` never descends past it). Returns the destination store path.
    """
    src_root = cache / store_relpath
    if not src_root.exists():
        raise FileNotFoundError(f"cache missing store: {src_root}")
    dst_root = workdir / experiment / store_relpath
    dst_root.mkdir(parents=True, exist_ok=True)
    # BFS the source tree creating real dirs, stopping once we have created the
    # level-`depth` dirs (we never list a dir at level == depth, so the heavy
    # array/chunk level below a position is never touched).
    frontier: list[tuple[Path, int]] = [(src_root, 0)]
    while frontier:
        src_dir, lvl = frontier.pop()
        if lvl >= depth:
            continue
        for child in sorted(src_dir.iterdir()):
            if not child.is_dir():
                continue
            (dst_root / child.relative_to(src_root)).mkdir(
                parents=True, exist_ok=True
            )
            frontier.append((child, lvl + 1))
    return dst_root


def rebuild_upstream_in_process(
    workdir: Path,
    cache: Path,
    chain: list[tuple[Callable[..., Any], dict[str, Any]]],
    *,
    experiment: str = REFERENCE_EXPERIMENT,
) -> Path:
    """Run upstream stage callables in-process to produce a not-seedable input.

    Some stages need an upstream output that cannot simply be symlinked from the
    read-only `cache`: e.g. a store written in a NGFF version the stage-under-test
    will not read, or intermediate shards (virtual-staining pre-inference) the
    reference run did not retain. For those, run the producing stage(s) *in this
    process* so the input is materialized fresh in `workdir`'s OpsDataset layout
    under the current OPS env (already pointed at `workdir` by `real_data_workdir`).

    `chain` is an ordered list of ``(fn, kwargs)`` tuples; each `fn` is a pipeline
    stage entry point invoked as ``fn(**kwargs)``. Seed any inputs the chain reads
    via `seed_from_cache` BEFORE calling this. Returns `workdir` for chaining.

    Note: runs on the calling node (no SLURM), so reserve for light CPU rebuilds;
    GPU-heavy producers should be seeded from a completed reference run instead.
    """
    for fn, kwargs in chain:
        fn(**kwargs)
    return workdir
