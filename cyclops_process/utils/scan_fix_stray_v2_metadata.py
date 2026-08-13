#!/usr/bin/env python3
"""Scan v3 zarr stores for stray zarr-v2 metadata files and (optionally) remove them.

A zarr-v3 store uses only ``zarr.json`` (chunks live under ``c/``). When a
v2-era reader/writer touches a v3 store it can drop v2 markers
(``.zgroup``, ``.zarray``, ``.zattrs``, ``.zmetadata``) beside the v3 nodes.
Those break strict readers: e.g. napari's built-in zarr reader sees a v2
``.zgroup``, treats the node as a v2 group, tries to open the v3 chunk dir
``c/`` as a member, and dies with::

    ValueError: Not a zarr dataset or group: .../A/1/0/2/c

This script walks every ``*_v3.zarr`` store in each experiment, finds those
stray v2 files, and removes them. Dry-run by default.

Safety:
  - Only a store whose ROOT has ``zarr.json`` (confirmed v3) is ever touched.
    A genuinely-v2 store (no root ``zarr.json``) is skipped, never modified.
  - Every removed file (path + original content) is recorded to a manifest
    JSON before deletion, so the removal is fully reversible.

Usage:
    # scan everything, dry-run (no deletions)
    python -m cyclops_process.utils.scan_fix_stray_v2_metadata
    # scan one experiment
    python -m cyclops_process.utils.scan_fix_stray_v2_metadata -e 180
    # actually delete the stray v2 files
    python -m cyclops_process.utils.scan_fix_stray_v2_metadata -e 180 --fix
    # fan out one SLURM job per experiment (mirrors audit_v3_all)
    python -m cyclops_process.utils.scan_fix_stray_v2_metadata --slurm --fix
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from cyclops_process.paths import BASE_PATH

# Experiment roots to scan. The first is the canonical OPS tree; the second
# holds the 11+20-round ISS re-runs (iss_ops2) under a nested base dir.
BASES = [
    f"{BASE_PATH}",
    f"{BASE_PATH}/iss_ops2",
]
V2_MARKERS = {".zgroup", ".zarray", ".zattrs", ".zmetadata"}
CHUNK_DIR = "c"           # v3 chunk subdir — prune when walking a store
RESULTS_DIR = Path(os.environ.get("OPS_STRAY_V2_RESULTS", "stray_v2_results"))
BACKUP_ROOT = Path(os.environ.get("OPS_STRAY_V2_BACKUPS", "stray_v2_backups"))


def _norm_prefix(spec: str) -> str:
    """Normalize an experiment spec to its ``ops0NNN`` prefix ('180'->'ops0180')."""
    s = str(spec)
    return s if s.startswith("ops") else f"ops{int(s):04d}"


def discover_experiments(specific=None) -> list[tuple[str, str]]:
    """Return ``(base, experiment_dir_name)`` pairs across all BASES.

    ``specific`` matches by ``ops0NNN`` prefix (or exact dir name) in any base.
    """
    all_pairs: list[tuple[str, str]] = []
    for base in BASES:
        if not Path(base).is_dir():
            continue
        for d in sorted(os.listdir(base)):
            if d.startswith("ops0") and (Path(base) / d).is_dir():
                all_pairs.append((base, d))
    if not specific:
        return all_pairs
    out: list[tuple[str, str]] = []
    for spec in specific:
        pref = _norm_prefix(spec)
        matches = [(b, e) for (b, e) in all_pairs if e == str(spec) or e.startswith(pref + "_") or e == pref]
        if not matches:
            print(f"  WARNING: no experiment matching {spec!r} in any base")
        out.extend(matches)
    return out


def find_v3_stores(base: str, experiment: str) -> list[Path]:
    """Deterministic v3 store roots from ``OpsDataset.store_paths`` (keys
    containing ``_v3``), built against ``base``. No filesystem walk — instant.
    Only existing stores are returned."""
    from cyclops_utils.data.experiment import OpsDataset
    # OpsDataset builds all paths off these env vars at construction time.
    os.environ["OPS_OUTPUT_BASE_DIR"] = base
    os.environ["OPS_FAST_OUTPUT_BASE_DIR"] = base
    ds = OpsDataset(experiment)
    stores = {Path(p) for k, p in ds.store_paths.items()
              if "_v3" in k and Path(p).exists()}
    return sorted(stores)


def scan_store(store: Path) -> dict:
    """Scan one store for stray v2 markers.

    Returns ``{store, is_v3, stray: [paths]}``. ``is_v3`` is False (and stray
    is empty) when the store root lacks ``zarr.json`` — such stores are left
    untouched.
    """
    if not (store / "zarr.json").exists():
        return {"store": str(store), "is_v3": False, "stray": []}
    stray: list[str] = []
    for dirpath, dirnames, filenames in os.walk(store):
        if CHUNK_DIR in dirnames:
            dirnames.remove(CHUNK_DIR)      # prune v3 chunk trees
        for f in filenames:
            if f in V2_MARKERS:
                stray.append(str(Path(dirpath) / f))
    return {"store": str(store), "is_v3": True, "stray": sorted(stray)}


def process_experiment(base: str, experiment: str, fix: bool = False) -> dict:
    """Scan (and optionally fix) all v3 stores in one experiment."""
    result = {"base": base, "experiment": experiment, "stores": [], "removed": [],
              "skipped_non_v3": [], "n_stray": 0, "n_removed": 0}

    backups: dict[str, str] = {}
    for store in find_v3_stores(base, experiment):
        info = scan_store(store)
        if not info["is_v3"]:
            result["skipped_non_v3"].append(info["store"])
            continue
        stray = info["stray"]
        result["stores"].append({"store": info["store"], "n_stray": len(stray)})
        result["n_stray"] += len(stray)
        for p in stray:
            pp = Path(p)
            try:
                backups[p] = pp.read_text()
            except Exception:
                backups[p] = None
            if fix:
                pp.unlink()
                result["removed"].append(p)
    result["n_removed"] = len(result["removed"])

    # Persist a restore manifest whenever we actually deleted something.
    if fix and result["removed"]:
        BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest = BACKUP_ROOT / f"{experiment}_{stamp}.json"
        manifest.write_text(json.dumps(
            {p: backups[p] for p in result["removed"]}, indent=2))
        result["manifest"] = str(manifest)
    return result


def _process_and_save(base: str, experiment: str, fix: bool) -> str:
    """SLURM worker: process one experiment, save result JSON, return name."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        res = process_experiment(base, experiment, fix=fix)
    except Exception as e:
        res = {"base": base, "experiment": experiment, "error": str(e)}
    (RESULTS_DIR / f"{experiment}.json").write_text(json.dumps(res, default=str))
    return experiment


def _print_summary(results: list[dict], fix: bool) -> None:
    total_stray = sum(r.get("n_stray", 0) for r in results)
    total_removed = sum(r.get("n_removed", 0) for r in results)
    dirty = [r for r in results if r.get("n_stray", 0)]
    errors = [r for r in results if r.get("error")]

    print(f"\n{'=' * 70}")
    print("  STRAY V2 METADATA SCAN" + ("  (--fix: DELETED)" if fix else "  (dry-run)"))
    print(f"{'=' * 70}")
    print(f"\n  Experiments scanned: {len(results)}")
    print(f"  With stray v2 files: {len(dirty)}")
    print(f"  Stray files found:   {total_stray}")
    print(f"  Files removed:       {total_removed}")
    print(f"  Errors:              {len(errors)}")

    if dirty:
        print(f"\n{'-' * 70}\n  EXPERIMENTS WITH STRAY V2 FILES\n{'-' * 70}")
        for r in sorted(dirty, key=lambda x: x["experiment"]):
            print(f"\n  {r['experiment']}: {r['n_stray']} stray"
                  + (f", {r['n_removed']} removed" if fix else ""))
            prefix = f"{r.get('base', '')}/{r['experiment']}/"
            for s in r.get("stores", []):
                if s["n_stray"]:
                    rel = s["store"].replace(prefix, "")
                    print(f"    [{s['n_stray']:3d}]  {rel}")
    if errors:
        print(f"\n{'-' * 70}\n  ERRORS\n{'-' * 70}")
        for r in errors:
            print(f"  {r['experiment']}: {r['error']}")
    if not fix and total_stray:
        print(f"\n  Re-run with --fix to delete these {total_stray} files.")
    print(f"\n{'=' * 70}")


def main():
    p = argparse.ArgumentParser(description="Scan/fix stray zarr-v2 metadata in v3 stores")
    p.add_argument("-e", "--experiments", nargs="*",
                   help="Specific experiments (default: all under BASE)")
    p.add_argument("--fix", action="store_true",
                   help="Delete stray v2 files (default: dry-run, report only)")
    p.add_argument("--slurm", action="store_true",
                   help="Fan out one SLURM job per experiment")
    args = p.parse_args()

    experiments = discover_experiments(args.experiments)
    mode = "SLURM" if args.slurm else "local"
    print(f"Scanning {len(experiments)} experiments ({mode}, "
          f"{'FIX' if args.fix else 'dry-run'})...\n")

    if not args.slurm:
        results = []
        for i, (base, exp) in enumerate(experiments):
            r = process_experiment(base, exp, fix=args.fix)
            print(f"  [{i+1}/{len(experiments)}] {exp}: "
                  f"{r.get('n_stray', 0)} stray"
                  + (f", {r.get('n_removed', 0)} removed" if args.fix else "")
                  + (f"  ERROR: {r['error']}" if r.get("error") else ""))
            results.append(r)
        _print_summary(results, args.fix)
        return

    from cyclops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    import shutil
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    jobs = [{"name": f"strayv2_{exp}", "func": _process_and_save,
             "kwargs": {"base": base, "experiment": exp, "fix": args.fix}}
            for base, exp in experiments]
    submit_parallel_jobs(
        jobs_to_submit=jobs,
        experiment="stray_v2_scan",
        slurm_params={"timeout_min": 10, "mem": "8GB",
                      "cpus_per_task": 2, "slurm_partition": "cpu,gpu"},
        log_dir="slurm_stray_v2_logs",
        manifest_prefix="strayv2",
        wait_for_completion=True,
        verbose=True,
        print_resource_summary=False,
    )
    results = []
    for base, exp in experiments:
        rp = RESULTS_DIR / f"{exp}.json"
        results.append(json.loads(rp.read_text()) if rp.exists()
                       else {"base": base, "experiment": exp, "error": "no result file"})
    _print_summary(results, args.fix)


if __name__ == "__main__":
    main()
