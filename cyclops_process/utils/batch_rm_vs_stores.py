#!/usr/bin/env python
"""
Batch delete virtual-staining stores (the virtual_staining/ subdirs) for track
and pheno in parallel. These are large and regenerable from the phase recon +
viscy inference.

Dry-run by default. Each process's VS stores are only deleted when its
downstream product exists:
  - pheno  -> phenotyping_v3.zarr (VS baked in as a channel; pheno_assembled_v3)
  - track  -> tracking_segmentation_stitched.zarr (lc_5x_segmentation)
Override the guard with --no-require-guard.

Deletion uses async_delete_path (instant rename + background rm), so it does
not block on slow per-file NFS unlinks.

python -m cyclops_process.utils.batch_rm_vs_stores                       # dry-run, all
python -m cyclops_process.utils.batch_rm_vs_stores --execute             # delete
python -m cyclops_process.utils.batch_rm_vs_stores --show-paths
python -m cyclops_process.utils.batch_rm_vs_stores --process pheno       # pheno only
python -m cyclops_process.utils.batch_rm_vs_stores --experiments ops0160_20260520
python -m cyclops_process.utils.batch_rm_vs_stores --execute --workers 8
"""
import sys
import os
import argparse
import time
from pathlib import Path

from joblib import Parallel, delayed
from tqdm import tqdm
from cyclops_process.paths import BASE_PATH

sys.path.insert(0, os.getcwd())

BASE_DIR = f"{BASE_PATH}"
VS_REL = "1-preprocess/live_imaging/virtual_staining"

# Per-process VS store keys (all live under virtual_staining/) and the
# downstream product whose existence makes them safe to delete.
PROCESS_TARGETS = {
    "track": {
        "keys": ["lc_5x_vs", "lc_5x_vs_intermediate", "lc_5x_vs_max_proj"],
        "guard": "lc_5x_segmentation",
    },
    "pheno": {
        "keys": ["lc_20x_vs", "lc_20x_vs_intermediate", "lc_20x_vs_max_proj"],
        "guard": "pheno_assembled_v3",
    },
}


def discover_experiments():
    """Experiments with a virtual_staining/ dir, via fixed-depth glob."""
    base = Path(BASE_DIR)
    return sorted({p.relative_to(base).parts[0] for p in base.glob(f"*/{VS_REL}")})


def _plan(experiment, processes, require_guard):
    """Resolve (present_targets, guarded_reason) per process for one experiment."""
    from cyclops_utils.data.experiment import OpsDataset

    ds = OpsDataset(experiment)
    plan = {}
    for proc in processes:
        cfg = PROCESS_TARGETS[proc]
        present = [ds.store_paths[k] for k in cfg["keys"] if ds.store_paths[k].exists()]
        guard = ds.store_paths[cfg["guard"]]
        reason = None
        if present and require_guard and not guard.exists():
            reason = f"no {cfg['guard']} (guarded)"
        plan[proc] = (present, guard.exists(), reason)
    return plan


def rm_experiment(experiment, processes, dry_run=True, require_guard=True):
    """Delete one experiment's VS stores. Returns (exp, status, msg, elapsed)."""
    from cyclops_utils.data.filesystem import async_delete_path

    start = time.time()
    plan = _plan(experiment, processes, require_guard)

    if not any(present for present, _, _ in plan.values()):
        return experiment, "skip", "no VS stores", time.time() - start

    to_delete, skips = [], []
    for proc, (present, _guard_ok, reason) in plan.items():
        if not present:
            continue
        if reason:
            skips.append(f"{proc}: {reason}")
        else:
            to_delete.extend(present)

    if not to_delete:
        return experiment, "skip", "; ".join(skips) or "guarded", time.time() - start

    names = ", ".join(p.name for p in to_delete)
    if dry_run:
        msg = f"would rm {names}" + (f" (kept {'; '.join(skips)})" if skips else "")
        return experiment, "dry-run", msg, time.time() - start

    deleted, errors = [], []
    for p in to_delete:
        try:
            async_delete_path(p)
            deleted.append(p.name)
        except PermissionError as e:
            errors.append(f"{p.name}: EPERM ({e.strerror})")
        except Exception as e:
            errors.append(f"{p.name}: {str(e)[:60]}")
    if errors:
        return experiment, "error", "; ".join(errors), time.time() - start
    return experiment, "ok", f"deleted {', '.join(deleted)}", time.time() - start


def show_paths(experiments, processes, require_guard):
    print(f"VS stores to delete ({len(experiments)} experiments):")
    for proc in processes:
        cfg = PROCESS_TARGETS[proc]
        print(f"  {proc}: {', '.join(cfg['keys'])}  guard={cfg['guard']}")
    print("-" * 60)
    for exp in experiments:
        try:
            plan = _plan(exp, processes, require_guard)
        except Exception as e:
            print(f"  {exp}: ERROR resolving ({str(e)[:50]})")
            continue
        parts = []
        for proc in processes:
            present, guard_ok, reason = plan[proc]
            g = "+" if guard_ok else "-"
            tag = " [GUARDED]" if reason else ""
            parts.append(f"{proc}:{len(present)} store(s) guard:{g}{tag}")
        print(f"  {exp}: " + " | ".join(parts))
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Batch delete track/pheno virtual-staining stores")
    parser.add_argument("--process", nargs="+", default=["track", "pheno"],
                        choices=["track", "pheno"], help="Which processes to clear")
    parser.add_argument("--experiments", nargs="+", help="Specific experiments (default: all with a VS dir)")
    parser.add_argument("--execute", action="store_true", help="Actually delete (default is dry-run)")
    parser.add_argument("--show-paths", action="store_true", help="Show targets and exit")
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers (default: auto)")
    parser.add_argument("--no-require-guard", dest="require_guard", action="store_false",
                        help="Delete even without a downstream product")
    parser.set_defaults(require_guard=True)
    args = parser.parse_args()

    if args.experiments:
        from cyclops_utils.data.filesystem import resolve_experiment_name
        experiments = [resolve_experiment_name(e, autoselect=True) for e in args.experiments]
    else:
        experiments = discover_experiments()

    if args.show_paths:
        show_paths(experiments, args.process, args.require_guard)
        return

    if args.workers is not None:
        workers = args.workers
    else:
        from cyclops_utils.hpc.resource_manager import get_optimal_workers
        workers = max(1, get_optimal_workers(use_gpu=False, verbose=False))

    dry_run = not args.execute
    mode_str = "(dry-run)" if dry_run else "(EXECUTE — deleting)"
    print(f"Deleting VS stores for {len(experiments)} experiments {mode_str}")
    print(f"Processes: {', '.join(args.process)}")
    print(f"Guard: {'require downstream product' if args.require_guard else 'NONE (--no-require-guard)'}")
    print(f"Workers: {workers}")
    print("-" * 60)

    total_start = time.time()
    results = Parallel(n_jobs=workers)(
        delayed(rm_experiment)(exp, args.process, dry_run, args.require_guard)
        for exp in tqdm(experiments, desc="Deleting", unit="exp")
    )
    total_elapsed = time.time() - total_start

    print("\n" + "-" * 60)
    print("Results:")
    print("-" * 60)
    stats = {}
    for exp, status, msg, elapsed in sorted(results):
        stats[status] = stats.get(status, 0) + 1
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins}m{secs:02d}s" if mins else f"{secs}s"
        symbol = {"ok": "+", "dry-run": ".", "skip": "-", "error": "!"}.get(status, "?")
        print(f"{symbol} {exp}: {status} ({msg}) [{time_str}]")

    total_mins, total_secs = divmod(int(total_elapsed), 60)
    print("-" * 60)
    print(f"Done: {stats.get('ok', 0)} deleted, {stats.get('dry-run', 0)} dry-run, "
          f"{stats.get('skip', 0)} skipped, {stats.get('error', 0)} errors")
    print(f"Total time: {total_mins}m{total_secs:02d}s" if total_mins else f"Total time: {total_secs}s")
    if stats.get("error"):
        print("\n(errors are likely EPERM — store owned by another user under a sticky parent; "
              "re-run as the owner)")
    if not dry_run:
        print("\n(deletions rename to sibling .trash_* and rm in background — space frees as those finish)")
    if dry_run:
        print("\n(dry-run — re-run with --execute to actually delete)")


if __name__ == "__main__":
    main()
