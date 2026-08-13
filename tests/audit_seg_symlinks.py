#!/usr/bin/env python3
"""Audit segmentation label groups (nuclear_seg / seg) across all experiments.

For every seg-bearing zarr store (iss / track / pheno, v2 + v3), samples a couple
of FOVs and checks each seg label's level-0 entry for:

  - STALE_PREFIX : symlink target points at a retired mount
  - BROKEN       : symlink does not resolve (dangling — covers stale + malformed targets,
                   e.g. a target one level too deep like .../A/1/0/0/0)
  - INVALID      : resolves but is not a valid zarr array (no .zarray / zarr.json) — a
                   "level issue" where the link points at the wrong depth
  - MISSING      : the seg group exists but has no level-0 entry

Only non-OK findings are printed. This is a fast, SAMPLED check (a few FOVs per store),
not an exhaustive walk.

Usage:
    python tests/audit_seg_symlinks.py
    python tests/audit_seg_symlinks.py --fovs 3
    python tests/audit_seg_symlinks.py --experiment 94
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "cyclops_utils" / "src"))
from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.data.filesystem import resolve_experiment_name
from cyclops_process.paths import BASE_PATH

BASE = Path(BASE_PATH)
STALE_PREFIXES = (f"{BASE_PATH}/",)

# (store_key, [seg label subpaths]). v3 stores keep labels under labels/.
SEG_STORES = [
    ("iss_stitch", ["nuclear_seg"]),
    ("iss_stitch_registered", ["nuclear_seg"]),
    ("lc_5x_phase_2d_stitched", ["nuclear_seg"]),
    ("lc_5x_phase_2d_stitched_v3", ["labels/nuclear_seg"]),
    ("pheno_assembled", ["nuclear_seg", "seg"]),
    ("pheno_assembled_v3", ["labels/nuclear_seg", "labels/seg"]),
]

# Base-data symlink stores (convert stage): A/well/fov per-FOV arrays whose chunks
# symlink to raw acquisition data. We sample FOVs and check the base array, not seg.
BASE_STORES = [
    ("lc_5x", "tracking_symlink"),
    ("iss", "bc_symlink"),
]


def _sample_fovs(store: Path, n: int) -> list[str]:
    """First n 'A/c/fov' positions containing a level-0 image array."""
    out = []
    for fov_dir in sorted(store.glob("A/*/*")):
        if fov_dir.is_dir() and (fov_dir / "0").exists():
            out.append(str(fov_dir.relative_to(store)))
            if len(out) >= n:
                break
    return out


def _first_symlink(root: Path) -> Path | None:
    """First symlink found anywhere under root (depth-first); None if none."""
    stack = [str(root)]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.is_symlink():
                        return Path(e.path)
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            continue
    return None


def _is_array(p: Path) -> bool:
    return (p / ".zarray").exists() or (p / "zarr.json").exists()


def _check_base_fov(fov: Path) -> tuple[str, str] | None:
    """Check a base-store FOV: level-0 array validity + a sample chunk symlink."""
    arr0 = fov / "0"
    if not arr0.exists():
        return ("MISSING", str(arr0))
    if not _is_array(arr0):
        return ("INVALID", str(arr0))
    link = _first_symlink(arr0)
    if link is not None:
        target = os.readlink(link)
        if any(p in target for p in STALE_PREFIXES):
            return ("STALE_PREFIX", target)
        if not link.exists():
            return ("BROKEN", f"{link.name} -> {target}")
    return None


def _check_level0(seg0: Path) -> tuple[str, str] | None:
    """Classify a seg label's level-0 entry. Returns (status, detail) or None if OK."""
    is_link = seg0.is_symlink()
    if is_link:
        target = os.readlink(seg0)
        if any(p in target for p in STALE_PREFIXES):
            return ("STALE_PREFIX", target)
        if not seg0.exists():  # dangling: stale OR malformed (wrong depth)
            return ("BROKEN", target)
    elif not seg0.exists():
        return ("MISSING", str(seg0))
    # Resolves (link or materialized dir) — confirm it is a real zarr array.
    if not ((seg0 / ".zarray").exists() or (seg0 / "zarr.json").exists()):
        return ("INVALID", os.readlink(seg0) if is_link else str(seg0))
    return None


def audit_experiment(experiment: str, n_fovs: int) -> list[dict]:
    try:
        ds = OpsDataset(experiment)
    except Exception as e:
        return [{"experiment": experiment, "store": "-", "status": "DATASET_ERROR", "detail": str(e)}]

    findings = []
    for store_key, seg_labels in SEG_STORES:
        store = Path(ds.store_paths.get(store_key, ""))
        if not store.exists():
            continue
        for pos in _sample_fovs(store, n_fovs):
            for seg in seg_labels:
                seg_group = store / pos / seg
                if not seg_group.exists():
                    continue  # this store/pos legitimately may not carry this label
                res = _check_level0(seg_group / "0")
                if res:
                    findings.append({
                        "experiment": experiment, "store": store_key,
                        "pos": pos, "seg": seg, "status": res[0], "detail": res[1],
                    })

    # Base-data symlink stores: sample a few FOVs, check the base array + chunk links.
    for store_key, label in BASE_STORES:
        store = Path(ds.store_paths.get(store_key, ""))
        if not store.exists():
            continue
        for pos in _sample_fovs(store, n_fovs):
            res = _check_base_fov(store / pos)
            if res:
                findings.append({
                    "experiment": experiment, "store": label,
                    "pos": pos, "seg": "base/0", "status": res[0], "detail": res[1],
                })
    return findings


def main():
    ap = argparse.ArgumentParser(description="Audit seg label symlinks/levels across experiments")
    ap.add_argument("--experiment", "-e", default=None, help="Single experiment (e.g. 94)")
    ap.add_argument("--fovs", type=int, default=2, help="FOVs to sample per store (default: 2)")
    ap.add_argument("--workers", type=int, default=16, help="Threads (default: 16)")
    args = ap.parse_args()

    if args.experiment:
        experiments = [resolve_experiment_name(args.experiment, autoselect=True)]
    else:
        experiments = sorted(d.name for d in BASE.iterdir() if d.is_dir() and d.name.startswith("ops"))

    print(f"Auditing {len(experiments)} experiment(s), {args.fovs} FOV(s)/store, "
          f"stores={[s for s, _ in SEG_STORES]}\n")

    all_findings = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for findings in ex.map(lambda e: audit_experiment(e, args.fovs), experiments):
            all_findings.extend(findings)

    if not all_findings:
        print("✓ No stale / broken / invalid seg label entries found.")
        return

    by_status: dict[str, int] = {}
    for f in sorted(all_findings, key=lambda x: (x["status"], x["experiment"])):
        by_status[f["status"]] = by_status.get(f["status"], 0) + 1
        print(f"  {f['status']:13s} {f['experiment']:24s} {f.get('store',''):28s} "
              f"{f.get('pos','')}/{f.get('seg','')}  -> {f['detail']}")
    print("\n--- Summary ---")
    for st, n in sorted(by_status.items()):
        print(f"  {st}: {n}")
    print(f"  total problem entries: {len(all_findings)}")


if __name__ == "__main__":
    main()
