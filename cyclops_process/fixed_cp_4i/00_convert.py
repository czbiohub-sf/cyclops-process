#!/usr/bin/env python3
"""Convert raw fixed-cell TIFF/NDTiff acquisitions to OME-Zarr (cell painting + 4i).

Step 00 of the unified fixed-cell pipeline. Discovers the per-unit raw acquisition
dirs (CP "parts", 4i "rounds"), maps unit N -> <stem>{N}.zarr under
0-convert/<subdir>, and submits one resumable SLURM convert job per unit.

Discovery is the one genuinely modality-specific bit:
  - cp: real round leaf dirs under the dragonfly "…cellpainting…" root (round{N}_1),
        rejecting test/junk folders and picking the max-tiff leaf per round.
  - 4i: per-round instrument dirs from four_i_config (hardcoded per experiment).

Conversion uses convert_raw._convert_single(resume=True) — the 16-worker parallel
writer that can finish a timed-out convert on rerun (a fixed-cell round is
thousands of FOVs and NDTiff reads don't thread well).

Usage:
    python -m cyclops_process.fixed_cp_4i.00_convert -e ops0174 --modality cp
    python -m cyclops_process.fixed_cp_4i.00_convert -e ops0144 --modality 4i --dry-run
    python -m cyclops_process.fixed_cp_4i.00_convert -e ops0174 --modality cp --resume
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

from cyclops_process.fixed_cp_4i.configs.modality_config import get_modality

# CP raw layout: a "…cellpainting…" root holds one leaf tiff dir per round
# (round{N}_1, 7035 tifs), plus junk (round{N}_test_M, extra_random_stuff).
_CP_ROOT_RE = re.compile(r"cell\s*painting", re.IGNORECASE)
_ROUND_RE = re.compile(r"round[_-]?(\d+)", re.IGNORECASE)
_JUNK_RE = re.compile(r"test|extra_random_stuff|thumb|display_?settings", re.IGNORECASE)
_SKIP_DIR_RE = re.compile(r"^(phenotyping_|tracking_|\.|0-convert)", re.IGNORECASE)
_MIN_TIFFS_PER_ROUND = 10

SLURM_PARAMS = {
    "timeout_min": 720,  # NDTiff reads don't thread well (~5s/FOV); thousands of FOVs
    "mem": "200GB",
    "cpus_per_task": 64,
    "slurm_partition": "cpu,gpu",
}


def _move_to_trash(path: Path) -> None:
    """Rename to .trash_* (instant), delete in background (shared util)."""
    from ops_utils.data.filesystem import async_delete_path
    trash = async_delete_path(path)
    if trash is not None:
        print(f"    renamed to {trash.name} (deleting in background)")


def convert_unit(input_dir: str, output_dir: str) -> str:
    """Convert one raw unit dir to <stem>{N}.zarr (resumable). SLURM payload."""
    from cyclops_process.convert.raw_to_zarr import _convert_single
    return _convert_single(src_dir=input_dir, out_zarr=output_dir, resume=True)


def _count_tiffs(directory: Path) -> int:
    tiff_suffixes = {".tif", ".tiff"}
    n = 0
    with os.scandir(directory) as it:
        for entry in it:
            if entry.is_file() and (Path(entry.name).suffix.lower() in tiff_suffixes
                                    or entry.name.lower().endswith(".ome.tif")):
                n += 1
    return n


def _find_cp_root(dragonfly_dir: Path) -> Path:
    if not dragonfly_dir.exists():
        return dragonfly_dir
    for child in sorted(p for p in dragonfly_dir.iterdir() if p.is_dir()):
        if _CP_ROOT_RE.search(child.name):
            return child
    return dragonfly_dir


def _discover_cp_round_dirs(dragonfly_dir: Path, min_tiffs: int = _MIN_TIFFS_PER_ROUND) -> dict[int, Path]:
    """Map CP round number -> real leaf tiff dir (max-tiff leaf per round, junk dropped)."""
    cp_root = _find_cp_root(Path(dragonfly_dir))
    best: dict[int, tuple[int, Path]] = {}
    for cur_dir, subdirs, _files in os.walk(cp_root):
        cur = Path(cur_dir)
        rel = cur.relative_to(cp_root)
        rel_str = str(rel)
        subdirs[:] = [d for d in subdirs if not _JUNK_RE.search(d) and not _SKIP_DIR_RE.search(d)]
        if rel_str != "." and (_JUNK_RE.search(rel_str) or _SKIP_DIR_RE.search(rel.parts[0])):
            continue
        m = _ROUND_RE.search(rel_str)
        if not m:
            continue
        n_tiffs = _count_tiffs(cur)
        if n_tiffs < min_tiffs:
            continue
        rnd = int(m.group(1))
        if rnd not in best or n_tiffs > best[rnd][0]:
            best[rnd] = (n_tiffs, cur)
    return {rnd: leaf for rnd, (_cnt, leaf) in sorted(best.items())}


def _discover_4i_round_dirs(experiment: str) -> dict[int, Path]:
    """Map 4i round number -> instrument acquisition dir (from four_i_config)."""
    from cyclops_process.fixed_cp_4i.configs.four_i_config import get_all_round_input_dirs
    return dict(sorted(get_all_round_input_dirs().items()))


def discover_round_dirs(experiment: str, modality: str) -> dict[int, Path]:
    """Modality-aware raw unit discovery -> {unit_number: raw_dir}."""
    if modality == "4i":
        return _discover_4i_round_dirs(experiment)
    from ops_utils.data.experiment import OpsDataset
    return _discover_cp_round_dirs(Path(OpsDataset(experiment).lc_dragonfly_dir))


def run_experiment_mode(experiment: str, modality: str, units: list[int] | None = None,
                        force: bool = False, resume: bool = False,
                        dry_run: bool = False, local: bool = False) -> int:
    from ops_utils.data.experiment import OpsDataset
    from ops_utils.data.filesystem import resolve_experiment_name
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

    experiment = resolve_experiment_name(experiment, autoselect=True)
    m = get_modality(modality)
    dataset = OpsDataset(experiment)

    round_dirs = discover_round_dirs(experiment, modality)
    if not round_dirs:
        print(f"No {m.unit_word} directories discovered for {experiment} ({modality}).")
        return 4
    if units:
        round_dirs = {u: d for u, d in round_dirs.items() if u in set(units)}
        if not round_dirs:
            print(f"None of the requested {m.unit_word}s {sorted(set(units))} were discovered.")
            return 4

    out_root = m.convert_dir(experiment)
    print(f"Experiment:  {experiment}  (modality={modality})")
    print(f"Output root: {out_root}")
    print(f"Discovered {len(round_dirs)} {m.unit_word}(s):")
    for u, leaf in round_dirs.items():
        print(f"  {m.unit_word} {u} -> {m.unit_stem(u)}.zarr   [{leaf}]")

    out_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for u, leaf in round_dirs.items():
        dest = out_root / f"{m.unit_stem(u)}.zarr"
        if dest.exists():
            if force:
                if not dry_run:
                    print(f"FORCE: clearing existing {dest.name}")
                    _move_to_trash(dest)
            elif resume:
                print(f"RESUME: {dest.name} exists — will fill only missing positions")
            else:
                print(f"SKIP: {dest.name} exists (use --resume to finish it, --force to redo)")
                continue
        jobs.append({
            "name": f"convert_{modality}_{m.unit_stem(u)}",
            "func": convert_unit,
            "kwargs": {"input_dir": str(leaf), "output_dir": str(dest)},
            "metadata": {"type": "convert_fixed", "modality": modality, "unit": u},
        })

    if not jobs:
        print("\nNothing to convert (all units already exist).")
        return 0
    if dry_run:
        print(f"\n[dry-run] Would submit {len(jobs)} SLURM job(s): " + ", ".join(j["name"] for j in jobs))
        return 0
    if local:
        for job in jobs:
            print(f"\n{'='*60}\nRunning {job['name']} locally...")
            job["func"](**job["kwargs"])
        print(f"\nConversion complete. Output root: {out_root}")
        return 0

    result = submit_parallel_jobs(
        jobs_to_submit=jobs, experiment=experiment, slurm_params=SLURM_PARAMS,
        log_dir=dataset.experiment_path / "slurm_logs" / f"convert_{modality}",
        manifest_prefix=f"convert_{modality}", dry_run=False, step_name=f"convert_{modality}",
    )
    if isinstance(result, dict) and result.get("failed"):
        print(f"\n⚠️  {len(result['failed'])} convert job(s) failed")
        return 3
    print(f"\nConversion complete. Output root: {out_root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert raw fixed-cell acquisitions to OME-Zarr (cp/4i)")
    p.add_argument("--experiment", "-e", required=True, help="Experiment name/shorthand")
    p.add_argument("--modality", choices=["cp", "4i"], default="cp")
    p.add_argument("--parts", "--rounds", "--units", dest="units", nargs="+", type=int, default=None,
                   help="Restrict to these unit numbers (default: all discovered)")
    p.add_argument("--force", action="store_true", help="Reconvert even if <stem>{N}.zarr exists")
    p.add_argument("--resume", action="store_true", help="Reuse an existing store and fill only missing positions")
    p.add_argument("--dry-run", action="store_true", help="Print discovery plan without converting")
    p.add_argument("--local", action="store_true", help="Run inline instead of submitting to SLURM")
    args = p.parse_args(argv)
    return run_experiment_mode(args.experiment, args.modality, units=args.units,
                               force=args.force, resume=args.resume,
                               dry_run=args.dry_run, local=args.local)


if __name__ == "__main__":
    raise SystemExit(main())
