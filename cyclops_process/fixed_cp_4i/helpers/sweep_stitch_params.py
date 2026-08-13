#!/usr/bin/env python3
"""Sweep stitch parameters (overlap / flip / rotation / channel) and rank by confidence.

Runs the fast estimate-only stitch (assemble.estimate_stitch) on a small adjacent
tile grid for every parameter combination, reads the per-well edge confidence each
one writes, and prints a ranked table so you can pick orientation/channel/overlap
before committing a full stitch. Empty wells (all-zero confidence, e.g. an unimaged
control well) are ignored when scoring.

Works for both fixed-cell modalities via --modality {cp,4i}; the modality config
resolves the per-unit convert store paths.

Combos run locally in parallel by default, or fan out one SLURM job each with
--slurm (via submit_parallel_jobs) for large grids.

Usage:
    # Find working orientation+channel for a cell painting experiment (parts 1 2):
    python -m cyclops_process.fixed_cp_4i.sweep_stitch_params -e ops0174 --parts 1 \
        --rotations 0 1 --flips none fliplr --channels 0 1 2 3 --overlaps 100

    # Sweep a 4i experiment's rounds:
    python -m cyclops_process.fixed_cp_4i.sweep_stitch_params --modality 4i -e ops0144 --parts 1 \
        --rotations 0 1 --flips none fliplr --channels 0 --overlaps 184

    # Explicit store(s), fan out on SLURM:
    python -m cyclops_process.fixed_cp_4i.sweep_stitch_params --stores /path/a.zarr --slurm
"""

from __future__ import annotations

import argparse
import itertools
import os
import statistics
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, os.getcwd())

import yaml

from cyclops_process.fixed_cp_4i.configs.modality_config import get_modality

# (flipud, fliplr) for each --flips token
FLIP_TOKENS = {
    "none": (False, False),
    "fliplr": (False, True),
    "flipud": (True, False),
    "both": (True, True),
}

# Estimate-only is CPU + light; a small grid is fast, so modest resources.
SLURM_PARAMS = {
    "timeout_min": 30,
    "mem": "64G",
    "cpus_per_task": 16,
    "slurm_partition": "cpu",
}


def estimate_confidence_combo(
    store_path: str,
    cfg_path: str,
    flipud: bool,
    fliplr: bool,
    rot90: int,
    overlap: int,
    channel: int,
    limit_positions: int,
    use_clahe: bool = True,
    clahe_clip_limit: float = 0.02,
    tile_size: tuple = (2048, 2048),
) -> str:
    """Estimate shifts for one combo on a small grid; writes confidence to cfg_path.

    Module-level so it is picklable for SLURM. Returns cfg_path (results are read
    from the written YAML afterward, uniformly for local and SLURM runs).
    """
    from stitch.stitch import assemble

    assemble.estimate_stitch(
        input_store_path=store_path,
        output_config_path=Path(cfg_path),
        flipud=flipud,
        fliplr=fliplr,
        rot90=rot90,
        tile_size=tuple(tile_size),
        overlap=overlap,
        limit_positions=limit_positions,
        channel=channel,
        use_clahe=use_clahe,
        clahe_clip_limit=clahe_clip_limit,
        verbose=False,
    )
    return str(cfg_path)


def _well_medians(cfg_path: Path) -> dict[str, float]:
    """Per-well median edge confidence from a stitch config; empty wells omitted."""
    try:
        conf = (yaml.safe_load(open(cfg_path)) or {}).get("confidence", {})
    except Exception:
        return {}
    out = {}
    for well, edges in conf.items():
        vals = [float(v[2]) for v in edges.values()]
        # Drop wells with no signal (all-zero confidence → unimaged/blank well).
        if vals and max(vals) > 0:
            out[well] = statistics.median(vals)
    return out


def _resolve_stores(args) -> list[tuple[str, Path]]:
    """Return (label, store_path) pairs from --experiment/--parts or --stores."""
    if args.stores:
        return [(Path(s).stem, Path(s)) for s in args.stores]
    from ops_utils.data.filesystem import resolve_experiment_name

    m = get_modality(args.modality)
    experiment = resolve_experiment_name(args.experiment, autoselect=True)
    convert_dir = m.convert_dir(experiment)
    units = args.parts if args.parts is not None else m.default_units
    stores = []
    for part in units:
        sp = convert_dir / f"{m.unit_stem(part)}_max_proj_flatfield.zarr"
        if not sp.exists():
            sp = convert_dir / f"{m.unit_stem(part)}_max_proj.zarr"
        if sp.exists():
            stores.append((m.unit_stem(part), sp))
        else:
            print(f"WARNING: no store for {m.unit_word} {part} under {convert_dir}")
    return stores


def sweep(args) -> list[dict]:
    """Run the sweep and return a list of result rows (one per combo)."""
    stores = _resolve_stores(args)
    if not stores:
        print("No stores to sweep.")
        return []

    limit_positions = int(args.grid) ** 2
    sweep_dir = Path(args.output_dir) if args.output_dir else (stores[0][1].parent / "stitch_sweep")
    sweep_dir.mkdir(parents=True, exist_ok=True)

    flip_pairs = [(tok, *FLIP_TOKENS[tok]) for tok in args.flips]
    combos = list(itertools.product(stores, args.overlaps, flip_pairs, args.rotations, args.channels))
    print(f"Sweeping {len(combos)} combo(s) across {len(stores)} store(s) "
          f"on a {args.grid}x{args.grid} grid ({limit_positions} tiles) -> {sweep_dir}")

    # Build a job per combo; each writes a uniquely-named config we read back.
    jobs, rows = [], []
    for (label, store), overlap, (ftok, fud, flr), rot, ch in combos:
        tag = f"{label}_ov{overlap}_{ftok}_r{rot}_c{ch}"
        cfg = sweep_dir / f"{tag}.yaml"
        row = {"store": label, "overlap": overlap, "flips": ftok, "rot90": rot,
               "channel": ch, "cfg": cfg, "tag": tag}
        rows.append(row)
        jobs.append({
            "name": f"sweep_{tag}",
            "func": estimate_confidence_combo,
            "kwargs": {
                "store_path": str(store), "cfg_path": str(cfg),
                "flipud": fud, "fliplr": flr, "rot90": rot, "overlap": overlap,
                "channel": ch, "limit_positions": limit_positions,
                "use_clahe": not args.no_clahe, "clahe_clip_limit": args.clahe_clip_limit,
            },
            "metadata": {"type": "stitch_sweep", "tag": tag},
        })

    if args.dry_run:
        for r in rows:
            print(f"  {r['tag']}")
        print(f"Would run {len(jobs)} combo(s).")
        return []

    if args.slurm:
        from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
        experiment = resolve_exp_label(args)
        submit_parallel_jobs(
            jobs_to_submit=jobs, experiment=experiment, slurm_params=SLURM_PARAMS,
            log_dir=sweep_dir / "logs", manifest_prefix="stitch_sweep", dry_run=False,
        )
    else:
        from joblib import Parallel, delayed
        Parallel(n_jobs=args.workers)(
            delayed(estimate_confidence_combo)(**j["kwargs"]) for j in jobs
        )

    # Aggregate: read each combo's confidence YAML.
    for r in rows:
        r["well_medians"] = _well_medians(r["cfg"])
        vals = list(r["well_medians"].values())
        r["min_well"] = min(vals) if vals else 0.0     # worst non-empty well
        r["mean_well"] = statistics.mean(vals) if vals else 0.0
    return rows


def resolve_exp_label(args) -> str:
    if args.experiment:
        from ops_utils.data.filesystem import resolve_experiment_name
        return resolve_experiment_name(args.experiment, autoselect=True)
    return "stitch_sweep"


def _print_ranked(rows: list[dict], min_confidence: float) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["min_well"], reverse=True)
    print(f"\n{'='*88}\nStitch parameter sweep — ranked by worst non-empty well (threshold {min_confidence})\n{'='*88}")
    print(f"{'rank':>4} {'store':>7} {'overlap':>7} {'flips':>7} {'rot':>3} {'chan':>4} "
          f"{'min_well':>9} {'mean_well':>9}  per-well")
    for i, r in enumerate(rows, 1):
        flag = "✓" if r["min_well"] >= min_confidence else " "
        per = ", ".join(f"{w}={m:.3f}" for w, m in sorted(r["well_medians"].items()))
        print(f"{i:>4} {r['store']:>7} {r['overlap']:>7} {r['flips']:>7} {r['rot90']:>3} "
              f"{r['channel']:>4} {r['min_well']:>9.3f} {r['mean_well']:>9.3f} {flag} {per}")
    best = rows[0]
    print(f"\nBest: {best['store']} overlap={best['overlap']} flips={best['flips']} "
          f"rot90={best['rot90']} channel={best['channel']} "
          f"(min well {best['min_well']:.3f})")


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Sweep stitch params (overlap/flip/rotation/channel) and rank by confidence")
    p.add_argument("--modality", choices=["cp", "4i"], default="cp", help="Imaging modality (default: cp)")
    p.add_argument("--experiment", "-e", default=None, help="Experiment name/shorthand (with --parts)")
    p.add_argument("--parts", nargs="+", type=int, default=None, help="Units to sweep (default: modality default parts/rounds)")
    p.add_argument("--stores", nargs="+", default=None, help="Explicit tiled store paths (instead of --experiment)")
    p.add_argument("--overlaps", nargs="+", type=int, default=[75, 100, 150], help="Overlap values in px")
    p.add_argument("--flips", nargs="+", default=["none"], choices=list(FLIP_TOKENS), help="Flip combos to try")
    p.add_argument("--rotations", nargs="+", type=int, default=[0], help="rot90 values (0-3)")
    p.add_argument("--channels", nargs="+", type=int, default=[0], help="Registration channel indices")
    p.add_argument("--grid", type=int, default=5, help="Adjacent NxN debug tile grid per well (default: 5)")
    p.add_argument("--no-clahe", action="store_true", help="Disable CLAHE preprocessing")
    p.add_argument("--clahe-clip-limit", type=float, default=0.02)
    p.add_argument("--min-confidence", type=float, default=0.8, help="Flag combos whose worst well is >= this")
    p.add_argument("--output-dir", default=None, help="Where to write sweep configs (default: <store>/../stitch_sweep)")
    p.add_argument("--workers", type=int, default=6, help="Local parallel workers (default: 6)")
    p.add_argument("--slurm", action="store_true", help="Fan out one SLURM job per combo")
    p.add_argument("--dry-run", action="store_true", help="List combos without running")
    args = p.parse_args(argv)

    if not args.experiment and not args.stores:
        p.error("provide --experiment (with --parts) or --stores")

    rows = sweep(args)
    _print_ranked(rows, args.min_confidence)


if __name__ == "__main__":
    main()
