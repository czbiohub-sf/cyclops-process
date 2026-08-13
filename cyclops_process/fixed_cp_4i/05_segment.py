#!/usr/bin/env python3
"""Segment nuclei for the unified fixed-cell pipeline (cell painting + 4i).

Step 05. Produces the 5x nuclear segmentation used as the REGISTRATION input:
segments the flatfield-corrected max-projection store for each unit and writes
``<unit_stem>_max_proj_flatfield_segmentation.zarr`` (the store step 06 reads).

Modality-aware via ``--modality {cp,4i}`` (default ``cp``); all naming/paths come
from ``modality_config`` so ``cp`` behaves exactly as the mature
``cell_painting/05_segment_cp.py`` and ``4i`` uses round stems + 2304px tiles.

Calls segment_and_stitch with process="cell_paint" (4x downsampling, nuclei
model, diameter 30). Requires the <stem>_stitch.yaml shifts produced by step 04
(stitch), so it runs after it.

Usage:
    python -m cyclops_process.fixed_cp_4i.05_segment segment --experiment ops0174 --parts 1 2 --part1-fliplr
    python -m cyclops_process.fixed_cp_4i.05_segment segment --modality 4i --experiment ops0144 --parts 1 2 3 4 5
    python -m cyclops_process.fixed_cp_4i.05_segment segment --input-store-path /path/to/part1_max_proj_flatfield.zarr --local --fliplr
    python -m cyclops_process.fixed_cp_4i.05_segment upscale --seg-store-path /path/to/segmentation.zarr
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path
from typing import Sequence

sys.path.insert(0, os.getcwd())

from cyclops_process.processes.segment import segment_and_stitch, upscale_nuclear_segmentations
from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.data.filesystem import resolve_experiment_name
from cyclops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
from cyclops_process.fixed_cp_4i.configs.modality_config import get_modality

# Cellpose segmentation runs on a local MultiGPUCluster, so each unit must run
# on a GPU node (submitted here, like the pheno segmentation step).
SLURM_PARAMS = {
    "gpus_per_node": 2,
    "slurm_constraint": "[h100|h200|6000_blackwell]",
    "timeout_min": 120,
    "mem": "150G",
    "cpus_per_task": 32,
    "slurm_partition": "gpu",
}


def _resolve_segment_paths(input_store: Path, config_path=None, output_store=None):
    """Derive the stitch-shifts config and segmentation output next to the input."""
    input_store = Path(input_store)
    if config_path is None:
        config_path = input_store.with_name(f"{input_store.stem}_stitch.yaml")
        if not config_path.exists():
            stem_base = input_store.stem.replace("_max_proj", "")
            config_path = input_store.with_name(f"{stem_base}_stitch.yaml")
    else:
        config_path = Path(config_path)
    if output_store is None:
        output_store = input_store.with_name(f"{input_store.stem}_segmentation.zarr")
    else:
        output_store = Path(output_store)
    return config_path, output_store


def _segment_store(
    input_store: Path,
    config_path=None,
    output_store=None,
    downsample: int = 4,
    diameter: int = 30,
    num_workers: int | None = None,
    debug_n_tiles: int | None = None,
    no_preprocess: bool = False,
    flipud: bool = False,
    fliplr: bool = False,
    rot90: int = 0,
    full_res_tile_size: int = 2048,
) -> str:
    """Segment + stitch nuclei for one store. Module-level so it's picklable for SLURM.

    flipud/fliplr/rot90 MUST match the orientation used in step 04 stitch — the
    segmentation reuses that step's shift YAML, so a mismatch misplaces tiles.
    """
    input_store = Path(input_store)
    config_path, output_store = _resolve_segment_paths(input_store, config_path, output_store)
    tile_size = (full_res_tile_size // downsample, full_res_tile_size // downsample)

    print(f"\n{'='*60}\nNuclei Segmentation and Stitching\n{'='*60}")
    print(f"Input store:  {input_store}")
    print(f"Config:       {config_path}")
    print(f"Output store: {output_store}")
    print(f"Orientation:  flipud={flipud}, fliplr={fliplr}, rot90={rot90} | downsample={downsample}x tile={tile_size} full_res={full_res_tile_size}\n")

    segment_and_stitch(
        experiment=None,  # Direct path mode
        process="cell_paint",  # 4x downsampling + shift scaling handled internally
        input_store_path=input_store,
        input_config_path=config_path,
        output_store_path=output_store,
        flipud=flipud,
        fliplr=fliplr,
        rot90=int(rot90) % 4,
        tile_size=tile_size,
        full_res_tile_size=full_res_tile_size,
        num_workers=num_workers,
        debug_n_tiles=debug_n_tiles,
        debug_output_suffix="_debug",
        use_preprocess=not no_preprocess,
        clahe_clip_limit=0.01,
        clahe_kernel_size=None,
    )
    print(f"\n✅ Segmentation complete! Output: {output_store}\n")
    return f"Done: {output_store}"


def run_segment_unit(
    experiment: str, modality: str, unit: int, downsample: int = 4, diameter: int = 30,
    num_workers: int | None = None, no_preprocess: bool = False,
    flipud: bool = False, fliplr: bool = False, rot90: int = 0,
) -> str:
    """Resolve a unit's flatfield max-proj store and segment it (SLURM payload)."""
    m = get_modality(modality)
    convert_dir = m.convert_dir(experiment)
    store = convert_dir / f"{m.unit_stem(unit)}_max_proj_flatfield.zarr"
    if not store.exists():
        store = convert_dir / f"{m.unit_stem(unit)}_max_proj.zarr"
    return _segment_store(
        store, downsample=downsample, diameter=diameter, num_workers=num_workers,
        no_preprocess=no_preprocess, flipud=flipud, fliplr=fliplr, rot90=rot90,
        full_res_tile_size=m.full_res_tile_size,
    )


def run_segment(args) -> None:
    """Manual single-store segmentation (direct --input-store-path)."""
    m = get_modality(args.modality)
    _segment_store(
        Path(args.input_store_path),
        config_path=args.config_path,
        output_store=args.output_store_path,
        downsample=args.downsample,
        diameter=args.diameter,
        num_workers=args.num_workers,
        debug_n_tiles=args.debug_n_tiles,
        no_preprocess=args.no_preprocess,
        flipud=args.flipud,
        fliplr=args.fliplr,
        rot90=args.rot90,
        full_res_tile_size=m.full_res_tile_size,
    )


def _submit_segment_experiment(args) -> None:
    """Submit one GPU SLURM segmentation job per unit for an experiment."""
    m = get_modality(args.modality)
    experiment = resolve_experiment_name(args.experiment, autoselect=True)
    dataset = OpsDataset(experiment)
    convert_dir = m.convert_dir(experiment)
    units = args.parts if args.parts is not None else m.default_units

    jobs = []
    for unit in units:
        store = convert_dir / f"{m.unit_stem(unit)}_max_proj_flatfield.zarr"
        if not store.exists():
            store = convert_dir / f"{m.unit_stem(unit)}_max_proj.zarr"
        if not store.exists():
            print(f"WARNING: no store found for unit {unit} in {convert_dir}, skipping")
            continue
        # Per-unit flips: global flags OR unit-specific flags (mirrors 04 stitch)
        flipud = args.flipud or (unit == 1 and args.part1_flipud) or (unit == 2 and args.part2_flipud)
        fliplr = args.fliplr or (unit == 1 and args.part1_fliplr) or (unit == 2 and args.part2_fliplr)
        jobs.append({
            "name": f"segment_{m.name}_{m.unit_stem(unit)}",
            "func": run_segment_unit,
            "kwargs": {
                "experiment": experiment, "modality": m.name, "unit": unit,
                "downsample": args.downsample, "diameter": args.diameter,
                "num_workers": args.num_workers, "no_preprocess": args.no_preprocess,
                "flipud": bool(flipud), "fliplr": bool(fliplr), "rot90": args.rot90,
            },
            "metadata": {"type": f"segment_{m.name}", "unit": unit},
        })

    if not jobs:
        print("No units to segment.")
        return

    if args.dry_run:
        for j in jobs:
            print(f"  {j['name']}: flipud={j['kwargs']['flipud']} fliplr={j['kwargs']['fliplr']}")
        print(f"Would submit {len(jobs)} GPU job(s).")
        return

    if args.local:
        for job in jobs:
            print(f"\n{'='*60}\nRunning {job['name']} locally...")
            job["func"](**job["kwargs"])
        return

    submit_parallel_jobs(
        jobs_to_submit=jobs,
        experiment=experiment,
        slurm_params=SLURM_PARAMS,
        log_dir=dataset.experiment_path / "slurm_logs" / f"segment_{m.name}",
        manifest_prefix=f"segment_{m.name}",
        dry_run=False,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Unified fixed-cell nuclei segmentation tools (cell painting + 4i)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Segment subcommand
    seg_parser = subparsers.add_parser(
        "segment",
        help="Segment nuclei from flatfield max projections",
    )
    seg_parser.add_argument(
        "--modality", choices=["cp", "4i"], default="cp",
        help="Imaging modality (default: cp)",
    )
    seg_parser.add_argument(
        "--experiment", "-e", default=None,
        help="Experiment name/shorthand; segments <unit_stem>_max_proj_flatfield.zarr on GPU SLURM",
    )
    seg_parser.add_argument(
        "--parts", nargs="+", type=int, default=None,
        help="Units (parts for cp, rounds for 4i); default: modality default_units",
    )
    seg_parser.add_argument(
        "--input-store-path",
        type=str,
        default=None,
        help="Input tiles store (OME-Zarr), max-projected (manual mode)",
    )
    seg_parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Shifts YAML (default: <input_stem>_stitch.yaml)",
    )
    seg_parser.add_argument(
        "--output-store-path",
        type=str,
        default=None,
        help="Output segmentation store (default: <input_stem>_segmentation.zarr)",
    )
    seg_parser.add_argument(
        "--downsample",
        type=int,
        default=4,
        help="Downsample factor for nuclei channel (default: 4, giving 512x512 tiles)",
    )
    seg_parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Dask workers (default: auto = num GPUs or 4)",
    )
    seg_parser.add_argument(
        "--debug-n-tiles",
        type=int,
        default=None,
        help="Debug: sample N tiles near center",
    )
    seg_parser.add_argument(
        "--diameter",
        type=int,
        default=30,
        help="Cellpose diameter parameter (default: 30)",
    )
    seg_parser.add_argument(
        "--no-preprocess", action="store_true", help="Disable CLAHE preprocessing"
    )
    seg_parser.add_argument(
        "--flipud", action="store_true", help="Apply vertical flip to all units"
    )
    seg_parser.add_argument(
        "--fliplr", action="store_true", help="Apply horizontal flip to all units"
    )
    seg_parser.add_argument("--part1-flipud", action="store_true", help="Flip unit 1 vertically only")
    seg_parser.add_argument("--part1-fliplr", action="store_true", help="Flip unit 1 horizontally only")
    seg_parser.add_argument("--part2-flipud", action="store_true", help="Flip unit 2 vertically only")
    seg_parser.add_argument("--part2-fliplr", action="store_true", help="Flip unit 2 horizontally only")
    seg_parser.add_argument("--rot90", type=int, default=0, help="Rotate tiles 90deg k times — MUST match step 04 stitch orientation (default: 0)")
    seg_parser.add_argument("--dry-run", action="store_true", help="Print SLURM plan without submitting")
    seg_parser.add_argument("--local", action="store_true", help="Run inline (needs a GPU node) instead of submitting to SLURM")

    # Upscale subcommand
    upscale_parser = subparsers.add_parser(
        "upscale",
        help="Upscale segmentations from 5x to 20x resolution",
    )
    upscale_parser.add_argument(
        "--seg-store-path",
        type=str,
        required=True,
        help="Path to segmentation zarr store",
    )
    upscale_parser.add_argument(
        "--dest-store-path",
        type=str,
        default=None,
        help="Path to destination store for symlinks (optional)",
    )
    upscale_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing upscaled segmentations",
    )

    args = parser.parse_args(argv)

    # Handle no command - show help
    if args.command is None:
        parser.print_help()
        return

    if args.command == "segment":
        if args.experiment:
            _submit_segment_experiment(args)
        elif args.input_store_path:
            run_segment(args)
        else:
            seg_parser.error("provide --experiment or --input-store-path")
    elif args.command == "upscale":
        upscale_nuclear_segmentations(
            seg_store_path=Path(args.seg_store_path),
            dest_store_path=Path(args.dest_store_path) if args.dest_store_path else None,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()


# Example commands (run AFTER step 04 stitch, which writes the shifts YAML):
# Segment cell painting (GPU SLURM, per-unit flips like ops0094):
#   python -m cyclops_process.fixed_cp_4i.05_segment segment -e ops0174 --parts 1 2 --part1-fliplr
# Segment 4i rounds:
#   python -m cyclops_process.fixed_cp_4i.05_segment segment --modality 4i -e ops0144 --parts 1 2 3 4 5
# Upscale seg to 20x:
#   python -m cyclops_process.fixed_cp_4i.05_segment upscale --seg-store-path <…>/part1_max_proj_flatfield_segmentation.zarr
