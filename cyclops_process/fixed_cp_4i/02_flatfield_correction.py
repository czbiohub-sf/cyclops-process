#!/usr/bin/env python3
"""Flatfield correction for fixed-cell max-projected stores via SLURM.

Step 02 in the unified pipeline. Submits one SLURM job per unit that calls
flatfield_correction.py on the max-projected store before stitching.
Modality-aware via ``--modality {cp,4i}``; unit vocabulary and convert dir
come from the config.

Usage:
    python -m cyclops_process.fixed_cp_4i.02_flatfield_correction --experiment ops0094 --parts 1 2
    python -m cyclops_process.fixed_cp_4i.02_flatfield_correction --experiment ops0094 --modality 4i --parts 1 3 5
    python -m cyclops_process.fixed_cp_4i.02_flatfield_correction --experiment ops0094 --parts 1 --channels Hoechst TOMM20
    python -m cyclops_process.fixed_cp_4i.02_flatfield_correction --experiment ops0094 --parts 1 --dry-run
    python -m cyclops_process.fixed_cp_4i.02_flatfield_correction --experiment ops0094 --parts 1 --local
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.getcwd())

from ops_utils.data.experiment import OpsDataset
from ops_utils.data.filesystem import resolve_experiment_name
from cyclops_process.processes.flatfield_correction import correct_flatfield
from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
from cyclops_process.fixed_cp_4i.configs.modality_config import get_modality

SLURM_PARAMS = {
    "timeout_min": 120,
    "mem": "250GB",
    "cpus_per_task": 64,
    "slurm_partition": "cpu",
}


def _resolve_paths(dataset, unit, modality):
    """Resolve source and output paths for a fixed-cell unit."""
    m = get_modality(modality)
    stem = m.unit_stem(unit)
    convert_dir = m.convert_dir(dataset.experiment)
    source_path = convert_dir / f"{stem}_max_proj.zarr"
    if not source_path.exists():
        source_path = convert_dir / f"{stem}.zarr"

    if "_max_proj" in source_path.stem:
        corrected_path = source_path.with_name(
            source_path.stem.replace("_max_proj", "_max_proj_flatfield") + ".zarr"
        )
    else:
        corrected_path = source_path.with_name(f"{source_path.stem}_flatfield.zarr")

    output_dir = dataset.experiment_path / "3-assembly" / "illumination_correction" / f"{m.name}_{stem}"
    return source_path, corrected_path, output_dir


def run_flatfield_part(
    experiment: str,
    part: int,
    modality: str = "cp",
    num_samples: int = 500,
    num_workers: int = None,
    channels: list[str] = None,
    sigma: int = 75,
    camera_offset: float = 100.0,
    debug_n_positions: int = None,
):
    """Run flatfield correction for a single unit. Called by SLURM or locally."""
    dataset = OpsDataset(experiment)
    source_path, corrected_path, output_dir = _resolve_paths(dataset, part, modality)

    if not source_path.exists():
        print(f"ERROR: No store found for unit {part}: {source_path}")
        return

    print(f"Unit {part}: {source_path.name} -> {corrected_path.name}")

    correct_flatfield(
        experiment=experiment,
        num_samples=num_samples,
        num_workers=num_workers,
        fluor_channels=channels,
        sigma=sigma,
        camera_offset=camera_offset,
        debug_n_positions=debug_n_positions,
        source_path_override=source_path,
        corrected_path_override=corrected_path,
        output_dir_override=output_dir,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Flatfield correction for fixed-cell stores (pre-stitch, via SLURM)",
    )
    parser.add_argument("--experiment", required=True, help="Experiment name (e.g., ops0094)")
    parser.add_argument("--modality", choices=["cp", "4i"], default="cp", help="Imaging modality (default: cp)")
    parser.add_argument("--parts", nargs="+", type=int, default=None, help="Units to correct (default: modality default units)")
    parser.add_argument("--num-samples", type=int, default=500, help="FOVs to sample for estimation (default: 500)")
    parser.add_argument("--num-workers", type=int, default=None, help="Parallel workers (default: auto)")
    parser.add_argument("--channels", nargs="+", default=None, help="Channel names to correct (default: all)")
    parser.add_argument("--sigma", type=int, default=75, help="Gaussian sigma (default: 75)")
    parser.add_argument("--camera-offset", type=float, default=100.0, help="Camera offset in counts (default: 100)")
    parser.add_argument("--debug-n-positions", type=int, default=None, help="Limit to N positions for testing")
    parser.add_argument("--dry-run", action="store_true", help="Print SLURM plan without submitting")
    parser.add_argument("--local", action="store_true", help="Run locally instead of via SLURM")

    args = parser.parse_args()
    args.experiment = resolve_experiment_name(args.experiment, autoselect=True)

    m = get_modality(args.modality)
    units = args.parts if args.parts is not None else m.default_units

    dataset = OpsDataset(args.experiment)

    # Validate paths before submitting
    jobs = []
    for unit in units:
        source_path, corrected_path, output_dir = _resolve_paths(dataset, unit, args.modality)
        if not source_path.exists():
            print(f"WARNING: No store found for unit {unit}: {source_path}, skipping")
            continue

        jobs.append({
            "name": f"flatfield_{m.name}_{m.unit_stem(unit)}",
            "func": run_flatfield_part,
            "kwargs": {
                "experiment": args.experiment,
                "part": unit,
                "modality": args.modality,
                "num_samples": args.num_samples,
                "num_workers": args.num_workers,
                "channels": args.channels,
                "sigma": args.sigma,
                "camera_offset": args.camera_offset,
                "debug_n_positions": args.debug_n_positions,
            },
            "metadata": {"type": f"flatfield_{m.name}", "unit": unit},
        })

    if not jobs:
        print("No units to process.")
        return

    if args.local:
        for job in jobs:
            print(f"\n{'='*60}")
            job["func"](**job["kwargs"])
    else:
        submit_parallel_jobs(
            jobs_to_submit=jobs,
            experiment=args.experiment,
            slurm_params=SLURM_PARAMS,
            log_dir=dataset.experiment_path / "slurm_logs" / f"flatfield_{m.name}",
            manifest_prefix=f"flatfield_{m.name}",
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
