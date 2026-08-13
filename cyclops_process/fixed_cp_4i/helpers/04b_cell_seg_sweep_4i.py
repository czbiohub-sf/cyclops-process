#!/usr/bin/env python3
"""Submit 4i cell segmentation channel sweep as a SLURM job.

Tests multiple (nuclei, membrane) channel combinations on a 2x2 tile grid to
find the best channels for cell segmentation. Writes each config's output to
a separate `4i_sweep_<name>` label group for visual comparison in napari.

Usage:
    python -m cyclops_process.fixed_cp_4i.helpers.04b_cell_seg_sweep_4i
    python -m cyclops_process.fixed_cp_4i.helpers.04b_cell_seg_sweep_4i --wells 1 --position A/1/0
    python -m cyclops_process.fixed_cp_4i.helpers.04b_cell_seg_sweep_4i --local
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.getcwd())

from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
from ops_utils.data.experiment import OpsDataset
from cyclops_process.processes.cell_seg.cell_segmentation import (
    run_cp_sweep_preview,
    FOUR_I_SWEEP_CONFIGS,
)
from cyclops_process.fixed_cp_4i.configs.four_i_config import EXPERIMENT

SLURM_PARAMS = {
    "timeout_min": 120,
    "mem": "250GB",
    "cpus_per_task": 32,
    "gpus_per_node": 1,
    "slurm_partition": "gpu",
    "slurm_constraint": "[h100|h200]",
}


def run_sweep_worker(experiment: str, position: str, store_override: str):
    """SLURM worker: run the 4i sweep on one position."""
    result = run_cp_sweep_preview(
        experiment=experiment,
        position=position,
        configs=FOUR_I_SWEEP_CONFIGS,
        label_prefix="4i_sweep",
        sweep_name="4i CHANNEL SWEEP",
        store_override=store_override,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Submit 4i cell seg channel sweep to SLURM")
    parser.add_argument("--experiment", "-e", default=EXPERIMENT)
    parser.add_argument("--position", "-p", default="A/1/0",
                        help="Position to run sweep on (default: A/1/0)")
    parser.add_argument("--store-name", default="phenotyping_v3_with_4i_unregistered.zarr",
                        help="v3 store filename under 3-assembly/")
    parser.add_argument("--local", action="store_true", help="Run locally instead of SLURM")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset = OpsDataset(args.experiment)
    store_path = dataset.experiment_path / "3-assembly" / args.store_name

    if not store_path.exists():
        print(f"ERROR: Store not found: {store_path}")
        sys.exit(1)

    print(f"Store: {store_path}")
    print(f"Position: {args.position}")
    print(f"Sweep configs: {len(FOUR_I_SWEEP_CONFIGS)}")

    job = {
        "name": f"4i_sweep_{args.position.replace('/', '_')}",
        "func": run_sweep_worker,
        "kwargs": {
            "experiment": args.experiment,
            "position": args.position,
            "store_override": str(store_path),
        },
        "metadata": {"position": args.position},
    }

    if args.local:
        run_sweep_worker(**job["kwargs"])
    else:
        submit_parallel_jobs(
            jobs_to_submit=[job],
            experiment=args.experiment,
            slurm_params=SLURM_PARAMS,
            log_dir="four_i/cell_seg_sweep",
            manifest_prefix="4i_sweep",
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
