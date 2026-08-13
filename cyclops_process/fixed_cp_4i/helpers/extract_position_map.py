#!/usr/bin/env python3
"""Extract position label mapping from NDTiff datasets.

Reads the NDTiff index to get original position labels (e.g., "A1-Site_040052")
and saves a JSON mapping from zarr position names to original labels.
This is much faster than re-converting — only reads metadata, no image data.

Usage:
    # Extract maps for all rounds (submits SLURM jobs)
    python -m cyclops_process.fixed_cp_4i.helpers.extract_position_map

    # Extract for specific rounds
    python -m cyclops_process.fixed_cp_4i.helpers.extract_position_map --rounds 1 5

    # Run locally
    python -m cyclops_process.fixed_cp_4i.helpers.extract_position_map --local
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.getcwd())

from cyclops_process.fixed_cp_4i.configs.four_i_config import (
    ROUNDS,
    NUM_ROUNDS,
    get_round_input_dir,
    get_default_output_dir,
)


def extract_map_for_round(src_dir: str, output_json: str) -> str:
    """Read MicroManager summary from first TIFF to get position labels.

    Uses tifffile to read just the metadata header — no pixel data,
    no full index scan. Should complete in seconds, not hours.
    """
    import json
    import time
    from pathlib import Path
    import tifffile

    src = Path(src_dir)
    out_json = Path(output_json)
    name = src.parent.name

    print(f"[{name}] Reading position labels from TIFF metadata...", flush=True)
    start = time.perf_counter()

    # Find the first NDTiffStack TIFF file
    tiff_files = sorted(src.glob("*NDTiffStack*.tif"))
    if not tiff_files:
        tiff_files = sorted(src.glob("*.tif"))
    if not tiff_files:
        raise FileNotFoundError(f"No TIFF files found in {src}")

    print(f"[{name}] Reading metadata from {tiff_files[0].name}...", flush=True)

    with tifffile.TiffFile(tiff_files[0]) as tif:
        # MicroManager stores summary metadata with StagePositions
        mm_meta = tif.micromanager_metadata
        if mm_meta is None:
            raise ValueError(f"No MicroManager metadata found in {tiff_files[0]}")

        summary = mm_meta.get("Summary", {})
        if isinstance(summary, str):
            summary = json.loads(summary)

        stage_positions = summary.get("StagePositions", [])
        if not stage_positions:
            # Try alternative keys
            stage_positions = summary.get("InitialPositionList", [])

        if not stage_positions:
            print(f"[{name}] WARNING: No StagePositions found. Summary keys: {list(summary.keys())}", flush=True)
            # Dump summary for debugging
            debug_path = out_json.with_suffix(".debug.json")
            with open(debug_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"[{name}] Saved summary to {debug_path} for inspection", flush=True)
            return f"Failed: no StagePositions in {name}"

    elapsed = time.perf_counter() - start
    print(f"[{name}] Found {len(stage_positions)} positions in {elapsed:.1f}s", flush=True)

    # Build mapping: sequential index -> original label
    pos_map = {}
    for i, sp in enumerate(stage_positions):
        label = sp.get("Label", sp.get("label", str(i)))
        pos_map[str(i)] = label

    print(f"[{name}] First 5: {list(pos_map.values())[:5]}", flush=True)
    print(f"[{name}] Last 5: {list(pos_map.values())[-5:]}", flush=True)

    with open(out_json, "w") as f:
        json.dump(pos_map, f, indent=1)

    print(f"[{name}] Saved {len(pos_map)} position labels to {out_json}", flush=True)
    return f"Done: {out_json.name} ({len(pos_map)} positions, {elapsed:.1f}s)"


def main():
    parser = argparse.ArgumentParser(
        description="Extract position label mapping from NDTiff datasets",
    )
    parser.add_argument(
        "--rounds",
        nargs="+",
        type=int,
        default=list(range(1, NUM_ROUNDS + 1)),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to save JSON maps (default: <experiment>/0-convert/4i/)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local", action="store_true")

    args = parser.parse_args()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else get_default_output_dir()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    from cyclops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

    jobs = []
    for rnd in args.rounds:
        src = get_round_input_dir(rnd)
        out_json = output_dir / f"round{rnd}_position_map.json"

        if out_json.exists():
            print(f"SKIP: round {rnd} (exists: {out_json})")
            continue

        if not src.exists():
            print(f"WARNING: Input not found for round {rnd}: {src}")
            continue

        jobs.append({
            "name": f"posmap_4i_round{rnd}",
            "func": extract_map_for_round,
            "kwargs": {"src_dir": str(src), "output_json": str(out_json)},
            "metadata": {"type": "posmap_4i", "round": rnd},
        })

    if not jobs:
        print("No rounds to process.")
        return

    if args.local:
        for job in jobs:
            print(f"\n{'='*60}")
            job["func"](**job["kwargs"])
    else:
        submit_parallel_jobs(
            jobs_to_submit=jobs,
            experiment="four_i",
            slurm_params={
                "timeout_min": 15,
                "mem": "16GB",
                "cpus_per_task": 4,
                "slurm_partition": "cpu,gpu",
            },
            log_dir="four_i/posmap",
            manifest_prefix="posmap_4i",
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
