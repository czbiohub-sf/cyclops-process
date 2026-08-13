#!/usr/bin/env python3
"""Single v3-convert wrapper for the fixed-cell pipeline (cell painting + 4i).

Both modalities extend the pheno v3 store with the fixed-cell channels warped in
via the registration affines. This is the ONE CLI for that step — it calls the
modality backend functions directly (no second CLI layer):

  - cp:  v3_fixed.submit_cell_paint_conversion_job
         (or submit_cp_seg_only_job for --mode cp_seg_only)
  - 4i:  v3_fixed._run_full_pipeline

Both modalities live in the one implementation module convert/v3_fixed.py
(the shared chunked-affine warp is convert/v3_warp.py). v3_fixed
keeps a staged main() because the 4i full-pipeline orchestrator drives it as a
subprocess; it shares the base convert_v3.py + convert_v3_slurm.py in processes/.

Usage:
    python -m cyclops_process.convert.v3_fixed_cli -e ops0174 --modality cp --parts 1 2
    python -m cyclops_process.convert.v3_fixed_cli -e ops0174 --modality cp --mode cp_seg_only --parts 1 2
    python -m cyclops_process.convert.v3_fixed_cli -e ops0144 --modality 4i
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace

sys.path.insert(0, os.getcwd())


def _run_cp(experiment: str, mode: str, units: list[int], wells, args) -> int:
    """Call the cell-paint backend submit function directly."""
    from cyclops_process.convert.v3_fixed import (
        submit_cell_paint_conversion_job,
        submit_cp_seg_only_job,
    )
    # Backend submit fns read a fixed, small set of args; build it explicitly.
    ns = Namespace(
        parts=units, wells=wells, force=args.force, dry_run=args.dry_run,
        no_wait=args.no_wait, quiet=args.quiet, skip_overlays=args.skip_overlays,
        dest_name=args.dest_name,
    )
    fn = submit_cp_seg_only_job if mode == "cp_seg_only" else submit_cell_paint_conversion_job
    result = fn(experiment, ns) or {}
    if result.get("dry_run"):
        return 0
    return 0 if result.get("success", False) else 1


def _run_4i(experiment: str, dry_run: bool, dest_name: str = None) -> int:
    """Call the 4i backend's full-pipeline orchestrator directly.

    _run_full_pipeline runs every stage (init+copy → transforms/pyramids ‖
    labels/seg-pyramids) and sys.exit()s with the combined return code.
    """
    from cyclops_process.convert.v3_fixed import _run_full_pipeline
    _run_full_pipeline(experiment, dry_run, dest_name)  # sys.exit()s internally
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert pheno v3 -> expanded v3 with fixed-cell channels (cp/4i)")
    p.add_argument("--experiment", "-e", required=True)
    p.add_argument("--modality", choices=["cp", "4i"], default="cp")
    p.add_argument("--mode", choices=["cell_paint", "cp_seg_only"], default="cell_paint",
                   help="[cp] cell_paint (default) or cp_seg_only (add CP nuclear seg to an existing v3 store)")
    p.add_argument("--units", "--parts", "--rounds", dest="units", nargs="+", type=int, default=None,
                   help="Units to include (default: modality default)")
    p.add_argument("--wells", nargs="+", type=int, default=None)
    p.add_argument("--force", "-f", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-wait", action="store_true")
    p.add_argument("--quiet", "-q", action="store_true")
    p.add_argument("--skip-overlays", action="store_true")
    p.add_argument("--dest-name", default=None,
                   help="[cp] write the output to this adjacent store name (e.g. phenotyping_v3_rerun.zarr) "
                        "instead of the canonical phenotyping_v3.zarr — leaves source + baseline untouched for A/B comparison")
    args = p.parse_args(argv)

    from cyclops_utils.data.filesystem import resolve_experiment_name
    from cyclops_process.fixed_cp_4i.configs.modality_config import get_modality

    experiment = resolve_experiment_name(args.experiment, autoselect=True)
    m = get_modality(args.modality)
    units = args.units or list(m.default_units)

    print(f"[convert_v3] modality={args.modality} experiment={experiment} units={units}")
    if args.modality == "cp":
        return _run_cp(experiment, args.mode, units, args.wells, args)
    return _run_4i(experiment, args.dry_run, args.dest_name)


if __name__ == "__main__":
    raise SystemExit(main())
