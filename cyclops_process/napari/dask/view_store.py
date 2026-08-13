#!/usr/bin/env python
"""Open a stitched OME-Zarr store directly in napari by path.

Unlike view_dask.py (which resolves stores via experiment + mode store-key
lookup), this points at any built .zarr store and opens it, so custom store
names (e.g. phenotyping_v3_4i_rerun.zarr) work without registration.

Example:
    python cyclops_process/napari/dask/view_store.py \
        /path/to/ops_data/ops0144_20260406/3-assembly/phenotyping_v3_4i_rerun.zarr
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())  # match view_dask.py so cyclops_process imports resolve

from cyclops_utils.data import filesystem as _fs
from cyclops_utils.data.filesystem import extract_ops_key
from cyclops_utils.io.zarr_utils import _iter_position_paths
from cyclops_process.napari.dask.view_dask import view_inplace_pyramid_in_napari


def main():
    parser = argparse.ArgumentParser(
        description="Open a stitched OME-Zarr store directly in napari by path."
    )
    parser.add_argument("store", type=str, help="Path to the .zarr store")
    parser.add_argument(
        "--wells",
        "-w",
        type=str,
        nargs="*",
        default=None,
        help="Optional well prefixes to view (e.g. A/1 A/2). Default: all.",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="pheno",
        help="Layer/broadcast mode (default: pheno).",
    )
    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        default=None,
        help="Experiment name for channel-marker coloring. Default: inferred from the path.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    _fs.VERBOSE = bool(args.verbose)

    store = Path(args.store)
    if not store.exists():
        raise SystemExit(f"Store not found: {store}")

    # Infer experiment from the path (for channel-marker labels) unless given.
    experiment = args.experiment or extract_ops_key(str(store))

    all_wells = _iter_position_paths(store)
    if args.wells:
        positions = [
            w for w in all_wells if any(str(w).startswith(str(s)) for s in args.wells)
        ]
    else:
        positions = all_wells

    view_inplace_pyramid_in_napari(
        source_store=store,
        positions=positions,
        name=store.stem,
        mode=args.mode,
        experiment=experiment,
        show_tracks=False,
    )


if __name__ == "__main__":
    main()
