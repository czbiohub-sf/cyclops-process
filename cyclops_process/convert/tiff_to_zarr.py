"""OME-TIFF → OME-Zarr conversion (iohub TIFFConverter path).

The live-cell raw-convert step: converts OME-TIF acquisitions (iss / lc_5x /
lc_20x) to OME-Zarr, by experiment convention or explicit input/output dirs.
Used as the pipeline's ``convert_iss`` (and lc) step and by processes/assemble.py.
(The fixed-cell pipeline's raw writer is convert/raw_to_zarr.py instead.)
"""

from ops_utils.data.experiment import OpsDataset
from ops_utils.data.filesystem import ensure_output_path
import os
from tqdm import tqdm
from pathlib import Path
import natsort
from iohub.ngff import open_ome_zarr
from iohub.ngff import TransformationMeta
from iohub.convert import TIFFConverter
import glob
import json
import re
import shutil
import yaml
import numpy as np
from typing import List, Optional
from joblib import Parallel, delayed
from ops_utils.hpc.resource_manager import get_optimal_workers

from ops_utils.data.image_utils import augment_tile


from ops_utils.io.zarr_utils import ensure_position_array
from ops_utils.io.zarr_precreate import create_hcs_store_fast
from ops_utils.data.filesystem import (
    decide_overwrite_resume_skip,
    canonicalize_channel_name,
    build_channel_index_map,
    well_to_prefix,
    convert_position_to_hcs,
)




def convert(
    experiment: str = None,
    process: str = None,
    input_dir: str = None,
    output_dir: str = None,
    overwrite: bool | None = None,
) -> None:
    """
    Convert OME-TIF to Zarr using iohub functions
    - Can either run using the OPS convention by providing an experiment name,
    or can access directly by providing input and output directories

    Args:
        experiment (str):
            Experiment name i.e. ops{num}_{YYYMMDD}
        process (str):
            Options are 'iss', 'lc_5x', 'lc_20x'
        input_dir (str):
            Path to directory containing OME-TIF files
        output_dir (str):
            Path to directory where Zarr files will be saved
        overwrite (bool | None):
            Override overwrite behavior. If None (default), uses interactive/non-interactive detection.
            If True, force overwrite. If False, skip existing outputs.
    """

    if experiment is None:
        if input_dir is None or output_dir is None:
            raise ValueError(
                "Either experiment or input/output directories must be provided."
            )
        indv_rnd_dirs = [Path(input_dir)]
        output_dir = Path(output_dir)
        output_paths = [output_dir]

    else:
        dataset = OpsDataset(experiment)

        if process == "iss":
            experiment_dir = dataset.iss_tif_dir
            output_dir = dataset.store_paths["iss"]
            # Match any well-round dir ("A1_1", "B2_3", ...) across all rows; excludes "DAPI_round10".
            indv_rnd_dirs = [
                p
                for p in experiment_dir.iterdir()
                if p.is_dir() and re.match(r"^[A-Za-z]+\d+", p.name)
            ]
            output_paths = [
                output_dir.parent / f"{ird.name}.zarr" for ird in indv_rnd_dirs
            ]

            # Check for DAPI_round10 subfolder
            dapi_round10_dir = experiment_dir / "DAPI_round10"
            if dapi_round10_dir.exists() and dapi_round10_dir.is_dir():
                # Find well directories inside DAPI_round10 (any row)
                dapi_wells = [
                    p
                    for p in dapi_round10_dir.iterdir()
                    if p.is_dir() and re.match(r"^[A-Za-z]+\d+", p.name)
                ]
                if dapi_wells:
                    print(f"Found {len(dapi_wells)} wells in DAPI_round10 subfolder")
                    indv_rnd_dirs.extend(dapi_wells)
                    # Name pattern: DAPI_round10_A1_1.zarr
                    output_paths.extend([
                        output_dir.parent / f"DAPI_round10_{well.name}.zarr"
                        for well in dapi_wells
                    ])
                else:
                    print("WARNING: DAPI_round10 directory exists but contains no well directories")

        if process == "20x_beads":
            path = (
                dataset.lc_dragonfly_dir
                / f"{experiment.split('_')[0].upper()}_beads"
                / "1um_beads_1"
            )
            if not path.exists():
                path = glob.glob(str(path.parent / "*"))[0]
            indv_rnd_dirs = [path]
            output_paths = [dataset.store_paths["lc_20x_beads"]]

        if process == "lc_20x":
            raise NotImplementedError("5x conversion not implemented yet")

    # Determine overwrite behavior once for all outputs to avoid repeated prompts
    existing_outputs = [Path(p) for p in output_paths if Path(p).exists()]
    overwrite_all: Optional[bool] = overwrite  # Use explicit parameter if provided
    if existing_outputs and overwrite_all is None:
        # Check if running in interactive mode (has a TTY)
        import sys
        if sys.stdin.isatty():
            # Interactive mode - prompt user
            resp = (
                input(
                    f"{len(existing_outputs)} output path(s) already exist. Overwrite ALL? [y/N]: "
                )
                .strip()
                .lower()
            )
            overwrite_all = True if resp in ("y", "yes") else False
        else:
            # Non-interactive mode (Slurm job) - skip existing outputs by default
            print(f"{len(existing_outputs)} output path(s) already exist. Skipping existing outputs (non-interactive mode).")
            overwrite_all = False

    # Define conversion worker function for parallel processing
    def _convert_single_well(ird, op, overwrite_decision):
        """Convert a single well from TIFF to Zarr."""
        can_write = ensure_output_path(op, prompt_user=False, overwrite=overwrite_decision)
        if not can_write:
            return f"Skipped: {ird.name} (existing output retained)"

        converter = TIFFConverter(input_dir=ird, output_dir=op)
        converter()
        return f"Completed: {ird.name}"

    # Determine number of parallel workers
    # For I/O-bound TIFF conversion, use all available CPUs from Slurm allocation
    n_jobs = get_optimal_workers(
        use_gpu=False,
        model_ram_gb=0.5,  # Minimal overhead per worker
        data_ram_gb=2.0,   # Estimated memory per conversion task
        verbose=True,
    )
    n_jobs = max(1, min(n_jobs, len(indv_rnd_dirs)))  # Don't use more workers than tasks

    print(f"Converting {len(indv_rnd_dirs)} wells using {n_jobs} parallel workers...")

    # Run conversions in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_convert_single_well)(ird, op, overwrite_all)
        for ird, op in zip(indv_rnd_dirs, output_paths)
    )

    # Print results summary
    for result in results:
        print(result)

    if process == "20x_beads":
        # need to change channel names to the correct format
        # 0: GFP
        # 1: mCherry
        # 2: BF
        store = open_ome_zarr(output_paths[0], mode="r+")

        summary = store.zattrs["Summary"]
        old_names = store["0/0/0"].channel_names
        new_names = ["GFP", "mCherry", "BF"]
        for old_n, new_n in zip(old_names, new_names):
            store["0/0/0"].rename_channel(old_n, new_n)
        summary["ChNames"] = new_names
        store.zattrs["Summary"] = summary
        store.close()

    return
