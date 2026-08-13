# from iohub.ngff.convert import TIFFConverter
from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.profiling.decorators import versioned_function
from cyclops_utils.data.filesystem import ensure_output_path
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
from typing import List, Optional, Literal
from joblib import Parallel, delayed
from cyclops_utils.hpc.resource_manager import get_optimal_workers

from cyclops_utils.data.image_utils import augment_tile


from cyclops_utils.io.zarr_utils import ensure_position_array
from cyclops_utils.io.zarr_precreate import create_hcs_store_fast
from cyclops_utils.data.filesystem import (
    async_delete_path,
    canonicalize_channel_name,
    build_channel_index_map,
    well_to_prefix,
    convert_position_to_hcs,
)


def _assert_raw_convert_complete(zarr_paths: list, kind: str, n_sample: int = 24) -> None:
    """Fail fast if a raw_convert store the link step is about to read isn't fully
    written — i.e. the link step was fired before raw_to_zarr finished.

    raw_to_zarr pre-creates each store (all-zero) then fills positions in place,
    so a too-early read sees a store that exists but has unwritten (all-zero)
    positions. Every genuinely imaged FOV carries at least the camera offset, so
    an all-zero position reliably means "not yet converted".

    For speed (thousands of FOVs over NFS), this SAMPLES ``n_sample`` positions
    per store, evenly spaced (convert fills in parallel, so unwritten positions
    are scattered), and reads only a single 2D corner chunk of each. This is a
    probabilistic gate: a store missing only a handful of its positions may slip
    through, but the common "fired too early" case (store absent or largely
    unfilled) is caught reliably and near-instantly.
    """
    from concurrent.futures import ThreadPoolExecutor
    import os
    import zarr
    from cyclops_process.pipelinerunner.exceptions import PipelineHalted

    zarr_paths = [Path(p) for p in zarr_paths]
    missing = [p for p in zarr_paths if not p.exists()]
    if missing:
        names = ", ".join(p.name for p in missing)
        raise PipelineHalted(
            f"Raw conversion incomplete for {kind}: {len(missing)} source store(s) "
            f"missing ({names}). The convert step has not finished — wait for "
            f"raw_to_zarr to complete before running the link step."
        )

    def _subdirs(d):
        return sorted(e.name for e in os.scandir(d)
                      if e.is_dir() and not e.name.startswith("."))

    def _corner_max(rc):
        # Read a single 2D corner of the level-0 array directly via zarr — iohub's
        # per-position Position object is ~100x slower to construct over NFS.
        p, r, c = rc
        fovs = _subdirs(p / r / c)
        if not fovs:
            return f"{r}/{c}", 0
        pos_name = f"{r}/{c}/{fovs[0]}"
        arr = zarr.open(str(p / pos_name / "0"), mode="r")  # multiscale level 0
        idx = (0,) * (arr.ndim - 2)  # first frame across leading dims (T/C/Z)
        return pos_name, int(np.asarray(arr[idx][:32, :32]).max())

    for p in zarr_paths:
        # Enumerate position groups (row/col) via cheap filesystem scandir — iohub's
        # ds.positions() opens per-position metadata (~26s for thousands of FOVs).
        pos_cols = [(p, r, c) for r in _subdirs(p) for c in _subdirs(p / r)]
        if not pos_cols:
            raise PipelineHalted(
                f"Raw conversion incomplete for {kind}: {p.name} has no "
                f"positions. The convert step is still running or failed."
            )
        if len(pos_cols) > n_sample:
            step = (len(pos_cols) - 1) / (n_sample - 1)
            sample = [pos_cols[round(i * step)] for i in range(n_sample)]
        else:
            sample = pos_cols

        with ThreadPoolExecutor(max_workers=min(16, len(sample))) as ex:
            for pos_name, mx in ex.map(_corner_max, sample):
                if mx == 0:
                    raise PipelineHalted(
                        f"Raw conversion incomplete for {kind}: {p.name} has an "
                        f"unwritten position ({pos_name} is all-zero). The convert "
                        f"step is still running or was interrupted — wait for "
                        f"raw_to_zarr to complete before running the link step."
                    )


def _filter_positions_by_wells(
    dataset: OpsDataset, all_positions: list[str]
) -> list[str]:
    """Filter positions based on wells_to_process from experiment config.

    Args:
        dataset: OpsDataset instance for the experiment
        all_positions: List of all position names (e.g., ["A1-Site_0", "A2-Site_1", ...])

    Returns:
        Filtered list of positions matching wells_to_process, or all positions if not specified
    """
    with open(dataset.config_paths["exp_config"], "r") as f:
        exp_config = yaml.safe_load(f)
    wells_to_process = exp_config.get("wells_to_process", None)

    if wells_to_process:
        well_prefixes = [well_to_prefix(w) for w in wells_to_process]
        filtered_positions = [
            pos
            for pos in all_positions
            if any(pos.startswith(prefix) for prefix in well_prefixes)
        ]
        print(
            f"Filtering positions for wells {wells_to_process}: {len(filtered_positions)}/{len(all_positions)} positions"
        )
        return filtered_positions
    else:
        return all_positions


@versioned_function("v1.0")
def convert(
    experiment: str = None,
    process: str = None,
    input_dir: str = None,
    output_dir: str = None,
    overwrite: bool | None = None,
    ngff_version: Literal["0.4", "0.5"] = "0.5"
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
        ngff_version (Literal['0.4', 0.5]):
            Version of zarr to write in. 0.4 corresponds to zarr v2 and 0.5 corresponds to zarr v3
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
    def _convert_single_well(ird, op, overwrite_decision, ngff_version="0.5"):
        """Convert a single well from TIFF to Zarr."""
        can_write = ensure_output_path(op, prompt_user=False, overwrite=overwrite_decision)
        if not can_write:
            return f"Skipped: {ird.name} (existing output retained)"

        converter = TIFFConverter(input_dir=ird, output_dir=op, version=ngff_version)
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
        delayed(_convert_single_well)(ird, op, overwrite_all, ngff_version)
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


@versioned_function("v1.0")
def stack_symlinks(
    experiment: str,
    pre_nuclei_round: bool = False,
    skip_pre_dapi_round: bool = False,
    overwrite: bool | None = None,
    ngff_version: Literal["0.4", "0.5"] = "0.4",
    tile_size: tuple = (2048, 2048),
):
    """
    After conversion to OME-Zarr each round of ISS exists as an individual zarr store.
    This function assembles the zarr stores into a single zarr store with each round as time points

    Args:
        experiment (str):
            Experiment name i.e. ops{num}_{YYYMMDD}
        pre_nuclei_round (bool):
            If True, the first round contains only DAPI imaging. The DAPI from round 0
            will be copied into round 1's DAPI channel, and conversion starts from round 1
            as the new round 0, maintaining backward compatibility.
        skip_pre_dapi_round (bool):
            If True, the first round (pre-DAPI round) is simply dropped with no DAPI
            copying. Use this when the pre-DAPI round should be ignored entirely.
        overwrite (bool | None):
            Override overwrite behavior. If None (default), uses interactive/non-interactive detection.
            If True, force overwrite. If False, skip existing outputs.
    """

    dataset = OpsDataset(experiment)
    tile_size = tuple(tile_size)
    print(f"[stack_symlinks] Using tile_size={tile_size[0]}x{tile_size[1]} for {experiment}")

    # wells to process

    source_path = dataset.convert_in_situ
    # Discover wells dynamically based on available per-well zarrs, e.g., A1_*.zarr, B3_*.zarr
    all_zarrs = natsort.natsorted(glob.glob(str(source_path / "*.zarr")))
    well_token_to_zarrs = {}
    for zpath in all_zarrs:
        name = Path(zpath).name
        # Expect pattern like "A1_*.zarr" or "AA4_*.zarr"
        import re

        m = re.match(r"^([A-Za-z]+\d+)_.*\.zarr$", name)
        if not m:
            continue
        well_token = m.group(1)  # e.g., A1, B3, AA4
        well_token_to_zarrs.setdefault(well_token, []).append(zpath)

    # Sort wells and their round zarrs
    sorted_well_tokens = natsort.natsorted(well_token_to_zarrs.keys())
    zarr_lists = [natsort.natsorted(well_token_to_zarrs[w]) for w in sorted_well_tokens]
    # print(f"zarr_lists, found {len(zarr_lists)} wells: {zarr_lists}")

    if not zarr_lists:
        print(f"No well zarrs found in {source_path}; expected files like A1_*.zarr")
        return

    # Handle pre_nuclei_round: Check for DAPI_round10 or use round 0
    pre_nuclei_zarrs = None
    dapi_round10_zarrs = None
    use_pre_round_dapi = False
    use_dapi_round10 = False
    dapi_mode = None  # Will be 'dapi_round10', 'round0_to_round1', or 'skip_round0'

    if pre_nuclei_round:
        print("\n=== Pre-nuclei round detected ===")

        # Check for DAPI_round10 zarr files
        dapi_round10_pattern = str(source_path / "*DAPI_round10*.zarr")
        potential_dapi_round10 = glob.glob(dapi_round10_pattern)

        if potential_dapi_round10:
            # Found DAPI_round10 - organize by well
            print(f"Found DAPI_round10 zarr file(s): {len(potential_dapi_round10)} file(s)")
            dapi_round10_by_well = {}
            for dpath in potential_dapi_round10:
                name = Path(dpath).name
                import re
                # Match pattern: DAPI_round10_A1_1.zarr or A1_DAPI_round10.zarr
                m = re.match(r"^(?:DAPI_round10_)?([A-Za-z]+\d+)_.*\.zarr$", name)
                if m:
                    well_token = m.group(1)
                    dapi_round10_by_well[well_token] = dpath

            # Check if we have DAPI_round10 for all wells
            if all(w in dapi_round10_by_well for w in sorted_well_tokens):
                dapi_round10_zarrs = [dapi_round10_by_well[w] for w in sorted_well_tokens]

                # Automatically use DAPI_round10 - no other options needed
                dapi_mode = "dapi_round10"
                use_dapi_round10 = True
                print("\n✓ Using DAPI_round10 as DAPI source for all rounds.")
                print("  Round 0 spots will be preserved.")
            else:
                print("DAPI_round10 not found for all wells. Falling back to round 0/1 options.")
                dapi_round10_zarrs = None

        # If no DAPI_round10 or incomplete, automatically use option 1
        if dapi_round10_zarrs is None:
            print("\n" + "=" * 80)
            print("WARNING: Automatically using round 0 DAPI and replacing round 1 DAPI")
            print("         (discards round 0 spots)")
            print("=" * 80)
            print()

            dapi_mode = "round0_to_round1"
            use_pre_round_dapi = True
            print("Using round 0 DAPI for round 1. Round 0 spots will be discarded.")
            pre_nuclei_zarrs = [zarr_list[0] for zarr_list in zarr_lists]
            zarr_lists = [zarr_list[1:] for zarr_list in zarr_lists]

        # Validate we have enough rounds
        if not all(zarr_lists):
            raise ValueError(
                "pre_nuclei_round=True but not all wells have sufficient rounds"
            )

    if skip_pre_dapi_round:
        print("\n=== skip_pre_dapi_round=True: dropping round 0 entirely ===")
        zarr_lists = [zarr_list[1:] for zarr_list in zarr_lists]
        if not all(zarr_lists):
            raise ValueError("skip_pre_dapi_round=True but not all wells have rounds remaining after skipping round 0")

    dest_store_path = dataset.store_paths["iss"]

    # Confirm overwrite of assembled ISS store if it already exists
    # Handle non-interactive mode (SLURM jobs)
    if overwrite is None and dest_store_path.exists():
        import sys
        if not sys.stdin.isatty():
            # Non-interactive mode - default to overwrite=True for automation
            print(f"Output exists at {dest_store_path}. Overwriting (non-interactive mode).")
            overwrite = True

    can_write = ensure_output_path(dest_store_path, prompt_user=False, overwrite=overwrite)
    if not can_write:
        print(
            f"Skipping stacking symlinks (existing output retained at {dest_store_path})."
        )
        return

    # get high level info that is common to all rounds
    with open_ome_zarr(zarr_lists[0][0]) as ds:
        source_channel_names = ds.channel_names

    # For backward compatibility: always create 5-channel output
    # If source has 4 channels (no DAPI), add DAPI as channel 0
    if len(source_channel_names) == 4:
        channel_names = ["DAPI"] + list(source_channel_names)
        print(
            f"WARNING: Detected 4-channel source (no DAPI). Creating 5-channel output with DAPI channel 0 empty (all zeros)."
        )
    else:
        channel_names = source_channel_names

    # First pass: collect all positions and metadata
    all_positions = []
    well_info = []  # Store (path_list, position_list, num_channels_list) for each well

    for path_list in zarr_lists:
        num_channels_list = []
        for path in path_list:
            num_channels_list.append(len(open_ome_zarr(path).channel_names))

        with open_ome_zarr(path_list[0]) as ds:
            position_list = [x for x in ds.positions()]
            all_positions.extend([pos for pos, _ in position_list])

            if not well_info:  # Get metadata from first well
                output_chunk_size = ds[position_list[0][0]].data.chunks
                output_dtype = ds[position_list[0][0]].data.dtype
                output_scale = ds[position_list[0][0]].scale
                max_shape = tuple(tile_size)
                output_shape = (len(path_list),) + (5, 1) + tuple(max_shape)

        well_info.append((path_list, position_list, num_channels_list))

    # Use fast precreation for all positions at once
    create_hcs_store_fast(
        store_path=dest_store_path,
        positions=all_positions,
        shape=output_shape,
        chunks=output_chunk_size,
        dtype=output_dtype,
        scale=output_scale,
        channel_names=channel_names,
        version=ngff_version,
    )

    # Open the store for symlinking
    output_store = open_ome_zarr(dest_store_path, mode="a")

    # Second pass: create symlinks
    chunk_t0 = "0/c/0" if ngff_version == "0.5" else "0/0"
    for well_idx, (path_list, position_list, num_channels_list) in enumerate(well_info):
        for pos, _ in tqdm(position_list, desc=f"Linking positions for well"):
            for num, source_zarr_path in enumerate(path_list):
                # after a few rounds we stop collecting DAPI
                num_chan = num_channels_list[num]

                # For zarr v3 chunks live at array/c/T/C/...; for v2 at array/T/C/...
                if ngff_version == "0.5":
                    round_dir = Path(dest_store_path, pos, "0", "c", str(num))
                else:
                    round_dir = Path(dest_store_path, pos, "0", str(num))
                round_dir.mkdir(parents=True, exist_ok=True)

                if num_chan == 5:
                    itterator = range(5)

                    # Option 1: Use DAPI_round10 for round 0 only
                    if use_dapi_round10 and dapi_round10_zarrs and num == 0:
                        dapi_round10_zarr_path = dapi_round10_zarrs[well_idx]
                        dapi_src = Path(dapi_round10_zarr_path, pos, chunk_t0, "0")
                        dapi_dst = round_dir / "0"

                        # Use symlink for DAPI from DAPI_round10 (replaces source channel 0)
                        if not dapi_src.exists():
                            raise FileNotFoundError(
                                f"DAPI channel not found in DAPI_round10 at {dapi_src}. "
                                "use_dapi_round10=True requires DAPI in DAPI_round10."
                            )
                        os.symlink(dapi_src, dapi_dst, target_is_directory=True)

                        # Link spot channels (1-4) from source, skipping source channel 0 (old DAPI)
                        for chan in range(1, 5):
                            src = Path(source_zarr_path, pos, chunk_t0, str(chan))
                            dst = round_dir / str(chan)
                            os.symlink(src, dst, target_is_directory=True)

                    # Option 2: Use round 0 DAPI for round 1 (originally round 1 becomes round 0)
                    elif use_pre_round_dapi and num == 0 and pre_nuclei_zarrs:
                        # Copy DAPI from pre-nuclei round 0 into channel 0 of this round
                        pre_nuclei_zarr_path = pre_nuclei_zarrs[well_idx]
                        dapi_src = Path(pre_nuclei_zarr_path, pos, chunk_t0, "0")
                        dapi_dst = round_dir / "0"
                        print("dapidapidapi", pre_nuclei_zarr_path, pre_nuclei_zarrs, well_idx, dapi_src)
                        # Use symlink for DAPI from pre-nuclei round
                        if not dapi_src.exists():
                            raise FileNotFoundError(
                                f"DAPI channel not found in pre-nuclei round at {dapi_src}. "
                                "use_pre_round_dapi=True requires DAPI in round 0."
                            )
                        os.symlink(dapi_src, dapi_dst, target_is_directory=True)

                        # Link remaining channels (1-4) from the actual round 1 data
                        for chan in range(1, 5):
                            src = Path(source_zarr_path, pos, chunk_t0, str(chan))
                            dst = round_dir / str(chan)
                            os.symlink(src, dst, target_is_directory=True)
                    else:
                        # Normal processing for all other rounds (including option 3: skip round 0)
                        for chan in itterator:
                            src = Path(source_zarr_path, pos, chunk_t0, str(chan))
                            dst = round_dir / str(chan)
                            os.symlink(src, dst, target_is_directory=True)

                if num_chan == 4:
                    itterator = range(4)

                    # Option 1: Use DAPI_round10 for round 0 only (4-channel source)
                    if use_dapi_round10 and dapi_round10_zarrs and num == 0:
                        dapi_round10_zarr_path = dapi_round10_zarrs[well_idx]
                        dapi_src = Path(dapi_round10_zarr_path, pos, chunk_t0, "0")
                        dapi_dst = round_dir / "0"

                        # Use symlink for DAPI from DAPI_round10 (adds to channel 0)
                        if not dapi_src.exists():
                            raise FileNotFoundError(
                                f"DAPI channel not found in DAPI_round10 at {dapi_src}. "
                                "use_dapi_round10=True requires DAPI in DAPI_round10."
                            )
                        os.symlink(dapi_src, dapi_dst, target_is_directory=True)

                        # Shift 4-channel source (0-3) to dest channels (1-4) to make room for DAPI
                        for chan in itterator:
                            src = Path(source_zarr_path, pos, chunk_t0, str(chan))
                            dst = round_dir / str(chan + 1)
                            os.symlink(src, dst, target_is_directory=True)

                    # Option 2: Use round 0 DAPI for round 1 (4-channel source)
                    elif use_pre_round_dapi and num == 0 and pre_nuclei_zarrs:
                        # Copy DAPI from pre-nuclei round 0 into channel 0 of this round
                        pre_nuclei_zarr_path = pre_nuclei_zarrs[well_idx]
                        dapi_src = Path(pre_nuclei_zarr_path, pos, chunk_t0, "0")
                        dapi_dst = round_dir / "0"

                        # Use symlink for DAPI from pre-nuclei round
                        if not dapi_src.exists():
                            raise FileNotFoundError(
                                f"DAPI channel not found in pre-nuclei round at {dapi_src}. "
                                "use_pre_round_dapi=True requires DAPI in round 0."
                            )
                        os.symlink(dapi_src, dapi_dst, target_is_directory=True)

                        # Link remaining channels (1-4) from the actual round 1 data (4-channel source)
                        for chan in itterator:
                            src = Path(source_zarr_path, pos, chunk_t0, str(chan))
                            dst = round_dir / str(chan + 1)
                            os.symlink(src, dst, target_is_directory=True)
                    else:
                        # Normal processing for all other rounds (including option 3: skip round 0)
                        for chan in itterator:
                            src = Path(source_zarr_path, pos, chunk_t0, str(chan))
                            dst = round_dir / str(chan + 1)
                            os.symlink(src, dst, target_is_directory=True)

    return


@versioned_function("v1.1")
def link_phenotyping(
    experiment: str,
    phase_flipud: bool = False,
    phase_fliplr: bool = False,
    phase_rot90: int = 0,
    gfp_flipud: bool = False,
    gfp_fliplr: bool = False,
    gfp_rot90: int = 0,
    mCherry_flipud: bool = False,
    mCherry_fliplr: bool = False,
    mCherry_rot90: int = 0,
    cy5_flipud: bool = False,
    cy5_fliplr: bool = True,
    cy5_rot90: int = 0,
) -> None:
    """
    Clean up and re-organize the 20x phenotyping data, importantly remanes the FOV positions,
    which are needed for stitching
    Args:
        experiment (str):
            Experiment name i.e. ops{num}_{YYYMMDD}
        channel_list (list):
            List of channels to be used in the 20x data
    """

    print(f"linking phenotyping for {experiment}")
    dataset = OpsDataset(experiment)

    # Prefer fast partition raw_convert/ if it exists, fallback to dragonfly
    fast_convert = dataset.experiment_path_fast / "0-convert" / "live_imaging" / "raw_convert"
    dragonfly_convert = dataset.lc_dragonfly_dir / "0-convert"
    if fast_convert.exists() and any(fast_convert.glob("phenotyping_well_*.zarr")):
        source_path = fast_convert
        print(f"  Using fast partition: {source_path}")
    else:
        source_path = dragonfly_convert
        print(f"  Using dragonfly: {source_path}")
    dest_path = dataset.store_paths["lc_20x"]

    # Refuse to link if the raw_convert stores aren't fully written yet (link step
    # fired before raw_to_zarr finished). Only the fast raw_convert path is
    # pre-created-then-filled in place; the dragonfly fallback is pre-existing data.
    # Chunks A1..A3 mirror start_chunk=1/end_chunk=3 used below.
    if source_path == fast_convert:
        _assert_raw_convert_complete(
            [source_path / f"phenotyping_well_A{c}_1.zarr" for c in range(1, 4)],
            kind="phenotyping",
        )

    # Build expected position list from config (check ops_convert/ first)
    pos_list_path = dataset.config_paths["lc_20x_position_list"]
    fast_pos_list = source_path / "pheno_position_list.json"
    if fast_pos_list.exists():
        pos_list_path = fast_pos_list
    with open(pos_list_path, "r") as f:
        correct_pheno_pos_list = json.load(f)
        all_positions = list(correct_pheno_pos_list.keys())

    # Filter positions based on wells_to_process from experiment config
    expected_positions = _filter_positions_by_wells(dataset, all_positions)

    # derive channel list from dataset config (NO fallback - config must be correct)
    ch_to_org = dataset.channel_map_data or {}
    print(f"channel map: {ch_to_org}")

    if not ch_to_org:
        raise ValueError(
            f"No channel_map found in experiment config for {experiment}. "
            "The exp_config.yaml must define a channel_map to ensure correct "
            "channel-to-flip parameter mapping."
        )

    # Drop fixed-cell panel channels (CP/4i) flagged in the config — they stay in
    # channel_map for labeling but aren't part of the live-cell phenotyping store.
    fixed = set(getattr(dataset, "fixed_channels", []) or [])
    live_channels = [ch for ch in ch_to_org.keys() if ch not in fixed]
    if fixed:
        print(f"channel map (excluding {len(fixed)} fixed channel(s)): {live_channels}")

    # Canonicalize common channel names
    dataset_channels = [canonicalize_channel_name(ch) for ch in live_channels]
    # Prefer a common imaging order
    preferred_order = ["GFP", "mCherry", "Cy5", "BF"]
    ordered = [ch for ch in preferred_order if ch in dataset_channels]
    remaining = [ch for ch in dataset_channels if ch not in ordered]
    channel_list = ordered + remaining
    print(f"channel list: {channel_list}")

    # Convert expected positions to HCS paths for validation
    expected_hcs_positions = [convert_position_to_hcs(p) for p in expected_positions]

    # Rebuild from scratch: every position is relinked below.
    async_delete_path(dest_path)
    print(f"Creating data at {dest_path}.")

    start_chunk = 1
    end_chunk = 3

    # Validate source data exists
    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")
    print(
        f"Source path exists with {len(list(source_path.glob('*.zarr')))} zarr stores"
    )

    with open_ome_zarr(
        source_path / f"phenotyping_well_A{start_chunk}_1.zarr", mode="r"
    ) as ds_chunks:
        num_positions = len(ds_chunks["0"])
        _, pos = next(ds_chunks.positions())
        shape = pos.data.shape
        chunk_size = pos.data.chunks
        _scale = pos.scale
        dtype = pos.data.dtype

    # Use the filtered expected_positions instead of all positions
    correct_pheno_pos_keys = expected_positions

    # Build a mapping from position name to its index in the original list
    # This is needed to find the correct source position in the zarr chunks
    all_pos_in_config = list(correct_pheno_pos_list.keys())
    pos_to_original_idx = {pos: idx for idx, pos in enumerate(all_pos_in_config)}

    # HARDCODE scale and channel names
    scale = _scale[:-2] + [
        0.65,
        0.65,
    ]  # TODO: make this dynamic based on the scale of the 20x data

    # Pre-create all positions.
    # Convert position names to HCS format (e.g., "A1-Site_0" -> "A/1/0")
    hcs_positions = []
    for position in correct_pheno_pos_keys:
        hsc_correct_name = convert_position_to_hcs(position)
        hcs_positions.append(hsc_correct_name)

    # Use fast precreation for all positions at once
    # Use channel_list length for channel dimension to match the mapping logic
    create_hcs_store_fast(
        store_path=dest_path,
        positions=hcs_positions,
        shape=(shape[0], len(channel_list), *shape[2:]),
        chunks=chunk_size,
        dtype=dtype,
        scale=scale,
        channel_names=channel_list,
        version="0.5",
    )

    # Open the store for subsequent operations
    ds = open_ome_zarr(
        dest_path,
        mode="a",  # Always append mode since store is created above or already exists
    )

    any_augmentation = any(
        [
            phase_flipud,
            phase_fliplr,
            phase_rot90 != 0,
            gfp_flipud,
            gfp_fliplr,
            gfp_rot90 != 0,
            mCherry_flipud,
            mCherry_fliplr,
            mCherry_rot90 != 0,
            cy5_flipud,
            cy5_fliplr,
            cy5_rot90 != 0,
        ]
    )

    total_positions = len(correct_pheno_pos_keys)

    if not any_augmentation:
        print("No orientation changes specified, creating symlinks.")
        for i in tqdm(range(total_positions), desc="Linking positions"):
            correct_position = correct_pheno_pos_keys[i]
            # Get the original index to find correct source chunk and position
            original_idx = pos_to_original_idx[correct_position]
            chk = original_idx // num_positions + start_chunk
            pos_index_in_chk = original_idx % num_positions
            current_position = f"0/{pos_index_in_chk}/0"

            hsc_correct_name = convert_position_to_hcs(correct_position)

            # zarr v3 stores chunks under a literal "c/" prefix dir: 0/c/{t}/...
            dst = Path(dest_path, Path(hsc_correct_name, "0", "c", "0"))
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() or dst.is_symlink(): # TODO (aliddell: symlinks are not valid in Zarr V3, so this may need to be adjusted for compatibility)
                continue

            src = Path(
                source_path,
                f"phenotyping_well_A{chk}_1.zarr",
                Path(current_position, "0", "0"),
            )
            os.symlink(src, dst, target_is_directory=True)
    else:
        # if augmentation, create augmented data (new zarr store)
        print("Applying orientation changes and writing new data.")

        # Print channel transformations upfront
        print("\nChannel transformations:")
        for c_dst, channel_name in enumerate(channel_list):
            if channel_name == "BF":
                flipud, fliplr, rot90 = phase_flipud, phase_fliplr, phase_rot90
            elif channel_name == "GFP":
                flipud, fliplr, rot90 = gfp_flipud, gfp_fliplr, gfp_rot90
            elif channel_name == "mCherry":
                flipud, fliplr, rot90 = mCherry_flipud, mCherry_fliplr, mCherry_rot90
            elif channel_name == "Cy5":
                flipud, fliplr, rot90 = cy5_flipud, cy5_fliplr, cy5_rot90
            else:
                flipud, fliplr, rot90 = False, False, 0

            transforms = []
            if flipud:
                transforms.append("flipud")
            if fliplr:
                transforms.append("fliplr")
            if rot90 != 0:
                transforms.append(f"rot90={rot90}")
            transform_str = ", ".join(transforms) if transforms else "none"
            print(f"  {channel_name}: {transform_str}")
        print()

        def _process_augmented_position(i: int):
            correct_position = correct_pheno_pos_keys[i]
            # Get the original index to find correct source chunk and position
            original_idx = pos_to_original_idx[correct_position]
            chk = original_idx // num_positions + start_chunk
            pos_index_in_chk = original_idx % num_positions
            source_position = f"0/{pos_index_in_chk}/0"

            hsc_correct_name = convert_position_to_hcs(correct_position)

            # Read source data at position level (skip plate metadata parsing)
            source_zarr_path = source_path / f"phenotyping_well_A{chk}_1.zarr"
            try:
                with open_ome_zarr(source_zarr_path, mode="r") as source_hcs:
                    source_ds = source_hcs[source_position]
                    source_data = np.asarray(source_ds.data)
                    src_names = list(source_ds.channel_names)

                # Warn if source data is empty
                if not source_data.any():
                    print(
                        f"[WARN] Position {i}: Source data is all zeros for '{source_position}' in chunk {chk}"
                    )
            except KeyError as e:
                print(
                    f"[ERROR] Position {i}: '{source_position}' not found in {source_zarr_path.name}"
                )
                print(
                    f"  Trying to get positions from chunk {chk}, index {pos_index_in_chk}"
                )
                return
            except Exception as e:
                print(
                    f"[ERROR] Position {i}: Failed to read '{source_position}' from {source_zarr_path.name}: {e}"
                )
                return

            T, C_src, Z, Y, X = source_data.shape

            # Open dest position directly (skip plate metadata parsing)
            dest_hcs = open_ome_zarr(dest_path, mode="r+")
            dest_arr = dest_hcs[hsc_correct_name].data
            _, C_dst, _, _, _ = dest_arr.shape

            out_chunk = np.zeros(
                (int(T), int(C_dst), int(Z), int(Y), int(X)), dtype=dest_arr.dtype
            )

            # Map dest channels to source indices (by name, or positional if generic)
            src_index_map = build_channel_index_map(src_names, channel_list)
            channels_copied = 0
            for c_dst in range(int(C_dst)):
                src_idx = src_index_map[c_dst]
                if src_idx is not None and int(src_idx) < int(C_src):
                    channel_name = (
                        channel_list[c_dst] if c_dst < len(channel_list) else None
                    )
                    if channel_name == "BF":
                        flipud, fliplr, rot90 = phase_flipud, phase_fliplr, phase_rot90
                    elif channel_name == "GFP":
                        flipud, fliplr, rot90 = gfp_flipud, gfp_fliplr, gfp_rot90
                    elif channel_name == "mCherry":
                        flipud, fliplr, rot90 = (
                            mCherry_flipud,
                            mCherry_fliplr,
                            mCherry_rot90,
                        )
                    elif channel_name == "Cy5":
                        flipud, fliplr, rot90 = (
                            cy5_flipud,
                            cy5_fliplr,
                            cy5_rot90,
                        )
                    else:
                        flipud, fliplr, rot90 = False, False, 0

                    if not any([flipud, fliplr, rot90 != 0]):
                        arr_c = source_data[:, int(src_idx), ...]
                        out_chunk[:, c_dst, ...] = np.asarray(arr_c)
                    else:
                        for t in range(int(T)):
                            for z in range(int(Z)):
                                img_slice = source_data[
                                    int(t), int(src_idx), int(z), :, :
                                ]
                                aug_slice = augment_tile(
                                    img_slice,
                                    flipud=flipud,
                                    fliplr=fliplr,
                                    rot90=rot90,
                                )
                                out_chunk[int(t), int(c_dst), int(z), :, :] = (
                                    np.asarray(aug_slice)
                                )
                    channels_copied += 1
                else:
                    continue

            # Write to destination
            dest_arr[:] = out_chunk
            dest_hcs.close()

        # Choose workers and run in parallel
        # Close the shared handle before parallel to avoid metadata races
        try:
            ds.close()
        except Exception:
            pass

        # For image augmentation (I/O bound), use realistic memory estimates:
        # - No heavy model loading needed
        # - Each worker processes one image at a time
        # - Actual memory per worker: ~300-500 MB (small images, simple augmentation)
        n_jobs = get_optimal_workers(
            use_gpu=False,
            model_ram_gb=0.05,  # Minimal model overhead
            data_ram_gb=0.05,  # Single image in memory (~500MB per worker total)
            verbose=True,
        )
        n_jobs = max(1, int(min(n_jobs, total_positions)))
        Parallel(n_jobs=n_jobs)(
            delayed(_process_augmented_position)(i)
            for i in tqdm(range(total_positions), desc="Processing positions")
        )

    # Validate output images
    from cyclops_utils.io.zarr_utils import _validate_output_images

    _validate_output_images(dest_path, n_samples=5)

    return


@versioned_function("v1.0")
def link_tracking(experiment: str) -> None:
    """
    Clean up and re-organize the 5x tracking data, importantly remanes the FOV positions,
    which are needed for stitching
    Args:
        experiment (str):
            Experiment name i.e. ops{num}_{YYYMMDD}
    """
    print(f"linking tracking for {experiment}")
    dataset = OpsDataset(experiment)

    # Prefer fast partition raw_convert/ if it exists, fallback to dragonfly
    fast_convert = dataset.experiment_path_fast / "0-convert" / "live_imaging" / "raw_convert"
    dragonfly_convert = dataset.lc_dragonfly_dir / "0-convert"
    if fast_convert.exists() and any(fast_convert.glob("tracking_*.zarr")):
        source_path = fast_convert
        print(f"  Using fast partition: {source_path}")
    else:
        source_path = dragonfly_convert
        print(f"  Using dragonfly: {source_path}")
    dest_path = dataset.store_paths["lc_5x"]

    # Discover available tracking chunks dynamically (e.g., tracking_1.zarr, tracking_2.zarr, ...)
    tracking_zarrs = natsort.natsorted(glob.glob(str(source_path / "tracking_*.zarr")))
    if not tracking_zarrs:
        print(f"No tracking_*.zarr chunks found in {source_path}")
        return

    # Refuse to link if any discovered raw_convert tracking store isn't fully
    # written yet (link step fired before raw_to_zarr finished). Only guards the
    # fast raw_convert path; the dragonfly fallback is pre-existing data.
    if source_path == fast_convert:
        _assert_raw_convert_complete(tracking_zarrs, kind="tracking")

    # Read actual position indices directly from source zarr files (source of truth)
    source_position_indices = set()  # Use set to avoid duplicates

    for zpath in tracking_zarrs:
        ds_temp = open_ome_zarr(zpath, mode="r")
        positions_list = list(ds_temp.positions())

        if not positions_list:
            print(f"Warning: No positions found in {zpath}")
            continue

        for pos_path, _ in positions_list:
            # pos_path is like "0/148/0" - extract the middle number as the source index
            parts = pos_path.split('/')
            if len(parts) >= 2:
                source_idx = int(parts[1])
                source_position_indices.add(source_idx)

    if not source_position_indices:
        print(f"ERROR: No positions found in any tracking zarr files")
        return

    source_position_indices = sorted(list(source_position_indices))

    print(f"Found {len(source_position_indices)} unique positions across {len(tracking_zarrs)} tracking zarr chunks")
    print(f"Source position index range: {min(source_position_indices)} to {max(source_position_indices)}")
    if len(source_position_indices) <= 10:
        print(f"Source indices: {source_position_indices}")

    # Load position list ONLY to map source indices to human-readable names
    # Check raw_convert/ first for position list
    pos_list_path = dataset.config_paths["lc_5x_position_list"]
    fast_pos_list = source_path / "tracking_position_list.json"
    if fast_pos_list.exists():
        pos_list_path = fast_pos_list
    with open(pos_list_path, "r") as f:
        correct_tracking_pos_list = json.load(f)

    all_pos_in_config = list(correct_tracking_pos_list.keys())

    # Map source indices to position names
    correct_tracking_pos_names = [all_pos_in_config[idx] for idx in source_position_indices]
    pos_to_original_idx = {pos: idx for idx, pos in enumerate(all_pos_in_config)}

    print(f"Using {len(correct_tracking_pos_names)} positions from source data")
    print(f"Position range: {correct_tracking_pos_names[0]} to {correct_tracking_pos_names[-1]}")

    # Rebuild from scratch: every position is relinked below.
    async_delete_path(dest_path)
    source_positions = []
    all_channels = []
    shapes = []
    chunks = []
    scales = []
    timepoints = []
    dtypes = []
    positions_per_chunk = []  # Number of positions in each chunk

    for zpath in tracking_zarrs:
        ds_chunks = open_ome_zarr(zpath, mode="r")
        positions_list = list(ds_chunks.positions())
        if not positions_list:
            print(f"Warning: No positions found in {zpath}")
            continue
        positions_per_chunk.append(len(positions_list))
        source_positions.extend([p[0] for p in positions_list])
        # Get channel names and metadata from the first position
        _, pos = positions_list[0]
        if hasattr(ds_chunks, "channel_names"):
            all_channels.extend(ds_chunks.channel_names)
        elif hasattr(pos, "channel_names"):
            all_channels.extend(pos.channel_names)
        shapes.append(pos.data.shape)
        chunks.append(pos.data.chunks)
        scales.append(pos.scale)
        timepoints.append(pos.data.shape[0])
        dtypes.append(pos.data.dtype)

    if not source_positions:
        raise ValueError(f"No positions found in any tracking zarrs at {source_path}")

    source_positions = list(dict.fromkeys(source_positions))
    all_channels = list(dict.fromkeys(all_channels))
    assert all([s == shapes[0] for s in shapes])
    assert all([c == chunks[0] for c in chunks])
    # assert(all([s == scales[0] for s in scales]))
    assert all([d == dtypes[0] for d in dtypes])

    czyx_shape = shapes[0][1:]
    chunk_size = chunks[0]

    dtype = dtypes[0]
    sizeT = sum(timepoints)

    # HARDCODE scale and channel names
    all_channels = ["BF"]
    scale = [1.0, 1.0, 25.0, 1.3, 1.3]

    # note: MicroManager metadata is not transferred
    # Create positions if not in resume mode
    if choice in ("create", "overwrite"):
        # Convert position names to HCS format (e.g., "A1-Site_0" -> "A/1/0")
        hcs_positions = []
        for correct_name in correct_tracking_pos_names:
            hsc_correct_name = convert_position_to_hcs(correct_name)
            hcs_positions.append(hsc_correct_name)

        # Use fast precreation for all positions at once
        create_hcs_store_fast(
            store_path=dest_path,
            positions=hcs_positions,
            shape=(sizeT,) + czyx_shape,
            chunks=chunk_size,
            dtype=dtype,
            scale=scale,
            channel_names=all_channels,
            version="0.5",
        )

    # Open the store for subsequent operations
    ds = open_ome_zarr(dest_path, mode="a")

    for correct_name in tqdm(correct_tracking_pos_names, desc="Linking"):
        # Get the original index - this is consistent across all zarr chunks
        original_idx = pos_to_original_idx[correct_name]

        # All tracking zarr chunks contain the same positions with the same indices
        # The position path uses the original index directly
        source_position = f"0/{original_idx}/0"

        hsc_correct_name = convert_position_to_hcs(correct_name)
        # zarr v3 stores chunks under a literal "c/" prefix dir: 0/c/{t}/...
        # Symlinking 0/c/{t} -> source 0/{t} is valid because the internal
        # {c_idx}/{z}/{y}/{x} structure is identical between v2 and v3.
        c_dir = Path(dest_path, hsc_correct_name, "0", "c")
        c_dir.mkdir(parents=True, exist_ok=True)
        t = 0
        for zpath, t_chk in zip(tracking_zarrs, timepoints):
            for _t in range(t_chk):
                src = Path(zpath) / source_position / "0" / str(_t)
                dst = c_dir / str(t)
                os.symlink(src, dst, target_is_directory=True)
                t += 1

    return
