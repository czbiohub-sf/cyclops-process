from pathlib import Path
import shutil

import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from iohub import open_ome_zarr
from iohub.ngff import TransformationMeta
from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.hpc.resource_manager import get_optimal_workers, _measure_ram
from cyclops_utils.profiling.decorators import versioned_function
from cyclops_utils.io.zarr_utils import _validate_output_images
from cyclops_utils.io.zarr_precreate import create_hcs_store_fast


def _test_projection_ram(pos, source_path, slices, projection):
    """Test function to measure RAM usage for one position's Z projection (max or sum)."""
    with open_ome_zarr(source_path / pos, layout="fov", mode="r") as ds:
        if slices == "all":
            data = ds.data
        else:
            slice_list = [int(num) for num in slices]
            data = ds.data[:, :, slice_list, :, :]

    if str(projection).lower() == "sum":
        proj = np.sum(data, axis=2)
    else:
        proj = np.max(data, axis=2)
    return proj


def _focus_slice_ranges_from_tilt(tilt_base, positions, z_depth) -> dict:
    """Per-FOV in-focus z-slice range from the tilt-calibration subtile offsets.

    Reads ``<tilt_base>/<well_tag>/tilt_params_all.csv`` (written by the tilt
    reconstruction), where each row is a subtile with a ``z_offset``. The focus
    slice for a subtile is ``round(z_depth//2 - z_offset)``: the reconstruction
    defines ``z_offset = focus_idx - Z//2``, but the max-proj source z-axis is
    flipped relative to that fit, so the sign is inverted here (a -3 offset
    selects the +3 slice). A FOV's focus range is the min..max across its
    subtiles — capturing the focal tilt across the FOV.

    Returns ``{position: [i0, i1, ...]}`` for positions found in the CSVs;
    positions without tilt data are omitted (caller falls back).
    """
    import csv

    half = z_depth // 2
    well_tags = sorted({"_".join(p.split("/")[:2]) for p in positions})
    pos_minmax: dict = {}
    for wt in well_tags:
        well_dir = tilt_base / wt
        # Two on-disk layouts: a consolidated tilt_params_all.csv (newer runs)
        # or per-tile split files under csvs/ (e.g. ops0166). Read whichever exist.
        csv_paths = list(well_dir.glob("tilt_params_all.csv")) + \
            list((well_dir / "csvs").glob("tilt_params_*.csv"))
        for csv_path in csv_paths:
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    pos = row.get("position")
                    try:
                        fi = int(round(half - float(row["z_offset"])))
                    except (KeyError, TypeError, ValueError):
                        continue
                    fi = max(0, min(fi, z_depth - 1))
                    lo, hi = pos_minmax.get(pos, (fi, fi))
                    pos_minmax[pos] = (min(lo, fi), max(hi, fi))
    return {pos: list(range(lo, hi + 1)) for pos, (lo, hi) in pos_minmax.items()}


@versioned_function("v1.0")
def create_max_projection(
    experiment: str,
    process: str,
    slices: list | str = "all",
    projection: str = "max",  # "max" or "sum"
) -> None:
    # slices:
    #   "all" (default) -> project every Z plane (standard behavior).
    #   "focus" (alias "auto") -> lc_20x projects each FOV over its per-FOV
    #       in-focus range from the tilt calibration.
    #   an explicit list (e.g. [3, 4]) -> project exactly those Z planes.

    # # Max projection is memory-light: ~0.5GB per worker for model/data
    # num_workers = get_optimal_workers(use_gpu=False, model_ram_gb=0.25, data_ram_gb=0.25)
    # print(f"Creating max projection for {experiment} using {num_workers} workers")

    dataset = OpsDataset(experiment)

    if process == "lc_20x":
        source_path = dataset.store_paths["lc_20x_vs"]
        proj_path = dataset.store_paths["lc_20x_vs_max_proj"]
        output_scale = dataset.store_props["20x_scale"]

    elif process == "lc_5x":
        source_path = dataset.store_paths["lc_5x_vs"]
        proj_path = dataset.store_paths["lc_5x_vs_max_proj"]
        output_scale = dataset.store_props["5x_scale"]

    elif process == "lc_20x_fluor":
        # Project raw 20x fluorescence channels into 2D (sum or max over Z)
        source_path = dataset.store_paths["lc_20x"]
        proj_path = dataset.store_paths["lc_20x_fluor_2d"]
        output_scale = dataset.store_props["20x_scale"]

    else:
        raise ValueError("experiment_type must be either 'lc_20x' or 'lc_5x'")

    # Only lc_20x has a per-FOV tilt focus model; elsewhere "focus" == project all.
    if slices in ("focus", "auto") and process != "lc_20x":
        slices = "all"

    print("Gathering metadata from source store using fast glob method...")

    # Fast discovery of position paths using glob, assuming consistent metadata
    position_paths = sorted(source_path.glob("*/*/*"))
    # Filter for actual position directories (must contain a '0' array)
    position_paths = [p for p in position_paths if (p / "0").is_dir()]

    if not position_paths:
        raise ValueError(f"No positions found in {source_path}")

    # Convert to relative string paths for iohub
    position_list = [str(p.relative_to(source_path)) for p in position_paths]

    # "focus"/"auto": project each FOV over its OWN in-focus range rather than a
    # fixed slice pair. The focus range is derived from the tilt-calibration
    # per-subtile z_offsets; FOVs without tilt data fall back to the depth-centered
    # pair [Z//2, Z//2+1]. ("all" projects every plane; an explicit list projects
    # exactly those planes — neither enters this branch.)
    per_pos_slices: dict = {}
    if process == "lc_20x" and slices in ("focus", "auto"):
        with open_ome_zarr(source_path / position_list[0], layout="fov", mode="r") as _fov:
            _z = int(_fov["0"].shape[-3])
        _center = _z // 2
        slices = [_center, _center + 1]  # depth-centered fallback
        recon_dir = dataset.store_paths["lc_20x_phase_3d_optimized"].parent
        tilt_base = recon_dir / "tilt_calibration" / "pheno"
        per_pos_slices = _focus_slice_ranges_from_tilt(tilt_base, position_list, _z)
        if per_pos_slices:
            print(f"[max-proj] {_z}-slice stack: per-FOV focus ranges from "
                  f"tilt CSVs for {len(per_pos_slices)}/{len(position_list)} FOVs "
                  f"(fallback {slices})")
        else:
            print(f"[max-proj] {_z}-slice stack: no tilt CSVs found, "
                  f"using depth-centered slices {slices}")

    # Measure RAM usage on first position to determine optimal workers
    measured_ram_gb = _measure_ram(
        _test_projection_ram, position_list[0], source_path, slices, projection
    )
    proj_label = str(projection).lower()
    print(
        f"Measured RAM usage for one position's {proj_label} projection: {measured_ram_gb:.3f} GB"
    )
    num_workers = get_optimal_workers(
        use_gpu=False, model_ram_gb=measured_ram_gb, data_ram_gb=0.0
    )
    print(f"Creating {proj_label} projection for {experiment} using {num_workers} workers")

    # Read metadata from the first position and top-level attributes, assuming uniformity
    with open_ome_zarr(source_path, mode="r") as ds:
        # Get top-level info without scanning all positions
        channel_names = ds.channel_names
        # For fluorescence, keep only non-BF channels
        selected_channel_indices = None
        if process == "lc_20x_fluor":
            try:
                selected_channel_indices = [
                    i for i, n in enumerate(channel_names) if str(n) != "BF"
                ]
                channel_names = [channel_names[i] for i in selected_channel_indices]
            except Exception:
                # Fallback: assume channel 0 is BF, use others
                first_pos_tmp = ds[position_list[0]]
                Ctmp = int(first_pos_tmp.data.shape[1])
                selected_channel_indices = list(range(1, Ctmp))
                channel_names = [f"ch{i}" for i in selected_channel_indices]

        # Get position-specific info from only the FIRST position
        first_pos = ds[position_list[0]]
        input_shape = first_pos.data.shape
        dtype = first_pos.data.dtype

        # Print metadata: pixel size / scales (Z, Y, X)
        try:
            scales = first_pos.scale
            if scales and len(scales) >= 3:
                # Scales are in (T, C, Z, Y, X) order
                print(
                    f"Input zarr pixel sizes - Z: {scales[-3]}, Y: {scales[-2]}, X: {scales[-1]}"
                )
        except Exception as e:
            print(f"Could not retrieve pixel size metadata from input zarr: {e}")

    # If there are no fluorescence channels to project, skip gracefully
    if process == "lc_20x_fluor":
        no_fluor_channels = (
            channel_names is None
            or (isinstance(channel_names, (list, tuple)) and len(channel_names) == 0)
            or (
                selected_channel_indices is not None
                and len(selected_channel_indices) == 0
            )
        )
        if no_fluor_channels:
            print(
                "No fluorescence channels (non-BF) found in source. Skipping 2D projection and continuing."
            )
            return

    output_store_transform = TransformationMeta(type="scale", scale=output_scale)
    if process == "lc_20x_fluor" and channel_names is not None:
        out_C = len(channel_names)
    else:
        out_C = input_shape[1]
    output_shape = (input_shape[0], out_C, 1, input_shape[3], input_shape[4])

    # Pre-allocate the output zarr store sequentially to prevent race conditions
    print(f"Initializing output store at {proj_path}")

    # Always recreate from scratch — never resume (stale slices are unhelpful).
    if proj_path.exists():
        shutil.rmtree(proj_path)

    # Use fast_zarr_precreate for O(1) scaling instead of iohub's O(n) overhead
    # Chunks: (1, 1, 1, Y, X) - chunk per 2D slice
    chunks = (1, 1, 1, output_shape[3], output_shape[4])

    # Scale: (T, C, Z, Y, X) from output_scale which is [T, C, Z, Y, X]
    scale_tuple = tuple(output_scale) if output_scale else (1.0, 1.0, 1.0, 1.0, 1.0)

    print(f"  Using fast_zarr_precreate for {len(position_list)} positions...")
    create_hcs_store_fast(
        store_path=proj_path,
        positions=position_list,
        shape=output_shape,
        chunks=chunks,
        dtype=dtype,
        scale=scale_tuple,
        channel_names=channel_names,
    )

    # Define worker function for parallel processing
    # Open positions directly with layout="fov" to skip plate metadata parsing.
    # See iohub_perf_issue.md for details on the O(N^2) overhead this avoids.
    def _process_and_write_projection(pos):
        pos_slices = per_pos_slices.get(pos, slices)  # per-FOV focus range or fallback
        with open_ome_zarr(source_path / pos, layout="fov", mode="r") as ds:
            if pos_slices == "all":
                data = ds.data  # (T,C,Z,Y,X)
            else:
                slice_list = [int(num) for num in pos_slices]
                data = ds.data[:, :, slice_list, :, :]

        # Select fluorescence channels if requested
        if process == "lc_20x_fluor" and selected_channel_indices is not None:
            data = data[:, selected_channel_indices, :, :, :]

        # Project over Z
        if str(projection).lower() == "sum":
            proj = np.sum(data, axis=2)
        else:
            proj = np.max(data, axis=2)

        # Ensure float32
        # proj = proj.astype(np.float32, copy=False)
        ome_proj = np.expand_dims(proj, axis=(2))

        with open_ome_zarr(proj_path / pos, layout="fov", mode="r+") as store:
            store.data[:] = ome_proj

    # Run in parallel
    print(f"Applying {proj_label} projection in parallel...")
    Parallel(n_jobs=num_workers)(
        delayed(_process_and_write_projection)(pos)
        for pos in tqdm(position_list, desc="Projecting positions")
    )
    print(f"{proj_label.capitalize()} projection complete.")

    # Validate output store
    try:
        _validate_output_images(proj_path, n_samples=3)
    except Exception as e:
        print(f"Warning: Output validation failed: {e}")

    return
