from __future__ import annotations

import os
import yaml
import threading
from tqdm import tqdm
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cellpose import models

import shutil
try:
    import cupy as cp
except ModuleNotFoundError:  # optional GPU dep; used only in GPU runtime paths
    cp = None
import numpy as np
import dask.array as da
try:
    from cupyx.scipy import ndimage as cundi
except ModuleNotFoundError:  # optional GPU dep; used only in GPU runtime paths
    cundi = None
try:
    import torch
except ModuleNotFoundError:  # optional GPU dep; used only in GPU runtime paths
    torch = None
import scipy.ndimage

from skimage.transform import downscale_local_mean
from skimage.exposure import equalize_adapthist
from dask.distributed import Client, LocalCluster, as_completed, get_worker
from iohub.ngff import open_ome_zarr, TransformationMeta
from stitch.connect import read_shifts_biahub
from stitch.stitch.assemble import get_output_shape
from cyclops_utils.data.image_utils import augment_tile

from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.profiling.decorators import versioned_function
from cyclops_utils.hpc.resource_manager import get_optimal_workers
from cyclops_utils.data.filesystem import (
    ensure_output_path,
    async_delete_path,
)
from cyclops_utils.hpc.gpu_utils import _setup_gpu_environment
from cyclops_utils.hpc.parallel_utils import (
    _cleanup_worker_memory,
    MultiGPUCluster,
)
from cyclops_utils.hpc.resource_manager import _measure_vram, compute_gpu_workers
from cyclops_utils.io.zarr_utils import (
    _group_shifts_by_position,
    _discover_positions,
    _maybe_sample_positions,
    _resolve_output_path_for_debug,
    _validate_output_images,
    _validate_all_positions_have_data,
    add_missing_zarr_metadata,
)
from cyclops_utils.io.zarr_precreate import create_hcs_store_fast
from cyclops_process.utils.segmentation_utils import (
    torch_fastremap,
    match_labels,
    _remove_edge_labels,
)


# ==============================================================================
# UTILITY FUNCTIONS FOR REDUCING BOILERPLATE
# ==============================================================================


def preprocess_pheno_cells(
    image_data: np.ndarray,
    channel_0: int = 0,
    channel_1: int = 1,
    clahe_params: dict = None,
    debug_save_path: str = None,
) -> np.ndarray:
    """
    Preprocessing function for pheno_cells segmentation.

    Takes max projection of channels 0 and 1, then applies CLAHE (Contrast Limited
    Adaptive Histogram Equalization) to enhance contrast for better cellpose segmentation.

    Parameters
    ----------
    image_data : np.ndarray
        Input image with shape (T, C, Z, Y, X) or (C, Z, Y, X)
    channel_0 : int, optional
        First channel index for max projection (default: 0)
    channel_1 : int, optional
        Second channel index for max projection (default: 1)
    clahe_params : dict, optional
        Parameters to pass to equalize_adapthist. Key parameters:
        - 'clip_limit': Controls contrast enhancement (default: 0.1)
          Lower (0.01-0.02) = subtle, higher (0.05-0.10) = aggressive
        - 'kernel_size': Tuple (height, width) for local region size (default: None = 1/8 of image)
          Smaller = more local contrast, larger = more global
    debug_save_path : str, optional
        If provided, saves before/after preprocessing images to this path

    Returns
    -------
    np.ndarray
        Preprocessed image with enhanced contrast, shape (Y, X)
    """
    if clahe_params is None:
        clahe_params = {"clip_limit": 0.01}

    # Extract channel(s) - optimize for single channel case when channel_0 == channel_1
    single_channel_mode = channel_0 == channel_1

    if image_data.ndim == 5:  # (T, C, Z, Y, X)
        ch0 = image_data[0, channel_0, 0, :, :]  # Primary channel
        ch1 = None if single_channel_mode else image_data[0, channel_1, 0, :, :]
    elif image_data.ndim == 4:  # (C, Z, Y, X)
        ch0 = image_data[channel_0, 0, :, :]
        ch1 = None if single_channel_mode else image_data[channel_1, 0, :, :]
    else:
        raise ValueError(f"Unexpected image shape: {image_data.shape}")

    # Store original dtype
    original_dtype = ch0.dtype

    # Normalize primary channel to [0, 1] using percentiles
    ch0_min, ch0_max = np.percentile(ch0, [1, 99.5])
    ch0_norm = np.clip(
        (ch0.astype(np.float32) - ch0_min) / (ch0_max - ch0_min + 1e-8), 0, 1
    )

    # Single channel mode: skip second channel and max projection
    if single_channel_mode:
        max_proj = ch0_norm

        # Save normalized channel if debug mode
        if debug_save_path:
            import tifffile
            base, ext = os.path.splitext(debug_save_path)
            tifffile.imwrite(
                f"{base}_ch0_normalized{ext}", ch0_norm.astype(np.float32)
            )
    else:
        # Two channel mode: normalize second channel and take max projection
        ch1_min, ch1_max = np.percentile(ch1, [1, 99.5])
        ch1_norm = np.clip(
            (ch1.astype(np.float32) - ch1_min) / (ch1_max - ch1_min + 1e-8), 0, 1
        )

        # Save normalized channels if debug mode
        if debug_save_path:
            import tifffile
            base, ext = os.path.splitext(debug_save_path)
            tifffile.imwrite(
                f"{base}_ch0_nucleus_normalized{ext}", ch0_norm.astype(np.float32)
            )
            tifffile.imwrite(
                f"{base}_ch1_membrane_normalized{ext}", ch1_norm.astype(np.float32)
            )

        # Take max projection of normalized channels
        stacked = np.stack([ch0_norm, ch1_norm], axis=0)
        max_proj = np.max(stacked, axis=0)

    # Save pre-CLAHE max projection image if debug path provided
    if debug_save_path:
        import tifffile

        base, ext = os.path.splitext(debug_save_path)
        pre_clahe_path = f"{base}_pre_clahe{ext}"
        tifffile.imwrite(pre_clahe_path, max_proj.astype(np.float32))

    # Apply CLAHE to the max projection
    enhanced = equalize_adapthist(max_proj, **clahe_params)

    # Convert back to original dtype range
    if np.issubdtype(original_dtype, np.integer):
        enhanced = (enhanced * np.iinfo(original_dtype).max).astype(original_dtype)
    else:
        # Keep as float in [0, 1] range for float types
        enhanced = enhanced.astype(original_dtype)

    # Save post-CLAHE enhanced image if debug path provided
    if debug_save_path:
        import tifffile

        base, ext = os.path.splitext(debug_save_path)
        post_clahe_path = f"{base}_post_clahe{ext}"
        tifffile.imwrite(post_clahe_path, enhanced.astype(np.float32))

    return enhanced


def _setup_experiment_config(experiment: str, process: str) -> dict:
    """Extract experiment configuration logic."""
    dataset = OpsDataset(experiment)

    # Load experiment config for overrides (e.g., flipud for bioe.ops2)
    exp_cfg = {}
    exp_cfg_path = dataset.config_paths.get("exp_config")
    if exp_cfg_path and Path(exp_cfg_path).exists():
        with open(exp_cfg_path, "r") as f:
            exp_cfg = yaml.safe_load(f) or {}

    if process == "iss":
        all_shifts = read_shifts_biahub(dataset.config_paths["iss_stitch"])
        seg_params = exp_cfg.get("segment_and_stitch_iss_params", {})
        print(f"[segment ISS] flipud={seg_params.get('flipud', True)}, fliplr={seg_params.get('fliplr', False)}, rot90={seg_params.get('rot90', 0)}, tile_size={dataset.store_props['tile_size']}")
        return {
            "dataset": dataset,
            "source_path": dataset.store_paths["iss_drift_corrected"],
            "output_path": dataset.store_paths["iss_segmentation"],
            "all_shifts": all_shifts,
            "tile_size": dataset.store_props["tile_size"],
            "rot90": seg_params.get("rot90", 0),
            "t_pts": 1,
            "flipud": seg_params.get("flipud", True),
            "fliplr": seg_params.get("fliplr", False),
            "model_type": "nuclei",
            "diameter": 30,
            "output_scale": dataset.store_props["5x_scale"],
            "channel_index": 0,
            "use_preprocess": False,
        }
    elif process == "track":
        all_shifts = read_shifts_biahub(dataset.config_paths["lc_5x_stitch"])
        return {
            "dataset": dataset,
            "source_path": dataset.store_paths[
                "lc_5x_vs"
            ],  # TODO: decide between direct 2d recon (lc_5x_phase_2d) segmentation or tracking VS 2d segmentation (lc_5x_vs)
            "output_path": dataset.store_paths["lc_5x_segmentation"],
            "all_shifts": all_shifts,
            "tile_size": (2048, 2048),
            "rot90": 1,
            "t_pts": 4,
            "flipud": False,
            "fliplr": True,
            "model_type": "nuclei",
            "diameter": 30,
            "output_scale": dataset.store_props["5x_scale"],
            "channel_index": 0,
            "use_preprocess": False,
        }
    # NOTE: process=="pheno" (legacy 5x nuclei seg -> lc_20x_segmentation) is
    # retired; nuclei are segmented at native 20x by submit_nuclei_segmentation_jobs.
    elif (
        process == "pheno_cells"
    ):  # NOTE: attempting segment and stitch of VS membranes to be combined in pheno assembled with symlink
        all_shifts = read_shifts_biahub(dataset.config_paths["lc_20x_stitch"])
        return {
            "dataset": dataset,
            "source_path": dataset.store_paths["lc_20x_vs_max_proj"],
            "output_path": dataset.store_paths["lc_20x_segmentation_cells"],
            "all_shifts": all_shifts,
            "tile_size": (2048, 2048),
            "rot90": 0,
            "t_pts": 1,
            "flipud": False,
            "fliplr": False,
            "model_type": "cyto3",
            "diameter": 100,
            "output_scale": dataset.store_props["20x_scale"],
            "channel_index": 1,
            "use_preprocess": True,
        }
    else:
        raise ValueError(f"Unknown process: {process}")


def _setup_direct_config(
    input_store_path: Path,
    output_store_path: Path,
    input_config_path: Path = None,
    tile_size: tuple = (2048, 2048),
) -> dict:
    """Set up a dummy config to access config args without
    conforming to ops_experiment file constraints"""
    dataset = OpsDataset("")  # dummy dataset for store properties
    config = {
        "dataset": dataset,
        "source_path": Path(input_store_path),
        "output_path": Path(output_store_path),
        "tile_size": tile_size,
        "t_pts": 1,
        "model_type": "cyto3",  # default
        "diameter": 100,
        "output_scale": dataset.store_props["20x_scale"],
        "channel_index": 0,
    }

    if input_config_path:
        config["all_shifts"] = read_shifts_biahub(input_config_path)

    return config


def _create_cellpose_model(model_type: str, gpu: bool = True) -> models.CellposeModel:
    """Create a Cellpose model with standard settings."""
    from cellpose import models
    return models.CellposeModel(
        gpu=gpu,
        # model_type=model_type, # TODO: model type is no longer used in Cellpose v4.0.1
        device=torch.device("cuda" if gpu else "cpu"),
        # diam_mean=100,
    )


def _initialize_segmentation_store(
    output_path: Path,
    position_list: list,
    dataset: OpsDataset,
    output_scale: list,
    source_path: Path = None,
) -> None:
    """Initialize output Zarr store for basic segmentation using fast precreation."""
    print(f"Initializing Zarr store at {output_path}")
    print(f"Pre-creating {len(position_list)} positions using fast method...")

    if source_path:
        # Get shape from first position (all tiles should have same shape)
        with open_ome_zarr(source_path, mode="r") as source_store:
            first_pos_shape = source_store[position_list[0]]["0"].shape
            seg_shape = (1, 1, 1, first_pos_shape[3], first_pos_shape[4])

        # Use fast precreation method (128x faster for large stores)
        create_hcs_store_fast(
            store_path=output_path,
            positions=position_list,
            shape=seg_shape,
            chunks=dataset.store_props["chunk_size"],
            dtype=np.int32,
            scale=output_scale,
            channel_names=["segmentation"],
        )


def _initialize_stitching_store(
    output_path: Path,
    grouped_shifts: dict,
    dataset: OpsDataset,
    tile_size: tuple,
    source_path: Path,
    process: str = None,
) -> None:
    """Initialize output Zarr store for stitched segmentation using fast precreation."""
    print(f"Initializing final Zarr store for stitched output at {output_path}")
    print(f"Pre-creating {len(grouped_shifts)} positions using fast method...")

    # Group positions by shape (in case they vary)
    with open_ome_zarr(source_path, mode="r") as source_store:
        positions_by_shape = {}

        for g in grouped_shifts.keys():
            shifts = grouped_shifts[g]
            position = f"{g}/0"  # g is "row/col" (e.g. "A/1", "B/2") -> "A/1/0"
            first_tile_pos = list(shifts.keys())[0]

            # Get actual time points from source data
            source_t_pts = source_store[first_tile_pos]["0"].shape[0]

            # For iss/pheno/pheno_cells, only segment first timepoint; for track, segment all timepoints
            if process in ("iss", "pheno", "pheno_cells"):
                actual_t_pts = 1
            else:
                actual_t_pts = source_t_pts

            final_shape_xy = get_output_shape(shifts, tile_size=tile_size)
            final_shape = (actual_t_pts, 1, 1) + final_shape_xy

            # Group by shape
            if final_shape not in positions_by_shape:
                positions_by_shape[final_shape] = []
            positions_by_shape[final_shape].append(position)

    # Create positions grouped by shape using fast method. Wells can stitch to
    # differing shapes (>1 group); the first group creates the store ("w"), the
    # rest append ("a") — otherwise each "w" call would wipe prior positions.
    for i, (shape, positions) in enumerate(positions_by_shape.items()):
        print(f"  Creating {len(positions)} positions with shape {shape}")
        create_hcs_store_fast(
            store_path=output_path,
            positions=positions,
            shape=shape,
            chunks=dataset.store_props["chunk_size"],
            dtype=np.int32,
            scale=dataset.store_props["5x_scale"],
            channel_names=["segmentation"],
            mode="w" if i == 0 else "a",
            version="0.5",
        )


def run_cellpose(
    image: np.ndarray, model_type: str, diameter: float, model_cache: dict = None
) -> tuple:
    """
    Runs Cellpose on a given image.

    A simple wrapper around model.eval() to facilitate caching the model
    within a dask worker.

    Parameters
    ----------
    image : np.ndarray
        The input image for segmentation.
    model_type : str
        The Cellpose model type ('cyto3', 'nuclei', etc.).
    diameter : float
        The estimated diameter of the objects to segment.
    model_cache : dict, optional
        A dictionary to cache the loaded model, by default None.

    Returns
    -------
    tuple
        A tuple containing masks, flows, and styles.
    """
    from cellpose import models
    if model_cache is None:
        model_cache = {}

    model_key = (model_type, "cuda")
    if model_key not in model_cache:
        model_cache[model_key] = models.CellposeModel(
            gpu=True,
            device=torch.device(
                "cuda"
            ),  # TODO: model_type is no longer used in Cellpose v4.0.1
        )
    model = model_cache[model_key]

    # Note: Cellpose expects channels_last for 2D, so (y, x, c)
    # Our data is channel-first (c, y, x) or just (y, x)
    # model.eval handles this detection.
    masks, flows, styles = model.eval(image, diameter=diameter, channels=[0, 0])
    return masks, flows, styles


def run_parallel_segmentation(
    source_zarr: str | Path,
    output_zarr: str | Path,
    channel_to_segment: str,
    output_mask_name: str,
    model_type: str,
    diameter: float,
    num_workers: int = None,
):
    """
    Generic, parallelized segmentation function using Dask.

    This function reads a specific channel from a source Zarr store, segments it
    tile by tile using Cellpose in a distributed manner, and writes the
    resulting masks to a specified group in an output Zarr store.

    Parameters
    ----------
    source_zarr : str or Path
        Path to the input OME-Zarr store.
    output_zarr : str or Path
        Path to the output OME-Zarr store to write masks to.
    channel_to_segment : str
        Name of the channel in the source store to use for segmentation.
    output_mask_name : str
        Name of the array to save the segmentation mask under (in the 'seg' group).
    model_type : str
        Name of the Cellpose model to use (e.g., 'cyto3', 'nuclei').
    diameter : float
        Expected average diameter of objects for Cellpose.
    num_workers : int, optional
        Number of Dask workers. If None, it will be determined automatically.
    """
    source_zarr = Path(source_zarr)
    output_zarr = Path(output_zarr)

    with open_ome_zarr(source_zarr, mode="r") as ds:
        # Get a list of position path strings, not Position objects
        position_list = [path for path, _ in ds.positions()]
        channel_index = ds.channel_names.index(channel_to_segment)

    if not position_list:
        raise ValueError(f"No positions found in {source_zarr}")

    # Auto-determine workers if not specified
    if num_workers is None:
        num_workers = get_optimal_workers(use_gpu=True)

    # Dask worker setup
    available_gpus = _setup_gpu_environment()
    workers_per_gpu = num_workers // max(1, len(available_gpus))
    print(
        f"Running parallel segmentation for '{output_mask_name}' with {num_workers} workers "
        f"({workers_per_gpu}/GPU)."
    )

    def segment_worker(pos_path):
        """Worker function to read data, segment it, and return the mask."""
        image_to_segment, source_scale, chunks = None, None, None
        try:
            model_cache = {}
            with open_ome_zarr(source_zarr, mode="r") as ds:
                source_pos = ds[pos_path]
                image_data = source_pos.data
                source_scale = source_pos.scale
                chunks = image_data.chunks
                image_to_segment = np.squeeze(image_data[0, channel_index, ...])

            masks, _ = run_cellpose(
                image_to_segment, model_type, diameter, model_cache=model_cache
            )
            seg = np.expand_dims(masks, axis=(0, 1, 2)).astype(np.int32)

            return pos_path, seg, source_scale, chunks

        finally:
            del image_to_segment, source_scale, chunks, model_cache
            torch.cuda.empty_cache()

    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    with MultiGPUCluster(available_gpus, workers_per_gpu) as mgc, \
         open_ome_zarr(output_zarr, mode="r+") as dest_store:

        print(f"Dask dashboard: {mgc.dashboard_link}")
        futures = mgc.map(segment_worker, position_list)

        for future in tqdm(
            as_completed(futures),
            total=len(position_list),
            desc=f"Segmenting and writing {output_mask_name}",
        ):
            pos_path, seg_mask, scale, chunks = future.result()

            if seg_mask is not None:
                dest_pos = dest_store[pos_path]
                # Use iohub's create_label method for proper OME-Zarr structure
                dest_pos.create_label(
                    name=output_mask_name,
                    data=seg_mask,
                    chunks=chunks,
                    transform=[TransformationMeta(type="scale", scale=scale)],
                )

    print(f"Finished parallel segmentation for '{output_mask_name}'.")


def dilate_masks(segmentation_path, itters: int, device: str = "cpu") -> None:
    """
    Dilate the nuclear segmentation masks
    - ISS spots are often at the edges of nuclei, or slightly outside of them,
    dialating the nuclear masks captures some of these spots

    Notes:
     - Many spots still map back to no nuclei
    """

    with open_ome_zarr(segmentation_path, mode="r+") as ds:
        position_list = [a[0] for a in ds.positions()]

        for pos in position_list:
            print(f"Processing {pos} on {device}")

            data = ds[pos].data[0, 0, 0, :, :]

            if device == "gpu":
                cp_data = cp.asarray(data)
                for i in tqdm(range(itters), desc="Dilating masks"):
                    temp = cundi.grey_dilation(cp_data, size=(3, 3))
                    cp_data = cp.where(cp_data == 0, temp, cp_data)
                dilated = np.expand_dims(cp_data.get(), (0, 1, 2))
            elif device == "cpu":
                for i in tqdm(range(itters), desc="Dilating masks"):
                    temp = scipy.ndimage.grey_dilation(data, size=(3, 3))
                    data = np.where(data == 0, temp, data)
                dilated = np.expand_dims(data, (0, 1, 2))
            else:
                raise ValueError(f"Unknown device: {device}")

            # Check if image "1" already exists and overwrite its data directly
            pos_group = ds[pos]
            if "1" in pos_group:
                print(f"Dilated mask '1' already exists for {pos}, overwriting data...")
                pos_group["1"][:] = dilated
            else:
                print(f"Creating new dilated mask '1' for {pos}...")
                pos_group.create_image(
                    name="1",
                    data=dilated,
                    chunks=ds[pos].data.chunks,
                    transform=[TransformationMeta(type="scale", scale=ds[pos].scale)],
                )
    return


@versioned_function("v1.0")
def segmentation(
    experiment: Optional[str] = None,
    process: Optional[str] = None,
    input_store_path: Optional[Path] = None,
    output_store_path: Optional[Path] = None,
    num_workers: int = None,
    debug_n_tiles: int | None = None,
    debug_output_suffix: str = "_debug",
    use_preprocess: bool = False,
    clahe_clip_limit: float = 0.01,
    clahe_kernel_size: int = None,
    positions: Optional[list[str]] = None,
) -> None:
    """
    Wrapper to apply a cellpose segmentation tile-wise to each fov in a zarr store.
    This function processes each FOV in parallel.

    Parameters
    ----------
    positions : list of str, optional
        List of specific position names to segment (e.g., ['A/1/028034', 'B/2/005012']).
        If None, all positions are segmented.
    """
    # Setup configuration
    if experiment is None:
        if input_store_path is None or output_store_path is None:
            raise ValueError(
                "Either experiment or input/output directories must be provided."
            )
        config = _setup_direct_config(input_store_path, output_store_path)
    else:
        config = _setup_experiment_config(experiment, process)

    # Extract config values
    dataset = config["dataset"]
    source_path = config["source_path"]
    output_path = config["output_path"]
    model_type = config["model_type"]
    diameter = config.get("diameter")
    output_scale = config["output_scale"]
    channel_index = config.get("channel_index", 0)
    use_preprocess = config.get("use_preprocess", False)

    if use_preprocess:
        print("Using preprocessing for segmentation for process: {process}")
        print(f"CLAHE clip limit: {clahe_clip_limit}")
        print(f"CLAHE kernel size: {clahe_kernel_size}")
    else:
        print("NOT using preprocessing for segmentation")

    # Discover positions and setup model
    print("Discovering positions using fast glob method...")
    position_list = _discover_positions(source_path)

    # Handle debug mode (fast balanced sampling)
    if debug_n_tiles is not None and debug_n_tiles > 0:
        print(
            f"DEBUG: Sampling {debug_n_tiles} positions for debug run (balanced selection)."
        )
        position_list = _maybe_sample_positions(position_list, int(debug_n_tiles))
        output_path = _resolve_output_path_for_debug(
            output_path, debug_n_tiles, debug_output_suffix
        )

    # Add specific positions if provided
    if positions is not None:
        print(f"Adding {len(positions)} specified positions to the list...")
        # Validate positions exist
        valid_positions = [
            p for p in positions if p in _discover_positions(source_path)
        ]
        if not valid_positions:
            raise ValueError(
                f"None of the specified positions found in source store: {positions}"
            )
        # Add to position list (avoid duplicates)
        position_list = valid_positions
        print(f"Total positions to segment: {len(position_list)}")
        if debug_n_tiles is None or debug_n_tiles == 0:
            output_path = _resolve_output_path_for_debug(
                output_path, len(position_list), debug_output_suffix
            )

    # Determine optimal workers if not specified
    if num_workers is None:
        def _vram_test():
            model = _create_cellpose_model("cyto3", gpu=True)
            with open_ome_zarr(str(source_path), mode="r") as ds:
                vs_fov = np.asarray(ds[position_list[0]].data[0, channel_index, 0, :, :])
            model.eval(vs_fov[::-1], flow_threshold=0.7, diameter=diameter)
            del model

        model_vram_gb = _measure_vram(_vram_test)
        print(f"Measured model VRAM: {model_vram_gb} GB")
        num_workers = get_optimal_workers(use_gpu=True, model_vram_gb=model_vram_gb)

    print(f"Segmenting for {experiment} using {num_workers} workers")

    # Always run fresh (no resume) — create_hcs_store_fast (mode="w") overwrites
    # any existing store. Resume was dropped because its "precreated → resume"
    # heuristic misclassified half-populated stores and silently no-op'd the step.
    if output_path.exists():
        print(f"Overwriting existing Zarr store at {output_path}")
    _initialize_segmentation_store(
        output_path, position_list, dataset, output_scale, source_path
    )

    # Setup parallel processing
    print("Starting parallel segmentation with Dask...")
    available_gpus = _setup_gpu_environment()
    workers_per_gpu = num_workers // max(1, len(available_gpus))

    def seg_and_write(pos):
        """Worker function to segment one position and write the result."""
        model = None
        vs_fov = out = seg = None
        try:
            model = _create_cellpose_model(model_type)

            with open_ome_zarr(source_path / pos, layout="fov", mode="r") as ds:
                if use_preprocess:
                    # Load full data for preprocessing (channels 0 and 1)
                    full_data = np.asarray(ds.data[:])

                    # Setup CLAHE parameters
                    clahe_params = {"clip_limit": clahe_clip_limit}
                    if clahe_kernel_size is not None:
                        clahe_params["kernel_size"] = (
                            clahe_kernel_size,
                            clahe_kernel_size,
                        )

                    # Setup debug path if in debug mode - save to debug folder in output path
                    debug_path = None
                    if debug_n_tiles is not None:
                        debug_folder = (
                            output_path.parent / f"{output_path.stem}_preprocess_debug"
                        )
                        debug_folder.mkdir(exist_ok=True, parents=True)
                        pos_safe = pos.replace("/", "_")
                        debug_path = str(debug_folder / f"preprocess_{pos_safe}.tif")

                    vs_fov = preprocess_pheno_cells(
                        full_data,
                        channel_0=0,
                        channel_1=1,
                        clahe_params=clahe_params,
                        debug_save_path=debug_path,
                    )
                else:
                    vs_fov = np.asarray(ds.data[0, channel_index, 0, :, :])

            out, _, _ = model.eval(vs_fov[::-1], flow_threshold=0.7, diameter=diameter)
            # Flip back to match original image orientation
            out = out[::-1]
            seg = np.expand_dims(np.asarray(out), axis=(0, 1, 2))

            with open_ome_zarr(output_path / pos, layout="fov", mode="r+") as store:
                store["0"][:] = seg
        finally:
            _cleanup_worker_memory(model, vs_fov, out, seg)
        return

    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    with MultiGPUCluster(available_gpus, workers_per_gpu) as mgc:
        print("\n--- Dask Cluster Information ---")
        print(f"Dashboard: {mgc.dashboard_link}")
        print("--------------------------------\n")

        futures = mgc.map(seg_and_write, position_list)

        print("Segmenting positions...")
        for future in tqdm(as_completed(futures), total=len(position_list)):
            future.result()

    print("Segmentation complete.")

    # Validate output images
    _validate_output_images(output_path, n_samples=3, raise_on_blank=True)

    return


def _create_segmentation_worker_function(
    source_path: Path,
    temp_seg_path: Path,
    process: str,
    available_gpus: list,
    flipud: bool,
    fliplr: bool,
    rot90: int,
    tile_size: tuple,
    t_pts: int,
    dataset: OpsDataset,
    channel_index: int,
    model_type: str,
    diameter: float,
    use_preprocess: bool = False,
    clahe_params: dict = None,
):
    """Create the segmentation worker function for Dask."""

    def _segment_and_augment_tile(positions_batch):
        """Dask worker function. Segments a BATCH of tiles in a tight loop.

        Accepts a list of position strings. Processing multiple tiles per Dask task
        eliminates per-tile scheduling overhead (~5ms/tile × 7035 tiles = ~35s).
        The Cellpose model and seg function are resolved once per batch, not per tile.
        """
        import zarr

        # Cache the model per worker (persistent across batches)
        worker = get_worker()
        if not hasattr(worker, "cellpose_model"):
            worker.cellpose_model = _create_cellpose_model(model_type)
        model = worker.cellpose_model

        # Resolve seg function ONCE per batch (not per tile).
        # These closures capture `model` which is stable for this worker.
        def seg_fcn_iss(tile):
            tile = tile[0, 0, 0, :, :]
            ds_masks, _, _ = model.eval(tile, flow_threshold=0.7)
            return np.expand_dims(ds_masks, axis=0)

        def seg_fcn_track(tile):
            tile = tile[:, channel_index, 0, :, :]
            seg_list = []
            for i in range(tile.shape[0]):
                masks, _, _ = model.eval(tile[i], flow_threshold=0.7)
                seg_list.append(np.expand_dims(masks, axis=0))
            return np.concatenate(seg_list, axis=0)

        def seg_fcn_pheno(tile):
            tile = tile[0, 0, 0, :, :]
            h, w = tile.shape
            ds_tile = tile.reshape(h // 4, 4, w // 4, 4).mean(axis=(1, 3))
            ds_masks, _, _ = model.eval(ds_tile, flow_threshold=0.7)
            return np.expand_dims(ds_masks, axis=0)

        def seg_fcn_pheno_cells(tile):
            if use_preprocess:
                print("Preprocessing pheno cells tile...")
                print(f"CLAHE clip limit: {clahe_params['clip_limit']}")
                tile_preprocessed = preprocess_pheno_cells(
                    tile,
                    channel_0=0,
                    channel_1=1,
                    clahe_params=clahe_params or {"clip_limit": 0.01},
                    debug_save_path=None,
                )
                print("Using diameter: ", diameter)
                ds_masks, _, _ = model.eval(
                    tile_preprocessed, flow_threshold=0.7, diameter=diameter
                )
            else:
                print("Not using preprocessing for pheno cells")
                tile = tile[0, 0, 0, :, :]
                ds_masks, _, _ = model.eval(
                    tile, flow_threshold=0.7, diameter=diameter
                )
            return np.expand_dims(ds_masks, axis=0)

        seg_functions = {
            "iss": seg_fcn_iss,
            "track": seg_fcn_track,
            "pheno": seg_fcn_pheno,
            "pheno_cells": seg_fcn_pheno_cells,
            "cell_paint": seg_fcn_pheno,
        }
        if process not in seg_functions:
            raise ValueError(f"Unknown process: {process}")
        seg_fcn = seg_functions[process]

        # Determine read strategy once (same for all tiles in batch)
        slim_read = process in ("iss", "pheno", "cell_paint") or (
            process == "pheno_cells" and not use_preprocess
        )

        # Init async write state
        if not hasattr(worker, "_write_thread"):
            worker._write_thread = None

        def _write_tile(path, data):
            dest_arr = zarr.open(str(path), mode="r+")
            dest_arr[:] = data

        # Use the prefetch pipeline for all single-channel Cellpose processes.
        # Background threads read+preprocess tiles from NFS.
        # N images are stacked as "z-slices" and passed to run_net() in one
        # GPU forward pass, bypassing eval()'s per-image loop.
        # For track (T>1 timepoints), each (tile, timepoint) pair is treated
        # as an independent image in the pipeline.
        # pheno_cells with preprocessing needs 2 channels + CLAHE — uses fallback.
        use_prefetch_pipeline = process in ("pheno", "iss", "cell_paint", "track")

        n_done = 0
        if use_prefetch_pipeline:
            from concurrent.futures import ThreadPoolExecutor
            from collections import deque
            from cellpose import transforms as cp_transforms
            from cellpose import dynamics as cp_dynamics
            from cellpose import core as cp_core

            PREFETCH_DEPTH = 32  # tiles buffered ahead of GPU
            NUM_READ_THREADS = 4  # parallel NFS readers
            # Images per GPU forward pass. 8 is the sweet spot:
            # 12% faster than 1, fits on any GPU (14GB VRAM), gains plateau beyond 16.
            MULTI_BATCH = int(os.environ.get("SEG_MULTI_BATCH", "8"))

            def _read_and_preprocess(pos):
                """Read, preprocess, and normalize tile images (runs in thread).

                Returns a list of (img_norm, t_idx) tuples. For single-timepoint
                processes (pheno/iss/cell_paint) returns 1 element. For track
                returns T elements (one per timepoint).
                """
                source_arr = zarr.open(str(source_path / pos / "0"), mode="r")
                if process == "track":
                    # Read all timepoints, single channel: (T, 1, 1, Y, X)
                    tile_data = source_arr[:, channel_index:channel_index+1, 0:1, :, :]
                    results = []
                    for t in range(tile_data.shape[0]):
                        img = tile_data[t, 0, 0, :, :]
                        img_conv = cp_transforms.convert_image(img)
                        img_norm = cp_transforms.normalize_img(
                            img_conv, normalize=True, norm3D=False
                        )
                        results.append((img_norm, t))
                    return results
                else:
                    # Single timepoint: (1, 1, 1, Y, X)
                    tile_data = source_arr[0:1, channel_index:channel_index+1, 0:1, :, :]
                    img = tile_data[0, 0, 0, :, :]
                    if process == "pheno" or process == "cell_paint":
                        h, w = img.shape
                        img = img.reshape(h // 4, 4, w // 4, 4).mean(axis=(1, 3))
                    img_conv = cp_transforms.convert_image(img)
                    img_norm = cp_transforms.normalize_img(
                        img_conv, normalize=True, norm3D=False
                    )
                    return [(img_norm, 0)]

            def _postprocess_and_write(pos, masks_by_timepoint):
                """Augment and write tile result (runs in thread).

                masks_by_timepoint: dict of {t_idx: masks_2d_array}.
                For single-timepoint: {0: masks}. For track: {0: m0, 1: m1, ...}.
                """
                if not masks_by_timepoint:
                    raise ValueError(
                        f"Empty masks_by_timepoint for position {pos}. "
                        f"Expected at least 1 timepoint mask."
                    )
                # Stack timepoints in order: (T, Y, X)
                t_indices = sorted(masks_by_timepoint.keys())
                stacked_masks = np.stack(
                    [masks_by_timepoint[t] for t in t_indices], axis=0
                )
                augmented_seg = augment_tile(
                    stacked_masks, flipud=flipud, fliplr=fliplr, rot90=rot90
                )
                if augmented_seg.ndim == 2:
                    augmented_seg = augmented_seg[np.newaxis, np.newaxis, np.newaxis, ...]
                elif augmented_seg.ndim == 3:
                    augmented_seg = augmented_seg[:, np.newaxis, np.newaxis, ...]
                # Validate shape matches destination before writing
                dest_arr = zarr.open(str(temp_seg_path / pos / "0"), mode="r+")
                if augmented_seg.shape != dest_arr.shape:
                    raise ValueError(
                        f"Shape mismatch for {pos}: data {augmented_seg.shape} "
                        f"vs dest {dest_arr.shape}. "
                        f"masks_by_timepoint had keys {list(masks_by_timepoint.keys())}"
                    )
                dest_arr[:] = augmented_seg

            io_pool = ThreadPoolExecutor(
                max_workers=NUM_READ_THREADS + 1,
                thread_name_prefix="seg_io",
            )
            # Parallel mask computation: compute_masks for N images concurrently.
            # follow_flows uses torch.grid_sample (releases GIL), so threads
            # can overlap GPU dynamics work across images.
            NUM_MASK_THREADS = int(os.environ.get("SEG_MASK_THREADS", "4"))
            masks_pool = ThreadPoolExecutor(
                max_workers=NUM_MASK_THREADS,
                thread_name_prefix="seg_masks",
            )

            def _fast_get_masks_torch(pt, inds, shape0, rpad=20, max_size_fraction=0.4):
                """Optimized get_masks_torch using F.max_pool2d instead of custom loops.

                The original uses max_pool1d with Python loops over kernel offsets
                (5 iterations × 2 axes = 10 loop iterations with slice+maximum).
                F.max_pool2d is a single fused CUDA kernel — much faster.
                """
                import torch as _torch
                import torch.nn.functional as _F
                import fastremap

                device = pt.device
                shape0_arr = np.array(shape0)

                pt = pt + rpad
                pt = _torch.clamp(pt, min=0)
                for i in range(len(pt)):
                    pt[i] = _torch.clamp(pt[i], max=shape0_arr[i] + rpad - 1)

                shape = tuple(shape0_arr + 2 * rpad)

                # Build histogram via sparse COO → dense
                coo = _torch.sparse_coo_tensor(
                    pt, _torch.ones(pt.shape[1], device=device, dtype=_torch.int), shape)
                h1 = coo.to_dense()
                del coo

                # Find seeds: local maxima with count > 10
                # Use F.max_pool2d (single fused kernel) instead of custom max_pool1d loops
                h1_4d = h1.unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)
                hmax1 = _F.max_pool2d(h1_4d, kernel_size=5, stride=1, padding=2)
                hmax1 = hmax1.squeeze().int()
                seeds1 = _torch.nonzero((h1 - hmax1 > -1e-6) * (h1 > 10))
                del hmax1, h1_4d

                if len(seeds1) == 0:
                    return np.zeros(shape0, dtype="uint16")

                npts = h1[tuple(seeds1.T)]
                isort1 = npts.argsort()
                seeds1 = seeds1[isort1]
                n_seeds = len(seeds1)

                # Extract 11×11 patches around seeds — vectorized via unfold
                ndim = len(shape0)
                if ndim == 2:
                    # unfold creates sliding window view: (H-10, W-10, 11, 11)
                    patches = h1.unfold(0, 11, 1).unfold(1, 11, 1)
                    h_slc = patches[seeds1[:, 0] - 5, seeds1[:, 1] - 5]  # (n_seeds, 11, 11)
                else:
                    # 3D fallback: use loop
                    h_slc = _torch.zeros((n_seeds, *[11]*ndim), device=device)
                    for k in range(n_seeds):
                        slc = tuple([slice(seeds1[k][j]-5, seeds1[k][j]+6) for j in range(ndim)])
                        h_slc[k] = h1[slc]
                del h1

                seed_masks = _torch.zeros((n_seeds, *[11]*ndim), device=device)
                if ndim == 2:
                    seed_masks[:, 5, 5] = 1
                else:
                    seed_masks[:, 5, 5, 5] = 1

                # Grow seed masks using F.max_pool2d (batched over all seeds)
                for _iter in range(5):
                    sm_4d = seed_masks.unsqueeze(1)  # (n_seeds, 1, 11, 11)
                    sm_4d = _F.max_pool2d(sm_4d, kernel_size=3, stride=1, padding=1)
                    seed_masks = sm_4d.squeeze(1)
                    seed_masks *= (h_slc > 2)
                del h_slc

                # Vectorized label assignment: batch nonzero + scatter
                batch_idx, row_idx, col_idx = _torch.nonzero(seed_masks, as_tuple=True)
                global_row = row_idx + seeds1[batch_idx, 0] - 5
                global_col = col_idx + seeds1[batch_idx, 1] - 5
                del seed_masks

                dtype = _torch.int32 if n_seeds < 2**16 else _torch.int64
                M1 = _torch.zeros(shape, dtype=dtype, device=device)
                M1[global_row, global_col] = (1 + batch_idx).to(dtype)

                M1 = M1[tuple(pt)]
                M1 = M1.cpu().numpy()

                dtype = "uint16" if n_seeds < 2**16 else "uint32"
                M0 = np.zeros(shape0, dtype=dtype)
                M0[inds] = M1

                # Remove big masks
                uniq, counts = fastremap.unique(M0, return_counts=True)
                big = np.prod(shape0) * max_size_fraction
                bigc = uniq[counts > big]
                if len(bigc) > 0 and (len(bigc) > 1 or bigc[0] != 0):
                    M0 = fastremap.mask(M0, bigc)
                fastremap.renumber(M0, in_place=True)
                M0 = M0.reshape(tuple(shape0))
                return M0

            def _compute_masks_fast(dP, cellprob, niter, cellprob_threshold,
                                    flow_threshold, device):
                """Optimized compute_masks: uses _fast_get_masks_torch, no profiling overhead."""
                from cellpose import utils as cp_utils
                import torch as _torch

                if (cellprob > cellprob_threshold).sum() == 0:
                    return np.zeros(cellprob.shape, "uint16")
                inds = np.nonzero(cellprob > cellprob_threshold)
                if len(inds[0]) == 0:
                    return np.zeros(cellprob.shape, "uint16")

                # follow_flows (GPU Euler integration)
                p_final = cp_dynamics.follow_flows(
                    dP * (cellprob > cellprob_threshold) / 5.,
                    inds=inds, niter=niter, device=device)
                if not _torch.is_tensor(p_final):
                    p_final = _torch.from_numpy(p_final).to(device, dtype=_torch.int)
                else:
                    p_final = p_final.int()

                # get_masks — optimized with F.max_pool2d
                if device.type == "mps":
                    p_final = p_final.to(_torch.device("cpu"))
                mask = _fast_get_masks_torch(
                    p_final, inds, dP.shape[1:], max_size_fraction=0.4)
                del p_final

                # remove_bad_flow_masks (flow QC)
                if mask.max() > 0 and flow_threshold is not None and flow_threshold > 0:
                    mask = cp_dynamics.remove_bad_flow_masks(
                        mask, dP, threshold=flow_threshold, device=device)

                if mask.max() < 2**16 and mask.dtype != "uint16":
                    mask = mask.astype("uint16")

                # fill_holes_and_remove_small_masks (CPU cleanup)
                mask = cp_utils.fill_holes_and_remove_small_masks(mask, min_size=15)
                return mask

            try:
                import time as _time

                def _run_gpu_batch(stacked, batch_count):
                    """Run GPU forward pass (can run in thread — GIL released by torch)."""
                    return cp_core.run_net(
                        model.net, stacked,
                        batch_size=batch_count * 16,
                        bsize=256,
                    )

                # Flatten tile reads into individual (pos, t_idx, img) entries.
                # _read_and_preprocess returns [(img, t_idx), ...] — one entry
                # for pheno/iss/cell_paint, T entries for track.
                # The flat_q feeds images to _collect_batch regardless of source.
                flat_q = deque()  # (pos, t_idx, img) ready for batching
                read_q = deque()  # (pos, Future) pending NFS reads

                def _prime_reads(n):
                    """Submit n tile reads to the io_pool."""
                    nonlocal next_prefetch_idx
                    while len(read_q) < n and next_prefetch_idx < len(positions_batch):
                        pos = positions_batch[next_prefetch_idx]
                        read_q.append((pos, io_pool.submit(_read_and_preprocess, pos)))
                        next_prefetch_idx += 1

                def _drain_reads_to_flat():
                    """Move completed reads into flat_q."""
                    while read_q and read_q[0][1].done():
                        pos, fut = read_q.popleft()
                        for img, t_idx in fut.result():
                            flat_q.append((pos, t_idx, img))

                def _ensure_flat_images(n):
                    """Ensure at least n images in flat_q (blocking)."""
                    while len(flat_q) < n and read_q:
                        pos, fut = read_q.popleft()
                        for img, t_idx in fut.result():
                            flat_q.append((pos, t_idx, img))
                    _prime_reads(PREFETCH_DEPTH)

                next_prefetch_idx = 0
                _prime_reads(PREFETCH_DEPTH)

                # Track how many timepoints each tile expects
                # (determined from first read result per position)
                tile_t_counts = {}  # pos -> total timepoints expected
                # Accumulate masks per tile until all timepoints arrive
                pending_tiles = {}  # pos -> {t_idx: masks}

                prev_write_future = None
                n_images = 0  # total images consumed

                # Count total images (tiles × timepoints)
                # For track: each tile has T timepoints
                # For pheno/iss: 1 image per tile
                total_images = len(positions_batch)  # approximate, refined as reads complete

                # Timing accumulators
                t_collect = 0.0
                t_gpu = 0.0
                t_masks = 0.0
                n_batches = 0

                def _collect_batch():
                    """Collect up to MULTI_BATCH images from flat queue."""
                    _ensure_flat_images(min(MULTI_BATCH, MULTI_BATCH))
                    batch_count = min(MULTI_BATCH, len(flat_q))
                    if batch_count == 0:
                        return [], [], None
                    batch_keys = []  # (pos, t_idx)
                    batch_imgs = []
                    for _ in range(batch_count):
                        pos, t_idx, img = flat_q.popleft()
                        batch_keys.append((pos, t_idx))
                        batch_imgs.append(img)
                    stacked = np.stack(batch_imgs, axis=0)
                    _prime_reads(PREFETCH_DEPTH)
                    return batch_keys, batch_imgs, stacked

                def _process_mask_results(batch_keys, mask_futures):
                    """Collect mask results, accumulate per tile, write complete tiles."""
                    nonlocal prev_write_future, n_done
                    for (pos, t_idx), mf in zip(batch_keys, mask_futures):
                        masks = mf.result()
                        # Track timepoint count from first read
                        if pos not in tile_t_counts:
                            # Infer from read results — for pheno it's 1,
                            # for track it's the number of images read for this pos
                            tile_t_counts[pos] = 1  # updated below if track
                        if pos not in pending_tiles:
                            pending_tiles[pos] = {}
                        pending_tiles[pos][t_idx] = masks

                    # Write tiles that have all timepoints
                    for (pos, _) in batch_keys:
                        if pos not in pending_tiles:
                            continue
                        # For track, we need to know T. Infer from the zarr shape
                        # on first encounter, or from the number of images queued.
                        t_expected = tile_t_counts.get(pos, 1)
                        if len(pending_tiles[pos]) >= t_expected:
                            if prev_write_future is not None:
                                prev_write_future.result()
                            prev_write_future = io_pool.submit(
                                _postprocess_and_write, pos, pending_tiles.pop(pos)
                            )
                            n_done += 1

                # Determine T per position (may vary across positions for track)
                if process == "track":
                    total_images = 0
                    for p in positions_batch:
                        _src = zarr.open(str(source_path / p / "0"), mode="r")
                        t = _src.shape[0]
                        tile_t_counts[p] = t
                        total_images += t
                else:
                    for p in positions_batch:
                        tile_t_counts[p] = 1
                    total_images = len(positions_batch)
                _ensure_flat_images(1)

                # Pipeline: overlap GPU forward pass (batch N+1) with
                # CPU mask computation (batch N) using a thread.
                gpu_future = None
                prev_batch_keys = None
                prev_yf = None

                while n_images < total_images or flat_q:
                    t0 = _time.monotonic()
                    batch_keys, batch_imgs, stacked = _collect_batch()
                    if stacked is None:
                        break
                    batch_count = len(batch_keys)
                    t1 = _time.monotonic()
                    t_collect += t1 - t0

                    # Submit GPU work for this batch in a thread
                    gpu_future = io_pool.submit(_run_gpu_batch, stacked, batch_count)

                    # While GPU processes current batch, compute masks
                    # for the PREVIOUS batch (if any) — parallel across images
                    if prev_yf is not None:
                        t2 = _time.monotonic()
                        mask_futures = []
                        for j in range(len(prev_batch_keys)):
                            dP = prev_yf[j, :, :, :2].transpose(2, 0, 1)
                            cellprob = prev_yf[j, :, :, 2]
                            mask_futures.append(masks_pool.submit(
                                _compute_masks_fast,
                                dP, cellprob, niter=100,
                                cellprob_threshold=0.0,
                                flow_threshold=0.7,
                                device=model.device,
                            ))
                        _process_mask_results(prev_batch_keys, mask_futures)
                        t3 = _time.monotonic()
                        t_masks += t3 - t2

                    # Wait for GPU to finish this batch
                    t4 = _time.monotonic()
                    yf, styles = gpu_future.result()
                    t5 = _time.monotonic()
                    t_gpu += t5 - t4  # only counts wait time (overlap is free)

                    prev_batch_keys = batch_keys
                    prev_yf = yf
                    n_images += batch_count
                    n_batches += 1

                # Process the final batch's masks (no next GPU batch to overlap)
                if prev_yf is not None:
                    t2 = _time.monotonic()
                    mask_futures = []
                    for j in range(len(prev_batch_keys)):
                        dP = prev_yf[j, :, :, :2].transpose(2, 0, 1)
                        cellprob = prev_yf[j, :, :, 2]
                        mask_futures.append(masks_pool.submit(
                            _compute_masks_fast,
                            dP, cellprob, niter=100,
                            cellprob_threshold=0.0,
                            flow_threshold=0.7,
                            device=model.device,
                        ))
                    _process_mask_results(prev_batch_keys, mask_futures)
                    t3 = _time.monotonic()
                    t_masks += t3 - t2

                # Wait for final write
                if prev_write_future is not None:
                    prev_write_future.result()

                # Drain any remaining read futures
                for _, fut in read_q:
                    fut.result()

                # Print timing breakdown
                print(f"[Seg timing] {n_batches} batches of {MULTI_BATCH}: "
                      f"collect={t_collect:.1f}s, gpu_wait={t_gpu:.1f}s, "
                      f"masks={t_masks:.1f}s "
                      f"(threads={NUM_MASK_THREADS})")

            except Exception:
                # Drain futures on error to prevent thread leaks
                if prev_write_future is not None:
                    try:
                        prev_write_future.result()
                    except Exception:
                        pass
                for _, fut in read_q:
                    try:
                        fut.result()
                    except Exception:
                        pass
                raise
            finally:
                masks_pool.shutdown(wait=False)
                io_pool.shutdown(wait=False)
        else:
            # Fallback: per-tile eval for track/pheno_cells (multi-timepoint or preprocessing)
            for pos in positions_batch:
                try:
                    source_arr = zarr.open(str(source_path / pos / "0"), mode="r")
                    if slim_read:
                        tile_data = source_arr[0:1, channel_index:channel_index+1, 0:1, :, :]
                    else:
                        tile_data = source_arr[:]

                    seg_new = seg_fcn(tile_data)
                    augmented_seg = augment_tile(
                        seg_new, flipud=flipud, fliplr=fliplr, rot90=rot90
                    )

                    if augmented_seg.ndim == 2:
                        augmented_seg = augmented_seg[np.newaxis, np.newaxis, np.newaxis, ...]
                    elif augmented_seg.ndim == 3:
                        augmented_seg = augmented_seg[:, np.newaxis, np.newaxis, ...]

                    if worker._write_thread is not None:
                        worker._write_thread.join()
                    worker._write_thread = threading.Thread(
                        target=_write_tile,
                        args=(temp_seg_path / pos / "0", augmented_seg),
                    )
                    worker._write_thread.start()
                    n_done += 1

                except Exception:
                    if hasattr(worker, "_write_thread") and worker._write_thread is not None:
                        worker._write_thread.join()
                    raise

        return n_done

    return _segment_and_augment_tile


def _stitch_well_gpu(
    well_position: str,
    shifts: dict,
    tile_size: tuple,
    temp_seg_path,
    output_path,
    final_shape_xy: tuple,
    max_t_pts: int,
):
    """Stitch one well with timepoint parallelism and GPU match_labels.

    Runs as a Dask worker. Uses ThreadPoolExecutor internally so each
    timepoint is processed in parallel (independent canvas, independent
    running_max). GPU is used for match_labels sparse IoU computation.
    """
    from concurrent.futures import ThreadPoolExecutor

    # GPU setup (PID-based assignment, same pattern as Phase 1 segmentation)
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        gpu_id = os.getpid() % num_gpus
        torch.cuda.set_device(gpu_id)
        device = torch.device("cuda")
        print(f"[Stitch {well_position}] Using GPU {gpu_id} (PID={os.getpid()})")
    else:
        device = torch.device("cpu")
        print(f"[Stitch {well_position}] No GPU available, using CPU")

    # Per-timepoint canvases — fully independent, no shared state between threads
    canvases = [
        torch.zeros((1, 1, 1) + final_shape_xy, dtype=torch.int32)
        for _ in range(max_t_pts)
    ]

    def _process_timepoint(t):
        """Process all tiles for one timepoint. Runs in a thread.
        Uses prefetch to overlap NFS reads with GPU match_labels."""
        import zarr
        from collections import deque

        canvas_t = canvases[t]
        running_max = 0
        positions = list(shifts.items())

        PREFETCH = 8

        def _read_and_prep(pos, shift, t):
            """Read tile from NFS and preprocess (runs in reader thread)."""
            pos_arr = zarr.open(str(temp_seg_path / pos / "0"), mode="r")
            if t >= pos_arr.shape[0]:
                return None  # This tile doesn't have this timepoint
            seg_new = pos_arr[t]
            seg_new = np.squeeze(seg_new)
            tile_new = torch.tensor(seg_new.copy(), dtype=torch.int32)
            tile_new = _remove_edge_labels(tile_new)
            tile_new = torch_fastremap(tile_new)
            return tile_new

        # Use a small thread pool for prefetching tile reads
        from concurrent.futures import ThreadPoolExecutor as _TPE
        read_pool = _TPE(max_workers=4, thread_name_prefix="stitch_read")

        try:
            # Prime prefetch queue
            prefetch_q = deque()
            prime_count = min(PREFETCH, len(positions))
            for k in range(prime_count):
                pos, shift = positions[k]
                prefetch_q.append((pos, shift, read_pool.submit(_read_and_prep, pos, shift, t)))
            next_idx = prime_count

            for idx in range(len(positions)):
                pos, shift, read_future = prefetch_q.popleft()
                tile_new = read_future.result()

                # Refill prefetch
                if next_idx < len(positions):
                    npos, nshift = positions[next_idx]
                    prefetch_q.append((npos, nshift, read_pool.submit(_read_and_prep, npos, nshift, t)))
                    next_idx += 1

                if tile_new is None:
                    continue

                i_low, i_high = int(shift[0]), int(shift[0]) + tile_size[0]
                j_low, j_high = int(shift[1]), int(shift[1]) + tile_size[1]
                tile_bulk = canvas_t[..., i_low:i_high, j_low:j_high]

                if tile_new.max() > 0:
                    tile_new[tile_new > 0] += running_max
                    tile_bulk_gpu = tile_bulk[0, 0, 0].to(device)
                    tile_new_gpu = tile_new.to(device)
                    _, remapped_gpu = match_labels(
                        tile_bulk_gpu, tile_new_gpu, threshold=0.1
                    )
                    remapped = remapped_gpu.cpu()
                    tile_bulk[0, 0, 0][remapped > 0] = remapped[remapped > 0].int()

                running_max = max(running_max, tile_bulk.max().item())

            # Drain remaining prefetch
            for _, _, fut in prefetch_q:
                fut.result()
        finally:
            read_pool.shutdown(wait=False)

        return t

    # Parallel timepoint processing via threads.
    # PyTorch CUDA ops release the GIL during kernel execution,
    # so threads overlap NFS I/O with GPU compute.
    print(f"[Stitch {well_position}] Processing {len(shifts)} tiles × {max_t_pts} timepoints...")
    with ThreadPoolExecutor(max_workers=max_t_pts) as executor:
        list(executor.map(_process_timepoint, range(max_t_pts)))

    # Assemble timepoints and write to output store asynchronously.
    # The canvas write (~2.7 GB to NFS) runs in a daemon thread so the
    # next well can start stitching while this well's data writes to disk.
    canvas = torch.cat(canvases, dim=0)  # (T, 1, 1, Y, X)
    canvas_np = canvas.numpy()  # Convert before passing to thread (no GIL issues)
    del canvas, canvases  # Free GPU/CPU memory

    def _write_canvas():
        with open_ome_zarr(output_path / well_position, layout="fov", mode="r+", version="0.5") as store:
            store["0"][:] = canvas_np

    write_thread = threading.Thread(target=_write_canvas, daemon=True)
    write_thread.start()

    print(f"[Stitch {well_position}] Complete ({max_t_pts} timepoints stitched)")
    return f"Completed: {well_position}", write_thread


def _perform_parallel_stitching(
    temp_seg_path: Path,
    output_path: Path,
    grouped_shifts: dict,
    tile_size: tuple,
) -> None:
    """Stitch pre-segmented tiles with timepoint parallelism and GPU match_labels.

    Uses Dask LocalCluster (spawn-based) so workers can access GPU.
    Each well is processed by one Dask worker, which internally parallelizes
    across timepoints using ThreadPoolExecutor.
    """
    print("\n--- Phase 2: Parallel Stitching (GPU + timepoint parallelism) ---")

    # Build work items: one per well (direct zarr — skip NGFF metadata scan)
    import zarr
    work_items = []
    for g, shifts in grouped_shifts.items():
        well_position = f"{g}/0"  # g is "row/col" -> "A/1/0" / "B/2/0"
        max_t_pts = max(
            zarr.open(str(temp_seg_path / pos / "0"), mode="r").shape[0]
            for pos in shifts.keys()
        )
        final_shape_xy = get_output_shape(shifts, tile_size=tile_size)
        work_items.append((
            well_position, shifts, tile_size,
            temp_seg_path, output_path, final_shape_xy, max_t_pts,
        ))

    n_workers = len(work_items)
    print(f"Stitching {n_workers} wells sequentially (GPU match_labels + timepoint parallelism)...")

    # Run stitching in the main process with async canvas writes.
    # The canvas write (~2.7 GB to NFS per well) runs in a background thread
    # while the next well starts stitching — overlapping NFS I/O with GPU work.
    prev_write_thread = None
    for args in work_items:
        # Wait for previous well's write to finish before reusing GPU
        if prev_write_thread is not None:
            prev_write_thread.join()
        result, write_thread = _stitch_well_gpu(*args)
        prev_write_thread = write_thread
        print(f"  {result}")
    # Wait for final well's write
    if prev_write_thread is not None:
        prev_write_thread.join()

    # Cleanup temporary directory (async — rename is instant, delete in background)
    print(f"\n--- Phase 3: Cleaning up temporary directory ---")
    if temp_seg_path.exists():
        async_delete_path(temp_seg_path)
        print(f"Removed {temp_seg_path} (async cleanup)")


def _perform_sequential_stitching(
    temp_seg_path: Path,
    output_path: Path,
    grouped_shifts: dict,
    tile_size: tuple,
) -> None:
    """Perform stitching of pre-segmented tiles, parallelized across wells."""
    print("\n--- Phase 2: Sequential Stitching ---")

    def _stitch_single_well(g, shifts, tile_size, temp_seg_path, output_path):
        """Worker function to stitch a single well."""
        position = f"{g}/0"  # g is "row/col" -> "A/1/0" / "B/2/0"

        try:
            with open_ome_zarr(temp_seg_path, mode="r") as temp_store:
                # Determine actual time points from first tile
                first_pos = list(shifts.keys())[0]
                actual_t_pts = temp_store[first_pos]["0"].shape[0]

                final_shape_xy = get_output_shape(shifts, tile_size=tile_size)
                final_shape = (actual_t_pts, 1, 1) + final_shape_xy
                canvas = torch.zeros(final_shape, dtype=torch.int32)
                running_max = 0

                for pos, shift in shifts.items():
                    i_low, i_high = int(shift[0]), int(shift[0]) + tile_size[0]
                    j_low, j_high = int(shift[1]), int(shift[1]) + tile_size[1]
                    tile_bulk = canvas[..., i_low:i_high, j_low:j_high]

                    # Load the pre-segmented, augmented tile
                    seg_new = temp_store[pos]["0"][:]  # is (t, c, z, y, x)
                    seg_new = np.squeeze(seg_new)  # becomes (t, y, x) or (y,x)
                    if seg_new.ndim == 2:
                        seg_new = seg_new[np.newaxis, ...]  # becomes (t=1, y, x)

                    # Stitching logic
                    tile_new_tensor = torch.tensor(seg_new.copy(), dtype=torch.int32)
                    for t in range(actual_t_pts):
                        tile_new = _remove_edge_labels(tile_new_tensor[t])
                        tile_new = torch_fastremap(tile_new)
                        if tile_new.max() > 0:
                            tile_new[tile_new > 0] = (
                                tile_new[tile_new > 0] + running_max
                            )
                            remapped = match_labels(
                                tile_bulk[t], tile_new, threshold=0.1
                            )[1]
                            tile_bulk[t, 0, 0][remapped > 0] = remapped[
                                remapped > 0
                            ].int()

                    running_max = max(running_max, tile_bulk.max())

            # Write the final stitched canvas to the output store
            with open_ome_zarr(output_path / position, layout="fov", mode="r+") as store:
                seg = canvas.cpu().numpy()
                store["0"][:] = seg

            return f"Completed: {position}"
        except Exception as e:
            return f"Failed: {position} - {str(e)}"

    try:
        # Determine number of parallel workers for stitching
        # Stitching is CPU and memory intensive, so use fewer workers
        from joblib import Parallel, delayed
        from cyclops_utils.hpc.resource_manager import get_optimal_workers

        # Use CPU-bound estimation with high memory requirement per task
        num_stitch_workers = get_optimal_workers(
            use_gpu=False,
            model_ram_gb=1.0,  # Minimal overhead
            data_ram_gb=8.0,   # Stitching is memory-intensive (~8GB per well)
            verbose=True,
        )
        num_stitch_workers = max(1, min(num_stitch_workers, len(grouped_shifts)))

        print(f"Stitching {len(grouped_shifts)} wells using {num_stitch_workers} parallel workers...")

        # Run stitching in parallel across wells
        results = Parallel(n_jobs=num_stitch_workers)(
            delayed(_stitch_single_well)(g, shifts, tile_size, temp_seg_path, output_path)
            for g, shifts in tqdm(grouped_shifts.items(), desc="Stitching Wells")
        )

        # Check for failures and raise if any
        failures = [r for r in results if r.startswith("Failed:")]
        for result in results:
            print(result)
        if failures:
            raise RuntimeError(f"Stitching failed for {len(failures)} position(s):\n" + "\n".join(failures))

    finally:
        # Cleanup temporary directory (async — rename is instant, delete in background)
        print(f"\n--- Phase 3: Cleaning up temporary directory ---")
        if temp_seg_path.exists():
            async_delete_path(temp_seg_path)
            print(f"Removed {temp_seg_path} (async cleanup)")


@versioned_function("v1.0")
def segment_and_stitch(
    experiment: Optional[str] = None,
    process: Optional[str] = None,
    input_store_path: Optional[Path] = None,
    input_config_path: Optional[Path] = None,
    output_store_path: Optional[Path] = None,
    flipud: bool = False,
    fliplr: bool = False,
    rot90: int = 0,
    tile_size: tuple = (2048, 2048),
    full_res_tile_size: int | None = None,
    num_workers: int = None,
    debug_n_tiles: int | None = None,
    debug_output_suffix: str = "_debug",
    use_preprocess: bool = False,
    clahe_clip_limit: float = 0.01,
    clahe_kernel_size: int = None,
) -> None:
    """
    Segment nuclei with a cellpose model, and then apply stitching, using the parameters estimated from
    the phase contrast images. This function is a two-phase process:
    1. Massively Parallel Segmentation: All tiles from all groups are segmented in parallel using Dask
    2. Sequential Stitching: Each group (e.g., well) is stitched sequentially
    """
    print(f"--- Entering segment_and_stitch for process '{process}' ---")

    # Setup configuration
    if experiment is None:
        if input_store_path is None or output_store_path is None:
            raise ValueError(
                "Either experiment or input/output directories must be provided."
            )
        config = _setup_direct_config(
            input_store_path, output_store_path, input_config_path, tile_size
        )
        config.update({"flipud": flipud, "fliplr": fliplr, "rot90": rot90})
        all_shifts = config.get("all_shifts", {})
    else:
        config = _setup_experiment_config(experiment, process)
        all_shifts = config["all_shifts"]

    # Extract config values
    dataset = config["dataset"]
    source_path = config["source_path"]
    output_path = config["output_path"]
    tile_size = config["tile_size"]
    t_pts = config["t_pts"]
    flipud = config.get("flipud", flipud)
    fliplr = config.get("fliplr", fliplr)
    rot90 = config.get("rot90", rot90)
    channel_index = config.get("channel_index", 0)
    use_preprocess = config.get("use_preprocess", False)

    # Scale shifts for cell_paint process (4x downsampling like pheno)
    # This is needed for direct config mode since experiment mode handles it in _setup_experiment_config
    if process == "cell_paint" and experiment is None:
        _full_res = full_res_tile_size or 2048  # default 2048 for dragonfly, override for other microscopes
        downsample_factor = _full_res // tile_size[0]  # e.g., 2048/512 = 4, or 2304/576 = 4
        if downsample_factor > 1:
            print(f"Scaling shifts by 1/{downsample_factor} for {downsample_factor}x downsampling (full_res={_full_res})")
            all_shifts = {k: [int(x / downsample_factor) for x in v] for k, v in all_shifts.items()}


    if use_preprocess:
        print("Using preprocessing for segmentation for process: {process}")
        print(f"CLAHE clip limit: {clahe_clip_limit}")
        print(f"CLAHE kernel size: {clahe_kernel_size}")
    else:
        print("NOT using preprocessing for segmentation")

    # Handle debug mode (fast balanced sampling)
    if debug_n_tiles is not None and debug_n_tiles > 0:
        print(
            f"DEBUG: Sampling {debug_n_tiles} tiles for debug run (balanced selection)."
        )
        # Convert shifts dict to a list of positions and sample
        all_positions = list(all_shifts.keys())
        sampled_positions = _maybe_sample_positions(all_positions, int(debug_n_tiles))
        all_shifts = {k: all_shifts[k] for k in sampled_positions if k in all_shifts}
        output_path = _resolve_output_path_for_debug(
            output_path, debug_n_tiles, debug_output_suffix
        )

    # Setup paths and grouping
    temp_seg_path = output_path.with_name(f"{output_path.name}_temp_seg")
    grouped_shifts = _group_shifts_by_position(all_shifts)
    is_debug_mode = debug_n_tiles is not None and debug_n_tiles > 0

    # Decide output store behavior (create/overwrite/resume/skip)
    # Check output store
    expected_stitched_positions = [f"{g}/0" for g in grouped_shifts.keys()]
    expected_shapes = {
        f"{g}/0": get_output_shape(shifts, tile_size=tile_size)
        for g, shifts in grouped_shifts.items()
    }
    # Always run fresh — overwrite if the store exists, create if not. We don't
    # use decide_overwrite_resume_skip here because its "precreated → resume"
    # heuristic was misclassifying half-populated stores (e.g. populated
    # 20x_nuclear_seg + empty `0`) and silently no-op'ing the whole step.
    sas_choice = "overwrite" if output_path.exists() else "create"
    expected_temp_positions = list(all_shifts.keys())
    temp_seg_choice = "overwrite" if temp_seg_path.exists() else "create"

    # Setup model and determine workers
    model_type = config["model_type"]
    diameter = config["diameter"]

    if num_workers is None:
        # Known VRAM for Cellpose cyto3: 1.718 GB (consistent across GPU types).
        # Skipping _measure_vram saves ~15s of model load + dummy inference.
        model_vram_gb = 1.72
        print(f"Using known model VRAM: {model_vram_gb} GB (Cellpose {model_type})")

        # Use compute_gpu_workers which accounts for per-process CUDA context
        # overhead (~1 GB). The old get_optimal_workers didn't account for this,
        # causing OOM when too many workers were spawned.
        _, num_workers = compute_gpu_workers(
            measured_vram_gb=model_vram_gb,
            target_utilization=0.85,
            cuda_context_overhead_gb=1.0,
        )

    print(f"Segmenting and stitching for {experiment} using {num_workers} workers")

    # Initialize final stitched store according to decision
    # Always trash any existing output store and recreate fresh.
    if sas_choice == "overwrite":
        print(f"Removing existing output store for clean initialization: {output_path}")
        async_delete_path(output_path)
    _initialize_stitching_store(
        output_path, grouped_shifts, dataset, tile_size, source_path, process
    )

    # Initialize temporary store for segmentation
    print(f"\n--- Phase 1: Parallel Segmentation ---")
    actual_t_pts = None
    if temp_seg_choice == "overwrite":
        print(f"Removing existing temp store for clean initialization: {temp_seg_path}")
        async_delete_path(temp_seg_path)
    print(f"Initializing temporary Zarr store for segmented tiles at {temp_seg_path}")
    print(f"Pre-creating {len(all_shifts)} tile positions using fast method...")

    # Group positions by shape (varies by process and time points)
    with open_ome_zarr(source_path, mode="r") as source_store:
        positions_by_shape = {}

        for pos in all_shifts.keys():
            source_pos = source_store[pos]
            source_shape = source_pos["0"].shape
            source_chunks = source_pos["0"].chunks

            # For iss/pheno/pheno_cells, only segment first timepoint; for track, segment all timepoints
            if process in ("iss", "pheno", "pheno_cells", "cell_paint"):
                actual_t_pts = 1
            else:
                actual_t_pts = source_shape[0]

            if process in ("pheno", "cell_paint"):
                seg_shape = (actual_t_pts, 1, 1, tile_size[0], tile_size[1])
            else:
                seg_shape = (
                    actual_t_pts,
                    1,
                    1,
                    source_shape[3],
                    source_shape[4],
                )

            # Group by shape and chunks
            shape_key = (seg_shape, source_chunks)
            if shape_key not in positions_by_shape:
                positions_by_shape[shape_key] = []
            positions_by_shape[shape_key].append(pos)

    # Create positions grouped by shape using fast method
    for (seg_shape, source_chunks), positions in positions_by_shape.items():
        print(f"  Creating {len(positions)} positions with shape {seg_shape}")
        create_hcs_store_fast(
            store_path=temp_seg_path,
            positions=positions,
            shape=seg_shape,
            chunks=source_chunks,
            dtype=np.int32,
            scale=dataset.store_props["5x_scale"],
            channel_names=["segmentation"],
        )

    # Run parallel segmentation
    available_gpus = _setup_gpu_environment()

    # Setup CLAHE parameters if preprocessing is enabled
    clahe_params = None
    if use_preprocess:
        clahe_params = {"clip_limit": clahe_clip_limit}
        if clahe_kernel_size is not None:
            clahe_params["kernel_size"] = (clahe_kernel_size, clahe_kernel_size)

    worker_function = _create_segmentation_worker_function(
        source_path,
        temp_seg_path,
        process,
        available_gpus,
        flipud,
        fliplr,
        rot90,
        tile_size,
        actual_t_pts,
        dataset,
        channel_index,
        model_type,
        diameter,
        use_preprocess,
        clahe_params,
    )

    # Clear parent CUDA_VISIBLE_DEVICES (MultiGPUCluster sets it per-cluster)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    # Scale workers per GPU based on VRAM. Each worker uses ~14 GB
    # (model + MB=8 activations + CUDA context). More workers = more
    # concurrent inference via separate CUDA contexts, filling SMs that
    # a single 512×512 Cellpose pipeline can't saturate.
    # Benchmarked on 2000 tiles (H100):
    #   2 workers: 189.9s, 74% SM, 28 GB VRAM
    #   3 workers: 200.9s, 72% SM, 41 GB VRAM (5.8% slower — contention)
    # 3 workers only helps on H200 (140 GB) where memory bandwidth
    # is sufficient for 3 concurrent contexts.
    # - A40/A6000 (48 GB): 2 workers (70% → 81% SM)
    # - H100/A100-80 (80 GB): 2 workers (35% → 74% SM)
    # - H200 (140 GB): 3 workers (34% → 48% SM)
    # Use nvidia-smi instead of torch.cuda to avoid initializing CUDA in
    # the parent process. torch.cuda.mem_get_info() triggers CUDA init, which
    # poisons child processes — all Dask workers end up on GPU 0 regardless
    # of CUDA_VISIBLE_DEVICES set by MultiGPUCluster.
    import subprocess as _sp
    _nvsmi = _sp.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5)
    total_vram_gb = int(_nvsmi.stdout.strip().split("\n")[0]) / 1024
    workers_per_gpu = int(os.environ.get("SEG_WORKERS_PER_GPU", 0)) or (
        3 if total_vram_gb > 100 else 2
    )
    print(f"[Seg] {workers_per_gpu} worker(s)/GPU "
          f"(VRAM={total_vram_gb:.0f}GB)")

    with MultiGPUCluster(available_gpus, workers_per_gpu, memory_limit=0) as mgc:
        print(f"\n--- Dask Cluster Information for Segmentation ---")
        print(f"Dashboard: {mgc.dashboard_link}")
        print("--------------------------------\n")

        # Split tiles evenly across workers. Each worker runs the full
        # prefetch pipeline internally with ThreadPoolExecutor.
        all_positions = list(all_shifts.keys())
        n_total_workers = workers_per_gpu * len(available_gpus)
        tiles_per_batch = max(1, len(all_positions) // n_total_workers)
        batches = [
            all_positions[i:i + tiles_per_batch]
            for i in range(0, len(all_positions), tiles_per_batch)
        ]
        print(f"Submitting {len(batches)} batches ({tiles_per_batch} tiles/batch, {len(all_positions)} total)")

        futures = mgc.map(worker_function, batches)

        print("Segmenting all tiles...")
        n_completed = 0
        for future in tqdm(as_completed(futures), total=len(batches)):
            n_completed += future.result()

        # Drain any in-flight async writes before closing the cluster.
        # Each worker may have a _write_thread still running for its last tile.
        # Submit one drain task per worker to ensure all writes complete.
        def _drain_writes(_):
            w = get_worker()
            if hasattr(w, "_write_thread") and w._write_thread is not None:
                w._write_thread.join()
                w._write_thread = None
        drain_futures = mgc.map(_drain_writes, list(range(n_total_workers)))
        for f in drain_futures:
            f.result()

    print("--- Phase 1: Parallel Segmentation Complete ---")

    # Validate temp segmentation tiles before stitching
    _validate_output_images(temp_seg_path, n_samples=3, raise_on_blank=True)

    # Perform GPU-parallel stitching (timepoint parallelism + GPU match_labels)
    _perform_parallel_stitching(temp_seg_path, output_path, grouped_shifts, tile_size)
    print("--- Phase 2: Parallel Stitching Complete ---")

    # Apply post-processing
    if process == "iss":
        print("Dilating masks")
        dilate_masks(output_path, itters=8, device="gpu")
    if process == "pheno":
        print("Upscaling nuclear segmentations for phenotyping...")
        upscale_nuclear_segmentations(experiment=experiment, overwrite=True)

    print("Segmentation and stitching complete.")
    return


def upscale_nuclear_segmentations(
    experiment: str = None,
    seg_store_path: Path = None,
    dest_store_path: Path = None,
    dest_subgroup_name: str = "nuclear_seg",
    upscaled_array_name: str = "20x_nuclear_seg",
    wells: list = None,
    overwrite: bool = False,
):
    """
    Upscale nuclear segmentations from 5x (4x downsampled) to full 20x resolution.

    Supports two modes:
    1. Experiment mode (backward compatible): Provide experiment name to use standard paths
       - Source: lc_20x_segmentation
       - Dest: pheno_assembled/{pos}/nuclear_seg
    2. Direct path mode: Provide seg_store_path directly (with optional dest_store_path for symlinks)
       - Source: seg_store_path
       - Dest: dest_store_path/{pos}/{dest_subgroup_name} (if provided)

    This function will:
        1) Upsample segmentations to full 20x res (stored as {upscaled_array_name} in source)
        2) Optionally symlink the upscaled segmentations into destination store

    Args:
        experiment: Experiment name (e.g., ops0042_20250520). If provided, uses standard paths.
        seg_store_path: Direct path to segmentation store. Required if experiment is None.
        dest_store_path: Path to destination store for symlinks (optional in direct mode).
        dest_subgroup_name: Name for the symlinked subgroup in destination (default: "nuclear_seg")
        upscaled_array_name: Name for the upscaled array in source store (default: "20x_nuclear_seg")
        wells: List of wells to process (default: all wells found)
        overwrite: If True, overwrite existing upscaled segmentations and symlinks
    """
    from scipy.ndimage import zoom

    # Determine paths based on mode
    if experiment is not None:
        # Experiment mode (backward compatible)
        dataset = OpsDataset(experiment)
        nuclear_seg_path = dataset.store_paths["lc_20x_segmentation"]
        dest_store_path = dataset.store_paths["pheno_assembled_v3"]
        chunk_size = dataset.store_props["chunk_size"]
        scale_20x = dataset.store_props["20x_scale"]
    elif seg_store_path is not None:
        # Direct path mode
        nuclear_seg_path = Path(seg_store_path)
        if dest_store_path is not None:
            dest_store_path = Path(dest_store_path)
        # Get default store properties from OpsDataset
        dataset = OpsDataset("")  # dummy dataset for store properties
        chunk_size = dataset.store_props["chunk_size"]
        scale_20x = dataset.store_props["20x_scale"]
    else:
        raise ValueError("Either experiment or seg_store_path must be provided")

    if not nuclear_seg_path.exists():
        raise FileNotFoundError(f"Segmentation store not found: {nuclear_seg_path}")

    print(f"\n{'='*60}")
    print(f"Upscaling Nuclear Segmentations (4x -> 20x)")
    print(f"{'='*60}")
    print(f"Source store: {nuclear_seg_path}")
    print(f"Dest store: {dest_store_path or 'None (no symlinks)'}")
    print(f"Upscaled array name: {upscaled_array_name}")
    print(f"Dest subgroup name: {dest_subgroup_name}")
    print(f"Wells: {wells or 'all'}")
    print(f"Overwrite: {overwrite}")
    print(f"{'='*60}\n")

    nuclear_seg_store = open_ome_zarr(nuclear_seg_path, mode="a")
    positions = _discover_positions(nuclear_seg_path)

    # Filter by wells if specified
    if wells is not None:
        positions = [p for p in positions if int(p.split("/")[1]) in wells]

    print(f"Found {len(positions)} positions to upscale")

    for pos in tqdm(positions, desc="Upscaling nuclear segmentations"):
        upscaled_path = nuclear_seg_path / pos / upscaled_array_name

        # Check if upscaled array already exists
        if upscaled_path.exists():
            if not overwrite:
                print(f"{upscaled_array_name} already exists for {pos}, skipping upscale...")
            else:
                print(f"{upscaled_array_name} exists for {pos}, overwriting...")
                shutil.rmtree(upscaled_path)

        if not upscaled_path.exists():
            import time as _time
            import zarr as _zarr

            t_up = _time.time()

            # Tiled upscaling: read source chunks, repeat 4x, write directly to zarr.
            # Uses ~268 MB peak memory vs ~44 GB for the old np.repeat approach.
            # The old approach materialized the entire 104K×104K array in memory,
            # which caused 40+ min timeouts under memory pressure.
            src_arr = _zarr.open(str(nuclear_seg_path / pos / "0"), mode="r")
            src_shape = src_arr.shape  # (1, 1, 1, ~26K, ~26K)
            src_chunks = src_arr.chunks

            up_shape = list(src_shape)
            up_shape[-2] *= 4
            up_shape[-1] *= 4
            up_shape = tuple(up_shape)

            # Create array directly at 20x_nuclear_seg/ (not nested at 0/)
            # to match iohub create_image structure. The symlink
            # nuclear_seg/0 → 20x_nuclear_seg then points directly to the array.
            dest_arr_path = nuclear_seg_path / pos / upscaled_array_name
            dest_arr_path.mkdir(parents=True, exist_ok=True)
            dest_arr = _zarr.open(
                str(dest_arr_path), mode="w",
                shape=up_shape,
                chunks=tuple(chunk_size),
                dtype=np.int32,
                zarr_format=2,
                dimension_separator="/",
            )

            h_src, w_src = src_shape[-2], src_shape[-1]
            ch = src_chunks[-2] if len(src_chunks) >= 2 else 2048
            cw = src_chunks[-1] if len(src_chunks) >= 1 else 2048
            n_y = (h_src + ch - 1) // ch
            n_x = (w_src + cw - 1) // cw

            for ty in range(n_y):
                for tx in range(n_x):
                    sy0, sy1 = ty * ch, min((ty + 1) * ch, h_src)
                    sx0, sx1 = tx * cw, min((tx + 1) * cw, w_src)
                    chunk = src_arr[0, 0, 0, sy0:sy1, sx0:sx1]
                    upchunk = np.repeat(np.repeat(chunk, 4, axis=0), 4, axis=1).astype(np.int32)
                    dest_arr[0, 0, 0, sy0*4:sy1*4, sx0*4:sx1*4] = upchunk

            # Write zarr metadata for the upscaled array
            import json
            zattrs = {
                "multiscales": [{
                    "datasets": [{"coordinateTransformations": [
                        {"type": "scale", "scale": list(scale_20x)}
                    ], "path": "0"}],
                    "version": "0.4",
                }]
            }
            zattrs_path = nuclear_seg_path / pos / upscaled_array_name / ".zattrs"
            zattrs_path.write_text(json.dumps(zattrs))

            elapsed = _time.time() - t_up
            print(f"  Upscaled {pos}: {src_shape} -> {up_shape} in {elapsed:.1f}s")

        # Symlink into destination store if provided
        if dest_store_path:
            src_path = nuclear_seg_path / pos / upscaled_array_name
            dest_path = dest_store_path / pos / dest_subgroup_name
            print(f"src_path: {src_path}")
            print(f"dest_path: {dest_path}")

            dest_path.mkdir(parents=True, exist_ok=True)

            # Ensure .zgroup metadata exists for zarr v2 compatibility
            zgroup_path = dest_path / ".zgroup"
            if not zgroup_path.exists():
                zgroup_path.write_text('{"zarr_format": 2}')

            symlink_target = dest_path / "0"

            # Check if symlink already exists
            if symlink_target.exists() or symlink_target.is_symlink():
                if overwrite:
                    symlink_target.unlink()
                else:
                    print(f"  Symlink {symlink_target} already exists, skipping...")
                    continue

            try:
                os.symlink(str(src_path.resolve()), str(symlink_target))
                print(f"[attach_seg_labels_symlink] linked {symlink_target} -> {src_path}")
            except Exception as e:
                print(
                    f"[attach_seg_labels_symlink] ERROR: failed to symlink {dest_path}: {e}"
                )

    nuclear_seg_store.close()

    # Validate upscaled data: check format and non-zero content
    print(f"\nValidating upscaled nuclear segmentations...")
    from cyclops_utils.io.zarr_utils import level_has_data
    for pos in positions:
        parent_path = nuclear_seg_path / pos / upscaled_array_name

        # Must be v2 format (.zarray at array level)
        if not (parent_path / ".zarray").exists():
            raise RuntimeError(
                f"Validation failed: {pos}/{upscaled_array_name}/.zarray missing. "
                f"Array was likely created as zarr v3 instead of v2."
            )
        if (parent_path / ".zgroup").exists():
            raise RuntimeError(
                f"Validation failed: {pos}/{upscaled_array_name}/.zgroup exists. "
                f"This breaks convert_v3. Remove it."
            )
        # Non-zero pixel check
        if not level_has_data(parent_path, check_pixels=True, level_index=0):
            raise RuntimeError(
                f"Validation failed: {pos}/{upscaled_array_name} has all-zero data. "
                f"Upscale writes did not persist."
            )

        print(f"  ✓ {pos}: format=v2, data=non-zero")

    print(f"\n{'='*60}")
    print(f"✅ Upscaling complete and validated!")
    print(f"{'='*60}\n")

    return


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run segmentation utilities directly (debug-friendly)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Helper function to add common arguments
    def add_common_args(subparser):
        subparser.add_argument(
            "--experiment", type=str, help="Experiment name (e.g., ops0042_20250520)"
        )
        subparser.add_argument(
            "--num-workers",
            type=int,
            default=None,
            help="Override auto-detected workers",
        )
        subparser.add_argument(
            "--debug-n-tiles", type=int, default=None, help="Debug: sample N tiles only"
        )
        subparser.add_argument(
            "--debug-output-suffix",
            type=str,
            default="_debug",
            help="Suffix for debug output store",
        )
        subparser.add_argument(
            "--use-preprocess",
            action="store_true",
            help="Enable preprocessing (max channels 0+1 and apply CLAHE) for pheno_cells",
        )
        subparser.add_argument(
            "--clahe-clip-limit",
            type=float,
            default=0.01,
            help="CLAHE clip limit for contrast enhancement (0.01=subtle, 0.05-0.10=aggressive)",
        )
        subparser.add_argument(
            "--clahe-kernel-size",
            type=int,
            default=None,
            help="CLAHE kernel size in pixels (smaller=local, larger=global contrast, default=256)",
        )

    # segment_and_stitch subcommand
    sas = subparsers.add_parser(
        "segment_and_stitch",
        help="Run segment_and_stitch for ISS/track/pheno/pheno_cells",
    )
    add_common_args(sas)
    sas.add_argument(
        "--process",
        type=str,
        choices=["iss", "track", "pheno", "pheno_cells"],
        help="Process type",
    )
    # Direct-path mode (no experiment)
    sas.add_argument(
        "--input-store-path",
        type=str,
        help="Direct path to input zarr (no experiment mode)",
    )
    sas.add_argument(
        "--input-config-path",
        type=str,
        help="Direct path to shifts config (no experiment mode)",
    )
    sas.add_argument(
        "--output-store-path",
        type=str,
        help="Direct path to output zarr (no experiment mode)",
    )

    # segmentation subcommand
    seg = subparsers.add_parser(
        "segmentation",
        help="Run standalone segmentation for pheno_cells or direct paths",
    )
    add_common_args(seg)
    seg.add_argument(
        "--process",
        type=str,
        default="pheno_cells",
        help="Process type (default: pheno_cells)",
    )
    seg.add_argument(
        "--positions",
        nargs="+",
        type=str,
        default=None,
        help="Specific position names to segment (e.g., A/1/028034 B/2/005012)",
    )
    # Direct-path mode (no experiment)
    seg.add_argument(
        "--input-store-path",
        type=str,
        help="Direct path to input zarr (no experiment mode)",
    )
    seg.add_argument(
        "--output-store-path",
        type=str,
        help="Direct path to output zarr (no experiment mode)",
    )

    # upscale_nuclear_segmentations subcommand
    upscale = subparsers.add_parser(
        "upscale_nuclear_segmentations",
        help="Upscale pheno nuclear segmentations from pseudo-5x to full 20x resolution",
    )
    upscale.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment name (e.g., ops0042_20250520)",
    )

    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.command == "segment_and_stitch":
        if args.experiment:
            segment_and_stitch(
                experiment=args.experiment,
                process=args.process,
                num_workers=args.num_workers,
                debug_n_tiles=args.debug_n_tiles,
                debug_output_suffix=args.debug_output_suffix,
                use_preprocess=args.use_preprocess,
                clahe_clip_limit=args.clahe_clip_limit,
                clahe_kernel_size=args.clahe_kernel_size,
            )
        else:
            if not (
                args.input_store_path
                and args.input_config_path
                and args.output_store_path
            ):
                parser.error(
                    "For direct-path mode, --input-store-path, --input-config-path and --output-store-path are required."
                )
            segment_and_stitch(
                experiment=None,
                process=args.process,
                input_store_path=Path(args.input_store_path),
                input_config_path=Path(args.input_config_path),
                output_store_path=Path(args.output_store_path),
                num_workers=args.num_workers,
                debug_n_tiles=args.debug_n_tiles,
                debug_output_suffix=args.debug_output_suffix,
                use_preprocess=args.use_preprocess,
                clahe_clip_limit=args.clahe_clip_limit,
                clahe_kernel_size=args.clahe_kernel_size,
            )
        return

    if args.command == "segmentation":
        if args.experiment:
            segmentation(
                experiment=args.experiment,
                process=args.process,
                num_workers=args.num_workers,
                debug_n_tiles=args.debug_n_tiles,
                debug_output_suffix=args.debug_output_suffix,
                use_preprocess=args.use_preprocess,
                clahe_clip_limit=args.clahe_clip_limit,
                clahe_kernel_size=args.clahe_kernel_size,
                positions=args.positions,
            )
        else:
            if not (args.input_store_path and args.output_store_path):
                parser.error(
                    "For direct-path mode, --input-store-path and --output-store-path are required."
                )
            segmentation(
                experiment=None,
                process=args.process,
                input_store_path=Path(args.input_store_path),
                output_store_path=Path(args.output_store_path),
                num_workers=args.num_workers,
                debug_n_tiles=args.debug_n_tiles,
                debug_output_suffix=args.debug_output_suffix,
                use_preprocess=args.use_preprocess,
                clahe_clip_limit=args.clahe_clip_limit,
                clahe_kernel_size=args.clahe_kernel_size,
                positions=args.positions,
            )
        return

    if args.command == "upscale_nuclear_segmentations":
        upscale_nuclear_segmentations(experiment=args.experiment)
        return


if __name__ == "__main__":
    main()
# to run ISS segment_and_stitch:
# python -m cyclops_process.processes.segment segment_and_stitch --experiment ops0058_20250805 --process iss --debug-n-tiles 10
# python -m cyclops_process.processes.segment segment_and_stitch --experiment ops0058_20250805 --process track --debug-n-tiles 10
# python -m cyclops_process.processes.segment segment_and_stitch --experiment ops0058_20250805 --process pheno --debug-n-tiles 10
# python -m cyclops_process.processes.segment segmentation --experiment ops0033_20250429 --process pheno_cells --debug-n-tiles 5
# python -m cyclops_process.processes.segment segmentation --experiment ops0046_20250611 --process pheno_cells --debug-n-tiles 5 --use-preprocess --clahe-clip-limit 0.01 --positions A/1/032031
