"""
Convert Zarr v2 stores to Zarr v3 format.
"""

from pathlib import Path
import zarr
import iohub
import dask.array as da
from tqdm import tqdm
import json
from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.data.filesystem import ensure_output_path
from cyclops_utils.profiling.decorators import versioned_function

# Import metadata utilities from convert_v3_metadata module
from cyclops_process.convert.v3_metadata import (
    build_channels_metadata,
    build_label_metadata,
    OVERLAY_METADATA,
)


# OPS phenotyping native YX pixel size (µm/px). The Dragonfly acquisition
# writes 0.325 µm per pixel; older v2 assembly pipelines wrote 0.65 into the
# OME-Zarr metadata (a 2× over-declaration that propagated into every
# downstream feature value). This is the authoritative correct value.
OPS_NATIVE_YX_UM_PER_PX = 0.325
# Historical buggy value to detect + auto-correct during v3 conversion.
OPS_BUGGY_YX_UM_PER_PX = 0.65


def _sanitize_coordinate_transformations(transform_meta, position_key: str, level_key: str):
    """Auto-correct a coordinate-transformation block that declares the
    historical buggy 0.65 µm/px spacing. Halves Y and X scale values so the
    written v3 metadata reflects the actual native 0.325 µm/px sampling.

    Returns the (possibly modified) transform_meta and a `corrected` flag.
    Operates on iohub-style `TransformationMeta` objects: each has
    `.type == "scale"` and `.scale` as a list of per-axis floats (T, C, Z, Y, X
    for 5D arrays).
    """
    if not transform_meta:
        return transform_meta, False
    corrected = False
    for t in transform_meta:
        if getattr(t, "type", None) != "scale":
            continue
        scale = getattr(t, "scale", None)
        if scale is None or len(scale) < 2:
            continue
        y, x = scale[-2], scale[-1]
        # Detect the exact buggy doubling. Only divide when the source clearly
        # has the bug — never touch already-correct stores or unexpected scales.
        if y == OPS_BUGGY_YX_UM_PER_PX and x == OPS_BUGGY_YX_UM_PER_PX:
            scale[-2] = y / 2.0
            scale[-1] = x / 2.0
            corrected = True
        elif y == 2 * OPS_BUGGY_YX_UM_PER_PX and x == 2 * OPS_BUGGY_YX_UM_PER_PX:
            # Coarser pyramid level (level 1 in buggy schema was 1.3 → fix to 0.65)
            scale[-2] = y / 2.0
            scale[-1] = x / 2.0
            corrected = True
        elif y == 4 * OPS_BUGGY_YX_UM_PER_PX and x == 4 * OPS_BUGGY_YX_UM_PER_PX:
            scale[-2] = y / 2.0
            scale[-1] = x / 2.0
            corrected = True
        elif y == 8 * OPS_BUGGY_YX_UM_PER_PX and x == 8 * OPS_BUGGY_YX_UM_PER_PX:
            scale[-2] = y / 2.0
            scale[-1] = x / 2.0
            corrected = True
        elif y == 16 * OPS_BUGGY_YX_UM_PER_PX and x == 16 * OPS_BUGGY_YX_UM_PER_PX:
            scale[-2] = y / 2.0
            scale[-1] = x / 2.0
            corrected = True
    if corrected:
        print(f"  [scale-fix] {position_key}/{level_key}: corrected Y/X scale "
              f"(was 0.65-based, now 0.325-based native µm/px)")
    return transform_meta, corrected


# Metadata for annotation subgroups (basic structure for quick lookups)
# For comprehensive metadata, use build_label_metadata() from convert_v3_metadata
SUBGROUP_METADATA = {
    "seg": {
        "annotation_type": "cell_segmentation",
        "description": "Cell segmentation masks",
        "is_ome_label": True
    },
    "nuclear_seg": {
        "annotation_type": "nuclear_segmentation",
        "description": "Nuclear segmentation masks",
        "is_ome_label": True
    },
    **OVERLAY_METADATA,  # grid_edges, grid_props, iss_points, etc.
}


def collect_per_level_clims(zarr_v2_path: Path, position_key: str, src_pos) -> dict:
    """Collect per-level clims metadata from v2 pyramid levels.

    Reads .zattrs from each pyramid level (0, 1, 2, 3, 4) and returns
    a dict mapping level name to clims metadata:
    {
        "0": {"contrast_limits": [...], "contrast_limits_per_channel": [...], ...},
        "1": {...},
        ...
    }
    """
    per_level_clims = {}
    for k, src_array in src_pos.images():
        # Read per-level .zattrs from v2 store
        level_dir = zarr_v2_path / position_key / k
        zattrs_path = level_dir / ".zattrs"
        if zattrs_path.exists():
            try:
                with open(zattrs_path, "r") as f:
                    level_attrs = json.load(f)
                # Store clims metadata for this level
                level_clims = {}
                if "contrast_limits" in level_attrs:
                    level_clims["contrast_limits"] = level_attrs["contrast_limits"]
                if "contrast_limits_per_channel" in level_attrs:
                    level_clims["contrast_limits_per_channel"] = level_attrs["contrast_limits_per_channel"]
                if "contrast_limits_method" in level_attrs:
                    level_clims["contrast_limits_method"] = level_attrs["contrast_limits_method"]
                if level_clims:
                    per_level_clims[k] = level_clims
            except Exception:
                pass
    return per_level_clims


def copy_array(src_array: "ts.TensorStore", dst_array: "ts.TensorStore"):
    """Copies content from src_array to dst_array using spatial chunking for efficiency."""
    # Handle arrays with fewer than 2 dimensions (e.g., 1D arrays like grid_props/id)
    if len(src_array.shape) < 2:
        dst_array.write(src_array.read().result()).result()
        return

    # Use spatial chunking for ALL arrays - copy in ~49K x 49K pixel tiles
    # This is fast and memory-efficient regardless of array dimensionality
    chunk_shape = dst_array.chunk_layout.read_chunk.shape

    # Determine spatial dimension indices based on array layout:
    # - 2D: (Y, X) -> y_dim=0, x_dim=1
    # - 3D overlays: (Y, X, C) -> y_dim=0, x_dim=1
    # - 4D: (T, C, Y, X) or similar -> y_dim=-2, x_dim=-1
    # - 5D: (T, C, Z, Y, X) -> y_dim=-2, x_dim=-1
    ndim = len(src_array.shape)
    if ndim <= 3:
        # For 2D and 3D (overlays), spatial dims are at start: (Y, X) or (Y, X, C)
        y_dim, x_dim = 0, 1
    else:
        # For 4D and 5D, spatial dims are at end: (..., Y, X)
        y_dim, x_dim = -2, -1

    # Calculate spatial step size: ~48 chunks per tile (~49K pixels if chunk=1024)
    spatial_multiplier = 48
    y_chunk = chunk_shape[y_dim] if len(chunk_shape) > abs(y_dim) else 1024
    x_chunk = chunk_shape[x_dim] if len(chunk_shape) > abs(x_dim) else 1024
    y_step = y_chunk * spatial_multiplier
    x_step = x_chunk * spatial_multiplier

    y_size = src_array.shape[y_dim]
    x_size = src_array.shape[x_dim]

    n_y = (y_size + y_step - 1) // y_step
    n_x = (x_size + x_step - 1) // x_step
    total_iterations = n_y * n_x

    print(f"    Copying with spatial chunking: {y_step}x{x_step} per tile ({total_iterations} tiles)")

    iteration = 0
    for y_start in range(0, y_size, y_step):
        y_stop = min(y_start + y_step, y_size)
        for x_start in range(0, x_size, x_step):
            x_stop = min(x_start + x_step, x_size)
            iteration += 1
            if iteration == 1 or iteration == total_iterations or iteration % max(1, total_iterations // 10) == 0:
                print(f"    Progress: {iteration}/{total_iterations} ({100*iteration/total_iterations:.1f}%)")

            # Build slice based on dimensionality - fast spatial chunking for all
            if ndim == 2:
                dst_array[y_start:y_stop, x_start:x_stop].write(
                    src_array[y_start:y_stop, x_start:x_stop]).result()
            elif ndim == 3:
                # 3D overlays: (Y, X, C)
                dst_array[y_start:y_stop, x_start:x_stop, :].write(
                    src_array[y_start:y_stop, x_start:x_stop, :]).result()
            elif ndim == 4:
                dst_array[:, :, y_start:y_stop, x_start:x_stop].write(
                    src_array[:, :, y_start:y_stop, x_start:x_stop]).result()
            elif ndim == 5:
                dst_array[:, :, :, y_start:y_stop, x_start:x_stop].write(
                    src_array[:, :, :, y_start:y_stop, x_start:x_stop]).result()
            elif ndim == 6:
                dst_array[:, :, :, :, y_start:y_stop, x_start:x_stop].write(
                    src_array[:, :, :, :, y_start:y_stop, x_start:x_stop]).result()
            else:
                raise ValueError(f"Unsupported array dimensionality: {ndim}D. Expected 2-6D arrays.")


def _set_custom_metadata(container, metadata: dict = None, use_zattrs: bool = True):
    """Set custom_metadata with provided metadata dict, merging with existing metadata."""
    if metadata:
        attrs = container.zattrs if use_zattrs else container.attrs
        # Get existing custom_metadata if it exists
        existing_metadata = attrs.get("custom_metadata", {})
        # Merge new metadata with existing (new metadata takes precedence for duplicate keys)
        merged_metadata = {**existing_metadata, **metadata}
        attrs["custom_metadata"] = merged_metadata


def copy_zarr_array_v2_to_v3(
    src_array,
    dest_container,
    array_key: str,
    zarr_v2_path: Path = None,
    zarr_v3_path: Path = None,
    src_pos_key: str = None,
    subgroup_name: str = None,
    dest_subgroup_name: str = None,
    chunks: tuple = None,
    shards_ratio: tuple = (1, 1, 1, 64, 64),
    transform_meta=None,
    custom_metadata: dict = None
):
    """
    Core routine to copy any zarr array from v2 to v3 format.
    Works with both iohub ImageArrays and raw zarr arrays/subgroups.

    Args:
        src_array: Source array (iohub ImageArray or zarr array)
        dest_container: Destination container (iohub Position or zarr subgroup)
        array_key: Array key/name (e.g., "0", "1", "seg", array name)
        zarr_v2_path: Root path to source zarr v2 store (required for subgroups)
        zarr_v3_path: Root path to destination zarr v3 store (required for subgroups)
        src_pos_key: Position key like "A/1/0" (required for subgroups)
        subgroup_name: Subgroup name in SOURCE v2 (e.g., "seg", "grid_edges")
        dest_subgroup_name: Subgroup path in DEST v3 (e.g., "labels/seg", "labels/grid_edges")
                           If None, uses subgroup_name
        chunks: Chunk dimensions for 5D image data. For subgroups, uses last 2 dims or source chunks
        shards_ratio: Sharding ratio for v3 format (only for iohub)
        transform_meta: Optional coordinate transformation metadata (only for iohub)
        custom_metadata: Optional metadata dict to attach
    """
    is_iohub = hasattr(dest_container, 'create_zeros')
    import numpy as np
    import tensorstore as ts

    # Determine chunks based on dimensionality
    if len(src_array.shape) == 5:
        dest_chunks = chunks if chunks else src_array.chunks
    elif len(src_array.shape) == 2:
        dest_chunks = (4096, 4096)  # Larger chunks for 2D arrays
    else:
        dest_chunks = src_array.chunks

    # PATH 1: 5D arrays (images + seg) - use create_zeros with sharding
    if len(src_array.shape) == 5:
        print(f"  copying 5D array {subgroup_name or 'image'}/{array_key} with sharding")

        if is_iohub:
            # For iohub Position containers: check if array already exists (resume mode)
            try:
                existing_array = dest_container[array_key]
                existing_chunks = tuple(existing_array.chunks) if hasattr(existing_array, 'chunks') else None
                expected_chunks = tuple(dest_chunks)

                if (existing_array.shape == src_array.shape
                        and existing_array.dtype == src_array.dtype
                        and existing_chunks == expected_chunks):
                    print(f"    -> Array {array_key} already exists with correct shape/dtype/chunks, skipping")
                    return
                else:
                    # Mismatch — delete ONLY this specific array, then recreate
                    print(f"    -> Array {array_key} exists but mismatches, removing and recreating:")
                    print(f"       shape: {existing_array.shape} vs expected {src_array.shape}")
                    print(f"       dtype: {existing_array.dtype} vs expected {src_array.dtype}")
                    print(f"       chunks: {existing_chunks} vs expected {expected_chunks}")
                    import shutil
                    # Resolve the on-disk path for this specific array only
                    store_path = Path(dest_container.zgroup.store.root)
                    array_disk_path = store_path / dest_container.zgroup.name.lstrip('/') / array_key
                    if array_disk_path.exists() and array_disk_path.is_dir():
                        shutil.rmtree(array_disk_path)
                        print(f"    -> Deleted {array_disk_path}")
                    else:
                        print(f"    -> WARNING: Could not find array on disk at {array_disk_path}, trying overwrite")
            except (KeyError, AttributeError):
                # Array doesn't exist, proceed with creation
                pass

            # For iohub Position containers: use create_zeros
            dest_array = dest_container.create_zeros(
                array_key, shape=src_array.shape, dtype=src_array.dtype,
                chunks=dest_chunks, shards_ratio=shards_ratio, transform=transform_meta
            )
            _set_custom_metadata(dest_container, metadata=custom_metadata, use_zattrs=True)
            # Older iohub exposed .tensorstore() on ImageArray; iohub 0.3.x
            # removed it. Use it if present, otherwise open via tensorstore
            # directly against the underlying on-disk path. Detect source
            # driver from the marker file: v3 source has `zarr.json` at the
            # array dir, v2 has `.zarray`.
            if hasattr(src_array, "tensorstore") and callable(src_array.tensorstore):
                _src_ts = src_array.tensorstore()
            else:
                _src_root = Path(src_array.native.store.root)
                _src_rel = src_array.native.path
                _src_arr_path = _src_root / _src_rel
                _src_driver = "zarr3" if (_src_arr_path / "zarr.json").exists() else "zarr"
                _src_ts = ts.open({
                    "driver": _src_driver,
                    "kvstore": {"driver": "file", "path": str(_src_arr_path)},
                }).result()
            if hasattr(dest_array, "tensorstore") and callable(dest_array.tensorstore):
                _dest_ts = dest_array.tensorstore()
            else:
                _dest_root = Path(dest_container.zgroup.store.root)
                _dest_rel = dest_container.zgroup.name.lstrip("/")
                _dest_arr_path = _dest_root / _dest_rel / array_key
                _dest_ts = ts.open({
                    "driver": "zarr3",
                    "kvstore": {"driver": "file", "path": str(_dest_arr_path)},
                }).result()
            copy_array(_src_ts, _dest_ts)
        else:
            # For raw zarr groups (subgroups): use create_array with sharding
            shards = tuple(c * r for c, r in zip(dest_chunks, shards_ratio))
            dest_container.create_array(
                name=array_key,
                shape=src_array.shape,
                dtype=src_array.dtype,
                chunks=dest_chunks,
                shards=shards,
                overwrite=True,
                fill_value=0,
            )
            _set_custom_metadata(dest_container, metadata=custom_metadata, use_zattrs=False)

            # Open both with tensorstore and copy
            src_ts = ts.open({
                'driver': 'zarr',
                'kvstore': {'driver': 'file', 'path': str(zarr_v2_path / src_pos_key / subgroup_name / array_key)},
            }).result()
            dest_path = dest_subgroup_name if dest_subgroup_name else subgroup_name
            dest_ts = ts.open({
                'driver': 'zarr3',
                'kvstore': {'driver': 'file', 'path': str(zarr_v3_path / src_pos_key / dest_path / array_key)},
            }).result()
            copy_array(src_ts, dest_ts)

    # PATH 2: String arrays - use zarr directly (tensorstore doesn't support strings)
    elif np.issubdtype(src_array.dtype, np.str_) or np.issubdtype(src_array.dtype, np.bytes_):
        print(f"  copying string array {subgroup_name}/{array_key}")
        dest_array = dest_container.create_dataset(
            array_key, shape=src_array.shape, dtype=src_array.dtype,
            chunks=dest_chunks, fill_value="", overwrite=True
        )
        _set_custom_metadata(dest_container, metadata=custom_metadata, use_zattrs=False)
        dest_array[:] = src_array[:]

    # PATH 3: Other numeric arrays (overlays, grid_edges, etc) - use sharding like 5D arrays
    else:
        print(f"  copying {len(src_array.shape)}D array {subgroup_name}/{array_key} with sharding")

        # Calculate shards_ratio for this array dimensionality
        # For 3D overlays (Y, X, C): use (32, 32, 1) - same spatial grouping as 5D
        # For 2D: use (32, 32)
        # For 1D: no sharding
        ndim = len(src_array.shape)
        if ndim == 3:
            overlay_shards_ratio = (32, 32, 1)
        elif ndim == 2:
            overlay_shards_ratio = (32, 32)
        else:
            overlay_shards_ratio = tuple([1] * ndim)

        shards = tuple(c * r for c, r in zip(dest_chunks, overlay_shards_ratio))
        dest_container.create_array(
            name=array_key,
            shape=src_array.shape,
            dtype=src_array.dtype,
            chunks=dest_chunks,
            shards=shards,
            overwrite=True,
            fill_value=0,
        )
        _set_custom_metadata(dest_container, metadata=custom_metadata, use_zattrs=False)

        dest_path = dest_subgroup_name if dest_subgroup_name else subgroup_name
        src_ts = ts.open({
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': str(zarr_v2_path / src_pos_key / subgroup_name / array_key)},
        }).result()
        dest_ts = ts.open({
            'driver': 'zarr3',
            'kvstore': {'driver': 'file', 'path': str(zarr_v3_path / src_pos_key / dest_path / array_key)},
        }).result()
        copy_array(src_ts, dest_ts)


def validate_arrays(src_plate, dest_plate, max_positions=1, validation_chunk_size=(1, 1, 1, 4096, 4096), store_name=None):
    """
    Iterates through positions on two plates and compares that the pixel values are exactly the same.

    Args:
        src_plate: Source plate to validate from
        dest_plate: Destination plate to validate against
        max_positions: Maximum number of positions to validate. If None, validates all positions. (default: 3)
        validation_chunk_size: Chunk size to use for dask array comparison (default: (1, 1, 1, 4096, 4096))
        store_name: Optional name of the store being validated (e.g., "pheno_assembled", "track_assembled")
    """
    store_label = f" [{store_name}]" if store_name else ""
    positions_checked = 0
    for k_pos, src_pos in src_plate.positions():
        if max_positions is not None and positions_checked >= max_positions:
            print(f"Validation{store_label} complete: checked {positions_checked} positions")
            break

        dest_pos = dest_plate[k_pos]
        for k_array, src_image in src_pos.images():
            print(f"Validating{store_label} {k_pos} {k_array}")
            assert (as_dask_array(src_image, chunks=validation_chunk_size) == as_dask_array(dest_pos[k_array], chunks=validation_chunk_size)).all().compute()

        positions_checked += 1

    if max_positions is None:
        print(f"Validation{store_label} complete: checked all {positions_checked} positions")


def as_dask_array(img: iohub.ngff.nodes.ImageArray, **kwargs):
    return da.from_zarr(img.store.root, component=img.path, **kwargs)


def calculate_channel_based_shards(
    num_channels: int,
    chunks: tuple = (1, 1, 1, 512, 512),
    target_chunks_per_shard: int = 4096
):
    """
    Calculate optimal sharding ratio that prioritizes grouping channels together
    while targeting ~1GB shards.

    Strategy: Group all available channels together, then reduce spatial sharding
    to keep total chunks per shard approximately constant.

    Args:
        num_channels: Number of channels in the dataset (typically 1-5)
        chunks: Base chunk size (T, C, Z, Y, X). Default: (1, 1, 1, 512, 512)
        target_chunks_per_shard: Target total chunks per shard. Default: 4096 (~1GB)
                                Each chunk is 0.25 MB, so 4096 chunks = 1024 MB

    Returns:
        tuple: Sharding ratio as (T_shard, C_shard, Z_shard, Y_shard, X_shard)

    Example:
        With 1 channel, target_chunks_per_shard=4096, chunks=(1,1,1,512,512):
        - Returns (1, 1, 1, 64, 64) -> 1×64×64 = 4096 chunks
        - Shard size: 32,768×32,768 pixels, ~1024 MB

        With 4 channels, target_chunks_per_shard=4096:
        - Returns (1, 4, 1, 32, 32) -> 4×32×32 = 4096 chunks
        - Shard size: 8,192×8,192 pixels per channel, ~1024 MB total

        With 5 channels, target_chunks_per_shard=4096:
        - Returns (1, 5, 1, 28, 28) -> 5×28×28 = 3920 chunks
        - Shard size: 7,168×7,168 pixels per channel, ~980 MB total
    """
    # Always group all available channels together
    c_shard = num_channels

    # Calculate spatial sharding needed to reach target total chunks
    # target_chunks_per_shard = c_shard × y_shard × x_shard
    # Assuming square spatial shards: y_shard = x_shard = sqrt(target / c_shard)
    spatial_chunks_needed = target_chunks_per_shard / c_shard
    spatial_shard_ratio = int(spatial_chunks_needed ** 0.5)

    # Ensure at least 1 chunk per dimension
    spatial_shard_ratio = max(spatial_shard_ratio, 1)

    shards_ratio = (1, c_shard, 1, spatial_shard_ratio, spatial_shard_ratio)

    # Calculate actual values for reporting
    chunk_size = chunks[-1]  # Y or X dimension (assumed square)
    actual_chunks_per_shard = c_shard * spatial_shard_ratio * spatial_shard_ratio
    actual_shard_pixels = chunk_size * spatial_shard_ratio

    print(f"Channel-based sharding: {num_channels} channels")
    print(f"  Shards ratio: {shards_ratio}")
    print(f"  Chunks per shard: {actual_chunks_per_shard} (target: {target_chunks_per_shard})")
    print(f"  Shard shape: (1, {c_shard}, 1, {actual_shard_pixels}, {actual_shard_pixels})")

    return shards_ratio


def initialize_v3_store(
    zarr_v2_path: Path,
    zarr_v3_path: Path,
    overwrite: bool = False,
    skip_overlays: bool = False,
    exclude_groups: set = None,
    experiment: str = None,
    skip_labels: bool = False,
):
    """
    Initialize the v3 zarr store structure with all positions and subgroups.
    This should be called once before parallel conversion jobs.

    Args:
        zarr_v2_path (Path): Root path to source zarr v2 store
        zarr_v3_path (Path): Root path to destination zarr v3 store
        overwrite (bool): If True, overwrite existing store
        skip_overlays (bool): If True, skip creating overlay groups in labels
        exclude_groups (set): Set of group names to exclude (e.g., {'seg'} for track/iss stores)
        experiment (str): Experiment name for loading channel metadata from ops_channel_maps.yaml
        skip_labels (bool): If True, skip creating labels groups entirely (for base-only conversion)

    Returns:
        None
    """
    print(f"Initializing v3 store structure at {zarr_v3_path}")

    # Check if destination exists
    if zarr_v3_path.exists():
        if not overwrite:
            print("  v3 store already exists, skipping initialization")
            return
        else:
            import shutil
            shutil.rmtree(zarr_v3_path)
            print("  removed existing v3 store")

    # Open source to get structure
    with iohub.open_ome_zarr(zarr_v2_path, mode="r") as source_plate:
        channel_names = list(source_plate.channel_names)

        # Create v3 plate
        with iohub.open_ome_zarr(
            zarr_v3_path, mode="w", layout="hcs",
            channel_names=channel_names, version="0.5"
        ) as dest_plate:
            # Create all positions
            for pos_key, _ in source_plate.positions():
                dest_plate.create_position(*pos_key.split("/"))
                print(f"  created position {pos_key}")

    # Build and write channel metadata to plate-level zattrs
    channels_metadata = build_channels_metadata(channel_names, experiment=experiment)
    if channels_metadata:
        dest_zarr_store_temp = zarr.open(str(zarr_v3_path), mode="r+")
        existing_attrs = dict(dest_zarr_store_temp.attrs)
        existing_attrs["channels_metadata"] = channels_metadata
        dest_zarr_store_temp.attrs.update(existing_attrs)
        print(f"  added channel metadata for {len(channels_metadata)} channels")

    # Skip labels creation entirely if skip_labels=True (for base-only conversion)
    if skip_labels:
        print("  skipping labels groups (base-only mode)")
        print("v3 store initialization complete\n")
        return

    # Create labels group and all subgroups under it
    # These are skipped when skip_overlays=True
    OVERLAY_GROUPS = {"grid_edges", "grid_props", "iss_points", "iss_points_props", "iss_gene_image", "iss_guide_image"}

    if exclude_groups is None:
        exclude_groups = set()

    src_zarr_store = zarr.open(str(zarr_v2_path), mode="r")
    dest_zarr_store = zarr.open(str(zarr_v3_path), mode="r+")

    with iohub.open_ome_zarr(zarr_v2_path, mode="r") as source_plate:
        for pos_key, _ in source_plate.positions():
            src_pos_group = src_zarr_store[pos_key]
            dest_pos_group = dest_zarr_store[pos_key]

            # Create labels group at position level
            labels_group = dest_pos_group.create_group("labels", overwrite=True)
            print(f"  created labels group {pos_key}/labels")

            # Get all subgroup names from source
            all_subgroups = list(src_pos_group.group_keys())

            # Filter out overlays if skip_overlays is True
            groups_to_skip = set()
            if skip_overlays:
                groups_to_skip.update(OVERLAY_GROUPS & set(all_subgroups))

            # Filter out excluded groups
            groups_to_skip.update(exclude_groups & set(all_subgroups))

            subgroups_to_create = [name for name in all_subgroups if name not in groups_to_skip]

            if groups_to_skip:
                print(f"  skipping groups: {groups_to_skip}")

            # Filter to only OME-compliant labels (segmentations) from the filtered list
            ome_label_names = [name for name in subgroups_to_create
                              if SUBGROUP_METADATA.get(name, {}).get("is_ome_label", False)]

            # Set labels metadata with both OME labels and all labels (filtered)
            if subgroups_to_create:
                labels_attrs = {}
                if ome_label_names:
                    labels_attrs["ome"] = {
                        "version": "0.5",
                        "labels": ome_label_names
                    }
                # Also store all label names for reference (filtered)
                labels_attrs["labels"] = subgroups_to_create
                labels_group.attrs.update(labels_attrs)
                print(f"  set OME labels metadata: {ome_label_names}")
                print(f"  set all labels: {subgroups_to_create}")

            # Create each subgroup under labels/ (filtered)
            for group_name in subgroups_to_create:
                labels_group.create_group(group_name, overwrite=True)
                print(f"  created subgroup {pos_key}/labels/{group_name}")

            # Write comprehensive metadata to each segmentation label subgroup
            for group_name in subgroups_to_create:
                label_metadata = build_label_metadata(
                    group_name, channel_names, experiment
                )
                if group_name in labels_group:
                    labels_group[group_name].attrs["segmentation_metadata"] = label_metadata
            print(f"  wrote segmentation_metadata for {len(subgroups_to_create)} labels")

    print("v3 store initialization complete\n")


def write_seg_label_v3(
    seg_source_store,
    dest_v3_store,
    label_name: str = "nuclear_seg",
    source_label: str = "0",
    experiment: str = None,
    channel_names=None,
    quiet: bool = False,
) -> int:
    """Write a segmentation label directly into a v3 store's ``labels/<label_name>/0``
    as a properly-sharded array, reading the source (v2 OR v3) via TensorStore.

    Replaces the fragile ``_attach_seg_labels_symlink`` staging for stores whose
    segmentation source is v3 (zarr.json) — the symlink path only handled v2
    (.zarray). Level 0 only; build_pyramids fills levels 1..N.

    Returns the number of positions written.
    """
    import tensorstore as ts
    import zarr
    from pathlib import Path
    from iohub.ngff import open_ome_zarr

    src_root = Path(seg_source_store)
    dst_root = Path(dest_v3_store)
    if not src_root.exists():
        if not quiet:
            print(f"  [seg-label] skip {label_name}: source seg store missing {src_root}")
        return 0
    if not dst_root.exists():
        if not quiet:
            print(f"  [seg-label] skip {label_name}: dest v3 store missing {dst_root}")
        return 0

    # Positions the v3 store already has (base image + labels group precreated).
    with open_ome_zarr(dst_root, mode="r") as dst:
        positions = [p for p, _ in dst.positions()]
        if channel_names is None:
            try:
                channel_names = list(dst.channel_names)
            except Exception:
                channel_names = []

    written = 0
    for pos in positions:
        src_arr_dir = src_root / pos / source_label
        if not src_arr_dir.exists():
            if not quiet:
                print(f"  [seg-label] {pos}: source array missing {src_arr_dir}; skip")
            continue

        # Auto-detect source driver: v3 has zarr.json, v2 has .zarray.
        src_driver = "zarr3" if (src_arr_dir / "zarr.json").exists() else "zarr"
        src_ts = ts.open(
            {"driver": src_driver, "kvstore": {"driver": "file", "path": str(src_arr_dir)}}
        ).result()
        shape = tuple(int(s) for s in src_ts.shape)
        dtype = src_ts.dtype.numpy_dtype

        # Chunk 512² spatial; shard = 8×8 chunks (4096²) so writes are full-shard
        # and file count stays low. Leading (T,C,Z) dims stay size-1.
        ndim = len(shape)
        chunks = tuple([1] * max(0, ndim - 2) + [512, 512])[:ndim]
        shard_ratio = tuple([1] * max(0, ndim - 2) + [8, 8])[:ndim]
        shards = tuple(min(c * r, s) if i >= ndim - 2 else c
                       for i, (c, r, s) in enumerate(zip(chunks, shard_ratio, shape)))

        label_grp_dir = dst_root / pos / "labels" / label_name
        label_grp = zarr.open_group(str(label_grp_dir), mode="a")
        if "0" in label_grp:
            del label_grp["0"]
        label_grp.create_array(
            name="0", shape=shape, dtype=dtype, chunks=chunks, shards=shards,
            overwrite=True, fill_value=0,
        )
        dest_ts = ts.open(
            {"driver": "zarr3", "kvstore": {"driver": "file", "path": str(label_grp_dir / "0")}}
        ).result()
        copy_array(src_ts, dest_ts)

        # Attach segmentation metadata on the label group (best-effort — matches
        # what convert writes for real seg labels; not required for audit/napari,
        # which discover labels by directory like grid_overlay).
        try:
            label_grp.attrs["segmentation_metadata"] = build_label_metadata(
                label_name, channel_names, experiment
            )
        except Exception as e:
            if not quiet:
                print(f"  [seg-label] {pos}: seg-metadata warn: {e}")
        # Register the label in the parent labels-group OME list (best-effort).
        try:
            labels_parent = zarr.open_group(str(dst_root / pos / "labels"), mode="a")
            ome = dict(labels_parent.attrs.get("ome") or {})
            ome["version"] = ome.get("version", "0.5")
            ome["labels"] = list(dict.fromkeys(list(ome.get("labels") or []) + [label_name]))
            labels_parent.attrs["ome"] = ome
            labels_parent.attrs["labels"] = list(
                dict.fromkeys(list(labels_parent.attrs.get("labels") or []) + [label_name])
            )
        except Exception as e:
            if not quiet:
                print(f"  [seg-label] {pos}: ome-metadata warn: {e}")

        written += 1
        if not quiet:
            print(f"  [seg-label] wrote {pos}/labels/{label_name}/0 shape={shape} shards={shards}")

    if not quiet:
        print(f"  [seg-label] {label_name}: wrote {written}/{len(positions)} positions (sharded, v3-native)")
    return written


def copy_position_group_zarrv2_to_zarrv3(
    zarr_v2_path: Path,
    zarr_v3_path: Path,
    position_key: str,
    group_name: str = None,
    chunks: tuple = None,
    shards_ratio: tuple = (1, 1, 1, 64, 64),
    experiment: str = None,
):
    """
    Converts a single position+group from Zarr v2 to v3 format.
    Assumes v3 store structure already exists (call initialize_v3_store first).

    Args:
        zarr_v2_path (Path): Root path to source zarr v2 store
        zarr_v3_path (Path): Root path to destination zarr v3 store
        position_key (str): Position key like "A/1/0"
        group_name (str): Name of subgroup to copy (e.g., "seg", "nuclear_seg").
                         If None, copies base images only.
        chunks (tuple): Chunk dimensions. If None, uses source chunks.
        shards_ratio (tuple): Sharding ratio for v3 format.
        experiment (str): Experiment name for loading channel metadata from ops_channel_maps.yaml

    Returns:
        None
    """
    if group_name:
        print(f"Converting position {position_key}, group {group_name}")
    else:
        print(f"Converting position {position_key}, base images")

    # Open source zarr v2 for accessing subgroups
    src_zarr_store = zarr.open(str(zarr_v2_path), mode="r")

    # Open existing v3 store (should already be initialized)
    with iohub.open_ome_zarr(zarr_v2_path, mode="r") as source_plate:
        with iohub.open_ome_zarr(zarr_v3_path, mode="r+", channel_names=source_plate.channel_names) as dest_plate:
            src_pos = source_plate[position_key]
            dest_pos = dest_plate[position_key]

            if group_name is None:
                # Copy main images (0, 1, 2, 3, 4)
                # Collect per-level clims from all pyramid levels ONCE
                per_level_clims = collect_per_level_clims(zarr_v2_path, position_key, src_pos)

                for k, src_array in src_pos.images():
                    # Find coordinate transformations for this pyramid level
                    matching_datasets = [x for x in src_pos.metadata.multiscales[0].datasets if x.path == k]
                    if matching_datasets:
                        transform_meta = matching_datasets[0].coordinate_transformations
                    else:
                        # Pyramid level exists but not in metadata - skip it
                        print(f"  WARNING: pyramid level {k} not found in multiscales metadata, skipping")
                        continue

                    # Sanitize: source v2 stores may declare the buggy 0.65 µm/px
                    # at level 0 (true native is 0.325). Halve Y/X here so the
                    # v3 store inherits the correct value.
                    transform_meta, _ = _sanitize_coordinate_transformations(
                        transform_meta, position_key, k,
                    )

                    # Collect all custom metadata from the array-level attrs.
                    # iohub 0.3.x ImageArray does NOT expose .attrs (only .zattrs,
                    # which raises if there's no group .zattrs file). The raw zarr
                    # array is reachable via .native (NGFFArray._handle); fall back
                    # to that. If the array has no attrs at all (clean v3 store),
                    # use an empty dict — downstream pop() calls handle that fine.
                    if hasattr(src_array, "attrs"):
                        _raw_attrs = src_array.attrs
                    elif hasattr(src_array, "native") and hasattr(src_array.native, "attrs"):
                        _raw_attrs = src_array.native.attrs
                    else:
                        _raw_attrs = {}
                    try:
                        array_zattrs = dict(_raw_attrs)
                    except Exception:
                        array_zattrs = {}

                    # Remove per-level clims from array attrs (they'll be in clims_per_level dict instead)
                    array_zattrs.pop("contrast_limits", None)
                    array_zattrs.pop("contrast_limits_per_channel", None)
                    array_zattrs.pop("contrast_limits_method", None)

                    # Merge with position-level normalization if it exists
                    pos_normalization = src_pos.zattrs.get("normalization")
                    if pos_normalization:
                        array_zattrs["normalization"] = pos_normalization

                    # Add per-level clims dict (only on first iteration)
                    if k == "0" and per_level_clims:
                        array_zattrs["clims_per_level"] = per_level_clims

                    copy_zarr_array_v2_to_v3(
                        src_array=src_array, dest_container=dest_pos, array_key=k,
                        chunks=chunks, shards_ratio=shards_ratio, transform_meta=transform_meta,
                        custom_metadata=array_zattrs if array_zattrs else None
                    )
            else:
                # Copy specific subgroup (should already exist under labels/)
                src_pos_group = src_zarr_store[position_key]

                # Use filesystem detection since some groups lack .zgroup files
                src_pos_path = Path(zarr_v2_path) / position_key
                group_path = src_pos_path / group_name
                group_exists = group_path.is_dir()

                if group_exists:
                    # Use filesystem to detect arrays and subgroups (no .zgroup/.zarray required)
                    # Arrays have .zarray file, subgroups are directories without .zarray
                    array_keys = [p.name for p in group_path.iterdir()
                                  if p.is_dir() and (p / ".zarray").exists()]
                    subgroup_keys = [p.name for p in group_path.iterdir()
                                     if p.is_dir() and not (p / ".zarray").exists() and p.name.isdigit()]

                    if not array_keys and not subgroup_keys:
                        print(f"  Skipping empty group {group_name}")
                        return

                    # Don't open the group - it may lack .zgroup. Open arrays directly instead.

                    # Use raw zarr group (NOT Position wrapper) for subgroups
                    dest_zarr_store = zarr.open(str(zarr_v3_path), mode="r+")
                    dest_pos_group = dest_zarr_store[position_key]

                    # Ensure labels group exists
                    if "labels" not in dest_pos_group.group_keys():
                        dest_pos_group.create_group("labels")
                    dest_labels = dest_pos_group["labels"]

                    # Create the specific label group if it doesn't exist
                    if group_name not in dest_labels.group_keys():
                        dest_labels.create_group(group_name)
                        print(f"  created missing group labels/{group_name}")
                    dest_label_group = dest_labels[group_name]

                    # Build and write comprehensive segmentation metadata for this label
                    # This ensures metadata is present even when resuming or re-converting labels
                    channel_names = list(source_plate.channel_names)
                    label_metadata = build_label_metadata(group_name, channel_names, experiment)
                    dest_label_group.attrs["segmentation_metadata"] = label_metadata
                    print(f"  wrote segmentation_metadata for {group_name}")
                    
                    # Handle nested pyramid structure (e.g., iss_gene_image/0, /1, /2)
                    # Note: Some groups may have a mix of subgroups (v3 format levels) and
                    # direct arrays (v2 format levels) due to incremental builds or format changes
                    if subgroup_keys:
                        print(f"  copying nested pyramid structure with {len(subgroup_keys)} subgroup levels")
                        for subgroup_key in subgroup_keys:
                            # Open subgroup directly by path (parent may lack .zgroup)
                            subgroup_path = group_path / subgroup_key
                            src_subgroup = zarr.open_group(str(subgroup_path), mode="r")
                            nested_array_keys = list(src_subgroup.array_keys())

                            if not nested_array_keys:
                                continue

                            # Create subgroup in destination if it doesn't exist
                            if subgroup_key not in dest_label_group.group_keys():
                                dest_label_group.create_group(subgroup_key)

                            dest_subgroup = dest_label_group[subgroup_key]

                            for array_name in nested_array_keys:
                                copy_zarr_array_v2_to_v3(
                                    src_array=src_subgroup[array_name],
                                    dest_container=dest_subgroup,  # Pass nested subgroup
                                    array_key=array_name,
                                    zarr_v2_path=zarr_v2_path, zarr_v3_path=zarr_v3_path,
                                    src_pos_key=position_key,
                                    subgroup_name=f"{group_name}/{subgroup_key}",  # Source v2 path
                                    dest_subgroup_name=f"labels/{group_name}/{subgroup_key}",  # Dest v3 path
                                    chunks=chunks,
                                    shards_ratio=shards_ratio,
                                    custom_metadata=SUBGROUP_METADATA.get(group_name)
                                )

                    # Handle flat structure (arrays directly in group)
                    # This handles both pure flat structures AND mixed formats (some levels as arrays)
                    if array_keys:
                        print(f"  copying {len(array_keys)} direct array levels")
                        for array_name in array_keys:
                            # Open array directly by path (parent may lack .zgroup)
                            src_array = zarr.open_array(str(group_path / array_name), mode="r")
                            copy_zarr_array_v2_to_v3(
                                src_array=src_array,
                                dest_container=dest_label_group,  # Pass raw zarr group
                                array_key=array_name,
                                zarr_v2_path=zarr_v2_path, zarr_v3_path=zarr_v3_path,
                                src_pos_key=position_key,
                                subgroup_name=group_name,  # Source v2 path (no labels/)
                                dest_subgroup_name=f"labels/{group_name}",  # Dest v3 path (with labels/)
                                chunks=chunks,
                                shards_ratio=shards_ratio,
                                custom_metadata=SUBGROUP_METADATA.get(group_name)
                            )
                else:
                    print(f"  Warning: group {group_name} not found in position {position_key}")

    print(f"  Completed {position_key}/{group_name or 'base'}")


def copy_zarrv2_to_zarrv3(zarr_v2_path: Path, zarr_v3_path: Path, chunks: tuple = None, shards_ratio: tuple = (1, 1, 1, 64, 64), overwrite: bool = False, experiment: str = None):
    """
    Creates a new Zarr v3 and copies the contents of a given Zarr v2 into v3.

    Args:
        zarr_v2_path (Path): A Path object to the root of a zarr v2 group. The source data to be converted/copied
        zarr_v3_path (Path): A Path object to the root of a zarr v3 group. The destination for the data to be copied to
        chunks (Tuple[Int]): A tuple of integers representing chunk dimensions. If None, uses source chunks.
        shards_ratio (Tuple[Int]): A tuple of integers for sharding ratio in v3 format.
        overwrite (bool): If True, overwrite existing v3 store during initialization
        experiment (str): Experiment name for loading channel metadata from ops_channel_maps.yaml

    Returns:
        None
    """
    print("Starting copy")

    # Initialize v3 store structure first
    initialize_v3_store(zarr_v2_path, zarr_v3_path, overwrite=overwrite, experiment=experiment)

    src_zarr_store = zarr.open(str(zarr_v2_path), mode="r")

    with iohub.open_ome_zarr(zarr_v2_path, mode="r") as source_plate:
        position_keys = [pos_key for pos_key, _ in source_plate.positions()]

    # Process each position + group combination
    for position_key in tqdm(position_keys, desc="Converting positions", unit="pos"):
        # First copy base images
        copy_position_group_zarrv2_to_zarrv3(zarr_v2_path, zarr_v3_path, position_key, None, chunks, shards_ratio, experiment=experiment)

        # Then copy each subgroup
        src_pos_group = src_zarr_store[position_key]
        for group_name in src_pos_group.group_keys():
            copy_position_group_zarrv2_to_zarrv3(zarr_v2_path, zarr_v3_path, position_key, group_name, chunks, shards_ratio, experiment=experiment)

    print("Copy completed")


@versioned_function("v1.0")
def convert_position_group_to_v3(
    experiment: str = None,
    position_key: str = None,
    group_name: str = None,
    source_store: str = "pheno_assembled",
    source_path: str = None,
    dest_path: str = None,
    chunks: tuple = (1, 1, 1, 512, 512),
    shards_ratio: tuple = (1, 1, 1, 64, 64),
):
    """
    Convert a single position+group from Zarr v2 to v3 format.

    Args:
        experiment: Experiment name (optional if source_path and dest_path provided)
        position_key: Position key like "A/1/0"
        group_name: Subgroup name (e.g., "seg", "nuclear_seg"). If None, converts base images.
        source_store: Key for source store in OpsDataset.store_paths (default: 'pheno_assembled')
        source_path: Direct path to source store (optional, overrides experiment lookup)
        dest_path: Direct path to destination store (optional, overrides experiment lookup)
        chunks: Chunk dimensions for v3 store (default: (1, 1, 1, 512, 512))
        shards_ratio: Sharding ratio for v3 format (default: (1, 1, 1, 64, 64) ~1GB shards)
    """
    # Handle direct paths or experiment-based lookup
    if source_path and dest_path:
        source_path = Path(source_path)
        dest_path = Path(dest_path)
    elif experiment:
        dataset = OpsDataset(experiment)
        source_path = dataset.store_paths.get(source_store)
        if not source_path:
            raise ValueError(f"Unknown source store key: {source_store}")

        # Infer dest path by appending _v3 to source_store key
        dest_store_key = f"{source_store}_v3"
        dest_path = dataset.store_paths.get(dest_store_key)
        if not dest_path:
            raise ValueError(f"Unknown destination store key: {dest_store_key}")
    else:
        raise ValueError("Either (source_path + dest_path) or experiment must be provided")

    group_label = group_name or "base"
    print(f"Converting {experiment or 'store'} position {position_key}/{group_label}\n"
          f"  Source: {source_path}\n"
          f"  Dest: {dest_path}")

    copy_position_group_zarrv2_to_zarrv3(
        source_path, dest_path, position_key, group_name, chunks, shards_ratio,
        experiment=experiment
    )


def validate_v3_conversion(
    experiment: str,
    source_store: str = "pheno_assembled",
    max_positions: int = None,
    validation_chunk_size: tuple = (1, 1, 1, 4096, 4096)
):
    """
    Validate that a Zarr v3 conversion matches the source v2 store.

    Args:
        experiment: Experiment name
        source_store: Key for source store in OpsDataset.store_paths (default: 'pheno_assembled')
        max_positions: Maximum positions to validate. If None, validates all (default: None)
        validation_chunk_size: Chunk size for dask array comparison (default: (1, 1, 1, 4096, 4096))
    """
    dataset = OpsDataset(experiment)
    source_path = dataset.store_paths.get(source_store)
    if not source_path:
        raise ValueError(f"Unknown source store key: {source_store}")

    dest_path = dataset.store_paths[f"{source_store}_v3"]

    if not source_path.exists():
        raise FileNotFoundError(f"Source store not found: {source_path}")
    if not dest_path.exists():
        raise FileNotFoundError(f"Destination v3 store not found: {dest_path}")

    print(f"Validating v3 conversion for {experiment}")
    print(f"  Source: {source_path}")
    print(f"  Dest: {dest_path}")
    print(f"  Max positions: {max_positions or 'all'}")
    print(f"  Chunk size: {validation_chunk_size}\n")

    with iohub.open_ome_zarr(source_path, mode="r") as src_plate, \
         iohub.open_ome_zarr(dest_path, mode="r") as dest_plate:
        validate_arrays(src_plate, dest_plate, max_positions=max_positions,
                       validation_chunk_size=validation_chunk_size, store_name=source_store)

    print("\n✓ Validation completed successfully - all data matches!")


def convert_to_v3(
    experiment: str,
    source_store: str = None,
    mode: str = "all",
    chunks: tuple = (1, 1, 1, 512, 512),
    shards_ratio: tuple = None,
    use_channel_sharding: bool = True,
    validate: bool = False,
    validate_max_positions: int = 3,
    overwrite: bool = None
):
    """
    Pipeline step wrapper that converts a Zarr v2 store to Zarr v3 format.
    Converts entire experiment sequentially (all positions and groups).

    For parallel conversion via SLURM, use convert_v3_slurm.py instead.

    Args:
        experiment: Experiment name
        source_store: Key for source store in OpsDataset.store_paths (deprecated, use mode instead)
        mode: Conversion mode: 'pheno', 'track', 'iss', or 'all' (default: 'all')
        chunks: Chunk dimensions for v3 store. If None, uses source chunks (default: (1, 1, 1, 512, 512))
        shards_ratio: Sharding ratio for v3 format. If None, auto-calculates based on channel count (default: None)
        use_channel_sharding: If True and shards_ratio is None, uses channel-based sharding (default: True)
        validate: If True, validate copied data matches source (default: False)
        validate_max_positions: Max positions to validate. If None, validates all (default: 3 for ~5min validation)
        overwrite: If True, overwrite existing output. If False, skip. If None, prompt user (default: None)
    """
    dataset = OpsDataset(experiment)

    # Map mode to source_store if mode is provided
    if mode and mode != "all":
        mode_to_source_store = {
            "pheno": "pheno_assembled",
            "track": "lc_5x_phase_2d_stitched",
            "iss": "iss_stitch_registered",
        }
        source_store = mode_to_source_store.get(mode, source_store)
    elif not source_store:
        # Default to pheno if no mode or source_store specified
        source_store = "pheno_assembled"

    # Handle 'all' mode - convert all three stores sequentially
    if mode == "all":
        stores_to_convert = [
            ("pheno", "pheno_assembled"),
            ("track", "lc_5x_phase_2d_stitched"),
            ("iss", "iss_stitch_registered"),
        ]

        for label, store_key in stores_to_convert:
            source_path = dataset.store_paths.get(store_key)
            dest_store_key = f"{store_key}_v3"
            dest_path = dataset.store_paths.get(dest_store_key)

            # Skip if paths don't exist
            if not source_path or not dest_path:
                print(f"⚠ Skipping {label}: store paths not configured")
                continue
            if not source_path.exists():
                print(f"⚠ Skipping {label}: source not found at {source_path}")
                continue

            print(f"\n{'='*70}")
            print(f"Converting {label} store ({store_key} → {dest_store_key})")
            print(f"{'='*70}\n")

            _convert_single_store(
                experiment=experiment,
                source_store=store_key,
                source_path=source_path,
                dest_path=dest_path,
                chunks=chunks,
                shards_ratio=shards_ratio,
                use_channel_sharding=use_channel_sharding,
                validate=validate,
                validate_max_positions=validate_max_positions,
                overwrite=overwrite,
            )

        print(f"\n{'='*70}")
        print("Completed conversion for all stores")
        print(f"{'='*70}\n")
        return

    # Single store conversion
    source_path = dataset.store_paths.get(source_store)
    if not source_path:
        raise ValueError(f"Unknown source store key: {source_store}")

    dest_store_key = f"{source_store}_v3"
    dest_path = dataset.store_paths.get(dest_store_key)
    if not dest_path:
        raise ValueError(f"Unknown destination store key: {dest_store_key}")

    _convert_single_store(
        experiment=experiment,
        source_store=source_store,
        source_path=source_path,
        dest_path=dest_path,
        chunks=chunks,
        shards_ratio=shards_ratio,
        use_channel_sharding=use_channel_sharding,
        validate=validate,
        validate_max_positions=validate_max_positions,
        overwrite=overwrite,
    )


def _convert_single_store(
    experiment: str,
    source_store: str,
    source_path: Path,
    dest_path: Path,
    chunks: tuple,
    shards_ratio: tuple,
    use_channel_sharding: bool,
    validate: bool,
    validate_max_positions: int,
    overwrite: bool,
):
    """Helper function to convert a single store."""

    # Check if destination exists and handle overwrite BEFORE initialization
    if not ensure_output_path(dest_path, prompt_user=True, overwrite=overwrite):
        print("Conversion skipped - destination exists and overwrite was declined")
        return

    # Determine sharding strategy
    if shards_ratio is None and use_channel_sharding:
        # Open source to get channel count
        with iohub.open_ome_zarr(source_path, mode="r") as source_plate:
            num_channels = len(source_plate.channel_names)

        print(f"\n{'='*70}")
        print(f"SHARDING STRATEGY: Channel-based (grouping all {num_channels} channels)")
        print(f"{'='*70}")
        shards_ratio = calculate_channel_based_shards(num_channels, chunks=chunks)
        print(f"{'='*70}\n")
    elif shards_ratio is None:
        # Default spatial-only sharding (~1GB shards)
        shards_ratio = (1, 1, 1, 64, 64)
        print(f"\n{'='*70}")
        print(f"SHARDING STRATEGY: Spatial-only (default)")
        print(f"  Shards ratio: {shards_ratio}")
        print(f"  Shard shape: (1, 1, 1, 16384, 16384) ~1GB per shard")
        print(f"{'='*70}\n")
    else:
        # User-provided shards_ratio
        chunk_size = chunks[-1]
        spatial_shard_size = chunk_size * shards_ratio[-1]
        print(f"\n{'='*70}")
        print(f"SHARDING STRATEGY: Custom")
        print(f"  Shards ratio: {shards_ratio}")
        print(f"  Shard shape: (1, {shards_ratio[1]}, 1, {spatial_shard_size}, {spatial_shard_size})")
        print(f"{'='*70}\n")

    print(f"Converting {source_store} to v3 for {experiment}\n"
          f"  Source: {source_path}\n"
          f"  Dest: {dest_path}\n"
          f"  Chunks: {chunks or 'source'}")

    # Pass overwrite=True to initialization since we've already handled cleanup above
    copy_zarrv2_to_zarrv3(source_path, dest_path, chunks=chunks, shards_ratio=shards_ratio, overwrite=True, experiment=experiment)

    if validate:
        validation_chunk_size = (tuple(c * s for c, s in zip(chunks, shards_ratio))
                                if chunks else (1, 1, 1, 4096, 4096))
        print(f"Validating (max_positions={validate_max_positions}, chunk_size={validation_chunk_size})")
        with iohub.open_ome_zarr(source_path, mode="r") as src_plate, \
             iohub.open_ome_zarr(dest_path, mode="r") as dest_plate:
            validate_arrays(src_plate, dest_plate, max_positions=validate_max_positions,
                          validation_chunk_size=validation_chunk_size, store_name=source_store)
        print("Validation completed")


def copy_pheno_channels_to_v3(
    pheno_v2_path,
    dest_v3_path,
    position_key: str,
    num_pheno_channels: int,
    store_custom_metadata: bool = False,
):
    """Copy the phenotyping channels (all pyramid levels) from a v2 store into the
    first ``num_pheno_channels`` channels of the pre-created v3 store, per position.

    Shared by the cell-painting and 4i converters (which both prepend pheno channels
    before appending fixed-cell channels). When ``store_custom_metadata`` is set, the
    source per-level clims + normalization are copied onto the dest position (the
    cell-painting path does this; 4i does not).
    """
    import tensorstore as ts

    pheno_v2_path = Path(pheno_v2_path)
    dest_v3_path = Path(dest_v3_path)

    with iohub.open_ome_zarr(pheno_v2_path, mode="r") as source_plate:
        src_pos = source_plate[position_key]
        per_level_clims = (
            collect_per_level_clims(pheno_v2_path, position_key, src_pos)
            if store_custom_metadata else None
        )
        for k, src_array in src_pos.images():
            src_shape = src_array.shape
            print(f"    [{position_key}] level {k}: copy {num_pheno_channels} pheno channels, shape {src_shape[-2:]}")
            src_ts = src_array.tensorstore()
            dst_ts = ts.open({
                'driver': 'zarr3',
                'kvstore': {'driver': 'file', 'path': str(dest_v3_path / position_key / k)},
            }).result()
            copy_array(src_ts, dst_ts[:, 0:num_pheno_channels, :, :, :])

        if store_custom_metadata:
            with iohub.open_ome_zarr(dest_v3_path, mode="r+") as dest_plate:
                dest_pos = dest_plate[position_key]
                array_zattrs = {}
                pos_normalization = src_pos.zattrs.get("normalization")
                if pos_normalization:
                    array_zattrs["normalization"] = pos_normalization
                if per_level_clims:
                    array_zattrs["clims_per_level"] = per_level_clims
                if array_zattrs:
                    dest_pos.zattrs["custom_metadata"] = array_zattrs
