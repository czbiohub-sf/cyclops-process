#!/usr/bin/env python3
"""Unified v3 conversion implementation for the fixed-cell pipeline (cell painting + 4i).

Merges the former processes/convert_v3_slurm_cp.py and convert_v3_slurm_4i.py into
one module. Both modalities warp fixed-cell channels/labels into the pheno 20x v3
frame; the shared chunked-affine warp is convert_v3_warp.warp_channels_into_v3, and
the modality-specific parts (channel specs, affine-chain topology, store init,
orchestration) live side by side here.

Called by the single CLI convert/v3_fixed_cli.py:
  - cp: submit_cell_paint_conversion_job / submit_cp_seg_only_job
  - 4i: _run_full_pipeline (drives this module's staged main() as a subprocess)
"""

from __future__ import annotations

import argparse
import sys
import os
import gc
import json
import re
import shutil
from pathlib import Path

import iohub
import numpy as np
import tensorstore as ts
import yaml
import zarr
from scipy.ndimage import zoom

sys.path.insert(0, os.getcwd())

from cyclops_process.convert.v3_common import (
    copy_pheno_channels_to_v3,
    convert_to_v3,
    convert_position_group_to_v3,
    initialize_v3_store,
    SUBGROUP_METADATA,
    calculate_channel_based_shards,
    validate_v3_conversion,
    copy_position_group_zarrv2_to_zarrv3,
    copy_zarr_array_v2_to_v3,
)
from cyclops_process.convert.v3_livecell import get_position_group_combinations
from cyclops_utils.hpc.slurm_batch_utils import (
    detect_experiments_needing_processing,
    handle_single_experiment_cli,
    submit_parallel_jobs,
    wait_for_multiple_job_arrays,
)
from cyclops_utils.io.zarr_utils import create_zarr_array
from cyclops_process.processes.pyramids.build_dask import build_seg_pyramid_only
from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.data.filesystem import parse_well
from cyclops_process.convert.v3_warp import (
    warp_channels_into_v3, affine_3x3_from_4x4,
    load_affine_from_yaml, compose_affines, _scale_affine_translation,
)
from cyclops_process.fixed_cp_4i.configs.four_i_config import (
    ROUNDS,
    NUM_ROUNDS,
    FOUR_I_CHANNELS,
    FOUR_I_COLORS,
    EXPERIMENT,
    get_default_output_dir,
    get_channel_name as _fouri_channel_name,
)

# ============================================================================
# Cell painting section
# ============================================================================

sys.path.insert(0, os.getcwd())


BASE_SLURM_PARAMS = {
    "timeout_min": 60,  # Base images take ~8-10min with 96× multiplier
    "mem": "250GB",  # Use ~50-60GB with 96× spatial chunking
    "cpus_per_task": 32,
    "slurm_partition": "cpu",
}


SEG_SLURM_PARAMS = {
    "timeout_min": 15,  # Increased from 5 - seg can have many labels requiring multiple spatial iterations
    "mem": "250GB",  # Need same memory as base for large arrays
    "cpus_per_task": 16,
    "slurm_partition": "cpu",
}


CELL_PAINT_SLURM_PARAMS = {
    "timeout_min": 240,  # Cell painting registration + pyramids takes longer
    "mem": "128GB",  # More memory for affine transforms + pyramids
    "cpus_per_task": 32,
    "slurm_partition": "cpu",
}


LABEL_SHARDS_RATIO = (1, 1, 1, 32, 32)


CELL_PAINTING_CHANNELS = {
    1: {  # Part 1
        0: {"name": "nuclei", "marker": "Hoechst", "structure": "Nucleus"},
        1: {"name": "mitochondria", "marker": "TOMM20", "structure": "Mitochondria"},
        2: {"name": "plasma_membrane", "marker": "WGA", "structure": "Plasma Membrane", "full_marker": "Wheat Germ Agglutinin"},
        3: {"name": "f_actin", "marker": "Phalloidin", "structure": "F-actin"},
    },
    2: {  # Part 2
        0: {"name": "nuclei", "marker": "Hoechst", "structure": "Nucleus"},
        1: {"name": "nucleoli", "marker": "NPM1", "structure": "Nucleoli"},
        2: {"name": "microtubules", "marker": "Tubulin", "structure": "Microtubules"},
        3: {"name": "ER", "marker": "ConA", "structure": "Endoplasmic Reticulum", "full_marker": "Concanavalin A"},
    },
}


CP_COLORS = {
    "nuclei": "0000FF",        # Blue
    "mitochondria": "FF0000",  # Red
    "plasma_membrane": "00FF00",  # Green
    "f_actin": "FF00FF",       # Magenta
    "nucleoli": "00FFFF",      # Cyan
    "microtubules": "FFFF00",  # Yellow
    "ER": "FFA500",            # Orange
}


CP_SEG_LABELS = {
    1: "CP1_nuclear_seg",
    2: "CP2_nuclear_seg",
}


CP_SEG_METADATA = {
    "CP1_nuclear_seg": {
        "annotation_type": "nuclear_segmentation",
        "description": "Nuclear segmentation from cell painting part 1 (Hoechst)",
        "is_ome_label": True,
        "part": 1,
        "source_channel": "nuclei",
        "marker": "Hoechst",
    },
    "CP2_nuclear_seg": {
        "annotation_type": "nuclear_segmentation",
        "description": "Nuclear segmentation from cell painting part 2 (Hoechst)",
        "is_ome_label": True,
        "part": 2,
        "source_channel": "nuclei",
        "marker": "Hoechst",
    },
}


def get_channel_name(part: int, channel: int, format: str = "short") -> str:
    """Get channel name for a given part and channel index."""
    info = CELL_PAINTING_CHANNELS.get(part, {}).get(channel, {})
    name = info.get("name", f"ch{channel}")
    marker = info.get("marker", "")
    if format == "short":
        return f"CP{part}_{name}"
    elif format == "full":
        return f"CP{part}_{name}_{marker}"
    return f"CP{part}_{name}"


def get_channel_metadata(part: int, channel: int) -> dict:
    """Get full metadata for a channel."""
    info = CELL_PAINTING_CHANNELS.get(part, {}).get(channel, {})
    name = info.get("name", f"ch{channel}")
    return {
        "name": get_channel_name(part, channel, format="short"),
        "full_name": get_channel_name(part, channel, format="full"),
        "marker": info.get("marker", "unknown"),
        "structure": info.get("structure", "unknown"),
        "full_marker": info.get("full_marker", info.get("marker", "unknown")),
        "part": part,
        "source_channel": channel,
        "color": CP_COLORS.get(name, "FFFFFF"),
    }


def build_channel_names(pheno_channel_names: list, parts: list) -> list:
    """Build complete channel name list for cell painting mode."""
    channel_names = list(pheno_channel_names)
    for part in parts:
        for ch in range(4):
            channel_names.append(get_channel_name(part, ch, format="full"))
    return channel_names


def load_composed_cp_affine(
    tracking_dir: Path,
    well,
    part: int,
    resolution_scale: float = 4.0,
) -> np.ndarray:
    """Compose CP→pheno affine from chained register YAMLs (compose-then-scale).

    Convention (set in 06_register_4i.py --cp):
      - A{well}_cell_painting1_register.yml stores pheno→CP1
      - A{well}_cell_painting{N}_register.yml (N>1) stores CP{N-1}→CP{N}

    YAMLs store the inverse (output→input) form scipy.ndimage.affine_transform consumes.
    Composition happens at 5x (the resolution at which each link was *computed*);
    the composed translation is then scaled 5x→20x. Returned 4x4 is at the
    destination's 20x resolution.
    """
    row, col = parse_well(well)  # accepts full unit ("A/1/0"), token, or int
    well_token = f"{row}{col}"  # row-A byte-identical to old A{well}
    composed = None
    for p in range(1, part + 1):
        reg_yaml = tracking_dir / f"{well_token}_cell_painting{p}_register.yml"
        if not reg_yaml.exists():
            raise FileNotFoundError(f"Missing CP register YAML in chain: {reg_yaml}")
        affine_p = load_affine_from_yaml(reg_yaml)
        composed = affine_p if composed is None else (affine_p @ composed)
    composed = composed.copy()
    composed[1, 3] *= resolution_scale
    composed[2, 3] *= resolution_scale
    return composed


def initialize_v3_store_with_cell_paint(
    pheno_v2_path: Path,
    output_v3_path: Path,
    parts: list,
    overwrite: bool = False,
    skip_overlays: bool = False,
    exclude_groups: set = None,
    experiment: str = None,
) -> tuple:
    """
    Initialize a v3 zarr store with expanded channel count for cell painting.

    This replaces the standard initialize_v3_store for cell_paint mode:
    - Reads phenotyping v2 store structure
    - Calculates total channels (pheno + cell painting parts)
    - Creates v3 store with correct sharding for all channels
    - Pre-creates all pyramid level arrays with proper sharding

    Returns:
        tuple: (target_shape, pheno_channel_names, all_channel_names, shards_ratio)
    """
    print(f"\nInitializing v3 store with cell painting channels at {output_v3_path}")

    if exclude_groups is None:
        exclude_groups = set()

    if output_v3_path.exists():
        if overwrite:
            print(f"  Removing existing store...")
            shutil.rmtree(output_v3_path)
        else:
            raise FileExistsError(f"Output exists: {output_v3_path}. Use --force to overwrite.")

    # Read phenotyping v2 metadata - get per-position shapes since they can differ
    with iohub.open_ome_zarr(pheno_v2_path, mode="r") as pheno_plate:
        pheno_channel_names = list(pheno_plate.channel_names)
        num_pheno_channels = len(pheno_channel_names)

        # Get position keys
        position_keys = [pos_key for pos_key, _ in pheno_plate.positions()]

        # Collect per-position level shapes (different wells can have different spatial dims)
        position_level_shapes = {}  # {pos_key: {level_key: shape}}
        level_transforms = {}  # Same transforms for all positions

        for pos_key, pos in pheno_plate.positions():
            position_level_shapes[pos_key] = {}
            for level in range(5):
                level_key = str(level)
                try:
                    position_level_shapes[pos_key][level_key] = pos[level_key].shape
                    # Get transforms from first position only (same for all)
                    if not level_transforms.get(level_key):
                        transforms = [x for x in pos.metadata.multiscales[0].datasets
                                     if x.path == level_key]
                        if transforms:
                            level_transforms[level_key] = transforms[0].coordinate_transformations
                except KeyError:
                    break

        # Get shape from first position for reference logging
        first_pos_key = position_keys[0]
        pheno_shape = position_level_shapes[first_pos_key]["0"]
        num_levels = len(position_level_shapes[first_pos_key])

    # Build combined channel names
    all_channel_names = build_channel_names(pheno_channel_names, parts)
    num_total_channels = len(all_channel_names)
    num_cp_channels = 4 * len(parts)

    print(f"  Phenotyping channels: {num_pheno_channels} ({pheno_channel_names})")
    print(f"  Cell painting channels: {num_cp_channels} ({len(parts)} parts × 4 channels)")
    print(f"  Total channels: {num_total_channels}")
    print(f"  Channel names: {all_channel_names}")

    # Calculate target shape (reference shape from first position)
    target_shape = (pheno_shape[0], num_total_channels, pheno_shape[2], pheno_shape[3], pheno_shape[4])
    print(f"  Reference shape (first position): {target_shape}")
    print(f"  Pyramid levels: {num_levels}")

    # Calculate sharding using the same function as standard conversion
    # This ensures consistent ~1GB shard sizes with fixed spatial ratios across all pyramid levels
    chunks = (1, 1, 1, 512, 512)
    shards_ratio = calculate_channel_based_shards(num_total_channels, chunks=chunks)
    shard_size = tuple(c * r for c, r in zip(chunks, shards_ratio))

    print(f"  Chunks: {chunks}")
    print(f"  Shards ratio: {shards_ratio}")
    print(f"  Shard size: {shard_size}")

    # Estimate shard file size
    shard_voxels = np.prod(shard_size)
    shard_bytes = shard_voxels * 4  # float32
    shard_gb = shard_bytes / (1024 ** 3)
    print(f"  Shard file size (uncompressed): ~{shard_gb:.2f} GB per shard")

    # Create v3 store with iohub - create positions AND arrays in one context
    # IMPORTANT: Each position uses its own shape from the source (wells can differ)
    with iohub.open_ome_zarr(
        output_v3_path,
        mode="w",
        layout="hcs",
        channel_names=all_channel_names,
        version="0.5"
    ) as dest_plate:
        # Create all positions with their pyramid arrays
        for pos_key in position_keys:
            row, col, fov = pos_key.split("/")
            dest_pos = dest_plate.create_position(row, col, fov)

            # Get this position's level shapes (different wells can have different spatial dims)
            pos_level_shapes = position_level_shapes[pos_key]

            # Create all pyramid level arrays for this position
            for level_key, src_shape in pos_level_shapes.items():
                # Calculate target shape for this level (expanded channels)
                level_target_shape = (
                    src_shape[0],
                    num_total_channels,
                    src_shape[2],
                    src_shape[3],
                    src_shape[4],
                )

                # Use the SAME fixed shards_ratio for all levels (matching standard conversion)
                # This results in fewer shards at lower resolution levels, which is the expected behavior

                # Get transform metadata for this level
                transform_meta = level_transforms.get(level_key)

                # Create destination array with expanded shape
                dest_pos.create_zeros(
                    level_key,
                    shape=level_target_shape,
                    dtype=np.float32,
                    chunks=chunks,
                    shards_ratio=shards_ratio,  # Use same fixed ratio for all levels
                    transform=transform_meta,
                )

            pos_shape = pos_level_shapes["0"]
            print(f"  Created position {pos_key} with {len(pos_level_shapes)} levels, shape {pos_shape[-2:]} (Y,X)")

    # Create labels group and subgroups
    OVERLAY_GROUPS = {"grid_edges", "grid_props", "grid_overlay", "iss_points", "iss_points_props", "iss_gene_image", "iss_guide_image"}

    src_zarr_store = zarr.open(str(pheno_v2_path), mode="r")
    dest_zarr_store = zarr.open(str(output_v3_path), mode="r+")

    for pos_key in position_keys:
        src_pos_group = src_zarr_store[pos_key]
        dest_pos_group = dest_zarr_store[pos_key]

        # Create labels group at position level
        labels_group = dest_pos_group.create_group("labels", overwrite=True)
        print(f"  Created labels group {pos_key}/labels")

        # Get all subgroup names from source. v3 sources nest label groups under a
        # `labels/` group, so `group_keys()` returns just ["labels"] — that container
        # is NOT itself a label; recreating it produced a spurious labels/labels group.
        # (Legacy v2 sources put label groups at position level, hence the fallback.)
        all_subgroups = [g for g in src_pos_group.group_keys() if g != "labels"]

        # Filter out overlays if skip_overlays is True
        groups_to_skip = set()
        if skip_overlays:
            groups_to_skip.update(OVERLAY_GROUPS & set(all_subgroups))

        # Filter out excluded groups
        groups_to_skip.update(exclude_groups & set(all_subgroups))

        subgroups_to_create = [name for name in all_subgroups if name not in groups_to_skip]

        # Add cell painting segmentation labels for each part
        cp_seg_labels = [CP_SEG_LABELS[part] for part in parts if part in CP_SEG_LABELS]
        subgroups_to_create.extend(cp_seg_labels)

        if groups_to_skip:
            print(f"  Skipping groups: {groups_to_skip}")

        # Filter to only OME-compliant labels (segmentations) from the filtered list
        # Include both standard labels and CP seg labels
        ome_label_names = [name for name in subgroups_to_create
                          if SUBGROUP_METADATA.get(name, {}).get("is_ome_label", False)
                          or CP_SEG_METADATA.get(name, {}).get("is_ome_label", False)]

        # Set labels metadata with both OME labels and all labels (filtered)
        if subgroups_to_create:
            labels_attrs = {}
            if ome_label_names:
                labels_attrs["ome"] = {
                    "version": "0.5",
                    "labels": ome_label_names
                }
            labels_attrs["labels"] = subgroups_to_create
            labels_group.attrs.update(labels_attrs)
            print(f"  Set OME labels metadata: {ome_label_names}")
            print(f"  Set all labels: {subgroups_to_create}")

        # Create each subgroup under labels/ (filtered)
        for group_name in subgroups_to_create:
            labels_group.create_group(group_name, overwrite=True)
            print(f"  Created subgroup {pos_key}/labels/{group_name}")

        # Write comprehensive metadata to each segmentation label subgroup
        from cyclops_process.convert.v3_metadata import build_label_metadata
        for group_name in subgroups_to_create:
            label_metadata = build_label_metadata(
                group_name, all_channel_names, experiment
            )
            if group_name in labels_group:
                labels_group[group_name].attrs["segmentation_metadata"] = label_metadata
        print(f"  Wrote segmentation_metadata for {len(subgroups_to_create)} labels")

    # Set OMERO metadata for all positions (channel labels, colors, cell_painting info)
    print("\n  Setting OMERO channel metadata...")
    for pos_key in position_keys:
        _update_omero_metadata(
            dest_v3_path=output_v3_path,
            position_key=pos_key,
            pheno_channel_names=pheno_channel_names,
            parts=parts,
            pheno_v2_path=pheno_v2_path,
        )

    # Write channels_metadata to plate-level zarr.json
    # _update_omero_metadata only writes position-level zarr.json; the plate root
    # needs attributes.channels_metadata for downstream tools.
    from cyclops_process.convert.v3_metadata import build_channels_metadata
    import re as _re
    channels_meta = build_channels_metadata(all_channel_names, experiment=experiment)
    # Enrich CP channels - match by trying both CP{part}_{name}_{marker} and CP{part}_{name}
    for i, name in enumerate(all_channel_names):
        if not name.startswith("CP"):
            continue
        for part, part_channels in CELL_PAINTING_CHANNELS.items():
            for ch_idx, ch_info in part_channels.items():
                ch_name = ch_info["name"]
                ch_marker = ch_info["marker"]
                full = f"CP{part}_{ch_name}_{ch_marker}"
                short = f"CP{part}_{ch_name}"
                if name not in (full, short):
                    continue
                marker = ch_info.get("full_marker", ch_marker)
                structure = ch_info["structure"]
                channels_meta[i]["channel_type"] = "fluorescent"
                channels_meta[i]["biological_annotation"] = {
                    "organelle": structure,
                    "marker": marker,
                    "marker_type": "nuclear_dye" if ch_marker == "Hoechst" else "direct_label",
                    "full_label": f"{structure}, {marker}",
                }
                channels_meta[i]["description"] = (
                    f"Cell painting {structure} visualized via {marker} (Part {part})"
                )
                break
    plate_json_path = output_v3_path / "zarr.json"
    with open(plate_json_path, "r") as f:
        plate_meta = json.load(f)
    plate_meta.setdefault("attributes", {})["channels_metadata"] = channels_meta
    with open(plate_json_path, "w") as f:
        json.dump(plate_meta, f, indent=2)
    print(f"  Set plate-level channels_metadata ({len(channels_meta)} channels)")

    print("v3 store initialization complete\n")

    return target_shape, pheno_channel_names, all_channel_names, shards_ratio


def convert_position_with_cell_paint(
    experiment: str,
    position_key: str,
    pheno_v2_path: str,
    dest_v3_path: str,
    cell_painting_dir: str,
    tracking_dir: str,
    parts: list,
    pheno_channel_names: list,
    chunks: tuple = (1, 1, 1, 512, 512),
    shards_ratio: tuple = (1, 1, 1, 64, 64),
):
    """
    Convert a single position from phenotyping v2 to v3 with cell painting channels.

    This function:
    1. Converts phenotyping channels from v2 to v3
    2. Applies registration and writes cell painting channels
    3. Builds pyramid levels for all channels

    Note: Target shape is determined at runtime from destination arrays since
    different wells can have different spatial dimensions.

    Args:
        experiment: Experiment name
        position_key: Position key like "A/1/0"
        pheno_v2_path: Path to phenotyping v2 zarr
        dest_v3_path: Path to destination v3 zarr
        cell_painting_dir: Path to cell painting zarr directory
        tracking_dir: Path to tracking directory with registration YAMLs
        parts: List of cell painting parts to include (e.g., [1, 2])
        pheno_channel_names: List of phenotyping channel names
        chunks: Chunk dimensions
        shards_ratio: Sharding ratio for v3 format
    """
    pheno_v2_path = Path(pheno_v2_path)
    dest_v3_path = Path(dest_v3_path)
    cell_painting_dir = Path(cell_painting_dir)
    tracking_dir = Path(tracking_dir)

    # Carry the full row/col unit; helpers derive row,col via parse_well
    well = position_key

    print(f"\n{'='*60}")
    print(f"Converting position {position_key} with cell painting")
    print(f"{'='*60}")
    print(f"  Phenotyping source: {pheno_v2_path}")
    print(f"  Destination: {dest_v3_path}")
    print(f"  Cell painting dir: {cell_painting_dir}")
    print(f"  Parts: {parts}")

    num_pheno_channels = len(pheno_channel_names)
    num_cp_channels = 4 * len(parts)
    num_total_channels = num_pheno_channels + num_cp_channels

    # ========================================================================
    # Step 1: Copy phenotyping channels from v2 to v3 (all pyramid levels)
    # Arrays are pre-created during initialization, just need to write data
    # Uses copy_array from convert_v3.py which has optimized spatial chunking
    # ========================================================================
    print(f"\n  Step 1: Copying {num_pheno_channels} phenotyping channels...")
    copy_pheno_channels_to_v3(
        pheno_v2_path, dest_v3_path, position_key, num_pheno_channels,
        store_custom_metadata=True,
    )

    # ========================================================================
    # Step 2: Transform and write cell painting channels (level 0 only)
    # ========================================================================
    print(f"\n  Step 2: Adding cell painting channels...")

    for i, part in enumerate(parts):
        start_channel = num_pheno_channels + (i * 4)
        cp_path = cell_painting_dir / f"part{part}_max_proj_flatfield_stitched.zarr"

        if not cp_path.exists():
            print(f"    [WARNING] Cell painting part {part} not found: {cp_path}")
            continue
        try:
            composed_affine = load_composed_cp_affine(tracking_dir, well, part)
        except FileNotFoundError as e:
            print(f"    [WARNING] {e}")
            continue

        print(f"    Part {part}: channels {start_channel}-{start_channel+3} (composed {part}-step affine chain)")
        _transform_and_write_cell_painting(
            dest_v3_path=dest_v3_path,
            cell_painting_path=cp_path,
            affine_4x4=composed_affine,
            position_key=position_key,
            well=well,
            part=part,
            start_channel=start_channel,
        )

    # ========================================================================
    # Step 3: Build pyramid levels for cell painting channels
    # ========================================================================
    print(f"\n  Step 3: Building pyramid levels for cell painting channels...")

    cp_channels = list(range(num_pheno_channels, num_total_channels))
    _build_pyramid_levels(
        dest_v3_path=dest_v3_path,
        position_key=position_key,
        channels_to_update=cp_channels,
    )

    # NOTE: Step 4 (CP nuclear segmentation) is handled by separate parallel jobs
    # (cp_nuclear_seg_jobs) to allow concurrent execution with base image conversion.
    # See submit_cell_paint_conversion_job() for job submission logic.

    # Note: OMERO metadata (channel labels, colors, cell_painting info) is set
    # during initialization, not here, to avoid redundant writes

    print(f"\n  Position {position_key} complete!")


def _transform_and_write_cell_painting(
    dest_v3_path: Path,
    cell_painting_path: Path,
    affine_4x4: np.ndarray,
    position_key: str,
    well,
    part: int,
    start_channel: int,
    spatial_chunk_multiplier: int = 48,
):
    """Warp cell-painting channels (order=1) into the pheno v3 frame (level 0).

    affine_4x4 is the composed pheno->CP{part} transform at the destination's 20x
    resolution. The chunked affine warp itself is the shared engine in convert_v3_warp.
    """
    affine_3x3 = affine_3x3_from_4x4(affine_4x4)
    print(f"      Affine 3x3 (YX): scale={affine_3x3[0,0]:.4f}, "
          f"translate=({affine_3x3[0,2]:.2f}, {affine_3x3[1,2]:.2f})")

    src_row, src_col = parse_well(well)  # row-agnostic source position
    source_store = zarr.open(cell_painting_path, mode="r")
    source_array = source_store[f"{src_row}/{src_col}/0/0"]  # (T, C, Z, Y, X)
    num_src_channels = source_array.shape[1]

    dest_ts = ts.open({
        'driver': 'zarr3',
        'kvstore': {'driver': 'file', 'path': str(dest_v3_path / position_key / "0")},
    }).result()

    for src_ch in range(num_src_channels):
        ch_meta = get_channel_metadata(part, src_ch)
        print(f"        Channel {src_ch} ({ch_meta['name']}) -> Channel {start_channel + src_ch}")

    warp_channels_into_v3(
        source_array, dest_ts, affine_3x3,
        [(c, start_channel + c) for c in range(num_src_channels)],
        order=1, dtype=np.float32,
        spatial_chunk_multiplier=spatial_chunk_multiplier, label=f"CP{part}",
    )

    del source_store
    gc.collect()


def _build_pyramid_levels(
    dest_v3_path: Path,
    position_key: str,
    channels_to_update: list = None,
    num_levels: int = 5,
    spatial_chunk_multiplier: int = 32,  # Use larger chunks for efficiency
):
    """Generate downsampled pyramid levels 1-4 from level 0.

    Uses large spatial chunks for efficient processing, similar to copy_array optimization.
    """
    import time
    print(f"    Building pyramids for channels {channels_to_update}")

    # Open level 0 to get shape
    level0_ts = ts.open({
        'driver': 'zarr3',
        'kvstore': {'driver': 'file', 'path': str(dest_v3_path / position_key / "0")},
    }).result()

    shape = level0_ts.shape
    num_channels = shape[1]

    if channels_to_update is None:
        channels_to_update = list(range(num_channels))

    # For each level 1-4
    for level in range(1, num_levels):
        level_start_time = time.time()
        src_level = level - 1

        src_ts = ts.open({
            'driver': 'zarr3',
            'kvstore': {'driver': 'file', 'path': str(dest_v3_path / position_key / str(src_level))},
        }).result()

        dest_ts = ts.open({
            'driver': 'zarr3',
            'kvstore': {'driver': 'file', 'path': str(dest_v3_path / position_key / str(level))},
        }).result()

        src_shape = src_ts.shape
        dest_shape = dest_ts.shape

        # Use large chunks (32× base chunk = ~16K pixels)
        # Scale down for higher pyramid levels since they're smaller
        level_multiplier = max(1, spatial_chunk_multiplier // (2 ** (level - 1)))
        chunk_size = 512 * level_multiplier

        dest_h, dest_w = dest_shape[-2:]
        y_iterations = (dest_h + chunk_size - 1) // chunk_size
        x_iterations = (dest_w + chunk_size - 1) // chunk_size
        total_iterations = y_iterations * x_iterations * len(channels_to_update)

        print(f"      Level {level}: {src_shape[-2:]} -> {dest_shape[-2:]} (2x downsample, {chunk_size}×{chunk_size} chunks)")

        # Downsample each channel with progress reporting
        iteration = 0
        last_report_time = time.time()
        for c in channels_to_update:
            for y_start in range(0, dest_h, chunk_size):
                y_end = min(y_start + chunk_size, dest_h)
                for x_start in range(0, dest_w, chunk_size):
                    x_end = min(x_start + chunk_size, dest_w)
                    iteration += 1

                    if iteration % 10 == 0 or iteration == 1:
                        now = time.time()
                        elapsed = now - level_start_time
                        iter_time = now - last_report_time
                        avg_per_iter = elapsed / iteration
                        remaining = avg_per_iter * (total_iterations - iteration)
                        print(f"        Progress: {iteration}/{total_iterations} ({100*iteration/total_iterations:.1f}%) "
                              f"[{iter_time:.1f}s last 10, {avg_per_iter:.2f}s/iter, ~{remaining:.0f}s remaining]")
                        last_report_time = now

                    # Read from source (2x the coordinates)
                    src_y_start = y_start * 2
                    src_y_end = min(y_end * 2, src_shape[-2])
                    src_x_start = x_start * 2
                    src_x_end = min(x_end * 2, src_shape[-1])

                    src_data = src_ts[0, c, 0, src_y_start:src_y_end, src_x_start:src_x_end].read().result()

                    # Downsample 2x using scipy zoom
                    dest_h_chunk = y_end - y_start
                    dest_w_chunk = x_end - x_start

                    if src_data.shape[0] > 0 and src_data.shape[1] > 0:
                        downsampled = zoom(src_data.astype(np.float32), 0.5, order=1)
                        actual_h = min(downsampled.shape[0], dest_h_chunk)
                        actual_w = min(downsampled.shape[1], dest_w_chunk)
                        dest_ts[0, c, 0, y_start:y_start + actual_h, x_start:x_start + actual_w].write(
                            downsampled[:actual_h, :actual_w]
                        ).result()
                        del downsampled

                    del src_data

        level_time = time.time() - level_start_time
        print(f"      Level {level} completed in {level_time:.1f}s")


def convert_cp_nuclear_seg_to_v3(
    dest_v3_path: Path,
    cell_painting_dir: Path,
    tracking_dir: Path,
    position_key: str,
    parts: list,
    chunks: tuple = (1, 1, 1, 512, 512),
    shards_ratio: tuple = (1, 1, 1, 32, 32),
    spatial_chunk_multiplier: int = 32,
    build_pyramids: bool = True,
    num_levels: int = 5,
    source_path=None,
):
    """
    Convert cell painting nuclear segmentations to v3 labels format, and carry the
    source store's pheno label groups (nuclear_seg, cell_seg, overlays) into the
    cp store. This is the sole per-position writer of the labels/ group, so it also
    finalizes the labels-group OME attrs at the end (no cross-job attrs race).

    This function:
    1. Reads the upscaled 20x nuclear segmentation from part{N}_max_proj_segmentation.zarr
    2. Applies the same registration transform as the cell painting channels
    3. Writes to the v3 store's labels group (e.g., labels/CP1_nuclear_seg)
    4. Builds pyramid levels for the segmentation (optional)

    Processing is done in spatial chunks to avoid OOM on large arrays (~100K x 100K).

    The segmentation must have been upscaled first using:
        python -m cyclops_process.ops0094.segment_94 upscale --seg-store-path <path>

    Args:
        dest_v3_path: Path to destination v3 zarr store
        cell_painting_dir: Path to cell painting directory (contains part{N}_max_proj_segmentation.zarr)
        tracking_dir: Path to tracking directory with registration YAMLs
        position_key: Position key like "A/1/0"
        parts: List of cell painting parts (e.g., [1, 2])
        chunks: Chunk dimensions for v3 format
        shards_ratio: Sharding ratio for v3 format
        spatial_chunk_multiplier: Multiplier for spatial chunk size (default: 32 -> 16K chunks)
        build_pyramids: Whether to build pyramid levels after base conversion (default: True)
        num_levels: Number of pyramid levels to build (default: 5)
    """
    from scipy.ndimage import affine_transform as scipy_affine_transform
    import time

    dest_v3_path = Path(dest_v3_path)
    cell_painting_dir = Path(cell_painting_dir)
    tracking_dir = Path(tracking_dir)

    # Carry the full unit; derive row,col for source positions
    well = position_key
    row, col = parse_well(position_key)

    print(f"\n  Converting CP nuclear segmentations for {position_key}")

    # Carry the source pheno labels (nuclear_seg, cell_seg, iss overlays) into the
    # cp store. Both are v3 in the same coordinate space, so this is a direct copy.
    # Skips any group already written (the CP seg labels are created below).
    if source_path:
        copy_v3_label_groups(source_path, dest_v3_path, position_key,
                             skip_names=set(CP_SEG_LABELS.values()))

    for part in parts:
        seg_label = CP_SEG_LABELS.get(part)
        if not seg_label:
            print(f"    [WARN] No label defined for part {part}")
            continue

        # Path to upscaled segmentation
        seg_store_path = cell_painting_dir / f"part{part}_max_proj_segmentation.zarr"
        upscaled_seg_path = seg_store_path / f"{row}/{col}/0" / "20x_nuclear_seg"

        if not upscaled_seg_path.exists():
            print(f"    [WARN] Upscaled segmentation not found: {upscaled_seg_path}")
            print(f"           Run: python -m cyclops_process.ops0094.segment_94 upscale --seg-store-path {seg_store_path}")
            continue

        # Get registration transform (composed chain pheno→CP{part}, 20x-scaled)
        try:
            affine_4x4 = load_composed_cp_affine(tracking_dir, well, part)
        except FileNotFoundError as e:
            print(f"    [WARN] {e}")
            continue

        print(f"    Part {part}: {seg_label}")
        print(f"      Source: {upscaled_seg_path}")
        print(f"      Composed {part}-step affine chain (pheno→CP{part}, 20x resolution)")

        # Extract 2D 3x3 affine (YX plane with translation)
        affine_3x3 = np.identity(3)
        affine_3x3[0:2, 0:2] = affine_4x4[1:3, 1:3]
        affine_3x3[0:2, 2] = affine_4x4[1:3, 3]

        # Open source segmentation (v2 zarr) - don't load into memory yet!
        # iohub.create_image stores the array directly at 20x_nuclear_seg (not 20x_nuclear_seg/0)
        source_store = zarr.open(str(seg_store_path), mode="r")
        source_array = source_store[f"{row}/{col}/0/20x_nuclear_seg"]
        source_shape = source_array.shape
        # Handle 5D (T, C, Z, Y, X) or 2D (Y, X) shapes
        if len(source_shape) == 5:
            source_h, source_w = source_shape[-2:]
        else:
            source_h, source_w = source_shape

        # Get target shape from destination v3 store
        dest_ts = ts.open({
            'driver': 'zarr3',
            'kvstore': {'driver': 'file', 'path': str(dest_v3_path / position_key / "0")},
        }).result()
        target_h, target_w = dest_ts.shape[-2:]

        print(f"      Source shape: {source_h}×{source_w}, Target shape: {target_h}×{target_w}")

        # Create the destination array first
        dest_label_path = dest_v3_path / position_key / "labels" / seg_label

        dest_zarr = zarr.open(str(dest_v3_path), mode="r+")
        labels_group = dest_zarr[position_key]["labels"]

        # Create the label subgroup if it doesn't exist (for stores created before CP seg support)
        if seg_label not in labels_group:
            print(f"      Creating labels subgroup: {seg_label}")
            labels_group.create_group(seg_label)

            # Update the labels metadata to include the new label
            labels_attrs = dict(labels_group.attrs)

            # Update ome.labels list
            if "ome" not in labels_attrs:
                labels_attrs["ome"] = {"version": "0.5", "labels": []}
            if seg_label not in labels_attrs["ome"].get("labels", []):
                labels_attrs["ome"]["labels"] = labels_attrs["ome"].get("labels", []) + [seg_label]

            # Update labels list
            if "labels" not in labels_attrs:
                labels_attrs["labels"] = []
            if seg_label not in labels_attrs["labels"]:
                labels_attrs["labels"] = labels_attrs["labels"] + [seg_label]

            labels_group.attrs.update(labels_attrs)
            print(f"      Updated labels metadata to include: {seg_label}")

        # Create the array (shape: 1, 1, 1, Y, X for 5D consistency)
        output_shape = (1, 1, 1, target_h, target_w)
        create_zarr_array(
            path=str(dest_label_path / "0"),
            shape=output_shape,
            dtype=np.int32,
            chunks=chunks,
            zarr_format=3,
            shards_ratio=shards_ratio,  # Use same sharding as seg/nuclear_seg
            fill_value=0,
            overwrite=True,
        )

        # Open destination with tensorstore for writing
        dest_label_ts = ts.open({
            'driver': 'zarr3',
            'kvstore': {'driver': 'file', 'path': str(dest_label_path / "0")},
        }).result()

        # Warp the label into the pheno frame (nearest-neighbor), shared engine.
        warp_channels_into_v3(
            source_array, dest_label_ts, affine_3x3, [(0, 0)],
            order=0, dtype=np.int32, source_is_2d=(len(source_shape) != 5),
            spatial_chunk_multiplier=spatial_chunk_multiplier, label=seg_label,
        )

        # Build pyramid levels if requested
        if build_pyramids and num_levels > 1:
            print(f"      Building {num_levels - 1} pyramid levels for {seg_label}...")
            pyramid_start = time.time()
            build_seg_pyramid_only(
                source_store=dest_v3_path,
                levels=num_levels,
                positions=[position_key],
                resume=False,  # Always build fresh since we just created base
                seg_types=[seg_label],
                shards_ratio=shards_ratio,
                chunks=chunks,
            )
            print(f"      Pyramid build completed in {time.time() - pyramid_start:.1f}s")

        # Set metadata for the label (OME-NGFF image-label + custom metadata)
        label_metadata = CP_SEG_METADATA.get(seg_label, {})

        # Build datasets list for all pyramid levels
        datasets_list = [{"path": str(lvl)} for lvl in range(num_levels if build_pyramids else 1)]

        # OME-NGFF image-label metadata structure
        ome_label_metadata = {
            "image-label": {
                "version": "0.5",
                "colors": [
                    {"label-value": 0, "rgba": [0, 0, 0, 0]},  # Background transparent
                ],
                "source": {
                    "image": f"../../0",  # Reference to base image
                },
            },
            "multiscales": [{
                "version": "0.5",
                "name": seg_label,
                "axes": [
                    {"name": "t", "type": "time"},
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": datasets_list,
            }],
            "custom_metadata": label_metadata,
        }

        # Re-open label group to update attrs (may have been modified by pyramid builder)
        dest_zarr = zarr.open(str(dest_v3_path), mode="r+")
        dest_label_group = dest_zarr[position_key]["labels"][seg_label]
        dest_label_group.attrs.update(ome_label_metadata)

        print(f"      Written {seg_label} with {len(datasets_list)} levels to {dest_label_path}")

        gc.collect()

    # Authoritative: make the labels-group OME attrs match the groups on disk
    # (copied pheno labels + CP seg labels).
    finalize_labels_attrs(dest_v3_path, position_key)

    print(f"  CP nuclear segmentation conversion complete for {position_key}")


def _update_omero_metadata(
    dest_v3_path: Path,
    position_key: str,
    pheno_channel_names: list,
    parts: list,
    pheno_v2_path: Path = None,
):
    """Update OMERO channel metadata with proper labels and colors."""
    pos_path = dest_v3_path / position_key
    pos_json_path = pos_path / "zarr.json"

    # Read existing metadata from output
    with open(pos_json_path, "r") as f:
        pos_meta = json.load(f)

    # Read source phenotyping metadata if available
    src_omero_channels = []
    src_custom_metadata = {}
    if pheno_v2_path is not None:
        # Try reading from v2 .zattrs
        src_pos_zattrs = pheno_v2_path / position_key / ".zattrs"
        if src_pos_zattrs.exists():
            with open(src_pos_zattrs, "r") as f:
                src_attrs = json.load(f)
            src_omero = src_attrs.get("omero", {})
            src_omero_channels = src_omero.get("channels", [])
            # Custom metadata might be at position level
            src_custom_metadata = src_attrs.get("custom_metadata", {})
            if src_omero_channels:
                print(f"    Loaded {len(src_omero_channels)} source OMERO channels from v2")

    # Build channel metadata
    channels = []

    # Phenotyping channels - preserve original metadata if available
    for i, name in enumerate(pheno_channel_names):
        if i < len(src_omero_channels):
            channels.append(src_omero_channels[i])
        else:
            channels.append({
                "active": True,
                "coefficient": 1.0,
                "color": "FFFFFF",
                "family": "linear",
                "inverted": False,
                "label": name,
                "window": {
                    "start": 0.0,
                    "end": 65535.0,
                    "min": 0.0,
                    "max": 65535.0,
                },
            })

    # Cell painting channels
    for part in parts:
        for ch in range(4):
            meta = get_channel_metadata(part, ch)
            channels.append({
                "active": True,
                "coefficient": 1.0,
                "color": meta["color"],
                "family": "linear",
                "inverted": False,
                "label": meta["full_name"],
                "window": {
                    "start": 0.0,
                    "end": 65535.0,
                    "min": 0.0,
                    "max": 65535.0,
                },
                "cell_painting": {
                    "part": part,
                    "source_channel": ch,
                    "marker": meta["marker"],
                    "full_marker": meta["full_marker"],
                    "structure": meta["structure"],
                },
            })

    # Update OMERO metadata in the ome block
    if "attributes" not in pos_meta:
        pos_meta["attributes"] = {}
    if "ome" not in pos_meta["attributes"]:
        pos_meta["attributes"]["ome"] = {}

    pos_meta["attributes"]["ome"]["omero"] = {
        "id": 0,
        "name": "0",
        "version": "0.5",
        "channels": channels,
        "rdefs": {
            "defaultT": 0,
            "defaultZ": 0,
            "model": "color",
            "projection": "normal"
        }
    }

    # Preserve custom_metadata from source if available
    if src_custom_metadata:
        pos_meta["attributes"]["custom_metadata"] = src_custom_metadata
        print(f"    Preserved custom_metadata from source")

    # Write updated metadata
    with open(pos_json_path, "w") as f:
        json.dump(pos_meta, f, indent=2)

    print(f"    Set {len(channels)} channel entries")


def submit_cp_seg_only_job(experiment: str, args) -> dict:
    """Submit SLURM jobs to add CP nuclear segmentations to an existing v3 store.

    This mode is for adding cell painting nuclear segmentations to a v3 store
    that was already created (e.g., via --mode cell_paint). It does NOT re-convert
    the base images or phenotyping channels.

    Prerequisites:
    1. The v3 store must already exist (run --mode cell_paint first)
    2. Segmentations must be upscaled (run segment_94.py upscale first)
    """
    dataset = OpsDataset(experiment)

    # Get paths
    dest_path = dataset.store_paths.get("pheno_assembled_v3")
    exp_path = dataset.experiment_path
    cell_painting_dir = exp_path / "0-convert" / "cell_painting"
    tracking_dir = exp_path / "2-tracking"

    # Validate paths
    if not dest_path.exists():
        print(f"v3 store not found: {dest_path}")
        print("Run --mode cell_paint first to create the v3 store.")
        return {"submitted": 0, "failed": 0}

    if not cell_painting_dir.exists():
        print(f"Cell painting directory not found: {cell_painting_dir}")
        return {"submitted": 0, "failed": 0}

    # Get position keys from existing v3 store
    with iohub.open_ome_zarr(dest_path, mode="r") as dest_plate:
        position_keys = sorted([pos_key for pos_key, _ in dest_plate.positions()])

    # Optionally filter by --wells argument
    if args.wells:
        position_keys = [pk for pk in position_keys if int(pk.split("/")[1]) in args.wells]

    if not position_keys:
        print(f"No positions found for {experiment}")
        return {"submitted": 0, "failed": 0}

    # Print job summary
    print(f"\n{'='*60}")
    print(f"CP Segmentation-Only Conversion Job Summary")
    print(f"{'='*60}")
    print(f"Experiment: {experiment}")
    print(f"Dest: {dest_path}")
    print(f"Positions: {len(position_keys)}")
    print(f"Parts: {args.parts}")
    print(f"{'='*60}\n")

    # Create jobs for each position
    chunks = (1, 1, 1, 512, 512)
    shards_ratio = (1, 1, 1, 32, 32)

    jobs = []
    for position_key in position_keys:
        job_name = f"cp_seg_{position_key.replace('/', '_')}"
        job_spec = {
            "name": job_name,
            "func": convert_cp_nuclear_seg_to_v3,
            "kwargs": {
                "dest_v3_path": str(dest_path),
                "cell_painting_dir": str(cell_painting_dir),
                "tracking_dir": str(tracking_dir),
                "position_key": position_key,
                "parts": args.parts,
                "chunks": chunks,
                "shards_ratio": shards_ratio,
            },
            "metadata": {
                "experiment": experiment,
                "position": position_key,
                "group": "cp_seg",
            },
        }
        jobs.append(job_spec)

    results = {"submitted": 0, "failed": 0}

    # Use lighter SLURM params since this is just writing segmentation labels
    cp_seg_slurm_params = {
        "timeout_min": 30,
        "mem": "250GB",  # Pyramid builder loads full level 0 (~41GB int32 for 104K×104K)
        "cpus_per_task": 16,
        "slurm_partition": "cpu",
    }

    if jobs:
        print(f"Submitting {len(jobs)} CP segmentation jobs with: "
              f"{cp_seg_slurm_params['timeout_min']}min, {cp_seg_slurm_params['mem']}, "
              f"{cp_seg_slurm_params['cpus_per_task']} CPUs")

        submit_result = submit_parallel_jobs(
            jobs_to_submit=jobs,
            experiment=experiment,
            slurm_params=cp_seg_slurm_params,
            log_dir="slurm_convert_v3_logs/%j",
            manifest_prefix="convert_v3_cp_seg",
            dry_run=args.dry_run,
            wait_for_completion=False,
            verbose=not args.quiet,
        )
        if submit_result.get("success"):
            results["submitted"] += len(jobs)

    # Wait for completion if not --no-wait
    if not args.no_wait and not args.dry_run:
        if jobs and submit_result and submit_result.get("success"):
            job_arrays = [{
                "submitted_jobs": submit_result.get("submitted_jobs", []),
                "base_job_id": submit_result.get("base_job_id"),
                "label": "cp_seg",
                "slurm_params": cp_seg_slurm_params,
            }]

            wait_results = wait_for_multiple_job_arrays(
                job_arrays=job_arrays,
                experiment=experiment,
                verbose=not args.quiet,
            )

            if wait_results.get("array_results"):
                for array_label, array_result in wait_results["array_results"].items():
                    results["failed"] += len(array_result.get("failed", []))

    # handle_single_experiment_cli maps exit code off these keys — without them a
    # fully-successful run (failed==0) is misread as failure (exit 1).
    results["success"] = (results.get("failed", 0) == 0)
    if args.dry_run:
        results["dry_run"] = True
    return results


def submit_cell_paint_conversion_job(experiment: str, args) -> dict:
    """Submit SLURM jobs for cell painting conversion.

    This matches the behavior of the standard pheno conversion but with:
    - 12 channels instead of 4 (pheno + cell painting)
    - Additional cell painting registration applied to channels 4-11
    - Same segmentation/label group jobs as standard pheno mode

    Supports --groups flag:
    - 'all' (default): convert base images + all label groups
    - 'base': only convert base images (pyramid levels), preserve labels
    - 'labels': only convert label groups, skip base images
    """
    from cyclops_utils.data.filesystem import decide_overwrite_resume_skip

    dataset = OpsDataset(experiment)

    # Pheno now stitches directly to v3 (no v2 pheno_assembled store), so source
    # the existing pheno-only v3 store. To avoid reading and writing the same
    # store, move it aside to <name>_without_cp.zarr and write the merged
    # 12-channel result back to the canonical _v3 name (downstream unchanged).
    final_v3 = dataset.store_paths.get("pheno_assembled_v3")
    source_path = final_v3.with_name(f"{final_v3.stem}_without_cp.zarr")
    dest_path = final_v3
    _dest_override = getattr(args, "dest_name", None)
    if source_path.exists():
        pass  # already moved aside on a previous run
    elif final_v3.exists() and not _dest_override:
        if args.dry_run:
            print(f"[dry-run] would move {final_v3.name} -> {source_path.name}")
            source_path = final_v3  # read the current store for planning only
        else:
            print(f"Moving pheno-only v3 aside: {final_v3.name} -> {source_path.name}")
            final_v3.rename(source_path)  # instant (metadata rename)
    elif final_v3.exists():
        pass  # dest-override: read canonical as source below, don't rename
    else:
        print(f"ERROR: no pheno v3 store found ({source_path.name} / {final_v3.name}). "
              f"Run the phenotyping pipeline first.")
        return {"submitted": 0, "failed": 0}

    if _dest_override:
        # A/B validation: write a NEW store adjacent; leave source + existing baseline untouched.
        if not source_path.exists():
            source_path = final_v3  # read canonical pheno-v3 as source (no rename)
        dest_path = final_v3.with_name(_dest_override)
        print(f"[dest-override] NEW output -> {dest_path.name} | source(read-only)={source_path.name} "
              f"| baseline preserved: {final_v3.name}")
    exp_path = dataset.experiment_path
    cell_painting_dir = exp_path / "0-convert" / "cell_painting"
    tracking_dir = exp_path / "2-tracking"

    # Check for cell painting data
    if not cell_painting_dir.exists():
        print(f"Cell painting directory not found: {cell_painting_dir}")
        return {"submitted": 0, "failed": 0}

    # Parse --groups flag to determine what to convert
    groups_list = getattr(args, 'groups', ['all'])
    if isinstance(groups_list, str):
        groups_list = [groups_list]
    groups_set = set(groups_list)

    include_base = "all" in groups_set or "base" in groups_set
    include_labels = "all" in groups_set or "labels" in groups_set

    print(f"\n--groups: {groups_set}")
    print(f"  Include base images: {include_base}")
    print(f"  Include labels: {include_labels}")

    # Get all position+group combinations (same as standard pheno mode)
    combinations = get_position_group_combinations(
        experiment=None,
        source_store=None,
        skip_overlays=args.skip_overlays,
        source_path=source_path,
        # v3 pheno source nests all label groups under a `labels/` container; a
        # "labels" copy job would create a spurious labels/labels. The cp store's
        # final labels (CP{N}_nuclear_seg, cp_cell_seg) come from convert + the
        # native-20x seg step, so the source's pheno labels aren't copied here.
        exclude_groups={"labels"},
    )

    if not combinations:
        print(f"No positions found for {experiment}")
        return {"submitted": 0, "failed": 0}

    # Get position keys
    position_keys = sorted(set(pos_key for pos_key, _ in combinations))

    # Optionally filter by --wells argument
    if args.wells:
        position_keys = [pk for pk in position_keys if int(pk.split("/")[1]) in args.wells]
        combinations = [(pk, grp) for pk, grp in combinations if pk in position_keys]

    # Get channel info from source
    with iohub.open_ome_zarr(source_path, mode="r") as source_plate:
        pheno_channel_names = list(source_plate.channel_names)
        first_pos = source_plate[position_keys[0]]
        src_shape = first_pos["0"].shape

    # Calculate total channels and target shape
    num_pheno_channels = len(pheno_channel_names)
    num_cp_channels = 4 * len(args.parts)
    all_channel_names = build_channel_names(pheno_channel_names, args.parts)
    num_total_channels = len(all_channel_names)
    target_shape = (src_shape[0], num_total_channels, src_shape[2], src_shape[3], src_shape[4])

    # Calculate sharding - all channels in one shard, 32x32 spatial chunks
    chunks = (1, 1, 1, 512, 512)
    shards_ratio = (1, num_total_channels, 1, 32, 32)
    shard_size = tuple(c * r for c, r in zip(chunks, shards_ratio))

    # Handle --groups base: only convert base images, preserve labels
    # This MUST set action="resume" to prevent decide_overwrite_resume_skip from deleting the store
    if dest_path.exists() and "base" in groups_set and "all" not in groups_set:
        if args.force:
            # --force with --groups base: delete pyramid levels only
            if args.dry_run:
                print(f"\n[--groups base --force] DRY RUN: Would delete pyramid levels, preserving labels...")
                for pos_key in position_keys:
                    pos_path = dest_path / pos_key
                    if pos_path.exists():
                        for item in pos_path.iterdir():
                            if item.is_dir() and item.name.isdigit():
                                print(f"  Would remove {pos_key}/{item.name} (pyramid level)")
                print("DRY RUN: Pyramid levels would be removed.\n")
            else:
                print(f"\n[--groups base --force] Only deleting pyramid levels, preserving labels...")
                for pos_key in position_keys:
                    pos_path = dest_path / pos_key
                    if pos_path.exists():
                        for item in pos_path.iterdir():
                            # Delete only numeric directories (pyramid levels: 0, 1, 2, 3, 4)
                            if item.is_dir() and item.name.isdigit():
                                print(f"  Removing {pos_key}/{item.name} (pyramid level)")
                                shutil.rmtree(item)
                print("Pyramid levels removed. Recreating empty arrays for cell painting channels...\n")

                # Recreate pyramid level arrays so tensorstore/workers can write to them
                # Use create_zarr_array directly (bypasses iohub channel count validation
                # since the plate metadata still has the old channel count)
                _shards = calculate_channel_based_shards(num_total_channels, chunks=chunks)
                with iohub.open_ome_zarr(source_path, mode="r") as _src_plate:
                    for pos_key in position_keys:
                        _src_pos = _src_plate[pos_key]
                        for level in range(5):
                            level_key = str(level)
                            try:
                                src_shape = _src_pos[level_key].shape
                            except KeyError:
                                break
                            level_target_shape = (
                                src_shape[0],
                                num_total_channels,
                                src_shape[2],
                                src_shape[3],
                                src_shape[4],
                            )
                            # Create array directly via zarr_utils (no iohub channel validation)
                            array_path = str(dest_path / pos_key / level_key)
                            create_zarr_array(
                                path=array_path,
                                shape=level_target_shape,
                                dtype=np.float32,
                                chunks=chunks,
                                zarr_format=3,
                                shards_ratio=_shards,
                                fill_value=0,
                                overwrite=True,
                            )
                        print(f"  Recreated pyramid arrays for {pos_key}")
                print("Array recreation complete.")

                # Update plate-level and position-level metadata for expanded channel count
                # (the original store may have been created with only pheno channels)
                print("  Updating metadata for expanded channel count...")

                # 1. Update OMERO metadata per position (channel labels, colors, CP info)
                for pos_key in position_keys:
                    _update_omero_metadata(
                        dest_v3_path=dest_path,
                        position_key=pos_key,
                        pheno_channel_names=pheno_channel_names,
                        parts=args.parts,
                        pheno_v2_path=source_path,
                    )

                # 2. Update plate-level channels_metadata
                from cyclops_process.convert.v3_metadata import build_channels_metadata
                channels_meta = build_channels_metadata(all_channel_names, experiment=experiment)
                # Enrich CP channels with biological annotation
                for i, name in enumerate(all_channel_names):
                    if not name.startswith("CP"):
                        continue
                    for _part, part_channels in CELL_PAINTING_CHANNELS.items():
                        for _ch_idx, ch_info in part_channels.items():
                            ch_name = ch_info["name"]
                            ch_marker = ch_info["marker"]
                            full = f"CP{_part}_{ch_name}_{ch_marker}"
                            short = f"CP{_part}_{ch_name}"
                            if name not in (full, short):
                                continue
                            _marker = ch_info.get("full_marker", ch_marker)
                            _structure = ch_info["structure"]
                            channels_meta[i]["channel_type"] = "fluorescent"
                            channels_meta[i]["biological_annotation"] = {
                                "organelle": _structure,
                                "marker": _marker,
                                "marker_type": "nuclear_dye" if ch_marker == "Hoechst" else "direct_label",
                                "full_label": f"{_structure}, {_marker}",
                            }
                            channels_meta[i]["description"] = (
                                f"Cell painting {_structure} visualized via {_marker} (Part {_part})"
                            )
                            break

                plate_json_path = dest_path / "zarr.json"
                with open(plate_json_path, "r") as f:
                    plate_meta = json.load(f)
                plate_meta.setdefault("attributes", {})["channels_metadata"] = channels_meta
                with open(plate_json_path, "w") as f:
                    json.dump(plate_meta, f, indent=2)
                print(f"  Updated plate-level channels_metadata ({len(channels_meta)} channels)\n")

        else:
            # --groups base without --force: just resume, don't delete anything
            print(f"\n[--groups base] Preserving existing labels, will overwrite base images only...")
        # CRITICAL: Always set action="resume" for --groups base to prevent store deletion
        action = "resume"
    else:
        # Check store state normally (only for --groups all or new stores)
        action = decide_overwrite_resume_skip(
            dest_path,
            is_debug=args.force,
            expected_positions=position_keys
        )

    if action == "skip":
        print(f"Conversion skipped for {experiment}")
        return {"submitted": 0, "failed": 0}

    # Initialize store if needed
    if not args.dry_run:
        if action in ("create", "overwrite"):
            print(f"\nInitializing v3 store with cell painting for {experiment}...")
            target_shape, pheno_names, all_names, shards_ratio = initialize_v3_store_with_cell_paint(
                pheno_v2_path=source_path,
                output_v3_path=dest_path,
                parts=args.parts,
                overwrite=(action == "overwrite"),
                skip_overlays=args.skip_overlays,
                experiment=experiment,
            )
            print("Initialization complete. Submitting parallel conversion jobs...\n")
        elif action == "resume":
            print(f"\nResuming conversion for {experiment}...")
    else:
        if action in ("create", "overwrite"):
            print(f"\nDRY RUN: Would initialize v3 store with cell painting for {experiment}")
            print(f"  Action: {action}")
        elif action == "resume":
            print(f"\nDRY RUN: Would resume conversion for {experiment}\n")

    # Print job summary
    print(f"\n{'='*60}")
    print(f"Cell Painting Conversion Job Summary")
    print(f"{'='*60}")
    print(f"Experiment: {experiment}")
    print(f"Source: {source_path}")
    print(f"Dest: {dest_path}")
    print(f"Positions: {len(position_keys)}")
    print(f"Parts: {args.parts}")
    print(f"Phenotyping channels: {num_pheno_channels} ({pheno_channel_names})")
    print(f"Cell painting channels: {num_cp_channels}")
    print(f"Total channels: {num_total_channels}")
    print(f"Channel names: {all_channel_names}")
    print(f"Reference shape (first position): {target_shape}")
    print(f"Chunks: {chunks}")
    print(f"Shards ratio: {shards_ratio}")
    print(f"Shard size: {shard_size}")
    print(f"{'='*60}\n")

    # Separate jobs by type (same as standard pheno mode)
    base_jobs = []
    seg_jobs = []

    for position_key, group_name in combinations:
        if group_name is None:
            # Base image job - includes cell painting conversion
            job_name = f"cp_convert_{position_key.replace('/', '_')}_base"
            job_spec = {
                "name": job_name,
                "func": convert_position_with_cell_paint,
                "kwargs": {
                    "experiment": experiment,
                    "position_key": position_key,
                    "pheno_v2_path": str(source_path),
                    "dest_v3_path": str(dest_path),
                    "cell_painting_dir": str(cell_painting_dir),
                    "tracking_dir": str(tracking_dir),
                    "parts": args.parts,
                    "pheno_channel_names": pheno_channel_names,
                    "chunks": chunks,
                    "shards_ratio": shards_ratio,
                },
                "metadata": {
                    "experiment": experiment,
                    "position": position_key,
                    "group": "base",
                },
            }
            base_jobs.append(job_spec)
        else:
            # Segmentation/label group job - same as standard pheno mode
            # Use single-channel sharding for labels (seg, nuclear_seg, etc.)
            job_name = f"cp_convert_{position_key.replace('/', '_')}_{group_name}"
            job_spec = {
                "name": job_name,
                "func": convert_position_group_to_v3,
                "kwargs": {
                    "experiment": experiment,
                    "position_key": position_key,
                    "group_name": group_name,
                    "source_path": str(source_path),
                    "dest_path": str(dest_path),
                    "chunks": chunks,
                    "shards_ratio": LABEL_SHARDS_RATIO,
                },
                "metadata": {
                    "experiment": experiment,
                    "position": position_key,
                    "group": group_name,
                },
            }
            seg_jobs.append(job_spec)

    # Add cell painting nuclear segmentation jobs (CP1_nuclear_seg, CP2_nuclear_seg)
    # These read from part{N}_max_proj_segmentation.zarr and write to labels/
    cp_nuclear_seg_jobs = []
    for position_key in position_keys:
        job_name = f"cp_convert_{position_key.replace('/', '_')}_cp_nuclear_seg"
        job_spec = {
            "name": job_name,
            "func": convert_cp_nuclear_seg_to_v3,
            "kwargs": {
                "dest_v3_path": str(dest_path),
                "cell_painting_dir": str(cell_painting_dir),
                "tracking_dir": str(tracking_dir),
                "position_key": position_key,
                "parts": args.parts,
                "source_path": str(source_path),
                "chunks": chunks,
                "shards_ratio": LABEL_SHARDS_RATIO,
            },
            "metadata": {
                "experiment": experiment,
                "position": position_key,
                "group": "cp_nuclear_seg",
            },
        }
        cp_nuclear_seg_jobs.append(job_spec)

    results = {"submitted": 0, "failed": 0}

    # Filter jobs based on --groups flag
    # - Base jobs write to pyramid levels (0/, 1/, 2/, 3/, 4/)
    # - Seg jobs write to labels/ subgroups (seg, nuclear_seg, etc.)
    # - CP nuclear seg jobs write to labels/CP1_nuclear_seg, labels/CP2_nuclear_seg
    jobs_to_submit = []
    if include_base:
        jobs_to_submit.extend(base_jobs)
    if include_labels:
        jobs_to_submit.extend(seg_jobs)
        jobs_to_submit.extend(cp_nuclear_seg_jobs)

    # For summary, show what's being submitted vs skipped
    skipped_base = len(base_jobs) if not include_base else 0
    skipped_labels = (len(seg_jobs) + len(cp_nuclear_seg_jobs)) if not include_labels else 0
    if skipped_base or skipped_labels:
        print(f"[--groups {' '.join(groups_set)}] Skipping {skipped_base} base jobs, {skipped_labels} label jobs")

    all_jobs = jobs_to_submit

    if all_jobs:
        n_base = len(base_jobs) if include_base else 0
        n_seg = len(seg_jobs) if include_labels else 0
        n_cp_seg = len(cp_nuclear_seg_jobs) if include_labels else 0
        print(f"Submitting {len(all_jobs)} jobs ({n_base} base + {n_seg} seg + {n_cp_seg} cp_nuclear_seg) with: "
              f"{CELL_PAINT_SLURM_PARAMS['timeout_min']}min, {CELL_PAINT_SLURM_PARAMS['mem']}, "
              f"{CELL_PAINT_SLURM_PARAMS['cpus_per_task']} CPUs")

        submit_result = submit_parallel_jobs(
            jobs_to_submit=all_jobs,
            experiment=experiment,
            slurm_params=CELL_PAINT_SLURM_PARAMS,
            log_dir="slurm_convert_v3_logs/%j",
            manifest_prefix="convert_v3_cell_paint",
            dry_run=args.dry_run,
            wait_for_completion=False,
            verbose=not args.quiet,
        )
        if submit_result.get("success"):
            results["submitted"] += len(all_jobs)

    # Wait for completion if not --no-wait
    if not args.no_wait and not args.dry_run:
        if all_jobs and submit_result and submit_result.get("success"):
            job_arrays = [{
                "submitted_jobs": submit_result.get("submitted_jobs", []),
                "base_job_id": submit_result.get("base_job_id"),
                "label": "cell_paint",
                "slurm_params": CELL_PAINT_SLURM_PARAMS,
            }]

            wait_results = wait_for_multiple_job_arrays(
                job_arrays=job_arrays,
                experiment=experiment,
                verbose=not args.quiet,
            )

            if wait_results.get("array_results"):
                for array_label, array_result in wait_results["array_results"].items():
                    results["failed"] += len(array_result.get("failed", []))

    # handle_single_experiment_cli maps exit code off these keys — without them a
    # fully-successful run (failed==0) is misread as failure (exit 1).
    results["success"] = (results.get("failed", 0) == 0)
    if args.dry_run:
        results["dry_run"] = True
    return results


# ============================================================================
# 4i section
# ============================================================================

sys.path.insert(0, os.getcwd())


SLURM_PARAMS = {
    "timeout_min": 360,
    "mem": "500GB",
    "cpus_per_task": 64,
    "slurm_partition": "cpu,gpu",
}


SLURM_PARAMS_PYRAMID = {
    "timeout_min": 35,
    "mem": "250GB",
    "cpus_per_task": 32,
    "slurm_partition": "cpu,gpu",
}


SLURM_PARAMS_TRANSFORM = {
    "timeout_min": 600,
    "mem": "250GB",
    "cpus_per_task": 32,
    "slurm_partition": "cpu,gpu",
}


SLURM_PARAMS_RESHARD = {
    "timeout_min": 60,
    "mem": "800GB",
    "cpus_per_task": 8,
    "slurm_partition": "cpu,gpu",
}


def build_4i_channel_names(pheno_channel_names: list, rounds: list) -> list:
    """Build complete channel name list: pheno channels + 4i channels."""
    channel_names = list(pheno_channel_names)
    for rnd in rounds:
        for ch in range(3):  # DAPI, mouse_488, rabbit_647
            channel_names.append(_fouri_channel_name(rnd, ch, format="full"))
    return channel_names


def get_4i_channel_metadata(rnd: int, channel: int) -> dict:
    """Get metadata for a 4i channel (mirrors CP get_channel_metadata)."""
    info = FOUR_I_CHANNELS.get(rnd, {}).get(channel, {})
    name = info.get("name", f"ch{channel}")
    marker = info.get("marker", "unknown")
    return {
        "name": _fouri_channel_name(rnd, channel, format="short"),
        "full_name": _fouri_channel_name(rnd, channel, format="full"),
        "marker": marker,
        "structure": info.get("structure", "unknown"),
        "full_marker": info.get("full_marker", marker),
        "round": rnd,
        "source_channel": channel,
        "color": FOUR_I_COLORS.get(marker, "FFFFFF"),
    }


def get_registration_affine(
    round_num: int,
    well,
    registration_dir: Path,
    resolution_scale: float = 4.0,
) -> np.ndarray:
    """Get the full affine transform for a 4i round to phenotyping space.

    For round 1: direct registration (round1 -> pheno)
    For rounds 2-5: chained (roundN -> round1, then round1 -> pheno)

    CRITICAL: Each registration YAML stores an affine computed at SEG resolution
    (4x downsampled). We need to:
      1. Scale each affine's translation by `resolution_scale` to full resolution FIRST
      2. Then compose them (matrix multiply)

    If we compose first and then scale, the math is wrong because composition
    mixes rotation with translation.
    """
    row, col = parse_well(well)  # accepts full unit, token, or int
    well_token = f"{row}{col}"  # row-A byte-identical to old A{well}
    # Round 1 -> pheno (scale translation to full res before anything else)
    r1_to_pheno_yaml = registration_dir / f"{well_token}_4i_round1_register.yml"
    if not r1_to_pheno_yaml.exists():
        raise FileNotFoundError(f"Missing round1->pheno registration: {r1_to_pheno_yaml}")
    r1_to_pheno = _scale_affine_translation(load_affine_from_yaml(r1_to_pheno_yaml), resolution_scale)

    if round_num == 1:
        return r1_to_pheno

    # Round N -> round 1 (scale translation to full res), then compose
    rn_to_r1_yaml = registration_dir / f"{well_token}_4i_round{round_num}_register.yml"
    if not rn_to_r1_yaml.exists():
        raise FileNotFoundError(f"Missing round{round_num}->round1 registration: {rn_to_r1_yaml}")
    rn_to_r1 = _scale_affine_translation(load_affine_from_yaml(rn_to_r1_yaml), resolution_scale)

    # Compose: the stored YAMLs are scipy-style (output→input) affines.
    #   r1_to_pheno maps pheno_coord → r1_coord
    #   rn_to_r1   maps r1_coord   → rn_coord
    # For a pheno output pixel, to find the corresponding rn input coord:
    #   rn_coord = rn_to_r1 @ (r1_to_pheno @ pheno_coord)
    #           = (rn_to_r1 @ r1_to_pheno) @ pheno_coord
    # So: composed = rn_to_r1 @ r1_to_pheno
    return rn_to_r1 @ r1_to_pheno


def _run_4i_preview(experiment: str, wells: list, rounds: list, crop_size: int = 2048):
    """Render small-crop overlay PNGs to visually confirm 4i alignment at 20x.

    For each (well, round), composes the 4i registration affine via
    `get_registration_affine` (same logic _transform_and_write_4i_round uses),
    warps that round's nuclear seg into 3 pheno-space crops at different radii,
    and saves an RGB overlay per (well, round, crop): R=pheno, G=round_warped.
    Round-pheno mask overlap printed per crop.
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import affine_transform as scipy_affine_transform
    from scipy.ndimage import zoom
    from cyclops_utils.data.filesystem import resolve_experiment_name

    experiment = resolve_experiment_name(experiment, autoselect=True)
    dataset = OpsDataset(experiment)
    four_i_dir = get_default_output_dir(experiment)
    registration_dir = four_i_dir / "registration"
    out_dir = registration_dir / "convert_preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    pheno_seg_path = dataset.store_paths.get("lc_20x_segmentation")
    if pheno_seg_path is None or not Path(pheno_seg_path).exists():
        print(f"ERROR: pheno seg not found at {pheno_seg_path}")
        return

    def _upscale_5x_to_20x_crop(arr_5x, y0_20x, y1_20x, x0_20x, x1_20x):
        """Read a small 20x crop by upscaling the corresponding 5x region (order=0)."""
        if arr_5x.ndim == 5:
            src_h_5x, src_w_5x = arr_5x.shape[-2:]
        else:
            src_h_5x, src_w_5x = arr_5x.shape
        y0_5x = max(0, y0_20x // 4 - 1)
        y1_5x = min(src_h_5x, (y1_20x + 3) // 4 + 1)
        x0_5x = max(0, x0_20x // 4 - 1)
        x1_5x = min(src_w_5x, (x1_20x + 3) // 4 + 1)
        if arr_5x.ndim == 5:
            chunk_5x = np.asarray(arr_5x[0, 0, 0, y0_5x:y1_5x, x0_5x:x1_5x], dtype=np.int32)
        else:
            chunk_5x = np.asarray(arr_5x[y0_5x:y1_5x, x0_5x:x1_5x], dtype=np.int32)
        if chunk_5x.size == 0:
            return np.zeros((y1_20x - y0_20x, x1_20x - x0_20x), dtype=np.int32)
        chunk_20x = zoom(chunk_5x, 4, order=0)
        chunk_origin_y = y0_5x * 4
        chunk_origin_x = x0_5x * 4
        oy0 = y0_20x - chunk_origin_y
        ox0 = x0_20x - chunk_origin_x
        h = y1_20x - y0_20x
        w = x1_20x - x0_20x
        out = np.zeros((h, w), dtype=np.int32)
        avail_y = min(h, max(0, chunk_20x.shape[0] - oy0))
        avail_x = min(w, max(0, chunk_20x.shape[1] - ox0))
        if avail_y > 0 and avail_x > 0 and oy0 >= 0 and ox0 >= 0:
            out[:avail_y, :avail_x] = chunk_20x[oy0:oy0 + avail_y, ox0:ox0 + avail_x]
        return out

    for well in wells:
        row, col = parse_well(well)  # row-agnostic; well may be int or "B2"
        well_token = f"{row}{col}"
        try:
            pheno_arr = zarr.open(str(Path(pheno_seg_path) / f"{row}/{col}/0/0"), mode="r")
        except Exception as e:
            print(f"  W{well_token}: pheno seg open failed: {e}")
            continue
        if pheno_arr.ndim == 5:
            pheno_h_5x, pheno_w_5x = pheno_arr.shape[-2:]
        else:
            pheno_h_5x, pheno_w_5x = pheno_arr.shape
        pheno_h, pheno_w = pheno_h_5x * 4, pheno_w_5x * 4

        cy, cx = pheno_h // 2, pheno_w // 2
        crop_centers = [
            ("center", cy, cx),
            ("mid-NW", cy - int(0.15 * pheno_h), cx - int(0.15 * pheno_w)),
            ("outer-SE", cy + int(0.25 * pheno_h), cx + int(0.25 * pheno_w)),
        ]
        print(f"  W{well_token}: pheno 20x={pheno_h}x{pheno_w}, crop_size={crop_size}, rounds={rounds}")

        crops_data = []
        for crop_label, ccy, ccx in crop_centers:
            y0 = max(0, ccy - crop_size // 2)
            y1 = min(pheno_h, y0 + crop_size)
            x0 = max(0, ccx - crop_size // 2)
            x1 = min(pheno_w, x0 + crop_size)
            ch, cw = y1 - y0, x1 - x0
            print(f"    {crop_label}: 20x=[{y0}:{y1}, {x0}:{x1}]")
            pheno_crop = _upscale_5x_to_20x_crop(pheno_arr, y0, y1, x0, x1)
            pheno_mask = (pheno_crop > 0).astype(np.uint8)
            crops_data.append({
                "label": crop_label, "y0": y0, "y1": y1, "x0": x0, "x1": x1,
                "ch": ch, "cw": cw, "pheno_mask": pheno_mask, "warped": {},
            })

        # Compose + warp per round
        for rnd in rounds:
            try:
                affine_4x4 = get_registration_affine(rnd, well, registration_dir, resolution_scale=4.0)
            except FileNotFoundError as e:
                print(f"    Round {rnd}: skipped — {e}")
                continue

            affine_3x3 = np.identity(3)
            affine_3x3[0:2, 0:2] = affine_4x4[1:3, 1:3]
            affine_3x3[0:2, 2] = affine_4x4[1:3, 3]
            print(f"    Round {rnd}: composed affine 20x — scale={affine_3x3[0,0]:.4f} "
                  f"trans=Y{affine_3x3[0,2]:+.1f},X{affine_3x3[1,2]:+.1f}")

            seg_path = four_i_dir / f"round{rnd}_max_proj_flatfield_segmentation.zarr"
            if not seg_path.exists():
                print(f"      seg missing: {seg_path}")
                continue
            cp_arr = zarr.open(str(seg_path / f"{row}/{col}/0/0"), mode="r")  # 5x

            for crop in crops_data:
                y0, y1, x0, x1 = crop["y0"], crop["y1"], crop["x0"], crop["x1"]
                ch, cw = crop["ch"], crop["cw"]

                corners_out = np.array([
                    [y0, x0, 1], [y0, x1, 1], [y1, x0, 1], [y1, x1, 1]
                ], dtype=np.float64)
                corners_in = (affine_3x3 @ corners_out.T).T[:, :2]
                pad = 50
                in_y_min = int(np.floor(corners_in[:, 0].min())) - pad
                in_y_max = int(np.ceil(corners_in[:, 0].max())) + pad
                in_x_min = int(np.floor(corners_in[:, 1].min())) - pad
                in_x_max = int(np.ceil(corners_in[:, 1].max())) + pad

                cp_chunk = _upscale_5x_to_20x_crop(cp_arr, in_y_min, in_y_max, in_x_min, in_x_max)

                chunk_aff = affine_3x3.copy()
                scale_2x2 = affine_3x3[0:2, 0:2]
                output_offset = np.array([y0, x0], dtype=np.float64)
                input_crop_offset = np.array([in_y_min, in_x_min], dtype=np.float64)
                chunk_aff[0:2, 2] = scale_2x2 @ output_offset + affine_3x3[0:2, 2] - input_crop_offset

                out = scipy_affine_transform(
                    cp_chunk.astype(np.float32),
                    chunk_aff[:2, :2],
                    offset=chunk_aff[:2, 2],
                    output_shape=(ch, cw),
                    order=0, mode='constant', cval=0,
                )
                crop["warped"][rnd] = (out > 0).astype(np.uint8)

        # One viridis composite per crop showing all rounds stacked together.
        viridis = plt.get_cmap('viridis')
        for crop in crops_data:
            pheno_mask = crop["pheno_mask"]
            tgt_sum = float(pheno_mask.sum())
            warped = crop["warped"]
            sorted_rnds = sorted(warped.keys())

            stats = []
            for rnd in sorted_rnds:
                inter = float((warped[rnd] & pheno_mask).sum())
                pct = (inter / tgt_sum * 100) if tgt_sum > 0 else 0.0
                stats.append(f"R{rnd}∩Pheno={pct:.1f}%")
            for i in range(1, len(sorted_rnds)):
                a, b = sorted_rnds[i - 1], sorted_rnds[i]
                inter = float((warped[a] & warped[b]).sum())
                base = max(1.0, float(warped[a].sum()))
                stats.append(f"R{b}∩R{a}={(inter/base*100):.1f}%")
            print(f"    [{crop['label']}] " + " | ".join(stats))

            if not sorted_rnds:
                continue

            # Per-pixel count of rounds with mask present, colored via viridis.
            # 0 rounds = black, all rounds = bright yellow. Pheno outline in red.
            n_rounds = len(sorted_rnds)
            count = np.zeros((crop["ch"], crop["cw"]), dtype=np.float32)
            for rnd in sorted_rnds:
                count = count + warped[rnd].astype(np.float32)
            norm = count / n_rounds
            rgba = viridis(norm)
            rgb = rgba[..., :3].copy()
            rgb[count == 0] = 0
            rgb[..., 0] = np.maximum(rgb[..., 0], pheno_mask * 0.9)
            rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            comp_path = out_dir / f"W{well_token}_{crop['label']}_overlay.png"
            plt.imsave(comp_path, rgb_uint8)


def _transform_and_write_4i_round(
    dest_v3_path: Path,
    four_i_store_path: Path,
    affine_4x4: np.ndarray,
    position_key: str,
    well,
    round_num: int,
    start_channel: int,
    spatial_chunk_multiplier: int = 48,
):
    """Warp 4i channels (order=1) into the pheno v3 frame via the shared engine."""
    affine_3x3 = affine_3x3_from_4x4(affine_4x4)
    print(f"      Affine 3x3 (YX): scale={affine_3x3[0,0]:.4f}, "
          f"translate=({affine_3x3[0,2]:.1f}, {affine_3x3[1,2]:.1f})")

    src_row, src_col = parse_well(well)  # row-agnostic source position
    source_store = zarr.open(str(four_i_store_path), mode="r")
    source_array = source_store[f"{src_row}/{src_col}/0/0"]
    num_src_channels = source_array.shape[1]  # DAPI, 488, 647

    dest_ts = ts.open({
        'driver': 'zarr3',
        'kvstore': {'driver': 'file', 'path': str(dest_v3_path / position_key / "0")},
    }).result()

    warp_channels_into_v3(
        source_array, dest_ts, affine_3x3,
        [(c, start_channel + c) for c in range(num_src_channels)],
        order=1, dtype=np.float32,
        spatial_chunk_multiplier=spatial_chunk_multiplier, label=f"R{round_num}",
    )


def convert_position_with_4i(
    experiment: str,
    position_key: str,
    pheno_v2_path: str,
    dest_v3_path: str,
    four_i_dir: str,
    registration_dir: str,
    rounds: list,
    pheno_channel_names: list,
    chunks: tuple = (1, 1, 1, 512, 512),
    shards_ratio: tuple = (1, 1, 1, 64, 64),
    skip_copy: bool = False,
    skip_transform: bool = False,
    skip_pyramids: bool = False,
    skip_labels: bool = False,
    add_4i_seg_labels: bool = True,
):
    """Convert a single position from phenotyping v2 to v3 with 4i channels.

    1. Copies phenotyping channels from v2 to v3
    2. For each 4i round, applies chained registration and writes channels
    3. Builds pyramid levels for all channels

    Args:
        skip_copy: Skip step 1 (pheno channel copy) — use if already done
        skip_transform: Skip step 2 (4i transform+write) — use if already done
    """
    pheno_v2_path = Path(pheno_v2_path)
    dest_v3_path = Path(dest_v3_path)
    four_i_dir = Path(four_i_dir)
    registration_dir = Path(registration_dir)

    well = position_key  # full unit; helpers derive row,col via parse_well

    print(f"\n{'='*60}")
    print(f"Converting position {position_key} with 4i")
    print(f"{'='*60}")

    num_pheno_channels = len(pheno_channel_names)
    num_4i_channels = 3 * len(rounds)
    num_total_channels = num_pheno_channels + num_4i_channels

    # Step 1: Copy phenotyping channels from v2 to v3
    if skip_copy:
        print(f"\n  Step 1: SKIPPED (--skip-copy)")
    else:
        print(f"\n  Step 1: Copying {num_pheno_channels} phenotyping channels...")
        copy_pheno_channels_to_v3(
            pheno_v2_path, dest_v3_path, position_key, num_pheno_channels,
            store_custom_metadata=False,
        )

    # Step 2: Transform and write 4i channels
    if skip_transform:
        print(f"\n  Step 2: SKIPPED (--skip-transform)")
    else:
        print(f"\n  Step 2: Writing {num_4i_channels} 4i channels ({len(rounds)} rounds)...")
        channel_offset = num_pheno_channels
        for rnd in rounds:
            stitched_store = four_i_dir / f"round{rnd}_max_proj_flatfield_stitched.zarr"
            if not stitched_store.exists():
                print(f"    WARNING: Stitched store not found for round {rnd}: {stitched_store}")
                channel_offset += 3
                continue

            try:
                affine = get_registration_affine(rnd, well, registration_dir)
            except FileNotFoundError as e:
                print(f"    WARNING: {e}")
                channel_offset += 3
                continue

            print(f"\n    Round {rnd} -> channels {channel_offset}-{channel_offset + 2}")
            _transform_and_write_4i_round(
                dest_v3_path=dest_v3_path,
                four_i_store_path=stitched_store,
                affine_4x4=affine,
                position_key=position_key,
                well=well,
                round_num=rnd,
                start_channel=channel_offset,
            )
            channel_offset += 3

    # Step 3: Build pyramids using the batch script's proven builder
    if skip_pyramids:
        print(f"\n  Step 3: SKIPPED (--skip-pyramids)")
    else:
        print(f"\n  Step 3: Building pyramids...")
        from cyclops_process.processes.pyramids.audit_fix import _build_cell_painting_pyramids

        # Re-use the CP pyramid builder — it handles any v3 store with multi-channel
        # (works for both CP and 4i since it's just dask-based multi-channel pyramid)
        result = _build_cell_painting_pyramids(
            experiment=experiment,
            channel_start=0,
            channel_end=num_total_channels,
            num_levels=5,
            resume=False,
        )
        print(f"    {result}")

    # Step 3.5: Write NGFF multiscales + omero metadata (makes store iohub-readable)
    if not skip_pyramids:
        _update_4i_ngff_metadata(
            dest_v3_path=dest_v3_path,
            position_key=position_key,
            pheno_channel_names=pheno_channel_names,
            rounds=rounds,
            pheno_v2_path=pheno_v2_path,
        )

    # Step 4: Copy labels group from existing v3 store (cell_seg, nuclear_seg, etc.)
    if skip_labels:
        print(f"\n  Step 4: SKIPPED (--skip-labels)")
    else:
        _copy_existing_labels(dest_v3_path, position_key)
        if add_4i_seg_labels:
            _add_4i_nuclear_seg_labels(
                dest_v3_path=dest_v3_path,
                four_i_dir=four_i_dir,
                registration_dir=registration_dir,
                position_key=position_key,
                well=well,
                rounds=rounds,
            )

    print(f"\n  Done: {position_key}")


def build_seg_pyramid_unit_worker(
    source_store: str,
    position: str,
    seg_name: str,
    levels: int = 5,
    resume: bool = False,
):
    """SLURM worker: build seg pyramids for ONE (position, seg_label) pair.

    Uses build_seg_pyramid_only which auto-creates level arrays with correct
    sharding (default LABEL_SHARDS_RATIO=(1,1,1,32,32)) and chunks.
    """
    from cyclops_process.processes.pyramids.build_dask import build_seg_pyramid_only
    print(f"[{position} {seg_name}] Building seg pyramids (levels 1..{levels-1})", flush=True)
    build_seg_pyramid_only(
        source_store=Path(source_store),
        levels=levels,
        positions=[position],
        resume=resume,
        seg_types=[seg_name],
    )
    print(f"[{position} {seg_name}] Done", flush=True)


def transform_all_rounds_for_position_worker(
    position_key: str,
    rounds: list,
    dest_v3_path: str,
    four_i_dir: str,
    registration_dir: str,
    start_channel: int,
):
    """SLURM worker: transform+write ALL rounds for one position (serialized).

    Writes each round sequentially into the same shared shard file. This avoids
    concurrent-write conflicts on level-0 sharded zarr arrays (v3 sharding is
    not safe for multiple writers to the same shard).
    """
    dest_v3_path = Path(dest_v3_path)
    four_i_dir = Path(four_i_dir)
    registration_dir = Path(registration_dir)
    well = position_key  # full unit; helpers derive row,col via parse_well

    for i, round_num in enumerate(rounds):
        ch = start_channel + i * 3
        stitched_store = four_i_dir / f"round{round_num}_max_proj_flatfield_stitched.zarr"
        if not stitched_store.exists():
            print(f"[{position_key} R{round_num}] WARNING: stitched store not found: {stitched_store}")
            continue
        try:
            affine = get_registration_affine(round_num, well, registration_dir)
        except FileNotFoundError as e:
            print(f"[{position_key} R{round_num}] WARNING: {e}")
            continue

        print(f"[{position_key} R{round_num}] Transforming -> channels {ch}-{ch + 2}", flush=True)
        _transform_and_write_4i_round(
            dest_v3_path=dest_v3_path,
            four_i_store_path=stitched_store,
            affine_4x4=affine,
            position_key=position_key,
            well=well,
            round_num=round_num,
            start_channel=ch,
        )
    print(f"[{position_key}] All rounds done", flush=True)


def transform_position_round_worker(
    position_key: str,
    round_num: int,
    dest_v3_path: str,
    four_i_dir: str,
    registration_dir: str,
    start_channel: int,
):
    """SLURM worker: transform+write one (position, round) of 4i channels.

    Applies the registration affine and writes 3 channels to the v3 store.
    """
    dest_v3_path = Path(dest_v3_path)
    four_i_dir = Path(four_i_dir)
    registration_dir = Path(registration_dir)
    well = position_key  # full unit; helpers derive row,col via parse_well

    stitched_store = four_i_dir / f"round{round_num}_max_proj_flatfield_stitched.zarr"
    if not stitched_store.exists():
        print(f"[{position_key} R{round_num}] ERROR: stitched store not found: {stitched_store}")
        return

    affine = get_registration_affine(round_num, well, registration_dir)
    print(f"[{position_key} R{round_num}] Transforming -> channels {start_channel}-{start_channel + 2}")
    _transform_and_write_4i_round(
        dest_v3_path=dest_v3_path,
        four_i_store_path=stitched_store,
        affine_4x4=affine,
        position_key=position_key,
        well=well,
        round_num=round_num,
        start_channel=start_channel,
    )
    print(f"[{position_key} R{round_num}] Done")


def copy_pheno_channels_worker(
    position_key: str,
    pheno_v2_path: str,
    dest_v3_path: str,
    num_pheno_channels: int,
):
    """SLURM worker: copy pheno channels from v2 to v3 for one position."""
    pheno_v2_path = Path(pheno_v2_path)
    dest_v3_path = Path(dest_v3_path)

    print(f"[{position_key}] Copying {num_pheno_channels} pheno channels")
    copy_pheno_channels_to_v3(
        pheno_v2_path, dest_v3_path, position_key, num_pheno_channels,
        store_custom_metadata=False,
    )
    print(f"[{position_key}] Pheno copy done")


def _write_plate_root_metadata(
    dest_v3_path: Path,
    position_keys: list,
    all_channel_names: list,
):
    """Write root-level + row-level + well-level zarr.json files for HCS plate."""
    # Parse positions into rows/columns/wells
    rows = sorted(set(p.split("/")[0] for p in position_keys))
    cols = sorted(set(p.split("/")[1] for p in position_keys), key=lambda x: int(x) if x.isdigit() else x)
    wells = sorted(set(f"{p.split('/')[0]}/{p.split('/')[1]}" for p in position_keys))

    plate = {
        "version": "0.5",
        "acquisitions": [{"id": 0}],
        "rows": [{"name": r} for r in rows],
        "columns": [{"name": c} for c in cols],
        "wells": [
            {
                "path": w,
                "rowIndex": rows.index(w.split("/")[0]),
                "columnIndex": cols.index(w.split("/")[1]),
            }
            for w in wells
        ],
    }

    # Build channels_metadata using the shared helper, then enrich 4i channels
    # with biological annotation (same pattern CP uses in convert_v3_slurm_cp.py)
    from cyclops_process.convert.v3_metadata import build_channels_metadata
    channels_metadata = build_channels_metadata(all_channel_names)

    # Enrich 4i channels with biological annotation (mirrors CP enrichment logic)
    for i, name in enumerate(all_channel_names):
        if not name.startswith("4i_R"):
            continue
        for rnd in ROUNDS.keys():
            matched = False
            for ch_idx in range(3):
                m = get_4i_channel_metadata(rnd, ch_idx)
                if m["full_name"] != name:
                    continue
                matched = True
                structure = m["structure"]
                marker = m["full_marker"]
                marker_type = "nuclear_dye" if m["marker"] == "DAPI" else "antibody"
                channels_metadata[i]["channel_type"] = "fluorescent"
                channels_metadata[i]["biological_annotation"] = {
                    "organelle": structure,
                    "marker": marker,
                    "marker_type": marker_type,
                    "full_label": f"{structure}, {marker}",
                }
                channels_metadata[i]["description"] = (
                    f"4i Round {rnd}: {structure} visualized via {marker}"
                )
                channels_metadata[i]["four_i"] = {
                    "round": rnd,
                    "source_channel": ch_idx,
                    "marker": m["marker"],
                    "full_marker": m["full_marker"],
                    "structure": structure,
                }
                break
            if matched:
                break

    # Root zarr.json (order matches existing v3 format: attributes first)
    root_json = {
        "attributes": {
            "ome": {"plate": plate, "version": "0.5"},
            "channels_metadata": channels_metadata,
        },
        "zarr_format": 3,
        "node_type": "group",
    }
    with open(dest_v3_path / "zarr.json", "w") as f:
        json.dump(root_json, f, indent=2)
    print(f"  Wrote plate root zarr.json ({len(wells)} wells, {len(all_channel_names)} channels)")

    # Row-level zarr.json
    for r in rows:
        row_dir = dest_v3_path / r
        row_dir.mkdir(parents=True, exist_ok=True)
        with open(row_dir / "zarr.json", "w") as f:
            json.dump({"attributes": {}, "zarr_format": 3, "node_type": "group"}, f, indent=2)

    # Well-level zarr.json (fov listing)
    for pos in position_keys:
        row, col, fov = pos.split("/")
        well_dir = dest_v3_path / row / col
        well_dir.mkdir(parents=True, exist_ok=True)
        with open(well_dir / "zarr.json", "w") as f:
            json.dump({
                "attributes": {
                    "ome": {
                        "well": {
                            "version": "0.5",
                            "images": [{"acquisition": 0, "path": fov}],
                        },
                        "version": "0.5",
                    },
                },
                "zarr_format": 3,
                "node_type": "group",
            }, f, indent=2)
    print(f"  Wrote row + well-level zarr.json")


def _update_4i_ngff_metadata(
    dest_v3_path: Path,
    position_key: str,
    pheno_channel_names: list,
    rounds: list,
    pheno_v2_path: Path = None,
    num_levels: int = 5,
):
    """Write NGFF multiscales + omero metadata to position-level zarr.json.

    Makes the store readable by iohub/napari/ome-ngff tools.
    """
    pos_path = dest_v3_path / position_key
    pos_json_path = pos_path / "zarr.json"

    # Read existing metadata (created by create_zarr_array)
    if pos_json_path.exists():
        with open(pos_json_path, "r") as f:
            pos_meta = json.load(f)
    else:
        pos_meta = {"zarr_format": 3, "node_type": "group", "attributes": {}}

    # Load source omero channels if available (preserves v2 pheno metadata)
    src_omero_channels = []
    src_custom_metadata = {}
    if pheno_v2_path is not None:
        src_pos_zattrs = pheno_v2_path / position_key / ".zattrs"
        if src_pos_zattrs.exists():
            with open(src_pos_zattrs) as f:
                src_attrs = json.load(f)
            src_omero = src_attrs.get("omero", {})
            src_omero_channels = src_omero.get("channels", [])
            src_custom_metadata = src_attrs.get("custom_metadata", {})

    # Build channel list: pheno channels first, then 4i rounds
    channels = []
    for i, name in enumerate(pheno_channel_names):
        if i < len(src_omero_channels):
            channels.append(src_omero_channels[i])
        else:
            channels.append({
                "active": True, "coefficient": 1.0, "color": "FFFFFF",
                "family": "linear", "inverted": False, "label": name,
                "window": {"start": 0.0, "end": 65535.0, "min": 0.0, "max": 65535.0},
            })

    for rnd in rounds:
        for ch in range(3):
            meta = get_4i_channel_metadata(rnd, ch)
            channels.append({
                "active": True, "coefficient": 1.0, "color": meta["color"],
                "family": "linear", "inverted": False, "label": meta["full_name"],
                "window": {"start": 0.0, "end": 65535.0, "min": 0.0, "max": 65535.0},
                "four_i": {
                    "round": rnd,
                    "source_channel": ch,
                    "marker": meta["marker"],
                    "structure": meta["structure"],
                },
            })

    # Datasets list for multiscales (one per pyramid level)
    datasets_list = []
    for lvl in range(num_levels):
        level_dir = pos_path / str(lvl)
        if not level_dir.exists():
            break
        scale_factor = 2 ** lvl
        datasets_list.append({
            "path": str(lvl),
            "coordinateTransformations": [{
                "type": "scale",
                "scale": [1.0, 1.0, 1.0, 0.325 * scale_factor, 0.325 * scale_factor],
            }],
        })

    # NGFF multiscales metadata
    multiscales = [{
        "version": "0.5",
        "name": "0",
        "axes": [
            {"name": "t", "type": "time"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        "datasets": datasets_list,
    }]

    omero = {
        "id": 0,
        "name": "0",
        "version": "0.5",
        "channels": channels,
        "rdefs": {"defaultT": 0, "defaultZ": 0, "model": "color", "projection": "normal"},
    }

    if "attributes" not in pos_meta:
        pos_meta["attributes"] = {}
    if "ome" not in pos_meta["attributes"]:
        pos_meta["attributes"]["ome"] = {}

    pos_meta["attributes"]["ome"]["multiscales"] = multiscales
    pos_meta["attributes"]["ome"]["omero"] = omero
    # Also write at top-level attributes for iohub compatibility
    pos_meta["attributes"]["multiscales"] = multiscales
    pos_meta["attributes"]["omero"] = omero

    if src_custom_metadata:
        pos_meta["attributes"]["custom_metadata"] = src_custom_metadata

    with open(pos_json_path, "w") as f:
        json.dump(pos_meta, f, indent=2)

    print(f"    Wrote NGFF metadata ({len(channels)} channels, {len(datasets_list)} levels)")


_OVERLAY_LABEL_NAMES = {
    "grid_edges", "grid_props", "grid_overlay",
    "iss_points", "iss_points_props", "iss_gene_image", "iss_guide_image",
}


def copy_v3_label_groups(source_v3_path, dest_v3_path, position_key: str, skip_names=()):
    """Copy label groups from one v3 store's ``labels/`` into another's, per position.

    Both stores are already v3 and share the same coordinate space, so each label
    group (e.g. nuclear_seg, cell_seg, iss overlays) is a direct ``copytree`` — no
    v2->v3 conversion. Individual label groups already present in the destination
    (freshly-written CP/4i labels) and any in ``skip_names`` are left untouched.
    Returns the list of copied group names.
    """
    import shutil
    src_labels = Path(source_v3_path) / position_key / "labels"
    dst_labels = Path(dest_v3_path) / position_key / "labels"
    if not src_labels.exists():
        print(f"    no source labels at {src_labels}")
        return []
    dst_labels.mkdir(parents=True, exist_ok=True)
    copied = []
    for src_group in sorted(p for p in src_labels.iterdir() if p.is_dir()):
        if src_group.name in skip_names or (dst_labels / src_group.name).exists():
            continue
        shutil.copytree(src_group, dst_labels / src_group.name)
        copied.append(src_group.name)
    if copied:
        print(f"    copied {len(copied)} label group(s) from {Path(source_v3_path).name}: {copied}")
    return copied


def finalize_labels_attrs(dest_v3_path, position_key: str):
    """Set the position's ``labels`` group OME attrs to match the groups on disk.

    Scans the labels/ directory (authoritative) so newly copied/written label
    groups are listed. Overlays are excluded from ``ome.labels`` (they are not
    segmentations) but kept in the plain ``labels`` list.
    """
    dst_labels = Path(dest_v3_path) / position_key / "labels"
    if not dst_labels.exists():
        return
    names = sorted(p.name for p in dst_labels.iterdir() if p.is_dir())
    z = zarr.open(str(dest_v3_path), mode="r+")
    lg = z[f"{position_key}/labels"]
    ome_names = [n for n in names if n not in _OVERLAY_LABEL_NAMES]
    attrs = dict(lg.attrs)
    attrs["labels"] = names
    attrs["ome"] = {"version": "0.5", "labels": ome_names}
    lg.attrs.update(attrs)
    print(f"    labels attrs: {len(names)} groups ({len(ome_names)} ome labels)")


def _copy_existing_labels(dest_v3_path: Path, position_key: str, source_v3_path=None):
    """Copy the labels/ subtree from an existing v3 store into a new one (per position).

    Defaults to the canonical phenotyping_v3.zarr (4i flow); pass ``source_v3_path``
    to copy from a different store (e.g. the cp flow's pheno-only _without_cp source).
    """
    existing_v3 = Path(source_v3_path) if source_v3_path else dest_v3_path.parent / "phenotyping_v3.zarr"
    copy_v3_label_groups(existing_v3, dest_v3_path, position_key)


def _upscale_4i_seg_if_needed(seg_store_path: Path, well) -> Path:
    """Upscale 4i nuclear segmentation from 4x downsampled to full resolution."""
    import time
    row, col = parse_well(well)  # row-agnostic source position
    upscaled = seg_store_path / f"{row}/{col}/0" / "20x_nuclear_seg"
    if upscaled.exists():
        print(f"      [upscale] using cached {upscaled.name}", flush=True)
        return upscaled

    from scipy.ndimage import zoom
    t0 = time.time()
    seg_4x = zarr.open(f"{seg_store_path}/{row}/{col}/0/0", mode="r")
    upscaled_shape = (seg_4x.shape[0], seg_4x.shape[1], seg_4x.shape[2],
                      seg_4x.shape[3] * 4, seg_4x.shape[4] * 4)
    print(f"      [upscale] 4x -> shape {upscaled_shape[-2:]}", flush=True)

    create_zarr_array(
        path=str(upscaled),
        shape=upscaled_shape,
        dtype=np.int32,
        chunks=(1, 1, 1, 2048, 2048),
        zarr_format=2,
        fill_value=0,
        overwrite=True,
    )
    upscaled_arr = zarr.open(str(upscaled), mode="r+")
    print(f"      [upscale] loading {seg_4x.shape[-2:]} data into memory...", flush=True)
    data_4x = np.asarray(seg_4x[0, 0, 0])
    print(f"      [upscale] zooming 4x (order=0)...", flush=True)
    data_full = zoom(data_4x, 4, order=0)
    upscaled_arr[0, 0, 0] = data_full
    return upscaled


def _add_4i_nuclear_seg_labels(
    dest_v3_path: Path,
    four_i_dir: Path,
    registration_dir: Path,
    position_key: str,
    well,
    rounds: list,
):
    """Add 4i nuclear segmentations as label groups with pyramids.

    For each round:
    1. Upscale segmentation from 4x to full resolution (if needed)
    2. Apply registration affine
    3. Write to labels/4i_R<n>_nuclear_seg
    4. Build pyramids
    """
    from scipy.ndimage import affine_transform as scipy_affine_transform

    labels_dir = dest_v3_path / position_key / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Adding 4i nuclear seg labels with pyramids...")

    for rnd in rounds:
        seg_store = four_i_dir / f"round{rnd}_max_proj_flatfield_segmentation.zarr"
        if not seg_store.exists():
            print(f"    WARNING: No segmentation for round {rnd}: {seg_store}")
            continue

        try:
            affine = get_registration_affine(rnd, well, registration_dir)
        except FileNotFoundError as e:
            print(f"    WARNING: {e}")
            continue

        label_name = f"4i_R{rnd}_nuclear_seg"
        dst_label_path = labels_dir / label_name
        if dst_label_path.exists():
            print(f"    {label_name}: already exists, skipping")
            continue

        print(f"    {label_name}: upscaling + transforming...")
        upscaled_path = _upscale_4i_seg_if_needed(seg_store, well)

        affine_3x3 = np.identity(3)
        affine_3x3[0:2, 0:2] = affine[1:3, 1:3]
        affine_3x3[0:2, 2] = affine[1:3, 3]

        source_array = zarr.open(str(upscaled_path), mode="r")
        src_h, src_w = source_array.shape[-2:]

        dest_ts = ts.open({
            'driver': 'zarr3',
            'kvstore': {'driver': 'file', 'path': str(dest_v3_path / position_key / "0")},
        }).result()
        target_h, target_w = dest_ts.shape[-2:]

        label_arr_path = str(dst_label_path / "0")
        create_zarr_array(
            path=label_arr_path,
            shape=(1, 1, 1, target_h, target_w),
            dtype=np.int32,
            chunks=(1, 1, 1, 512, 512),
            zarr_format=3,
            shards_ratio=(1, 1, 1, 32, 32),
            fill_value=0,
            overwrite=True,
        )

        label_ts = ts.open({
            'driver': 'zarr3',
            'kvstore': {'driver': 'file', 'path': label_arr_path},
        }).result()

        chunk_size = 512 * 32
        pad = 10
        for y0 in range(0, target_h, chunk_size):
            y1 = min(y0 + chunk_size, target_h)
            for x0 in range(0, target_w, chunk_size):
                x1 = min(x0 + chunk_size, target_w)
                corners = np.array([[y0, x0, 1], [y0, x1, 1], [y1, x0, 1], [y1, x1, 1]], dtype=np.float64)
                in_corners = (affine_3x3 @ corners.T).T[:, :2]
                iy0 = max(0, int(np.floor(in_corners[:, 0].min())) - pad)
                iy1 = min(src_h, int(np.ceil(in_corners[:, 0].max())) + pad)
                ix0 = max(0, int(np.floor(in_corners[:, 1].min())) - pad)
                ix1 = min(src_w, int(np.ceil(in_corners[:, 1].max())) + pad)

                if iy1 <= iy0 or ix1 <= ix0:
                    continue

                chunk_aff = affine_3x3.copy()
                chunk_aff[0:2, 2] = affine_3x3[0:2, :2] @ np.array([y0, x0]) + affine_3x3[0:2, 2] - np.array([iy0, ix0])

                src_chunk = np.asarray(source_array[0, 0, 0, iy0:iy1, ix0:ix1], dtype=np.int32)
                if src_chunk.size == 0:
                    continue

                out_chunk = scipy_affine_transform(
                    src_chunk, chunk_aff[:2, :2], offset=chunk_aff[:2, 2],
                    output_shape=(y1 - y0, x1 - x0), order=0, mode='constant', cval=0,
                )
                label_ts[0, 0, 0, y0:y1, x0:x1].write(out_chunk.astype(np.int32)).result()

        print(f"    {label_name}: wrote level 0 (pyramids built separately via --seg-pyramids-only)", flush=True)


def initialize_v3_store_with_4i(
    pheno_v2_path: Path,
    output_v3_path: Path,
    rounds: list,
    overwrite: bool = False,
    experiment: str = None,
) -> tuple:
    """Initialize a v3 zarr store with expanded channel count for 4i."""
    print(f"\nInitializing v3 store with 4i channels at {output_v3_path}")

    if output_v3_path.exists():
        if overwrite:
            shutil.rmtree(output_v3_path)
        else:
            raise FileExistsError(f"Output exists: {output_v3_path}. Use --force to overwrite.")

    with iohub.open_ome_zarr(pheno_v2_path, mode="r") as pheno_plate:
        pheno_channel_names = list(pheno_plate.channel_names)
        num_pheno_channels = len(pheno_channel_names)
        position_keys = [pos_key for pos_key, _ in pheno_plate.positions()]

        # Get shapes from first position
        first_pos = pheno_plate[position_keys[0]]
        pheno_shape = first_pos["0"].shape
        num_levels = sum(1 for k in range(10) if str(k) in dict(first_pos.images()))

    all_channel_names = build_4i_channel_names(pheno_channel_names, rounds)
    num_total_channels = len(all_channel_names)
    target_shape = (pheno_shape[0], num_total_channels, pheno_shape[2], pheno_shape[3], pheno_shape[4])

    chunks = (1, 1, 1, 512, 512)
    shards_ratio = calculate_channel_based_shards(num_total_channels, chunks=chunks)

    print(f"  Pheno channels: {num_pheno_channels}")
    print(f"  4i channels: {3 * len(rounds)} ({len(rounds)} rounds x 3)")
    print(f"  Total channels: {num_total_channels}")
    print(f"  Channel names: {all_channel_names}")
    print(f"  Target shape: {target_shape}")
    print(f"  Positions: {len(position_keys)}")

    # Create zarr v3 arrays for each position and level
    for pos_key in position_keys:
        with iohub.open_ome_zarr(pheno_v2_path, mode="r") as src:
            src_pos = src[pos_key]
            for level in range(num_levels):
                level_key = str(level)
                try:
                    src_shape = src_pos[level_key].shape
                except KeyError:
                    break

                level_target_shape = (
                    src_shape[0],
                    num_total_channels,
                    src_shape[2],
                    src_shape[3],
                    src_shape[4],
                )
                array_path = str(output_v3_path / pos_key / level_key)
                create_zarr_array(
                    path=array_path,
                    shape=level_target_shape,
                    dtype=np.float32,
                    chunks=chunks,
                    zarr_format=3,
                    shards_ratio=shards_ratio,
                    fill_value=0,
                    overwrite=True,
                )

    print(f"  Created {len(position_keys)} positions x {num_levels} levels")

    # Write HCS plate metadata (root + row + well level zarr.json files)
    _write_plate_root_metadata(
        dest_v3_path=output_v3_path,
        position_keys=position_keys,
        all_channel_names=all_channel_names,
    )
    return target_shape, pheno_channel_names, all_channel_names, shards_ratio


def _run_full_pipeline(experiment: str, dry_run: bool, dest_name: str = None):
    """Orchestrate all stages end-to-end. Stage 1 serial, Tracks A+B parallel.

    dest_name (optional) writes to an adjacent store and is threaded to every stage;
    all stages honor it (they take dest_v3_path/source_store), so the canonical store
    is never touched.

    Dependency graph:
        Stage 1 (init+copy pheno)
            |
            +-- Track A: Stage 2 (transforms) -> Stage 3 (base pyramids)
            +-- Track B: Stage 4 (labels+seg)  -> Stage 5 (seg pyramids)
    """
    import subprocess
    import time
    from threading import Thread

    def run_stage(label: str, flags: list) -> int:
        cmd = ["python", "-m", "cyclops_process.convert.v3_fixed",
               "--experiment", experiment] + flags
        if dest_name:
            cmd += ["--dest-name", dest_name]
        if dry_run:
            cmd.append("--dry-run")
        print(f"\n{'='*80}\n  {label}\n{'='*80}\n  {' '.join(cmd)}\n")
        t0 = time.time()
        rc = subprocess.run(cmd).returncode
        print(f"\n  {label} finished in {(time.time()-t0)/60:.1f} min "
              f"({'OK' if rc == 0 else f'FAILED {rc}'})")
        return rc

    def run_track(name: str, stages: list, result: list):
        for label, flags in stages:
            rc = run_stage(f"[{name}] {label}", flags)
            if rc != 0:
                result.append(rc)
                print(f"\n!!! {name} stopped: {label} failed !!!")
                return
        result.append(0)

    print(f"\n{'#'*80}\n#  Full 4i V3 Pipeline: {experiment}  (dry_run={dry_run})\n{'#'*80}")
    t_total = time.time()

    # Stage 1 — must complete before tracks can start
    rc = run_stage("Stage 1: init + copy pheno", ["--force", "--copy-only"])
    if rc != 0:
        print("\nStage 1 failed — aborting.")
        sys.exit(1)

    # Tracks A and B in parallel
    track_a_result, track_b_result = [], []
    track_a = Thread(target=run_track, args=(
        "Track A", [
            ("Stage 2: transform 4i",  ["--transforms-only"]),
            ("Stage 3: base pyramids", ["--pyramids-only"]),
        ], track_a_result))
    track_b = Thread(target=run_track, args=(
        "Track B", [
            ("Stage 4: labels + 4i seg", ["--labels-only"]),
            ("Stage 5: 4i seg pyramids", ["--seg-pyramids-only"]),
        ], track_b_result))
    track_a.start(); track_b.start()
    track_a.join(); track_b.join()

    rc_a = track_a_result[0] if track_a_result else 1
    rc_b = track_b_result[0] if track_b_result else 1
    print(f"\n{'#'*80}")
    print(f"#  Pipeline finished in {(time.time()-t_total)/60:.1f} min")
    print(f"#    Track A: {'OK' if rc_a == 0 else 'FAILED'}")
    print(f"#    Track B: {'OK' if rc_b == 0 else 'FAILED'}")
    print(f"{'#'*80}\n")
    sys.exit(rc_a or rc_b)


def _build_base_reshard_jobs(
    position_keys: list,
    dest_path: Path,
    num_levels: int,
    experiment: str,
) -> list:
    """Build reshard jobs for base image pyramid levels 1..num_levels-1 per position."""
    from cyclops_process.processes.pyramids.workers import reshard_level_worker

    jobs = []
    for pos in position_keys:
        pos_label = pos.replace("/", "_")
        for level in range(1, num_levels):
            level_path = dest_path / pos / str(level)
            if not level_path.exists():
                print(f"SKIP reshard: {pos} L{level} (level dir missing)")
                continue
            jobs.append({
                "name": f"4i_reshard_{pos_label}_L{level}",
                "func": reshard_level_worker,
                "kwargs": {
                    "experiment": experiment,
                    "position": pos,
                    "level": level,
                    "source_store": str(dest_path),
                },
                "metadata": {"position": pos, "level": level, "type": "reshard"},
            })
    return jobs


def main():
    parser = argparse.ArgumentParser(
        description="Add 4i channels to phenotyping v3 zarr store",
    )
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--rounds", nargs="+", type=int, default=list(range(1, NUM_ROUNDS + 1)))
    parser.add_argument("--wells", nargs="+", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dest-name", default=None,
                        help="Write 4i output to this adjacent store name (e.g. "
                             "phenotyping_v3_4i_rerun.zarr) instead of the canonical "
                             "phenotyping_v3.zarr. Reads the pheno-only source read-only and "
                             "NEVER renames/touches the canonical store — safe A/B validation.")
    parser.add_argument("--skip-copy", action="store_true",
                        help="Skip step 1 (pheno channel copy) - use when resuming")
    parser.add_argument("--skip-transform", action="store_true",
                        help="Skip step 2 (4i channel transform+write) - use when resuming")
    parser.add_argument("--skip-pyramids", action="store_true",
                        help="Skip step 3 (pyramid build)")
    parser.add_argument("--skip-labels", action="store_true",
                        help="Skip step 4 (copy labels + add 4i seg labels)")
    parser.add_argument("--no-4i-seg-labels", action="store_true",
                        help="Don't add 4i nuclear seg as label groups (still copies existing labels)")
    parser.add_argument("--pyramids-only", action="store_true",
                        help="Only build base image pyramids (skip steps 1,2,4, skip init)")
    parser.add_argument("--labels-only", action="store_true",
                        help="Only do step 4 (copy existing labels + add 4i seg labels). Skip init/copy/transform/pyramids.")
    parser.add_argument("--seg-pyramids-only", action="store_true",
                        help="Only build pyramids for 4i_R<n>_nuclear_seg labels (per-unit SLURM jobs).")
    parser.add_argument("--reshard-base", action="store_true",
                        help="Only reshard base image pyramid levels 1..N-1 (channel-based sharding). "
                             "Use after --pyramids-only if reshard was skipped or failed.")
    parser.add_argument("--full", action="store_true",
                        help="Run the full pipeline end-to-end. Stage 1 (init+copy) runs first, "
                             "then Track A (transforms→pyramids) and Track B (labels→seg pyramids) "
                             "run in parallel.")
    parser.add_argument("--copy-only", action="store_true",
                        help="Only copy pheno channels (per-position jobs)")
    parser.add_argument("--transforms-only", action="store_true",
                        help="Only run 4i transforms as per-(position, round) jobs")
    parser.add_argument("--no-init", action="store_true",
                        help="Skip store initialization (use existing store)")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Only rewrite NGFF + plate-root metadata (omero channels, "
                             "multiscales, biological_annotation). Fast local op — no SLURM, "
                             "no image data touched. Use after swapping labels in four_i_config.")
    parser.add_argument("--preview", action="store_true",
                        help="Render small-crop overlay PNGs to confirm 4i alignment at 20x "
                             "(no SLURM, no v3 writes). For each well, 3 crops at different "
                             "radii × N rounds. Each PNG: R=pheno, G=round_warped. "
                             "Output: registration/convert_preview/W{well}_R{rnd}_{crop}.png.")
    parser.add_argument("--preview-crop-size", type=int, default=2048,
                        help="Edge length of preview crop in 20x pixels (default: 2048).")

    args = parser.parse_args()

    if args.preview:
        _run_4i_preview(
            experiment=args.experiment,
            wells=args.wells or [1, 2, 3],
            rounds=args.rounds,
            crop_size=args.preview_crop_size,
        )
        return

    # --full: orchestrate all stages. Stage 1 serial, then Tracks A+B in parallel.
    if args.full:
        _run_full_pipeline(args.experiment, args.dry_run, getattr(args, "dest_name", None))
        return

    dataset = OpsDataset(args.experiment)
    # Pheno stitches directly to v3 (no v2 pheno_assembled store), so source the
    # pheno-only v3 store. Mirror the cp flow: move the pheno-only store aside to
    # <name>_without_4i.zarr and write the merged result back to the canonical name.
    final_v3 = dataset.store_paths.get("pheno_assembled_v3")
    source_path = final_v3.with_name(f"{final_v3.stem}_without_4i.zarr")
    _dest_override = getattr(args, "dest_name", None)
    if source_path.exists():
        pass  # pheno-only already moved aside on a previous run
    elif final_v3.exists() and not _dest_override:
        if args.dry_run:
            print(f"[dry-run] would move {final_v3.name} -> {source_path.name}")
            source_path = final_v3
        else:
            print(f"Moving pheno-only v3 aside: {final_v3.name} -> {source_path.name}")
            final_v3.rename(source_path)
    elif final_v3.exists():
        source_path = final_v3  # dest-override: read canonical as source, NEVER rename it
    else:
        print(f"ERROR: no pheno v3 store found ({source_path.name} / {final_v3.name}).")
        return
    dest_path = final_v3.with_name(_dest_override) if _dest_override else final_v3
    if _dest_override:
        print(f"[dest-override] NEW 4i output -> {dest_path.name} | source(read-only)={source_path.name} "
              f"| canonical {final_v3.name} untouched")
    four_i_dir = get_default_output_dir(args.experiment)
    registration_dir = four_i_dir / "registration"

    if not source_path.exists():
        print(f"Phenotyping v3 source not found: {source_path}")
        return

    # Get position keys
    with iohub.open_ome_zarr(source_path, mode="r") as source_plate:
        pheno_channel_names = list(source_plate.channel_names)
        position_keys = sorted([pk for pk, _ in source_plate.positions()])

    if args.wells:
        position_keys = [pk for pk in position_keys if int(pk.split("/")[1]) in args.wells]

    all_channel_names = build_4i_channel_names(pheno_channel_names, args.rounds)
    num_total_channels = len(all_channel_names)

    print(f"\n{'='*60}")
    print(f"4i V3 Conversion")
    print(f"{'='*60}")
    print(f"Experiment: {args.experiment}")
    print(f"Source: {source_path}")
    print(f"Dest: {dest_path}")
    print(f"4i dir: {four_i_dir}")
    print(f"Registration dir: {registration_dir}")
    print(f"Positions: {len(position_keys)}")
    print(f"Rounds: {args.rounds}")
    print(f"Pheno channels: {len(pheno_channel_names)}")
    print(f"4i channels: {3 * len(args.rounds)}")
    print(f"Total channels: {num_total_channels}")
    print(f"{'='*60}\n")

    # --metadata-only: rewrite omero channel labels + multiscales in zarr.json
    # (fast local op — no SLURM, no image writes). Use after swapping marker
    # names in four_i_config.ROUNDS.
    if args.metadata_only:
        print("[metadata-only] Rewriting NGFF + root metadata...")
        _write_plate_root_metadata(dest_path, position_keys, all_channel_names)
        for pk in position_keys:
            print(f"  [{pk}] updating NGFF metadata")
            _update_4i_ngff_metadata(
                dest_v3_path=dest_path,
                position_key=pk,
                pheno_channel_names=pheno_channel_names,
                rounds=args.rounds,
                pheno_v2_path=source_path,
            )
        print("[metadata-only] Done.")
        return

    # --pyramids-only: only rebuild base pyramids
    if args.pyramids_only:
        args.skip_copy = True
        args.skip_transform = True
        args.skip_labels = True
        args.no_init = True

    # --labels-only: only copy labels + add 4i seg labels
    if args.labels_only:
        args.skip_copy = True
        args.skip_transform = True
        args.skip_pyramids = True
        args.no_init = True

    # --seg-pyramids-only: only build 4i seg label pyramids (per-unit)
    if args.seg_pyramids_only:
        args.no_init = True

    # --reshard-base: only reshard existing pyramid levels (no init, no build)
    if args.reshard_base:
        args.no_init = True

    # --copy-only / --transforms-only: skip init if store already exists
    if (args.copy_only or args.transforms_only) and dest_path.exists():
        args.no_init = True

    # Initialize store
    if not args.dry_run and not args.no_init:
        if not dest_path.exists() or args.force:
            initialize_v3_store_with_4i(
                pheno_v2_path=source_path,
                output_v3_path=dest_path,
                rounds=args.rounds,
                overwrite=args.force and dest_path.exists(),
                experiment=args.experiment,
            )
        else:
            print(f"Dest exists: {dest_path}. Use --force to overwrite.")
            return

    # Build jobs
    chunks = (1, 1, 1, 512, 512)
    shards_ratio = calculate_channel_based_shards(num_total_channels, chunks=chunks)

    jobs = []
    slurm_params = SLURM_PARAMS

    if args.copy_only:
        # One job per position for step 1 (pheno copy)
        slurm_params = SLURM_PARAMS_TRANSFORM
        for pos in position_keys:
            jobs.append({
                "name": f"4i_copy_{pos.replace('/', '_')}",
                "func": copy_pheno_channels_worker,
                "kwargs": {
                    "position_key": pos,
                    "pheno_v2_path": str(source_path),
                    "dest_v3_path": str(dest_path),
                    "num_pheno_channels": len(pheno_channel_names),
                },
                "metadata": {"position": pos},
            })
    elif args.transforms_only:
        # One job per position (all rounds serialized within the job).
        # We CANNOT split by round because level-0 zarr v3 arrays are sharded
        # across all channels — multiple jobs writing different channels to the
        # same position's shard file corrupts the shard (stale file handle).
        slurm_params = SLURM_PARAMS_TRANSFORM
        for pos in position_keys:
            jobs.append({
                "name": f"4i_xfm_{pos.replace('/', '_')}",
                "func": transform_all_rounds_for_position_worker,
                "kwargs": {
                    "position_key": pos,
                    "rounds": args.rounds,
                    "dest_v3_path": str(dest_path),
                    "four_i_dir": str(four_i_dir),
                    "registration_dir": str(registration_dir),
                    "start_channel": len(pheno_channel_names),
                },
                "metadata": {"position": pos, "rounds": args.rounds},
            })
    elif args.seg_pyramids_only:
        # Per (position, seg_label) jobs for 4i nuclear seg pyramids
        slurm_params = SLURM_PARAMS_PYRAMID
        seg_labels = [f"4i_R{rnd}_nuclear_seg" for rnd in args.rounds]
        for pos in position_keys:
            for seg in seg_labels:
                # Only submit if level 0 exists (label was written by --labels-only)
                label_dir = dest_path / pos / "labels" / seg / "0"
                if not label_dir.exists():
                    print(f"SKIP: {pos}/{seg} (no level 0)")
                    continue
                jobs.append({
                    "name": f"4i_segpyr_{pos.replace('/', '_')}_{seg}",
                    "func": build_seg_pyramid_unit_worker,
                    "kwargs": {
                        "source_store": str(dest_path),
                        "position": pos,
                        "seg_name": seg,
                        "levels": 5,
                        "resume": False,
                    },
                    "metadata": {"position": pos, "seg_name": seg},
                })
    elif args.reshard_base:
        # Only reshard existing base pyramid levels (1..N-1). No pyramid build.
        slurm_params = SLURM_PARAMS_RESHARD
        jobs = _build_base_reshard_jobs(
            position_keys=position_keys,
            dest_path=dest_path,
            num_levels=5,
            experiment=args.experiment,
        )
    elif args.pyramids_only:
        # Use existing per-unit pyramid builder from batch_build_pyramids
        slurm_params = SLURM_PARAMS_PYRAMID
        from cyclops_process.processes.pyramids.workers import build_pyramid_unit_worker
        from cyclops_utils.io.zarr_utils import ensure_pyramid_levels_unsharded

        # Write NGFF metadata up-front so enumerate_units works
        for position_key in position_keys:
            _update_4i_ngff_metadata(
                dest_v3_path=dest_path,
                position_key=position_key,
                pheno_channel_names=pheno_channel_names,
                rounds=args.rounds,
                pheno_v2_path=source_path,
            )

        # Pre-init unsharded pyramid levels so workers don't contend
        for pos in position_keys:
            ensure_pyramid_levels_unsharded(dest_path, pos, 5, force=True, factor=2)

        # Skip enumerate_units (it requires iohub-compliant plate metadata we don't have).
        # We know the structure: 1 timepoint × num_total_channels channels per position.
        units = [(pos, 0, c) for pos in position_keys for c in range(num_total_channels)]
        print(f"Built {len(units)} (pos, t, c) units")

        for pos, t, c in units:
            pos_label = pos.replace("/", "_")
            jobs.append({
                "name": f"4i_pyr_{pos_label}_t{t}_c{c}",
                "func": build_pyramid_unit_worker,
                "kwargs": {
                    "experiment": args.experiment,
                    "position": pos,
                    "t": t,
                    "c": c,
                    "source_store": str(dest_path),
                    "levels": 5,
                    "factor": 2,
                    "resume": False,
                },
                "metadata": {"position": pos, "t": t, "c": c},
            })
    else:
        # One job per position (covers copy, transform, pyramids, labels)
        for position_key in position_keys:
            jobs.append({
                "name": f"4i_v3_{position_key.replace('/', '_')}",
                "func": convert_position_with_4i,
                "kwargs": {
                    "experiment": args.experiment,
                    "position_key": position_key,
                    "pheno_v2_path": str(source_path),
                    "dest_v3_path": str(dest_path),
                    "four_i_dir": str(four_i_dir),
                    "registration_dir": str(registration_dir),
                    "rounds": args.rounds,
                    "pheno_channel_names": pheno_channel_names,
                    "chunks": chunks,
                    "shards_ratio": shards_ratio,
                    "skip_copy": args.skip_copy,
                    "skip_transform": args.skip_transform,
                    "skip_pyramids": args.skip_pyramids,
                    "skip_labels": args.skip_labels,
                    "add_4i_seg_labels": not args.no_4i_seg_labels,
                },
                "metadata": {"position": position_key},
            })

    if not jobs:
        print("No positions to process.")
        return

    if args.dry_run:
        print(f"DRY RUN: Would submit {len(jobs)} jobs")
        return

    result = submit_parallel_jobs(
        jobs_to_submit=jobs,
        experiment=args.experiment,
        slurm_params=slurm_params,
        log_dir="four_i/convert_v3",
        manifest_prefix="4i_v3",
        dry_run=False,
    )

    # Phase 2: after pyramid build, reshard levels 1..N-1 in-place (unsharded -> channel-sharded).
    # Skipped if any build job failed — resharding an incomplete level would corrupt it.
    if args.pyramids_only and not args.dry_run:
        failed = len(result.get("failed", [])) if isinstance(result, dict) else 0
        if failed == 0:
            reshard_jobs = _build_base_reshard_jobs(
                position_keys=position_keys,
                dest_path=dest_path,
                num_levels=5,
                experiment=args.experiment,
            )
            if reshard_jobs:
                print(f"\n{'=' * 60}")
                print(f"Phase 2: Submitting {len(reshard_jobs)} reshard jobs for base pyramid levels 1..4")
                print(f"{'=' * 60}\n")
                submit_parallel_jobs(
                    jobs_to_submit=reshard_jobs,
                    experiment=f"{args.experiment}_reshard",
                    slurm_params=SLURM_PARAMS_RESHARD,
                    log_dir="four_i/convert_v3",
                    manifest_prefix="4i_reshard",
                    dry_run=False,
                )
        else:
            print(f"\n[SKIP Phase 2 reshard] {failed} pyramid build job(s) failed — fix those first, then run --reshard-base")


if __name__ == "__main__":
    main()
