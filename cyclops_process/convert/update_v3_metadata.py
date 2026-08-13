"""
Update metadata on existing Zarr v3 stores.

This CLI updates metadata without reconverting data. It smartly merges metadata:
- Updates convert_v3 metadata (channels_metadata, core label fields) when out of date
- Preserves downstream metadata (segmentation_metadata from organelle_seg) written by other processes

Usage:
------
# Update metadata for a single experiment (default: pheno mode)
python -m cyclops_process.convert.update_v3_metadata --experiment ops0033_20250429

# Use shorthand experiment number (auto-resolves to full name)
python -m cyclops_process.convert.update_v3_metadata -e 33

# Update specific store types using --mode
python -m cyclops_process.convert.update_v3_metadata --experiment ops0033_20250429 --mode pheno
python -m cyclops_process.convert.update_v3_metadata --experiment ops0033_20250429 --mode track
python -m cyclops_process.convert.update_v3_metadata --experiment ops0033_20250429 --mode all

# Update all experiments
python -m cyclops_process.convert.update_v3_metadata --all --mode pheno

# Preview changes without applying (dry run)
python -m cyclops_process.convert.update_v3_metadata --experiment ops0033_20250429 --dry-run

# Force update all metadata (ignore existing downstream metadata)
python -m cyclops_process.convert.update_v3_metadata --experiment ops0033_20250429 --force

# Only update channel metadata (skip label metadata)
python -m cyclops_process.convert.update_v3_metadata --experiment ops0033_20250429 --channels-only

# Only update label metadata (skip channel metadata)
python -m cyclops_process.convert.update_v3_metadata --experiment ops0033_20250429 --labels-only

# Audit mode: show all metadata at every level without making changes
python -m cyclops_process.convert.update_v3_metadata --experiment ops0033_20250429 --audit
"""

import argparse
import sys
from pathlib import Path

import zarr
import iohub

from ops_utils.data.experiment import OpsDataset
from cyclops_process.convert.v3_metadata import (
    build_channels_metadata,
    build_label_metadata,
    build_labels_metadata,
    OVERLAY_METADATA,
)
from ops_utils.data.filesystem import resolve_experiment_name
from cyclops_process.paths import BASE_PATH


# Metadata keys that convert_v3 owns (will be updated)
# NOTE: If statistics is present, the label has complete downstream metadata
# and we should NOT update description/biological_annotation (they're already better)
CONVERT_V3_OWNED_KEYS = {
    "label_name",
    "annotation_type",
    "is_ome_label",
    "description",
    # For cell/nuclear seg from convert_v3
    "source_channel",
    "biological_annotation",
}

# Keys that indicate the label has complete metadata from a downstream pipeline
# When these are present, we skip updating entirely - the pipeline metadata is better
COMPLETE_METADATA_INDICATORS = {"statistics", "parameters", "stitching"}

# Labels that have their own dedicated metadata builders in downstream pipelines
# For these, if complete metadata exists, we skip update entirely
PIPELINE_OWNED_LABELS = {"cell_seg", "seg"}

# Metadata keys that downstream processes own (will be preserved)
# These are added by organelle_seg, cell_seg, and other feature extraction processes
# Note: 'segmentation' is NOT in this list because it's shared - convert_v3 sets method/version,
# while downstream adds parameters/stitching/structure_type. We handle merging separately.
DOWNSTREAM_OWNED_KEYS = {
    "preprocessing",      # CLAHE params from organelle_seg
    "detection_params",   # Frangi/blob params from organelle_seg
    "postprocessing",     # Post-processing params from organelle_seg
    "mask",               # Mask info (e.g., nuclear_seg for nucleoli)
    "statistics",         # Cell counts, etc. from cell_seg/organelle_seg
}


def is_organelle_seg_label(existing: dict) -> bool:
    """
    Check if a label was created by organelle_seg pipeline.

    organelle_seg labels have:
    - preprocessing (CLAHE params)
    - detection_params (Frangi/blob params)
    - segmentation.version containing "position-based"
    """
    if not existing:
        return False

    if "preprocessing" in existing or "detection_params" in existing:
        return True

    existing_seg = existing.get("segmentation", {})
    version = str(existing_seg.get("version", ""))
    if "position-based" in version:
        return True

    return False


def has_complete_pipeline_metadata(existing: dict, label_name: str) -> bool:
    """
    Check if a label has complete metadata from its dedicated pipeline.

    For labels like cell_seg that have their own metadata builder (in cell_segmentation.py),
    we should NOT update their metadata if it's already complete - the pipeline-generated
    metadata is always better than our generic builder.

    Args:
        existing: Existing metadata dict from the zarr store
        label_name: Name of the label

    Returns:
        True if the label has complete pipeline-generated metadata that should be preserved
    """
    if not existing:
        return False

    # Check for complete metadata indicators
    has_statistics = "statistics" in existing

    # Check nested segmentation parameters
    existing_seg = existing.get("segmentation", {})
    has_parameters = existing_seg.get("parameters") is not None
    has_stitching = existing_seg.get("stitching") is not None

    # For pipeline-owned labels (cell_seg), if we have statistics OR detailed segmentation,
    # the metadata is complete from the pipeline and shouldn't be touched
    if label_name in PIPELINE_OWNED_LABELS:
        if has_statistics or has_parameters or has_stitching:
            return True

    return False


def rebuild_organelle_seg_metadata(existing: dict, label_name: str, channel_names: list, experiment: str = None) -> dict:
    """
    Rebuild organelle_seg metadata using the proper metadata builder.

    This preserves the downstream-owned keys (preprocessing, detection_params, etc.)
    while regenerating the biological_annotation and description correctly.

    Args:
        existing: Existing metadata with preprocessing/detection_params
        label_name: Label name (e.g., "phase2d_tubular_seg")
        channel_names: List of channel names from the store
        experiment: Experiment name for YAML lookup

    Returns:
        Rebuilt metadata dict with correct biological_annotation
    """
    from organelle_profiler.organelle_seg.metadata import (
        _build_segmentation_metadata,
        get_channel_index,
    )
    from cyclops_process.convert.v3_metadata import load_channel_map_for_experiment

    # Extract info from existing metadata
    source_channel_info = existing.get("source_channel", {})
    source_channel = source_channel_info.get("name")
    existing_seg = existing.get("segmentation", {})
    structure_type = existing_seg.get("structure_type")
    segmenter_type = existing_seg.get("method", "frangi")

    # Get channel index
    channel_index = get_channel_index(channel_names, source_channel) if source_channel else -1

    # Get channel label from YAML
    yaml_label_lookup = load_channel_map_for_experiment(experiment) if experiment else {}
    channel_label = yaml_label_lookup.get(source_channel) if source_channel else None

    # If no YAML label, try to infer from channel name (Cell Painting format)
    if not channel_label and source_channel:
        parts = source_channel.split("_")
        if len(parts) >= 3 and parts[0].startswith("CP"):
            # Cell Painting format: CP1_organelle_marker
            organelle_part = parts[1]
            marker_part = "_".join(parts[2:])
            channel_label = f"{organelle_part}, {marker_part}"

    # Parse base_name from label_name for organelle_name
    base_name = label_name
    if label_name.endswith("_seg"):
        base_name = label_name[:-4]
    # Remove structure type suffix
    for suffix in ["_tubular", "_vesicular_dark", "_vesicular"]:
        if suffix in base_name:
            base_name = base_name.replace(suffix, "")
            break

    # Build new metadata using the proper builder
    new_meta = _build_segmentation_metadata(
        label_name=label_name,
        organelle_name=base_name,
        channel_name=source_channel,
        channel_label=channel_label,
        channel_index=channel_index,
        segmenter_type=segmenter_type,
        channel_names=channel_names,
        structure_type=structure_type,
        clahe_params=existing.get("preprocessing", {}).get("clahe"),
        detection_params=existing.get("detection_params"),
        postprocess_params=existing.get("postprocessing"),
    )

    # Preserve any additional downstream keys that might exist
    for key in ["mask"]:
        if key in existing and key not in new_meta:
            new_meta[key] = existing[key]

    return new_meta


def merge_label_metadata(existing: dict, new: dict, label_name: str = None, force: bool = False,
                         channel_names: list = None, experiment: str = None) -> tuple:
    """
    Merge existing label metadata with new metadata.

    Strategy:
    - If force=True, completely replace with new metadata
    - If label has complete pipeline metadata (e.g., cell_seg with statistics/parameters),
      skip update entirely - the pipeline metadata is better
    - Otherwise, update convert_v3 owned keys while preserving downstream keys
    - For nested dicts like 'segmentation', merge at the nested level

    Args:
        existing: Existing metadata dict from the zarr store
        new: New metadata dict from build_label_metadata()
        label_name: Name of the label (used to check for pipeline-owned labels)
        force: If True, completely replace existing metadata

    Returns:
        Tuple of (merged_metadata, merge_details) where merge_details is a dict with:
            - 'keys_updated': list of keys that were updated
            - 'keys_preserved': list of keys that were preserved from existing
            - 'keys_added': list of new keys added
            - 'is_downstream': whether existing had downstream metadata
            - 'skip_entirely': whether to skip update entirely (complete pipeline metadata)
    """
    merge_details = {
        "keys_updated": [],
        "keys_preserved": [],
        "keys_added": [],
        "is_downstream": False,
        "skip_entirely": False,
    }

    if force or not existing:
        merge_details["keys_added"] = list(new.keys())
        return new, merge_details

    # For pipeline-owned labels with complete metadata, skip update entirely
    # The pipeline (e.g., cell_segmentation.py) generates better metadata than our generic builder
    if label_name and has_complete_pipeline_metadata(existing, label_name):
        merge_details["skip_entirely"] = True
        merge_details["is_downstream"] = True
        merge_details["keys_preserved"] = list(existing.keys())
        return existing, merge_details

    # For organelle_seg labels, rebuild metadata using the proper builder
    # This ensures biological_annotation and description are generated correctly
    if is_organelle_seg_label(existing) and channel_names:
        rebuilt = rebuild_organelle_seg_metadata(existing, label_name, channel_names, experiment)
        merge_details["is_downstream"] = True
        merge_details["keys_updated"] = list(rebuilt.keys())
        return rebuilt, merge_details

    # Check if existing has downstream metadata
    downstream_keys_present = [key for key in DOWNSTREAM_OWNED_KEYS if key in existing]
    merge_details["is_downstream"] = len(downstream_keys_present) > 0
    merge_details["keys_preserved"] = downstream_keys_present

    # Check if existing has complete metadata from downstream pipeline
    # If so, we should NOT overwrite description/biological_annotation (they're already better)
    has_complete_metadata = any(key in existing for key in COMPLETE_METADATA_INDICATORS)
    if has_complete_metadata:
        # Also check nested: existing["segmentation"]["parameters"] or ["stitching"]
        existing_seg = existing.get("segmentation", {})
        has_complete_metadata = (
            "statistics" in existing or
            existing_seg.get("parameters") is not None or
            existing_seg.get("stitching") is not None
        )

    # Start with existing metadata
    merged = dict(existing)

    # Update convert_v3 owned keys
    # But skip description/biological_annotation if we have complete downstream metadata
    keys_to_skip = {"description", "biological_annotation"} if has_complete_metadata else set()

    for key in CONVERT_V3_OWNED_KEYS:
        if key in keys_to_skip:
            # Preserve existing value, don't update
            if key in existing:
                if key not in merge_details["keys_preserved"]:
                    merge_details["keys_preserved"].append(key)
            continue

        if key in new:
            if key in existing and existing[key] != new[key]:
                merge_details["keys_updated"].append(key)
            elif key not in existing:
                merge_details["keys_added"].append(key)
            merged[key] = new[key]

    # Special handling for 'segmentation' - merge nested dict
    # convert_v3 sets: method, version
    # Downstream processes set more detailed info: structure_type, parameters, stitching, etc.
    if "segmentation" in new:
        existing_seg = existing.get("segmentation", {})
        new_seg = new.get("segmentation", {})

        # Check if existing has detailed downstream metadata
        has_detailed_segmentation = (
            existing_seg.get("structure_type") is not None or  # organelle_seg
            existing_seg.get("parameters") is not None or      # cellpose-sam or other
            existing_seg.get("stitching") is not None or       # cellpose-sam
            "position-based" in str(existing_seg.get("version", "")) or
            "tiled" in str(existing_seg.get("version", ""))
        )

        if has_detailed_segmentation:
            # Preserve detailed downstream segmentation metadata
            merge_details["keys_preserved"].append("segmentation")
        elif existing_seg != new_seg:
            # Different and no detailed metadata - update
            if "segmentation" not in merged:
                merged["segmentation"] = {}
            merged["segmentation"].update(new_seg)
            merge_details["keys_updated"].append("segmentation")
        # If existing_seg == new_seg, do nothing (already matches)

    return merged, merge_details


def format_metadata_diff(label_name: str, existing: dict, new: dict, merge_details: dict) -> list:
    """
    Format a detailed diff of metadata changes for a label.

    Returns list of formatted strings for display.
    """
    lines = []

    if merge_details["is_downstream"]:
        lines.append(f"    {label_name}: HAS DOWNSTREAM METADATA (will merge)")
        if merge_details["keys_preserved"]:
            lines.append(f"      PRESERVE: {', '.join(merge_details['keys_preserved'])}")
        if merge_details["keys_updated"]:
            lines.append(f"      UPDATE:   {', '.join(merge_details['keys_updated'])}")
    else:
        lines.append(f"    {label_name}: convert_v3 metadata only (will replace)")
        if merge_details["keys_updated"]:
            lines.append(f"      CHANGE:   {', '.join(merge_details['keys_updated'])}")

    return lines


def get_existing_label_names(store_path: Path, position_key: str) -> list:
    """Get list of label names that exist in the store for a position."""
    store = zarr.open(str(store_path), mode="r")
    try:
        labels_group = store[position_key]["labels"]
        return list(labels_group.group_keys())
    except (KeyError, AttributeError):
        return []


def update_labels_group_metadata(
    store_path: Path,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Update the labels group metadata (ome.labels and labels list) to sync with
    what label subgroups actually exist in the store.

    This catches cases where downstream processes (like organelle_seg) added new
    labels that aren't reflected in the labels group attrs.

    Args:
        store_path: Path to v3 zarr store
        dry_run: If True, only report what would change
        verbose: Print progress

    Returns:
        Dict with:
            - 'positions_updated': count of positions with changes
            - 'labels_added': list of label names that were added to metadata
            - 'labels_removed': list of label names removed from metadata (no longer exist)
            - 'per_position_details': dict mapping position to change details
    """
    result = {
        "positions_updated": 0,
        "labels_added": set(),
        "labels_removed": set(),
        "per_position_details": {},
    }

    # Get positions
    with iohub.open_ome_zarr(store_path, mode="r") as plate:
        positions = [pos_key for pos_key, _ in plate.positions()]

    if not positions:
        if verbose:
            print("  - No positions found in store")
        return result

    store = zarr.open(str(store_path), mode="r+")

    for pos_key in positions:
        try:
            labels_group = store[pos_key]["labels"]
        except (KeyError, AttributeError):
            continue

        # Get what subgroups actually exist
        actual_subgroups = set(labels_group.group_keys())

        # Get what's in metadata
        labels_attrs = dict(labels_group.attrs)
        existing_labels = set(labels_attrs.get("labels", []))
        existing_ome = labels_attrs.get("ome", {"version": "0.5", "labels": []})
        existing_ome_labels = set(existing_ome.get("labels", []))

        # Determine what needs to change
        # Labels to add: exist in store but not in metadata
        labels_to_add = actual_subgroups - existing_labels

        # Labels to remove: in metadata but no longer exist in store
        labels_to_remove = existing_labels - actual_subgroups

        # For OME labels, we add all new labels that are OME-compliant (not overlays)
        ome_labels_to_add = labels_to_add - set(OVERLAY_METADATA.keys())
        ome_labels_to_remove = existing_ome_labels - actual_subgroups

        pos_details = {
            "actual_subgroups": sorted(actual_subgroups),
            "metadata_labels": sorted(existing_labels),
            "metadata_ome_labels": sorted(existing_ome_labels),
            "labels_to_add": sorted(labels_to_add),
            "labels_to_remove": sorted(labels_to_remove),
            "ome_labels_to_add": sorted(ome_labels_to_add),
            "ome_labels_to_remove": sorted(ome_labels_to_remove),
            "needs_update": bool(labels_to_add or labels_to_remove),
        }
        result["per_position_details"][pos_key] = pos_details

        # Track globally
        result["labels_added"].update(labels_to_add)
        result["labels_removed"].update(labels_to_remove)

        # Apply changes if needed
        if labels_to_add or labels_to_remove:
            result["positions_updated"] += 1

            if not dry_run:
                # Build new labels list
                new_labels = sorted(actual_subgroups)
                new_ome_labels = sorted(actual_subgroups - set(OVERLAY_METADATA.keys()))

                labels_group.attrs["labels"] = new_labels
                labels_group.attrs["ome"] = {
                    "version": "0.5",
                    "labels": new_ome_labels
                }

    # Convert sets to sorted lists for return
    result["labels_added"] = sorted(result["labels_added"])
    result["labels_removed"] = sorted(result["labels_removed"])

    # Print summary
    if verbose:
        print(f"\n  Labels Group Metadata Analysis:")
        print(f"  " + "-" * 50)
        print(f"  Target: {{position}}/labels/zarr.json -> ome.labels, labels")

        if result["labels_added"]:
            print(f"\n  LABELS TO ADD TO METADATA (exist in store but not tracked):")
            for label in result["labels_added"]:
                print(f"    + {label}")

        if result["labels_removed"]:
            print(f"\n  LABELS TO REMOVE FROM METADATA (tracked but no longer exist):")
            for label in result["labels_removed"]:
                print(f"    - {label}")

        if not result["labels_added"] and not result["labels_removed"]:
            print(f"\n  Labels group metadata is in sync with actual subgroups")

        print(f"\n  " + "-" * 50)

        if dry_run:
            print(f"\n  [DRY RUN] Would update labels group in {result['positions_updated']} positions")
        else:
            if result["positions_updated"] > 0:
                print(f"\n  Updated labels group metadata in {result['positions_updated']} positions")

    return result


def audit_store_metadata(store_path: Path, verbose: bool = True) -> dict:
    """
    Audit all metadata at every level of a zarr v3 store.

    Shows metadata at:
    - Plate level: ome, channels_metadata
    - Position level: ome, custom_metadata (clims_per_level, normalization)
    - Array level: custom_metadata on pyramid levels
    - Labels group: ome, labels list
    - Label subgroups: segmentation_metadata, custom_metadata

    Args:
        store_path: Path to v3 zarr store
        verbose: Print detailed output

    Returns:
        Dict with comprehensive metadata audit results
    """
    store = zarr.open(str(store_path), mode="r")

    audit = {
        "plate_level": {},
        "positions": {},
        "labels_summary": {},
    }

    # === PLATE LEVEL ===
    if verbose:
        print(f"\n  {'='*56}")
        print(f"  PLATE LEVEL METADATA")
        print(f"  {'='*56}")

    plate_attrs = dict(store.attrs)
    audit["plate_level"]["attrs_keys"] = list(plate_attrs.keys())

    # OME metadata
    ome = plate_attrs.get("ome", {})
    if verbose:
        print(f"\n  ome:")
        if "plate" in ome:
            plate_info = ome["plate"]
            n_rows = len(plate_info.get("rows", []))
            n_cols = len(plate_info.get("columns", []))
            n_wells = len(plate_info.get("wells", []))
            print(f"    plate: {n_rows} rows × {n_cols} cols = {n_wells} wells")
        print(f"    version: {ome.get('version', 'N/A')}")

    # Channels metadata
    channels_meta = plate_attrs.get("channels_metadata", [])
    audit["plate_level"]["channels_metadata_count"] = len(channels_meta)
    if verbose:
        print(f"\n  channels_metadata: {len(channels_meta)} channels")
        for i, ch in enumerate(channels_meta):
            ch_name = ch.get("name", f"ch_{i}")
            ch_type = ch.get("type", "unknown")
            ch_label = ch.get("label", "")
            label_str = f" ({ch_label})" if ch_label else ""
            print(f"    [{i}] {ch_name}: {ch_type}{label_str}")

    # Get positions
    with iohub.open_ome_zarr(store_path, mode="r") as plate:
        positions = [pos_key for pos_key, _ in plate.positions()]
        channel_names = list(plate.channel_names)

    audit["plate_level"]["channel_names"] = channel_names
    audit["plate_level"]["n_positions"] = len(positions)

    if verbose:
        print(f"\n  channel_names: {channel_names}")
        print(f"  positions: {len(positions)}")

    # === POSITION LEVEL (sample first position) ===
    if positions:
        first_pos = positions[0]
        if verbose:
            print(f"\n  {'='*56}")
            print(f"  POSITION LEVEL METADATA (sample: {first_pos})")
            print(f"  {'='*56}")

        pos_group = store[first_pos]
        pos_attrs = dict(pos_group.attrs)
        audit["positions"][first_pos] = {"attrs_keys": list(pos_attrs.keys())}

        # Position OME metadata
        pos_ome = pos_attrs.get("ome", {})
        if verbose:
            print(f"\n  ome:")
            if "multiscales" in pos_ome:
                ms = pos_ome["multiscales"]
                if ms and len(ms) > 0:
                    n_datasets = len(ms[0].get("datasets", []))
                    print(f"    multiscales: {n_datasets} pyramid levels")
                    axes = ms[0].get("axes", [])
                    if axes:
                        axis_str = ", ".join([f"{a.get('name', '?')}({a.get('type', '?')})" for a in axes])
                        print(f"    axes: {axis_str}")

        # Position custom_metadata (clims, normalization)
        pos_custom = pos_attrs.get("custom_metadata", {})
        audit["positions"][first_pos]["custom_metadata_keys"] = list(pos_custom.keys())
        if verbose:
            print(f"\n  custom_metadata:")
            if "clims_per_level" in pos_custom:
                clims = pos_custom["clims_per_level"]
                print(f"    clims_per_level: {len(clims)} pyramid levels")
                for level, level_clims in sorted(clims.items()):
                    method = level_clims.get("contrast_limits_method", "N/A")
                    n_channels = len(level_clims.get("contrast_limits_per_channel", []))
                    print(f"      level {level}: {n_channels} channels, method={method}")
            if "normalization" in pos_custom:
                norm = pos_custom["normalization"]
                print(f"    normalization: {norm}")
            if not pos_custom:
                print(f"    (empty)")

        # === LABELS GROUP ===
        if verbose:
            print(f"\n  {'='*56}")
            print(f"  LABELS GROUP METADATA ({first_pos}/labels)")
            print(f"  {'='*56}")

        try:
            labels_group = store[first_pos]["labels"]
            labels_attrs = dict(labels_group.attrs)

            # Labels OME
            labels_ome = labels_attrs.get("ome", {})
            ome_labels = labels_ome.get("labels", [])
            all_labels = labels_attrs.get("labels", [])

            audit["labels_summary"]["ome_labels"] = ome_labels
            audit["labels_summary"]["all_labels"] = all_labels

            if verbose:
                print(f"\n  ome.labels (OME-compliant): {len(ome_labels)}")
                for label in ome_labels:
                    print(f"    • {label}")

                non_ome = set(all_labels) - set(ome_labels)
                if non_ome:
                    print(f"\n  Non-OME labels (overlays): {len(non_ome)}")
                    for label in sorted(non_ome):
                        print(f"    • {label}")

            # === LABEL SUBGROUP METADATA ===
            if verbose:
                print(f"\n  {'='*56}")
                print(f"  LABEL SUBGROUP METADATA")
                print(f"  {'='*56}")

            label_subgroups = list(labels_group.group_keys())
            audit["labels_summary"]["subgroups"] = {}

            for label_name in sorted(label_subgroups):
                try:
                    subgroup = labels_group[label_name]
                    subgroup_attrs = dict(subgroup.attrs)
                    attr_keys = list(subgroup_attrs.keys())

                    seg_meta = subgroup_attrs.get("segmentation_metadata", {})
                    custom_meta = subgroup_attrs.get("custom_metadata", {})

                    audit["labels_summary"]["subgroups"][label_name] = {
                        "attrs_keys": attr_keys,
                        "has_segmentation_metadata": bool(seg_meta),
                        "has_custom_metadata": bool(custom_meta),
                        "segmentation_metadata_keys": list(seg_meta.keys()) if seg_meta else [],
                    }

                    if verbose:
                        print(f"\n  {label_name}:")
                        if seg_meta:
                            annotation_type = seg_meta.get("annotation_type", "unknown")
                            has_downstream = any(k in seg_meta for k in DOWNSTREAM_OWNED_KEYS)
                            downstream_marker = " [HAS DOWNSTREAM]" if has_downstream else ""
                            print(f"    segmentation_metadata: {annotation_type}{downstream_marker}")
                            print(f"      keys: {list(seg_meta.keys())}")
                            if has_downstream:
                                downstream_present = [k for k in DOWNSTREAM_OWNED_KEYS if k in seg_meta]
                                print(f"      downstream keys: {downstream_present}")
                        if custom_meta:
                            print(f"    custom_metadata: {list(custom_meta.keys())}")
                        if not seg_meta and not custom_meta:
                            print(f"    (no metadata)")

                except Exception as e:
                    if verbose:
                        print(f"\n  {label_name}: ERROR - {e}")

        except (KeyError, AttributeError) as e:
            if verbose:
                print(f"  No labels group found: {e}")

    return audit


def update_channels_metadata(
    store_path: Path,
    experiment: str,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Update channel metadata at the plate level.

    Args:
        store_path: Path to v3 zarr store
        experiment: Experiment name for channel map lookup
        dry_run: If True, only report what would change
        verbose: Print progress

    Returns:
        Dict with 'updated' bool and 'changes' list
    """
    result = {"updated": False, "changes": [], "details": {}}

    # Get current channel names from store
    with iohub.open_ome_zarr(store_path, mode="r") as plate:
        channel_names = list(plate.channel_names)

    # Build new metadata
    new_metadata = build_channels_metadata(channel_names, experiment=experiment)

    # Get existing metadata
    store = zarr.open(str(store_path), mode="r+")
    existing_metadata = store.attrs.get("channels_metadata", [])

    # Detailed comparison
    channels_changed = []
    channels_unchanged = []
    channels_missing = []

    if len(existing_metadata) == len(new_metadata):
        for i, (old, new) in enumerate(zip(existing_metadata, new_metadata)):
            ch_name = new.get("name", f"channel_{i}")
            if old != new:
                # Find what changed, with before/after values
                changed_fields = []
                for key in set(old.keys()) | set(new.keys()):
                    if old.get(key) != new.get(key):
                        changed_fields.append((key, old.get(key, "(missing)"), new.get(key, "(missing)")))
                channels_changed.append((ch_name, changed_fields))
            else:
                channels_unchanged.append(ch_name)
    elif len(existing_metadata) == 0:
        channels_missing = [m.get("name", f"ch_{i}") for i, m in enumerate(new_metadata)]
    else:
        # Different number of channels
        result["changes"].append(f"channels_metadata count changed: {len(existing_metadata)} -> {len(new_metadata)}")

    result["details"] = {
        "channels_changed": channels_changed,
        "channels_unchanged": channels_unchanged,
        "channels_missing": channels_missing,
    }

    # Compare
    if existing_metadata != new_metadata:
        if verbose:
            print(f"\n  Channel Metadata Analysis:")
            print(f"  " + "-" * 50)
            print(f"  Target: zarr.json -> channels_metadata (plate level)")

            if channels_missing:
                print(f"\n  MISSING (will add):")
                for ch_name in channels_missing:
                    print(f"    + {ch_name}")

            if channels_changed:
                print(f"\n  WILL UPDATE:")
                for ch_name, changed_fields in channels_changed:
                    print(f"    📝 {ch_name}")
                    for field, old_val, new_val in changed_fields:
                        print(f"       \033[91m- {field}: {old_val}\033[0m")
                        print(f"       \033[92m+ {field}: {new_val}\033[0m")

            if channels_unchanged:
                print(f"\n  UNCHANGED: {len(channels_unchanged)} channels")
                # Only show first few
                for ch_name in channels_unchanged[:3]:
                    print(f"    ✓ {ch_name}")
                if len(channels_unchanged) > 3:
                    print(f"    ... and {len(channels_unchanged) - 3} more")

            print(f"\n  " + "-" * 50)

        if not dry_run:
            existing_attrs = dict(store.attrs)
            existing_attrs["channels_metadata"] = new_metadata
            store.attrs.update(existing_attrs)
            result["updated"] = True
            if verbose:
                print(f"\n  ✓ Updated channels_metadata ({len(new_metadata)} channels)")
        else:
            if verbose:
                print(f"\n  [DRY RUN] Would update channels_metadata ({len(new_metadata)} channels)")
    else:
        if verbose:
            print(f"  ✓ channels_metadata already up to date ({len(existing_metadata)} channels)")

    return result


def update_label_metadata(
    store_path: Path,
    experiment: str,
    store_type: str = None,
    dry_run: bool = False,
    force: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Update label metadata for all positions and labels.

    Args:
        store_path: Path to v3 zarr store
        experiment: Experiment name for channel map lookup
        store_type: Type of store ('pheno', 'track', 'iss') - affects metadata for some labels
        dry_run: If True, only report what would change
        force: If True, completely replace existing metadata (don't preserve downstream)
        verbose: Print progress

    Returns:
        Dict with 'updated_count', 'preserved_count', 'changes' list, 'details' dict
    """
    result = {
        "updated_count": 0,
        "preserved_count": 0,
        "changes": [],
        "details": {
            "labels_with_downstream": [],
            "labels_to_update": [],
            "labels_up_to_date": [],
            "per_label_details": {},
        }
    }

    # Get channel names from store
    with iohub.open_ome_zarr(store_path, mode="r") as plate:
        channel_names = list(plate.channel_names)
        positions = [pos_key for pos_key, _ in plate.positions()]

    store = zarr.open(str(store_path), mode="r+")

    # Process first position to get label list (assume consistent across positions)
    if not positions:
        if verbose:
            print("  - No positions found in store")
        return result

    first_pos = positions[0]
    label_names = get_existing_label_names(store_path, first_pos)

    if not label_names:
        if verbose:
            print("  - No labels found in store")
        return result

    if verbose:
        print(f"  Found {len(label_names)} labels across {len(positions)} positions")

    # Build new metadata for all labels
    new_labels_metadata = build_labels_metadata(label_names, channel_names, experiment, store_type=store_type)

    # First pass: analyze what will change (use first position as representative)
    # This allows us to show detailed info before making changes
    for label_name in label_names:
        try:
            labels_group = store[first_pos]["labels"]
            if label_name not in labels_group:
                continue
            subgroup = labels_group[label_name]
        except (KeyError, AttributeError):
            continue

        # Determine metadata key based on label type
        is_overlay = label_name in OVERLAY_METADATA
        metadata_key = "custom_metadata" if is_overlay else "segmentation_metadata"

        # Get existing and new metadata
        existing = dict(subgroup.attrs.get(metadata_key, {}))
        new = new_labels_metadata.get(label_name, {})

        # Get merge analysis
        merged, merge_details = merge_label_metadata(
            existing, new, label_name=label_name, force=force,
            channel_names=channel_names, experiment=experiment
        )

        # Store analysis
        result["details"]["per_label_details"][label_name] = {
            "existing_keys": list(existing.keys()),
            "new_keys": list(new.keys()),
            "merge_details": merge_details,
            "has_changes": merged != existing,
            "metadata_key": metadata_key,
        }

        if merge_details.get("skip_entirely"):
            # Complete pipeline metadata - skip entirely
            result["details"]["labels_up_to_date"].append(label_name)
        elif merge_details["is_downstream"] and not force:
            result["details"]["labels_with_downstream"].append(label_name)
        elif merged != existing:
            result["details"]["labels_to_update"].append(label_name)
        else:
            result["details"]["labels_up_to_date"].append(label_name)

    # Print detailed analysis in verbose/dry-run mode
    if verbose:
        print(f"\n  Label Metadata Analysis:")
        print(f"  " + "-" * 50)
        print(f"  Target: {{position}}/labels/{{label}}/zarr.json -> segmentation_metadata or custom_metadata")

        # Labels with downstream metadata (will be merged/preserved)
        if result["details"]["labels_with_downstream"]:
            print(f"\n  DOWNSTREAM METADATA (will merge, preserving organelle_seg data):")
            for label_name in result["details"]["labels_with_downstream"]:
                details = result["details"]["per_label_details"][label_name]
                merge_info = details["merge_details"]
                meta_key = details["metadata_key"]
                print(f"    🔒 {label_name}")
                print(f"       Path: {{pos}}/labels/{label_name}/zarr.json -> {meta_key}")
                if merge_info["keys_preserved"]:
                    print(f"       Preserve: {', '.join(merge_info['keys_preserved'])}")
                if merge_info["keys_updated"]:
                    print(f"       Update:   {', '.join(merge_info['keys_updated'])}")

        # Labels to update (no downstream, will be replaced)
        if result["details"]["labels_to_update"]:
            print(f"\n  WILL UPDATE (convert_v3 metadata only):")
            for label_name in result["details"]["labels_to_update"]:
                details = result["details"]["per_label_details"][label_name]
                merge_info = details["merge_details"]
                meta_key = details["metadata_key"]
                print(f"    📝 {label_name}")
                print(f"       Path: {{pos}}/labels/{label_name}/zarr.json -> {meta_key}")
                if merge_info["keys_updated"]:
                    print(f"       Changed: {', '.join(merge_info['keys_updated'])}")
                if merge_info["keys_added"]:
                    print(f"       Added:   {', '.join(merge_info['keys_added'])}")

        # Labels already up to date
        if result["details"]["labels_up_to_date"]:
            print(f"\n  ALREADY UP TO DATE:")
            for label_name in result["details"]["labels_up_to_date"]:
                details = result["details"]["per_label_details"][label_name]
                merge_info = details["merge_details"]
                meta_key = details["metadata_key"]
                if merge_info.get("skip_entirely"):
                    print(f"    ✓ {label_name} (complete pipeline metadata - skip)")
                else:
                    print(f"    ✓ {label_name} ({{pos}}/labels/{label_name}/zarr.json -> {meta_key})")

        print(f"\n  " + "-" * 50)

    # Second pass: actually update metadata for all positions
    for pos_key in positions:
        try:
            labels_group = store[pos_key]["labels"]
        except (KeyError, AttributeError):
            continue

        for label_name in label_names:
            if label_name not in labels_group:
                continue

            subgroup = labels_group[label_name]

            # Get stored analysis
            label_details = result["details"]["per_label_details"].get(label_name)
            if not label_details:
                continue

            metadata_key = label_details["metadata_key"]
            existing = dict(subgroup.attrs.get(metadata_key, {}))
            new = new_labels_metadata.get(label_name, {})

            # Get merge result
            merged, merge_details = merge_label_metadata(
                existing, new, label_name=label_name, force=force,
                channel_names=channel_names, experiment=experiment
            )

            # Check if we should skip entirely (complete pipeline metadata)
            if merge_details.get("skip_entirely"):
                result["preserved_count"] += 1
                continue

            # Check if this has downstream metadata that should be preserved
            has_downstream = merge_details["is_downstream"]

            if has_downstream and not force:
                # Merge, preserving downstream metadata
                if merged != existing:
                    if not dry_run:
                        subgroup.attrs[metadata_key] = merged
                        result["updated_count"] += 1
                    result["changes"].append(f"{pos_key}/labels/{label_name}: merged (preserved downstream)")
                else:
                    result["preserved_count"] += 1
            else:
                # No downstream metadata or force mode - replace entirely
                if existing != new:
                    if not dry_run:
                        subgroup.attrs[metadata_key] = new
                        result["updated_count"] += 1
                    result["changes"].append(f"{pos_key}/labels/{label_name}: updated")

    if verbose:
        if dry_run:
            n_positions = len(positions)
            n_to_update = len(result["details"]["labels_to_update"])
            n_downstream = len(result["details"]["labels_with_downstream"])
            n_up_to_date = len(result["details"]["labels_up_to_date"])

            print(f"\n  [DRY RUN] Summary for {n_positions} positions:")
            print(f"    Would update: {n_to_update} labels × {n_positions} positions = {n_to_update * n_positions} entries")
            print(f"    Would merge:  {n_downstream} labels × {n_positions} positions = {n_downstream * n_positions} entries")
            print(f"    Up to date:   {n_up_to_date} labels")
        else:
            print(f"\n  ✓ Updated {result['updated_count']} label metadata entries")
            if result["preserved_count"] > 0:
                print(f"  ✓ Preserved {result['preserved_count']} entries with downstream metadata")

    return result


def get_store_path_for_mode(mode: str, dataset: OpsDataset) -> tuple:
    """
    Get store path for a given mode.

    Returns:
        Tuple of (label, store_path) or (None, None) if not found
    """
    mode_map = {
        "pheno": ("pheno", "pheno_assembled_v3"),
        "track": ("track", "lc_5x_phase_2d_stitched_v3"),
        "iss": ("iss", "iss_stitch_registered_v3"),
    }

    if mode not in mode_map:
        return None, None

    label, store_key = mode_map[mode]
    store_path = dataset.store_paths.get(store_key)

    if store_path is None or not store_path.exists():
        return label, None

    return label, store_path


def update_experiment_metadata(
    experiment: str,
    mode: str = "pheno",
    dry_run: bool = False,
    force: bool = False,
    channels_only: bool = False,
    labels_only: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Update metadata for an experiment's v3 store.

    Args:
        experiment: Experiment name
        mode: Store mode ('pheno', 'track', 'iss', or 'all')
        dry_run: If True, only report what would change
        force: If True, completely replace existing metadata
        channels_only: Only update channel metadata
        labels_only: Only update label metadata
        verbose: Print progress

    Returns:
        Dict with results summary
    """
    dataset = OpsDataset(experiment)

    # Determine which stores to process
    if mode == "all":
        modes_to_process = ["pheno", "track", "iss"]
    else:
        modes_to_process = [mode]

    results = {
        "experiment": experiment,
        "stores_updated": [],
        "stores_skipped": [],
        "total_channel_updates": 0,
        "total_labels_group_updates": 0,
        "total_label_updates": 0,
        "total_preserved": 0,
    }

    for m in modes_to_process:
        label, store_path = get_store_path_for_mode(m, dataset)

        if store_path is None:
            if verbose:
                print(f"\n  ⚠ {m}: store not found, skipping")
            results["stores_skipped"].append(m)
            continue

        if verbose:
            print(f"\n  Processing {m} store: {store_path}")

        # Update channel metadata
        if not labels_only:
            channel_result = update_channels_metadata(
                store_path, experiment, dry_run=dry_run, verbose=verbose
            )
            if channel_result["updated"]:
                results["total_channel_updates"] += 1

        # Update labels group metadata (sync with actual subgroups)
        if not channels_only:
            labels_group_result = update_labels_group_metadata(
                store_path, dry_run=dry_run, verbose=verbose
            )
            results["total_labels_group_updates"] += labels_group_result["positions_updated"]

        # Update label metadata
        if not channels_only:
            label_result = update_label_metadata(
                store_path, experiment, store_type=m, dry_run=dry_run, force=force, verbose=verbose
            )
            results["total_label_updates"] += label_result["updated_count"]
            results["total_preserved"] += label_result["preserved_count"]

        results["stores_updated"].append(m)

    return results


def detect_experiments_with_v3_stores(mode: str = "pheno", verbose: bool = True) -> list:
    """
    Scan for experiments that have v3 stores.

    Args:
        mode: Store mode to check for ('pheno', 'track', 'iss')
        verbose: Print progress

    Returns:
        List of experiment names with v3 stores
    """
    mode_to_store_key = {
        "pheno": "pheno_assembled_v3",
        "track": "lc_5x_phase_2d_stitched_v3",
        "iss": "iss_stitch_registered_v3",
    }

    store_key = mode_to_store_key.get(mode)
    if not store_key:
        return []

    ops_dir = Path(f"{BASE_PATH}")
    experiments = sorted([
        d.name for d in ops_dir.iterdir()
        if d.is_dir() and d.name.startswith("ops")
    ])

    experiments_with_stores = []

    if verbose:
        print(f"\nScanning {len(experiments)} experiments for {mode} v3 stores...")

    for experiment in experiments:
        try:
            dataset = OpsDataset(experiment)
            store_path = dataset.store_paths.get(store_key)
            if store_path and store_path.exists():
                experiments_with_stores.append(experiment)
        except Exception:
            continue

    if verbose:
        print(f"Found {len(experiments_with_stores)} experiments with {mode} v3 stores\n")

    return experiments_with_stores


def main():
    """CLI entry point for metadata update."""
    parser = argparse.ArgumentParser(
        description="Update metadata on existing Zarr v3 stores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--experiment", "-e", type=str, nargs="+",
        help="Experiment name(s) (e.g., ops0033_20250429 or shorthand 33 64 94). Required unless --all is used.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process all experiments that have v3 stores",
    )
    parser.add_argument(
        "--mode", type=str, choices=["pheno", "track", "iss", "all"], default="pheno",
        help="Store mode: 'pheno', 'track', 'iss', or 'all' (default: pheno)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be changed without making changes",
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Force update all metadata, overwriting downstream metadata (organelle_seg, etc.)",
    )
    parser.add_argument(
        "--channels-only", action="store_true",
        help="Only update channel metadata (skip label metadata)",
    )
    parser.add_argument(
        "--labels-only", action="store_true",
        help="Only update label metadata (skip channel metadata)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Reduce verbosity",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip confirmation prompt (use with --all)",
    )
    parser.add_argument(
        "--audit", action="store_true",
        help="Audit mode: show all metadata at every level without making changes",
    )

    args = parser.parse_args()

    # Validation
    if not args.all and not args.experiment and not args.audit:
        parser.error("--experiment is required unless --all or --audit is used")

    if args.audit and not args.experiment:
        parser.error("--audit requires --experiment")

    if args.channels_only and args.labels_only:
        parser.error("Cannot specify both --channels-only and --labels-only")

    verbose = not args.quiet

    # Resolve experiment names from shorthand (e.g., "33" -> "ops0033_20250429")
    if args.experiment:
        args.experiment = [resolve_experiment_name(e, autoselect=True) for e in args.experiment]

    # Handle --audit mode
    if args.audit:
        audit_exp = args.experiment[0] if args.experiment else None
        if not audit_exp:
            parser.error("--audit requires --experiment")
        print(f"\n{'='*60}")
        print(f"METADATA AUDIT: {audit_exp}")
        print(f"Mode: {args.mode}")
        print(f"{'='*60}")

        dataset = OpsDataset(audit_exp)

        # Determine which stores to audit
        if args.mode == "all":
            modes_to_audit = ["pheno", "track", "iss"]
        else:
            modes_to_audit = [args.mode]

        for m in modes_to_audit:
            label, store_path = get_store_path_for_mode(m, dataset)

            if store_path is None:
                print(f"\n  ⚠ {m}: store not found, skipping")
                continue

            print(f"\n{'='*60}")
            print(f"  STORE: {m}")
            print(f"  PATH: {store_path}")
            print(f"{'='*60}")

            audit_store_metadata(store_path, verbose=True)

        print()
        sys.exit(0)

    # Handle --all mode
    if args.all:
        # For --all with mode='all', we need to check each mode separately
        detection_mode = args.mode if args.mode != "all" else "pheno"

        experiments = detect_experiments_with_v3_stores(mode=detection_mode, verbose=verbose)

        if not experiments:
            print(f"No experiments found with {detection_mode} v3 stores.")
            sys.exit(0)

        # Print summary
        print(f"\n{'='*60}")
        print(f"Metadata Update: {len(experiments)} experiments")
        print(f"{'='*60}\n")

        for exp in experiments:
            print(f"  • {exp}")

        # Prompt for confirmation
        if not args.yes and not args.dry_run:
            try:
                response = input(f"\nUpdate metadata for {len(experiments)} experiments? [y/N]: ").strip().lower()
                if response not in ['y', 'yes']:
                    print("\nCancelled by user.\n")
                    sys.exit(0)
            except (KeyboardInterrupt, EOFError):
                print("\n\nCancelled by user.\n")
                sys.exit(0)
            print()

        # Process each experiment
        total_results = {
            "experiments_processed": 0,
            "total_channel_updates": 0,
            "total_labels_group_updates": 0,
            "total_label_updates": 0,
            "total_preserved": 0,
        }

        for idx, experiment in enumerate(experiments, 1):
            print(f"\n{'='*60}")
            print(f"[{idx}/{len(experiments)}] {experiment}")
            print(f"{'='*60}")

            result = update_experiment_metadata(
                experiment=experiment,
                mode=args.mode,
                dry_run=args.dry_run,
                force=args.force,
                channels_only=args.channels_only,
                labels_only=args.labels_only,
                verbose=verbose,
            )

            total_results["experiments_processed"] += 1
            total_results["total_channel_updates"] += result["total_channel_updates"]
            total_results["total_labels_group_updates"] += result["total_labels_group_updates"]
            total_results["total_label_updates"] += result["total_label_updates"]
            total_results["total_preserved"] += result["total_preserved"]

        # Summary
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        print(f"Experiments processed: {total_results['experiments_processed']}")
        print(f"Channel metadata updates: {total_results['total_channel_updates']}")
        print(f"Labels group sync updates: {total_results['total_labels_group_updates']}")
        print(f"Label metadata updates: {total_results['total_label_updates']}")
        print(f"Preserved (downstream): {total_results['total_preserved']}")
        if args.dry_run:
            print("\n[DRY RUN] No changes were made.")
        print()

    else:
        # Experiment list mode
        experiments = args.experiment
        total_results = {
            "experiments_processed": 0,
            "total_channel_updates": 0,
            "total_labels_group_updates": 0,
            "total_label_updates": 0,
            "total_preserved": 0,
        }

        for idx, experiment in enumerate(experiments, 1):
            print(f"\n{'='*60}")
            if len(experiments) > 1:
                print(f"[{idx}/{len(experiments)}] Updating metadata for {experiment}")
            else:
                print(f"Updating metadata for {experiment}")
            print(f"Mode: {args.mode}")
            if args.dry_run:
                print("[DRY RUN MODE]")
            if args.force:
                print("[FORCE MODE - will overwrite downstream metadata]")
            print(f"{'='*60}")

            result = update_experiment_metadata(
                experiment=experiment,
                mode=args.mode,
                dry_run=args.dry_run,
                force=args.force,
                channels_only=args.channels_only,
                labels_only=args.labels_only,
                verbose=verbose,
            )

            total_results["experiments_processed"] += 1
            total_results["total_channel_updates"] += result["total_channel_updates"]
            total_results["total_labels_group_updates"] += result["total_labels_group_updates"]
            total_results["total_label_updates"] += result["total_label_updates"]
            total_results["total_preserved"] += result["total_preserved"]

        # Summary
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        if len(experiments) > 1:
            print(f"Experiments processed: {total_results['experiments_processed']}")
        print(f"Channel metadata updates: {total_results['total_channel_updates']}")
        print(f"Labels group sync updates: {total_results['total_labels_group_updates']}")
        print(f"Label metadata updates: {total_results['total_label_updates']}")
        print(f"Preserved (downstream): {total_results['total_preserved']}")
        if args.dry_run:
            print("\n[DRY RUN] No changes were made.")
        print()


if __name__ == "__main__":
    main()
