"""
Metadata Building for Zarr v3 Conversion
=========================================

This module provides functions for building comprehensive metadata
for channels and labels during v3 zarr conversion.

The metadata format matches the organelle_seg metadata module for consistency
across all segmentation labels in the zarr stores.

Channel metadata is stored at the plate level in zarr.json:
    zarr.json -> attributes -> channels_metadata

Label metadata is stored on each label subgroup:
    {position}/labels/{label_name} -> segmentation_metadata
"""

import re
import yaml
from pathlib import Path

from ops_utils.data.experiment import OpsDataset
from ops_utils.data.naming import (
    parse_channel_label,
    build_channel_metadata,
    get_channel_type as _get_channel_type_base,
)


# =============================================================================
# CHANNEL TYPE DETECTION
# =============================================================================

# Virtual stain channel patterns (from convert_v3 phenotyping pipeline)
VIRTUAL_STAIN_PATTERNS = ["_prediction", "nuclei_prediction", "membrane_prediction"]

# Label-free microscopy channels (from organelle_seg configs)
LABELFREE_CHANNELS = {"Phase2D", "Focus3D", "Phase3D", "Raw"}


def _cp_channel_label(channel_name: str) -> str | None:
    """Build a channel label for a cell painting channel using convert_v3_impl definitions.

    E.g. "CP1_mitochondria_TOMM20" -> "Mitochondria, TOMM20"
         "CP1_plasma_membrane_WGA" -> "Plasma Membrane, Wheat Germ Agglutinin"
    """
    from cyclops_process.convert.v3_fixed import CELL_PAINTING_CHANNELS

    # Parse CPn_ prefix to get part number
    import re
    m = re.match(r"^CP(\d+)_", channel_name)
    if not m:
        return None
    part = int(m.group(1))

    # Search for matching channel in the definitions
    for ch_idx, info in CELL_PAINTING_CHANNELS.get(part, {}).items():
        full_name = f"CP{part}_{info['name']}_{info['marker']}"
        if channel_name == full_name:
            structure = info.get("structure", info["name"].capitalize())
            marker = info.get("full_marker", info["marker"])
            return f"{structure}, {marker}"

    return None


def _classify_no_label(label: str | None) -> str | None:
    """Classify a channel label into a no-label subtype, or None if it's a real label.

    Handles all known YAML label formats:
        "no label"                    -> "empty"
        "empty, no label"            -> "empty"
        "bleedthrough, no label"     -> "bleedthrough"
        "bleedthrough, mCherry"      -> "bleedthrough"  (older experiments)
        "autofluorescence, no label" -> "autofluorescence"

    Returns:
        "empty"            - no expected signal (default for bare "no label" or missing)
        "bleedthrough"     - signal from another channel bleeding through
        "autofluorescence" - cellular autofluorescence
        None               - not a no-label channel (has a real biological label)
    """
    if label is None:
        return None  # no YAML entry — let channel name parsing handle it
    low = label.lower().strip()
    if low == "no label":
        return "empty"
    # Any label starting with bleedthrough or autofluorescence is not a real marker,
    # regardless of suffix ("no label", channel name, etc.)
    prefix = low.split(",")[0].strip()
    if prefix.startswith("bleed"):
        return "bleedthrough"
    if prefix.startswith("autofluor"):
        return "autofluorescence"
    if "no label" in low:
        if prefix == "empty":
            return "empty"
        return "empty"
    return None


def get_channel_type(channel_name: str) -> str:
    """
    Determine channel type for metadata purposes.

    Extends the organelle_seg get_channel_type to also handle:
    - Virtual stain channels (prediction outputs)
    - Label-free channels (Phase2D, Focus3D, etc.)

    Args:
        channel_name: Channel name from the zarr store

    Returns:
        Channel type: "virtual_stain", "labelfree", or "fluorescent"
    """
    ch_lower = channel_name.lower()

    # Check for virtual stain patterns first
    for pattern in VIRTUAL_STAIN_PATTERNS:
        if pattern.lower() in ch_lower:
            return "virtual_stain"

    # Check for label-free channels
    if channel_name in LABELFREE_CHANNELS:
        return "labelfree"

    # Raw brightfield z-slices (BF_z0..BF_z6) from the titration pipeline are
    # label-free, like Phase2D/Focus3D.
    if channel_name.startswith("BF_z"):
        return "labelfree"

    # Default to fluorescent for all other channels
    return "fluorescent"


# =============================================================================
# CHANNEL METADATA
# =============================================================================


def load_channel_map_for_experiment(experiment: str) -> dict:
    """
    Load channel mapping from ops_channel_maps.yaml for an experiment.

    Args:
        experiment: Experiment name (e.g., "ops0094_20251217" or "ops0094")

    Returns:
        Dict mapping channel_name -> label string, or empty dict if not found.
    """
    from ops_utils.data.filesystem import extract_ops_key

    # Get config path
    dataset = OpsDataset("dummy")  # Just to get config paths
    config_path = dataset.channel_maps

    if not config_path.exists():
        return {}

    with open(config_path, 'r') as f:
        channel_maps = yaml.safe_load(f)

    if not channel_maps:
        return {}

    # Extract ops key (e.g., "ops0094" from "ops0094_20251217")
    ops_key = extract_ops_key(experiment)

    # Try various key formats
    exp_config = None
    for key in [experiment, ops_key, experiment.lower()]:
        if key and key in channel_maps:
            exp_config = channel_maps[key]
            break

    if exp_config is None:
        return {}

    # Build lookup dict from YAML entries
    label_lookup = {}
    if isinstance(exp_config, list):
        for entry in exp_config:
            if isinstance(entry, dict) and "channel_name" in entry:
                label_lookup[entry["channel_name"]] = entry.get("label", "")

    return label_lookup


def build_channels_metadata(
    channel_names: list,
    experiment: str = None,
) -> list:
    """
    Build metadata for all channels in a zarr store.

    Args:
        channel_names: List of channel names from the zarr store
        experiment: Optional experiment name to load channel maps from YAML

    Returns:
        List of per-channel metadata dicts
    """
    # Load channel map from YAML if experiment provided
    yaml_label_lookup = {}
    if experiment:
        yaml_label_lookup = load_channel_map_for_experiment(experiment)

    # Build metadata for each channel
    channels_metadata = []

    for idx, ch_name in enumerate(channel_names):
        # Get label from YAML if available
        ch_label = yaml_label_lookup.get(ch_name)

        # Handle prediction channels (virtual stain outputs)
        if ch_label is None:
            if "nuclei_prediction" in ch_name.lower():
                ch_label = "nuclei, virtual stain"
            elif "membrane_prediction" in ch_name.lower():
                ch_label = "membrane, virtual stain"

        # Handle cell painting channels using definitions from convert_v3_slurm_cp
        if ch_label is None and ch_name.startswith("CP"):
            ch_label = _cp_channel_label(ch_name)

        # Get channel type
        ch_type = get_channel_type(ch_name)

        # Handle unlabeled channels: empty, bleedthrough, or autofluorescence.
        # YAML formats:
        #   "no label"                      -> empty (no expected signal)
        #   "empty, no label"               -> empty
        #   "bleedthrough, no label"         -> bleedthrough from another channel
        #   "autofluorescence, no label"     -> cellular autofluorescence
        # Missing label (None)              -> empty
        no_label_type = _classify_no_label(ch_label)

        ch_metadata = build_channel_metadata(
            channel_name=ch_name,
            channel_index=idx,
            channel_label=None if no_label_type else ch_label,
            channel_type=ch_type,
        )

        if no_label_type:
            _NO_LABEL_META = {
                "empty": {
                    "marker_type": "empty",
                    "description": f"Max projected no label empty from {ch_name} channel",
                },
                "bleedthrough": {
                    "marker_type": "bleedthrough",
                    "description": f"Max projected no label bleedthrough from {ch_name} channel",
                },
                "autofluorescence": {
                    "marker_type": "autofluorescence",
                    "description": f"Max projected no label autofluorescence from {ch_name} channel",
                },
            }
            meta = _NO_LABEL_META[no_label_type]
            ch_metadata["biological_annotation"] = {
                "organelle": "no label",
                "marker": "no label",
                "marker_type": meta["marker_type"],
                "full_label": f"no label, {meta['marker_type']}",
            }
            ch_metadata["description"] = meta["description"]

        # Fix marker_type and description for cell painting channels
        # (naming.py doesn't know nuclear_dye/direct_label types or CP description format)
        if ch_name.startswith("CP") and "biological_annotation" in ch_metadata:
            bio = ch_metadata["biological_annotation"]
            marker_val = bio.get("marker", "")
            if marker_val and "Hoechst" in marker_val:
                bio["marker_type"] = "nuclear_dye"
            elif bio.get("marker_type") == "endogenous_tag":
                bio["marker_type"] = "direct_label"
            part = ch_name[2]  # "1" or "2" from "CP1_..." or "CP2_..."
            organelle = bio.get("organelle", "")
            ch_metadata["description"] = f"Cell painting {organelle} visualized via {marker_val} (Part {part})"

        # Label-free channels: override the auto-built metadata. When the
        # experiment's ops_channel_maps.yaml declares ``BF`` (not the
        # downstream ``Phase2D`` / ``Focus3D`` derivatives), ``yaml_label_lookup``
        # returns None for these channels, and the upstream
        # ``build_channel_metadata`` stamps a bogus
        # ``biological_annotation: {marker: 'no label', marker_type: 'empty', ...}``
        # as a fallback. Phase2D/Focus3D are NOT "missing labels" — they're
        # label-free brightfield reconstructions. Strip the bogus block and
        # set the proper description.
        if ch_name == "Phase2D":
            ch_metadata.pop("biological_annotation", None)
            ch_metadata["description"] = "Projected 2D reconstruction of label-free brightfield imaging"
        elif ch_name == "Focus3D":
            ch_metadata.pop("biological_annotation", None)
            ch_metadata["description"] = "Reconstructed focal slice from 3D reconstruction of label-free brightfield"
        elif ch_name.startswith("BF_z"):
            ch_metadata.pop("biological_annotation", None)
            ch_metadata["description"] = f"Raw label-free brightfield z-slice {ch_name[len('BF_z'):]}"

        channels_metadata.append(ch_metadata)

    return channels_metadata


# =============================================================================
# LABEL METADATA
# =============================================================================

# Basic metadata for non-segmentation label groups
OVERLAY_METADATA = {
    "grid_overlay": {
        "annotation_type": "grid_overlay",
        "description": "Grid overlay with tile boundaries (bright blue) and tile IDs as text labels (RGBA image)",
        "is_ome_label": False,
    },
    # Legacy entries (kept for backward compatibility with old stores)
    "grid_edges": {
        "annotation_type": "grid_overlay",
        "description": "Grid edge overlay for tile boundaries (tiles named XXXYYY where XXX=column, YYY=row)",
        "is_ome_label": False,
    },
    "grid_props": {
        "annotation_type": "grid_properties",
        "description": "Grid tile properties (ID, coordinates, names using XXXYYY convention where XXX=column, YYY=row)",
        "is_ome_label": False,
    },
    "iss_points": {
        "annotation_type": "iss_overlay",
        "description": "ISS point detection overlay",
        "is_ome_label": False,
    },
    "iss_points_props": {
        "annotation_type": "iss_properties",
        "description": "ISS point properties (coordinates, labels)",
        "is_ome_label": False,
    },
    "iss_gene_image": {
        "annotation_type": "iss_gene_overlay",
        "description": "OPS guideRNA gene KO RGBA overlay from in-situ sequencing",
        "is_ome_label": False,
    },
    "iss_guide_image": {
        "annotation_type": "iss_guide_overlay",
        "description": "OPS guideRNA KO RGBA overlay from in-situ sequencing",
        "is_ome_label": False,
    },
}


def build_label_metadata(
    label_name: str,
    channel_names: list,
    experiment: str = None,
    store_type: str = None,
) -> dict:
    """
    Build comprehensive metadata for a segmentation label.

    Matches the format used by organelle_seg metadata module for consistency.

    Args:
        label_name: Label name (e.g., "seg", "nuclear_seg", "grid_edges")
        channel_names: List of channel names from the zarr store
        experiment: Optional experiment name to load channel maps from YAML
        store_type: Type of store ('pheno', 'track', 'iss') - affects metadata for some labels

    Returns:
        Dictionary with comprehensive segmentation metadata
    """
    # Check if this is an overlay (non-segmentation) label
    if label_name in OVERLAY_METADATA:
        return {
            "label_name": label_name,
            **OVERLAY_METADATA[label_name],
        }

    # Determine source channel and annotation based on label type
    if label_name == "seg" or label_name == "cell_seg":
        # Cell segmentation from membrane_prediction virtual stain
        source_channel = "membrane_prediction"
        organelle = "cell_membrane"
        marker = "virtual stain"
        marker_type = "virtual_stain"
        annotation_type = "cell_segmentation"
        method = "cellpose"
        description = "Cell segmentation from membrane virtual stain using Cellpose"

    elif label_name == "nuclear_seg":
        # Nuclear segmentation - source depends on store type
        if store_type == "iss":
            # ISS stores use DAPI channel directly (no virtual staining)
            source_channel = "DAPI"
            organelle = "nuclei"
            marker = "DAPI"
            marker_type = "fluorescent_dye"
            annotation_type = "nuclear_segmentation"
            method = "cellpose"
            description = "Nuclear segmentation from DAPI channel using Cellpose"
        else:
            # Pheno/track stores use nuclei_prediction virtual stain
            source_channel = "nuclei_prediction"
            organelle = "nuclei"
            marker = "virtual stain"
            marker_type = "virtual_stain"
            annotation_type = "nuclear_segmentation"
            method = "cellpose"
            description = "Nuclear segmentation from nuclei virtual stain using Cellpose"

    elif label_name.startswith("CP") and "nuclear_seg" in label_name:
        # Cell painting nuclear segmentation (CP1_nuclear_seg, CP2_nuclear_seg)
        part = label_name.split("_")[0]  # "CP1" or "CP2"
        source_channel = "Hoechst"
        organelle = "nuclei"
        marker = "Hoechst"
        marker_type = "live_cell_dye"
        annotation_type = "nuclear_segmentation"
        method = "cellpose"
        description = f"Nuclear segmentation from {part} cell painting Hoechst using Cellpose"

    elif label_name.endswith("_seg") or label_name.endswith("_vesselness"):
        # Organelle segmentation label - use the organelle_seg metadata builder
        # Labels are produced by organelle_seg with format: {organelle}_{marker}_{structure}_seg
        # e.g., "mitochondria_tomm20_tubular_seg", "phase_2d_vesicular_seg", "nucleoli_phase2d_seg"
        from organelle_profiler.organelle_seg.metadata import (
            _build_segmentation_metadata,
            _build_vesselness_metadata,
        )

        # Parse label name to extract info
        is_vesselness = label_name.endswith("_vesselness")
        base_name = label_name[:-11] if is_vesselness else label_name[:-4]  # Remove suffix

        # Determine structure type from label name
        structure_type = None
        if "_tubular" in base_name:
            structure_type = "tubular"
            base_name = base_name.replace("_tubular", "")
        elif "_vesicular_dark" in base_name:
            structure_type = "vesicular_dark"
            base_name = base_name.replace("_vesicular_dark", "")
        elif "_vesicular" in base_name:
            structure_type = "vesicular"
            base_name = base_name.replace("_vesicular", "")

        # Load YAML labels for channel lookup
        yaml_label_lookup = load_channel_map_for_experiment(experiment) if experiment else {}

        # Deterministic channel matching based on known label formats from organelle_seg
        # Format: {organelle}_{marker} where marker often maps to channel name
        source_channel = None
        channel_index = -1
        channel_label = None

        # Known label-to-channel mappings (deterministic)
        LABEL_CHANNEL_MAP = {
            # Label-free channels
            "phase_2d": "Phase2D",
            "focus_3d": "Focus3D",
            # Nucleoli use Phase2D or Focus3D
            "nucleoli_phase2d": "Phase2D",
            "nucleoli_focus3d": "Focus3D",
            "nuclo_phase": "Phase2D",
            "nuclo_focus": "Focus3D",
        }

        # Try direct mapping first
        if base_name in LABEL_CHANNEL_MAP:
            source_channel = LABEL_CHANNEL_MAP[base_name]
        else:
            # For fluorescent channels, the label format is {organelle}_{channel}
            # e.g., "mitochondria_mcherry" -> mCherry channel
            # Check each channel to see if it appears in the label
            for ch in channel_names:
                ch_lower = ch.lower().replace("_", "")  # Normalize underscores
                base_normalized = base_name.replace("_", "")
                if ch_lower in base_normalized or base_normalized.endswith(ch_lower):
                    source_channel = ch
                    break

        # Get channel index and label
        if source_channel and source_channel in channel_names:
            channel_index = channel_names.index(source_channel)
            # First try YAML lookup
            channel_label = yaml_label_lookup.get(source_channel)

            # If no YAML label, try to infer from channel name
            # Cell Painting channels have format: CP{N}_{organelle}_{marker}
            # e.g., "CP2_microtubules_Tubulin" -> "microtubules, Tubulin"
            if not channel_label and source_channel:
                parts = source_channel.split("_")
                if len(parts) >= 3 and parts[0].startswith("CP"):
                    # Cell Painting format: CP1_organelle_marker
                    organelle_part = parts[1]
                    marker_part = "_".join(parts[2:])
                    channel_label = f"{organelle_part}, {marker_part}"
                elif len(parts) >= 2:
                    # Generic format: organelle_marker or just channel name
                    channel_label = ", ".join(parts)

        if is_vesselness:
            return _build_vesselness_metadata(
                label_name=label_name,
                organelle_name=base_name,
                channel_name=source_channel,
                channel_label=channel_label,
                channel_index=channel_index,
                channel_names=channel_names,
            )
        else:
            return _build_segmentation_metadata(
                label_name=label_name,
                organelle_name=base_name,
                channel_name=source_channel,
                channel_label=channel_label,
                channel_index=channel_index,
                segmenter_type="frangi",
                channel_names=channel_names,
                structure_type=structure_type,
            )

    else:
        # Unknown label - use basic metadata
        return {
            "label_name": label_name,
            "annotation_type": "unknown",
            "is_ome_label": True,
            "description": f"Label: {label_name}",
        }

    # Get channel index
    channel_index = -1
    if source_channel and source_channel in channel_names:
        channel_index = channel_names.index(source_channel)

    # Get channel type
    channel_type = get_channel_type(source_channel) if source_channel else None

    # Build metadata dictionary matching organelle_seg format
    metadata = {
        # Core identification
        "label_name": label_name,
        "annotation_type": annotation_type,
        "is_ome_label": True,

        # Source channel information
        "source_channel": {
            "name": source_channel,
            "index": channel_index,
            "type": channel_type,
            "all_channels": channel_names,
        },

        # Biological annotation
        "biological_annotation": {
            "organelle": organelle,
            "marker": marker,
            "marker_type": marker_type,
            "full_label": f"{organelle}, {marker}" if organelle and marker else None,
        },

        # Segmentation method
        "segmentation": {
            "method": method,
            "version": "phenotyping-v3",
        },

        # Human-readable description
        "description": description,
    }

    return metadata


def build_labels_metadata(
    label_names: list,
    channel_names: list,
    experiment: str = None,
    store_type: str = None,
) -> dict:
    """
    Build metadata for all labels in a zarr store.

    Args:
        label_names: List of label names (e.g., ["seg", "nuclear_seg", "grid_edges"])
        channel_names: List of channel names from the zarr store
        experiment: Optional experiment name to load channel maps from YAML
        store_type: Type of store ('pheno', 'track', 'iss') - affects metadata for some labels

    Returns:
        Dict mapping label_name -> metadata dict
    """
    labels_metadata = {}
    for label_name in label_names:
        labels_metadata[label_name] = build_label_metadata(
            label_name, channel_names, experiment, store_type=store_type
        )
    return labels_metadata


def write_label_metadata_to_store(
    store_path: Path,
    position_key: str,
    label_name: str,
    metadata: dict,
):
    """
    Write metadata to a label subgroup in the zarr store.

    For segmentations: writes to {position}/labels/{label_name} -> segmentation_metadata
    For overlays: writes to {position}/labels/{label_name} -> custom_metadata

    Args:
        store_path: Path to zarr v3 store
        position_key: Position key like "A/1/0"
        label_name: Name of the label (e.g., "seg", "nuclear_seg", "iss_gene_image")
        metadata: Metadata dictionary to write
    """
    import zarr

    store = zarr.open(str(store_path), mode="r+")

    try:
        labels_group = store[position_key]["labels"]
        if label_name in labels_group:
            subgroup = labels_group[label_name]
            # Use custom_metadata for overlays, segmentation_metadata for segmentations
            is_overlay = label_name in OVERLAY_METADATA
            metadata_key = "custom_metadata" if is_overlay else "segmentation_metadata"
            subgroup.attrs[metadata_key] = metadata
    except (KeyError, AttributeError):
        # Label doesn't exist yet, skip
        pass


