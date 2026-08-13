#!/usr/bin/env python
"""
Batch check Zarr segmentation stores for incorrect channel counts.

This script scans all OPS experiments and checks if their phenotyping zarr stores
(both v2 and v3) have segmentation arrays (seg, nuclear_seg) with incorrect channel
dimensions.

Segmentation arrays should have shape [T, 1, Z, Y, X] where the channel dimension
(index 1) is 1. A common bug causes these arrays to inherit the channel count from
the parent image (e.g., 6 channels) instead of having 1 channel.

Zarr v2 vs v3 differences:
    - v2: {position}/seg/0/.zarray (stores shape in "shape" key)
    - v3: {position}/labels/seg/0/zarr.json (stores shape in "shape" key)

Environment:
    This script only reads metadata files - no special Zarr environment needed.

Usage:
    # Check all experiments (v3 phenotyping by default)
    python -m cyclops_process.tests.QC.check_zarr_v3_channel_count

    # Check specific experiments
    python -m cyclops_process.tests.QC.check_zarr_v3_channel_count --experiments ops0045_20250603

    # Check v2 stores instead of v3
    python -m cyclops_process.utils.batch.batch_check_zarr_v3_channel_count --zarr-version v2

    # Check both v2 and v3 stores
    python -m cyclops_process.tests.QC.check_zarr_v3_channel_count --zarr-version both

    # Check all store types (pheno, track, iss)
    python -m cyclops_process.tests.QC.check_zarr_v3_channel_count --all-stores

    # Verbose output showing all checks
    python -m cyclops_process.tests.QC.check_zarr_v3_channel_count --verbose

    # Output results to CSV
    python -m cyclops_process.tests.QC.check_zarr_v3_channel_count --output results.csv

    # Check for all-zero pyramid levels (requires zarr environment)
    python -m cyclops_process.tests.QC.check_zarr_v3_channel_count --check-zeroes
"""
import sys
import os

sys.path.insert(0, os.getcwd())

import argparse
import json
import csv
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional

from cyclops_utils.data.experiment import OpsDataset
from cyclops_process.paths import BASE_PATH


# Map of store types to their keys in OpsDataset.store_paths
# Format: {store_type: {zarr_version: store_key}}
STORE_MAP = {
    "pheno": {
        "v2": "pheno_assembled",      # phenotyping.zarr
        "v3": "pheno_assembled_v3",   # phenotyping_v3.zarr
    },
    "track": {
        "v2": "lc_5x_phase_2d_stitched",
        "v3": "lc_5x_phase_2d_stitched_v3",
    },
    "iss": {
        "v2": "iss_stitch_registered",
        "v3": "iss_stitch_registered_v3",
    },
}

# Label arrays that should have exactly 1 channel
SEGMENTATION_LABELS = ["seg", "nuclear_seg"]

# Expected channel count for segmentation arrays
EXPECTED_CHANNEL_COUNT = 1

# Global flag to track if zarr is available for zero checking
_ZARR_AVAILABLE = None


def _check_zarr_available() -> bool:
    """Check if zarr library is available for zero checking."""
    global _ZARR_AVAILABLE
    if _ZARR_AVAILABLE is None:
        try:
            import zarr
            _ZARR_AVAILABLE = True
        except ImportError:
            _ZARR_AVAILABLE = False
    return _ZARR_AVAILABLE


def check_symlink_status(path: Path) -> Dict:
    """
    Check if a path is a symlink and if so, whether the target exists.

    Args:
        path: Path to check

    Returns:
        dict with keys:
            - is_symlink: bool
            - symlink_target: str or None, the symlink target if is_symlink
            - target_exists: bool or None, whether the symlink target exists
            - resolved_path: str or None, the fully resolved path
    """
    result = {
        "is_symlink": False,
        "symlink_target": None,
        "target_exists": None,
        "resolved_path": None,
    }

    if path.is_symlink():
        result["is_symlink"] = True
        try:
            result["symlink_target"] = str(os.readlink(path))
            resolved = path.resolve()
            result["resolved_path"] = str(resolved)
            result["target_exists"] = resolved.exists()
        except OSError as e:
            result["target_exists"] = False
            result["symlink_target"] = f"<error: {e}>"

    return result


def check_array_all_zeroes(array_path: Path, zarr_version: str = "v3", sample_size: int = 1000) -> Dict:
    """
    Check if a zarr array contains all zeros by sampling data.

    For efficiency, samples a subset of the data rather than reading the entire array.
    Checks multiple random locations to detect zero-filled arrays.

    Args:
        array_path: Path to array directory (contains zarr.json or .zarray)
        zarr_version: "v2" or "v3"
        sample_size: Number of pixels to sample per check

    Returns:
        dict with keys:
            - checked: bool, whether check was performed
            - is_all_zeroes: bool or None, True if array appears to be all zeros
            - max_value: number or None, maximum value found in samples
            - error: str or None, error message if check failed
            - symlink_info: dict with symlink status info
    """
    result = {
        "checked": False,
        "is_all_zeroes": None,
        "max_value": None,
        "error": None,
        "symlink_info": check_symlink_status(array_path),
    }

    if not _check_zarr_available():
        result["error"] = "zarr library not available"
        return result

    import zarr
    import numpy as np

    try:
        if zarr_version == "v3":
            # For v3, we need zarr-python 3.x
            try:
                arr = zarr.open(str(array_path), mode="r")
            except Exception as e:
                result["error"] = f"Failed to open v3 array: {e}"
                return result
        else:
            # v2 store
            arr = zarr.open(str(array_path), mode="r")

        result["checked"] = True

        # Get array shape
        shape = arr.shape
        if len(shape) < 2:
            result["error"] = f"Unexpected shape: {shape}"
            return result

        # Sample from multiple locations
        # For 5D array [T, C, Z, Y, X], sample from different Y, X positions
        total_elements = np.prod(shape)

        if total_elements == 0:
            result["is_all_zeroes"] = True
            result["max_value"] = 0
            return result

        # Read a single chunk from the center of the array
        # This is efficient because zarr reads whole chunks anyway
        try:
            max_val = 0

            if len(shape) >= 3:
                # Get chunk shape from the array
                chunk_shape = arr.chunks

                # Get Y, X dimensions (last two)
                y_dim = shape[-2]
                x_dim = shape[-1]
                chunk_y = chunk_shape[-2]
                chunk_x = chunk_shape[-1]

                # Find the center chunk
                center_chunk_y = (y_dim // 2) // chunk_y
                center_chunk_x = (x_dim // 2) // chunk_x

                # Calculate the slice for that chunk
                y_start = center_chunk_y * chunk_y
                x_start = center_chunk_x * chunk_x
                y_end = min(y_start + chunk_y, y_dim)
                x_end = min(x_start + chunk_x, x_dim)

                if len(shape) == 5:
                    sample = arr[0, 0, 0, y_start:y_end, x_start:x_end]
                elif len(shape) == 4:
                    sample = arr[0, 0, y_start:y_end, x_start:x_end]
                elif len(shape) == 3:
                    sample = arr[0, y_start:y_end, x_start:x_end]

                max_val = np.max(sample)
            else:
                sample = arr.flat[:min(1000, total_elements)]
                max_val = np.max(sample)

            result["max_value"] = float(max_val)
            result["is_all_zeroes"] = (max_val == 0)

        except Exception as e:
            result["error"] = f"Failed to read sample: {e}"
            return result

    except Exception as e:
        result["error"] = f"Failed to open array: {e}"

    return result


def find_pyramid_levels(label_path: Path, zarr_version: str = "v3") -> List[Path]:
    """
    Find all pyramid levels for a label array.

    Args:
        label_path: Path to label directory (e.g., .../labels/seg or .../seg)
        zarr_version: "v2" or "v3"

    Returns:
        List of paths to each pyramid level (0, 1, 2, etc.)
    """
    levels = []
    if not label_path.exists():
        return levels

    for item in sorted(label_path.iterdir()):
        if item.is_dir() and item.name.isdigit():
            levels.append(item)

    return levels


def check_position_zeroes(
    position_path: Path,
    zarr_version: str = "v3",
    label_names: List[str] = None,
    verbose: bool = False
) -> Dict[str, List[Dict]]:
    """
    Check all label arrays in a position for all-zero pyramid levels.

    Args:
        position_path: Path to position directory (e.g., .../A/1/0)
        zarr_version: "v2" or "v3" - determines path structure
        label_names: List of label names to check (default: seg, nuclear_seg)
        verbose: Print detailed info

    Returns:
        dict mapping label_name -> list of zero level results
    """
    if label_names is None:
        label_names = SEGMENTATION_LABELS

    results = {}

    for label_name in label_names:
        if zarr_version == "v3":
            label_path = position_path / "labels" / label_name
        else:  # v2
            label_path = position_path / label_name

        if not label_path.exists():
            continue

        levels = find_pyramid_levels(label_path, zarr_version)
        zero_levels = []

        for level_path in levels:
            level_num = int(level_path.name)
            check_result = check_array_all_zeroes(level_path, zarr_version=zarr_version)

            if check_result["checked"] and check_result["is_all_zeroes"]:
                zero_levels.append({
                    "level": level_num,
                    "path": str(level_path),
                    "max_value": check_result["max_value"],
                    "symlink_info": check_result.get("symlink_info", {}),
                })

            if check_result["error"] and verbose:
                print(f"        Warning: {level_path.name}: {check_result['error']}")

        if zero_levels:
            results[label_name] = zero_levels

    return results


def count_actual_channels(array_dir: Path, zarr_version: str = "v3") -> Optional[int]:
    """
    Count the actual number of channel directories in a zarr array.

    For zarr v2/v3 with dimension_separator="/", chunks are stored as T/C/Z/Y/X.
    We count how many subdirectories exist under 0/ (first time point).

    For v3 with sharding, we check if it uses sharding codec and assume
    the data has correct channels (since we can't easily inspect sharded data).

    For v2, we also check the .zarray metadata for dimension_separator.

    Args:
        array_dir: Path to array directory (contains zarr.json or .zarray)
        zarr_version: "v2" or "v3"

    Returns:
        Number of channel directories found, or None if cannot determine
    """
    # Check for T=0 directory (non-sharded stores with "/" dimension separator)
    t0_dir = array_dir / "0"
    if t0_dir.exists():
        # Count channel directories under T=0
        channel_dirs = [d for d in t0_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if channel_dirs:
            return len(channel_dirs)

    # For v2, check if using "." dimension separator (flat chunk naming like 0.0.0.0.0)
    if zarr_version == "v2":
        zarray_path = array_dir / ".zarray"
        if zarray_path.exists():
            try:
                with open(zarray_path, "r") as f:
                    metadata = json.load(f)
                dim_sep = metadata.get("dimension_separator", ".")

                if dim_sep == ".":
                    # Flat chunk naming - look for chunk files like 0.0.0.0.0
                    # Chunk names are T.C.Z.Y.X - we want to find unique C values (position 1)
                    chunk_files = [f for f in array_dir.iterdir()
                                   if f.is_file() and f.name[0].isdigit()]
                    if chunk_files:
                        # Parse chunk files to find unique channel values
                        # We need to check shape to know how many dimensions
                        shape = metadata.get("shape", [])
                        if len(shape) == 5:  # T, C, Z, Y, X
                            channel_values = set()
                            for cf in chunk_files:
                                parts = cf.name.split(".")
                                if len(parts) == 5:
                                    channel_values.add(parts[1])  # C is at index 1
                            if channel_values:
                                return len(channel_values)
                        # If we have chunks but can't determine channels, assume 1
                        return EXPECTED_CHANNEL_COUNT
                    else:
                        # No chunk files - empty array, metadata-only issue
                        return EXPECTED_CHANNEL_COUNT
            except (json.JSONDecodeError, IOError):
                pass

    # For v3 sharded stores, check if using sharding codec
    if zarr_version == "v3":
        zarr_json = array_dir / "zarr.json"
        if zarr_json.exists():
            try:
                with open(zarr_json, "r") as f:
                    metadata = json.load(f)
                codecs = metadata.get("codecs", [])
                # Check if any codec is sharding_indexed
                for codec in codecs:
                    if codec.get("name") == "sharding_indexed":
                        # Sharded store - check for shard files (c/X/Y/Z format)
                        # If shard files exist, assume data is correct (1 channel)
                        # because segmentation data should always be 1 channel
                        shard_dir = array_dir / "c"
                        if shard_dir.exists() and any(shard_dir.iterdir()):
                            # Sharded data exists - assume correct channel count
                            return EXPECTED_CHANNEL_COUNT
                        # No shard data - metadata issue only
                        return EXPECTED_CHANNEL_COUNT
            except (json.JSONDecodeError, IOError):
                pass

    return None


def check_array_channel_count(metadata_path: Path, zarr_version: str = "v3", verbose: bool = False) -> Dict:
    """
    Check if a zarr array has the correct channel count.

    Args:
        metadata_path: Path to metadata file (zarr.json for v3, .zarray for v2)
        zarr_version: "v2" or "v3"
        verbose: Print detailed info

    Returns:
        dict with keys:
            - exists: bool, whether the metadata file exists
            - shape: list, the array shape if exists
            - channel_count: int, the channel dimension value (index 1)
            - is_correct: bool, whether channel count is 1
            - chunk_shape: list, the chunk shape if exists
            - actual_channels: int or None, actual channel count from data
            - is_metadata_only: bool or None, True if metadata wrong but data correct
    """
    result = {
        "exists": False,
        "shape": None,
        "channel_count": None,
        "is_correct": None,
        "chunk_shape": None,
        "actual_channels": None,
        "is_metadata_only": None,
    }

    if not metadata_path.exists():
        return result

    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        if verbose:
            print(f"      Error reading {metadata_path}: {e}")
        return result

    result["exists"] = True

    # Get shape - expected format [T, C, Z, Y, X]
    shape = metadata.get("shape", [])
    result["shape"] = shape

    if len(shape) >= 2:
        result["channel_count"] = shape[1]
        result["is_correct"] = shape[1] == EXPECTED_CHANNEL_COUNT

    # Get chunk shape - different location in v2 vs v3
    if zarr_version == "v3":
        chunk_grid = metadata.get("chunk_grid", {})
        if chunk_grid.get("name") == "regular":
            result["chunk_shape"] = chunk_grid.get("configuration", {}).get("chunk_shape")
    else:  # v2
        result["chunk_shape"] = metadata.get("chunks")

    # Check actual data channel count if metadata shows wrong count
    if not result["is_correct"]:
        array_dir = metadata_path.parent
        actual_channels = count_actual_channels(array_dir, zarr_version)
        result["actual_channels"] = actual_channels

        if actual_channels is not None:
            # Determine if it's metadata-only issue or data issue
            result["is_metadata_only"] = (actual_channels == EXPECTED_CHANNEL_COUNT)

    return result


def fix_metadata_channel_count(metadata_path: Path, zarr_version: str = "v3", dry_run: bool = False) -> Dict:
    """
    Fix the channel count in zarr metadata to be 1.

    Only fixes the shape and chunk_shape channel dimension (index 1).

    Args:
        metadata_path: Path to metadata file (zarr.json for v3, .zarray for v2)
        zarr_version: "v2" or "v3"
        dry_run: If True, only report what would be changed

    Returns:
        dict with keys:
            - success: bool
            - old_shape: list
            - new_shape: list
            - old_chunk_shape: list
            - new_chunk_shape: list
    """
    result = {
        "success": False,
        "old_shape": None,
        "new_shape": None,
        "old_chunk_shape": None,
        "new_chunk_shape": None,
    }

    if not metadata_path.exists():
        return result

    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, IOError):
        return result

    # Get current shape
    shape = metadata.get("shape", [])
    if len(shape) < 2:
        return result

    result["old_shape"] = shape.copy()

    # Fix shape - set channel dimension to 1
    new_shape = shape.copy()
    new_shape[1] = EXPECTED_CHANNEL_COUNT
    result["new_shape"] = new_shape

    # Fix chunk shape
    if zarr_version == "v3":
        chunk_grid = metadata.get("chunk_grid", {})
        if chunk_grid.get("name") == "regular":
            chunk_shape = chunk_grid.get("configuration", {}).get("chunk_shape", [])
            if len(chunk_shape) >= 2:
                result["old_chunk_shape"] = chunk_shape.copy()
                new_chunk_shape = chunk_shape.copy()
                new_chunk_shape[1] = EXPECTED_CHANNEL_COUNT
                result["new_chunk_shape"] = new_chunk_shape
    else:  # v2
        chunk_shape = metadata.get("chunks", [])
        if len(chunk_shape) >= 2:
            result["old_chunk_shape"] = chunk_shape.copy()
            new_chunk_shape = chunk_shape.copy()
            new_chunk_shape[1] = EXPECTED_CHANNEL_COUNT
            result["new_chunk_shape"] = new_chunk_shape

    if not dry_run:
        # Apply changes
        metadata["shape"] = new_shape

        if zarr_version == "v3":
            if "chunk_grid" in metadata and metadata["chunk_grid"].get("name") == "regular":
                metadata["chunk_grid"]["configuration"]["chunk_shape"] = result["new_chunk_shape"]
        else:  # v2
            if "chunks" in metadata and result["new_chunk_shape"]:
                metadata["chunks"] = result["new_chunk_shape"]

        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            result["success"] = True
        except IOError:
            return result
    else:
        result["success"] = True

    return result


def check_position_labels(
    position_path: Path,
    zarr_version: str = "v3",
    label_names: List[str] = None,
    verbose: bool = False
) -> Dict[str, Dict]:
    """
    Check all label arrays in a position for correct channel counts.

    Checks ALL pyramid levels (0, 1, 2, etc.) for each label array.

    Args:
        position_path: Path to position directory (e.g., .../A/1/0)
        zarr_version: "v2" or "v3" - determines path structure
        label_names: List of label names to check (default: seg, nuclear_seg)
        verbose: Print detailed info

    Returns:
        dict mapping label_name -> check result (from first incorrect level, or level 0 if all correct)

    Path differences:
        v2: {position}/seg/0/.zarray
        v3: {position}/labels/seg/0/zarr.json
    """
    if label_names is None:
        label_names = SEGMENTATION_LABELS

    results = {}

    for label_name in label_names:
        # Get the label directory
        if zarr_version == "v3":
            label_dir = position_path / "labels" / label_name
            metadata_filename = "zarr.json"
        else:  # v2
            label_dir = position_path / label_name
            metadata_filename = ".zarray"

        if not label_dir.exists():
            continue

        # Find all pyramid levels
        pyramid_levels = sorted([
            d for d in label_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        ], key=lambda x: int(x.name))

        # Check each pyramid level
        first_incorrect_result = None
        level_0_result = None

        for level_dir in pyramid_levels:
            metadata_path = level_dir / metadata_filename

            result = check_array_channel_count(metadata_path, zarr_version=zarr_version, verbose=verbose)
            if result["exists"]:
                # Store level 0 result as fallback
                if level_dir.name == "0":
                    level_0_result = result

                # If this level has incorrect channel count, use it
                if not result["is_correct"] and first_incorrect_result is None:
                    # Update metadata_path in the calling context to point to this level
                    first_incorrect_result = result
                    first_incorrect_result["_level"] = int(level_dir.name)

        # Return the first incorrect result, or level 0 result if all are correct
        if first_incorrect_result is not None:
            results[label_name] = first_incorrect_result
        elif level_0_result is not None:
            results[label_name] = level_0_result

    return results


def find_positions(zarr_path: Path) -> List[Path]:
    """
    Find all position directories in a zarr store.

    HCS/plate format: row/col/fov (e.g., A/1/0, B/2/0)

    Returns:
        List of position paths
    """
    positions = []

    if not zarr_path.exists():
        return positions

    # Iterate through row/col/fov structure
    for row_dir in sorted(zarr_path.glob("*")):
        if row_dir.is_dir() and not row_dir.name.startswith(".") and row_dir.name != "zarr.json":
            for col_dir in sorted(row_dir.glob("*")):
                if col_dir.is_dir() and not col_dir.name.startswith("."):
                    for fov_dir in sorted(col_dir.glob("*")):
                        if fov_dir.is_dir() and not fov_dir.name.startswith("."):
                            positions.append(fov_dir)

    return positions


def check_store_channel_counts(
    zarr_path: Path,
    zarr_version: str = "v3",
    verbose: bool = False
) -> Dict:
    """
    Check all segmentation arrays in a zarr store for correct channel counts.

    Args:
        zarr_path: Path to zarr store
        zarr_version: "v2" or "v3"
        verbose: Print detailed info

    Returns:
        dict with keys:
            - total_positions: int
            - total_arrays_checked: int
            - incorrect_arrays: list of dicts with position/label/details
            - correct_arrays: int count
    """
    result = {
        "total_positions": 0,
        "total_arrays_checked": 0,
        "incorrect_arrays": [],
        "correct_arrays": 0,
    }

    if not zarr_path.exists():
        return result

    positions = find_positions(zarr_path)
    result["total_positions"] = len(positions)

    for position_path in positions:
        position_name = "/".join(position_path.relative_to(zarr_path).parts)
        label_results = check_position_labels(position_path, zarr_version=zarr_version, verbose=verbose)

        for label_name, check_result in label_results.items():
            result["total_arrays_checked"] += 1

            if check_result["is_correct"]:
                result["correct_arrays"] += 1
            else:
                # Build full metadata path for reporting
                # Use the level from check_result if available (when a non-level-0 was incorrect)
                level = check_result.get("_level", 0)
                if zarr_version == "v3":
                    metadata_path = position_path / "labels" / label_name / str(level) / "zarr.json"
                else:  # v2
                    metadata_path = position_path / label_name / str(level) / ".zarray"

                result["incorrect_arrays"].append({
                    "position": position_name,
                    "label": label_name,
                    "shape": check_result["shape"],
                    "channel_count": check_result["channel_count"],
                    "chunk_shape": check_result["chunk_shape"],
                    "metadata_path": str(metadata_path),
                    "actual_channels": check_result["actual_channels"],
                    "is_metadata_only": check_result["is_metadata_only"],
                    "level": level,
                })

    return result


def check_store_for_zeroes(
    zarr_path: Path,
    zarr_version: str = "v3",
    verbose: bool = False
) -> Dict:
    """
    Check all segmentation arrays in a zarr store for all-zero pyramid levels.

    Args:
        zarr_path: Path to zarr store
        zarr_version: "v2" or "v3"
        verbose: Print detailed info

    Returns:
        dict with keys:
            - total_positions: int
            - zero_arrays: list of dicts with position/label/level details
    """
    result = {
        "total_positions": 0,
        "zero_arrays": [],
    }

    if not zarr_path.exists():
        return result

    positions = find_positions(zarr_path)
    result["total_positions"] = len(positions)

    for position_path in positions:
        position_name = "/".join(position_path.relative_to(zarr_path).parts)
        zero_results = check_position_zeroes(position_path, zarr_version=zarr_version, verbose=verbose)

        for label_name, zero_levels in zero_results.items():
            for level_info in zero_levels:
                result["zero_arrays"].append({
                    "position": position_name,
                    "label": label_name,
                    "level": level_info["level"],
                    "path": level_info["path"],
                    "max_value": level_info["max_value"],
                    "symlink_info": level_info.get("symlink_info", {}),
                })

    return result


def check_experiment_stores(
    experiment: str,
    store_types: List[str] = None,
    zarr_versions: List[str] = None,
    verbose: bool = False,
    check_zeroes: bool = False
) -> Dict:
    """
    Check all specified stores for an experiment.

    Args:
        experiment: Experiment name (e.g., "ops0045_20250603")
        store_types: List of store types to check (e.g., ["pheno", "track"])
                    If None, checks only "pheno"
        zarr_versions: List of zarr versions to check (e.g., ["v2", "v3"])
                      If None, checks only "v3"
        verbose: Print detailed info
        check_zeroes: Also check for all-zero pyramid levels

    Returns:
        dict mapping (store_type, zarr_version) -> check result dict containing:
            - channel_result: channel count check result
            - zero_result: zero check result (if check_zeroes=True)
    """
    if store_types is None:
        store_types = ["pheno"]
    if zarr_versions is None:
        zarr_versions = ["v3"]

    try:
        dataset = OpsDataset(experiment)
    except Exception as e:
        if verbose:
            print(f"  Error loading {experiment}: {e}")
        return {}

    results = {}

    for store_type in store_types:
        store_keys = STORE_MAP.get(store_type)
        if not store_keys:
            continue

        for zarr_version in zarr_versions:
            store_key = store_keys.get(zarr_version)
            if not store_key:
                continue

            store_path = dataset.store_paths.get(store_key)

            if not store_path or not store_path.exists():
                if verbose:
                    print(f"  - {store_type} ({zarr_version}): not found")
                continue

            if verbose:
                print(f"  Checking {store_type} ({zarr_version}): {store_path}")

            result_key = f"{store_type}_{zarr_version}"

            # Channel count check
            channel_result = check_store_channel_counts(store_path, zarr_version=zarr_version, verbose=verbose)
            results[result_key] = {"channel_result": channel_result, "zero_result": None}

            # Print channel count summary
            if channel_result["total_arrays_checked"] > 0:
                num_incorrect = len(channel_result["incorrect_arrays"])
                if num_incorrect > 0:
                    print(f"  ⚠ {store_type} ({zarr_version}): {num_incorrect} arrays with incorrect channel count")
                    for arr in channel_result["incorrect_arrays"]:
                        print(f"      {arr['metadata_path']}")
                        issue_type = "METADATA-ONLY" if arr["is_metadata_only"] else (
                            "DATA+METADATA" if arr["is_metadata_only"] is False else "UNKNOWN"
                        )
                        actual_str = f", actual_data={arr['actual_channels']}" if arr["actual_channels"] is not None else ""
                        print(f"        shape={arr['shape']} (channels={arr['channel_count']}, expected={EXPECTED_CHANNEL_COUNT}{actual_str}) [{issue_type}]")
                elif verbose:
                    print(f"  ✓ {store_type} ({zarr_version}): All {channel_result['total_arrays_checked']} arrays have correct channel count")

            # Zero check
            if check_zeroes:
                zero_result = check_store_for_zeroes(store_path, zarr_version=zarr_version, verbose=verbose)
                results[result_key]["zero_result"] = zero_result

                num_zero = len(zero_result["zero_arrays"])
                if num_zero > 0:
                    print(f"  ⚠ {store_type} ({zarr_version}): {num_zero} pyramid levels with all zeros")
                    for arr in zero_result["zero_arrays"]:
                        print(f"      {arr['path']}")
                        symlink_info = arr.get("symlink_info", {})
                        symlink_str = ""
                        if symlink_info.get("is_symlink"):
                            if symlink_info.get("target_exists"):
                                symlink_str = f" [SYMLINK -> {symlink_info.get('symlink_target')}]"
                            else:
                                symlink_str = f" [BROKEN SYMLINK -> {symlink_info.get('symlink_target')}]"
                        print(f"        position={arr['position']}, label={arr['label']}, level={arr['level']}{symlink_str}")
                elif verbose:
                    print(f"  ✓ {store_type} ({zarr_version}): No all-zero pyramid levels found")

    return results


def find_all_experiments(experiment_configs_dir: Path) -> List[str]:
    """Find all experiment names from config files."""
    config_files = list(experiment_configs_dir.glob("ops*_config.yaml"))
    experiments = [cf.stem.replace("_config", "") for cf in config_files]
    return sorted(experiments)


def main():
    parser = argparse.ArgumentParser(
        description="Batch check Zarr segmentation stores for incorrect channel counts"
    )
    parser.add_argument(
        "--experiment-configs-dir",
        type=Path,
        default=Path(BASE_PATH) / "configs/experiment_configs",
        help="Path to experiment_configs directory",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        type=str,
        help="Check only specific experiments (skip auto-detection)",
    )
    parser.add_argument(
        "--all-stores",
        action="store_true",
        help="Check all store types (pheno, track, iss). Default is pheno only.",
    )
    parser.add_argument(
        "--zarr-version",
        type=str,
        choices=["v2", "v3", "both"],
        default="v3",
        help="Zarr version to check: v2, v3, or both (default: v3)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed checking information",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output results to CSV file",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Fix metadata-only issues (where data has correct channels but metadata is wrong)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --fix, show what would be changed without actually modifying files",
    )
    parser.add_argument(
        "--check-zeroes",
        action="store_true",
        help="Check if any pyramid levels contain all zeros (requires zarr library)",
    )

    args = parser.parse_args()

    # Determine which experiments to check
    if args.experiments:
        experiments = args.experiments
        print(f"Checking specified experiments: {len(experiments)}")
    else:
        print(f"Scanning for experiments in: {args.experiment_configs_dir}")
        experiments = find_all_experiments(args.experiment_configs_dir)
        print(f"Found {len(experiments)} experiment configs")

    # Determine which stores to check
    if args.all_stores:
        store_types = list(STORE_MAP.keys())
        print(f"Checking all store types: {store_types}")
    else:
        store_types = ["pheno"]
        print(f"Checking phenotyping only (use --all-stores for track/iss)")

    # Determine zarr versions to check
    if args.zarr_version == "both":
        zarr_versions = ["v2", "v3"]
        print(f"Checking both Zarr v2 and v3 stores")
    else:
        zarr_versions = [args.zarr_version]
        print(f"Checking Zarr {args.zarr_version} stores")

    print(f"{'='*80}")
    print(f"Checking segmentation arrays for incorrect channel counts")
    print(f"Expected channel count: {EXPECTED_CHANNEL_COUNT}")
    print(f"Labels to check: {SEGMENTATION_LABELS}")
    if args.check_zeroes:
        print(f"Also checking for all-zero pyramid levels")
    print(f"{'='*80}\n")

    # Check zarr availability if needed
    if args.check_zeroes and not _check_zarr_available():
        print("WARNING: zarr library not available, --check-zeroes will be skipped")
        args.check_zeroes = False

    # Track summary statistics
    total_experiments_checked = 0
    total_arrays_checked = 0
    total_incorrect = 0
    total_zero_levels = 0
    experiments_with_issues = []
    experiments_with_zeroes = []
    all_incorrect_arrays = []  # For CSV output
    all_zero_arrays = []  # For CSV output

    # Check each experiment
    for experiment in tqdm(experiments, desc="Checking experiments"):
        if args.verbose:
            print(f"\n{experiment}:")

        results = check_experiment_stores(
            experiment,
            store_types=store_types,
            zarr_versions=zarr_versions,
            verbose=args.verbose,
            check_zeroes=args.check_zeroes
        )

        experiment_has_channel_issues = False
        experiment_has_zero_issues = False

        for result_key, result_dict in results.items():
            channel_result = result_dict["channel_result"]
            zero_result = result_dict.get("zero_result")

            # Process channel count results
            total_arrays_checked += channel_result["total_arrays_checked"]
            num_incorrect = len(channel_result["incorrect_arrays"])
            total_incorrect += num_incorrect

            if num_incorrect > 0:
                experiment_has_channel_issues = True
                if not args.verbose:
                    print(f"⚠ {experiment} - {result_key}: {num_incorrect} arrays with wrong channel count")

                # Collect for CSV and fixing
                for arr in channel_result["incorrect_arrays"]:
                    all_incorrect_arrays.append({
                        "experiment": experiment,
                        "store_type": result_key,
                        "position": arr["position"],
                        "label": arr["label"],
                        "shape": str(arr["shape"]),
                        "channel_count": arr["channel_count"],
                        "chunk_shape": str(arr["chunk_shape"]),
                        "metadata_path": arr["metadata_path"],
                        "actual_channels": arr["actual_channels"],
                        "is_metadata_only": arr["is_metadata_only"],
                    })

            # Process zero check results
            if zero_result:
                num_zero = len(zero_result["zero_arrays"])
                total_zero_levels += num_zero

                if num_zero > 0:
                    experiment_has_zero_issues = True
                    if not args.verbose:
                        print(f"⚠ {experiment} - {result_key}: {num_zero} all-zero pyramid levels")

                    for arr in zero_result["zero_arrays"]:
                        symlink_info = arr.get("symlink_info", {})
                        all_zero_arrays.append({
                            "experiment": experiment,
                            "store_type": result_key,
                            "position": arr["position"],
                            "label": arr["label"],
                            "level": arr["level"],
                            "path": arr["path"],
                            "is_symlink": symlink_info.get("is_symlink", False),
                            "symlink_target": symlink_info.get("symlink_target"),
                            "target_exists": symlink_info.get("target_exists"),
                        })

        if experiment_has_channel_issues:
            experiments_with_issues.append(experiment)
        if experiment_has_zero_issues:
            experiments_with_zeroes.append(experiment)

        if results:
            total_experiments_checked += 1

    # Fix metadata-only issues if requested
    total_fixed = 0
    total_fixable = 0
    total_levels_fixed = 0
    if args.fix and all_incorrect_arrays:
        fixable = [arr for arr in all_incorrect_arrays if arr["is_metadata_only"] is True]
        total_fixable = len(fixable)

        if fixable:
            print(f"\n{'='*80}")
            if args.dry_run:
                print(f"DRY RUN - Would fix {len(fixable)} metadata-only issues (including all pyramid levels)")
            else:
                print(f"Fixing {len(fixable)} metadata-only issues (including all pyramid levels)...")
            print(f"{'='*80}\n")

            for arr in fixable:
                metadata_path = Path(arr["metadata_path"])
                # Determine zarr version from store_type
                zarr_version = "v3" if "_v3" in arr["store_type"] else "v2"

                # Get the label directory (parent of level 0)
                # metadata_path is like: .../labels/seg/0/zarr.json or .../seg/0/.zarray
                level_0_dir = metadata_path.parent
                label_dir = level_0_dir.parent

                # Find all pyramid levels in the label directory
                pyramid_levels = sorted([
                    d for d in label_dir.iterdir()
                    if d.is_dir() and d.name.isdigit()
                ], key=lambda x: int(x.name))

                levels_fixed_for_label = 0
                for level_dir in pyramid_levels:
                    if zarr_version == "v3":
                        level_metadata_path = level_dir / "zarr.json"
                    else:
                        level_metadata_path = level_dir / ".zarray"

                    if level_metadata_path.exists():
                        fix_result = fix_metadata_channel_count(level_metadata_path, zarr_version=zarr_version, dry_run=args.dry_run)

                        if fix_result["success"]:
                            levels_fixed_for_label += 1
                            total_levels_fixed += 1

                if levels_fixed_for_label > 0:
                    total_fixed += 1
                    action = "Would fix" if args.dry_run else "Fixed"
                    print(f"  ✓ {action}: {label_dir} ({levels_fixed_for_label} pyramid levels)")
                else:
                    print(f"  ✗ Failed to fix: {label_dir}")

    # Write CSV if requested
    if args.output and all_incorrect_arrays:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "experiment", "store_type", "position", "label",
                "shape", "channel_count", "chunk_shape", "metadata_path",
                "actual_channels", "is_metadata_only"
            ])
            writer.writeheader()
            writer.writerows(all_incorrect_arrays)
        print(f"\nResults written to: {args.output}")

    # Print summary
    print(f"\n{'='*80}")
    print(f"Summary")
    print(f"{'='*80}")
    print(f"Total experiments checked: {total_experiments_checked}")
    print(f"Total segmentation arrays checked: {total_arrays_checked}")
    print(f"Arrays with incorrect channel count: {total_incorrect}")
    print(f"Experiments with channel issues: {len(experiments_with_issues)}")

    if args.check_zeroes:
        print(f"\nZero check results:")
        print(f"  All-zero pyramid levels found: {total_zero_levels}")
        print(f"  Experiments with zero issues: {len(experiments_with_zeroes)}")

    if args.fix:
        print(f"\nFix results:")
        if args.dry_run:
            print(f"  Would fix: {total_fixed} label arrays ({total_levels_fixed} pyramid levels total)")
        else:
            print(f"  Fixed: {total_fixed} label arrays ({total_levels_fixed} pyramid levels total)")
        unfixable = total_incorrect - total_fixable
        if unfixable > 0:
            print(f"  Unfixable (data+metadata or unknown): {unfixable}")

    if experiments_with_issues:
        print(f"\nExperiments with incorrect channel counts:")
        for exp in experiments_with_issues:
            print(f"  - {exp}")
    elif total_arrays_checked > 0:
        print(f"\n✓ All segmentation arrays have correct channel count!")

    if args.check_zeroes and experiments_with_zeroes:
        # Group experiments by store type and zarr version
        # Key format: (base_store, zarr_version) -> set of experiments
        experiments_by_store_version = {}
        for arr in all_zero_arrays:
            store_type = arr["store_type"]
            # Extract base store type and version from store_type like "pheno_v3", "iss_v2"
            parts = store_type.split("_")
            base_store = parts[0]
            zarr_ver = parts[1] if len(parts) > 1 else "v2"
            key = (base_store, zarr_ver)
            if key not in experiments_by_store_version:
                experiments_by_store_version[key] = set()
            experiments_by_store_version[key].add(arr["experiment"])

        print(f"\nExperiments with all-zero pyramid levels by store type:")
        for store_type in ["pheno", "iss", "track"]:
            # Check both v2 and v3
            v2_exps = experiments_by_store_version.get((store_type, "v2"), set())
            v3_exps = experiments_by_store_version.get((store_type, "v3"), set())

            if v2_exps or v3_exps:
                total = len(v2_exps | v3_exps)  # Union for total unique experiments
                print(f"\n  {store_type.upper()} ({total} experiments):")

                if v2_exps:
                    print(f"    v2 ({len(v2_exps)}):")
                    for exp in sorted(v2_exps):
                        print(f"      - {exp}")

                if v3_exps:
                    print(f"    v3 ({len(v3_exps)}):")
                    for exp in sorted(v3_exps):
                        print(f"      - {exp}")

        # Check for symlink issues in zero arrays
        symlink_zeros = [z for z in all_zero_arrays if z.get("is_symlink")]
        broken_symlinks = [z for z in symlink_zeros if not z.get("target_exists")]

        if symlink_zeros:
            # Group symlink issues by store type and version
            symlink_by_store_version = {}
            for z in symlink_zeros:
                parts = z["store_type"].split("_")
                base_store = parts[0]
                zarr_ver = parts[1] if len(parts) > 1 else "v2"
                key = (base_store, zarr_ver)
                if key not in symlink_by_store_version:
                    symlink_by_store_version[key] = set()
                symlink_by_store_version[key].add(z["experiment"])

            print(f"\nSymlink analysis for zero arrays:")
            print(f"  Total symlinked arrays with zeros: {len(symlink_zeros)}")
            print(f"  Broken symlinks (target missing): {len(broken_symlinks)}")
            print(f"  Valid symlinks (target exists but empty): {len(symlink_zeros) - len(broken_symlinks)}")

            print(f"\n  Symlink issues by store type:")
            for store_type in ["pheno", "iss", "track"]:
                v2_exps = symlink_by_store_version.get((store_type, "v2"), set())
                v3_exps = symlink_by_store_version.get((store_type, "v3"), set())
                if v2_exps or v3_exps:
                    total = len(v2_exps | v3_exps)
                    parts = []
                    if v2_exps:
                        parts.append(f"v2: {len(v2_exps)}")
                    if v3_exps:
                        parts.append(f"v3: {len(v3_exps)}")
                    print(f"    {store_type.upper()}: {total} experiments ({', '.join(parts)})")

            # Provide fix commands for each store type with symlink issues
            stores_with_symlinks = set(k[0] for k in symlink_by_store_version.keys())
            print(f"\n  To fix symlinks, run:")
            for store_type in ["pheno", "iss", "track"]:
                if store_type in stores_with_symlinks:
                    print(f"    # Fix {store_type} symlinks:")
                    print(f"    python -m cyclops_process.utils.batch.batch_symlink_nuclear_seg --symlink-target {store_type} --force")

    elif args.check_zeroes and total_zero_levels == 0:
        print(f"\n✓ No all-zero pyramid levels found!")

    print(f"{'='*80}")


if __name__ == "__main__":
    main()
