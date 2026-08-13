"""
Cell Painting Linking Pipeline
==============================

Links cells across multiple imaging modalities for cell painting experiments:
- CP1 (Cell Painting Part 1) - Hoechst nuclear segmentation
- CP2 (Cell Painting Part 2) - Hoechst nuclear segmentation
- Phenotyping (20x live cell imaging) - nuclear segmentation
- ISS (In Situ Sequencing) - bc_segmentation

Linking strategy:
1. CP1 <-> CP2: Nearest neighbor between CP1_nuclear_seg and CP2_nuclear_seg centroids
2. CP1 <-> Pheno: Nearest neighbor between CP1_nuclear_seg and nuclear_seg centroids
3. CP1 <-> ISS: Nearest neighbor between CP1_nuclear_seg and ISS bc_segmentation centroids
4. Merge with existing phenotyping links (from link_calls_tracks) to get barcodes/genes

Output CSV format extends the standard linked_results format with cell painting columns.

Usage:
    python -m cyclops_process.data.link_cell_painting --experiment ops0094 --wells A/1/0 A/2/0 A/3/0
    python -m cyclops_process.data.link_cell_painting -e 94  # shorthand
"""

import argparse
import numpy as np
import pandas as pd
import zarr
from pathlib import Path
from scipy.spatial import KDTree
from skimage.measure import regionprops
from tqdm import tqdm
from typing import Optional

import sys
import os
sys.path.insert(0, os.getcwd())

from ops_utils.data.experiment import OpsDataset


# =============================================================================
# Configuration
# =============================================================================

# Pyramid level for segmentation processing (level 2 = 4x downsampled)
# This dramatically reduces memory usage while maintaining sufficient precision for linking
# Level 2 is 4x downsampled from full res
PYRAMID_LEVEL = 2
DOWNSAMPLE_FACTOR = 4  # Pyramid level 2 = 4x downsampled from full res

# Maximum distance (in pixels) for nearest neighbor matching
# CP1/CP2/Pheno distances are at pyramid level 2 (4x downsampled)
MAX_MATCH_DISTANCE_CP1_CP2 = 10  # CP1 and CP2 should be well-aligned (~50px at full res)
MAX_MATCH_DISTANCE_CP1_PHENO = 50  # Pheno matching with shape weighting - larger radius (~200px at full res)
# ISS distance is in ISS space (4x downsampled)
MAX_MATCH_DISTANCE_CP1_ISS = 20  # ISS matching - slightly looser for registration tolerance


# =============================================================================
# Mode configuration: CP (2 rounds) vs 4i (5 rounds)
# =============================================================================
# Same pipeline, different label names + output CSV. Internal centroid columns
# stay as cp1/cp2 regardless of mode to minimize code churn — only the final
# output columns are renamed to the mode-specific names.
MODE_CONFIG = {
    "cp": {
        "primary_label": "CP1_nuclear_seg",
        "secondary_labels": ["CP2_nuclear_seg"],
        "cell_seg_label": "cp_cell_seg",
        "seg_id_col": "cp_cell_seg_id",
        "bbox_col": "cp_bbox",
        "output_template": "{well_short}_linked_pheno_iss_cp.csv",
        "cache_subdir": "cp_links",
    },
    "4i": {
        "primary_label": "4i_R1_nuclear_seg",
        "secondary_labels": [
            "4i_R2_nuclear_seg",
            "4i_R3_nuclear_seg",
            "4i_R4_nuclear_seg",
            "4i_R5_nuclear_seg",
        ],
        "cell_seg_label": "4i_cell_seg",
        "seg_id_col": "4i_segmentation_id",
        "bbox_col": "4i_bbox",
        "output_template": "{well_short}_linked_pheno_iss_4i.csv",
        "cache_subdir": "four_i_links",
    },
}


# =============================================================================
# Utility Functions
# =============================================================================

def get_centroids_from_segmentation(seg_array: np.ndarray, desc: str = None) -> pd.DataFrame:
    """
    Extract cell centroids from a 2D segmentation array (in-memory version).

    Args:
        seg_array: 2D numpy array of segmentation labels
        desc: Optional description for progress bar

    Returns DataFrame with columns: label, y_centroid, x_centroid
    """
    props = regionprops(seg_array)
    data = []
    iterator = tqdm(props, desc=desc, leave=False) if desc and len(props) > 1000 else props
    for prop in iterator:
        data.append({
            "label": prop.label,
            "y_centroid": prop.centroid[0],
            "x_centroid": prop.centroid[1],
        })
    return pd.DataFrame(data)


def _process_chunk(zarr_array, chunk_idx: int, n_chunks_x: int, chunk_size: int, h: int, w: int, is_5d: bool) -> dict:
    """
    Process a single chunk and return label statistics.

    Returns dict mapping label -> {"sum_y": float, "sum_x": float, "count": int}
    count is the pixel count (area) which is used for shape similarity matching.
    """
    cy = chunk_idx // n_chunks_x
    cx = chunk_idx % n_chunks_x

    y0 = cy * chunk_size
    y1 = min(y0 + chunk_size, h)
    x0 = cx * chunk_size
    x1 = min(x0 + chunk_size, w)

    # Load chunk
    if is_5d:
        chunk = np.asarray(zarr_array[0, 0, 0, y0:y1, x0:x1])
    else:
        chunk = np.asarray(zarr_array[y0:y1, x0:x1])

    # Find unique labels in chunk (excluding background 0)
    unique_labels = np.unique(chunk)
    unique_labels = unique_labels[unique_labels > 0]

    chunk_stats = {}
    for label in unique_labels:
        mask = chunk == label
        ys, xs = np.where(mask)

        # Add global offset - count is used for both centroid calculation and as area metric
        chunk_stats[int(label)] = {
            "sum_y": float((ys + y0).sum()),
            "sum_x": float((xs + x0).sum()),
            "count": int(len(ys)),  # This accumulates to total cell area
        }

    return chunk_stats


def get_centroids_from_zarr_chunked(
    zarr_array,
    chunk_size: int = 2048,
    desc: str = None,
    n_jobs: int = None,
) -> pd.DataFrame:
    """
    Extract cell centroids from a zarr array using parallel chunk-wise processing.

    This avoids loading the entire segmentation into memory by:
    1. Processing the array in chunks (parallelized with joblib)
    2. Accumulating pixel sums and counts per label
    3. Computing centroids from accumulated statistics

    Args:
        zarr_array: 2D zarr array (or 5D with singleton T,C,Z dims)
        chunk_size: Size of chunks to process (default: 2048)
        desc: Optional description for progress bar
        n_jobs: Number of parallel workers (default: auto-detect)

    Returns DataFrame with columns: label, y_centroid, x_centroid
    """
    from collections import defaultdict
    from joblib import Parallel, delayed
    from ops_utils.hpc.resource_manager import get_optimal_workers

    # Determine workers
    if n_jobs is None:
        n_jobs = get_optimal_workers(use_gpu=False, verbose=False)

    # Handle 5D arrays (T, C, Z, Y, X) - squeeze to 2D
    is_5d = len(zarr_array.shape) == 5
    if is_5d:
        h, w = zarr_array.shape[-2:]
    else:
        h, w = zarr_array.shape

    # Calculate number of chunks
    n_chunks_y = (h + chunk_size - 1) // chunk_size
    n_chunks_x = (w + chunk_size - 1) // chunk_size
    total_chunks = n_chunks_y * n_chunks_x

    # Process chunks in parallel with tqdm progress bar
    pbar_desc = desc.strip() if desc else "Chunks"
    chunk_results = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(_process_chunk)(
            zarr_array, chunk_idx, n_chunks_x, chunk_size, h, w, is_5d
        )
        for chunk_idx in range(total_chunks)
    )

    # Wrap with tqdm for progress tracking
    chunk_results = list(tqdm(
        chunk_results,
        total=total_chunks,
        desc=f"    {pbar_desc}",
        leave=False,
        ncols=80,
    ))

    # Merge results from all chunks
    label_stats = defaultdict(lambda: {"sum_y": 0.0, "sum_x": 0.0, "count": 0})
    for chunk_stats in chunk_results:
        for label, stats in chunk_stats.items():
            label_stats[label]["sum_y"] += stats["sum_y"]
            label_stats[label]["sum_x"] += stats["sum_x"]
            label_stats[label]["count"] += stats["count"]

    # Compute centroids and include area for shape matching
    data = []
    for label, stats in sorted(label_stats.items()):
        if stats["count"] > 0:
            data.append({
                "label": int(label),
                "y_centroid": stats["sum_y"] / stats["count"],
                "x_centroid": stats["sum_x"] / stats["count"],
                "area": stats["count"],  # Cell area in pixels for shape similarity
            })

    return pd.DataFrame(data)


def get_centroids_cached(
    zarr_array,
    cache_path: Path,
    desc: str = None,
    n_jobs: int = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Get centroids with caching support.

    If cache exists and force_recompute is False, loads from cache.
    Otherwise computes centroids and saves to cache.

    Args:
        zarr_array: zarr array to extract centroids from
        cache_path: Path to cache file (parquet format)
        desc: Description for progress bar
        n_jobs: Number of parallel workers
        force_recompute: If True, recompute even if cache exists

    Returns:
        DataFrame with columns: label, y_centroid, x_centroid
    """
    if cache_path.exists() and not force_recompute:
        return pd.read_parquet(cache_path)

    # Compute centroids
    centroids = get_centroids_from_zarr_chunked(zarr_array, desc=desc, n_jobs=n_jobs)

    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    centroids.to_parquet(cache_path, index=False)

    return centroids


def nearest_neighbor_match(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    source_y_col: str,
    source_x_col: str,
    target_y_col: str,
    target_x_col: str,
    max_distance: float,
) -> pd.DataFrame:
    """
    Find nearest neighbor matches between source and target points.

    Returns DataFrame with columns:
        - source_idx: index in source_df
        - target_idx: index in target_df
        - distance: Euclidean distance between matched points
    """
    if len(source_df) == 0 or len(target_df) == 0:
        return pd.DataFrame(columns=["source_idx", "target_idx", "distance"])

    source_points = source_df[[source_y_col, source_x_col]].values
    target_points = target_df[[target_y_col, target_x_col]].values

    # Build KDTree on target points
    tree = KDTree(target_points)

    # Query nearest neighbor for each source point
    distances, indices = tree.query(source_points, k=1)

    # Filter by max distance
    valid_mask = distances <= max_distance

    matches = pd.DataFrame({
        "source_idx": np.arange(len(source_df))[valid_mask],
        "target_idx": indices[valid_mask],
        "distance": distances[valid_mask],
    })

    return matches


def shape_weighted_match(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    source_y_col: str,
    source_x_col: str,
    target_y_col: str,
    target_x_col: str,
    source_area_col: str,
    target_area_col: str,
    max_distance: float,
    k_neighbors: int = 5,
    distance_weight: float = 0.6,
    shape_weight: float = 0.4,
) -> pd.DataFrame:
    """
    Find best matches using combined distance and shape similarity scoring.

    For each source cell, finds k nearest neighbors in target, then scores each
    candidate based on:
    - Distance score: closer = better (normalized by max_distance)
    - Shape score: similar area = better (using area ratio)

    Final score = distance_weight * dist_score + shape_weight * shape_score

    Args:
        source_df: Source cells DataFrame
        target_df: Target cells DataFrame
        source_y_col, source_x_col: Column names for source coordinates
        target_y_col, target_x_col: Column names for target coordinates
        source_area_col: Column name for source cell area
        target_area_col: Column name for target cell area
        max_distance: Maximum distance threshold
        k_neighbors: Number of nearest neighbors to consider
        distance_weight: Weight for distance in combined score (default 0.6)
        shape_weight: Weight for shape similarity in combined score (default 0.4)

    Returns DataFrame with columns:
        - source_idx: index in source_df
        - target_idx: index in target_df
        - distance: Euclidean distance between matched points
        - score: Combined distance + shape similarity score (higher = better)
    """
    if len(source_df) == 0 or len(target_df) == 0:
        return pd.DataFrame(columns=["source_idx", "target_idx", "distance", "score"])

    source_points = source_df[[source_y_col, source_x_col]].values
    target_points = target_df[[target_y_col, target_x_col]].values

    # Get areas (use 1.0 as fallback if area column missing)
    if source_area_col in source_df.columns:
        source_areas = source_df[source_area_col].values
    else:
        source_areas = np.ones(len(source_df))

    if target_area_col in target_df.columns:
        target_areas = target_df[target_area_col].values
    else:
        target_areas = np.ones(len(target_df))

    # Build KDTree on target points
    tree = KDTree(target_points)

    # Query k nearest neighbors for each source point
    k = min(k_neighbors, len(target_df))
    distances, indices = tree.query(source_points, k=k)

    # Handle single neighbor case (distances/indices are 1D)
    if k == 1:
        distances = distances.reshape(-1, 1)
        indices = indices.reshape(-1, 1)

    # For each source, find best match among k neighbors
    best_matches = []
    for src_idx in range(len(source_df)):
        src_area = source_areas[src_idx]

        best_score = -1
        best_tgt_idx = -1
        best_dist = np.inf

        for neighbor_rank in range(k):
            tgt_idx = indices[src_idx, neighbor_rank]
            dist = distances[src_idx, neighbor_rank]

            # Skip if beyond max distance
            if dist > max_distance:
                continue

            tgt_area = target_areas[tgt_idx]

            # Distance score: 1.0 when dist=0, 0.0 when dist=max_distance
            dist_score = 1.0 - (dist / max_distance)

            # Shape score: area similarity using ratio (1.0 when identical)
            # Use min/max ratio so it's symmetric and bounded [0, 1]
            if src_area > 0 and tgt_area > 0:
                area_ratio = min(src_area, tgt_area) / max(src_area, tgt_area)
            else:
                area_ratio = 0.0
            shape_score = area_ratio

            # Combined score
            combined_score = distance_weight * dist_score + shape_weight * shape_score

            if combined_score > best_score:
                best_score = combined_score
                best_tgt_idx = tgt_idx
                best_dist = dist

        if best_tgt_idx >= 0:
            best_matches.append({
                "source_idx": src_idx,
                "target_idx": best_tgt_idx,
                "distance": best_dist,
                "score": best_score,
            })

    if not best_matches:
        return pd.DataFrame(columns=["source_idx", "target_idx", "distance", "score"])

    matches_df = pd.DataFrame(best_matches)

    # CRITICAL: Deduplicate by target - if multiple sources matched to the same target,
    # keep only the best match (highest score). This ensures 1:1 mapping so that
    # downstream feature extraction doesn't have duplicate cell_ids.
    n_before = len(matches_df)
    if len(matches_df) > 0:
        # Sort by score descending, then drop duplicates keeping first (best) match per target
        matches_df = matches_df.sort_values("score", ascending=False)
        matches_df = matches_df.drop_duplicates(subset=["target_idx"], keep="first")
        matches_df = matches_df.sort_values("source_idx").reset_index(drop=True)
        n_after = len(matches_df)
        if n_before != n_after:
            # This will be printed at the call site with context
            pass  # Removed {n_before - n_after} duplicate target matches

    return matches_df


def load_nuclear_seg_zarr(dataset: OpsDataset, well: str, label_name: str, pyramid_level: int = PYRAMID_LEVEL):
    """Load any nuclear segmentation label from the v3 store."""
    v3_path = dataset.store_paths.get("pheno_assembled_v3")
    if v3_path is None or not v3_path.exists():
        return None, 0, None

    try:
        store = zarr.open(str(v3_path), mode="r")
        label_path = f"{well}/labels/{label_name}/{pyramid_level}"
        if label_path not in store:
            # Fall back to level 0 if requested level doesn't exist
            label_path = f"{well}/labels/{label_name}/0"
            if label_path not in store:
                print(f"  Warning: {label_name} not found")
                return None, 0, None

        zarr_arr = store[label_path]
        shape = zarr_arr.shape[-2:]  # (Y, X)

        # Get max label by sampling corners and center (avoid loading entire array)
        max_label = _estimate_max_label(zarr_arr)

        return zarr_arr, max_label, shape
    except Exception as e:
        print(f"  Error loading {label_name}: {e}")
        return None, 0, None


def load_cp_nuclear_seg_zarr(dataset: OpsDataset, well: str, part: int, pyramid_level: int = PYRAMID_LEVEL):
    """Backwards-compat wrapper: load CP{part}_nuclear_seg."""
    return load_nuclear_seg_zarr(dataset, well, f"CP{part}_nuclear_seg", pyramid_level)


def load_pheno_nuclear_seg_zarr(dataset: OpsDataset, well: str, pyramid_level: int = PYRAMID_LEVEL):
    """
    Load phenotyping nuclear segmentation as zarr array at specified pyramid level.

    Returns:
        Tuple of (zarr_array, max_label, shape) or (None, 0, None) if not found
    """
    v3_path = dataset.store_paths.get("pheno_assembled_v3")
    if v3_path is None or not v3_path.exists():
        return None, 0, None

    try:
        store = zarr.open(str(v3_path), mode="r")
        # Try nuclear_seg first, then fall back to seg
        for label_name in ["nuclear_seg", "seg"]:
            label_path = f"{well}/labels/{label_name}/{pyramid_level}"
            if label_path not in store:
                label_path = f"{well}/labels/{label_name}/0"
            if label_path in store:
                zarr_arr = store[label_path]
                shape = zarr_arr.shape[-2:]
                max_label = _estimate_max_label(zarr_arr)
                return zarr_arr, max_label, shape

        print(f"  Warning: No nuclear segmentation found for phenotyping")
        return None, 0, None
    except Exception as e:
        print(f"  Error loading pheno nuclear seg: {e}")
        return None, 0, None


def load_cell_seg_zarr(dataset: OpsDataset, well: str, label_name: str = "cp_cell_seg"):
    """Load cell segmentation (cp_cell_seg or 4i_cell_seg) at full resolution."""
    v3_path = dataset.store_paths.get("pheno_assembled_v3")
    if v3_path is None or not v3_path.exists():
        return None, False, "pheno_assembled_v3 store not found"

    try:
        store = zarr.open(str(v3_path), mode="r")
        labels_path = f"{well}/labels/{label_name}/0"
        if labels_path not in store:
            return None, False, f"{label_name} label not found at {labels_path}"
        lazy_seg_array = store[labels_path]
        segmentation_array = np.asarray(lazy_seg_array[0, 0, 0, :, :])
        return segmentation_array, True, f"{label_name} (v3)"
    except Exception as e:
        return None, False, f"Error loading {label_name} from v3: {e}"


def load_cp_cell_seg_zarr(dataset: OpsDataset, well: str):
    """Backwards-compat wrapper for cp_cell_seg."""
    return load_cell_seg_zarr(dataset, well, "cp_cell_seg")


def compute_cp_cell_bboxes(
    unified: pd.DataFrame,
    dataset: OpsDataset,
    well: str,
    cell_seg_label: str = "cp_cell_seg",
    bbox_col: str = "cp_bbox",
    seg_id_col: str = "cp_cell_seg_id",
) -> pd.DataFrame:
    """
    Add bounding boxes from cp_cell_seg for each CP1 cell.

    Uses CP1 centroid coordinates (scaled to full res) to look up the cell label
    in cp_cell_seg, then computes bbox from regionprops.

    Args:
        unified: DataFrame with y_cp1, x_cp1 columns (at pyramid level 2)
        dataset: OpsDataset instance
        well: Well identifier

    Returns:
        DataFrame with added 'bbox' and 'cp_cell_seg_id' columns
    """
    # Load cell seg at full resolution
    seg_array, success, message = load_cell_seg_zarr(dataset, well, cell_seg_label)

    if not success:
        print(f"[bbox] Warning: {message}")
        unified[bbox_col] = None
        unified[seg_id_col] = np.nan
        return unified

    print(f"[bbox] ✓ Loaded {message} for {well}, shape={seg_array.shape}")

    # Compute bboxes once using regionprops
    label_props = regionprops(seg_array, cache=False)
    bbox_lookup = {obj.label: obj.bbox for obj in label_props}
    print(f"[bbox] Computed bboxes for {len(bbox_lookup)} cells")

    # Scale CP1 coordinates from pyramid level 2 to full resolution
    # Pyramid level 2 = 4x downsampled, so multiply by 4
    y_fullres = (unified["y_cp1"].values * DOWNSAMPLE_FACTOR).astype(int)
    x_fullres = (unified["x_cp1"].values * DOWNSAMPLE_FACTOR).astype(int)

    # Look up cell label at each coordinate
    h, w = seg_array.shape
    bbox_list = []
    seg_id_list = []

    fallback_bbox_size = 200  # 200x200 pixel box for cells without valid segmentation
    half_size = fallback_bbox_size // 2

    for i in range(len(unified)):
        y, x = int(y_fullres[i]), int(x_fullres[i])

        # Check bounds
        if y < 0 or x < 0 or y >= h or x >= w:
            # Out of bounds - create fallback bbox (ensure Python ints for clean CSV output)
            min_row = int(max(0, y - half_size))
            min_col = int(max(0, x - half_size))
            max_row = int(min(h, y + half_size))
            max_col = int(min(w, x + half_size))
            bbox_list.append((min_row, min_col, max_row, max_col))
            seg_id_list.append(np.nan)
            continue

        seg_id = int(seg_array[y, x])

        if seg_id <= 0 or seg_id not in bbox_lookup:
            # No valid cell at this location - create fallback bbox (ensure Python ints for clean CSV output)
            min_row = int(max(0, y - half_size))
            min_col = int(max(0, x - half_size))
            max_row = int(min(h, y + half_size))
            max_col = int(min(w, x + half_size))
            bbox_list.append((min_row, min_col, max_row, max_col))
            seg_id_list.append(np.nan)
        else:
            bbox_list.append(bbox_lookup[seg_id])
            seg_id_list.append(seg_id)

    unified[bbox_col] = bbox_list
    unified[seg_id_col] = seg_id_list

    n_valid = sum(1 for s in seg_id_list if not (isinstance(s, float) and np.isnan(s)))
    n_fallback = len(seg_id_list) - n_valid
    print(f"[bbox] Assigned {n_valid} valid bboxes, {n_fallback} fallback bboxes")

    return unified


def load_iss_segmentation_zarr(dataset: OpsDataset, well: str):
    """
    Load ISS segmentation as zarr array.

    Note: ISS segmentation is already at tracking resolution (4x downsampled),
    similar to our pyramid level 2. No additional downsampling needed.

    Returns:
        Tuple of (zarr_array, max_label, shape) or (None, 0, None) if not found
    """
    iss_seg_path = dataset.store_paths.get("iss_segmentation")
    if iss_seg_path is None or not iss_seg_path.exists():
        return None, 0, None

    try:
        store = zarr.open(str(iss_seg_path), mode="r")
        zarr_arr = store[f"{well}/0"]
        shape = zarr_arr.shape[-2:]
        max_label = _estimate_max_label(zarr_arr)
        return zarr_arr, max_label, shape
    except Exception as e:
        print(f"  Error loading ISS segmentation: {e}")
        return None, 0, None


def _estimate_max_label(zarr_arr, sample_size: int = 512) -> int:
    """
    Estimate max label by sampling regions of the array.
    This avoids loading the entire array to find the maximum.
    """
    if len(zarr_arr.shape) == 5:
        h, w = zarr_arr.shape[-2:]
        def get_region(y0, y1, x0, x1):
            return np.asarray(zarr_arr[0, 0, 0, y0:y1, x0:x1])
    else:
        h, w = zarr_arr.shape[-2:]
        def get_region(y0, y1, x0, x1):
            return zarr_arr[y0:y1, x0:x1]

    max_label = 0

    # Sample corners and center
    regions = [
        (0, min(sample_size, h), 0, min(sample_size, w)),  # top-left
        (0, min(sample_size, h), max(0, w - sample_size), w),  # top-right
        (max(0, h - sample_size), h, 0, min(sample_size, w)),  # bottom-left
        (max(0, h - sample_size), h, max(0, w - sample_size), w),  # bottom-right
        (h//2 - sample_size//2, h//2 + sample_size//2, w//2 - sample_size//2, w//2 + sample_size//2),  # center
    ]

    for y0, y1, x0, x1 in regions:
        y0, y1 = max(0, y0), min(h, y1)
        x0, x1 = max(0, x0), min(w, x1)
        if y1 > y0 and x1 > x0:
            region = get_region(y0, y1, x0, x1)
            region_max = int(np.max(region))
            max_label = max(max_label, region_max)

    return max_label


def load_seg_as_numpy_for_preview(zarr_arr, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """
    Load a crop of segmentation as numpy array (for preview generation).
    """
    if zarr_arr is None:
        return None
    if len(zarr_arr.shape) == 5:
        return np.asarray(zarr_arr[0, 0, 0, y0:y1, x0:x1])
    else:
        return np.asarray(zarr_arr[y0:y1, x0:x1])


def _has_tracking_for_experiment(dataset: OpsDataset, well: str) -> bool:
    """Mirror view_dask's _has_tracking_for_pos: check skip_track flag + ops>=69 A3 rule."""
    exp_config_path = dataset.config_paths.get("exp_config")
    if exp_config_path and Path(exp_config_path).exists():
        try:
            import yaml as _yaml
            with open(exp_config_path) as _f:
                _exp_cfg = _yaml.safe_load(_f) or {}
            if _exp_cfg.get("auto_register_params", {}).get("skip_track", False):
                return False
        except Exception:
            pass
    try:
        well_token = well.replace("/", "")
        well_token_short = well_token[:2] if len(well_token) >= 2 else well_token
        ops_num = int(dataset.experiment.split("_")[0].replace("ops", ""))
        if ops_num >= 69 and well_token_short == "A3":
            return False
    except (ValueError, IndexError):
        pass
    return True


def transform_cp1_to_iss(
    cp1_points: np.ndarray,
    dataset: OpsDataset,
    well: str,
) -> np.ndarray:
    """
    Transform pheno-pyramid-level-2 coordinates to ISS segmentation space.

    Mirrors the composition view_dask uses for ISS→Pheno overlays, inverted for
    our pheno→ISS direction:
      - skip_track=True  (no tracking): auto_iss_register stores Pheno→ISS; apply it directly.
      - with tracking:    auto_iss_register stores ISS→Track, auto_pheno_register stores Track→Pheno.
        view_dask composes T_iss_to_pheno = T_pheno @ inv(T_iss). Inverted for our direction:
        T_pheno_to_iss = T_iss @ inv(T_pheno).

    Args:
        cp1_points: Nx2 array of (y, x) coordinates at pyramid level 2 (4x downsampled)
        dataset: OpsDataset instance
        well: Well identifier (e.g., "A/1/0")

    Returns:
        Nx2 array of (y, x) coordinates in ISS segmentation space
    """
    # Locate ISS registration YAML
    iss_reg_yaml = dataset.append_well("auto_iss_register", well)
    if not Path(iss_reg_yaml).exists():
        iss_reg_yaml = dataset.config_paths.get("auto_iss_register")
        if iss_reg_yaml is None or not Path(iss_reg_yaml).exists():
            iss_reg_yaml = dataset.config_paths.get("iss_seg_register")

    if iss_reg_yaml is None or not Path(str(iss_reg_yaml)).exists():
        print(f"  Warning: No ISS registration found, returning input coords")
        return cp1_points

    try:
        from stitch.registration.register import read_transform_biahub
        T_iss_raw = np.asarray(read_transform_biahub(str(iss_reg_yaml)))

        has_tracking = _has_tracking_for_experiment(dataset, well)

        if has_tracking:
            # Need auto_pheno_register (Track→Pheno) to compose
            pheno_reg_yaml = dataset.append_well("auto_pheno_register", well)
            if not Path(pheno_reg_yaml).exists():
                pheno_reg_yaml = dataset.config_paths.get("auto_pheno_register")
            if pheno_reg_yaml is None or not Path(str(pheno_reg_yaml)).exists():
                print(f"  Warning: has_tracking=True but no auto_pheno_register — falling back to direct ISS transform")
                T_pheno_to_iss_4x4 = T_iss_raw
            else:
                T_track_to_pheno = np.asarray(read_transform_biahub(str(pheno_reg_yaml)))
                # Pheno → ISS = T_iss_raw @ inv(T_track_to_pheno)
                T_pheno_to_iss_4x4 = T_iss_raw @ np.linalg.inv(T_track_to_pheno)
        else:
            # skip_track: yaml stores Pheno→ISS; apply directly
            T_pheno_to_iss_4x4 = T_iss_raw

        # Reduce 4x4 ZYX affine → 3x3 YX affine (drop Z row/col at index 1)
        T_2d = np.identity(3)
        T_2d[0:2, 0:2] = T_pheno_to_iss_4x4[1:3, 1:3]
        T_2d[0:2, 2] = T_pheno_to_iss_4x4[1:3, 3]

        # Apply to (y, x) points. Homogeneous: [y, x, 1]
        pts_h = np.column_stack([cp1_points, np.ones(len(cp1_points))])
        iss_points = (T_2d @ pts_h.T).T[:, :2]
        return iss_points
    except Exception as e:
        print(f"  Warning: Could not apply ISS transform: {e}")
        return cp1_points


# =============================================================================
# Preview Visualization
# =============================================================================

def generate_linking_preview(
    experiment: str,
    well: str,
    crop_size: int = 1024,
    center_offset: tuple = None,
    output_path: str = None,
    mode: str = "cp",
) -> Optional[Path]:
    """
    Generate a preview image showing the actual linked data from the saved CSV.

    Loads the saved cell_painting_linked CSV and visualizes the matches
    within a crop region. Each panel shows two segmentations overlaid with
    different colors, with arrows connecting matched cell centroids.

    Creates 3 overlay panels (if data available):
    - CP1 (cyan) + CP2 (magenta) overlay with arrows
    - CP1 (cyan) + Pheno (yellow) overlay with arrows
    - CP1 (cyan) + ISS (red) overlay with arrows

    Args:
        experiment: Experiment name
        well: Well identifier (e.g., "A/1/0")
        crop_size: Size of center crop in pixels (default: 1024)
        center_offset: Optional (y, x) offset from center. If None, uses image center.
        output_path: Optional output path. If None, saves to debug_output folder.

    Returns:
        Path to saved preview image, or None if failed
    """
    import matplotlib.pyplot as plt

    dataset = OpsDataset(experiment)

    print(f"\n{'='*60}")
    print(f"Generating linking preview: {experiment} - {well}")
    print(f"{'='*60}")

    # -------------------------------------------------------------------------
    # Load the saved linking CSV
    # -------------------------------------------------------------------------
    print("\n[1] Loading saved linking data...")

    mcfg = MODE_CONFIG[mode]
    _parts = well.split("/")
    csv_path = dataset.results_fast / mcfg["output_template"].format(
        well_safe=well.replace("/", "_"),
        well_short=f"{_parts[0]}{_parts[1]}",
    )
    if not csv_path.exists():
        print(f"  ERROR: Linked CSV not found at {csv_path}")
        print("  Run linking first before generating preview.")
        return None

    linked_df = pd.read_csv(csv_path)
    print(f"  Loaded {len(linked_df)} cells from {csv_path.name}")

    # Debug: show coordinate ranges to verify scaling
    if "y_cp1" in linked_df.columns:
        print(f"  CSV y_cp1 range: [{linked_df['y_cp1'].min():.0f} - {linked_df['y_cp1'].max():.0f}]")
        print(f"  CSV x_cp1 range: [{linked_df['x_cp1'].min():.0f} - {linked_df['x_cp1'].max():.0f}]")

    # -------------------------------------------------------------------------
    # Load segmentations for visualization
    # -------------------------------------------------------------------------
    print("\n[2] Loading segmentations...")

    # Get zarr arrays (lazy loading) — primary + first secondary label from the mode config
    cp1_zarr, _, cp1_shape = load_nuclear_seg_zarr(dataset, well, mcfg["primary_label"], PYRAMID_LEVEL)
    sec_label = mcfg["secondary_labels"][0] if mcfg["secondary_labels"] else None
    cp2_zarr, _, _ = load_nuclear_seg_zarr(dataset, well, sec_label, PYRAMID_LEVEL) if sec_label else (None, 0, None)
    pheno_zarr, _, pheno_shape = load_pheno_nuclear_seg_zarr(dataset, well, pyramid_level=PYRAMID_LEVEL)
    iss_zarr, _, iss_shape = load_iss_segmentation_zarr(dataset, well)

    if cp1_zarr is None:
        print(f"  ERROR: {mcfg['primary_label']} segmentation not found")
        return None

    # Determine crop region (center of CP1 image at pyramid level 2)
    h, w = cp1_shape
    if center_offset is None:
        cy, cx = h // 2, w // 2
    else:
        cy, cx = h // 2 + center_offset[0], w // 2 + center_offset[1]

    y0 = max(0, cy - crop_size // 2)
    y1 = min(h, cy + crop_size // 2)
    x0 = max(0, cx - crop_size // 2)
    x1 = min(w, cx + crop_size // 2)

    print(f"  Crop region: y=[{y0}:{y1}], x=[{x0}:{x1}] (at pyramid level {PYRAMID_LEVEL})")

    # Load crop regions only
    cp1_crop = load_seg_as_numpy_for_preview(cp1_zarr, y0, y1, x0, x1)
    cp2_crop = load_seg_as_numpy_for_preview(cp2_zarr, y0, y1, x0, x1) if cp2_zarr is not None else None
    pheno_crop = load_seg_as_numpy_for_preview(pheno_zarr, y0, y1, x0, x1) if pheno_zarr is not None else None

    # ISS segmentation is in ISS native space, need to transform crop bounds
    # from CP1 pyramid level 2 space to ISS space using the same transform
    iss_crop = None
    iss_y0, iss_y1, iss_x0, iss_x1 = 0, 0, 0, 0
    if iss_zarr is not None and iss_shape is not None:
        # Transform the four corners of the crop region to ISS space
        crop_corners = np.array([
            [y0, x0],  # top-left
            [y0, x1],  # top-right
            [y1, x0],  # bottom-left
            [y1, x1],  # bottom-right
        ])
        iss_corners = transform_cp1_to_iss(crop_corners, dataset, well)

        # Get bounding box in ISS space
        iss_y0 = int(np.floor(iss_corners[:, 0].min()))
        iss_y1 = int(np.ceil(iss_corners[:, 0].max()))
        iss_x0 = int(np.floor(iss_corners[:, 1].min()))
        iss_x1 = int(np.ceil(iss_corners[:, 1].max()))

        # Clamp to valid range
        iss_h, iss_w = iss_shape
        iss_y0, iss_y1 = max(0, iss_y0), min(iss_h, iss_y1)
        iss_x0, iss_x1 = max(0, iss_x0), min(iss_w, iss_x1)

        if iss_y1 > iss_y0 and iss_x1 > iss_x0:
            iss_crop = load_seg_as_numpy_for_preview(iss_zarr, iss_y0, iss_y1, iss_x0, iss_x1)
            print(f"  ISS crop region: y=[{iss_y0}:{iss_y1}], x=[{iss_x0}:{iss_x1}]")

    # -------------------------------------------------------------------------
    # Filter linked data to crop region (using CP1 coordinates)
    # -------------------------------------------------------------------------
    print("\n[3] Filtering to crop region...")

    # The saved CSV has y_cp1/x_cp1 at full resolution (scaled 4x from pyramid level 2)
    # Scale crop bounds to full resolution to match
    scale = DOWNSAMPLE_FACTOR  # 4x from pyramid level 2
    y0_full, y1_full = y0 * scale, y1 * scale
    x0_full, x1_full = x0 * scale, x1 * scale

    print(f"  Crop bounds (pyr lvl 2): y=[{y0}:{y1}], x=[{x0}:{x1}]")
    print(f"  Crop bounds (full res):  y=[{y0_full}:{y1_full}], x=[{x0_full}:{x1_full}]")

    # Filter to cells whose CP1 centroid is in the crop region (full res coords)
    crop_df = linked_df[
        (linked_df["y_cp1"] >= y0_full) & (linked_df["y_cp1"] < y1_full) &
        (linked_df["x_cp1"] >= x0_full) & (linked_df["x_cp1"] < x1_full)
    ].copy()

    # Convert to crop-relative coordinates (scaled back to pyramid level 2 for display)
    crop_df["y_cp1_crop"] = (crop_df["y_cp1"] - y0_full) / scale
    crop_df["x_cp1_crop"] = (crop_df["x_cp1"] - x0_full) / scale

    print(f"  {len(crop_df)} cells in crop region")

    # Build match lists from the actual data
    stats = {"cp1_in_crop": len(crop_df)}

    # CP2 matches (cells with valid cp2_label)
    # Note: y_cp2/x_cp2 are at full resolution, need to scale to pyramid level 2 for display
    crop_height = y1 - y0  # pyramid level 2 crop size
    crop_width = x1 - x0
    cp2_matches = []
    if "cp2_label" in crop_df.columns and "y_cp2" in crop_df.columns:
        cp2_linked = crop_df[crop_df["cp2_label"].notna()]
        for _, row in cp2_linked.iterrows():
            # Scale from full res to pyramid level 2
            cp2_y_crop = (row["y_cp2"] - y0_full) / scale
            cp2_x_crop = (row["x_cp2"] - x0_full) / scale
            if 0 <= cp2_y_crop < crop_height and 0 <= cp2_x_crop < crop_width:
                cp2_matches.append({
                    "cp1_y": row["y_cp1_crop"], "cp1_x": row["x_cp1_crop"],
                    "cp2_y": cp2_y_crop, "cp2_x": cp2_x_crop,
                })
        stats["cp2_in_crop"] = len(cp2_linked)
        stats["cp1_cp2_matched"] = len(cp2_matches)

    # Pheno matches (cells with valid pheno_label)
    # Note: y_pheno_centroid/x_pheno_centroid are at full resolution
    pheno_matches = []
    if "pheno_label" in crop_df.columns and "y_pheno_centroid" in crop_df.columns:
        pheno_linked = crop_df[crop_df["pheno_label"].notna()]
        for _, row in pheno_linked.iterrows():
            # Scale from full res to pyramid level 2
            pheno_y_crop = (row["y_pheno_centroid"] - y0_full) / scale
            pheno_x_crop = (row["x_pheno_centroid"] - x0_full) / scale
            if 0 <= pheno_y_crop < crop_height and 0 <= pheno_x_crop < crop_width:
                pheno_matches.append({
                    "cp1_y": row["y_cp1_crop"], "cp1_x": row["x_cp1_crop"],
                    "pheno_y": pheno_y_crop, "pheno_x": pheno_x_crop,
                })
        stats["pheno_in_crop"] = len(pheno_linked)
        stats["cp1_pheno_matched"] = len(pheno_matches)

    # ISS matches (cells with valid iss_label)
    iss_matches = []
    iss_scale_factor = 1.0
    if "iss_label" in crop_df.columns and "y_iss_centroid" in crop_df.columns:
        iss_linked = crop_df[crop_df["iss_label"].notna()]
        # Calculate scale factor for ISS -> CP1 coordinate space
        if iss_crop is not None:
            iss_scale_factor = crop_size / max(1, (iss_y1 - iss_y0))
        for _, row in iss_linked.iterrows():
            iss_y_crop = row["y_iss_centroid"] - iss_y0
            iss_x_crop = row["x_iss_centroid"] - iss_x0
            if 0 <= iss_y_crop < (iss_y1 - iss_y0) and 0 <= iss_x_crop < (iss_x1 - iss_x0):
                iss_matches.append({
                    "cp1_y": row["y_cp1_crop"], "cp1_x": row["x_cp1_crop"],
                    "iss_y": iss_y_crop * iss_scale_factor,  # Scale to CP1 space
                    "iss_x": iss_x_crop * iss_scale_factor,
                })
        stats["iss_in_crop"] = len(iss_linked)
        stats["cp1_iss_matched"] = len(iss_matches)

    print(f"  CP2 matches: {len(cp2_matches)}, Pheno matches: {len(pheno_matches)}, ISS matches: {len(iss_matches)}")

    # -------------------------------------------------------------------------
    # Create visualization - 2 rows:
    # Row 1: Source segmentations with distinct colors (CP1 cyan, target magenta/yellow/red)
    # Row 2: Matched cells colored by match ID (same color = same cell)
    # -------------------------------------------------------------------------
    print("\n[4] Creating visualization...")

    # Define colors for each modality (RGBA)
    COLOR_CP1 = (0.0, 0.8, 0.8, 0.7)    # Cyan for CP1 (source)
    COLOR_CP2 = (0.9, 0.0, 0.9, 0.7)    # Magenta for CP2 (target)
    COLOR_PHENO = (0.9, 0.9, 0.0, 0.7)  # Yellow for Pheno (target)
    COLOR_ISS = (0.9, 0.2, 0.2, 0.7)    # Red for ISS (target)
    COLOR_UNMATCHED = (0.5, 0.5, 0.5, 0.3)  # Gray for unmatched cells

    def seg_to_colored_mask(seg, color):
        """Convert segmentation to colored RGBA mask."""
        mask = np.zeros((*seg.shape, 4), dtype=np.float32)
        cell_mask = seg > 0
        mask[cell_mask] = color
        return mask

    def overlay_masks(base, overlay):
        """Blend two RGBA masks."""
        result = base.copy()
        overlay_mask = overlay[..., 3] > 0
        result[overlay_mask] = (
            base[overlay_mask] * (1 - overlay[overlay_mask, 3:4]) +
            overlay[overlay_mask] * overlay[overlay_mask, 3:4]
        )
        result[..., 3] = np.maximum(base[..., 3], overlay[..., 3])
        return result

    def generate_distinct_colors(n_colors, seed=42):
        """Generate n distinct colors using HSV color space."""
        np.random.seed(seed)
        colors = []
        for i in range(n_colors):
            # Use golden ratio to spread hues evenly
            hue = (i * 0.618033988749895) % 1.0
            sat = 0.7 + np.random.random() * 0.3  # 0.7-1.0
            val = 0.8 + np.random.random() * 0.2  # 0.8-1.0
            # Convert HSV to RGB
            import colorsys
            r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
            colors.append((r, g, b, 0.8))
        return colors

    def create_match_colored_seg(source_seg, target_seg, crop_df, source_label_col, target_label_col):
        """
        Create two segmentation masks where matched cells share the same color.

        Returns:
            Tuple of (source_colored, target_colored) as RGBA arrays
        """
        # Get matched pairs from crop_df
        matched_df = crop_df[crop_df[target_label_col].notna()].copy()
        n_matches = len(matched_df)

        # Generate distinct colors for each match
        match_colors = generate_distinct_colors(max(n_matches, 1))

        # Initialize output masks
        source_colored = np.zeros((*source_seg.shape, 4), dtype=np.float32)
        target_colored = np.zeros((*target_seg.shape, 4), dtype=np.float32)

        # Color unmatched cells gray
        source_unmatched_mask = source_seg > 0
        target_unmatched_mask = target_seg > 0
        source_colored[source_unmatched_mask] = COLOR_UNMATCHED
        target_colored[target_unmatched_mask] = COLOR_UNMATCHED

        # Color matched cells with same color
        for i, (_, row) in enumerate(matched_df.iterrows()):
            source_label = int(row[source_label_col])
            target_label = int(row[target_label_col])
            color = match_colors[i % len(match_colors)]

            # Color source cell
            source_mask = source_seg == source_label
            source_colored[source_mask] = color

            # Color target cell
            target_mask = target_seg == target_label
            target_colored[target_mask] = color

        return source_colored, target_colored, n_matches

    # Build list of panel pairs (each pair = source/target comparison)
    panel_pairs = []
    if cp2_crop is not None and "cp2_label" in crop_df.columns:
        cp2_matched = crop_df["cp2_label"].notna().sum()
        if cp2_matched > 0:
            panel_pairs.append(("CP1", "CP2", cp1_crop, cp2_crop, COLOR_CP1, COLOR_CP2, "cp1_label", "cp2_label"))
    if pheno_crop is not None and "pheno_label" in crop_df.columns:
        pheno_matched = crop_df["pheno_label"].notna().sum()
        if pheno_matched > 0:
            panel_pairs.append(("CP1", "Pheno", cp1_crop, pheno_crop, COLOR_CP1, COLOR_PHENO, "cp1_label", "pheno_label"))
    if iss_crop is not None and "iss_label" in crop_df.columns:
        iss_matched = crop_df["iss_label"].notna().sum()
        if iss_matched > 0:
            from skimage.transform import resize
            iss_upscaled = resize(iss_crop, (crop_size, crop_size), order=0, preserve_range=True).astype(iss_crop.dtype)
            panel_pairs.append(("CP1", "ISS", cp1_crop, iss_upscaled, COLOR_CP1, COLOR_ISS, "cp1_label", "iss_label"))

    n_pairs = max(1, len(panel_pairs))
    fig, axes = plt.subplots(2, n_pairs, figsize=(5 * n_pairs, 10))

    # Handle single column case
    if n_pairs == 1:
        axes = axes.reshape(2, 1)

    for col_idx, (source_name, target_name, source_seg, target_seg, source_color, target_color, source_col, target_col) in enumerate(panel_pairs):
        # Row 1: Source-colored view (distinct colors per modality)
        ax_top = axes[0, col_idx]
        source_mask = seg_to_colored_mask(source_seg, source_color)
        target_mask = seg_to_colored_mask(target_seg, target_color)
        background = np.ones((*source_seg.shape, 4), dtype=np.float32)
        background[..., 3] = 1.0
        combined = overlay_masks(background, source_mask)
        combined = overlay_masks(combined, target_mask)
        ax_top.imshow(combined)
        n_source = (source_seg > 0).max()  # rough count
        n_target = (target_seg > 0).max()
        ax_top.set_title(f"{source_name} (cyan) + {target_name} (color)\nOverlay view", fontsize=10, fontweight='bold')
        ax_top.axis("off")

        # Row 2: Match-colored view (same color = matched cell pair)
        ax_bot = axes[1, col_idx]
        source_colored, target_colored, n_matches = create_match_colored_seg(
            source_seg, target_seg, crop_df, source_col, target_col
        )
        background = np.ones((*source_seg.shape, 4), dtype=np.float32)
        background[..., 3] = 1.0
        combined_match = overlay_masks(background, source_colored)
        combined_match = overlay_masks(combined_match, target_colored)
        ax_bot.imshow(combined_match)
        ax_bot.set_title(f"Matched cells: same color = linked\n{n_matches} matches (gray = unmatched)", fontsize=10, fontweight='bold')
        ax_bot.axis("off")

    # If no panels to show, display a message
    if len(panel_pairs) == 0:
        for ax in axes.flat:
            ax.text(0.5, 0.5, "No matches to display\n(CP2, Pheno, or ISS data missing)",
                    ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.axis("off")

    # Add overall stats at bottom
    stats_text = f"Crop: {crop_size}x{crop_size} @ pyramid level {PYRAMID_LEVEL} | "
    stats_text += f"CP1 cells in crop: {stats['cp1_in_crop']}"
    if "cp1_cp2_matched" in stats:
        stats_text += f" | CP2 matched: {stats.get('cp1_cp2_matched', 0)}"
    if "cp1_pheno_matched" in stats:
        stats_text += f" | Pheno matched: {stats.get('cp1_pheno_matched', 0)}"
    if "cp1_iss_matched" in stats:
        stats_text += f" | ISS matched: {stats.get('cp1_iss_matched', 0)}"

    fig.text(0.5, 0.01, stats_text, ha='center', fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Title
    fig.suptitle(f"Cell Painting Linking: {experiment} - {well}", fontsize=14, fontweight='bold', y=0.99)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Save
    if output_path is None:
        debug_dir = dataset.results_fast / "debug_output"
        debug_dir.mkdir(parents=True, exist_ok=True)
        output_path = debug_dir / f"linking_preview_{well.replace('/', '_')}.png"
    else:
        output_path = Path(output_path)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\n  Preview saved to: {output_path}")
    return output_path


# =============================================================================
# Per-Well Processing Function
# =============================================================================

def _process_single_well(
    experiment: str,
    well: str,
    verbose: bool = True,
    n_jobs: int = None,
    mode: str = "cp",
) -> dict:
    """
    Process a single well for cell painting linking.

    Uses pyramid level 2 (4x downsampled) for efficient processing.
    Chunk processing is parallelized within each well.

    Args:
        experiment: Experiment name
        well: Well identifier (e.g., "A/1/0")
        verbose: Print progress
        n_jobs: Number of parallel workers for chunk processing

    Returns:
        Dict with well results and metrics, or None if processing failed
    """
    import time
    start_time = time.time()

    dataset = OpsDataset(experiment)

    if mode not in MODE_CONFIG:
        raise ValueError(f"Unknown mode={mode!r}; must be one of {list(MODE_CONFIG)}")
    mcfg = MODE_CONFIG[mode]

    # Always print a start message so user knows processing began
    print(f"\n[{well}] Starting {mode} linking (pyramid level {PYRAMID_LEVEL})...")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing {experiment} - {well}")
        print(f"{'='*60}")

    well_num = well.split("/")[1]  # Extract "1", "2", or "3"
    metrics = {"well": well}

    # -------------------------------------------------------------------------
    # Step 1: Load all segmentations (lazy zarr arrays at pyramid level 2)
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n[1] Loading segmentations (level {PYRAMID_LEVEL}, {DOWNSAMPLE_FACTOR}x downsampled)...")

    cp1_zarr, cp1_max, cp1_shape = load_nuclear_seg_zarr(dataset, well, mcfg["primary_label"])
    # Secondary rounds (CP2 for cp mode; R2..R5 for 4i mode)
    secondary_zarrs = []
    for sec_label in mcfg["secondary_labels"]:
        z, m, sh = load_nuclear_seg_zarr(dataset, well, sec_label)
        if z is not None:
            secondary_zarrs.append((sec_label, z, m, sh))
    # Keep cp2_* for the first secondary so downstream CP1<->CP2 logic works unchanged
    if secondary_zarrs:
        _, cp2_zarr, cp2_max, cp2_shape = secondary_zarrs[0]
    else:
        cp2_zarr, cp2_max, cp2_shape = None, 0, None
    pheno_zarr, pheno_max, pheno_shape = load_pheno_nuclear_seg_zarr(dataset, well)
    iss_zarr, iss_max, iss_shape = load_iss_segmentation_zarr(dataset, well)

    if cp1_zarr is None:
        print(f"[{well}] ERROR: {mcfg['primary_label']} not found, skipping")
        return None

    # Brief status
    seg_status = f"{mcfg['primary_label']}:~{cp1_max} ({cp1_shape[0]}x{cp1_shape[1]})"
    for sec_label, _, m, _ in secondary_zarrs:
        seg_status += f" {sec_label}:~{m}"
    if pheno_zarr is not None:
        seg_status += f" Pheno:~{pheno_max}"
    if iss_zarr is not None:
        seg_status += f" ISS:~{iss_max}"
    print(f"[{well}] Loaded zarr arrays: {seg_status}")

    # -------------------------------------------------------------------------
    # Step 2: Extract centroids (chunk-wise, parallelized, with caching)
    # -------------------------------------------------------------------------
    # Cache directory for centroids
    well_safe = well.replace("/", "_")
    cache_dir = dataset.results_fast / mcfg["cache_subdir"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{well}] Extracting centroids (with caching in cp_links/)...")

    # CP1 centroids
    cp1_cache = cache_dir / f"centroids_cp1_{well_safe}.parquet"
    if cp1_cache.exists():
        cp1_centroids = pd.read_parquet(cp1_cache)
        print(f"[{well}] CP1: {len(cp1_centroids)} cells (cached)")
    else:
        cp1_centroids = get_centroids_from_zarr_chunked(cp1_zarr, desc="  CP1", n_jobs=n_jobs)
        cp1_centroids.to_parquet(cp1_cache, index=False)
        print(f"[{well}] CP1: {len(cp1_centroids)} cells (computed)")
    cp1_centroids = cp1_centroids.rename(columns={
        "label": "cp1_label",
        "y_centroid": "y_cp1",
        "x_centroid": "x_cp1",
        "area": "area_cp1",
    })
    metrics["cp1_cells"] = len(cp1_centroids)

    # CP2 centroids
    if cp2_zarr is not None:
        cp2_cache = cache_dir / f"centroids_cp2_{well_safe}.parquet"
        if cp2_cache.exists():
            cp2_centroids = pd.read_parquet(cp2_cache)
            print(f"[{well}] CP2: {len(cp2_centroids)} cells (cached)")
        else:
            cp2_centroids = get_centroids_from_zarr_chunked(cp2_zarr, desc="  CP2", n_jobs=n_jobs)
            cp2_centroids.to_parquet(cp2_cache, index=False)
            print(f"[{well}] CP2: {len(cp2_centroids)} cells (computed)")
        cp2_centroids = cp2_centroids.rename(columns={
            "label": "cp2_label",
            "y_centroid": "y_cp2",
            "x_centroid": "x_cp2",
        })
        metrics["cp2_cells"] = len(cp2_centroids)
    else:
        cp2_centroids = None
        metrics["cp2_cells"] = 0

    # Pheno centroids
    if pheno_zarr is not None:
        pheno_cache = cache_dir / f"centroids_pheno_{well_safe}.parquet"
        if pheno_cache.exists():
            pheno_centroids = pd.read_parquet(pheno_cache)
            print(f"[{well}] Pheno: {len(pheno_centroids)} cells (cached)")
        else:
            pheno_centroids = get_centroids_from_zarr_chunked(pheno_zarr, desc="  Pheno", n_jobs=n_jobs)
            pheno_centroids.to_parquet(pheno_cache, index=False)
            print(f"[{well}] Pheno: {len(pheno_centroids)} cells (computed)")
        pheno_centroids = pheno_centroids.rename(columns={
            "label": "pheno_label",
            "y_centroid": "y_pheno_centroid",
            "x_centroid": "x_pheno_centroid",
            "area": "area_pheno",
        })
        metrics["pheno_cells"] = len(pheno_centroids)
    else:
        pheno_centroids = None
        metrics["pheno_cells"] = 0

    # ISS centroids
    if iss_zarr is not None:
        iss_cache = cache_dir / f"centroids_iss_{well_safe}.parquet"
        if iss_cache.exists():
            iss_centroids = pd.read_parquet(iss_cache)
            print(f"[{well}] ISS: {len(iss_centroids)} cells (cached)")
        else:
            iss_centroids = get_centroids_from_zarr_chunked(iss_zarr, desc="  ISS", n_jobs=n_jobs)
            iss_centroids.to_parquet(iss_cache, index=False)
            print(f"[{well}] ISS: {len(iss_centroids)} cells (computed)")
        iss_centroids = iss_centroids.rename(columns={
            "label": "iss_label",
            "y_centroid": "y_iss_centroid",
            "x_centroid": "x_iss_centroid",
        })
        metrics["iss_cells"] = len(iss_centroids)
    else:
        iss_centroids = None
        metrics["iss_cells"] = 0

    # -------------------------------------------------------------------------
    # Step 3: Link CP1 <-> CP2 (both at pyramid level 2, same space)
    # -------------------------------------------------------------------------
    print(f"[{well}] Linking CP1 <-> CP2...")

    if cp2_centroids is not None and len(cp2_centroids) > 0:
        # CP1 and CP2 are in the same space at pyramid level 2, direct matching
        cp1_cp2_matches = nearest_neighbor_match(
            cp1_centroids, cp2_centroids,
            "y_cp1", "x_cp1",
            "y_cp2", "x_cp2",
            MAX_MATCH_DISTANCE_CP1_CP2,
        )
        metrics["cp1_cp2_matches"] = len(cp1_cp2_matches)
        print(f"[{well}] CP1<->CP2: {len(cp1_cp2_matches)}/{len(cp1_centroids)} matched")
    else:
        cp1_cp2_matches = pd.DataFrame(columns=["source_idx", "target_idx", "distance"])
        metrics["cp1_cp2_matches"] = 0

    # -------------------------------------------------------------------------
    # Step 4: Link CP1 <-> Pheno (shape-weighted matching)
    # CP1_nuclear_seg and nuclear_seg are BOTH in pheno_assembled_v3 at the same
    # pyramid level, so they're already co-registered.
    # Use shape-weighted matching: closer distance + similar area = better match
    # -------------------------------------------------------------------------
    print(f"[{well}] Linking CP1 <-> Pheno (shape-weighted)...")

    if pheno_centroids is not None and len(pheno_centroids) > 0:
        # CP1 and Pheno are in the same coordinate space (pheno_assembled_v3)
        # Use shape-weighted matching for better accuracy
        cp1_pheno_matches = shape_weighted_match(
            cp1_centroids, pheno_centroids,
            "y_cp1", "x_cp1",
            "y_pheno_centroid", "x_pheno_centroid",
            "area_cp1", "area_pheno",  # Area columns for shape similarity
            MAX_MATCH_DISTANCE_CP1_PHENO,
            k_neighbors=5,  # Consider 5 nearest neighbors
            distance_weight=0.6,  # 60% weight on distance
            shape_weight=0.4,  # 40% weight on shape similarity
        )
        metrics["cp1_pheno_matches"] = len(cp1_pheno_matches)
        print(f"[{well}] CP1<->Pheno: {len(cp1_pheno_matches)}/{len(cp1_centroids)} matched (shape-weighted)")
        if verbose:
            print(f"  Matched {len(cp1_pheno_matches)} / {len(cp1_centroids)} CP1 cells to Pheno")
    else:
        cp1_pheno_matches = pd.DataFrame(columns=["source_idx", "target_idx", "distance", "score"])
        metrics["cp1_pheno_matches"] = 0

    # -------------------------------------------------------------------------
    # Step 5: Link CP1 <-> ISS
    # CP1 is in pheno_assembled_v3 at pyramid level 2 (4x downsampled)
    # ISS segmentation is in ISS space (separate coordinate system)
    #
    # For skip_track=True (ops0094), we apply a single affine transform
    # directly from pheno space (pyramid level 2) to ISS space.
    # -------------------------------------------------------------------------
    print(f"[{well}] Linking CP1 <-> ISS...")

    if iss_centroids is not None and len(iss_centroids) > 0:
        # CP1 coords at pyramid level 2 - pass directly to transform function
        cp1_points = cp1_centroids[["y_cp1", "x_cp1"]].values

        # Transform CP1 pyramid level 2 -> ISS space
        cp1_in_iss = transform_cp1_to_iss(cp1_points, dataset, well)

        cp1_iss_df = cp1_centroids.copy()
        cp1_iss_df["y_cp1_in_iss"] = cp1_in_iss[:, 0]
        cp1_iss_df["x_cp1_in_iss"] = cp1_in_iss[:, 1]

        cp1_iss_matches = nearest_neighbor_match(
            cp1_iss_df, iss_centroids,
            "y_cp1_in_iss", "x_cp1_in_iss",
            "y_iss_centroid", "x_iss_centroid",
            MAX_MATCH_DISTANCE_CP1_ISS,
        )
        metrics["cp1_iss_matches"] = len(cp1_iss_matches)
        print(f"[{well}] CP1<->ISS: {len(cp1_iss_matches)}/{len(cp1_centroids)} matched")
    else:
        cp1_iss_matches = pd.DataFrame(columns=["source_idx", "target_idx", "distance"])
        cp1_iss_df = cp1_centroids.copy()
        metrics["cp1_iss_matches"] = 0

    # -------------------------------------------------------------------------
    # Step 6: Build unified cell table
    # -------------------------------------------------------------------------
    print(f"[{well}] Building unified cell table...")

    unified = cp1_centroids.copy()
    unified["cp1_idx"] = unified.index

    if len(cp1_cp2_matches) > 0:
        cp2_matched = cp2_centroids.iloc[cp1_cp2_matches["target_idx"].values].reset_index(drop=True)
        cp2_matched["cp1_idx"] = cp1_cp2_matches["source_idx"].values
        cp2_matched["cp1_cp2_distance"] = cp1_cp2_matches["distance"].values
        unified = unified.merge(cp2_matched, on="cp1_idx", how="left")

    if len(cp1_pheno_matches) > 0:
        pheno_matched = pheno_centroids.iloc[cp1_pheno_matches["target_idx"].values].reset_index(drop=True)
        pheno_matched["cp1_idx"] = cp1_pheno_matches["source_idx"].values
        pheno_matched["cp1_pheno_distance"] = cp1_pheno_matches["distance"].values
        unified = unified.merge(pheno_matched, on="cp1_idx", how="left")

    if len(cp1_iss_matches) > 0:
        iss_matched = iss_centroids.iloc[cp1_iss_matches["target_idx"].values].reset_index(drop=True)
        iss_matched["cp1_idx"] = cp1_iss_matches["source_idx"].values
        iss_matched["cp1_iss_distance"] = cp1_iss_matches["distance"].values
        unified = unified.merge(iss_matched, on="cp1_idx", how="left")

    # -------------------------------------------------------------------------
    # Step 7: Link to existing phenotyping results (for barcodes/genes)
    # Match using coordinates since pheno_label (pyramid level 2) != segmentation_id (full res)
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[7] Linking to phenotyping results...")

    linked_results_path = dataset.append_well("linked_results", well)
    if Path(linked_results_path).exists():
        pheno_results = pd.read_csv(linked_results_path)
        if verbose:
            print(f"  Loaded {len(pheno_results)} rows from linked_results ({len(pheno_results.columns)} columns)")

        # Match using coordinates: linked_results has y_pheno/x_pheno (full res)
        # Our unified table has y_pheno_centroid/x_pheno_centroid (pyramid level 2)
        if "y_pheno_centroid" in unified.columns and "y_pheno" in pheno_results.columns:
            # Scale our pyramid level 2 coords to full res for matching
            unified_with_pheno = unified[unified["y_pheno_centroid"].notna()].copy()

            if len(unified_with_pheno) > 0:
                # Convert pyramid level 2 coords to full res
                unified_pheno_fullres_y = unified_with_pheno["y_pheno_centroid"].values * DOWNSAMPLE_FACTOR
                unified_pheno_fullres_x = unified_with_pheno["x_pheno_centroid"].values * DOWNSAMPLE_FACTOR

                # linked_results coords are already full res
                pheno_results_valid = pheno_results[pheno_results["y_pheno"].notna()].copy()

                if len(pheno_results_valid) > 0:
                    from scipy.spatial import cKDTree

                    # Build KDTree from linked_results coordinates
                    linked_coords = pheno_results_valid[["y_pheno", "x_pheno"]].values
                    tree = cKDTree(linked_coords)

                    # Query for each unified cell with pheno match
                    unified_coords = np.column_stack([unified_pheno_fullres_y, unified_pheno_fullres_x])
                    distances, indices = tree.query(unified_coords, k=1)

                    # Max distance threshold (in full-res pixels) - should be very small since these are same cells
                    max_coord_distance = 50  # ~10 pixels at pyramid level 2

                    # Add linked_results columns to unified
                    cols_to_add = [c for c in pheno_results_valid.columns
                                   if c not in unified.columns and c not in ["y_pheno", "x_pheno"]]

                    # Vectorized matching - much faster than row-by-row
                    valid_mask = distances < max_coord_distance
                    matched_count = valid_mask.sum()

                    # Get the unified indices and corresponding pheno_results indices for valid matches
                    unified_indices = unified_with_pheno.index[valid_mask].values
                    pheno_indices = indices[valid_mask]

                    # Extract matched data from pheno_results in one go
                    matched_data = pheno_results_valid.iloc[pheno_indices][cols_to_add].reset_index(drop=True)
                    matched_data.index = unified_indices

                    # Join the matched columns to unified
                    for col in cols_to_add:
                        unified[col] = np.nan
                        unified.loc[unified_indices, col] = matched_data[col].values

                    if verbose:
                        print(f"  Matched {matched_count}/{len(unified_with_pheno)} pheno cells to linked_results by coordinates")

                    n_with_barcode = unified["barcode"].notna().sum() if "barcode" in unified.columns else 0
                    n_with_gene = unified["gene_name"].notna().sum() if "gene_name" in unified.columns else 0
                    if verbose:
                        print(f"  Linked {n_with_barcode} cells to barcodes, {n_with_gene} to genes")
                        print(f"  Added {len(cols_to_add)} columns from linked_results")
                    metrics["cells_with_barcode"] = int(n_with_barcode)
                    metrics["cells_with_gene"] = int(n_with_gene)
                else:
                    if verbose:
                        print("  No valid coordinates in linked_results for matching")
                    metrics["cells_with_barcode"] = 0
                    metrics["cells_with_gene"] = 0
            else:
                if verbose:
                    print("  No cells with pheno matches to link")
                metrics["cells_with_barcode"] = 0
                metrics["cells_with_gene"] = 0
        else:
            if verbose:
                print("  Missing coordinate columns for linking")
            metrics["cells_with_barcode"] = 0
            metrics["cells_with_gene"] = 0
    else:
        if verbose:
            print(f"  No linked_results found, skipping barcode/gene linking")
        metrics["cells_with_barcode"] = 0
        metrics["cells_with_gene"] = 0

    # -------------------------------------------------------------------------
    # Step 8: Add ISS barcodes AND gene metadata for cells not matched via phenotyping
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[8] Adding ISS barcodes for direct CP1-ISS matches...")

    if "iss_label" in unified.columns:
        iss_reads_path = dataset.append_well("reads", well)
        if Path(iss_reads_path).exists():
            # Mirrors datasets.link_tracking_iss with method="mine": take all reads
            # as-is, no quality threshold (matches the canonical link_calls_tracks).
            iss_reads = pd.read_csv(iss_reads_path)

            cell_barcode_lookup = iss_reads.set_index("cell")["barcode"].to_dict()

            mask_no_barcode = unified["barcode"].isna() & unified["iss_label"].notna()
            unified.loc[mask_no_barcode, "barcode_from_iss"] = unified.loc[mask_no_barcode, "iss_label"].apply(
                lambda x: cell_barcode_lookup.get(int(x), None) if pd.notna(x) else None
            )

            # Get effective ISS rounds for this well (accounting for failed rounds)
            # Need this BEFORE filling barcode column so we can truncate ISS barcodes
            from cyclops_process.metrics.plate_stats.match_reads import _get_effective_iss_rounds
            import yaml

            # Load failed rounds config
            failed_rounds_by_well = {}
            if dataset.failed_rounds.exists():
                with open(dataset.failed_rounds, "r") as f:
                    failed_config = yaml.safe_load(f) or {}
                    failed_rounds_by_well = failed_config.get(experiment, {})

            # The assay's actual round count comes from the per-experiment
            # iss_rounds_manifest.yaml — written by iss_metrics.py to
            # ``dataset.results_iss / "iss_rounds_manifest.yaml"`` and also
            # consumed by iss_read_freq_analysis.py the same way.
            # Hardcoding ``list(range(10))`` here was the bug that left
            # direct CP1-ISS matches stamped with 10-mer barcodes when the
            # library is 9-mer — confirmed 2026-05-17 by inspecting
            # A1_linked_pheno_iss_cp.csv (155k rows had 10-mer barcodes
            # that should have been truncated).
            iss_manifest_path = dataset.results_iss / "iss_rounds_manifest.yaml"
            default_iss_rounds = list(range(10))
            if iss_manifest_path.exists():
                with open(iss_manifest_path, "r") as f:
                    iss_manifest = yaml.safe_load(f) or {}
                manifest_base = iss_manifest.get("base_iss_rounds")
                if isinstance(manifest_base, list) and manifest_base:
                    default_iss_rounds = list(manifest_base)
            well_iss_rounds = _get_effective_iss_rounds(default_iss_rounds, well, failed_rounds_by_well)

            # Truncate ALL direct ISS barcodes to match effective rounds BEFORE adding them
            # This ensures consistent barcode lengths with pheno-matched cells (which already
            # have truncated barcodes from linked_results)
            def truncate_barcode(x):
                if pd.isna(x):
                    return x
                x_str = str(x)
                if len(x_str) >= max(well_iss_rounds) + 1:
                    return "".join([x_str[i] for i in well_iss_rounds])
                return x_str  # Already truncated or too short

            unified.loc[mask_no_barcode, "barcode_from_iss"] = unified.loc[mask_no_barcode, "barcode_from_iss"].apply(truncate_barcode)

            unified["barcode"] = unified["barcode"].fillna(unified.get("barcode_from_iss"))

            n_new_barcodes = (mask_no_barcode & unified["barcode"].notna()).sum()
            if verbose:
                print(f"  Added {n_new_barcodes} barcodes from direct CP1-ISS matches (truncated to {len(well_iss_rounds)} rounds)")
            metrics["barcodes_from_direct_iss"] = int((unified["barcode_from_iss"].notna()).sum())

            # Step 8b: Add gene metadata for direct ISS matches
            # These cells have barcodes from direct ISS matching but no gene_name yet
            if verbose:
                print("  Adding gene metadata for direct ISS matches...")

            # Load gene index
            gene_df = dataset.load_gene_index()

            # Truncate gene_df barcodes to match this well's effective rounds
            gene_df_well = gene_df.copy()
            gene_df_well["barcode_truncated"] = gene_df_well["barcode"].apply(
                lambda x: "".join([x[i] for i in well_iss_rounds]) if pd.notna(x) and len(x) >= max(well_iss_rounds)+1 else None
            )

            # Find cells that have barcode but no gene_name (from direct ISS matches)
            mask_barcode_no_gene = unified["barcode"].notna() & unified["gene_name"].isna()
            n_missing_gene = mask_barcode_no_gene.sum()

            if n_missing_gene > 0:
                # Create lookup from truncated barcode to gene metadata
                # Use all columns from gene_df except barcode columns (matching datasets.py merge behavior)
                gene_cols = [c for c in gene_df_well.columns if c not in ["barcode", "barcode_truncated"]]

                # Verify expected columns are present (sanity check)
                expected_cols = ["sgRNA", "subpool", "NCBI_ID", "dep_map_gene_name", "gene_effect", "Gene name"]
                missing_expected = [c for c in expected_cols if c not in gene_cols]
                if missing_expected:
                    print(f"  WARNING: gene_index missing expected columns: {missing_expected}")

                gene_lookup = gene_df_well.set_index("barcode_truncated")[gene_cols].to_dict("index")

                # Apply gene metadata to cells missing it
                # Cell barcodes may be raw (10-char) or truncated (9-char) - truncate for lookup
                for idx in unified[mask_barcode_no_gene].index:
                    bc = unified.loc[idx, "barcode"]
                    # Truncate barcode to effective rounds for lookup (handles length mismatch)
                    if pd.notna(bc) and len(str(bc)) >= max(well_iss_rounds) + 1:
                        bc_truncated = "".join([str(bc)[i] for i in well_iss_rounds])
                    else:
                        bc_truncated = bc
                    if bc_truncated in gene_lookup:
                        gene_info = gene_lookup[bc_truncated]
                        for col, val in gene_info.items():
                            target_col = "gene_name" if col == "Gene name" else col
                            if target_col not in unified.columns:
                                unified[target_col] = np.nan
                            unified.loc[idx, target_col] = val

                n_with_gene_now = unified.loc[mask_barcode_no_gene, "gene_name"].notna().sum()
                if verbose:
                    print(f"  Added gene metadata for {n_with_gene_now}/{n_missing_gene} cells with barcodes")
                    if n_missing_gene - n_with_gene_now > 0:
                        print(f"  ({n_missing_gene - n_with_gene_now} barcodes not found in gene index - may be invalid/low-quality reads)")

    # -------------------------------------------------------------------------
    # Step 9: Compute bounding boxes from cp_cell_seg
    # -------------------------------------------------------------------------
    print(f"[{well}] Computing bounding boxes from {mcfg['cell_seg_label']}...")

    unified = compute_cp_cell_bboxes(
        unified, dataset, well,
        cell_seg_label=mcfg["cell_seg_label"],
        bbox_col=mcfg["bbox_col"],
        seg_id_col=mcfg["seg_id_col"],
    )
    n_valid_bbox = unified[mcfg["seg_id_col"]].notna().sum()
    metrics["cells_with_valid_bbox"] = int(n_valid_bbox)
    metrics["cells_with_fallback_bbox"] = int(len(unified) - n_valid_bbox)

    # -------------------------------------------------------------------------
    # Step 10: Scale coordinates to full resolution for output
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[10] Scaling coordinates to full resolution...")

    # Scale all pyramid level 2 coordinates to full resolution (4x)
    # CP1, CP2, and Pheno centroids are all at pyramid level 2
    coord_cols_to_scale = [
        "y_cp1", "x_cp1",
        "y_cp2", "x_cp2",
        "y_pheno_centroid", "x_pheno_centroid",
    ]
    for col in coord_cols_to_scale:
        if col in unified.columns:
            unified[col] = unified[col] * DOWNSAMPLE_FACTOR

    # Also scale match distances (they were computed at pyramid level 2)
    distance_cols_to_scale = ["cp1_cp2_distance", "cp1_pheno_distance"]
    for col in distance_cols_to_scale:
        if col in unified.columns:
            unified[col] = unified[col] * DOWNSAMPLE_FACTOR

    # Note: ISS coordinates (y_iss_centroid, x_iss_centroid) are already in ISS space
    # and cp1_iss_distance is in ISS space, so we don't scale those

    # -------------------------------------------------------------------------
    # Step 11: Deduplicate by segmentation_id (only for cells WITH segmentation_id)
    # -------------------------------------------------------------------------
    # WHY DUPLICATES OCCUR:
    # CP1 centroids come from cp_nuclear_seg (nuclear segmentation) at pyramid level 2.
    # segmentation_id comes from pheno linked_results (cell segmentation at full res).
    # Multiple nuclear centroids can map to the same cell segmentation_id because:
    #   1. Multiple CP1 nuclei fall within the same pheno cell boundary
    #   2. Coordinate-based matching to linked_results maps multiple CP1 cells to same row
    #
    # DESIGN CHOICE - CELL OVER NUCLEUS:
    # For feature extraction, we prioritize ONE row per CELL (not per nucleus).
    # We deduplicate by segmentation_id, keeping the best match (has barcode, closest).
    # This differs from datasets.py which allows multiple nuclei per segmentation_id.
    #
    # IMPORTANT: Only deduplicate rows that HAVE a segmentation_id. CP cells without
    # a pheno match (segmentation_id is NaN) should be kept - they're valid CP cells
    # that just didn't match to pheno/ISS.
    if "segmentation_id" in unified.columns:
        # Split into cells with and without segmentation_id
        has_seg_id = unified["segmentation_id"].notna()
        unified_with_seg = unified[has_seg_id].copy()
        unified_without_seg = unified[~has_seg_id].copy()

        n_with_seg = len(unified_with_seg)
        n_without_seg = len(unified_without_seg)
        n_unique_seg = unified_with_seg["segmentation_id"].nunique()

        if n_with_seg > n_unique_seg:
            # Sort by match quality: prefer cells with ISS match, then by distance
            sort_cols = []
            sort_ascending = []

            # Prefer cells with ISS barcode
            if "barcode" in unified_with_seg.columns:
                unified_with_seg["_has_barcode"] = unified_with_seg["barcode"].notna().astype(int)
                sort_cols.append("_has_barcode")
                sort_ascending.append(False)  # True (1) first

            # Then by CP1-pheno distance (smaller = better)
            if "cp1_pheno_distance" in unified_with_seg.columns:
                sort_cols.append("cp1_pheno_distance")
                sort_ascending.append(True)  # Smaller first

            if sort_cols:
                unified_with_seg = unified_with_seg.sort_values(sort_cols, ascending=sort_ascending)

            # Keep first (best) match per segmentation_id
            unified_with_seg = unified_with_seg.drop_duplicates(subset=["segmentation_id"], keep="first")

            # Clean up temp column
            if "_has_barcode" in unified_with_seg.columns:
                unified_with_seg = unified_with_seg.drop(columns=["_has_barcode"])

            n_deduped = len(unified_with_seg)
            n_removed = n_with_seg - n_deduped
            print(f"[{well}] Deduplicated by segmentation_id: {n_with_seg} -> {n_deduped} ({n_removed} duplicates removed)")
            metrics["duplicates_removed"] = int(n_removed)

        # -------------------------------------------------------------------------
        # Step 11b: Deduplicate CP-only cells by cp_cell_seg_id
        # -------------------------------------------------------------------------
        # WHY DUPLICATES OCCUR:
        # CP1 centroids come from cp_nuclear_seg (nuclear segmentation) at pyramid level 2.
        # cp_cell_seg_id comes from cp_cell_seg (cell segmentation) at full resolution.
        # Multiple nuclear centroids can fall within the same cell boundary because:
        #   1. Multinucleated cells (one cell body contains multiple nuclei)
        #   2. Segmentation differences between nuclear and cell boundaries
        #   3. One CP cell matches multiple direct ISS barcodes
        #
        # DESIGN CHOICE - CELL OVER NUCLEUS:
        # Same as Step 11 - we prioritize ONE row per CELL for feature extraction.
        # We deduplicate by cp_cell_seg_id, keeping the best match (has barcode, closest).
        # Also drop orphan cells that have no valid identifier.
        if len(unified_without_seg) > 0:
            cell_seg_id_col = mcfg["seg_id_col"]
            if cell_seg_id_col in unified_without_seg.columns:
                has_cp_id = unified_without_seg[cell_seg_id_col].notna()
            else:
                has_cp_id = pd.Series(False, index=unified_without_seg.index)

            cp_only_with_id = unified_without_seg[has_cp_id].copy()
            cp_only_no_id = unified_without_seg[~has_cp_id]

            # Drop orphans (cells with no valid identifier)
            if len(cp_only_no_id) > 0:
                print(f"[{well}] Dropped {len(cp_only_no_id)} orphan cells (no segmentation_id, no {cell_seg_id_col})")
                metrics["orphans_dropped"] = int(len(cp_only_no_id))

            # Deduplicate remaining CP-only cells by cp_cell_seg_id
            n_cp_before = len(cp_only_with_id)
            if n_cp_before > 0:
                sort_cols = []
                sort_ascending = []

                if "barcode" in cp_only_with_id.columns:
                    cp_only_with_id["_has_barcode"] = cp_only_with_id["barcode"].notna().astype(int)
                    sort_cols.append("_has_barcode")
                    sort_ascending.append(False)  # Prefer cells with barcode

                if "cp1_iss_distance" in cp_only_with_id.columns:
                    sort_cols.append("cp1_iss_distance")
                    sort_ascending.append(True)  # Prefer closer matches

                if sort_cols:
                    cp_only_with_id = cp_only_with_id.sort_values(sort_cols, ascending=sort_ascending)

                # Keep first (best) match per cell_seg_id
                cp_only_with_id = cp_only_with_id.drop_duplicates(subset=[cell_seg_id_col], keep="first")

                if "_has_barcode" in cp_only_with_id.columns:
                    cp_only_with_id = cp_only_with_id.drop(columns=["_has_barcode"])

                n_cp_after = len(cp_only_with_id)
                n_cp_removed = n_cp_before - n_cp_after
                if n_cp_removed > 0:
                    print(f"[{well}] Deduplicated by {cell_seg_id_col}: {n_cp_before} -> {n_cp_after} ({n_cp_removed} duplicates removed)")
                    metrics["cp_only_duplicates_removed"] = int(n_cp_removed)

            # Update unified_without_seg to only include valid CP-only cells (orphans are dropped)
            unified_without_seg = cp_only_with_id
            n_without_seg = len(unified_without_seg)

        # Recombine: deduped cells with seg_id + deduped CP-only cells
        unified = pd.concat([unified_with_seg, unified_without_seg], ignore_index=True)
        print(f"[{well}] Final: {len(unified)} cells ({len(unified_with_seg)} with pheno match, {n_without_seg} CP-only)")

    # -------------------------------------------------------------------------
    # Step 12: Fill NTC gene_name for cells with NCBI_ID = -1
    # -------------------------------------------------------------------------
    # Non-targeting controls (NTC) have NCBI_ID = -1 in the gene index
    # Fill their gene_name as "NTC" for clarity
    if "NCBI_ID" in unified.columns and "gene_name" in unified.columns:
        ntc_mask = unified["NCBI_ID"] == -1
        n_ntc = ntc_mask.sum()
        if n_ntc > 0:
            unified.loc[ntc_mask, "gene_name"] = "NTC"
            if verbose:
                print(f"\n[12a] Labeled {n_ntc} cells with NCBI_ID=-1 as NTC")

    # -------------------------------------------------------------------------
    # Step 12b: Filter to cells with valid codebook matches only
    # -------------------------------------------------------------------------
    # Following the pattern from datasets.py link_calls_tracks() which uses an inner
    # join with gene_df to filter to valid codebook matches. Cells without valid sgRNA
    # have ISS reads that didn't match the codebook - they shouldn't be in the output
    # as they can't be assigned to a guide/gene.
    if "sgRNA" in unified.columns:
        n_before = len(unified)
        has_valid_sgrna = unified["sgRNA"].notna() & (unified["sgRNA"] != "") & (unified["sgRNA"].astype(str) != "None")
        n_valid = has_valid_sgrna.sum()
        n_invalid = n_before - n_valid

        if n_invalid > 0:
            unified = unified[has_valid_sgrna].reset_index(drop=True)
            print(f"[{well}] Filtered to cells with valid codebook matches: {n_before:,} -> {n_valid:,} ({n_invalid:,} unmatched ISS reads dropped)")
            metrics["cells_no_codebook_match"] = int(n_invalid)
        else:
            if verbose:
                print(f"\n[12b] All {n_valid:,} cells have valid codebook matches")

    # -------------------------------------------------------------------------
    # Step 13: Save results
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[13] Saving results...")

    # In 4i mode, rename internal cp1/cp2 column prefixes to 4i / 4i_r2 so the
    # output columns match the mode (R1 is the "primary" 4i round; R2 is the
    # cross-round secondary that was merged in).
    if mode == "4i":
        col_rename = {
            "cp1_idx": "4i_idx",
            "cp1_label": "4i_label",
            "y_cp1": "y_4i",
            "x_cp1": "x_4i",
            "area_cp1": "area_4i",
            "cp2_label": "4i_r2_label",
            "y_cp2": "y_4i_r2",
            "x_cp2": "x_4i_r2",
            "cp1_cp2_distance": "4i_r1_r2_distance",
            "cp1_pheno_distance": "4i_pheno_distance",
            "cp1_iss_distance": "4i_iss_distance",
        }
        unified = unified.rename(columns={k: v for k, v in col_rename.items() if k in unified.columns})
        primary_priority = [
            "4i_idx", "4i_label", "y_4i", "x_4i",
            "4i_r2_label", "y_4i_r2", "x_4i_r2", "4i_r1_r2_distance",
            "pheno_label", "y_pheno_centroid", "x_pheno_centroid", "4i_pheno_distance",
            "iss_label", "y_iss_centroid", "x_iss_centroid", "4i_iss_distance",
        ]
    else:
        primary_priority = [
            "cp1_idx", "cp1_label", "y_cp1", "x_cp1",
            "cp2_label", "y_cp2", "x_cp2", "cp1_cp2_distance",
            "pheno_label", "y_pheno_centroid", "x_pheno_centroid", "cp1_pheno_distance",
            "iss_label", "y_iss_centroid", "x_iss_centroid", "cp1_iss_distance",
        ]

    # Put linking columns first, then everything else from linked_results.
    # og_index is the primary unique identifier (from tracking graph) - put it first
    priority_cols = [
        "og_index",
        *primary_priority,
        mcfg["seg_id_col"], mcfg["bbox_col"],
        # From linked_results (original pipeline) - keep these names as-is
        "barcode", "gene_name", "bbox", "segmentation_id",
    ]
    existing_priority = [c for c in priority_cols if c in unified.columns]
    other_cols = [c for c in unified.columns if c not in priority_cols]
    unified = unified[existing_priority + other_cols]

    _parts = well.split("/")
    output_path = dataset.results_fast / mcfg["output_template"].format(
        well_safe=well.replace("/", "_"),
        well_short=f"{_parts[0]}{_parts[1]}",
    )
    unified.to_csv(output_path, index=False)
    print(f"[{well}] Saved {len(unified)} cells to CSV")
    if verbose:
        print(f"  Path: {output_path}")

    metrics["total_cells"] = len(unified)

    if verbose:
        print(f"\n  Summary:")
        print(f"    Total CP1 cells: {metrics['cp1_cells']}")
        print(f"    Matched to CP2: {metrics['cp1_cp2_matches']}")
        print(f"    Matched to Pheno: {metrics['cp1_pheno_matches']}")
        print(f"    Matched to ISS: {metrics['cp1_iss_matches']}")
        print(f"    With barcode: {metrics.get('cells_with_barcode', 0)}")
        print(f"    With gene: {metrics.get('cells_with_gene', 0)}")

    # -------------------------------------------------------------------------
    # Step 14: Generate linking preview
    # -------------------------------------------------------------------------
    print(f"[{well}] Generating linking preview...")

    try:
        preview_path = generate_linking_preview(
            experiment=experiment,
            well=well,
            crop_size=1024,
            mode=mode,
        )
        if preview_path:
            metrics["preview_path"] = str(preview_path)
            print(f"[{well}] Preview saved: {preview_path.name}")
    except Exception as e:
        print(f"[{well}] Warning: Could not generate preview: {e}")

    # Final completion message with timing
    elapsed = time.time() - start_time
    print(f"[{well}] DONE in {elapsed:.1f}s - {metrics['total_cells']} cells, "
          f"{metrics.get('cells_with_barcode', 0)} with barcode")

    return {
        "well": well,
        "metrics": metrics,
        "output_path": str(output_path),
    }


# =============================================================================
# Linking Report Generation
# =============================================================================

def generate_linking_report(
    experiment: str,
    wells: list[str],
    results: dict,
) -> Optional[Path]:
    """
    Generate a summary report of cell painting linking statistics.

    Creates a CSV with per-well and total statistics showing:
    - Total CP1 cells
    - Cells matched to CP2
    - Cells matched to Pheno
    - Cells matched to ISS
    - Cells linked to barcodes (from original linked_results)
    - Cells linked to genes

    Args:
        experiment: Experiment name
        wells: List of wells processed
        results: Results dict from link_cell_painting()

    Returns:
        Path to saved report CSV, or None if failed
    """
    dataset = OpsDataset(experiment)

    # Build report rows
    report_rows = []
    totals = {
        "cp1_cells": 0,
        "cp2_cells": 0,
        "pheno_cells": 0,
        "iss_cells": 0,
        "cp1_cp2_matches": 0,
        "cp1_pheno_matches": 0,
        "cp1_iss_matches": 0,
        "cells_with_barcode": 0,
        "cells_with_gene": 0,
        "barcodes_from_direct_iss": 0,
    }

    for well in wells:
        if well not in results:
            continue

        m = results[well]["metrics"]

        row = {
            "well": well,
            "cp1_cells": m.get("cp1_cells", 0),
            "cp2_cells": m.get("cp2_cells", 0),
            "pheno_cells": m.get("pheno_cells", 0),
            "iss_cells": m.get("iss_cells", 0),
            "cp1_to_cp2_matched": m.get("cp1_cp2_matches", 0),
            "cp1_to_pheno_matched": m.get("cp1_pheno_matches", 0),
            "cp1_to_iss_matched": m.get("cp1_iss_matches", 0),
            "cells_with_barcode": m.get("cells_with_barcode", 0),
            "cells_with_gene": m.get("cells_with_gene", 0),
            "barcodes_from_direct_iss": m.get("barcodes_from_direct_iss", 0),
        }

        # Calculate percentages
        if row["cp1_cells"] > 0:
            row["pct_cp2_matched"] = round(100 * row["cp1_to_cp2_matched"] / row["cp1_cells"], 1)
            row["pct_pheno_matched"] = round(100 * row["cp1_to_pheno_matched"] / row["cp1_cells"], 1)
            row["pct_iss_matched"] = round(100 * row["cp1_to_iss_matched"] / row["cp1_cells"], 1)
            row["pct_with_barcode"] = round(100 * row["cells_with_barcode"] / row["cp1_cells"], 1)
            row["pct_with_gene"] = round(100 * row["cells_with_gene"] / row["cp1_cells"], 1)
        else:
            row["pct_cp2_matched"] = 0.0
            row["pct_pheno_matched"] = 0.0
            row["pct_iss_matched"] = 0.0
            row["pct_with_barcode"] = 0.0
            row["pct_with_gene"] = 0.0

        report_rows.append(row)

        # Accumulate totals
        for key in totals:
            totals[key] += m.get(key, 0)

    # Add totals row
    total_row = {
        "well": "TOTAL",
        "cp1_cells": totals["cp1_cells"],
        "cp2_cells": totals["cp2_cells"],
        "pheno_cells": totals["pheno_cells"],
        "iss_cells": totals["iss_cells"],
        "cp1_to_cp2_matched": totals["cp1_cp2_matches"],
        "cp1_to_pheno_matched": totals["cp1_pheno_matches"],
        "cp1_to_iss_matched": totals["cp1_iss_matches"],
        "cells_with_barcode": totals["cells_with_barcode"],
        "cells_with_gene": totals["cells_with_gene"],
        "barcodes_from_direct_iss": totals["barcodes_from_direct_iss"],
    }

    if totals["cp1_cells"] > 0:
        total_row["pct_cp2_matched"] = round(100 * totals["cp1_cp2_matches"] / totals["cp1_cells"], 1)
        total_row["pct_pheno_matched"] = round(100 * totals["cp1_pheno_matches"] / totals["cp1_cells"], 1)
        total_row["pct_iss_matched"] = round(100 * totals["cp1_iss_matches"] / totals["cp1_cells"], 1)
        total_row["pct_with_barcode"] = round(100 * totals["cells_with_barcode"] / totals["cp1_cells"], 1)
        total_row["pct_with_gene"] = round(100 * totals["cells_with_gene"] / totals["cp1_cells"], 1)
    else:
        total_row["pct_cp2_matched"] = 0.0
        total_row["pct_pheno_matched"] = 0.0
        total_row["pct_iss_matched"] = 0.0
        total_row["pct_with_barcode"] = 0.0
        total_row["pct_with_gene"] = 0.0

    report_rows.append(total_row)

    # Create DataFrame
    report_df = pd.DataFrame(report_rows)

    # Reorder columns for readability
    col_order = [
        "well",
        "cp1_cells", "cp2_cells", "pheno_cells", "iss_cells",
        "cp1_to_cp2_matched", "pct_cp2_matched",
        "cp1_to_pheno_matched", "pct_pheno_matched",
        "cp1_to_iss_matched", "pct_iss_matched",
        "cells_with_barcode", "pct_with_barcode",
        "cells_with_gene", "pct_with_gene",
        "barcodes_from_direct_iss",
    ]
    report_df = report_df[[c for c in col_order if c in report_df.columns]]

    # Save report
    report_path = dataset.results_fast / "cell_painting_linking_report.csv"
    report_df.to_csv(report_path, index=False)

    # Print report
    print(f"\n{'='*80}")
    print("CELL PAINTING LINKING REPORT")
    print(f"{'='*80}")
    print(f"Experiment: {experiment}")
    print(f"Wells processed: {len(wells)}")
    print()

    # Print table header
    print(f"{'Well':<10} {'CP1':<8} {'->CP2':<8} {'->Pheno':<10} {'->ISS':<8} {'Barcode':<10} {'Gene':<10}")
    print("-" * 80)

    for row in report_rows:
        well_str = row["well"]
        cp1 = row["cp1_cells"]
        cp2_m = f"{row['cp1_to_cp2_matched']} ({row['pct_cp2_matched']}%)"
        pheno_m = f"{row['cp1_to_pheno_matched']} ({row['pct_pheno_matched']}%)"
        iss_m = f"{row['cp1_to_iss_matched']} ({row['pct_iss_matched']}%)"
        bc = f"{row['cells_with_barcode']} ({row['pct_with_barcode']}%)"
        gene = f"{row['cells_with_gene']} ({row['pct_with_gene']}%)"

        if well_str == "TOTAL":
            print("-" * 80)
        print(f"{well_str:<10} {cp1:<8} {cp2_m:<8} {pheno_m:<10} {iss_m:<8} {bc:<10} {gene:<10}")

    print(f"\nReport saved to: {report_path}")

    return report_path


# =============================================================================
# Main Linking Function
# =============================================================================

def link_cell_painting(
    experiment: str,
    wells: list[str] = None,
    verbose: bool = True,
    n_jobs: int = None,
    mode: str = "cp",
) -> dict:
    """
    Link cells across cell painting modalities.

    Processes wells sequentially, but chunk processing within each well is parallelized.
    This approach is more memory-efficient than parallel well processing.

    Args:
        experiment: Experiment name
        wells: List of wells to process (default: ["A/1/0", "A/2/0", "A/3/0"])
        verbose: Print progress
        n_jobs: Number of parallel workers for chunk processing within each well.
                If None, uses resource_manager to determine optimal count.

    Returns:
        Dict with per-well results and metrics
    """
    from ops_utils.hpc.resource_manager import get_optimal_workers

    if wells is None:
        wells = ["A/1/0", "A/2/0", "A/3/0"]

    # Determine number of workers for chunk processing
    if n_jobs is None:
        n_jobs = get_optimal_workers(use_gpu=False, verbose=False)

    print(f"\n{'='*60}")
    print(f"Cell Painting Linking: {experiment}")
    print(f"Wells: {wells}")
    print(f"Pyramid level: {PYRAMID_LEVEL} ({DOWNSAMPLE_FACTOR}x downsampled)")
    print(f"Chunk workers: {n_jobs}")
    print(f"{'='*60}")

    # Process wells sequentially (chunk processing is parallelized within each well)
    well_results = []
    for i, well in enumerate(wells):
        print(f"\n[{i+1}/{len(wells)}] Processing well {well}...")
        result = _process_single_well(
            experiment=experiment,
            well=well,
            verbose=verbose,
            n_jobs=n_jobs,
            mode=mode,
        )
        well_results.append(result)

    # Collect results
    results = {}
    for result in well_results:
        if result is not None:
            well = result["well"]
            results[well] = {
                "metrics": result["metrics"],
                "output_path": result["output_path"],
            }

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for well, data in results.items():
        m = data["metrics"]
        print(f"\n{well}:")
        print(f"  CP1 cells: {m['cp1_cells']}, CP2 matches: {m['cp1_cp2_matches']}")
        print(f"  Pheno matches: {m['cp1_pheno_matches']}, ISS matches: {m['cp1_iss_matches']}")
        print(f"  With barcode: {m.get('cells_with_barcode', 0)}, With gene: {m.get('cells_with_gene', 0)}")

    # Generate and save linking report
    report_path = generate_linking_report(experiment, wells, results)
    if report_path:
        print(f"\nLinking report saved to: {report_path}")

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Link cells across cell painting modalities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m cyclops_process.data.link_cell_painting -e ops0094
  python -m cyclops_process.data.link_cell_painting -e 94 --wells A/1/0
        """,
    )

    parser.add_argument(
        "--experiment", "-e",
        type=str,
        required=True,
        help="Experiment name (supports shorthand like '94' or 'ops94')",
    )

    parser.add_argument(
        "--wells", "-w",
        type=str,
        nargs="+",
        default=["A/1/0", "A/2/0", "A/3/0"],
        help="Wells to process (default: A/1/0 A/2/0 A/3/0)",
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce verbosity",
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel workers for chunk processing (default: auto-detect)",
    )

    parser.add_argument(
        "--4i", dest="four_i", action="store_true",
        help="4i mode (R1_nuclear_seg primary, 4i_cell_seg for bboxes, outputs four_i_linked_<well>.csv)",
    )

    parser.add_argument(
        "--preview-only", action="store_true",
        help="Skip linking; just regenerate the overlay preview PNG from the existing linked CSV.",
    )

    parser.add_argument(
        "--crop-size", type=int, default=1024,
        help="Preview crop size in pixels at pyramid level 2 (default: 1024)",
    )

    args = parser.parse_args()

    # Resolve experiment name
    from ops_utils.data.filesystem import resolve_experiment_name
    experiment = resolve_experiment_name(args.experiment, allow_interactive=True)
    if experiment is None:
        print("Error: Could not resolve experiment name")
        return 1

    mode = "4i" if args.four_i else "cp"

    # --preview-only: skip linking, just regenerate the overlay PNG
    if args.preview_only:
        for w in args.wells:
            generate_linking_preview(
                experiment=experiment, well=w,
                crop_size=args.crop_size,
                mode=mode,
            )
        return 0

    # Full linking mode (preview is generated automatically for each well)
    results = link_cell_painting(
        experiment=experiment,
        wells=args.wells,
        verbose=not args.quiet,
        n_jobs=args.n_jobs,
        mode=mode,
    )

    # Print final summary
    print(f"\n{'='*60}")
    print("Final Summary")
    print(f"{'='*60}")
    for well, data in results.items():
        m = data["metrics"]
        print(f"  {well}: {m['total_cells']} cells, "
              f"{m.get('cells_with_barcode', 0)} with barcode, "
              f"{m.get('cells_with_gene', 0)} with gene")

    return 0


if __name__ == "__main__":
    exit(main())
