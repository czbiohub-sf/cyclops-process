"""
Core utility functions for automatic registration.

Contains:
- Centroid extraction from segmentation masks
- Affine transformation utilities
- YAML I/O for affine transforms
- Comparison utilities
"""

import os
import numpy as np
import yaml
import zarr
from pathlib import Path
from skimage.measure import regionprops, moments_hu
from scipy.spatial.distance import cdist


_PHENO_V3_STORE_NAME = "phenotyping_v3.zarr"
_PHENO_NUCLEAR_SEG_5X_LEVEL = 2  # 20x level 0 -> 5x at level 2


def resolve_seg_array_path(seg_zarr_path, position: str, level: int = 0) -> Path:
    """Zarr array path for a seg store + position.

    Standalone seg stores resolve to ``{position}/{level}``; the v3 pheno store
    redirects to its ``nuclear_seg`` label at the 5x pyramid level.
    """
    seg_zarr_path = Path(seg_zarr_path)
    if seg_zarr_path.name == _PHENO_V3_STORE_NAME:
        return (
            seg_zarr_path / position / "labels"
            / "nuclear_seg" / str(_PHENO_NUCLEAR_SEG_5X_LEVEL)
        )
    return seg_zarr_path / position / str(level)


def _parallel_centroids_and_areas(label_image, n_threads=16):
    """Compute centroids and areas for all labels using parallel chunked bincount.

    Splits the flat label array into chunks, runs bincount per chunk in parallel,
    sums partial results. ~4x faster than scipy.ndimage.center_of_mass for large images.

    Returns (areas, sum_y, sum_x) arrays indexed by label.
    """
    from concurrent.futures import ThreadPoolExecutor

    labels_flat = label_image.ravel()
    Y, X = label_image.shape
    n_pixels = len(labels_flat)
    n_labels = int(labels_flat.max()) + 1

    chunk_size = (n_pixels + n_threads - 1) // n_threads

    def _process_chunk(start):
        end = min(start + chunk_size, n_pixels)
        chunk_labels = labels_flat[start:end]
        chunk_areas = np.bincount(chunk_labels, minlength=n_labels)
        # Compute row and col from flat indices
        chunk_rows = np.arange(start, end, dtype=np.float64) // X
        chunk_cols = np.arange(start, end, dtype=np.float64) % X
        chunk_sum_y = np.bincount(chunk_labels, weights=chunk_rows, minlength=n_labels)
        chunk_sum_x = np.bincount(chunk_labels, weights=chunk_cols, minlength=n_labels)
        return chunk_areas, chunk_sum_y, chunk_sum_x

    starts = list(range(0, n_pixels, chunk_size))

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        results = list(pool.map(_process_chunk, starts))

    # Sum partial results
    areas = np.zeros(n_labels, dtype=np.int64)
    sum_y = np.zeros(n_labels, dtype=np.float64)
    sum_x = np.zeros(n_labels, dtype=np.float64)
    for chunk_areas, chunk_sum_y, chunk_sum_x in results:
        areas += chunk_areas
        sum_y += chunk_sum_y
        sum_x += chunk_sum_x

    return areas, sum_y, sum_x


def _load_seg_2d(seg_zarr_path, position, t_idx=0, n_threads=16):
    """Load a 2D label image from a segmentation zarr store.

    Uses threaded chunk reads for ~12x faster NFS I/O (0.5s vs 6.5s for 27K×27K).
    """
    from concurrent.futures import ThreadPoolExecutor

    arr = zarr.open(str(resolve_seg_array_path(seg_zarr_path, position)), mode="r")
    h, w = arr.shape[-2], arr.shape[-1]
    ch, cw = arr.chunks[-2], arr.chunks[-1]

    # Determine which slice indices to use for leading dims.
    # Clamp t_idx to the leading (time) dim's actual length: ISS segmentation is
    # single-timepoint (T=1), but iss_to_track registration is called with a
    # tracking timepoint index >= 1 -> previously raised
    # `BoundsCheckError: index out of bounds for dimension with length 1`.
    # For a length-1 dim this yields index 0 (the only/correct timepoint); for a
    # store with a valid t_idx it is a no-op. (Fix 0n, 2026-06-11.)
    t = min(t_idx, arr.shape[0] - 1) if arr.ndim >= 4 else 0
    if arr.ndim == 5:
        lead = (t, 0, 0)
    elif arr.ndim == 4:
        lead = (t, 0)
    elif arr.ndim == 3:
        lead = (0,)
    else:
        lead = ()

    out = np.empty((h, w), dtype=arr.dtype)
    ny = (h + ch - 1) // ch
    nx = (w + cw - 1) // cw

    def _read_chunk(ty_tx):
        ty, tx = ty_tx
        y0, y1 = ty * ch, min((ty + 1) * ch, h)
        x0, x1 = tx * cw, min((tx + 1) * cw, w)
        out[y0:y1, x0:x1] = arr[lead + (slice(y0, y1), slice(x0, x1))]

    tiles = [(ty, tx) for ty in range(ny) for tx in range(nx)]
    if n_threads > 1 and len(tiles) > 1:
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            list(pool.map(_read_chunk, tiles))
    else:
        for t in tiles:
            _read_chunk(t)

    return out

try:
    import cupy as xp
    from cupyx.scipy import ndimage as cundi

    if xp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("No CUDA device available")
except (ModuleNotFoundError, ImportError, RuntimeError):
    import numpy as xp
    from scipy import ndimage as cundi


import hashlib


def _get_modality_from_path(seg_path: Path) -> str:
    """Extract human-readable modality name from segmentation path."""
    seg_name = seg_path.name.replace("_segmentation_stitched.zarr", "").replace("_segmentation.zarr", "")
    modality_map = {
        "tracking": "track5x",
        "phenotyping": "pheno20x",
        "bc": "iss",
    }
    return modality_map.get(seg_name, seg_name)


def _format_position(position: str) -> str:
    """Format position string (A/1/0 -> A1, B/2/0 -> B2)."""
    parts = position.split("/")
    return f"{parts[0]}{parts[1]}"


class AutoRegistrationError(Exception):
    """Raised when auto-registration fails validation checks."""
    pass


def validate_registration_results(
    experiment: str,
    result: dict,
    wells: list = [1, 2, 3],
    registration_types: list[str] = ["iss", "pheno"],
    min_centroid_overlap_threshold: float = 9.0,
    verbose: bool = True,
    skip_track: bool = False,
) -> tuple[bool, list[str]]:
    """
    Validate auto-registration results by checking individual metrics files.

    Checks:
    1. Any SLURM jobs that failed
    2. Any registration with overlap below threshold (from individual auto_overlays metrics)

    Parameters
    ----------
    experiment : str
        Experiment name
    result : dict
        Result dict from submit_registration_jobs containing 'failed' and 'completed' lists
    wells : list[int]
        Wells that were processed
    registration_types : list[str]
        Registration types that were run ("iss", "pheno")
    min_centroid_overlap_threshold : float
        Minimum acceptable overlap percentage (default: 10.0%)
    verbose : bool
        Print validation details

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, error_messages) - is_valid is True if all checks pass
    """
    import re
    import pandas as pd
    from ops_utils.data.experiment import OpsDataset
    from ops_utils.data.filesystem import parse_well

    errors = []
    # Full row/col tokens (e.g. {"A1", "B1"}) so A/B rows never collide on col alone
    well_tokens = {f"{r}{c}" for r, c in (parse_well(w) for w in wells)}

    # Check for failed jobs
    failed_jobs = result.get("failed", [])
    if failed_jobs:
        error_msg = f"{len(failed_jobs)} registration job(s) failed:"
        for failed_item in failed_jobs:
            if isinstance(failed_item, tuple):
                name, job_id = failed_item
                error_msg += f"\n  - {name} [job: {job_id}]"
            else:
                error_msg += f"\n  - {failed_item}"
        errors.append(error_msg)

    # Check overlap metrics from individual auto_overlays directories
    dataset = OpsDataset(experiment)
    auto_overlays_dir = dataset.tracking / "auto_overlays"

    if auto_overlays_dir.exists():
        # Scan for all overlay subdirectories matching pattern
        pattern = re.compile(r"([A-Za-z]+\d+)_(iss|pheno)_to_(track|pheno|iss)")
        low_overlap_jobs = []

        for overlay_subdir in auto_overlays_dir.iterdir():
            if not overlay_subdir.is_dir():
                continue

            match = pattern.match(overlay_subdir.name)
            if not match:
                continue

            well_token = match.group(1)  # full row/col token, e.g. "A1" or "B1"
            source_type = match.group(2)  # "iss" or "pheno"
            target_type = match.group(3)  # "track", "pheno", or "iss"

            # Skip if this well/type wasn't in our job list
            if well_token not in well_tokens:
                continue
            if source_type not in registration_types:
                continue

            # Filter by expected target based on skip_track mode
            # skip_track: iss->pheno, pheno->iss; normal: iss->track, pheno->track
            if skip_track and target_type == "track":
                continue
            if not skip_track and target_type in ("pheno", "iss"):
                continue

            metrics_csv = overlay_subdir / "auto_register_metrics.csv"
            if not metrics_csv.exists():
                continue

            try:
                df_metrics = pd.read_csv(metrics_csv, index_col=0)
                col = df_metrics.columns[0]
                centroid = float(df_metrics.loc["overlap_forward_overlap_percent", col]) if "overlap_forward_overlap_percent" in df_metrics.index else 0.0
                mask = float(df_metrics.loc["mask_overlap_forward_overlap_percent", col]) if "mask_overlap_forward_overlap_percent" in df_metrics.index else None
                if pd.isna(centroid):
                    centroid = 0.0
                if mask is not None and pd.isna(mask):
                    mask = None

                # Use registration_passed from auto_register_orchestrator
                from cyclops_process.processes.auto_register.auto_register_orchestrator import registration_passed
                if not registration_passed(centroid, mask):
                    mask_str = f", mask={mask:.1f}%" if mask is not None else ""
                    low_overlap_jobs.append(f"{overlay_subdir.name}: centroid={centroid:.1f}%{mask_str}")
            except Exception:
                pass

        if low_overlap_jobs:
            error_msg = f"{len(low_overlap_jobs)} registration(s) failed quality gate:"
            for job in low_overlap_jobs:
                error_msg += f"\n  - {job}"
            errors.append(error_msg)

    is_valid = len(errors) == 0

    if verbose:
        if is_valid:
            print(f"\n✓ Registration validation passed for {experiment}")
        else:
            print(f"\n{'='*60}")
            print(f"🚨 REGISTRATION VALIDATION FAILED for {experiment}")
            print(f"{'='*60}")
            for error in errors:
                print(f"\n{error}")
            print(f"\n{'='*60}")
            print("Pipeline will stop here. Please review and fix registration issues before continuing.")
            print("Check SLURM logs for failed job details: slurm_logs/slurm_auto_register_logs/")
            print(f"{'='*60}\n")

    return is_valid, errors


def print_overlap_quality_warning(overlap_percent: float, indent: str = "", threshold: float = 9.0) -> None:
    """
    Print quality warning/status based on overlap percentage.

    Parameters
    ----------
    overlap_percent : float
        Overlap percentage (0-100)
    indent : str
        String to prepend to each line for indentation
    threshold : float
        Minimum acceptable overlap percentage (default: 9.0%)
    """
    warn_threshold = threshold * 2  # warning zone above the failure threshold

    if overlap_percent < threshold:
        print(f"\n{indent}🚨 CRITICAL: Final overlap ({overlap_percent:.1f}%) is below threshold ({threshold}%)")
        print(f"{indent}Possible causes:")
        print(f"{indent}  - Poor PCC pre-alignment")
        print(f"{indent}  - Insufficient high-quality graph matches")
        print(f"{indent}  - Low RANSAC inlier ratio")
        print(f"{indent}  - Large tissue movement/deformation between modalities")
        print(f"{indent}Recommendation: Review graph match visualization and RANSAC metrics\n")
    elif overlap_percent < warn_threshold:
        print(f"\n{indent}⚠️  WARNING: Final overlap is low ({overlap_percent:.1f}%)")
        print(f"{indent}Above threshold ({threshold}%) but below {warn_threshold:.0f}%")
        print(f"{indent}Registration may be suboptimal - review graph match visualization\n")
    else:
        print(f"\n{indent}✓ Good overlap ({overlap_percent:.1f}%) - registration successful!\n")


def _create_cache_hash(cache_key: str, length: int = 8) -> str:
    """Create a short hash from cache key."""
    return hashlib.md5(cache_key.encode()).hexdigest()[:length]


def _get_centroid_cache_path(
    seg_path: Path,
    position: str,
    t_idx: int,
    min_cell_area: int,
    center_fraction: float,
    bins_to_select: int = None,
    grid_size: int = None,
    cache_subdir: str = "2-tracking/cache/centroids",
) -> Path:
    """Generate cache file path for extracted centroids."""
    # Create a unique hash from input parameters
    if bins_to_select is not None:
        cache_key = f"{seg_path}_{position}_{t_idx}_{min_cell_area}_{center_fraction}_{bins_to_select}_{grid_size}"
        subsample_str = f"_bins{bins_to_select}"
    else:
        cache_key = f"{seg_path}_{position}_{t_idx}_{min_cell_area}_{center_fraction}"
        subsample_str = ""

    cache_hash = _create_cache_hash(cache_key)

    # Navigate from segmentation path to experiment root
    # seg_path structure: experiment_root/1-preprocess/modality/segmentation/name.zarr
    experiment_root = seg_path.parents[3]  # Go up from 1-preprocess/modality/segmentation/name.zarr to experiment_root
    cache_dir = experiment_root / cache_subdir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create human-readable prefix
    modality = _get_modality_from_path(seg_path)
    pos_str = _format_position(position)
    cf_str = f"cf{center_fraction:.1f}".replace(".", "p")
    human_prefix = f"{modality}_{pos_str}_t{t_idx}_{cf_str}{subsample_str}"

    return cache_dir / f"{human_prefix}_{cache_hash}.npy"


def spatial_grid_subsample(
    source_centroids: np.ndarray,
    target_centroids: np.ndarray,
    bins_to_select: int,
    grid_size: int = 50,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Subsample centroids using spatial grid to preserve local correspondence.

    Divides the well into a grid and selects ALL cells from a subset of grid bins.
    This ensures that every cell in the selected region has potential matches nearby,
    preserving cell-to-cell correspondence much better than random sampling.

    Parameters
    ----------
    source_centroids : np.ndarray
        (N, 2) array of source centroids in (y, x) format.
    target_centroids : np.ndarray
        (M, 2) array of target centroids in (y, x) format.
    bins_to_select : int
        Number of bins to randomly select from grid.
    grid_size : int
        Number of bins per dimension (total bins = grid_size²).

    Returns
    -------
    tuple
        (source_indices, target_indices, grid_info)
        - source_indices: Indices of sampled source cells
        - target_indices: Indices of sampled target cells
        - grid_info: Dict with grid bounds and selected bins for visualization
    """
    # Compute bounding box covering both source and target
    all_centroids = np.vstack([source_centroids, target_centroids])
    y_min, x_min = all_centroids.min(axis=0)
    y_max, x_max = all_centroids.max(axis=0)

    # Create bin edges
    y_bins = np.linspace(y_min, y_max, grid_size + 1)
    x_bins = np.linspace(x_min, x_max, grid_size + 1)

    # Assign each centroid to a grid cell
    source_y_idx = np.digitize(source_centroids[:, 0], y_bins) - 1
    source_x_idx = np.digitize(source_centroids[:, 1], x_bins) - 1
    target_y_idx = np.digitize(target_centroids[:, 0], y_bins) - 1
    target_x_idx = np.digitize(target_centroids[:, 1], x_bins) - 1

    # Clip to valid range (edge cases)
    source_y_idx = np.clip(source_y_idx, 0, grid_size - 1)
    source_x_idx = np.clip(source_x_idx, 0, grid_size - 1)
    target_y_idx = np.clip(target_y_idx, 0, grid_size - 1)
    target_x_idx = np.clip(target_x_idx, 0, grid_size - 1)

    # Count cells per bin
    bin_counts_target = {}
    bin_counts_source = {}
    for i in range(grid_size):
        for j in range(grid_size):
            target_in_bin = (target_y_idx == i) & (target_x_idx == j)
            source_in_bin = (source_y_idx == i) & (source_x_idx == j)
            target_count_bin = np.sum(target_in_bin)
            source_count_bin = np.sum(source_in_bin)
            # Only include bins that have cells in BOTH source and target
            if target_count_bin > 0 and source_count_bin > 0:
                bin_counts_target[(i, j)] = target_count_bin
                bin_counts_source[(i, j)] = source_count_bin

    # Strategy: Sample bins randomly across the well for spatial distribution
    # Exclude outer regions (edges often have artifacts or fewer cells)
    # Only select from inner 60% region (20% margin on each side)
    edge_margin = int(grid_size * 0.20)  # 20% margin on each side for inner 60%
    inner_i_min = edge_margin
    inner_i_max = grid_size - edge_margin
    inner_j_min = edge_margin
    inner_j_max = grid_size - edge_margin

    # Filter to only include bins in inner region
    inner_bins = [
        bin_id
        for bin_id in bin_counts_target.keys()
        if (
            inner_i_min <= bin_id[0] < inner_i_max
            and inner_j_min <= bin_id[1] < inner_j_max
        )
    ]

    # Shuffle bins for random spatial sampling
    np.random.seed(42)  # Reproducible
    np.random.shuffle(inner_bins)

    # Select specified number of bins from inner region
    n_bins_to_select = min(bins_to_select, len(inner_bins))
    selected_bins = inner_bins[:n_bins_to_select]

    # Now take ALL cells from selected bins
    sampled_source = []
    sampled_target = []

    for i, j in selected_bins:
        # Find all cells in this bin
        source_in_bin = (source_y_idx == i) & (source_x_idx == j)
        target_in_bin = (target_y_idx == i) & (target_x_idx == j)

        source_bin_idx = np.where(source_in_bin)[0]
        target_bin_idx = np.where(target_in_bin)[0]

        # Take ALL cells from both source and target in this bin
        sampled_source.extend(source_bin_idx)
        sampled_target.extend(target_bin_idx)

    source_indices = np.array(sampled_source, dtype=np.int64)
    target_indices = np.array(sampled_target, dtype=np.int64)

    # Store grid info for visualization
    grid_info = {
        "y_bins": y_bins,
        "x_bins": x_bins,
        "grid_size": grid_size,
        "y_min": y_min,
        "y_max": y_max,
        "x_min": x_min,
        "x_max": x_max,
        "selected_bins": selected_bins,  # List of (i, j) tuples for selected bins
    }

    return source_indices, target_indices, grid_info


def extract_centroids_from_segmentation_subsampled(
    seg_zarr_path: Path,
    position: str,
    t_idx: int = 0,
    min_area: int = 100,
    bins_to_select: int = 100,
    grid_size: int = 50,
    cache_subdir: str = "2-tracking/cache/centroids",
) -> np.ndarray:
    """
    Extract cell centroids from SUBSAMPLED spatial grid bins (fast).

    Only processes selected grid bins from inner 60% of well, dramatically
    reducing computation time for registration purposes.

    Parameters
    ----------
    seg_zarr_path : Path
        Path to segmentation zarr.
    position : str
        Position string (e.g., "A/1/0").
    t_idx : int
        Time index.
    min_area : int
        Minimum cell area filter.
    bins_to_select : int
        Number of spatial grid bins to process.
    grid_size : int
        Grid size (total bins = grid_size²).
    cache_subdir : str
        Cache subdirectory relative to experiment root (default: "2-tracking/cache/centroids").

    Returns
    -------
    np.ndarray
        (N, 2) array of centroids from sampled regions.
    """
    # Check cache first
    cache_path = _get_centroid_cache_path(
        seg_zarr_path, position, t_idx, min_area, 1.0, bins_to_select, grid_size, cache_subdir
    )
    if cache_path.exists():
        try:
            return np.load(cache_path)
        except (EOFError, ValueError, OSError):
            pass  # corrupt or partial write from concurrent job — recompute and overwrite

    # Read zarr metadata to get shape without loading full image
    arr = zarr.open(str(resolve_seg_array_path(seg_zarr_path, position)), mode="r")
    Y, X = arr.shape[-2], arr.shape[-1]

    # Determine leading slice indices
    if arr.ndim == 5:
        lead = (t_idx, 0, 0)
    elif arr.ndim == 4:
        lead = (t_idx, 0)
    elif arr.ndim == 3:
        lead = (0,)
    else:
        lead = ()

    # Create spatial grid bins
    y_bins = np.linspace(0, Y, grid_size + 1)
    x_bins = np.linspace(0, X, grid_size + 1)

    # Select from inner 60% region (20% margin on each side)
    edge_margin = int(grid_size * 0.20)
    inner_i_min = edge_margin
    inner_i_max = grid_size - edge_margin
    inner_j_min = edge_margin
    inner_j_max = grid_size - edge_margin

    # Get all inner bins
    inner_bins = [
        (i, j)
        for i in range(inner_i_min, inner_i_max)
        for j in range(inner_j_min, inner_j_max)
    ]

    # Shuffle for random spatial sampling (reproducible)
    np.random.seed(42)
    np.random.shuffle(inner_bins)

    # Select bins
    n_bins_to_select = min(bins_to_select, len(inner_bins))
    selected_bins = inner_bins[:n_bins_to_select]

    # Read only the selected bins from zarr (not the full image).
    # Uses parallel reads + parallel bincount for centroids.
    from concurrent.futures import ThreadPoolExecutor

    def _extract_bin(ij):
        i, j = ij
        y_start = int(y_bins[i])
        y_end = int(y_bins[i + 1])
        x_start = int(x_bins[j])
        x_end = int(x_bins[j + 1])

        # Read only this bin's region from zarr
        region = np.asarray(arr[lead + (slice(y_start, y_end), slice(x_start, x_end))])

        # Get centroids via bincount (faster than regionprops)
        region_flat = region.ravel()
        n_labels = int(region_flat.max()) + 1 if region_flat.size > 0 else 0
        if n_labels <= 1:
            return []

        rh, rw = region.shape[-2], region.shape[-1]
        rows = np.arange(region_flat.size, dtype=np.float64) // rw
        cols = np.arange(region_flat.size, dtype=np.float64) % rw
        areas = np.bincount(region_flat, minlength=n_labels)
        sum_y = np.bincount(region_flat, weights=rows, minlength=n_labels)
        sum_x = np.bincount(region_flat, weights=cols, minlength=n_labels)

        valid = np.where((areas >= min_area) & (np.arange(n_labels) > 0))[0]
        if len(valid) == 0:
            return []
        cy = sum_y[valid] / areas[valid] + y_start
        cx = sum_x[valid] / areas[valid] + x_start
        return list(np.column_stack([cy, cx]))

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(_extract_bin, selected_bins))

    all_centroids = [c for batch in results for c in batch]
    centroids = np.array(all_centroids) if all_centroids else np.empty((0, 2))

    # Save to cache via per-PID temp file + atomic rename to avoid concurrent write races
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(f'.npy.{os.getpid()}.tmp')
    np.save(tmp_path, centroids)
    try:
        tmp_path.rename(cache_path)
    except OSError:
        tmp_path.unlink(missing_ok=True)  # another job won the race; data is identical

    return centroids


def extract_centroids_from_segmentation(
    seg_zarr_path: Path,
    position: str,
    t_idx: int = 0,
    min_area: int = 100,
    center_fraction: float = 1.0,
) -> np.ndarray:
    """Extract cell centroids from segmentation mask.

    Uses parallel chunked bincount for centroids and areas — utilizes all cores.
    """
    label_image = _load_seg_2d(seg_zarr_path, position, t_idx)
    Y, X = label_image.shape

    # Parallel centroid + area computation using all available cores
    areas, sum_y, sum_x = _parallel_centroids_and_areas(label_image)

    # Valid labels: area >= min_area, not background
    n_labels = len(areas)
    valid_labels = np.where((areas >= min_area) & (np.arange(n_labels) > 0))[0]

    # Centroids = sum of coords / area
    cy = sum_y[valid_labels] / areas[valid_labels]
    cx = sum_x[valid_labels] / areas[valid_labels]
    centroids = np.column_stack([cy, cx])

    # Filter to central region if requested
    if center_fraction < 1.0 and len(centroids) > 0:
        y_center, x_center = Y / 2, X / 2
        y_half_width = (Y * center_fraction) / 2
        x_half_width = (X * center_fraction) / 2

        mask = (
            (centroids[:, 0] >= y_center - y_half_width)
            & (centroids[:, 0] <= y_center + y_half_width)
            & (centroids[:, 1] >= x_center - x_half_width)
            & (centroids[:, 1] <= x_center + x_half_width)
        )
        centroids = centroids[mask]

    return centroids


def _compute_hu_for_single_region(region_moments_normalized):
    """
    Helper function to compute Hu moments from pre-computed normalized moments.

    Takes only normalized moments (not entire region object) to avoid pickling issues.
    """
    hu = moments_hu(region_moments_normalized)
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu_log


def compute_hu_moments_for_cells(
    seg_zarr_path: Path,
    position: str,
    centroids: np.ndarray,
    t_idx: int = 0,
    min_area: int = 100,
    n_workers: int = None,
    verbose: bool = False,
    cache_dir: Path = None,
) -> tuple[np.ndarray, np.ndarray, None]:
    """
    Compute Hu moments for each cell.

    Hu moments are 7 scale/rotation/translation invariant shape descriptors.

    Parameters
    ----------
    seg_zarr_path : Path
        Path to segmentation zarr.
    position : str
        Position string (e.g., "A/1/0").
    centroids : np.ndarray
        (N, 2) array of centroids to match with cell labels.
    t_idx : int
        Time index.
    min_area : int
        Minimum cell area filter.
    n_workers : int, optional
        Number of parallel workers (default: auto-detect via resource_manager).
    verbose : bool
        Show progress bar.
    cache_dir : Path, optional
        Directory for caching regionprops and Hu moments.

    Returns
    -------
    tuple
        (hu_moments, cell_labels, None)
        - hu_moments: (N, 7) array of Hu moments for each centroid
        - cell_labels: (N,) array of corresponding cell labels
        - None: Placeholder for backward compatibility
    """
    from multiprocessing import Pool
    from tqdm import tqdm
    from ops_utils.hpc.resource_manager import get_optimal_workers
    import time
    import hashlib
    import pickle

    t_start = time.time()

    # Create cache key based on parameters
    cache_key_str = f"{seg_zarr_path}_{position}_{t_idx}_{min_area}_{len(centroids)}"
    cache_hash = _create_cache_hash(cache_key_str)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create human-readable prefix
        modality = _get_modality_from_path(seg_zarr_path)
        pos_str = _format_position(position)
        human_prefix = f"{modality}_{pos_str}_t{t_idx}_n{len(centroids)}"

        regionprops_cache = cache_dir / f"regionprops_{human_prefix}_{cache_hash}.pkl"
        hu_cache = cache_dir / f"hu_moments_{human_prefix}_{cache_hash}.npy"
        labels_cache = cache_dir / f"hu_labels_{human_prefix}_{cache_hash}.npy"

        # Check if Hu moments cache exists
        if hu_cache.exists() and labels_cache.exists():
            if verbose:
                print(f"      Using cached Hu moments: {hu_cache.name}")
            hu_moments = np.load(hu_cache)
            cell_labels = np.load(labels_cache)
            return hu_moments, cell_labels, None

    # Load image with parallel chunk reads
    label_image = _load_seg_2d(seg_zarr_path, position, t_idx)

    if verbose:
        print(f"      Loaded label image ({time.time()-t_start:.2f}s)")

    # Match centroids to labels via pixel lookup + KDTree fallback for misses.
    # Pixel lookup is instant; KDTree handles ISS centroids that land on background.
    t_match = time.time()
    centroid_y = np.clip(np.round(centroids[:, 0]).astype(int), 0, label_image.shape[0] - 1)
    centroid_x = np.clip(np.round(centroids[:, 1]).astype(int), 0, label_image.shape[1] - 1)
    matched_labels_arr = label_image[centroid_y, centroid_x]

    # For centroids that landed on background (label=0), use KDTree to find nearest cell
    # Vectorized centroid computation via bincount (replaces slow center_of_mass)
    unmatched = matched_labels_arr == 0
    n_unmatched = int(np.sum(unmatched))
    if n_unmatched > 0 and n_unmatched < len(centroids):
        areas, sum_y, sum_x = _parallel_centroids_and_areas(label_image)
        n_lab = len(areas)
        valid_mask = (areas >= min_area) & (np.arange(n_lab) > 0)
        valid_labels = np.where(valid_mask)[0]
        if len(valid_labels) > 0:
            from scipy.spatial import cKDTree
            cy = sum_y[valid_labels] / areas[valid_labels]
            cx = sum_x[valid_labels] / areas[valid_labels]
            all_coms = np.column_stack([cy, cx])
            tree = cKDTree(all_coms)
            _, nearest_idx = tree.query(centroids[unmatched], k=1)
            matched_labels_arr[unmatched] = valid_labels[nearest_idx]

    if verbose:
        print(f"      Matched labels: {len(centroids) - n_unmatched} direct, {n_unmatched} KDTree fallback ({time.time()-t_match:.2f}s)")

    # Compute Hu moments per cell using bounding-box extraction (parallel).
    # Uses scipy.ndimage.find_objects for O(pixels) bbox computation (not O(labels × pixels)).
    from skimage.measure import moments as _moments, moments_normalized as _moments_norm
    from scipy.ndimage import find_objects as _find_objects
    from concurrent.futures import ThreadPoolExecutor

    t_hu = time.time()

    unique_labels = np.unique(matched_labels_arr)
    unique_labels = unique_labels[unique_labels > 0]

    # find_objects returns bounding box slices for ALL labels in one pass — O(pixels)
    t_bbox = time.time()
    all_slices = _find_objects(label_image)  # index i = label i+1
    if verbose:
        print(f"      find_objects: {time.time()-t_bbox:.2f}s for {len(all_slices)} labels")

    def _compute_hu_for_label(lbl):
        """Extract cell's bounding box, compute normalized moments → Hu moments."""
        idx = int(lbl) - 1  # find_objects is 0-indexed for label 1
        if idx < 0 or idx >= len(all_slices) or all_slices[idx] is None:
            return np.zeros(7)
        sl = all_slices[idx]
        patch = (label_image[sl] == lbl).astype(np.float64)
        m = _moments(patch, order=3)
        if m[0, 0] == 0:
            return np.zeros(7)
        mn = _moments_norm(m, order=3)
        return moments_hu(mn)

    # Compute Hu moments for all unique matched labels in parallel (32 threads)
    hu_moments = np.zeros((len(centroids), 7), dtype=np.float64)
    cell_labels = matched_labels_arr.copy()

    label_to_hu = {}
    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = {lbl: pool.submit(_compute_hu_for_label, lbl) for lbl in unique_labels}
        for lbl, fut in futures.items():
            label_to_hu[lbl] = fut.result()

    for i, lbl in enumerate(matched_labels_arr):
        if lbl > 0 and lbl in label_to_hu:
            hu_moments[i] = label_to_hu[lbl]

    if verbose:
        n_computed = len(label_to_hu)
        print(f"      Computed Hu moments for {n_computed} unique cells, {len(centroids)} centroids ({time.time()-t_hu:.2f}s)")

    # Cache Hu moments if cache_dir provided
    if cache_dir is not None:
        np.save(hu_cache, hu_moments)
        np.save(labels_cache, cell_labels)
        if verbose:
            print(f"      Cached Hu moments to: {hu_cache.name}")

    return hu_moments, cell_labels, None


def _compute_best_hu_match_knn(args):
    """
    Helper function for parallel k-NN Hu moment matching.

    Must be at module level for multiprocessing pickling.
    """
    src_idx, tgt_candidates_idx, src_hu, target_hu_all = args
    # Get Hu moments for k candidate targets
    tgt_hu_candidates = target_hu_all[tgt_candidates_idx]  # (k, 7)
    # Compute L2 distance in Hu space
    hu_dists = np.linalg.norm(tgt_hu_candidates - src_hu, axis=1)  # (k,)
    # Select best Hu match among k candidates
    best_k_idx = np.argmin(hu_dists)
    best_tgt_idx = tgt_candidates_idx[best_k_idx]
    best_hu_dist = hu_dists[best_k_idx]
    return (src_idx, best_tgt_idx, best_hu_dist)


def match_cells_by_hu_moments(
    source_centroids: np.ndarray,
    target_centroids: np.ndarray,
    source_hu_moments: np.ndarray,
    target_hu_moments: np.ndarray,
    max_spatial_distance: float = 50.0,
    max_hu_distance: float = 0.5,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match cells between source and target using Hu moments similarity.

    Strategy:
    1. Find spatially nearby pairs (within max_spatial_distance)
    2. Among nearby pairs, keep only those with similar Hu moments (< max_hu_distance)
    3. Use 1-to-1 matching (greedy: best matches first)

    Parameters
    ----------
    source_centroids : np.ndarray
        (N, 2) source centroids (should be PCC-aligned).
    target_centroids : np.ndarray
        (M, 2) target centroids.
    source_hu_moments : np.ndarray
        (N, 7) Hu moments for source cells.
    target_hu_moments : np.ndarray
        (M, 7) Hu moments for target cells.
    max_spatial_distance : float
        Maximum spatial distance (pixels) to consider as potential match.
    max_hu_distance : float
        Maximum Hu moments distance (L2 norm) to accept as match.
    verbose : bool
        Print statistics.

    Returns
    -------
    tuple
        (source_indices, target_indices, hu_distances)
        - source_indices: Matched source cell indices
        - target_indices: Matched target cell indices
        - hu_distances: Hu moment distances for each match
    """
    n_source = len(source_centroids)
    n_target = len(target_centroids)

    # Compute spatial distance matrix
    spatial_dists = cdist(source_centroids, target_centroids)

    # Compute Hu moment distance matrix
    hu_dists = cdist(source_hu_moments, target_hu_moments)

    # Find candidate pairs (spatially close)
    candidates = np.argwhere(spatial_dists < max_spatial_distance)

    if len(candidates) == 0:
        if verbose:
            print(
                f"      WARNING: No spatially close pairs found (max_dist={max_spatial_distance}px)"
            )
        return np.array([]), np.array([]), np.array([])

    # Filter by Hu moment similarity
    valid_matches = []
    for src_idx, tgt_idx in candidates:
        hu_dist = hu_dists[src_idx, tgt_idx]
        if hu_dist < max_hu_distance:
            valid_matches.append((src_idx, tgt_idx, hu_dist))

    if len(valid_matches) == 0:
        if verbose:
            print(
                f"      WARNING: No Hu moment matches found (max_hu_dist={max_hu_distance})"
            )
            print(
                f"      Spatial candidates: {len(candidates)}, but all failed Hu similarity"
            )
        return np.array([]), np.array([]), np.array([])

    # Sort by Hu distance (best matches first)
    valid_matches.sort(key=lambda x: x[2])

    # Greedy 1-to-1 matching
    used_source = set()
    used_target = set()
    final_matches = []

    for src_idx, tgt_idx, hu_dist in valid_matches:
        if src_idx not in used_source and tgt_idx not in used_target:
            final_matches.append((src_idx, tgt_idx, hu_dist))
            used_source.add(src_idx)
            used_target.add(tgt_idx)

    if len(final_matches) == 0:
        return np.array([]), np.array([]), np.array([])

    # Unpack results
    source_indices = np.array([m[0] for m in final_matches])
    target_indices = np.array([m[1] for m in final_matches])
    hu_distances = np.array([m[2] for m in final_matches])

    if verbose:
        print(
            f"      Hu moment matching: {len(final_matches)}/{n_source} source cells matched"
        )
        print(
            f"        Spatial candidates: {len(candidates)}, Hu-filtered: {len(valid_matches)}, Final 1-to-1: {len(final_matches)}"
        )
        print(
            f"        Hu distance: mean={hu_distances.mean():.3f}, median={np.median(hu_distances):.3f}, max={hu_distances.max():.3f}"
        )

    return source_indices, target_indices, hu_distances


def match_cells_by_hu_moments_knn(
    source_centroids: np.ndarray,
    target_centroids: np.ndarray,
    source_hu_moments: np.ndarray,
    target_hu_moments: np.ndarray,
    k_neighbors: int = 5,
    percentile_threshold: float = 1.0,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match cells using k-NN spatial search + Hu moments, keeping only top percentile.

    Strategy:
    1. For each source cell, find k nearest target neighbors (spatially)
    2. Compute Hu moment distance to each of the k neighbors
    3. Select best Hu match among the k candidates (if any pass threshold)
    4. Keep only top percentile_threshold% of matches (e.g., top 1% = 0.01)
    5. Apply greedy 1-to-1 matching on remaining candidates

    Parameters
    ----------
    source_centroids : np.ndarray
        (N, 2) source centroids (should be PCC-aligned).
    target_centroids : np.ndarray
        (M, 2) target centroids.
    source_hu_moments : np.ndarray
        (N, 7) Hu moments for source cells.
    target_hu_moments : np.ndarray
        (M, 7) Hu moments for target cells.
    k_neighbors : int
        Number of spatial nearest neighbors to evaluate per source cell (default: 5).
    percentile_threshold : float
        Keep only top X% of matches by Hu distance (e.g., 1.0 = top 1%). Range: [0.1, 100].
    verbose : bool
        Print statistics.

    Returns
    -------
    tuple
        (source_indices, target_indices, hu_distances)
        - source_indices: Matched source cell indices
        - target_indices: Matched target cell indices
        - hu_distances: Hu moment distances for each match
    """
    from sklearn.neighbors import NearestNeighbors

    n_source = len(source_centroids)
    n_target = len(target_centroids)

    if k_neighbors > n_target:
        k_neighbors = n_target
        if verbose:
            print(
                f"      Reducing k_neighbors to {k_neighbors} (max available targets)"
            )

    # Build k-NN tree on target centroids
    if verbose:
        print(f"      Building k-NN tree for {n_target} target cells...")
    knn = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean", n_jobs=-1)
    knn.fit(target_centroids)

    # Find k nearest spatial neighbors for each source cell
    if verbose:
        print(
            f"      Finding {k_neighbors} nearest neighbors for {n_source} source cells..."
        )
    distances, indices = knn.kneighbors(source_centroids)

    # For each source cell, evaluate Hu moments of k candidates (parallelized)
    from multiprocessing import Pool
    from tqdm import tqdm
    from ops_utils.hpc.resource_manager import get_optimal_workers
    import time

    t_match_start = time.time()

    # Prepare arguments for parallel processing
    args_list = [
        (src_idx, indices[src_idx], source_hu_moments[src_idx], target_hu_moments)
        for src_idx in range(n_source)
    ]

    # Determine number of workers
    n_workers = get_optimal_workers(
        use_gpu=False,
        model_ram_gb=0.001,
        data_ram_gb=0.001,
        verbose=False,
    )

    if verbose:
        print(
            f"      Evaluating Hu distances for {n_source} source cells (workers={n_workers})..."
        )

    # Parallel Hu distance evaluation with tqdm
    if n_workers > 1:
        with Pool(n_workers) as pool:
            candidate_matches = list(
                tqdm(
                    pool.imap(_compute_best_hu_match_knn, args_list, chunksize=100),
                    total=len(args_list),
                    desc="Matching by Hu moments",
                    disable=not verbose,
                )
            )
    else:
        # Single-threaded fallback
        candidate_matches = [
            _compute_best_hu_match_knn(args)
            for args in tqdm(
                args_list, desc="Matching by Hu moments", disable=not verbose
            )
        ]

    if verbose:
        print(
            f"      Hu distance evaluation completed ({time.time()-t_match_start:.2f}s)"
        )

    # Filter to top percentile by Hu distance
    t_filter = time.time()
    candidate_matches.sort(key=lambda x: x[2])  # Sort by Hu distance (ascending)
    n_keep = max(1, int(len(candidate_matches) * (percentile_threshold / 100.0)))
    top_matches = candidate_matches[:n_keep]

    if verbose:
        print(
            f"      Filtered to top {percentile_threshold}% ({time.time()-t_filter:.2f}s)"
        )
        all_hu_dists = np.array([m[2] for m in candidate_matches])
        top_hu_dists = np.array([m[2] for m in top_matches])
        print(
            f"      k-NN Hu matching: {len(candidate_matches)} candidates → {len(top_matches)} after top {percentile_threshold}% filter"
        )
        print(
            f"        All candidates - Hu distance: mean={all_hu_dists.mean():.3f}, median={np.median(all_hu_dists):.3f}"
        )
        print(
            f"        Top {percentile_threshold}% - Hu distance: mean={top_hu_dists.mean():.3f}, median={np.median(top_hu_dists):.3f}, max={top_hu_dists.max():.3f}"
        )

    # Greedy 1-to-1 matching on top percentile (already sorted by quality)
    t_greedy = time.time()
    used_source = set()
    used_target = set()
    final_matches = []

    for src_idx, tgt_idx, hu_dist in top_matches:
        if src_idx not in used_source and tgt_idx not in used_target:
            final_matches.append((src_idx, tgt_idx, hu_dist))
            used_source.add(src_idx)
            used_target.add(tgt_idx)

    if verbose:
        print(f"      Greedy 1-to-1 matching completed ({time.time()-t_greedy:.2f}s)")

    if len(final_matches) == 0:
        return np.array([]), np.array([]), np.array([])

    # Unpack results
    source_indices = np.array([m[0] for m in final_matches])
    target_indices = np.array([m[1] for m in final_matches])
    hu_distances = np.array([m[2] for m in final_matches])

    if verbose:
        print(
            f"        Final 1-to-1 matches: {len(final_matches)}/{n_source} source cells"
        )

    return source_indices, target_indices, hu_distances


def _compute_best_multifeature_match_knn(args):
    """
    Helper function for parallel k-NN multi-feature matching.

    Computes combined distance using Hu moments + normalized shape features.

    Must be at module level for multiprocessing pickling.
    """
    (
        src_idx,
        tgt_candidates_idx,
        src_hu,
        target_hu_all,
        src_shape,
        target_shape_all,
        hu_weight,
    ) = args

    # Get Hu moments for k candidate targets
    tgt_hu_candidates = target_hu_all[tgt_candidates_idx]  # (k, 7)
    tgt_shape_candidates = target_shape_all[tgt_candidates_idx]  # (k, 5)

    # Compute L2 distance in Hu space (already log-normalized)
    hu_dists = np.linalg.norm(tgt_hu_candidates - src_hu, axis=1)  # (k,)

    # Compute L2 distance in normalized shape space
    # Shape features: [area, perimeter, eccentricity, solidity, extent]
    # Normalize by standard deviation to equalize importance
    shape_dists = np.linalg.norm(tgt_shape_candidates - src_shape, axis=1)  # (k,)

    # Combined distance: weighted average
    combined_dists = hu_weight * hu_dists + (1.0 - hu_weight) * shape_dists

    # Select best match among k candidates
    best_k_idx = np.argmin(combined_dists)
    best_tgt_idx = tgt_candidates_idx[best_k_idx]
    best_combined_dist = combined_dists[best_k_idx]
    best_hu_dist = hu_dists[best_k_idx]
    best_shape_dist = shape_dists[best_k_idx]

    return (src_idx, best_tgt_idx, best_combined_dist, best_hu_dist, best_shape_dist)


def match_cells_by_multifeature_knn(
    source_centroids: np.ndarray,
    target_centroids: np.ndarray,
    source_hu_moments: np.ndarray,
    target_hu_moments: np.ndarray,
    source_shape_features: np.ndarray,
    target_shape_features: np.ndarray,
    k_neighbors: int = 5,
    percentile_threshold: float = 1.0,
    hu_weight: float = 0.7,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match cells using k-NN + Hu moments + shape features (multi-feature matching).

    Strategy:
    1. For each source cell, find k nearest target neighbors (spatially)
    2. Compute combined distance: hu_weight * Hu_dist + (1-hu_weight) * shape_dist
    3. Select best match among the k candidates
    4. Keep only top percentile_threshold% of matches
    5. Apply greedy 1-to-1 matching on remaining candidates

    Parameters
    ----------
    source_centroids : np.ndarray
        (N, 2) source centroids (should be PCC-aligned).
    target_centroids : np.ndarray
        (M, 2) target centroids.
    source_hu_moments : np.ndarray
        (N, 7) Hu moments for source cells.
    target_hu_moments : np.ndarray
        (M, 7) Hu moments for target cells.
    source_shape_features : np.ndarray
        (N, 5) shape features [area, perimeter, eccentricity, solidity, extent].
    target_shape_features : np.ndarray
        (M, 5) shape features [area, perimeter, eccentricity, solidity, extent].
    k_neighbors : int
        Number of spatial nearest neighbors to evaluate per source cell (default: 5).
    percentile_threshold : float
        Keep only top X% of matches by combined distance (e.g., 1.0 = top 1%).
    hu_weight : float
        Weight for Hu moments in combined distance (0-1). Default: 0.7 (70% Hu, 30% shape).
    verbose : bool
        Print statistics.

    Returns
    -------
    tuple
        (source_indices, target_indices, combined_distances)
        - source_indices: Matched source cell indices
        - target_indices: Matched target cell indices
        - combined_distances: Combined distance metric for each match
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    n_source = len(source_centroids)
    n_target = len(target_centroids)

    if k_neighbors > n_target:
        k_neighbors = n_target
        if verbose:
            print(
                f"      Reducing k_neighbors to {k_neighbors} (max available targets)"
            )

    # Normalize shape features using z-score (combine source + target for consistent scaling)
    if verbose:
        print(
            f"      Normalizing shape features (area, perimeter, eccentricity, solidity, extent)..."
        )
    combined_shape = np.vstack([source_shape_features, target_shape_features])
    scaler = StandardScaler()
    scaler.fit(combined_shape)
    source_shape_norm = scaler.transform(source_shape_features)
    target_shape_norm = scaler.transform(target_shape_features)

    # Build k-NN tree on target centroids
    if verbose:
        print(f"      Building k-NN tree for {n_target} target cells...")
    knn = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean", n_jobs=-1)
    knn.fit(target_centroids)

    # Find k nearest spatial neighbors for each source cell
    if verbose:
        print(
            f"      Finding {k_neighbors} nearest neighbors for {n_source} source cells..."
        )
    distances, indices = knn.kneighbors(source_centroids)

    # For each source cell, evaluate combined distance (Hu + shape) for k candidates
    from multiprocessing import Pool
    from tqdm import tqdm
    from ops_utils.hpc.resource_manager import get_optimal_workers
    import time

    t_match_start = time.time()

    # Prepare arguments for parallel processing
    args_list = [
        (
            src_idx,
            indices[src_idx],
            source_hu_moments[src_idx],
            target_hu_moments,
            source_shape_norm[src_idx],
            target_shape_norm,
            hu_weight,
        )
        for src_idx in range(n_source)
    ]

    # Determine number of workers
    n_workers = get_optimal_workers(
        use_gpu=False,
        model_ram_gb=0.001,
        data_ram_gb=0.001,
        verbose=False,
    )

    if verbose:
        print(
            f"      Evaluating multi-feature distances (Hu={hu_weight:.1%}, shape={1-hu_weight:.1%}) for {n_source} cells (workers={n_workers})..."
        )

    # Parallel multi-feature distance evaluation with tqdm
    if n_workers > 1:
        with Pool(n_workers) as pool:
            candidate_matches = list(
                tqdm(
                    pool.imap(
                        _compute_best_multifeature_match_knn, args_list, chunksize=100
                    ),
                    total=len(args_list),
                    desc="Multi-feature matching",
                    disable=not verbose,
                )
            )
    else:
        # Single-threaded fallback
        candidate_matches = [
            _compute_best_multifeature_match_knn(args)
            for args in tqdm(
                args_list, desc="Multi-feature matching", disable=not verbose
            )
        ]

    if verbose:
        print(
            f"      Multi-feature distance evaluation completed ({time.time()-t_match_start:.2f}s)"
        )

    # Filter to top percentile by combined distance
    t_filter = time.time()
    candidate_matches.sort(key=lambda x: x[2])  # Sort by combined distance (ascending)
    n_keep = max(1, int(len(candidate_matches) * (percentile_threshold / 100.0)))
    top_matches = candidate_matches[:n_keep]

    if verbose:
        print(
            f"      Filtered to top {percentile_threshold}% ({time.time()-t_filter:.2f}s)"
        )
        all_combined_dists = np.array([m[2] for m in candidate_matches])
        all_hu_dists = np.array([m[3] for m in candidate_matches])
        all_shape_dists = np.array([m[4] for m in candidate_matches])
        top_combined_dists = np.array([m[2] for m in top_matches])
        top_hu_dists = np.array([m[3] for m in top_matches])
        top_shape_dists = np.array([m[4] for m in top_matches])

        print(
            f"      Multi-feature k-NN: {len(candidate_matches)} candidates → {len(top_matches)} after top {percentile_threshold}% filter"
        )
        print(
            f"        All candidates - Combined: mean={all_combined_dists.mean():.3f}, Hu: mean={all_hu_dists.mean():.3f}, Shape: mean={all_shape_dists.mean():.3f}"
        )
        print(
            f"        Top {percentile_threshold}% - Combined: mean={top_combined_dists.mean():.3f}, Hu: mean={top_hu_dists.mean():.3f}, Shape: mean={top_shape_dists.mean():.3f}"
        )

    # Greedy 1-to-1 matching on top percentile (already sorted by quality)
    t_greedy = time.time()
    used_source = set()
    used_target = set()
    final_matches = []

    for src_idx, tgt_idx, combined_dist, hu_dist, shape_dist in top_matches:
        if src_idx not in used_source and tgt_idx not in used_target:
            final_matches.append((src_idx, tgt_idx, combined_dist))
            used_source.add(src_idx)
            used_target.add(tgt_idx)

    if verbose:
        print(f"      Greedy 1-to-1 matching completed ({time.time()-t_greedy:.2f}s)")

    if len(final_matches) == 0:
        return np.array([]), np.array([]), np.array([])

    # Unpack results
    source_indices = np.array([m[0] for m in final_matches])
    target_indices = np.array([m[1] for m in final_matches])
    combined_distances = np.array([m[2] for m in final_matches])

    if verbose:
        print(
            f"        Final 1-to-1 matches: {len(final_matches)}/{n_source} source cells"
        )

    return source_indices, target_indices, combined_distances


def affine_3x3_to_4x4_zyx(affine_2d: np.ndarray) -> np.ndarray:
    """Convert 2D affine (3x3) to 4x4 homogeneous in ZYX order."""
    affine_4x4 = np.eye(4)
    affine_4x4[0, 0] = 1.0  # Z: identity
    affine_4x4[0, 3] = 0.0
    affine_4x4[1:3, 1:3] = affine_2d[:2, :2]  # YX: rotation/scale
    affine_4x4[1:3, 3] = affine_2d[:2, 2]  # YX: translation
    return affine_4x4


def save_affine_to_yaml(
    affine_4x4: np.ndarray,
    output_path: Path,
    source_channel_names: list[str] = None,
    target_channel_name: str = "segmentation",
):
    """Save affine to biahub-compatible YAML format. Atomic (tmp + os.replace)
    so a concurrent reader never sees a partial file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "source_channel_names": source_channel_names or ["segmentation"],
        "target_channel_name": target_channel_name,
        "affine_transform_zyx": affine_4x4.tolist(),
        "keep_overhang": False,
        "interpolation": "linear",
        "time_indices": "all",
    }

    tmp = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    os.replace(tmp, output_path)


def calculate_mask_overlap(
    mask_src: np.ndarray, mask_tgt: np.ndarray, downsample: int = 8
) -> dict:
    """
    Calculate overlap metrics between binary masks.

    Parameters
    ----------
    mask_src : np.ndarray
        Source mask (already transformed/aligned).
    mask_tgt : np.ndarray
        Target mask.
    downsample : int
        Downsample factor for faster computation (default 8x).

    Returns
    -------
    dict
        Overlap metrics including IoU, Dice coefficient, and overlap percentage.
    """
    from skimage.transform import downscale_local_mean

    # Downsample for faster computation (overlap % is resolution-independent)
    if downsample > 1:
        mask_src = downscale_local_mean(mask_src, (downsample, downsample))
        mask_tgt = downscale_local_mean(mask_tgt, (downsample, downsample))

    # Ensure same shape
    min_h = min(mask_src.shape[0], mask_tgt.shape[0])
    min_w = min(mask_src.shape[1], mask_tgt.shape[1])
    mask_src = mask_src[:min_h, :min_w]
    mask_tgt = mask_tgt[:min_h, :min_w]

    # Convert to binary
    binary_src = (mask_src > 0).astype(bool)
    binary_tgt = (mask_tgt > 0).astype(bool)

    # Calculate overlap metrics
    intersection = np.logical_and(binary_src, binary_tgt).sum()
    union = np.logical_or(binary_src, binary_tgt).sum()
    src_area = binary_src.sum()
    tgt_area = binary_tgt.sum()

    # IoU (Intersection over Union)
    iou = intersection / union if union > 0 else 0.0

    # Dice coefficient (2 * intersection / (src + tgt))
    dice = (
        2 * intersection / (src_area + tgt_area) if (src_area + tgt_area) > 0 else 0.0
    )

    # Overlap percentage (intersection / target area)
    overlap_pct = 100 * intersection / tgt_area if tgt_area > 0 else 0.0

    return {
        "iou": float(iou),
        "dice": float(dice),
        "overlap_percent": float(overlap_pct),
        "intersection_pixels": int(intersection),
        "union_pixels": int(union),
        "source_area_pixels": int(src_area),
        "target_area_pixels": int(tgt_area),
    }


def compute_binary_mask_overlap_metrics(
    mask_src: np.ndarray,
    mask_tgt: np.ndarray,
    pcc_shift: np.ndarray,
    affine_3x3: np.ndarray,
    downsample_factor: int = 8,
    verbose: bool = True,
) -> dict:
    """Compute segmentation mask binary mask overlap at before/pcc/forward stages.

    Unlike centroid-based overlap, this measures whether the segmented tissue
    regions overlap — robust to modality differences where ISS and tracking
    detect different individual cells but image the same tissue.

    Parameters
    ----------
    mask_src, mask_tgt : np.ndarray
        Full-resolution 2D segmentation masks.
    pcc_shift : np.ndarray
        PCC translation [dy, dx].
    affine_3x3 : np.ndarray
        Final 3x3 affine (PCC + RANSAC composition).
    downsample_factor : int
        Downsample for speed (default 8x).
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        ``{"before": {...}, "pcc": {...}, "forward": {...}}`` with IoU/Dice/overlap.
    """
    from scipy import ndimage

    import time

    if verbose:
        print(f"    Computing binary mask overlap (segmentation mask, {downsample_factor}x ds)...")

    t0 = time.time()

    # Downsample
    ds = downsample_factor
    src_ds = (mask_src[::ds, ::ds] > 0).astype(np.float32)
    tgt_ds = (mask_tgt[::ds, ::ds] > 0).astype(np.uint8)

    # Crop to same shape
    min_h = min(src_ds.shape[0], tgt_ds.shape[0])
    min_w = min(src_ds.shape[1], tgt_ds.shape[1])
    src_ds = src_ds[:min_h, :min_w]
    tgt_ds = tgt_ds[:min_h, :min_w]

    # 1. Before alignment
    m_before = calculate_mask_overlap(src_ds.astype(np.uint8), tgt_ds, downsample=1)

    # 2. After PCC only
    if pcc_shift is not None:
        pcc_ds = pcc_shift / ds
        src_pcc = ndimage.shift(src_ds, pcc_ds, order=0)[:min_h, :min_w]
    else:
        src_pcc = src_ds
    m_pcc = calculate_mask_overlap(src_pcc.astype(np.uint8), tgt_ds, downsample=1)

    # 3. After full affine (PCC + RANSAC)
    affine_ds = affine_3x3.copy()
    affine_ds[:2, 2] /= ds
    inv_ds = np.linalg.inv(affine_ds)
    src_fwd = ndimage.affine_transform(
        src_ds, inv_ds[:2, :2], offset=inv_ds[:2, 2],
        output_shape=(min_h, min_w), order=0,
    )
    m_forward = calculate_mask_overlap((src_fwd > 0.5).astype(np.uint8), tgt_ds, downsample=1)

    dt = time.time() - t0
    if verbose:
        print(f"      Before:  IoU={m_before['iou']:.3f}  Dice={m_before['dice']:.3f}  Overlap={m_before['overlap_percent']:.1f}%")
        print(f"      PCC:     IoU={m_pcc['iou']:.3f}  Dice={m_pcc['dice']:.3f}  Overlap={m_pcc['overlap_percent']:.1f}%")
        print(f"      Forward: IoU={m_forward['iou']:.3f}  Dice={m_forward['dice']:.3f}  Overlap={m_forward['overlap_percent']:.1f}%")
        print(f"      ({dt:.2f}s)")

    return {
        "before": m_before,
        "pcc": m_pcc,
        "forward": m_forward,
    }


def compute_registration_overlap_metrics(
    source_centroids: np.ndarray,
    target_centroids: np.ndarray,
    mask_shape: tuple,
    pcc_shift: np.ndarray,
    affine_3x3: np.ndarray,
    centroid_radius: int = 10,
    manual_yaml_path: Path = None,
    output_csv_path: Path = None,
    verbose: bool = True,
) -> dict:
    """
    Compute overlap metrics at multiple registration stages using centroids.

    Parameters
    ----------
    source_centroids : np.ndarray
        (N, 2) array of source centroids in (y, x) format.
    target_centroids : np.ndarray
        (M, 2) array of target centroids in (y, x) format.
    mask_shape : tuple
        Shape of the mask (height, width).
    pcc_shift : np.ndarray
        PCC translation [dy, dx].
    affine_3x3 : np.ndarray
        Final 3x3 affine (PCC + RANSAC composition, forward transform).
    centroid_radius : int
        Radius of circles to draw around each centroid (default 10 for 20x20 circles).
    manual_yaml_path : Path, optional
        Path to manual affine YAML for comparison.
    output_csv_path : Path, optional
        Path to save metrics CSV file.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Overlap metrics at multiple stages: before, pcc, forward, inverse, [manual].
    """
    from skimage.draw import disk
    from skimage.transform import AffineTransform
    import time

    if verbose:
        print(
            f"    Calculating centroid-based overlap metrics (radius={centroid_radius}px)..."
        )

    t_overlap_start = time.time()

    # Downsample mask shape for faster rasterization (we'll downsample by 8x anyway for overlap calc)
    # This is the key speedup - don't rasterize at full resolution!
    downsample_factor = 8
    downsampled_shape = (
        mask_shape[0] // downsample_factor,
        mask_shape[1] // downsample_factor,
    )
    downsampled_radius = max(1, centroid_radius // downsample_factor)

    # Vectorized rasterization: stamp all centroids at once using broadcasting
    def rasterize_centroids(
        centroids: np.ndarray, shape: tuple, radius: int
    ) -> np.ndarray:
        """Create binary mask from centroids — fully vectorized, no Python loop."""
        mask = np.zeros(shape, dtype=bool)
        if len(centroids) == 0:
            return mask

        # Pre-compute disk template
        disk_template_rr, disk_template_cc = disk((0, 0), radius)
        n_disk = len(disk_template_rr)

        # Round centroids to int
        cy = np.round(centroids[:, 0]).astype(np.int32)
        cx = np.round(centroids[:, 1]).astype(np.int32)

        # Broadcast: all_rr[i*n_disk:(i+1)*n_disk] = disk offsets + centroid[i]
        all_rr = np.repeat(cy, n_disk) + np.tile(disk_template_rr, len(cy))
        all_cc = np.repeat(cx, n_disk) + np.tile(disk_template_cc, len(cx))

        # Clip to valid range
        valid = (all_rr >= 0) & (all_rr < shape[0]) & (all_cc >= 0) & (all_cc < shape[1])
        mask[all_rr[valid], all_cc[valid]] = True
        return mask

    # Helper function to transform centroids
    def transform_centroids(
        centroids: np.ndarray, affine_3x3: np.ndarray
    ) -> np.ndarray:
        """Transform centroids using affine matrix."""
        # Convert to homogeneous coordinates
        ones = np.ones((centroids.shape[0], 1))
        centroids_hom = np.hstack([centroids, ones])  # (N, 3) in (y, x, 1) format

        # Apply affine transform
        transformed = (affine_3x3 @ centroids_hom.T).T  # (N, 3)
        return transformed[:, :2]  # Return (y, x)

    # Downsample centroids to match downsampled shape
    source_centroids_ds = source_centroids / downsample_factor
    target_centroids_ds = target_centroids / downsample_factor
    pcc_shift_ds = pcc_shift / downsample_factor

    # Create target mask once (doesn't change)
    if verbose:
        print(f"      Rasterizing target centroids...")
    t_raster = time.time()
    mask_tgt = rasterize_centroids(
        target_centroids_ds, downsampled_shape, downsampled_radius
    )
    if verbose:
        print(f"        ({time.time()-t_raster:.2f}s)")

    # Compute all 3 overlap stages in parallel (each rasterizes + computes IoU independently)
    from concurrent.futures import ThreadPoolExecutor as _OverlapStagePool

    affine_3x3_ds = affine_3x3.copy()
    affine_3x3_ds[:2, 2] /= downsample_factor
    source_centroids_pcc = source_centroids_ds + pcc_shift_ds
    source_centroids_forward = transform_centroids(source_centroids_ds, affine_3x3_ds)

    def _compute_one_overlap(src_centroids, label):
        mask_src = rasterize_centroids(src_centroids, downsampled_shape, downsampled_radius)
        return calculate_mask_overlap(mask_src, mask_tgt, downsample=1)

    if verbose:
        print(f"      Computing 3 overlap stages in parallel...")
    t_ovl = time.time()

    with _OverlapStagePool(max_workers=3) as ovl_pool:
        f_before = ovl_pool.submit(_compute_one_overlap, source_centroids_ds, "before")
        f_pcc = ovl_pool.submit(_compute_one_overlap, source_centroids_pcc, "pcc")
        f_forward = ovl_pool.submit(_compute_one_overlap, source_centroids_forward, "forward")
        overlap_before = f_before.result()
        overlap_pcc = f_pcc.result()
        overlap_forward = f_forward.result()

    if verbose:
        print(f"        ({time.time()-t_ovl:.2f}s for all 3 stages)")

    # 4. Manual affine (if provided)
    results = {
        "before": overlap_before,
        "pcc": overlap_pcc,
        "forward": overlap_forward,
    }

    if manual_yaml_path and manual_yaml_path.exists():
        if verbose:
            print(f"      Computing overlap: Manual affine...")
        t4 = time.time()

        # Load manual affine from YAML
        import yaml

        with open(manual_yaml_path, "r") as f:
            manual_data = yaml.safe_load(f)
        manual_affine_4x4 = np.array(manual_data["affine_transform_zyx"])

        # Extract 2D affine and invert (YAML stores target→source, need source→target)
        manual_affine_3x3 = np.eye(3)
        manual_affine_3x3[:2, :2] = manual_affine_4x4[1:3, 1:3]
        manual_affine_3x3[:2, 2] = manual_affine_4x4[1:3, 3]
        manual_affine_3x3_fwd = np.linalg.inv(manual_affine_3x3)

        # Scale to downsampled space
        manual_affine_3x3_fwd_ds = manual_affine_3x3_fwd.copy()
        manual_affine_3x3_fwd_ds[:2, 2] /= downsample_factor

        # Transform centroids
        source_centroids_manual = transform_centroids(
            source_centroids_ds, manual_affine_3x3_fwd_ds
        )
        mask_src_manual = rasterize_centroids(
            source_centroids_manual, downsampled_shape, downsampled_radius
        )
        overlap_manual = calculate_mask_overlap(mask_src_manual, mask_tgt, downsample=1)
        results["manual"] = overlap_manual

        if verbose:
            print(f"        ({time.time()-t4:.2f}s)")

    dt_overlap = time.time() - t_overlap_start
    if verbose:
        print(f"\n    Centroid Overlap Metrics (total time: {dt_overlap:.2f}s):")
        print(
            f"      Before alignment:     IoU={overlap_before['iou']:.3f}, Dice={overlap_before['dice']:.3f}, Overlap={overlap_before['overlap_percent']:.1f}%"
        )
        print(
            f"      After PCC:            IoU={overlap_pcc['iou']:.3f}, Dice={overlap_pcc['dice']:.3f}, Overlap={overlap_pcc['overlap_percent']:.1f}%"
        )
        print(
            f"      After PCC+RANSAC:     IoU={overlap_forward['iou']:.3f}, Dice={overlap_forward['dice']:.3f}, Overlap={overlap_forward['overlap_percent']:.1f}%"
        )
        if "manual" in results:
            print(
                f"      Manual affine:        IoU={results['manual']['iou']:.3f}, Dice={results['manual']['dice']:.3f}, Overlap={results['manual']['overlap_percent']:.1f}%"
            )

        # Check if final overlap is too low
        print_overlap_quality_warning(overlap_forward['overlap_percent'], indent="    ")

    # Save to CSV if path provided
    if output_csv_path:
        import pandas as pd

        rows = []
        for stage_name, stage_metrics in results.items():
            rows.append(
                {
                    "stage": stage_name,
                    "iou": stage_metrics["iou"],
                    "dice": stage_metrics["dice"],
                    "overlap_percent": stage_metrics["overlap_percent"],
                }
            )
        df = pd.DataFrame(rows)
        df.to_csv(output_csv_path, index=False)
        if verbose:
            print(f"    Saved overlap metrics to: {output_csv_path.name}")

    return results


def compare_affines(manual_yaml: Path, auto_yaml: Path) -> dict:
    """Compare manual vs automatic affine transforms.

    Note: Both biahub and automatic registration save inverse affines (target->source).
    We compare them directly without any inversion.
    """
    with open(manual_yaml, "r") as f:
        manual_data = yaml.safe_load(f)
    with open(auto_yaml, "r") as f:
        auto_data = yaml.safe_load(f)

    manual_affine = np.array(manual_data["affine_transform_zyx"])
    auto_affine = np.array(auto_data["affine_transform_zyx"])

    # Extract 2D components (YX)
    manual_2d = manual_affine[1:3, 1:3]
    auto_2d = auto_affine[1:3, 1:3]
    manual_trans = manual_affine[1:3, 3]
    auto_trans = auto_affine[1:3, 3]

    trans_diff = np.linalg.norm(manual_trans - auto_trans)

    # Print actual translation values for debugging
    print(f"    Manual translation: Y={manual_trans[0]:.2f}, X={manual_trans[1]:.2f}")
    print(f"    Auto translation:   Y={auto_trans[0]:.2f}, X={auto_trans[1]:.2f}")
    print(
        f"    Difference:         dY={manual_trans[0]-auto_trans[0]:.2f}, dX={manual_trans[1]-auto_trans[1]:.2f}"
    )

    manual_rot = np.arctan2(manual_2d[1, 0], manual_2d[0, 0])
    auto_rot = np.arctan2(auto_2d[1, 0], auto_2d[0, 0])
    rot_diff_deg = np.abs(np.degrees(manual_rot - auto_rot))

    manual_scale = np.sqrt(np.linalg.det(manual_2d))
    auto_scale = np.sqrt(np.linalg.det(auto_2d))
    scale_diff_pct = 100 * np.abs(manual_scale - auto_scale) / manual_scale

    return {
        "translation_diff_pixels": float(trans_diff),
        "rotation_diff_degrees": float(rot_diff_deg),
        "scale_diff_percent": float(scale_diff_pct),
        "manual_determinant": float(np.linalg.det(manual_affine)),
        "auto_determinant": float(np.linalg.det(auto_affine)),
    }


def extract_spots_from_intensity_subsampled(
    zarr_path: Path,
    position: str,
    t_idx: int = 0,
    channel_indices: list = None,
    threshold: float = 400,
    min_distance: int = 3,
    bins_to_select: int = 100,
    grid_size: int = 50,
    max_spots_per_bin: int = None,
    cache_subdir: str = "in_situ_sequencing/register/iss_spot_cache",
) -> np.ndarray:
    """
    Extract spot centroids from SUBSAMPLED spatial grid bins (fast).

    Only processes selected grid bins from inner 60% of well, dramatically
    reducing computation time for registration purposes.

    Parameters
    ----------
    zarr_path : Path
        Path to zarr store containing intensity images.
    position : str
        Position string (e.g., "A/1/0").
    t_idx : int
        Time index (for ISS: round index).
    channel_indices : list
        Channels to sum (e.g., [1, 2, 3, 4] for ISS spot channels).
    threshold : float
        Minimum intensity for spot detection.
    min_distance : int
        Minimum distance between detected spots (pixels).
    bins_to_select : int
        Number of spatial grid bins to process.
    grid_size : int
        Grid size (total bins = grid_size²).
    max_spots_per_bin : int
        Maximum spots to keep per bin (keeps brightest, default: None = all spots).
    cache_subdir : str
        Cache subdirectory relative to experiment root (default: "1-preprocess/register/iss_spot_cache").

    Returns
    -------
    np.ndarray
        (N, 2) array of spot centroids in (y, x) format from sampled regions.
    """
    # Check cache first
    cache_path = _get_spot_cache_path(
        zarr_path, position, t_idx, channel_indices, threshold, 1.0,
        bins_to_select, grid_size, max_spots_per_bin, cache_subdir
    )
    if cache_path.exists():
        try:
            return np.load(cache_path)
        except (EOFError, ValueError, OSError):
            pass  # corrupt or partial write from concurrent job — recompute and overwrite

    # Load and sum intensity channels with parallel chunk reads
    import zarr as _zarr_spots
    from concurrent.futures import ThreadPoolExecutor as _ChPool

    arr = _zarr_spots.open(str(zarr_path / position / "0"), mode="r")
    if channel_indices is None:
        if arr.ndim >= 4:
            channel_indices = list(range(arr.shape[1]))
        elif arr.ndim == 3:
            channel_indices = list(range(arr.shape[0]))

    def _load_channel(ch):
        """Load one channel with parallel chunk reads (releases GIL)."""
        if arr.ndim == 5:
            return _load_seg_2d(zarr_path, position, t_idx * arr.shape[1] + ch).astype(np.float32) if False else arr[t_idx, ch, 0, :, :].astype(np.float32)
        elif arr.ndim == 4:
            return arr[t_idx, ch, :, :].astype(np.float32)
        elif arr.ndim == 3:
            return arr[ch, :, :].astype(np.float32)
        return arr[:, :].astype(np.float32)

    # Load all channels concurrently (zarr chunk decompression releases GIL)
    with _ChPool(max_workers=min(len(channel_indices), 32)) as chpool:
        channels = list(chpool.map(_load_channel, channel_indices))
    intensity = channels[0]
    for ch_data in channels[1:]:
        intensity = intensity + ch_data
    del channels

    Y, X = intensity.shape

    # Create spatial grid bins
    y_bins = np.linspace(0, Y, grid_size + 1)
    x_bins = np.linspace(0, X, grid_size + 1)

    # Select from inner 60% region (20% margin on each side)
    edge_margin = int(grid_size * 0.20)
    inner_i_min = edge_margin
    inner_i_max = grid_size - edge_margin
    inner_j_min = edge_margin
    inner_j_max = grid_size - edge_margin

    # Get all inner bins
    inner_bins = [
        (i, j)
        for i in range(inner_i_min, inner_i_max)
        for j in range(inner_j_min, inner_j_max)
    ]

    # Shuffle for random spatial sampling (reproducible)
    np.random.seed(42)
    np.random.shuffle(inner_bins)

    # Select bins
    n_bins_to_select = min(bins_to_select, len(inner_bins))
    selected_bins = inner_bins[:n_bins_to_select]

    # Extract spots from each selected bin — parallel across bins (releases GIL in C code)
    from skimage.feature import peak_local_max

    def _detect_spots_in_bin(bin_ij):
        i, j = bin_ij
        y_start = int(y_bins[i])
        y_end = int(y_bins[i + 1])
        x_start = int(x_bins[j])
        x_end = int(x_bins[j + 1])

        region = intensity[y_start:y_end, x_start:x_end]
        region_peaks = peak_local_max(
            region, min_distance=min_distance,
            threshold_abs=threshold, exclude_border=True,
        )

        if len(region_peaks) > 0:
            if max_spots_per_bin is not None and len(region_peaks) > max_spots_per_bin:
                peak_intensities = region[region_peaks[:, 0], region_peaks[:, 1]]
                top_indices = np.argsort(peak_intensities)[::-1][:max_spots_per_bin]
                region_peaks = region_peaks[top_indices]
            # Convert to global coordinates
            region_peaks = region_peaks + np.array([y_start, x_start])
        return region_peaks

    # Spot detection is GIL-bound (peak_local_max) — sequential is nearly as fast as parallel
    bin_results = [_detect_spots_in_bin(b) for b in selected_bins]

    all_peaks = []
    for region_peaks in bin_results:
        if len(region_peaks) > 0:
            all_peaks.append(region_peaks)

    peaks = np.vstack(all_peaks) if all_peaks else np.empty((0, 2))

    # Save to cache via per-PID temp file + atomic rename to avoid concurrent write races
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(f'.npy.{os.getpid()}.tmp')
    np.save(tmp_path, peaks)
    try:
        tmp_path.rename(cache_path)
    except OSError:
        tmp_path.unlink(missing_ok=True)  # another job won the race; data is identical

    return peaks


def extract_spots_from_intensity(
    zarr_path: Path,
    position: str,
    t_idx: int = 0,
    channel_indices: list = None,
    threshold: float = 400,
    min_distance: int = 3,
    min_area: int = 100,  # Reuse existing param for consistency
    center_fraction: float = 1.0,  # Reuse existing param
    spatial_bins_to_select: int = None,  # Added missing parameter
    spatial_grid_size: int = 100,  # Added missing parameter
) -> np.ndarray:
    """
    Extract spot centroids from summed intensity image.

    Similar to extract_centroids_from_segmentation, but works on continuous
    intensity images (e.g., ISS spots). Detects local maxima above threshold.

    Parameters
    ----------
    zarr_path : Path
        Path to zarr store containing intensity images.
    position : str
        Position string (e.g., "A/1/0").
    t_idx : int
        Time index (for ISS: round index).
    channel_indices : list
        Channels to sum (e.g., [1, 2, 3, 4] for ISS spot channels).
        If None, uses all channels.
    threshold : float
        Minimum intensity for spot detection.
    min_distance : int
        Minimum distance between detected spots (pixels).
    min_area : int
        Minimum spot area (unused, kept for param compatibility).
    center_fraction : float
        Fraction of image to use from center (0-1, default: 1.0 = full image).
    spatial_bins_to_select : int
        If specified, only extract spots from N randomly selected spatial bins
        (for speed). If None, extract from full image.
    spatial_grid_size : int
        Grid size for spatial subsampling (only used if spatial_bins_to_select is set).

    Returns
    -------
    np.ndarray
        (N, 2) array of spot centroids in (y, x) format.

    Notes
    -----
    - Results are cached using the same strategy as centroid extraction
    - Cache key includes channel_indices, threshold, and spatial_bins_to_select
    """
    # Check cache first
    cache_path = _get_spot_cache_path(
        zarr_path, position, t_idx, channel_indices, threshold, center_fraction,
        spatial_bins_to_select, spatial_grid_size
    )
    if cache_path.exists():
        try:
            return np.load(cache_path)
        except (EOFError, ValueError, OSError):
            pass  # corrupt or partial write from concurrent job — recompute and overwrite

    # Load intensity data (direct zarr read)
    data = zarr.open(str(zarr_path / position / "0"), mode="r")

    # Handle different data shapes
    if data.ndim == 5:  # (T, C, Z, Y, X)
        if channel_indices is None:
            channel_indices = list(range(data.shape[1]))
        intensity = np.sum([data[t_idx, ch, 0, :, :] for ch in channel_indices], axis=0)
    elif data.ndim == 4:  # (T, C, Y, X) - no Z
        if channel_indices is None:
            channel_indices = list(range(data.shape[1]))
        intensity = np.sum([data[t_idx, ch, :, :] for ch in channel_indices], axis=0)
    elif data.ndim == 3:  # (C, Y, X) - single timepoint
        if channel_indices is None:
            channel_indices = list(range(data.shape[0]))
        intensity = np.sum([data[ch, :, :] for ch in channel_indices], axis=0)
    else:  # (Y, X) - single channel
        intensity = data[:, :]
    intensity = np.asarray(intensity, dtype=np.float32)

    # Apply center fraction crop if specified
    if center_fraction < 1.0:
        Y, X = intensity.shape
        crop_h = int(Y * center_fraction)
        crop_w = int(X * center_fraction)
        start_h = (Y - crop_h) // 2
        start_w = (X - crop_w) // 2
        intensity_crop = intensity[start_h : start_h + crop_h, start_w : start_w + crop_w]
    else:
        intensity_crop = intensity
        start_h, start_w = 0, 0

    # Detect local maxima
    from skimage.feature import peak_local_max

    # Find peaks above threshold
    peaks = peak_local_max(
        intensity_crop,
        min_distance=min_distance,
        threshold_abs=threshold,
        exclude_border=True,
    )

    # Adjust coordinates if we cropped
    if center_fraction < 1.0:
        peaks[:, 0] += start_h
        peaks[:, 1] += start_w

    # Save to cache via per-PID temp file + atomic rename to avoid concurrent write races
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(f'.npy.{os.getpid()}.tmp')
    np.save(tmp_path, peaks)
    try:
        tmp_path.rename(cache_path)
    except OSError:
        tmp_path.unlink(missing_ok=True)  # another job won the race; data is identical

    return peaks


def _get_spot_cache_path(
    zarr_path: Path,
    position: str,
    t_idx: int,
    channel_indices: list,
    threshold: float,
    center_fraction: float,
    bins_to_select: int = None,
    grid_size: int = None,
    max_spots_per_bin: int = None,
    cache_subdir: str = "in_situ_sequencing/register/iss_spot_cache",
) -> Path:
    """Generate cache file path for extracted spot centroids."""
    # Create a unique hash from input parameters
    ch_str = "_".join(map(str, channel_indices)) if channel_indices else "all"

    if bins_to_select is not None:
        max_spots_str = f"_{max_spots_per_bin}" if max_spots_per_bin is not None else ""
        cache_key = f"{zarr_path}_{position}_{t_idx}_{ch_str}_{threshold}_{center_fraction}_{bins_to_select}_{grid_size}{max_spots_str}"
        subsample_str = f"_bins{bins_to_select}"
    else:
        cache_key = f"{zarr_path}_{position}_{t_idx}_{ch_str}_{threshold}_{center_fraction}"
        subsample_str = ""

    cache_hash = _create_cache_hash(cache_key)

    # Navigate from zarr path to 1-preprocess directory
    # zarr_path structure: experiment_root/1-preprocess/modality/stitch/name.zarr
    preprocess_dir = zarr_path.parents[2]  # Go up from modality/stitch/name.zarr to 1-preprocess
    cache_dir = preprocess_dir / cache_subdir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create human-readable prefix
    modality = zarr_path.stem.replace("_stitched", "").replace(".zarr", "")
    pos_str = _format_position(position)
    cf_str = f"cf{center_fraction:.1f}".replace(".", "p")
    thr_str = f"thr{int(threshold)}"
    human_prefix = f"{modality}_{pos_str}_t{t_idx}_{thr_str}_{cf_str}{subsample_str}"

    return cache_dir / f"{human_prefix}_{cache_hash}.npy"

