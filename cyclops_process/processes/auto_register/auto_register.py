"""
Automatic registration using centroid matching + RANSAC.

Algorithm overview:
------------------
1. PCC pre-alignment: Coarse translation using phase cross-correlation on binary masks
2. Centroid extraction: Extract cell centroids from source and target segmentations
3. KDTree matching: Fast nearest-neighbor matching with distance filtering
4. Hu moments graph matching: Shape-based filtering using graph consistency (optional)
5. RANSAC: Robust affine estimation from matched point pairs
6. Compose transforms: Final = RANSAC ∘ PCC
7. Validation overlays: Generate comparison images

Performance optimizations:
--------------------------
- Adaptive PCC downsampling: 8x-64x based on image size (>95% speedup)
- KDTree matching: 10-20x faster than Hungarian, better inlier ratios
- Graph-based Hu moment filtering: Shape consistency validation
- Validation warnings for poor RANSAC quality
"""

import numpy as np
import time
import json
import hashlib
import yaml
from pathlib import Path
from typing import Literal

from iohub import open_ome_zarr
from scipy.spatial.distance import cdist

import sys
import os

sys.path.insert(0, os.getcwd())


from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.data.filesystem import parse_well

# Import from modular components
from cyclops_process.processes.auto_register.auto_register_utils import (
    extract_centroids_from_segmentation,
    resolve_seg_array_path,
    affine_3x3_to_4x4_zyx,
    save_affine_to_yaml,
    compare_affines,
    calculate_mask_overlap,
    compute_registration_overlap_metrics,
    compute_binary_mask_overlap_metrics,
    spatial_grid_subsample,
    _get_centroid_cache_path,
    compute_hu_moments_for_cells,
    match_cells_by_hu_moments,
    match_cells_by_hu_moments_knn,
)
from cyclops_process.processes.auto_register.auto_register_pcc import (
    estimate_translation_pcc,
)


from cyclops_process.processes.auto_register.auto_register_ransac import (
    kdtree_matching,
    estimate_affine_ransac,
)
from cyclops_process.processes.auto_register.auto_register_visualization import (
    create_validation_overlays,
    create_pcc_overlays,
    create_final_alignment_overlays,
    save_centroid_overlay,
    save_spatial_sampling_grid,
    load_mask_2d,
    visualize_hu_moment_matches,
)

def resolve_registration_paths(
    dataset,
    experiment: str,
    well,
    registration_type: str,
    skip_track: bool = False,
) -> dict:
    """Resolve seg paths and time indices for a registration type.

    Centralises the ops_num / well routing logic so it isn't duplicated
    across ``auto_register_iss_to_track``, ``auto_register_pheno_to_track``,
    ``check_yaml_registration``, and the CLI overlap commands.

    Parameters
    ----------
    dataset : OpsDataset
        Initialised dataset object.
    experiment : str
        Experiment name (used to derive ops number).
    well : int
        Well number (1, 2, or 3).
    registration_type : str
        ``"iss"`` or ``"pheno"``.
    skip_track : bool
        If True, skip the tracking store and register directly between
        ISS ↔ Pheno.

    Returns
    -------
    dict
        ``source_seg_path``, ``target_seg_path``, ``t_idx_source``,
        ``t_idx_target``, ``position``.
    """
    ops_num = int(experiment.split("_")[0].replace("ops", ""))
    row, col = parse_well(well)
    position = f"{row}/{col}/0"

    if registration_type == "iss":
        source_seg_path = dataset.store_paths["iss_segmentation"]
        t_idx_source = 0

        if skip_track:
            target_seg_path = dataset.store_paths["pheno_assembled_v3"]
            t_idx_target = 0
        elif ops_num < 69 or col in [1, 2]:
            target_seg_path = dataset.store_paths["lc_5x_segmentation"]
            if ops_num < 69:
                # Clamp t_idx to valid range — well 3 may have fewer timepoints
                # (e.g., T=3 for W3 vs T=4+ for W1/W2)
                from iohub import open_ome_zarr as _open
                _store = _open(target_seg_path, mode="r")
                n_t = _store[position].data.shape[0]
                t_idx_target = min(3, n_t - 1)
            else:
                t_idx_target = 1
        else:  # ops >= 69, well 3
            target_seg_path = dataset.store_paths["pheno_assembled_v3"]
            t_idx_target = 0

    else:  # pheno
        t_idx_source = 0

        if skip_track:
            source_seg_path = dataset.store_paths["pheno_assembled_v3"]
            target_seg_path = dataset.store_paths["iss_segmentation"]
            t_idx_target = 0
        elif ops_num < 69 or col in [1, 2]:
            source_seg_path = dataset.store_paths["pheno_assembled_v3"]
            target_seg_path = dataset.store_paths["lc_5x_segmentation"]
            if ops_num < 69:
                # Clamp t_idx to valid range — well 3 may have fewer timepoints
                from iohub import open_ome_zarr as _open
                _store = _open(target_seg_path, mode="r")
                n_t = _store[position].data.shape[0]
                t_idx_target = min(col, n_t - 1)
            else:
                t_idx_target = col - 1
        else:  # ops >= 69, well 3 — pheno register is ISS→Pheno
            source_seg_path = dataset.store_paths["iss_segmentation"]
            target_seg_path = dataset.store_paths["pheno_assembled_v3"]
            t_idx_target = 0

    return {
        "source_seg_path": source_seg_path,
        "target_seg_path": target_seg_path,
        "t_idx_source": t_idx_source,
        "t_idx_target": t_idx_target,
        "position": position,
    }


# Default parameters optimized for cell segmentation matching
DEFAULT_PARAMS = {
    # RANSAC
    "min_samples": 3,
    "residual_threshold": 8.0,  # Increased from 5.0 to handle centroid jitter between modalities
    "max_trials": 50000,  # Increased from 10000 to handle low inlier ratios (~10%)
    "stop_probability": 0.99,
    "transform_type": "similarity",  # "affine", "similarity" (isotropic scale), or "euclidean" (no scale)
    # Filtering
    "min_cell_area": 100,
    # Spatial subsampling (takes all cells from randomly selected bins across well)
    "spatial_grid_size": 50,  # Grid size (NxN bins total), larger = finer spatial resolution
    "spatial_bins_to_select": 100,  # Number of bins to randomly select (None = no subsampling)
    # PCC pre-alignment (adaptive downsampling based on image size)
    "skip_pcc": False,  # Enable PCC for initial coarse alignment
    "pcc_downsample_factor": None,  # None = auto-select based on image size
    "pcc_center_fraction": 0.5,  # Use center 30% of image for PCC (overridden to 1.0 for ISS-to-Track)
    # Hu moments filtering (shape-based matching for RANSAC input)
    "use_hu_moments": True,  # Enable graph-based Hu matching before RANSAC
    # Graph-based neighborhood matching parameters
    "max_match_distance": 2000.0,  # Maximum spatial search distance (pixels) - all cells in this radius evaluated
    "graph_top_k_candidates": 500,  # Top N candidates by Hu similarity to pass to graph filtering (use all spatial candidates)
    "graph_k_neighbors": 8,  # Neighborhood size for graph construction
    # Graph scoring weights (5 components total)
    # Note: Weights tuned based on variance in top matches - high variance = high discrimination
    "graph_hu_weight": 0.1,  # Weight for individual cell shape (already pre-filtered by Hu)
    "graph_neighbor_hu_weight": 0.15,  # Weight for neighbor shape consistency (low variance in top matches)
    "graph_edge_length_weight": 0.1,  # Weight for edge length consistency (medium variance)
    "graph_angular_spacing_weight": 0.95,  # Weight for angular spacing (highest variance, most discriminative)
    "graph_clustering_weight": 0.1,  # Weight for clustering coefficient (neighbor interconnectedness)
    # Top-N selection for RANSAC (best matches globally)
    "graph_max_score_threshold": 100,  # Take top N best matches for RANSAC (integer = count, not threshold)
    "graph_min_matches_per_cell": 0,  # Unused (kept for compatibility)
    "graph_min_total_matches": 10,  # Minimum total matches required (else throw warning)
    "hu_n_workers": None,  # Number of parallel workers for Hu computation (None = CPU count - 1)
}


def _get_stitched_path_for_pyramids(seg_path: Path) -> Path:
    """
    Map segmentation path to corresponding stitched store path for pyramid access.

    Segmentation stores (bc_segmentation.zarr, tracking_segmentation_stitched.zarr, etc.)
    do not have pyramids, but the stitched image stores contain nuclear_seg subdirectories
    with pyramids. Paths updated for the v3-native pipeline: the old v2 stitched stores
    (tracking_phase_2d_stitched.zarr, pheno_phase_stitched.zarr) no longer exist after the
    convert_v3 migration.
    """
    seg_path = Path(seg_path)
    seg_str = str(seg_path)

    # ISS: segmentation/bc_segmentation.zarr -> stitch/bc_stitched.zarr
    if "bc_segmentation.zarr" in seg_str:
        return seg_path.parent.parent / "stitch/bc_stitched.zarr"

    # Tracking (5x): segmentation/tracking_segmentation_stitched.zarr -> v3 stitched store
    elif "tracking_segmentation_stitched.zarr" in seg_str:
        return seg_path.parent.parent / "stitch/tracking_phase_2d_stitched_v3.zarr"

    # Phenotyping (20x): v3 assembled store lives under <exp>/3-assembly/, not stitch/.
    # seg_path = <exp>/1-preprocess/live_imaging/segmentation/phenotyping_segmentation_stitched.zarr
    elif "phenotyping_segmentation_stitched.zarr" in seg_str:
        return seg_path.parents[3] / "3-assembly/phenotyping_v3.zarr"

    # If already a stitched path or unknown, return as-is
    return seg_path


def refine_affine_by_mask_iou(
    affine_3x3: np.ndarray,
    mask_src_ds: np.ndarray,
    mask_tgt_ds: np.ndarray,
    ds_factor: int = 8,
) -> tuple[np.ndarray, dict]:
    """Refine an affine via coarse-then-fine translation grid maximizing mask IoU.

    Parameters
    ----------
    affine_3x3 : 3x3 forward affine (source→target, YX order).
    mask_src_ds : Downsampled binary source mask.
    mask_tgt_ds : Downsampled binary target mask.
    ds_factor : Downsample factor used for the masks.

    Returns
    -------
    (refined_affine_3x3, info_dict)
    """
    from scipy.ndimage import affine_transform as _aff

    base_scaled = affine_3x3.copy()
    base_scaled[:2, 2] /= ds_factor
    h, w = mask_tgt_ds.shape

    def _iou(dy, dx):
        perturb = np.array([[1, 0, dy / ds_factor], [0, 1, dx / ds_factor], [0, 0, 1]])
        composed = perturb @ base_scaled
        inv_A = np.linalg.inv(composed)
        warped = _aff(mask_src_ds, inv_A[:2, :2], offset=inv_A[:2, 2],
                       order=0, output_shape=(h, w))
        inter = (warped & mask_tgt_ds).sum()
        union = (warped | mask_tgt_ds).sum()
        return float(inter / union) if union > 0 else 0.0

    base_iou = _iou(0, 0)
    best_iou, best_dy, best_dx = base_iou, 0, 0

    # Coarse grid: ±50px, step=10 (121 evals)
    for dy in range(-50, 51, 10):
        for dx in range(-50, 51, 10):
            v = _iou(dy, dx)
            if v > best_iou:
                best_iou, best_dy, best_dx = v, dy, dx

    # Fine grid around best: ±8px, step=2 (81 evals)
    if best_iou > base_iou:
        for dy in range(best_dy - 8, best_dy + 9, 2):
            for dx in range(best_dx - 8, best_dx + 9, 2):
                v = _iou(dy, dx)
                if v > best_iou:
                    best_iou, best_dy, best_dx = v, dy, dx

    info = {"base_iou": base_iou, "best_iou": best_iou,
            "dy": best_dy, "dx": best_dx}

    if best_iou > base_iou:
        perturb = np.array([[1, 0, best_dy], [0, 1, best_dx], [0, 0, 1]])
        return perturb @ affine_3x3, info

    return affine_3x3, info


def auto_estimate_registration(
    source_seg_path: Path,
    target_seg_path: Path,
    position: str,
    output_yaml_path: Path,
    t_idx_source: int = 0,
    t_idx_target: int = 0,
    params: dict = None,
    create_overlays: bool = True,
    overlay_output_dir: Path = None,
    verbose: bool = True,
    manual_yaml_path: Path = None,
    use_cache: bool = False,
    seed_affine_path: Path = None,
) -> dict:
    """
    Automatically estimate affine registration between segmentation masks.

    Parameters
    ----------
    source_seg_path : Path
        Source segmentation zarr path.
    target_seg_path : Path
        Target segmentation zarr path.
    position : str
        Position string (e.g., "A/1/0").
    output_yaml_path : Path
        Output YAML file path for affine.
    t_idx_source : int
        Source time index.
    t_idx_target : int
        Target time index.
    params : dict
        Registration parameters (uses defaults if None).
    create_overlays : bool
        Whether to create validation overlays.
    overlay_output_dir : Path
        Output directory for overlays.
    verbose : bool
        Print progress messages.
    manual_yaml_path : Path
        Optional path to manual affine for comparison.
    use_cache : bool
        If True, load from cache when available. If False, recompute everything
        but still save to cache for future use.

    Returns
    -------
    dict
        Results including affine, metrics, and paths.
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    # print  params as pretty table
    from prettytable import PrettyTable

    table = PrettyTable()
    table.field_names = ["Param", "Value"]
    for param, value in params.items():
        table.add_row([param, value])
    print(table)

    t_start = time.time()

    # Get image dimensions for megapixel calculations (direct zarr read)
    import zarr as _zarr
    _src_arr = _zarr.open(str(resolve_seg_array_path(source_seg_path, position)), mode="r")
    seg_src_shape = _src_arr.shape
    if len(seg_src_shape) >= 2:
        img_h, img_w = seg_src_shape[-2:]
        megapixels = (img_h * img_w) / 1e6
    else:
        megapixels = None

    if verbose:
        print(f"Auto-registration: {position}")
        print(f"  Source: {source_seg_path}")
        print(f"  Target: {target_seg_path}")
        if megapixels:
            print(f"  Image size: {img_h} x {img_w} ({megapixels:.1f} MP)")

    # Step 0: PCC pre-alignment for coarse translation
    center_fraction = params.get("center_fraction", 1.0)
    skip_pcc = params.get("skip_pcc", False)

    # Seed affine: load existing affine and use its 2D transform as pre-alignment
    # This replaces PCC entirely — useful for refining an existing registration
    _seed_prealign_3x3 = None
    if seed_affine_path is not None and Path(seed_affine_path).exists():
        import yaml as _yaml
        with open(seed_affine_path) as _f:
            _seed_4x4 = np.array(_yaml.safe_load(_f)["affine_transform_zyx"])
        # The YAML stores the inverse transform (target→source for scipy).
        # For pre-aligning source centroids to target space, we need the inverse of the inverse = forward.
        # But actually the registration pipeline expects: source_aligned = source + pcc_shift
        # and the YAML affine maps target→source. So we need the inverse (source→target).
        _seed_prealign_3x3 = np.linalg.inv(np.eye(3))  # start with identity
        # Extract 2D 3x3 from 4x4: rows/cols 1-2 (Y,X) + translation
        _seed_2d = np.eye(3)
        _seed_2d[:2, :2] = _seed_4x4[1:3, 1:3]
        _seed_2d[:2, 2] = _seed_4x4[1:3, 3]
        # Invert: YAML is target→source, we want source→target for pre-alignment
        _seed_prealign_3x3 = np.linalg.inv(_seed_2d)
        if verbose:
            _t = _seed_prealign_3x3[:2, 2]
            print(f"  [0/7] Using seed affine from {Path(seed_affine_path).name}")
            print(f"    Seed translation: dy={_t[0]:.1f}, dx={_t[1]:.1f}")
        skip_pcc = True  # Don't run PCC when we have a seed

    if skip_pcc:
        if verbose and _seed_prealign_3x3 is None:
            print("  [0/7] PCC pre-alignment... SKIPPED")
        pcc_shift = np.array([0.0, 0.0])
        dt0 = 0.0
        pcc_cache_hit = False  # No cache when skipping PCC
    else:
        if verbose:
            print(
                f"  [0/7] PCC pre-alignment, using downsample factor of {params.get('pcc_downsample_factor', 32)}..."
            )

        t0 = time.time()

        # Adaptive downsampling based on image size
        # PCC only needs coarse translation, so aggressive downsampling is fine
        pcc_downsample = params.get("pcc_downsample_factor", None)
        if pcc_downsample is None:
            if megapixels is None or megapixels < 100:
                pcc_downsample = 8  # Small images
            elif megapixels < 500:
                pcc_downsample = 32  # Medium images
            else:
                pcc_downsample = 64  # Large images (>500 MP)

            if verbose:
                print(
                    f"    Auto-selected downsample factor: {pcc_downsample}x ({megapixels:.1f} MP → {megapixels/(pcc_downsample**2):.2f} MP)"
                )

        pcc_shift, pcc_cache_hit = estimate_translation_pcc(
            source_seg_path,
            target_seg_path,
            position,
            position,
            t_idx_source,
            downsample_factor=pcc_downsample,
            center_fraction=params.get("pcc_center_fraction", 1.0),
            use_cache=use_cache,
        )
        dt0 = time.time() - t0
        if verbose:
            if megapixels:
                print(
                    f"    Translation: dy={pcc_shift[0]:.1f}, dx={pcc_shift[1]:.1f} ({dt0:.2f}s, {dt0/megapixels:.3f}s/MP)"
                )
            else:
                print(
                    f"    Translation: dy={pcc_shift[0]:.1f}, dx={pcc_shift[1]:.1f} ({dt0:.2f}s)"
                )

    # Create PCC alignment preview if overlay directory is specified
    # Skip if PCC was loaded from cache and overlays already exist
    pcc_overlays_exist = False 

    if create_overlays and overlay_output_dir is not None and not skip_pcc:
        # Check if PCC overlays already exist (when using cached PCC)
        if pcc_cache_hit:
            pcc_overlay_files = [
                overlay_output_dir / "01_pcc_alignment_8x.png",
                overlay_output_dir / "02_pcc_detail_grid_pos0.png",
            ]
            pcc_overlays_exist = all(f.exists() for f in pcc_overlay_files)
            if pcc_overlays_exist and verbose:
                print(f"    PCC overlays already exist, skipping creation")

    if (
        create_overlays
        and overlay_output_dir is not None
        and ((not skip_pcc and not pcc_overlays_exist))
    ):
        t_overlay = time.time()

        # First, create timepoint comparison if this is pheno-to-track (multi-timepoint target)
        # Check if target has multiple timepoints
        _tgt_arr = _zarr.open(str(resolve_seg_array_path(target_seg_path, position)), mode="r")
        target_shape = _tgt_arr.shape
        n_target_timepoints = target_shape[0] if target_shape[0] > 1 else 1

        if n_target_timepoints > 1 and verbose:
            print(
                f"    Creating timepoint comparison grid (target has {n_target_timepoints} timepoints)..."
            )

            # Get crop regions first
            _src_shape = _zarr.open(str(resolve_seg_array_path(source_seg_path, position)), mode="r").shape
            mask_height, mask_width = _src_shape[-2:]
            from cyclops_process.processes.auto_register.auto_register_visualization import (
                compute_crop_regions,
            )

            crop_regions_for_comparison = compute_crop_regions(
                (mask_height, mask_width), crop_size=1024
            )

            from cyclops_process.processes.auto_register.auto_register_visualization import (
                create_timepoint_comparison_grid,
            )

            create_timepoint_comparison_grid(
                source_seg_path=source_seg_path,
                target_seg_path=target_seg_path,
                position=position,
                t_idx_source=t_idx_source,
                t_idx_target=t_idx_target,
                n_target_timepoints=n_target_timepoints,
                output_dir=overlay_output_dir,
                crop_regions=crop_regions_for_comparison,
                verbose=verbose,
            )

        if verbose:
            print(f"    Creating PCC alignment overlays...")

        mask_src, mask_tgt, crop_region = create_pcc_overlays(
            source_seg_path=source_seg_path,
            target_seg_path=target_seg_path,
            position=position,
            pcc_shift=pcc_shift,
            output_dir=overlay_output_dir,
            t_idx_source=t_idx_source,
            t_idx_target=t_idx_target,
            center_fraction=center_fraction,
            affine_3x3=None,
            verbose=verbose,
        )

        # Store crop region and masks for later use
        params["_debug_crop_region"] = crop_region
        params["_cached_mask_src"] = mask_src
        params["_cached_mask_tgt"] = mask_tgt

        dt_overlay_total = time.time() - t_overlay
        if verbose:
            print(f"    PCC comparison images saved ({dt_overlay_total:.2f}s total)")

    # If PCC overlays were skipped but we still need final overlays, compute crop regions now
    if create_overlays and "_debug_crop_region" not in params:
        # Need to determine mask shape to compute crop regions
        # Load a quick shape check from source segmentation (direct zarr)
        _src_shape_arr = _zarr.open(str(resolve_seg_array_path(source_seg_path, position)), mode="r")
        if _src_shape_arr.ndim >= 2:
            mask_shape = _src_shape_arr.shape[-2:]
        else:
            mask_shape = _src_shape_arr.shape

        # Import the helper function
        from cyclops_process.processes.auto_register.auto_register_visualization import (
            compute_crop_regions,
        )

        params["_debug_crop_region"] = compute_crop_regions(mask_shape, crop_size=1024)

    # Steps 1-2: Extract source and target centroids in parallel
    center_fraction = params.get("center_fraction", 1.0)
    if verbose:
        print("  [1-2/7] Extracting source + target centroids (parallel)...")
        if center_fraction < 1.0:
            print(f"    Using central {center_fraction*100:.0f}% of well")

    source_cache_path = _get_centroid_cache_path(
        source_seg_path, position, t_idx_source, params["min_cell_area"], center_fraction)
    target_cache_path = _get_centroid_cache_path(
        target_seg_path, position, t_idx_target, params["min_cell_area"], center_fraction)

    from concurrent.futures import ThreadPoolExecutor as _CentroidPool

    def _extract_or_cache(seg_path, pos, t_idx, cache_path):
        if use_cache and cache_path.exists():
            return np.load(cache_path), True
        centroids = extract_centroids_from_segmentation(
            seg_path, pos, t_idx, params["min_cell_area"], center_fraction)
        np.save(cache_path, centroids)
        return centroids, False

    t_centroids = time.time()
    with _CentroidPool(max_workers=2) as cpool:
        src_future = cpool.submit(_extract_or_cache, source_seg_path, position, t_idx_source, source_cache_path)
        tgt_future = cpool.submit(_extract_or_cache, target_seg_path, position, t_idx_target, target_cache_path)
        source_centroids, src_cached = src_future.result()
        target_centroids, tgt_cached = tgt_future.result()

    dt_centroids = time.time() - t_centroids
    if verbose:
        src_msg = "cached" if src_cached else "extracted"
        tgt_msg = "cached" if tgt_cached else "extracted"
        print(f"    Source: {len(source_centroids)} cells ({src_msg})")
        print(f"    Target: {len(target_centroids)} cells ({tgt_msg})")
        print(f"    Total: {dt_centroids:.2f}s (parallel)")

    # Apply pre-alignment to source centroids BEFORE subsampling
    # This ensures grid binning happens in the same coordinate space
    if _seed_prealign_3x3 is not None:
        # Full affine pre-alignment (rotation + scale + translation)
        ones = np.ones((len(source_centroids), 1))
        homogeneous = np.hstack([source_centroids, ones])  # (N, 3)
        source_centroids_aligned = (homogeneous @ _seed_prealign_3x3.T)[:, :2]
    else:
        source_centroids_aligned = source_centroids + pcc_shift  # (dy, dx)

    # Optionally subsample centroids using spatial grid
    # This preserves local cell correspondence by selecting all cells from a subset of grid regions
    bins_to_select = params.get("spatial_bins_to_select", None)
    grid_info = None  # Will store grid info if subsampling is performed

    if bins_to_select is not None:
        if verbose:
            print(f"    Subsampling centroids using spatial grid...")
            print(
                f"    Before: {len(source_centroids_aligned)} source, {len(target_centroids)} target cells"
            )

        # Use spatial grid subsampling on PCC-aligned centroids
        # Select all cells from randomly chosen bins across well
        grid_size = params.get("spatial_grid_size", 50)
        source_indices, target_indices, grid_info = spatial_grid_subsample(
            source_centroids=source_centroids_aligned,  # Use PCC-aligned source
            target_centroids=target_centroids,
            bins_to_select=bins_to_select,
            grid_size=grid_size,
        )

        # Apply subsampling
        source_centroids_aligned_subsampled = source_centroids_aligned[source_indices]
        target_centroids_subsampled = target_centroids[target_indices]

        if verbose:
            print(
                f"    After: {len(source_centroids_aligned_subsampled)} source, {len(target_centroids_subsampled)} target cells"
            )
            print(
                f"    Selected {len(grid_info['selected_bins'])} bins from {grid_info['grid_size']}x{grid_info['grid_size']} grid"
            )

        # Save spatial sampling visualization if creating overlays (skip if already exists)
        grid_viz_path = (
            overlay_output_dir / "00c_spatial_sampling_grid.png"
            if overlay_output_dir
            else None
        )
        grid_viz_exists = grid_viz_path and grid_viz_path.exists()

        if create_overlays and overlay_output_dir is not None:
            if grid_viz_exists and verbose:
                print(f"    Spatial grid visualization already exists, skipping")

        if create_overlays and overlay_output_dir is not None and not grid_viz_exists:
            if verbose:
                print(f"    Saving spatial sampling grid visualization...")
            t_grid_viz = time.time()

            # Load masks at pyramid level 4 (16x downsampled) for faster visualization
            # The grid and centroids are still accurate, just the background is lower res
            ds_factor = 16
            pyramid_level = 4  # 2^4 = 16x downsampled

            # Load directly from pyramid level (much faster than full-res + downsample!)
            # Use stitched stores which have nuclear_seg pyramids
            from cyclops_process.processes.auto_register.auto_register_visualization import (
                load_mask_2d_from_level,
            )

            source_stitch_path = _get_stitched_path_for_pyramids(source_seg_path)
            target_stitch_path = _get_stitched_path_for_pyramids(target_seg_path)

            # For now, use fallback (full-res + downsample) until pyramid loading is fixed
            # TODO: Fix pyramid level loading in load_mask_2d_from_level()
            from skimage.transform import downscale_local_mean

            mask_src_full = load_mask_2d(source_seg_path, position, t_idx_source)
            mask_tgt_full = load_mask_2d(target_seg_path, position, t_idx_target)
            mask_src_ds = downscale_local_mean(mask_src_full, (ds_factor, ds_factor))
            mask_tgt_ds = downscale_local_mean(mask_tgt_full, (ds_factor, ds_factor))
            del mask_src_full, mask_tgt_full  # Free memory immediately

            # Scale centroids and grid info to match downsampled coordinate space
            source_centroids_ds = source_centroids_aligned / ds_factor
            target_centroids_ds = target_centroids / ds_factor
            grid_info_ds = {
                "y_bins": grid_info["y_bins"] / ds_factor,
                "x_bins": grid_info["x_bins"] / ds_factor,
                "grid_size": grid_info["grid_size"],
                "selected_bins": grid_info["selected_bins"],
            }

            save_spatial_sampling_grid(
                mask_src=mask_src_ds,
                mask_tgt=mask_tgt_ds,
                source_centroids=source_centroids_ds,
                target_centroids=target_centroids_ds,
                source_indices=source_indices,
                target_indices=target_indices,
                grid_info=grid_info_ds,
                pcc_shift=pcc_shift
                / ds_factor,  # Scale PCC shift to match downsampled masks
                output_path=overlay_output_dir / "00c_spatial_sampling_grid.png",
                downsample=1,  # Already downsampled, don't downsample again
            )

            dt_grid_viz = time.time() - t_grid_viz
            if verbose:
                print(f"      Saved spatial sampling grid ({dt_grid_viz:.2f}s)")

        # Save original centroids for visualization (before overwriting with subsampled)
        source_centroids_for_viz = source_centroids.copy()
        target_centroids_for_viz = target_centroids.copy()

        # Use subsampled centroids for matching
        source_centroids_aligned = source_centroids_aligned_subsampled
        target_centroids = target_centroids_subsampled

        # Check if subsampling resulted in no overlapping cells
        if len(source_centroids_aligned) == 0 or len(target_centroids) == 0:
            error_msg = (
                f"No overlapping cells found after PCC alignment and spatial subsampling.\n"
                f"  Source cells after alignment: {len(source_centroids_aligned)}\n"
                f"  Target cells: {len(target_centroids)}\n"
                f"  PCC translation: dy={pcc_shift[0]:.1f}, dx={pcc_shift[1]:.1f}\n"
                f"  This likely indicates poor PCC alignment or non-overlapping regions."
            )
            raise ValueError(error_msg)
    else:
        # No subsampling - use original centroids for visualization
        source_centroids_for_viz = source_centroids.copy()
        target_centroids_for_viz = target_centroids.copy()

    # Save centroid overlays if creating overlays (skip if already exist)
    centroid_viz_paths = [
        overlay_output_dir / "00a_source_centroids.png" if overlay_output_dir else None,
        overlay_output_dir / "00b_target_centroids.png" if overlay_output_dir else None,
    ]
    centroids_exist = all(p and p.exists() for p in centroid_viz_paths if p)

    if create_overlays and overlay_output_dir is not None:
        if centroids_exist and verbose:
            print(f"    Centroid overlays already exist, skipping")

    if create_overlays and overlay_output_dir is not None and not centroids_exist:
        if verbose:
            print(f"    Saving centroid overlays...")
        t_centroid_viz = time.time()

        # Load masks for centroid visualization at full resolution
        # Always load fresh to ensure we have full resolution (not downsampled cache)
        mask_src = load_mask_2d(source_seg_path, position, t_idx_source)
        mask_tgt = load_mask_2d(target_seg_path, position, t_idx_target)

        # Create small high-resolution crops to verify centroid placement on individual cells
        # Use same sampling strategy as detail crops: pick from center-fraction region
        crop_size = 256  # Small enough to see individual cells clearly
        Y, X = mask_src.shape

        # If using center_fraction, sample from that region (like detail crops)
        # Otherwise use geometric center with slight offset for variety
        if center_fraction < 1.0:
            # Sample from within the center_fraction region
            y_center_region = Y / 2
            x_center_region = X / 2
            offset_frac = 0.10  # Same as detail crops
            y_offset = int(Y * center_fraction * offset_frac)
            x_offset = int(X * center_fraction * offset_frac)
            y_sample = int(y_center_region + y_offset)
            x_sample = int(x_center_region + x_offset)
        else:
            # For full well, offset slightly from center
            y_sample = int(Y / 2 + Y * 0.1)
            x_sample = int(X / 2 + X * 0.1)

        y_start = max(0, y_sample - crop_size // 2)
        y_end = min(Y, y_start + crop_size)
        x_start = max(0, x_sample - crop_size // 2)
        x_end = min(X, x_start + crop_size)

        # Crop masks
        mask_src_crop = mask_src[y_start:y_end, x_start:x_end]
        mask_tgt_crop = mask_tgt[y_start:y_end, x_start:x_end]

        # Filter centroids to those within the crop and adjust coordinates
        # Use the saved original centroids (not subsampled) for visualization
        src_in_crop = (
            (source_centroids_for_viz[:, 0] >= y_start)
            & (source_centroids_for_viz[:, 0] < y_end)
            & (source_centroids_for_viz[:, 1] >= x_start)
            & (source_centroids_for_viz[:, 1] < x_end)
        )
        tgt_in_crop = (
            (target_centroids_for_viz[:, 0] >= y_start)
            & (target_centroids_for_viz[:, 0] < y_end)
            & (target_centroids_for_viz[:, 1] >= x_start)
            & (target_centroids_for_viz[:, 1] < x_end)
        )

        source_centroids_viz = source_centroids_for_viz[src_in_crop] - np.array(
            [y_start, x_start]
        )
        target_centroids_viz = target_centroids_for_viz[tgt_in_crop] - np.array(
            [y_start, x_start]
        )

        # Save at full resolution (downsample=1) to verify centroid accuracy
        save_centroid_overlay(
            mask_src_crop,
            source_centroids_viz,
            overlay_output_dir / "00a_source_centroids.png",
            title=f"Source Centroids (256×256 @ 1x)",
            color="green",
            downsample=1,
        )
        save_centroid_overlay(
            mask_tgt_crop,
            target_centroids_viz,
            overlay_output_dir / "00b_target_centroids.png",
            title=f"Target Centroids (256×256 @ 1x)",
            color="red",
            downsample=1,
        )

        dt_centroid_viz = time.time() - t_centroid_viz
        if verbose:
            print(f"      Saved centroid overlays ({dt_centroid_viz:.2f}s)")

    # Note: PCC translation already applied to source_centroids_aligned above
    # source_centroids_aligned will be used for KDTree matching and RANSAC affine estimation

    # Step 3: KDTree matching - fast nearest-neighbor matching
    if verbose:
        print("  [3/7] KDTree nearest-neighbor matching...")

    t4 = time.time()
    source_idx, target_idx = kdtree_matching(
        source_centroids_aligned,
        target_centroids,
        params["max_match_distance"],
    )
    dt4 = time.time() - t4

    # CRITICAL FIX: Use aligned centroids for RANSAC (matched with aligned points)
    matched_source = source_centroids_aligned[source_idx]
    matched_target = target_centroids[target_idx]

    # Compute residuals before RANSAC (after PCC alignment) for QA
    residuals_pre_ransac = np.linalg.norm(matched_source - matched_target, axis=1)
    median_res_pre = np.median(residuals_pre_ransac)
    p95_res_pre = np.percentile(residuals_pre_ransac, 95)

    if verbose:
        print(
            f"    Found {len(matched_source)} matches ({dt4:.2f}s, {dt4/megapixels:.3f}s/MP)"
        )
        print(
            f"    Pre-RANSAC residuals (after PCC): median={median_res_pre:.2f}px, 95th%={p95_res_pre:.2f}px"
        )

    # Optional: Hu moments filtering for higher quality RANSAC input
    use_hu_moments = params.get("use_hu_moments", False)
    hu_filtered_matches = None

    if use_hu_moments:
        if verbose:
            print(f"  [4.5/7] Hu moments shape filtering (k-NN + top percentile)...")

        t_hu = time.time()

        # Use ALL subsampled centroids (not just KDTree matches) for wider search
        source_centroids_for_hu = source_centroids_aligned
        target_centroids_for_hu = target_centroids

        # Compute Hu moments for ALL subsampled cells (parallelized with tqdm)
        # Use cache directory for regionprops and Hu moments
        from cyclops_utils.data.experiment import OpsDataset

        dataset = OpsDataset(
            experiment=source_seg_path.parent.parent.parent.parent.name
        )
        hu_cache_dir = dataset.tracking / "cache" / "hu_moments"

        if verbose:
            print(f"      Hu moments cache directory: {hu_cache_dir}")

        n_workers = params.get("hu_n_workers", None)

        # Compute Hu moments for source and target concurrently
        # Each call loads a mask + runs regionprops + KDTree match + moments_hu
        from concurrent.futures import ThreadPoolExecutor as _HuPool

        def _compute_hu(seg_path, centroids, t_idx):
            return compute_hu_moments_for_cells(
                seg_path, position, centroids, t_idx, params["min_cell_area"],
                n_workers=n_workers, verbose=verbose, cache_dir=hu_cache_dir)

        with _HuPool(max_workers=2) as hu_pool:
            src_hu_future = hu_pool.submit(_compute_hu, source_seg_path, source_centroids_for_hu, t_idx_source)
            tgt_hu_future = hu_pool.submit(_compute_hu, target_seg_path, target_centroids_for_hu, t_idx_target)
            source_hu_moments, source_labels, _ = src_hu_future.result()
            target_hu_moments, target_labels, _ = tgt_hu_future.result()

        # Graph-based neighborhood matching
        from cyclops_process.processes.auto_register.auto_register_graph import match_cells_by_graph_consistency

        weights = {
            "hu": params.get("graph_hu_weight", 0.1),
            "neighbor_hu": params.get("graph_neighbor_hu_weight", 0.5),
            "edge_length": params.get("graph_edge_length_weight", 0.2),
            "angular_spacing": params.get("graph_angular_spacing_weight", 0.1),
            "clustering": params.get("graph_clustering_weight", 0.1),
        }

        hu_source_idx, hu_target_idx, hu_distances, source_graphs, target_graphs = match_cells_by_graph_consistency(
            source_centroids_for_hu,
            target_centroids_for_hu,
            source_hu_moments,
            target_hu_moments,
            search_radius=params.get("max_match_distance", 600.0),
            k_neighbors=params.get("graph_k_neighbors", 8),
            top_k_candidates=params.get("graph_top_k_candidates", 200),
            weights=weights,
            max_score_threshold=params.get("graph_max_score_threshold", 0.1),
            min_matches_per_cell=params.get("graph_min_matches_per_cell", 0),
            min_total_matches=params.get("graph_min_total_matches", 10),
            cache_dir=hu_cache_dir,
            verbose=verbose,
        )

        dt_hu = time.time() - t_hu

        if len(hu_source_idx) > 0:
            # Store for visualization (show all candidates considered)
            hu_filtered_matches = {
                "all_source": source_centroids_for_hu,
                "all_target": target_centroids_for_hu,
                "hu_source_idx": hu_source_idx,
                "hu_target_idx": hu_target_idx,
                "hu_distances": hu_distances,
                "source_graphs": source_graphs,
                "target_graphs": target_graphs,
            }

            # REPLACE matched centroids with Hu-filtered ones
            matched_source = source_centroids_for_hu[hu_source_idx]
            matched_target = target_centroids_for_hu[hu_target_idx]

            if verbose:
                print(
                    f"    Hu filtering: {len(matched_source)} shape-based matches selected ({dt_hu:.2f}s)"
                )
                print(f"      (Replaced {len(source_idx)} distance-based matches)")
        else:
            if verbose:
                print(
                    f"    WARNING: Hu filtering found no matches, falling back to distance-based matches"
                )

    if verbose:
        print(f"  [5/7] Running RANSAC affine estimation...")

    skip_ransac = params.get("skip_ransac", False)
    min_matches_required = params["min_samples"]
    if skip_ransac:
        if verbose:
            print(f"    skip_ransac=True — using PCC-only alignment (no RANSAC refinement)")
        ransac_affine_3x3 = np.eye(3)
        inliers = np.array([], dtype=bool)
        metrics = {
            "n_matches": len(matched_source),
            "n_inliers": 0,
            "inlier_ratio": 0.0,
            "residual_mean": 0.0,
            "residual_std": 0.0,
            "residual_max": 0.0,
        }
        dt5b = 0.0
    elif len(matched_source) < min_matches_required:
        if verbose:
            print(
                f"    WARNING: Insufficient matches ({len(matched_source)} < {min_matches_required})"
            )
            print(f"    Falling back to PCC-only alignment (no RANSAC refinement)")

        # Use identity for RANSAC (no refinement)
        ransac_affine_3x3 = np.eye(3)
        inliers = np.array([], dtype=bool)
        metrics = {
            "n_matches": len(matched_source),
            "n_inliers": 0,
            "inlier_ratio": 0.0,
            "residual_mean": 0.0,
            "residual_std": 0.0,
            "residual_max": 0.0,
        }
        dt5b = 0.0
    else:
        # Step 5: RANSAC affine estimation — run multiple times, keep best
        # RANSAC is near-instant (<0.02s), so multiple runs cost nothing
        # but stabilize the inlier count against random sampling variance
        t5b = time.time()
        n_ransac_runs = params.get("n_ransac_runs", 5)
        best_affine, best_inliers, best_metrics = None, None, {"n_inliers": -1}
        for _ in range(n_ransac_runs):
            r_affine, r_inliers, r_metrics = estimate_affine_ransac(
                matched_source,
                matched_target,
                params["min_samples"],
                params["residual_threshold"],
                params["max_trials"],
                params["stop_probability"],
                params.get("transform_type", "similarity"),
            )
            if r_metrics["n_inliers"] > best_metrics["n_inliers"]:
                best_affine, best_inliers, best_metrics = r_affine, r_inliers, r_metrics
        ransac_affine_3x3, inliers, metrics = best_affine, best_inliers, best_metrics
        dt5b = time.time() - t5b

    if verbose:
        if metrics["n_matches"] > 0:
            print(
                f"    RANSAC: {metrics['n_inliers']}/{metrics['n_matches']} inliers ({dt5b:.2f}s, {dt5b/megapixels:.3f}s/MP)"
            )
            print(f"    Inlier ratio: {metrics['inlier_ratio']:.2%}")
            print(
                f"    Residual: {metrics['residual_mean']:.2f} ± {metrics['residual_std']:.2f} px"
            )
        else:
            print(f"    No RANSAC refinement (insufficient matches)")

    # Compose pre-alignment and RANSAC transforms
    # RANSAC was fit to pre-aligned points, so it refines the initial alignment
    # Final transform: T_final = T_ransac ∘ T_prealign (apply pre-align first, then RANSAC)

    # Compose pre-alignment with RANSAC refinement
    if _seed_prealign_3x3 is not None:
        prealign_affine_3x3 = _seed_prealign_3x3
    else:
        prealign_affine_3x3 = np.eye(3)
        prealign_affine_3x3[:2, 2] = pcc_shift  # [dy, dx]

    affine_3x3 = ransac_affine_3x3 @ prealign_affine_3x3

    if verbose:
        print(f"    Composed transforms:")
        if _seed_prealign_3x3 is not None:
            _st = _seed_prealign_3x3[:2, 2]
            print(f"      Seed affine: dy={_st[0]:.1f}, dx={_st[1]:.1f} (+ rotation/scale)")
        else:
            print(f"      PCC shift: dy={pcc_shift[0]:.1f}, dx={pcc_shift[1]:.1f}")

        # Extract RANSAC refinement
        ransac_trans = ransac_affine_3x3[:2, 2]
        print(
            f"      RANSAC refinement: dy={ransac_trans[0]:.1f}, dx={ransac_trans[1]:.1f}"
        )

        # Final translation
        final_trans = affine_3x3[:2, 2]
        final_rot = np.arctan2(affine_3x3[1, 0], affine_3x3[0, 0])
        final_scale_x = np.sqrt(affine_3x3[0, 0] ** 2 + affine_3x3[1, 0] ** 2)
        final_scale_y = np.sqrt(affine_3x3[0, 1] ** 2 + affine_3x3[1, 1] ** 2)
        print(
            f"      Final: dy={final_trans[0]:.1f}, dx={final_trans[1]:.1f}, "
            f"rot={np.degrees(final_rot):.2f}°, scale=({final_scale_x:.4f}, {final_scale_y:.4f})"
        )

        # Sanity check: RANSAC refinement magnitude
        ransac_magnitude = np.linalg.norm(ransac_trans)
        if ransac_magnitude > 100:
            print(
                f"    WARNING: Large RANSAC refinement ({ransac_magnitude:.1f} px) - may indicate poor matching"
            )
        if metrics["inlier_ratio"] < 0.3:
            print(
                f"    WARNING: Low inlier ratio ({metrics['inlier_ratio']:.1%}) - alignment quality may be poor"
            )

    # Step 6b: Iterative RANSAC refinement — warp source centroids with current
    # affine, re-match at tighter radius, refit RANSAC on original points.
    if metrics.get("n_inliers", 0) >= 3:
        t6b = time.time()
        refine_radius = min(params.get("max_match_distance", 25.0), metrics["residual_mean"] * 2)
        for refine_iter in range(3):
            src_h = np.column_stack([source_centroids, np.ones(len(source_centroids))])
            src_warped = (affine_3x3 @ src_h.T).T[:, :2]

            ref_src_idx, ref_tgt_idx = kdtree_matching(
                src_warped, target_centroids, max_distance=refine_radius,
            )
            if len(ref_src_idx) < params.get("min_samples", 3):
                break

            ref_affine, ref_inliers, ref_metrics = estimate_affine_ransac(
                source_centroids[ref_src_idx], target_centroids[ref_tgt_idx],
                params["min_samples"], params["residual_threshold"],
                params["max_trials"], params["stop_probability"],
                params.get("transform_type", "similarity"),
            )

            improved = ref_metrics["residual_mean"] < metrics["residual_mean"]
            if verbose:
                print(f"  [6b] Iterative refinement [{refine_iter+1}]: "
                      f"{ref_metrics['n_inliers']}/{ref_metrics['n_matches']} inliers, "
                      f"residual {ref_metrics['residual_mean']:.2f}px"
                      f"{' ✓' if improved else ' (no improvement)'}")

            if not improved:
                break

            # RANSAC was fit on original source centroids → target centroids directly,
            # so ref_affine is already the full source→target transform
            affine_3x3 = ref_affine
            metrics = ref_metrics
            refine_radius = max(ref_metrics["residual_mean"] * 1.5, 5.0)

        dt6b = time.time() - t6b
        if verbose:
            print(f"  [6b] Iterative refinement done ({dt6b:.1f}s)")

    # Step 6c: Quality gate + grid search on mask IoU
    # Compare the candidate transforms (identity, PCC-only, post-RANSAC-and-6b)
    # by mask IoU and keep whichever wins. This protects against RANSAC fitting a
    # phantom rotation/scale on already-aligned same-modality pairs (e.g. CP→CP),
    # where PCC alone is the optimum and RANSAC degrades it. Then polish the
    # winner with the existing translation grid search.
    #
    # Evaluate at 4x downsample so sub-pixel translations (e.g. PCC shift of ~14
    # native px = ~0.9 px at 16x) are still resolvable. With order=1 bilinear
    # interpolation we get smooth IoU as a function of translation, avoiding the
    # nearest-neighbor rounding tie that previously caused the gate to keep
    # identity over a real PCC shift.
    t6c = time.time()
    from scipy.ndimage import affine_transform as _aff_eval
    _ds = 4
    _mask_src = (load_mask_2d(source_seg_path, position, t_idx_source)[::_ds, ::_ds] > 0).astype(np.uint8)
    _mask_tgt = (load_mask_2d(target_seg_path, position, t_idx_target)[::_ds, ::_ds] > 0).astype(np.uint8)
    _tgt_sum = float(_mask_tgt.sum())

    def _mask_overlap_pct(a3):
        """% of target mask covered by warped source (intersection / |target|).

        More sensitive than IoU for small translations because the denominator
        is constant — only the numerator (intersection) varies with the affine.
        """
        a = a3.copy()
        a[:2, 2] = a[:2, 2] / _ds
        inv_a = np.linalg.inv(a)
        warped = _aff_eval(
            _mask_src.astype(np.float32),
            inv_a[:2, :2], offset=inv_a[:2, 2],
            order=1, output_shape=_mask_tgt.shape,
        )
        warped_bin = (warped > 0.5).astype(np.uint8)
        inter = float((warped_bin & _mask_tgt).sum())
        return (inter / _tgt_sum * 100.0) if _tgt_sum > 0 else 0.0

    candidates = {
        # Order matters: when scores tie, max() keeps the first key. Put the
        # less-trivial transforms first so a real PCC shift wins over identity.
        "ransac+6b":     affine_3x3,
        "prealign":      prealign_affine_3x3,
        "identity":      np.eye(3),
    }
    scores = {name: _mask_overlap_pct(a) for name, a in candidates.items()}
    best_name = max(scores, key=scores.get)
    affine_3x3 = candidates[best_name].copy()
    if verbose:
        print(f"  [Quality gate] Mask overlap % (ds={_ds}x, |target| denom): "
              + ", ".join(f"{n}={v:.1f}%" for n, v in scores.items())
              + f" → keeping '{best_name}'")

    # Polish the winner with translation grid search (mask IoU)
    affine_3x3, refine_info = refine_affine_by_mask_iou(affine_3x3, _mask_src, _mask_tgt, _ds)
    dt6c = time.time() - t6c
    if verbose:
        print(f"  [6c] Mask grid refinement: IoU {refine_info['base_iou']:.3f} → {refine_info['best_iou']:.3f} "
              f"(+{(refine_info['best_iou']-refine_info['base_iou'])*100:.1f}pp, "
              f"dy={refine_info['dy']} dx={refine_info['dx']}, {dt6c:.1f}s)")

    # Step 7: Save affine to YAML
    t6 = time.time()
    affine_4x4 = affine_3x3_to_4x4_zyx(affine_3x3)

    # Invert affine to match biahub convention (biahub saves inverse transforms)
    affine_4x4_inv = np.linalg.inv(affine_4x4)
    save_affine_to_yaml(affine_4x4_inv, output_yaml_path)

    dt6 = time.time() - t6
    if verbose:
        print(f"  [7/7] Saved affine to: {output_yaml_path} ({dt6:.2f}s)")

    # Compare with manual affine if provided
    comparison = None
    if manual_yaml_path is not None and manual_yaml_path.exists():
        if verbose:
            print(f"\n  Comparing with manual affine: {manual_yaml_path}")
        comparison = compare_affines(manual_yaml_path, output_yaml_path)
        if verbose:
            print(
                f"    Translation diff: {comparison['translation_diff_pixels']:.2f} pixels"
            )
            print(
                f"    Rotation diff: {comparison['rotation_diff_degrees']:.2f} degrees"
            )
            print(f"    Scale diff: {comparison['scale_diff_percent']:.2f}%")

    # Track overlap metrics across overlay creation
    overlap_metrics = None

    # Create validation overlays
    if create_overlays:
        if overlay_output_dir is None:
            overlay_output_dir = output_yaml_path.parent / "overlays"

        if verbose:
            print(f"    Creating validation overlays in: {overlay_output_dir}")

        t7 = time.time()

        # Create final alignment comparison overlays (same views as PCC)
        if "_debug_crop_region" in params:
            # Reuse cached masks from PCC step if available
            if "_cached_mask_src" in params and "_cached_mask_tgt" in params:
                if verbose:
                    print(f"    Using cached masks from PCC step")
                mask_src = params["_cached_mask_src"]
                mask_tgt = params["_cached_mask_tgt"]
            else:
                if verbose:
                    print(f"    Loading masks...")
                t_load = time.time()
                mask_src = load_mask_2d(source_seg_path, position, t_idx_source)
                mask_tgt = load_mask_2d(target_seg_path, position, t_idx_target)
                dt_load = time.time() - t_load
                if verbose:
                    print(f"      Loaded masks ({dt_load:.2f}s)")

            # Create final alignment overlays using visualization module
            # Pass pcc_shift to show post-PCC as BEFORE (reveals RANSAC contribution)
            # Pass auto_yaml_path to load and apply auto affine the same way as manual
            create_final_alignment_overlays(
                mask_src=mask_src,
                mask_tgt=mask_tgt,
                affine_3x3=None,  # Will load from YAML internally
                crop_region=params["_debug_crop_region"],
                output_dir=overlay_output_dir,
                pcc_shift=pcc_shift,
                center_fraction=center_fraction,
                manual_yaml_path=(
                    manual_yaml_path
                    if manual_yaml_path and manual_yaml_path.exists()
                    else None
                ),
                auto_yaml_path=output_yaml_path,
                verbose=verbose,
            )

            # Run overlap metrics and graph viz concurrently (all independent)
            from concurrent.futures import ThreadPoolExecutor as _OverlapPool

            def _compute_centroid_overlap():
                return compute_registration_overlap_metrics(
                    source_centroids=source_centroids_for_viz,
                    target_centroids=target_centroids_for_viz,
                    mask_shape=mask_src.shape,
                    pcc_shift=pcc_shift,
                    affine_3x3=affine_3x3,
                    centroid_radius=10,
                    manual_yaml_path=(
                        manual_yaml_path
                        if manual_yaml_path and manual_yaml_path.exists()
                        else None
                    ),
                    output_csv_path=None,
                    verbose=verbose,
                )

            def _compute_mask_overlap():
                return compute_binary_mask_overlap_metrics(
                    mask_src=mask_src,
                    mask_tgt=mask_tgt,
                    pcc_shift=pcc_shift,
                    affine_3x3=affine_3x3,
                    downsample_factor=8,
                    verbose=verbose,
                )

            with _OverlapPool(max_workers=2) as ovl_pool:
                centroid_future = ovl_pool.submit(_compute_centroid_overlap)
                mask_future = ovl_pool.submit(_compute_mask_overlap)
                overlap_metrics = centroid_future.result()
                mask_overlap_metrics = mask_future.result()

            # Collect all metrics into comprehensive CSV
            import pandas as pd

            # Extract RANSAC refinement magnitude
            ransac_trans = ransac_affine_3x3[:2, 2]
            ransac_magnitude = np.linalg.norm(ransac_trans)

            all_metrics = {
                # Timing metrics (minutes) - rounded to 2 decimal places
                "time_pcc_min": round(dt0 / 60, 2) if dt0 is not None else None,
                "time_extract_centroids_min": round(dt_centroids / 60, 2),
                "time_matching_min": round(dt4 / 60, 2),
                "time_hu_graph_filtering_min": round(dt_hu / 60, 2) if use_hu_moments else None,
                "time_ransac_min": round(dt5b / 60, 2),
                "time_total_min": round((time.time() - t_start) / 60, 2),
                # PCC metrics - rounded to 2 decimal places
                "pcc_shift_dy": round(pcc_shift[0], 2) if pcc_shift is not None else None,
                "pcc_shift_dx": round(pcc_shift[1], 2) if pcc_shift is not None else None,
                # Centroid counts (integers, no rounding needed)
                "n_source_centroids_total": len(source_centroids),
                "n_target_centroids_total": len(target_centroids),
                "n_source_centroids_subsampled": len(source_centroids_for_hu) if use_hu_moments else None,
                "n_target_centroids_subsampled": len(target_centroids_for_hu) if use_hu_moments else None,
                # Graph matching metrics - rounded to 2 decimal places
                "graph_matches": len(hu_source_idx) if use_hu_moments and len(hu_source_idx) > 0 else None,
                "graph_mean_score": round(np.mean(hu_distances), 2) if use_hu_moments and len(hu_distances) > 0 else None,
                "graph_median_score": round(np.median(hu_distances), 2) if use_hu_moments and len(hu_distances) > 0 else None,
                "graph_min_score": round(np.min(hu_distances), 2) if use_hu_moments and len(hu_distances) > 0 else None,
                "graph_max_score": round(np.max(hu_distances), 2) if use_hu_moments and len(hu_distances) > 0 else None,
                # RANSAC metrics - rounded to 2 decimal places
                "ransac_n_matches": metrics["n_matches"],
                "ransac_n_inliers": metrics["n_inliers"],
                "ransac_inlier_ratio": round(metrics["inlier_ratio"], 2),
                "ransac_residual_mean": round(metrics["residual_mean"], 2),
                "ransac_residual_std": round(metrics["residual_std"], 2),
                "ransac_residual_max": round(metrics.get("residual_max"), 2) if metrics.get("residual_max") is not None else None,
                "ransac_refinement_dy": round(ransac_trans[0], 2),
                "ransac_refinement_dx": round(ransac_trans[1], 2),
                "ransac_refinement_magnitude": round(ransac_magnitude, 2),
                # Final transformation - rounded to 2 decimal places
                "final_translation_dy": round(affine_3x3[1, 2], 2),
                "final_translation_dx": round(affine_3x3[0, 2], 2),
                "final_rotation_deg": round(np.degrees(np.arctan2(affine_3x3[1, 0], affine_3x3[0, 0])), 2),
                "final_scale_x": round(np.sqrt(affine_3x3[0, 0]**2 + affine_3x3[1, 0]**2), 2),
                "final_scale_y": round(np.sqrt(affine_3x3[0, 1]**2 + affine_3x3[1, 1]**2), 2),
                # Comparison with manual (if available) - rounded to 2 decimal places
                "manual_translation_diff_px": round(comparison["translation_diff_pixels"], 2) if comparison else None,
                "manual_rotation_diff_deg": round(comparison["rotation_diff_degrees"], 2) if comparison else None,
                "manual_scale_diff_pct": round(comparison["scale_diff_percent"], 2) if comparison else None,
            }

            # Flatten centroid overlap metrics into separate rows
            # overlap_metrics = {"before": {"iou": 0.5, ...}, "pcc": {...}, ...}
            for stage, stage_metrics in overlap_metrics.items():
                if isinstance(stage_metrics, dict):
                    for metric_name, metric_value in stage_metrics.items():
                        all_metrics[f"overlap_{stage}_{metric_name}"] = round(metric_value, 2) if isinstance(metric_value, (int, float)) else metric_value
                else:
                    all_metrics[f"overlap_{stage}"] = round(stage_metrics, 2) if isinstance(stage_metrics, (int, float)) else stage_metrics

            # Flatten binary mask (segmentation mask) overlap metrics
            for stage, stage_metrics in mask_overlap_metrics.items():
                if isinstance(stage_metrics, dict):
                    for metric_name, metric_value in stage_metrics.items():
                        all_metrics[f"mask_overlap_{stage}_{metric_name}"] = round(metric_value, 2) if isinstance(metric_value, (int, float)) else metric_value

            # Convert to DataFrame with metrics as rows (not columns)
            df_metrics = pd.DataFrame([
                {"metric": k, "value": v} for k, v in all_metrics.items()
            ])
            metrics_csv_path = overlay_output_dir / "auto_register_metrics.csv"
            df_metrics.to_csv(metrics_csv_path, index=False)
            if verbose:
                print(f"    Saved comprehensive metrics to: {metrics_csv_path.name}")

            # Create graph-matched cell visualization (always create, no caching)
            if use_hu_moments and hu_filtered_matches is not None:
                if verbose:
                    print(f"    Creating graph-matched cell visualization (100 cells)...")

                t_hu_viz = time.time()

                visualize_hu_moment_matches(
                    mask_src=mask_src,
                    mask_tgt=mask_tgt,
                    all_source_centroids=hu_filtered_matches["all_source"],
                    all_target_centroids=hu_filtered_matches["all_target"],
                    hu_source_idx=hu_filtered_matches["hu_source_idx"],
                    hu_target_idx=hu_filtered_matches["hu_target_idx"],
                    hu_distances=hu_filtered_matches["hu_distances"],
                    source_graphs=hu_filtered_matches.get("source_graphs"),
                    target_graphs=hu_filtered_matches.get("target_graphs"),
                    output_path=overlay_output_dir / "00d_graph_matched_cells_100.png",
                    pcc_shift=pcc_shift,
                    n_display=100,  # Show 100 cells (10×10 grid)
                    crop_size=64,   # Smaller crops for compact layout
                    n_cols=10,      # 10 pairs per row
                    verbose=verbose,
                )

                dt_hu_viz = time.time() - t_hu_viz
                if verbose:
                    print(f"      Created graph match visualization ({dt_hu_viz:.2f}s)")

        if verbose:
            print(f"    Created overlays ({time.time()-t7:.2f}s total)")

    results = {
        "affine_3x3": affine_3x3,
        "affine_4x4": affine_4x4,
        "metrics": metrics,
        "output_yaml": output_yaml_path,
        "overlay_dir": overlay_output_dir if create_overlays else None,
        "comparison": comparison,
        "overlap_percent": (
            overlap_metrics.get("forward", {}).get("overlap_percent", 0.0)
            if overlap_metrics
            else 0.0
        ),
    }

    return results


# Ordered list of registration retry strategies.
# Each strategy is submitted as its own SLURM job — see auto_register_orchestrator.py for details.
RETRY_STRATEGY_ORDER = ["default", "toggle_pcc", "compose"]


def get_retry_strategies(pcc_was_skipped: bool) -> dict[str, dict]:
    """Return parameter overrides for each retry strategy.

    The overrides depend on the default PCC state (which varies by well).
    """
    return {
        "default": {},
        "toggle_pcc": {"skip_pcc": not pcc_was_skipped},
        "refine": {"use_seed_affine": True},  # load existing affine as pre-alignment, skip PCC
        "compose": {},  # handled specially in auto_register_orchestrator.py
    }


def compose_registration(
    experiment: str,
    well,
    reg_type: str = "iss",
    verbose: bool = True,
    skip_track: bool = False,
    **kwargs,
) -> dict | None:
    """
    Compose a registration from two existing registrations when direct fails.

    For ISS→Track: compose ISS→Pheno (run on the fly) + Pheno→Track (existing).
    For Pheno→Track: compose Pheno→ISS (inverse of ISS→Pheno) + ISS→Track (existing).

    Parameters
    ----------
    experiment : str
        Experiment name.
    well : int
        Well number (1, 2, or 3).
    reg_type : str
        Which registration failed: "iss" or "pheno".
    verbose : bool
        Print progress.

    Returns
    -------
    dict or None
        Results dict with composed affine, or None if composition not possible.
    """
    import yaml as _yaml
    from cyclops_utils.data.experiment import OpsDataset

    dataset = OpsDataset(experiment)
    row, col = parse_well(well)
    position = f"{row}/{col}/0"

    if reg_type == "iss":
        # ISS→Track failed. Compose: Pheno→Track @ ISS→Pheno
        pheno_track_yaml = dataset.append_well("auto_pheno_register", position)
        if not pheno_track_yaml.exists():
            if verbose:
                print(f"  compose: Pheno→Track YAML not found at {pheno_track_yaml}, cannot compose")
            return None

        if verbose:
            print(f"\n{'='*60}")
            print(f"COMPOSE strategy: ISS→Track = Pheno→Track @ ISS→Pheno")
            print(f"  Step 1: Run ISS→Pheno registration")
            print(f"{'='*60}")

        # Run ISS→Pheno by calling auto_register_iss_to_track with skip_track=True.
        # Force PCC on: the well 2 PCC auto-skip was only tested against tracking,
        # PCC may work fine for ISS→Pheno (different modality pair).
        iss_pheno_result = auto_register_iss_to_track(
            experiment=experiment,
            well=well,
            verbose=verbose,
            skip_track=True,  # registers ISS→Pheno instead of ISS→Track
            force_pcc=True,   # override well 2 auto-skip
            strategy=None,
        )

        iss_pheno_yaml = dataset.append_well("auto_iss_register", position)
        if not iss_pheno_yaml.exists():
            if verbose:
                print(f"  compose: ISS→Pheno registration failed, cannot compose")
            return None

        # Load both affines
        with open(iss_pheno_yaml) as f:
            iss_pheno_affine = np.array(_yaml.safe_load(f)["affine_transform_zyx"])
        with open(pheno_track_yaml) as f:
            pheno_track_affine = np.array(_yaml.safe_load(f)["affine_transform_zyx"])

        # YAMLs store inverse transforms. For inverse composition:
        # inv(ISS→Track) = inv(ISS→Pheno) @ inv(Pheno→Track)
        composed_affine = iss_pheno_affine @ pheno_track_affine

        if verbose:
            iss_pheno_trans = iss_pheno_affine[1:3, 3]
            pheno_track_trans = pheno_track_affine[1:3, 3]
            composed_trans = composed_affine[1:3, 3]
            print(f"\n  Step 2: Compose affines")
            print(f"    ISS→Pheno trans:   Y={iss_pheno_trans[0]:.1f}, X={iss_pheno_trans[1]:.1f}")
            print(f"    Pheno→Track trans: Y={pheno_track_trans[0]:.1f}, X={pheno_track_trans[1]:.1f}")
            print(f"    Composed trans:    Y={composed_trans[0]:.1f}, X={composed_trans[1]:.1f}")

        # Save composed result as the ISS→Track YAML
        # composed_affine is already in biahub convention (inverse) because
        # both input YAMLs store inverse transforms:
        #   Track→Pheno @ Pheno→ISS = Track→ISS = inv(ISS→Track)
        output_yaml = dataset.append_well("auto_iss_register", position)
        affine_data = {
            "affine_transform_zyx": composed_affine.tolist(),
            "interpolation": "linear",
            "keep_overhang": False,
            "source_channel_names": ["segmentation"],
            "target_channel_name": "segmentation",
            "time_indices": "all",
            "_composed_from": {
                "iss_to_pheno": str(iss_pheno_yaml),
                "pheno_to_track": str(pheno_track_yaml),
            },
        }
        # Atomic write so concurrent SLURM readers can't see a half-written file.
        _tmp = output_yaml.with_name(f".{output_yaml.name}.tmp.{os.getpid()}")
        with open(_tmp, "w") as f:
            _yaml.dump(affine_data, f, default_flow_style=False)
        os.replace(_tmp, output_yaml)

        if verbose:
            print(f"    Saved composed ISS→Track to: {output_yaml}")

        # Compute mask overlap to validate the composed registration
        paths = resolve_registration_paths(dataset, experiment, well, "iss", skip_track=skip_track)
        overlap_metrics = compute_overlap_metrics(
            source_seg_path=paths["source_seg_path"],
            target_seg_path=paths["target_seg_path"],
            position=position,
            affine_yaml_path=output_yaml,
            t_idx_source=0,
            t_idx_target=paths["t_idx_target"],
            verbose=verbose,
        )
        mask_overlap = overlap_metrics.get("mask_forward", {}).get("overlap_percent", 0.0)
        if verbose:
            print(f"    Composed mask overlap: {mask_overlap:.1f}%")

        # Compare with manual if available
        manual_yaml = dataset.append_well("iss_seg_register", position)
        comparison = None
        if manual_yaml.exists():
            with open(manual_yaml) as f:
                manual_affine = np.array(_yaml.safe_load(f)["affine_transform_zyx"])
            manual_trans = manual_affine[1:3, 3]
            composed_trans = composed_affine[1:3, 3]
            trans_diff = np.linalg.norm(manual_trans - composed_trans)
            if verbose:
                print(f"    Manual trans:     Y={manual_trans[0]:.1f}, X={manual_trans[1]:.1f}")
                print(f"    Translation diff: {trans_diff:.1f} px")
            comparison = {"translation_diff_pixels": trans_diff}

            # Also compute mask overlap for manual registration as reference
            manual_overlap = compute_overlap_metrics(
                source_seg_path=paths["source_seg_path"],
                target_seg_path=paths["target_seg_path"],
                position=position,
                affine_yaml_path=manual_yaml,
                t_idx_source=0,
                t_idx_target=paths["t_idx_target"],
                verbose=False,
            )
            manual_mask_overlap = manual_overlap.get("mask_forward", {}).get("overlap_percent", 0.0)
            if verbose:
                print(f"    Manual mask overlap:   {manual_mask_overlap:.1f}%")

        # Write metrics to the standard overlay directory so validation picks them up
        import pandas as pd
        overlay_dir = dataset.tracking / "auto_overlays" / f"{row}{col}_iss_to_{'pheno' if skip_track else 'track'}"
        overlay_dir.mkdir(parents=True, exist_ok=True)

        all_metrics = {}
        for stage, stage_data in overlap_metrics.items():
            if isinstance(stage_data, dict):
                for metric_name, metric_value in stage_data.items():
                    # mask stages: mask_before, mask_pcc, mask_forward -> mask_overlap_{stage}_{metric}
                    # centroid stages: before, pcc, forward -> overlap_{stage}_{metric}
                    if stage.startswith("mask_"):
                        key = f"mask_overlap_{stage.removeprefix('mask_')}_{metric_name}"
                    else:
                        key = f"overlap_{stage}_{metric_name}"
                    all_metrics[key] = round(metric_value, 2) if isinstance(metric_value, (int, float)) else metric_value

        df_metrics = pd.DataFrame([
            {"metric": k, "value": v} for k, v in all_metrics.items()
        ])
        metrics_csv_path = overlay_dir / "auto_register_metrics.csv"
        df_metrics.to_csv(metrics_csv_path, index=False)
        if verbose:
            print(f"    Saved compose metrics to: {metrics_csv_path}")

        # Generate overlay images for visual confirmation
        from cyclops_process.processes.auto_register.auto_register_visualization import (
            create_final_alignment_overlays,
            load_mask_2d,
            compute_crop_regions,
        )
        compose_overlay_dir = overlay_dir / "compose"
        compose_overlay_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"    Generating compose overlay images → {compose_overlay_dir}")
        mask_src = load_mask_2d(paths["source_seg_path"], position, 0)
        mask_tgt = load_mask_2d(paths["target_seg_path"], position, paths["t_idx_target"])
        pcc_shift, _ = estimate_translation_pcc(
            paths["source_seg_path"], paths["target_seg_path"],
            position, position, 0,
            downsample_factor=8, use_cache=False, center_fraction=1.0,
        )
        crop_regions = compute_crop_regions(mask_src.shape, crop_size=1024)
        create_final_alignment_overlays(
            mask_src=mask_src,
            mask_tgt=mask_tgt,
            affine_3x3=None,
            crop_region=crop_regions,
            output_dir=compose_overlay_dir,
            pcc_shift=pcc_shift,
            center_fraction=1.0,
            auto_yaml_path=output_yaml,
            verbose=verbose,
        )

        return {
            "composed": True,
            "affine_4x4": composed_affine,
            "output_yaml": output_yaml,
            "comparison": comparison,
            "overlap_percent": mask_overlap,
            "iss_pheno_result": iss_pheno_result,
        }

    if verbose:
        print(f"  compose: No composition path available for reg_type={reg_type}")
    return None


def auto_register_iss_to_track(
    experiment: str,
    well,
    t_idx_source: int = 0,
    t_idx_target: int = 3,
    compare_with_manual: bool = True,
    center_fraction: float = 1.0,
    skip_pcc: bool = False,
    pcc_center_fraction: float = None,
    verbose: bool = True,
    use_cache: bool = False,
    skip_track: bool = False,
    force_pcc: bool = False,
    strategy: str = None,
) -> dict:
    """
    Automatically register ISS segmentation to tracking segmentation for a well.

    Parameters
    ----------
    experiment : str
        Experiment name (e.g., "ops0031_20250424").
    well : int
        Well number (1, 2, or 3).
    t_idx_source : int
        ISS time index.
    t_idx_target : int
        Tracking time index.
    compare_with_manual : bool
        Compare with manual affine if it exists.
    center_fraction : float
        Fraction of well to use from center (0.0-1.0).
        1.0 = full well, 0.3 = central 30% (for debug).
    skip_pcc : bool
        Skip PCC pre-alignment step.
    pcc_center_fraction : float, optional
        Fraction of well center to use for PCC (0.0-1.0). Default: 1.0 for ISS-to-Track.
    verbose : bool
        Print progress.
    skip_track : bool
        Skip track and register ISS directly to pheno (default: False).

    Returns
    -------
    dict
        Results including metrics and paths.
    """
    from cyclops_utils.data.experiment import OpsDataset

    dataset = OpsDataset(experiment)
    row, col = parse_well(well)
    well_token = f"{row}{col}"
    paths = resolve_registration_paths(dataset, experiment, well, "iss", skip_track=skip_track)
    source_seg_path = paths["source_seg_path"]
    target_seg_path = paths["target_seg_path"]
    position = paths["position"]
    t_idx_target = paths["t_idx_target"]

    if skip_track and verbose:
        print(f"  WARNING: Skipping track - registering ISS directly to pheno segmentation")

    # Add _debug suffix if in debug mode
    is_debug = center_fraction < 1.0

    output_yaml = dataset.append_well("auto_iss_register", position)
    if is_debug:
        output_yaml = output_yaml.with_stem(output_yaml.stem + "_debug")

    overlay_dir = dataset.tracking / "auto_overlays" / f"{well_token}_iss_to_{'pheno' if skip_track else 'track'}"
    if is_debug:
        overlay_dir = overlay_dir.with_name(overlay_dir.name + "_debug")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Auto-registering ISS→{'Pheno' if skip_track else 'Track'} for {experiment} well {well}")
        if is_debug:
            print(f"  DEBUG MODE: center_fraction={center_fraction}")
        print(f"{'='*60}")

    # Run registration with debug parameters
    params = DEFAULT_PARAMS.copy()
    params["center_fraction"] = center_fraction

    # Auto-skip PCC for well 2 when targeting track (sparse signal causes poor PCC)
    # When skip_track=True (ISS→Pheno), PCC works fine — pheno has dense signal
    # Override with force_pcc=True to use PCC anyway
    skip_pcc = skip_pcc or (col == 2 and not skip_track and not force_pcc)
    if col == 2 and not skip_track and not force_pcc and not params.get("skip_pcc", False):
        print(f"  WARNING: Auto-enabling skip_pcc for well 2 (iss_w{well_token}) - PCC often fails for this well")
    params["skip_pcc"] = skip_pcc

    # ISS-to-Track specific overrides: use full well and aggressive downsampling for PCC
    # ISS has sparse signal, so we need full well context and can tolerate more downsampling
    params["pcc_center_fraction"] = pcc_center_fraction if pcc_center_fraction is not None else 1.0
    params["pcc_downsample_factor"] = None  # Let adaptive logic select based on image size

    print(f"  ISS→Track mode: center_fraction={center_fraction}, pcc_center_fraction={params['pcc_center_fraction']}, pcc_downsample={params['pcc_downsample_factor']}x, skip_pcc={skip_pcc}")

    # Get manual yaml path if requested
    manual_yaml = None
    if compare_with_manual:
        manual_yaml = dataset.append_well("iss_seg_register", position)
        if not manual_yaml.exists():
            if verbose:
                print(f"\nWarning: No manual affine found at {manual_yaml}")
            manual_yaml = None

    # Apply strategy overrides if requested (see RETRY_STRATEGY_ORDER / auto_register_orchestrator.py)
    main_overlay_dir = overlay_dir
    is_retry = strategy and strategy != "default"

    if is_retry:
        pcc_was_skipped = params["skip_pcc"]
        strategy_overrides = get_retry_strategies(pcc_was_skipped)

        if strategy not in strategy_overrides:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Available: {list(strategy_overrides.keys())}"
            )

        params.update(strategy_overrides[strategy])

        if verbose:
            print(f"  Strategy '{strategy}' overrides: {strategy_overrides[strategy]}")

        # Use separate overlay dir for non-default strategies to preserve all outputs
        overlay_dir = overlay_dir / strategy

        # Read existing overlap from the main overlay dir so we only overwrite
        # the output YAML + metrics if this strategy produces a better result.
        existing_overlap = _read_existing_overlap(main_overlay_dir)
        if verbose and existing_overlap is not None:
            print(f"  Existing overlap from prior strategy: {existing_overlap:.1f}%")

        # Back up the current output YAML before the retry overwrites it
        import shutil
        yaml_backup = None
        if output_yaml.exists():
            yaml_backup = output_yaml.with_suffix(".yml.bak")
            shutil.copyfile(output_yaml, yaml_backup)

    # For "refine" strategy, use existing output YAML as seed affine
    _seed_path = None
    if strategy == "refine" and output_yaml.exists():
        _seed_path = output_yaml
        if verbose:
            print(f"  Refine mode: using existing affine as seed: {output_yaml.name}")

    result = auto_estimate_registration(
        source_seg_path=source_seg_path,
        target_seg_path=target_seg_path,
        position=position,
        output_yaml_path=output_yaml,
        t_idx_source=t_idx_source,
        t_idx_target=t_idx_target,
        params=params,
        create_overlays=True,
        overlay_output_dir=overlay_dir,
        verbose=verbose,
        manual_yaml_path=manual_yaml,
        use_cache=use_cache,
        seed_affine_path=_seed_path,
    )

    overlap = result.get("overlap_percent", 0.0)
    strategy_name = strategy or "default"

    if verbose:
        print(f"  Strategy '{strategy_name}': overlap = {overlap:.1f}%")

    # For retry strategies: only promote result to main overlay dir if it improved
    if is_retry:
        existing_overlap = existing_overlap if existing_overlap is not None else -1.0
        if overlap > existing_overlap:
            if verbose:
                print(f"  ✓ Strategy '{strategy_name}' improved overlap: "
                      f"{existing_overlap:.1f}% → {overlap:.1f}%")
                print(f"    Promoting result to main overlay dir and output YAML")
            # Copy metrics CSV to main overlay dir so validation picks it up
            strategy_metrics = overlay_dir / "auto_register_metrics.csv"
            main_metrics = main_overlay_dir / "auto_register_metrics.csv"
            if strategy_metrics.exists():
                shutil.copyfile(strategy_metrics, main_metrics)
            # output_yaml was already written by auto_estimate_registration
            # Clean up backup
            if yaml_backup and yaml_backup.exists():
                yaml_backup.unlink()
        else:
            if verbose:
                print(f"  ✗ Strategy '{strategy_name}' did not improve overlap "
                      f"({overlap:.1f}% vs existing {existing_overlap:.1f}%)")
            # Restore the previous best YAML since auto_estimate_registration
            # already overwrote it with the worse result
            if yaml_backup and yaml_backup.exists():
                shutil.copyfile(yaml_backup, output_yaml)
                yaml_backup.unlink()
                if verbose:
                    print(f"    Restored previous best YAML from backup")

    return result


def _read_existing_overlap(overlay_dir) -> float | None:
    """Read overlap_forward_overlap_percent from an overlay dir's metrics CSV."""
    import pandas as pd
    metrics_csv = overlay_dir / "auto_register_metrics.csv"
    if not metrics_csv.exists():
        return None
    try:
        df = pd.read_csv(metrics_csv, index_col=0)
        if "overlap_forward_overlap_percent" in df.index:
            val = df.loc["overlap_forward_overlap_percent", df.columns[0]]
            return float(val) if pd.notna(val) else None
    except Exception:
        return None
    return None


def auto_register_pheno_to_track(
    experiment: str,
    well,
    t_idx_source: int = None,
    t_idx_target: int = None,
    compare_with_manual: bool = True,
    center_fraction: float = 1.0,
    skip_pcc: bool = False,
    pcc_center_fraction: float = None,
    verbose: bool = True,
    use_cache: bool = False,
    skip_track: bool = False,
    force_pcc: bool = False,
    strategy: str = None,
) -> dict:
    """
    Automatically register phenotyping segmentation to tracking segmentation for a well.

    Parameters
    ----------
    experiment : str
        Experiment name (e.g., "ops0031_20250424").
    well : int
        Well number (1, 2, or 3).
    t_idx_source : int, optional
        Phenotyping time index. If None, auto-determined from well/ops number.
    t_idx_target : int, optional
        Tracking time index. If None, auto-determined from well/ops number.
    compare_with_manual : bool
        Compare with manual affine if it exists.
    center_fraction : float
        Fraction of well to use from center (0.0-1.0).
        1.0 = full well, 0.3 = central 30% (for debug).
    skip_pcc : bool
        Skip PCC pre-alignment step.
    pcc_center_fraction : float, optional
        Fraction of well center to use for PCC (0.0-1.0). Default: 0.5 for Pheno-to-Track.
    verbose : bool
        Print progress.
    skip_track : bool
        Skip track and register pheno directly to ISS (default: False).

    Returns
    -------
    dict
        Results including metrics and paths.
    """
    from cyclops_utils.data.experiment import OpsDataset

    dataset = OpsDataset(experiment)
    row, col = parse_well(well)
    well_token = f"{row}{col}"
    paths = resolve_registration_paths(dataset, experiment, well, "pheno", skip_track=skip_track)
    source_seg_path = paths["source_seg_path"]
    target_seg_path = paths["target_seg_path"]
    position = paths["position"]

    if t_idx_source is None:
        t_idx_source = paths["t_idx_source"]
    if t_idx_target is None:
        t_idx_target = paths["t_idx_target"]

    if skip_track and verbose:
        print(f"  WARNING: Skipping track - registering pheno directly to ISS segmentation")

    # Add _debug suffix if in debug mode
    is_debug = center_fraction < 1.0

    output_yaml = dataset.append_well("auto_pheno_register", position)
    if is_debug:
        output_yaml = output_yaml.with_stem(output_yaml.stem + "_debug")

    overlay_dir = dataset.tracking / "auto_overlays" / f"{well_token}_pheno_to_{'iss' if skip_track else 'track'}"
    if is_debug:
        overlay_dir = overlay_dir.with_name(overlay_dir.name + "_debug")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Auto-registering Pheno→{'ISS' if skip_track else 'Track'} for {experiment} well {well}")
        if is_debug:
            print(f"  DEBUG MODE: center_fraction={center_fraction}")
        print(f"  t_idx_source={t_idx_source}, t_idx_target={t_idx_target}")
        print(f"{'='*60}")

    # Run registration with debug parameters
    params = DEFAULT_PARAMS.copy()
    params["center_fraction"] = center_fraction

    # Auto-skip PCC for well 2 (pheno_w2) - known to have poor PCC alignment
    # Override with force_pcc=True to use PCC anyway
    skip_pcc = skip_pcc or (col == 2 and not force_pcc)
    if col == 2 and not force_pcc and not params.get("skip_pcc", False):
        print(f"  WARNING: Auto-enabling skip_pcc for well 2 (pheno_w{well_token}) - PCC often fails for this well")

    params["skip_pcc"] = skip_pcc
    if pcc_center_fraction is not None:
        params["pcc_center_fraction"] = pcc_center_fraction

    print(f"  Pheno→Track mode: Setting skip_pcc={skip_pcc}, pcc_center_fraction={params.get('pcc_center_fraction', 0.5)}")

    # Get manual yaml path if requested
    manual_yaml = None
    if compare_with_manual:
        manual_yaml = dataset.append_well("lc_20x_seg_register", position)
        if not manual_yaml.exists():
            if verbose:
                print(f"\nWarning: No manual affine found at {manual_yaml}")
            manual_yaml = None

    # Apply strategy overrides (same system as ISS, see auto_register_iss_to_track)
    main_overlay_dir = overlay_dir
    is_retry = strategy and strategy != "default"

    if is_retry:
        pcc_was_skipped = params["skip_pcc"]
        strategy_overrides = get_retry_strategies(pcc_was_skipped)

        if strategy not in strategy_overrides:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Available: {list(strategy_overrides.keys())}"
            )

        params.update(strategy_overrides[strategy])

        if verbose:
            print(f"  Strategy '{strategy}' overrides: {strategy_overrides[strategy]}")

        overlay_dir = overlay_dir / strategy
        existing_overlap = _read_existing_overlap(main_overlay_dir)
        if verbose and existing_overlap is not None:
            print(f"  Existing overlap from prior strategy: {existing_overlap:.1f}%")

        # Back up the current output YAML before the retry overwrites it
        import shutil
        yaml_backup = None
        if output_yaml.exists():
            yaml_backup = output_yaml.with_suffix(".yml.bak")
            shutil.copyfile(output_yaml, yaml_backup)

    # For "refine" strategy, use existing output YAML as seed affine
    _seed_path = None
    if strategy == "refine" and output_yaml.exists():
        _seed_path = output_yaml
        if verbose:
            print(f"  Refine mode: using existing affine as seed: {output_yaml.name}")

    results = auto_estimate_registration(
        source_seg_path=source_seg_path,
        target_seg_path=target_seg_path,
        position=position,
        output_yaml_path=output_yaml,
        t_idx_source=t_idx_source,
        t_idx_target=t_idx_target,
        params=params,
        create_overlays=True,
        overlay_output_dir=overlay_dir,
        verbose=verbose,
        manual_yaml_path=manual_yaml,
        use_cache=use_cache,
        seed_affine_path=_seed_path,
    )

    overlap = results.get("overlap_percent", 0.0)
    strategy_name = strategy or "default"

    if verbose:
        print(f"  Strategy '{strategy_name}': overlap = {overlap:.1f}%")

    # For retry strategies: only promote result to main overlay dir if it improved
    if is_retry:
        existing_overlap = existing_overlap if existing_overlap is not None else -1.0
        if overlap > existing_overlap:
            if verbose:
                print(f"  ✓ Strategy '{strategy_name}' improved overlap: "
                      f"{existing_overlap:.1f}% → {overlap:.1f}%")
                print(f"    Promoting result to main overlay dir and output YAML")
            strategy_metrics = overlay_dir / "auto_register_metrics.csv"
            main_metrics = main_overlay_dir / "auto_register_metrics.csv"
            if strategy_metrics.exists():
                shutil.copyfile(strategy_metrics, main_metrics)
            if yaml_backup and yaml_backup.exists():
                yaml_backup.unlink()
        else:
            if verbose:
                print(f"  ✗ Strategy '{strategy_name}' did not improve overlap "
                      f"({overlap:.1f}% vs existing {existing_overlap:.1f}%)")
            if yaml_backup and yaml_backup.exists():
                shutil.copyfile(yaml_backup, output_yaml)
                yaml_backup.unlink()
                if verbose:
                    print(f"    Restored previous best YAML from backup")

    return results


def auto_register_cell_painting_to_pheno(
    experiment: str,
    well,
    part: int = 1,
    t_idx_source: int = 0,
    t_idx_target: int = 0,
    compare_with_manual: bool = False,
    center_fraction: float = 1.0,
    verbose: bool = True,
    use_cache: bool = False,
) -> dict:
    """
    Automatically register cell painting segmentation to phenotyping segmentation for a well.

    Parameters
    ----------
    experiment : str
        Experiment name (e.g., "ops0094_20251217").
    well : int
        Well number (1, 2, or 3).
    part : int
        Cell painting part number (1 or 2).
    t_idx_source : int
        Cell painting time index (default: 0).
    t_idx_target : int
        Phenotyping time index (default: 0).
    compare_with_manual : bool
        Compare with manual affine if it exists.
    center_fraction : float
        Fraction of well to use from center (0.0-1.0).
        1.0 = full well, 0.3 = central 30% (for debug).
    verbose : bool
        Print progress.
    use_cache : bool
        Use cached centroid data if available.

    Returns
    -------
    dict
        Results including metrics and paths.
    """
    from cyclops_utils.data.experiment import OpsDataset
    from pathlib import Path

    dataset = OpsDataset(experiment)
    row, col = parse_well(well)
    well_token = f"{row}{col}"
    position = f"{row}/{col}/0"

    # Source is cell painting segmentation
    # Path format: /path/to/ops_data/.../0-convert/cell_painting/part{N}_max_proj_segmentation.zarr
    cell_painting_base = dataset.experiment_path / "0-convert" / "cell_painting"
    source_seg_path = cell_painting_base / f"part{part}_max_proj_segmentation.zarr"

    if not source_seg_path.exists():
        raise FileNotFoundError(f"Cell painting segmentation not found: {source_seg_path}")

    # Target is phenotyping segmentation (20x stitched)
    target_seg_path = dataset.store_paths["pheno_assembled_v3"]

    if not target_seg_path.exists():
        raise FileNotFoundError(f"Phenotyping segmentation not found: {target_seg_path}")

    # Add _debug suffix if in debug mode
    is_debug = center_fraction < 1.0

    # Output YAML path, e.g. A{well}_cell_painting{part}_register.yml
    output_yaml = dataset.tracking / f"{well_token}_cell_painting{part}_register.yml"
    if is_debug:
        output_yaml = output_yaml.with_stem(output_yaml.stem + "_debug")

    overlay_dir = dataset.tracking / "auto_overlays" / f"{well_token}_cell_painting{part}_to_pheno"
    if is_debug:
        overlay_dir = overlay_dir.with_name(overlay_dir.name + "_debug")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Auto-registering CellPainting{part}→Pheno for {experiment} well {well}")
        if is_debug:
            print(f"  DEBUG MODE: center_fraction={center_fraction}")
        print(f"  Source: {source_seg_path}")
        print(f"  Target: {target_seg_path}")
        print(f"  Output: {output_yaml}")
        print(f"{'='*60}")

    # Run registration with parameters
    params = DEFAULT_PARAMS.copy()
    params["center_fraction"] = center_fraction

    # Cell painting to pheno: use full well and moderate downsampling
    params["pcc_center_fraction"] = 1.0
    params["pcc_downsample_factor"] = 8

    if verbose:
        print(f"  CellPainting→Pheno mode: PCC full well with {params['pcc_downsample_factor']}x downsampling")

    # Get manual yaml path if requested (unlikely to exist for cell painting)
    manual_yaml = None
    if compare_with_manual:
        manual_yaml = dataset.tracking / f"{well_token}_cell_painting{part}_register_manual.yml"
        if not manual_yaml.exists():
            if verbose:
                print(f"\nNote: No manual affine found at {manual_yaml}")
            manual_yaml = None

    results = auto_estimate_registration(
        source_seg_path=source_seg_path,
        target_seg_path=target_seg_path,
        position=position,
        output_yaml_path=output_yaml,
        t_idx_source=t_idx_source,
        t_idx_target=t_idx_target,
        params=params,
        create_overlays=True,
        overlay_output_dir=overlay_dir,
        verbose=verbose,
        manual_yaml_path=manual_yaml,
        use_cache=use_cache,
    )

    return results


def compute_overlap_metrics(
    source_seg_path: Path,
    target_seg_path: Path,
    position: str,
    affine_yaml_path: Path,
    t_idx_source: int = 0,
    t_idx_target: int = 0,
    center_fraction: float = 1.0,
    min_area: int = 100,
    verbose: bool = True,
) -> dict:
    """
    Compute overlap metrics for an existing registration.

    Parameters
    ----------
    source_seg_path : Path
        Path to source segmentation zarr.
    target_seg_path : Path
        Path to target segmentation zarr.
    position : str
        Position string (e.g., "A/1/0").
    affine_yaml_path : Path
        Path to saved affine YAML file.
    t_idx_source : int
        Source time index.
    t_idx_target : int
        Target time index.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Overlap metrics at different stages.
    """
    import yaml
    from scipy import ndimage
    from skimage.transform import warp, AffineTransform

    if verbose:
        print(f"Computing overlap metrics for: {affine_yaml_path.name}")
        print(f"  Source: {source_seg_path}")
        print(f"  Target: {target_seg_path}")
        print(f"  Position: {position}")

    # Load affine from YAML
    with open(affine_yaml_path, "r") as f:
        affine_data = yaml.safe_load(f)

    affine_4x4_inv = np.array(affine_data["affine_transform_zyx"])

    # Convert back to forward transform (saved as inverse)
    affine_4x4 = np.linalg.inv(affine_4x4_inv)

    # Extract 2D 3x3 affine (YX plane with translation)
    affine_3x3 = np.eye(3)
    affine_3x3[:2, :2] = affine_4x4[1:3, 1:3]  # Rotation/scale
    affine_3x3[:2, 2] = affine_4x4[1:3, 3]  # Translation

    # Load PCC from cache
    if verbose:
        print(f"  Loading PCC from cache...")
    pcc_shift, pcc_cache_hit = estimate_translation_pcc(
        source_seg_path,
        target_seg_path,
        position,
        position,
        t_idx_source,
        downsample_factor=8,
        use_cache=False,
        center_fraction=1.0,  # Full image for overlap metrics
    )
    if verbose:
        print(
            f"    PCC shift: dy={pcc_shift[0]:.1f}, dx={pcc_shift[1]:.1f} (cached: {pcc_cache_hit})"
        )

    # Load centroids from cache (should already be cached from registration run)
    if verbose:
        print(f"  Loading centroids from cache...")
    t_load = time.time()

    source_cache_path = _get_centroid_cache_path(
        source_seg_path, position, t_idx_source, min_area, center_fraction
    )
    target_cache_path = _get_centroid_cache_path(
        target_seg_path, position, t_idx_target, min_area, center_fraction
    )

    if verbose:
        print(f"    Looking for source cache: {source_cache_path}")
        print(f"    Looking for target cache: {target_cache_path}")

    if source_cache_path.exists():
        source_centroids = np.load(source_cache_path)
        if verbose:
            print(f"    Using cached source centroids: {len(source_centroids)} cells")
    else:
        if verbose:
            print(f"    Cache miss for source, extracting...")
        source_centroids = extract_centroids_from_segmentation(
            source_seg_path, position, t_idx_source, min_area, center_fraction
        )

    if target_cache_path.exists():
        target_centroids = np.load(target_cache_path)
        if verbose:
            print(f"    Using cached target centroids: {len(target_centroids)} cells")
    else:
        if verbose:
            print(f"    Cache miss for target, extracting...")
        target_centroids = extract_centroids_from_segmentation(
            target_seg_path, position, t_idx_target, min_area, center_fraction
        )

    # Get mask shape (needed for rasterization)
    store_src = open_ome_zarr(source_seg_path, mode="r")
    seg_src = store_src[position].data
    if seg_src.ndim >= 2:
        mask_shape = seg_src.shape[-2:]
    else:
        mask_shape = seg_src.shape

    if verbose:
        print(
            f"    Source: {len(source_centroids)} cells, Target: {len(target_centroids)} cells ({time.time()-t_load:.2f}s)"
        )

    # Compute centroid-based overlap metrics using shared function
    overlap_metrics = compute_registration_overlap_metrics(
        source_centroids=source_centroids,
        target_centroids=target_centroids,
        mask_shape=mask_shape,
        pcc_shift=pcc_shift,
        affine_3x3=affine_3x3,
        centroid_radius=10,  # 20x20 pixel circles
        verbose=verbose,
    )

    # Compute binary mask overlap (segmentation mask)
    if verbose:
        print(f"\n  Loading masks for segmentation mask overlap...")
    mask_src = load_mask_2d(source_seg_path, position, t_idx_source)
    mask_tgt = load_mask_2d(target_seg_path, position, t_idx_target)
    mask_overlap = compute_binary_mask_overlap_metrics(
        mask_src=mask_src,
        mask_tgt=mask_tgt,
        pcc_shift=pcc_shift,
        affine_3x3=affine_3x3,
        downsample_factor=8,
        verbose=verbose,
    )

    # Merge both into results
    overlap_metrics["mask_before"] = mask_overlap["before"]
    overlap_metrics["mask_pcc"] = mask_overlap["pcc"]
    overlap_metrics["mask_forward"] = mask_overlap["forward"]

    return overlap_metrics


def check_yaml_registration(
    experiment: str,
    well,
    registration_type: str,
    yaml_path: Path,
    verbose: bool = True,
    skip_track: bool = False,
) -> dict:
    """Recompute overlap metrics and generate overlay images for an existing affine YAML.

    Parameters
    ----------
    experiment : str
        Experiment name (e.g., "ops0100_20251218").
    well : int
        Well number (1, 2, or 3).
    registration_type : str
        "iss" or "pheno".
    yaml_path : Path
        Path to existing affine YAML file.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Combined centroid + mask overlap metrics.
    """
    import pandas as pd
    from cyclops_process.processes.auto_register.auto_register_visualization import (
        create_final_alignment_overlays,
        load_mask_2d,
        compute_crop_regions,
    )
    from cyclops_process.processes.auto_register.auto_register_utils import (
        print_overlap_quality_warning,
    )

    yaml_path = Path(yaml_path)
    dataset = OpsDataset(experiment)
    row, col = parse_well(well)
    well_token = f"{row}{col}"
    paths = resolve_registration_paths(dataset, experiment, well, registration_type, skip_track=skip_track)
    source_seg_path = paths["source_seg_path"]
    target_seg_path = paths["target_seg_path"]
    t_idx_source = paths["t_idx_source"]
    t_idx_target = paths["t_idx_target"]
    position = paths["position"]

    print(f"\n{'='*60}")
    print(f"Recomputing overlap metrics from YAML")
    print(f"  Experiment: {experiment}")
    print(f"  Well: {well}  Position: {position}")
    print(f"  Type: {registration_type}")
    print(f"  YAML: {yaml_path}")
    print(f"  t_idx: source={t_idx_source}, target={t_idx_target}")
    print(f"{'='*60}\n")

    # Compute centroid + mask overlap metrics
    metrics = compute_overlap_metrics(
        source_seg_path=source_seg_path,
        target_seg_path=target_seg_path,
        position=position,
        affine_yaml_path=yaml_path,
        t_idx_source=t_idx_source,
        t_idx_target=t_idx_target,
        center_fraction=1.0,
        min_area=100,
        verbose=verbose,
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"Overlap Metrics Summary")
    print(f"{'='*60}")
    print(f"  Centroid-based (individual cell co-localization):")
    for stage in ("before", "pcc", "forward"):
        if stage in metrics:
            m = metrics[stage]
            print(f"    {stage:>10s}:  overlap={m['overlap_percent']:.1f}%  "
                  f"IoU={m['iou']:.3f}  Dice={m['dice']:.3f}")

    print(f"\n  Binary mask (segmentation mask alignment):")
    for stage in ("mask_before", "mask_pcc", "mask_forward"):
        if stage in metrics:
            m = metrics[stage]
            label = stage.replace("mask_", "")
            print(f"    {label:>10s}:  overlap={m['overlap_percent']:.1f}%  "
                  f"IoU={m['iou']:.3f}  Dice={m['dice']:.3f}")

    mask_pct = metrics.get("mask_forward", {}).get("overlap_percent", 0.0)
    centroid_pct = metrics.get("forward", {}).get("overlap_percent", 0.0)
    print(f"\n  Mask overlap: {mask_pct:.1f}%  |  Centroid overlap: {centroid_pct:.1f}%")
    print_overlap_quality_warning(mask_pct, indent="  ")

    # Generate overlay images
    if skip_track:
        target = "pheno" if registration_type == "iss" else "iss"
    else:
        target = "track"
    overlay_dir = yaml_path.parent / "check_yaml" / f"{well_token}_{registration_type}_to_{target}"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating overlay images → {overlay_dir}")

    t_idx_source = paths["t_idx_source"]
    t_idx_target = paths["t_idx_target"]

    print("  Loading masks...")
    mask_src = load_mask_2d(source_seg_path, position, t_idx_source)
    mask_tgt = load_mask_2d(target_seg_path, position, t_idx_target)

    print("  Computing PCC shift...")
    pcc_shift, _ = estimate_translation_pcc(
        source_seg_path, target_seg_path,
        position, position,
        t_idx_source,
        downsample_factor=8,
        use_cache=False,
        center_fraction=1.0,
    )

    crop_regions = compute_crop_regions(mask_src.shape, crop_size=1024)

    print("  Creating final alignment overlays...")
    create_final_alignment_overlays(
        mask_src=mask_src,
        mask_tgt=mask_tgt,
        affine_3x3=None,
        crop_region=crop_regions,
        output_dir=overlay_dir,
        pcc_shift=pcc_shift,
        center_fraction=1.0,
        auto_yaml_path=yaml_path,
        verbose=verbose,
    )

    # Save metrics CSV
    metrics_rows = []
    for stage in ("before", "pcc", "forward"):
        if stage in metrics:
            for k, v in metrics[stage].items():
                metrics_rows.append({"metric": f"overlap_{stage}_{k}", "value": v})
    for stage in ("mask_before", "mask_pcc", "mask_forward"):
        if stage in metrics:
            for k, v in metrics[stage].items():
                metrics_rows.append({"metric": f"{stage}_{k}", "value": v})
    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(
            overlay_dir / "auto_register_metrics.csv", index=False
        )

    print(f"\n✓ Done — overlays and metrics saved to {overlay_dir}")
    return metrics


def check_all_yaml_registrations(
    experiment: str,
    wells: list = None,
    registration_types: list = None,
    verbose: bool = True,
    skip_track: bool = False,
) -> dict:
    """Recompute overlap metrics and overlays for all existing affine YAMLs in parallel.

    Discovers all ``A{well}_auto_register.yml`` (ISS) and
    ``A{well}_auto_pheno_register.yml`` (pheno) files in the experiment's
    tracking directory and runs ``check_yaml_registration`` for each.

    Parameters
    ----------
    experiment : str
        Experiment name.
    wells : list, optional
        Wells to check (default: [1, 2, 3]).
    registration_types : list, optional
        Types to check (default: ["iss", "pheno"]).
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        ``{job_label: metrics_dict}`` for each completed check.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if wells is None:
        wells = [1, 2, 3]
    if registration_types is None:
        # In skip_track mode, pheno is derived from ISS inverse — no pheno YAMLs exist
        registration_types = ["iss"] if skip_track else ["iss", "pheno"]

    dataset = OpsDataset(experiment)
    tracking_dir = dataset.tracking

    # Discover existing YAMLs
    jobs = []
    for well in wells:
        row, col = parse_well(well)
        well_token = f"{row}{col}"
        for reg_type in registration_types:
            if reg_type == "iss":
                yaml_name = f"{well_token}_auto_register.yml"
            else:
                yaml_name = f"{well_token}_auto_pheno_register.yml"
            yaml_path = tracking_dir / yaml_name
            if yaml_path.exists():
                jobs.append({
                    "well": well,
                    "registration_type": reg_type,
                    "yaml_path": yaml_path,
                    "label": f"{reg_type}_w{well_token}",
                })

    if not jobs:
        print(f"No affine YAMLs found in {tracking_dir}")
        return {}

    print(f"\n{'='*60}")
    print(f"Checking {len(jobs)} registrations for {experiment}")
    print(f"{'='*60}")
    for j in jobs:
        print(f"  - {j['label']}: {j['yaml_path'].name}")
    print()

    # Run in parallel
    all_results = {}
    with ProcessPoolExecutor(max_workers=min(len(jobs), 6)) as pool:
        futures = {}
        for j in jobs:
            fut = pool.submit(
                check_yaml_registration,
                experiment=experiment,
                well=j["well"],
                registration_type=j["registration_type"],
                yaml_path=j["yaml_path"],
                verbose=verbose,
                skip_track=skip_track,
            )
            futures[fut] = j["label"]

        for fut in as_completed(futures):
            label = futures[fut]
            try:
                all_results[label] = fut.result()
            except Exception as exc:
                print(f"\n  ✗ {label} FAILED: {exc}")
                all_results[label] = {"error": str(exc)}

    # Print combined summary table
    print(f"\n{'='*60}")
    print(f"Combined Registration Quality — {experiment}")
    print(f"{'='*60}")
    print(f"  {'Registration':<16s}  {'Centroid':>10s}  {'Mask':>10s}  {'Status'}")
    print(f"  {'-'*16}  {'-'*10}  {'-'*10}  {'-'*10}")

    for j in jobs:
        label = j["label"]
        m = all_results.get(label, {})
        if "error" in m:
            print(f"  {label:<16s}  {'ERROR':>10s}  {'ERROR':>10s}  ✗")
            continue
        centroid_pct = m.get("forward", {}).get("overlap_percent", 0.0)
        mask_pct = m.get("mask_forward", {}).get("overlap_percent", 0.0)
        if mask_pct >= 20:
            status = "✓ Good"
        elif mask_pct >= 10:
            status = "⚠ Low"
        else:
            status = "✗ Bad"
        print(f"  {label:<16s}  {centroid_pct:>9.1f}%  {mask_pct:>9.1f}%  {status}")

    print(f"{'='*60}\n")

    # Save combined metrics CSV to check_yaml/
    import pandas as pd
    check_yaml_dir = dataset.tracking / "check_yaml"
    check_yaml_dir.mkdir(parents=True, exist_ok=True)

    combined_rows = []
    for j in jobs:
        label = j["label"]
        m = all_results.get(label, {})
        if "error" in m:
            combined_rows.append({"registration": label, "error": m["error"]})
            continue
        row = {"registration": label}
        # Flatten all metric stages
        for stage_key in ("before", "pcc", "forward", "mask_before", "mask_pcc", "mask_forward"):
            if stage_key in m and isinstance(m[stage_key], dict):
                for metric_name, metric_value in m[stage_key].items():
                    row[f"{stage_key}_{metric_name}"] = round(metric_value, 2) if isinstance(metric_value, (int, float)) else metric_value
        combined_rows.append(row)

    if combined_rows:
        combined_csv = check_yaml_dir / "auto_register_metrics_combined.csv"

        # Backup previous CSV and compare
        prev_df = None
        if combined_csv.exists():
            import shutil
            backup_csv = check_yaml_dir / "auto_register_metrics_combined_prev.csv"
            shutil.copyfile(combined_csv, backup_csv)
            prev_df = pd.read_csv(backup_csv)

        new_df = pd.DataFrame(combined_rows)
        new_df.to_csv(combined_csv, index=False)
        print(f"Saved combined metrics: {combined_csv}")

        # Print before→after comparison if previous exists
        if prev_df is not None:
            print(f"\n{'='*80}")
            print(f"  Check-All Comparison (previous → current)")
            print(f"{'='*80}")
            print(f"  {'Registration':<16} {'Centroid':>18} {'Mask':>18}")
            print(f"  {'':<16} {'prev→now':>18} {'prev→now':>18}")
            print(f"  {'-'*16} {'-'*18} {'-'*18}")
            for _, row in new_df.iterrows():
                reg = row.get("registration", "?")
                prev_row = prev_df[prev_df["registration"] == reg]
                c_now = row.get("forward_overlap_percent", float("nan"))
                m_now = row.get("mask_forward_overlap_percent", float("nan"))
                if len(prev_row) > 0:
                    c_prev = prev_row.iloc[0].get("forward_overlap_percent", float("nan"))
                    m_prev = prev_row.iloc[0].get("mask_forward_overlap_percent", float("nan"))
                    c_str = f"{c_prev:.1f}→{c_now:.1f}%" if pd.notna(c_prev) else f"→{c_now:.1f}%"
                    m_str = f"{m_prev:.1f}→{m_now:.1f}%" if pd.notna(m_prev) else f"→{m_now:.1f}%"
                else:
                    c_str = f"→{c_now:.1f}%"
                    m_str = f"→{m_now:.1f}%"
                print(f"  {reg:<16} {c_str:>18} {m_str:>18}")
            print(f"{'='*80}")

    # Generate summary heatmap plot
    _plot_check_all_summary(jobs, all_results, experiment, check_yaml_dir)

    return all_results


def _plot_check_all_summary(jobs, all_results, experiment, output_dir):
    """Generate a 2-panel heatmap summarising centroid and mask overlap per well/type."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np

    reg_types = []
    wells = []
    for j in jobs:
        if j["registration_type"] not in reg_types:
            reg_types.append(j["registration_type"])
        if j["well"] not in wells:
            wells.append(j["well"])
    wells = sorted(wells)

    # Build matrices: rows = reg types, cols = wells
    centroid_matrix = np.full((len(reg_types), len(wells)), np.nan)
    mask_matrix = np.full((len(reg_types), len(wells)), np.nan)

    for j in jobs:
        m = all_results.get(j["label"], {})
        if "error" in m:
            continue
        ri = reg_types.index(j["registration_type"])
        ci = wells.index(j["well"])
        centroid_matrix[ri, ci] = m.get("forward", {}).get("overlap_percent", np.nan)
        mask_matrix[ri, ci] = m.get("mask_forward", {}).get("overlap_percent", np.nan)

    # Colormap: red < 10, yellow 10-15, green > 15
    cmap = LinearSegmentedColormap.from_list(
        "reg_quality",
        [(0.0, "#d32f2f"), (0.4, "#ff9800"), (0.6, "#fdd835"), (1.0, "#43a047")],
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    well_labels = [f"Well {w}" for w in wells]

    for ax, matrix, title in [
        (axes[0], centroid_matrix, "Centroid Overlap %"),
        (axes[1], mask_matrix, "Mask Overlap %"),
    ]:
        vmin, vmax = 0, 35
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

        ax.set_xticks(range(len(wells)))
        ax.set_xticklabels(well_labels, fontsize=11)
        ax.set_yticks(range(len(reg_types)))
        ax.set_yticklabels([t.upper() for t in reg_types], fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")

        # Annotate cells
        for ri in range(len(reg_types)):
            for ci in range(len(wells)):
                val = matrix[ri, ci]
                if np.isnan(val):
                    ax.text(ci, ri, "N/A", ha="center", va="center",
                            fontsize=12, color="gray", fontweight="bold")
                else:
                    text_color = "white" if val < 10 else "black"
                    ax.text(ci, ri, f"{val:.1f}%", ha="center", va="center",
                            fontsize=13, color=text_color, fontweight="bold")

    fig.suptitle(f"Registration Quality — {experiment}", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.colorbar(im, ax=axes[1], location="right", shrink=0.8, label="Overlap %", pad=0.04)

    out = output_dir / "registration_quality_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary plot: {output_dir / 'registration_quality_summary.png'}")


def main():
    """CLI for automatic registration."""
    import argparse
    import sys
    from cyclops_process.pipelinerunner.orchestrator import resolve_experiment_config

    parser = argparse.ArgumentParser(
        description="Automatic registration using graph-based matching + RANSAC"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ISS to Track registration
    iss_track = subparsers.add_parser(
        "iss-to-track", help="Register ISS segmentation to tracking segmentation"
    )
    iss_track.add_argument(
        "-e",
        "--exp",
        "--experiment",
        dest="experiment",
        type=str,
        required=True,
        help="Experiment name; partial names allowed (e.g., 'ops0042' or 'ops0042_20250520')",
    )
    iss_track.add_argument(
        "-w",
        "--well",
        type=str,
        required=True,
        help="Well number (1, 2, 3) or 'all' for all wells",
    )
    iss_track.add_argument(
        "--t-idx-source", type=int, default=0, help="ISS time index (default: 0)"
    )
    iss_track.add_argument(
        "--t-idx-target", type=int, default=None, help="Track time index (auto if None)"
    )
    iss_track.add_argument(
        "--no-compare", action="store_true", help="Skip comparison with manual affine"
    )
    iss_track.add_argument(
        "--center-fraction",
        type=float,
        default=1.0,
        help="Fraction of well to use from center (0.0-1.0, default: 1.0=full well). Use 0.3 for debug.",
    )
    iss_track.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache loading (force recomputation, but still save cache)",
    )
    iss_track.add_argument(
        "--skip-track",
        action="store_true",
        help="Skip track and register ISS directly to pheno segmentation",
    )
    iss_track.add_argument(
        "--skip-pcc", action="store_true", help="Skip PCC pre-alignment step"
    )
    iss_track.add_argument(
        "--pcc-center-fraction",
        type=float,
        default=None,
        help="Fraction of well center to use for PCC (0.0-1.0, default: 1.0 for ISS-to-Track)",
    )

    # Pheno to Track registration
    pheno_track = subparsers.add_parser(
        "pheno-to-track",
        help="Register phenotyping segmentation to tracking segmentation",
    )
    pheno_track.add_argument(
        "-e",
        "--exp",
        "--experiment",
        dest="experiment",
        type=str,
        required=True,
        help="Experiment name; partial names allowed (e.g., 'ops0042' or 'ops0042_20250520')",
    )
    pheno_track.add_argument(
        "-w",
        "--well",
        type=str,
        required=True,
        help="Well number (1, 2, 3) or 'all' for all wells",
    )
    pheno_track.add_argument(
        "--t-idx-source", type=int, default=None, help="Pheno time index (auto if None)"
    )
    pheno_track.add_argument(
        "--t-idx-target", type=int, default=None, help="Track time index (auto if None)"
    )
    pheno_track.add_argument(
        "--no-compare", action="store_true", help="Skip comparison with manual affine"
    )
    pheno_track.add_argument(
        "--center-fraction",
        type=float,
        default=1.0,
        help="Fraction of well to use from center (0.0-1.0, default: 1.0=full well). Use 0.3 for debug.",
    )
    pheno_track.add_argument(
        "--skip-pcc", action="store_true", help="Skip PCC pre-alignment step"
    )
    pheno_track.add_argument(
        "--pcc-center-fraction",
        type=float,
        default=None,
        help="Fraction of well center to use for PCC (0.0-1.0, default: 0.5 for Pheno-to-Track)",
    )
    pheno_track.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache loading (force recomputation, but still save cache)",
    )
    pheno_track.add_argument(
        "--skip-track",
        action="store_true",
        help="Skip track and register pheno directly to ISS segmentation",
    )

    # All registrations (both ISS→Track and Pheno→Track for all wells)
    all_cmd = subparsers.add_parser(
        "all", help="Register all wells for both ISS→Track and Pheno→Track"
    )
    all_cmd.add_argument(
        "-e",
        "--exp",
        "--experiment",
        dest="experiment",
        type=str,
        required=True,
        help="Experiment name; partial names allowed (e.g., 'ops0042' or 'ops0042_20250520')",
    )
    all_cmd.add_argument(
        "--center-fraction",
        type=float,
        default=1.0,
        help="Fraction of well to use from center (0.0-1.0, default: 1.0=full well). Use 0.3 for debug.",
    )
    all_cmd.add_argument(
        "--no-compare", action="store_true", help="Skip comparison with manual affine"
    )
    all_cmd.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache loading (force recomputation, but still save cache)",
    )

    # Compare affines
    compare = subparsers.add_parser(
        "compare", help="Compare manual vs automatic affines"
    )
    compare.add_argument("--manual", type=str, required=True, help="Manual YAML path")
    compare.add_argument("--auto", type=str, required=True, help="Automatic YAML path")

    # Compute overlap metrics for ISS-to-track
    overlap_iss = subparsers.add_parser(
        "overlap-iss-to-track",
        help="Compute overlap metrics for existing ISS→Track registration",
    )
    overlap_iss.add_argument(
        "-e",
        "--exp",
        "--experiment",
        dest="experiment",
        type=str,
        required=True,
        help="Experiment name; partial names allowed (e.g., 'ops0042' or 'ops0042_20250520')",
    )
    overlap_iss.add_argument(
        "-w",
        "--well",
        type=str,
        required=True,
        help="Well unit (1, B2, A/1/0)",
    )
    overlap_iss.add_argument(
        "--t-idx-source", type=int, default=0, help="ISS time index"
    )
    overlap_iss.add_argument(
        "--t-idx-target", type=int, default=3, help="Track time index (default: 3)"
    )

    # Compute overlap metrics for Pheno-to-track
    overlap_pheno = subparsers.add_parser(
        "overlap-pheno-to-track",
        help="Compute overlap metrics for existing Pheno→Track registration",
    )
    overlap_pheno.add_argument(
        "-e",
        "--exp",
        "--experiment",
        dest="experiment",
        type=str,
        required=True,
        help="Experiment name; partial names allowed (e.g., 'ops0042' or 'ops0042_20250520')",
    )
    overlap_pheno.add_argument(
        "-w",
        "--well",
        type=str,
        required=True,
        help="Well unit (1, B2, A/1/0)",
    )
    overlap_pheno.add_argument(
        "--t-idx-source", type=int, default=0, help="Pheno time index"
    )
    overlap_pheno.add_argument(
        "--t-idx-target", type=int, default=0, help="Track time index"
    )

    args = parser.parse_args()

    # Resolve experiment name for commands that need it (not "compare")
    if args.command in ["iss-to-track", "pheno-to-track", "all", "overlap-iss-to-track", "overlap-pheno-to-track"]:
        config_path = resolve_experiment_config(args.experiment, allow_interactive=True)
        if config_path is None:
            print("No experiment selected or found. Exiting.")
            sys.exit(1)

        # Extract experiment name from config path
        experiment = config_path.stem.replace("_config", "")
    else:
        experiment = getattr(args, "experiment", None)

    if args.command == "iss-to-track":
        # Handle "all" wells; keep raw well unit (parse_well resolves it downstream)
        wells = [1, 2, 3] if args.well.lower() == "all" else [args.well]

        for well in wells:
            # Build kwargs
            kwargs = {
                "experiment": experiment,
                "well": well,
                "t_idx_source": args.t_idx_source,
                "t_idx_target": args.t_idx_target if args.t_idx_target is not None else 0,
                "compare_with_manual": not args.no_compare,
                "center_fraction": args.center_fraction,
                "verbose": True,
                "use_cache": not args.no_cache,
                "skip_track": args.skip_track,
            }
            # Only override skip_pcc if --skip-pcc flag was provided
            if args.skip_pcc:
                kwargs["skip_pcc"] = True
            # Pass pcc_center_fraction if provided
            if args.pcc_center_fraction is not None:
                kwargs["pcc_center_fraction"] = args.pcc_center_fraction

            results = auto_register_iss_to_track(**kwargs)
            print(f"\n✓ Registration complete for well {well}!")
            print(f"  Output: {results['output_yaml']}")
            print(f"  Overlays: {results['overlay_dir']}")

    elif args.command == "pheno-to-track":
        # Handle "all" wells; keep raw well unit (parse_well resolves it downstream)
        wells = [1, 2, 3] if args.well.lower() == "all" else [args.well]

        for well in wells:
            # Build kwargs, only pass skip_pcc if explicitly provided via CLI
            kwargs = {
                "experiment": experiment,
                "well": well,
                "t_idx_source": args.t_idx_source,
                "t_idx_target": args.t_idx_target,
                "compare_with_manual": not args.no_compare,
                "center_fraction": args.center_fraction,
                "verbose": True,
                "use_cache": not args.no_cache,
            }
            # Only override skip_pcc if --skip-pcc flag was provided
            if args.skip_pcc:
                kwargs["skip_pcc"] = True
            # Pass pcc_center_fraction if provided
            if args.pcc_center_fraction is not None:
                kwargs["pcc_center_fraction"] = args.pcc_center_fraction
            # Pass skip_track if provided
            if args.skip_track:
                kwargs["skip_track"] = True

            results = auto_register_pheno_to_track(**kwargs)
            print(f"\n✓ Registration complete for well {well}!")
            print(f"  Output: {results['output_yaml']}")
            print(f"  Overlays: {results['overlay_dir']}")

    elif args.command == "all":
        # Run both ISS→Track and Pheno→Track for all wells
        wells = [1, 2, 3]

        for well in wells:
            print(f"\n{'='*60}")
            print(f"Processing well {well}")
            print(f"{'='*60}")

            # ISS → Track
            print(f"\n[1/2] ISS → Track registration...")
            results_iss = auto_register_iss_to_track(
                experiment=experiment,
                well=well,
                compare_with_manual=not args.no_compare,
                center_fraction=args.center_fraction,
                verbose=True,
                use_cache=not args.no_cache,
            )

            # Pheno → Track
            print(f"\n[2/2] Pheno → Track registration...")
            results_pheno = auto_register_pheno_to_track(
                experiment=experiment,
                well=well,
                compare_with_manual=not args.no_compare,
                center_fraction=args.center_fraction,
                verbose=True,
                use_cache=not args.no_cache,
            )

            print(f"\n✓ Well {well} complete!")
            print(f"  ISS→Track output: {results_iss['output_yaml']}")
            print(f"  Pheno→Track output: {results_pheno['output_yaml']}")

        print(f"\n{'='*60}")
        print(f"✓ All registrations complete!")
        print(f"{'='*60}")

    elif args.command == "compare":
        metrics = compare_affines(Path(args.manual), Path(args.auto))
        print(f"\nAffine Comparison:")
        print(f"  Translation diff: {metrics['translation_diff_pixels']:.2f} pixels")
        print(f"  Rotation diff: {metrics['rotation_diff_degrees']:.2f} degrees")
        print(f"  Scale diff: {metrics['scale_diff_percent']:.2f}%")
        print(f"  Manual determinant: {metrics['manual_determinant']:.4f}")
        print(f"  Auto determinant: {metrics['auto_determinant']:.4f}")

    elif args.command == "overlap-iss-to-track":
        from cyclops_utils.data.experiment import OpsDataset

        dataset = OpsDataset(experiment)
        _row, _col = parse_well(args.well)
        paths = resolve_registration_paths(dataset, experiment, args.well, "iss")
        affine_yaml = dataset.tracking / f"{_row}{_col}_auto_register.yml"

        metrics = compute_overlap_metrics(
            source_seg_path=paths["source_seg_path"],
            target_seg_path=paths["target_seg_path"],
            position=paths["position"],
            affine_yaml_path=affine_yaml,
            t_idx_source=args.t_idx_source or paths["t_idx_source"],
            t_idx_target=args.t_idx_target or paths["t_idx_target"],
            center_fraction=1.0,
            min_area=100,
            verbose=True,
        )

        print(f"\n✓ ISS→Track overlap computation complete!")

    elif args.command == "overlap-pheno-to-track":
        from cyclops_utils.data.experiment import OpsDataset

        dataset = OpsDataset(experiment)
        _row, _col = parse_well(args.well)
        paths = resolve_registration_paths(dataset, experiment, args.well, "pheno")
        affine_yaml = dataset.tracking / f"{_row}{_col}_auto_pheno_register.yml"

        metrics = compute_overlap_metrics(
            source_seg_path=paths["source_seg_path"],
            target_seg_path=paths["target_seg_path"],
            position=paths["position"],
            affine_yaml_path=affine_yaml,
            t_idx_source=args.t_idx_source or paths["t_idx_source"],
            t_idx_target=args.t_idx_target or paths["t_idx_target"],
            center_fraction=1.0,  # Must match what was used during registration
            min_area=100,  # Must match what was used during registration
            verbose=True,
        )

        print(f"\n✓ Pheno→Track overlap computation complete!")


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
# QUICK REFERENCE
# -----------------------------------------------------------------------------
#
# Single well, single type:
#   python -m cyclops_process.processes.auto_register iss-to-track --experiment ops0031_20250424 --well 1
#
# All wells, single type:
#   python -m cyclops_process.processes.auto_register iss-to-track --experiment ops0031_20250424 --well all
#
# All wells, both types:
#   python -m cyclops_process.processes.auto_register all --experiment ops0031_20250424
#
# Debug mode (any of the above + --center-fraction 0.3):
#   python -m cyclops_process.processes.auto_register all --experiment ops0031_20250424 --center-fraction 0.3

# python -m cyclops_process.processes.auto_register.auto_register pheno-to-track  --experiment ops0094_20251217  --well 3
# python -m cyclops_process.processes.auto_register.auto_register iss-to-track  --experiment 138  --well 2 --skip-pcc
# python -m cyclops_process.processes.auto_register.auto_register iss-to-track  --experiment ops0117_20260128 --well 1 --skip-tracks
