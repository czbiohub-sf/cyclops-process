"""
ISS Round-to-Round Registration Module.

Registers ISS imaging rounds using spot-based matching adapted from the proven
auto_register.py approach. Works in stitched well space to compute full affine
transforms per well.

Key Features:
- Round -1 (nucleus) → Round 0 (spots) registration using graph matching
- Sequential round-to-round registration (R0→R1, R1→R2, ..., R8→R9)
- Graph-based matching + RANSAC affine estimation
- Comprehensive QA overlays and metrics
- Caching for speed

Usage:
    # Single well
    python -m cyclops_process.processes.auto_register.iss_cycle_register --experiment ops0032_20250428 --well 1
    python -m cyclops_process.processes.auto_register.iss_cycle_register --experiment ops0090_20251120 --well 1
    # All wells
    python -m cyclops_process.processes.auto_register.iss_cycle_register \\
        --experiment ops0032_20250428 --well all
"""

import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from skimage.morphology import white_tophat, disk
from skimage.exposure import adjust_gamma
from skimage.registration import phase_cross_correlation
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading

from iohub import open_ome_zarr
import sys
import os
sys.path.insert(0, os.getcwd())
from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.data.filesystem import parse_well

# GPU-accelerated affine transforms (falls back to CPU if no GPU)
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cundi_gpu
    # Test if GPU is actually available
    _ = cp.array([1.0])
    xp = cp
    cundi = cundi_gpu
    HAS_GPU = True
except (ModuleNotFoundError, ImportError, RuntimeError):
    import numpy as xp
    from scipy import ndimage as cundi
    HAS_GPU = False

# Import from existing auto_register modules
from cyclops_process.processes.auto_register.auto_register_utils import (
    extract_spots_from_intensity_subsampled,
    extract_centroids_from_segmentation_subsampled,
    affine_3x3_to_4x4_zyx,
    save_affine_to_yaml,
)
from cyclops_process.processes.auto_register.auto_register_ransac import (
    kdtree_matching,
    estimate_affine_ransac,
)
from cyclops_process.processes.auto_register.auto_register_graph import (
    match_cells_by_graph_consistency,
)

# Default parameters (adapted from auto_register DEFAULT_PARAMS)
DEFAULT_ISS_PARAMS = {
    # Spot detection
    "spot_threshold": 1000,  # Minimum spot intensity (strict - only bright spots)
    "spot_min_distance": 5,  # Minimum distance between spots (pixels)
    "max_spots_per_bin": 50,  # Maximum spots to keep per spatial bin (top N by intensity)
    "nucleus_threshold": 200,  # Minimum nucleus intensity (fallback)
    # RANSAC
    "min_samples": 3,
    "residual_threshold": 8.0,  # For round-to-round registration (tight)
    "residual_threshold_nucleus": 25.0,  # For nucleus-to-spots (more lenient due to mismatched density)
    "max_trials": 50000,
    "stop_probability": 0.99,
    "transform_type": "similarity",  # "affine", "similarity", or "euclidean"
    # Matching
    "max_match_distance": 100.0,  # Maximum spatial search distance for round-to-round (pixels)
    # Graph matching
    "graph_k_neighbors": 8,
    "graph_top_k_candidates": 5,  # Reduced - skip Hu filtering when weights are 0
    "nucleus_search_radius": 200,  # Increased to handle larger offsets between nucleus and spots
    # Spatial subsampling (for speed)
    "use_spatial_subsample": True,
    "subsample_bins_to_select": 100,  # Number of grid bins to sample (out of 50x50 grid)
    "subsample_grid_size": 50,  # Grid size for subsampling
    # Output
    "min_cell_area": 100,  # For consistency with auto_register
    "center_fraction": 1.0,  # Use full well
}


def _extract_spots_worker(args):
    """Module-level worker for ProcessPoolExecutor spot extraction."""
    zarr_path, position, round_idx, channels, threshold, min_dist, bins, grid, max_spots = args
    from pathlib import Path
    return extract_spots_from_intensity_subsampled(
        Path(zarr_path), position, t_idx=round_idx,
        channel_indices=channels, threshold=threshold,
        min_distance=min_dist, bins_to_select=bins,
        grid_size=grid, max_spots_per_bin=max_spots,
        cache_subdir="in_situ_sequencing/register/iss_spot_cache",
    )


def create_all_rounds_overlay(
    zarr_path: Path,
    position: str,
    affines_cumulative: dict,
    output_path: Path,
    crop_size: int = 500,
    n_crops: int = 6,
    spot_channels: list = [1, 2, 3, 4],
    include_nucleus: bool = False,
    include_segmentation: bool = False,
    seg_zarr_path: Path = None,
    precomputed_crops: dict = None,
):
    """
    Create before/after overlay showing all rounds stacked together.

    Uses viridis colormap to assign different colors to different rounds.
    Shows unregistered (before) vs registered (after) for all rounds simultaneously.

    Parameters
    ----------
    include_nucleus : bool
        If True, include nucleus (Round 0, ch0) in CYAN as the global anchor.
    include_segmentation : bool
        If True, include segmentation mask in RED as the global anchor.
    seg_zarr_path : Path, optional
        Path to segmentation zarr (required if include_segmentation=True).
    """
    import time as _t

    _t0_total = _t.time()
    _t0_open = _t.time()
    with open_ome_zarr(zarr_path, mode="r", version="0.5") as store:
        data = store[position].data

        # Get dimensions without loading full images
        # Exclude nucleus (-1) - only show spot rounds (0-9)
        rounds = sorted([r for r in affines_cumulative.keys() if r >= 0])
        Y, X = data.shape[-2:]

        # --- Detect blown-out / saturated rounds (self-calibrating) ---
        # A saturated spot channel turns a round into a near-uniform bright field
        # whose background is orders of magnitude above the other rounds. Left in,
        # its per-crop percentile normalization washes the whole composite to white
        # (observed: ops0137 well 2 R0 MiSeq G/T ~125x the other rounds' background).
        # Flag a round whose background (median of the summed spot channels, sampled
        # from a center region) exceeds 4x the median across rounds, and drop it from
        # the overlay so the good rounds still render. Using the cross-round median as
        # the reference is robust to up to ~half the rounds being blown, and needs no
        # dataset-specific absolute threshold. Cast to float before summing to avoid
        # uint16 overflow on bright fields.
        _cy, _cx = Y // 2, X // 2
        _sy, _sx = max(0, _cy - 256), max(0, _cx - 256)
        _round_bg = {}
        for _r in rounds:
            _samp = np.sum(
                [np.asarray(data[_r, ch, 0, _sy:_sy + 512, _sx:_sx + 512]).astype(np.float32)
                 for ch in spot_channels],
                axis=0,
            )
            _round_bg[_r] = float(np.median(_samp))
        _bg_ref = float(np.median(list(_round_bg.values()))) if _round_bg else 0.0
        blown_rounds = {
            _r for _r, _b in _round_bg.items() if _bg_ref > 0 and _b > 4.0 * _bg_ref
        }
        if blown_rounds:
            print(
                f"    [overlay] excluding saturated rounds {sorted(blown_rounds)} "
                f"(background > 4x cross-round median {_bg_ref:.0f}): "
                + ", ".join(f"R{_r}={_round_bg[_r]:.0f}" for _r in sorted(blown_rounds)),
                flush=True,
            )
    print(f"    [overlay-profile] Open zarr + get shape: {_t.time()-_t0_open:.2f}s", flush=True)

    # Sample from inner 50% of well (avoid edges)
    margin_y = int(Y * 0.25)
    margin_x = int(X * 0.25)
    inner_height = Y - 2 * margin_y
    inner_width = X - 2 * margin_x

    # Fixed grid positions
    y_positions = [
        margin_y,
        margin_y,
        margin_y + inner_height // 2 - crop_size // 2,
        margin_y + inner_height // 2 - crop_size // 2,
        margin_y + inner_height - crop_size,
        margin_y + inner_height - crop_size
    ]
    x_positions = [
        margin_x,
        margin_x + inner_width - crop_size,
        margin_x,
        margin_x + inner_width - crop_size,
        margin_x,
        margin_x + inner_width - crop_size
    ]
    crop_positions = [(y_positions[i], x_positions[i]) for i in range(min(n_crops, len(y_positions)))]

    # Create viridis colors for each round
    cmap = plt.cm.viridis
    if len(rounds) <= 1:
        colors = [cmap(0.5)]  # Single color if only one round
    else:
        colors = [cmap(i / (len(rounds) - 1)) for i in range(len(rounds))]

    # Cyan color for nucleus (if included)
    nucleus_color = np.array([0, 1, 1])  # Cyan (R=0, G=1, B=1)

    # Pre-calculate white top-hat structural element for background removal
    selem = disk(20)

    # Create figure with square geometry (matches 2 columns of square images)
    plot_size = 4  # inches per subplot
    fig = Figure(figsize=(plot_size * 2, plot_size * n_crops * 1.05))
    axes = fig.subplots(n_crops, 2, gridspec_kw={'wspace': 0.01, 'hspace': 0.1})
    if n_crops == 1:
        axes = axes.reshape(1, -1)

    _t0_open2 = _t.time()
    with open_ome_zarr(zarr_path, mode="r", version="0.5") as store:
        data = store[position].data
        print(f"    [overlay-profile] Open zarr (2nd): {_t.time()-_t0_open2:.2f}s", flush=True)

        _t0_crops = _t.time()
        _t_load_total = 0.0
        _t_tophat_total = 0.0
        _t_normalize_total = 0.0
        _t_transform_total = 0.0
        _t_seg_total = 0.0
        _t_nucleus_total = 0.0
        _n_ops = 0

        # Pre-open segmentation zarr array once (not per-crop)
        _seg_arr_cached = None
        if include_segmentation and seg_zarr_path is not None:
            import zarr as _zarr_seg
            _seg_store = _zarr_seg.open(str(seg_zarr_path), mode='r')
            _seg_group = _seg_store[position]
            if hasattr(_seg_group, 'shape'):
                _seg_arr_cached = _seg_group
            else:
                _seg_arr_cached = _seg_group['0'] if '0' in _seg_group else _seg_group[list(_seg_group.keys())[0]]

        for idx, (y_start, x_start) in enumerate(crop_positions):
            y_end = y_start + crop_size
            x_end = x_start + crop_size

            # Create RGB composite for before and after
            overlay_before = np.zeros((crop_size, crop_size, 3), dtype=np.float32)
            overlay_after = np.zeros((crop_size, crop_size, 3), dtype=np.float32)

            # Offset matrix for crop coordinate transformation
            T_offset = np.eye(3)
            T_offset[0, 2] = y_start
            T_offset[1, 2] = x_start
            T_offset_inv = np.linalg.inv(T_offset)

            # Optionally add segmentation mask - RED, anchor (never moves)
            if include_segmentation and seg_zarr_path is not None:
                # Read crop slice directly from zarr (not full 2.8GB array)
                seg_crop = _seg_arr_cached[0, 0, 0, y_start:y_end, x_start:x_end] if _seg_arr_cached.ndim == 5 else _seg_arr_cached[y_start:y_end, x_start:x_end]

                # Convert to binary and normalize
                seg_binary = (seg_crop > 0).astype(np.float32)
                seg_frac = np.mean(seg_binary)

                # Only add if there's actually segmentation in this crop
                if seg_frac > 0.01:  # At least 1% segmented pixels
                    # Add segmentation to both before and after (it's the anchor, never moves)
                    # Use 0.4 alpha so we can see DAPI underneath
                    seg_color = np.array([1.0, 0.0, 0.0])  # RED
                    seg_rgb = (seg_binary[:, :, np.newaxis] * seg_color) * 0.4  # 40% opacity
                    overlay_before = np.maximum(overlay_before, seg_rgb)
                    overlay_after = np.maximum(overlay_after, seg_rgb)

            # Optionally add nucleus channel (Round 0, ch0) - CYAN
            if include_nucleus:
                if precomputed_crops is not None and (idx, "nucleus") in precomputed_crops:
                    nucleus_norm = precomputed_crops[(idx, "nucleus")]
                else:
                    nucleus_crop = data[0, 0, 0, y_start:y_end, x_start:x_end]
                    if hasattr(nucleus_crop, "compute"):
                        nucleus_crop = nucleus_crop.compute()
                    nucleus_crop = np.array(nucleus_crop, dtype=np.float32)
                    nucleus_filtered = white_tophat(nucleus_crop, selem)
                    bg_floor = np.percentile(nucleus_filtered, 50)
                    nucleus_subtracted = np.clip(nucleus_filtered - bg_floor, 0, None)
                    p_max = np.percentile(nucleus_subtracted, 99.9)
                    if p_max == 0:
                        p_max = 1
                    nucleus_norm = np.clip(nucleus_subtracted / p_max, 0, 1)
                    nucleus_norm = adjust_gamma(nucleus_norm, 1.5)

                # Add nucleus to before (unaligned)
                nucleus_rgb_before = nucleus_norm[:, :, np.newaxis] * nucleus_color
                overlay_before = np.maximum(overlay_before, nucleus_rgb_before)

                # For "after", transform nucleus if segmentation is anchor (not nucleus)
                # Check if segmentation exists in affines_cumulative (key -2)
                if -2 in affines_cumulative and -1 in affines_cumulative:
                    # Segmentation is anchor, so nucleus needs to be transformed
                    # affines_cumulative[-1] is T_nuc_to_seg (Global Forward)
                    affine_global = affines_cumulative[-1]
                    
                    # Convert to Local Forward (taking into account crop offset)
                    # Local_out = T_offset_inv @ Global_out
                    # Global_out = M_global @ Global_in
                    # Global_in = T_offset @ Local_in
                    # => Local_out = T_offset_inv @ M_global @ T_offset @ Local_in
                    affine_local_fwd = T_offset_inv @ affine_global @ T_offset
                    
                    # Invert for scipy (Local Output -> Local Input)
                    affine_local_inv = np.linalg.inv(affine_local_fwd)
                    
                    nucleus_aligned = ndi.affine_transform(
                        nucleus_norm, affine_local_inv[:2, :], order=1,
                        output_shape=nucleus_norm.shape, cval=0
                    )
                    nucleus_rgb_after = nucleus_aligned[:, :, np.newaxis] * nucleus_color
                else:
                    # Nucleus is the anchor, doesn't move
                    nucleus_rgb_after = nucleus_rgb_before

                overlay_after = np.maximum(overlay_after, nucleus_rgb_after)

            for round_idx, r in enumerate(rounds):
                # Skip saturated/blown-out rounds (detected globally above) so a
                # single bad channel doesn't wash the whole composite to white.
                if r in blown_rounds:
                    continue
                if precomputed_crops is not None and (idx, r) in precomputed_crops:
                    crop_norm = precomputed_crops[(idx, r)]
                else:
                    _t_ld = _t.time()
                    crop = np.sum([data[r, ch, 0, y_start:y_end, x_start:x_end] for ch in spot_channels], axis=0)
                    if hasattr(crop, "compute"):
                        crop = crop.compute()
                    crop = np.array(crop, dtype=np.float32)
                    _t_load_total += _t.time() - _t_ld

                    _t_th = _t.time()
                    crop_filtered = white_tophat(crop, selem)
                    _t_tophat_total += _t.time() - _t_th

                    _t_nm = _t.time()
                    bg_floor = np.percentile(crop_filtered, 50)
                    crop_subtracted = np.clip(crop_filtered - bg_floor, 0, None)
                    p_max = np.percentile(crop_subtracted, 99.9)
                    if p_max == 0:
                        p_max = 1
                    crop_norm = np.clip(crop_subtracted / p_max, 0, 1)
                    crop_norm = adjust_gamma(crop_norm, 1.5)
                    _t_normalize_total += _t.time() - _t_nm
                _n_ops += 1

                # Before: unregistered - use MAXIMUM projection (not sum)
                round_color = crop_norm[:, :, np.newaxis] * np.array(colors[round_idx][:3])
                overlay_before = np.maximum(overlay_before, round_color)

                # After: apply transform with crop adjustment
                affine_global = affines_cumulative[r]

                # Adjust for crop position using offset conjugation
                # Global coords = Local coords + Crop offset
                # We need: Local_out = Transform(Local_in + Crop_offset) - Crop_offset
                affine_global_3x3 = np.eye(3)
                affine_global_3x3[:2, :2] = affine_global[:2, :2]
                affine_global_3x3[:2, 2] = affine_global[:2, 2]

                # Conjugate with crop offset
                affine_crop_fwd = T_offset_inv @ affine_global_3x3 @ T_offset

                # For Round 0 (nucleus registration), use translation-only
                # For other rounds, use full affine (rotation + scale + translation)
                if r == 0:
                    # Translation-only (negate translation for scipy semantics)
                    affine_matrix = np.array([
                        [1, 0, -affine_crop_fwd[0, 2]],
                        [0, 1, -affine_crop_fwd[1, 2]]
                    ])

                    # DEBUG: Print for Round 0
                    if idx == 0:
                        print(f"\n  DEBUG Overlay Round 0 (translation-only):")
                        print(f"    affine_global: dy={affine_global[0,2]:.2f}, dx={affine_global[1,2]:.2f}")
                        print(f"    affine_crop_fwd: dy={affine_crop_fwd[0,2]:.2f}, dx={affine_crop_fwd[1,2]:.2f}")
                        print(f"    Negated shift applied: dy={-affine_crop_fwd[0,2]:.2f}, dx={-affine_crop_fwd[1,2]:.2f}")
                else:
                    # Full affine for round-to-round
                    # Properly invert the full 3x3 matrix, then extract 2x3 for scipy
                    affine_crop_inv = np.linalg.inv(affine_crop_fwd)
                    affine_matrix = affine_crop_inv[:2, :]

                # Apply transform directly
                _t_xf = _t.time()
                crop_aligned = ndi.affine_transform(
                    crop_norm,
                    affine_matrix,
                    order=1,
                    output_shape=crop_norm.shape
                )

                _t_transform_total += _t.time() - _t_xf
                _n_ops += 1

                # After: use MAXIMUM projection (not sum)
                aligned_color = crop_aligned[:, :, np.newaxis] * np.array(colors[round_idx][:3])
                overlay_after = np.maximum(overlay_after, aligned_color)

            # Display
            axes[idx, 0].imshow(overlay_before)
            axes[idx, 0].set_title(f"Region {idx+1}: Before", fontsize=10)
            axes[idx, 0].axis('off')
            axes[idx, 0].set_aspect('equal', 'box')

            axes[idx, 1].imshow(overlay_after)
            axes[idx, 1].set_title(f"Region {idx+1}: After (Registered)", fontsize=10)
            axes[idx, 1].axis('off')
            axes[idx, 1].set_aspect('equal', 'box')

    print(f"    [overlay-profile] Processing {_n_ops} round×crop ops: "
          f"load={_t_load_total:.2f}s tophat={_t_tophat_total:.2f}s "
          f"normalize={_t_normalize_total:.2f}s transform={_t_transform_total:.2f}s "
          f"crops_total={_t.time()-_t0_crops:.2f}s", flush=True)

    _t0_save = _t.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"    [overlay-profile] Figure save: {_t.time()-_t0_save:.2f}s", flush=True)
    print(f"    [overlay-profile] TOTAL this overlay: {_t.time()-_t0_total:.2f}s", flush=True)


def create_drift_trajectory_plot(
    affines_cumulative: dict,
    output_path: Path,
):
    """
    Create drift trajectory plot showing translation relative to Round 0 spots.

    Round 0 spots are at (0,0). Shows drift forward through ISS rounds and
    backward to nucleus and segmentation.
    """
    # Separate segmentation/nucleus offsets from round offsets
    seg_keys = sorted([k for k in affines_cumulative.keys() if k < 0])
    round_keys = sorted([k for k in affines_cumulative.keys() if k >= 0])

    # Get Round 0 translation as reference
    round0_trans = affines_cumulative[0][:2, 2] if 0 in affines_cumulative else np.zeros(2)

    # Calculate translations relative to Round 0
    round_translations = np.array([affines_cumulative[r][:2, 2] - round0_trans for r in round_keys])
    dy_rounds = round_translations[:, 0]
    dx_rounds = round_translations[:, 1]

    # Build complete series including backward drift to nucleus/seg
    all_keys = seg_keys + round_keys
    all_translations = []
    all_labels = []

    for k in all_keys:
        trans = affines_cumulative[k][:2, 2] - round0_trans
        all_translations.append(trans)
        if k == -2:
            all_labels.append('Seg')
        elif k == -1:
            all_labels.append('Nuc')
        else:
            all_labels.append(f'R{k}')

    all_translations = np.array(all_translations)
    dy_all = all_translations[:, 0]
    dx_all = all_translations[:, 1]

    fig = Figure(figsize=(14, 6)); axes = fig.subplots(1, 2)

    # Left: Translation components - show full trajectory including backward
    axes[0].plot(all_keys, dy_all, 'o-', label='Y drift', linewidth=2, markersize=8)
    axes[0].plot(all_keys, dx_all, 's-', label='X drift', linewidth=2, markersize=8)

    # Mark Round 0 reference
    round0_idx = all_keys.index(0)
    axes[0].plot(0, 0, 'g*', markersize=15, label='Round 0 (reference)', zorder=10)

    axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0].set_xlabel('Round (negative = pre-ISS)', fontsize=12)
    axes[0].set_ylabel('Translation from Round 0 (pixels)', fontsize=12)
    axes[0].set_title('Drift Relative to Round 0 Spots', fontsize=14)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Right: 2D trajectory
    axes[1].plot(dx_all, dy_all, 'o-', linewidth=2, markersize=8, color='tab:blue', alpha=0.7)

    # Mark key points
    axes[1].plot(0, 0, 'g*', markersize=15, label='Round 0 (reference)', zorder=10)
    axes[1].plot(dx_rounds[-1], dy_rounds[-1], 'rs', markersize=12, label=f'Round {round_keys[-1]}', zorder=9)

    # Mark segmentation and nucleus if present
    if seg_keys:
        for sk in seg_keys:
            idx = all_keys.index(sk)
            label = 'Segmentation' if sk == -2 else 'Nucleus'
            marker = 'r^' if sk == -2 else 'c^'
            axes[1].plot(dx_all[idx], dy_all[idx], marker, markersize=12, label=label, zorder=9)

    # Annotate all points
    for i, (k, label) in enumerate(zip(all_keys, all_labels)):
        axes[1].annotate(label, (dx_all[i], dy_all[i]), xytext=(5, 5),
                        textcoords='offset points', fontsize=9)

    axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[1].axvline(0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_xlabel('X Translation from R0 (pixels)', fontsize=12)
    axes[1].set_ylabel('Y Translation from R0 (pixels)', fontsize=12)
    axes[1].set_title('2D Drift Trajectory (R0 = origin)', fontsize=14)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].axis('equal')

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_spot_match_visualization(
    zarr_path: Path,
    position: str,
    round_source: int,
    round_target: int,
    source_spots: np.ndarray,
    target_spots: np.ndarray,
    source_idx: np.ndarray,
    target_idx: np.ndarray,
    affine_3x3: np.ndarray,
    output_path: Path,
    max_matches_to_show: int = 200,
):
    """
    Create visualization showing spot matches between rounds.

    Shows matched spots connected by lines, with colors indicating match quality.
    """
    # Subsample matches for clarity
    n_matches = len(source_idx)
    if n_matches > max_matches_to_show:
        show_idx = np.random.choice(n_matches, max_matches_to_show, replace=False)
        src_idx_show = source_idx[show_idx]
        tgt_idx_show = target_idx[show_idx]
    else:
        src_idx_show = source_idx
        tgt_idx_show = target_idx

    matched_src = source_spots[src_idx_show]
    matched_tgt = target_spots[tgt_idx_show]

    # Apply transform to source
    src_homog = np.column_stack([matched_src, np.ones(len(matched_src))])
    src_aligned = (affine_3x3 @ src_homog.T).T[:, :2]

    # Calculate distances after alignment
    distances = np.linalg.norm(src_aligned - matched_tgt, axis=1)

    fig = Figure(figsize=(16, 8)); axes = fig.subplots(1, 2)

    # Left: Before alignment
    axes[0].scatter(matched_tgt[:, 1], matched_tgt[:, 0], c='red', s=20, alpha=0.6, label='Target')
    axes[0].scatter(matched_src[:, 1], matched_src[:, 0], c='green', s=20, alpha=0.6, label='Source')

    # Draw lines connecting matches
    for i in range(len(matched_src)):
        axes[0].plot([matched_src[i, 1], matched_tgt[i, 1]],
                    [matched_src[i, 0], matched_tgt[i, 0]],
                    'b-', alpha=0.1, linewidth=0.5)

    axes[0].set_xlabel('X (pixels)', fontsize=12)
    axes[0].set_ylabel('Y (pixels)', fontsize=12)
    axes[0].set_title(f'Before: Round {round_source} → Round {round_target}\n({n_matches} matches, showing {len(matched_src)})', fontsize=12)
    axes[0].legend()
    axes[0].invert_yaxis()
    axes[0].set_aspect('equal')

    # Right: After alignment (colored by residual)
    scatter = axes[1].scatter(matched_tgt[:, 1], matched_tgt[:, 0], c='red', s=20, alpha=0.6, label='Target')
    axes[1].scatter(src_aligned[:, 1], src_aligned[:, 0], c=distances, cmap='RdYlGn_r',
                   s=20, alpha=0.6, vmin=0, vmax=np.percentile(distances, 95))

    # Draw lines colored by distance
    for i in range(len(src_aligned)):
        color = plt.cm.RdYlGn_r(distances[i] / np.percentile(distances, 95))
        axes[1].plot([src_aligned[i, 1], matched_tgt[i, 1]],
                    [src_aligned[i, 0], matched_tgt[i, 0]],
                    color=color, alpha=0.3, linewidth=0.8)

    axes[1].set_xlabel('X (pixels)', fontsize=12)
    axes[1].set_ylabel('Y (pixels)', fontsize=12)
    axes[1].set_title(f'After: Aligned\nMean residual: {np.mean(distances):.2f}px', fontsize=12)
    axes[1].invert_yaxis()
    axes[1].set_aspect('equal')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(vmin=0, vmax=np.percentile(distances, 95)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[1])
    cbar.set_label('Residual (pixels)', fontsize=10)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_spot_extraction_visualization(
    zarr_path: Path,
    position: str,
    round_num: int,
    extracted_spots: np.ndarray,
    output_path: Path,
    crop_size: int = 500,
    spot_channels: list = [1, 2, 3, 4],
):
    """
    Create visualization showing extracted spot locations overlaid on raw image.

    Shows a single 500x500 crop with the raw image and extracted spots marked.
    """
    with open_ome_zarr(zarr_path, mode="r", version="0.5") as store:
        data = store[position].data

        # Load and sum spot channels
        spots_img = np.sum([data[round_num, ch, 0, :, :] for ch in spot_channels], axis=0)

        # Compute if dask
        if hasattr(spots_img, "compute"):
            spots_img = spots_img.compute()

    Y, X = spots_img.shape

    if len(extracted_spots) == 0:
        print(f"      WARNING: No extracted spots for Round {round_num}")
        return

    # Pick a random spot and center a 500x500 crop on it
    # This guarantees we show a region with at least one spot
    random_idx = np.random.randint(0, len(extracted_spots))
    center_y = int(extracted_spots[random_idx, 0])
    center_x = int(extracted_spots[random_idx, 1])

    y_start = max(0, min(Y - crop_size, center_y - crop_size // 2))
    x_start = max(0, min(X - crop_size, center_x - crop_size // 2))
    y_end = y_start + crop_size
    x_end = x_start + crop_size

    # Extract crop
    img_crop = spots_img[y_start:y_end, x_start:x_end]

    # Find all spots within this crop region
    spot_mask = (
        (extracted_spots[:, 0] >= y_start) &
        (extracted_spots[:, 0] < y_end) &
        (extracted_spots[:, 1] >= x_start) &
        (extracted_spots[:, 1] < x_end)
    )
    spots_in_crop = extracted_spots[spot_mask]
    spots_in_crop_local = spots_in_crop - np.array([y_start, x_start])

    # Normalize image with strong background subtraction
    p30 = np.percentile(img_crop, 30)
    p99 = np.percentile(img_crop, 99)
    img_clipped = np.clip(img_crop, p30, p99)
    img_norm = (img_clipped - img_clipped.min()) / (img_clipped.max() - img_clipped.min() + 1e-10)

    # Create figure
    fig = Figure(figsize=(10, 10)); ax = fig.subplots(1, 1)

    # Show image in grayscale
    ax.imshow(img_norm, cmap='gray', vmin=0, vmax=1)

    # Overlay spots as yellow circles with no fill (just border)
    ax.scatter(spots_in_crop_local[:, 1], spots_in_crop_local[:, 0],
               facecolors='none', s=60, edgecolors='yellow', linewidths=1,
               marker='o', label=f'{len(spots_in_crop)} detected spots')

    ax.set_title(f'Round {round_num}: Extracted Spots Overlay\n{len(spots_in_crop)} spots in this 500x500px region',
                 fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.axis('off')

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        plt.close(fig)
        print(f"      WARNING: Failed to save spot extraction viz: {e}")


def create_nucleus_to_spots_overlay(
    zarr_path: Path,
    position: str,
    affine_spots_to_nuc: np.ndarray,
    output_path: Path,
    crop_size: int = 500,
    n_crops: int = 6,
):
    """
    Create before/after overlay showing spots → nucleus registration (both from Round 0).

    Nucleus is the global anchor (green channel, never moves).
    Spots (red channel) are warped to align with nucleus.

    Colors: Nucleus=Green (reference), Spots=Red (before/after alignment), Yellow=Overlap
    """
    with open_ome_zarr(zarr_path, mode="r", version="0.5") as store:
        data = store[position].data

        # Get image dimensions without loading data
        Y, X = data.shape[-2:]

    # Sample from inner 50% of well (avoid edges)
    margin_y = int(Y * 0.25)
    margin_x = int(X * 0.25)
    inner_height = Y - 2 * margin_y
    inner_width = X - 2 * margin_x

    # Fixed grid positions: top-left, top-right, center-left, center-right, bottom-left, bottom-right
    y_positions = [
        margin_y,                                      # Top
        margin_y,                                      # Top
        margin_y + inner_height // 2 - crop_size // 2, # Center
        margin_y + inner_height // 2 - crop_size // 2, # Center
        margin_y + inner_height - crop_size,           # Bottom
        margin_y + inner_height - crop_size            # Bottom
    ]
    x_positions = [
        margin_x,                                      # Left
        margin_x + inner_width - crop_size,            # Right
        margin_x,                                      # Left
        margin_x + inner_width - crop_size,            # Right
        margin_x,                                      # Left
        margin_x + inner_width - crop_size             # Right
    ]
    crop_positions = [(y_positions[i], x_positions[i]) for i in range(min(n_crops, len(y_positions)))]

    # Create figure with square geometry (matches 2 columns of square images)
    plot_size = 4  # inches per subplot
    fig = Figure(figsize=(plot_size * 2, plot_size * n_crops * 1.05))
    axes = fig.subplots(n_crops, 2, gridspec_kw={'wspace': 0.01, 'hspace': 0.1})
    if n_crops == 1:
        axes = axes.reshape(1, -1)

    # Load crops on-demand from Zarr (more memory efficient)
    with open_ome_zarr(zarr_path, mode="r", version="0.5") as store:
        data = store[position].data

        for idx, (y_start, x_start) in enumerate(crop_positions):
            y_end = y_start + crop_size
            x_end = x_start + crop_size

            # Load only the crop region from Zarr (avoid loading full image)
            nucleus_crop = data[0, 0, 0, y_start:y_end, x_start:x_end]
            spots_crop = np.sum([data[0, ch, 0, y_start:y_end, x_start:x_end] for ch in [1, 2, 3, 4]], axis=0)

            # Compute if dask
            if hasattr(nucleus_crop, "compute"):
                nucleus_crop = nucleus_crop.compute()
            if hasattr(spots_crop, "compute"):
                spots_crop = spots_crop.compute()

            # Cast to float32
            nucleus_crop = np.array(nucleus_crop, dtype=np.float32)
            spots_crop = np.array(spots_crop, dtype=np.float32)

            # High contrast normalization: clip p50→p99.5, then gamma>1
            nucleus_p50 = np.percentile(nucleus_crop, 50)
            nucleus_p99 = np.percentile(nucleus_crop, 99.5)
            nucleus_norm = np.clip((nucleus_crop - nucleus_p50) / (nucleus_p99 - nucleus_p50 + 1e-10), 0, 1)

            spots_p50 = np.percentile(spots_crop, 50)
            spots_p99 = np.percentile(spots_crop, 99.5)
            spots_norm = np.clip((spots_crop - spots_p50) / (spots_p99 - spots_p50 + 1e-10), 0, 1)

            # Apply Gamma > 1 to crush remaining background
            nucleus_norm = adjust_gamma(nucleus_norm, 1.5)
            spots_norm = adjust_gamma(spots_norm, 1.5)

            # Apply transform to SPOTS (not nucleus)
            # affine_spots_to_nuc contains the spots→nucleus transform
            from scipy.ndimage import affine_transform as scipy_affine_transform

            # affine_spots_to_nuc is spots→nucleus, but scipy needs nucleus→spots (inverse mapping)
            # So we invert it back to get nucleus→spots for scipy
            affine_nuc_to_spots = np.linalg.inv(affine_spots_to_nuc)

            # Extract translation (nucleus→spots)
            shift_y = affine_nuc_to_spots[0, 2]
            shift_x = affine_nuc_to_spots[1, 2]

            # Create matrix for scipy affine_transform (already in correct direction)
            affine_matrix = np.array([
                [1, 0, shift_y],  # No negation needed - already correct direction
                [0, 1, shift_x]   # No negation needed - already correct direction
            ])

            # Apply transform
            spots_aligned = scipy_affine_transform(
                spots_norm,
                affine_matrix,
                order=1,
                mode='constant',
                cval=0.0
            )

            # Create RGB overlays
            # Before: spots (red) vs nucleus (green) - misaligned
            overlay_before = np.zeros((*nucleus_norm.shape, 3), dtype=np.float32)
            overlay_before[..., 0] = spots_norm     # Red: spots (misaligned)
            overlay_before[..., 1] = nucleus_norm   # Green: nucleus (reference)

            # After: spots aligned to nucleus
            overlay_after = np.zeros((*nucleus_norm.shape, 3), dtype=np.float32)
            overlay_after[..., 0] = spots_aligned   # Red: spots (aligned)
            overlay_after[..., 1] = nucleus_norm    # Green: nucleus (reference)

            # Display
            axes[idx, 0].imshow(overlay_before)
            axes[idx, 0].set_title(f"Region {idx+1}: Before", fontsize=10)
            axes[idx, 0].axis('off')
            axes[idx, 0].set_aspect('equal', 'box')

            axes[idx, 1].imshow(overlay_after)
            axes[idx, 1].set_title(f"Region {idx+1}: After (Aligned)", fontsize=10)
            axes[idx, 1].axis('off')
            axes[idx, 1].set_aspect('equal', 'box')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def create_iss_registration_overlay(
    zarr_path: Path,
    position: str,
    round_source: int,
    round_target: int,
    affine_3x3: np.ndarray,
    output_path: Path,
    crop_size: int = 500,
    n_crops: int = 6,
    spot_channels: list = [1, 2, 3, 4],
):
    """
    Create before/after overlay showing ISS round registration correction.

    Shows a grid of cropped regions (500x500px) sampled across the well at full resolution.
    Each region shows before/after registration side-by-side.

    Colors: Target=Red, Source=Green, Overlap=Yellow
    """
    with open_ome_zarr(zarr_path, mode="r", version="0.5") as store:
        data = store[position].data

        # Get image dimensions without loading data
        Y, X = data.shape[-2:]

    # Sample from inner 50% of well (avoid edges)
    margin_y, margin_x = int(Y * 0.25), int(X * 0.25)
    inner_h, inner_w = Y - 2 * margin_y, X - 2 * margin_x

    y_pos = [margin_y, margin_y, margin_y + inner_h//2, margin_y + inner_h//2, margin_y + inner_h - crop_size, margin_y + inner_h - crop_size]
    x_pos = [margin_x, margin_x + inner_w - crop_size, margin_x, margin_x + inner_w - crop_size, margin_x, margin_x + inner_w - crop_size]
    crop_positions = list(zip(y_pos, x_pos))[:n_crops]

    # Create figure with square geometry (matches 2 columns of square images)
    plot_size = 4  # inches per subplot
    fig = Figure(figsize=(plot_size * 2, plot_size * n_crops * 1.05))
    axes = fig.subplots(n_crops, 2, gridspec_kw={'wspace': 0.01, 'hspace': 0.1})
    if n_crops == 1:
        axes = axes.reshape(1, -1)

    # Use larger TopHat selem
    selem = disk(20)

    # High contrast normalization helper
    def high_contrast_norm(img, selem):
        # 1. TopHat filter
        filtered = white_tophat(img, selem)
        # 2. Subtract median background
        bg_floor = np.percentile(filtered, 50)
        subtracted = np.clip(filtered - bg_floor, 0, None)
        # 3. Normalize to max
        p_max = np.percentile(subtracted, 99.9) or 1
        norm = np.clip(subtracted / p_max, 0, 1)
        # 4. Gamma > 1 to crush blacks
        return adjust_gamma(norm, 1.5)

    # Load crops on-demand from Zarr (more memory efficient)
    with open_ome_zarr(zarr_path, mode="r", version="0.5") as store:
        data = store[position].data

        for idx, (y_start, x_start) in enumerate(crop_positions):
            y_end = y_start + crop_size
            x_end = x_start + crop_size

            # Load only the crop region from Zarr (avoid loading full image)
            src_crop = np.sum([data[round_source, ch, 0, y_start:y_end, x_start:x_end] for ch in spot_channels], axis=0)
            tgt_crop = np.sum([data[round_target, ch, 0, y_start:y_end, x_start:x_end] for ch in spot_channels], axis=0)

            # Compute if dask
            if hasattr(src_crop, "compute"):
                src_crop = src_crop.compute()
            if hasattr(tgt_crop, "compute"):
                tgt_crop = tgt_crop.compute()

            # Cast to float32
            src_crop = np.array(src_crop, dtype=np.float32)
            tgt_crop = np.array(tgt_crop, dtype=np.float32)

            # Apply high contrast normalization
            src_norm = high_contrast_norm(src_crop, selem)
            tgt_norm = high_contrast_norm(tgt_crop, selem)

            # Transform Source
            T_offset = np.eye(3)
            T_offset[0, 2], T_offset[1, 2] = y_start, x_start

            affine_crop_fwd = np.linalg.inv(T_offset) @ affine_3x3 @ T_offset
            affine_crop_inv = np.linalg.inv(affine_crop_fwd)

            src_aligned = ndi.affine_transform(
                src_norm,
                affine_crop_inv[:2, :2],
                offset=affine_crop_inv[:2, 2],
                order=1,
                output_shape=src_norm.shape
            )

            # Create Overlays
            # Red = Target, Green = Source
            overlay_before = np.zeros((*tgt_norm.shape, 3), dtype=np.float32)
            overlay_before[..., 0] = tgt_norm
            overlay_before[..., 1] = src_norm

            overlay_after = np.zeros((*tgt_norm.shape, 3), dtype=np.float32)
            overlay_after[..., 0] = tgt_norm
            overlay_after[..., 1] = src_aligned

            axes[idx, 0].imshow(overlay_before)
            axes[idx, 0].set_title(f"Region {idx+1}: Before", fontsize=10)
            axes[idx, 0].axis('off')
            axes[idx, 0].set_aspect('equal', 'box')

            axes[idx, 1].imshow(overlay_after)
            axes[idx, 1].set_title(f"Region {idx+1}: After", fontsize=10)
            axes[idx, 1].axis('off')
            axes[idx, 1].set_aspect('equal', 'box')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def create_iss_registration_overlay_custom_zarrs(
    source_zarr: Path,
    target_zarr: Path,
    position: str,
    source_t_idx: int,
    target_t_idx: int,
    affine_3x3: np.ndarray,
    output_path: Path,
    crop_size: int = 500,
    n_crops: int = 6,
    spot_channels: list = [1, 2, 3, 4],
):
    """
    Create before/after overlay for registration between two different zarr stores.

    Used for DAPI_round10 → Round 9 registration where source and target are in different zarrs.

    Parameters
    ----------
    source_zarr : Path
        Source zarr (e.g., DAPI_round10_stitched.zarr)
    target_zarr : Path
        Target zarr (e.g., bc_stitched.zarr)
    source_t_idx : int
        Time index in source zarr
    target_t_idx : int
        Time index in target zarr
    affine_3x3 : np.ndarray
        3x3 affine transform (source → target)
    """
    # Get dimensions from target zarr
    with open_ome_zarr(target_zarr, mode="r") as store:
        data = store[position].data
        Y, X = data.shape[-2:]

    # Sample from inner 50% of well (avoid edges)
    margin_y, margin_x = int(Y * 0.25), int(X * 0.25)
    inner_h, inner_w = Y - 2 * margin_y, X - 2 * margin_x

    y_pos = [margin_y, margin_y, margin_y + inner_h//2, margin_y + inner_h//2, margin_y + inner_h - crop_size, margin_y + inner_h - crop_size]
    x_pos = [margin_x, margin_x + inner_w - crop_size, margin_x, margin_x + inner_w - crop_size, margin_x, margin_x + inner_w - crop_size]
    crop_positions = list(zip(y_pos, x_pos))[:n_crops]

    # Create figure
    plot_size = 4
    fig = Figure(figsize=(plot_size * 2, plot_size * n_crops * 1.05))
    axes = fig.subplots(n_crops, 2, gridspec_kw={'wspace': 0.01, 'hspace': 0.1})
    if n_crops == 1:
        axes = axes.reshape(1, -1)

    selem = disk(20)

    def high_contrast_norm(img, selem):
        filtered = white_tophat(img, selem)
        bg_floor = np.percentile(filtered, 50)
        subtracted = np.clip(filtered - bg_floor, 0, None)
        p_max = np.percentile(subtracted, 99.9) or 1
        norm = np.clip(subtracted / p_max, 0, 1)
        return adjust_gamma(norm, 1.5)

    # Load crops from both zarrs
    with open_ome_zarr(source_zarr, mode="r") as src_store, \
         open_ome_zarr(target_zarr, mode="r") as tgt_store:

        src_data = src_store[position].data
        tgt_data = tgt_store[position].data

        for idx, (y_start, x_start) in enumerate(crop_positions):
            y_end = y_start + crop_size
            x_end = x_start + crop_size

            # Load crops - handle different zarr structures
            # Source might be single-round (no time dimension)
            if len(src_data.shape) == 5:
                # Has time dimension: (T, C, Z, Y, X)
                src_crop = np.sum([src_data[source_t_idx, ch, 0, y_start:y_end, x_start:x_end] for ch in spot_channels], axis=0)
            elif len(src_data.shape) == 4:
                # No time dimension: (C, Z, Y, X)
                src_crop = np.sum([src_data[ch, 0, y_start:y_end, x_start:x_end] for ch in spot_channels], axis=0)
            else:
                raise ValueError(f"Unexpected source zarr shape: {src_data.shape}")

            tgt_crop = np.sum([tgt_data[target_t_idx, ch, 0, y_start:y_end, x_start:x_end] for ch in spot_channels], axis=0)

            # Compute if dask
            if hasattr(src_crop, "compute"):
                src_crop = src_crop.compute()
            if hasattr(tgt_crop, "compute"):
                tgt_crop = tgt_crop.compute()

            src_crop = np.array(src_crop, dtype=np.float32)
            tgt_crop = np.array(tgt_crop, dtype=np.float32)

            # Normalize
            src_norm = high_contrast_norm(src_crop, selem)
            tgt_norm = high_contrast_norm(tgt_crop, selem)

            # Transform source
            T_offset = np.eye(3)
            T_offset[0, 2], T_offset[1, 2] = y_start, x_start

            affine_crop_fwd = np.linalg.inv(T_offset) @ affine_3x3 @ T_offset
            affine_crop_inv = np.linalg.inv(affine_crop_fwd)

            src_aligned = ndi.affine_transform(
                src_norm,
                affine_crop_inv[:2, :2],
                offset=affine_crop_inv[:2, 2],
                order=1,
                output_shape=src_norm.shape
            )

            # Create overlays (Red=Target, Green=Source)
            overlay_before = np.zeros((*tgt_norm.shape, 3), dtype=np.float32)
            overlay_before[..., 0] = tgt_norm
            overlay_before[..., 1] = src_norm

            overlay_after = np.zeros((*tgt_norm.shape, 3), dtype=np.float32)
            overlay_after[..., 0] = tgt_norm
            overlay_after[..., 1] = src_aligned

            axes[idx, 0].imshow(overlay_before)
            axes[idx, 0].set_title(f"Region {idx+1}: Before", fontsize=10)
            axes[idx, 0].axis('off')
            axes[idx, 0].set_aspect('equal', 'box')

            axes[idx, 1].imshow(overlay_after)
            axes[idx, 1].set_title(f"Region {idx+1}: After", fontsize=10)
            axes[idx, 1].axis('off')
            axes[idx, 1].set_aspect('equal', 'box')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def compute_overlap_metrics(
    zarr_path: Path,
    position: str,
    round_source: int,
    round_target: int,
    affine_3x3: np.ndarray,
    crop_size: int = 1000,
    spot_channels: list = [1, 2, 3, 4],
) -> dict:
    """
    Compute overlap percentage before/after registration for both direct and inverse transforms.

    Uses a single 1k x 1k crop from the center of the well.
    Overlap is computed as the percentage of pixels where both images have non-zero values
    after thresholding (using median as threshold).

    Parameters
    ----------
    zarr_path : Path
        Path to zarr store
    position : str
        Position string
    round_source : int
        Source round index
    round_target : int
        Target round index
    affine_3x3 : np.ndarray
        3x3 affine transformation matrix (source → target)
    crop_size : int
        Size of crop to use (default: 1000)
    spot_channels : list
        Channels to sum for spot signal

    Returns
    -------
    dict
        Metrics including before/after overlap for direct and inverse transforms
    """
    # Load images
    with open_ome_zarr(zarr_path, mode="r", version="0.5") as store:
        data = store[position].data

        # Sum spot channels
        src_img = np.sum([data[round_source, ch, 0, :, :] for ch in spot_channels], axis=0)
        tgt_img = np.sum([data[round_target, ch, 0, :, :] for ch in spot_channels], axis=0)

        # Compute if dask
        if hasattr(src_img, "compute"):
            src_img = src_img.compute()
        if hasattr(tgt_img, "compute"):
            tgt_img = tgt_img.compute()

    Y, X = src_img.shape

    # Extract center crop
    y_start = (Y - crop_size) // 2
    x_start = (X - crop_size) // 2
    y_end = y_start + crop_size
    x_end = x_start + crop_size

    src_crop = src_img[y_start:y_end, x_start:x_end]
    tgt_crop = tgt_img[y_start:y_end, x_start:x_end]

    # Convert to float32 and move to GPU if available
    src_crop = xp.asarray(src_crop, dtype=xp.float32)
    tgt_crop = xp.asarray(tgt_crop, dtype=xp.float32)

    # Normalize to [0, 1] using percentile clipping (removes background)
    src_p50 = xp.percentile(src_crop, 50)
    src_p99 = xp.percentile(src_crop, 99.5)
    tgt_p50 = xp.percentile(tgt_crop, 50)
    tgt_p99 = xp.percentile(tgt_crop, 99.5)

    src_norm = xp.clip((src_crop - src_p50) / (src_p99 - src_p50 + 1e-10), 0, 1)
    tgt_norm = xp.clip((tgt_crop - tgt_p50) / (tgt_p99 - tgt_p50 + 1e-10), 0, 1)

    # Compute overlap BEFORE alignment using element-wise minimum (intersection-over-union)
    overlap_before = float(xp.sum(xp.minimum(src_norm, tgt_norm)) / xp.sum(xp.maximum(src_norm, tgt_norm)))

    # ===== DIRECT: Apply affine to source (source → target) =====
    # Adjust affine for crop offset using conjugation
    T_offset = np.eye(3)
    T_offset[0, 2] = y_start
    T_offset[1, 2] = x_start
    affine_crop = np.linalg.inv(T_offset) @ affine_3x3 @ T_offset

    # Compute inverse on CPU (more stable than GPU cuSOLVER)
    # NOTE: scipy/cupyx ndimage.affine_transform uses PULL semantics:
    #   out[x] = input[matrix @ x + offset]
    # So we need the INVERSE of the forward transform
    affine_inv = np.linalg.inv(affine_crop)

    # Convert to GPU array if needed
    affine_inv_mat = xp.asarray(affine_inv[:2, :2], dtype=xp.float32)
    affine_offset = xp.asarray(affine_inv[:2, 2], dtype=xp.float32)

    # Transform source using GPU-accelerated function
    src_aligned = cundi.affine_transform(
        src_norm,
        affine_inv_mat,
        offset=affine_offset,
        order=1,
        output_shape=src_norm.shape,
    )

    # Compute overlap AFTER alignment (direct)
    direct_overlap_after = float(xp.sum(xp.minimum(src_aligned, tgt_norm)) / xp.sum(xp.maximum(src_aligned, tgt_norm)))
    direct_improvement = direct_overlap_after - overlap_before

    # ===== INVERSE: Apply forward affine to target (target → source) =====
    # For inverse direction, we use affine_3x3 directly (no inversion)
    # because ndimage.affine_transform already expects the inverse
    # and affine_3x3 is source→target, so we want target→source = use it directly in pull mode
    # Adjust using conjugation
    T_offset_inv = np.eye(3)
    T_offset_inv[0, 2] = y_start
    T_offset_inv[1, 2] = x_start
    affine_forward_crop = np.linalg.inv(T_offset_inv) @ affine_3x3 @ T_offset_inv

    # Convert to GPU array if needed
    affine_fwd_mat = xp.asarray(affine_forward_crop[:2, :2], dtype=xp.float32)
    affine_fwd_offset = xp.asarray(affine_forward_crop[:2, 2], dtype=xp.float32)

    # Transform target using GPU-accelerated function
    tgt_aligned = cundi.affine_transform(
        tgt_norm,
        affine_fwd_mat,
        offset=affine_fwd_offset,
        order=1,
        output_shape=tgt_norm.shape,
    )

    # Compute overlap AFTER alignment (inverse)
    inverse_overlap_after = float(xp.sum(xp.minimum(src_norm, tgt_aligned)) / xp.sum(xp.maximum(src_norm, tgt_aligned)))
    inverse_improvement = inverse_overlap_after - overlap_before

    return {
        "direct_before": overlap_before,
        "direct_after": direct_overlap_after,
        "direct_improvement": direct_improvement,
        "inverse_before": overlap_before,
        "inverse_after": inverse_overlap_after,
        "inverse_improvement": inverse_improvement,
    }


def register_round_pair(
    zarr_path: Path,
    position: str,
    round_i: int,
    round_j: int,
    params: dict,
    transforms_dir: Path,
    overlays_dir: Path,
    verbose: bool = True,
    compute_metrics: bool = False,
    precomputed_source_spots: np.ndarray = None,
    precomputed_target_spots: np.ndarray = None,
    generate_overlays: bool = False,
) -> dict:
    """
    Register Round j → Round i using spot-based matching.

    Parameters
    ----------
    zarr_path : Path
        Path to stitched ISS zarr (e.g., bc_stitched.zarr).
    position : str
        Position string (e.g., "A/1/0").
    round_i : int
        Reference round index.
    round_j : int
        Moving round index (to be registered to round_i).
    params : dict
        Registration parameters (DEFAULT_ISS_PARAMS).
    transforms_dir : Path
        Output directory for affine YAML files.
    overlays_dir : Path
        Output directory for visualization overlays.
    verbose : bool
        Print progress messages.
    compute_metrics : bool
        Whether to compute overlap metrics (slow). Default False.
    generate_overlays : bool
        Whether to generate overlay images. Default False.

    Returns
    -------
    dict
        Results including affine_3x3, metrics, and output paths.
    """
    if verbose:
        print(f"  Registering Round {round_j} → Round {round_i}...")

    t_start = time.time()

    # Use precomputed spots if available (from parallel pre-extraction)
    if precomputed_source_spots is not None:
        source_spots = precomputed_source_spots
        if verbose:
            print(f"    Round {round_j} (source): {len(source_spots)} spots (precomputed)")
    else:
        if verbose:
            print(f"    Extracting spots from Round {round_j} (source)...")
        t_extract_src = time.time()
        source_spots = extract_spots_from_intensity_subsampled(
            zarr_path, position, t_idx=round_j,
            channel_indices=[1, 2, 3, 4],
            threshold=params["spot_threshold"],
            min_distance=params["spot_min_distance"],
            bins_to_select=params["subsample_bins_to_select"],
            grid_size=params["subsample_grid_size"],
            max_spots_per_bin=params["max_spots_per_bin"],
            cache_subdir="in_situ_sequencing/register/iss_spot_cache",
        )
        if verbose:
            print(f"      Found {len(source_spots)} spots ({time.time()-t_extract_src:.2f}s)")

    if precomputed_target_spots is not None:
        target_spots = precomputed_target_spots
        if verbose:
            print(f"    Round {round_i} (target): {len(target_spots)} spots (precomputed)")
    else:
        if verbose:
            print(f"    Extracting spots from Round {round_i} (target)...")
        t_extract_tgt = time.time()
        target_spots = extract_spots_from_intensity_subsampled(
            zarr_path, position, t_idx=round_i,
            channel_indices=[1, 2, 3, 4],
            threshold=params["spot_threshold"],
            min_distance=params["spot_min_distance"],
            bins_to_select=params["subsample_bins_to_select"],
            grid_size=params["subsample_grid_size"],
            max_spots_per_bin=params["max_spots_per_bin"],
            cache_subdir="in_situ_sequencing/register/iss_spot_cache",
        )
        if verbose:
            print(f"      Found {len(target_spots)} spots ({time.time()-t_extract_tgt:.2f}s)")

    # Validate that we have enough spots for registration
    min_spots_required = params["graph_k_neighbors"] + 1  # Need at least k+1 for k-NN
    if len(source_spots) < min_spots_required:
        error_msg = f"Insufficient source spots: {len(source_spots)} < {min_spots_required} required"
        if verbose:
            print(f"    ✗ SKIPPING: {error_msg}")
            print(f"    Creating identity transform to maintain cumulative chain...")

        # Create identity transform (no change) and save it
        identity_3x3 = np.eye(3)
        identity_4x4 = affine_3x3_to_4x4_zyx(identity_3x3)
        identity_4x4_inv = np.linalg.inv(identity_4x4)
        output_yaml = transforms_dir / f"round{round_j}_to_round{round_i}.yml"
        save_affine_to_yaml(identity_4x4_inv, output_yaml)

        if verbose:
            print(f"    ✓ Saved identity transform: {output_yaml.name}")

        return {
            "success": False,
            "error": error_msg,
            "n_matches": 0,
            "affine_3x3": identity_3x3,
            "transform_yaml": str(output_yaml),
        }

    if len(target_spots) < min_spots_required:
        error_msg = f"Insufficient target spots: {len(target_spots)} < {min_spots_required} required"
        if verbose:
            print(f"    ✗ SKIPPING: {error_msg}")
            print(f"    Creating identity transform to maintain cumulative chain...")

        # Create identity transform (no change) and save it
        identity_3x3 = np.eye(3)
        identity_4x4 = affine_3x3_to_4x4_zyx(identity_3x3)
        identity_4x4_inv = np.linalg.inv(identity_4x4)
        output_yaml = transforms_dir / f"round{round_j}_to_round{round_i}.yml"
        save_affine_to_yaml(identity_4x4_inv, output_yaml)

        if verbose:
            print(f"    ✓ Saved identity transform: {output_yaml.name}")

        return {
            "success": False,
            "error": error_msg,
            "n_matches": 0,
            "affine_3x3": identity_3x3,
            "transform_yaml": str(output_yaml),
        }

    # Graph-based matching (spatial + neighborhood consistency)
    if verbose:
        print(f"    Graph-based matching...")
        print(f"      Source: {len(source_spots)} spots, Target: {len(target_spots)} spots")
        print(f"      Search radius: {params['max_match_distance']}px, k_neighbors: {params['graph_k_neighbors']}")
    t_match = time.time()

    # Create dummy Hu moments (all zeros) since spots don't have shape
    source_hu = np.zeros((len(source_spots), 7))
    target_hu = np.zeros((len(target_spots), 7))

    # Weights: spatial only (no Hu moments, only edge/angle/clustering)
    weights = {
        "hu": 0.0,
        "neighbor_hu": 0.0,
        "edge_length": 0.5,
        "angular_spacing": 0.4,
        "clustering": 0.1,
    }

    source_idx, target_idx, distances, _, _ = match_cells_by_graph_consistency(
        source_spots,
        target_spots,
        source_hu,
        target_hu,
        search_radius=params["max_match_distance"],
        k_neighbors=params["graph_k_neighbors"],
        top_k_candidates=params["graph_top_k_candidates"],
        weights=weights,
        max_score_threshold=100,  # Take top 100 matches
        min_matches_per_cell=0,
        min_total_matches=10,
        cache_dir=None,  # No cache for ISS round matching (different each time)
        verbose=True,  # Show progress
    )

    dt_match = time.time() - t_match

    matched_source = source_spots[source_idx]
    matched_target = target_spots[target_idx]

    if verbose:
        print(f"      Found {len(matched_source)} matches ({dt_match:.2f}s)")

    # Check if we have enough matches for RANSAC
    min_matches_required = params["min_samples"]
    if len(matched_source) < min_matches_required:
        if verbose:
            print(f"      WARNING: Insufficient matches ({len(matched_source)} < {min_matches_required})")
            print(f"      Falling back to PCC-only alignment (no RANSAC refinement)")

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
    else:
        # RANSAC affine estimation
        if verbose:
            print(f"    RANSAC affine estimation...")
        t_ransac = time.time()
        ransac_affine_3x3, inliers, metrics = estimate_affine_ransac(
            matched_source,
            matched_target,
            params["min_samples"],
            params["residual_threshold"],
            params["max_trials"],
            params["stop_probability"],
            params["transform_type"],
        )
        dt_ransac = time.time() - t_ransac

        if verbose:
            print(f"      RANSAC: {metrics['n_inliers']}/{metrics['n_matches']} inliers ({dt_ransac:.2f}s)")
            print(f"      Inlier ratio: {metrics['inlier_ratio']:.2%}")
            print(f"      Residual: {metrics['residual_mean']:.2f} ± {metrics['residual_std']:.2f} px")

        # Iterative refinement: if residual is borderline, transform source spots
        # using current affine, re-match at tighter radius, and refit.
        refine_max_residual = 10.0
        if metrics["residual_mean"] > refine_max_residual and metrics.get("inlier_ratio", 0) > 0.75:
            refine_radius = min(params["max_match_distance"], refine_max_residual * 2)
            for refine_iter in range(3):
                src_h = np.column_stack([source_spots, np.ones(len(source_spots))])
                src_warped = (ransac_affine_3x3 @ src_h.T).T[:, :2]

                ref_src_idx, ref_tgt_idx = kdtree_matching(
                    src_warped, target_spots, max_distance=refine_radius,
                )
                if len(ref_src_idx) < params["min_samples"]:
                    break

                ref_affine, ref_inliers, ref_metrics = estimate_affine_ransac(
                    source_spots[ref_src_idx], target_spots[ref_tgt_idx],
                    params["min_samples"], params["residual_threshold"],
                    params["max_trials"], params["stop_probability"],
                    params["transform_type"],
                )

                improved = ref_metrics["residual_mean"] < metrics["residual_mean"]
                if verbose:
                    print(f"    Iterative refinement [{refine_iter+1}]: "
                          f"{ref_metrics['n_inliers']}/{ref_metrics['n_matches']} inliers, "
                          f"residual {ref_metrics['residual_mean']:.2f}px, "
                          f"inlier ratio {ref_metrics['inlier_ratio']:.1%}"
                          f"{' ✓' if improved else ' (no improvement)'}")

                if not improved:
                    break

                ransac_affine_3x3 = ref_affine
                inliers = ref_inliers
                metrics = ref_metrics
                refine_radius = max(ref_metrics["residual_mean"] * 1.5, 5.0)

    # Use RANSAC transform directly (no PCC pre-alignment)
    affine_3x3 = ransac_affine_3x3

    # --- Quality gate ---
    # Consecutive ISS rounds share the stitched canvas, so a good round-pair fit
    # has a high inlier ratio (healthy wells: ~70-100%) and a near-identity shift.
    # A fit built on very few inliers is unreliable and can emit a large spurious
    # translation; because the cumulative chain composes every round onto R0, that
    # poisons the whole well (observed: well 2 R1→R0 fit on 9/79 inliers produced a
    # 25px jump that propagated through rounds 1-9). Reject such fits and fall back
    # to identity (no added drift) — strictly safer than a spurious shift — and
    # flag it in the metrics/log for review. (The too-few-matches case above
    # already falls back to identity, so this only fires on real RANSAC fits.)
    ROUND_PAIR_MIN_INLIER_RATIO = 0.35
    ROUND_PAIR_MIN_ABS_INLIERS = 20
    _n_inliers = int(metrics.get("n_inliers", 0))
    _inlier_ratio = float(metrics.get("inlier_ratio", 0.0))
    if len(matched_source) >= min_matches_required and (
        _inlier_ratio < ROUND_PAIR_MIN_INLIER_RATIO
        or _n_inliers < ROUND_PAIR_MIN_ABS_INLIERS
    ):
        if verbose:
            print(
                f"    ⚠️  QUALITY GATE: low-confidence round-pair fit "
                f"({_n_inliers} inliers, {_inlier_ratio:.0%} ratio < "
                f"{ROUND_PAIR_MIN_INLIER_RATIO:.0%}/{ROUND_PAIR_MIN_ABS_INLIERS}) — "
                f"rejecting transform "
                f"(dy={ransac_affine_3x3[0,2]:.1f}, dx={ransac_affine_3x3[1,2]:.1f}) "
                f"and falling back to identity to avoid corrupting the cumulative chain"
            )
        metrics["quality_gate"] = "rejected_low_confidence"
        metrics["rejected_affine_dy"] = float(ransac_affine_3x3[0, 2])
        metrics["rejected_affine_dx"] = float(ransac_affine_3x3[1, 2])
        affine_3x3 = np.eye(3)
    else:
        metrics["quality_gate"] = "passed"

    # Extract final transform parameters
    final_trans = affine_3x3[:2, 2]
    final_rot = np.arctan2(affine_3x3[1, 0], affine_3x3[0, 0])
    final_scale_x = np.sqrt(affine_3x3[0, 0] ** 2 + affine_3x3[1, 0] ** 2)
    final_scale_y = np.sqrt(affine_3x3[0, 1] ** 2 + affine_3x3[1, 1] ** 2)

    if verbose:
        print(f"    Final transform:")
        print(f"      Translation: dy={final_trans[0]:.1f}, dx={final_trans[1]:.1f}")
        print(f"      Rotation: {np.degrees(final_rot):.2f}°")
        print(f"      Scale: ({final_scale_x:.4f}, {final_scale_y:.4f})")

    # Save affine to YAML (biahub convention: save inverse)
    affine_4x4 = affine_3x3_to_4x4_zyx(affine_3x3)
    affine_4x4_inv = np.linalg.inv(affine_4x4)
    output_yaml = transforms_dir / f"round{round_j}_to_round{round_i}.yml"
    save_affine_to_yaml(affine_4x4_inv, output_yaml)

    # Generate visualizations
    if verbose:
        print(f"    Generating visualizations...")

    # # 1. Spot extraction visualization (source)
    # spot_extract_src_path = overlays_dir / f"round{round_j}_extracted_spots.png"
    # try:
    #     create_spot_extraction_visualization(
    #         zarr_path, position, round_j, source_spots, spot_extract_src_path, crop_size=500
    #     )
    #     if verbose:
    #         print(f"      Spots (R{round_j}): {spot_extract_src_path.name}")
    # except Exception as e:
    #     if verbose:
    #         print(f"      WARNING: Spot extraction viz failed: {e}")

    # # 2. Spot extraction visualization (target)
    # spot_extract_tgt_path = overlays_dir / f"round{round_i}_extracted_spots.png"
    # try:
    #     create_spot_extraction_visualization(
    #         zarr_path, position, round_i, target_spots, spot_extract_tgt_path, crop_size=500
    #     )
    #     if verbose:
    #         print(f"      Spots (R{round_i}): {spot_extract_tgt_path.name}")
    # except Exception as e:
    #     if verbose:
    #         print(f"      WARNING: Spot extraction viz failed: {e}")

    # 3. Before/after overlay
    overlay_path = overlays_dir / f"round{round_j}_to_round{round_i}_overlay.png"
    if generate_overlays:
        try:
            create_iss_registration_overlay(
                zarr_path, position, round_j, round_i, affine_3x3, overlay_path,
                crop_size=500, n_crops=6
            )
            if verbose:
                print(f"      Overlay: {overlay_path.name}")
        except Exception as e:
            if verbose:
                print(f"      WARNING: Overlay failed: {e}")

    # 2. Spot match visualization
    matches_path = overlays_dir / f"round{round_j}_to_round{round_i}_matches.png"
    if generate_overlays:
        try:
            create_spot_match_visualization(
                zarr_path, position, round_j, round_i,
                source_spots, target_spots, source_idx, target_idx,
                affine_3x3, matches_path, max_matches_to_show=200
            )
            if verbose:
                print(f"      Matches: {matches_path.name}")
        except Exception as e:
            if verbose:
                print(f"      WARNING: Match visualization failed: {e}")

    # Compute overlap metrics on a 1k x 1k crop (optional)
    overlap_metrics = None
    if compute_metrics:
        if verbose:
            print(f"    Computing overlap metrics...")
        try:
            overlap_metrics = compute_overlap_metrics(
                zarr_path, position, round_j, round_i, affine_3x3,
                crop_size=1000, spot_channels=[1, 2, 3, 4]
            )
            if verbose:
                print(f"      Direct (source→target):")
                print(f"        Before: {overlap_metrics['direct_before']:.1%} overlap")
                print(f"        After:  {overlap_metrics['direct_after']:.1%} overlap (Δ={overlap_metrics['direct_improvement']:.1%})")
                print(f"      Inverse (target→source):")
                print(f"        Before: {overlap_metrics['inverse_before']:.1%} overlap")
                print(f"        After:  {overlap_metrics['inverse_after']:.1%} overlap (Δ={overlap_metrics['inverse_improvement']:.1%})")
        except Exception as e:
            if verbose:
                print(f"      WARNING: Overlap computation failed: {e}")
            overlap_metrics = None
    else:
        if verbose:
            print(f"    Skipping overlap metrics computation (compute_metrics=False)")

    dt_total = time.time() - t_start
    if verbose:
        print(f"    Total time: {dt_total:.2f}s\n")

    results = {
        "affine_3x3": affine_3x3,
        "affine_4x4": affine_4x4,
        "metrics": metrics,
        "overlap_metrics": overlap_metrics,
        "yaml": output_yaml,
        "overlay": overlay_path if overlay_path.exists() else None,
        "n_source_spots": len(source_spots),
        "n_target_spots": len(target_spots),
    }

    return results


def centroids_to_image_crop(centroids: np.ndarray, crop_size: int = 3000, sigma: float = 5.0) -> tuple:
    """
    Convert centroid coordinates to a smoothed density image for PCC.
    Uses a central crop region for speed (avoids allocating huge arrays).

    Parameters
    ----------
    centroids : np.ndarray
        Nx2 array of (y, x) coordinates
    crop_size : int
        Size of square crop region (e.g., 3000 = 3000x3000px)
    sigma : float
        Gaussian smoothing sigma for robustness

    Returns
    -------
    tuple
        (density_image, crop_offset_y, crop_offset_x)
    """
    # Find center of centroid cloud
    center_y = np.mean(centroids[:, 0])
    center_x = np.mean(centroids[:, 1])

    # Define crop region centered on centroids
    crop_y_start = int(center_y - crop_size // 2)
    crop_x_start = int(center_x - crop_size // 2)

    # Create small crop image
    img = np.zeros((crop_size, crop_size), dtype=np.float32)

    # Convert centroids to crop-local coordinates
    coords_local = centroids - np.array([crop_y_start, crop_x_start])
    coords_local = coords_local.astype(int)

    # Filter in-bounds coordinates
    valid = (coords_local[:, 0] >= 0) & (coords_local[:, 0] < crop_size) & \
            (coords_local[:, 1] >= 0) & (coords_local[:, 1] < crop_size)

    # Place points at centroid locations
    img[coords_local[valid, 0], coords_local[valid, 1]] = 1.0

    # Smooth with Gaussian for better PCC
    if sigma > 0:
        img = gaussian_filter(img, sigma=sigma)

    return img, crop_y_start, crop_x_start


def compute_dapi_to_dapi_pcc_from_original(
    dataset: "OpsDataset",
    position: str,
    overlays_dir: Path = None,
    verbose: bool = True,
) -> tuple[np.ndarray, float]:
    """
    Compute PCC offset between pre-nuclei round DAPI (round 0) and first ISS round DAPI (round 1).

    Uses the original raw NDTiff MicroManager data to measure the DAPI offset between:
    - Pre-nuclei round (round 0): DAPI-only imaging
    - First ISS round (round 1): First round with spots + DAPI

    This offset is then used to pre-align nucleus segmentation to spots, handling the
    microscope stage shift that occurs between these two acquisitions.

    The DAPI channel is located by NAME in each round. If the first ISS round has
    no DAPI channel (e.g. a no-incorporation round 0 + extra end round layout),
    the PCC is skipped: a zero shift flagged unreliable is returned so the caller
    falls back to raw-centroid (no_pcc) matching instead of a bogus pre-alignment.

    Parameters
    ----------
    dataset : OpsDataset
        Dataset containing paths to raw ISS NDTiff data.
    position : str
        Position string (e.g., "A/1/0").
    verbose : bool
        Print progress messages.

    Returns
    -------
    tuple
        (pcc_shift, pcc_error)
        - pcc_shift: Translation offset (dy, dx) in pixels
        - pcc_error: PCC error metric (lower is better)
    """
    from ndtiff import Dataset
    import re

    # Get paths to original raw NDTiff data
    experiment_dir = dataset.iss_tif_dir

    # Extract well identifier from position (e.g., "A/1/0" -> "A1")
    well_parts = position.split("/")
    well = f"{well_parts[0]}{well_parts[1]}"  # e.g., "A1"

    # Find all round directories for this well (e.g., A1_1, A1_2, ...)
    all_dirs = [d for d in experiment_dir.iterdir() if d.is_dir()]
    round_dirs = []
    for d in all_dirs:
        # Match pattern {well}_{round} e.g. A1_1
        m = re.match(rf"^{well}_(\d+)$", d.name)
        if m:
            round_num = int(m.group(1))
            round_dirs.append((round_num, d))

    # DEBUG: Print ALL found directories before sorting
    if verbose:
        print(f"    DEBUG: Found {len(round_dirs)} round directories for well {well}:")
        for round_num, d in round_dirs:
            print(f"      - {d.name} (round number: {round_num})")

    # Sort by round number
    round_dirs.sort(key=lambda x: x[0])

    if len(round_dirs) < 2:
        raise FileNotFoundError(
            f"Need at least 2 rounds of raw data in {experiment_dir}. "
            f"Found {len(round_dirs)} directories matching pattern '{well}_*'"
        )

    round0_dir = round_dirs[0][1]  # Pre-nuclei round
    round1_dir = round_dirs[1][1]  # First ISS round

    if verbose:
        print(f"    Loading DAPI from raw MicroManager NDTiff data...")
        print(f"    DEBUG: Selected directories for PCC computation:")
        print(f"      Round 0 (pre-nuclei): {round0_dir.name} (index [0] in sorted list)")
        print(f"      Round 1 (first ISS):  {round1_dir.name} (index [1] in sorted list)")
        print(f"      Full path round 0: {round0_dir}")
        print(f"      Full path round 1: {round1_dir}")

    try:
        # Load round 0 (pre-nuclei / nucleus round; DAPI is channel 0)
        ds0 = Dataset(str(round0_dir))
        axes0 = ds0.axes

        # Some experiments (e.g. a no-incorporation round 0 compensated by an extra
        # end round) image the first ISS round WITHOUT a DAPI channel. Blindly using
        # channel[0] there correlates DAPI against a spot channel and yields a bogus
        # pre-alignment, so skip the PCC and match on raw centroids (no_pcc) instead.
        ds1 = Dataset(str(round1_dir))
        axes1 = ds1.axes
        def _has_dapi(axes):
            return any('dapi' in str(c).lower() for c in axes.get('channel', []))
        if not (_has_dapi(axes0) and _has_dapi(axes1)):
            ds0.close(); ds1.close()
            if verbose:
                print(f"    First ISS round has no DAPI channel "
                      f"(round1 channels={list(axes1.get('channel', []))}); skipping "
                      f"DAPI-to-DAPI PCC → raw-centroid (no_pcc) matching.")
            unreliable_quality = {
                "shift_std": np.array([999.0, 999.0]),
                "shift_std_filtered": np.array([999.0, 999.0]),
                "shift_median": np.zeros(2),
                "shift_std_raw": np.array([999.0, 999.0]),
                "shift_median_raw": np.zeros(2),
                "n_tiles": 0,
                "n_tiles_used": 0,
                "bimodal_filtered": True,
            }
            return np.zeros(2), 1.0, unreliable_quality

        # Get position names (e.g., "Pos0", "Pos1", ...)
        position_names = list(axes0.get('position', []))
        num_positions = len(position_names)

        if num_positions == 0:
            raise ValueError("No positions found in dataset")

        # Sample many tiles across the entire well to check for spatial variation
        # Use at least 20 tiles, or 10% of available positions (whichever is larger)
        n_samples = max(20, num_positions // 10)
        n_samples = min(n_samples, num_positions)  # Don't exceed available positions

        # Sample evenly across the entire well (not just middle)
        if n_samples >= num_positions:
            sampled_indices = list(range(num_positions))
        else:
            step = num_positions / n_samples
            sampled_indices = [int(i * step) for i in range(n_samples)]

        if verbose:
            print(f"      Total positions: {num_positions}")
            print(f"      Sampling {len(sampled_indices)} tiles across entire well")
            print(f"      Position indices: {sampled_indices[:10]}{'...' if len(sampled_indices) > 10 else ''}")

        # Collect PCC shifts from multiple tiles
        pcc_shifts = []
        pcc_errors = []
        pcc_tile_indices = []  # Track which tiles were sampled
        dapi0_samples = []  # For visualization (keep first few)
        dapi1_samples = []

        for tile_idx in sampled_indices:
            pos_name = position_names[tile_idx]

            # Read DAPI from round 0 (channel 0)
            read_kwargs = {'position': pos_name}
            if 'time' in axes0:
                read_kwargs['time'] = list(axes0['time'])[0]
            if 'z' in axes0:
                read_kwargs['z'] = list(axes0['z'])[0]
            if 'channel' in axes0:
                read_kwargs['channel'] = list(axes0['channel'])[0]  # DAPI (channel 0)

            dapi0 = ds0.read_image(**read_kwargs)

            # Read DAPI from round 1 (ds1 opened above; DAPI is channel 0)
            read_kwargs = {'position': pos_name}
            if 'time' in axes1:
                read_kwargs['time'] = list(axes1['time'])[0]
            if 'z' in axes1:
                read_kwargs['z'] = list(axes1['z'])[0]
            if 'channel' in axes1:
                read_kwargs['channel'] = list(axes1['channel'])[0]  # DAPI (channel 0)

            dapi1 = ds1.read_image(**read_kwargs)

            # Store first 10 tiles for visualization
            if len(dapi0_samples) < 5:
                dapi0_samples.append(dapi0)
                dapi1_samples.append(dapi1)

            # Compute PCC for this tile
            from skimage.transform import downscale_local_mean
            downsample_factor = 4

            dapi0_ds = downscale_local_mean(dapi0.astype(np.float32), (downsample_factor, downsample_factor))
            dapi1_ds = downscale_local_mean(dapi1.astype(np.float32), (downsample_factor, downsample_factor))

            # Compute phase cross-correlation
            pcc_shift_raw, pcc_error, _ = phase_cross_correlation(
                dapi0_ds, dapi1_ds, upsample_factor=10
            )

            pcc_shifts.append(pcc_shift_raw)
            pcc_errors.append(pcc_error)
            pcc_tile_indices.append(tile_idx)

            if verbose and tile_idx in sampled_indices[:10]:  # Only print first 10
                print(f"        Tile {pos_name}: shift dy={pcc_shift_raw[0]:.2f}px, dx={pcc_shift_raw[1]:.2f}px (ds), error={pcc_error:.4f}")

        ds0.close()
        ds1.close()

        # Convert to numpy array for analysis
        pcc_shifts = np.array(pcc_shifts)
        pcc_errors = np.array(pcc_errors)

        # Check for bimodal tile distribution and filter noise tiles if detected
        # Noise tiles (no DAPI signal) return PCC shifts near (0,0), producing
        # a bimodal distribution that corrupts both the mean and median.
        shift_magnitudes = np.linalg.norm(pcc_shifts, axis=1)
        shift_std = np.std(pcc_shifts, axis=0)
        shift_median = np.median(pcc_shifts, axis=0)
        std_mag = np.linalg.norm(shift_std)
        median_mag = np.linalg.norm(shift_median)

        # Detect bimodality: high std relative to median shift
        is_bimodal = (median_mag > 0 and std_mag / median_mag > 0.4) or (median_mag == 0 and std_mag > 5.0)

        if is_bimodal and len(pcc_shifts) >= 4:
            # Otsu's method on shift magnitudes to find optimal split between clusters
            sorted_mags = np.sort(shift_magnitudes)
            best_thresh = 0
            best_variance = -1
            for i in range(1, len(sorted_mags)):
                left, right = sorted_mags[:i], sorted_mags[i:]
                w0, w1 = len(left) / len(sorted_mags), len(right) / len(sorted_mags)
                between_var = w0 * w1 * (left.mean() - right.mean()) ** 2
                if between_var > best_variance:
                    best_variance = between_var
                    best_thresh = (sorted_mags[i - 1] + sorted_mags[i]) / 2

            # Keep the cluster with LARGER magnitude (real DAPI signal, not noise at ~0)
            signal_mask = shift_magnitudes > best_thresh
            noise_mask = ~signal_mask
            if signal_mask.sum() == 0:
                signal_mask = noise_mask  # fallback: keep all
            elif noise_mask.sum() > 0:
                signal_mean = shift_magnitudes[signal_mask].mean()
                noise_mean = shift_magnitudes[noise_mask].mean()
                # Verify signal cluster really is bigger than noise cluster
                if signal_mean < noise_mean:
                    signal_mask = noise_mask

            inlier_mask = signal_mask
            if verbose:
                print(f"      BIMODAL DETECTED: Otsu threshold={best_thresh:.1f}px, "
                      f"keeping {inlier_mask.sum()}/{len(pcc_shifts)} signal tiles")
        else:
            inlier_mask = np.ones(len(pcc_shifts), dtype=bool)

        pcc_shifts_filtered = pcc_shifts[inlier_mask]
        pcc_shift_raw = np.mean(pcc_shifts_filtered, axis=0)
        pcc_error = np.mean(pcc_errors[inlier_mask])

        if verbose:
            print(f"      Sampled {len(pcc_shifts)} tiles across well")
            if is_bimodal:
                print(f"      Tile shift magnitudes: {np.round(shift_magnitudes, 1).tolist()}")
                print(f"      Inlier tiles: {inlier_mask.sum()}/{len(pcc_shifts)}")
            print(f"      PCC shift (filtered): dy={pcc_shift_raw[0]:.2f}px, dx={pcc_shift_raw[1]:.2f}px (ds)")
            print(f"      Median PCC shift:     dy={shift_median[0]:.2f}px, dx={shift_median[1]:.2f}px (ds)")
            print(f"      Std dev (all tiles):  dy={shift_std[0]:.2f}px, dx={shift_std[1]:.2f}px (ds)")
            print(f"      Range dy: [{pcc_shifts[:, 0].min():.2f}, {pcc_shifts[:, 0].max():.2f}]px")
            print(f"      Range dx: [{pcc_shifts[:, 1].min():.2f}, {pcc_shifts[:, 1].max():.2f}]px")
            print(f"      Average PCC error: {pcc_error:.4f}")

        # Create distribution plot of PCC shifts
        if overlays_dir is not None:
            try:
                fig = Figure(figsize=(12, 5)); axes = fig.subplots(1, 2)

                # Histogram of dy shifts
                axes[0].hist(pcc_shifts[:, 0], bins=30, alpha=0.7, edgecolor='black')
                axes[0].axvline(pcc_shift_raw[0], color='red', linestyle='--', linewidth=2, label=f'Mean: {pcc_shift_raw[0]:.2f}px')
                axes[0].axvline(shift_median[0], color='green', linestyle='--', linewidth=2, label=f'Median: {shift_median[0]:.2f}px')
                axes[0].set_xlabel('dy shift (downsampled px)', fontsize=12)
                axes[0].set_ylabel('Count', fontsize=12)
                axes[0].set_title(f'PCC dy shift distribution\nStd: {shift_std[0]:.2f}px', fontsize=12, fontweight='bold')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

                # Histogram of dx shifts
                axes[1].hist(pcc_shifts[:, 1], bins=30, alpha=0.7, edgecolor='black')
                axes[1].axvline(pcc_shift_raw[1], color='red', linestyle='--', linewidth=2, label=f'Mean: {pcc_shift_raw[1]:.2f}px')
                axes[1].axvline(shift_median[1], color='green', linestyle='--', linewidth=2, label=f'Median: {shift_median[1]:.2f}px')
                axes[1].set_xlabel('dx shift (downsampled px)', fontsize=12)
                axes[1].set_ylabel('Count', fontsize=12)
                axes[1].set_title(f'PCC dx shift distribution\nStd: {shift_std[1]:.2f}px', fontsize=12, fontweight='bold')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

                fig.suptitle(f'DAPI-to-DAPI PCC Shift Distribution (n={len(pcc_shifts)} tiles)',
                            fontsize=14, fontweight='bold')
                fig.tight_layout()

                dist_plot_path = overlays_dir / "dapi_pcc_shift_distribution.png"
                fig.savefig(dist_plot_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

                if verbose:
                    print(f"      Saved PCC distribution plot: {dist_plot_path.name}")
            except Exception as e:
                if verbose:
                    print(f"      WARNING: PCC distribution plot failed: {e}")

        # Use first tile for visualization
        dapi0 = dapi0_samples[0]
        dapi1 = dapi1_samples[0]

    except Exception as e:
        raise RuntimeError(
            f"Failed to load DAPI images from raw NDTiff data: {e}"
        )

    if verbose:
        print(f"      DAPI tile shape: {dapi0.shape}")

    # pcc_shift_raw is the shift to align dapi1→dapi0 (spots→nucleus)
    # Store it as spots→nucleus (positive direction) for simpler downstream application
    pcc_shift_spots_to_nuc = pcc_shift_raw * downsample_factor  # spots→nucleus, full res
    pcc_shift = pcc_shift_raw * downsample_factor  # Store same as spots→nucleus for consistency

    if verbose:
        print(f"      PCC shift (spots→nucleus): dy={pcc_shift[0]:.2f}px, dx={pcc_shift[1]:.2f}px")
        print(f"      Stored with POSITIVE sign for both axes (simpler downstream application)")

    # Create before/after overlay showing DAPI alignment for all tiles
    if overlays_dir is not None:
        try:
            from scipy.ndimage import affine_transform

            if verbose:
                print(f"    Generating DAPI-to-DAPI PCC overlay (all {len(dapi0_samples)} tiles)...")

            # Apply shift to align spots to nucleus (use the raw shift direction)
            shift_to_apply = pcc_shift_spots_to_nuc  # spots→nucleus

            # Create affine transform matrix for scipy
            affine_matrix = np.array([
                [1, 0, -shift_to_apply[0]],  # dy
                [0, 1, -shift_to_apply[1]]   # dx
            ])

            # Create figure with 2 rows (before/after) × N columns (tiles)
            n_tiles = len(dapi0_samples)
            fig = Figure(figsize=(6 * n_tiles, 12)); axes = fig.subplots(2, n_tiles)
            if n_tiles == 1:
                axes = axes.reshape(2, 1)  # Ensure 2D array

            for tile_idx, (dapi0, dapi1) in enumerate(zip(dapi0_samples, dapi1_samples)):
                # Normalize images for visualization with aggressive background removal
                dapi0_norm = dapi0.astype(np.float32)
                dapi1_norm = dapi1.astype(np.float32)

                # Aggressive background removal: subtract median and clip low percentile
                p50_0 = np.percentile(dapi0_norm, 50)
                p50_1 = np.percentile(dapi1_norm, 50)
                p99_0 = np.percentile(dapi0_norm, 99.5)
                p99_1 = np.percentile(dapi1_norm, 99.5)

                # Subtract median (background) and normalize to p99.5
                dapi0_norm = np.clip((dapi0_norm - p50_0) / (p99_0 - p50_0 + 1e-10), 0, 1)
                dapi1_norm = np.clip((dapi1_norm - p50_1) / (p99_1 - p50_1 + 1e-10), 0, 1)

                # Apply gamma to further crush background
                dapi0_norm = np.power(dapi0_norm, 1.5)
                dapi1_norm = np.power(dapi1_norm, 1.5)

                # Apply alignment to dapi1
                dapi1_aligned = affine_transform(
                    dapi1_norm,
                    affine_matrix,
                    order=1,  # Bilinear interpolation
                    mode='constant',
                    cval=0.0
                )

                # Before: dapi0 (red) vs dapi1 (green) - should be offset
                before_rgb = np.zeros((*dapi0.shape, 3), dtype=np.float32)
                before_rgb[..., 0] = dapi0_norm  # Red = nucleus (pre-nuclei round)
                before_rgb[..., 1] = dapi1_norm  # Green = spots (first ISS round)

                axes[0, tile_idx].imshow(before_rgb)
                if tile_idx == 0:
                    axes[0, tile_idx].set_title(f'Before PCC - Tile {tile_idx+1}\nDAPI0 (red) vs DAPI1 (green)',
                                                fontsize=10)
                else:
                    axes[0, tile_idx].set_title(f'Tile {tile_idx+1}', fontsize=10)
                axes[0, tile_idx].axis('off')

                # After: dapi0 (red) vs dapi1_aligned (green) - should overlap (yellow)
                after_rgb = np.zeros((*dapi0.shape, 3), dtype=np.float32)
                after_rgb[..., 0] = dapi0_norm        # Red = nucleus
                after_rgb[..., 1] = dapi1_aligned     # Green = aligned spots

                axes[1, tile_idx].imshow(after_rgb)
                if tile_idx == 0:
                    axes[1, tile_idx].set_title(f'After PCC - Tile {tile_idx+1}\nYellow = good overlap',
                                                fontsize=10)
                else:
                    axes[1, tile_idx].set_title(f'Tile {tile_idx+1}', fontsize=10)
                axes[1, tile_idx].axis('off')

            # Add overall figure title
            fig.suptitle(f'DAPI-to-DAPI PCC Correction\nShift to align spots→nucleus: dy={shift_to_apply[0]:.1f}px, dx={shift_to_apply[1]:.1f}px',
                        fontsize=12, fontweight='bold')

            fig.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for suptitle
            overlay_path = overlays_dir / "dapi_to_dapi_pcc_correction.png"
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(overlay_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            if verbose:
                print(f"      Saved DAPI-to-DAPI overlay: {overlay_path.name}")
        except Exception as e:
            if verbose:
                print(f"      WARNING: DAPI-to-DAPI overlay failed: {e}")

    # Compute filtered std (only inlier tiles) for downstream quality reporting
    shift_std_filtered = np.std(pcc_shifts_filtered, axis=0) if len(pcc_shifts_filtered) > 1 else np.zeros(2)

    pcc_quality = {
        "shift_std": shift_std * downsample_factor,  # full-res, ALL tiles
        "shift_std_filtered": shift_std_filtered * downsample_factor,  # full-res, inlier tiles only
        "shift_median": shift_median * downsample_factor,  # full-res pixels
        "shift_std_raw": shift_std,  # downsampled pixels
        "shift_median_raw": shift_median,  # downsampled pixels
        "n_tiles": len(pcc_shifts),
        "n_tiles_used": int(inlier_mask.sum()),
        "bimodal_filtered": bool(is_bimodal),
    }

    return pcc_shift, pcc_error, pcc_quality


def compute_segmentation_to_nucleus_pcc(
    seg_zarr_path: Path,
    iss_zarr_path: Path,
    position: str,
    crop_size: int = 2048,
    overlays_dir: Path = None,
    verbose: bool = True,
) -> tuple[np.ndarray, float]:
    """
    Compute PCC offset between segmentation mask and Round 0 Channel 0 DAPI.

    This measures the translation offset between the segmentation image and the
    nuclear DAPI channel, using a center crop for efficient computation.

    Parameters
    ----------
    seg_zarr_path : Path
        Path to segmentation zarr (bc_segmentation.zarr)
    iss_zarr_path : Path
        Path to ISS zarr with registered rounds
    position : str
        HCS position (e.g., "A/1/0")
    crop_size : int
        Size of center crop to use for PCC (default: 2048)
    overlays_dir : Path, optional
        Directory to save before/after overlay
    verbose : bool
        Print progress messages

    Returns
    -------
    pcc_shift : np.ndarray
        Translation [dy, dx] to move Round 0 DAPI to segmentation (seg→dapi)
    pcc_error : float
        PCC error metric
    """
    import zarr
    from skimage.registration import phase_cross_correlation
    from scipy.ndimage import affine_transform

    if verbose:
        print(f"    Computing segmentation→nucleus PCC offset...")
        print(f"      seg_zarr_path: {seg_zarr_path}")
        print(f"      iss_zarr_path: {iss_zarr_path}")
        print(f"      position: {position}")

    # Load segmentation mask (center crop)
    if verbose:
        print(f"      Loading segmentation...")
    seg_store = zarr.open(str(seg_zarr_path), mode='r')

    # Access the array - could be at position directly or position/0
    seg_group = seg_store[position]
    if verbose:
        print(f"      Seg group type: {type(seg_group)}")
        if hasattr(seg_group, 'keys'):
            print(f"      Seg group keys: {list(seg_group.keys())}")

    # Try to get the array - it might be at index 0 or directly accessible
    if hasattr(seg_group, 'shape'):
        # It's already an array
        seg_array = seg_group
    else:
        # It's a group, get the first array (usually '0')
        seg_array = seg_group['0'] if '0' in seg_group else seg_group[list(seg_group.keys())[0]]

    if verbose:
        print(f"      Seg array shape: {seg_array.shape}")

    # Handle different zarr formats
    # We need dimensions to calculate crop, but avoid reading full data
    if hasattr(seg_array, 'shape'):
        shape = seg_array.shape
        # Assume last two dims are Y, X
        H, W = shape[-2:]
    else:
        # Fallback if shape not directly available (unlikely)
        seg_full = np.array(seg_array).squeeze()
        H, W = seg_full.shape[-2:]

    # Use smaller crop (half size) offset 25% toward center from geometric center
    actual_crop_size = crop_size // 2
    center_y = H // 2
    center_x = W // 2

    # Offset 25% from center (toward more central region)
    offset = int(crop_size * 0.25)
    crop_y_start = center_y - actual_crop_size // 2 - offset
    crop_x_start = center_x - actual_crop_size // 2 - offset
    crop_y_end = crop_y_start + actual_crop_size
    crop_x_end = crop_x_start + actual_crop_size

    # Read ONLY the crop region using slicing
    # Handle various dimensions: (Y, X), (1, Y, X), (1, 1, 1, Y, X)
    if len(shape) == 2:
        seg_crop = seg_array[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    elif len(shape) == 3:
        seg_crop = seg_array[0, crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    elif len(shape) == 4:
        seg_crop = seg_array[0, 0, crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    elif len(shape) == 5:
        seg_crop = seg_array[0, 0, 0, crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    else:
        # Fallback: load full and crop (slow)
        print(f"      WARNING: Unknown shape {shape}, falling back to full load")
        seg_full = np.array(seg_array).squeeze()
        seg_crop = seg_full[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

    seg_crop = np.array(seg_crop)  # Ensure numpy array

    # Convert segmentation to binary mask (any label > 0)
    seg_binary = (seg_crop > 0).astype(np.float32)

    # Load Round 0 Channel 0 DAPI (center crop)
    if verbose:
        print(f"      Loading DAPI from ISS zarr...")
    with open_ome_zarr(iss_zarr_path, mode="r") as iss_store:
        iss_data = iss_store[position].data
        # iss_data is likely a dask array or zarr array
        # Shape is typically (T, C, Z, Y, X)
        # We want T=0, C=0, Z=0, crop Y, crop X

        dapi_crop = iss_data[0, 0, 0, crop_y_start:crop_y_end, crop_x_start:crop_x_end]

        # If it's a dask array, compute it
        if hasattr(dapi_crop, "compute"):
            dapi_crop = dapi_crop.compute()

        dapi_crop = np.array(dapi_crop)

    # Convert to float32 to avoid precision issues with float16 in percentile calculations
    if dapi_crop.dtype == np.float16:
        dapi_crop = dapi_crop.astype(np.float32)

    # Threshold DAPI to binary for better PCC
    # Use nanpercentile to handle potential NaN values in the data
    # Check if all values are NaN (invalid data)
    if np.isnan(dapi_crop).all():
        if verbose:
            print(f"      WARNING: All DAPI values are NaN! Using zeros instead.")
        dapi_binary = np.zeros_like(dapi_crop, dtype=np.float32)
        dapi_thresh = 0.0
    else:
        dapi_thresh = np.nanpercentile(dapi_crop, 70)
        # Replace NaNs with 0 before thresholding
        dapi_crop_clean = np.nan_to_num(dapi_crop, nan=0.0)
        dapi_binary = (dapi_crop_clean > dapi_thresh).astype(np.float32)
    if verbose:
        print(f"      DAPI threshold: {dapi_thresh:.1f}")
        print(f"      Computing PCC...")

    # Compute PCC (seg as reference, dapi as moving)
    pcc_result = phase_cross_correlation(
        seg_binary, dapi_binary, upsample_factor=10
    )
    if verbose:
        print(f"      PCC result type: {type(pcc_result)}, len: {len(pcc_result) if hasattr(pcc_result, '__len__') else 'N/A'}")
        print(f"      PCC result: {pcc_result}")
    pcc_shift_raw = pcc_result[0]
    pcc_error = pcc_result[1] if len(pcc_result) > 1 else 0.0

    # Handle NaN errors
    if np.isnan(pcc_error):
        if verbose:
            print(f"      WARNING: PCC error is NaN (likely due to all-zero or all-NaN input)")
        pcc_error = -1.0  # Sentinel value to indicate invalid/unreliable registration

    # pcc_shift_raw is how to move DAPI to align with seg (seg→dapi direction)
    # We want seg→dapi for our framework (segmentation is anchor)
    pcc_shift = pcc_shift_raw

    if verbose:
        print(f"      PCC shift (Nucleus→Seg): dy={pcc_shift[0]:.2f}px, dx={pcc_shift[1]:.2f}px")
        print(f"      PCC error: {pcc_error:.6f}")

    # Create overlay if requested
    if overlays_dir is not None:
        try:
            import matplotlib.pyplot as plt

            # Normalize for visualization
            seg_norm = seg_binary
            dapi_norm = dapi_binary

            # Apply shift to DAPI to align it with seg
            shift_to_apply = pcc_shift_raw
            affine_matrix = np.array([
                [1, 0, -shift_to_apply[0]],
                [0, 1, -shift_to_apply[1]]
            ])
            dapi_aligned = affine_transform(
                dapi_norm, affine_matrix, order=1,
                output_shape=dapi_norm.shape, cval=0
            )

            # Create figure
            fig = Figure(figsize=(12, 6)); axes = fig.subplots(1, 2)

            # Before: seg (red) vs dapi (green)
            before_rgb = np.zeros((*seg_crop.shape, 3), dtype=np.float32)
            before_rgb[..., 0] = seg_norm     # Red = segmentation
            before_rgb[..., 1] = dapi_norm    # Green = DAPI

            axes[0].imshow(before_rgb)
            axes[0].set_title('Before PCC\nSeg (red) vs DAPI (green)', fontsize=12)
            axes[0].axis('off')

            # After: seg (red) vs dapi_aligned (green) - should overlap (yellow)
            after_rgb = np.zeros((*seg_crop.shape, 3), dtype=np.float32)
            after_rgb[..., 0] = seg_norm          # Red = segmentation
            after_rgb[..., 1] = dapi_aligned      # Green = aligned DAPI

            axes[1].imshow(after_rgb)
            axes[1].set_title('After PCC\nYellow = good overlap', fontsize=12)
            axes[1].axis('off')

            fig.suptitle(f'Segmentation→Nucleus PCC Correction\nShift (Nucleus→Seg): dy={pcc_shift[0]:.1f}px, dx={pcc_shift[1]:.1f}px',
                        fontsize=14, fontweight='bold')

            fig.tight_layout(rect=[0, 0, 1, 0.96])
            overlay_path = overlays_dir / "seg_to_nucleus_pcc_correction.png"
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(overlay_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            if verbose:
                print(f"      Saved seg→nucleus overlay: {overlay_path.name}")
        except Exception as e:
            if verbose:
                print(f"      WARNING: Seg→nucleus overlay failed: {e}")

    return pcc_shift, pcc_error


def register_segmentation_to_nucleus(
    iss_zarr_path: Path,
    seg_zarr_path: Path,
    position: str,
    transforms_dir: Path,
    overlays_dir: Path,
    verbose: bool = True,
) -> dict:
    """
    Register segmentation → Round 0 Channel 0 DAPI using PCC.

    This is the first step in the registration pipeline, establishing the
    segmentation as the global anchor. All subsequent transforms will compose
    through this transform.

    Parameters
    ----------
    iss_zarr_path : Path
        Path to stitched ISS zarr
    seg_zarr_path : Path
        Path to segmentation zarr (bc_segmentation.zarr)
    position : str
        HCS position (e.g., "A/1/0")
    transforms_dir : Path
        Output directory for transform YAML files
    overlays_dir : Path
        Output directory for visualization overlays
    verbose : bool
        Print progress messages

    Returns
    -------
    dict
        Results including affine_3x3, metrics, and output paths
    """
    if verbose:
        print(f"  Registering segmentation → nucleus (Round 0, ch0)...")

    t_start = time.time()

    # Compute PCC between segmentation and Round 0 DAPI
    pcc_shift, pcc_error = compute_segmentation_to_nucleus_pcc(
        seg_zarr_path, iss_zarr_path, position,
        crop_size=2048, overlays_dir=overlays_dir, verbose=verbose
    )

    # Create translation-only affine (segmentation→nucleus)
    affine_3x3 = np.eye(3)
    affine_3x3[:2, 2] = pcc_shift

    # Save transform (biahub convention: save inverse)
    affine_4x4 = affine_3x3_to_4x4_zyx(affine_3x3)
    affine_4x4_inv = np.linalg.inv(affine_4x4)

    output_yaml = transforms_dir / "segmentation_to_nucleus.yaml"
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    save_affine_to_yaml(affine_4x4_inv, output_yaml)

    elapsed = time.time() - t_start

    if verbose:
        print(f"    ✓ Segmentation→nucleus registration complete ({elapsed:.2f}s)")
        print(f"      Shift: dy={pcc_shift[0]:.2f}px, dx={pcc_shift[1]:.2f}px")
        print(f"      Saved: {output_yaml.name}")

    return {
        "success": True,
        "affine_3x3": affine_3x3,
        "pcc_shift": pcc_shift,
        "pcc_error": pcc_error,
        "yaml_path": output_yaml,
        "elapsed_sec": elapsed,
    }


def manual_nucleus_registration(
    well,
    transforms_dir: Path,
    manual_offsets: dict,
    verbose: bool = True,
) -> dict:
    """
    Create nucleus→spots affine from manual (dy, dx) offset for a specific well.

    For DAPI_round10 experiments where automated nucleus→spots registration
    is difficult (no Round 0 nucleus available for PCC alignment).

    Parameters
    ----------
    well : int
        Well number (1, 2, 3, etc.)
    transforms_dir : Path
        Output directory for affine YAML files (e.g., register/transforms/A{well}).
    manual_offsets : dict
        Dict mapping well numbers to (dy, dx) tuples.
        Example: {1: (-8.5, 2.1), 2: (-7.2, 1.8), 3: (-9.1, 2.3)}
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict
        Registration result with affine and YAML path.
    """
    if well not in manual_offsets:
        raise ValueError(
            f"Manual offset for well {well} not provided. "
            f"Available wells: {list(manual_offsets.keys())}"
        )

    dy, dx = manual_offsets[well]

    row, col = parse_well(well)  # accept full row/col unit, default row A
    well_token = f"{row}{col}"
    if verbose:
        print(f"\n  Using manual nucleus registration for Well {well_token}:")
        print(f"    dy={dy:.2f}px, dx={dx:.2f}px")

    # Create translation-only affine (nucleus→spots)
    affine_3x3 = np.eye(3)
    affine_3x3[:2, 2] = [dy, dx]

    # Save transform (biahub convention: save inverse for spots→nucleus)
    affine_4x4 = affine_3x3_to_4x4_zyx(affine_3x3)
    affine_4x4_inv = np.linalg.inv(affine_4x4)

    output_yaml = transforms_dir / "nucleus_to_round0.yml"
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    save_affine_to_yaml(affine_4x4_inv, output_yaml)

    if verbose:
        print(f"    ✓ Manual nucleus→spots registration saved")
        print(f"      Saved: {output_yaml.name}")

    return {
        "success": True,
        "affine_3x3": affine_3x3,
        "manual": True,
        "shift": [dy, dx],
        "yaml": output_yaml,
        "yaml_path": output_yaml,
    }


def register_nucleus_to_round0(
    iss_zarr_path: Path,
    seg_zarr_path: Path,
    position: str,
    params: dict,
    transforms_dir: Path,
    overlays_dir: Path,
    dataset: "OpsDataset" = None,
    verbose: bool = True,
    spots_round: int = 0,
) -> dict:
    """
    Register nucleus (Round 0, ch0) → spots (Round ``spots_round``, ch1-4).

    ``spots_round`` is the anchor spots round (default 0). Set it >0 when the
    physical round 0 has no usable spots (e.g. a no-incorporation cycle): the
    nucleus is then aligned to that round's spots instead, and the registration
    chain anchors there. The nucleus/DAPI is still taken from round 0.

    Uses DAPI-to-DAPI PCC from original MicroManager data to pre-align nuclei,
    then KDTree matching for final correspondence.
    Falls back to intensity-based nucleus detection if segmentation unavailable.

    Parameters
    ----------
    iss_zarr_path : Path
        Path to stitched ISS zarr (e.g., bc_stitched.zarr).
    seg_zarr_path : Path
        Path to ISS segmentation zarr (if exists).
    position : str
        Position string (e.g., "A/1/0").
    params : dict
        Registration parameters.
    transforms_dir : Path
        Output directory for affine YAML files.
    overlays_dir : Path
        Output directory for visualization overlays.
    dataset : OpsDataset, optional
        Dataset object for accessing original MicroManager data (required for DAPI-to-DAPI PCC).
        If None, falls back to centroid-based PCC.
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict
        Results including affine_3x3, metrics, and output paths.
    """
    if verbose:
        print(f"  Registering nucleus (Round 0, ch0) → spots (Round 0, ch1-4)...")

    t_start = time.time()

    # Extract nucleus centroids from subsampled spatial grid bins (FAST)
    if seg_zarr_path.exists():
        if verbose:
            print(f"    Extracting nucleus centroids from segmentation (subsampled)...")
        t_extract_nuc = time.time()
        try:
            nucleus_centroids = extract_centroids_from_segmentation_subsampled(
                seg_zarr_path,
                position,
                t_idx=0,
                min_area=params["min_cell_area"],
                bins_to_select=params["subsample_bins_to_select"],
                grid_size=params["subsample_grid_size"],
                cache_subdir="1-preprocess/in_situ_sequencing/register/centroid_cache",
            )
            dt_extract_nuc = time.time() - t_extract_nuc
            if verbose:
                print(f"      Found {len(nucleus_centroids)} nucleus centroids ({dt_extract_nuc:.2f}s)")
        except Exception as e:
            if verbose:
                print(f"      WARNING: Failed to load segmentation: {e}")
                print(f"      Falling back to intensity-based nucleus detection...")
            t_extract_nuc = time.time()
            nucleus_centroids = extract_spots_from_intensity_subsampled(
                iss_zarr_path,
                position,
                t_idx=0,
                channel_indices=[0],  # Nucleus channel
                threshold=params["nucleus_threshold"],
                min_distance=5,  # Larger min distance for nuclei
                bins_to_select=params["subsample_bins_to_select"],
                grid_size=params["subsample_grid_size"],
                cache_subdir="in_situ_sequencing/register/iss_spot_cache",
            )
            dt_extract_nuc = time.time() - t_extract_nuc
            if verbose:
                print(f"      Found {len(nucleus_centroids)} nuclei ({dt_extract_nuc:.2f}s)")
    else:
        if verbose:
            print(f"    Segmentation not found, using intensity-based nucleus detection (subsampled)...")
        t_extract_nuc = time.time()
        nucleus_centroids = extract_spots_from_intensity_subsampled(
            iss_zarr_path,
            position,
            t_idx=0,
            channel_indices=[0],
            threshold=params["nucleus_threshold"],
            min_distance=5,
            bins_to_select=params["subsample_bins_to_select"],
            grid_size=params["subsample_grid_size"],
            cache_subdir="in_situ_sequencing/register/iss_spot_cache",
        )
        dt_extract_nuc = time.time() - t_extract_nuc
        if verbose:
            print(f"      Found {len(nucleus_centroids)} nuclei ({dt_extract_nuc:.2f}s)")

    # Extract spot centroids from the anchor spots round (ch1-4) - subsampled extraction
    if verbose:
        print(f"    Extracting spot centroids from Round {spots_round} (ch1-4, subsampled)...")
    t_extract_spots = time.time()
    spot_centroids = extract_spots_from_intensity_subsampled(
        iss_zarr_path,
        position,
        t_idx=spots_round,
        channel_indices=[1, 2, 3, 4],
        threshold=params["spot_threshold"],
        min_distance=params["spot_min_distance"],
        bins_to_select=params["subsample_bins_to_select"],
        grid_size=params["subsample_grid_size"],
        max_spots_per_bin=params["max_spots_per_bin"],
        cache_subdir="in_situ_sequencing/register/iss_spot_cache",
    )
    dt_extract_spots = time.time() - t_extract_spots
    if verbose:
        print(f"      Found {len(spot_centroids)} spots ({dt_extract_spots:.2f}s)")

    # PRE-REGISTRATION DIAGNOSTICS: Measure spatial overlap and compute PCC
    if verbose:
        print(f"    Pre-registration diagnostics...")

    # Calculate bounding boxes and center offset (for reference)
    nuc_bbox = (nucleus_centroids.min(axis=0), nucleus_centroids.max(axis=0))
    spot_bbox = (spot_centroids.min(axis=0), spot_centroids.max(axis=0))
    nuc_center = nucleus_centroids.mean(axis=0)
    spot_center = spot_centroids.mean(axis=0)
    center_offset = spot_center - nuc_center

    if verbose:
        print(f"      Nucleus bbox: ({nuc_bbox[0][0]:.0f}, {nuc_bbox[0][1]:.0f}) → ({nuc_bbox[1][0]:.0f}, {nuc_bbox[1][1]:.0f})")
        print(f"      Spot bbox:    ({spot_bbox[0][0]:.0f}, {spot_bbox[0][1]:.0f}) → ({spot_bbox[1][0]:.0f}, {spot_bbox[1][1]:.0f})")
        print(f"      Center offset (spots - nucleus): dy={center_offset[0]:.1f}px, dx={center_offset[1]:.1f}px (unreliable due to density mismatch)")

    # Compute PCC using DAPI-to-DAPI alignment from original MicroManager data
    # This measures the microscope stage shift between pre-nuclei and first ISS round
    if verbose:
        print(f"    Computing DAPI-to-DAPI PCC from original MicroManager data...")
    t_pcc = time.time()

    # Use original unconverted data to measure DAPI offset
    if dataset is None:
        raise ValueError("dataset parameter is required for DAPI-to-DAPI PCC computation")
    pcc_shift, pcc_error, pcc_quality = compute_dapi_to_dapi_pcc_from_original(
        dataset, position, overlays_dir=overlays_dir, verbose=verbose
    )

    dt_pcc = time.time() - t_pcc

    if verbose:
        print(f"      Total PCC computation time: {dt_pcc:.2f}s")
        print(f"      Final PCC shift (nucleus→spots): dy={pcc_shift[0]:.2f}px, dx={pcc_shift[1]:.2f}px")
        print(f"      PCC error: {pcc_error:.4f}")
        if pcc_quality["bimodal_filtered"]:
            print(f"      NOTE: Bimodal tiles detected — used {pcc_quality['n_tiles_used']}/{pcc_quality['n_tiles']} signal tiles")
        print(f"      Tile shift std (filtered): dy={pcc_quality['shift_std_filtered'][0]:.2f}px, dx={pcc_quality['shift_std_filtered'][1]:.2f}px")

    # --- Quality gate thresholds ---
    # Nucleus→spots matches at a different density than round-to-round (nuclei and
    # spots don't correspond 1:1), so use the lenient nucleus tolerances rather than
    # the strict round-pair gate — a ~50% inlier ratio is normal/healthy here.
    min_matches = 50
    max_residual = params.get("residual_threshold_nucleus", 25.0)  # px (round-pairs use ~8)
    min_inlier_ratio = 0.4  # nucleus↔spot density mismatch → far below round-pair 88%

    # Determine PCC pre-alignment strategies to try.
    # pcc_error measures image similarity (are images identical?), NOT shift confidence.
    # Images from different rounds often look different (intensity/staining), giving
    # error=1.0 even when PCC found the correct shift.  Use tile-to-tile shift std
    # as the real reliability signal: consistent shifts across tiles = PCC is trustworthy.
    shift_std_full = pcc_quality["shift_std_filtered"]  # full-res pixels
    shift_is_consistent = shift_std_full[0] < 5.0 and shift_std_full[1] < 5.0
    pcc_unreliable = pcc_quality["bimodal_filtered"] or not shift_is_consistent
    if pcc_unreliable:
        strategies = [
            ("no_pcc", "PCC unreliable (shift std dy={:.1f}px dx={:.1f}px) — matching on raw centroids".format(
                shift_std_full[0], shift_std_full[1])),
            ("with_pcc", "Retry with PCC pre-alignment"),
        ]
    else:
        strategies = [
            ("with_pcc", "PCC-aligned matching (shift std dy={:.1f}px dx={:.1f}px)".format(
                shift_std_full[0], shift_std_full[1])),
            ("no_pcc", "Retry without PCC pre-alignment"),
        ]

    best_result = None  # (affine_3x3, inliers, metrics, method)

    for strategy_idx, (strategy, strategy_desc) in enumerate(strategies):
        is_retry = strategy_idx > 0
        if is_retry and verbose:
            print(f"\n    ⚠ First attempt failed quality gate — retrying...")

        if strategy == "no_pcc":
            nucleus_centroids_for_matching = nucleus_centroids
            if verbose:
                print(f"    [{strategy_desc}] KDTree matching on raw centroids (radius={params['nucleus_search_radius']}px)...")
        else:
            nucleus_centroids_for_matching = nucleus_centroids - pcc_shift
            if verbose:
                print(f"    [{strategy_desc}] KDTree matching (radius={params['nucleus_search_radius']}px)...")

        if verbose:
            print(f"      {len(nucleus_centroids_for_matching)} nuclei → {len(spot_centroids)} spots")

        t_match = time.time()
        nuc_idx, spot_idx = kdtree_matching(
            nucleus_centroids_for_matching,
            spot_centroids,
            max_distance=params["nucleus_search_radius"],
        )

        if len(nuc_idx) > 0:
            matched_nuc = nucleus_centroids_for_matching[nuc_idx]
            matched_spots_arr = spot_centroids[spot_idx]
            match_distances = np.linalg.norm(matched_nuc - matched_spots_arr, axis=1)
        else:
            matched_spots_arr = np.array([])
            match_distances = np.array([])

        dt_match = time.time() - t_match

        if verbose:
            if len(nuc_idx) > 0:
                print(f"      Found {len(nuc_idx)} matches ({dt_match:.2f}s)")
                print(f"      Match rate: {100*len(nuc_idx)/min(len(nucleus_centroids), len(spot_centroids)):.1f}%")
                print(f"      Distance: {np.mean(match_distances):.2f}px ± {np.std(match_distances):.2f}px")
                print(f"      Distance range: [{np.min(match_distances):.1f}, {np.max(match_distances):.1f}]px")
            else:
                print(f"      Found 0 matches - nucleus and spots may not overlap!")

        # RANSAC refinement on matched pairs (using ORIGINAL nucleus centroids)
        matched_nucleus_original = nucleus_centroids[nuc_idx]
        matched_spots_arr = spot_centroids[spot_idx] if len(nuc_idx) > 0 else np.array([])

        if len(nuc_idx) >= params["min_samples"]:
            if verbose:
                print(f"    RANSAC refinement on {len(nuc_idx)} matched pairs...")
            t_ransac = time.time()
            attempt_affine, attempt_inliers, ransac_metrics = estimate_affine_ransac(
                matched_nucleus_original,
                matched_spots_arr,
                params["min_samples"],
                params["residual_threshold_nucleus"],
                params["max_trials"],
                params["stop_probability"],
                params["transform_type"],
            )
            dt_ransac = time.time() - t_ransac

            if verbose:
                print(f"      RANSAC: {ransac_metrics['n_inliers']}/{ransac_metrics['n_matches']} inliers ({dt_ransac:.2f}s)")
                print(f"      Inlier ratio: {ransac_metrics['inlier_ratio']:.2%}")
                print(f"      Residual: {ransac_metrics['residual_mean']:.2f} ± {ransac_metrics['residual_std']:.2f} px")

            attempt_method = f"{strategy}_ransac"
            attempt_metrics = {
                **ransac_metrics,
                "method": attempt_method,
                "pcc_tiles_used": pcc_quality["n_tiles_used"],
                "pcc_tiles_total": pcc_quality["n_tiles"],
                "pcc_bimodal_filtered": pcc_quality["bimodal_filtered"],
                "pcc_error": float(pcc_error),
                "strategy": strategy,
            }
        else:
            if verbose:
                print(f"    Too few matches for RANSAC ({len(nuc_idx)})")
            # Can't produce a valid result — skip to next strategy
            continue

        # Check quality gate for this attempt
        attempt_errors = []
        if attempt_metrics["n_matches"] < min_matches:
            attempt_errors.append(f"Too few matches: {attempt_metrics['n_matches']}")
        if attempt_metrics["residual_mean"] > max_residual:
            attempt_errors.append(f"Residual too high: {attempt_metrics['residual_mean']:.1f}px")
        if attempt_metrics.get("inlier_ratio", 1.0) < min_inlier_ratio:
            attempt_errors.append(f"Inlier ratio too low: {attempt_metrics['inlier_ratio']:.1%}")

        # Iterative refinement: if quality gate failed but result is borderline,
        # transform nuclei using current affine, re-match at tighter radius, refit.
        # This cleans up residual by finding better correspondences.
        if attempt_errors and attempt_metrics["residual_mean"] < max_residual * 2.0 and attempt_metrics.get("inlier_ratio", 0) > 0.75:
            refine_radius = min(params["nucleus_search_radius"], max_residual * 2)
            for refine_iter in range(3):
                # Transform all nucleus centroids using current affine
                nuc_h = np.column_stack([nucleus_centroids, np.ones(len(nucleus_centroids))])
                nuc_warped = (attempt_affine @ nuc_h.T).T[:, :2]

                # Re-match with tighter radius
                ref_nuc_idx, ref_spot_idx = kdtree_matching(
                    nuc_warped, spot_centroids, max_distance=refine_radius,
                )
                if len(ref_nuc_idx) < params["min_samples"]:
                    break

                # Refit on original (untransformed) nucleus positions
                ref_matched_nuc = nucleus_centroids[ref_nuc_idx]
                ref_matched_spots = spot_centroids[ref_spot_idx]
                ref_affine, ref_inliers, ref_metrics = estimate_affine_ransac(
                    ref_matched_nuc, ref_matched_spots,
                    params["min_samples"], params["residual_threshold_nucleus"],
                    params["max_trials"], params["stop_probability"],
                    params["transform_type"],
                )

                improved = ref_metrics["residual_mean"] < attempt_metrics["residual_mean"]
                if verbose:
                    print(f"    Iterative refinement [{refine_iter+1}]: "
                          f"{ref_metrics['n_inliers']}/{ref_metrics['n_matches']} inliers, "
                          f"residual {ref_metrics['residual_mean']:.2f}px, "
                          f"inlier ratio {ref_metrics['inlier_ratio']:.1%}"
                          f"{' ✓' if improved else ' (no improvement)'}")

                if not improved:
                    break

                attempt_affine = ref_affine
                attempt_inliers = ref_inliers
                attempt_metrics = {
                    **ref_metrics,
                    "method": f"{strategy}_ransac_refined",
                    "pcc_tiles_used": pcc_quality["n_tiles_used"],
                    "pcc_tiles_total": pcc_quality["n_tiles"],
                    "pcc_bimodal_filtered": pcc_quality["bimodal_filtered"],
                    "pcc_error": float(pcc_error),
                    "strategy": strategy,
                }
                matched_nucleus_original = ref_matched_nuc
                matched_spots_arr = ref_matched_spots
                refine_radius = max(ref_metrics["residual_mean"] * 1.5, 5.0)

            # Re-check quality gate after refinement
            attempt_errors = []
            if attempt_metrics["n_matches"] < min_matches:
                attempt_errors.append(f"Too few matches: {attempt_metrics['n_matches']}")
            if attempt_metrics["residual_mean"] > max_residual:
                attempt_errors.append(f"Residual too high: {attempt_metrics['residual_mean']:.1f}px")
            if attempt_metrics.get("inlier_ratio", 1.0) < min_inlier_ratio:
                attempt_errors.append(f"Inlier ratio too low: {attempt_metrics['inlier_ratio']:.1%}")

        if not attempt_errors:
            # Passed quality gate — use this result
            if verbose and is_retry:
                print(f"    ✓ Retry succeeded!")
            best_result = (attempt_affine, attempt_inliers, attempt_metrics, matched_nucleus_original, matched_spots_arr)
            break
        else:
            if verbose:
                print(f"    Quality gate failed: {'; '.join(attempt_errors)}")
            # Keep as candidate if it's the best we've seen
            if best_result is None:
                best_result = (attempt_affine, attempt_inliers, attempt_metrics, matched_nucleus_original, matched_spots_arr)
            elif attempt_metrics["residual_mean"] < best_result[2]["residual_mean"]:
                best_result = (attempt_affine, attempt_inliers, attempt_metrics, matched_nucleus_original, matched_spots_arr)

    # Unpack best result
    if best_result is None:
        raise RuntimeError(
            "Nucleus → Round 0 registration FAILED: no strategy produced enough matches for RANSAC"
        )
    affine_3x3, inliers, metrics, matched_nucleus_original, matched_spots = best_result

    # POST-REGISTRATION DIAGNOSTICS: Measure improvement
    if len(matched_nucleus_original) > 0:
        # Transform original nucleus centroids to spot space using final composed transform
        nuc_homogeneous = np.column_stack([matched_nucleus_original, np.ones(len(matched_nucleus_original))])
        nuc_transformed = (affine_3x3 @ nuc_homogeneous.T).T[:, :2]

        # Measure distances before/after for all matches
        distances_before = np.linalg.norm(matched_nucleus_original - matched_spots, axis=1)
        distances_after = np.linalg.norm(nuc_transformed - matched_spots, axis=1)

        # Calculate improvement
        mean_before = np.mean(distances_before)
        mean_after = np.mean(distances_after)
        improvement = mean_before - mean_after
        improvement_pct = 100 * improvement / mean_before if mean_before > 0 else 0

        # Store diagnostics in metrics
        metrics["post_reg_residual_mean"] = float(mean_after)
        metrics["post_reg_improvement_pct"] = float(improvement_pct)

        if verbose:
            print(f"    Post-registration diagnostics...")
            print(f"      All matches ({len(matched_nucleus_original)}):")
            print(f"        Before: {mean_before:.2f}px ± {np.std(distances_before):.2f}px")
            print(f"        After:  {mean_after:.2f}px ± {np.std(distances_after):.2f}px")
            print(f"        Improvement: {improvement:.2f}px ({improvement_pct:.1f}%)")

            # Also show inlier-only stats
            inlier_distances_before = distances_before[inliers]
            inlier_distances_after = distances_after[inliers]
            print(f"      Inliers only ({metrics['n_inliers']}):")
            print(f"        Before: {np.mean(inlier_distances_before):.2f}px ± {np.std(inlier_distances_before):.2f}px")
            print(f"        After:  {np.mean(inlier_distances_after):.2f}px ± {np.std(inlier_distances_after):.2f}px")

    # --- Final quality gate: fail hard if best result is still bad ---
    errors = []
    if metrics["n_matches"] < min_matches:
        errors.append(
            f"Too few matches: {metrics['n_matches']} (minimum {min_matches})"
        )
    if metrics["n_matches"] > 0 and metrics["residual_mean"] > max_residual:
        errors.append(
            f"Residual too high: {metrics['residual_mean']:.1f}px (maximum {max_residual}px)"
        )
    if metrics.get("inlier_ratio", 1.0) < min_inlier_ratio:
        errors.append(
            f"Inlier ratio too low: {metrics['inlier_ratio']:.1%} (minimum {min_inlier_ratio:.0%})"
        )

    if errors:
        error_msg = (
            f"Nucleus → Round 0 registration FAILED quality check after all strategies "
            f"(best method={metrics.get('method', 'unknown')}):\n"
            + "\n".join(f"  - {e}" for e in errors)
            + f"\n  Metrics: {metrics}"
        )
        raise RuntimeError(error_msg)

    # --- Candidate-overlap selection + grid refinement ---
    # RANSAC residual is measured on inliers only; if RANSAC converged to a
    # self-consistent but globally wrong solution (e.g. wrong sign in dy),
    # the residual looks fine but spots don't actually land on nuclei.
    # Score each candidate by the actual spot→nearest-warped-nucleus median
    # distance over ALL spots, then grid-refine the winner.
    from scipy.spatial import cKDTree as _cKDTree

    def _spot_to_warpednuc_median(a3: np.ndarray) -> float:
        nuc_h = np.column_stack([nucleus_centroids, np.ones(len(nucleus_centroids))])
        warped = (a3 @ nuc_h.T).T[:, :2]
        d, _ = _cKDTree(warped).query(spot_centroids, k=1)
        return float(np.median(d))

    pcc_only_affine = np.eye(3)
    pcc_only_affine[0, 2] = -pcc_shift[0]
    pcc_only_affine[1, 2] = -pcc_shift[1]

    candidates = {
        # Order matters when scores tie: prefer non-trivial transforms.
        "ransac":   affine_3x3,
        "pcc_only": pcc_only_affine,
        "identity": np.eye(3),
    }
    cand_scores = {n: _spot_to_warpednuc_median(a) for n, a in candidates.items()}
    best_cand = min(cand_scores, key=cand_scores.get)
    if verbose:
        print(f"    [Candidate overlap] median spot→warped-nuc dist: "
              + ", ".join(f"{n}={v:.1f}px" for n, v in cand_scores.items())
              + f" → keeping '{best_cand}'")

    # If RANSAC was clearly worse than a trivial fallback, swap.
    if best_cand != "ransac" and cand_scores[best_cand] < cand_scores["ransac"] - 1.0:
        affine_3x3 = candidates[best_cand].copy()
        metrics["method"] = f"{metrics.get('method','unknown')}_overridden_by_{best_cand}"
        metrics["candidate_overlap_scores"] = cand_scores
        metrics["candidate_chosen"] = best_cand

    # Coarse-then-fine translation grid search to escape RANSAC's wrong minimum.
    # Coarse: ±50px step 10 (121 evals); fine: ±8px step 2 around best.
    #
    # GATE: this objective measures spot→nearest-nucleus-CENTROID distance, whose
    # floor (~nucleus radius) is set by spot/nucleus geometry, not registration
    # error. When identity scores about as well as the best candidate, the metric
    # can't discriminate alignment quality, so refining against it just chases
    # spatial-density noise and corrupts an already-good transform (observed:
    # RANSAC residual ~3.7px / 100% inliers, yet refine pushed the anchor 5-8px
    # off in inconsistent directions per well). Only refine when the objective is
    # discriminative (identity meaningfully worse than the best candidate) AND the
    # gain clears a real threshold — otherwise trust the RANSAC/PCC fit.
    GRID_REFINE_MIN_GAIN_PX = 3.0
    best_overlap = min(cand_scores.values())
    metric_discriminative = (cand_scores["identity"] - best_overlap) >= 1.0

    base_score = _spot_to_warpednuc_median(affine_3x3)
    best_dy, best_dx, best_score = 0, 0, base_score
    if metric_discriminative:
        for dy in range(-50, 51, 10):
            for dx in range(-50, 51, 10):
                perturb = np.eye(3); perturb[0, 2] = dy; perturb[1, 2] = dx
                s = _spot_to_warpednuc_median(perturb @ affine_3x3)
                if s < best_score:
                    best_score, best_dy, best_dx = s, dy, dx
        if best_score < base_score:
            for dy in range(best_dy - 8, best_dy + 9, 2):
                for dx in range(best_dx - 8, best_dx + 9, 2):
                    perturb = np.eye(3); perturb[0, 2] = dy; perturb[1, 2] = dx
                    s = _spot_to_warpednuc_median(perturb @ affine_3x3)
                    if s < best_score:
                        best_score, best_dy, best_dx = s, dy, dx

    applied_gain = base_score - best_score
    if metric_discriminative and applied_gain >= GRID_REFINE_MIN_GAIN_PX:
        perturb = np.eye(3); perturb[0, 2] = best_dy; perturb[1, 2] = best_dx
        affine_3x3 = perturb @ affine_3x3
        metrics["method"] = f"{metrics.get('method','unknown')}_grid_refined"
        metrics["grid_refine_dy"] = int(best_dy)
        metrics["grid_refine_dx"] = int(best_dx)
    else:
        # Reject the perturbation — keep the RANSAC/PCC transform untouched.
        best_dy, best_dx, best_score = 0, 0, base_score

    metrics["overlap_score_median_px"] = float(best_score)
    if verbose:
        if not metric_discriminative:
            print(f"    [Grid refine] skipped — overlap metric non-discriminative "
                  f"(identity {cand_scores['identity']:.1f}px ≈ best {best_overlap:.1f}px); "
                  f"trusting RANSAC/PCC fit")
        else:
            status = ("applied" if applied_gain >= GRID_REFINE_MIN_GAIN_PX
                      else f"rejected (gain {applied_gain:.2f}px < {GRID_REFINE_MIN_GAIN_PX:.0f}px)")
            print(f"    [Grid refine] median spot→warped-nuc: {base_score:.2f}px → {best_score:.2f}px "
                  f"(perturb dy={best_dy}, dx={best_dx}) [{status}]")

    # Extract final transform parameters
    final_trans = affine_3x3[:2, 2]
    final_rot = np.arctan2(affine_3x3[1, 0], affine_3x3[0, 0])
    final_scale_x = np.sqrt(affine_3x3[0, 0] ** 2 + affine_3x3[1, 0] ** 2)
    final_scale_y = np.sqrt(affine_3x3[0, 1] ** 2 + affine_3x3[1, 1] ** 2)

    if verbose:
        print(f"    Final transform (stored with consistent sign convention):")
        print(f"      Translation: dy={final_trans[0]:.1f}px, dx={final_trans[1]:.1f}px (both same sign for scipy)")
        print(f"      Rotation: {np.degrees(final_rot):.2f}°")
        print(f"      Scale: ({final_scale_x:.4f}, {final_scale_y:.4f})")

    # Save affine to YAML
    affine_4x4 = affine_3x3_to_4x4_zyx(affine_3x3)
    affine_4x4_inv = np.linalg.inv(affine_4x4)
    output_yaml = transforms_dir / "nucleus_to_round0.yml"
    save_affine_to_yaml(affine_4x4_inv, output_yaml)

    if verbose:
        print(f"    Saved transform to: {output_yaml.name}")
        print(f"      Transform direction: spots→nucleus (to move spots to nucleus anchor)")
        print(f"      Translation: dy={affine_4x4_inv[1,3]:.2f}, dx={affine_4x4_inv[2,3]:.2f}px")

    # Generate visualization overlay (nucleus ch0 vs spots ch1-4)
    if verbose:
        print(f"    Generating overlay visualization...")
    overlay_path = overlays_dir / "spots_to_nucleus_overlay.png"
    try:
        # Specialized overlay showing spots warped to nucleus
        # Invert the nucleus→spots transform to get spots→nucleus
        affine_spots_to_nuc = np.linalg.inv(affine_3x3)
        create_nucleus_to_spots_overlay(
            iss_zarr_path, position, affine_spots_to_nuc, overlay_path,
            crop_size=500, n_crops=6
        )
        if verbose:
            print(f"    Saved overlay: {overlay_path.name}")
    except Exception as e:
        if verbose:
            print(f"    WARNING: Overlay generation failed: {e}")

    dt_total = time.time() - t_start
    if verbose:
        print(f"    Total time: {dt_total:.2f}s\n")

    results = {
        "affine_3x3": affine_3x3,
        "affine_4x4": affine_4x4,
        "metrics": metrics,
        "yaml": output_yaml,
        "overlay": overlay_path if overlay_path.exists() else None,
        "n_nucleus_centroids": len(nucleus_centroids),
        "n_spot_centroids": len(spot_centroids),
    }

    return results


def auto_register_iss_rounds(
    experiment: str,
    well,
    spot_threshold: float = 400,
    nucleus_threshold: float = 200,
    transform_type: str = "similarity",
    apply_transforms: bool = True,
    create_overlays: bool = False,
    skip_nucleus_registration: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Main entry point: register all ISS rounds for a well.

    Performs:
    1. Nucleus (Round 0, ch0) → Spots (Round 0, ch1-4) registration (optional)
    2. Sequential round-to-round registration (R0→R1, R1→R2, ..., R8→R9)
    3. Composes cumulative affines (all rounds → Round 0 coordinate system)
    4. Applies transforms and writes registered output zarr
    5. Generates QA overlays and metrics CSV

    Parameters
    ----------
    experiment : str
        Experiment name (e.g., "ops0035_20250501").
    well : int
        Well number (1, 2, or 3).
    spot_threshold : float
        Minimum spot intensity for detection.
    nucleus_threshold : float
        Minimum nucleus intensity (fallback if no segmentation).
    transform_type : str
        Transform type: "similarity", "euclidean", or "affine".
    apply_transforms : bool
        Apply transforms and write registered output zarr.
    create_overlays : bool
        Generate QA overlay images (NYI).
    skip_nucleus_registration : bool
        Skip nucleus→spots registration (use identity). Set to True for experiments
        without a pre-DAPI round where nucleus and spots are already aligned.
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict
        Registration results including affines, metrics, and output paths.
    """
    row, col = parse_well(well)  # accept full row/col unit, default row A
    well_token = f"{row}{col}"
    if verbose:
        print(f"\n{'='*80}")
        print(f"ISS Round-to-Round Registration")
        print(f"Experiment: {experiment}, Well: {well_token}")
        print(f"{'='*80}\n")

    t_start_total = time.time()

    # Setup paths with organized subdirectories
    dataset = OpsDataset(experiment)
    position = f"{row}/{col}/0"

    iss_zarr = dataset.store_paths["iss_stitch"]
    seg_zarr = dataset.store_paths["iss_segmentation"]

    # Organized output structure
    register_root = dataset.preprocess_in_situ / "register"
    transforms_dir = register_root / f"transforms/{well_token}"
    overlays_dir = register_root / f"overlays/{well_token}"
    metrics_dir = register_root / "metrics"

    transforms_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Input zarr:  {iss_zarr}")
        print(f"Transforms:  {transforms_dir}")
        print(f"Overlays:    {overlays_dir}")
        print(f"Metrics:     {metrics_dir}\n")

    # Setup parameters
    params = DEFAULT_ISS_PARAMS.copy()
    params["spot_threshold"] = spot_threshold
    params["nucleus_threshold"] = nucleus_threshold
    params["transform_type"] = transform_type

    results = {}

    # Phase 0: Segmentation → Nucleus (always runs first)
    if verbose:
        print(f"[Phase 0/4] Segmentation → Nucleus Registration")
        print(f"-" * 80)

    results["segmentation"] = register_segmentation_to_nucleus(
        iss_zarr, seg_zarr, position,
        transforms_dir, overlays_dir, verbose=verbose
    )

    # Phase 1: Nucleus → Round 0 (optional - skip if no pre-DAPI round)
    if skip_nucleus_registration:
        print(f"[Phase 1/4] Nucleus → Round 0 Registration: SKIPPED")
        print(f"-" * 80)
        if verbose:
            print(f"  Skipping nucleus→spots registration (no pre-DAPI round)")
            print(f"  Using identity transform (nucleus and spots already aligned)")
        # Create dummy results with identity transform
        results["nucleus"] = {
            "affine_3x3": np.eye(3),
            "affine_4x4": np.eye(4),
            "metrics": {
                "n_matches": 0,
                "n_inliers": 0,
                "inlier_ratio": 0.0,
                "residual_mean": 0.0,
                "residual_std": 0.0,
                "residual_max": 0.0,
            },
            "yaml": None,
            "overlay": None,
            "n_nucleus_centroids": 0,
            "n_spot_centroids": 0,
        }
        print()
    else:
        print(f"[Phase 1/4] Nucleus → Round 0 Registration")
        print(f"-" * 80)
        results["nucleus"] = register_nucleus_to_round0(
            iss_zarr, seg_zarr, position, params, transforms_dir, overlays_dir,
            dataset=dataset, verbose=verbose
        )
        print()

    # Phase 2: Round-to-round sequential registration
    print(f"[Phase 2/4] Round-to-Round Sequential Registration")
    print(f"-" * 80)

    affines_cumulative = {}

    # 1. Establish Anchor Chain
    # Chain: Spots -> Nucleus -> Segmentation (Anchor)

    # Step A: Get T_nuc_to_seg (Nucleus -> Segmentation)
    # If segmentation registration failed or wasn't run, this is Identity
    if results.get("segmentation", {}).get("success"):
        affine_nuc_to_seg = results["segmentation"]["affine_3x3"]
        affines_cumulative[-2] = np.eye(3)  # Key -2: Segmentation is the global anchor (Identity)
        if verbose:
            print(f"  Anchor: Segmentation")
            print(f"  Nucleus→Segmentation: dy={affine_nuc_to_seg[0, 2]:.2f}, dx={affine_nuc_to_seg[1, 2]:.2f}")
    else:
        affine_nuc_to_seg = np.eye(3)
        if verbose:
            print(f"  Anchor: Nucleus (Segmentation not available)")

    # Step B: Get T_spots_to_nuc (Spots -> Nucleus)
    if "nucleus" in results and results["nucleus"].get("affine_3x3") is not None:
        affine_spots_to_nuc = results["nucleus"]["affine_3x3"]
        if verbose:
            print(f"  Spots→Nucleus: dy={affine_spots_to_nuc[0, 2]:.2f}, dx={affine_spots_to_nuc[1, 2]:.2f}")
    else:
        affine_spots_to_nuc = np.eye(3)

    # Step C: Store Cumulative Transforms relative to Anchor

    # Key -1: Nucleus (Round 0, ch0) -> Anchor
    affines_cumulative[-1] = affine_nuc_to_seg

    # Key 0: Round 0 Spots (Round 0, ch1-4) -> Anchor
    # affine_spots_to_nuc is spots→nucleus (stored with consistent sign convention)
    # Use it directly for composition
    # T_spots0_to_anchor = T_nuc_to_seg @ T_spots_to_nuc
    affines_cumulative[0] = affine_nuc_to_seg @ affine_spots_to_nuc

    if verbose:
        t0 = affines_cumulative[0]
        print(f"  Cumulative Round 0 (Spots)→Anchor: dy={t0[0, 2]:.2f}, dx={t0[1, 2]:.2f}\n")

    # Pre-extract spots for ALL 10 rounds using ProcessPool.
    # Spot detection (peak_local_max) holds the GIL — threads can't parallelize it.
    # ProcessPool gives each round its own GIL → true parallel spot detection.
    if verbose:
        print(f"\n[Pre-extraction] Extracting spots for 10 rounds (ProcessPool, 10 workers)...")
    t_preextract = time.time()
    from concurrent.futures import ProcessPoolExecutor as _ProcPool
    from concurrent.futures import ThreadPoolExecutor as _ThreadPool

    spot_args = [
        (str(iss_zarr), position, r, [1, 2, 3, 4],
         params["spot_threshold"], params["spot_min_distance"],
         params["subsample_bins_to_select"], params["subsample_grid_size"],
         params["max_spots_per_bin"])
        for r in range(10)
    ]

    with _ProcPool(max_workers=10) as proc_pool:
        round_spots = list(proc_pool.map(_extract_spots_worker, spot_args))

    if verbose:
        for r, spots in enumerate(round_spots):
            print(f"  Round {r}: {len(spots)} spots")
        print(f"  Pre-extraction total: {time.time()-t_preextract:.1f}s\n")

    # Register 9 round-pairs sequentially (each uses Pool(31) for graph matching,
    # so running them in parallel causes massive process contention).
    # With precomputed spots, each pair takes ~1.8s — total ~16s.
    for i in range(9):
        if verbose:
            print(f"\nRound {i+1} → Round {i} ({i+1}/9)")
            print(f"-" * 40)

        result = register_round_pair(
            iss_zarr, position, round_i=i, round_j=i + 1, params=params,
            transforms_dir=transforms_dir, overlays_dir=overlays_dir, verbose=verbose,
            precomputed_source_spots=round_spots[i + 1],
            precomputed_target_spots=round_spots[i],
        )
        results[f"round_{i+1}_to_{i}"] = result

        affines_cumulative[i + 1] = affines_cumulative[i] @ result["affine_3x3"]

        cumulative_yaml = transforms_dir / f"round{i+1}_to_round0_cumulative.yml"
        affine_4x4 = affine_3x3_to_4x4_zyx(affines_cumulative[i + 1])
        save_affine_to_yaml(np.linalg.inv(affine_4x4), cumulative_yaml)

        if verbose:
            cumul_trans = affines_cumulative[i + 1][:2, 2]
            print(f"    Cumulative translation (R{i+1}→R0): dy={cumul_trans[0]:.1f}, dx={cumul_trans[1]:.1f}")

    # Generate drift trajectory plot
    if verbose:
        print(f"\nGenerating drift trajectory plot...")
    drift_plot_path = overlays_dir / "drift_trajectory.png"
    try:
        create_drift_trajectory_plot(affines_cumulative, drift_plot_path)
        if verbose:
            print(f"  Saved: {drift_plot_path.name}")
    except Exception as e:
        if verbose:
            print(f"  WARNING: Drift plot failed: {e}")

    # Pre-compute tophat+normalized crops for all rounds×positions ONCE with 32 threads.
    # white_tophat releases the GIL → 15.6x speedup (49s → 3s for 60 crops).
    # Cache is reused by all 3 overlay calls, avoiding redundant computation.
    if verbose:
        print(f"\nPre-computing tophat-filtered crops for overlays (32 threads)...")
    t_precompute = time.time()

    from concurrent.futures import ThreadPoolExecutor as _TophatPool
    from skimage.morphology import white_tophat as _wth
    from skimage.morphology import disk as _disk
    from skimage.exposure import adjust_gamma as _ag

    _selem = _disk(20)
    crop_size_ov = 500
    n_crops_ov = 6
    rounds_ov = sorted([r for r in affines_cumulative.keys() if r >= 0])
    spot_channels = [1, 2, 3, 4]

    # Get image dimensions + crop positions
    import zarr as _zarr_ov
    _arr_ov = _zarr_ov.open(str(iss_zarr / position / "0"), mode="r")
    Y_ov, X_ov = _arr_ov.shape[-2], _arr_ov.shape[-1]
    margin_y = int(Y_ov * 0.25)
    margin_x = int(X_ov * 0.25)
    inner_h = Y_ov - 2 * margin_y
    inner_w = X_ov - 2 * margin_x
    y_pos = [margin_y, margin_y, margin_y + inner_h//2 - crop_size_ov//2,
             margin_y + inner_h//2 - crop_size_ov//2, margin_y + inner_h - crop_size_ov,
             margin_y + inner_h - crop_size_ov]
    x_pos = [margin_x, margin_x + inner_w - crop_size_ov, margin_x,
             margin_x + inner_w - crop_size_ov, margin_x, margin_x + inner_w - crop_size_ov]
    crop_positions_ov = [(y_pos[i], x_pos[i]) for i in range(min(n_crops_ov, len(y_pos)))]

    def _process_crop(args):
        r, ch_list, y_start, x_start = args
        crop = np.sum([_arr_ov[r, ch, 0, y_start:y_start+crop_size_ov,
                        x_start:x_start+crop_size_ov] for ch in ch_list], axis=0)
        crop = np.array(crop, dtype=np.float32)
        crop_filtered = _wth(crop, _selem)
        bg_floor = np.percentile(crop_filtered, 50)
        crop_sub = np.clip(crop_filtered - bg_floor, 0, None)
        p_max = np.percentile(crop_sub, 99.9)
        if p_max == 0: p_max = 1
        crop_norm = np.clip(crop_sub / p_max, 0, 1)
        return _ag(crop_norm, 1.5)

    # Pre-compute spot crops (10 rounds × 6 positions) + nucleus crops (6 positions)
    tasks = [(r, spot_channels, y, x) for y, x in crop_positions_ov for r in rounds_ov]
    nucleus_tasks = [(0, [0], y, x) for y, x in crop_positions_ov]
    all_tasks = tasks + nucleus_tasks

    with _TophatPool(max_workers=32) as tp:
        results_flat = list(tp.map(_process_crop, all_tasks))

    # Build cache: (crop_idx, round) → normalized image for spots
    precomputed_crops = {}
    idx = 0
    for crop_idx, (y, x) in enumerate(crop_positions_ov):
        for r in rounds_ov:
            precomputed_crops[(crop_idx, r)] = results_flat[idx]
            idx += 1
    # Add nucleus crops: key = (crop_idx, "nucleus")
    for crop_idx in range(len(crop_positions_ov)):
        precomputed_crops[(crop_idx, "nucleus")] = results_flat[idx]
        idx += 1

    if verbose:
        print(f"  Pre-computed {len(results_flat)} crops in {time.time()-t_precompute:.1f}s")

    # Generate all 3 overlays using pre-computed crops
    if verbose:
        print(f"\nGenerating all-rounds overlay (spots only)...")
    all_rounds_overlay_path = overlays_dir / "all_rounds_overlay.png"
    try:
        create_all_rounds_overlay(
            iss_zarr, position, affines_cumulative, all_rounds_overlay_path,
            crop_size=crop_size_ov, n_crops=n_crops_ov,
            precomputed_crops=precomputed_crops,
        )
        if verbose:
            print(f"  Saved: {all_rounds_overlay_path.name}")
    except Exception as e:
        if verbose:
            print(f"  WARNING: All-rounds overlay failed: {e}")

    if verbose:
        print(f"\nGenerating all-rounds overlay (with nucleus)...")
    all_rounds_nucleus_overlay_path = overlays_dir / "all_rounds_overlay_with_nucleus.png"
    try:
        create_all_rounds_overlay(
            iss_zarr, position, affines_cumulative, all_rounds_nucleus_overlay_path,
            crop_size=crop_size_ov, n_crops=n_crops_ov, include_nucleus=True,
            precomputed_crops=precomputed_crops,
        )
        if verbose:
            print(f"  Saved: {all_rounds_nucleus_overlay_path.name}")
    except Exception as e:
        if verbose:
            print(f"  WARNING: All-rounds+nucleus overlay failed: {e}")

    # Generate all-rounds overlay WITH segmentation (final validation)
    if verbose:
        print(f"\nGenerating all-rounds overlay (with segmentation)...")
    final_registration_path = overlays_dir / "final_registration_with_segmentation.png"
    try:
        create_all_rounds_overlay(
            iss_zarr, position, affines_cumulative, final_registration_path,
            crop_size=crop_size_ov, n_crops=n_crops_ov,
            include_segmentation=True, seg_zarr_path=seg_zarr,
            precomputed_crops=precomputed_crops,
        )
        if verbose:
            print(f"  Saved: {final_registration_path.name}")
    except Exception as e:
        if verbose:
            print(f"  WARNING: All-rounds+segmentation overlay failed: {e}")

    print()

    # Phase 3: Apply transforms and write output
    if apply_transforms:
        print(f"[Phase 3/4] Applying Transforms & Writing Registered Output")
        print(f"-" * 80)

        # aliddell: _v3 is beta/nextflow, fallback to iss_stitch_registered (main) if not present, finally fallback to default path if not defined in experiment.py yet
        output_zarr = dataset.store_paths.get("iss_stitch_registered_v3")
        if output_zarr is None:
            output_zarr = dataset.store_paths.get("iss_stitch_registered")
        if output_zarr is None:
            # Fallback if not defined in experiment.py yet
            output_zarr = dataset.preprocess_in_situ / "stitch" / "bc_stitched_registered.zarr"

        apply_iss_transforms(
            iss_zarr,
            position,
            affines_cumulative,
            output_zarr,
            verbose=verbose,
        )
        print()
    else:
        if verbose:
            print(f"[Phase 3/4] Skipping transform application (apply_transforms=False)\n")

    # Generate QA overlays (optional)
    if create_overlays:
        if verbose:
            print(f"[Phase 4/4] Generating QA Overlays")
            print(f"-" * 80)
        # TODO: Implement overlay generation
        if verbose:
            print("    Overlay generation not yet implemented (TODO)\n")

    # Summary
    dt_total = time.time() - t_start_total
    if verbose:
        print(f"{'='*80}")
        print(f"Registration Complete!")
        print(f"Total time: {dt_total:.1f}s ({dt_total/60:.1f} min)")
        print(f"Output directory: {register_root}")
        if apply_transforms:
            print(f"Registered zarr: {output_zarr}")
        print(f"{'='*80}\n")

    # Save metrics CSV
    metrics_file = metrics_dir / f"registration_metrics_{well_token}.csv"
    _save_metrics_csv_for_well(results, metrics_file, verbose=verbose)

    return results


def _get_worker_count():
    """
    Get optimal number of worker threads for parallel tile processing.

    Returns the number of CPUs allocated by SLURM, or the number of available
    CPUs if not running under SLURM. For I/O-bound zarr writes, using all
    available CPUs is beneficial even though GPU transforms are serialized.
    """
    cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if cpus:
        return int(cpus)
    return len(os.sched_getaffinity(0))


def _process_tile_worker(input_zarr_path, output_zarr_path, position, round_idx, c, y_out, x_out,
                         affine_matrix, tile_size, padding, Y, X,
                         shm_name=None, shm_shape=None, shm_dtype=None):
    """
    Worker function for ProcessPoolExecutor - processes a single tile.

    This function is at module level so it can be pickled for multiprocessing.
    Each process opens its own zarr handles and processes one tile independently.

    When ``shm_name`` is provided, the warped tile is also written into the
    shared-memory ndarray with that name (caller-owned). This lets a single
    parent process collect all tiles into one in-memory (T, C, Z, Y, X)
    float32 array without going through the output zarr — used by the merge
    orchestrator that hands the registered array to detect_spots and
    base_calling without an NFS round-trip.

    Either ``output_zarr_path`` or ``shm_name`` (or both) must be set.
    """
    import time
    from iohub import open_ome_zarr
    from scipy import ndimage as cpu_ndi
    from pathlib import Path

    t_start = time.time()

    # Output tile dimensions
    h_tile = min(tile_size, Y - y_out)
    w_tile = min(tile_size, X - x_out)

    # Input region with padding
    y_in_start = max(0, y_out - padding)
    x_in_start = max(0, x_out - padding)
    y_in_end = min(Y, y_out + h_tile + padding)
    x_in_end = min(X, x_out + w_tile + padding)

    # Open input zarr at position level (skip plate metadata parsing)
    with open_ome_zarr(Path(input_zarr_path) / position, layout="fov", mode="r") as store_in:
        data = store_in.data
        input_crop = np.asarray(data[round_idx, c, 0, y_in_start:y_in_end, x_in_start:x_in_end], dtype=np.float32)

    # Adjust affine matrix for crop position
    crop_out_off = np.array([y_out, x_out])
    crop_in_off = np.array([y_in_start, x_in_start])

    mat = affine_matrix[:, :2]
    off = affine_matrix[:, 2]
    new_offset = mat @ crop_out_off + off - crop_in_off

    # Apply transform (CPU only for ProcessPoolExecutor)
    output_tile = cpu_ndi.affine_transform(
        input_crop,
        mat,
        offset=new_offset,
        output_shape=(h_tile, w_tile),
        order=1,
        mode='constant',
        cval=0.0
    )

    # Write to output zarr if a path was given
    if output_zarr_path is not None:
        with open_ome_zarr(Path(output_zarr_path) / position, layout="fov", mode="r+") as store_out:
            out_array = store_out["0"]
            out_array[round_idx, c, 0, y_out:y_out+h_tile, x_out:x_out+w_tile] = output_tile

    # Write to shared-memory ndarray if a name was given
    if shm_name is not None and shm_shape is not None and shm_dtype is not None:
        from multiprocessing import shared_memory as _shm

        shm = _shm.SharedMemory(name=shm_name)
        try:
            shm_arr = np.ndarray(shm_shape, dtype=np.dtype(shm_dtype), buffer=shm.buf)
            shm_arr[round_idx, c, 0, y_out:y_out+h_tile, x_out:x_out+w_tile] = output_tile
            # Drop the numpy view before close() to avoid "active references" warning
            shm_arr = None  # noqa: F841
        finally:
            shm.close()

    t_elapsed = time.time() - t_start
    return (y_out, x_out, t_elapsed)


def apply_iss_transforms(
    input_zarr: Path,
    position: str,
    affines_cumulative: dict,
    output_zarr: Path | None = None,
    tile_size: int = 4096,
    padding: int = 256,
    verbose: bool = True,
    *,
    output_shm_name: str | None = None,
    output_shm_shape: tuple | None = None,
):
    """
    Apply transforms using tiled CPU processing.

    Strategy:
    1. Iterate over output grid (tiles).
    2. For each tile, calculate the corresponding input region (with padding for drift).
    3. Apply transform with corrected coordinate offset.
    4. Write each warped tile to the output zarr and/or a caller-supplied
       shared-memory ndarray.

    If segmentation exists (-2 in affines_cumulative), segmentation is the global anchor.
    Otherwise, nucleus (Round 0, Channel 0) is the anchor and remains untransformed.
    All other channels are transformed to align with the anchor coordinate system.

    Parameters
    ----------
    input_zarr : Path
        Input stitched zarr (e.g., bc_stitched.zarr).
    position : str
        Position string (e.g., "A/1/0").
    affines_cumulative : dict
        Dictionary mapping round index to cumulative 3x3 affine (round_i → nucleus).
    output_zarr : Path or None
        Output registered zarr path. May be ``None`` if writing only to a
        shared-memory ndarray (``output_shm_name`` set).
    tile_size : int
        Output tile size (matches zarr chunk size for speed).
    padding : int
        Padding around input crop (enough to cover max drift).
    verbose : bool
        Print progress messages.
    output_shm_name : str, optional
        Name of a caller-allocated ``multiprocessing.shared_memory`` block
        backing a float32 ndarray of shape ``output_shm_shape``. When set,
        every warped tile is also written into that shared array. The merge
        orchestrator uses this to capture the registered (T, C, Z, Y, X)
        array in host RAM without a zarr round-trip, and hand it to
        detect_spots / base_calling.
    output_shm_shape : tuple, optional
        Shape of the shared-memory ndarray, must equal (T, C, Z, Y, X) of the
        input zarr. Required when ``output_shm_name`` is set.
    """
    if output_zarr is None and output_shm_name is None:
        raise ValueError(
            "apply_iss_transforms needs at least one of output_zarr or output_shm_name"
        )
    if output_shm_name is not None and output_shm_shape is None:
        raise ValueError(
            "output_shm_shape is required when output_shm_name is provided"
        )

    if verbose:
        print(f"  Using CPU with Tiled Processing (ProcessPoolExecutor for true parallelism)")
        print(f"  Tile size: {tile_size}px, Padding: {padding}px")
        print(f"  Input:  {input_zarr}")
        print(f"  Output zarr: {output_zarr if output_zarr is not None else '(skipped)'}")
        if output_shm_name is not None:
            print(f"  Output shm:  {output_shm_name} shape={output_shm_shape}")

    import time
    t_start = time.time()

    # Open input store at position level (skip plate metadata parsing)
    with open_ome_zarr(input_zarr / position, layout="fov", mode="r") as store_in:
        data = store_in.data
        T, C, Z, Y, X = data.shape
        channel_names = store_in.channel_names

        if verbose:
            print(f"  Shape: T={T}, C={C}, Z={Z}, Y={Y}, X={X}")

    # Validate shm shape against zarr shape if both are in use
    if output_shm_name is not None and tuple(output_shm_shape) != (T, C, Z, Y, X):
        raise ValueError(
            f"output_shm_shape {output_shm_shape} does not match input zarr "
            f"shape (T={T}, C={C}, Z={Z}, Y={Y}, X={X})"
        )

    # Open output store in append mode (store must be pre-created with all positions
    # before parallel finalization jobs are submitted to avoid race conditions)
    from cyclops_utils.io.zarr_utils import ensure_position_array

    if output_zarr is not None:
        if not output_zarr.exists():
            # Fallback: create store if not pre-created (e.g. single-well mode)
            if verbose:
                print(f"  Creating new zarr: {output_zarr}")
            store_mode = "w"
            with open_ome_zarr(output_zarr, layout="hcs", mode=store_mode, channel_names=channel_names) as store_out:
                ensure_position_array(
                    store_out,
                    position,
                    shape=(T, C, Z, Y, X),
                    chunk_size=(1, 1, 1, tile_size, tile_size),
                    dtype=np.float32,
                    scale=[1, 1, 1, 1, 1],
                )
        else:
            if verbose:
                print(f"  Opening pre-created zarr: {output_zarr}")

    # If we're writing to shm, attach a parent-side view so the identity-shortcut
    # branch (no actual warp needed) can fill tiles directly without dispatching
    # to a worker. The workers attach their own view by name.
    parent_shm = None
    parent_shm_view = None
    if output_shm_name is not None:
        from multiprocessing import shared_memory as _shm
        parent_shm = _shm.SharedMemory(name=output_shm_name)
        parent_shm_view = np.ndarray(
            output_shm_shape, dtype=np.float32, buffer=parent_shm.buf
        )

    # Output zarr context (or null if we're shm-only)
    from contextlib import nullcontext
    if output_zarr is not None:
        zarr_out_ctx = open_ome_zarr(output_zarr / position, layout="fov", mode="r+")
    else:
        zarr_out_ctx = nullcontext(None)

    with zarr_out_ctx as store_out:
        out_array = store_out["0"] if store_out is not None else None

        # Pre-calculate tile grid
        y_starts = list(range(0, Y, tile_size))
        x_starts = list(range(0, X, tile_size))
        n_tiles = len(y_starts) * len(x_starts)

        # Get worker count and create persistent ProcessPool
        n_workers = _get_worker_count()

        if verbose:
            print(f"  Using {n_workers} parallel processes for {n_tiles} tiles per channel")

        # Create ProcessPool once and reuse for all rounds/channels (much faster than per-channel spawning)
        # Use spawn context to avoid fork-vs-zarr deadlock: workers reading v3-sharded
        # bc_stitched.zarr hang indefinitely if forked from a parent that has already
        # imported zarr/iohub (inherited async/file-handle state).
        import multiprocessing as _mp
        _ctx = _mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=_ctx) as executor:
            # Apply transforms per round
            for round_idx in tqdm(range(T), desc="  Applying transforms", disable=not verbose):
                # Load full round to CPU memory (faster than repeated zarr seeks)
                round_slice = data[round_idx, :, 0, :, :]
                if hasattr(round_slice, 'compute'):
                    round_data = round_slice.compute()  # (C, Y, X)
                else:
                    round_data = np.asarray(round_slice)  # (C, Y, X)

                # Transform each channel
                for c in range(C):
                    # Determine which transform to use
                    if round_idx == 0 and c == 0:
                        # Nucleus channel (Round 0, ch0) -> Use key -1
                        affine_3x3 = affines_cumulative.get(-1, np.eye(3))
                    else:
                        # Spot channels (Round 0 ch1-4, or any other round) -> Use round index key
                        affine_3x3 = affines_cumulative.get(round_idx, np.eye(3))

                    # Check if transform is identity
                    is_identity = np.allclose(affine_3x3, np.eye(3), atol=1e-6)

                    if is_identity:
                        chan_f32 = round_data[c, :, :].astype(np.float32, copy=False)
                        if out_array is not None:
                            out_array[round_idx, c, 0, :, :] = chan_f32
                        if parent_shm_view is not None:
                            parent_shm_view[round_idx, c, 0, :, :] = chan_f32
                        continue

                    # Prepare affine matrix for scipy/cupyx
                    # For translation-only (Nucleus/Round0), we can optimize, but full affine handles it too.
                    # Note: scipy.ndimage.affine_transform expects inverse transform (output->input).
                    # We have Forward transform (input->output/anchor). So we need Inverse.
                    affine_3x3_inv = np.linalg.inv(affine_3x3)
                    affine_matrix = affine_3x3_inv[:2, :].astype(np.float32)

                    # Create list of all tiles for this channel
                    tiles = [(y, x) for y in y_starts for x in x_starts]
                    n_tiles_channel = len(tiles)

                    # Process tiles in parallel using the persistent ProcessPool (bypasses GIL)
                    completed = 0
                    t_channel_start = time.time()

                    # Submit all tiles to the existing executor.
                    # Workers will write to output zarr (if a path was given)
                    # and to shared memory (if a name was given).
                    out_path_arg = str(output_zarr) if output_zarr is not None else None
                    futures = {
                        executor.submit(
                            _process_tile_worker,
                            str(input_zarr), out_path_arg, position,
                            round_idx, c, y_out, x_out,
                            affine_matrix, tile_size, padding, Y, X,
                            output_shm_name, output_shm_shape, "float32",
                        ): (y_out, x_out)
                        for y_out, x_out in tiles
                    }

                    # Collect results and track progress
                    tile_times = []
                    for future in as_completed(futures):
                        try:
                            y_out, x_out, t_tile = future.result()  # Get tile coordinates and timing
                            tile_times.append(t_tile)
                            completed += 1

                            # Print progress every 10% or 500 tiles
                            if completed % max(1, n_tiles_channel // 10) == 0 or completed % 500 == 0:
                                pct = 100.0 * completed / n_tiles_channel
                                avg_tile_time = np.mean(tile_times) if tile_times else 0
                                if verbose:
                                    print(f"    Round {round_idx+1}/{T}, Channel {c+1}/{C}: {completed}/{n_tiles_channel} tiles ({pct:.1f}%) - avg tile time: {avg_tile_time:.3f}s")
                        except Exception as e:
                            # Log error but continue processing other tiles
                            print(f"    ERROR processing tile {futures[future]}: {e}")
                            raise  # Re-raise to stop processing

                    t_channel_elapsed = time.time() - t_channel_start
                    if verbose:
                        avg_tile_time = np.mean(tile_times) if tile_times else 0
                        print(f"    Channel complete: {t_channel_elapsed:.1f}s total, {avg_tile_time:.3f}s avg per tile")

    # Drop parent shm view + close handle (caller still owns the shm block)
    if parent_shm is not None:
        parent_shm_view = None  # noqa: F841
        parent_shm.close()

    dt = time.time() - t_start
    if verbose:
        sinks = []
        if output_zarr is not None:
            sinks.append("zarr")
        if output_shm_name is not None:
            sinks.append("shm")
        print(f"  Transforms applied to {' + '.join(sinks)} ({dt:.1f}s)")


def _save_metrics_csv_for_well(results: dict, output_path: Path, verbose: bool = True):
    """Save registration metrics to CSV."""
    import pandas as pd

    rows = []

    # Nucleus metrics
    if "nucleus" in results:
        nuc = results["nucleus"]
        rows.append(
            {
                "round_pair": "nucleus_to_round0",
                "n_matches": nuc["metrics"]["n_matches"],
                "n_inliers": nuc["metrics"]["n_inliers"],
                "inlier_ratio": nuc["metrics"]["inlier_ratio"],
                "residual_mean": nuc["metrics"]["residual_mean"],
                "residual_std": nuc["metrics"]["residual_std"],
                "residual_max": nuc["metrics"]["residual_max"],
            }
        )

    # Round-to-round metrics
    for i in range(9):
        key = f"round_{i+1}_to_{i}"
        if key in results:
            rnd = results[key]
            rows.append(
                {
                    "round_pair": f"round{i+1}_to_round{i}",
                    "n_matches": rnd["metrics"]["n_matches"],
                    "n_inliers": rnd["metrics"]["n_inliers"],
                    "inlier_ratio": rnd["metrics"]["inlier_ratio"],
                    "residual_mean": rnd["metrics"]["residual_mean"],
                    "residual_std": rnd["metrics"]["residual_std"],
                    "residual_max": rnd["metrics"]["residual_max"],
                }
            )

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    if verbose:
        print(f"  Saved metrics: {output_path}")


def main():
    """CLI entry point."""
    import argparse
    import sys
    from cyclops_process.pipelinerunner.orchestrator import resolve_experiment_config

    parser = argparse.ArgumentParser(
        description="ISS Round-to-Round Registration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register single well (using -e and -w flags)
  python -m cyclops_process.processes.auto_register.iss_cycle_register -e ops0035 -w 1

  # Register all wells (partial experiment name matching)
  python -m cyclops_process.processes.auto_register.iss_cycle_register -e ops0035 -w all

  # Long-form flags
  python -m cyclops_process.processes.auto_register.iss_cycle_register \\
      --experiment ops0035_20250501 --well 1

  # Custom thresholds
  python -m cyclops_process.processes.auto_register.iss_cycle_register \\
      -e ops0035 -w 1 --spot-threshold 500 --nucleus-threshold 300
        """,
    )

    parser.add_argument(
        "-e",
        "--exp",
        "--experiment",
        dest="experiment",
        type=str,
        required=True,
        help="Experiment name; partial names allowed (e.g., 'ops0042' or 'ops0042_20250520')",
    )
    parser.add_argument(
        "-w",
        "--well",
        type=str,
        required=True,
        help="Well number (1, 2, 3, or 'all')",
    )
    parser.add_argument("--spot-threshold", type=float, default=400, help="Spot intensity threshold (default: 400)")
    parser.add_argument(
        "--nucleus-threshold", type=float, default=200, help="Nucleus intensity threshold (default: 200)"
    )
    parser.add_argument(
        "--transform-type",
        type=str,
        default="similarity",
        choices=["affine", "similarity", "euclidean"],
        help="Transform type (default: similarity)",
    )
    parser.add_argument("--no-apply", action="store_true", help="Skip applying transforms (compute only)")
    parser.add_argument("--create-overlays", action="store_true", help="Generate QA overlay images (NYI)")
    parser.add_argument(
        "--skip-nucleus-registration",
        action="store_true",
        help="Skip nucleus→spots registration (use identity). For experiments without pre-DAPI round.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    # Resolve experiment config (includes name resolution with interactive selection)
    config_path = resolve_experiment_config(args.experiment, allow_interactive=True)
    if config_path is None:
        print("No experiment selected or found. Exiting.")
        sys.exit(1)

    # Extract experiment name from config path
    experiment = config_path.stem.replace("_config", "")

    # Parse wells (full row/col units accepted; parsed downstream, default row A)
    if args.well.lower() == "all":
        wells = [1, 2, 3]
    else:
        wells = [args.well]

    # Run registration for each well
    for well in wells:
        auto_register_iss_rounds(
            experiment=experiment,
            well=well,
            spot_threshold=args.spot_threshold,
            nucleus_threshold=args.nucleus_threshold,
            transform_type=args.transform_type,
            apply_transforms=not args.no_apply,
            create_overlays=args.create_overlays,
            skip_nucleus_registration=args.skip_nucleus_registration,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()