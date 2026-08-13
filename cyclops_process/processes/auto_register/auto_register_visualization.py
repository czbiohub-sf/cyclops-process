"""
Visualization and overlay generation for automatic registration.

Contains:
- PCC alignment comparison overlays
- Final registration validation overlays
- Multi-region grid sampling
- Side-by-side before/after comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import zarr as _zarr_mod
from skimage.transform import downscale_local_mean
from scipy import ndimage
import skimage.io
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

try:
    import cupy as xp
    from cupyx.scipy import ndimage as cundi

    if xp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("No CUDA device available")
except (ModuleNotFoundError, ImportError, RuntimeError):
    import numpy as xp
    from scipy import ndimage as cundi


def create_timepoint_comparison_grid(
    source_seg_path: Path,
    target_seg_path: Path,
    position: str,
    t_idx_source: int,
    t_idx_target: int,
    n_target_timepoints: int,
    output_dir: Path,
    crop_regions: list,
    verbose: bool = True,
):
    """
    Create side-by-side comparison of source (pheno) and all target (track) timepoints.

    Shows 3 crop positions x (1 source + N target timepoints), with the selected
    target timepoint highlighted.

    Parameters
    ----------
    source_seg_path : Path
        Source segmentation path (phenotyping).
    target_seg_path : Path
        Target segmentation path (tracking).
    position : str
        Position string.
    t_idx_source : int
        Source timepoint index.
    t_idx_target : int
        Selected target timepoint index.
    n_target_timepoints : int
        Total number of target timepoints available.
    output_dir : Path
        Output directory for images.
    crop_regions : list
        List of (h_start, h_end, w_start, w_end) crop regions.
    verbose : bool
        Print progress.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load source + all target timepoint masks concurrently
    from concurrent.futures import ThreadPoolExecutor as _TPPool

    def _safe_load(seg_path, pos, t):
        try:
            return load_mask_2d(seg_path, pos, t)
        except:
            return None

    with _TPPool(max_workers=min(n_target_timepoints + 1, 8)) as tpool:
        src_future = tpool.submit(load_mask_2d, source_seg_path, position, t_idx_source)
        tgt_futures = [tpool.submit(_safe_load, target_seg_path, position, t)
                       for t in range(n_target_timepoints)]
        mask_src = src_future.result()
        target_masks = [f.result() for f in tgt_futures]
    if verbose and any(m is None for m in target_masks):
        for t, m in enumerate(target_masks):
            if m is None:
                print(f"      Warning: Could not load target timepoint {t}")

    # Create grid for each crop position
    for pos_idx, (h_start, h_end, w_start, w_end) in enumerate(crop_regions):
        # Crop source
        src_crop = mask_src[h_start:h_end, w_start:w_end]
        src_binary = (src_crop > 0).astype(np.uint8) * 255

        # Crop all target timepoints
        target_crops = []
        for mask_tgt in target_masks:
            if mask_tgt is not None:
                tgt_crop = mask_tgt[h_start:h_end, w_start:w_end]
                tgt_binary = (tgt_crop > 0).astype(np.uint8) * 255
                target_crops.append(tgt_binary)
            else:
                target_crops.append(None)

        # Create figure: 3 rows (one per crop), columns = 1 source + N targets
        n_cols = 1 + n_target_timepoints
        fig = Figure(figsize=(3 * n_cols, 3), dpi=100)
        canvas = FigureCanvasAgg(fig)

        # Column 0: Source (pheno)
        ax_src = fig.add_subplot(1, n_cols, 1)
        ax_src.imshow(src_binary, cmap="gray")
        ax_src.set_title(
            f"Pheno t={t_idx_source}", fontsize=10, fontweight="bold", color="blue"
        )
        ax_src.axis("off")

        # Columns 1-N: Target timepoints
        for t in range(n_target_timepoints):
            ax_tgt = fig.add_subplot(1, n_cols, t + 2)

            if target_crops[t] is not None:
                ax_tgt.imshow(target_crops[t], cmap="gray")

                # Highlight selected timepoint
                if t == t_idx_target:
                    title_color = "red"
                    title = f"Track t={t} ★ SELECTED"
                else:
                    title_color = "black"
                    title = f"Track t={t}"

                ax_tgt.set_title(
                    title, fontsize=10, fontweight="bold", color=title_color
                )
            else:
                ax_tgt.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=16)
                ax_tgt.set_title(f"Track t={t}", fontsize=10)

            ax_tgt.axis("off")

        fig.suptitle(
            f"Timepoint Comparison - Position {pos_idx}", fontsize=12, fontweight="bold"
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        # Save
        output_path = output_dir / f"00_timepoint_comparison_pos{pos_idx}.png"
        canvas.draw()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
        buf_rgb = buf[:, :, :3]
        skimage.io.imsave(output_path, buf_rgb)
        plt.close(fig)

        if verbose:
            print(f"      Saved {output_path.name}")


def compute_crop_regions(mask_shape: tuple, crop_size: int = 1024) -> list:
    """
    Compute 3 crop regions for detail overlays (center and 2 offset positions).

    Parameters
    ----------
    mask_shape : tuple
        Shape of the mask (height, width).
    crop_size : int
        Size of crop in pixels.

    Returns
    -------
    list
        List of 3 crop regions as (h_start, h_end, w_start, w_end) tuples.
    """
    center_h, center_w = mask_shape[0] // 2, mask_shape[1] // 2

    # Define 3 sampling positions: center, and two offset positions
    # 10% offset to stay closer to center
    offset_frac = 0.10
    offset_h = int(mask_shape[0] * offset_frac)
    offset_w = int(mask_shape[1] * offset_frac)

    crop_positions = [
        (center_h, center_w),  # Position 0: center
        (center_h + offset_h, center_w + offset_w),  # Position 1: lower-right
        (center_h - offset_h, center_w + offset_w),  # Position 2: upper-right
    ]

    crop_regions = []
    for crop_center_h, crop_center_w in crop_positions:
        h_start = max(0, crop_center_h - crop_size // 2)
        h_end = min(mask_shape[0], crop_center_h + crop_size // 2)
        w_start = max(0, crop_center_w - crop_size // 2)
        w_end = min(mask_shape[1], crop_center_w + crop_size // 2)
        crop_regions.append((h_start, h_end, w_start, w_end))

    return crop_regions


def load_mask_2d(seg_path: Path, position: str, t_idx: int) -> np.ndarray:
    """
    Load 2D mask from zarr store, handling different dimensionalities.

    Parameters
    ----------
    seg_path : Path
        Path to segmentation zarr.
    position : str
        Position string (e.g., "A/1/0").
    t_idx : int
        Time index.

    Returns
    -------
    np.ndarray
        2D mask array.
    """
    # Use parallel chunk reader for fast NFS I/O (16 threads)
    from cyclops_process.processes.auto_register.auto_register_utils import (
        _load_seg_2d,
        resolve_seg_array_path,
    )

    # Clamp t_idx for arrays with fewer timepoints
    seg = _zarr_mod.open(str(resolve_seg_array_path(seg_path, position)), mode="r")
    if seg.ndim >= 4:
        t_idx = min(t_idx, seg.shape[0] - 1)

    return _load_seg_2d(seg_path, position, t_idx)


def _load_mask_2d_legacy(seg_path, position, t_idx):
    """Legacy single-threaded load (unused, kept for reference)."""
    seg = _zarr_mod.open(str(seg_path / position / "0"), mode="r")
    if seg.ndim == 5:
        t_idx_clamped = min(t_idx, seg.shape[0] - 1)
        mask = seg[t_idx_clamped, 0, 0, :, :]
    elif seg.ndim == 4:
        t_idx_clamped = min(t_idx, seg.shape[0] - 1)
        mask = seg[t_idx_clamped, 0, :, :]
    elif seg.ndim == 3:
        mask = seg[0, :, :]
    else:
        mask = seg[:, :]

    return mask


def load_mask_2d_from_level(
    seg_path: Path, position: str, t_idx: int, level: int = 0
) -> np.ndarray:
    """
    Load 2D mask from specific pyramid level.

    This is much faster for visualization when you need downsampled masks,
    as it loads the pre-computed pyramid level instead of loading full resolution
    and downsampling manually.

    Parameters
    ----------
    seg_path : Path
        Path to segmentation zarr.
    position : str
        Position string (e.g., "A/1/0").
    t_idx : int
        Time index.
    level : int
        Pyramid level (0=full res, 1=2x down, 2=4x down, 3=8x down, 4=16x down).

    Returns
    -------
    np.ndarray
        2D mask array at specified pyramid level.

    Raises
    ------
    KeyError
        If pyramid level doesn't exist in the store.
    """
    store = _zarr_mod.open(str(seg_path), mode="r")

    # Try different possible pyramid structures
    seg = None
    tried_paths = []

    # Debug: Check what's available
    try:
        pos_node = store[position]
        pos_keys = list(pos_node.keys()) if hasattr(pos_node, "keys") else []
    except:
        pos_keys = []

    # Structure 1 (v3 canonical, OME-NGFF v0.5): position/labels/nuclear_seg/level
    try:
        seg = store[position]["labels"]["nuclear_seg"][str(level)].data
    except (KeyError, AttributeError):
        tried_paths.append(f"{position}/labels/nuclear_seg/{level}")

    # Structure 2 (v3 canonical, cell seg): position/labels/cell_seg/level
    if seg is None:
        try:
            seg = store[position]["labels"]["cell_seg"][str(level)].data
        except (KeyError, AttributeError):
            tried_paths.append(f"{position}/labels/cell_seg/{level}")

    # Structure 3 (legacy v2 top-level symlink, retiring): position/nuclear_seg/level
    if seg is None:
        try:
            nuc_seg = store[position]["nuclear_seg"]
            seg = nuc_seg[str(level)].data
        except (KeyError, AttributeError):
            tried_paths.append(
                f"{position}/nuclear_seg/{level} (available at position: {pos_keys})"
            )

    # Structure 4 (legacy v2 top-level cell seg): position/seg/level
    if seg is None:
        try:
            seg = store[position]["seg"][str(level)].data
        except (KeyError, AttributeError):
            tried_paths.append(f"{position}/seg/{level}")

    # Structure 5 (flat — some stores have levels directly): position/level
    if seg is None:
        try:
            seg = store[position][str(level)].data
        except (KeyError, AttributeError):
            tried_paths.append(f"{position}/{level}")

    if seg is None:
        raise KeyError(
            f"Could not find pyramid level {level} in {seg_path}. "
            f"Tried paths: {', '.join(tried_paths)}"
        )

    # Extract 2D mask
    if seg.ndim == 5:
        mask = seg[t_idx, 0, 0, :, :]
    elif seg.ndim == 4:
        mask = seg[t_idx, 0, :, :]
    elif seg.ndim == 3:
        mask = seg[0, :, :]
    else:
        mask = seg[:, :]

    # Compute if dask/cupy array
    if hasattr(mask, "compute"):
        mask = mask.compute()
    if hasattr(mask, "get"):
        mask = mask.get()

    return mask


def create_overlay_rgb(mask_target: np.ndarray, mask_source: np.ndarray) -> np.ndarray:
    """
    Create RGB overlay from binary masks.

    Red channel: target
    Green channel: source
    Yellow: overlap

    Parameters
    ----------
    mask_target : np.ndarray
        Target mask (any dtype, will be binarized).
    mask_source : np.ndarray
        Source mask (any dtype, will be binarized).

    Returns
    -------
    np.ndarray
        RGB overlay (uint8, shape (..., 3)).
    """
    binary_tgt = (mask_target > 0).astype(np.uint8) * 255
    binary_src = (mask_source > 0).astype(np.uint8) * 255

    overlay = np.zeros((*binary_tgt.shape, 3), dtype=np.uint8)
    overlay[..., 0] = binary_tgt  # Red
    overlay[..., 1] = binary_src  # Green

    return overlay


def create_pcc_overlays(
    source_seg_path: Path,
    target_seg_path: Path,
    position: str,
    pcc_shift: np.ndarray,
    output_dir: Path,
    t_idx_source: int = 0,
    t_idx_target: int = 0,
    center_fraction: float = 1.0,
    affine_3x3: np.ndarray = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, tuple]:
    """
    Create PCC alignment comparison overlays (before/after).

    Generates:
    - Full well overlay at 8x downsample (before/after side-by-side)
    - Detail crop at 1x (before/after side-by-side)
    - 3x2 grid showing individual channels and overlay

    Parameters
    ----------
    source_seg_path : Path
        Source segmentation zarr path.
    target_seg_path : Path
        Target segmentation zarr path.
    position : str
        Position string.
    pcc_shift : np.ndarray
        PCC translation offset (dy, dx).
    output_dir : Path
        Output directory for overlays.
    t_idx_source : int
        Source time index.
    t_idx_target : int
        Target time index.
    center_fraction : float
        Fraction of well to use (for debug mode).
    verbose : bool
        Print progress messages.

    Returns
    -------
    tuple
        (mask_src, mask_tgt, crop_region) for reuse in final overlays.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load masks concurrently
    from concurrent.futures import ThreadPoolExecutor as _MaskPool
    with _MaskPool(max_workers=2) as mpool:
        src_f = mpool.submit(load_mask_2d, source_seg_path, position, t_idx_source)
        tgt_f = mpool.submit(load_mask_2d, target_seg_path, position, t_idx_target)
        mask_src = src_f.result()
        mask_tgt = tgt_f.result()

    # Calculate 3 crop regions using shared helper function
    crop_regions = compute_crop_regions(mask_src.shape, crop_size=1024)

    # Full well at 8x (side by side)
    before_8x = _create_shifted_overlay(mask_src, mask_tgt, np.array([0.0, 0.0]), 8)
    after_pcc_8x = _create_shifted_overlay(mask_src, mask_tgt, pcc_shift, 8)
    _save_side_by_side(before_8x, after_pcc_8x, output_dir / "01_pcc_alignment_8x.png")
    if verbose:
        print(f"      Saved 01_pcc_alignment_8x.png")

    # Apply PCC shift OR full affine to full source mask BEFORE cropping
    # This is critical: shift values are in full-well coordinates
    if affine_3x3 is not None:
        # Apply full affine transformation (rotation + scale + translation)
        # YAML stores inverse transform (target→source), which is what ndimage.affine_transform expects
        # Do NOT invert - use directly like manual affine
        mask_src_shifted = ndimage.affine_transform(
            mask_src, affine_3x3, order=0, output_shape=mask_src.shape
        )
    else:
        # Simple translation (PCC or hardcoded shift)
        mask_src_shifted = ndimage.shift(mask_src, pcc_shift, order=0)

    # Detail crops at 1x (3 positions) — render in parallel processes (avoids GIL)
    from concurrent.futures import ProcessPoolExecutor as _PccRenderPool

    pcc_detail_args = []
    for pos_idx, (h_start, h_end, w_start, w_end) in enumerate(crop_regions):
        pcc_detail_args.append((
            mask_src[h_start:h_end, w_start:w_end].copy(),
            mask_src_shifted[h_start:h_end, w_start:w_end].copy(),
            mask_tgt[h_start:h_end, w_start:w_end].copy(),
            str(output_dir / f"02_pcc_detail_grid_pos{pos_idx}.png"),
            "BEFORE PCC",
            "AFTER PCC",
        ))

    with _PccRenderPool(max_workers=len(pcc_detail_args)) as rpool:
        list(rpool.map(_save_detail_grid_preapplied_mp, pcc_detail_args))

    if verbose:
        for pos_idx in range(len(crop_regions)):
            print(f"      Saved 02_pcc_detail_grid_pos{pos_idx}.png")

    return mask_src, mask_tgt, crop_regions


def create_final_alignment_overlays(
    mask_src: np.ndarray,
    mask_tgt: np.ndarray,
    affine_3x3: np.ndarray = None,
    crop_region: tuple | list = None,
    output_dir: Path = None,
    pcc_shift: np.ndarray = None,
    center_fraction: float = 1.0,
    manual_yaml_path: Path = None,
    auto_yaml_path: Path = None,
    verbose: bool = True,
):
    """
    Create final alignment overlays (after PCC + RANSAC).

    Generates:
    - Full well overlay at 8x downsample
    - Detail crops at 1x (3 positions)
    - 3x2 grids showing before/after final alignment (3 positions)

    IMPORTANT: "BEFORE" shows post-PCC state (not original unaligned).
    This allows us to see RANSAC's contribution clearly.

    Parameters
    ----------
    mask_src : np.ndarray
        Source mask (original, not warped).
    mask_tgt : np.ndarray
        Target mask.
    affine_3x3 : np.ndarray, optional
        Final 3x3 affine transform (PCC + RANSAC composition).
        If None, will load from auto_yaml_path.
    crop_region : tuple or list of tuples
        Single (h_start, h_end, w_start, w_end) for detail crop, or list of 3 crop regions.
    output_dir : Path
        Output directory.
    pcc_shift : np.ndarray, optional
        PCC translation shift [dy, dx] to show post-PCC as BEFORE.
        If None, uses original unaligned source for BEFORE.
    center_fraction : float
        Fraction of well used (for debug mode).
    manual_yaml_path : Path, optional
        Path to manual affine YAML for comparison.
    auto_yaml_path : Path, optional
        Path to auto affine YAML (used if affine_3x3 is None).
    verbose : bool
        Print progress messages.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load auto affine from YAML if not provided (apply same as manual)
    if affine_3x3 is None and auto_yaml_path is not None:
        import yaml

        if verbose:
            print(f"    Loading auto affine from YAML...")
        with open(auto_yaml_path, "r") as f:
            auto_data = yaml.safe_load(f)
        auto_affine_4x4 = np.array(auto_data["affine_transform_zyx"])

        # Extract 2D affine (YX plane)
        affine_3x3 = np.eye(3)
        affine_3x3[:2, :2] = auto_affine_4x4[1:3, 1:3]
        affine_3x3[:2, 2] = auto_affine_4x4[1:3, 3]

    # Apply PCC shift and full affine concurrently (both operate on full-res mask)
    from concurrent.futures import ThreadPoolExecutor as _XformPool
    from scipy import ndimage

    if verbose:
        print(f"    Creating BEFORE + AFTER images (concurrent)...")

    def _apply_pcc():
        if pcc_shift is not None:
            return ndimage.shift(mask_src, pcc_shift, order=0)
        return mask_src

    def _apply_affine():
        return ndimage.affine_transform(
            mask_src, affine_3x3, order=0, output_shape=mask_src.shape)

    with _XformPool(max_workers=2) as xpool:
        pcc_future = xpool.submit(_apply_pcc)
        warp_future = xpool.submit(_apply_affine)
        mask_src_pcc = pcc_future.result()
        mask_src_warped = warp_future.result()

    if verbose:
        print(f"    Downsampling for full-well visualization...")

    # Downsample all 3 masks concurrently
    downsample_for_viz = 8

    def _ds(m):
        return downscale_local_mean(m, (downsample_for_viz, downsample_for_viz))

    with _XformPool(max_workers=3) as dspool:
        f1 = dspool.submit(_ds, mask_src_pcc)
        f2 = dspool.submit(_ds, mask_src_warped)
        f3 = dspool.submit(_ds, mask_tgt)
        mask_src_pcc_ds = f1.result()
        mask_src_warped_ds = f2.result()
        mask_tgt_ds = f3.result()

    # Ensure masks have the same shape (they may differ slightly after downsampling)
    min_h = min(
        mask_src_pcc_ds.shape[0], mask_src_warped_ds.shape[0], mask_tgt_ds.shape[0]
    )
    min_w = min(
        mask_src_pcc_ds.shape[1], mask_src_warped_ds.shape[1], mask_tgt_ds.shape[1]
    )
    mask_src_pcc_ds = mask_src_pcc_ds[:min_h, :min_w]
    mask_src_warped_ds = mask_src_warped_ds[:min_h, :min_w]
    mask_tgt_ds = mask_tgt_ds[:min_h, :min_w]

    # Full well at 8x - show before/after side-by-side
    # BEFORE = post-PCC (shows RANSAC's contribution)
    # AFTER = post-PCC + RANSAC (final alignment)
    before_final_8x = create_overlay_rgb(mask_tgt_ds, mask_src_pcc_ds)
    after_final_8x = create_overlay_rgb(mask_tgt_ds, mask_src_warped_ds)
    _save_side_by_side(
        before_final_8x, after_final_8x, output_dir / "03_final_alignment_8x.png"
    )
    if verbose:
        print(f"      Saved 03_final_alignment_8x.png")

    # Handle single crop region (legacy) or list of crop regions
    if isinstance(crop_region, tuple):
        crop_regions = [crop_region]
    else:
        crop_regions = crop_region

    # Detail crops for each position — render all in parallel (ProcessPool avoids GIL)
    if verbose:
        print(f"    Creating detail overlays for {len(crop_regions)} positions...")

    from concurrent.futures import ProcessPoolExecutor as _RenderPool

    # Prepare crop data for parallel rendering
    detail_args = []
    for pos_idx, (h_start, h_end, w_start, w_end) in enumerate(crop_regions):
        detail_args.append((
            mask_src_pcc[h_start:h_end, w_start:w_end].copy(),
            mask_src_warped[h_start:h_end, w_start:w_end].copy(),
            mask_tgt[h_start:h_end, w_start:w_end].copy(),
            str(output_dir / f"04_final_detail_grid_pos{pos_idx}.png"),
        ))

    with _RenderPool(max_workers=len(detail_args)) as rpool:
        list(rpool.map(_save_final_detail_grid_mp, detail_args))

    if verbose:
        for pos_idx in range(len(crop_regions)):
            print(f"      Saved 04_final_detail_grid_pos{pos_idx}.png")

    # Optional: Create manual affine comparison grids (if manual YAML provided)
    if manual_yaml_path is not None and manual_yaml_path.exists():
        if verbose:
            print(
                f"    Creating manual affine comparison for {len(crop_regions)} positions..."
            )

        # Load manual affine from YAML
        import yaml

        with open(manual_yaml_path, "r") as f:
            manual_data = yaml.safe_load(f)
        manual_affine_4x4 = np.array(manual_data["affine_transform_zyx"])

        # Extract 2D affine (YX plane)
        # YAML stores inverse transform (target→source), which is what ndimage.affine_transform expects
        manual_affine_3x3 = np.eye(3)
        manual_affine_3x3[:2, :2] = manual_affine_4x4[1:3, 1:3]
        manual_affine_3x3[:2, 2] = manual_affine_4x4[1:3, 3]

        # Apply manual affine to source mask (following track.py - do NOT invert)
        mask_src_manual = ndimage.affine_transform(
            mask_src, manual_affine_3x3, order=0, output_shape=mask_src.shape
        )

        # Create comparison grids for same 3 positions
        for pos_idx, (h_start, h_end, w_start, w_end) in enumerate(crop_regions):
            mask_src_crop_auto = mask_src_warped[h_start:h_end, w_start:w_end]
            mask_src_crop_manual = mask_src_manual[h_start:h_end, w_start:w_end]
            mask_tgt_crop = mask_tgt[h_start:h_end, w_start:w_end]

            # Create 2x3 grid: [Manual, Auto] x [Target, Source, Overlay]
            _save_manual_vs_auto_grid(
                mask_src_crop_manual,
                mask_src_crop_auto,
                mask_tgt_crop,
                output_dir / f"05_manual_vs_auto_pos{pos_idx}.png",
            )
            if verbose:
                print(f"      Saved 05_manual_vs_auto_pos{pos_idx}.png")


def create_validation_overlays(
    source_seg_path: Path,
    target_seg_path: Path,
    affine_3x3: np.ndarray,
    position: str,
    output_dir: Path,
    t_idx_source: int = 0,
    t_idx_target: int = 0,
    n_regions: int = 9,
    region_size: int = 500,
):
    """
    Create validation overlay images from different regions of the well.

    Samples n_regions evenly distributed across the well (grid pattern)
    and saves 500×500 pixel RGB overlays showing:
    - Target segmentation (red channel)
    - Registered source segmentation (green channel)
    - Overlap appears yellow

    Parameters
    ----------
    source_seg_path : Path
        Source segmentation zarr path.
    target_seg_path : Path
        Target segmentation zarr path.
    affine_3x3 : np.ndarray
        2D affine transform (3x3).
    position : str
        Position string.
    output_dir : Path
        Output directory for overlay images.
    t_idx_source : int
        Source time index.
    t_idx_target : int
        Target time index.
    n_regions : int
        Number of regions to sample (will form sqrt(n) × sqrt(n) grid).
    region_size : int
        Size of each region in pixels.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load segmentations (direct zarr read)
    source_seg = _zarr_mod.open(str(source_seg_path / position / "0"), mode="r")
    target_seg = _zarr_mod.open(str(target_seg_path / position / "0"), mode="r")

    # Handle dimensionality
    if source_seg.ndim == 5:
        source_img = source_seg[t_idx_source, 0, 0, :, :]
        target_img = target_seg[t_idx_target, 0, 0, :, :]
    elif source_seg.ndim == 4:
        source_img = source_seg[t_idx_source, 0, :, :]
        target_img = target_seg[t_idx_target, 0, :, :]
    else:
        source_img = source_seg[0, :, :]
        target_img = target_seg[0, :, :]

    # Apply affine to source
    source_img_float = source_img.astype(np.float32)
    affine_inv = np.linalg.inv(affine_3x3)
    source_registered = cundi.affine_transform(
        xp.asarray(source_img_float),
        xp.asarray(affine_inv[:2, :2]),
        offset=xp.asarray(affine_inv[:2, 2]),
        order=0,
        output_shape=target_img.shape,
        mode="constant",
        cval=0,
    )
    if hasattr(source_registered, "get"):
        source_registered = source_registered.get()

    # Get image dimensions
    Y, X = target_img.shape

    # Create grid of sampling positions
    grid_size = int(np.ceil(np.sqrt(n_regions)))
    y_positions = np.linspace(
        region_size // 2, Y - region_size // 2, grid_size, dtype=int
    )
    x_positions = np.linspace(
        region_size // 2, X - region_size // 2, grid_size, dtype=int
    )

    region_idx = 0
    for yi, y_center in enumerate(y_positions):
        for xi, x_center in enumerate(x_positions):
            if region_idx >= n_regions:
                break

            # Define region bounds
            y_start = max(0, y_center - region_size // 2)
            y_end = min(Y, y_center + region_size // 2)
            x_start = max(0, x_center - region_size // 2)
            x_end = min(X, x_center + region_size // 2)

            # Extract regions
            target_region = target_img[y_start:y_end, x_start:x_end]
            source_region = source_registered[y_start:y_end, x_start:x_end]

            # Create RGB overlay
            rgb = create_overlay_rgb(target_region, source_region)

            # Save overlay
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(rgb)
            ax.set_title(
                f"Position {position} | Region {region_idx + 1}/{n_regions}\n"
                f"Center: Y={y_center}, X={x_center}\n"
                f"Red=Target, Green=Source, Yellow=Overlap"
            )
            ax.axis("off")

            output_path = (
                output_dir / f"{position.replace('/', '_')}_region_{region_idx:02d}.png"
            )
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            region_idx += 1


# =============================================================================
# Private helper functions
# =============================================================================


def _create_shifted_overlay(
    mask_src: np.ndarray, mask_tgt: np.ndarray, shift: np.ndarray, ds_factor: int
) -> np.ndarray:
    """Create overlay at given downsampling factor with shift applied."""
    if ds_factor > 1:
        mask_src_ds = downscale_local_mean(mask_src, (ds_factor, ds_factor))
        mask_tgt_ds = downscale_local_mean(mask_tgt, (ds_factor, ds_factor))
    else:
        mask_src_ds = mask_src.copy()
        mask_tgt_ds = mask_tgt.copy()

    # Pad to same shape if needed
    if mask_src_ds.shape != mask_tgt_ds.shape:
        max_h = max(mask_src_ds.shape[0], mask_tgt_ds.shape[0])
        max_w = max(mask_src_ds.shape[1], mask_tgt_ds.shape[1])

        if mask_src_ds.shape != (max_h, max_w):
            pad_h = max_h - mask_src_ds.shape[0]
            pad_w = max_w - mask_src_ds.shape[1]
            mask_src_ds = np.pad(
                mask_src_ds,
                ((0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=0,
            )

        if mask_tgt_ds.shape != (max_h, max_w):
            pad_h = max_h - mask_tgt_ds.shape[0]
            pad_w = max_w - mask_tgt_ds.shape[1]
            mask_tgt_ds = np.pad(
                mask_tgt_ds,
                ((0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=0,
            )

    # Apply shift
    shift_ds = shift / ds_factor
    mask_src_shifted = ndimage.shift(mask_src_ds, shift_ds, order=0)

    # Create overlay
    return create_overlay_rgb(mask_tgt_ds, mask_src_shifted)


def _create_final_overlay(
    mask_src: np.ndarray, mask_tgt: np.ndarray, ds_factor: int
) -> np.ndarray:
    """Create final alignment overlay at given downsampling factor."""
    if ds_factor > 1:
        mask_src_ds = downscale_local_mean(mask_src, (ds_factor, ds_factor))
        mask_tgt_ds = downscale_local_mean(mask_tgt, (ds_factor, ds_factor))
    else:
        mask_src_ds = mask_src
        mask_tgt_ds = mask_tgt

    # Pad to same shape if needed
    if mask_src_ds.shape != mask_tgt_ds.shape:
        max_h = max(mask_src_ds.shape[0], mask_tgt_ds.shape[0])
        max_w = max(mask_src_ds.shape[1], mask_tgt_ds.shape[1])

        if mask_src_ds.shape != (max_h, max_w):
            pad_h = max_h - mask_src_ds.shape[0]
            pad_w = max_w - mask_src_ds.shape[1]
            mask_src_ds = np.pad(
                mask_src_ds,
                ((0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=0,
            )

        if mask_tgt_ds.shape != (max_h, max_w):
            pad_h = max_h - mask_tgt_ds.shape[0]
            pad_w = max_w - mask_tgt_ds.shape[1]
            mask_tgt_ds = np.pad(
                mask_tgt_ds,
                ((0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=0,
            )

    return create_overlay_rgb(mask_tgt_ds, mask_src_ds)


def _save_side_by_side(
    before_overlay: np.ndarray, after_overlay: np.ndarray, filename: Path
):
    """Save two overlays side by side with labels."""
    # Add 10 pixel white separator
    separator = np.ones((before_overlay.shape[0], 10, 3), dtype=np.uint8) * 255
    combined = np.concatenate([before_overlay, separator, after_overlay], axis=1)

    # Add text labels (scale font size to image dimensions)
    # Use 3% of image height as a good scaling factor
    fontsize = max(20, int(combined.shape[0] * 0.03))
    text_space_px = int(fontsize * 2.5)  # Space for text above image

    # Create figure with the combined image plus space for text
    dpi = 100
    fig_height = (combined.shape[0] + text_space_px) / dpi
    fig_width = combined.shape[1] / dpi
    fig = Figure(figsize=(fig_width, fig_height), dpi=dpi)
    canvas = FigureCanvasAgg(fig)

    # Position axes to leave room for text at top
    ax_height = combined.shape[0] / (combined.shape[0] + text_space_px)
    ax = fig.add_axes([0, 0, 1, ax_height])
    ax.axis("off")

    # Display image
    ax.imshow(combined)

    # Add text labels above the image (in figure coordinates)
    before_x = (before_overlay.shape[1] / 2) / combined.shape[1]
    after_x = (
        before_overlay.shape[1] + 10 + after_overlay.shape[1] / 2
    ) / combined.shape[1]
    text_y = ax_height + (1 - ax_height) / 2  # Center of text space

    fig.text(
        before_x,
        text_y,
        "BEFORE",
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.8),
    )
    fig.text(
        after_x,
        text_y,
        "AFTER",
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.8),
    )

    # Render to array
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
    buf_rgb = buf[:, :, :3]  # Drop alpha channel

    skimage.io.imsave(filename, buf_rgb)
    plt.close(fig)


def _save_detail_grid(
    mask_src_crop: np.ndarray,
    mask_tgt_crop: np.ndarray,
    shift: np.ndarray,
    filename: Path,
    title_before: str = "BEFORE",
    title_after: str = "AFTER",
):
    """Save 3x2 grid: [target, source, overlay] x [before, after]"""
    # Apply shift to source
    mask_src_shifted = ndimage.shift(mask_src_crop, shift, order=0)

    # Create binary masks
    binary_tgt = (mask_tgt_crop > 0).astype(np.uint8) * 255
    binary_src_before = (mask_src_crop > 0).astype(np.uint8) * 255
    binary_src_after = (mask_src_shifted > 0).astype(np.uint8) * 255

    # Create colored RGB versions to match overlay colors
    # Target = red, Source = green (matching create_overlay_rgb)
    tgt_rgb = np.stack(
        [binary_tgt, np.zeros_like(binary_tgt), np.zeros_like(binary_tgt)], axis=-1
    )
    src_before_rgb = np.stack(
        [
            np.zeros_like(binary_src_before),
            binary_src_before,
            np.zeros_like(binary_src_before),
        ],
        axis=-1,
    )
    src_after_rgb = np.stack(
        [
            np.zeros_like(binary_src_after),
            binary_src_after,
            np.zeros_like(binary_src_after),
        ],
        axis=-1,
    )

    # Create overlays
    overlay_before = create_overlay_rgb(mask_tgt_crop, mask_src_crop)
    overlay_after = create_overlay_rgb(mask_tgt_crop, mask_src_shifted)

    # Create figure with 3x2 grid
    fig = Figure(figsize=(12, 8), dpi=100)
    canvas = FigureCanvasAgg(fig)

    # Row 1: Before
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(tgt_rgb)
    ax1.set_title("Target (red)", fontsize=10, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(src_before_rgb)
    ax2.set_title(f"Source {title_before} (green)", fontsize=10, fontweight="bold")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(overlay_before)
    ax3.set_title(f"Overlay {title_before}", fontsize=10, fontweight="bold")
    ax3.axis("off")

    # Row 2: After
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(tgt_rgb)
    ax4.set_title("Target (red)", fontsize=10, fontweight="bold")
    ax4.axis("off")

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(src_after_rgb)
    ax5.set_title(f"Source {title_after} (green)", fontsize=10, fontweight="bold")
    ax5.axis("off")

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.imshow(overlay_after)
    ax6.set_title(f"Overlay {title_after}", fontsize=10, fontweight="bold")
    ax6.axis("off")

    fig.tight_layout()

    # Render to array
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
    buf_rgb = buf[:, :, :3]

    skimage.io.imsave(filename, buf_rgb)
    plt.close(fig)


def _save_detail_grid_preapplied_mp(args):
    """Module-level wrapper for ProcessPoolExecutor."""
    before, after, tgt, filename, title_before, title_after = args
    _save_detail_grid_preapplied(before, after, tgt, Path(filename), title_before, title_after)


def _save_detail_grid_preapplied(
    mask_src_crop_before: np.ndarray,
    mask_src_crop_after: np.ndarray,
    mask_tgt_crop: np.ndarray,
    filename: Path,
    title_before: str = "BEFORE",
    title_after: str = "AFTER",
):
    """Save 3x2 grid: [target, source, overlay] x [before, after] with pre-shifted source masks."""
    # Create binary masks
    binary_tgt = (mask_tgt_crop > 0).astype(np.uint8) * 255
    binary_src_before = (mask_src_crop_before > 0).astype(np.uint8) * 255
    binary_src_after = (mask_src_crop_after > 0).astype(np.uint8) * 255

    # Create colored RGB versions to match overlay colors
    # Target = red, Source = green (matching create_overlay_rgb)
    tgt_rgb = np.stack(
        [binary_tgt, np.zeros_like(binary_tgt), np.zeros_like(binary_tgt)], axis=-1
    )
    src_before_rgb = np.stack(
        [
            np.zeros_like(binary_src_before),
            binary_src_before,
            np.zeros_like(binary_src_before),
        ],
        axis=-1,
    )
    src_after_rgb = np.stack(
        [
            np.zeros_like(binary_src_after),
            binary_src_after,
            np.zeros_like(binary_src_after),
        ],
        axis=-1,
    )

    # Create overlays
    overlay_before = create_overlay_rgb(mask_tgt_crop, mask_src_crop_before)
    overlay_after = create_overlay_rgb(mask_tgt_crop, mask_src_crop_after)

    # Create figure with 3x2 grid
    fig = Figure(figsize=(12, 8), dpi=100)
    canvas = FigureCanvasAgg(fig)

    # Row 1: Before
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(tgt_rgb)
    ax1.set_title("Target (red)", fontsize=10, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(src_before_rgb)
    ax2.set_title(f"Source {title_before} (green)", fontsize=10, fontweight="bold")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(overlay_before)
    ax3.set_title(f"Overlay {title_before}", fontsize=10, fontweight="bold")
    ax3.axis("off")

    # Row 2: After
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(tgt_rgb)
    ax4.set_title("Target (red)", fontsize=10, fontweight="bold")
    ax4.axis("off")

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(src_after_rgb)
    ax5.set_title(f"Source {title_after} (green)", fontsize=10, fontweight="bold")
    ax5.axis("off")

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.imshow(overlay_after)
    ax6.set_title(f"Overlay {title_after}", fontsize=10, fontweight="bold")
    ax6.axis("off")

    fig.tight_layout()

    # Render to array
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
    buf_rgb = buf[:, :, :3]

    skimage.io.imsave(filename, buf_rgb)
    plt.close(fig)


def save_spatial_sampling_grid(
    mask_src: np.ndarray,
    mask_tgt: np.ndarray,
    source_centroids: np.ndarray,
    target_centroids: np.ndarray,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    grid_info: dict,
    output_path: Path,
    pcc_shift: np.ndarray = None,
    downsample: int = 8,
):
    """
    Visualize spatial grid sampling strategy in post-PCC space.

    Shows grid overlay on well with sampled source (green) and target (red) centroids.
    Source centroids are expected to be PCC-aligned already.

    Parameters
    ----------
    mask_src : np.ndarray
        Source segmentation mask (original, not shifted).
    mask_tgt : np.ndarray
        Target segmentation mask.
    source_centroids : np.ndarray
        All source centroids (N, 2) - should be PCC-aligned.
    target_centroids : np.ndarray
        All target centroids (M, 2).
    source_indices : np.ndarray
        Indices of sampled source cells.
    target_indices : np.ndarray
        Indices of sampled target cells.
    grid_info : dict
        Grid bounds from spatial_grid_subsample().
    output_path : Path
        Output image path.
    pcc_shift : np.ndarray, optional
        PCC shift [dy, dx] to apply to source mask for visualization.
    downsample : int
        Downsampling factor for mask visualization.
    """
    # Apply PCC shift to source mask first (to match PCC-aligned centroids)
    if pcc_shift is not None:
        from scipy import ndimage

        mask_src_shifted = ndimage.shift(mask_src, pcc_shift, order=0)
    else:
        mask_src_shifted = mask_src

    # Downsample masks
    if downsample > 1:
        mask_src_ds = downscale_local_mean(mask_src_shifted, (downsample, downsample))
        mask_tgt_ds = downscale_local_mean(mask_tgt, (downsample, downsample))
    else:
        mask_src_ds = mask_src_shifted
        mask_tgt_ds = mask_tgt

    # Ensure masks have same shape (they may differ slightly)
    min_h = min(mask_src_ds.shape[0], mask_tgt_ds.shape[0])
    min_w = min(mask_src_ds.shape[1], mask_tgt_ds.shape[1])
    mask_src_ds = mask_src_ds[:min_h, :min_w]
    mask_tgt_ds = mask_tgt_ds[:min_h, :min_w]

    # Create base overlay (target=red, source=green)
    binary_tgt = (mask_tgt_ds > 0).astype(np.uint8) * 255
    binary_src = (mask_src_ds > 0).astype(np.uint8) * 255

    overlay = np.zeros((*binary_tgt.shape, 3), dtype=np.uint8)
    overlay[..., 0] = binary_tgt  # Red
    overlay[..., 1] = binary_src  # Green

    # Create figure
    fig = Figure(figsize=(14, 14), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)

    # Show base overlay (very transparent to focus on centroids)
    ax.imshow(overlay, alpha=0.15)

    # Draw grid lines
    y_bins = grid_info["y_bins"] / downsample
    x_bins = grid_info["x_bins"] / downsample

    for y in y_bins:
        ax.axhline(y, color="white", linewidth=0.5, alpha=0.5)
    for x in x_bins:
        ax.axvline(x, color="white", linewidth=0.5, alpha=0.5)

    # Highlight selected bins with yellow boxes
    if "selected_bins" in grid_info:
        from matplotlib.patches import Rectangle

        for i, j in grid_info["selected_bins"]:
            y_start = y_bins[i]
            y_end = y_bins[i + 1]
            x_start = x_bins[j]
            x_end = x_bins[j + 1]
            width = x_end - x_start
            height = y_end - y_start
            rect = Rectangle(
                (x_start, y_start),
                width,
                height,
                linewidth=2,
                edgecolor="yellow",
                facecolor="none",
                alpha=0.8,
            )
            ax.add_patch(rect)

    # Plot sampled centroids
    sampled_source = source_centroids[source_indices] / downsample
    sampled_target = target_centroids[target_indices] / downsample

    ax.scatter(
        sampled_source[:, 1],
        sampled_source[:, 0],
        c="lime",
        s=2,
        alpha=0.8,
        label=f"Source ({len(source_indices)} cells)",
    )
    ax.scatter(
        sampled_target[:, 1],
        sampled_target[:, 0],
        c="red",
        s=2,
        alpha=0.8,
        label=f"Target ({len(target_indices)} cells)",
    )

    n_selected = len(grid_info.get("selected_bins", []))
    total_bins = grid_info["grid_size"] ** 2
    ax.set_title(
        f"Spatial Grid Sampling: {n_selected}/{total_bins} bins selected",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.axis("off")

    fig.tight_layout()

    # Render to array
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
    buf_rgb = buf[:, :, :3]

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    skimage.io.imsave(output_path, buf_rgb)
    plt.close(fig)


def save_centroid_overlay(
    mask: np.ndarray,
    centroids: np.ndarray,
    output_path: Path,
    title: str = "Centroids",
    color: str = "red",
    downsample: int = 4,
):
    """
    Save overlay of centroids on mask.

    Parameters
    ----------
    mask : np.ndarray
        Segmentation mask (2D).
    centroids : np.ndarray
        Centroid coordinates (N, 2) in (y, x) format.
    output_path : Path
        Output image path.
    title : str
        Title for the plot.
    color : str
        Color for centroid markers.
    downsample : int
        Downsampling factor for display.
    """
    # Downsample mask for visualization
    if downsample > 1:
        mask_ds = downscale_local_mean(mask, (downsample, downsample))
        centroids_ds = centroids / downsample
    else:
        mask_ds = mask
        centroids_ds = centroids

    # Create binary mask for display
    binary_mask = (mask_ds > 0).astype(np.uint8) * 255

    # Create figure
    fig = Figure(figsize=(10, 10), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)

    # Show mask (dim background)
    ax.imshow(binary_mask, cmap="gray", alpha=0.3)

    # Overlay centroids with bright, visible markers
    # Use much larger size and bright colors with strong edge contrast
    if color == "green":
        marker_color = "lime"
        edge_color = "darkgreen"
    elif color == "red":
        marker_color = "red"
        edge_color = "darkred"
    else:
        marker_color = color
        edge_color = "black"

    # Use + markers which are more visible, much larger size (200), thick lines
    ax.scatter(
        centroids_ds[:, 1],
        centroids_ds[:, 0],
        c=marker_color,
        s=200,
        alpha=1.0,
        marker="+",
        linewidths=3,
        edgecolors=edge_color,
    )

    ax.set_title(f"{title} ({len(centroids)} cells)", fontsize=12, fontweight="bold")
    ax.axis("off")

    # Render and save
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
    buf_rgb = buf[:, :, :3]

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    skimage.io.imsave(output_path, buf_rgb)
    plt.close(fig)


def _save_final_detail_grid_mp(args):
    """Module-level wrapper for ProcessPoolExecutor (unpacks tuple args)."""
    mask_pcc, mask_final, mask_tgt, filename = args
    _save_final_detail_grid(mask_pcc, mask_final, mask_tgt, Path(filename))


def _save_final_detail_grid(
    mask_src_crop_pcc: np.ndarray,
    mask_src_crop_final: np.ndarray,
    mask_tgt_crop: np.ndarray,
    filename: Path,
):
    """
    Save 3x2 grid: [target, source, overlay] x [before RANSAC, after RANSAC].

    BEFORE = post-PCC (shows RANSAC contribution)
    AFTER = post-PCC + RANSAC (final alignment)
    """
    # Create binary masks
    binary_tgt = (mask_tgt_crop > 0).astype(np.uint8) * 255
    binary_src_before = (mask_src_crop_pcc > 0).astype(np.uint8) * 255
    binary_src_after = (mask_src_crop_final > 0).astype(np.uint8) * 255

    # Create colored RGB versions to match overlay colors
    # Target = red, Source = green (matching create_overlay_rgb)
    tgt_rgb = np.stack(
        [binary_tgt, np.zeros_like(binary_tgt), np.zeros_like(binary_tgt)], axis=-1
    )
    src_before_rgb = np.stack(
        [
            np.zeros_like(binary_src_before),
            binary_src_before,
            np.zeros_like(binary_src_before),
        ],
        axis=-1,
    )
    src_after_rgb = np.stack(
        [
            np.zeros_like(binary_src_after),
            binary_src_after,
            np.zeros_like(binary_src_after),
        ],
        axis=-1,
    )

    # Create overlays
    overlay_before = create_overlay_rgb(mask_tgt_crop, mask_src_crop_pcc)
    overlay_after = create_overlay_rgb(mask_tgt_crop, mask_src_crop_final)

    # Create figure with 3x2 grid
    fig = Figure(figsize=(12, 8), dpi=100)
    canvas = FigureCanvasAgg(fig)

    # Row 1: Before RANSAC
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(tgt_rgb)
    ax1.set_title("Target (red)", fontsize=10, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(src_before_rgb)
    ax2.set_title("Source BEFORE (green)", fontsize=10, fontweight="bold")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(overlay_before)
    ax3.set_title("Overlay BEFORE", fontsize=10, fontweight="bold")
    ax3.axis("off")

    # Row 2: After RANSAC
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(tgt_rgb)
    ax4.set_title("Target (red)", fontsize=10, fontweight="bold")
    ax4.axis("off")

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(src_after_rgb)
    ax5.set_title("Source AFTER (green)", fontsize=10, fontweight="bold")
    ax5.axis("off")

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.imshow(overlay_after)
    ax6.set_title("Overlay AFTER", fontsize=10, fontweight="bold")
    ax6.axis("off")

    fig.tight_layout()

    # Render to array
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
    buf_rgb = buf[:, :, :3]

    skimage.io.imsave(filename, buf_rgb)
    plt.close(fig)


def _save_manual_vs_auto_grid(
    mask_src_crop_manual: np.ndarray,
    mask_src_crop_auto: np.ndarray,
    mask_tgt_crop: np.ndarray,
    filename: Path,
):
    """
    Save 2x3 grid comparing manual vs automatic registration.

    Layout:
    Row 1 (Manual): [Target, Source, Overlay]
    Row 2 (Auto):   [Target, Source, Overlay]

    This allows direct visual comparison of manual vs automatic alignment quality.
    """
    # Create binary masks
    binary_tgt = (mask_tgt_crop > 0).astype(np.uint8) * 255
    binary_src_manual = (mask_src_crop_manual > 0).astype(np.uint8) * 255
    binary_src_auto = (mask_src_crop_auto > 0).astype(np.uint8) * 255

    # Create colored RGB versions
    # Target = red, Source = green (matching create_overlay_rgb)
    tgt_rgb = np.stack(
        [binary_tgt, np.zeros_like(binary_tgt), np.zeros_like(binary_tgt)], axis=-1
    )
    src_manual_rgb = np.stack(
        [
            np.zeros_like(binary_src_manual),
            binary_src_manual,
            np.zeros_like(binary_src_manual),
        ],
        axis=-1,
    )
    src_auto_rgb = np.stack(
        [
            np.zeros_like(binary_src_auto),
            binary_src_auto,
            np.zeros_like(binary_src_auto),
        ],
        axis=-1,
    )

    # Create overlays
    overlay_manual = create_overlay_rgb(mask_tgt_crop, mask_src_crop_manual)
    overlay_auto = create_overlay_rgb(mask_tgt_crop, mask_src_crop_auto)

    # Create figure with 2x3 grid
    fig = Figure(figsize=(12, 8), dpi=100)
    canvas = FigureCanvasAgg(fig)

    # Row 1: Manual Registration
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(tgt_rgb)
    ax1.set_title("Target (red)", fontsize=10, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(src_manual_rgb)
    ax2.set_title("Source MANUAL (green)", fontsize=10, fontweight="bold")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(overlay_manual)
    ax3.set_title("Overlay MANUAL", fontsize=10, fontweight="bold")
    ax3.axis("off")

    # Row 2: Automatic Registration
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(tgt_rgb)
    ax4.set_title("Target (red)", fontsize=10, fontweight="bold")
    ax4.axis("off")

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(src_auto_rgb)
    ax5.set_title("Source AUTO (green)", fontsize=10, fontweight="bold")
    ax5.axis("off")

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.imshow(overlay_auto)
    ax6.set_title("Overlay AUTO", fontsize=10, fontweight="bold")
    ax6.axis("off")

    fig.tight_layout()

    # Render to array
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
    buf_rgb = buf[:, :, :3]

    skimage.io.imsave(filename, buf_rgb)
    plt.close(fig)


def visualize_hu_moment_matches(
    mask_src: np.ndarray,
    mask_tgt: np.ndarray,
    all_source_centroids: np.ndarray,
    all_target_centroids: np.ndarray,
    hu_source_idx: np.ndarray,
    hu_target_idx: np.ndarray,
    hu_distances: np.ndarray,
    output_path: Path,
    pcc_shift: np.ndarray = None,
    source_graphs: dict = None,
    target_graphs: dict = None,
    n_display: int = 100,
    crop_size: int = 64,
    n_cols: int = 10,
    verbose: bool = True,
):
    """
    Visualize graph-matched cell pairs in multi-column grid layout.

    Creates a grid showing n_display matched pairs in n_cols columns, with source on left
    and target on right for each pair. Pairs are sorted by graph score (best to worst).
    Color-coded by match quality (green=excellent, yellow=good, orange=fair).

    Parameters
    ----------
    mask_src : np.ndarray
        Source segmentation mask (original, not shifted).
    mask_tgt : np.ndarray
        Target segmentation mask.
    all_source_centroids : np.ndarray
        (N, 2) all source centroids (PCC-aligned).
    all_target_centroids : np.ndarray
        (M, 2) all target centroids.
    hu_source_idx : np.ndarray
        Indices of graph-matched source cells.
    hu_target_idx : np.ndarray
        Indices of graph-matched target cells.
    hu_distances : np.ndarray
        Graph consistency scores for each match (lower is better).
    output_path : Path
        Output image path.
    pcc_shift : np.ndarray, optional
        PCC shift [dy, dx] to apply to source mask.
    n_display : int
        Number of cell pairs to display (default: 100).
    crop_size : int
        Size of crop around each cell (default: 64px for compact grid).
    n_cols : int
        Number of pair columns in grid (default: 10, showing 10 pairs per row).
    verbose : bool
        Print progress.
    """
    if len(hu_source_idx) == 0:
        if verbose:
            print(f"      No graph matches to visualize")
        return

    # Apply PCC shift to source mask
    if pcc_shift is not None:
        mask_src_shifted = ndimage.shift(mask_src, pcc_shift, order=0)
    else:
        mask_src_shifted = mask_src

    # Select n_display matches (best matches sorted by graph score)
    n_matches = len(hu_source_idx)
    n_display = min(n_display, n_matches)

    # Sort by graph score (lower is better)
    sorted_indices = np.argsort(hu_distances)[:n_display]

    # Get matched centroids (best matches)
    matched_source = all_source_centroids[hu_source_idx[sorted_indices]]
    matched_target = all_target_centroids[hu_target_idx[sorted_indices]]
    matched_scores = hu_distances[sorted_indices]

    # Calculate grid dimensions: n_cols pairs per row, each pair has 2 images (source+target)
    n_rows = int(np.ceil(n_display / n_cols))

    # Figure size: 2 inches per pair (source+target), with rows scaled appropriately
    fig_width = n_cols * 2
    fig_height = n_rows * 2

    # Add extra spacing between pairs (wspace controls horizontal spacing between subplots)
    fig = Figure(figsize=(fig_width, fig_height), dpi=100)
    canvas = FigureCanvasAgg(fig)

    # Adjust subplot spacing: add whitespace between pair columns
    fig.subplots_adjust(wspace=0.1, hspace=0.5)  # wspace adds horizontal gap between pairs

    for idx in range(n_display):
        src_idx_global = hu_source_idx[sorted_indices[idx]]
        tgt_idx_global = hu_target_idx[sorted_indices[idx]]
        src_y, src_x = matched_source[idx]
        tgt_y, tgt_x = matched_target[idx]
        graph_score = matched_scores[idx]

        # Determine color based on graph score (green=excellent, yellow=good, orange=fair)
        if graph_score < 0.3:
            color = "green"
            quality = "excellent"
        elif graph_score < 0.5:
            color = "yellow"
            quality = "good"
        else:
            color = "orange"
            quality = "fair"

        # Calculate subplot position
        row = idx // n_cols
        col = idx % n_cols

        # Each pair takes 2 subplot positions (source + target)
        total_cols = n_cols * 2  # Each pair column contains 2 images

        # Crop around source cell
        src_y_start = int(max(0, src_y - crop_size // 2))
        src_y_end = int(min(mask_src_shifted.shape[0], src_y + crop_size // 2))
        src_x_start = int(max(0, src_x - crop_size // 2))
        src_x_end = int(min(mask_src_shifted.shape[1], src_x + crop_size // 2))

        src_crop = mask_src_shifted[src_y_start:src_y_end, src_x_start:src_x_end]
        src_centroid_crop = np.array([src_y - src_y_start, src_x - src_x_start])

        # Crop around target cell
        tgt_y_start = int(max(0, tgt_y - crop_size // 2))
        tgt_y_end = int(min(mask_tgt.shape[0], tgt_y + crop_size // 2))
        tgt_x_start = int(max(0, tgt_x - crop_size // 2))
        tgt_x_end = int(min(mask_tgt.shape[1], tgt_x + crop_size // 2))

        tgt_crop = mask_tgt[tgt_y_start:tgt_y_end, tgt_x_start:tgt_x_end]
        tgt_centroid_crop = np.array([tgt_y - tgt_y_start, tgt_x - tgt_x_start])

        # Source (left of pair)
        subplot_idx_src = row * total_cols + col * 2 + 1
        ax_src = fig.add_subplot(n_rows, total_cols, subplot_idx_src)
        ax_src.imshow(src_crop > 0, cmap="gray")

        # Draw graph edges for source if available
        if source_graphs is not None and src_idx_global in source_graphs:
            src_graph = source_graphs[src_idx_global]
            neighbor_indices = src_graph["neighbor_indices"]
            neighbor_dists = src_graph["neighbor_distances"]

            # Draw edges to neighbors
            for i, (neighbor_idx, dist) in enumerate(zip(neighbor_indices, neighbor_dists)):
                neighbor_centroid = all_source_centroids[neighbor_idx]
                # Check if neighbor is in crop
                if (src_y_start <= neighbor_centroid[0] < src_y_end and
                    src_x_start <= neighbor_centroid[1] < src_x_end):
                    neighbor_y_crop = neighbor_centroid[0] - src_y_start
                    neighbor_x_crop = neighbor_centroid[1] - src_x_start
                    ax_src.plot([src_centroid_crop[1], neighbor_x_crop],
                               [src_centroid_crop[0], neighbor_y_crop],
                               'cyan', alpha=0.5, linewidth=1)
                    ax_src.scatter([neighbor_x_crop], [neighbor_y_crop],
                                 c='cyan', s=10, marker='o', alpha=0.7)

        ax_src.scatter(
            [src_centroid_crop[1]],
            [src_centroid_crop[0]],
            c="lime",
            s=50,
            marker="+",
            linewidths=2,
        )

        # Add edge count annotation if graphs available
        edge_text = ""
        if source_graphs is not None and src_idx_global in source_graphs:
            n_edges = len(source_graphs[src_idx_global]["neighbor_indices"])
            mean_dist = np.mean(source_graphs[src_idx_global]["neighbor_distances"])
            edge_text = f"S{idx+1} | {n_edges}nbr | {mean_dist:.0f}px"
        else:
            edge_text = f"S{idx+1}"

        ax_src.set_title(edge_text, fontsize=6, fontweight="bold", color=color)
        ax_src.axis("off")

        # Target (right of pair)
        subplot_idx_tgt = row * total_cols + col * 2 + 2
        ax_tgt = fig.add_subplot(n_rows, total_cols, subplot_idx_tgt)
        ax_tgt.imshow(tgt_crop > 0, cmap="gray")

        # Draw graph edges for target if available
        if target_graphs is not None and tgt_idx_global in target_graphs:
            tgt_graph = target_graphs[tgt_idx_global]
            neighbor_indices = tgt_graph["neighbor_indices"]
            neighbor_dists = tgt_graph["neighbor_distances"]

            # Draw edges to neighbors
            for i, (neighbor_idx, dist) in enumerate(zip(neighbor_indices, neighbor_dists)):
                neighbor_centroid = all_target_centroids[neighbor_idx]
                # Check if neighbor is in crop
                if (tgt_y_start <= neighbor_centroid[0] < tgt_y_end and
                    tgt_x_start <= neighbor_centroid[1] < tgt_x_end):
                    neighbor_y_crop = neighbor_centroid[0] - tgt_y_start
                    neighbor_x_crop = neighbor_centroid[1] - tgt_x_start
                    ax_tgt.plot([tgt_centroid_crop[1], neighbor_x_crop],
                               [tgt_centroid_crop[0], neighbor_y_crop],
                               'magenta', alpha=0.5, linewidth=1)
                    ax_tgt.scatter([neighbor_x_crop], [neighbor_y_crop],
                                 c='magenta', s=10, marker='o', alpha=0.7)

        ax_tgt.scatter(
            [tgt_centroid_crop[1]],
            [tgt_centroid_crop[0]],
            c="red",
            s=50,
            marker="+",
            linewidths=2,
        )

        # Add edge count and score annotation if graphs available
        tgt_text = f"T{idx+1} | score={graph_score:.3f}"
        if target_graphs is not None and tgt_idx_global in target_graphs:
            n_edges = len(target_graphs[tgt_idx_global]["neighbor_indices"])
            mean_dist = np.mean(target_graphs[tgt_idx_global]["neighbor_distances"])
            tgt_text = f"T{idx+1} | {n_edges}nbr | {mean_dist:.0f}px | s={graph_score:.3f}"

        ax_tgt.set_title(tgt_text, fontsize=5, fontweight="bold", color=color)
        ax_tgt.axis("off")

    fig.suptitle(
        f"Graph-Matched Cell Pairs (n={n_display}/{n_matches}, sorted by quality)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    # Render to array
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
    buf_rgb = buf[:, :, :3]

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    skimage.io.imsave(output_path, buf_rgb)
    plt.close(fig)

    if verbose:
        print(f"      Saved graph match visualization: {output_path.name}")
