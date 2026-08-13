"""
Phase Cross-Correlation (PCC) pre-alignment module.

Contains functions for:
- Multi-approach PCC estimation (binary, edges, distance transform, blurred)
- Adaptive downsampling based on image size
- Translation-only alignment using phase correlation
- PCC result caching for faster repeated runs
"""

import numpy as np
import json
import hashlib
import zarr
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from skimage.transform import downscale_local_mean
from skimage.registration import phase_cross_correlation


def _get_pcc_cache_path(
    source_seg_path: Path,
    target_seg_path: Path,
    source_position: str,
    target_position: str,
    t_idx: int,
    downsample_factor: int,
    center_fraction: float = 1.0,
) -> Path:
    """Generate cache file path for PCC results."""
    from cyclops_process.processes.auto_register.auto_register_utils import (
        _get_modality_from_path,
        _format_position,
        _create_cache_hash,
    )

    # Create a unique hash from input parameters
    cache_key = f"{source_seg_path}_{target_seg_path}_{source_position}_{target_position}_{t_idx}_{downsample_factor}_{center_fraction}"
    cache_hash = _create_cache_hash(cache_key)

    # Store cache in 2-tracking/cache directory (visible to user)
    # Navigate from segmentation path to experiment root, then to 2-tracking/cache
    # e.g., /path/to/ops0031_20250424/1-preprocess/.../segmentation -> /path/to/ops0031_20250424/2-tracking/cache
    experiment_root = target_seg_path.parents[
        3
    ]  # Go up from 1-preprocess/modality/segmentation
    cache_dir = experiment_root / "2-tracking" / "cache" / "pcc"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create human-readable prefix
    src_mod = _get_modality_from_path(source_seg_path)
    tgt_mod = _get_modality_from_path(target_seg_path)
    pos_str = _format_position(source_position)
    cf_str = f"cf{center_fraction:.1f}".replace(".", "p")
    human_prefix = f"pcc_{src_mod}_to_{tgt_mod}_{pos_str}_t{t_idx}_ds{downsample_factor}_{cf_str}"

    return cache_dir / f"{human_prefix}_{cache_hash}.json"


def estimate_translation_pcc(
    source_seg_path: Path,
    target_seg_path: Path,
    source_position: str,
    target_position: str,
    t_idx: int = 0,
    downsample_factor: int = 8,
    use_cache: bool = False,
    center_fraction: float = 1.0,
) -> tuple[np.ndarray, bool]:
    """
    Estimate translation offset using phase cross-correlation on binary masks.

    Results are cached for faster repeated runs with the same parameters.

    Parameters
    ----------
    source_seg_path : Path
        Source segmentation zarr path.
    target_seg_path : Path
        Target segmentation zarr path.
    source_position : str
        Source position (e.g., "A/1/0").
    target_position : str
        Target position (e.g., "A/1/0").
    t_idx : int
        Time index.
    downsample_factor : int
        Factor to downsample images for faster PCC.
    use_cache : bool
        Whether to use cached PCC results if available.
    center_fraction : float
        Fraction of image to use from center (0-1, default: 1.0 = full image).
        Using center region (e.g., 0.5 = 50%) ignores edges with stitching artifacts.

    Returns
    -------
    tuple
        (shift, cache_hit)
        - shift: Translation offset (dy, dx) in pixels
        - cache_hit: True if loaded from cache, False if computed
    """
    # Check cache first
    if use_cache:
        cache_path = _get_pcc_cache_path(
            source_seg_path,
            target_seg_path,
            source_position,
            target_position,
            t_idx,
            downsample_factor,
            center_fraction,
        )
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    cache_data = json.load(f)
                shift = np.array(cache_data["shift"])
                print(
                    f"    Using cached PCC shift: dy={shift[0]:.1f}, dx={shift[1]:.1f}"
                )
                print(f"    Cache: {cache_path}")
                return shift, True  # Cache hit
            except Exception as e:
                print(f"    WARNING: Failed to load PCC cache: {e}")
                # Continue with normal computation
    # Load source and target masks in parallel (each uses 16-thread chunk reads)
    from concurrent.futures import ThreadPoolExecutor as _IOPool
    from cyclops_process.processes.auto_register.auto_register_utils import _load_seg_2d

    with _IOPool(max_workers=2) as io_pool:
        src_future = io_pool.submit(_load_seg_2d, source_seg_path, source_position, t_idx)
        tgt_future = io_pool.submit(_load_seg_2d, target_seg_path, target_position, t_idx)
        mask_src = src_future.result()
        mask_tgt = tgt_future.result()

    # Convert to binary
    binary_src = (mask_src > 0).astype(np.float32)
    binary_tgt = (mask_tgt > 0).astype(np.float32)

    # Check for empty masks before continuing
    src_coverage = binary_src.mean()
    tgt_coverage = binary_tgt.mean()
    print(f"    DEBUG: Mask coverage - src: {src_coverage:.3%}, tgt: {tgt_coverage:.3%}")

    if src_coverage < 0.001 or tgt_coverage < 0.001:
        print(f"    WARNING: Very low mask coverage detected (src: {src_coverage:.3%}, tgt: {tgt_coverage:.3%})")
        print(f"    PCC may fail - consider checking segmentation quality")

    # Downsample for speed
    if downsample_factor > 1:
        binary_src = downscale_local_mean(
            binary_src, (downsample_factor, downsample_factor)
        )
        binary_tgt = downscale_local_mean(
            binary_tgt, (downsample_factor, downsample_factor)
        )

    # Crop to center fraction around the center of mass of each mask.
    # Using center-of-mass (not image center) ensures the crop captures
    # the densest cell region even if the well is off-center.
    if center_fraction < 1.0:
        from scipy.ndimage import center_of_mass as _com

        def crop_around_com(img, fraction):
            h, w = img.shape
            crop_h = int(h * fraction)
            crop_w = int(w * fraction)
            # Use center of mass if there's signal, else fall back to image center
            if img.sum() > 0:
                cy, cx = _com(img)
                cy, cx = int(round(cy)), int(round(cx))
            else:
                cy, cx = h // 2, w // 2
            # Clamp so the crop stays within bounds
            start_h = max(0, min(cy - crop_h // 2, h - crop_h))
            start_w = max(0, min(cx - crop_w // 2, w - crop_w))
            return img[start_h : start_h + crop_h, start_w : start_w + crop_w]

        binary_src = crop_around_com(binary_src, center_fraction)
        binary_tgt = crop_around_com(binary_tgt, center_fraction)
        print(
            f"    DEBUG: Cropped to {center_fraction:.0%} around center-of-mass - "
            f"new shapes: src={binary_src.shape}, tgt={binary_tgt.shape}"
        )

    # Pad to same shape if needed (for phenotyping vs tracking which may differ slightly)
    if binary_src.shape != binary_tgt.shape:
        max_h = max(binary_src.shape[0], binary_tgt.shape[0])
        max_w = max(binary_src.shape[1], binary_tgt.shape[1])

        # Pad source
        pad_h_src = max_h - binary_src.shape[0]
        pad_w_src = max_w - binary_src.shape[1]
        if pad_h_src > 0 or pad_w_src > 0:
            binary_src = np.pad(
                binary_src,
                ((0, pad_h_src), (0, pad_w_src)),
                mode="constant",
                constant_values=0,
            )

        # Pad target
        pad_h_tgt = max_h - binary_tgt.shape[0]
        pad_w_tgt = max_w - binary_tgt.shape[1]
        if pad_h_tgt > 0 or pad_w_tgt > 0:
            binary_tgt = np.pad(
                binary_tgt,
                ((0, pad_h_tgt), (0, pad_w_tgt)),
                mode="constant",
                constant_values=0,
            )

    # Try multiple preprocessing approaches in parallel and pick the best one
    from scipy.ndimage import gaussian_filter, distance_transform_edt
    from skimage.filters import sobel

    print(f"    DEBUG: Trying PCC with downsample_factor={downsample_factor}")
    print(
        f"    DEBUG: Downsampled shapes - src: {binary_src.shape}, tgt: {binary_tgt.shape}"
    )

    # Prepare all 4 approaches (preprocessing is fast after downsampling)
    edge_src = sobel(binary_src)
    edge_tgt = sobel(binary_tgt)

    dist_src = distance_transform_edt(binary_src)
    dist_tgt = distance_transform_edt(binary_tgt)
    if dist_src.max() > 0:
        dist_src = dist_src / dist_src.max()
    if dist_tgt.max() > 0:
        dist_tgt = dist_tgt / dist_tgt.max()

    blur_src = gaussian_filter(binary_src, sigma=1.0)
    blur_tgt = gaussian_filter(binary_tgt, sigma=1.0)

    approaches = [
        ("binary", binary_src, binary_tgt),
        ("edges", edge_src, edge_tgt),
        ("distance", dist_src, dist_tgt),
        ("blurred", blur_src, blur_tgt),
    ]

    # Run all 4 PCC approaches concurrently
    best_shift = None
    best_error = float("inf")
    best_approach = None

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                phase_cross_correlation, tgt, src, upsample_factor=10, normalization=None
            ): name
            for name, src, tgt in approaches
        }
        for future in futures:
            name = futures[future]
            try:
                shift, error, diffphase = future.result()
                print(f"    DEBUG: {name:10s} - shift: {shift}, error: {error:.6f}")
                if np.isfinite(error) and error < best_error:
                    best_error = error
                    best_shift = shift
                    best_approach = name
            except Exception as e:
                print(f"    DEBUG: {name:10s} - FAILED: {e}")

    shift = best_shift
    print(f"    DEBUG: Best approach: {best_approach} (error: {best_error:.6f})")

    print(f"    DEBUG: Raw PCC shift (downsampled): {shift}, error: {best_error:.6f}")

    # Check if PCC failed (all approaches returned NaN or invalid shifts)
    if shift is None or not np.isfinite(best_error):
        print(f"    WARNING: PCC failed (all approaches returned invalid results)")
        print(f"    Falling back to zero shift (identity transform)")
        shift = np.array([0.0, 0.0])
    else:
        # Scale shift back to original resolution
        shift = shift * downsample_factor

    print(f"    DEBUG: Scaled PCC shift (full res): {shift}")

    # Save to cache for future runs
    if use_cache:
        cache_path = _get_pcc_cache_path(
            source_seg_path,
            target_seg_path,
            source_position,
            target_position,
            t_idx,
            downsample_factor,
            center_fraction,
        )
        try:
            cache_data = {
                "shift": shift.tolist(),
                "error": float(best_error),
                "approach": best_approach,
                "downsample_factor": downsample_factor,
            }
            with open(cache_path, "w") as f:
                json.dump(cache_data, f, indent=2)
            print(f"    Saved PCC result to cache: {cache_path}")
        except Exception as e:
            print(f"    WARNING: Failed to save PCC cache: {e}")

    return shift, False  # (dy, dx), not from cache
