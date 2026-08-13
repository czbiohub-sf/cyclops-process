"""Flatfield (illumination) correction for 20x phenotyping fluorescence channels.

Estimates per-channel flatfield profiles from the images themselves (no calibration
data required) and applies a multiplicative correction to each position.

Two estimation methods:
  1. max (default) — mean+smooth, max-normalized (only boosts dim areas)
  2. mean — mean+smooth, mean-normalized with auto-strength optimization

Usage:
    python flatfield_correction.py <experiment> [--method max|mean] [--num-samples 500]
"""

import os
import sys
import argparse
import time

import numpy as np
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from iohub import open_ome_zarr
from iohub.ngff import TransformationMeta

sys.path.insert(0, os.getcwd())
from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.hpc.resource_manager import get_optimal_workers
from cyclops_utils.io.zarr_utils import (
    _discover_positions,
    _maybe_sample_positions,
    _resolve_output_path_for_debug,
)
from cyclops_utils.io.zarr_precreate import create_hcs_store_fast
from cyclops_utils.data.filesystem import async_delete_path
from cyclops_utils.profiling.decorators import versioned_function


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_single_position(source_path: str, pos_name: str, ch_idx: int, z_method: str) -> np.ndarray:
    """Read and max-project (or mid-slice) a single position/channel. Thread-safe via zarr."""
    import zarr as zarr_mod

    arr = zarr_mod.open(str(Path(source_path) / pos_name / "0"), mode="r")
    vol = np.asarray(arr[0, ch_idx])  # (Z, H, W)
    if z_method == "max_projection":
        return vol.max(axis=0).astype(np.float32)
    else:
        return vol[vol.shape[0] // 2].astype(np.float32)



# ---------------------------------------------------------------------------
# Flatfield estimation
# ---------------------------------------------------------------------------



def estimate_flatfield_max(
    source_path: str | Path,
    channel_indices: list[int],
    num_samples: int = 500,
    num_workers: int = 8,
    sigma: int | None = None,
    camera_offset: float = 100.0,
    z_method: str = "max_projection",
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Estimate flatfield using Gaussian-smoothed mean, max-normalized.

    Camera offset is subtracted before computing the mean so the flatfield
    profile represents the true illumination pattern of the signal only,
    not the signal + constant dark current.

    Returns dict mapping channel_index -> (flatfield, darkfield, sample_stack).
    """
    from scipy.ndimage import gaussian_filter

    print(f"    Discovering positions (fast glob)...")
    t_open = time.time()
    positions = _discover_positions(Path(source_path))
    print(f"    Found {len(positions)} positions in {time.time() - t_open:.1f}s")

    if num_samples and len(positions) > num_samples:
        rng = np.random.default_rng(42)
        positions = list(rng.choice(positions, size=num_samples, replace=False))

    results: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for ch_idx in channel_indices:
        t0 = time.time()
        print(f"  Estimating flatfield for channel {ch_idx} using max method ({len(positions)} images)...")

        print(f"    Reading {len(positions)} images with {num_workers} threads...")
        images = [None] * len(positions)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {
                executor.submit(_read_single_position, str(source_path), pos, ch_idx, z_method): i
                for i, pos in enumerate(positions)
            }
            for future in tqdm(as_completed(future_to_idx), total=len(positions), desc=f"    Reading ch{ch_idx}"):
                images[future_to_idx[future]] = future.result()
        t_read = time.time() - t0
        print(f"    Read {len(positions)} images in {t_read:.1f}s")

        t1 = time.time()
        print(f"    Stacking {len(images)} images...")
        stack = np.stack(images, axis=0)  # (N, H, W)
        print(f"      Stack shape: {stack.shape}, {stack.nbytes / 1e9:.1f} GB  ({time.time() - t1:.1f}s)")

        t2 = time.time()
        print(f"    Subtracting camera offset ({camera_offset}) and computing mean across {stack.shape[0]} images...")
        stack_sub = np.clip(stack - camera_offset, 0, None)
        mean_img = np.mean(stack_sub, axis=0)  # (H, W)
        del stack_sub
        print(f"      Mean done ({time.time() - t2:.1f}s)")

        t3 = time.time()
        _sigma = sigma if sigma is not None else 75
        print(f"    Gaussian smoothing (sigma={_sigma})...")
        smoothed = gaussian_filter(mean_img, sigma=_sigma)

        # Fractal normalization: divide by max so profile is in (0, 1]
        # Correction = raw / profile -> only boosts dim areas
        flatfield = smoothed / smoothed.max()
        # Clamp floor to avoid extreme boost at very dim edges
        flatfield = np.clip(flatfield, 0.2, 1.0)

        darkfield = np.zeros_like(flatfield)
        t_compute = time.time() - t1
        print(f"      Smoothing done ({time.time() - t3:.1f}s)")

        results[ch_idx] = (flatfield, darkfield, stack)
        print(f"    Flatfield range: [{flatfield.min():.4f}, {flatfield.max():.4f}]")
        print(f"    Compute in {t_compute:.1f}s  |  Total: {time.time() - t0:.1f}s")

    return results



def _spatial_cv(image: np.ndarray, grid: int = 8) -> float:
    """Coefficient of variation of block means across an NxN grid.

    Lower CV = more spatially uniform illumination.
    """
    h, w = image.shape
    bh, bw = h // grid, w // grid
    block_means = np.array([
        image[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw].mean()
        for r in range(grid) for c in range(grid)
    ])
    return block_means.std() / block_means.mean()


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

def _flatfield_correct_worker(
    pos: str,
    source_path: str,
    corrected_path: str,
    corrections: dict[int, tuple[np.ndarray, np.ndarray]],
    camera_offset: float = 100.0,
):
    """Apply flatfield correction to one position (all channels in corrections dict).

    Subtracts camera offset before correction and adds it back after,
    preventing the constant dark current from being distorted by the
    illumination profile division.
    """
    import zarr as zarr_mod

    src_arr = zarr_mod.open(str(Path(source_path) / pos / "0"), mode="r")
    data = np.asarray(src_arr)  # (T, C, Z, H, W)
    corrected = data.copy().astype(np.float32)

    for ch_idx, (flatfield, darkfield) in corrections.items():
        # Subtract camera offset + darkfield, divide by flatfield, add offset back
        corrected[:, ch_idx] = (
            (corrected[:, ch_idx] - camera_offset - darkfield[np.newaxis, np.newaxis])
            / flatfield[np.newaxis, np.newaxis]
            + camera_offset
        )

    # Cast back to original dtype
    corrected = np.clip(corrected, 0, np.iinfo(data.dtype).max if np.issubdtype(data.dtype, np.integer) else np.finfo(data.dtype).max)
    corrected = corrected.astype(data.dtype)

    dst_arr = zarr_mod.open(str(Path(corrected_path) / pos / "0"), mode="r+")
    dst_arr[:] = corrected


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@versioned_function("v1.0")
def correct_flatfield(
    experiment: str,
    process: str = "fluor",
    num_samples: int = 500,
    num_workers: int | None = None,
    fluor_channels: list[str] | None = None,
    sigma: int | None = None,
    camera_offset: float = 100.0,
    debug_n_positions: int | None = None,
    debug_output_suffix: str = "_debug",
    save_profiles: bool = True,
    source_path_override: Path | str | None = None,
    corrected_path_override: Path | str | None = None,
    output_dir_override: Path | str | None = None,
):
    """Estimate and apply flatfield correction to fluorescence channels.

    Uses max-normalized Gaussian-smoothed mean image as illumination profile.
    Correction only boosts dim areas (divides by profile normalized to max).
    Camera offset is subtracted before correction and added back after to prevent
    the constant dark current from being distorted by the illumination division.

    Parameters
    ----------
    experiment : str
        Experiment name (resolved via OpsDataset).
    num_samples : int
        Number of positions to sample for flatfield estimation.
    num_workers : int, optional
        Parallel workers for the correction pass.
    fluor_channels : list[str], optional
        Channel names to correct. If None, all non-BF/Phase channels are corrected.
    sigma : int, optional
        Gaussian sigma for smoothing. Default 75.
    camera_offset : float, optional
        Camera dark current offset to subtract before correction. If None,
        auto-estimated from the 1st percentile of sample images.
    debug_n_positions : int, optional
        Limit to N positions for testing.
    save_profiles : bool
        If True, save estimated flatfield/darkfield arrays as .npy files.
    source_path_override : Path, optional
        Override the source store path (default: resolved from OpsDataset).
    corrected_path_override : Path, optional
        Override the output store path (default: resolved from OpsDataset).
    """
    t_total = time.time()

    if num_workers is None:
        num_workers = get_optimal_workers(use_gpu=False, verbose=False)
    print(f"Using {num_workers} workers")

    dataset = OpsDataset(experiment)

    if source_path_override:
        source_path = Path(source_path_override)
        print(f"Using override source: {source_path}")
    else:
        # Source: the 20x phenotyping fluorescence store
        source_path = dataset.store_paths["lc_20x_fluor_2d"]
        if not source_path.exists():
            source_path = dataset.store_paths["lc_20x"]
            print(f"Using raw 20x store: {source_path}")
        else:
            print(f"Using fluor 2D store: {source_path}")

    if corrected_path_override:
        corrected_path = Path(corrected_path_override)
    else:
        corrected_path = dataset.store_paths["lc_20x_fluor_2d_flatfield"]

    # Discover positions (fast filesystem glob, no iohub plate parsing)
    print("Discovering positions (fast glob)...")
    t_meta = time.time()
    position_list = _discover_positions(Path(source_path))
    print(f"  Found {len(position_list)} positions in {time.time() - t_meta:.1f}s")

    # Read metadata from first position using layout="fov" (fast, no plate parsing)
    print("Reading metadata from first position...")
    first_pos = position_list[0]
    with open_ome_zarr(str(source_path / first_pos), layout="fov", mode="r") as pos_ds:
        channel_names = list(pos_ds.channel_names)
        output_scale = pos_ds.scale
        src_shape = pos_ds.data.shape
        dtype = pos_ds.data.dtype
        src_chunks = pos_ds.data.chunks
        output_transform = TransformationMeta(type="scale", scale=output_scale)

    print(f"Source store: {source_path}")
    print(f"  Channels: {channel_names}")
    print(f"  Shape: {src_shape}, dtype: {dtype}")
    print(f"  Total positions: {len(position_list)}")

    # Determine which channels are fluorescence
    non_fluor_keywords = {"BF", "Phase", "Retardance", "Orientation", "phase", "bf"}
    if fluor_channels is not None:
        ch_indices = [channel_names.index(ch) for ch in fluor_channels]
    else:
        ch_indices = [
            i for i, name in enumerate(channel_names)
            if not any(kw.lower() in name.lower() for kw in non_fluor_keywords)
        ]

    if not ch_indices:
        print("No fluorescence channels found to correct. Exiting.")
        return

    fluor_names = [channel_names[i] for i in ch_indices]
    print(f"  Fluorescence channels to correct: {fluor_names} (indices {ch_indices})")
    # --- Step 1: Estimate flatfield profiles ---
    t_est = time.time()
    print(f"\nStep 1: Estimating flatfield profile (max-normalized, sigma={sigma or 75})...")
    raw_results = estimate_flatfield_max(source_path, ch_indices, num_samples, num_workers, sigma, camera_offset)
    print(f"  Estimation complete in {time.time() - t_est:.1f}s")

    # Split out corrections (discard sample stacks)
    corrections: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for ch_idx, (flatfield, darkfield, _stack) in raw_results.items():
        corrections[ch_idx] = (flatfield, darkfield)
    del raw_results
    print(f"  Camera offset: {camera_offset:.1f} counts")

    # Save profiles for inspection / reuse
    if save_profiles:
        if output_dir_override:
            profile_dir = Path(output_dir_override) / "flatfield_profiles"
        else:
            profile_dir = dataset.configs / "flatfield_profiles"
        # Fall back to per-experiment writable path if configs/ is read-only.
        # This lets sandboxed runs (e.g., OPS_CONFIGS_DIR pointing at a
        # shared dir owned by another user) compute and save flatfields
        # without permission errors. Production runs with write access to
        # the shared configs dir keep their behavior.
        try:
            profile_dir.mkdir(parents=True, exist_ok=True)
            # Probe writability with a temp file (mkdir -p succeeds even
            # when dir already exists and isn't writable).
            _probe = profile_dir / ".write_probe"
            _probe.touch()
            _probe.unlink()
        except (PermissionError, OSError):
            fallback = dataset.experiment_path / "configs" / "flatfield_profiles"
            print(
                f"  WARN: {profile_dir} not writable; falling back to "
                f"{fallback}"
            )
            profile_dir = fallback
            profile_dir.mkdir(parents=True, exist_ok=True)
        for ch_idx, (flatfield, darkfield) in corrections.items():
            ch_name = channel_names[ch_idx]
            np.save(profile_dir / f"flatfield_{ch_name}.npy", flatfield)
            np.save(profile_dir / f"darkfield_{ch_name}.npy", darkfield)
        print(f"  Saved flatfield profiles to {profile_dir}")

    # --- Step 2: Apply correction to all positions ---
    t_apply = time.time()
    position_list = _maybe_sample_positions(position_list, debug_n_positions)
    corrected_path = _resolve_output_path_for_debug(
        corrected_path, debug_n_positions, debug_output_suffix
    )

    print(f"\nStep 2: Applying correction to {len(position_list)} positions...")
    print(f"  Output store: {corrected_path}")
    # Rebuild from scratch: every position is recorrected below, and reusing an
    # existing store would keep its stale shape/channel metadata.
    async_delete_path(corrected_path)
    print(f"  Pre-creating {len(position_list)} positions...")
    create_hcs_store_fast(
        store_path=corrected_path,
        positions=position_list,
        shape=src_shape,
        chunks=src_chunks,
        dtype=dtype,
        scale=output_transform.scale,
        channel_names=channel_names,
    )

    print(f"  Correcting with {num_workers} workers...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_flatfield_correct_worker, pos, str(source_path), str(corrected_path), corrections, camera_offset): pos
            for pos in position_list
        }
        for future in tqdm(as_completed(futures), total=len(position_list), desc="  Correcting positions"):
            future.result()
    print(f"  Correction complete in {time.time() - t_apply:.1f}s")

    # --- Step 3: Save QC comparison PNGs ---
    t_viz = time.time()
    print(f"\nStep 3: Saving QC comparison PNGs...")

    # Debug mode: use the debug positions; full mode: pick 5 seeded random tiles
    if debug_n_positions:
        qc_positions = position_list
    else:
        qc_rng = np.random.default_rng(42)
        qc_positions = list(qc_rng.choice(position_list, size=min(5, len(position_list)), replace=False))

    # Re-measure flatfield profile from QC tiles to validate
    print(f"  Re-measuring flatfield profile from {len(qc_positions)} QC tiles...")
    corrected_profiles: dict[int, np.ndarray] = {}
    for ch_idx in ch_indices:
        corrected_imgs = [None] * len(qc_positions)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {
                executor.submit(_read_single_position, str(corrected_path), pos, ch_idx, "max_projection"): i
                for i, pos in enumerate(qc_positions)
            }
            for future in as_completed(future_to_idx):
                corrected_imgs[future_to_idx[future]] = future.result()
        corrected_profiles[ch_idx] = np.mean(np.stack(corrected_imgs, axis=0), axis=0)

    # Debug mode: save to cwd; full mode: save to experiment assembly folder
    if output_dir_override:
        qc_output_dir = Path(output_dir_override)
    elif debug_n_positions:
        qc_output_dir = Path.cwd() / "flatfield_debug_previews"
    else:
        qc_output_dir = dataset.experiment_path / "3-assembly" / "illumination_correction"

    _save_debug_comparison(
        source_path=source_path,
        corrected_path=corrected_path,
        positions=qc_positions,
        channel_names=channel_names,
        ch_indices=ch_indices,
        corrections=corrections,
        corrected_profiles=corrected_profiles,
        output_dir=qc_output_dir,
        experiment=experiment,
        camera_offset=camera_offset,
        sigma=sigma,
        num_samples=num_samples,
        num_positions_total=len(position_list),
    )
    print(f"  QC saved in {time.time() - t_viz:.1f}s")
    print(f"\nFlatfield correction complete. Total time: {time.time() - t_total:.1f}s")


def _save_debug_comparison(
    source_path,
    corrected_path,
    positions: list[str],
    channel_names: list[str],
    ch_indices: list[int],
    corrections: dict,
    corrected_profiles: dict[int, np.ndarray],
    output_dir: Path,
    experiment: str = "",
    camera_offset: float = 100.0,
    sigma: int | None = None,
    num_samples: int = 500,
    num_positions_total: int = 0,
):
    """Save before/after PNG comparisons and QC metrics CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import zarr as zarr_mod

    output_dir.mkdir(parents=True, exist_ok=True)
    qc_rows = []

    for pos in tqdm(positions, desc="  Saving previews"):
        src_arr = zarr_mod.open(str(Path(str(source_path)) / pos / "0"), mode="r")
        dst_arr = zarr_mod.open(str(Path(str(corrected_path)) / pos / "0"), mode="r")
        src_data = np.asarray(src_arr)   # (T, C, Z, H, W)
        dst_data = np.asarray(dst_arr)

        for ch_idx in ch_indices:
            ch_name = channel_names[ch_idx]
            flatfield, _darkfield = corrections[ch_idx]

            # Max-project over Z for visualization (t=0)
            before = src_data[0, ch_idx].max(axis=0).astype(np.float32)
            after = dst_data[0, ch_idx].max(axis=0).astype(np.float32)

            def _saturate(img, lo_pct=10, hi_pct=90):
                lo, hi = np.percentile(img, [lo_pct, hi_pct])
                return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)

            before_sat = _saturate(before)
            after_sat = _saturate(after)

            cv_before = _spatial_cv(before)
            cv_after = _spatial_cv(after)

            # SNR: signal = mean of top quartile, noise = std of bottom quartile
            def _snr(img):
                q25, q75 = np.percentile(img, [25, 75])
                bg = img[img <= q25]
                noise = bg.std() if len(bg) > 0 and bg.std() > 0 else 1e-6
                signal = img[img >= q75].mean() - bg.mean()
                return signal / noise

            snr_before = _snr(before)
            snr_after = _snr(after)

            # Collect QC metrics (zero extra I/O — reuses loaded data)
            qc_rows.append({
                "experiment": experiment,
                "position": pos,
                "channel": ch_name,
                "cv_before": round(cv_before, 6),
                "cv_after": round(cv_after, 6),
                "cv_improvement": round(cv_before - cv_after, 6),
                "cv_ratio": round(cv_after / cv_before, 4) if cv_before > 0 else None,
                "snr_before": round(snr_before, 2),
                "snr_after": round(snr_after, 2),
                "snr_change": round(snr_after - snr_before, 2),
                "mean_before": round(float(before.mean()), 2),
                "mean_after": round(float(after.mean()), 2),
                "flatfield_min": round(float(flatfield.min()), 4),
                "flatfield_max": round(float(flatfield.max()), 4),
                "flatfield_range": round(float(flatfield.max() - flatfield.min()), 4),
                "camera_offset": camera_offset,
                "sigma": sigma or 75,
                "num_samples": num_samples,
                "num_positions_corrected": num_positions_total,
            })

            # Get the re-measured corrected profile for this channel
            corr_profile = corrected_profiles[ch_idx]

            pos_label = pos.replace("/", "_")
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))

            # Row 1: before, after, radial profile
            axes[0, 0].imshow(before_sat, cmap="inferno", vmin=0, vmax=1)
            axes[0, 0].set_title(f"Before (saturated) — CV={cv_before:.4f}")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(after_sat, cmap="inferno", vmin=0, vmax=1)
            axes[0, 1].set_title(f"After (saturated) — CV={cv_after:.4f}")
            axes[0, 1].axis("off")

            # Radial profiles
            h, w = before.shape
            cy, cx = h // 2, w // 2
            y, x = np.ogrid[0:h, 0:w]
            r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
            r_max = min(cy, cx)
            radii = np.arange(0, r_max)
            prof_before = np.array([before[r == rr].mean() for rr in radii])
            prof_after = np.array([after[r == rr].mean() for rr in radii])
            prof_before = prof_before / prof_before[0]
            prof_after = prof_after / prof_after[0]

            axes[0, 2].plot(radii, prof_before, label="Before", alpha=0.8)
            axes[0, 2].plot(radii, prof_after, label="After", alpha=0.8)
            axes[0, 2].axhline(1.0, color="gray", ls="--", alpha=0.5)
            axes[0, 2].set_xlabel("Radius from center (px)")
            axes[0, 2].set_ylabel("Normalized mean intensity")
            axes[0, 2].set_title("Radial intensity profile (this tile)")
            axes[0, 2].legend()
            axes[0, 2].set_ylim(0.5, 1.5)

            # Row 2: flatfield profile before, corrected profile after
            # Each gets its own colorbar range to show full contrast
            im1 = axes[1, 0].imshow(flatfield, cmap="inferno")
            axes[1, 0].set_title(f"Estimated illumination profile")
            axes[1, 0].axis("off")
            plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)

            im2 = axes[1, 1].imshow(corr_profile, cmap="inferno")
            axes[1, 1].set_title(f"Re-measured profile (after correction)")
            axes[1, 1].axis("off")
            plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)

            # Radial profiles of the flatfield heatmaps (normalized to center value)
            prof_ff = np.array([flatfield[r == rr].mean() for rr in radii])
            prof_cp = np.array([corr_profile[r == rr].mean() for rr in radii])
            prof_ff = prof_ff / prof_ff[0]
            prof_cp = prof_cp / prof_cp[0]

            axes[1, 2].plot(radii, prof_ff, label="Before correction", alpha=0.8)
            axes[1, 2].plot(radii, prof_cp, label="After correction", alpha=0.8)
            axes[1, 2].axhline(1.0, color="gray", ls="--", alpha=0.5)
            axes[1, 2].set_xlabel("Radius from center (px)")
            axes[1, 2].set_ylabel("Normalized profile intensity")
            axes[1, 2].set_title("Illumination profile radial comparison")
            axes[1, 2].legend()

            fig.suptitle(f"{pos}  |  {ch_name}", fontsize=13)
            fig.tight_layout()

            exp_prefix = f"{experiment}_" if experiment else ""
            out_path = output_dir / f"{exp_prefix}{pos_label}_ill_correction.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

    # Save QC metrics CSV
    df = pd.DataFrame(qc_rows)
    csv_path = output_dir / f"{experiment}_flatfield_qc.csv" if experiment else output_dir / "flatfield_qc.csv"
    df.to_csv(csv_path, index=False)
    print(f"  QC metrics saved to {csv_path}")
    print(f"  Debug previews saved to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate and apply flatfield correction to 20x fluorescence channels"
    )
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of positions to sample for profile estimation (default: 500)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers for correction pass",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Specific fluorescence channel names to correct (default: auto-detect)",
    )
    parser.add_argument(
        "--sigma",
        type=int,
        default=None,
        help="Gaussian sigma in pixels (default: 75). Lower captures more optical structure.",
    )
    parser.add_argument(
        "--camera-offset",
        type=float,
        default=100.0,
        help="Camera dark current offset in counts (default: 100)",
    )
    parser.add_argument(
        "--debug-n-positions",
        type=int,
        default=None,
        help="Limit to N positions for testing",
    )

    args = parser.parse_args()

    from cyclops_utils.data.filesystem import resolve_experiment_name
    args.experiment = resolve_experiment_name(args.experiment, autoselect=True)

    correct_flatfield(
        experiment=args.experiment,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
        fluor_channels=args.channels,
        sigma=args.sigma,
        camera_offset=args.camera_offset,
        debug_n_positions=args.debug_n_positions,
    )
