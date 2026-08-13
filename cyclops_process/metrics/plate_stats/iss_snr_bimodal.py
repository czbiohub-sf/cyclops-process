"""
SNR Heatmap Generation for ISS Data (Bimodal Distribution Approach)

This module generates spatial heatmaps of Signal-to-Noise Ratio (SNR) across wells
using a bimodal distribution analysis (Gaussian Mixture Model) of image intensities,
instead of relying on pre-computed spots.

It creates three levels of detail:
1. Overall mean SNR (averaged across all channels and rounds)
2. Per-channel mean SNR (averaged across all rounds)
3. Per-channel, per-round SNR (full detail)

Usage:
    python -m cyclops_process.metrics.plate_stats.iss_snr_bimodal --experiment ops0033_2025042
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import sys
import os

sys.path.insert(0, os.getcwd())


from cyclops_utils.data.experiment import OpsDataset
from cyclops_process.metrics.plate_stats.metrics_iss_utils import (
    calculate_snr,
    calculate_sbr,
    calculate_lld,
    calculate_z_prime,
)
from typing import Dict, List, Optional, Tuple
import click
import pandas as pd
from sklearn.mixture import GaussianMixture
from joblib import Parallel, delayed
from cyclops_utils.hpc.resource_manager import get_optimal_workers
from scipy.stats import linregress


def calculate_tile_stats_bimodal(
    tile_data: np.ndarray,
    channels: List[str] = ["G", "T", "A", "C"]
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Calculate SNR and comprehensive stats using bimodal distribution analysis (GMM).

    Args:
        tile_data: Image data with shape (n_cycles, n_channels, H, W)
        channels: List of channel names

    Returns:
        Tuple of:
            - SNR values (n_cycles, n_channels)
            - DataFrame with comprehensive stats
    """
    n_cycles, n_channels, H, W = tile_data.shape
    snr_values = np.zeros((n_cycles, n_channels))
    stats_list = []

    for cycle_idx in range(n_cycles):
        for chan_idx in range(n_channels):
            # Use original data
            image_data = tile_data[cycle_idx, chan_idx, :, :].flatten()

            # Simple check for empty/zero image
            if len(image_data) == 0 or np.all(image_data == 0):
                 snr_values[cycle_idx, chan_idx] = 0
                 continue

            # Fit GMM to pixel intensities
            # Reshape for sklearn: (n_samples, n_features) -> (H*W, 1)
            X = image_data.reshape(-1, 1)

            # Subsample for speed if image is large (>10k pixels)
            if len(X) > 10000:
                # Use random sampling
                rng = np.random.RandomState(42 + cycle_idx + chan_idx)
                indices = rng.choice(len(X), 10000, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X

            try:
                # Fit 2-component GMM (Signal vs Background)
                gmm = GaussianMixture(n_components=2, random_state=42)
                gmm.fit(X_sample)

                means = gmm.means_.flatten()
                covariances = gmm.covariances_.flatten()
                weights = gmm.weights_.flatten()

                # Identify signal (higher mean) and background (lower mean)
                idx_bg = np.argmin(means)
                idx_sig = np.argmax(means)

                bg_mean = means[idx_bg]
                bg_std = np.sqrt(covariances[idx_bg])
                sig_mean = means[idx_sig]
                sig_std = np.sqrt(covariances[idx_sig])

                # Calculate SNR
                if bg_std > 0:
                    snr = (sig_mean - bg_mean) / bg_std
                    snr = max(0, snr)
                else:
                    snr = 0
                snr_values[cycle_idx, chan_idx] = snr

                # Add stats
                stats_list.append({
                    "Cycle": cycle_idx + 1,
                    "Channel": channels[chan_idx],
                    "Type": "Signal",
                    "mean": sig_mean,
                    "std": sig_std,
                    "count": int(weights[idx_sig] * len(X)),
                    "median": sig_mean, # Approximation from Gaussian mean
                    "median_top_10pct": sig_mean + 1.28 * sig_std # Approximation: 90th percentile of Gaussian
                })
                stats_list.append({
                    "Cycle": cycle_idx + 1,
                    "Channel": channels[chan_idx],
                    "Type": "Background",
                    "mean": bg_mean,
                    "std": bg_std,
                    "count": int(weights[idx_bg] * len(X)),
                    "median": bg_mean, # Approximation
                    "median_top_10pct": np.nan
                })

            except Exception:
                # Fallback if GMM fails (e.g. too uniform data)
                snr_values[cycle_idx, chan_idx] = 0

    return snr_values, pd.DataFrame(stats_list)


def _load_and_compute_tile_bimodal(
    store_path_str: str,
    well_hcs: str,
    pos_path: str,
    channel_start: int,
    n_iss_channels: int = 4,
    crop_size: int = None,
) -> Tuple[int, int, np.ndarray, pd.DataFrame]:
    """
    Load a single tile from zarr and compute bimodal GMM stats.
    Each call opens its own zarr connection for process-safety with joblib.

    Args:
        store_path_str: Path to zarr store (string for pickle compatibility)
        well_hcs: Well in HCS format (e.g., "A/1")
        pos_path: Position path within well (e.g., "001002" = row 1, col 2)
        channel_start: Channel offset (0 or 1 to skip DAPI)
        n_iss_channels: Number of ISS channels (default 4: G,T,A,C)
        crop_size: If set, load only center crop of this size (reduces I/O)

    Returns:
        (row, col, snr_values, tile_stats)
    """
    from iohub.ngff import open_ome_zarr
    from pathlib import Path

    full_path = f"{well_hcs}/{pos_path}"
    # Position names use XXXYYY convention (first 3 digits = row, last 3 = col)
    pos_name = pos_path.split("/")[-1]
    row = int(pos_name[:3])
    col = int(pos_name[3:])
    channels = ["G", "T", "A", "C"][:n_iss_channels]

    # Open position directly with layout="fov" to skip plate metadata parsing.
    with open_ome_zarr(Path(store_path_str) / full_path, layout="fov", mode="r") as ds:
        if crop_size is not None:
            shape = ds.data.shape  # (T, C, Z, Y, X)
            H, W = shape[-2], shape[-1]
            h0 = max(0, (H - crop_size) // 2)
            w0 = max(0, (W - crop_size) // 2)
            tile_data = np.asarray(
                ds.data[
                    :, channel_start:channel_start + n_iss_channels, 0,
                    h0:h0 + crop_size, w0:w0 + crop_size
                ]
            )
        else:
            tile_data = np.asarray(
                ds.data[
                    :, channel_start:channel_start + n_iss_channels, 0, :, :
                ]
            )

    snr_values, tile_stats = calculate_tile_stats_bimodal(tile_data, channels)
    return row, col, snr_values, tile_stats


def _safe_load_and_compute_tile_bimodal(store_path_str, well_hcs, pos_path,
                                        channel_start, n_iss_channels, crop_size):
    """Tile worker that never raises — a single bad tile yields a skipped (nan)
    result instead of aborting the whole flat pool."""
    try:
        return _load_and_compute_tile_bimodal(
            store_path_str, well_hcs, pos_path, channel_start, n_iss_channels, crop_size
        )
    except Exception as e:
        pos_name = pos_path.split("/")[-1]
        row, col = int(pos_name[:3]), int(pos_name[3:])
        print(f"  ✗ tile {well_hcs}/{pos_path} failed ({type(e).__name__}: {e}); skipping")
        # Shape (1, 1) won't match (n_cycles, n_channels), so assemble leaves it nan.
        return row, col, np.full((1, 1), np.nan), pd.DataFrame()


def _well_snr_setup(dataset: "OpsDataset", well: str, crop_size: int):
    """Cheap per-well metadata read (positions, grid, channels). Returns a setup
    dict, or None when the well is missing/empty. No tile data is loaded."""
    from iohub.ngff import open_ome_zarr
    import re

    well_hcs = well
    if "/" not in well:
        m = re.match(r"^([A-Za-z]+)(\d+)$", well)
        if m:
            well_hcs = f"{m.group(1)}/{m.group(2)}"

    store_path = dataset.store_paths["iss"]
    ds = open_ome_zarr(store_path, mode="r")
    try:
        position_list = list(ds[well_hcs].positions())
    except KeyError:
        print(f"  Well {well_hcs} not found in zarr store at {store_path}")
        ds.close()
        return None
    if not position_list:
        print(f"  No positions found for well {well_hcs}")
        ds.close()
        return None

    first_pos_path, _ = position_list[0]
    data_shape = ds[f"{well_hcs}/{first_pos_path}"].data.shape  # (T, C, Z, Y, X)
    num_channels = data_shape[1]
    channel_start = 1 if num_channels == 5 else 0  # skip DAPI on 5-channel stores
    n_cycles = data_shape[0]
    n_channels = 4

    grid_coords = []
    for pos_path, _ in position_list:
        pos_name = pos_path.split("/")[-1]
        grid_coords.append((int(pos_name[:3]), int(pos_name[3:])))
    grid_rows = max(r for r, c in grid_coords) + 1
    grid_cols = max(c for r, c in grid_coords) + 1

    ds.close()
    return {
        "well": well, "well_hcs": well_hcs, "store_path": str(store_path),
        "position_paths": [p for p, _ in position_list],
        "channel_start": channel_start, "n_channels": n_channels, "n_cycles": n_cycles,
        "grid_rows": grid_rows, "grid_cols": grid_cols,
        "n_tiles": grid_rows * grid_cols, "crop_size": crop_size,
    }


def _assemble_well_snr(setup: dict, tile_results: list):
    """Fold per-tile (row, col, snr_values, tile_stats) into a well's snr array
    and comprehensive-stats DataFrame."""
    grid_cols, n_cycles, n_channels = setup["grid_cols"], setup["n_cycles"], setup["n_channels"]
    snr_results = np.full((setup["n_tiles"], n_cycles, n_channels), np.nan)
    all_tile_stats = []
    for row, col, snr_values, tile_stats in tile_results:
        tile_idx = row * grid_cols + col
        if snr_values.ndim == 2 and snr_values.shape == (n_cycles, n_channels):
            snr_results[tile_idx] = snr_values
        if not tile_stats.empty:
            tile_stats = tile_stats.copy()
            tile_stats["tile_idx"] = tile_idx
            tile_stats["row"] = row
            tile_stats["col"] = col
            tile_stats["well"] = setup["well"]
            all_tile_stats.append(tile_stats)
    comprehensive_stats = pd.concat(all_tile_stats, ignore_index=True) if all_tile_stats else pd.DataFrame()
    return snr_results, comprehensive_stats


def process_wells_snr_bimodal_batched(dataset: "OpsDataset", wells: list, crop_size: int = 512) -> dict:
    """Process ALL wells' tiles in a single flat joblib pool so workers stay
    saturated across wells (no sequential per-well barrier). Peak memory is
    unchanged — still one tile per worker. Returns {well: (snr_results,
    comprehensive_stats, grid_rows, grid_cols, load_time, compute_time)}."""
    import time
    import gc

    setups = {}
    for well in wells:
        s = _well_snr_setup(dataset, well, crop_size)
        if s is not None:
            setups[well] = s
    if not setups:
        return {}

    # Flat (well, pos_path) task list across all wells.
    flat_tasks = [
        (well, pos_path)
        for well, s in setups.items()
        for pos_path in s["position_paths"]
    ]
    n_jobs = get_optimal_workers(use_gpu=False, verbose=False)
    print(f"  Processing {len(flat_tasks)} tiles across {len(setups)} wells "
          f"with {n_jobs} workers (single flat pool)...")

    compute_start = time.time()
    flat_results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_safe_load_and_compute_tile_bimodal)(
            setups[well]["store_path"], setups[well]["well_hcs"], pos_path,
            setups[well]["channel_start"], setups[well]["n_channels"], crop_size,
        )
        for well, pos_path in flat_tasks
    )
    compute_time = time.time() - compute_start

    # Regroup tile results by well, then assemble.
    by_well: dict = {well: [] for well in setups}
    for (well, _pos), res in zip(flat_tasks, flat_results):
        by_well[well].append(res)

    total_tiles = len(flat_tasks)
    out = {}
    for well, s in setups.items():
        snr_results, comprehensive_stats = _assemble_well_snr(s, by_well[well])
        # Attribute wall time proportionally for the per-well timing report.
        well_ct = compute_time * (len(by_well[well]) / total_tiles) if total_tiles else 0.0
        out[well] = (snr_results, comprehensive_stats, s["grid_rows"], s["grid_cols"], 0.0, well_ct)

    del flat_results
    gc.collect()
    print(f"✓ Completed {len(setups)} wells in {compute_time:.1f}s (flat parallel pool)")
    return out


def process_well_snr_tiles_bimodal(
    dataset: "OpsDataset",
    well: str,
    crop_size: int = 512,
) -> Tuple[np.ndarray, pd.DataFrame, int, int, float, float]:
    """Single-well wrapper around the batched flat-pool path (kept for API
    stability). Returns (snr_results, comprehensive_stats, grid_rows, grid_cols,
    load_time, compute_time)."""
    result = process_wells_snr_bimodal_batched(dataset, [well], crop_size)
    if well not in result:
        return np.full((0, 1, 4), np.nan), pd.DataFrame(), 0, 0, 0.0, 0.0
    return result[well]


def create_combined_overall_heatmaps(
    all_snr_results: Dict[str, np.ndarray],
    grid_rows: int,
    grid_cols: int,
    experiment: str,
    save_path: Path,
    vmin: float = 0,
    vmax: float = None,
    cmap: str = "viridis",
) -> None:
    """
    Create combined overall SNR heatmaps for all wells in one canvas.
    Layout: 1 row × n_wells columns
    """
    wells = list(all_snr_results.keys())
    num_wells = len(wells)

    # Create figure with wells as columns
    fig, axes = plt.subplots(1, num_wells, figsize=(8 * num_wells, 8), squeeze=False)
    axes = axes.flatten()

    for well_idx, well in enumerate(wells):
        snr_results = all_snr_results[well]
        ax = axes[well_idx]

        # Calculate overall mean SNR (excluding tiles with no spots using nanmean)
        mean_snr = np.nanmean(
            snr_results, axis=(1, 2)
        )  # Average over cycles and channels

        # Reshape to grid
        snr_grid = np.zeros((grid_rows, grid_cols))
        snr_grid.fill(np.nan)

        n_tiles = len(mean_snr)
        for idx in range(min(n_tiles, grid_rows * grid_cols)):
            i = idx // grid_cols
            j = idx % grid_cols
            snr_grid[i, j] = mean_snr[idx]

        # Plot
        im = ax.imshow(
            snr_grid, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(well, fontsize=32, pad=10)

        # Add colorbar for each subplot
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=20)

    fig.suptitle(
        f"Mean SNR (All Channels & Rounds) - Bimodal\n{experiment}",
        fontsize=40,
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def create_combined_per_channel_heatmaps(
    all_snr_results: Dict[str, np.ndarray],
    grid_rows: int,
    grid_cols: int,
    experiment: str,
    save_path: Path,
    vmin: float = 0,
    vmax: float = None,
    cmap: str = "viridis",
) -> None:
    """
    Create combined per-channel heatmaps for all wells in one canvas.
    Layout: 4 rows (channels) × n_wells columns
    """
    channels = ["G", "T", "A", "C"]
    wells = list(all_snr_results.keys())
    num_wells = len(wells)

    # Create figure: 4 rows (channels) × n_wells columns
    fig, axes = plt.subplots(4, num_wells, figsize=(7 * num_wells, 24), squeeze=False)

    for chan_idx, chan_name in enumerate(channels):
        for well_idx, well in enumerate(wells):
            snr_results = all_snr_results[well]
            ax = axes[chan_idx, well_idx]

            # Average across rounds for this channel (excluding tiles with no spots)
            per_channel_snr = np.nanmean(
                snr_results, axis=1
            )  # Shape: (n_tiles, n_channels)
            snr_data = per_channel_snr[:, chan_idx]

            # Reshape to grid
            snr_grid = np.zeros((grid_rows, grid_cols))
            snr_grid.fill(np.nan)

            n_tiles = len(snr_data)
            for idx in range(min(n_tiles, grid_rows * grid_cols)):
                i = idx // grid_cols
                j = idx % grid_cols
                snr_grid[i, j] = snr_data[idx]

            # Plot
            im = ax.imshow(
                snr_grid, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"
            )
            ax.set_xticks([])
            ax.set_yticks([])

            # Label first column with channel names
            if well_idx == 0:
                ax.set_ylabel(
                    chan_name,
                    fontsize=28,
                    rotation=0,
                    labelpad=50,
                    ha="right",
                    va="center",
                )

            # Label first row with well names
            if chan_idx == 0:
                ax.set_title(well, fontsize=28, pad=10)

            # Add colorbar to last column
            if well_idx == num_wells - 1:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=20)

    fig.suptitle(
        f"Mean SNR per Channel (Averaged Across Rounds) - Bimodal\n{experiment}",
        fontsize=44,
        y=0.9995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.998])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def create_per_channel_per_round_heatmaps(
    snr_results: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    well: str,
    experiment: str,
    save_path: Path,
    vmin: float = 0,
    vmax: float = None,
    cmap: str = "viridis",
    normalize_to_cycle0: bool = False,
) -> None:
    """
    Create grid of heatmaps (4 channels × n_cycles).

    Args:
        snr_results: SNR values (n_tiles, n_cycles, n_channels)
        grid_rows: Number of rows in grid
        grid_cols: Number of columns in grid
        well: Well identifier
        experiment: Experiment name
        save_path: Output path
        vmin: Colormap minimum
        vmax: Colormap maximum
        cmap: Colormap name
        normalize_to_cycle0: If True, normalize each channel by its cycle 0 values
    """
    channels = ["G", "T", "A", "C"]
    n_tiles, n_cycles, n_channels = snr_results.shape

    # Normalize if requested
    if normalize_to_cycle0:
        snr_results_plot = snr_results.copy()
        # For each channel, divide by the MEAN of cycle 0 values across all tiles
        for chan_idx in range(n_channels):
            cycle0_values = snr_results_plot[:, 0, chan_idx]  # (n_tiles,)
            # Calculate mean across all tiles for this channel's cycle 0
            cycle0_mean = np.nanmean(cycle0_values)
            # Avoid division by zero
            if cycle0_mean > 0:
                # Normalize all cycles and all tiles for this channel by the cycle 0 mean
                snr_results_plot[:, :, chan_idx] = snr_results_plot[:, :, chan_idx] / cycle0_mean
            else:
                # If mean is 0 or nan, set to nan
                snr_results_plot[:, :, chan_idx] = np.nan
    else:
        snr_results_plot = snr_results

    # Create grid: n_cycles rows × 4 columns
    fig, axes = plt.subplots(n_cycles, 4, figsize=(18, 4.5 * n_cycles), squeeze=False)

    for cycle_idx in range(n_cycles):
        for chan_idx, chan_name in enumerate(channels):
            ax = axes[cycle_idx, chan_idx]
            snr_data = snr_results_plot[:, cycle_idx, chan_idx]

            # Reshape to grid
            snr_grid = np.zeros((grid_rows, grid_cols))
            snr_grid.fill(np.nan)

            for idx in range(min(n_tiles, grid_rows * grid_cols)):
                i = idx // grid_cols
                j = idx % grid_cols
                snr_grid[i, j] = snr_data[idx]

            # Plot
            im = ax.imshow(
                snr_grid, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"
            )
            ax.set_xticks([])
            ax.set_yticks([])

            # Label first column with cycle number
            if chan_idx == 0:
                ax.set_ylabel(
                    f"Cycle {cycle_idx + 1}",
                    fontsize=24,
                    rotation=0,
                    labelpad=50,
                    ha="right",
                    va="center",
                )

            # Label first row with channel names
            if cycle_idx == 0:
                ax.set_title(f"{chan_name}", fontsize=28, pad=10)

            # Add minimal colorbar to last column
            if chan_idx == 3:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=16)

    # Update title based on normalization
    if normalize_to_cycle0:
        title = f"SNR per Channel per Round (Normalized to Cycle 1)\n{experiment} - {well}"
    else:
        title = f"SNR per Channel per Round (Bimodal)\n{experiment} - {well}"

    fig.suptitle(title, fontsize=44, y=0.9995)
    plt.tight_layout(rect=[0, 0, 1, 0.998])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def save_comprehensive_stats_to_csv(
    pooled_stats: pd.DataFrame,
    dataset: OpsDataset,
) -> None:
    """
    Save comprehensive imaging quality metrics to CSV.
    """
    if pooled_stats.empty:
        print("\n⚠️  Warning: No comprehensive stats to save")
        return

    # === 1. Save per-tile comprehensive stats ===
    stats_with_metrics = pooled_stats.copy()

    # Separate signal and background rows
    signal_df = stats_with_metrics[stats_with_metrics["Type"] == "Signal"].copy()
    background_df = stats_with_metrics[
        stats_with_metrics["Type"] == "Background"
    ].copy()

    # Merge to get both signal and background stats on same row
    signal_df = signal_df.rename(
        columns={
            "mean": "signal_mean",
            "median": "signal_median",
            "std": "signal_std",
            "count": "signal_count",
            "median_top_10pct": "signal_top10pct",
        }
    ).drop(columns=["Type"])

    background_df = background_df.rename(
        columns={
            "mean": "background_mean",
            "median": "background_median",
            "std": "background_std",
            "count": "background_count",
        }
    ).drop(columns=["Type", "median_top_10pct"], errors="ignore")

    merged_df = pd.merge(
        signal_df,
        background_df,
        on=["well", "tile_idx", "row", "col", "Cycle", "Channel"],
        how="outer",
    )

    # Calculate all metrics using discrete functions
    merged_df["snr"] = merged_df.apply(
        lambda row: (
            calculate_snr(
                row["signal_mean"], row["background_mean"], row["background_std"]
            )
            if pd.notna(row["signal_mean"])
            else np.nan
        ),
        axis=1,
    )
    merged_df["sbr"] = merged_df.apply(
        lambda row: (
            calculate_sbr(row["signal_mean"], row["background_mean"])
            if pd.notna(row["signal_mean"])
            else np.nan
        ),
        axis=1,
    )
    merged_df["lld"] = merged_df.apply(
        lambda row: (
            calculate_lld(
                row["background_std"], row["signal_mean"], row["background_mean"]
            )
            if pd.notna(row["signal_mean"])
            else np.nan
        ),
        axis=1,
    )
    merged_df["z_prime"] = merged_df.apply(
        lambda row: (
            calculate_z_prime(
                row["signal_mean"],
                row["signal_std"],
                row["background_mean"],
                row["background_std"],
            )
            if pd.notna(row["signal_mean"])
            else np.nan
        ),
        axis=1,
    )

    # Save per-tile data
    # Use a new suffix for bimodal results if desired, or overwrite existing
    csv_path_per_tile = dataset.metrics_paths["snr_per_tile_data"].with_name("snr_per_tile_data_bimodal.csv")
    merged_df.to_csv(csv_path_per_tile, index=False, float_format="%.3f")
    print(f"\n✓ Saved comprehensive per-tile stats to: {csv_path_per_tile}")

    # === 2. Save mean metrics per round per channel (averaged across tiles) ===
    summary_rows = []
    channels = ["G", "T", "A", "C"]

    for well in merged_df["well"].unique():
        well_data = merged_df[merged_df["well"] == well]

        for cycle in well_data["Cycle"].unique():
            for channel in channels:
                subset = well_data[
                    (well_data["Cycle"] == cycle) & (well_data["Channel"] == channel)
                ]

                if len(subset) > 0:
                    summary_rows.append(
                        {
                            "well": well,
                            "cycle": int(cycle),
                            "channel": channel,
                            "median_signal": subset["signal_mean"].median(),
                            "std_signal": subset["signal_mean"].std(),
                            "median_signal_top10pct": subset["signal_top10pct"].median(),
                            "median_background": subset["background_mean"].median(),
                            "std_background": subset["background_std"].median(),
                            "median_snr": subset["snr"].median(),
                            "std_snr": subset["snr"].std(),
                            "median_sbr": subset["sbr"].median(),
                            "std_sbr": subset["sbr"].std(),
                            "median_lld": subset["lld"].median(),
                            "median_z_prime": subset["z_prime"].median(),
                            "n_tiles_used": len(subset), # All tiles used since we don't filter by spots
                        }
                    )

    summary_df = pd.DataFrame(summary_rows)

    # Save summary data
    csv_path_summary = dataset.metrics_paths["snr_mean_per_round_per_channel"].with_name("snr_mean_per_round_per_channel_bimodal.csv")
    summary_df.to_csv(csv_path_summary, index=False, float_format="%.3f")
    print(f"\n✓ Saved mean metrics per round per channel to: {csv_path_summary}")


def plot_metric_vs_cycle(
    summary_df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    save_path: Path,
    experiment: str,
    ylim: Optional[Tuple[float, float]] = None,
    channel_colors: Dict[str, str] = None,
    annotate: bool = True,
) -> None:
    """
    Generic function to plot any metric vs cycle with per-channel lines.
    """
    if channel_colors is None:
        channel_colors = {"G": "green", "T": "red", "A": "blue", "C": "orange"}

    wells = sorted(summary_df["well"].unique())
    num_wells = len(wells)

    fig, axes = plt.subplots(1, num_wells, figsize=(6 * num_wells, 5), sharey=True)
    if num_wells == 1:
        axes = [axes]

    for ax, well in zip(axes, wells):
        well_data = summary_df[summary_df["well"] == well]

        for channel in ["G", "T", "A", "C"]:
            channel_data = well_data[well_data["channel"] == channel].sort_values(
                "cycle"
            )
            if not channel_data.empty:
                ax.plot(
                    channel_data["cycle"],
                    channel_data[metric_col],
                    marker="o",
                    linestyle="-",
                    color=channel_colors[channel],
                    label=channel,
                    linewidth=2,
                    markersize=8,
                )

                # Annotate values
                if annotate:
                    for _, row in channel_data.iterrows():
                        val = row[metric_col]
                        if pd.notna(val) and np.isfinite(val):
                            ax.annotate(
                                f"{val:.1f}",
                                (row["cycle"], val),
                                textcoords="offset points",
                                xytext=(0, 5),
                                ha="center",
                                fontsize=9,
                                alpha=0.7,
                            )

        ax.set_xlabel("ISS Cycle", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(well, fontsize=16)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=12)

        if ylim:
            ax.set_ylim(ylim)

    fig.suptitle(f"{title}\n{experiment}", fontsize=18)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {save_path}")


def create_all_metric_plots(
    summary_df: pd.DataFrame,
    dataset: OpsDataset,
    experiment: str,
) -> None:
    """
    Create all metric vs cycle line plots from summary statistics.
    """
    print(f"\n{'='*60}")
    print("Generating metric vs cycle plots (Bimodal)")
    print(f"{'='*60}\n")

    # Helper to add suffix to filenames
    def bimodal_path(original_path):
        return original_path.with_name(original_path.stem + "_bimodal" + original_path.suffix)

    # 1. Signal intensity vs cycle
    plot_metric_vs_cycle(
        summary_df,
        metric_col="median_signal",
        ylabel="Median Signal Intensity (AU)",
        title="Signal Intensity vs Cycle (Bimodal)",
        save_path=bimodal_path(dataset.metrics_paths["iss_signal_vs_cycle"]),
        experiment=experiment,
    )

    # 2. Top 10% signal intensity vs cycle
    plot_metric_vs_cycle(
        summary_df,
        metric_col="median_signal_top10pct",
        ylabel="Median Top 10% Signal Intensity (AU)",
        title="Top 10% Signal Intensity vs Cycle (Bimodal)",
        save_path=bimodal_path(dataset.metrics_paths["iss_median_top10pct_vs_cycle"]),
        experiment=experiment,
    )

    # 3. Background noise (std) vs cycle
    plot_metric_vs_cycle(
        summary_df,
        metric_col="std_background",
        ylabel="Median Background Noise (Std Dev)",
        title="Background Noise vs Cycle (Bimodal)",
        save_path=bimodal_path(dataset.metrics_paths["iss_background_noise_vs_cycle"]),
        experiment=experiment,
    )

    # 4. Background median vs cycle
    plot_metric_vs_cycle(
        summary_df,
        metric_col="median_background",
        ylabel="Median Background Intensity (AU)",
        title="Background Intensity vs Cycle (Bimodal)",
        save_path=bimodal_path(dataset.metrics_paths["iss_background_mean_vs_cycle"]),
        experiment=experiment,
    )

    # 5. SNR vs cycle
    plot_metric_vs_cycle(
        summary_df,
        metric_col="median_snr",
        ylabel="Median Signal-to-Noise Ratio (SNR)",
        title="SNR vs Cycle (Bimodal)",
        save_path=bimodal_path(dataset.metrics_paths["iss_snr_vs_cycle"]),
        experiment=experiment,
        ylim=(0, None),
    )

    # 6. SBR vs cycle
    plot_metric_vs_cycle(
        summary_df,
        metric_col="median_sbr",
        ylabel="Median Signal-to-Background Ratio (SBR)",
        title="SBR vs Cycle (Bimodal)",
        save_path=bimodal_path(dataset.metrics_paths["iss_sbr_vs_cycle"]),
        experiment=experiment,
        ylim=(0, None),
    )

    # 7. LLD vs cycle (lower is better)
    plot_metric_vs_cycle(
        summary_df,
        metric_col="median_lld",
        ylabel="Median Lower Limit of Detection (LLD)",
        title="LLD vs Cycle (Lower is Better) (Bimodal)",
        save_path=bimodal_path(dataset.metrics_paths["iss_lld_vs_cycle"]),
        experiment=experiment,
    )

    # 8. Z'-factor vs cycle
    plot_metric_vs_cycle(
        summary_df,
        metric_col="median_z_prime",
        ylabel="Median Z' Factor",
        title="Z' Factor vs Cycle (Bimodal)",
        save_path=bimodal_path(dataset.metrics_paths["iss_zprime_vs_cycle"]),
        experiment=experiment,
    )

    print(f"\n✓ Completed all metric vs cycle plots\n")


def calculate_summary_stats_for_metrics(summary_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Calculate per-well summary statistics.
    """
    wells = summary_df["well"].unique()
    well_stats = {}

    for well in wells:
        well_data = summary_df[summary_df["well"] == well]

        well_stats[well] = {
            "snr_median_snr": well_data["median_snr"].median(),
            "snr_median_sbr": well_data["median_sbr"].median(),
            "snr_median_lld": well_data["median_lld"].median(),
            "snr_median_z_prime": well_data["median_z_prime"].median(),
            "snr_median_max_intensity": well_data["median_signal"].median(),
            "snr_median_top10pct_intensity": well_data["median_signal_top10pct"].median(),
            "snr_median_background_mean": well_data["median_background"].median(),
            "snr_median_background_std": well_data["std_background"].median(),
            "snr_snr_slope": 0.0,
            "snr_sbr_slope": 0.0,
        }

        try:
            # SNR slope
            snr_per_cycle = well_data.groupby("cycle")["median_snr"].median()
            if len(snr_per_cycle) > 1:
                slope_snr, _, _, _, _ = linregress(
                    snr_per_cycle.index, snr_per_cycle.values
                )
                well_stats[well]["snr_snr_slope"] = round(slope_snr, 2)

            # SBR slope
            sbr_per_cycle = well_data.groupby("cycle")["median_sbr"].median()
            if len(sbr_per_cycle) > 1:
                slope_sbr, _, _, _, _ = linregress(
                    sbr_per_cycle.index, sbr_per_cycle.values
                )
                well_stats[well]["snr_sbr_slope"] = round(slope_sbr, 2)
        except Exception:
            pass

    return well_stats


def load_snr_from_csv(dataset: OpsDataset) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, Dict[str, int], Dict[str, int], bool]:
    """
    Load pre-computed SNR results from CSV files.

    Returns:
        Tuple of (all_snr_results, per_tile_stats, well_grid_rows, well_grid_cols, success)
        - all_snr_results: Dict mapping well -> SNR array (n_tiles, n_cycles, n_channels)
        - per_tile_stats: DataFrame with per-tile comprehensive stats
        - well_grid_rows: Dict mapping well -> grid_rows
        - well_grid_cols: Dict mapping well -> grid_cols
        - success: True if CSVs were loaded successfully
    """
    csv_path_per_tile = dataset.metrics_paths["snr_per_tile_data"].with_name("snr_per_tile_data_bimodal.csv")

    if not csv_path_per_tile.exists():
        return {}, pd.DataFrame(), {}, {}, False

    print(f"\n{'='*60}")
    print("Found existing CSV data - Loading pre-computed results")
    print(f"{'='*60}\n")
    print(f"Loading: {csv_path_per_tile}")

    try:
        per_tile_stats = pd.read_csv(csv_path_per_tile)

        # Reconstruct SNR arrays from per-tile stats
        all_snr_results = {}
        well_grid_rows = {}
        well_grid_cols = {}

        # Get unique wells
        wells = sorted(per_tile_stats["well"].unique())

        for well in wells:
            well_data = per_tile_stats[per_tile_stats["well"] == well]

            # Get dimensions from actual row/col values
            # Handle both 'row'/'col' and 'row_x'/'col_x' (due to pandas merge suffixes)
            row_col = None
            if "row" in well_data.columns and "col" in well_data.columns:
                row_col = ("row", "col")
            elif "row_x" in well_data.columns and "col_x" in well_data.columns:
                row_col = ("row_x", "col_x")

            if row_col:
                grid_rows = int(well_data[row_col[0]].max() + 1)
                grid_cols = int(well_data[row_col[1]].max() + 1)
                n_tiles = grid_rows * grid_cols
            else:
                # Fallback to tile_idx if row/col not available
                n_tiles = int(well_data["tile_idx"].max() + 1)
                grid_cols = int(np.sqrt(n_tiles))
                grid_rows = (n_tiles + grid_cols - 1) // grid_cols

            n_cycles = int(well_data["Cycle"].max())
            channels = ["G", "T", "A", "C"]
            n_channels = len(channels)

            # Initialize SNR array
            snr_array = np.zeros((n_tiles, n_cycles, n_channels))
            snr_array.fill(np.nan)

            # Fill in SNR values from the merged data
            # The CSV has both signal and background rows, we need to reconstruct SNR
            for _, row_data in well_data.iterrows():
                # Use row and col to compute the correct linear tile index
                if "row" in row_data and "col" in row_data and pd.notna(row_data["row"]) and pd.notna(row_data["col"]):
                    tile_row = int(row_data["row"])
                    tile_col = int(row_data["col"])
                    tile_idx = tile_row * grid_cols + tile_col
                else:
                    # Fallback to stored tile_idx
                    tile_idx = int(row_data["tile_idx"])

                cycle_idx = int(row_data["Cycle"]) - 1  # Convert to 0-indexed
                channel = row_data["Channel"]

                if channel in channels and 0 <= tile_idx < n_tiles:
                    chan_idx = channels.index(channel)
                    # Use the pre-calculated SNR value
                    if "snr" in row_data and pd.notna(row_data["snr"]):
                        snr_array[tile_idx, cycle_idx, chan_idx] = row_data["snr"]

            all_snr_results[well] = snr_array
            well_grid_rows[well] = grid_rows
            well_grid_cols[well] = grid_cols
            print(f"  Loaded well {well}: {n_tiles} tiles ({grid_rows}x{grid_cols}), {n_cycles} cycles")

        print(f"\n✓ Successfully loaded data for {len(wells)} wells from CSV")
        return all_snr_results, per_tile_stats, well_grid_rows, well_grid_cols, True

    except Exception as e:
        print(f"✗ Error loading CSV data: {e}")
        import traceback
        traceback.print_exc()
        return {}, pd.DataFrame(), {}, {}, False


def iss_snr_bimodal(
    experiment: str,
    force_recompute: bool = False,
    crop_size: int = None,
) -> Dict[str, np.ndarray]:
    """
    Generate SNR heatmaps for an experiment using bimodal analysis.
    Processes all tiles in all wells using the converted ISS zarr store.
    Grid dimensions are automatically detected from the zarr data.

    Args:
        experiment: Experiment name
        force_recompute: If True, reprocess data even if CSVs exist
        crop_size: Center crop per tile for faster I/O (None = full tile)
    """
    dataset = OpsDataset(experiment)

    # Check if CSV files exist and load them if available
    all_snr_results = {}
    pooled_stats = pd.DataFrame()
    skip_processing = False

    if not force_recompute:
        all_snr_results, pooled_stats, well_grid_rows, well_grid_cols, csv_loaded = load_snr_from_csv(dataset)

        if csv_loaded and all_snr_results:
            # We have CSV data - skip to visualization
            print("\n✓ Using pre-computed data from CSV files")
            print("  (Use --force-recompute to reprocess raw data)\n")

            # Grid dimensions are already loaded from CSV
            # Skip to visualization section
            skip_processing = True
    else:
        print("\n⚠️  Force recompute enabled - will reprocess raw data\n")

    # Process raw data if needed
    well_load_times = {}
    well_compute_times = {}

    if not skip_processing:
        # Get list of wells from the converted ISS zarr store
        try:
            from iohub.ngff import open_ome_zarr
            import re

            store_path = dataset.store_paths["iss"]
            if not store_path.exists():
                print(f"ISS zarr store not found at {store_path}")
                print("Run convert + stack_symlinks first.")
                return {}

            with open_ome_zarr(store_path, mode="r") as ds:
                # Discover wells from positions (e.g., "A/1/000001" -> well "A/1" -> "A1")
                well_set = set()
                for pos_path, _ in ds.positions():
                    parts = Path(pos_path).parts
                    if len(parts) >= 2:
                        well_hcs = f"{parts[0]}/{parts[1]}"
                        # Convert HCS format to short form: "A/1" -> "A1"
                        well_short = f"{parts[0]}{parts[1]}"
                        well_set.add(well_short)
                wells = sorted(list(well_set))
        except Exception as e:
            print(f"Error listing wells from zarr store: {e}")
            return {}

        if not wells:
            print("No wells found in ISS zarr store.")
            return {}

        print(f"\n{'='*60}")
        print(f"Generating SNR Heatmaps (Bimodal) for {experiment}")
        print(f"Wells: {wells}")
        print(f"{'='*60}\n")

        all_snr_results = {}
        all_comprehensive_stats = []

        # Store grid dimensions (assuming all wells have same dimensions)
        well_grid_rows = {}
        well_grid_cols = {}

        # Process all wells in one flat parallel pool (no per-well barrier).
        batched = process_wells_snr_bimodal_batched(dataset, wells, crop_size=crop_size)
        for well in wells:
            if well not in batched:
                print(f"✗ No SNR data for well {well}; skipping")
                continue
            snr_results, comprehensive_stats, grid_rows, grid_cols, load_time, compute_time = batched[well]
            all_snr_results[well] = snr_results
            well_grid_rows[well] = grid_rows
            well_grid_cols[well] = grid_cols
            well_load_times[well] = load_time
            well_compute_times[well] = compute_time

            total_time = load_time + compute_time
            print(f"  ⏱️  Well {well}: Load={load_time:.1f}s, Compute={compute_time:.1f}s, Total={total_time:.1f}s")

            if not comprehensive_stats.empty:
                comprehensive_stats = comprehensive_stats.copy()
                comprehensive_stats["well"] = well
                all_comprehensive_stats.append(comprehensive_stats)

        if not all_snr_results:
            print("No SNR data was processed. Aborting visualization.")
            return {}

        # Aggregate stats and save CSVs
        if all_comprehensive_stats:
            pooled_stats = pd.concat(all_comprehensive_stats, ignore_index=True)
            try:
                save_comprehensive_stats_to_csv(pooled_stats, dataset)
                print(f"\n✓ Saved CSV files for future use")
            except Exception as e:
                print(f"Error saving comprehensive stats: {e}")

        # Print timing report
        if well_load_times:
            from prettytable import PrettyTable

            table = PrettyTable()
            table.field_names = ["Well", "Load Time (s)", "Compute Time (s)", "Total Time (s)"]
            table.align["Well"] = "l"
            table.align["Load Time (s)"] = "r"
            table.align["Compute Time (s)"] = "r"
            table.align["Total Time (s)"] = "r"

            total_load = 0
            total_compute = 0

            for well in sorted(well_load_times.keys()):
                load_t = well_load_times[well]
                compute_t = well_compute_times[well]
                total_t = load_t + compute_t

                total_load += load_t
                total_compute += compute_t

                table.add_row([well, f"{load_t:.1f}", f"{compute_t:.1f}", f"{total_t:.1f}"])

            # Add summary row
            total_total = total_load + total_compute
            table.add_row(["─" * 10, "─" * 14, "─" * 16, "─" * 14])
            table.add_row(["TOTAL", f"{total_load:.1f}", f"{total_compute:.1f}", f"{total_total:.1f}"])

            print(f"\n{'='*60}")
            print("Timing Report")
            print(f"{'='*60}")
            print(table)

    # === VISUALIZATION SECTION (runs whether data was loaded or computed) ===#
    if not all_snr_results:
        print("No SNR data available. Aborting visualization.")
        return {}

    # Global colormap range
    all_snr_values = np.concatenate(
        [results[~np.isnan(results)].flatten() for results in all_snr_results.values()]
    )

    if len(all_snr_values) > 0:
        vmin = 0
        vmax = min(np.percentile(all_snr_values, 95), 30)  # Cap at 30 SNR
        print(f"\nUsing colormap range: {vmin:.1f} - {vmax:.1f}")
    else:
        vmin, vmax = 0, 1

    # Generate visualizations
    print(f"\n{'='*60}")
    print("Generating visualizations")
    print(f"{'='*60}\n")

    snr_dir = dataset.results_iss / "SNR"
    snr_dir.mkdir(parents=True, exist_ok=True)

    # Use the first well's grid dimensions for combined plots
    # (assuming all wells have the same dimensions)
    wells = list(all_snr_results.keys())
    first_well = wells[0] if wells else None
    if first_well:
        combined_grid_rows = well_grid_rows[first_well]
        combined_grid_cols = well_grid_cols[first_well]

        try:
            # Use different filenames for bimodal outputs
            save_path_overall = dataset.metrics_paths["snr_heatmap_overall"].with_name("snr_heatmap_overall_bimodal.png")
            create_combined_overall_heatmaps(
                all_snr_results,
                combined_grid_rows,
                combined_grid_cols,
                experiment,
                save_path_overall,
                vmin=vmin,
                vmax=vmax,
            )
        except Exception as e:
            print(f"Error creating combined overall heatmap: {e}")

        try:
            save_path_per_channel = dataset.metrics_paths["snr_heatmap_per_channel"].with_name("snr_heatmap_per_channel_bimodal.png")
            create_combined_per_channel_heatmaps(
                all_snr_results,
                combined_grid_rows,
                combined_grid_cols,
                experiment,
                save_path_per_channel,
                vmin=vmin,
                vmax=vmax,
            )
        except Exception as e:
            print(f"Error creating combined per-channel heatmap: {e}")

    # Per-well visualizations
    print(f"\n--- Generating per-well detailed visualizations ---")
    per_round_dir = dataset.metrics_paths["snr_heatmap_per_channel_per_round"]
    per_round_dir.mkdir(parents=True, exist_ok=True)

    for well in all_snr_results.keys():
        try:
            snr_results = all_snr_results[well]
            well_sanitized = well.replace("/", "_")

            # Get this well's grid dimensions
            grid_rows = well_grid_rows[well]
            grid_cols = well_grid_cols[well]

            # Generate absolute SNR heatmap
            save_path_detailed = (
                per_round_dir
                / f"snr_heatmap_per_channel_per_round_{well_sanitized}_bimodal.png"
            )
            create_per_channel_per_round_heatmaps(
                snr_results,
                grid_rows,
                grid_cols,
                well,
                experiment,
                save_path_detailed,
                vmin=vmin,
                vmax=vmax,
            )

            # Generate normalized SNR heatmap (normalized to cycle 0 per channel)
            save_path_normalized = (
                per_round_dir
                / f"snr_heatmap_per_channel_per_round_{well_sanitized}_bimodal_normalized.png"
            )

            # Calculate dynamic vmin/vmax for normalized data to exclude outliers
            # Temporarily normalize to get the data distribution
            snr_normalized_temp = snr_results.copy()
            n_channels = snr_normalized_temp.shape[2]
            for chan_idx in range(n_channels):
                cycle0_values = snr_normalized_temp[:, 0, chan_idx]
                cycle0_mean = np.nanmean(cycle0_values)
                if cycle0_mean > 0:
                    snr_normalized_temp[:, :, chan_idx] = snr_normalized_temp[:, :, chan_idx] / cycle0_mean
                else:
                    snr_normalized_temp[:, :, chan_idx] = np.nan

            # Use 5th and 90th percentiles to exclude outliers (especially high outliers)
            valid_normalized_values = snr_normalized_temp[~np.isnan(snr_normalized_temp)]
            if len(valid_normalized_values) > 0:
                vmin_norm = max(0, np.percentile(valid_normalized_values, 5))
                vmax_norm = np.percentile(valid_normalized_values, 90)
                print(f"  Normalized colormap range for {well}: {vmin_norm:.2f} - {vmax_norm:.2f}")
            else:
                vmin_norm, vmax_norm = 0.5, 1.5

            create_per_channel_per_round_heatmaps(
                snr_results,
                grid_rows,
                grid_cols,
                well,
                experiment,
                save_path_normalized,
                vmin=vmin_norm,
                vmax=vmax_norm,
                normalize_to_cycle0=True,
            )
        except Exception as e:
            print(f"Error creating detailed heatmap for well {well}: {e}")

    # Load or use summary data for metric plots
    summary_df = None
    summary_csv = dataset.metrics_paths["snr_mean_per_round_per_channel"].with_name("snr_mean_per_round_per_channel_bimodal.csv")

    if summary_csv.exists():
        try:
            summary_df = pd.read_csv(summary_csv)
        except Exception as e:
            print(f"Error loading summary CSV: {e}")

    # Metric plots
    if summary_df is not None and not summary_df.empty:
        try:
            create_all_metric_plots(summary_df, dataset, experiment)
        except Exception as e:
            print(f"Error creating metric plots: {e}")

    # Summary stats
    well_summary_stats = {}
    if summary_df is not None and not summary_df.empty:
        try:
            well_summary_stats = calculate_summary_stats_for_metrics(summary_df)
            # No crosstalk in this method
            for stats in well_summary_stats.values():
                stats["mean_bleedthrough"] = 0.0
        except Exception as e:
            print(f"Error calculating summary stats: {e}")

    print(f"\n{'='*60}")
    print(f"SNR Analysis (Bimodal) Complete")
    print(f"Processed {len(all_snr_results)} wells")
    print(f"{'='*60}\n")

    return well_summary_stats


@click.command()
@click.option(
    "--experiment",
    required=True,
    type=str,
    help='Experiment name (e.g., "ops0033_20250429")',
)
@click.option(
    "--force-recompute",
    is_flag=True,
    default=False,
    help="Force recomputation from raw data even if CSV files exist",
)
@click.option(
    "--crop",
    is_flag=True,
    default=False,
    help="Use center 512x512 crop per tile for faster I/O (default: full tile)",
)
def main(
    experiment: str,
    force_recompute: bool,
    crop: bool,
):
    """
    Generate SNR heatmaps for an ISS experiment using Bimodal Analysis (no spots required).
    Processes all tiles in all wells. Grid dimensions are auto-detected.

    By default, if pre-computed CSV files exist, they will be used to regenerate visualizations.
    Use --force-recompute to reprocess raw data and overwrite existing CSVs.
    """
    import time
    from cyclops_utils.data.filesystem import resolve_experiment_name

    # Resolve experiment name (e.g. "98" -> "ops0098_2025...")
    experiment = resolve_experiment_name(experiment, verbose=True, allow_interactive=True)

    start_time = time.time()

    crop_size = 512 if crop else None
    results = iss_snr_bimodal(
        experiment=experiment,
        force_recompute=force_recompute,
        crop_size=crop_size,
    )

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print(f"\nTotal runtime: {minutes}m {seconds}s")

    if results:
        # Print summary statistics
        print("\nSummary Statistics:")
        for well, stats in results.items():
            print(f"  {well}:")
            print(f"    Median SNR: {stats['median_snr']:.1f}")
            print(f"    Median SBR: {stats['median_sbr']:.1f}")
            print(f"    Median LLD: {stats['median_lld']:.3f}")
            print(f"    Z' Factor: {stats['median_z_prime']:.3f}")


if __name__ == "__main__":
    main()
