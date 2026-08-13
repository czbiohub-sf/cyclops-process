"""
SNR Heatmap Generation for ISS Data

This module generates spatial heatmaps of Signal-to-Noise Ratio (SNR) across wells
with three levels of detail:
1. Overall mean SNR (averaged across all channels and rounds)
2. Per-channel mean SNR (averaged across all rounds)
3. Per-channel, per-round SNR (full detail)

Uses efficient tile-based sampling with lazy loading for fast processing.
Target: <3 minutes per well for 30x30 tile grid.

Usage:
    python -m cyclops_process.metrics.metrics_snr --experiment ops0033_20250429 --grid-size 30
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from cyclops_utils.data.experiment import OpsDataset
from cyclops_process.metrics.plate_stats.metrics_iss_utils import process_well_snr_tiles
from typing import Dict, List, Optional, Tuple
import click
from tqdm import tqdm
import pandas as pd


def create_combined_overall_heatmaps(
    all_snr_results: Dict[str, np.ndarray],
    grid_size: int,
    experiment: str,
    save_path: Path,
    vmin: float = 0,
    vmax: float = None,
    cmap: str = "viridis",
) -> None:
    """
    Create combined overall SNR heatmaps for all wells in one canvas.
    Layout: 1 row × n_wells columns

    Args:
        all_snr_results: Dictionary mapping well names to SNR arrays
        grid_size: Size of grid
        experiment: Experiment name
        save_path: Path to save figure
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        cmap: Colormap name
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
        snr_grid = np.zeros((grid_size, grid_size))
        snr_grid.fill(np.nan)

        n_tiles = len(mean_snr)
        for idx in range(min(n_tiles, grid_size * grid_size)):
            i = idx // grid_size
            j = idx % grid_size
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
        f"Mean SNR (All Channels & Rounds)\n{experiment}",
        fontsize=40,
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def create_combined_per_channel_heatmaps(
    all_snr_results: Dict[str, np.ndarray],
    grid_size: int,
    experiment: str,
    save_path: Path,
    vmin: float = 0,
    vmax: float = None,
    cmap: str = "viridis",
) -> None:
    """
    Create combined per-channel heatmaps for all wells in one canvas.
    Layout: 4 rows (channels) × n_wells columns

    Args:
        all_snr_results: Dictionary mapping well names to SNR arrays
        grid_size: Size of grid
        experiment: Experiment name
        save_path: Path to save figure
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        cmap: Colormap name
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
            snr_grid = np.zeros((grid_size, grid_size))
            snr_grid.fill(np.nan)

            n_tiles = len(snr_data)
            for idx in range(min(n_tiles, grid_size * grid_size)):
                i = idx // grid_size
                j = idx % grid_size
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
        f"Mean SNR per Channel (Averaged Across Rounds)\n{experiment}",
        fontsize=44,
        y=0.9995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.998])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def create_per_channel_per_round_heatmaps(
    snr_results: np.ndarray,
    grid_size: int,
    well: str,
    experiment: str,
    save_path: Path,
    vmin: float = 0,
    vmax: float = None,
    cmap: str = "viridis",
) -> None:
    """
    Create grid of heatmaps (4 channels × n_cycles).

    Args:
        snr_results: Array with shape (n_tiles, n_cycles, n_channels)
        grid_size: Size of grid
        well: Well identifier
        experiment: Experiment name
        save_path: Path to save figure
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        cmap: Colormap name
    """
    channels = ["G", "T", "A", "C"]
    n_tiles, n_cycles, n_channels = snr_results.shape

    # Create grid: n_cycles rows × 4 columns
    fig, axes = plt.subplots(n_cycles, 4, figsize=(18, 4.5 * n_cycles), squeeze=False)

    for cycle_idx in range(n_cycles):
        for chan_idx, chan_name in enumerate(channels):
            ax = axes[cycle_idx, chan_idx]
            snr_data = snr_results[:, cycle_idx, chan_idx]

            # Reshape to grid
            snr_grid = np.zeros((grid_size, grid_size))
            snr_grid.fill(np.nan)

            for idx in range(min(n_tiles, grid_size * grid_size)):
                i = idx // grid_size
                j = idx % grid_size
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

    fig.suptitle(
        f"SNR per Channel per Round\n{experiment} - {well}",
        fontsize=44,
        y=0.9995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.998])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def save_comprehensive_stats_to_csv(
    pooled_stats: pd.DataFrame,
    all_snr_results: Dict[str, np.ndarray],
    dataset: OpsDataset,
) -> None:
    """
    Save comprehensive imaging quality metrics to CSV for later analysis and reuse.

    This saves two files:
    1. Per-tile comprehensive stats (signal/background mean, std, SNR, etc.)
    2. Mean metrics per round per channel (averaged across tiles)

    Args:
        pooled_stats: DataFrame with comprehensive stats from all wells and tiles
        all_snr_results: Dictionary mapping well names to SNR arrays
        dataset: OpsDataset object for path resolution
    """
    from cyclops_process.metrics.plate_stats.metrics_iss_utils import (
        calculate_snr,
        calculate_sbr,
        calculate_lld,
        calculate_z_prime,
    )

    if pooled_stats.empty:
        print("\n⚠️  Warning: No comprehensive stats to save")
        return

    # === 1. Save per-tile comprehensive stats ===
    # Add calculated metrics to the pooled stats
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
        on=["well", "tile_idx", "Cycle", "Channel"],
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
    csv_path_per_tile = dataset.metrics_paths["snr_per_tile_data"]
    merged_df.to_csv(csv_path_per_tile, index=False, float_format="%.3f")
    print(f"\n✓ Saved comprehensive per-tile stats to: {csv_path_per_tile}")
    print(f"  Total rows: {len(merged_df):,}")
    print(f"  Wells: {merged_df['well'].nunique()}")
    print(f"  Tiles: {merged_df['tile_idx'].nunique()}")
    print(f"  Cycles: {merged_df['Cycle'].nunique()}")
    print(f"  Channels: {merged_df['Channel'].nunique()}")
    print(
        f"  Metrics: signal (mean/std/top10pct), background (mean/std), snr, sbr, lld, z_prime"
    )

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
                    # Calculate mean and std across tiles, excluding NaN
                    summary_rows.append(
                        {
                            "well": well,
                            "cycle": int(cycle),
                            "channel": channel,
                            "mean_signal": subset["signal_mean"].mean(),
                            "std_signal": subset["signal_mean"].std(),
                            "mean_signal_top10pct": subset["signal_top10pct"].mean(),
                            "mean_background": subset["background_mean"].mean(),
                            "std_background": subset["background_std"].mean(),
                            "mean_snr": subset["snr"].mean(),
                            "std_snr": subset["snr"].std(),
                            "mean_sbr": subset["sbr"].mean(),
                            "std_sbr": subset["sbr"].std(),
                            "mean_lld": subset["lld"].mean(),
                            "mean_z_prime": subset["z_prime"].mean(),
                            "n_tiles_used": len(subset[subset["signal_count"] > 0]),
                        }
                    )

    summary_df = pd.DataFrame(summary_rows)

    # Save summary data
    csv_path_summary = dataset.metrics_paths["snr_mean_per_round_per_channel"]
    summary_df.to_csv(csv_path_summary, index=False, float_format="%.3f")
    print(f"\n✓ Saved mean metrics per round per channel to: {csv_path_summary}")
    print(f"  Total rows: {len(summary_df):,}")
    print(f"  Wells: {summary_df['well'].nunique()}")
    print(f"  Cycles: {summary_df['cycle'].nunique()}")
    print(f"  Channels: {summary_df['channel'].nunique()}")
    print(f"  Metrics: mean ± std for signal, background, snr, sbr, lld, z_prime")


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

    Args:
        summary_df: DataFrame with columns [well, cycle, channel, <metric_col>]
        metric_col: Name of the metric column to plot
        ylabel: Y-axis label
        title: Plot title
        save_path: Path to save the figure
        experiment: Experiment name for subtitle
        ylim: Optional y-axis limits
        channel_colors: Optional custom colors for channels
        annotate: Whether to annotate values on points
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

    Args:
        summary_df: DataFrame with summary metrics per well/cycle/channel
        dataset: OpsDataset for path resolution
        experiment: Experiment name
    """
    print(f"\n{'='*60}")
    print("Generating metric vs cycle plots")
    print(f"{'='*60}\n")

    # 1. Signal intensity vs cycle
    plot_metric_vs_cycle(
        summary_df,
        metric_col="mean_signal",
        ylabel="Mean Signal Intensity (AU)",
        title="Signal Intensity vs Cycle",
        save_path=dataset.metrics_paths["iss_signal_vs_cycle"],
        experiment=experiment,
    )

    # 2. Top 10% signal intensity vs cycle
    plot_metric_vs_cycle(
        summary_df,
        metric_col="mean_signal_top10pct",
        ylabel="Median Top 10% Signal Intensity (AU)",
        title="Top 10% Signal Intensity vs Cycle",
        save_path=dataset.metrics_paths["iss_median_top10pct_vs_cycle"],
        experiment=experiment,
    )

    # 3. Background noise (std) vs cycle
    plot_metric_vs_cycle(
        summary_df,
        metric_col="std_background",
        ylabel="Background Noise (Std Dev)",
        title="Background Noise vs Cycle",
        save_path=dataset.metrics_paths["iss_background_noise_vs_cycle"],
        experiment=experiment,
    )

    # 4. Background mean vs cycle
    plot_metric_vs_cycle(
        summary_df,
        metric_col="mean_background",
        ylabel="Mean Background Intensity (AU)",
        title="Background Intensity vs Cycle",
        save_path=dataset.metrics_paths["iss_background_mean_vs_cycle"],
        experiment=experiment,
    )

    # 5. SNR vs cycle
    plot_metric_vs_cycle(
        summary_df,
        metric_col="mean_snr",
        ylabel="Signal-to-Noise Ratio (SNR)",
        title="SNR vs Cycle",
        save_path=dataset.metrics_paths["iss_snr_vs_cycle"],
        experiment=experiment,
        ylim=(0, None),  # SNR should start at 0
    )

    # 6. SBR vs cycle
    plot_metric_vs_cycle(
        summary_df,
        metric_col="mean_sbr",
        ylabel="Signal-to-Background Ratio (SBR)",
        title="SBR vs Cycle",
        save_path=dataset.metrics_paths["iss_sbr_vs_cycle"],
        experiment=experiment,
        ylim=(0, None),
    )

    # 7. LLD vs cycle (lower is better)
    plot_metric_vs_cycle(
        summary_df,
        metric_col="mean_lld",
        ylabel="Lower Limit of Detection (LLD)",
        title="LLD vs Cycle (Lower is Better)",
        save_path=dataset.metrics_paths["iss_lld_vs_cycle"],
        experiment=experiment,
    )

    # 8. Z'-factor vs cycle
    plot_metric_vs_cycle(
        summary_df,
        metric_col="mean_z_prime",
        ylabel="Z' Factor",
        title="Z' Factor vs Cycle (>0.5 is Excellent)",
        save_path=dataset.metrics_paths["iss_zprime_vs_cycle"],
        experiment=experiment,
        ylim=(-1, 1),  # Z' typically ranges from -inf to 1
    )

    print(f"\n✓ Completed all metric vs cycle plots\n")


def plot_crosstalk_heatmap(
    crosstalk_matrices: Dict[str, pd.DataFrame],
    save_path: Path,
    experiment: str,
) -> None:
    """
    Create crosstalk heatmap showing spectral bleeding between channels.

    Args:
        crosstalk_matrices: Dictionary mapping well names to crosstalk matrices
        save_path: Path to save the figure
        experiment: Experiment name
    """
    import seaborn as sns

    wells = sorted(crosstalk_matrices.keys())
    num_wells = len(wells)

    fig, axes = plt.subplots(1, num_wells, figsize=(6 * num_wells, 5))
    if num_wells == 1:
        axes = [axes]

    for ax, well in zip(axes, wells):
        matrix = crosstalk_matrices[well]

        # Plot heatmap
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            square=True,
            cbar_kws={"label": "Fraction of Signal"},
            ax=ax,
            linewidths=0.5,
            linecolor="gray",
        )

        ax.set_title(well, fontsize=16)
        ax.set_xlabel("Detected in Channel", fontsize=12)
        ax.set_ylabel("Brightest in Channel", fontsize=12)

        # Rotate labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.suptitle(f"Estimated Crosstalk Matrix\n{experiment}", fontsize=18)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {save_path}")


def save_crosstalk_matrices(
    crosstalk_matrices: Dict[str, pd.DataFrame],
    dataset: OpsDataset,
) -> None:
    """
    Save crosstalk matrices to CSV in two formats:
    1. Long format (all wells combined with well identifier)
    2. Matrix format (per-well, preserving channel x channel structure)

    Args:
        crosstalk_matrices: Dictionary mapping well names to crosstalk matrices
        dataset: OpsDataset for path resolution
    """
    # 1. Save long-format CSV (all wells combined)
    csv_path = dataset.metrics_paths["estimated_crosstalk_matrix"]

    # Combine all wells into one CSV with well identifier
    rows = []
    for well, matrix in crosstalk_matrices.items():
        for src_channel in matrix.index:
            for dst_channel in matrix.columns:
                rows.append(
                    {
                        "well": well,
                        "brightest_in": src_channel,
                        "detected_in": dst_channel,
                        "fraction": matrix.loc[src_channel, dst_channel],
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, float_format="%.4f")

    print(f"\n✓ Saved crosstalk matrices (long format) to: {csv_path}")
    print(f"  Total rows: {len(df):,}")
    print(f"  Wells: {df['well'].nunique()}")
    print(f"  Channels: {len(matrix.index)}")

    # 2. Save per-well matrix-format CSVs (preserving channel x channel structure)
    for well, matrix in crosstalk_matrices.items():
        per_well_path = dataset.append_well("estimated_crosstalk_matrix", well)
        per_well_path.parent.mkdir(parents=True, exist_ok=True)
        matrix.to_csv(per_well_path, float_format="%.4f")

    print(
        f"✓ Saved per-well crosstalk matrices (matrix format) for {len(crosstalk_matrices)} wells"
    )


def calculate_summary_stats_for_metrics(summary_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Calculate per-well summary statistics for integration with metrics.py.

    Args:
        summary_df: DataFrame with summary metrics per well/cycle/channel

    Returns:
        Dictionary mapping well names to their summary statistics
    """
    from cyclops_process.metrics.plate_stats.metrics_iss_utils import calculate_summary_metrics

    wells = summary_df["well"].unique()
    well_stats = {}

    for well in wells:
        well_data = summary_df[summary_df["well"] == well]

        # Calculate overall averages across all cycles and channels
        well_stats[well] = {
            "snr_mean_snr": well_data["mean_snr"].mean(),
            "snr_mean_sbr": well_data["mean_sbr"].mean(),
            "snr_mean_lld": well_data["mean_lld"].mean(),
            "snr_mean_z_prime": well_data["mean_z_prime"].mean(),
            "snr_mean_max_intensity": well_data["mean_signal"].mean(),
            "snr_median_top10pct_intensity": well_data["mean_signal_top10pct"].mean(),
            "snr_mean_background_mean": well_data["mean_background"].mean(),
            "snr_mean_background_std": well_data["std_background"].mean(),
            # Calculate slopes for SNR and SBR vs cycle
            "snr_snr_slope": 0.0,  # Will be calculated below
            "snr_sbr_slope": 0.0,  # Will be calculated below
        }

        # Calculate slopes (averaged across channels)
        try:
            from scipy.stats import linregress

            # SNR slope
            snr_per_cycle = well_data.groupby("cycle")["mean_snr"].mean()
            if len(snr_per_cycle) > 1:
                slope_snr, _, _, _, _ = linregress(
                    snr_per_cycle.index, snr_per_cycle.values
                )
                well_stats[well]["snr_snr_slope"] = round(slope_snr, 2)

            # SBR slope
            sbr_per_cycle = well_data.groupby("cycle")["mean_sbr"].mean()
            if len(sbr_per_cycle) > 1:
                slope_sbr, _, _, _, _ = linregress(
                    sbr_per_cycle.index, sbr_per_cycle.values
                )
                well_stats[well]["snr_sbr_slope"] = round(slope_sbr, 2)
        except Exception:
            pass  # Keep default 0.0 values

    return well_stats


def generate_snr_heatmaps(
    experiment: str,
    grid_size: int = 30,
    spots_per_tile: int = 100,
    method: Optional[str] = None,
    tile_sample_fraction: float = 0.1,
    well_inner_fraction: float = 0.75,
) -> Dict[str, np.ndarray]:
    """
    Generate SNR heatmaps for an experiment.

    Creates three types of visualizations:
    1. Overall mean SNR heatmap
    2. Per-channel mean SNR heatmaps (4 maps)
    3. Per-channel, per-round SNR heatmaps (n_cycles × 4 maps)

    Args:
        experiment: Experiment name
        grid_size: Number of tiles per dimension (default: 30)
        spots_per_tile: Maximum spots to sample per tile (default: 100)
        method: Base calling method (unused, for compatibility)
        tile_sample_fraction: Fraction of tile to load (default: 0.1)
        well_inner_fraction: Fraction of well to tile (default: 0.75)

    Returns:
        Dictionary mapping well names to SNR results arrays
    """
    dataset = OpsDataset(experiment, method=method)

    # Get list of wells
    from iohub.ngff import open_ome_zarr

    try:
        with open_ome_zarr(
            dataset.store_paths["iss_segmentation"], mode="r"
        ) as seg_store:
            wells = [a[0] for a in seg_store.positions()]
    except Exception as e:
        print(f"Error loading segmentation store: {e}")
        return {}

    if not wells:
        print("No wells found to process.")
        return {}

    print(f"\n{'='*60}")
    print(f"Generating SNR Heatmaps for {experiment}")
    print(f"Wells: {wells}")
    print(f"Grid size: {grid_size}x{grid_size} ({grid_size**2} tiles)")
    print(f"Spots per tile: {spots_per_tile}")
    print(f"{'='*60}\n")

    all_snr_results = {}
    all_comprehensive_stats = []
    all_crosstalk_matrices = {}

    # Process all wells to get SNR data, comprehensive stats, and crosstalk
    for well in wells:
        print(f"\n--- Processing well: {well} ---")
        try:
            # Process well to get SNR for all tiles, comprehensive stats, and crosstalk
            snr_results, comprehensive_stats, crosstalk_matrix = process_well_snr_tiles(
                dataset,
                well,
                grid_size=grid_size,
                spots_per_tile=spots_per_tile,
                signal_radius=5,
                tile_sample_fraction=tile_sample_fraction,
                well_inner_fraction=well_inner_fraction,
            )
            all_snr_results[well] = snr_results
            all_crosstalk_matrices[well] = crosstalk_matrix

            # Add well identifier to comprehensive stats and collect
            if not comprehensive_stats.empty:
                comprehensive_stats = comprehensive_stats.copy()
                comprehensive_stats["well"] = well
                all_comprehensive_stats.append(comprehensive_stats)

            # Get dimensions
            n_tiles, n_cycles, n_channels = snr_results.shape
            print(f"  SNR results shape: {snr_results.shape}")
            print(f"  Comprehensive stats rows: {len(comprehensive_stats):,}")

            # Calculate SNR range excluding NaN (tiles with no spots)
            valid_snr = snr_results[~np.isnan(snr_results)]
            if len(valid_snr) > 0:
                print(f"  SNR range: {valid_snr.min():.1f} - {valid_snr.max():.1f}")
            else:
                print("  Warning: All tiles have no spots or zero SNR")

            print(f"✓ Completed SNR processing for well {well}")

        except Exception as e:
            print(f"✗ Error processing well {well}: {e}")
            import traceback

            traceback.print_exc()
            continue

    if not all_snr_results:
        print("No SNR data was processed. Aborting visualization.")
        return {}

    # Calculate global colormap range across ALL wells for consistent scaling
    # Exclude NaN (tiles with no spots) from colormap calculation
    all_snr_values = np.concatenate(
        [results[~np.isnan(results)].flatten() for results in all_snr_results.values()]
    )

    if len(all_snr_values) > 0:
        vmin = 0
        vmax = np.percentile(
            all_snr_values, 95
        )  # Use 95th percentile to avoid outliers
        print(
            f"\nGlobal SNR range across all wells: {all_snr_values.min():.1f} - {all_snr_values.max():.1f}"
        )
        print(f"Using colormap range: {vmin:.1f} - {vmax:.1f}")
    else:
        vmin, vmax = 0, 1
        print("Warning: No valid SNR values found across all wells")

    # Generate combined visualizations
    print(f"\n--- Generating combined visualizations ---")

    # Ensure SNR output directory exists
    snr_dir = dataset.results_iss / "SNR"
    snr_dir.mkdir(parents=True, exist_ok=True)

    # 1. Combined overall mean SNR heatmap (all wells in one canvas)
    try:
        save_path_overall = dataset.metrics_paths["snr_heatmap_overall"]
        create_combined_overall_heatmaps(
            all_snr_results,
            grid_size,
            experiment,
            save_path_overall,
            vmin=vmin,
            vmax=vmax,
        )
    except Exception as e:
        print(f"Error creating combined overall heatmap: {e}")

    # 2. Combined per-channel mean SNR heatmaps (all wells in one canvas)
    try:
        save_path_per_channel = dataset.metrics_paths["snr_heatmap_per_channel"]
        create_combined_per_channel_heatmaps(
            all_snr_results,
            grid_size,
            experiment,
            save_path_per_channel,
            vmin=vmin,
            vmax=vmax,
        )
    except Exception as e:
        print(f"Error creating combined per-channel heatmap: {e}")

    # 3. Per-channel, per-round SNR heatmaps (separate file per well - too detailed for combined)
    print(f"\n--- Generating per-well detailed visualizations ---")
    # Ensure per-channel-per-round directory exists
    per_round_dir = dataset.metrics_paths["snr_heatmap_per_channel_per_round"]
    per_round_dir.mkdir(parents=True, exist_ok=True)

    for well in all_snr_results.keys():
        try:
            snr_results = all_snr_results[well]
            well_sanitized = well.replace("/", "_")
            save_path_detailed = (
                per_round_dir
                / f"snr_heatmap_per_channel_per_round_{well_sanitized}.png"
            )
            create_per_channel_per_round_heatmaps(
                snr_results,
                grid_size,
                well,
                experiment,
                save_path_detailed,
                vmin=vmin,
                vmax=vmax,
            )
        except Exception as e:
            print(f"Error creating detailed heatmap for well {well}: {e}")

    # Aggregate comprehensive stats from all wells
    if all_comprehensive_stats:
        pooled_stats = pd.concat(all_comprehensive_stats, ignore_index=True)
        print(f"\n--- Aggregated comprehensive stats ---")
        print(f"  Total rows: {len(pooled_stats):,}")
        print(f"  Wells: {pooled_stats['well'].nunique()}")
        print(f"  Cycles: {pooled_stats['Cycle'].nunique()}")
        print(f"  Channels: {pooled_stats['Channel'].nunique()}")
    else:
        pooled_stats = pd.DataFrame()
        print("\n⚠️  Warning: No comprehensive stats collected from any well")

    # Save comprehensive stats to CSV and get summary DataFrame for plotting
    summary_df = None
    if not pooled_stats.empty:
        try:
            save_comprehensive_stats_to_csv(pooled_stats, all_snr_results, dataset)
            # Read back the summary CSV for plotting
            summary_df = pd.read_csv(
                dataset.metrics_paths["snr_mean_per_round_per_channel"]
            )
        except Exception as e:
            print(f"Error saving comprehensive stats to CSV: {e}")
            import traceback

            traceback.print_exc()

    # Generate metric vs cycle line plots
    if summary_df is not None and not summary_df.empty:
        try:
            create_all_metric_plots(summary_df, dataset, experiment)
        except Exception as e:
            print(f"Error creating metric plots: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("\n⚠️  Skipping metric plots - no summary data available")

    # Save and plot crosstalk matrices
    if all_crosstalk_matrices:
        try:
            save_crosstalk_matrices(all_crosstalk_matrices, dataset)
            plot_crosstalk_heatmap(
                all_crosstalk_matrices,
                dataset.metrics_paths["estimated_crosstalk_heatmap"],
                experiment,
            )
        except Exception as e:
            print(f"Error processing crosstalk matrices: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("\n⚠️  No crosstalk matrices available")

    # Calculate summary statistics for downstream use (metrics.py)
    well_summary_stats = {}
    if summary_df is not None and not summary_df.empty:
        try:
            well_summary_stats = calculate_summary_stats_for_metrics(summary_df)

            # Add mean bleedthrough from crosstalk matrices
            for well, stats in well_summary_stats.items():
                if well in all_crosstalk_matrices:
                    matrix = all_crosstalk_matrices[well]
                    diagonal_mean = np.diag(matrix.values).mean()
                    stats["mean_bleedthrough"] = round(1 - diagonal_mean, 3)
                else:
                    stats["mean_bleedthrough"] = 0.0
        except Exception as e:
            print(f"Error calculating summary stats: {e}")

    print(f"\n{'='*60}")
    print(f"SNR Analysis Complete")
    print(f"Processed {len(all_snr_results)} wells successfully")
    print(f"Generated:")
    print(f"  - SNR heatmaps (3 types)")
    print(f"  - Metric line plots (8 types)")
    print(f"  - Crosstalk heatmap")
    print(f"  - Comprehensive CSV exports")
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
    "--grid-size",
    default=30,
    type=int,
    help="Grid size for tiling (default: 30 for 30x30 grid)",
)
@click.option(
    "--spots-per-tile",
    default=100,
    type=int,
    help="Maximum number of spots to sample per tile (default: 100)",
)
@click.option(
    "--tile-sample-fraction",
    default=0.1,
    type=float,
    help="Fraction of tile to load (0.1 = center 10%, faster; 1.0 = full tile, slower) (default: 0.1)",
)
@click.option(
    "--well-inner-fraction",
    default=0.85,
    type=float,
    help="Fraction of well to tile (0.75 = inner 75%, avoids edges with no spots) (default: 0.75)",
)
def main(
    experiment: str,
    grid_size: int,
    spots_per_tile: int,
    tile_sample_fraction: float,
    well_inner_fraction: float,
):
    """
    Generate SNR heatmaps for an ISS experiment.

    This script creates three types of SNR visualizations:
    1. Overall mean SNR (averaged across all channels and rounds)
    2. Per-channel mean SNR (4 heatmaps, averaged across rounds)
    3. Per-channel, per-round SNR (full detail: n_cycles × 4 heatmaps)

    All heatmaps use the same colormap scale for direct comparison.

    Example:
        python -m cyclops_process.metrics.metrics_snr \\
            --experiment ops0033_20250429 \\
            --grid-size 30 \\
            --spots-per-tile 100
    """
    import time

    start_time = time.time()

    results = generate_snr_heatmaps(
        experiment=experiment,
        grid_size=grid_size,
        spots_per_tile=spots_per_tile,
        tile_sample_fraction=tile_sample_fraction,
        well_inner_fraction=well_inner_fraction,
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
            print(f"    Mean SNR: {stats['mean_snr']:.1f}")
            print(f"    Mean SBR: {stats['mean_sbr']:.1f}")
            print(f"    Mean LLD: {stats['mean_lld']:.3f} (lower is better)")
            print(f"    Z' Factor: {stats['mean_z_prime']:.3f} (>0.5 is excellent)")
            print(f"    Mean Signal: {stats['mean_max_intensity']:.1f}")
            print(f"    Mean Background: {stats['mean_background_mean']:.1f}")
            print(f"    Mean Noise (Bg Std): {stats['mean_background_std']:.1f}")
            print(
                f"    Mean Bleedthrough: {stats['mean_bleedthrough']:.3f} (lower is better)"
            )


if __name__ == "__main__":
    main()
    # usage: python -m cyclops_process.metrics.metrics_snr --experiment ops0092_20251027 --grid-size 30 --spots-per-tile 100 --tile-sample-fraction 0.1
