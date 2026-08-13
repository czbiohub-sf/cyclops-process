import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ops_utils.data.experiment import OpsDataset
from iohub.ngff import open_ome_zarr
from pathlib import Path
import seaborn as sns
import click

# Import common functions
from cyclops_process.metrics.plate_stats.metrics_iss_utils import (
    _iss_filter_points,
    _iss_split_into_tiles,
    _detect_spots_on_fly,
    calculate_signal_background_stats,
    estimate_crosstalk_matrix,
    calculate_summary_metrics,
    _plot_metric_by_channel,
    resolve_iss_registered_store,
)








def snr_analysis_ISS(
    dataset: OpsDataset,
    well: str,
    grid_size: int = 13,
    save_debug_tifs: bool = False,
    debug_output_dir: Path = None,
):
    """
    Analyzes the signal and background noise
    Detects spots, defines signal regions around them, and treats the rest of the
    image as background. Calculates statistics and returns them.

    Args:
        dataset: OpsDataset object
        well: Well identifier
        grid_size: Grid size for tile splitting (0 for entire well)
        save_debug_tifs: If True, save intermediate TIF files for debugging
        debug_output_dir: Directory to save debug TIFs
    """
    # print(f"\n--- Analyzing noise for well: {well} ---")

    if not (grid_size == 0 or (grid_size % 2 != 0 and grid_size >= 1)):
        print(
            f"WARNING: `grid_size` must be 0 or an odd integer. Got {grid_size}. Defaulting to 5."
        )
        grid_size = 9
    if grid_size == 1:
        grid_size = 0

    sanitized_well = well.replace("/", "_")
    location_str = (
        "Entire Well"
        if grid_size == 0
        else f"Center Tile of {grid_size}x{grid_size} Grid"
    )

    try:
        stitch_ds = open_ome_zarr(resolve_iss_registered_store(dataset) / well, layout="fov", mode="r")
        shape = stitch_ds.data.shape[-2:]

        # Load pre-computed spots for the entire well, or detect on-the-fly
        spots_file = dataset.append_well("spots", well)
        if spots_file.exists():
            all_points = np.load(spots_file)
            print(f"Loaded {len(all_points)} pre-computed spots from {spots_file}")
        else:
            print(
                f"No pre-computed spots found at {spots_file}. Will detect on-the-fly after loading image data."
            )
            all_points = None  # Will detect after loading tile data

        if grid_size == 0:
            row_start, row_stop, col_start, col_stop = 0, shape[0], 0, shape[1]
        else:
            tiles, _ = _iss_split_into_tiles(shape, grid_size, 0)
            center_tile_index = (grid_size // 2) * grid_size + (grid_size // 2)
            row_start, row_stop, col_start, col_stop = tiles[center_tile_index]

        # print(f"Loading image for well {well}, {location_str}...")
        # Detect number of channels and skip DAPI (channel 0) only if 5 channels present
        # Note: stitch_ds is already opened at the well/FOV level, so access .data directly
        num_channels = stitch_ds.data.shape[1]

        # For backward compatibility: skip channel 0 (DAPI) if 5 channels, use all if 4
        channel_start = 1 if num_channels == 5 else 0

        tile_data_all_cycles = (
            stitch_ds
            .data[:, channel_start:, 0, row_start:row_stop, col_start:col_stop]
            .astype("float32")
        )

        # If spots weren't pre-computed, detect them on-the-fly from first cycle
        if all_points is None:
            first_cycle_data = tile_data_all_cycles[0, :, :, :]  # Shape: (C, Y, X)
            all_points = _detect_spots_on_fly(first_cycle_data)

        # Filter points to the selected tile if needed
        if grid_size == 0:
            points = all_points
        else:
            points = _iss_filter_points(tiles[center_tile_index], all_points)

    except (Exception, FileNotFoundError, KeyError) as e:
        print(
            f"Could not load prerequisite data for well {well} (e.g. stitched image or spots.npy). Skipping noise analysis. Error: {e}"
        )
        return None, None, None

    if points.shape[0] == 0:
        print(
            "No spots were detected/loaded for the selected region. Cannot perform noise analysis."
        )
        return None, None, None

    # Print sampling info for validation
    tile_height = row_stop - row_start
    tile_width = col_stop - col_start
    tile_area_pixels = tile_height * tile_width
    well_area_pixels = shape[0] * shape[1]
    coverage_pct = (tile_area_pixels / well_area_pixels) * 100
    print(
        f"SNR analysis region: {tile_height}x{tile_width} pixels ({coverage_pct:.1f}% of well), {points.shape[0]} spots"
    )

    # Set up debug output directory if needed
    if save_debug_tifs and debug_output_dir is None:
        debug_output_dir = dataset.results / "iss_metrics_debug"

    # Create debug prefix from well name
    debug_prefix = sanitized_well

    # Calculate signal and background statistics using common function
    summary_stats = calculate_signal_background_stats(
        tile_data_all_cycles,
        points,
        save_debug_tifs=save_debug_tifs,
        debug_output_dir=debug_output_dir,
        debug_prefix=debug_prefix,
    )

    # Estimate crosstalk matrix using common function
    crosstalk_matrix = estimate_crosstalk_matrix(tile_data_all_cycles, points)

    # Save crosstalk matrix if it exists
    if crosstalk_matrix is not None:
        crosstalk_save_path = dataset.append_well("estimated_crosstalk_matrix", well)
        crosstalk_save_path.parent.mkdir(parents=True, exist_ok=True)
        crosstalk_matrix.to_csv(crosstalk_save_path, float_format="%.4f")

    # Calculate summary metrics using common function
    well_stats = calculate_summary_metrics(summary_stats, crosstalk_matrix)
    print(f"Well stats: {well_stats}")

    return summary_stats, crosstalk_matrix, well_stats


def plot_noise_and_crosstalk_metrics(
    dataset: OpsDataset,
    all_summary_stats: dict,
    all_crosstalk_matrices: dict,
    distribution_assumption: str,
):
    """
    Generates combined, side-by-side plots for noise and crosstalk from pre-calculated data.
    """
    if not all_summary_stats:
        print("No summary stats data provided. Aborting plotting.")
        return

    # --- Plotting ---
    channels = ["G", "T", "A", "C"]
    channel_colors = {"G": "green", "T": "red", "A": "blue", "C": "orange"}
    num_wells = len(all_summary_stats)

    # --- Font sizes ---
    TITLE_SIZE = 20
    LABEL_SIZE = 16
    TICK_SIZE = 12
    LEGEND_SIZE = 12
    ANNOTATION_SIZE = 10

    # Plot 1: Mean Signal vs. Cycle
    fig_signal, axes_signal = plt.subplots(
        1, num_wells, figsize=(6 * num_wells, 5), sharey=True
    )
    if num_wells == 1:
        axes_signal = [axes_signal]
    for i, (well, stats) in enumerate(all_summary_stats.items()):
        ax = axes_signal[i]
        for chan_name in channels:
            signal_data = stats[
                (stats["Channel"] == chan_name) & (stats["Type"] == "Signal")
            ]
            if not signal_data.empty:
                # For Gaussian, use Standard Error of the Mean (SEM) instead of Standard Deviation
                y_err = (
                    np.sqrt(signal_data["mean"])
                    if distribution_assumption == "poisson"
                    else signal_data["std"] / np.sqrt(signal_data["count"])
                )
                plot_df = signal_data.copy()
                plot_df["y_mean"] = plot_df["mean"]
                plot_df["y_err"] = y_err
                _plot_metric_by_channel(
                    ax=ax,
                    df=plot_df,
                    channel_name=chan_name,
                    channel_color=channel_colors[chan_name],
                    x_col="Cycle",
                    y_col="y_mean",
                    yerr_col="y_err",
                    marker="o",
                    linestyle="-",
                    capsize=3,
                    annotate=True,
                    annotation_fontsize=ANNOTATION_SIZE,
                    label=chan_name,
                )
        ax.set_title(f"Well: {well}", fontsize=LABEL_SIZE)
        ax.set_xlabel("ISS Cycle", fontsize=LABEL_SIZE)
        ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
        ax.grid(True, linestyle="--")
    # Global y-limit across wells based on data (not first subplot limits)
    try:
        _sig_max = max(
            (
                df[df["Type"] == "Signal"]["mean"].max()
                for df in all_summary_stats.values()
            )
        )
        for _ax in axes_signal:
            _ax.set_ylim(0, _sig_max * 1.05)
    except Exception:
        pass
    axes_signal[0].set_ylabel(
        "Mean Spot Max Intensity"
        + (" (Error Bars: SEM)" if distribution_assumption == "gaussian" else ""),
        fontsize=LABEL_SIZE,
    )
    fig_signal.suptitle(
        f"Mean Spot Max Intensity vs. Cycle (Assumption: {distribution_assumption.capitalize()})\n{dataset.experiment}",
        fontsize=TITLE_SIZE,
    )
    handles, labels = axes_signal[-1].get_legend_handles_labels()
    fig_signal.legend(
        handles,
        labels,
        loc="upper right",
        title="Channel",
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_SIZE,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(dataset.metrics_paths["iss_signal_vs_cycle"], dpi=300)
    plt.close(fig_signal)

    # Plot 1b: Median Top 10% Brightest Spots vs. Cycle
    fig_top10, axes_top10 = plt.subplots(
        1, num_wells, figsize=(6 * num_wells, 5), sharey=True
    )
    if num_wells == 1:
        axes_top10 = [axes_top10]
    for i, (well, stats) in enumerate(all_summary_stats.items()):
        ax = axes_top10[i]
        for chan_name in channels:
            signal_data = stats[
                (stats["Channel"] == chan_name) & (stats["Type"] == "Signal")
            ]
            if not signal_data.empty and "median_top_10pct" in signal_data.columns:
                plot_df = signal_data.copy()
                plot_df["y_val"] = plot_df["median_top_10pct"]
                _plot_metric_by_channel(
                    ax=ax,
                    df=plot_df,
                    channel_name=chan_name,
                    channel_color=channel_colors[chan_name],
                    x_col="Cycle",
                    y_col="y_val",
                    yerr_col=None,
                    marker="s",
                    linestyle="-",
                    capsize=0,
                    annotate=True,
                    annotation_fontsize=ANNOTATION_SIZE,
                    label=chan_name,
                )
        ax.set_title(f"Well: {well}", fontsize=LABEL_SIZE)
        ax.set_xlabel("ISS Cycle", fontsize=LABEL_SIZE)
        ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
        ax.grid(True, linestyle="--")
    # Global y-limit across wells based on data
    try:
        _top10_max = max(
            (
                df[df["Type"] == "Signal"]["median_top_10pct"].max()
                for df in all_summary_stats.values()
                if "median_top_10pct" in df.columns
            )
        )
        for _ax in axes_top10:
            _ax.set_ylim(0, _top10_max * 1.05)
    except Exception:
        pass
    axes_top10[0].set_ylabel(
        "Median Top 10% Spot Intensity",
        fontsize=LABEL_SIZE,
    )
    fig_top10.suptitle(
        f"Median Top 10% Brightest Spots vs. Cycle\n{dataset.experiment}",
        fontsize=TITLE_SIZE,
    )
    handles, labels = axes_top10[-1].get_legend_handles_labels()
    fig_top10.legend(
        handles,
        labels,
        loc="upper right",
        title="Channel",
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_SIZE,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # Save to new path
    top10_path = dataset.results_iss / "iss_median_top10pct_vs_cycle_all_wells.png"
    plt.savefig(top10_path, dpi=300)
    plt.close(fig_top10)

    # Plot 2: Background Noise vs. Cycle
    fig_noise, axes_noise = plt.subplots(
        1, num_wells, figsize=(6 * num_wells, 5), sharey=True
    )
    if num_wells == 1:
        axes_noise = [axes_noise]
    for i, (well, stats) in enumerate(all_summary_stats.items()):
        ax = axes_noise[i]
        for chan_name in channels:
            background_data = stats[
                (stats["Channel"] == chan_name) & (stats["Type"] == "Background")
            ]
            if not background_data.empty:
                noise_y = (
                    np.sqrt(background_data["mean"])
                    if distribution_assumption == "poisson"
                    else background_data["std"]
                )
                plot_df = background_data.copy()
                plot_df["y_val"] = noise_y
                _plot_metric_by_channel(
                    ax=ax,
                    df=plot_df,
                    channel_name=chan_name,
                    channel_color=channel_colors[chan_name],
                    x_col="Cycle",
                    y_col="y_val",
                    yerr_col=None,
                    marker="x",
                    linestyle="--",
                    capsize=0,
                    annotate=True,
                    annotation_fontsize=ANNOTATION_SIZE,
                    label=chan_name,
                )
        ax.set_title(f"Well: {well}", fontsize=LABEL_SIZE)
        ax.set_xlabel("ISS Cycle", fontsize=LABEL_SIZE)
        ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
        ax.grid(True, linestyle="--")
    # Global y-limit across wells based on data (not first subplot limits)
    try:
        if distribution_assumption == "poisson":
            _noise_max = max(
                (
                    np.sqrt(df[df["Type"] == "Background"]["mean"]).max()
                    for df in all_summary_stats.values()
                )
            )
        else:
            _noise_max = max(
                (
                    df[df["Type"] == "Background"]["std"].max()
                    for df in all_summary_stats.values()
                )
            )
        for _ax in axes_noise:
            _ax.set_ylim(0, _noise_max * 1.05)
    except Exception:
        pass
    axes_noise[0].set_ylabel(
        "Background Noise (Standard Deviation)", fontsize=LABEL_SIZE
    )
    fig_noise.suptitle(
        f"Background Noise vs. Cycle (Assumption: {distribution_assumption.capitalize()})\n{dataset.experiment}",
        fontsize=TITLE_SIZE,
    )
    handles, labels = axes_noise[-1].get_legend_handles_labels()
    fig_noise.legend(
        handles,
        labels,
        loc="upper right",
        title="Channel",
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_SIZE,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(dataset.metrics_paths["iss_background_noise_vs_cycle"], dpi=300)
    plt.close(fig_noise)

    # Plot 3: SNR vs. Cycle
    fig_snr, axes_snr = plt.subplots(
        1, num_wells, figsize=(6 * num_wells, 5), sharey=True
    )
    if num_wells == 1:
        axes_snr = [axes_snr]
    for i, (well, stats) in enumerate(all_summary_stats.items()):
        ax = axes_snr[i]
        signal_stats = stats[stats["Type"] == "Signal"][
            ["Cycle", "Channel", "mean", "std", "count"]
        ].rename(
            columns={
                "mean": "signal_mean",
                "std": "signal_std",
                "count": "signal_count",
            }
        )
        background_stats = stats[stats["Type"] == "Background"][
            ["Cycle", "Channel", "mean", "std", "count"]
        ].rename(
            columns={
                "mean": "background_mean",
                "std": "background_std",
                "count": "background_count",
            }
        )
        snr_df = pd.merge(signal_stats, background_stats, on=["Cycle", "Channel"])

        background_std_safe = snr_df["background_std"] + 1e-9
        snr_df["snr"] = (
            snr_df["signal_mean"] - snr_df["background_mean"]
        ) / background_std_safe
        delta_A = snr_df["signal_std"] / np.sqrt(snr_df["signal_count"])
        delta_B = snr_df["background_std"] / np.sqrt(
            2 * (snr_df["background_count"] - 1)
        )
        signal_mean_safe = snr_df["signal_mean"] + 1e-9
        snr_df["snr_err"] = np.abs(snr_df["snr"]) * np.sqrt(
            (delta_A / signal_mean_safe) ** 2 + (delta_B / background_std_safe) ** 2
        )

        for chan_name in channels:
            channel_snr_data = snr_df[snr_df["Channel"] == chan_name]
            if not channel_snr_data.empty:
                plot_df = channel_snr_data.copy()
                plot_df["y_val"] = plot_df["snr"]
                plot_df["y_err"] = plot_df["snr_err"]
                _plot_metric_by_channel(
                    ax=ax,
                    df=plot_df,
                    channel_name=chan_name,
                    channel_color=channel_colors[chan_name],
                    x_col="Cycle",
                    y_col="y_val",
                    yerr_col="y_err",
                    marker="o",
                    linestyle="-",
                    capsize=3,
                    annotate=True,
                    annotation_fontsize=ANNOTATION_SIZE,
                    label=chan_name,
                )
        ax.set_ylim(bottom=0)
        ax.set_title(f"Well: {well}", fontsize=LABEL_SIZE)
        ax.set_xlabel("ISS Cycle", fontsize=LABEL_SIZE)
        ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
        ax.grid(True, linestyle="--")
    # Global y-limit across wells based on actual plotted SNR values
    try:
        _snr_max_vals = []
        for _df in all_summary_stats.values():
            _s = _df[_df["Type"] == "Signal"][["Cycle", "Channel", "mean"]].rename(
                columns={"mean": "signal_mean"}
            )
            _b = _df[_df["Type"] == "Background"][["Cycle", "Channel", "mean", "std"]].rename(
                columns={"mean": "background_mean", "std": "background_std"}
            )
            _m = pd.merge(_s, _b, on=["Cycle", "Channel"], how="inner")
            if not _m.empty:
                # Use same SNR formula as plotted: (signal - background) / background_std
                _snr = (_m["signal_mean"] - _m["background_mean"]) / (_m["background_std"] + 1e-9)
                _snr_max_vals.append(_snr.max())
        if _snr_max_vals:
            _snr_max = float(np.nanmax(_snr_max_vals))
            for _ax in axes_snr:
                _ax.set_ylim(0, _snr_max * 1.05)
    except Exception:
        pass
    axes_snr[0].set_ylabel("Signal-to-Noise Ratio (SNR)", fontsize=LABEL_SIZE)
    fig_snr.suptitle(
        f"Signal-to-Noise Ratio vs. Cycle\n{dataset.experiment}", fontsize=TITLE_SIZE
    )
    handles, labels = axes_snr[-1].get_legend_handles_labels()
    fig_snr.legend(
        handles,
        labels,
        loc="upper right",
        title="Channel",
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_SIZE,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(dataset.metrics_paths["iss_snr_vs_cycle"], dpi=300)
    plt.close(fig_snr)

    # Plot 4: SBR vs. Cycle
    fig_sbr, axes_sbr = plt.subplots(
        1, num_wells, figsize=(6 * num_wells, 5), sharey=True
    )
    if num_wells == 1:
        axes_sbr = [axes_sbr]
    for i, (well, stats) in enumerate(all_summary_stats.items()):
        ax = axes_sbr[i]
        signal_stats = stats[stats["Type"] == "Signal"][
            ["Cycle", "Channel", "mean"]
        ].rename(columns={"mean": "signal_mean"})
        background_stats = stats[stats["Type"] == "Background"][
            ["Cycle", "Channel", "mean"]
        ].rename(columns={"mean": "background_mean"})
        sbr_df = pd.merge(signal_stats, background_stats, on=["Cycle", "Channel"])

        background_mean_safe = sbr_df["background_mean"] + 1e-9
        sbr_df["sbr"] = sbr_df["signal_mean"] / background_mean_safe

        for chan_name in channels:
            channel_sbr_data = sbr_df[sbr_df["Channel"] == chan_name]
            if not channel_sbr_data.empty:
                plot_df = channel_sbr_data.copy()
                plot_df["y_val"] = plot_df["sbr"]
                _plot_metric_by_channel(
                    ax=ax,
                    df=plot_df,
                    channel_name=chan_name,
                    channel_color=channel_colors[chan_name],
                    x_col="Cycle",
                    y_col="y_val",
                    yerr_col=None,
                    marker="o",
                    linestyle="-",
                    capsize=3,
                    annotate=True,
                    annotation_fontsize=ANNOTATION_SIZE,
                    label=chan_name,
                )
        ax.set_ylim(bottom=0)
        ax.set_title(f"Well: {well}", fontsize=LABEL_SIZE)
        ax.set_xlabel("ISS Cycle", fontsize=LABEL_SIZE)
        ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
        ax.grid(True, linestyle="--")
    # Global y-limit across wells based on data
    try:
        _sbr_max_vals = []
        for _df in all_summary_stats.values():
            _s = _df[_df["Type"] == "Signal"][["Cycle", "Channel", "mean"]].rename(
                columns={"mean": "signal_mean"}
            )
            _b = _df[_df["Type"] == "Background"][["Cycle", "Channel", "mean"]].rename(
                columns={"mean": "background_mean"}
            )
            _m = pd.merge(_s, _b, on=["Cycle", "Channel"], how="inner")
            if not _m.empty:
                _sbr_max_vals.append(
                    (_m["signal_mean"] / (_m["background_mean"] + 1e-9)).max()
                )
        if _sbr_max_vals:
            _sbr_max = float(np.nanmax(_sbr_max_vals))
            for _ax in axes_sbr:
                _ax.set_ylim(0, _sbr_max * 1.05)
    except Exception:
        pass
    axes_sbr[0].set_ylabel("Signal-to-Background Ratio (SBR)", fontsize=LABEL_SIZE)
    fig_sbr.suptitle(
        f"Signal-to-Background Ratio vs. Cycle\n{dataset.experiment}",
        fontsize=TITLE_SIZE,
    )
    handles, labels = axes_sbr[-1].get_legend_handles_labels()
    fig_sbr.legend(
        handles,
        labels,
        loc="upper right",
        title="Channel",
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_SIZE,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(dataset.metrics_paths["iss_sbr_vs_cycle"], dpi=300)
    plt.close(fig_sbr)

    # Plot 5: LLD vs. Cycle
    fig_lld, axes_lld = plt.subplots(
        1, num_wells, figsize=(6 * num_wells, 5), sharey=True
    )
    if num_wells == 1:
        axes_lld = [axes_lld]
    for i, (well, stats) in enumerate(all_summary_stats.items()):
        ax = axes_lld[i]
        signal_stats = stats[stats["Type"] == "Signal"][
            ["Cycle", "Channel", "mean"]
        ].rename(columns={"mean": "signal_mean"})
        background_stats = stats[stats["Type"] == "Background"][
            ["Cycle", "Channel", "mean", "std"]
        ].rename(columns={"mean": "background_mean", "std": "background_std"})
        lld_df = pd.merge(signal_stats, background_stats, on=["Cycle", "Channel"])

        signal_minus_background = (
            lld_df["signal_mean"] - lld_df["background_mean"]
        ) + 1e-9
        lld_df["lld"] = (3 * lld_df["background_std"]) / signal_minus_background

        for chan_name in channels:
            channel_lld_data = lld_df[lld_df["Channel"] == chan_name]
            if not channel_lld_data.empty:
                plot_df = channel_lld_data.copy()
                plot_df["y_val"] = plot_df["lld"]
                _plot_metric_by_channel(
                    ax=ax,
                    df=plot_df,
                    channel_name=chan_name,
                    channel_color=channel_colors[chan_name],
                    x_col="Cycle",
                    y_col="y_val",
                    yerr_col=None,
                    marker="o",
                    linestyle="-",
                    capsize=3,
                    annotate=True,
                    annotation_fontsize=ANNOTATION_SIZE,
                    label=chan_name,
                )
        ax.set_ylim(bottom=0)
        ax.set_title(f"Well: {well}", fontsize=LABEL_SIZE)
        ax.set_xlabel("ISS Cycle", fontsize=LABEL_SIZE)
        ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
        ax.grid(True, linestyle="--")
    # Global y-limit across wells based on data
    try:
        _lld_max_vals = []
        for _df in all_summary_stats.values():
            _s = _df[_df["Type"] == "Signal"][["Cycle", "Channel", "mean"]].rename(
                columns={"mean": "signal_mean"}
            )
            _b = _df[_df["Type"] == "Background"][
                ["Cycle", "Channel", "mean", "std"]
            ].rename(columns={"mean": "background_mean", "std": "background_std"})
            _m = pd.merge(_s, _b, on=["Cycle", "Channel"], how="inner")
            if not _m.empty:
                _lld_max_vals.append(
                    (
                        (3 * _m["background_std"])
                        / ((_m["signal_mean"] - _m["background_mean"]) + 1e-9)
                    ).max()
                )
        if _lld_max_vals:
            _lld_max = float(np.nanmax(_lld_max_vals))
            for _ax in axes_lld:
                _ax.set_ylim(0, _lld_max * 1.05)
    except Exception:
        pass
    axes_lld[0].set_ylabel("Lower Limit of Detection (LLD)", fontsize=LABEL_SIZE)
    fig_lld.suptitle(
        f"Lower Limit of Detection vs. Cycle\n{dataset.experiment}", fontsize=TITLE_SIZE
    )
    handles, labels = axes_lld[-1].get_legend_handles_labels()
    fig_lld.legend(
        handles,
        labels,
        loc="upper right",
        title="Channel",
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_SIZE,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(dataset.metrics_paths["iss_lld_vs_cycle"], dpi=300)
    plt.close(fig_lld)

    # Plot 6: Z'-factor vs. Cycle
    fig_zprime, axes_zprime = plt.subplots(
        1, num_wells, figsize=(6 * num_wells, 5), sharey=True
    )
    if num_wells == 1:
        axes_zprime = [axes_zprime]
    for i, (well, stats) in enumerate(all_summary_stats.items()):
        ax = axes_zprime[i]
        signal_stats = stats[stats["Type"] == "Signal"][
            ["Cycle", "Channel", "mean", "std"]
        ].rename(columns={"mean": "signal_mean", "std": "signal_std"})
        background_stats = stats[stats["Type"] == "Background"][
            ["Cycle", "Channel", "mean", "std"]
        ].rename(columns={"mean": "background_mean", "std": "background_std"})
        zprime_df = pd.merge(signal_stats, background_stats, on=["Cycle", "Channel"])

        zprime_df["z_prime"] = 1 - (
            3 * (zprime_df["signal_std"] + zprime_df["background_std"])
        ) / (np.abs(zprime_df["signal_mean"] - zprime_df["background_mean"]) + 1e-9)

        for chan_name in channels:
            channel_zprime_data = zprime_df[zprime_df["Channel"] == chan_name]
            if not channel_zprime_data.empty:
                plot_df = channel_zprime_data.copy()
                plot_df["y_val"] = plot_df["z_prime"]
                _plot_metric_by_channel(
                    ax=ax,
                    df=plot_df,
                    channel_name=chan_name,
                    channel_color=channel_colors[chan_name],
                    x_col="Cycle",
                    y_col="y_val",
                    yerr_col=None,
                    marker="o",
                    linestyle="-",
                    capsize=3,
                    annotate=True,
                    annotation_fontsize=ANNOTATION_SIZE,
                    label=chan_name,
                )
        # Z' ranges from <0 to ~1, with 0.5 being good
        ax.axhline(
            y=0.5,
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label="Good threshold (0.5)",
        )
        ax.set_title(f"Well: {well}", fontsize=LABEL_SIZE)
        ax.set_xlabel("ISS Cycle", fontsize=LABEL_SIZE)
        ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
        ax.grid(True, linestyle="--")
    # Global y-limit across wells based on data (Z' can be negative)
    try:
        _zprime_min_vals = []
        _zprime_max_vals = []
        for _df in all_summary_stats.values():
            _s = _df[_df["Type"] == "Signal"][
                ["Cycle", "Channel", "mean", "std"]
            ].rename(columns={"mean": "signal_mean", "std": "signal_std"})
            _b = _df[_df["Type"] == "Background"][
                ["Cycle", "Channel", "mean", "std"]
            ].rename(columns={"mean": "background_mean", "std": "background_std"})
            _m = pd.merge(_s, _b, on=["Cycle", "Channel"], how="inner")
            if not _m.empty:
                _z = 1 - (3 * (_m["signal_std"] + _m["background_std"])) / (
                    np.abs(_m["signal_mean"] - _m["background_mean"]) + 1e-9
                )
                _zprime_min_vals.append(_z.min())
                _zprime_max_vals.append(_z.max())
        if _zprime_min_vals and _zprime_max_vals:
            _zprime_min = float(np.nanmin(_zprime_min_vals))
            _zprime_max = float(np.nanmax(_zprime_max_vals))
            # Add padding
            y_range = _zprime_max - _zprime_min
            for _ax in axes_zprime:
                _ax.set_ylim(_zprime_min - y_range * 0.05, _zprime_max + y_range * 0.05)
    except Exception:
        pass
    axes_zprime[0].set_ylabel("Z'-factor", fontsize=LABEL_SIZE)
    fig_zprime.suptitle(
        f"Z'-factor vs. Cycle\n{dataset.experiment}", fontsize=TITLE_SIZE
    )
    handles, labels = axes_zprime[-1].get_legend_handles_labels()
    fig_zprime.legend(
        handles,
        labels,
        loc="upper right",
        title="Channel",
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_SIZE,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(dataset.metrics_paths["iss_zprime_vs_cycle"], dpi=300)
    plt.close(fig_zprime)

    # Plot 7: Mean Background vs. Cycle
    fig_bg_mean, axes_bg_mean = plt.subplots(
        1, num_wells, figsize=(6 * num_wells, 5), sharey=True
    )
    if num_wells == 1:
        axes_bg_mean = [axes_bg_mean]
    for i, (well, stats) in enumerate(all_summary_stats.items()):
        ax = axes_bg_mean[i]
        for chan_name in channels:
            background_data = stats[
                (stats["Channel"] == chan_name) & (stats["Type"] == "Background")
            ]
            if not background_data.empty:
                _plot_metric_by_channel(
                    ax=ax,
                    df=background_data,
                    channel_name=chan_name,
                    channel_color=channel_colors[chan_name],
                    x_col="Cycle",
                    y_col="mean",
                    yerr_col=None,
                    marker="o",
                    linestyle="-",
                    capsize=3,
                    annotate=True,
                    annotation_fontsize=ANNOTATION_SIZE,
                    label=chan_name,
                )
        ax.set_ylim(bottom=0)
        ax.set_title(f"Well: {well}", fontsize=LABEL_SIZE)
        ax.set_xlabel("ISS Cycle", fontsize=LABEL_SIZE)
        ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
        ax.grid(True, linestyle="--")
    # Global y-limit across wells based on data
    try:
        _bg_mean_max_vals = []
        for _df in all_summary_stats.values():
            _bg = _df[_df["Type"] == "Background"]["mean"]
            if not _bg.empty:
                _bg_mean_max_vals.append(_bg.max())
        if _bg_mean_max_vals:
            _bg_mean_max = float(np.nanmax(_bg_mean_max_vals))
            for _ax in axes_bg_mean:
                _ax.set_ylim(0, _bg_mean_max * 1.05)
    except Exception:
        pass
    axes_bg_mean[0].set_ylabel("Mean Background Intensity", fontsize=LABEL_SIZE)
    fig_bg_mean.suptitle(
        f"Mean Background Intensity vs. Cycle\n{dataset.experiment}",
        fontsize=TITLE_SIZE,
    )
    handles, labels = axes_bg_mean[-1].get_legend_handles_labels()
    fig_bg_mean.legend(
        handles,
        labels,
        loc="upper right",
        title="Channel",
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_SIZE,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(dataset.metrics_paths["iss_background_mean_vs_cycle"], dpi=300)
    plt.close(fig_bg_mean)

    # Plot 8: Crosstalk Matrix Heatmaps
    if all_crosstalk_matrices:
        num_crosstalk_wells = len(all_crosstalk_matrices)
        fig_crosstalk, axes_crosstalk = plt.subplots(
            1, num_crosstalk_wells, figsize=(7 * num_crosstalk_wells, 6)
        )
        if num_crosstalk_wells == 1:
            axes_crosstalk = [axes_crosstalk]
        for i, (well, matrix) in enumerate(all_crosstalk_matrices.items()):
            ax = axes_crosstalk[i]
            sns.heatmap(
                matrix,
                annot=True,
                fmt=".2%",
                cmap="viridis",
                ax=ax,
                cbar=(i == num_crosstalk_wells - 1),
                vmin=0,
                vmax=1,
                annot_kws={"size": ANNOTATION_SIZE},
            )
            ax.set_title(f"Well: {well}", fontsize=LABEL_SIZE)
            ax.set_xlabel("Measured Channel", fontsize=LABEL_SIZE)
            ax.tick_params(axis="y", labelsize=TICK_SIZE)
            ax.tick_params(
                axis="x", labelsize=TICK_SIZE, rotation=45
            )  # rotate for readability
            if i == 0:
                ax.set_ylabel("Primary Signal Channel", fontsize=LABEL_SIZE)
            if i == num_crosstalk_wells - 1:
                if ax.collections:
                    cbar = ax.collections[0].colorbar
                    if cbar:
                        cbar.ax.tick_params(labelsize=TICK_SIZE)

        # --- Calculate and add mean crosstalk legend ---
        all_diagonals = []
        channels_list = list(all_crosstalk_matrices.values())[0].columns
        for matrix in all_crosstalk_matrices.values():
            # The diagonal represents on-target signal. Crosstalk is 1 - on-target.
            all_diagonals.append(1 - np.diag(matrix.values))

        mean_crosstalk_per_channel = np.mean(all_diagonals, axis=0)

        legend_parts = []
        for i, chan in enumerate(channels_list):
            legend_parts.append(f"{chan}: {mean_crosstalk_per_channel[i]:.1%}")
        legend_str = "Mean Crosstalk: " + " | ".join(legend_parts)

        fig_crosstalk.text(
            0.5,
            0.01,
            legend_str,
            ha="center",
            va="bottom",
            fontsize=LEGEND_SIZE,
            bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5),
        )

        fig_crosstalk.suptitle(
            f"Estimated Spectral Crosstalk Matrix\n{dataset.experiment}",
            fontsize=TITLE_SIZE,
        )
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        plt.savefig(dataset.metrics_paths["estimated_crosstalk_heatmap"], dpi=300)
        plt.close(fig_crosstalk)



def main_for_metrics(
    experiment: str,
    method: str,
    spotiflow_grid_size: int = 13,
    save_debug_tifs: bool = False,
    iss_rounds: list[int] | None = None,
    failed_rounds_by_well: dict[str, list[int]] | None = None,
    force: bool = False,
):
    """
    Main function to drive ISS signal/noise/crosstalk metrics for the primary metrics pipeline.
    It calculates stats for all wells and returns them in a dictionary.
    Optionally, it can run the plotting function for the first well if needed for debugging.

    Args:
        iss_rounds: List of ISS round indices to use (e.g., [0,1,2,3,4,5,6,7,8,9] or [10,11,12,13,14,15,16,17,18,19]).
                   If None, defaults to sequential rounds [0, 1, ..., len_reads-1].
        failed_rounds_by_well: Dictionary mapping wells to lists of failed round indices to exclude
        force: If True, regenerate stats even if cache exists. Default False.
    """
    # Default to sequential rounds if not specified
    if iss_rounds is None:
        iss_rounds = list(range(10))
    dataset = OpsDataset(experiment, method=method)

    # Check for cached stats — but only skip recomputation if plots also exist
    cache_path = dataset.results_iss / f"snr_stats_cache_{method}.csv"
    key_plots = [
        dataset.metrics_paths["iss_snr_vs_cycle"],
        dataset.metrics_paths["iss_signal_vs_cycle"],
        dataset.metrics_paths["estimated_crosstalk_heatmap"],
    ]
    plots_exist = all(p.exists() for p in key_plots)

    if cache_path.exists() and plots_exist and not force:
        print(f"Loading cached SNR stats from: {cache_path}")
        print("Use --force to regenerate.")
        try:
            cached_df = pd.read_csv(cache_path, index_col=0)
            # Convert DataFrame back to dict of dicts
            all_well_stats = cached_df.to_dict(orient="index")
            return all_well_stats
        except Exception as e:
            print(f"Failed to load cache, regenerating: {e}")
    elif cache_path.exists() and not plots_exist and not force:
        print(f"Cache exists but plots are missing — regenerating plots.")

    # Set up debug output directory if needed
    debug_output_dir = None
    if save_debug_tifs:
        debug_output_dir = dataset.results / "iss_metrics_debug"
        debug_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Debug TIFs will be saved to: {debug_output_dir}")

    try:
        with open_ome_zarr(
            dataset.store_paths["iss_segmentation"], mode="r"
        ) as seg_store:
            wells_to_process = [a[0] for a in seg_store.positions()]
    except (FileNotFoundError, KeyError):
        print("Could not find segmentation data. Cannot determine wells to process.")
        return {}

    if not wells_to_process:
        print("No wells found to process.")
        return {}

    # --- Stat Gathering ---
    all_well_stats = {}
    if not wells_to_process:
        return {}

    # print(f"\n--- Gathering Probabilistic Stats for Wells: {wells_to_process} ---")

    # 1. Get noise, bleedthrough, etc. for all wells
    noise_stats = {}
    all_summary_stats = {}
    all_crosstalk_matrices = {}
    for well in wells_to_process:
        print(f"--- Processing noise/bleedthrough for well: {well} ---")
        # Use 5x5 grid for larger center tile (avoids edge effects, more representative sampling)
        summary_stats, crosstalk_matrix, single_well_noise = snr_analysis_ISS(
            dataset,
            well,
            grid_size=9,
            save_debug_tifs=save_debug_tifs,
            debug_output_dir=debug_output_dir,
        )
        if summary_stats is not None:
            all_summary_stats[well] = summary_stats
        if crosstalk_matrix is not None:
            all_crosstalk_matrices[well] = crosstalk_matrix
        if single_well_noise is not None:
            noise_stats[well] = single_well_noise

    # --- Generate Noise/SNR/Crosstalk Plots ---
    if all_summary_stats:
        plot_noise_and_crosstalk_metrics(
            dataset=dataset,
            all_summary_stats=all_summary_stats,
            all_crosstalk_matrices=all_crosstalk_matrices,
            distribution_assumption="gaussian",  # Hardcoding for now
        )

    # 2. Merge all stats together
    for well in wells_to_process:
        all_well_stats[well] = {
            **noise_stats.get(well, {}),
        }

    # Save stats to cache
    if all_well_stats:
        try:
            stats_df = pd.DataFrame.from_dict(all_well_stats, orient="index")
            cache_path = dataset.results_iss / f"snr_stats_cache_{method}.csv"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            stats_df.to_csv(cache_path)
            print(f"Saved SNR stats cache to: {cache_path}")
        except Exception as e:
            print(f"Failed to save stats cache: {e}")

    return all_well_stats


@click.command()
@click.option(
    "--experiment",
    required=True,
    type=str,
    help='Experiment name (e.g., "ops12_20231024").',
)
@click.option(
    "--wells",
    default="all",
    type=str,
    help='Wells to analyze. Can be a comma-separated list (e.g., "A/1/0,A/2/0") or "all". The noise analysis will run on the first well in the list.',
)
@click.option(
    "--spotiflow-grid-size",
    default=5,
    type=int,
    help="Grid size for sampling. 0 for whole well, or an odd integer (3, 5, etc.).",
)
@click.option(
    "--distribution-assumption",
    default="gaussian",
    type=click.Choice(["gaussian", "poisson"]),
    help="Assumption for noise distribution analysis.",
)
@click.option(
    "--save-debug-tifs",
    is_flag=True,
    help="Save intermediate TIF files showing signal/background masks for debugging.",
)
def main(
    experiment: str,
    wells: str,
    spotiflow_grid_size: int,
    distribution_assumption: str,
    save_debug_tifs: bool,
):
    """
    Generates a set of plots to characterize the signal and background noise in ISS data.
    """
    dataset = OpsDataset(experiment, method="mine")

    # --- Parse Wells ---
    if wells.lower() == "all":
        try:
            iss_seg_store = open_ome_zarr(
                dataset.store_paths["iss_segmentation"], mode="r"
            )
            wells_to_process = [a[0] for a in iss_seg_store.positions()][
                :3
            ]  # Default to first 3 wells
            print(f"Found {len(wells_to_process)} wells. Analyzing: {wells_to_process}")
        except Exception as e:
            print(
                f"Could not open ISS segmentation store to get all wells. Using primary well 'A/1/0'. Error: {e}"
            )
            wells_to_process = ["A/1/0"]
    else:
        wells_to_process = [well.strip() for well in wells.split(",")]
        print(f"Processing specified wells: {wells_to_process}")

    if not wells_to_process:
        print("No wells specified or found. Aborting.")
        return

    # Define and create output directory
    output_dir = dataset.results / "base_calling"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Run Analysis ---
    all_well_stats = main_for_metrics(
        experiment,
        method="mine",  # Pass method to main_for_metrics
        spotiflow_grid_size=spotiflow_grid_size,
        save_debug_tifs=save_debug_tifs,
    )

    # --- Plotting ---
    # This part is now handled within main_for_metrics
    # The main function will call main_for_metrics and then plot if needed.
    # For now, we just return the stats.
    print("\n--- Summary of All Wells ---")
    for well, stats in all_well_stats.items():
        print(f"\nWell: {well}")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    # --- Save stats to CSV ---
    if all_well_stats:
        stats_df = pd.DataFrame.from_dict(all_well_stats, orient="index")
        stats_csv_path = output_dir / f"{experiment}_well_stats_summary.csv"
        stats_df.to_csv(stats_csv_path)
        print(f"\n--- Saved well stats summary to: {stats_csv_path} ---")


if __name__ == "__main__":
    main()
