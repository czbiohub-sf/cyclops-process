"""
Common functions and utilities for ISS metrics analysis.
Shared between metrics_probs_ISS.py and metrics_iss_tiles.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ops_utils.data.experiment import OpsDataset
from scipy.spatial import KDTree
from typing import Tuple, List, Optional
try:
    import torch
except ModuleNotFoundError:  # optional GPU dep; used only in GPU runtime paths
    torch = None
from pathlib import Path
import seaborn as sns
import scipy
from scipy.stats import linregress
import matplotlib.ticker as mtick
from skimage import io as skio
from joblib import Parallel, delayed
from ops_utils.hpc.resource_manager import get_optimal_workers


# --- Helper functions ---

def resolve_iss_registered_store(dataset) -> Path:
    """Path to the registered ISS store, preferring the v3 store.

    The pipeline converts iss to v3 right after merge and async-deletes the v2
    source, so v3 is the live store. Falls back to the v2 path for legacy
    experiments that were never re-run (iohub reads both layouts).
    """
    v3 = dataset.store_paths.get("iss_stitch_registered_v3")
    if v3 is not None and v3.exists():
        return v3
    return dataset.store_paths["iss_stitch_registered"]


def bf_normalize_bases(df):
    """Normalize the channel intensities by the median brightness of each channel in all spots.

    Args:
        df (pandas.DataFrame): DataFrame containing spot intensity data.

    Returns:
        pandas.DataFrame: DataFrame with normalized intensity values.
    """
    # Calculate median brightness of each channel
    df_medians = df.groupby("channel").intensity.median()

    # Vectorized normalization - map channel to median, then divide
    df_out = df.copy()
    df_out.intensity = df_out.intensity / df_out.channel.map(df_medians)

    return df_out

from ops_utils.io.tiling import split_into_tiles


def _iss_max_filter(data: np.array, width: int = 3) -> np.array:
    """Apply a maximum filter in a window of `width`."""
    return scipy.ndimage.filters.maximum_filter(data, size=(1, 1, width, width))


def _iss_filter_points(tile: Tuple, points: np.array) -> np.array:
    """Helper function to select points that fall within a given tile's boundaries."""
    # tile is expected to be (row_start, row_stop, col_start, col_stop)
    row_start, row_stop, col_start, col_stop = tile
    points_in_tile = points[
        (points[:, 0] >= row_start)
        & (points[:, 0] < row_stop)
        & (points[:, 1] >= col_start)
        & (points[:, 1] < col_stop)
    ]
    # Return points with coordinates relative to the tile's origin
    return points_in_tile - np.array([row_start, col_start])


def _iss_split_into_tiles(arr_shape: Tuple, n: int, overlap: int) -> List[Tuple[int]]:
    """Helper function to divide a large FOV into smaller, processable tiles."""
    tiles = []
    height, width = arr_shape
    tile_height = height // n
    tile_width = width // n
    row_stride = tile_height - overlap
    col_stride = tile_width - overlap
    for i in range(n):
        row_start = i * row_stride
        row_stop = row_start + tile_height
        if row_stop > height:
            row_stop = height
        for j in range(n):
            col_start = j * col_stride
            col_stop = col_start + tile_width
            if col_stop > width:
                col_stop = width
            tiles.append((row_start, row_stop, col_start, col_stop))
    return tiles, list(range(len(tiles)))


def get_tile_coords_from_snake(position: str, total_positions: int) -> Tuple[int, int]:
    """
    Extract tile number and convert to grid coordinates assuming snake pattern.

    Args:
        position: Position string (e.g., "A/1/005014")
        total_positions: Total number of positions in the well

    Returns:
        Tuple of (row, col) grid coordinates
    """
    tile_num = int(position.split("/")[-1])
    grid_size = int(np.sqrt(total_positions))
    row = tile_num // grid_size
    # Snake pattern: even rows go left-to-right, odd rows go right-to-left
    col = (
        tile_num % grid_size
        if row % 2 == 0
        else (grid_size - 1 - (tile_num % grid_size))
    )
    return row, col


def select_center_positions(
    well_positions: List[str],
    num_positions: int,
    seed: int,
    center_fraction: float = 0.5,
) -> List[str]:
    """
    Select positions from the center region of a well, avoiding edge tiles.

    Args:
        well_positions: List of position strings
        num_positions: Number of positions to select
        seed: Random seed for reproducibility
        center_fraction: Fraction of tiles closest to center to sample from (default 0.5 = inner 50%)

    Returns:
        List of selected position strings
    """
    rng = np.random.RandomState(seed)
    total_positions = len(well_positions)

    if num_positions >= total_positions:
        return well_positions

    # Calculate center of grid and distance from center for each position
    grid_size = int(np.sqrt(total_positions))
    center = grid_size / 2.0
    distances = []

    for i, pos in enumerate(well_positions):
        try:
            row, col = get_tile_coords_from_snake(pos, total_positions)
            dist = np.sqrt((row - center) ** 2 + (col - center) ** 2)
            distances.append((i, dist))
        except:
            distances.append((i, 999))  # Large distance for parsing errors

    # Select from inner tiles (closest to center)
    distances.sort(key=lambda x: x[1])
    inner_tile_count = max(num_positions, int(len(distances) * center_fraction))
    available_indices = [idx for idx, _ in distances[:inner_tile_count]]

    # Randomly sample positions from center tiles
    indices = rng.choice(available_indices, size=num_positions, replace=False)
    indices = sorted(indices)
    return [well_positions[i] for i in indices]


def _detect_spots_on_fly(
    image_data: np.ndarray, min_distance: int = 5, prob_threshold: float = 0.7
) -> np.ndarray:
    """
    Detect spots on-the-fly using Spotiflow when pre-computed spots don't exist.

    Args:
        image_data: Image data with shape (C, Y, X) where C is number of channels (4 for GTAC)
        min_distance: Minimum distance between spots
        prob_threshold: Probability threshold for spot detection

    Returns:
        Array of spot coordinates (N, 2) in (y, x) format
    """
    from spotiflow.model import Spotiflow

    def normalize(x, mi, ma, eps: float = 1e-20):
        return (x - mi) / (ma - mi + eps)

    print("Detecting spots on-the-fly using Spotiflow...")
    peak_model = Spotiflow.from_pretrained("general")

    all_channel_points = []
    for ch_idx in range(image_data.shape[0]):
        channel_img = image_data[ch_idx, :, :]
        p1 = np.percentile(channel_img, 1)
        p998 = np.percentile(channel_img, 99.8)

        points, details = peak_model.predict(
            channel_img,
            min_distance=min_distance,
            prob_thresh=prob_threshold,
            n_tiles=(10, 10),
            normalizer=lambda x: normalize(x, p1, p998),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            verbose=False,
        )
        all_channel_points.append(points)

    # Combine points from all channels
    all_points = np.vstack(all_channel_points)

    # Remove duplicates using KDTree
    if len(all_points) > 0:
        point_tree = KDTree(all_points)
        pairs = point_tree.query_pairs(min_distance, output_type="ndarray")
        if len(pairs) > 0:
            index_to_remove = np.unique(pairs[:, 1])
            all_points = np.delete(all_points, index_to_remove, axis=0)

    print(f"Detected {len(all_points)} spots")
    return all_points


# --- Signal/Background Analysis Functions ---


def create_background_mask(
    points: np.ndarray, image_shape: Tuple[int, int], signal_radius: int = 5
) -> np.ndarray:
    """
    Create a boolean mask where True indicates background pixels.

    Args:
        points: Array of spot coordinates (N, 2) in (y, x) format
        image_shape: Shape of the image (H, W)
        signal_radius: Radius around each spot to exclude from background

    Returns:
        Boolean array of shape (H, W) where True = background
    """
    H, W = image_shape
    background_mask = np.ones((H, W), dtype=bool)
    points_int = np.round(points).astype(int)

    # Ensure points are within bounds
    points_valid = points_int[
        (points_int[:, 0] >= 0)
        & (points_int[:, 0] < H)
        & (points_int[:, 1] >= 0)
        & (points_int[:, 1] < W)
    ]

    y, x = np.ogrid[
        -signal_radius : signal_radius + 1, -signal_radius : signal_radius + 1
    ]
    disk = x**2 + y**2 <= signal_radius**2

    for r, c in points_valid:
        r_start, r_end = max(0, r - signal_radius), min(H, r + signal_radius + 1)
        c_start, c_end = max(0, c - signal_radius), min(W, c + signal_radius + 1)
        disk_r_start, disk_r_end = signal_radius - (r - r_start), signal_radius + (
            r_end - r
        )
        disk_c_start, disk_c_end = signal_radius - (c - c_start), signal_radius + (
            c_end - c
        )
        background_mask[r_start:r_end, c_start:c_end] &= ~disk[
            disk_r_start:disk_r_end, disk_c_start:disk_c_end
        ]

    return background_mask


def calculate_signal_background_stats(
    tile_data_all_cycles: np.ndarray,
    points: np.ndarray,
    channels: List[str] = ["G", "T", "A", "C"],
    signal_radius: int = 5,
    save_debug_tifs: bool = False,
    debug_output_dir: Optional[Path] = None,
    debug_prefix: str = "debug",
) -> pd.DataFrame:
    """
    Calculate signal and background statistics for all cycles and channels.

    Args:
        tile_data_all_cycles: Image data with shape (n_cycles, n_channels, H, W)
        points: Array of spot coordinates (N, 2) in (y, x) format
        channels: List of channel names
        signal_radius: Radius around each spot for signal measurement
        save_debug_tifs: If True, save intermediate TIF files for debugging
        debug_output_dir: Directory to save debug TIFs (required if save_debug_tifs=True)
        debug_prefix: Prefix for debug filenames

    Returns:
        DataFrame with columns: Cycle, Channel, Type, mean, median, std, count
    """
    n_cycles, n_channels, H, W = tile_data_all_cycles.shape

    # Create background mask
    background_mask = create_background_mask(points, (H, W), signal_radius)

    # Get valid point coordinates
    points_int = np.round(points).astype(int)
    points_valid = points_int[
        (points_int[:, 0] >= 0)
        & (points_int[:, 0] < H)
        & (points_int[:, 1] >= 0)
        & (points_int[:, 1] < W)
    ]
    points_y, points_x = points_valid[:, 0], points_valid[:, 1]

    # Save debug TIFs if requested
    if save_debug_tifs:
        if debug_output_dir is None:
            raise ValueError(
                "debug_output_dir must be provided when save_debug_tifs=True"
            )

        debug_output_dir = Path(debug_output_dir)
        debug_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Saving debug TIFs to {debug_output_dir}")

        # Create a signal mask (inverse of background mask)
        signal_mask = ~background_mask

        # Save masks
        skio.imsave(
            str(debug_output_dir / f"0-{debug_prefix}_background_mask.tif"),
            background_mask.astype(np.uint8) * 255,
            check_contrast=False,
        )
        skio.imsave(
            str(debug_output_dir / f"1-{debug_prefix}_signal_mask.tif"),
            signal_mask.astype(np.uint8) * 255,
            check_contrast=False,
        )

        # Create composite RGB images from all 4 channels
        # Channel colors: G=green, T=red, A=blue, C=orange
        channel_colors = {
            "G": [0, 1, 0],  # Green
            "T": [1, 0, 0],  # Red
            "A": [0, 0, 1],  # Blue
            "C": [1, 0.65, 0],  # Orange
        }

        def normalize_for_display(img, percentile_low=1, percentile_high=99.8):
            """Normalize image to 0-1 range using percentile scaling."""
            p_low = np.percentile(img, percentile_low)
            p_high = np.percentile(img, percentile_high)
            img_norm = np.clip((img - p_low) / (p_high - p_low + 1e-10), 0, 1)
            return img_norm

        # Create composite images
        composite_raw = np.zeros((H, W, 3), dtype=np.float32)
        composite_signal = np.zeros((H, W, 3), dtype=np.float32)
        composite_background = np.zeros((H, W, 3), dtype=np.float32)

        for chan_idx, chan_name in enumerate(channels):
            raw_image = tile_data_all_cycles[0, chan_idx, :, :].copy()
            normalized = normalize_for_display(raw_image)
            color = np.array(channel_colors[chan_name])

            # Add to composite raw image
            for c in range(3):
                composite_raw[:, :, c] += normalized * color[c]

            # Signal-masked version
            signal_only = normalized.copy()
            signal_only[background_mask] = 0
            for c in range(3):
                composite_signal[:, :, c] += signal_only * color[c]

            # Background-masked version
            background_only = normalized.copy()
            background_only[signal_mask] = 0
            for c in range(3):
                composite_background[:, :, c] += background_only * color[c]

        # Normalize composites to 0-255 range
        composite_raw = np.clip(composite_raw * 255, 0, 255).astype(np.uint8)
        composite_signal = np.clip(composite_signal * 255, 0, 255).astype(np.uint8)
        composite_background = np.clip(composite_background * 255, 0, 255).astype(
            np.uint8
        )

        # Save composite images
        skio.imsave(
            str(debug_output_dir / f"2-{debug_prefix}_cycle1_composite_raw.tif"),
            composite_raw,
            check_contrast=False,
        )
        skio.imsave(
            str(
                debug_output_dir
                / f"3-{debug_prefix}_cycle1_composite_signal_masked.tif"
            ),
            composite_signal,
            check_contrast=False,
        )
        skio.imsave(
            str(
                debug_output_dir
                / f"4-{debug_prefix}_cycle1_composite_background_masked.tif"
            ),
            composite_background,
            check_contrast=False,
        )

        # Create an RGB overlay showing masks on top of grayscale data
        # Use first channel as base grayscale image
        base_gray = normalize_for_display(tile_data_all_cycles[0, 0, :, :])
        overlay = np.stack([base_gray, base_gray, base_gray], axis=-1)
        overlay = (overlay * 0.5 * 255).astype(np.uint8)  # Dim the base image

        # Highlight signal regions with red outline
        from scipy.ndimage import binary_erosion

        signal_edges = signal_mask & ~binary_erosion(signal_mask, iterations=1)
        overlay[signal_edges, 0] = 255  # Red edges
        overlay[signal_edges, 1] = 0
        overlay[signal_edges, 2] = 0

        # Mark spot centers with bright green crosses
        for y, x in zip(points_y, points_x):
            if 0 <= y < H and 0 <= x < W:
                # Draw cross
                overlay[max(0, y - 2) : min(H, y + 3), x] = [0, 255, 0]  # Vertical
                overlay[y, max(0, x - 2) : min(W, x + 3)] = [0, 255, 0]  # Horizontal

        skio.imsave(
            str(debug_output_dir / f"5-{debug_prefix}_mask_overlay.tif"),
            overlay,
            check_contrast=False,
        )

        print(f"Saved debug images:")
        print(
            f"  - 0: Background mask (white=background): 0-{debug_prefix}_background_mask.tif"
        )
        print(
            f"  - 1: Signal mask (white=signal regions): 1-{debug_prefix}_signal_mask.tif"
        )
        print(
            f"  - 2: Composite raw (G=green, T=red, A=blue, C=orange): 2-{debug_prefix}_cycle1_composite_raw.tif"
        )
        print(
            f"  - 3: Composite signal-masked: 3-{debug_prefix}_cycle1_composite_signal_masked.tif"
        )
        print(
            f"  - 4: Composite background-masked: 4-{debug_prefix}_cycle1_composite_background_masked.tif"
        )
        print(
            f"  - 5: Mask overlay (gray=data, red=signal region edge, green cross=spot center): 5-{debug_prefix}_mask_overlay.tif"
        )

    all_stats_list = []

    # Apply max filter to get local maxima around spots (width=10 to match base calling)
    tile_data_max_filtered = _iss_max_filter(tile_data_all_cycles, width=10)

    for cycle_idx in range(n_cycles):
        for chan_idx, chan_name in enumerate(channels):
            # Use max-filtered data for signal to get peak intensity around each spot
            channel_image_maxfilt = tile_data_max_filtered[cycle_idx, chan_idx, :, :]
            signal_pixels = channel_image_maxfilt[points_y, points_x]

            # Use original data for background
            channel_image_orig = tile_data_all_cycles[cycle_idx, chan_idx, :, :]
            background_pixels = channel_image_orig[background_mask]

            # Downsample background if too large
            if len(background_pixels) > 500_000:
                background_pixels = np.random.choice(
                    background_pixels, 500_000, replace=False
                )

            # Calculate median of top 10% brightest spots
            if len(signal_pixels) > 0:
                top_10pct_count = max(1, int(len(signal_pixels) * 0.1))
                top_10pct_intensities = np.partition(signal_pixels, -top_10pct_count)[
                    -top_10pct_count:
                ]
                median_top_10pct = np.median(top_10pct_intensities)
            else:
                median_top_10pct = 0

            all_stats_list.append(
                {
                    "Cycle": cycle_idx + 1,
                    "Channel": chan_name,
                    "Type": "Signal",
                    "mean": np.mean(signal_pixels),
                    "median": np.median(signal_pixels),
                    "median_top_10pct": median_top_10pct,
                    "std": np.std(signal_pixels),
                    "count": len(signal_pixels),
                }
            )
            all_stats_list.append(
                {
                    "Cycle": cycle_idx + 1,
                    "Channel": chan_name,
                    "Type": "Background",
                    "mean": np.mean(background_pixels),
                    "median": np.median(background_pixels),
                    "std": np.std(background_pixels),
                    "count": len(background_pixels),
                }
            )

    return pd.DataFrame(all_stats_list)


def estimate_crosstalk_matrix(
    tile_data_all_cycles: np.ndarray,
    points: np.ndarray,
    channels: List[str] = ["G", "T", "A", "C"],
) -> pd.DataFrame:
    """
    Estimate spectral crosstalk matrix from first cycle data.

    Args:
        tile_data_all_cycles: Image data with shape (n_cycles, n_channels, H, W)
        points: Array of spot coordinates (N, 2) in (y, x) format
        channels: List of channel names

    Returns:
        DataFrame representing crosstalk matrix (channels x channels)
    """
    n_cycles, n_channels, H, W = tile_data_all_cycles.shape

    # Get valid point coordinates
    points_int = np.round(points).astype(int)
    points_valid = points_int[
        (points_int[:, 0] >= 0)
        & (points_int[:, 0] < H)
        & (points_int[:, 1] >= 0)
        & (points_int[:, 1] < W)
    ]
    points_y, points_x = points_valid[:, 0], points_valid[:, 1]
    read_ids = np.arange(len(points_valid))

    # Collect first cycle data
    plot_data_list = []
    for chan_idx, chan_name in enumerate(channels):
        channel_image = tile_data_all_cycles[0, chan_idx, :, :]
        signal_pixels = channel_image[points_y, points_x]
        plot_data_list.append(
            pd.DataFrame(
                {
                    "Intensity": signal_pixels,
                    "Channel": chan_name,
                    "read": read_ids,
                }
            )
        )

    if not plot_data_list:
        return None

    cycle1_signal_df = pd.concat(plot_data_list, ignore_index=True)
    if cycle1_signal_df.empty:
        return None

    cycle1_intensities_wide = cycle1_signal_df.pivot(
        index="read", columns="Channel", values="Intensity"
    )[channels]
    brightest_channel_indices = cycle1_intensities_wide.values.argmax(axis=1)
    brightest_channel_names = cycle1_intensities_wide.columns[brightest_channel_indices]
    cycle1_intensities_wide["brightest_channel"] = brightest_channel_names

    median_vectors = cycle1_intensities_wide.groupby("brightest_channel")[
        channels
    ].median()
    median_vectors = median_vectors.reindex(channels, fill_value=0)

    row_sums = median_vectors.sum(axis=1)
    crosstalk_matrix = median_vectors.div(row_sums, axis=0).fillna(0)

    return crosstalk_matrix


# --- Discrete Metric Calculation Functions ---


def calculate_snr(
    signal_mean: float, background_mean: float, background_std: float
) -> float:
    """
    Calculate Signal-to-Noise Ratio (SNR).

    SNR = (signal_mean - background_mean) / background_std

    Args:
        signal_mean: Mean signal intensity
        background_mean: Mean background intensity
        background_std: Standard deviation of background

    Returns:
        SNR value
    """
    return (signal_mean - background_mean) / (background_std + 1e-9)


def calculate_sbr(signal_mean: float, background_mean: float) -> float:
    """
    Calculate Signal-to-Background Ratio (SBR).

    SBR = signal_mean / background_mean

    Args:
        signal_mean: Mean signal intensity
        background_mean: Mean background intensity

    Returns:
        SBR value
    """
    return signal_mean / (background_mean + 1e-9)


def calculate_lld(
    background_std: float, signal_mean: float, background_mean: float
) -> float:
    """
    Calculate Lower Limit of Detection (LLD).

    LLD = 3 * background_std / (signal_mean - background_mean)
    Lower values indicate better detection capability.

    Args:
        background_std: Standard deviation of background
        signal_mean: Mean signal intensity
        background_mean: Mean background intensity

    Returns:
        LLD value
    """
    return (3 * background_std) / ((signal_mean - background_mean) + 1e-9)


def calculate_z_prime(
    signal_mean: float,
    signal_std: float,
    background_mean: float,
    background_std: float,
) -> float:
    """
    Calculate Z'-factor (Z-prime), a measure of assay quality.

    Z' = 1 - (3 * (signal_std + background_std)) / |signal_mean - background_mean|

    Values > 0.5 indicate excellent separation between signal and background.

    Args:
        signal_mean: Mean signal intensity
        signal_std: Standard deviation of signal
        background_mean: Mean background intensity
        background_std: Standard deviation of background

    Returns:
        Z'-factor value
    """
    return 1 - (3 * (signal_std + background_std)) / (
        np.abs(signal_mean - background_mean) + 1e-9
    )


def calculate_all_metrics(
    signal_mean: float,
    signal_std: float,
    background_mean: float,
    background_std: float,
) -> dict:
    """
    Calculate all imaging quality metrics from signal/background statistics.

    Args:
        signal_mean: Mean signal intensity
        signal_std: Standard deviation of signal
        background_mean: Mean background intensity
        background_std: Standard deviation of background

    Returns:
        Dictionary with keys: snr, sbr, lld, z_prime
    """
    return {
        "snr": calculate_snr(signal_mean, background_mean, background_std),
        "sbr": calculate_sbr(signal_mean, background_mean),
        "lld": calculate_lld(background_std, signal_mean, background_mean),
        "z_prime": calculate_z_prime(
            signal_mean, signal_std, background_mean, background_std
        ),
    }


def calculate_summary_metrics(
    summary_stats: pd.DataFrame,
    crosstalk_matrix: pd.DataFrame = None,
) -> dict:
    """
    Calculate summary metrics from signal/background stats.

    Args:
        summary_stats: DataFrame with signal/background statistics
        crosstalk_matrix: Optional crosstalk matrix

    Returns:
        Dictionary of summary metrics
    """
    signal_df = summary_stats[summary_stats["Type"] == "Signal"][
        ["Cycle", "Channel", "mean", "median_top_10pct", "std"]
    ].rename(columns={"mean": "signal_mean", "std": "signal_std"})
    background_df = summary_stats[summary_stats["Type"] == "Background"][
        ["Cycle", "Channel", "mean", "std"]
    ].rename(columns={"mean": "background_mean", "std": "background_std"})

    if signal_df.empty or background_df.empty:
        return {
            "snr_mean_snr": 0,
            "snr_mean_sbr": 0,
            "snr_mean_lld": 0,
            "snr_mean_z_prime": 0,
            "snr_mean_max_intensity": 0,
            "snr_median_top10pct_intensity": 0,
            "snr_mean_background_mean": 0,
            "snr_mean_background_std": 0,
            "snr_mean_bleedthrough": 0,
            "snr_snr_slope": 0.0,
            "snr_sbr_slope": 0.0,
        }

    merged_stats = pd.merge(signal_df, background_df, on=["Cycle", "Channel"])
    mean_max_intensity = np.round(merged_stats["signal_mean"].mean(), 1)
    median_top_10pct_intensity = np.round(merged_stats["median_top_10pct"].mean(), 1)
    mean_background_std = np.round(merged_stats["background_std"].mean(), 1)
    mean_background_mean = np.round(merged_stats["background_mean"].mean(), 1)

    # Use discrete metric functions
    merged_stats["snr"] = merged_stats.apply(
        lambda row: calculate_snr(
            row["signal_mean"], row["background_mean"], row["background_std"]
        ),
        axis=1,
    )
    merged_stats["sbr"] = merged_stats.apply(
        lambda row: calculate_sbr(row["signal_mean"], row["background_mean"]), axis=1
    )
    merged_stats["lld"] = merged_stats.apply(
        lambda row: calculate_lld(
            row["background_std"], row["signal_mean"], row["background_mean"]
        ),
        axis=1,
    )
    merged_stats["z_prime"] = merged_stats.apply(
        lambda row: calculate_z_prime(
            row["signal_mean"],
            row["signal_std"],
            row["background_mean"],
            row["background_std"],
        ),
        axis=1,
    )

    mean_snr = np.round(merged_stats["snr"].mean(), 1)
    mean_sbr = np.round(merged_stats["sbr"].mean(), 1)
    mean_lld = np.round(merged_stats["lld"].mean(), 3)
    mean_z_prime = np.round(merged_stats["z_prime"].mean(), 3)

    # Calculate slopes
    snr_per_cycle = merged_stats.groupby("Cycle")["snr"].mean()
    slope_snr, _, _, _, _ = (
        linregress(snr_per_cycle.index, snr_per_cycle.values)
        if len(snr_per_cycle) > 1
        else (0.0, 0, 0, 0, 0)
    )
    snr_slope = np.round(slope_snr, 2)

    sbr_per_cycle = merged_stats.groupby("Cycle")["sbr"].mean()
    slope_sbr, _, _, _, _ = (
        linregress(sbr_per_cycle.index, sbr_per_cycle.values)
        if len(sbr_per_cycle) > 1
        else (0.0, 0, 0, 0, 0)
    )
    sbr_slope = np.round(slope_sbr, 2)

    mean_bleedthrough = (
        1 - np.diag(crosstalk_matrix.values).mean()
        if crosstalk_matrix is not None
        else 0
    )

    return {
        "snr_mean_snr": mean_snr,
        "snr_mean_sbr": mean_sbr,
        "snr_mean_lld": mean_lld,
        "snr_mean_z_prime": mean_z_prime,
        "snr_mean_max_intensity": mean_max_intensity,
        "snr_median_top10pct_intensity": median_top_10pct_intensity,
        "snr_mean_background_mean": mean_background_mean,
        "snr_mean_background_std": mean_background_std,
        "snr_mean_bleedthrough": np.round(mean_bleedthrough, 2),
        "snr_snr_slope": snr_slope,
        "snr_sbr_slope": sbr_slope,
    }


def calculate_per_channel_metrics(
    summary_stats: pd.DataFrame, crosstalk_matrix: pd.DataFrame = None
) -> dict:
    """Calculate per-channel summary metrics averaged across all cycles.

    Args:
        summary_stats: DataFrame with columns [Cycle, Channel, Type, mean, std, count]
        crosstalk_matrix: Optional crosstalk matrix

    Returns:
        Dictionary mapping channel names to their metrics
    """
    channels = ["G", "T", "A", "C"]
    per_channel_metrics = {}

    signal_df = summary_stats[summary_stats["Type"] == "Signal"][
        ["Cycle", "Channel", "mean", "median_top_10pct", "std"]
    ].rename(columns={"mean": "signal_mean", "std": "signal_std"})
    background_df = summary_stats[summary_stats["Type"] == "Background"][
        ["Cycle", "Channel", "mean", "std"]
    ].rename(columns={"mean": "background_mean", "std": "background_std"})

    if signal_df.empty or background_df.empty:
        return {ch: {} for ch in channels}

    merged_stats = pd.merge(signal_df, background_df, on=["Cycle", "Channel"])

    # Calculate metrics for each channel using discrete metric functions
    merged_stats["snr"] = merged_stats.apply(
        lambda row: calculate_snr(
            row["signal_mean"], row["background_mean"], row["background_std"]
        ),
        axis=1,
    )
    merged_stats["sbr"] = merged_stats.apply(
        lambda row: calculate_sbr(row["signal_mean"], row["background_mean"]), axis=1
    )
    merged_stats["lld"] = merged_stats.apply(
        lambda row: calculate_lld(
            row["background_std"], row["signal_mean"], row["background_mean"]
        ),
        axis=1,
    )
    merged_stats["z_prime"] = merged_stats.apply(
        lambda row: calculate_z_prime(
            row["signal_mean"],
            row["signal_std"],
            row["background_mean"],
            row["background_std"],
        ),
        axis=1,
    )

    for channel in channels:
        channel_data = merged_stats[merged_stats["Channel"] == channel]
        if not channel_data.empty:
            per_channel_metrics[channel] = {
                "snr": np.round(channel_data["snr"].mean(), 2),
                "sbr": np.round(channel_data["sbr"].mean(), 2),
                "lld": np.round(channel_data["lld"].mean(), 3),
                "z_prime": np.round(channel_data["z_prime"].mean(), 3),
                "max_intensity": np.round(channel_data["signal_mean"].mean(), 1),
                "median_top_10pct": np.round(
                    channel_data["median_top_10pct"].mean(), 1
                ),
                "background_mean": np.round(channel_data["background_mean"].mean(), 1),
                "background_std": np.round(channel_data["background_std"].mean(), 1),
            }
        else:
            per_channel_metrics[channel] = {
                "snr": 0,
                "sbr": 0,
                "lld": 0,
                "z_prime": 0,
                "max_intensity": 0,
                "median_top_10pct": 0,
                "background_mean": 0,
                "background_std": 0,
            }

    return per_channel_metrics


# --- Plotting Helper Functions ---


def _annotate_points(ax, x_values, y_values, fontsize: int):
    """Add value annotations to plot points."""
    for x_val, y_val in zip(x_values, y_values):
        try:
            if np.isfinite(y_val):
                ax.annotate(
                    f"{float(y_val):.1f}",
                    (x_val, y_val),
                    textcoords="offset points",
                    xytext=(0, 4),
                    ha="center",
                    fontsize=fontsize,
                )
        except Exception:
            pass


def _plot_metric_by_channel(
    ax,
    df: pd.DataFrame,
    channel_name: str,
    channel_color: str,
    x_col: str,
    y_col: str,
    yerr_col: str = None,
    marker: str = "o",
    linestyle: str = "-",
    capsize: int = 3,
    annotate: bool = True,
    annotation_fontsize: int = 10,
    label: str = None,
):
    """Plot a metric for a single channel."""
    chan_df = df[df["Channel"] == channel_name]
    if chan_df.empty:
        return
    x_vals = chan_df[x_col]
    y_vals = chan_df[y_col]
    if yerr_col is not None and yerr_col in chan_df.columns:
        ax.errorbar(
            x=x_vals,
            y=y_vals,
            yerr=chan_df[yerr_col],
            marker=marker,
            capsize=capsize,
            linestyle=linestyle,
            color=channel_color,
            label=label or channel_name,
        )
    else:
        ax.plot(
            x_vals,
            y_vals,
            marker=marker,
            linestyle=linestyle,
            color=channel_color,
            label=label or channel_name,
        )
    if annotate:
        _annotate_points(ax, x_vals, y_vals, annotation_fontsize)


# --- Fast SNR Heatmap Functions ---


def sample_spots_from_tile(
    spots: np.ndarray, tile_bounds: Tuple[int, int, int, int], max_spots: int = 100
) -> np.ndarray:
    """
    Sample spots within a tile region.

    Args:
        spots: Array of spot coordinates (N, 2) in (y, x) format for entire well
        tile_bounds: Tuple of (row_start, row_stop, col_start, col_stop)
        max_spots: Maximum number of spots to sample

    Returns:
        Array of sampled spot coordinates relative to tile origin
    """
    row_start, row_stop, col_start, col_stop = tile_bounds

    # Filter spots to tile
    spots_in_tile = spots[
        (spots[:, 0] >= row_start)
        & (spots[:, 0] < row_stop)
        & (spots[:, 1] >= col_start)
        & (spots[:, 1] < col_stop)
    ]

    if len(spots_in_tile) == 0:
        return np.array([])

    # Sample if too many spots
    if len(spots_in_tile) > max_spots:
        indices = np.random.choice(len(spots_in_tile), max_spots, replace=False)
        spots_sampled = spots_in_tile[indices]
    else:
        spots_sampled = spots_in_tile

    # Convert to tile-relative coordinates
    spots_relative = spots_sampled - np.array([row_start, col_start])

    return spots_relative


def calculate_tile_snr_fast(
    tile_data: np.ndarray,
    spots_relative: np.ndarray,
    signal_radius: int = 5,
    background_region_size: int = 100,
) -> np.ndarray:
    """
    Calculate SNR for a tile using sampled spots.

    Args:
        tile_data: Image data with shape (n_cycles, n_channels, H, W)
        spots_relative: Spot coordinates relative to tile origin (N, 2) in (y, x)
        signal_radius: Radius for signal mask
        background_region_size: Size of region to sample for background

    Returns:
        Array of SNR values with shape (n_cycles, n_channels)
    """
    n_cycles, n_channels, H, W = tile_data.shape

    if len(spots_relative) == 0:
        return np.zeros((n_cycles, n_channels))

    # Apply max filter to match base calling approach
    tile_data_max = _iss_max_filter(tile_data, width=10)

    # Create background mask (True = background)
    background_mask = create_background_mask(spots_relative, (H, W), signal_radius)

    # Get valid spot coordinates
    spots_int = np.round(spots_relative).astype(int)
    valid_mask = (
        (spots_int[:, 0] >= 0)
        & (spots_int[:, 0] < H)
        & (spots_int[:, 1] >= 0)
        & (spots_int[:, 1] < W)
    )
    spots_valid = spots_int[valid_mask]

    if len(spots_valid) == 0:
        return np.zeros((n_cycles, n_channels))

    points_y, points_x = spots_valid[:, 0], spots_valid[:, 1]

    # Calculate SNR for each cycle and channel
    snr_values = np.zeros((n_cycles, n_channels))

    for cycle_idx in range(n_cycles):
        for chan_idx in range(n_channels):
            # Signal: max-filtered values at spot locations
            signal_image = tile_data_max[cycle_idx, chan_idx, :, :]
            signal_pixels = signal_image[points_y, points_x]

            # Background: original data in background regions
            background_image = tile_data[cycle_idx, chan_idx, :, :]
            background_pixels = background_image[background_mask]

            # Downsample background if too large
            if len(background_pixels) > 50000:
                background_pixels = np.random.choice(
                    background_pixels, 50000, replace=False
                )

            if len(signal_pixels) == 0 or len(background_pixels) == 0:
                snr_values[cycle_idx, chan_idx] = 0
                continue

            # SNR = (mean_signal - mean_background) / std_background
            signal_mean = np.mean(signal_pixels)
            background_mean = np.mean(background_pixels)
            background_std = np.std(background_pixels)

            if background_std > 0:
                snr = (signal_mean - background_mean) / background_std
                snr_values[cycle_idx, chan_idx] = max(0, snr)  # Clip negative SNR
            else:
                snr_values[cycle_idx, chan_idx] = 0

    return snr_values


def calculate_tile_comprehensive_stats(
    tile_data: np.ndarray,
    spots_relative: np.ndarray,
    signal_radius: int = 5,
) -> pd.DataFrame:
    """
    Calculate comprehensive signal and background statistics for a tile.

    Returns DataFrame with same format as calculate_signal_background_stats:
        Columns: Cycle, Channel, Type, mean, median, std, count, median_top_10pct

    Args:
        tile_data: Shape (n_cycles, n_channels, H, W)
        spots_relative: Spot coordinates relative to tile
        signal_radius: Radius for signal region

    Returns:
        DataFrame with statistics
    """
    channels = ["G", "T", "A", "C"]
    n_cycles, n_channels, H, W = tile_data.shape

    if len(spots_relative) == 0:
        # Return empty stats
        return pd.DataFrame(
            columns=[
                "Cycle",
                "Channel",
                "Type",
                "mean",
                "median",
                "std",
                "count",
                "median_top_10pct",
            ]
        )

    # Apply max filter for signal
    tile_data_max = _iss_max_filter(tile_data, width=10)

    # Create background mask
    background_mask = create_background_mask(spots_relative, (H, W), signal_radius)

    # Get valid spot coordinates
    spots_int = np.round(spots_relative).astype(int)
    valid_mask = (
        (spots_int[:, 0] >= 0)
        & (spots_int[:, 0] < H)
        & (spots_int[:, 1] >= 0)
        & (spots_int[:, 1] < W)
    )
    spots_valid = spots_int[valid_mask]

    if len(spots_valid) == 0:
        return pd.DataFrame(
            columns=[
                "Cycle",
                "Channel",
                "Type",
                "mean",
                "median",
                "std",
                "count",
                "median_top_10pct",
            ]
        )

    points_y, points_x = spots_valid[:, 0], spots_valid[:, 1]

    # Collect stats
    rows = []

    for cycle_idx in range(n_cycles):
        for chan_idx in range(n_channels):
            # Signal: max-filtered values at spot locations
            signal_image = tile_data_max[cycle_idx, chan_idx, :, :]
            signal_pixels = signal_image[points_y, points_x]

            # Background: original data in background regions
            background_image = tile_data[cycle_idx, chan_idx, :, :]
            background_pixels = background_image[background_mask]

            # Downsample background if too large
            if len(background_pixels) > 50000:
                background_pixels = np.random.choice(
                    background_pixels, 50000, replace=False
                )

            # Signal stats
            if len(signal_pixels) > 0:
                top_10pct_count = max(1, len(signal_pixels) // 10)
                top_10pct_values = np.partition(signal_pixels, -top_10pct_count)[
                    -top_10pct_count:
                ]

                rows.append(
                    {
                        "Cycle": cycle_idx + 1,
                        "Channel": channels[chan_idx],
                        "Type": "Signal",
                        "mean": np.mean(signal_pixels),
                        "median": np.median(signal_pixels),
                        "std": np.std(signal_pixels),
                        "count": len(signal_pixels),
                        "median_top_10pct": np.median(top_10pct_values),
                    }
                )

            # Background stats
            if len(background_pixels) > 0:
                rows.append(
                    {
                        "Cycle": cycle_idx + 1,
                        "Channel": channels[chan_idx],
                        "Type": "Background",
                        "mean": np.mean(background_pixels),
                        "median": np.median(background_pixels),
                        "std": np.std(background_pixels),
                        "count": len(background_pixels),
                        "median_top_10pct": np.nan,  # Not applicable for background
                    }
                )

    return pd.DataFrame(rows)


def _process_single_tile(
    tile_idx: int,
    tile_bounds: Tuple[int, int, int, int],
    spots: np.ndarray,
    stitch_path: str,
    well: str,
    channel_start: int,
    spots_per_tile: int,
    signal_radius: int,
    tile_sample_fraction: float,
) -> Tuple[int, np.ndarray, int, pd.DataFrame, pd.DataFrame]:
    """
    Process a single tile to calculate SNR values and comprehensive stats.

    This function is designed to be called in parallel by joblib.

    Args:
        tile_idx: Index of the tile being processed
        tile_bounds: (row_start, row_stop, col_start, col_stop)
        spots: Array of all spot coordinates for the well
        stitch_path: Path to the stitched image dataset
        well: Well identifier
        channel_start: Starting channel index (to skip DAPI if present)
        spots_per_tile: Maximum spots to sample
        signal_radius: Radius for signal region mask
        tile_sample_fraction: Fraction of tile to load

    Returns:
        Tuple of (tile_idx, snr_values, n_spots_used, tile_stats, crosstalk_matrix)
    """
    from iohub.ngff import open_ome_zarr
    import dask.array as da

    row_start, row_stop, col_start, col_stop = tile_bounds

    # Calculate center region to load
    tile_height = row_stop - row_start
    tile_width = col_stop - col_start

    sample_height = max(100, int(tile_height * tile_sample_fraction))
    sample_width = max(100, int(tile_width * tile_sample_fraction))

    row_margin = (tile_height - sample_height) // 2
    col_margin = (tile_width - sample_width) // 2

    sample_row_start = row_start + row_margin
    sample_row_stop = sample_row_start + sample_height
    sample_col_start = col_start + col_margin
    sample_col_stop = sample_col_start + sample_width

    # Filter spots to the sample region
    spots_in_sample = spots[
        (spots[:, 0] >= sample_row_start)
        & (spots[:, 0] < sample_row_stop)
        & (spots[:, 1] >= sample_col_start)
        & (spots[:, 1] < sample_col_stop)
    ]

    # Exclude tiles with insufficient spots for reliable statistics
    MIN_SPOTS_THRESHOLD = 20

    if len(spots_in_sample) == 0 or len(spots_in_sample) < MIN_SPOTS_THRESHOLD:
        # Return NaN for tiles with no spots or too few spots (excluded from downstream analysis)
        # We need to know the shape, so load once to get dimensions
        stitch_ds = open_ome_zarr(Path(stitch_path) / well, layout="fov", mode="r")
        fov_dask = da.from_array(stitch_ds.data)
        n_cycles = fov_dask.shape[0]
        n_channels_used = 4
        empty_stats = pd.DataFrame(
            columns=[
                "Cycle",
                "Channel",
                "Type",
                "mean",
                "median",
                "std",
                "count",
                "median_top_10pct",
            ]
        )
        empty_crosstalk = pd.DataFrame()  # Empty crosstalk matrix
        return (
            tile_idx,
            np.full((n_cycles, n_channels_used), np.nan),
            0,
            empty_stats,
            empty_crosstalk,
        )

    # Sample spots if too many
    if len(spots_in_sample) > spots_per_tile:
        indices = np.random.choice(len(spots_in_sample), spots_per_tile, replace=False)
        spots_sampled = spots_in_sample[indices]
    else:
        spots_sampled = spots_in_sample

    n_spots_used = len(spots_sampled)

    # Convert to sample-region-relative coordinates
    spots_relative = spots_sampled - np.array([sample_row_start, sample_col_start])

    # Load the data for this tile
    stitch_ds = open_ome_zarr(Path(stitch_path) / well, layout="fov", mode="r")
    fov_dask = da.from_array(stitch_ds.data)

    tile_data = (
        fov_dask[
            :,
            channel_start:,
            0,
            sample_row_start:sample_row_stop,
            sample_col_start:sample_col_stop,
        ]
        .compute()
        .astype("float32")
    )

    # Calculate SNR
    snr_values = calculate_tile_snr_fast(
        tile_data, spots_relative, signal_radius=signal_radius
    )

    # Calculate comprehensive stats
    tile_stats = calculate_tile_comprehensive_stats(
        tile_data, spots_relative, signal_radius=signal_radius
    )

    # Calculate crosstalk matrix from first cycle
    crosstalk_matrix = estimate_crosstalk_matrix(
        tile_data, spots_relative, channels=["G", "T", "A", "C"]
    )

    return tile_idx, snr_values, n_spots_used, tile_stats, crosstalk_matrix


def process_well_snr_tiles(
    dataset: "OpsDataset",
    well: str,
    grid_size: int = 30,
    spots_per_tile: int = 100,
    signal_radius: int = 5,
    tile_sample_fraction: float = 0.1,
    well_inner_fraction: float = 0.75,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Process a well to calculate SNR, comprehensive stats, and crosstalk for each tile using parallel processing.

    For speed, loads only a central fraction of each tile (default 10%).
    Tiles only the inner region of the well to avoid edges with no spots.
    Uses multiprocessing to process tiles in parallel (worker count auto-detected by resource_manager).

    Args:
        dataset: OpsDataset object
        well: Well identifier
        grid_size: Number of tiles per dimension (e.g., 30 for 30x30 grid)
        spots_per_tile: Maximum spots to sample per tile
        signal_radius: Radius for signal region mask
        tile_sample_fraction: Fraction of tile dimensions to load (0.1 = center 10% on each side)
        well_inner_fraction: Fraction of well dimensions to tile (0.75 = inner 75%, avoiding edges)

    Returns:
        Tuple of:
            - Array of SNR values with shape (n_tiles, n_cycles, n_channels)
            - DataFrame with comprehensive stats (columns: tile_idx, Cycle, Channel, Type, mean, median, std, count, median_top_10pct)
            - DataFrame with aggregated crosstalk matrix (channels x channels)
    """
    from iohub.ngff import open_ome_zarr
    import dask.array as da

    # Load data at position level (skip plate metadata parsing)
    stitch_ds = open_ome_zarr(resolve_iss_registered_store(dataset) / well, layout="fov", mode="r")
    fov_dask = da.from_array(stitch_ds.data)
    full_shape = fov_dask.shape[-2:]

    # Load spots
    spots_file = dataset.append_well("spots", well)
    if not spots_file.exists():
        raise FileNotFoundError(f"Spots file not found: {spots_file}")
    spots = np.load(spots_file)

    # Calculate inner region (e.g., center 75% of well)
    full_height, full_width = full_shape
    inner_height = int(full_height * well_inner_fraction)
    inner_width = int(full_width * well_inner_fraction)

    # Center the inner region
    row_margin = (full_height - inner_height) // 2
    col_margin = (full_width - inner_width) // 2

    inner_row_start = row_margin
    inner_row_stop = row_margin + inner_height
    inner_col_start = col_margin
    inner_col_stop = col_margin + inner_width

    inner_shape = (inner_height, inner_width)

    # Get tile list for the INNER region only
    from ops_utils.io.tiling import split_into_tiles

    tile_list, _ = split_into_tiles(inner_shape, grid_size, 0)

    # Offset tiles to global coordinates
    tile_list_global = [
        (
            row_start + inner_row_start,
            row_stop + inner_row_start,
            col_start + inner_col_start,
            col_stop + inner_col_start,
        )
        for row_start, row_stop, col_start, col_stop in tile_list
    ]

    # Detect number of channels and skip DAPI if present
    num_channels = fov_dask.shape[1]
    channel_start = 1 if num_channels == 5 else 0
    n_cycles = fov_dask.shape[0]
    n_channels_used = 4  # Always GTAC

    # Determine optimal number of workers
    n_jobs = get_optimal_workers(use_gpu=False, verbose=False)

    print(f"\n{'='*60}")
    print(f"Processing well {well} with {n_jobs} parallel workers")
    print(f"{'='*60}\n")

    # Pre-allocate results
    snr_results = np.zeros((len(tile_list_global), n_cycles, n_channels_used))

    # Calculate approximate tile dimensions for info
    if tile_list_global:
        sample_tile = tile_list_global[0]
        tile_height = sample_tile[1] - sample_tile[0]
        tile_width = sample_tile[3] - sample_tile[2]
        sample_height = int(tile_height * tile_sample_fraction)
        sample_width = int(tile_width * tile_sample_fraction)
        print(f"Processing {len(tile_list_global)} tiles for well {well}...")
        print(f"  Full well: {full_shape[0]}x{full_shape[1]} pixels")
        print(
            f"  Tiling inner {well_inner_fraction*100:.0f}%: {inner_height}x{inner_width} pixels "
            f"(excluding {row_margin} px margins)"
        )
        print(
            f"  Loading center {tile_sample_fraction*100:.0f}% of each tile "
            f"(~{sample_height}x{sample_width} px vs full {tile_height}x{tile_width} px)"
        )

    # Get path to stitched dataset for workers
    stitch_path = str(resolve_iss_registered_store(dataset))

    # Process tiles in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_process_single_tile)(
            tile_idx,
            tile_bounds,
            spots,
            stitch_path,
            well,
            channel_start,
            spots_per_tile,
            signal_radius,
            tile_sample_fraction,
        )
        for tile_idx, tile_bounds in enumerate(tile_list_global)
    )

    # Collect results
    spot_counts = []
    tiles_with_no_spots = 0
    all_tile_stats = []
    all_crosstalk_matrices = []

    for tile_idx, snr_values, n_spots_used, tile_stats, crosstalk_matrix in results:
        snr_results[tile_idx] = snr_values
        spot_counts.append(n_spots_used)
        if n_spots_used == 0:
            tiles_with_no_spots += 1

        # Collect tile stats if not empty
        if not tile_stats.empty:
            tile_stats = tile_stats.copy()
            tile_stats["tile_idx"] = tile_idx
            all_tile_stats.append(tile_stats)

        # Collect crosstalk matrices if not empty
        if crosstalk_matrix is not None and not crosstalk_matrix.empty:
            all_crosstalk_matrices.append(crosstalk_matrix)

    # Print spot count statistics
    if spot_counts:
        spot_counts_arr = np.array(spot_counts)
        spots_with_data = spot_counts_arr[spot_counts_arr > 0]

        print(f"Completed processing {len(tile_list_global)} tiles for well {well}")
        print(f"  Spot statistics:")
        print(
            f"    Tiles with spots: {len(spots_with_data)}/{len(tile_list_global)} ({len(spots_with_data)/len(tile_list_global)*100:.1f}%)"
        )
        print(f"    Tiles with no spots: {tiles_with_no_spots}")

        if len(spots_with_data) > 0:
            print(
                f"    Spots per tile (with spots): mean={spots_with_data.mean():.1f}, "
                f"median={np.median(spots_with_data):.0f}, "
                f"min={spots_with_data.min()}, max={spots_with_data.max()}"
            )

            # Report tiles excluded due to insufficient spots
            tiles_with_few_spots = np.sum(spots_with_data < 20)
            if tiles_with_few_spots > 0:
                pct_excluded = tiles_with_few_spots / len(tile_list_global) * 100
                print(
                    f"    ⚠️  EXCLUDED from analysis: {tiles_with_few_spots} tiles with <20 spots ({pct_excluded:.1f}% of total)"
                )
                print(f"       Reason: Insufficient spots for reliable SNR statistics")
    else:
        print(f"Completed processing {len(tile_list_global)} tiles for well {well}")
        print(f"  ⚠️  Warning: No spots found in any tiles!")

    # Aggregate tile stats into single DataFrame
    if all_tile_stats:
        comprehensive_stats = pd.concat(all_tile_stats, ignore_index=True)
    else:
        comprehensive_stats = pd.DataFrame(
            columns=[
                "Cycle",
                "Channel",
                "Type",
                "mean",
                "median",
                "std",
                "count",
                "median_top_10pct",
                "tile_idx",
            ]
        )

    # Aggregate crosstalk matrices across tiles (median aggregation)
    if all_crosstalk_matrices:
        # Stack all matrices and take median across tiles
        channels = ["G", "T", "A", "C"]
        stacked_values = {src: {dst: [] for dst in channels} for src in channels}

        for matrix in all_crosstalk_matrices:
            for src in channels:
                if src in matrix.index:
                    for dst in channels:
                        if dst in matrix.columns:
                            stacked_values[src][dst].append(matrix.loc[src, dst])

        # Calculate median for each cell
        aggregated_crosstalk = pd.DataFrame(index=channels, columns=channels)
        for src in channels:
            for dst in channels:
                values = stacked_values[src][dst]
                if values:
                    aggregated_crosstalk.loc[src, dst] = np.median(values)
                else:
                    aggregated_crosstalk.loc[src, dst] = 0.0

        aggregated_crosstalk = aggregated_crosstalk.astype(float)
        print(
            f"\n  Aggregated crosstalk matrix from {len(all_crosstalk_matrices)} tiles"
        )
    else:
        # Return empty DataFrame if no crosstalk matrices
        channels = ["G", "T", "A", "C"]
        aggregated_crosstalk = pd.DataFrame(index=channels, columns=channels, data=0.0)
        print(f"\n  No crosstalk matrices calculated (no tiles with sufficient spots)")

    return snr_results, comprehensive_stats, aggregated_crosstalk
