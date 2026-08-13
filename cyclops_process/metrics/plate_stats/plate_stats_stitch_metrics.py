"""
ISS Downstream Metrics Helper Functions.

This module provides helper functions for computing QC metrics from downstream
pipeline outputs (stitching, registration, etc.) for inclusion in plate_stats.csv.
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from cyclops_utils.data.filesystem import parse_well


@dataclass
class StitchConfidenceStats:
    """Statistics from stitch confidence analysis."""
    mean: float
    median: float
    std: float
    min: float
    max: float
    num_edges: int


def load_stitch_config(config_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load a stitch config YAML file.

    Args:
        config_path: Path to the stitch settings YAML file

    Returns:
        Parsed YAML dict or None if file doesn't exist or is invalid
    """
    if not config_path.exists():
        return None

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config if isinstance(config, dict) else None
    except Exception:
        return None


def compute_stitch_confidence_stats(
    config_path: Path,
    well: Optional[str] = None,
) -> Optional[StitchConfidenceStats]:
    """
    Compute stitch confidence statistics from a stitch config file.

    The stitch config contains a 'confidence' key with per-well edge confidence
    values. Each edge has format: {edge_id: [[pos1], [pos2], confidence_value]}

    Args:
        config_path: Path to the stitch settings YAML file
        well: Optional well to filter (e.g., "A/1/0"). If None, computes across all wells.

    Returns:
        StitchConfidenceStats or None if no confidence data available
    """
    config = load_stitch_config(config_path)
    if config is None or "confidence" not in config:
        return None

    confidence_dict = config["confidence"]
    if not confidence_dict:
        return None

    # Collect all confidence values
    all_confidences = []

    # Normalize requested well to config key form "row/col" (row-agnostic)
    well_normalized = None
    if well is not None:
        row, col = parse_well(well)
        well_normalized = f"{row}/{col}"

    for well_key, well_conf in confidence_dict.items():
        # If specific well requested, filter against normalized "row/col" key
        if well_normalized is not None and well_key != well_normalized:
            continue

        if not isinstance(well_conf, dict):
            continue

        for edge_id, edge_data in well_conf.items():
            try:
                # Edge data format: [[pos1_y, pos1_x], [pos2_y, pos2_x], confidence]
                if isinstance(edge_data, (list, tuple)) and len(edge_data) >= 3:
                    conf_value = float(edge_data[2])
                    all_confidences.append(conf_value)
            except (IndexError, TypeError, ValueError):
                continue

    if not all_confidences:
        return None

    return StitchConfidenceStats(
        mean=float(np.round(np.mean(all_confidences), 4)),
        median=float(np.round(np.median(all_confidences), 4)),
        std=float(np.round(np.std(all_confidences), 4)),
        min=float(np.round(np.min(all_confidences), 4)),
        max=float(np.round(np.max(all_confidences), 4)),
        num_edges=len(all_confidences),
    )


def get_stitch_confidence_for_experiment(
    dataset,
    well: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get stitch confidence statistics for all processes (ISS, track, pheno).

    This is the main entry point for metrics.py to get stitch confidence stats.

    Args:
        dataset: OpsDataset instance
        well: Optional well to filter. If None, computes across all wells.

    Returns:
        Dictionary with stitch confidence stats for each process, ready to add to plate_stats
    """
    result = {
        # ISS stitch confidence
        "iss_stitch_conf_mean": None,
        "iss_stitch_conf_median": None,
        "iss_stitch_conf_std": None,
        "iss_stitch_conf_min": None,
        "iss_stitch_conf_max": None,
        # Track (5x) stitch confidence
        "track_stitch_conf_mean": None,
        "track_stitch_conf_median": None,
        "track_stitch_conf_std": None,
        "track_stitch_conf_min": None,
        "track_stitch_conf_max": None,
        # Pheno (20x) stitch confidence
        "pheno_stitch_conf_mean": None,
        "pheno_stitch_conf_median": None,
        "pheno_stitch_conf_std": None,
        "pheno_stitch_conf_min": None,
        "pheno_stitch_conf_max": None,
    }

    # ISS stitch confidence
    iss_config = dataset.config_paths.get("iss_stitch")
    if iss_config:
        stats = compute_stitch_confidence_stats(Path(iss_config), well)
        if stats:
            result["iss_stitch_conf_mean"] = stats.mean
            result["iss_stitch_conf_median"] = stats.median
            result["iss_stitch_conf_std"] = stats.std
            result["iss_stitch_conf_min"] = stats.min
            result["iss_stitch_conf_max"] = stats.max

    # Track (5x) stitch confidence
    track_config = dataset.config_paths.get("lc_5x_stitch")
    if track_config:
        stats = compute_stitch_confidence_stats(Path(track_config), well)
        if stats:
            result["track_stitch_conf_mean"] = stats.mean
            result["track_stitch_conf_median"] = stats.median
            result["track_stitch_conf_std"] = stats.std
            result["track_stitch_conf_min"] = stats.min
            result["track_stitch_conf_max"] = stats.max

    # Pheno (20x) stitch confidence
    pheno_config = dataset.config_paths.get("lc_20x_stitch")
    if pheno_config:
        stats = compute_stitch_confidence_stats(Path(pheno_config), well)
        if stats:
            result["pheno_stitch_conf_mean"] = stats.mean
            result["pheno_stitch_conf_median"] = stats.median
            result["pheno_stitch_conf_std"] = stats.std
            result["pheno_stitch_conf_min"] = stats.min
            result["pheno_stitch_conf_max"] = stats.max

    return result


def plot_stitch_confidence_heatmap(
    config_path: Path,
    output_path: Path,
    process_name: str,
    experiment: str,
) -> bool:
    """
    Generate and save a stitch confidence heatmap for a single process.

    Args:
        config_path: Path to the stitch settings YAML file
        output_path: Path to save the output PNG file
        process_name: Name of the process (e.g., "ISS", "Track", "Pheno")
        experiment: Experiment name for the plot title

    Returns:
        True if plot was generated successfully, False otherwise
    """
    config = load_stitch_config(config_path)
    if config is None or "confidence" not in config:
        print(f"[stitch_heatmap] No confidence data found in {config_path}")
        return False

    confidence_dict = config["confidence"]
    if not confidence_dict:
        print(f"[stitch_heatmap] Confidence dictionary is empty for {process_name}")
        return False

    num_wells = len(confidence_dict.keys())
    print(f"[stitch_heatmap] Creating {process_name} plot with {num_wells} wells")

    fig, ax = plt.subplots(1, num_wells, figsize=(6 * num_wells + 2, 6))
    if num_wells == 1:
        ax = [ax]

    cmap = cm.magma

    for i, well_key in enumerate(confidence_dict.keys()):
        well_conf = confidence_dict[well_key]
        print(f"[stitch_heatmap] Processing well {well_key} with {len(well_conf)} edges")

        ax[i].set_facecolor("black")
        ax[i].set_xticks([])
        ax[i].set_yticks([])

        for edge_id, edge_data in well_conf.items():
            try:
                ax[i].plot(
                    [edge_data[0][1], edge_data[1][1]],
                    [edge_data[0][0], edge_data[1][0]],
                    color=cmap(float(edge_data[2])),
                    linewidth=8,
                )
            except (IndexError, TypeError, ValueError) as e:
                print(f"[stitch_heatmap WARNING] Failed to plot edge {edge_id}: {e}")
                continue

        ax[i].set_title(well_key)

    sm = cm.ScalarMappable(cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Stitching Confidence")
    fig.suptitle(f"Stitching Confidence - {process_name}\n{experiment}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[stitch_heatmap] Saved {process_name} heatmap to {output_path}")
    return True


def generate_all_stitch_confidence_heatmaps(
    dataset,
    experiment: str,
    force: bool = False,
) -> Dict[str, bool]:
    """
    Generate stitch confidence heatmaps for all processes (ISS, Track, Pheno).

    Heatmaps are saved to the stitch/ subdirectory under results_iss.
    Stats are cached to stitch_confidence_stats.csv for faster re-runs.

    Args:
        dataset: OpsDataset instance
        experiment: Experiment name for plot titles
        force: If True, regenerate even if cache exists. Default False.

    Returns:
        Dictionary with process names as keys and success status as values
    """
    results = {}

    # Get the stitch output directory
    stitch_dir = dataset.results_iss / "stitch"
    stitch_dir.mkdir(parents=True, exist_ok=True)

    # Cache file path (use CSV as the skip indicator)
    cache_path = stitch_dir / "stitch_confidence_stats.csv"

    # Check if cache exists and skip if not forcing
    if cache_path.exists() and not force:
        print(f"[stitch_heatmap] Stitch confidence stats already cached at: {cache_path}")
        print("[stitch_heatmap] Skipping. Use --force or force=True to regenerate.")
        return {"ISS": True, "Track (5x)": True, "Pheno (20x)": True}

    # Process configurations: (config_key, output_filename, display_name)
    processes = [
        ("iss_stitch", "iss_stitch_confidence.png", "ISS"),
        ("lc_5x_stitch", "track_stitch_confidence.png", "Track (5x)"),
        ("lc_20x_stitch", "pheno_stitch_confidence.png", "Pheno (20x)"),
    ]

    # Collect stats for caching
    all_stats = {}

    for config_key, output_filename, display_name in processes:
        config_path = dataset.config_paths.get(config_key)
        if config_path is None:
            print(f"[stitch_heatmap] Config path not found for {config_key}")
            results[display_name] = False
            continue

        config_path = Path(config_path)
        if not config_path.exists():
            print(f"[stitch_heatmap] Config file does not exist: {config_path}")
            results[display_name] = False
            continue

        output_path = stitch_dir / output_filename
        success = plot_stitch_confidence_heatmap(
            config_path=config_path,
            output_path=output_path,
            process_name=display_name,
            experiment=experiment,
        )
        results[display_name] = success

        # Collect stats for this process
        if success:
            stats = compute_stitch_confidence_stats(config_path)
            if stats:
                all_stats[display_name] = {
                    "mean": stats.mean,
                    "median": stats.median,
                    "std": stats.std,
                    "min": stats.min,
                    "max": stats.max,
                    "num_edges": stats.num_edges,
                }

    # Save stats to cache CSV
    if all_stats:
        try:
            stats_df = pd.DataFrame.from_dict(all_stats, orient="index")
            stats_df.to_csv(cache_path)
            print(f"[stitch_heatmap] Saved stitch confidence stats to: {cache_path}")
        except Exception as e:
            print(f"[stitch_heatmap] Failed to save stats cache: {e}")

    return results
