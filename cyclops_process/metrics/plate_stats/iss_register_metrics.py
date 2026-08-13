"""
ISS Registration Metrics Helper Functions.

This module provides helper functions for computing QC metrics from registration
outputs (ISS cycle registration and auto-registration) for inclusion in plate_stats.csv.

Data Sources:
1. ISS Cycle Registration: 1-preprocess/in_situ_sequencing/register/metrics/registration_metrics_{row}{col}.csv
   - Per-round drift and alignment metrics (inlier ratio, residuals)

2. Auto-Registration: 2-tracking/auto_overlays/{row}{col}_{type}_to_{target}/auto_register_metrics.csv
   - ISS->Track and Pheno->Track overlap and inlier metrics per well

3. ISS Cycle Drift: 1-preprocess/in_situ_sequencing/register/transforms/{row}{col}/
   - Per-round and cumulative drift from affine transforms
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from cyclops_process.utils.batch.batch_collect_drift_trajectories import (
    collect_drift_data_for_well,
)
from ops_utils.data.filesystem import parse_well


@dataclass
class ISSCycleRegisterStats:
    """Statistics from ISS cycle registration."""
    mean_inlier_ratio: float
    min_inlier_ratio: float
    mean_residual: float
    max_residual: float
    n_rounds: int


@dataclass
class DriftStats:
    """Statistics from ISS cycle drift analysis."""
    mean_per_round_drift: float  # Mean drift magnitude per round
    max_per_round_drift: float   # Max drift magnitude in any single round
    cumulative_drift: float      # Total cumulative drift at final round
    n_rounds: int
    per_round_drifts: Dict[int, float]  # Individual drift per round (round_num -> magnitude)


@dataclass
class AutoRegisterStats:
    """Statistics from auto-registration (ISS->Track or Pheno->Track)."""
    overlap_percent: float
    inlier_ratio: float


def load_iss_cycle_metrics(
    dataset,
    well,
) -> Optional[pd.DataFrame]:
    """
    Load ISS cycle registration metrics CSV for a well.

    Args:
        dataset: OpsDataset instance
        well: Well identifier (1, "A1", "B2", "A/1/0")

    Returns:
        DataFrame with registration metrics or None if not found
    """
    row, col = parse_well(well)  # row-agnostic well token
    metrics_path = dataset.preprocess_in_situ / "register" / "metrics" / f"registration_metrics_{row}{col}.csv"

    if not metrics_path.exists():
        return None

    try:
        df = pd.read_csv(metrics_path)
        if len(df) == 0:
            return None
        return df
    except Exception:
        return None


def compute_iss_cycle_stats(
    df: pd.DataFrame,
) -> Optional[ISSCycleRegisterStats]:
    """
    Compute aggregate statistics from ISS cycle registration metrics.

    Args:
        df: DataFrame with columns: round_pair, n_matches, n_inliers, inlier_ratio,
            residual_mean, residual_std, residual_max

    Returns:
        ISSCycleRegisterStats or None if no data
    """
    if df is None or df.empty:
        return None

    # Filter to round-to-round registrations (exclude nucleus)
    round_df = df[df["round_pair"].str.startswith("round")]

    if round_df.empty:
        return None

    return ISSCycleRegisterStats(
        mean_inlier_ratio=float(np.round(round_df["inlier_ratio"].mean(), 4)),
        min_inlier_ratio=float(np.round(round_df["inlier_ratio"].min(), 4)),
        mean_residual=float(np.round(round_df["residual_mean"].mean(), 4)),
        max_residual=float(np.round(round_df["residual_max"].max(), 4)),
        n_rounds=len(round_df),
    )


def compute_drift_stats_for_well(
    experiment: str,
    well,
) -> Optional[DriftStats]:
    """
    Compute drift statistics for a single well using cumulative transforms.

    Args:
        experiment: Experiment name
        well: Well identifier (1, "A1", "B2", "A/1/0")

    Returns:
        DriftStats or None if no drift data available
    """
    drift_data = collect_drift_data_for_well(experiment, well)

    if not drift_data:
        return None

    # Extract magnitudes from each round, keyed by round number
    round_magnitudes = {entry["round"]: entry["magnitude"] for entry in drift_data}

    if not round_magnitudes:
        return None

    # Compute per-round drift (delta between consecutive rounds)
    # round 0 is reference (magnitude 0), so drift 0->1 is magnitude of round 1
    per_round_drifts_dict = {}
    sorted_rounds = sorted(round_magnitudes.keys())

    prev_magnitude = 0
    prev_round = 0
    for round_num in sorted_rounds:
        magnitude = round_magnitudes[round_num]
        drift = abs(magnitude - prev_magnitude)
        per_round_drifts_dict[round_num] = float(np.round(drift, 2))
        prev_magnitude = magnitude
        prev_round = round_num

    per_round_drifts_list = list(per_round_drifts_dict.values())

    return DriftStats(
        mean_per_round_drift=float(np.round(np.mean(per_round_drifts_list), 2)),
        max_per_round_drift=float(np.round(np.max(per_round_drifts_list), 2)),
        cumulative_drift=float(np.round(max(round_magnitudes.values()), 2)),
        n_rounds=len(round_magnitudes),
        per_round_drifts=per_round_drifts_dict,
    )


def load_auto_register_metrics(
    dataset,
    well,
    reg_type: str,  # "iss" or "pheno"
    target: str = "track",  # "track", "pheno", or "iss"
) -> Optional[pd.DataFrame]:
    """
    Load auto-registration metrics CSV for a well.

    Args:
        dataset: OpsDataset instance
        well: Well identifier (1, "A1", "B2", "A/1/0")
        reg_type: Registration type ("iss" or "pheno")
        target: Registration target ("track", "pheno", or "iss")

    Returns:
        DataFrame with metrics or None if not found
    """
    row, col = parse_well(well)  # row-agnostic overlay dir token
    auto_overlays_dir = dataset.tracking / "auto_overlays"
    overlay_subdir = auto_overlays_dir / f"{row}{col}_{reg_type}_to_{target}"
    metrics_csv = overlay_subdir / "auto_register_metrics.csv"

    if not metrics_csv.exists():
        return None

    try:
        df = pd.read_csv(metrics_csv, index_col=0)
        if len(df) == 0:
            return None
        return df
    except Exception:
        return None


def compute_auto_register_stats(
    df: pd.DataFrame,
) -> Optional[AutoRegisterStats]:
    """
    Compute statistics from auto-registration metrics.

    Args:
        df: DataFrame with metrics indexed by metric name

    Returns:
        AutoRegisterStats or None if no data
    """
    if df is None or df.empty:
        return None

    overlap = None
    inlier_ratio = None

    if "overlap_forward_overlap_percent" in df.index:
        overlap = float(df.loc["overlap_forward_overlap_percent", df.columns[0]])

    if "ransac_inlier_ratio" in df.index:
        inlier_ratio = float(df.loc["ransac_inlier_ratio", df.columns[0]])

    if overlap is None and inlier_ratio is None:
        return None

    return AutoRegisterStats(
        overlap_percent=overlap if overlap is not None else 0.0,
        inlier_ratio=inlier_ratio if inlier_ratio is not None else 0.0,
    )


def get_iss_cycle_register_stats_for_experiment(
    dataset,
    wells: List[int] = [1, 2, 3],
) -> Dict[str, Any]:
    """
    Get ISS cycle registration statistics across all wells.

    Args:
        dataset: OpsDataset instance
        wells: List of wells to process

    Returns:
        Dictionary with aggregate ISS cycle registration stats including drift
    """
    result = {
        # Inlier/residual stats (commented out until metrics CSV is generated)
        # "iss_cycle_mean_inlier_ratio": None,
        # "iss_cycle_min_inlier_ratio": None,
        # "iss_cycle_mean_residual": None,
        # "iss_cycle_max_residual": None,
        # Drift stats
        "iss_cycle_mean_per_round_drift": None,
        "iss_cycle_max_per_round_drift": None,
        "iss_cycle_cumulative_drift": None,
    }

    # all_inlier_ratios = []
    # all_residuals_mean = []
    # all_residuals_max = []
    all_per_round_drifts = []
    all_max_per_round_drifts = []
    all_cumulative_drifts = []
    # Per-round drift aggregation (max across wells for each round)
    per_round_drift_by_round: Dict[int, List[float]] = {}

    experiment = dataset.experiment

    for well in wells:
        # Get inlier/residual stats from metrics CSV (commented out until metrics CSV is generated)
        # df = load_iss_cycle_metrics(dataset, well)
        # if df is not None:
        #     stats = compute_iss_cycle_stats(df)
        #     if stats:
        #         all_inlier_ratios.append(stats.mean_inlier_ratio)
        #         all_inlier_ratios.append(stats.min_inlier_ratio)
        #         all_residuals_mean.append(stats.mean_residual)
        #         all_residuals_max.append(stats.max_residual)

        # Get drift stats from affine transforms
        drift_stats = compute_drift_stats_for_well(experiment, well)
        if drift_stats:
            all_per_round_drifts.append(drift_stats.mean_per_round_drift)
            all_max_per_round_drifts.append(drift_stats.max_per_round_drift)
            all_cumulative_drifts.append(drift_stats.cumulative_drift)
            # Collect per-round drifts
            for round_num, drift in drift_stats.per_round_drifts.items():
                if round_num not in per_round_drift_by_round:
                    per_round_drift_by_round[round_num] = []
                per_round_drift_by_round[round_num].append(drift)

    # Aggregate inlier/residual stats (commented out until metrics CSV is generated)
    # if all_inlier_ratios:
    #     result["iss_cycle_mean_inlier_ratio"] = float(np.round(np.mean(all_inlier_ratios), 4))
    #     result["iss_cycle_min_inlier_ratio"] = float(np.round(np.min(all_inlier_ratios), 4))
    #
    # if all_residuals_mean:
    #     result["iss_cycle_mean_residual"] = float(np.round(np.mean(all_residuals_mean), 4))
    #
    # if all_residuals_max:
    #     result["iss_cycle_max_residual"] = float(np.round(np.max(all_residuals_max), 4))

    # Aggregate drift stats
    if all_per_round_drifts:
        result["iss_cycle_mean_per_round_drift"] = float(np.round(np.mean(all_per_round_drifts), 2))

    if all_max_per_round_drifts:
        result["iss_cycle_max_per_round_drift"] = float(np.round(np.max(all_max_per_round_drifts), 2))

    if all_cumulative_drifts:
        result["iss_cycle_cumulative_drift"] = float(np.round(np.max(all_cumulative_drifts), 2))

    # Add per-round drift values (max across wells for each round)
    for round_num in sorted(per_round_drift_by_round.keys()):
        prev_round = round_num - 1
        max_drift = float(np.round(np.max(per_round_drift_by_round[round_num]), 2))
        result[f"iss_round_{prev_round}to{round_num}_drift"] = max_drift

    return result


def get_iss_drift_stats_for_well(
    dataset,
    well,
) -> Dict[str, Any]:
    """
    Get ISS drift statistics for a single well.

    Args:
        dataset: OpsDataset instance
        well: Well identifier (1, "A1", "B2", "A/1/0")

    Returns:
        Dictionary with drift stats for this well
    """
    result = {
        "iss_cycle_mean_per_round_drift": None,
        "iss_cycle_max_per_round_drift": None,
        "iss_cycle_cumulative_drift": None,
    }

    experiment = dataset.experiment
    drift_stats = compute_drift_stats_for_well(experiment, well)

    if drift_stats:
        result["iss_cycle_mean_per_round_drift"] = drift_stats.mean_per_round_drift
        result["iss_cycle_max_per_round_drift"] = drift_stats.max_per_round_drift
        result["iss_cycle_cumulative_drift"] = drift_stats.cumulative_drift

        # Add per-round drift values
        for round_num in sorted(drift_stats.per_round_drifts.keys()):
            prev_round = round_num - 1
            drift_val = drift_stats.per_round_drifts[round_num]
            result[f"iss_round_{prev_round}to{round_num}_drift"] = drift_val

    return result


def get_auto_register_stats_for_well(
    dataset,
    well,
) -> Dict[str, Any]:
    """
    Get auto-registration statistics for a single well.

    Args:
        dataset: OpsDataset instance
        well: Well identifier (1, "A1", "B2", "A/1/0")

    Returns:
        Dictionary with auto-registration stats for this well (without well number in keys)
    """
    result = {
        "iss_to_track_overlap": None,
        "iss_to_track_inlier_ratio": None,
        "pheno_to_track_overlap": None,
        "pheno_to_track_inlier_ratio": None,
    }

    # ISS -> Track
    df = load_auto_register_metrics(dataset, well, "iss", "track")
    if df is None:
        # Try skip_track mode (ISS -> Pheno)
        df = load_auto_register_metrics(dataset, well, "iss", "pheno")

    if df is not None:
        stats = compute_auto_register_stats(df)
        if stats:
            result["iss_to_track_overlap"] = stats.overlap_percent
            result["iss_to_track_inlier_ratio"] = stats.inlier_ratio

    # Pheno -> Track
    df = load_auto_register_metrics(dataset, well, "pheno", "track")
    if df is None:
        # Try skip_track mode (Pheno -> ISS)
        df = load_auto_register_metrics(dataset, well, "pheno", "iss")

    if df is not None:
        stats = compute_auto_register_stats(df)
        if stats:
            result["pheno_to_track_overlap"] = stats.overlap_percent
            result["pheno_to_track_inlier_ratio"] = stats.inlier_ratio

    return result


def get_auto_register_stats_for_experiment(
    dataset,
    wells: List[int] = [1, 2, 3],
) -> Dict[str, Any]:
    """
    Get aggregate auto-registration statistics across all wells.

    Args:
        dataset: OpsDataset instance
        wells: List of wells to process

    Returns:
        Dictionary with aggregate auto-registration stats (mean, min overlap)
    """
    result = {
        "iss_to_track_mean_overlap": None,
        "iss_to_track_min_overlap": None,
        "pheno_to_track_mean_overlap": None,
        "pheno_to_track_min_overlap": None,
    }

    iss_overlaps = []
    pheno_overlaps = []

    for well in wells:
        well_stats = get_auto_register_stats_for_well(dataset, well)
        if well_stats["iss_to_track_overlap"] is not None and well_stats["iss_to_track_overlap"] > 0:
            iss_overlaps.append(well_stats["iss_to_track_overlap"])
        if well_stats["pheno_to_track_overlap"] is not None and well_stats["pheno_to_track_overlap"] > 0:
            pheno_overlaps.append(well_stats["pheno_to_track_overlap"])

    # Compute aggregate stats
    if iss_overlaps:
        result["iss_to_track_mean_overlap"] = float(np.round(np.mean(iss_overlaps), 2))
        result["iss_to_track_min_overlap"] = float(np.round(np.min(iss_overlaps), 2))

    if pheno_overlaps:
        result["pheno_to_track_mean_overlap"] = float(np.round(np.mean(pheno_overlaps), 2))
        result["pheno_to_track_min_overlap"] = float(np.round(np.min(pheno_overlaps), 2))

    return result


def get_registration_stats_for_experiment(
    dataset,
    wells: List[int] = [1, 2, 3],
) -> Dict[str, Any]:
    """
    Get all registration statistics for an experiment.

    This is the main entry point for metrics.py to get registration stats.

    Args:
        dataset: OpsDataset instance
        wells: List of wells to process

    Returns:
        Dictionary with all registration stats, ready to add to plate_stats
    """
    result = {}

    # Get ISS cycle registration stats
    iss_cycle_stats = get_iss_cycle_register_stats_for_experiment(dataset, wells)
    result.update(iss_cycle_stats)

    # Get auto-registration stats
    auto_register_stats = get_auto_register_stats_for_experiment(dataset, wells)
    result.update(auto_register_stats)

    return result
