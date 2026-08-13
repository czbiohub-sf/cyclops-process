"""
ISS Reconstruction Metrics Helper Functions.

This module provides helper functions for computing QC metrics from reconstruction
outputs for inclusion in plate_stats.csv.

Supports two reconstruction backends:
  1. Tilt-corrected (reconstruct_tilt_corrected) — reads per-well
     tilt_params_all.csv with columns: z_offset, zenith, azimuth.
  2. Legacy subtile recon (reconstruct) — reads {exp}_{process}_subtile_metadata.csv
     with column: z_focus_offset.

The tilt-corrected backend is tried first; if not found, falls back to legacy.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class TiltStats:
    """Statistics from tilt-corrected reconstruction."""
    z_offset_mean: float
    z_offset_median: float
    z_offset_std: float
    z_offset_min: float
    z_offset_max: float
    zenith_mean: float
    zenith_median: float
    zenith_std: float
    azimuth_mean: float
    azimuth_median: float
    azimuth_std: float
    n_subtiles: int


@dataclass
class ZOffsetStats:
    """Statistics from legacy z-offset analysis."""
    mean: float
    median: float
    std: float
    min: float
    max: float
    n_subtiles: int


# ---------------------------------------------------------------------------
# Tilt-corrected loader
# ---------------------------------------------------------------------------

def _load_well_tilt_params(well_dir: Path) -> Optional[pd.DataFrame]:
    """Load tilt params for a single well directory.

    Tries tilt_params_all.csv first, then falls back to concatenating
    per-tile CSVs from the csvs/ subdirectory (track process stores
    params as individual csvs/tilt_params_*.csv files).
    """
    if not well_dir.exists():
        return None

    # Try consolidated file first
    csv_path = well_dir / "tilt_params_all.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            return df if len(df) > 0 else None
        except Exception:
            pass

    # Fall back to per-tile CSVs in csvs/ subdirectory
    csvs_dir = well_dir / "csvs"
    if csvs_dir.exists():
        tile_csvs = sorted(csvs_dir.glob("tilt_params_*.csv"))
        if tile_csvs:
            dfs: List[pd.DataFrame] = []
            for tc in tile_csvs:
                try:
                    df = pd.read_csv(tc)
                    if len(df) > 0:
                        dfs.append(df)
                except Exception:
                    continue
            if dfs:
                return pd.concat(dfs, ignore_index=True)

    return None


def _load_tilt_params(
    dataset,
    process: str = "pheno",
    well: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Load tilt_params_all.csv for the given process and (optionally) well.

    The tilt-corrected step writes per-well CSVs under:
        reconstruction/tilt_calibration/{process}/{well_tag}/tilt_params_all.csv

    If ``well`` is provided, load that single well's CSV. Otherwise
    concatenate all wells found for the process.
    """
    recon_dir = dataset.preprocess_live / "reconstruction"
    tilt_base = recon_dir / "tilt_calibration" / process

    if not tilt_base.exists():
        return None

    if well is not None:
        # Normalize "A/1/0" -> "A/1" -> "A_1"
        well_prefix = "/".join(well.split("/")[:2])
        well_tag = well_prefix.replace("/", "_")
        well_dir = tilt_base / well_tag
        df = _load_well_tilt_params(well_dir)
        return df

    # No well filter — concatenate all wells
    dfs: List[pd.DataFrame] = []
    for well_dir in sorted(tilt_base.iterdir()):
        if not well_dir.is_dir():
            continue
        df = _load_well_tilt_params(well_dir)
        if df is not None:
            dfs.append(df)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def _compute_tilt_stats(df: pd.DataFrame) -> Optional[TiltStats]:
    """Compute statistics from a tilt_params DataFrame."""
    if df is None or df.empty:
        return None

    required = {"z_offset", "zenith", "azimuth"}
    if not required.issubset(df.columns):
        return None

    z = df["z_offset"].values
    zen = df["zenith"].values
    azi = df["azimuth"].values

    return TiltStats(
        z_offset_mean=float(np.round(np.mean(z), 4)),
        z_offset_median=float(np.round(np.median(z), 4)),
        z_offset_std=float(np.round(np.std(z), 4)),
        z_offset_min=float(np.round(np.min(z), 4)),
        z_offset_max=float(np.round(np.max(z), 4)),
        zenith_mean=float(np.round(np.mean(zen), 6)),
        zenith_median=float(np.round(np.median(zen), 6)),
        zenith_std=float(np.round(np.std(zen), 6)),
        azimuth_mean=float(np.round(np.mean(azi), 4)),
        azimuth_median=float(np.round(np.median(azi), 4)),
        azimuth_std=float(np.round(np.std(azi), 4)),
        n_subtiles=len(z),
    )


# ---------------------------------------------------------------------------
# Legacy subtile metadata loader (kept for backwards compatibility)
# ---------------------------------------------------------------------------

def load_subtile_metadata(
    dataset,
    metadata_type: str = "pheno",
) -> Optional[pd.DataFrame]:
    """
    Load subtile metadata CSV for an experiment (legacy reconstruct step).

    Args:
        dataset: OpsDataset instance
        metadata_type: "pheno" or "track"

    Returns:
        DataFrame with subtile metadata or None if not found
    """
    recon_dir = dataset.preprocess_live / "reconstruction"
    exp_name = dataset.experiment
    pattern = f"{exp_name}_{metadata_type}-2d_subtile_metadata.csv"
    metadata_path = recon_dir / pattern

    if not metadata_path.exists():
        # Try alternate patterns
        alt_patterns = list(recon_dir.glob(f"*{metadata_type}-2d_subtile_metadata.csv"))
        if alt_patterns:
            metadata_path = alt_patterns[0]
        else:
            return None

    try:
        df = pd.read_csv(metadata_path)
        if len(df) == 0:
            return None
        return df
    except Exception:
        return None


def compute_z_offset_stats(
    df: pd.DataFrame,
    well: Optional[str] = None,
) -> Optional[ZOffsetStats]:
    """
    Compute z-offset statistics from legacy subtile metadata.

    Args:
        df: Subtile metadata DataFrame with 'z_focus_offset' column
        well: Optional well to filter (e.g., "A/1/0"). If None, computes across all.

    Returns:
        ZOffsetStats or None if no data available
    """
    if df is None or df.empty or "z_focus_offset" not in df.columns:
        return None

    # Filter by well if specified and well column exists
    if well is not None:
        well_col = None
        for col in ["well", "position", "pos", "fov"]:
            if col in df.columns:
                well_col = col
                break

        if well_col is not None:
            if "/" in well:
                well_prefix = "/".join(well.split("/")[:2])  # "A/1/0" -> "A/1"
                df = df[df[well_col].str.startswith(well_prefix)]
            else:
                df = df[df[well_col] == well]
            if df.empty:
                return None

    z_offsets = df["z_focus_offset"].values

    if len(z_offsets) == 0:
        return None

    return ZOffsetStats(
        mean=float(np.round(np.mean(z_offsets), 4)),
        median=float(np.round(np.median(z_offsets), 4)),
        std=float(np.round(np.std(z_offsets), 4)),
        min=float(np.round(np.min(z_offsets), 4)),
        max=float(np.round(np.max(z_offsets), 4)),
        n_subtiles=len(z_offsets),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _get_stats_for_process(
    dataset,
    process: str,
    well: Optional[str] = None,
) -> Dict[str, Any]:
    """Get reconstruction stats for a single process (pheno or track).

    Tries tilt-corrected data first, falls back to legacy subtile metadata.

    Returns dict with keys prefixed by process name, e.g.:
        pheno_z_offset_mean, pheno_zenith_mean, etc.
    """
    result: Dict[str, Any] = {
        f"{process}_z_offset_mean": None,
        f"{process}_z_offset_median": None,
        f"{process}_z_offset_std": None,
        f"{process}_z_offset_min": None,
        f"{process}_z_offset_max": None,
        f"{process}_zenith_mean": None,
        f"{process}_zenith_median": None,
        f"{process}_zenith_std": None,
        f"{process}_azimuth_mean": None,
        f"{process}_azimuth_median": None,
        f"{process}_azimuth_std": None,
        f"{process}_recon_n_subtiles": None,
    }

    # Try tilt-corrected first
    tilt_df = _load_tilt_params(dataset, process=process, well=well)
    if tilt_df is not None:
        stats = _compute_tilt_stats(tilt_df)
        if stats is not None:
            result[f"{process}_z_offset_mean"] = stats.z_offset_mean
            result[f"{process}_z_offset_median"] = stats.z_offset_median
            result[f"{process}_z_offset_std"] = stats.z_offset_std
            result[f"{process}_z_offset_min"] = stats.z_offset_min
            result[f"{process}_z_offset_max"] = stats.z_offset_max
            result[f"{process}_zenith_mean"] = stats.zenith_mean
            result[f"{process}_zenith_median"] = stats.zenith_median
            result[f"{process}_zenith_std"] = stats.zenith_std
            result[f"{process}_azimuth_mean"] = stats.azimuth_mean
            result[f"{process}_azimuth_median"] = stats.azimuth_median
            result[f"{process}_azimuth_std"] = stats.azimuth_std
            result[f"{process}_recon_n_subtiles"] = stats.n_subtiles
            return result

    # Fallback to legacy subtile metadata
    legacy_df = load_subtile_metadata(dataset, metadata_type=process)
    if legacy_df is not None:
        stats = compute_z_offset_stats(legacy_df, well)
        if stats is not None:
            result[f"{process}_z_offset_mean"] = stats.mean
            result[f"{process}_z_offset_median"] = stats.median
            result[f"{process}_z_offset_std"] = stats.std
            result[f"{process}_z_offset_min"] = stats.min
            result[f"{process}_z_offset_max"] = stats.max
            result[f"{process}_recon_n_subtiles"] = stats.n_subtiles

    return result


def get_z_offset_stats_for_experiment(
    dataset,
    well: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get reconstruction QC statistics for pheno and track.

    This is the main entry point for metrics.py / iss_stats.py to get
    reconstruction stats. Tries tilt-corrected data first, falls back
    to legacy subtile metadata.

    Args:
        dataset: OpsDataset instance
        well: Optional well to filter. If None, computes across all wells.

    Returns:
        Dictionary with reconstruction stats for each process, ready to
        merge into plate_stats. Keys include z_offset, zenith, azimuth
        stats prefixed by process name (pheno_ / track_).
    """
    result = {}
    result.update(_get_stats_for_process(dataset, "pheno", well))
    result.update(_get_stats_for_process(dataset, "track", well))
    return result
