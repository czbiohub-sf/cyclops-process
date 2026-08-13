"""Flatfield correction QC metrics for inclusion in plate_stats.csv.

Reads the per-tile QC CSV produced by correct_flatfield and aggregates
to per-well statistics for the main metrics pipeline.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional


def get_flatfield_stats_for_well(
    dataset,
    well: str,
) -> Dict[str, Any]:
    """Get flatfield correction QC stats for a given well.

    Reads the QC CSV from 3-assembly/illumination_correction/ and returns
    mean metrics across tiles in the specified well.

    Args:
        dataset: OpsDataset instance
        well: Well identifier (e.g., "A/1/0")

    Returns:
        Dictionary with flatfield QC stats, ready to add to plate_stats
    """
    result = {
        "flatfield_cv_before": None,
        "flatfield_cv_after": None,
        "flatfield_cv_improvement": None,
        "flatfield_snr_before": None,
        "flatfield_snr_after": None,
        "flatfield_snr_change": None,
        "flatfield_profile_range": None,
    }

    qc_dir = dataset.experiment_path / "3-assembly" / "illumination_correction"
    # Find the QC CSV (pattern: <experiment>_flatfield_qc.csv)
    csv_files = list(qc_dir.glob("*_flatfield_qc.csv"))
    if not csv_files:
        return result

    df = pd.read_csv(csv_files[0])
    if df.empty:
        return result

    # Extract well prefix from position (e.g., "A/1/029031" -> "A/1")
    well_prefix = "/".join(well.split("/")[:2])
    well_rows = df[df["position"].str.startswith(well_prefix)]

    if well_rows.empty:
        return result

    result["flatfield_cv_before"] = round(float(well_rows["cv_before"].mean()), 6)
    result["flatfield_cv_after"] = round(float(well_rows["cv_after"].mean()), 6)
    result["flatfield_cv_improvement"] = round(float(well_rows["cv_improvement"].mean()), 6)
    result["flatfield_snr_before"] = round(float(well_rows["snr_before"].mean()), 2)
    result["flatfield_snr_after"] = round(float(well_rows["snr_after"].mean()), 2)
    result["flatfield_snr_change"] = round(float(well_rows["snr_change"].mean()), 2)
    result["flatfield_profile_range"] = round(float(well_rows["flatfield_range"].mean()), 4)

    return result
