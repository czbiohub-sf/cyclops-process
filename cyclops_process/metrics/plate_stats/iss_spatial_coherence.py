"""
Per-well spatial coherence statistics for plate_stats.csv.

Computes sister cell (matching sgRNA) proximity metrics using a KDTree
on pheno coordinates from linked_results CSVs. Returns summary statistics
(mean, median, std, min, max) of sister_count, sister_ratio,
region_homogeneity and miscall_score aggregated across all sgRNAs in
the well.

Per-cell columns produced by ``_compute_spatial_coherence``:

* ``neighbor_count``     — total KDTree neighbors within ``radius`` (excl. self).
* ``sister_count``       — neighbors that share this cell's sgRNA.
* ``sister_ratio``       — sister_count / neighbor_count (0 if no neighbors).
* ``region_homogeneity`` — fraction of this cell's neighbors that share the
                           MOST-common sgRNA in the neighborhood (independent
                           of the cell's own sgRNA). High = clonal patch;
                           low = mixed neighborhood.
* ``miscall_score``      — max(0, region_homogeneity − sister_ratio). High
                           iff the region is dominated by a single sgRNA AND
                           this cell does not match — the spatial signature
                           of a likely in-situ-sequencing barcode miscall.

Uses cached results from the QC script (per_cell_spatial_coherence.csv)
when available to avoid expensive recomputation. When recomputing from
scratch, saves per-cell results to the same cache path so QC scripts
(qc_sister_effect_analysis.py, qc_cohens_d_ridge.py) can use them
without requiring a separate run of spatial_coherence_analysis.py.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from typing import Dict, Any

from ops_utils.data.experiment import OpsDataset


def _compute_spatial_coherence(
    df: pd.DataFrame,
    radius: float = 500.0,
) -> pd.DataFrame:
    """
    Compute per-cell spatial coherence metrics for a set of cells.

    Uses a sparse distance matrix from cKDTree for vectorized counting.

    Args:
        df: DataFrame with y_pheno, x_pheno, sgRNA columns
        radius: Search radius in pixels

    Returns:
        DataFrame with added columns: neighbor_count, sister_count, sister_ratio
    """
    coords = df[["y_pheno", "x_pheno"]].values.astype(np.float64)
    tree = cKDTree(coords)

    sparse_dist = tree.sparse_distance_matrix(
        tree, max_distance=radius, output_type="coo_matrix"
    )

    row = sparse_dist.row
    col = sparse_dist.col

    # Exclude self-pairs
    mask = row != col
    row = row[mask]
    col = col[mask]

    n_cells = len(df)

    neighbor_counts = np.bincount(row, minlength=n_cells)

    sgrna_values = df["sgRNA"].values
    sgrna_matches = sgrna_values[row] == sgrna_values[col]
    sister_counts = np.bincount(row[sgrna_matches], minlength=n_cells)

    # region_homogeneity: for each cell, fraction of neighbors that share
    # the most-common sgRNA in the neighborhood (independent of the cell's
    # own sgRNA). Vectorized via a sparse (cell × sgRNA-code) count matrix
    # — per-row max gives the dominant sgRNA count per cell.
    sgrna_codes = pd.factorize(sgrna_values)[0]
    n_sgrnas = int(sgrna_codes.max()) + 1 if len(sgrna_codes) else 1
    nbr_count_matrix = coo_matrix(
        (np.ones_like(row, dtype=np.int32), (row, sgrna_codes[col])),
        shape=(n_cells, n_sgrnas),
    ).tocsr()
    region_dominant_count = np.asarray(
        nbr_count_matrix.max(axis=1).todense()
    ).flatten()

    df = df.copy()
    df["neighbor_count"] = neighbor_counts
    df["sister_count"] = sister_counts
    df["sister_ratio"] = np.where(
        neighbor_counts > 0,
        sister_counts / neighbor_counts,
        0.0,
    )
    df["region_homogeneity"] = np.where(
        neighbor_counts > 0,
        region_dominant_count / np.maximum(neighbor_counts, 1),
        0.0,
    )
    # Likely-barcode-miscall signal: region is coherent (high homogeneity)
    # AND the cell doesn't match (low sister_ratio). Clipped at 0.
    df["miscall_score"] = np.maximum(
        0.0, df["region_homogeneity"] - df["sister_ratio"]
    )

    return df


def _summarize_well_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute summary statistics from a DataFrame that already has
    neighbor_count, sister_count, sister_ratio columns.

    Returns dict with sc_-prefixed stats.
    """
    result = {}

    result["sc_n_cells"] = len(df)
    result["sc_n_sgRNAs"] = int(df["sgRNA"].nunique())

    result["sc_neighbor_count_mean"] = float(np.round(df["neighbor_count"].mean(), 2))
    result["sc_neighbor_count_median"] = float(np.round(df["neighbor_count"].median(), 2))

    result["sc_sister_count_mean"] = float(np.round(df["sister_count"].mean(), 4))
    result["sc_sister_count_median"] = float(np.round(df["sister_count"].median(), 4))
    result["sc_sister_count_std"] = float(np.round(df["sister_count"].std(), 4))
    result["sc_sister_count_min"] = int(df["sister_count"].min())
    result["sc_sister_count_max"] = int(df["sister_count"].max())

    result["sc_sister_ratio_mean"] = float(np.round(df["sister_ratio"].mean(), 6))
    result["sc_sister_ratio_median"] = float(np.round(df["sister_ratio"].median(), 6))
    result["sc_sister_ratio_std"] = float(np.round(df["sister_ratio"].std(), 6))
    result["sc_sister_ratio_min"] = float(np.round(df["sister_ratio"].min(), 6))
    result["sc_sister_ratio_max"] = float(np.round(df["sister_ratio"].max(), 6))

    result["sc_pct_isolated"] = float(
        np.round(100 * (df["neighbor_count"] == 0).sum() / len(df), 2)
    )
    result["sc_pct_no_sisters"] = float(
        np.round(100 * (df["sister_count"] == 0).sum() / len(df), 2)
    )

    # region_homogeneity / miscall_score summaries (added in v2; missing on
    # caches written before this change — guard for backward compat).
    if "region_homogeneity" in df.columns:
        rh = df["region_homogeneity"]
        result["sc_region_homog_mean"] = float(np.round(rh.mean(), 6))
        result["sc_region_homog_median"] = float(np.round(rh.median(), 6))
        result["sc_region_homog_p90"] = float(np.round(rh.quantile(0.90), 6))
        result["sc_region_homog_max"] = float(np.round(rh.max(), 6))
    if "miscall_score" in df.columns:
        ms = df["miscall_score"]
        result["sc_miscall_score_mean"] = float(np.round(ms.mean(), 6))
        result["sc_miscall_score_median"] = float(np.round(ms.median(), 6))
        result["sc_miscall_score_p90"] = float(np.round(ms.quantile(0.90), 6))
        result["sc_miscall_score_max"] = float(np.round(ms.max(), 6))
        result["sc_pct_miscall_ge_03"] = float(
            np.round(100 * (ms >= 0.3).sum() / len(df), 2)
        )
        result["sc_pct_miscall_ge_05"] = float(
            np.round(100 * (ms >= 0.5).sum() / len(df), 2)
        )

    return result


def _normalize_well_token(well: str) -> str:
    """
    Normalize a well identifier to the token format stored in the cache CSV.

    Examples: "A/1/0" -> "A1", "A1" -> "A1"
    """
    import re
    if "/" in well:
        parts = [p for p in well.split("/") if p]
        return f"{parts[0]}{parts[1]}" if len(parts) >= 2 else well
    m = re.match(r"^([A-Za-z]+)(\d+)$", well)
    return f"{m.group(1)}{m.group(2)}" if m else well


def _load_cached_well_data(
    dataset: OpsDataset,
    well: str,
) -> pd.DataFrame | None:
    """
    Try to load pre-computed spatial coherence data from the cache.

    Per-cell results are stored at:
        fast_ops/{exp}/3-assembly/spatial_coherence/per_cell_spatial_coherence.csv

    This file is written either by spatial_coherence_analysis.py (QC script)
    or by get_spatial_coherence_stats_for_well when it recomputes from scratch.

    Args:
        dataset: OpsDataset instance
        well: Well identifier (e.g., "A/1/0")

    Returns:
        DataFrame for this well with neighbor_count/sister_count/sister_ratio,
        or None if cache not found.
    """
    cache_path = dataset.results_fast / "spatial_coherence" / "per_cell_spatial_coherence.csv"
    if not cache_path.exists():
        return None

    try:
        df = pd.read_csv(cache_path)
    except Exception:
        return None

    if df.empty or "well" not in df.columns:
        return None

    well_token = _normalize_well_token(well)
    well_df = df[df["well"] == well_token]

    required_cols = ["sgRNA", "neighbor_count", "sister_count", "sister_ratio"]
    if well_df.empty or not all(c in well_df.columns for c in required_cols):
        return None

    return well_df


def _save_well_to_cache(
    dataset: OpsDataset,
    well_token: str,
    df: pd.DataFrame,
) -> None:
    """
    Save per-cell spatial coherence data for one well to the shared cache CSV.

    Merges with any existing cache, replacing rows for this well so that
    re-runs stay consistent. Matches the format written by
    spatial_coherence_analysis.py: all wells concatenated with a "well" column.

    Args:
        dataset: OpsDataset instance
        well_token: Normalized well token (e.g., "A1")
        df: Per-cell DataFrame with neighbor_count/sister_count/sister_ratio columns
    """
    cache_dir = dataset.results_fast / "spatial_coherence"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "per_cell_spatial_coherence.csv"

    df_well = df.copy()
    df_well["well"] = well_token

    if cache_path.exists():
        existing = pd.read_csv(cache_path)
        # Drop any stale rows for this well, then append fresh ones
        existing = existing[existing["well"] != well_token]
        combined = pd.concat([existing, df_well], ignore_index=True)
    else:
        combined = df_well

    combined.to_csv(cache_path, index=False)


def get_spatial_coherence_stats_for_well(
    dataset: OpsDataset,
    well: str,
    radius: float = 500.0,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Compute spatial coherence statistics for a single well.

    Checks for cached results first. If not available (or force=True),
    recomputes from the linked_results CSV and saves per-cell results to
    the shared cache (fast_ops/{exp}/3-assembly/spatial_coherence/
    per_cell_spatial_coherence.csv) so QC scripts can use them without a
    separate run of spatial_coherence_analysis.py.

    Args:
        dataset: OpsDataset instance
        well: Well identifier (e.g., "A/1/0")
        radius: Search radius in pixels (default 500)
        force: If True, skip cache and recompute from scratch

    Returns:
        Dictionary with spatial coherence stats for this well, prefixed with
        "sc_" (spatial coherence). Returns empty dict if data unavailable.
    """
    # Try cached data first
    if not force:
        cached_df = _load_cached_well_data(dataset, well)
        if cached_df is not None:
            return _summarize_well_stats(cached_df)

    # No cache available - compute from linked_results
    link_path = dataset.append_well("linked_results", well)
    if not link_path.exists():
        return {}

    df = pd.read_csv(link_path)

    if df.empty:
        return {}

    # Normalize barcode column name: linked_results may use "barcode" instead of "sgRNA"
    if "sgRNA" not in df.columns and "barcode" in df.columns:
        df = df.rename(columns={"barcode": "sgRNA"})

    required_cols = ["y_pheno", "x_pheno", "sgRNA"]
    if not all(c in df.columns for c in required_cols):
        return {}

    df = df.dropna(subset=required_cols)
    if df.empty:
        return {}

    df = _compute_spatial_coherence(df, radius=radius)

    well_token = _normalize_well_token(well)
    _save_well_to_cache(dataset, well_token, df)

    return _summarize_well_stats(df)
