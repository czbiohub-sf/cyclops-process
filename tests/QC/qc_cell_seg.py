"""
Cell Segmentation QC - BBox Shape Analysis
==========================================

Analyzes cell bounding box properties across all experiments using
bbox dimensions from the linked CSV (height, width, area, aspect ratio).

For each experiment:
1. Calls cell_size_metrics() from iss_cell_size.py to compute per-cell metrics
2. Collects experiment-level summary statistics
3. Runs outlier detection across experiments
4. Generates ridge plots (one per metric, mean+std columns, rows ordered by ops number),
   violin/strip plots, and individual 4-panel metric plots

Per-experiment outputs stored in: fast_ops/{exp}/3-assembly/cell_sizes/
QC plots stored in: tests/QC/cell_seg_qc/

Usage
-----
    python tests/QC/qc_cell_seg.py
    python tests/QC/qc_cell_seg.py -v
    python tests/QC/qc_cell_seg.py -o tests/QC/cell_seg_qc -j 8
    python tests/QC/qc_cell_seg.py --z-threshold 3.0
"""

import sys
import os
import re
import time
import yaml
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from typing import Optional
from tqdm import tqdm
from joblib import Parallel, delayed

sys.path.insert(0, os.getcwd())

from ops_utils.data.experiment import OpsDataset
from ops_utils.hpc.resource_manager import get_optimal_workers
from ops_utils.hpc.slurm_utils import print_slurm_job_header
from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
from ops_utils.data.bad_experiments import is_excluded
from cyclops_process.metrics.plate_stats.iss_cell_size import (
    cell_size_metrics,
    _SHAPE_COLUMNS,
)


def _extract_ops_number(exp_name: str) -> int:
    """Extract ops number from experiment name for sorting.

    'ops0033_20250429' -> 33
    """
    m = re.search(r"ops0*(\d+)", exp_name)
    return int(m.group(1)) if m else 999999


# === Core Analysis ===


def analyze_experiment_cell_seg(
    dataset: OpsDataset,
    exp_name: str,
    method: str = "mine",
    force: bool = False,
    verbose: bool = False,
) -> dict | None:
    """
    Analyze cell segmentation shape for a single experiment.

    Delegates to cell_size_metrics() from iss_cell_size.py for the actual
    computation and per-cell CSV generation.

    Returns experiment-level summary statistics dict or None if no data found.
    """
    t_start = time.time()

    try:
        well_stats = cell_size_metrics(
            exp_name, method=method, force=force
        )
    except Exception as e:
        if verbose:
            print(f"  Error computing cell size metrics for {exp_name}: {e}")
        return None

    if not well_stats:
        if verbose:
            print(f"  No cell size data for {exp_name}")
        return None

    # Read back the summary CSV to get pooled stats
    summary_path = dataset.metrics_paths.get("cell_size_summary")
    if summary_path and summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        pooled = summary_df[summary_df["well"] == "ALL"]
        if pooled.empty:
            pooled = summary_df  # use all rows if no pooled
    else:
        return None

    elapsed = time.time() - t_start

    # Build experiment-level stats dict
    result = {
        "experiment": exp_name,
        "n_cells": int(pooled.iloc[0].get("cell_count", 0)),
        "n_wells": len(well_stats),
    }

    # Add mean and std for each shape metric
    for col in _SHAPE_COLUMNS:
        for agg in ("mean", "median", "min", "max", "std"):
            key = f"{col}_{agg}"
            if key in pooled.columns:
                result[key] = float(pooled.iloc[0][key])

    # Add large cell metrics
    if "large_cell_count" in pooled.columns:
        result["large_cell_count"] = int(pooled.iloc[0]["large_cell_count"])
    if "large_cell_pct" in pooled.columns:
        result["large_cell_pct"] = float(pooled.iloc[0]["large_cell_pct"])

    if verbose:
        large_pct = result.get("large_cell_pct", 0)
        print(f"  {exp_name}: {result['n_cells']:,} cells, {large_pct:.1f}% large, {elapsed:.1f}s")

    return result


def _process_single_experiment(
    config_path: Path,
    method: str,
    force: bool,
    verbose: bool,
) -> dict | None:
    """Process a single experiment config file. Returns stats dict or None."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            if not config or "experiment_name" not in config:
                return None

        exp_name = config["experiment_name"]

        # Filter: only standard format opsXXXX_YYYYMMDD
        if not re.match(r"^ops\d{4}_\d{8}$", exp_name):
            return None

        # Skip excluded experiments
        if is_excluded(exp_name):
            return None

        dataset = OpsDataset(exp_name, method=method)

        return analyze_experiment_cell_seg(
            dataset, exp_name, method=method, force=force, verbose=verbose
        )

    except Exception as e:
        print(f"\nError processing {config_path.stem}: {e}")
        return None


def collect_all_experiment_stats(
    method: str = "mine",
    force: bool = False,
    verbose: bool = False,
    n_jobs: int = None,
) -> pd.DataFrame:
    """
    Collect cell size statistics from all experiments.

    Uses joblib for parallel processing — bbox parsing from CSVs is lightweight
    so memory is not a concern.

    Returns DataFrame with one row per experiment, sorted by ops number.
    """
    dummy_dataset = OpsDataset("dummy")
    config_root_dir = dummy_dataset.config_paths["exp_config_dir"]

    if not config_root_dir.is_dir():
        print(f"Configuration directory not found at {config_root_dir}. Aborting.")
        return pd.DataFrame()

    config_files = sorted(list(config_root_dir.glob("*_config.yaml")))

    if n_jobs is None:
        n_jobs = get_optimal_workers(
            use_gpu=False, model_ram_gb=1.0, data_ram_gb=0.5, verbose=True
        )

    print(f"Processing {len(config_files)} configs with {n_jobs} workers")

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_process_single_experiment)(config_path, method, force, verbose)
        for config_path in tqdm(config_files, desc="Analyzing experiments")
    )

    all_stats = [r for r in results if r is not None]
    n_skipped = len(results) - len(all_stats)

    if n_skipped > 0:
        print(
            f"\n{n_skipped} experiments skipped (excluded, non-standard, or no data)"
        )
    print(f"\nAnalyzed {len(all_stats)} experiments with cell seg data\n")

    df = pd.DataFrame(all_stats)

    # Sort by ops number
    if not df.empty and "experiment" in df.columns:
        df["_ops_num"] = df["experiment"].apply(_extract_ops_number)
        df = df.sort_values("_ops_num").reset_index(drop=True)
        df = df.drop(columns=["_ops_num"])

    return df


# === Outlier Detection (matching QC script approach) ===


def calculate_zscore_outliers(
    series: pd.Series, threshold: float = 2.5
) -> pd.Series:
    """Identify outliers using z-score method."""
    mean = series.mean()
    std = series.std()
    if std == 0:
        return pd.Series([False] * len(series), index=series.index)
    z_scores = (series - mean) / std
    return np.abs(z_scores) > threshold


def calculate_modified_zscore_outliers(
    series: pd.Series, threshold: float = 3.5
) -> pd.Series:
    """Identify outliers using modified z-score (MAD-based)."""
    median = series.median()
    mad = np.median(np.abs(series - median))
    if mad == 0:
        mad = np.mean(np.abs(series - median))
    if mad == 0:
        return pd.Series([False] * len(series), index=series.index)
    modified_z = 0.6745 * (series - median) / mad
    return np.abs(modified_z) > threshold


def calculate_iqr_outliers(
    series: pd.Series, multiplier: float = 1.5
) -> pd.Series:
    """Identify outliers using IQR method."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    return (series < lower_bound) | (series > upper_bound)


def analyze_distribution_shifts(
    df: pd.DataFrame,
    metrics: Optional[list[str]] = None,
    z_threshold: float = 2.5,
    mod_z_threshold: float = 3.5,
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Analyze all metrics for distribution shifts using 3-method consensus."""
    if metrics is None:
        # All mean and std variants for each shape column + large cell pct
        metrics = []
        for col in _SHAPE_COLUMNS:
            for agg in ("mean", "std"):
                metrics.append(f"{col}_{agg}")
        metrics.append("large_cell_count")
        metrics.append("large_cell_pct")

    metrics = [m for m in metrics if m in df.columns]
    results = df[["experiment"]].copy()
    if "n_cells" in df.columns:
        results["n_cells"] = df["n_cells"]

    for metric in metrics:
        series = df[metric].dropna()
        if len(series) < 3:
            continue

        # Re-index to match df
        full_series = df[metric]

        z_outlier = calculate_zscore_outliers(full_series.fillna(full_series.median()), z_threshold)
        mod_z_outlier = calculate_modified_zscore_outliers(
            full_series.fillna(full_series.median()), mod_z_threshold
        )
        iqr_outlier = calculate_iqr_outliers(full_series.fillna(full_series.median()), iqr_multiplier)

        mean = full_series.mean()
        std = full_series.std()
        z_scores = (
            (full_series - mean) / std if std > 0 else pd.Series([0] * len(full_series))
        )

        outlier_count = (
            z_outlier.astype(int)
            + mod_z_outlier.astype(int)
            + iqr_outlier.astype(int)
        )
        combined_outlier = outlier_count >= 2

        results[f"{metric}_value"] = full_series
        results[f"{metric}_zscore"] = z_scores
        results[f"{metric}_outlier"] = combined_outlier
        results[f"{metric}_direction"] = np.where(
            combined_outlier, np.where(z_scores > 0, "HIGH", "LOW"), ""
        )

    outlier_cols = [c for c in results.columns if c.endswith("_outlier")]
    results["total_outlier_flags"] = results[outlier_cols].sum(axis=1)
    results["is_anomalous"] = results["total_outlier_flags"] >= 2

    return results


# === Visualization Functions ===


def _sort_df_by_ops_number(df: pd.DataFrame) -> pd.DataFrame:
    """Sort DataFrame by ops number extracted from experiment name."""
    df = df.copy()
    df["_ops_num"] = df["experiment"].apply(_extract_ops_number)
    df = df.sort_values("_ops_num").reset_index(drop=True)
    df = df.drop(columns=["_ops_num"])
    return df


def create_ridge_plots(
    df: pd.DataFrame,
    analysis: pd.DataFrame,
    output_dir: Path,
    overlap: float = 0.6,
) -> None:
    """Create one ridge plot canvas per shape metric, each with mean and std columns.

    Rows are ordered by ops number ascending.
    """
    # Sort by ops number
    sorted_df = _sort_df_by_ops_number(df)
    sorted_analysis = (
        analysis.set_index("experiment").loc[sorted_df["experiment"]].reset_index()
    )

    n_experiments = len(sorted_df)
    experiments = sorted_df["experiment"].values
    n_cells_arr = (
        sorted_df["n_cells"].values
        if "n_cells" in sorted_df.columns
        else np.zeros(n_experiments)
    )

    for col in _SHAPE_COLUMNS:
        mean_col = f"{col}_mean"
        std_col = f"{col}_std"

        if mean_col not in df.columns:
            continue

        has_std = std_col in df.columns

        n_cols = 2 if has_std else 1
        fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, max(8, n_experiments * 0.35)))

        if n_cols == 1:
            axes = [axes]

        sub_metrics = [(mean_col, f"{col} Mean")]
        if has_std:
            sub_metrics.append((std_col, f"{col} Std"))

        for ax, (metric, label) in zip(axes, sub_metrics):
            if metric not in sorted_df.columns:
                ax.set_visible(False)
                continue

            values = sorted_df[metric].values
            outlier_col = f"{metric}_outlier"
            is_outlier = (
                sorted_analysis[outlier_col].values
                if outlier_col in sorted_analysis.columns
                else np.zeros(n_experiments, dtype=bool)
            )

            global_mean = np.nanmean(values)
            global_std = np.nanstd(values)

            val_range = np.nanmax(values) - np.nanmin(values)
            if val_range == 0:
                val_range = 1
            x_min = np.nanmin(values) - 0.15 * val_range
            x_max = np.nanmax(values) + 0.15 * val_range
            x_range = np.linspace(x_min, x_max, 200)

            row_height = 1.0

            for i, (val, exp, n_cells, outlier) in enumerate(
                zip(values, experiments, n_cells_arr, is_outlier)
            ):
                if np.isnan(val):
                    continue

                bandwidth = max(global_std * 0.15, 0.001)
                y_offset = i * row_height * (1 - overlap)

                density = np.exp(-0.5 * ((x_range - val) / bandwidth) ** 2)
                density = density / density.max() * row_height * 0.8

                if outlier:
                    color = "red"
                    alpha = 0.8
                    linewidth = 1.5
                else:
                    color = "steelblue"
                    alpha = 0.4
                    linewidth = 0.8

                ax.fill_between(
                    x_range,
                    y_offset,
                    y_offset + density,
                    color=color,
                    alpha=alpha,
                    linewidth=0,
                )
                ax.plot(
                    x_range,
                    y_offset + density,
                    color=color,
                    alpha=min(1, alpha + 0.3),
                    linewidth=linewidth,
                )

                # Label with experiment name and cell count
                label_color = "red" if outlier else "black"
                label_weight = "bold" if outlier else "normal"
                exp_label = f"{exp[:12]} ({int(n_cells / 1000)}k)"
                ax.text(
                    x_min - 0.02 * (x_max - x_min),
                    y_offset,
                    exp_label,
                    fontsize=6,
                    ha="right",
                    va="center",
                    color=label_color,
                    fontweight=label_weight,
                )

            ax.axvline(
                global_mean,
                color="green",
                linestyle="-",
                lw=2,
                alpha=0.7,
                label=f"Mean: {global_mean:.1f}",
            )
            if global_std > 0:
                ax.axvline(
                    global_mean - 2.5 * global_std,
                    color="orange",
                    linestyle="--",
                    lw=1.5,
                    alpha=0.7,
                    label="-2.5\u03c3",
                )
                ax.axvline(
                    global_mean + 2.5 * global_std,
                    color="orange",
                    linestyle="--",
                    lw=1.5,
                    alpha=0.7,
                    label="+2.5\u03c3",
                )

            ax.set_xlabel(label, fontsize=12)
            ax.set_ylabel("Experiments (ordered by ops #)", fontsize=10)
            ax.set_title(
                f"{label}\n(red = outlier)", fontsize=12, fontweight="bold"
            )
            ax.legend(loc="upper right", fontsize=8)
            ax.set_yticks([])
            ax.set_xlim(x_min - 0.2 * (x_max - x_min), x_max)

        col_clean = col.replace("_", " ").title()
        plt.suptitle(
            f"Cell Seg {col_clean} Distribution Across Experiments\n"
            "(Ridge Plot - rows ordered by ops number)",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(
            output_dir / f"ridge_{col}.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

    # --- Large cell bar chart: count + pct side by side ---
    large_metrics = []
    if "large_cell_count" in sorted_df.columns:
        large_metrics.append(("large_cell_count", "Large Cell Count"))
    if "large_cell_pct" in sorted_df.columns:
        large_metrics.append(("large_cell_pct", "Large Cell %"))

    if large_metrics:
        n_cols = len(large_metrics)
        fig, axes = plt.subplots(
            1, n_cols, figsize=(7 * n_cols, max(8, n_experiments * 0.25))
        )
        if n_cols == 1:
            axes = [axes]

        exp_labels = [
            f"{exp[:12]} ({int(nc / 1000)}k)"
            for exp, nc in zip(experiments, n_cells_arr)
        ]
        y_pos = np.arange(n_experiments)

        for ax, (metric, label) in zip(axes, large_metrics):
            if metric not in sorted_df.columns:
                ax.set_visible(False)
                continue

            values = sorted_df[metric].fillna(0).values
            outlier_col = f"{metric}_outlier"
            is_outlier = (
                sorted_analysis[outlier_col].values
                if outlier_col in sorted_analysis.columns
                else np.zeros(n_experiments, dtype=bool)
            )

            colors = ["red" if o else "steelblue" for o in is_outlier]

            ax.barh(y_pos, values, color=colors, alpha=0.7, edgecolor="none")

            global_mean = np.nanmean(values)
            ax.axvline(
                global_mean, color="green", linestyle="-", lw=2, alpha=0.7,
                label=f"Mean: {global_mean:.1f}",
            )

            ax.set_yticks(y_pos)
            ax.set_yticklabels(exp_labels, fontsize=5)
            ax.invert_yaxis()
            ax.set_xlabel(label, fontsize=12)
            ax.set_title(
                f"{label} (bbox >= 300x300)\n(red = outlier)",
                fontsize=12, fontweight="bold",
            )
            ax.legend(loc="lower right", fontsize=8)

        plt.suptitle(
            "Large Cell Metrics Across Experiments\n(ordered by ops number)",
            fontsize=14, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        plt.savefig(
            output_dir / "bar_large_cell.png", dpi=150, bbox_inches="tight"
        )
        plt.close()


def _load_per_cell_data_for_experiments(
    df: pd.DataFrame,
    method: str = "mine",
) -> dict[str, np.ndarray]:
    """Load per-cell bbox_area arrays from per-well CSVs for each experiment.

    Returns dict mapping experiment name -> array of bbox_area values (all cells).
    """
    from ops_utils.data.experiment import OpsDataset

    result = {}
    for _, row in df.iterrows():
        exp_name = row["experiment"]
        try:
            dataset = OpsDataset(exp_name, method=method)
            cell_sizes_dir = dataset.results / "cell_sizes"
            if not cell_sizes_dir.exists():
                continue

            csvs = sorted(cell_sizes_dir.glob("*_cell_sizes.csv"))
            if not csvs:
                continue

            areas = []
            for csv_path in csvs:
                chunk = pd.read_csv(csv_path, usecols=["bbox_area"])
                areas.append(chunk["bbox_area"].dropna().values)

            if not areas:
                continue

            result[exp_name] = np.concatenate(areas)
        except Exception as e:
            print(f"  Warning: could not load per-cell data for {exp_name}: {e}")
            continue

    return result


def create_area_distribution_ridge_plot(
    df: pd.DataFrame,
    analysis: pd.DataFrame,
    output_dir: Path,
    method: str = "mine",
) -> None:
    """Ridge plot of per-cell bbox_area KDE distributions, one row per experiment.

    Shows the actual shape of cell size distributions — reveals bimodality,
    fat tails, and shifts that summary stats miss.
    """
    sorted_df = _sort_df_by_ops_number(df)
    sorted_analysis = (
        analysis.set_index("experiment").loc[sorted_df["experiment"]].reset_index()
    )

    print("  Loading per-cell area data for ridge plot...")
    per_cell = _load_per_cell_data_for_experiments(sorted_df, method=method)

    if not per_cell:
        print("  No per-cell data found, skipping area distribution ridge plot")
        return

    experiments = sorted_df["experiment"].values
    n_cells_arr = (
        sorted_df["n_cells"].values
        if "n_cells" in sorted_df.columns
        else np.zeros(len(sorted_df))
    )
    n_experiments = len(experiments)

    # Get outlier flags for bbox_area_mean
    outlier_col = "bbox_area_mean_outlier"
    is_outlier_arr = (
        sorted_analysis[outlier_col].values
        if outlier_col in sorted_analysis.columns
        else np.zeros(n_experiments, dtype=bool)
    )

    # Determine global x range from all data
    all_vals = np.concatenate([v for v in per_cell.values()])
    x_min = np.percentile(all_vals, 0.5)
    x_max = np.percentile(all_vals, 99.5)
    x_range = np.linspace(x_min, x_max, 300)

    fig, ax = plt.subplots(figsize=(14, max(8, n_experiments * 0.35)))

    row_height = 1.0
    overlap = 0.6
    plotted = 0

    for i, (exp, n_cells, is_outlier) in enumerate(
        zip(experiments, n_cells_arr, is_outlier_arr)
    ):
        if exp not in per_cell:
            continue

        areas = per_cell[exp]
        y_offset = plotted * row_height * (1 - overlap)

        # KDE
        try:
            kde = stats.gaussian_kde(areas, bw_method=0.15)
            density = kde(x_range)
            density = density / density.max() * row_height * 0.8
        except Exception:
            continue

        color = "red" if is_outlier else "steelblue"
        alpha = 0.7 if is_outlier else 0.4

        ax.fill_between(
            x_range, y_offset, y_offset + density,
            color=color, alpha=alpha, linewidth=0,
        )
        ax.plot(
            x_range, y_offset + density,
            color=color, alpha=min(1, alpha + 0.3), linewidth=0.8,
        )

        # Label
        label_color = "red" if is_outlier else "black"
        label_weight = "bold" if is_outlier else "normal"
        exp_label = f"{exp[:12]} ({int(n_cells / 1000)}k)"
        ax.text(
            x_min - 0.02 * (x_max - x_min), y_offset,
            exp_label, fontsize=6, ha="right", va="center",
            color=label_color, fontweight=label_weight,
        )
        plotted += 1

    # Global median line
    global_median = np.median(all_vals)
    ax.axvline(global_median, color="green", linestyle="-", lw=2, alpha=0.7,
               label=f"Global Median: {global_median:.0f}")

    ax.set_xlabel("Cell BBox Area (px)", fontsize=12)
    ax.set_ylabel("Experiments (ordered by ops #)", fontsize=10)
    ax.set_title(
        "Per-Cell BBox Area Distribution (KDE)\nAcross Experiments",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.set_yticks([])
    ax.set_xlim(x_min - 0.15 * (x_max - x_min), x_max)

    plt.tight_layout()
    plt.savefig(output_dir / "ridge_area_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved ridge_area_distribution.png ({plotted} experiments)")


def create_area_bin_heatmap(
    df: pd.DataFrame,
    output_dir: Path,
    method: str = "mine",
) -> None:
    """Heatmap showing fraction of cells in each area bin per experiment.

    Bins: 0-5k, 5k-10k, 10k-25k, 25k-50k, 50k-100k, 100k+
    Immediately reveals which experiments are skewed toward larger cells.
    """
    sorted_df = _sort_df_by_ops_number(df)

    print("  Loading per-cell area data for heatmap...")
    per_cell = _load_per_cell_data_for_experiments(sorted_df, method=method)

    if not per_cell:
        print("  No per-cell data found, skipping area bin heatmap")
        return

    bin_edges = [0, 2500, 5000, 10000, 15000, 20000, 30000, 50000, 75000, 100000, 300000, 500000, np.inf]
    # 0.325 µm/pixel -> area conversion: 1 px² = 0.325² = 0.1056 µm²
    um2_per_px2 = 0.325 ** 2
    um_per_px = 0.325
    bin_labels = []
    for k in range(len(bin_edges) - 1):
        lo_px = bin_edges[k]
        hi_px = bin_edges[k + 1]
        lo_um2 = lo_px * um2_per_px2
        # Approximate diameter: treat bbox as square, side = sqrt(area_px) * um_per_px
        lo_diam = np.sqrt(lo_px) * um_per_px
        if np.isinf(hi_px):
            bin_labels.append(
                f"{lo_px / 1000:.0f}k+ px\n({lo_um2:.0f}+ \u00b5m\u00b2)\n(\u2265{lo_diam:.0f}\u00b5m diam)"
            )
        else:
            hi_um2 = hi_px * um2_per_px2
            hi_diam = np.sqrt(hi_px) * um_per_px
            bin_labels.append(
                f"{lo_px / 1000:.1f}-{hi_px / 1000:.0f}k px\n({lo_um2:.0f}-{hi_um2:.0f} \u00b5m\u00b2)\n(~{lo_diam:.0f}-{hi_diam:.0f}\u00b5m diam)"
            )

    experiments = sorted_df["experiment"].values
    n_cells_arr = (
        sorted_df["n_cells"].values
        if "n_cells" in sorted_df.columns
        else np.zeros(len(sorted_df))
    )

    rows = []
    count_rows = []
    exp_labels = []
    for exp, n_cells in zip(experiments, n_cells_arr):
        if exp not in per_cell:
            continue
        areas = per_cell[exp]
        counts, _ = np.histogram(areas, bins=bin_edges)
        fractions = counts / counts.sum() if counts.sum() > 0 else counts.astype(float)
        rows.append(fractions)
        count_rows.append(counts)
        exp_labels.append(f"{exp[:12]} ({int(n_cells / 1000)}k)")

    if not rows:
        return

    heatmap_data = np.array(rows)
    count_data = np.array(count_rows)

    # Compute per-column z-scores so coloring shows outliers within each bin
    col_means = heatmap_data.mean(axis=0)
    col_stds = heatmap_data.std(axis=0)
    col_stds[col_stds == 0] = 1  # avoid division by zero
    zscore_data = (heatmap_data - col_means) / col_stds

    # Symmetric clamp for diverging colormap
    z_abs_max = min(np.abs(zscore_data).max(), 3.0)

    fig, ax = plt.subplots(figsize=(18, max(8, len(rows) * 0.3)))

    im = ax.imshow(
        zscore_data, aspect="auto", cmap="RdBu_r",
        interpolation="nearest", vmin=-z_abs_max, vmax=z_abs_max,
    )

    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, fontsize=8)
    ax.set_yticks(range(len(exp_labels)))
    ax.set_yticklabels(exp_labels, fontsize=6)

    ax.set_xlabel("BBox Area Bin (px)", fontsize=12)
    ax.set_ylabel("Experiment (ordered by ops #)", fontsize=10)
    ax.set_title(
        "Cell Size Distribution Heatmap\n(z-score per bin column — red = more cells than typical, blue = fewer)",
        fontsize=13, fontweight="bold",
    )

    cbar = plt.colorbar(im, ax=ax, shrink=0.6, label="Z-score (within bin)")

    # Annotate cells with actual percentages for readability
    for i in range(len(rows)):
        for j in range(len(bin_labels)):
            frac = heatmap_data[i, j]
            z = zscore_data[i, j]
            if frac > 0:
                text_color = "white" if abs(z) > 1.5 else "black"
                n_cells_bin = int(count_data[i, j])
                if frac >= 0.01:
                    label_text = f"{frac:.0%}"
                else:
                    label_text = f"{frac:.2%}\n({n_cells_bin})"
                ax.text(
                    j, i, label_text,
                    ha="center", va="center", fontsize=5, color=text_color,
                )

    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_area_bins.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved heatmap_area_bins.png ({len(rows)} experiments)")


def create_violin_strip_plot(
    df: pd.DataFrame,
    analysis: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create violin + strip plot for key cell seg metrics."""
    metrics = [
        ("bbox_area_mean", "BBox Area Mean (px)"),
        ("bbox_height_mean", "BBox Height Mean (px)"),
        ("bbox_width_mean", "BBox Width Mean (px)"),
        ("bbox_aspect_ratio_mean", "BBox Aspect Ratio Mean"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (metric, label) in zip(axes, metrics):
        if metric not in df.columns:
            ax.set_visible(False)
            continue

        values = df[metric].dropna().values
        full_values = df[metric].values
        outlier_col = f"{metric}_outlier"
        is_outlier = (
            analysis[outlier_col].values
            if outlier_col in analysis.columns
            else np.zeros(len(df), dtype=bool)
        )

        # Only use non-NaN for violin
        valid_mask = ~np.isnan(full_values)
        valid_values = full_values[valid_mask]
        valid_outlier = is_outlier[valid_mask]
        valid_exps = df["experiment"].values[valid_mask]

        if len(valid_values) == 0:
            ax.set_visible(False)
            continue

        parts = ax.violinplot(
            [valid_values], positions=[0], showmeans=True, showmedians=True
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("steelblue")
            pc.set_alpha(0.3)

        normal_vals = valid_values[~valid_outlier]
        outlier_vals = valid_values[valid_outlier]
        outlier_exps = valid_exps[valid_outlier]

        jitter_normal = np.random.uniform(-0.15, 0.15, len(normal_vals))
        jitter_outlier = np.random.uniform(-0.15, 0.15, len(outlier_vals))

        ax.scatter(
            jitter_normal, normal_vals, c="steelblue", alpha=0.5, s=30, label="Normal"
        )
        ax.scatter(
            jitter_outlier,
            outlier_vals,
            c="red",
            alpha=0.9,
            s=80,
            marker="*",
            edgecolors="darkred",
            linewidth=0.5,
            label="Outlier",
            zorder=5,
        )

        for j, (x, y, exp) in enumerate(
            zip(jitter_outlier, outlier_vals, outlier_exps)
        ):
            ax.annotate(
                exp[:10],
                (x, y),
                fontsize=6,
                ha="left",
                va="bottom",
                xytext=(5, 2),
                textcoords="offset points",
                color="red",
            )

        mean = valid_values.mean()
        std = valid_values.std()
        ax.axhline(mean, color="green", linestyle="-", lw=1.5, alpha=0.7)
        if std > 0:
            ax.axhline(
                mean + 2.5 * std, color="orange", linestyle="--", lw=1, alpha=0.7
            )
            ax.axhline(
                mean - 2.5 * std, color="orange", linestyle="--", lw=1, alpha=0.7
            )

        ax.set_ylabel(label)
        ax.set_title(f"{label}", fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_xlim(-0.5, 0.5)

    plt.suptitle(
        "Cell Seg Metric Distributions with Outliers Highlighted\n"
        "(Violin + Strip Plot)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "violin_strip_cell_seg.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


def create_individual_metric_plots(
    df: pd.DataFrame,
    analysis: pd.DataFrame,
    output_dir: Path,
    figsize: tuple = (14, 10),
) -> None:
    """Create individual 4-panel plots for each metric."""
    metrics = []
    for col in _SHAPE_COLUMNS:
        col_label = col.replace("_", " ").title()
        metrics.append((f"{col}_mean", f"{col_label} Mean"))
        metrics.append((f"{col}_std", f"{col_label} Std"))

    for metric, label in metrics:
        if metric not in df.columns:
            continue

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(
            f"Distribution Analysis: {label}", fontsize=14, fontweight="bold"
        )

        values = df[metric]
        experiments = df["experiment"]
        n_cells_series = (
            df["n_cells"] if "n_cells" in df.columns else pd.Series([0] * len(df))
        )
        outlier_col = f"{metric}_outlier"
        is_outlier = (
            analysis[outlier_col]
            if outlier_col in analysis.columns
            else pd.Series([False] * len(df))
        )

        # Drop NaN for plotting
        valid = values.notna()
        values_valid = values[valid]
        experiments_valid = experiments[valid]
        is_outlier_valid = is_outlier[valid]

        if len(values_valid) == 0:
            plt.close()
            continue

        # 1. Histogram with KDE
        ax1 = axes[0, 0]
        ax1.hist(
            values_valid,
            bins=20,
            density=True,
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
        )
        if len(values_valid) > 1 and values_valid.std() > 0:
            try:
                kde_x = np.linspace(values_valid.min(), values_valid.max(), 100)
                kde = stats.gaussian_kde(values_valid)
                ax1.plot(kde_x, kde(kde_x), "r-", lw=2, label="KDE")
            except Exception:
                pass
        ax1.axvline(
            values_valid.mean(),
            color="green",
            linestyle="--",
            lw=2,
            label=f"Mean: {values_valid.mean():.1f}",
        )
        ax1.axvline(
            values_valid.median(),
            color="orange",
            linestyle="--",
            lw=2,
            label=f"Median: {values_valid.median():.1f}",
        )
        ax1.set_xlabel(label)
        ax1.set_ylabel("Density")
        ax1.set_title("Distribution")
        ax1.legend(fontsize=8)

        outlier_vals = values_valid[is_outlier_valid]
        outlier_exps = experiments_valid[is_outlier_valid]
        for val, exp in zip(outlier_vals, outlier_exps):
            ax1.axvline(val, color="red", linestyle="-", lw=1.5, alpha=0.7)

        # 2. Box plot with labeled outliers
        ax2 = axes[0, 1]
        bp = ax2.boxplot(values_valid, vert=True, patch_artist=True)
        bp["boxes"][0].set_facecolor("steelblue")
        bp["boxes"][0].set_alpha(0.7)
        if len(outlier_vals) > 0:
            jitter = np.random.uniform(0.9, 1.1, len(outlier_vals))
            ax2.scatter(
                jitter,
                outlier_vals,
                color="red",
                s=100,
                zorder=5,
                marker="*",
                label="Flagged",
            )
            for j, (x, y, exp) in enumerate(
                zip(jitter, outlier_vals, outlier_exps)
            ):
                ax2.annotate(
                    exp[:12],
                    (x, y),
                    fontsize=7,
                    ha="left",
                    va="bottom",
                    xytext=(5, 2),
                    textcoords="offset points",
                    color="red",
                    fontweight="bold",
                )
            ax2.legend()
        ax2.set_ylabel(label)
        ax2.set_title("Box Plot with Outliers Labeled")
        ax2.set_xticklabels(["All Experiments"])

        # 3. Bar chart sorted by value
        ax3 = axes[1, 0]
        sorted_idx = values_valid.argsort()
        sorted_vals = values_valid.iloc[sorted_idx]
        sorted_exps = experiments_valid.iloc[sorted_idx]
        sorted_outliers = is_outlier_valid.iloc[sorted_idx]

        colors = ["red" if o else "steelblue" for o in sorted_outliers]
        ax3.barh(range(len(sorted_vals)), sorted_vals, color=colors, alpha=0.7)

        outlier_indices = [i for i, o in enumerate(sorted_outliers) if o]
        n_labels = min(12, len(sorted_exps))
        step = max(1, len(sorted_exps) // n_labels)
        normal_indices = list(range(0, len(sorted_exps), step))
        all_labeled = sorted(set(outlier_indices + normal_indices))

        ax3.set_yticks(all_labeled)
        ytick_labels = [sorted_exps.iloc[i][:12] for i in all_labeled]
        ax3.set_yticklabels(ytick_labels, fontsize=7)

        for tick_label, idx in zip(ax3.get_yticklabels(), all_labeled):
            if sorted_outliers.iloc[idx]:
                tick_label.set_color("red")
                tick_label.set_fontweight("bold")

        ax3.set_xlabel(label)
        ax3.set_title("Experiments Ranked by Value (red = flagged)")
        ax3.axvline(
            values_valid.mean(), color="green", linestyle="--", lw=1.5, alpha=0.7
        )

        # 4. Z-score plot
        ax4 = axes[1, 1]
        z_col = f"{metric}_zscore"
        if z_col in analysis.columns:
            z_scores = analysis.loc[valid.values, z_col]
        else:
            z_scores = (
                (values_valid - values_valid.mean()) / values_valid.std()
                if values_valid.std() > 0
                else pd.Series([0] * len(values_valid))
            )
        sorted_z = z_scores.iloc[sorted_idx.values] if hasattr(sorted_idx, 'values') else z_scores.iloc[sorted_idx]

        colors = ["red" if o else "steelblue" for o in sorted_outliers]
        ax4.barh(range(len(sorted_z)), sorted_z, color=colors, alpha=0.7)
        ax4.axvline(0, color="black", linestyle="-", lw=1)
        ax4.axvline(-2.5, color="orange", linestyle="--", lw=1.5, alpha=0.7, label="z = -2.5")
        ax4.axvline(2.5, color="orange", linestyle="--", lw=1.5, alpha=0.7, label="z = +2.5")

        ax4.set_yticks(all_labeled)
        ax4.set_yticklabels(ytick_labels, fontsize=7)
        for tick_label, idx in zip(ax4.get_yticklabels(), all_labeled):
            if sorted_outliers.iloc[idx]:
                tick_label.set_color("red")
                tick_label.set_fontweight("bold")

        ax4.set_xlabel("Z-Score")
        ax4.set_title("Z-Scores (red = flagged)")
        ax4.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(
            output_dir / f"distribution_{metric}.png", dpi=150, bbox_inches="tight"
        )
        plt.close()


# === Report Generation ===


def generate_report(
    df: pd.DataFrame,
    analysis: pd.DataFrame,
    output_path: str | Path,
) -> str:
    """Generate a text report of cell segmentation QC analysis."""
    lines = []
    lines.append("=" * 80)
    lines.append("CELL SEGMENTATION QC REPORT")
    lines.append("Morphological Shape Analysis (regionprops)")
    lines.append("=" * 80)
    lines.append("")

    n_experiments = len(df)
    anomalous = analysis[analysis["is_anomalous"]]
    n_anomalous = len(anomalous)

    lines.append(f"Total experiments analyzed: {n_experiments}")
    lines.append(
        f"Experiments with distribution shifts: {n_anomalous} "
        f"({100 * n_anomalous / n_experiments:.1f}%)"
    )
    lines.append("")

    # Global statistics
    lines.append("-" * 80)
    lines.append("GLOBAL STATISTICS (Reference)")
    lines.append("-" * 80)

    for col in _SHAPE_COLUMNS:
        mean_col = f"{col}_mean"
        if mean_col in df.columns:
            vals = df[mean_col].dropna()
            lines.append(
                f"  {mean_col:35s}: mean={vals.mean():10.2f}, "
                f"median={vals.median():10.2f}, std={vals.std():10.2f}"
            )
    lines.append("")

    # Flagged experiments
    if n_anomalous > 0:
        lines.append("-" * 80)
        lines.append("FLAGGED EXPERIMENTS (Distribution Shifts Detected)")
        lines.append("-" * 80)
        lines.append("")

        for _, row in anomalous.iterrows():
            exp = row["experiment"]
            n_cells = row.get("n_cells", "N/A")
            flags = row["total_outlier_flags"]
            lines.append(f"EXPERIMENT: {exp} ({n_cells} cells)")
            lines.append(f"  Total flags: {flags}")

            flagged_metrics = []
            for c in analysis.columns:
                if c.endswith("_outlier") and row[c]:
                    metric_name = c.replace("_outlier", "")
                    direction = row.get(f"{metric_name}_direction", "")
                    value = row.get(f"{metric_name}_value", "")
                    zscore = row.get(f"{metric_name}_zscore", "")
                    if isinstance(value, (int, float)) and isinstance(
                        zscore, (int, float)
                    ):
                        flagged_metrics.append(
                            f"    - {metric_name}: {value:.2f} "
                            f"(z={zscore:.2f}, {direction})"
                        )

            lines.append("  Flagged metrics:")
            lines.extend(flagged_metrics)
            lines.append("")
    else:
        lines.append("No significant distribution shifts detected.")
        lines.append("")

    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    report_content = "\n".join(lines)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_content)

    return report_content


# === Main Orchestration ===


def run_full_analysis(
    output_dir: str | Path = "tests/QC/cell_seg_qc",
    method: str = "mine",
    force: bool = False,
    verbose: bool = False,
    n_jobs: int = None,
    z_threshold: float = 2.5,
    mod_z_threshold: float = 3.5,
    iqr_multiplier: float = 1.5,
) -> dict:
    """
    Run complete cell segmentation QC analysis.

    Args:
        output_dir: Directory for QC output files (plots, reports)
        method: Base calling method (default: mine)
        force: Force regeneration of per-experiment metrics
        verbose: Print detailed progress
        n_jobs: Number of parallel workers (auto-detected if None)
        z_threshold: Z-score threshold for outlier detection
        mod_z_threshold: Modified z-score threshold
        iqr_multiplier: IQR multiplier

    Returns:
        Dict with analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print SLURM header if running inside a SLURM job
    if os.environ.get("SLURM_JOB_ID"):
        print_slurm_job_header(title="Cell Seg QC")

    print("\n" + "=" * 100)
    print("CELL SEGMENTATION QC - Morphological Shape Analysis")
    print(f"Method: {method}")
    print("=" * 100 + "\n")

    # Collect stats from all experiments
    df = collect_all_experiment_stats(
        method=method, force=force, verbose=verbose, n_jobs=n_jobs
    )

    if len(df) == 0:
        print("No experiments with cell segmentation data found!")
        return {}

    # Save raw stats
    df.to_csv(output_dir / "cell_seg_stats.csv", index=False)

    print("Running distribution shift analysis...")
    analysis = analyze_distribution_shifts(
        df,
        z_threshold=z_threshold,
        mod_z_threshold=mod_z_threshold,
        iqr_multiplier=iqr_multiplier,
    )

    print("Generating ridge plots (one per metric, ordered by ops #)...")
    create_ridge_plots(df, analysis, output_dir)

    print("Generating per-cell area distribution ridge plot...")
    create_area_distribution_ridge_plot(df, analysis, output_dir, method=method)

    print("Generating area bin heatmap...")
    create_area_bin_heatmap(df, output_dir, method=method)

    print("Generating violin/strip plots...")
    create_violin_strip_plot(df, analysis, output_dir)

    print("Generating individual metric plots...")
    create_individual_metric_plots(df, analysis, output_dir)

    print("Generating report...")
    report_path = output_dir / "cell_seg_qc_report.txt"
    report = generate_report(df, analysis, report_path)
    print(report)

    print("Saving analysis CSV...")
    analysis_path = output_dir / "cell_seg_analysis.csv"
    analysis.to_csv(analysis_path, index=False)

    n_anomalous = analysis["is_anomalous"].sum()
    print(f"\n{'=' * 60}")
    print("Analysis complete!")
    print(f"  Experiments analyzed: {len(df)}")
    print(f"  Experiments with shifts: {n_anomalous}")
    print(f"  Output directory: {output_dir}")
    print(f"{'=' * 60}")

    return {
        "df": df,
        "analysis": analysis,
        "output_dir": output_dir,
        "report_path": report_path,
        "analysis_csv_path": analysis_path,
        "n_anomalous": n_anomalous,
    }


def submit_slurm_job(
    output_dir: str = "tests/QC/cell_seg_qc",
    method: str = "mine",
    force: bool = False,
    verbose: bool = False,
    n_jobs: int | None = None,
    z_threshold: float = 2.5,
    iqr_multiplier: float = 1.5,
    mem_gb: int = 256,
    cpus: int = 64,
    timeout_min: int = 5,
    partition: str = "cpu",
    wait_for_completion: bool = True,
    dry_run: bool = False,
) -> dict:
    """Submit qc_cell_seg as a SLURM job with sufficient memory.

    Uses submit_parallel_jobs for consistent monitoring, manifests, and
    resource reporting matching other pipeline scripts.

    Returns submit_parallel_jobs result dict.
    """
    slurm_params = {
        "timeout_min": timeout_min,
        "mem": f"{mem_gb}GB",
        "cpus_per_task": cpus,
        "slurm_partition": partition,
        "name": "qc_cell_seg",
    }

    jobs_to_submit = [
        {
            "name": "qc_cell_seg",
            "func": run_full_analysis,
            "kwargs": {
                "output_dir": output_dir,
                "method": method,
                "force": force,
                "verbose": verbose,
                "n_jobs": n_jobs,
                "z_threshold": z_threshold,
                "iqr_multiplier": iqr_multiplier,
            },
            "metadata": {
                "type": "qc_cell_seg",
                "method": method,
                "output_dir": output_dir,
            },
        }
    ]

    result = submit_parallel_jobs(
        jobs_to_submit=jobs_to_submit,
        experiment="qc_cell_seg",
        slurm_params=slurm_params,
        log_dir="slurm_cell_size",
        manifest_prefix="qc_cell_seg",
        dry_run=dry_run,
        wait_for_completion=wait_for_completion,
        verbose=True,
    )

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cell segmentation QC - morphological shape analysis across experiments"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="tests/QC/cell_seg_qc",
        help="Output directory for plots and reports",
    )
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        default="mine",
        choices=["mine", "probabilistic"],
        help="Base calling method (default: mine)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of per-experiment cell size metrics",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed progress for each experiment",
    )
    parser.add_argument(
        "--n-jobs",
        "-j",
        type=int,
        default=None,
        help="Number of parallel workers (auto-detected if not set)",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=2.5,
        help="Z-score threshold for outlier detection",
    )
    parser.add_argument(
        "--iqr-multiplier",
        type=float,
        default=1.5,
        help="IQR multiplier for outlier detection",
    )
    # SLURM submission options
    slurm_group = parser.add_argument_group("SLURM options")
    slurm_group.add_argument(
        "--slurm",
        action="store_true",
        help="Submit as a SLURM job instead of running locally",
    )
    slurm_group.add_argument(
        "--mem-gb",
        type=int,
        default=256,
        help="Memory in GB for SLURM job (default: 256)",
    )
    slurm_group.add_argument(
        "--cpus",
        type=int,
        default=64,
        help="CPUs for SLURM job (default: 64)",
    )
    slurm_group.add_argument(
        "--timeout",
        type=int,
        default=5,
        help="Timeout in minutes for SLURM job (default: 5)",
    )
    slurm_group.add_argument(
        "--partition",
        type=str,
        default="cpu",
        help="SLURM partition (default: cpu)",
    )
    slurm_group.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit SLURM job and return immediately without waiting",
    )
    slurm_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be submitted without actually submitting",
    )

    args = parser.parse_args()

    if args.slurm:
        result = submit_slurm_job(
            output_dir=args.output_dir,
            method=args.method,
            force=args.force,
            verbose=args.verbose,
            n_jobs=args.n_jobs,
            z_threshold=args.z_threshold,
            iqr_multiplier=args.iqr_multiplier,
            mem_gb=args.mem_gb,
            cpus=args.cpus,
            timeout_min=args.timeout,
            partition=args.partition,
            wait_for_completion=not args.no_wait,
            dry_run=args.dry_run,
        )
        if result.get("dry_run"):
            sys.exit(0)
        elif result.get("success"):
            if result.get("all_completed") is not None:
                sys.exit(0 if result.get("all_completed") else 1)
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        run_full_analysis(
            output_dir=args.output_dir,
            method=args.method,
            force=args.force,
            verbose=args.verbose,
            n_jobs=args.n_jobs,
            z_threshold=args.z_threshold,
            iqr_multiplier=args.iqr_multiplier,
        )
