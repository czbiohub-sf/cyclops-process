"""
Spatial Coherence Analysis - Sister Cell Proximity
===================================================

Analyzes sister cell (matching sgRNA) spatial clustering within experiments.

For each cell in linked_results CSV:
1. Build KDTree from pheno coordinates (per well)
2. Find all neighbors within configurable radius (default 500px)
3. Count neighbors with matching sgRNA (sister count)
4. Calculate sister_ratio = sister_count / neighbor_count

Generates:
- Per-cell CSV with neighbor_count, sister_count, sister_ratio
- Per-sgRNA statistics CSV (mean, median, std, min, max, sum)
- Ridge plots showing spatial coherence distributions across experiments
- Violin/strip plots for key metrics
- Summary report of detected shifts

Per-experiment outputs stored in: fast_ops/{exp}/3-assembly/spatial_coherence/
QC plots stored in: tests/QC/spatial_coherence/
"""

import sys
import os
import re
import time
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.spatial import cKDTree
from typing import Optional
from tqdm import tqdm
from joblib import Parallel, delayed

sys.path.insert(0, os.getcwd())

from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.hpc.resource_manager import get_optimal_workers


# === Core Analysis ===


def compute_spatial_coherence_for_well(
    df: pd.DataFrame,
    radius: float = 500.0,
) -> pd.DataFrame:
    """
    Compute per-cell spatial coherence metrics for a single well.

    Uses a sparse distance matrix from cKDTree for fully vectorized counting.

    Args:
        df: DataFrame with y_pheno, x_pheno, sgRNA columns
        radius: Search radius in pixels

    Returns:
        Input DataFrame with added columns: neighbor_count, sister_count, sister_ratio
    """
    coords = df[["y_pheno", "x_pheno"]].values.astype(np.float64)
    tree = cKDTree(coords)

    # Sparse distance matrix for all pairs within radius
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

    # Total neighbor count per cell
    neighbor_counts = np.bincount(row, minlength=n_cells)

    # Sister count: pairs where sgRNA matches
    sgrna_values = df["sgRNA"].values
    sgrna_matches = sgrna_values[row] == sgrna_values[col]
    sister_counts = np.bincount(row[sgrna_matches], minlength=n_cells)

    df = df.copy()
    df["neighbor_count"] = neighbor_counts
    df["sister_count"] = sister_counts
    df["sister_ratio"] = np.where(
        neighbor_counts > 0,
        sister_counts / neighbor_counts,
        0.0,
    )

    return df


def analyze_experiment_spatial_coherence(
    dataset: OpsDataset,
    exp_name: str,
    radius: float = 500.0,
    verbose: bool = False,
) -> dict | None:
    """
    Analyze spatial coherence for a single experiment.

    Loads linked_results for all available wells, computes per-cell metrics,
    saves per-cell and per-sgRNA CSVs to fast_ops/{exp}/3-assembly/spatial_coherence/.

    Returns summary statistics dict or None if no data found.
    """
    t_start = time.time()

    # Find available linked_results files (try fast partition first)
    link_dir = dataset.results_fast
    link_files = sorted(link_dir.glob("*_linked_pheno_iss.csv"))

    if not link_files:
        link_dir = dataset.results
        link_files = sorted(link_dir.glob("*_linked_pheno_iss.csv"))

    if not link_files:
        if verbose:
            print(f"  No linked results found for {exp_name}")
        return None

    all_well_dfs = []

    for link_path in link_files:
        try:
            df = pd.read_csv(link_path)
        except Exception as e:
            if verbose:
                print(f"  Error reading {link_path}: {e}")
            continue

        if len(df) == 0:
            continue

        # Verify required columns
        required_cols = ["y_pheno", "x_pheno", "sgRNA"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            if verbose:
                print(f"  Missing columns {missing} in {link_path.name}")
            continue

        # Drop rows with NaN in required columns
        df = df.dropna(subset=["y_pheno", "x_pheno", "sgRNA"])
        if len(df) == 0:
            continue

        # Extract well token from filename (e.g., "A1_linked_pheno_iss.csv" -> "A1")
        well_token = link_path.name.split("_linked_pheno_iss.csv")[0]

        if verbose:
            print(f"  Processing {well_token}: {len(df):,} cells")

        # Compute spatial coherence per well (each well has its own coordinate space)
        df = compute_spatial_coherence_for_well(df, radius=radius)
        df["well"] = well_token
        all_well_dfs.append(df)

    if not all_well_dfs:
        return None

    df_all = pd.concat(all_well_dfs, ignore_index=True)

    if len(df_all) == 0:
        return None

    # Save per-cell CSV
    output_dir = dataset.results_fast / "spatial_coherence"
    output_dir.mkdir(parents=True, exist_ok=True)

    per_cell_path = output_dir / "per_cell_spatial_coherence.csv"
    df_all.to_csv(per_cell_path, index=False)

    # Per-sgRNA aggregation
    sgrna_stats = (
        df_all.groupby("sgRNA")
        .agg(
            n_cells=("sister_count", "count"),
            sister_count_mean=("sister_count", "mean"),
            sister_count_median=("sister_count", "median"),
            sister_count_std=("sister_count", "std"),
            sister_count_min=("sister_count", "min"),
            sister_count_max=("sister_count", "max"),
            sister_count_sum=("sister_count", "sum"),
            sister_ratio_mean=("sister_ratio", "mean"),
            sister_ratio_median=("sister_ratio", "median"),
            sister_ratio_std=("sister_ratio", "std"),
            sister_ratio_min=("sister_ratio", "min"),
            sister_ratio_max=("sister_ratio", "max"),
            sister_ratio_sum=("sister_ratio", "sum"),
        )
        .reset_index()
    )

    sgrna_stats_path = output_dir / "per_sgRNA_spatial_stats.csv"
    sgrna_stats.to_csv(sgrna_stats_path, index=False)

    elapsed = time.time() - t_start

    if verbose:
        print(
            f"  {exp_name}: {len(df_all):,} cells, "
            f"{df_all['sgRNA'].nunique()} sgRNAs, {elapsed:.1f}s"
        )

    # Return experiment-level summary
    return {
        "experiment": exp_name,
        "n_cells": len(df_all),
        "n_wells": df_all["well"].nunique(),
        "n_sgRNAs": df_all["sgRNA"].nunique(),
        "mean_neighbor_count": float(df_all["neighbor_count"].mean()),
        "median_neighbor_count": float(df_all["neighbor_count"].median()),
        "std_neighbor_count": float(df_all["neighbor_count"].std()),
        "mean_sister_count": float(df_all["sister_count"].mean()),
        "median_sister_count": float(df_all["sister_count"].median()),
        "std_sister_count": float(df_all["sister_count"].std()),
        "mean_sister_ratio": float(df_all["sister_ratio"].mean()),
        "median_sister_ratio": float(df_all["sister_ratio"].median()),
        "std_sister_ratio": float(df_all["sister_ratio"].std()),
        "pct_isolated": float(
            100 * (df_all["neighbor_count"] == 0).sum() / len(df_all)
        ),
        "pct_no_sisters": float(
            100 * (df_all["sister_count"] == 0).sum() / len(df_all)
        ),
    }


def _process_single_experiment(
    config_path: Path,
    radius: float,
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

        dataset = OpsDataset(exp_name)

        stats = analyze_experiment_spatial_coherence(
            dataset, exp_name, radius=radius, verbose=verbose
        )

        return stats

    except Exception as e:
        print(f"\nError processing {config_path.stem}: {e}")
        return None


def collect_all_experiment_stats(
    radius: float = 500.0,
    verbose: bool = False,
    n_jobs: int = None,
) -> pd.DataFrame:
    """
    Collect spatial coherence statistics from all experiments using OpsDataset.

    Uses joblib for parallel processing across experiments.

    Args:
        radius: Search radius in pixels
        verbose: Print detailed progress
        n_jobs: Number of parallel workers (auto-detected if None)

    Returns DataFrame with one row per experiment.
    """
    dummy_dataset = OpsDataset("dummy")
    config_root_dir = dummy_dataset.config_paths["exp_config_dir"]

    if not config_root_dir.is_dir():
        print(f"Configuration directory not found at {config_root_dir}. Aborting.")
        return pd.DataFrame()

    config_files = sorted(list(config_root_dir.glob("*_config.yaml")))

    # Determine number of workers
    if n_jobs is None:
        n_jobs = get_optimal_workers(use_gpu=False, verbose=False)
        n_jobs = max(1, n_jobs)

    print(f"Processing {len(config_files)} configs with {n_jobs} workers")

    # Parallel processing across experiments
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_process_single_experiment)(config_path, radius, verbose)
        for config_path in tqdm(config_files, desc="Analyzing experiments")
    )

    # Filter out None results
    all_stats = [r for r in results if r is not None]
    n_no_data = len(results) - len(all_stats)

    if n_no_data > 0:
        print(f"\n{n_no_data} experiments skipped (non-standard or no linked data)")
    print(f"\nAnalyzed {len(all_stats)} experiments with spatial coherence data\n")

    return pd.DataFrame(all_stats)


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
        metrics = [
            "mean_sister_count",
            "median_sister_count",
            "std_sister_count",
            "mean_sister_ratio",
            "median_sister_ratio",
            "std_sister_ratio",
            "mean_neighbor_count",
            "pct_isolated",
            "pct_no_sisters",
        ]

    metrics = [m for m in metrics if m in df.columns]
    results = df[["experiment"]].copy()
    if "n_cells" in df.columns:
        results["n_cells"] = df["n_cells"]

    for metric in metrics:
        series = df[metric]

        z_outlier = calculate_zscore_outliers(series, z_threshold)
        mod_z_outlier = calculate_modified_zscore_outliers(series, mod_z_threshold)
        iqr_outlier = calculate_iqr_outliers(series, iqr_multiplier)

        mean = series.mean()
        std = series.std()
        z_scores = (
            (series - mean) / std if std > 0 else pd.Series([0] * len(series))
        )

        outlier_count = (
            z_outlier.astype(int)
            + mod_z_outlier.astype(int)
            + iqr_outlier.astype(int)
        )
        combined_outlier = outlier_count >= 2

        results[f"{metric}_value"] = series
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


def create_ridge_plot(
    df: pd.DataFrame,
    analysis: pd.DataFrame,
    output_dir: Path,
    metrics: list[tuple[str, str]],
    overlap: float = 0.6,
) -> None:
    """Create ridge plot showing spatial coherence metrics across experiments."""
    n_metrics = len([m for m, _ in metrics if m in df.columns])
    if n_metrics == 0:
        return

    fig, axes = plt.subplots(1, n_metrics, figsize=(7 * n_metrics, 14))

    if n_metrics == 1:
        axes = [axes]

    # Sort by mean_sister_ratio for consistent ordering
    sort_col = (
        "mean_sister_ratio" if "mean_sister_ratio" in df.columns else df.columns[1]
    )
    sorted_df = df.sort_values(sort_col).reset_index(drop=True)
    sorted_analysis = (
        analysis.set_index("experiment").loc[sorted_df["experiment"]].reset_index()
    )

    n_experiments = len(sorted_df)

    for ax, (metric, label) in zip(axes, metrics):
        if metric not in df.columns:
            continue

        values = sorted_df[metric].values
        experiments = sorted_df["experiment"].values
        n_cells_arr = (
            sorted_df["n_cells"].values
            if "n_cells" in sorted_df.columns
            else np.zeros(n_experiments)
        )
        outlier_col = f"{metric}_outlier"
        is_outlier = (
            sorted_analysis[outlier_col].values
            if outlier_col in sorted_analysis.columns
            else np.zeros(n_experiments, dtype=bool)
        )

        global_mean = values.mean()
        global_std = values.std()

        val_range = values.max() - values.min()
        x_min = values.min() - 0.15 * (val_range + 0.1)
        x_max = values.max() + 0.15 * (val_range + 0.1)
        x_range = np.linspace(x_min, x_max, 200)

        row_height = 1.0

        for i, (val, exp, n_cells, outlier) in enumerate(
            zip(values, experiments, n_cells_arr, is_outlier)
        ):
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
            exp_label = f"{exp[:12]} ({int(n_cells/1000)}k)"
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
            label=f"Mean: {global_mean:.4f}",
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
        ax.set_ylabel("Experiments (sorted by sister ratio)", fontsize=10)
        ax.set_title(f"{label}\n(red = outlier)", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_yticks([])
        ax.set_xlim(x_min - 0.2 * (x_max - x_min), x_max)

    plt.suptitle(
        "Spatial Coherence Distribution Across Experiments\n"
        "(Ridge Plot - Each row is one experiment)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "ridge_plot_spatial_coherence.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


def create_violin_strip_plot(
    df: pd.DataFrame,
    analysis: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create violin + strip plot for key spatial coherence metrics."""
    metrics = [
        ("mean_sister_count", "Mean Sister Count"),
        ("mean_sister_ratio", "Mean Sister Ratio"),
        ("pct_no_sisters", "% Cells with No Sisters"),
        ("mean_neighbor_count", "Mean Neighbor Count"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (metric, label) in zip(axes, metrics):
        if metric not in df.columns:
            ax.set_visible(False)
            continue

        values = df[metric].values
        outlier_col = f"{metric}_outlier"
        is_outlier = (
            analysis[outlier_col].values
            if outlier_col in analysis.columns
            else np.zeros(len(df), dtype=bool)
        )

        parts = ax.violinplot(
            [values], positions=[0], showmeans=True, showmedians=True
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("steelblue")
            pc.set_alpha(0.3)

        normal_vals = values[~is_outlier]
        outlier_vals = values[is_outlier]
        outlier_exps = df["experiment"].values[is_outlier]

        jitter_normal = np.random.uniform(-0.15, 0.15, len(normal_vals))
        jitter_outlier = np.random.uniform(-0.15, 0.15, len(outlier_vals))

        ax.scatter(
            jitter_normal,
            normal_vals,
            c="steelblue",
            alpha=0.5,
            s=30,
            label="Normal",
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

        mean = values.mean()
        std = values.std()
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
        "Spatial Coherence Metric Distributions with Outliers Highlighted\n"
        "(Violin + Strip Plot)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "violin_strip_spatial_coherence.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


def create_individual_metric_plots(
    df: pd.DataFrame,
    analysis: pd.DataFrame,
    output_dir: Path,
    figsize: tuple = (14, 10),
) -> None:
    """Create individual 4-panel plots for each metric."""
    metrics = [
        ("mean_sister_count", "Mean Sister Count"),
        ("mean_sister_ratio", "Mean Sister Ratio"),
        ("std_sister_ratio", "Std Dev Sister Ratio"),
        ("pct_no_sisters", "% Cells with No Sisters"),
        ("pct_isolated", "% Isolated Cells (No Neighbors)"),
        ("mean_neighbor_count", "Mean Neighbor Count"),
    ]

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

        # 1. Histogram with KDE
        ax1 = axes[0, 0]
        ax1.hist(
            values,
            bins=20,
            density=True,
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
        )
        if len(values) > 1 and values.std() > 0:
            try:
                kde_x = np.linspace(values.min(), values.max(), 100)
                kde = stats.gaussian_kde(values)
                ax1.plot(kde_x, kde(kde_x), "r-", lw=2, label="KDE")
            except Exception:
                pass
        ax1.axvline(
            values.mean(),
            color="green",
            linestyle="--",
            lw=2,
            label=f"Mean: {values.mean():.4f}",
        )
        ax1.axvline(
            values.median(),
            color="orange",
            linestyle="--",
            lw=2,
            label=f"Median: {values.median():.4f}",
        )
        ax1.set_xlabel(label)
        ax1.set_ylabel("Density")
        ax1.set_title("Distribution")
        ax1.legend(fontsize=8)

        outlier_vals = values[is_outlier]
        outlier_exps = experiments[is_outlier]
        for val, exp in zip(outlier_vals, outlier_exps):
            ax1.axvline(val, color="red", linestyle="-", lw=1.5, alpha=0.7)

        # 2. Box plot with labeled outliers
        ax2 = axes[0, 1]
        bp = ax2.boxplot(values, vert=True, patch_artist=True)
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
        sorted_idx = values.argsort()
        sorted_vals = values.iloc[sorted_idx]
        sorted_exps = experiments.iloc[sorted_idx]
        sorted_ncells = n_cells_series.iloc[sorted_idx]
        sorted_outliers = is_outlier.iloc[sorted_idx]

        colors = ["red" if o else "steelblue" for o in sorted_outliers]
        ax3.barh(range(len(sorted_vals)), sorted_vals, color=colors, alpha=0.7)

        outlier_indices = [i for i, o in enumerate(sorted_outliers) if o]
        n_labels = min(12, len(sorted_exps))
        step = max(1, len(sorted_exps) // n_labels)
        normal_indices = list(range(0, len(sorted_exps), step))
        all_labeled = sorted(set(outlier_indices + normal_indices))

        ax3.set_yticks(all_labeled)
        ytick_labels = [
            f"{sorted_exps.iloc[i][:10]} ({int(sorted_ncells.iloc[i] / 1000)}k)"
            for i in all_labeled
        ]
        ax3.set_yticklabels(ytick_labels, fontsize=7)

        for tick_label, idx in zip(ax3.get_yticklabels(), all_labeled):
            if sorted_outliers.iloc[idx]:
                tick_label.set_color("red")
                tick_label.set_fontweight("bold")

        ax3.set_xlabel(label)
        ax3.set_title("Experiments Ranked by Value (red = flagged)")
        ax3.axvline(values.mean(), color="green", linestyle="--", lw=1.5, alpha=0.7)

        # 4. Z-score plot
        ax4 = axes[1, 1]
        z_scores = (
            analysis[f"{metric}_zscore"]
            if f"{metric}_zscore" in analysis.columns
            else (values - values.mean()) / values.std()
        )
        sorted_z = z_scores.iloc[sorted_idx]

        colors = ["red" if o else "steelblue" for o in sorted_outliers]
        ax4.barh(range(len(sorted_z)), sorted_z, color=colors, alpha=0.7)
        ax4.axvline(0, color="black", linestyle="-", lw=1)
        ax4.axvline(
            -2.5,
            color="orange",
            linestyle="--",
            lw=1.5,
            alpha=0.7,
            label="z = -2.5",
        )
        ax4.axvline(
            2.5,
            color="orange",
            linestyle="--",
            lw=1.5,
            alpha=0.7,
            label="z = +2.5",
        )

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
    """Generate a text report of spatial coherence analysis."""
    lines = []
    lines.append("=" * 80)
    lines.append("SPATIAL COHERENCE ANALYSIS REPORT")
    lines.append("Sister Cell (sgRNA) Proximity Analysis")
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

    key_metrics = [
        "mean_sister_count",
        "mean_sister_ratio",
        "mean_neighbor_count",
        "pct_no_sisters",
        "pct_isolated",
    ]
    for metric in key_metrics:
        if metric in df.columns:
            vals = df[metric]
            lines.append(
                f"  {metric:30s}: mean={vals.mean():10.4f}, "
                f"median={vals.median():10.4f}, std={vals.std():10.4f}"
            )
    lines.append("")

    # Cell count breakdown
    lines.append("-" * 80)
    lines.append("CELL COUNT BREAKDOWN")
    lines.append("-" * 80)
    if "n_cells" in df.columns:
        for _, row in df.sort_values("n_cells", ascending=False).iterrows():
            lines.append(
                f"  {row['experiment']:25s}: {int(row['n_cells']):>10,} cells, "
                f"sister_ratio={row['mean_sister_ratio']:.4f}"
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
            for col in analysis.columns:
                if col.endswith("_outlier") and row[col]:
                    metric_name = col.replace("_outlier", "")
                    direction = row.get(f"{metric_name}_direction", "")
                    value = row.get(f"{metric_name}_value", "")
                    zscore = row.get(f"{metric_name}_zscore", "")
                    if isinstance(value, (int, float)) and isinstance(
                        zscore, (int, float)
                    ):
                        flagged_metrics.append(
                            f"    - {metric_name}: {value:.4f} "
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
    output_dir: str | Path = "tests/QC/spatial_coherence",
    radius: float = 500.0,
    verbose: bool = False,
    n_jobs: int = None,
    z_threshold: float = 2.5,
    mod_z_threshold: float = 3.5,
    iqr_multiplier: float = 1.5,
) -> dict:
    """
    Run complete spatial coherence analysis.

    Args:
        output_dir: Directory for QC output files (plots, reports)
        radius: Search radius in pixels for neighbor detection
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

    print("\n" + "=" * 100)
    print("SPATIAL COHERENCE ANALYSIS - Sister Cell (sgRNA) Proximity")
    print(f"Search radius: {radius} pixels")
    print("=" * 100 + "\n")

    # Collect stats from all experiments
    df = collect_all_experiment_stats(radius=radius, verbose=verbose, n_jobs=n_jobs)

    if len(df) == 0:
        print("No experiments with linked data found!")
        return {}

    # Save raw stats
    df.to_csv(output_dir / "spatial_coherence_stats.csv", index=False)

    print("Running distribution shift analysis...")
    analysis = analyze_distribution_shifts(
        df,
        z_threshold=z_threshold,
        mod_z_threshold=mod_z_threshold,
        iqr_multiplier=iqr_multiplier,
    )

    print("Generating ridge plot...")
    ridge_metrics = [
        ("mean_sister_ratio", "Mean Sister Ratio"),
        ("mean_sister_count", "Mean Sister Count"),
        ("pct_no_sisters", "% Cells with No Sisters"),
    ]
    create_ridge_plot(df, analysis, output_dir, ridge_metrics)

    print("Generating violin/strip plots...")
    create_violin_strip_plot(df, analysis, output_dir)

    print("Generating individual metric plots...")
    create_individual_metric_plots(df, analysis, output_dir)

    print("Generating report...")
    report_path = output_dir / "spatial_coherence_report.txt"
    report = generate_report(df, analysis, report_path)
    print(report)

    print("Saving analysis CSV...")
    analysis_path = output_dir / "spatial_coherence_analysis.csv"
    analysis.to_csv(analysis_path, index=False)

    n_anomalous = analysis["is_anomalous"].sum()
    print(f"\n{'=' * 60}")
    print(f"Analysis complete!")
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze spatial coherence (sister cell proximity) across experiments"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="tests/QC/spatial_coherence",
        help="Output directory for plots and reports",
    )
    parser.add_argument(
        "--radius",
        "-r",
        type=float,
        default=500.0,
        help="Search radius in pixels (default: 500)",
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
        help="Number of parallel workers (auto-detected from resource_manager if not set)",
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

    args = parser.parse_args()

    run_full_analysis(
        output_dir=args.output_dir,
        radius=args.radius,
        verbose=args.verbose,
        n_jobs=args.n_jobs,
        z_threshold=args.z_threshold,
        iqr_multiplier=args.iqr_multiplier,
    )
    # To run: python tests/QC/spatial_coherence_analysis.py
    # To run with verbose: python tests/QC/spatial_coherence_analysis.py -v
    # To run with custom radius: python tests/QC/spatial_coherence_analysis.py -r 300
