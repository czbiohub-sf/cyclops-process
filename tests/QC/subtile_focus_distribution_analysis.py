"""
Subtile Focus Distribution Shift Detection
===========================================




Analyzes subtile focus statistics across experiments to detect distribution shifts
and identify experiments where Z-stack range may be insufficient.

Uses multiple statistical methods to identify outliers:
1. Z-score method for mean/median shifts
2. Modified Z-score for robust outlier detection
3. IQR (Interquartile Range) method

Generates:
- Ridge plots showing focus distribution across experiments
- Summary statistics with outlier flags
- Edge clipping analysis
- Detailed report of detected shifts

Usage:
python tests/QC/subtile_focus_distribution_analysis.py
python tests/QC/subtile_focus_distribution_analysis.py --output-dir tests/subtile_focus_distribution_analysis
python tests/QC/subtile_focus_distribution_analysis.py --output-dir tests/subtile_focus_distribution_analysis --metadata-type track
python tests/QC/subtile_focus_distribution_analysis.py --output-dir tests/subtile_focus_distribution_analysis --metadata-type pheno
python tests/QC/subtile_focus_distribution_analysis.py --output-dir tests/subtile_focus_distribution_analysis --metadata-type track --skip-low-count-exps
python tests/QC/subtile_focus_distribution_analysis.py --output-dir tests/subtile_focus_distribution_analysis --metadata-type pheno --skip-low-count-exps
python tests/QC/subtile_focus_distribution_analysis.py --output-dir tests/subtile_focus_distribution_analysis --metadata-type track --skip-low-count-exps --verbose-timing
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
from typing import Optional
from tqdm import tqdm

sys.path.insert(0, os.getcwd())

from ops_utils.data.experiment import OpsDataset
from ops_utils.data.bad_experiments import is_excluded
from cyclops_process.paths import BASE_PATH


def analyze_experiment_subtile_focus(
    dataset: OpsDataset,
    exp_name: str,
    metadata_type: str = "pheno",
    verbose: bool = False,
) -> dict | None:
    """
    Analyze subtile focus statistics for a single experiment.

    Args:
        dataset: OpsDataset instance for the experiment
        exp_name: Experiment name
        metadata_type: "pheno" or "track" for phenotyping or tracking
        verbose: Print timing info

    Returns:
        Dict with statistics or None if no metadata found
    """
    timings = {}
    t_start = time.time()

    # Find the subtile metadata CSV, checking new (fast_ops) and old (ops) partitions
    t0 = time.time()
    pattern = f"{exp_name}_{metadata_type}-2d_subtile_metadata.csv"
    glob_pattern = f"*{metadata_type}-2d_subtile_metadata.csv"
    metadata_path = None

    recon_subpath = Path("1-preprocess/live_imaging/reconstruction")
    base_dirs = [Path(BASE_PATH)]

    for base_dir in base_dirs:
        recon_dir = base_dir / exp_name / recon_subpath
        candidate = recon_dir / pattern
        if candidate.exists():
            metadata_path = candidate
            break
        if recon_dir.exists():
            alt = list(recon_dir.glob(glob_pattern))
            if alt:
                metadata_path = alt[0]
                break

    if metadata_path is None:
        return None

    timings['find_file'] = time.time() - t0

    # Load the CSV
    t0 = time.time()
    try:
        df = pd.read_csv(metadata_path)
        if len(df) == 0:
            return None
    except Exception as e:
        if verbose:
            print(f"  Warning: Error reading {metadata_path}: {e}")
        return None
    timings['read_csv'] = time.time() - t0

    # Calculate statistics
    t0 = time.time()
    z_size = int(df['z_stack_size'].iloc[0])
    middle = z_size // 2
    max_idx = z_size - 1
    n_subtiles = len(df)

    # Edge analysis
    at_bottom = int((df['focus_index'] == 0).sum())
    at_top = int((df['focus_index'] == max_idx).sum())
    at_edges = at_bottom + at_top
    near_bottom = int((df['focus_index'] <= 1).sum())
    near_top = int((df['focus_index'] >= max_idx - 1).sum())
    near_edges = int(((df['focus_index'] <= 1) | (df['focus_index'] >= max_idx - 1)).sum())

    # Clipping analysis
    clipped_low = int((df['focus_index_float'] < 0.5).sum())
    clipped_high = int((df['focus_index_float'] > max_idx - 0.5).sum())

    stats_row = {
        'experiment': exp_name,
        'csv_path': str(metadata_path),
        'z_stack_size': z_size,
        'n_subtiles': n_subtiles,

        # Focus index stats
        'focus_index_mean': float(df['focus_index'].mean()),
        'focus_index_median': float(df['focus_index'].median()),
        'focus_index_std': float(df['focus_index'].std()),
        'focus_index_min': int(df['focus_index'].min()),
        'focus_index_max': int(df['focus_index'].max()),

        # Focus index float (subpixel) stats
        'focus_float_mean': float(df['focus_index_float'].mean()),
        'focus_float_median': float(df['focus_index_float'].median()),
        'focus_float_std': float(df['focus_index_float'].std()),

        # Z offset stats (distance from middle)
        'z_offset_mean': float(df['z_focus_offset'].mean()),
        'z_offset_median': float(df['z_focus_offset'].median()),
        'z_offset_std': float(df['z_focus_offset'].std()),
        'z_offset_min': float(df['z_focus_offset'].min()),
        'z_offset_max': float(df['z_focus_offset'].max()),

        # Edge counts
        'at_bottom_edge': at_bottom,
        'at_top_edge': at_top,
        'at_any_edge': at_edges,
        'near_bottom': near_bottom,
        'near_top': near_top,
        'near_any_edge': near_edges,

        # Percentages
        'pct_at_bottom': 100 * at_bottom / n_subtiles,
        'pct_at_top': 100 * at_top / n_subtiles,
        'pct_at_edges': 100 * at_edges / n_subtiles,
        'pct_near_edges': 100 * near_edges / n_subtiles,

        # Clipping
        'clipped_low': clipped_low,
        'clipped_high': clipped_high,
        'pct_clipped_low': 100 * clipped_low / n_subtiles,
        'pct_clipped_high': 100 * clipped_high / n_subtiles,

        # Distribution shape
        'focus_skewness': float(df['focus_index_float'].skew()),
        'focus_kurtosis': float(df['focus_index_float'].kurtosis()),

        # Normalized offset (relative to z_size for cross-experiment comparison)
        'normalized_offset_mean': float(df['z_focus_offset'].mean() / (z_size / 2)) if z_size > 0 else 0,
        'normalized_offset_std': float(df['z_focus_offset'].std() / (z_size / 2)) if z_size > 0 else 0,
    }
    timings['calculate_stats'] = time.time() - t0

    timings['total'] = time.time() - t_start

    if verbose:
        print(f"\n[TIMING] {exp_name} ({n_subtiles:,} subtiles):")
        for key, val in sorted(timings.items(), key=lambda x: -x[1]):
            pct = (val / timings['total'] * 100) if timings['total'] > 0 else 0
            print(f"  {key:25s}: {val:6.3f}s ({pct:5.1f}%)")

    return stats_row


def collect_all_experiment_stats(
    metadata_type: str = "pheno",
    skip_low_count_exps: bool = False,
    verbose_timing: bool = False,
    include_bad_exps: bool = False,
) -> pd.DataFrame:
    """
    Collect subtile focus statistics from all experiments using OpsDataset.

    Args:
        metadata_type: "pheno" or "track"
        skip_low_count_exps: Skip known low-count experiments
        verbose_timing: Print detailed timing info

    Returns:
        DataFrame with one row per experiment
    """
    # Low-count experiments to exclude
    LOW_COUNT_EXPERIMENTS = [
        "ops0028_20250417",
        "ops0011_20250205",
        "ops0012_20250206",
        "ops0023_20250317",
    ]

    dummy_dataset = OpsDataset("dummy")
    config_root_dir = dummy_dataset.config_paths["exp_config_dir"]

    if not config_root_dir.is_dir():
        print(f"Configuration directory not found at {config_root_dir}. Aborting.")
        return pd.DataFrame()

    config_files = sorted(list(config_root_dir.glob("*_config.yaml")))

    all_stats = []
    skipped_experiments = []
    low_count_skipped = []
    bad_exp_skipped = []
    no_data_experiments = []

    for config_path in tqdm(config_files, desc="Analyzing experiments", disable=verbose_timing):
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                if not config or "experiment_name" not in config:
                    continue

            exp_name = config["experiment_name"]

            # Skip low-count experiments if flag is set
            if skip_low_count_exps and exp_name in LOW_COUNT_EXPERIMENTS:
                low_count_skipped.append(exp_name)
                continue

            # Filter experiments: only include standard format opsXXXX_YYYYMMDD
            standard_format = re.match(r"^ops\d{4}_\d{8}$", exp_name)
            if not standard_format:
                skipped_experiments.append(exp_name)
                continue

            # Skip bad/excluded experiments unless explicitly included
            if not include_bad_exps and is_excluded(exp_name):
                bad_exp_skipped.append(exp_name)
                continue

            dataset = OpsDataset(exp_name)

            # Analyze subtile focus for this experiment
            stats = analyze_experiment_subtile_focus(
                dataset, exp_name, metadata_type=metadata_type, verbose=verbose_timing
            )

            if stats is None:
                no_data_experiments.append(exp_name)
                continue

            all_stats.append(stats)

        except Exception as e:
            print(f"\nError processing {config_path.stem}: {e}")
            continue

    if low_count_skipped:
        print(f"\nExcluded {len(low_count_skipped)} low-count experiments")

    if bad_exp_skipped:
        print(f"\nExcluded {len(bad_exp_skipped)} bad/excluded experiments")

    if skipped_experiments:
        print(f"\nSkipped {len(skipped_experiments)} non-standard experiments")

    if no_data_experiments:
        print(f"\nNo subtile metadata found for {len(no_data_experiments)} experiments")

    print(f"\nAnalyzed {len(all_stats)} experiments with subtile focus data\n")

    return pd.DataFrame(all_stats)


def calculate_zscore_outliers(series: pd.Series, threshold: float = 2.5) -> pd.Series:
    """Identify outliers using z-score method."""
    mean = series.mean()
    std = series.std()
    if std == 0:
        return pd.Series([False] * len(series), index=series.index)
    z_scores = (series - mean) / std
    return np.abs(z_scores) > threshold


def calculate_modified_zscore_outliers(series: pd.Series, threshold: float = 3.5) -> pd.Series:
    """Identify outliers using modified z-score (MAD-based)."""
    median = series.median()
    mad = np.median(np.abs(series - median))
    if mad == 0:
        mad = np.mean(np.abs(series - median))
    if mad == 0:
        return pd.Series([False] * len(series), index=series.index)
    modified_z = 0.6745 * (series - median) / mad
    return np.abs(modified_z) > threshold


def calculate_iqr_outliers(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
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
    """Analyze all metrics for distribution shifts."""
    if metrics is None:
        metrics = [
            'z_offset_mean', 'z_offset_median', 'z_offset_std',
            'pct_at_edges', 'pct_near_edges',
            'pct_clipped_high', 'pct_clipped_low',
            'focus_float_mean', 'focus_float_std',
            'normalized_offset_mean',
        ]

    metrics = [m for m in metrics if m in df.columns]
    results = df[['experiment', 'z_stack_size']].copy()

    for metric in metrics:
        series = df[metric]

        z_outlier = calculate_zscore_outliers(series, z_threshold)
        mod_z_outlier = calculate_modified_zscore_outliers(series, mod_z_threshold)
        iqr_outlier = calculate_iqr_outliers(series, iqr_multiplier)

        mean = series.mean()
        std = series.std()
        z_scores = (series - mean) / std if std > 0 else pd.Series([0] * len(series))

        outlier_count = z_outlier.astype(int) + mod_z_outlier.astype(int) + iqr_outlier.astype(int)
        combined_outlier = outlier_count >= 2

        results[f'{metric}_value'] = series
        results[f'{metric}_zscore'] = z_scores
        results[f'{metric}_outlier'] = combined_outlier
        results[f'{metric}_direction'] = np.where(
            combined_outlier,
            np.where(z_scores > 0, 'HIGH', 'LOW'),
            ''
        )

    outlier_cols = [c for c in results.columns if c.endswith('_outlier')]
    results['total_outlier_flags'] = results[outlier_cols].sum(axis=1)
    results['is_anomalous'] = results['total_outlier_flags'] >= 2

    return results


def create_ridge_plot(
    df: pd.DataFrame,
    analysis: pd.DataFrame,
    output_dir: Path,
    metrics: list[tuple[str, str]],
    overlap: float = 0.6,
) -> None:
    """Create ridge plot showing focus distributions across experiments."""
    n_metrics = len([m for m, _ in metrics if m in df.columns])
    fig, axes = plt.subplots(1, n_metrics, figsize=(7 * n_metrics, 14))

    if n_metrics == 1:
        axes = [axes]

    # Sort by experiment number (e.g. ops0033_20250429 -> 33)
    def _extract_ops_num(name):
        m = re.search(r'ops0*(\d+)', name)
        return int(m.group(1)) if m else 0
    sorted_df = df.copy()
    sorted_df['_ops_num'] = sorted_df['experiment'].apply(_extract_ops_num)
    sorted_df = sorted_df.sort_values('_ops_num').reset_index(drop=True)
    sorted_df = sorted_df.drop(columns='_ops_num')
    sorted_analysis = analysis.set_index('experiment').loc[sorted_df['experiment']].reset_index()

    n_experiments = len(sorted_df)

    for ax, (metric, label) in zip(axes, metrics):
        if metric not in df.columns:
            continue

        values = sorted_df[metric].values
        experiments = sorted_df['experiment'].values
        z_sizes = sorted_df['z_stack_size'].values
        outlier_col = f'{metric}_outlier'
        is_outlier = sorted_analysis[outlier_col].values if outlier_col in sorted_analysis.columns else np.zeros(n_experiments, dtype=bool)

        global_mean = values.mean()
        global_std = values.std()

        x_min = values.min() - 0.15 * (values.max() - values.min() + 0.1)
        x_max = values.max() + 0.15 * (values.max() - values.min() + 0.1)
        x_range = np.linspace(x_min, x_max, 200)

        row_height = 1.0

        for i, (val, exp, z_size, outlier) in enumerate(zip(values, experiments, z_sizes, is_outlier)):
            bandwidth = max(global_std * 0.15, 0.01)
            y_offset = i * row_height * (1 - overlap)

            density = np.exp(-0.5 * ((x_range - val) / bandwidth) ** 2)
            density = density / density.max() * row_height * 0.8

            if outlier:
                color = 'red'
                alpha = 0.8
                linewidth = 1.5
            else:
                color = 'steelblue'
                alpha = 0.4
                linewidth = 0.8

            ax.fill_between(x_range, y_offset, y_offset + density,
                           color=color, alpha=alpha, linewidth=0)
            ax.plot(x_range, y_offset + density, color=color, alpha=min(1, alpha + 0.3),
                   linewidth=linewidth)

            # Label with experiment name and z_size
            label_color = 'red' if outlier else 'black'
            label_weight = 'bold' if outlier else 'normal'
            exp_label = f"{exp[:12]} ({z_size}z)"
            ax.text(x_min - 0.02 * (x_max - x_min), y_offset,
                   exp_label, fontsize=6, ha='right', va='center',
                   color=label_color, fontweight=label_weight)

        ax.axvline(global_mean, color='green', linestyle='-', lw=2, alpha=0.7, label=f'Mean: {global_mean:.2f}')
        ax.axvline(0, color='black', linestyle='--', lw=1.5, alpha=0.5, label='Zero (middle)')
        if global_std > 0:
            ax.axvline(global_mean - 2.5 * global_std, color='orange', linestyle='--', lw=1.5, alpha=0.7, label='-2.5σ')
            ax.axvline(global_mean + 2.5 * global_std, color='orange', linestyle='--', lw=1.5, alpha=0.7, label='+2.5σ')

        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel('Experiments (sorted by offset)', fontsize=10)
        ax.set_title(f'{label}\n(red = outlier)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_yticks([])
        ax.set_xlim(x_min - 0.2 * (x_max - x_min), x_max)

    plt.suptitle('Subtile Focus Distribution Shifts Across Experiments\n(Ridge Plot - Each row is one experiment)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'ridge_plot_focus_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_edge_clipping_plot(
    df: pd.DataFrame,
    analysis: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create plot showing edge clipping percentages across experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Z-Stack Edge Clipping Analysis', fontsize=14, fontweight='bold')

    # Sort by pct_at_edges
    sorted_df = df.sort_values('pct_at_edges', ascending=False).reset_index(drop=True)
    sorted_analysis = analysis.set_index('experiment').loc[sorted_df['experiment']].reset_index()

    is_outlier = sorted_analysis['pct_at_edges_outlier'].values if 'pct_at_edges_outlier' in sorted_analysis.columns else np.zeros(len(sorted_df), dtype=bool)

    # 1. Bar chart of edge clipping percentage
    ax1 = axes[0, 0]
    colors = ['red' if o else 'steelblue' for o in is_outlier]
    bars = ax1.barh(range(len(sorted_df)), sorted_df['pct_at_edges'], color=colors, alpha=0.7)

    # Label top experiments
    n_labels = min(15, len(sorted_df))
    ax1.set_yticks(range(n_labels))
    ax1.set_yticklabels([f"{e[:12]} ({z}z)" for e, z in zip(sorted_df['experiment'][:n_labels], sorted_df['z_stack_size'][:n_labels])], fontsize=7)
    for i, tick in enumerate(ax1.get_yticklabels()):
        if is_outlier[i]:
            tick.set_color('red')
            tick.set_fontweight('bold')

    ax1.set_xlabel('% Subtiles at Edge (idx=0 or max)')
    ax1.set_title('Edge Clipping (focus at idx=0 or idx=max)')
    ax1.axvline(sorted_df['pct_at_edges'].mean(), color='green', linestyle='--', lw=1.5, label=f"Mean: {sorted_df['pct_at_edges'].mean():.1f}%")
    ax1.legend()

    # 2. Scatter: z_stack_size vs pct_at_edges
    ax2 = axes[0, 1]
    scatter_colors = ['red' if o else 'steelblue' for o in is_outlier]
    ax2.scatter(sorted_df['z_stack_size'], sorted_df['pct_at_edges'], c=scatter_colors, alpha=0.7, s=60)

    for i, (x, y, exp, outlier) in enumerate(zip(sorted_df['z_stack_size'], sorted_df['pct_at_edges'], sorted_df['experiment'], is_outlier)):
        if outlier or y > sorted_df['pct_at_edges'].quantile(0.9):
            ax2.annotate(exp[:10], (x, y), fontsize=6, ha='left', va='bottom',
                        xytext=(3, 3), textcoords='offset points', color='red' if outlier else 'black')

    ax2.set_xlabel('Z Stack Size')
    ax2.set_ylabel('% Subtiles at Edge')
    ax2.set_title('Edge Clipping vs Z-Stack Depth')

    # 3. Histogram of z_offset_mean
    ax3 = axes[1, 0]
    ax3.hist(df['z_offset_mean'], bins=25, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    ax3.axvline(0, color='black', linestyle='--', lw=2, label='Zero (middle)')
    ax3.axvline(df['z_offset_mean'].mean(), color='green', linestyle='-', lw=2, label=f"Mean: {df['z_offset_mean'].mean():.2f}")
    ax3.set_xlabel('Mean Z Focus Offset')
    ax3.set_ylabel('Density')
    ax3.set_title('Distribution of Mean Z Offset Across Experiments')
    ax3.legend()

    # Mark outliers
    outlier_vals = df.loc[analysis['z_offset_mean_outlier'] == True, 'z_offset_mean'] if 'z_offset_mean_outlier' in analysis.columns else []
    for val in outlier_vals:
        ax3.axvline(val, color='red', linestyle='-', lw=1.5, alpha=0.5)

    # 4. Box plot comparison by z_stack_size
    ax4 = axes[1, 1]
    z_sizes = sorted(df['z_stack_size'].unique())
    data_by_z = [df[df['z_stack_size'] == z]['pct_at_edges'].values for z in z_sizes]

    bp = ax4.boxplot(data_by_z, labels=[str(z) for z in z_sizes], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)

    ax4.set_xlabel('Z Stack Size')
    ax4.set_ylabel('% Subtiles at Edge')
    ax4.set_title('Edge Clipping by Z-Stack Depth')

    plt.tight_layout()
    plt.savefig(output_dir / 'edge_clipping_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_violin_strip_plot(
    df: pd.DataFrame,
    analysis: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create violin + strip plot for key focus metrics."""
    metrics = [
        ('z_offset_mean', 'Mean Z Offset'),
        ('z_offset_std', 'Z Offset Std Dev'),
        ('pct_at_edges', '% at Edges (idx=0 or max)'),
        ('pct_near_edges', '% Near Edges (idx≤1 or ≥max-1)'),
    ]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (metric, label) in zip(axes, metrics):
        if metric not in df.columns:
            ax.set_visible(False)
            continue

        values = df[metric].values
        outlier_col = f'{metric}_outlier'
        is_outlier = analysis[outlier_col].values if outlier_col in analysis.columns else np.zeros(len(df), dtype=bool)

        parts = ax.violinplot([values], positions=[0], showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('steelblue')
            pc.set_alpha(0.3)

        normal_vals = values[~is_outlier]
        outlier_vals = values[is_outlier]
        outlier_exps = df['experiment'].values[is_outlier]

        jitter_normal = np.random.uniform(-0.15, 0.15, len(normal_vals))
        jitter_outlier = np.random.uniform(-0.15, 0.15, len(outlier_vals))

        ax.scatter(jitter_normal, normal_vals, c='steelblue', alpha=0.5, s=30, label='Normal')
        ax.scatter(jitter_outlier, outlier_vals, c='red', alpha=0.9, s=80, marker='*',
                  edgecolors='darkred', linewidth=0.5, label='Outlier', zorder=5)

        for j, (x, y, exp) in enumerate(zip(jitter_outlier, outlier_vals, outlier_exps)):
            ax.annotate(exp[:10], (x, y), fontsize=6, ha='left', va='bottom',
                       xytext=(5, 2), textcoords='offset points', color='red')

        mean = values.mean()
        std = values.std()
        ax.axhline(mean, color='green', linestyle='-', lw=1.5, alpha=0.7)
        if std > 0:
            ax.axhline(mean + 2.5 * std, color='orange', linestyle='--', lw=1, alpha=0.7)
            ax.axhline(mean - 2.5 * std, color='orange', linestyle='--', lw=1, alpha=0.7)

        # Add zero reference line for offset metrics
        if 'offset' in metric.lower():
            ax.axhline(0, color='black', linestyle='--', lw=1, alpha=0.5)

        ax.set_ylabel(label)
        ax.set_title(f'{label}', fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_xlim(-0.5, 0.5)

    plt.suptitle('Subtile Focus Metric Distributions with Outliers Highlighted\n(Violin + Strip Plot)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'violin_strip_focus_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_individual_metric_plots(
    df: pd.DataFrame,
    analysis: pd.DataFrame,
    output_dir: Path,
    figsize: tuple = (14, 10),
) -> None:
    """Create individual 4-panel plots for each metric."""
    metrics = [
        ('z_offset_mean', 'Mean Z Focus Offset'),
        ('z_offset_std', 'Z Offset Std Dev'),
        ('pct_at_edges', '% at Edges (idx=0 or max)'),
        ('pct_near_edges', '% Near Edges (idx≤1 or ≥max-1)'),
        ('focus_float_mean', 'Mean Focus Index (float)'),
        ('normalized_offset_mean', 'Normalized Offset Mean'),
    ]

    for metric, label in metrics:
        if metric not in df.columns:
            continue

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Distribution Analysis: {label}', fontsize=14, fontweight='bold')

        values = df[metric]
        experiments = df['experiment']
        z_sizes = df['z_stack_size']
        outlier_col = f'{metric}_outlier'
        is_outlier = analysis[outlier_col] if outlier_col in analysis.columns else pd.Series([False] * len(df))

        # 1. Histogram with KDE
        ax1 = axes[0, 0]
        ax1.hist(values, bins=20, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        if len(values) > 1 and values.std() > 0:
            try:
                kde_x = np.linspace(values.min(), values.max(), 100)
                kde = stats.gaussian_kde(values)
                ax1.plot(kde_x, kde(kde_x), 'r-', lw=2, label='KDE')
            except Exception:
                pass
        ax1.axvline(values.mean(), color='green', linestyle='--', lw=2, label=f'Mean: {values.mean():.2f}')
        ax1.axvline(values.median(), color='orange', linestyle='--', lw=2, label=f'Median: {values.median():.2f}')
        if 'offset' in metric.lower():
            ax1.axvline(0, color='black', linestyle='--', lw=1.5, alpha=0.5, label='Zero')
        ax1.set_xlabel(label)
        ax1.set_ylabel('Density')
        ax1.set_title('Distribution')
        ax1.legend(fontsize=8)

        outlier_vals = values[is_outlier]
        outlier_exps = experiments[is_outlier]
        for val, exp in zip(outlier_vals, outlier_exps):
            ax1.axvline(val, color='red', linestyle='-', lw=1.5, alpha=0.7)

        # 2. Box plot with labeled outliers
        ax2 = axes[0, 1]
        bp = ax2.boxplot(values, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('steelblue')
        bp['boxes'][0].set_alpha(0.7)
        if len(outlier_vals) > 0:
            jitter = np.random.uniform(0.9, 1.1, len(outlier_vals))
            ax2.scatter(jitter, outlier_vals, color='red', s=100, zorder=5, marker='*', label='Flagged')
            for j, (x, y, exp) in enumerate(zip(jitter, outlier_vals, outlier_exps)):
                ax2.annotate(exp[:12], (x, y), fontsize=7, ha='left', va='bottom',
                           xytext=(5, 2), textcoords='offset points', color='red', fontweight='bold')
            ax2.legend()
        ax2.set_ylabel(label)
        ax2.set_title('Box Plot with Outliers Labeled')
        ax2.set_xticklabels(['All Experiments'])

        # 3. Bar chart sorted by value
        ax3 = axes[1, 0]
        sorted_idx = values.argsort()
        sorted_vals = values.iloc[sorted_idx]
        sorted_exps = experiments.iloc[sorted_idx]
        sorted_zsizes = z_sizes.iloc[sorted_idx]
        sorted_outliers = is_outlier.iloc[sorted_idx]

        colors = ['red' if o else 'steelblue' for o in sorted_outliers]
        ax3.barh(range(len(sorted_vals)), sorted_vals, color=colors, alpha=0.7)

        outlier_indices = [i for i, o in enumerate(sorted_outliers) if o]
        n_labels = min(12, len(sorted_exps))
        step = max(1, len(sorted_exps) // n_labels)
        normal_indices = list(range(0, len(sorted_exps), step))
        all_labeled = sorted(set(outlier_indices + normal_indices))

        ax3.set_yticks(all_labeled)
        ytick_labels = [f"{sorted_exps.iloc[i][:10]} ({sorted_zsizes.iloc[i]}z)" for i in all_labeled]
        ax3.set_yticklabels(ytick_labels, fontsize=7)

        for tick_label, idx in zip(ax3.get_yticklabels(), all_labeled):
            if sorted_outliers.iloc[idx]:
                tick_label.set_color('red')
                tick_label.set_fontweight('bold')

        ax3.set_xlabel(label)
        ax3.set_title('Experiments Ranked by Value (red = flagged)')
        ax3.axvline(values.mean(), color='green', linestyle='--', lw=1.5, alpha=0.7)
        if 'offset' in metric.lower():
            ax3.axvline(0, color='black', linestyle='--', lw=1, alpha=0.5)

        # 4. Z-score plot
        ax4 = axes[1, 1]
        z_scores = analysis[f'{metric}_zscore'] if f'{metric}_zscore' in analysis.columns else (values - values.mean()) / values.std()
        sorted_z = z_scores.iloc[sorted_idx]

        colors = ['red' if o else 'steelblue' for o in sorted_outliers]
        ax4.barh(range(len(sorted_z)), sorted_z, color=colors, alpha=0.7)
        ax4.axvline(0, color='black', linestyle='-', lw=1)
        ax4.axvline(-2.5, color='orange', linestyle='--', lw=1.5, alpha=0.7, label='z = -2.5')
        ax4.axvline(2.5, color='orange', linestyle='--', lw=1.5, alpha=0.7, label='z = +2.5')

        ax4.set_yticks(all_labeled)
        ax4.set_yticklabels(ytick_labels, fontsize=7)
        for tick_label, idx in zip(ax4.get_yticklabels(), all_labeled):
            if sorted_outliers.iloc[idx]:
                tick_label.set_color('red')
                tick_label.set_fontweight('bold')

        ax4.set_xlabel('Z-Score')
        ax4.set_title('Z-Scores (red = flagged)')
        ax4.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(output_dir / f'distribution_{metric}.png', dpi=150, bbox_inches='tight')
        plt.close()


def generate_report(
    df: pd.DataFrame,
    analysis: pd.DataFrame,
    output_path: str | Path,
) -> str:
    """Generate a text report of detected distribution shifts."""
    lines = []
    lines.append("=" * 80)
    lines.append("SUBTILE FOCUS DISTRIBUTION SHIFT ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")

    n_experiments = len(df)
    anomalous = analysis[analysis['is_anomalous']]
    n_anomalous = len(anomalous)

    lines.append(f"Total experiments analyzed: {n_experiments}")
    lines.append(f"Experiments with distribution shifts: {n_anomalous} ({100*n_anomalous/n_experiments:.1f}%)")
    lines.append("")

    # Global statistics
    lines.append("-" * 80)
    lines.append("GLOBAL STATISTICS (Reference)")
    lines.append("-" * 80)

    key_metrics = ['z_offset_mean', 'z_offset_std', 'pct_at_edges', 'pct_near_edges', 'focus_float_mean']
    for metric in key_metrics:
        if metric in df.columns:
            vals = df[metric]
            lines.append(f"  {metric:25s}: mean={vals.mean():10.2f}, median={vals.median():10.2f}, std={vals.std():10.2f}")
    lines.append("")

    # Z-stack size breakdown
    lines.append("-" * 80)
    lines.append("Z-STACK SIZE BREAKDOWN")
    lines.append("-" * 80)
    for z_size in sorted(df['z_stack_size'].unique()):
        subset = df[df['z_stack_size'] == z_size]
        lines.append(f"  {z_size} slices: {len(subset)} experiments, mean offset={subset['z_offset_mean'].mean():.2f}, mean edge%={subset['pct_at_edges'].mean():.1f}%")
    lines.append("")

    # Flagged experiments
    if n_anomalous > 0:
        lines.append("-" * 80)
        lines.append("FLAGGED EXPERIMENTS (Distribution Shifts Detected)")
        lines.append("-" * 80)
        lines.append("")

        for _, row in anomalous.iterrows():
            exp = row['experiment']
            z_size = row['z_stack_size']
            flags = row['total_outlier_flags']
            lines.append(f"EXPERIMENT: {exp} ({z_size} slices)")
            lines.append(f"  Total flags: {flags}")

            flagged_metrics = []
            for col in analysis.columns:
                if col.endswith('_outlier') and row[col]:
                    metric_name = col.replace('_outlier', '')
                    direction = row.get(f'{metric_name}_direction', '')
                    value = row.get(f'{metric_name}_value', '')
                    zscore = row.get(f'{metric_name}_zscore', '')
                    if isinstance(value, (int, float)) and isinstance(zscore, (int, float)):
                        flagged_metrics.append(f"    - {metric_name}: {value:.2f} (z={zscore:.2f}, {direction})")

            lines.append("  Flagged metrics:")
            lines.extend(flagged_metrics)
            lines.append("")
    else:
        lines.append("No significant distribution shifts detected.")
        lines.append("")

    # Special cases
    lines.append("-" * 80)
    lines.append("SPECIAL CASES - POTENTIAL Z-STACK ISSUES")
    lines.append("-" * 80)

    # High edge clipping
    high_edge = df[df['pct_at_edges'] > 5]
    if len(high_edge) > 0:
        lines.append("")
        lines.append("Experiments with HIGH edge clipping (>5% at edges):")
        for _, row in high_edge.sort_values('pct_at_edges', ascending=False).iterrows():
            lines.append(f"  - {row['experiment']}: {row['pct_at_edges']:.1f}% at edges ({row['z_stack_size']} slices)")

    # Strongly negative offset (focus below range)
    neg_offset = df[df['z_offset_mean'] < -0.5]
    if len(neg_offset) > 0:
        lines.append("")
        lines.append("Experiments with strongly NEGATIVE offset (focus below Z range):")
        for _, row in neg_offset.sort_values('z_offset_mean').iterrows():
            lines.append(f"  - {row['experiment']}: mean offset={row['z_offset_mean']:.2f} ({row['z_stack_size']} slices)")

    # Strongly positive offset (focus above range)
    pos_offset = df[df['z_offset_mean'] > 1.0]
    if len(pos_offset) > 0:
        lines.append("")
        lines.append("Experiments with strongly POSITIVE offset (focus above Z range):")
        for _, row in pos_offset.sort_values('z_offset_mean', ascending=False).iterrows():
            lines.append(f"  - {row['experiment']}: mean offset={row['z_offset_mean']:.2f} ({row['z_stack_size']} slices)")

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    report_content = "\n".join(lines)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report_content)

    return report_content


def run_full_analysis(
    output_dir: str | Path = "tests/subtile_focus_distribution_analysis",
    metadata_type: str = "pheno",
    skip_low_count_exps: bool = False,
    verbose_timing: bool = False,
    z_threshold: float = 2.5,
    mod_z_threshold: float = 3.5,
    iqr_multiplier: float = 1.5,
    include_bad_exps: bool = False,
) -> dict:
    """Run complete subtile focus distribution analysis.

    Args:
        output_dir: Directory for output files
        metadata_type: "pheno" or "track"
        skip_low_count_exps: Skip known low-count experiments
        verbose_timing: Print detailed timing info
        z_threshold: Z-score threshold for outlier detection
        mod_z_threshold: Modified z-score threshold
        iqr_multiplier: IQR multiplier

    Returns:
        Dict with analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print("SUBTILE FOCUS DISTRIBUTION ANALYSIS")
    print("=" * 100 + "\n")

    # Collect stats from all experiments using OpsDataset
    df = collect_all_experiment_stats(
        metadata_type=metadata_type,
        skip_low_count_exps=skip_low_count_exps,
        verbose_timing=verbose_timing,
        include_bad_exps=include_bad_exps,
    )

    if len(df) == 0:
        print("No experiments with subtile focus data found!")
        return {}

    # Save raw stats
    df.to_csv(output_dir / 'subtile_focus_stats.csv', index=False)

    print("Running distribution shift analysis...")
    analysis = analyze_distribution_shifts(
        df,
        z_threshold=z_threshold,
        mod_z_threshold=mod_z_threshold,
        iqr_multiplier=iqr_multiplier,
    )

    print("Generating ridge plot...")
    ridge_metrics = [
        ('z_offset_mean', 'Mean Z Focus Offset'),
        ('pct_at_edges', '% at Edges (idx=0 or max)'),
        ('pct_near_edges', '% Near Edges (idx≤1 or ≥max-1)'),
    ]
    create_ridge_plot(df, analysis, output_dir, ridge_metrics)

    print("Generating edge clipping analysis plot...")
    create_edge_clipping_plot(df, analysis, output_dir)

    print("Generating violin/strip plots...")
    create_violin_strip_plot(df, analysis, output_dir)

    print("Generating individual metric plots...")
    create_individual_metric_plots(df, analysis, output_dir)

    print("Generating report...")
    report_path = output_dir / 'subtile_focus_distribution_report.txt'
    report = generate_report(df, analysis, report_path)
    print(report)

    print("Saving analysis CSV...")
    analysis_path = output_dir / 'subtile_focus_distribution_analysis.csv'
    analysis.to_csv(analysis_path, index=False)

    n_anomalous = analysis['is_anomalous'].sum()
    print(f"\n{'='*60}")
    print(f"Analysis complete!")
    print(f"  Experiments analyzed: {len(df)}")
    print(f"  Experiments with shifts: {n_anomalous}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")

    return {
        'df': df,
        'analysis': analysis,
        'output_dir': output_dir,
        'report_path': report_path,
        'analysis_csv_path': analysis_path,
        'n_anomalous': n_anomalous,
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Analyze subtile focus distribution shifts across experiments')
    parser.add_argument('--output-dir', '-o',
                       default='tests/subtile_focus_distribution_analysis',
                       help='Output directory for plots and reports')
    parser.add_argument('--metadata-type', '-m',
                       default='pheno',
                       choices=['pheno', 'track'],
                       help='Metadata type: pheno or track')
    parser.add_argument('--skip-low-count-exps',
                       action='store_true',
                       help='Skip known low-count experiments')
    parser.add_argument('--verbose-timing',
                       action='store_true',
                       help='Print detailed timing breakdown for each experiment')
    parser.add_argument('--z-threshold', type=float, default=2.5,
                       help='Z-score threshold for outlier detection')
    parser.add_argument('--iqr-multiplier', type=float, default=1.5,
                       help='IQR multiplier for outlier detection')
    parser.add_argument('--include-bad-exps',
                       action='store_true',
                       help='Include bad/excluded experiments (skipped by default)')

    args = parser.parse_args()

    run_full_analysis(
        output_dir=args.output_dir,
        metadata_type=args.metadata_type,
        skip_low_count_exps=args.skip_low_count_exps,
        verbose_timing=args.verbose_timing,
        z_threshold=args.z_threshold,
        iqr_multiplier=args.iqr_multiplier,
        include_bad_exps=args.include_bad_exps,
    )
    # To run: python tests/subtile_focus_distribution_analysis.py
    # To run for tracking: python tests/subtile_focus_distribution_analysis.py -m track
    # To run with verbose timing: python tests/subtile_focus_distribution_analysis.py --verbose-timing
