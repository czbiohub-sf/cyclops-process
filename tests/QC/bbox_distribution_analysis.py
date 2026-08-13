"""
Bbox Distribution Shift Detection
==================================

Analyzes bbox statistics across experiments to detect distribution shifts.
Uses multiple statistical methods to identify outliers:
1. Z-score method for mean/median shifts
2. Modified Z-score for robust outlier detection
3. IQR (Interquartile Range) method
4. Coefficient of Variation analysis

Generates:
- Distribution plots for each metric
- Summary statistics with outlier flags
- Detailed report of detected shifts

# usage
python tests/QC/bbox_distribution_analysis.py
python tests/QC/bbox_distribution_analysis.py --output-dir tests/bbox_distribution_analysis
python tests/QC/bbox_distribution_analysis.py --output-dir tests/bbox_distribution_analysis --z-threshold 3.0
python tests/QC/bbox_distribution_analysis.py --output-dir tests/bbox_distribution_analysis --mod-z-threshold 4.0
python tests/QC/bbox_distribution_analysis.py --output-dir tests/bbox_distribution_analysis --iqr-multiplier 2.0


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from typing import Optional


def load_bbox_stats(csv_path: str | Path) -> pd.DataFrame:
    """Load bbox debug stats CSV."""
    df = pd.read_csv(csv_path)
    # Clean experiment names (remove any whitespace)
    df['experiment'] = df['experiment'].str.strip()
    return df


def calculate_zscore_outliers(series: pd.Series, threshold: float = 2.5) -> pd.Series:
    """
    Identify outliers using z-score method.

    Args:
        series: Data series to analyze
        threshold: Z-score threshold (default 2.5 = ~1.2% in each tail)

    Returns:
        Boolean series indicating outliers
    """
    mean = series.mean()
    std = series.std()
    if std == 0:
        return pd.Series([False] * len(series), index=series.index)
    z_scores = (series - mean) / std
    return np.abs(z_scores) > threshold


def calculate_modified_zscore_outliers(series: pd.Series, threshold: float = 3.5) -> pd.Series:
    """
    Identify outliers using modified z-score (MAD-based).
    More robust to outliers than standard z-score.

    Args:
        series: Data series to analyze
        threshold: Modified z-score threshold (default 3.5)

    Returns:
        Boolean series indicating outliers
    """
    median = series.median()
    mad = np.median(np.abs(series - median))
    if mad == 0:
        # Use mean absolute deviation if MAD is 0
        mad = np.mean(np.abs(series - median))
    if mad == 0:
        return pd.Series([False] * len(series), index=series.index)

    # 0.6745 is the scaling factor for normal distribution
    modified_z = 0.6745 * (series - median) / mad
    return np.abs(modified_z) > threshold


def calculate_iqr_outliers(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    Identify outliers using IQR method.

    Args:
        series: Data series to analyze
        multiplier: IQR multiplier (1.5 = standard, 3.0 = extreme)

    Returns:
        Boolean series indicating outliers
    """
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
    """
    Analyze all metrics for distribution shifts.

    Args:
        df: DataFrame with bbox stats
        metrics: List of metrics to analyze (default: all numeric)
        z_threshold: Z-score threshold
        mod_z_threshold: Modified z-score threshold
        iqr_multiplier: IQR multiplier

    Returns:
        DataFrame with outlier flags and scores for each experiment
    """
    if metrics is None:
        # Default metrics to analyze
        metrics = [
            'height_mean', 'height_median', 'height_std',
            'width_mean', 'width_median', 'width_std',
            'area_mean', 'area_median', 'area_std',
            'fallback_count', 'pct_large_bbox',
        ]

    # Filter to metrics that exist in the dataframe
    metrics = [m for m in metrics if m in df.columns]

    results = df[['experiment']].copy()

    for metric in metrics:
        series = df[metric]

        # Calculate all outlier methods
        z_outlier = calculate_zscore_outliers(series, z_threshold)
        mod_z_outlier = calculate_modified_zscore_outliers(series, mod_z_threshold)
        iqr_outlier = calculate_iqr_outliers(series, iqr_multiplier)

        # Calculate z-scores for reporting
        mean = series.mean()
        std = series.std()
        z_scores = (series - mean) / std if std > 0 else pd.Series([0] * len(series))

        # Combined outlier flag (flagged by at least 2 methods)
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

    # Calculate overall outlier score (how many metrics flagged)
    outlier_cols = [c for c in results.columns if c.endswith('_outlier')]
    results['total_outlier_flags'] = results[outlier_cols].sum(axis=1)
    results['is_anomalous'] = results['total_outlier_flags'] >= 2  # At least 2 metrics flagged

    return results


def generate_distribution_plots(
    df: pd.DataFrame,
    output_dir: str | Path,
    metrics: Optional[list[str]] = None,
    figsize: tuple = (14, 10),
) -> None:
    """
    Generate distribution plots including:
    1. Ridge plots showing all experiment distributions overlaid
    2. Individual metric plots with outliers labeled
    3. Summary comparison plots

    Args:
        df: DataFrame with bbox stats
        output_dir: Directory to save plots
        metrics: List of metrics to plot
        figsize: Figure size
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if metrics is None:
        metrics = [
            ('height_mean', 'Height Mean (pixels)'),
            ('height_median', 'Height Median (pixels)'),
            ('height_std', 'Height Std Dev'),
            ('width_mean', 'Width Mean (pixels)'),
            ('width_median', 'Width Median (pixels)'),
            ('width_std', 'Width Std Dev'),
            ('area_mean', 'Area Mean (sq pixels)'),
            ('area_median', 'Area Median (sq pixels)'),
            ('area_std', 'Area Std Dev'),
        ]

    # Run analysis to get outlier flags
    analysis = analyze_distribution_shifts(df)

    # Create the main ridge plot for key metrics (subset for clarity)
    ridge_metrics = [
        ('height_mean', 'Height Mean (pixels)'),
        ('width_mean', 'Width Mean (pixels)'),
        ('area_mean', 'Area Mean (sq pixels)'),
        ('area_std', 'Area Std Dev'),
    ]
    create_ridge_plot(df, analysis, output_dir, ridge_metrics)

    # Create individual metric plots with outliers labeled
    create_individual_metric_plots(df, analysis, output_dir, metrics, figsize)

    # Create summary comparison plot
    create_summary_plot(df, analysis, output_dir, figsize)


def create_individual_metric_plots(
    df: pd.DataFrame,
    analysis: pd.DataFrame,
    output_dir: Path,
    metrics: list[tuple[str, str]],
    figsize: tuple = (14, 10),
) -> None:
    """
    Create individual 4-panel plots for each metric with outlier experiments labeled.
    """
    for metric, label in metrics:
        if metric not in df.columns:
            continue

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Distribution Analysis: {label}', fontsize=14, fontweight='bold')

        values = df[metric]
        experiments = df['experiment']
        outlier_col = f'{metric}_outlier'
        is_outlier = analysis[outlier_col] if outlier_col in analysis.columns else pd.Series([False] * len(df))

        # 1. Histogram with KDE
        ax1 = axes[0, 0]
        ax1.hist(values, bins=20, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        if len(values) > 1:
            try:
                kde_x = np.linspace(values.min(), values.max(), 100)
                kde = stats.gaussian_kde(values)
                ax1.plot(kde_x, kde(kde_x), 'r-', lw=2, label='KDE')
            except Exception:
                pass
        ax1.axvline(values.mean(), color='green', linestyle='--', lw=2, label=f'Mean: {values.mean():.1f}')
        ax1.axvline(values.median(), color='orange', linestyle='--', lw=2, label=f'Median: {values.median():.1f}')
        ax1.set_xlabel(label)
        ax1.set_ylabel('Density')
        ax1.set_title('Distribution')
        ax1.legend(fontsize=8)

        # Mark outliers on histogram
        outlier_vals = values[is_outlier]
        outlier_exps = experiments[is_outlier]
        for val, exp in zip(outlier_vals, outlier_exps):
            ax1.axvline(val, color='red', linestyle='-', lw=1.5, alpha=0.7)
            ax1.text(val, ax1.get_ylim()[1] * 0.95, exp[:10], fontsize=6, rotation=90,
                    ha='right', va='top', color='red', fontweight='bold')

        # 2. Box plot with labeled outliers
        ax2 = axes[0, 1]
        bp = ax2.boxplot(values, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('steelblue')
        bp['boxes'][0].set_alpha(0.7)
        # Overlay outlier points with labels
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

        # 3. Bar chart sorted by value with outliers labeled
        ax3 = axes[1, 0]
        sorted_idx = values.argsort()
        sorted_vals = values.iloc[sorted_idx]
        sorted_exps = experiments.iloc[sorted_idx]
        sorted_outliers = is_outlier.iloc[sorted_idx]

        colors = ['red' if o else 'steelblue' for o in sorted_outliers]
        bars = ax3.barh(range(len(sorted_vals)), sorted_vals, color=colors, alpha=0.7)

        # Show labels for all outliers and every Nth normal experiment
        outlier_indices = [i for i, o in enumerate(sorted_outliers) if o]
        n_labels = min(12, len(sorted_exps))
        step = max(1, len(sorted_exps) // n_labels)
        normal_indices = list(range(0, len(sorted_exps), step))

        # Combine and deduplicate
        all_labeled = sorted(set(outlier_indices + normal_indices))

        ax3.set_yticks(all_labeled)
        ytick_labels = []
        for i in all_labeled:
            exp_name = sorted_exps.iloc[i][:12]
            if sorted_outliers.iloc[i]:
                ytick_labels.append(exp_name)
            else:
                ytick_labels.append(exp_name)
        ax3.set_yticklabels(ytick_labels, fontsize=7)

        # Color outlier labels red
        for tick_label, idx in zip(ax3.get_yticklabels(), all_labeled):
            if sorted_outliers.iloc[idx]:
                tick_label.set_color('red')
                tick_label.set_fontweight('bold')

        ax3.set_xlabel(label)
        ax3.set_title('Experiments Ranked by Value (red = flagged)')
        ax3.axvline(values.mean(), color='green', linestyle='--', lw=1.5, alpha=0.7)

        # 4. Z-score plot with outliers labeled
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


def create_ridge_plot(
    df: pd.DataFrame,
    analysis: pd.DataFrame,
    output_dir: Path,
    metrics: list[tuple[str, str]],
    overlap: float = 0.6,
) -> None:
    """
    Create ridge plot (joy plot) showing overlapping distributions for all experiments.

    Each experiment gets a horizontal density curve, stacked vertically with partial overlap.
    Outlier experiments are highlighted in red.

    Args:
        df: DataFrame with bbox stats
        analysis: Analysis results with outlier flags
        output_dir: Output directory
        metrics: List of (column_name, display_label) tuples
        overlap: How much distributions overlap (0-1, higher = more overlap)
    """
    n_metrics = len([m for m, _ in metrics if m in df.columns])
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 12))

    if n_metrics == 1:
        axes = [axes]

    # Sort experiments by area_mean for consistent ordering
    sort_col = 'area_mean' if 'area_mean' in df.columns else df.columns[1]
    sorted_df = df.sort_values(sort_col).reset_index(drop=True)
    sorted_analysis = analysis.set_index('experiment').loc[sorted_df['experiment']].reset_index()

    n_experiments = len(sorted_df)

    for ax, (metric, label) in zip(axes, metrics):
        if metric not in df.columns:
            continue

        values = sorted_df[metric].values
        experiments = sorted_df['experiment'].values
        outlier_col = f'{metric}_outlier'
        is_outlier = sorted_analysis[outlier_col].values if outlier_col in sorted_analysis.columns else np.zeros(n_experiments, dtype=bool)

        # Global stats for reference lines
        global_mean = values.mean()
        global_std = values.std()

        # Calculate x range with padding
        x_min = values.min() - 0.1 * (values.max() - values.min())
        x_max = values.max() + 0.1 * (values.max() - values.min())
        x_range = np.linspace(x_min, x_max, 200)

        # Height per distribution row
        row_height = 1.0

        for i, (val, exp, outlier) in enumerate(zip(values, experiments, is_outlier)):
            # Create a narrow gaussian centered at the experiment's value
            # Width based on global std to show relative position
            bandwidth = global_std * 0.15  # Narrow bandwidth for clear peaks
            y_offset = i * row_height * (1 - overlap)

            # Gaussian density centered at the experiment's value
            density = np.exp(-0.5 * ((x_range - val) / bandwidth) ** 2)
            density = density / density.max() * row_height * 0.8  # Normalize height

            # Color based on outlier status
            if outlier:
                color = 'red'
                alpha = 0.8
                linewidth = 1.5
            else:
                color = 'steelblue'
                alpha = 0.4
                linewidth = 0.8

            # Fill the density curve
            ax.fill_between(x_range, y_offset, y_offset + density,
                           color=color, alpha=alpha, linewidth=0)
            ax.plot(x_range, y_offset + density, color=color, alpha=min(1, alpha + 0.3),
                   linewidth=linewidth)

            # Add experiment label on the left side of the plot, aligned with the baseline
            label_color = 'red' if outlier else 'black'
            label_weight = 'bold' if outlier else 'normal'
            ax.text(x_min - 0.02 * (x_max - x_min), y_offset,
                   exp[:12], fontsize=6, ha='right', va='center',
                   color=label_color, fontweight=label_weight)

        # Reference lines
        ax.axvline(global_mean, color='green', linestyle='-', lw=2, alpha=0.7, label=f'Mean: {global_mean:.0f}')
        ax.axvline(global_mean - 2.5 * global_std, color='orange', linestyle='--', lw=1.5, alpha=0.7, label='-2.5σ')
        ax.axvline(global_mean + 2.5 * global_std, color='orange', linestyle='--', lw=1.5, alpha=0.7, label='+2.5σ')

        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel('Experiments (sorted by area)', fontsize=10)
        ax.set_title(f'{label}\n(red = outlier)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)

        # Clean up y-axis (no need for tick labels)
        ax.set_yticks([])
        # Extend x-axis to the left to make room for labels
        ax.set_xlim(x_min - 0.15 * (x_max - x_min), x_max)

    plt.suptitle('Bbox Distribution Shifts Across Experiments\n(Ridge Plot - Each row is one experiment)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'ridge_plot_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Also create a combined violin + strip plot for quick comparison
    create_violin_strip_plot(df, analysis, output_dir, metrics)


def create_violin_strip_plot(
    df: pd.DataFrame,
    analysis: pd.DataFrame,
    output_dir: Path,
    metrics: list[tuple[str, str]],
) -> None:
    """
    Create violin + strip plot showing distribution of each metric with outliers highlighted.
    Compact view that shows both the overall distribution and individual experiment values.
    """
    n_metrics = len([m for m, _ in metrics if m in df.columns])
    fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (metric, label) in zip(axes, metrics):
        if metric not in df.columns:
            ax.set_visible(False)
            continue

        values = df[metric].values
        outlier_col = f'{metric}_outlier'
        is_outlier = analysis[outlier_col].values if outlier_col in analysis.columns else np.zeros(len(df), dtype=bool)

        # Create violin plot for overall distribution
        parts = ax.violinplot([values], positions=[0], showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('steelblue')
            pc.set_alpha(0.3)

        # Overlay individual points
        normal_vals = values[~is_outlier]
        outlier_vals = values[is_outlier]
        outlier_exps = df['experiment'].values[is_outlier]

        # Jittered x positions for strip plot effect
        jitter_normal = np.random.uniform(-0.15, 0.15, len(normal_vals))
        jitter_outlier = np.random.uniform(-0.15, 0.15, len(outlier_vals))

        ax.scatter(jitter_normal, normal_vals, c='steelblue', alpha=0.5, s=30, label='Normal')
        ax.scatter(jitter_outlier, outlier_vals, c='red', alpha=0.9, s=80, marker='*',
                  edgecolors='darkred', linewidth=0.5, label='Outlier', zorder=5)

        # Label outliers
        for j, (x, y, exp) in enumerate(zip(jitter_outlier, outlier_vals, outlier_exps)):
            ax.annotate(exp[:10], (x, y), fontsize=6, ha='left', va='bottom',
                       xytext=(5, 2), textcoords='offset points', color='red')

        # Stats
        mean = values.mean()
        std = values.std()
        ax.axhline(mean, color='green', linestyle='-', lw=1.5, alpha=0.7)
        ax.axhline(mean + 2.5 * std, color='orange', linestyle='--', lw=1, alpha=0.7)
        ax.axhline(mean - 2.5 * std, color='orange', linestyle='--', lw=1, alpha=0.7)

        ax.set_ylabel(label)
        ax.set_title(f'{label}', fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_xlim(-0.5, 0.5)

    # Hide extra axes
    for ax in axes[len(metrics):]:
        ax.set_visible(False)

    plt.suptitle('Bbox Metric Distributions with Outliers Highlighted\n(Violin + Strip Plot)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'violin_strip_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_plot(
    df: pd.DataFrame,
    analysis: pd.DataFrame,
    output_dir: Path,
    figsize: tuple = (16, 12),
) -> None:
    """Create a summary plot showing all flagged experiments."""

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Bbox Distribution Summary - Outlier Detection', fontsize=14, fontweight='bold')

    # Key metrics to summarize
    key_metrics = [
        ('height_mean', 'Height Mean'),
        ('width_mean', 'Width Mean'),
        ('area_mean', 'Area Mean'),
        ('height_std', 'Height Std'),
        ('width_std', 'Width Std'),
        ('area_std', 'Area Std'),
    ]

    for ax, (metric, label) in zip(axes.flat, key_metrics):
        if metric not in df.columns:
            ax.set_visible(False)
            continue

        values = df[metric]
        outlier_col = f'{metric}_outlier'
        is_outlier = analysis[outlier_col] if outlier_col in analysis.columns else pd.Series([False] * len(df))

        # Scatter plot: x = experiment index, y = value
        colors = ['red' if o else 'steelblue' for o in is_outlier]
        ax.scatter(range(len(values)), values, c=colors, alpha=0.7, s=50)

        # Add mean and std bands
        mean = values.mean()
        std = values.std()
        ax.axhline(mean, color='green', linestyle='-', lw=1.5, alpha=0.7, label='Mean')
        ax.axhline(mean + 2.5 * std, color='orange', linestyle='--', lw=1, alpha=0.7, label='+2.5 std')
        ax.axhline(mean - 2.5 * std, color='orange', linestyle='--', lw=1, alpha=0.7, label='-2.5 std')
        ax.fill_between(range(len(values)), mean - std, mean + std, alpha=0.1, color='green')

        # Label outliers
        outlier_idx = np.where(is_outlier)[0]
        for idx in outlier_idx:
            exp_name = df['experiment'].iloc[idx][:10]
            ax.annotate(exp_name, (idx, values.iloc[idx]), fontsize=6, rotation=45)

        ax.set_xlabel('Experiment Index')
        ax.set_ylabel(label)
        ax.set_title(f'{label} Distribution')
        if ax == axes[0, 0]:
            ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_dir / 'summary_outlier_detection.png', dpi=150, bbox_inches='tight')
    plt.close()


def generate_report(
    df: pd.DataFrame,
    analysis: pd.DataFrame,
    output_path: str | Path,
) -> str:
    """
    Generate a text report of detected distribution shifts.

    Args:
        df: Original DataFrame with bbox stats
        analysis: Analysis results with outlier flags
        output_path: Path to save the report

    Returns:
        Report content as string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("BBOX DISTRIBUTION SHIFT ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Summary statistics
    n_experiments = len(df)
    anomalous = analysis[analysis['is_anomalous']]
    n_anomalous = len(anomalous)

    lines.append(f"Total experiments analyzed: {n_experiments}")
    lines.append(f"Experiments with distribution shifts: {n_anomalous} ({100*n_anomalous/n_experiments:.1f}%)")
    lines.append("")

    # Global statistics for reference
    lines.append("-" * 80)
    lines.append("GLOBAL STATISTICS (Reference)")
    lines.append("-" * 80)

    key_metrics = ['height_mean', 'height_median', 'width_mean', 'width_median', 'area_mean', 'area_median']
    for metric in key_metrics:
        if metric in df.columns:
            vals = df[metric]
            lines.append(f"  {metric:20s}: mean={vals.mean():10.2f}, median={vals.median():10.2f}, std={vals.std():10.2f}")
    lines.append("")

    # Detailed outlier analysis
    if n_anomalous > 0:
        lines.append("-" * 80)
        lines.append("FLAGGED EXPERIMENTS (Distribution Shifts Detected)")
        lines.append("-" * 80)
        lines.append("")

        for _, row in anomalous.iterrows():
            exp = row['experiment']
            flags = row['total_outlier_flags']
            lines.append(f"EXPERIMENT: {exp}")
            lines.append(f"  Total flags: {flags}")

            # List which metrics are flagged
            flagged_metrics = []
            for col in analysis.columns:
                if col.endswith('_outlier') and row[col]:
                    metric_name = col.replace('_outlier', '')
                    direction = row.get(f'{metric_name}_direction', '')
                    value = row.get(f'{metric_name}_value', '')
                    zscore = row.get(f'{metric_name}_zscore', '')
                    flagged_metrics.append(f"    - {metric_name}: {value:.2f} (z={zscore:.2f}, {direction})")

            lines.append("  Flagged metrics:")
            lines.extend(flagged_metrics)
            lines.append("")
    else:
        lines.append("No significant distribution shifts detected.")
        lines.append("")

    # Special cases to check
    lines.append("-" * 80)
    lines.append("SPECIAL CASES")
    lines.append("-" * 80)

    # Check for experiments with very high fallback counts
    if 'fallback_count' in df.columns and 'valid_seg_count' in df.columns:
        df_temp = df.copy()
        df_temp['fallback_pct'] = df_temp['fallback_count'] / (df_temp['valid_seg_count'] + df_temp['fallback_count']) * 100
        high_fallback = df_temp[df_temp['fallback_pct'] > 15]  # >15% fallback
        if len(high_fallback) > 0:
            lines.append("")
            lines.append("Experiments with HIGH fallback rate (>15%):")
            for _, row in high_fallback.iterrows():
                lines.append(f"  - {row['experiment']}: {row['fallback_pct']:.1f}% fallback ({row['fallback_count']:,} cells)")

    # Check for experiments with unusual std (very low = uniform sizes, very high = variable)
    if 'height_std' in df.columns:
        std_thresh_low = df['height_std'].quantile(0.05)
        std_thresh_high = df['height_std'].quantile(0.95)

        low_std = df[df['height_std'] < std_thresh_low]
        high_std = df[df['height_std'] > std_thresh_high]

        if len(low_std) > 0:
            lines.append("")
            lines.append("Experiments with UNUSUALLY LOW height std (uniform bbox sizes):")
            for _, row in low_std.iterrows():
                lines.append(f"  - {row['experiment']}: height_std={row['height_std']:.2f}")

        if len(high_std) > 0:
            lines.append("")
            lines.append("Experiments with UNUSUALLY HIGH height std (variable bbox sizes):")
            for _, row in high_std.iterrows():
                lines.append(f"  - {row['experiment']}: height_std={row['height_std']:.2f}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    report_content = "\n".join(lines)

    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report_content)

    return report_content


def save_analysis_csv(
    analysis: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save the analysis results to CSV with outlier flags."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    analysis.to_csv(output_path, index=False)


def run_full_analysis(
    input_csv: str | Path,
    output_dir: str | Path,
    z_threshold: float = 2.5,
    mod_z_threshold: float = 3.5,
    iqr_multiplier: float = 1.5,
) -> dict:
    """
    Run complete distribution shift analysis.

    Args:
        input_csv: Path to bbox_debug_stats.csv
        output_dir: Directory for output files
        z_threshold: Z-score threshold for outlier detection
        mod_z_threshold: Modified z-score threshold
        iqr_multiplier: IQR multiplier

    Returns:
        dict with analysis results and paths to generated files
    """
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {input_csv}...")
    df = load_bbox_stats(input_csv)
    print(f"Loaded {len(df)} experiments")

    print("Running distribution shift analysis...")
    analysis = analyze_distribution_shifts(
        df,
        z_threshold=z_threshold,
        mod_z_threshold=mod_z_threshold,
        iqr_multiplier=iqr_multiplier,
    )

    print("Generating distribution plots...")
    generate_distribution_plots(df, output_dir)

    print("Generating report...")
    report_path = output_dir / 'distribution_shift_report.txt'
    report = generate_report(df, analysis, report_path)
    print(report)

    print("Saving analysis CSV...")
    analysis_path = output_dir / 'bbox_distribution_analysis.csv'
    save_analysis_csv(analysis, analysis_path)

    # Summary
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

    parser = argparse.ArgumentParser(description='Analyze bbox distribution shifts across experiments')
    parser.add_argument('input_csv', nargs='?',
                       default='tests/QC/bbox_debug_stats.csv',
                       help='Path to bbox_debug_stats.csv')
    parser.add_argument('--output-dir', '-o',
                       default='tests/QC/bbox_distribution_analysis',
                       help='Output directory for plots and reports')
    parser.add_argument('--z-threshold', type=float, default=2.5,
                       help='Z-score threshold for outlier detection')
    parser.add_argument('--iqr-multiplier', type=float, default=1.5,
                       help='IQR multiplier for outlier detection')

    args = parser.parse_args()

    run_full_analysis(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        z_threshold=args.z_threshold,
        iqr_multiplier=args.iqr_multiplier,
    )
