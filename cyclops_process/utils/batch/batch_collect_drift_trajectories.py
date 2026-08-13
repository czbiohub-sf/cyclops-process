#!/usr/bin/env python
"""
Batch collect ISS cycle drift trajectories across all OPS experiments.

This script gathers drift trajectory data from all experiments where ISS cycle
registration has been run. It collects the cumulative affine transforms from
YAML files and computes drift magnitude and direction statistics across all
experiments/wells.

Output:
- drift_trajectory_summary.csv: Mean and variance of drift magnitude and direction
  Saved in the same location as metrics_tracking.py outputs

Usage:
    python -m cyclops_process.utils.batch.batch_collect_drift_trajectories --reference-exp ops0107_20251208 
    python -m cyclops_process.utils.batch.batch_collect_drift_trajectories --reference-exp ops0045_20250603 --verbose
"""
import argparse
from pathlib import Path
from tqdm import tqdm
import sys
import os
import re
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
try:
    from matplotlib import colormaps
    get_cmap = colormaps.get_cmap
except ImportError:
    get_cmap = cm.get_cmap

from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from datetime import datetime
from typing import Dict, List, Tuple

sys.path.insert(0, os.getcwd())

from ops_utils.data.experiment import OpsDataset
from cyclops_process.paths import BASE_PATH


def load_affine_from_yaml(yaml_path: Path) -> np.ndarray:
    """
    Load 4x4 affine matrix from YAML file.

    Returns 3x3 affine in YX coordinates (discarding Z dimension).
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # YAML stores 4x4 matrix, extract 2D affine (YX)
    if 'affine' in data:
        affine_4x4 = np.array(data['affine'])
    elif 'affine_transform_zyx' in data:
        affine_4x4 = np.array(data['affine_transform_zyx'])
    else:
        raise KeyError(f"No affine key found in {yaml_path}. Keys: {list(data.keys())}")

    # Extract YX 2D affine from 4x4 (rows/cols 1,2 for Y,X)
    affine_3x3 = np.eye(3)
    affine_3x3[:2, :2] = affine_4x4[1:3, 1:3]  # Rotation/scale
    affine_3x3[:2, 2] = affine_4x4[1:3, 3]     # Translation

    return affine_3x3


def compute_drift_metrics(affine_3x3: np.ndarray, reference_affine: np.ndarray = None) -> Dict:
    """
    Compute drift metrics from affine transform.

    Args:
        affine_3x3: 3x3 affine matrix (YX coordinates)
        reference_affine: Reference affine to compute drift relative to (default: identity)

    Returns:
        Dictionary with drift metrics:
        - dy, dx: Translation in Y and X
        - magnitude: Euclidean distance of drift
        - angle_deg: Direction of drift in degrees (0=right, 90=up, etc.)
    """
    if reference_affine is None:
        reference_affine = np.eye(3)

    # Compute relative translation
    dy = affine_3x3[0, 2] - reference_affine[0, 2]
    dx = affine_3x3[1, 2] - reference_affine[1, 2]

    # Magnitude
    magnitude = np.sqrt(dy**2 + dx**2)

    # Angle (atan2 returns angle in radians, convert to degrees)
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)

    return {
        'dy': dy,
        'dx': dx,
        'magnitude': magnitude,
        'angle_deg': angle_deg,
    }


def collect_drift_data_for_well(
    experiment: str,
    well: str,
) -> List[Dict]:
    """
    Collect drift trajectory data for a single well using cumulative transforms.

    Returns list of dictionaries, one per round, with drift metrics.
    Also includes nucleus_to_round0 drift if available.
    """
    from ops_utils.data.filesystem import parse_well
    dataset = OpsDataset(experiment)

    # Construct path to transforms directory (row-aware token, e.g. "A1"/"B2")
    row, col = parse_well(well)
    register_root = dataset.preprocess_in_situ / "register"
    transforms_dir = register_root / f"transforms/{row}{col}"

    if not transforms_dir.exists():
        return []

    drift_data = []

    # Round 0 is the reference (identity)
    round0_affine = np.eye(3)

    # Check for segmentation_to_nucleus drift
    seg_yaml = transforms_dir / "segmentation_to_nucleus.yaml"
    has_seg_registration = seg_yaml.exists()
    seg_drift = None

    if has_seg_registration:
        try:
            affine_inv = load_affine_from_yaml(seg_yaml)
            affine = np.linalg.inv(affine_inv)
            seg_metrics = compute_drift_metrics(affine, round0_affine)
            seg_drift = {
                'dy': seg_metrics['dy'],
                'dx': seg_metrics['dx'],
                'magnitude': seg_metrics['magnitude'],
                'angle_deg': seg_metrics['angle_deg'],
            }
        except Exception:
            pass

    # Check for nucleus_to_round0 drift (DAPI pre-registration)
    nucleus_yaml = transforms_dir / "nucleus_to_round0.yml"
    has_nucleus_registration = nucleus_yaml.exists()
    nucleus_drift = None

    if has_nucleus_registration:
        try:
            affine_inv = load_affine_from_yaml(nucleus_yaml)
            affine = np.linalg.inv(affine_inv)
            nucleus_metrics = compute_drift_metrics(affine, round0_affine)
            nucleus_drift = {
                'dy': nucleus_metrics['dy'],
                'dx': nucleus_metrics['dx'],
                'magnitude': nucleus_metrics['magnitude'],
                'angle_deg': nucleus_metrics['angle_deg'],
            }
        except Exception:
            pass

    # Collect cumulative transforms for rounds 1-9
    for round_num in range(1, 10):
        # Try both .yaml and .yml extensions
        yaml_path = transforms_dir / f"round{round_num}_to_round0_cumulative.yaml"
        if not yaml_path.exists():
            yaml_path = transforms_dir / f"round{round_num}_to_round0_cumulative.yml"
            if not yaml_path.exists():
                continue

        try:
            # Load affine and invert it (YAML stores inverse transform)
            affine_inv = load_affine_from_yaml(yaml_path)
            affine = np.linalg.inv(affine_inv)

            # Compute drift relative to Round 0
            metrics = compute_drift_metrics(affine, round0_affine)

            entry = {
                'experiment': experiment,
                'well': well,
                'round': round_num,
                'dy': metrics['dy'],
                'dx': metrics['dx'],
                'magnitude': metrics['magnitude'],
                'angle_deg': metrics['angle_deg'],
                'has_nucleus_registration': has_nucleus_registration,
                'has_seg_registration': has_seg_registration,
            }

            # Add segmentation drift info
            if seg_drift:
                entry['seg_dy'] = seg_drift['dy']
                entry['seg_dx'] = seg_drift['dx']
                entry['seg_magnitude'] = seg_drift['magnitude']
                entry['seg_angle_deg'] = seg_drift['angle_deg']

            # Add nucleus drift info
            if nucleus_drift:
                entry['nucleus_dy'] = nucleus_drift['dy']
                entry['nucleus_dx'] = nucleus_drift['dx']
                entry['nucleus_magnitude'] = nucleus_drift['magnitude']
                entry['nucleus_angle_deg'] = nucleus_drift['angle_deg']

            drift_data.append(entry)

        except Exception as e:
            # print(f"Error loading {yaml_path}: {e}", file=sys.stderr)
            continue

    return drift_data


def collect_all_drift_trajectories(
    reference_experiment: str,
    verbose: bool = False,
    experiments_filter: List[str] = None,
) -> pd.DataFrame:
    """
    Collect drift trajectories across all experiments.

    Args:
        reference_experiment: Reference experiment for getting config directory
        verbose: Print progress information
        experiments_filter: Optional list of experiments to process (default: all)

    Returns:
        DataFrame with columns: experiment, well, round, dy, dx, magnitude, angle_deg
    """
    dataset = OpsDataset(reference_experiment)

    # Get all experiment configs
    config_dir = dataset.config_paths["exp_config_dir"]
    config_files = sorted(list(config_dir.glob("ops*.yaml")))

    print(f"Found {len(config_files)} experiment configs")

    all_drift_data = []
    experiments_processed = 0
    experiments_with_data = 0
    total_errors = 0

    # Process each experiment with progress bar
    for config_file in tqdm(config_files, desc="Processing experiments", unit="exp"):
        exp_name = config_file.stem.replace("_config", "")

        # Filter experiments if requested
        if experiments_filter and exp_name not in experiments_filter:
            continue

        experiments_processed += 1

        try:
            exp_dataset = OpsDataset(exp_name)
            wells = exp_dataset.infer_wells()

            # Process each well
            exp_has_data = False
            for well in wells:
                # Convert well format "A/1/0" to just "1"
                well_number = well.split('/')[1] if '/' in well else well.replace('A', '')

                well_drift_data = collect_drift_data_for_well(exp_name, well_number)
                if well_drift_data:
                    exp_has_data = True
                all_drift_data.extend(well_drift_data)

            if exp_has_data:
                experiments_with_data += 1

        except Exception as e:
            total_errors += 1
            if verbose:
                tqdm.write(f"  ERROR processing {exp_name}: {e}")
            continue

    # Convert to DataFrame
    if not all_drift_data:
        print(f"\nWARNING: No drift data collected!")
        print(f"  Experiments processed: {experiments_processed}")
        print(f"  Experiments with data: {experiments_with_data}")
        print(f"  Errors encountered: {total_errors}")

        # Sample experiments to diagnose the issue - check all to find which have data
        print(f"\nDiagnosing issue - scanning all experiments for registration data:")
        exps_with_register = []
        exps_with_transforms = []
        exps_with_cumulative = []

        for config_file in sorted(config_files):
            exp_name = config_file.stem.replace("_config", "")
            try:
                exp_dataset = OpsDataset(exp_name)
                register_root = exp_dataset.preprocess_in_situ / "register"

                if register_root.exists():
                    exps_with_register.append(exp_name)

                    wells = exp_dataset.infer_wells()
                    if wells:
                        well_num = wells[0].split('/')[1] if '/' in wells[0] else wells[0].replace('A', '')
                        transforms_dir = register_root / f"transforms/A{well_num}"

                        if transforms_dir.exists():
                            exps_with_transforms.append(exp_name)

                            # Check for both .yaml and .yml extensions
                            yaml_files = list(transforms_dir.glob("round*_to_round0_cumulative.yaml"))
                            yml_files = list(transforms_dir.glob("round*_to_round0_cumulative.yml"))
                            all_cumulative = yaml_files + yml_files
                            if all_cumulative:
                                exps_with_cumulative.append((exp_name, len(all_cumulative)))
            except Exception:
                pass

        print(f"  Experiments with register/ directory: {len(exps_with_register)}")
        print(f"  Experiments with transforms/ directory: {len(exps_with_transforms)}")
        print(f"  Experiments with cumulative YAMLs: {len(exps_with_cumulative)}")

        if exps_with_cumulative:
            print(f"\n  Examples with cumulative transforms:")
            for exp_name, n_files in exps_with_cumulative[:5]:
                print(f"    - {exp_name}: {n_files} cumulative YAMLs")
        elif exps_with_transforms:
            print(f"\n  First experiment with transforms (but no cumulative):")
            exp_name = exps_with_transforms[0]
            exp_dataset = OpsDataset(exp_name)
            register_root = exp_dataset.preprocess_in_situ / "register"
            wells = exp_dataset.infer_wells()
            well_num = wells[0].split('/')[1] if '/' in wells[0] else wells[0].replace('A', '')
            transforms_dir = register_root / f"transforms/A{well_num}"
            all_files = sorted(list(transforms_dir.glob("*.yaml")))
            print(f"    {exp_name}:")
            print(f"      Files in transforms dir: {len(all_files)}")
            for f in all_files[:5]:
                print(f"        - {f.name}")

        print(f"\nPossible reasons:")
        print(f"  - ISS registration has not been run yet (missing cumulative transform YAMLs)")
        print(f"  - Only {len(exps_with_cumulative)} out of {experiments_processed} experiments have completed registration")
        print(f"  - Use --verbose to see detailed error messages")
        return pd.DataFrame()

    df = pd.DataFrame(all_drift_data)

    print(f"\nCollected drift data from {len(df)} measurements")
    print(f"  Experiments: {df['experiment'].nunique()}")
    print(f"  Wells: {len(df.groupby(['experiment', 'well']))}")

    return df


def compute_summary_statistics(drift_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for drift trajectories.

    Returns DataFrame with mean and variance of magnitude and direction
    grouped by round number.
    """
    if drift_df.empty:
        return pd.DataFrame()

    # Compute statistics per round
    summary = drift_df.groupby('round').agg({
        'magnitude': ['mean', 'std', 'min', 'max', 'count'],
        'angle_deg': ['mean', 'std'],
        'dy': ['mean', 'std'],
        'dx': ['mean', 'std'],
    }).reset_index()

    # Flatten column names
    summary.columns = [
        'round',
        'magnitude_mean', 'magnitude_std', 'magnitude_min', 'magnitude_max', 'n_measurements',
        'angle_deg_mean', 'angle_deg_std',
        'dy_mean', 'dy_std',
        'dx_mean', 'dx_std',
    ]

    return summary


def create_drift_scatterplots(drift_df: pd.DataFrame, output_dir: Path):
    """
    Create scatterplots of X vs Y drift with each point being an experiment-well pair.

    Creates separate plots for each round showing INCREMENTAL drift (drift from previous round).
    Points are color-coded by experiment (viridis) and shaped by well.
    Axes limited to +/- 75 pixels.
    """
    if drift_df.empty:
        return

    # Compute incremental drift (drift from previous round)
    # Add round 0 as origin for all experiment-well pairs
    pairs = drift_df[['experiment', 'well']].drop_duplicates()
    round0 = pairs.copy()
    round0['round'] = 0
    round0['dx'] = 0.0
    round0['dy'] = 0.0
    round0['magnitude'] = 0.0
    round0['angle_deg'] = 0.0

    # Combine and sort
    cols = ['experiment', 'well', 'round', 'dx', 'dy', 'magnitude', 'angle_deg']
    full_df = pd.concat([drift_df[cols], round0[cols]], ignore_index=True)
    full_df = full_df.sort_values(['experiment', 'well', 'round'])

    # Compute incremental drift (difference from previous round)
    full_df['dx_inc'] = full_df.groupby(['experiment', 'well'])['dx'].diff()
    full_df['dy_inc'] = full_df.groupby(['experiment', 'well'])['dy'].diff()
    full_df['magnitude_inc'] = np.sqrt(full_df['dx_inc']**2 + full_df['dy_inc']**2)

    # Filter out round 0 (which has NaN diffs)
    inc_df = full_df[full_df['round'] > 0].copy()

    # Create one plot per round showing incremental drift
    rounds = sorted(inc_df['round'].unique())
    experiments = sorted(inc_df['experiment'].unique())
    wells = sorted(inc_df['well'].unique(), key=lambda x: int(x) if x.isdigit() else x)

    # Color map for experiments
    cmap = get_cmap('viridis')
    exp_colors = {exp: cmap(i / (len(experiments) - 1)) if len(experiments) > 1 else cmap(0) for i, exp in enumerate(experiments)}

    # Marker map for wells
    available_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
    # Cycle through markers if more wells than markers
    well_markers = {well: available_markers[i % len(available_markers)] for i, well in enumerate(wells)}

    # Check how many additional registration plots we'll need
    seg_data = drift_df[drift_df['has_seg_registration'] == True].copy()
    has_seg_plot = not seg_data.empty and 'seg_dx' in seg_data.columns and 'seg_dy' in seg_data.columns

    nucleus_data = drift_df[drift_df['has_nucleus_registration'] == True].copy()
    has_nucleus_plot = not nucleus_data.empty and 'nucleus_dx' in nucleus_data.columns and 'nucleus_dy' in nucleus_data.columns

    # Create figure with subplots for each round + additional registration plots
    n_rounds = len(rounds)
    n_additional = (1 if has_seg_plot else 0) + (1 if has_nucleus_plot else 0)
    n_total_plots = n_rounds + n_additional
    ncols = 3
    nrows = (n_total_plots + ncols - 1) // ncols

    # Create first figure: unlabeled points (for visual distribution)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, round_num in enumerate(rounds):
        ax = axes[idx]

        # Filter data for this round (incremental drift)
        round_data = inc_df[inc_df['round'] == round_num]

        # Plot each point (incremental drift)
        for _, row in round_data.iterrows():
            ax.scatter(
                row['dx_inc'],
                row['dy_inc'],
                color=exp_colors[row['experiment']],
                marker=well_markers[row['well']],
                alpha=0.7,
                s=60,
                edgecolors='black',
                linewidth=0.5
            )

        # Add origin
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.scatter([0], [0], color='red', s=100, marker='x', linewidth=2, label='No Drift', zorder=10)

        # Labels and title (showing it's incremental drift)
        ax.set_xlabel('X Drift (pixels)', fontsize=10)
        ax.set_ylabel('Y Drift (pixels)', fontsize=10)
        prev_round = round_num - 1
        ax.set_title(f'Round {prev_round}→{round_num} Drift (n={len(round_data)})', fontsize=11, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)

        # Set limits (incremental drift should be smaller than cumulative)
        ax.set_xlim(-75, 75)
        ax.set_ylim(-75, 75)

        # Add statistics text for incremental drift
        mean_mag = round_data['magnitude_inc'].mean()
        median_mag = round_data['magnitude_inc'].median()
        std_mag = round_data['magnitude_inc'].std()
        stats_text = f'Mean: {mean_mag:.1f}±{std_mag:.1f} px\nMedian: {median_mag:.1f} px'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add segmentation→nucleus drift plot (if data available)
    plot_idx = len(rounds)
    if has_seg_plot:
        seg_data_filtered = seg_data[['experiment', 'well', 'seg_dx', 'seg_dy', 'seg_magnitude']].drop_duplicates()

        ax = axes[plot_idx]

        # Plot each point
        for _, row in seg_data_filtered.iterrows():
            ax.scatter(
                row['seg_dx'],
                row['seg_dy'],
                color=exp_colors[row['experiment']],
                marker=well_markers[row['well']],
                alpha=0.7,
                s=60,
                edgecolors='black',
                linewidth=0.5
            )

        # Add origin
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.scatter([0], [0], color='red', s=100, marker='x', linewidth=2, label='No Drift', zorder=10)

        # Labels and title
        ax.set_xlabel('X Drift (pixels)', fontsize=10)
        ax.set_ylabel('Y Drift (pixels)', fontsize=10)
        ax.set_title(f'Segmentation→Nucleus Drift (n={len(seg_data_filtered)})', fontsize=11, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-75, 75)
        ax.set_ylim(-75, 75)

        # Add statistics
        mean_mag = seg_data_filtered['seg_magnitude'].mean()
        median_mag = seg_data_filtered['seg_magnitude'].median()
        std_mag = seg_data_filtered['seg_magnitude'].std()
        stats_text = f'Mean: {mean_mag:.1f}±{std_mag:.1f} px\nMedian: {median_mag:.1f} px'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))

        plot_idx += 1

    # Add nucleus→round0 drift plot (if data available)
    if has_nucleus_plot:
        nucleus_data_filtered = nucleus_data[['experiment', 'well', 'nucleus_dx', 'nucleus_dy', 'nucleus_magnitude']].drop_duplicates()

        ax = axes[plot_idx]

        # Plot each point
        for _, row in nucleus_data_filtered.iterrows():
            ax.scatter(
                row['nucleus_dx'],
                row['nucleus_dy'],
                color=exp_colors[row['experiment']],
                marker=well_markers[row['well']],
                alpha=0.7,
                s=60,
                edgecolors='black',
                linewidth=0.5
            )

        # Add origin
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.scatter([0], [0], color='red', s=100, marker='x', linewidth=2, label='No Drift', zorder=10)

        # Labels and title
        ax.set_xlabel('X Drift (pixels)', fontsize=10)
        ax.set_ylabel('Y Drift (pixels)', fontsize=10)
        ax.set_title(f'Nucleus→Round 0 Drift (n={len(nucleus_data_filtered)})', fontsize=11, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-75, 75)
        ax.set_ylim(-75, 75)

        # Add statistics
        mean_mag = nucleus_data_filtered['nucleus_magnitude'].mean()
        median_mag = nucleus_data_filtered['nucleus_magnitude'].median()
        std_mag = nucleus_data_filtered['nucleus_magnitude'].std()
        stats_text = f'Mean: {mean_mag:.1f}±{std_mag:.1f} px\nMedian: {median_mag:.1f} px'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        plot_idx += 1

    # Create legend handles manually
    # Experiment legend (if not too many)
    if len(experiments) <= 20:
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=exp_colors[e], label=e, markersize=8) for e in experiments]
        fig.legend(handles, [h.get_label() for h in handles], loc='upper right', bbox_to_anchor=(1.15, 0.9), title="Experiments")

    # Well legend (if not too many)
    if len(wells) <= 20:
        handles = [plt.Line2D([0], [0], marker=well_markers[w], color='w', markerfacecolor='gray', markeredgecolor='k', label=f"Well {w}", markersize=8) for w in wells]
        fig.legend(handles, [h.get_label() for h in handles], loc='lower right', bbox_to_anchor=(1.15, 0.1), title="Wells")

    # Hide extra subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    # Save figure
    plot_path = output_dir / "drift_trajectory_scatterplots_incremental.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved incremental drift scatterplots: {plot_path}")

    # Create second figure: labeled points for top 15 drifting
    create_labeled_scatterplots(inc_df, drift_df, output_dir, rounds, exp_colors, well_markers,
                                has_seg_plot, has_nucleus_plot, seg_data, nucleus_data)

    # Create zoomed-in scatterplots (+/- 20 px)
    create_drift_scatterplots_zoomed(inc_df, drift_df, output_dir, rounds, exp_colors, well_markers,
                                     has_seg_plot, has_nucleus_plot, seg_data, nucleus_data)

    # Also create a combined plot showing all rounds with trajectory lines
    create_combined_trajectory_plot(drift_df, output_dir)
    create_combined_trajectory_plot(drift_df, output_dir, labeled=True)
    create_combined_trajectory_plot(drift_df, output_dir, labeled=True, limit_axes=False)


def create_drift_scatterplots_zoomed(
    inc_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    output_dir: Path,
    rounds: List[int],
    exp_colors: Dict,
    well_markers: Dict,
    has_seg_plot: bool,
    has_nucleus_plot: bool,
    seg_data: pd.DataFrame,
    nucleus_data: pd.DataFrame
):
    """
    Create zoomed-in scatterplots of X vs Y INCREMENTAL drift (+/- 20 pixels).
    """
    n_rounds = len(rounds)
    n_additional = (1 if has_seg_plot else 0) + (1 if has_nucleus_plot else 0)
    n_total_plots = n_rounds + n_additional
    ncols = 3
    nrows = (n_total_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, round_num in enumerate(rounds):
        ax = axes[idx]
        round_data = inc_df[inc_df['round'] == round_num]

        # Plot each point (incremental drift)
        for _, row in round_data.iterrows():
            ax.scatter(
                row['dx_inc'],
                row['dy_inc'],
                color=exp_colors[row['experiment']],
                marker=well_markers[row['well']],
                alpha=0.6,
                s=60,
                edgecolors='black',
                linewidth=0.5
            )

        # Add origin
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.scatter([0], [0], color='red', s=100, marker='x', linewidth=2, label='No Drift')

        # Labels and title
        ax.set_xlabel('X Drift (pixels)', fontsize=10)
        ax.set_ylabel('Y Drift (pixels)', fontsize=10)
        prev_round = round_num - 1
        ax.set_title(f'Round {prev_round}→{round_num} Drift (Zoomed +/- 20px)', fontsize=11, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)

        # Set limits
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)

        # Add statistics text for incremental drift
        mean_mag = round_data['magnitude_inc'].mean()
        median_mag = round_data['magnitude_inc'].median()
        std_mag = round_data['magnitude_inc'].std()
        stats_text = f'Global Stats:\nMean: {mean_mag:.1f}±{std_mag:.1f} px\nMedian: {median_mag:.1f} px'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add registration plots
    plot_idx = len(rounds)

    if has_seg_plot:
        seg_data_filtered = seg_data[['experiment', 'well', 'seg_dx', 'seg_dy', 'seg_magnitude']].drop_duplicates()
        ax = axes[plot_idx]

        for _, row in seg_data_filtered.iterrows():
            ax.scatter(row['seg_dx'], row['seg_dy'], color=exp_colors[row['experiment']],
                      marker=well_markers[row['well']], alpha=0.6, s=60, edgecolors='black', linewidth=0.5)

        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.scatter([0], [0], color='red', s=100, marker='x', linewidth=2, label='No Drift')
        ax.set_xlabel('X Drift (pixels)', fontsize=10)
        ax.set_ylabel('Y Drift (pixels)', fontsize=10)
        ax.set_title(f'Segmentation→Nucleus Drift (Zoomed)', fontsize=11, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)

        mean_mag = seg_data_filtered['seg_magnitude'].mean()
        median_mag = seg_data_filtered['seg_magnitude'].median()
        std_mag = seg_data_filtered['seg_magnitude'].std()
        stats_text = f'Mean: {mean_mag:.1f}±{std_mag:.1f} px\nMedian: {median_mag:.1f} px'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))
        plot_idx += 1

    if has_nucleus_plot:
        nucleus_data_filtered = nucleus_data[['experiment', 'well', 'nucleus_dx', 'nucleus_dy', 'nucleus_magnitude']].drop_duplicates()
        ax = axes[plot_idx]

        for _, row in nucleus_data_filtered.iterrows():
            ax.scatter(row['nucleus_dx'], row['nucleus_dy'], color=exp_colors[row['experiment']],
                      marker=well_markers[row['well']], alpha=0.6, s=60, edgecolors='black', linewidth=0.5)

        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.scatter([0], [0], color='red', s=100, marker='x', linewidth=2, label='No Drift')
        ax.set_xlabel('X Drift (pixels)', fontsize=10)
        ax.set_ylabel('Y Drift (pixels)', fontsize=10)
        ax.set_title(f'Nucleus→Round 0 Drift (Zoomed)', fontsize=11, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)

        mean_mag = nucleus_data_filtered['nucleus_magnitude'].mean()
        median_mag = nucleus_data_filtered['nucleus_magnitude'].median()
        std_mag = nucleus_data_filtered['nucleus_magnitude'].std()
        stats_text = f'Mean: {mean_mag:.1f}±{std_mag:.1f} px\nMedian: {median_mag:.1f} px'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        plot_idx += 1

    # Hide extra subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plot_path = output_dir / "drift_trajectory_scatterplots_incremental_zoomed.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved zoomed incremental drift scatterplots: {plot_path}")


def create_labeled_scatterplots(
    inc_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    output_dir: Path,
    rounds: List[int],
    exp_colors: Dict,
    well_markers: Dict,
    has_seg_plot: bool,
    has_nucleus_plot: bool,
    seg_data: pd.DataFrame,
    nucleus_data: pd.DataFrame
):
    """
    Create scatterplots with top 15 highest INCREMENTAL drifting points labeled.
    """
    n_rounds = len(rounds)
    n_additional = (1 if has_seg_plot else 0) + (1 if has_nucleus_plot else 0)
    n_total_plots = n_rounds + n_additional
    ncols = 3
    nrows = (n_total_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, round_num in enumerate(rounds):
        ax = axes[idx]
        round_data = inc_df[inc_df['round'] == round_num].copy()

        # Sort by incremental magnitude to find top drift
        round_data = round_data.sort_values('magnitude_inc', ascending=False)
        top_15 = round_data.head(15)

        # Plot all points faintly
        for _, row in round_data.iterrows():
            ax.scatter(
                row['dx_inc'],
                row['dy_inc'],
                color=exp_colors[row['experiment']],
                marker=well_markers[row['well']],
                alpha=0.3,
                s=40,
                edgecolors='none'
            )

        # Plot top 15 prominently and label them
        from adjustText import adjust_text
        texts = []
        for _, row in top_15.iterrows():
            ax.scatter(
                row['dx_inc'],
                row['dy_inc'],
                color=exp_colors[row['experiment']],
                marker=well_markers[row['well']],
                alpha=1.0,
                s=80,
                edgecolors='black',
                linewidth=1.0
            )
            label = f"{row['experiment']}\n{row['well']}"
            texts.append(ax.text(row['dx_inc'], row['dy_inc'], label, fontsize=6))

        # Add origin
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.scatter([0], [0], color='red', s=100, marker='x', linewidth=2, label='No Drift')

        # Labels and title
        ax.set_xlabel('X Drift (pixels)', fontsize=10)
        ax.set_ylabel('Y Drift (pixels)', fontsize=10)
        prev_round = round_num - 1
        ax.set_title(f'Round {prev_round}→{round_num} Drift (Top 15 labeled)', fontsize=11, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)

        # Set limits
        ax.set_xlim(-75, 75)
        ax.set_ylim(-75, 75)

        # Adjust text labels to avoid overlap
        try:
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        except Exception:
            pass  # Fallback if adjustText is not installed or fails

        # Add statistics for incremental drift
        mean_mag = round_data['magnitude_inc'].mean()
        median_mag = round_data['magnitude_inc'].median()
        std_mag = round_data['magnitude_inc'].std()
        stats_text = f'Mean: {mean_mag:.1f}±{std_mag:.1f} px\nMedian: {median_mag:.1f} px'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add registration plots with labels
    plot_idx = len(rounds)

    if has_seg_plot:
        from adjustText import adjust_text
        seg_data_filtered = seg_data[['experiment', 'well', 'seg_dx', 'seg_dy', 'seg_magnitude']].drop_duplicates()
        seg_data_filtered = seg_data_filtered.sort_values('seg_magnitude', ascending=False)
        top_15 = seg_data_filtered.head(15)

        ax = axes[plot_idx]

        # Plot all points faintly
        for _, row in seg_data_filtered.iterrows():
            ax.scatter(row['seg_dx'], row['seg_dy'], color=exp_colors[row['experiment']],
                      marker=well_markers[row['well']], alpha=0.3, s=40, edgecolors='none')

        # Plot top 15 prominently and label
        texts = []
        for _, row in top_15.iterrows():
            ax.scatter(row['seg_dx'], row['seg_dy'], color=exp_colors[row['experiment']],
                      marker=well_markers[row['well']], alpha=1.0, s=80, edgecolors='black', linewidth=1.5, zorder=5)
            label = f"{row['experiment']}_W{row['well']}"
            texts.append(ax.text(row['seg_dx'], row['seg_dy'], label, fontsize=7, ha='center'))

        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.scatter([0], [0], color='red', s=100, marker='x', linewidth=2, label='No Drift')
        ax.set_xlabel('X Drift (pixels)', fontsize=10)
        ax.set_ylabel('Y Drift (pixels)', fontsize=10)
        ax.set_title(f'Segmentation→Nucleus Drift (Top 15 Labeled)', fontsize=11, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-75, 75)
        ax.set_ylim(-75, 75)

        try:
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        except Exception:
            pass

        mean_mag = seg_data_filtered['seg_magnitude'].mean()
        median_mag = seg_data_filtered['seg_magnitude'].median()
        std_mag = seg_data_filtered['seg_magnitude'].std()
        stats_text = f'Mean: {mean_mag:.1f}±{std_mag:.1f} px\nMedian: {median_mag:.1f} px'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))
        plot_idx += 1

    if has_nucleus_plot:
        from adjustText import adjust_text
        nucleus_data_filtered = nucleus_data[['experiment', 'well', 'nucleus_dx', 'nucleus_dy', 'nucleus_magnitude']].drop_duplicates()
        nucleus_data_filtered = nucleus_data_filtered.sort_values('nucleus_magnitude', ascending=False)
        top_15 = nucleus_data_filtered.head(15)

        ax = axes[plot_idx]

        # Plot all points faintly
        for _, row in nucleus_data_filtered.iterrows():
            ax.scatter(row['nucleus_dx'], row['nucleus_dy'], color=exp_colors[row['experiment']],
                      marker=well_markers[row['well']], alpha=0.3, s=40, edgecolors='none')

        # Plot top 15 prominently and label
        texts = []
        for _, row in top_15.iterrows():
            ax.scatter(row['nucleus_dx'], row['nucleus_dy'], color=exp_colors[row['experiment']],
                      marker=well_markers[row['well']], alpha=1.0, s=80, edgecolors='black', linewidth=1.5, zorder=5)
            label = f"{row['experiment']}_W{row['well']}"
            texts.append(ax.text(row['nucleus_dx'], row['nucleus_dy'], label, fontsize=7, ha='center'))

        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.scatter([0], [0], color='red', s=100, marker='x', linewidth=2, label='No Drift')
        ax.set_xlabel('X Drift (pixels)', fontsize=10)
        ax.set_ylabel('Y Drift (pixels)', fontsize=10)
        ax.set_title(f'Nucleus→Round 0 Drift (Top 15 Labeled)', fontsize=11, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-75, 75)
        ax.set_ylim(-75, 75)

        try:
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        except Exception:
            pass

        mean_mag = nucleus_data_filtered['nucleus_magnitude'].mean()
        median_mag = nucleus_data_filtered['nucleus_magnitude'].median()
        std_mag = nucleus_data_filtered['nucleus_magnitude'].std()
        stats_text = f'Mean: {mean_mag:.1f}±{std_mag:.1f} px\nMedian: {median_mag:.1f} px'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        plot_idx += 1

    # Hide extra subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plot_path = output_dir / "drift_trajectory_scatterplots_incremental_labeled.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved labeled incremental drift scatterplots: {plot_path}")


def create_combined_trajectory_plot(
    drift_df: pd.DataFrame, 
    output_dir: Path, 
    labeled: bool = False,
    limit_axes: bool = True
):
    """
    Create a single plot showing drift trajectories for all experiment-well pairs.

    Each trajectory is a line from Round 0 (origin) through each round.
    Segments are colored by round number (viridis colormap).
    
    Args:
        drift_df: DataFrame with drift data
        output_dir: Directory to save plot
        labeled: Whether to label the final points
        limit_axes: Whether to limit axes to +/- 75 pixels (default: True)
    """
    if drift_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 10))

    # Prepare segments for LineCollection
    # Group by experiment-well pair
    grouped = drift_df.groupby(['experiment', 'well'])

    # We will collect segments by round index to color them
    # rounds are 1-based. Segment i connects round i-1 to round i.
    max_round = drift_df['round'].max()
    segments_by_round = {r: [] for r in range(1, max_round + 1)}
    
    # Store final points for labeling/scatter
    final_points = [] # (x, y, exp, well)

    for (exp, well), group in grouped:
        # Sort by round
        group_sorted = group.sort_values('round')
        
        # Get coordinates including origin
        xs = [0] + group_sorted['dx'].tolist()
        ys = [0] + group_sorted['dy'].tolist()
        rounds = [0] + group_sorted['round'].tolist()
        
        # Create segments
        for i in range(1, len(xs)):
            r = rounds[i]
            if r in segments_by_round:
                segments_by_round[r].append([(xs[i-1], ys[i-1]), (xs[i], ys[i])])
        
        # Store final point
        if len(xs) > 1:
            final_points.append((xs[-1], ys[-1], exp, well))

    # Colormap for rounds
    cmap = get_cmap('viridis')
    colors = [cmap(i/(max_round-1)) if max_round > 1 else cmap(0) for i in range(max_round)]
    
    # Add segments to plot
    for r, segments in segments_by_round.items():
        if not segments:
            continue
        # r is 1-based, index 0 is round 0->1
        color = colors[r-1]
        lc = LineCollection(segments, colors=[color], linewidths=1.0, alpha=0.6, label=f"Round {r-1} → {r}")
        ax.add_collection(lc)

    # Plot final points (using darker orange/gold for better contrast)
    if final_points:
        fxs, fys, fexps, fwells = zip(*final_points)
        ax.scatter(fxs, fys, c='darkorange', s=30, alpha=0.9, edgecolors='black', linewidth=0.5, zorder=10)
        
        if labeled:
            texts = []
            for x, y, e, w in final_points:
                 texts.append(ax.text(x, y, f"{e}-{w}", fontsize=6, color='black'))
            
            try:
                from adjustText import adjust_text
                adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
            except ImportError:
                pass

    # Add origin
    ax.scatter([0], [0], color='red', s=100, marker='x', linewidth=2, label='Origin', zorder=100)

    # Labels and title
    ax.set_xlabel('X Drift (pixels)', fontsize=12)
    ax.set_ylabel('Y Drift (pixels)', fontsize=12)
    
    title_parts = ['Drift Trajectories Across All Wells', f'(n={len(grouped)} wells)']
    if labeled:
        title_parts.append('(Labeled)')
    if not limit_axes:
        title_parts.append('(Full Scale)')
        
    ax.set_title(" ".join(title_parts), fontsize=14, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    
    # Legend for rounds
    legend_elements = [plt.Line2D([0], [0], color=colors[i], lw=2, label=f'Round {i} → {i+1}') for i in range(max_round)]
    legend_elements.append(plt.Line2D([0], [0], marker='x', color='w', markeredgecolor='red', label='Origin', markersize=10))
    ax.legend(handles=legend_elements, fontsize=8, loc='best')
    
    # Auto-scale view if not limiting, or if line collections confuse autoscale
    if not limit_axes:
        ax.autoscale_view()
    
    # Set limits if requested
    if limit_axes:
        ax.set_xlim(-75, 75)
        ax.set_ylim(-75, 75)

    # Add statistics for final drift (Round 9 if available)
    final_round = drift_df['round'].max()
    final_data = drift_df[drift_df['round'] == final_round]
    if not final_data.empty:
        mean_mag = final_data['magnitude'].mean()
        std_mag = final_data['magnitude'].std()
        stats_text = f'Round {final_round} Final Drift:\n{mean_mag:.1f}±{std_mag:.1f} px'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()

    # Save figure
    suffix_parts = []
    if labeled:
        suffix_parts.append("labeled")
    if not limit_axes:
        suffix_parts.append("full_scale")
        
    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    plot_path = output_dir / f"drift_trajectories_combined{suffix}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved combined plot: {plot_path}")


def create_round_comparison_plot(drift_df: pd.DataFrame, output_dir: Path):
    """
    Create violin plots comparing drift distributions between rounds.
    
    Top subplot: X Drift distribution per round.
    Bottom subplot: Y Drift distribution per round.
    Data points are experiment means of incremental drift for that round.
    Includes statistical tests (ANOVA) and mean/std annotations.
    """
    if drift_df.empty:
        return

    # 1. Compute incremental (per-round) drift
    # Need to account for Round 0 = (0,0)
    
    # Get all experiment-well pairs
    pairs = drift_df[['experiment', 'well']].drop_duplicates()
    round0 = pairs.copy()
    round0['round'] = 0
    round0['dx'] = 0.0
    round0['dy'] = 0.0
    round0['magnitude'] = 0.0
    round0['angle_deg'] = 0.0
    
    # Combine and sort
    # Ensure columns match for concat
    cols = ['experiment', 'well', 'round', 'dx', 'dy']
    full_df = pd.concat([drift_df[cols], round0[cols]], ignore_index=True)
    full_df = full_df.sort_values(['experiment', 'well', 'round'])
    
    # Compute diff
    # We group by exp, well and take diff.
    # The diff at Round N is Val(N) - Val(N-1).
    full_df['dx_inc'] = full_df.groupby(['experiment', 'well'])['dx'].diff()
    full_df['dy_inc'] = full_df.groupby(['experiment', 'well'])['dy'].diff()
    
    # Filter out Round 0 (which has NaN diffs or is start point)
    # We want rows where round > 0
    inc_df = full_df[full_df['round'] > 0].copy()
    
    # Now group by experiment and round to get mean incremental drift for that experiment/round
    exp_means = inc_df.groupby(['experiment', 'round'])[['dx_inc', 'dy_inc']].mean().reset_index()
    
    rounds = sorted(exp_means['round'].unique())
    
    # Prepare data for plotting
    data_dx = [exp_means[exp_means['round'] == r]['dx_inc'].values for r in rounds]
    data_dy = [exp_means[exp_means['round'] == r]['dy_inc'].values for r in rounds]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # --- X Drift Plot ---
    ax_x = axes[0]
    parts = ax_x.violinplot(data_dx, positions=rounds, showmeans=True, showmedians=False)
    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_alpha(0.7)
    
    # Overlay stats
    means = [np.mean(d) for d in data_dx]
    stds = [np.std(d) for d in data_dx]
    for i, r in enumerate(rounds):
        ax_x.text(r, np.max(data_dx[i]) + 0.5, f"{means[i]:.1f}\n±{stds[i]:.1f}", 
                  ha='center', va='bottom', fontsize=10, 
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))

    # Statistical test (ANOVA + Tukey)
    try:
        f_stat, p_val = stats.f_oneway(*data_dx)
        title_suffix = f" (ANOVA p={p_val:.2e})"
        
        # If significant, run Tukey's HSD
        if p_val < 0.05:
            # Prepare data for Tukey
            values = []
            labels = []
            for i, r in enumerate(rounds):
                values.extend(data_dx[i])
                labels.extend([r] * len(data_dx[i]))
            
            tukey = pairwise_tukeyhsd(endog=values, groups=labels, alpha=0.05)
            # Add significant comparisons to plot (simplified: just note significant pairs in title or similar?
            # Or add connecting bars? Connecting bars can get messy with many groups.
            # Let's add a text box with significant pairs if not too many
            sig_pairs = []
            for i, row in enumerate(tukey.summary().data[1:]):
                if row[6]: # reject == True
                    sig_pairs.append(f"{row[0]} vs {row[1]}")
            
            if sig_pairs:
                sig_text = "Sig. Diff:\n" + "\n".join(sig_pairs[:5]) # Limit to first 5
                if len(sig_pairs) > 5:
                    sig_text += "\n..."
                ax_x.text(0.02, 0.95, sig_text, transform=ax_x.transAxes, fontsize=8,
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    except Exception:
        title_suffix = ""

    ax_x.set_ylabel('Mean Incremental X Drift (pixels)')
    ax_x.set_title(f'X Drift Distribution by Round (Incremental){title_suffix}', fontweight='bold')
    ax_x.grid(True, alpha=0.3, axis='y')
    
    # --- Y Drift Plot ---
    ax_y = axes[1]
    parts = ax_y.violinplot(data_dy, positions=rounds, showmeans=True, showmedians=False)
    for pc in parts['bodies']:
        pc.set_facecolor('#357ABD')
        pc.set_alpha(0.7)

    # Overlay stats
    means = [np.mean(d) for d in data_dy]
    stds = [np.std(d) for d in data_dy]
    for i, r in enumerate(rounds):
        ax_y.text(r, np.max(data_dy[i]) + 0.5, f"{means[i]:.1f}\n±{stds[i]:.1f}", 
                  ha='center', va='bottom', fontsize=10,
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))

    # Statistical test (ANOVA + Tukey)
    try:
        f_stat, p_val = stats.f_oneway(*data_dy)
        title_suffix = f" (ANOVA p={p_val:.2e})"
        
        if p_val < 0.05:
            values = []
            labels = []
            for i, r in enumerate(rounds):
                values.extend(data_dy[i])
                labels.extend([r] * len(data_dy[i]))
            
            tukey = pairwise_tukeyhsd(endog=values, groups=labels, alpha=0.05)
            sig_pairs = []
            for i, row in enumerate(tukey.summary().data[1:]):
                if row[6]: # reject == True
                    sig_pairs.append(f"{row[0]} vs {row[1]}")
            
            if sig_pairs:
                sig_text = "Sig. Diff:\n" + "\n".join(sig_pairs[:5])
                if len(sig_pairs) > 5:
                    sig_text += "\n..."
                ax_y.text(0.02, 0.95, sig_text, transform=ax_y.transAxes, fontsize=8,
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    except Exception:
        title_suffix = ""

    ax_y.set_ylabel('Mean Incremental Y Drift (pixels)')
    ax_y.set_title(f'Y Drift Distribution by Round (Incremental){title_suffix}', fontweight='bold')
    ax_y.set_xlabel('Round')
    ax_y.set_xticks(rounds)
    ax_y.set_xticklabels([f"Round {r}" for r in rounds])
    ax_y.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = output_dir / "drift_comparison_by_round.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved round comparison plot: {plot_path}")


def create_well_comparison_plot(drift_df: pd.DataFrame, output_dir: Path):
    """
    Create violin plots comparing drift distributions between wells.
    
    Top subplot: X Drift distribution per well.
    Bottom subplot: Y Drift distribution per well.
    Data points are experiment means for that well (averaged across rounds).
    Includes statistical tests (Kruskal-Wallis) and mean/std annotations.
    """
    if drift_df.empty:
        return

    # Calculate means per experiment per well (averaging across rounds to get "well behavior in this exp")
    well_exp_means = drift_df.groupby(['experiment', 'well'])[['dx', 'dy']].mean().reset_index()
    
    wells = sorted(well_exp_means['well'].unique(), key=lambda x: int(x) if x.isdigit() else x)
    # If too many wells, this plot might be crowded, but user requested it.
    
    # Prepare data for plotting
    data_dx = [well_exp_means[well_exp_means['well'] == w]['dx'].values for w in wells]
    data_dy = [well_exp_means[well_exp_means['well'] == w]['dy'].values for w in wells]
    
    # Check if we have enough data per well for violins
    # Violinplot needs at least one point, but looks bad with few.
    # If any well has < 2 points, maybe just scatter/strip plot? 
    # But user requested violins. Matplotlib handles single points (as lines).
    
    # Dynamic width based on number of wells
    width = max(12, len(wells) * 0.5)
    fig, axes = plt.subplots(2, 1, figsize=(width, 10), sharex=True)
    
    positions = range(len(wells))
    
    # --- X Drift Plot ---
    ax_x = axes[0]
    parts = ax_x.violinplot(data_dx, positions=positions, showmeans=True, showmedians=False)
    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_alpha(0.7)
    
    # Overlay stats
    means = [np.mean(d) if len(d) > 0 else 0 for d in data_dx]
    stds = [np.std(d) if len(d) > 1 else 0 for d in data_dx]
    
    # Only label every Nth well if too many? Or rotate labels.
    # We'll rotate stats text if crowded.
    for i, w in enumerate(wells):
        if len(data_dx[i]) > 0:
            top_val = np.max(data_dx[i])
            ax_x.text(i, top_val + 0.5, f"{means[i]:.1f}\n±{stds[i]:.1f}", 
                      ha='center', va='bottom', fontsize=10, rotation=0,
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))

    # Statistical test (ANOVA + Tukey)
    try:
        # Kruskal requires at least 2 groups
        if len(data_dx) > 1:
            # Filter empty
            valid_indices = [i for i, d in enumerate(data_dx) if len(d) > 0]
            valid_data = [data_dx[i] for i in valid_indices]
            
            if len(valid_data) > 1:
                f_stat, p_val = stats.f_oneway(*valid_data)
                title_suffix = f" (ANOVA p={p_val:.2e})"
                
                if p_val < 0.05:
                    values = []
                    labels = []
                    for i in valid_indices:
                        values.extend(data_dx[i])
                        labels.extend([wells[i]] * len(data_dx[i]))
                    
                    if len(set(labels)) > 1:
                        tukey = pairwise_tukeyhsd(endog=values, groups=labels, alpha=0.05)
                        
                        sig_diffs = []
                        summary_data = tukey.summary().data[1:]
                        for row in summary_data:
                            if row[6]: # reject == True
                                p_adj = row[3]
                                try:
                                    idx1 = wells.index(row[0])
                                    idx2 = wells.index(row[1])
                                    sig_diffs.append((idx1, idx2, p_adj))
                                except ValueError:
                                    continue
                        
                        sig_diffs.sort(key=lambda x: x[2])
                        current_h = max(max_y_vals) + 5
                        step = 3
                        
                        # Only show top 5 for wells if many comparisons
                        for idx1, idx2, p in sig_diffs[:5]:
                            label = "*" if p < 0.05 else "ns"
                            if p < 0.01: label = "**"
                            if p < 0.001: label = "***"
                            draw_significance_bracket(ax_x, idx1, idx2, current_h, 1, label)
                            current_h += step
            else:
                title_suffix = ""
        else:
            title_suffix = ""
    except Exception:
        title_suffix = ""

    ax_x.set_ylabel('Mean X Drift (pixels)')
    ax_x.set_title(f'X Drift Distribution by Well{title_suffix}', fontweight='bold')
    ax_x.grid(True, alpha=0.3, axis='y')
    
    # --- Y Drift Plot ---
    ax_y = axes[1]
    parts = ax_y.violinplot(data_dy, positions=positions, showmeans=True, showmedians=False)
    for pc in parts['bodies']:
        pc.set_facecolor('#357ABD')
        pc.set_alpha(0.7)

    means = [np.mean(d) if len(d) > 0 else 0 for d in data_dy]
    stds = [np.std(d) if len(d) > 1 else 0 for d in data_dy]
    max_y_vals = [np.max(d) if len(d) > 0 else 0 for d in data_dy]
    
    for i, w in enumerate(wells):
        if len(data_dy[i]) > 0:
            ax_y.text(i, max_y_vals[i] + 0.5, f"{means[i]:.1f}\n±{stds[i]:.1f}", 
                      ha='center', va='bottom', fontsize=10, rotation=0,
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))

    try:
        if len(data_dy) > 1:
            valid_indices = [i for i, d in enumerate(data_dy) if len(d) > 0]
            valid_data = [data_dy[i] for i in valid_indices]
            
            if len(valid_data) > 1:
                f_stat, p_val = stats.f_oneway(*valid_data)
                title_suffix = f" (ANOVA p={p_val:.2e})"
                
                if p_val < 0.05:
                    values = []
                    labels = []
                    for i in valid_indices:
                        values.extend(data_dy[i])
                        labels.extend([wells[i]] * len(data_dy[i]))
                    
                    if len(set(labels)) > 1:
                        tukey = pairwise_tukeyhsd(endog=values, groups=labels, alpha=0.05)
                        sig_diffs = []
                        summary_data = tukey.summary().data[1:]
                        for row in summary_data:
                            if row[6]:
                                p_adj = row[3]
                                try:
                                    idx1 = wells.index(row[0])
                                    idx2 = wells.index(row[1])
                                    sig_diffs.append((idx1, idx2, p_adj))
                                except ValueError:
                                    continue
                        
                        sig_diffs.sort(key=lambda x: x[2])
                        current_h = max(max_y_vals) + 5
                        step = 3
                        
                        for idx1, idx2, p in sig_diffs[:5]:
                            label = "*" if p < 0.05 else "ns"
                            if p < 0.01: label = "**"
                            if p < 0.001: label = "***"
                            draw_significance_bracket(ax_y, idx1, idx2, current_h, 1, label)
                            current_h += step
            else:
                title_suffix = ""
        else:
            title_suffix = ""
    except Exception:
        title_suffix = ""

    ax_y.set_ylabel('Mean Y Drift (pixels)')
    ax_y.set_title(f'Y Drift Distribution by Well{title_suffix}', fontweight='bold')
    ax_y.set_xlabel('Well')
    ax_y.set_xticks(positions)
    ax_y.set_xticklabels(wells, rotation=45)
    ax_y.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = output_dir / "drift_comparison_by_well.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved well comparison plot: {plot_path}")


def collect_iss_matching_data(drift_df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Collect ISS matching rates for each experiment-well pair.

    Returns DataFrame with experiment, well, total_cells_iss, cells_with_matched_reads, match_rate
    """
    matching_data = []

    exp_wells = drift_df[['experiment', 'well']].drop_duplicates()

    for _, row in exp_wells.iterrows():
        experiment = row['experiment']
        well = row['well']

        try:
            dataset = OpsDataset(experiment)
            # Use the mine results path (same as report_pipeline_status.py line 258)
            stats_path = dataset.results / "ISS" / "mine" / "plate_stats.csv"

            if not stats_path.exists():
                if verbose:
                    print(f"  Warning: Stats file not found for {experiment}: {stats_path}")
                continue

            plate_stats = pd.read_csv(stats_path, index_col=0)

            # Well format conversion: "1" -> "A/1/0" (site notation)
            well_col = f"A/{well}/0"

            if well_col not in plate_stats.columns:
                if verbose:
                    print(f"  Warning: Well column {well_col} not found for {experiment}")
                continue

            # Get pre-computed ISS metrics from the CSV
            pct_cells_with_reads = None
            pct_cells_with_matched_reads = None
            pct_matched_of_cells_with_reads = None

            if "percent_cells_with_reads" in plate_stats.index:
                pct_cells_with_reads = plate_stats.loc["percent_cells_with_reads", well_col] if pd.notna(plate_stats.loc["percent_cells_with_reads", well_col]) else None

            if "percent_cells_with_matched_reads" in plate_stats.index:
                pct_cells_with_matched_reads = plate_stats.loc["percent_cells_with_matched_reads", well_col] if pd.notna(plate_stats.loc["percent_cells_with_matched_reads", well_col]) else None

            if "percent_matched_cells_of_cells_with_reads" in plate_stats.index:
                pct_matched_of_cells_with_reads = plate_stats.loc["percent_matched_cells_of_cells_with_reads", well_col] if pd.notna(plate_stats.loc["percent_matched_cells_of_cells_with_reads", well_col]) else None

            # Skip if all metrics are None
            if all(x is None for x in [pct_cells_with_reads, pct_cells_with_matched_reads, pct_matched_of_cells_with_reads]):
                continue

            matching_data.append({
                'experiment': experiment,
                'well': well,
                'percent_cells_with_reads': pct_cells_with_reads,
                'percent_cells_with_matched_reads': pct_cells_with_matched_reads,
                'percent_matched_cells_of_cells_with_reads': pct_matched_of_cells_with_reads,
            })

        except Exception as e:
            if verbose:
                print(f"  Warning: Could not load ISS data for {experiment} well {well}: {e}")
            continue

    return pd.DataFrame(matching_data)


def create_drift_vs_iss_correlation_plot(drift_df: pd.DataFrame, output_dir: Path, verbose: bool = False):
    """
    Create scatter plots correlating total cumulative drift with ISS matching metrics.

    Creates three separate plots for:
    1. percent_cells_with_reads
    2. percent_cells_with_matched_reads
    3. percent_matched_cells_of_cells_with_reads

    Each point is an experiment-well pair.
    X-axis: Total cumulative drift (final round magnitude)
    Y-axis: ISS metric (%)
    """
    print("\n  Analyzing drift vs ISS matching correlation...")

    # Get ISS matching data
    iss_data = collect_iss_matching_data(drift_df, verbose=verbose)

    if iss_data.empty:
        print("    Warning: No ISS matching data found")
        return

    # Get final round drift for each experiment-well
    final_round = drift_df['round'].max()
    final_drift = drift_df[drift_df['round'] == final_round][['experiment', 'well', 'magnitude']].copy()
    final_drift = final_drift.rename(columns={'magnitude': 'final_drift_magnitude'})

    # Merge drift and ISS data
    merged = pd.merge(final_drift, iss_data, on=['experiment', 'well'], how='inner')

    if len(merged) == 0:
        print("    Warning: No matching data after merge")
        return

    print(f"    Found {len(merged)} experiment-well pairs with both drift and ISS data")

    # Import stats
    from scipy.stats import pearsonr, spearmanr

    # Define metrics to plot
    metrics = [
        ('percent_cells_with_reads', '% Cells with Reads', 'cells_with_reads'),
        ('percent_cells_with_matched_reads', '% Cells with Matched Reads', 'cells_with_matched_reads'),
        ('percent_matched_cells_of_cells_with_reads', '% Matched Cells (of cells with reads)', 'matched_of_reads'),
    ]

    for metric_col, metric_label, metric_short in metrics:
        # Filter out rows where this metric is None
        plot_data = merged[merged[metric_col].notna()].copy()

        if len(plot_data) == 0:
            print(f"    Warning: No data for {metric_label}")
            continue

        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))

        scatter = ax.scatter(
            plot_data['final_drift_magnitude'],
            plot_data[metric_col],
            alpha=0.6,
            s=60,
            c='steelblue',
            edgecolors='black',
            linewidth=0.5
        )

        # Compute correlation
        pearson_r, pearson_p = pearsonr(plot_data['final_drift_magnitude'], plot_data[metric_col])
        spearman_r, spearman_p = spearmanr(plot_data['final_drift_magnitude'], plot_data[metric_col])

        # Add regression line
        z = np.polyfit(plot_data['final_drift_magnitude'], plot_data[metric_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(plot_data['final_drift_magnitude'].min(), plot_data['final_drift_magnitude'].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'Linear fit (slope={z[0]:.3f})')

        # Labels and title
        ax.set_xlabel(f'Total Cumulative Drift at Round {final_round} (pixels)', fontsize=12)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_title(f'Correlation: Drift vs {metric_label}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add correlation stats
        stats_text = f'Pearson r = {pearson_r:.3f} (p={pearson_p:.2e})\nSpearman ρ = {spearman_r:.3f} (p={spearman_p:.2e})\nn = {len(plot_data)} wells'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

        ax.legend(fontsize=10)
        plt.tight_layout()

        plot_path = output_dir / f"drift_vs_iss_{metric_short}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"    Saved correlation plot: {plot_path}")
        print(f"      Pearson: r={pearson_r:.3f}, p={pearson_p:.2e}")
        print(f"      Spearman: ρ={spearman_r:.3f}, p={spearman_p:.2e}")


def create_nucleus_registration_comparison_plot(drift_df: pd.DataFrame, output_dir: Path):
    """
    Compare Round 0→1 drift for experiments with vs without nucleus registration.

    Creates scatter plots showing:
    1. Round 0→1 incremental drift (with/without nucleus correction)
    2. Nucleus drift magnitude for experiments that have it
    """
    print("\n  Analyzing nucleus registration impact on Round 0→1 drift...")

    # Filter for round 1 only
    round1_data = drift_df[drift_df['round'] == 1].copy()

    if 'has_nucleus_registration' not in round1_data.columns:
        print("    Warning: No nucleus registration data found in drift_df")
        return

    # Separate into two groups
    with_nucleus = round1_data[round1_data['has_nucleus_registration'] == True]
    without_nucleus = round1_data[round1_data['has_nucleus_registration'] == False]

    print(f"    Experiments WITH nucleus registration: {len(with_nucleus)}")
    print(f"    Experiments WITHOUT nucleus registration: {len(without_nucleus)}")

    if len(with_nucleus) == 0 and len(without_nucleus) == 0:
        print("    Warning: No data available")
        return

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Plot 1: Grouped violin plot with dots for Round 0→1 drift ---
    ax1 = axes[0]

    # Prepare data for violin plot
    plot_data = []
    plot_labels = []

    if len(without_nucleus) > 0:
        plot_data.append(without_nucleus['magnitude'].values)
        plot_labels.append('Without\nNucleus Reg')

    if len(with_nucleus) > 0:
        plot_data.append(with_nucleus['magnitude'].values)
        plot_labels.append('With\nNucleus Reg')

    if plot_data:
        # Create violin plot
        positions = list(range(len(plot_data)))
        parts = ax1.violinplot(plot_data, positions=positions, showmeans=False, showmedians=False, widths=0.7)

        # Color the violins
        colors = ['red', 'blue']
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i] if i < len(colors) else 'gray')
            pc.set_alpha(0.6)

        # Overlay individual points with jitter
        for i, (data, label) in enumerate(zip(plot_data, plot_labels)):
            # Add jitter to x-coordinates
            jitter = np.random.normal(0, 0.04, size=len(data))
            x_coords = np.full(len(data), i) + jitter
            ax1.scatter(x_coords, data, alpha=0.5, s=30,
                       c=colors[i] if i < len(colors) else 'gray',
                       edgecolors='black', linewidth=0.5, zorder=3)

        # Add mean markers (white diamonds)
        for i, data in enumerate(plot_data):
            mean_val = np.mean(data)
            ax1.plot(i, mean_val, marker='D', markersize=10, color='white',
                    markeredgecolor='black', markeredgewidth=2, zorder=4)

    ax1.set_xticks(positions)
    ax1.set_xticklabels(plot_labels, fontsize=11)
    ax1.set_ylabel('Round 0→1 Drift Magnitude (pixels)', fontsize=11)
    ax1.set_title('Round 0→1 Drift: With vs Without Nucleus Registration', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add statistical comparison
    if len(with_nucleus) > 0 and len(without_nucleus) > 0:
        from scipy.stats import mannwhitneyu
        stat, p_val = mannwhitneyu(with_nucleus['magnitude'], without_nucleus['magnitude'], alternative='two-sided')

        mean_with = with_nucleus['magnitude'].mean()
        mean_without = without_nucleus['magnitude'].mean()

        stats_text = f'n (without): {len(without_nucleus)}\nn (with): {len(with_nucleus)}\n\nMean drift:\n  Without: {mean_without:.2f} px\n  With: {mean_with:.2f} px\n\nMann-Whitney U:\n  p = {p_val:.2e}'
        ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # --- Plot 2: Nucleus drift magnitude distribution ---
    ax2 = axes[1]

    if len(with_nucleus) > 0 and 'nucleus_magnitude' in with_nucleus.columns:
        nucleus_mags = with_nucleus['nucleus_magnitude'].dropna()

        if len(nucleus_mags) > 0:
            # Histogram
            ax2.hist(nucleus_mags, bins=30, alpha=0.7, color='purple', edgecolor='black')

            # Add vertical lines for mean/median
            mean_val = nucleus_mags.mean()
            median_val = nucleus_mags.median()
            ax2.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f} px')
            ax2.axvline(median_val, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f} px')

            ax2.set_xlabel('Nucleus→Round0 Drift Magnitude (pixels)', fontsize=11)
            ax2.set_ylabel('Count', fontsize=11)
            ax2.set_title(f'Nucleus Registration Drift Distribution (n={len(nucleus_mags)})', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3, axis='y')

            # Add statistics
            stats_text = f'Mean: {mean_val:.2f} px\nMedian: {median_val:.2f} px\nStd: {nucleus_mags.std():.2f} px\nMin: {nucleus_mags.min():.2f} px\nMax: {nucleus_mags.max():.2f} px'
            ax2.text(0.65, 0.95, stats_text, transform=ax2.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    else:
        ax2.text(0.5, 0.5, 'No nucleus registration data available',
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)

    plt.tight_layout()

    plot_path = output_dir / "nucleus_registration_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"    Saved nucleus registration comparison: {plot_path}")


def create_drift_over_time_plot(drift_df: pd.DataFrame, output_dir: Path):
    """
    Create two plots showing drift over time:
    1. Stacked area chart showing incremental (round-to-round) drift contributions by round
    2. Line plot showing drift over time by well
    """
    print("\n  Creating drift over time plots...")

    if drift_df.empty:
        print("    Warning: No drift data available")
        return

    # Extract date from experiment name for sorting
    def extract_date(exp_name: str) -> str:
        """Extract date suffix from experiment name (e.g., 'ops0061_20250728' -> '20250728')"""
        match = re.search(r"_(\d{8})$", exp_name)
        return match.group(1) if match else exp_name

    # Get unique experiments and sort by date
    experiments = sorted(drift_df['experiment'].unique(), key=extract_date)
    rounds = sorted(drift_df['round'].unique())
    wells = sorted(drift_df['well'].unique(), key=lambda x: int(x) if x.isdigit() else x)

    # Compute incremental (round-to-round) drift for each experiment-well
    # Add round 0 as origin
    pairs = drift_df[['experiment', 'well']].drop_duplicates()
    round0 = pairs.copy()
    round0['round'] = 0
    round0['magnitude'] = 0.0

    cols = ['experiment', 'well', 'round', 'magnitude']
    full_df = pd.concat([drift_df[cols], round0[cols]], ignore_index=True)
    full_df = full_df.sort_values(['experiment', 'well', 'round'])

    # Compute incremental drift (difference from previous round)
    full_df['magnitude_inc'] = full_df.groupby(['experiment', 'well'])['magnitude'].diff()
    inc_df = full_df[full_df['round'] > 0].copy()

    # ========== PLOT 1: Stacked area chart by round ==========
    fig1, ax1 = plt.subplots(figsize=(20, 8))

    # For each round, compute mean incremental drift per experiment
    round_contributions = {}
    for round_num in rounds:
        round_means = []
        for exp_name in experiments:
            exp_round_data = inc_df[(inc_df['experiment'] == exp_name) & (inc_df['round'] == round_num)]
            if not exp_round_data.empty:
                round_means.append(exp_round_data['magnitude_inc'].mean())
            else:
                round_means.append(0.0)
        round_contributions[round_num] = round_means

    # Color map for rounds (viridis)
    cmap = plt.colormaps.get('viridis')
    round_colors = {r: cmap(i / max(len(rounds) - 1, 1)) for i, r in enumerate(rounds)}

    # Create stacked area chart
    x_positions = np.arange(len(experiments))
    cumulative = np.zeros(len(experiments))
    for round_num in rounds:
        values = np.array(round_contributions[round_num])
        ax1.fill_between(x_positions, cumulative, cumulative + values,
                        color=round_colors[round_num], alpha=0.7,
                        label=f'Round {round_num-1}→{round_num}')
        cumulative += values

    # Plot total incremental drift line
    ax1.plot(x_positions, cumulative, '-o', color='black', linewidth=2.5,
           markersize=8, alpha=0.9, label='Total (sum of increments)', zorder=10)

    # Formatting
    ax1.set_xlabel('Experiment (chronological)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Total Drift (pixels)', fontsize=13, fontweight='bold')
    ax1.set_title('Drift Over Time: Round-to-Round Contributions (Stacked)', fontsize=15, fontweight='bold')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(experiments, rotation=45, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(loc='upper left', fontsize=10, ncol=3)

    plt.tight_layout()
    plot_path1 = output_dir / "drift_over_time_by_round.png"
    plt.savefig(plot_path1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"    Saved drift over time (by round): {plot_path1}")

    # ========== PLOT 2: Line plot by well ==========
    fig2, ax2 = plt.subplots(figsize=(20, 8))

    # Get final round data
    final_round = max(rounds)
    final_round_data = drift_df[drift_df['round'] == final_round]

    # Color map for wells
    cmap_wells = plt.colormaps.get('tab10')
    well_colors = {w: cmap_wells(i % 10) for i, w in enumerate(wells)}

    # Plot each well as a line
    for well in wells:
        well_means = []
        for exp_name in experiments:
            exp_well_data = final_round_data[(final_round_data['experiment'] == exp_name) &
                                            (final_round_data['well'] == well)]
            if not exp_well_data.empty:
                well_means.append(exp_well_data['magnitude'].mean())
            else:
                well_means.append(np.nan)

        # Plot line
        ax2.plot(x_positions, well_means, '-o', color=well_colors[well],
                linewidth=2, markersize=6, alpha=0.8, label=f'Well {well}')

    # Plot overall mean
    total_means = []
    for exp_name in experiments:
        exp_final_data = final_round_data[final_round_data['experiment'] == exp_name]
        if not exp_final_data.empty:
            total_means.append(exp_final_data['magnitude'].mean())
        else:
            total_means.append(np.nan)

    ax2.plot(x_positions, total_means, '-o', color='black', linewidth=3,
           markersize=8, alpha=0.9, label='Mean (all wells)', zorder=10)

    # Formatting
    ax2.set_xlabel('Experiment (chronological)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Total Drift (pixels)', fontsize=13, fontweight='bold')
    ax2.set_title('Drift Over Time: By Well', fontsize=15, fontweight='bold')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(experiments, rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(loc='upper left', fontsize=9, ncol=4)

    plt.tight_layout()
    plot_path2 = output_dir / "drift_over_time_by_well.png"
    plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"    Saved drift over time (by well): {plot_path2}")


def save_results(
    drift_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    reference_experiment: str,
):
    """
    Save drift trajectory results to CSV files and generate plots.

    Saves in the same location as metrics_tracking.py outputs:
    /path/to/ops_data/ops_summary/over_time/
    """
    # Determine output directory (same as metrics_tracking.py)
    date_str = datetime.now().strftime("%Y%m%d")
    output_dir = Path(f"{BASE_PATH}/ops_data_report/{date_str}/drift_trajectories")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full drift data
    full_data_path = output_dir / "drift_trajectory_full_data.csv"
    drift_df.to_csv(full_data_path, index=False)

    # Save summary statistics
    summary_path = output_dir / "drift_trajectory_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\nResults saved:")
    print(f"  Full data: {full_data_path}")
    print(f"  Summary:   {summary_path}")

    # Create scatterplots
    create_drift_scatterplots(drift_df, output_dir)
    create_round_comparison_plot(drift_df, output_dir)
    create_well_comparison_plot(drift_df, output_dir)

    # NEW: Create correlation analysis plots
    create_drift_vs_iss_correlation_plot(drift_df, output_dir, verbose=False)
    create_nucleus_registration_comparison_plot(drift_df, output_dir)

    # NEW: Create drift over time plot
    create_drift_over_time_plot(drift_df, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Collect ISS cycle drift trajectories across all experiments"
    )
    parser.add_argument(
        "--reference-exp",
        type=str,
        required=True,
        help="Reference experiment name for determining config directory",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        help="Optional: Specific experiments to process (default: all)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed progress information",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Collecting ISS Cycle Drift Trajectories")
    print("=" * 80)
    print(f"Reference experiment: {args.reference_exp}")
    if args.experiments:
        print(f"Filtering to experiments: {args.experiments}")
    print()

    # Collect drift data
    drift_df = collect_all_drift_trajectories(
        reference_experiment=args.reference_exp,
        verbose=args.verbose,
        experiments_filter=args.experiments,
    )

    if drift_df.empty:
        print("\nNo drift data collected. Exiting.")
        return

    # Compute summary statistics
    print("\nComputing summary statistics...")
    summary_df = compute_summary_statistics(drift_df)

    # Save results
    save_results(drift_df, summary_df, args.reference_exp)

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
