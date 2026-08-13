#!/usr/bin/env python
"""
Optimize failed round configurations for ISS wells.

This script systematically tests dropout and shift combinations to find
the best configuration for each well, while validating against false positives
caused by barcode collisions or repeat matches.

Usage:
    # Single well (entropy-based validation, default)
    python -m cyclops_process.metrics.plate_stats.optimize_failed_rounds ops0108_20251209 --well "A/3/0"

    # All wells in experiment
    python -m cyclops_process.metrics.plate_stats.optimize_failed_rounds ops0108_20251209 --all-wells

    # Custom search parameters
    python -m cyclops_process.metrics.plate_stats.optimize_failed_rounds ops0108_20251209 --well "A/3/0" \
        --max-dropouts 3 --max-shifts 2 --min-effective-rounds 7

OPS-Median Validation Mode:
    Use --ops-median to switch from entropy-based validation to correlation-based
    validation against a reference guide frequency distribution built from the
    top 10 most correlated experiments across all OPS experiments.

    Validation Logic:
    - Compute baseline correlation (full 10 rounds) against reference
    - A dropout/shift config is VALID if correlation >= baseline (strict)
    - This ensures configs don't produce spurious matches with wrong guide distribution

    # Use ops-median validation (correlation-based)
    python -m cyclops_process.metrics.plate_stats.optimize_failed_rounds ops0108_20251209 --well "A/3/0" \
        --ops-median

    # Force rebuild the reference cache
    python -m cyclops_process.metrics.plate_stats.optimize_failed_rounds ops0108_20251209 --well "A/3/0" \
        --ops-median --rebuild-cache

    # Use a different OPS base directory
    python -m cyclops_process.metrics.plate_stats.optimize_failed_rounds ops0108_20251209 --well "A/3/0" \
        --ops-median --ops-dir /path/to/custom/ops/

Reference Cache:
    When using --ops-median, the script builds a reference distribution from the
    top 10 most correlated experiments across all OPS experiments. This
    reference is cached to disk for reuse:

    Cache location: tests/QC/reference_cache/
    Files:
      - reference_guide_freq_global.csv: Median frequency per barcode
      - reference_guide_freq_global_meta.csv: Top 10 experiments and their correlations

    The cache is built once and reused for all future runs. Use --rebuild-cache
    to force a rebuild if new experiments have been added to the OPS directory.

    The current experiment being optimized is automatically excluded from the
    reference to avoid self-correlation bias.
"""

import argparse
import os
import re
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations, product
from dataclasses import dataclass
from typing import Optional

from collections import Counter
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.stats import pearsonr

import sys
sys.path.insert(0, os.getcwd())

from ops_utils.data.experiment import OpsDataset
from cyclops_process.metrics.plate_stats.match_reads import (
    _get_shift_round_mapping,
    _parse_failed_rounds_spec,
)
from cyclops_process.metrics.plate_stats.match_reads import match_reads
from ops_utils.hpc.resource_manager import get_optimal_workers
from ops_utils.data.filesystem import resolve_experiment_name
from ops_utils.data.bad_experiments import get_category
from cyclops_process.paths import BASE_PATH


# Default paths for reference data
DEFAULT_OPS_DIR = f"{BASE_PATH}/"
DEFAULT_REF_TABLE = f"{BASE_PATH}/configs/library/twist1k_pool_CERES.csv"
# Cache directory for reference guide distributions
CACHE_DIR = Path(__file__).parent.parent.parent.parent / "tests" / "QC" / "reference_cache"


def calculate_entropy(seq: str) -> float:
    """Calculate Shannon entropy of a sequence (in bits). Max is 2.0 for 4 bases."""
    if not seq:
        return 0.0
    counts = Counter(seq)
    probs = np.array(list(counts.values())) / len(seq)
    return float(-np.sum(probs * np.log2(probs + 1e-9)))


# =============================================================================
# OPS Median Validation - Reference Distribution Functions
# =============================================================================

def scan_ops_experiments(ops_dir: str = DEFAULT_OPS_DIR) -> list[str]:
    """
    Scan parent directory for folders matching OPS experiment pattern.

    Returns:
        Sorted list of OPS experiment folder names (e.g., 'ops0108_20251209')
    """
    pattern = re.compile(r"^ops\d{4}_\d{8}$")
    experiments = []

    for entry in os.listdir(ops_dir):
        if pattern.match(entry):
            experiments.append(entry)

    return sorted(experiments)


def load_experiment_frequency_table(
    ops_dir: str,
    experiment: str,
) -> Optional[pd.DataFrame]:
    """
    Load aggregated frequency table for an experiment.

    Args:
        ops_dir: Base OPS directory
        experiment: Experiment name (e.g., 'ops0108_20251209')

    Returns:
        DataFrame with 'barcode' and 'count' columns, or None if not found
    """
    # Path: {experiment}/3-assembly/ISS/mine/frequency_table.csv
    freq_path = Path(ops_dir) / experiment / "3-assembly" / "ISS" / "mine" / "frequency_table.csv"

    if not freq_path.exists():
        return None

    try:
        freq_df = pd.read_csv(freq_path)
        if 'barcode' not in freq_df.columns or 'count' not in freq_df.columns:
            return None
        return freq_df
    except Exception:
        return None


def append_barcode_freq(ref_tbl: pd.DataFrame, ops_path: str, exp_name: str) -> pd.DataFrame:
    """
    Append BASELINE barcode frequency data from an OPS experiment to reference table.

    Computes baseline frequencies by:
    1. Loading raw reads from all wells
    2. Matching against codebook using ALL 10 rounds (no dropouts) via match_reads()
    3. Computing frequency table from matched reads via frequency_table()

    This ensures the reference distribution is always based on baseline (no dropout configs).

    Args:
        ref_tbl: Reference table with 'barcode' column
        ops_path: Full path to OPS experiment folder
        exp_name: Experiment name for column naming

    Returns:
        Updated reference table with count column for this experiment
    """
    from glob import glob
    from cyclops_process.metrics.plate_stats.match_reads import match_reads
    from cyclops_process.metrics.plate_stats.iss_metrics import frequency_table

    # Find all wells with reads in base_calling/mine directory
    # Per-well reads: 1-preprocess/in_situ_sequencing/base_calling/mine/*_reads.csv (A1_reads.csv, etc.)
    reads_pattern = os.path.join(ops_path, "1-preprocess/in_situ_sequencing/base_calling/mine/*_reads.csv")
    reads_files = glob(reads_pattern)

    if not reads_files:
        # Try single combined reads file
        single_reads = os.path.join(ops_path, "1-preprocess/in_situ_sequencing/base_calling/mine/reads.csv")
        if os.path.exists(single_reads):
            reads_files = [single_reads]
        else:
            return ref_tbl

    # Load codebook via OpsDataset so codebook_round_offset and overrides are applied
    try:
        dataset = OpsDataset(exp_name, method="mine")
        codebook_db = dataset.load_codebook()
        if 'sgRNA' not in codebook_db.columns:
            return ref_tbl
    except Exception:
        return ref_tbl

    # Baseline uses all 10 rounds (no dropouts)
    baseline_rounds = list(range(10))

    # Load and match reads from all wells using baseline (all 10 rounds)
    all_matched_reads = []
    for reads_file in reads_files:
        try:
            reads_df = pd.read_csv(reads_file)
            if 'barcode' not in reads_df.columns:
                continue

            # Use the same match_reads function as metrics.py
            matched = match_reads(reads_df, codebook_db, iss_rounds=baseline_rounds, debug=False)
            if not matched.empty:
                all_matched_reads.append(matched)
        except Exception:
            continue

    if not all_matched_reads:
        return ref_tbl

    # Combine all matched reads and compute frequency table
    combined_reads = pd.concat(all_matched_reads, ignore_index=True)
    freq_tbl = frequency_table(combined_reads)

    # Barcode length is always 10 for baseline
    bc_len = 10

    ref_tbl_short = ref_tbl.copy()
    ref_tbl_short['barcode_tmp'] = ref_tbl_short['barcode'].str[:bc_len]

    merged = ref_tbl_short.merge(
        freq_tbl,
        left_on='barcode_tmp',
        right_on='barcode',
        how='left',
        suffixes=('_ref', '')
    )

    merged['count'] = merged['count'].fillna(0).astype(int)
    ref_tbl_short[exp_name] = merged['count'].tolist()

    return ref_tbl_short.drop(columns=['barcode_tmp'])


def compute_experiment_correlations(
    ops_dir: str,
    ref_table_path: str = DEFAULT_REF_TABLE,
    exclude_experiment: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute pairwise correlations between OPS experiments using library barcodes.

    Uses the same approach as iss_read_freq_analysis.py:
    - Load library reference table (twist1k_pool_CERES.csv)
    - Match library barcodes to experiment frequency tables
    - Compute correlation using log2((count + 1) / (mean + 1))

    Args:
        ops_dir: Base OPS directory
        ref_table_path: Path to library reference table
        exclude_experiment: Experiment to exclude (the one being optimized)
        verbose: Print progress

    Returns:
        DataFrame with columns: experiment, avg_correlation, frequency_data
    """
    # Load library reference table
    ref_tbl = pd.read_csv(ref_table_path)
    ref_tbl = ref_tbl[['barcode', 'Gene name']]

    if verbose:
        print(f"  Loaded {len(ref_tbl)} library barcodes from {ref_table_path}")

    experiments = scan_ops_experiments(ops_dir)

    if exclude_experiment and exclude_experiment in experiments:
        experiments.remove(exclude_experiment)

    if verbose:
        print(f"  Scanning {len(experiments)} experiments...")

    # Compute baseline frequency data from each experiment (all 10 rounds, no dropouts)
    for exp in tqdm(experiments, desc="Computing baseline frequencies", disable=not verbose):
        ops_path = os.path.join(ops_dir, exp)
        ref_tbl = append_barcode_freq(ref_tbl, ops_path, exp)

    # Get OPS columns (those that were successfully added)
    ops_columns = [col for col in ref_tbl.columns if col not in ['barcode', 'Gene name']]

    if len(ops_columns) < 3:
        if verbose:
            print(f"  Warning: Only {len(ops_columns)} experiments found with data")
        return pd.DataFrame()

    if verbose:
        print(f"  Found {len(ops_columns)} experiments with frequency data")

    # Compute pairwise correlations using same formula as iss_read_freq_analysis.py
    n_exp = len(ops_columns)
    corr_matrix = np.zeros((n_exp, n_exp))

    if verbose:
        print(f"  Computing pairwise correlations ({n_exp * (n_exp - 1) // 2} pairs)...")

    for i, ops1 in enumerate(ops_columns):
        for j, ops2 in enumerate(ops_columns):
            if i == j:
                corr_matrix[i, j] = 1.0
            elif i < j:
                # Match iss_read_freq_analysis.py normalization exactly
                x = np.log2((ref_tbl[ops1].values + 1) / (ref_tbl[ops1].values.mean() + 1))
                y = np.log2((ref_tbl[ops2].values + 1) / (ref_tbl[ops2].values.mean() + 1))

                valid = ~np.isinf(x) & ~np.isnan(x) & ~np.isinf(y) & ~np.isnan(y)
                if valid.sum() > 10:
                    corr, _ = pearsonr(x[valid], y[valid])
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr

    # Compute average correlation per experiment
    avg_corr = (corr_matrix.sum(axis=1) - 1) / (n_exp - 1)

    results = pd.DataFrame({
        'experiment': ops_columns,
        'avg_correlation': avg_corr,
    })

    # Store frequency data for building reference (normalize to frequencies)
    # Use library barcodes as keys (truncated to 10 chars like the library)
    library_barcodes = ref_tbl['barcode'].str[:10].tolist()
    results['freq_data'] = [
        dict(zip(library_barcodes, (ref_tbl[exp] / ref_tbl[exp].sum()).values))
        for exp in ops_columns
    ]

    # Also return ref_tbl and ops_columns for heatmap generation
    results.attrs['ref_tbl'] = ref_tbl
    results.attrs['ops_columns'] = ops_columns

    return results.sort_values('avg_correlation', ascending=False)


def build_reference_distribution(
    correlation_df: pd.DataFrame,
    top_n: int = 10,
) -> pd.Series:
    """
    Build reference guide frequency distribution from top N correlated experiments.

    Args:
        correlation_df: Output from compute_experiment_correlations
        top_n: Number of top experiments to use

    Returns:
        Series with barcode -> median frequency
    """
    if len(correlation_df) == 0:
        return pd.Series(dtype=float)

    top_experiments = correlation_df.head(top_n)

    # Combine frequency data from top experiments
    freq_dfs = []
    for _, row in top_experiments.iterrows():
        freq_series = pd.Series(row['freq_data'])
        freq_dfs.append(freq_series)

    if not freq_dfs:
        return pd.Series(dtype=float)

    # Compute median frequency per barcode
    combined = pd.concat(freq_dfs, axis=1)
    median_freq = combined.median(axis=1)

    return median_freq


# Experiment exclusion lists — loaded from shared bad_experiment.yaml
BAD_EXPERIMENTS = get_category("bad")
ISS_ONLY_EXPERIMENT_NUMS = get_category("iss_only")


def _create_single_heatmap(
    ref_tbl: pd.DataFrame,
    ops_columns: list[str],
    output_path: Path,
    title: str,
    verbose: bool = True,
) -> None:
    """Helper function to create a single correlation heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_exp = len(ops_columns)
    if n_exp < 2:
        if verbose:
            print(f"  Not enough experiments for heatmap: {output_path.name}")
        return

    # Compute correlation matrix
    corr_matrix = np.zeros((n_exp, n_exp))
    for i, ops1 in enumerate(ops_columns):
        for j, ops2 in enumerate(ops_columns):
            if i == j:
                corr_matrix[i, j] = 1.0
            elif i < j:
                x = np.log2((ref_tbl[ops1].values + 1) / (ref_tbl[ops1].values.mean() + 1))
                y = np.log2((ref_tbl[ops2].values + 1) / (ref_tbl[ops2].values.mean() + 1))
                valid = ~np.isinf(x) & ~np.isnan(x) & ~np.isinf(y) & ~np.isnan(y)
                if valid.sum() > 10:
                    corr, _ = pearsonr(x[valid], y[valid])
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr

    # Sort by ops number
    def get_ops_num(name):
        match = re.match(r'ops(\d+)', name)
        return int(match.group(1)) if match else 9999

    sorted_indices = sorted(range(n_exp), key=lambda i: get_ops_num(ops_columns[i]))
    sorted_exp_cols = [ops_columns[i] for i in sorted_indices]
    sorted_corr = corr_matrix[np.ix_(sorted_indices, sorted_indices)]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(22, 20))
    sns.heatmap(
        sorted_corr,
        xticklabels=sorted_exp_cols,
        yticklabels=sorted_exp_cols,
        cmap='RdYlBu_r',
        center=0.5,
        vmin=0,
        vmax=1,
        square=True,
        ax=ax,
        cbar_kws={'label': 'Pearson Correlation', 'shrink': 0.8}
    )
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Experiment', fontsize=11)
    ax.set_ylabel('Experiment', fontsize=11)
    plt.xticks(rotation=90, fontsize=5)
    plt.yticks(rotation=0, fontsize=5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Correlation heatmap saved to: {output_path}")

    # Print statistics
    upper_tri = sorted_corr[np.triu_indices_from(sorted_corr, k=1)]
    avg_corr = sorted_corr.mean(axis=1)

    if verbose:
        print(f"\n  Correlation Statistics ({output_path.stem}):")
        print(f"    Mean: {upper_tri.mean():.3f}, Std: {upper_tri.std():.3f}")
        print(f"    Min:  {upper_tri.min():.3f}, Max: {upper_tri.max():.3f}")

        # Outliers
        outlier_indices = np.argsort(avg_corr)[:3]
        print(f"\n  Lowest avg correlation (potential outliers):")
        for idx in outlier_indices:
            print(f"    {sorted_exp_cols[idx]}: avg r={avg_corr[idx]:.3f}")


def generate_correlation_heatmap(
    ref_tbl: pd.DataFrame,
    ops_columns: list[str],
    output_path: Path,
    verbose: bool = True,
) -> None:
    """
    Generate correlation heatmaps from experiment frequency data.

    Creates two heatmaps:
    1. All experiments (original behavior)
    2. Filtered: excludes BAD_EXPERIMENTS and ISS_ONLY_EXPERIMENT_NUMS

    Args:
        ref_tbl: DataFrame with barcode column and experiment frequency columns
        ops_columns: List of experiment column names
        output_path: Path to save the main heatmap PNG (filtered version uses _filtered suffix)
        verbose: Print progress
    """
    # 1. Generate heatmap with ALL experiments
    _create_single_heatmap(
        ref_tbl=ref_tbl,
        ops_columns=ops_columns,
        output_path=output_path,
        title='Pairwise Experiment Correlation\n(Baseline Frequencies, All 10 Rounds, Log2-Normalized)',
        verbose=verbose,
    )

    # 2. Generate filtered heatmap (exclude bad experiments and ISS-only experiments)
    def should_exclude(exp_name: str) -> bool:
        # Check if in BAD_EXPERIMENTS list
        if exp_name in BAD_EXPERIMENTS:
            return True
        # Check if experiment number is in ISS_ONLY list
        match = re.match(r'ops(\d+)', exp_name)
        if match:
            exp_num = int(match.group(1))
            if exp_num in ISS_ONLY_EXPERIMENT_NUMS:
                return True
        return False

    filtered_columns = [col for col in ops_columns if not should_exclude(col)]
    excluded_count = len(ops_columns) - len(filtered_columns)

    if verbose:
        print(f"\n  Filtered heatmap: excluded {excluded_count} experiments (bad + ISS-only)")

    # Create filtered output path
    filtered_path = output_path.parent / f"{output_path.stem}_filtered{output_path.suffix}"

    _create_single_heatmap(
        ref_tbl=ref_tbl,
        ops_columns=filtered_columns,
        output_path=filtered_path,
        title='Pairwise Experiment Correlation (Filtered)\n(Excludes Bad Experiments & ISS-Only)',
        verbose=verbose,
    )


def get_or_build_reference_cache(
    ops_dir: str = DEFAULT_OPS_DIR,
    exclude_experiment: Optional[str] = None,
    rebuild: bool = False,
    top_n: int = 10,
    verbose: bool = True,
) -> tuple[pd.Series, list[str]]:
    """
    Load cached reference distribution or build and cache it.

    The reference is built from the top N most correlated experiments
    across ALL experiments in the OPS directory. This provides a robust
    reference for validating guide frequency distributions.

    Args:
        ops_dir: Base OPS directory
        exclude_experiment: Experiment to exclude from reference (the one being optimized)
        rebuild: Force rebuild of cache
        top_n: Number of top experiments to use
        verbose: Print progress

    Returns:
        Tuple of (reference_freq_series, list_of_top_experiments)
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Cache files use "baseline" suffix to indicate they're computed from baseline (all 10 rounds)
    cache_path = CACHE_DIR / "reference_guide_freq_global_baseline.csv"
    meta_path = CACHE_DIR / "reference_guide_freq_global_baseline_meta.csv"

    # Check if cache exists and is valid
    if not rebuild and cache_path.exists() and meta_path.exists():
        if verbose:
            print(f"  Loading cached reference from {cache_path}")
        ref_df = pd.read_csv(cache_path)
        meta_df = pd.read_csv(meta_path)
        ref_series = pd.Series(ref_df['median_freq'].values, index=ref_df['barcode'])
        top_experiments = meta_df['experiment'].tolist()
        if verbose:
            print(f"\n  Top {len(top_experiments)} reference experiments (by avg correlation):")
            for _, row in meta_df.iterrows():
                print(f"    {row['experiment']}: r={row['avg_correlation']:.3f}")
        return ref_series, top_experiments

    # Build reference
    if verbose:
        print(f"  Building global reference distribution...")

    corr_df = compute_experiment_correlations(
        ops_dir, exclude_experiment=exclude_experiment, verbose=verbose
    )

    if len(corr_df) == 0:
        if verbose:
            print(f"  Warning: Could not build reference distribution")
        return pd.Series(dtype=float), []

    ref_series = build_reference_distribution(corr_df, top_n=top_n)
    top_experiments = corr_df.head(top_n)['experiment'].tolist()

    # Save cache
    ref_df = pd.DataFrame({
        'barcode': ref_series.index,
        'median_freq': ref_series.values
    })
    ref_df.to_csv(cache_path, index=False)

    top_correlations = corr_df.head(top_n)['avg_correlation'].values
    meta_df = pd.DataFrame({
        'experiment': top_experiments,
        'avg_correlation': top_correlations
    })
    meta_df.to_csv(meta_path, index=False)

    if verbose:
        print(f"  Cached reference to {cache_path}")
        print(f"\n  Top {top_n} reference experiments (by avg correlation):")
        for exp, corr in zip(top_experiments, top_correlations):
            print(f"    {exp}: r={corr:.3f}")

    # Cache the full frequency matrix for fast heatmap regeneration
    if hasattr(corr_df, 'attrs') and 'ref_tbl' in corr_df.attrs:
        freq_matrix_path = CACHE_DIR / "experiment_frequency_matrix.csv"
        corr_df.attrs['ref_tbl'].to_csv(freq_matrix_path, index=False)
        if verbose:
            print(f"  Cached frequency matrix to {freq_matrix_path}")

        # Generate correlation heatmap
        heatmap_path = CACHE_DIR / "experiment_correlation_heatmap.png"
        generate_correlation_heatmap(
            corr_df.attrs['ref_tbl'],
            corr_df.attrs['ops_columns'],
            heatmap_path,
            verbose=verbose
        )

    return ref_series, top_experiments


def validate_against_reference(
    matched_df: pd.DataFrame,
    reference_freq: pd.Series,
    read_positions: list[int],
) -> tuple[float, str]:
    """
    Compute correlation of matched reads against reference guide distribution.

    Uses Pearson correlation on log2-transformed frequencies. The log2 transform
    handles the long-tailed distribution of guide counts, ensuring all guides
    contribute more equally to the correlation.

    The reference uses 10-char library barcodes. We truncate both the reference
    and current barcodes to the effective round count for comparison.

    Note: Validation is done later by comparing to baseline correlation.
    A configuration is valid if correlation >= baseline (strict, no tolerance).

    Args:
        matched_df: DataFrame of matched reads with 'barcode' column
        reference_freq: Reference frequency distribution (barcode -> freq)
        read_positions: Positions used for barcode extraction

    Returns:
        Tuple of (correlation, notes)
    """
    if len(matched_df) == 0:
        return 0.0, "No matches"

    if len(reference_freq) == 0:
        return 0.0, "No reference available"

    # Extract barcodes at read positions from matched reads
    matched_barcodes = matched_df['barcode'].apply(
        lambda x: ''.join(x[p] for p in read_positions if p < len(x))
    )

    # Compute frequency distribution of current barcodes
    barcode_counts = matched_barcodes.value_counts()
    current_freq = barcode_counts / barcode_counts.sum()

    # Extract reference barcodes at the SAME read_positions as current barcodes
    # (not just first N chars - must account for dropout positions)
    ref_truncated = reference_freq.copy()
    ref_truncated.index = ref_truncated.index.astype(str).map(
        lambda x: ''.join(x[p] for p in read_positions if p < len(x))
    )
    # Aggregate any collisions from position extraction
    ref_truncated = ref_truncated.groupby(ref_truncated.index).sum()

    # Find common barcodes
    common_barcodes = set(current_freq.index) & set(ref_truncated.index)

    if len(common_barcodes) < 50:
        return 0.0, f"Only {len(common_barcodes)} common barcodes with reference (need 50+)"

    # Compute correlation on common barcodes
    current_arr = current_freq.reindex(list(common_barcodes)).fillna(0).values
    ref_arr = ref_truncated.reindex(list(common_barcodes)).fillna(0).values

    # Log2 transform to handle long-tailed distribution of guide frequencies
    # This ensures all guides contribute more equally (not dominated by high-count guides)
    epsilon = 1e-9
    current_log2 = np.log2(current_arr + epsilon)
    ref_log2 = np.log2(ref_arr + epsilon)

    # Pearson correlation on log2-transformed frequencies
    corr, p_value = pearsonr(current_log2, ref_log2)

    pct_common = len(common_barcodes) / len(reference_freq) * 100
    notes = f"Reference corr: {corr:.3f}, {pct_common:.1f}% guides in common"

    return corr, notes


def _get_col(df, candidates: list[str]) -> str | None:
    """Find first matching column name from candidates list."""
    if df is None:
        return None
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _format_baseline(baseline: float, fmt: str = ".1f") -> str:
    """Format baseline value for comparison display."""
    if fmt == ".3f":
        return f" (baseline: {baseline:.3f})"
    elif fmt == "int":
        return f" (baseline: {int(baseline):,})"
    else:
        return f" (baseline: {baseline:.1f})"


@dataclass
class EntropyStats:
    """Statistics from entropy analysis for comparison."""
    slope: float  # reads per entropy bit
    r2: float  # R-squared
    p_value: float
    top_guide_ratio: float  # top guide count / mean count
    top_gene_ratio: float  # top gene count / mean count
    mean_entropy: float  # mean barcode entropy
    unique_barcodes: int
    unique_genes: int


def compute_entropy_stats(
    matched_df: pd.DataFrame,
    codebook_db: pd.DataFrame,
    read_positions: list[int],
    codebook_positions: list[int],
    gene_index_db: pd.DataFrame = None,
) -> Optional["EntropyStats"]:
    """
    Compute entropy statistics for a matched DataFrame.

    Returns EntropyStats or None if not enough data.
    """
    from scipy import stats as scipy_stats

    if len(matched_df) == 0:
        return None

    # Extract barcodes at read positions
    matched_barcodes = matched_df['barcode'].apply(
        lambda x: ''.join(x[p] for p in read_positions if p < len(x))
    )

    # Create codebook lookup
    codebook_truncated = codebook_db['sgRNA'].apply(
        lambda x: ''.join(x[p] for p in codebook_positions if p < len(x))
    )
    codebook_lookup = dict(zip(codebook_truncated, codebook_db['gene_id']))

    # Create gene_name lookup
    gene_name_lookup = {}
    if gene_index_db is not None and 'barcode' in gene_index_db.columns:
        gene_name_col = _get_col(gene_index_db, ["Gene name", "dep_map_gene_name", "gene_name"])
        if gene_name_col:
            gi = gene_index_db.copy()
            gi['barcode_truncated'] = gi['barcode'].apply(
                lambda x: ''.join(x[p] for p in codebook_positions if p < len(x))
            )
            gene_name_lookup = dict(zip(gi['barcode_truncated'], gi[gene_name_col]))

    # Exclude designed spike-in controls (ratio>1 in codebook) from bias stats.
    # These are high-frequency, often low-entropy barcodes by design and would
    # otherwise dominate the slope/top-guide-ratio metrics.
    spiked_barcodes_set = _get_spiked_barcodes(codebook_db, codebook_positions)
    if spiked_barcodes_set:
        matched_barcodes = matched_barcodes[~matched_barcodes.isin(spiked_barcodes_set)]

    # Calculate entropy for each unique barcode
    barcode_counts = matched_barcodes.value_counts()
    barcode_entropies = {bc: calculate_entropy(bc) for bc in barcode_counts.index}

    if len(barcode_counts) < 3:
        return None

    # Entropy vs count regression
    entropies_arr = np.array([barcode_entropies[bc] for bc in barcode_counts.index])
    counts_arr = barcode_counts.values
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(entropies_arr, counts_arr)

    # Top guide ratio
    top_count = barcode_counts.iloc[0]
    mean_count = barcode_counts.mean()
    top_guide_ratio = top_count / max(mean_count, 1)

    # Top gene ratio
    if gene_name_lookup:
        matched_genes = matched_barcodes.map(gene_name_lookup)
    else:
        matched_genes = matched_barcodes.map(codebook_lookup)
    matched_genes_valid = matched_genes.dropna()
    matched_genes_real = matched_genes_valid[matched_genes_valid.astype(str) != '-1']
    gene_counts = matched_genes_real.value_counts()

    if len(gene_counts) > 0:
        top_gene_count = gene_counts.iloc[0]
        gene_mean = gene_counts.mean()
        top_gene_ratio = top_gene_count / max(gene_mean, 1)
    else:
        top_gene_ratio = 0.0

    return EntropyStats(
        slope=slope,
        r2=r_value ** 2,
        p_value=p_value,
        top_guide_ratio=top_guide_ratio,
        top_gene_ratio=top_gene_ratio,
        mean_entropy=np.mean(list(barcode_entropies.values())),
        unique_barcodes=len(barcode_counts),
        unique_genes=len(gene_counts),
    )


def full_entropy_analysis(
    matched_df: pd.DataFrame,
    codebook_db: pd.DataFrame,
    read_positions: list[int],
    codebook_positions: list[int],
    config_str: str,
    gene_index_db: pd.DataFrame = None,
    baseline_stats: Optional[EntropyStats] = None,
    verbose: bool = False,
) -> None:
    """
    Run comprehensive entropy analysis on matched reads and print detailed report.

    This analyzes:
    - Entropy vs count regression (like metrics.py) to detect systematic bias
    - Distribution of barcode entropies (verbose only)
    - Top low-entropy barcodes (verbose only)
    - Per-round base entropy (verbose only)
    - Gene-level entropy statistics (verbose only)

    Args:
        baseline_stats: Optional EntropyStats from baseline for comparison
        verbose: If True, show detailed barcode/gene analysis sections
    """
    from scipy import stats as scipy_stats

    if len(matched_df) == 0:
        print("  No matched reads to analyze.")
        return

    print(f"\n{'='*70}")
    print(f"FULL ENTROPY ANALYSIS: {config_str}")
    if baseline_stats:
        print(f"  (showing comparisons vs baseline)")
    print(f"{'='*70}")

    # Extract barcodes at read positions
    matched_barcodes = matched_df['barcode'].apply(
        lambda x: ''.join(x[p] for p in read_positions if p < len(x))
    )

    # Create codebook lookup for gene_id (needed for gene mapping throughout)
    codebook_truncated = codebook_db['sgRNA'].apply(
        lambda x: ''.join(x[p] for p in codebook_positions if p < len(x))
    )
    codebook_lookup = dict(zip(codebook_truncated, codebook_db['gene_id']))

    # Create gene_name lookup from gene_index if available
    gene_name_lookup = {}
    if gene_index_db is not None and 'barcode' in gene_index_db.columns:
        gene_name_col = _get_col(gene_index_db, ["Gene name", "dep_map_gene_name", "gene_name"])
        if gene_name_col:
            gi = gene_index_db.copy()
            gi['barcode_truncated'] = gi['barcode'].apply(
                lambda x: ''.join(x[p] for p in codebook_positions if p < len(x))
            )
            gene_name_lookup = dict(zip(gi['barcode_truncated'], gi[gene_name_col]))

    # Exclude designed spike-in controls (ratio>1 in codebook) so reported entropy
    # stats match the QC view and aren't dominated by C1/C2-style low-entropy spikes.
    spiked_barcodes_set = _get_spiked_barcodes(codebook_db, codebook_positions)
    if spiked_barcodes_set:
        n_before = len(matched_barcodes)
        matched_barcodes = matched_barcodes[~matched_barcodes.isin(spiked_barcodes_set)]
        n_excluded = n_before - len(matched_barcodes)
        if n_excluded > 0:
            print(f"  (Excluding {n_excluded:,} reads from {len(spiked_barcodes_set)} spike-in barcodes for entropy analysis)")

    total_reads = len(matched_barcodes)

    # Calculate entropy for each unique barcode
    barcode_counts = matched_barcodes.value_counts()
    barcode_entropies = {bc: calculate_entropy(bc) for bc in barcode_counts.index}

    # === ENTROPY vs COUNT REGRESSION (like metrics.py) ===
    # This detects if low-entropy barcodes are systematically over-represented
    entropies_arr = np.array([barcode_entropies[bc] for bc in barcode_counts.index])
    counts_arr = barcode_counts.values

    if len(entropies_arr) >= 3:
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(entropies_arr, counts_arr)
        r2 = r_value ** 2

        print(f"\n--- ENTROPY vs COUNT REGRESSION (bias detection) ---")
        slope_cmp = _format_baseline(baseline_stats.slope) if baseline_stats else ""
        print(f"  Slope: {slope:.1f} reads per entropy bit{slope_cmp}")
        r2_cmp = _format_baseline(baseline_stats.r2, ".3f") if baseline_stats else ""
        print(f"  R²: {r2:.3f}{r2_cmp}")
        print(f"  P-value: {p_value:.2e}")

        # Interpretation
        if p_value < 0.05:
            if slope < -50:
                bias_text = "⚠️ STRONG NEGATIVE BIAS: Low-entropy barcodes are significantly over-represented"
            elif slope < 0:
                bias_text = "⚠️ Negative bias: Some preference for low-entropy barcodes"
            elif slope > 50:
                bias_text = "✓ Positive slope: High-entropy barcodes favored (unusual)"
            else:
                bias_text = "✓ Weak relationship between entropy and count"
        else:
            bias_text = "✓ No significant entropy bias detected (p > 0.05)"

        print(f"  Interpretation: {bias_text}")

        # Show top guide info (like metrics.py) - use MEAN not median
        top_bc = barcode_counts.index[0]
        top_count = barcode_counts.iloc[0]
        mean_count = barcode_counts.mean()
        ratio = top_count / max(mean_count, 1)
        # Use gene_name if available, fallback to gene_id
        top_gene_name = gene_name_lookup.get(top_bc) or codebook_lookup.get(top_bc, "NOT_IN_CODEBOOK")

        warn_guide = "⚠️ " if ratio > 10 else ""
        ratio_cmp = _format_baseline(baseline_stats.top_guide_ratio) if baseline_stats else ""
        print(f"  {warn_guide}Top guide: '{top_gene_name}' [{top_bc}] has {top_count:,} reads ({ratio:.1f}x mean{ratio_cmp})")

        # Top gene analysis (aggregate all barcodes for each gene) - exclude '-1'
        # Use gene_name_lookup if available for proper gene names
        if gene_name_lookup:
            matched_genes = matched_barcodes.map(gene_name_lookup)
        else:
            matched_genes = matched_barcodes.map(codebook_lookup)
        matched_genes_valid = matched_genes.dropna()
        matched_genes_real = matched_genes_valid[matched_genes_valid.astype(str) != '-1']
        gene_counts = matched_genes_real.value_counts()
        if len(gene_counts) > 0:
            top_gene_display = gene_counts.index[0]
            top_gene_count = gene_counts.iloc[0]
            gene_mean = gene_counts.mean()
            gene_ratio = top_gene_count / max(gene_mean, 1)
            warn_gene = "⚠️ " if gene_ratio > 2 else ""
            gene_ratio_cmp = _format_baseline(baseline_stats.top_gene_ratio) if baseline_stats else ""
            print(f"  {warn_gene}Top gene: '{top_gene_display}' has {top_gene_count:,} reads ({gene_ratio:.1f}x mean{gene_ratio_cmp})")

        print(f"{'='*70}")

    # Detailed analysis sections (verbose only)
    if verbose:
        all_entropies = list(barcode_entropies.values())
        print(f"\n--- BARCODE ENTROPY DISTRIBUTION ---")
        bc_cmp = _format_baseline(baseline_stats.unique_barcodes, "int") if baseline_stats else ""
        print(f"  Total unique barcodes: {len(all_entropies):,}{bc_cmp}")
        print(f"  Entropy range: {min(all_entropies):.3f} - {max(all_entropies):.3f} bits")
        mean_ent = np.mean(all_entropies)
        mean_ent_cmp = _format_baseline(baseline_stats.mean_entropy, ".3f") if baseline_stats else ""
        print(f"  Mean entropy: {mean_ent:.3f} bits{mean_ent_cmp}")
        print(f"  Median entropy: {np.median(all_entropies):.3f} bits")

        # Entropy percentiles
        percentiles = [10, 25, 50, 75, 90]
        pct_values = np.percentile(all_entropies, percentiles)
        print(f"  Percentiles: " + ", ".join([f"p{p}={v:.2f}" for p, v in zip(percentiles, pct_values)]))

        # Count barcodes by entropy ranges
        low_entropy = sum(1 for e in all_entropies if e < 1.0)
        mid_entropy = sum(1 for e in all_entropies if 1.0 <= e < 1.5)
        high_entropy = sum(1 for e in all_entropies if e >= 1.5)
        print(f"\n  Entropy categories:")
        print(f"    Low (<1.0 bits):  {low_entropy:,} ({low_entropy/len(all_entropies)*100:.1f}%)")
        print(f"    Mid (1.0-1.5):    {mid_entropy:,} ({mid_entropy/len(all_entropies)*100:.1f}%)")
        print(f"    High (>=1.5):     {high_entropy:,} ({high_entropy/len(all_entropies)*100:.1f}%)")

        # Top 10 most frequent barcodes with entropy
        print(f"\n--- TOP 10 MOST FREQUENT BARCODES ---")
        print(f"  {'Barcode':<12} {'Count':>10} {'%Reads':>8} {'Entropy':>8} {'Gene':<20}")
        print(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*20}")
        for bc, count in barcode_counts.head(10).items():
            entropy = barcode_entropies[bc]
            gene = gene_name_lookup.get(bc) or codebook_lookup.get(bc, "NOT_IN_CODEBOOK")
            pct = count / total_reads * 100
            flag = " ⚠️ LOW" if entropy < 1.0 else ""
            print(f"  {bc:<12} {count:>10,} {pct:>7.2f}% {entropy:>7.2f}{flag}  {gene:<20}")

        # Top 10 LOW entropy barcodes (potential repeat sequences)
        low_entropy_bcs = [(bc, barcode_entropies[bc], barcode_counts[bc])
                           for bc in barcode_counts.index if barcode_entropies[bc] < 1.2]
        low_entropy_bcs.sort(key=lambda x: (-x[2], x[1]))  # Sort by count desc, then entropy asc

        if low_entropy_bcs:
            print(f"\n--- TOP 10 LOW-ENTROPY BARCODES (potential repeats) ---")
            print(f"  {'Barcode':<12} {'Entropy':>8} {'Count':>10} {'%Reads':>8} {'Gene':<20}")
            print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*8} {'-'*20}")
            for bc, entropy, count in low_entropy_bcs[:10]:
                gene = gene_name_lookup.get(bc) or codebook_lookup.get(bc, "NOT_IN_CODEBOOK")
                pct = count / total_reads * 100
                print(f"  {bc:<12} {entropy:>7.3f} {count:>10,} {pct:>7.2f}%  {gene:<20}")

            # Summary of low-entropy impact
            low_entropy_reads = sum(c for _, _, c in low_entropy_bcs)
            print(f"\n  Low-entropy barcodes account for {low_entropy_reads:,} reads ({low_entropy_reads/total_reads*100:.1f}% of matches)")

        # Per-round base entropy analysis
        print(f"\n--- PER-ROUND BASE ENTROPY ---")
        print(f"  (Higher = more uniform distribution, Max = 2.0 bits for 4 bases)")
        round_entropies = []
        for i, pos in enumerate(read_positions):
            bases_at_pos = matched_barcodes.str[i]
            base_counts = bases_at_pos.value_counts()
            probs = base_counts.values / base_counts.sum()
            pos_entropy = float(-np.sum(probs * np.log2(probs + 1e-9)))
            round_entropies.append((pos, pos_entropy, base_counts.to_dict()))

        for pos, entropy, dist in round_entropies:
            dist_str = " ".join([f"{b}:{dist.get(b, 0)/sum(dist.values())*100:.0f}%" for b in "ACGT"])
            flag = " ⚠️ BIASED" if entropy < 1.5 else ""
            print(f"  Round {pos}: {entropy:.3f} bits  [{dist_str}]{flag}")

        avg_round_entropy = np.mean([e for _, e, _ in round_entropies])
        print(f"\n  Average per-round entropy: {avg_round_entropy:.3f} bits")

        # Gene-level analysis - exclude '-1' (not in codebook)
        # Use gene_name_lookup if available for proper gene names
        if gene_name_lookup:
            matched_genes = matched_barcodes.map(gene_name_lookup)
        else:
            matched_genes = matched_barcodes.map(codebook_lookup)
        matched_genes_valid = matched_genes.dropna()
        matched_genes_real = matched_genes_valid[matched_genes_valid.astype(str) != '-1']
        gene_counts = matched_genes_real.value_counts()

        print(f"\n--- GENE-LEVEL STATISTICS ---")
        genes_cmp = _format_baseline(baseline_stats.unique_genes, "int") if baseline_stats else ""
        print(f"  Unique genes matched: {len(gene_counts):,}{genes_cmp}")
        real_reads = len(matched_genes_real)
        print(f"  Top 5 genes: {gene_counts.head(5).sum()/real_reads*100:.1f}% of reads" if real_reads > 0 else "  Top 5 genes: 0%")
        print(f"  Top 10 genes: {gene_counts.head(10).sum()/real_reads*100:.1f}% of reads" if real_reads > 0 else "  Top 10 genes: 0%")

        # Gene concentration curve
        if real_reads > 0:
            cumulative = gene_counts.cumsum() / real_reads * 100
            for pct in [25, 50, 75, 90]:
                genes_needed = (cumulative <= pct).sum() + 1
                print(f"  {pct}% of reads covered by top {genes_needed:,} genes")

        print(f"\n{'='*70}")


@dataclass
class MatchResult:
    """Results from a matching configuration."""
    config_type: str  # e.g., "dropout", "shift", "dropout+shift"
    dropouts: list[int]
    shifts: list[int]
    cells_with_matched_reads: int
    cells_with_reads: int
    match_rate: float  # percent_matched_cells_of_cells_with_reads
    unique_barcodes: int
    unique_genes: int
    gene_coverage: float  # percentage of codebook genes matched
    effective_rounds: int
    top5_concentration: float  # % of reads in top 5 genes
    avg_entropy: float  # average entropy of matched barcodes (0-2 bits)
    entropy_slope: float  # slope of counts vs entropy regression (negative = low-entropy bias)
    entropy_pvalue: float  # p-value of entropy-count regression (low = significant relationship)
    top_barcode: str  # most frequent matched barcode
    top_barcode_entropy: float  # entropy of most frequent barcode
    top_barcode_count: int  # count of most frequent barcode
    is_valid: bool  # False if likely false positive
    validation_notes: str
    correlation: float = 0.0  # ops-median correlation (0 if not using ops-median)
    top_gene_vs_median: float = 0.0  # ratio of top gene count to median gene count
    top_guide_vs_median: float = 0.0  # ratio of top guide count to median guide count
    weighted_entropy: float = 0.0  # read-count-weighted avg entropy (lower = bias toward low-entropy barcodes)


def extract_positions(barcode: str, positions: list[int]) -> str:
    """Extract characters at specified positions from a barcode."""
    return ''.join(barcode[p] for p in positions if p < len(barcode))


def _get_spiked_barcodes(
    codebook_db: pd.DataFrame,
    codebook_positions: list[int],
) -> set[str]:
    """Return position-extracted barcodes for codebook entries with ratio > min.

    Some pools encode pos_ctrl/neg_ctrl spike-ins via a `ratio` column
    (e.g. ratio=5 vs ratio=1 for binders). These designed high-frequency, often
    low-entropy barcodes confound entropy/concentration QC and should be
    excluded from bias detection.
    """
    if "ratio" not in codebook_db.columns:
        return set()
    barcode_col = "sgRNA" if "sgRNA" in codebook_db.columns else "barcode"
    if barcode_col not in codebook_db.columns:
        return set()
    min_ratio = codebook_db["ratio"].min()
    spiked = codebook_db.loc[codebook_db["ratio"] > min_ratio, barcode_col]
    if len(spiked) == 0:
        return set()
    return {extract_positions(x, codebook_positions) for x in spiked}


def analyze_base_distribution(reads_df: pd.DataFrame, num_rounds: int = 10) -> dict:
    """
    Analyze base distribution per round to identify problematic rounds.

    Returns dict mapping round -> (dominant_base, dominant_pct, is_problematic)
    """
    results = {}
    for i in range(num_rounds):
        col = f"peak_{i}"
        if col in reads_df.columns:
            counts = reads_df[col].value_counts(normalize=True) * 100
            dominant_base = counts.idxmax()
            dominant_pct = counts.max()
            # >40% single base indicates problematic round
            is_problematic = dominant_pct > 40
            results[i] = {
                "dominant_base": dominant_base,
                "dominant_pct": dominant_pct,
                "is_problematic": is_problematic,
                "distribution": {base: counts.get(base, 0) for base in "ACGT"}
            }
    return results


def get_codebook_uniqueness(
    codebook_db: pd.DataFrame,
    positions: list[int],
    barcode_col: str = "sgRNA"
) -> tuple[int, set]:
    """
    Check codebook uniqueness at specified positions.

    Returns (unique_count, set_of_truncated_barcodes)
    """
    truncated = codebook_db[barcode_col].apply(lambda x: extract_positions(x, positions))
    return truncated.nunique(), set(truncated)


def check_entropy_slope_bias(
    barcode_counts: pd.Series,
    barcode_entropies: dict,
) -> tuple[bool, str | None, float, float]:
    """
    Check if entropy slope indicates low-entropy barcode over-representation.

    Returns (is_biased, note_string, slope, p_value) where is_biased=True means config should be invalidated.

    Validation criteria:
    - Hard cutoff: slope < -50 is always flagged (extreme bias)
    - P-value check: any negative slope with p < 0.05 indicates significant bias
    """
    from scipy import stats as scipy_stats

    if len(barcode_counts) < 10 or len(barcode_entropies) < 10:
        return False, None, 0.0, 0.0

    entropies_arr = np.array([barcode_entropies[bc] for bc in barcode_counts.index if bc in barcode_entropies])
    counts_arr = np.array([barcode_counts[bc] for bc in barcode_counts.index if bc in barcode_entropies])

    if len(entropies_arr) < 10:
        return False, None, 0.0, 0.0

    slope, _, r_value, p_value, _ = scipy_stats.linregress(entropies_arr, counts_arr)
    r_squared = r_value ** 2

    # Hard cutoff for extreme slopes - always flag regardless of p-value
    if slope < -50:
        return True, f"Extreme entropy bias: slope={slope:.1f} reads/bit (hard cutoff)", slope, p_value

    # P-value only check: if entropy-count correlation is statistically significant, flag as biased
    # We require slope < 0 (negative correlation) AND p < 0.05 (significant)
    if slope < 0 and p_value < 0.05:
        return True, f"Entropy bias: slope={slope:.1f} reads/bit, p={p_value:.2e} (significant negative correlation)", slope, p_value

    return False, None, slope, p_value


def validate_matches(
    matched_df: pd.DataFrame,
    codebook_db: pd.DataFrame,
    read_positions: list[int],
    codebook_positions: list[int],
    total_guides: int,
    barcode_col: str = "sgRNA",
    use_ops_median: bool = False,
    reference_freq: Optional[pd.Series] = None,
    baseline_correlation: float = 0.0,
    baseline_top_guide_vs_median: float = 0.0,
    baseline_weighted_entropy: float = 0.0,
    baseline_entropy_slope: float = 0.0,
    baseline_entropy_pvalue: float = 1.0,
) -> tuple[int, int, float, float, float, str, float, int, bool, str, float, float, float, float, float, float]:
    """
    Validate matched reads against codebook to detect false positives.

    Returns:
        (unique_barcodes, unique_genes, gene_coverage_pct, top5_concentration,
         avg_entropy, top_barcode, top_barcode_entropy, top_barcode_count,
         is_valid, notes, correlation, entropy_slope, top_gene_vs_median, top_guide_vs_median,
         weighted_entropy, entropy_pvalue)
    """
    from scipy import stats as scipy_stats

    if len(matched_df) == 0:
        return 0, 0, 0.0, 0.0, 0.0, "", 0.0, 0, True, "No matches", 0.0, 0.0, 0.0, 0.0, 0.0, 1.0

    notes = []

    # Extract barcodes at read positions
    matched_barcodes = matched_df['barcode'].apply(
        lambda x: extract_positions(x, read_positions)
    )
    unique_barcodes = matched_barcodes.nunique()

    # Create codebook lookup at codebook positions
    codebook_truncated = codebook_db[barcode_col].apply(
        lambda x: extract_positions(x, codebook_positions)
    )
    codebook_unique = codebook_truncated.nunique()

    # Check for codebook collisions (fewer unique truncated than total guides)
    if codebook_unique < total_guides:
        collision_rate = (total_guides - codebook_unique) / total_guides * 100
        notes.append(f"Codebook has {collision_rate:.1f}% collisions at these positions")

    # Map barcodes to genes
    codebook_lookup = dict(zip(codebook_truncated, codebook_db['gene_id']))
    matched_genes = matched_barcodes.map(codebook_lookup)
    matched_genes_valid = matched_genes.dropna()

    # Filter out '-1' (not in codebook) from gene statistics - these aren't real gene matches
    matched_genes_real = matched_genes_valid[matched_genes_valid.astype(str) != '-1']

    unique_genes = int(matched_genes_real.nunique()) if len(matched_genes_real) > 0 else 0
    gene_coverage = unique_genes / total_guides * 100 if total_guides > 0 else 0

    # Check gene concentration (top 5 genes should not dominate) - excluding '-1'
    if len(matched_genes_real) > 0:
        gene_counts = matched_genes_real.value_counts()
        top5_concentration = gene_counts.head(5).sum() / len(matched_genes_real) * 100
    else:
        gene_counts = pd.Series(dtype=int)
        top5_concentration = 0.0

    # Exclude designed spike-in controls (e.g. pos_ctrl/neg_ctrl with
    # ratio>1) from bias detection. These are high-frequency, often low-entropy
    # barcodes by design and would falsely trip the entropy-slope and top1/median
    # QC checks. Match-rate / unique_barcodes / unique_genes above are unaffected.
    spiked_barcodes_set = _get_spiked_barcodes(codebook_db, codebook_positions)
    if spiked_barcodes_set:
        n_before = len(matched_barcodes)
        matched_barcodes = matched_barcodes[~matched_barcodes.isin(spiked_barcodes_set)]
        n_excluded = n_before - len(matched_barcodes)
        if n_excluded > 0:
            notes.append(
                f"Excluded {n_excluded:,} reads from {len(spiked_barcodes_set)} "
                f"spike-in barcodes (ratio>min) before bias QC"
            )

    # === ENTROPY ANALYSIS ===
    # Calculate entropy for matched barcodes
    barcode_counts = matched_barcodes.value_counts()
    top_barcode = barcode_counts.index[0] if len(barcode_counts) > 0 else ""
    top_barcode_count = int(barcode_counts.iloc[0]) if len(barcode_counts) > 0 else 0
    top_barcode_entropy = calculate_entropy(top_barcode) if top_barcode else 0.0

    # Calculate average entropy across unique matched barcodes
    unique_matched = matched_barcodes.unique()
    if len(unique_matched) > 0:
        entropies = [calculate_entropy(bc) for bc in unique_matched]
        avg_entropy = float(np.mean(entropies))
        barcode_entropies = dict(zip(unique_matched, entropies))
    else:
        avg_entropy = 0.0
        barcode_entropies = {}

    # Calculate WEIGHTED average entropy (weighted by read count)
    # Low-entropy bias = weighted entropy drops because low-entropy barcodes have more reads
    if len(barcode_counts) > 0 and len(barcode_entropies) > 0:
        total_reads = float(barcode_counts.sum())
        weighted_sum = sum(
            barcode_entropies.get(bc, 0.0) * count
            for bc, count in barcode_counts.items()
        )
        weighted_entropy = weighted_sum / total_reads if total_reads > 0 else 0.0
    else:
        weighted_entropy = 0.0

    # Get gene name for top barcode if available
    top_gene = codebook_lookup.get(top_barcode, "unknown")

    # =========================================================================
    # VALIDATION: Match rate is the PRIMARY metric
    # Only invalidate for clear signs of false positives, not for having many matches
    # =========================================================================
    is_valid = True

    # Track correlation for ops-median mode (0.0 if not using ops-median)
    correlation = 0.0

    if use_ops_median and reference_freq is not None and len(reference_freq) > 0:
        # =====================================================================
        # OPS MEDIAN VALIDATION: Compare correlation against baseline
        # A config is valid if correlation >= baseline (strict)
        # =====================================================================
        correlation, ref_notes = validate_against_reference(
            matched_df, reference_freq, read_positions
        )
        notes.append(ref_notes)

        # Validate against baseline (must be >= baseline, no tolerance)
        if baseline_correlation > 0:
            if correlation < baseline_correlation:
                is_valid = False
                notes.append(f"Correlation {correlation:.3f} < baseline ({baseline_correlation:.3f})")

        # Compute entropy slope and p-value for bias detection
        _, _, entropy_slope, entropy_pvalue = check_entropy_slope_bias(barcode_counts, barcode_entropies)

        # P-VALUE CHECK: reject if entropy-count correlation is statistically significant
        # Hard cutoff at slope < -50, otherwise check p-value < 0.05 with negative slope
        PVALUE_THRESHOLD = 0.05
        if entropy_slope < -50:
            is_valid = False
            notes.append(f"Extreme entropy bias: slope={entropy_slope:.1f} (hard cutoff)")
        elif entropy_slope < 0 and entropy_pvalue < PVALUE_THRESHOLD:
            is_valid = False
            notes.append(f"Entropy bias: slope={entropy_slope:.1f}, p={entropy_pvalue:.2e} (significant negative correlation)")

        # TOP GUIDE vs MEDIAN CHECK: invalidate if ratio exceeds 1.2x baseline OR absolute cap
        # Catches single dominant guide bias (e.g., contamination, systematic error)
        # Threshold justification (from 95 baseline experiments):
        #   - Median ratio: 3.6x, 75th pctl: 4.2x, 95th pctl: ~11x
        #   - 10x threshold catches 7.4% of experiments (pathological cases with low median counts)
        #   - Experiments >10x tend to have very low median counts (1-30), indicating poor data quality
        RATIO_ALLOWANCE = 1.2
        ABSOLUTE_MAX_RATIO = 10.0  # Hard cap - catches pathological cases while allowing normal variation
        if len(barcode_counts) >= 1:
            median_guide_count = float(barcode_counts.median())
            top_guide_count = float(barcode_counts.iloc[0])
            current_top_guide_ratio = top_guide_count / max(median_guide_count, 1)
        else:
            current_top_guide_ratio = 0.0

        # Check both relative threshold (1.1x baseline) and absolute cap (10x)
        if current_top_guide_ratio > ABSOLUTE_MAX_RATIO:
            is_valid = False
            notes.append(f"Top1/median ratio {current_top_guide_ratio:.1f}x > {ABSOLUTE_MAX_RATIO}x absolute cap")
        elif baseline_top_guide_vs_median > 0 and current_top_guide_ratio > baseline_top_guide_vs_median * RATIO_ALLOWANCE:
            is_valid = False
            notes.append(f"Top1/median ratio {current_top_guide_ratio:.1f}x > {RATIO_ALLOWANCE}x baseline ({baseline_top_guide_vs_median:.1f}x)")

    else:
        # =====================================================================
        # ENTROPY-BASED VALIDATION (original method)
        # =====================================================================

        # INVALIDATING CHECK 1: Per-round base entropy collapse
        # If any round has <0.8 bits entropy in matched barcodes, the config is wrong
        # (e.g., wrong shift aligning to constant region)
        if len(read_positions) > 0 and len(matched_df) > 500:
            for i, pos in enumerate(read_positions):
                if i < len(matched_barcodes.iloc[0]) if len(matched_barcodes) > 0 else False:
                    bases_at_pos = matched_barcodes.str[i]
                    base_counts = bases_at_pos.value_counts()
                    probs = base_counts.values / base_counts.sum()
                    pos_entropy = float(-np.sum(probs * np.log2(probs + 1e-9)))
                    if pos_entropy < 0.8:
                        dominant_base = base_counts.idxmax()
                        dominant_pct = base_counts.max() / base_counts.sum() * 100
                        is_valid = False
                        notes.append(f"Round {pos} entropy collapse: {pos_entropy:.2f} bits ({dominant_base}={dominant_pct:.0f}%)")
                        break

        # INVALIDATING CHECK 2: Extreme single-barcode dominance with low entropy
        # Only if one barcode is >15% of ALL matches AND has very low entropy (<1.0)
        if len(barcode_counts) > 0 and len(matched_df) > 500:
            top_guide_pct = top_barcode_count / len(matched_df) * 100
            if top_guide_pct > 15 and top_barcode_entropy < 1.0:
                is_valid = False
                notes.append(f"Single low-entropy barcode dominates: '{top_barcode}' is {top_guide_pct:.1f}% of matches ({top_barcode_entropy:.2f} bits)")

        # INVALIDATING CHECK 3: Low-entropy barcodes account for too many matches
        # If barcodes with entropy <1.0 bits make up >5% of matches, config is suspect
        if len(barcode_counts) > 0 and len(matched_df) > 1000 and len(barcode_entropies) > 0:
            low_entropy_reads = sum(
                count for bc, count in barcode_counts.items()
                if bc in barcode_entropies and barcode_entropies[bc] < 1.0
            )
            low_entropy_pct = low_entropy_reads / len(matched_df) * 100
            if low_entropy_pct > 5:
                is_valid = False
                notes.append(f"Low-entropy barcodes (<1.0 bits) account for {low_entropy_pct:.1f}% of matches ({low_entropy_reads:,} reads)")

        # INVALIDATING CHECK 4: Entropy regression slope too negative
        is_biased, bias_note, entropy_slope, entropy_pvalue = check_entropy_slope_bias(barcode_counts, barcode_entropies)
        if is_biased:
            is_valid = False
            notes.append(bias_note)

    # INFORMATIONAL NOTES (do not invalidate - just report for debugging)
    # Top gene concentration
    if len(gene_counts) > 0:
        top_gene_count = int(gene_counts.iloc[0])
        top_gene_name = str(gene_counts.index[0])
        gene_mean = gene_counts.mean() if len(gene_counts) > 1 else 1
        gene_ratio = top_gene_count / max(gene_mean, 1)
        if gene_ratio > 10:
            notes.append(f"Info: Top gene '{top_gene_name}' is {gene_ratio:.0f}x mean ({top_gene_count:,} reads)")

    # Top barcode concentration
    if len(barcode_counts) > 0:
        mean_count = barcode_counts.mean() if len(barcode_counts) > 1 else 1
        top_ratio = top_barcode_count / max(mean_count, 1)
        if top_ratio > 20:
            notes.append(f"Info: Top barcode '{top_barcode}' [{top_gene}] is {top_ratio:.0f}x mean")

    # Compute top gene vs median gene ratio (skew metric)
    if len(gene_counts) > 1:
        median_gene_count = float(gene_counts.median())
        top_gene_vs_median = float(gene_counts.iloc[0] / max(median_gene_count, 1))
    else:
        top_gene_vs_median = 0.0

    # Compute top 5 guides sum / median ratio (skew metric)
    # Note: variable name says "top_guide" but we use sum(top5)/median for robustness
    if len(barcode_counts) >= 5:
        median_guide_count = float(barcode_counts.median())
        top5_sum = float(barcode_counts.iloc[:5].sum())
        top_guide_vs_median = top5_sum / max(median_guide_count, 1)
    else:
        top_guide_vs_median = 0.0

    return unique_barcodes, unique_genes, gene_coverage, top5_concentration, avg_entropy, top_barcode, top_barcode_entropy, top_barcode_count, is_valid, "; ".join(notes) if notes else "OK", correlation, entropy_slope, top_gene_vs_median, top_guide_vs_median, weighted_entropy, entropy_pvalue


def test_configuration(
    reads_df: pd.DataFrame,
    codebook_db: pd.DataFrame,
    well: str,
    dropouts: list[int],
    shifts: list[int],
    iss_rounds: list[int],
    total_guides: int,
    use_ops_median: bool = False,
    reference_freq: Optional[pd.Series] = None,
    baseline_correlation: float = 0.0,
    baseline_top_guide_vs_median: float = 0.0,
    baseline_weighted_entropy: float = 0.0,
    baseline_entropy_slope: float = 0.0,
    baseline_entropy_pvalue: float = 1.0,
) -> MatchResult:
    """Test a specific dropout/shift configuration and return validated results."""

    total_reads = len(reads_df)

    # Determine config type
    if dropouts and shifts:
        config_type = "dropout+shift"
    elif dropouts:
        config_type = "dropout"
    elif shifts:
        config_type = "shift"
    else:
        config_type = "baseline"

    # Build failed_rounds config
    if dropouts or shifts:
        failed_config = {well: {}}
        if dropouts:
            failed_config[well]["dropout"] = dropouts
        if shifts:
            failed_config[well]["shift"] = shifts
    else:
        failed_config = None

    # Get the position mapping
    if shifts:
        read_positions, codebook_positions = _get_shift_round_mapping(
            iss_rounds, well, failed_config
        )
    else:
        # For dropout-only, positions are the same (excluding dropouts)
        effective_rounds = [r for r in iss_rounds if r not in dropouts]
        read_positions = effective_rounds
        codebook_positions = effective_rounds

    effective_round_count = len(read_positions)

    # Run matching
    if failed_config:
        matched = match_reads(
            reads_df.copy(),
            codebook_db,
            iss_rounds=iss_rounds,
            well_name=well,
            failed_rounds_by_well=failed_config,
        )
    else:
        matched = match_reads(
            reads_df.copy(),
            codebook_db,
            iss_rounds=iss_rounds,
            well_name=well,
        )

    matched_reads_count = len(matched)

    # Calculate percent_matched_cells_of_cells_with_reads (consistent with metrics.py)
    # cells_with_reads = unique cells that have ANY read
    # cells_with_matched_reads = unique cells that have at least one MATCHED read
    if "cell" in reads_df.columns:
        cells_with_reads = reads_df["cell"][reads_df["cell"] > 0].nunique()
        cells_with_matched_reads = matched["cell"][matched["cell"] > 0].nunique() if len(matched) > 0 and "cell" in matched.columns else 0
        match_rate = cells_with_matched_reads / cells_with_reads * 100 if cells_with_reads > 0 else 0
    else:
        cells_with_reads = 0
        cells_with_matched_reads = 0
        match_rate = 0

    # Validate results (entropy-based or ops-median based)
    (unique_barcodes, unique_genes, gene_coverage, top5_conc,
     avg_entropy, top_barcode, top_barcode_entropy, top_barcode_count,
     is_valid, notes, correlation, entropy_slope,
     top_gene_vs_median, top_guide_vs_median, weighted_entropy, entropy_pvalue) = validate_matches(
        matched, codebook_db, read_positions, codebook_positions, total_guides,
        use_ops_median=use_ops_median,
        reference_freq=reference_freq,
        baseline_correlation=baseline_correlation,
        baseline_top_guide_vs_median=baseline_top_guide_vs_median,
        baseline_weighted_entropy=baseline_weighted_entropy,
        baseline_entropy_slope=baseline_entropy_slope,
        baseline_entropy_pvalue=baseline_entropy_pvalue,
    )

    return MatchResult(
        config_type=config_type,
        dropouts=dropouts,
        shifts=shifts,
        cells_with_matched_reads=cells_with_matched_reads,
        cells_with_reads=cells_with_reads,
        match_rate=match_rate,
        unique_barcodes=unique_barcodes,
        unique_genes=unique_genes,
        gene_coverage=gene_coverage,
        effective_rounds=effective_round_count,
        top5_concentration=top5_conc,
        avg_entropy=avg_entropy,
        entropy_slope=entropy_slope,
        entropy_pvalue=entropy_pvalue,
        top_barcode=top_barcode,
        top_barcode_entropy=top_barcode_entropy,
        top_barcode_count=top_barcode_count,
        is_valid=is_valid,
        validation_notes=notes,
        correlation=correlation,
        top_gene_vs_median=top_gene_vs_median,
        top_guide_vs_median=top_guide_vs_median,
        weighted_entropy=weighted_entropy,
    )


def optimize_well(
    experiment: str,
    well: str,
    max_dropouts: int = 3,
    max_shifts: int = 2,
    min_effective_rounds: int = 7,
    improvement_threshold: float = 1.0,
    verbose: bool = True,
    n_jobs: int = None,
    use_ops_median: bool = False,
    rebuild_cache: bool = False,
    ops_dir: str = DEFAULT_OPS_DIR,
    entropy_verbose: bool = False,
) -> pd.DataFrame:
    """
    Find optimal failed round configuration for a single well.

    Uses a 2-phase approach:
    1. Test all single dropouts and single shifts individually
    2. Only combine rounds that show >= improvement_threshold improvement

    When using ops-median validation, configs are validated by comparing their
    correlation to baseline - a config is valid if correlation >= baseline (strict).

    Args:
        experiment: Experiment name
        well: Well identifier (e.g., "A/3/0")
        max_dropouts: Maximum dropout rounds to combine (1-2)
        max_shifts: Maximum shift rounds to combine (0-2)
        min_effective_rounds: Minimum effective rounds required (default 7)
        improvement_threshold: Min % improvement to consider a round promising (default 1.0)
        verbose: Print progress
        n_jobs: Number of parallel workers (None = auto-detect using resource_manager)
        use_ops_median: Use correlation-based validation against reference distribution
        rebuild_cache: Force rebuild of reference cache
        ops_dir: Base OPS directory for scanning experiments

    Returns:
        DataFrame with all tested configurations, sorted by valid match rate
    """
    dataset = OpsDataset(experiment, method="mine")
    codebook_db = dataset.load_codebook()
    reads_df = pd.read_csv(dataset.append_well("reads", well))

    # Load gene_index for gene name lookups (optional, falls back to gene_id)
    gene_index_db = None
    try:
        gene_index_db = dataset.load_gene_index()
    except Exception:
        pass  # Will use gene_id if gene_index not available

    total_reads = len(reads_df)
    total_guides = len(codebook_db)

    # ISS reads at most 10 rounds even if codebook barcodes are longer.
    barcode_col = "sgRNA" if "sgRNA" in codebook_db.columns else "barcode"
    n_iss_rounds = min(10, int(codebook_db[barcode_col].str.len().max()))
    constant_rounds = []
    iss_rounds = list(range(n_iss_rounds))

    # Calculate cells_with_reads for display (used in match_rate denominator)
    cells_with_reads_total = reads_df["cell"][reads_df["cell"] > 0].nunique() if "cell" in reads_df.columns else 0

    # Get optimal number of workers for CPU-bound task
    if n_jobs is None:
        n_jobs = get_optimal_workers(use_gpu=False, model_ram_gb=0.5, data_ram_gb=0.5, verbose=False)

    if verbose:
        print(f"\n{'='*70}")
        print(f"Optimizing: {experiment} - {well}")
        print(f"{'='*70}")
        print(f"Total reads: {total_reads:,}")
        print(f"Cells with reads: {cells_with_reads_total:,}")
        print(f"Total guides: {total_guides:,}")
        if constant_rounds:
            print(f"Constant rounds excluded: {constant_rounds} (single base in codebook)")
        print(f"ISS rounds: {iss_rounds}")
        print(f"Min effective rounds: {min_effective_rounds}")
        print(f"Improvement threshold: {improvement_threshold}%")
        print(f"Parallel workers: {n_jobs}")
        print(f"Validation mode: {'OPS Median (baseline-relative correlation)' if use_ops_median else 'Entropy-based'}")
        if use_ops_median:
            print(f"  Validation: configs valid if correlation >= baseline (strict)")
        print(f"Metric: percent_matched_cells_of_cells_with_reads")

    # Load reference distribution if using ops-median validation
    reference_freq = None
    top_ref_experiments = []
    if use_ops_median:
        if verbose:
            print(f"\n--- Loading reference distribution for ops-median validation ---")
        reference_freq, top_ref_experiments = get_or_build_reference_cache(
            ops_dir=ops_dir,
            exclude_experiment=experiment,
            rebuild=rebuild_cache,
            top_n=10,
            verbose=verbose,
        )
        if len(reference_freq) == 0:
            if verbose:
                print(f"  Warning: Could not load reference, falling back to entropy-based validation")
            use_ops_median = False
        elif verbose:
            print(f"  Reference loaded: {len(reference_freq)} barcodes from {len(top_ref_experiments)} experiments")

    # Analyze base distribution to identify candidate problematic rounds
    base_dist = analyze_base_distribution(reads_df)
    problematic_rounds = [r for r, info in base_dist.items() if info["is_problematic"]]

    if verbose and problematic_rounds:
        print(f"\nProblematic rounds (>40% single base):")
        for r in problematic_rounds:
            info = base_dist[r]
            print(f"  Round {r}: {info['dominant_pct']:.1f}% {info['dominant_base']}")

    results = []

    # Phase 1: Test baseline
    if verbose:
        print(f"\n--- Phase 1: Testing baseline and individual rounds ---")

    baseline = test_configuration(
        reads_df, codebook_db, well, [], [], iss_rounds, total_guides,
        use_ops_median=use_ops_median, reference_freq=reference_freq, baseline_correlation=0.0,
        baseline_top_guide_vs_median=0.0, baseline_weighted_entropy=0.0,
        baseline_entropy_slope=0.0, baseline_entropy_pvalue=1.0,
    )
    results.append(baseline)
    baseline_rate = baseline.match_rate
    baseline_correlation = baseline.correlation
    baseline_top_guide_vs_median = baseline.top_guide_vs_median
    baseline_weighted_entropy = baseline.weighted_entropy
    baseline_entropy_slope = baseline.entropy_slope
    baseline_entropy_pvalue = baseline.entropy_pvalue
    if verbose:
        baseline_info = f"  Baseline: {baseline_rate:.2f}% ({baseline.unique_genes} genes)"
        if use_ops_median and baseline_correlation > 0:
            baseline_info += f" [corr: {baseline_correlation:.3f}]"
        print(baseline_info)

    # Phase 1: Test all single dropouts (parallel)
    promising_dropouts = []
    single_dropout_results = []
    if verbose:
        print(f"\n  Testing single dropouts (parallel with {n_jobs} workers):")

    single_dropout_configs = [[r] for r in iss_rounds]
    single_results = Parallel(n_jobs=n_jobs)(
        delayed(test_configuration)(
            reads_df, codebook_db, well, d, [], iss_rounds, total_guides,
            use_ops_median, reference_freq, baseline_correlation, baseline_top_guide_vs_median,
            baseline_weighted_entropy, baseline_entropy_slope, baseline_entropy_pvalue
        )
        for d in single_dropout_configs
    )
    single_threshold = _get_dynamic_threshold(1)  # 1 round = 1%
    for r, result in zip(iss_rounds, single_results):
        results.append(result)
        single_dropout_results.append((r, result))
        improvement = result.match_rate - baseline_rate
        if result.is_valid and improvement >= single_threshold:
            promising_dropouts.append(r)

    # Show valid single dropout results sorted by match rate
    if verbose:
        single_dropout_results.sort(key=lambda x: x[1].match_rate, reverse=True)
        for r, result in single_dropout_results:
            if result.is_valid:
                improvement = result.match_rate - baseline_rate
                marker = " ✓ PROMISING" if r in promising_dropouts else ""
                print(f"    dropout [{r}]: {result.match_rate:.2f}% ({improvement:+.2f}%) {result.unique_genes} genes{marker}")
        print(f"\n  Promising single dropouts (>={single_threshold}% improvement): {promising_dropouts}")

    # Phase 2: Test ALL double dropout combinations
    if verbose:
        print(f"\n--- Phase 2: Testing double dropout combinations ---")

    promising_double_dropouts = []
    double_dropout_configs = []
    double_dropout_results = []

    total_rounds = len(iss_rounds)
    if max_dropouts >= 2 and total_rounds - 2 >= min_effective_rounds:
        # Test ALL double dropout combinations
        double_dropout_configs = [list(combo) for combo in combinations(iss_rounds, 2)]

    if verbose:
        print(f"  Testing {len(double_dropout_configs)} double dropout combinations...")

    double_results = Parallel(n_jobs=n_jobs)(
        delayed(test_configuration)(
            reads_df, codebook_db, well, d, [], iss_rounds, total_guides,
            use_ops_median, reference_freq, baseline_correlation, baseline_top_guide_vs_median,
            baseline_weighted_entropy, baseline_entropy_slope, baseline_entropy_pvalue
        )
        for d in tqdm(double_dropout_configs, desc="Testing double dropouts", disable=not verbose)
    )
    double_threshold = _get_dynamic_threshold(2)  # 2 rounds = 3%
    for dropouts, result in zip(double_dropout_configs, double_results):
        results.append(result)
        double_dropout_results.append((dropouts, result))
        improvement = result.match_rate - baseline_rate
        if result.is_valid and improvement >= double_threshold:
            promising_double_dropouts.append(dropouts)

    # Show top 10 valid double dropout results sorted by match rate
    if verbose:
        double_dropout_results.sort(key=lambda x: x[1].match_rate, reverse=True)
        valid_double = [(d, r) for d, r in double_dropout_results if r.is_valid][:10]
        print(f"\n  Top 10 valid double dropout combinations:")
        for dropouts, result in valid_double:
            improvement = result.match_rate - baseline_rate
            marker = " ✓ PROMISING" if dropouts in promising_double_dropouts else ""
            print(f"    dropout {dropouts}: {result.match_rate:.2f}% ({improvement:+.2f}%) {result.unique_genes} genes{marker}")
        print(f"\n  Promising double dropouts (>={double_threshold}% improvement): {promising_double_dropouts}")

    # Phase 3: Test ALL triple dropout combinations
    if verbose:
        print(f"\n--- Phase 3: Testing triple dropout combinations ---")

    promising_triple_dropouts = []
    triple_dropout_configs = []
    triple_dropout_results = []

    if max_dropouts >= 3 and total_rounds - 3 >= min_effective_rounds:
        # Test ALL triple dropout combinations
        triple_dropout_configs = [list(combo) for combo in combinations(iss_rounds, 3)]

    if verbose:
        print(f"  Testing {len(triple_dropout_configs)} triple dropout combinations...")

    if triple_dropout_configs:
        triple_results = Parallel(n_jobs=n_jobs)(
            delayed(test_configuration)(
                reads_df, codebook_db, well, d, [], iss_rounds, total_guides,
                use_ops_median, reference_freq, baseline_correlation, baseline_top_guide_vs_median,
                baseline_weighted_entropy
            )
            for d in tqdm(triple_dropout_configs, desc="Testing triple dropouts", disable=not verbose)
        )
        triple_threshold = _get_dynamic_threshold(3)  # 3 rounds = 5%
        for dropouts, result in zip(triple_dropout_configs, triple_results):
            results.append(result)
            triple_dropout_results.append((dropouts, result))
            improvement = result.match_rate - baseline_rate
            if result.is_valid and improvement >= triple_threshold:
                promising_triple_dropouts.append(dropouts)

    # Show top 10 valid triple dropout results sorted by match rate
    if verbose and triple_dropout_results:
        triple_dropout_results.sort(key=lambda x: x[1].match_rate, reverse=True)
        valid_triple = [(d, r) for d, r in triple_dropout_results if r.is_valid][:10]
        triple_threshold = _get_dynamic_threshold(3)
        if valid_triple:
            print(f"\n  Top 10 valid triple dropout combinations:")
            for dropouts, result in valid_triple:
                improvement = result.match_rate - baseline_rate
                marker = " ✓ PROMISING" if dropouts in promising_triple_dropouts else ""
                print(f"    dropout {dropouts}: {result.match_rate:.2f}% ({improvement:+.2f}%) {result.unique_genes} genes{marker}")
            print(f"\n  Promising triple dropouts (>={triple_threshold}% improvement): {promising_triple_dropouts}")
        else:
            print(f"\n  No valid triple dropout combinations found (all failed QC)")

    # === SHIFT TESTING ===
    # Shifts can only be combined with dropouts while maintaining min effective rounds.
    # total_rounds accounts for constant rounds already excluded from iss_rounds.

    # Phase 4: Test shifts independently (no dropouts)
    single_shift_results = []
    promising_single_shifts = []

    if max_shifts >= 1:
        if verbose:
            print(f"\n--- Phase 4: Testing shifts independently (no dropouts) ---")
            print(f"  Testing single shifts (parallel):")

        single_shift_configs = [[r] for r in iss_rounds if total_rounds - 1 >= min_effective_rounds]
    else:
        single_shift_configs = []
        if verbose:
            print(f"\n--- Phase 4: Skipped (--no-shifts) ---")
    if single_shift_configs:
        shift_results = Parallel(n_jobs=n_jobs)(
            delayed(test_configuration)(
                reads_df, codebook_db, well, [], s, iss_rounds, total_guides,
                use_ops_median, reference_freq, baseline_correlation, baseline_top_guide_vs_median,
                baseline_weighted_entropy
            )
            for s in single_shift_configs
        )
        single_shift_threshold = _get_dynamic_threshold(1)  # 1 round = 1%
        for shift_cfg, result in zip(single_shift_configs, shift_results):
            r = shift_cfg[0]
            results.append(result)
            single_shift_results.append((r, result))
            improvement = result.match_rate - baseline_rate
            if result.is_valid and improvement >= single_shift_threshold:
                promising_single_shifts.append(r)

    if verbose and single_shift_results:
        single_shift_results.sort(key=lambda x: x[1].match_rate, reverse=True)
        single_shift_threshold = _get_dynamic_threshold(1)
        for r, result in single_shift_results:
            if result.is_valid:
                improvement = result.match_rate - baseline_rate
                marker = " ✓ PROMISING" if r in promising_single_shifts else ""
                print(f"    shift [{r}]: {result.match_rate:.2f}% ({improvement:+.2f}%) {result.unique_genes} genes{marker}")
        print(f"\n  Promising single shifts (>={single_shift_threshold}% improvement): {promising_single_shifts}")

    # Test ALL double shift combinations
    double_shift_results = []
    promising_double_shifts = []

    if max_shifts >= 2 and total_rounds - 2 >= min_effective_rounds:
        if verbose:
            print(f"\n  Testing double shifts:")

        # Test ALL double shift combinations
        double_shift_configs = [list(combo) for combo in combinations(iss_rounds, 2)]
        if double_shift_configs:
            dbl_shift_results = Parallel(n_jobs=n_jobs)(
                delayed(test_configuration)(
                    reads_df, codebook_db, well, [], s, iss_rounds, total_guides,
                    use_ops_median, reference_freq, baseline_correlation, baseline_top_guide_vs_median,
                    baseline_weighted_entropy
                )
                for s in double_shift_configs
            )
            double_shift_threshold = _get_dynamic_threshold(2)  # 2 rounds = 3%
            for shift_cfg, result in zip(double_shift_configs, dbl_shift_results):
                results.append(result)
                double_shift_results.append((shift_cfg, result))
                improvement = result.match_rate - baseline_rate
                if result.is_valid and improvement >= double_shift_threshold:
                    promising_double_shifts.append(shift_cfg)

        if verbose:
            double_shift_results.sort(key=lambda x: x[1].match_rate, reverse=True)
            valid_dbl_shifts = [(s, r) for s, r in double_shift_results if r.is_valid][:10]
            double_shift_threshold = _get_dynamic_threshold(2)
            print(f"\n  Top 10 valid double shift combinations:")
            for shifts, result in valid_dbl_shifts:
                improvement = result.match_rate - baseline_rate
                marker = " ✓ PROMISING" if shifts in promising_double_shifts else ""
                print(f"    shift {shifts}: {result.match_rate:.2f}% ({improvement:+.2f}%) {result.unique_genes} genes{marker}")
            print(f"\n  Promising double shifts (>={double_shift_threshold}% improvement): {promising_double_shifts}")

    # Phase 5: Test TOP shift + TOP dropout combinations only (not exhaustive)
    # Get top performers from each category to combine
    single_shift_results.sort(key=lambda x: x[1].match_rate, reverse=True)
    single_dropout_results.sort(key=lambda x: x[1].match_rate, reverse=True)
    double_dropout_results.sort(key=lambda x: x[1].match_rate, reverse=True)

    # Get top 3 VALID configs from each category for targeted combination testing
    # IMPORTANT: Only select from valid configurations to avoid propagating QC failures
    top_single_shifts = [r for r, res in single_shift_results if res.is_valid][:3]
    top_single_dropouts = [r for r, res in single_dropout_results if res.is_valid][:3]
    top_double_dropouts = [d for d, res in double_dropout_results if res.is_valid][:3]
    top_double_shifts = [s for s, res in double_shift_results if res.is_valid][:3] if double_shift_results else []

    shift_dropout_combo_results = []
    combo_configs = []

    if max_shifts >= 1:
        if verbose:
            print(f"\n--- Phase 5: Testing TOP shift + dropout combinations (targeted) ---")
            print(f"  Top single shifts: {top_single_shifts}")
            print(f"  Top single dropouts: {top_single_dropouts}")
            print(f"  Top double dropouts: {top_double_dropouts}")
            if top_double_shifts:
                print(f"  Top double shifts: {top_double_shifts}")

        # Top single shifts + top single dropouts
        for shift_r in top_single_shifts:
            for dropout_r in top_single_dropouts:
                if shift_r == dropout_r:
                    continue
                if total_rounds - 1 - 1 >= min_effective_rounds:
                    combo_configs.append(([dropout_r], [shift_r]))

        # Top single shifts + top double dropouts
        for shift_r in top_single_shifts:
            for dropout_combo in top_double_dropouts:
                if shift_r in dropout_combo:
                    continue
                if total_rounds - 2 - 1 >= min_effective_rounds:
                    combo_configs.append((dropout_combo, [shift_r]))

        # Top double shifts + top single dropouts
        if max_shifts >= 2 and top_double_shifts:
            for shift_combo in top_double_shifts:
                for dropout_r in top_single_dropouts:
                    if dropout_r in shift_combo:
                        continue
                    if total_rounds - 1 - 2 >= min_effective_rounds:
                        combo_configs.append(([dropout_r], shift_combo))

        # Run all shift+dropout combos in parallel
        if combo_configs:
            combo_results = Parallel(n_jobs=n_jobs)(
                delayed(test_configuration)(
                    reads_df, codebook_db, well, d, s, iss_rounds, total_guides,
                    use_ops_median, reference_freq, baseline_correlation, baseline_top_guide_vs_median,
                    baseline_weighted_entropy, baseline_entropy_slope, baseline_entropy_pvalue
                )
                for d, s in combo_configs
            )
            for (dropouts, shifts), result in zip(combo_configs, combo_results):
                results.append(result)
                shift_dropout_combo_results.append((dropouts, shifts, result))

        if verbose:
            shift_dropout_combo_results.sort(key=lambda x: x[2].match_rate, reverse=True)
            valid_combos = [(d, s, r) for d, s, r in shift_dropout_combo_results if r.is_valid][:10]
            print(f"\n  Top 10 valid shift+dropout combinations:")
            for dropouts, shifts, result in valid_combos:
                improvement = result.match_rate - baseline_rate
                n_rounds = len(dropouts) + len(shifts)
                combo_threshold = _get_dynamic_threshold(n_rounds)
                marker = " ✓ PROMISING" if improvement >= combo_threshold else ""
                print(f"    dropout {dropouts} + shift {shifts}: {result.match_rate:.2f}% ({improvement:+.2f}%) {result.unique_genes} genes{marker}")
    elif verbose:
        print(f"\n--- Phase 5: Skipped (--no-shifts) ---")

    if verbose:
        print(f"\n  Total configurations tested: {len(results)}")

    # Convert to DataFrame
    results_df = pd.DataFrame([
        {
            "config_type": r.config_type,
            "dropouts": r.dropouts,
            "shifts": r.shifts,
            "config_str": format_config(r.dropouts, r.shifts),
            "cells_with_matched_reads": r.cells_with_matched_reads,
            "cells_with_reads": r.cells_with_reads,
            "match_rate": r.match_rate,  # percent_matched_cells_of_cells_with_reads
            "unique_barcodes": r.unique_barcodes,
            "unique_genes": r.unique_genes,
            "gene_coverage": r.gene_coverage,
            "effective_rounds": r.effective_rounds,
            "top5_concentration": r.top5_concentration,
            "avg_entropy": r.avg_entropy,
            "top_barcode": r.top_barcode,
            "top_barcode_entropy": r.top_barcode_entropy,
            "top_barcode_count": r.top_barcode_count,
            "is_valid": r.is_valid,
            "validation_notes": r.validation_notes,
            "correlation": r.correlation,
            "entropy_slope": r.entropy_slope,
            "entropy_pvalue": r.entropy_pvalue,
            "top_gene_vs_median": r.top_gene_vs_median,
            "top_guide_vs_median": r.top_guide_vs_median,
        }
        for r in results
    ])

    # CRITICAL: Invalidate configs with match rate far below baseline
    # A config that's worse than baseline is never a valid "optimization"
    min_acceptable_rate = baseline_rate * 0.5  # Must be at least 50% of baseline
    low_rate_mask = (results_df["match_rate"] < min_acceptable_rate) & (results_df["config_type"] != "baseline")
    if low_rate_mask.any():
        results_df.loc[low_rate_mask, "is_valid"] = False
        results_df.loc[low_rate_mask, "validation_notes"] = results_df.loc[low_rate_mask, "validation_notes"].apply(
            lambda x: f"Match rate below 50% of baseline; {x}" if x else "Match rate below 50% of baseline"
        )

    # Report OPS-median validation results (validation already done during testing)
    if use_ops_median and baseline_correlation > 0 and verbose:
        valid_count = results_df[results_df["is_valid"] & (results_df["config_type"] != "baseline")].shape[0]
        print(f"\n  OPS-median validation: {valid_count} configs pass (correlation >= {baseline_correlation:.3f})")

    # Add complexity score (number of rounds affected - prefer simpler interventions)
    results_df["complexity"] = results_df.apply(
        lambda r: len(r["dropouts"]) + len(r["shifts"]), axis=1
    )

    # Add preference for dropouts over shifts (0 = dropout only, 1 = has shifts)
    results_df["has_shifts"] = results_df.apply(
        lambda r: 1 if len(r["shifts"]) > 0 else 0, axis=1
    )

    # Sort by: valid first, then by match rate descending
    results_df = results_df.sort_values(
        ["is_valid", "match_rate"],
        ascending=[False, False]
    ).reset_index(drop=True)

    # CONFIG SELECTION LOGIC:
    # - Always maximize match_rate among valid configs
    # - Correlation is used as a THRESHOLD (>= baseline), not an optimization target
    # - This avoids overfitting to the OPS-median reference distribution
    valid_df = results_df[results_df["is_valid"]]

    # Sort by: valid first, then match_rate (highest first)
    # Correlation >= baseline is enforced by is_valid flag, not by sorting
    results_df = results_df.sort_values(
        ["is_valid", "match_rate"],
        ascending=[False, False]
    ).reset_index(drop=True)

    if verbose:
        print(f"\n{'='*70}")
        print("TOP 10 VALID CONFIGURATIONS (sorted by match rate):")
        print(f"{'='*70}")

        # Filter valid configs by dynamic threshold
        # A config is truly valid only if it meets the improvement threshold for its complexity
        # Thresholds are relative: 2 rounds must beat best 1-round by 5%, 3 rounds must beat best 2-round by 10%
        valid_df = results_df[results_df["is_valid"]].copy()
        if len(valid_df) > 0:
            # Add column for dynamic threshold check using relative thresholds
            valid_df["meets_threshold"] = valid_df.apply(
                lambda row: _meets_dynamic_threshold(row, baseline_rate, results_df), axis=1
            )
            valid_df = valid_df[valid_df["meets_threshold"]]

        # Show selection rationale
        if len(valid_df) > 1 and use_ops_median:
            best_config = valid_df.iloc[0]
            if best_config["complexity"] > 0:
                print(f"  ✓ Selecting by highest match_rate (baseline corr: {baseline_correlation:.3f})")
                print()

        for i, row in valid_df.head(10).iterrows():
            complexity_str = f"[{row['complexity']} rounds]"
            if use_ops_median:
                marker = " ← BEST (highest match_rate)" if i == 0 else ""
            else:
                marker = " ← BEST" if i == 0 else ""
            corr_str = f" corr={row['correlation']:.3f}" if use_ops_median and row['correlation'] > 0 else ""
            threshold = _get_dynamic_threshold(row['complexity']) if row['complexity'] > 0 else 0
            improvement = row['match_rate'] - baseline_rate
            print(f"  {row['config_str']:<30} {row['match_rate']:>6.2f}% "
                  f"({row['unique_genes']:>4} genes, {row['effective_rounds']} eff rounds) {complexity_str}{corr_str}{marker}")

        if len(valid_df) == 0:
            print("  No configurations meet the dynamic improvement threshold!")
            print(f"  (1 round: >1% vs baseline, 2 rounds: >5% vs best 1-round, 3 rounds: >10% vs best 2-round)")
            print(f"  Baseline ({baseline_rate:.2f}%) is the best option.")

        # Always show top invalid configs for comparison
        invalid_df = results_df[~results_df["is_valid"]]
        if len(invalid_df) > 0:
            # Sort invalid by match rate to find highest-matching configs that failed QC
            invalid_sorted = invalid_df.sort_values("match_rate", ascending=False)

            print(f"\n{'='*70}")
            print("TOP 1 HIGHEST MATCH-RATE CONFIG THAT FAILED QC:")
            print(f"{'='*70}")
            for i, row in invalid_sorted.head(1).iterrows():
                print(f"\n  Config: {row['config_str']}")
                print(f"    Match rate:      {row['match_rate']:>6.2f}% ({row['cells_with_matched_reads']:,} / {row['cells_with_reads']:,} cells)")
                print(f"    Unique barcodes: {row['unique_barcodes']:,}")
                print(f"    Unique genes:    {row['unique_genes']:,} ({row['gene_coverage']:.1f}% of codebook)")
                print(f"    Top 5 conc:      {row['top5_concentration']:.1f}%")
                print(f"    Eff rounds:      {row['effective_rounds']}")
                print(f"    ❌ FAILED: {row['validation_notes']}")

            # Show comparison if we have both valid and invalid results
            if len(valid_df) > 0:
                best_valid = valid_df.iloc[0]
                best_invalid = invalid_sorted.iloc[0]
                gap = best_invalid['match_rate'] - best_valid['match_rate']
                if gap > 0:
                    print(f"\n  ⚠️  Best invalid config has {gap:.2f}% higher match rate than best valid!")
                    print(f"      Consider reviewing QC thresholds if this gap is significant.")

    # Run full entropy analysis on the best valid configuration that meets threshold
    if verbose:
        valid_df = results_df[results_df["is_valid"]].copy()
        # Apply dynamic threshold filtering - only select from configs that meet the threshold
        if len(valid_df) > 0:
            valid_df = valid_df[valid_df.apply(lambda row: _meets_dynamic_threshold(row, baseline_rate, results_df), axis=1)]

        if len(valid_df) > 0:
            best = valid_df.iloc[0]
            best_dropouts = best["dropouts"]
            best_shifts = best["shifts"]
            config_str = format_config(best_dropouts, best_shifts)

            # Build failed_rounds config for the best configuration
            if best_dropouts or best_shifts:
                failed_config = {well: {}}
                if best_dropouts:
                    failed_config[well]["dropout"] = best_dropouts
                if best_shifts:
                    failed_config[well]["shift"] = best_shifts
            else:
                failed_config = None

            # Get position mapping for best config
            if best_shifts:
                read_positions, codebook_positions = _get_shift_round_mapping(
                    iss_rounds, well, failed_config
                )
            else:
                effective_rounds = [r for r in iss_rounds if r not in best_dropouts]
                read_positions = effective_rounds
                codebook_positions = effective_rounds

            # Re-run matching on full data for the best config
            if failed_config:
                best_matched = match_reads(
                    reads_df.copy(),
                    codebook_db,
                    iss_rounds=iss_rounds,
                    well_name=well,
                    failed_rounds_by_well=failed_config,
                )
            else:
                best_matched = match_reads(
                    reads_df.copy(),
                    codebook_db,
                    iss_rounds=iss_rounds,
                    well_name=well,
                )

            # Compute baseline entropy stats for comparison
            baseline_stats = None
            baseline_matched = None
            baseline_positions = iss_rounds  # All non-constant rounds for baseline

            if best_dropouts or best_shifts:  # Only compute if best config != baseline
                baseline_matched = match_reads(
                    reads_df.copy(),
                    codebook_db,
                    iss_rounds=iss_rounds,
                    well_name=well,
                )
                baseline_stats = compute_entropy_stats(
                    baseline_matched,
                    codebook_db,
                    baseline_positions,
                    baseline_positions,
                    gene_index_db=gene_index_db,
                )

                # Print BASELINE entropy analysis first
                full_entropy_analysis(
                    baseline_matched,
                    codebook_db,
                    baseline_positions,
                    baseline_positions,
                    "baseline",
                    gene_index_db=gene_index_db,
                    baseline_stats=None,  # No comparison for baseline itself
                    verbose=entropy_verbose,
                )

            # Run full entropy analysis for best config (with baseline comparison if available)
            full_entropy_analysis(
                best_matched,
                codebook_db,
                read_positions,
                codebook_positions,
                config_str,
                gene_index_db=gene_index_db,
                baseline_stats=baseline_stats,
                verbose=entropy_verbose,
            )

            # Print recovery summary vs baseline
            best_match_rate = best["match_rate"]
            improvement = best_match_rate - baseline_rate
            recovery_factor = best_match_rate / baseline_rate if baseline_rate > 0 else 0

            print(f"\n{'='*70}")
            print("RECOVERY SUMMARY")
            print(f"{'='*70}")
            print(f"  Baseline match rate:     {baseline_rate:.2f}%")
            print(f"  Best valid match rate:   {best_match_rate:.2f}%")
            print(f"  Absolute improvement:    {improvement:+.2f}%")
            print(f"  Relative improvement:    {recovery_factor:.2f}x baseline ({(recovery_factor-1)*100:+.1f}%)")
            print(f"  Configuration:           {config_str}")
            if constant_rounds:
                full_dropouts = sorted(set(best_dropouts + constant_rounds))
                print(f"  Full dropout (incl. constant rounds {constant_rounds}): {full_dropouts}")
            # Print correlation if computed (ops-median mode or if correlation > 0)
            if baseline_correlation > 0 or best["correlation"] > 0:
                corr_improvement = best["correlation"] - baseline_correlation
                print(f"  Baseline correlation:    {baseline_correlation:.3f}")
                print(f"  Best correlation:        {best['correlation']:.3f} ({corr_improvement:+.3f})")
            print(f"{'='*70}")
        else:
            # No configs meet the dynamic threshold - recommend baseline
            print(f"\n{'='*70}")
            print("RECOVERY SUMMARY")
            print(f"{'='*70}")
            print(f"  Baseline match rate:     {baseline_rate:.2f}%")
            print(f"  Best valid match rate:   {baseline_rate:.2f}% (baseline)")
            if constant_rounds:
                print(f"  Configuration:           dropout {constant_rounds} (constant rounds only)")
            else:
                print(f"  Configuration:           baseline")
            print(f"  ℹ️  No configurations exceeded the dynamic improvement threshold.")
            print(f"      (1 round: >1% vs baseline, 2 rounds: >5% vs best 1-round, 3 rounds: >10% vs best 2-round)")
            # Print correlation if computed
            if baseline_correlation > 0:
                print(f"  Baseline correlation:    {baseline_correlation:.3f}")
            print(f"{'='*70}")

    return results_df


def format_config(dropouts: list[int], shifts: list[int]) -> str:
    """Format configuration as readable string."""
    parts = []
    if dropouts:
        parts.append(f"dropout {dropouts}")
    if shifts:
        parts.append(f"shift {shifts}")
    return " + ".join(parts) if parts else "baseline"


def optimize_experiment(
    experiment: str,
    max_dropouts: int = 3,
    max_shifts: int = 2,
    min_effective_rounds: int = 7,
    improvement_threshold: float = 1.0,
    output_dir: Optional[str] = None,
    verbose: bool = True,
    use_ops_median: bool = False,
    rebuild_cache: bool = False,
    ops_dir: str = DEFAULT_OPS_DIR,
) -> dict[str, pd.DataFrame]:
    """
    Optimize failed round configurations for all wells in an experiment.

    Returns:
        Dictionary mapping well -> results DataFrame
    """
    dataset = OpsDataset(experiment, method="mine")

    # Get all wells by finding *_reads.csv files in the reads directory
    reads_dir = dataset.result_paths["reads"].parent
    wells = []
    if reads_dir.exists():
        for reads_file in reads_dir.glob("*_reads.csv"):
            # Extract well ID from filename (e.g., "A1_reads.csv" -> "A1")
            well_id = reads_file.stem.replace("_reads", "")
            wells.append(well_id)

    if verbose:
        print(f"Found {len(wells)} wells in {experiment}")

    all_results = {}
    best_configs = []

    for well in tqdm(sorted(wells), desc="Processing wells", disable=not verbose):
        try:
            results_df = optimize_well(
                experiment, well,
                max_dropouts=max_dropouts,
                max_shifts=max_shifts,
                min_effective_rounds=min_effective_rounds,
                improvement_threshold=improvement_threshold,
                verbose=False,  # Suppress inner verbose when processing multiple wells
                use_ops_median=use_ops_median,
                rebuild_cache=rebuild_cache,
                ops_dir=ops_dir,
            )
            all_results[well] = results_df

            # Get best valid config that meets dynamic threshold
            valid = results_df[results_df["is_valid"]].copy()
            baseline_row = results_df[results_df["config_type"] == "baseline"]
            baseline_rate = baseline_row.iloc[0]["match_rate"] if len(baseline_row) > 0 else 0

            # Filter by dynamic threshold (relative: 2 rounds vs best 1 round, 3 rounds vs best 2 round)
            if len(valid) > 0:
                valid = valid[valid.apply(lambda row: _meets_dynamic_threshold(row, baseline_rate, results_df), axis=1)]

            if len(valid) > 0:
                best = valid.iloc[0]
                best_configs.append({
                    "well": well,
                    "config": best["config_str"],
                    "dropouts": best["dropouts"],
                    "shifts": best["shifts"],
                    "match_rate": best["match_rate"],
                    "baseline_rate": baseline_rate,
                    "improvement": best["match_rate"] - baseline_rate,
                    "unique_genes": best["unique_genes"],
                    "effective_rounds": best["effective_rounds"],
                })
            else:
                best_configs.append({
                    "well": well,
                    "config": "baseline",
                    "dropouts": [],
                    "shifts": [],
                    "match_rate": baseline_rate,
                    "baseline_rate": baseline_rate,
                    "improvement": 0.0,
                    "unique_genes": baseline_row.iloc[0]["unique_genes"] if len(baseline_row) > 0 else 0,
                    "effective_rounds": 10,
                })
        except Exception as e:
            if verbose:
                tqdm.write(f"  ERROR processing {well}: {e}")

    # Print summary of results per well
    if verbose and best_configs:
        print(f"\n{'='*70}")
        print(f"RESULTS: {experiment} ({len(best_configs)} wells)")
        print(f"{'='*70}")
        for bc in best_configs:
            improvement_str = f"({bc['improvement']:+.2f}%)" if bc['improvement'] != 0 else "(no improvement)"
            print(f"  {bc['well']:<10} {bc['match_rate']:>6.2f}% {improvement_str:<16} "
                  f"{bc['unique_genes']:>4} genes  {bc['config']}")

        # Print YAML snippet for ops_failed_rounds.yaml
        has_any_failed = any(bc["config"] != "baseline" for bc in best_configs)
        print(f"\n{'='*70}")
        print(f"ops_failed_rounds.yaml entry:")
        print(f"{'='*70}")
        print(f"{experiment}:")
        print(f"  failed_rounds_by_well:")
        for bc in best_configs:
            dropouts = bc["dropouts"]
            print(f'    "{bc["well"]}": {dropouts}')
        if not has_any_failed:
            print(f"  # All wells clean - no failed rounds detected")

    # Save summary
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        summary_df = pd.DataFrame(best_configs)
        summary_path = output_path / f"failed_rounds_summary_{experiment}.csv"
        summary_df.to_csv(summary_path, index=False)
        if verbose:
            print(f"\nSummary saved to: {summary_path}")

        # Save detailed results per well
        for well, results_df in all_results.items():
            well_safe = well.replace("/", "_")
            detail_path = output_path / f"failed_rounds_detail_{experiment}_{well_safe}.csv"
            results_df.to_csv(detail_path, index=False)

    return all_results


def optimize_all_experiments(
    ops_dir: str = DEFAULT_OPS_DIR,
    max_dropouts: int = 3,
    max_shifts: int = 2,
    min_effective_rounds: int = 7,
    improvement_threshold: float = 1.0,
    use_ops_median: bool = True,
    rebuild_cache: bool = False,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Process ALL experiments and wells, collecting detailed optimization stats.

    Compares optimized configs against current settings in ops_failed_rounds.yaml.

    Returns a DataFrame with columns:
    - experiment, well
    - baseline_match_rate, best_match_rate, absolute_improvement, relative_improvement
    - current_config_str, current_match_rate, improvement_vs_current
    - config_str, dropouts, shifts
    - baseline_correlation, best_correlation, correlation_improvement
    - slope, baseline_slope, r2, baseline_r2
    - top_guide_ratio, baseline_top_guide_ratio, top_gene_ratio, baseline_top_gene_ratio
    """
    from cyclops_process.metrics.plate_stats.match_reads import match_reads

    experiments = scan_ops_experiments(ops_dir)
    if verbose:
        print(f"Found {len(experiments)} experiments in {ops_dir}")

    # Load current ops_failed_rounds.yaml for comparison
    current_yaml_path = Path(__file__).parent.parent.parent / "configs" / "ops_failed_rounds.yaml"
    current_configs = {}
    if current_yaml_path.exists():
        with open(current_yaml_path) as f:
            current_configs = yaml.safe_load(f) or {}
        if verbose:
            print(f"Loaded current config from: {current_yaml_path}")

    all_stats = []

    for experiment in tqdm(experiments, desc="Processing experiments", disable=not verbose):
        try:
            dataset = OpsDataset(experiment, method="mine")

            # Get all wells by finding *_reads.csv files in the reads directory
            reads_dir = dataset.result_paths["reads"].parent
            wells = []
            if reads_dir.exists():
                for reads_file in reads_dir.glob("*_reads.csv"):
                    # Extract well ID from filename (e.g., "A1_reads.csv" -> "A1")
                    well_id = reads_file.stem.replace("_reads", "")
                    wells.append(well_id)

            codebook_db = dataset.load_codebook()
            total_guides = len(codebook_db)
            barcode_col = "sgRNA" if "sgRNA" in codebook_db.columns else "barcode"
            n_iss_rounds = min(10, int(codebook_db[barcode_col].str.len().max()))
            iss_rounds = [r for r in range(n_iss_rounds)
                          if codebook_db[barcode_col].str[r].nunique() > 1]

            # Load gene_index for gene name lookups
            gene_index_db = None
            try:
                gene_index_db = dataset.load_gene_index()
            except Exception:
                pass

            for well in wells:
                try:
                    reads_df = pd.read_csv(dataset.append_well("reads", well))

                    # Run optimization
                    results_df = optimize_well(
                        experiment, well,
                        max_dropouts=max_dropouts,
                        max_shifts=max_shifts,
                        min_effective_rounds=min_effective_rounds,
                        improvement_threshold=improvement_threshold,
                        verbose=False,
                        use_ops_median=use_ops_median,
                        rebuild_cache=rebuild_cache,
                        ops_dir=ops_dir,
                    )

                    # Get baseline and best config
                    baseline_row = results_df[results_df["config_type"] == "baseline"]
                    valid_df = results_df[results_df["is_valid"]].copy()

                    if len(baseline_row) == 0 or len(valid_df) == 0:
                        continue

                    baseline = baseline_row.iloc[0]
                    baseline_rate = baseline["match_rate"]

                    # Filter by dynamic threshold (relative: 2 rounds vs best 1 round, 3 rounds vs best 2 round)
                    valid_df = valid_df[valid_df.apply(lambda row: _meets_dynamic_threshold(row, baseline_rate, results_df), axis=1)]

                    if len(valid_df) == 0:
                        continue

                    best = valid_df.iloc[0]

                    # Calculate stats
                    best_rate = best["match_rate"]
                    absolute_improvement = best_rate - baseline_rate
                    relative_improvement = best_rate / baseline_rate if baseline_rate > 0 else 0

                    # Get current config from ops_failed_rounds.yaml and test it
                    current_config_str = "baseline"
                    current_match_rate = baseline_rate
                    current_dropouts = []
                    current_shifts = []

                    if experiment in current_configs:
                        exp_config = current_configs[experiment]
                        if "failed_rounds_by_well" in exp_config and well in exp_config["failed_rounds_by_well"]:
                            well_config = exp_config["failed_rounds_by_well"][well]

                            # Parse the config format
                            if isinstance(well_config, list):
                                # Simple dropout list: [3, 9]
                                current_dropouts = well_config
                            elif isinstance(well_config, dict):
                                # Dict format: {"dropout": [...], "shift": [...]}
                                current_dropouts = well_config.get("dropout", [])
                                current_shifts = well_config.get("shift", [])

                            # Build config string
                            if current_dropouts and current_shifts:
                                current_config_str = f"dropout {current_dropouts} + shift {current_shifts}"
                            elif current_shifts:
                                current_config_str = f"shift {current_shifts}"
                            elif current_dropouts:
                                current_config_str = f"dropout {current_dropouts}"

                            # Test current config to get match rate
                            if current_dropouts or current_shifts:
                                current_failed_config = {well: {}}
                                if current_dropouts:
                                    current_failed_config[well]["dropout"] = current_dropouts
                                if current_shifts:
                                    current_failed_config[well]["shift"] = current_shifts

                                current_matched = match_reads(
                                    reads_df.copy(), codebook_db,
                                    iss_rounds=iss_rounds, well_name=well,
                                    failed_rounds_by_well=current_failed_config,
                                )
                                cells_with_reads = reads_df["cell"].nunique() if "cell" in reads_df.columns else len(reads_df)
                                cells_with_current_matched = current_matched["cell"][current_matched["cell"] > 0].nunique() if len(current_matched) > 0 and "cell" in current_matched.columns else 0
                                current_match_rate = (cells_with_current_matched / cells_with_reads * 100) if cells_with_reads > 0 else 0

                    improvement_vs_current = best_rate - current_match_rate

                    # Get entropy stats for baseline and best config
                    baseline_stats = None
                    best_stats = None

                    # Match reads for baseline
                    baseline_matched = match_reads(
                        reads_df.copy(), codebook_db,
                        iss_rounds=iss_rounds, well_name=well,
                    )
                    baseline_positions = iss_rounds
                    baseline_stats = compute_entropy_stats(
                        baseline_matched, codebook_db,
                        baseline_positions, baseline_positions,
                        gene_index_db=gene_index_db,
                    )

                    # Match reads for best config (if different from baseline)
                    if best["dropouts"] or best["shifts"]:
                        failed_config = {well: {}}
                        if best["dropouts"]:
                            failed_config[well]["dropout"] = best["dropouts"]
                        if best["shifts"]:
                            failed_config[well]["shift"] = best["shifts"]

                        if best["shifts"]:
                            read_positions, codebook_positions = _get_shift_round_mapping(
                                iss_rounds, well, failed_config
                            )
                        else:
                            effective_rounds = [r for r in iss_rounds if r not in best["dropouts"]]
                            read_positions = effective_rounds
                            codebook_positions = effective_rounds

                        best_matched = match_reads(
                            reads_df.copy(), codebook_db,
                            iss_rounds=iss_rounds, well_name=well,
                            failed_rounds_by_well=failed_config,
                        )
                        best_stats = compute_entropy_stats(
                            best_matched, codebook_db,
                            read_positions, codebook_positions,
                            gene_index_db=gene_index_db,
                        )
                    else:
                        best_stats = baseline_stats

                    # Collect all stats
                    stats = {
                        "experiment": experiment,
                        "well": well,
                        "baseline_match_rate": baseline_rate,
                        "current_config_str": current_config_str,
                        "current_match_rate": current_match_rate,
                        "best_match_rate": best_rate,
                        "absolute_improvement": absolute_improvement,
                        "relative_improvement": relative_improvement,
                        "improvement_vs_current": improvement_vs_current,
                        "config_str": best["config_str"],
                        "dropouts": str(best["dropouts"]),
                        "shifts": str(best["shifts"]),
                        "current_dropouts": str(current_dropouts),
                        "current_shifts": str(current_shifts),
                        "baseline_correlation": baseline["correlation"],
                        "best_correlation": best["correlation"],
                        "correlation_improvement": best["correlation"] - baseline["correlation"],
                        "unique_genes": best["unique_genes"],
                        "effective_rounds": best["effective_rounds"],
                    }

                    # Add entropy stats if available
                    if baseline_stats:
                        stats["baseline_slope"] = baseline_stats.slope
                        stats["baseline_r2"] = baseline_stats.r2
                        stats["baseline_top_guide_ratio"] = baseline_stats.top_guide_ratio
                        stats["baseline_top_gene_ratio"] = baseline_stats.top_gene_ratio
                    if best_stats:
                        stats["best_slope"] = best_stats.slope
                        stats["best_r2"] = best_stats.r2
                        stats["best_top_guide_ratio"] = best_stats.top_guide_ratio
                        stats["best_top_gene_ratio"] = best_stats.top_gene_ratio

                    all_stats.append(stats)

                except Exception as e:
                    if verbose:
                        tqdm.write(f"  ERROR {experiment}/{well}: {e}")

        except Exception as e:
            if verbose:
                tqdm.write(f"ERROR processing experiment {experiment}: {e}")

    # Create DataFrame
    summary_df = pd.DataFrame(all_stats)

    # Save if output path provided
    if output_path:
        summary_df.to_csv(output_path, index=False)
        if verbose:
            print(f"\nSummary saved to: {output_path}")

        # Also save YAML file with optimized configurations
        yaml_path = Path(output_path).with_suffix(".yaml")
        _save_optimized_yaml(summary_df, yaml_path, verbose=verbose)

    return summary_df


def _get_dynamic_threshold(n_rounds: int) -> float:
    """
    Get the minimum improvement threshold based on number of rounds removed.

    Thresholds are relative to the best result from the previous complexity level:
    - 1 round: must be >1% better than baseline
    - 2 rounds: must be >5% better than best 1-round config
    - 3+ rounds: must be >10% better than best 2-round config
    """
    if n_rounds <= 1:
        return 1.0   # 1 round: >1% vs baseline
    elif n_rounds == 2:
        return 5.0   # 2 rounds: >5% vs best 1-round
    else:
        return 10.0  # 3+ rounds: >10% vs best 2-round


def _meets_dynamic_threshold(row, baseline_rate: float, all_results: "pd.DataFrame" = None) -> bool:
    """
    Check if a config row meets the dynamic improvement threshold.

    Thresholds are relative to the best result from the previous complexity level:
    - 1 round: must be >1% better than baseline
    - 2 rounds: must be >5% better than best valid 1-round config
    - 3+ rounds: must be >10% better than best valid 2-round config
    """
    if row["complexity"] == 0:  # baseline always meets threshold
        return True

    complexity = row["complexity"]
    threshold = _get_dynamic_threshold(complexity)

    # Determine comparison rate based on complexity
    if complexity == 1:
        # Compare against baseline
        comparison_rate = baseline_rate
    else:
        # Compare against best valid result from previous complexity level
        if all_results is None:
            # Fallback to baseline if no results available
            comparison_rate = baseline_rate
        else:
            prev_complexity = complexity - 1
            prev_level = all_results[
                (all_results["complexity"] == prev_complexity) &
                (all_results["is_valid"])
            ]
            if len(prev_level) > 0:
                comparison_rate = prev_level["match_rate"].max()
            else:
                # No valid configs at previous level, compare to baseline
                comparison_rate = baseline_rate

    improvement = row["match_rate"] - comparison_rate
    return improvement > threshold


def _save_optimized_yaml(
    summary_df: pd.DataFrame,
    yaml_path: Path,
    verbose: bool = True,
) -> None:
    """
    Save optimized configurations as a YAML file in ops_failed_rounds.yaml format.

    Uses dynamic threshold based on number of rounds removed (relative to previous level):
    - 1 round: >1% improvement vs baseline
    - 2 rounds: >5% improvement vs best 1-round
    - 3+ rounds: >10% improvement vs best 2-round

    Format:
        experiment_name:
          failed_rounds_by_well:
            "well": [dropouts]           # dropout-only
            "well": {"dropout": [...], "shift": [...]}  # mixed mode
    """
    import ast

    # Group by experiment
    yaml_data = {}

    for experiment in sorted(summary_df["experiment"].unique()):
        exp_df = summary_df[summary_df["experiment"] == experiment]
        failed_rounds_by_well = {}

        for _, row in exp_df.iterrows():
            well = row["well"]

            # Parse dropouts and shifts from string representation
            dropouts = ast.literal_eval(row["dropouts"]) if row["dropouts"] else []
            shifts = ast.literal_eval(row["shifts"]) if row["shifts"] else []

            # Dynamic threshold based on number of rounds removed
            n_rounds = len(dropouts) + len(shifts)
            min_threshold = _get_dynamic_threshold(n_rounds)

            # Check if improvement exceeds threshold
            improvement = row.get("improvement_vs_current", 0)
            if improvement <= min_threshold:
                # Improvement below threshold - use baseline (empty list)
                failed_rounds_by_well[well] = []
            elif shifts and dropouts:
                # Mixed mode: use dict format
                failed_rounds_by_well[well] = {"dropout": dropouts, "shift": shifts}
            elif shifts:
                # Shift-only: use dict format
                failed_rounds_by_well[well] = {"shift": shifts}
            else:
                # Dropout-only or baseline: use list format
                failed_rounds_by_well[well] = dropouts

        yaml_data[experiment] = {
            "failed_rounds_by_well": failed_rounds_by_well
        }

    # Write YAML with custom formatting for readability
    yaml_lines = [
        "# Optimized failed rounds configuration",
        "# Auto-generated by optimize_failed_rounds.py --all",
        f"# Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "# Dynamic improvement thresholds (relative): 1 round>1% vs baseline, 2 rounds>5% vs best 1-round, 3 rounds>10% vs best 2-round",
        "#",
        "# Format matches ops_failed_rounds.yaml:",
        "#   experiment_name:",
        "#     failed_rounds_by_well:",
        '#       "well": [dropouts]  OR  {"dropout": [...], "shift": [...]}',
        "",
    ]

    for experiment, exp_config in sorted(yaml_data.items()):
        yaml_lines.append(f"{experiment}:")
        yaml_lines.append("  failed_rounds_by_well:")

        failed_rounds = exp_config["failed_rounds_by_well"]
        for well in sorted(failed_rounds.keys()):
            config = failed_rounds[well]
            if isinstance(config, dict):
                # Mixed or shift-only mode
                parts = []
                if "dropout" in config and config["dropout"]:
                    parts.append(f'"dropout": {config["dropout"]}')
                if "shift" in config and config["shift"]:
                    parts.append(f'"shift": {config["shift"]}')
                yaml_lines.append(f'    "{well}": {{{", ".join(parts)}}}')
            else:
                # Dropout-only or baseline (list format)
                yaml_lines.append(f'    "{well}": {config}')

        yaml_lines.append("")  # Blank line between experiments

    # Write to file
    with open(yaml_path, "w") as f:
        f.write("\n".join(yaml_lines))

    if verbose:
        print(f"YAML config saved to: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Optimize failed round configurations for ISS wells",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single well
    python -m cyclops_process.scripts.optimize_failed_rounds ops0108_20251209 --well "A/3/0"

    # All wells
    python -m cyclops_process.scripts.optimize_failed_rounds ops0108_20251209 --all-wells

    # Custom parameters
    python -m cyclops_process.scripts.optimize_failed_rounds ops0108_20251209 --well "A/3/0" \\
        --max-dropouts 2 --max-shifts 1 --min-effective-rounds 8
        """
    )
    parser.add_argument("experiment", nargs="?", help="Experiment name (e.g., ops0108_20251209). Not required with --all.")
    parser.add_argument("--well", help="Single well to optimize (e.g., A/3/0)")
    parser.add_argument("--all-wells", action="store_true", help="Optimize all wells in one experiment")
    parser.add_argument("--all", action="store_true",
                        help="Process ALL experiments and wells, output summary CSV")
    parser.add_argument("--max-dropouts", type=int, default=3,
                        help="Maximum dropout rounds to test (default: 3)")
    parser.add_argument("--max-shifts", type=int, default=2,
                        help="Maximum shift rounds to test (default: 2)")
    parser.add_argument("--min-effective-rounds", type=int, default=7,
                        help="Minimum effective rounds required (default: 7)")
    parser.add_argument("--improvement-threshold", type=float, default=1.0,
                        help="Min %% improvement to consider a round promising (default: 1.0)")
    parser.add_argument("--output-dir", help="Directory to save results")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="Number of parallel workers (default: auto-detect)")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed entropy analysis (barcode distribution, per-round stats, etc.)")

    # OPS Median validation options
    parser.add_argument("--ops-median", action="store_true",
                        help="Use correlation-based validation against reference distribution from top 10 correlated experiments. "
                             "Configs are valid if correlation >= baseline (strict).")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Force rebuild of reference cache")
    parser.add_argument("--regenerate-heatmap", action="store_true",
                        help="Regenerate correlation heatmaps from cached data (no full rebuild)")
    parser.add_argument("--ops-dir", default=DEFAULT_OPS_DIR,
                        help=f"Base OPS directory for scanning experiments (default: {DEFAULT_OPS_DIR})")

    args = parser.parse_args()

    # Handle --regenerate-heatmap mode (loads from cached frequency matrix - fast!)
    if args.regenerate_heatmap:
        print("Regenerating correlation heatmaps from cached frequency matrix...")

        freq_matrix_path = CACHE_DIR / "experiment_frequency_matrix.csv"
        if not freq_matrix_path.exists():
            print(f"  ERROR: Cached frequency matrix not found at {freq_matrix_path}")
            print("  Run with --rebuild-cache first to create it.")
            sys.exit(1)

        # Load cached frequency matrix (fast - just reading CSV)
        ref_tbl = pd.read_csv(freq_matrix_path)
        ops_columns = [col for col in ref_tbl.columns if col not in ['barcode', 'Gene name']]

        if len(ops_columns) < 2:
            print(f"  ERROR: Only {len(ops_columns)} experiments in cached data")
            sys.exit(1)

        print(f"  Loaded {len(ops_columns)} experiments from cache")

        heatmap_path = CACHE_DIR / "experiment_correlation_heatmap.png"
        generate_correlation_heatmap(
            ref_tbl,
            ops_columns,
            heatmap_path,
            verbose=not args.quiet
        )
        print("\nDone! Heatmaps regenerated.")
        sys.exit(0)

    # Handle --all mode (processes all experiments)
    if args.all:
        if args.well or args.all_wells:
            parser.error("Cannot use --all with --well or --all-wells")

        # Default output path (QC folder)
        output_path = args.output_dir or str(CACHE_DIR.parent / "optimized_iss_failed_rounds.csv")

        print(f"Processing ALL experiments with --ops-median validation...")
        summary_df = optimize_all_experiments(
            ops_dir=args.ops_dir,
            max_dropouts=args.max_dropouts,
            max_shifts=args.max_shifts,
            min_effective_rounds=args.min_effective_rounds,
            improvement_threshold=args.improvement_threshold,
            use_ops_median=True,  # Always use ops-median for --all
            rebuild_cache=args.rebuild_cache,
            output_path=output_path,
            verbose=not args.quiet,
        )

        print(f"\n{'='*70}")
        print(f"SUMMARY: Processed {len(summary_df)} wells across all experiments")
        print(f"{'='*70}")
        if len(summary_df) > 0:
            # Improvement vs baseline (no failed rounds)
            improved_vs_baseline = summary_df[summary_df["absolute_improvement"] > 0]
            print(f"\n  VS BASELINE (no failed rounds):")
            print(f"    Wells with improvement: {len(improved_vs_baseline)} / {len(summary_df)} ({len(improved_vs_baseline)/len(summary_df)*100:.1f}%)")
            print(f"    Mean improvement: {summary_df['absolute_improvement'].mean():.2f}%")
            print(f"    Max improvement: {summary_df['absolute_improvement'].max():.2f}%")

            # Improvement vs current ops_failed_rounds.yaml settings
            improved_vs_current = summary_df[summary_df["improvement_vs_current"] > 0.1]  # >0.1% threshold
            unchanged = summary_df[abs(summary_df["improvement_vs_current"]) <= 0.1]
            worse_vs_current = summary_df[summary_df["improvement_vs_current"] < -0.1]

            print(f"\n  VS CURRENT CONFIG (ops_failed_rounds.yaml):")
            print(f"    Wells improved:  {len(improved_vs_current)} / {len(summary_df)} ({len(improved_vs_current)/len(summary_df)*100:.1f}%)")
            print(f"    Wells unchanged: {len(unchanged)} / {len(summary_df)} ({len(unchanged)/len(summary_df)*100:.1f}%)")
            print(f"    Wells worse:     {len(worse_vs_current)} / {len(summary_df)} ({len(worse_vs_current)/len(summary_df)*100:.1f}%)")
            print(f"    Mean improvement vs current: {summary_df['improvement_vs_current'].mean():.2f}%")
            if len(improved_vs_current) > 0:
                print(f"    Max improvement vs current: {summary_df['improvement_vs_current'].max():.2f}%")

            # Global totals
            print(f"\n  GLOBAL TOTALS:")
            total_cells = len(summary_df)  # Approximate - each well represents many cells
            total_current_rate = summary_df["current_match_rate"].mean()
            total_best_rate = summary_df["best_match_rate"].mean()
            total_improvement = total_best_rate - total_current_rate
            print(f"    Avg current match rate: {total_current_rate:.2f}%")
            print(f"    Avg optimized match rate: {total_best_rate:.2f}%")
            print(f"    Avg improvement: {total_improvement:+.2f}%")

            print(f"\n  CORRELATION:")
            print(f"    Mean correlation improvement: {summary_df['correlation_improvement'].mean():.3f}")

        yaml_path = Path(output_path).with_suffix(".yaml")
        print(f"\nResults saved to:")
        print(f"  CSV:  {output_path}")
        print(f"  YAML: {yaml_path}")
        return

    # Validate args for single experiment modes
    if not args.experiment:
        parser.error("experiment is required (unless using --all)")

    if not args.well and not args.all_wells:
        parser.error("Must specify either --well, --all-wells, or --all")

    if args.well and args.all_wells:
        parser.error("Cannot specify both --well and --all-wells")

    # Resolve experiment name (supports shorthand like "108" -> "ops0108_20251209")
    args.experiment = resolve_experiment_name(args.experiment, autoselect=True)

    # Get default output directory from OpsDataset (ISS/mine results folder)
    dataset = OpsDataset(args.experiment, method="mine")
    default_output_dir = dataset.results_iss

    # Ensure the output directory exists
    Path(default_output_dir).mkdir(parents=True, exist_ok=True)

    if args.all_wells:
        optimize_experiment(
            experiment=args.experiment,
            max_dropouts=args.max_dropouts,
            max_shifts=args.max_shifts,
            min_effective_rounds=args.min_effective_rounds,
            improvement_threshold=args.improvement_threshold,
            output_dir=args.output_dir or str(default_output_dir),
            verbose=not args.quiet,
            use_ops_median=args.ops_median,
            rebuild_cache=args.rebuild_cache,
            ops_dir=args.ops_dir,
        )
    else:
        results = optimize_well(
            experiment=args.experiment,
            well=args.well,
            max_dropouts=args.max_dropouts,
            max_shifts=args.max_shifts,
            min_effective_rounds=args.min_effective_rounds,
            improvement_threshold=args.improvement_threshold,
            verbose=not args.quiet,
            n_jobs=args.n_jobs,
            use_ops_median=args.ops_median,
            rebuild_cache=args.rebuild_cache,
            ops_dir=args.ops_dir,
            entropy_verbose=args.verbose,
        )

        # Save results to ISS/mine results folder (or custom output dir if specified)
        output_dir = args.output_dir or str(default_output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        well_safe = args.well.replace("/", "_")
        output_path = f"{output_dir}/optimize_failed_rounds_{args.experiment}_{well_safe}.csv"
        results.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

        # Print YAML config suggestion for best valid config that meets threshold
        valid = results[results["is_valid"]].copy()
        # Get baseline for improvement calculation
        baseline_row = results[results["config_type"] == "baseline"]
        baseline_rate = baseline_row.iloc[0]["match_rate"] if len(baseline_row) > 0 else 0

        # Filter valid configs by dynamic threshold
        if len(valid) > 0:
            valid = valid[valid.apply(lambda row: _meets_dynamic_threshold(row, baseline_rate), axis=1)]

        if len(valid) > 0:
            best = valid.iloc[0]
            improvement = best["match_rate"] - baseline_rate

            # Load current config from ops_failed_rounds.yaml for comparison
            from cyclops_process.metrics.plate_stats.match_reads import match_reads as match_reads_fn
            current_yaml_path = Path(__file__).parent.parent.parent / "configs" / "ops_failed_rounds.yaml"
            current_config_str = "baseline"
            current_match_rate = baseline_rate
            current_dropouts = []
            current_shifts = []

            if current_yaml_path.exists():
                with open(current_yaml_path) as f:
                    current_configs = yaml.safe_load(f) or {}

                if args.experiment in current_configs:
                    exp_config = current_configs[args.experiment]
                    if "failed_rounds_by_well" in exp_config and args.well in exp_config["failed_rounds_by_well"]:
                        well_config = exp_config["failed_rounds_by_well"][args.well]

                        # Parse the config format
                        if isinstance(well_config, list):
                            current_dropouts = well_config
                        elif isinstance(well_config, dict):
                            current_dropouts = well_config.get("dropout", [])
                            current_shifts = well_config.get("shift", [])

                        # Build config string
                        if current_dropouts and current_shifts:
                            current_config_str = f"dropout {current_dropouts} + shift {current_shifts}"
                        elif current_shifts:
                            current_config_str = f"shift {current_shifts}"
                        elif current_dropouts:
                            current_config_str = f"dropout {current_dropouts}"

                        # Test current config to get match rate
                        if current_dropouts or current_shifts:
                            dataset = OpsDataset(args.experiment, method="mine")
                            reads_df = pd.read_csv(dataset.append_well("reads", args.well))
                            codebook_db = dataset.load_codebook()
                            barcode_col = "sgRNA" if "sgRNA" in codebook_db.columns else "barcode"
                            n_iss_rounds = min(10, int(codebook_db[barcode_col].str.len().max()))
                            iss_rounds = [r for r in range(n_iss_rounds)
                                          if codebook_db[barcode_col].str[r].nunique() > 1]

                            current_failed_config = {args.well: {}}
                            if current_dropouts:
                                current_failed_config[args.well]["dropout"] = current_dropouts
                            if current_shifts:
                                current_failed_config[args.well]["shift"] = current_shifts

                            current_matched = match_reads_fn(
                                reads_df.copy(), codebook_db,
                                iss_rounds=iss_rounds, well_name=args.well,
                                failed_rounds_by_well=current_failed_config,
                            )
                            cells_with_reads = reads_df["cell"].nunique() if "cell" in reads_df.columns else len(reads_df)
                            cells_with_current_matched = current_matched["cell"][current_matched["cell"] > 0].nunique() if len(current_matched) > 0 and "cell" in current_matched.columns else 0
                            current_match_rate = (cells_with_current_matched / cells_with_reads * 100) if cells_with_reads > 0 else 0

            improvement_vs_current = best["match_rate"] - current_match_rate

            print(f"\n{'='*70}")
            print("SUGGESTED YAML CONFIG:")
            print(f"{'='*70}")

            # Show current config comparison
            print(f"  Current config: {current_config_str} ({current_match_rate:.2f}%)")

            # Best config already passed the dynamic threshold filter vs baseline
            # So if it has dropouts/shifts, suggest it; otherwise suggest baseline
            if best["dropouts"] or best["shifts"]:
                config_parts = {}
                if best["dropouts"]:
                    config_parts["dropout"] = best["dropouts"]
                if best["shifts"]:
                    config_parts["shift"] = best["shifts"]
                print(f'  "{args.well}": {config_parts}  # {best["match_rate"]:.2f}% ({improvement_vs_current:+.2f}% vs current, {improvement:+.2f}% vs baseline)')
            else:
                # Best config is baseline itself
                print(f'  "{args.well}": []  # baseline ({baseline_rate:.2f}%) - no optimizations exceed threshold')
        else:
            # No valid configs meet the threshold - suggest baseline
            print(f"\n{'='*70}")
            print("SUGGESTED YAML CONFIG:")
            print(f"{'='*70}")
            print(f'  "{args.well}": []  # baseline ({baseline_rate:.2f}%) - no optimizations exceed threshold')


if __name__ == "__main__":
    main()
