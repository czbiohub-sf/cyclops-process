"""
ISS Entropy and Correlation Statistics Helper Functions.

This module provides helper functions for computing entropy statistics and
correlation metrics for ISS reads, used by the main metrics.py statistics function.
"""

import numpy as np
import pandas as pd
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path
from scipy import stats as scipy_stats
from scipy.stats import pearsonr


# Cache directory for reference guide distributions
CACHE_DIR = Path(__file__).parent.parent.parent.parent / "tests" / "QC" / "reference_cache"


def calculate_entropy(seq: str) -> float:
    """Calculate Shannon entropy of a sequence (in bits). Max is 2.0 for 4 bases."""
    if not seq:
        return 0.0
    counts = Counter(seq)
    probs = np.array(list(counts.values())) / len(seq)
    return float(-np.sum(probs * np.log2(probs + 1e-9)))


@dataclass
class EntropyStats:
    """Statistics from entropy analysis."""
    slope: float  # reads per entropy bit (entropy-count regression slope)
    r2: float  # R-squared of entropy-count regression
    p_value: float  # p-value of entropy-count regression
    top_guide_ratio: float  # top guide count / median count
    mean_entropy: float  # mean barcode entropy
    unique_barcodes: int  # number of unique barcodes


@dataclass
class CorrelationStats:
    """Statistics from correlation to reference distribution."""
    correlation: float  # Pearson correlation to reference
    common_barcodes: int  # number of barcodes in common with reference
    pct_common: float  # percentage of reference barcodes found


def compute_entropy_stats_for_well(
    matched_df: pd.DataFrame,
    iss_rounds: list[int],
) -> Optional[EntropyStats]:
    """
    Compute entropy statistics for matched reads in a well.

    This computes:
    - Entropy vs count regression (slope, r2, p_value) to detect low-entropy bias
    - Top guide ratio (top guide count / median count)
    - Mean barcode entropy

    Args:
        matched_df: DataFrame of matched reads with 'barcode' column
        iss_rounds: List of ISS round indices used for matching

    Returns:
        EntropyStats or None if not enough data
    """
    if matched_df.empty or 'barcode' not in matched_df.columns:
        return None

    # Extract barcodes at effective positions
    matched_barcodes = matched_df['barcode'].apply(
        lambda x: ''.join(x[p] for p in iss_rounds if p < len(x))
    )

    # Calculate entropy for each unique barcode
    barcode_counts = matched_barcodes.value_counts()

    if len(barcode_counts) < 3:
        return None

    barcode_entropies = {bc: calculate_entropy(bc) for bc in barcode_counts.index}

    # Entropy vs count regression
    entropies_arr = np.array([barcode_entropies[bc] for bc in barcode_counts.index])
    counts_arr = barcode_counts.values

    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
        entropies_arr, counts_arr
    )

    # Top guide ratio: top guide count / median count
    top_count = barcode_counts.iloc[0]
    median_count = barcode_counts.median()
    top_guide_ratio = top_count / max(median_count, 1)

    return EntropyStats(
        slope=float(np.round(slope, 2)),
        r2=float(np.round(r_value ** 2, 4)),
        p_value=float(p_value),
        top_guide_ratio=float(np.round(top_guide_ratio, 2)),
        mean_entropy=float(np.round(np.mean(list(barcode_entropies.values())), 4)),
        unique_barcodes=len(barcode_counts),
    )


def load_reference_distribution() -> Optional[pd.Series]:
    """
    Load the cached reference guide frequency distribution.

    Returns:
        Series with barcode -> median frequency, or None if cache doesn't exist
    """
    cache_path = CACHE_DIR / "reference_guide_freq_global_baseline.csv"

    if not cache_path.exists():
        return None

    try:
        ref_df = pd.read_csv(cache_path)
        if 'barcode' not in ref_df.columns or 'median_freq' not in ref_df.columns:
            return None
        return pd.Series(ref_df['median_freq'].values, index=ref_df['barcode'])
    except Exception:
        return None


def compute_correlation_to_reference(
    matched_df: pd.DataFrame,
    iss_rounds: list[int],
    reference_freq: Optional[pd.Series] = None,
) -> Optional[CorrelationStats]:
    """
    Compute correlation of matched reads against reference guide distribution.

    Uses Pearson correlation on log2-transformed frequencies. The log2 transform
    handles the long-tailed distribution of guide counts, ensuring all guides
    contribute more equally to the correlation (not dominated by a few high-count guides).

    Note: The reference stores median frequencies from top 10 experiments.
    We normalize the current well's counts to frequencies for comparison.

    Args:
        matched_df: DataFrame of matched reads with 'barcode' column
        iss_rounds: List of ISS round indices used for matching
        reference_freq: Reference frequency distribution. If None, loads from cache.

    Returns:
        CorrelationStats or None if not enough data or no reference available
    """
    if matched_df.empty or 'barcode' not in matched_df.columns:
        return None

    if reference_freq is None:
        reference_freq = load_reference_distribution()

    if reference_freq is None or len(reference_freq) == 0:
        return None

    # Extract barcodes at effective positions
    matched_barcodes = matched_df['barcode'].apply(
        lambda x: ''.join(x[p] for p in iss_rounds if p < len(x))
    )

    # Compute frequency distribution of current barcodes
    barcode_counts = matched_barcodes.value_counts()
    current_freq = barcode_counts / barcode_counts.sum()

    # Truncate reference barcodes to same positions
    ref_truncated = reference_freq.copy()
    ref_truncated.index = ref_truncated.index.astype(str).map(
        lambda x: ''.join(x[p] for p in iss_rounds if p < len(x))
    )
    # Aggregate any collisions from position extraction
    ref_truncated = ref_truncated.groupby(ref_truncated.index).sum()

    # Find common barcodes
    common_barcodes = set(current_freq.index) & set(ref_truncated.index)

    if len(common_barcodes) < 50:
        return CorrelationStats(
            correlation=0.0,
            common_barcodes=len(common_barcodes),
            pct_common=len(common_barcodes) / len(reference_freq) * 100 if len(reference_freq) > 0 else 0.0,
        )

    # Get frequencies for common barcodes
    current_arr = current_freq.reindex(list(common_barcodes)).fillna(0).values
    ref_arr = ref_truncated.reindex(list(common_barcodes)).fillna(0).values

    # Log2 transform to handle long-tailed distribution
    # Small epsilon prevents log(0) = -inf while not affecting non-zero values
    epsilon = 1e-9
    current_log2 = np.log2(current_arr + epsilon)
    ref_log2 = np.log2(ref_arr + epsilon)

    # Pearson correlation on log2-transformed frequencies
    corr, p_value = pearsonr(current_log2, ref_log2)

    return CorrelationStats(
        correlation=float(np.round(corr, 4)),
        common_barcodes=len(common_barcodes),
        pct_common=float(np.round(len(common_barcodes) / len(reference_freq) * 100, 2)),
    )


def compute_iss_quality_stats(
    matched_df: pd.DataFrame,
    iss_rounds: list[int],
    reference_freq: Optional[pd.Series] = None,
) -> dict:
    """
    Compute all ISS quality statistics for a well in one call.

    This is the main entry point for metrics.py to get entropy and correlation stats.

    Args:
        matched_df: DataFrame of matched reads with 'barcode' column
        iss_rounds: List of ISS round indices used for matching
        reference_freq: Optional reference frequency distribution

    Returns:
        Dictionary with all computed statistics, ready to add to plate_stats
    """
    result = {
        'entropy_slope': None,
        'entropy_r2': None,
        'entropy_pvalue': None,
        'top_guide_ratio': None,
        'mean_entropy': None,
        'correlation_to_ref': None,
        'pct_common_with_ref': None,
    }

    # Compute entropy stats
    entropy_stats = compute_entropy_stats_for_well(matched_df, iss_rounds)
    if entropy_stats is not None:
        result['entropy_slope'] = entropy_stats.slope
        result['entropy_r2'] = entropy_stats.r2
        result['entropy_pvalue'] = entropy_stats.p_value
        result['top_guide_ratio'] = entropy_stats.top_guide_ratio
        result['mean_entropy'] = entropy_stats.mean_entropy

    # Compute correlation stats
    corr_stats = compute_correlation_to_reference(matched_df, iss_rounds, reference_freq)
    if corr_stats is not None:
        result['correlation_to_ref'] = corr_stats.correlation
        result['pct_common_with_ref'] = corr_stats.pct_common

    return result
