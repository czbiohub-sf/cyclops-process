"""Cell-level sister effect analysis: does sister co-localization amplify phenotypes?

Loads cell-level dino features and per-cell spatial coherence, pools cells across
experiments per sgRNA, bins by a spatial metric, and computes Cohen's d and cosine
distance for each (guide × bin) vs NTC. Tests whether guides with higher sister
co-localization show larger perturbation effect sizes.

Outputs are saved to two parallel subdirectory trees for comparison:
  <output_dir>/
    native/                  # All cells per (guide, bin) — no downsampling
      sister_ratio/
      neighbor_count/
      sister_count/
    downsampled/             # Subsampled to k=min(cells) across bins per guide
      sister_ratio/
      neighbor_count/
      sister_count/

Three binning metrics:
  • sister_ratio   — same-guide fraction of neighbors (density-normalised)
  • neighbor_count — total neighbors regardless of guide (pure density proxy)
  • sister_count   — raw same-guide neighbor count

Default mode pools cells across all experiments before binning and filtering,
maximizing the number of guides that survive the min-cells threshold. Cells
are z-scored within their own experiment (for Cohen's d) and L2-normalized
(for cosine distance) before pooling.

Memory-efficient pooled mode (default) uses a two-pass streaming approach:
  Pass A: Load each experiment's h5ad obs metadata (no feature matrix) and
          spatial CSV to match cells and collect bin-metric values. Fixed
          equal-width bin edges are computed from the pooled metric values.
  Pass B: Load each experiment's full feature matrix once, compute NTC stats,
          and accumulate per-(guide, bin) Z-score and L2-normed centroid sums
          for ALL metrics simultaneously, then free the matrix immediately.

This ensures at most one experiment's feature matrix resides in RAM at any time,
and each h5ad is opened only twice (obs-only + full X) regardless of metric count.

Usage:
    # Single experiment (pooled is a no-op, but same code path):
    python cyclops_process/tests/QC/qc_sister_effect_analysis.py -e 89

    # All experiments, pooled across experiments (default, memory-efficient):
    python cyclops_process/tests/QC/qc_sister_effect_analysis.py --all --remove-bad

    # All experiments, per-experiment (old behavior):
    python cyclops_process/tests/QC/qc_sister_effect_analysis.py --all --remove-bad --per-experiment

    # Custom min cells per bin:
    python cyclops_process/tests/QC/qc_sister_effect_analysis.py --all --remove-bad --min-cells 50

    # Force recompute even if a cached CSV exists:
    python cyclops_process/tests/QC/qc_sister_effect_analysis.py --all --remove-bad --force

    # Submit as a SLURM job (400GB, 32 CPUs, 30 min):
    python cyclops_process/tests/QC/qc_sister_effect_analysis.py --all --remove-bad --slurm

    # Submit and don't wait for completion:
    python cyclops_process/tests/QC/qc_sister_effect_analysis.py --all --remove-bad --slurm --no-wait
"""

import argparse
import logging
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from tqdm import tqdm

from cyclops_utils.data.bad_experiments import DEFAULT_EXCLUDE_CATEGORIES, is_excluded
from cyclops_utils.hpc.resource_manager import get_optimal_workers
from cyclops_process.paths import BASE_PATH

EXCLUDE_CATEGORIES = DEFAULT_EXCLUDE_CATEGORIES + ("need_rescue",)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

FEATURE_DIR = "dino_features_v1"
CHANNEL = "Phase2D"
STORAGE_ROOTS = [
    Path(BASE_PATH),
]
HIGHLIGHT_EXPS = {"ops0094", "ops0052"}


# --- Discovery ---

def _discover_cell_h5ad(exclude_bad=False):
    """Find cell-level features_processed h5ad paths across all storage roots."""
    found = {}
    for root in STORAGE_ROOTS:
        if not root.exists():
            continue
        for exp_dir in sorted(root.glob("ops*")):
            exp_id = exp_dir.name.split("_")[0]
            if exp_id in found:
                continue
            if exclude_bad and is_excluded(exp_dir.name, categories=EXCLUDE_CATEGORIES):
                continue
            for sub in [exp_dir / "3-assembly" / FEATURE_DIR / "anndata_objects",
                        exp_dir / "3-assembly" / "results" / "feature_extraction" / FEATURE_DIR / "anndata_objects"]:
                for fn in [f"features_processed_{CHANNEL}.h5ad", "features_processed_Phase.h5ad"]:
                    p = sub / fn
                    if p.exists() and exp_id not in found:
                        found[exp_id] = p
    return found


def _discover_spatial_coherence(exclude_bad=False):
    """Find per_cell_spatial_coherence.csv paths across all storage roots."""
    found = {}
    for root in STORAGE_ROOTS:
        if not root.exists():
            continue
        for exp_dir in sorted(root.glob("ops*")):
            exp_id = exp_dir.name.split("_")[0]
            if exp_id in found:
                continue
            if exclude_bad and is_excluded(exp_dir.name, categories=EXCLUDE_CATEGORIES):
                continue
            sc_path = exp_dir / "3-assembly" / "spatial_coherence" / "per_cell_spatial_coherence.csv"
            if sc_path.exists():
                found[exp_id] = sc_path
    return found


def discover_experiments(exclude_bad=False):
    """Find experiments with both cell-level h5ad and spatial coherence cache.

    Searches all storage roots independently for each file type, then intersects.
    """
    h5ad_map = _discover_cell_h5ad(exclude_bad=exclude_bad)
    sc_map = _discover_spatial_coherence(exclude_bad=exclude_bad)
    common = set(h5ad_map) & set(sc_map)
    h5ad_only = sorted(set(h5ad_map) - set(sc_map))
    sc_only = sorted(set(sc_map) - set(h5ad_map))
    log.info(f"Discovery: {len(h5ad_map)} with h5ad, {len(sc_map)} with spatial, "
             f"{len(common)} with both")
    if h5ad_only:
        log.info(f"  h5ad but NO spatial ({len(h5ad_only)}): {', '.join(h5ad_only)}")
    if sc_only:
        log.info(f"  spatial but NO h5ad ({len(sc_only)}): {', '.join(sc_only)}")
    return {
        eid: {"h5ad": h5ad_map[eid], "spatial": sc_map[eid]}
        for eid in sorted(common)
    }


def resolve_experiment(exp_arg):
    """Resolve a shorthand like '89' or 'ops0089' to a full experiment."""
    if not exp_arg.startswith("ops"):
        exp_arg = f"ops{exp_arg.zfill(4)}"
    all_exps = discover_experiments(exclude_bad=False)
    if exp_arg in all_exps:
        return {exp_arg: all_exps[exp_arg]}
    log.error(f"Experiment {exp_arg} not found (need both cell h5ad + spatial coherence)")
    return {}


# --- Core computation ---

N_SR_BINS = 10  # Number of bins for all metrics
MIN_CELLS_PER_BIN = 20  # Minimum cells per (guide, bin) to count that bin
MIN_BIN_FRAC = 0.7    # Fraction of bins that must meet MIN_CELLS_PER_BIN

# Bin strategy per metric:
#   "fixed":   use np.linspace(0, 1, N_SR_BINS+1) — only valid for sister_ratio
#   "integer": dynamically compute N_SR_BINS equal-width integer bins from
#              the observed [min, max] range at runtime
BIN_METRIC_STRATEGY = {
    "sister_ratio":   "fixed",
    "neighbor_count": "integer",
    "sister_count":   "integer",
}


def _normalize_well(well_str):
    """Normalize well format: 'A/1/0_ops0089_20251119' -> 'A1'."""
    import re
    m = re.match(r"^([A-Z]+)/(\d+)/", well_str)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return well_str


def _make_fixed_bins(values, bin_metric, n_bins=N_SR_BINS):
    """Compute equal-width bin edges and labels for a metric.

    Strategy per metric (from BIN_METRIC_STRATEGY):
      "fixed":   np.linspace(0, 1, n_bins+1) — for sister_ratio ∈ [0,1]
      "integer": n_bins equal-width integer bins spanning [observed_min,
                 observed_max], with the last bin open-ended (…, inf] to
                 capture any outliers above the max used for edge computation.
                 Bin width = max(1, ceil((p95 - p5) / n_bins)) so the bins
                 reflect the bulk of the distribution rather than extreme
                 outliers pulling edges apart.

    Returns (bin_edges, bin_labels) as np.ndarray and pd.IntervalIndex,
    or raises ValueError if fewer than 2 non-empty bins result.
    """
    strategy = BIN_METRIC_STRATEGY.get(bin_metric, "fixed")
    finite_vals = values[np.isfinite(values)]

    if strategy == "fixed":
        edges = np.round(np.linspace(0.0, 1.0, n_bins + 1), decimals=10)
    else:  # "integer"
        # Use p95 to set bin width; always start from 0 so low-count cells
        # are included (biologically meaningful baseline).
        # The last bin is capped at max(observed) so midpoints stay finite
        # (no inf bin — outliers above p95 land in the last finite bin).
        p95 = np.percentile(finite_vals, 95)
        span = max(1, int(np.ceil(p95)))
        width = max(1, int(np.ceil(span / n_bins)))
        int_edges = np.arange(0, (n_bins + 1) * width, width, dtype=float)
        # Extend last edge to cover the observed max so no data is dropped.
        obs_max = float(np.max(finite_vals))
        if int_edges[-1] < obs_max:
            int_edges = np.append(int_edges, int_edges[-1] + width)
        edges = int_edges

    # Determine which bins contain at least one observed value
    bin_assignments = pd.cut(finite_vals, bins=edges, labels=False,
                             right=True, include_lowest=True)
    occupied = sorted(set(int(b) for b in bin_assignments if not np.isnan(b)))

    if not occupied:
        raise ValueError(f"No observed values fell into any bin for {bin_metric}")
    if len(occupied) < 2:
        raise ValueError(f"Fewer than 2 occupied bins for {bin_metric}")

    # Keep only contiguous range from first to last occupied bin
    first, last = occupied[0], occupied[-1]
    used_edges = edges[first: last + 2]  # +2: need both boundaries of last bin
    labels = pd.IntervalIndex.from_breaks(used_edges, closed="right")
    return used_edges, labels


def _parse_obs(a):
    """Extract obs DataFrame and perturbation array from an open AnnData.

    Normalises sgRNA/perturbation/label_str columns from categorical to str.
    Returns (obs, perts) or raises KeyError if sgRNA is missing.
    """
    obs = a.obs.copy().reset_index(drop=True)
    for c in ["sgRNA", "perturbation", "label_str"]:
        if c in obs.columns and hasattr(obs[c], "cat"):
            obs[c] = obs[c].astype(str)
    if "perturbation" not in obs.columns and "label_str" in obs.columns:
        obs["perturbation"] = obs["label_str"]
    if "sgRNA" not in obs.columns:
        raise KeyError("h5ad has no sgRNA column")
    perts = obs["perturbation"].values if "perturbation" in obs.columns else obs["sgRNA"].values
    return obs, perts


def _compute_effect(X_group, ntc_mean, ntc_std, ntc_normed):
    """Compute Cohen's d and cosine distance for a group of cells vs NTC."""
    group_mean = X_group.mean(axis=0)
    d = float(np.sqrt(np.mean(((group_mean - ntc_mean) / ntc_std) ** 2)))

    g_norms = np.linalg.norm(X_group, axis=1, keepdims=True)
    g_norms[g_norms == 0] = 1.0
    group_normed_centroid = (X_group / g_norms).mean(axis=0)
    g_c_norm = np.linalg.norm(group_normed_centroid)
    if g_c_norm > 0:
        group_normed_centroid = group_normed_centroid / g_c_norm
    cos_dist = float(1.0 - np.dot(group_normed_centroid, ntc_normed))

    return d, cos_dist


# --- New 2-pass pooled functions ---

def _load_metadata_and_spatial(exp_id, paths, all_bin_metrics):
    """Pass A: open h5ad once (obs only, NO X) + spatial CSV once.

    Returns (non_ntc_df, ntc_df) tuple.  Each DataFrame has columns:
        [_idx, sgRNA, gene_name, <all bin metrics>]
    non_ntc_df: perturbation cells (None if < 100 cells).
    ntc_df:     NTC cells with individual sgRNA names (None if < 10 cells).
    """
    log.info(f"  [pass A] {exp_id}: loading obs + spatial CSV...")
    try:
        a = ad.read_h5ad(paths["h5ad"])
        obs, perts = _parse_obs(a)
        del a  # free immediately — X never loaded
    except (KeyError, Exception) as e:
        log.warning(f"  {exp_id}: failed to load h5ad obs: {e}")
        return None, None

    if "x_position" not in obs.columns or "y_position" not in obs.columns:
        log.warning(f"  {exp_id}: h5ad missing x_position/y_position columns")
        return None, None

    sc_df = pd.read_csv(paths["spatial"], low_memory=False)
    required = ["sgRNA", "sister_ratio", "neighbor_count", "sister_count",
                "well", "x_pheno", "y_pheno", "gene_name"]
    if not all(c in sc_df.columns for c in required):
        log.warning(f"  {exp_id}: spatial CSV missing required columns")
        return None, None

    obs["well_norm"] = obs["well"].apply(_normalize_well)
    obs["x_r"] = obs["x_position"].round(0).astype(int)
    obs["y_r"] = obs["y_position"].round(0).astype(int)
    obs["_idx"] = np.arange(len(obs))

    sc_df["well_norm"] = sc_df["well"].astype(str)
    sc_df["x_r"] = sc_df["x_pheno"].round(0).astype(int)
    sc_df["y_r"] = sc_df["y_pheno"].round(0).astype(int)
    sc_dedup = sc_df.drop_duplicates(subset=["well_norm", "x_r", "y_r"], keep="first")

    keep_cols = ["well_norm", "x_r", "y_r", "gene_name"] + list(all_bin_metrics)
    merged = obs.merge(sc_dedup[keep_cols], on=["well_norm", "x_r", "y_r"], how="inner")

    matched_perts = perts[merged["_idx"].values]
    is_ntc = np.isin(matched_perts, ["NTC", "non-targeting"])

    out_cols = ["_idx", "sgRNA", "gene_name"] + list(all_bin_metrics)

    non_ntc_df = merged[~is_ntc].reset_index(drop=True)
    if len(non_ntc_df) < 100:
        log.warning(f"  {exp_id}: too few matched non-NTC cells ({len(non_ntc_df)})")
        non_ntc_df = None
    else:
        non_ntc_df = non_ntc_df[out_cols].copy()

    ntc_df = merged[is_ntc].reset_index(drop=True)
    if len(ntc_df) < 10:
        ntc_df = None
    else:
        ntc_df = ntc_df[out_cols].copy()

    return non_ntc_df, ntc_df


def _process_experiment_all_metrics(exp_id, paths, matched_df, bin_edges_map,
                                     bin_labels_map, all_bin_metrics,
                                     ntc_matched_df=None):
    """Pass B: open h5ad ONCE (full X). Compute NTC stats + accumulate
    all metrics simultaneously in a single pass over the feature matrix.

    matched_df: pre-cached DataFrame from _load_metadata_and_spatial (Pass A).
    ntc_matched_df: NTC cells from Pass A (optional, for null hypothesis).
    bin_edges_map / bin_labels_map: global bin edges computed after Pass A.

    Returns:
        (ntc_stats, accum_map, ntc_accum_map)
        ntc_stats: dict with ntc_mean, ntc_std, ntc_normed, n_ntc  (or None)
        accum_map: {bin_metric: accum_dict}  for perturbation guides
        ntc_accum_map: {bin_metric: accum_dict}  for NTC guides (or empty)
    """
    log.info(f"  [pass B] {exp_id}: loading full X matrix...")
    try:
        a = ad.read_h5ad(paths["h5ad"])
        X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
        X = X.astype(np.float32)
        obs, perts = _parse_obs(a)
        del a
    except (KeyError, Exception) as e:
        log.warning(f"  {exp_id}: failed to load h5ad in pass B: {e}")
        return None, {m: None for m in all_bin_metrics}, {}

    # --- 1. NTC stats ---
    ntc_mask = np.isin(perts, ["NTC", "non-targeting"])
    X_ntc = X[ntc_mask]
    if len(X_ntc) < 2:
        log.warning(f"  {exp_id}: not enough NTC cells ({len(X_ntc)})")
        del X
        return None, {m: None for m in all_bin_metrics}, {}

    ntc_mean = X_ntc.mean(axis=0)
    ntc_std = X_ntc.std(axis=0, ddof=1)
    ntc_std[ntc_std == 0] = 1.0
    norms_ntc = np.linalg.norm(X_ntc, axis=1, keepdims=True)
    norms_ntc[norms_ntc == 0] = 1.0
    ntc_normed = (X_ntc / norms_ntc).mean(axis=0)
    del X_ntc
    ntc_c_norm = np.linalg.norm(ntc_normed)
    if ntc_c_norm > 0:
        ntc_normed = ntc_normed / ntc_c_norm

    ntc_stats = {
        "ntc_mean": ntc_mean,
        "ntc_std": ntc_std,
        "ntc_normed": ntc_normed.astype(np.float64),
        "n_ntc": int(ntc_mask.sum()),
    }

    # --- 2. Matched subsets (extract before freeing X) ---
    matched_idx = matched_df["_idx"].values
    X_matched = X[matched_idx]

    X_ntc_matched = None
    if ntc_matched_df is not None and len(ntc_matched_df) > 0:
        ntc_idx = ntc_matched_df["_idx"].values
        X_ntc_matched = X[ntc_idx]

    del X  # free full matrix

    # --- 3. Z-score + L2-normalise ONCE for all metrics ---
    Z = (X_matched - ntc_mean) / ntc_std
    norms = np.linalg.norm(X_matched, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_normed = X_matched / norms
    del X_matched, norms

    # Z-score + L2-normalise NTC matched cells (for null hypothesis)
    Z_ntc_m = None
    X_ntc_normed_m = None
    if X_ntc_matched is not None:
        Z_ntc_m = (X_ntc_matched - ntc_mean) / ntc_std
        norms_ntc_m = np.linalg.norm(X_ntc_matched, axis=1, keepdims=True)
        norms_ntc_m[norms_ntc_m == 0] = 1.0
        X_ntc_normed_m = X_ntc_matched / norms_ntc_m
        del X_ntc_matched, norms_ntc_m

    sgrnas = matched_df["sgRNA"].values
    genes = matched_df["gene_name"].values
    n_feat = Z.shape[1]

    # --- 4. Accumulate for ALL metrics in a single cell loop ---
    accum_map = {}
    for m in all_bin_metrics:
        if m not in bin_edges_map:
            accum_map[m] = None
            continue
        bin_edges = bin_edges_map[m]
        bin_labels = bin_labels_map[m]
        sr_vals = matched_df[m].values
        sr_bin_idx = np.searchsorted(bin_edges[1:-1], sr_vals, side="right")
        sr_bin_idx = np.clip(sr_bin_idx, 0, len(bin_labels) - 1)

        accum = {}
        for i in range(len(sgrnas)):
            guide = sgrnas[i]
            gene = genes[i]
            b_idx = int(sr_bin_idx[i])
            key = (guide, b_idx)
            if key not in accum:
                accum[key] = {
                    "Z_sum": np.zeros(n_feat, dtype=np.float64),
                    "X_normed_sum": np.zeros(n_feat, dtype=np.float64),
                    "count": 0,
                    "gene_name": gene,
                }
            accum[key]["Z_sum"] += Z[i].astype(np.float64)
            accum[key]["X_normed_sum"] += X_normed[i].astype(np.float64)
            accum[key]["count"] += 1

        accum_map[m] = accum
        log.info(f"  [pass B] {exp_id} [{m}]: accumulated {len(matched_df)} cells, "
                 f"{len(set(g for g, _ in accum))} guides")

    del Z, X_normed

    # --- 5. Accumulate NTC guides for null hypothesis ---
    ntc_accum_map = {}
    if Z_ntc_m is not None:
        ntc_sgrnas = ntc_matched_df["sgRNA"].values
        ntc_genes = ntc_matched_df["gene_name"].values
        for m in all_bin_metrics:
            if m not in bin_edges_map:
                continue
            bin_edges = bin_edges_map[m]
            bin_labels = bin_labels_map[m]
            sr_vals = ntc_matched_df[m].values
            sr_bin_idx = np.searchsorted(bin_edges[1:-1], sr_vals, side="right")
            sr_bin_idx = np.clip(sr_bin_idx, 0, len(bin_labels) - 1)

            ntc_accum = {}
            for i in range(len(ntc_sgrnas)):
                guide = ntc_sgrnas[i]
                gene = ntc_genes[i]
                b_idx = int(sr_bin_idx[i])
                key = (guide, b_idx)
                if key not in ntc_accum:
                    ntc_accum[key] = {
                        "Z_sum": np.zeros(n_feat, dtype=np.float64),
                        "X_normed_sum": np.zeros(n_feat, dtype=np.float64),
                        "count": 0,
                        "gene_name": gene,
                    }
                ntc_accum[key]["Z_sum"] += Z_ntc_m[i].astype(np.float64)
                ntc_accum[key]["X_normed_sum"] += X_ntc_normed_m[i].astype(np.float64)
                ntc_accum[key]["count"] += 1

            ntc_accum_map[m] = ntc_accum
            log.info(f"  [pass B] {exp_id} [{m}]: NTC accumulated {len(ntc_matched_df)} cells, "
                     f"{len(set(g for g, _ in ntc_accum))} NTC guides")

        del Z_ntc_m, X_ntc_normed_m

    return ntc_stats, accum_map, ntc_accum_map


def _scan_metadata_all_experiments(exp_items, all_bin_metrics, n_sr_bins, n_workers):
    """Pass A orchestration: scan obs + spatial CSV for all experiments.

    Runs _load_metadata_and_spatial in parallel. Caches matched DataFrames
    for reuse in Pass B. Computes global bin edges from concatenated metric
    values across all experiments.

    Returns dict with:
        "matched_df_map":  {exp_id: DataFrame}
        "valid_exp_items": [(exp_id, paths), ...]
        "bin_edges_map":   {bin_metric: np.ndarray}
        "bin_labels_map":  {bin_metric: pd.IntervalIndex}
    or None on failure.
    """
    exp_items = list(exp_items)
    n_exps = len(exp_items)

    log.info(f"Pass A: scanning metadata + spatial CSVs "
             f"({n_exps} experiments, {n_workers} workers)...")
    pa_results = Parallel(n_jobs=n_workers, backend="loky")(
        delayed(_load_metadata_and_spatial)(eid, paths, all_bin_metrics)
        for eid, paths in tqdm(exp_items, desc="Pass A: metadata scan")
    )

    matched_df_map = {}
    ntc_matched_df_map = {}
    metric_vals = {m: [] for m in all_bin_metrics}
    valid_exp_items = []

    for (eid, paths), (meta, ntc_meta) in zip(exp_items, pa_results):
        if meta is None:
            continue
        matched_df_map[eid] = meta
        if ntc_meta is not None:
            ntc_matched_df_map[eid] = ntc_meta
        valid_exp_items.append((eid, paths))
        for m in all_bin_metrics:
            metric_vals[m].append(meta[m].values)

    if not valid_exp_items:
        log.error("No experiments produced valid metadata in pass A")
        return None

    log.info(f"  Pass A: {len(valid_exp_items)}/{n_exps} experiments passed validation")

    bin_edges_map = {}
    bin_labels_map = {}
    for m in all_bin_metrics:
        all_vals = np.concatenate(metric_vals[m])
        try:
            edges, labels = _make_fixed_bins(all_vals, m, n_bins=n_sr_bins)
        except ValueError as e:
            log.error(f"Cannot create {m} bins: {e} — skipping metric")
            continue
        if len(labels) < 2:
            log.error(f"Fewer than 2 {m} bins ({len(labels)}), skipping metric")
            continue
        bin_edges_map[m] = edges
        bin_labels_map[m] = labels
        log.info(f"  Fixed {m} bins ({len(labels)}): {[str(b) for b in labels]}")

    if not bin_edges_map:
        log.error("No valid bin edges produced for any metric")
        return None

    return {
        "matched_df_map": matched_df_map,
        "ntc_matched_df_map": ntc_matched_df_map,
        "valid_exp_items": valid_exp_items,
        "bin_edges_map": bin_edges_map,
        "bin_labels_map": bin_labels_map,
    }


def _finalize_one_metric(bin_metric, all_exp_accums, global_ntc_normed,
                          bin_labels, min_cells_per_bin, downsample=True):
    """Finalize: merge per-experiment accumulators for one metric, filter guides,
    compute Cohen's d and cosine distance.

    all_exp_accums: list of (exp_id, accum_dict) tuples for this metric.
    global_ntc_normed: np.ndarray — the global NTC reference direction.
    bin_labels: pd.IntervalIndex for this metric.
    min_cells_per_bin: minimum cells per (guide, bin) to include.
    downsample: if True, k = min(count across bins) and k >= min_cells_per_bin;
                if False (native), each bin must independently have >= min_cells_per_bin.

    Returns DataFrame with one row per (guide, bin), or None on failure.
    """
    actual_bins = len(bin_labels)
    variant = "downsampled" if downsample else "native"

    # Merge per-experiment accumulators.
    # Store per-experiment partial sums so downsampled mode can subsample.
    global_accum = {}
    n_exps_processed = 0
    for eid, exp_accum in all_exp_accums:
        if exp_accum is None:
            continue
        n_exps_processed += 1
        for (guide, b_idx), vals in exp_accum.items():
            key = (guide, b_idx)
            if key not in global_accum:
                global_accum[key] = {
                    "Z_sum": np.zeros_like(vals["Z_sum"]),
                    "X_normed_sum": np.zeros_like(vals["X_normed_sum"]),
                    "count": 0,
                    "gene_name": vals["gene_name"],
                    "exp_set": set(),
                    # Per-experiment partial sums for downsampled subsampling
                    "exp_parts": [],
                }
            global_accum[key]["Z_sum"] += vals["Z_sum"]
            global_accum[key]["X_normed_sum"] += vals["X_normed_sum"]
            global_accum[key]["count"] += vals["count"]
            global_accum[key]["exp_set"].add(eid)
            global_accum[key]["exp_parts"].append({
                "Z_sum": vals["Z_sum"],
                "X_normed_sum": vals["X_normed_sum"],
                "count": vals["count"],
            })

    log.info(f"  [{bin_metric}/{variant}] Merged {n_exps_processed} experiments; "
             f"{len(set(g for g, _ in global_accum))} unique guides")

    if not global_accum:
        log.error(f"No data accumulated for {bin_metric}/{variant}")
        return None

    # Guide filtering and effect size computation
    guide_bins = {}
    for (guide, b_idx), vals in global_accum.items():
        guide_bins.setdefault(guide, {})[b_idx] = vals

    n_total_guides = len(guide_bins)
    n_missing_bins = 0
    n_too_few_cells = 0
    n_guides_kept = 0
    rows = []
    k_values = []
    rng = np.random.RandomState(42)

    # Native mode: allow guides present in >= 2 bins (partial coverage ok).
    # Downsampled mode: require ALL bins (needed for fair k=min subsampling).
    min_bins_required = 2 if not downsample else actual_bins

    for guide, bins_dict in guide_bins.items():
        if len(bins_dict) < min_bins_required:
            n_missing_bins += 1
            continue

        if downsample:
            # Downsampled: k = min cells across all bins; k must meet threshold
            k = min(v["count"] for v in bins_dict.values())
            if k < min_cells_per_bin:
                n_too_few_cells += 1
                continue
        else:
            # Native: no downsampling — only require at least 1 cell per bin.
            # All cells are used; the centroid is the true mean, not subsampled.
            min_bin = min(v["count"] for v in bins_dict.values())
            if min_bin < 1:
                n_too_few_cells += 1
                continue
            k = sum(v["count"] for v in bins_dict.values())  # total cells

        n_guides_kept += 1
        k_values.append(k)
        gene = next(iter(bins_dict.values()))["gene_name"]
        n_exps_guide = len(set().union(*(v["exp_set"] for v in bins_dict.values())))

        for rank, b in enumerate(bin_labels):
            if rank not in bins_dict:
                continue  # guide has no cells in this bin (native partial coverage)
            vals = bins_dict[rank]
            count = vals["count"]

            if downsample and count > k:
                # Subsample k cells from per-experiment partial sums.
                # Build an array where each experiment contributes `count_exp`
                # virtual cells; randomly pick k of them, then recompute the
                # centroid from the selected experiments weighted by how many
                # cells were picked from each.
                parts = vals["exp_parts"]
                exp_counts = np.array([p["count"] for p in parts])
                # Expand experiment indices: [0,0,..,1,1,..,2,..]
                exp_idx = np.repeat(np.arange(len(parts)), exp_counts)
                chosen = rng.choice(exp_idx, size=k, replace=False)
                # Count how many cells chosen from each experiment
                chosen_counts = np.bincount(chosen, minlength=len(parts))
                Z_sum_sub = sum(
                    p["Z_sum"] * (c / p["count"]) if p["count"] > 0 else 0.0
                    for p, c in zip(parts, chosen_counts) if c > 0
                )
                Xn_sum_sub = sum(
                    p["X_normed_sum"] * (c / p["count"]) if p["count"] > 0 else 0.0
                    for p, c in zip(parts, chosen_counts) if c > 0
                )
                mean_Z = Z_sum_sub / k
                mean_X_normed = Xn_sum_sub / k
                use_count = k
            else:
                mean_Z = vals["Z_sum"] / count
                mean_X_normed = vals["X_normed_sum"] / count
                use_count = count

            d = float(np.sqrt(np.mean(mean_Z ** 2)))

            c_norm = np.linalg.norm(mean_X_normed)
            if c_norm > 0:
                mean_X_normed = mean_X_normed / c_norm
            cos_dist = float(1.0 - np.dot(mean_X_normed, global_ntc_normed))

            rows.append({
                "experiment": "pooled",
                "sgRNA": guide,
                "gene_name": gene,
                "sr_bin": str(b),
                "sr_bin_mid": float(b.mid),
                "sr_bin_rank": rank,
                "n_cells_total": sum(v["count"] for v in bins_dict.values()),
                "n_cells_bin": use_count,
                "n_experiments": n_exps_guide,
                "k_subsampled": k,
                "cohens_d": d,
                "cosine_dist": cos_dist,
            })

    log.info(f"  [{bin_metric}/{variant}] Guide filter: {n_total_guides} total -> "
             f"{n_missing_bins} missing bins (need >={min_bins_required}), "
             f"{n_too_few_cells} too few cells, "
             f"{n_guides_kept} kept ({100*n_guides_kept/max(n_total_guides,1):.1f}%)")

    if not rows:
        log.error(f"No guides survived filtering for {bin_metric}/{variant}")
        return None

    result = pd.DataFrame(rows)
    k_median = int(np.median(k_values))
    k_min, k_max = int(np.min(k_values)), int(np.max(k_values))
    log.info(f"  [{bin_metric}/{variant}] k: min={k_min}, median={k_median}, max={k_max}")
    log.info(f"  [{bin_metric}/{variant}] Output: {n_guides_kept} guides x {actual_bins} bins "
             f"= {len(result)} rows")
    return result


def _aggregate_global_ntc(ntc_stats_list):
    """Aggregate per-experiment NTC stats into a global NTC normed direction.

    Enforces consistent feature dimension (keeps most common).
    Returns (global_ntc_normed, canonical_dim, valid_ntc_stats) or None.
    """
    if not ntc_stats_list:
        return None

    dim_counts = {}
    for s in ntc_stats_list:
        d = len(s["ntc_normed"])
        dim_counts.setdefault(d, []).append(s)
    canonical_dim = max(dim_counts, key=lambda d: len(dim_counts[d]))
    if len(dim_counts) > 1:
        dropped = {d: len(ss) for d, ss in dim_counts.items() if d != canonical_dim}
        log.warning(f"  Multiple feature dims: {dropped} dropped (canonical={canonical_dim})")

    valid_stats = dim_counts[canonical_dim]
    total_ntc = sum(s["n_ntc"] for s in valid_stats)
    global_ntc_normed = np.zeros(canonical_dim, dtype=np.float64)
    for s in valid_stats:
        global_ntc_normed += s["ntc_normed"] * s["n_ntc"]
    global_ntc_normed /= total_ntc
    norm = np.linalg.norm(global_ntc_normed)
    if norm > 0:
        global_ntc_normed /= norm

    return global_ntc_normed, canonical_dim, valid_stats


def compute_pooled_effects_all_metrics(exp_items, all_bin_metrics, n_sr_bins=N_SR_BINS,
                                        min_cells_per_bin=MIN_CELLS_PER_BIN, n_workers=1):
    """2-pass memory-efficient pooled computation for multiple bin metrics.

    Pass A (parallel): load h5ad obs + spatial CSV once per experiment.
        Cache matched DataFrames, compute global bin edges.
    Pass B (parallel): load h5ad full X once per experiment.
        Compute per-experiment NTC stats, accumulate Z-scored + L2-normed
        centroids for ALL metrics simultaneously.
    Finalize (serial): aggregate NTC, merge accumulators, filter guides,
        compute effect sizes — twice per metric (native + downsampled).

    I/O cost: 2 h5ad opens + 1 spatial CSV open per experiment.

    Returns dict mapping bin_metric -> {"native": DataFrame, "downsampled": DataFrame}.
    """
    # ── Pass A ────────────────────────────────────────────────────────────────
    scan = _scan_metadata_all_experiments(exp_items, all_bin_metrics, n_sr_bins, n_workers)
    if scan is None:
        return {m: {"native": None, "downsampled": None} for m in all_bin_metrics}

    matched_df_map = scan["matched_df_map"]
    ntc_matched_df_map = scan["ntc_matched_df_map"]
    valid_exp_items = scan["valid_exp_items"]
    bin_edges_map = scan["bin_edges_map"]
    bin_labels_map = scan["bin_labels_map"]

    valid_metrics = [m for m in all_bin_metrics if m in bin_edges_map]

    # ── Pass B ────────────────────────────────────────────────────────────────
    log.info(f"Pass B: loading full X + accumulating all metrics "
             f"({len(valid_exp_items)} experiments, {n_workers} workers)...")
    pb_results = Parallel(n_jobs=n_workers, backend="loky")(
        delayed(_process_experiment_all_metrics)(
            eid, paths, matched_df_map[eid], bin_edges_map, bin_labels_map, valid_metrics,
            ntc_matched_df=ntc_matched_df_map.get(eid)
        )
        for eid, paths in tqdm(valid_exp_items, desc="Pass B: accumulate")
    )

    # ── Aggregate NTC stats ───────────────────────────────────────────────────
    ntc_stats_list = [ntc_s for (ntc_s, _, _) in pb_results if ntc_s is not None]
    agg = _aggregate_global_ntc(ntc_stats_list)
    if agg is None:
        log.error("No valid NTC stats from pass B")
        return {m: {"native": None, "downsampled": None} for m in all_bin_metrics}

    global_ntc_normed, canonical_dim, _ = agg

    # ── Finalize each metric (native + downsampled + null) ──────────────────
    results = {}
    for m in all_bin_metrics:
        if m not in bin_labels_map:
            results[m] = {"native": None, "downsampled": None, "null": None}
            continue

        # Collect per-experiment accumulators for this metric
        all_exp_accums = []
        all_ntc_accums = []
        for (eid, _), (ntc_s, accum_map, ntc_accum_map) in zip(valid_exp_items, pb_results):
            if ntc_s is None or len(ntc_s["ntc_normed"]) != canonical_dim:
                continue
            all_exp_accums.append((eid, accum_map.get(m)))
            if ntc_accum_map.get(m):
                all_ntc_accums.append((eid, ntc_accum_map[m]))

        results[m] = {
            "native": _finalize_one_metric(
                bin_metric=m, all_exp_accums=all_exp_accums,
                global_ntc_normed=global_ntc_normed, bin_labels=bin_labels_map[m],
                min_cells_per_bin=min_cells_per_bin, downsample=False),
            "downsampled": _finalize_one_metric(
                bin_metric=m, all_exp_accums=all_exp_accums,
                global_ntc_normed=global_ntc_normed, bin_labels=bin_labels_map[m],
                min_cells_per_bin=min_cells_per_bin, downsample=True),
            "null": _finalize_one_metric(
                bin_metric=m, all_exp_accums=all_ntc_accums,
                global_ntc_normed=global_ntc_normed, bin_labels=bin_labels_map[m],
                min_cells_per_bin=1, downsample=False) if all_ntc_accums else None,
        }

    return results


def compute_pooled_effects_streaming(exp_items, n_sr_bins=N_SR_BINS,
                                     min_cells_per_bin=MIN_CELLS_PER_BIN,
                                     n_workers=1, bin_metric="sister_ratio"):
    """Single-metric convenience wrapper around compute_pooled_effects_all_metrics.

    Returns dict {"native": DataFrame, "downsampled": DataFrame}.
    """
    results = compute_pooled_effects_all_metrics(
        exp_items, all_bin_metrics=[bin_metric],
        n_sr_bins=n_sr_bins, min_cells_per_bin=min_cells_per_bin,
        n_workers=n_workers)
    return results.get(bin_metric, {"native": None, "downsampled": None})


def process_experiment(exp_id, paths, n_sr_bins=N_SR_BINS,
                       min_cells_per_bin=MIN_CELLS_PER_BIN,
                       bin_metric="sister_ratio"):
    """Process one experiment: bin cells by bin_metric, compute per-(guide, bin) effect sizes.

    1. Match h5ad cells to spatial CSV cells via (well, x_round, y_round)
    2. Assign each matched non-NTC cell a bin_metric quantile bin
    3. For each guide: find min cells across bins → k; drop guides missing from any bin
       or with k < min_cells_per_bin
    4. Subsample each (guide, bin) to k cells, compute centroid → Cohen's d and cosine distance

    bin_metric: spatial CSV column to bin cells by ("sister_ratio" or "neighbor_count").

    Returns DataFrame with one row per (guide, sr_bin):
        experiment, sgRNA, gene_name, sr_bin, sr_bin_mid, n_cells_total,
        k_subsampled, cohens_d, cosine_dist
    """
    log.info(f"Processing {exp_id}...")

    # Load cell-level features
    try:
        a = ad.read_h5ad(paths["h5ad"])
    except Exception as e:
        log.warning(f"  {exp_id}: failed to read h5ad: {e}")
        return None
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    X = X.astype(np.float64)

    try:
        obs, perts = _parse_obs(a)
    except KeyError:
        log.warning(f"  {exp_id}: no sgRNA column in h5ad")
        del a, X
        return None
    del a

    # NTC reference (use all NTC cells, not matched subset)
    ntc_mask = np.isin(perts, ["NTC", "non-targeting"])
    X_ntc = X[ntc_mask]
    if len(X_ntc) < 2:
        log.warning(f"  {exp_id}: not enough NTC cells ({len(X_ntc)})")
        return None

    ntc_mean = X_ntc.mean(axis=0)
    ntc_std = X_ntc.std(axis=0, ddof=1)
    ntc_std[ntc_std == 0] = 1.0

    norms_ntc = np.linalg.norm(X_ntc, axis=1, keepdims=True)
    norms_ntc[norms_ntc == 0] = 1.0
    ntc_normed = (X_ntc / norms_ntc).mean(axis=0)
    ntc_c_norm = np.linalg.norm(ntc_normed)
    if ntc_c_norm > 0:
        ntc_normed = ntc_normed / ntc_c_norm

    # Load per-cell spatial coherence and match to h5ad cells
    sc_df = pd.read_csv(paths["spatial"], low_memory=False)
    required = ["sgRNA", "sister_ratio", "neighbor_count", "sister_count", "well", "x_pheno", "y_pheno"]
    if not all(c in sc_df.columns for c in required):
        log.warning(f"  {exp_id}: spatial CSV missing required columns")
        return None

    # Match cells: normalize wells and round coordinates
    obs["well_norm"] = obs["well"].apply(_normalize_well)
    obs["x_r"] = obs["x_position"].round(0).astype(int) if "x_position" in obs.columns else None
    obs["y_r"] = obs["y_position"].round(0).astype(int) if "y_position" in obs.columns else None
    if obs["x_r"] is None:
        log.warning(f"  {exp_id}: h5ad missing x_position/y_position columns")
        return None

    sc_df["well_norm"] = sc_df["well"].astype(str)
    sc_df["x_r"] = sc_df["x_pheno"].round(0).astype(int)
    sc_df["y_r"] = sc_df["y_pheno"].round(0).astype(int)
    sc_dedup = sc_df.drop_duplicates(subset=["well_norm", "x_r", "y_r"], keep="first")

    obs["_idx"] = np.arange(len(obs))
    merged = obs.merge(
        sc_dedup[["well_norm", "x_r", "y_r", bin_metric, "gene_name"]],
        on=["well_norm", "x_r", "y_r"], how="inner",
    )
    log.info(f"  {exp_id}: matched {len(merged)}/{len(obs)} cells "
             f"({100*len(merged)/len(obs):.1f}%)")

    # Filter to non-NTC matched cells
    matched_idx = merged["_idx"].values
    matched_perts = perts[matched_idx]
    non_ntc = ~np.isin(matched_perts, ["NTC", "non-targeting"])
    merged = merged[non_ntc].reset_index(drop=True)

    if len(merged) < 100:
        log.warning(f"  {exp_id}: too few matched non-NTC cells ({len(merged)})")
        return None

    # Bin cells by fixed equal-width bins for this metric
    sr_values = merged[bin_metric].values
    try:
        bin_edges, bin_label_index = _make_fixed_bins(sr_values, bin_metric)
    except ValueError:
        log.warning(f"  {exp_id}: cannot create {bin_metric} bins (too few unique values)")
        return None
    merged["sr_bin"] = pd.cut(sr_values, bins=bin_edges, labels=bin_label_index,
                               right=True, include_lowest=True)
    merged = merged[merged["sr_bin"].notna()].reset_index(drop=True)
    matched_idx = merged["_idx"].values
    X_matched = X[matched_idx]
    sgrnas_matched = obs["sgRNA"].values[matched_idx]

    bin_labels = list(bin_label_index)
    actual_bins = len(bin_labels)
    if actual_bins < 2:
        log.warning(f"  {exp_id}: fewer than 2 {bin_metric} bins, skipping")
        return None

    # Gene name mapping from spatial CSV
    gene_map = merged.groupby("sgRNA")["gene_name"].first().to_dict()

    # For each guide: count cells per bin, find min → k for that guide
    # Only keep guides present in ALL bins with k >= min_cells_per_bin
    rng = np.random.RandomState(42)
    rows = []
    n_guides_kept = 0
    k_values = []

    all_guides = np.unique(sgrnas_matched)
    n_total_guides = len(all_guides)
    n_missing_bins = 0
    n_too_few_cells = 0

    for guide in all_guides:
        guide_mask = sgrnas_matched == guide
        guide_bins = merged.loc[guide_mask, "sr_bin"].values
        guide_X = X_matched[guide_mask]

        # Count per bin
        bin_counts = {}
        bin_indices = {}
        for i, b in enumerate(guide_bins):
            if b not in bin_counts:
                bin_counts[b] = 0
                bin_indices[b] = []
            bin_counts[b] += 1
            bin_indices[b].append(i)

        # Must be present in all bins
        if len(bin_counts) < actual_bins:
            n_missing_bins += 1
            continue
        k = min(bin_counts.values())
        if k < min_cells_per_bin:
            n_too_few_cells += 1
            continue

        n_guides_kept += 1
        k_values.append(k)
        gene = gene_map.get(guide, guide)

        for rank, b in enumerate(bin_labels):
            idx = np.array(bin_indices[b])
            if len(idx) > k:
                idx = rng.choice(idx, size=k, replace=False)
            X_group = guide_X[idx]

            d, cos_dist = _compute_effect(X_group, ntc_mean, ntc_std, ntc_normed)

            rows.append({
                "experiment": exp_id,
                "sgRNA": guide,
                "gene_name": gene,
                "sr_bin": str(b),
                "sr_bin_mid": float(b.mid),
                "sr_bin_rank": rank,
                "n_cells_total": int(guide_mask.sum()),
                "k_subsampled": k,
                "cohens_d": d,
                "cosine_dist": cos_dist,
            })

    # Filtering report
    log.info(f"  {exp_id} guide filter report:")
    log.info(f"    Total non-NTC guides (matched): {n_total_guides}")
    log.info(f"    Dropped — missing from >=1 SR bin: {n_missing_bins}")
    log.info(f"    Dropped — min cells/bin < {min_cells_per_bin}: {n_too_few_cells}")
    log.info(f"    Kept: {n_guides_kept} "
             f"({100*n_guides_kept/max(n_total_guides,1):.1f}%)")

    if not rows:
        log.warning(f"  {exp_id}: no guides survived filtering")
        return None

    result = pd.DataFrame(rows)
    k_median = int(np.median(k_values))
    k_min, k_max = int(np.min(k_values)), int(np.max(k_values))
    log.info(f"    k (cells/guide/bin): min={k_min}, median={k_median}, max={k_max}")
    log.info(f"    Output: {n_guides_kept} guides x {actual_bins} bins = {len(result)} rows")
    return result


# --- Plotting ---


def _per_bin_cell_stats(df, col="n_cells_bin"):
    """Return per-bin min/max of a cell-count column as {sr_bin_mid: (min, max)}.

    col="n_cells_bin" for native (actual cells), "k_subsampled" for downsampled
    (effective cells used per bin after subsampling).
    """
    if col not in df.columns:
        return {}
    grp = df.groupby("sr_bin_mid")[col]
    return {mid: (int(g.min()), int(g.max())) for mid, g in grp}




def _annotate_regression(ax, x, y):
    """Add Pearson r, Spearman rho, and regression line to an axis."""
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 5:
        return
    r, p_r = stats.pearsonr(x, y)
    rho, p_rho = stats.spearmanr(x, y)
    try:
        slope, intercept = np.polyfit(x, y, 1)
        x_fit = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_fit, slope * x_fit + intercept, "r--", lw=1.5, alpha=0.6)
    except (np.linalg.LinAlgError, ValueError):
        slope = np.nan
    ax.text(0.05, 0.95,
            f"Pearson r={r:.3f}, p={p_r:.2e}\n"
            f"Spearman \u03c1={rho:.3f}, p={p_rho:.2e}\n"
            f"slope={slope:.4f}",
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))


def create_dose_response_grid(all_data, output_path, y_col="cohens_d",
                               ylabel="Cohen's d", subtitle="",
                               bin_metric_label="Sister Ratio",
                               cell_col="n_cells_bin",
                               null_data=None):
    """Grid of per-experiment dose-response: sr_bin_mid vs effect size.

    Each point = mean effect size across all guides at that bin_metric bin (IQR shaded).
    Data has one row per (guide x sr_bin).
    """
    experiments = sorted(all_data["experiment"].unique())
    n = len(experiments)
    if n == 0:
        return pd.DataFrame()

    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    regression_rows = []
    for idx, exp_id in enumerate(experiments):
        ax = axes[idx // ncols][idx % ncols]
        edf = all_data[all_data["experiment"] == exp_id]

        x = edf["sr_bin_mid"].values
        y = edf[y_col].values
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        n_guides = edf["sgRNA"].nunique()

        if len(x) < 10:
            ax.text(0.5, 0.5, f"n={len(x)}\n(too few)", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(exp_id, fontsize=9)
            regression_rows.append({"experiment": exp_id, "n_guides": n_guides,
                                     "pearson_r": np.nan, "spearman_rho": np.nan, "slope": np.nan})
            continue

        # Aggregate: mean + quartiles at each sr_bin_mid
        bin_df = pd.DataFrame({"x": x, "y": y})
        grp = bin_df.groupby("x")["y"]
        bin_stats = grp.agg(["mean", "std", "count"]).reset_index()
        bin_stats["q25"] = grp.quantile(0.25).values
        bin_stats["q75"] = grp.quantile(0.75).values

        ax.plot(bin_stats["x"], bin_stats["mean"], "o-", color="#1565C0", ms=6, lw=2)
        ax.fill_between(bin_stats["x"], bin_stats["q25"], bin_stats["q75"],
                         alpha=0.15, color="#1565C0", label="IQR")

        # NTC null overlay (gray band behind perturbation data)
        if null_data is not None and not null_data.empty:
            null_grp = null_data.groupby("sr_bin_mid")[y_col]
            null_stats = null_grp.agg(["mean"]).reset_index()
            null_stats["q25"] = null_grp.quantile(0.25).values
            null_stats["q75"] = null_grp.quantile(0.75).values
            null_stats = null_stats.sort_values("sr_bin_mid")
            ax.fill_between(null_stats["sr_bin_mid"], null_stats["q25"], null_stats["q75"],
                             alpha=0.10, color="gray", label="NTC null (IQR)", zorder=0)
            ax.plot(null_stats["sr_bin_mid"], null_stats["mean"], "--", color="gray",
                    lw=1.5, alpha=0.5, label="NTC null (mean)", zorder=0)

        cpb = _per_bin_cell_stats(edf, col=cell_col)
        for _, r in bin_stats.iterrows():
            mid = r["x"]
            lo, hi = cpb.get(mid, (0, 0))
            cell_lbl = f"{lo}\u2013{hi}c" if lo != hi else f"{lo}c"
            ax.text(mid, r["mean"],
                    f"\n{int(r['count'])}g, {cell_lbl}", fontsize=5,
                    ha="center", va="top", alpha=0.5)

        _annotate_regression(ax, x, y)

        is_highlight = exp_id in HIGHLIGHT_EXPS
        n_exps_for_title = (all_data["n_experiments"].max()
                            if exp_id == "pooled" and "n_experiments" in all_data.columns
                            else None)
        panel_title = (f"POOLED ({n_exps_for_title} exps, {n_guides} guides)"
                       if n_exps_for_title is not None
                       else f"{exp_id} ({n_guides} guides)")
        ax.set_title(panel_title, fontsize=9,
                     fontweight="bold" if is_highlight else "normal",
                     color="red" if is_highlight else "black")
        ax.tick_params(labelsize=7)

        # Regression stats
        r_val, rho, slope = np.nan, np.nan, np.nan
        if np.std(x) > 0 and np.std(y) > 0:
            r_val, _ = stats.pearsonr(x, y)
            rho, _ = stats.spearmanr(x, y)
            try:
                slope = np.polyfit(x, y, 1)[0]
            except (np.linalg.LinAlgError, ValueError):
                pass
        regression_rows.append({"experiment": exp_id, "n_guides": n_guides,
                                 "pearson_r": r_val, "spearman_rho": rho, "slope": slope})

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    for row_axes in axes:
        row_axes[0].set_ylabel(ylabel, fontsize=9)
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel(f"{bin_metric_label} (bin midpoint)", fontsize=9)

    is_pooled = len(experiments) == 1 and experiments[0] == "pooled"
    mode_label = "Pooled" if is_pooled else "Per Experiment"
    title = f"{bin_metric_label} Bin vs {ylabel} \u2014 {mode_label}"
    if subtitle:
        title += f"\n{subtitle}"
    fig.suptitle(title, fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {output_path}")

    return pd.DataFrame(regression_rows)


def create_pooled_dose_response(all_data, output_path, subtitle="",
                                bin_metric_label="Sister Ratio",
                                cell_col="n_cells_bin",
                                null_data=None):
    """Pooled dose-response across all experiments (z-scored within experiment).

    Top row: Cohen's d, Bottom row: Cosine distance.
    Left: lineplot by sr_bin_rank, Right: boxplot by sr_bin_rank.
    Uses sr_bin_rank (ordinal 0..N) so bins align across experiments.
    """
    df = all_data.copy()

    # Z-score within experiment to remove experiment-level variation
    for col in ["cohens_d", "cosine_dist"]:
        df[f"{col}_z"] = df.groupby("experiment")[col].transform(
            lambda s: (s - s.mean()) / s.std() if s.std() > 0 else 0.0
        )

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ranks_sorted = sorted(df["sr_bin_rank"].unique())

    # Compute mean sr_bin_mid per rank for axis labels
    rank_mid = df.groupby("sr_bin_rank")["sr_bin_mid"].mean().to_dict()

    # Per-rank cell count stats
    rank_cpb = {}
    if cell_col in df.columns:
        for rank, g in df.groupby("sr_bin_rank")[cell_col]:
            rank_cpb[rank] = (int(g.min()), int(g.max()))

    for row, (zcol, label) in enumerate([
        ("cohens_d_z", "Cohen's d (z-scored)"),
        ("cosine_dist_z", "Cosine Distance (z-scored)"),
    ]):
        # Left: lineplot by sr_bin_rank
        ax = axes[row, 0]
        grp = df.groupby("sr_bin_rank")[zcol]
        bin_stats = grp.agg(["mean", "std", "count"]).reset_index()
        bin_stats["q25"] = grp.quantile(0.25).values
        bin_stats["q75"] = grp.quantile(0.75).values
        bin_stats = bin_stats.sort_values("sr_bin_rank")

        ax.plot(bin_stats["sr_bin_rank"], bin_stats["mean"], "o-", color="#1565C0", ms=6, lw=2)
        ax.fill_between(bin_stats["sr_bin_rank"],
                         bin_stats["q25"], bin_stats["q75"],
                         alpha=0.15, color="#1565C0", label="IQR")

        # NTC null overlay on lineplot
        if null_data is not None and not null_data.empty:
            null_col = zcol.replace("_z", "")  # use raw metric for null
            if null_col in null_data.columns:
                null_grp = null_data.groupby("sr_bin_rank")[null_col]
                null_s = null_grp.agg(["mean"]).reset_index()
                null_s["q25"] = null_grp.quantile(0.25).values
                null_s["q75"] = null_grp.quantile(0.75).values
                null_s = null_s.sort_values("sr_bin_rank")
                ax.fill_between(null_s["sr_bin_rank"], null_s["q25"], null_s["q75"],
                                 alpha=0.10, color="gray", label="NTC null (IQR)", zorder=0)
                ax.plot(null_s["sr_bin_rank"], null_s["mean"], "--", color="gray",
                        lw=1.5, alpha=0.5, label="NTC null", zorder=0)

        for _, r in bin_stats.iterrows():
            rk = int(r["sr_bin_rank"])
            lo, hi = rank_cpb.get(rk, (0, 0))
            cell_lbl = f"{lo}\u2013{hi}c" if lo != hi else f"{lo}c"
            ax.text(r["sr_bin_rank"], r["mean"],
                    f"\n{int(r['count'])}g, {cell_lbl}", fontsize=5,
                    ha="center", va="top", alpha=0.5)

        _annotate_regression(ax, df["sr_bin_rank"].values, df[zcol].values)
        ax.set_xticks(ranks_sorted)
        tick_labels = []
        for i in ranks_sorted:
            lo, hi = rank_cpb.get(i, (0, 0))
            cell_range = f"{lo}\u2013{hi}" if lo != hi else str(lo)
            tick_labels.append(f"{i}\n(~{rank_mid.get(i, 0):.3f})\n[{cell_range}c]")
        ax.set_xticklabels(tick_labels, fontsize=7)
        ax.set_xlabel(f"{bin_metric_label} Bin (low \u2192 high)", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f"Pooled: {bin_metric_label} vs {label}", fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Right: boxplot by sr_bin_rank
        ax2 = axes[row, 1]
        box_data = [df.loc[df["sr_bin_rank"] == r, zcol].values for r in ranks_sorted]
        box_labels = []
        for r in ranks_sorted:
            lo, hi = rank_cpb.get(r, (0, 0))
            cell_range = f"{lo}\u2013{hi}" if lo != hi else str(lo)
            box_labels.append(f"Bin {r}\n(~{rank_mid.get(r, 0):.3f})\n[{cell_range}c]")
        bp = ax2.boxplot(box_data, labels=box_labels,
                         patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor("#1565C0")
            patch.set_alpha(0.4)
        ax2.set_xlabel(f"{bin_metric_label} Bin (low \u2192 high)", fontsize=10)
        ax2.set_ylabel(label, fontsize=10)
        ax2.set_title(f"Pooled: {label} by {bin_metric_label} Bin", fontsize=11, fontweight="bold")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

    n_exps = (df["n_experiments"].max() if "n_experiments" in df.columns
              else df["experiment"].nunique())
    n_guides = df["sgRNA"].nunique()
    n_per_bin = df.groupby("sr_bin_rank").size().iloc[0] if len(ranks_sorted) > 0 else 0
    title = (f"Pooled Sister Effect Analysis ({n_exps} exps, "
             f"{n_guides:,} guides, {n_per_bin:,}/bin)")
    if subtitle:
        title += f"\n{subtitle}"
    fig.suptitle(title, fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {output_path}")


def create_effect_histograms(all_data, output_path, subtitle="",
                             cell_col="n_cells_bin",
                             bin_metric_label="Sister Ratio",
                             null_data=None):
    """Overlaid histograms of per-(guide x bin) effect sizes, colored by sr_bin.

    One row per experiment, 2 columns (Cohen's d, cosine distance).
    For --all mode, also includes a pooled row.
    """
    experiments = sorted(all_data["experiment"].unique())
    n_exps_actual = (all_data["n_experiments"].max()
                     if "n_experiments" in all_data.columns
                     else len(experiments))
    show_pooled = len(experiments) > 1
    n_rows = len(experiments) + (1 if show_pooled else 0)

    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3.5 * n_rows), squeeze=False)

    sr_mids = sorted(m for m in all_data["sr_bin_mid"].unique() if np.isfinite(m))
    mid_to_label = all_data.groupby("sr_bin_mid")["sr_bin"].first().to_dict()
    cmap = plt.cm.coolwarm
    color_norm = plt.Normalize(vmin=min(sr_mids), vmax=max(sr_mids))

    cpb = _per_bin_cell_stats(all_data, col=cell_col)

    def _plot_hist(ax, df, metric, title):
        cpb_local = _per_bin_cell_stats(df, col=cell_col) if df is not all_data else cpb
        for mid in sr_mids:
            subset = df.loc[df["sr_bin_mid"] == mid, metric].values
            if len(subset) > 0:
                color = cmap(color_norm(mid))
                bin_lbl = mid_to_label.get(mid, f"{mid:.3f}")
                lo, hi = cpb_local.get(mid, (0, 0))
                cell_range = f"{lo}\u2013{hi}c" if lo != hi else f"{lo}c"
                ax.hist(subset, bins=40, alpha=0.35, color=color,
                        label=f"{bin_lbl} (n={len(subset)}, {cell_range})",
                        density=True)
        # NTC null overlay: dashed gray KDE of all NTC effect sizes
        if null_data is not None and not null_data.empty and metric in null_data.columns:
            null_vals = null_data[metric].dropna().values
            if len(null_vals) > 5:
                try:
                    kde_null = stats.gaussian_kde(null_vals, bw_method=0.3)
                    x_grid = np.linspace(null_vals.min(), null_vals.max(), 200)
                    ax.plot(x_grid, kde_null(x_grid), "--", color="gray", lw=1.5,
                            alpha=0.6, label="NTC null", zorder=5)
                except Exception:
                    pass
        ax.legend(fontsize=5, title=f"{bin_metric_label} Bin", title_fontsize=6)
        ax.set_xlabel(metric.replace("_", " ").title(), fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx, exp_id in enumerate(experiments):
        edf = all_data[all_data["experiment"] == exp_id]
        is_hl = exp_id in HIGHLIGHT_EXPS
        color = "red" if is_hl else "black"
        weight = "bold" if is_hl else "normal"
        _plot_hist(axes[idx, 0], edf, "cohens_d", f"{exp_id} \u2014 Cohen's d")
        _plot_hist(axes[idx, 1], edf, "cosine_dist", f"{exp_id} \u2014 Cosine Distance")
        for c in range(2):
            axes[idx, c].title.set_color(color)
            axes[idx, c].title.set_fontweight(weight)

    if show_pooled:
        df_pooled = all_data.copy()
        for col in ["cohens_d", "cosine_dist"]:
            df_pooled[col] = df_pooled.groupby("experiment")[col].transform(
                lambda s: (s - s.mean()) / s.std() if s.std() > 0 else 0.0
            )
        _plot_hist(axes[-1, 0], df_pooled, "cohens_d",
                   f"POOLED ({n_exps_actual} exps) \u2014 Cohen's d (z-scored)")
        _plot_hist(axes[-1, 1], df_pooled, "cosine_dist",
                   f"POOLED ({n_exps_actual} exps) \u2014 Cosine Distance (z-scored)")

    title = f"Effect Size Distributions by {bin_metric_label} Bin"
    if subtitle:
        title += f"\n{subtitle}"
    fig.suptitle(title, fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {output_path}")


def create_ridge_by_sister_ratio(all_data, output_path, metric="cohens_d",
                                  xlabel="Cohen's d", title_suffix="",
                                  min_guides=10, overlap=0.55,
                                  bin_metric_label="Sister Ratio",
                                  cell_col="n_cells_bin",
                                  null_data=None):
    """Ridge plot: one KDE row per bin_metric bin, x-axis = effect size.

    Shows how the distribution of per-(guide x bin) effect sizes shifts
    as bin_metric increases. Uses the sr_bin/sr_bin_mid columns directly.
    """
    df = all_data.copy()

    sr_mids = sorted(m for m in df["sr_bin_mid"].unique() if np.isfinite(m))
    # Filter bins with enough data
    counts = df.groupby("sr_bin_mid").size()
    valid_mids = [m for m in sr_mids if counts.get(m, 0) >= min_guides]
    if len(valid_mids) < 2:
        log.warning(f"Ridge plot: fewer than 2 valid bins for {metric}, skipping")
        return

    all_vals = df.loc[df["sr_bin_mid"].isin(valid_mids), metric].values
    p_lo, p_hi = np.percentile(all_vals, [0.5, 99.5])
    p_range = max(p_hi - p_lo, 1e-6)
    x_min = p_lo - 0.10 * p_range
    x_max = p_hi + 0.10 * p_range
    x_range = np.linspace(x_min, x_max, 300)

    row_height = 1.0
    n_rows_plot = len(valid_mids)
    fig_h = max(6, 0.6 * n_rows_plot + 2)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=min(valid_mids), vmax=max(valid_mids))

    # Build sr_bin_mid -> sr_bin label mapping
    sr_mid_to_label = df.groupby("sr_bin_mid")["sr_bin"].first().to_dict()
    cpb = _per_bin_cell_stats(df, col=cell_col)

    for i, mid in enumerate(valid_mids):
        vals = df.loc[df["sr_bin_mid"] == mid, metric].values
        n = len(vals)
        y_offset = i * row_height * (1 - overlap)

        try:
            kde = stats.gaussian_kde(vals, bw_method=0.2)
            density = kde(x_range)
            density = density / density.max() * row_height * 0.8
        except Exception:
            continue

        # NTC null KDE behind perturbation KDE
        if null_data is not None and not null_data.empty and metric in null_data.columns:
            null_bin = null_data.loc[null_data["sr_bin_mid"] == mid, metric].values
            if len(null_bin) > 5:
                try:
                    null_kde = stats.gaussian_kde(null_bin, bw_method=0.2)
                    null_density = null_kde(x_range)
                    null_density = null_density / max(null_density.max(), 1e-12) * row_height * 0.8
                    ax.fill_between(x_range, y_offset, y_offset + null_density,
                                     color="gray", alpha=0.12, linewidth=0, zorder=0)
                    ax.plot(x_range, y_offset + null_density, color="gray",
                            alpha=0.3, linewidth=0.8, zorder=0)
                except Exception:
                    pass

        color = cmap(norm(mid))
        ax.fill_between(x_range, y_offset, y_offset + density,
                         color=color, alpha=0.5, linewidth=0)
        ax.plot(x_range, y_offset + density, color=color, alpha=0.8, linewidth=1.0)

        label = sr_mid_to_label.get(mid, f"{mid:.3f}")
        lo, hi = cpb.get(mid, (0, 0))
        cell_range = f"{lo}\u2013{hi}c" if lo != hi else f"{lo}c"
        ax.text(x_min - 0.02 * (x_max - x_min), y_offset + row_height * 0.1,
                f"{label}  (n={n}, {cell_range})", fontsize=8, ha="right",
                va="center")

        mean_val = np.mean(vals)
        ax.plot(mean_val, y_offset + row_height * 0.05, "v", color=color,
                ms=5, alpha=0.8)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_yticks([])
    ax.set_xlim(x_min - 0.25 * (x_max - x_min), x_max)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    n_exps = (df["n_experiments"].max() if "n_experiments" in df.columns
              else df["experiment"].nunique())
    n_guides = df["sgRNA"].nunique()
    title = (f"Effect Size Distribution by {bin_metric_label} Bin "
             f"({n_exps} exps, {n_guides:,} guides x {len(valid_mids)} bins)")
    if title_suffix:
        title += f"\n{title_suffix.strip()}"
    ax.set_title(title, fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {output_path}")


def create_regression_summary(reg_df, output_path, metric_label="Cohen's d",
                               subtitle="", bin_metric_label="Sister Ratio"):
    """Bar chart of per-experiment regression slopes: effect ~ sr_bin_mid."""
    if reg_df is None or reg_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, max(6, 0.4 * len(reg_df))))

    df = reg_df.dropna(subset=["slope"]).sort_values("slope", ascending=True)
    if df.empty:
        plt.close(fig)
        return

    y_pos = np.arange(len(df))
    colors = [("red" if e in HIGHLIGHT_EXPS else "#1565C0") for e in df["experiment"]]
    ax.barh(y_pos, df["slope"].values, color=colors, alpha=0.8, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{e} ({n}g)" for e, n in zip(df["experiment"], df["n_guides"])],
        fontsize=8,
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(f"Slope ({metric_label} ~ {bin_metric_label} bin)", fontsize=10)
    ax.set_title(f"Per-Experiment Regression: {metric_label} vs {bin_metric_label} Bin",
                 fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["slope"], i, f" r={row['pearson_r']:.2f}", va="center",
                fontsize=6, alpha=0.7)

    title = "Per-Experiment Regression Slopes"
    if subtitle:
        title += f"\n{subtitle}"
    fig.suptitle(title, fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {output_path}")


def create_evo_genes_validation(all_data, evo_genes_path, output_path, subtitle="",
                                bin_metric_label="Sister Ratio",
                                cell_col="n_cells_bin"):
    """Validation plot: individual guide lines for evo_genes subset.

    Shows each guide's effect size across sister_ratio bins as a separate line,
    confirming that the trend is visible at the individual guide level.
    """
    evo_df = pd.read_csv(evo_genes_path)
    evo_genes = set(evo_df["gene_name"].unique())

    mask = all_data["gene_name"].isin(evo_genes)
    df = all_data[mask].copy()

    if df.empty:
        available = set(all_data["gene_name"].unique())
        missing = evo_genes - available
        log.warning(
            f"No evo_genes found in data — {len(missing)}/{len(evo_genes)} genes absent "
            f"from the {len(available)} guides that passed filtering. "
            f"Missing: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}"
        )
        return

    genes_found = sorted(df["gene_name"].unique())
    log.info(f"Evo genes validation: {len(genes_found)} genes found in data")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    cmap = plt.cm.tab20
    colors = {g: cmap(i / max(len(genes_found) - 1, 1))
              for i, g in enumerate(genes_found)}

    for ax, metric, ylabel in [
        (axes[0], "cohens_d", "Cohen's d"),
        (axes[1], "cosine_dist", "Cosine Distance"),
    ]:
        for gene in genes_found:
            gene_df = df[df["gene_name"] == gene]
            first_sgrna = True
            for sgrna in gene_df["sgRNA"].unique():
                sdf = gene_df[gene_df["sgRNA"] == sgrna].sort_values("sr_bin_mid")
                ax.plot(sdf["sr_bin_mid"], sdf[metric], "-o", color=colors[gene],
                        alpha=0.5, ms=3, lw=1.0,
                        label=gene if first_sgrna else "")
                first_sgrna = False

        # Mean trend: evo genes
        mean_evo = df.groupby("sr_bin_mid")[metric].mean()
        ax.plot(mean_evo.index, mean_evo.values, "k-", lw=3, alpha=0.8,
                label="Mean (evo genes)")

        # Mean trend: all genes (for reference)
        mean_all = all_data.groupby("sr_bin_mid")[metric].mean()
        ax.plot(mean_all.index, mean_all.values, "k--", lw=2, alpha=0.4,
                label="Mean (all genes)")

        # Annotate per-bin cell counts on x-axis
        cpb = _per_bin_cell_stats(all_data, col=cell_col)
        mids_sorted = sorted(m for m in all_data["sr_bin_mid"].unique()
                             if np.isfinite(m))
        ax.set_xticks(mids_sorted)
        xlabels = []
        for m in mids_sorted:
            lo, hi = cpb.get(m, (0, 0))
            cell_range = f"{lo}\u2013{hi}" if lo != hi else str(lo)
            xlabels.append(f"{m:.2f}\n[{cell_range}c]")
        ax.set_xticklabels(xlabels, fontsize=7)
        ax.set_xlabel(f"{bin_metric_label} (bin midpoint)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{ylabel} vs {bin_metric_label} Bin", fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=6, loc="upper left",
                  ncol=2, framealpha=0.7)

    n_guides = df["sgRNA"].nunique()
    n_exps = (df["n_experiments"].max() if "n_experiments" in df.columns
              else df["experiment"].nunique())
    title = (f"Evo Genes Validation ({len(genes_found)} genes, "
             f"{n_guides} guides, {n_exps} exps)")
    if subtitle:
        title += f"\n{subtitle}"
    fig.suptitle(title, fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {output_path}")


def create_null_guides_validation(null_data, native_data, output_path, subtitle="",
                                   bin_metric_label="Sister Ratio",
                                   cell_col="n_cells_bin", n_guides=15, seed=42):
    """Null validation plot: individual NTC sgRNA lines across bins.

    Analogous to create_evo_genes_validation but for NTC guides. Randomly
    picks ~n_guides NTC sgRNAs and plots their individual effect-size trends.
    Overlays the mean NTC trend and the mean perturbation trend (from native_data)
    as reference lines.
    """
    if null_data is None or null_data.empty:
        log.warning("No null data for null guides validation plot")
        return

    # Randomly pick n_guides NTC sgRNAs
    rng = np.random.RandomState(seed)
    all_sgrnas = sorted(null_data["sgRNA"].unique())
    n_pick = min(n_guides, len(all_sgrnas))
    chosen = rng.choice(all_sgrnas, size=n_pick, replace=False)
    df = null_data[null_data["sgRNA"].isin(chosen)].copy()

    if df.empty:
        log.warning("No NTC guides survived for null validation plot")
        return

    log.info(f"Null guides validation: {n_pick} NTC sgRNAs selected")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    cmap = plt.cm.tab20
    colors = {g: cmap(i / max(n_pick - 1, 1)) for i, g in enumerate(sorted(chosen))}

    for ax, metric, ylabel in [
        (axes[0], "cohens_d", "Cohen's d"),
        (axes[1], "cosine_dist", "Cosine Distance"),
    ]:
        for sgrna in sorted(chosen):
            sdf = df[df["sgRNA"] == sgrna].sort_values("sr_bin_mid")
            ax.plot(sdf["sr_bin_mid"], sdf[metric], "-o", color=colors[sgrna],
                    alpha=0.5, ms=3, lw=1.0, label=sgrna)

        # Mean NTC trend
        mean_ntc = null_data.groupby("sr_bin_mid")[metric].mean()
        ax.plot(mean_ntc.index, mean_ntc.values, "k-", lw=3, alpha=0.8,
                label="Mean (all NTC)")

        # Mean perturbation trend (reference from native data)
        if native_data is not None and not native_data.empty:
            mean_pert = native_data.groupby("sr_bin_mid")[metric].mean()
            ax.plot(mean_pert.index, mean_pert.values, "r--", lw=2, alpha=0.5,
                    label="Mean (perturbations)")

        # Annotate per-bin cell counts on x-axis
        cpb = _per_bin_cell_stats(null_data, col=cell_col)
        mids_sorted = sorted(m for m in null_data["sr_bin_mid"].unique()
                             if np.isfinite(m))
        ax.set_xticks(mids_sorted)
        xlabels = []
        for m in mids_sorted:
            lo, hi = cpb.get(m, (0, 0))
            cell_range = f"{lo}\u2013{hi}" if lo != hi else str(lo)
            xlabels.append(f"{m:.2f}\n[{cell_range}c]")
        ax.set_xticklabels(xlabels, fontsize=7)
        ax.set_xlabel(f"{bin_metric_label} (bin midpoint)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{ylabel} vs {bin_metric_label} Bin (NTC Null)", fontsize=12,
                     fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axhline(0, color="gray", lw=0.8, ls=":", alpha=0.5)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=6, loc="upper left",
                  ncol=2, framealpha=0.7)

    n_total_ntc = null_data["sgRNA"].nunique()
    n_exps = (null_data["n_experiments"].max() if "n_experiments" in null_data.columns
              else null_data["experiment"].nunique())
    title = (f"NTC Null Validation ({n_pick}/{n_total_ntc} NTC sgRNAs, "
             f"{n_exps} exps)")
    if subtitle:
        title += f"\n{subtitle}"
    fig.suptitle(title, fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {output_path}")


def create_regressed_plots(native_data, null_data, out_dir, bin_metric,
                            bin_metric_label="Sister Ratio", subtitle="",
                            evo_path=None):
    """Create NTC-regressed plots: subtract per-bin NTC baseline from perturbation data.

    For each bin, computes the mean NTC effect size and subtracts it from every
    perturbation guide's effect size in that bin. This removes the spatial/density
    confound revealed by the NTC null analysis.

    Produces:
      - Dose-response grid (regressed Cohen's d and cosine distance)
      - Ridge plots (regressed distributions)
      - Summary comparison plot (raw vs regressed vs NTC null trends)
      - Evo genes validation (regressed)

    Outputs go to out_dir / regressed / bin_metric /.
    """
    if native_data is None or native_data.empty:
        log.warning("No native data for regressed plots")
        return
    if null_data is None or null_data.empty:
        log.warning("No null data for regressed plots — cannot compute regression")
        return

    sub_dir = out_dir / "regressed" / bin_metric
    sub_dir.mkdir(parents=True, exist_ok=True)

    # Compute per-bin NTC baseline (mean effect size)
    ntc_baseline = null_data.groupby("sr_bin_mid")[["cohens_d", "cosine_dist"]].mean()
    ntc_baseline.columns = ["ntc_cohens_d", "ntc_cosine_dist"]

    # Merge NTC baseline into native data and subtract
    reg = native_data.merge(ntc_baseline, left_on="sr_bin_mid", right_index=True, how="left")
    reg["cohens_d_raw"] = reg["cohens_d"]
    reg["cosine_dist_raw"] = reg["cosine_dist"]
    reg["cohens_d"] = reg["cohens_d"] - reg["ntc_cohens_d"].fillna(0)
    reg["cosine_dist"] = reg["cosine_dist"] - reg["ntc_cosine_dist"].fillna(0)

    n_exps = (reg["n_experiments"].max() if "n_experiments" in reg.columns
              else reg["experiment"].nunique())
    n_guides = reg["sgRNA"].nunique()
    mode_str = f"NTC-regressed, {n_exps} exps, binned by {bin_metric_label}"

    # Save regressed CSV
    reg.to_csv(sub_dir / f"sister_effect_regressed__{bin_metric}.csv", index=False)

    # 1. Dose-response grid (regressed)
    create_dose_response_grid(
        reg, sub_dir / "sr_bin_vs_cohens_d_regressed.png",
        y_col="cohens_d", ylabel="Cohen's d (NTC-regressed)",
        subtitle=f"{mode_str}, {subtitle}".strip(", "),
        bin_metric_label=bin_metric_label, cell_col="n_cells_bin")
    create_dose_response_grid(
        reg, sub_dir / "sr_bin_vs_cosine_dist_regressed.png",
        y_col="cosine_dist", ylabel="Cosine Distance (NTC-regressed)",
        subtitle=f"{mode_str}, {subtitle}".strip(", "),
        bin_metric_label=bin_metric_label, cell_col="n_cells_bin")

    # 2. Ridge plots (regressed)
    create_ridge_by_sister_ratio(
        reg, sub_dir / "ridge_cohens_d_regressed.png",
        metric="cohens_d", xlabel="Cohen's d (NTC-regressed)",
        title_suffix=f"\n({mode_str})",
        bin_metric_label=bin_metric_label, cell_col="n_cells_bin")
    create_ridge_by_sister_ratio(
        reg, sub_dir / "ridge_cosine_dist_regressed.png",
        metric="cosine_dist", xlabel="Cosine Distance (NTC-regressed)",
        title_suffix=f"\n({mode_str})",
        bin_metric_label=bin_metric_label, cell_col="n_cells_bin")

    # 3. Histograms (regressed)
    create_effect_histograms(
        reg, sub_dir / "effect_histograms_regressed.png",
        subtitle=f"{mode_str}",
        cell_col="n_cells_bin", bin_metric_label=bin_metric_label)

    # 4. Summary comparison: raw vs NTC null vs regressed trends
    _plot_regression_summary_comparison(
        native_data, null_data, reg, sub_dir / "raw_vs_regressed_comparison.png",
        bin_metric_label=bin_metric_label)

    # 5. Evo genes validation (regressed)
    if evo_path and Path(evo_path).exists():
        create_evo_genes_validation(
            reg, evo_path, sub_dir / "evo_genes_validation_regressed.png",
            subtitle=f"{mode_str}",
            bin_metric_label=bin_metric_label, cell_col="n_cells_bin")

    log.info(f"  All regressed outputs in: {sub_dir}")


def _plot_regression_summary_comparison(native_data, null_data, regressed_data,
                                         output_path, bin_metric_label="Sister Ratio"):
    """3-panel comparison: raw perturbation vs NTC null vs NTC-regressed.

    Shows the mean trend per bin for each, making the confound removal visible.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, metric, ylabel in [
        (axes[0], "cohens_d", "Cohen's d"),
        (axes[1], "cosine_dist", "Cosine Distance"),
    ]:
        # Raw perturbation trend
        raw = native_data.groupby("sr_bin_mid")[metric].agg(["mean", "std"])
        raw_q25 = native_data.groupby("sr_bin_mid")[metric].quantile(0.25)
        raw_q75 = native_data.groupby("sr_bin_mid")[metric].quantile(0.75)
        mids = raw.index.values

        ax.fill_between(mids, raw_q25.values, raw_q75.values,
                         alpha=0.12, color="#1565C0")
        ax.plot(mids, raw["mean"].values, "o-", color="#1565C0", lw=2.5, ms=6,
                label="Perturbation (raw)", zorder=3)

        # NTC null trend
        ntc = null_data.groupby("sr_bin_mid")[metric].agg(["mean"])
        ntc_q25 = null_data.groupby("sr_bin_mid")[metric].quantile(0.25)
        ntc_q75 = null_data.groupby("sr_bin_mid")[metric].quantile(0.75)
        ntc_mids = ntc.index.values

        ax.fill_between(ntc_mids, ntc_q25.values, ntc_q75.values,
                         alpha=0.12, color="gray")
        ax.plot(ntc_mids, ntc["mean"].values, "s--", color="gray", lw=2, ms=5,
                label="NTC null", zorder=2)

        # Regressed perturbation trend
        reg = regressed_data.groupby("sr_bin_mid")[metric].agg(["mean"])
        reg_q25 = regressed_data.groupby("sr_bin_mid")[metric].quantile(0.25)
        reg_q75 = regressed_data.groupby("sr_bin_mid")[metric].quantile(0.75)
        reg_mids = reg.index.values

        ax.fill_between(reg_mids, reg_q25.values, reg_q75.values,
                         alpha=0.12, color="#E65100")
        ax.plot(reg_mids, reg["mean"].values, "D-", color="#E65100", lw=2.5, ms=6,
                label="Perturbation (NTC-regressed)", zorder=4)

        ax.axhline(0, color="black", lw=0.5, ls=":", alpha=0.4)

        # Annotate the fraction of signal that's confound
        raw_range = raw["mean"].max() - raw["mean"].min()
        ntc_range = ntc["mean"].max() - ntc["mean"].min() if len(ntc) > 1 else 0
        confound_pct = (ntc_range / raw_range * 100) if raw_range > 0 else 0
        ax.text(0.02, 0.98,
                f"Confound: {confound_pct:.0f}% of raw trend\n"
                f"Raw range: {raw_range:.4f}\n"
                f"NTC range: {ntc_range:.4f}\n"
                f"Residual: {raw_range - ntc_range:.4f}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

        ax.set_xlabel(f"{bin_metric_label} (bin midpoint)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{ylabel}: Raw vs NTC Null vs Regressed", fontsize=12,
                     fontweight="bold")
        ax.legend(fontsize=9, loc="upper left" if metric == "cosine_dist" else "best")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    n_exps = (native_data["n_experiments"].max() if "n_experiments" in native_data.columns
              else native_data["experiment"].nunique())
    fig.suptitle(f"NTC Confound Regression — {bin_metric_label} ({n_exps} exps)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {output_path}")


# --- Main ---

# Metrics to bin by, with human-readable labels for plot axis / subtitles.
# Three metrics tell a clean comparative story:
#   sister_ratio   — same-guide fraction of neighbors (density-normalised)
#   neighbor_count — total neighbors regardless of guide (pure confluence/density proxy)
#   sister_count   — raw same-guide neighbor count (unnormalised co-localisation)
BIN_METRICS = {
    "sister_ratio": "Sister Ratio",
    "neighbor_count": "Neighbor Count (all guides)",
    "sister_count": "Sister Count (same-guide neighbors)",
}

VARIANTS = ["native", "downsampled"]


def _run_one_metric_variant(bin_metric, variant, all_data, exp_items, out_dir,
                             mode, min_cells, n_workers, per_experiment, evo_path,
                             force, null_data=None, native_data=None):
    """Run the plot pipeline for one (bin_metric, variant) combination.

    all_data: pre-computed DataFrame for this metric+variant (or None to load from cache).
    Outputs go into out_dir / variant / bin_metric /.
    """
    metric_label = BIN_METRICS.get(bin_metric, bin_metric)
    sub_dir = out_dir / variant / bin_metric
    sub_dir.mkdir(parents=True, exist_ok=True)

    cache_path = sub_dir / (
        f"sister_effect_merged__{mode}__{variant}__bins{N_SR_BINS}"
        f"__mincells{min_cells}__{bin_metric}.csv"
    )

    if all_data is None:
        if not force and cache_path.exists():
            log.info(f"  Loading cached [{variant}/{bin_metric}] from {cache_path}")
            all_data = pd.read_csv(cache_path)
        else:
            log.error(f"  No data for [{variant}/{bin_metric}] and no cache")
            return
    else:
        all_data.to_csv(cache_path, index=False)
        log.info(f"  Saved: {cache_path}")

    if all_data.empty:
        log.error(f"  Empty data for [{variant}/{bin_metric}]")
        return

    n_exps_loaded = (all_data["n_experiments"].max()
                     if "n_experiments" in all_data.columns
                     else all_data["experiment"].nunique()
                     if "experiment" in all_data.columns else 0)
    n_guides = all_data["sgRNA"].nunique()
    n_bins = all_data["sr_bin_mid"].nunique()
    log.info(f"  [{variant}/{bin_metric}]: {n_guides:,} guides x {n_bins} bins = "
             f"{len(all_data):,} rows ({n_exps_loaded} exps, {mode})")

    k_values = all_data["k_subsampled"]
    k_min, k_max, k_med = int(k_values.min()), int(k_values.max()), int(k_values.median())
    subsample_str = (f"k={k_min}\u2013{k_max} cells/guide/bin (median {k_med})"
                     if variant == "downsampled"
                     else f"no downsampling, all cells")
    mode_str = f"{mode}, {variant}, {n_exps_loaded} exps, binned by {metric_label}"

    # For downsampled, show k_subsampled (effective cells used); for native, n_cells_bin.
    cell_col = "k_subsampled" if variant == "downsampled" else "n_cells_bin"

    # 1. Dose-response grid
    reg_d = create_dose_response_grid(
        all_data, sub_dir / "sr_bin_vs_cohens_d_per_exp.png",
        y_col="cohens_d", ylabel="Cohen's d",
        subtitle=f"{mode_str}, {subsample_str}",
        bin_metric_label=metric_label, cell_col=cell_col,
        null_data=null_data)

    reg_cos = create_dose_response_grid(
        all_data, sub_dir / "sr_bin_vs_cosine_dist_per_exp.png",
        y_col="cosine_dist", ylabel="Cosine Distance",
        subtitle=f"{mode_str}, {subsample_str}",
        bin_metric_label=metric_label, cell_col=cell_col,
        null_data=null_data)

    # 2. Regression summary (per-experiment mode only)
    if per_experiment:
        create_regression_summary(
            reg_d, sub_dir / "regression_summary_cohens_d.png",
            metric_label="Cohen's d", subtitle=subsample_str,
            bin_metric_label=metric_label)
        create_regression_summary(
            reg_cos, sub_dir / "regression_summary_cosine_dist.png",
            metric_label="Cosine Distance", subtitle=subsample_str,
            bin_metric_label=metric_label)
        for name, df in [("cohens_d", reg_d), ("cosine_dist", reg_cos)]:
            if df is not None and not df.empty:
                df.to_csv(sub_dir / f"regression_{name}.csv", index=False)

    # 3. Histograms
    create_effect_histograms(all_data, sub_dir / "effect_histograms_by_sr_bin.png",
                             subtitle=f"{mode_str}, {subsample_str}",
                             cell_col=cell_col,
                             bin_metric_label=metric_label,
                             null_data=null_data)

    # 4. Ridge plots
    create_ridge_by_sister_ratio(
        all_data, sub_dir / "ridge_cohens_d_by_sr_bin.png",
        metric="cohens_d", xlabel="Cohen's d",
        title_suffix=f"\n({mode_str}, {subsample_str})",
        bin_metric_label=metric_label, cell_col=cell_col,
        null_data=null_data)
    create_ridge_by_sister_ratio(
        all_data, sub_dir / "ridge_cosine_dist_by_sr_bin.png",
        metric="cosine_dist", xlabel="Cosine Distance",
        title_suffix=f"\n({mode_str}, {subsample_str})",
        bin_metric_label=metric_label, cell_col=cell_col,
        null_data=null_data)

    # 5. Per-experiment pooled dose-response (per-experiment mode with >1 exps)
    if per_experiment and all_data["experiment"].nunique() > 1:
        create_pooled_dose_response(all_data, sub_dir / "pooled_dose_response.png",
                                     subtitle=subsample_str,
                                     bin_metric_label=metric_label,
                                     cell_col=cell_col,
                                     null_data=null_data)

    # 6. Evo genes / null guides validation
    if variant == "null":
        # For null variant: show individual NTC guide trends
        create_null_guides_validation(
            all_data, native_data, sub_dir / "null_guides_validation.png",
            subtitle=f"{mode_str}, {subsample_str}",
            bin_metric_label=metric_label, cell_col=cell_col)
    elif evo_path and Path(evo_path).exists():
        create_evo_genes_validation(
            all_data, evo_path, sub_dir / "evo_genes_validation.png",
            subtitle=f"{mode_str}, {subsample_str}",
            bin_metric_label=metric_label, cell_col=cell_col)
    else:
        log.info("  Skipping evo_genes validation (file not found, use --evo-genes)")

    log.info(f"  All outputs in: {sub_dir}")


def run_analysis(args):
    """Execute the full sister-effect analysis. Called locally or as a SLURM job."""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover experiments
    if args.experiment:
        experiments = resolve_experiment(args.experiment)
    else:
        experiments = discover_experiments(exclude_bad=args.remove_bad)

    log.info(f"Found {len(experiments)} experiments with cell h5ad + spatial coherence")
    if not experiments:
        log.error("No experiments found")
        return

    n_workers = get_optimal_workers(use_gpu=False, data_ram_gb=3.0)
    exp_items = sorted(experiments.items())
    min_cells = args.min_cells
    mode = "per-experiment" if args.per_experiment else "pooled"
    log.info(f"Settings: mode={mode}, {N_SR_BINS} bins, min {min_cells} cells/guide/bin, "
             f"{n_workers} workers")

    # Resolve evo_genes path once
    evo_path = args.evo_genes
    if evo_path is None:
        candidates = [
            Path(__file__).resolve().parents[3] / "evo_genes.csv",
            Path(__file__).resolve().parents[3] / "cyclops_process" / "cyclops_process" / "configs" / "evo_genes.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                evo_path = str(candidate)
                break

    if args.per_experiment:
        # Per-experiment mode: process each experiment independently (only downsampled)
        for bin_metric in BIN_METRICS:
            log.info(f"\n{'='*60}")
            log.info(f"  bin_metric={bin_metric}  ({BIN_METRICS[bin_metric]})")
            log.info(f"{'='*60}")

            # Check cache for downsampled variant
            cache_path = (out_dir / "downsampled" / bin_metric /
                          f"sister_effect_merged__{mode}__downsampled__bins{N_SR_BINS}"
                          f"__mincells{min_cells}__{bin_metric}.csv")
            if not args.force and cache_path.exists():
                all_data = pd.read_csv(cache_path)
            else:
                results = Parallel(n_jobs=n_workers, backend="loky")(
                    delayed(process_experiment)(
                        eid, paths, min_cells_per_bin=min_cells, bin_metric=bin_metric)
                    for eid, paths in tqdm(exp_items, desc=f"Processing [{bin_metric}]")
                )
                all_data = pd.concat([r for r in results if r is not None], ignore_index=True)

            if all_data is None or all_data.empty:
                log.error(f"  No data after processing for bin_metric={bin_metric}")
                continue

            _run_one_metric_variant(
                bin_metric, "downsampled", all_data, exp_items, out_dir,
                mode, min_cells, n_workers, True, evo_path, args.force)
    else:
        # Pooled mode: 2-pass design for all metrics
        # Determine which metrics need recomputing (check both variants)
        metrics_needing_compute = []
        for m in BIN_METRICS:
            for v in VARIANTS:
                cache = (out_dir / v / m /
                         f"sister_effect_merged__{mode}__{v}__bins{N_SR_BINS}"
                         f"__mincells{min_cells}__{m}.csv")
                if args.force or not cache.exists():
                    if m not in metrics_needing_compute:
                        metrics_needing_compute.append(m)

        # Also check null variant caches
        for m in BIN_METRICS:
            null_cache = (out_dir / "null" / m /
                         f"sister_effect_merged__{mode}__null__bins{N_SR_BINS}"
                         f"__mincells{min_cells}__{m}.csv")
            if args.force or not null_cache.exists():
                if m not in metrics_needing_compute:
                    metrics_needing_compute.append(m)

        all_results = {}
        if metrics_needing_compute:
            log.info(f"Running 2-pass computation for metrics: {metrics_needing_compute}")
            n_workers_pooled = get_optimal_workers(use_gpu=False, data_ram_gb=2.0)

            # This runs Pass A + Pass B + finalization for all metrics at once
            all_results = compute_pooled_effects_all_metrics(
                exp_items,
                all_bin_metrics=metrics_needing_compute,
                n_sr_bins=N_SR_BINS,
                min_cells_per_bin=min_cells,
                n_workers=n_workers_pooled,
            )
        else:
            log.info("All metrics have cached results — running plots only")

        # Run plotting for all metrics (computed + cached)
        for m in BIN_METRICS:
            # Get null data for this metric (for overlay on perturbation plots)
            null_df = all_results.get(m, {}).get("null")
            if null_df is None:
                # Try loading from cache
                null_cache = (out_dir / "null" / m /
                             f"sister_effect_merged__{mode}__null__bins{N_SR_BINS}"
                             f"__mincells{min_cells}__{m}.csv")
                if null_cache.exists():
                    null_df = pd.read_csv(null_cache)

            for v in VARIANTS:
                is_computed = m in metrics_needing_compute
                log.info(f"\n{'='*60}")
                log.info(f"  {v}/{m}  ({BIN_METRICS.get(m, m)})"
                         f"{'' if is_computed else ' (cached)'}")
                log.info(f"{'='*60}")
                data = all_results.get(m, {}).get(v) if is_computed else None
                _run_one_metric_variant(
                    m, v, data, exp_items, out_dir,
                    mode, min_cells, n_workers, False, evo_path, args.force,
                    null_data=null_df)

            # Null variant plots
            null_data_for_plots = all_results.get(m, {}).get("null") if m in metrics_needing_compute else None
            native_df = all_results.get(m, {}).get("native")
            if native_df is None:
                # Try loading native from cache for reference
                native_cache = (out_dir / "native" / m /
                               f"sister_effect_merged__{mode}__native__bins{N_SR_BINS}"
                               f"__mincells{min_cells}__{m}.csv")
                if native_cache.exists():
                    native_df = pd.read_csv(native_cache)

            log.info(f"\n{'='*60}")
            log.info(f"  null/{m}  ({BIN_METRICS.get(m, m)})")
            log.info(f"{'='*60}")
            _run_one_metric_variant(
                m, "null", null_data_for_plots, exp_items, out_dir,
                mode, min_cells, n_workers, False, evo_path, args.force,
                native_data=native_df)

            # Regressed plots: subtract NTC null baseline from native data
            log.info(f"\n{'='*60}")
            log.info(f"  regressed/{m}  ({BIN_METRICS.get(m, m)})")
            log.info(f"{'='*60}")
            create_regressed_plots(
                native_df, null_df, out_dir, m,
                bin_metric_label=BIN_METRICS.get(m, m),
                evo_path=evo_path)

    log.info(f"\nAll outputs in: {out_dir}/{{native,downsampled,null,regressed}}/"
             f"{{sister_ratio,neighbor_count,sister_count}}/")


def main():
    parser = argparse.ArgumentParser(
        description="Cell-level sister effect analysis: does sister co-localization amplify phenotypes?")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-e", "--experiment", type=str,
                       help="Single experiment (e.g., 89, ops0089)")
    group.add_argument("--all", action="store_true",
                       help="Process all experiments with cell h5ad + spatial coherence")
    parser.add_argument("-o", "--output-dir", default="scripts/qc_phase2d_output/sister_effect",
                        help="Output directory")
    parser.add_argument("--remove-bad", action="store_true",
                        help="Exclude bad experiments (with --all)")
    parser.add_argument("--min-cells", type=int, default=MIN_CELLS_PER_BIN,
                        help=f"Min cells per (guide, bin) to keep a guide (default {MIN_CELLS_PER_BIN})")
    parser.add_argument("--per-experiment", action="store_true",
                        help="Process each experiment independently (old behavior). "
                             "Default pools cells across experiments for more guides.")
    parser.add_argument("--evo-genes", type=str, default=None,
                        help="Path to evo_genes.csv for validation plot")
    parser.add_argument("--force", action="store_true",
                        help="Ignore cached results and recompute from scratch")

    # SLURM submission options
    slurm_group = parser.add_argument_group("SLURM options")
    slurm_group.add_argument("--slurm", action="store_true",
                             help="Submit the full analysis as a single SLURM job")
    slurm_group.add_argument("--no-wait", action="store_true",
                             help="Don't wait for SLURM job to complete (with --slurm)")
    slurm_group.add_argument("--slurm-memory", type=str, default="400GB",
                             help="SLURM memory (default: 400GB)")
    slurm_group.add_argument("--slurm-time", type=int, default=15,
                             help="SLURM time limit in minutes (default: 15)")
    slurm_group.add_argument("--slurm-cpus", type=int, default=32,
                             help="SLURM CPUs (default: 32)")

    args = parser.parse_args()

    if args.slurm:
        import copy
        from cyclops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

        args_for_job = copy.copy(args)
        args_for_job.slurm = False

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        slurm_params = {
            "timeout_min": args.slurm_time,
            "mem": args.slurm_memory,
            "cpus_per_task": args.slurm_cpus,
            "slurm_partition": "cpu",
        }

        job_spec = {
            "name": "qc_sister_effect_analysis",
            "func": run_analysis,
            "kwargs": {"args": args_for_job},
        }

        log.info(f"Submitting to SLURM: mem={args.slurm_memory}, "
                 f"cpus={args.slurm_cpus}, time={args.slurm_time}min")

        result = submit_parallel_jobs(
            jobs_to_submit=[job_spec],
            experiment="qc_sister_effect",
            slurm_params=slurm_params,
            log_dir=str(out_dir / "slurm_logs"),
            manifest_prefix="sister_effect",
            wait_for_completion=not args.no_wait,
        )

        if result.get("success"):
            log.info(f"SLURM job completed: {result.get('base_job_id')}")
        else:
            log.error(f"SLURM job failed: {result}")
        return

    run_analysis(args)


if __name__ == "__main__":
    main()
