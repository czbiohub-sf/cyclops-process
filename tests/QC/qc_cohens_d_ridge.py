"""QC: Ridge plot of Cohen's d vs Cosine Distance across ALL Phase dino experiments.

Discovers every experiment with guide_bulked_Phase2D.h5ad, computes per-perturbation:
  - Cohen's d analog: RMS z-score distance from NTC centroid (normalized by NTC std)
  - Cosine distance: 1 - cosine_similarity(pert_centroid, NTC_centroid)

Produces side-by-side ridge plots showing Cohen's d captures experiment quality
differences while cosine distance does NOT.

Two canvases:
  1. Native: guide_bulked directly (fast)
  2. Downsampled: cell-level, all experiments downsampled to smallest (slow)

Usage:
    python scripts/qc_cohens_d_ridge.py                  # native only (fast)
    python scripts/qc_cohens_d_ridge.py --downsample      # also produce downsampled
"""

import argparse
import logging
from pathlib import Path

import anndata as ad
import h5py
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from tqdm import tqdm

from cyclops_utils.data.bad_experiments import DEFAULT_EXCLUDE_CATEGORIES, is_excluded

EXCLUDE_CATEGORIES = DEFAULT_EXCLUDE_CATEGORIES + ("need_rescue",)
from cyclops_utils.hpc.resource_manager import get_optimal_workers
from cyclops_process.paths import BASE_PATH

CP_CHALLENGE_CONTROL_SUMMARY = Path(
    f"{BASE_PATH}/ops0094_20251217/3-assembly/feature_extraction/"
    "graphs/2_guide_level/12_cp_challenge_dino/ntc_norm/cp_challenge_control_summary.csv"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

FEATURE_DIR = "dino_features_v1"
CHANNEL = "Phase2D"
STORAGE_ROOTS = [Path(BASE_PATH)]


# --- Discovery ---

def discover_experiments(exclude_bad=False):
    """Find all experiments with guide_bulked Phase dino h5ad files."""
    found = {}
    for root in STORAGE_ROOTS:
        if not root.exists():
            continue
        for exp_dir in sorted(root.glob("ops*")):
            exp_id = exp_dir.name.split("_")[0]
            if exp_id in found:
                continue
            if exclude_bad and is_excluded(exp_dir.name, categories=EXCLUDE_CATEGORIES):
                log.info(f"  Excluding bad experiment: {exp_dir.name}")
                continue
            for sub in [exp_dir / "3-assembly" / FEATURE_DIR / "anndata_objects",
                        exp_dir / "3-assembly" / "results" / "feature_extraction" / FEATURE_DIR / "anndata_objects"]:
                for fn in [f"guide_bulked_{CHANNEL}.h5ad", "guide_bulked_Phase.h5ad"]:
                    p = sub / fn
                    if p.exists() and exp_id not in found:
                        found[exp_id] = p
    return found


def discover_cell_level(exp_ids, exclude_bad=False):
    """Find features_processed paths for given experiment IDs."""
    found = {}
    for root in STORAGE_ROOTS:
        if not root.exists():
            continue
        for exp_dir in sorted(root.glob("ops*")):
            exp_id = exp_dir.name.split("_")[0]
            if exp_id not in exp_ids or exp_id in found:
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


# --- Loading ---

def load_guide_bulked(path):
    """Load guide_bulked h5ad, return (X, perturbation_labels, total_cells) or None."""
    a = ad.read_h5ad(path)
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    X = X.astype(np.float64)

    obs = a.obs.copy()
    if "label_str" in obs.columns and "perturbation" not in obs.columns:
        obs["perturbation"] = obs["label_str"]
    for c in ["perturbation"]:
        if c in obs.columns and hasattr(obs[c], "cat"):
            obs[c] = obs[c].astype(str)

    perts = obs["perturbation"].values if "perturbation" in obs.columns else None
    if perts is None:
        return None

    # Total cells across all guides
    total_cells = int(obs["n_cells"].sum()) if "n_cells" in obs.columns else None

    valid = pd.notna(perts) & (perts != "") & (perts != "None")
    return X[valid], perts[valid], total_cells


def load_cell_and_aggregate(path, target_cells, seed=42):
    """Load cell-level, downsample, aggregate to guide by mean."""
    a = ad.read_h5ad(path)
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    X = X.astype(np.float32)
    feat_cols = list(a.var_names)

    obs = a.obs.copy()
    for c in ["perturbation", "label_str", "sgRNA"]:
        if c in obs.columns and hasattr(obs[c], "cat"):
            obs[c] = obs[c].astype(str)
    if "perturbation" not in obs.columns and "label_str" in obs.columns:
        obs["perturbation"] = obs["label_str"]

    n = len(X)
    if n > target_cells:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, target_cells, replace=False)
        X = X[idx]
        obs = obs.iloc[idx].reset_index(drop=True)

    df = pd.DataFrame(X, columns=feat_cols)
    df["sgRNA"] = obs["sgRNA"].values if "sgRNA" in obs.columns else np.arange(len(df)).astype(str)
    df["perturbation"] = obs["perturbation"].values if "perturbation" in obs.columns else ""

    valid = df["sgRNA"].notna() & (df["sgRNA"] != "") & (df["sgRNA"] != "None")
    df = df[valid]

    guide_X = df.groupby("sgRNA", observed=True)[feat_cols].mean()
    guide_meta = df.groupby("sgRNA", observed=True)["perturbation"].first()
    guide_n = df.groupby("sgRNA", observed=True).size()

    keep = guide_n >= 3
    guide_X = guide_X[keep]
    guide_meta = guide_meta[keep]

    X_out = guide_X.values.astype(np.float64)
    perts = guide_meta.values
    valid_p = pd.notna(perts) & (perts != "") & (perts != "None")
    return X_out[valid_p], perts[valid_p]


# --- Metrics ---

def compute_both_metrics(X, perts):
    """Compute per-perturbation Cohen's d and cosine distance from NTC.

    Cosine distance uses the cyclops_model convention: L2-normalize each guide
    vector independently first, then compute centroids and similarity.
    (see cyclops_model/data/embeddings/cosine_similarity.py)

    Returns:
        cohens_d: dict {pert: d}
        cosine_dist: dict {pert: 1-cosine_sim}
        mean_d: float
        mean_cos: float
    """
    ntc_mask = np.isin(perts, ["NTC", "non-targeting"])
    X_ntc = X[ntc_mask]
    X_pert = X[~ntc_mask]
    perts_only = perts[~ntc_mask]

    if len(X_ntc) < 2:
        return {}, {}, 0.0, 0.0

    # Cohen's d uses raw features
    ntc_mean = X_ntc.mean(axis=0)
    ntc_std = X_ntc.std(axis=0, ddof=1)
    ntc_std[ntc_std == 0] = 1.0

    # Cosine uses L2-normalized guide vectors (cyclops_model convention)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_normed = X / norms

    X_ntc_normed = X_normed[ntc_mask]
    X_pert_normed = X_normed[~ntc_mask]

    # NTC centroid of normalized vectors, re-normalized
    ntc_centroid = X_ntc_normed.mean(axis=0)
    ntc_c_norm = np.linalg.norm(ntc_centroid)
    if ntc_c_norm > 0:
        ntc_centroid = ntc_centroid / ntc_c_norm

    cohens_d = {}
    cosine_dist = {}
    for p in np.unique(perts_only):
        idx = perts_only == p
        if idx.sum() < 2:
            continue
        p_mean = X_pert[idx].mean(axis=0)

        # Cohen's d: RMS of z-scored differences
        d = np.sqrt(np.mean(((p_mean - ntc_mean) / ntc_std) ** 2))
        cohens_d[p] = d

        # Cosine distance: centroid of L2-normalized vectors, re-normalized
        p_centroid = X_pert_normed[idx].mean(axis=0)
        p_c_norm = np.linalg.norm(p_centroid)
        if p_c_norm > 0:
            p_centroid = p_centroid / p_c_norm
            cos_sim = np.dot(p_centroid, ntc_centroid)
            cosine_dist[p] = 1.0 - cos_sim
        else:
            cosine_dist[p] = 0.0

    mean_d = float(np.mean(list(cohens_d.values()))) if cohens_d else 0.0
    mean_cos = float(np.mean(list(cosine_dist.values()))) if cosine_dist else 0.0
    return cohens_d, cosine_dist, mean_d, mean_cos


# --- Visualization ---

def _draw_ridge_on_axis(ax, exp_data, metric_key, mean_key, xlabel, title,
                        overlap=0.6, exp_order=None, clamp_x_min_zero=True):
    """Draw a single ridge plot on the given axis.

    exp_data values must have keys: metric_key (dict of per-pert values),
    mean_key (float), "n_guides" (int).
    exp_order: optional list of exp_ids in desired display order (bottom to top).
               If None, sorts by mean_key.
    clamp_x_min_zero: if True, x-axis minimum is clamped to 0.
    """
    if exp_order is not None:
        sorted_exps = [e for e in exp_order if e in exp_data]
    else:
        sorted_exps = sorted(exp_data.keys(), key=lambda e: exp_data[e][mean_key])
    n_experiments = len(sorted_exps)
    if n_experiments == 0:
        return

    # Collect all values for x-range
    all_vals = []
    for exp_id in sorted_exps:
        all_vals.extend(exp_data[exp_id][metric_key].values())
    all_vals = np.array(all_vals)
    global_std = np.std(all_vals)

    mean_ds = np.array([exp_data[e][mean_key] for e in sorted_exps])
    global_mean = mean_ds.mean()
    global_mean_std = mean_ds.std()

    row_height = 1.0
    # Use percentile range to focus on bulk of data, not extreme outliers
    p1, p99 = np.percentile(all_vals, [1, 97])
    p_range = p99 - p1
    x_min = p1 - 0.15 * max(p_range, 1e-6)
    if clamp_x_min_zero:
        x_min = max(0, x_min)
    x_max = p99 + 0.10 * max(p_range, 1e-6)
    x_range = np.linspace(x_min, x_max, 300)

    for i, exp_id in enumerate(sorted_exps):
        vals = np.array(list(exp_data[exp_id][metric_key].values()))
        if len(vals) == 0:
            continue

        mean_val = exp_data[exp_id][mean_key]

        is_outlier = False
        if global_mean_std > 0:
            z = (mean_val - global_mean) / global_mean_std
            is_outlier = abs(z) > 2.5

        y_offset = i * row_height * (1 - overlap)

        try:
            kde = stats.gaussian_kde(vals, bw_method=0.15)
            density = kde(x_range)
            density = density / density.max() * row_height * 0.8
        except Exception:
            continue

        color = "red" if is_outlier else "steelblue"
        alpha = 0.7 if is_outlier else 0.35
        lw = 1.5 if is_outlier else 0.8

        ax.fill_between(x_range, y_offset, y_offset + density,
                        color=color, alpha=alpha, linewidth=0)
        ax.plot(x_range, y_offset + density,
                color=color, alpha=min(1, alpha + 0.3), linewidth=lw)

        label_color = "red" if is_outlier else "black"
        label_weight = "bold" if is_outlier else "normal"
        exp_label = f"{exp_id} ({mean_val:.4f})"
        ax.text(x_min - 0.02 * (x_max - x_min), y_offset + row_height * 0.1,
                exp_label, fontsize=12, ha="right", va="center",
                color=label_color, fontweight=label_weight)

    ax.axvline(global_mean, color="green", linestyle="-", lw=2, alpha=0.7,
               label=f"Mean: {global_mean:.4f}")
    if global_mean_std > 0:
        ax.axvline(global_mean - 2.5 * global_mean_std, color="orange",
                   linestyle="--", lw=1.5, alpha=0.7, label="-2.5\u03c3")
        ax.axvline(global_mean + 2.5 * global_mean_std, color="orange",
                   linestyle="--", lw=1.5, alpha=0.7, label="+2.5\u03c3")

    ax.set_xlabel(xlabel, fontsize=21)
    ax.set_title(title, fontsize=21, fontweight="bold")
    ax.legend(loc="upper right", fontsize=15)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=16)
    ax.set_xlim(x_min - 0.25 * (x_max - x_min), x_max)


def _ops_number_order(exp_data):
    """Return experiment IDs sorted by their numeric ops number (e.g. ops0052 → 52)."""
    import re
    def _num(e):
        m = re.search(r"ops(\d+)", e)
        return int(m.group(1)) if m else 0
    return sorted(exp_data.keys(), key=_num)


def create_dual_ridge_plot(exp_data, output_path, title_suffix="", overlap=0.6,
                           sort_by="metric"):
    """Create side-by-side ridge plots: Cohen's d (left) and Cosine Distance (right).

    sort_by: "metric" (each panel sorted by its own metric, default) or
             "ops_number" (both panels use the same ops-number order).
    """
    n_experiments = len(exp_data)
    if n_experiments == 0:
        return

    if sort_by == "ops_number":
        exp_order = _ops_number_order(exp_data)
        sort_label = "sorted by ops number"
    else:
        exp_order = None  # each panel sorts independently by its metric
        sort_label = "sorted independently by each metric"

    fig_height = max(12, n_experiments * 0.35)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(22, fig_height))

    _draw_ridge_on_axis(
        ax_left, exp_data,
        metric_key="cohens_d", mean_key="mean_d",
        xlabel="Cohen's d analog (RMS z-score distance from NTC)",
        title=f"Cohen's d{title_suffix}",
        overlap=overlap, exp_order=exp_order, clamp_x_min_zero=True,
    )

    _draw_ridge_on_axis(
        ax_right, exp_data,
        metric_key="cosine_dist", mean_key="mean_cos",
        xlabel="Cosine Distance (1 - cos_sim to NTC centroid)",
        title=f"Cosine Distance{title_suffix}",
        overlap=overlap, exp_order=exp_order, clamp_x_min_zero=False,
    )

    fig.suptitle(
        f"Phase2D Dino Features: Cohen's d vs Cosine Distance{title_suffix}\n"
        f"({n_experiments} experiments — {sort_label}, red = outlier)",
        fontsize=24, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {output_path}")


def create_pairwise_correlation_heatmaps(exp_data, output_path, title_suffix=""):
    """Create side-by-side heatmaps of pairwise experiment correlation.

    For each pair of experiments, finds shared perturbations and computes
    Pearson correlation of their per-perturbation metric vectors.
    Left heatmap: Cohen's d profiles. Right heatmap: cosine distance profiles.
    """
    exps = sorted(exp_data.keys(), key=lambda e: int(
        __import__("re").search(r"ops(\d+)", e).group(1)
        if __import__("re").search(r"ops(\d+)", e) else 0
    ))
    n = len(exps)
    if n < 3:
        log.warning("Not enough experiments for correlation heatmap")
        return

    corr_d = np.full((n, n), np.nan)
    corr_cos = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(n):
            d_i = exp_data[exps[i]]["cohens_d"]
            d_j = exp_data[exps[j]]["cohens_d"]
            cos_i = exp_data[exps[i]]["cosine_dist"]
            cos_j = exp_data[exps[j]]["cosine_dist"]

            # Shared perturbations for Cohen's d
            shared_d = sorted(set(d_i.keys()) & set(d_j.keys()))
            if len(shared_d) >= 5:
                vi = np.array([d_i[p] for p in shared_d])
                vj = np.array([d_j[p] for p in shared_d])
                if vi.std() > 0 and vj.std() > 0:
                    corr_d[i, j], _ = stats.pearsonr(vi, vj)

            # Shared perturbations for cosine distance
            shared_cos = sorted(set(cos_i.keys()) & set(cos_j.keys()))
            if len(shared_cos) >= 5:
                vi = np.array([cos_i[p] for p in shared_cos])
                vj = np.array([cos_j[p] for p in shared_cos])
                if vi.std() > 0 and vj.std() > 0:
                    corr_cos[i, j], _ = stats.pearsonr(vi, vj)

    # Shared color scale across both heatmaps
    all_valid = np.concatenate([
        corr_d[~np.isnan(corr_d)], corr_cos[~np.isnan(corr_cos)]
    ])
    vmin, vmax = (all_valid.min(), all_valid.max()) if len(all_valid) > 0 else (0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(26, 10),
                             gridspec_kw={"wspace": 0.35})

    for ax, mat, metric_name in [
        (axes[0], corr_d, "Cohen's d"),
        (axes[1], corr_cos, "Cosine Distance"),
    ]:
        im = ax.imshow(mat, cmap="inferno_r", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(exps, rotation=90, fontsize=9)
        ax.set_yticklabels(exps, fontsize=9)
        ax.set_title(f"{metric_name} Profile Correlation{title_suffix}",
                     fontsize=14, fontweight="bold")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Pearson r (darker = lower)", fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        f"Pairwise Experiment Correlation of Per-Perturbation Profiles{title_suffix}",
        fontsize=16, fontweight="bold", y=1.02,
    )
    fig.subplots_adjust(top=0.90)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {output_path}")


# --- Plate stats + mAP ---

def load_plate_stats(exp_id):
    """Load plate_stats.csv for an experiment, return dict of ALL numeric metrics.

    All metrics are averaged across plates — Pearson r is invariant to
    sum vs mean when the number of plates is constant across experiments.
    """
    for root in STORAGE_ROOTS:
        for exp_dir in root.glob(f"{exp_id}*"):
            p = exp_dir / "3-assembly" / "ISS" / "mine" / "plate_stats.csv"
            if p.exists():
                df = pd.read_csv(p, index_col=0)
                result = {}
                for metric in df.index:
                    vals = pd.to_numeric(df.loc[metric], errors="coerce").dropna()
                    if len(vals) == 0:
                        continue
                    result[metric] = float(vals.mean())
                return result
    return None


def compute_hopkins_statistic(coords, n_samples=None, seed=42):
    """Hopkins statistic for spatial clustering vs complete spatial randomness.

    H ~ 0.5 → random, H → 1.0 → clustered, H → 0.0 → uniform/regular.
    """
    rng = np.random.RandomState(seed)
    n = len(coords)
    if n < 10:
        return np.nan
    if n_samples is None:
        n_samples = min(100, max(10, n // 10))

    # Sample n_samples data points
    idx = rng.choice(n, size=n_samples, replace=False)
    sampled = coords[idx]

    # Build tree on ALL points, query nearest neighbor for sampled (excluding self)
    tree = cKDTree(coords)
    data_dists, _ = tree.query(sampled, k=2)  # k=2: [self=0, nearest neighbor]
    data_nn = data_dists[:, 1]  # skip self

    # Generate random points in bounding box
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    random_pts = rng.uniform(mins, maxs, size=(n_samples, coords.shape[1]))
    random_dists, _ = tree.query(random_pts, k=1)

    # Hopkins statistic
    sum_random = (random_dists ** 2).sum()
    sum_data = (data_nn ** 2).sum()
    if sum_random + sum_data == 0:
        return np.nan
    return float(sum_random / (sum_random + sum_data))


def load_experiment_hopkins(exp_id):
    """Compute Hopkins statistic for an experiment from cell coordinates.

    Tries spatial_coherence cache first, falls back to linked_pheno_iss CSVs.
    Returns mean Hopkins across wells, or None if data unavailable.
    """
    # Find experiment directory
    exp_dir = None
    for root in STORAGE_ROOTS:
        for d in root.glob(f"{exp_id}*"):
            exp_dir = d
            break
        if exp_dir:
            break
    if exp_dir is None:
        return None

    assembly = exp_dir / "3-assembly"
    hopkins_values = []

    # Try spatial_coherence cache (has y_pheno, x_pheno already)
    cache_path = assembly / "spatial_coherence" / "per_cell_spatial_coherence.csv"
    if cache_path.exists():
        try:
            df = pd.read_csv(cache_path, low_memory=False)
            if "y_pheno" in df.columns and "x_pheno" in df.columns:
                for well, wdf in df.groupby("well"):
                    coords = wdf[["y_pheno", "x_pheno"]].dropna().values
                    if len(coords) >= 10:
                        hopkins_values.append(compute_hopkins_statistic(coords))
                if hopkins_values:
                    return float(np.nanmean(hopkins_values))
        except Exception as e:
            log.debug(f"  {exp_id}: cache read failed: {e}")

    # Fall back to linked_pheno_iss CSVs
    for p in sorted(assembly.glob("*_linked_pheno_iss.csv")):
        try:
            df = pd.read_csv(p, usecols=["y_pheno", "x_pheno"])
            coords = df.dropna().values
            if len(coords) >= 10:
                hopkins_values.append(compute_hopkins_statistic(coords))
        except Exception:
            continue

    # Also try ISS/mine subdirectory
    if not hopkins_values:
        for sub in ["ISS/mine", "ISS/prob"]:
            for p in sorted((assembly / sub).glob("*_linked_pheno_iss.csv")) if (assembly / sub).exists() else []:
                try:
                    df = pd.read_csv(p, usecols=["y_pheno", "x_pheno"])
                    coords = df.dropna().values
                    if len(coords) >= 10:
                        hopkins_values.append(compute_hopkins_statistic(coords))
                except Exception:
                    continue

    if hopkins_values:
        return float(np.nanmean(hopkins_values))
    return None


def load_precomputed_active_ratios():
    """Load precomputed mAP active ratios from CP challenge summary CSV.

    Returns dict {exp_id: mean_active_ratio} averaged across organelle comparisons.
    """
    if not CP_CHALLENGE_CONTROL_SUMMARY.exists():
        log.warning(f"CP challenge control summary not found: {CP_CHALLENGE_CONTROL_SUMMARY}")
        return {}
    df = pd.read_csv(CP_CHALLENGE_CONTROL_SUMMARY)
    # One row per live-cell experiment, all Phase2D control comparisons
    df["live_id"] = df["live_experiment"].str.split("_").str[0]
    ratios = dict(zip(df["live_id"], df["live_active_ratio"]))
    # Also include CP experiment (ops0094) Phase2D active ratio (same across rows, take first)
    cp_id = df["cp_experiment"].iloc[0].split("_")[0]
    ratios[cp_id] = df["cp_active_ratio"].iloc[0]
    log.info(f"Loaded Phase2D active ratios for {len(ratios)} experiments from control summary")
    return ratios


GENE_EFFECT_CSV = Path(
    f"{BASE_PATH}/configs/library/twist1k_pool_CERES.csv"
)

_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "cyclops_process" / "configs"

RELG4_CSV = _CONFIGS_DIR / "df_tau_relg4.csv"
CHANNEL_MAPS_YAML = _CONFIGS_DIR / "ops_channel_maps.yaml"

HIGHLIGHT_EXPS = {"ops0094", "ops0052"}


def _draw_scatter(ax, x, y, labels, xlabel, ylabel, title):
    """Draw a single scatter with regression line and Pearson r."""
    highlight = np.array([lab in HIGHLIGHT_EXPS for lab in labels])
    # Normal points
    if (~highlight).any():
        ax.scatter(x[~highlight], y[~highlight], s=40, alpha=0.7,
                   c="#1565C0", edgecolors="white", lw=0.5)
    # Highlighted points
    if highlight.any():
        ax.scatter(x[highlight], y[highlight], s=80, alpha=0.9,
                   c="red", edgecolors="black", lw=1.0, zorder=5)
    for xi, yi, lab in zip(x, y, labels):
        weight = "bold" if lab in HIGHLIGHT_EXPS else "normal"
        color = "red" if lab in HIGHLIGHT_EXPS else "black"
        fontsize = 7.5 if lab in HIGHLIGHT_EXPS else 6
        ax.annotate(lab, (xi, yi), fontsize=fontsize, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points", alpha=0.85,
                    fontweight=weight, color=color)
    if len(x) >= 3:
        r, p_val = stats.pearsonr(x, y)
        rho, rho_p = stats.spearmanr(x, y)
        slope, intercept = np.polyfit(x, y, 1)
        x_fit = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_fit, slope * x_fit + intercept, "r--", lw=1.5, alpha=0.6)
        ax.text(0.05, 0.95,
                f"Pearson r={r:.3f}, p={p_val:.2e}\n"
                f"Spearman ρ={rho:.3f}, p={rho_p:.2e}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def create_correlation_scatters(exp_data, out_dir, filename="cohens_d_correlations.png",
                                title_suffix=""):
    """Create paired scatter plots: Cohen's d (left col) and Cosine Dist (right col) vs QC metrics."""
    has_cos = any("mean_cos" in exp_data[e] for e in exp_data)

    # Load precomputed mAP active ratios from CP challenge summary
    active_ratios = load_precomputed_active_ratios()
    map_results = {e: {"active_ratio": r} for e, r in active_ratios.items() if e in exp_data}

    # Load plate stats for ALL experiments
    log.info("Loading plate_stats for all experiments...")
    plate_stats = {}
    for exp_id in sorted(exp_data.keys()):
        ps = load_plate_stats(exp_id)
        if ps:
            plate_stats[exp_id] = ps
    log.info(f"  Plate stats found for {len(plate_stats)} / {len(exp_data)} experiments")

    # Build rows: each row is (ylabel, title_desc, experiment_list, y_values)
    row_specs = []

    # 1. mAP active ratio (CP challenge experiments only)
    if map_results:
        common = sorted(set(map_results.keys()) & set(exp_data.keys()))
        if len(common) >= 3:
            row_specs.append((
                "mAP Active Ratio", "mAP Active Ratio (CP challenge)",
                common, [map_results[e]["active_ratio"] for e in common],
            ))

    # 2. Cell count from anndata
    exps_with_cells = sorted(e for e in exp_data if "n_cells" in exp_data[e])
    if len(exps_with_cells) >= 3:
        row_specs.append((
            "Total Cells (anndata)", "Cell Count in AnnData",
            exps_with_cells, [exp_data[e]["n_cells"] for e in exps_with_cells],
        ))

    # 3-5. Key plate stats
    for metric, ylabel, title_desc in [
        ("num_cells", "Total Cells (plates)", "Total Cell Count (plates)"),
        ("percent_cells_with_matched_reads", "% Cells Matched Reads", "ISS Read Matching Rate"),
        ("snr_mean_snr", "ISS Mean SNR", "ISS Signal-to-Noise Ratio"),
    ]:
        common = sorted(e for e in exp_data if e in plate_stats and metric in plate_stats[e])
        if len(common) >= 3:
            row_specs.append((
                ylabel, title_desc,
                common, [plate_stats[e][metric] for e in common],
            ))

    if not row_specs:
        log.warning("No scatter plots to create (insufficient data)")
        return

    nrows = len(row_specs)
    ncols = 2 if has_cos else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows),
                             squeeze=False)

    for i, (ylabel, title_desc, exps, y_vals) in enumerate(row_specs):
        x_d = np.array([exp_data[e]["mean_d"] for e in exps])
        y = np.array(y_vals)

        _draw_scatter(axes[i, 0], x_d, y, exps,
                      "Mean Cohen's d", ylabel,
                      f"Cohen's d vs {title_desc}{title_suffix}")

        if has_cos:
            # Only include experiments that have cosine data
            cos_exps = [e for e in exps if "mean_cos" in exp_data[e]]
            if len(cos_exps) >= 3:
                x_cos = np.array([exp_data[e]["mean_cos"] for e in cos_exps])
                y_cos = np.array([y_vals[exps.index(e)] for e in cos_exps])
                _draw_scatter(axes[i, 1], x_cos, y_cos, cos_exps,
                              "Mean Cosine Distance", ylabel,
                              f"Cosine Dist vs {title_desc}{title_suffix}")
            else:
                axes[i, 1].set_visible(False)

    fig.suptitle(f"Phase2D: Cohen's d vs Cosine Distance Correlations{title_suffix}",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {out_path}")


def create_qc_metric_ranking(exp_data, output_path, title_suffix=""):
    """Succinct summary: correlate ALL plate_stats metrics with Cohen's d & cosine distance.

    Produces a single figure with:
      Top: heatmap of Pearson r (metrics × [Cohen's d, Cosine dist]), sorted by |r|
      Bottom: paired horizontal bar chart of r values
    Also saves CSV with r, p, slope, n for every metric.
    """
    # Load plate stats for all experiments
    plate_stats = {}
    for exp_id in sorted(exp_data.keys()):
        ps = load_plate_stats(exp_id)
        if ps:
            plate_stats[exp_id] = ps

    # Compute Hopkins statistic per experiment and inject into plate_stats
    log.info("Computing Hopkins statistic for confluency...")
    exp_ids_sorted = sorted(exp_data.keys())
    hopkins_results = Parallel(n_jobs=-1, backend="loky")(
        delayed(load_experiment_hopkins)(eid)
        for eid in tqdm(exp_ids_sorted, desc="Hopkins statistic")
    )
    n_hopkins = 0
    for exp_id, h in zip(exp_ids_sorted, hopkins_results):
        if h is not None:
            plate_stats.setdefault(exp_id, {})["hopkins_statistic"] = h
            n_hopkins += 1
    log.info(f"  Hopkins computed for {n_hopkins}/{len(exp_data)} experiments")

    if len(plate_stats) < 3:
        log.warning("Not enough experiments with plate_stats for QC ranking")
        return

    has_cos = any("mean_cos" in exp_data[e] for e in exp_data)

    # Collect all metric names
    all_metrics = set()
    for ps in plate_stats.values():
        all_metrics.update(ps.keys())

    rows = []
    for metric in sorted(all_metrics):
        common = sorted(e for e in exp_data if e in plate_stats and metric in plate_stats[e])
        if len(common) < 3:
            continue
        y = np.array([plate_stats[e][metric] for e in common])
        x_d = np.array([exp_data[e]["mean_d"] for e in common])

        r_d, p_d = stats.pearsonr(x_d, y)
        rho_d, rho_p_d = stats.spearmanr(x_d, y)
        slope_d = np.polyfit(x_d, y, 1)[0] if np.std(x_d) > 0 else np.nan
        row = {"metric": metric, "n": len(common),
               "pearson_r_cohens_d": r_d, "pearson_p_cohens_d": p_d,
               "spearman_r_cohens_d": rho_d, "spearman_p_cohens_d": rho_p_d,
               "slope_cohens_d": slope_d,
               "pearson_r_cosine_dist": np.nan, "pearson_p_cosine_dist": np.nan,
               "spearman_r_cosine_dist": np.nan, "spearman_p_cosine_dist": np.nan,
               "slope_cosine_dist": np.nan}

        if has_cos:
            cos_common = [e for e in common if "mean_cos" in exp_data[e]]
            if len(cos_common) >= 3:
                x_cos = np.array([exp_data[e]["mean_cos"] for e in cos_common])
                y_cos = np.array([plate_stats[e][metric] for e in cos_common])
                r_cos, p_cos = stats.pearsonr(x_cos, y_cos)
                rho_cos, rho_p_cos = stats.spearmanr(x_cos, y_cos)
                slope_cos = np.polyfit(x_cos, y_cos, 1)[0] if np.std(x_cos) > 0 else np.nan
                row["pearson_r_cosine_dist"] = r_cos
                row["pearson_p_cosine_dist"] = p_cos
                row["spearman_r_cosine_dist"] = rho_cos
                row["spearman_p_cosine_dist"] = rho_p_cos
                row["slope_cosine_dist"] = slope_cos
        rows.append(row)

    if not rows:
        log.warning("No metrics with enough experiments for ranking")
        return

    df = pd.DataFrame(rows)

    # Save CSV (sorted by |pearson_r_cohens_d|)
    csv_path = output_path.with_suffix(".csv")
    df.sort_values("pearson_r_cohens_d", key=lambda s: s.abs(), ascending=False).to_csv(csv_path, index=False)
    log.info(f"Saved QC metric ranking CSV: {csv_path} ({len(df)} metrics)")

    # --- One figure per (corr_method × effect_metric), each sorted independently ---
    # 4 canvases: Pearson×Cohen's d, Pearson×Cosine, Spearman×Cohen's d, Spearman×Cosine
    panel_specs = [
        ("pearson_r_cohens_d", "pearson_p_cohens_d", "Pearson r", "Cohen's d", "#1565C0"),
        ("spearman_r_cohens_d", "spearman_p_cohens_d", "Spearman rho", "Cohen's d", "#1565C0"),
    ]
    if has_cos and df["pearson_r_cosine_dist"].notna().any():
        panel_specs += [
            ("pearson_r_cosine_dist", "pearson_p_cosine_dist", "Pearson r", "Cosine Distance", "#E65100"),
            ("spearman_r_cosine_dist", "spearman_p_cosine_dist", "Spearman rho", "Cosine Distance", "#E65100"),
        ]

    stem = output_path.stem  # e.g. "qc_metric_ranking_native"
    for r_col, p_col, corr_label, effect_label, color in panel_specs:
        df_panel = df.dropna(subset=[r_col]).copy()
        df_panel["abs_r"] = df_panel[r_col].abs()
        df_panel = df_panel.sort_values("abs_r", ascending=True).reset_index(drop=True)
        n = len(df_panel)
        fig_height = max(8, n * 0.28 + 2)

        fig, (ax_heat, ax_bar) = plt.subplots(
            1, 2, figsize=(12, fig_height),
            gridspec_kw={"width_ratios": [0.6, 2], "wspace": 0.02},
        )

        # Heatmap: single column of r/rho values
        r_vals = df_panel[r_col].values[:, np.newaxis]
        ax_heat.imshow(r_vals, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax_heat.set_xticks([0])
        ax_heat.set_xticklabels([corr_label], fontsize=8)
        ax_heat.set_yticks(range(n))
        ax_heat.set_yticklabels(df_panel["metric"].values, fontsize=6.5)
        for i in range(n):
            r_val = df_panel.iloc[i][r_col]
            p_val = df_panel.iloc[i][p_col]
            star = "*" if p_val < 0.05 else ""
            txt_color = "white" if abs(r_val) > 0.6 else "black"
            ax_heat.text(0, i, f"{r_val:.2f}{star}", ha="center", va="center",
                         fontsize=6, color=txt_color,
                         fontweight="bold" if star else "normal")

        # Bar chart
        y_pos = np.arange(n)
        vals = df_panel[r_col].values
        colors = [color if v >= 0 else "#C62828" for v in vals]
        ax_bar.barh(y_pos, vals, color=colors, edgecolor="white", linewidth=0.3, height=0.7)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels([])
        ax_bar.axvline(0, color="black", lw=0.8)
        ax_bar.set_xlabel(f"{corr_label} with mean {effect_label}", fontsize=10)
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        ax_bar.set_ylim(-0.5, n - 0.5)
        ax_heat.set_ylim(-0.5, n - 0.5)
        ax_heat.invert_yaxis()
        ax_bar.invert_yaxis()

        fig.suptitle(
            f"QC Metrics Ranked by {corr_label} with {effect_label}{title_suffix}\n"
            f"(* = p < 0.05, {len(plate_stats)} experiments)",
            fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        corr_suffix = corr_label.lower().replace(" ", "_")
        effect_suffix = effect_label.lower().replace("'", "").replace(" ", "_")
        fig_path = output_path.with_name(f"{stem}_{corr_suffix}_{effect_suffix}.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"Saved: {fig_path}")

    return df, plate_stats


def create_top_qc_scatters(exp_data, ranking_df, plate_stats, output_path,
                            top_n=5, title_suffix=""):
    """Scatter plots for the top-N most positively and negatively correlated QC metrics.

    20 panels total: 5 rows × 4 columns.
      Col 0: top-5 positive r with Cohen's d
      Col 1: top-5 negative r with Cohen's d
      Col 2: top-5 positive r with Cosine dist
      Col 3: top-5 negative r with Cosine dist
    Uses the same _draw_scatter style (highlighted ops0052/ops0094, regression line, labels).
    """
    valid_d = ranking_df.dropna(subset=["pearson_r_cohens_d"])
    top_pos_d = valid_d.nlargest(top_n, "pearson_r_cohens_d")["metric"].tolist()
    top_neg_d = valid_d.nsmallest(top_n, "pearson_r_cohens_d")["metric"].tolist()

    has_cos = "pearson_r_cosine_dist" in ranking_df.columns
    if has_cos:
        valid_cos = ranking_df.dropna(subset=["pearson_r_cosine_dist"])
        top_pos_cos = valid_cos.nlargest(top_n, "pearson_r_cosine_dist")["metric"].tolist()
        top_neg_cos = valid_cos.nsmallest(top_n, "pearson_r_cosine_dist")["metric"].tolist()
    else:
        top_pos_cos, top_neg_cos = [], []

    ncols = 4 if has_cos else 2
    fig, axes = plt.subplots(top_n, ncols, figsize=(7 * ncols, 5 * top_n), squeeze=False)

    col_specs = [
        (top_pos_d, "mean_d", "Cohen's d", "Top + r (Cohen's d)"),
        (top_neg_d, "mean_d", "Cohen's d", "Top - r (Cohen's d)"),
    ]
    if has_cos:
        col_specs += [
            (top_pos_cos, "mean_cos", "Cosine Distance", "Top + r (Cosine dist)"),
            (top_neg_cos, "mean_cos", "Cosine Distance", "Top - r (Cosine dist)"),
        ]

    for col_idx, (metrics_list, x_key, x_label, col_title) in enumerate(col_specs):
        axes[0, col_idx].set_title(col_title, fontsize=11, fontweight="bold", pad=12)
        for row_idx in range(top_n):
            ax = axes[row_idx, col_idx]
            if row_idx >= len(metrics_list):
                ax.set_visible(False)
                continue
            metric = metrics_list[row_idx]
            common = sorted(
                e for e in exp_data
                if e in plate_stats and metric in plate_stats[e] and x_key in exp_data[e]
            )
            if len(common) < 3:
                ax.set_visible(False)
                continue

            x = np.array([exp_data[e][x_key] for e in common])
            y = np.array([plate_stats[e][metric] for e in common])
            ylabel = metric.replace("_", " ")

            _draw_scatter(ax, x, y, common,
                          f"Mean {x_label}", ylabel,
                          f"{x_label} vs {ylabel}")

    fig.suptitle(f"Top {top_n} Most Positive & Negative QC Correlations{title_suffix}",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {output_path}")


def create_confluency_control_analysis(exp_data, plate_stats, output_path,
                                        title_suffix=""):
    """Test whether sister-guide spatial effect is confounded by confluency.

    Uses sc_neighbor_count_mean as the confluency proxy (local cell density)
    rather than Hopkins (which saturates near 1.0 for cell imaging data).

    For each effect metric (Cohen's d, Cosine dist), 4 columns:
      1. Neighbor count (density) vs effect size — does confluency predict effect?
      2. Sister ratio (mean) vs effect size — raw correlation
      3. Sister ratio (median) vs effect size — raw correlation
      4. Partial: sister ratio vs effect size, controlling for neighbor count

    Also includes Hopkins vs num_cells validation row.
    """
    density_key = "sc_neighbor_count_mean"
    sister_mean_key = "sc_sister_ratio_mean"
    sister_med_key = "sc_sister_ratio_median"
    hopkins_key = "hopkins_statistic"
    num_cells_key = "num_cells"

    # Experiments with density + sister ratio
    common = sorted(
        e for e in exp_data
        if e in plate_stats
        and density_key in plate_stats[e]
        and sister_mean_key in plate_stats[e]
        and sister_med_key in plate_stats[e]
    )
    if len(common) < 4:
        log.warning(f"Only {len(common)} experiments with density + sister_ratio — need >=4")
        return

    has_cos = any("mean_cos" in exp_data[e] for e in common)

    # --- Build row specs: (effect_key, effect_label, sister_key, sister_label) ---
    effect_metrics = [("mean_d", "Mean Cohen's d")]
    if has_cos:
        effect_metrics.append(("mean_cos", "Mean Cosine Distance"))

    # Each effect metric gets 4 columns across 1 row
    ncols = 4
    nrows = len(effect_metrics)

    # Add Hopkins validation row if Hopkins data exists
    has_hopkins = any(
        hopkins_key in plate_stats.get(e, {}) and num_cells_key in plate_stats.get(e, {})
        for e in common
    )
    if has_hopkins:
        nrows += 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.5 * nrows), squeeze=False)

    for row, (eff_key, eff_label) in enumerate(effect_metrics):
        row_common = [e for e in common if eff_key in exp_data[e]]
        if len(row_common) < 4:
            for j in range(ncols):
                axes[row, j].set_visible(False)
            continue

        effect = np.array([exp_data[e][eff_key] for e in row_common])
        density = np.array([plate_stats[e][density_key] for e in row_common])
        s_mean = np.array([plate_stats[e][sister_mean_key] for e in row_common])
        s_med = np.array([plate_stats[e][sister_med_key] for e in row_common])
        labels = row_common

        # Col 0: density (neighbor count) vs effect size
        _draw_scatter(axes[row, 0], effect, density, labels,
                      eff_label, "Neighbor Count (mean)",
                      f"Cell Density vs {eff_label}")

        # Col 1: sister ratio mean vs effect size
        _draw_scatter(axes[row, 1], effect, s_mean, labels,
                      eff_label, "Sister Ratio (mean)",
                      f"Sister Ratio Mean vs {eff_label}")

        # Col 2: sister ratio median vs effect size
        _draw_scatter(axes[row, 2], effect, s_med, labels,
                      eff_label, "Sister Ratio (median)",
                      f"Sister Ratio Median vs {eff_label}")

        # Col 3: partial — sister ratio mean vs effect, controlling for density
        if np.std(density) > 0:
            sl_s, int_s = np.polyfit(density, s_mean, 1)
            sl_e, int_e = np.polyfit(density, effect, 1)
            s_resid = s_mean - (sl_s * density + int_s)
            e_resid = effect - (sl_e * density + int_e)
        else:
            s_resid, e_resid = s_mean, effect

        _draw_scatter(axes[row, 3], e_resid, s_resid, labels,
                      f"{eff_label}\n(density-residual)",
                      "Sister Ratio (density-residual)",
                      f"Partial: Sister Ratio vs {eff_label}\n(controlling for neighbor count)")

    # Hopkins validation row: Hopkins vs num_cells, Hopkins vs density
    if has_hopkins:
        hop_row = nrows - 1
        hop_common = sorted(
            e for e in common
            if hopkins_key in plate_stats.get(e, {})
            and num_cells_key in plate_stats.get(e, {})
        )
        if len(hop_common) >= 3:
            h = np.array([plate_stats[e][hopkins_key] for e in hop_common])
            nc = np.array([plate_stats[e][num_cells_key] for e in hop_common])
            dens = np.array([plate_stats[e][density_key] for e in hop_common])

            _draw_scatter(axes[hop_row, 0], nc, h, hop_common,
                          "Num Cells", "Hopkins Statistic",
                          "Hopkins vs Cell Count (validation)")
            _draw_scatter(axes[hop_row, 1], dens, h, hop_common,
                          "Neighbor Count (mean)", "Hopkins Statistic",
                          "Hopkins vs Neighbor Density (validation)")

            # Also: density vs num_cells
            _draw_scatter(axes[hop_row, 2], nc, dens, hop_common,
                          "Num Cells", "Neighbor Count (mean)",
                          "Cell Density vs Cell Count")
        # Hide unused panel
        for j in range(3 if len(hop_common) >= 3 else 0, ncols):
            axes[hop_row, j].set_visible(False)

    fig.suptitle(
        f"Confluency Control: Is Sister-Guide Effect Independent of Cell Density?{title_suffix}\n"
        f"({len(common)} experiments, density = sc_neighbor_count_mean)",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {output_path}")


# --- Gene effect (CERES) correlation ---

def load_gene_effect_map():
    """Load gene-level CERES scores from twist1k_pool_CERES.csv.

    Returns dict {gene_name: gene_effect} using the 'Gene name' column
    which matches perturbation names in guide_bulked h5ad files.
    """
    if not GENE_EFFECT_CSV.exists():
        log.warning(f"Gene effect CSV not found: {GENE_EFFECT_CSV}")
        return {}
    df = pd.read_csv(GENE_EFFECT_CSV)
    # 'Gene name' gives 100% match with perturbation names; 'dep_map_gene_name' ~97%
    gene_effect = df.groupby("Gene name")["gene_effect"].first().to_dict()
    # Remove NTC entries (NCBI_ID == -1 → no gene name mapping)
    ntc_genes = set(df.loc[df["NCBI_ID"] == -1, "Gene name"].unique())
    for g in ntc_genes:
        gene_effect.pop(g, None)
    log.info(f"Loaded gene effect scores for {len(gene_effect)} genes")
    return gene_effect


def load_relg4_growth_rates():
    """Load relg4 growth rates from df_tau_relg4.csv.

    Returns dict {marker: relg4} where marker is extracted from the cellline
    column (e.g., "A549-doxCas9-C8 + EEA1" → "EEA1").
    """
    if not RELG4_CSV.exists():
        log.warning(f"relg4 CSV not found: {RELG4_CSV}")
        return {}
    df = pd.read_csv(RELG4_CSV)
    marker_to_relg4 = {}
    for _, row in df.iterrows():
        cellline = row["cellline"]
        if " + " in cellline:
            marker = cellline.split(" + ", 1)[1].strip()
            marker_to_relg4[marker] = row["relg4"]
    log.info(f"Loaded relg4 growth rates for {len(marker_to_relg4)} cell lines")
    return marker_to_relg4


def build_experiment_marker_map(exp_ids):
    """Map experiment IDs to their GFP marker using ops_channel_maps.yaml.

    For each experiment, looks up the GFP channel label and extracts the marker
    protein (e.g., ops0089 → "early endosome, EEA1" → "EEA1").
    Falls back to mCherry if GFP has "no label".

    Returns dict {exp_id: marker_name}.
    """
    from cyclops_utils.data.feature_metadata import FeatureMetadata
    meta = FeatureMetadata(metadata_path=str(CHANNEL_MAPS_YAML))

    exp_markers = {}
    for exp_id in exp_ids:
        # Try GFP first, then mCherry
        for channel in ["GFP", "mCherry"]:
            info = meta.get_channel_info(exp_id, channel)
            marker = info.get("marker")
            if marker and marker not in ("None", "unknown"):
                exp_markers[exp_id] = marker
                break
    log.info(f"Mapped {len(exp_markers)}/{len(exp_ids)} experiments to markers")
    return exp_markers


def create_relg4_correlation(exp_data, exp_markers, relg4_map, output_path,
                              title_suffix=""):
    """Scatter plots of experiment mean Cohen's d / cosine dist vs relg4 growth rate.

    Matches experiments to cell lines via their GFP marker, then plots:
      Left: mean Cohen's d vs relg4
      Right: mean cosine distance vs relg4
    """
    # Build matched data: experiment → (mean_d, mean_cos, relg4, marker)
    matched = []
    for exp_id in sorted(exp_data.keys()):
        marker = exp_markers.get(exp_id)
        if marker is None:
            continue
        relg4 = relg4_map.get(marker)
        if relg4 is None:
            continue
        matched.append({
            "experiment": exp_id,
            "marker": marker,
            "mean_d": exp_data[exp_id]["mean_d"],
            "mean_cos": exp_data[exp_id]["mean_cos"],
            "relg4": relg4,
        })

    if len(matched) < 3:
        log.warning(f"Only {len(matched)} experiments matched to relg4 — need at least 3")
        return

    df = pd.DataFrame(matched)
    log.info(f"relg4 correlation: {len(df)} experiments matched")

    fig, (ax_d, ax_cos) = plt.subplots(1, 2, figsize=(16, 7))

    for ax, y_col, ylabel, title in [
        (ax_d, "mean_d", "Mean Cohen's d", "Cohen's d vs relg4 Growth Rate"),
        (ax_cos, "mean_cos", "Mean Cosine Distance", "Cosine Distance vs relg4 Growth Rate"),
    ]:
        x = df["relg4"].values
        y = df[y_col].values
        labels = df["experiment"].values

        _draw_scatter(ax, x, y, labels, "relg4 Growth Rate", ylabel,
                      f"{title}{title_suffix}")

        # Add marker labels as secondary annotation
        for xi, yi, marker in zip(x, y, df["marker"].values):
            ax.annotate(marker, (xi, yi), fontsize=5.5, ha="right", va="top",
                        xytext=(-3, -3), textcoords="offset points",
                        alpha=0.5, color="gray", style="italic")

    fig.suptitle(f"Phase2D Metrics vs relg4 Growth Rate{title_suffix}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {output_path}")

    # Save matched data CSV
    csv_path = output_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    log.info(f"Saved: {csv_path}")


def create_gene_effect_scatter_grid(exp_data, gene_effect_map, output_path,
                                     metric_key="cohens_d", metric_label="Cohen's d",
                                     title_suffix=""):
    """Create a grid of per-experiment scatter plots: gene effect vs metric.

    Each subplot shows one experiment with regression line, Pearson r, and R².
    Returns a DataFrame of per-experiment regression stats.
    """
    exps = sorted(exp_data.keys(), key=lambda e: int(
        __import__("re").search(r"ops(\d+)", e).group(1)
        if __import__("re").search(r"ops(\d+)", e) else 0
    ))
    n = len(exps)
    if n == 0:
        return pd.DataFrame()

    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    regression_rows = []
    for idx, exp_id in enumerate(exps):
        ax = axes[idx // ncols][idx % ncols]
        metric_vals = exp_data[exp_id][metric_key]

        # Match perturbations to gene effect
        shared = sorted(set(metric_vals.keys()) & set(gene_effect_map.keys()))
        if len(shared) < 5:
            ax.text(0.5, 0.5, f"n={len(shared)}\n(too few)", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(exp_id, fontsize=9)
            regression_rows.append({
                "experiment": exp_id, "n_matched": len(shared),
                "pearson_r": np.nan, "r2": np.nan, "slope": np.nan,
            })
            continue

        x = np.array([gene_effect_map[p] for p in shared])
        y = np.array([metric_vals[p] for p in shared])

        # Drop entries where either value is NaN/inf
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]

        if len(x) < 5:
            ax.text(0.5, 0.5, f"n={len(x)}\n(too few finite)", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(exp_id, fontsize=9)
            regression_rows.append({
                "experiment": exp_id, "n_matched": len(x),
                "pearson_r": np.nan, "r2": np.nan, "slope": np.nan,
            })
            continue

        ax.scatter(x, y, s=8, alpha=0.4, c="steelblue", edgecolors="none")

        # Regression
        r, slope, r2 = np.nan, np.nan, np.nan
        if x.std() > 0 and y.std() > 0:
            try:
                r, p_val = stats.pearsonr(x, y)
                slope, intercept = np.polyfit(x, y, 1)
                r2 = r ** 2
                x_fit = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_fit, slope * x_fit + intercept, "r--", lw=1.2, alpha=0.7)
            except (np.linalg.LinAlgError, ValueError):
                r, slope, r2 = np.nan, np.nan, np.nan

        ax.text(0.05, 0.95, f"r={r:.3f}\nR²={r2:.3f}\nn={len(shared)}",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat", alpha=0.5))

        is_highlight = exp_id in HIGHLIGHT_EXPS
        ax.set_title(exp_id, fontsize=9,
                     fontweight="bold" if is_highlight else "normal",
                     color="red" if is_highlight else "black")
        ax.tick_params(labelsize=7)

        regression_rows.append({
            "experiment": exp_id, "n_matched": len(shared),
            "pearson_r": r, "r2": r2, "slope": slope,
        })

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Shared axis labels
    for row_axes in axes:
        row_axes[0].set_ylabel(metric_label, fontsize=9)
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("Gene Effect (CERES)", fontsize=9)

    fig.suptitle(
        f"{metric_label} vs Gene Effect (CERES) — Per Experiment{title_suffix}\n"
        f"({n} experiments)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {output_path}")

    return pd.DataFrame(regression_rows)


def create_gene_effect_summary(regression_df, output_path, metric_label="Cohen's d",
                                title_suffix=""):
    """Bar chart of Pearson r across experiments for gene effect correlation.

    Shows which experiments have the strongest metric ↔ gene effect relationship.
    """
    df = regression_df.dropna(subset=["pearson_r"]).copy()
    if len(df) < 2:
        return

    df = df.sort_values("pearson_r")

    fig, (ax_r, ax_slope) = plt.subplots(1, 2, figsize=(16, max(6, len(df) * 0.3)))

    colors_r = ["red" if e in HIGHLIGHT_EXPS else "steelblue" for e in df["experiment"]]
    ax_r.barh(df["experiment"], df["pearson_r"], color=colors_r, alpha=0.7, edgecolor="white")
    ax_r.set_xlabel("Pearson r", fontsize=11)
    ax_r.set_title(f"{metric_label} vs Gene Effect: Pearson r{title_suffix}", fontsize=12, fontweight="bold")
    ax_r.axvline(0, color="gray", lw=0.8, ls="--")
    mean_r = df["pearson_r"].mean()
    ax_r.axvline(mean_r, color="green", lw=1.5, alpha=0.7, label=f"Mean: {mean_r:.3f}")
    ax_r.legend(fontsize=9)
    ax_r.tick_params(labelsize=9)

    colors_s = ["red" if e in HIGHLIGHT_EXPS else "steelblue" for e in df["experiment"]]
    ax_slope.barh(df["experiment"], df["slope"], color=colors_s, alpha=0.7, edgecolor="white")
    ax_slope.set_xlabel("Slope", fontsize=11)
    ax_slope.set_title(f"{metric_label} vs Gene Effect: Slope{title_suffix}", fontsize=12, fontweight="bold")
    ax_slope.axvline(0, color="gray", lw=0.8, ls="--")
    ax_slope.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {output_path}")


def create_gene_effect_combined_summary(reg_cd, reg_cos, output_path, title_suffix=""):
    """Side-by-side bar chart comparing Cohen's d vs Cosine Distance gene effect correlations.

    Left: Pearson r for both metrics per experiment (grouped bars).
    Right: Heatmap of experiments × metric showing Pearson r (darker = weaker).
    """
    cd = reg_cd.dropna(subset=["pearson_r"]).copy() if reg_cd is not None else pd.DataFrame()
    cos = reg_cos.dropna(subset=["pearson_r"]).copy() if reg_cos is not None else pd.DataFrame()

    if len(cd) < 2 and len(cos) < 2:
        return

    # Merge on experiment
    merged = pd.merge(
        cd[["experiment", "pearson_r", "slope"]].rename(
            columns={"pearson_r": "r_cohens_d", "slope": "slope_cohens_d"}),
        cos[["experiment", "pearson_r", "slope"]].rename(
            columns={"pearson_r": "r_cosine_dist", "slope": "slope_cosine_dist"}),
        on="experiment", how="outer",
    )

    import re
    merged["_sort"] = merged["experiment"].apply(
        lambda e: int(re.search(r"ops(\d+)", e).group(1)) if re.search(r"ops(\d+)", e) else 0
    )
    merged = merged.sort_values("_sort")

    n = len(merged)
    fig, (ax_bar, ax_heat) = plt.subplots(1, 2, figsize=(20, max(8, n * 0.35)),
                                            gridspec_kw={"width_ratios": [3, 1], "wspace": 0.3})

    # --- Left: grouped horizontal bar chart ---
    y_pos = np.arange(n)
    bar_h = 0.35
    ax_bar.barh(y_pos + bar_h / 2, merged["r_cohens_d"], bar_h,
                color=["red" if e in HIGHLIGHT_EXPS else "steelblue" for e in merged["experiment"]],
                alpha=0.7, label="Cohen's d")
    ax_bar.barh(y_pos - bar_h / 2, merged["r_cosine_dist"], bar_h,
                color=["darkred" if e in HIGHLIGHT_EXPS else "darkorange" for e in merged["experiment"]],
                alpha=0.7, label="Cosine Distance")
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(merged["experiment"], fontsize=9)
    ax_bar.set_xlabel("Pearson r (metric vs Gene Effect CERES)", fontsize=11)
    ax_bar.set_title(f"Gene Effect Correlation by Experiment{title_suffix}", fontsize=12, fontweight="bold")
    ax_bar.axvline(0, color="gray", lw=0.8, ls="--")
    mean_cd = merged["r_cohens_d"].mean()
    mean_cos = merged["r_cosine_dist"].mean()
    ax_bar.axvline(mean_cd, color="steelblue", lw=1.5, ls=":", alpha=0.7,
                   label=f"Cohen's d mean: {mean_cd:.3f}")
    ax_bar.axvline(mean_cos, color="darkorange", lw=1.5, ls=":", alpha=0.7,
                   label=f"Cosine dist mean: {mean_cos:.3f}")
    ax_bar.legend(fontsize=9, loc="lower right")
    ax_bar.tick_params(labelsize=9)

    # --- Right: heatmap ---
    heat_data = merged[["r_cohens_d", "r_cosine_dist"]].values
    im = ax_heat.imshow(heat_data, cmap="RdYlGn", aspect="auto",
                        vmin=np.nanmin(heat_data), vmax=np.nanmax(heat_data))
    ax_heat.set_xticks([0, 1])
    ax_heat.set_xticklabels(["Cohen's d", "Cosine Dist"], fontsize=10, rotation=45, ha="right")
    ax_heat.set_yticks(range(n))
    ax_heat.set_yticklabels(merged["experiment"], fontsize=9)
    ax_heat.set_title("Pearson r", fontsize=11, fontweight="bold")

    # Annotate cells with values
    for i in range(n):
        for j in range(2):
            val = heat_data[i, j]
            if np.isfinite(val):
                ax_heat.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r", fontsize=9)

    fig.suptitle(
        f"Gene Effect (CERES) Correlation Summary{title_suffix}",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {output_path}")


# --- Parallel helpers ---

def _process_native(exp_id, path):
    """Load guide_bulked and compute metrics for one experiment."""
    result = load_guide_bulked(path)
    if result is None:
        return None
    X, perts, total_cells = result
    cd, cosd, mean_d, mean_cos = compute_both_metrics(X, perts)
    if not cd:
        return None
    data = {
        "cohens_d": cd, "cosine_dist": cosd,
        "mean_d": mean_d, "mean_cos": mean_cos,
        "n_guides": len(X),
    }
    if total_cells is not None:
        data["n_cells"] = total_cells
    return exp_id, data


def _process_downsampled(exp_id, path, target_cells):
    """Load cell-level, downsample, aggregate, compute metrics for one experiment."""
    X, perts = load_cell_and_aggregate(path, target_cells)
    cd, cosd, mean_d, mean_cos = compute_both_metrics(X, perts)
    if not cd:
        return None
    return exp_id, {
        "cohens_d": cd, "cosine_dist": cosd,
        "mean_d": mean_d, "mean_cos": mean_cos,
        "n_guides": len(X),
    }


# --- Cache helpers ---

def _load_cached_data(out_dir, prefix="native"):
    """Reconstruct exp_data dict from cached per-perturbation + summary CSVs.

    Returns exp_data dict or None if cache is missing/invalid.
    """
    pert_path = out_dir / f"metrics_{prefix}_per_perturbation.csv"
    summary_path = out_dir / f"metrics_{prefix}_summary.csv"
    if not pert_path.exists() or not summary_path.exists():
        return None

    try:
        pert_df = pd.read_csv(pert_path)
        summary_df = pd.read_csv(summary_path)
    except Exception as e:
        log.warning(f"Failed to read cached CSVs: {e}")
        return None

    # Build summary lookup
    summary_lookup = {}
    for _, row in summary_df.iterrows():
        summary_lookup[row["experiment"]] = row

    exp_data = {}
    for exp_id, grp in pert_df.groupby("experiment"):
        cd = dict(zip(grp["perturbation"], grp["cohens_d"]))
        cosd = dict(zip(grp["perturbation"], grp["cosine_dist"]))
        # Drop NaN entries
        cd = {k: v for k, v in cd.items() if pd.notna(v)}
        cosd = {k: v for k, v in cosd.items() if pd.notna(v)}

        mean_d = float(np.mean(list(cd.values()))) if cd else 0.0
        mean_cos = float(np.mean(list(cosd.values()))) if cosd else 0.0

        data = {
            "cohens_d": cd, "cosine_dist": cosd,
            "mean_d": mean_d, "mean_cos": mean_cos,
            "n_guides": int(summary_lookup[exp_id]["n_guides"]) if exp_id in summary_lookup else len(cd),
        }
        if exp_id in summary_lookup and "n_cells" in summary_lookup[exp_id].index:
            val = summary_lookup[exp_id].get("n_cells")
            if pd.notna(val):
                data["n_cells"] = int(val)
        exp_data[exp_id] = data

    log.info(f"Loaded cached {prefix} data: {len(exp_data)} experiments from {pert_path.name}")
    return exp_data


# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description="Ridge plot of Cohen's d vs Cosine Distance across all Phase dino experiments")
    parser.add_argument("-o", "--output-dir", default="scripts/qc_phase2d_output",
                        help="Output directory")
    parser.add_argument("--downsample", action="store_true",
                        help="Also produce downsampled-to-smallest plot (slow)")
    parser.add_argument("--remove-bad", action="store_true",
                        help="Exclude bad experiments via cyclops_utils bad_experiments.yaml")
    parser.add_argument("--force", action="store_true",
                        help="Recompute metrics even if cached CSVs exist")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_jobs = get_optimal_workers(use_gpu=False, verbose=False)
    n_jobs = max(1, n_jobs)
    log.info(f"Using {n_jobs} parallel workers")

    # --- 1. Discover ---
    guide_paths = discover_experiments(exclude_bad=args.remove_bad)
    log.info(f"Found {len(guide_paths)} experiments with Phase dino guide_bulked")

    # --- 2. Native (guide_bulked) ---
    native_data = None
    if not args.force:
        native_data = _load_cached_data(out_dir, prefix="native")

    if native_data is None:
        log.info("Computing metrics from guide_bulked (native)...")
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_process_native)(exp_id, path)
            for exp_id, path in sorted(guide_paths.items())
        )
        native_data = {}
        for r in results:
            if r is not None:
                exp_id, data = r
                native_data[exp_id] = data
                log.info(f"  {exp_id}: d={data['mean_d']:.4f}, cos_dist={data['mean_cos']:.6f}")

        log.info(f"\n{len(native_data)} experiments loaded")

        # Summary
        rows = []
        for exp_id in sorted(native_data.keys(), key=lambda e: native_data[e]["mean_d"]):
            d = native_data[exp_id]
            cd_arr = np.array(list(d["cohens_d"].values()))
            cos_arr = np.array(list(d["cosine_dist"].values()))
            row = {
                "experiment": exp_id, "n_guides": d["n_guides"],
                "mean_cohens_d": d["mean_d"], "median_cohens_d": float(np.median(cd_arr)),
                "mean_cosine_dist": d["mean_cos"], "median_cosine_dist": float(np.median(cos_arr)),
            }
            if "n_cells" in d:
                row["n_cells"] = d["n_cells"]
            rows.append(row)
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(out_dir / "metrics_native_summary.csv", index=False)

        # Per-perturbation CSV: one row per experiment × perturbation
        pert_rows = []
        for exp_id in sorted(native_data.keys()):
            d = native_data[exp_id]
            all_perts = sorted(set(d["cohens_d"].keys()) | set(d["cosine_dist"].keys()))
            for p in all_perts:
                pert_rows.append({
                    "experiment": exp_id,
                    "perturbation": p,
                    "cohens_d": d["cohens_d"].get(p, np.nan),
                    "cosine_dist": d["cosine_dist"].get(p, np.nan),
                })
        pert_df = pd.DataFrame(pert_rows)
        pert_df.to_csv(out_dir / "metrics_native_per_perturbation.csv", index=False)
        log.info(f"Saved per-perturbation CSV: {len(pert_df)} rows across {len(native_data)} experiments")

    # Build summary from native_data (works for both cached and fresh)
    summary_rows = []
    for exp_id in sorted(native_data.keys(), key=lambda e: native_data[e]["mean_d"]):
        d = native_data[exp_id]
        cd_arr = np.array(list(d["cohens_d"].values()))
        cos_arr = np.array(list(d["cosine_dist"].values()))
        summary_rows.append({
            "experiment": exp_id, "n_guides": d.get("n_guides", len(cd_arr)),
            "mean_cohens_d": d["mean_d"], "median_cohens_d": float(np.median(cd_arr)),
            "mean_cosine_dist": d["mean_cos"], "median_cosine_dist": float(np.median(cos_arr)),
        })
    summary_df = pd.DataFrame(summary_rows)

    print(f"\n{'='*100}")
    print(f"Native (guide_bulked) — Cohen's d vs Cosine Distance")
    print(f"{'='*100}")
    print(f"{'experiment':>10s}  {'mean_d':>8s}  {'median_d':>9s}  {'mean_cos':>10s}  {'median_cos':>11s}")
    print("-" * 100)
    for _, r in summary_df.iterrows():
        print(f"  {r['experiment']:>10s}  {r['mean_cohens_d']:8.4f}  {r['median_cohens_d']:9.4f}  "
              f"{r['mean_cosine_dist']:10.6f}  {r['median_cosine_dist']:11.6f}")
    print(f"{'='*100}")

    # Cohen's d range vs cosine distance range
    d_range = summary_df["mean_cohens_d"].max() - summary_df["mean_cohens_d"].min()
    d_cv = summary_df["mean_cohens_d"].std() / summary_df["mean_cohens_d"].mean()
    cos_range = summary_df["mean_cosine_dist"].max() - summary_df["mean_cosine_dist"].min()
    cos_cv = summary_df["mean_cosine_dist"].std() / summary_df["mean_cosine_dist"].mean()
    print(f"\nCohen's d  — range: {d_range:.4f}, CV: {d_cv:.4f}")
    print(f"Cosine dist — range: {cos_range:.6f}, CV: {cos_cv:.4f}")
    print(f"Cohen's d spreads {d_cv/cos_cv:.1f}x more across experiments (by CV)")

    create_dual_ridge_plot(native_data,
                           out_dir / "ridge_cohens_d_vs_cosine_native.png",
                           title_suffix=" — Native (all cells)")
    create_dual_ridge_plot(native_data,
                           out_dir / "ridge_cohens_d_vs_cosine_native_by_ops.png",
                           title_suffix=" — Native (all cells)",
                           sort_by="ops_number")

    create_pairwise_correlation_heatmaps(
        native_data, out_dir / "pairwise_correlation_heatmaps_native.png",
        title_suffix=" — Native")

    # --- Correlation scatters: Cohen's d vs ALL QC metrics ---
    create_correlation_scatters(native_data, out_dir,
                                filename="cohens_d_correlations.png",
                                title_suffix=" (native)")

    # --- QC metric ranking by correlation strength ---
    ranking_result = create_qc_metric_ranking(
        native_data, out_dir / "qc_metric_ranking_native.png",
        title_suffix=" — Native")
    if ranking_result is not None:
        ranking_df, ps_data = ranking_result
        create_top_qc_scatters(
            native_data, ranking_df, ps_data,
            out_dir / "qc_top_correlations_native.png",
            title_suffix=" — Native")
        create_confluency_control_analysis(
            native_data, ps_data,
            out_dir / "confluency_control_native.png",
            title_suffix=" — Native")

    # --- Gene effect (CERES) correlation ---
    gene_effect_map = load_gene_effect_map()
    if gene_effect_map:
        # Cohen's d vs gene effect
        reg_df_cd = create_gene_effect_scatter_grid(
            native_data, gene_effect_map,
            out_dir / "gene_effect_vs_cohens_d_native.png",
            metric_key="cohens_d", metric_label="Cohen's d",
            title_suffix=" — Native",
        )
        if not reg_df_cd.empty:
            create_gene_effect_summary(
                reg_df_cd, out_dir / "gene_effect_vs_cohens_d_summary_native.png",
                metric_label="Cohen's d", title_suffix=" — Native",
            )
            reg_df_cd.to_csv(out_dir / "gene_effect_regression_native.csv", index=False)

        # Cosine distance vs gene effect
        reg_df_cos = create_gene_effect_scatter_grid(
            native_data, gene_effect_map,
            out_dir / "gene_effect_vs_cosine_dist_native.png",
            metric_key="cosine_dist", metric_label="Cosine Distance",
            title_suffix=" — Native",
        )
        if not reg_df_cos.empty:
            create_gene_effect_summary(
                reg_df_cos, out_dir / "gene_effect_vs_cosine_dist_summary_native.png",
                metric_label="Cosine Distance", title_suffix=" — Native",
            )

        # Combined summary: both metrics side-by-side
        create_gene_effect_combined_summary(
            reg_df_cd, reg_df_cos,
            out_dir / "gene_effect_combined_summary_native.png",
            title_suffix=" — Native",
        )

    # --- relg4 growth rate correlation ---
    relg4_map = load_relg4_growth_rates()
    exp_markers = build_experiment_marker_map(list(native_data.keys()))
    if relg4_map and exp_markers:
        create_relg4_correlation(
            native_data, exp_markers, relg4_map,
            out_dir / "relg4_vs_metrics_native.png",
            title_suffix=" — Native",
        )

    # --- Downsampled correlation scatters (from cached CSV if available) ---
    ds_summary_path = out_dir / "cohens_d_downsampled_summary.csv"
    if not ds_summary_path.exists():
        ds_summary_path = out_dir / "metrics_downsampled_summary.csv"
    if ds_summary_path.exists():
        log.info(f"Loading cached downsampled data from {ds_summary_path}")
        ds_csv = pd.read_csv(ds_summary_path)
        ds_data_from_csv = {}
        for _, row in ds_csv.iterrows():
            d = {"mean_d": row["mean_cohens_d"]}
            if "mean_cosine_dist" in row.index and pd.notna(row.get("mean_cosine_dist")):
                d["mean_cos"] = row["mean_cosine_dist"]
            ds_data_from_csv[row["experiment"]] = d
        create_correlation_scatters(ds_data_from_csv, out_dir,
                                    filename="cohens_d_correlations_downsampled.png",
                                    title_suffix=" (downsampled)")
        ds_ranking = create_qc_metric_ranking(
            ds_data_from_csv, out_dir / "qc_metric_ranking_downsampled.png",
            title_suffix=" — Downsampled (cached)")
        if ds_ranking is not None:
            ds_rdf, ds_ps = ds_ranking
            create_top_qc_scatters(
                ds_data_from_csv, ds_rdf, ds_ps,
                out_dir / "qc_top_correlations_downsampled.png",
                title_suffix=" — Downsampled (cached)")
            create_confluency_control_analysis(
                ds_data_from_csv, ds_ps,
                out_dir / "confluency_control_downsampled.png",
                title_suffix=" — Downsampled (cached)")
    else:
        log.info("No cached downsampled summary found — run with --downsample first to generate")

    # --- 3. Downsampled ---
    if args.downsample:
        log.info("\n=== DOWNSAMPLED MODE ===")

        ds_data = None
        min_cells = None
        min_exp = None
        if not args.force:
            ds_data = _load_cached_data(out_dir, prefix="downsampled")

        if ds_data is None:
            cell_paths = discover_cell_level(set(guide_paths.keys()), exclude_bad=args.remove_bad)
            log.info(f"Found {len(cell_paths)} experiments with features_processed")

            cell_counts = {}
            for exp_id, path in sorted(cell_paths.items()):
                try:
                    with h5py.File(path, "r") as f:
                        x = f["X"]
                        if hasattr(x, "shape") and len(x.shape) >= 1:
                            cell_counts[exp_id] = int(x.shape[0])
                        elif "shape" in x.attrs:
                            cell_counts[exp_id] = int(x.attrs["shape"][0])
                        else:
                            cell_counts[exp_id] = len(f["obs"]["_index"])
                    log.info(f"  {exp_id}: {cell_counts.get(exp_id, '?')} cells")
                except Exception as e:
                    log.warning(f"  {exp_id}: failed to read shape: {e}")

            min_cells = min(cell_counts.values())
            min_exp = min(cell_counts, key=cell_counts.get)
            log.info(f"Downsampling all to {min_exp}'s count: {min_cells} cells")

            log.info(f"Processing {len(cell_paths)} experiments in parallel (n_jobs={n_jobs})...")
            ds_results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_process_downsampled)(exp_id, path, min_cells)
                for exp_id, path in sorted(cell_paths.items())
            )
            ds_data = {}
            for r in ds_results:
                if r is not None:
                    exp_id, data = r
                    ds_data[exp_id] = data
                    log.info(f"  {exp_id}: {data['n_guides']} guides, d={data['mean_d']:.4f}, cos={data['mean_cos']:.6f}")

            ds_rows = []
            for exp_id in sorted(ds_data.keys(), key=lambda e: ds_data[e]["mean_d"]):
                d = ds_data[exp_id]
                cd_arr = np.array(list(d["cohens_d"].values()))
                cos_arr = np.array(list(d["cosine_dist"].values()))
                ds_rows.append({
                    "experiment": exp_id, "n_guides": d["n_guides"],
                    "mean_cohens_d": d["mean_d"], "median_cohens_d": float(np.median(cd_arr)),
                    "mean_cosine_dist": d["mean_cos"], "median_cosine_dist": float(np.median(cos_arr)),
                })
            ds_df = pd.DataFrame(ds_rows)
            ds_df.to_csv(out_dir / "metrics_downsampled_summary.csv", index=False)

            # Per-perturbation CSV for downsampled
            ds_pert_rows = []
            for exp_id in sorted(ds_data.keys()):
                d = ds_data[exp_id]
                all_perts = sorted(set(d["cohens_d"].keys()) | set(d["cosine_dist"].keys()))
                for p in all_perts:
                    ds_pert_rows.append({
                        "experiment": exp_id,
                        "perturbation": p,
                        "cohens_d": d["cohens_d"].get(p, np.nan),
                        "cosine_dist": d["cosine_dist"].get(p, np.nan),
                    })
            ds_pert_df = pd.DataFrame(ds_pert_rows)
            ds_pert_df.to_csv(out_dir / "metrics_downsampled_per_perturbation.csv", index=False)
            log.info(f"Saved downsampled per-perturbation CSV: {len(ds_pert_df)} rows across {len(ds_data)} experiments")

        ds_suffix = f" — Downsampled to {min_cells:,} cells" if min_cells else " — Downsampled"

        print(f"\n{'='*100}")
        print(f"Downsampled — Cohen's d vs Cosine Distance ({len(ds_data)} experiments)")
        print(f"{'='*100}")
        for exp_id in sorted(ds_data.keys(), key=lambda e: ds_data[e]["mean_d"]):
            d = ds_data[exp_id]
            cd_arr = np.array(list(d["cohens_d"].values()))
            cos_arr = np.array(list(d["cosine_dist"].values()))
            print(f"  {exp_id:>10s}  {d['mean_d']:8.4f}  {float(np.median(cd_arr)):9.4f}  "
                  f"{d['mean_cos']:10.6f}  {float(np.median(cos_arr)):11.6f}")
        print(f"{'='*100}")

        mean_ds = np.array([ds_data[e]["mean_d"] for e in ds_data])
        mean_cos_arr = np.array([ds_data[e]["mean_cos"] for e in ds_data])
        d_range = mean_ds.max() - mean_ds.min()
        d_cv = mean_ds.std() / mean_ds.mean()
        cos_range = mean_cos_arr.max() - mean_cos_arr.min()
        cos_cv = mean_cos_arr.std() / mean_cos_arr.mean()
        print(f"\nCohen's d  — range: {d_range:.4f}, CV: {d_cv:.4f}")
        print(f"Cosine dist — range: {cos_range:.6f}, CV: {cos_cv:.4f}")
        print(f"Cohen's d spreads {d_cv/max(cos_cv, 1e-9):.1f}x more across experiments (by CV)")

        create_dual_ridge_plot(ds_data,
                               out_dir / "ridge_cohens_d_vs_cosine_downsampled.png",
                               title_suffix=ds_suffix)
        create_dual_ridge_plot(ds_data,
                               out_dir / "ridge_cohens_d_vs_cosine_downsampled_by_ops.png",
                               title_suffix=ds_suffix,
                               sort_by="ops_number")

        create_pairwise_correlation_heatmaps(
            ds_data, out_dir / "pairwise_correlation_heatmaps_downsampled.png",
            title_suffix=ds_suffix)

        # Regenerate downsampled correlations from fresh data
        ds_data_slim = {e: {"mean_d": ds_data[e]["mean_d"], "mean_cos": ds_data[e]["mean_cos"]}
                        for e in ds_data}
        create_correlation_scatters(ds_data_slim, out_dir,
                                    filename="cohens_d_correlations_downsampled.png",
                                    title_suffix=f" ({ds_suffix.lstrip(' —')})")

        # QC metric ranking for downsampled
        ds_ranking = create_qc_metric_ranking(
            ds_data_slim, out_dir / "qc_metric_ranking_downsampled.png",
            title_suffix=ds_suffix)
        if ds_ranking is not None:
            ds_rdf, ds_ps = ds_ranking
            create_top_qc_scatters(
                ds_data_slim, ds_rdf, ds_ps,
                out_dir / "qc_top_correlations_downsampled.png",
                title_suffix=ds_suffix)
            create_confluency_control_analysis(
                ds_data_slim, ds_ps,
                out_dir / "confluency_control_downsampled.png",
                title_suffix=ds_suffix)

        # Gene effect correlation for downsampled
        if gene_effect_map:
            ds_reg_cd = create_gene_effect_scatter_grid(
                ds_data, gene_effect_map,
                out_dir / "gene_effect_vs_cohens_d_downsampled.png",
                metric_key="cohens_d", metric_label="Cohen's d",
                title_suffix=ds_suffix,
            )
            if not ds_reg_cd.empty:
                create_gene_effect_summary(
                    ds_reg_cd, out_dir / "gene_effect_vs_cohens_d_summary_downsampled.png",
                    metric_label="Cohen's d", title_suffix=ds_suffix,
                )
                ds_reg_cd.to_csv(out_dir / "gene_effect_regression_downsampled.csv", index=False)

            ds_reg_cos = create_gene_effect_scatter_grid(
                ds_data, gene_effect_map,
                out_dir / "gene_effect_vs_cosine_dist_downsampled.png",
                metric_key="cosine_dist", metric_label="Cosine Distance",
                title_suffix=ds_suffix,
            )
            if not ds_reg_cos.empty:
                create_gene_effect_summary(
                    ds_reg_cos, out_dir / "gene_effect_vs_cosine_dist_summary_downsampled.png",
                    metric_label="Cosine Distance", title_suffix=ds_suffix,
                )

            create_gene_effect_combined_summary(
                ds_reg_cd, ds_reg_cos,
                out_dir / "gene_effect_combined_summary_downsampled.png",
                title_suffix=ds_suffix,
            )

        # relg4 growth rate correlation for downsampled
        if relg4_map and exp_markers:
            ds_markers = {e: exp_markers[e] for e in ds_data if e in exp_markers}
            create_relg4_correlation(
                ds_data, ds_markers, relg4_map,
                out_dir / "relg4_vs_metrics_downsampled.png",
                title_suffix=ds_suffix,
            )

    print(f"\nOutputs in: {out_dir}")


if __name__ == "__main__":
    main()
