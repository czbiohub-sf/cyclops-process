"""
ISS Cell Size / Shape Metrics
=============================

Computes per-cell bounding box metrics from the linked CSV, avoiding the need
to load large segmentation zarr arrays.

Metrics per cell: bbox_height, bbox_width, bbox_area, bbox_aspect_ratio
Summary stats per well: mean/median/min/max/std for each metric + cell count

Produces per experiment:
  - Per-cell CSV per well (under 3-assembly/cell_sizes/)
  - Summary statistics CSV (under results_iss)
  - Distribution plot (under results_iss)

CLI Usage
---------
    python -m cyclops_process.metrics.plate_stats.iss_cell_size 33
    python -m cyclops_process.metrics.plate_stats.iss_cell_size -e 33
    python -m cyclops_process.metrics.plate_stats.iss_cell_size --experiment ops0033 --force
"""

import ast
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from cyclops_utils.data.experiment import OpsDataset

# Shape columns used for summary statistics (all get mean/median/min/max/std)
_SHAPE_COLUMNS = [
    "bbox_height",
    "bbox_width",
    "bbox_area",
    "bbox_aspect_ratio",
]

# Threshold for "large cell" classification (width AND height >= this)
_LARGE_CELL_THRESHOLD_PX = 300


def _parse_bbox(bbox_str) -> tuple | None:
    """Parse bbox string '(min_row, min_col, max_row, max_col)' into tuple."""
    try:
        if pd.isna(bbox_str):
            return None
        bbox = ast.literal_eval(str(bbox_str))
        if len(bbox) == 4:
            return tuple(bbox)
        return None
    except Exception:
        return None


def _compute_cell_props_from_linked(linked_df: pd.DataFrame, well: str) -> pd.DataFrame:
    """Extract cell size metrics from bbox column in linked CSV."""
    if "bbox" not in linked_df.columns:
        return pd.DataFrame()

    # Parse bboxes
    bboxes = linked_df["bbox"].apply(_parse_bbox)
    valid = bboxes.notna()
    if valid.sum() == 0:
        return pd.DataFrame()

    df = linked_df.loc[valid].copy()
    parsed = bboxes[valid]

    df["bbox_height"] = parsed.apply(lambda b: b[2] - b[0])
    df["bbox_width"] = parsed.apply(lambda b: b[3] - b[1])
    df["bbox_area"] = df["bbox_height"] * df["bbox_width"]
    df["bbox_aspect_ratio"] = np.where(
        df["bbox_width"] > 0,
        df["bbox_height"] / df["bbox_width"],
        np.nan,
    )
    df["is_large"] = (
        (df["bbox_width"] >= _LARGE_CELL_THRESHOLD_PX)
        & (df["bbox_height"] >= _LARGE_CELL_THRESHOLD_PX)
    )
    df["well"] = well

    # Keep relevant columns
    keep_cols = ["well", "bbox_height", "bbox_width", "bbox_area", "bbox_aspect_ratio", "is_large"]
    if "segmentation_id" in df.columns:
        keep_cols.insert(0, "segmentation_id")

    # Add gene/barcode info if available
    for col in ("gene_name", "Gene name", "sgRNA"):
        if col in df.columns:
            df = df.rename(columns={col: "gene_name"}) if col != "gene_name" else df
            if "gene_name" not in keep_cols:
                keep_cols.append("gene_name")
            break
    if "barcode" in df.columns:
        keep_cols.append("barcode")

    return df[[c for c in keep_cols if c in df.columns]].reset_index(drop=True)


def _compute_summary(all_cells_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-well and pooled summary statistics."""

    def _stats(group):
        result = {"cell_count": len(group)}
        for col in _SHAPE_COLUMNS:
            if col in group.columns:
                series = group[col].dropna()
                result[f"{col}_mean"] = series.mean() if len(series) else np.nan
                result[f"{col}_median"] = series.median() if len(series) else np.nan
                result[f"{col}_min"] = series.min() if len(series) else np.nan
                result[f"{col}_max"] = series.max() if len(series) else np.nan
                result[f"{col}_std"] = series.std() if len(series) else np.nan
        if "is_large" in group.columns:
            n_large = int(group["is_large"].sum())
            result["large_cell_count"] = n_large
            result["large_cell_pct"] = (
                n_large / len(group) * 100 if len(group) else 0.0
            )
        return pd.Series(result)

    per_well = all_cells_df.groupby("well").apply(_stats).reset_index()
    pooled = _stats(all_cells_df).to_frame().T
    pooled["well"] = "ALL"
    return pd.concat([per_well, pooled], ignore_index=True)


# ISS pixel scale (20x objective): 0.325 µm per pixel
_UM_PER_PX = 0.325


def _px_to_um(px: float) -> float:
    """Convert pixel length to microns."""
    return px * _UM_PER_PX


def _area_px_to_um2(area_px: float) -> float:
    """Convert pixel area to µm²."""
    return area_px * _UM_PER_PX ** 2


def _equivalent_diameter_um(area_px: float) -> float:
    """Convert pixel area to equivalent circular diameter in µm."""
    return 2.0 * np.sqrt(_area_px_to_um2(area_px) / np.pi)


def _add_dual_xaxis(ax, label_px: str, label_um: str):
    """Add a secondary x-axis showing micron values alongside pixels."""
    ax.set_xlabel(label_px)
    ax2 = ax.twiny()
    ax2.set_xlim([v * _UM_PER_PX for v in ax.get_xlim()])
    ax2.set_xlabel(label_um, fontsize=10, color="grey")
    ax2.tick_params(axis="x", labelsize=9, colors="grey")
    return ax2


def _add_dual_xaxis_area(ax):
    """Add secondary x-axis converting area (px²) to equivalent diameter (µm)."""
    ax.set_xlabel("Cell BBox Area (px²)")
    ax2 = ax.twiny()
    lo, hi = ax.get_xlim()
    ax2.set_xlim([_equivalent_diameter_um(max(lo, 1)), _equivalent_diameter_um(max(hi, 1))])
    ax2.set_xlabel("Equiv. Circular Diameter (µm)", fontsize=10, color="grey")
    ax2.tick_params(axis="x", labelsize=9, colors="grey")
    return ax2


# Key cell diameter reference points (in µm) for annotating plots
_DIAMETER_REFS_UM = [10, 20, 50, 100, 200]


def _diameter_um_to_area_px(diam_um: float) -> float:
    """Convert a circular diameter (µm) to bounding-box area (px²)."""
    radius_um = diam_um / 2.0
    area_um2 = np.pi * radius_um ** 2
    return area_um2 / (_UM_PER_PX ** 2)


def _add_diameter_refs_area(ax, color="teal", alpha=0.5):
    """Add vertical reference lines at key cell diameters on an area-axis plot."""
    lo, hi = ax.get_xlim()
    for diam in _DIAMETER_REFS_UM:
        area_px = _diameter_um_to_area_px(diam)
        if lo < area_px < hi:
            ax.axvline(area_px, color=color, linestyle=":", alpha=alpha, linewidth=0.8)
            ax.text(
                area_px, ax.get_ylim()[1] * 0.92,
                f"~{diam}µm\ndia",
                ha="center", va="top", fontsize=7, color=color, alpha=0.8,
            )


def _add_diameter_refs_width(ax, color="teal", alpha=0.5):
    """Add vertical reference lines at key cell sizes (µm) on a width-axis plot."""
    lo, hi = ax.get_xlim()
    for diam in _DIAMETER_REFS_UM:
        width_px = diam / _UM_PER_PX
        if lo < width_px < hi:
            ax.axvline(width_px, color=color, linestyle=":", alpha=alpha, linewidth=0.8)
            ax.text(
                width_px, ax.get_ylim()[1] * 0.92,
                f"{diam}µm",
                ha="center", va="top", fontsize=7, color=color, alpha=0.8,
            )


def _plot_distributions(
    all_cells_df: pd.DataFrame, output_path: Path, experiment: str
):
    """Generate cell size distribution plots (linear + log scale)."""
    wells = sorted(all_cells_df["well"].unique())
    areas = all_cells_df["bbox_area"].dropna()
    if areas.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- Top-left: pooled histogram (linear) ---
    ax = axes[0, 0]
    ax.hist(areas, bins=100, edgecolor="black", linewidth=0.3, alpha=0.7, color="steelblue")
    ax.axvline(areas.median(), color="red", linestyle="--",
               label=f"Median: {areas.median():,.0f} px² ({_equivalent_diameter_um(areas.median()):.1f} µm)")
    ax.axvline(areas.mean(), color="green", linestyle="--",
               label=f"Mean: {areas.mean():,.0f} px² ({_equivalent_diameter_um(areas.mean()):.1f} µm)")
    ax.axvline(_LARGE_CELL_THRESHOLD_PX ** 2, color="orange", linestyle=":",
               label=f"Large threshold: {_LARGE_CELL_THRESHOLD_PX}² px²")
    ax.set_ylabel("Count")
    ax.set_title("Pooled BBox Area Distribution (linear)")
    ax.legend(fontsize=9)
    _add_diameter_refs_area(ax)
    _add_dual_xaxis_area(ax)

    # --- Top-right: pooled histogram (log scale) ---
    ax = axes[0, 1]
    log_areas = areas[areas > 0]
    bins_log = np.logspace(np.log10(log_areas.min()), np.log10(log_areas.max()), 100)
    ax.hist(log_areas, bins=bins_log, edgecolor="black", linewidth=0.3, alpha=0.7, color="steelblue")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(areas.median(), color="red", linestyle="--", label=f"Median: {areas.median():,.0f}")
    ax.axvline(areas.mean(), color="green", linestyle="--", label=f"Mean: {areas.mean():,.0f}")
    ax.axvline(_LARGE_CELL_THRESHOLD_PX ** 2, color="orange", linestyle=":",
               label=f"Large threshold")
    ax.set_xlabel("Cell BBox Area (px², log)")
    ax.set_ylabel("Count (log)")
    ax.set_title("Pooled BBox Area Distribution (log-log)")
    ax.legend(fontsize=9)
    _add_diameter_refs_area(ax)
    _add_dual_xaxis_area(ax)

    # --- Bottom-left: boxplot per well (linear) ---
    ax = axes[1, 0]
    well_labels = [w.replace("/", "") for w in wells]
    data_per_well = [
        all_cells_df[all_cells_df["well"] == w]["bbox_area"].dropna().values
        for w in wells
    ]
    bp = ax.boxplot(data_per_well, labels=well_labels, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.6)
    # Annotate n per well
    for i, w in enumerate(wells):
        n = len(all_cells_df[all_cells_df["well"] == w])
        ax.text(i + 1, ax.get_ylim()[0], f"n={n}", ha="center", va="top",
                fontsize=8, color="grey")
    ax.set_xlabel("Well")
    ax.set_ylabel("Cell BBox Area (px²)")
    ax.set_title("BBox Area by Well")

    # Secondary y-axis for diameter
    ax_r = ax.twinx()
    lo, hi = ax.get_ylim()
    ax_r.set_ylim([_equivalent_diameter_um(max(lo, 1)), _equivalent_diameter_um(max(hi, 1))])
    ax_r.set_ylabel("Equiv. Diameter (µm)", fontsize=10, color="grey")
    ax_r.tick_params(axis="y", labelsize=9, colors="grey")

    # --- Bottom-right: bbox width histogram with dual axis ---
    ax = axes[1, 1]
    widths = all_cells_df["bbox_width"].dropna()
    ax.hist(widths, bins=80, edgecolor="black", linewidth=0.3, alpha=0.7, color="mediumpurple")
    ax.axvline(widths.median(), color="red", linestyle="--",
               label=f"Median: {widths.median():.0f} px ({_px_to_um(widths.median()):.1f} µm)")
    ax.axvline(_LARGE_CELL_THRESHOLD_PX, color="orange", linestyle=":",
               label=f"Large threshold: {_LARGE_CELL_THRESHOLD_PX} px ({_px_to_um(_LARGE_CELL_THRESHOLD_PX):.0f} µm)")
    ax.set_ylabel("Count")
    ax.set_title("Pooled BBox Width Distribution")
    ax.legend(fontsize=9)
    _add_diameter_refs_width(ax)
    _add_dual_xaxis(ax, "BBox Width (px)", "BBox Width (µm)")

    # Stats box
    n_total = len(all_cells_df)
    n_large = int(all_cells_df["is_large"].sum()) if "is_large" in all_cells_df.columns else 0
    stats_text = (
        f"Total cells: {n_total:,}\n"
        f"Large cells (>{_LARGE_CELL_THRESHOLD_PX}px): {n_large:,} ({100*n_large/n_total:.1f}%)\n"
        f"Pixel scale: {_UM_PER_PX} µm/px (20x)"
    )
    fig.text(0.99, 0.01, stats_text, ha="right", va="bottom", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle(f"Cell Size Metrics — {experiment}", fontsize=14)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    # --- Separate large-cell analysis plot ---
    _plot_large_cells(all_cells_df, output_path.parent, experiment)


def _plot_large_cells(
    all_cells_df: pd.DataFrame, out_dir: Path, experiment: str
):
    """Separate plot highlighting very large cells across wells."""
    if "is_large" not in all_cells_df.columns:
        return

    large = all_cells_df[all_cells_df["is_large"]].copy()
    if large.empty:
        print("[cell_size] No large cells found, skipping large cell plot")
        return

    wells = sorted(all_cells_df["well"].unique())
    well_labels = [w.replace("/", "") for w in wells]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # --- Panel 1: large cell count per well ---
    ax = axes[0]
    large_counts = []
    total_counts = []
    for w in wells:
        well_df = all_cells_df[all_cells_df["well"] == w]
        total_counts.append(len(well_df))
        large_counts.append(int(well_df["is_large"].sum()))
    pcts = [100 * lc / tc if tc > 0 else 0 for lc, tc in zip(large_counts, total_counts)]

    x = np.arange(len(wells))
    bars = ax.bar(x, large_counts, color="tomato", edgecolor="white", linewidth=0.5)
    for i, (bar, pct) in enumerate(zip(bars, pcts)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(well_labels)
    ax.set_xlabel("Well")
    ax.set_ylabel("Large Cell Count")
    ax.set_title(f"Large Cells per Well\n(bbox > {_LARGE_CELL_THRESHOLD_PX}px = {_px_to_um(_LARGE_CELL_THRESHOLD_PX):.0f} µm)")
    ax.grid(axis="y", alpha=0.3)

    # --- Panel 2: area distribution of large cells per well (violin) ---
    ax = axes[1]
    data_per_well = []
    labels_used = []
    for w, lbl in zip(wells, well_labels):
        vals = large[large["well"] == w]["bbox_area"].dropna().values
        if len(vals) > 0:
            data_per_well.append(vals)
            labels_used.append(lbl)

    if data_per_well:
        parts = ax.violinplot(data_per_well, showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("tomato")
            pc.set_alpha(0.6)
        ax.set_xticks(range(1, len(labels_used) + 1))
        ax.set_xticklabels(labels_used)

    ax.set_xlabel("Well")
    ax.set_ylabel("BBox Area (px²)")
    ax.set_title("Large Cell Area Distribution by Well")

    # Secondary y-axis for diameter
    ax_r = ax.twinx()
    lo, hi = ax.get_ylim()
    ax_r.set_ylim([_equivalent_diameter_um(max(lo, 1)), _equivalent_diameter_um(max(hi, 1))])
    ax_r.set_ylabel("Equiv. Diameter (µm)", fontsize=10, color="grey")
    ax_r.tick_params(axis="y", labelsize=9, colors="grey")
    ax.grid(axis="y", alpha=0.3)

    # --- Panel 3: scatter of large cell width vs height ---
    ax = axes[2]
    cmap = plt.cm.tab10
    well_colors = {w: cmap(i % cmap.N) for i, w in enumerate(wells)}
    for w in wells:
        sub = large[large["well"] == w]
        if sub.empty:
            continue
        lbl = w.replace("/", "")
        ax.scatter(sub["bbox_width"] * _UM_PER_PX, sub["bbox_height"] * _UM_PER_PX,
                   s=15, alpha=0.5, color=well_colors[w], label=f"{lbl} (n={len(sub)})")
    # Square reference line
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], color="grey", linestyle="--", alpha=0.5, label="Square")
    ax.set_xlabel("BBox Width (µm)")
    ax.set_ylabel("BBox Height (µm)")
    ax.set_title("Large Cell Width vs Height")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    n_large = len(large)
    n_total = len(all_cells_df)
    fig.suptitle(
        f"Large Cell Analysis — {experiment}\n"
        f"{n_large:,} / {n_total:,} cells ({100*n_large/n_total:.1f}%) exceed "
        f"{_LARGE_CELL_THRESHOLD_PX}px ({_px_to_um(_LARGE_CELL_THRESHOLD_PX):.0f} µm) threshold",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(out_dir / "large_cell_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[cell_size] Large cell plot saved to {out_dir / 'large_cell_analysis.png'}")


def _discover_wells(dataset: OpsDataset) -> list[tuple[str, Path]]:
    """Discover wells by globbing linked_results files.

    Returns list of (well_path, linked_csv_path) tuples.
    """
    link_dir = dataset.results_fast
    link_files = sorted(link_dir.glob("*_linked_pheno_iss.csv"))
    if not link_files:
        link_dir = dataset.results
        link_files = sorted(link_dir.glob("*_linked_pheno_iss.csv"))

    wells = []
    for f in link_files:
        # e.g. "A1_linked_pheno_iss.csv" -> "A1"
        token = f.name.split("_linked_pheno_iss.csv")[0]
        # Convert token to well path: "A1" -> "A/1/0"
        if len(token) >= 2:
            well_path = f"{token[0]}/{token[1:]}/0"
            wells.append((well_path, f))
    return wells


def get_cell_size_stats_for_well(
    dataset: OpsDataset, well: str, force: bool = False
) -> dict:
    """Read per-well cell sizes CSV and return summary stats dict for plate_stats.

    Returns dict with keys like cell_seg_bbox_area_mean, etc.
    """
    csv_path = dataset.append_well("cell_sizes", well)
    if not Path(str(csv_path)).exists():
        return {}

    df = pd.read_csv(csv_path)
    if df.empty:
        return {}

    result = {"cell_seg_count": len(df)}
    for col in _SHAPE_COLUMNS:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        prefix = f"cell_seg_{col}"
        result[f"{prefix}_mean"] = float(series.mean()) if len(series) else None
        result[f"{prefix}_median"] = float(series.median()) if len(series) else None
        result[f"{prefix}_min"] = float(series.min()) if len(series) else None
        result[f"{prefix}_max"] = float(series.max()) if len(series) else None
        result[f"{prefix}_std"] = float(series.std()) if len(series) else None

    if "is_large" in df.columns:
        n_large = int(df["is_large"].sum())
        result["cell_seg_large_cell_count"] = n_large
        result["cell_seg_large_cell_pct"] = (
            n_large / len(df) * 100 if len(df) else 0.0
        )

    return result


def cell_size_metrics(
    experiment: str,
    method: str = "mine",
    force: bool = False,
) -> dict:
    """
    Compute per-cell bbox metrics for all wells in an experiment.

    Reads bbox from the linked CSV — no segmentation loading required.

    Args:
        experiment: Experiment name/identifier.
        method: Base calling method ('mine' or 'probabilistic').
        force: If True, regenerate even if outputs exist.

    Returns:
        Dict mapping well -> summary stats dict (for plate_stats integration).
    """
    dataset = OpsDataset(experiment, method=method)

    # Check cache
    summary_path = dataset.metrics_paths["cell_size_summary"]
    plot_path = dataset.metrics_paths["cell_size_distribution"]

    if summary_path.exists() and not force:
        print(f"[cell_size] Summary exists: {summary_path}")
        wells = _discover_wells(dataset)

        # Replot from cached CSVs if the plot is missing
        if not plot_path.exists():
            print("[cell_size] Plot missing, regenerating from cached CSVs...")
            all_cells = []
            for well_path, _ in wells:
                csv_path = dataset.append_well("cell_sizes", well_path)
                if Path(str(csv_path)).exists():
                    all_cells.append(pd.read_csv(csv_path))
            if all_cells:
                _plot_distributions(
                    pd.concat(all_cells, ignore_index=True), plot_path, experiment
                )
                print(f"[cell_size] Plot saved to {plot_path}")
        else:
            print("[cell_size] Skipping. Use force=True to regenerate.")

        return {w: get_cell_size_stats_for_well(dataset, w) for w, _ in wells}

    # Create output directory for per-cell CSVs
    cell_sizes_dir = dataset.results / "cell_sizes"
    cell_sizes_dir.mkdir(parents=True, exist_ok=True)

    # Discover wells
    wells = _discover_wells(dataset)
    if not wells:
        print("[cell_size] No linked_results found, skipping cell size metrics")
        return {}

    print(f"[cell_size] Processing {len(wells)} wells")

    all_cells = []
    well_stats = {}
    for well_path, linked_csv_path in wells:
        linked_df = pd.read_csv(linked_csv_path)
        if linked_df.empty:
            print(f"[cell_size] WARNING: Empty linked CSV for {well_path}")
            continue

        cell_df = _compute_cell_props_from_linked(linked_df, well_path)
        if cell_df.empty:
            print(f"[cell_size] WARNING: No valid bboxes for {well_path}")
            continue

        # Save per-well CSV
        well_csv_path = dataset.append_well("cell_sizes", well_path)
        Path(str(well_csv_path)).parent.mkdir(parents=True, exist_ok=True)
        cell_df.to_csv(well_csv_path, index=False)
        print(f"[cell_size] {well_path}: {len(cell_df)} cells -> {well_csv_path}")

        all_cells.append(cell_df)
        well_stats[well_path] = get_cell_size_stats_for_well(dataset, well_path)

    if not all_cells:
        print("[cell_size] No cells found across any well.")
        return {}

    all_cells_df = pd.concat(all_cells, ignore_index=True)

    # Summary stats
    summary_df = _compute_summary(all_cells_df)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"[cell_size] Summary saved to {summary_path}")

    # Distribution plots
    _plot_distributions(all_cells_df, plot_path, experiment)
    print(f"[cell_size] Plot saved to {plot_path}")

    return well_stats


if __name__ == "__main__":
    import argparse
    from cyclops_utils.data.filesystem import resolve_experiment_name

    parser = argparse.ArgumentParser(
        description="Compute cell size/shape metrics from linked CSV bboxes"
    )
    parser.add_argument(
        "exp",
        type=str,
        nargs="?",
        default=None,
        help="Experiment name or shorthand (e.g. '33', 'ops33', 'ops0033_20250429')",
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        default=None,
        help="Experiment name (alternative to positional arg)",
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="mine",
        choices=["mine", "probabilistic"],
        help="Base calling method (default: mine)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force regeneration even if outputs exist",
    )

    args = parser.parse_args()
    exp_input = args.exp or args.experiment
    if not exp_input:
        parser.error("experiment is required (positional or via -e/--experiment)")
    exp = resolve_experiment_name(exp_input, autoselect=True)
    print(f"Processing experiment: {exp}")
    cell_size_metrics(exp, method=args.method, force=args.force)
