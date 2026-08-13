import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
from tqdm import tqdm
import argparse
import yaml
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from matplotlib.ticker import ScalarFormatter
from ops_utils.io.tiling import split_into_tiles
from ops_utils.data.experiment import OpsDataset
from iohub.ngff import open_ome_zarr
import dask.array as da
from cyclops_process.paths import BASE_PATH


# usage: python -m cyclops_process.metrics.metrics_tracking ops0113_20251219 --verbose --force

def plot_tracking_metrics_over_time(
    reference_experiment: str,
    verbose: bool = False,
    force: bool = False,
):
    """
    Analyze linked tracking data across all experiments and generate over-time plots.

    This function loads linked_pheno_iss.csv files from all experiments and generates
    time-series plots showing:
    1. Total cells per experiment
    2. Cells per well (with individual well dots)
    3. Mean cells per gene per experiment
    4. Max cells per gene per experiment
    5. Growth effect correlation metrics over time

    Args:
        reference_experiment: Reference experiment name for determining config directory
        verbose: Print detailed progress information
    """
    dataset = OpsDataset(reference_experiment, method=None)

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    # Get all experiment configs
    config_dir = dataset.config_paths["exp_config_dir"]
    config_files = sorted(list(config_dir.glob("ops*.yaml")))

    # List of experiments to exclude
    exp_to_exclude = [
        "ops0012_20250206"
        "ops0080_20250904",
        "ops0079_rnd10_cardboard",
        "ops0079_black2pt0",
        "ops0079_black_oracal",
        "ops0079_blk_anodized",
        "ops0079_delrin",
        "ops0079_milled_acry",
        "ops0079_orig_cap",
        "ops0079_PETG",
        "ops0079_PLA",
    ]

    # Exclude experiments
    if exp_to_exclude:
        excluded_prefixes = {f"{exp}_" for exp in exp_to_exclude}
        config_files = [
            cf
            for cf in config_files
            if not any(cf.name.startswith(pref) for pref in excluded_prefixes)
        ]

    _log(f"Found {len(config_files)} experiment configs. Loading tracking data...")

    # Data structures to hold metrics
    experiment_data = {}
    skipped_count = 0

    for config_file in tqdm(config_files, desc="Processing experiments"):
        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)
            exp_name = config_data.get("experiment_name")

        if not exp_name:
            _log(f"--> Skipping config file {config_file.name}: no 'experiment_name'.")
            continue

        exp_dataset = OpsDataset(exp_name, method="mine")

        # Try to load linked results for this experiment
        wells = exp_dataset.infer_wells()
        all_well_data = []
        well_counts = {}

        for well in wells:
            linked_file = exp_dataset.append_well("linked_results", well)
            if linked_file.exists():
                try:
                    df = pd.read_csv(linked_file)
                    if not df.empty:
                        df["well"] = well
                        all_well_data.append(df)
                        well_counts[well] = len(df)
                except Exception as e:
                    _log(f"Could not load {linked_file}: {e}")

        if not all_well_data:
            skipped_count += 1
            continue

        # Combine all wells for this experiment
        combined_df = pd.concat(all_well_data, ignore_index=True)

        # Get ISS cell count from plate_stats.csv (cells with matched reads)
        iss_cell_count = 0
        well_iss_counts = {}
        try:
            plate_stats_path = exp_dataset.metrics_paths["statistics"]
            if plate_stats_path.exists():
                plate_stats = pd.read_csv(plate_stats_path, index_col=0)
                # Sum cells_with_matched_reads across all columns (wells)
                if "cells_with_matched_reads" in plate_stats.index:
                    row_values = plate_stats.loc["cells_with_matched_reads"]
                    iss_cell_count = int(row_values.sum())
                    # Store per-well ISS counts
                    for col in plate_stats.columns:
                        well_iss_counts[col] = int(row_values[col]) if pd.notna(row_values[col]) else 0
        except Exception as e:
            if verbose:
                _log(f"Could not load ISS cell count for {exp_name}: {e}")
            iss_cell_count = 0

        # Calculate metrics
        metrics = {
            "experiment": exp_name,
            "total_cells": len(combined_df),  # post-tracking
            "total_cells_iss": iss_cell_count,  # post-ISS segmentation
            "well_counts": well_counts,
            "well_iss_counts": well_iss_counts,
            "num_wells": len(well_counts),
        }

        # Mean and max cells per gene
        if "dep_map_gene_name" in combined_df.columns:
            gene_col = "dep_map_gene_name"
        elif "Gene name" in combined_df.columns:
            gene_col = "Gene name"
        else:
            gene_col = None

        if gene_col:
            cells_per_gene = combined_df.groupby(gene_col).size()
            metrics["mean_cells_per_gene"] = cells_per_gene.mean()
            metrics["max_cells_per_gene"] = cells_per_gene.max()
            metrics["median_cells_per_gene"] = cells_per_gene.median()
        else:
            metrics["mean_cells_per_gene"] = np.nan
            metrics["max_cells_per_gene"] = np.nan
            metrics["median_cells_per_gene"] = np.nan

        # Growth effect analysis (if gene_effect column exists)
        if "gene_effect" in combined_df.columns:
            growth_stats = _calculate_growth_effect_stats(combined_df, gene_col)
            metrics.update(growth_stats)
        else:
            metrics["growth_effect_slope"] = np.nan
            metrics["growth_effect_r2"] = np.nan
            metrics["growth_effect_fc"] = np.nan

        experiment_data[exp_name] = metrics

    if not experiment_data:
        print("No tracking data found across any experiments.")
        return

    # Sort experiments by date
    def extract_date(exp_name: str) -> str:
        """Extract date suffix from experiment name (e.g., 'ops0061_20250728' -> '20250728')"""
        match = re.search(r"_(\d{8})$", exp_name)
        return match.group(1) if match else exp_name

    experiments_sorted = sorted(experiment_data.keys(), key=extract_date)

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    output_dir = Path(f"{BASE_PATH}/ops_data_report/{timestamp}")
    output_dir.mkdir(exist_ok=True, parents=True)

    _log(f"Generating plots in {output_dir}")

    # Generate all over-time plots
    plots_to_generate = [
        ("total_cells.png", lambda: _plot_total_cells(experiment_data, experiments_sorted, output_dir, reference_experiment)),
        ("cells_per_well.png", lambda: _plot_cells_per_well(experiment_data, experiments_sorted, output_dir, reference_experiment)),
        ("mean_cells_per_gene.png", lambda: _plot_mean_cells_per_gene(experiment_data, experiments_sorted, output_dir, reference_experiment)),
        ("max_cells_per_gene.png", lambda: _plot_max_cells_per_gene(experiment_data, experiments_sorted, output_dir, reference_experiment)),
        ("growth_effect_metrics.png", lambda: _plot_growth_effect_metrics(experiment_data, experiments_sorted, output_dir, reference_experiment)),
        ("growth_effect_fc_vs_total_cells.png", lambda: _plot_growth_effect_vs_cells(experiment_data, experiments_sorted, output_dir, reference_experiment)),
        ("cell_loss_iss_to_tracking.png", lambda: _plot_cell_loss_iss_to_tracking(experiment_data, experiments_sorted, output_dir, reference_experiment)),
        ("cell_loss_per_well.png", lambda: _plot_cell_loss_per_well(experiment_data, experiments_sorted, output_dir, reference_experiment)),
    ]

    for plot_file, plot_func in plots_to_generate:
        plot_path = output_dir / plot_file
        if force or not plot_path.exists():
            plot_func()

    # Generate per-experiment gene histograms
    _log("Generating per-experiment gene count histograms...")
    gene_hist_dir = output_dir / "gene_hist"
    gene_hist_dir.mkdir(exist_ok=True, parents=True)

    for exp_name in tqdm(experiments_sorted, desc="Creating gene histograms"):
        gene_hist_path = gene_hist_dir / f"{exp_name}_gene_counts.png"
        if force or not gene_hist_path.exists():
            _generate_gene_histogram(exp_name, gene_hist_dir, _log)

    # Generate per-experiment well heatmaps
    _log("Generating per-experiment tracked cell count heatmaps...")
    well_heatmap_dir = output_dir / "well_heatmaps"
    well_heatmap_dir.mkdir(exist_ok=True, parents=True)

    for exp_name in tqdm(experiments_sorted, desc="Creating well heatmaps"):
        well_heatmap_path = well_heatmap_dir / f"{exp_name}_well_heatmap.png"
        if force or not well_heatmap_path.exists():
            _generate_well_heatmap(exp_name, well_heatmap_dir, _log)

    # Generate per-experiment growth effect plots
    _log("Generating per-experiment growth effect plots...")
    growth_sum_dir = output_dir / "growth_effect_sum_well"
    growth_per_dir = output_dir / "growth_effect_per_well"
    growth_sum_dir.mkdir(exist_ok=True, parents=True)
    growth_per_dir.mkdir(exist_ok=True, parents=True)

    for exp_name in tqdm(experiments_sorted, desc="Creating growth effect plots"):
        growth_sum_path = growth_sum_dir / f"{exp_name}_growth_effect_sum_well.png"
        # Note: per-well plots have multiple files, so we check the sum plot as proxy
        if force or not growth_sum_path.exists():
            _generate_growth_effect_plots(exp_name, growth_sum_dir, growth_per_dir, _log)

    # Generate per-experiment normalized tracking and loss heatmaps (combined for efficiency)
    _log("Generating per-experiment normalized tracking and tracking loss heatmaps...")
    norm_tracking_dir = output_dir / "normalized_tracking_heatmaps"
    tracking_loss_dir = output_dir / "tracking_loss_heatmaps"
    norm_tracking_dir.mkdir(exist_ok=True, parents=True)
    tracking_loss_dir.mkdir(exist_ok=True, parents=True)

    for exp_name in tqdm(experiments_sorted, desc="Creating tracking heatmaps"):
        norm_tracking_path = norm_tracking_dir / f"{exp_name}_normalized_tracking.png"
        tracking_loss_path = tracking_loss_dir / f"{exp_name}_tracking_loss.png"

        # Generate both heatmaps in one pass if either is missing
        if force or not norm_tracking_path.exists() or not tracking_loss_path.exists():
            _generate_tracking_heatmaps_combined(exp_name, norm_tracking_dir, tracking_loss_dir, _log)

    # Save summary CSV
    summary_df = pd.DataFrame.from_dict(experiment_data, orient="index")
    summary_df = summary_df.reindex(experiments_sorted)
    summary_df.to_csv(output_dir / "tracking_metrics_summary.csv")

    print(f"\n[metrics_tracking] Processed {len(config_files)} configs: {len(experiments_sorted)} with data, {skipped_count} skipped")
    print(f"[metrics_tracking] Output directory: {output_dir}")


def _generate_gene_histogram(exp_name: str, output_dir: Path, log_func) -> None:
    """
    Generate a histogram of top 50 genes by cell count for a single experiment.

    Args:
        exp_name: Experiment name
        output_dir: Directory to save histogram
        log_func: Logging function
    """
    try:
        exp_dataset = OpsDataset(exp_name, method="mine")
        wells = exp_dataset.infer_wells()

        # Load all linked results
        all_well_data = []
        for well in wells:
            linked_file = exp_dataset.append_well("linked_results", well)
            if linked_file.exists():
                try:
                    df = pd.read_csv(linked_file)
                    if not df.empty:
                        all_well_data.append(df)
                except Exception:
                    continue

        if not all_well_data:
            log_func(f"No data for {exp_name}, skipping histogram")
            return

        # Combine all wells
        combined_df = pd.concat(all_well_data, ignore_index=True)

        # Determine gene column
        if "dep_map_gene_name" in combined_df.columns:
            gene_col = "dep_map_gene_name"
        elif "Gene name" in combined_df.columns:
            gene_col = "Gene name"
        else:
            log_func(f"No gene column found for {exp_name}, skipping histogram")
            return

        # Count cells per gene (using segmentation_id as proxy for unique cells)
        if "segmentation_id" in combined_df.columns:
            cells_per_gene = combined_df.groupby(gene_col)["segmentation_id"].nunique()
        else:
            # Fallback to counting rows
            cells_per_gene = combined_df.groupby(gene_col).size()

        if cells_per_gene.empty:
            log_func(f"No gene data for {exp_name}, skipping histogram")
            return

        # Get top 50 genes
        top_50 = cells_per_gene.nlargest(50).sort_values(ascending=False)

        # Create bar plot
        fig, ax = plt.subplots(figsize=(16, 8))
        x_pos = np.arange(len(top_50))
        ax.bar(x_pos, top_50.values, color="steelblue", alpha=0.7)

        ax.set_xlabel("Gene", fontsize=14)
        ax.set_ylabel("Number of Cells", fontsize=14)
        ax.set_title(f"Top 50 Genes by Cell Count - {exp_name}", fontsize=16)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(top_50.index, rotation=90, ha="right", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        # Add summary stats
        mean_val = top_50.mean()
        median_val = top_50.median()
        ax.axhline(mean_val, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label=f"Mean: {mean_val:.1f}")
        ax.axhline(median_val, color="green", linestyle="--", linewidth=1.5, alpha=0.7, label=f"Median: {median_val:.1f}")
        ax.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(output_dir / f"{exp_name}_gene_counts.png", dpi=300, bbox_inches="tight")
        plt.close()

    except Exception as e:
        log_func(f"Error generating histogram for {exp_name}: {e}")


def _generate_well_heatmap(exp_name: str, output_dir: Path, log_func) -> None:
    """
    Generate a well heatmap showing tracked cell counts for each well in an experiment.

    Args:
        exp_name: Experiment name
        output_dir: Directory to save heatmap
        log_func: Logging function
    """
    try:
        exp_dataset = OpsDataset(exp_name, method="mine")
        wells = exp_dataset.infer_wells()

        if not wells:
            log_func(f"No wells found for {exp_name}, skipping heatmap")
            return

        num_wells = len(wells)
        fig, axes = plt.subplots(1, num_wells, figsize=(15, 5) if num_wells > 1 else (6, 5))
        if num_wells == 1:
            axes = [axes]

        well_data_loaded = False

        for idx, well in enumerate(wells):
            ax = axes[idx]
            linked_file = exp_dataset.append_well("linked_results", well)

            if not linked_file.exists():
                ax.text(0.5, 0.5, f"No data\n{well}", ha="center", va="center", fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(well)
                continue

            try:
                df = pd.read_csv(linked_file)
                if df.empty:
                    ax.text(0.5, 0.5, f"Empty\n{well}", ha="center", va="center", fontsize=12)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(well)
                    continue

                # Check for required coordinate columns
                if "y_pheno" not in df.columns or "x_pheno" not in df.columns:
                    ax.text(0.5, 0.5, f"Missing coords\n{well}", ha="center", va="center", fontsize=10)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(well)
                    continue

                # Get shape from coordinates
                y_coords = df["y_pheno"].values
                x_coords = df["x_pheno"].values

                if len(y_coords) == 0 or len(x_coords) == 0:
                    ax.text(0.5, 0.5, f"No coords\n{well}", ha="center", va="center", fontsize=12)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(well)
                    continue

                # Determine shape from max coordinates
                shape = (int(np.max(y_coords)) + 100, int(np.max(x_coords)) + 100)

                # Split into tiles (30x30 grid like in metrics.py)
                tile_list, indx = split_into_tiles(shape, 30, 0)

                # Count cells per tile
                cell_counts = []
                for tile in tile_list:
                    y_min, y_max, x_min, x_max = tile

                    # Count unique cells (using segmentation_id if available)
                    if "segmentation_id" in df.columns:
                        cells_in_tile = df[
                            (df["y_pheno"] >= y_min)
                            & (df["y_pheno"] < y_max)
                            & (df["x_pheno"] >= x_min)
                            & (df["x_pheno"] < x_max)
                        ]["segmentation_id"].nunique()
                    else:
                        # Fallback: count rows
                        cells_in_tile = len(df[
                            (df["y_pheno"] >= y_min)
                            & (df["y_pheno"] < y_max)
                            & (df["x_pheno"] >= x_min)
                            & (df["x_pheno"] < x_max)
                        ])

                    cell_counts.append(cells_in_tile)

                # Create heatmap array
                indx_i = [a[0] for a in indx]
                indx_j = [a[1] for a in indx]
                n = int(np.sqrt(len(indx_i)))
                heatmap_array = np.zeros((n, n))
                heatmap_array[indx_i, indx_j] = cell_counts

                # Plot heatmap
                im = ax.imshow(heatmap_array, cmap="viridis", interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(well, fontsize=12)

                well_data_loaded = True

            except Exception as e:
                ax.text(0.5, 0.5, f"Error\n{well}", ha="center", va="center", fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(well)
                log_func(f"Error processing well {well} for {exp_name}: {e}")
                continue

        if not well_data_loaded:
            plt.close(fig)
            log_func(f"No valid well data for {exp_name}, skipping heatmap")
            return

        # Add colorbar to the right side without overlapping
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(im, cax=cbar_ax, label="Tracked cells per tile")
        fig.suptitle(f"Tracked Cell Counts per Well - {exp_name}", fontsize=16)

        plt.savefig(output_dir / f"{exp_name}_well_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

    except Exception as e:
        log_func(f"Error generating well heatmap for {exp_name}: {e}")


def _generate_growth_effect_plots(exp_name: str, sum_dir: Path, per_dir: Path, log_func) -> None:
    """
    Generate growth effect plots for an experiment (pooled and per-well).

    Args:
        exp_name: Experiment name
        sum_dir: Directory for pooled (sum of wells) plots
        per_dir: Directory for per-well plots
        log_func: Logging function
    """
    try:
        exp_dataset = OpsDataset(exp_name, method="mine")
        wells = exp_dataset.infer_wells()

        # --- Pooled Plot (all wells combined) ---
        all_well_data = []
        for well in wells:
            linked_file = exp_dataset.append_well("linked_results", well)
            if linked_file.exists():
                try:
                    df = pd.read_csv(linked_file)
                    if not df.empty:
                        all_well_data.append(df)
                except Exception:
                    continue

        if all_well_data:
            combined_df = pd.concat(all_well_data, ignore_index=True)

            # Check if gene_effect column exists
            if "gene_effect" not in combined_df.columns:
                log_func(f"No gene_effect column in linked data for {exp_name}")
                return

            # Determine gene column
            if "dep_map_gene_name" in combined_df.columns:
                gene_col = "dep_map_gene_name"
            elif "Gene name" in combined_df.columns:
                gene_col = "Gene name"
            else:
                log_func(f"No gene column found for {exp_name}")
                return

            # Count cells per gene (using segmentation_id if available)
            if "segmentation_id" in combined_df.columns:
                cells_per_gene = combined_df.groupby(gene_col)["segmentation_id"].nunique().reset_index(name="count")
            else:
                cells_per_gene = combined_df.groupby(gene_col).size().reset_index(name="count")

            # Merge with gene_effect and NCBI_ID from the linked data itself
            gene_info = combined_df[[gene_col, "gene_effect"]].drop_duplicates()
            if "NCBI_ID" in combined_df.columns:
                gene_info = combined_df[[gene_col, "gene_effect", "NCBI_ID"]].drop_duplicates()

            merged_df = pd.merge(cells_per_gene, gene_info, on=gene_col, how="left")

            # Generate pooled plot
            fig, ax = plt.subplots(figsize=(6, 4))
            _plot_growth_effect_helper(ax, merged_df, f"{exp_name}")
            plt.tight_layout()
            plt.savefig(sum_dir / f"{exp_name}_growth_effect.png", dpi=300, bbox_inches="tight")
            plt.close()

        # --- Per-Well Plots ---
        if not wells:
            return

        num_wells = len(wells)
        fig, axes = plt.subplots(1, num_wells, figsize=(5 * num_wells, 4), sharey=True)
        if num_wells == 1:
            axes = [axes]

        # First pass: collect all data to determine global y-axis limits
        well_data_list = []
        global_max_y = 0

        for well in wells:
            linked_file = exp_dataset.append_well("linked_results", well)
            if not linked_file.exists():
                well_data_list.append(None)
                continue

            try:
                df = pd.read_csv(linked_file)
                if df.empty or "gene_effect" not in df.columns:
                    well_data_list.append(None)
                    continue

                # Determine gene column
                if "dep_map_gene_name" in df.columns:
                    gene_col = "dep_map_gene_name"
                elif "Gene name" in df.columns:
                    gene_col = "Gene name"
                else:
                    well_data_list.append(None)
                    continue

                # Count cells per gene
                if "segmentation_id" in df.columns:
                    cells_per_gene = df.groupby(gene_col)["segmentation_id"].nunique().reset_index(name="count")
                else:
                    cells_per_gene = df.groupby(gene_col).size().reset_index(name="count")

                # Merge with gene_effect and NCBI_ID
                gene_info = df[[gene_col, "gene_effect"]].drop_duplicates()
                if "NCBI_ID" in df.columns:
                    gene_info = df[[gene_col, "gene_effect", "NCBI_ID"]].drop_duplicates()

                merged_df = pd.merge(cells_per_gene, gene_info, on=gene_col, how="left")
                well_data_list.append(merged_df)

                # Update global max
                if not merged_df.empty and "count" in merged_df.columns:
                    well_max = merged_df["count"].quantile(0.90) * 1.4
                    global_max_y = max(global_max_y, well_max)

            except Exception:
                well_data_list.append(None)

        # Second pass: plot with consistent y-axis
        for i, well in enumerate(wells):
            ax = axes[i]
            merged_df = well_data_list[i]

            if merged_df is None:
                ax.text(0.5, 0.5, f"No data\n{well}", ha="center", va="center", fontsize=12)
                ax.set_title(well)
                continue

            try:
                _plot_growth_effect_helper(ax, merged_df, well, ylim=global_max_y if global_max_y > 0 else None)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error\n{well}", ha="center", va="center", fontsize=12)
                ax.set_title(well)
                log_func(f"Error plotting well {well} for {exp_name}: {e}")

        fig.suptitle(f"Per-Well Growth Effect - {exp_name}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(per_dir / f"{exp_name}_growth_effect_per_well.png", dpi=300, bbox_inches="tight")
        plt.close()

    except Exception as e:
        log_func(f"Error generating growth effect plots for {exp_name}: {e}")


def _plot_growth_effect_helper(ax, merged_df, title, ylim=None):
    """
    Helper function to generate growth effect scatter plot (adapted from metrics.py).

    Args:
        ax: Matplotlib axis
        merged_df: DataFrame with 'gene_effect' and 'count' columns
        title: Plot title
        ylim: Optional y-axis upper limit (if None, auto-calculate from data)
    """
    x = merged_df["gene_effect"].dropna()
    y = merged_df["count"].loc[x.index]

    if x.empty or y.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(title, fontsize=10)
        return

    # Scatter plot
    ax.scatter(x, y, s=5, alpha=0.4)
    ax.set_xlabel("Growth effect (CERES score)", fontsize=10)
    ax.set_ylabel("Cell count", fontsize=10)
    ax.set_title(title, fontsize=10)

    # Overlay NTC guides
    if "NCBI_ID" in merged_df.columns:
        ntc = merged_df[merged_df["NCBI_ID"] == -1]
        if not ntc.empty:
            ntc_x = ntc["gene_effect"]
            ntc_y = ntc["count"]
            ax.scatter(ntc_x, ntc_y, s=5, alpha=0.4, color="orange", label="NTC")

            ax.axhline(y=ntc_y.median(), color="orange", linestyle=":", linewidth=1, alpha=0.7)
            ax.axhline(y=ntc_y.quantile(0.9), color="orange", linestyle=":", linewidth=0.5, alpha=0.5)
            ax.axhline(y=ntc_y.quantile(0.1), color="orange", linestyle=":", linewidth=0.5, alpha=0.5)

    # Set y-limit
    if ylim is not None and ylim > 0:
        ax.set_ylim(0, ylim)
    elif not y.empty:
        ax.set_ylim(0, y.quantile(0.90) * 1.4)

    # Bin data and plot median
    try:
        bins = np.linspace(x.min(), x.max(), 10)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        df_temp = pd.DataFrame({"x": x, "y": y})
        df_temp["bin"] = pd.cut(df_temp["x"], bins=bins)
        bin_medians = df_temp.groupby("bin", observed=False)["y"].median()

        ax.plot(
            bin_centers, bin_medians, color="red", marker="o",
            linestyle="-", linewidth=1.5, markersize=3, label="Binned Median"
        )

        # Linear regression on binned medians (gene_effect <= 0.5)
        valid = ~np.isnan(bin_medians) & (bin_centers <= 0.5)
        if valid.sum() > 1:
            bin_centers_valid = bin_centers[valid].reshape(-1, 1)
            bin_medians_valid = bin_medians[valid]

            model = LinearRegression()
            model.fit(bin_centers_valid, bin_medians_valid)
            y_pred = model.predict(bin_centers_valid)

            slope = model.coef_[0]
            r2 = r2_score(bin_medians_valid, y_pred)

            x_fit = np.linspace(bin_centers_valid.min(), bin_centers_valid.max(), 100).reshape(-1, 1)
            y_fit = model.predict(x_fit)
            ax.plot(x_fit, y_fit, color="black", linestyle=":", linewidth=1)

            # Annotation
            annotation_text = f"Slope: {slope:.1f}\n$R^2$: {r2:.2f}"
            ax.text(
                0.95, 0.05, annotation_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="bottom", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            )
    except Exception:
        pass

    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.3)


def _calculate_growth_effect_stats(df: pd.DataFrame, gene_col: str) -> dict:
    """
    Calculate growth effect regression statistics similar to metrics.py.

    Returns dict with slope, r2, and fold_change.
    """
    if gene_col is None or "gene_effect" not in df.columns:
        return {
            "growth_effect_slope": np.nan,
            "growth_effect_r2": np.nan,
            "growth_effect_fc": np.nan,
        }

    # Group by gene and count cells
    gene_counts = df.groupby(gene_col).size().reset_index(name="count")

    # Merge with gene_effect
    if "gene_effect" in df.columns:
        gene_effects = df[[gene_col, "gene_effect"]].drop_duplicates()
        merged = pd.merge(gene_counts, gene_effects, on=gene_col, how="left")
    else:
        return {
            "growth_effect_slope": np.nan,
            "growth_effect_r2": np.nan,
            "growth_effect_fc": np.nan,
        }

    # Drop NaN values
    merged = merged.dropna(subset=["gene_effect", "count"])

    if len(merged) < 2:
        return {
            "growth_effect_slope": np.nan,
            "growth_effect_r2": np.nan,
            "growth_effect_fc": np.nan,
        }

    # Bin the data and calculate median per bin
    x = merged["gene_effect"].values
    y = merged["count"].values

    try:
        bins = np.linspace(x.min(), x.max(), 10)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        df_binned = pd.DataFrame({"x": x, "y": y})
        df_binned["bin"] = pd.cut(df_binned["x"], bins=bins)
        bin_medians = df_binned.groupby("bin", observed=False)["y"].median()

        # Linear regression on binned medians (only for gene_effect <= 0.5)
        valid = ~np.isnan(bin_medians) & (bin_centers <= 0.5)

        if valid.sum() > 1:
            bin_centers_valid = bin_centers[valid].reshape(-1, 1)
            bin_medians_valid = bin_medians[valid]

            model = LinearRegression()
            model.fit(bin_centers_valid, bin_medians_valid)
            y_pred = model.predict(bin_centers_valid)

            slope = model.coef_[0]
            r2 = r2_score(bin_medians_valid, y_pred)

            # Calculate fold change
            x_fit = np.linspace(bin_centers_valid.min(), bin_centers_valid.max(), 100).reshape(-1, 1)
            y_fit = model.predict(x_fit)
            fold_change = y_fit[-1] / y_fit[0] if y_fit[0] != 0 else 0

            return {
                "growth_effect_slope": float(slope),
                "growth_effect_r2": float(r2),
                "growth_effect_fc": float(fold_change),
            }
    except Exception:
        pass

    return {
        "growth_effect_slope": np.nan,
        "growth_effect_r2": np.nan,
        "growth_effect_fc": np.nan,
    }


def _generate_tracking_heatmaps_combined(exp_name: str, norm_output_dir: Path, loss_output_dir: Path, log_func) -> None:
    """
    Generate both normalized tracking and tracking loss heatmaps in a single pass for efficiency.

    Args:
        exp_name: Experiment name
        norm_output_dir: Directory to save normalized tracking heatmap
        loss_output_dir: Directory to save tracking loss heatmap
        log_func: Logging function
    """
    try:
        exp_dataset = OpsDataset(exp_name, method="mine")

        # Open ISS segmentation store and get positions
        iss_seg_store = open_ome_zarr(exp_dataset.store_paths["iss_segmentation"])
        pos_list = [a[0] for a in iss_seg_store.positions()]

        if not pos_list:
            log_func(f"No positions found for {exp_name}, skipping tracking heatmaps")
            return

        num_wells = len(pos_list)

        # Create figures for both heatmaps
        fig_norm, axes_norm = plt.subplots(1, num_wells, figsize=(15, 5) if num_wells > 1 else (6, 5))
        fig_loss, axes_loss = plt.subplots(1, num_wells, figsize=(15, 5) if num_wells > 1 else (6, 5))
        if num_wells == 1:
            axes_norm = [axes_norm]
            axes_loss = [axes_loss]

        norm_data_loaded = False
        loss_data_loaded = False

        for idx, pos in enumerate(pos_list):
            ax_norm = axes_norm[idx]
            ax_loss = axes_loss[idx]
            linked_file = exp_dataset.append_well("linked_results", pos)

            if not linked_file.exists():
                for ax in [ax_norm, ax_loss]:
                    ax.text(0.5, 0.5, f"No data\n{pos}", ha="center", va="center", fontsize=12)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(pos)
                continue

            try:
                # Load tracked cells data
                df = pd.read_csv(linked_file)
                if df.empty or "y_iss" not in df.columns or "x_iss" not in df.columns:
                    for ax in [ax_norm, ax_loss]:
                        ax.text(0.5, 0.5, f"Missing ISS coords\n{pos}", ha="center", va="center", fontsize=10)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_title(pos)
                    continue

                # Get ISS segmentation data and load into memory once
                iss_seg_data = da.array(iss_seg_store[pos].data[0, 0, 0, :, :])
                shape = iss_seg_data.shape[-2:]
                iss_seg_np = iss_seg_data.compute()  # Load once for both heatmaps

                # Split into tiles
                tile_list, indx = split_into_tiles(shape, 30, 0)

                # Prepare coordinate arrays - use ISS coordinates to match ISS segmentation
                y_coords = df["y_iss"].values
                x_coords = df["x_iss"].values
                seg_ids = df["segmentation_id"].values if "segmentation_id" in df.columns else None

                # Compute both metrics in one pass
                normalized_counts = []
                percent_loss_values = []

                for tile in tile_list:
                    y_min, y_max, x_min, x_max = tile

                    # Count total cells in tile (used by both metrics)
                    tile_labels = iss_seg_np[y_min:y_max, x_min:x_max]
                    total_cells = len(np.unique(tile_labels)) - 1

                    # Exclude tiles with less than 100 cells
                    if total_cells < 100:
                        normalized_counts.append(np.nan)
                        percent_loss_values.append(np.nan)
                        continue

                    # Count tracked cells in tile (used by both metrics)
                    in_tile = (y_coords >= y_min) & (y_coords < y_max) & (x_coords >= x_min) & (x_coords < x_max)
                    if seg_ids is not None:
                        tracked_cells = len(np.unique(seg_ids[in_tile]))
                    else:
                        tracked_cells = np.sum(in_tile)

                    # Calculate both metrics
                    normalized_counts.append(tracked_cells / total_cells)
                    percent_loss = 100 * (1 - tracked_cells / total_cells)
                    percent_loss_values.append(max(0, min(100, percent_loss)))

                # Renormalize the normalized_counts to 0-1 range (excluding NaN values)
                valid_norm_counts = [x for x in normalized_counts if not np.isnan(x)]
                if valid_norm_counts:
                    min_val = min(valid_norm_counts)
                    max_val = max(valid_norm_counts)
                    if max_val > min_val:
                        normalized_counts_rescaled = [
                            (x - min_val) / (max_val - min_val) if not np.isnan(x) else np.nan
                            for x in normalized_counts
                        ]
                    else:
                        normalized_counts_rescaled = normalized_counts
                else:
                    normalized_counts_rescaled = normalized_counts

                # Create heatmap arrays
                indx_i = [a[0] for a in indx]
                indx_j = [a[1] for a in indx]
                n = int(np.sqrt(len(indx_i)))

                # Normalized tracking heatmap
                heatmap_norm = np.full((n, n), np.nan)
                heatmap_norm[indx_i, indx_j] = normalized_counts_rescaled
                im_norm = ax_norm.imshow(heatmap_norm, cmap="viridis", interpolation="nearest", vmin=0, vmax=1)
                ax_norm.set_xticks([])
                ax_norm.set_yticks([])
                ax_norm.set_title(pos, fontsize=12)
                norm_data_loaded = True

                # Tracking loss heatmap
                heatmap_loss = np.full((n, n), np.nan)
                heatmap_loss[indx_i, indx_j] = percent_loss_values
                im_loss = ax_loss.imshow(heatmap_loss, cmap="Reds", interpolation="nearest", vmin=0, vmax=100)
                ax_loss.set_xticks([])
                ax_loss.set_yticks([])
                ax_loss.set_title(pos, fontsize=12)
                loss_data_loaded = True

            except Exception as e:
                for ax in [ax_norm, ax_loss]:
                    ax.text(0.5, 0.5, f"Error\n{pos}", ha="center", va="center", fontsize=12)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(pos)
                log_func(f"Error processing position {pos} for {exp_name}: {e}")
                continue

        # Save normalized tracking heatmap
        if norm_data_loaded:
            fig_norm.subplots_adjust(right=0.9)
            cbar_ax = fig_norm.add_axes([0.92, 0.15, 0.02, 0.7])
            fig_norm.colorbar(im_norm, cax=cbar_ax, label="Tracked / Total (rescaled 0-1)")
            fig_norm.suptitle(f"Normalized Tracking (Tracked/Total, ≥100 cells) - {exp_name}", fontsize=16)
            plt.figure(fig_norm.number)
            plt.savefig(norm_output_dir / f"{exp_name}_normalized_tracking.png", dpi=300, bbox_inches="tight")
        else:
            log_func(f"No valid data for {exp_name}, skipping normalized tracking heatmap")
        plt.close(fig_norm)

        # Save tracking loss heatmap
        if loss_data_loaded:
            fig_loss.subplots_adjust(right=0.9)
            cbar_ax = fig_loss.add_axes([0.92, 0.15, 0.02, 0.7])
            fig_loss.colorbar(im_loss, cax=cbar_ax, label="Cell Loss (%)")
            fig_loss.suptitle(f"Tracking Loss (ISS to Tracking, ≥100 cells) - {exp_name}", fontsize=16)
            plt.figure(fig_loss.number)
            plt.savefig(loss_output_dir / f"{exp_name}_tracking_loss.png", dpi=300, bbox_inches="tight")
        else:
            log_func(f"No valid data for {exp_name}, skipping tracking loss heatmap")
        plt.close(fig_loss)

    except Exception as e:
        log_func(f"Error generating tracking heatmaps for {exp_name}: {e}")


def _generate_normalized_tracking_heatmap(exp_name: str, output_dir: Path, log_func) -> None:
    """
    Generate a heatmap showing tracked cells normalized by total cell count (confluency) per tile.

    Args:
        exp_name: Experiment name
        output_dir: Directory to save heatmap
        log_func: Logging function
    """
    try:
        exp_dataset = OpsDataset(exp_name, method="mine")

        # Open ISS segmentation store and get positions
        iss_seg_store = open_ome_zarr(exp_dataset.store_paths["iss_segmentation"])
        pos_list = [a[0] for a in iss_seg_store.positions()]

        if not pos_list:
            log_func(f"No positions found for {exp_name}, skipping normalized tracking heatmap")
            return

        num_wells = len(pos_list)
        fig, axes = plt.subplots(1, num_wells, figsize=(15, 5) if num_wells > 1 else (6, 5))
        if num_wells == 1:
            axes = [axes]

        well_data_loaded = False

        for idx, pos in enumerate(pos_list):
            ax = axes[idx]
            linked_file = exp_dataset.append_well("linked_results", pos)

            if not linked_file.exists():
                ax.text(0.5, 0.5, f"No data\n{pos}", ha="center", va="center", fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(pos)
                continue

            try:
                # Load tracked cells data
                df = pd.read_csv(linked_file)
                if df.empty or "y_pheno" not in df.columns or "x_pheno" not in df.columns:
                    ax.text(0.5, 0.5, f"Missing coords\n{pos}", ha="center", va="center", fontsize=10)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(pos)
                    continue

                # Get ISS segmentation data for total cell counts and shape
                iss_seg_data = da.array(iss_seg_store[pos].data[0, 0, 0, :, :])
                shape = iss_seg_data.shape[-2:]

                # Load ISS data into memory once for this position (major speedup)
                iss_seg_np = iss_seg_data.compute()

                # Split into tiles (30x30 grid)
                tile_list, indx = split_into_tiles(shape, 30, 0)

                # Prepare coordinate arrays for vectorized operations
                y_coords = df["y_pheno"].values
                x_coords = df["x_pheno"].values
                seg_ids = df["segmentation_id"].values if "segmentation_id" in df.columns else None

                # Count tracked cells and total cells per tile
                normalized_counts = []
                for tile in tile_list:
                    y_min, y_max, x_min, x_max = tile

                    # Count total cells in tile from ISS segmentation (already in memory)
                    tile_labels = iss_seg_np[y_min:y_max, x_min:x_max]
                    total_cells = len(np.unique(tile_labels)) - 1

                    # Exclude tiles with less than 100 cells (insufficient data)
                    if total_cells < 100:
                        normalized_counts.append(np.nan)
                        continue

                    # Count tracked cells in tile using vectorized operations
                    in_tile = (y_coords >= y_min) & (y_coords < y_max) & (x_coords >= x_min) & (x_coords < x_max)

                    if seg_ids is not None:
                        tracked_cells = len(np.unique(seg_ids[in_tile]))
                    else:
                        tracked_cells = np.sum(in_tile)

                    # Calculate normalized ratio
                    normalized_counts.append(tracked_cells / total_cells)

                # Create heatmap array
                indx_i = [a[0] for a in indx]
                indx_j = [a[1] for a in indx]
                n = int(np.sqrt(len(indx_i)))
                heatmap_array = np.full((n, n), np.nan)
                heatmap_array[indx_i, indx_j] = normalized_counts

                # Plot heatmap (NaN tiles will appear as gray/masked)
                im = ax.imshow(heatmap_array, cmap="viridis", interpolation="nearest", vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(pos, fontsize=12)

                well_data_loaded = True

            except Exception as e:
                ax.text(0.5, 0.5, f"Error\n{pos}", ha="center", va="center", fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(pos)
                log_func(f"Error processing position {pos} for {exp_name}: {e}")
                continue

        if not well_data_loaded:
            plt.close(fig)
            log_func(f"No valid well data for {exp_name}, skipping normalized tracking heatmap")
            return

        # Add colorbar
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="Tracked cells / Total cells")
        fig.suptitle(f"Normalized Tracking (Tracked/Total) - {exp_name}", fontsize=16)

        plt.savefig(output_dir / f"{exp_name}_normalized_tracking.png", dpi=300, bbox_inches="tight")
        plt.close()

    except Exception as e:
        log_func(f"Error generating normalized tracking heatmap for {exp_name}: {e}")


def _generate_tracking_loss_heatmap(exp_name: str, output_dir: Path, log_func) -> None:
    """
    Generate a heatmap showing % cell loss from ISS to tracking per tile.

    Args:
        exp_name: Experiment name
        output_dir: Directory to save heatmap
        log_func: Logging function
    """
    try:
        exp_dataset = OpsDataset(exp_name, method="mine")

        # Open ISS segmentation store and get positions
        iss_seg_store = open_ome_zarr(exp_dataset.store_paths["iss_segmentation"])
        pos_list = [a[0] for a in iss_seg_store.positions()]

        if not pos_list:
            log_func(f"No positions found for {exp_name}, skipping tracking loss heatmap")
            return

        num_wells = len(pos_list)
        fig, axes = plt.subplots(1, num_wells, figsize=(15, 5) if num_wells > 1 else (6, 5))
        if num_wells == 1:
            axes = [axes]

        well_data_loaded = False

        for idx, pos in enumerate(pos_list):
            ax = axes[idx]
            linked_file = exp_dataset.append_well("linked_results", pos)

            if not linked_file.exists():
                ax.text(0.5, 0.5, f"No data\n{pos}", ha="center", va="center", fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(pos)
                continue

            try:
                # Load tracked cells data
                df = pd.read_csv(linked_file)
                if df.empty or "y_pheno" not in df.columns or "x_pheno" not in df.columns:
                    ax.text(0.5, 0.5, f"Missing coords\n{pos}", ha="center", va="center", fontsize=10)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(pos)
                    continue

                # Get ISS segmentation data for ISS cell counts and shape
                iss_seg_data = da.array(iss_seg_store[pos].data[0, 0, 0, :, :])
                shape = iss_seg_data.shape[-2:]

                # Load ISS data into memory once for this position (major speedup)
                iss_seg_np = iss_seg_data.compute()

                # Split into tiles (30x30 grid)
                tile_list, indx = split_into_tiles(shape, 30, 0)

                # Prepare coordinate arrays for vectorized operations
                y_coords = df["y_pheno"].values
                x_coords = df["x_pheno"].values
                seg_ids = df["segmentation_id"].values if "segmentation_id" in df.columns else None

                # Count ISS cells and tracked cells per tile, compute % loss
                percent_loss_values = []
                for tile in tile_list:
                    y_min, y_max, x_min, x_max = tile

                    # Count tracked cells in tile using vectorized operations
                    in_tile = (y_coords >= y_min) & (y_coords < y_max) & (x_coords >= x_min) & (x_coords < x_max)

                    if seg_ids is not None:
                        tracked_cells = len(np.unique(seg_ids[in_tile]))
                    else:
                        tracked_cells = np.sum(in_tile)

                    # Count ISS cells in tile (already in memory)
                    tile_labels = iss_seg_np[y_min:y_max, x_min:x_max]
                    iss_cells = len(np.unique(tile_labels)) - 1

                    # Calculate percent loss
                    if iss_cells > 0:
                        percent_loss = 100 * (1 - tracked_cells / iss_cells)
                        percent_loss_values.append(max(0, min(100, percent_loss)))
                    else:
                        percent_loss_values.append(0)

                # Create heatmap array
                indx_i = [a[0] for a in indx]
                indx_j = [a[1] for a in indx]
                n = int(np.sqrt(len(indx_i)))
                heatmap_array = np.zeros((n, n))
                heatmap_array[indx_i, indx_j] = percent_loss_values

                # Plot heatmap
                im = ax.imshow(heatmap_array, cmap="Reds", interpolation="nearest", vmin=0, vmax=100)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(pos, fontsize=12)

                well_data_loaded = True

            except Exception as e:
                ax.text(0.5, 0.5, f"Error\n{pos}", ha="center", va="center", fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(pos)
                log_func(f"Error processing position {pos} for {exp_name}: {e}")
                continue

        if not well_data_loaded:
            plt.close(fig)
            log_func(f"No valid well data for {exp_name}, skipping tracking loss heatmap")
            return

        # Add colorbar
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="Cell Loss (%)")
        fig.suptitle(f"Tracking Loss (ISS to Tracking) - {exp_name}", fontsize=16)

        plt.savefig(output_dir / f"{exp_name}_tracking_loss.png", dpi=300, bbox_inches="tight")
        plt.close()

    except Exception as e:
        log_func(f"Error generating tracking loss heatmap for {exp_name}: {e}")


def _format_sig_no_sci(value: float, sig: int = 2) -> str:
    """Format a number with 'sig' significant digits, never using scientific notation."""
    try:
        if value == 0 or np.isclose(value, 0.0):
            return "0"
        import math
        decimals = max(0, sig - int(math.floor(math.log10(abs(value)))) - 1)
        return f"{value:.{decimals}f}"
    except Exception:
        return f"{value:.2f}"


def _plot_total_cells(data: dict, experiments: list, output_dir: Path, ref_exp: str):
    """Plot total cells per experiment over time."""
    total_cells = [data[exp]["total_cells"] / 1e6 for exp in experiments]  # Convert to millions

    fig, (main_ax, violin_ax) = plt.subplots(
        1, 2, figsize=(20, 10), sharey=True,
        gridspec_kw={"width_ratios": [20, 1], "wspace": 0.02},
        constrained_layout=True,
    )

    main_ax.plot(range(len(experiments)), total_cells, "-o", linewidth=2, markersize=8)

    # Highlight reference experiment
    ref_idx = None
    if ref_exp in experiments:
        ref_idx = experiments.index(ref_exp)
        main_ax.axvline(x=ref_idx, color="red", linestyle="--", linewidth=2, alpha=0.2)
        main_ax.plot(ref_idx, total_cells[ref_idx], "ro", markersize=12, alpha=0.5)

    main_ax.set_xlabel("Experiment", fontsize=18)
    main_ax.set_ylabel("Total Cells (Millions)", fontsize=18)
    main_ax.set_title("Total Cells per Experiment Over Time", fontsize=24)
    main_ax.set_xticks(range(len(experiments)))
    main_ax.set_xticklabels(experiments, rotation=45, ha="right")
    main_ax.tick_params(axis="both", which="major", labelsize=14)
    main_ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Bold current experiment label
    if ref_idx is not None:
        labels = main_ax.get_xticklabels()
        labels[ref_idx].set_fontweight("bold")
        labels[ref_idx].set_fontsize(14)

    # Violin plot on the right
    parts = violin_ax.violinplot(
        total_cells, vert=True, widths=0.8, showmeans=False, showmedians=False, positions=[1]
    )
    for pc in parts["bodies"]:
        pc.set_facecolor("gray")
        pc.set_edgecolor("gray")
        pc.set_alpha(0.2)
        pc.set_linewidth(1)

    jitter = np.random.normal(0, 0.04, size=len(total_cells))
    violin_ax.plot(1 + jitter, total_cells, "o", color="black", alpha=0.4, markersize=2)

    overall_mean = np.mean(total_cells)
    violin_ax.axhline(overall_mean, color="green", linestyle=":", linewidth=1.5)
    violin_ax.text(
        1.1, overall_mean, f" {_format_sig_no_sci(overall_mean, 2)}",
        verticalalignment="bottom", color="green", fontweight="bold", fontsize=18,
    )

    if ref_idx is not None:
        current_exp_val = total_cells[ref_idx]
        violin_ax.axhline(current_exp_val, color="red", linestyle=":", linewidth=1.5)
        violin_ax.text(
            0.9, current_exp_val, f"{_format_sig_no_sci(current_exp_val, 2)} ",
            verticalalignment="bottom", horizontalalignment="right",
            color="red", fontweight="bold", fontsize=18,
        )

    violin_ax.set_title("All Exp.\nMeans", fontsize=15)
    violin_ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    violin_ax.spines["top"].set_visible(False)
    violin_ax.spines["right"].set_visible(False)
    violin_ax.spines["bottom"].set_visible(False)

    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    main_ax.yaxis.set_major_formatter(formatter)

    plt.savefig(output_dir / "total_cells_over_time.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_cells_per_well(data: dict, experiments: list, output_dir: Path, ref_exp: str):
    """Plot cells per well with individual well dots and mean line."""
    fig, (main_ax, violin_ax) = plt.subplots(
        1, 2, figsize=(20, 10), sharey=True,
        gridspec_kw={"width_ratios": [20, 1], "wspace": 0.02},
        constrained_layout=True,
    )

    # Get all unique wells across all experiments
    all_wells = set()
    for exp in experiments:
        all_wells.update(data[exp]["well_counts"].keys())
    sorted_wells = sorted(list(all_wells))

    # Color map for wells
    color_map = plt.colormaps.get("Set1")
    well_colors = {well: color_map(i % color_map.N) for i, well in enumerate(sorted_wells)}

    means = []

    for i, exp in enumerate(experiments):
        well_counts = data[exp]["well_counts"]
        well_values = [v / 1e6 for v in well_counts.values()]  # Convert to millions
        means.append(np.mean(well_values) if well_values else 0)

        # Plot individual wells
        for well, count in well_counts.items():
            main_ax.plot(i, count / 1e6, "o", color=well_colors.get(well, "gray"), alpha=0.6, markersize=8)

    # Plot mean line
    main_ax.plot(range(len(experiments)), means, "-", color="dodgerblue", linewidth=3, alpha=0.7, label="Mean")

    # Highlight reference experiment
    ref_idx = None
    if ref_exp in experiments:
        ref_idx = experiments.index(ref_exp)
        main_ax.axvline(x=ref_idx, color="red", linestyle="--", linewidth=2, alpha=0.2)

    main_ax.set_xlabel("Experiment", fontsize=18)
    main_ax.set_ylabel("Cell Count (Millions)", fontsize=18)
    main_ax.set_title("Cells per Well per Experiment Over Time", fontsize=24)
    main_ax.set_xticks(range(len(experiments)))
    main_ax.set_xticklabels(experiments, rotation=45, ha="right")
    main_ax.tick_params(axis="both", which="major", labelsize=14)
    main_ax.grid(axis="y", linestyle="--", alpha=0.7)
    main_ax.legend(loc="upper left")

    # Bold current experiment label
    if ref_idx is not None:
        labels = main_ax.get_xticklabels()
        labels[ref_idx].set_fontweight("bold")
        labels[ref_idx].set_fontsize(14)

    # Violin plot on the right
    parts = violin_ax.violinplot(
        means, vert=True, widths=0.8, showmeans=False, showmedians=False, positions=[1]
    )
    for pc in parts["bodies"]:
        pc.set_facecolor("gray")
        pc.set_edgecolor("gray")
        pc.set_alpha(0.2)
        pc.set_linewidth(1)

    jitter = np.random.normal(0, 0.04, size=len(means))
    violin_ax.plot(1 + jitter, means, "o", color="black", alpha=0.4, markersize=2)

    overall_mean = np.mean(means)
    violin_ax.axhline(overall_mean, color="green", linestyle=":", linewidth=1.5)
    violin_ax.text(
        1.1, overall_mean, f" {_format_sig_no_sci(overall_mean, 2)}",
        verticalalignment="bottom", color="green", fontweight="bold", fontsize=18,
    )

    if ref_idx is not None:
        current_exp_val = means[ref_idx]
        violin_ax.axhline(current_exp_val, color="red", linestyle=":", linewidth=1.5)
        violin_ax.text(
            0.9, current_exp_val, f"{_format_sig_no_sci(current_exp_val, 2)} ",
            verticalalignment="bottom", horizontalalignment="right",
            color="red", fontweight="bold", fontsize=18,
        )

    violin_ax.set_title("All Exp.\nMeans", fontsize=15)
    violin_ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    violin_ax.spines["top"].set_visible(False)
    violin_ax.spines["right"].set_visible(False)
    violin_ax.spines["bottom"].set_visible(False)

    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    main_ax.yaxis.set_major_formatter(formatter)

    plt.savefig(output_dir / "cells_per_well_over_time.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_mean_cells_per_gene(data: dict, experiments: list, output_dir: Path, ref_exp: str):
    """Plot mean cells per gene over time."""
    mean_cells_per_gene = [data[exp]["mean_cells_per_gene"] for exp in experiments]

    fig, (main_ax, violin_ax) = plt.subplots(
        1, 2, figsize=(20, 10), sharey=True,
        gridspec_kw={"width_ratios": [20, 1], "wspace": 0.02},
        constrained_layout=True,
    )

    main_ax.plot(range(len(experiments)), mean_cells_per_gene, "-o", linewidth=2, markersize=8, color="green")

    # Highlight reference experiment
    ref_idx = None
    if ref_exp in experiments:
        ref_idx = experiments.index(ref_exp)
        main_ax.axvline(x=ref_idx, color="red", linestyle="--", linewidth=2, alpha=0.2)
        main_ax.plot(ref_idx, mean_cells_per_gene[ref_idx], "ro", markersize=12, alpha=0.5)

    main_ax.set_xlabel("Experiment", fontsize=18)
    main_ax.set_ylabel("Mean Cells per Gene", fontsize=18)
    main_ax.set_title("Mean Cells per Gene per Experiment Over Time", fontsize=24)
    main_ax.set_xticks(range(len(experiments)))
    main_ax.set_xticklabels(experiments, rotation=45, ha="right")
    main_ax.tick_params(axis="both", which="major", labelsize=14)
    main_ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Bold current experiment label
    if ref_idx is not None:
        labels = main_ax.get_xticklabels()
        labels[ref_idx].set_fontweight("bold")
        labels[ref_idx].set_fontsize(14)

    # Violin plot on the right
    parts = violin_ax.violinplot(
        mean_cells_per_gene, vert=True, widths=0.8, showmeans=False, showmedians=False, positions=[1]
    )
    for pc in parts["bodies"]:
        pc.set_facecolor("gray")
        pc.set_edgecolor("gray")
        pc.set_alpha(0.2)
        pc.set_linewidth(1)

    jitter = np.random.normal(0, 0.04, size=len(mean_cells_per_gene))
    violin_ax.plot(1 + jitter, mean_cells_per_gene, "o", color="black", alpha=0.4, markersize=2)

    overall_mean = np.mean(mean_cells_per_gene)
    violin_ax.axhline(overall_mean, color="green", linestyle=":", linewidth=1.5)
    violin_ax.text(
        1.1, overall_mean, f" {_format_sig_no_sci(overall_mean, 2)}",
        verticalalignment="bottom", color="green", fontweight="bold", fontsize=18,
    )

    if ref_idx is not None:
        current_exp_val = mean_cells_per_gene[ref_idx]
        violin_ax.axhline(current_exp_val, color="red", linestyle=":", linewidth=1.5)
        violin_ax.text(
            0.9, current_exp_val, f"{_format_sig_no_sci(current_exp_val, 2)} ",
            verticalalignment="bottom", horizontalalignment="right",
            color="red", fontweight="bold", fontsize=18,
        )

    violin_ax.set_title("All Exp.\nMeans", fontsize=15)
    violin_ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    violin_ax.spines["top"].set_visible(False)
    violin_ax.spines["right"].set_visible(False)
    violin_ax.spines["bottom"].set_visible(False)

    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    main_ax.yaxis.set_major_formatter(formatter)

    plt.savefig(output_dir / "mean_cells_per_gene_over_time.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_max_cells_per_gene(data: dict, experiments: list, output_dir: Path, ref_exp: str):
    """Plot max cells per gene over time."""
    max_cells_per_gene = [data[exp]["max_cells_per_gene"] for exp in experiments]

    fig, (main_ax, violin_ax) = plt.subplots(
        1, 2, figsize=(20, 10), sharey=True,
        gridspec_kw={"width_ratios": [20, 1], "wspace": 0.02},
        constrained_layout=True,
    )

    main_ax.plot(range(len(experiments)), max_cells_per_gene, "-o", linewidth=2, markersize=8, color="purple")

    # Highlight reference experiment
    ref_idx = None
    if ref_exp in experiments:
        ref_idx = experiments.index(ref_exp)
        main_ax.axvline(x=ref_idx, color="red", linestyle="--", linewidth=2, alpha=0.2)
        main_ax.plot(ref_idx, max_cells_per_gene[ref_idx], "ro", markersize=12, alpha=0.5)

    main_ax.set_xlabel("Experiment", fontsize=18)
    main_ax.set_ylabel("Max Cells per Gene", fontsize=18)
    main_ax.set_title("Max Cells per Gene per Experiment Over Time", fontsize=24)
    main_ax.set_xticks(range(len(experiments)))
    main_ax.set_xticklabels(experiments, rotation=45, ha="right")
    main_ax.tick_params(axis="both", which="major", labelsize=14)
    main_ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Bold current experiment label
    if ref_idx is not None:
        labels = main_ax.get_xticklabels()
        labels[ref_idx].set_fontweight("bold")
        labels[ref_idx].set_fontsize(14)

    # Violin plot on the right
    parts = violin_ax.violinplot(
        max_cells_per_gene, vert=True, widths=0.8, showmeans=False, showmedians=False, positions=[1]
    )
    for pc in parts["bodies"]:
        pc.set_facecolor("gray")
        pc.set_edgecolor("gray")
        pc.set_alpha(0.2)
        pc.set_linewidth(1)

    jitter = np.random.normal(0, 0.04, size=len(max_cells_per_gene))
    violin_ax.plot(1 + jitter, max_cells_per_gene, "o", color="black", alpha=0.4, markersize=2)

    overall_mean = np.mean(max_cells_per_gene)
    violin_ax.axhline(overall_mean, color="green", linestyle=":", linewidth=1.5)
    violin_ax.text(
        1.1, overall_mean, f" {_format_sig_no_sci(overall_mean, 2)}",
        verticalalignment="bottom", color="green", fontweight="bold", fontsize=18,
    )

    if ref_idx is not None:
        current_exp_val = max_cells_per_gene[ref_idx]
        violin_ax.axhline(current_exp_val, color="red", linestyle=":", linewidth=1.5)
        violin_ax.text(
            0.9, current_exp_val, f"{_format_sig_no_sci(current_exp_val, 2)} ",
            verticalalignment="bottom", horizontalalignment="right",
            color="red", fontweight="bold", fontsize=18,
        )

    violin_ax.set_title("All Exp.\nMeans", fontsize=15)
    violin_ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    violin_ax.spines["top"].set_visible(False)
    violin_ax.spines["right"].set_visible(False)
    violin_ax.spines["bottom"].set_visible(False)

    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    main_ax.yaxis.set_major_formatter(formatter)

    plt.savefig(output_dir / "max_cells_per_gene_over_time.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_growth_effect_metrics(data: dict, experiments: list, output_dir: Path, ref_exp: str):
    """Plot growth effect correlation metrics over time."""
    slopes = [data[exp].get("growth_effect_slope", np.nan) for exp in experiments]
    r2_values = [data[exp].get("growth_effect_r2", np.nan) for exp in experiments]
    fold_changes = [data[exp].get("growth_effect_fc", np.nan) for exp in experiments]

    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # Slope plot
    axes[0].plot(range(len(experiments)), slopes, "-o", linewidth=2, markersize=8, color="orange")
    axes[0].set_ylabel("Slope", fontsize=14)
    axes[0].set_title("Growth Effect Regression Slope Over Time", fontsize=14)
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    # R² plot
    axes[1].plot(range(len(experiments)), r2_values, "-o", linewidth=2, markersize=8, color="blue")
    axes[1].set_ylabel("R²", fontsize=14)
    axes[1].set_title("Growth Effect R² Over Time", fontsize=14)
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)
    axes[1].set_ylim(0, 1)

    # Fold change plot
    axes[2].plot(range(len(experiments)), fold_changes, "-o", linewidth=2, markersize=8, color="red")
    axes[2].set_ylabel("Fold Change", fontsize=14)
    axes[2].set_title("Growth Effect Fold Change Over Time", fontsize=14)
    axes[2].set_xlabel("Experiment", fontsize=14)
    axes[2].grid(axis="y", linestyle="--", alpha=0.7)

    # Highlight reference experiment on all subplots
    if ref_exp in experiments:
        ref_idx = experiments.index(ref_exp)
        for ax in axes:
            ax.axvline(x=ref_idx, color="red", linestyle="--", linewidth=2, alpha=0.3)

    # Set x-axis labels on all subplots
    for ax in axes:
        ax.set_xticks(range(len(experiments)))
        ax.set_xticklabels(experiments, rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_dir / "growth_effect_metrics_over_time.png", dpi=300)
    plt.close()


def _plot_cell_loss_iss_to_tracking(data: dict, experiments: list, output_dir: Path, ref_exp: str):
    """Plot cell counts post-ISS vs post-tracking with shaded loss area."""
    # Extract data
    cells_iss = []
    cells_tracking = []
    exp_with_data = []

    for exp in experiments:
        iss_count = data[exp].get("total_cells_iss", 0)
        tracking_count = data[exp]["total_cells"]

        # Only include if we have ISS data
        if iss_count > 0:
            cells_iss.append(iss_count / 1e6)  # Convert to millions
            cells_tracking.append(tracking_count / 1e6)
            exp_with_data.append(exp)

    if not cells_iss:
        print("WARNING: No ISS cell count data available for cell loss plot.")
        print(f"  Checked {len(experiments)} experiments but none had valid plate_stats.csv with 'cells_with_matched_reads'.")
        print("  Make sure experiments have 3-assembly/ISS/mine/plate_stats.csv files.")
        return

    fig, (main_ax, violin_ax) = plt.subplots(
        1, 2, figsize=(20, 10), sharey=False,
        gridspec_kw={"width_ratios": [20, 1], "wspace": 0.02},
        constrained_layout=True,
    )

    x_pos = range(len(exp_with_data))

    # Plot lines
    main_ax.plot(x_pos, cells_iss, "-o", linewidth=2, markersize=8, color="blue", label="Post-ISS", zorder=3)
    main_ax.plot(x_pos, cells_tracking, "-o", linewidth=2, markersize=8, color="green", label="Post-Tracking", zorder=3)

    # Fill the area between the lines (cell loss)
    main_ax.fill_between(x_pos, cells_iss, cells_tracking, alpha=0.3, color="red", label="Cell Loss")

    # Calculate percent loss
    percent_loss = [100 * (1 - track / iss) if iss > 0 else 0 for iss, track in zip(cells_iss, cells_tracking)]

    # Highlight reference experiment
    ref_idx = None
    if ref_exp in exp_with_data:
        ref_idx = exp_with_data.index(ref_exp)
        main_ax.axvline(x=ref_idx, color="red", linestyle="--", linewidth=2, alpha=0.2, zorder=1)
        main_ax.plot(ref_idx, cells_iss[ref_idx], "o", color="blue", markersize=14, markeredgecolor="red", markeredgewidth=2, zorder=4)
        main_ax.plot(ref_idx, cells_tracking[ref_idx], "o", color="green", markersize=14, markeredgecolor="red", markeredgewidth=2, zorder=4)

    main_ax.set_xlabel("Experiment", fontsize=18)
    main_ax.set_ylabel("Total Cells (Millions)", fontsize=18)
    main_ax.set_title("Cell Loss from ISS to Tracking", fontsize=24)
    main_ax.set_xticks(x_pos)
    main_ax.set_xticklabels(exp_with_data, rotation=45, ha="right")
    main_ax.tick_params(axis="both", which="major", labelsize=14)
    main_ax.grid(axis="y", linestyle="--", alpha=0.7)
    main_ax.legend(loc="upper left", fontsize=14)

    # Bold current experiment label
    if ref_idx is not None:
        labels = main_ax.get_xticklabels()
        labels[ref_idx].set_fontweight("bold")
        labels[ref_idx].set_fontsize(14)

    # Add text annotations for percent loss
    for i, (iss, track, loss) in enumerate(zip(cells_iss, cells_tracking, percent_loss)):
        mid_y = (iss + track) / 2
        main_ax.text(i, mid_y, f"{loss:.1f}%", ha="center", va="center", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    # Violin plot showing percent loss distribution
    parts = violin_ax.violinplot(
        percent_loss, vert=True, widths=0.8, showmeans=False, showmedians=False, positions=[1]
    )
    for pc in parts["bodies"]:
        pc.set_facecolor("red")
        pc.set_edgecolor("red")
        pc.set_alpha(0.3)
        pc.set_linewidth(1)

    jitter = np.random.normal(0, 0.04, size=len(percent_loss))
    violin_ax.plot(1 + jitter, percent_loss, "o", color="red", alpha=0.6, markersize=4)

    overall_mean = np.mean(percent_loss)
    violin_ax.axhline(overall_mean, color="darkred", linestyle=":", linewidth=1.5)
    violin_ax.text(
        1.1, overall_mean, f" {overall_mean:.1f}%",
        verticalalignment="center", color="darkred", fontweight="bold", fontsize=18,
    )

    if ref_idx is not None:
        current_exp_val = percent_loss[ref_idx]
        violin_ax.axhline(current_exp_val, color="red", linestyle=":", linewidth=1.5)
        violin_ax.text(
            0.9, current_exp_val, f"{current_exp_val:.1f}% ",
            verticalalignment="center", horizontalalignment="right",
            color="red", fontweight="bold", fontsize=18,
        )

    violin_ax.set_title("Loss %\nDist.", fontsize=15)
    violin_ax.set_ylabel("Cell Loss (%)", fontsize=14)
    violin_ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    violin_ax.spines["top"].set_visible(False)
    violin_ax.spines["right"].set_visible(False)
    violin_ax.spines["bottom"].set_visible(False)

    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    main_ax.yaxis.set_major_formatter(formatter)

    plt.savefig(output_dir / "cell_loss_iss_to_tracking.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_cell_loss_per_well(data: dict, experiments: list, output_dir: Path, ref_exp: str):
    """Plot cell counts per well post-ISS vs post-tracking with shaded loss area."""
    # Get all unique wells
    all_wells = set()
    for exp in experiments:
        all_wells.update(data[exp].get("well_counts", {}).keys())
    sorted_wells = sorted(list(all_wells))

    # Color map for wells
    color_map = plt.colormaps.get("Set1")
    well_colors = {well: color_map(i % color_map.N) for i, well in enumerate(sorted_wells)}

    # Extract per-well data
    well_data = {}
    for well in sorted_wells:
        cells_iss = []
        cells_tracking = []
        exp_with_data = []

        for exp in experiments:
            well_iss_counts = data[exp].get("well_iss_counts", {})
            well_counts = data[exp].get("well_counts", {})

            # Match well names (ISS uses "A/1/0", tracking uses "A/1")
            iss_count = 0
            for iss_well, count in well_iss_counts.items():
                if iss_well.startswith(well + "/") or iss_well == well:
                    iss_count = count
                    break

            tracking_count = well_counts.get(well, 0)

            # Only include if we have ISS data
            if iss_count > 0:
                cells_iss.append(iss_count / 1e3)  # Convert to thousands
                cells_tracking.append(tracking_count / 1e3)
                exp_with_data.append(exp)

        if cells_iss:
            well_data[well] = {
                "cells_iss": cells_iss,
                "cells_tracking": cells_tracking,
                "exp_with_data": exp_with_data
            }

    if not well_data:
        print("No well data available for per-well cell loss plot")
        return

    fig, (main_ax, violin_ax) = plt.subplots(
        1, 2, figsize=(20, 10), sharey=False,
        gridspec_kw={"width_ratios": [20, 1], "wspace": 0.02},
        constrained_layout=True,
    )

    all_percent_losses = []

    # Plot each well
    for well, wdata in well_data.items():
        cells_iss = wdata["cells_iss"]
        cells_tracking = wdata["cells_tracking"]
        exp_with_data = wdata["exp_with_data"]

        x_pos = range(len(exp_with_data))
        well_color = well_colors[well]

        # Plot lines with blue for ISS and green for tracking (same as aggregate plot)
        main_ax.plot(x_pos, cells_iss, "-", linewidth=1.5, color="blue", alpha=0.3, zorder=2)
        main_ax.plot(x_pos, cells_tracking, "-", linewidth=1.5, color="green", alpha=0.3, zorder=2)

        # Plot dots with well-specific colors
        main_ax.plot(x_pos, cells_iss, "o", markersize=6, color=well_color, alpha=0.6, zorder=3)
        main_ax.plot(x_pos, cells_tracking, "o", markersize=6, color=well_color, alpha=0.6, zorder=3, label=f"Well {well}")

        # Fill the area between the lines (cell loss) with well color at low alpha
        main_ax.fill_between(x_pos, cells_iss, cells_tracking, alpha=0.1, color=well_color, zorder=1)

        # Calculate percent loss
        percent_loss = [100 * (1 - track / iss) if iss > 0 else 0 for iss, track in zip(cells_iss, cells_tracking)]
        all_percent_losses.extend(percent_loss)

        # Add text annotations for percent loss
        for i, (iss, track, loss) in enumerate(zip(cells_iss, cells_tracking, percent_loss)):
            mid_y = (iss + track) / 2
            main_ax.text(i, mid_y, f"{loss:.1f}%", ha="center", va="center", fontsize=6,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    # Add legend entries for Post-ISS and Post-Tracking at the top
    main_ax.plot([], [], "-o", linewidth=1.5, markersize=6, color="blue", alpha=0.5, label="Post-ISS")
    main_ax.plot([], [], "-o", linewidth=1.5, markersize=6, color="green", alpha=0.5, label="Post-Tracking")

    # Highlight reference experiment
    ref_idx = None
    if ref_exp in experiments:
        ref_idx = experiments.index(ref_exp)
        main_ax.axvline(x=ref_idx, color="red", linestyle="--", linewidth=2, alpha=0.2, zorder=1)

    main_ax.set_xlabel("Experiment", fontsize=18)
    main_ax.set_ylabel("Cells per Well (Thousands)", fontsize=18)
    main_ax.set_title("Cell Loss per Well from ISS to Tracking", fontsize=24)
    main_ax.set_xticks(range(len(experiments)))
    main_ax.set_xticklabels(experiments, rotation=45, ha="right")
    main_ax.tick_params(axis="both", which="major", labelsize=14)
    main_ax.grid(axis="y", linestyle="--", alpha=0.7)
    main_ax.legend(loc="upper left", fontsize=11, ncol=1)

    # Bold current experiment label
    if ref_idx is not None:
        labels = main_ax.get_xticklabels()
        labels[ref_idx].set_fontweight("bold")
        labels[ref_idx].set_fontsize(14)

    # Violin plot showing percent loss distribution per well
    from scipy import stats

    # Organize losses by well
    well_losses = {}
    for well, wdata in well_data.items():
        cells_iss = wdata["cells_iss"]
        cells_tracking = wdata["cells_tracking"]
        percent_loss = [100 * (1 - track / iss) if iss > 0 else 0 for iss, track in zip(cells_iss, cells_tracking)]
        well_losses[well] = percent_loss

    # Create violin plot
    positions = list(range(1, len(sorted_wells) + 1))
    violin_data = [well_losses[well] for well in sorted_wells]

    parts = violin_ax.violinplot(
        violin_data, positions=positions, vert=True, widths=0.7, showmeans=False, showmedians=False
    )

    for i, pc in enumerate(parts["bodies"]):
        well = sorted_wells[i]
        color = well_colors[well]
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        pc.set_alpha(0.3)
        pc.set_linewidth(1)

    # Plot individual points with jitter
    for i, well in enumerate(sorted_wells):
        losses = well_losses[well]
        jitter = np.random.normal(0, 0.04, size=len(losses))
        violin_ax.plot(positions[i] + jitter, losses, "o", color=well_colors[well], alpha=0.6, markersize=4)

    # Calculate means and perform t-tests
    well_means = {}
    for well in sorted_wells:
        well_means[well] = np.mean(well_losses[well])
        # Plot mean line
        violin_ax.axhline(well_means[well], color=well_colors[well], linestyle=":", linewidth=1, alpha=0.5)
        # Annotate mean
        violin_ax.text(
            positions[sorted_wells.index(well)] + 0.35, well_means[well], f"{well_means[well]:.1f}%",
            verticalalignment="center", color=well_colors[well], fontweight="bold", fontsize=10,
        )

    # Perform pairwise Wilcoxon signed-rank tests (paired, non-parametric) and add significance stars
    if len(sorted_wells) > 1:
        y_max = max([max(losses) for losses in well_losses.values()])
        y_offset = y_max * 0.05

        for i in range(len(sorted_wells) - 1):
            well1 = sorted_wells[i]
            well2 = sorted_wells[i + 1]

            # Use Wilcoxon signed-rank test (paired, non-parametric)
            # Need to ensure both arrays have the same length (paired samples)
            losses1 = well_losses[well1]
            losses2 = well_losses[well2]
            min_len = min(len(losses1), len(losses2))

            if min_len > 0:
                stat, p_value = stats.wilcoxon(losses1[:min_len], losses2[:min_len])

                # Determine significance stars
                if p_value < 0.001:
                    sig_text = "***"
                elif p_value < 0.01:
                    sig_text = "**"
                elif p_value < 0.05:
                    sig_text = "*"
                else:
                    sig_text = "ns"

                # Draw bracket and add stars
                y_pos = y_max + y_offset * (i + 1)
                violin_ax.plot([positions[i], positions[i+1]], [y_pos, y_pos], "k-", linewidth=1)
                violin_ax.text((positions[i] + positions[i+1]) / 2, y_pos + y_offset * 0.2, sig_text,
                              ha="center", va="bottom", fontsize=10, fontweight="bold")

    violin_ax.set_title("Loss % per Well", fontsize=15)
    violin_ax.set_ylabel("Cell Loss (%)", fontsize=14)
    violin_ax.set_xticks(positions)
    violin_ax.set_xticklabels(sorted_wells, rotation=45, ha="right", fontsize=10)
    violin_ax.spines["top"].set_visible(False)
    violin_ax.spines["right"].set_visible(False)

    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    main_ax.yaxis.set_major_formatter(formatter)

    plt.savefig(output_dir / "cell_loss_per_well.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_growth_effect_vs_cells(data: dict, experiments: list, output_dir: Path, ref_exp: str):
    """Plot scatter plots of growth effect metrics vs total cell count."""
    # Generate fold change scatter
    _plot_metric_vs_cells(
        data, experiments, output_dir, ref_exp,
        metric_key="growth_effect_fc",
        metric_label="Growth Effect Fold Change",
        filename="growth_effect_fc_vs_total_cells.png",
        add_baseline=1.0,
        baseline_label="No effect (FC=1)"
    )

    # Generate slope scatter
    _plot_metric_vs_cells(
        data, experiments, output_dir, ref_exp,
        metric_key="growth_effect_slope",
        metric_label="Growth Effect Slope",
        filename="growth_effect_slope_vs_total_cells.png",
        add_baseline=0.0,
        baseline_label="No correlation (slope=0)"
    )


def _plot_metric_vs_cells(
    data: dict,
    experiments: list,
    output_dir: Path,
    ref_exp: str,
    metric_key: str,
    metric_label: str,
    filename: str,
    add_baseline: float = None,
    baseline_label: str = None
):
    """
    Generic function to plot a metric vs total cell count.

    Args:
        data: Experiment data dictionary
        experiments: List of experiment names
        output_dir: Output directory for plot
        ref_exp: Reference experiment name
        metric_key: Key in data dict for the metric (e.g., "growth_effect_fc")
        metric_label: Y-axis label
        filename: Output filename
        add_baseline: If provided, add horizontal line at this y-value
        baseline_label: Label for baseline line
    """
    # Extract data
    total_cells = []
    metric_values = []
    exp_labels = []

    for exp in experiments:
        metric_val = data[exp].get(metric_key, np.nan)
        cells = data[exp]["total_cells"]

        # Only include experiments with valid metric
        if not np.isnan(metric_val):
            total_cells.append(cells / 1e6)  # Convert to millions
            metric_values.append(metric_val)
            exp_labels.append(exp)

    if not total_cells:
        print(f"No valid {metric_key} data for scatter plot")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot all experiments
    ax.scatter(total_cells, metric_values, s=100, alpha=0.6, color="steelblue", edgecolors="black", linewidth=1)

    # Highlight reference experiment
    if ref_exp in exp_labels:
        ref_idx = exp_labels.index(ref_exp)
        ax.scatter(
            total_cells[ref_idx],
            metric_values[ref_idx],
            s=200,
            color="red",
            alpha=0.8,
            edgecolors="black",
            linewidth=2,
            marker="*",
            label=f"Current: {ref_exp}",
            zorder=10
        )

    # Add trend line if enough points
    if len(total_cells) > 2:
        try:
            # Fit linear regression
            X = np.array(total_cells).reshape(-1, 1)
            y = np.array(metric_values)

            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)

            # Plot trend line
            sort_idx = np.argsort(total_cells)
            ax.plot(
                np.array(total_cells)[sort_idx],
                y_pred[sort_idx],
                "r--",
                linewidth=2,
                alpha=0.5,
                label=f"Trend (R²={r2_score(y, y_pred):.3f})"
            )
        except Exception:
            pass

    ax.set_xlabel("Total Cells (Millions)", fontsize=14)
    ax.set_ylabel(metric_label, fontsize=14)
    ax.set_title(f"{metric_label} vs Total Cell Count", fontsize=16)
    ax.grid(True, linestyle="--", alpha=0.3)

    # Add baseline if specified
    if add_baseline is not None:
        ax.axhline(y=add_baseline, color="gray", linestyle=":", linewidth=1.5, alpha=0.5, label=baseline_label)

    ax.legend(loc="best")

    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot tracking metrics over time for all experiments."
    )
    parser.add_argument(
        "experiment",
        type=str,
        help="Reference experiment name to determine config directory.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all plots even if they already exist.",
    )
    args = parser.parse_args()

    plot_tracking_metrics_over_time(
        args.experiment,
        verbose=args.verbose,
        force=args.force,
    )
