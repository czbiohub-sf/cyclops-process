import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from ops_utils.data.experiment import OpsDataset
from iohub.ngff import open_ome_zarr
from cyclops_process.metrics.plate_stats.match_reads import (
    _get_effective_iss_rounds,
)


def _plot_growth_effect(ax, merged_df, title):
    """
    Helper function to generate the cell count vs growth effect scatter plot,
    including linear regression and annotations.
    Returns a dictionary with the calculated regression stats.
    """
    x = merged_df["gene_effect"].dropna()
    y = merged_df["count"].loc[x.index]  # Ensure y matches the filtered x

    if x.empty or y.empty:
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center")
        ax.set_title(title)
        return {}

    ax.scatter(x, y, 5, alpha=0.4)
    ax.set_xlabel("Growth effect (CERES score)")
    ax.set_ylabel("OPS cell count")
    if " - " in title:
        main_title, subtitle = title.split(" - ", 1)
        ax.get_figure().suptitle(main_title)
        ax.set_title(subtitle, fontsize=10)
    else:
        ax.set_title(title)

    # Overlap NTC guides on plot
    ntc = merged_df[merged_df["NCBI_ID"] == -1]
    ntc_x = ntc["gene_effect"]
    ntc_y = ntc["count"]
    ax.scatter(ntc_x, ntc_y, 5, alpha=0.4, label="NTC")

    if not ntc_y.empty:
        ax.axhline(y=ntc_y.median(), color="orange", linestyle=":", label="NTC Median")
        ax.axhline(y=ntc_y.quantile(0.9), color="orange", linestyle=":", alpha=0.5)
        ax.axhline(y=ntc_y.quantile(0.1), color="orange", linestyle=":", alpha=0.5)

    if not y.empty:
        ax.set_ylim(0, y.quantile(0.90) * 1.4)

    # Define bin edges, group data into bins, calculate median, and plot median
    bins = np.linspace(min(x, default=0), max(x, default=1), 10)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    df = pd.DataFrame({"x": x, "y": y})
    df["bin"] = pd.cut(df["x"], bins=bins)
    bin_medians = df.groupby("bin", observed=False)["y"].median()
    ax.plot(
        bin_centers,
        bin_medians,
        color="red",
        marker="o",
        linestyle="-",
        linewidth=2,
        label="Binned Median",
    )

    # --- Linear Regression on Binned Medians ---
    valid = ~np.isnan(bin_medians) & (bin_centers <= 0.5)
    slope, r2, growth_effect_fc = 0, 0, 0
    if valid.sum() > 1:
        bin_centers_valid = bin_centers[valid].reshape(-1, 1)
        bin_medians_valid = bin_medians[valid]
        model = LinearRegression()
        model.fit(bin_centers_valid, bin_medians_valid)
        y_pred = model.predict(bin_centers_valid)

        slope = model.coef_[0]
        r2 = r2_score(bin_medians_valid, y_pred)

        x_fit = np.linspace(
            min(bin_centers_valid)[0], max(bin_centers_valid)[0], 100
        ).reshape(-1, 1)
        y_fit = model.predict(x_fit)
        ax.plot(x_fit, y_fit, color="black", linestyle=":")
        growth_effect_fc = y_fit[-1] / y_fit[0] if y_fit[0] != 0 else 0

        # Add annotation for slope and R2
        annotation_text = f"Slope: {slope:.2f}\n$R^2$: {r2:.2f}"
        ax.text(
            0.95,
            0.05,
            annotation_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    ax.legend()

    return {
        "growth_effect_slope": np.round(slope, 1),
        "growth_effect_r2": np.round(r2, 2),
        "growth_effect_fold_change": np.round(growth_effect_fc, 2),
    }





def cell_count_vs_growth_effect(
    experiment: str,
    debug: bool = False,
    method=None,
    iss_rounds: list[int] | None = None,
    failed_rounds_by_well: dict[str, list[int]] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Plots OPS cell counts vs growth effect scores and returns associated stats.
    Generates one plot for all wells pooled, and another with per-well subplots.
    Stats are cached to growth_effect_stats.csv for faster re-runs.

    Args:
        iss_rounds: List of ISS round indices to use for filtering gene effect database barcodes
        failed_rounds_by_well: Dictionary mapping wells to lists of failed round indices
        force: If True, regenerate even if cache exists. Default False.
    """

    dataset = OpsDataset(experiment, method=method)

    # Cache file path (use CSV as the skip indicator)
    cache_path = dataset.results_iss / "growth_effect_stats.csv"

    # Check if cache exists and skip if not forcing
    if cache_path.exists() and not force:
        print(f"[growth_effect] Growth effect stats already cached at: {cache_path}")
        print("[growth_effect] Skipping. Use --force or force=True to regenerate.")
        try:
            cached_df = pd.read_csv(cache_path, index_col=0)
            return cached_df
        except Exception as e:
            print(f"[growth_effect] Failed to load cache, regenerating: {e}")

    if debug:
        print(
            f"--- Running cell_count_vs_growth_effect for {experiment} with DEBUG ON ---"
        )

    try:
        dep_map_gene_effect_db = dataset.load_gene_index()
        if debug:
            print(
                f"\n[DEBUG] Loaded gene effect data. Shape: {dep_map_gene_effect_db.shape}"
            )
            print(dep_map_gene_effect_db.head())
    except FileNotFoundError:
        print(f"ERROR: Gene index file not found at {dataset.gene_index}")
        return pd.DataFrame()

    if "gene_effect" not in dep_map_gene_effect_db.columns:
        # Fallback for libraries without gene_effect (e.g. custom-perturbation pools):
        # use gene_target to look up gene_effect from the default reference
        gene_target_col = dataset.iss_secondary_gene_column or "gene_target"
        if gene_target_col in dep_map_gene_effect_db.columns:
            try:
                default_ref_path = dataset.configs / "library" / "twist1k_pool_CERES.csv"
                ref_df = pd.read_csv(default_ref_path)
                ref_effects = ref_df.drop_duplicates("Gene name")[["Gene name", "gene_effect", "NCBI_ID"]].dropna(subset=["gene_effect"])
                dep_map_gene_effect_db = dep_map_gene_effect_db.merge(
                    ref_effects,
                    left_on=gene_target_col,
                    right_on="Gene name",
                    how="left",
                    suffixes=("", "_ref"),
                )
                matched = dep_map_gene_effect_db["gene_effect"].notna().sum()
                total = len(dep_map_gene_effect_db)
                print(f"[growth_effect] Mapped gene_effect via '{gene_target_col}': {matched}/{total} entries matched")
            except Exception as e:
                print(f"Skipping growth effect analysis: fallback gene_effect lookup failed: {e}")
                return {}
        else:
            print(f"Skipping growth effect analysis: 'gene_effect' column not found in gene index.")
            return {}

    # --- 1. Pooled Analysis and Plot ---
    print("--- Generating pooled cell count vs growth effect plot ---")

    # For pooled plot: merge each well's frequency table with its well-specific filtered gene effect DB
    # then combine all results (since wells may have different failed rounds)
    seg_store = open_ome_zarr(dataset.store_paths["iss_segmentation"], mode="r")
    position_list = [a[0] for a in seg_store.positions()]

    merged_well_dfs = []
    for pos in position_list:
        try:
            # Get well-specific ISS rounds
            well_iss_rounds = _get_effective_iss_rounds(
                iss_rounds, pos, failed_rounds_by_well
            )

            # Load this well's frequency table
            sgRNA_counts_well = pd.read_csv(
                dataset.append_well("frequency_table", pos), index_col=0
            )

            # Filter gene effect database to this well's specific ISS rounds
            dep_map_well = dep_map_gene_effect_db.copy()
            offset = dataset.codebook_round_offset
            if offset:
                # Gene index barcodes are sgRNA[:10], but freq table uses sgRNA[offset:offset+N]
                # Map via raw codebook, then filter by well_iss_rounds for dropped rounds
                raw_cb = pd.read_csv(dataset.codebook)
                bc_map = dict(zip(raw_cb["sgRNA"].str[:10], raw_cb["sgRNA"].str[offset:offset + 10]))
                dep_map_well["barcode"] = dep_map_well["barcode"].map(bc_map).apply(
                    lambda bc: "".join([bc[i] for i in well_iss_rounds if i < len(bc)]) if pd.notna(bc) else ""
                )
            else:
                dep_map_well["barcode"] = dep_map_well["barcode"].apply(
                    lambda bc: "".join([bc[i] for i in well_iss_rounds if i < len(bc)])
                )
            dep_map_well = dep_map_well.drop_duplicates(subset=["barcode"], keep="first")

            # Merge this well's data
            merged_well = pd.merge(
                sgRNA_counts_well, dep_map_well, on="barcode", how="left"
            )
            merged_well_dfs.append(merged_well)

            if debug:
                print(f"\n[DEBUG] Well {pos}: ISS rounds {well_iss_rounds}, merged shape: {merged_well.shape}")
        except FileNotFoundError:
            if debug:
                print(f"[DEBUG] Frequency table for {pos} not found, skipping")
            continue

    if not merged_well_dfs:
        print("No frequency tables found. Skipping growth effect plots.")
        return pd.DataFrame()

    # Combine all per-well merged dataframes
    merged_df_pooled = pd.concat(merged_well_dfs, ignore_index=True)

    if debug:
        print(f"\n[DEBUG] Pooled merged dataframe shape: {merged_df_pooled.shape}")
    if debug:
        print(
            f"\n[DEBUG] Merged frequency table with gene effect data on 'barcode'. Shape: {merged_df_pooled.shape}"
        )
        print(
            f"  - NaN values created in 'gene_effect' column after merge: {merged_df_pooled['gene_effect'].isnull().sum()}"
        )
        # Show rows that failed to merge to help diagnose barcode mismatches
        print("  - Example rows that failed to find a gene_effect:")
        print(merged_df_pooled[merged_df_pooled["gene_effect"].isnull()].head())

        final_plot_data = merged_df_pooled.dropna(subset=["gene_effect"])
        print(
            f"\n[DEBUG] Data after dropping NaNs (this is what gets plotted). Shape: {final_plot_data.shape}"
        )
        if final_plot_data.empty:
            print(
                "[DEBUG] No data remains after merge and dropna. This is why the plot is empty."
            )

    fig_pooled, ax_pooled = plt.subplots(figsize=(6, 4))
    pooled_stats = _plot_growth_effect(
        ax_pooled,
        merged_df_pooled,
        title=f"Cell Count vs Growth Effect (All Wells) - {experiment}",
    )

    # --- Calculate and save statistics from the POOLED data ---
    targeting_guides = merged_df_pooled[merged_df_pooled["NCBI_ID"] != -1]
    ntc = merged_df_pooled[merged_df_pooled["NCBI_ID"] == -1]
    ntc_y = ntc["count"]

    if not ntc_y.empty:
        pooled_stats["num_tar_guides_below_NTC_10%"] = (
            targeting_guides["count"] < ntc_y.quantile(0.1)
        ).sum()
    else:
        pooled_stats["num_tar_guides_below_NTC_10%"] = 0

    plt.savefig(dataset.metrics_paths["cell_count_vs_growth_effect"], dpi=300)
    plt.close(fig_pooled)

    plot_metrics_df = pd.DataFrame([pooled_stats])
    # This file is now temporary, as the main stats function will append to it.
    # plot_metrics_df.T.to_csv(dataset.metrics_paths["statistics_bio_plot"])

    # --- 2. Per-Well Analysis and Plot ---
    print("--- Generating per-well cell count vs growth effect plots ---")
    seg_store = open_ome_zarr(dataset.store_paths["iss_segmentation"], mode="r")
    position_list = [a[0] for a in seg_store.positions()]

    num_wells = len(position_list)
    fig_per_well, axes = plt.subplots(
        1, num_wells, figsize=(5 * num_wells, 4), sharey=True
    )
    if num_wells == 1:
        axes = [axes]  # ensure axes is always iterable

    per_well_stats = {}
    for i, pos in enumerate(position_list):
        ax = axes[i]
        try:
            # Get well-specific ISS rounds
            well_iss_rounds = _get_effective_iss_rounds(
                iss_rounds, pos, failed_rounds_by_well
            )

            # Load this well's frequency table
            sgRNA_counts_well = pd.read_csv(
                dataset.append_well("frequency_table", pos), index_col=0
            )

            # Filter gene effect database to this well's specific ISS rounds
            dep_map_well = dep_map_gene_effect_db.copy()
            offset = dataset.codebook_round_offset
            if offset:
                # Gene index barcodes are sgRNA[:10], but freq table uses sgRNA[offset:offset+N]
                # Map via raw codebook, then filter by well_iss_rounds for dropped rounds
                raw_cb = pd.read_csv(dataset.codebook)
                bc_map = dict(zip(raw_cb["sgRNA"].str[:10], raw_cb["sgRNA"].str[offset:offset + 10]))
                dep_map_well["barcode"] = dep_map_well["barcode"].map(bc_map).apply(
                    lambda bc: "".join([bc[i] for i in well_iss_rounds if i < len(bc)]) if pd.notna(bc) else ""
                )
            else:
                dep_map_well["barcode"] = dep_map_well["barcode"].apply(
                    lambda bc: "".join([bc[i] for i in well_iss_rounds if i < len(bc)])
                )
            dep_map_well = dep_map_well.drop_duplicates(subset=["barcode"], keep="first")

            # Merge this well's data
            merged_df_well = pd.merge(
                sgRNA_counts_well, dep_map_well, on="barcode", how="left"
            )
            stats = _plot_growth_effect(ax, merged_df_well, title=pos)
            per_well_stats[pos] = stats
        except FileNotFoundError:
            ax.text(0.5, 0.5, f"Data not found for\n{pos}", ha="center", va="center")
            ax.set_title(pos)
            continue

    fig_per_well.suptitle(
        f"Per-Well Cell Count vs Growth Effect - {experiment}", fontsize=16
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(dataset.metrics_paths["cell_count_vs_growth_effect_per_well"], dpi=300)
    plt.close(fig_per_well)

    # --- Append per-well stats to the main statistics file ---
    per_well_stats_df = pd.DataFrame.from_dict(per_well_stats, orient="index")

    # Read the existing main stats CSV
    # stats_df = pd.read_csv(dataset.metrics_paths["statistics"], index_col=0)

    # Transpose the per-well stats to align wells as columns, matching the main stats file
    per_well_stats_df_T = per_well_stats_df.T

    # Save stats to cache CSV
    try:
        per_well_stats_df_T.to_csv(cache_path)
        print(f"[growth_effect] Saved growth effect stats to: {cache_path}")
    except Exception as e:
        print(f"[growth_effect] Failed to save stats cache: {e}")

    return per_well_stats_df_T

