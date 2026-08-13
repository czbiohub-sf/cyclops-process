"""
View cells for a specific gene in a tiled napari grid.

Usage:
    python cyclops_process/napari/view_by_gene.py <experiment> <gene> [OPTIONS]

Examples:
    # View 100 cells for gene RBPJ (experiment can be shorthand like "12" or "ops12")
    python cyclops_process/napari/view_by_gene.py 12 RBPJ

    # View cells with larger crops
    python cyclops_process/napari/view_by_gene.py 12 RBPJ --crop_size 256

    # View cells from a specific well
    python cyclops_process/napari/view_by_gene.py 12 RBPJ --well A1

    # View more cells in a wider grid
    python cyclops_process/napari/view_by_gene.py 113 RBPJ --max_cells 200 --grid_width 20

    # View randomly sampled NTC controls (4 random NTC guides)
    python cyclops_process/napari/view_by_gene.py 113 NTC

    # Attention-atlas mode — top-N attention cells per marker, cross-
    # experiment, mirroring the PDF attention atlas in napari with NTC
    # rows interleaved beneath each KO row. One row for Phase, plus one
    # row per fluor channel that has top-attention cells for `<gene>`.
    # Loads ALL channels of each cell's source zarr so phase / fluor /
    # nuclei overlays can all be toggled per row. The first positional
    # `<experiment>` is ignored in this mode (pass any placeholder).
    # Implementation lives in `view_by_gene_attention.py` (sibling
    # module); this script just routes `--attention` through to it.
    python cyclops_process/napari/view_by_gene.py _ ABCE1 --attention --top 10

    # Override the attention source CSVs (defaults to v3 cohort under
    # /path/to/ops_data/.../v3/attention_v3/). Each CSV is
    # auto-cached as parquet on first use, and per-(exp, well) NTC pools
    # are cached too. Both live under /path/to/ops_data/cache/
    # (override via $OPS_CACHE_DIR) so all users share the build cost.
    python cyclops_process/napari/view_by_gene.py _ ABCE1 --attention \\
        --phase-csv /path/to/pma_top_phase_cells_v4.csv \\
        --fluor-csv /path/to/pma_top_fluorescent_cells_v4.csv

    # Disable the per-row natural-language captions (left gutter)
    python cyclops_process/napari/view_by_gene.py _ ABCE1 --attention --captions-csv ""

Options:
    --crop_size      Size of square crop around each cell (default: 256)
    --max_cells      Maximum number of cells to display (default: 80)
    --grid_width     Number of cells per row in grid (default: 10)
    --mask-dilation  Pixels to dilate the inverse mask (default: 10)
    --well           Specific well to load (e.g., 'A/1' or 'A1')
    --no-by-guide    Disable organizing cells by guide (on by default)
    --include-no-seg Include cells without segmentation ID (excluded by default)
    --attention      PMA attention atlas: top-N cells per marker for `<gene>`,
                     cross-experiment, with all channels stacked as overlays
                     and NTC strips interleaved beneath each KO row.
    --top            Top-N cells per marker in attention mode (default: 10)
    --phase-csv      Override the attention phase CSV path
    --fluor-csv      Override the attention fluor CSV path
    --captions-csv   Override the per-gene captions CSV (or pass "" to disable)
    --marker-map-csv Override the mAP matrix used for top-marker selection.
                     Defaults to mAP DISTINCTIVENESS at gene level
                     (gene_reporter_distinctiveness_raw.csv), mAP CONSISTENCY
                     at complex level (complex_reporter_chad_consistency.csv).
    --chad-config    Override the CHAD positive-controls YAML used to map
                     complex_num → name (only used at --aggregation-level complex).
    --aggregation-level
                     `gene` (default) for gene-KO atlas, `complex` for CHAD
                     pathway atlas. Selects which mAP matrix is used for
                     top-marker ranking. At `complex`, pass the complex name
                     (e.g. "subu Proteasome 19s") as <gene>.
    --top-markers    Render only the top-N fluor markers by mAP (default: 3).
                     The atlas always shows the Phase row in addition. Pass 0
                     to render every marker that has top-attention cells for
                     the gene (the v3 cohort has ~50-60 markers, which makes
                     napari layer creation very slow).
    --preload-layers (--attention only) Pre-create a napari layer for every
                     (row, channel) pair so any channel can be toggled visible
                     from the layer panel. Adds ~30-90 s of startup with
                     ~570 layers; off by default. Without it, only each row's
                     primary channel + a phase backdrop get layers.
"""

import pandas as pd
import napari
import numpy as np

import sys
import os

sys.path.insert(0, os.getcwd())


from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.data.bbox_utils import BaseDataset
from cyclops_utils.data.filesystem import resolve_experiment_name
from iohub import open_ome_zarr
import click


@click.command()
@click.argument("experiment")
@click.argument("gene")
@click.option(
    "--crop_size", default=256, help="Crop size in pixels (square)", type=int
)
@click.option(
    "--max_cells", default=80, help="Maximum number of cells to display", type=int
)
@click.option(
    "--grid_width", default=10, help="Number of cells per row in grid", type=int
)
@click.option(
    "--mask-dilation", "mask_dilation", default=10, help="Pixels to dilate inverse mask (default: 10)", type=int
)
@click.option(
    "--well", default=None, help="Specific well to load (e.g., 'A/1')", type=str
)
@click.option(
    "--by-guide/--no-by-guide", "by_guide", default=True, help="Organize cells by guide (default: on)"
)
@click.option(
    "--include-no-seg", "include_no_seg", is_flag=True, default=False, help="Include cells without segmentation ID (excluded by default)"
)
@click.option(
    "--4i", "four_i", is_flag=True, default=False,
    help="Use 4i_cell_seg mask and 4i_bbox column (if present) instead of cell_seg/bbox"
)
@click.option(
    "--attention", "attention_mode", is_flag=True, default=False,
    help="Use the PMA attention atlas to pick the top-N cells per marker "
         "for <gene> (cross-experiment). Ignores <experiment>.",
)
@click.option(
    "--top", "top_n", default=10, type=int,
    help="Top-N cells per marker in --attention mode (default: 10)",
)
@click.option(
    "--phase-csv", "phase_csv", default=None, type=str,
    help="Override the attention phase CSV path",
)
@click.option(
    "--fluor-csv", "fluor_csv", default=None, type=str,
    help="Override the attention fluor CSV path",
)
@click.option(
    "--captions-csv", "captions_csv", default=None, type=str,
    help="Override the attention captions CSV (per-gene per-channel "
         "natural-language description). Falls back to the default v4 "
         "captions; pass an empty string to disable.",
)
@click.option(
    "--marker-map-csv", "marker_map_csv", default=None, type=str,
    help="(--attention) Override the mAP matrix used for top-marker "
         "selection. Defaults: distinctiveness "
         "(gene_reporter_distinctiveness_raw.csv) at gene level, "
         "consistency (complex_reporter_chad_consistency.csv) at "
         "complex level — same matrices the SHAP atlas uses.",
)
@click.option(
    "--chad-config", "chad_config", default=None, type=str,
    help="(--attention --aggregation-level complex) Override the CHAD "
         "positive-controls YAML used to map complex_num → name in the "
         "consistency CSV. Defaults to chad_positive_controls_v5_hierarchy.yml.",
)
@click.option(
    "--aggregation-level", "aggregation_level",
    type=click.Choice(["gene", "complex"]), default="gene",
    help="(--attention) `gene` = gene-KO atlas, picks markers by mAP "
         "DISTINCTIVENESS. `complex` = CHAD pathway atlas, picks markers "
         "by mAP CONSISTENCY. Pass the complex name (e.g. "
         "'subu Proteasome 19s') as <gene> at complex level.",
)
@click.option(
    "--top-markers", "top_markers", default=3, type=int,
    help="(--attention) Render only the top-N fluor markers ranked "
         "by mAP (default: 3). The atlas still always shows the Phase "
         "row. Pass 0 to render every marker that has top-attention "
         "cells for the gene (slow, ~50-60 rows for v3).",
)
@click.option(
    "--preload-layers", "preload_layers", is_flag=True, default=False,
    help="(--attention only) Pre-create a napari layer for every "
         "(row, channel) pair, so the full set of channels is "
         "toggleable from the layer panel. Adds ~30-90 s to startup "
         "with ~570 layers — only enable when you need to flip "
         "between non-default channels.",
)
def view_by_gene_cli(experiment, gene, crop_size, max_cells, grid_width, mask_dilation, well, by_guide, include_no_seg, four_i, attention_mode, top_n, phase_csv, fluor_csv, captions_csv, marker_map_csv, chad_config, aggregation_level, top_markers, preload_layers):
    if attention_mode:
        # Imported lazily so the heavy attention-only deps (and the
        # ~700-line implementation) only load when actually needed.
        from view_by_gene_attention import view_by_gene_attention
        return view_by_gene_attention(
            gene,
            top_n=top_n,
            crop_size=crop_size,
            mask_dilation=mask_dilation,
            phase_csv=phase_csv,
            fluor_csv=fluor_csv,
            captions_csv=captions_csv,
            marker_map_csv=marker_map_csv,
            chad_config=chad_config,
            aggregation_level=aggregation_level,
            top_markers=top_markers,
            preload_layers=preload_layers,
        )
    return view_by_gene(
        experiment,
        gene,
        crop_size=crop_size,
        max_cells=max_cells,
        grid_width=grid_width,
        mask_dilation=mask_dilation,
        well=well,
        by_guide=by_guide,
        include_no_seg=include_no_seg,
        four_i=four_i,
    )


def view_by_gene(
    experiment,
    gene,
    crop_size=128,
    max_cells=80,
    grid_width=10,
    mask_dilation=10,
    well=None,
    by_guide=True,
    include_no_seg=False,
    four_i=False,
):
    """
    View cells for a specific gene from the stitched phenotyping zarr using BaseDataset.

    Args:
        experiment: Experiment name (e.g., 'ops0033_20240808')
        gene: Gene name to filter cells by
        crop_size: Size of square crop around each cell (default 128)
        max_cells: Maximum number of cells to display (default 100)
        grid_width: Number of cells per row in the tiled view (default 10)
        mask_dilation: Pixels to dilate the inverse mask (default 10)
        well: Specific well to load (e.g., 'A/1'), or None for all wells
        by_guide: If True (default), organize cells by guide (one row per guide) with labels
        include_no_seg: If True, include cells without segmentation ID (default False)
    """
    import re
    import warnings

    # Resolve experiment shorthand (e.g., "33" -> "ops0033_20250429")
    experiment = resolve_experiment_name(experiment)
    dataset = OpsDataset(experiment)

    # Open the stitched phenotyping store (v3 zarr format)
    # Always use phenotyping_v3.zarr — 4i mode just swaps mask/bbox columns.
    # Suppress zarr v2/v3 coexistence warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="zarr")
        pheno_store = open_ome_zarr(dataset.store_paths["pheno_assembled_v3"], mode="r")

    # Determine wells to process
    if well is not None:
        # Normalize well format (e.g., "A1" -> "A/1")
        if "/" not in well:
            m = re.match(r"^([A-Za-z]+)(\d+)$", well)
            if m:
                well = f"{m.group(1)}/{m.group(2)}"
        wells = [well]
    else:
        wells = [f"A/{i}" for i in pheno_store["A"].group_keys()]

    # Load linked results for selected wells
    # In --4i mode, prefer four_i_linked_<well>.csv (has 4i_bbox + gene calls)
    results_list = []
    for w in wells:
        results_path = None
        if four_i:
            four_i_csv = dataset.results_fast / f"four_i_linked_{w.replace('/', '_')}_0.csv"
            if four_i_csv.exists():
                results_path = four_i_csv
                print(f"Using 4i linked CSV: {four_i_csv.name}")
        if results_path is None:
            results_path = dataset.append_well("linked_results", w)
        if results_path.exists():
            df = pd.read_csv(results_path)
            df["well"] = f"{w}/0"  # Format for BaseDataset: "A/1/0"
            df["store_key"] = "pheno_assembled_v3"
            results_list.append(df)

    if not results_list:
        print(f"No linked results found for experiment {experiment}")
        return

    results_df = pd.concat(results_list, ignore_index=True)

    # 4i mode: swap bbox/seg columns to use 4i-specific values if present
    if four_i:
        if "4i_bbox" in results_df.columns:
            print("Using '4i_bbox' column for bounding boxes")
            results_df["bbox"] = results_df["4i_bbox"]
        else:
            print("Note: '4i_bbox' column not found — using default 'bbox'")
        if "4i_segmentation_id" in results_df.columns:
            print("Using '4i_segmentation_id' for segmentation IDs")
            results_df["segmentation_id"] = results_df["4i_segmentation_id"]

    # Determine the gene name column: use the configured custom output column when set
    gene_col = "gene_name"
    secondary_gene_col = None
    if dataset.gene_name_output_column and dataset.gene_name_output_column in results_df.columns:
        gene_col = dataset.gene_name_output_column
        print(f"Using custom gene column: '{gene_col}'")
    if dataset.iss_secondary_gene_column and dataset.iss_secondary_gene_column in results_df.columns:
        secondary_gene_col = dataset.iss_secondary_gene_column
        print(f"Secondary gene column (target): '{secondary_gene_col}'")

    # Fill NaN gene names with NTC + barcode
    results_df[gene_col] = results_df[gene_col].fillna(
        "NTC_" + results_df["barcode"].astype(str)
    )

    # Add total_index if not present (required by BaseDataset)
    if "total_index" not in results_df.columns:
        results_df["total_index"] = results_df.index

    # Special handling for "NTC" - randomly sample 4 NTC guides
    if gene.upper() == "NTC":
        # Find all NTC genes (start with "NTC_")
        ntc_genes = [g for g in results_df[gene_col].unique() if g and g.startswith("NTC_")]
        if len(ntc_genes) == 0:
            print("No NTC genes found in the dataset")
            return
        # Randomly sample 4 NTC guides (or all if fewer than 4)
        n_ntc_guides = min(4, len(ntc_genes))
        np.random.seed(42)
        selected_ntc_genes = np.random.choice(ntc_genes, size=n_ntc_guides, replace=False)
        print(f"Selected {n_ntc_guides} random NTC guides: {list(selected_ntc_genes)}")
        single_gene_df = results_df[results_df[gene_col].isin(selected_ntc_genes)].copy()
        gene = "NTC"  # Keep display name as NTC
    else:
        # Filter for the requested gene - search both primary and secondary gene columns
        single_gene_df = results_df[results_df[gene_col] == gene].copy()

        # If no match on primary column, try searching the secondary gene column (e.g., gene_target)
        if len(single_gene_df) == 0 and secondary_gene_col:
            single_gene_df = results_df[results_df[secondary_gene_col] == gene].copy()
            if len(single_gene_df) > 0:
                print(f"Matched '{gene}' on secondary column '{secondary_gene_col}'")

    total_cells = len(single_gene_df)

    # Build a display label that includes the target gene when configured
    display_gene = gene
    if secondary_gene_col and len(single_gene_df) > 0:
        target_genes = single_gene_df[secondary_gene_col].dropna().unique()
        if len(target_genes) == 1:
            display_gene = f"{gene} (target: {target_genes[0]})"
        elif len(target_genes) > 1:
            display_gene = f"{gene} (targets: {', '.join(str(t) for t in target_genes[:5])})"

    # Filter out cells without segmentation ID (unless --include-no-seg is set)
    if not include_no_seg and "segmentation_id" in single_gene_df.columns:
        cells_with_seg = single_gene_df["segmentation_id"].notna().sum()
        single_gene_df = single_gene_df[single_gene_df["segmentation_id"].notna()].copy()
        excluded = total_cells - len(single_gene_df)
        if excluded > 0:
            print(f"Found {total_cells} cells for '{display_gene}', excluded {excluded} without segmentation")
        else:
            print(f"Found {len(single_gene_df)} cells for '{display_gene}' in experiment {experiment}")
    else:
        print(f"Found {len(single_gene_df)} cells for '{display_gene}' in experiment {experiment}")

    if len(single_gene_df) == 0:
        print(f"No cells found for gene '{gene}'.")

        # Find similar gene names using simple string matching
        all_genes = results_df[gene_col].dropna().unique()
        # Also include secondary gene column values in suggestions
        if secondary_gene_col:
            secondary_genes = results_df[secondary_gene_col].dropna().unique()
            all_genes = np.unique(np.concatenate([all_genes, secondary_genes]))
        gene_upper = gene.upper()
        similar_genes = []
        for g in all_genes:
            g_upper = g.upper()
            # Check for substring match, prefix match, or similar characters
            if (gene_upper in g_upper or
                g_upper in gene_upper or
                g_upper.startswith(gene_upper[:3]) if len(gene_upper) >= 3 else False):
                similar_genes.append(g)

        if similar_genes:
            print(f"\nSimilar gene names:")
            gene_counts = results_df[results_df[gene_col].isin(similar_genes)][gene_col].value_counts().head(10)
            for g, count in gene_counts.items():
                print(f"  {g}: {count} cells")

        print(f"\nMost abundant genes (top 20):")
        gene_counts = results_df[gene_col].value_counts().head(20)
        for g, count in gene_counts.items():
            print(f"  {g}: {count} cells")
        return

    # Get guide column name (sgRNA or barcode)
    guide_col = "sgRNA" if "sgRNA" in single_gene_df.columns else "barcode"

    # GL_MAX_TEXTURE_SIZE limit - avoid downsampling
    GL_MAX_TEXTURE_SIZE = 16384
    cell_size_with_border_est = crop_size + 2  # 1 pixel border on each side
    max_cells_per_row = (GL_MAX_TEXTURE_SIZE // 2) // cell_size_with_border_est  # Use half the limit for safety

    # Warn if user's grid_width would exceed safe texture width (napari will downsample)
    requested_row_px = grid_width * cell_size_with_border_est
    if grid_width > max_cells_per_row:
        print(
            f"WARNING: grid_width={grid_width} at crop_size={crop_size} produces a "
            f"{requested_row_px}px-wide texture, exceeding the {max_cells_per_row}-cell "
            f"({max_cells_per_row * cell_size_with_border_est}px) safe limit "
            f"(GL_MAX_TEXTURE_SIZE={GL_MAX_TEXTURE_SIZE}). napari will downsample the "
            f"image. Capping grid_width to {max_cells_per_row}."
        )
        grid_width = max_cells_per_row

    if by_guide:
        # Organize by guide: cells grouped by guide, wrapping to multiple rows if needed
        guides = single_gene_df[guide_col].dropna().unique()
        print(f"Found {len(guides)} unique guides for gene '{gene}'")

        # Sample cells per guide to fit max_cells
        cells_per_guide = max(1, max_cells // len(guides)) if len(guides) > 0 else max_cells
        sampled_dfs = []
        for g in guides:
            guide_df = single_gene_df[single_gene_df[guide_col] == g]
            if len(guide_df) > cells_per_guide:
                guide_df = guide_df.sample(n=cells_per_guide, random_state=42)
            sampled_dfs.append(guide_df)

        if sampled_dfs:
            single_gene_df = pd.concat(sampled_dfs, ignore_index=True)
        print(f"Sampled {len(single_gene_df)} cells across {len(guides)} guides ({cells_per_guide} max per guide)")

        # Sort by guide so cells are grouped
        single_gene_df = single_gene_df.sort_values(by=guide_col).reset_index(drop=True)
        # Note: guides with more cells than grid_width will wrap (handled in cell_positions below)
    else:
        # Sample cells if we have more than max_cells
        if len(single_gene_df) > max_cells:
            single_gene_df = single_gene_df.sample(n=max_cells, random_state=42)
            print(f"Sampled {max_cells} cells for display")

    single_gene_df = single_gene_df.reset_index(drop=True)

    # Ensure "gene_name" column exists for BaseDataset (expects it for label lookup)
    if "gene_name" not in single_gene_df.columns and gene_col != "gene_name":
        single_gene_df["gene_name"] = single_gene_df[gene_col]

    # Create stores dict for BaseDataset
    stores = {"pheno_assembled_v3": pheno_store}

    # Get channel info
    channel_names = pheno_store.channel_names
    n_channels = len(channel_names)

    # Create BaseDataset (never apply mask internally - we'll handle it separately)
    base_dataset = BaseDataset(
        stores=stores,
        labels_df=single_gene_df,
        initial_yx_patch_size=(crop_size, crop_size),
        final_yx_patch_size=(crop_size, crop_size),
        out_channels="all",
        mask_cell=False,
    )

    # 4i mode: monkey-patch mask loader to use 4i_cell_seg instead of cell_seg.
    # __getitem__ calls add_mask_to_batch (hardcoded to 'cell_seg'), so we override it.
    if four_i:
        print("Using '4i_cell_seg' label for masks")
        import types

        def _add_mask_to_batch_4i(self, ci, bbox):
            mask_h = bbox[2] - bbox[0]
            mask_w = bbox[3] - bbox[1]
            if pd.isna(ci.segmentation_id):
                return np.ones((1, mask_h, mask_w), dtype=bool)
            try:
                mask_arr = self.load_label_array(ci.store_key, ci.well, "4i_cell_seg", bbox)
                if mask_arr is None:
                    return np.ones((1, mask_h, mask_w), dtype=bool)
                sc_mask = (mask_arr == int(ci.segmentation_id))
                return np.expand_dims(sc_mask, axis=0)
            except Exception as e:
                print(f"  Warning: 4i_cell_seg load failed for {ci.well}: {e}")
                return np.ones((1, mask_h, mask_w), dtype=bool)

        base_dataset.add_mask_to_batch = types.MethodType(_add_mask_to_batch_4i, base_dataset)

    # Cache channel index once (instead of re-reading zattrs per cell)
    _cached_channel_index = list(range(n_channels))
    _cached_channel_names = list(channel_names)

    def _get_channels_cached(self, ci):
        return _cached_channel_names, _cached_channel_index

    import types as _types
    base_dataset._get_channels = _types.MethodType(_get_channels_cached, base_dataset)

    # Calculate grid dimensions
    n_cells = len(single_gene_df)
    cell_size_with_border = crop_size + 2  # 1 pixel border on each side

    # Add spacing between rows for guide labels
    label_spacing = 20 if by_guide else 0  # pixels for text between rows
    row_height = cell_size_with_border + label_spacing

    # For by_guide mode, calculate rows needed considering that guides can wrap
    if by_guide:
        # Build a mapping of cell index -> (row, col) and track guide boundaries
        guide_row_mapping = []  # List of (guide_name, start_row, num_rows)
        cell_positions = []  # List of (row, col) for each cell
        current_row = 0
        current_col = 0
        prev_guide = None

        for i, row_data in single_gene_df.iterrows():
            cell_guide = row_data[guide_col]

            # Check if we're starting a new guide
            if cell_guide != prev_guide:
                # New guide starts on a new row (only if we're not already at start of a row)
                if prev_guide is not None and current_col > 0:
                    current_row += 1
                current_col = 0
                guide_start_row = current_row
                prev_guide = cell_guide

            # Record position for this cell
            cell_positions.append((current_row, current_col))

            # Move to next column
            current_col += 1

            # Wrap to next row if we exceed grid_width
            if current_col >= grid_width:
                current_col = 0
                current_row += 1

        # Calculate total rows needed
        grid_h = current_row + 1 if current_col > 0 else current_row

        # Build guide row mapping for labels
        prev_guide = None
        guide_start_row = 0
        for idx, (row_idx, col_idx) in enumerate(cell_positions):
            cell_guide = single_gene_df.iloc[idx][guide_col]
            if cell_guide != prev_guide:
                if prev_guide is not None:
                    guide_row_mapping.append((prev_guide, guide_start_row, row_idx - guide_start_row))
                guide_start_row = row_idx
                prev_guide = cell_guide
        # Add the last guide
        if prev_guide is not None:
            last_row = cell_positions[-1][0] if cell_positions else 0
            guide_row_mapping.append((prev_guide, guide_start_row, last_row - guide_start_row + 1))
    else:
        grid_h = int(np.ceil(n_cells / grid_width))
        cell_positions = None
        guide_row_mapping = None

    # Initialize tiled array for all channels
    tiled_array = np.zeros(
        (n_channels, grid_h * row_height, grid_width * cell_size_with_border),
        dtype=np.float32,
    )

    # Initialize mask arrays
    from scipy.ndimage import binary_dilation
    # Binary mask (no dilation) - shows the exact segmentation
    binary_mask_array = np.zeros(
        (grid_h * row_height, grid_width * cell_size_with_border),
        dtype=np.uint8,
    )
    # Inverse mask (with dilation) - carves out the cell from background
    inverse_mask_array = np.ones(
        (grid_h * row_height, grid_width * cell_size_with_border),
        dtype=np.uint8,
    )
    print(f"Inverse mask dilation: {mask_dilation} pixels")

    # Determine centroid column names (y_pheno/x_pheno or fallbacks)
    y_col = next((c for c in ["y_pheno", "centroid_y", "y"] if c in single_gene_df.columns), None)
    x_col = next((c for c in ["x_pheno", "centroid_x", "x"] if c in single_gene_df.columns), None)

    coord_points = []  # (y, x) positions in the tiled grid
    coord_texts = []   # "well, y, x" strings

    # Parallel cell loading: zarr reads are I/O-bound and release the GIL, so
    # ThreadPoolExecutor with oversubscription is fine (and better than what
    # resource_manager reports in interactive/noVNC sessions with tight cgroups).
    from concurrent.futures import ThreadPoolExecutor
    n_workers = min(n_cells, max(8, os.cpu_count() or 8))
    print(f"Loading {n_cells} cells into {grid_h}x{grid_width} grid ({n_workers} parallel workers)...")

    def _load_cell(i):
        batch = base_dataset[i]
        return i, batch["data"].numpy(), batch["mask"].numpy()

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        loaded = list(ex.map(_load_cell, range(n_cells)))

    for i, data, mask in loaded:
        # Add 1-pixel border (padding)
        crop_with_border = np.pad(
            data,
            ((0, 0), (1, 1), (1, 1)),
            mode="constant",
            constant_values=0,
        )

        # Get row/col position - use precomputed positions for by_guide mode
        if by_guide and cell_positions is not None:
            row, col = cell_positions[i]
        else:
            row = i // grid_width
            col = i % grid_width

        # Collect coordinate annotation for this cell
        if y_col and x_col:
            cell_row = single_gene_df.iloc[i]
            well_label = cell_row["well"].replace("/0", "").replace("/", "")
            cy = cell_row.get(y_col, 0)
            cx = cell_row.get(x_col, 0)
            # Place text at top-left corner of the cell tile
            grid_y = row * row_height + label_spacing + 4
            grid_x = col * cell_size_with_border + 4
            coord_points.append([grid_y, grid_x])
            coord_texts.append(f"{well_label} ({int(cy)},{int(cx)})")

        # Position cells below the label spacing area
        y_start = row * row_height + label_spacing
        x_start = col * cell_size_with_border
        tiled_array[
            :,
            y_start:y_start + cell_size_with_border,
            x_start:x_start + cell_size_with_border,
        ] = crop_with_border

        # Add masks to arrays
        cell_mask = mask[0].astype(bool)  # (H, W)

        # Binary mask (no dilation) - exact segmentation outline
        # Use label ID 4 for better default color
        binary_mask_with_border = np.pad(
            (cell_mask.astype(np.uint8) * 4),
            ((1, 1), (1, 1)),
            mode="constant",
            constant_values=0,
        )
        binary_mask_array[
            y_start:y_start + cell_size_with_border,
            x_start:x_start + cell_size_with_border,
        ] = binary_mask_with_border

        # Inverse mask (with dilation) - carve out cell from background
        dilated_mask = cell_mask
        if mask_dilation > 0:
            dilated_mask = binary_dilation(cell_mask, iterations=mask_dilation)
        # Inverse: 0 where cell is, 1 elsewhere (for carving out)
        inverse_mask = (~dilated_mask).astype(np.uint8)
        inverse_mask_with_border = np.pad(
            inverse_mask,
            ((1, 1), (1, 1)),
            mode="constant",
            constant_values=1,  # Border should be background (1)
        )
        inverse_mask_array[
            y_start:y_start + cell_size_with_border,
            x_start:x_start + cell_size_with_border,
        ] = inverse_mask_with_border

    # Create napari viewer
    viewer = napari.Viewer()

    # Color mapping for channels
    color_dict = {
        "GFP": "green",
        "mCherry": "magenta",
        "Phase": "gray",
        "Phase2D": "gray",
        "BF": "gray",
        "VS": "gray",
        "Cy5": "cyan",
        "farred": "cyan",
        "Focus3D": "gray",
        "nuclei_prediction": "blue",
        "membrane_prediction": "magenta",
    }

    # Channels to hide by default
    hidden_channels = {"Focus3D", "nuclei_prediction", "membrane_prediction"}

    # Channels with custom contrast limits
    contrast_limits_dict = {
        "Phase2D": (-0.5, 0.8),
        "Focus3D": (-0.5, 0.8),
    }

    # In --4i mode, hide all 4i_* channels except the single p21 channel
    four_i_visible = None
    if four_i:
        four_i_visible = next((c for c in channel_names if c.endswith("_p21")), None)
        if four_i_visible:
            print(f"4i mode: only '{four_i_visible}' visible; all other 4i_* channels hidden")

    # Add each channel individually to control visibility and contrast limits
    for ch_idx, ch_name in enumerate(channel_names):
        colormap = color_dict.get(ch_name, "gray")
        visible = ch_name not in hidden_channels
        if four_i and ch_name.startswith("4i_"):
            visible = (ch_name == four_i_visible)
        contrast_limits = contrast_limits_dict.get(ch_name, None)

        layer = viewer.add_image(
            tiled_array[ch_idx],
            name=f"{display_gene}_{ch_name}",
            colormap=colormap,
            blending="additive",
            visible=visible,
            contrast_limits=contrast_limits,
        )
        layer.gamma = 0.75

    # Add mask layers
    # Binary mask - exact segmentation (no dilation) - uses label ID 4
    from napari.utils.colormaps import DirectLabelColormap
    seg_mask_layer = viewer.add_labels(
        binary_mask_array,
        name="Segmentation Mask",
        opacity=0.12,
        visible=False,
        colormap=DirectLabelColormap(color_dict={4: "red", None: "transparent"}),  # Red segmentation outline
    )

    # Inverse mask - carves out the cell (with dilation)
    inv_mask_layer = viewer.add_labels(
        inverse_mask_array,
        name=f"Inverse Mask (dilation={mask_dilation}px)",
        opacity=0.2,
        colormap=DirectLabelColormap(color_dict={1: (0.2, 0.25, 0.9, 1.0), None: "transparent"}),  # bluish-purple inverse mask
    )

    # Add guide labels as text overlay if --by-guide
    if by_guide and guide_row_mapping:
        # Create points in the spacing area above each guide's first row
        label_points = []
        label_texts = []
        for guide_name, start_row, num_rows in guide_row_mapping:
            # Position label in the spacing area above the guide's first row
            y_pos = start_row * row_height + label_spacing // 2
            label_points.append([y_pos, 5])  # y, x coordinates
            rows_str = f" ({num_rows} rows)" if num_rows > 1 else ""
            label_texts.append(f"{guide_name}{rows_str}")

        # Add as points layer with text
        text_properties = {
            "string": label_texts,
            "color": "white",
            "size": 12,
            "anchor": "upper_left",
        }
        viewer.add_points(
            np.array(label_points),
            name="Guide Labels",
            text=text_properties,
            size=0,
            face_color="transparent",
        )
        print(f"Added {len(guide_row_mapping)} guide labels")

    # Add cell coordinate annotations (hidden by default)
    if coord_points:
        coord_text_properties = {
            "string": coord_texts,
            "color": "yellow",
            "size": 16,
            "anchor": "upper_left",
        }
        coord_layer = viewer.add_points(
            np.array(coord_points),
            name="Cell Coordinates",
            text=coord_text_properties,
            size=0,
            face_color="transparent",
            visible=False,
        )

    # Add title
    viewer.title = f"{experiment} - {display_gene} ({n_cells} cells)"

    print(f"Displaying {n_cells} cells for '{display_gene}'")
    print(f"Channels: {channel_names}")

    napari.run()

    return


if __name__ == "__main__":
    view_by_gene_cli()
