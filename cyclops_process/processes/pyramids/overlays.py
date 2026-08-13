"""ISS gene/guide + grid overlay rendering for pyramids."""
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict, List
import dask.array as da
import numpy as np
import json
import pandas as pd
import logging
from tqdm import tqdm
from iohub import open_ome_zarr

from cyclops_process.napari.dask.dask_utils import (
    _load_total_translations,
    _synthesize_edges_from_rects,
    _candidate_pos_prefixes,
)
from cyclops_process.napari.dask.channel_clims import (
    match_profile,
    compute_position_clims,
)
from cyclops_utils.io.zarr_utils import (
    _iter_position_paths,
    write_component_attrs,
    write_zarr_slice_direct,
    list_numeric_levels,
    get_level0_shape,
    get_channel_dim,
    ensure_pyramid_levels,
    enumerate_units,
    add_missing_zarr_metadata,
    detect_zarr_format,
    has_zarr_array_metadata,
    create_zarr_array,
)
import zarr
from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.data.filesystem import (
    decide_overwrite_resume_skip,
    prompt_overwrite_resume_skip,
)

from joblib import Parallel, delayed
from cyclops_utils.hpc.resource_manager import get_optimal_workers

from cyclops_process.processes.pyramids.reshard import _reshard_overlay_arrays
from cyclops_process.paths import BASE_PATH

# Standard gene-name column candidates (checked in order)
_DEFAULT_GENE_COLUMNS = ["perturbation", "gene_name", "Gene name", "Gene_name", "Gene Name"]

def _apply_library_map_overrides(ds: OpsDataset, experiment: str) -> None:
    """
    Load ops_library_map.yaml and apply per-experiment overrides to *ds*.

    OpsDataset.__init__ sets ``self.library_map`` but never loads/applies it.
    This helper reads the YAML, resolves defaults + experiment overrides, and
    calls ``ds.apply_experiment_config(...)`` so fields like
    ``gene_name_output_column`` and ``iss_secondary_gene_column`` are populated.
    """
    import yaml

    lm_path = ds.library_map
    if not lm_path.exists():
        return
    with open(lm_path, "r") as f:
        lm = yaml.safe_load(f) or {}

    # Merge defaults with experiment-specific overrides
    # Strip experiment prefix (e.g. "ops0138_20260305" -> "ops0138")
    exp_key = experiment.split("_")[0]
    defaults = lm.get("default", {})
    overrides = lm.get("overrides", {}).get(exp_key, {})
    if not overrides:
        return

    merged = {**defaults, **overrides}
    ds.apply_experiment_config(merged)


def _resolve_gene_columns(experiment: Optional[str] = None) -> Tuple[List[str], Optional[str]]:
    """
    Return (candidate_columns, secondary_column) for gene-name labelling.

    Parameters
    ----------
    experiment : Optional[str]
        Experiment name. If given and the library map defines a custom
        ``gene_name_output_column`` (a custom perturbation-name column for some
        experiments), that column is prepended so it is checked first.

    Returns
    -------
    candidates : List[str]
        Ordered list of candidate primary gene-name column names.
    secondary : Optional[str]
        If the library map defines ``iss_secondary_gene_column``
        (e.g. ``gene_target``), return it so labels can be combined
        as ``"PRIMARY | SECONDARY"``.  None for standard experiments.
    """
    extra: List[str] = []
    secondary: Optional[str] = None
    if experiment is not None:
        try:
            ds = OpsDataset(experiment)
            # OpsDataset.__init__ doesn't load the library map automatically;
            # load it here and apply overrides so gene_name_output_column and
            # iss_secondary_gene_column are set from ops_library_map.yaml.
            _apply_library_map_overrides(ds, experiment)
            if ds.gene_name_output_column:
                extra = [ds.gene_name_output_column]
            if ds.iss_secondary_gene_column:
                secondary = ds.iss_secondary_gene_column
        except Exception:
            pass
    # Custom column first, then the standard fallbacks
    return extra + _DEFAULT_GENE_COLUMNS, secondary


# -----------------------------
# ISS Overlay Color Helpers
# -----------------------------
def _get_gene_color_palette(
    gene_names: Sequence[str],
    colormap: str = "hsv",
    existing_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    min_brightness: float = 0.85,
) -> Dict[str, Tuple[int, int, int]]:
    """
    Generate consistent, deterministic bright colors for genes.

    Same gene always gets the same color across runs due to sorted ordering.
    All colors are boosted to ensure high brightness for visibility on dark backgrounds.

    Parameters
    ----------
    gene_names : Sequence[str]
        List of gene names (can contain duplicates)
    colormap : str
        Matplotlib colormap name (default: "hsv" for bright, distinct colors)
    existing_colors : Optional[Dict]
        Pre-existing gene→color mapping to use/extend
    min_brightness : float
        Minimum brightness (0-1) for generated colors (default: 0.85)

    Returns
    -------
    Dict[str, Tuple[int, int, int]]
        Mapping of gene name to RGB tuple (0-255)
    """
    import colorsys
    import matplotlib.pyplot as plt

    if existing_colors is None:
        existing_colors = {}

    # Get unique genes, sorted for determinism
    unique_genes = sorted(set(gene_names))

    # Filter out genes that already have colors
    genes_needing_colors = [g for g in unique_genes if g not in existing_colors]

    if genes_needing_colors:
        # Separate NTC genes from regular genes
        ntc_genes = [g for g in genes_needing_colors if g.startswith("NTC_") or g == "NTC"]
        regular_genes = [g for g in genes_needing_colors if not (g.startswith("NTC_") or g == "NTC")]

        # Assign bright white to all NTC genes
        for gene in ntc_genes:
            existing_colors[gene] = (255, 255, 255)

        # Generate colormap colors only for regular genes
        if regular_genes:
            cmap = plt.get_cmap(colormap, len(regular_genes))
            for i, gene in enumerate(regular_genes):
                rgba = cmap(i)
                r, g, b = rgba[0], rgba[1], rgba[2]

                # Convert to HSV and boost brightness/saturation
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                # Ensure high saturation and brightness
                s = max(s, 0.7)  # Keep colors vivid
                v = max(v, min_brightness)  # Ensure bright
                r, g, b = colorsys.hsv_to_rgb(h, s, v)

                existing_colors[gene] = (
                    int(r * 255),
                    int(g * 255),
                    int(b * 255),
                )

    return existing_colors


def _get_guide_color_variants(
    gene_colors: Dict[str, Tuple[int, int, int]],
    guide_to_gene: Dict[str, str],
) -> Dict[str, Tuple[int, int, int]]:
    """
    Generate color variants for guide sequences based on their parent gene color.

    Each gene can have up to 4 unique guide sequences. Each guide gets a slightly
    different color variant of the gene's base color (shifted in hue and saturation).

    Parameters
    ----------
    gene_colors : Dict[str, Tuple[int, int, int]]
        Mapping of gene name to RGB tuple (0-255)
    guide_to_gene : Dict[str, str]
        Mapping of guide sequence (first 10 bases) to gene name

    Returns
    -------
    Dict[str, Tuple[int, int, int]]
        Mapping of guide sequence to RGB tuple (0-255)
    """
    import colorsys

    guide_colors = {}

    # Group guides by gene
    gene_to_guides: Dict[str, list] = {}
    for guide, gene in guide_to_gene.items():
        if gene not in gene_to_guides:
            gene_to_guides[gene] = []
        gene_to_guides[gene].append(guide)

    # Sort guides within each gene for deterministic ordering
    for gene in gene_to_guides:
        gene_to_guides[gene] = sorted(gene_to_guides[gene])

    # Generate color variants for each guide
    for gene, guides in gene_to_guides.items():
        if gene not in gene_colors:
            # Default bright white for unknown/NTC genes
            base_color = (255, 255, 255)
        else:
            base_color = gene_colors[gene]

        # Convert base color to HSV
        r, g, b = base_color[0] / 255.0, base_color[1] / 255.0, base_color[2] / 255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)

        # Generate variants by shifting hue slightly for each guide
        # Typical gene has 4 guides, so shift by small amounts
        n_guides = len(guides)
        for i, guide in enumerate(guides):
            # Shift hue by small increments (-0.03 to +0.03 range)
            # Also slightly vary saturation to make guides more distinguishable
            if n_guides > 1:
                hue_shift = (i - (n_guides - 1) / 2) * 0.02  # Center around base hue
                sat_shift = (i - (n_guides - 1) / 2) * 0.05  # Slight saturation variation
            else:
                hue_shift = 0
                sat_shift = 0

            new_h = (h + hue_shift) % 1.0  # Wrap hue around
            new_s = max(0.4, min(1.0, s + sat_shift))  # Keep saturation in valid range
            new_v = max(0.7, v)  # Ensure brightness

            new_r, new_g, new_b = colorsys.hsv_to_rgb(new_h, new_s, new_v)
            guide_colors[guide] = (
                int(new_r * 255),
                int(new_g * 255),
                int(new_b * 255),
            )

    return guide_colors


def _render_iss_guide_tile(
    coords_yx: np.ndarray,
    guide_labels: np.ndarray,
    guide_colors: Dict[str, Tuple[int, int, int]],
    tile_y_start: int,
    tile_x_start: int,
    tile_h: int,
    tile_w: int,
    dot_radius: int = 3,
    font_size: int = 12,
    render_text: bool = True,
    y_offset: int = 0,
    failed_rounds: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    Render a single tile of the ISS guide overlay.

    Similar to _render_iss_tile but for guide sequences, with text positioned
    below and to the right of the dot (offset from gene name position).

    Supports rendering failed/dropout rounds in white while keeping valid bases
    in their guide color.

    Parameters
    ----------
    coords_yx : np.ndarray
        Shape (N, 2) array of Y, X coordinates (global coordinates)
    guide_labels : np.ndarray
        Shape (N,) array of guide sequence strings (first 10 bases)
    guide_colors : Dict[str, Tuple[int, int, int]]
        Mapping of guide sequence to RGB tuple
    tile_y_start, tile_x_start : int
        Top-left corner of tile in global coordinates
    tile_h, tile_w : int
        Height and width of this tile
    dot_radius : int
        Radius of dots in pixels
    font_size : int
        Font size for text labels
    render_text : bool
        Whether to render text labels
    y_offset : int
        Vertical offset for text (to position below gene name)
    failed_rounds : Optional[Sequence[int]]
        List of round indices (0-9) that are dropouts. These bases will be
        rendered in gray while valid bases keep their guide color.

    Returns
    -------
    np.ndarray
        RGBA uint8 array of shape (tile_h, tile_w, 4)
    """
    from PIL import Image, ImageDraw, ImageFont

    # Create transparent RGBA canvas for this tile
    img = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Convert failed_rounds to a set for fast lookup
    failed_set = set(failed_rounds) if failed_rounds else set()
    dropout_color = (255, 255, 255)  # White for dropout bases

    # Try to load a good font, fall back to default
    font = None
    if render_text:
        try:
            for font_name in [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/TTF/DejaVuSans.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "arial.ttf",
            ]:
                try:
                    font = ImageFont.truetype(font_name, font_size)
                    break
                except (OSError, IOError):
                    continue
            if font is None:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

    # Pre-compute disk template
    r = max(1, dot_radius)
    rrng = np.arange(-r, r + 1, dtype=np.int32)
    dy, dx = np.meshgrid(rrng, rrng, indexing="ij")
    disk_mask = (dy * dy + dx * dx) <= (r * r)
    disk_offsets = np.stack([dy[disk_mask], dx[disk_mask]], axis=1)

    default_color = (255, 255, 255)  # Bright white for unknown/NTC guides

    # Tile boundaries with padding for dots/text that may overlap
    text_padding = font_size * 12 if render_text else 0  # Approximate max text width
    y_min = tile_y_start - r - y_offset - font_size - 1
    y_max = tile_y_start + tile_h + r + y_offset + font_size + 1
    x_min = tile_x_start - r - text_padding - 1
    x_max = tile_x_start + tile_w + r + 1

    # Filter points that could affect this tile
    in_tile = (
        (coords_yx[:, 0] >= y_min) & (coords_yx[:, 0] < y_max) &
        (coords_yx[:, 1] >= x_min) & (coords_yx[:, 1] < x_max)
    )
    tile_indices = np.where(in_tile)[0]

    # Render points in this tile
    for i in tile_indices:
        y_c, x_c = int(coords_yx[i, 0]), int(coords_yx[i, 1])
        label = str(guide_labels[i]) if i < len(guide_labels) else ""
        color = guide_colors.get(label, default_color)

        # Convert to tile-local coordinates
        y_local = y_c - tile_y_start
        x_local = x_c - tile_x_start

        # Draw colored disk (same position as gene dot)
        for dy_off, dx_off in disk_offsets:
            py, px = y_local + dy_off, x_local + dx_off
            if 0 <= py < tile_h and 0 <= px < tile_w:
                draw.point((px, py), fill=(*color, 255))

        # Draw text label if enabled - positioned BELOW the gene name
        # Gene name is at (x + r + 2, y - font_size // 2)
        # Guide text is offset down by y_offset (typically font_size + 4)
        if render_text and font is not None and label:
            text_x = x_local + r + 2
            text_y = y_local - font_size // 2 + y_offset

            # If we have failed rounds, render each character separately
            # with dropout bases in white and valid bases in the guide color
            if failed_set:
                current_x = text_x
                for char_idx, char in enumerate(label):
                    if char_idx in failed_set:
                        # Dropout round - render in white
                        char_color = dropout_color
                    else:
                        # Valid round - render in guide color
                        char_color = color
                    draw.text((current_x, text_y), char, font=font, fill=(*char_color, 255))
                    # Get character width for positioning next character
                    try:
                        char_bbox = font.getbbox(char)
                        char_width = char_bbox[2] - char_bbox[0]
                    except AttributeError:
                        # Fallback for older PIL versions
                        char_width = font_size * 0.6
                    current_x += char_width
            else:
                # No failed rounds - render entire string in guide color
                draw.text((text_x, text_y), label, font=font, fill=(*color, 255))

    return np.array(img)


def _render_iss_tile(
    coords_yx: np.ndarray,
    labels: np.ndarray,
    gene_colors: Dict[str, Tuple[int, int, int]],
    tile_y_start: int,
    tile_x_start: int,
    tile_h: int,
    tile_w: int,
    dot_radius: int = 3,
    font_size: int = 12,
    render_text: bool = True,
) -> np.ndarray:
    """
    Render a single tile of the ISS overlay.

    Parameters
    ----------
    coords_yx : np.ndarray
        Shape (N, 2) array of Y, X coordinates (global coordinates)
    labels : np.ndarray
        Shape (N,) array of gene name strings
    gene_colors : Dict[str, Tuple[int, int, int]]
        Mapping of gene name to RGB tuple
    tile_y_start, tile_x_start : int
        Top-left corner of tile in global coordinates
    tile_h, tile_w : int
        Height and width of this tile
    dot_radius : int
        Radius of dots in pixels
    font_size : int
        Font size for text labels
    render_text : bool
        Whether to render text labels

    Returns
    -------
    np.ndarray
        RGBA uint8 array of shape (tile_h, tile_w, 4)
    """
    from PIL import Image, ImageDraw, ImageFont

    # Create transparent RGBA canvas for this tile
    img = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Try to load a good font, fall back to default
    font = None
    if render_text:
        try:
            for font_name in [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/TTF/DejaVuSans.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "arial.ttf",
            ]:
                try:
                    font = ImageFont.truetype(font_name, font_size)
                    break
                except (OSError, IOError):
                    continue
            if font is None:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

    # Pre-compute disk template
    r = max(1, dot_radius)
    rrng = np.arange(-r, r + 1, dtype=np.int32)
    dy, dx = np.meshgrid(rrng, rrng, indexing="ij")
    disk_mask = (dy * dy + dx * dx) <= (r * r)
    disk_offsets = np.stack([dy[disk_mask], dx[disk_mask]], axis=1)

    default_color = (255, 255, 255)  # Bright white for unknown/NTC genes

    # Tile boundaries with padding for dots/text that may overlap
    text_padding = font_size * 10 if render_text else 0  # Approximate max text width
    y_min = tile_y_start - r - 1
    y_max = tile_y_start + tile_h + r + 1
    x_min = tile_x_start - r - text_padding - 1
    x_max = tile_x_start + tile_w + r + 1

    # Filter points that could affect this tile
    in_tile = (
        (coords_yx[:, 0] >= y_min) & (coords_yx[:, 0] < y_max) &
        (coords_yx[:, 1] >= x_min) & (coords_yx[:, 1] < x_max)
    )
    tile_indices = np.where(in_tile)[0]

    # Render points in this tile
    for i in tile_indices:
        y_c, x_c = int(coords_yx[i, 0]), int(coords_yx[i, 1])
        label = str(labels[i]) if i < len(labels) else ""
        color = gene_colors.get(label, default_color)

        # Convert to tile-local coordinates
        y_local = y_c - tile_y_start
        x_local = x_c - tile_x_start

        # Draw colored disk
        for dy_off, dx_off in disk_offsets:
            py, px = y_local + dy_off, x_local + dx_off
            if 0 <= py < tile_h and 0 <= px < tile_w:
                draw.point((px, py), fill=(*color, 255))

        # Draw text label if enabled
        if render_text and font is not None and label:
            text_x = x_local + r + 2
            text_y = y_local - font_size // 2
            draw.text((text_x, text_y), label, font=font, fill=(*color, 255))

    return np.array(img)


def _process_single_iss_tile(
    tile_spec: dict,
) -> str:
    """
    Process a single ISS tile work unit. Designed for parallel execution.

    Parameters
    ----------
    tile_spec : dict
        Dictionary containing all info needed to render one tile:
        - output_store: Path to zarr store
        - output_component: Component path within zarr store
        - coords_yx: Shape (N, 2) array of Y, X coordinates
        - labels: Shape (N,) array of gene name strings
        - gene_colors: Mapping of gene name to RGB tuple
        - y_start, x_start, tile_h, tile_w: Tile bounds
        - dot_radius, font_size, render_text: Rendering params
        - pos, lvl: For logging

    Returns
    -------
    str
        Status message for this tile
    """
    try:
        # Open zarr array for writing
        z = zarr.open(f"{tile_spec['output_store']}/{tile_spec['output_component']}", mode="r+")

        # Render tile
        tile_data = _render_iss_tile(
            coords_yx=tile_spec['coords_yx'],
            labels=tile_spec['labels'],
            gene_colors=tile_spec['gene_colors'],
            tile_y_start=tile_spec['y_start'],
            tile_x_start=tile_spec['x_start'],
            tile_h=tile_spec['tile_h'],
            tile_w=tile_spec['tile_w'],
            dot_radius=tile_spec['dot_radius'],
            font_size=tile_spec['font_size'],
            render_text=tile_spec['render_text'],
        )

        # Write to zarr
        z[tile_spec['y_start']:tile_spec['y_start'] + tile_spec['tile_h'],
          tile_spec['x_start']:tile_spec['x_start'] + tile_spec['tile_w'], :] = tile_data

        return f"[OK] {tile_spec['pos']}/lvl{tile_spec['lvl']}/tile({tile_spec['y_start']},{tile_spec['x_start']})"
    except Exception as e:
        return f"[ERROR] {tile_spec['pos']}/lvl{tile_spec['lvl']}/tile({tile_spec['y_start']},{tile_spec['x_start']}): {e}"


def _init_iss_zarr_array(
    output_store: str,
    output_component: str,
    canvas_shape: Tuple[int, int],
    zarr_format: int = 3,
    tile_size: int = 4096,
) -> None:
    """
    Initialize an empty zarr array for ISS overlay rendering.

    Called once per position/level before parallel tile processing.
    Creates UNSHARDED arrays to avoid concurrent write conflicts and
    optimize for sequential access patterns. Arrays remain unsharded
    permanently for better I/O performance with sparse overlay data.
    """
    h, w = canvas_shape
    chunk_size = min(1024, tile_size)

    create_zarr_array(
        path=f"{output_store}/{output_component}",
        shape=(h, w, 4),
        chunks=(chunk_size, chunk_size, 4),
        dtype=np.uint8,
        zarr_format=zarr_format,
        fill_value=0,
        overwrite=True,
        shards_ratio=(1, 1, 1),  # Disable sharding to avoid concurrent write conflicts
    )


def _render_iss_tiled_to_zarr(
    coords_yx: np.ndarray,
    labels: np.ndarray,
    gene_colors: Dict[str, Tuple[int, int, int]],
    canvas_shape: Tuple[int, int],
    output_store: str,
    output_component: str,
    dot_radius: int = 3,
    font_size: int = 12,
    render_text: bool = True,
    tile_size: int = 4096,
    zarr_format: int = 3,
) -> None:
    """
    Render ISS overlay directly to zarr in tiles to avoid memory issues.

    NOTE: This is the sequential version used when called from single-position
    processing. For batch processing, use _collect_iss_tile_specs() and
    _process_single_iss_tile() with joblib for better parallelization.

    Parameters
    ----------
    coords_yx : np.ndarray
        Shape (N, 2) array of Y, X coordinates
    labels : np.ndarray
        Shape (N,) array of gene name strings
    gene_colors : Dict[str, Tuple[int, int, int]]
        Mapping of gene name to RGB tuple
    canvas_shape : Tuple[int, int]
        (height, width) of output canvas
    output_store : str
        Path to zarr store
    output_component : str
        Component path within zarr store
    dot_radius : int
        Radius of dots in pixels
    font_size : int
        Font size for text labels
    render_text : bool
        Whether to render text labels
    tile_size : int
        Size of tiles to process (default 4096)
    zarr_format : int
        Zarr format version (2 or 3)
    """
    h, w = canvas_shape
    chunk_size = min(1024, tile_size)

    # Create zarr array
    z = create_zarr_array(
        path=f"{output_store}/{output_component}",
        shape=(h, w, 4),
        chunks=(chunk_size, chunk_size, 4),
        dtype=np.uint8,
        zarr_format=zarr_format,
        fill_value=0,
        overwrite=True,
    )

    # Process tiles sequentially
    n_tiles_y = (h + tile_size - 1) // tile_size
    n_tiles_x = (w + tile_size - 1) // tile_size

    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            y_start = ty * tile_size
            x_start = tx * tile_size
            tile_h = min(tile_size, h - y_start)
            tile_w = min(tile_size, w - x_start)

            tile_data = _render_iss_tile(
                coords_yx=coords_yx,
                labels=labels,
                gene_colors=gene_colors,
                tile_y_start=y_start,
                tile_x_start=x_start,
                tile_h=tile_h,
                tile_w=tile_w,
                dot_radius=dot_radius,
                font_size=font_size,
                render_text=render_text,
            )
            z[y_start:y_start + tile_h, x_start:x_start + tile_w, :] = tile_data


def _generate_tile_specs(
    canvas_shape: Tuple[int, int],
    tile_size: int = 4096,
) -> List[Tuple[int, int, int, int]]:
    """
    Generate tile specifications for a canvas.

    Returns list of (y_start, x_start, tile_h, tile_w) tuples.
    """
    h, w = canvas_shape
    n_tiles_y = (h + tile_size - 1) // tile_size
    n_tiles_x = (w + tile_size - 1) // tile_size

    tile_specs = []
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            y_start = ty * tile_size
            x_start = tx * tile_size
            tile_h = min(tile_size, h - y_start)
            tile_w = min(tile_size, w - x_start)
            tile_specs.append((y_start, x_start, tile_h, tile_w))

    return tile_specs


def _render_colored_iss_overlay(
    coords_yx: np.ndarray,
    labels: np.ndarray,
    gene_colors: Dict[str, Tuple[int, int, int]],
    canvas_shape: Tuple[int, int],
    dot_radius: int = 3,
    font_size: int = 12,
    render_text: bool = True,
) -> np.ndarray:
    """
    Render colored dots (and optionally text labels) to an RGBA array.

    NOTE: For large images (>10K pixels), use _render_iss_tiled_to_zarr instead
    to avoid memory issues.

    Parameters
    ----------
    coords_yx : np.ndarray
        Shape (N, 2) array of Y, X coordinates
    labels : np.ndarray
        Shape (N,) array of gene name strings
    gene_colors : Dict[str, Tuple[int, int, int]]
        Mapping of gene name to RGB tuple
    canvas_shape : Tuple[int, int]
        (height, width) of output canvas
    dot_radius : int
        Radius of dots in pixels
    font_size : int
        Font size for text labels (only used if render_text=True)
    render_text : bool
        Whether to render text labels (True for levels 0-1, False for 2+)

    Returns
    -------
    np.ndarray
        RGBA uint8 array of shape (H, W, 4)
    """
    h, w = canvas_shape

    # For small images, render in one go using the tile function
    return _render_iss_tile(
        coords_yx=coords_yx,
        labels=labels,
        gene_colors=gene_colors,
        tile_y_start=0,
        tile_x_start=0,
        tile_h=h,
        tile_w=w,
        dot_radius=dot_radius,
        font_size=font_size,
        render_text=render_text,
    )


def _process_single_iss_guide_tile(
    tile_spec: dict,
) -> str:
    """
    Process a single ISS guide tile work unit. Designed for parallel execution.

    Parameters
    ----------
    tile_spec : dict
        Dictionary containing all info needed to render one guide tile:
        - output_store: Path to zarr store
        - output_component: Component path within zarr store
        - coords_yx: Shape (N, 2) array of Y, X coordinates
        - guide_labels: Shape (N,) array of guide sequence strings
        - guide_colors: Mapping of guide sequence to RGB tuple
        - y_start, x_start, tile_h, tile_w: Tile bounds
        - dot_radius, font_size, render_text, y_offset: Rendering params
        - failed_rounds: List of dropout round indices
        - pos, lvl: For logging

    Returns
    -------
    str
        Status message for this tile
    """
    try:
        # Open zarr array for writing
        z = zarr.open(f"{tile_spec['output_store']}/{tile_spec['output_component']}", mode="r+")

        # Render tile
        tile_data = _render_iss_guide_tile(
            coords_yx=tile_spec['coords_yx'],
            guide_labels=tile_spec['guide_labels'],
            guide_colors=tile_spec['guide_colors'],
            tile_y_start=tile_spec['y_start'],
            tile_x_start=tile_spec['x_start'],
            tile_h=tile_spec['tile_h'],
            tile_w=tile_spec['tile_w'],
            dot_radius=tile_spec['dot_radius'],
            font_size=tile_spec['font_size'],
            render_text=tile_spec['render_text'],
            y_offset=tile_spec['y_offset'],
            failed_rounds=tile_spec['failed_rounds'],
        )

        # Write to zarr
        z[tile_spec['y_start']:tile_spec['y_start'] + tile_spec['tile_h'],
          tile_spec['x_start']:tile_spec['x_start'] + tile_spec['tile_w'], :] = tile_data

        return f"[OK] {tile_spec['pos']}/lvl{tile_spec['lvl']}/tile({tile_spec['y_start']},{tile_spec['x_start']})"
    except Exception as e:
        return f"[ERROR] {tile_spec['pos']}/lvl{tile_spec['lvl']}/tile({tile_spec['y_start']},{tile_spec['x_start']}): {e}"


def _render_iss_guide_tiled_to_zarr(
    coords_yx: np.ndarray,
    guide_labels: np.ndarray,
    guide_colors: Dict[str, Tuple[int, int, int]],
    canvas_shape: Tuple[int, int],
    output_store: str,
    output_component: str,
    dot_radius: int = 3,
    font_size: int = 12,
    render_text: bool = True,
    tile_size: int = 4096,
    zarr_format: int = 3,
    y_offset: int = 0,
    failed_rounds: Optional[Sequence[int]] = None,
) -> None:
    """
    Render ISS guide overlay directly to zarr in tiles to avoid memory issues.

    NOTE: This is the sequential version used when called from single-position
    processing. For batch processing, use _process_single_iss_guide_tile()
    with joblib for better parallelization.

    Parameters
    ----------
    coords_yx : np.ndarray
        Shape (N, 2) array of Y, X coordinates
    guide_labels : np.ndarray
        Shape (N,) array of guide sequence strings (first 10 bases)
    guide_colors : Dict[str, Tuple[int, int, int]]
        Mapping of guide sequence to RGB tuple
    canvas_shape : Tuple[int, int]
        (height, width) of output canvas
    output_store : str
        Path to zarr store
    output_component : str
        Component path within zarr store
    dot_radius : int
        Radius of dots in pixels
    font_size : int
        Font size for text labels
    render_text : bool
        Whether to render text labels
    tile_size : int
        Size of tiles to process (default 4096)
    zarr_format : int
        Zarr format version (2 or 3)
    y_offset : int
        Vertical offset for text (to position below gene name)
    failed_rounds : Optional[Sequence[int]]
        List of round indices (0-9) that are dropouts. These bases will be
        rendered in gray while valid bases keep their guide color.
    """
    h, w = canvas_shape
    chunk_size = min(1024, tile_size)

    # Create zarr array
    z = create_zarr_array(
        path=f"{output_store}/{output_component}",
        shape=(h, w, 4),
        chunks=(chunk_size, chunk_size, 4),
        dtype=np.uint8,
        zarr_format=zarr_format,
        fill_value=0,
        overwrite=True,
    )

    # Process tiles sequentially
    n_tiles_y = (h + tile_size - 1) // tile_size
    n_tiles_x = (w + tile_size - 1) // tile_size

    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            y_start = ty * tile_size
            x_start = tx * tile_size
            tile_h = min(tile_size, h - y_start)
            tile_w = min(tile_size, w - x_start)

            tile_data = _render_iss_guide_tile(
                coords_yx=coords_yx,
                guide_labels=guide_labels,
                guide_colors=guide_colors,
                tile_y_start=y_start,
                tile_x_start=x_start,
                tile_h=tile_h,
                tile_w=tile_w,
                dot_radius=dot_radius,
                font_size=font_size,
                render_text=render_text,
                y_offset=y_offset,
                failed_rounds=failed_rounds,
            )
            z[y_start:y_start + tile_h, x_start:x_start + tile_w, :] = tile_data


def _render_colored_iss_guide_overlay(
    coords_yx: np.ndarray,
    guide_labels: np.ndarray,
    guide_colors: Dict[str, Tuple[int, int, int]],
    canvas_shape: Tuple[int, int],
    dot_radius: int = 3,
    font_size: int = 12,
    render_text: bool = True,
    y_offset: int = 0,
    failed_rounds: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    Render colored guide overlay dots (and optionally text labels) to an RGBA array.

    NOTE: For large images (>10K pixels), use _render_iss_guide_tiled_to_zarr instead
    to avoid memory issues.

    Parameters
    ----------
    coords_yx : np.ndarray
        Shape (N, 2) array of Y, X coordinates
    guide_labels : np.ndarray
        Shape (N,) array of guide sequence strings (first 10 bases)
    guide_colors : Dict[str, Tuple[int, int, int]]
        Mapping of guide sequence to RGB tuple
    canvas_shape : Tuple[int, int]
        (height, width) of output canvas
    dot_radius : int
        Radius of dots in pixels
    font_size : int
        Font size for text labels (only used if render_text=True)
    render_text : bool
        Whether to render text labels (True for levels 0-1, False for 2+)
    y_offset : int
        Vertical offset for text (to position below gene name)
    failed_rounds : Optional[Sequence[int]]
        List of round indices (0-9) that are dropouts. These bases will be
        rendered in gray while valid bases keep their guide color.

    Returns
    -------
    np.ndarray
        RGBA uint8 array of shape (H, W, 4)
    """
    h, w = canvas_shape

    # For small images, render in one go using the tile function
    return _render_iss_guide_tile(
        coords_yx=coords_yx,
        guide_labels=guide_labels,
        guide_colors=guide_colors,
        tile_y_start=0,
        tile_x_start=0,
        tile_h=h,
        tile_w=w,
        dot_radius=dot_radius,
        font_size=font_size,
        render_text=render_text,
        y_offset=y_offset,
        failed_rounds=failed_rounds,
    )


def _prepare_iss_position_data(
    source_store: Path,
    pos: str,
    experiment: Optional[str],
    point_radius_px: int,
    font_size: int,
    text_levels: Sequence[int],
    gene_colors: Optional[Dict[str, Tuple[int, int, int]]],
    colormap: str,
    build_levels: Optional[Sequence[int]],
    overwrite: bool,
    zarr_format: int,
    tile_size: int = 4096,
) -> Tuple[Optional[List[dict]], Optional[List[dict]], str]:
    """
    Prepare ISS position data and collect tile specs for parallel processing.

    This function:
    1. Loads CSV data for the position
    2. Prepares coordinates and labels
    3. Creates zarr array directories and metadata
    4. Initializes empty zarr arrays for each level
    5. Returns tile specs for parallel rendering

    Returns
    -------
    Tuple of (gene_tile_specs, guide_tile_specs, status_message)
        - gene_tile_specs: List of dicts for gene overlay tiles, or None on error
        - guide_tile_specs: List of dicts for guide overlay tiles, or None on error
        - status_message: Status string (starts with [OK], [SKIP], or [ERROR])
    """
    from iohub import open_ome_zarr

    ds: Optional[OpsDataset] = None
    if experiment is not None:
        try:
            ds = OpsDataset(experiment)
        except Exception:
            ds = None

    gene_tile_specs = []
    guide_tile_specs = []

    try:
        with open_ome_zarr(Path(source_store) / pos, layout="fov", mode="r") as fov:
            level_names = list_numeric_levels(source_store, pos)
            if not level_names:
                return None, None, f"[SKIP] {pos}: No levels found"

            # Determine level 0 shape
            y0, x0 = int(fov["0"].shape[-2]), int(fov["0"].shape[-1])

            # Resolve well key and load CSV
            parts = [p for p in str(pos).split("/") if p]
            well_key = f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else str(pos)
            df = None

            if ds is not None:
                try:
                    csv_path = ds.append_well("linked_results", well_key)
                    if Path(csv_path).exists():
                        df = pd.read_csv(csv_path)
                    else:
                        return None, None, f"[SKIP] {pos}: CSV missing at {csv_path}"
                except Exception as e:
                    return None, None, f"[ERROR] {pos}: Could not resolve CSV: {e}"

            if df is None:
                return None, None, f"[SKIP] {pos}: No CSV dataframe available"

            # Prepare coordinates
            y_col, x_col = None, None
            if {"y_global_pheno", "x_global_pheno"}.issubset(df.columns):
                y_col, x_col = "y_global_pheno", "x_global_pheno"
            elif {"y_pheno", "x_pheno"}.issubset(df.columns):
                y_col, x_col = "y_pheno", "x_pheno"

            if y_col is None or x_col is None:
                return None, None, f"[ERROR] {pos}: Missing coordinate columns"

            coords = df[[y_col, x_col]].copy().dropna()
            if len(coords) == 0:
                return None, None, f"[SKIP] {pos}: No valid coordinates"

            yy = np.clip(coords[y_col].to_numpy(dtype=np.int64), 0, y0 - 1)
            xx = np.clip(coords[x_col].to_numpy(dtype=np.int64), 0, x0 - 1)
            coords_yx_np = np.stack([yy.astype(np.float32), xx.astype(np.float32)], axis=1)

            # Build labels from gene names (supports custom gene_name_output_column)
            gene_col_candidates, secondary_col = _resolve_gene_columns(experiment)
            gene = None
            gene_col_used = None
            for gene_col in gene_col_candidates:
                if gene_col in df.columns:
                    gene = df[gene_col]
                    gene_col_used = gene_col
                    break
            if secondary_col and secondary_col in df.columns:
                sec_msg = f", combined with secondary '{secondary_col}' -> labels like 'primary | secondary'"
            elif secondary_col:
                sec_msg = f" (secondary '{secondary_col}' configured but not found in CSV columns: {list(df.columns)})"
            else:
                sec_msg = ""
            print(f"[ISS] {pos}: using gene column '{gene_col_used}'{sec_msg}")

            barcode = None
            for bc_col in ["barcode", "Barcode"]:
                if bc_col in df.columns:
                    barcode = df[bc_col]
                    break

            if gene is None:
                gene = pd.Series([None] * len(df))
            if barcode is None:
                barcode = pd.Series(["UNKNOWN"] * len(df))

            primary_labels_s = gene.astype("string")
            primary_labels_s = primary_labels_s.mask(
                primary_labels_s.isna() | (primary_labels_s.str.len() == 0),
                other=("NTC_" + barcode.astype("string").fillna("UNKNOWN")),
            )

            # Combine with secondary column if available (e.g. "MB_A | KRAS")
            if secondary_col and secondary_col in df.columns:
                secondary_s = df[secondary_col].astype("string").fillna("")
                labels_s = primary_labels_s + " | " + secondary_s
                # Clean up trailing " | " for rows where secondary is empty
                labels_s = labels_s.str.replace(r"\s*\|\s*$", "", regex=True)
            else:
                labels_s = primary_labels_s

            labels_s = labels_s.loc[coords.index].astype(str)
            labels_np = labels_s.to_numpy(dtype="U64")
            # Keep primary-only labels for color generation (combined labels
            # like "MB_A | KRAS" inherit color from primary key "MB_A")
            primary_labels_np = primary_labels_s.loc[coords.index].astype(str).to_numpy(dtype="U64")

            # Guide labels (first 10 bases of barcode)
            guide_labels_np = None
            if barcode is not None:
                guide_s = barcode.loc[coords.index].astype(str).str[:10]
                guide_labels_np = guide_s.to_numpy(dtype="U10")

            # Generate gene color palette from PRIMARY labels (not combined)
            # so "MB_A" gets a color, then map combined labels like "MB_A | KRAS"
            # to the same color as their primary key
            if gene_colors is None:
                pos_gene_colors = _get_gene_color_palette(primary_labels_np.tolist(), colormap=colormap)
            else:
                pos_gene_colors = dict(gene_colors)  # copy so we don't mutate caller's dict

            # Map combined labels to their primary label's color
            for primary, combined in zip(primary_labels_np, labels_np):
                if combined not in pos_gene_colors and primary in pos_gene_colors:
                    pos_gene_colors[combined] = pos_gene_colors[primary]

            # Build guide-to-gene mapping and generate guide colors
            guide_colors_dict = None
            if guide_labels_np is not None and gene is not None:
                guide_to_gene: Dict[str, str] = {}
                gene_aligned = gene.loc[coords.index].astype(str).to_numpy()
                for i, guide in enumerate(guide_labels_np):
                    if guide not in guide_to_gene:
                        guide_to_gene[guide] = gene_aligned[i]

                # Generate guide colors as variants of gene colors
                guide_colors_dict = _get_guide_color_variants(pos_gene_colors, guide_to_gene)

            # Setup directories and metadata
            pos_dir = source_store / pos
            if zarr_format == 3:
                iss_gene_dir = pos_dir / "labels" / "iss_gene_image"
                iss_guide_dir = pos_dir / "labels" / "iss_guide_image"
            else:
                iss_gene_dir = pos_dir / "iss_gene_image"
                iss_guide_dir = pos_dir / "iss_guide_image"

            iss_gene_dir.mkdir(parents=True, exist_ok=True)
            iss_guide_dir.mkdir(parents=True, exist_ok=True)

            # Ensure .zgroup metadata exists for zarr v2 compatibility
            if zarr_format == 2:
                for iss_dir in [iss_gene_dir, iss_guide_dir]:
                    zgroup_path = iss_dir / ".zgroup"
                    if not zgroup_path.exists():
                        zgroup_path.write_text('{"zarr_format": 2}')

            # Write metadata
            if zarr_format == 3:
                from cyclops_process.convert.v3_metadata import OVERLAY_METADATA

                # Gene metadata
                zarr_json_path = iss_gene_dir / "zarr.json"
                zarr_meta = {
                    "zarr_format": 3,
                    "node_type": "group",
                    "attributes": {
                        "custom_metadata": {
                            "label_name": "iss_gene_image",
                            **OVERLAY_METADATA["iss_gene_image"],
                        }
                    }
                }
                with open(zarr_json_path, "w") as f:
                    json.dump(zarr_meta, f, indent=2)

                # Guide metadata
                zarr_json_path = iss_guide_dir / "zarr.json"
                zarr_meta = {
                    "zarr_format": 3,
                    "node_type": "group",
                    "attributes": {
                        "custom_metadata": {
                            "label_name": "iss_guide_image",
                            **OVERLAY_METADATA["iss_guide_image"],
                        }
                    }
                }
                with open(zarr_json_path, "w") as f:
                    json.dump(zarr_meta, f, indent=2)
            else:
                from cyclops_process.convert.v3_metadata import OVERLAY_METADATA
                with open(iss_gene_dir / ".zattrs", "w") as f:
                    json.dump(OVERLAY_METADATA["iss_gene_image"], f)
                with open(iss_guide_dir / ".zattrs", "w") as f:
                    json.dump(OVERLAY_METADATA["iss_guide_image"], f)

            # Determine levels to build
            lvls = sorted([int(l) for l in level_names])
            if build_levels is not None:
                lvls = [int(l) for l in build_levels if str(l) in level_names]

            # For each level: init zarr arrays and collect tile specs
            for lvl in lvls:
                if not overwrite:
                    continue

                yl, xl = int(fov[str(lvl)].shape[-2]), int(fov[str(lvl)].shape[-1])

                scale_y = y0 / max(1, yl)
                scale_x = x0 / max(1, xl)
                coords_lvl = coords_yx_np.copy()
                coords_lvl[:, 0] = coords_lvl[:, 0] / scale_y
                coords_lvl[:, 1] = coords_lvl[:, 1] / scale_x
                coords_lvl[:, 0] = np.clip(coords_lvl[:, 0], 0, yl - 1)
                coords_lvl[:, 1] = np.clip(coords_lvl[:, 1], 0, xl - 1)

                render_text = lvl in text_levels
                scale_factor = max(scale_y, scale_x)
                lvl_radius = max(3, int(point_radius_px / scale_factor) + int(lvl))
                lvl_font_size = max(16, int(font_size / scale_factor))

                # Init zarr arrays for gene and guide
                iss_gene_component = str(iss_gene_dir.relative_to(source_store) / str(lvl))
                iss_guide_component = str(iss_guide_dir.relative_to(source_store) / str(lvl))

                _init_iss_zarr_array(
                    output_store=str(source_store),
                    output_component=iss_gene_component,
                    canvas_shape=(yl, xl),
                    zarr_format=zarr_format,
                    tile_size=tile_size,
                )
                _init_iss_zarr_array(
                    output_store=str(source_store),
                    output_component=iss_guide_component,
                    canvas_shape=(yl, xl),
                    zarr_format=zarr_format,
                    tile_size=tile_size,
                )

                # Generate tile specs for this level
                raw_tile_specs = _generate_tile_specs((yl, xl), tile_size=tile_size)

                for y_start, x_start, tile_h, tile_w in raw_tile_specs:
                    # Gene tile spec
                    gene_tile_specs.append({
                        'output_store': str(source_store),
                        'output_component': iss_gene_component,
                        'coords_yx': coords_lvl,
                        'labels': labels_np,
                        'gene_colors': pos_gene_colors,
                        'y_start': y_start,
                        'x_start': x_start,
                        'tile_h': tile_h,
                        'tile_w': tile_w,
                        'dot_radius': lvl_radius,
                        'font_size': lvl_font_size,
                        'render_text': render_text,
                        'pos': pos,
                        'lvl': lvl,
                    })

                    # Guide tile spec
                    if guide_labels_np is not None:
                        guide_tile_specs.append({
                            'output_store': str(source_store),
                            'output_component': iss_guide_component,
                            'coords_yx': coords_lvl,
                            'guide_labels': guide_labels_np,
                            'guide_colors': guide_colors_dict if guide_colors_dict else pos_gene_colors,
                            'y_start': y_start,
                            'x_start': x_start,
                            'tile_h': tile_h,
                            'tile_w': tile_w,
                            'dot_radius': lvl_radius,
                            'font_size': lvl_font_size,
                            'render_text': render_text,
                            'y_offset': 40,  # guide_text_offset
                            'failed_rounds': None,  # Will be set in caller
                            'pos': pos,
                            'lvl': lvl,
                        })

        return gene_tile_specs, guide_tile_specs, f"[OK] {pos}: {len(gene_tile_specs)} gene tiles, {len(guide_tile_specs)} guide tiles"

    except Exception as e:
        import traceback
        return None, None, f"[ERROR] {pos}: {e}\n{traceback.format_exc()}"


def _build_iss_for_single_position(
    source_store: Path,
    pos: str,
    experiment: Optional[str],
    point_radius_px: int,
    font_size: int,
    text_levels: Sequence[int],
    gene_colors: Optional[Dict[str, Tuple[int, int, int]]],
    colormap: str,
    build_levels: Optional[Sequence[int]],
    overwrite: bool,
    zarr_format: int,
) -> str:
    """
    Process a single position for ISS gene name overlay building.

    Renders gene names as colored text labels with dots at ISS call positions.
    Output is written to iss_gene_image/<level> as RGBA images.

    This is a helper function designed to be called in parallel via joblib.
    Returns the position name on success, or an error message on failure.
    """
    from iohub import open_ome_zarr

    ds: Optional[OpsDataset] = None
    if experiment is not None:
        try:
            ds = OpsDataset(experiment)
        except Exception:
            ds = None

    try:
        with open_ome_zarr(Path(source_store) / pos, layout="fov", mode="r") as fov:
            level_names = list_numeric_levels(source_store, pos)
            if not level_names:
                return f"[SKIP] {pos}: No levels found"

            # Determine level 0 shape
            y0, x0 = int(fov["0"].shape[-2]), int(fov["0"].shape[-1])

            # Resolve well key and load CSV if available
            parts = [p for p in str(pos).split("/") if p]
            well_key = f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else str(pos)
            df = None
            csv_path = None

            if ds is not None:
                try:
                    csv_path = ds.append_well("linked_results", well_key)
                    exists_flag = Path(csv_path).exists()
                    if exists_flag:
                        try:
                            df = pd.read_csv(csv_path)
                        except Exception as _e_csv:
                            return f"[ERROR] {pos}: Failed to read CSV: {_e_csv}"
                    else:
                        return f"[SKIP] {pos}: CSV missing at {csv_path}"
                except Exception as _e_path:
                    return f"[ERROR] {pos}: Could not resolve CSV path: {_e_path}"

            if df is None:
                return f"[SKIP] {pos}: No CSV dataframe available"

            # Prepare coords and labels from CSV
            coords_yx_np: Optional[np.ndarray] = None
            labels_np: Optional[np.ndarray] = None

            # Support both old column names and new ones
            y_col, x_col = None, None
            if {"y_global_pheno", "x_global_pheno"}.issubset(df.columns):
                y_col, x_col = "y_global_pheno", "x_global_pheno"
            elif {"y_pheno", "x_pheno"}.issubset(df.columns):
                y_col, x_col = "y_pheno", "x_pheno"

            if y_col is None or x_col is None:
                return f"[ERROR] {pos}: Missing coordinate columns. Found: {list(df.columns)}"

            coords = df[[y_col, x_col]].copy().dropna()
            if len(coords) == 0:
                return f"[SKIP] {pos}: No valid coordinates after dropna"

            yy = np.clip(coords[y_col].to_numpy(dtype=np.int64), 0, y0 - 1)
            xx = np.clip(coords[x_col].to_numpy(dtype=np.int64), 0, x0 - 1)
            coords_yx_np = np.stack([yy.astype(np.float32), xx.astype(np.float32)], axis=1)

            # Build labels from gene names (supports custom gene_name_output_column)
            gene_col_candidates, secondary_col = _resolve_gene_columns(experiment)
            gene = None
            gene_col_used = None
            for gene_col in gene_col_candidates:
                if gene_col in df.columns:
                    gene = df[gene_col]
                    gene_col_used = gene_col
                    break
            if secondary_col and secondary_col in df.columns:
                sec_msg = f", combined with secondary '{secondary_col}' -> labels like 'primary | secondary'"
            elif secondary_col:
                sec_msg = f" (secondary '{secondary_col}' configured but not found in CSV columns: {list(df.columns)})"
            else:
                sec_msg = ""
            print(f"[ISS] {pos}: using gene column '{gene_col_used}'{sec_msg}")

            barcode = None
            for bc_col in ["barcode", "Barcode"]:
                if bc_col in df.columns:
                    barcode = df[bc_col]
                    break

            if gene is None:
                gene = pd.Series([None] * len(df))
            if barcode is None:
                barcode = pd.Series(["UNKNOWN"] * len(df))

            primary_labels_s = gene.astype("string")
            primary_labels_s = primary_labels_s.mask(
                primary_labels_s.isna() | (primary_labels_s.str.len() == 0),
                other=("NTC_" + barcode.astype("string").fillna("UNKNOWN")),
            )

            # Combine with secondary column if available (e.g. "MB_A | KRAS")
            if secondary_col and secondary_col in df.columns:
                secondary_s = df[secondary_col].astype("string").fillna("")
                labels_s = primary_labels_s + " | " + secondary_s
                # Clean up trailing " | " for rows where secondary is empty
                labels_s = labels_s.str.replace(r"\s*\|\s*$", "", regex=True)
            else:
                labels_s = primary_labels_s

            labels_s = labels_s.loc[coords.index].astype(str)
            labels_np = labels_s.to_numpy(dtype="U64")
            # Keep primary-only labels for color generation (combined labels
            # like "MB_A | KRAS" inherit color from primary key "MB_A")
            primary_labels_np = primary_labels_s.loc[coords.index].astype(str).to_numpy(dtype="U64")

            # Generate gene color palette from PRIMARY labels (not combined)
            # so "MB_A" gets a color, then map combined labels like "MB_A | KRAS"
            # to the same color as their primary key
            if gene_colors is None:
                pos_gene_colors = _get_gene_color_palette(primary_labels_np.tolist(), colormap=colormap)
            else:
                pos_gene_colors = dict(gene_colors)  # copy so we don't mutate caller's dict

            # Map combined labels to their primary label's color
            for primary, combined in zip(primary_labels_np, labels_np):
                if combined not in pos_gene_colors and primary in pos_gene_colors:
                    pos_gene_colors[combined] = pos_gene_colors[primary]

            # Ensure directory and attrs
            # v2 zarr: write directly under position (pos/iss_gene_image)
            # v3 zarr: write under labels subfolder (pos/labels/iss_gene_image)
            pos_dir = source_store / pos
            if zarr_format == 3:
                iss_gene_dir = pos_dir / "labels" / "iss_gene_image"
            else:
                iss_gene_dir = pos_dir / "iss_gene_image"
            iss_gene_dir.mkdir(parents=True, exist_ok=True)

            # Store metadata
            if zarr_format == 3:
                # For zarr v3, write compact custom_metadata to zarr.json
                from cyclops_process.convert.v3_metadata import OVERLAY_METADATA

                zarr_json_path = iss_gene_dir / "zarr.json"
                zarr_meta = {
                    "zarr_format": 3,
                    "node_type": "group",
                    "attributes": {
                        "custom_metadata": {
                            "label_name": "iss_gene_image",
                            **OVERLAY_METADATA["iss_gene_image"],
                        }
                    }
                }
                with open(zarr_json_path, "w") as f:
                    json.dump(zarr_meta, f, indent=2)
            else:
                # For zarr v2, write compact metadata to .zattrs (don't store full gene_colors library)
                from cyclops_process.convert.v3_metadata import OVERLAY_METADATA

                # Ensure .zgroup metadata exists for zarr v2 compatibility
                zgroup_path = iss_gene_dir / ".zgroup"
                if not zgroup_path.exists():
                    zgroup_path.write_text('{"zarr_format": 2}')

                with open(iss_gene_dir / ".zattrs", "w") as f:
                    json.dump(OVERLAY_METADATA["iss_gene_image"], f)

            # Determine levels to build
            lvls = sorted([int(l) for l in level_names])
            if build_levels is not None:
                lvls = [int(l) for l in build_levels if str(l) in level_names]

            # Render and write each level
            for lvl in lvls:
                if not overwrite:
                    continue

                yl, xl = int(fov[str(lvl)].shape[-2]), int(fov[str(lvl)].shape[-1])

                scale_y = y0 / max(1, yl)
                scale_x = x0 / max(1, xl)
                coords_lvl = coords_yx_np.copy()
                coords_lvl[:, 0] = coords_lvl[:, 0] / scale_y
                coords_lvl[:, 1] = coords_lvl[:, 1] / scale_x
                coords_lvl[:, 0] = np.clip(coords_lvl[:, 0], 0, yl - 1)
                coords_lvl[:, 1] = np.clip(coords_lvl[:, 1], 0, xl - 1)

                render_text = lvl in text_levels
                # Scale font and radius - use larger minimums for visibility at all levels
                # At higher levels (more zoomed out), keep dots and text visible
                scale_factor = max(scale_y, scale_x)
                lvl_radius = max(3, int(point_radius_px / scale_factor) + int(lvl))  # Increase radius at higher levels
                lvl_font_size = max(16, int(font_size / scale_factor))

                print(f"[ISS gene] Rendering level {lvl} for {pos} ({len(coords_lvl)} points, "
                      f"text={render_text}, radius={lvl_radius}px, shape=({yl}, {xl}))")

                # Use correct path based on zarr version
                iss_gene_component = str(iss_gene_dir.relative_to(source_store) / str(lvl))

                # Always use tiled rendering for consistent sharding via create_zarr_array
                _render_iss_tiled_to_zarr(
                    coords_yx=coords_lvl,
                    labels=labels_np,
                    gene_colors=pos_gene_colors,
                    canvas_shape=(yl, xl),
                    output_store=str(source_store),
                    output_component=iss_gene_component,
                    dot_radius=lvl_radius,
                    font_size=lvl_font_size,
                    render_text=render_text,
                    tile_size=4096,
                    zarr_format=zarr_format,
                )

        return f"[OK] {pos}: iss_gene_image built ({len(coords_yx_np)} points, {len(lvls)} levels)"

    except Exception as e:
        import traceback
        return f"[ERROR] {pos}: {e}\n{traceback.format_exc()}"


def _build_iss_guide_for_single_position(
    source_store: Path,
    pos: str,
    experiment: Optional[str],
    point_radius_px: int,
    font_size: int,
    text_levels: Sequence[int],
    gene_colors: Dict[str, Tuple[int, int, int]],
    colormap: str,
    build_levels: Optional[Sequence[int]],
    overwrite: bool,
    zarr_format: int,
    guide_text_offset: int,
    failed_rounds_by_well: Optional[Dict[str, list]] = None,
) -> str:
    """
    Process a single position for ISS guide overlay building.

    Renders guide sequences (first 10 bases of sgRNA) as colored text labels
    with dots, positioned below gene names. Output is written to
    iss_guide_image/<level> as RGBA images.

    Parameters
    ----------
    source_store : Path
        Path to zarr store
    pos : str
        Position path (e.g., "A/1/0")
    experiment : Optional[str]
        Experiment name for resolving linked_results CSV
    point_radius_px : int
        Radius of dots in pixels
    font_size : int
        Font size for text labels
    text_levels : Sequence[int]
        Pyramid levels that include text labels
    gene_colors : Dict[str, Tuple[int, int, int]]
        Pre-computed gene color mapping (used to derive guide colors)
    colormap : str
        Matplotlib colormap for auto-generating colors
    build_levels : Optional[Sequence[int]]
        Specific levels to build
    overwrite : bool
        Whether to overwrite existing overlay data
    zarr_format : int
        Zarr format version (2 or 3)
    guide_text_offset : int
        Vertical offset for guide text (to position below gene name)
    failed_rounds_by_well : Optional[Dict[str, list]]
        Mapping of well keys to lists of failed round indices (0-9).
        Failed rounds will be rendered in gray while valid bases keep their color.

    Returns
    -------
    str
        Status message describing result
    """
    from iohub import open_ome_zarr

    ds: Optional[OpsDataset] = None
    if experiment is not None:
        try:
            ds = OpsDataset(experiment)
        except Exception:
            ds = None

    try:
        with open_ome_zarr(Path(source_store) / pos, layout="fov", mode="r") as fov:
            level_names = list_numeric_levels(source_store, pos)
            if not level_names:
                return f"[SKIP] {pos}: No levels found"

            # Determine level 0 shape
            y0, x0 = int(fov["0"].shape[-2]), int(fov["0"].shape[-1])

            # Resolve well key and load CSV if available
            parts = [p for p in str(pos).split("/") if p]
            well_key = f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else str(pos)

            # Get failed rounds for this well
            well_key_slash = f"{parts[0]}/{parts[1]}/0" if len(parts) >= 2 else str(pos)
            failed_rounds: Optional[list] = None
            if failed_rounds_by_well:
                failed_rounds = failed_rounds_by_well.get(well_key_slash, None)
            df = None
            csv_path = None

            if ds is not None:
                try:
                    csv_path = ds.append_well("linked_results", well_key)
                    exists_flag = Path(csv_path).exists()
                    if exists_flag:
                        try:
                            df = pd.read_csv(csv_path)
                        except Exception as _e_csv:
                            return f"[ERROR] {pos}: Failed to read CSV: {_e_csv}"
                    else:
                        return f"[SKIP] {pos}: CSV missing at {csv_path}"
                except Exception as _e_path:
                    return f"[ERROR] {pos}: Could not resolve CSV path: {_e_path}"

            if df is None:
                return f"[SKIP] {pos}: No CSV dataframe available"

            # Prepare coords from CSV
            coords_yx_np: Optional[np.ndarray] = None

            # Support both old column names and new ones
            y_col, x_col = None, None
            if {"y_global_pheno", "x_global_pheno"}.issubset(df.columns):
                y_col, x_col = "y_global_pheno", "x_global_pheno"
            elif {"y_pheno", "x_pheno"}.issubset(df.columns):
                y_col, x_col = "y_pheno", "x_pheno"

            if y_col is None or x_col is None:
                return f"[ERROR] {pos}: Missing coordinate columns. Found: {list(df.columns)}"

            coords = df[[y_col, x_col]].copy().dropna()
            if len(coords) == 0:
                return f"[SKIP] {pos}: No valid coordinates after dropna"

            yy = np.clip(coords[y_col].to_numpy(dtype=np.int64), 0, y0 - 1)
            xx = np.clip(coords[x_col].to_numpy(dtype=np.int64), 0, x0 - 1)
            coords_yx_np = np.stack([yy.astype(np.float32), xx.astype(np.float32)], axis=1)

            # Extract guide sequences (first 10 bases of sgRNA column)
            sgRNA = None
            for sgRNA_col in ["sgRNA", "sgrna", "SGRNA", "guide", "Guide"]:
                if sgRNA_col in df.columns:
                    sgRNA = df[sgRNA_col]
                    break

            if sgRNA is None:
                return f"[SKIP] {pos}: No sgRNA column found. Available: {list(df.columns)}"

            # Extract first 10 bases of each sgRNA as the guide label
            guide_labels_s = sgRNA.astype("string").str[:10].fillna("UNKNOWN")
            guide_labels_s = guide_labels_s.loc[coords.index].astype(str)
            guide_labels_np = guide_labels_s.to_numpy(dtype="U10")

            # Get gene names for mapping guides to gene colors (supports custom column)
            gene_col_candidates, _ = _resolve_gene_columns(experiment)
            gene = None
            for gene_col in gene_col_candidates:
                if gene_col in df.columns:
                    gene = df[gene_col]
                    break

            if gene is None:
                gene = pd.Series(["UNKNOWN"] * len(df))

            # Build guide -> gene mapping
            guide_to_gene: Dict[str, str] = {}
            gene_aligned = gene.loc[coords.index].astype(str).to_numpy()
            for i, guide in enumerate(guide_labels_np):
                if guide not in guide_to_gene:
                    guide_to_gene[guide] = gene_aligned[i]

            # Generate guide colors as variants of gene colors
            guide_colors = _get_guide_color_variants(gene_colors, guide_to_gene)

            # Ensure directory and attrs for guide overlay
            # v2 zarr: write directly under position (pos/iss_guide_image)
            # v3 zarr: write under labels subfolder (pos/labels/iss_guide_image)
            pos_dir = source_store / pos
            if zarr_format == 3:
                iss_guide_dir = pos_dir / "labels" / "iss_guide_image"
            else:
                iss_guide_dir = pos_dir / "iss_guide_image"
            iss_guide_dir.mkdir(parents=True, exist_ok=True)

            # Store metadata
            if zarr_format == 3:
                # For zarr v3, write compact custom_metadata to zarr.json
                from cyclops_process.convert.v3_metadata import OVERLAY_METADATA

                zarr_json_path = iss_guide_dir / "zarr.json"
                zarr_meta = {
                    "zarr_format": 3,
                    "node_type": "group",
                    "attributes": {
                        "custom_metadata": {
                            "label_name": "iss_guide_image",
                            **OVERLAY_METADATA["iss_guide_image"],
                        }
                    }
                }
                with open(zarr_json_path, "w") as f:
                    json.dump(zarr_meta, f, indent=2)
            else:
                # For zarr v2, write compact metadata to .zattrs (don't store full guide_colors/guide_to_gene libraries)
                from cyclops_process.convert.v3_metadata import OVERLAY_METADATA

                # Ensure .zgroup metadata exists for zarr v2 compatibility
                zgroup_path = iss_guide_dir / ".zgroup"
                if not zgroup_path.exists():
                    zgroup_path.write_text('{"zarr_format": 2}')

                with open(iss_guide_dir / ".zattrs", "w") as f:
                    json.dump(OVERLAY_METADATA["iss_guide_image"], f)

            # Determine levels to build
            lvls = sorted([int(l) for l in level_names])
            if build_levels is not None:
                lvls = [int(l) for l in build_levels if str(l) in level_names]

            # Render and write each level
            for lvl in lvls:
                if not overwrite:
                    continue

                yl, xl = int(fov[str(lvl)].shape[-2]), int(fov[str(lvl)].shape[-1])

                scale_y = y0 / max(1, yl)
                scale_x = x0 / max(1, xl)
                coords_lvl = coords_yx_np.copy()
                coords_lvl[:, 0] = coords_lvl[:, 0] / scale_y
                coords_lvl[:, 1] = coords_lvl[:, 1] / scale_x
                coords_lvl[:, 0] = np.clip(coords_lvl[:, 0], 0, yl - 1)
                coords_lvl[:, 1] = np.clip(coords_lvl[:, 1], 0, xl - 1)

                render_text = lvl in text_levels
                # Scale font and radius - use larger minimums for visibility at all levels
                # At higher levels (more zoomed out), keep dots and text visible
                scale_factor = max(scale_y, scale_x)
                lvl_radius = max(3, int(point_radius_px / scale_factor) + int(lvl))  # Increase radius at higher levels
                lvl_font_size = max(16, int(font_size / scale_factor))
                lvl_y_offset = max(8, int(guide_text_offset / scale_factor))

                print(f"[ISS guide] Rendering level {lvl} for {pos} ({len(coords_lvl)} points, "
                      f"text={render_text}, radius={lvl_radius}px, y_offset={lvl_y_offset}, shape=({yl}, {xl}))")

                # Use correct path based on zarr version
                iss_guide_component = str(iss_guide_dir.relative_to(source_store) / str(lvl))

                # Always use tiled rendering for consistent sharding via create_zarr_array
                _render_iss_guide_tiled_to_zarr(
                    coords_yx=coords_lvl,
                    guide_labels=guide_labels_np,
                    guide_colors=guide_colors,
                    canvas_shape=(yl, xl),
                    output_store=str(source_store),
                    output_component=iss_guide_component,
                    dot_radius=lvl_radius,
                    font_size=lvl_font_size,
                    render_text=render_text,
                    tile_size=4096,
                    zarr_format=zarr_format,
                    y_offset=lvl_y_offset,
                    failed_rounds=failed_rounds,
                )

        return f"[OK] {pos}: iss_guide_image built ({len(coords_yx_np)} points, {len(lvls)} levels)"

    except Exception as e:
        import traceback
        return f"[ERROR] {pos}: {e}\n{traceback.format_exc()}"


# -----------------------------
# Small helpers (concise calls)
# -----------------------------


def _render_grid_tile(
    rects: List[Tuple[int, int, int, int]],
    tile_names: List[str],
    tile_coords: np.ndarray,
    tile_y_start: int,
    tile_x_start: int,
    tile_h: int,
    tile_w: int,
    line_width: int = 2,
    font_size: int = 24,
    render_text: bool = True,
) -> np.ndarray:
    """
    Render a single tile of the grid overlay.

    Parameters
    ----------
    rects : List[Tuple[int, int, int, int]]
        List of (top, left, height, width) rectangles for grid tiles (global coordinates)
    tile_names : List[str]
        List of tile name strings (e.g., "A/1/000001")
    tile_coords : np.ndarray
        Shape (N, 2) array of (y, x) tile center coordinates (global coordinates)
    tile_y_start, tile_x_start : int
        Top-left corner of tile in global coordinates
    tile_h, tile_w : int
        Height and width of this tile
    line_width : int
        Width of grid lines in pixels
    font_size : int
        Font size for tile name labels
    render_text : bool
        Whether to render tile name text

    Returns
    -------
    np.ndarray
        RGBA uint8 array of shape (tile_h, tile_w, 4)
    """
    from PIL import Image, ImageDraw, ImageFont

    # Create transparent RGBA canvas for this tile
    img = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Bright blue color for grid lines (R=0, G=191, B=255)
    grid_color = (0, 191, 255, 255)

    # Tile boundaries with padding for lines/text that may overlap
    text_padding = font_size * 10 if render_text else 0
    y_min = tile_y_start - line_width - 1
    y_max = tile_y_start + tile_h + line_width + 1
    x_min = tile_x_start - line_width - text_padding - 1
    x_max = tile_x_start + tile_w + line_width + 1

    # Draw rectangles that intersect with this tile
    for top, left, height, width in rects:
        right = left + width
        bottom = top + height

        # Check if this rectangle intersects the tile (with padding)
        if not (bottom < y_min or top > y_max or right < x_min or left > x_max):
            # Convert to tile-local coordinates
            local_left = left - tile_x_start
            local_top = top - tile_y_start
            local_right = right - tile_x_start
            local_bottom = bottom - tile_y_start

            # Draw rectangle outline (PIL will clip to canvas)
            draw.rectangle(
                [(local_left, local_top), (local_right, local_bottom)],
                outline=grid_color,
                width=line_width,
            )

    # Render tile names as white text if requested
    if render_text and len(tile_names) > 0:
        # Load font
        font = None
        try:
            for font_name in [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/TTF/DejaVuSans.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "arial.ttf",
            ]:
                try:
                    font = ImageFont.truetype(font_name, font_size)
                    break
                except (OSError, IOError):
                    continue
            if font is None:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        # Draw tile names that fall within this tile
        for tile_name, (y, x) in zip(tile_names, tile_coords):
            # Check if text center is near this tile
            if y_min <= y < y_max and x_min <= x < x_max:
                # Convert to tile-local coordinates
                local_x = x - tile_x_start
                local_y = y - tile_y_start
                draw.text(
                    (local_x, local_y),
                    tile_name,
                    fill=(255, 255, 255, 255),
                    font=font,
                    anchor="mm",  # Middle-middle anchor
                )

    # Convert to numpy array
    rgba = np.array(img)

    return rgba


def _process_single_grid_tile(tile_spec: dict) -> str:
    """
    Process a single grid tile work unit. Designed for parallel execution.

    Parameters
    ----------
    tile_spec : dict
        Dictionary containing all info needed to render one tile:
        - output_store: Path to zarr store
        - output_component: Component path within zarr store
        - rects: List of grid rectangles
        - tile_names: List of tile name strings
        - tile_coords: Tile center coordinates
        - y_start, x_start, tile_h, tile_w: Tile bounds
        - line_width, font_size, render_text: Rendering params
        - pos, lvl: For logging

    Returns
    -------
    str
        Status message for this tile
    """
    try:
        # Open zarr array for writing
        # Use zarr.open_array with explicit zarr_format for v3 compatibility
        array_path = f"{tile_spec['output_store']}/{tile_spec['output_component']}"
        try:
            # Try zarr v3 first
            z = zarr.open_array(array_path, mode="r+", zarr_format=3)
        except Exception:
            # Fallback to auto-detection
            z = zarr.open_array(array_path, mode="r+")

        # Render tile
        tile_data = _render_grid_tile(
            rects=tile_spec['rects'],
            tile_names=tile_spec['tile_names'],
            tile_coords=tile_spec['tile_coords'],
            tile_y_start=tile_spec['y_start'],
            tile_x_start=tile_spec['x_start'],
            tile_h=tile_spec['tile_h'],
            tile_w=tile_spec['tile_w'],
            line_width=tile_spec['line_width'],
            font_size=tile_spec['font_size'],
            render_text=tile_spec['render_text'],
        )

        # Write to zarr (even if all zeros - zarr handles this efficiently)
        z[tile_spec['y_start']:tile_spec['y_start'] + tile_spec['tile_h'],
          tile_spec['x_start']:tile_spec['x_start'] + tile_spec['tile_w'], :] = tile_data

        return f"[OK] {tile_spec['pos']}/lvl{tile_spec['lvl']}/tile({tile_spec['y_start']},{tile_spec['x_start']})"
    except Exception as e:
        return f"[ERROR] {tile_spec['pos']}/lvl{tile_spec['lvl']}/tile({tile_spec['y_start']},{tile_spec['x_start']}): {e}"


def build_grid_overlay_in_place(
    source_store: str | Path,
    positions: Optional[Sequence[str]] = None,
    tile_size_yx: Optional[Sequence[int]] = None,
    origin_yx: Optional[Sequence[int]] = None,
    line_width_px: int = 2,
    font_size: int = 120,
    stitch_config_path: Optional[str | Path] = None,
    dataset: Optional[OpsDataset] = None,
    zarr_format: Optional[int] = None,
) -> Path:
    """
    Build (or rebuild) a lightweight grid overlay inside an existing OME-Zarr store.

    Creates a unified RGBA image overlay with bright blue grid lines and tile IDs as text labels.
    - v2 format: Writes to pos/grid_overlay/<level>
    - v3 format: Writes to pos/labels/grid_overlay/<level>
    - Text labels are rendered on levels 0, 1, 2 only (not 3, 4)
    - Uses existing pyramid levels and shapes; does not recompute the image pyramid

    Parameters
    ----------
    source_store : str | Path
        Path to the OME-Zarr store
    positions : Optional[Sequence[str]]
        Specific positions to process (default: all)
    tile_size_yx : Optional[Sequence[int]]
        Tile size in pixels [height, width] (default: [2048, 2048])
    origin_yx : Optional[Sequence[int]]
        Origin offset in pixels [y, x] (default: [0, 0])
    line_width_px : int
        Line width for grid boundaries (default: 2)
    font_size : int
        Font size for tile ID labels (default: 24)
    stitch_config_path : Optional[str | Path]
        Path to stitch config YAML with tile translations
    dataset : Optional[OpsDataset]
        Dataset object for resolving metadata
    zarr_format : Optional[int]
        Zarr format version (2 or 3). If None, auto-detects from store
    """

    def _get_mode_from_store_path(
        store_path: Path, dataset: OpsDataset
    ) -> Optional[str]:
        if dataset is None:
            return None
        for mode, path in dataset.store_paths.items():
            if str(store_path) == str(path):
                if "iss" in mode:
                    return "iss"
                elif "lc_5x" in mode:
                    return "track"
                elif "lc_20x" in mode or "pheno" in mode:
                    return "pheno"
        return None

    from cyclops_process.convert.v3_metadata import OVERLAY_METADATA

    source_store = Path(source_store)

    # Detect zarr format for compatibility with both v2 and v3 stores (if not explicitly provided)
    if zarr_format is None:
        zarr_format = detect_zarr_format(source_store)

    origin_yx = origin_yx or (0, 0)
    if tile_size_yx is not None and len(tile_size_yx) != 2:
        raise ValueError("tile_size_yx must be a sequence of length 2 [ty, tx]")

    mode = _get_mode_from_store_path(source_store, dataset) if dataset else None

    # Collect all grid tiles across all positions for parallel processing
    all_grid_tiles = []

    with open_ome_zarr(source_store, mode="r+") as store:
        # Convert to list to allow multiple iterations (for loop + resharding)
        pos_paths = list(positions) if positions else list(_iter_position_paths(source_store))

        # Load stitch translations once (if provided)
        translations = (
            _load_total_translations(Path(stitch_config_path))
            if stitch_config_path
            else {}
        )
        if translations:
            sample_trans_keys = list(translations.keys())[:5]
            print(
                f"Loaded {len(translations)} translations from {stitch_config_path}. Sample keys: {sample_trans_keys}"
            )
        else:
            print(f"No translations loaded (stitch_config_path={stitch_config_path})")

        for pos in tqdm(pos_paths, desc="Grid overlay", leave=False):
            pos_dir = source_store / pos
            fov = store[pos]
            level_names = list_numeric_levels(source_store, pos)
            if not level_names:
                if not (pos_dir / "0").exists():
                    vprintf("No levels found under %s; skipping grid overlay", pos)
                    continue
                level_names = ["0"]

            # Determine level 0 shape to derive tile count and scale
            lvl0 = str(sorted(level_names, key=lambda s: int(s))[0])
            base_arr = fov[lvl0]
            if base_arr.ndim < 2:
                vprintf("Base level has <2 dims for %s; skipping", pos)
                continue
            y0, x0 = get_level0_shape(source_store, pos)

            # Determine tile size and tile origins
            if tile_size_yx is not None:
                ty, tx = int(tile_size_yx[0]), int(tile_size_yx[1])
            else:
                ty, tx = 2048, 2048

            # If stitch translations available, filter for current pos and compute rects
            rects_level0: list[Tuple[int, int, int, int]] = []
            oy, ox = (
                (int(origin_yx[0]), int(origin_yx[1]))
                if origin_yx is not None
                else (0, 0)
            )
            if mode in ["iss", "track"]:
                oy, ox = 0, 0

            # Build rectangles and tile info together in one pass to maintain tile names
            tile_info: list[dict] = []
            if translations:
                pos_prefixes = _candidate_pos_prefixes(str(pos))
                tile_id = 1
                for k, v in translations.items():
                    if isinstance(k, str) and any(
                        k.startswith(pp) for pp in pos_prefixes
                    ):
                        y_s, x_s = int(v[0]), int(v[1])
                        rects_level0.append((y_s, x_s, ty, tx))
                        # Construct proper tile name from the translation key
                        parts_k = [p for p in k.split("/") if p]
                        tile_token = parts_k[-1] if parts_k else ""
                        well_prefix = (
                            "/".join([str(parts_k[0]), str(parts_k[1])])
                            if len(parts_k) >= 3
                            else str(pos)
                        )
                        tile_name = (
                            f"{well_prefix}/{tile_token}" if tile_token else well_prefix
                        )
                        tile_info.append(
                            {
                                "id": tile_id,
                                "y": float(y_s),
                                "x": float(x_s),
                                "name": tile_name,
                            }
                        )
                        tile_id += 1

                if not rects_level0:
                    # Debug: print why no matches were found
                    sample_keys = list(translations.keys())[:3]
                    print(
                        f"Warning: No translation matches for pos '{pos}' (prefixes: {pos_prefixes}). Sample translation keys: {sample_keys}"
                    )

            # Skip position if no rectangles were found
            if not rects_level0:
                print(
                    f"Skipping grid overlay for position {pos} (no translation matches)"
                )
                continue

            # Extract tile names and coordinates for text rendering
            tile_names = [t["name"] for t in tile_info]
            tile_coords_level0 = np.array(
                [[t["y"] + ty / 2, t["x"] + tx / 2] for t in tile_info],
                dtype=np.float32,
            )

            # Determine grid overlay path based on zarr format
            if zarr_format == 3:
                # v3: pos/labels/grid_overlay
                grid_dir = pos_dir / "labels" / "grid_overlay"
            else:
                # v2: pos/grid_overlay
                grid_dir = pos_dir / "grid_overlay"

            grid_dir.mkdir(parents=True, exist_ok=True)

            # Ensure .zgroup metadata exists for zarr v2 compatibility
            if zarr_format == 2:
                zgroup_path = grid_dir / ".zgroup"
                if not zgroup_path.exists():
                    zgroup_path.write_text('{"zarr_format": 2}')

            # Write metadata for grid overlay group
            grid_metadata = OVERLAY_METADATA["grid_overlay"].copy()
            grid_metadata.update(
                {
                    "line_width_px": int(line_width_px),
                    "tile_size_px": [int(ty), int(tx)],
                    "origin_px": [int(oy), int(ox)],
                }
            )

            if zarr_format == 3:
                # For zarr v3, write metadata to zarr.json with attributes
                zarr_json_path = grid_dir / "zarr.json"
                zarr_meta = {
                    "zarr_format": 3,
                    "node_type": "group",
                    "attributes": {
                        "custom_metadata": {
                            "label_name": "grid_overlay",
                            **grid_metadata,
                        }
                    }
                }
                with open(zarr_json_path, "w") as f:
                    json.dump(zarr_meta, f, indent=2)
            else:
                # For zarr v2, write metadata to .zattrs
                write_component_attrs(grid_dir, grid_metadata)

            # Collect tile specs for all levels (this position)
            for lvl_idx, lvl in enumerate(level_names):
                lvl_arr = fov[str(lvl)]
                yl, xl = int(lvl_arr.shape[-2]), int(lvl_arr.shape[-1])
                # Derive scale vs level 0; avoid assuming a fixed factor
                scale_y = y0 / max(1, yl)
                scale_x = x0 / max(1, xl)

                # Scale rectangles
                rects_lvl = []
                for yy, xx, hh, ww in rects_level0:
                    yy_l = int(np.floor(yy / scale_y))
                    xx_l = int(np.floor(xx / scale_x))
                    hh_l = max(1, int(np.ceil(hh / scale_y)))
                    ww_l = max(1, int(np.ceil(ww / scale_x)))
                    rects_lvl.append((yy_l, xx_l, hh_l, ww_l))

                # Scale tile coordinates for this level
                tile_coords_lvl = tile_coords_level0 / np.array(
                    [[scale_y, scale_x]], dtype=np.float32
                )

                # Scale font size for higher pyramid levels (smaller images)
                # Use larger minimum font size to keep text readable at all levels
                lvl_font_size = max(24, int(font_size / max(scale_y, scale_x)))

                # Render text on levels 0, 1, 2, 3 (not 4 or 5)
                render_text = int(lvl) in [0, 1, 2, 3]

                # Render tiles into in-memory canvas (cell_seg pattern), then
                # write canvas to sharded zarr in one shot — no reshard needed.
                from concurrent.futures import ThreadPoolExecutor
                grid_component = str(grid_dir.relative_to(source_store) / str(lvl))
                canvas = np.zeros((yl, xl, 4), dtype=np.uint8)
                raw_tile_specs = _generate_tile_specs((yl, xl), tile_size=4096)
                def _render_into(spec):
                    y_start, x_start, tile_h, tile_w = spec
                    tile_data = _render_grid_tile(
                        rects=rects_lvl, tile_names=tile_names,
                        tile_coords=tile_coords_lvl,
                        tile_y_start=y_start, tile_x_start=x_start,
                        tile_h=tile_h, tile_w=tile_w,
                        line_width=int(line_width_px), font_size=lvl_font_size,
                        render_text=render_text,
                    )
                    canvas[y_start:y_start + tile_h, x_start:x_start + tile_w, :] = tile_data
                n_workers = min(16, len(raw_tile_specs))
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    list(pool.map(_render_into, raw_tile_specs))
                # Write via parallel shard-row strips (cell_seg pattern).
                chunk_shape = (min(1024, yl), min(1024, xl), 4)
                create_zarr_array(
                    path=f"{source_store}/{grid_component}",
                    shape=(yl, xl, 4),
                    chunks=chunk_shape,
                    dtype=np.uint8,
                    zarr_format=zarr_format,
                    fill_value=0,
                    overwrite=True,
                    shards_ratio=(32, 32, 1),
                )
                shard_y = chunk_shape[0] * 32
                n_strips = max(1, (yl + shard_y - 1) // shard_y)
                def _write_strip(s):
                    a = zarr.open(f"{source_store}/{grid_component}", mode="r+")
                    y0 = s * shard_y
                    y1 = min(yl, (s + 1) * shard_y)
                    a[y0:y1, :, :] = canvas[y0:y1, :, :]
                with ThreadPoolExecutor(max_workers=n_strips) as pool:
                    list(pool.map(_write_strip, range(n_strips)))
                del canvas

    return source_store




def build_iss_overlay_in_place(
    source_store: str | Path,
    experiment: Optional[str] = None,
    positions: Optional[Sequence[str]] = None,
    point_radius_px: int = 5,
    font_size: int = 36,
    text_levels: Sequence[int] = (0, 1),
    gene_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    colormap: str = "hsv",
    build_levels: Optional[Sequence[int]] = None,
    guide_text_offset: int = 40,
    zarr_format: Optional[int] = None,
    kinds: Sequence[str] = ("gene", "guide"),
) -> Path:
    """
    Build ISS overlays inside the OME-Zarr store.

    Creates two rendered image layers:
    - iss_gene_image: Gene names as colored text labels with dots
    - iss_guide_image: Guide sequences (first 10 bases of sgRNA) below gene names,
      with dropout/failed rounds rendered in white

    Parameters
    ----------
    source_store : str | Path
        Path to the OME-Zarr store
    experiment : Optional[str]
        Experiment name for resolving linked_results CSV and failed rounds config
    positions : Optional[Sequence[str]]
        Specific positions to process (default: all)
    point_radius_px : int
        Radius of dots in pixels (default: 3)
    font_size : int
        Font size for text labels (default: 24)
    text_levels : Sequence[int]
        Pyramid levels that include text labels (default: (0, 1))
    gene_colors : Optional[Dict]
        Pre-defined gene→RGB color mapping (default: auto-generate)
    colormap : str
        Matplotlib colormap for auto-generating colors (default: "hsv")
    build_levels : Optional[Sequence[int]]
        Specific levels to build (default: all available)
    guide_text_offset : int
        Vertical offset to position guide text below gene name (default: 28)
    zarr_format : Optional[int]
        Zarr format version (2 or 3). If None, auto-detects from store (default: None)
    """
    import yaml

    source_store = Path(source_store)

    # Detect zarr format for compatibility with both v2 and v3 stores (if not explicitly provided)
    if zarr_format is None:
        zarr_format = detect_zarr_format(source_store)

    # Load failed rounds configuration for guide overlay
    failed_rounds_by_well: Optional[Dict[str, list]] = None
    if experiment:
        try:
            config_path = Path(f"{BASE_PATH}/configs/ops_failed_rounds.yaml")
            if config_path.exists():
                with open(config_path, "r") as f:
                    all_failed_rounds = yaml.safe_load(f) or {}
                exp_config = all_failed_rounds.get(experiment, {})
                failed_rounds_by_well = exp_config.get("failed_rounds_by_well", None)
                if failed_rounds_by_well:
                    print(f"[ISS] Loaded failed_rounds config for {experiment}")
        except Exception as e:
            print(f"[ISS] Warning: Could not load failed_rounds config: {e}")

    # Check if ISS overlay components already exist
    overwrite = True

    try:
        pos_sample = positions[0] if positions else None
        if pos_sample is None:
            pos_paths_sample = _iter_position_paths(source_store)
            pos_sample = pos_paths_sample[0] if pos_paths_sample else None

        if pos_sample is not None:
            # Check for existing overlays (either old names or new names)
            if zarr_format == 3:
                iss_gene_path = source_store / pos_sample / "labels" / "iss_gene_image"
                iss_guide_path = source_store / pos_sample / "labels" / "iss_guide_image"
            else:
                iss_gene_path = source_store / pos_sample / "iss_gene_image"
                iss_guide_path = source_store / pos_sample / "iss_guide_image"

            if iss_gene_path.exists() or iss_guide_path.exists():
                print(f"\nISS overlay exists")
                choice = prompt_overwrite_resume_skip(iss_gene_path, default="O")
                if choice == "skip":
                    print(f"Skipping ISS overlay build (user choice)")
                    return source_store
                overwrite = choice == "overwrite"

    except Exception as e:
        print(f"Warning: could not check ISS overlay status: {e}")
        overwrite = True

    # Get list of positions to process (must be a list for multiple iterations)
    pos_paths = list(positions) if positions else list(_iter_position_paths(source_store))

    # Auto-generate gene colors if not provided
    if gene_colors is None:
        gene_colors = {}
        if experiment:
            try:
                ds = OpsDataset(experiment)
                gene_col_candidates, secondary_col = _resolve_gene_columns(experiment)
                print(f"[ISS] Gene column candidates: {gene_col_candidates}"
                      + (f", secondary: {secondary_col}" if secondary_col else ""))
                for pos in pos_paths[:3]:
                    parts = [p for p in str(pos).split("/") if p]
                    well_key = f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else str(pos)
                    csv_path = ds.append_well("linked_results", well_key)
                    if Path(csv_path).exists():
                        df = pd.read_csv(csv_path)
                        for col in gene_col_candidates:
                            if col in df.columns:
                                genes = df[col].dropna().unique()
                                gene_colors = _get_gene_color_palette(
                                    genes, colormap=colormap, existing_colors=gene_colors
                                )
                                break
            except Exception as e:
                print(f"[ISS] Warning: Could not pre-compute gene colors: {e}")

    # Determine optimal number of workers
    n_workers = get_optimal_workers(
        use_gpu=False,
        model_ram_gb=2.0,
        data_ram_gb=4.0,
        verbose=False,
    )

    print(f"[ISS] Building iss_gene_image and iss_guide_image for {len(pos_paths)} positions")

    # Phase 1: Collect all position data and tile specs (sequential - needs zarr init)
    print("\n[ISS] Phase 1: Preparing position data and initializing zarr arrays...")
    all_gene_tiles = []
    all_guide_tiles = []
    pos_errors = []
    pos_skipped = []

    for pos in tqdm(pos_paths, desc="Preparing positions", unit="pos"):
        gene_tiles, guide_tiles, status = _prepare_iss_position_data(
            source_store=source_store,
            pos=pos,
            experiment=experiment,
            point_radius_px=point_radius_px,
            font_size=font_size,
            text_levels=text_levels,
            gene_colors=gene_colors,
            colormap=colormap,
            build_levels=build_levels,
            overwrite=overwrite,
            zarr_format=zarr_format,
            tile_size=4096,
        )

        if gene_tiles is not None:
            # Set failed_rounds for guide tiles based on well
            parts = [p for p in str(pos).split("/") if p]
            well_key = f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else str(pos)
            failed_rounds = None
            if failed_rounds_by_well and well_key in failed_rounds_by_well:
                failed_rounds = failed_rounds_by_well[well_key]

            for tile in guide_tiles:
                tile['failed_rounds'] = failed_rounds

            all_gene_tiles.extend(gene_tiles)
            all_guide_tiles.extend(guide_tiles)
        elif status.startswith("[ERROR]"):
            pos_errors.append(status)
        else:
            pos_skipped.append(status)

    # Drop tile lists for kinds we're not building this run (gene/guide split
    # across two SLURM jobs in the fix pipeline).
    _kinds = set(kinds)
    if "gene" not in _kinds:
        all_gene_tiles = []
    if "guide" not in _kinds:
        all_guide_tiles = []

    print(f"  Prepared {len(pos_paths) - len(pos_errors) - len(pos_skipped)} positions"
          f" — kinds={sorted(_kinds)}")
    print(f"  Total tiles: {len(all_gene_tiles)} gene + {len(all_guide_tiles)} guide = {len(all_gene_tiles) + len(all_guide_tiles)} total")
    if pos_skipped:
        print(f"  Skipped: {len(pos_skipped)} positions")
    if pos_errors:
        print(f"  Errors during prep: {len(pos_errors)} positions")

    # Phase 2: Process all tiles in parallel using loky (multiprocessing).
    # Multiprocessing bypasses the GIL — PIL/Cython rendering of dots+text is
    # CPU-bound, so threads spend 90%+ blocked on GIL. The canvas-into-shards
    # variant we briefly used (134f82b) clocked 47 min wall at 3.6% CPU; this
    # multiprocessing path was the fast version.
    if all_gene_tiles or all_guide_tiles:
        n_workers = min(n_workers, len(all_gene_tiles) + len(all_guide_tiles))
        print(f"\n[ISS] Phase 2: Rendering {len(all_gene_tiles) + len(all_guide_tiles)} tiles with {n_workers} workers...")

        if all_gene_tiles:
            print(f"  Rendering {len(all_gene_tiles)} gene tiles...")
            gene_results = Parallel(n_jobs=n_workers, backend="loky")(
                delayed(_process_single_iss_tile)(tile)
                for tile in tqdm(all_gene_tiles, desc="Gene tiles", unit="tile")
            )
            gene_ok = sum(1 for r in gene_results if r.startswith("[OK]"))
            gene_err = sum(1 for r in gene_results if r.startswith("[ERROR]"))
        else:
            gene_results = []
            gene_ok, gene_err = 0, 0

        if all_guide_tiles:
            print(f"  Rendering {len(all_guide_tiles)} guide tiles...")
            guide_results = Parallel(n_jobs=n_workers, backend="loky")(
                delayed(_process_single_iss_guide_tile)(tile)
                for tile in tqdm(all_guide_tiles, desc="Guide tiles", unit="tile")
            )
            guide_ok = sum(1 for r in guide_results if r.startswith("[OK]"))
            guide_err = sum(1 for r in guide_results if r.startswith("[ERROR]"))
        else:
            guide_results = []
            guide_ok, guide_err = 0, 0
    else:
        gene_results, guide_results = [], []
        gene_ok, gene_err, guide_ok, guide_err = 0, 0, 0, 0

    # Phase 3: Reshard overlay arrays — only for the kinds we built this run.
    _overlay_names = []
    if "gene" in _kinds:
        _overlay_names.append("iss_gene_image")
    if "guide" in _kinds:
        _overlay_names.append("iss_guide_image")
    if _overlay_names:
        _reshard_overlay_arrays(
            source_store, pos_paths,
            overlay_names=_overlay_names,
            zarr_format=zarr_format,
            label="ISS",
        )

    # Print summary
    print(f"\n[ISS] Build complete:")
    print(f"  iss_gene_image:  {gene_ok}/{len(gene_results)} tiles OK" + (f", {gene_err} errors" if gene_err else ""))
    print(f"  iss_guide_image: {guide_ok}/{len(guide_results)} tiles OK" + (f", {guide_err} errors" if guide_err else ""))

    # Print errors if any
    if pos_errors:
        print(f"\n  Position prep errors ({len(pos_errors)}):")
        for err in pos_errors[:5]:  # Limit to first 5
            print(f"    {err}")
        if len(pos_errors) > 5:
            print(f"    ... and {len(pos_errors) - 5} more")

    if gene_err > 0:
        print(f"\n  Gene tile errors ({gene_err}):")
        for result in gene_results[:5]:
            if result.startswith("[ERROR]"):
                print(f"    {result}")

    if guide_err > 0:
        print(f"\n  Guide tile errors ({guide_err}):")
        for result in guide_results[:5]:
            if result.startswith("[ERROR]"):
                print(f"    {result}")

    return source_store


