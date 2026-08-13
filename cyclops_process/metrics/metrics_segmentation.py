#%%
"""
Segmentation metrics comparison: Cellpose baseline vs cpsam vs Ground Truth

This script evaluates segmentation quality by computing IoU-based metrics
(recall, precision) against hand-annotated ground truth labels.

Compares:
- Cellpose baseline (existing predictions from segment.py)
- cpsam (Cellpose-SAM model from cellpose v4+)

Default Parameters (must match segment.py pheno_cells config):
- diameter: 100
- flow_threshold: 0.7
- cellprob_threshold: 0.0

Usage:
    # Run full metrics analysis (in sam3 conda environment)
    python -m cyclops_process.metrics.metrics_segmentation

    # Compare on a specific tile from any experiment/position
    python -m cyclops_process.metrics.metrics_segmentation --compare-tile \\
        --experiment ops0033_20250429 \\
        --position A/1/025022 \\
        --diameter 100 \\
        --flow-threshold 0.7

    # Run interactively in Jupyter/VSCode cells

Regenerating Cellpose Baseline Predictions (update CELLPOSE_BASELINE paths after):
    # ops0033 - position A/1/025022
    python -m cyclops_process.metrics.processes.segment segmentation \\
        --experiment ops0033_20250429 --process pheno_cells \\
        --use-preprocess --positions A/1/025022

    # ops0036 - position A/1/010026
    python -m cyclops_process.metrics.processes.segment segmentation \\
        --experiment ops0036_20250505 --process pheno_cells \\
        --use-preprocess --positions A/1/010026

    # ops0063 - position A/2/040013
    python -m cyclops_process.metrics.processes.segment segmentation \\
        --experiment ops0063_20250731 --process pheno_cells \\
        --use-preprocess --positions A/2/040013

    # ops0086 - position A/3/042040
    python -m cyclops_process.metrics.processes.segment segmentation \\
        --experiment ops0086_20250922 --process pheno_cells \\
        --use-preprocess --positions A/3/042040
"""
import warnings
warnings.filterwarnings("ignore", message=".*depricated.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

import numpy as np
from pathlib import Path
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from scipy.optimize import linear_sum_assignment
from skimage.exposure import equalize_adapthist

# Try imports that may not be available in all environments
try:
    from iohub import open_ome_zarr
    IOHUB_AVAILABLE = True
except ImportError:
    IOHUB_AVAILABLE = False
    import zarr

try:
    from ultrack.core.segmentation.node import _fast_iou_with_bbox
    ULTRACK_AVAILABLE = True
except ImportError:
    ULTRACK_AVAILABLE = False

try:
    import torch
    import os
    from cellpose import models as cellpose_models
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False


# ==============================================================================
# IOU METRICS (from segmentation_metrics.py)
# ==============================================================================


def _compute_iou(bbox1, bbox2, mask1, mask2):
    """Compute IoU between two objects given their bounding boxes and masks."""
    # Get bounding box coordinates
    r1_min, c1_min, r1_max, c1_max = bbox1
    r2_min, c2_min, r2_max, c2_max = bbox2

    # Check for overlap
    if r1_max <= r2_min or r2_max <= r1_min or c1_max <= c2_min or c2_max <= c1_min:
        return 0.0

    # Compute intersection bounds
    r_min = max(r1_min, r2_min)
    r_max = min(r1_max, r2_max)
    c_min = max(c1_min, c2_min)
    c_max = min(c1_max, c2_max)

    # Extract overlapping regions from masks
    mask1_region = mask1[r_min - r1_min:r_max - r1_min, c_min - c1_min:c_max - c1_min]
    mask2_region = mask2[r_min - r2_min:r_max - r2_min, c_min - c2_min:c_max - c2_min]

    intersection = np.sum(mask1_region & mask2_region)
    union = np.sum(mask1) + np.sum(mask2) - intersection

    return intersection / union if union > 0 else 0.0


def multi_object_iou(input: np.ndarray, target: np.ndarray):
    """
    Compute IoU-based metrics between predicted and ground truth segmentations.

    Parameters
    ----------
    input : np.ndarray
        Predicted segmentation masks, shape (batch, H, W).
    target : np.ndarray
        Ground truth segmentation masks, shape (batch, H, W).

    Returns
    -------
    ious : np.ndarray
        IoU values for matched objects.
    metrics : dict
        Dictionary with recall, precision, false_pos, false_neg at various thresholds.
    """
    ious = []
    input_size = target_size = 0

    for b in range(input.shape[0]):
        inp_objs = list(regionprops(input[b]))
        tgt_objs = list(regionprops(target[b]))
        cost = np.zeros((len(inp_objs), len(tgt_objs)))

        for i, i_obj in enumerate(inp_objs):
            for j, j_obj in enumerate(tgt_objs):
                if ULTRACK_AVAILABLE:
                    cost[i, j] = _fast_iou_with_bbox(
                        i_obj.bbox, j_obj.bbox, i_obj.image, j_obj.image
                    )
                else:
                    cost[i, j] = _compute_iou(
                        i_obj.bbox, j_obj.bbox, i_obj.image, j_obj.image
                    )

        input_size += len(inp_objs)
        target_size += len(tgt_objs)
        rows, cols = linear_sum_assignment(-cost)
        if len(rows) > 0:
            ious.append(cost[rows, cols])

    ious = np.concatenate(ious) if ious else np.array([])
    thresholds = list(np.arange(0.5, 1.0, 0.01))
    metrics = {"recall": {}, "precision": {}, "false_pos": {}, "false_neg": {}}

    for t in thresholds + [1.0]:
        tp = np.sum(ious > t)
        fp = input_size - tp
        fn = target_size - tp
        metrics["recall"][t] = tp / target_size if target_size else 0
        metrics["precision"][t] = tp / input_size if input_size else 0
        metrics["false_pos"][t] = fp
        metrics["false_neg"][t] = fn

    return ious, {**metrics, "n_matches": ious.size, "total_true_cells": target_size, "total_pred_cells": input_size}


# ==============================================================================
# CELLPOSE INFERENCE (cpsam model)
# ==============================================================================


def load_cellpose_model(model_type: str = "cpsam"):
    """
    Load Cellpose model (cpsam by default).

    Note: In cellpose v4+, cpsam (Cellpose-SAM) is the default model and
    provides the best performance for most cell types.

    Parameters
    ----------
    model_type : str
        Model type (default: cpsam). Other options like cyto3 will also
        use cpsam internally in cellpose v4.
    """
    if not CELLPOSE_AVAILABLE:
        print("ERROR: Cellpose not available.")
        return None

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model: {model_type}...")
        model = cellpose_models.CellposeModel(
            gpu=True,
            model_type=model_type,
            device=device,
        )
        print(f"Loaded Cellpose model ({model_type}) on {device}")
        return model
    except Exception as e:
        # Handle CP3/CP4 incompatibility by removing cached model
        if "CP3 models are not compatible" in str(e):
            model_dir = os.path.expanduser("~/.cellpose/models")
            model_path = os.path.join(model_dir, model_type)
            if os.path.exists(model_path):
                print(f"  Removing incompatible CP3 cached model: {model_path}")
                os.remove(model_path)
                # Retry loading
                try:
                    model = cellpose_models.CellposeModel(
                        gpu=True,
                        model_type=model_type,
                        device=device,
                    )
                    print(f"Loaded Cellpose model ({model_type}) on {device} after cache clear")
                    return model
                except Exception as e2:
                    print(f"ERROR loading model after cache clear: {e2}")
                    return None
        print(f"ERROR loading model: {e}")
        return None


def preprocess_for_segmentation(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for segmentation (percentile normalization + CLAHE).

    Parameters
    ----------
    image : np.ndarray
        Raw image, can be any dtype.

    Returns
    -------
    np.ndarray
        Preprocessed image in [0, 1] range.
    """
    # Percentile normalization
    vmin, vmax = np.percentile(image, [1, 99.5])
    normalized = np.clip((image.astype(np.float32) - vmin) / (vmax - vmin + 1e-8), 0, 1)

    # CLAHE enhancement
    enhanced = equalize_adapthist(normalized, clip_limit=0.01)

    return enhanced.astype(np.float32)


def segment_with_cellpose(
    image: np.ndarray,
    model,
    diameter: float = 100,
    flow_threshold: float = 0.8,
    cellprob_threshold: float = 0.0,
) -> np.ndarray:
    """
    Run Cellpose segmentation on a preprocessed image.

    Parameters
    ----------
    image : np.ndarray
        Preprocessed image in [0, 1] range.
    model : CellposeModel
        Loaded Cellpose model.
    diameter : float
        Cell diameter parameter.
    flow_threshold : float
        Flow threshold parameter.
    cellprob_threshold : float
        Cell probability threshold parameter.

    Returns
    -------
    np.ndarray
        Segmentation mask with unique labels per cell.
    """
    result = model.eval(
        image,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    # Handle both v3 (4 returns) and v4 (3 returns)
    masks = result[0]
    return masks


def segment_with_gap_filling(
    image: np.ndarray,
    model,
    primary_diameter: float = 100,
    scale_factor: float = 0.25,
    flow_threshold: float = 0.7,
    cellprob_threshold: float = 0.0,
    min_gap_size: int = 200,
    padding: int = 250,
    max_overlap_fraction: float = 0.1,
) -> tuple[np.ndarray, dict]:
    """
    Two-pass segmentation: primary pass for most cells, secondary pass for large gaps.

    This approach efficiently handles heterogeneous cell sizes by:
    1. Running segmentation at primary_diameter (optimal for most cells)
    2. Finding large unsegmented regions (gaps) bigger than min_gap_size
    3. Downscaling gap regions so large cells appear ~100px diameter
    4. Running segmentation at primary_diameter on downscaled crops
    5. Upscaling masks back and merging new cells

    Parameters
    ----------
    image : np.ndarray
        Preprocessed image in [0, 1] range.
    model : CellposeModel
        Loaded Cellpose model.
    primary_diameter : float
        Diameter for segmentation (default: 100, optimal for most cells).
        Used for both passes (second pass uses downscaling instead of different diameter).
    scale_factor : float
        Scale factor for downscaling gap crops (default: 0.25 = 4x downscale).
        A 400px cell becomes 100px after 0.25x downscale.
    flow_threshold : float
        Flow threshold parameter.
    cellprob_threshold : float
        Cell probability threshold parameter.
    min_gap_size : int
        Minimum gap size in pixels to trigger second pass (default: 200).
        Gaps smaller than min_gap_size x min_gap_size are ignored.
    padding : int
        Padding around gap regions for context (default: 50).
    max_overlap_fraction : float
        Maximum allowed overlap with existing cells (default: 0.1 = 10%).
        New cells with more overlap are discarded.

    Returns
    -------
    masks : np.ndarray
        Combined segmentation mask with unique labels per cell.
    stats : dict
        Statistics about the two-pass process:
        - n_cells_pass1: cells from first pass
        - n_gaps_found: number of large gaps detected
        - n_cells_pass2: cells added from second pass
        - n_cells_total: total cells in final mask
        - gap_regions: list of (bbox, n_cells_added, scale_factor) for each gap
    """
    from skimage.measure import label as sk_label
    from skimage.transform import resize

    h, w = image.shape
    stats = {
        "n_cells_pass1": 0,
        "n_gaps_found": 0,
        "n_cells_pass2": 0,
        "n_cells_total": 0,
        "gap_regions": [],
        "scale_factor": scale_factor,
    }

    # === PASS 1: Primary segmentation ===
    masks = segment_with_cellpose(
        image, model,
        diameter=primary_diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )

    if masks is None:
        return None, stats

    stats["n_cells_pass1"] = int(masks.max())
    next_label = stats["n_cells_pass1"] + 1

    # === Find large unsegmented gaps ===
    unsegmented = masks == 0
    gap_labels = sk_label(unsegmented)
    gap_props = regionprops(gap_labels)

    # Filter to gaps larger than min_gap_size x min_gap_size
    min_area = min_gap_size * min_gap_size
    large_gaps = [p for p in gap_props if p.area >= min_area]
    stats["n_gaps_found"] = len(large_gaps)

    if len(large_gaps) == 0:
        stats["n_cells_total"] = stats["n_cells_pass1"]
        return masks, stats

    # === PASS 2: Process each large gap with downscaling ===
    for gap in large_gaps:
        # Get bounding box with padding
        min_row, min_col, max_row, max_col = gap.bbox

        # Add padding (clamp to image bounds)
        min_row_pad = max(0, min_row - padding)
        min_col_pad = max(0, min_col - padding)
        max_row_pad = min(h, max_row + padding)
        max_col_pad = min(w, max_col + padding)

        # Extract crop
        crop = image[min_row_pad:max_row_pad, min_col_pad:max_col_pad]
        crop_h, crop_w = crop.shape
        existing_mask_crop = masks[min_row_pad:max_row_pad, min_col_pad:max_col_pad]

        # Downscale crop so large cells appear ~100px diameter
        new_h = int(crop_h * scale_factor)
        new_w = int(crop_w * scale_factor)

        if new_h < 50 or new_w < 50:
            # Crop too small after downscaling, skip
            stats["gap_regions"].append({
                "bbox": (min_row_pad, min_col_pad, max_row_pad, max_col_pad),
                "n_cells_added": 0,
                "skipped": "crop too small after downscaling",
            })
            continue

        # Downscale image (preserve intensity range)
        crop_downscaled = resize(crop, (new_h, new_w), order=1, preserve_range=True, anti_aliasing=True)

        # Run segmentation at primary diameter on downscaled crop
        gap_masks_small = segment_with_cellpose(
            crop_downscaled, model,
            diameter=primary_diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )

        if gap_masks_small is None or gap_masks_small.max() == 0:
            stats["gap_regions"].append({
                "bbox": (min_row_pad, min_col_pad, max_row_pad, max_col_pad),
                "n_cells_added": 0,
            })
            continue

        # Upscale mask back to original size (use nearest neighbor to preserve labels)
        gap_masks = resize(gap_masks_small, (crop_h, crop_w), order=0, preserve_range=True, anti_aliasing=False)
        gap_masks = gap_masks.astype(np.int32)

        # Merge new cells that don't overlap significantly with existing ones
        cells_added = 0
        for cell_id in range(1, int(gap_masks.max()) + 1):
            cell_mask = gap_masks == cell_id
            cell_area = np.sum(cell_mask)

            if cell_area == 0:
                continue

            # Check overlap with existing cells
            overlap_area = np.sum((existing_mask_crop > 0) & cell_mask)
            overlap_fraction = overlap_area / cell_area

            if overlap_fraction <= max_overlap_fraction:
                # Add this cell to the main mask (only in unsegmented regions)
                target_region = masks[min_row_pad:max_row_pad, min_col_pad:max_col_pad]
                # Only fill pixels that are currently unsegmented
                new_pixels = cell_mask & (target_region == 0)
                if np.sum(new_pixels) > 0:
                    target_region[new_pixels] = next_label
                    next_label += 1
                    cells_added += 1

        stats["n_cells_pass2"] += cells_added
        stats["gap_regions"].append({
            "bbox": (min_row_pad, min_col_pad, max_row_pad, max_col_pad),
            "n_cells_added": cells_added,
        })

    stats["n_cells_total"] = int(masks.max())

    # Verify labels are contiguous (relabel if needed)
    unique_labels = np.unique(masks)
    unique_labels = unique_labels[unique_labels > 0]
    if len(unique_labels) > 0 and unique_labels.max() != len(unique_labels):
        # Relabel to ensure contiguous IDs
        from skimage.measure import label as sk_label
        masks = sk_label(masks > 0, connectivity=1)
        # Preserve original cell boundaries by re-labeling
        new_masks = np.zeros_like(masks)
        for new_id, old_id in enumerate(np.unique(masks)[1:], start=1):
            new_masks[masks == old_id] = new_id
        masks = new_masks
        stats["n_cells_total"] = int(masks.max())

    return masks, stats


# ==============================================================================
# VISUALIZATION
# ==============================================================================


def plot_comparison(
    cellpose_metrics: dict,
    cpsam_metrics: dict,
    metric_name: str = "recall",
    title_suffix: str = "",
    save_path: Path = None,
):
    """
    Plot comparison of metrics between Cellpose baseline and cpsam.

    Parameters
    ----------
    cellpose_metrics : dict
        Metrics dictionary from multi_object_iou for Cellpose baseline.
    cpsam_metrics : dict
        Metrics dictionary from multi_object_iou for cpsam.
    metric_name : str
        Which metric to plot ("recall" or "precision").
    title_suffix : str
        Additional text for plot title.
    save_path : Path, optional
        Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Cellpose baseline
    cp_x = list(cellpose_metrics[metric_name].keys())
    cp_y = list(cellpose_metrics[metric_name].values())
    ax.plot(cp_x, cp_y, linewidth=2.5, label=f"Cellpose baseline ({metric_name}@0.5: {cellpose_metrics[metric_name][0.5]:.2f})", color="tab:blue")

    # cpsam
    cpsam_x = list(cpsam_metrics[metric_name].keys())
    cpsam_y = list(cpsam_metrics[metric_name].values())
    ax.plot(cpsam_x, cpsam_y, linewidth=2.5, label=f"cpsam ({metric_name}@0.5: {cpsam_metrics[metric_name][0.5]:.2f})", color="tab:orange")

    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("IoU Threshold", fontsize=12)
    ax.set_ylabel(metric_name.capitalize(), fontsize=12)
    ax.set_title(f"{metric_name.capitalize()} vs IoU Threshold {title_suffix}", fontsize=14)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_combined_metrics(
    cellpose_metrics: dict,
    cpsam_metrics: dict,
    save_path: Path = None,
):
    """Plot recall and precision side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, metric_name in zip(axes, ["recall", "precision"]):
        # Cellpose baseline
        cp_x = list(cellpose_metrics[metric_name].keys())
        cp_y = list(cellpose_metrics[metric_name].values())
        ax.plot(cp_x, cp_y, linewidth=2.5,
                label=f"Cellpose baseline ({cellpose_metrics[metric_name][0.5]:.2f}@0.5)",
                color="tab:blue")

        # cpsam
        cpsam_x = list(cpsam_metrics[metric_name].keys())
        cpsam_y = list(cpsam_metrics[metric_name].values())
        ax.plot(cpsam_x, cpsam_y, linewidth=2.5,
                label=f"cpsam ({cpsam_metrics[metric_name][0.5]:.2f}@0.5)",
                color="tab:orange")

        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("IoU Threshold", fontsize=12)
        ax.set_ylabel(metric_name.capitalize(), fontsize=12)
        ax.set_title(f"{metric_name.capitalize()} vs IoU Threshold", fontsize=14)
        # Move legend to right side
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10)
        ax.grid(True, alpha=0.3)

    # Add summary stats
    cp_cells = cellpose_metrics.get("total_pred_cells", "?")
    cpsam_cells = cpsam_metrics.get("total_pred_cells", "?")
    gt_cells = cellpose_metrics.get("total_true_cells", "?")
    fig.suptitle(f"Segmentation Comparison (GT: {gt_cells} cells, Cellpose baseline: {cp_cells}, cpsam: {cpsam_cells})",
                 fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def create_label_colormap(n_labels: int):
    """Create a colormap for segmentation labels with white background."""
    import matplotlib.colors as mcolors
    if n_labels == 0:
        return mcolors.ListedColormap(['white'])
    # Start with white for background (label 0), then use tab20 for cells
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, n_labels)))
    if n_labels > 20:
        colors = np.vstack([colors] * (n_labels // 20 + 1))[:n_labels]
    # Prepend white for background (index 0)
    colors_with_bg = np.vstack([[[1, 1, 1, 1]], colors])  # white RGBA
    return mcolors.ListedColormap(colors_with_bg)


def create_disagreement_overlay(model_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """
    Create an RGB overlay showing disagreement between model and ground truth.

    Uses colorblind-friendly palette:
    - Teal (#21918C): Agreement (both GT and model segmented, same cell boundary)
    - Orange (#F59E0B): Boundary mismatch (both segmented, but different cell IDs)
    - Yellow (#FDE725): False negative (GT has seg, model doesn't)
    - Magenta (#F768A1): False positive (model has seg, GT doesn't)
    - Background (black): Neither has segmentation

    Parameters
    ----------
    model_mask : np.ndarray
        Model segmentation mask (labels).
    gt_mask : np.ndarray
        Ground truth mask (labels).

    Returns
    -------
    np.ndarray
        RGB image (H, W, 3) with disagreement visualization.
    """
    h, w = gt_mask.shape
    overlay = np.zeros((h, w, 3), dtype=np.float32)

    gt_binary = gt_mask > 0
    model_binary = model_mask > 0

    # Both have segmentation
    both_segmented = gt_binary & model_binary
    # False negative: GT has seg, model doesn't
    false_neg = gt_binary & ~model_binary
    # False positive: model has seg, GT doesn't
    false_pos = ~gt_binary & model_binary

    # For pixels where both have segmentation, check if cell IDs match
    # We need to match GT cells to model cells first using IoU
    # For simplicity, we check if the labels are consistent within each GT cell region
    # A boundary mismatch occurs when GT and model both segment a pixel but assign
    # it to cells that don't correspond (different cell identity)

    # Build a mapping from GT cell IDs to their best-matching model cell IDs
    gt_labels = np.unique(gt_mask[gt_mask > 0])
    gt_to_model_map = {}

    for gt_label in gt_labels:
        gt_region = gt_mask == gt_label
        # Find which model labels overlap with this GT cell
        overlapping_model_labels = model_mask[gt_region]
        overlapping_model_labels = overlapping_model_labels[overlapping_model_labels > 0]
        if len(overlapping_model_labels) > 0:
            # The best match is the most common model label in this GT region
            unique, counts = np.unique(overlapping_model_labels, return_counts=True)
            best_model_label = unique[np.argmax(counts)]
            gt_to_model_map[gt_label] = best_model_label

    # Now classify each pixel where both have segmentation
    # Agreement: GT cell maps to model cell and this pixel has that model cell
    # Boundary mismatch: GT cell maps to model cell X, but pixel has model cell Y
    agreement = np.zeros((h, w), dtype=bool)
    boundary_mismatch = np.zeros((h, w), dtype=bool)

    for gt_label, model_label in gt_to_model_map.items():
        gt_region = gt_mask == gt_label
        # Pixels where GT says this cell AND model says the matched cell
        matched_pixels = gt_region & (model_mask == model_label)
        agreement |= matched_pixels
        # Pixels where GT says this cell but model says a different cell (not 0)
        mismatched_pixels = gt_region & (model_mask > 0) & (model_mask != model_label)
        boundary_mismatch |= mismatched_pixels

    # Handle GT cells with no model match - they count as false negative where model has no seg
    # and boundary mismatch where model has different seg
    for gt_label in gt_labels:
        if gt_label not in gt_to_model_map:
            gt_region = gt_mask == gt_label
            # Where model also has segmentation, it's a mismatch
            boundary_mismatch |= (gt_region & model_binary)

    # Teal (#21918C) for agreement - RGB: (33, 145, 140)
    overlay[agreement, 0] = 33 / 255
    overlay[agreement, 1] = 145 / 255
    overlay[agreement, 2] = 140 / 255

    # Orange (#F59E0B) for boundary mismatch - RGB: (245, 158, 11)
    overlay[boundary_mismatch, 0] = 245 / 255
    overlay[boundary_mismatch, 1] = 158 / 255
    overlay[boundary_mismatch, 2] = 11 / 255

    # Magenta (#F768A1) for false negative - RGB: (247, 104, 161)
    overlay[false_neg, 0] = 247 / 255
    overlay[false_neg, 1] = 104 / 255
    overlay[false_neg, 2] = 161 / 255

    # Yellow (#FDE725) for false positive - RGB: (253, 231, 37)
    overlay[false_pos, 0] = 253 / 255
    overlay[false_pos, 1] = 231 / 255
    overlay[false_pos, 2] = 37 / 255

    return overlay


def create_comparison_canvas(
    raw_image: np.ndarray,
    preprocessed_image: np.ndarray,
    cellpose_mask: np.ndarray,
    cpsam_mask: np.ndarray,
    ground_truth: np.ndarray,
    position_name: str,
    output_path: Path,
) -> None:
    """
    Create a comparison canvas showing raw, preprocessed, masks, overlays, and disagreement.

    Layout (3 rows x 4 cols):
    Row 0: Raw, CLAHE, Ground Truth, GT Overlay
    Row 1: Cellpose Mask, cpsam Mask, Cellpose Overlay, cpsam Overlay
    Row 2: Cellpose vs GT Disagreement, cpsam vs GT Disagreement, Legend, (empty)

    Disagreement colors:
    - Teal: Agreement (both have segmentation)
    - Orange: Boundary mismatch
    - Magenta: False negative (GT has seg, model doesn't)
    - Yellow: False positive (model has seg, GT doesn't)

    Parameters
    ----------
    raw_image : np.ndarray
        Raw image (single channel).
    preprocessed_image : np.ndarray
        CLAHE preprocessed image.
    cellpose_mask : np.ndarray
        Cellpose baseline segmentation mask.
    cpsam_mask : np.ndarray
        cpsam segmentation mask.
    ground_truth : np.ndarray
        Ground truth annotation mask.
    position_name : str
        Name of the position for title.
    output_path : Path
        Where to save the PNG.
    """
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f"Segmentation Comparison: {position_name}", fontsize=14, fontweight="bold", y=1.02)

    # Row 0: Raw, CLAHE, Ground Truth, GT Overlay
    axes[0, 0].imshow(raw_image, cmap="gray")
    axes[0, 0].set_title("Raw", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(preprocessed_image, cmap="gray")
    axes[0, 1].set_title("CLAHE Preprocessed", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    # Ground truth mask
    gt_n_cells = int(ground_truth.max())
    gt_cmap = create_label_colormap(gt_n_cells)
    axes[0, 2].imshow(ground_truth, cmap=gt_cmap, interpolation="nearest")
    axes[0, 2].set_title(f"Ground Truth ({gt_n_cells} cells)", fontsize=11, fontweight="bold")
    axes[0, 2].axis("off")

    # Ground truth overlay
    axes[0, 3].imshow(preprocessed_image, cmap="gray")
    gt_overlay = np.ma.masked_where(ground_truth == 0, ground_truth)
    axes[0, 3].imshow(gt_overlay, cmap=gt_cmap, alpha=0.5, interpolation="nearest")
    axes[0, 3].set_title("GT Overlay", fontsize=11, fontweight="bold")
    axes[0, 3].axis("off")

    # Row 1: Cellpose Mask, cpsam Mask, Cellpose Overlay, cpsam Overlay
    cp_n_cells = int(cellpose_mask.max())
    cp_cmap = create_label_colormap(cp_n_cells)
    axes[1, 0].imshow(cellpose_mask, cmap=cp_cmap, interpolation="nearest")
    axes[1, 0].set_title(f"Cellpose baseline ({cp_n_cells} cells)", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")

    cpsam_n_cells = int(cpsam_mask.max()) if cpsam_mask is not None else 0
    cpsam_cmap = create_label_colormap(cpsam_n_cells)
    if cpsam_mask is not None:
        axes[1, 1].imshow(cpsam_mask, cmap=cpsam_cmap, interpolation="nearest")
    else:
        axes[1, 1].imshow(preprocessed_image, cmap="gray", alpha=0.3)
        axes[1, 1].text(0.5, 0.5, "N/A", transform=axes[1, 1].transAxes, fontsize=16, ha="center", va="center")
    axes[1, 1].set_title(f"cpsam ({cpsam_n_cells} cells)", fontsize=11, fontweight="bold")
    axes[1, 1].axis("off")

    # Cellpose overlay (no highlight for unsegmented areas)
    axes[1, 2].imshow(preprocessed_image, cmap="gray")
    cp_overlay = np.ma.masked_where(cellpose_mask == 0, cellpose_mask)
    axes[1, 2].imshow(cp_overlay, cmap=cp_cmap, alpha=0.5, interpolation="nearest")
    axes[1, 2].set_title("Cellpose baseline Overlay", fontsize=11, fontweight="bold")
    axes[1, 2].axis("off")

    # cpsam overlay (no highlight for unsegmented areas)
    axes[1, 3].imshow(preprocessed_image, cmap="gray")
    if cpsam_mask is not None:
        cpsam_overlay = np.ma.masked_where(cpsam_mask == 0, cpsam_mask)
        axes[1, 3].imshow(cpsam_overlay, cmap=cpsam_cmap, alpha=0.5, interpolation="nearest")
    else:
        axes[1, 3].text(0.5, 0.5, "N/A", transform=axes[1, 3].transAxes, fontsize=16, ha="center", va="center")
    axes[1, 3].set_title("cpsam Overlay", fontsize=11, fontweight="bold")
    axes[1, 3].axis("off")

    # Row 2: Disagreement overlays
    # Cellpose vs GT disagreement
    cp_disagreement = create_disagreement_overlay(cellpose_mask, ground_truth)
    axes[2, 0].imshow(cp_disagreement, interpolation="nearest")
    axes[2, 0].set_title("Cellpose baseline vs GT", fontsize=11, fontweight="bold")
    axes[2, 0].axis("off")

    # cpsam vs GT disagreement
    if cpsam_mask is not None:
        cpsam_disagreement = create_disagreement_overlay(cpsam_mask, ground_truth)
        axes[2, 1].imshow(cpsam_disagreement, interpolation="nearest")
    else:
        axes[2, 1].imshow(preprocessed_image, cmap="gray", alpha=0.3)
        axes[2, 1].text(0.5, 0.5, "N/A", transform=axes[2, 1].transAxes, fontsize=16, ha="center", va="center")
    axes[2, 1].set_title("cpsam vs GT", fontsize=11, fontweight="bold")
    axes[2, 1].axis("off")

    # Legend with properly aligned color patches and text
    axes[2, 2].axis("off")
    axes[2, 2].set_xlim(0, 1)
    axes[2, 2].set_ylim(0, 1)

    # Title
    axes[2, 2].text(0.5, 0.95, "Disagreement Legend", transform=axes[2, 2].transAxes,
                    fontsize=11, fontweight="bold", ha="center", va="top")

    # Legend entries: (y_position, color, label, description)
    # 4 colors: agreement, boundary mismatch, false negative, false positive
    legend_entries = [
        (0.82, "#21918C", "Agreement", "(same cell assignment)"),
        (0.60, "#F59E0B", "Boundary Mismatch", "(different cell IDs)"),
        (0.38, "#F768A1", "False Negative", "(GT segmented, model missed)"),
        (0.16, "#FDE725", "False Positive", "(model segmented, GT empty)"),
    ]

    for y_pos, color, label, desc in legend_entries:
        # Color patch
        axes[2, 2].add_patch(plt.Rectangle((0.08, y_pos - 0.04), 0.08, 0.08,
                                            transform=axes[2, 2].transAxes,
                                            facecolor=color, edgecolor='black', linewidth=1))
        # Label text (aligned with patch center)
        axes[2, 2].text(0.20, y_pos, f"{label}", transform=axes[2, 2].transAxes,
                        fontsize=10, fontweight="bold", va="center", ha="left")
        # Description text (below label)
        axes[2, 2].text(0.20, y_pos - 0.08, desc, transform=axes[2, 2].transAxes,
                        fontsize=9, va="center", ha="left", color="gray")

    # Empty cell
    axes[2, 3].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved canvas: {output_path}")


# ==============================================================================
# DATA LOADING
# ==============================================================================


# Annotation paths (ground truth)
ANNOTATIONS = {
    "ops0033_A_1_025022": f"{BASE_PATH}/metrics/segmentation/annotations/labels_ops0033_A_1_025022.tif",
    "ops0036_A_1_010026": f"{BASE_PATH}/metrics/segmentation/annotations/labels_ops0036_A_1_010026.tif",
    "ops0063_A_2_040013": f"{BASE_PATH}/metrics/segmentation/annotations/labels_ops0063_A_2_040013.tif",
    "ops0086_A_3_042040": f"{BASE_PATH}/metrics/segmentation/annotations/labels_ops0086_A_3_042040.tif",
}

# Cellpose baseline predictions
CELLPOSE_BASELINE = {
    "ops0033_A_1_025022": f"{BASE_PATH}/metrics/segmentation/cellpose_baseline/model_pred_ops0033_A_1_025022.zarr",
    "ops0036_A_1_010026": f"{BASE_PATH}/metrics/segmentation/cellpose_baseline/model_pred_ops0036_A_1_010026.zarr",
    "ops0063_A_2_040013": f"{BASE_PATH}/metrics/segmentation/cellpose_baseline/model_pred_ops0063_A_2_040013.zarr",
    "ops0086_A_3_042040": f"{BASE_PATH}/metrics/segmentation/cellpose_baseline/model_pred_ops0086_A_3_042040.zarr",
}

# Raw image paths (for cpsam inference)
# Use the phenotyping_max_proj.zarr stores with correct experiment dates
RAW_IMAGE_STORES = {
    "ops0033": f"{BASE_PATH}/ops0033_20250429/1-preprocess/live_imaging/virtual_staining/phenotyping_max_proj.zarr",
    "ops0036": f"{BASE_PATH}/ops0036_20250505/1-preprocess/live_imaging/virtual_staining/phenotyping_max_proj.zarr",
    "ops0063": f"{BASE_PATH}/ops0063_20250731/1-preprocess/live_imaging/virtual_staining/phenotyping_max_proj.zarr",
    "ops0086": f"{BASE_PATH}/ops0086_20250922/1-preprocess/live_imaging/virtual_staining/phenotyping_max_proj.zarr",
}

# Position paths within zarr stores
POSITION_PATHS = {
    "ops0033_A_1_025022": ("ops0033", "A/1/025022"),
    "ops0036_A_1_010026": ("ops0036", "A/1/010026"),
    "ops0063_A_2_040013": ("ops0063", "A/2/040013"),
    "ops0086_A_3_042040": ("ops0086", "A/3/042040"),
}


class OpsDataset:
    """Minimal OpsDataset for path resolution only (avoids iohub import issues)."""

    def __init__(self, experiment: str):
        self.experiment = experiment
        self.experiment_path = Path(f"{BASE_PATH}/{experiment}")
        self.preprocess_live = self.experiment_path / "1-preprocess/live_imaging"

        self.store_paths = {
            "lc_20x_vs_max_proj": self.preprocess_live / "virtual_staining/phenotyping_max_proj.zarr",
            "lc_20x_segmentation_cells": self.preprocess_live / "segmentation/phenotyping_segmentation_cells.zarr",
        }


def load_ground_truth(key: str) -> np.ndarray:
    """Load ground truth annotation."""
    path = ANNOTATIONS[key]
    img = imread(path)
    # Handle different shapes
    if img.ndim == 4:
        return img[0, 0, :, :]
    elif img.ndim == 3:
        return img[0, :, :]
    return img


def load_cellpose_prediction(key: str) -> np.ndarray:
    """Load Cellpose baseline prediction."""
    path = CELLPOSE_BASELINE[key]
    pos_key = key.split("_", 1)[1]  # e.g., "A_1_025022" -> "A/1/025022"
    pos_path = pos_key.replace("_", "/")

    if IOHUB_AVAILABLE:
        with open_ome_zarr(path, mode='r') as store:
            return store[pos_path].data[0, 0, 0, :, :]
    else:
        store = zarr.open(path, mode='r')
        return np.asarray(store[pos_path]["0"][0, 0, 0, :, :])


def load_raw_image(key: str) -> np.ndarray:
    """Load raw image for Omnipose inference."""
    exp_key, pos_path = POSITION_PATHS[key]
    store_path = RAW_IMAGE_STORES[exp_key]

    if not Path(store_path).exists():
        print(f"WARNING: Raw image store not found: {store_path}")
        return None

    if IOHUB_AVAILABLE:
        with open_ome_zarr(store_path, mode='r') as store:
            # Get both channels for preprocessing
            data = store[pos_path].data[:]
            return data
    else:
        store = zarr.open(store_path, mode='r')
        return np.asarray(store[pos_path]["0"][:])


# ==============================================================================
# SWEEP MODE VISUALIZATION
# ==============================================================================


def plot_flow_sweep_metrics(
    cellpose_metrics: dict,
    cpsam_sweep_metrics: dict,
    diameter: float,
    save_path: Path = None,
):
    """
    Plot recall and precision for flow threshold sweep (single variable).

    Parameters
    ----------
    cellpose_metrics : dict
        Metrics dictionary from multi_object_iou for Cellpose baseline.
    cpsam_sweep_metrics : dict
        Dictionary mapping flow_threshold -> metrics dict.
    diameter : float
        Fixed diameter value used in sweep.
    save_path : Path, optional
        Path to save the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    n_params = len(cpsam_sweep_metrics)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_params))

    for ax, metric_name in zip(axes, ["recall", "precision"]):
        # Cellpose baseline (thicker, dashed line)
        cp_x = list(cellpose_metrics[metric_name].keys())
        cp_y = list(cellpose_metrics[metric_name].values())
        ax.plot(cp_x, cp_y, linewidth=3, linestyle="--",
                label=f"Cellpose baseline ({cellpose_metrics[metric_name][0.5]:.2f}@0.5)",
                color="tab:red")

        # cpsam flow threshold sweep
        for idx, (ft, metrics) in enumerate(sorted(cpsam_sweep_metrics.items())):
            cpsam_x = list(metrics[metric_name].keys())
            cpsam_y = list(metrics[metric_name].values())
            label = f"cpsam ft={ft} ({metrics[metric_name][0.5]:.2f}@0.5)"
            ax.plot(cpsam_x, cpsam_y, linewidth=2, label=label, color=colors[idx])

        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("IoU Threshold", fontsize=12)
        ax.set_ylabel(metric_name.capitalize(), fontsize=12)
        ax.set_title(f"{metric_name.capitalize()} vs IoU Threshold", fontsize=14)
        # Move legend to right side, outside the plot
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
        ax.grid(True, alpha=0.3)

    gt_cells = cellpose_metrics.get("total_true_cells", "?")
    fig.suptitle(f"cpsam Flow Threshold Sweep (d={diameter}, GT: {gt_cells} cells)",
                 fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_model_sweep_metrics(
    cellpose_metrics: dict,
    model_sweep_metrics: dict,
    save_path: Path = None,
):
    """
    Plot recall and precision for model type comparison.

    Parameters
    ----------
    cellpose_metrics : dict
        Metrics dictionary from multi_object_iou for Cellpose baseline.
    model_sweep_metrics : dict
        Dictionary mapping model_type -> metrics dict.
    save_path : Path, optional
        Path to save the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    n_models = len(model_sweep_metrics)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_models, 3)))

    for ax, metric_name in zip(axes, ["recall", "precision"]):
        # Cellpose baseline (thicker, dashed line)
        cp_x = list(cellpose_metrics[metric_name].keys())
        cp_y = list(cellpose_metrics[metric_name].values())
        ax.plot(cp_x, cp_y, linewidth=3, linestyle="--",
                label=f"Cellpose baseline ({cellpose_metrics[metric_name][0.5]:.2f}@0.5)",
                color="tab:red")

        # Cellpose model comparison
        for idx, (model_type, metrics) in enumerate(model_sweep_metrics.items()):
            m_x = list(metrics[metric_name].keys())
            m_y = list(metrics[metric_name].values())
            label = f"{model_type} ({metrics[metric_name][0.5]:.2f}@0.5)"
            ax.plot(m_x, m_y, linewidth=2, label=label, color=colors[idx])

        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("IoU Threshold", fontsize=12)
        ax.set_ylabel(metric_name.capitalize(), fontsize=12)
        ax.set_title(f"{metric_name.capitalize()} vs IoU Threshold", fontsize=14)
        # Move legend to right side, outside the plot
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
        ax.grid(True, alpha=0.3)

    gt_cells = cellpose_metrics.get("total_true_cells", "?")
    fig.suptitle(f"Cellpose Model Comparison (GT: {gt_cells} cells)",
                 fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def preprocess_raw_image(raw: np.ndarray, membrane_only: bool = False):
    """
    Preprocess raw image data: extract channels, normalize, max project, CLAHE.

    Parameters
    ----------
    raw : np.ndarray
        Raw image data with shape (T, C, Z, Y, X), (C, Z, Y, X), or (Y, X).
    membrane_only : bool
        If True, use only the membrane_prediction channel (channel 1) instead of
        max projection of nuclei and membrane channels.

    Returns
    -------
    raw_image_2d : np.ndarray
        Raw membrane channel for display.
    preprocessed : np.ndarray
        CLAHE-enhanced image for segmentation.
    """
    if raw.ndim == 5:  # (T, C, Z, Y, X)
        ch0 = raw[0, 0, 0, :, :]
        ch1 = raw[0, 1, 0, :, :]
        raw_image_2d = ch1
    elif raw.ndim == 4:  # (C, Z, Y, X)
        ch0 = raw[0, 0, :, :]
        ch1 = raw[1, 0, :, :]
        raw_image_2d = ch1
    else:
        ch0 = ch1 = raw
        raw_image_2d = raw

    # Normalize each channel
    ch0_norm = (ch0.astype(np.float32) - np.percentile(ch0, 1)) / (np.percentile(ch0, 99.5) - np.percentile(ch0, 1) + 1e-8)
    ch1_norm = (ch1.astype(np.float32) - np.percentile(ch1, 1)) / (np.percentile(ch1, 99.5) - np.percentile(ch1, 1) + 1e-8)
    ch0_norm = np.clip(ch0_norm, 0, 1)
    ch1_norm = np.clip(ch1_norm, 0, 1)

    # Use membrane only or max projection
    if membrane_only:
        combined = ch1_norm
    else:
        combined = np.maximum(ch0_norm, ch1_norm)

    preprocessed = equalize_adapthist(combined, clip_limit=0.01).astype(np.float32)

    return raw_image_2d, preprocessed


def run_single_evaluation(
    output_dir: Path,
    diameter: float = 100,
    flow_threshold: float = 0.7,
    model_type: str = "cpsam",
    membrane_only: bool = False,
):
    """
    Run single cpsam evaluation (original behavior).

    Parameters
    ----------
    output_dir : Path
        Directory to save outputs.
    diameter : float
        Cellpose diameter parameter (default: 100).
    flow_threshold : float
        Cellpose flow threshold parameter (default: 0.7).
    model_type : str
        Cellpose model type (default: cpsam).
    membrane_only : bool
        If True, use only the membrane_prediction channel for segmentation.
    """
    print("=" * 60)
    print("Segmentation Metrics: Cellpose baseline vs cpsam")
    print(f"cpsam params: d={diameter}, ft={flow_threshold}, model={model_type}")
    if membrane_only:
        print("Input: membrane_prediction channel only")
    print("=" * 60)

    output_dir.mkdir(exist_ok=True)

    # Load cpsam model
    cpsam_model = load_cellpose_model(model_type) if CELLPOSE_AVAILABLE else None

    # Collect predictions
    gt_images = []
    cellpose_images = []
    cpsam_images = []
    valid_keys = []

    for key in ANNOTATIONS.keys():
        print(f"\nProcessing: {key}")

        # Load ground truth
        gt = load_ground_truth(key)
        print(f"  Ground truth: {gt.shape}, {int(gt.max())} cells")

        # Load Cellpose baseline prediction
        try:
            cp = load_cellpose_prediction(key)
            print(f"  Cellpose baseline: {cp.shape}, {int(cp.max())} cells")
        except Exception as e:
            print(f"  Cellpose baseline: ERROR - {e}")
            continue

        # Load raw image and run cpsam
        raw_image_2d = None
        preprocessed = None
        cpsam_mask = None

        if cpsam_model is not None:
            raw = load_raw_image(key)
            if raw is not None:
                raw_image_2d, preprocessed = preprocess_raw_image(raw, membrane_only=membrane_only)

                # Run cpsam
                cpsam_mask = segment_with_cellpose(preprocessed, cpsam_model, diameter=diameter, flow_threshold=flow_threshold)
                print(f"  cpsam: {cpsam_mask.shape}, {int(cpsam_mask.max())} cells")

                # Save comparison canvas
                create_comparison_canvas(
                    raw_image=raw_image_2d,
                    preprocessed_image=preprocessed,
                    cellpose_mask=cp,
                    cpsam_mask=cpsam_mask,
                    ground_truth=gt,
                    position_name=key,
                    output_path=output_dir / f"comparison_{key}.png",
                )
            else:
                print(f"  cpsam: SKIPPED (no raw image)")
                cpsam_mask = np.zeros_like(gt)
        else:
            print(f"  cpsam: SKIPPED (model not available)")
            cpsam_mask = np.zeros_like(gt)

        gt_images.append(gt)
        cellpose_images.append(cp)
        cpsam_images.append(cpsam_mask)
        valid_keys.append(key)

    # Stack images
    gt_stack = np.stack(gt_images, axis=0)
    cp_stack = np.stack(cellpose_images, axis=0)
    cpsam_stack = np.stack(cpsam_images, axis=0)

    print(f"\n{'=' * 60}")
    print(f"Evaluation on {len(valid_keys)} FOVs")
    print("=" * 60)

    # Compute metrics
    print("\nComputing Cellpose baseline metrics...")
    cp_ious, cp_metrics = multi_object_iou(cp_stack, gt_stack)
    print(f"  Recall@0.5: {cp_metrics['recall'][0.5]:.3f}")
    print(f"  Precision@0.5: {cp_metrics['precision'][0.5]:.3f}")
    print(f"  Total predictions: {cp_metrics['total_pred_cells']}")

    print("\nComputing cpsam metrics...")
    cpsam_ious, cpsam_metrics = multi_object_iou(cpsam_stack, gt_stack)
    print(f"  Recall@0.5: {cpsam_metrics['recall'][0.5]:.3f}")
    print(f"  Precision@0.5: {cpsam_metrics['precision'][0.5]:.3f}")
    print(f"  Total predictions: {cpsam_metrics['total_pred_cells']}")

    # Plot metrics
    print("\nGenerating metric plots...")
    plot_combined_metrics(cp_metrics, cpsam_metrics, save_path=output_dir / "comparison_recall_precision.png")

    print(f"\n{'=' * 60}")
    print(f"Done! Results saved to: {output_dir}")
    print("=" * 60)


def _load_data_cache(membrane_only: bool = False):
    """Load all ground truth, cellpose predictions, and raw images into cache."""
    data_cache = {}  # key -> (gt, cp, raw_2d, preprocessed)

    for key in ANNOTATIONS.keys():
        print(f"  Loading: {key}")

        gt = load_ground_truth(key)
        try:
            cp = load_cellpose_prediction(key)
        except Exception as e:
            print(f"    Cellpose: ERROR - {e}")
            continue

        raw = load_raw_image(key)
        if raw is None:
            print(f"    Raw image: NOT FOUND")
            continue

        raw_image_2d, preprocessed = preprocess_raw_image(raw, membrane_only=membrane_only)
        data_cache[key] = (gt, cp, raw_image_2d, preprocessed)
        print(f"    GT: {int(gt.max())} cells, Cellpose: {int(cp.max())} cells")

    return data_cache


def run_flow_sweep(
    output_dir: Path,
    diameter: float = 100,
    flow_thresholds: list = None,
    model_type: str = "cpsam",
    membrane_only: bool = False,
):
    """
    Run single-variable sweep over flow_threshold (diameter fixed).

    Parameters
    ----------
    output_dir : Path
        Directory to save outputs.
    diameter : float
        Fixed diameter value (default: 100).
    flow_thresholds : list
        List of flow threshold values to sweep.
    model_type : str
        Cellpose model type (default: cpsam).
    membrane_only : bool
        If True, use only the membrane_prediction channel for segmentation.
    """
    if flow_thresholds is None:
        flow_thresholds = [0.6, 0.65, 0.7, 0.75, 0.8]

    print("=" * 60)
    print("Segmentation Metrics: cpsam Flow Threshold Sweep")
    print("=" * 60)
    print(f"Model: {model_type}")
    print(f"Fixed diameter: {diameter}")
    print(f"Flow thresholds: {flow_thresholds}")
    print(f"Total runs: {len(flow_thresholds)}")
    if membrane_only:
        print("Input: membrane_prediction channel only")
    print("=" * 60)

    output_dir.mkdir(exist_ok=True)

    # Load cpsam model
    cpsam_model = load_cellpose_model(model_type) if CELLPOSE_AVAILABLE else None
    if cpsam_model is None:
        print("ERROR: Cellpose not available. Cannot run sweep.")
        return

    # Load data
    print("\nLoading data...")
    data_cache = _load_data_cache(membrane_only=membrane_only)
    if not data_cache:
        print("ERROR: No valid data found.")
        return

    # Compute Cellpose baseline metrics
    gt_stack = np.stack([data_cache[k][0] for k in data_cache], axis=0)
    cp_stack = np.stack([data_cache[k][1] for k in data_cache], axis=0)

    print("\nComputing Cellpose baseline metrics...")
    _, cp_metrics = multi_object_iou(cp_stack, gt_stack)
    print(f"  Recall@0.5: {cp_metrics['recall'][0.5]:.3f}")
    print(f"  Precision@0.5: {cp_metrics['precision'][0.5]:.3f}")

    # Sweep over flow thresholds only
    cpsam_sweep_metrics = {}  # ft -> metrics

    for ft in flow_thresholds:
        print(f"\n--- cpsam d={diameter}, ft={ft} ---")

        cpsam_masks = {}
        for key, (gt, cp, raw_2d, preprocessed) in data_cache.items():
            mask = segment_with_cellpose(preprocessed, cpsam_model, diameter=diameter, flow_threshold=ft)
            cpsam_masks[key] = mask
            print(f"  {key}: {int(mask.max())} cells")

        # Stack and compute metrics
        cpsam_stack = np.stack([cpsam_masks[k] for k in data_cache], axis=0)
        _, cpsam_metrics = multi_object_iou(cpsam_stack, gt_stack)
        cpsam_sweep_metrics[ft] = cpsam_metrics

        print(f"  Recall@0.5: {cpsam_metrics['recall'][0.5]:.3f}")
        print(f"  Precision@0.5: {cpsam_metrics['precision'][0.5]:.3f}")

        # Save comparison canvases
        param_dir = output_dir / f"ft{ft}"
        param_dir.mkdir(exist_ok=True)

        for key, (gt, cp, raw_2d, preprocessed) in data_cache.items():
            create_comparison_canvas(
                raw_image=raw_2d,
                preprocessed_image=preprocessed,
                cellpose_mask=cp,
                cpsam_mask=cpsam_masks[key],
                ground_truth=gt,
                position_name=f"{key} (d={diameter}, ft={ft})",
                output_path=param_dir / f"comparison_{key}.png",
            )

    # Plot combined sweep metrics
    print("\nGenerating sweep metric plots...")
    plot_flow_sweep_metrics(cp_metrics, cpsam_sweep_metrics, diameter, save_path=output_dir / "flow_sweep_recall_precision.png")

    # Print summary table
    print(f"\n{'=' * 60}")
    print(f"cpsam FLOW THRESHOLD SWEEP SUMMARY (d={diameter})")
    print("=" * 60)
    print(f"{'Flow Threshold':<16} {'Recall@0.5':<12} {'Precision@0.5':<12} {'Pred Cells':<12}")
    print("-" * 60)
    print(f"{'Cellpose base':<16} {cp_metrics['recall'][0.5]:<12.3f} {cp_metrics['precision'][0.5]:<12.3f} {cp_metrics['total_pred_cells']:<12}")
    for ft, metrics in sorted(cpsam_sweep_metrics.items()):
        print(f"{f'cpsam ft={ft}':<16} {metrics['recall'][0.5]:<12.3f} {metrics['precision'][0.5]:<12.3f} {metrics['total_pred_cells']:<12}")
    print("=" * 60)

    print(f"\nDone! Results saved to: {output_dir}")


def run_model_sweep(
    output_dir: Path,
    diameter: float = 100,
    flow_threshold: float = 0.7,
    model_types: list = None,
    membrane_only: bool = False,
):
    """
    Run sweep comparing different Cellpose model types (d & ft fixed).

    Note: In cellpose v4+, most model types will use cpsam internally.

    Parameters
    ----------
    output_dir : Path
        Directory to save outputs.
    diameter : float
        Fixed diameter value (default: 100).
    flow_threshold : float
        Fixed flow threshold value (default: 0.7).
    model_types : list
        List of model types to compare.
    membrane_only : bool
        If True, use only the membrane_prediction channel for segmentation.
    """
    if model_types is None:
        model_types = ["cpsam", "cyto3", "cyto2"]

    print("=" * 60)
    print("Segmentation Metrics: Cellpose Model Type Comparison")
    print("=" * 60)
    print(f"Fixed diameter: {diameter}")
    print(f"Fixed flow_threshold: {flow_threshold}")
    print(f"Models to compare: {model_types}")
    if membrane_only:
        print("Input: membrane_prediction channel only")
    print("=" * 60)

    output_dir.mkdir(exist_ok=True)

    # Load data
    print("\nLoading data...")
    data_cache = _load_data_cache(membrane_only=membrane_only)
    if not data_cache:
        print("ERROR: No valid data found.")
        return

    # Compute Cellpose baseline metrics
    gt_stack = np.stack([data_cache[k][0] for k in data_cache], axis=0)
    cp_stack = np.stack([data_cache[k][1] for k in data_cache], axis=0)

    print("\nComputing Cellpose baseline metrics...")
    _, cp_metrics = multi_object_iou(cp_stack, gt_stack)
    print(f"  Recall@0.5: {cp_metrics['recall'][0.5]:.3f}")
    print(f"  Precision@0.5: {cp_metrics['precision'][0.5]:.3f}")

    # Sweep over model types
    model_sweep_metrics = {}  # model_type -> metrics

    for model_type in model_types:
        print(f"\n--- Model: {model_type} ---")

        cellpose_model = load_cellpose_model(model_type)
        if cellpose_model is None:
            print(f"  SKIPPED (model not available)")
            continue

        cellpose_masks = {}
        for key, (gt, cp, raw_2d, preprocessed) in data_cache.items():
            mask = segment_with_cellpose(preprocessed, cellpose_model, diameter=diameter, flow_threshold=flow_threshold)
            cellpose_masks[key] = mask
            print(f"  {key}: {int(mask.max())} cells")

        # Stack and compute metrics
        model_stack = np.stack([cellpose_masks[k] for k in data_cache], axis=0)
        _, model_metrics = multi_object_iou(model_stack, gt_stack)
        model_sweep_metrics[model_type] = model_metrics

        print(f"  Recall@0.5: {model_metrics['recall'][0.5]:.3f}")
        print(f"  Precision@0.5: {model_metrics['precision'][0.5]:.3f}")

        # Save comparison canvases
        param_dir = output_dir / model_type
        param_dir.mkdir(exist_ok=True)

        for key, (gt, cp, raw_2d, preprocessed) in data_cache.items():
            create_comparison_canvas(
                raw_image=raw_2d,
                preprocessed_image=preprocessed,
                cellpose_mask=cp,
                cpsam_mask=cellpose_masks[key],
                ground_truth=gt,
                position_name=f"{key} ({model_type})",
                output_path=param_dir / f"comparison_{key}.png",
            )

    # Plot combined sweep metrics
    print("\nGenerating model comparison plots...")
    plot_model_sweep_metrics(cp_metrics, model_sweep_metrics, save_path=output_dir / "model_comparison_recall_precision.png")

    # Print summary table
    print(f"\n{'=' * 60}")
    print(f"CELLPOSE MODEL COMPARISON SUMMARY (d={diameter}, ft={flow_threshold})")
    print("=" * 60)
    print(f"{'Model':<20} {'Recall@0.5':<12} {'Precision@0.5':<12} {'Pred Cells':<12}")
    print("-" * 60)
    print(f"{'Cellpose baseline':<20} {cp_metrics['recall'][0.5]:<12.3f} {cp_metrics['precision'][0.5]:<12.3f} {cp_metrics['total_pred_cells']:<12}")
    for model_type, metrics in model_sweep_metrics.items():
        print(f"{model_type:<20} {metrics['recall'][0.5]:<12.3f} {metrics['precision'][0.5]:<12.3f} {metrics['total_pred_cells']:<12}")
    print("=" * 60)

    print(f"\nDone! Results saved to: {output_dir}")


# ==============================================================================
# TILE COMPARISON MODE
# ==============================================================================


def create_tile_comparison_canvas(
    raw_image: np.ndarray,
    preprocessed_image: np.ndarray,
    existing_mask: np.ndarray,
    new_mask: np.ndarray,
    position_name: str,
    params_str: str,
    output_path: Path,
) -> None:
    """
    Create a comparison canvas for tile comparison mode (no ground truth).

    Layout (2 rows x 4 cols):
    Row 0: Raw, CLAHE, Existing Seg, New Seg
    Row 1: Existing Overlay, New Overlay, Disagreement, Metrics

    Parameters
    ----------
    raw_image : np.ndarray
        Raw image (single channel).
    preprocessed_image : np.ndarray
        CLAHE preprocessed image.
    existing_mask : np.ndarray
        Existing segmentation mask (from segment.py).
    new_mask : np.ndarray
        New segmentation mask (from this script).
    position_name : str
        Name of the position for title.
    params_str : str
        Parameter string (e.g., "d=100, ft=0.7").
    output_path : Path
        Where to save the PNG.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Tile Comparison: {position_name} ({params_str})", fontsize=14, fontweight="bold", y=1.02)

    # Row 0: Raw, CLAHE, Existing Seg, New Seg
    axes[0, 0].imshow(raw_image, cmap="gray")
    axes[0, 0].set_title("Raw", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(preprocessed_image, cmap="gray")
    axes[0, 1].set_title("CLAHE Preprocessed", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    # Existing segmentation mask
    existing_n_cells = int(existing_mask.max())
    existing_cmap = create_label_colormap(existing_n_cells)
    axes[0, 2].imshow(existing_mask, cmap=existing_cmap, interpolation="nearest")
    axes[0, 2].set_title(f"Existing (segment.py)\n{existing_n_cells} cells", fontsize=11, fontweight="bold")
    axes[0, 2].axis("off")

    # New segmentation mask
    new_n_cells = int(new_mask.max())
    new_cmap = create_label_colormap(new_n_cells)
    axes[0, 3].imshow(new_mask, cmap=new_cmap, interpolation="nearest")
    axes[0, 3].set_title(f"New (this script)\n{new_n_cells} cells", fontsize=11, fontweight="bold")
    axes[0, 3].axis("off")

    # Row 1: Existing Overlay, New Overlay, Disagreement, Metrics
    # Existing overlay
    axes[1, 0].imshow(preprocessed_image, cmap="gray")
    existing_overlay = np.ma.masked_where(existing_mask == 0, existing_mask)
    axes[1, 0].imshow(existing_overlay, cmap=existing_cmap, alpha=0.5, interpolation="nearest")
    axes[1, 0].set_title("Existing Overlay", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")

    # New overlay
    axes[1, 1].imshow(preprocessed_image, cmap="gray")
    new_overlay = np.ma.masked_where(new_mask == 0, new_mask)
    axes[1, 1].imshow(new_overlay, cmap=new_cmap, alpha=0.5, interpolation="nearest")
    axes[1, 1].set_title("New Overlay", fontsize=11, fontweight="bold")
    axes[1, 1].axis("off")

    # Disagreement (new vs existing)
    disagreement = create_disagreement_overlay(new_mask, existing_mask)
    axes[1, 2].imshow(disagreement, interpolation="nearest")
    axes[1, 2].set_title("Disagreement (New vs Existing)", fontsize=11, fontweight="bold")
    axes[1, 2].axis("off")

    # Legend
    axes[1, 3].axis("off")
    axes[1, 3].set_xlim(0, 1)
    axes[1, 3].set_ylim(0, 1)

    axes[1, 3].text(0.5, 0.95, "Disagreement Legend", transform=axes[1, 3].transAxes,
                    fontsize=11, fontweight="bold", ha="center", va="top")

    legend_entries = [
        (0.75, "#21918C", "Agreement"),
        (0.55, "#F59E0B", "Boundary Mismatch"),
        (0.35, "#F768A1", "Only Existing"),
        (0.15, "#FDE725", "Only New"),
    ]

    for y_pos, color, label in legend_entries:
        axes[1, 3].add_patch(plt.Rectangle((0.08, y_pos - 0.04), 0.08, 0.08,
                                            transform=axes[1, 3].transAxes,
                                            facecolor=color, edgecolor='black', linewidth=1))
        axes[1, 3].text(0.20, y_pos, label, transform=axes[1, 3].transAxes,
                        fontsize=10, fontweight="bold", va="center", ha="left")

    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def run_tile_comparison(
    experiment: str,
    position: str,
    output_dir: Path,
    diameter: float = 100,
    flow_threshold: float = 0.7,
    model_type: str = "cpsam",
    source_zarr: str = None,
    seg_zarr: str = None,
    flip_y: bool = False,
    gap_filling: bool = False,
    scale_factor: float = 0.25,
    min_gap_size: int = 200,
    membrane_only: bool = False,
):
    """
    Process a specific tile from a zarr store and compare with existing segmentation.

    This mode allows comparing the segmentation output from this script with
    segment.py output to ensure consistency.

    Parameters
    ----------
    experiment : str
        Experiment name (e.g., ops0100_20251218). Used to resolve paths via OpsDataset.
    position : str
        Position path within zarr stores (e.g., A/2/029032).
    output_dir : Path
        Directory to save outputs.
    diameter : float
        Diameter parameter for segmentation.
    flow_threshold : float
        Flow threshold parameter for segmentation.
    model_type : str
        Cellpose model type (default: cpsam).
    source_zarr : str, optional
        Override source zarr store path. If None, uses OpsDataset.store_paths["lc_20x_vs_max_proj"].
    seg_zarr : str, optional
        Override segmentation zarr store path. If None, uses OpsDataset.store_paths["lc_20x_segmentation_cells"].
    flip_y : bool
        If True, flip image along Y axis before segmentation (to match segment.py behavior).
    membrane_only : bool
        If True, use only the membrane_prediction channel for segmentation.
    """
    # OpsDataset class is defined locally in this file to avoid iohub import issues

    # Resolve paths using OpsDataset
    ds = OpsDataset(experiment)

    if source_zarr is None:
        source_zarr = str(ds.store_paths["lc_20x_vs_max_proj"])
    if seg_zarr is None:
        # Check for debug store first (has per-tile segmentation), otherwise use main store
        debug_seg_path = ds.preprocess_live / "segmentation/phenotyping_segmentation_cells_debug.zarr"
        if debug_seg_path.exists():
            seg_zarr = str(debug_seg_path)
            print(f"[Using debug segmentation store with per-tile data]")
        else:
            seg_zarr = str(ds.store_paths["lc_20x_segmentation_cells"])

    print("=" * 60)
    print("Tile Comparison Mode")
    print(f"Experiment: {experiment}")
    print(f"Source zarr: {source_zarr}")
    print(f"Seg zarr: {seg_zarr}")
    print(f"Position: {position}")
    print(f"Params: d={diameter}, ft={flow_threshold}, model={model_type}")
    print("=" * 60)

    output_dir.mkdir(exist_ok=True, parents=True)

    # Load source image from zarr
    print(f"\nLoading source image from {source_zarr}/{position}...")
    source_path = Path(source_zarr)

    raw_image = None
    if IOHUB_AVAILABLE:
        try:
            with open_ome_zarr(source_path, layout="fov", mode="r") as dataset:
                pos = dataset[position]
                raw_image = pos.data[:]
                print(f"  Shape: {raw_image.shape}, dtype: {raw_image.dtype}")
        except Exception as e:
            print(f"  iohub failed: {e}, falling back to zarr")

    if raw_image is None:
        # Direct zarr access
        import zarr
        store = zarr.open(source_path, mode="r")
        pos_data = store[position]
        if hasattr(pos_data, "0"):
            raw_image = pos_data["0"][:]
        else:
            raw_image = pos_data[:]
        print(f"  Shape: {raw_image.shape}, dtype: {raw_image.dtype}")

    # Load existing segmentation
    print(f"\nLoading existing segmentation from {seg_zarr}/{position}...")
    seg_path = Path(seg_zarr)

    existing_seg = None
    if IOHUB_AVAILABLE:
        try:
            with open_ome_zarr(seg_path, layout="fov", mode="r") as dataset:
                pos = dataset[position]
                existing_seg = pos.data[:]
                print(f"  Shape: {existing_seg.shape}, dtype: {existing_seg.dtype}")
        except Exception as e:
            print(f"  iohub failed: {e}, falling back to zarr")

    if existing_seg is None:
        store = zarr.open(seg_path, mode="r")
        pos_data = store[position]
        if hasattr(pos_data, "0"):
            existing_seg = pos_data["0"][:]
        else:
            existing_seg = pos_data[:]
        print(f"  Shape: {existing_seg.shape}, dtype: {existing_seg.dtype}")

    # Preprocess raw image (same as segment.py)
    print("\nPreprocessing image...")
    if membrane_only:
        print("  [membrane_only=True] Using only membrane_prediction channel")
    raw_2d, preprocessed = preprocess_raw_image(raw_image, membrane_only=membrane_only)
    print(f"  Raw 2D shape: {raw_2d.shape}")
    print(f"  Preprocessed shape: {preprocessed.shape}, range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")

    # Extract 2D segmentation mask
    if existing_seg.ndim == 5:  # (T, C, Z, Y, X)
        existing_seg_2d = existing_seg[0, 0, 0, :, :]
    elif existing_seg.ndim == 4:  # (C, Z, Y, X)
        existing_seg_2d = existing_seg[0, 0, :, :]
    elif existing_seg.ndim == 3:  # (Z, Y, X) or (C, Y, X)
        existing_seg_2d = existing_seg[0, :, :]
    else:
        existing_seg_2d = existing_seg

    print(f"  Existing seg 2D: {existing_seg_2d.shape}, {int(existing_seg_2d.max())} cells")

    # Load model and run segmentation
    print(f"\nLoading model and running segmentation...")
    if flip_y:
        print("  [flip_y=True] Flipping image along Y axis (to match segment.py)")
    model = load_cellpose_model(model_type) if CELLPOSE_AVAILABLE else None

    if model is None:
        print("ERROR: Could not load model. Make sure cellpose is installed.")
        return

    # Flip Y axis before segmentation if requested (matches segment.py behavior)
    seg_input = preprocessed[::-1] if flip_y else preprocessed

    if gap_filling:
        print(f"  [gap_filling=True] Two-pass segmentation: d={diameter}, scale={scale_factor}, min_gap={min_gap_size}")
        new_seg, gap_stats = segment_with_gap_filling(
            seg_input, model,
            primary_diameter=diameter,
            scale_factor=scale_factor,
            flow_threshold=flow_threshold,
            min_gap_size=min_gap_size,
        )
        print(f"  Gap-filling stats:")
        print(f"    Pass 1 cells: {gap_stats['n_cells_pass1']}")
        print(f"    Gaps found (>={min_gap_size}x{min_gap_size}): {gap_stats['n_gaps_found']}")
        print(f"    Pass 2 cells added: {gap_stats['n_cells_pass2']}")
        print(f"    Total cells: {gap_stats['n_cells_total']}")
        for i, region in enumerate(gap_stats["gap_regions"]):
            bbox = region["bbox"]
            print(f"    Gap {i+1}: bbox={bbox}, cells_added={region['n_cells_added']}")
    else:
        new_seg = segment_with_cellpose(seg_input, model, diameter=diameter, flow_threshold=flow_threshold)

    # Flip back if we flipped before
    if flip_y:
        new_seg = new_seg[::-1]
    print(f"  New segmentation: {new_seg.shape}, {int(new_seg.max())} cells")

    # Compute metrics comparing new vs existing
    print("\nComparing new segmentation vs existing segmentation...")

    # Stack for batch processing
    new_batch = new_seg[np.newaxis, ...]
    existing_batch = existing_seg_2d[np.newaxis, ...]

    ious, metrics = multi_object_iou(new_batch, existing_batch)

    print(f"\n{'=' * 40}")
    print("COMPARISON METRICS (New vs Existing)")
    print("=" * 40)
    print(f"New segmentation cells: {int(new_seg.max())}")
    print(f"Existing segmentation cells: {int(existing_seg_2d.max())}")
    print(f"Recall@0.5: {metrics['recall'][0.5]:.3f}")
    print(f"Precision@0.5: {metrics['precision'][0.5]:.3f}")
    if len(ious) > 0:
        print(f"Mean IoU (matched): {np.mean(ious):.3f}")
    else:
        print("Mean IoU: N/A (no matches)")
    print("=" * 40)

    # Create visualization canvas
    print("\nGenerating comparison canvas...")
    if gap_filling:
        params_str = f"d={diameter}, scale={scale_factor}, ft={flow_threshold}, model={model_type}, gap_fill"
    else:
        params_str = f"d={diameter}, ft={flow_threshold}, model={model_type}"
    create_tile_comparison_canvas(
        raw_image=raw_2d,
        preprocessed_image=preprocessed,
        existing_mask=existing_seg_2d,
        new_mask=new_seg,
        position_name=position,
        params_str=params_str,
        output_path=output_dir / f"tile_comparison_{position.replace('/', '_')}.png",
    )

    print(f"\nDone! Results saved to: {output_dir}")


# ==============================================================================
# CLI
# ==============================================================================

import argparse
from cyclops_process.paths import BASE_PATH

# Available Cellpose models (cellpose v4+)
CELLPOSE_MODELS = [
    "cpsam",           # Cellpose-SAM (default, best performance)
    "cyto3",           # Legacy cytoplasm model (uses cpsam in v4)
    "cyto2",           # Legacy cytoplasm model
    "cyto",            # Original cytoplasm model
    "nuclei",          # Nuclei segmentation
]


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation quality: Cellpose baseline vs cpsam vs Ground Truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single evaluation (default: d=100, ft=0.7, model=cpsam)
  python -m cyclops_process.metrics.graphs.metrics_segmentation

  # Single evaluation with custom params
  python -m cyclops_process.metrics.graphs.metrics_segmentation --diameter 100 --flow-threshold 0.65

  # Sweep flow thresholds (diameter fixed)
  python -m cyclops_process.metrics.graphs.metrics_segmentation --sweep-flow

  # Sweep flow with custom values
  python -m cyclops_process.metrics.graphs.metrics_segmentation --sweep-flow \\
      --flow-thresholds 0.5 0.6 0.7 0.8 0.9

  # Compare different Cellpose models
  python -m cyclops_process.metrics.graphs.metrics_segmentation --sweep-models

  # Compare specific models
  python -m cyclops_process.metrics.graphs.metrics_segmentation --sweep-models \\
      --model-types cpsam cyto3 cyto2

  # Compare tile from zarr with existing segmentation (uses OpsDataset paths)
  python -m cyclops_process.metrics.graphs.metrics_segmentation --compare-tile \\
      --experiment ops0100_20251218 --position A/2/029032

Available Cellpose models (v4+):
  cpsam    - Cellpose-SAM (default, best performance)
  cyto3    - Legacy cytoplasm model (uses cpsam in v4)
  cyto2    - Legacy cytoplasm model
  cyto     - Original cytoplasm model
  nuclei   - Nuclei segmentation
""",
    )

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--sweep-flow",
        action="store_true",
        help="Sweep flow_threshold values (diameter fixed)",
    )
    mode_group.add_argument(
        "--sweep-models",
        action="store_true",
        help="Compare different Omnipose model types (d & ft fixed)",
    )
    mode_group.add_argument(
        "--compare-tile",
        action="store_true",
        help="Process a specific tile from zarr and compare with existing segmentation",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save outputs (default: auto-generated based on mode)",
    )

    # Model selection
    parser.add_argument(
        "--model-type",
        type=str,
        default="cpsam",
        choices=CELLPOSE_MODELS,
        help="Cellpose model for single evaluation (default: cpsam)",
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        type=str,
        default=["cpsam", "cyto3", "cyto2"],
        help="Model types to compare in --sweep-models mode",
    )

    # Segmentation parameters
    parser.add_argument(
        "--diameter",
        type=str,
        default="100",
        help="Cellpose diameter (default: 100). Use 'auto' or '0' for automatic estimation.",
    )
    parser.add_argument(
        "--flow-threshold",
        type=float,
        default=0.7,
        help="Cellpose flow threshold (default: 0.7)",
    )

    # Flow sweep parameters
    parser.add_argument(
        "--flow-thresholds",
        nargs="+",
        type=float,
        default=[0.6, 0.65, 0.7, 0.75, 0.8],
        help="Flow threshold values to sweep (default: 0.6 0.65 0.7 0.75 0.8)",
    )

    # Tile comparison parameters
    parser.add_argument(
        "--experiment",
        type=str,
        help="Experiment name (e.g., ops0100_20251218) for --compare-tile mode",
    )
    parser.add_argument(
        "--source-zarr",
        type=str,
        default=None,
        help="Override source zarr store path (default: OpsDataset.lc_20x_vs_max_proj)",
    )
    parser.add_argument(
        "--seg-zarr",
        type=str,
        default=None,
        help="Override existing segmentation zarr store path (default: OpsDataset.lc_20x_segmentation_cells)",
    )
    parser.add_argument(
        "--position",
        type=str,
        default="A/2/029032",
        help="Position path within zarr stores (e.g., A/2/029032)",
    )
    parser.add_argument(
        "--flip-y",
        action="store_true",
        help="Flip image along Y axis before segmentation (matches segment.py behavior)",
    )

    # Gap-filling parameters
    parser.add_argument(
        "--gap-filling",
        action="store_true",
        help="Enable two-pass gap-filling segmentation (d=100, then downscale large gaps)",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=0.25,
        help="Scale factor for downscaling gap crops (default: 0.25 = 4x downscale, so 400px cells appear 100px)",
    )
    parser.add_argument(
        "--min-gap-size",
        type=int,
        default=200,
        help="Minimum gap size in pixels to trigger second pass (default: 200)",
    )

    # Input channel selection
    parser.add_argument(
        "--membrane-only",
        action="store_true",
        help="Use only the membrane_prediction channel for segmentation (instead of max projection of nuclei + membrane)",
    )

    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Parse diameter: support 'auto', '0', or numeric value
    if args.diameter.lower() == "auto" or args.diameter == "0":
        diameter = None  # Cellpose will auto-estimate
        print("[diameter=None] Using automatic diameter estimation")
    else:
        diameter = float(args.diameter)

    if args.sweep_flow:
        output_dir = Path(args.output_dir) if args.output_dir else Path("./segmentation_flow_sweep")
        run_flow_sweep(
            output_dir=output_dir,
            diameter=diameter,
            flow_thresholds=args.flow_thresholds,
            model_type=args.model_type,
            membrane_only=args.membrane_only,
        )
    elif args.sweep_models:
        output_dir = Path(args.output_dir) if args.output_dir else Path("./segmentation_model_sweep")
        run_model_sweep(
            output_dir=output_dir,
            diameter=diameter,
            flow_threshold=args.flow_threshold,
            model_types=args.model_types,
            membrane_only=args.membrane_only,
        )
    elif args.compare_tile:
        if not args.experiment:
            parser.error("--compare-tile requires --experiment")
        output_dir = Path(args.output_dir) if args.output_dir else Path("./tile_comparison_output")
        run_tile_comparison(
            experiment=args.experiment,
            position=args.position,
            output_dir=output_dir,
            diameter=diameter,
            flow_threshold=args.flow_threshold,
            model_type=args.model_type,
            source_zarr=args.source_zarr,
            seg_zarr=args.seg_zarr,
            flip_y=args.flip_y,
            gap_filling=args.gap_filling,
            scale_factor=args.scale_factor,
            min_gap_size=args.min_gap_size,
            membrane_only=args.membrane_only,
        )
    else:
        output_dir = Path(args.output_dir) if args.output_dir else Path("./segmentation_metrics_output")
        run_single_evaluation(
            output_dir=output_dir,
            diameter=diameter,
            flow_threshold=args.flow_threshold,
            model_type=args.model_type,
            membrane_only=args.membrane_only,
        )


#%%
# ==============================================================================
# INTERACTIVE MODE (for Jupyter/VSCode cells)
# ==============================================================================

if __name__ == "__main__":
    main()

# %%
