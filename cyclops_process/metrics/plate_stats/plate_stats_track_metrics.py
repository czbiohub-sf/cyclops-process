"""
ISS Tracking Metrics Helper Functions.

This module provides helper functions for computing QC metrics from tracking
outputs (GEFF graphs and linked results) for inclusion in plate_stats.csv.

Data Sources:
1. Tracking GEFF: 2-tracking/A{well}_tracks.geff
   - Graph-based tracking data with cell positions and lineages

2. Linked Results: dataset.result_paths["linked_results"] -> results_fast/A{well}_linked_pheno_iss.csv
   - Matched tracking-ISS-pheno cell data with barcodes and segmentation IDs
   - Path: /path/to/ops_data/{experiment}/3-assembly/

3. Link Metrics: dataset.result_paths["link_metrics"] -> results_fast/link_metrics.csv
   - Per-well pipeline progression stats (tracks through ISS/pheno linking)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict

from ops_utils.data.filesystem import parse_well


@dataclass
class TrackingGraphStats:
    """Statistics from tracking graph analysis."""
    total_nodes: int
    total_edges: int
    num_timepoints: int
    division_events: int  # Nodes with >1 child
    disappearance_events: int  # Nodes with no children at t < T-1
    appearance_events: int  # Nodes at t > 0 with no parents
    full_length_tracks: int  # Tracks spanning all timepoints


@dataclass
class MovementStats:
    """Statistics for cell movement between timepoints."""
    mean_distance: float
    median_distance: float
    std_distance: float
    min_distance: float
    max_distance: float
    n_movements: int


@dataclass
class DaughterBarcodeStats:
    """Statistics for daughter cell barcode matching."""
    total_divisions: int
    divisions_with_barcodes: int
    matching_barcode_pairs: int
    pct_matching: float


@dataclass
class TrackingLossStats:
    """Statistics for cell loss from ISS to tracking."""
    iss_matched_cells: int
    tracked_cells: int
    pct_tracked: float


_tracksdata_warning_shown = False

def load_tracking_graph(
    dataset,
    well: str,
) -> Optional[Tuple[Any, dict]]:
    """
    Load tracking GEFF graph for a well.

    Uses fast loading with only essential node properties (no masks).

    Args:
        dataset: OpsDataset instance
        well: Well identifier (e.g., "A/1/0")

    Returns:
        Tuple of (graph, metadata) or None if not found
    """
    global _tracksdata_warning_shown

    try:
        import tracksdata as td
    except ImportError:
        if not _tracksdata_warning_shown:
            print("Warning: tracksdata package not available. Track metrics will be skipped. "
                  "Use dask_viewer_env to enable tracking metrics.")
            _tracksdata_warning_shown = True
        return None

    try:
        geff_path = dataset.append_well("tracking_geff", well)
        if not Path(geff_path).exists():
            return None

        # Fast loading with only essential node properties (skip masks)
        node_props = [
            td.DEFAULT_ATTR_KEYS.T,
            "y",
            "x",
            td.DEFAULT_ATTR_KEYS.SOLUTION,
        ]

        graph, metadata = td.graph.IndexedRXGraph.from_geff(
            geff_path,
            geff_read_kwargs={"node_props": node_props}
        )

        # Compute tracklet_ids from solution if not already present
        if td.DEFAULT_ATTR_KEYS.TRACKLET_ID not in graph.node_attr_keys():
            graph.filter(
                td.NodeAttr(td.DEFAULT_ATTR_KEYS.SOLUTION) == True,
                td.EdgeAttr(td.DEFAULT_ATTR_KEYS.SOLUTION) == True,
            ).subgraph().assign_tracklet_ids(
                output_key=td.DEFAULT_ATTR_KEYS.TRACKLET_ID
            )

        return graph, metadata
    except Exception as e:
        print(f"Warning: Failed to load tracking graph for {well}: {e}")
        return None


def compute_tracking_graph_stats(
    graph,
    well: str,
) -> Optional[TrackingGraphStats]:
    """
    Compute statistics from a tracking graph.

    Args:
        graph: tracksdata graph instance (IndexedRXGraph or InMemoryGraph)
        well: Well identifier for context

    Returns:
        TrackingGraphStats or None if computation fails
    """
    try:
        import rustworkx as rx

        node_ids = list(graph.node_ids())
        if not node_ids:
            return None

        # Get timepoints present in the graph
        timepoints = set()
        node_timepoints = {}
        for node_id in node_ids:
            node_data = graph.rx_graph[node_id]
            t = node_data.get("t", 0)
            timepoints.add(t)
            node_timepoints[node_id] = t

        min_t = min(timepoints)
        max_t = max(timepoints)
        num_timepoints = len(timepoints)

        # Count edges
        total_edges = graph.rx_graph.num_edges()

        # Build parent/child relationships
        children_of = defaultdict(list)
        parents_of = defaultdict(list)

        for edge in graph.rx_graph.edge_list():
            src, tgt = edge
            children_of[src].append(tgt)
            parents_of[tgt].append(src)

        # Count division events (nodes with >1 child)
        division_events = sum(1 for node_id in node_ids if len(children_of[node_id]) > 1)

        # Count disappearance events (nodes at t < max_t with no children)
        disappearance_events = 0
        for node_id in node_ids:
            t = node_timepoints[node_id]
            if t < max_t and len(children_of[node_id]) == 0:
                disappearance_events += 1

        # Count appearance events (nodes at t > min_t with no parents)
        appearance_events = 0
        for node_id in node_ids:
            t = node_timepoints[node_id]
            if t > min_t and len(parents_of[node_id]) == 0:
                appearance_events += 1

        # Count full-length tracks (nodes at max_t with ancestors spanning all timepoints)
        full_length_tracks = 0
        for node_id in node_ids:
            if node_timepoints[node_id] == max_t:
                ancestors = rx.ancestors(graph.rx_graph, node_id)
                if len(ancestors) == num_timepoints - 1:
                    full_length_tracks += 1

        return TrackingGraphStats(
            total_nodes=len(node_ids),
            total_edges=total_edges,
            num_timepoints=num_timepoints,
            division_events=division_events,
            disappearance_events=disappearance_events,
            appearance_events=appearance_events,
            full_length_tracks=full_length_tracks,
        )
    except Exception:
        return None


def compute_movement_stats(
    graph,
) -> Optional[MovementStats]:
    """
    Compute movement statistics from tracking graph edges.

    Calculates distance moved between connected cells across timepoints.

    Args:
        graph: tracksdata InMemoryGraph instance

    Returns:
        MovementStats or None if not enough data
    """
    try:
        distances = []

        for edge in graph.rx_graph.edge_list():
            src, tgt = edge
            src_data = graph.rx_graph[src]
            tgt_data = graph.rx_graph[tgt]

            # Get coordinates
            src_y = src_data.get("y", 0)
            src_x = src_data.get("x", 0)
            tgt_y = tgt_data.get("y", 0)
            tgt_x = tgt_data.get("x", 0)

            # Compute Euclidean distance
            dist = np.sqrt((tgt_y - src_y) ** 2 + (tgt_x - src_x) ** 2)
            distances.append(dist)

        if not distances:
            return None

        return MovementStats(
            mean_distance=float(np.round(np.mean(distances), 2)),
            median_distance=float(np.round(np.median(distances), 2)),
            std_distance=float(np.round(np.std(distances), 2)),
            min_distance=float(np.round(np.min(distances), 2)),
            max_distance=float(np.round(np.max(distances), 2)),
            n_movements=len(distances),
        )
    except Exception:
        return None


def get_per_track_totals(
    graph,
) -> List[float]:
    """
    Get the total distance traveled for each unique cell track.

    For each unique cell track, sums all edge distances along that track.

    Args:
        graph: tracksdata InMemoryGraph instance

    Returns:
        List of total distances (one per track). Empty list if computation fails.
    """
    try:
        # Build track ID to total distance mapping
        track_distances: Dict[Any, float] = defaultdict(float)

        for edge in graph.rx_graph.edge_list():
            src, tgt = edge
            src_data = graph.rx_graph[src]
            tgt_data = graph.rx_graph[tgt]

            # Get tracklet_id to identify which track this edge belongs to
            tracklet_id = src_data.get("tracklet_id", None)
            if tracklet_id is None:
                continue

            # Get coordinates
            src_y = src_data.get("y", 0)
            src_x = src_data.get("x", 0)
            tgt_y = tgt_data.get("y", 0)
            tgt_x = tgt_data.get("x", 0)

            # Compute Euclidean distance
            dist = np.sqrt((tgt_y - src_y) ** 2 + (tgt_x - src_x) ** 2)

            # Add to this track's total distance
            track_distances[tracklet_id] += dist

        return list(track_distances.values())
    except Exception:
        return []


def compute_per_track_total_distance_stats(
    graph,
) -> Dict[str, float]:
    """
    Compute per-track total distance statistics.

    For each unique cell track, sums all edge distances along that track,
    then computes mean/median of those track totals.

    Args:
        graph: tracksdata InMemoryGraph instance

    Returns:
        Dictionary with track_mean_total_distance and track_median_total_distance.
        Returns empty dict if computation fails.
    """
    track_totals = get_per_track_totals(graph)

    if not track_totals:
        return {}

    return {
        "track_mean_total_distance": float(np.round(np.mean(track_totals), 2)),
        "track_median_total_distance": float(np.round(np.median(track_totals), 2)),
    }


def compute_per_timepoint_movement_stats(
    graph,
) -> Dict[str, float]:
    """
    Compute per-timepoint movement statistics from tracking graph edges.

    Groups edges by their source timepoint and computes mean/median distance
    for each timepoint transition (e.g., 0to1, 1to2, etc.).

    Args:
        graph: tracksdata InMemoryGraph instance

    Returns:
        Dictionary with keys like 'track_mean_distance_0to1', 'track_median_distance_0to1', etc.
        Also includes 'track_mean_distance_per_timepoint' and 'track_median_distance_per_timepoint'
        as aggregated stats across all timepoint transitions.
        Returns empty dict if computation fails.
    """
    try:
        import tracksdata as td

        # Group distances by source timepoint
        distances_by_timepoint: Dict[int, List[float]] = defaultdict(list)

        for edge in graph.rx_graph.edge_list():
            src, tgt = edge
            src_data = graph.rx_graph[src]
            tgt_data = graph.rx_graph[tgt]

            # Get timepoints
            src_t = src_data.get(td.DEFAULT_ATTR_KEYS.T, None)
            tgt_t = tgt_data.get(td.DEFAULT_ATTR_KEYS.T, None)

            if src_t is None or tgt_t is None:
                continue

            # Get coordinates
            src_y = src_data.get("y", 0)
            src_x = src_data.get("x", 0)
            tgt_y = tgt_data.get("y", 0)
            tgt_x = tgt_data.get("x", 0)

            # Compute Euclidean distance
            dist = np.sqrt((tgt_y - src_y) ** 2 + (tgt_x - src_x) ** 2)

            # Group by source timepoint (transition from src_t to tgt_t)
            distances_by_timepoint[int(src_t)].append(dist)

        result = {}
        per_timepoint_means = []
        per_timepoint_medians = []

        for src_t in sorted(distances_by_timepoint.keys()):
            tgt_t = src_t + 1
            dists = distances_by_timepoint[src_t]
            if dists:
                mean_dist = float(np.round(np.mean(dists), 2))
                median_dist = float(np.round(np.median(dists), 2))
                result[f"track_mean_distance_{src_t}to{tgt_t}"] = mean_dist
                result[f"track_median_distance_{src_t}to{tgt_t}"] = median_dist
                per_timepoint_means.append(mean_dist)
                per_timepoint_medians.append(median_dist)

        # Add aggregated stats across all timepoint transitions
        if per_timepoint_means:
            result["track_mean_distance_per_timepoint"] = float(np.round(np.mean(per_timepoint_means), 2))
            result["track_median_distance_per_timepoint"] = float(np.round(np.median(per_timepoint_means), 2))

        return result
    except Exception:
        return {}


def load_linked_results(
    dataset,
    well: str,
) -> Optional[pd.DataFrame]:
    """
    Load linked results CSV for a well.

    Args:
        dataset: OpsDataset instance
        well: Well identifier (e.g., "A/1/0")

    Returns:
        DataFrame with linked results or None if not found
    """
    try:
        linked_path = dataset.append_well("linked_results", well)
        if not Path(linked_path).exists():
            return None

        df = pd.read_csv(linked_path)
        if len(df) == 0:
            return None
        return df
    except Exception:
        return None


def compute_doublet_stats(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compute doublet statistics from linked results.

    Doublets are defined as multiple tracks mapping to the same segmentation_id.

    Args:
        df: Linked results DataFrame with segmentation_id column

    Returns:
        Dictionary with doublet statistics
    """
    result = {
        "doublet_count": 0,
        "doublet_pct": 0.0,
        "total_unique_cells": 0,
    }

    if df is None or df.empty or "segmentation_id" not in df.columns:
        return result

    # Filter to valid segmentation IDs
    valid_df = df[df["segmentation_id"].notna()]
    if valid_df.empty:
        return result

    # Count tracks per segmentation_id
    tracks_per_seg = valid_df.groupby("segmentation_id").size()

    # Doublets: seg_ids with >1 track
    doublet_seg_ids = tracks_per_seg[tracks_per_seg > 1]
    doublet_count = len(doublet_seg_ids)
    total_unique_cells = len(tracks_per_seg)

    result["doublet_count"] = doublet_count
    result["doublet_pct"] = float(np.round(100 * doublet_count / total_unique_cells, 2)) if total_unique_cells > 0 else 0.0
    result["total_unique_cells"] = total_unique_cells

    return result


def compute_daughter_barcode_stats(
    df: pd.DataFrame,
) -> Optional[DaughterBarcodeStats]:
    """
    Compute statistics for daughter cell barcode matching.

    Looks for cells that map to the same segmentation_id (potential divisions)
    and checks if they have matching barcodes.

    Args:
        df: Linked results DataFrame with segmentation_id and barcode columns

    Returns:
        DaughterBarcodeStats or None if not enough data
    """
    if df is None or df.empty:
        return None

    if "segmentation_id" not in df.columns or "barcode" not in df.columns:
        return None

    # Filter to valid segmentation IDs and barcodes
    valid_df = df[df["segmentation_id"].notna()].copy()
    if valid_df.empty:
        return None

    # Find seg_ids with multiple tracks (potential divisions)
    tracks_per_seg = valid_df.groupby("segmentation_id").size()
    division_seg_ids = tracks_per_seg[tracks_per_seg > 1].index

    if len(division_seg_ids) == 0:
        return DaughterBarcodeStats(
            total_divisions=0,
            divisions_with_barcodes=0,
            matching_barcode_pairs=0,
            pct_matching=0.0,
        )

    total_divisions = len(division_seg_ids)
    divisions_with_barcodes = 0
    matching_barcode_pairs = 0

    for seg_id in division_seg_ids:
        seg_cells = valid_df[valid_df["segmentation_id"] == seg_id]
        barcodes = seg_cells["barcode"].dropna().unique()

        # Filter out zero/empty barcodes
        valid_barcodes = [b for b in barcodes if b != 0 and b != "0" and str(b).strip() != ""]

        if len(valid_barcodes) > 0:
            divisions_with_barcodes += 1
            # If all barcodes match (only 1 unique barcode), it's a match
            if len(valid_barcodes) == 1:
                matching_barcode_pairs += 1

    pct_matching = float(np.round(100 * matching_barcode_pairs / divisions_with_barcodes, 2)) if divisions_with_barcodes > 0 else 0.0

    return DaughterBarcodeStats(
        total_divisions=total_divisions,
        divisions_with_barcodes=divisions_with_barcodes,
        matching_barcode_pairs=matching_barcode_pairs,
        pct_matching=pct_matching,
    )


def compute_post_tracking_cells_per_gene(
    linked_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compute cells per gene statistics from linked results (post-tracking).

    Args:
        linked_df: Linked results DataFrame with gene_name column

    Returns:
        Dictionary with post-tracking cells per gene stats
    """
    result = {
        "post_track_mean_cells_per_gene": None,
        "post_track_median_cells_per_gene": None,
        "post_track_total_tracked_cells": None,
        "post_track_genes_detected": None,
    }

    if linked_df is None or linked_df.empty:
        return result

    # Use gene_name or library-specific gene column
    gene_col = None
    for col in _gene_col_candidates():
        if col in linked_df.columns:
            gene_col = col
            break

    if gene_col is None:
        return result

    # Filter to valid gene assignments
    valid_df = linked_df[linked_df[gene_col].notna()].copy()
    if valid_df.empty:
        return result

    # Count cells per gene
    cells_per_gene = valid_df.groupby(gene_col).size()

    result["post_track_total_tracked_cells"] = len(valid_df)
    result["post_track_genes_detected"] = len(cells_per_gene)
    result["post_track_mean_cells_per_gene"] = float(np.round(cells_per_gene.mean(), 2))
    result["post_track_median_cells_per_gene"] = float(np.round(cells_per_gene.median(), 2))

    return result


def get_post_tracking_stats_for_well(
    dataset,
    well: str,
) -> Dict[str, Any]:
    """
    Get post-tracking cells per gene statistics for a single well.

    Args:
        dataset: OpsDataset instance
        well: Well identifier (e.g., "A/1/0")

    Returns:
        Dictionary with post-tracking stats for this well
    """
    linked_df = load_linked_results(dataset, well)
    return compute_post_tracking_cells_per_gene(linked_df)


def get_tracking_graph_stats_for_well(
    dataset,
    well: str,
) -> Dict[str, Any]:
    """
    Get tracking graph and doublet/division statistics for a single well.

    Args:
        dataset: OpsDataset instance
        well: Well identifier (e.g., "A/1/0")

    Returns:
        Dictionary with tracking stats for this well
    """
    result = {
        # Graph stats
        "track_total_nodes": None,
        "track_total_edges": None,
        "track_division_events": None,
        "track_disappearance_events": None,
        "track_appearance_events": None,
        "track_full_length_tracks": None,
        # Movement stats
        "track_mean_distance": None,
        "track_median_distance": None,
        "track_std_distance": None,
        "track_max_distance": None,
        # Per-track total distance stats (mean/median of per-track totals)
        "track_mean_total_distance": None,
        "track_median_total_distance": None,
        # Per-timepoint aggregated stats (mean/median of per-timepoint means)
        "track_mean_distance_per_timepoint": None,
        "track_median_distance_per_timepoint": None,
        # Doublet stats
        "track_doublet_count": None,
        "track_doublet_pct": None,
        # Daughter barcode stats
        "track_divisions_total": None,
        "track_divisions_with_barcodes": None,
        "track_daughter_barcode_match_pct": None,
    }

    # Graph stats
    graph_result = load_tracking_graph(dataset, well)
    if graph_result is not None:
        graph, _ = graph_result
        graph_stats = compute_tracking_graph_stats(graph, well)
        if graph_stats:
            result["track_total_nodes"] = graph_stats.total_nodes
            result["track_total_edges"] = graph_stats.total_edges
            result["track_division_events"] = graph_stats.division_events
            result["track_disappearance_events"] = graph_stats.disappearance_events
            result["track_appearance_events"] = graph_stats.appearance_events
            result["track_full_length_tracks"] = graph_stats.full_length_tracks
            # Use division_events from graph for track_divisions_total
            result["track_divisions_total"] = graph_stats.division_events

        # Movement stats
        movement_stats = compute_movement_stats(graph)
        if movement_stats:
            result["track_mean_distance"] = movement_stats.mean_distance
            result["track_median_distance"] = movement_stats.median_distance
            result["track_std_distance"] = movement_stats.std_distance
            result["track_max_distance"] = movement_stats.max_distance

        # Per-track total distance stats (mean/median of total path length per cell track)
        per_track_stats = compute_per_track_total_distance_stats(graph)
        result.update(per_track_stats)

        # Per-timepoint movement stats (includes aggregated per-timepoint stats)
        per_timepoint_stats = compute_per_timepoint_movement_stats(graph)
        result.update(per_timepoint_stats)

    # Linked results stats
    linked_df = load_linked_results(dataset, well)
    if linked_df is not None:
        # Doublet stats
        doublet_stats = compute_doublet_stats(linked_df)
        result["track_doublet_count"] = doublet_stats["doublet_count"]
        if doublet_stats["total_unique_cells"] > 0:
            result["track_doublet_pct"] = doublet_stats["doublet_pct"]

        # Daughter barcode stats (divisions_with_barcodes from linked data)
        daughter_stats = compute_daughter_barcode_stats(linked_df)
        if daughter_stats:
            # Use division_events from graph for total (already set above if graph exists)
            # Only set from linked if graph wasn't available
            if result["track_divisions_total"] is None:
                result["track_divisions_total"] = daughter_stats.total_divisions
            result["track_divisions_with_barcodes"] = daughter_stats.divisions_with_barcodes
            if daughter_stats.divisions_with_barcodes > 0:
                result["track_daughter_barcode_match_pct"] = daughter_stats.pct_matching

    return result


def compute_tracking_loss_stats(
    dataset,
    well: str,
) -> Optional[TrackingLossStats]:
    """
    Compute cell loss from ISS to tracking.

    Note: This function is currently disabled due to circular dependency
    (it would need to read plate_stats.csv which we're generating).

    Args:
        dataset: OpsDataset instance
        well: Well identifier (e.g., "A/1/0")

    Returns:
        TrackingLossStats or None if data not available
    """
    # Disabled due to circular dependency - would need plate_stats.csv
    # to get cells_with_matched_reads, but we're generating that file
    return None


def load_link_metrics(
    dataset,
) -> Optional[pd.DataFrame]:
    """
    Load link_metrics.csv containing per-well tracking pipeline progression stats.

    This CSV is generated by link_calls_tracks() in datasets.py and contains:
    - 1_full_length_tracks: Total tracks spanning all timepoints
    - 2_iss_linked: Tracks after ISS linking
    - 3_iss_linked_with_barcode: Tracks with non-zero barcode
    - 4_pheno_linked: Tracks after pheno linking
    - 5_tracks_with_gene_matched: Tracks after gene merge
    - 6_after_bbox: Final output tracks

    Args:
        dataset: OpsDataset instance

    Returns:
        DataFrame with link metrics or None if not found
    """
    try:
        link_metrics_path = dataset.result_paths.get("link_metrics")
        if link_metrics_path is None or not Path(link_metrics_path).exists():
            return None

        df = pd.read_csv(link_metrics_path)
        if len(df) == 0:
            return None
        return df
    except Exception:
        return None


def compute_link_pipeline_stats(
    link_metrics_df: pd.DataFrame,
    well: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute link pipeline statistics from link_metrics.csv.

    Reports key pipeline progression stats without duplicates:
    - link_full_length_tracks: Total full-length tracks from tracking
    - link_with_barcode: Tracks that have ISS barcodes
    - link_iss_pheno_matched: Tracks matched to genes via ISS+pheno linking (final output)
    - link_fallback_bbox: Tracks using fallback bounding box (no valid segmentation)

    Args:
        link_metrics_df: DataFrame from link_metrics.csv
        well: Optional well to filter (e.g., "A/1/0"). If None, sums across all wells.

    Returns:
        Dictionary with pipeline progression stats
    """
    result = {
        "link_full_length_tracks": None,
        "link_with_barcode": None,
        "link_iss_pheno_matched": None,
        "link_fallback_bbox": None,
    }

    if link_metrics_df is None or link_metrics_df.empty:
        return result

    df = link_metrics_df

    # Filter by well if specified
    if well is not None and "well" in df.columns:
        # Normalize well format: "A/1/0" -> "A1", "A/1" -> "A1"
        if "/" in well:
            parts = well.split("/")
            well_normalized = f"{parts[0]}{parts[1]}"  # "A/1/0" -> "A1"
        else:
            well_normalized = well

        # Try matching against the well column
        df = df[df["well"].str.replace("/", "", regex=False).str.replace("-", "", regex=False) == well_normalized]
        if df.empty:
            # Try exact match
            df = link_metrics_df[link_metrics_df["well"] == well]
        if df.empty:
            return result

    # Sum across matched wells (or single well)
    if "1_full_length_tracks" in df.columns:
        result["link_full_length_tracks"] = int(df["1_full_length_tracks"].sum())

    if "3_iss_linked_with_barcode" in df.columns:
        result["link_with_barcode"] = int(df["3_iss_linked_with_barcode"].sum())

    if "5_tracks_with_gene_matched" in df.columns:
        result["link_iss_pheno_matched"] = int(df["5_tracks_with_gene_matched"].sum())

    if "6_fallback_bbox" in df.columns:
        result["link_fallback_bbox"] = int(df["6_fallback_bbox"].sum())

    return result


def get_link_stats_for_well(
    dataset,
    well: str,
) -> Dict[str, Any]:
    """
    Get link pipeline statistics for a single well.

    Args:
        dataset: OpsDataset instance
        well: Well identifier (e.g., "A/1/0")

    Returns:
        Dictionary with link stats for this well
    """
    link_metrics_df = load_link_metrics(dataset)
    return compute_link_pipeline_stats(link_metrics_df, well=well)


def get_tracking_stats_for_experiment(
    dataset,
    wells: List[str] = ["A/1/0", "A/2/0", "A/3/0"],
) -> Dict[str, Any]:
    """
    Get all tracking statistics for an experiment.

    This is the main entry point for metrics.py to get tracking stats.

    Args:
        dataset: OpsDataset instance
        wells: List of wells to process

    Returns:
        Dictionary with all tracking stats, ready to add to plate_stats
    """
    result = {
        # Link pipeline progression stats (from link_metrics.csv)
        "link_full_length_tracks": None,
        "link_iss_linked": None,
        "link_iss_linked_with_barcode": None,
        "link_pheno_linked": None,
        "link_gene_matched": None,
        "link_final_output": None,
        "link_valid_segmentation": None,
        "link_fallback_bbox": None,
        # Graph stats (aggregated across wells)
        "track_total_nodes": None,
        "track_total_edges": None,
        "track_division_events": None,
        "track_disappearance_events": None,
        "track_appearance_events": None,
        "track_full_length_tracks": None,
        # Movement stats
        "track_mean_distance": None,
        "track_median_distance": None,
        "track_std_distance": None,
        "track_max_distance": None,
        # Per-track total distance stats (mean/median of per-track totals)
        "track_mean_total_distance": None,
        "track_median_total_distance": None,
        # Per-timepoint aggregated stats (mean/median of per-timepoint means)
        "track_mean_distance_per_timepoint": None,
        "track_median_distance_per_timepoint": None,
        # Doublet stats
        "track_doublet_count": None,
        "track_doublet_pct": None,
        # Daughter barcode stats
        "track_divisions_total": None,
        "track_divisions_with_barcodes": None,
        "track_daughter_barcode_match_pct": None,
    }

    # Load link pipeline metrics from link_metrics.csv
    link_metrics_df = load_link_metrics(dataset)
    if link_metrics_df is not None:
        link_stats = compute_link_pipeline_stats(link_metrics_df)
        result.update(link_stats)

    # Aggregation accumulators
    all_nodes = 0
    all_edges = 0
    all_divisions = 0
    all_disappearances = 0
    all_appearances = 0
    all_full_tracks = 0

    all_distances = []
    all_track_totals = []  # Per-track total distances for aggregation
    all_timepoint_means = []  # Per-timepoint mean distances for aggregation

    all_doublets = 0
    all_unique_cells = 0

    all_division_events = 0
    all_divisions_with_barcodes = 0
    all_matching_pairs = 0

    for well in wells:
        # Graph stats
        graph_result = load_tracking_graph(dataset, well)
        if graph_result is not None:
            graph, _ = graph_result
            graph_stats = compute_tracking_graph_stats(graph, well)
            if graph_stats:
                all_nodes += graph_stats.total_nodes
                all_edges += graph_stats.total_edges
                all_divisions += graph_stats.division_events
                all_disappearances += graph_stats.disappearance_events
                all_appearances += graph_stats.appearance_events
                all_full_tracks += graph_stats.full_length_tracks

            # Movement stats
            movement_stats = compute_movement_stats(graph)
            if movement_stats and movement_stats.n_movements > 0:
                # We need to collect raw distances for proper aggregation
                # For now, use weighted average approach
                all_distances.extend([movement_stats.mean_distance] * movement_stats.n_movements)

            # Per-track total distance stats - collect raw totals for proper aggregation
            track_totals = get_per_track_totals(graph)
            all_track_totals.extend(track_totals)

            # Per-timepoint stats
            per_timepoint_stats = compute_per_timepoint_movement_stats(graph)
            if per_timepoint_stats:
                # Extract the per-timepoint means for aggregation
                for key, val in per_timepoint_stats.items():
                    if key.startswith("track_mean_distance_") and "per_timepoint" not in key:
                        all_timepoint_means.append(val)

        # Linked results stats
        linked_df = load_linked_results(dataset, well)
        if linked_df is not None:
            # Doublet stats
            doublet_stats = compute_doublet_stats(linked_df)
            all_doublets += doublet_stats["doublet_count"]
            all_unique_cells += doublet_stats["total_unique_cells"]

            # Daughter barcode stats
            daughter_stats = compute_daughter_barcode_stats(linked_df)
            if daughter_stats:
                all_division_events += daughter_stats.total_divisions
                all_divisions_with_barcodes += daughter_stats.divisions_with_barcodes
                all_matching_pairs += daughter_stats.matching_barcode_pairs

    # Populate aggregated results
    if all_nodes > 0:
        result["track_total_nodes"] = all_nodes
        result["track_total_edges"] = all_edges
        result["track_division_events"] = all_divisions
        result["track_disappearance_events"] = all_disappearances
        result["track_appearance_events"] = all_appearances
        result["track_full_length_tracks"] = all_full_tracks

    if all_distances:
        result["track_mean_distance"] = float(np.round(np.mean(all_distances), 2))
        result["track_median_distance"] = float(np.round(np.median(all_distances), 2))
        result["track_std_distance"] = float(np.round(np.std(all_distances), 2))
        result["track_max_distance"] = float(np.round(np.max(all_distances), 2))

    # Per-track total distance stats (mean/median of total path length per cell track)
    if all_track_totals:
        result["track_mean_total_distance"] = float(np.round(np.mean(all_track_totals), 2))
        result["track_median_total_distance"] = float(np.round(np.median(all_track_totals), 2))

    # Per-timepoint aggregated stats
    if all_timepoint_means:
        result["track_mean_distance_per_timepoint"] = float(np.round(np.mean(all_timepoint_means), 2))
        result["track_median_distance_per_timepoint"] = float(np.round(np.median(all_timepoint_means), 2))

    if all_unique_cells > 0:
        result["track_doublet_count"] = all_doublets
        result["track_doublet_pct"] = float(np.round(100 * all_doublets / all_unique_cells, 2))

    if all_division_events > 0:
        result["track_divisions_total"] = all_division_events
        result["track_divisions_with_barcodes"] = all_divisions_with_barcodes
        if all_divisions_with_barcodes > 0:
            result["track_daughter_barcode_match_pct"] = float(np.round(100 * all_matching_pairs / all_divisions_with_barcodes, 2))

    return result


def get_tracking_stats_per_well(
    dataset,
    wells: List[str] = ["A/1/0", "A/2/0", "A/3/0"],
) -> Dict[str, Any]:
    """
    Get per-well tracking statistics for an experiment.

    Args:
        dataset: OpsDataset instance
        wells: List of wells to process

    Returns:
        Dictionary with per-well tracking stats
    """
    result = {}

    for well in wells:
        row, col = parse_well(well)  # full token avoids A1/B1 key collision
        prefix = f"track_w{row}{col}"

        # Initialize per-well keys
        result[f"{prefix}_nodes"] = None
        result[f"{prefix}_divisions"] = None
        result[f"{prefix}_disappearances"] = None
        result[f"{prefix}_appearances"] = None
        result[f"{prefix}_full_tracks"] = None
        result[f"{prefix}_doublet_pct"] = None
        result[f"{prefix}_iss_to_track_pct"] = None

        # Graph stats
        graph_result = load_tracking_graph(dataset, well)
        if graph_result is not None:
            graph, _ = graph_result
            graph_stats = compute_tracking_graph_stats(graph, well)
            if graph_stats:
                result[f"{prefix}_nodes"] = graph_stats.total_nodes
                result[f"{prefix}_divisions"] = graph_stats.division_events
                result[f"{prefix}_disappearances"] = graph_stats.disappearance_events
                result[f"{prefix}_appearances"] = graph_stats.appearance_events
                result[f"{prefix}_full_tracks"] = graph_stats.full_length_tracks

        # Doublet stats
        linked_df = load_linked_results(dataset, well)
        if linked_df is not None:
            doublet_stats = compute_doublet_stats(linked_df)
            if doublet_stats["total_unique_cells"] > 0:
                result[f"{prefix}_doublet_pct"] = doublet_stats["doublet_pct"]

        # Tracking loss
        loss_stats = compute_tracking_loss_stats(dataset, well)
        if loss_stats:
            result[f"{prefix}_iss_to_track_pct"] = loss_stats.pct_tracked

    return result


# ---------------------------------------------------------------------------
# Link-level histogram plots (from linked CSV files, post-tracking)
# ---------------------------------------------------------------------------

_GENE_COL_CANDIDATES_CACHE = None


def _gene_col_candidates() -> list:
    """Gene/perturbation-column names to look for in produced tables: the standard
    names plus any custom `gene_name_output_column` values configured per experiment
    in the library map (so custom perturbation-column names are sourced from config,
    never hardcoded here)."""
    global _GENE_COL_CANDIDATES_CACHE
    if _GENE_COL_CANDIDATES_CACHE is None:
        custom = []
        try:
            from ops_utils.data.bad_experiments import load_library_map
            lm = load_library_map() or {}
            default = (lm.get("default") or {}).get("gene_name_output_column")
            if default:
                custom.append(default)
            for ov in (lm.get("overrides") or {}).values():
                c = (ov or {}).get("gene_name_output_column")
                if c and c not in custom:
                    custom.append(c)
        except Exception:
            pass
        _GENE_COL_CANDIDATES_CACHE = ["gene_name", *custom, "dep_map_gene_name", "NCBI_ID"]
    return _GENE_COL_CANDIDATES_CACHE
_GUIDE_COL_CANDIDATES = ["sgRNA", "barcode"]


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return first matching column name from *candidates* present in *df*."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _load_all_linked_for_experiment(
    experiment: str,
    method: str = "mine",
):
    """Create dataset, discover wells, and load all linked CSVs.

    Returns (dataset, pooled_df) or (dataset, None) if no linked data found.
    """
    from ops_utils.data.experiment import OpsDataset
    from iohub.ngff import open_ome_zarr

    dataset = OpsDataset(experiment, method=method)
    try:
        seg_store = open_ome_zarr(dataset.store_paths["iss_segmentation"], mode="r")
        wells = [a[0] for a in seg_store.positions()]
    except Exception as e:
        print(f"Could not load segmentation data to discover wells: {e}")
        return dataset, None

    frames = []
    for well in wells:
        df = load_linked_results(dataset, well)
        if df is not None:
            df["_well"] = well
            frames.append(df)
    if not frames:
        return dataset, None
    return dataset, pd.concat(frames, ignore_index=True)


def plot_link_cells_per_gene_histogram(
    experiment: str,
    method: str = "mine",
) -> None:
    """Plot histogram of cells per gene from linked CSV files (post-tracking).

    Mirrors :func:`iss_histrogram.plot_cells_per_gene_histogram` but uses the
    linked results instead of raw ISS reads.

    Args:
        experiment: Experiment name (e.g. ``"ops0141_20260101"``)
        method: Base calling method (``"mine"`` or ``"probabilistic"``)
    """
    print("--- Generating linked cells-per-gene histogram ---")
    dataset, pooled = _load_all_linked_for_experiment(experiment, method)
    if pooled is None or pooled.empty:
        print("No linked results found. Aborting linked cells-per-gene histogram.")
        return

    gene_col = _find_col(pooled, _gene_col_candidates())
    if gene_col is None:
        print("No gene column found in linked results. Aborting histogram.")
        return

    valid = pooled[pooled[gene_col].notna()]
    if valid.empty:
        print("No valid gene assignments in linked results. Aborting histogram.")
        return

    # Each row in the linked CSV is one tracked cell
    cells_per_gene = valid.groupby(gene_col).size()
    total_cells = len(valid)

    mean_cpg = cells_per_gene.mean()
    median_cpg = cells_per_gene.median()
    std_cpg = cells_per_gene.std()
    xlim_upper = cells_per_gene.quantile(0.95)

    TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE = 20, 16, 12, 12

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(cells_per_gene, bins=100, range=(0, xlim_upper), alpha=0.7,
            label="Cells per Gene Distribution")
    ax.axvline(mean_cpg, color="red", linestyle="dashed", linewidth=2,
               label=f"Mean: {mean_cpg:.2f}")
    ax.axvline(median_cpg, color="green", linestyle="dashed", linewidth=2,
               label=f"Median: {median_cpg:.2f}")

    fig.suptitle("Linked Cells per Gene (All Wells Pooled)", fontsize=TITLE_SIZE)
    ax.set_title(dataset.experiment, fontsize=LABEL_SIZE - 4)
    ax.set_xlabel("Number of Cells", fontsize=LABEL_SIZE)
    ax.set_ylabel("Number of Genes", fontsize=LABEL_SIZE)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim(0, xlim_upper)
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)

    legend_text = (
        f"Total Linked Cells: {total_cells}\n"
        f"Mean Cells/Gene: {mean_cpg:.2f}\n"
        f"Median Cells/Gene: {median_cpg:.2f}\n"
        f"Std Dev: {std_cpg:.2f}"
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(0.95, 0.95, legend_text, transform=ax.transAxes, fontsize=LEGEND_SIZE,
            verticalalignment="top", horizontalalignment="right", bbox=props)
    ax.legend(loc="upper left", fontsize=LEGEND_SIZE)

    plt.tight_layout()
    save_path = dataset.metrics_paths["link_cells_per_gene_histogram"]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Saved linked cells-per-gene histogram to {save_path}")
    plt.close()


def plot_link_top_genes_by_cell_count(
    experiment: str,
    method: str = "mine",
    top_n: int = 50,
) -> None:
    """Bar plot of top N genes by cell count from linked CSV files (post-tracking).

    Mirrors :func:`iss_histrogram.plot_top_genes_by_cell_count`.

    Args:
        experiment: Experiment name
        method: Base calling method
        top_n: Number of top genes to display
    """
    print(f"--- Generating linked top {top_n} genes by cell count ---")
    dataset, pooled = _load_all_linked_for_experiment(experiment, method)
    if pooled is None or pooled.empty:
        print("No linked results found. Aborting linked top genes plot.")
        return

    gene_col = _find_col(pooled, _gene_col_candidates())
    if gene_col is None:
        print("No gene column found in linked results. Aborting top genes plot.")
        return

    valid = pooled[pooled[gene_col].notna()]

    # Separate NTCs if NCBI_ID is available
    if "NCBI_ID" in valid.columns:
        ntc = valid[valid["NCBI_ID"] == -1]
        targeting = valid[valid["NCBI_ID"] != -1]
    else:
        ntc = pd.DataFrame()
        targeting = valid

    if targeting.empty:
        print("No targeting gene assignments in linked results. Aborting top genes plot.")
        return

    cells_per_gene = targeting.groupby(gene_col).size()
    top_genes = cells_per_gene.nlargest(top_n).sort_values(ascending=False)

    if top_genes.empty:
        print("No genes with cell counts. Aborting top genes plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 18))
    ax.barh(range(len(top_genes)), top_genes.values, color="steelblue", alpha=0.8)
    ax.set_yticks(range(len(top_genes)))
    ax.set_yticklabels(top_genes.index.tolist(), fontsize=14)
    ax.set_ylabel("Gene", fontsize=20)
    ax.set_xlabel("Number of Cells", fontsize=20)
    ax.set_title(
        f"Top {top_n} Targeting Genes by Cell Count (Linked, All Wells Pooled)\n{dataset.experiment}",
        fontsize=22,
    )
    ax.grid(True, axis="x", linestyle="--", alpha=0.6)
    ax.invert_yaxis()
    ax.set_ylim(len(top_genes) - 0.5, -0.5)
    ax.tick_params(axis="x", labelsize=16)

    plt.tight_layout()
    save_path = dataset.metrics_paths["link_top_genes_by_cell_count"]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Saved linked top genes plot to {save_path}")
    plt.close()

    # NTC guides subplot (if available)
    if not ntc.empty:
        guide_col = _find_col(ntc, _GUIDE_COL_CANDIDATES)
        if guide_col is not None:
            cells_per_ntc = ntc.groupby(guide_col).size()
            ntc_sorted = cells_per_ntc.nlargest(top_n).sort_values(ascending=False)
            if not ntc_sorted.empty:
                # Build labels with gene name if available
                ntc_labels = []
                if gene_col in ntc.columns:
                    guide_to_gene = ntc.drop_duplicates(guide_col).set_index(guide_col)[gene_col].to_dict()
                    for g in ntc_sorted.index:
                        ntc_labels.append(f"{g}\n({guide_to_gene.get(g, 'Unknown')})")
                else:
                    ntc_labels = ntc_sorted.index.tolist()

                fig_height = max(8, len(ntc_sorted) * 0.4)
                fig_ntc, ax_ntc = plt.subplots(figsize=(12, fig_height))
                ax_ntc.barh(range(len(ntc_sorted)), ntc_sorted.values, color="orange", alpha=0.8)
                ax_ntc.set_yticks(range(len(ntc_sorted)))
                ax_ntc.set_yticklabels(ntc_labels, fontsize=12)
                ax_ntc.set_ylabel("NTC Guide (Gene Name)", fontsize=18)
                ax_ntc.set_xlabel("Number of Cells", fontsize=18)
                ax_ntc.set_title(
                    f"Top {top_n} NTC Guides by Cell Count (Linked, All Wells Pooled)\n{dataset.experiment}",
                    fontsize=20,
                )
                ax_ntc.grid(True, axis="x", linestyle="--", alpha=0.6)
                ax_ntc.invert_yaxis()
                ax_ntc.set_ylim(len(ntc_sorted) - 0.5, -0.5)
                ax_ntc.tick_params(axis="x", labelsize=14)
                plt.tight_layout()

                ntc_path = save_path.parent / "link_ntc_guides_by_cell_count.png"
                plt.savefig(ntc_path, dpi=300)
                print(f"Saved linked NTC guides plot to {ntc_path}")
                plt.close()


def plot_link_top_guides_by_cell_count(
    experiment: str,
    method: str = "mine",
    top_n: int = 50,
) -> None:
    """Bar plot of top N individual guides (sgRNAs) by cell count from linked CSVs.

    Mirrors :func:`iss_histrogram.plot_top_guides_by_cell_count`.

    Args:
        experiment: Experiment name
        method: Base calling method
        top_n: Number of top guides to display
    """
    print(f"--- Generating linked top {top_n} guides by cell count ---")
    dataset, pooled = _load_all_linked_for_experiment(experiment, method)
    if pooled is None or pooled.empty:
        print("No linked results found. Aborting linked top guides plot.")
        return

    guide_col = _find_col(pooled, _GUIDE_COL_CANDIDATES)
    if guide_col is None:
        print("No guide column found in linked results. Aborting top guides plot.")
        return

    valid = pooled[pooled[guide_col].notna()]
    if valid.empty:
        print("No valid guide assignments in linked results. Aborting top guides plot.")
        return

    cells_per_guide = valid.groupby(guide_col).size()
    top_guides = cells_per_guide.nlargest(top_n).sort_values(ascending=False)

    if top_guides.empty:
        print("No guides with cell counts. Aborting top guides plot.")
        return

    # Build labels with gene name if available
    gene_col = _find_col(valid, _gene_col_candidates())
    if gene_col is not None:
        guide_to_gene = valid.drop_duplicates(guide_col).set_index(guide_col)[gene_col].to_dict()
        guide_labels = [f"{g} ({guide_to_gene.get(g, 'Unknown')})" for g in top_guides.index]
    else:
        guide_labels = top_guides.index.tolist()

    fig, ax = plt.subplots(figsize=(12, 18))
    ax.barh(range(len(top_guides)), top_guides.values, color="teal", alpha=0.8)
    ax.set_yticks(range(len(top_guides)))
    ax.set_yticklabels(guide_labels, fontsize=14)
    ax.set_ylabel("Guide (Gene)", fontsize=20)
    ax.set_xlabel("Number of Cells", fontsize=20)
    ax.set_title(
        f"Top {top_n} Guides by Cell Count (Linked, All Wells Pooled)\n{dataset.experiment}",
        fontsize=22,
    )
    ax.grid(True, axis="x", linestyle="--", alpha=0.6)
    ax.invert_yaxis()
    ax.set_ylim(len(top_guides) - 0.5, -0.5)
    ax.tick_params(axis="x", labelsize=16)

    plt.tight_layout()
    save_path = dataset.metrics_paths["link_top_guides_by_cell_count"]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Saved linked top guides plot to {save_path}")
    plt.close()
