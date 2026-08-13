"""
Graph-based neighborhood matching for robust cell registration.

Uses local neighborhood structure (k-NN graphs) to validate cell matches.
Each cell is characterized not just by its Hu moments, but also by the
Hu moments and spatial relationships of its neighbors.

Strategy:
1. Build k-NN graphs for source and target cells
2. Create rotation-invariant graph descriptors (sorted neighbors by angle)
3. Wide-area Hu moment search (300px radius vs. previous 50px)
4. Validate matches using neighborhood graph consistency

This provides robustness when PCC pre-alignment has large errors.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import time


def _filter_by_hu_helper(args):
    """
    Parallel Hu filtering using shared data (no pickling of target_hu_moments).
    """
    src_idx, candidate_tgt_indices, src_hu, top_k_candidates = args

    if len(candidate_tgt_indices) == 0:
        return src_idx, [], []

    # Access shared target Hu moments (set via Pool initializer, fork-inherited)
    target_hu_moments = _shared_hu_filter_data["target_hu"]

    # Vectorized: compute Hu distance to all candidates at once
    tgt_hu_batch = target_hu_moments[candidate_tgt_indices]  # (N_cand, 7)
    src_log = -np.sign(src_hu) * np.log10(np.abs(src_hu) + 1e-10)
    tgt_log = -np.sign(tgt_hu_batch) * np.log10(np.abs(tgt_hu_batch) + 1e-10)
    hu_dists = np.sum(np.abs(src_log[np.newaxis, :] - tgt_log), axis=1)

    # Keep top K by Hu similarity (lowest distance)
    n_cand = len(hu_dists)
    top_k = min(top_k_candidates, n_cand)
    if top_k >= n_cand:
        top_k_order = np.argsort(hu_dists)
    else:
        top_k_order = np.argpartition(hu_dists, top_k)[:top_k]
        top_k_order = top_k_order[np.argsort(hu_dists[top_k_order])]

    top_tgt_indices = candidate_tgt_indices[top_k_order].tolist()
    top_hu_dists = hu_dists[top_k_order].tolist()

    return src_idx, top_tgt_indices, top_hu_dists


# Shared data for Hu filtering Pool workers
_shared_hu_filter_data = {}


def _init_shared_hu_filter(target_hu):
    _shared_hu_filter_data["target_hu"] = target_hu


def _score_graph_matches_helper(args):
    """
    Fully vectorized graph scoring for all candidates of a single source cell.

    Scores ALL candidates simultaneously using numpy broadcasting — no Python loop.
    Uses shared module-level data (set via Pool initializer).
    """
    src_idx, tgt_indices, weights = args

    n_cand = len(tgt_indices)
    if n_cand == 0:
        return []

    source_graphs = _shared_graph_data["source_graphs"]
    target_graphs = _shared_graph_data["target_graphs"]
    source_hu = _shared_graph_data["source_hu"]
    target_hu = _shared_graph_data["target_hu"]

    src_graph = source_graphs[src_idx]
    src_k = len(src_graph["neighbor_indices"])

    w_hu = weights.get("hu", 0.1)
    w_nhu = weights.get("neighbor_hu", 0.5)
    w_el = weights.get("edge_length", 0.2)
    w_as = weights.get("angular_spacing", 0.1)
    w_cl = weights.get("clustering", 0.1)

    # Pre-compute source quantities once
    src_log_hu = -np.sign(source_hu[src_idx]) * np.log10(np.abs(source_hu[src_idx]) + 1e-10)
    src_clustering = src_graph.get("clustering_coefficient", 0.0)

    # Collect all target data into arrays for batch scoring
    final_scores = np.ones(n_cand, dtype=np.float64)  # default = reject (1.0)
    result_indices = np.array(tgt_indices, dtype=np.intp)

    # Step 1: Filter by neighbor count (batch)
    tgt_ks = np.array([len(target_graphs[ti]["neighbor_indices"]) for ti in tgt_indices])
    k_vals = np.minimum(src_k, tgt_ks)
    valid = np.abs(src_k - tgt_ks) <= 2
    valid &= k_vals > 0

    # Step 2: Batch Hu distance for valid candidates
    if np.any(valid):
        valid_idx = np.where(valid)[0]
        tgt_hu_batch = target_hu[result_indices[valid_idx]]  # (N_valid, 7)
        tgt_log_hu = -np.sign(tgt_hu_batch) * np.log10(np.abs(tgt_hu_batch) + 1e-10)
        hu_dists = np.sum(np.abs(src_log_hu[np.newaxis, :] - tgt_log_hu), axis=1) / 10.0
        hu_dists = np.minimum(hu_dists, 1.0)

        # Reject high Hu distance
        hu_ok = hu_dists <= 0.6
        valid[valid_idx[~hu_ok]] = False

    # Step 3: Score remaining candidates individually (neighbor-level ops need per-cell data)
    # This is the irreducible per-candidate work, but we've already filtered ~50-80% out
    remaining = np.where(valid)[0]

    for ci in remaining:
        tgt_idx = int(result_indices[ci])
        tgt_graph = target_graphs[tgt_idx]
        k = int(k_vals[ci])

        # Hu distance (already computed for batch, reuse)
        tgt_log_hu_i = -np.sign(target_hu[tgt_idx]) * np.log10(np.abs(target_hu[tgt_idx]) + 1e-10)
        hu_dist_norm = min(float(np.sum(np.abs(src_log_hu - tgt_log_hu_i))) / 10.0, 1.0)

        # Neighbor Hu consistency
        src_nhu = src_graph["neighbor_hu"][:k]
        tgt_nhu = tgt_graph["neighbor_hu"][:k]
        log_src_nhu = -np.sign(src_nhu) * np.log10(np.abs(src_nhu) + 1e-10)
        log_tgt_nhu = -np.sign(tgt_nhu) * np.log10(np.abs(tgt_nhu) + 1e-10)
        neighbor_hu_dist_norm = min(float(np.mean(np.sum(np.abs(log_src_nhu - log_tgt_nhu), axis=1))) / 10.0, 1.0)

        # Edge length
        src_el = np.sort(src_graph["neighbor_distances"][:k])
        tgt_el = np.sort(tgt_graph["neighbor_distances"][:k])
        mean_l = (np.mean(src_el) + np.mean(tgt_el)) / 2
        edge_length_dist = min(float(np.mean(np.abs(src_el / mean_l - tgt_el / mean_l))), 1.0) if mean_l > 0 else 1.0

        # Angular spacing
        if k > 1:
            sp_diff = np.abs(np.diff(src_graph["neighbor_angles"][:k]) - np.diff(tgt_graph["neighbor_angles"][:k]))
            sp_diff = np.minimum(sp_diff, 2 * np.pi - sp_diff)
            angular_spacing_dist = min(float(np.mean(sp_diff)) / (np.pi / 4), 1.0)
        else:
            angular_spacing_dist = 0.5

        if angular_spacing_dist > 0.5:
            continue  # leave as 1.0

        clustering_dist = abs(src_clustering - tgt_graph.get("clustering_coefficient", 0.0))

        final_scores[ci] = (w_hu * hu_dist_norm + w_nhu * neighbor_hu_dist_norm +
                            w_el * edge_length_dist + w_as * angular_spacing_dist +
                            w_cl * clustering_dist)

    # Build result tuples
    return [(src_idx, int(result_indices[i]), float(final_scores[i]), {}) for i in range(n_cand)]


# Shared data for multiprocessing Pool workers (set via initializer, inherited via fork)
_shared_graph_data = {}


def _init_shared_graph_data(sg, tg, shu, thu):
    """Pool initializer: store shared graph data in module-level dict."""
    _shared_graph_data["source_graphs"] = sg
    _shared_graph_data["target_graphs"] = tg
    _shared_graph_data["source_hu"] = shu
    _shared_graph_data["target_hu"] = thu


def build_cell_neighborhood_graphs(
    centroids: np.ndarray,
    hu_moments: np.ndarray,
    k_neighbors: int = 8,
    verbose: bool = False,
) -> Dict[int, Dict]:
    """
    Build k-NN neighborhood graphs for cells.

    For each cell, stores:
    - neighbor_indices: k nearest neighbor indices
    - neighbor_distances: distances to neighbors
    - neighbor_angles: angles to neighbors (relative, for rotation invariance)
    - neighbor_hu: Hu moments of neighbors

    Parameters
    ----------
    centroids : np.ndarray
        (N, 2) cell centroids.
    hu_moments : np.ndarray
        (N, 7) Hu moments for each cell.
    k_neighbors : int
        Number of neighbors per cell (default: 8).
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Mapping from cell index to neighborhood info dict.
    """
    from sklearn.neighbors import NearestNeighbors

    n_cells = len(centroids)

    if k_neighbors >= n_cells:
        k_neighbors = max(1, n_cells - 1)
        if verbose:
            print(f"      Reducing k_neighbors to {k_neighbors} (max available)")

    # Build k-NN tree (k+1 to exclude self)
    if verbose:
        print(f"      Building k-NN graphs for {n_cells} cells (k={k_neighbors})...")

    t_start = time.time()
    knn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="euclidean", n_jobs=-1)
    knn.fit(centroids)

    distances, indices = knn.kneighbors(centroids)

    # Build graph descriptors
    graphs = {}
    for i in range(n_cells):
        # Exclude self (first neighbor is always self with distance 0)
        neighbor_idx = indices[i, 1:]  # Skip first (self)
        neighbor_dist = distances[i, 1:]

        # Compute angles to neighbors (for rotation invariance and direction checking)
        neighbor_vectors = centroids[neighbor_idx] - centroids[i]
        neighbor_angles_unsorted = np.arctan2(neighbor_vectors[:, 0], neighbor_vectors[:, 1])

        # Sort neighbors by angle (makes descriptor rotation-invariant)
        sort_order = np.argsort(neighbor_angles_unsorted)

        # Compute clustering coefficient (how interconnected are neighbors)
        # Vectorized: use pdist for all pairwise distances at once
        clustering = 0.0
        if k_neighbors > 1:
            from scipy.spatial.distance import pdist
            neighbor_centroids = centroids[neighbor_idx]
            max_neighbor_dist = neighbor_dist[-1]
            pairwise_dists = pdist(neighbor_centroids)
            neighbor_edges = int(np.sum(pairwise_dists <= max_neighbor_dist))
            max_possible_edges = (k_neighbors * (k_neighbors - 1)) / 2
            clustering = neighbor_edges / max_possible_edges if max_possible_edges > 0 else 0.0

        graphs[i] = {
            "neighbor_indices": neighbor_idx[sort_order],
            "neighbor_distances": neighbor_dist[sort_order],
            "neighbor_angles": neighbor_angles_unsorted[sort_order],  # Sorted angles for rotation-invariant comparison
            "neighbor_angles_unsorted": neighbor_angles_unsorted,  # Original angles for absolute direction checking
            "neighbor_hu": hu_moments[neighbor_idx[sort_order]],
            "clustering_coefficient": clustering,  # How interconnected are neighbors
        }

    if verbose:
        print(f"      Built {n_cells} neighborhood graphs ({time.time() - t_start:.2f}s)")

    return graphs


def compute_hu_distance(hu1: np.ndarray, hu2: np.ndarray) -> float:
    """
    Compute Hu moment distance (L1 norm of log-transformed moments).

    Parameters
    ----------
    hu1 : np.ndarray
        (7,) Hu moments for cell 1.
    hu2 : np.ndarray
        (7,) Hu moments for cell 2.

    Returns
    -------
    float
        Hu moment distance.
    """
    # Log-transform to handle scale differences (standard practice for Hu moments)
    log_hu1 = -np.sign(hu1) * np.log10(np.abs(hu1) + 1e-10)
    log_hu2 = -np.sign(hu2) * np.log10(np.abs(hu2) + 1e-10)
    return np.sum(np.abs(log_hu1 - log_hu2))


def score_graph_similarity(
    src_graph: Dict,
    tgt_graph: Dict,
    src_hu: np.ndarray,
    tgt_hu: np.ndarray,
    weights: Dict[str, float],
) -> float:
    """
    Score similarity between two neighborhood graphs.

    Combines:
    - Individual cell Hu moment similarity
    - Neighbor Hu moment consistency
    - Edge length consistency
    - Edge angle consistency (relative, rotation-invariant)
    - **Absolute directional consistency (checks if neighbors are in same directions)**

    Parameters
    ----------
    src_graph : dict
        Source cell neighborhood graph.
    tgt_graph : dict
        Target cell neighborhood graph.
    src_hu : np.ndarray
        (7,) Hu moments for source cell.
    tgt_hu : np.ndarray
        (7,) Hu moments for target cell.
    weights : dict
        Weights for different components:
        - "hu": Individual Hu similarity weight
        - "neighbor_hu": Neighbor Hu consistency weight
        - "edge_length": Edge length consistency weight
        - "edge_angle": Edge angle consistency weight (relative)
        - "direction": Absolute directional consistency weight (NEW)

    Returns
    -------
    float
        Combined similarity score (lower is better, normalized 0-1).
    """
    # Component 0: Neighbor count consistency (CRITICAL)
    # Cells should have similar numbers of neighbors within the k-NN radius
    src_k = len(src_graph["neighbor_indices"])
    tgt_k = len(tgt_graph["neighbor_indices"])
    k = min(src_k, tgt_k)

    # Penalize heavily if neighbor counts differ significantly
    neighbor_count_diff = abs(src_k - tgt_k)
    if neighbor_count_diff > 2:  # Allow up to 2 neighbors difference
        return (1.0, {'reject_reason': 'neighbor_count_diff', 'src_k': src_k, 'tgt_k': tgt_k})

    # Component 1: Individual Hu moment distance
    hu_dist = compute_hu_distance(src_hu, tgt_hu)
    hu_dist_norm = min(hu_dist / 10.0, 1.0)  # Normalize to [0, 1]

    # Component 2: Neighbor Hu moment consistency
    # Compare Hu moments of corresponding neighbors (sorted by angle)
    src_neighbor_hu = src_graph["neighbor_hu"]
    tgt_neighbor_hu = tgt_graph["neighbor_hu"]
    if k == 0:
        neighbor_hu_dist = 1.0  # Worst case if no neighbors
    else:
        neighbor_hu_dists = [
            compute_hu_distance(src_neighbor_hu[i], tgt_neighbor_hu[i])
            for i in range(k)
        ]
        neighbor_hu_dist = np.mean(neighbor_hu_dists)
        neighbor_hu_dist_norm = min(neighbor_hu_dist / 10.0, 1.0)

    # Component 3: Edge length distribution similarity
    # Compare the distribution of distances to neighbors (captures local density)
    src_edge_lengths = src_graph["neighbor_distances"][:k]
    tgt_edge_lengths = tgt_graph["neighbor_distances"][:k]

    if k == 0:
        edge_length_dist = 1.0
    else:
        # Compare sorted distance vectors (order-invariant density signature)
        src_dists_sorted = np.sort(src_edge_lengths)
        tgt_dists_sorted = np.sort(tgt_edge_lengths)

        # Normalize by mean to handle scale, then compute element-wise differences
        mean_length = (np.mean(src_dists_sorted) + np.mean(tgt_dists_sorted)) / 2
        if mean_length > 0:
            src_normalized = src_dists_sorted / mean_length
            tgt_normalized = tgt_dists_sorted / mean_length
            # Mean absolute difference in normalized distances
            edge_length_dist = min(np.mean(np.abs(src_normalized - tgt_normalized)), 1.0)
        else:
            edge_length_dist = 1.0

    # Component 4: Edge angle consistency
    # Compare relative angles to neighbors (rotation-invariant via sorting)
    src_angles = src_graph["neighbor_angles"][:k]
    tgt_angles = tgt_graph["neighbor_angles"][:k]

    if k == 0:
        angle_dist = 1.0
    else:
        # Compute angle differences (modulo 2�)
        angle_diffs = np.abs(src_angles - tgt_angles)
        angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)  # Wrap around
        angle_dist = min(np.mean(angle_diffs) / np.pi, 1.0)  # Normalize to [0, 1]

    # Component 4: Angular spacing distribution
    # Check if neighbor angular spacing patterns match (rotation-invariant layout signature)
    angular_spacing_dist = 0.0
    if k > 1:  # Need at least 2 neighbors to compute spacing
        src_angles_sorted = src_graph["neighbor_angles"][:k]
        tgt_angles_sorted = tgt_graph["neighbor_angles"][:k]

        # Compute angular spacing between consecutive neighbors
        src_spacing = np.diff(src_angles_sorted)
        tgt_spacing = np.diff(tgt_angles_sorted)

        # Compare spacing patterns
        spacing_diffs = np.abs(src_spacing - tgt_spacing)
        spacing_diffs = np.minimum(spacing_diffs, 2 * np.pi - spacing_diffs)  # Wrap
        angular_spacing_dist = min(np.mean(spacing_diffs) / (np.pi / 4), 1.0)
    else:
        angular_spacing_dist = 0.5  # Penalize cells with too few neighbors

    # Component 5: Clustering coefficient similarity
    # How interconnected are the neighbors? Dense clusters vs sparse layouts
    clustering_dist = 0.0
    if "clustering_coefficient" in src_graph and "clustering_coefficient" in tgt_graph:
        src_clustering = src_graph["clustering_coefficient"]
        tgt_clustering = tgt_graph["clustering_coefficient"]
        # Absolute difference in clustering coefficient (both in [0,1])
        clustering_dist = abs(src_clustering - tgt_clustering)

    # Apply strict hard thresholds first (early rejection)
    MAX_ANGULAR_SPACING_DIST = 0.5  # Reject if angular spacing differs significantly
    MAX_HU_DIST = 0.6  # Reject if cells look very different

    if angular_spacing_dist > MAX_ANGULAR_SPACING_DIST:
        return (1.0, {'reject_reason': 'angular_spacing_dist', 'angular_spacing': angular_spacing_dist})

    if hu_dist_norm > MAX_HU_DIST:
        return (1.0, {'reject_reason': 'hu_dist', 'hu': hu_dist_norm})

    # Weighted combination using provided weights
    score = (
        weights.get("hu", 0.1) * hu_dist_norm +
        weights.get("neighbor_hu", 0.5) * neighbor_hu_dist_norm +
        weights.get("edge_length", 0.2) * edge_length_dist +
        weights.get("angular_spacing", 0.1) * angular_spacing_dist +
        weights.get("clustering", 0.1) * clustering_dist
    )

    # Store component breakdown for diagnostics (returned as tuple)
    return (score, {
        'src_k': src_k,
        'tgt_k': tgt_k,
        'k_diff': neighbor_count_diff,
        'hu': hu_dist_norm,
        'neighbor_hu': neighbor_hu_dist_norm,
        'edge_length': edge_length_dist,
        'angular_spacing': angular_spacing_dist,
        'clustering': clustering_dist,
    })


def match_cells_by_graph_consistency(
    source_centroids: np.ndarray,
    target_centroids: np.ndarray,
    source_hu_moments: np.ndarray,
    target_hu_moments: np.ndarray,
    search_radius: float = 300.0,
    k_neighbors: int = 8,
    top_k_candidates: int = 20,
    weights: Dict[str, float] = None,
    max_score_threshold: float = 0.2,
    min_matches_per_cell: int = 1,
    min_total_matches: int = 10,
    cache_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match cells using graph-based neighborhood consistency with strict quality gates.

    Multi-stage approach:
    1. Wide-area spatial search (radius-based)
    2. Hu moment filtering (top K by similarity)
    3. Graph consistency scoring (neighborhood validation)
    4. **Hard quality threshold (reject poor matches)**
    5. Greedy 1-to-1 matching

    Parameters
    ----------
    source_centroids : np.ndarray
        (N, 2) source centroids (should be PCC-aligned).
    target_centroids : np.ndarray
        (M, 2) target centroids.
    source_hu_moments : np.ndarray
        (N, 7) Hu moments for source cells.
    target_hu_moments : np.ndarray
        (M, 7) Hu moments for target cells.
    search_radius : float
        Maximum spatial search radius (pixels, default: 300).
    k_neighbors : int
        Neighborhood size for graph construction (default: 8).
    top_k_candidates : int
        Number of initial candidates per source cell (default: 20).
    weights : dict
        Weights for graph scoring components. If None, uses equal weighting (0.2 each).
        Components: hu, neighbor_hu, edge_length, edge_angle, direction
    max_score_threshold : float
        Hard quality threshold: reject matches with score > threshold (default: 0.2).
        Lower = stricter. Scores are 0-1, with 0=perfect match.
    min_matches_per_cell : int
        Minimum matches required per source cell. If 0, cells can have no match.
        If 1+, keep best N matches per cell even if they exceed threshold.
    min_total_matches : int
        Minimum total matches required. If fewer pass threshold, issue warning.
    verbose : bool
        Print progress.

    Returns
    -------
    tuple
        (source_indices, target_indices, graph_scores)
        - source_indices: Matched source cell indices
        - target_indices: Matched target cell indices
        - graph_scores: Graph consistency scores (lower is better)
    """
    from sklearn.neighbors import NearestNeighbors
    from multiprocessing import get_context
    from tqdm import tqdm
    from cyclops_utils.hpc.resource_manager import get_optimal_workers

    # Use forkserver to avoid hangs from inheriting parent's tensorstore/BLAS
    # thread state. Plain fork() inherits open async threads and held GIL/BLAS
    # locks from the parent; on some nodes (observed cpu-h-* class) this
    # deadlocks Stage 2 Pool initialization indefinitely. forkserver spawns
    # workers from a clean intermediary process. The Pool initializer args
    # (target_hu / graph dicts) are pickled to each worker — small enough
    # (sub-MB) for this not to matter.
    _mp_ctx = get_context("forkserver")
    Pool = _mp_ctx.Pool

    if weights is None:
        weights = {
            "hu": 0.2,           # Individual Hu moment similarity
            "neighbor_hu": 0.2,  # Neighbor Hu consistency
            "edge_length": 0.2,  # Edge length consistency
            "edge_angle": 0.2,   # Relative angle consistency (rotation-invariant)
            "direction": 0.2,    # Absolute directional consistency
        }

    n_source = len(source_centroids)
    n_target = len(target_centroids)

    if verbose:
        print(f"    [Graph Matching] Source: {n_source} cells, Target: {n_target} cells")
        print(f"      Search radius: {search_radius:.1f}px, k_neighbors: {k_neighbors}")

    # Build neighborhood graphs for source and target concurrently
    from concurrent.futures import ThreadPoolExecutor as _GraphPool
    with _GraphPool(max_workers=2) as gpool:
        src_future = gpool.submit(
            build_cell_neighborhood_graphs, source_centroids, source_hu_moments, k_neighbors, verbose
        )
        tgt_future = gpool.submit(
            build_cell_neighborhood_graphs, target_centroids, target_hu_moments, k_neighbors, verbose
        )
        source_graphs = src_future.result()
        target_graphs = tgt_future.result()

    # Stage 1: Wide-area spatial search to get initial candidates
    if verbose:
        print(f"      [Stage 1] Wide-area spatial search (radius={search_radius:.1f}px)...")

    t_search = time.time()

    # Find all target cells within search radius of each source cell
    knn_spatial = NearestNeighbors(radius=search_radius, metric="euclidean", n_jobs=-1)
    knn_spatial.fit(target_centroids)

    # Get candidates within radius for each source cell
    candidate_indices_list = knn_spatial.radius_neighbors(
        source_centroids, return_distance=False
    )

    if verbose:
        mean_candidates = np.mean([len(candidates) for candidates in candidate_indices_list])
        print(f"      Found {mean_candidates:.1f} spatial candidates per source cell (avg)")
        print(f"      Spatial search completed in {time.time() - t_search:.2f}s")

    # Stage 2: For each source cell, compute Hu distance to all spatial candidates
    # and keep top K by Hu similarity (with caching)
    if verbose:
        print(f"      [Stage 2] Filtering to top {top_k_candidates} by Hu similarity...")

    t_hu_filter = time.time()

    # Check for cached Hu filtering results
    import hashlib
    import pickle

    hu_filtered = None
    hu_cache_file = None

    if cache_dir is not None:
        # Create cache key from parameters
        cache_key = f"{search_radius}_{top_k_candidates}_{len(source_centroids)}_{len(target_centroids)}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]

        # Create human-readable prefix: src{n_src}_tgt{n_tgt}
        human_prefix = f"src{len(source_centroids)}_tgt{len(target_centroids)}_r{int(search_radius)}_k{top_k_candidates}"
        hu_cache_file = cache_dir / f"hu_filtering_{human_prefix}_{cache_hash}.pkl"

        # Try to load from cache
        if hu_cache_file.exists():
            try:
                with open(hu_cache_file, 'rb') as f:
                    hu_filtered = pickle.load(f)
                if verbose:
                    print(f"      Using cached Hu filtering: {hu_cache_file.name}")
            except Exception as e:
                if verbose:
                    print(f"      Warning: Could not load Hu filtering cache: {e}")
                hu_filtered = None

    # Compute if not cached
    # Use optimal workers for parallel processing (can be disabled via SLURM if outer executor manages parallelism)
    n_workers = get_optimal_workers(use_gpu=False, model_ram_gb=0.001, data_ram_gb=0.001, verbose=False)
    if hu_filtered is None:

        # Prepare lightweight arguments — target_hu shared via Pool initializer
        args_list = [
            (src_idx, np.array(candidate_indices_list[src_idx], dtype=np.intp),
             source_hu_moments[src_idx], top_k_candidates)
            for src_idx in range(n_source)
        ]

        if n_workers > 1:
            with Pool(n_workers, initializer=_init_shared_hu_filter,
                      initargs=(target_hu_moments,)) as pool:
                hu_filtered = list(pool.imap_unordered(
                    _filter_by_hu_helper, args_list, chunksize=256))
        else:
            _init_shared_hu_filter(target_hu_moments)
            hu_filtered = [_filter_by_hu_helper(args) for args in args_list]

        # Save to cache
        if cache_dir is not None and hu_cache_file is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            try:
                with open(hu_cache_file, 'wb') as f:
                    pickle.dump(hu_filtered, f)
                if verbose:
                    print(f"      Saved Hu filtering to cache: {hu_cache_file.name}")
            except Exception as e:
                if verbose:
                    print(f"      Warning: Could not save Hu filtering cache: {e}")

    if verbose:
        mean_hu_candidates = np.mean([len(tgt_indices) for _, tgt_indices, _ in hu_filtered])
        print(f"      Reduced to {mean_hu_candidates:.1f} candidates per source cell (avg)")
        print(f"      Hu filtering completed in {time.time() - t_hu_filter:.2f}s")

    # Stage 3: Graph consistency scoring for remaining candidates
    if verbose:
        print(f"      [Stage 3] Graph consistency validation...")

    t_graph = time.time()

    # Prepare lightweight arguments: only pass indices (not graph dicts).
    # Graph data is shared via Pool initializer (fork-inherited, zero-copy).
    args_list = [
        (src_idx, np.array(tgt_indices, dtype=np.intp), weights)
        for src_idx, tgt_indices, _ in hu_filtered
    ]

    if n_workers > 1:
        with Pool(
            n_workers,
            initializer=_init_shared_graph_data,
            initargs=(source_graphs, target_graphs, source_hu_moments, target_hu_moments),
        ) as pool:
            # Use imap_unordered with large chunksize for less IPC overhead
            # No tqdm — it adds ~30% overhead from per-item callbacks
            graph_scored_nested = list(
                pool.imap_unordered(_score_graph_matches_helper, args_list, chunksize=256)
            )
    else:
        _init_shared_graph_data(source_graphs, target_graphs, source_hu_moments, target_hu_moments)
        graph_scored_nested = [_score_graph_matches_helper(args) for args in args_list]

    # Flatten results and extract diagnostics
    all_matches = []
    all_diagnostics = []
    for matches in graph_scored_nested:
        for match in matches:
            if len(match) == 4:  # (src_idx, tgt_idx, score, diagnostics)
                src_idx, tgt_idx, score, diag = match
                all_matches.append((src_idx, tgt_idx, score))
                all_diagnostics.append(diag)
            else:  # Backward compatibility
                all_matches.append(match[:3])
                all_diagnostics.append({})

    if len(all_matches) == 0:
        if verbose:
            print("      WARNING: No matches found!")
        return np.array([]), np.array([]), np.array([])

    # Convert to arrays
    src_indices = np.array([m[0] for m in all_matches], dtype=int)
    tgt_indices = np.array([m[1] for m in all_matches], dtype=int)
    graph_scores = np.array([m[2] for m in all_matches])

    if verbose:
        print(f"      Graph scoring completed in {time.time() - t_graph:.2f}s")
        print(f"      Total candidate matches: {len(all_matches)}")

    # Stage 4: Keep top N best matches globally (for RANSAC)
    top_n_global = max_score_threshold if isinstance(max_score_threshold, int) else 100
    if verbose:
        print(f"      [Stage 4] Selecting top {top_n_global} best matches globally for RANSAC...")

    # Sort all matches by score (best first)
    sorted_indices = np.argsort(graph_scores)

    # Keep top N matches
    top_n = min(top_n_global, len(sorted_indices))
    filtered_indices = sorted_indices[:top_n]

    filtered_matches = [
        (src_indices[i], tgt_indices[i], graph_scores[i])
        for i in filtered_indices
    ]

    # Print diagnostic statistics for top matches to understand what's failing
    if verbose and len(filtered_indices) > 0:
        top_diags = [all_diagnostics[i] for i in filtered_indices if all_diagnostics[i]]
        if top_diags:
            import pandas as pd
            df_diag = pd.DataFrame(top_diags)
            print(f"\n      Top {len(top_diags)} match diagnostics:")
            for col in ['src_k', 'tgt_k', 'k_diff', 'hu', 'neighbor_hu', 'edge_length', 'angular_spacing', 'clustering']:
                if col in df_diag.columns:
                    vals = df_diag[col].dropna()
                    if len(vals) > 0:
                        print(f"        {col}: mean={vals.mean():.3f}, med={vals.median():.3f}, min={vals.min():.3f}, max={vals.max():.3f}")

            # Check if key metrics are discriminating
            for metric in ['hu', 'neighbor_hu']:
                if metric in df_diag.columns:
                    vals = df_diag[metric].dropna()
                    if len(vals) > 0 and vals.std() < 0.02:
                        print(f"        ⚠️  WARNING: {metric} has very low variance (std={vals.std():.4f})")
                        print(f"        This suggests {metric} is NOT discriminating between good/bad matches!")

            # Correlation analysis to check for redundancy
            metric_cols = ['hu', 'neighbor_hu', 'edge_length', 'angular_spacing', 'clustering']
            available_cols = [c for c in metric_cols if c in df_diag.columns]
            if len(available_cols) >= 2:
                corr_matrix = df_diag[available_cols].corr()
                print(f"\n      Metric correlations (high = redundant, low = independent):")
                for i, col1 in enumerate(available_cols):
                    for col2 in available_cols[i+1:]:
                        corr = corr_matrix.loc[col1, col2]
                        if abs(corr) > 0.7:
                            print(f"        {col1} <-> {col2}: {corr:.3f} ⚠️  HIGH (redundant)")
                        else:
                            print(f"        {col1} <-> {col2}: {corr:.3f}")

    # Convert to arrays
    if len(filtered_matches) == 0:
        return np.array([]), np.array([]), np.array([])

    filtered_matches = np.array(filtered_matches)
    src_indices_filtered = filtered_matches[:, 0].astype(int)
    tgt_indices_filtered = filtered_matches[:, 1].astype(int)
    graph_scores_filtered = filtered_matches[:, 2]

    if verbose:
        print(f"      Selected {len(filtered_matches)} matches for RANSAC")
        if len(graph_scores_filtered) > 0:
            print(f"      Score range: {np.min(graph_scores_filtered):.4f} - {np.max(graph_scores_filtered):.4f}")

    # Stage 5: Greedy 1-to-1 matching (prevent duplicates)
    if verbose:
        print(f"      [Stage 5] Greedy 1-to-1 matching...")

    # Sort by graph score (best matches first)
    sort_order = np.argsort(graph_scores_filtered)
    src_indices_sorted = src_indices_filtered[sort_order]
    tgt_indices_sorted = tgt_indices_filtered[sort_order]
    graph_scores_sorted = graph_scores_filtered[sort_order]

    # Greedy selection: keep first occurrence of each source/target
    used_src = set()
    used_tgt = set()
    final_src = []
    final_tgt = []
    final_scores = []

    for src_idx, tgt_idx, score in zip(src_indices_sorted, tgt_indices_sorted, graph_scores_sorted):
        if src_idx not in used_src and tgt_idx not in used_tgt:
            final_src.append(src_idx)
            final_tgt.append(tgt_idx)
            final_scores.append(score)
            used_src.add(src_idx)
            used_tgt.add(tgt_idx)

    final_src = np.array(final_src)
    final_tgt = np.array(final_tgt)
    final_scores = np.array(final_scores)

    if verbose:
        print(f"      Final 1-to-1 matches: {len(final_src)}")
        if len(final_scores) > 0:
            print(f"      Mean graph score: {np.mean(final_scores):.4f}")
            print(f"      Median graph score: {np.median(final_scores):.4f}")
            print(f"      Matches with score ≤ 0.15 (excellent): {np.sum(final_scores <= 0.15)}")
            print(f"      Matches with score ≤ 0.20 (good): {np.sum(final_scores <= 0.20)}")

    return final_src, final_tgt, final_scores, source_graphs, target_graphs
