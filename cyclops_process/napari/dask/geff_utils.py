"""Utility functions for loading and rendering cell tracking data from geff format."""

from pathlib import Path
import numpy as np

from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.data.filesystem import vprintf


def load_geff_tracks(geff_path: Path, shape: tuple = (1, 1, 1)):
    """
    Load tracks from a geff Zarr store using tracksdata.

    Args:
        geff_path: Path to the geff directory (Zarr store)
        shape: Shape tuple for the image data (not used for tracks-only, but required by API)

    Returns:
        tuple of (tracks_df, graph, labels):
            - tracks_df: DataFrame in napari tracks format
            - graph: dict mapping child track IDs to parent IDs for lineage visualization
            - labels: label image (None if mask_key is None)
    """
    import tracksdata as td

    # Only load the attrs needed for napari tracks — skip heavy columns
    # (bbox, mask, inertia_tensor, etc.) to cut load time.
    # Progressively drop optional props if they don't exist in older geffs.
    _node_props_sets = [
        [td.DEFAULT_ATTR_KEYS.T, "y", "x", "z", "tracklet_id", "solution"],
        [td.DEFAULT_ATTR_KEYS.T, "y", "x", "z", "solution"],
        [td.DEFAULT_ATTR_KEYS.T, "y", "x", "z"],
    ]

    # Load geff file using tracksdata
    # from_geff may return (graph, metadata) tuple or just a graph
    result_geff = None
    for props in _node_props_sets:
        try:
            result_geff = td.graph.InMemoryGraph().from_geff(
                str(geff_path), geff_read_kwargs={"node_props": props}
            )
            break
        except Exception:
            continue
    if result_geff is None:
        result_geff = td.graph.InMemoryGraph().from_geff(str(geff_path))
    if isinstance(result_geff, tuple):
        graph_obj = result_geff[0]
    else:
        graph_obj = result_geff

    vprintf("[geff-track] Loaded graph from geff")

    # Convert to napari format
    # solution_key=None means include all tracks (not just solution)
    # mask_key=None means don't generate label images
    result = td.functional.to_napari_format(
        graph_obj,
        shape,
        solution_key="solution",  # Filter by solution flag
        mask_key=None,  # Don't generate label images (tracks only)
    )

    # Handle different return formats (may return 2 or 3 values)
    if len(result) == 2:
        tracks_df, napari_graph = result
        labels = None
    else:
        tracks_df, napari_graph, labels = result

    vprintf("[geff-track] Converted to %d track points", len(tracks_df))

    return tracks_df, napari_graph, labels


def render_tracks_from_geff(
    experiment: str,
    pos: str,
    offsets_x: dict,
    v,
    shape: tuple = None,
):
    """
    Load and add raw tracks from a geff Zarr store for a given position.

    The tracks layer is added WITHOUT any affine transform so that the calling
    code in view_dask.py can apply the same transform pipeline used for image
    layers (scale, registration, well separation).

    Args:
        experiment: Experiment name
        pos: Position/well identifier (e.g., 'A/1/0')
        offsets_x: Dictionary of horizontal offsets per position (used for translate)
        v: Napari viewer instance
        shape: Optional shape tuple for mask rendering (unused, kept for compatibility)

    Returns:
        The napari Tracks layer, or None if tracks could not be loaded.
    """
    ds = OpsDataset(experiment)

    # Get geff directory path using append_well (handles well token conversion)
    try:
        geff_path = Path(ds.append_well("tracking_geff", pos))
        vprintf("[geff-track] Resolved geff path for %s: %s", pos, str(geff_path))
    except Exception as e:
        vprintf("[geff-track] Could not resolve geff path for %s: %s", pos, str(e))
        vprintf("[geff-track] Dataset experiment_path: %s", ds.experiment_path)
        vprintf(
            "[geff-track] Available store_paths keys: %s",
            list(ds.store_paths.keys()) if hasattr(ds, "store_paths") else "N/A",
        )
        return None

    if not geff_path.exists():
        vprintf("[geff-track] No geff file found for %s at %s", pos, str(geff_path))
        return None

    try:
        vprintf("[geff-track] Loading geff from %s", str(geff_path))

        # Load tracks data and lineage graph using tracksdata
        # If shape is None, use a dummy shape (tracks don't need actual image dimensions)
        if shape is None:
            shape = (1, 1, 1, 1)  # (t, z, y, x) dummy shape

        result = load_geff_tracks(geff_path, shape=shape)
        if len(result) == 2:
            tracks_df, napari_graph = result
            labels = None
        else:
            tracks_df, napari_graph, labels = result

        if len(tracks_df) == 0:
            vprintf("[geff-track] No track data to display for %s", pos)
            return None

        # Convert polars DataFrame to numpy for napari
        tracks_array = tracks_df.to_numpy()

        # Tracks are in format: [track_id, t, z, y, x] - ndim = 4 (TZYX)
        ndim = 4

        # Extract time values for each track point to use for coloring
        time_values = tracks_array[:, 1].copy()  # Column 1 is time

        # Apply horizontal offset for multi-well display via translate
        offset_x = float(offsets_x.get(pos, 0))
        translate_tuple = (0.0, 0.0, 0.0, offset_x)  # T, Z, Y, X format

        vprintf(
            "[geff-track] Adding %d raw track points for %s (no transform)",
            len(tracks_array),
            pos,
        )

        # Add tracks layer with NO affine — the caller applies transforms
        tracks_layer = v.add_tracks(
            tracks_array,
            graph=napari_graph if napari_graph else None,
            name=f"tracks:{pos}",
            properties={"time": time_values},
            translate=translate_tuple,
            visible=True,
            tail_width=2,
            tail_length=10,
            head_length=0,
            colormap="viridis",
            color_by="time",
        )

        vprintf("[geff-track] Successfully added tracks layer for %s", pos)
        return tracks_layer

    except Exception as e:
        vprintf("[geff-track] Failed to load tracks for %s: %s", pos, str(e))
        import traceback

        traceback.print_exc()
        return None
