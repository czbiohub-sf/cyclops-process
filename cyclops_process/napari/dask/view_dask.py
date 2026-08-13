from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Iterable

import numpy as np
import pandas as pd
import dask.array as da

import sys
import os

sys.path.insert(0, os.getcwd())

from stitch.registration.register import read_transform_biahub

from iohub import open_ome_zarr
from cyclops_utils.data.experiment import OpsDataset

from tqdm import tqdm
from cyclops_utils.io.zarr_utils import (
    _iter_position_paths,
    _infer_channel_axis_from_store,
    _read_component_attrs,
    read_per_level_clims,
)
from cyclops_process.napari.dask.dask_utils import (
    _list_numeric_level_names,
    _load_total_translations,
    _prune_extra_levels,
    _candidate_pos_prefixes,
    _resolve_stitch_config_path,
    _apply_clims_to_layer,
    _apply_clims_on_zoom,
)
from cyclops_utils.data.filesystem import (
    vprintf,
    resolve_experiment_name,
    canonicalize_well_path,
    extract_ops_key,
)
from cyclops_process.processes.pyramids.build_dask import (
    build_pyramid_in_place,
    build_seg_pyramid_only,
    build_grid_overlay_in_place,
    build_iss_overlay_in_place,
    build_clims_in_place,
)
from cyclops_process.napari.dask.geff_utils import render_tracks_from_geff

# Print only one clims table per run
CLIMS_REPORT_PRINTED: bool = False

from cyclops_utils.data import filesystem as _fs
from cyclops_process.paths import BASE_PATH


_OPS_CHANNEL_MAP_CACHE: dict[str, dict[str, str]] | None = None


def _load_ops_channel_map_yaml() -> dict[str, dict[str, str]]:
    """Parse ops_channel_maps.yaml into {ops_prefix: {channel_name: label}}."""
    global _OPS_CHANNEL_MAP_CACHE
    if _OPS_CHANNEL_MAP_CACHE is not None:
        return _OPS_CHANNEL_MAP_CACHE
    yaml_path = None
    try:
        yaml_path = Path(OpsDataset(resolve_experiment_name("ops0107_20251208")).channel_maps)
    except Exception:
        yaml_path = Path(f"{BASE_PATH}/configs/ops_channel_maps.yaml")
    out: dict[str, dict[str, str]] = {}
    try:
        import yaml as _yaml
        with open(yaml_path) as f:
            data = _yaml.safe_load(f) or {}
        for prefix, entries in data.items():
            if not isinstance(entries, list):
                continue
            cm: dict[str, str] = {}
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                ch = entry.get("channel_name")
                lbl = entry.get("label")
                if ch and lbl:
                    cm[str(ch)] = str(lbl)
            if cm:
                out[str(prefix)] = cm
    except Exception:
        out = {}
    _OPS_CHANNEL_MAP_CACHE = out
    return out


def _marker_suffix_for_label(label: str | None) -> str | None:
    """Short marker tag from a channel label (e.g. 'FastAct_SPY555' from
    'actin filament, FastAct_SPY555 Live Cell Dye'). Truncated to 15 chars.
    Returns None for Phase / empty markers."""
    if not label:
        return None
    marker = label.split(",", 1)[1].strip() if "," in label else label.strip()
    marker = marker[:15].strip()
    if not marker or marker.lower() == "phase":
        return None
    return marker


def _well_token(pos_str: str) -> str:
    """Derive well token like 'A1' from 'A/1/0'."""
    try:
        parts = [p for p in str(pos_str).split("/") if p]
        row = parts[0]
        col = parts[1] if len(parts) > 1 else ""
        return f"{row}{col}"
    except Exception:
        return str(pos_str).replace("/", "")


def _find_per_well_yaml(ds: OpsDataset, config_key: str, pos: str) -> Path | None:
    """Look up per-well registration YAML, falling back to base YAML.

    Tries candidates in order:
        {tok}_{base_name}, {tok}_{base_stem}.yml, {tok}_{base_stem}.yaml, base_path

    Returns the first existing path, or None if nothing found.
    """
    yaml_p = ds.config_paths.get(config_key)
    if yaml_p is None:
        return None

    base_path = Path(yaml_p)
    yaml_dir = base_path.parent
    base_stem = base_path.stem
    tok = _well_token(pos)

    candidates = [
        yaml_dir / f"{tok}_{base_path.name}",
        yaml_dir / f"{tok}_{base_stem}.yml",
        yaml_dir / f"{tok}_{base_stem}.yaml",
        base_path,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _has_tracking_for_pos(experiment: str, pos: str, ds: "OpsDataset") -> bool:
    """Return True if tracking data exists for this position.

    Checks two signals (in order):
    1. ``skip_track`` flag in the experiment config YAML.
    2. Experiment number ≥ 69 with well A/3/0 (or A3) — these wells were not
       time-lapse tracked starting from ops0069.

    Used by both the geff-track overlay and the ISS→Pheno compose path so that
    wells without tracking fall back to the ISS→Pheno affine directly.
    """
    # Check experiment config for skip_track flag
    exp_config_path = ds.config_paths.get("exp_config")
    if exp_config_path and Path(exp_config_path).exists():
        try:
            import yaml as _yaml

            with open(exp_config_path) as _f:
                _exp_cfg = _yaml.safe_load(_f) or {}
            if _exp_cfg.get("auto_register_params", {}).get("skip_track", False):
                return False
        except Exception:
            pass

    # Well A/3/0 has no tracking only for ops >= 69
    try:
        well_token = _well_token(pos)
        ops_num = int(experiment.split("_")[0].replace("ops", ""))
        if ops_num >= 69 and well_token in ("A3", "A30"):
            return False
    except (ValueError, IndexError):
        pass

    return True


def _apply_registration_affine_to_layer(
    lyr,
    T_reg: np.ndarray | None,
    scale_yx: tuple[float, float],
    store_key_for_yaml: str | None = None,
):
    """Apply the full registration + scale + well-separation affine to a layer.

    This is the single source of truth for composing layer affines.
    Composition order (right to left):
        final = T_well_sep @ T_offset @ S_global @ T_reg_embedded

    The layer's existing ``translate`` is consumed as the well-separation
    offset, then ``translate`` and ``scale`` are reset to identity so the
    affine alone drives positioning.
    """
    ndim = lyr.ndim

    # Global upscale on YX
    S_global = np.identity(ndim + 1)
    if ndim >= 2:
        S_global[ndim - 2, ndim - 2] = scale_yx[0]  # Y
        S_global[ndim - 1, ndim - 1] = scale_yx[1]  # X

    # Embed 4×4 ZYX registration into full (ndim+1, ndim+1) matrix
    T_reg_embedded = np.identity(ndim + 1)
    if T_reg is not None:
        if ndim >= 3:  # 3D or more (e.g., TZYX)
            T_reg_embedded[-(3 + 1):, -(3 + 1):] = T_reg
        elif ndim == 2:  # 2D (e.g., YX grid)
            T_2d = np.identity(3)
            T_2d[0:2, 0:2] = T_reg[1:3, 1:3]     # YX rotation/scale
            T_2d[0:2, 2] = T_reg[1:3, 3]           # YX translation
            T_reg_embedded[-(2 + 1):, -(2 + 1):] = T_2d

    # Cell painting offset correction
    # NOTE: These hardcoded offsets were empirical corrections that should
    # now be captured in the registration YAML. Commenting out to test
    # if registration alone provides correct alignment.
    T_offset = np.identity(ndim + 1)
    # if store_key_for_yaml == "lc_cell_painting1_register":
    #     if ndim >= 2:
    #         T_offset[ndim - 2, -1] = 27.0   # Y offset
    #         T_offset[ndim - 1, -1] = 135.0  # X offset
    # elif store_key_for_yaml == "lc_cell_painting2_register":
    #     if ndim >= 2:
    #         T_offset[ndim - 2, -1] = -8.0    # Y offset (negative = upward)
    #         T_offset[ndim - 1, -1] = 845.0   # X offset

    # Well separation transform (from the layer's current translate)
    T_well_sep = np.identity(ndim + 1)
    T_well_sep[:ndim, -1] = lyr.translate

    # Compose: final_affine = well_sep * offset * global_scale * registration
    final_affine = T_well_sep @ T_offset @ S_global @ T_reg_embedded

    vprintf("[apply-affine] layer=%s ndim=%d scale_yx=%s T_reg=%s translate=%s",
            lyr.name, ndim, str(scale_yx), "yes" if T_reg is not None else "no", str(list(lyr.translate)))
    lyr.affine = final_affine
    lyr.scale = [1.0] * ndim
    lyr.translate = [0.0] * ndim


def view_inplace_pyramid_in_napari(
    source_store: str | Path,
    positions: Optional[Sequence[str]] = None,
    contrast_limits: Optional[Sequence[float]] = None,
    name: Optional[str] = None,
    channel_axis: Optional[int] = None,
    show_grid_edges: bool = True,
    show_grid_ids: bool = True,
    show_tile_numbers: bool = True,
    show_exp_grids: bool = False,
    cell_paint: bool = False,
    v: Optional[object] = None,
    run_app: bool = True,
    mode: Optional[str] = None,
    scale_yx: Optional[Tuple[float, float]] = None,
    experiment: Optional[str] = None,
    show_tracks: bool = False,
    broadcast_shape: Optional[
        Tuple[int, int, int]
    ] = None,  # (ISS_max, Track_max, Point_max)
    skip_iss_overlays: bool = False,  # Skip ISS points/props for cell painting experiments
    offsets_x: Optional[dict[str, int]] = None,  # Pre-computed well separation offsets
    show_organelle_labels: bool = True,  # Auto-load organelle segmentation labels (*_seg in labels/)
):
    import napari

    source_store = Path(source_store)
    # Try to enable experimental async rendering for smoother UX
    from napari.settings import get_settings  # type: ignore

    print("Loading pyramid for positions: ", positions)

    settings = get_settings()
    experimental = getattr(settings, "experimental", None)
    if experimental is not None:
        setattr(experimental, "async_", True)
        vprintf("Napari experimental async rendering enabled by async_")

    # Choose positions to view
    if positions is not None and len(positions) > 0:
        selected_positions = list(positions)
    else:
        selected_positions = _iter_position_paths(source_store)
        if not selected_positions:
            raise FileNotFoundError(f"No positions found in {source_store}")

    # Compute per-well diameters and horizontal offsets so wells are spaced apart
    # by their diameter plus a 5% gap. This helps ensure that viewing one well
    # does not trigger loading of distant wells.
    # If offsets_x is pre-computed (from view_registered_stores_in_one_viewer), use it directly
    if offsets_x is None:
        offsets_x: dict[str, int] = {}
        diameters: dict[str, int] = {}
        try:
            with open_ome_zarr(source_store, mode="r") as _store_meta:
                for _pos in selected_positions:
                    try:
                        _base = _store_meta[_pos]["0"]
                        _y0, _x0 = int(_base.shape[-2]), int(_base.shape[-1])
                    except Exception:
                        # Fallback via dask shape if needed
                        _arr0 = da.from_zarr(
                            str(source_store), component=str(Path(_pos) / "0")
                        )
                        _y0, _x0 = int(_arr0.shape[-2]), int(_arr0.shape[-1])
                    diameters[_pos] = max(_y0, _x0)
        except Exception:
            diameters = {p: 0 for p in selected_positions}

        _current_offset = 0
        _prev_d = None
        for _pos in selected_positions:
            if _prev_d is None:
                offsets_x[_pos] = 0
                _prev_d = diameters.get(_pos, 0)
            else:
                _sep = int(max(_prev_d, diameters.get(_pos, 0)) * 1.05)
                _current_offset += _sep
                offsets_x[_pos] = _current_offset
                _prev_d = diameters.get(_pos, 0)

    # Infer channel axis, catching errors from stores with non-position directories (e.g., nuclear_seg)
    try:
        inferred_channel_axis = channel_axis or _infer_channel_axis_from_store(
            source_store
        )
    except (StopIteration, Exception) as e:
        vprintf(
            "[channel-axis] Could not infer channel axis from store: %s, using fallback",
            str(e),
        )
        # Fallback: check array shape directly from first position
        if channel_axis is not None:
            inferred_channel_axis = channel_axis
        elif selected_positions and len(selected_positions) > 0:
            try:
                first_pos = selected_positions[0]
                level_names_fallback = _list_numeric_level_names(
                    source_store / first_pos
                )
                if level_names_fallback:
                    test_arr = da.from_zarr(
                        str(source_store),
                        component=str(Path(first_pos) / level_names_fallback[0]),
                    )
                    # Typical shapes: (T, C, Z, Y, X) or (C, Z, Y, X) - channel usually at index 1
                    if test_arr.ndim >= 4:
                        inferred_channel_axis = 1
                    else:
                        inferred_channel_axis = None
                    vprintf(
                        "[channel-axis] Fallback detected channel_axis=%s from array shape %s",
                        str(inferred_channel_axis),
                        str(test_arr.shape),
                    )
                else:
                    inferred_channel_axis = None
            except Exception as e2:
                vprintf("[channel-axis] Fallback also failed: %s", str(e2))
                inferred_channel_axis = None
        else:
            inferred_channel_axis = None

    # Store the original inferred channel axis before it gets modified in the loop
    original_channel_axis = inferred_channel_axis

    v = v if v is not None else napari.Viewer()
    # viewer_layers_before = set(v.layers)
    # Ensure global verbosity is respected across modules
    try:
        _fs.VERBOSE = bool(_fs.VERBOSE)
    except Exception:
        pass
    # For wells: add one layer per well, no mosaic
    for pos in selected_positions:
        level_names = _list_numeric_level_names(source_store / pos)
        if len(level_names) <= 1:
            vprintf("No multiscale levels found under %s; skipping", pos)
            continue
        arrays = [
            da.from_zarr(str(source_store), component=str(Path(pos) / lvl))
            for lvl in level_names
        ]
        # Ensure a Z spatial axis exists so users can toggle 2D/3D in napari
        promoted_arrays = []
        for _arr in arrays:
            try:
                nd = int(getattr(_arr, "ndim", 0))
            except Exception:
                nd = 0
            if channel_axis is None and nd >= 3:
                promoted_arrays.append(_arr)
            else:
                try:
                    promoted_arrays.append(
                        da.expand_dims(_arr, axis=-2)
                    )  # insert Z before Y
                except Exception:
                    promoted_arrays.append(_arr)
        arrays = promoted_arrays

        # # Debug: log original shape before broadcasting
        # if arrays and len(arrays) > 0:
        #     vprintf(
        #         "[broadcast] Original shape before broadcast: %s, channel_axis=%s",
        #         str(arrays[0].shape),
        #         str(inferred_channel_axis),
        #     )

        # Broadcast arrays for dimension slider approach
        # ISS images: add singleton dims at positions 1, 2 → (T_iss, 1, 1, ...)
        # Track images: add singleton dims at positions 0, 2 → (1, T_track, 1, ...)
        # Pheno images: add singleton dims at positions 0, 1 → (1, 1, T_pheno, ...)
        # IMPORTANT: Update channel_axis after adding dimensions
        # If broadcast_shape is provided, use da.broadcast_to() to replicate across all time dims
        if mode == "iss":
            # ISS varies in dim 0 (ISS_Time)
            broadcast_arrays = []
            for arr in arrays:
                # Insert singleton dimensions at axis 1 and 2
                broadcasted = da.expand_dims(da.expand_dims(arr, axis=1), axis=2)
                # If broadcast_shape provided, replicate across Track and Point dims
                if broadcast_shape is not None:
                    iss_dim, track_dim, point_dim = broadcast_shape
                    # Current shape: (T_iss, 1, 1, ...), target: (T_iss, track_dim, point_dim, ...)
                    target_shape = (
                        broadcasted.shape[0],
                        track_dim,
                        point_dim,
                    ) + broadcasted.shape[3:]
                    broadcasted = da.broadcast_to(broadcasted, target_shape)
                    # vprintf("[broadcast] ISS: broadcast_to shape %s", str(target_shape))
                broadcast_arrays.append(broadcasted)
            arrays = broadcast_arrays
            # Update channel_axis: added 2 dims before the original axes (at positions 1, 2)
            # IMPORTANT: Use original_channel_axis to avoid accumulating adjustments across loop iterations
            if original_channel_axis is not None:
                inferred_channel_axis = original_channel_axis + 2
            # vprintf(
            #     "[broadcast] ISS images: added singleton dims at 1, 2, updated channel_axis to %s",
            #     str(inferred_channel_axis),
            # )
        elif mode == "track":
            # Track varies in dim 1 (Track_Time)
            broadcast_arrays = []
            for arr in arrays:
                # Insert singleton dimensions at axis 0 and 2
                broadcasted = da.expand_dims(arr, axis=0)
                broadcasted = da.expand_dims(broadcasted, axis=2)
                # If broadcast_shape provided, replicate across ISS and Point dims
                if broadcast_shape is not None:
                    iss_dim, track_dim, point_dim = broadcast_shape
                    # Current shape: (1, T_track, 1, ...), target: (iss_dim, T_track, point_dim, ...)
                    target_shape = (
                        iss_dim,
                        broadcasted.shape[1],
                        point_dim,
                    ) + broadcasted.shape[3:]
                    broadcasted = da.broadcast_to(broadcasted, target_shape)
                    # vprintf(
                    #     "[broadcast] Track: broadcast_to shape %s", str(target_shape)
                    # )
                broadcast_arrays.append(broadcasted)
            arrays = broadcast_arrays
            # Update channel_axis: added 2 dims (at positions 0 and 2), so channel_axis shifts by count of dims added before it
            # Original axis shifts by +1 (dim at 0), then we add another at position 2, channel shifts by +1 again if it was after pos 2
            # IMPORTANT: Use original_channel_axis to avoid accumulating adjustments across loop iterations
            if original_channel_axis is not None:
                inferred_channel_axis = original_channel_axis + 2  # Both dims added are before the channel axis
            # vprintf(
            #     "[broadcast] Track images: added singleton dims at 0, 2, updated channel_axis to %s",
            #     str(inferred_channel_axis),
            # )
        elif mode == "pheno":
            # Pheno typically has T=1 (single timepoint), so treat as static
            # Remove the time dimension if present, then add 3 new time dims for broadcasting
            broadcast_arrays = []
            for arr in arrays:
                original_shape = arr.shape
                # If pheno has a time dimension (first axis), remove it by selecting index 0
                if arr.shape[0] == 1:
                    # Remove singleton time dimension: (1, C, Z, Y, X) -> (C, Z, Y, X)
                    arr = arr[0]
                    # vprintf(
                    #     "[broadcast] Pheno: removed singleton time dimension %s -> %s",
                    #     str(original_shape),
                    #     str(arr.shape),
                    # )
                # else:
                #     vprintf(
                #         "[broadcast] Pheno: no singleton time dimension to remove, shape=%s",
                #         str(arr.shape),
                #     )

                # Insert 3 singleton dimensions at beginning for ISS, Track, Point
                broadcasted = da.expand_dims(arr, axis=0)  # ISS dim
                broadcasted = da.expand_dims(broadcasted, axis=1)  # Track dim
                broadcasted = da.expand_dims(broadcasted, axis=2)  # Point dim
                # vprintf(
                #     "[broadcast] Pheno: after adding 3 dims, shape=%s",
                #     str(broadcasted.shape),
                # )
                # Now shape is: (1, 1, 1, C, Z, Y, X)

                # If broadcast_shape provided, replicate across all 3 time dims
                if broadcast_shape is not None:
                    iss_dim, track_dim, point_dim = broadcast_shape
                    # Current shape: (1, 1, 1, ...), target: (iss_dim, track_dim, point_dim, ...)
                    target_shape = (iss_dim, track_dim, point_dim) + broadcasted.shape[
                        3:
                    ]
                    broadcasted = da.broadcast_to(broadcasted, target_shape)
                    # vprintf(
                    #     "[broadcast] Pheno: broadcast_to final shape %s (static across all time dims)",
                    #     str(target_shape),
                    # )
                # else:
                #     vprintf(
                #         "[broadcast] Pheno: no broadcast_shape provided, keeping shape=%s",
                #         str(broadcasted.shape),
                #     )
                broadcast_arrays.append(broadcasted)
            arrays = broadcast_arrays
            # Update channel_axis: we removed time dim (was at 0), then added 3 dims
            # Original: (T=1, C, Z, Y, X) with channel_axis=1
            # After removing T: (C, Z, Y, X) with channel_axis=0 (subtract 1 since we removed a dim before it)
            # After adding 3 dims: (1, 1, 1, C, Z, Y, X) with channel_axis=3 (add 3 for new dims)
            # Net effect: channel_axis goes from 1 -> 0 (remove T) -> 3 (add 3 dims)
            # IMPORTANT: Use original_channel_axis to avoid accumulating adjustments across loop iterations
            if original_channel_axis is not None:
                # print(f"[DEBUG-CHANNEL-AXIS] BEFORE adjustment: original_channel_axis={original_channel_axis}")
                # print(f"[DEBUG-CHANNEL-AXIS] Array shape after broadcast: {arrays[0].shape if arrays else 'empty'}")
                # Start with the original channel axis value
                inferred_channel_axis = original_channel_axis
                # First account for removed time dimension if it was before channel axis
                if inferred_channel_axis > 0:  # Time dim was at 0, so if channel was after it
                    inferred_channel_axis -= 1
                # Then add 3 for the new dimensions at the beginning
                inferred_channel_axis += 3
                # print(f"[DEBUG-CHANNEL-AXIS] AFTER adjustment: inferred_channel_axis={inferred_channel_axis}")
            # vprintf(
            #     "[broadcast] Pheno images: shape=%s, updated channel_axis to %s",
            #     str(arrays[0].shape if arrays else "empty"),
            #     str(inferred_channel_axis),
            # )

        # Read precomputed per-level (and optional per-channel) contrast limits if present
        per_level_clims: list[Tuple[float, float] | None] = []
        per_level_clims_per_channel: list[list[Tuple[float, float] | None] | None] = []
        gamma_per_channel: list[float] = []
        # Determine number of channels for this position
        try:
            with open_ome_zarr(source_store, mode="r") as _store_ch:
                _fov = _store_ch[pos]
                c_dim = _fov.data.shape[1] if _fov.data.ndim >= 2 else 1
        except Exception as e:
            # Fallback: infer from array shape if channel axis is known
            if inferred_channel_axis is not None and len(arrays) > 0:
                c_dim = arrays[0].shape[inferred_channel_axis]
            else:
                c_dim = 1
            vprintf(
                "[clims] Could not read channel count from metadata: %s, using c_dim=%d",
                str(e),
                c_dim,
            )
        # Read per-level clims (supports both v2 and v3 formats)
        # print(f"[DEBUG-CLIMS] About to read clims for pos={pos}, store={source_store}")
        # print(f"[DEBUG-CLIMS] level_names={level_names}, c_dim={c_dim}")
        per_level_clims, per_level_clims_per_channel, gamma_per_channel = read_per_level_clims(
            source_store, pos, level_names, c_dim
        )
        # print(f"[DEBUG-CLIMS] Result: per_level_clims={per_level_clims}")
        # print(f"[DEBUG-CLIMS] Result: gamma_per_channel={gamma_per_channel}")
        # vprintf("[clims] Final per_level_clims: %s", str(per_level_clims))

        # Choose initial clims: use coarsest level (typically level 4) to prevent napari auto-calc
        # Napari usually starts at the coarsest level when first displaying a multiscale layer
        initial_cl = None
        if len(per_level_clims) > 0:
            # Use the last (coarsest) level's clims as initial
            for i in range(len(per_level_clims) - 1, -1, -1):
                if per_level_clims[i] is not None:
                    initial_cl = per_level_clims[i]
                    break

        # Build a translate vector whose length matches image dims (after removing channel dim)
        try:
            arr0 = arrays[0]
            effective_ndim = int(getattr(arr0, "ndim", 2)) - (
                1 if inferred_channel_axis is not None else 0
            )
            translate_vec = [0.0] * max(1, effective_ndim)
            # Offset along last spatial axis (X)
            translate_vec[-1] = float(offsets_x.get(pos, 0))
            translate_tuple = tuple(translate_vec)
        except Exception:
            translate_tuple = (0.0, float(offsets_x.get(pos, 0)))

        # Pre-determine channel names to allow early visibility setting
        ch_names = None
        try:
            with open_ome_zarr(source_store, mode="r") as _store_meta2:
                try:
                    ch_names = list(getattr(_store_meta2, "channel_names", None) or [])
                except Exception:
                    ch_names = None
                if not ch_names:
                    try:
                        ch_names = list(
                            getattr(_store_meta2[pos], "channel_names", None) or []
                        )
                    except Exception:
                        ch_names = None
        except Exception:
            ch_names = None

        # Determine default gamma: phase channels use 1.0, others use 0.75
        gamma_val = 0.75
        if ch_names and len(ch_names) > 0:
            # Check if any channel has "phase" in its name
            has_phase = any("phase" in str(ch).lower() for ch in ch_names)
            if has_phase:
                gamma_val = 1.0

        # Add all layers as invisible initially to prevent async loader from caching
        # Set initial contrast_limits to prevent napari from auto-calculating when layer becomes visible
        img_added = v.add_image(
            arrays,
            multiscale=True,
            contrast_limits=initial_cl,
            colormap="gray",
            name=(name or source_store.stem) + f":{pos}",
            channel_axis=inferred_channel_axis,
            translate=translate_tuple,
            visible=False,  # Start ALL layers invisible
            blending="additive",
            gamma=gamma_val,
        )
        layers = img_added if isinstance(img_added, list) else [img_added]

        # Set per-layer gamma: prefer stored gamma_per_channel from build_clims, fallback to heuristics
        if ch_names and inferred_channel_axis is not None:
            for i, lyr in enumerate(layers):
                if i < len(gamma_per_channel):
                    # Use precomputed gamma from build_clims_in_place
                    lyr.gamma = gamma_per_channel[i]
                elif i < len(ch_names):
                    # Fallback to heuristic-based gamma
                    ch_name_lower = str(ch_names[i]).lower()
                    if "phase" in ch_name_lower or "focus" in ch_name_lower:
                        lyr.gamma = 1.0
                    else:
                        lyr.gamma = 0.75

        # Immediately set selective visibility for specific channels
        # Only do this AFTER renaming layers below so we can match by channel name
        layers_to_show = []

        # Helper function to find seg path (v2: pos/seg_name, v3: pos/labels/seg_name)
        def _find_seg_path(seg_name: str) -> Optional[Path]:
            """Find segmentation path, preferring v3 canonical (pos/labels/seg_name)
            over legacy v2 top-level symlink (pos/seg_name)."""
            # Try v3 canonical first: pos/labels/seg_name/
            v3_path = source_store / pos / "labels" / seg_name
            if v3_path.exists() and v3_path.is_dir():
                try:
                    level_names = _list_numeric_level_names(v3_path)
                    if level_names:
                        return Path(pos) / "labels" / seg_name
                except Exception:
                    pass

            # Fallback: legacy v2 top-level symlink at pos/seg_name/
            v2_path = source_store / pos / seg_name
            if v2_path.exists() and v2_path.is_dir():
                try:
                    level_names = _list_numeric_level_names(v2_path)
                    if level_names:
                        return Path(pos) / seg_name
                except Exception:
                    pass

            return None

        # Helper function to load and broadcast segmentation layers
        def _load_and_broadcast_seg(seg_name: str):
            """Load and broadcast a segmentation layer (seg or nuclear_seg)."""
            seg_path = _find_seg_path(seg_name)
            if seg_path is None:
                vprintf("[%s] Not found for %s (checked both pos/%s and pos/labels/%s)", seg_name, pos, seg_name, seg_name)
                return None

            try:
                level_names = _list_numeric_level_names(source_store / seg_path)
            except Exception:
                level_names = []

            if not level_names:
                return None

            try:
                # Load segmentation arrays, allowing automatic format detection
                seg_arrays = [
                    da.from_zarr(
                        str(source_store),
                        component=str(seg_path / lvl),
                    )
                    for lvl in level_names
                ]

                # Broadcast seg arrays to match viewer dimensionality
                # Seg has shape (T, C, Z, Y, X) where T can be 1 (pheno/ISS) or 4 (track time), C=1, Z=1
                # Labels don't have channels, target: (ISS, Track, Point, Z, Y, X) = 6D
                broadcast_seg_arrays = []
                for seg_arr in seg_arrays:
                    original_shape = seg_arr.shape

                    # Handle different modes:
                    # - pheno/ISS: (1, 1, 1, Y, X) -> remove T,C -> (Z, Y, X), then add 3 time dims
                    # - track: (T, 1, 1, Y, X) -> remove C only -> (T, Z, Y, X), then add ISS and Point dims
                    if seg_arr.ndim == 5:
                        # (T, C, Z, Y, X)
                        if mode == "track" and seg_arr.shape[0] > 1:
                            # Track mode with multiple timepoints: keep T dimension, remove C
                            # (T, C, Z, Y, X) -> (T, Z, Y, X)
                            seg_arr = seg_arr[:, 0]
                            # vprintf(
                            #     "[broadcast] %s: removed C dimension (kept T=%d) %s -> %s",
                            #     seg_name,
                            #     seg_arr.shape[0],
                            #     str(original_shape),
                            #     str(seg_arr.shape),
                            # )
                            # seg_arr is now (T, Z, Y, X) - need to add ISS and Point dims
                            # Target: (ISS, Track, Point, Z, Y, X) where Track=T
                            broadcasted_seg = da.expand_dims(
                                seg_arr, axis=0
                            )  # ISS dim at 0
                            broadcasted_seg = da.expand_dims(
                                broadcasted_seg, axis=2
                            )  # Point dim at 2
                            # Now: (1, T, 1, Z, Y, X) = (ISS, Track, Point, Z, Y, X)
                        else:
                            # Pheno/ISS mode or track with T=1: remove both T and C
                            # (1, 1, 1, Y, X) -> (Z, Y, X)
                            seg_arr = seg_arr[0, 0]
                            # vprintf(
                            #     "[broadcast] %s: removed T,C dimensions %s -> %s",
                            #     seg_name,
                            #     str(original_shape),
                            #     str(seg_arr.shape),
                            # )
                            # seg_arr is now (Z, Y, X) - add 3 time dims
                            broadcasted_seg = da.expand_dims(seg_arr, axis=0)  # ISS dim
                            broadcasted_seg = da.expand_dims(
                                broadcasted_seg, axis=1
                            )  # Track dim
                            broadcasted_seg = da.expand_dims(
                                broadcasted_seg, axis=2
                            )  # Point dim
                            # Now: (1, 1, 1, Z, Y, X) = (ISS, Track, Point, Z, Y, X)
                    else:
                        # Already has correct dimensionality or unexpected shape
                        broadcasted_seg = seg_arr

                    # Broadcast if broadcast_shape available
                    if broadcast_shape is not None:
                        iss_dim, track_dim, point_dim = broadcast_shape
                        current_shape = broadcasted_seg.shape
                        # Target: (iss_dim, track_dim, point_dim, ...) but respect existing Track dimension
                        if (
                            mode == "track"
                            and len(current_shape) >= 2
                            and current_shape[1] > 1
                        ):
                            # Track seg already has its own Track dimension - don't broadcast over it
                            target_shape = (
                                iss_dim,
                                current_shape[1],
                                point_dim,
                            ) + current_shape[3:]
                        else:
                            # Broadcast across all time dims
                            target_shape = (
                                iss_dim,
                                track_dim,
                                point_dim,
                            ) + current_shape[3:]
                        broadcasted_seg = da.broadcast_to(broadcasted_seg, target_shape)
                        # vprintf(
                        #     "[broadcast] %s: broadcast_to shape %s",
                        #     seg_name,
                        #     str(target_shape),
                        # )

                    broadcast_seg_arrays.append(broadcasted_seg)

                # Build translate tuple matching seg array dimensions
                # For multiscale, napari uses the highest resolution (first) array's ndim
                # We need translate to match this exactly
                seg_ndim = broadcast_seg_arrays[0].ndim

                # Translate must match the array's actual ndim
                # Only set X offset (last dimension)
                seg_translate = [0.0] * seg_ndim
                seg_translate[-1] = float(offsets_x.get(pos, 0))  # X offset

                # Build layer name to match image layer naming: store_tag:pos:layer_type
                store_tag = name or source_store.stem
                layer_name = f"{store_tag}:{pos}:{seg_name}"

                seg_layer = v.add_labels(
                    broadcast_seg_arrays,
                    multiscale=True,
                    name=layer_name,
                    translate=tuple(seg_translate),
                    visible=False,
                    blending="additive",
                )
                vprintf("[%s] Added segmentation layer for %s", seg_name, pos)
                return seg_layer
            except Exception as e:
                vprintf(
                    "[%s] Failed to add segmentation layer for %s: %s",
                    seg_name,
                    pos,
                    str(e),
                )
                import traceback

                traceback.print_exc()
                return None

        # Load segmentation layers - now all modes use same structure
        if mode == "pheno":
            # Pheno has both seg and nuclear_seg at pos/seg_name/
            _load_and_broadcast_seg("seg")
            _load_and_broadcast_seg("nuclear_seg")
        elif mode in ["iss", "track"]:
            # ISS/Track only have nuclear_seg (symlinked to match pheno structure at pos/nuclear_seg/)
            _load_and_broadcast_seg("nuclear_seg")

        # Auto-discover and load organelle segmentation labels (*_seg in labels/)
        if show_organelle_labels and mode == "pheno":
            labels_dir = source_store / pos / "labels"
            if labels_dir.exists():
                try:
                    # Discover organelle labels: end with '_seg', exclude 'seg' and 'nuclear_seg'
                    organelle_labels = sorted([
                        d.name for d in labels_dir.iterdir()
                        if d.is_dir() and not d.name.startswith('.')
                        and d.name.endswith('_seg')
                        and d.name not in ('seg', 'nuclear_seg')
                    ])
                    for org_label in organelle_labels:
                        _load_and_broadcast_seg(org_label)
                    if organelle_labels:
                        vprintf("[organelle-seg] Loaded %d organelle labels for %s: %s",
                               len(organelle_labels), pos, ", ".join(organelle_labels))
                except Exception as e:
                    vprintf("[organelle-seg] Failed to discover organelle labels for %s: %s", pos, str(e))

        # Rename layers to channel names and apply default colormaps by channel
        store_tag = name or source_store.stem
        # Look up channel→label map for this experiment so we can append the
        # biological marker (e.g. 'GFP' → 'GFP_FastAct_SPY555') from ops_channel_maps.yaml.
        _exp_marker_map: dict[str, str] = {}
        try:
            _ops_key = extract_ops_key(experiment) if experiment else None
            if _ops_key is None:
                _ops_key = extract_ops_key(name or source_store.stem)
            if _ops_key:
                _exp_marker_map = _load_ops_channel_map_yaml().get(_ops_key, {}) or {}
        except Exception:
            _exp_marker_map = {}
        # Fallback palette for channels not matched by keyword rules (e.g. 4i antibodies).
        # Cycled so each unmatched channel gets a distinct color instead of gray.
        _fallback_palette = [
            "bop orange", "red", "yellow", "green", "cyan", "magenta", "bop purple", "bop blue",
        ]
        _fallback_idx = 0
        try:
            if ch_names and len(ch_names) == len(layers):
                for lyr, ch_name in zip(layers, ch_names):
                    try:
                        ch_str = str(ch_name)
                        marker = _marker_suffix_for_label(_exp_marker_map.get(ch_str))
                        ch_label = f"{ch_str}_{marker}" if marker else ch_str
                        # Preserve store and position in layer name to allow downstream transforms to match by prefix
                        lyr.name = f"{store_tag}:{pos}:{ch_label}"
                    except Exception:
                        pass
                    cname_l = str(ch_name).lower()
                    cm = None
                    # ISS-specific channel coloring
                    if isinstance(mode, str) and mode == "iss":
                        import re as _re

                        if "dapi" in cname_l:
                            cm = "blue"
                        else:
                            base = None
                            m = _re.search(r"miseq[-_\s]*([acgt])", cname_l)
                            if m:
                                base = m.group(1)
                            elif cname_l.strip() in ("a", "c", "g", "t"):
                                base = cname_l.strip()
                            if base == "a":
                                cm = "green"
                            elif base == "c":
                                cm = "red"
                            elif base == "g":
                                cm = "yellow"
                            elif base == "t":
                                cm = "magenta"
                    if "gfp" in cname_l:
                        cm = "bop orange"
                    elif "mcherry" in cname_l:
                        cm = "bop purple"
                    elif "dapi" in cname_l or "hoechst" in cname_l or "nuclei" in cname_l:
                        cm = "blue"
                    elif "membrane" in cname_l:
                        cm = "bop purple"
                    elif "phase2d" in cname_l or "phase_2d" in cname_l or "focus3d" in cname_l or "focus_3d" in cname_l or cname_l == "raw" or cname_l.startswith("bf_z"):
                        cm = "gray"
                    if cm is None:
                        cm = _fallback_palette[_fallback_idx % len(_fallback_palette)]
                        _fallback_idx += 1
                    try:
                        lyr.colormap = cm
                    except Exception:
                        pass

                    # Determine which channels should be shown based on mode and channel name
                    # Only show Phase2D from the phenotyping store (mode == "pheno")
                    if "phase2d" in cname_l and mode == "pheno":
                        layers_to_show.append(lyr)
        except Exception:
            pass

        # Build per-layer clims map (prefer per-channel; fallback to level clims)
        per_layer_clims_map: dict[int, list[Tuple[float, float] | None]] = {}
        try:
            for idx, lyr in enumerate(layers):
                clims_for_layer: list[Tuple[float, float] | None] = []
                for li, _ in enumerate(level_names):
                    pc_list = (
                        per_level_clims_per_channel[li]
                        if li < len(per_level_clims_per_channel)
                        else None
                    )
                    if (
                        pc_list is not None
                        and idx < len(pc_list)
                        and pc_list[idx] is not None
                    ):
                        clims_for_layer.append(pc_list[idx])
                    else:
                        clims_for_layer.append(
                            per_level_clims[li] if li < len(per_level_clims) else None
                        )
                per_layer_clims_map[id(lyr)] = clims_for_layer
        except Exception:
            # Fallback: same per-level clims for all layers
            for lyr in layers:
                per_layer_clims_map[id(lyr)] = list(per_level_clims)

        # ISS: duplicate DAPI (round 0) across all rounds for the DAPI channel layer only
        try:
            if (
                isinstance(mode, str)
                and mode == "iss"
                and ch_names
                and len(ch_names) == len(layers)
            ):
                dapi_idx = None
                for i, nm in enumerate(ch_names):
                    if "dapi" in str(nm).lower() or str(nm).lower().startswith(
                        "1-dapi"
                    ):
                        dapi_idx = i
                        break
                if dapi_idx is not None and inferred_channel_axis is not None:
                    # Build per-level arrays list for DAPI with round 0 replicated across all rounds
                    dapi_ms: list = []
                    for arr_lvl in arrays:
                        try:
                            # Extract DAPI channel
                            arr_ch = da.take(
                                arr_lvl,
                                indices=int(dapi_idx),
                                axis=int(inferred_channel_axis),
                            )
                            # Round axis is at position 0
                            t_dim = (
                                int(getattr(arr_ch, "shape", (1,))[0])
                                if getattr(arr_ch, "ndim", 0) >= 1
                                else 1
                            )
                            if t_dim > 1:
                                # Take round 0 and replicate across all rounds
                                first_t = da.take(arr_ch, indices=0, axis=0)
                                rep = da.repeat(
                                    first_t[None, ...], repeats=t_dim, axis=0
                                )
                                dapi_ms.append(rep)
                            else:
                                dapi_ms.append(arr_ch)
                        except Exception:
                            dapi_ms.append(arr_lvl)
                    # Update the DAPI layer with replicated data
                    try:
                        layers[int(dapi_idx)].data = dapi_ms
                    except Exception:
                        pass
        except Exception:
            pass

        # Apply initial clims discretely for all layers
        for lyr in layers:
            _apply_clims_to_layer(
                lyr, per_layer_clims_map.get(id(lyr), per_level_clims), pos
            )

        # Track last applied level per layer to avoid redundant updates
        _last_level: dict[int, int | None] = {id(lyr): None for lyr in layers}

        # Bind discrete contrast updates only when multiscale level changes
        def _on_zoom(
            event=None,
            layers=layers,
            _pos=pos,
            _last=_last_level,
            _clims_map=per_layer_clims_map,
            _default=per_level_clims,
        ):
            _apply_clims_on_zoom(
                layers=layers,
                pos=_pos,
                _last_level=_last,
                per_layer_clims_map=_clims_map,
                per_level_clims=_default,
            )

        try:
            v.camera.events.zoom.connect(_on_zoom)
        except Exception:
            pass
        # Also react to napari's multiscale level changes per layer
        try:
            for lyr in layers:

                def _on_level_change(event=None, _lyr=lyr):
                    _on_zoom()

                lyr.events.data_level.connect(_on_level_change)
        except Exception:
            pass
        # React to dims step (e.g., channel/time slice changes)
        try:
            v.dims.events.current_step.connect(
                lambda _e=None, _layers=layers, _pos=pos, _last=_last_level, _map=per_layer_clims_map, _def=per_level_clims: (
                    _apply_clims_on_zoom(
                        event=_e,
                        layers=_layers,
                        pos=_pos,
                        _last_level=_last,
                        per_layer_clims_map=_map,
                        per_level_clims=_def,
                    )
                )
            )
        except Exception:
            pass
        # Apply after first draw to catch initial level selection
        try:
            from qtpy.QtCore import QTimer  # type: ignore

            QTimer.singleShot(100, _on_zoom)
            QTimer.singleShot(500, _on_zoom)
        except Exception:
            pass

        # Apply selective visibility
        for lyr in layers_to_show:
            lyr.visible = True
            # vprintf("Set layer visible: %s", lyr.name)

        # Listen to data_level changes to apply per-level clims on zoom
        # Also fires when layer first loads, applying correct clims for initial zoom level
        for lyr in layers:

            def _on_data_level_change(
                event=None,
                _lyr=lyr,
                _map=per_layer_clims_map,
                _def=per_level_clims,
                _pos=pos,
            ):
                if _lyr.visible:  # Only apply if layer is visible
                    _apply_clims_to_layer(_lyr, _map.get(id(_lyr), _def), _pos)

            try:
                lyr.events.data_level.connect(_on_data_level_change)
            except Exception:
                pass

        # ISS overlay: prefer prebuilt image overlay; check both v2 and v3 locations
        # Skip ISS overlays if this is a cell painting experiment
        # Supports both new names (iss_gene_image, iss_guide_image) and legacy (iss_points, iss_points_props)
        if not skip_iss_overlays:
            # Check for new naming convention first, then fall back to legacy
            # New names: iss_gene_image, iss_guide_image (self-contained RGBA images)
            # Legacy names: iss_points, iss_points_props (grayscale + separate props)
            iss_gene_v2 = source_store / pos / "iss_gene_image"
            iss_gene_v3 = source_store / pos / "labels" / "iss_gene_image"
            iss_points_v2 = source_store / pos / "iss_points"
            iss_points_v3 = source_store / pos / "labels" / "iss_points"

            iss_overlay_exists = (
                iss_gene_v2.exists() or iss_gene_v3.exists() or
                iss_points_v2.exists() or iss_points_v3.exists()
            )

            vprintf(
                "[DEBUG] Checking ISS overlay: mode=%s, gene_v2=%s, gene_v3=%s, points_v2=%s, points_v3=%s",
                str(mode),
                str(iss_gene_v2.exists()),
                str(iss_gene_v3.exists()),
                str(iss_points_v2.exists()),
                str(iss_points_v3.exists()),
            )
            if isinstance(mode, str) and mode == "pheno" and iss_overlay_exists:
                try:
                    vprintf("[ISS overlay] Rendering ISS image for %s", pos)
                    render_iss_image(source_store, pos, offsets_x, v)
                except Exception as e:
                    vprintf("[ISS overlay-img] Exception: %s", str(e))
                    import traceback

                    traceback.print_exc()
            elif isinstance(mode, str) and mode == "pheno":
                vprintf("WARNING: No ISS overlay found for %s (checked iss_gene_image and iss_points in both v2/v3 locations)", pos)
        else:
            vprintf("[cell-paint] Skipping ISS overlay for %s (cell painting mode)", pos)
        # Optional overlays per position
        # Skip grid overlays if this is a cell painting experiment
        if not skip_iss_overlays:
            # Grid overlay (RGBA image with blue lines + text) - check both new and legacy formats
            grid_overlay_v2 = source_store / pos / "grid_overlay"
            grid_overlay_v3 = source_store / pos / "labels" / "grid_overlay"
            grid_edges_v2 = source_store / pos / "grid_edges"
            grid_edges_v3 = source_store / pos / "labels" / "grid_edges"
            if show_grid_edges and (
                grid_overlay_v2.exists() or grid_overlay_v3.exists() or
                grid_edges_v2.exists() or grid_edges_v3.exists()
            ):
                try:
                    render_grid_overlay(
                        source_store, pos, offsets_x, v, store_tag=Path(source_store).stem
                    )
                except Exception as e:
                    vprintf("Skipping grid overlay for %s: %s", pos, str(e))
            # Tile numbers as points (legacy format only - new format has embedded text)
            grid_props_v2 = source_store / pos / "grid_props"
            grid_props_v3 = source_store / pos / "labels" / "grid_props"
            if show_grid_ids and (grid_props_v2.exists() or grid_props_v3.exists()):
                try:
                    render_grid_props(
                        source_store,
                        pos,
                        show_tile_numbers,
                        offsets_x,
                        v,
                        store_tag=Path(source_store).stem,
                    )
                except Exception as e:
                    vprintf("Skipping grid_props for %s: %s", pos, str(e))
            else:
                vprintf(
                    "No grid_props found for %s (checked pos/grid_props and pos/labels/grid_props)",
                    pos,
                )
        else:
            vprintf("[cell-paint] Skipping grid overlays (grid_edges, grid_props) for %s (cell painting mode)", pos)

    # Tracks overlay from geff files - loaded last so other layers appear first.
    # Apply the same registration affine as image layers via _apply_registration_affine_to_layer.
    if show_tracks and experiment is not None:
        _ds_tracks = OpsDataset(experiment)
        # Tracks scale: 5x tracking → 20x pheno = 4x upscale
        _tracks_scale = scale_yx if scale_yx is not None else (4.0, 4.0)

        for pos in selected_positions:
            # Determine if tracking data exists for this position.
            # For wells without tracking (skip_track=True or ops>=69 A/3/0), fall back
            # to the ISS→Pheno registration so the tracks are still placed correctly.
            _has_track = _has_tracking_for_pos(experiment, pos, _ds_tracks)
            vprintf("[geff-track] pos=%s has_tracking=%s", pos, str(_has_track))

            # Look up the appropriate registration YAML for this position.
            # - With tracking: use Track→Pheno (auto_pheno_register)
            # - Without tracking: coords are already in pheno space (per datasets.py),
            #   so no registration transform — only the 4x upscale (handled by _apply_registration_affine_to_layer)
            T_reg = None
            if _has_track:
                chosen_yaml = _find_per_well_yaml(_ds_tracks, "auto_pheno_register", pos)
                if chosen_yaml is not None:
                    try:
                        T_reg = read_transform_biahub(chosen_yaml)
                        vprintf("[geff-track] Using Track→Pheno YAML %s for %s", str(chosen_yaml), pos)
                    except Exception as _e_yaml:
                        vprintf("[geff-track] Failed to read registration YAML: %s", str(_e_yaml))
            else:
                vprintf("[geff-track] No tracking for %s: coords already in pheno space, using identity (4x upscale only)", pos)

            # 1) Current tracks (from tracking_geff)
            try:
                vprintf("[geff-track] Attempting to render tracks for %s", pos)
                tracks_layer = render_tracks_from_geff(experiment, pos, offsets_x, v)
                if tracks_layer is not None:
                    _apply_registration_affine_to_layer(
                        tracks_layer, T_reg, _tracks_scale
                    )
                    vprintf("[geff-track] Applied registration affine to tracks for %s", pos)
            except Exception as e:
                vprintf("[geff-track] Skipping tracks for %s: %s", pos, str(e))

    # Set custom axis labels for dimension slider approach
    # After all layers are loaded, set the axis labels based on viewer dimensionality
    try:
        if v.dims.ndim >= 6:
            # 6D: (ISS_Time, Track_Time, Point_Time, Z, Y, X)
            v.dims.axis_labels = ["ISS_Time", "Track_Time", "Point_Time", "Z", "Y", "X"]
            vprintf("[dims] Set 6D axis labels: %s", str(v.dims.axis_labels))
        elif v.dims.ndim == 5:
            # 5D: (ISS_Time, Track_Time, Point_Time, Y, X)
            v.dims.axis_labels = ["ISS_Time", "Track_Time", "Point_Time", "Y", "X"]
            vprintf("[dims] Set 5D axis labels: %s", str(v.dims.axis_labels))
        else:
            vprintf(
                "[dims] Viewer has %d dimensions, not setting custom labels",
                v.dims.ndim,
            )
    except Exception as e:
        vprintf("[dims] Could not set axis labels: %s", str(e))

    # Note: experiment-labels and experiment-stamps overlays have been removed.
    # Grid tile names are now rendered directly in the grid_overlay layer built by build_dask.py.
    # The show_exp_grids parameter is kept for backward compatibility but is now a no-op.
    # Optional: add Cell Painting overlay to the well immediately right of the first selected
    if cell_paint:
        try:
            # Import lazily to avoid circular import at module load time
            from cyclops_process.napari.dask.cell_paint import _add_cell_painting_overlay  # type: ignore

            _add_cell_painting_overlay(v, source_store, selected_positions, offsets_x)
        except Exception as e:
            vprintf("Failed to add cell painting overlay: %s", str(e))

    # # apply 4x scale to iss and track layers
    # if scale_yx is not None:
    #     viewer_layers_after = set(v.layers)
    #     newly_added_layers = viewer_layers_after - viewer_layers_before
    #     vprintf("[rescale] applying %s scale to %d new layers", str(scale_yx), len(newly_added_layers))
    #     for lyr in newly_added_layers:
    #         try:
    #             ndim = lyr.ndim
    #             T_sep = np.identity(ndim + 1)
    #             T_sep[:ndim, -1] = lyr.translate
    #             S_global = np.identity(ndim + 1)
    #             if ndim >= 2:
    #                 S_global[ndim - 2, ndim - 2] = scale_yx[0]  # Y
    #                 S_global[ndim - 1, ndim - 1] = scale_yx[1]  # X
    #             final_affine = T_sep @ S_global
    #             lyr.affine = final_affine
    #             lyr.scale = [1.0] * ndim
    #             lyr.translate = [0.0] * ndim
    #         except Exception as e:
    #             vprintf("[rescale] failed for layer %s: %s", lyr.name, str(e))

    if run_app:
        napari.run()
    return v


def view_registered_stores_in_one_viewer(
    experiment: str,
    stores: Sequence[str | Path],
    positions: Optional[Sequence[str]] = None,
    contrast_limits: Optional[Sequence[float]] = None,
    channel_axis: Optional[int] = None,
    name: Optional[str] = None,
    show_exp_grids: bool = False,
    show_tracks: bool = True,
    show_organelle_labels: bool = True,
):
    import napari

    if not stores:
        raise ValueError("No stores provided")
    ds = OpsDataset(experiment)

    # Determine max time dimensions across all stores for broadcasting
    # This ensures all layers are visible at all slider positions
    pos_list = (
        list(positions) if positions is not None else _iter_position_paths(stores[0])
    )
    if pos_list:
        sample_pos = pos_list[0]
    else:
        sample_pos = None

    max_iss_time = 1
    max_track_time = 1
    max_point_time = 1

    if sample_pos:
        for store_path in stores:
            store_path = Path(store_path)
            try:
                # Determine mode for this store
                store_mode = None
                if str(store_path) == str(ds.store_paths.get("iss_stitch_registered_v3")):
                    store_mode = "iss"
                elif str(store_path) == str(
                    ds.store_paths.get("lc_5x_phase_2d_stitched_v3")
                ):
                    store_mode = "track"
                elif str(store_path) == str(ds.store_paths.get("pheno_assembled_v3")) or str(store_path) == str(ds.store_paths.get("pheno_assembled")):
                    store_mode = "pheno"

                # Load a sample array to check time dimension
                lvl_names = _list_numeric_level_names(store_path / sample_pos)
                if lvl_names:
                    sample_arr = da.from_zarr(
                        str(store_path), component=str(Path(sample_pos) / lvl_names[0])
                    )
                    # Assume first dimension is time (T, C, Z, Y, X) or (T, Z, Y, X)
                    if store_mode == "iss" and sample_arr.ndim >= 1:
                        max_iss_time = max(max_iss_time, sample_arr.shape[0])
                    elif store_mode == "track" and sample_arr.ndim >= 1:
                        max_track_time = max(max_track_time, sample_arr.shape[0])
                    # Pheno typically doesn't have time, stays at 1
            except Exception as e:
                vprintf(
                    "[broadcast] Could not determine dimensions for %s: %s",
                    str(store_path),
                    str(e),
                )

    # Point_Time dimension: pheno (1) + ISS (1) + track timepoints (varies by well)
    # This ensures all point-based overlays (pheno cells, ISS calls, track points) can be displayed
    max_point_time = 1 + 1 + max_track_time  # pheno + ISS + tracks
    # vprintf(
    #     "[broadcast] Set max_point_time=%d (pheno=1 + iss=1 + track=%d)",
    #     max_point_time,
    #     max_track_time,
    # )

    broadcast_shape = (max_iss_time, max_track_time, max_point_time)
    # vprintf(
    #     "[broadcast] Determined broadcast shape: ISS=%d, Track=%d, Point=%d",
    #     max_iss_time,
    #     max_track_time,
    #     max_point_time,
    # )

    # Check if this is a cell painting experiment (has cell painting stores)
    has_cell_painting = any(
        "part1_max_proj_stitched" in str(Path(s).name) or "part2_max_proj_stitched" in str(Path(s).name)
        for s in stores
    )
    if has_cell_painting:
        vprintf("[cell-paint] Detected cell painting stores - will skip ISS and track layers")

    # Compute well separation offsets once for all stores to ensure consistent positioning
    # This is critical for proper registration across multiple stores
    base = Path(stores[0])
    offsets_x: dict[str, int] = {}
    diameters: dict[str, int] = {}
    try:
        with open_ome_zarr(base, mode="r") as _store_meta:
            for _pos in pos_list:
                try:
                    _base = _store_meta[_pos]["0"]
                    _y0, _x0 = int(_base.shape[-2]), int(_base.shape[-1])
                except Exception:
                    # Fallback via dask shape if needed
                    _arr0 = da.from_zarr(
                        str(base), component=str(Path(_pos) / "0")
                    )
                    _y0, _x0 = int(_arr0.shape[-2]), int(_arr0.shape[-1])
                diameters[_pos] = max(_y0, _x0)
    except Exception:
        diameters = {p: 0 for p in pos_list}

    _current_offset = 0
    _prev_d = None
    for _pos in pos_list:
        if _prev_d is None:
            offsets_x[_pos] = 0
            _prev_d = diameters.get(_pos, 0)
        else:
            _sep = int(max(_prev_d, diameters.get(_pos, 0)) * 1.05)
            _current_offset += _sep
            offsets_x[_pos] = _current_offset
            _prev_d = diameters.get(_pos, 0)

    vprintf("[offsets] Computed well separation offsets: %s", str(offsets_x))

    # Determine mode for base store
    base_mode = None
    if str(base) == str(ds.store_paths.get("pheno_assembled_v3")) or str(base) == str(ds.store_paths.get("pheno_assembled")):
        base_mode = "pheno"
    elif str(base) == str(ds.store_paths.get("iss_stitch_registered_v3")):
        base_mode = "iss"
    elif str(base) == str(ds.store_paths.get("lc_5x_phase_2d_stitched_v3")):
        base_mode = "track"

    v = view_inplace_pyramid_in_napari(
        source_store=base,
        positions=positions,
        contrast_limits=contrast_limits,
        name=base.stem,
        channel_axis=channel_axis,
        show_grid_edges=True,
        show_grid_ids=True,
        show_tile_numbers=True,
        show_exp_grids=bool(show_exp_grids),
        cell_paint=False,
        v=None,
        run_app=False,
        mode=base_mode,
        experiment=experiment,
        show_tracks=False,  # Load tracks after all stores are loaded
        broadcast_shape=broadcast_shape,
        skip_iss_overlays=has_cell_painting,
        offsets_x=offsets_x,  # Pass shared offsets for consistent positioning
        show_organelle_labels=show_organelle_labels,
    )
    # Registered overlays: determine registration YAML per known store
    for extra in stores[1:]:
        store_path = Path(extra)

        # Skip ISS and track stores if this is a cell painting experiment
        if has_cell_painting:
            if str(store_path) == str(ds.store_paths.get("iss_stitch_registered_v3")):
                vprintf("[cell-paint] Skipping ISS store: %s", store_path.stem)
                continue
            if str(store_path) == str(ds.store_paths.get("lc_5x_phase_2d_stitched_v3")):
                vprintf("[cell-paint] Skipping track store: %s", store_path.stem)
                continue

        store_key_for_yaml = None
        # Identify which registration file to use based on known keys
        # iss overlay - use auto-registration
        if str(store_path) == str(ds.store_paths.get("iss_stitch_registered_v3")):
            store_key_for_yaml = "auto_iss_register"
        # tracking overlay - use auto-registration
        if str(store_path) == str(ds.store_paths.get("lc_5x_phase_2d_stitched_v3")):
            store_key_for_yaml = "auto_pheno_register"
        # cell painting overlays for experiment 72
        store_name = store_path.name
        if "part1_max_proj_stitched" in store_name:
            store_key_for_yaml = "lc_cell_painting1_register"
        elif "part2_max_proj_stitched" in store_name:
            store_key_for_yaml = "lc_cell_painting2_register"
        # Fallback: no registration
        scale_yx, offset_yx = (1.0, 1.0), (0.0, 0.0)
        yaml_p = None
        if store_key_for_yaml is not None:
            yaml_p = ds.config_paths.get(store_key_for_yaml)
            vprintf("[register] store=%s key=%s base_yaml=%s exists=%s", store_path.stem, store_key_for_yaml, str(yaml_p), str(Path(yaml_p).exists() if yaml_p is not None else False))

        # In 'all' mode, upscale ISS and track stores by 4x to match pheno 20x
        try:
            base_stem = base.stem
        except Exception:
            base_stem = ""
        try:
            if base_stem and (
                store_path == Path(ds.store_paths.get("iss_stitch_registered_v3", ""))
                or store_path == Path(ds.store_paths.get("lc_5x_phase_2d_stitched_v3", ""))
            ):
                scale_yx = (float(scale_yx[0]) * 4.0, float(scale_yx[1]) * 4.0)
                # vprintf("[register] applying additional 4x upscale for %s (scale only)", store_path.stem)
        except Exception:
            pass
        # Render this store with standard viewer
        # Determine mode for this overlay store
        overlay_mode = None
        if str(store_path) == str(ds.store_paths.get("iss_stitch_registered_v3")):
            overlay_mode = "iss"
        elif str(store_path) == str(ds.store_paths.get("lc_5x_phase_2d_stitched_v3")):
            overlay_mode = "track"
        elif "part1_max_proj" in str(store_path) or "part2_max_proj" in str(store_path):
            overlay_mode = "pheno"  # Cell painting overlays treated as pheno

        v = view_inplace_pyramid_in_napari(
            source_store=store_path,
            positions=positions,
            contrast_limits=contrast_limits,
            name=store_path.stem,
            channel_axis=channel_axis,
            show_grid_edges=True,
            show_grid_ids=True,
            show_tile_numbers=True,
            show_exp_grids=False,
            cell_paint=False,
            v=v,
            run_app=False,
            mode=overlay_mode,
            scale_yx=scale_yx,
            broadcast_shape=broadcast_shape,
            skip_iss_overlays=has_cell_painting,
            offsets_x=offsets_x,  # Use shared offsets for consistent positioning
            show_organelle_labels=False,  # Only load organelle labels from base store
        )

        # Apply colormaps for cell painting layers
        # (Contrast is handled automatically by the dynamic system in view_inplace_pyramid_in_napari
        #  which reads per-level contrast limits from zarr attributes)
        # print(f"[DEBUG] store_key_for_yaml = '{store_key_for_yaml}'")
        if store_key_for_yaml in (
            "lc_cell_painting1_register",
            "lc_cell_painting2_register",
        ):
            # print(f"[DEBUG] Matched cell painting store: {store_key_for_yaml}")
            # Define color schemes for each cell painting store
            cp_colors = {
                "lc_cell_painting1_register": ["blue", "bop orange", "green", "cyan"],
                "lc_cell_painting2_register": ["blue", "bop orange", "bop purple", "bop blue"],
            }

            colors = cp_colors.get(store_key_for_yaml, [])
            # print(f"[DEBUG] Selected colors: {colors}")

            # Get newly added image layers for this store (exclude grid overlays, points, shapes)
            newly_added_for_color = [
                lyr
                for lyr in v.layers
                if lyr.name.startswith(store_path.stem)
                and hasattr(lyr, "data")
                and hasattr(lyr, "colormap")
                and not any(x in lyr.name for x in [":grid:", ":tile-ids:"])
            ]
            # print(f"[DEBUG] Found {len(newly_added_for_color)} layers to color")
            # print(f"[DEBUG] Layer names: {[lyr.name for lyr in newly_added_for_color]}")

            for layer_idx, lyr in enumerate(newly_added_for_color):
                # Assign colormap cyclically if we have more layers than colors
                if colors:
                    color_idx = layer_idx % len(colors)
                    try:
                        # print(f"[DEBUG] Setting layer {lyr.name} (idx {layer_idx}) to color {colors[color_idx]}")
                        lyr.colormap = colors[color_idx]
                    except Exception as e:
                        pass
                        # print(f"[DEBUG] Failed to set colormap for {lyr.name}: {e}")

        # Apply registration transform to layers from this store
        # Layer names are of the form f"{store_stem}:{pos}:{channel}" for images and
        # f"{store_stem}:grid:{pos}", f"{store_stem}:tile-ids:{pos}" for overlays
        newly_added_layers = [
            lyr for lyr in v.layers if lyr.name.startswith(store_path.stem)
        ]
        pos_list = (
            list(positions)
            if positions is not None
            else _iter_position_paths(store_path)
        )

        # Apply the global scale-up (e.g. 4x track/ISS) without a registration
        # affine — used both when no YAML exists and when its parse fails.
        def _apply_scale_only(p):
            if scale_yx is None or scale_yx == (1.0, 1.0):
                return
            prefixes = [f"{store_path.stem}:{x}" for x in (f"{p}:", f"grid:{p}", f"tile-ids:{p}")]
            for lyr in newly_added_layers:
                if any(lyr.name.startswith(pref) for pref in prefixes):
                    _apply_registration_affine_to_layer(lyr, None, scale_yx)

        # Try per-position YAMLs first (e.g., A1_pheno_register.yaml), then fall back to base YAML
        for p in pos_list:
            if store_key_for_yaml is not None and yaml_p is not None:
                try:
                    chosen = _find_per_well_yaml(ds, store_key_for_yaml, p)
                    # vprintf("[register] pos=%s using YAML=%s", str(p), str(chosen))
                    if chosen is not None:
                        try:
                            # T_reg is the 4x4 affine for ZYX, mapping source (5x) to target (downsampled 5x pheno)
                            T_reg = read_transform_biahub(chosen)

                            # vprintf("[register] pos=%s T_reg from YAML:\n%s", str(p), T_reg)

                            # Cell painting: YAML has cell_paint→pheno, we're overlaying cell_paint on pheno base
                            # So we need to invert to get the transform from cell_paint data coords to pheno world coords
                            # DISABLED: Testing if double-application causes misalignment in v3 stores
                            # if store_key_for_yaml in (
                            #     "lc_cell_painting1_register",
                            #     "lc_cell_painting2_register",
                            # ):
                            #     T_reg = np.linalg.inv(T_reg)
                            #     # vprintf("[register] pos=%s T_reg after invert:\n%s", str(p), T_reg)

                            # If overlay is ISS and base is Pheno, determine how to compose transforms:
                            # - If tracking data exists: compose inv(ISS->Track) then (Track->Pheno)
                            # - If no tracking data (skip_track mode or ops>=69 A/3/0): use ISS->Pheno directly
                            if str(store_path) == str(
                                ds.store_paths.get("iss_stitch_registered_v3")
                            ) and (str(base) == str(ds.store_paths.get("pheno_assembled_v3")) or str(base) == str(ds.store_paths.get("pheno_assembled"))):
                                has_tracking = _has_tracking_for_pos(experiment, p, ds)
                                vprintf("[register] pos=%s has_tracking=%s", str(p), str(has_tracking))

                                if has_tracking:
                                    # Normal case: tracking data exists
                                    # Load Track→Pheno transform and compose with ISS→Track
                                    ph_chosen = _find_per_well_yaml(ds, "auto_pheno_register", p)

                                    if ph_chosen is not None:
                                        # Compose: Track→Pheno @ inv(ISS→Track) to get ISS→Pheno
                                        T_track_to_pheno = read_transform_biahub(ph_chosen)
                                        T_reg = T_track_to_pheno @ np.linalg.inv(T_reg)
                                        vprintf("[register] ISS overlay: using inv(ISS→Track) then (Track→Pheno) for %s", str(p))
                                    else:
                                        vprintf("[register] WARNING: has_tracking=True but auto_pheno_register not found for %s", str(p))
                                else:
                                    # No tracking data: auto_iss_register already contains ISS→Pheno directly
                                    # (this happens with skip_track=True or for ops>=69 A/3/0)
                                    # YAML stores Pheno→ISS (biahub convention), invert to get ISS→Pheno
                                    T_reg = np.linalg.inv(T_reg)
                                    vprintf("[register] ISS overlay: using inv(direct) ISS→Pheno for %s (no tracking)", str(p))

                            # Find layers for this position
                            prefixes = [
                                f"{store_path.stem}:{p}:",
                                f"{store_path.stem}:grid:{p}",
                                f"{store_path.stem}:tile-ids:{p}",
                            ]

                            matched_layers = [lyr for lyr in newly_added_layers if any(lyr.name.startswith(pref) for pref in prefixes)]
                            vprintf("[register] pos=%s applying transform to %d layers: %s", str(p), len(matched_layers), [lyr.name for lyr in matched_layers])

                            for lyr in newly_added_layers:
                                if not any(
                                    lyr.name.startswith(pref) for pref in prefixes
                                ):
                                    continue

                                _apply_registration_affine_to_layer(
                                    lyr, T_reg, scale_yx, store_key_for_yaml
                                )

                        except Exception as _e_pos_yaml:
                            # A malformed/corrupt YAML must not drop the scale-up.
                            vprintf("[register] parse/compose failed for %s: %s — applying scale-only", str(chosen), str(_e_pos_yaml))
                            _apply_scale_only(p)
                    elif scale_yx is not None and scale_yx != (1.0, 1.0):
                        # No registration YAML found, but still apply the scale-up.
                        vprintf("[register] No YAML for pos=%s, applying scale-only (%.1f, %.1f)", str(p), scale_yx[0], scale_yx[1])
                        _apply_scale_only(p)
                except Exception:
                    pass

    # Load geff tracks last so all other layers appear first.
    # Tracks live in tracking (5x) coordinate space, so they need the same
    # transform as the tracking store overlay: auto_pheno_register YAML + 4x
    # upscale + well separation.  We run them through _apply_registration_affine_to_layer
    # so they get the exact same affine composition as image layers.
    if show_tracks:
        from cyclops_process.napari.dask.geff_utils import render_tracks_from_geff

        pos_list_for_tracks = (
            list(positions) if positions is not None else _iter_position_paths(stores[0])
        )

        # Tracks scale: 5x tracking → 20x pheno = 4x upscale (same as tracking store)
        tracks_scale_yx = (4.0, 4.0)

        for pos in pos_list_for_tracks:
            # Determine if tracking data exists for this position.
            # For wells without tracking (skip_track=True or ops>=69 A/3/0), fall back
            # to the ISS→Pheno registration so the tracks are still placed correctly.
            _has_track = _has_tracking_for_pos(experiment, pos, ds)
            vprintf("[geff-track] pos=%s has_tracking=%s", pos, str(_has_track))

            # Look up the appropriate registration YAML for this position.
            # - With tracking: use Track→Pheno (auto_pheno_register)
            # - Without tracking: coords are already in pheno space (per datasets.py),
            #   so no registration transform — only the 4x upscale (handled by _apply_registration_affine_to_layer)
            T_reg = None
            if _has_track:
                chosen_yaml = _find_per_well_yaml(ds, "auto_pheno_register", pos)
                if chosen_yaml is not None:
                    try:
                        T_reg = read_transform_biahub(chosen_yaml)
                        vprintf("[geff-track] Using Track→Pheno YAML %s for %s", str(chosen_yaml), pos)
                    except Exception as _e_yaml:
                        vprintf("[geff-track] Failed to read registration YAML: %s", str(_e_yaml))
            else:
                vprintf("[geff-track] No tracking for %s: coords already in pheno space, using identity (4x upscale only)", pos)

            # 1) Current tracks (from tracking_geff)
            try:
                vprintf("[geff-track] Attempting to render tracks for %s", pos)
                tracks_layer = render_tracks_from_geff(
                    experiment, pos, offsets_x, v
                )
                if tracks_layer is not None:
                    _apply_registration_affine_to_layer(
                        tracks_layer, T_reg, tracks_scale_yx
                    )
                    vprintf("[geff-track] Applied registration affine to tracks for %s", pos)
            except Exception as e:
                vprintf("[geff-track] Skipping tracks for %s: %s", pos, str(e))

    napari.run()
    return v


def render_grid_overlay(source_store, pos, offsets_x, v, store_tag: Optional[str] = None):
    """
    Render grid overlay as a single RGBA image layer with blue lines and tile ID text.

    Tries to load:
    1. New format: grid_overlay (RGBA image with embedded text)
    2. Legacy format: grid_edges (grayscale) - for backward compatibility
    """
    # Try new grid_overlay format first (v2 and v3)
    grid_path = None
    v2_path = source_store / pos / "grid_overlay"
    v3_path = source_store / pos / "labels" / "grid_overlay"

    if v2_path.exists() and v2_path.is_dir():
        try:
            levels = _list_numeric_level_names(v2_path)
            if levels:
                grid_path = Path(pos) / "grid_overlay"
        except Exception:
            pass

    if grid_path is None and v3_path.exists() and v3_path.is_dir():
        try:
            levels = _list_numeric_level_names(v3_path)
            if levels:
                grid_path = Path(pos) / "labels" / "grid_overlay"
        except Exception:
            pass

    # If new format found, render as RGBA image
    if grid_path is not None:
        levels = _list_numeric_level_names(source_store / grid_path)
        if levels:
            grid_arrays = [
                da.from_zarr(
                    str(source_store), component=str(grid_path / lvl)
                )
                for lvl in levels
            ]
            translate_tuple = (0.0, float(offsets_x.get(pos, 0)))

            v.add_image(
                grid_arrays,
                multiscale=True,
                blending="additive",
                opacity=0.85,
                name=(f"{store_tag}:grid:{pos}" if store_tag else f"grid:{pos}"),
                translate=translate_tuple,
                visible=False,
                rgb=True,
            )
            return

    # Fall back to legacy grid_edges format (grayscale)
    grid_edges_path = None
    v2_edges = source_store / pos / "grid_edges"
    v3_edges = source_store / pos / "labels" / "grid_edges"

    if v2_edges.exists() and v2_edges.is_dir():
        try:
            edge_levels = _list_numeric_level_names(v2_edges)
            if edge_levels:
                grid_edges_path = Path(pos) / "grid_edges"
        except Exception:
            pass

    if grid_edges_path is None and v3_edges.exists() and v3_edges.is_dir():
        try:
            edge_levels = _list_numeric_level_names(v3_edges)
            if edge_levels:
                grid_edges_path = Path(pos) / "labels" / "grid_edges"
        except Exception:
            pass

    if grid_edges_path is None:
        return

    edge_levels = _list_numeric_level_names(source_store / grid_edges_path)
    if edge_levels:
        edge_arrays = [
            da.from_zarr(
                str(source_store), component=str(grid_edges_path / lvl)
            )
            for lvl in edge_levels
        ]
        translate_tuple = (0.0, float(offsets_x.get(pos, 0)))

        v.add_image(
            edge_arrays,
            multiscale=True,
            blending="additive",
            opacity=0.85,
            contrast_limits=(0, 255),
            colormap="cyan",
            name=(f"{store_tag}:grid:{pos}" if store_tag else f"grid:{pos}"),
            translate=translate_tuple,
            visible=False,
            gamma=0.75,
        )


def render_grid_props(
    source_store, pos, show_tile_numbers, offsets_x, v, store_tag: Optional[str] = None
):
    """
    Render tile IDs as a points layer with labels (legacy format only).

    NOTE: This is for backward compatibility with old grid_props format.
    New grid_overlay format has text embedded in the RGBA image.
    """
    if not show_tile_numbers:
        return

    # Check if new grid_overlay exists - if so, skip props rendering (text is embedded)
    v2_overlay = source_store / pos / "grid_overlay"
    v3_overlay = source_store / pos / "labels" / "grid_overlay"
    if v2_overlay.exists() or v3_overlay.exists():
        return

    # Try v2 style first: pos/grid_props, then v3 style: pos/labels/grid_props
    props_path = None
    v2_path = source_store / pos / "grid_props"
    v3_path = source_store / pos / "labels" / "grid_props"

    if v2_path.exists() and v2_path.is_dir():
        props_path = Path(pos) / "grid_props"
    elif v3_path.exists() and v3_path.is_dir():
        props_path = Path(pos) / "labels" / "grid_props"
    else:
        return

    try:
        coords_da = da.from_zarr(
            str(source_store), component=str(props_path / "coords_yx")
        )
        names_da = da.from_zarr(str(source_store), component=str(props_path / "name"))
        coords = coords_da.compute()
        names = names_da.compute()

        if coords.ndim == 2 and coords.shape[0] > 0:
            viewer_offset_yx = (0.0, float(offsets_x.get(pos, 0)))
            pts = v.add_points(
                coords,
                name=(
                    f"{store_tag}:tile-ids:{pos}" if store_tag else f"tile-ids:{pos}"
                ),
                size=1,
                face_color="transparent",
                border_color="transparent",
                properties={"id": names.astype("U")},
                blending="additive",
                translate=viewer_offset_yx,
                visible=False,
            )
            pts.text = {
                "string": "{id}",
                "size": 12,
                "color": "yellow",
                "anchor": "upper_left",
            }

            vprintf("Added numeric tile ID points for %s (hidden by default)", pos)
    except Exception as e:
        vprintf("[grid_props] Could not add tile points for %s: %s", pos, str(e))


def render_iss_image(source_store, pos, offsets_x, v):
    """Render ISS overlays for a position.

    Supports both new and legacy naming conventions:
    - New: iss_gene_image, iss_guide_image (self-contained RGBA images with text baked in)
    - Legacy: iss_points, iss_points_props (grayscale + separate props for napari text)
    """
    offset = (0.0, 0.0)
    translate_tuple = (offset[0], float(offsets_x.get(pos, 0)) + offset[1])

    # Helper to find an ISS overlay by name (checks v2 and v3 locations)
    def _find_iss_overlay(name: str):
        v2_path = source_store / pos / name
        v3_path = source_store / pos / "labels" / name
        for path, rel_path in [(v2_path, Path(pos) / name), (v3_path, Path(pos) / "labels" / name)]:
            if path.exists() and path.is_dir():
                try:
                    lvl_names = _list_numeric_level_names(path)
                    if lvl_names:
                        return rel_path, lvl_names
                except Exception:
                    pass
        return None, None

    # Try new naming convention first (iss_gene_image, iss_guide_image)
    # These are self-contained RGBA images with text baked in - no props needed
    gene_path, gene_levels = _find_iss_overlay("iss_gene_image")
    guide_path, guide_levels = _find_iss_overlay("iss_guide_image")

    if gene_path is not None:
        vprintf("[ISS overlay] Found new-style iss_gene_image for %s", pos)
        arrays = [
            da.from_zarr(str(source_store), component=str(gene_path / lvl))
            for lvl in gene_levels
        ]
        v.add_image(
            arrays,
            multiscale=True,
            blending="translucent",
            opacity=0.9,
            name=f"ISS-gene:{pos}",
            translate=translate_tuple,
            visible=False,
            rgb=True,
        )

    if guide_path is not None:
        vprintf("[ISS overlay] Found new-style iss_guide_image for %s", pos)
        arrays = [
            da.from_zarr(str(source_store), component=str(guide_path / lvl))
            for lvl in guide_levels
        ]
        v.add_image(
            arrays,
            multiscale=True,
            blending="translucent",
            opacity=0.9,
            name=f"ISS-guide:{pos}",
            translate=translate_tuple,
            visible=False,
            rgb=True,
        )

    # If new-style overlays found, we're done (no legacy props needed)
    if gene_path is not None or guide_path is not None:
        return

    # Fall back to legacy naming convention (iss_points, iss_points_props)
    iss_points_path, lvl_names = _find_iss_overlay("iss_points")
    if iss_points_path is None:
        vprintf("[ISS overlay] No ISS overlay found for %s", pos)
        return

    vprintf("[ISS overlay] Found legacy iss_points for %s", pos)
    arrays = [
        da.from_zarr(str(source_store), component=str(iss_points_path / lvl))
        for lvl in lvl_names
    ]

    # Detect if this is RGBA (shape ends with 4) or grayscale
    # RGBA images have colored dots + text rendered directly
    is_rgba = False
    if arrays and len(arrays) > 0:
        sample_shape = arrays[0].shape
        if len(sample_shape) >= 3 and sample_shape[-1] == 4:
            is_rgba = True
            vprintf("[ISS overlay] Detected RGBA ISS overlay for %s", pos)

    if is_rgba:
        # RGBA: napari displays directly with embedded colors + text
        v.add_image(
            arrays,
            multiscale=True,
            blending="translucent",
            opacity=0.9,
            name=f"ISS-img:{pos}",
            translate=translate_tuple,
            visible=False,
            rgb=True,
        )
    else:
        # Grayscale (legacy format): use colormap
        v.add_image(
            arrays,
            multiscale=True,
            blending="additive",
            opacity=0.7,
            contrast_limits=(0, 255),
            colormap="magenta",
            name=f"ISS-img:{pos}",
            translate=translate_tuple,
            visible=False,
            gamma=0.75,
        )

    # Load label props once at a single level (0 preferred, else 1) and render all labels (no bbox filtering).
    # Try v2 style first: pos/iss_points_props, then v3 style: pos/labels/iss_points_props
    iss_props_path, props_levels = _find_iss_overlay("iss_points_props")
    if iss_props_path is None:
        vprintf("[ISS props] no iss_points_props found for %s (new-style overlays have text baked in)", str(pos))
        return

    level_to_use = 0 if "0" in props_levels else (1 if "1" in props_levels else None)
    if level_to_use is None:
        vprintf(
            "[ISS props] neither level 0 nor 1 available for %s (found: %s)",
            str(pos),
            ",".join(props_levels),
        )
        return
    vprintf("[ISS props] simple render: using level=%d (load all)", int(level_to_use))
    c_da = da.from_zarr(
        str(source_store),
        component=str(iss_props_path / str(level_to_use) / "coords_yx"),
    )
    l_da = da.from_zarr(
        str(source_store),
        component=str(iss_props_path / str(level_to_use) / "labels"),
    )
    coords = np.asarray(c_da.compute())
    labels = np.asarray(l_da.compute()).astype("U")
    vprintf(
        "[ISS props] loaded %d labels at level=%d (no bbox)",
        int(coords.shape[0] if coords.ndim == 2 else 0),
        int(level_to_use),
    )
    if coords.ndim == 2 and coords.shape[0] == labels.shape[0] and coords.shape[1] == 2:
        pts = v.add_points(
            data=coords,
            name=f"ISS:{pos}",
            size=12,
            translate=translate_tuple,
            face_color=[1.0, 0.0, 1.0, 0.15],
            border_color=[1.0, 0.0, 1.0, 0.6],
            properties={"label": labels},
            blending="translucent",
            visible=False,
        )
        pts.text = {
            "string": "{label}",
            "anchor": "upper_left",
            "color": "white",  # High contrast white text
            "size": 24,  # Larger text size
        }
        # Note: Napari doesn't directly support text backgrounds, but we can make text more readable
        # by using high-contrast white text on semi-transparent point backgrounds
        vprintf("[ISS props] points layer added with %d labels", int(labels.shape[0]))
    else:
        vprintf(
            "[ISS props] unexpected coords/labels shapes: coords=%s labels=%s",
            str(getattr(coords, "shape", None)),
            str(getattr(labels, "shape", None)),
        )


def build_and_view(
    experiment: Optional[str] = None,
    source_store: Optional[str | Path] = None,
    store_key: Optional[str] = None,
    num_levels: int = 5,
    factor: int = 2,
    contrast_limits: Optional[Sequence[float]] = None,
    channel_axis: Optional[int] = None,
    name: Optional[str] = None,
    with_grid: bool = False,
    grid_line_width: int = 3,
    show_grid_edges: bool = True,
    show_grid_ids: bool = True,
):
    """
    Convenience wrapper: build (or ensure) a pyramid and launch napari.
    """
    if experiment is not None:
        ds = OpsDataset(experiment)
        # Default to phenotyping stitched 2D recon unless a specific key is requested
        chosen_key = store_key or "lc_20x_phase_2d_stitched"
        if chosen_key not in ds.store_paths:
            raise KeyError(f"Unknown store_key '{chosen_key}'.")
        source_store = source_store or ds.store_paths[chosen_key]
    if source_store is None:
        raise ValueError("Either experiment or source_store must be provided")

    vprintf(
        "Build-and-view: experiment=%s, source=%s, levels=%d, factor=%d",
        str(experiment),
        str(source_store),
        num_levels,
        factor,
    )
    build_pyramid_in_place(source_store, levels=num_levels, factor=factor)
    if with_grid:
        mode = "pheno"
        if "iss" in chosen_key:
            mode = "iss"
        elif "track" in chosen_key or "lc_5x" in chosen_key:
            mode = "track"

        build_grid_overlay_in_place(
            source_store=source_store,
            positions=None,
            line_width_px=grid_line_width,
            stitch_config_path=_resolve_stitch_config_path(experiment, mode),
            dataset=ds if experiment is not None else None,
        )
    return view_inplace_pyramid_in_napari(
        source_store=source_store,
        contrast_limits=contrast_limits,
        channel_axis=channel_axis,
        name=name,
        show_grid_edges=show_grid_edges,
        show_grid_ids=show_grid_ids,
        show_exp_grids=False,
    )


__all__ = [
    "build_pyramid_in_place",
    "view_inplace_pyramid_in_napari",
    "build_and_view",
    "build_grid_overlay_in_place",
]


def main():
    """
    CLI entrypoint to build and view a multiscale pyramid for an OPS experiment.
    Lets the user choose phenotyping (20x) or tracking (5x) stitched 2D recon stores.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Build and/or view in-place multiscale pyramid in napari for OPS experiments."
    )
    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        required=True,
        help="Experiment name or shorthand (e.g., '33' for ops0033_20250429)",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["pheno", "track", "iss", "all", "cell_paint", "bf"],
        default="all",
        help="Dataset mode: 'pheno' (20x), 'track' (5x), 'iss' (bc_stitched.zarr), 'all' (pheno+track+iss), "
             "'cell_paint', or 'bf' (pheno v3 + the BF-slice titration v3 store overlaid in one viewer)",
    )
    parser.add_argument(
        "--action",
        "-a",
        type=str,
        choices=[
            "build",
            "view",
            "both",
            "build-grid",
            "verify-grid",
            "build-clim",
            "build-iss",
            "build-seg",
        ],
        default="view",
        help="Choose: build pyramid, view, both, build grid overlay, verify overlays, build per-level contrast limits, build ISS overlay, or rebuild segmentation pyramids only",
    )
    parser.add_argument(
        "--num-levels", type=int, default=5, help="Number of pyramid levels to build"
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=2,
        help="Spatial downsampling factor between consecutive levels",
    )
    parser.add_argument(
        "--well",
        "-w",
        type=str,
        default=None,
        help="Single well identifier (e.g., '1' for A/1, 'B2' for B/2)",
    )
    parser.add_argument(
        "--wells",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of well keys to process (e.g., A/1 B/2). Default: all wells",
    )
    parser.add_argument(
        "--contrast-limits",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Optional contrast limits [min max]",
    )
    parser.add_argument(
        "--channel-axis",
        type=int,
        default=None,
        help="Override inferred channel axis (OME-Zarr inferred by default)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose (DEBUG) logging"
    )
    # Grid overlay options
    parser.add_argument(
        "--with-grid",
        action="store_true",
        help="When building, also build the grid overlay",
    )
    parser.add_argument(
        "--grid-line-width", type=int, default=3, help="Grid line width in pixels"
    )
    parser.add_argument(
        "--skip-overlays",
        action="store_true",
        help="Skip building grid_edges, grid_props, iss_points, and iss_points_props overlays (only build base images and segmentation pyramids)",
    )
    parser.add_argument(
        "--show-exp-grids",
        action="store_true",
        help="Overlay experiment grid circles and labels inferred from available configs",
    )
    parser.add_argument(
        "--cell-paint",
        action="store_true",
        help="Overlay Cell Painting label and two OME-TIFF layers to the right-adjacent well",
    )
    parser.add_argument(
        "--show-tracks",
        action="store_true",
        default=True,
        help="Overlay tracks from geff files (default: True)",
    )
    parser.add_argument(
        "--no-tracks",
        action="store_false",
        dest="show_tracks",
        help="Disable tracks overlay",
    )
    parser.add_argument(
        "--pheno-version",
        type=str,
        choices=["v2", "v3"],
        default="v3",
        help="Phenotyping assembled zarr version (default: v3)",
    )
    parser.add_argument(
        "--no-organelle-labels",
        action="store_true",
        help="Disable auto-loading of organelle segmentation labels (*_seg in labels/)",
    )
    parser.add_argument(
        "--extra-stores",
        type=str,
        nargs="*",
        default=None,
        help="Extra v3 zarr store path(s) to overlay on the (mode) viewer in the same "
             "coordinate frame (no registration, no tracks). Like --mode bf but for any "
             "store stitched on the same canvas.",
    )
    # Overlays now auto-display when present; keeping flags minimal
    # Removed --position for simplicity; viewer opens the first processed position

    args = parser.parse_args()

    # Configurevprintf verbosity
    # Set global verbosity used by vprintf across modules
    try:
        from cyclops_utils.data import filesystem as _fs

        _fs.VERBOSE = bool(args.verbose)
    except Exception:
        pass

    # Resolve experiment shorthand to full name
    experiment = resolve_experiment_name(args.experiment, verbose=args.verbose, autoselect=True)

    # Convert single --well argument to --wells format if provided
    if args.well is not None and args.wells is None:
        args.wells = [canonicalize_well_path(args.well)]

    # Map mode to the appropriate store key or directory enumeration
    ds = OpsDataset(experiment)
    
    # Determine pheno store key based on version argument
    pheno_store_key = f"pheno_assembled_{args.pheno_version}" if args.pheno_version == "v3" else "pheno_assembled"
    
    if args.mode == "pheno":
        store_key = pheno_store_key
        display_name = f"{experiment} {args.mode} assembled ({args.pheno_version})"
        if store_key not in ds.store_paths:
            raise KeyError(
                f"store_key '{store_key}' not found. Available: {sorted(ds.store_paths)}"
            )
        source_store = ds.store_paths[store_key]
        target_stores = [source_store]
    elif args.mode == "track":
        store_key = "lc_5x_phase_2d_stitched_v3"
        display_name = f"{experiment} {args.mode} 2D stitched"
        if store_key not in ds.store_paths:
            raise KeyError(
                f"store_key '{store_key}' not found. Available: {sorted(ds.store_paths)}"
            )
        source_store = ds.store_paths[store_key]
        target_stores = [source_store]
    elif args.mode == "iss":
        store_key = "iss_stitch_registered_v3"
        display_name = f"{experiment} {args.mode} stitched"
        if store_key not in ds.store_paths:
            raise KeyError(
                f"store_key '{store_key}' not found. Available: {sorted(ds.store_paths)}"
            )
        source_store = ds.store_paths[store_key]
        target_stores = [source_store]
    elif args.mode == "all":
        display_name = f"{experiment} all modes ({args.pheno_version})"
        keys = [
            pheno_store_key,
            "lc_5x_phase_2d_stitched_v3",
            "iss_stitch_registered_v3",
        ]
        target_stores = [ds.store_paths[k] for k in keys if k in ds.store_paths]

        # For experiment 72, also include cell painting max projection stitched stores
        if "ops0072_20250904" in experiment:
            part1_path = (
                ds.experiment_path
                / "0-convert"
                / "cell_painting"
                / "part1_max_proj_stitched_v3.zarr"
            )
            part2_path = (
                ds.experiment_path
                / "0-convert"
                / "cell_painting"
                / "part2_max_proj_stitched_v3.zarr"
            )
            if part1_path.exists() and part1_path.is_dir():
                target_stores.append(part1_path)
                print(
                    f"Added Cell Painting 1 part1_max_proj_stitched_v3.zarr to target_stores: {part1_path}"
                )
            if part2_path.exists() and part2_path.is_dir():
                target_stores.append(part2_path)
                print(
                    f"Added Cell Painting 2 part2_max_proj_stitched_v3.zarr to target_stores: {part2_path}"
                )

        if not target_stores:
            raise KeyError(
                f"No stores found for 'all' mode (need at least one of {pheno_store_key}, lc_5x_phase_2d_stitched_v3, iss_stitch_registered_v3)"
            )
    elif args.mode == "bf":
        # BF-slice titration QC: overlay the production pheno v3 (Phase2D/Focus3D +
        # cell masks) with the sibling BF-slice v3 (BF_z0..z6). Both were stitched
        # with the same shifts, so they live in the identical coordinate frame and
        # overlay directly — no registration needed.
        display_name = f"{experiment} pheno+BF slices ({args.pheno_version})"
        bf_key = "bf_slices_assembled_v3"
        keys = [pheno_store_key, bf_key]
        missing = [k for k in keys
                   if k not in ds.store_paths or not Path(ds.store_paths[k]).exists()]
        if missing:
            raise KeyError(
                f"'bf' mode needs both stores built: {keys}. Missing/not found: {missing}"
            )
        target_stores = [ds.store_paths[k] for k in keys]
    else:
        # cell_paint: enumerate all Zarr stores under cell_painting/stitch
        display_name = f"{experiment} cell_paint assembled"

        # For experiment 72, use the part1/part2 stores from 0-convert instead of cell_painting/stitch
        if "ops0072_20250904" in experiment:
            target_stores = []
            part1_path = (
                ds.experiment_path
                / "0-convert"
                / "cell_painting"
                / "part1_max_proj_stitched_v3.zarr"
            )
            part2_path = (
                ds.experiment_path
                / "0-convert"
                / "cell_painting"
                / "part2_max_proj_stitched_v3.zarr"
            )
            if part1_path.exists() and part1_path.is_dir():
                target_stores.append(part1_path)
            if part2_path.exists() and part2_path.is_dir():
                target_stores.append(part2_path)
        else:
            # For other experiments, enumerate from cell_painting/stitch
            base_dir = ds.experiment_path / "cell_painting" / "stitch"
            target_stores = sorted([p for p in base_dir.glob("*.zarr") if p.is_dir()])

        if not target_stores:
            raise FileNotFoundError(f"No cell_paint Zarr stores found for {experiment}")

    # Append arbitrary extra stores to overlay in the same coordinate frame (like
    # --mode bf, but for any v3 store stitched on the same canvas).
    if args.extra_stores:
        for extra in args.extra_stores:
            ep = Path(extra)
            if not ep.exists():
                raise FileNotFoundError(f"--extra-stores path not found: {ep}")
            target_stores.append(ep)
        display_name = f"{display_name} + {len(args.extra_stores)} extra"

    # Resolve wells to process (per store). For multi-store (cell_paint), this is computed inside loops.
    if len(target_stores) == 1:
        source_store = target_stores[0]
        all_wells = _iter_position_paths(source_store)
        if args.wells:
            selected_positions = [
                w
                for w in all_wells
                if any(str(w).startswith(str(sel)) for sel in args.wells)
            ]
        else:
            selected_positions = all_wells
        vprintf("Selected %d/%d wells", len(selected_positions), len(all_wells))

    if args.action in ("build", "both"):
        # Loop over one or many target stores depending on mode
        for src in target_stores:
            wells_this = _iter_position_paths(src)
            if len(target_stores) == 1:
                positions_this = selected_positions
            else:
                if args.wells:
                    positions_this = [
                        w
                        for w in wells_this
                        if any(str(w).startswith(str(sel)) for sel in args.wells)
                    ]
                else:
                    positions_this = wells_this
                vprintf(
                    "[cell_paint] %s: Selected %d/%d wells",
                    str(src),
                    len(positions_this),
                    len(wells_this),
                )

            build_pyramid_in_place(
                source_store=src,
                levels=args.num_levels,
                factor=args.factor,
                positions=positions_this,
            )
            _prune_extra_levels(src, positions_this, keep_count=args.num_levels)
            build_clims_in_place(
                source_store=src,
                positions=positions_this,
                scale_factor=args.factor,
            )
            # Build overlays unless --skip-overlays flag is set
            if args.with_grid and not args.skip_overlays:
                # Determine per-store mode for stitch config resolution
                if args.mode in ("all", "cell_paint"):
                    _src_str = str(src)
                    if _src_str == str(ds.store_paths.get("iss_stitch_registered_v3")):
                        _build_mode = "iss"
                    elif _src_str == str(ds.store_paths.get("lc_5x_phase_2d_stitched_v3")):
                        _build_mode = "track"
                    else:
                        _build_mode = "pheno"
                else:
                    _build_mode = args.mode
                build_grid_overlay_in_place(
                    source_store=src,
                    positions=positions_this,
                    line_width_px=args.grid_line_width,
                    stitch_config_path=_resolve_stitch_config_path(
                        experiment, _build_mode
                    ),
                    dataset=ds,
                )
                build_iss_overlay_in_place(
                    source_store=src,
                    experiment=experiment,
                    positions=positions_this,
                )
            elif args.skip_overlays:
                print(f"Skipping overlay builds (grid_edges, grid_props, iss_points, iss_points_props) for {src.name}")

    if args.action == "build-grid":
        for src in target_stores:
            wells_this = _iter_position_paths(src)
            if len(target_stores) == 1:
                positions_this = selected_positions
            else:
                if args.wells:
                    positions_this = [
                        w
                        for w in wells_this
                        if any(str(w).startswith(str(sel)) for sel in args.wells)
                    ]
                else:
                    positions_this = wells_this

            # Determine mode from store path when args.mode == "all"
            if args.mode == "all":
                src_str = str(src)
                if src_str == str(ds.store_paths.get("iss_stitch_registered_v3")):
                    store_mode = "iss"
                elif src_str == str(ds.store_paths.get("lc_5x_phase_2d_stitched_v3")):
                    store_mode = "track"
                else:
                    store_mode = "pheno"  # default fallback
            elif args.mode == "cell_paint":
                store_mode = "pheno"
            else:
                store_mode = args.mode

            build_grid_overlay_in_place(
                source_store=src,
                positions=positions_this,
                line_width_px=args.grid_line_width,
                stitch_config_path=_resolve_stitch_config_path(experiment, store_mode),
                dataset=ds,
            )
            build_iss_overlay_in_place(
                source_store=src,
                experiment=experiment,
                positions=positions_this,
            )

    if args.action == "build-clim":
        for src in target_stores:
            wells_this = _iter_position_paths(src)
            if len(target_stores) == 1:
                positions_this = selected_positions
            else:
                if args.wells:
                    positions_this = [
                        w
                        for w in wells_this
                        if any(str(w).startswith(str(sel)) for sel in args.wells)
                    ]
                else:
                    positions_this = wells_this

            # Reset the global flag to allow printing clims table for each store
            import cyclops_process.processes.pyramids.build_dask as build_module

            build_module.CLIMS_REPORT_PRINTED = False

            build_clims_in_place(
                source_store=src,
                positions=positions_this,
                scale_factor=args.factor,
            )

    if args.action == "build-iss":
        for src in target_stores:
            wells_this = _iter_position_paths(src)
            if len(target_stores) == 1:
                positions_this = selected_positions
            else:
                if args.wells:
                    positions_this = [
                        w
                        for w in wells_this
                        if any(str(w).startswith(str(sel)) for sel in args.wells)
                    ]
                else:
                    positions_this = wells_this
            build_iss_overlay_in_place(
                source_store=src,
                experiment=experiment,
                positions=positions_this,
            )

    if args.action == "build-seg":
        print(f"Rebuilding segmentation pyramids only (levels={args.num_levels})...")
        for src in target_stores:
            wells_this = _iter_position_paths(src)
            if len(target_stores) == 1:
                positions_this = selected_positions
            else:
                if args.wells:
                    positions_this = [
                        w
                        for w in wells_this
                        if any(str(w).startswith(str(sel)) for sel in args.wells)
                    ]
                else:
                    positions_this = wells_this
            print(f"\nRebuilding segmentation pyramid for {src.name}...")
            build_seg_pyramid_only(
                source_store=src,
                levels=args.num_levels,
                positions=positions_this,
                resume=True,
            )
            print(f"✓ Completed segmentation pyramid rebuild for {src.name}")

    if args.action == "verify-grid":
        # Verify presence and basic stats of grid overlays
        import pprint as _pp

        report = {}
        for src in target_stores:
            wells_this = _iter_position_paths(src)
            if len(target_stores) == 1:
                positions_this = selected_positions
            else:
                if args.wells:
                    positions_this = [
                        w
                        for w in wells_this
                        if any(str(w).startswith(str(sel)) for sel in args.wells)
                    ]
                else:
                    positions_this = wells_this
            for pos in positions_this:
                pos_dir = Path(src) / pos
                edges_dir = pos_dir / "grid_edges"
                ids_dir = pos_dir / "grid_ids"
                info = {"edges_levels": [], "ids_shape": None, "ids_nonzero": None}
                if edges_dir.exists():
                    info["edges_levels"] = _list_numeric_level_names(edges_dir)
                if ids_dir.exists():
                    try:
                        ids_da = da.from_zarr(
                            str(src), component=str(Path(pos) / "grid_ids")
                        )
                        info["ids_shape"] = tuple(int(s) for s in ids_da.shape)
                        nz = int(da.count_nonzero(ids_da).compute())
                        info["ids_nonzero"] = nz
                    except Exception:
                        pass
                report[f"{Path(str(src)).name}:{pos}"] = info
        print("Grid verification report:")
        _pp.pprint(report)
        return

    if args.action in ("view", "both"):
        # For view-only, do not implicitly build; require pyramid to exist
        try:
            if args.mode == "cell_paint":
                # Overlay view across multiple Zarr stores with code-defined per-overlay offsets
                # Include the experiment's assembled phenotyping store as an additional overlay if present
                # Determine base well selection from the first store to respect --wells
                try:
                    base_store_for_view = target_stores[0]
                except Exception:
                    base_store_for_view = source_store
                base_wells = _iter_position_paths(base_store_for_view)
                if args.wells:
                    selected_positions = [
                        w
                        for w in base_wells
                        if any(str(w).startswith(str(sel)) for sel in args.wells)
                    ]
                else:
                    selected_positions = base_wells
                # Respect --wells selection even when multiple stores are shown
                pos_for_view = (
                    selected_positions
                    if args.wells
                    else (None if len(target_stores) > 1 else selected_positions)
                )
                stores_for_view = list(target_stores)
                try:
                    assembled = ds.store_paths.get(pheno_store_key)
                    if assembled is not None and Path(assembled).exists():
                        # Append only if it's multiscale for at least one requested well
                        wells_check = pos_for_view or _iter_position_paths(assembled)
                        wells_check = (
                            list(wells_check) if wells_check is not None else []
                        )
                        if not wells_check:
                            wells_check = _iter_position_paths(assembled)
                        is_ms_any = False
                        for _w in wells_check:
                            lvl_names = _list_numeric_level_names(Path(assembled) / _w)
                            if len(lvl_names) > 1:
                                is_ms_any = True
                                break
                        if is_ms_any:
                            stores_for_view.append(Path(assembled))
                        else:
                            vprintf("Skipping pheno_assembled overlay (not multiscale)")
                except Exception:
                    pass

                # Lazy import to avoid circular import at module import time
                from cyclops_process.napari.dask.cell_paint import view_cell_paint_overlays  # type: ignore

                view_cell_paint_overlays(
                    stores=stores_for_view,
                    positions=pos_for_view,
                    contrast_limits=args.contrast_limits,
                    channel_axis=args.channel_axis,
                    name=display_name,
                    show_exp_grids=bool(args.show_exp_grids),
                )
            elif args.mode == "bf" or args.extra_stores:
                # pheno v3 + extra v3 store(s) in one shared viewer. Same coordinate
                # frame (stitched with the same shifts / same affine), so load all
                # with view_inplace into a shared viewer — no registration, no tracks.
                base_wells = _iter_position_paths(target_stores[0])
                if args.wells:
                    selected_positions = [
                        w for w in base_wells
                        if any(str(w).startswith(str(sel)) for sel in args.wells)
                    ]
                else:
                    selected_positions = base_wells
                v = None
                for i, src in enumerate(target_stores):
                    v = view_inplace_pyramid_in_napari(
                        source_store=src,
                        positions=selected_positions,
                        contrast_limits=args.contrast_limits,
                        channel_axis=args.channel_axis,
                        name=f"{experiment} {Path(src).stem}",
                        show_grid_edges=True,
                        show_grid_ids=True,
                        show_tile_numbers=True,
                        show_exp_grids=bool(args.show_exp_grids),
                        mode="pheno",
                        experiment=experiment,
                        show_tracks=False,
                        # Load cell/organelle masks once, from the prod store only
                        # (the BF store symlinks the same labels — avoid duplicates).
                        show_organelle_labels=(i == 0) and (not args.no_organelle_labels),
                        v=v,
                        run_app=(i == len(target_stores) - 1),
                    )
            elif args.mode == "all":
                # Overlay pheno + track + iss using registration YAMLs to align to pheno
                base_store_for_view = target_stores[0]
                base_wells = _iter_position_paths(base_store_for_view)
                if args.wells:
                    selected_positions = [
                        w
                        for w in base_wells
                        if any(str(w).startswith(str(sel)) for sel in args.wells)
                    ]
                else:
                    selected_positions = base_wells
                view_registered_stores_in_one_viewer(
                    experiment=experiment,
                    stores=target_stores,
                    positions=selected_positions,
                    contrast_limits=args.contrast_limits,
                    channel_axis=args.channel_axis,
                    name=display_name,
                    show_exp_grids=bool(args.show_exp_grids),
                    show_tracks=args.show_tracks,
                    show_organelle_labels=not args.no_organelle_labels,
                )
            else:
                # Single-store view
                src_for_view = target_stores[0]
                pos_for_view = None
                if len(target_stores) == 1:
                    src_for_view = target_stores[0]
                    pos_for_view = selected_positions

                scale_yx = (4.0, 4.0) if args.mode in ("track", "iss") else None

                view_inplace_pyramid_in_napari(
                    source_store=src_for_view,
                    contrast_limits=args.contrast_limits,
                    channel_axis=args.channel_axis,
                    name=display_name,
                    positions=pos_for_view,
                    show_grid_edges=True,
                    show_grid_ids=True,
                    show_tile_numbers=True,
                    show_exp_grids=bool(args.show_exp_grids),
                    cell_paint=bool(args.cell_paint),
                    mode=args.mode if isinstance(args.mode, str) else None,
                    experiment=experiment,
                    show_tracks=bool(args.show_tracks),
                    show_organelle_labels=not args.no_organelle_labels,
                    # scale_yx=scale_yx,
                )
        except FileNotFoundError as e:
            raise SystemExit(
                f"Pyramid not found. Run with --action build or both first. Details: {e}"
            )


if __name__ == "__main__":
    main()


# usage: python cyclops_process/napari/dask/view_dask.py --experiment ops0033_20250429 --mode pheno --action build --num-levels 5 --verbose --with-grid
# usage: python cyclops_process/napari/dask/view_dask.py --experiment ops0033_20250429 --mode pheno --action view --num-levels 5 --verbose
# usage: python cyclops_process/napari/dask/view_dask.py -e ops0033_20250429 -m pheno -a build-grid --wells A/1
# usage: python cyclops_process/napari/dask/view_dask.py -e ops0033_20250429 -m pheno -a build-iss --pheno-version v3
# usage: python cyclops_process/napari/dask/view_dask.py -e ops0033_20250429 -m pheno -a build-clim
# usage: python cyclops_process/napari/dask/view_dask.py -e ops0033_20250429 -m pheno -a build-seg --num-levels 5
# usage: python cyclops_process/napari/dask/view_dask.py -e ops0033_20250429 -m all -a build-seg --wells A/1
# usage: python cyclops_process/napari/dask/view_dask.py -e ops0072_20250904 -m cell_paint -a build-clim --num-levels 5 --with-grid
