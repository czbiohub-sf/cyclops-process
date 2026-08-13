from pathlib import Path
from typing import Sequence, Tuple

import shutil
import sys
import os
import numpy as np
import yaml

sys.path.insert(0, os.getcwd())

from cyclops_utils.data.filesystem import vprintf


def _list_numeric_level_names(position_dir: Path) -> list[str]:
    """Return sorted list of existing numeric level names under a position directory."""
    level_names: list[str] = []
    if not position_dir.exists():
        return level_names
    for child in sorted(position_dir.iterdir()):
        if child.name.isdigit() and (child.is_dir() or child.is_symlink()):
            level_names.append(child.name)
    return level_names


def _load_total_translations(stitch_config_path: Path) -> dict:
    try:
        with open(stitch_config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("total_translation", {}) or {}
    except Exception:
        return {}


def _prune_extra_levels(
    source_store: Path | str, positions: Sequence[str], keep_count: int
) -> None:
    """Delete numeric level folders >= keep_count for each position.

    - Never delete level "0" (original data)
    - Also prune matching levels in grid_edges
    """
    root = Path(source_store)
    keep_levels = {str(i) for i in range(max(0, int(keep_count)))}
    keep_levels.add("0")
    for pos in positions:
        pos_dir = root / pos
        # base pyramid
        for lvl in _list_numeric_level_names(pos_dir):
            if lvl not in keep_levels and lvl != "0":
                try:
                    shutil.rmtree(pos_dir / lvl)
                    vprintf("Pruned level %s for %s", lvl, pos)
                except Exception as e:
                    vprintf("Failed to prune %s/%s: %s", pos, lvl, str(e))
        # grid_edges pyramid
        ge_dir = pos_dir / "grid_edges"
        if ge_dir.exists():
            for lvl in _list_numeric_level_names(ge_dir):
                if lvl not in keep_levels and lvl != "0":
                    try:
                        shutil.rmtree(ge_dir / lvl)
                        vprintf("Pruned grid_edges level %s for %s", lvl, pos)
                    except Exception as e:
                        vprintf(
                            "Failed to prune grid_edges %s/%s: %s", pos, lvl, str(e)
                        )


def _synthesize_edges_mask(
    y: int,
    x: int,
    tile_h: int,
    tile_w: int,
    origin_y: int,
    origin_x: int,
    line_width: int,
) -> np.ndarray:
    """Generate a binary YX mask (uint8) with grid lines at tile boundaries.

    Lines placed at rows y = origin_y + k*tile_h and cols x = origin_x + k*tile_w.
    """
    mask = np.zeros((y, x), dtype=np.uint8)
    if tile_h <= 0 or tile_w <= 0:
        return mask
    # Horizontal lines
    r = max(0, origin_y)
    while r < y:
        rr = slice(r, min(y, r + line_width))
        mask[rr, :] = 255
        r += tile_h
    # Vertical lines
    c = max(0, origin_x)
    while c < x:
        cc = slice(c, min(x, c + line_width))
        mask[:, cc] = 255
        c += tile_w
    return mask


from typing import Iterable, Tuple


def _synthesize_edges_from_rects(
    y: int, x: int, rects: Iterable[Tuple[int, int, int, int]], line_width: int
) -> np.ndarray:
    """Generate a binary YX mask (uint8) drawing rectangle edges.

    rects: iterable of (top, left, height, width) in pixels.
    """
    mask = np.zeros((y, x), dtype=np.uint8)
    lw = max(1, int(line_width))
    for top, left, height, width in rects:
        t = max(0, int(top))
        l = max(0, int(left))
        b = min(y, t + max(0, int(height)))
        r = min(x, l + max(0, int(width)))
        if t >= b or l >= r:
            continue
        # Top edge
        mask[t : min(b, t + lw), l:r] = 255
        # Bottom edge
        mask[max(t, b - lw) : b, l:r] = 255
        # Left edge
        mask[t:b, l : min(r, l + lw)] = 255
        # Right edge
        mask[t:b, max(l, r - lw) : r] = 255
    return mask


def _candidate_pos_prefixes(pos: str) -> list[str]:
    """Return possible YAML key prefixes for a position name.

    Handles positions like 'A/1/0' by stripping the trailing numeric index,
    and returns both 'A/1/' and 'A1/' variants.
    """
    s = str(pos)
    parts = [p for p in s.split("/") if p != ""]
    # Drop trailing numeric segment (e.g., '0' from 'A/1/0')
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    base = "/".join(parts)
    prefixes = []
    if base:
        prefixes.append(base + "/")
        try:
            import re as _re

            m = _re.match(r"^([A-Za-z]+)/?(\d+)$", base)
            if m:
                prefixes.append(f"{m.group(1)}{m.group(2)}/")
        except Exception:
            pass
    return list(dict.fromkeys(prefixes))


from typing import Optional
from cyclops_utils.data.experiment import OpsDataset


def _resolve_stitch_config_path(experiment: Optional[str], mode: str) -> Optional[Path]:
    """Resolve default stitch config YAML path from OpsDataset and mode."""
    if experiment is None:
        return None
    try:
        ds = OpsDataset(experiment)
        if mode == "pheno":
            key = "lc_20x_stitch"
        elif mode == "track":
            key = "lc_5x_stitch"
        elif mode == "iss":
            key = "iss_stitch"
        else:
            key = None
        if key is None:
            return None
        p = ds.config_paths.get(key)
        return Path(p) if p is not None else None
    except Exception:
        return None


def _apply_clims_to_layer(layer, per_level_clims, pos):
    lvl = getattr(layer, "data_level", None)
    lvl_idx = int(lvl) if lvl is not None else None
    if (
        isinstance(lvl_idx, int)
        and 0 <= lvl_idx < len(per_level_clims)
        and per_level_clims[lvl_idx] is not None
    ):
        cl = per_level_clims[lvl_idx]
        layer.contrast_limits = cl
        return True
    return False


# Enable continuous contrast updates (auto reset on zoom/level changes)
def _apply_clims_on_zoom(
    event=None,
    layers=None,
    pos=None,
    _last_level=None,
    per_layer_clims_map=None,
    per_level_clims=None,
):
    # Discrete per-level application
    for lyr in layers:
        lvl = getattr(lyr, "data_level", None)
        prev = _last_level.get(id(lyr))
        if lvl != prev:
            clims = None
            try:
                if per_layer_clims_map is not None:
                    clims = per_layer_clims_map.get(id(lyr))
            except Exception:
                clims = None
            if clims is None:
                clims = per_level_clims
            ok = _apply_clims_to_layer(lyr, clims, pos)
            if not ok:
                vprintf(
                    "[contrast] apply_clims returned False for %s",
                    str(getattr(lyr, "name", "")),
                )
            _last_level[id(lyr)] = lvl


# --------- Pyramid building utilities ---------


def determine_target_levels(
    source_store: Path, pos_path: str, levels: int, resume: bool,
    t: int | None = None, c: int | None = None,
) -> list[int]:
    """Determine which pyramid levels need to be built for a position.

    Args:
        source_store: Path to zarr store
        pos_path: Position path (e.g., 'A/1/0')
        levels: Total number of levels desired
        resume: If True, skip levels that already have data
        t: Time index — when provided, zero-check samples only this t slice
        c: Channel index — when provided, zero-check samples only this c slice

    Returns:
        Sorted list of level indices to build (e.g., [1, 2, 3, 4])
    """
    from cyclops_utils.io.zarr_utils import list_numeric_levels

    existing = {
        int(k) for k in list_numeric_levels(source_store, pos_path) if int(k) > 0
    }
    desired = set(range(1, int(levels)))

    if not resume:
        return sorted(desired if desired else existing)

    # resume mode: build if missing or zero-filled for the specific (t, c) slice
    from cyclops_process.processes.pyramids.build_dask import _is_zero_like_component

    targets = []
    for lvl in sorted(desired):
        if lvl not in existing or _is_zero_like_component(
            source_store, str(Path(pos_path) / str(lvl)), t=t, c=c
        ):
            targets.append(lvl)
    return targets


def downsample_array_sequential(
    data: np.ndarray, targets: list[int], factor: int, dtype: np.dtype
) -> dict[int, np.ndarray]:
    """Downsample data sequentially through multiple pyramid levels.

    Args:
        data: Input array to downsample
        targets: Level indices to generate (e.g., [1, 2, 3])
        factor: Downsampling factor (typically 2)
        dtype: Data type to preserve

    Returns:
        Dictionary mapping level index to downsampled array
    """
    from skimage.transform import downscale_local_mean as cpu_downscale

    results = {}
    curr = data
    for lvl in targets:
        curr = cpu_downscale(curr, (factor,) * curr.ndim)
        curr = (
            np.rint(curr).astype(dtype, copy=False)
            if np.issubdtype(dtype, np.integer)
            else curr.astype(dtype, copy=False)
        )
        results[lvl] = curr.copy()
    return results


def get_chunk_positions(
    chunks_y: Sequence[int], chunks_x: Sequence[int]
) -> list[Tuple[int, int, int, int]]:
    """Calculate chunk positions in the full array.

    Args:
        chunks_y: Chunk sizes along Y axis
        chunks_x: Chunk sizes along X axis

    Returns:
        List of (y_idx, x_idx, y_offset, x_offset) tuples
    """
    positions = []
    y_offset = 0
    for y_idx, cy in enumerate(chunks_y):
        x_offset = 0
        for x_idx, cx in enumerate(chunks_x):
            positions.append((y_idx, x_idx, y_offset, x_offset))
            x_offset += cx
        y_offset += cy
    return positions


def place_data_in_output(
    output_array: np.ndarray, data: np.ndarray, y_out: int, x_out: int
) -> None:
    """Place downsampled chunk data into output array, handling Z-dim and cropping.

    Args:
        output_array: Target array (2D or 3D with Z)
        data: Source data to place
        y_out: Y position in output array
        x_out: X position in output array
    """
    # Get available space in output array
    out_shape = output_array.shape
    y_available = out_shape[-2] - y_out
    x_available = out_shape[-1] - x_out

    # Crop data if needed to fit
    y_size = min(data.shape[-2], y_available)
    x_size = min(data.shape[-1], x_available)

    # Handle different array dimensions (with or without Z)
    if output_array.ndim == 3:
        # Has Z dimension: shape is (Z, Y, X)
        output_array[:, y_out : y_out + y_size, x_out : x_out + x_size] = data[
            ..., :y_size, :x_size
        ]
    else:
        # No Z dimension: shape is (Y, X)
        output_array[y_out : y_out + y_size, x_out : x_out + x_size] = data[
            ..., :y_size, :x_size
        ]
