"""Bounded-I/O normalization-statistics sampler for large assembled stores.

viscy's ``generate_normalization_metadata`` reads each full Y*X slice into RAM
(``mp_utils.sample_im_pixels``: ``image_zarr[t, c, z, :, :]``) before
grid-subsampling. For per-FOV tiles (~2k*2k, ~16 MB) that's fine, but the
assembled phenotyping canvas is ~100k*100k per well (tens of GB per channel),
so the full read makes the ``viscy_normalize`` step time out.

This samples a bounded set of chunk-aligned blocks instead — a few hundred MB
regardless of canvas size — and writes identical ``normalization`` metadata
(same mean/std/median/iqr stats and OME-NGFF layout as viscy) so downstream
readers (clims/portal/napari) are unaffected.
"""

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import iohub.ngff as ngff
# viscy (-> torch) is optional at IMPORT so this module loads in torch-less envs (CI):
# the torch-free sampling helpers are exercised by the unit tests. The viscy-backed
# names below are None only where viscy/torch isn't installed; the functions that use
# them (generate_normalization_metadata_fast, benchmark_sampling) are viscy paths.
try:
    from viscy.utils.meta_utils import write_meta_field
    from viscy.utils.mp_utils import get_val_stats
except ModuleNotFoundError:
    write_meta_field = get_val_stats = None


def _block_origins(extent: int, block: int, max_along_axis: int) -> list[int]:
    """Chunk-aligned block start coords along one axis, capped to max_along_axis."""
    origins = list(range(0, max(1, extent - 1), block))
    if len(origins) <= max_along_axis:
        return origins
    step = int(math.ceil(len(origins) / max_along_axis))
    return origins[::step]


def sample_position_channel(
    position,
    channel: int,
    *,
    block: int = 512,
    max_blocks: int = 400,
    grid_spacing: int = 4,
    num_workers: int = 8,
) -> np.ndarray:
    """Sample pixel values from a bounded grid of chunk-aligned blocks.

    Reads at most ~``max_blocks`` windows of ``block``x``block`` (one inner shard
    chunk), distributed across the full canvas, and sub-samples each window every
    ``grid_spacing`` px. Never materializes a full Y*X slice. Returns a flat
    float array of sampled values.
    """
    arr = position.data  # (T, C, Z, Y, X)
    T, _, Z, Y, X = arr.shape
    per_axis = max(1, int(math.sqrt(max_blocks)))
    ys = _block_origins(Y, block, per_axis)
    xs = _block_origins(X, block, per_axis)

    coords = [
        (t, z, y0, x0)
        for t in range(T)
        for z in range(Z)
        for y0 in ys
        for x0 in xs
    ]

    def _read(c):
        t, z, y0, x0 = c
        y1, x1 = min(y0 + block, Y), min(x0 + block, X)
        win = np.asarray(arr[t, channel, z, y0:y1, x0:x1])
        if grid_spacing > 1:
            win = win[::grid_spacing, ::grid_spacing]
        return win.ravel()

    with ThreadPoolExecutor(max(1, num_workers)) as ex:
        parts = list(ex.map(_read, coords))
    return np.concatenate(parts) if parts else np.zeros(0, np.float32)


def generate_normalization_metadata_fast(
    zarr_dir,
    channel_ids=-1,
    num_workers: int = 8,
    *,
    block: int = 512,
    max_blocks: int = 400,
    grid_spacing: int = 4,
) -> None:
    """Drop-in for viscy.generate_normalization_metadata with bounded I/O.

    Writes the same ``normalization`` field (``dataset_statistics`` at plate
    level + ``fov_statistics`` per FOV; keys mean/std/median/iqr) using viscy's
    own ``get_val_stats`` / ``write_meta_field`` so the format is byte-compatible.
    Only the sampling strategy differs.
    """
    plate = ngff.open_ome_zarr(zarr_dir, mode="r+")
    position_map = list(plate.positions())

    if channel_ids == -1:
        channel_ids = range(len(plate.channel_names))
    elif isinstance(channel_ids, int):
        channel_ids = [channel_ids]

    for channel in channel_ids:
        channel_name = plate.channel_names[channel]

        fov_sample_values = [
            sample_position_channel(
                pos, channel, block=block, max_blocks=max_blocks,
                grid_spacing=grid_spacing, num_workers=num_workers,
            )
            for _, pos in position_map
        ]

        nonempty = [v for v in fov_sample_values if v.size]
        dataset_sample_values = (
            np.concatenate(nonempty) if nonempty else np.zeros(0, np.float32)
        )
        dataset_statistics = {"dataset_statistics": get_val_stats(dataset_sample_values)}
        write_meta_field(plate, dataset_statistics, "normalization", channel_name)

        for j, (_, pos) in enumerate(position_map):
            position_statistics = dataset_statistics | {
                "fov_statistics": get_val_stats(fov_sample_values[j]),
            }
            write_meta_field(pos, position_statistics, "normalization", channel_name)

    plate.close()


def benchmark_sampling(zarr_dir, channel: int = 0, **kwargs) -> dict:
    """Benchmark the bounded sampler on one channel of a store.

    Returns timing + how many pixels were read vs the full-slice size, so the
    read-amplification savings vs viscy's full ``[:, :]`` read are explicit.
    """
    import time

    plate = ngff.open_ome_zarr(zarr_dir, mode="r")
    _, pos = list(plate.positions())[0]
    shape = pos.data.shape
    full_slice_px = int(shape[0] * shape[2] * shape[-2] * shape[-1])

    t0 = time.perf_counter()
    vals = sample_position_channel(pos, channel, **kwargs)
    elapsed = time.perf_counter() - t0
    plate.close()

    sampled_px = int(vals.size)
    return {
        "shape": tuple(shape),
        "full_slice_px": full_slice_px,
        "sampled_px": sampled_px,
        "fraction_read": sampled_px / full_slice_px if full_slice_px else 0.0,
        "seconds": elapsed,
        "px_per_sec": sampled_px / elapsed if elapsed else 0.0,
        "stats": get_val_stats(vals) if vals.size else None,
    }


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Benchmark bounded normalization sampling")
    ap.add_argument("zarr_dir", help="Path to an OME-Zarr store (e.g. phenotyping_v3.zarr)")
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--max-blocks", type=int, default=400)
    ap.add_argument("--grid-spacing", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    result = benchmark_sampling(
        args.zarr_dir, channel=args.channel, block=args.block,
        max_blocks=args.max_blocks, grid_spacing=args.grid_spacing,
        num_workers=args.num_workers,
    )
    print(json.dumps(result, indent=2))
