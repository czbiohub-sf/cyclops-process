"""Tests + benchmark for the bounded-I/O normalization sampler.

Covers cyclops_process.utils.fast_normalize, which replaces viscy's full-slice
read (image_zarr[t, c, z, :, :]) — fine for small FOVs but tens of GB on the
assembled ~100k*100k canvas — with bounded chunk-aligned block sampling.

Validates: (1) it does NOT read the whole slice, (2) stats stay close to the
full-array stats, (3) the written metadata matches viscy's format, and prints a
sampling-rate benchmark.
"""
import os
import time
import tempfile
from pathlib import Path

import numpy as np
import pytest
from iohub import open_ome_zarr

os.environ["OPS_DISABLE_NOTIFICATIONS"] = "1"

from cyclops_process.utils.fast_normalize import (
    sample_position_channel,
    generate_normalization_metadata_fast,
    benchmark_sampling,
)

import importlib.util
# generate_normalization_metadata_fast / benchmark_sampling call viscy's
# get_val_stats / write_meta_field; skip those tests where viscy (-> torch) isn't
# installed (e.g. CI). The pure-sampling tests above run everywhere.
requires_viscy = pytest.mark.skipif(
    importlib.util.find_spec('viscy') is None,
    reason='needs viscy (write_meta_field/get_val_stats), not installed in this env',
)


def _make_store(path, *, shape=(1, 2, 1, 4096, 4096), chunk=512, seed=0):
    """HCS store with one large FOV and known per-channel distributions."""
    store = open_ome_zarr(path, layout="hcs", mode="w",
                          channel_names=[f"ch{c}" for c in range(shape[1])])
    rng = np.random.default_rng(seed)
    pos = store.create_position("A", "1", "000000")
    chunks = (1, 1, 1, chunk, chunk)
    pos.create_zeros("0", shape=shape, dtype=np.float32, chunks=chunks)
    # Channel c ~ Normal(mean=c+1, std=c+1) so each channel's stats are distinct.
    for c in range(shape[1]):
        data = rng.normal(loc=c + 1.0, scale=c + 1.0, size=shape[2:]).astype(np.float32)
        pos["0"][0, c, :, :, :] = data
    store.close()
    return path


def test_sampler_does_not_read_whole_slice():
    """With max_blocks capped, the sampler reads a small fraction of the slice."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_store(Path(tmp) / "s.zarr", shape=(1, 1, 1, 4096, 4096), chunk=512)
        with open_ome_zarr(path, layout="hcs", mode="r") as ds:
            _, pos = list(ds.positions())[0]
            full_px = pos.data.shape[-1] * pos.data.shape[-2]
            vals = sample_position_channel(
                pos, 0, block=512, max_blocks=16, grid_spacing=4, num_workers=4
            )
        # 16 blocks of 512px subsampled /4 -> 16 * 128 * 128 = 262144 px, far below 16.7M
        assert vals.size < full_px * 0.05, f"read too much: {vals.size}/{full_px}"
        assert vals.size > 0


def test_sampler_stats_close_to_full():
    """Bounded-sample stats track the full-array stats within tolerance."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_store(Path(tmp) / "s.zarr", shape=(1, 1, 1, 4096, 4096), chunk=512, seed=7)
        with open_ome_zarr(path, layout="hcs", mode="r") as ds:
            _, pos = list(ds.positions())[0]
            full = np.asarray(pos.data[0, 0, 0, :, :]).ravel()
            vals = sample_position_channel(
                pos, 0, block=512, max_blocks=200, grid_spacing=4, num_workers=4
            )
        # channel 0 ~ N(1, 1)
        assert abs(float(np.mean(vals)) - float(np.mean(full))) < 0.05
        assert abs(float(np.std(vals)) - float(np.std(full))) < 0.05
        assert abs(float(np.median(vals)) - float(np.median(full))) < 0.05


@requires_viscy
def test_metadata_format_matches_viscy_layout():
    """generate_normalization_metadata_fast writes plate + per-FOV stats with the 4 keys."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_store(Path(tmp) / "s.zarr", shape=(1, 2, 1, 2048, 2048), chunk=512)
        generate_normalization_metadata_fast(str(path), channel_ids=-1, num_workers=2,
                                              block=512, max_blocks=64, grid_spacing=4)
        with open_ome_zarr(path, layout="hcs", mode="r") as ds:
            ch_names = list(ds.channel_names)
            _, pos = list(ds.positions())[0]
            norm = dict(pos.zattrs)["normalization"]
        for ch in ch_names:
            assert "dataset_statistics" in norm[ch]
            assert "fov_statistics" in norm[ch]
            for key in ("mean", "std", "median", "iqr"):
                assert np.isfinite(norm[ch]["dataset_statistics"][key])
                assert np.isfinite(norm[ch]["fov_statistics"][key])
        # channel stats are distinct (ch1 ~ N(2,2) wider than ch0 ~ N(1,1))
        assert norm[ch_names[1]]["fov_statistics"]["std"] > norm[ch_names[0]]["fov_statistics"]["std"]


@requires_viscy
def test_benchmark_bounded_faster_than_full_read():
    """Bounded sampling beats a full-slice read on a large-ish synthetic canvas."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_store(Path(tmp) / "big.zarr", shape=(1, 1, 1, 8192, 8192), chunk=512)
        with open_ome_zarr(path, layout="hcs", mode="r") as ds:
            _, pos = list(ds.positions())[0]
            t0 = time.perf_counter()
            _ = np.asarray(pos.data[0, 0, 0, :, :])  # viscy's full read
            full_t = time.perf_counter() - t0

        res = benchmark_sampling(str(path), channel=0, block=512, max_blocks=200,
                                 grid_spacing=4, num_workers=8)
        print(f"\n[bench] full-read {full_t*1000:.0f} ms | bounded {res['seconds']*1000:.0f} ms "
              f"| fraction_read {res['fraction_read']*100:.3f}% "
              f"| {res['px_per_sec']/1e6:.1f} Mpx/s")
        assert res["fraction_read"] < 0.05
        assert res["seconds"] < full_t  # bounded must be faster than full read


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
