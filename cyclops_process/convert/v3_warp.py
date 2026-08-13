"""Shared affine-warp engine for fixed-cell v3 conversion (cell painting + 4i).

Both modalities warp fixed-cell channels/labels into the pheno 20x frame the same
way: for each output spatial chunk, map its corners back through the affine to find
the needed input region, build a chunk-local affine, run scipy.ndimage.affine_transform,
and write the result to the destination tensorstore. cp and 4i previously each carried
their own copy of this loop (images, order=1) plus a third copy for label warping
(order=0). This is the single shared implementation.

Only the loop is shared; the affine *composition* (cp compose-then-scale vs 4i
scale-then-compose) and store init stay modality-specific.
"""

from __future__ import annotations

import gc
import time

import yaml
import numpy as np
import tensorstore as ts
from scipy.ndimage import affine_transform as scipy_affine_transform


def affine_3x3_from_4x4(affine_4x4: np.ndarray) -> np.ndarray:
    """Extract the 2D YX affine (3x3) from a 4x4 ZYX affine (drop Z row/col)."""
    affine_4x4 = np.asarray(affine_4x4, dtype=float)
    a3 = np.identity(3)
    a3[0:2, 0:2] = affine_4x4[1:3, 1:3]   # YX scale/rotation
    a3[0:2, 2] = affine_4x4[1:3, 3]       # YX translation
    return a3


def _read_src_region(source_array, src_ch, y0, y1, x0, x1, *, source_is_2d):
    """Read a (y0:y1, x0:x1) region for one channel from a v2/stitched source array."""
    if source_is_2d:
        chunk = source_array[y0:y1, x0:x1]
    else:
        chunk = source_array[0, src_ch, 0, y0:y1, x0:x1]
    if hasattr(chunk, "compute"):
        chunk = chunk.compute()
    return np.asarray(chunk)


def warp_channels_into_v3(
    source_array,
    dest_ts,
    affine_3x3: np.ndarray,
    channel_pairs: list[tuple[int, int]],
    *,
    order: int,
    dtype,
    source_is_2d: bool = False,
    spatial_chunk_multiplier: int = 48,
    base_chunk_px: int = 512,
    chunk_size: int | None = None,
    interp_padding: int = 10,
    progress_every: int = 10,
    label: str = "",
):
    """Affine-warp source channels into an (already-created) destination tensorstore.

    Args:
        source_array: zarr/tensorstore-like source. Indexed [0, ch, 0, y, x] unless
            ``source_is_2d`` (then [y, x], and channel_pairs' src index is ignored).
        dest_ts: opened destination tensorstore, shape (T, C, Z, Y, X). Written at
            [0, dst_ch, 0, y0:y1, x0:x1].
        affine_3x3: 2D YX affine mapping OUTPUT coords -> INPUT coords (scipy convention).
        channel_pairs: list of (src_ch, dst_ch). For labels use [(0, 0)].
        order: interpolation order — 1 (bilinear) for images, 0 (nearest) for labels.
        dtype: output/write dtype (np.float32 for images, np.int32 for labels).
        chunk_size: output spatial chunk edge in px. Defaults to
            ``base_chunk_px * spatial_chunk_multiplier`` (512 * 48 = 24576) when None.
            Pass an explicit value to override (e.g. smaller on memory-tight nodes).

    Returns the elapsed seconds.
    """
    affine_3x3 = np.asarray(affine_3x3, dtype=float)
    scale_2x2 = affine_3x3[0:2, 0:2]

    if source_is_2d:
        source_h, source_w = source_array.shape[-2:]
    else:
        source_h, source_w = source_array.shape[-2:]
    target_h, target_w = dest_ts.shape[-2:]

    if chunk_size is None:
        chunk_size = base_chunk_px * spatial_chunk_multiplier
    y_chunks = (target_h + chunk_size - 1) // chunk_size
    x_chunks = (target_w + chunk_size - 1) // chunk_size
    total_chunks = y_chunks * x_chunks

    print(f"      [warp{(' ' + label) if label else ''}] src {source_h}x{source_w} -> "
          f"tgt {target_h}x{target_w}, {len(channel_pairs)} ch, order={order}, "
          f"{total_chunks} chunks")

    start_time = time.time()
    chunk_idx = 0

    for y_start in range(0, target_h, chunk_size):
        y_end = min(y_start + chunk_size, target_h)
        out_chunk_h = y_end - y_start

        for x_start in range(0, target_w, chunk_size):
            x_end = min(x_start + chunk_size, target_w)
            out_chunk_w = x_end - x_start
            chunk_idx += 1

            # Map output-chunk corners back to input space to find the region needed.
            corners_out = np.array([
                [y_start, x_start, 1],
                [y_start, x_end, 1],
                [y_end, x_start, 1],
                [y_end, x_end, 1],
            ], dtype=np.float64)
            corners_in = (affine_3x3 @ corners_out.T).T[:, :2]

            in_y_min = int(np.floor(corners_in[:, 0].min())) - interp_padding
            in_y_max = int(np.ceil(corners_in[:, 0].max())) + interp_padding
            in_x_min = int(np.floor(corners_in[:, 1].min())) - interp_padding
            in_x_max = int(np.ceil(corners_in[:, 1].max())) + interp_padding

            in_y_min_c = max(0, in_y_min)
            in_y_max_c = min(source_h, in_y_max)
            in_x_min_c = max(0, in_x_min)
            in_x_max_c = min(source_w, in_x_max)

            # Chunk-local affine: shift translation for the output offset and the
            # cropped input origin (same for every channel in this chunk).
            chunk_affine = affine_3x3.copy()
            output_offset = np.array([y_start, x_start], dtype=np.float64)
            input_crop_offset = np.array([in_y_min, in_x_min], dtype=np.float64)
            chunk_affine[0:2, 2] = scale_2x2 @ output_offset + affine_3x3[0:2, 2] - input_crop_offset

            for src_ch, dst_ch in channel_pairs:
                src_chunk = _read_src_region(source_array, src_ch,
                                             in_y_min_c, in_y_max_c, in_x_min_c, in_x_max_c,
                                             source_is_2d=source_is_2d)
                src_chunk = np.asarray(src_chunk, dtype=dtype)
                if src_chunk.size == 0:
                    continue

                # Pad back to the unclamped region so coordinates stay consistent.
                if in_y_min < 0 or in_x_min < 0 or in_y_max > source_h or in_x_max > source_w:
                    padded = np.zeros((in_y_max - in_y_min, in_x_max - in_x_min), dtype=dtype)
                    dst_y = in_y_min_c - in_y_min
                    dst_x = in_x_min_c - in_x_min
                    padded[dst_y:dst_y + src_chunk.shape[0],
                           dst_x:dst_x + src_chunk.shape[1]] = src_chunk
                    src_chunk = padded

                transformed = scipy_affine_transform(
                    src_chunk,
                    chunk_affine,
                    output_shape=(out_chunk_h, out_chunk_w),
                    order=order,
                    mode="constant",
                    cval=0,
                )

                dest_ts[0, dst_ch, 0, y_start:y_end, x_start:x_end].write(
                    transformed.astype(dtype)
                ).result()

                del src_chunk, transformed

            if progress_every and (chunk_idx % progress_every == 0 or chunk_idx == total_chunks):
                elapsed = time.time() - start_time
                cps = chunk_idx / elapsed if elapsed > 0 else 0
                print(f"        chunk {chunk_idx}/{total_chunks} "
                      f"({100 * chunk_idx / total_chunks:.1f}%) | {cps:.2f} chunks/s")

    gc.collect()
    total = time.time() - start_time
    print(f"      [warp{(' ' + label) if label else ''}] done in {total:.1f}s")
    return total


# --- generic affine helpers (shared by cp + 4i chain builders in v3_fixed) ---

def load_affine_from_yaml(yaml_path: Path) -> np.ndarray:
    """Load 4x4 affine matrix from registration YAML file."""
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return np.array(config["affine_transform_zyx"])

def compose_affines(affine_a: np.ndarray, affine_b: np.ndarray) -> np.ndarray:
    """Compose two 4x4 affine transforms: result maps through A then B.

    For round N -> round 1 -> pheno:
      composed = compose_affines(round_to_r1, r1_to_pheno)
    """
    return affine_b @ affine_a

def _scale_affine_translation(affine: np.ndarray, factor: float) -> np.ndarray:
    """Scale only the translation part of a 4x4 affine (rotation/scale unchanged).

    This is how to convert an affine computed at one resolution to apply at
    another resolution. The rotation and scale don't change — only translation.
    """
    out = affine.copy()
    out[1, 3] *= factor
    out[2, 3] *= factor
    return out
