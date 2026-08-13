"""
Subtile-based autofocus reconstruction system.

This module implements an advanced autofocus system that divides each tile into N subtiles,
performs individual autofocus reconstruction on each subtile, then stitches them back together
with linear blending to create seamless final images.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Any
import tempfile
import uuid

import os
import shutil
import logging
from contextlib import redirect_stdout, redirect_stderr
from joblib import Parallel, delayed
from scipy.ndimage import laplace, distance_transform_edt
from iohub import open_ome_zarr
from iohub.ngff import TransformationMeta

# Suppress iohub warnings about channel mismatch
logging.getLogger("iohub.ngff.nodes").setLevel(logging.ERROR)

import sys

sys.path.insert(0, os.getcwd())


# waveorder (-> torch) is imported lazily at its use site (in _apply, below) so this
# module imports cleanly in torch-less envs (CI); other waveorder imports here are
# already function-local.
from cyclops_process.utils.waveorder_utils import model_to_yaml
from ops_utils.io.zarr_utils import (
    _get_or_create_reusable_input_position,
    _ensure_store_position,
    _write_plane_to_store,
    _create_overlapping_subtile_bounds,
    _validate_subtile_grid,
)
from cyclops_process.utils.recon_utils import _infer_focus_index, _get_or_compute_tf_cached


# Cache for EDT-based blending weight maps to avoid recomputation per subtile
_EDT_WEIGHT_CACHE = {}


def _extract_subtile_data(
    full_data: np.ndarray, bounds: Tuple[int, int, int, int]
) -> np.ndarray:
    """Extract subtile data from full image stack.

    Args:
        full_data: Full image stack (T, C, Z, Y, X)
        bounds: (y_start, y_end, x_start, x_end)

    Returns:
        Subtile data with same dimensionality
    """
    y_start, y_end, x_start, x_end = bounds
    return full_data[:, :, :, y_start:y_end, x_start:x_end]


def _compute_subtile_focus(
    subtile_data: np.ndarray,
    optical_params: dict,
    subtile_id: int,
    enable_subpixel_precision: bool = False,
    polynomial_fit_order: int | None = None,
    midband_fractions: Tuple[float, float] = (0.125, 0.25),
    device: str = "cpu",
) -> Tuple[float, int]:
    """Compute focus index for a subtile.

    Args:
        subtile_data: Subtile image stack (T, C, Z, Y, X)
        optical_params: Dict with NA_det, lambda_ill, pixel_size
        subtile_id: ID for logging/debugging
        enable_subpixel_precision: Enable subpixel precision for focus finding
        polynomial_fit_order: Order of polynomial for subpixel fitting
        midband_fractions: (low, high) fractions of max frequency for bandpass
        device: Device to use for computation ('cpu' or 'cuda')

    Returns:
        Tuple of (focus_idx, Z_size)
    """
    # Extract Z stack for focus computation (assume T=0, C=0)
    zyx_stack = subtile_data[0, 0]  # (Z, Y, X)

    Z = zyx_stack.shape[0]
    try:
        focus_idx = _infer_focus_index(
            zyx_stack,
            NA_det=optical_params["NA_det"],
            lambda_ill=optical_params["lambda_ill"],
            pixel_size=optical_params["pixel_size"],
            enable_subpixel_precision=enable_subpixel_precision,
            polynomial_fit_order=polynomial_fit_order,
            midband_fractions=midband_fractions,
            device=device,  # BUG FIX: Pass device for GPU acceleration
        )
        if focus_idx is None:
            focus_idx = float(Z // 2)  # Fallback to middle
    except Exception:
        focus_idx = float(Z // 2)  # Fallback to middle

    return focus_idx, Z


def _load_tf_singular_system(tf_zarr_path: Path, device: str = "cpu"):
    """Load transfer function singular system (U, S, Vh) from cached zarr store.

    Returns tensors on the specified device.
    """
    import torch
    import zarr as zarr_lib
    tf_store = zarr_lib.open(str(tf_zarr_path), mode="r")
    U = torch.from_numpy(np.array(tf_store["singular_system_U"][0])).to(device)
    S = torch.from_numpy(np.array(tf_store["singular_system_S"][0, 0])).to(device)
    Vh = torch.from_numpy(np.array(tf_store["singular_system_Vh"][0])).to(device)
    return (U, S, Vh)


def _reconstruct_subtile_2d_direct(
    subtile_data: np.ndarray,
    singular_system,
    regularization_strength: float = 1e-3,
    bg_filter: bool = False,
    device: str = "cpu",
) -> Tuple[np.ndarray, bool]:
    """Reconstruct a single subtile by calling the GPU model directly.

    Bypasses all file-based I/O (no temp zarr stores, no config files).
    The singular_system (U, S, Vh) should already be on the target device.

    Args:
        subtile_data: Subtile image stack (T, C, Z, Y, X) - only BF channel
        singular_system: Tuple of (U, S, Vh) tensors on device
        regularization_strength: Tikhonov regularization parameter
        bg_filter: Whether to apply background filtering
        device: Device for computation ('cpu' or 'cuda')

    Returns:
        Tuple of (reconstructed_2d_data, success_flag)
    """
    import torch
    from waveorder.models.isotropic_thin_3d import apply_inverse_transfer_function

    try:
        # Convert input to float32 tensor on device: shape (Z, Y, X)
        zyx_data = torch.tensor(
            np.int32(subtile_data[0, 0]),  # (T=0, C=0) -> (Z, Y, X)
            dtype=torch.float32,
            device=device,
        )

        # Diagnostic: log shapes and stats for first call only
        if not hasattr(_reconstruct_subtile_2d_direct, "_logged"):
            _reconstruct_subtile_2d_direct._logged = True
            U, S, Vh = singular_system
            print(f"[DirectRecon DIAG] input shape={zyx_data.shape}, dtype={zyx_data.dtype}, "
                  f"min={zyx_data.min().item():.1f}, max={zyx_data.max().item():.1f}, "
                  f"mean={zyx_data.mean().item():.1f}")
            print(f"[DirectRecon DIAG] U={U.shape} {U.dtype}, S={S.shape} {S.dtype}, Vh={Vh.shape} {Vh.dtype}")
            print(f"[DirectRecon DIAG] S range: [{S.min().item():.6f}, {S.max().item():.6f}]")
            print(f"[DirectRecon DIAG] reg_strength={regularization_strength}, bg_filter={bg_filter}")

        # Call the GPU model directly — no file I/O
        absorption_yx, phase_yx = apply_inverse_transfer_function(
            zyx_data,
            singular_system,
            reconstruction_algorithm="Tikhonov",
            regularization_strength=regularization_strength,
            bg_filter=bg_filter,
        )

        # Move result to CPU numpy
        reconstructed_2d = phase_yx.cpu().numpy().astype(np.float32)

        # Diagnostic: log output stats for first few calls
        if not hasattr(_reconstruct_subtile_2d_direct, "_out_logged"):
            _reconstruct_subtile_2d_direct._out_logged = True
            nan_count = np.isnan(reconstructed_2d).sum()
            inf_count = (~np.isfinite(reconstructed_2d)).sum()
            print(f"[DirectRecon DIAG] output shape={reconstructed_2d.shape}, "
                  f"min={np.nanmin(reconstructed_2d):.6f}, max={np.nanmax(reconstructed_2d):.6f}, "
                  f"nan_count={nan_count}, inf_count={inf_count}")

        # Validate
        if np.isnan(reconstructed_2d).any() or not np.isfinite(reconstructed_2d).all():
            return np.full_like(reconstructed_2d, np.nan), False

        return reconstructed_2d, True

    except Exception as e:
        print(f"[SubtileRecon][ERROR] Direct 2D reconstruction failed: {e}")
        return np.full(
            (subtile_data.shape[3], subtile_data.shape[4]), np.nan, dtype=np.float32
        ), False


def _reconstruct_subtile_2d(
    subtile_data: np.ndarray,
    cfg2d_model,
    z_focus_offset: float,
    temp_dir: Path,
    subtile_id: int,
    position_scale,
    input_channel_names,
    original_metadata: dict | None = None,
    tf_cache_dir: Path | None = None,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Reconstruct a single subtile with 2D phase reconstruction.

    Legacy file-based path -- kept as fallback. Prefer _reconstruct_subtile_2d_direct.
    """
    import time

    unique_token = uuid.uuid4().hex[:8]
    temp_config_path = temp_dir / f"config_{subtile_id:02d}_{unique_token}.yaml"
    temp_output_store = temp_dir / f"output_{subtile_id:02d}_{unique_token}.zarr"

    reconstructed_2d = np.full(
        (subtile_data.shape[3], subtile_data.shape[4]), np.nan, dtype=np.float32
    )

    try:
        cfg2d_model.input_channel_names = input_channel_names
        model_to_yaml(cfg2d_model, temp_config_path)
        subtile_data_f32 = subtile_data.astype(np.float32, copy=False)

        filtered_metadata = None
        if original_metadata:
            filtered_metadata = dict(original_metadata)
            if "plate_zattrs" in filtered_metadata:
                filtered_metadata["plate_zattrs"] = {k: v for k, v in filtered_metadata["plate_zattrs"].items() if k != "omero"}
            if "position_zattrs" in filtered_metadata:
                filtered_metadata["position_zattrs"] = {k: v for k, v in filtered_metadata["position_zattrs"].items() if k != "omero"}
        input_position_path = _get_or_create_reusable_input_position(
            base_temp_dir=temp_dir,
            shape=subtile_data_f32.shape,
            position_scale=position_scale,
            input_channel_names=input_channel_names,
            original_metadata=filtered_metadata,
        )
        with open_ome_zarr(input_position_path, mode="r+") as input_pos_store:
            input_pos_store["0"][:] = subtile_data_f32

        if tf_cache_dir is None:
            tf_cache_dir = temp_dir / "tf_cache"
        tf_cache_dir.mkdir(parents=True, exist_ok=True)
        temp_tf_path = _get_or_compute_tf_cached(
            dataset=None,
            cfg2d_base_model=cfg2d_model,
            abs_offset=float(z_focus_offset),
            example_input_position_dirpath=input_position_path,
            cache_dir=tf_cache_dir,
            height=subtile_data_f32.shape[-2],
            width=subtile_data_f32.shape[-1],
            verbose=False,
        )

        with open_ome_zarr(
            temp_output_store, layout="hcs", mode="w-", channel_names=["Phase3D"]
        ) as output_store:
            output_pos = output_store.create_position("0", "0", "0")
            output_pos.create_zeros(
                name="0",
                shape=(1, 1, 1, subtile_data.shape[3], subtile_data.shape[4]),
                dtype=np.float32,
                transform=[TransformationMeta(type="scale", scale=position_scale)],
            )

        output_position_path = temp_output_store / "0" / "0" / "0"

        def _apply(tf_path: Path):
            from waveorder.cli.apply_inverse_transfer_function import (
                apply_inverse_transfer_function_single_position,
            )
            with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
                apply_inverse_transfer_function_single_position(
                    input_position_dirpath=input_position_path,
                    transfer_function_dirpath=tf_path,
                    config_filepath=temp_config_path,
                    output_position_dirpath=output_position_path,
                    num_processes=1,
                    output_channel_names=["Phase3D"],
                )

        try:
            _apply(temp_tf_path)
        except Exception as e_apply:
            msg = str(e_apply)
            if "Cannot open Zarr root group" in msg or "singular_system" in msg:
                try:
                    if os.path.isdir(temp_tf_path):
                        shutil.rmtree(temp_tf_path)
                except Exception:
                    pass
                temp_tf_path = _get_or_compute_tf_cached(
                    dataset=None,
                    cfg2d_base_model=cfg2d_model,
                    abs_offset=float(z_focus_offset),
                    example_input_position_dirpath=input_position_path,
                    cache_dir=tf_cache_dir,
                    height=subtile_data_f32.shape[-2],
                    width=subtile_data_f32.shape[-1],
                    verbose=False,
                )
                _apply(temp_tf_path)
            else:
                raise

        with open_ome_zarr(output_position_path, mode="r") as output_pos_store:
            reconstructed_2d = np.asarray(output_pos_store["0"][0, 0, 0, :, :])

        if np.isnan(reconstructed_2d).any():
            raise ValueError(f"Subtile {subtile_id}: NaN values in reconstruction")
        if not np.isfinite(reconstructed_2d).all():
            raise ValueError(f"Subtile {subtile_id}: non-finite values in reconstruction")

        Z = subtile_data.shape[2]
        abs_focus_idx = int(round(z_focus_offset + (Z // 2)))
        abs_focus_idx = max(0, min(abs_focus_idx, Z - 1))
        focus_slice = subtile_data[0, 0, abs_focus_idx, :, :]
        return reconstructed_2d, focus_slice, True

    except Exception as e:
        print(f"[SubtileRecon][ERROR] Subtile {subtile_id}: 2D reconstruction failed: {e}")
        raise e

    finally:
        if not np.isnan(reconstructed_2d).any():
            try:
                shutil.rmtree(temp_output_store, ignore_errors=True)
                temp_config_path.unlink(missing_ok=True)
            except Exception:
                pass


def _stitch_subtiles_overlapping(
    subtile_results: List[
        Tuple[np.ndarray, Tuple[int, int, int, int], Tuple[int, int]]
    ],
    full_shape: Tuple[int, int],
    blend_pixels: int,
    grid_size: int,
    blending_exponent: float = 1.0,
) -> np.ndarray:
    """Stitch overlapping subtiles with cross-boundary blending.

    Args:
        subtile_results: List of (subtile_data, overlap_bounds, core_position) tuples
        full_shape: (Y, X) dimensions of full output image
        blend_pixels: Overlap width for blending
        grid_size: Grid dimension
        blend_function: Blending function euclidean distance transform (edt)
        blending_exponent: Exponent for EDT-based blending

    Returns:
        Stitched image with smooth blending
    """
    Y, X = full_shape
    stitched = np.zeros((Y, X), dtype=np.float32)
    weight_sum = np.zeros((Y, X), dtype=np.float32)

    base_height = Y // grid_size
    base_width = X // grid_size

    # Create a copy of the stitched array to avoid modifying the input data in-place
    output_image = stitched.copy()

    if blend_pixels <= 0:
        # No blending: simple placement of non-overlapping subtiles
        for subtile_data, _, (core_row, core_col) in subtile_results:
            core_y_start = core_row * base_height
            core_y_end = (core_row + 1) * base_height if core_row < grid_size - 1 else Y
            core_x_start = core_col * base_width
            core_x_end = (core_col + 1) * base_width if core_col < grid_size - 1 else X

            # If blend_pixels is 0, subtile_data is the core region.
            output_image[core_y_start:core_y_end, core_x_start:core_x_end] = (
                subtile_data
            )

    else:
        # Vectorized blending approach
        for subtile_data, overlap_bounds, (core_row, core_col) in subtile_results:
            y_start, y_end, x_start, x_end = overlap_bounds

            # Calculate core region boundaries
            core_y_start = core_row * base_height
            core_y_end = (core_row + 1) * base_height if core_row < grid_size - 1 else Y
            core_x_start = core_col * base_width
            core_x_end = (core_col + 1) * base_width if core_col < grid_size - 1 else X

            # Create coordinate grids LOCAL to the subtile
            subtile_height = y_end - y_start
            subtile_width = x_end - x_start

            # y_coords_local = np.arange(subtile_height)[:, None]
            # x_coords_local = np.arange(subtile_width)[None, :]

            # # Calculate distances from EDGES of the subtile
            # dist_from_top = y_coords_local
            # dist_from_bottom = subtile_height - 1 - y_coords_local
            # dist_from_left = x_coords_local
            # dist_from_right = subtile_width - 1 - x_coords_local

            # Create weight matrix
            # Cache key includes size, exponent, and which edges require blending
            cache_key = (
                subtile_height,
                subtile_width,
                (
                    int(blending_exponent * 1e6)
                    if isinstance(blending_exponent, float)
                    else blending_exponent
                ),
                True,  # marker for EDT
            )
            weights = _EDT_WEIGHT_CACHE.get(cache_key)
            if weights is None:
                mask_dt = np.zeros((subtile_height, subtile_width), dtype=bool)
                if subtile_height > 2 and subtile_width > 2:
                    mask_dt[1:-1, 1:-1] = True
                distances = distance_transform_edt(mask_dt).astype(np.float32)
                distances += 1e-6
                weights = np.power(distances, blending_exponent, where=(distances > 0))
                _EDT_WEIGHT_CACHE[cache_key] = weights

            # Add weighted contribution to the output image
            output_image[y_start:y_end, x_start:x_end] += subtile_data * weights
            weight_sum[y_start:y_end, x_start:x_end] += weights

        # Normalize by weights
        mask = weight_sum > 0
        output_image[mask] /= weight_sum[mask]

    return output_image


def _process_single_subtile(
    subtile_id: int,
    bounds: tuple,
    t: int,
    grid_size: int,
    focus_full_data: np.ndarray,
    bf_recon_data: np.ndarray,
    optical_params: dict,
    enable_subpixel_precision: bool,
    polynomial_fit_order: int,
    midband_fractions: Tuple[float, float],
    cfg2d_base_model,
    temp_dir: Path,
    position_scale,
    raw_store_path: Path,
    pos: str,
    device: str = "cpu",  # BUG FIX: Add device parameter for GPU acceleration
    tf_cache_dir: Path | None = None,
) -> Dict[str, Any]:
    """Process a single subtile - thread-safe function for parallel execution.

    Returns:
        Dictionary containing:
        - focus_slice: Phase3D focus slice
        - reconstructed_2d: 2D reconstruction
        - bounds: Subtile bounds
        - grid_coords: (r, c) grid coordinates
        - metadata: Subtile metadata dict
        - success: Reconstruction success flag
    """
    try:
        r, c = subtile_id // grid_size, subtile_id % grid_size

        # Extract stacks for this subtile
        subtile_phase_stack = _extract_subtile_data(
            focus_full_data[t : t + 1, :, :, :, :], bounds
        )
        subtile_bf_stack = _extract_subtile_data(
            bf_recon_data[t : t + 1, :, :, :, :], bounds
        )

        # Compute focus and offsets directly
        focus_idx, Z = _compute_subtile_focus(
            subtile_phase_stack,
            optical_params,
            subtile_id,
            enable_subpixel_precision=enable_subpixel_precision,
            polynomial_fit_order=(
                polynomial_fit_order if enable_subpixel_precision else None
            ),
            midband_fractions=midband_fractions,
            device=device,  # BUG FIX: Pass device for GPU acceleration
        )
        Z = int(Z)
        focus_idx_bounded = max(0, min(int(round(focus_idx)), Z - 1))
        z_focus_offset = float(focus_idx - (Z // 2))
        # Quantize z offset to one decimal place to improve TF cache reuse
        z_focus_offset_q = float(round(z_focus_offset, 1))

        # Focus slice for stitched focus image (use Phase3D recon, not BF)
        focus_slice = subtile_phase_stack[0, 0, focus_idx_bounded, :, :].copy()

        # Prepare metadata and configuration
        original_metadata = {
            "position_scale": position_scale,
            "channel_names": cfg2d_base_model.input_channel_names,
        }
        try:
            with open_ome_zarr(raw_store_path, mode="r") as raw_store:
                pos_node = raw_store[pos]
                try:
                    plate = open_ome_zarr(raw_store_path.parent, mode="r")
                    original_metadata["plate_zattrs"] = dict(plate.zattrs)
                except Exception:
                    pass
                original_metadata["position_zattrs"] = dict(pos_node.zattrs)
                original_metadata["position_scale"] = pos_node.scale
        except Exception:
            pass

        cfg2d_subtile = cfg2d_base_model.model_copy(deep=True)
        cfg2d_subtile.phase.transfer_function.z_focus_offset = z_focus_offset_q

        reconstructed_2d, _, success = _reconstruct_subtile_2d(
            subtile_bf_stack,
            cfg2d_subtile,
            z_focus_offset_q,
            temp_dir,
            subtile_id,
            position_scale,
            cfg2d_base_model.input_channel_names,
            original_metadata=original_metadata,
            tf_cache_dir=tf_cache_dir,
        )

        meta = {
            "subtile_id": subtile_id,
            "bounds": bounds,
            "focus_index": focus_idx_bounded,
            "focus_index_float": float(focus_idx),
            "z_stack_size": Z,
            "z_focus_offset": z_focus_offset_q,
            "reconstruction_success": success,
            "time_index": int(t),
        }

        return {
            "focus_slice": focus_slice,
            "reconstructed_2d": reconstructed_2d,
            "bounds": bounds,
            "grid_coords": (r, c),
            "metadata": meta,
            "success": success,
        }

    except Exception as e:
        import traceback
        print(f"[SubtileRecon][ERROR] Subtile {subtile_id} failed: {e}")
        traceback.print_exc()
        # Return failure result
        return {
            "focus_slice": None,
            "reconstructed_2d": None,
            "bounds": bounds,
            "grid_coords": (subtile_id // grid_size, subtile_id % grid_size),
            "metadata": {
                "subtile_id": subtile_id,
                "bounds": bounds,
                "reconstruction_success": False,
                "time_index": int(t),
            },
            "success": False,
        }


def _process_subtile_wrapper_mp(args):
    """Wrapper for multiprocessing - must be at module level to be picklable."""
    return _process_single_subtile(**args)


def _process_position_subtile_flat_wrapper(args):
    """Wrapper for Dask - unpacks tuple of arguments for _process_position_subtile_flat.

    Must be at module level to be picklable for Dask serialization.
    """
    return _process_position_subtile_flat(*args)


def _process_position_subtile_flat(
    pos: str,
    subtile_id: int,
    bounds: tuple,
    t: int,
    grid_size: int,
    phase3d_store_path: Path,
    raw_store_path: Path,
    config_path: Path,
    optical_params: dict,
    enable_subpixel_precision: bool,
    polynomial_fit_order: int,
    midband_fractions: Tuple[float, float],
    temp_base_dir: Path,
    position_scale,
    device: str = "cpu",  # BUG FIX: Add device parameter for GPU acceleration
    tf_cache_dir: Path | None = None,
) -> Dict[str, Any]:
    """Process a single subtile from a single position (for flattened parallelization).

    This function is designed to be called by joblib Parallel to process subtiles
    from any position independently. Each worker opens zarr stores as needed.

    Returns:
        Dict with position, subtile_id, focus_slice, reconstructed_2d, metadata
    """
    try:
        # Open zarr stores (read-only, safe for concurrent access)
        with open_ome_zarr(phase3d_store_path, mode="r") as phase3d_store:
            focus_full_data = np.asarray(phase3d_store[pos]["0"])  # (T, C, Z, Y, X)

        with open_ome_zarr(raw_store_path, mode="r") as raw_store:
            recon_full_data = np.asarray(raw_store[pos]["0"])
            raw_channel_names = raw_store.channel_names

            # Find BF channel
            try:
                bf_channel_index = raw_channel_names.index("BF")
            except ValueError:
                bf_channel_index = 0

            # Validate selected channel has data; fall back to first non-zero channel
            if not np.any(recon_full_data[:, bf_channel_index]):
                for c_idx in range(recon_full_data.shape[1]):
                    if np.any(recon_full_data[:, c_idx]):
                        bf_channel_index = c_idx
                        break

            bf_recon_data = recon_full_data[
                :, bf_channel_index : bf_channel_index + 1, :, :, :
            ].copy()

            # Get metadata
            original_metadata = {
                "position_scale": position_scale,
                "channel_names": [raw_channel_names[bf_channel_index]],
            }
            try:
                pos_node = raw_store[pos]
                original_metadata["position_zattrs"] = dict(pos_node.zattrs)
                original_metadata["position_scale"] = pos_node.scale
            except Exception:
                pass

        # Extract subtile data
        subtile_phase_stack = _extract_subtile_data(
            focus_full_data[t : t + 1, :, :, :, :], bounds
        )
        subtile_bf_stack = _extract_subtile_data(
            bf_recon_data[t : t + 1, :, :, :, :], bounds
        )

        # Compute focus
        focus_idx, Z = _compute_subtile_focus(
            subtile_phase_stack,
            optical_params,
            subtile_id,
            enable_subpixel_precision=enable_subpixel_precision,
            polynomial_fit_order=(
                polynomial_fit_order if enable_subpixel_precision else None
            ),
            midband_fractions=midband_fractions,
            device=device,  # BUG FIX: Pass device for GPU acceleration
        )
        Z = int(Z)
        focus_idx_bounded = max(0, min(int(round(focus_idx)), Z - 1))
        z_focus_offset = float(focus_idx - (Z // 2))
        z_focus_offset_q = float(round(z_focus_offset, 1))

        # Focus slice
        focus_slice = subtile_phase_stack[0, 0, focus_idx_bounded, :, :].copy()

        # Load config and prepare for reconstruction
        from waveorder.io import utils
        from waveorder.cli.settings import ReconstructionSettings
        import copy
        cfg2d_model = utils.yaml_to_model(config_path, ReconstructionSettings)
        cfg2d_model.input_channel_names = [raw_channel_names[bf_channel_index]]
        cfg2d_subtile = copy.deepcopy(cfg2d_model)
        cfg2d_subtile.phase.transfer_function.z_focus_offset = z_focus_offset_q

        # Use scratchpad for temporary files
        import tempfile
        temp_dir = Path(tempfile.mkdtemp(prefix=f"subtile_{subtile_id}_"))

        # Reconstruct subtile
        reconstructed_2d, _, success = _reconstruct_subtile_2d(
            subtile_bf_stack,
            cfg2d_subtile,
            z_focus_offset_q,
            temp_dir,
            subtile_id,
            position_scale,
            cfg2d_model.input_channel_names,
            original_metadata=original_metadata,
            tf_cache_dir=tf_cache_dir,
        )

        # Prepare metadata
        r, c = subtile_id // grid_size, subtile_id % grid_size
        meta = {
            "subtile_id": subtile_id,
            "bounds": bounds,
            "focus_index": focus_idx_bounded,
            "focus_index_float": float(focus_idx),
            "z_stack_size": Z,
            "z_focus_offset": z_focus_offset_q,
            "reconstruction_success": success,
            "time_index": int(t),
        }

        return {
            "position": pos,
            "subtile_id": subtile_id,
            "t": t,
            "focus_slice": focus_slice if success else None,
            "reconstructed_2d": reconstructed_2d if success else None,
            "bounds": bounds,
            "grid_coords": (r, c),
            "metadata": meta,
            "success": success,
        }

    except Exception as e:
        import traceback
        print(f"[SubtileRecon][ERROR] Failed processing {pos} subtile {subtile_id}: {e}")
        traceback.print_exc()
        r, c = subtile_id // grid_size, subtile_id % grid_size
        return {
            "position": pos,
            "subtile_id": subtile_id,
            "t": t,
            "focus_slice": None,
            "reconstructed_2d": None,
            "bounds": bounds,
            "grid_coords": (r, c),
            "metadata": {
                "subtile_id": subtile_id,
                "bounds": bounds,
                "reconstruction_success": False,
                "time_index": int(t),
            },
            "success": False,
        }


def reconstruct_subtile_autofocus(
    pos: str,
    phase3d_store_path: Path,
    raw_store_path: Path,
    phase2d_store_path: Path,
    dataset,
    cfg2d_base_model,
    optical_params: dict,
    T: int,
    Y: int,
    X: int,
    position_scale,
    n_subtiles: int = 256,
    blend_pixels: int = 25,
    verbose: bool = False,
    enable_subpixel_precision: bool = True,
    polynomial_fit_order: int = 3,
    midband_fractions: Tuple[float, float] = (0.125, 0.25),
    device: str = "cpu",  # BUG FIX: Add device parameter for GPU acceleration
) -> Dict[str, Any]:
    """Reconstruct a single position using subtile-based autofocus."""
    import time
    import sys
    pos_total_start = time.time()
    timings = {}

    # Write timing to a separate file for real-time monitoring (unbuffered)
    timing_log = phase2d_store_path.parent / "reconstruction_timing.log"

    def log_timing(msg):
        """Write timing message to log file (unbuffered) and flush immediately"""
        with open(timing_log, "a") as f:
            f.write(f"{msg}\n")
            f.flush()

    grid_size = _validate_subtile_grid(n_subtiles, Y, X)

    # Load 3D phase data for this position (for focus finding only)
    # Use direct zarr.open instead of open_ome_zarr to skip plate metadata walk
    # (open_ome_zarr on a 7035-position plate takes ~8s per call)
    import zarr as zarr_lib
    import json as json_lib
    load_start = time.time()
    try:
        phase3d_arr = zarr_lib.open(str(phase3d_store_path / pos / "0"), mode="r")
        focus_full_data = np.asarray(phase3d_arr)  # (T, C, Z, Y, X)
    except Exception:
        print(
            f"[SubtileRecon][ERROR] {pos}: Position not found in phase store {phase3d_store_path}"
        )
        return {
            "reconstruction_success": False,
            "error": f"Position {pos} not found in phase store",
        }

    # Load raw data for this position (for reconstruction input)
    try:
        raw_arr = zarr_lib.open(str(raw_store_path / pos / "0"), mode="r")
        recon_full_data = np.asarray(raw_arr)
        # Read channel names — v3 stores metadata in zarr.json, v2 in .zattrs
        zarr_json_path = raw_store_path / pos / "zarr.json"
        zattrs_path = raw_store_path / pos / ".zattrs"
        if zarr_json_path.exists():
            with open(zarr_json_path) as f:
                raw = json_lib.load(f)
            raw_pos_attrs = raw.get("attributes", raw)
            # NGFF 0.5 nests omero/multiscales under "ome"; 0.4 has them flat
            if "ome" in raw_pos_attrs:
                raw_pos_attrs = raw_pos_attrs["ome"]
        else:
            with open(zattrs_path) as f:
                raw_pos_attrs = json_lib.load(f)
        raw_channel_names = [c["label"] for c in raw_pos_attrs.get("omero", {}).get("channels", [])]
    except Exception:
        print(
            f"[SubtileRecon][ERROR] {pos}: Position not found in raw store {raw_store_path}"
        )
        return {
            "reconstruction_success": False,
            "error": f"Position {pos} not found in raw store",
        }

    timings["data_loading"] = time.time() - load_start
    log_timing(f"[TIMING] {pos} - Data loading: {timings['data_loading']:.2f}s")

    # Find the index of the channel for reconstruction
    # If config has a valid channel name that exists in raw data, use it
    # Otherwise, look for 'BF' as the standard brightfield channel
    try:
        if cfg2d_base_model.input_channel_names:
            target_channel = cfg2d_base_model.input_channel_names[0]
            try:
                bf_channel_index = raw_channel_names.index(target_channel)
                print(
                    f"[SubtileRecon][DEBUG] Using channel '{target_channel}' at index {bf_channel_index} from raw_channel_names={raw_channel_names}"
                )
            except ValueError:
                # Config channel not found in raw data, try to find 'BF' instead
                print(
                    f"[SubtileRecon][WARN] Config channel '{target_channel}' not in raw data {raw_channel_names}, looking for 'BF'"
                )
                bf_channel_index = raw_channel_names.index("BF")
                print(f"[SubtileRecon][DEBUG] Using 'BF' at index {bf_channel_index}")
        else:
            # No config channel, try 'BF'
            bf_channel_index = raw_channel_names.index("BF")
            print(
                f"[SubtileRecon][DEBUG] No config channel, using 'BF' at index {bf_channel_index}"
            )
    except (ValueError, AttributeError, IndexError) as e:
        # Fallback to the first channel if nothing else works
        bf_channel_index = 0
        print(
            f"[SubtileRecon][WARN] No valid channel found ({e}), falling back to index 0 ('{raw_channel_names[0] if raw_channel_names else 'unknown'}'). raw_channel_names={raw_channel_names}"
        )

    # Validate the selected channel has actual data; if all zeros, find a non-zero channel.
    # This handles cases where link_phenotyping mislabeled channels (e.g., BF data at C=0
    # labeled "GFP" because the source had only 1 channel and the fallback channel list
    # put GFP first). The 3D reconstruction sidesteps this by always using channel [0].
    selected_data = recon_full_data[:, bf_channel_index : bf_channel_index + 1, :, :, :]
    if not np.any(selected_data):
        log_timing(
            f"[WARN] {pos}: Channel '{raw_channel_names[bf_channel_index]}' (C={bf_channel_index}) is all zeros, scanning for non-zero channel..."
        )
        for c_idx in range(recon_full_data.shape[1]):
            if np.any(recon_full_data[:, c_idx]):
                log_timing(
                    f"[WARN] {pos}: Using channel '{raw_channel_names[c_idx]}' (C={c_idx}) instead (has data)"
                )
                bf_channel_index = c_idx
                break
        else:
            log_timing(f"[ERROR] {pos}: ALL channels are zeros — skipping position")
            return {
                "reconstruction_success": False,
                "error": f"All channels are zeros for position {pos}",
            }

    # Isolate the brightfield data for all downstream tasks
    # Shape of recon_full_data is (T, C, Z, Y, X), so we slice on the C dimension
    bf_recon_data = recon_full_data[
        :, bf_channel_index : bf_channel_index + 1, :, :, :
    ].copy()
    print(
        f"[SubtileRecon][DEBUG] bf_recon_data shape: {bf_recon_data.shape}, dtype: {bf_recon_data.dtype}, min: {bf_recon_data.min()}, max: {bf_recon_data.max()}"
    )

    # Update the config to use the actual channel name we found
    actual_input_channel = raw_channel_names[bf_channel_index]
    cfg2d_base_model.input_channel_names = [actual_input_channel]
    print(
        f"[SubtileRecon][DEBUG] Updated config input_channel_names to ['{actual_input_channel}']"
    )

    # Calculate overlapping subtile bounds
    subtile_bounds = _create_overlapping_subtile_bounds(Y, X, grid_size, blend_pixels)

    # Use /dev/shm (RAM-backed tmpfs) for temp zarr stores — eliminates all disk I/O
    # for intermediate files. Falls back to /tmp if /dev/shm is unavailable.
    import tempfile
    shm_path = Path("/dev/shm")
    local_scratch = shm_path if shm_path.exists() else Path(os.environ.get("TMPDIR", "/tmp"))
    temp_dir = Path(tempfile.mkdtemp(dir=local_scratch, prefix=f"subtile_{pos.replace('/', '_')}_"))

    subtile_id_with_nan = None
    reconstruction_failed = False

    try:
        # Accumulate per-timepoint results to stitch after the loop
        t_to_focus_results: Dict[int, list] = {}
        t_to_2d_results: Dict[int, list] = {}
        subtile_metadata = []

        # Timing counters for subtile operations
        subtile_start = time.time()
        total_autofocus_time = 0.0
        total_recon_time = 0.0
        total_zarr_open_time = 0.0

        # Pre-fetch position metadata ONCE (constant for all 256 subtiles)
        # Uses direct file reads instead of open_ome_zarr to avoid plate metadata walk
        original_metadata = {
            "position_scale": position_scale,
            "channel_names": cfg2d_base_model.input_channel_names,
        }
        try:
            pos_zattrs_path = raw_store_path / pos / ".zattrs"
            if pos_zattrs_path.exists():
                with open(pos_zattrs_path) as f:
                    original_metadata["position_zattrs"] = json_lib.load(f)
                # Extract scale from multiscales metadata
                pz = original_metadata["position_zattrs"]
                if "multiscales" in pz:
                    for ms in pz["multiscales"]:
                        for ds in ms.get("datasets", []):
                            if "coordinateTransformations" in ds:
                                for ct in ds["coordinateTransformations"]:
                                    if ct.get("type") == "scale":
                                        original_metadata["position_scale"] = ct["scale"]
                                        break
                                break
                        break
            plate_zattrs_path = raw_store_path / ".zattrs"
            if plate_zattrs_path.exists():
                with open(plate_zattrs_path) as f:
                    original_metadata["plate_zattrs"] = json_lib.load(f)
        except Exception:
            pass

        # Direct GPU reconstruction: load TF singular systems on GPU and call
        # the model directly, bypassing all file-based zarr I/O per subtile.
        # Cache TFs per (z_offset, height, width) since edge subtiles differ in size.
        _tf_cache_on_device: Dict[tuple, tuple] = {}
        regularization_strength = cfg2d_base_model.phase.apply_inverse.regularization_strength
        Z_dim = bf_recon_data.shape[2]  # Z is the same for all subtiles

        def _get_tf_on_device(z_offset_q: float, subtile_h: int, subtile_w: int) -> tuple:
            """Get or load transfer function singular system for this offset and subtile size."""
            cache_key = (z_offset_q, subtile_h, subtile_w)
            if cache_key in _tf_cache_on_device:
                return _tf_cache_on_device[cache_key]
            cfg2d_tmp = cfg2d_base_model.model_copy(deep=True)
            cfg2d_tmp.phase.transfer_function.z_focus_offset = z_offset_q
            # Create reusable input position with SUBTILE dimensions (not full image)
            # so compute_transfer_function_cli generates TF at the correct spatial size.
            subtile_shape = (1, 1, Z_dim, subtile_h, subtile_w)
            input_pos_path = _get_or_create_reusable_input_position(
                base_temp_dir=temp_dir,
                shape=subtile_shape,
                position_scale=position_scale,
                input_channel_names=cfg2d_base_model.input_channel_names,
                original_metadata=original_metadata,
            )
            tf_cache_dir = dataset.config_paths["tf_cache_shared"]
            tf_cache_dir.mkdir(parents=True, exist_ok=True)
            tf_zarr_path = _get_or_compute_tf_cached(
                dataset=None,
                cfg2d_base_model=cfg2d_tmp,
                abs_offset=float(z_offset_q),
                example_input_position_dirpath=input_pos_path,
                cache_dir=tf_cache_dir,
                height=subtile_h,
                width=subtile_w,
                z_planes=Z_dim,
                verbose=False,
            )
            singular_system = _load_tf_singular_system(tf_zarr_path, device=device)
            _tf_cache_on_device[cache_key] = singular_system
            return singular_system

        # Loop over timepoints, constructing per-t reconstructions in a single pass
        for t in range(int(T)):
            focus_results: list = []
            recon_results: list = []
            for subtile_id, bounds in enumerate(subtile_bounds):
                r, c = subtile_id // grid_size, subtile_id % grid_size

                # Extract stacks for this subtile
                subtile_phase_stack = _extract_subtile_data(
                    focus_full_data[t : t + 1, :, :, :, :], bounds
                )
                subtile_bf_stack = _extract_subtile_data(
                    bf_recon_data[t : t + 1, :, :, :, :], bounds
                )

                # Compute focus and offsets directly
                focus_start = time.time()
                focus_idx, Z = _compute_subtile_focus(
                    subtile_phase_stack,
                    optical_params,
                    subtile_id,
                    enable_subpixel_precision=enable_subpixel_precision,
                    polynomial_fit_order=(
                        polynomial_fit_order if enable_subpixel_precision else None
                    ),
                    midband_fractions=midband_fractions,
                    device=device,
                )
                total_autofocus_time += time.time() - focus_start

                Z = int(Z)
                focus_idx_bounded = max(0, min(int(round(focus_idx)), Z - 1))
                z_focus_offset = float(focus_idx - (Z // 2))
                # Quantize z offset to one decimal place to improve TF cache reuse
                z_focus_offset_q = float(round(z_focus_offset, 1))

                # Focus slice for stitched focus image (use Phase3D recon, not BF)
                focus_slice = subtile_phase_stack[0, 0, focus_idx_bounded, :, :].copy()
                focus_results.append((focus_slice, bounds, (r, c)))

                # Direct GPU reconstruction — no temp zarr files
                recon_start = time.time()
                subtile_h = bounds[1] - bounds[0]
                subtile_w = bounds[3] - bounds[2]
                singular_system = _get_tf_on_device(z_focus_offset_q, subtile_h, subtile_w)
                reconstructed_2d, success = _reconstruct_subtile_2d_direct(
                    subtile_bf_stack,
                    singular_system,
                    regularization_strength=regularization_strength,
                    device=device,
                )
                total_recon_time += time.time() - recon_start

                # Diagnostic: log details for first subtile of first position
                if t == 0 and subtile_id == 0:
                    U, S, Vh = singular_system
                    bf_stats = f"min={subtile_bf_stack.min()}, max={subtile_bf_stack.max()}, dtype={subtile_bf_stack.dtype}"
                    tf_info = f"U={U.shape} {U.dtype}, S={S.shape} {S.dtype} [{S.min().item():.6f},{S.max().item():.6f}], Vh={Vh.shape} {Vh.dtype}"
                    r2d_nan = int(np.isnan(reconstructed_2d).sum())
                    r2d_inf = int((~np.isfinite(reconstructed_2d)).sum())
                    r2d_stats = f"shape={reconstructed_2d.shape}, min={np.nanmin(reconstructed_2d):.6f}, max={np.nanmax(reconstructed_2d):.6f}, nan={r2d_nan}, non-finite={r2d_inf}"
                    log_timing(f"[DIAG] {pos} subtile0: success={success}")
                    log_timing(f"[DIAG]   input: {bf_stats}")
                    log_timing(f"[DIAG]   TF: {tf_info}")
                    log_timing(f"[DIAG]   output: {r2d_stats}")

                meta = {
                    "subtile_id": subtile_id,
                    "bounds": bounds,
                    "focus_index": focus_idx_bounded,
                    "focus_index_float": float(focus_idx),
                    "z_stack_size": Z,
                    "z_focus_offset": z_focus_offset_q,
                    "reconstruction_success": success,
                    "time_index": int(t),
                }
                subtile_metadata.append(meta)

                if np.isnan(reconstructed_2d).any():
                    subtile_id_with_nan = subtile_id

                if success:
                    recon_results.append((reconstructed_2d.copy(), bounds, (r, c)))

            t_to_focus_results[int(t)] = focus_results
            t_to_2d_results[int(t)] = recon_results

        timings["subtile_processing"] = time.time() - subtile_start
        timings["autofocus_total"] = total_autofocus_time
        timings["reconstruction_total"] = total_recon_time
        timings["zarr_open_total"] = total_zarr_open_time

        log_timing(f"[TIMING] {pos} - Subtile processing breakdown:")
        log_timing(f"  Autofocus (all {n_subtiles} subtiles): {total_autofocus_time:.2f}s ({total_autofocus_time/n_subtiles*1000:.1f}ms/subtile)")
        log_timing(f"  2D Reconstruction (all subtiles): {total_recon_time:.2f}s ({total_recon_time/n_subtiles*1000:.1f}ms/subtile)")
        log_timing(f"  Zarr opens for metadata (256x): {total_zarr_open_time:.2f}s ({total_zarr_open_time/n_subtiles*1000:.1f}ms/open)")

    except Exception as e:
        # OOM means too many workers — fail fast, don't silently produce blank output
        if "out of memory" in str(e).lower() or "OutOfMemoryError" in type(e).__name__:
            raise

        reconstruction_failed = True
        import traceback

        print(f"[SubtileRecon][ERROR] Processing for {pos} failed: {e}")
        traceback.print_exc()
        # Initialize to empty metadata so downstream reporting doesn't error out
        subtile_metadata = []

    # Removed deprecated unblended diagnostic image composition

    # For each timepoint, stitch and write to unified store
    # Use direct zarr.open to avoid open_ome_zarr plate metadata walk (~5s per call)
    stitch_start = time.time()
    try:
        out_arr = zarr_lib.open(str(phase2d_store_path / pos / "0"), mode="r+")
        for t in range(int(T)):
            focus_list = t_to_focus_results.get(int(t), [])
            recon_list = t_to_2d_results.get(int(t), [])
            blending_exponent = getattr(cfg2d_base_model, "blending_exponent", 1.0)
            for stitch_list, channel_idx in ((focus_list, 1), (recon_list, 0)):
                if not stitch_list:
                    continue
                stitched = _stitch_subtiles_overlapping(
                    stitch_list,
                    (Y, X),
                    blend_pixels,
                    grid_size,
                    blending_exponent,
                )
                out_arr[int(t), int(channel_idx), 0, :, :] = stitched.astype(
                    np.float32, copy=False
                )
    finally:
        shutil.rmtree(temp_dir)

    timings["stitching_and_writing"] = time.time() - stitch_start
    timings["total_position_time"] = time.time() - pos_total_start

    log_timing(f"[TIMING] {pos} - Stitching and writing: {timings['stitching_and_writing']:.2f}s")
    log_timing(f"[TIMING] {pos} - TOTAL: {timings['total_position_time']:.2f}s")
    # Only log breakdown if all timing keys exist (they may be missing if subtile processing failed)
    if all(k in timings for k in ['autofocus_total', 'reconstruction_total', 'zarr_open_total']):
        log_timing(f"[TIMING] {pos} - Breakdown: Load={timings['data_loading']:.1f}s, Autofocus={timings['autofocus_total']:.1f}s, Recon={timings['reconstruction_total']:.1f}s, ZarrOpens={timings['zarr_open_total']:.1f}s, Stitch={timings['stitching_and_writing']:.1f}s")

    # Calculate summary statistics
    successful_subtiles = sum(
        1 for info in subtile_metadata if info and info["reconstruction_success"]
    )
    focus_indices = [info["focus_index"] for info in subtile_metadata if info]
    z_offsets = [info["z_focus_offset"] for info in subtile_metadata if info]

    result = {
        "position": pos,
        "n_subtiles": n_subtiles,
        "grid_size": grid_size,
        "blend_pixels": blend_pixels,
        "successful_subtiles": successful_subtiles,
        "failed_subtiles": n_subtiles - successful_subtiles,
        "subtile_metadata": subtile_metadata,
        "focus_index_range": (
            (min(focus_indices), max(focus_indices)) if focus_indices else (0, 0)
        ),
        "z_offset_range": (min(z_offsets), max(z_offsets)) if z_offsets else (0.0, 0.0),
        "reconstruction_success": successful_subtiles > 0,
    }

    return result