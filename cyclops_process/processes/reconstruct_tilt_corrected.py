"""Per-run tilt-corrected 3D+2D phase reconstruction.

Three CLI subcommands:

  calibrate  - Chain calibration along 4 cardinal spokes from center tile.
               Fits per-run model: zenith(r)=a*exp(b*r)+c, z_offset=const,
               azimuth=theta+pi.  Saves model.yaml + all_points.yaml.

  reconstruct - Apply calibrated model with per-subtile optimization.
                Optimizes tilt via fast 2D (isotropic_thin_3d), then applies
                to 3D (phase_thick_3d).  Produces dated 3D and 2D output
                stores with reflect padding and EDT blending.

Usage:
    # Calibrate
    uv run python -m cyclops_process.processes.reconstruct_tilt_corrected calibrate \
        --experiment ops0105_20260106 --well A/1

    # Reconstruct single FOV (debug)
    uv run python -m cyclops_process.processes.reconstruct_tilt_corrected reconstruct \
        --experiment ops0105_20260106 --well A/1 --process track --debug-n-positions 1

    # Submit calibration + reconstruction to SLURM (all wells)
    uv run python -m cyclops_process.processes.reconstruct_tilt_corrected submit \
        --experiment ops0105_20260106 --process track pheno --chunk-size 20

    # audit
    uv run python -m cyclops_process.processes.reconstruct_tilt_corrected audit --experiment 33 --process track pheno
"""

import argparse
import csv
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Cap BLAS/OMP thread pools to 1. Prevents numpy/torch CPU-side ops
# (esp. in waveorder loss_fn's coord-mask compute) from spawning 128 threads
# per operation → GIL/futex thrash in the multi-worker cgroup that runs
# ~30× slower per Adam iteration. Same pattern as iss.py and nuclei_pass.py.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import xarray as xr
import yaml
from iohub import open_ome_zarr
from scipy.ndimage import distance_transform_edt
from scipy.optimize import curve_fit

sys.path.insert(0, os.getcwd())

from cyclops_process.processes.reconstruct_subtile import (
    _stitch_subtiles_overlapping)
from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.profiling.decorators import versioned_function
from cyclops_utils.data.filesystem import get_experiment_wells
from cyclops_utils.io.zarr_precreate import create_hcs_store_fast
from cyclops_utils.io.zarr_utils import (
    _validate_all_positions_have_data,
    _write_plane_to_store,
)
from cyclops_process.paths import BASE_PATH

# ============================================================
# Constants
# ============================================================

OPS_BASE = Path(f"{BASE_PATH}")
DEFAULT_WELLS = ["A/1", "A/2", "A/3"]
CROP_SIZE = 256  # center crop for calibration tiles

# Subtile reconstruction defaults (validated on ops0105)
GRID_SIZE = 8
N_SUBTILES = GRID_SIZE * GRID_SIZE  # 64
BLEND_PIXELS = 50
REFLECT_PAD = 64
SUBTILE_NUM_ITERATIONS = 50
BATCH_SIZE = 4
REG = 0.001

# ============================================================
# Process-specific configurations
# ============================================================

PROCESS_CONFIGS = {
    "pheno": {
        "center_row": 29,
        "center_col": 29,
        "spokes": {
            "S": [{"tile": f"{r:03d}029", "row": r, "col": 29}
                  for r in [32, 36, 41, 46, 51, 54, 56]],
            "N": [{"tile": f"{r:03d}029", "row": r, "col": 29}
                  for r in [26, 22, 17, 12, 7, 4, 2]],
            "E": [{"tile": f"029{c:03d}", "row": 29, "col": c}
                  for c in [32, 36, 41, 46, 51, 54, 56]],
            "W": [{"tile": f"029{c:03d}", "row": 29, "col": c}
                  for c in [26, 22, 17, 12, 7, 4, 2]],
        },
        "universal_prior": {
            "z_focus_offset": -0.55,
            "tilt_angle_zenith": 0.13,
            "tilt_angle_azimuth": 0.0,
        },
        "transfer_function_params": dict(
            wavelength_illumination=0.45,
            yx_pixel_size=0.325,
            z_pixel_size=2.0,
            z_padding=5,
            index_of_refraction_media=1.0,
            numerical_aperture_detection=0.55,
            numerical_aperture_illumination=0.4,
            invert_phase_contrast=False,
        ),
        "apply_inverse_params": dict(
            reconstruction_algorithm="Tikhonov",
            regularization_strength=REG,
            TV_rho_strength=0.001,
            TV_iterations=1,
        ),
        "lr_z_offset": 0.05,
        "lr_zenith": 0.005,
        "lr_azimuth": 0.01,
        "swap_axes": False,  # 20x: atan2(dcol, drow)
        # NAdam optimizer step range (scales linearly with radius)
        "nadam_min_steps": 5,
        "nadam_max_steps": 15,
        # OpsDataset store keys
        "raw_store_key": "lc_20x",
        "phase3d_store_key": "lc_20x_phase",
        "output_3d_store_key": "lc_20x_phase_3d_optimized",
        "output_2d_store_key": "lc_20x_phase_2d_optimized",
    },
    "track": {
        "center_row": 8,
        "center_col": 8,
        "spokes": {
            "S": [{"tile": f"{r:03d}008", "row": r, "col": 8}
                  for r in [10, 12, 14]],
            "N": [{"tile": f"{r:03d}008", "row": r, "col": 8}
                  for r in [6, 4, 2]],
            "E": [{"tile": f"008{c:03d}", "row": 8, "col": c}
                  for c in [10, 12, 14]],
            "W": [{"tile": f"008{c:03d}", "row": 8, "col": c}
                  for c in [6, 4, 2]],
        },
        "universal_prior": {
            "z_focus_offset": 0.0,
            "tilt_angle_zenith": 0.05,
            "tilt_angle_azimuth": 0.0,
        },
        "transfer_function_params": dict(
            wavelength_illumination=0.45,
            yx_pixel_size=1.3,
            z_pixel_size=25.0,
            z_padding=5,
            index_of_refraction_media=1.0,
            numerical_aperture_detection=0.15,
            numerical_aperture_illumination=0.15,
            invert_phase_contrast=False,
        ),
        "apply_inverse_params": dict(
            reconstruction_algorithm="Tikhonov",
            regularization_strength=REG,
            TV_rho_strength=0.001,
            TV_iterations=1,
        ),
        "lr_z_offset": 0.05,
        "lr_zenith": 0.005,
        "lr_azimuth": 0.01,
        "swap_axes": True,  # 5x: atan2(drow, dcol) — different image orientation
        # NAdam optimizer step range (scales linearly with radius)
        # Track needs more steps than pheno — lower NA gives weaker signal
        "nadam_min_steps": 10,
        "nadam_max_steps": 25,
        "raw_store_key": "lc_5x_bf_corrected",
        "phase3d_store_key": "lc_5x_phase",
        "output_3d_store_key": "lc_5x_phase_3d_optimized",
        "output_2d_store_key": "lc_5x_phase_2d_optimized",
        # Validated spatial warm-start models for 5x (from run_5x_nsew_v4)
        "warmstart_z_plane": {"row_coeff": 0.065, "col_coeff": 0.036, "intercept": -0.674},
        "warmstart_zenith": {"base": 0.03, "slope": 0.015},
    },
}


def _get_process_config(process):
    """Get process config, defaulting to 'pheno' for backwards compatibility."""
    if process is None:
        process = "pheno"
    if process not in PROCESS_CONFIGS:
        raise ValueError(f"Unknown process '{process}'. Choose from: {list(PROCESS_CONFIGS.keys())}")
    return PROCESS_CONFIGS[process]


# Backwards-compatible module-level aliases (used by existing code)
# These reference 'pheno' config
TRANSFER_FUNCTION_PARAMS = PROCESS_CONFIGS["pheno"]["transfer_function_params"]
APPLY_INVERSE_PARAMS = PROCESS_CONFIGS["pheno"]["apply_inverse_params"]


# ============================================================
# Path resolution
# ============================================================

# OpsDataset may resolve paths to a fast partition (fast_ops/) that
# doesn't have the raw data for older experiments.  Fall back to the
# standard partition (ops/) when a path doesn't exist.
_FAST_PREFIX = f"{BASE_PATH}/"
_STD_PREFIX = f"{BASE_PATH}/"


def _resolve_path(p):
    """Return *p* if it exists, otherwise try swapping fast_ops/ -> ops/."""
    p = Path(p)
    if p.exists():
        return p
    s = str(p)
    if s.startswith(_FAST_PREFIX):
        alt = Path(_STD_PREFIX + s[len(_FAST_PREFIX) :])
        if alt.exists():
            return alt
    return p  # caller will get a clear FileNotFoundError


# ============================================================
# Geometry helpers
# ============================================================


def radial_distance(row, col, center_row=29, center_col=29):
    return np.sqrt((row - center_row) ** 2 + (col - center_col) ** 2)


def position_angle(row, col, center_row=29, center_col=29, swap_axes=False):
    # The azimuth convention depends on how the image axes map to plate
    # coordinates. 20x phenotyping: atan2(dcol, drow). 5x tracking: atan2(drow, dcol).
    if swap_axes:
        return np.arctan2(row - center_row, col - center_col)
    return np.arctan2(col - center_col, row - center_row)


def parse_tile_rc(tile_str):
    """Parse '029029' -> (29, 29)."""
    return int(tile_str[:3]), int(tile_str[3:])


# ============================================================
# Data loading helpers
# ============================================================


def _detect_bf_channel_index(zarr_path, fov_path):
    with open_ome_zarr(zarr_path, layout="hcs", mode="r") as store:
        channel_names = store[fov_path].channel_names
    return channel_names.index("BF")


def _load_bf_czyx(zarr_path, fov_path, bf_channel_index, crop_size=None):
    """Load BF data as CZYX xr.DataArray, optionally center-cropped."""
    with open_ome_zarr(zarr_path, layout="hcs", mode="r") as store:
        position = store[fov_path]
        bf_zyx = np.array(position.data[0, bf_channel_index])

    if crop_size is not None:
        _, Y, X = bf_zyx.shape
        y0 = (Y - crop_size) // 2
        x0 = (X - crop_size) // 2
        bf_zyx = bf_zyx[:, y0 : y0 + crop_size, x0 : x0 + crop_size]

    czyx_data = xr.DataArray(
        bf_zyx[None].astype(np.float32),
        dims=("c", "z", "y", "x"),
        coords={"c": ["BF"]},
    )
    return czyx_data


# ============================================================
# Calibration: optimize a single tile
# ============================================================


def _optimize_tile(czyx_data, init_params, lr, num_iterations,
                    tf_params=None, apply_inv_params=None):
    """Run tilt optimization on a single tile using isotropic_thin_3d.

    Uses the same reconstruction path as the production batched optimizer
    (isotropic_thin_3d.reconstruct with pupil_steepness=100) to avoid
    singularities when NA_ill >= NA_det.

    Returns dict with z_focus_offset, tilt_angle_zenith, tilt_angle_azimuth,
    final_loss.
    """
    import torch
    import torch.nn.functional as F
    from waveorder.models import isotropic_thin_3d
    from waveorder.optim.losses import MidbandPowerLossSettings, build_loss_fn

    if tf_params is None:
        tf_params = TRANSFER_FUNCTION_PARAMS
    if apply_inv_params is None:
        apply_inv_params = APPLY_INVERSE_PARAMS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reg = apply_inv_params.get("regularization_strength", REG)
    tf_no_z = {k: v for k, v in tf_params.items()
               if k not in ("z_pixel_size", "z_padding")}

    loss_fn = build_loss_fn(
        MidbandPowerLossSettings(),
        NA_det=tf_params["numerical_aperture_detection"],
        wavelength=tf_params["wavelength_illumination"],
        pixel_size=tf_params["yx_pixel_size"],
    )

    # czyx_data is xr.DataArray (1, Z, Y, X) — extract numpy
    bf_zyx = torch.tensor(
        czyx_data.values[0].astype(np.float32), device=device)
    Z = bf_zyx.shape[0]
    z_idx = -torch.arange(Z, device=device) + Z // 2

    # Optimizable parameters
    z_p = torch.tensor([init_params["z_focus_offset"]],
                       dtype=torch.float32, device=device, requires_grad=True)
    zen_p = torch.tensor([init_params["tilt_angle_zenith"]],
                         dtype=torch.float32, device=device, requires_grad=True)
    azi_p = torch.tensor([init_params["tilt_angle_azimuth"]],
                         dtype=torch.float32, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([
        {"params": [z_p], "lr": lr},
        {"params": [zen_p], "lr": lr * 0.25},
        {"params": [azi_p], "lr": lr * 0.5},
    ])

    last_good = (z_p.detach().clone(), zen_p.detach().clone(),
                 azi_p.detach().clone())
    last_loss = float("inf")

    bzyx = bf_zyx.unsqueeze(0)  # (1, Z, Y, X)
    pad = REFLECT_PAD
    bzyx_pad = F.pad(bzyx, (pad, pad, pad, pad), mode="reflect")

    for step in range(num_iterations):
        optimizer.zero_grad()
        z_positions = (z_idx + z_p[0]) * tf_params["z_pixel_size"]
        _, phase_byx = isotropic_thin_3d.reconstruct(
            bzyx_pad, z_position_list=z_positions,
            regularization_strength=reg,
            tilt_angle_zenith=zen_p,
            tilt_angle_azimuth=azi_p,
            pupil_steepness=100.0, **tf_no_z)
        phase_byx = phase_byx[:, pad:-pad, pad:-pad]

        if torch.isnan(phase_byx).any():
            break
        loss = loss_fn(phase_byx[0])
        if torch.isnan(loss):
            break
        last_good = (z_p.detach().clone(), zen_p.detach().clone(),
                     azi_p.detach().clone())
        last_loss = loss.item()
        loss.backward()
        optimizer.step()

    zg, zeng, azig = last_good
    return {
        "z_focus_offset": zg[0].item(),
        "tilt_angle_zenith": zeng[0].item(),
        "tilt_angle_azimuth": azig[0].item(),
        "final_loss": last_loss,
    }


# ============================================================
# Calibration: chain calibration + model fitting
# ============================================================


def _calibrate_spoke(direction, spoke_tiles, center_result,
                     cal_dir, raw_store_path, well, bf_idx,
                     center_row, center_col, tf_params, apply_inv_params, resume):
    """Process one spoke sequentially (tiles chain warm-start). Thread-safe."""
    spoke_points = []
    log_lines = [f"\n  Spoke {direction}:"]
    prev_params = {
        "z_focus_offset": center_result["z_focus_offset"],
        "tilt_angle_zenith": center_result["tilt_angle_zenith"],
        "tilt_angle_azimuth": center_result["tilt_angle_azimuth"],
    }

    for spoke_tile in spoke_tiles:
        tile = spoke_tile["tile"]
        row, col = spoke_tile["row"], spoke_tile["col"]
        r = radial_distance(row, col, center_row, center_col)
        theta = position_angle(row, col, center_row, center_col)

        tile_file = cal_dir / f"{direction}_{tile}_params.yaml"

        if resume and tile_file.exists():
            with open(tile_file) as f:
                result = yaml.safe_load(f)
            prev_params = {
                k: result[k]
                for k in [
                    "z_focus_offset",
                    "tilt_angle_zenith",
                    "tilt_angle_azimuth",
                ]
            }
            spoke_points.append(result)
            log_lines.append(
                f"    {tile} (r={r:.0f}): SKIPPED, "
                f"loss={result['final_loss']:.0f}"
            )
            continue

        # Check tile exists in store
        tile_path = raw_store_path / well / tile
        if not tile_path.exists():
            log_lines.append(f"    {tile} (r={r:.0f}): SKIPPING - tile not found")
            continue

        init_params = {
            "z_focus_offset": prev_params["z_focus_offset"],
            "tilt_angle_zenith": prev_params["tilt_angle_zenith"],
            "tilt_angle_azimuth": float(theta + np.pi),
        }
        log_lines.append(f"    {tile} (r={r:.0f}): warm from prev, 50 iter, lr=0.02...")

        czyx = _load_bf_czyx(
            str(raw_store_path), f"{well}/{tile}", bf_idx, crop_size=CROP_SIZE
        )
        t0 = time.time()
        result = _optimize_tile(
            czyx, init_params, lr=0.02, num_iterations=50,
            tf_params=tf_params, apply_inv_params=apply_inv_params,
        )
        elapsed = time.time() - t0

        result.update(
            tile=tile,
            row=row,
            col=col,
            radial_distance=float(r),
            direction=direction,
            theta=float(theta),
            elapsed_s=elapsed,
        )
        with open(tile_file, "w") as f:
            yaml.dump(result, f, default_flow_style=False)

        prev_params = {
            k: result[k]
            for k in [
                "z_focus_offset",
                "tilt_angle_zenith",
                "tilt_angle_azimuth",
            ]
        }
        spoke_points.append(result)
        log_lines.append(
            f"      z={result['z_focus_offset']:.4f} "
            f"zen={result['tilt_angle_zenith']:.4f} "
            f"azi={result['tilt_angle_azimuth']:.4f} "
            f"loss={result['final_loss']:.0f} ({elapsed:.1f}s)"
        )

    return spoke_points, log_lines


def calibrate(experiment, well=None, process=None, resume=False,
              output_dir=None):
    """Run chain calibration and fit per-run model for a single well.

    If the process config has hardcoded warm-start models (warmstart_zenith,
    warmstart_z_plane), writes those directly as the calibration model
    instead of running spoke optimization (which is unreliable at low NA).
    """
    pcfg = _get_process_config(process)
    dataset = OpsDataset(experiment)
    raw_store_path = _resolve_path(dataset.store_paths[pcfg["raw_store_key"]])
    phase3d_store_path = _resolve_path(dataset.store_paths[pcfg["phase3d_store_key"]])

    # Per-well output directories, namespaced by process
    well_tag = well.replace("/", "_")  # "A/2" -> "A_2"
    proc_name = process or "pheno"
    recon_base = Path(output_dir) if output_dir else phase3d_store_path.parent
    output_dir = recon_base / "tilt_calibration" / proc_name / well_tag
    cal_dir = output_dir / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)

    # If hardcoded warm-start models exist, write them directly as the
    # calibration model. At low NA (e.g. 5x, NA=0.15), spoke optimization
    # cannot reliably recover tilt angles — the midband loss is insensitive
    # to small tilts. The hardcoded models were validated experimentally.
    warmstart_zen = pcfg.get("warmstart_zenith")
    warmstart_z = pcfg.get("warmstart_z_plane")
    if warmstart_zen is not None:
        model = {
            "zenith": {
                "type": "linear",
                "base": warmstart_zen["base"],
                "slope": warmstart_zen["slope"],
            },
            "z_offset": {
                "type": "linear_plane",
                **(warmstart_z or {"row_coeff": 0.0, "col_coeff": 0.0, "intercept": 0.0}),
            },
            "azimuth": {"type": "theta_plus_pi"},
            "n_calibration_points": 0,
            "source": "hardcoded_warmstart",
        }
        model_path = output_dir / "model.yaml"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "w") as f:
            yaml.dump(model, f, default_flow_style=False, sort_keys=False)
        print(f"Tilt calibration for {experiment} well={well}")
        print(f"  Using hardcoded warm-start model (spoke optimization not reliable at low NA)")
        print(f"  zenith = {warmstart_zen['base']:.4f} + {warmstart_zen['slope']:.4f} * r")
        if warmstart_z:
            print(f"  z = {warmstart_z['row_coeff']:.4f}*row + {warmstart_z['col_coeff']:.4f}*col + {warmstart_z['intercept']:.4f}")
        print(f"  Model saved to {model_path}")
        return model

    center_tile = f"{pcfg['center_row']:03d}{pcfg['center_col']:03d}"
    bf_idx = _detect_bf_channel_index(str(raw_store_path), f"{well}/{center_tile}")

    # Read Z dimension from the raw store
    with open_ome_zarr(str(raw_store_path), layout="hcs", mode="r") as store:
        z_size = store[f"{well}/{center_tile}"].data.shape[2]

    print(f"Tilt calibration for {experiment} well={well}")
    print(f"  Raw BF store: {raw_store_path}")
    print(f"  Output dir:   {output_dir}")
    print(f"  BF channel:   {bf_idx}")
    print(f"  Z slices:     {z_size}")

    all_cal_points = []

    center_row = pcfg["center_row"]
    center_col = pcfg["center_col"]
    tf_params = pcfg["transfer_function_params"]
    apply_inv_params = pcfg["apply_inverse_params"]

    # --- Center tile ---
    center_file = cal_dir / "center_params.yaml"
    if resume and center_file.exists():
        with open(center_file) as f:
            center_result = yaml.safe_load(f)
        print(
            f"  Center ({center_tile}): SKIPPED (resume), "
            f"loss={center_result['final_loss']:.0f}"
        )
    else:
        print(f"  Center ({center_tile}): warm from universal prior, 50 iter, lr=0.1...")
        czyx = _load_bf_czyx(
            str(raw_store_path), f"{well}/{center_tile}", bf_idx, crop_size=CROP_SIZE
        )
        t0 = time.time()
        center_result = _optimize_tile(
            czyx, pcfg["universal_prior"], lr=0.1, num_iterations=50,
            tf_params=tf_params, apply_inv_params=apply_inv_params,
        )
        elapsed = time.time() - t0
        center_result.update(
            tile=center_tile,
            row=center_row,
            col=center_col,
            radial_distance=0.0,
            direction="center",
            elapsed_s=elapsed,
        )
        with open(center_file, "w") as f:
            yaml.dump(center_result, f, default_flow_style=False)
        print(
            f"    z={center_result['z_focus_offset']:.4f} "
            f"zen={center_result['tilt_angle_zenith']:.4f} "
            f"azi={center_result['tilt_angle_azimuth']:.4f} "
            f"loss={center_result['final_loss']:.0f} ({elapsed:.1f}s)"
        )

    # Validate center tile: if z drifted beyond half the z-stack, fall back.
    z_prior = pcfg["universal_prior"]["z_focus_offset"]
    max_z_drift = z_size // 2
    if abs(center_result["z_focus_offset"] - z_prior) > max_z_drift:
        print(
            f"    WARNING: center z drifted to {center_result['z_focus_offset']:.2f}, "
            f"falling back to universal prior ({z_prior})"
        )
        center_result["z_focus_offset"] = z_prior

    all_cal_points.append(center_result)

    # --- Spokes (parallel across directions, sequential within each) ---
    import torch
    from concurrent.futures import ThreadPoolExecutor
    n_spokes = len(pcfg["spokes"])
    threads_per_spoke = max(1, os.cpu_count() // max(n_spokes, 1))
    torch.set_num_threads(threads_per_spoke)

    with ThreadPoolExecutor(max_workers=n_spokes) as executor:
        futures = {
            executor.submit(
                _calibrate_spoke, direction, spoke_tiles, center_result,
                cal_dir, raw_store_path, well, bf_idx,
                center_row, center_col, tf_params, apply_inv_params, resume,
            ): direction
            for direction, spoke_tiles in pcfg["spokes"].items()
        }
        for future in futures:
            points, log_lines = future.result()
            for line in log_lines:
                print(line)
            all_cal_points.extend(points)

    # Save all calibration points
    all_points_path = cal_dir / "all_points.yaml"
    with open(all_points_path, "w") as f:
        yaml.dump(all_cal_points, f, default_flow_style=False)
    print(f"\n  Saved {len(all_cal_points)} calibration points to {all_points_path}")

    # --- Fit model ---
    model = _fit_model(all_cal_points)

    model_path = output_dir / "model.yaml"
    with open(model_path, "w") as f:
        yaml.dump(model, f, default_flow_style=False, sort_keys=False)
    print(f"  Model saved to {model_path}")

    # Print summary
    zen = model["zenith"]
    z = model["z_offset"]
    print(f"\n  Model summary:")
    if zen["type"] == "linear":
        print(f"    zenith  = {zen['base']:.4f} + {zen['slope']:.4f} * r")
    else:
        print(f"    zenith  = {zen['a']:.6f} * exp({zen['b']:.4f} * r) + {zen['c']:.4f}")
    if z["type"] == "linear_plane":
        print(f"    z_offset = {z['row_coeff']:.4f}*row + {z['col_coeff']:.4f}*col + {z['intercept']:.4f}")
    else:
        print(f"    z_offset = {z['value']:.4f} (constant)")
    print(f"    azimuth  = theta + pi")

    return model


def _fit_model(cal_points):
    """Fit zenith(r) = a*exp(b*r)+c and z_offset=const from calibration data."""
    r_vals = np.array([p["radial_distance"] for p in cal_points])
    zen_vals = np.array([abs(p["tilt_angle_zenith"]) for p in cal_points])
    z_vals = np.array([p["z_focus_offset"] for p in cal_points])

    # Zenith: a*exp(b*r) + c
    def zen_model(r, a, b, c):
        return a * np.exp(b * r) + c

    try:
        popt_zen, _ = curve_fit(
            zen_model, r_vals, zen_vals, p0=[0.0001, 0.3, 0.1], maxfev=5000
        )
    except RuntimeError:
        popt_zen = [0.0001, 0.28, float(np.mean(zen_vals[r_vals < 5]))]

    # z_offset: constant (mean)
    z_mean = float(np.mean(z_vals))
    z_std = float(np.std(z_vals))

    model = {
        "zenith": {
            "type": "a*exp(b*r)+c",
            "a": float(popt_zen[0]),
            "b": float(popt_zen[1]),
            "c": float(popt_zen[2]),
        },
        "z_offset": {
            "type": "constant",
            "value": z_mean,
            "std": z_std,
        },
        "azimuth": {"type": "theta_plus_pi"},
        "n_calibration_points": len(cal_points),
    }
    return model


# ============================================================
# Reconstruction: predict model parameters for a tile
# ============================================================


# ============================================================
# Reconstruction: subtile helpers
# ============================================================


def _create_subtile_bounds(Y, X, gs, bp):
    """Create overlapping subtile bounds for an 8x8 grid with blend overlap."""
    bh, bw = Y // gs, X // gs
    bounds = []
    for r in range(gs):
        for c in range(gs):
            y0 = r * bh - (bp if r > 0 else 0)
            y1 = ((r + 1) * bh if r < gs - 1 else Y) + (bp if r < gs - 1 else 0)
            x0 = c * bw - (bp if c > 0 else 0)
            x1 = ((c + 1) * bw if c < gs - 1 else X) + (bp if c < gs - 1 else 0)
            bounds.append((max(0, y0), min(Y, y1), max(0, x0), min(X, x1)))
    return bounds


def _gpu_optimize_tilt_params(
    bf_dev,
    z_idx,
    bounds,
    focus_offsets,
    init_zen,
    init_azi,
    init_z,
    tf_params,
    n_iter=SUBTILE_NUM_ITERATIONS,
    batch_size=BATCH_SIZE,
    reflect_pad=REFLECT_PAD,
    lr_z=0.05,
    lr_zenith=0.005,
    lr_azimuth=0.01,
):
    """Batched GPU optimization of tilt params via isotropic_thin_3d.

    Parameters
    ----------
    bf_dev : torch.Tensor
        BF z-stack (Z, Y, X) on GPU.
    z_idx : torch.Tensor
        Z index tensor (-arange(Z) + Z//2) on device.
    bounds : list of (y0, y1, x0, x1)
        Subtile bounds.
    focus_offsets : list of float
        Per-subtile focus offset from 3D phase.
    init_zen, init_azi, init_z : list of float
        Per-subtile initial tilt parameters.
    tf_params : dict
        Transfer function parameters.
    n_iter : int
        Adam iterations.
    batch_size : int
        Max tiles per batch.
    reflect_pad : int
        Reflect padding pixels.
    lr_z, lr_zenith, lr_azimuth : float
        Per-parameter learning rates.

    Returns
    -------
    dict
        Mapping subtile_id -> (z_offset, zenith, azimuth).
    """
    import torch
    import torch.nn.functional as F
    from waveorder.models import isotropic_thin_3d
    from waveorder.optim.losses import MidbandPowerLossSettings, build_loss_fn

    device = bf_dev.device
    reg = tf_params.get("regularization_strength", REG)
    tf_no_z = {k: v for k, v in tf_params.items()
               if k not in ("z_pixel_size", "z_padding", "regularization_strength")}

    loss_fn = build_loss_fn(
        MidbandPowerLossSettings(),
        NA_det=tf_params["numerical_aperture_detection"],
        wavelength=tf_params["wavelength_illumination"],
        pixel_size=tf_params["yx_pixel_size"],
    )

    # Group by focus offset for shared z_positions
    groups = defaultdict(list)
    for idx, (b, fo) in enumerate(zip(bounds, focus_offsets)):
        groups[fo].append((idx, b))

    opt_params = {}
    for offset, group in groups.items():
        for bs in range(0, len(group), batch_size):
            batch = group[bs:bs + batch_size]
            tiles = [bf_dev[:, y0:y1, x0:x1]
                     for _, (y0, y1, x0, x1) in batch]

            # Sub-group by shape for stacking
            sg = defaultdict(list)
            for t, (i, b) in zip(tiles, batch):
                sg[t.shape].append((t, i, b))

            for shape, items in sg.items():
                B = len(items)
                bzyx = torch.stack([t for t, _, _ in items])
                idxs = [i for _, i, _ in items]

                z_p = torch.tensor(
                    [(focus_offsets[i] + init_z[i]) / 2.0 for i in idxs],
                    dtype=torch.float32, device=device, requires_grad=True)
                zen_p = torch.tensor(
                    [init_zen[i] for i in idxs],
                    dtype=torch.float32, device=device, requires_grad=True)
                azi_p = torch.tensor(
                    [init_azi[i] for i in idxs],
                    dtype=torch.float32, device=device, requires_grad=True)
                optimizer = torch.optim.NAdam([
                    {"params": [z_p], "lr": lr_z * 2},
                    {"params": [zen_p], "lr": lr_zenith * 2},
                    {"params": [azi_p], "lr": lr_azimuth * 2},
                ])

                last_good = (z_p.detach().clone(), zen_p.detach().clone(),
                             azi_p.detach().clone())
                bzyx_pad = F.pad(bzyx, (reflect_pad,) * 4, mode="reflect")
                for step in range(n_iter):
                    optimizer.zero_grad()
                    z_positions = (z_idx + z_p.mean()) * tf_params["z_pixel_size"]
                    _, phase_byx = isotropic_thin_3d.reconstruct(
                        bzyx_pad, z_position_list=z_positions,
                        regularization_strength=reg,
                        tilt_angle_zenith=zen_p,
                        tilt_angle_azimuth=azi_p,
                        pupil_steepness=100.0, **tf_no_z)
                    # Crop padding
                    phase_byx = phase_byx[:, reflect_pad:-reflect_pad,
                                           reflect_pad:-reflect_pad]

                    if torch.isnan(phase_byx).any():
                        break
                    loss = torch.stack(
                        [loss_fn(phase_byx[b]) for b in range(B)]).sum()
                    if torch.isnan(loss):
                        break
                    last_good = (z_p.detach().clone(), zen_p.detach().clone(),
                                 azi_p.detach().clone())
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        z_half = len(z_idx) // 2
                        z_p.clamp_(-z_half, z_half)

                zg, zeng, azig = last_good
                for j, idx in enumerate(idxs):
                    opt_params[idx] = (zg[j].item(), zeng[j].item(),
                                       azig[j].item())

    return opt_params


def _reconstruct_subtiles_3d(
    bf_zyx_np,
    bounds,
    opt_params,
    tf_params,
    grid_size=GRID_SIZE,
    reflect_pad=REFLECT_PAD,
):
    """Reconstruct all subtiles with phase_thick_3d using optimized tilt.

    Groups subtiles by shape, computes batched transfer functions for each
    group, then applies per-subtile inverse via batched FFT.

    Parameters
    ----------
    bf_zyx_np : np.ndarray
        BF z-stack (Z, Y, X).
    bounds : list of (y0, y1, x0, x1)
        Subtile bounds.
    opt_params : dict
        subtile_id -> (z_offset, zenith, azimuth).
    tf_params : dict
        Transfer function parameters (must include z_padding).
    grid_size : int
        Grid dimension.
    reflect_pad : int
        Reflect padding pixels.

    Returns
    -------
    list of (np.ndarray, (y0,y1,x0,x1), (r,c))
        Per-subtile 3D volumes and metadata for stitching.
    """
    import torch
    import torch.nn.functional as F
    from waveorder.models import phase_thick_3d

    reg = tf_params.get("regularization_strength", REG)
    tf_3d = {k: v for k, v in tf_params.items()
             if k != "regularization_strength"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Group subtiles by padded shape for batched TF computation
    shape_groups = defaultdict(list)
    for sid, (y0, y1, x0, x1) in enumerate(bounds):
        subtile = torch.tensor(bf_zyx_np[:, y0:y1, x0:x1],
                               dtype=torch.float32)
        subtile_pad = F.pad(subtile.unsqueeze(0),
                            (reflect_pad,) * 4, mode="reflect").squeeze(0)
        shape_groups[subtile_pad.shape].append((sid, subtile_pad, (y0, y1, x0, x1)))

    results_dict = {}
    done = 0
    tf_sub_batch = 16  # sub-batch to fit 40GB+ GPU VRAM
    with torch.no_grad():
        for shape, group in shape_groups.items():
            sids = [g[0] for g in group]
            tiles = torch.stack([g[1] for g in group]).to(device)
            all_zeniths = torch.tensor([opt_params[s][1] for s in sids],
                                       device=device)
            all_azimuths = torch.tensor([opt_params[s][2] for s in sids],
                                        device=device)

            for sb_start in range(0, len(sids), tf_sub_batch):
                sb_end = min(sb_start + tf_sub_batch, len(sids))
                sb_sids = sids[sb_start:sb_end]
                sb_tiles = tiles[sb_start:sb_end]
                sb_zen = all_zeniths[sb_start:sb_end]
                sb_azi = all_azimuths[sb_start:sb_end]

                # Batched TF on GPU (shared optics + batched WOTF)
                real_tfs, imag_tfs = phase_thick_3d.calculate_transfer_function(
                    zyx_shape=shape,
                    tilt_angle_zenith=sb_zen,
                    tilt_angle_azimuth=sb_azi,
                    **tf_3d)
                real_tfs = real_tfs.to(device)
                imag_tfs = imag_tfs.to(device)

                # Batched inverse on GPU
                phase_bzyx = phase_thick_3d.apply_inverse_transfer_function(
                    sb_tiles, real_tfs, imag_tfs,
                    z_padding=tf_params.get("z_padding", 5),
                    regularization_strength=reg)
                phase_bzyx = phase_bzyx[:, :, reflect_pad:-reflect_pad,
                                               reflect_pad:-reflect_pad]

                for i, sid in enumerate(sb_sids):
                    r_g, c_g = sid // grid_size, sid % grid_size
                    results_dict[sid] = (phase_bzyx[i].cpu().numpy(),
                                         group[sb_start + i][2], (r_g, c_g))

            done += len(group)
            print(f"      3D recon: {done}/{len(bounds)}", flush=True)

    return [results_dict[sid] for sid in sorted(results_dict)]


def _stitch_subtiles_3d(results, full_shape, blend_pixels, grid_size):
    """Stitch 3D subtile results with EDT blending.

    Parameters
    ----------
    results : list of (zyx_data, bounds, (r, c))
        Per-subtile 3D volumes.
    full_shape : (Z, Y, X)
        Output volume shape.
    blend_pixels : int
        Overlap for blending.
    grid_size : int
        Grid dimension.

    Returns
    -------
    np.ndarray
        Stitched (Z, Y, X) volume.
    """
    Z, Y, X = full_shape
    output = np.zeros((Z, Y, X), dtype=np.float32)
    weight_sum = np.zeros((Y, X), dtype=np.float32)

    # Compute 2D weights once, apply per z-slice
    weight_maps = {}
    for data, bounds, (r, c) in results:
        y0, y1, x0, x1 = bounds
        h, w = y1 - y0, x1 - x0
        mask = np.zeros((h, w), dtype=bool)
        if h > 2 and w > 2:
            mask[1:-1, 1:-1] = True
        weights = distance_transform_edt(mask).astype(np.float32) + 1e-6
        weight_maps[(r, c)] = weights
        weight_sum[y0:y1, x0:x1] += weights

    valid = weight_sum > 0
    for data, bounds, (r, c) in results:
        y0, y1, x0, x1 = bounds
        weights = weight_maps[(r, c)]
        for z in range(Z):
            output[z, y0:y1, x0:x1] += data[z] * weights
    for z in range(Z):
        output[z][valid] /= weight_sum[valid]
    return output


def _reconstruct_subtiles_2d_final(
    bf_dev,
    z_idx,
    bounds,
    opt_params,
    tf_params,
    grid_size=GRID_SIZE,
    reflect_pad=REFLECT_PAD,
    batch_size=BATCH_SIZE,
):
    """Final 2D reconstruction with optimized params + reflect padding.

    Parameters
    ----------
    bf_dev : torch.Tensor
        BF z-stack (Z, Y, X) on GPU.
    z_idx : torch.Tensor
        Z index tensor on device.
    bounds : list of (y0, y1, x0, x1)
        Subtile bounds.
    opt_params : dict
        subtile_id -> (z_offset, zenith, azimuth).
    tf_params : dict
        Transfer function parameters.

    Returns
    -------
    list of (np.ndarray, (y0,y1,x0,x1), (r,c))
        Per-subtile 2D phase images and metadata for stitching.
    """
    import torch
    import torch.nn.functional as F
    from waveorder.models import isotropic_thin_3d

    device = bf_dev.device
    reg = tf_params.get("regularization_strength", REG)
    tf_no_z = {k: v for k, v in tf_params.items()
               if k not in ("z_pixel_size", "z_padding", "regularization_strength")}

    # Group by focus offset
    groups = defaultdict(list)
    for idx, b in enumerate(bounds):
        fo = opt_params[idx][0]  # z_offset
        groups[round(fo, 1)].append((idx, b))

    results = [None] * len(bounds)
    for offset, group in groups.items():
        for bs in range(0, len(group), batch_size):
            batch = group[bs:bs + batch_size]
            tiles = [bf_dev[:, y0:y1, x0:x1]
                     for _, (y0, y1, x0, x1) in batch]

            sg = defaultdict(list)
            for t, (i, b) in zip(tiles, batch):
                sg[t.shape].append((t, i, b))

            for shape, items in sg.items():
                sg_tiles = [t for t, _, _ in items]
                sg_data = [(i, b) for _, i, b in items]
                bzyx = torch.stack(sg_tiles)
                b_zen = torch.tensor(
                    [opt_params[i][1] for i, _ in sg_data], device=device)
                b_azi = torch.tensor(
                    [opt_params[i][2] for i, _ in sg_data], device=device)
                mean_z = np.mean([opt_params[i][0] for i, _ in sg_data])
                z_positions = (z_idx + mean_z) * tf_params["z_pixel_size"]

                with torch.no_grad():
                    bzyx_pad = F.pad(bzyx, (reflect_pad,) * 4, mode="reflect")
                    _, phase_byx = isotropic_thin_3d.reconstruct(
                        bzyx_pad, z_position_list=z_positions,
                        regularization_strength=reg,
                        tilt_angle_zenith=b_zen,
                        tilt_angle_azimuth=b_azi,
                        **tf_no_z)
                    phase_byx = phase_byx[:, reflect_pad:-reflect_pad,
                                           reflect_pad:-reflect_pad]

                    for j, (idx, b) in enumerate(sg_data):
                        tile_phase = phase_byx[j]
                        if torch.isnan(tile_phase).any():
                            # Fallback: no tilt
                            _, tile_phase = isotropic_thin_3d.reconstruct(
                                F.pad(sg_tiles[j].unsqueeze(0),
                                      (reflect_pad,) * 4, mode="reflect"),
                                z_position_list=z_positions,
                                regularization_strength=reg, **tf_no_z)
                            tile_phase = tile_phase[0, reflect_pad:-reflect_pad,
                                                       reflect_pad:-reflect_pad]
                        r_g, c_g = idx // grid_size, idx % grid_size
                        results[idx] = (tile_phase.cpu().numpy(), b, (r_g, c_g))

    return results


def _write_volume_to_store(store_path, pos, volume, channel_index, time_index):
    """Write a (Z, Y, X) volume into store at [time, channel, :, :, :].

    Parameters
    ----------
    store_path : Path
        Path to OME-Zarr store.
    pos : str
        Position path (e.g. "A/1/008008").
    volume : np.ndarray
        3D volume (Z, Y, X).
    channel_index : int
        Channel index in store.
    time_index : int
        Time index in store.
    """
    with open_ome_zarr(store_path, mode="r+") as store:
        store[pos]["0"][int(time_index), int(channel_index)] = volume.astype(
            np.float32, copy=True
        )


def _collect_tilt_rows(opt_params, grid_size, pos):
    """Convert opt_params dict to list of CSV row dicts.

    Parameters
    ----------
    opt_params : dict
        subtile_id -> (z_offset, zenith, azimuth).
    grid_size : int
        Grid dimension.
    pos : str
        Position path (e.g. "A/1/008008").

    Returns
    -------
    list of dict
        One dict per subtile with keys: position, tile_row, tile_col,
        subtile_id, sub_row, sub_col, z_offset, zenith, azimuth.
    """
    tile_str = pos.split("/")[-1]
    tile_row, tile_col = int(tile_str[:3]), int(tile_str[3:])
    rows = []
    for sid in sorted(opt_params.keys()):
        r_g, c_g = sid // grid_size, sid % grid_size
        z, zen, azi = opt_params[sid]
        rows.append({
            "position": pos,
            "tile_row": tile_row,
            "tile_col": tile_col,
            "subtile_id": sid,
            "sub_row": r_g,
            "sub_col": c_g,
            "z_offset": round(z, 4),
            "zenith": round(zen, 6),
            "azimuth": round(azi, 4),
        })
    return rows


def _save_tilt_csv(all_rows, csv_path, append=False):
    """Write all tilt parameters to a single CSV.

    Parameters
    ----------
    all_rows : list of dict
        Accumulated rows from _collect_tilt_rows across all positions.
    csv_path : Path
        Output CSV path.
    append : bool
        Append to an existing CSV instead of overwriting it.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["position", "tile_row", "tile_col", "subtile_id",
                  "sub_row", "sub_col", "z_offset", "zenith", "azimuth"]
    write_header = not (append and csv_path.exists())
    with open(csv_path, "a" if append else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(all_rows)
    print(f"  Tilt params: {len(all_rows)} rows -> {csv_path}")


def _collect_and_summarize_tilt(tilt_dir, grid_size, title):
    """Collect per-chunk CSVs and generate well-level summary.

    Parameters
    ----------
    tilt_dir : Path
        Well-level tilt calibration directory (contains csvs/ subdirectory).
    grid_size : int
        Subtile grid dimension.
    title : str
        Plot title.
    """
    import pandas as pd

    tilt_dir = Path(tilt_dir)
    csv_dir = tilt_dir / "csvs"
    csvs = sorted(csv_dir.glob("tilt_params_*.csv"))
    if not csvs:
        print(f"  No tilt CSVs found in {csv_dir}")
        return

    dfs = []
    for c in csvs:
        try:
            d = pd.read_csv(c)
            if len(d) > 0:
                dfs.append(d)
        except Exception:
            continue
    if not dfs:
        print(f"  All tilt CSVs empty in {csv_dir}")
        return
    df = pd.concat(dfs, ignore_index=True)

    # Save combined CSV at the well level
    combined_path = tilt_dir / "tilt_params_all.csv"
    df.to_csv(combined_path, index=False)
    print(f"  Tilt params: {len(df)} rows -> {combined_path}")

    # Generate well-level summary plot
    _save_tilt_summary_plot(
        df.to_dict("records"), grid_size,
        tilt_dir / "tilt_summary_well.png", title=title)


def _save_tilt_summary_plot(all_rows, grid_size, plot_path, title=None):
    """Save 3x3 summary plot of tilt parameters.

    Columns: z_offset, zenith, azimuth
    Rows:
      1. Full well at subtile resolution (tile_row*gs+sub_row, tile_col*gs+sub_col)
      2. Full well averaged per FOV (one value per tile)
      3. Single FOV averaged across FOVs (grid_size x grid_size)

    Parameters
    ----------
    all_rows : list of dict
        From _collect_tilt_rows.
    grid_size : int
        Subtile grid dimension.
    plot_path : Path
        Output PNG path.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_path = Path(plot_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        params = ["z_offset", "zenith", "azimuth"]

        # Build arrays
        tile_rows = sorted(set(r["tile_row"] for r in all_rows))
        tile_cols = sorted(set(r["tile_col"] for r in all_rows))
        tr_map = {v: i for i, v in enumerate(tile_rows)}
        tc_map = {v: i for i, v in enumerate(tile_cols)}
        n_tr, n_tc = len(tile_rows), len(tile_cols)

        # Row 1: full well at subtile resolution
        hi_h, hi_w = n_tr * grid_size, n_tc * grid_size
        hi_res = {p: np.full((hi_h, hi_w), np.nan) for p in params}

        # Row 2: per-FOV average
        fov_avg = {p: np.full((n_tr, n_tc), np.nan) for p in params}
        fov_accum = defaultdict(lambda: defaultdict(list))

        # Row 3: subtile average across FOVs
        sub_avg = {p: np.zeros((grid_size, grid_size)) for p in params}
        sub_count = np.zeros((grid_size, grid_size))

        for r in all_rows:
            tr_i = tr_map[r["tile_row"]]
            tc_i = tc_map[r["tile_col"]]
            sr, sc = r["sub_row"], r["sub_col"]
            for p in params:
                val = r[p]
                hi_res[p][tr_i * grid_size + sr, tc_i * grid_size + sc] = val
                fov_accum[(tr_i, tc_i)][p].append(val)
                sub_avg[p][sr, sc] += val
            sub_count[sr, sc] += 1

        for (tr_i, tc_i), pvals in fov_accum.items():
            for p in params:
                fov_avg[p][tr_i, tc_i] = np.mean(pvals[p])

        valid = sub_count > 0
        for p in params:
            sub_avg[p][valid] /= sub_count[valid]
            sub_avg[p][~valid] = np.nan

        # Plot
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        row_labels = ["subtile resolution", "per-FOV average", "subtile avg across FOVs"]
        for col_i, p in enumerate(params):
            for row_i, (data, label) in enumerate([
                (hi_res[p], row_labels[0]),
                (fov_avg[p], row_labels[1]),
                (sub_avg[p], row_labels[2]),
            ]):
                ax = axes[row_i, col_i]
                vmin, vmax = np.nanpercentile(data, [2, 98])
                im = ax.imshow(data, cmap="viridis", vmin=vmin, vmax=vmax,
                               aspect="equal", interpolation="nearest")
                plt.colorbar(im, ax=ax, fraction=0.046)
                if row_i == 0:
                    ax.set_title(p, fontsize=12, fontweight="bold")
                if col_i == 0:
                    ax.set_ylabel(label, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])

        n_fovs = len(set(r['position'] for r in all_rows))
        suptitle = title or f"Tilt parameters ({n_fovs} FOVs)"
        if title and n_fovs:
            suptitle = f"{title} — {n_fovs} FOVs"
        fig.suptitle(suptitle,
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Tilt summary plot: {plot_path}")
    except Exception as e:
        print(f"  WARNING: could not save tilt summary plot: {e}")


def _reconstruct_position_3d_2d(
    pos,
    raw_store_path,
    phase3d_store_path,
    out_3d_path,
    out_2d_path,
    pcfg,
    model,
    time_index=0,
    device="auto",
):
    """Full per-position pipeline: optimize tilt via 2D, apply to 3D.

    Parameters
    ----------
    pos : str
        Position path (e.g. "A/1/008008").
    raw_store_path : Path
        Path to raw BF store.
    phase3d_store_path : Path
        Path to existing 3D phase store (for focus finding).
    out_3d_path : Path
        Output 3D store path.
    out_2d_path : Path
        Output 2D store path.
    pcfg : dict
        Process config from PROCESS_CONFIGS.
    model : dict
        Calibrated tilt model.
    time_index : int
        Time index to process.
    device : str
        "auto", "cuda", or "cpu".
    """
    import torch
    import torch.nn.functional as F
    from waveorder.focus import focus_from_transverse_band
    from waveorder.models import isotropic_thin_3d

    tf_params = pcfg["transfer_function_params"]
    center_row = pcfg["center_row"]
    center_col = pcfg["center_col"]
    swap_axes = pcfg["swap_axes"]

    parts = pos.split("/")
    tile_str = parts[-1]
    row, col = parse_tile_rc(tile_str)

    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Step 1: Load data
    with open_ome_zarr(phase3d_store_path, mode="r") as store:
        phase3d_zyx = np.asarray(store[pos]["0"][time_index, 0], dtype=np.float32)

    with open_ome_zarr(raw_store_path, mode="r") as store:
        channel_names = store.channel_names
        try:
            bf_idx = channel_names.index("BF")
        except ValueError:
            bf_idx = 0
        bf_zyx = np.asarray(store[pos]["0"][time_index, bf_idx], dtype=np.float32)

    Z, Y, X = bf_zyx.shape
    gs = GRID_SIZE
    bounds = _create_subtile_bounds(Y, X, gs, BLEND_PIXELS)

    pixel_mm = tf_params["yx_pixel_size"] * 1e-3
    fov_mm = 2048 * pixel_mm

    # Step 2: Focus finding per subtile
    focus_offsets = []
    focus_results = []
    for sid, (y0, y1, x0, x1) in enumerate(bounds):
        r_g, c_g = sid // gs, sid % gs
        subtile = phase3d_zyx[:, y0:y1, x0:x1]
        if np.all(subtile == 0) or np.all(np.isnan(subtile)):
            raise ValueError(
                f"Subtile {sid} (y={y0}:{y1}, x={x0}:{x1}) is all zeros/NaN. "
                f"Check upstream 3D phase reconstruction."
            )
        try:
            fi = focus_from_transverse_band(
                subtile,
                NA_det=tf_params["numerical_aperture_detection"],
                lambda_ill=tf_params["wavelength_illumination"],
                pixel_size=tf_params["yx_pixel_size"],
                midband_fractions=(0.125, 0.25), mode="max",
                plot_path=None, threshold_FWHM=0,
                enable_subpixel_precision=True, polynomial_fit_order=3)
            if fi is None:
                fi = float(Z // 2)
        except Exception:
            fi = float(Z // 2)
        focus_idx = max(0, min(int(round(fi)), Z - 1))
        focus_offsets.append(round(float(fi) - Z // 2, 1))
        focus_results.append((phase3d_zyx[focus_idx, y0:y1, x0:x1].copy(),
                              (y0, y1, x0, x1), (r_g, c_g)))

    # Step 3: Per-subtile warm-start
    # Use validated spatial models when available, fall back to calibration model
    warmstart_z = pcfg.get("warmstart_z_plane")
    warmstart_zen = pcfg.get("warmstart_zenith")

    init_zeniths, init_azimuths, init_z_offsets = [], [], []
    for sid, (y0, y1, x0, x1) in enumerate(bounds):
        sy, sx = (y0 + y1) / 2, (x0 + x1) / 2
        sub_row = row + (sy - Y / 2) * pixel_mm / fov_mm
        sub_col = col + (sx - X / 2) * pixel_mm / fov_mm
        r_tilt = radial_distance(sub_row, sub_col, center_row, center_col)

        if warmstart_zen is not None:
            # Validated spatial model: zenith = base + slope * r
            zen_init = warmstart_zen["base"] + warmstart_zen["slope"] * r_tilt
        else:
            # Fall back to calibration model
            zen = model["zenith"]
            if zen["type"] == "linear":
                zen_init = zen["base"] + zen["slope"] * r_tilt
            else:
                zen_init = zen["a"] * np.exp(zen["b"] * r_tilt) + zen["c"]

        if warmstart_z is not None:
            # Validated spatial model: z = row_coeff*r + col_coeff*c + intercept
            z_init = (warmstart_z["row_coeff"] * sub_row
                      + warmstart_z["col_coeff"] * sub_col
                      + warmstart_z["intercept"])
        else:
            z_model = model["z_offset"]
            if z_model.get("type") == "linear_plane":
                z_init = (z_model["row_coeff"] * sub_row
                          + z_model["col_coeff"] * sub_col
                          + z_model["intercept"])
            else:
                z_init = z_model["value"]

        if r_tilt > 0.1:
            if swap_axes:
                azi_init = float(
                    np.arctan2(sub_row - center_row, sub_col - center_col) + np.pi)
            else:
                azi_init = float(
                    np.arctan2(sub_col - center_col, sub_row - center_row) + np.pi)
        else:
            azi_init = 0.0

        init_zeniths.append(float(zen_init))
        init_azimuths.append(azi_init)
        init_z_offsets.append(float(z_init))

    # Step 4: Batched GPU optimization via isotropic_thin_3d
    bf_dev = torch.tensor(bf_zyx, dtype=torch.float32, device=device)
    z_idx = -torch.arange(Z, device=device) + Z // 2

    # Scale optimization steps with radius: more steps where the warm-start
    # is further from optimal. NAdam 2x lr converges faster than Adam.
    r_tile = radial_distance(row, col, center_row, center_col)
    r_max = max(center_row, center_col)
    min_steps = pcfg.get("nadam_min_steps", 10)
    max_steps = pcfg.get("nadam_max_steps", 25)
    n_opt_steps = int(min_steps + (max_steps - min_steps) * min(r_tile / r_max, 1.0))

    print(f"    [{pos}] optimizing {len(bounds)} subtiles "
          f"({n_opt_steps} steps, r={r_tile:.1f}, {device})...", flush=True)
    t0 = time.time()
    opt_params = _gpu_optimize_tilt_params(
        bf_dev, z_idx, bounds, focus_offsets,
        init_zen=init_zeniths, init_azi=init_azimuths, init_z=init_z_offsets,
        tf_params=tf_params,
        n_iter=n_opt_steps,
        lr_z=pcfg["lr_z_offset"],
        lr_zenith=pcfg["lr_zenith"],
        lr_azimuth=pcfg["lr_azimuth"],
    )
    print(f"    [{pos}] 2D optimization done in {time.time()-t0:.1f}s", flush=True)

    # Step 5: 3D reconstruction with phase_thick_3d + optimized tilt
    t0 = time.time()
    results_3d = _reconstruct_subtiles_3d(
        bf_zyx, bounds, opt_params, tf_params, grid_size=gs)
    print(f"    [{pos}] 3D reconstruction done in {time.time()-t0:.1f}s", flush=True)

    # Step 6: Best-focus slice from tilt-corrected 3D → 2D focus
    focus_from_3d = []
    for sid, (data_3d, b, (r_g, c_g)) in enumerate(results_3d):
        fi = focus_offsets[sid]
        focus_idx = max(0, min(int(round(fi + Z // 2)), Z - 1))
        focus_from_3d.append((data_3d[focus_idx], b, (r_g, c_g)))

    # Step 7: Final 2D reconstruction with optimized params
    t0 = time.time()
    results_2d = _reconstruct_subtiles_2d_final(
        bf_dev, z_idx, bounds, opt_params, tf_params, grid_size=gs)
    print(f"    [{pos}] 2D reconstruction done in {time.time()-t0:.1f}s", flush=True)

    # Step 7b: Stitch
    stitched_3d = _stitch_subtiles_3d(results_3d, (Z, Y, X), BLEND_PIXELS, gs)
    stitched_2d = _stitch_subtiles_overlapping(
        results_2d, (Y, X), BLEND_PIXELS, gs)
    stitched_focus = _stitch_subtiles_overlapping(
        focus_from_3d, (Y, X), BLEND_PIXELS, gs)

    # Step 8: Write outputs
    _write_volume_to_store(out_3d_path, pos, stitched_3d,
                           channel_index=0, time_index=time_index)
    _write_plane_to_store(out_2d_path, pos, stitched_2d,
                          channel_index=0, time_index=time_index)
    _write_plane_to_store(out_2d_path, pos, stitched_focus,
                          channel_index=1, time_index=time_index)

    # Free GPU memory
    del bf_dev
    # torch.cuda.empty_cache()  # disabled: forces OS dealloc on every position; caching allocator reuses the block automatically

    return {
        "position": pos,
        "tile": tile_str,
        "row": row,
        "col": col,
        "n_subtiles": len(bounds),
        "opt_params": opt_params,
    }


# ============================================================
# Multi-GPU subprocess worker (top-level so it's picklable for spawn)
# ============================================================


def _detect_workers_per_gpu_from_vram(proc_name: str) -> int:
    """Pick a safe workers/GPU based on GPU model + smallest GPU's VRAM.

    The pheno tilt-recon worker holds ~14–25 GB VRAM during a 3D recon
    burst; the SM-bound steady state is hit around 2–3 workers/GPU on
    big-VRAM cards. Measured ceilings (2026-06):

      - H100 / H200 / Blackwell (80 GB+):          3 workers
      - A100-80GB / A40 / L40S / A6000 (45 GB+):   2 workers
      - A100-40GB / smaller VRAM:                  1 worker

    The 45 GB threshold (was 40 GB) excludes the A100-SXM4-40GB variant
    while including A40/A6000/L40S (which report ~47.99 GiB total). The
    earlier 48 GB threshold misfired on those cards.

    Blackwell joined the 3-worker tier 2026-06-30 after live measurement
    showed it sustains 3 workers/GPU cleanly under the updated tilt-recon
    workload (the earlier 2-worker cap was over-conservative).

    Track fan-out is per-well (small) — keep it single-worker.
    """
    if proc_name == "track":
        return 1
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        rows = [r.strip() for r in out.strip().splitlines() if r.strip()]
        if not rows:
            return 1
        names, mems = [], []
        for r in rows:
            parts = [p.strip() for p in r.split(",")]
            if len(parts) >= 2:
                names.append(parts[0])
                try:
                    mems.append(int(parts[1]))
                except ValueError:
                    pass
        if not mems:
            return 1
        min_gib = min(mems) / 1024
        # H100 / H200 / Blackwell — 3 workers/GPU (gated on model name
        # rather than raw VRAM so we don't accidentally over-subscribe
        # A100-80GB which has similar capacity but lower memory bandwidth).
        if names and all(
            ("H100" in n) or ("H200" in n) or ("Blackwell" in n)
            for n in names
        ):
            return 3
        # A100-80GB / A40 / L40S / A6000 — 2 workers fit above 45 GB.
        # Threshold is 45 (not 48) because A40 / A6000 / L40S report
        # ``memory.total = 49140 MiB`` (driver reserves 12 MiB framebuffer)
        # which is 47.99 GiB → would fail a ``>= 48`` check. The 45 GiB
        # threshold correctly excludes A100-40GB (exactly 40.0 GiB
        # reported) — where 2 workers OOM during long chunks — and
        # includes A40 / A6000 / L40S / A100-80GB / etc.
        if min_gib >= 45:
            return 2
        return 1
    except Exception:
        return 1


def _workqueue_build(
    queue_root,
    well_positions_by_well,
    bucket_size,
):
    """Build a work-stealing queue on the shared filesystem.

    Layout::

        queue_root/
          queue/B_NNNN/info.json   (well + list of (pos_idx, pos_name))
          taken/                    (claimed by some chunk, in progress)
          done/                     (processed; kept for debugging)

    Buckets contain positions from a SINGLE well so a worker subprocess
    can model-switch only when transitioning between buckets. Each
    bucket holds up to ``bucket_size`` positions; we pack within-well
    then start a new bucket per remaining well.

    Returns the total number of buckets created.
    """
    import json
    queue_root = Path(queue_root)
    queue_dir = queue_root / "queue"
    taken_dir = queue_root / "taken"
    done_dir  = queue_root / "done"
    queue_dir.mkdir(parents=True, exist_ok=True)
    taken_dir.mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(parents=True, exist_ok=True)

    bucket_idx = 0
    for well, positions in well_positions_by_well.items():
        for i in range(0, len(positions), bucket_size):
            bucket_positions = positions[i : i + bucket_size]
            bdir = queue_dir / f"B_{bucket_idx:05d}"
            bdir.mkdir(exist_ok=True)
            with open(bdir / "info.json", "w") as f:
                json.dump(
                    {"well": well, "positions": bucket_positions},
                    f,
                )
            bucket_idx += 1

    return bucket_idx


def _workqueue_claim(queue_dir, taken_dir, chunk_tag):
    """Atomically claim a bucket via os.rename (POSIX-atomic on NFS).

    Returns the new path in ``taken_dir`` on success, or None if the
    queue is empty. Races are tolerated: if our chosen bucket got
    snatched between listdir and rename, we just try the next one.
    """
    queue_dir = Path(queue_dir)
    taken_dir = Path(taken_dir)
    if not queue_dir.exists():
        return None
    # listdir may be stale across NFS clients — that's fine; rename is
    # the only mutating op and it's atomic.
    for bucket in sorted(queue_dir.iterdir()):
        if not bucket.is_dir():
            continue
        target = taken_dir / f"{chunk_tag}__{bucket.name}"
        try:
            os.rename(bucket, target)
            return target
        except (FileNotFoundError, OSError):
            # Lost the race or bucket vanished — try next
            continue
    return None


def _workqueue_release(taken_path, done_dir):
    """Move a finished bucket from ``taken/`` to ``done/``."""
    done_dir = Path(done_dir)
    target = done_dir / taken_path.name
    try:
        os.rename(taken_path, target)
    except (FileNotFoundError, OSError) as e:
        # Best-effort; not fatal — done/ is for debugging only.
        print(f"  [workqueue] release failed for {taken_path.name}: {e}",
              flush=True)


def _workqueue_chunk_tag():
    """A unique tag for this chunk's claims. Survives crashes since the
    SLURM array task id is stable across the job's lifetime."""
    arr_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    host = os.environ.get("HOSTNAME") or "unknown"
    if arr_id:
        return f"a{arr_id}_{host}"
    return f"j{job_id or os.getpid()}_{host}"


def _workqueue_batch_size_for_gpu(proc_name: str) -> int:
    """Return a per-claim batch size based on the worker's actual GPU class.

    Faster cards claim bigger batches so the ratio of NFS-claim overhead to
    GPU compute stays consistent across the heterogeneous cluster. Workers
    are independent (no parent-coordination), so this is purely a per-worker
    tuning knob — no divisibility constraint vs workers_per_gpu.
    """
    if proc_name == "track":
        return 3
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        rows = [r.strip() for r in out.strip().splitlines() if r.strip()]
        names = []
        mems = []
        for r in rows:
            parts = [p.strip() for p in r.split(",")]
            if len(parts) >= 2:
                names.append(parts[0])
                try:
                    mems.append(int(parts[1]))
                except ValueError:
                    pass
        if not mems:
            return 5
        min_gib = min(mems) / 1024
        # H100 / H200 (HBM3, fastest per-position compute) — bigger batches
        if names and all(("H100" in n) or ("H200" in n) for n in names):
            return 10
        # A100-80GB / Blackwell — medium speed, bigger VRAM
        if min_gib >= 80:
            return 8
        # A40 / L40S / A6000 / A100-40GB (smaller VRAM or older arch)
        return 5
    except Exception:
        return 5


def _workqueue_build_positions(queue_root, well_positions_by_well):
    """Per-position queue: one tiny file per position for stride-scan claim.

    Layout::

        queue_root/
          queue/pos_00000              (json: {"well": "A/1", "position": "A/1/000000"})
          queue/pos_00001
          ...
          taken/                       (worker mid-claim)
          done/                        (successfully processed)
          INDEX.json                   (total count, for stride wraparound)

    Returns total number of position files created.
    """
    import json
    queue_root = Path(queue_root)
    queue_dir = queue_root / "queue"
    taken_dir = queue_root / "taken"
    done_dir  = queue_root / "done"
    queue_dir.mkdir(parents=True, exist_ok=True)
    taken_dir.mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    for well, positions in well_positions_by_well.items():
        for pos_name in positions:
            with open(queue_dir / f"pos_{idx:05d}", "w") as f:
                json.dump({"well": well, "position": pos_name}, f)
            idx += 1

    with open(queue_root / "INDEX.json", "w") as f:
        json.dump({"n_total": idx}, f)
    return idx


def _build_worksteal_jobs(
    experiment,
    process,
    positions_by_well,
    recon_slurm_params,
    *,
    queue_tag="",
    name_prefix="recon",
    n_jobs=None,
    metadata_phase="reconstruct_workstealing",
):
    """Build a per-position work-steal queue + matching SLURM jobs.

    Shared by the full run and both rerun paths (auto-audit and targeted
    ``submit --positions``) so all three submit the identical work-steal
    shape: N GPUs/job, WS cpu/mem, no GPU-class constraint, workers draining
    a shared NFS queue via ``reconstruct_workstealing_v2``.

    positions_by_well : {well: [position_name, ...]}
    queue_tag         : suffix in the queue dir name ("" or e.g. "rerun")
    name_prefix       : SLURM job-name stem ("recon" / "audit_rerun")
    n_jobs            : explicit job count; if None, sized to the queue
                        (~350 positions/job, capped per-process, floored 1).

    Returns (jobs, slurm_params, queue_root, n_total, n_jobs, gpus_per_job).
    ``slurm_params`` is a COPY of ``recon_slurm_params`` reshaped for v2.
    """
    from datetime import datetime as _dt

    dataset = OpsDataset(experiment)
    pcfg = _get_process_config(process)
    proc_name = process or "pheno"
    phase3d_store_path = _resolve_path(dataset.store_paths[pcfg["phase3d_store_key"]])
    recon_dir = phase3d_store_path.parent

    tag = f"_{queue_tag}" if queue_tag else ""
    queue_root = recon_dir / f"_workqueue_{proc_name}{tag}_{_dt.now().strftime('%Y%m%d_%H%M%S')}"
    n_total = _workqueue_build_positions(queue_root, dict(positions_by_well))

    gpus_per_job = int(os.environ.get("OPS_TILT_GPUS_PER_JOB", 4))
    # Same env as the full run's fixed default; here it caps the sized count.
    cap = int(os.environ.get(
        "OPS_TILT_WORK_STEAL_JOBS", 20 if proc_name == "pheno" else 8))
    if n_jobs is None:
        n_jobs = max(1, min(cap, (n_total + 349) // 350))

    # N GPUs per job, no constraint — workers auto-adapt batch_size to
    # whatever GPU SLURM gives us.
    slurm_params = dict(recon_slurm_params)
    slurm_params["gpus_per_node"] = gpus_per_job
    slurm_params["cpus_per_task"] = int(
        os.environ.get("OPS_TILT_WS_CPUS_PER_JOB", 8 * gpus_per_job))
    slurm_params["mem"] = os.environ.get(
        "OPS_TILT_WS_MEM_PER_JOB", f"{64 * gpus_per_job}G")
    slurm_params.pop("slurm_constraint", None)

    jobs = [
        {
            "name": f"{name_prefix}_{proc_name}_ws_{i:03d}",
            "func": reconstruct_workstealing_v2,
            "kwargs": {
                "experiment": experiment,
                "process": process,
                "work_queue_dir": str(queue_root),
                # array_task_id folds into the decorator's log_key so each
                # SLURM child lands under a unique key instead of clobbering.
                "array_task_id": i,
            },
            "metadata": {"phase": metadata_phase, "job_id": i},
        }
        for i in range(n_jobs)
    ]
    return jobs, slurm_params, queue_root, n_total, n_jobs, gpus_per_job


def _workqueue_batch_claim_stride(
    queue_dir, taken_dir, worker_tag, batch_size, anchor, n_total
):
    """Stride-scan atomic claim of up to ``batch_size`` positions.

    Returns list of (taken_path, info_dict) tuples for whatever the worker
    actually managed to claim (may be shorter than ``batch_size``). The
    second return value is ``next_anchor`` (resume point for next iteration).
    A third return ``saw_misses`` reports the consecutive-miss count —
    callers use it as a "queue exhausted?" hint.

    No listdir in the hot path. Each candidate is a single rename attempt;
    failures (already taken / vanished) are silent and we move on.
    """
    import json
    claimed = []
    i = anchor % max(1, n_total)
    misses = 0
    scanned = 0
    while len(claimed) < batch_size and scanned < n_total:
        candidate = queue_dir / f"pos_{i:05d}"
        target = taken_dir / f"{worker_tag}__pos_{i:05d}"
        try:
            os.rename(candidate, target)
            try:
                with open(target) as f:
                    info = json.load(f)
            except Exception:
                info = {}
            claimed.append((target, info))
            misses = 0
        except (FileNotFoundError, OSError):
            misses += 1
        i = (i + 1) % max(1, n_total)
        scanned += 1
        # Early-exit: if we've scanned a stretch and got nothing, the
        # queue is likely empty (or we're chasing other workers). Let
        # caller decide whether to retry from a new anchor or stop.
        if misses >= max(50, batch_size * 5) and len(claimed) == 0:
            break
    return claimed, i, misses


def _workqueue_release_batch(taken_paths, done_dir):
    """Release multiple claimed positions atomically (rename → done/)."""
    done_dir = Path(done_dir)
    for tp in taken_paths:
        try:
            os.rename(tp, done_dir / tp.name)
        except (FileNotFoundError, OSError):
            pass


def _reconstruct_chunk_on_gpu(positions_chunk, gpu_id, worker_kwargs):
    """Run _reconstruct_position_3d_2d sequentially on one GPU subprocess.

    Called by reconstruct() via ProcessPoolExecutor(mp_context='spawn').
    The subprocess starts CUDA-clean because spawn doesn't inherit parent
    state — we pin to the assigned GPU here BEFORE any CUDA-using import
    fires inside _reconstruct_position_3d_2d.

    positions_chunk : list[str]
        Position paths this subprocess is responsible for.
    gpu_id : int
        Which GPU index (0..N-1) within the job's CUDA_VISIBLE_DEVICES.
    worker_kwargs : dict
        Fixed arguments passed to every position call (paths, model, etc.).
    """
    import os
    # Pin to this GPU before any CUDA touch.
    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if parent_cvd:
        # SLURM gave us a list of physical GPU ids; pick the gpu_id'th one.
        cvd_list = [g for g in parent_cvd.split(",") if g.strip() != ""]
        if 0 <= gpu_id < len(cvd_list):
            os.environ["CUDA_VISIBLE_DEVICES"] = cvd_list[gpu_id]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import time as _time
    import traceback
    # Late imports (after CVD is set)
    import torch
    try:
        _dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        _dev_uuid = str(torch.cuda.get_device_properties(0).uuid) if torch.cuda.is_available() else "-"
    except Exception:
        _dev_name, _dev_uuid = "?", "?"
    print(
        f"[gpu-subproc gpu_id={gpu_id} cvd={os.environ['CUDA_VISIBLE_DEVICES']}] "
        f"torch_dev='{_dev_name}' uuid={_dev_uuid} positions={len(positions_chunk)}",
        flush=True,
    )

    raw_store_path = worker_kwargs["raw_store_path"]
    phase3d_store_path = worker_kwargs["phase3d_store_path"]
    out_3d_path = worker_kwargs["out_3d_path"]
    out_2d_path = worker_kwargs["out_2d_path"]
    pcfg = worker_kwargs["pcfg"]
    model = worker_kwargs["model"]
    valid_time_indices = worker_kwargs["valid_time_indices"]

    results = []
    tilt_rows = []
    for i, pos in enumerate(positions_chunk):
        for t_idx in valid_time_indices:
            t_label = f" T={t_idx}" if len(valid_time_indices) > 1 else ""
            t0 = _time.time()
            try:
                result = _reconstruct_position_3d_2d(
                    pos,
                    raw_store_path,
                    phase3d_store_path,
                    out_3d_path,
                    out_2d_path,
                    pcfg,
                    model,
                    time_index=t_idx,
                )
                elapsed = _time.time() - t0
                result["elapsed_s"] = elapsed
                tilt_rows.extend(
                    _collect_tilt_rows(result.pop("opt_params"), GRID_SIZE, pos))
                results.append(result)
                print(
                    f"  [gpu_id={gpu_id} {i+1}/{len(positions_chunk)}] "
                    f"{pos}{t_label}  done ({elapsed:.1f}s)",
                    flush=True,
                )
            except Exception as e:
                elapsed = _time.time() - t0
                print(
                    f"  [gpu_id={gpu_id} {i+1}/{len(positions_chunk)}] "
                    f"{pos}{t_label}  FAILED ({elapsed:.1f}s): {e}",
                    flush=True,
                )
                traceback.print_exc()
    return results, tilt_rows


# ============================================================
# Output-store pre-creation strategies
# ============================================================
#
# Two concurrency-safe ways to pre-create the reconstruct() output stores are
# provided so the approach can be chosen in review. Approach A (flock + sentinel)
# is the original from ``main``; it was dropped from beta/nextflow by a bad merge.
# Approach B (atomic O_CREAT|O_EXCL claim + poll) is the current beta/nextflow
# implementation. Both share the same signature and are interchangeable via the
# ``_precreate_store`` alias below.
#
# Review note: approach B's leading ``store_path.exists()`` fast-path (and the
# losers' ``while not store_path.exists()`` poll) is exactly the check approach A's
# commentary warns is unsafe — ``create_hcs_store_fast`` mkdirs the store dir FIRST
# and writes metadata LAST, so a second process can observe the dir mid-creation and
# start writing into a partially-built zarr. Approach A avoids this by always taking
# the lock and gating on a sentinel touched only AFTER creation completes. Weigh that
# against B's cross-node atomicity (flock is advisory and unreliable on NFS/VAST).


def _precreate_store_flock_sentinel(
    store_path,
    positions,
    shape,
    chunks,
    dtype,
    scale,
    channel_names,
    version,
    timeout=None,  # unused; present for interchangeability with the O_EXCL variant
):
    """Pre-create one output store — Approach A: flock + sentinel completion marker.

    Original implementation from ``main`` (removed from beta/nextflow by a bad
    merge), restored here for comparison.

    Subtle race fix: the naive version checked ``store_path.exists()`` outside the
    lock as a fast-path skip. But ``create_hcs_store_fast`` mkdirs the store FIRST
    and writes metadata last — so a second process arriving while the first was
    mid-creation would see the dir exists, skip the lock, and start writing data
    into a partially-built zarr. With chunk_size=20 + 4 GPUs x 2 workers/GPU (pheno
    auto-detect), 354 chunks x 8 subprocesses raced on the metadata write and the
    resulting 3D store landed with 0 populated chunks (every reconstruct() call's
    writes silently no-op'd because the parent group's metadata was incomplete).
    Track happened to escape because workers_per_gpu=1 for proc=track keeps
    concurrency low.

    Fix: always take the lock; inside, check a sentinel file that is touched AFTER
    create_hcs_store_fast returns. The flock serializes the create; the sentinel
    makes the skip safe.
    """
    import fcntl

    lock_path = Path(str(store_path) + ".create_lock")
    sentinel = Path(str(store_path) + ".create_complete")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if sentinel.exists():
            print(f"  Reusing existing {store_path.name}")
        else:
            print(f"  Creating {store_path.name} ({len(positions)} positions)...")
            create_hcs_store_fast(
                store_path=store_path,
                positions=positions,
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                scale=scale,
                channel_names=channel_names,
                version=version,
            )
            sentinel.touch()


def _precreate_store_oexcl_poll(
    store_path,
    positions,
    shape,
    chunks,
    dtype,
    scale,
    channel_names,
    version,
    timeout=300,
):
    """Pre-create one output store — Approach B: atomic O_CREAT|O_EXCL claim + poll.

    Current beta/nextflow implementation. ``O_CREAT | O_EXCL`` is POSIX-atomic even
    on NFS/VAST, whereas ``fcntl.flock`` is advisory and unreliable cross-node. The
    winner creates the store; losers poll until it appears (up to ``timeout`` s).
    """
    if store_path.exists():
        print(f"  Reusing existing {store_path.name}")
        return

    lock_path = Path(str(store_path) + ".create_lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        # This node won the race — create the store.
        print(f"  Creating {store_path.name} ({len(positions)} positions)...")
        create_hcs_store_fast(
            store_path=store_path,
            positions=positions,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            scale=scale,
            channel_names=channel_names,
            version=version,
        )
        lock_path.unlink(missing_ok=True)
    except FileExistsError:
        # Another node is creating the store — wait for it to finish.
        t_wait = time.time()
        while not store_path.exists():
            if time.time() - t_wait > timeout:
                raise TimeoutError(
                    f"Timed out waiting for {store_path} to be created "
                    f"after {timeout}s"
                )
            time.sleep(0.5)


# Active strategy. Flip to _precreate_store_flock_sentinel to select approach A.
_precreate_store = _precreate_store_oexcl_poll


# ============================================================
# Reconstruction: main entry point
# ============================================================


@versioned_function("v1.0")
def reconstruct(
    experiment,
    well,
    process=None,
    debug_n_positions=None,
    position_start=None,
    position_end=None,
    output_dir=None,
    ngff_version="0.4",
):
    """Apply calibrated tilt model to reconstruct all positions for a single well.

    Produces 3D and 2D output stores with fixed names.
    """
    pcfg = _get_process_config(process)
    dataset = OpsDataset(experiment)
    raw_store_path = _resolve_path(dataset.store_paths[pcfg["raw_store_key"]])
    phase3d_store_path = _resolve_path(dataset.store_paths[pcfg["phase3d_store_key"]])
    recon_dir = Path(output_dir) if output_dir else phase3d_store_path.parent

    out_3d_path = _resolve_path(dataset.store_paths[pcfg["output_3d_store_key"]])
    out_2d_path = _resolve_path(dataset.store_paths[pcfg["output_2d_store_key"]])
    if output_dir:
        out_3d_path = recon_dir / out_3d_path.name
        out_2d_path = recon_dir / out_2d_path.name

    well_tag = well.replace("/", "_")
    proc_name = process or "pheno"
    model_path = recon_dir / "tilt_calibration" / proc_name / well_tag / "model.yaml"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run 'calibrate' first."
        )
    with open(model_path) as f:
        model = yaml.safe_load(f)

    print(f"Tilt-corrected 3D+2D reconstruction for {experiment} [{process}]")
    print(f"  Raw BF store:  {raw_store_path}")
    print(f"  Phase3D store: {phase3d_store_path}")
    print(f"  Output 3D:     {out_3d_path}")
    print(f"  Output 2D:     {out_2d_path}")
    print(f"  Model:         {model_path}")
    zen = model["zenith"]
    if zen["type"] == "linear":
        print(f"    zenith = {zen['base']:.4f} + {zen['slope']:.4f} * r")
    else:
        print(f"    zenith = {zen['a']:.6f} * exp({zen['b']:.4f} * r) + {zen['c']:.4f}")
    z_model = model["z_offset"]
    if z_model.get("type") == "linear_plane":
        print(f"    z_offset = {z_model['row_coeff']:.4f}*row + {z_model['col_coeff']:.4f}*col + {z_model['intercept']:.4f}")
    else:
        print(f"    z_offset = {z_model['value']:.4f}")

    # Discover positions and valid time indices
    with open_ome_zarr(phase3d_store_path, mode="r", version=ngff_version) as store:
        all_store_positions = [a[0] for a in store.positions()]
        well_positions = sorted(p for p in all_store_positions if p.startswith(well))
        ref = store[well_positions[0]]
        T, C, Z, Y, X = ref["0"].shape
        position_scale = ref.scale

        # Detect which time indices have data (check center pixel of mid-z)
        valid_time_indices = []
        for t in range(T):
            val = np.array(ref["0"][t, 0, Z // 2, Y // 2, X // 2])
            if val != 0:
                valid_time_indices.append(t)
        if not valid_time_indices:
            valid_time_indices = list(range(T))

    print(f"  Time indices with data: {valid_time_indices}")

    positions = list(well_positions)

    if debug_n_positions is not None:
        positions = positions[:debug_n_positions]

    all_positions_in_well = list(positions)

    if position_start is not None or position_end is not None:
        s = position_start or 0
        e = position_end or len(positions)
        positions = positions[s:e]

    print(f"  Positions: {len(positions)} (of {len(all_positions_in_well)} total)")

    # Pre-create output stores (concurrency-safe). Two interchangeable strategies
    # are defined above (_precreate_store_flock_sentinel vs _precreate_store_oexcl_poll);
    # _precreate_store selects the active one so the choice can be made in review.
    for store_path, shape, channels in [
        (out_3d_path, (T, 1, Z, Y, X), ["Phase3D"]),
        (out_2d_path, (T, 2, 1, Y, X), ["Phase2D", "Focus3D"]),
    ]:
        _precreate_store(
            store_path,
            positions=all_store_positions,
            shape=shape,
            chunks=(1, 1, shape[2], Y, X),
            dtype=np.float32,
            scale=position_scale,
            channel_names=channels,
            version=ngff_version,
        )

    # Process positions × time indices
    # Multi-GPU parallel dispatch: if more than one GPU is allocated to this
    # job (SLURM_GPUS_ON_NODE > 1), split positions into N sub-pools and run
    # them in parallel subprocesses, each pinned to its own GPU. Spawn (not
    # fork) so subprocesses don't inherit any parent CUDA context.
    # Set OPS_TILT_PARALLEL_GPUS to override the auto-detected GPU count.
    n_gpus_env = int(os.environ.get("OPS_TILT_PARALLEL_GPUS", "0") or "0")
    n_gpus_alloc = int(os.environ.get("SLURM_GPUS_ON_NODE", "1") or "1")
    n_gpus = n_gpus_env if n_gpus_env > 0 else n_gpus_alloc

    # Workers per GPU. Pheno workers hold ~14–20 GB VRAM each, so the safe
    # count depends on the smallest visible GPU. Explicit env override always
    # wins so callers can pin behavior on known hardware.
    # Override hierarchy:
    #   OPS_TILT_WORKERS_PER_GPU_{pheno,track}  — most specific
    #   OPS_TILT_WORKERS_PER_GPU                — fallback for both
    #   auto-detect via nvidia-smi (3 if ≥70 GB, 2 if ≥40 GB, 1 otherwise)
    _wpg_proc_key = f"OPS_TILT_WORKERS_PER_GPU_{proc_name}"
    _wpg_env = (
        os.environ.get(_wpg_proc_key)
        or os.environ.get("OPS_TILT_WORKERS_PER_GPU")
    )
    if _wpg_env:
        workers_per_gpu = int(_wpg_env)
    else:
        workers_per_gpu = _detect_workers_per_gpu_from_vram(proc_name)

    t_total = time.time()
    results = []
    all_tilt_rows = []
    if n_gpus > 1 and len(positions) > 1:
        # Parallel multi-GPU dispatch.
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as _mp
        ctx = _mp.get_context("spawn")

        n_workers = n_gpus * max(1, workers_per_gpu)
        n_workers = min(n_workers, len(positions))

        # Distribute positions across workers (round-robin so chunk sizes
        # differ by at most 1).
        chunks = [positions[i::n_workers] for i in range(n_workers)]
        chunks = [c for c in chunks if c]
        # Round-robin GPU assignment so each GPU gets workers_per_gpu workers.
        # worker 0..workers_per_gpu-1 → GPU 0; next group → GPU 1; etc.
        gpu_assignment = [w // max(1, workers_per_gpu) for w in range(len(chunks))]
        print(
            f"  Parallel dispatch ({proc_name}): {n_gpus} GPUs × "
            f"{workers_per_gpu} worker/GPU = {len(chunks)} subprocesses for "
            f"{len(positions)} positions × {len(valid_time_indices)} timepoints "
            f"(chunks: {[len(c) for c in chunks]}, gpus: {gpu_assignment})"
        )

        # Keep paths as Path objects — pathlib.Path is picklable, and
        # downstream writers (cyclops_utils.io.zarr_utils._write_plane_to_store)
        # use the `/` operator on these, which only works on Path. Earlier
        # versions stringified these and silently broke every 2D write.
        worker_kwargs = dict(
            raw_store_path=raw_store_path,
            phase3d_store_path=phase3d_store_path,
            out_3d_path=out_3d_path,
            out_2d_path=out_2d_path,
            pcfg=pcfg,
            model=model,
            valid_time_indices=valid_time_indices,
        )
        with ProcessPoolExecutor(max_workers=len(chunks), mp_context=ctx) as pool:
            futures = []
            for worker_idx, chunk in enumerate(chunks):
                gpu_id = gpu_assignment[worker_idx]
                futures.append(
                    pool.submit(_reconstruct_chunk_on_gpu, chunk, gpu_id, worker_kwargs)
                )
            for fut in futures:
                chunk_results, chunk_rows = fut.result()
                results.extend(chunk_results)
                all_tilt_rows.extend(chunk_rows)
        # Sequential path is skipped below.
        positions_for_sequential = []
    else:
        positions_for_sequential = positions

    for i, pos in enumerate(positions_for_sequential):
        for t_idx in valid_time_indices:
            t_label = f" T={t_idx}" if len(valid_time_indices) > 1 else ""
            print(f"\n  [{i + 1}/{len(positions_for_sequential)}] {pos}{t_label}")
            t0 = time.time()
            try:
                result = _reconstruct_position_3d_2d(
                    pos,
                    raw_store_path,
                    phase3d_store_path,
                    out_3d_path,
                    out_2d_path,
                    pcfg,
                    model,
                    time_index=t_idx,
                )
                elapsed = time.time() - t0
                result["elapsed_s"] = elapsed
                # Accumulate tilt params, then drop from result to keep it light
                all_tilt_rows.extend(
                    _collect_tilt_rows(result.pop("opt_params"), GRID_SIZE, pos))
                results.append(result)
                print(f"    Done ({elapsed:.1f}s)")
            except Exception as e:
                elapsed = time.time() - t0
                print(f"    FAILED ({elapsed:.1f}s): {e}")
                import traceback
                traceback.print_exc()

    total_elapsed = time.time() - t_total
    print(f"\n  Total: {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")
    print(f"  Successful: {len(results)}/{len(positions)}")

    # Save per-chunk tilt params CSV (into csvs/ subdirectory)
    if all_tilt_rows:
        tilt_dir = recon_dir / "tilt_calibration" / proc_name / well_tag
        s_tag = ""
        if position_start is not None or position_end is not None:
            s_tag = f"_{position_start or 0}_{position_end or 'end'}"
        _save_tilt_csv(all_tilt_rows,
                       tilt_dir / "csvs" / f"tilt_params{s_tag}.csv")

    return results


def reconstruct_workstealing(
    experiment,
    process=None,
    work_queue_dir=None,
    output_dir=None,
):
    """Work-stealing reconstruction loop. One SLURM job claims buckets
    from a shared filesystem queue and processes them until exhausted.

    Used by the orchestrator's ``OPS_TILT_WORK_STEAL=1`` path (default on).
    See ``_workqueue_build`` for the queue layout. All N concurrent SLURM
    jobs run this same function — load balance is dynamic; faster GPUs
    naturally claim more buckets.

    Atomicity is via ``os.rename`` (POSIX-atomic on NFS for same-FS
    renames). No flock / file content locks needed.
    """
    import json

    pcfg = _get_process_config(process)
    dataset = OpsDataset(experiment)
    raw_store_path = _resolve_path(dataset.store_paths[pcfg["raw_store_key"]])
    phase3d_store_path = _resolve_path(dataset.store_paths[pcfg["phase3d_store_key"]])
    recon_dir = Path(output_dir) if output_dir else phase3d_store_path.parent

    out_3d_path = _resolve_path(dataset.store_paths[pcfg["output_3d_store_key"]])
    out_2d_path = _resolve_path(dataset.store_paths[pcfg["output_2d_store_key"]])
    if output_dir:
        out_3d_path = recon_dir / out_3d_path.name
        out_2d_path = recon_dir / out_2d_path.name

    proc_name = process or "pheno"
    queue_root = Path(work_queue_dir)
    queue_dir = queue_root / "queue"
    taken_dir = queue_root / "taken"
    done_dir  = queue_root / "done"
    chunk_tag = _workqueue_chunk_tag()

    # Discover dims + ALL store positions + per-well model paths
    with open_ome_zarr(phase3d_store_path, mode="r") as store:
        all_store_positions = [a[0] for a in store.positions()]
        ref = store[all_store_positions[0]]
        T, C, Z, Y, X = ref["0"].shape
        position_scale = ref.scale

        valid_time_indices = []
        for t in range(T):
            val = np.array(ref["0"][t, 0, Z // 2, Y // 2, X // 2])
            if val != 0:
                valid_time_indices.append(t)
        if not valid_time_indices:
            valid_time_indices = list(range(T))

    # Pre-load every well's tilt model so workers can model-switch
    # between buckets without re-reading from disk every time.
    models_by_well = {}
    discovered_wells = sorted({"/".join(p.split("/")[:2]) for p in all_store_positions})
    for w in discovered_wells:
        well_tag = w.replace("/", "_")
        model_path = recon_dir / "tilt_calibration" / proc_name / well_tag / "model.yaml"
        if model_path.exists():
            with open(model_path) as f:
                models_by_well[w] = yaml.safe_load(f)

    print(f"Tilt-corrected work-stealing recon for {experiment} [{proc_name}]")
    print(f"  Queue:           {queue_root}")
    print(f"  Chunk tag:       {chunk_tag}")
    print(f"  Wells w/ model:  {sorted(models_by_well.keys())}")
    print(f"  Time indices:    {valid_time_indices}")

    # Pre-create output stores (sentinel-locked — see commentary in
    # reconstruct() at the same location).
    from cyclops_utils.io.zarr_precreate import create_hcs_store_fast
    import fcntl
    for store_path, shape, channels in [
        (out_3d_path, (T, 1, Z, Y, X), ["Phase3D"]),
        (out_2d_path, (T, 2, 1, Y, X), ["Phase2D", "Focus3D"]),
    ]:
        lock_path = Path(str(store_path) + ".create_lock")
        sentinel = Path(str(store_path) + ".create_complete")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "w") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            if not sentinel.exists():
                print(f"  Creating {store_path.name} ({len(all_store_positions)} positions)...")
                create_hcs_store_fast(
                    store_path=store_path,
                    positions=all_store_positions,
                    shape=shape,
                    chunks=(1, 1, shape[2], Y, X),
                    dtype=np.float32,
                    scale=position_scale,
                    channel_names=channels,
                )
                sentinel.touch()

    # Multi-GPU parallel dispatch parameters (same as reconstruct())
    n_gpus_env = int(os.environ.get("OPS_TILT_PARALLEL_GPUS", "0") or "0")
    n_gpus_alloc = int(os.environ.get("SLURM_GPUS_ON_NODE", "1") or "1")
    n_gpus = n_gpus_env if n_gpus_env > 0 else n_gpus_alloc
    _wpg_proc_key = f"OPS_TILT_WORKERS_PER_GPU_{proc_name}"
    _wpg_env = os.environ.get(_wpg_proc_key) or os.environ.get("OPS_TILT_WORKERS_PER_GPU")
    workers_per_gpu = int(_wpg_env) if _wpg_env else _detect_workers_per_gpu_from_vram(proc_name)
    n_workers = max(1, n_gpus * workers_per_gpu)

    # Long-lived ProcessPoolExecutor: spawn workers ONCE, reuse across
    # bucket claims. Each bucket distributes its positions across the pool.
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as _mp
    ctx = _mp.get_context("spawn")

    all_results = []
    all_tilt_rows_by_well = {}
    buckets_done = 0
    t_chunk_start = time.time()

    print(f"  Pool: {n_gpus} GPUs × {workers_per_gpu} worker/GPU = {n_workers} subprocesses")
    print(f"  Starting work-steal loop...")

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        while True:
            claimed = _workqueue_claim(queue_dir, taken_dir, chunk_tag)
            if claimed is None:
                break  # queue exhausted
            try:
                with open(claimed / "info.json") as f:
                    info = json.load(f)
                bucket_well = info["well"]
                bucket_positions = info["positions"]
                if bucket_well not in models_by_well:
                    print(f"  [{chunk_tag}] WARNING: no model for {bucket_well}; skipping bucket {claimed.name}",
                          flush=True)
                    _workqueue_release(claimed, done_dir)
                    continue

                # Distribute positions round-robin across the pool
                sub_chunks = [bucket_positions[i::n_workers] for i in range(n_workers)]
                sub_chunks = [c for c in sub_chunks if c]
                gpu_assignment = [w // max(1, workers_per_gpu) for w in range(len(sub_chunks))]

                worker_kwargs = dict(
                    raw_store_path=raw_store_path,
                    phase3d_store_path=phase3d_store_path,
                    out_3d_path=out_3d_path,
                    out_2d_path=out_2d_path,
                    pcfg=pcfg,
                    model=models_by_well[bucket_well],
                    valid_time_indices=valid_time_indices,
                )

                t_bucket = time.time()
                futures = [
                    pool.submit(_reconstruct_chunk_on_gpu,
                                sub_chunks[i], gpu_assignment[i], worker_kwargs)
                    for i in range(len(sub_chunks))
                ]
                for fut in futures:
                    sub_results, sub_rows = fut.result()
                    all_results.extend(sub_results)
                    all_tilt_rows_by_well.setdefault(bucket_well, []).extend(sub_rows)

                bucket_elapsed = time.time() - t_bucket
                buckets_done += 1
                print(
                    f"  [{chunk_tag}] bucket {buckets_done} ({claimed.name}, "
                    f"well={bucket_well}, {len(bucket_positions)} pos) "
                    f"done in {bucket_elapsed:.1f}s",
                    flush=True,
                )
                _workqueue_release(claimed, done_dir)
            except Exception as e:
                import traceback
                print(f"  [{chunk_tag}] bucket {claimed.name} FAILED: {e}", flush=True)
                traceback.print_exc()
                # Leave the bucket in taken/ for diagnostic; orchestrator
                # audit will catch unprocessed positions at the end.

    total_elapsed = time.time() - t_chunk_start
    print(f"\n[{chunk_tag}] Finished work-steal: {buckets_done} buckets in {total_elapsed:.1f}s")

    # Save per-chunk tilt params CSVs (one per well this chunk touched)
    for well, rows in all_tilt_rows_by_well.items():
        if not rows:
            continue
        well_tag = well.replace("/", "_")
        tilt_dir = recon_dir / "tilt_calibration" / proc_name / well_tag
        _save_tilt_csv(
            rows,
            tilt_dir / "csvs" / f"tilt_params_workstealing_{chunk_tag}.csv",
        )

    return all_results


def _workstealing_worker_loop(args_dict):
    """Single worker subprocess: claim positions, process, release. Loops
    until the queue is empty (consecutive-miss threshold reached).

    Workers are independent — they don't coordinate with sibling workers
    on the same GPU or across the SLURM job. Each pulls its own batch
    from the NFS queue, runs positions one at a time, releases the batch,
    loops back. Faster GPUs naturally claim bigger batches (per
    ``_workqueue_batch_size_for_gpu``) and end up processing more
    positions in the total run.

    If ``gpu_id`` is in args_dict, the worker pins to that GPU by
    rewriting CUDA_VISIBLE_DEVICES BEFORE any CUDA-using import.
    """
    import json
    import time
    import traceback
    import hashlib

    experiment = args_dict["experiment"]
    process = args_dict["process"]
    work_queue_dir = args_dict["work_queue_dir"]
    worker_idx = args_dict["worker_idx"]
    gpu_id = args_dict.get("gpu_id")

    # Pin to the assigned GPU before importing torch (spawn ctx keeps env
    # changes private to this subprocess).
    if gpu_id is not None:
        parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        cvd_list = [g for g in parent_cvd.split(",") if g.strip() != ""]
        if 0 <= gpu_id < len(cvd_list):
            os.environ["CUDA_VISIBLE_DEVICES"] = cvd_list[gpu_id]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    pcfg = _get_process_config(process)
    dataset = OpsDataset(experiment)
    raw_store_path = _resolve_path(dataset.store_paths[pcfg["raw_store_key"]])
    phase3d_store_path = _resolve_path(dataset.store_paths[pcfg["phase3d_store_key"]])
    out_3d_path = _resolve_path(dataset.store_paths[pcfg["output_3d_store_key"]])
    out_2d_path = _resolve_path(dataset.store_paths[pcfg["output_2d_store_key"]])
    recon_dir = phase3d_store_path.parent
    proc_name = process or "pheno"

    queue_root = Path(work_queue_dir)
    queue_dir = queue_root / "queue"
    taken_dir = queue_root / "taken"
    done_dir = queue_root / "done"

    with open(queue_root / "INDEX.json") as f:
        n_total = json.load(f)["n_total"]

    base_tag = _workqueue_chunk_tag()
    worker_tag = f"{base_tag}_w{worker_idx}"

    # Spread anchors so workers don't all start scanning from index 0
    anchor = int(hashlib.md5(worker_tag.encode()).hexdigest()[:8], 16) % max(1, n_total)
    initial_batch_size = _workqueue_batch_size_for_gpu(proc_name)
    batch_size = initial_batch_size  # shrinks adaptively as queue thins (see loop)

    # Discover valid time indices once
    with open_ome_zarr(phase3d_store_path, mode="r") as store:
        first_pos = next(iter(store.positions()))[0]
        ref = store[first_pos]
        T = ref["0"].shape[0]
        Z = ref["0"].shape[2]
        Yc = ref["0"].shape[3] // 2
        Xc = ref["0"].shape[4] // 2
        valid_time_indices = []
        for t in range(T):
            val = np.array(ref["0"][t, 0, Z // 2, Yc, Xc])
            if val != 0:
                valid_time_indices.append(t)
        if not valid_time_indices:
            valid_time_indices = list(range(T))

    models_by_well = {}  # lazy per-well cache

    print(
        f"[{worker_tag}] worker started "
        f"batch_size={batch_size} anchor={anchor} n_total={n_total} "
        f"valid_T={valid_time_indices}",
        flush=True,
    )

    t_start = time.time()
    positions_done = 0
    consecutive_empty_batches = 0

    while True:
        claimed, anchor, misses = _workqueue_batch_claim_stride(
            queue_dir, taken_dir, worker_tag, batch_size, anchor, n_total
        )

        if not claimed:
            consecutive_empty_batches += 1
            if consecutive_empty_batches >= 3:
                # Confirm queue is empty before exiting
                try:
                    if not any(queue_dir.iterdir()):
                        break
                except Exception:
                    break
                consecutive_empty_batches = 0
            time.sleep(2)
            continue
        consecutive_empty_batches = 0

        # Adaptive batch shrink: when the queue is thinning, claims start
        # returning partial batches or take many misses to fill. Halve the
        # batch size so the tail processes smaller bites — keeps fast
        # workers from grabbing the last 10 positions while slow workers
        # idle. Minimum 1 position per claim.
        if len(claimed) < batch_size or misses >= batch_size:
            new_bs = max(1, batch_size // 2)
            if new_bs != batch_size:
                print(
                    f"[{worker_tag}] tail-mode: batch_size {batch_size} → {new_bs} "
                    f"(claimed {len(claimed)}, misses {misses})",
                    flush=True,
                )
                batch_size = new_bs

        # Process claimed positions one at a time (sequential within worker)
        batch_rows_by_well = {}
        for taken_path, info in claimed:
            well = info.get("well")
            pos = info.get("position")
            if not well or not pos:
                continue
            if well not in models_by_well:
                well_tag = well.replace("/", "_")
                model_path = recon_dir / "tilt_calibration" / proc_name / well_tag / "model.yaml"
                try:
                    with open(model_path) as f:
                        models_by_well[well] = yaml.safe_load(f)
                except Exception as e:
                    print(f"[{worker_tag}] missing model {model_path}: {e}", flush=True)
                    continue

            for t_idx in valid_time_indices:
                try:
                    result = _reconstruct_position_3d_2d(
                        pos, raw_store_path, phase3d_store_path,
                        out_3d_path, out_2d_path, pcfg, models_by_well[well],
                        time_index=t_idx,
                    )
                except Exception as e:
                    print(f"[{worker_tag}] FAILED {pos} T={t_idx}: {e}", flush=True)
                    traceback.print_exc()
                    continue
                # Separate guard: the recon already landed in the zarr, so a
                # tilt-row failure must not report the position as FAILED.
                try:
                    batch_rows_by_well.setdefault(well, []).extend(
                        _collect_tilt_rows(result.pop("opt_params"), GRID_SIZE, pos))
                except Exception as e:
                    print(f"[{worker_tag}] tilt rows skipped {pos} T={t_idx}: {e}",
                          flush=True)
            positions_done += 1

        _workqueue_release_batch([t for t, _ in claimed], done_dir)

        # Flush per batch (~4 ms/batch vs minutes of GPU work): workers run
        # until the queue drains, so an end-of-loop write would lose
        # everything on preemption. Never let a diagnostics write kill a
        # compute worker.
        for well, rows in batch_rows_by_well.items():
            if not rows:
                continue
            try:
                _save_tilt_csv(
                    rows,
                    recon_dir / "tilt_calibration" / proc_name / well.replace("/", "_")
                    / "csvs" / f"tilt_params_ws_{worker_tag}.csv",
                    append=True,
                )
            except Exception as e:
                print(f"[{worker_tag}] tilt CSV write failed ({well}): {e}", flush=True)

    elapsed = time.time() - t_start
    rate = positions_done / elapsed * 60 if elapsed > 0 else 0
    print(
        f"[{worker_tag}] worker exiting: {positions_done} positions in "
        f"{elapsed:.0f}s ({rate:.1f} pos/min)",
        flush=True,
    )


@versioned_function("v1.0")
def reconstruct_workstealing_v2(experiment, process=None, work_queue_dir=None, output_dir=None, array_task_id=None):
    """SLURM-job entrypoint for per-position work-stealing.

    Each SLURM job runs ``workers_per_gpu`` independent worker subprocesses
    sharing the job's single GPU. Workers pull positions from a shared
    NFS queue via atomic ``os.rename``; no parent coordination, no
    chunk-level synchronization. Workers exit when the queue is empty.

    Compared to the legacy chunk-based dispatch, this design eliminates
    the per-bucket ``pool.submit`` overhead (which was the dominant cost
    in the previous run — workers idle ~75% of wall waiting on the
    chunk parent's coordination loop).
    """
    import multiprocessing as _mp

    pcfg = _get_process_config(process)
    dataset = OpsDataset(experiment)
    phase3d_store_path = _resolve_path(dataset.store_paths[pcfg["phase3d_store_key"]])
    out_3d_path = _resolve_path(dataset.store_paths[pcfg["output_3d_store_key"]])
    out_2d_path = _resolve_path(dataset.store_paths[pcfg["output_2d_store_key"]])
    proc_name = process or "pheno"

    # Discover dims so we can pre-create the output stores on this node
    # (sentinel-locked; safe with many parents racing).
    with open_ome_zarr(phase3d_store_path, mode="r") as store:
        all_store_positions = [a[0] for a in store.positions()]
        ref = store[all_store_positions[0]]
        T, C, Z, Y, X = ref["0"].shape
        position_scale = ref.scale

    from cyclops_utils.io.zarr_precreate import create_hcs_store_fast
    # NFS-safe cross-worker serialization. ``fcntl.flock`` silently no-ops on
    # NFS mounts without rpc.lockd — observed on ops0171 when 7 of 8 track
    # work-stealing workers concurrently entered create_hcs_store_fast and
    # raced inside zarr's ``open_group(mode="w") → delete_dir → shutil.rmtree``,
    # tripping FileNotFoundError as workers unlinked each other's files. Use
    # ``O_CREAT|O_EXCL`` for atomic lock claim, same primitive as the FCL
    # writer in cyclops_utils/profiling/decorators (commit 2bac0cb).
    from cyclops_utils.profiling.decorators import _acquire_nfs_lock, _release_nfs_lock
    for store_path, shape, channels in [
        (out_3d_path, (T, 1, Z, Y, X), ["Phase3D"]),
        (out_2d_path, (T, 2, 1, Y, X), ["Phase2D", "Focus3D"]),
    ]:
        sentinel = Path(str(store_path) + ".create_complete")
        # Fast path: if the sentinel already exists, some other worker
        # finished the create — no lock needed.
        if sentinel.exists():
            continue
        lock_path = str(store_path) + ".create_lock"
        Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
        # Long timeout: create_hcs_store_fast writes 2000+ position zarr groups
        # on NFS and takes 2-4 min at scale. Default 120 s force-unlinks the
        # lock while the winner is still legitimately creating, causing waiters
        # to race with it and trip the same rmtree race we're preventing.
        # Observed on ops0171 pheno: 17/20 workers force-unlinked at 120 s and
        # died. 1800 s covers even slow NFS create; if that's ever exceeded,
        # the actual work has other problems worth investigating.
        _acquire_nfs_lock(lock_path, timeout=1800)
        try:
            if not sentinel.exists():
                print(f"  Creating {store_path.name}...", flush=True)
                create_hcs_store_fast(
                    store_path=store_path,
                    positions=all_store_positions,
                    shape=shape,
                    chunks=(1, 1, shape[2], Y, X),
                    dtype=np.float32,
                    scale=position_scale,
                    channel_names=channels,
                )
                sentinel.touch()
        finally:
            _release_nfs_lock(lock_path)

    # Detect workers_per_gpu from this GPU's VRAM (env override allowed).
    _wpg_proc_key = f"OPS_TILT_WORKERS_PER_GPU_{proc_name}"
    _wpg_env = os.environ.get(_wpg_proc_key) or os.environ.get("OPS_TILT_WORKERS_PER_GPU")
    workers_per_gpu = int(_wpg_env) if _wpg_env else _detect_workers_per_gpu_from_vram(proc_name)

    # n_gpus per SLURM job. Discover from SLURM_GPUS_ON_NODE (set by
    # ``--gres=gpu:N``), or override via OPS_TILT_PARALLEL_GPUS.
    n_gpus_env = int(os.environ.get("OPS_TILT_PARALLEL_GPUS", "0") or "0")
    n_gpus_alloc = int(os.environ.get("SLURM_GPUS_ON_NODE", "1") or "1")
    n_gpus = n_gpus_env if n_gpus_env > 0 else n_gpus_alloc

    n_workers = n_gpus * workers_per_gpu

    print(
        f"[parent] {experiment} {proc_name}: queue={work_queue_dir}, "
        f"spawning {n_workers} workers ({n_gpus} GPUs × {workers_per_gpu} per GPU)",
        flush=True,
    )

    ctx = _mp.get_context("spawn")
    # Import the worker via its real module path so spawn can pickle it even
    # when this file was launched as `python -m ...` (__module__="__main__").
    from cyclops_process.processes.reconstruct_tilt_corrected import (
        _workstealing_worker_loop as _worker_target,
    )
    procs = []
    for w in range(n_workers):
        gpu_id = w // max(1, workers_per_gpu)  # round-robin pin: 0,0,0,1,1,1,...
        p = ctx.Process(
            target=_worker_target,
            args=({
                "experiment": experiment,
                "process": process,
                "work_queue_dir": str(work_queue_dir),
                "worker_idx": w,
                "gpu_id": gpu_id,
            },),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print(f"[parent] all {n_workers} workers exited", flush=True)
    return {"status": "ok"}


# ============================================================
# SLURM submission
# ============================================================


# ============================================================
# Pipeline-compatible entry points
# ============================================================


def _normalize_wells(wells):
    """Strip FOV index from well paths: 'A/1/0' -> 'A/1'."""
    seen = []
    for w in wells:
        parts = w.split("/")
        normalized = "/".join(parts[:2])
        if normalized not in seen:
            seen.append(normalized)
    return seen


def calibrate_tilt(experiment, process=None, wells=None):
    """Calibrate all wells in parallel. Called by PipelineRunner."""
    if wells is None:
        from cyclops_utils.data.filesystem import get_experiment_wells
        wells = get_experiment_wells(experiment, prefix_only=True)
    wells = _normalize_wells(wells)
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    # 'spawn' instead of the Linux default 'fork': iohub imports CuPy at module
    # level, which leaves the CUDA runtime in an error state on CPU-only nodes.
    # Forked workers inherit that broken state; when they later import torch,
    # torch.cuda.is_available() hangs or segfaults.  Spawned workers start a
    # fresh interpreter with no inherited CUDA state, so torch initialises
    # cleanly.  Worker startup is slightly slower (~5-10 s) but negligible
    # relative to the per-tile optimisation time.
    with ProcessPoolExecutor(
        max_workers=len(wells), mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        futures = [
            executor.submit(calibrate, experiment, well=w, process=process, resume=True)
            for w in wells
        ]
        for f in futures:
            f.result()  # propagate exceptions


def reconstruct_tilt_corrected(experiment, process=None, wells=None, skip_precheck=False):
    """Reconstruct all wells via internal SLURM scheduler. Called by PipelineRunner."""
    if wells is None:
        wells = get_experiment_wells(experiment, prefix_only=True)
    wells = _normalize_wells(wells)
    pcfg = _get_process_config(process)
    proc_name = process or "pheno"

    submit_jobs(
        experiment,
        wells=wells,
        process=process,
        chunk_size=150,
        wait_for_completion=True,
        skip_precheck=skip_precheck,
        _skip_calibration=True,  # PipelineRunner already ran calibration step
    )


def reconstruct_tilt_corrected_setup(experiment, process=None, skip_precheck=False):
    """Nextflow fan-out setup: discover per-chunk work units and print one line per chunk.

    Each output line has the format: ``well start end``
    (0-based position indices, end exclusive). Warnings go to stderr to avoid
    corrupting the Nextflow stdout channel.
    """
    chunk_size = 150
    wells = get_experiment_wells(experiment, prefix_only=True)
    wells = _normalize_wells(wells)
    pcfg = _get_process_config(process)
    proc_name = process or "pheno"

    dataset = OpsDataset(experiment)
    phase3d_store_path = _resolve_path(dataset.store_paths[pcfg["phase3d_store_key"]])

    with open_ome_zarr(phase3d_store_path, mode="r") as store:
        all_positions = [a[0] for a in store.positions()]

    if skip_precheck:
        print("Skipping input data precheck (--skip_precheck).", file=sys.stderr)
    else:
        SAMPLE_SIZE = 50
        WARN_EMPTY_RATIO = 0.01
        MAX_EMPTY_RATIO = 0.10
        ops_num = int(experiment.split("_")[0].replace("ops", ""))

        for well in wells:
            well_positions = sorted(p for p in all_positions if p.startswith(well))
            sample = random.sample(
                well_positions, min(SAMPLE_SIZE, len(well_positions))
            )
            skip_tp = (
                [0]
                if (ops_num >= 69 and well == "A/2" and proc_name == "track")
                else None
            )
            _, missing, empty = _validate_all_positions_have_data(
                phase3d_store_path, sample, skip_timepoints=skip_tp
            )
            n_bad = len(missing) + len(empty)
            ratio = n_bad / len(sample) if sample else 0
            if ratio > MAX_EMPTY_RATIO:
                raise RuntimeError(
                    f"Well {well} has too many empty/missing positions "
                    f"({n_bad}/{len(sample)} sampled = {ratio:.0%}, threshold {MAX_EMPTY_RATIO:.0%}). "
                    f"Fix upstream 3D reconstruction before running tilt-corrected reconstruction."
                )
            elif ratio > WARN_EMPTY_RATIO:
                print(
                    f"WARNING: Well {well} has {n_bad}/{len(sample)} sampled "
                    f"empty/missing positions ({ratio:.0%}).",
                    file=sys.stderr,
                )
        print(
            f"Input 3D positions spot-check passed (sampled {SAMPLE_SIZE}, "
            f"<={int(MAX_EMPTY_RATIO * 100)}% empty).",
            file=sys.stderr,
        )

    for well in wells:
        well_positions = [p for p in all_positions if p.startswith(well)]
        n_positions = len(well_positions)
        n_chunks = (n_positions + chunk_size - 1) // chunk_size
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min((chunk_idx + 1) * chunk_size, n_positions)
            print(f"{well} {start} {end}")


def get_n_positions(experiment, process, well):
    """Print the number of positions for a given well in the phase3d store."""
    pcfg = _get_process_config(process)
    dataset = OpsDataset(experiment)
    phase3d_store_path = _resolve_path(dataset.store_paths[pcfg["phase3d_store_key"]])
    with open_ome_zarr(phase3d_store_path, mode="r") as store:
        n = sum(1 for pos, _ in store.positions() if pos.startswith(well))
    print(n)


def reconstruct_tilt_corrected_job(
    experiment,
    well,
    process=None,
    position_start: Optional[int] = None,
    position_end: Optional[int] = None,
    ngff_version="0.4",
):
    """Nextflow per-chunk entry point: reconstruct one position range for a single well."""
    reconstruct(
        experiment=experiment,
        well=well,
        process=process,
        position_start=position_start,
        position_end=position_end,
        ngff_version=ngff_version,
    )


def submit_jobs(
    experiment,
    wells=None,
    process=None,
    chunk_size=None,
    dry_run=False,
    wait_for_completion=True,
    _skip_calibration=False,
    position_start=None,
    position_end=None,
    targeted_positions=None,
    skip_precheck=False,
):
    """Submit calibration + reconstruction jobs to SLURM.

    Submits calibration as one job per well, then reconstruction as
    chunked jobs (chunk_size positions per job) per well.

    chunk_size default is per-process: ``352`` for pheno (sized so a 7035-FOV
    experiment fits in one wave of 20 concurrent slots), ``20`` for track
    (~296 positions per well already needs only ~15 jobs). Override via the
    ``OPS_TILT_CHUNK_SIZE`` env var or the explicit kwarg.

    To resubmit specific failed tiles (skips calibration automatically):
        python -m cyclops_process.processes.reconstruct_tilt_corrected submit --experiment ops0066_20250820 --process track --positions A/1:31 A/1:61 A/1:77 A/2:132 A/3:69

    Parameters
    ----------
    experiment : str
        Experiment name.
    wells : list of str
        Wells to process (default: ["A/1", "A/2", "A/3"]).
    process : str
        "track" or "pheno".
    chunk_size : int
        Positions per reconstruction job.
    dry_run : bool
        Print plan without submitting.
    wait_for_completion : bool
        Block until all jobs complete.
    _skip_calibration : bool
        Skip calibration phase (used by PipelineRunner which runs
        calibration as a separate step).
    position_start : int, optional
        If set, submit only positions from this index. Skips calibration.
    position_end : int, optional
        If set, submit only positions up to this index (exclusive).
    """
    from cyclops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

    if wells is None:
        wells = get_experiment_wells(experiment, prefix_only=True)

    pcfg = _get_process_config(process)
    proc_name = process or "pheno"

    # Default chunk_size is per-process: pheno=352 (sized so a ~7035-FOV
    # experiment fits in one wave of 20 concurrent slots — see the chunk-
    # size sweep in scratch/nuclear_seg/tilt_sweep), track=20 (~296
    # positions per well already needs only ~15 jobs). The OPS_TILT_CHUNK_SIZE
    # env var overrides both; an explicit kwarg overrides everything.
    if chunk_size is None:
        chunk_size = 352 if proc_name == "pheno" else 20
    _env_chunk = os.environ.get("OPS_TILT_CHUNK_SIZE")
    if _env_chunk is not None:
        try:
            chunk_size = int(_env_chunk)
            print(f"  chunk_size from OPS_TILT_CHUNK_SIZE: {chunk_size}")
        except ValueError:
            pass

    # Determine timeout based on number of timepoints in the store.
    # Base: ~2.5 min per timepoint per position (chunk_size=1).
    dataset = OpsDataset(experiment)
    pcfg_temp = _get_process_config(process)
    phase3d_path = _resolve_path(dataset.store_paths[pcfg_temp["phase3d_store_key"]])
    try:
        with open_ome_zarr(phase3d_path, mode="r") as _store:
            # Use max T across all wells for timeout calculation
            n_timepoints = 0
            for _pos_name, _ in _store.positions():
                t = _store[_pos_name]["0"].shape[0]
                if t > n_timepoints:
                    n_timepoints = t
                    break  # All positions in a well share T, just need one per well
            # Check one position per well
            _seen_wells = set()
            for _pos_name, _ in _store.positions():
                _well_prefix = "/".join(_pos_name.split("/")[:2])
                if _well_prefix in _seen_wells:
                    continue
                _seen_wells.add(_well_prefix)
                t = _store[_pos_name]["0"].shape[0]
                n_timepoints = max(n_timepoints, t)
    except Exception:
        n_timepoints = 2  # fallback
    min_per_tp = 5.0 if proc_name == "pheno" else 2.5
    timeout_per_position = max(5, int(n_timepoints * min_per_tp))
    timeout_per_job = timeout_per_position * chunk_size
    print(f"  Timeout: {timeout_per_job} min/job ({timeout_per_position} min/position × {chunk_size} positions/job)")

    # Env-var overrides so the orchestrator can tune per-job size without
    # editing code. Useful when (a) the FOV count is high and the default
    # chunk_size=150 results in too many SLURM jobs, or (b) we want more
    # GPUs per job to push positions through faster.
    # Recon phase (per-batch position processing — benefits from extra GPUs):
    #   OPS_TILT_GPUS_PER_JOB (default 1) — GPUs per recon batch
    #   OPS_TILT_CPUS_PER_JOB (default 2) — CPUs per recon batch
    #   OPS_TILT_MEM_PER_JOB  (default 16G) — memory per recon batch
    #   OPS_TILT_SLURM_CONSTRAINT (default "[h100|h200|6000_blackwell|a100_80]")
    #     — restricts recon chunks to GPU classes with enough VRAM for the
    #     auto-detected workers_per_gpu setting. A100-40GB / A40 / A6000
    #     work for 1 worker but OOM at 2; the default constraint blocks
    #     them so chunks land on hardware that can sustain the 2-3 worker
    #     dispatch. Set to empty string to allow any GPU class.
    _gpus = int(os.environ.get("OPS_TILT_GPUS_PER_JOB", 1))
    _cpus = int(os.environ.get("OPS_TILT_CPUS_PER_JOB", 2))
    _mem  = os.environ.get("OPS_TILT_MEM_PER_JOB", "16G")
    _constraint = os.environ.get(
        "OPS_TILT_SLURM_CONSTRAINT",
        "[h100|h200|6000_blackwell|a100_80]",
    )
    recon_slurm_params = {
        "timeout_min": timeout_per_job,
        "mem": _mem,
        "cpus_per_task": _cpus,
        "gpus_per_node": _gpus,
        "slurm_partition": "gpu",
    }
    if _constraint:
        recon_slurm_params["slurm_constraint"] = _constraint
    # Calibration phase (one job per well, ~few-minutes each — does not
    # scale with extra GPUs, so use a smaller default to avoid wasting
    # GPUs reserved for the recon-batch shape). Same env-var pattern.
    _cal_gpus = int(os.environ.get("OPS_TILT_CAL_GPUS_PER_JOB", 1))
    _cal_cpus = int(os.environ.get("OPS_TILT_CAL_CPUS_PER_JOB", 8))
    _cal_mem  = os.environ.get("OPS_TILT_CAL_MEM_PER_JOB", "64G")
    cal_slurm_params = {
        # Calibration is per-well and fairly fast; keep its own
        # short timeout independent of recon's chunk_size.
        "timeout_min": max(30, timeout_per_position * 4),
        "mem": _cal_mem,
        "cpus_per_task": _cal_cpus,
        "gpus_per_node": _cal_gpus,
        "slurm_partition": "gpu",
    }
    if _gpus > 1 or _cpus > 2 or _mem != "16G":
        print(
            f"  Recon resource overrides: gpus={_gpus} cpus={_cpus} mem={_mem}"
        )
    if _cal_gpus != 1 or _cal_cpus != 8 or _cal_mem != "64G":
        print(
            f"  Cal   resource overrides: gpus={_cal_gpus} cpus={_cal_cpus} mem={_cal_mem}"
        )

    # --- Phase 1: Calibration jobs (one per well) ---
    if not _skip_calibration:
        cal_jobs = []
        for well in wells:
            cal_jobs.append({
                "name": f"cal_{proc_name}_{well.replace('/', '_')}",
                "func": calibrate,
                "kwargs": {
                    "experiment": experiment,
                    "well": well,
                    "process": process,
                    "resume": True,
                },
                "metadata": {"well": well, "phase": "calibrate"},
            })

        print(f"Submitting {len(cal_jobs)} calibration jobs ({proc_name})...")
        cal_result = submit_parallel_jobs(
            jobs_to_submit=cal_jobs,
            experiment=experiment,
            slurm_params=cal_slurm_params,
            log_dir=f"slurm_tilt_corrected_logs/{experiment}/{proc_name}/calibrate",
            manifest_prefix=f"tilt_cal_{proc_name}",
            dry_run=dry_run,
            wait_for_completion=True,  # Must complete before reconstruction
        )

        if dry_run:
            print("DRY RUN: skipping reconstruction submission")
            return cal_result

        if not cal_result.get("success"):
            print("Calibration failed, skipping reconstruction")
            return cal_result

    # --- Phase 2: Reconstruction jobs (chunked per well) ---
    recon_jobs = []

    # Work-stealing toggle — read once here so every branch (and the
    # post-submission subtask aggregation below) sees it, not just the
    # process-all-positions path. (default on, OPS_TILT_WORK_STEAL=0 disables)
    _work_steal = os.environ.get("OPS_TILT_WORK_STEAL", "1") not in ("0", "", "false", "False")

    if targeted_positions and _work_steal:
        # Targeted resubmission (list of (well, position_index) pairs, e.g.
        # the command `audit` prints) routed through the SAME work-steal
        # queue the full run uses, instead of one SLURM job per FOV.
        dataset = OpsDataset(experiment)
        phase3d_store_path = _resolve_path(
            dataset.store_paths[pcfg["phase3d_store_key"]])
        with open_ome_zarr(phase3d_store_path, mode="r") as store:
            all_positions = [a[0] for a in store.positions()]

        # Map (well, idx) → position name via the same sorted-well ordering
        # reconstruct() uses to interpret position_start/position_end.
        well_sorted = {}
        rerun_by_well = defaultdict(list)
        for well, pos_idx in targeted_positions:
            if well not in well_sorted:
                well_sorted[well] = sorted(
                    p for p in all_positions if p.startswith(well))
            rerun_by_well[well].append(well_sorted[well][pos_idx])

        jobs, recon_slurm_params, queue_root, n_total, n_jobs, gpus_per_job = (
            _build_worksteal_jobs(
                experiment, process, dict(rerun_by_well), recon_slurm_params,
                queue_tag="rerun", name_prefix="recon",
            )
        )
        recon_jobs.extend(jobs)
        print(f"  Targeted rerun via work-steal queue: {n_total} positions, "
              f"{n_jobs} jobs × {gpus_per_job} GPUs (queue: {queue_root})")
    elif targeted_positions:
        # Fallback (OPS_TILT_WORK_STEAL=0): one job per FOV.
        for well, pos_idx in targeted_positions:
            start, end = pos_idx, pos_idx + 1
            recon_jobs.append({
                "name": f"recon_{proc_name}_{well.replace('/', '_')}_"
                        f"{start:04d}_{end:04d}",
                "func": reconstruct,
                "kwargs": {
                    "experiment": experiment,
                    "well": well,
                    "process": process,
                    "position_start": start,
                    "position_end": end,
                },
                "metadata": {
                    "well": well,
                    "phase": "reconstruct",
                    "positions": f"{start}-{end}",
                },
            })
    elif position_start is not None or position_end is not None:
        # Targeted resubmission of a contiguous range within a well
        for well in wells:
            start = position_start or 0
            end = position_end
            n_positions = end - start
            n_chunks = (n_positions + chunk_size - 1) // chunk_size
            for chunk_idx in range(n_chunks):
                chunk_start = start + chunk_idx * chunk_size
                chunk_end = min(start + (chunk_idx + 1) * chunk_size, end)
                recon_jobs.append({
                    "name": f"recon_{proc_name}_{well.replace('/', '_')}_"
                            f"{chunk_start:04d}_{chunk_end:04d}",
                    "func": reconstruct,
                    "kwargs": {
                        "experiment": experiment,
                        "well": well,
                        "process": process,
                        "position_start": chunk_start,
                        "position_end": chunk_end,
                    },
                    "metadata": {
                        "well": well,
                        "phase": "reconstruct",
                        "positions": f"{chunk_start}-{chunk_end}",
                    },
                })
    else:
        # Discover position counts per well and chunk
        dataset = OpsDataset(experiment)
        phase3d_store_path = _resolve_path(
            dataset.store_paths[pcfg["phase3d_store_key"]])

        with open_ome_zarr(phase3d_store_path, mode="r") as store:
            all_positions = [a[0] for a in store.positions()]

        # Precheck: sample random positions to catch widespread empty data
        if skip_precheck:
            print("⏭️  Skipping input data precheck (--skip-precheck).")
        else:
            SAMPLE_SIZE = 50
            WARN_EMPTY_RATIO = 0.01  # warn if >1% of sampled tiles are empty
            MAX_EMPTY_RATIO = 0.10  # fail if >15% of sampled tiles are empty

            # For ops >= 69, well A/2 tracking t=0 is legitimately empty — skip it.
            ops_num = int(experiment.split("_")[0].replace("ops", ""))

            for well in wells:
                well_positions = sorted(
                    p for p in all_positions if p.startswith(well))
                sample = (random.sample(well_positions, min(SAMPLE_SIZE, len(well_positions))))

                skip_tp = [0] if (ops_num >= 69 and well == "A/2" and proc_name == "track") else None
                _, missing, empty = _validate_all_positions_have_data(
                    phase3d_store_path, sample, skip_timepoints=skip_tp)

                n_bad = len(missing) + len(empty)
                ratio = n_bad / len(sample) if sample else 0
                if ratio > MAX_EMPTY_RATIO:
                    raise RuntimeError(
                        f"\nERROR: Well {well} has too many empty/missing positions "
                        f"({n_bad}/{len(sample)} sampled = {ratio:.0%}, threshold {MAX_EMPTY_RATIO:.0%}).\n"
                        f"Store: {phase3d_store_path}\n"
                        f"  Missing: {missing[:10]}{'...' if len(missing) > 10 else ''}\n"
                        f"  Empty:   {empty[:10]}{'...' if len(empty) > 10 else ''}\n"
                        f"Fix upstream 3D reconstruction before running tilt-corrected reconstruction."
                    )
                elif ratio > WARN_EMPTY_RATIO:
                    print(
                        f"\n⚠️  WARNING: Well {well} has {n_bad}/{len(sample)} sampled "
                        f"empty/missing positions ({ratio:.0%}). Proceeding, but check "
                        f"upstream 3D reconstruction quality.\n"
                        f"  Missing: {missing[:10]}{'...' if len(missing) > 10 else ''}\n"
                        f"  Empty:   {empty[:10]}{'...' if len(empty) > 10 else ''}"
                    )

            print("✅ Input 3D positions spot-check passed (sampled {}, ≤{}% empty).".format(
                SAMPLE_SIZE, int(MAX_EMPTY_RATIO * 100)))

        # ── Work-stealing queue (default on, OPS_TILT_WORK_STEAL=0 disables) ──
        # Per-position queue + per-worker batch claim. Each SLURM job gets
        # ONE GPU and spawns ``workers_per_gpu`` independent subprocesses;
        # each subprocess claims its own batches of positions from the
        # shared NFS queue via atomic ``os.rename``. No chunk-level
        # synchronization — workers run their own loops independently and
        # naturally adapt batch size to their GPU class.
        # See _workqueue_build_positions / _workqueue_batch_claim_stride /
        # reconstruct_workstealing_v2 / _workstealing_worker_loop.
        if _work_steal:
            # Per-process defaults sized for typical position counts:
            # track wells contain ~296 positions total → 8 jobs × 4 GPUs
            # × ~2 workers/GPU ≈ 64 workers ≈ ~5 positions each. Pheno
            # has ~7 035 positions per experiment → 20 jobs handles it with
            # ~350 positions per job (~1.5 h per job on typical H200/H100
            # mix). OPS_TILT_WORK_STEAL_JOBS still overrides.
            n_jobs_env = os.environ.get("OPS_TILT_WORK_STEAL_JOBS")
            n_jobs = int(n_jobs_env) if n_jobs_env else (8 if proc_name == "track" else 20)
            positions_by_well = {
                well: sorted(p for p in all_positions if p.startswith(well))
                for well in wells
            }
            jobs, recon_slurm_params, queue_root, n_total, n_jobs, gpus_per_job = (
                _build_worksteal_jobs(
                    experiment, process, positions_by_well, recon_slurm_params,
                    name_prefix="recon", n_jobs=n_jobs,
                )
            )
            recon_jobs.extend(jobs)
            print(f"  Work-steal queue: {n_total} positions at {queue_root}")
            print(f"  Submitting {n_jobs} work-stealing jobs × {gpus_per_job} GPUs each")
        else:
            for well in wells:
                well_positions = [p for p in all_positions if p.startswith(well)]
                n_positions = len(well_positions)
                n_chunks = (n_positions + chunk_size - 1) // chunk_size

                for chunk_idx in range(n_chunks):
                    start = chunk_idx * chunk_size
                    end = min((chunk_idx + 1) * chunk_size, n_positions)
                    recon_jobs.append({
                        "name": f"recon_{proc_name}_{well.replace('/', '_')}_"
                                f"{start:04d}_{end:04d}",
                        "func": reconstruct,
                        "kwargs": {
                            "experiment": experiment,
                            "well": well,
                            "process": process,
                            "position_start": start,
                            "position_end": end,
                        },
                        "metadata": {
                            "well": well,
                            "phase": "reconstruct",
                            "positions": f"{start}-{end}",
                        },
                    })

    print(f"\nSubmitting {len(recon_jobs)} reconstruction jobs "
          f"({proc_name}, {chunk_size} positions/job)...")
    recon_result = submit_parallel_jobs(
        jobs_to_submit=recon_jobs,
        experiment=experiment,
        slurm_params=recon_slurm_params,
        log_dir=f"slurm_tilt_corrected_logs/{experiment}/{proc_name}/reconstruct",
        manifest_prefix=f"tilt_recon_{proc_name}",
        dry_run=dry_run,
        wait_for_completion=wait_for_completion,
    )

    # Auto-retry failed jobs (timeout, node failure, signal kills, etc.)
    if wait_for_completion and not dry_run:
        failed = recon_result.get("failed") or []
        if failed:
            import subprocess as _sp
            retry_jobs = []
            for item in failed:
                name, job_id = item if isinstance(item, tuple) else (item, None)
                # Check if the failure is retryable (TIMEOUT, CANCELLED, node failure)
                retryable = False
                if job_id:
                    try:
                        r = _sp.run(
                            ["sacct", "-j", job_id, "--format=State,ExitCode", "-n", "-P"],
                            capture_output=True, text=True, timeout=10,
                        )
                        retryable_states = {"TIMEOUT", "CANCELLED", "FAILED", "NODE_FAIL"}
                        retryable = any(
                            any(s in line for s in retryable_states)
                            for line in r.stdout.splitlines()
                        )
                    except Exception:
                        pass
                if retryable:
                    # Parse well and position range from job name: recon_<proc>_<well>_<start>_<end>
                    parts = name.split("_")
                    try:
                        pos_end = int(parts[-1])
                        pos_start = int(parts[-2])
                        well = "/".join(parts[-4:-2]).replace("_", "/")
                        # Reconstruct well from name format recon_proc_A_N_start_end
                        well = f"{parts[-4]}/{parts[-3]}"
                        retry_jobs.append({
                            "name": f"retry_{name}",
                            "func": reconstruct,
                            "kwargs": {
                                "experiment": experiment,
                                "well": well,
                                "process": process,
                                "position_start": pos_start,
                                "position_end": pos_end,
                            },
                            "metadata": {"well": well, "phase": "reconstruct_retry"},
                        })
                    except (ValueError, IndexError):
                        print(f"  ⚠️  Could not parse position range from job name: {name}")

            if retry_jobs:
                retry_params = {**recon_slurm_params, "timeout_min": recon_slurm_params["timeout_min"] * 2}
                print(f"\n🔁 Auto-retrying {len(retry_jobs)} failed job(s) "
                      f"with {retry_params['timeout_min']}min timeout...")
                recon_result = submit_parallel_jobs(
                    jobs_to_submit=retry_jobs,
                    experiment=experiment,
                    slurm_params=retry_params,
                    log_dir=f"slurm_tilt_corrected_logs/{experiment}/{proc_name}/reconstruct_retry",
                    manifest_prefix=f"tilt_recon_{proc_name}_retry",
                    dry_run=dry_run,
                    wait_for_completion=True,
                )

    # --- Phase 3: Auto-audit output store for empty tiles ---
    if wait_for_completion and not dry_run:
        print(f"\n{'='*60}")
        print(f"  Auto-auditing output store for empty tiles...")
        print(f"{'='*60}")

        empty = audit_output(experiment, wells=wells, process=process)

        if empty and _work_steal:
            # Rerun empty tiles through the SAME work-stealing queue the
            # full run uses, rather than one SLURM job per FOV (1 GPU each).
            # The empty set is sparse (non-contiguous positions across wells)
            # so the per-FOV path can't batch it; a work-steal queue can. At
            # scale (~2000 empty tiles) per-FOV jobs run ~1 FOV/hr/GPU; the
            # pool clears the same set in one wave across N jobs × gpus_per_job.
            rerun_by_well = defaultdict(list)
            for well, pos_idx, pos_name, empty_ts in empty:
                rerun_by_well[well].append(pos_name)

            audit_retry_jobs, audit_params, queue_root, n_total, n_jobs, gpus_per_job = (
                _build_worksteal_jobs(
                    experiment, process, dict(rerun_by_well), recon_slurm_params,
                    queue_tag="rerun", name_prefix="audit_rerun",
                    metadata_phase="audit_rerun_workstealing",
                )
            )
            print(f"\n🔁 Rerunning {n_total} empty positions via work-steal "
                  f"queue: {n_jobs} jobs × {gpus_per_job} GPUs "
                  f"(queue: {queue_root})")
            submit_parallel_jobs(
                jobs_to_submit=audit_retry_jobs,
                experiment=experiment,
                slurm_params=audit_params,
                log_dir=f"slurm_tilt_corrected_logs/{experiment}/{proc_name}/audit_rerun",
                manifest_prefix=f"tilt_recon_{proc_name}_audit_rerun",
                dry_run=dry_run,
                wait_for_completion=True,
            )
        elif empty:
            # Fallback (OPS_TILT_WORK_STEAL=0): one job per FOV. Single
            # position per task, so the multi-GPU dispatch path doesn't
            # fire — keep 1 GPU so we don't reserve the recon-batch shape.
            audit_retry_jobs = []
            for well, pos_idx, pos_name, empty_ts in empty:
                audit_retry_jobs.append({
                    "name": f"audit_rerun_{proc_name}_{pos_name.replace('/', '_')}",
                    "func": reconstruct,
                    "kwargs": {
                        "experiment": experiment,
                        "well": well,
                        "process": process,
                        "position_start": pos_idx,
                        "position_end": pos_idx + 1,
                    },
                    "metadata": {"well": well, "phase": "audit_rerun", "position": pos_name},
                })

            audit_params = {
                **recon_slurm_params,
                "timeout_min": recon_slurm_params["timeout_min"] * 3,
                "gpus_per_node": 1,
            }
            print(f"\n🔁 Rerunning {len(audit_retry_jobs)} empty positions "
                  f"with {audit_params['timeout_min']}min timeout...")
            submit_parallel_jobs(
                jobs_to_submit=audit_retry_jobs,
                experiment=experiment,
                slurm_params=audit_params,
                log_dir=f"slurm_tilt_corrected_logs/{experiment}/{proc_name}/audit_rerun",
                manifest_prefix=f"tilt_recon_{proc_name}_audit_rerun",
                dry_run=dry_run,
                wait_for_completion=True,
            )

        if empty:
            # Final audit
            print(f"\n{'='*60}")
            print(f"  Final audit after rerun...")
            print(f"{'='*60}")
            still_empty = audit_output(experiment, wells=wells, process=process)

            if still_empty:
                n_empty = len(still_empty)
                positions_str = ", ".join(f"{pos}" for _, _, pos, _ in still_empty[:10])
                if n_empty > 10:
                    positions_str += f"... and {n_empty - 10} more"
                raise RuntimeError(
                    f"\n{'='*60}\n"
                    f"  RECONSTRUCTION ERROR: {n_empty} positions still empty after audit rerun\n"
                    f"  Experiment: {experiment} ({proc_name})\n"
                    f"  Positions: {positions_str}\n"
                    f"  These positions may have persistent reconstruction failures.\n"
                    f"  Inspect the SLURM logs for details.\n"
                    f"{'='*60}"
                )

    # --- Phase 4: Collect per-chunk tilt CSVs into tilt_params_all.csv ---
    if wait_for_completion and not dry_run:
        dataset = OpsDataset(experiment)
        recon_dir = _resolve_path(dataset.store_paths[pcfg["phase3d_store_key"]]).parent
        phase_recon_dir = dataset.results / "phase_recon"
        phase_recon_dir.mkdir(parents=True, exist_ok=True)
        for well in wells:
            well_tag = well.replace("/", "_")
            tilt_dir = recon_dir / "tilt_calibration" / proc_name / well_tag
            _collect_and_summarize_tilt(
                tilt_dir, GRID_SIZE,
                title=f"{experiment} {proc_name} {well}")

            # Symlink plot and CSV into 3-assembly/phase_recon/ for QC dashboards
            for fname in ["tilt_summary_well.png", "tilt_params_all.csv"]:
                src = tilt_dir / fname
                if src.exists():
                    link = phase_recon_dir / f"{proc_name}_{well_tag}_{fname}"
                    link.unlink(missing_ok=True)
                    link.symlink_to(src)
                    print(f"  Symlinked {link.name} -> {src}")

    # Aggregate per-task GPU metrics from subtasks into a single per-step
    # summary so reporting tools don't have to walk the per-child entries.
    # In work-steal mode the children run ``reconstruct_workstealing_v2(...)``
    # keyed as ``reconstruct_workstealing_v2_<proc>_at<N>``; in the legacy
    # chunked path they ran ``reconstruct(...)`` keyed as
    # ``reconstruct_<proc>_<well>_ps<X>_pe<Y>``. Pick the right prefix.
    from cyclops_utils.profiling.decorators import emit_subtask_summary
    child_prefix = (
        f"reconstruct_workstealing_v2_{proc_name}"
        if _work_steal else
        f"reconstruct_{proc_name}"
    )
    summary = emit_subtask_summary(
        experiment,
        child_key_prefix=child_prefix,
        summary_key=f"reconstruct_tilt_corrected_{proc_name}",
    )
    if summary:
        print(
            f"  Per-step summary written: {summary.get('n_subtasks')} "
            f"subtasks; "
            f"avg SM util mean={summary.get('children_gpu_sm_util_avg_pct_mean')}%, "
            f"peak={summary.get('children_gpu_sm_util_max_pct')}%, "
            f"models={summary.get('children_gpu_models')}"
        )

    return recon_result


# ============================================================
# CLI
# ============================================================


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Per-run tilt-corrected 3D+2D phase reconstruction"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # calibrate
    cal = subparsers.add_parser(
        "calibrate", help="Chain calibration to fit per-run tilt model"
    )
    cal.add_argument("--experiment", required=True, help="Experiment name")
    cal.add_argument(
        "--well", required=True,
        help="Well to calibrate (e.g. A/1, A/2, A/3)",
    )
    cal.add_argument(
        "--process", default=None, choices=["pheno", "track"],
        help="Process type (default: pheno)",
    )
    cal.add_argument(
        "--resume", action="store_true",
        help="Skip tiles with existing params",
    )
    cal.add_argument(
        "--output-dir", default=None,
        help="Override base output directory (default: same as input stores)",
    )

    # reconstruct
    rec = subparsers.add_parser(
        "reconstruct", help="Apply tilt model: 3D+2D reconstruction"
    )
    rec.add_argument("--experiment", required=True, help="Experiment name")
    rec.add_argument(
        "--well", required=True,
        help="Well to reconstruct (e.g. A/1, A/2, A/3)",
    )
    rec.add_argument(
        "--process", default=None, choices=["pheno", "track"],
        help="Process type (default: pheno)",
    )
    rec.add_argument(
        "--debug-n-positions", type=int, default=None,
        help="Process only N positions (for testing)",
    )
    rec.add_argument(
        "--position-start", type=int, default=None,
        help="Start index in position list (for SLURM array splitting)",
    )
    rec.add_argument(
        "--position-end", type=int, default=None,
        help="End index (exclusive) in position list",
    )
    rec.add_argument(
        "--output-dir", default=None,
        help="Override base output directory (default: same as input stores)",
    )

    # submit
    sub = subparsers.add_parser(
        "submit", help="Submit calibration + reconstruction to SLURM"
    )
    sub.add_argument("--experiment", required=True, help="Experiment name")
    sub.add_argument(
        "--wells", nargs="+", default=None,
        help="Wells to process (default: A/1 A/2 A/3)",
    )
    sub.add_argument(
        "--process", nargs="+", default=None,
        choices=["pheno", "track"],
        help="Process type(s) (default: pheno). Pass both for all.",
    )
    sub.add_argument(
        "--chunk-size", type=int, default=20,
        help="Positions per reconstruction job (default: 1)",
    )
    sub.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without submitting",
    )
    sub.add_argument(
        "--no-wait", action="store_true",
        help="Don't wait for job completion",
    )
    sub.add_argument(
        "--positions", nargs="+", default=None, metavar="WELL:INDEX",
        help="Resubmit specific positions as 'well:index' pairs, e.g. A/1:31 A/1:61 A/2:132",
    )
    sub.add_argument(
        "--position-start", type=int, default=None,
        help="Resubmit a contiguous range from this index (single well, skips calibration)",
    )
    sub.add_argument(
        "--position-end", type=int, default=None,
        help="Resubmit a contiguous range up to this index (exclusive)",
    )

    # audit
    aud = subparsers.add_parser(
        "audit", help="Find empty/missing tiles in the output store and print a rerun command"
    )
    aud.add_argument("--experiment", required=True, help="Experiment name")
    aud.add_argument(
        "--wells", nargs="+", default=None,
        help="Wells to audit (default: A/1 A/2 A/3)",
    )
    aud.add_argument(
        "--process", default=None, choices=["pheno", "track"],
        help="Process type (default: pheno)",
    )

    # summarize
    summ = subparsers.add_parser(
        "summarize",
        help="Collect per-tile tilt CSVs into tilt_params_all.csv and generate summary plots (no recon)",
    )
    summ.add_argument("--experiment", required=True, help="Experiment name")
    summ.add_argument(
        "--wells", nargs="+", default=None,
        help="Wells to process (default: A/1 A/2 A/3)",
    )
    summ.add_argument(
        "--process", nargs="+", default=None,
        choices=["pheno", "track"],
        help="Process type(s) (default: both pheno and track)",
    )

    return parser


def audit_output(experiment, wells=None, process=None):
    """
    Scan the tilt-corrected output store for empty tiles (all-zero data
    at any timepoint). Prints a summary and a single submit command to
    rerun only the failed positions.
    """
    dataset = OpsDataset(experiment)
    pcfg = _get_process_config(process)
    proc_name = process or "pheno"
    output_2d_path = _resolve_path(dataset.store_paths[pcfg["output_2d_store_key"]])

    if wells is None:
        from cyclops_utils.data.filesystem import get_experiment_wells
        wells = get_experiment_wells(experiment, prefix_only=True)

    if not output_2d_path.exists():
        print(f"Output store does not exist: {output_2d_path}")
        return

    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor

    # For ops >= 69, A/2 tracking t=0 is legitimately empty — skip it
    ops_num = int(experiment.split("_")[0].replace("ops", ""))

    print(f"Auditing {output_2d_path} for empty positions...")

    with open_ome_zarr(output_2d_path, mode="r") as store:
        all_positions = [a[0] for a in store.positions()]

    empty_positions = []  # list of (well, position_index, position_name, empty_timepoints)

    for well in wells:
        well_positions = sorted(p for p in all_positions if p.startswith(well))
        skip_t0 = (ops_num >= 69 and well == "A/2" and proc_name == "track")

        def _check_position(args, _well=well, _skip_t0=skip_t0):
            idx, pos = args
            with open_ome_zarr(output_2d_path / pos, layout="fov", mode="r") as fov:
                arr = fov["0"]
                T = arr.shape[0]
                Y, X = arr.shape[-2], arr.shape[-1]
                # Check a 64x64 center crop per timepoint (TCZYX layout)
                y0, y1 = Y // 2 - 32, Y // 2 + 32
                x0, x1 = X // 2 - 32, X // 2 + 32
                empty_ts = []
                for t in range(T):
                    if _skip_t0 and t == 0:
                        continue
                    crop = np.array(arr[t, 0, 0, y0:y1, x0:x1])
                    if np.all(crop == 0):
                        empty_ts.append(t)
            return (_well, idx, pos, empty_ts) if empty_ts else None

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(tqdm(
                executor.map(_check_position, enumerate(well_positions)),
                total=len(well_positions),
                desc=f"  Auditing {well} ({proc_name}){' (skip t=0)' if skip_t0 else ''}",
            ))
        empty_positions.extend(r for r in results if r is not None)

    # Summary
    print(f"\n{'='*60}")
    print(f"  AUDIT RESULTS: {experiment} ({proc_name})")
    print(f"  Output store: {output_2d_path}")
    print(f"{'='*60}")

    if not empty_positions:
        print("  All positions have data. Nothing to rerun.")
        return

    print(f"\n  Found {len(empty_positions)} positions with empty timepoints:\n")
    for well, idx, pos, empty_ts in empty_positions:
        print(f"    {pos} (index {idx}): empty at t={empty_ts}")

    # Build rerun command
    pos_args = " ".join(f"{well}:{idx}" for well, idx, _, _ in empty_positions)
    cmd = (
        f"python -m cyclops_process.processes.reconstruct_tilt_corrected submit "
        f"--experiment {experiment} --process {proc_name} "
        f"--positions {pos_args}"
    )

    print(f"\n  To rerun these positions:\n")
    print(f"    {cmd}\n")

    return empty_positions


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "calibrate":
        calibrate(
            args.experiment,
            well=args.well,
            process=args.process,
            resume=args.resume,
            output_dir=args.output_dir,
        )
    elif args.command == "reconstruct":
        reconstruct(
            args.experiment,
            well=args.well,
            process=args.process,
            debug_n_positions=args.debug_n_positions,
            position_start=args.position_start,
            position_end=args.position_end,
            output_dir=args.output_dir,
        )
    elif args.command == "submit":
        processes = args.process or ["pheno"]
        position_start = getattr(args, "position_start", None)
        position_end = getattr(args, "position_end", None)
        targeted_positions = None
        if getattr(args, "positions", None):
            targeted_positions = []
            for token in args.positions:
                well, idx = token.rsplit(":", 1)
                targeted_positions.append((well, int(idx)))
        skip_cal = targeted_positions is not None or position_start is not None or position_end is not None
        for proc in processes:
            print(f"\n{'='*60}")
            print(f"  Process: {proc}")
            print(f"{'='*60}")
            submit_jobs(
                args.experiment,
                wells=args.wells,
                process=proc,
                chunk_size=args.chunk_size,
                dry_run=args.dry_run,
                wait_for_completion=not args.no_wait,
                _skip_calibration=skip_cal,
                position_start=position_start,
                position_end=position_end,
                targeted_positions=targeted_positions,
            )
    elif args.command == "audit":
        from cyclops_utils.data.filesystem import resolve_experiment_name
        resolved = resolve_experiment_name(args.experiment, allow_interactive=True)
        audit_output(
            resolved,
            wells=args.wells,
            process=args.process,
        )
    elif args.command == "summarize":
        from cyclops_utils.data.filesystem import resolve_experiment_name, get_experiment_wells
        resolved = resolve_experiment_name(args.experiment, allow_interactive=True)
        wells = args.wells or get_experiment_wells(resolved, prefix_only=True)
        wells = _normalize_wells(wells)
        processes = args.process or ["pheno", "track"]

        dataset = OpsDataset(resolved)
        phase_recon_dir = dataset.results / "phase_recon"
        phase_recon_dir.mkdir(parents=True, exist_ok=True)

        for proc in processes:
            pcfg = _get_process_config(proc)
            recon_dir = _resolve_path(dataset.store_paths[pcfg["phase3d_store_key"]]).parent
            print(f"\n{'='*60}")
            print(f"  Summarizing tilt params: {proc}")
            print(f"{'='*60}")
            for well in wells:
                well_tag = well.replace("/", "_")
                tilt_dir = recon_dir / "tilt_calibration" / proc / well_tag
                if not tilt_dir.exists():
                    print(f"  Skipping {proc} {well}: {tilt_dir} not found")
                    continue
                _collect_and_summarize_tilt(
                    tilt_dir, GRID_SIZE,
                    title=f"{resolved} {proc} {well}")
                for fname in ["tilt_summary_well.png", "tilt_params_all.csv"]:
                    src = tilt_dir / fname
                    if src.exists():
                        link = phase_recon_dir / f"{proc}_{well_tag}_{fname}"
                        link.unlink(missing_ok=True)
                        link.symlink_to(src)
                        print(f"  Symlinked {link.name} -> {src}")


if __name__ == "__main__":
    main()
