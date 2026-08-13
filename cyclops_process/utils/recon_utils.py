from iohub import open_ome_zarr
from pathlib import Path
import numpy as np
from tqdm import tqdm
from iohub.ngff import TransformationMeta
import os
import shutil
import time
import uuid
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
import time
import uuid
from pathlib import Path
from typing import Tuple, List, Dict, Any
import pandas as pd
import math

import sys

sys.path.insert(0, os.getcwd())


from cyclops_process.utils.waveorder_utils import ReconstructionConfig, yaml_to_model
from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.io.zarr_utils import (
    _discover_positions_fast_balanced,
    _maybe_sample_positions,
    _resolve_output_path_for_debug,
)
from cyclops_utils.io.zarr_precreate import create_hcs_store_fast
from cyclops_process.utils.waveorder_utils import (
    model_to_yaml,
    yaml_to_model,
    _normalize_subpixel_options,
    _get_wo_focus,
)

# Module-level cache for transfer functions keyed by optics + absolute offset + spatial dims (including Z)
_TF_CACHE: dict[tuple[float, float, float, float, int, int, int], Path] = {}


def _validate_subtile_grid(n_subtiles: int) -> int:
    """Validate that n_subtiles is a perfect square for grid layout."""
    sqrt_n = int(np.sqrt(n_subtiles))
    if sqrt_n * sqrt_n != n_subtiles:
        raise ValueError(
            f"n_subtiles ({n_subtiles}) must be a perfect square (4, 9, 16, 25, etc.)"
        )
    return sqrt_n


def _shape_key_from_shape(shape: Tuple[int, int, int, int, int]) -> str:
    t, c, z, y, x = shape
    return f"T{t}C{c}Z{z}Y{y}X{x}"


def _setup_auto2d_paths_and_config(
    experiment: str,
    process: str,
    debug_n_positions: int | None,
    debug_output_suffix: str,
    verbose: bool,
    ngff_version: str = "0.4",
):
    """Setup paths, configuration, and positions for 2D autofocus for either 20x phenotyping or 5x tracking.

    Selects correct input/output/config based on `process` ("pheno-2d" or "track-2d").
    """
    dataset = OpsDataset(experiment)
    if verbose:
        print(f"[Auto2D] Experiment: {experiment} | process={process}")

    is_pheno = process == "pheno-2d"
    # Paths and configs based on process
    if is_pheno:
        raw_store_path = dataset.store_paths["lc_20x"]
        phase3d_store_path = dataset.store_paths["lc_20x_phase_3d_optimized"]
        phase2d_store_path = dataset.store_paths["lc_20x_phase_2d_optimized"]
        config2d_path = dataset.config_paths["lc_20x_phase_recon_2d"]
    else:
        raw_store_path = dataset.store_paths["lc_5x_bf_corrected"]
        phase3d_store_path = dataset.store_paths["lc_5x_phase_3d_optimized"]
        phase2d_store_path = dataset.store_paths["lc_5x_phase_2d_optimized"]
        config2d_path = dataset.config_paths["lc_5x_phase_recon_2d"]

    # Load optical parameters directly from the 2D config using Pydantic models
    cfg2d_base_model = yaml_to_model(config2d_path, ReconstructionConfig)
    tf2d_model = cfg2d_base_model.phase.transfer_function
    optical_params = {
        "lambda_ill": tf2d_model.wavelength_illumination,
        "pixel_size": tf2d_model.yx_pixel_size,
        "NA_det": tf2d_model.numerical_aperture_detection,
    }

    # Use a transfer function consistent with the 2D config (matches existing stores)
    transfer_function_path = phase3d_store_path.parent / Path(
        "transfer_function_" + Path(config2d_path).stem + ".zarr"
    )

    # Discover positions with fast balanced mode for debug
    if debug_n_positions is not None and debug_n_positions > 0:
        positions = _discover_positions_fast_balanced(
            phase3d_store_path, int(debug_n_positions)
        )
    else:
        with open_ome_zarr(phase3d_store_path, mode="r", version=ngff_version) as phase3d:
            positions = [a[0] for a in phase3d.positions()]
        positions = _maybe_sample_positions(positions, debug_n_positions)

    if verbose:
        print(f"[Auto2D] Positions discovered: {len(positions)}")

    phase2d_store_path = _resolve_output_path_for_debug(
        phase2d_store_path, debug_n_positions, debug_output_suffix
    )

    return {
        "dataset": dataset,
        "raw_store_path": raw_store_path,
        "phase3d_store_path": phase3d_store_path,
        "phase2d_store_path": phase2d_store_path,
        "config2d_path": config2d_path,
        "cfg2d_base_model": cfg2d_base_model,
        "optical_params": optical_params,
        "transfer_function_path": transfer_function_path,
        "positions": positions,
    }


# no longer used but keeping for now if extreme subtiling focal differences become apparent
def _bound_by_neighbors(
    grid: np.ndarray, delta: float = 1.0, max_iters: int | None = None
) -> np.ndarray:
    g = grid.copy().astype(float)
    G = g.shape[0]
    if max_iters is None:
        max_iters = G * G
    for _ in range(int(max_iters)):
        changed = False
        for rr in range(G):
            for cc in range(G):
                val = g[rr, cc]
                low, high = -np.inf, np.inf
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < G and 0 <= nc < G and np.isfinite(g[nr, nc]):
                        low = max(low, g[nr, nc] - delta)
                        high = min(high, g[nr, nc] + delta)
                if low <= high:
                    new_val = min(max(val, low), high)
                    if new_val != val:
                        g[rr, cc] = new_val
                        changed = True
        if not changed:
            break
    return g


def _calculate_subtile_bounds(
    Y: int, X: int, grid_size: int
) -> List[Tuple[int, int, int, int]]:
    """Calculate bounds for each subtile in the grid.

    Returns:
        List of (y_start, y_end, x_start, x_end) tuples for each subtile
    """
    subtile_height = Y // grid_size
    subtile_width = X // grid_size

    bounds = []
    for row in range(grid_size):
        for col in range(grid_size):
            y_start = row * subtile_height
            y_end = (row + 1) * subtile_height if row < grid_size - 1 else Y
            x_start = col * subtile_width
            x_end = (col + 1) * subtile_width if col < grid_size - 1 else X
            bounds.append((y_start, y_end, x_start, x_end))

    return bounds


def _create_output_stores(
    phase3d_store_path: Path,
    phase2d_store_path: Path,
    positions: list[str],
    dataset: OpsDataset,
    cfg2d_base_model,
    verbose: bool,
    ngff_version: str = "0.4",
):
    """Create and initialize output store with channels: 0=Phase2D, 1=Focus3D.

    The time dimension (T) is inherited from the source 3D phase store. For 20x (T=1),
    and for 5x tracking (T>1), we allocate (T, 2, 1, Y, X).
    """
    with open_ome_zarr(phase3d_store_path, mode="r", version=ngff_version) as phase3d:
        ref_pos = phase3d[positions[0]]
        ref_arr = ref_pos["0"]
        T, C, Z, Y, X = ref_arr.shape
        # Prefer position scale; fallback to dataset 20x scale
        try:
            position_scale = ref_pos.scale
        except Exception:
            position_scale = dataset.store_props.get(
                "20x_scale", [1.0, 1.0, 1.0, 0.325, 0.325]
            )
        out_dtype = np.float32
        # Create the focus store structure (match source chunking for Y/X; Z chunk forced to 1)
        ref_chunks = ref_arr.chunks  # (T,C,Z,Y,X)
        focus_chunks = (1, 1, 1, ref_chunks[3], ref_chunks[4])

        if verbose:
            print(
                f"[Auto2D] Output shape per pos: (T={T}, C=2, Z=1, Y={Y}, X={X}) | dtype={out_dtype}"
            )

        # Use fast precreation method (38x faster, O(1) scaling)
        if verbose:
            print(
                f"[Auto2D] Pre-creating {len(positions)} positions using fast method..."
            )

        create_hcs_store_fast(
            store_path=phase2d_store_path,
            positions=positions,
            shape=(T, 2, 1, Y, X),
            chunks=dataset.store_props["chunk_size"],
            dtype=out_dtype,
            scale=position_scale,
            channel_names=["Phase2D", "Focus3D"],
            version=ngff_version,
        )

    return T, Y, X, position_scale, out_dtype


def _tf_cache_key(
    cfg2d_base_model, abs_offset: float, height: int, width: int, z_planes: int
) -> tuple[float, float, float, float, int, int, int]:
    tfm = cfg2d_base_model.phase.transfer_function
    return (
        float(tfm.wavelength_illumination),
        float(tfm.yx_pixel_size),
        float(tfm.numerical_aperture_detection),
        float(round(abs_offset, 6)),
        int(height),
        int(width),
        int(z_planes),
    )


def _make_tf_cache_path(
    cache_dir: Path, cfg2d_base_model, abs_offset: float, height: int, width: int, z_planes: int
) -> Path:
    tfm = cfg2d_base_model.phase.transfer_function
    # Offset tag like ZOFFP1P5/ZOFFM2
    sign = "M" if abs_offset < 0 else "P"
    val = abs(abs_offset)
    if abs(val - round(val)) < 1e-6:
        body = f"{int(round(val))}"
    elif abs(val - (int(val) + 0.5)) < 1e-6:
        body = f"{int(val)}P5"
    else:
        body = f"{val:.2f}".replace("-", "").replace(".", "P")
    offset_tag = f"ZOFF{sign}{body}"

    lam_str = f"{float(tfm.wavelength_illumination):.4f}".replace(".", "P")
    px_str = f"{float(tfm.yx_pixel_size):.4f}".replace(".", "P")
    na_str = f"{float(tfm.numerical_aperture_detection):.3f}".replace(".", "P")
    name = f"tf_cache_{offset_tag}_lam{lam_str}_px{px_str}_na{na_str}_Z{z_planes}_H{height}_W{width}.zarr"
    return cache_dir / name


def _get_or_compute_tf_cached(
    dataset: OpsDataset,
    cfg2d_base_model,
    abs_offset: float,
    example_input_position_dirpath: Path,
    cache_dir: Path,
    height: int | None = None,
    width: int | None = None,
    z_planes: int | None = None,
    verbose: bool = False,
) -> Path:
    """Return cached TF path for given optics/offset/shape; compute if missing.

    The TF depends on optics, z_focus_offset and spatial dimensions (H, W, Z). No padding is attempted.
    """
    from waveorder.cli.compute_transfer_function import compute_transfer_function_cli
    # Determine spatial dims if not provided
    if height is None or width is None or z_planes is None:
        try:
            with open_ome_zarr(example_input_position_dirpath, mode="r") as pos_store:
                try:
                    arr = pos_store["0"]
                except Exception:
                    with open_ome_zarr(
                        example_input_position_dirpath / "0", mode="r"
                    ) as arr_store:
                        arr = arr_store
                _, _, _Z, H, W = arr.shape
        except Exception:
            H, W, _Z = 256, 256, 1
    else:
        H, W, _Z = int(height), int(width), int(z_planes)

    # Use provided values or inferred values
    if height is None:
        height = H
    if width is None:
        width = W
    if z_planes is None:
        z_planes = _Z

    H, W, Z = int(height), int(width), int(z_planes)

    key = _tf_cache_key(cfg2d_base_model, abs_offset, H, W, Z)
    if key in _TF_CACHE and _TF_CACHE[key].exists():
        return _TF_CACHE[key]

    cache_dir.mkdir(parents=True, exist_ok=True)
    tf_cache_path = _make_tf_cache_path(cache_dir, cfg2d_base_model, abs_offset, H, W, Z)

    # Helper to validate an existing TF store (guards against partially written stores)
    def _is_valid_tf_store(p: Path) -> bool:
        try:
            if not p.exists() or not p.is_dir():
                return False
            # Consider valid if any immediate child (or one level deeper) has a .zarray
            for child in p.iterdir():
                if (child / ".zarray").exists():
                    return True
                if child.is_dir():
                    for grand in child.iterdir():
                        if (grand / ".zarray").exists():
                            return True
            return False
        except Exception:
            return False

    if tf_cache_path.exists():
        if _is_valid_tf_store(tf_cache_path):
            _TF_CACHE[key] = tf_cache_path
            return tf_cache_path
        else:
            # Remove corrupted/incomplete store and recompute
            try:
                shutil.rmtree(tf_cache_path)
            except Exception:
                pass

    # Compute with a simple file lock to avoid concurrent writers
    lock_path = Path(str(tf_cache_path) + ".lock")
    got_lock = False
    while not got_lock:
        try:
            # Atomic exclusive creation
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            got_lock = True
        except FileExistsError:
            # Another process is computing it; wait and then validate
            time.sleep(0.2)
            if tf_cache_path.exists() and _is_valid_tf_store(tf_cache_path):
                _TF_CACHE[key] = tf_cache_path
                return tf_cache_path

    try:
        # Double-check another process didn't finish while we waited
        if tf_cache_path.exists() and _is_valid_tf_store(tf_cache_path):
            _TF_CACHE[key] = tf_cache_path
            return tf_cache_path

        # Write temporary config with this absolute offset
        cfg2d_model = cfg2d_base_model.copy(deep=True)
        cfg2d_model.phase.transfer_function.z_focus_offset = float(abs_offset)
        tmp_cfg = cache_dir / f"temp_tf_cache_{uuid.uuid4().hex[:8]}.yaml"
        tmp_out = cache_dir / f"{tf_cache_path.name}.tmp_{uuid.uuid4().hex[:8]}"
        try:
            model_to_yaml(cfg2d_model, tmp_cfg)
            if verbose:
                print(
                    f"[Auto2D] Caching TF for abs_offset={abs_offset:+.1f} @ {Z}x{H}x{W} → {tf_cache_path.name}"
                )
            with open(os.devnull, "w") as devnull, redirect_stdout(
                devnull
            ), redirect_stderr(devnull):
                compute_transfer_function_cli(
                    input_position_dirpath=example_input_position_dirpath,
                    config_filepath=tmp_cfg,
                    output_dirpath=tmp_out,
                )
            # Validate temp output, then atomically rename into place
            if not _is_valid_tf_store(tmp_out):
                if verbose:
                    print(
                        f"[Auto2D][WARN] TF temp store validation failed for {tmp_out.name}; proceeding to finalize and relying on retry if needed."
                    )
            # Ensure destination does not exist, then move into place atomically if same FS
            if tf_cache_path.exists():
                try:
                    shutil.rmtree(tf_cache_path)
                except Exception:
                    pass
            try:
                os.rename(tmp_out, tf_cache_path)
            except Exception:
                shutil.move(str(tmp_out), str(tf_cache_path))
            # Optional: brief visibility backoff for networked FS
            for _ in range(10):
                if _is_valid_tf_store(tf_cache_path):
                    break
                time.sleep(0.05)
        finally:
            tmp_cfg.unlink(missing_ok=True)
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass

    _TF_CACHE[key] = tf_cache_path
    return tf_cache_path


def _infer_focus_index(
    zyx_stack: np.ndarray,
    NA_det: float,
    lambda_ill: float,
    pixel_size: float,
    enable_subpixel_precision: bool = False,
    polynomial_fit_order: int | None = None,
    midband_fractions: Tuple[float, float] = (0.125, 0.25),
    device: str = "cpu",
) -> float | int | None:
    """Return in-focus slice index using Waveorder's official method.

    When enable_subpixel_precision is True, returns a float index using polynomial fitting.
    Otherwise returns an integer index (default, backwards-compatible).

    Args:
        zyx_stack: 3D array (Z, Y, X)
        NA_det: Numerical aperture of detection
        lambda_ill: Illumination wavelength (microns)
        pixel_size: Pixel size (microns)
        enable_subpixel_precision: Enable subpixel focus finding
        polynomial_fit_order: Order for polynomial fitting
        midband_fractions: (low, high) fractions of max frequency for bandpass
        device: Device for computation ('cpu' or 'cuda')
    """
    if zyx_stack.ndim != 3:
        raise ValueError("zyx_stack must be 3D (Z,Y,X)")
    if zyx_stack.shape[0] == 1:
        return 0
    wo_focus = _get_wo_focus()
    if wo_focus is None:
        raise ImportError(
            "waveorder.focus.focus_from_transverse_band is not available in this environment. "
            "Please install/update Waveorder to a version exposing this function."
        )
    # Options based on API capabilities
    enabled, order, support = _normalize_subpixel_options(
        enable_subpixel_precision, polynomial_fit_order
    )

    # Build kwargs only with supported parameters to avoid TypeError on older versions
    kwargs = {
        "NA_det": NA_det,
        "lambda_ill": lambda_ill,
        "pixel_size": pixel_size,
        "midband_fractions": midband_fractions,
        "mode": "max",
        "plot_path": None,
        "threshold_FWHM": 0,
    }
    if support.get("polynomial_fit_order", False):
        kwargs["polynomial_fit_order"] = order if enabled else None
    if support.get("enable_subpixel_precision", False):
        kwargs["enable_subpixel_precision"] = bool(enabled)

    # Pass device if the waveorder version supports it (GPU-enabled version)
    import inspect
    sig = inspect.signature(wo_focus)
    if "device" in sig.parameters:
        kwargs["device"] = device

    return wo_focus(zyx_stack, **kwargs)


def save_subtile_metadata(
    subtile_results: Dict[str, Dict[str, Any]],
    output_dir: Path,
    experiment: str,
    tag: str | None = None,
) -> Path:
    """Save subtile metadata to CSV file for future analysis.

    Args:
        subtile_results: Dictionary of position -> subtile results
        output_dir: Output directory for CSV file
        experiment: Experiment name

    Returns:
        Path to saved CSV file
    """
    records = []

    for pos, result in subtile_results.items():
        for subtile_info in result["subtile_metadata"]:
            y_start, y_end, x_start, x_end = subtile_info["bounds"]
            records.append(
                {
                    "experiment": experiment,
                    "position": pos,
                    "subtile_id": subtile_info["subtile_id"],
                    "y_start": y_start,
                    "y_end": y_end,
                    "x_start": x_start,
                    "x_end": x_end,
                    "focus_index": int(
                        round(float(subtile_info.get("focus_index", 0)))
                    ),
                    # Float sub-pixel estimate, if present
                    "focus_index_float": float(
                        subtile_info.get(
                            "focus_index_float",
                            float(subtile_info.get("focus_index", 0.0)),
                        )
                    ),
                    "z_stack_size": subtile_info["z_stack_size"],
                    "z_focus_offset": subtile_info["z_focus_offset"],
                    "reconstruction_success": subtile_info["reconstruction_success"],
                }
            )

    df = pd.DataFrame(records)
    tag_local = (tag or "unknown").replace(" ", "_")
    csv_path = output_dir / f"{experiment}_{tag_local}_subtile_metadata.csv"
    print(f"[SubtileRecon] Saving subtile metadata to {csv_path}")
    df.to_csv(csv_path, index=False)

    return csv_path


def generate_subtile_report(
    subtile_results: Dict[str, Dict[str, Any]], experiment: str
):
    """Generate comprehensive report of subtile reconstruction results."""
    print("\n" + "=" * 80)
    print("SUBTILE AUTOFOCUS RECONSTRUCTION REPORT")
    print("=" * 80)
    print(f"Experiment: {experiment}")

    total_positions = len(subtile_results)
    total_subtiles = sum(result["n_subtiles"] for result in subtile_results.values())
    successful_positions = sum(
        1 for result in subtile_results.values() if result["reconstruction_success"]
    )
    total_successful_subtiles = sum(
        result["successful_subtiles"] for result in subtile_results.values()
    )

    print(f"Total positions: {total_positions}")
    print(f"Total subtiles: {total_subtiles}")
    print(f"Successful positions: {successful_positions}/{total_positions}")

    # Avoid division by zero if no subtiles were processed
    if total_subtiles > 0:
        success_pct = 100 * total_successful_subtiles / total_subtiles
        print(
            f"Successful subtiles: {total_successful_subtiles}/{total_subtiles} ({success_pct:.1f}%)"
        )
    else:
        print(f"Successful subtiles: 0/0 (N/A - no subtiles processed)")

    # If no results, exit early
    if not subtile_results:
        print("\n" + "=" * 80)
        return

    # Grid configuration summary
    grid_sizes = [result["grid_size"] for result in subtile_results.values()]
    blend_pixels = [result["blend_pixels"] for result in subtile_results.values()]

    print(f"\nConfiguration:")
    print(f"Grid size: {grid_sizes[0]}x{grid_sizes[0]} (n_subtiles={grid_sizes[0]**2})")
    print(f"Blend pixels: {blend_pixels[0]}")

    # Focus statistics across all subtiles
    all_focus_indices = []
    all_z_offsets = []

    for result in subtile_results.values():
        for subtile_info in result["subtile_metadata"]:
            all_focus_indices.append(subtile_info["focus_index"])
            all_z_offsets.append(subtile_info["z_focus_offset"])

    if all_focus_indices:
        print(f"\nFocus Statistics Across All Subtiles:")
        print(
            f"Focus indices: min={min(all_focus_indices)}, max={max(all_focus_indices)}, mean={np.mean(all_focus_indices):.1f}"
        )
        print(
            f"Z focus offsets: min={min(all_z_offsets):.2f}, max={max(all_z_offsets):.2f}, mean={np.mean(all_z_offsets):.2f}"
        )

    # Per-position summary
    print(f"\nPER-POSITION SUMMARY")
    print("-" * 80)
    print(
        f"{'Position':<20} {'Subtiles':<10} {'Success':<10} {'Focus_Range':<15} {'Offset_Range':<15}"
    )
    print("-" * 80)

    for pos in sorted(subtile_results.keys()):
        result = subtile_results[pos]
        focus_range = (
            f"{result['focus_index_range'][0]}-{result['focus_index_range'][1]}"
        )
        offset_range = (
            f"{result['z_offset_range'][0]:.1f}-{result['z_offset_range'][1]:.1f}"
        )

        print(
            f"{pos:<20} {result['n_subtiles']:<10} {result['successful_subtiles']:<10} {focus_range:<15} {offset_range:<15}"
        )

    print("=" * 80)
