from typing import Optional
from pathlib import Path
import shutil
import time
import sys
import os
import re

# Configure stitch package location via STITCH_PATH environment variable
# If STITCH_PATH is set, it will override the default stitch package location
# (which is typically set via a .pth file in site-packages)
#
# Usage:
#   export STITCH_PATH=/path/to/your/stitch/package
#   python cyclops_process/processes/run.py ...
#
# If STITCH_PATH is not set, the default stitch package (from .pth file) will be used
if "STITCH_PATH" in os.environ:
    stitch_path = os.environ["STITCH_PATH"]
    # Remove existing stitch-related paths from sys.path
    sys.path = [p for p in sys.path if 'stitch' not in p or 'site-packages' in p]
    # Insert custom stitch path at the beginning
    sys.path.insert(0, stitch_path)

from stitch.stitch import assemble

# Verify which stitch package is being used
import inspect
_assemble_path = inspect.getfile(assemble)
print(f"✓ Using stitch package from: {_assemble_path}")
del inspect, _assemble_path


# OPS phenotyping native YX pixel size (µm/px). The pre-fix upstream input store
# (lc_20x_phase_2d_optimized.zarr) declares 0.65 here, a 2× over-declaration:
# the Dragonfly raw acquisition is 0.325 µm/px and the optimized store is at
# native resolution (no spatial binning). For pheno-like processes we override
# the YX scale during stitching so the v3 output level-0 multiscale records the
# correct value; pyramid levels then halve correctly from there.
_PHENO_NATIVE_YX = 0.325
_PHENO_PROCESSES = {"pheno", "pheno-2d"}
# Track (5x) native YX is 4x pheno (20x): 0.325 * 4 = 1.3 µm/px. The optimized
# input store over-declares it, so force it here to keep the v3 level-0 scale
# correct — the viewer's 4x track→pheno upscale depends on this consistency.
_TRACK_NATIVE_YX = 1.3
_TRACK_PROCESSES = {"track", "track-2d"}


def _force_yx_scale_for(process: Optional[str]) -> Optional[tuple]:
    """Return the YX scale override (µm/px) for a given process, or None when
    the process's input store already declares the correct value."""
    proc = (process or "").lower()
    if proc in _PHENO_PROCESSES:
        return (_PHENO_NATIVE_YX, _PHENO_NATIVE_YX)
    if proc in _TRACK_PROCESSES:
        return (_TRACK_NATIVE_YX, _TRACK_NATIVE_YX)
    return None

from cyclops_process.metrics.plate_stats.plate_stats_stitch_metrics import (
    load_stitch_config,
    compute_stitch_confidence_stats,
)
from ops_utils.data.experiment import OpsDataset
from ops_utils.profiling.decorators import versioned_function
from ops_utils.data.filesystem import (
    ensure_output_path,
    async_delete_path,
)
from ops_utils.io.zarr_utils import (
    _resolve_output_path_for_debug,
    _discover_positions,
    add_missing_zarr_metadata,
    detect_zarr_format,
)
import os

try:
    import cupy as xp
    from cupyx.scipy import ndimage as cundi
except (ModuleNotFoundError, ImportError):
    import numpy as xp
    from scipy import ndimage as cundi


def check_stitch_confidence(
    config_path: str,
    threshold: float = 0.7,
    experiment: str = None,
) -> dict:
    """
    Check per-well mean stitch confidence from a stitch settings YAML.

    Reads the confidence section of the YAML, computes mean confidence per well,
    and raises an error if any well falls below the threshold.

    Args:
        config_path: Path to the stitch settings YAML file.
        threshold: Minimum acceptable mean confidence per well (default 0.7 = 70%).
        experiment: Experiment name. If provided, only checks wells listed in the
            experiment's wells_to_process config.

    Returns:
        Dict mapping well names to their StitchConfidenceStats.

    Raises:
        RuntimeError: If any well has mean confidence below threshold.
    """
    config_path = Path(config_path)
    config = load_stitch_config(config_path)
    if config is None or "confidence" not in config:
        print(f"[stitch_confidence_check] WARNING: No confidence data in {config_path}, skipping check.")
        return {}

    confidence_dict = config["confidence"]
    if not confidence_dict:
        print(f"[stitch_confidence_check] WARNING: Empty confidence dict in {config_path}, skipping check.")
        return {}

    # Filter to only wells in the experiment's config
    valid_wells = None
    if experiment:
        from ops_utils.data.filesystem import get_experiment_wells
        valid_wells = set(get_experiment_wells(experiment, prefix_only=True))

    per_well_stats = {}
    failing_wells = []

    for well_key in confidence_dict:
        if valid_wells is not None and well_key not in valid_wells:
            print(f"[stitch_confidence_check] Skipping {well_key} (not in wells_to_process)")
            continue
        stats = compute_stitch_confidence_stats(config_path, well=well_key)
        if stats is None:
            print(f"[stitch_confidence_check] WARNING: Could not compute stats for well {well_key}")
            continue
        per_well_stats[well_key] = stats
        status = "PASS" if stats.mean >= threshold else "FAIL"
        print(
            f"[stitch_confidence_check] Well {well_key}: mean={stats.mean:.4f}, "
            f"median={stats.median:.4f}, min={stats.min:.4f}, max={stats.max:.4f}, "
            f"edges={stats.num_edges} [{status}]"
        )
        if stats.mean < threshold:
            failing_wells.append((well_key, stats.mean))

    if failing_wells:
        details = ", ".join(f"{w} (mean={m:.4f})" for w, m in failing_wells)
        raise RuntimeError(
            f"Stitch confidence check FAILED. The following wells have mean confidence "
            f"below {threshold*100:.0f}%: {details}. "
            f"Config: {config_path}"
        )

    print(f"[stitch_confidence_check] All {len(per_well_stats)} wells passed (threshold={threshold*100:.0f}%).")
    return per_well_stats


@versioned_function("v1.0")
def stitch(
    experiment: str = None,
    process: str = None,
    input_store_path: str = None,
    output_store_path: str = None,
    config_path: str = None,
    flipud: bool = False,
    fliplr: bool = False,
    rot90: int = 0,
    debug_n_positions: int | None = None,
    debug_output_suffix: str = "_debug",
) -> None:
    """
    Mimic of biahub stitching function
    """

    if experiment is None:
        if input_store_path is None or output_store_path is None or config_path is None:
            raise ValueError(
                "Either experiment or input/output directories must be provided."
            )
        pass

    else:
        dataset = OpsDataset(experiment)

        if process == "iss":
            config_path = dataset.config_paths["iss_stitch"]
            input_store_path = dataset.store_paths[
                "iss_drift_corrected"
            ]  # iss_distortion_corrected
            output_store_path = dataset.store_paths["iss_stitch"]
            flipud = True
            fliplr = False
            rot90 = 0

        elif process == "lc_5x_phase":
            raise NotImplementedError("5x phase stitching not implemented yet")
            # assemble.stitch(
            #     config_path=,
            #     input_store_path=,
            #     output_store_path=,
            #     flipud=,
            #     fliplr=,
            #     rot90=,
            # )

        elif process == "lc_20x_phase":
            raise NotImplementedError("20x phase stitching not implemented yet")
            # assemble.stitch(
            #     config_path=,
            #     input_store_path=,
            #     output_store_path=,
            #     flipud=,
            #     fliplr=,
            #     rot90=,
            # )

        elif process == "lc_20x_fluorescence":
            raise NotImplementedError("20x fluorescence stitching not implemented yet")
            # assemble.stitch(
            #     config_path=,
            #     input_store_path=,
            #     output_store_path=,
            #     flipud=,
            #     fliplr=,
            #     rot90=,
            # )

    # Debug-aware output path handling
    if output_store_path is not None:
        output_store_path = _resolve_output_path_for_debug(
            output_store_path, debug_n_positions, debug_output_suffix
        )
        # Ensure it's a Path object
        output_store_path = Path(output_store_path)

    # Rebuild from scratch: stitching rewrites the whole canvas.
    async_delete_path(output_store_path)
    # Choose default numeric precision: 16-bit for ISS/Track, 32-bit for Pheno
    proc = (process or "").lower() if process is not None else ""
    _is_track = proc in ("track", "track-2d", "lc_5x_phase", "lc_5x_phase_2d")
    _is_iss = proc == "iss"
    default_bits = 16 if (_is_iss or _is_track) else 32

    assemble.stitch(
        config_path=config_path,
        input_store_path=input_store_path,
        output_store_path=output_store_path,
        flipud=flipud,
        fliplr=fliplr,
        rot90=rot90,
        value_precision_bits=default_bits,
        force_yx_scale=_force_yx_scale_for(process),
    )

    return


@versioned_function("v1.0")
def estimate_stitch_parameters(
    experiment: str = None,
    process: str = None,
    input_store_path: str = None,
    output_config_path: str = None,
    flipud: bool = False,
    fliplr: bool = False,
    rot90: int = 0,
    debug_n_positions: int | None = None,
    debug_output_suffix: str = "_debug",
    verbose: bool = False,
    overlap: int = 150,
    channel: int = 0,
    timepoint: int = 0,
    tile_size: tuple = (2048, 2048),
) -> None:
    """
    Mimic of biahub estimate stitch
    """

    # Default: no per-well timepoint overrides
    timepoint_per_well = None

    if experiment is None:
        if input_store_path is None or output_config_path is None:
            raise ValueError(
                "Either experiment or input/output directories must be provided."
            )
        pass

    else:
        dataset = OpsDataset(experiment)

        if process == "iss":
            print(f"Estimating shifts for ISS experiment: {experiment}")
            input_store_path = dataset.store_paths[
                "iss_drift_corrected"
            ]  # iss_distortion_corrected
            output_config_path = dataset.config_paths["iss_stitch"]
            # flipud/fliplr/rot90 come from config params (defaults set in generate_config_files.py)

        if process == "track":
            print(f"Estimating shifts for 5x phase (2D recon) experiment: {experiment}")
            input_store_path = dataset.store_paths["lc_5x_phase_2d_optimized"]
            output_config_path = dataset.config_paths["lc_5x_stitch"]
            flipud = False
            fliplr = True
            rot90 = 1

            # For experiments >= 69, use timepoint 1 for well A/2
            timepoint_per_well = None
            exp_match = re.search(r'ops(\d+)', experiment)
            if exp_match:
                exp_num = int(exp_match.group(1))
                if exp_num >= 69:
                    timepoint_per_well = {"A/2": 1}
                    print(f"  Using timepoint 1 for well A/2 (exp >= 69)")

        if process == "pheno":
            print(
                f"Estimating shifts for 20x phase (2D recon) experiment: {experiment}"
            )
            input_store_path = dataset.store_paths["lc_20x_phase_2d_optimized"]
            output_config_path = dataset.config_paths["lc_20x_stitch"]
            flipud = False
            fliplr = False
            rot90 = 0

    # Debug-aware config path handling
    if output_config_path is not None:
        from pathlib import Path as _P

        p = _P(output_config_path)
        if debug_n_positions is not None and int(debug_n_positions) > 0:
            output_config_path = str(
                p.with_name(f"{p.stem}{debug_output_suffix}{p.suffix}")
            )
    # Build kwargs for estimate_stitch, only including timepoint_per_well if not None
    estimate_kwargs = {
        "input_store_path": input_store_path,
        "output_config_path": output_config_path,
        "flipud": flipud,
        "fliplr": fliplr,
        "rot90": rot90,
        "verbose": verbose,
        "overlap": overlap,
        "channel": channel,
        "timepoint": timepoint,
        "tile_size": tuple(tile_size),
    }
    if timepoint_per_well is not None:
        estimate_kwargs["timepoint_per_well"] = timepoint_per_well

    shifts = assemble.estimate_stitch(**estimate_kwargs)

    # Check stitch confidence after estimation
    if output_config_path is not None:
        check_stitch_confidence(output_config_path, experiment=experiment)

    return


@versioned_function("v1.0")
def estimate_and_stitch(
    experiment: Optional[str] = None,
    process: Optional[str] = None,
    input_store_path: Optional[str] = None,
    output_config_path: Optional[str] = None,
    output_store_path: Optional[str] = None,
    seg_source_store: Optional[str] = None,
    flipud: bool = False,
    fliplr: bool = False,
    rot90: int = 0,
    channel: int = 0,
    overlap: int = 150,
    use_clahe: bool = False,
    clahe_clip_limit: float = 0.02,
    debug_n_positions: int | None = None,
    debug_output_suffix: str = "_debug",
    verbose: bool = False,
    restitch_base_only: bool = False,
    tile_size: tuple = (2048, 2048),
    zarr_version: str = "0.5",
) -> None:
    """
    Estimate the globally optimal position of each tile, then assemble the stitched well
    from all individual tiles

    Can either run with the ops convention by providing an experiment name, or can run
    directly by providing the input, config, and output directories.

    Args:
        experiment (str):
            Experiment name i.e. ops{num}_{YYYMMDD}
        process (str):
            options: 'iss', 'track-2d', 'pheno-2d'
        input_store_path (str):
            Path to store of individual tiles
        output_config_path (str):
            path that the config file, recording individual tile positions will be written to
        output_store_path (str):
            Path to store of stitched well
        flipud (bool):
            Option to flip the individual tile vertically before stitching
        fliplr (bool):
            Option to flip the individual tile horizontally before stitching
        rot90 (int):
            Option to rotate the individual tile 90 degrees k times before stitching
        debug_n_positions (int):
            Grid size for debug mode (e.g., 4 = 4×4 grid = 16 positions).
            Enables debug output paths with suffix.
        debug_output_suffix (str):
            Suffix to append to output paths in debug mode (default: "_debug")
        verbose (bool):
            Print confidence scores as edges are computed (default: False)
        restitch_base_only (bool):
            If True and output store exists, only replace the base image (level 0)
            while preserving pyramids and label groups (seg/, nuclear_seg/).
            Useful for re-stitching without losing downstream segmentation results.
    """

    if experiment is None:
        if input_store_path is None or output_store_path is None:
            raise ValueError(
                "Either experiment or input/output directories must be provided."
            )
        # Debug-aware paths
        if output_config_path is not None:
            from pathlib import Path as _P

            p = _P(output_config_path)
            if debug_n_positions is not None and int(debug_n_positions) > 0:
                output_config_path = str(
                    p.with_name(f"{p.stem}{debug_output_suffix}{p.suffix}")
                )
        if output_store_path is not None:
            output_store_path = _resolve_output_path_for_debug(
                output_store_path, debug_n_positions, debug_output_suffix
            )

        # If a config already exists, allow skipping estimation and stitch directly
        config_exists = output_config_path is not None and os.path.exists(
            output_config_path
        )
        is_debug = debug_n_positions is not None and int(debug_n_positions) > 0
        # Auto-skip estimation if config exists (SLURM-compatible, no prompts)
        if config_exists and not is_debug:
            print(
                f"Shifts config exists at {output_config_path}. Skipping estimation and stitching using existing shifts."
            )
            # Rebuild from scratch: stitching rewrites the whole canvas.
            async_delete_path(output_store_path)
            # Always stitch using existing shifts (whether output existed or not)
            assemble.stitch(
                config_path=output_config_path,
                input_store_path=input_store_path,
                output_store_path=output_store_path,
                flipud=flipud,
                fliplr=fliplr,
                rot90=rot90,
                value_precision_bits=(
                    16
                    if (process or "").lower()
                    in ("iss", "track", "track-2d", "lc_5x_phase", "lc_5x_phase_2d")
                    else 32
                ),
                zarr_version=zarr_version,
                experiment=experiment,
                force_yx_scale=_force_yx_scale_for(process),
            )
            return

        # Convert debug_n_positions (grid size) to actual position count: N×N grid
        limit_positions = None
        if debug_n_positions is not None:
            limit_positions = int(debug_n_positions) ** 2
            print(
                f"DEBUG: Limiting to {debug_n_positions}×{debug_n_positions} grid = {limit_positions} positions"
            )

        shifts = assemble.estimate_stitch(
            input_store_path=input_store_path,
            output_config_path=output_config_path,
            flipud=flipud,
            fliplr=fliplr,
            rot90=rot90,
            limit_positions=limit_positions,
            channel=channel,
            overlap=overlap,
            use_clahe=use_clahe,
            clahe_clip_limit=clahe_clip_limit,
            verbose=verbose,
            tile_size=tuple(tile_size),
        )
        # Check stitch confidence after estimation
        if output_config_path is not None:
            check_stitch_confidence(output_config_path, experiment=experiment)
        # Rebuild from scratch: stitching rewrites the whole canvas.
        async_delete_path(output_store_path)
        assemble.stitch(
            config_path=output_config_path,
            input_store_path=input_store_path,
            output_store_path=output_store_path,
            flipud=flipud,
            fliplr=fliplr,
            rot90=rot90,
            zarr_version=zarr_version,
            experiment=experiment,
            force_yx_scale=_force_yx_scale_for(process),
        )

    else:
        dataset = OpsDataset(experiment)

        if process == "iss":
            print(f"Estimating shifts for ISS experiment: {experiment}")
            # Debug-aware config path
            from pathlib import Path as _P

            cfg = dataset.config_paths["iss_stitch"]
            seg_source_store = dataset.store_paths["iss_segmentation"]
            if debug_n_positions is not None and int(debug_n_positions) > 0:
                cfg = str(
                    _P(cfg).with_name(
                        f"{_P(cfg).stem}{debug_output_suffix}{_P(cfg).suffix}"
                    )
                )
            in_store = dataset.store_paths["iss_drift_corrected"]
            # If config exists, optionally skip estimation
            cfg_exists = os.path.exists(cfg)
            do_estimate = True
            if cfg_exists and not (
                debug_n_positions is not None and int(debug_n_positions) > 0
            ):
                try:
                    resp = (
                        input(
                            "Shifts config exists. Skip estimation and stitch using it? [Y/n]: "
                        )
                        .strip()
                        .lower()
                    )
                except Exception:
                    resp = ""
                do_estimate = not (resp == "" or resp in ("y", "yes"))
            if do_estimate:
                # Convert debug_n_positions (grid size) to actual position count: N×N grid
                limit_positions = None
                if debug_n_positions is not None:
                    limit_positions = int(debug_n_positions) ** 2
                    print(
                        f"DEBUG: Limiting to {debug_n_positions}×{debug_n_positions} grid = {limit_positions} positions"
                    )

                shifts = assemble.estimate_stitch(
                    input_store_path=in_store,  # iss_distortion_corrected
                    output_config_path=cfg,
                    flipud=flipud,
                    fliplr=fliplr,
                    rot90=rot90,
                    limit_positions=limit_positions,
                    verbose=verbose,
                    tile_size=tuple(tile_size),
                )

                # Check stitch confidence after estimation
                check_stitch_confidence(cfg, experiment=experiment)

            print(f"Assembling stitched ISS experiment: {experiment}")
            out_path = _resolve_output_path_for_debug(
                dataset.store_paths["iss_stitch"],
                debug_n_positions,
                debug_output_suffix,
            )
            # Rebuild from scratch: stitching rewrites the whole canvas.
            async_delete_path(out_path)
            # register_iss_cycles now requires a v3 stitched store (opens it
            # version="0.5"), so the output must be v3-sharded — slow on the
            # single-process y_bands path (~7 min/well NFS shard-commit drain).
            # Use the fast shard_stripes path instead. The ISS band tensor is
            # deep (~50: 10 cycles × 5 ch), so bound host memory: t_chunk chunks
            # the leading axis and STITCH_WORKERS_PER_GPU caps concurrent
            # per-worker band copies (raise if the step has RAM headroom, drop
            # to 1 if it OOMs). Requires stitching PR #15 for correct
            # shard_stripes shift placement + mp-spawn GPU pinning.
            os.environ.setdefault("STITCH_WORKERS_PER_GPU", "2")
            assemble.stitch(
                config_path=cfg,
                input_store_path=dataset.store_paths[
                    "iss_drift_corrected"
                ],  # iss_distortion_corrected
                output_store_path=out_path,
                flipud=flipud,
                fliplr=fliplr,
                rot90=rot90,
                value_precision_bits=16,
                blending_method="average",
                parallel_mode="shard_stripes",
                t_chunk=5,
                zarr_version=zarr_version,
                experiment=experiment,
            )

            if seg_source_store is not None:
                _attach_seg_labels_symlink(seg_source_store, out_path, label_name="nuclear_seg")

        elif process == "track-2d":
            print(f"Estimating shifts and stitching tracking 2D recon: {experiment}")
            cfg = dataset.config_paths["lc_5x_stitch"]
            seg_source_store = dataset.store_paths["lc_5x_segmentation"]
            if debug_n_positions is not None and int(debug_n_positions) > 0:
                from pathlib import Path as _P

                cfg = str(
                    _P(cfg).with_name(
                        f"{_P(cfg).stem}{debug_output_suffix}{_P(cfg).suffix}"
                    )
                )

            # For experiments >= 69, use timepoint 1 for well A/2
            timepoint_per_well = None
            exp_match = re.search(r'ops(\d+)', experiment)
            if exp_match:
                exp_num = int(exp_match.group(1))
                if exp_num >= 69:
                    timepoint_per_well = {"A/2": 1}
                    print(f"  Using timepoint 1 for well A/2 (exp >= 69)")

            # Reuse pre-estimated track shifts when available; only estimate if missing.
            # Re-estimating unconditionally would overwrite shifts that the segmentation
            # store was stitched with, causing a canvas-size mismatch between image and
            # nuclear_seg (the seg store is symlinked, not re-stitched).
            cfg_exists = os.path.exists(cfg)
            if cfg_exists and not (
                debug_n_positions is not None and int(debug_n_positions) > 0
            ):
                print(
                    f"Stitching tracking 2D recon (using existing shifts): {experiment}"
                )
            else:
                if cfg_exists:
                    print(f"DEBUG mode: re-estimating shifts for {experiment}")
                else:
                    print(f"No stitch config found, estimating shifts for {experiment}")

                # Convert debug_n_positions (grid size) to actual position count: N×N grid
                limit_positions = None
                if debug_n_positions is not None:
                    limit_positions = int(debug_n_positions) ** 2
                    print(
                        f"DEBUG: Limiting to {debug_n_positions}×{debug_n_positions} grid = {limit_positions} positions"
                    )

                shifts = assemble.estimate_stitch(
                    input_store_path=dataset.store_paths["lc_5x_phase_2d_optimized"],
                    output_config_path=cfg,
                    flipud=False,
                    fliplr=True,
                    rot90=1,
                    limit_positions=limit_positions,
                    timepoint_per_well=timepoint_per_well,
                    tile_size=tuple(tile_size),
                )

                # Check stitch confidence after estimation
                check_stitch_confidence(cfg, experiment=experiment)

            track_out_key = (
                "lc_5x_phase_2d_stitched_v3"
                if str(zarr_version) in ("0.5", "3")
                else "lc_5x_phase_2d_stitched"
            )
            out_path = _resolve_output_path_for_debug(
                dataset.store_paths[track_out_key],
                debug_n_positions,
                debug_output_suffix,
            )
            # Rebuild from scratch: stitching rewrites the whole canvas.
            async_delete_path(out_path)
            assemble.stitch(
                config_path=cfg,
                input_store_path=dataset.store_paths["lc_5x_phase_2d_optimized"],
                output_store_path=out_path,
                flipud=False,
                fliplr=True,
                rot90=1,
                value_precision_bits=16,
                zarr_version=zarr_version,
                experiment=experiment,
                force_yx_scale=_force_yx_scale_for(process),
            )

            # v3-only: materialize nuclear_seg as real labels/ data + plate channel
            # metadata (the work the removed convert_v3 step did for track). The
            # legacy v2 top-level-symlink path is gone.
            if (str(zarr_version) in ("0.5", "3")
                    and seg_source_store is not None and Path(seg_source_store).exists()):
                _materialize_nuclear_seg_to_v3_labels(
                    out_path, seg_source_store, source_label="0")
                _write_plate_channels_metadata(out_path, experiment)

        elif process == "pheno-2d":
            # Reuse pre-estimated pheno shifts when available; only estimate if missing
            cfg = dataset.config_paths["lc_20x_stitch"]
            seg_source_store = dataset.store_paths["lc_20x_segmentation_cells"]
            if debug_n_positions is not None and int(debug_n_positions) > 0:
                from pathlib import Path as _P

                cfg = str(
                    _P(cfg).with_name(
                        f"{_P(cfg).stem}{debug_output_suffix}{_P(cfg).suffix}"
                    )
                )

            cfg_exists = os.path.exists(cfg)
            if cfg_exists:
                print(
                    f"Stitching unified phenotyping tiles (using existing shifts): {experiment}"
                )
            else:
                print(
                    f"Warning: Stich not found, estimating shifts (pheno) then stitching unified phenotyping tiles: {experiment}"
                )
                # Convert debug_n_positions (grid size) to actual position count: N×N grid
                limit_positions = None
                if debug_n_positions is not None:
                    limit_positions = int(debug_n_positions) ** 2
                    print(
                        f"DEBUG: Limiting to {debug_n_positions}×{debug_n_positions} grid = {limit_positions} positions"
                    )

                _ = assemble.estimate_stitch(
                    input_store_path=dataset.store_paths[
                        "pheno_tiles_unified"
                    ],  # lc_20x_phase_2d
                    output_config_path=cfg,
                    flipud=False,
                    fliplr=False,
                    rot90=0,
                    limit_positions=limit_positions,
                    tile_size=tuple(tile_size),
                )

                # Check stitch confidence after estimation
                check_stitch_confidence(cfg, experiment=experiment)

            # Stitch the unified multi-channel tiles built by register.prepare_unified_pheno_tiles
            pheno_out_key = (
                "pheno_assembled_v3"
                if str(zarr_version) in ("0.5", "3")
                else "pheno_assembled"
            )
            out_path = _resolve_output_path_for_debug(
                dataset.store_paths[pheno_out_key],
                debug_n_positions,
                debug_output_suffix,
            )

            # restitch_base_only keeps the store and swaps level 0 in below.
            if not restitch_base_only:
                async_delete_path(out_path)

            # Viewer-optimized chunking: T,C,Z singleton; YX moderately large
            target_chunks_yx = (1024, 1024)
            try:
                from numcodecs import Blosc

                compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
            except Exception:
                compressor = None

            # Determine whether to use temp store swap (preserve pyramids/labels)
            use_temp_swap = restitch_base_only and out_path.exists()

            if use_temp_swap:
                # Stitch to temp location, then swap only level 0 back
                temp_path = out_path.parent / f"{out_path.name}_restitch_temp"
                if temp_path.exists():
                    print(f"[restitch_base_only] Removing existing temp store from previous run: {temp_path}")
                    shutil.rmtree(temp_path)
                print(f"[restitch_base_only] Stitching to temp store: {temp_path}")
                print(f"[restitch_base_only] Will preserve pyramids and labels in: {out_path}")

                assemble.stitch(
                    config_path=cfg,
                    input_store_path=dataset.store_paths["pheno_tiles_unified"],
                    output_store_path=temp_path,
                    flipud=False,
                    fliplr=False,
                    rot90=0,
                    value_precision_bits=32,
                    target_chunks_yx=target_chunks_yx,
                    compressor=compressor,
                    blending_method="edt",
                    zarr_version=zarr_version,
                    experiment=experiment,
                    force_yx_scale=_force_yx_scale_for(process),
                )

                # Swap only level 0 from temp back to original
                swapped_count = 0
                for pos_dir in temp_path.glob("*/*/*"):
                    src_0 = pos_dir / "0"
                    rel_path = pos_dir.relative_to(temp_path)
                    dst_0 = out_path / rel_path / "0"
                    if src_0.exists():
                        if dst_0.exists():
                            shutil.rmtree(dst_0)
                        dst_0.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(src_0), str(dst_0))
                        swapped_count += 1
                        print(f"[restitch_base_only] Swapped level 0 for {rel_path}")

                # Clean up temp store
                shutil.rmtree(temp_path)
                print(f"[restitch_base_only] Swapped {swapped_count} positions, removed temp store")

            else:

                assemble.stitch(
                    config_path=cfg,
                    input_store_path=dataset.store_paths["pheno_tiles_unified"],
                    output_store_path=out_path,
                    flipud=False,
                    fliplr=False,
                    rot90=0,
                    value_precision_bits=32,
                    target_chunks_yx=target_chunks_yx,
                    compressor=compressor,
                    blending_method="edt",
                    zarr_version=zarr_version,
                    experiment=experiment,
                    force_yx_scale=_force_yx_scale_for(process),
                )
            # nuclear_seg is written natively at 20x by a later step, not
            # materialized here; just (re-)apply plate channel metadata.
            if str(zarr_version) == "0.5":
                _write_plate_channels_metadata(out_path, experiment)

    return


def _materialize_nuclear_seg_to_v3_labels(
    assembled_store, seg_source_store, source_label: str
) -> None:
    """Materialize nuclear_seg into the v3 store's ``labels/nuclear_seg/0`` as a
    real sharded array, reading the (v2 OR v3) segmentation source via TensorStore.

    Replaces the old stage-symlink + convert-lift, which used the v2-only
    ``.zarray`` symlink and silently skipped the now-v3 segmentation stores.
    Level 0 only; build_pyramids fills levels 1..N.
    """
    from cyclops_process.convert.v3_common import write_seg_label_v3

    write_seg_label_v3(
        seg_source_store, assembled_store,
        label_name="nuclear_seg", source_label=source_label,
    )


def _write_plate_channels_metadata(assembled_store, experiment) -> None:
    """Write plate-level ``channels_metadata`` to a v3 store.

    convert_v3 used to write this (``build_channels_metadata``); re-apply it
    here so v3-native stitched stores don't lose it now that convert_v3 is gone.
    """
    import zarr as _zarr
    from iohub.ngff import open_ome_zarr as _open_ome_zarr
    from cyclops_process.convert.v3_metadata import build_channels_metadata

    try:
        with _open_ome_zarr(assembled_store, mode="r") as plate:
            channel_names = list(plate.channel_names)
        channels_md = build_channels_metadata(channel_names, experiment=experiment)
        if channels_md:
            root = _zarr.open(str(assembled_store), mode="r+")
            attrs = dict(root.attrs)
            attrs["channels_metadata"] = channels_md
            root.attrs.update(attrs)
            print(f"  wrote channels_metadata for {len(channels_md)} channels")
    except Exception as e:
        print(f"  WARNING: could not write channels_metadata: {e}")


def _attach_seg_labels_symlink(
    seg_source_store: str,
    assembled_store: str,
    label_name: str = "seg",
    source_label: str = "0"
) -> None:
    """
    Create per-position symlinks for segmentation labels.

    Designed to handle 4 use cases (all end up as siblings to image pyramids):
    
    Note: _discover_positions from zarr_utils uses 3-level discovery (*/*/*),
    so pos is ALWAYS A/1/0 format for all stores.

    1. ISS nuclear seg (label_name="nuclear_seg", source_label="0"):
       Source: iss_segmentation/A/1/0/0
       Dest:   iss_stitch/A/1/0/nuclear_seg/0
       (pos discovered = A/1/0)

    2. Track nuclear seg (label_name="nuclear_seg", source_label="0"):
       Source: lc_5x_segmentation/A/1/0/0
       Dest:   lc_5x_phase_2d_stitched/A/1/0/nuclear_seg/0
       (pos discovered = A/1/0)

    3. Pheno nuclear seg (label_name="nuclear_seg", source_label="20x_nuclear_seg"):
       Source: lc_20x_segmentation/A/1/0/20x_nuclear_seg
       Dest:   pheno_assembled/A/1/0/nuclear_seg/0
       (pos discovered = A/1/0)

    4. Pheno cell seg (label_name="seg", source_label="0"):
       Source: lc_20x_segmentation_cells/A/1/0/0
       Dest:   pheno_assembled/A/1/0/seg/0
       (pos discovered = A/1/0)

    Parameters
    ----------
    seg_source_store : str
        Path to source segmentation zarr store
    assembled_store : str
        Path to destination assembled zarr store
    label_name : str
        Name of label group in destination (default: "seg")
        Use "seg" for cell segmentation, "nuclear_seg" for nuclear segmentation
    source_label : str
        Name of source array in source store (default: "0")
        Use "0" for standard segmentation, "20x_nuclear_seg" for upscaled pheno nuclear seg

    Notes
    -----
    - Uses source store positions as the basis for symlinking.
    - _discover_positions() from zarr_utils uses 3-level glob, returns A/1/0 format.
    - Existing symlinks are replaced.
    """
    from pathlib import Path as _P
    from iohub.ngff import open_ome_zarr as _open
    import shutil as _shutil

    src_root = _P(seg_source_store)
    dst_root = _P(assembled_store)

    print(
        f"[attach_seg_labels_symlink] src={src_root} dst={dst_root} label={label_name}"
    )

    # Detect zarr format of destination store to know whether to write .zgroup files
    zarr_format = detect_zarr_format(dst_root)

    # Use fast position discovery (no fallback)
    # _discover_positions from zarr_utils uses 3-level glob (*/*/*), returns A/1/0 format
    src_positions = sorted(_discover_positions(src_root))

    for pos in src_positions:
        # Build source path (pos is always A/1/0):
        # - Nuclear seg: A/1/0/source_label (e.g., A/1/0/0 or A/1/0/20x_nuclear_seg)
        # - Cell seg: A/1/0/source_label (e.g., A/1/0/0)
        src_array = src_root.joinpath(_P(pos), source_label)
        
        # Build destination path (both end up as siblings to image pyramids):
        # - Nuclear seg: A/1/0/nuclear_seg/0
        # - Cell seg: A/1/0/seg/0
        # Final structure at A/1/0/:
        #   ├── 0/              (pyramid level 0 - images)
        #   ├── 1/              (pyramid level 1)
        #   ├── seg/            (cell segmentation - sibling to pyramid levels)
        #   │   └── 0/
        #   └── nuclear_seg/    (nuclear segmentation - sibling to pyramid levels)
        #       └── 0/
        dst_dir = dst_root.joinpath(_P(pos), label_name)
        
        dst_link = dst_dir.joinpath("0")

        if not src_array.exists():
            print(
                f"[attach_seg_labels_symlink] WARN: missing source array for {pos}: {src_array}"
            )
            continue

        # Check and repair missing .zarray metadata before symlinking
        zarray_path = src_array / ".zarray"
        if not zarray_path.exists():
            print(
                f"[attach_seg_labels_symlink] WARN: {pos}/{source_label}/0 missing .zarray metadata - attempting to reconstruct"
            )
            success = add_missing_zarr_metadata(src_root, str(pos), source_label, level=0)
            if not success:
                print(
                    f"[attach_seg_labels_symlink] ERROR: Could not reconstruct metadata for {pos}/{source_label}/0 - skipping symlink"
                )
                continue

        dst_dir.mkdir(parents=True, exist_ok=True)

        # Ensure .zgroup metadata exists for zarr v2 compatibility
        # (zarr v3 uses zarr.json instead, which is handled elsewhere)
        if zarr_format == 2:
            zgroup_path = dst_dir / ".zgroup"
            if not zgroup_path.exists():
                zgroup_path.write_text('{"zarr_format": 2}')

        if dst_link.is_symlink() or dst_link.exists():
            try:
                if dst_link.is_symlink() or dst_link.is_file():
                    dst_link.unlink()
                else:
                    _shutil.rmtree(dst_link)
            except Exception as e:
                print(
                    f"[attach_seg_labels_symlink] ERROR: could not remove existing {dst_link}: {e}"
                )
                continue

        try:
            os.symlink(str(src_array.resolve()), str(dst_link))
            print(f"[attach_seg_labels_symlink] linked {dst_link} -> {src_array}")
        except Exception as e:
            print(
                f"[attach_seg_labels_symlink] ERROR: failed to symlink {dst_link}: {e}"
            )


# -----------------------------
# CLI (debug-friendly) support
# -----------------------------
def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Estimate stitch parameters and assemble stitched images"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # estimate_and_stitch subcommand
    eas = subparsers.add_parser(
        "estimate_and_stitch",
        help="Estimate global shifts then stitch tiles into a single OME-Zarr store",
    )
    eas.add_argument("--experiment", type=str, help="Experiment name (OPS convention)")
    eas.add_argument(
        "--process",
        type=str,
        choices=["iss", "track-2d", "pheno-2d"],
        help="Process key for OPS paths (when --experiment is given)",
    )
    eas.add_argument("--input-store-path", type=str, help="Direct input tiles store")
    eas.add_argument("--output-config-path", type=str, help="Path to write shifts YAML")
    eas.add_argument("--output-store-path", type=str, help="Output stitched store path")
    eas.add_argument(
        "--seg-source-store",
        type=str,
        default=None,
        help="Optional: source segmentation store to symlink into assembled store",
    )
    eas.add_argument(
        "--flipud", action="store_true", help="Flip vertically before stitching"
    )
    eas.add_argument(
        "--fliplr", action="store_true", help="Flip horizontally before stitching"
    )
    eas.add_argument(
        "--rot90", type=int, default=0, help="Rotate by 90° k times before stitching"
    )
    eas.add_argument(
        "--debug-n-positions",
        type=int,
        default=None,
        help="Grid size for debug mode (e.g., 4 = 4×4 grid = 16 positions). Writes to *_debug outputs.",
    )
    eas.add_argument(
        "--debug-output-suffix",
        type=str,
        default="_debug",
        help="Suffix for debug outputs",
    )
    eas.add_argument(
        "--overlap",
        type=int,
        default=150,
        help="Expected tile overlap in pixels (default: 150)",
    )
    eas.add_argument(
        "--verbose",
        action="store_true",
        help="Print confidence scores as edges are computed during estimation",
    )
    eas.add_argument(
        "--restitch-base-only",
        action="store_true",
        help="Only replace base image (level 0), preserving pyramids and label groups (seg/, nuclear_seg/)",
    )
    eas.add_argument(
        "--zarr-version",
        type=str,
        default="0.5",
        choices=["0.4", "0.5"],
        help="OME-Zarr version: 0.5 (zarr v3 sharded, default) or 0.4 (legacy zarr v2)",
    )

    # estimate_stitch_parameters subcommand
    est = subparsers.add_parser(
        "estimate_stitch_parameters",
        help="Estimate global shifts only and write config YAML",
    )
    est.add_argument("--experiment", type=str, help="Experiment name (OPS convention)")
    est.add_argument(
        "--process",
        type=str,
        choices=["iss", "track", "pheno"],
        help="Process key for OPS paths (when --experiment is given)",
    )
    est.add_argument("--input-store-path", type=str, help="Direct input tiles store")
    est.add_argument("--output-config-path", type=str, help="Path to write shifts YAML")
    est.add_argument(
        "--flipud", action="store_true", help="Flip vertically before stitching"
    )
    est.add_argument(
        "--fliplr", action="store_true", help="Flip horizontally before stitching"
    )
    est.add_argument(
        "--rot90", type=int, default=0, help="Rotate by 90° k times before stitching"
    )
    est.add_argument(
        "--debug-n-positions",
        type=int,
        default=None,
        help="Grid size for debug mode (e.g., 4 = 4×4 grid = 16 positions). Writes to *_debug outputs.",
    )
    est.add_argument(
        "--debug-output-suffix",
        type=str,
        default="_debug",
        help="Suffix for debug outputs",
    )
    est.add_argument(
        "--overlap",
        type=int,
        default=150,
        help="Expected tile overlap in pixels (default: 150)",
    )
    est.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Channel index to use for registration (default: 0)",
    )
    est.add_argument(
        "--timepoint",
        type=int,
        default=0,
        help="Timepoint index to use for registration (default: 0)",
    )
    est.add_argument(
        "--verbose",
        action="store_true",
        help="Print confidence scores as edges are computed during estimation",
    )

    # stitch subcommand (stitch using an existing config file)
    st = subparsers.add_parser(
        "stitch",
        help="Stitch tiles into a single OME-Zarr store using an existing config",
    )
    st.add_argument("--experiment", type=str, help="Experiment name (OPS convention)")
    st.add_argument(
        "--process",
        type=str,
        choices=["iss", "lc_5x_phase", "lc_20x_phase", "lc_20x_fluorescence"],
        help="Process key for OPS paths (when --experiment is given)",
    )
    st.add_argument("--input-store-path", type=str, help="Direct input tiles store")
    st.add_argument("--output-store-path", type=str, help="Output stitched store path")
    st.add_argument("--config-path", type=str, help="Existing shifts YAML path")
    st.add_argument(
        "--flipud", action="store_true", help="Flip vertically before stitching"
    )
    st.add_argument(
        "--fliplr", action="store_true", help="Flip horizontally before stitching"
    )
    st.add_argument(
        "--rot90", type=int, default=0, help="Rotate by 90° k times before stitching"
    )
    st.add_argument(
        "--debug-n-positions",
        type=int,
        default=None,
        help="Grid size for debug mode (e.g., 4 = 4×4 grid = 16 positions). Writes to *_debug outputs.",
    )
    st.add_argument(
        "--debug-output-suffix",
        type=str,
        default="_debug",
        help="Suffix for debug outputs",
    )

    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.command == "estimate_and_stitch":
        print(
            f"[ops_stitch][estimate_and_stitch] flipud={bool(args.flipud)}, fliplr={bool(args.fliplr)}, rot90={int(args.rot90) % 4}"
        )
        estimate_and_stitch(
            experiment=args.experiment,
            process=args.process,
            input_store_path=args.input_store_path,
            output_config_path=args.output_config_path,
            output_store_path=args.output_store_path,
            seg_source_store=args.seg_source_store,
            flipud=bool(args.flipud),
            fliplr=bool(args.fliplr),
            rot90=int(args.rot90) % 4,
            overlap=args.overlap,
            debug_n_positions=args.debug_n_positions,
            debug_output_suffix=args.debug_output_suffix,
            verbose=bool(args.verbose),
            restitch_base_only=bool(args.restitch_base_only),
            zarr_version=args.zarr_version,
        )
        return

    if args.command == "estimate_stitch_parameters":
        print(
            f"[ops_stitch][estimate_stitch_parameters] flipud={bool(args.flipud)}, fliplr={bool(args.fliplr)}, rot90={int(args.rot90) % 4}"
        )
        estimate_stitch_parameters(
            experiment=args.experiment,
            process=args.process,
            input_store_path=args.input_store_path,
            output_config_path=args.output_config_path,
            flipud=bool(args.flipud),
            fliplr=bool(args.fliplr),
            rot90=int(args.rot90) % 4,
            debug_n_positions=args.debug_n_positions,
            debug_output_suffix=args.debug_output_suffix,
            verbose=bool(args.verbose),
            overlap=args.overlap,
            channel=args.channel,
            timepoint=args.timepoint,
        )
        return

    if args.command == "stitch":
        print(
            f"[ops_stitch][stitch] flipud={bool(args.flipud)}, fliplr={bool(args.fliplr)}, rot90={int(args.rot90) % 4}"
        )
        stitch(
            experiment=args.experiment,
            process=args.process,
            input_store_path=args.input_store_path,
            output_store_path=args.output_store_path,
            config_path=args.config_path,
            flipud=bool(args.flipud),
            fliplr=bool(args.fliplr),
            rot90=int(args.rot90) % 4,
            debug_n_positions=args.debug_n_positions,
            debug_output_suffix=args.debug_output_suffix,
        )
        return


if __name__ == "__main__":
    main()
    # usage: python -m cyclops_process.processes.ops_stitch estimate_and_stitch --experiment ops0033_20250429 --process pheno-2d --debug-n-positions 4 --debug-output-suffix _debug
    # usage: python -m cyclops_process.processes.ops_stitch estimate_and_stitch --experiment ops0033_20250429 --process track-2d --debug-n-positions 4 --debug-output-suffix _debug
