"""Pyramid-build SLURM workers (job bodies executed on compute nodes).

Shared by the launcher (job submission) and audit_fix (post-fix rebuilds)."""
import json
import logging
from pathlib import Path
from typing import Optional, Sequence
from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.profiling.decorators import versioned_function


@versioned_function("v1.0")
def build_pyramid_unit_worker(
    experiment: str,
    position: str,
    t: int,
    c: int,
    source_store: str,
    levels: int = 5,
    factor: int = 2,
    resume: bool = True,
) -> dict:
    """
    SLURM-compatible worker for building pyramids for a single (pos, t, c) unit.

    This is the finest-grained parallelization - each unit can run independently.

    Parameters
    ----------
    experiment : str
        Experiment name
    position : str
        Position path (e.g., "A/1/0")
    t : int
        Time index
    c : int
        Channel index
    source_store : str
        Path to zarr store
    levels : int
        Number of pyramid levels (default: 5)
    factor : int
        Downsampling factor between levels (default: 2)
    resume : bool
        If True, skip already-built levels (default: True)

    Returns
    -------
    dict
        Result with position, t, c, status, and any errors
    """
    from pathlib import Path
    import traceback

    from cyclops_process.processes.pyramids.build_dask import _process_pyramid_unit
    from cyclops_process.napari.dask.dask_utils import determine_target_levels

    source_path = Path(source_store)

    try:
        print(f"\n{'='*60}")
        print(f"Building pyramid for unit: {position} t={t} c={c}")
        print(f"Store: {source_store}")
        print(f"{'='*60}\n")

        # Pyramid levels are pre-initialized before job submission - just determine targets
        targets = determine_target_levels(source_path, position, levels, resume, t=t, c=c)

        if not targets:
            print(f"All levels already built for {position} t={t} c={c}")
            return {
                "position": position,
                "t": t,
                "c": c,
                "status": "skipped",
                "reason": "already_built",
            }

        print(f"Building levels: {targets}")

        # Process the unit
        _process_pyramid_unit(source_path, position, t, c, targets, factor)

        print(f"\n✓ Completed pyramid build for {position} t={t} c={c}")
        return {
            "position": position,
            "t": t,
            "c": c,
            "status": "success",
            "levels_built": targets,
        }

    except Exception as e:
        error_msg = f"{e}\n{traceback.format_exc()}"
        print(f"\n✗ FAILED pyramid build for {position} t={t} c={c}: {error_msg}")
        # Re-raise so submitit's job.result() propagates the exception and the
        # orchestrator's wait code treats this unit as a real failure. Previously
        # we returned a {"status": "failed", ...} dict, which submitit packaged
        # as a "success" return value and the orchestrator marked the unit ✓.
        raise


@versioned_function("v1.0")
def build_seg_unit_worker(
    experiment: str,
    position: str,
    t: int,
    c: int,
    seg_name: str,
    source_store: str,
    levels: int = 5,
    in_labels_group: bool = False,
    preserve_dtype: bool = False,
    resume: bool = True,
) -> dict:
    """
    SLURM-compatible worker for building segmentation pyramids for a single unit.

    Parameters
    ----------
    experiment : str
        Experiment name
    position : str
        Position path (e.g., "A/1/0")
    t : int
        Time index
    c : int
        Channel index
    seg_name : str
        Segmentation name (e.g., "seg", "nuclear_seg", "mitoc_tomm20_seg")
    source_store : str
        Path to zarr store
    levels : int
        Number of pyramid levels (default: 5)
    in_labels_group : bool
        If True, seg is under pos/labels/seg_name (default: False)
    preserve_dtype : bool
        If True, preserve original dtype (default: False)
    resume : bool
        If True, skip already-built levels (default: True)

    Returns
    -------
    dict
        Result with position, seg_name, status, and any errors
    """
    from pathlib import Path
    import traceback

    from cyclops_process.processes.pyramids.build_dask import (
        _init_seg_levels,
        _process_seg_unit,
        _get_seg_dir_path,
        _get_seg_component_path,
        _is_zero_like_component,
    )

    source_path = Path(source_store)

    try:
        print(f"\n{'='*60}")
        print(f"Building {seg_name} pyramid for: {position} t={t} c={c}")
        print(f"Store: {source_store}")
        print(f"{'='*60}\n")

        # Check if base segmentation exists
        base_path = _get_seg_dir_path(source_path, position, seg_name, 0, in_labels_group)
        if not base_path.exists():
            return {
                "position": position,
                "seg_name": seg_name,
                "t": t,
                "c": c,
                "status": "skipped",
                "reason": "no_base_data",
            }

        # Seg levels are pre-initialized by the orchestrator dispatch; do NOT
        # init here — parallel workers racing on level zarr.json files cause
        # "atomic_write tmp.replace dest" FileNotFoundError on NFS.

        # Determine target levels
        targets = []
        desired_levels = set(range(1, int(levels)))
        for lvl in sorted(desired_levels):
            lvl_path = _get_seg_dir_path(source_path, position, seg_name, lvl, in_labels_group)
            lvl_component = _get_seg_component_path(position, seg_name, lvl, in_labels_group)
            exists = lvl_path.exists()
            if not resume:
                targets.append(lvl)
            else:
                if (not exists) or _is_zero_like_component(source_path, lvl_component):
                    targets.append(lvl)

        if not targets:
            print(f"All {seg_name} levels already built for {position} t={t} c={c}")
            return {
                "position": position,
                "seg_name": seg_name,
                "t": t,
                "c": c,
                "status": "skipped",
                "reason": "already_built",
            }

        print(f"Building {seg_name} levels: {targets}")

        # Process the seg unit
        _process_seg_unit(source_path, position, t, c, targets, seg_name,
                         in_labels_group=in_labels_group, preserve_dtype=preserve_dtype)

        print(f"\n✓ Completed {seg_name} pyramid build for {position} t={t} c={c}")
        return {
            "position": position,
            "seg_name": seg_name,
            "t": t,
            "c": c,
            "status": "success",
            "levels_built": targets,
        }

    except Exception as e:
        error_msg = f"{e}\n{traceback.format_exc()}"
        print(f"\n✗ FAILED {seg_name} pyramid build for {position} t={t} c={c}: {error_msg}")
        raise  # Surface failure to submitit/orchestrator (was swallowed before).


def build_overlays_worker(
    experiment: str,
    position: str,
    source_store: str,
    factor: int = 2,
    grid_line_width: int = 1,
) -> dict:
    """
    SLURM-compatible worker for building overlays (clims, grid, ISS) for a position.

    This runs after all pyramid units are complete.

    Parameters
    ----------
    experiment : str
        Experiment name
    position : str
        Position path (e.g., "A/1/0")
    source_store : str
        Path to zarr store
    factor : int
        Downsampling factor for clims scaling (default: 2)
    grid_line_width : int
        Width of grid overlay lines (default: 1)

    Returns
    -------
    dict
        Result with position, status, and any errors
    """
    from pathlib import Path
    import traceback

    from cyclops_process.processes.pyramids.build_dask import (
        build_clims_in_place,
        build_grid_overlay_in_place,
        build_iss_overlay_in_place,
    )
    from cyclops_process.napari.dask.dask_utils import _resolve_stitch_config_path

    source_path = Path(source_store)
    pos_paths = [position]

    try:
        print(f"\n{'='*60}")
        print(f"Building overlays for position: {position}")
        print(f"Store: {source_store}")
        print(f"{'='*60}\n")

        # Build contrast limits
        print("[1/3] Building contrast limits...")
        build_clims_in_place(
            source_store=source_path,
            positions=pos_paths,
            scale_factor=factor,
        )

        # Build grid overlay
        print("[2/3] Building grid overlay...")
        store_name = source_path.name.lower()
        if "iss" in store_name:
            mode = "iss"
        elif "tracking" in store_name or "lc_5x" in store_name:
            mode = "track"
        else:
            mode = "pheno"

        stitch_config_path = _resolve_stitch_config_path(experiment, mode)
        ds = OpsDataset(experiment)

        build_grid_overlay_in_place(
            source_store=source_path,
            positions=pos_paths,
            line_width_px=grid_line_width,
            stitch_config_path=stitch_config_path,
            dataset=ds,
        )

        # Build ISS overlay
        print("[3/3] Building ISS overlay...")
        build_iss_overlay_in_place(
            source_store=source_path,
            experiment=experiment,
            positions=pos_paths,
        )

        print(f"\n✓ Completed overlays for {position}")
        return {
            "position": position,
            "status": "success",
        }

    except Exception as e:
        error_msg = f"{e}\n{traceback.format_exc()}"
        print(f"\n✗ FAILED overlay build for {position}: {error_msg}")
        raise  # Surface failure to submitit/orchestrator (was swallowed before).


def reshard_level_worker(
    experiment: str,
    position: str,
    level: int,
    source_store: str,
) -> dict:
    """
    SLURM-compatible worker for resharding a single pyramid level for one position.

    After parallel pyramid building creates unsharded arrays (one file per chunk),
    this worker consolidates a single level into channel-based sharded format.

    Parameters
    ----------
    experiment : str
        Experiment name (for logging context)
    position : str
        Position path (e.g., "A/1/0")
    level : int
        Pyramid level to reshard (1, 2, 3, ...)
    source_store : str
        Path to zarr store

    Returns
    -------
    dict
        Result with position, level, status, and any error.
    """
    from pathlib import Path
    import traceback

    from cyclops_utils.io.zarr_utils import reshard_zarr_array, get_channel_dim
    from cyclops_process.convert.v3_common import calculate_channel_based_shards

    source_path = Path(source_store)

    try:
        level_path = source_path / position / str(level)
        if not level_path.exists():
            return {
                "position": position,
                "level": level,
                "status": "skipped",
            }

        num_channels = get_channel_dim(source_path, position)
        shards_ratio = calculate_channel_based_shards(
            num_channels, chunks=(1, 1, 1, 512, 512)
        )
        print(f"Resharding {position} level {level} (channels={num_channels}, shards_ratio={shards_ratio})")

        reshard_zarr_array(
            source_path=str(level_path),
            dest_path=None,  # in-place resharding
            chunks=(1, 1, 1, 512, 512),
            shards_ratio=shards_ratio,
            show_progress=False,
        )

        print(f"Completed resharding {position} level {level}")
        return {
            "position": position,
            "level": level,
            "status": "success",
        }

    except Exception as e:
        error_msg = f"{e}\n{traceback.format_exc()}"
        print(f"FAILED resharding {position} level {level}: {error_msg}")
        raise



@versioned_function("v1.0")
def build_position_pyramids_worker(
    experiment: str,
    position: str,
    source_store: str,
    levels: int = 5,
    factor: int = 2,
    grid_line_width: int = 1,
    skip_overlays: bool = False,
    resume: bool = True,
) -> dict:
    """
    SLURM-compatible worker for building pyramids for a single position.

    This function builds:
    1. Image pyramids (levels 1-4)
    2. Segmentation pyramids (seg, nuclear_seg)
    3. Contrast limits
    4. Grid overlay
    5. ISS overlay (if applicable)
    6. Organelle segmentation pyramids (if applicable)

    Parameters
    ----------
    experiment : str
        Experiment name
    position : str
        Position path (e.g., "A/1/0")
    source_store : str
        Path to zarr store
    levels : int
        Number of pyramid levels (default: 5)
    factor : int
        Downsampling factor between levels (default: 2)
    grid_line_width : int
        Width of grid overlay lines (default: 1)
    skip_overlays : bool
        If True, skip clims/grid/ISS overlays (default: False)
    resume : bool
        If True, skip already-built levels (default: True)

    Returns
    -------
    dict
        Result with position, status, and any errors
    """
    from pathlib import Path
    import traceback

    from cyclops_process.processes.pyramids.build_dask import (
        _init_image_levels,
        _init_seg_levels,
        _build_image_pyramid,
        _build_seg_pyramid,
        build_clims_in_place,
        build_grid_overlay_in_place,
        build_iss_overlay_in_place,
    )
    from cyclops_process.napari.dask.dask_utils import _resolve_stitch_config_path
    from cyclops_utils.io.zarr_utils import detect_zarr_format

    source_path = Path(source_store)
    pos_paths = [position]

    try:
        print(f"\n{'='*60}")
        print(f"Building pyramids for position: {position}")
        print(f"Store: {source_store}")
        print(f"{'='*60}\n")

        # Detect zarr format for correct path handling
        zarr_format = detect_zarr_format(source_path)
        in_labels_group = (zarr_format == 3)
        print(f"Detected zarr format: v{zarr_format}")

        # 1. Initialize and build image pyramid levels
        print("\n[1/6] Building image pyramids...")
        _init_image_levels(source_path, pos_paths, levels)
        _build_image_pyramid(source_path, pos_paths, levels, factor, resume)

        # 2. Build segmentation pyramids (seg, nuclear_seg)
        print("\n[2/6] Building segmentation pyramids...")
        seg_types = []
        for seg_type in ["seg", "nuclear_seg"]:
            if in_labels_group:
                has_seg = (source_path / position / "labels" / seg_type / "0").exists()
            else:
                has_seg = (source_path / position / seg_type / "0").exists()
            if has_seg:
                seg_types.append(seg_type)
                _init_seg_levels(source_path, pos_paths, levels, seg_type, in_labels_group=in_labels_group)
                _build_seg_pyramid(source_path, pos_paths, levels, resume, seg_type, in_labels_group=in_labels_group)
                print(f"  Built {seg_type} pyramid")

        if not seg_types:
            print("  No segmentation data found")

        # 3. Build organelle segmentation pyramids (if pheno store with labels/)
        print("\n[3/6] Building organelle segmentation pyramids...")
        labels_dir = source_path / position / "labels"
        if labels_dir.exists():
            # Discover organelle labels (exclude seg/nuclear_seg)
            organelle_labels = []
            for subdir in labels_dir.iterdir():
                if subdir.is_dir() and subdir.name not in ["seg", "nuclear_seg"]:
                    if (subdir / "0").exists():
                        organelle_labels.append(subdir.name)

            if organelle_labels:
                print(f"  Found organelle labels: {organelle_labels}")
                for label_name in organelle_labels:
                    _init_seg_levels(source_path, pos_paths, levels, label_name,
                                   in_labels_group=True, preserve_dtype=True)
                    _build_seg_pyramid(source_path, pos_paths, levels, resume, label_name,
                                     in_labels_group=True, preserve_dtype=True)
                    print(f"  Built {label_name} pyramid")
            else:
                print("  No organelle labels found")
        else:
            print("  No labels directory found")

        if not skip_overlays:
            # 4. Build contrast limits
            print("\n[4/6] Building contrast limits...")
            build_clims_in_place(
                source_store=source_path,
                positions=pos_paths,
                scale_factor=factor,
            )

            # 5. Build grid overlay
            print("\n[5/6] Building grid overlay...")
            # Determine mode from store name
            store_name = source_path.name.lower()
            if "iss" in store_name:
                mode = "iss"
            elif "tracking" in store_name or "lc_5x" in store_name:
                mode = "track"
            else:
                mode = "pheno"

            stitch_config_path = _resolve_stitch_config_path(experiment, mode)
            ds = OpsDataset(experiment)

            build_grid_overlay_in_place(
                source_store=source_path,
                positions=pos_paths,
                line_width_px=grid_line_width,
                stitch_config_path=stitch_config_path,
                dataset=ds,
            )

            # 6. Build ISS overlay
            print("\n[6/6] Building ISS overlay...")
            build_iss_overlay_in_place(
                source_store=source_path,
                experiment=experiment,
                positions=pos_paths,
            )
        else:
            print("\n[4-6/6] Skipping overlays (--skip-overlays)")

        print(f"\n✓ Completed pyramid build for {position}")
        return {
            "position": position,
            "status": "success",
            "seg_types": seg_types,
        }

    except Exception as e:
        error_msg = f"{e}\n{traceback.format_exc()}"
        print(f"\n✗ FAILED pyramid build for {position}: {error_msg}")
        raise  # Surface failure to submitit/orchestrator (was swallowed before).


