"""Batch build-drivers for pyramid components (ISS/grid overlays, seg/organelle/
base-image/cell-painting pyramids, clims) + store-path helpers.

Lower layer imported by audit_fix.py (the audit/fix orchestrator + CLI)."""
import itertools
from pathlib import Path
from tqdm import tqdm
from ops_utils.data.experiment import OpsDataset
from ops_utils.io.zarr_utils import (
    _iter_position_paths,
    _discover_last_position_per_well,
    detect_zarr_format,
    level_has_data,
)
from ops_utils.hpc.slurm_batch_utils import (
    detect_experiments_needing_processing,
    submit_parallel_jobs,
)
from ops_utils.data.filesystem import resolve_experiment_name


# =============================================================================
# Store path resolution
# =============================================================================


def _cleanup_top_level_seg_symlinks_for_experiment(experiment: str) -> None:
    """Clean up legacy top-level seg symlinks (e.g. <store>/<pos>/nuclear_seg/0)
    in all v3 stores for an experiment, now that the canonical
    labels/<name>/ pyramid has been materialized by convert_v3 + the
    seg-pyramid build step. See cleanup_top_level_seg_symlinks() in
    convert_v3_slurm for the per-store implementation.
    """
    from cyclops_process.convert.v3_livecell import cleanup_top_level_seg_symlinks
    from ops_utils.data.experiment import OpsDataset as _Ds

    try:
        ds = _Ds(experiment)
    except Exception as e:
        print(f"  [cleanup-symlink] could not load dataset for {experiment}: {e}")
        return

    total = 0
    for store_key in (
        "pheno_assembled_v3",
        "lc_5x_phase_2d_stitched_v3",
        "iss_stitch_registered_v3",
    ):
        store_path = ds.store_paths.get(store_key)
        if store_path is None or not store_path.exists():
            continue
        print(f"\n  [cleanup-symlink] scanning {store_key} -> {store_path}")
        removed = cleanup_top_level_seg_symlinks(store_path)
        if removed:
            print(f"  [cleanup-symlink] removed {removed} entries from {store_key}")
        total += removed
    if total:
        print(f"\n  [cleanup-symlink] total: {total} top-level seg entries removed")


# =============================================================================
# Store path resolution
# =============================================================================


def _filter_to_configured_wells(experiment: str, positions: list[str]) -> list[str]:
    """Filter positions to only those in the experiment's wells_to_process config.

    Returns the filtered list, or the original list if config is unavailable.
    Prints [SKIP] for any excluded positions.
    """
    try:
        from ops_utils.data.filesystem import get_experiment_wells
        configured = set(get_experiment_wells(experiment))
        if not configured:
            return positions
        filtered = [p for p in positions if p in configured]
        if not filtered:
            return positions  # don't filter to empty
        for s in sorted(set(positions) - set(filtered)):
            print(f"    [SKIP] {s} — not in experiment wells config")
        return filtered
    except Exception:
        return positions


def _get_store_path(dataset: OpsDataset, zarr_version: int, store_type: str) -> Path:
    """
    Get store path based on zarr version and store type.

    Args:
        dataset: OpsDataset instance
        zarr_version: 2 or 3
        store_type: "pheno", "iss", or "track"

    Returns:
        Path to the store, or None if not found
    """
    if zarr_version == 3:
        store_map = {
            "pheno": "pheno_assembled_v3",
            "iss": "iss_stitch_registered_v3",
            "track": "lc_5x_phase_2d_stitched_v3",
            "bf": "bf_slices_assembled_v3",  # BF-slice titration sibling store (v3 only)
        }
    else:  # v2
        store_map = {
            "pheno": "pheno_assembled",
            "iss": "iss_stitch_registered",
            "track": "lc_5x_phase_2d_stitched",
        }
    return dataset.store_paths.get(store_map.get(store_type))


# =============================================================================
# Contrast limits (clims) building
# =============================================================================


def _build_clims(
    experiment: str,
    zarr_version: int,
    store: str = "all",
    scale_factor: int = 2,
    positions: list[str] = None,
) -> str:
    """Rebuild contrast limits for an experiment across one or more store types.

    Args:
        experiment: Experiment name
        zarr_version: 2 or 3
        store: Store type ("pheno", "iss", "track", or "all")
        scale_factor: Pyramid scale factor for per-level clim expansion
        positions: Specific positions to process (None = all)

    Returns:
        Status string
    """
    from cyclops_process.processes.pyramids.build_dask import build_clims_in_place
    import cyclops_process.processes.pyramids.build_dask as build_module

    dataset = OpsDataset(experiment)

    stores_to_process = ["pheno", "iss", "track"] if store == "all" else [store]
    results = []

    for store_name in stores_to_process:
        store_path = _get_store_path(dataset, zarr_version, store_name)

        if not store_path or not store_path.exists():
            results.append(f"{store_name}: not found")
            continue

        # Verify zarr format matches
        try:
            actual_format = detect_zarr_format(store_path)
            if actual_format != zarr_version:
                results.append(f"{store_name}: v{actual_format} (expected v{zarr_version})")
                continue
        except Exception:
            pass

        # Discover positions if not provided
        pos_list = positions
        if pos_list is None:
            pos_list = _iter_position_paths(store_path)
            if not pos_list:
                results.append(f"{store_name}: no positions found")
                continue

        # Reset the clims report flag so each store gets a table printed
        build_module.CLIMS_REPORT_PRINTED = False

        print(f"  [Clims] {store_name} ({len(pos_list)} positions)...")
        build_clims_in_place(
            source_store=store_path,
            positions=pos_list,
            scale_factor=scale_factor,
        )
        results.append(f"{store_name}: OK ({len(pos_list)} positions)")

    return f"[OK] Clims: {'; '.join(results)}"


def _get_clims_outputs(dataset: OpsDataset, wells: list[int], zarr_version: int = 3) -> list[Path]:
    """Get expected clims output paths (checks for contrast_limits_per_channel in .zattrs)."""
    store_path = _get_store_path(dataset, zarr_version=zarr_version, store_type="pheno")
    if not store_path or not store_path.exists():
        return []

    try:
        positions = _iter_position_paths(store_path)
        if positions:
            # Check for .zattrs with contrast_limits on level 0
            zattrs_path = store_path / positions[0] / "0" / ".zattrs"
            return [zattrs_path]
    except Exception:
        pass
    return []


# =============================================================================
# ISS overlay building
# =============================================================================


def build_iss_overlay(
    experiment: str,
    zarr_version: int,
    font_size: int = 24,
    point_radius: int = 3,
    guide_text_offset: int = 28,
    force: bool = False,
    kinds: tuple = ("gene", "guide"),
) -> str:
    """Build ISS overlays for an experiment.

    Creates two rendered image layers (one or both, per ``kinds``):
    - iss_gene_image: Gene names as colored text labels with dots
    - iss_guide_image: Guide sequences (first 10 bases of sgRNA) below gene names,
      with dropout/failed rounds rendered in gray

    The fix pipeline submits gene and guide as two parallel SLURM jobs because
    each kind is independent and roughly half the wall time of running both.
    """
    from cyclops_process.processes.pyramids.build_dask import build_iss_overlay_in_place

    dataset = OpsDataset(experiment)
    store_path = _get_store_path(dataset, zarr_version, "pheno")

    if not store_path or not store_path.exists():
        return f"[SKIP] ISS: Store not found for v{zarr_version}"

    # Verify zarr format matches
    try:
        actual_format = detect_zarr_format(store_path)
        if actual_format != zarr_version:
            return f"[SKIP] ISS: Store is v{actual_format}, expected v{zarr_version}"
    except Exception:
        pass

    positions = _iter_position_paths(store_path)
    if not positions:
        # Try fallback: manually iterate HCS structure for v3 zarr
        try:
            from pathlib import Path as _Path
            store_p = _Path(store_path)
            positions = []
            for row_dir in sorted(store_p.iterdir()):
                if row_dir.is_dir() and row_dir.name.isalpha():
                    for col_dir in sorted(row_dir.iterdir()):
                        if col_dir.is_dir() and col_dir.name.isdigit():
                            for fov_dir in sorted(col_dir.iterdir()):
                                if fov_dir.is_dir() and fov_dir.name.isdigit():
                                    positions.append(f"{row_dir.name}/{col_dir.name}/{fov_dir.name}")
        except Exception:
            pass
    if not positions:
        return f"[SKIP] ISS: No positions found"

    # Check for linked_results CSV
    has_linked_results = False
    for pos in positions[:3]:
        parts = [p for p in str(pos).split("/") if p]
        well_key = f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else str(pos)
        try:
            csv_path = dataset.append_well("linked_results", well_key)
            if Path(csv_path).exists():
                has_linked_results = True
                break
        except Exception:
            continue

    if not has_linked_results:
        return f"[SKIP] ISS: No linked_results CSV found"

    # Clean up legacy ISS overlays (old names: iss_points, iss_guide) and handle force mode
    import shutil
    cleaned_wrong_location = 0
    for pos in positions:
        # Legacy names to clean up (iss_points, iss_points_props, iss_guide, iss_guide_props)
        # plus new names in wrong locations
        for name in ["iss_points", "iss_points_props", "iss_guide", "iss_guide_props", "iss_gene_image", "iss_guide_image"]:
            # Determine correct and wrong paths based on zarr version
            if zarr_version == 3:
                correct_path = store_path / pos / "labels" / name
                wrong_path = store_path / pos / name
            else:
                correct_path = store_path / pos / name
                wrong_path = store_path / pos / "labels" / name

            # Always clean up wrong location
            if wrong_path.exists():
                print(f"    [Cleanup] Removing {name} from wrong location: {wrong_path}")
                shutil.rmtree(wrong_path)
                cleaned_wrong_location += 1

            # Clean up legacy names (iss_points, iss_guide) from correct location too
            if name in ["iss_points", "iss_points_props", "iss_guide", "iss_guide_props"]:
                if correct_path.exists():
                    print(f"    [Cleanup] Removing legacy {name}: {correct_path}")
                    shutil.rmtree(correct_path)
                    cleaned_wrong_location += 1

            # Force mode: also delete new names from correct location
            if force and name in ["iss_gene_image", "iss_guide_image"] and correct_path.exists():
                shutil.rmtree(correct_path)

    if cleaned_wrong_location > 0:
        print(f"    [Cleanup] Removed {cleaned_wrong_location} ISS overlay(s)")

    kinds_tuple = tuple(kinds)
    kinds_label = " + ".join(f"iss_{k}_image" for k in kinds_tuple)
    print(f"  [ISS] Building {kinds_label} for {len(positions)} positions...")
    build_iss_overlay_in_place(
        source_store=store_path,
        experiment=experiment,
        positions=positions,
        point_radius_px=point_radius,
        font_size=font_size,
        guide_text_offset=guide_text_offset,
        zarr_format=zarr_version,
        kinds=kinds_tuple,
    )

    return f"[OK] ISS: Built {kinds_label} for {len(positions)} positions"


# =============================================================================
# Grid overlay building
# =============================================================================


def _check_grid_input(dataset: OpsDataset, zarr_version: int, store_type: str = "pheno") -> bool:
    """Check if experiment has stitch config for grid building."""
    store_path = _get_store_path(dataset, zarr_version, store_type)
    if not store_path or not store_path.exists():
        return False

    # Map store type to stitch config
    stitch_config_map = {
        "pheno": "lc_20x_stitch",
        "iss": "iss_stitch",
        "track": "lc_5x_stitch",
    }
    config_key = stitch_config_map.get(store_type)
    if not config_key:
        return False

    # Check for stitch config
    try:
        stitch_config = dataset.config_paths.get(config_key)
        return stitch_config and Path(stitch_config).exists()
    except:
        return False


def _get_grid_outputs(dataset: OpsDataset, wells: list[int], zarr_version: int) -> list[Path]:
    """Get expected grid overlay output paths."""
    store_path = _get_store_path(dataset, zarr_version, "pheno")
    if not store_path or not store_path.exists():
        return []

    positions = _discover_last_position_per_well(Path(store_path))
    if positions:
        if zarr_version == 3:
            grid_path = store_path / positions[0] / "labels" / "grid_overlay"
        else:
            grid_path = store_path / positions[0] / "grid_overlay"
        return [grid_path]
    return []


def _build_grid_overlay(
    experiment: str,
    zarr_version: int,
    store_type: str = "pheno",
    font_size: int = 24,
    line_width: int = 2,
    force: bool = False,
) -> str:
    """Build grid overlay for an experiment.

    Args:
        experiment: Experiment name
        zarr_version: 2 or 3
        store_type: "pheno", "iss", or "track"
        font_size: Font size for tile ID labels
        line_width: Line width for grid boundaries
        force: Force rebuild
    """
    from cyclops_process.processes.pyramids.build_dask import build_grid_overlay_in_place
    import shutil

    dataset = OpsDataset(experiment)
    store_path = _get_store_path(dataset, zarr_version, store_type)

    if not store_path or not store_path.exists():
        return f"[SKIP] Grid ({store_type}): Store not found for v{zarr_version}"

    # Map store type to stitch config
    stitch_config_map = {
        "pheno": "lc_20x_stitch",
        "iss": "iss_stitch",
        "track": "lc_5x_stitch",
    }
    config_key = stitch_config_map.get(store_type)
    if not config_key:
        return f"[SKIP] Grid ({store_type}): Unknown store type"

    stitch_config = dataset.config_paths.get(config_key)
    if not stitch_config or not Path(stitch_config).exists():
        return f"[SKIP] Grid ({store_type}): No stitch config found at {stitch_config}"

    positions = _iter_position_paths(store_path)
    if not positions:
        # Try filesystem fallback for v3 zarr
        try:
            positions = []
            for row_dir in sorted(store_path.iterdir()):
                if row_dir.is_dir() and row_dir.name.isalpha():
                    for col_dir in sorted(row_dir.iterdir()):
                        if col_dir.is_dir() and col_dir.name.isdigit():
                            for fov_dir in sorted(col_dir.iterdir()):
                                if fov_dir.is_dir() and fov_dir.name.isdigit():
                                    positions.append(f"{row_dir.name}/{col_dir.name}/{fov_dir.name}")
        except Exception:
            pass

    if not positions:
        return f"[SKIP] Grid ({store_type}): No positions found"

    # Clean up old grid overlays (both old and new names)
    cleaned = 0
    for pos in positions:
        for name in ["grid_edges", "grid_props", "grid_overlay"]:
            if zarr_version == 3:
                path = store_path / pos / "labels" / name
            else:
                path = store_path / pos / name

            if path.exists():
                if name == "grid_overlay" and not force:
                    # Skip existing grid_overlay unless force mode
                    continue
                try:
                    shutil.rmtree(path)
                    cleaned += 1
                except Exception:
                    pass

    print(f"  [Grid ({store_type})] Building grid_overlay for {len(positions)} positions...")
    build_grid_overlay_in_place(
        source_store=store_path,
        stitch_config_path=stitch_config,
        positions=None,
        line_width_px=line_width,
        font_size=font_size,
        zarr_format=zarr_version,
        dataset=dataset,
    )

    return f"[OK] Grid ({store_type}): Built grid_overlay for {len(positions)} positions"


# =============================================================================
# Segmentation pyramid building
# =============================================================================


def _build_seg_pyramids(
    experiment: str,
    zarr_version: int,
    seg_types: list[str],
    store: str = "pheno",
    num_levels: int = 5,
    resume: bool = True,
    positions: list[str] = None,
) -> str:
    """Build segmentation pyramids for an experiment."""
    from cyclops_process.processes.pyramids.build_dask import build_seg_pyramid_only

    dataset = OpsDataset(experiment)

    stores_to_process = ["pheno", "iss", "track"] if store == "all" else [store]
    results = []

    for store_name in stores_to_process:
        store_path = _get_store_path(dataset, zarr_version, store_name)

        if not store_path or not store_path.exists():
            results.append(f"{store_name}: not found")
            continue

        # ISS and track only have nuclear_seg
        seg_types_for_store = seg_types.copy()
        if store_name in ["iss", "track"] and "seg" in seg_types_for_store:
            seg_types_for_store.remove("seg")

        if not seg_types_for_store:
            results.append(f"{store_name}: no valid seg types")
            continue

        pos_desc = f" (positions: {positions})" if positions else ""
        print(f"  [Seg Pyramids] {store_name}: {', '.join(seg_types_for_store)}{pos_desc}...")
        build_seg_pyramid_only(
            source_store=store_path,
            levels=num_levels,
            positions=positions,
            resume=resume,
            seg_types=seg_types_for_store,
        )
        results.append(f"{store_name}: OK")

    return f"[OK] Seg pyramids: {'; '.join(results)}"


# =============================================================================
# Organelle segmentation pyramid building (v3 zarr labels/ group)
# =============================================================================


def _discover_organelle_labels(experiment: str) -> tuple[list[str], Path, list[str]]:
    """
    Discover organelle segmentation labels for an experiment.

    Returns:
        Tuple of (label_names, store_path, positions) or ([], None, []) if none found.
        Only returns labels ending with '_seg' (organelle segmentation format).
    """
    dataset = OpsDataset(experiment)
    store_path = _get_store_path(dataset, zarr_version=3, store_type="pheno")

    if not store_path or not store_path.exists():
        return [], None, []

    # Verify zarr format
    try:
        actual_format = detect_zarr_format(store_path)
        if actual_format != 3:
            return [], None, []
    except Exception:
        pass

    positions = _iter_position_paths(store_path)
    if not positions:
        # Try fallback for v3 zarr
        try:
            store_p = Path(store_path)
            positions = []
            for row_dir in sorted(store_p.iterdir()):
                if row_dir.is_dir() and row_dir.name.isalpha():
                    for col_dir in sorted(row_dir.iterdir()):
                        if col_dir.is_dir() and col_dir.name.isdigit():
                            for fov_dir in sorted(col_dir.iterdir()):
                                if fov_dir.is_dir() and fov_dir.name.isdigit():
                                    positions.append(f"{row_dir.name}/{col_dir.name}/{fov_dir.name}")
        except Exception:
            pass

    if not positions:
        return [], None, []

    # Check if any labels exist
    first_pos = positions[0]
    labels_dir = store_path / first_pos / "labels"
    if not labels_dir.exists():
        return [], None, []

    # Discover organelle segmentation labels (format: {org[:5]}_{marker[:6]}_seg)
    # Only include labels ending with '_seg' to exclude iss_points, grid_edges, etc.
    try:
        label_names = sorted([
            d.name for d in labels_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
            and d.name.endswith('_seg')
            and d.name not in ('seg', 'nuclear_seg')
        ])
    except Exception:
        label_names = []

    return label_names, store_path, positions


def _build_organelle_pyramids(
    experiment: str,
    num_levels: int = 5,
    resume: bool = True,
    label_names: list[str] = None,
    label_filter: list[str] = None,
) -> str:
    """Build organelle segmentation pyramids for an experiment (v3 zarr).

    Args:
        experiment: Experiment name
        num_levels: Number of pyramid levels
        resume: Skip already-built levels
        label_names: Explicit list of labels to build (overrides discovery)
        label_filter: Filter patterns to match against discovered labels (partial match)
    """
    from cyclops_process.processes.pyramids.build_dask import build_organelle_seg_pyramids

    # Discover labels and store info
    discovered_labels, store_path, positions = _discover_organelle_labels(experiment)

    if store_path is None:
        return f"[SKIP] Organelle pyramids: v3 store not found"

    if not positions:
        return f"[SKIP] Organelle pyramids: No positions found"

    # Use provided labels or discovered labels
    if label_names is None:
        label_names = discovered_labels

    # Apply filter if provided (partial match)
    if label_filter is not None and label_names:
        filtered = []
        for label in label_names:
            for pattern in label_filter:
                if pattern in label or label in pattern:
                    filtered.append(label)
                    break
        label_names = filtered

    if not label_names:
        return f"[SKIP] Organelle pyramids: No organelle labels found in labels/"

    print(f"  [Organelle Pyramids] Building for {len(label_names)} labels: {', '.join(label_names)}")
    build_organelle_seg_pyramids(
        source_store=store_path,
        levels=num_levels,
        positions=positions,
        resume=resume,
        label_names=label_names,
    )

    return f"[OK] Organelle pyramids: Built for {len(label_names)} labels"


# =============================================================================
# Base image pyramid building (main image channels)
# =============================================================================


def _build_base_image_pyramids(
    experiment: str,
    zarr_version: int,
    store: str = "pheno",
    num_levels: int = 5,
    resume: bool = True,
    positions: list[str] = None,
) -> str:
    """Build base image pyramids (main channels, not segmentation) for an experiment.

    Args:
        experiment: Experiment name
        zarr_version: 2 or 3
        store: Store type ("pheno", "iss", "track", or "all")
        num_levels: Number of pyramid levels
        resume: Skip already-built levels
        positions: List of specific positions to process (e.g., ["A/1/0"]). None = all positions.
    """
    from cyclops_process.processes.pyramids.build_dask import build_base_image_pyramids

    dataset = OpsDataset(experiment)

    stores_to_process = ["pheno", "iss", "track"] if store == "all" else [store]
    results = []

    for store_name in stores_to_process:
        store_path = _get_store_path(dataset, zarr_version, store_name)

        if not store_path or not store_path.exists():
            results.append(f"{store_name}: not found")
            continue

        # Verify zarr format matches
        try:
            actual_format = detect_zarr_format(store_path)
            if actual_format != zarr_version:
                results.append(f"{store_name}: v{actual_format} (expected v{zarr_version})")
                continue
        except Exception:
            pass

        pos_desc = f" ({len(positions)} positions)" if positions else ""
        print(f"  [Base Image Pyramids] {store_name}{pos_desc}...")
        build_base_image_pyramids(
            source_store=store_path,
            levels=num_levels,
            positions=positions,
            resume=resume,
        )
        results.append(f"{store_name}: OK")

    return f"[OK] Base image pyramids: {'; '.join(results)}"


# =============================================================================
# Cell painting channel pyramid building (v3 zarr image channels)
# =============================================================================


def _build_cell_painting_pyramids(
    experiment: str,
    channel_start: int = 4,
    channel_end: int = 12,
    num_levels: int = 5,
    resume: bool = True,
) -> str:
    """Build pyramids for cell painting channels in phenotyping_v3.zarr.

    Cell painting channels are added to phenotyping_v3.zarr starting at channel 4
    (after Phase2D, Focus3D, nuclei_prediction, membrane_prediction).
    Default is channels 4-11 (8 cell painting channels: 4 per part × 2 parts).

    Args:
        experiment: Experiment name
        channel_start: First channel index to build pyramids for (default: 4)
        channel_end: End channel index (exclusive, default: 12)
        num_levels: Number of pyramid levels (default: 5)
        resume: Skip already-built levels (default: True)

    Returns:
        Status string
    """
    from cyclops_process.processes.pyramids.build_dask import (
        _iter_position_paths,
        _init_image_levels,
        _process_pyramid_unit,
    )
    from cyclops_process.napari.dask.dask_utils import determine_target_levels
    import dask.array as da
    import time

    dataset = OpsDataset(experiment)
    store_path = _get_store_path(dataset, zarr_version=3, store_type="pheno")

    if not store_path or not store_path.exists():
        return f"[SKIP] Cell painting pyramids: v3 store not found"

    # Verify zarr format
    try:
        actual_format = detect_zarr_format(store_path)
        if actual_format != 3:
            return f"[SKIP] Cell painting pyramids: Store is v{actual_format}, expected v3"
    except Exception:
        pass

    # Get positions
    positions = _iter_position_paths(store_path)
    if not positions:
        # Try fallback for v3 zarr
        try:
            store_p = Path(store_path)
            positions = []
            for row_dir in sorted(store_p.iterdir()):
                if row_dir.is_dir() and row_dir.name.isalpha():
                    for col_dir in sorted(row_dir.iterdir()):
                        if col_dir.is_dir() and col_dir.name.isdigit():
                            for fov_dir in sorted(col_dir.iterdir()):
                                if fov_dir.is_dir() and fov_dir.name.isdigit():
                                    positions.append(f"{row_dir.name}/{col_dir.name}/{fov_dir.name}")
        except Exception:
            pass

    if not positions:
        return f"[SKIP] Cell painting pyramids: No positions found"

    # Check how many channels exist
    first_pos = positions[0]
    try:
        arr = da.from_zarr(str(store_path), component=f"{first_pos}/0")
        num_channels = arr.shape[1]
        print(f"  Store has {num_channels} channels")

        if channel_start >= num_channels:
            return f"[SKIP] Cell painting pyramids: channel_start ({channel_start}) >= num_channels ({num_channels})"

        # Adjust channel_end if needed
        actual_end = min(channel_end, num_channels)
        channels_to_build = list(range(channel_start, actual_end))

        if not channels_to_build:
            return f"[SKIP] Cell painting pyramids: No channels in range [{channel_start}, {channel_end})"

        print(f"  Building pyramids for channels {channels_to_build}")

    except Exception as e:
        return f"[ERROR] Cell painting pyramids: Failed to read store: {e}"

    # Build pyramids for each position and channel
    factor = 2
    total_units = len(positions) * len(channels_to_build)
    built_count = 0

    print(f"  [Cell Painting Pyramids] Processing {total_units} units ({len(positions)} positions × {len(channels_to_build)} channels)...")

    for pos_path in tqdm(positions, desc="Positions"):
        for c in channels_to_build:
            # Determine which levels need building
            # Process this unit (t=0 for cell painting)
            t = 0
            targets = determine_target_levels(store_path, pos_path, num_levels, resume, t=t, c=c)

            if not targets:
                continue
            try:
                _process_pyramid_unit(store_path, pos_path, t, c, targets, factor)
                built_count += 1
            except Exception as e:
                print(f"    [WARNING] Failed to build pyramid for {pos_path} c={c}: {e}")

    return f"[OK] Cell painting pyramids: Built {built_count} units for channels {channel_start}-{actual_end-1}"


# =============================================================================
# Unified job runner
# =============================================================================


def _run_single_experiment_build(
    experiment: str,
    build_iss: bool = False,
    build_seg_pyramids: bool = False,
    build_organelle_pyramids: bool = False,
    build_cell_painting_pyramids: bool = False,
    build_grid: bool = False,
    build_base_image: bool = False,
    build_clims_flag: bool = False,
    zarr_version: int = 3,
    seg_types: list[str] = None,
    store: str = "pheno",
    force: bool = False,
    font_size: int = 24,
    point_radius: int = 3,
    guide_text_offset: int = 28,
    grid_line_width: int = 2,
    grid_font_size: int = 120,
    num_levels: int = 5,
    resume: bool = True,
    label_filter: list[str] = None,
    channel_start: int = 4,
    channel_end: int = 12,
    positions: list[str] = None,
    scale_factor: int = 2,
) -> str:
    """
    Run requested build steps for a single experiment.

    Args:
        experiment: Experiment name
        build_iss: Build ISS overlays (iss_gene_image + iss_guide_image)
        build_seg_pyramids: Build seg/nuclear_seg pyramids
        build_organelle_pyramids: Build organelle segmentation pyramids (v3 labels/)
        build_cell_painting_pyramids: Build pyramids for cell painting channels
        build_grid: Build grid overlay (tile boundaries + IDs)
        build_base_image: Build base image pyramids (main channels)
        build_clims_flag: Rebuild contrast limits for all matching stores
        zarr_version: 2 or 3
        seg_types: List of seg types for pyramids
        store: Store to process ("pheno", "iss", "track", or "all")
        force: Force rebuild
        font_size: Font size for ISS labels
        point_radius: Radius for ISS dots
        guide_text_offset: Vertical offset for guide text (default: 28)
        grid_line_width: Line width for grid boundaries (default: 2)
        grid_font_size: Font size for tile ID labels (default: 120)
        num_levels: Number of pyramid levels
        resume: Resume mode for pyramids
        label_filter: Filter patterns for organelle labels
        channel_start: Start channel for cell painting pyramids (default: 4)
        channel_end: End channel for cell painting pyramids (default: 12)
        positions: List of specific positions to process (e.g., ["A/1/0"]). None = all positions.
        scale_factor: Scale factor for per-level clim expansion (default: 2)

    Returns:
        Status string describing results
    """
    try:
        results = []
        print(f"\n[{experiment}] Starting build (v{zarr_version} zarr)...")

        if build_iss:
            result = build_iss_overlay(
                experiment=experiment,
                zarr_version=zarr_version,
                font_size=font_size,
                point_radius=point_radius,
                guide_text_offset=guide_text_offset,
                force=force,
            )
            results.append(result)

        if build_seg_pyramids:
            result = _build_seg_pyramids(
                experiment=experiment,
                zarr_version=zarr_version,
                seg_types=seg_types or ["seg", "nuclear_seg"],
                store=store,
                num_levels=num_levels,
                resume=resume,
                positions=positions,
            )
            results.append(result)

        if build_organelle_pyramids:
            result = _build_organelle_pyramids(
                experiment=experiment,
                label_filter=label_filter,
                num_levels=num_levels,
                resume=resume,
            )
            results.append(result)

        if build_cell_painting_pyramids:
            result = _build_cell_painting_pyramids(
                experiment=experiment,
                channel_start=channel_start,
                channel_end=channel_end,
                num_levels=num_levels,
                resume=resume,
            )
            results.append(result)

        if build_grid:
            result = _build_grid_overlay(
                experiment=experiment,
                zarr_version=zarr_version,
                store_type=store,
                font_size=grid_font_size,
                line_width=grid_line_width,
                force=force,
            )
            results.append(result)

        if build_base_image:
            result = _build_base_image_pyramids(
                experiment=experiment,
                zarr_version=zarr_version,
                store=store,
                num_levels=num_levels,
                resume=resume,
                positions=positions,
            )
            results.append(result)

        if build_clims_flag:
            result = _build_clims(
                experiment=experiment,
                zarr_version=zarr_version,
                store=store,
                scale_factor=scale_factor,
                positions=positions,
            )
            results.append(result)

        return f"[{experiment}] " + " | ".join(results)

    except Exception as e:
        import traceback
        return f"[ERROR] {experiment}: {e}\n{traceback.format_exc()}"


# =============================================================================
# Detection functions for slurm_batch_utils
# =============================================================================


def _check_build_input(dataset: OpsDataset, zarr_version: int) -> bool:
    """Check if experiment has required store for the given zarr version."""
    store_path = _get_store_path(dataset, zarr_version, "pheno")
    if not store_path or not store_path.exists():
        return False

    try:
        actual_format = detect_zarr_format(store_path)
        return actual_format == zarr_version
    except Exception:
        return False


def _check_iss_input(dataset: OpsDataset) -> bool:
    """Check if experiment has inputs for ISS overlay (v3 only)."""
    return _check_build_input(dataset, zarr_version=3)


def _check_v2_input(dataset: OpsDataset) -> bool:
    """Check if experiment has v2 store."""
    return _check_build_input(dataset, zarr_version=2)


def _get_iss_outputs(dataset: OpsDataset, wells: list[int]) -> list[Path]:
    """Get expected ISS output paths (checks for NEW iss_gene_image/iss_guide_image names only).

    This function checks ONLY for the new naming convention (iss_gene_image/iss_guide_image).
    Experiments with old names (iss-image, iss-props-image, iss_points, iss_guide) will be
    detected as needing processing and will be rebuilt with the new names.

    Returns the expected output paths (whether they exist or not).
    The caller checks .exists() to determine completion status.
    """
    store_path = _get_store_path(dataset, zarr_version=3, store_type="pheno")
    if not store_path or not store_path.exists():
        return []

    # Use filesystem-based discovery (avoids iohub metadata issues with v3 zarr)
    positions = _discover_last_position_per_well(Path(store_path))
    if positions:
        # Return expected ISS overlay paths with NEW naming convention in labels/ (v3 zarr)
        # Only checking for iss_gene_image and iss_guide_image
        # Old names (iss-image, iss-props-image, iss_points, iss_guide) are intentionally NOT checked
        gene_path = store_path / positions[0] / "labels" / "iss_gene_image"
        guide_path = store_path / positions[0] / "labels" / "iss_guide_image"
        return [gene_path, guide_path]
    return []  # No positions found


def _get_seg_pyramid_outputs(dataset: OpsDataset, wells: list[int], zarr_version: int = 2) -> list[Path]:
    """Get expected seg pyramid output paths."""
    store_path = _get_store_path(dataset, zarr_version=zarr_version, store_type="pheno")
    if not store_path or not store_path.exists():
        return []

    try:
        positions = _iter_position_paths(store_path)
        if positions:
            # Check for pyramid level 1 of seg
            return [store_path / positions[0] / "0" / "seg" / "1"]
    except Exception:
        pass
    return []


def _get_organelle_outputs(dataset: OpsDataset, wells: list[int]) -> list[Path]:
    """Get expected organelle segmentation pyramid output paths (v3 labels/)."""
    store_path = _get_store_path(dataset, zarr_version=3, store_type="pheno")
    if not store_path or not store_path.exists():
        return []

    try:
        positions = _iter_position_paths(store_path)
        if not positions:
            # Try fallback
            store_p = Path(store_path)
            positions = []
            for row_dir in sorted(store_p.iterdir()):
                if row_dir.is_dir() and row_dir.name.isalpha():
                    for col_dir in sorted(row_dir.iterdir()):
                        if col_dir.is_dir() and col_dir.name.isdigit():
                            for fov_dir in sorted(col_dir.iterdir()):
                                if fov_dir.is_dir() and fov_dir.name.isdigit():
                                    positions.append(f"{row_dir.name}/{col_dir.name}/{fov_dir.name}")
                                    break
                            break
                    break

        if positions:
            # Check if labels/ directory exists with any organelle labels
            labels_dir = store_path / positions[0] / "labels"
            if labels_dir.exists():
                # Return labels dir as the "output" to check
                return [labels_dir]
    except Exception:
        pass
    return []


def detect_experiments_for_build(
    build_type: str,
    zarr_version: int = 3,
    force: bool = False,
    verbose: bool = True,
) -> tuple[list[tuple], list[tuple]]:
    """
    Detect experiments needing the specified build type.

    Args:
        build_type: "iss_image", "seg_pyramids", "organelle_pyramids", "clims", etc.
        zarr_version: 2 or 3
        force: Force rebuild all
        verbose: Print progress

    Returns:
        (experiments_to_process, experiments_completed)
    """
    if build_type == "iss_image":
        input_checker = _check_iss_input
        output_checker = _get_iss_outputs
    elif build_type == "organelle_pyramids":
        # Organelle pyramids are always v3
        input_checker = _check_iss_input
        output_checker = _get_organelle_outputs
    elif build_type == "grid":
        input_checker = lambda d: _check_grid_input(d, zarr_version)
        output_checker = lambda d, w: _get_grid_outputs(d, w, zarr_version)
    elif build_type == "clims":
        input_checker = lambda d: _check_build_input(d, zarr_version)
        output_checker = lambda d, w: _get_clims_outputs(d, w, zarr_version)
    else:  # seg_pyramids
        if zarr_version == 2:
            input_checker = _check_v2_input
        else:
            input_checker = _check_iss_input
        output_checker = lambda d, w: _get_seg_pyramid_outputs(d, w, zarr_version)

    return detect_experiments_needing_processing(
        input_checker=input_checker,
        output_checker=output_checker,
        wells=[1, 2, 3],
        force=force,
        verbose=verbose,
    )


# =============================================================================
# V3 Store Audit
# =============================================================================
