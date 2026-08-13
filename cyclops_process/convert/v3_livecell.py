"""
SLURM batch submission for Zarr v2 to v3 conversion.

Usage:
------
# Submit conversion for a single experiment (default: pheno mode)
python -m cyclops_process.convert.v3_livecell --experiment ops0033_20250429

# Convert specific store types using --mode
python -m cyclops_process.convert.v3_livecell --experiment ops0033_20250429 --mode pheno    # phenotyping.zarr
python -m cyclops_process.convert.v3_livecell --experiment ops0033_20250429 --mode track    # tracking_phase_2d_stitched.zarr
python -m cyclops_process.convert.v3_livecell --experiment ops0033_20250429 --mode iss      # bc_stitched_registered.zarr
python -m cyclops_process.convert.v3_livecell --experiment ops0033_20250429 --mode all      # all three stores

python -m cyclops_process.convert.v3_livecell --experiment ops0035_20250501 --mode pheno
# Run locally (without SLURM) for a single experiment
python -m cyclops_process.convert.v3_livecell --experiment ops0033_20250429 --local --mode track

# Process ALL experiments that need conversion (batch mode)
python -m cyclops_process.convert.v3_livecell --all --mode pheno

# Preview what --all mode would submit (dry run)
python -m cyclops_process.convert.v3_livecell --all --dry-run --mode track

# Enable validation (checks 3 positions by default)
python -m cyclops_process.convert.v3_livecell --experiment ops0033_20250429 --validate --mode iss

# Validate all positions (slower)
python -m cyclops_process.convert.v3_livecell --experiment ops0033_20250429 --validate --validate-all

# Validate only (without conversion) - for already converted zarr stores
python -m cyclops_process.convert.v3_livecell --experiment ops0033_20250429 --validate-only --mode track
python -m cyclops_process.convert.v3_livecell --experiment ops0033_20250429 --validate-only --validate-all --mode iss

# Convert only specific label groups (skip base images)
python -m cyclops_process.convert.v3_livecell --experiment ops0033_20250429 --only-labels seg nuclear_seg
python -m cyclops_process.convert.v3_livecell --experiment ops0036_20250505 --only-labels seg --mode pheno
python -m cyclops_process.convert.v3_livecell --experiment ops0033_20250429 --only-labels nuclear_seg --mode track

# Reconvert only specific positions (useful for fixing blank channels)
# --groups controls what to convert: 'base', 'labels', 'overlays', or 'all' (default)
python -m cyclops_process.convert.v3_livecell --experiment ops0042_20250520 --positions A/1/0 A/2/0 --force --mode pheno  # everything (default)
python -m cyclops_process.convert.v3_livecell --experiment ops0042_20250520 --positions A/1/0 A/2/0 --groups base --force --mode pheno  # base images only
python -m cyclops_process.convert.v3_livecell --experiment ops0042_20250520 --positions A/1/0 A/2/0 --groups labels --force --mode pheno  # labels only

# Resubmit specific experiment+position combinations from a file (useful for failed jobs)
python -m cyclops_process.convert.v3_livecell --resubmit-file failed_jobs.txt --mode pheno --force
# File format: one "experiment position" per line, e.g.:
#   ops0012_20250206 A/2/0
#   ops0015_20250213 A/2/0
#   ops0015_20250213 A/3/0

# Legacy: Using --source-store (deprecated, use --mode instead)
python -m cyclops_process.convert.v3_livecell --experiment ops0033_20250429 --source-store pheno_assembled
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.getcwd())

from cyclops_process.convert.v3_common import convert_to_v3, convert_position_group_to_v3, initialize_v3_store, SUBGROUP_METADATA, calculate_channel_based_shards, validate_v3_conversion
from cyclops_utils.hpc.slurm_batch_utils import (
    detect_experiments_needing_processing,
    handle_single_experiment_cli,
    submit_parallel_jobs,
    wait_for_multiple_job_arrays,
)
from cyclops_utils.data.experiment import OpsDataset
# from cyclops_utils.profiling.decorators import versioned_function
import zarr
import iohub
import pathlib
from pathlib import Path
from cyclops_process.paths import BASE_PATH

# SLURM resource configuration - single source of truth for all submission modes
# Base images: High memory/CPU for large array conversion with 96× spatial chunking
BASE_SLURM_PARAMS = {
    "timeout_min": 60,  # Base images take ~8-10min with 96× multiplier
    "mem": "250GB",  # Use ~50-60GB with 96× spatial chunking
    "cpus_per_task": 32,
    "slurm_partition": "cpu",
}

# Segmentation: Same memory as base due to large spatial dimensions
# Note: Seg arrays are same size (104K×104K) as base, just fewer channels
# Multiple label channels (seg/0-4) each requiring spatial chunking iterations
SEG_SLURM_PARAMS = {
    "timeout_min": 15,  # Increased from 5 - seg can have many labels requiring multiple spatial iterations
    "mem": "250GB",  # Need same memory as base for large arrays
    "cpus_per_task": 16,
    "slurm_partition": "cpu",
}

# Sharding ratio for single-channel label groups (seg, nuclear_seg, etc.)
# Labels are single-channel, so no benefit to multi-channel sharding.
# (1, 1, 1, 32, 32) = 32×32 chunks spatially = 16K×16K pixels per shard (~1GB for int32)
LABEL_SHARDS_RATIO = (1, 1, 1, 32, 32)


def cleanup_top_level_seg_symlinks(
    store_path: "Path",
    label_names: tuple = ("nuclear_seg", "seg"),
    quiet: bool = False,
) -> int:
    """Remove leftover top-level segmentation symlinks from a v3 store.

    The legacy symlink-then-convert flow stages seg data as a top-level
    symlink at ``<store>/<pos>/<label>/0`` so convert_v3 can read it.
    Once convert_v3 has populated the canonical labels-group copy at
    ``<store>/<pos>/labels/<label>/0..N``, the top-level symlink is
    redundant cruft (and OME-Zarr v0.5 considers it out-of-spec).

    For each position, this helper removes ``<pos>/<label>`` (the
    directory holding the staged ``/0`` symlink) **only if** the
    corresponding ``<pos>/labels/<label>/0`` exists as a real array,
    so we never leave a position with no copy of the seg data.

    Returns the number of (pos, label) entries removed.
    """
    from pathlib import Path as _P
    import shutil as _shutil

    store_path = _P(store_path)
    removed = 0

    # Iterate positions: <store>/A/<col>/<fov> (3-level discovery)
    if not store_path.exists():
        return 0

    for row_dir in sorted(store_path.iterdir()):
        if not row_dir.is_dir() or row_dir.name.startswith("."):
            continue
        for col_dir in sorted(row_dir.iterdir()):
            if not col_dir.is_dir() or col_dir.name.startswith("."):
                continue
            for fov_dir in sorted(col_dir.iterdir()):
                if not fov_dir.is_dir() or fov_dir.name.startswith("."):
                    continue

                labels_dir = fov_dir / "labels"
                for label in label_names:
                    top_level = fov_dir / label
                    if not top_level.exists():
                        continue

                    # Only safe to remove top-level if the canonical
                    # labels-group copy exists with at least level 0.
                    canonical = labels_dir / label / "0"
                    if not canonical.exists():
                        if not quiet:
                            print(
                                f"  [cleanup-symlink] keeping {top_level} "
                                f"— canonical {canonical} not present yet"
                            )
                        continue

                    try:
                        # Top-level <pos>/<label> contains the staged 0 (often
                        # a symlink) and sometimes a generated .zgroup. Drop
                        # the whole directory; the canonical copy is in labels/.
                        if top_level.is_symlink():
                            top_level.unlink()
                        else:
                            _shutil.rmtree(top_level)
                        removed += 1
                        if not quiet:
                            rel = top_level.relative_to(store_path)
                            print(f"  [cleanup-symlink] removed {rel}")
                    except Exception as e:
                        print(
                            f"  [cleanup-symlink] WARN: failed to remove "
                            f"{top_level}: {e}"
                        )

    return removed


def _get_mode_to_stores(source_zarr_version: int = 2) -> dict:
    """Map conversion mode to (source_store_key, dest_store_key).

    - source_zarr_version=2 (legacy v2→v3 path): source = unsuffixed
      store, dest = _v3 store. Used by ISS (register_iss_cycles still
      writes v2; convert_v3 transforms it to v3).
    - source_zarr_version=3 (v3-native): source = dest = _v3 store.
      Used by pheno/track because estimate_and_stitch now writes v3
      directly; convert_v3 only lifts symlinked seg data from
      top-level groups into the labels/ group within the same store.
    """
    if int(source_zarr_version) == 3:
        return {
            "pheno": ("pheno_assembled_v3", "pheno_assembled_v3"),
            "track": ("lc_5x_phase_2d_stitched_v3", "lc_5x_phase_2d_stitched_v3"),
            "iss": ("iss_stitch_registered_v3", "iss_stitch_registered_v3"),
        }
    return {
        "pheno": ("pheno_assembled", "pheno_assembled_v3"),
        "track": ("lc_5x_phase_2d_stitched", "lc_5x_phase_2d_stitched_v3"),
        "iss": ("iss_stitch_registered", "iss_stitch_registered_v3"),
    }


def detect_experiments_needing_conversion(
    force: bool = False,
    verbose: bool = True,
    mode: str = "pheno",
    source_zarr_version: int = 2,
) -> tuple[list[tuple[str, int, int, dict]], list[tuple[str, int, int, dict]]]:
    """
    Scan for experiments that need v3 conversion.

    Parameters
    ----------
    force : bool
        If True, include all experiments with valid inputs even if outputs exist
    verbose : bool
        Print progress during scan
    mode : str
        Conversion mode: 'pheno', 'track', or 'iss' (default: 'pheno')

    Returns
    -------
    tuple[list, list]
        (experiments_to_process, experiments_completed)
    """
    from pathlib import Path

    # Map mode to source/dest store keys (version-aware).
    mode_to_stores = _get_mode_to_stores(source_zarr_version)

    if mode not in mode_to_stores:
        raise ValueError(f"Invalid mode: {mode}. Must be one of: pheno, track, iss")

    source_store_key, dest_store_key = mode_to_stores[mode]

    ops_dir = Path(f"{BASE_PATH}")
    experiments = sorted([
        d.name for d in ops_dir.iterdir()
        if d.is_dir() and d.name.startswith("ops")
    ])

    experiments_to_process = []
    experiments_completed = []

    if verbose:
        print(f"\nScanning {len(experiments)} experiments...")
        print(f"Mode: {mode} (checking {source_store_key} → {dest_store_key})")
        print(f"Wells to check: []")
        print(f"{'='*60}\n")

    for experiment in experiments:
        try:
            dataset = OpsDataset(experiment)

            # Check if experiment has required input
            try:
                source_store = dataset.store_paths[source_store_key]
                if not source_store.exists():
                    continue
            except (KeyError, AttributeError):
                continue

            # Count positions in source store (fast directory scan, no metadata loading)
            try:
                # Fast: just count row/col/tile directories without opening zarr
                n_positions = 0
                for row_dir in source_store.iterdir():
                    if not row_dir.is_dir() or row_dir.name.startswith('.'):
                        continue
                    for col_dir in row_dir.iterdir():
                        if not col_dir.is_dir() or col_dir.name.startswith('.'):
                            continue
                        for tile_dir in col_dir.iterdir():
                            if tile_dir.is_dir() and not tile_dir.name.startswith('.'):
                                n_positions += 1
                if n_positions == 0:
                    n_positions = 3  # Fallback
            except Exception:
                n_positions = 3  # Default fallback

            # Check v3 store status
            try:
                v3_store = dataset.store_paths[dest_store_key]
            except (KeyError, AttributeError):
                # No v3 path configured - needs processing
                experiments_to_process.append((experiment, 0, n_positions, {}))
                continue

            # Store doesn't exist - needs processing
            if not v3_store.exists():
                experiments_to_process.append((experiment, 0, n_positions, {}))
                continue

            # Store exists - check if it has actual data
            if force:
                # Force mode - reprocess even if complete
                experiments_to_process.append((experiment, 0, n_positions, {}))
                continue

            try:
                # Check if v3 store has actual data by looking for shard files on disk
                # In Zarr v3 with sharding, data is stored in c/ (chunks) directory
                # Count positions with data
                from pathlib import Path
                v3_path = Path(v3_store)

                # Count any row/col/fov position with level-0 shard data (row-agnostic).
                positions_with_data = 0
                for row_dir in v3_path.iterdir():
                    if not row_dir.is_dir() or not row_dir.name.isalpha():
                        continue
                    for col_dir in row_dir.iterdir():
                        if not col_dir.is_dir():
                            continue
                        for fov_dir in col_dir.iterdir():
                            data_dir = fov_dir / "0" / "c"
                            if data_dir.exists() and any(data_dir.iterdir()):
                                positions_with_data += 1

                if positions_with_data >= n_positions:
                    experiments_completed.append((experiment, n_positions, n_positions, {}))
                else:
                    experiments_to_process.append((experiment, positions_with_data, n_positions, {}))

            except Exception as e:
                experiments_to_process.append((experiment, 0, n_positions, {}))

        except Exception as e:
            if verbose:
                print(f"  ✗ Error checking {experiment}: {e}")
            continue

    return experiments_to_process, experiments_completed


def get_position_group_combinations(experiment: str = None, source_store: str = "pheno_assembled",
                                   skip_overlays: bool = False, source_path: Path = None,
                                   exclude_groups: set = None, only_groups: set = None,
                                   only_positions: set = None, include_base: bool = None):
    """
    Get all position+group combinations for an experiment.

    Args:
        experiment: Experiment name (optional if source_path provided)
        source_store: Source store key (default: "pheno_assembled")
        skip_overlays: If True, skip overlay groups (grid_edges, grid_props, iss_points, iss_points_props, iss_gene_image, iss_guide_image)
        source_path: Direct path to source store (optional, overrides experiment lookup)
        exclude_groups: Set of group names to exclude (e.g., {'seg'} for track/iss stores)
        only_groups: If provided, only include these specific groups (e.g., {'seg', 'nuclear_seg'}).
                     When set, base images are skipped unless include_base=True.
        only_positions: If provided, only include these specific positions (e.g., {'A/1/0', 'A/2/0'}).
                       When set, only the specified positions are converted.
        include_base: If True, include base images. If False, skip base images.
                     If None (default), include base images only when only_groups is None.

    Returns:
        list[tuple[str, str | None]]: List of (position_key, group_name) tuples.
                                       group_name=None for base images.
    """
    # Define overlay group names to skip
    OVERLAY_GROUPS = {"grid_edges", "grid_props", "grid_overlay", "iss_points", "iss_points_props", "iss_gene_image", "iss_guide_image"}

    if exclude_groups is None:
        exclude_groups = set()

    if source_path is None:
        if experiment is None:
            raise ValueError("Either experiment or source_path must be provided")
        dataset = OpsDataset(experiment)
        source_path = dataset.store_paths.get(source_store)

    if not source_path or not source_path.exists():
        return []

    combinations = []

    # Open zarr stores to enumerate positions and groups
    src_zarr_store = zarr.open(str(source_path), mode="r")

    with iohub.open_ome_zarr(source_path, mode="r") as source_plate:
        for position_key, _ in source_plate.positions():
            # Skip positions not in the only_positions filter
            if only_positions is not None and position_key not in only_positions:
                continue

            # Determine whether to include base images
            # If include_base is explicitly set, use that; otherwise default to including when only_groups is None
            should_include_base = include_base if include_base is not None else (only_groups is None)
            if should_include_base:
                combinations.append((position_key, None))

            # Add subgroup jobs
            # Use filesystem scan to find all subgroups, since some (like overlays)
            # may not have .zgroup files and won't be found by zarr's group_keys()
            pos_path = source_path / position_key
            for item in pos_path.iterdir():
                # Skip non-directories and numeric pyramid levels
                if not item.is_dir() or item.name.isdigit():
                    continue
                group_name = item.name
                # If only_groups is specified, only include those groups
                if only_groups is not None and group_name not in only_groups:
                    continue
                # Skip overlay groups if flag is set
                if skip_overlays and group_name in OVERLAY_GROUPS:
                    continue
                # Skip excluded groups
                if group_name in exclude_groups:
                    continue
                combinations.append((position_key, group_name))

    return combinations


def calculate_shards_ratio(experiment: str = None, source_store: str = None, use_channel_sharding: bool = True,
                          source_path: Path = None):
    """Calculate shards_ratio based on channel count if channel sharding is enabled."""
    if not use_channel_sharding:
        return (1, 1, 1, 64, 64)  # Spatial-only sharding, ~1GB shards

    if source_path is None:
        if experiment is None or source_store is None:
            raise ValueError("Either source_path or (experiment + source_store) must be provided")
        dataset = OpsDataset(experiment)
        source_path = dataset.store_paths.get(source_store)

    with iohub.open_ome_zarr(source_path, mode="r") as source_plate:
        num_channels = len(source_plate.channel_names)

    return calculate_channel_based_shards(num_channels, chunks=(1, 1, 1, 512, 512))


def submit_conversion_batch(experiments: list[tuple[str, int, int, dict]], args) -> int:
    """
    Submit conversion jobs for multiple experiments with unified monitoring.

    Submits all base and seg arrays for all experiments immediately, then monitors
    them all together.

    Args:
        experiments: List of (experiment, n_done, n_total, extra_data) tuples
        args: CLI arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Collect all jobs and arrays to monitor
    all_base_jobs = []
    all_seg_jobs = []
    job_arrays_to_monitor = []
    total_submitted = 0
    total_failed = 0

    # First pass: Initialize stores (if not dry run) and collect jobs for all experiments
    if args.dry_run:
        print(f"\n{'='*60}")
        print(f"DRY RUN: Building job plan for {len(experiments)} experiments...")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print(f"Initializing stores for {len(experiments)} experiments...")
        print(f"{'='*60}\n")

    for experiment, _, _, extra_data in experiments:
        dataset = OpsDataset(experiment)

        # Determine source and dest paths based on mode or source-store
        if args.mode:
            store_pairs = get_store_pairs_for_mode(
                args.mode, dataset, source_zarr_version=args.source_zarr_version
            )
            if not store_pairs:
                print(f"  ⚠ {experiment}: No valid stores for mode '{args.mode}'")
                continue
            # For batch mode, only process first store
            store_label, source_path, dest_path = store_pairs[0]
            source_store_key = args.mode  # Use mode as identifier
        else:
            # Fallback to legacy --source-store behavior
            source_path = dataset.store_paths.get(args.source_store)
            dest_path = dataset.store_paths["pheno_assembled_v3"]
            source_store_key = args.source_store

        # Determine groups to exclude based on store type
        exclude_groups = set()
        if args.mode in ("track", "iss"):
            # Track and ISS stores should only have nuclear_seg, not seg (cell segmentation)
            exclude_groups = {"seg"}

        # Determine only_groups filter (if --only-labels specified)
        only_groups = set(args.only_labels) if hasattr(args, 'only_labels') and args.only_labels else None

        # Determine only_positions: from resubmit-file (extra_data), or from --positions
        only_positions = (extra_data or {}).get("only_positions")
        if only_positions is None:
            only_positions = set(args.positions) if hasattr(args, 'positions') and args.positions else None

        # Handle --groups flag (works with or without --positions)
        # Now supports multiple values: --groups labels overlays
        include_base = None
        groups_list = getattr(args, 'groups', ['all'])
        if isinstance(groups_list, str):
            groups_list = [groups_list]
        groups_set = set(groups_list)

        LABEL_GROUPS = {"seg", "nuclear_seg", "cell_seg"}
        OVERLAY_GROUPS = {"grid_edges", "grid_props", "grid_overlay", "iss_points", "iss_points_props", "iss_gene_image", "iss_guide_image"}

        if "all" in groups_set:
            include_base = True
            only_groups = None
        else:
            include_base = "base" in groups_set
            only_groups = set()
            if "labels" in groups_set:
                only_groups.update(LABEL_GROUPS)
            if "overlays" in groups_set:
                only_groups.update(OVERLAY_GROUPS)
            # If only base is selected, only_groups should be empty set (no subgroups)
            if not only_groups and not include_base:
                # Invalid: no groups selected
                print(f"  ⚠ {experiment}: No valid groups selected")
                continue

        # Get all position+group combinations
        combinations = get_position_group_combinations(experiment, source_store_key, skip_overlays=args.skip_overlays,
                                                      source_path=source_path, exclude_groups=exclude_groups,
                                                      only_groups=only_groups, only_positions=only_positions,
                                                      include_base=include_base)
        if not combinations:
            print(f"  ⚠ {experiment}: No positions found")
            continue

        # Skip initialization and shard calculation for dry run
        if args.dry_run:
            # Use placeholder shard ratio for dry run
            shards_ratio = (1, 1, 1, 64, 64)  # Default placeholder
        else:
            import shutil
            # Skip labels initialization if only base is selected (not labels or overlays)
            skip_labels = (groups_set == {"base"})

            if dest_path.exists():
                if "all" not in groups_set:
                    # Partial mode (base, labels, overlays, or combination): check for missing positions
                    print(f"  ⚙ {experiment}: Checking v3 store structure...")
                    missing_positions = []
                    for pos_key, _ in combinations:
                        pos_path = dest_path / pos_key
                        # Check if position has valid zarr.json metadata
                        if not (pos_path / "zarr.json").exists():
                            missing_positions.append(pos_key)

                    if missing_positions:
                        print(f"  🔧 {experiment}: Recreating {len(missing_positions)} missing positions...")
                        with iohub.open_ome_zarr(source_path, mode="r") as source_plate:
                            with iohub.open_ome_zarr(dest_path, mode="r+", channel_names=source_plate.channel_names) as dest_plate:
                                for pos_key in missing_positions:
                                    try:
                                        dest_plate.create_position(*pos_key.split("/"))
                                        print(f"    created position {pos_key}")
                                    except (FileExistsError, ValueError):
                                        pass  # Position already exists
                    else:
                        # Reuse same force/delete logic as single-experiment path: remove data so workers don't skip
                        if args.force:
                            positions_to_process = sorted(set(pos_key for pos_key, _ in combinations))
                            if "base" in groups_set:
                                print(f"  🗑 {experiment}: [--force] Removing base (pyramid levels) for {len(positions_to_process)} positions...")
                                for pos_key in positions_to_process:
                                    pos_path = dest_path / pos_key
                                    if pos_path.exists():
                                        for item in pos_path.iterdir():
                                            if item.is_dir() and item.name.isdigit():
                                                shutil.rmtree(item)
                            if "labels" in groups_set:
                                print(f"  🗑 {experiment}: [--force] Removing label groups for {len(positions_to_process)} positions...")
                                for pos_key in positions_to_process:
                                    for label_name in LABEL_GROUPS:
                                        label_path = dest_path / pos_key / "labels" / label_name
                                        if label_path.exists():
                                            shutil.rmtree(label_path)
                            if "overlays" in groups_set:
                                print(f"  🗑 {experiment}: [--force] Removing overlay groups for {len(positions_to_process)} positions...")
                                for pos_key in positions_to_process:
                                    for overlay_name in OVERLAY_GROUPS:
                                        overlay_path = dest_path / pos_key / "labels" / overlay_name
                                        if overlay_path.exists():
                                            shutil.rmtree(overlay_path)
                        else:
                            print(f"  ✓ {experiment}: All positions intact, workers will overwrite...")
                else:
                    # All mode: remove entire store for fresh initialization
                    print(f"  🗑 {experiment}: Removing existing v3 store for fresh initialization...")
                    shutil.rmtree(dest_path)

            # Initialize v3 store structure (only if store doesn't exist)
            if not dest_path.exists():
                print(f"  ⚙ {experiment}: Initializing v3 store...")
                initialize_v3_store(source_path, dest_path, overwrite=False, skip_overlays=args.skip_overlays,
                                  exclude_groups=exclude_groups, experiment=experiment, skip_labels=skip_labels)

            # Calculate shards_ratio for job building
            shards_ratio = calculate_shards_ratio(experiment, source_store_key, args.use_channel_sharding,
                                                  source_path=source_path)

        # Build jobs for this experiment
        for position_key, group_name in combinations:
            group_label = group_name or "base"
            job_name = f"{experiment}_convert_{position_key.replace('/', '_')}_{group_label}"

            # Use channel-based sharding for base images, single-channel sharding for labels
            job_shards_ratio = shards_ratio if group_name is None else LABEL_SHARDS_RATIO

            job_spec = {
                "name": job_name,
                "func": convert_position_group_to_v3,
                "kwargs": {
                    "experiment": experiment,
                    "position_key": position_key,
                    "group_name": group_name,
                    "source_path": str(source_path),
                    "dest_path": str(dest_path),
                    "chunks": (1, 1, 1, 512, 512),
                    "shards_ratio": job_shards_ratio,
                },
                "metadata": {
                    "experiment": experiment,
                    "position": position_key,
                    "group": group_label,
                },
            }

            if group_name is None:
                all_base_jobs.append(job_spec)
            else:
                all_seg_jobs.append(job_spec)

    print(f"\n{'='*60}")
    print(f"Submitting jobs to SLURM...")
    print(f"{'='*60}\n")

    # Submit all base jobs as one array
    if all_base_jobs:
        print(f"Submitting {len(all_base_jobs)} base image jobs "
              f"({BASE_SLURM_PARAMS['timeout_min']}min, {BASE_SLURM_PARAMS['mem']}, {BASE_SLURM_PARAMS['cpus_per_task']} CPUs)...")
        base_result = submit_parallel_jobs(
            jobs_to_submit=all_base_jobs,
            experiment=f"batch_convert_base_{len(experiments)}_experiments",
            slurm_params=BASE_SLURM_PARAMS,
            log_dir="slurm_convert_v3_logs/all/%j",
            manifest_prefix="convert_v3_batch_base",
            step_name="convert_iss_to_v3_batch_base",
            dry_run=args.dry_run,
            wait_for_completion=False,
            verbose=not args.quiet,
            post_completion_callback=None,
        )

        if base_result.get("success"):
            total_submitted += len(all_base_jobs)
            job_arrays_to_monitor.append({
                "submitted_jobs": base_result["submitted_jobs"],
                "base_job_id": base_result["base_job_id"],
                "label": "base",
                "slurm_params": BASE_SLURM_PARAMS,
            })

    # Submit all seg jobs as one array
    if all_seg_jobs:
        print(f"Submitting {len(all_seg_jobs)} segmentation jobs "
              f"({SEG_SLURM_PARAMS['timeout_min']}min, {SEG_SLURM_PARAMS['mem']}, {SEG_SLURM_PARAMS['cpus_per_task']} CPUs)...")
        seg_result = submit_parallel_jobs(
            jobs_to_submit=all_seg_jobs,
            experiment=f"batch_convert_seg_{len(experiments)}_experiments",
            slurm_params=SEG_SLURM_PARAMS,
            log_dir="slurm_convert_v3_logs/all/%j",
            manifest_prefix="convert_v3_batch_seg",
            step_name="convert_iss_to_v3_batch_seg",
            dry_run=args.dry_run,
            wait_for_completion=False,
            verbose=not args.quiet,
            post_completion_callback=None,
        )

        if seg_result.get("success"):
            total_submitted += len(all_seg_jobs)
            job_arrays_to_monitor.append({
                "submitted_jobs": seg_result["submitted_jobs"],
                "base_job_id": seg_result["base_job_id"],
                "label": "seg",
                "slurm_params": SEG_SLURM_PARAMS,
            })

    # Save experiment-to-job mapping in the all/ directory
    if not args.dry_run and (all_base_jobs or all_seg_jobs):
        from collections import defaultdict
        import yaml

        manifest_dir = Path("slurm_logs/slurm_convert_v3_logs/all")
        manifest_dir.mkdir(parents=True, exist_ok=True)

        # Save mapping for base jobs
        if all_base_jobs and base_result and base_result.get("success"):
            base_job_id = base_result.get("base_job_id")
            jobs_list = base_result.get("jobs", [])

            exp_to_jobs = defaultdict(list)
            for job in jobs_list:
                exp = job.get("experiment", "unknown")
                job_id = job.get("job_id", job.get("array_index", "?"))
                exp_to_jobs[exp].append(job_id)

            manifest_file = manifest_dir / f"experiment_mapping_base_{base_job_id}.yaml"
            mapping_data = {
                "slurm_array_id": base_job_id,
                "job_type": "base",
                "total_jobs": len(jobs_list),
                "experiments": {exp: {"job_ids": ids, "count": len(ids)} for exp, ids in sorted(exp_to_jobs.items())},
            }

            with open(manifest_file, "w") as f:
                yaml.dump(mapping_data, f, default_flow_style=False, sort_keys=False)
            print(f"\nBase experiment mapping saved: {manifest_file}")

        # Save mapping for seg jobs
        if all_seg_jobs and seg_result and seg_result.get("success"):
            seg_job_id = seg_result.get("base_job_id")
            jobs_list = seg_result.get("jobs", [])

            exp_to_jobs = defaultdict(list)
            for job in jobs_list:
                exp = job.get("experiment", "unknown")
                job_id = job.get("job_id", job.get("array_index", "?"))
                exp_to_jobs[exp].append(job_id)

            manifest_file = manifest_dir / f"experiment_mapping_seg_{seg_job_id}.yaml"
            mapping_data = {
                "slurm_array_id": seg_job_id,
                "job_type": "seg",
                "total_jobs": len(jobs_list),
                "experiments": {exp: {"job_ids": ids, "count": len(ids)} for exp, ids in sorted(exp_to_jobs.items())},
            }

            with open(manifest_file, "w") as f:
                yaml.dump(mapping_data, f, default_flow_style=False, sort_keys=False)
            print(f"Seg experiment mapping saved: {manifest_file}")

    if args.dry_run:
        return 0

    # Wait for all arrays together if requested
    if not args.no_wait and job_arrays_to_monitor:
        print(f"\n{'='*60}")
        print(f"Monitoring {len(job_arrays_to_monitor)} job arrays ({total_submitted} total jobs)...")
        print(f"{'='*60}\n")

        wait_results = wait_for_multiple_job_arrays(
            job_arrays=job_arrays_to_monitor,
            experiment=f"batch_convert_{len(experiments)}_experiments",
            verbose=not args.quiet,
        )

        # Aggregate failures
        if wait_results.get("array_results"):
            for array_label, array_result in wait_results["array_results"].items():
                total_failed += len(array_result.get("failed", []))

        return 0 if total_failed == 0 else 1

    return 0


# @versioned_function(version="1.0")
def get_store_pairs_for_mode(mode: str, dataset: OpsDataset,
                             source_zarr_version: int = 2) -> list[tuple[str, Path, Path]]:
    """
    Get list of (store_label, source_path, dest_path) tuples based on mode.

    Args:
        mode: One of 'pheno', 'track', 'iss', or 'all'
        dataset: OpsDataset instance
        source_zarr_version: 2 for legacy v2→v3 (source is unsuffixed); 3 for
            v3-native (source = dest = _v3 store, intra-store labels lift)

    Returns:
        List of (label, source_path, dest_path) tuples
    """
    per_mode = _get_mode_to_stores(source_zarr_version)
    if mode == "all":
        mode_map = {"all": [(label, src, dst) for label, (src, dst) in per_mode.items()]}
    else:
        mode_map = {mode: [(mode, *per_mode.get(mode, (None, None)))]}

    store_keys = mode_map.get(mode, [])
    result = []

    for label, source_key, dest_key in store_keys:
        source_path = dataset.store_paths.get(source_key)
        dest_path = dataset.store_paths.get(dest_key)

        # Skip if paths don't exist in store_paths
        if source_path is None or dest_path is None:
            print(f"  ⚠ Skipping {label}: store paths not configured")
            continue

        # Skip if source doesn't exist
        if not source_path.exists():
            print(f"  ⚠ Skipping {label}: source not found at {source_path}")
            continue

        result.append((label, source_path, dest_path))

    return result


def run_v3_conversion(
    experiment: str,
    mode: str = "all",
    force: bool = False,
    skip_overlays: bool = False,
    use_channel_sharding: bool = True,
    no_wait: bool = False,
    dry_run: bool = False,
    quiet: bool = False,
    source_zarr_version: int = 2,
    # Legacy parameters (ignored, kept for backwards compatibility)
    source_store: str = None,
    validate: bool = False,
    **kwargs,
) -> dict:
    """
    Orchestrator-friendly wrapper for Zarr v3 conversion.

    This function provides a clean interface for the pipeline orchestrator,
    avoiding the need for an argparse Namespace object.

    Args:
        experiment: Experiment name (e.g., 'ops0033_20250429')
        mode: Conversion mode - 'pheno', 'track', 'iss', or 'all' (default: 'all')
        force: If True, overwrite existing v3 store
        skip_overlays: If True, skip overlay groups (grid_edges, grid_props, etc.)
        use_channel_sharding: If True, use channel-based sharding strategy
        no_wait: If True, submit jobs and return immediately without waiting
        dry_run: If True, print what would be done without submitting
        quiet: If True, reduce verbosity

    Returns:
        dict with 'submitted' and 'failed' counts, aggregated across all stores
    """
    import argparse

    dataset = OpsDataset(experiment)

    # Get all store pairs to convert based on mode
    store_pairs = get_store_pairs_for_mode(mode, dataset, source_zarr_version=source_zarr_version)
    if not store_pairs:
        print(f"No valid stores to convert for mode '{mode}'")
        return {"submitted": 0, "failed": 0, "success": True}

    total_results = {"submitted": 0, "failed": 0}
    all_job_arrays = []  # Collect arrays across all stores for unified waiting

    # Submit all stores first (no_wait=True), then wait once at the end
    for store_label, source_path, dest_path in store_pairs:
        print(f"\n{'='*60}")
        print(f"Converting {store_label} store:")
        print(f"  Source: {source_path}")
        print(f"  Dest: {dest_path}")
        print(f"{'='*60}\n")

        # Create args namespace — always no_wait=True so we can batch-wait later
        args = argparse.Namespace(
            experiment=experiment,
            mode=store_label,  # Use individual store label, not 'all'
            force=force,
            skip_overlays=skip_overlays,
            use_channel_sharding=use_channel_sharding,
            no_wait=True,  # Submit only; wait below after all stores
            dry_run=dry_run,
            quiet=quiet,
            source_zarr_version=source_zarr_version,
            source_store="pheno_assembled",  # Legacy fallback
        )

        result = submit_conversion_job(experiment, {}, args)
        total_results["submitted"] += result.get("submitted", 0)
        total_results["failed"] += result.get("failed", 0)

        # Collect job arrays with store-specific labels
        for arr in result.get("_job_arrays", []):
            arr["label"] = f"{store_label}_{arr.get('label', '')}"
            all_job_arrays.append(arr)

    # Wait for ALL stores' job arrays together (PhaseTracker intercepts in DAG context)
    if not no_wait and not dry_run and all_job_arrays:
        wait_results = wait_for_multiple_job_arrays(
            job_arrays=all_job_arrays,
            experiment=experiment,
            verbose=not quiet,
        )

        if wait_results.get("array_results"):
            for array_label, array_result in wait_results["array_results"].items():
                total_results["failed"] += len(array_result.get("failed", []))

    return total_results


def _async_delete_store(store_path: Path) -> None:
    """Remove a zarr store without blocking the caller.

    Thin wrapper over the shared ``async_delete_path`` (rename instant, rm
    detached) so the DAG step returns immediately.
    """
    from cyclops_utils.data.filesystem import async_delete_path

    trash = async_delete_path(store_path)
    if trash is not None:
        print(f"  Renamed {store_path.name} -> {trash.name}; deleting in background")


def _ensure_iss_nuclear_seg_symlink(experiment: str, quiet: bool = False) -> None:
    """Re-attach nuclear_seg top-level symlinks on the v2 ISS register store.

    register_iss_cycles' symlink step is wrapped in a try/except that swallows
    errors, so a silent failure leaves the v2 store without nuclear_seg and
    convert_v3 has nothing to lift into ``labels/nuclear_seg``. Idempotent and
    cheap — replaces existing symlinks.
    """
    from cyclops_process.processes.ops_stitch import _attach_seg_labels_symlink

    dataset = OpsDataset(experiment)
    v2_path = dataset.store_paths.get("iss_stitch_registered")
    seg_path = dataset.store_paths.get("iss_segmentation")

    if v2_path is None or not Path(v2_path).exists():
        if not quiet:
            print(f"  [iss-symlink] skip: v2 register store missing at {v2_path}")
        return
    if seg_path is None or not Path(seg_path).exists():
        if not quiet:
            print(f"  [iss-symlink] skip: iss_segmentation missing at {seg_path}")
        return

    try:
        _attach_seg_labels_symlink(
            seg_source_store=str(seg_path),
            assembled_store=str(v2_path),
            label_name="nuclear_seg",
        )
    except Exception as e:
        print(f"  [iss-symlink] WARN: failed to attach nuclear_seg: {e}")


def convert_iss_to_v3(
    experiment: str,
    delete_v2: bool = True,
    force: bool = False,
    no_wait: bool = False,
    dry_run: bool = False,
    quiet: bool = False,
    **kwargs,
) -> dict:
    """Convert the ISS registered store to v3, then async-delete the v2 source.

    Runs right after ``merge_spots_base_calling``. ISS is independent of the
    track/pheno chains, so we convert immediately instead of waiting for the
    shared end-of-pipeline conversion. Pheno/track stitch v3-native (PR #96),
    so ISS is the only store that still needs an explicit convert.

    On a clean conversion the v2 ``iss_stitch_registered`` store is removed via
    rename-then-detached-rm to reclaim disk without blocking the DAG. All
    downstream ISS readers (metrics, napari, pyramid builds) point at the _v3
    store, so the v2 copy is dead once conversion succeeds.
    """
    result = run_v3_conversion(
        experiment,
        mode="iss",
        force=force,
        no_wait=no_wait,
        dry_run=dry_run,
        quiet=quiet,
        **kwargs,
    )

    dataset = OpsDataset(experiment)
    v2_path = dataset.store_paths.get("iss_stitch_registered")
    v3_path = dataset.store_paths.get("iss_stitch_registered_v3")

    # Write nuclear_seg directly into the v3 store (sharded) from the ISS
    # segmentation store. Replaces the v2-only staging symlink, which silently
    # skipped the now-v3 segmentation source. build_pyramids fills levels 1+.
    if not dry_run and result.get("failed", 0) == 0 and v3_path is not None and Path(v3_path).exists():
        from cyclops_process.convert.v3_common import write_seg_label_v3
        write_seg_label_v3(
            dataset.store_paths["iss_segmentation"], v3_path,
            label_name="nuclear_seg", experiment=experiment, quiet=quiet,
        )

    if dry_run or not delete_v2:
        return result

    # Only delete v2 once we're sure the v3 store exists and nothing failed.
    if result.get("failed", 0) > 0:
        print(f"  ⚠ Skipping v2 deletion: {result['failed']} conversion job(s) failed")
        return result
    if v3_path is None or not Path(v3_path).exists():
        print(f"  ⚠ Skipping v2 deletion: v3 store not found at {v3_path}")
        return result
    if v2_path is None or not Path(v2_path).exists():
        print(f"  v2 store already absent at {v2_path}; nothing to delete")
        return result

    _async_delete_store(Path(v2_path))
    return result


def submit_conversion_job(experiment: str, slurm_params: dict, args) -> dict:
    """Submit SLURM jobs for all position+group combinations in an experiment."""

    from cyclops_utils.data.filesystem import decide_overwrite_resume_skip

    dataset = OpsDataset(experiment)

    # (ISS nuclear_seg is written directly into the v3 store by write_seg_label_v3
    # in convert_iss_to_v3 — no v2 staging symlink needed here.)

    # Determine source and dest paths based on mode or source-store
    if args.mode:
        store_pairs = get_store_pairs_for_mode(
            args.mode, dataset, source_zarr_version=args.source_zarr_version
        )
        if not store_pairs:
            print(f"No valid stores to convert for mode '{args.mode}'")
            return {"submitted": 0, "failed": 0, "success": True}
        # For now, only support single store conversion per job
        # TODO: Could extend to handle multiple stores sequentially
        if len(store_pairs) > 1:
            print(f"Warning: mode '{args.mode}' would convert {len(store_pairs)} stores, but only first will be processed")
            print(f"  Stores: {[label for label, _, _ in store_pairs]}")
        store_label, source_path, dest_path = store_pairs[0]
        source_store_key = store_label  # Use label as identifier
        print(f"\nConverting {store_label} store:")
        print(f"  Source: {source_path}")
        print(f"  Dest: {dest_path}\n")
    else:
        # Fallback to legacy --source-store behavior
        source_path = dataset.store_paths.get(args.source_store)
        dest_path = dataset.store_paths["pheno_assembled_v3"]
        source_store_key = args.source_store

    # Determine groups to exclude based on store type
    exclude_groups = set()
    if args.mode in ("track", "iss"):
        # Track and ISS stores should only have nuclear_seg, not seg (cell segmentation)
        exclude_groups = {"seg"}

    # Determine only_groups filter (if --only-labels specified)
    only_groups = set(args.only_labels) if hasattr(args, 'only_labels') and args.only_labels else None

    # Determine only_positions filter (if --positions specified)
    only_positions = set(args.positions) if hasattr(args, 'positions') and args.positions else None

    # Handle --groups flag to filter what gets converted
    # Works with or without --positions, supports multiple values
    include_base = None  # Default: let get_position_group_combinations decide
    groups_list = getattr(args, 'groups', None) or ['all']
    if isinstance(groups_list, str):
        groups_list = [groups_list]
    groups_set = set(groups_list)

    LABEL_GROUPS = {"seg", "nuclear_seg", "cell_seg"}
    OVERLAY_GROUPS = {"grid_edges", "grid_props", "grid_overlay", "iss_points", "iss_points_props", "iss_gene_image", "iss_guide_image"}

    # --only-labels takes precedence over --groups default of 'all'
    _has_only_labels = hasattr(args, 'only_labels') and args.only_labels
    if _has_only_labels:
        # --only-labels explicitly specified: convert only those labels, no base images
        include_base = False
        # only_groups already set from line 847
    elif "all" in groups_set:
        include_base = True
        only_groups = None
    else:
        include_base = "base" in groups_set
        only_groups = set()
        if "labels" in groups_set:
            only_groups.update(LABEL_GROUPS)
        if "overlays" in groups_set:
            only_groups.update(OVERLAY_GROUPS)

    # Get all position+group combinations to determine expected positions
    combinations = get_position_group_combinations(experiment, source_store_key if not args.mode else None,
                                                  skip_overlays=args.skip_overlays, source_path=source_path,
                                                  exclude_groups=exclude_groups, only_groups=only_groups,
                                                  only_positions=only_positions, include_base=include_base)
    if not combinations:
        print(f"No positions found for {experiment}")
        return {"submitted": 0, "failed": 0, "success": True}

    # Extract unique position keys for structure validation
    expected_positions = sorted(set(pos_key for pos_key, _ in combinations))

    # Special handling for --groups with --force: delete based on groups_set
    # Skip this when --only-labels is specified (handled separately below)
    if groups_set and args.force and dest_path.exists() and not _has_only_labels:
        import shutil
        LABEL_GROUPS = {"seg", "nuclear_seg", "cell_seg"}
        OVERLAY_GROUPS = {"grid_edges", "grid_props", "grid_overlay", "iss_points", "iss_points_props", "iss_gene_image", "iss_guide_image"}

        # Determine which positions to process
        positions_to_process = list(only_positions) if only_positions else expected_positions

        # Delete based on what's being converted
        deleted_something = False

        if "base" in groups_set:
            print(f"\n[DELETE] Removing base images (all pyramid levels) for {len(positions_to_process)} positions...")
            for pos_key in positions_to_process:
                pos_path = dest_path / pos_key
                if pos_path.exists():
                    for item in pos_path.iterdir():
                        if item.is_dir() and item.name.isdigit():
                            print(f"  Removing {pos_key}/{item.name} (pyramid level)")
                            shutil.rmtree(item)
            deleted_something = True

        if "labels" in groups_set:
            print(f"\n[DELETE] Removing label groups for {len(positions_to_process)} positions...")
            for pos_key in positions_to_process:
                for label_name in LABEL_GROUPS:
                    label_path = dest_path / pos_key / "labels" / label_name
                    if label_path.exists():
                        print(f"  Removing {pos_key}/labels/{label_name}")
                        shutil.rmtree(label_path)
            deleted_something = True

        if "overlays" in groups_set:
            print(f"\n[DELETE] Removing overlay groups for {len(positions_to_process)} positions...")
            for pos_key in positions_to_process:
                for overlay_name in OVERLAY_GROUPS:
                    overlay_path = dest_path / pos_key / "labels" / overlay_name
                    if overlay_path.exists():
                        print(f"  Removing {pos_key}/labels/{overlay_name}")
                        shutil.rmtree(overlay_path)
            deleted_something = True

        if "all" in groups_set:
            # Delete only what we're converting - use combinations list which has exact items
            groups_per_position = {}
            for pos_key, group_name in combinations:
                if pos_key not in groups_per_position:
                    groups_per_position[pos_key] = {"base": False, "labels": set()}
                if group_name is None:
                    groups_per_position[pos_key]["base"] = True
                else:
                    groups_per_position[pos_key]["labels"].add(group_name)

            print(f"\n[DELETE] Removing data being converted for {len(groups_per_position)} positions...")
            for pos_key, groups in groups_per_position.items():
                pos_path = dest_path / pos_key
                if pos_path.exists():
                    if groups["base"]:
                        for item in pos_path.iterdir():
                            if item.is_dir() and item.name.isdigit():
                                print(f"  Removing {pos_key}/{item.name} (pyramid level)")
                                shutil.rmtree(item)
                    for label_name in groups["labels"]:
                        label_path = pos_path / "labels" / label_name
                        if label_path.exists():
                            print(f"  Removing {pos_key}/labels/{label_name}")
                            shutil.rmtree(label_path)
            deleted_something = True

        if deleted_something:
            print("Data removed. Submitting conversion jobs...\n")
        else:
            print("Submitting conversion jobs...\n")
        # Conversion workers will create missing groups as needed
        action = "resume"
    # Special handling for --only-labels with --force (legacy, without --groups)
    elif only_groups and args.force and dest_path.exists():
        import shutil
        print(f"\n[DELETE] Removing only specified label groups for re-conversion: {only_groups}")
        for pos_key in expected_positions:
            for label_name in only_groups:
                label_path = dest_path / pos_key / "labels" / label_name
                if label_path.exists():
                    print(f"  Removing {pos_key}/labels/{label_name}")
                    shutil.rmtree(label_path)
        print("Label groups removed. Submitting conversion jobs...\n")
        # Conversion workers will create missing groups as needed
        action = "resume"
    else:
        # Check store state and decide action: 'create', 'overwrite', 'resume', or 'skip'
        action = decide_overwrite_resume_skip(
            dest_path,
            is_debug=args.force,  # Force mode = debug mode (auto-overwrite)
            expected_positions=expected_positions
        )

    if action == "skip":
        print(f"Conversion skipped for {experiment}")
        return {"submitted": 0, "failed": 0, "success": True}

    # Initialize v3 store structure if needed (create or overwrite)
    # Skip initialization in dry-run mode
    if not args.dry_run:
        if action in ("create", "overwrite"):
            print(f"\nInitializing v3 store structure for {experiment}...")
            initialize_v3_store(source_path, dest_path, overwrite=(action == "overwrite"),
                              skip_overlays=args.skip_overlays, exclude_groups=exclude_groups,
                              experiment=experiment)
            print("Initialization complete. Submitting parallel conversion jobs...\n")
        elif action == "resume":
            if not only_groups:  # Don't print again if we already printed for --only-labels
                print(f"\nResuming conversion for {experiment} using existing precreated store...")
                print("Skipping initialization. Submitting parallel conversion jobs...\n")
    else:
        # Dry run mode - just print what would be done
        if only_groups and args.force:
            print(f"\nDRY RUN: Would remove and re-convert only: {only_groups}")
        elif action in ("create", "overwrite"):
            print(f"\nDRY RUN: Would initialize v3 store structure for {experiment}")
            print(f"  Action: {action}")
            print(f"  Source: {source_path}")
            print(f"  Dest: {dest_path}")
            if exclude_groups:
                print(f"  Exclude groups: {exclude_groups}")
            print()
        elif action == "resume":
            print(f"\nDRY RUN: Would resume conversion for {experiment}\n")

    if args.skip_overlays:
        print(f"Found {len(combinations)} position+group combinations to convert (skipping overlays: grid_edges, grid_props, iss_points, iss_points_props)")
    else:
        print(f"Found {len(combinations)} position+group combinations to convert")

    # Calculate shards_ratio and display sharding strategy
    shards_ratio = calculate_shards_ratio(experiment, source_store_key if not args.mode else None,
                                         args.use_channel_sharding, source_path=source_path)

    # Separate jobs by type for optimized resource allocation
    base_jobs = []
    seg_jobs = []

    for position_key, group_name in combinations:
        group_label = group_name or "base"
        job_name = f"convert_{position_key.replace('/', '_')}_{group_label}"

        # Use channel-based sharding for base images, single-channel sharding for labels
        job_shards_ratio = shards_ratio if group_name is None else LABEL_SHARDS_RATIO

        job_spec = {
            "name": job_name,
            "func": convert_position_group_to_v3,
            "kwargs": {
                "experiment": experiment,
                "position_key": position_key,
                "group_name": group_name,
                "source_path": str(source_path),
                "dest_path": str(dest_path),
                "chunks": (1, 1, 1, 512, 512),
                "shards_ratio": job_shards_ratio,
            },
            "metadata": {
                "experiment": experiment,
                "position": position_key,
                "group": group_label,
            },
        }

        if group_name is None:
            base_jobs.append(job_spec)
        else:
            seg_jobs.append(job_spec)

    results = {"submitted": 0, "failed": 0, "success": True}

    # Submit both arrays immediately without waiting
    if base_jobs:
        print(f"\nSubmitting {len(base_jobs)} base image jobs with: "
              f"{BASE_SLURM_PARAMS['timeout_min']}min, {BASE_SLURM_PARAMS['mem']}, {BASE_SLURM_PARAMS['cpus_per_task']} CPUs")
        base_result = submit_parallel_jobs(
            jobs_to_submit=base_jobs,
            experiment=experiment,
            slurm_params=BASE_SLURM_PARAMS,
            log_dir=f"slurm_convert_v3_logs/{experiment}",
            manifest_prefix="convert_v3_base",
            step_name="convert_iss_to_v3_base",
            dry_run=args.dry_run,
            wait_for_completion=False,  # Submit immediately, wait later
            verbose=not args.quiet,
            post_completion_callback=None,
        )
        if base_result.get("success"):
            results["submitted"] += len(base_jobs)

    if seg_jobs:
        print(f"\nSubmitting {len(seg_jobs)} segmentation jobs with: "
              f"{SEG_SLURM_PARAMS['timeout_min']}min, {SEG_SLURM_PARAMS['mem']}, {SEG_SLURM_PARAMS['cpus_per_task']} CPUs")
        seg_result = submit_parallel_jobs(
            jobs_to_submit=seg_jobs,
            experiment=experiment,
            slurm_params=SEG_SLURM_PARAMS,
            log_dir=f"slurm_convert_v3_logs/{experiment}",
            manifest_prefix="convert_v3_seg",
            step_name="convert_iss_to_v3_seg",
            dry_run=args.dry_run,
            wait_for_completion=False,  # Submit immediately, wait later
            verbose=not args.quiet,
            post_completion_callback=None,
        )
        if seg_result.get("success"):
            results["submitted"] += len(seg_jobs)

    # Collect job array info for callers that want to track themselves
    _job_arrays = []
    if base_jobs and base_result and base_result.get("success"):
        _job_arrays.append({
            "submitted_jobs": base_result.get("submitted_jobs", []),
            "base_job_id": base_result.get("base_job_id"),
            "label": "base",
            "slurm_params": BASE_SLURM_PARAMS,
        })
    if seg_jobs and seg_result and seg_result.get("success"):
        _job_arrays.append({
            "submitted_jobs": seg_result.get("submitted_jobs", []),
            "base_job_id": seg_result.get("base_job_id"),
            "label": "seg",
            "slurm_params": SEG_SLURM_PARAMS,
        })
    results["_job_arrays"] = _job_arrays

    # Now wait for completion if requested (after both arrays are submitted)
    if not args.no_wait and not args.dry_run and _job_arrays:
        # Wait for all arrays with unified monitoring
        wait_results = wait_for_multiple_job_arrays(
            job_arrays=_job_arrays,
            experiment=experiment,
            verbose=not args.quiet,
        )

        # Aggregate failures
        if wait_results.get("array_results"):
            for array_label, array_result in wait_results["array_results"].items():
                results["failed"] += len(array_result.get("failed", []))

        if results["failed"] > 0:
            results["success"] = False

    return results


def main():
    """CLI entry point for SLURM batch conversion submission."""
    parser = argparse.ArgumentParser(
        description="Submit Zarr v3 conversion jobs to SLURM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--experiment", "-e", type=str,
        help="Experiment name (e.g., ops0033_20250429). Required unless --all or --experiments is used.",
    )
    parser.add_argument(
        "--experiments", type=str,
        help="Range of experiment numbers to process (e.g., '6-69' for ops0006 through ops0069). "
             "Can also be comma-separated: '6,12,33' or mixed: '6-20,33,45-50'.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process all experiments that need conversion (batch submission)",
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Force re-run even if outputs exist (use with --all)",
    )
    parser.add_argument(
        "--source-store", type=str, default="pheno_assembled",
        help="Source store key (default: pheno_assembled). Deprecated: use --mode instead.",
    )
    parser.add_argument(
        "--mode", type=str, choices=["pheno", "track", "iss", "all"], default=None,
        help="Conversion mode: 'pheno' (phenotyping.zarr), 'track' (tracking_phase_2d_stitched.zarr), "
             "'iss' (bc_stitched_registered.zarr), or 'all' (convert all three). "
             "Takes precedence over --source-store.",
    )
    parser.add_argument(
        "--source-zarr-version", type=int, choices=[2, 3], default=2,
        help="Zarr version of the source store. 2 (default) = legacy v2→v3 "
             "conversion (read v2 store, write v3 store). 3 = v3-native intra-"
             "store labels lift (read and write the same _v3 store; convert "
             "copies symlinked seg data from top-level groups into the "
             "labels/ group). Pass 3 when invoking convert_v3 against pheno or "
             "track v3-native stitch outputs.",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Enable validation after conversion",
    )
    parser.add_argument(
        "--validate-all", action="store_true",
        help="Validate all positions (default: only 3 positions)",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only validate (skip conversion) - for already converted stores",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be submitted without actually submitting",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip confirmation prompt and submit immediately (use with --all)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Reduce verbosity (suppress job output)",
    )
    parser.add_argument(
        "--no-wait", action="store_true",
        help="Submit jobs and return immediately without waiting for completion",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Run locally instead of submitting to SLURM (only for single experiment)",
    )
    parser.add_argument(
        "--use-channel-sharding", action="store_true", default=True,
        help="Use channel-based sharding (groups all channels together). Default: True",
    )
    parser.add_argument(
        "--no-channel-sharding", dest="use_channel_sharding", action="store_false",
        help="Disable channel-based sharding (use spatial-only sharding)",
    )
    parser.add_argument(
        "--skip-overlays", action="store_true",
        help="Skip converting overlay groups (grid_edges, grid_props, iss_points, iss_points_props). Only convert base images and segmentation.",
    )
    parser.add_argument(
        "--only-labels", nargs="+", type=str, default=None,
        help="Only convert specific label groups (e.g., --only-labels seg nuclear_seg). "
             "When specified, base images are skipped and only the listed labels are converted. "
             "Useful for re-converting just segmentation labels without touching base images.",
    )
    parser.add_argument(
        "--positions", nargs="+", type=str, default=None,
        help="Only convert specific positions (e.g., --positions A/1/0 A/2/0). "
             "When specified, only the listed positions are converted. "
             "Use with --groups to specify what to convert within those positions.",
    )
    parser.add_argument(
        "--groups", nargs="+", type=str, default=["all"],
        help="What to convert (can specify multiple): "
             "'base' = base images (pyramid levels), "
             "'labels' = label groups (seg, nuclear_seg), "
             "'overlays' = overlay groups (grid_edges, iss_points, etc.), "
             "'all' (default) = everything. "
             "Examples: --groups labels overlays, --groups base labels",
    )
    parser.add_argument(
        "--resubmit-file", type=str, default=None,
        help="Path to file containing experiment+position combinations to resubmit. "
             "Format: one 'experiment position' per line (e.g., 'ops0012_20250206 A/2/0'). "
             "Lines starting with '#' are ignored. Use with --mode and --force.",
    )

    args = parser.parse_args()

    # Helper to parse experiment range (e.g., "6-69" or "6,12,33" or "6-20,33,45-50")
    def parse_experiment_range(range_str: str) -> set[int]:
        """Parse experiment range string into set of experiment numbers."""
        numbers = set()
        for part in range_str.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-", 1)
                numbers.update(range(int(start), int(end) + 1))
            else:
                numbers.add(int(part))
        return numbers

    # Validation
    if not args.all and not args.experiment and not args.experiments and not args.resubmit_file:
        parser.error("--experiment, --experiments, --all, or --resubmit-file is required")
    if args.resubmit_file and not args.mode:
        parser.error("--resubmit-file requires --mode (e.g., --mode pheno)")

    if args.local and (args.all or args.experiments):
        parser.error("--local can only be used with --experiment, not with --all or --experiments")

    if args.validate_only and (args.all or args.experiments):
        parser.error("--validate-only can only be used with --experiment, not with --all or --experiments")

    # Handle --resubmit-file: submit only the (experiment, position) pairs listed in the file
    if args.resubmit_file:
        resubmit_path = Path(args.resubmit_file)
        if not resubmit_path.exists():
            print(f"Error: Resubmit file not found: {resubmit_path}")
            sys.exit(1)
        lines = [ln.strip() for ln in resubmit_path.read_text().splitlines() if ln.strip() and not ln.strip().startswith("#")]
        pairs = []
        for ln in lines:
            parts = ln.split(None, 1)
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
        if not pairs:
            print(f"No valid 'experiment position' lines in {resubmit_path}")
            sys.exit(1)
        from collections import defaultdict
        by_exp = defaultdict(set)
        for exp, pos in pairs:
            by_exp[exp].add(pos)
        experiments_to_process = [(exp, 0, len(positions), {"only_positions": positions}) for exp, positions in sorted(by_exp.items())]
        print(f"\n{'='*60}")
        print(f"Resubmit mode: {len(experiments_to_process)} experiments, {len(pairs)} position(s)")
        print(f"{'='*60}\n")
        for exp, _, n_pos, _ in experiments_to_process:
            print(f"  • {exp}: {n_pos} position(s)")
        if not args.yes and not args.dry_run:
            try:
                response = input(f"\nSubmit conversion for these jobs? [y/N]: ").strip().lower()
                if response not in ['y', 'yes']:
                    print("\nCancelled by user. No jobs submitted.\n")
                    sys.exit(0)
            except (KeyboardInterrupt, EOFError):
                print("\n\nCancelled by user. No jobs submitted.\n")
                sys.exit(0)
        exit_code = submit_conversion_batch(experiments_to_process, args)
        sys.exit(exit_code)

    # Handle --all or --experiments mode (batch processing)
    if args.all or args.experiments:
        # Determine mode for batch processing
        # If mode is "all", default to "pheno" for detection (user would need to run separately for each mode)
        detection_mode = args.mode if args.mode and args.mode != "all" else "pheno"

        # Detect experiments needing conversion
        experiments_to_process, experiments_completed = detect_experiments_needing_conversion(
            force=args.force,
            verbose=not args.quiet,
            mode=detection_mode,
        )

        # Filter by experiment range if --experiments is specified
        if args.experiments:
            import re
            target_numbers = parse_experiment_range(args.experiments)
            print(f"\nFiltering to experiment numbers: {sorted(target_numbers)}")

            def get_exp_number(exp_name: str) -> int | None:
                """Extract experiment number from name like 'ops0033_20250429' -> 33"""
                match = re.match(r"ops(\d+)", exp_name)
                return int(match.group(1)) if match else None

            experiments_to_process = [
                (exp, n_done, n_total, extra)
                for exp, n_done, n_total, extra in experiments_to_process
                if get_exp_number(exp) in target_numbers
            ]
            experiments_completed = [
                (exp, n_done, n_total, extra)
                for exp, n_done, n_total, extra in experiments_completed
                if get_exp_number(exp) in target_numbers
            ]

        if not experiments_to_process:
            print(f"\n✓ All experiments are complete! No conversion jobs needed.\n")
            if not args.quiet and experiments_completed:
                print(f"Completed experiments ({len(experiments_completed)}):")
                for exp, n_done, n_total, _ in experiments_completed:
                    print(f"  ✓ {exp}")
            sys.exit(0)

        # Print summary
        print(f"\n{'='*60}")
        print(f"Batch Conversion Submission: {len(experiments_to_process)} experiments")
        print(f"{'='*60}\n")

        for exp, n_done, n_total, _ in experiments_to_process:
            status = f"{n_done}/{n_total}" if n_total > 0 else "pending"
            print(f"  • {exp}: {status}")

        if experiments_completed and not args.quiet:
            print(f"\nAlready completed ({len(experiments_completed)}):")
            for exp, n_done, n_total, _ in experiments_completed:
                print(f"  ✓ {exp}")

        # Prompt for confirmation
        if not args.yes and not args.dry_run:
            try:
                response = input(f"\nSubmit conversion for {len(experiments_to_process)} experiments? [y/N]: ").strip().lower()
                if response not in ['y', 'yes']:
                    print("\nCancelled by user. No jobs submitted.\n")
                    sys.exit(0)
            except (KeyboardInterrupt, EOFError):
                print("\n\nCancelled by user. No jobs submitted.\n")
                sys.exit(0)
            print()
        elif args.yes:
            print("\nProceeding with submission (--yes flag provided)...\n")

        # Submit all experiments as unified job arrays (one for base, one for seg)
        # This is much more efficient than submitting one experiment at a time
        exit_code = submit_conversion_batch(experiments_to_process, args)
        sys.exit(exit_code)

    # Single experiment mode
    else:
        # Determine source_store based on mode
        if args.mode:
            # Map mode to source_store key (version-aware) for validate/local operations
            _mode_pairs = _get_mode_to_stores(args.source_zarr_version)
            mode_to_source_store = {m: src for m, (src, _) in _mode_pairs.items()}
            source_store_for_op = mode_to_source_store.get(args.mode, args.source_store)
            if args.mode == "all":
                print("Error: --mode all is not supported for single experiment operations (--local or --validate-only)")
                print("Please specify one of: pheno, track, or iss")
                sys.exit(1)
        else:
            source_store_for_op = args.source_store

        # Handle validation-only mode
        if args.validate_only:
            print(f"Running validation only for {args.experiment}")
            if args.mode:
                print(f"Mode: {args.mode} (source: {source_store_for_op})")
            max_positions = None if args.validate_all else 3
            try:
                validate_v3_conversion(
                    experiment=args.experiment,
                    source_store=source_store_for_op,
                    max_positions=max_positions,
                    validation_chunk_size=(1, 1, 1, 4096, 4096)
                )
                sys.exit(0)
            except Exception as e:
                print(f"Error during validation: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)

        # Run locally if --local flag is set
        elif args.local:
            print(f"Running conversion locally for {args.experiment}")
            if args.mode:
                print(f"Mode: {args.mode} (source: {source_store_for_op})")
            validate_max_positions = None if args.validate_all else 3
            try:
                convert_to_v3(
                    experiment=args.experiment,
                    source_store=source_store_for_op,
                    chunks=(1, 1, 1, 512, 512),
                    shards_ratio=None,  # Let convert_to_v3 calculate based on channel count
                    use_channel_sharding=args.use_channel_sharding,
                    validate=args.validate,
                    validate_max_positions=validate_max_positions,
                    overwrite=None,  # Will prompt user
                )
                print(f"Conversion completed successfully for {args.experiment}")
                sys.exit(0)
            except Exception as e:
                print(f"Error during conversion: {e}")
                sys.exit(1)
        else:
            # Note: SLURM params now configured directly in submit_conversion_job()
            exit_code = handle_single_experiment_cli(
                submit_func=submit_conversion_job,
                args=args,
                slurm_params={},  # Not used anymore, params are hardcoded in submit_conversion_job
            )
            sys.exit(exit_code)


if __name__ == "__main__":
    main()
