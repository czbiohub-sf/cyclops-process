#!/usr/bin/env python
"""
Batch symlink nuclear segmentations to target zarr stores.

Symlinks nuclear segmentation from lc_20x_segmentation to pheno, iss, or track zarrs.
Does NOT perform upscaling - for upscaling to pheno, use batch_upscale_nuclear_seg.py instead.

python -m cyclops_process.utils.batch.batch_symlink_nuclear_seg --experiments ops0036_20250505 --symlink-target pheno --force
"""
import sys
import os

sys.path.insert(0, os.getcwd())

import argparse
from pathlib import Path
from tqdm import tqdm

from ops_utils.data.experiment import OpsDataset
from cyclops_process.paths import BASE_PATH


def _discover_positions(store_path: Path):
    """Discover position folders in a zarr store."""
    positions = []
    for row_dir in store_path.glob("*"):
        if row_dir.is_dir() and row_dir.name not in [".zattrs", ".zgroup"]:
            for col_dir in row_dir.glob("*"):
                if col_dir.is_dir() and col_dir.name not in [".zattrs", ".zgroup"]:
                    positions.append(f"{row_dir.name}/{col_dir.name}")
    return positions


def symlink_nuclear_seg(experiment: str, symlink_target: str = "iss",
                        zarr_version: int = 3):
    """
    Symlink nuclear segmentation from respective source to target zarr.

    Args:
        experiment: Name of the experiment
        symlink_target: Target zarr to symlink into ("pheno", "iss", or "track")
        zarr_version: 2 for legacy v2 destination, 3 for v3-native destination
            (pheno/track v3-native stitch writes phenotyping_v3.zarr /
             tracking_phase_2d_stitched_v3.zarr; symlinks must land there).

    Symlinks (v3-native, default):
        - pheno: uses lc_20x_segmentation -> pheno_assembled_v3
        - iss:   uses iss_segmentation    -> iss_stitch_registered_v3
        - track: uses lc_5x_segmentation  -> lc_5x_phase_2d_stitched_v3
    """
    dataset = OpsDataset(experiment)

    # Map target to source and destination store paths.
    # In v3-native mode (zarr_version=3) the dest is the _v3 path the
    # stitch step wrote to; in legacy v2 mode the dest is the unsuffixed
    # path that older pipeline runs produced.
    v3 = int(zarr_version) == 3
    mapping = {
        "pheno": {
            "source": "lc_20x_segmentation",
            "dest": "pheno_assembled_v3" if v3 else "pheno_assembled",
        },
        "iss": {
            "source": "iss_segmentation",
            "dest": "iss_stitch_registered_v3" if v3 else "iss_stitch_registered",
        },
        "track": {
            "source": "lc_5x_segmentation",
            "dest": "lc_5x_phase_2d_stitched_v3" if v3 else "lc_5x_phase_2d_stitched",
        },
    }

    if symlink_target not in mapping:
        raise ValueError(
            f"Invalid symlink_target: {symlink_target}. Must be one of {list(mapping.keys())}"
        )

    source_store_key = mapping[symlink_target]["source"]
    dest_store_key = mapping[symlink_target]["dest"]

    nuclear_seg_path = dataset.store_paths[source_store_key]
    dest_store_path = dataset.store_paths[dest_store_key]

    if not nuclear_seg_path.exists():
        raise FileNotFoundError(
            f"Source nuclear seg path does not exist: {nuclear_seg_path}"
        )

    if not dest_store_path.exists():
        raise FileNotFoundError(
            f"Destination store path does not exist: {dest_store_path}"
        )

    positions = _discover_positions(nuclear_seg_path)

    print(f"Symlinking from {source_store_key} to {dest_store_key}")

    for pos in tqdm(positions, desc=f"Symlinking nuclear seg to {symlink_target}"):
        # Determine source path based on target type
        # All sources are at pos/0/0 (the actual array zarr)
        # ISS has pos/0/0 (real) and pos/0/1 (dilated), track only has pos/0/0
        if symlink_target == "pheno":
            # For pheno: REQUIRE upscaled 20x version (do not fallback to 5x)
            source_20x_seg = nuclear_seg_path / pos / "0" / "20x_nuclear_seg"
            if source_20x_seg.exists():
                src_path = source_20x_seg
                print(f"  Using upscaled 20x seg for {pos}")
            else:
                print(
                    f"  ERROR: 20x_nuclear_seg not found for {pos} at {source_20x_seg}"
                )
                print(f"  Skipping {pos} - run batch_upscale_nuclear_seg.py first")
                continue
        else:
            # For ISS and track: use pos/0/0 (actual segmentation array)
            src_path = nuclear_seg_path / pos / "0" / "0"

        # Remove old symlink at wrong location (pos/nuclear_seg) if it exists
        old_location = dest_store_path / pos / "nuclear_seg"
        if old_location.exists():
            import shutil

            print(f"  Removing old symlink structure at {old_location}")
            shutil.rmtree(old_location)

        # Create symlink at pos/0/nuclear_seg/0 (matching pheno structure)
        # This ensures pyramid building logic works consistently across all stores
        dest_link = dest_store_path / pos / "0" / "nuclear_seg" / "0"

        print(f"  Linking {pos}: {src_path} -> {dest_link}")

        # Ensure parent directory exists
        dest_link.parent.mkdir(parents=True, exist_ok=True)

        # Remove existing symlink if it exists
        if dest_link.exists() or dest_link.is_symlink():
            dest_link.unlink()

        try:
            os.symlink(str(src_path.resolve()), str(dest_link))
            print(f"  ✓ Symlinked {dest_link} -> {src_path}")
        except Exception as e:
            print(f"  ✗ ERROR: failed to symlink {dest_link}: {e}")


def find_experiments_needing_symlink(
    experiment_configs_dir: Path, symlink_target: str = "iss", force: bool = False
):
    """
    Find all experiments that need nuclear segmentation symlinks.

    Args:
        experiment_configs_dir: Path to experiment configs directory
        symlink_target: Target zarr to check ("pheno", "iss", or "track")
        force: If True, include all experiments even if symlinks exist

    Returns list of experiment names that need symlinks.
    """
    experiments_to_process = []

    # Map target to source and destination store paths
    mapping = {
        "pheno": {
            "source": "lc_20x_segmentation",
            "dest": "pheno_assembled",
        },
        "iss": {
            "source": "iss_segmentation",
            "dest": "iss_stitch_registered",
        },
        "track": {
            "source": "lc_5x_segmentation",
            "dest": "lc_5x_phase_2d_stitched",
        },
    }

    if symlink_target not in mapping:
        raise ValueError(f"Invalid symlink_target: {symlink_target}")

    source_store_key = mapping[symlink_target]["source"]
    dest_store_key = mapping[symlink_target]["dest"]

    # Find all experiment config files
    config_files = list(experiment_configs_dir.glob("ops*_config.yaml"))
    print(f"Found {len(config_files)} experiment configs")
    print(
        f"Scanning for experiments needing symlinks from {source_store_key} to {dest_store_key}"
    )

    for config_file in tqdm(config_files, desc="Scanning experiments"):
        experiment = config_file.stem.replace("_config", "")

        try:
            dataset = OpsDataset(experiment)

            # Check if source segmentation exists
            nuclear_seg_path = dataset.store_paths.get(source_store_key)
            if not nuclear_seg_path or not nuclear_seg_path.exists():
                continue

            # Check if destination store exists
            dest_path = dataset.store_paths.get(dest_store_key)
            if not dest_path or not dest_path.exists():
                continue

            # Find all positions in destination store
            positions = []
            for row_dir in dest_path.glob("*"):
                if row_dir.is_dir() and row_dir.name not in [".zattrs", ".zgroup"]:
                    for col_dir in row_dir.glob("*"):
                        if col_dir.is_dir() and col_dir.name not in [
                            ".zattrs",
                            ".zgroup",
                        ]:
                            positions.append(f"{row_dir.name}/{col_dir.name}")

            # In force mode, include all experiments that have the required stores
            if force:
                needs_processing = True
            else:
                needs_processing = False
                for pos in positions:
                    # Check if nuclear_seg symlink exists at pos/0/nuclear_seg/0
                    dest_nuclear_seg_link = dest_path / pos / "0" / "nuclear_seg" / "0"
                    if not (
                        dest_nuclear_seg_link.exists()
                        or dest_nuclear_seg_link.is_symlink()
                    ):
                        needs_processing = True
                        break

            if needs_processing:
                experiments_to_process.append(experiment)
                mode_str = "(force mode)" if force else ""
                print(
                    f"  ✓ {experiment} needs processing for {symlink_target} {mode_str}"
                )

        except Exception as e:
            print(f"  ✗ Error checking {experiment}: {e}")
            continue

    return experiments_to_process


def main():
    parser = argparse.ArgumentParser(
        description="Batch symlink nuclear segmentations for OPS experiments"
    )
    parser.add_argument(
        "--experiment-configs-dir",
        type=Path,
        default=Path(
            f"{BASE_PATH}/configs/experiment_configs"
        ),
        help="Path to experiment_configs directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list experiments that need symlinks, don't process",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        type=str,
        help="Process only specific experiments (skip auto-detection)",
    )
    parser.add_argument(
        "--symlink-target",
        type=str,
        choices=["pheno", "iss", "track"],
        default="iss",
        help="Target zarr to symlink nuclear segmentation into: 'pheno' (pheno_assembled), 'iss' (iss_stitch), or 'track' (lc_5x_phase_2d_stitched)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-symlink even if symlinks already exist (use this to fix wrong symlinks)",
    )
    parser.add_argument(
        "--zarr-version",
        type=int,
        choices=[2, 3],
        default=3,
        help="Destination zarr version (default 3). v3-native pheno/track "
             "stitching writes the _v3-suffixed store, so symlinks must land "
             "there. Use 2 only for legacy v2 destinations.",
    )

    args = parser.parse_args()

    if args.experiments:
        experiments_to_process = args.experiments
        print(f"Processing specified experiments: {experiments_to_process}")
    else:
        print(f"Scanning for experiments in: {args.experiment_configs_dir}")
        if args.force:
            print("FORCE MODE: Will re-symlink all experiments even if symlinks exist")
        experiments_to_process = find_experiments_needing_symlink(
            args.experiment_configs_dir, args.symlink_target, args.force
        )

    print(f"\n{'='*60}")
    print(f"Found {len(experiments_to_process)} experiments needing symlinks:")
    for exp in experiments_to_process:
        print(f"  - {exp}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("Dry run - exiting without processing")
        return

    # Process each experiment
    for i, experiment in enumerate(experiments_to_process, 1):
        print(f"\n[{i}/{len(experiments_to_process)}] Processing {experiment}...")
        try:
            symlink_nuclear_seg(
                experiment=experiment,
                symlink_target=args.symlink_target,
                zarr_version=args.zarr_version,
            )
            print(f"  ✓ Completed {experiment}")
        except Exception as e:
            print(f"  ✗ Error processing {experiment}: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"Batch processing complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()


# Usage examples:
#
# 1. Symlink ISS segmentation to ISS stitched zarr (default):
#    python batch_symlink_nuclear_seg.py --experiments ops0042_20250520
#
# 2. Symlink tracking segmentation to tracking zarr:
#    python batch_symlink_nuclear_seg.py --experiments ops0042_20250520 --symlink-target track
#
# 3. Symlink 20x segmentation to phenotyping zarr (will use 20x_nuclear_seg if available):
#    python batch_symlink_nuclear_seg.py --experiments ops0042_20250520 --symlink-target pheno
#
# 4. Dry run to see what would be processed for ISS:
#    python batch_symlink_nuclear_seg.py --dry-run --symlink-target iss
#
# 5. Process all experiments that need ISS symlinks:
#    python batch_symlink_nuclear_seg.py --symlink-target iss
#
# 6. FORCE MODE - Re-symlink all ISS experiments (even if symlinks exist, to fix wrong symlinks):
#    python batch_symlink_nuclear_seg.py --symlink-target iss --force
#
# 7. Force re-symlink specific experiments:
#    python batch_symlink_nuclear_seg.py --experiments ops0042_20250520 --symlink-target iss --force
