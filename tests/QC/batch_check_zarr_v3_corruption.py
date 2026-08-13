#!/usr/bin/env python
"""
Batch check Zarr v3 stores for corruption by v2 metadata.

This script scans all OPS experiments and checks if their phenotyping_v3.zarr stores
(and optionally other v3 stores) have been corrupted by v2 metadata files
(.zattrs, .zgroup) which can cause issues when opening with iohub or other
v3-compliant tools.

A Zarr v3 store is considered corrupted if:
1. It has zarr.json files (indicating v3 format)
2. It also has .zattrs or .zgroup files (v2 metadata) in the same directories

Environment:
    This script can run in any environment - it only checks for file existence,
    not opening the stores. No special Zarr v3 environment needed.

Usage:
    # Check all experiments (phenotyping_v3.zarr by default)
    python -m tests/QC/batch_check_zarr_v3_corruption

    # Check specific experiments
    python -m tests/QC/batch_check_zarr_v3_corruption --experiments ops0094_20251217

    # Check all v3 store types (pheno, track, iss)
    python -m tests/QC/batch_check_zarr_v3_corruption --all-stores

    # Fix corrupted stores by removing v2 metadata files
    python -m tests/QC/batch_check_zarr_v3_corruption --uncorrupt

    # Dry run to see what would be removed (safe preview)
    python -m tests/QC/batch_check_zarr_v3_corruption --uncorrupt --dry-run

    # Fix specific experiments
    python -m tests/QC/batch_check_zarr_v3_corruption --experiments ops0094_20251217 --uncorrupt

    # Verbose output showing all checks
    python -m tests/QC/batch_check_zarr_v3_corruption --verbose
"""
import sys
import os

sys.path.insert(0, os.getcwd())

import argparse
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple

from ops_utils.data.experiment import OpsDataset
from cyclops_process.paths import BASE_PATH


# Map of store types to their keys in OpsDataset.store_paths
# pheno_assembled_v3 -> {experiment_path}/3-assembly/phenotyping_v3.zarr
STORE_MAP = {
    "pheno": "pheno_assembled_v3",  # phenotyping_v3.zarr
    "track": "lc_5x_phase_2d_stitched_v3",
    "iss": "iss_stitch_registered_v3",
}


def fix_zarr_v3_corruption(zarr_path: Path, dry_run: bool = False, verbose: bool = False) -> Dict:
    """
    Remove v2 metadata files (.zattrs, .zgroup) from a Zarr v3 store.

    Args:
        zarr_path: Path to zarr store
        dry_run: If True, only report what would be removed
        verbose: Print detailed info

    Returns:
        dict with keys:
            - files_removed: list of files that were removed
            - success: bool, whether operation succeeded
    """
    result = {
        "files_removed": [],
        "success": False,
    }

    if not zarr_path.exists():
        return result

    # Check for corruption first
    corruption_check = check_zarr_v3_corruption(zarr_path, verbose=False)

    if not corruption_check["is_corrupted"]:
        result["success"] = True
        return result

    # Get v2 files to remove
    v2_file_paths = []
    for rel_path in corruption_check["v2_files"]:
        v2_file_paths.append(zarr_path / rel_path)

    if verbose or dry_run:
        print(f"    Found {len(v2_file_paths)} v2 metadata files to remove:")
        for f in v2_file_paths:
            print(f"      - {f.relative_to(zarr_path)}")

    if not dry_run:
        # Remove the files
        for file_path in v2_file_paths:
            try:
                file_path.unlink()
                result["files_removed"].append(str(file_path.relative_to(zarr_path)))
            except Exception as e:
                print(f"      ✗ Failed to remove {file_path.relative_to(zarr_path)}: {e}")
                return result

        result["success"] = True
    else:
        result["files_removed"] = corruption_check["v2_files"]
        result["success"] = True

    return result


def check_zarr_v3_corruption(zarr_path: Path, verbose: bool = False) -> Dict:
    """
    Check if a Zarr store is v3 and whether it's corrupted by v2 metadata.

    Only checks key directories (store root and position levels like A/1/0) to avoid
    scanning millions of chunk files.

    Args:
        zarr_path: Path to zarr store
        verbose: Print detailed checking info

    Returns:
        dict with keys:
            - is_v3: bool, whether store has v3 metadata (zarr.json)
            - has_v2_metadata: bool, whether store has v2 metadata (.zattrs/.zgroup)
            - is_corrupted: bool, whether v3 store has v2 metadata
            - v2_files: list of paths to v2 metadata files found
            - v3_files: list of paths to v3 metadata files found
    """
    result = {
        "is_v3": False,
        "has_v2_metadata": False,
        "is_corrupted": False,
        "v2_files": [],
        "v3_files": [],
    }

    if not zarr_path.exists():
        return result

    # Check only key directories to avoid scanning chunk files
    # For HCS/plate stores: root, A/, A/1/, A/1/0/, A/1/0/labels/, etc.
    dirs_to_check = [zarr_path]

    # Add position directories (A/1/0, B/2/0, etc.)
    for row_dir in zarr_path.glob("*"):
        if row_dir.is_dir() and not row_dir.name.startswith("."):
            dirs_to_check.append(row_dir)
            for col_dir in row_dir.glob("*"):
                if col_dir.is_dir() and not col_dir.name.startswith("."):
                    dirs_to_check.append(col_dir)
                    for fov_dir in col_dir.glob("*"):
                        if fov_dir.is_dir() and not fov_dir.name.startswith("."):
                            dirs_to_check.append(fov_dir)
                            # Check labels subdirectory too
                            labels_dir = fov_dir / "labels"
                            if labels_dir.exists():
                                dirs_to_check.append(labels_dir)

    # Check for v3 and v2 metadata in these directories only
    v3_files = []
    v2_files = []

    for dir_path in dirs_to_check:
        # Check for zarr.json (v3)
        zarr_json = dir_path / "zarr.json"
        if zarr_json.exists():
            v3_files.append(zarr_json)

        # Check for .zattrs and .zgroup (v2)
        zattrs = dir_path / ".zattrs"
        zgroup = dir_path / ".zgroup"
        if zattrs.exists():
            v2_files.append(zattrs)
        if zgroup.exists():
            v2_files.append(zgroup)

    result["v3_files"] = [str(f.relative_to(zarr_path)) for f in v3_files]
    result["v2_files"] = [str(f.relative_to(zarr_path)) for f in v2_files]
    result["is_v3"] = len(v3_files) > 0
    result["has_v2_metadata"] = len(v2_files) > 0

    if verbose and result["is_v3"]:
        print(f"    Found {len(v3_files)} zarr.json files (Zarr v3)")

    if verbose and result["has_v2_metadata"]:
        print(f"    Found {len(v2_files)} v2 metadata files (.zattrs/.zgroup)")

    # Corruption = v3 store with v2 metadata
    result["is_corrupted"] = result["is_v3"] and result["has_v2_metadata"]

    return result


def check_experiment_stores(
    experiment: str,
    store_types: List[str] = None,
    verbose: bool = False
) -> Dict:
    """
    Check all specified stores for an experiment.

    Args:
        experiment: Experiment name (e.g., "ops0094_20251217")
        store_types: List of store types to check (e.g., ["pheno", "track"])
                    If None, checks only "pheno" (phenotyping_v3.zarr)
        verbose: Print detailed info

    Returns:
        dict mapping store_type -> corruption check result
    """
    if store_types is None:
        store_types = ["pheno"]  # Default to only phenotyping_v3.zarr

    try:
        dataset = OpsDataset(experiment)
    except Exception as e:
        if verbose:
            print(f"  ✗ Error loading {experiment}: {e}")
        return {}

    results = {}

    for store_type in store_types:
        store_key = STORE_MAP.get(store_type)
        if not store_key:
            continue

        store_path = dataset.store_paths.get(store_key)

        if not store_path or not store_path.exists():
            if verbose:
                print(f"  - {store_type}: not found")
            continue

        if verbose:
            print(f"  Checking {store_type}: {store_path}")

        result = check_zarr_v3_corruption(store_path, verbose=verbose)
        results[store_type] = result

        # Print summary
        if result["is_v3"]:
            if result["is_corrupted"]:
                print(f"  ⚠ {store_type}: CORRUPTED (v3 with {len(result['v2_files'])} v2 metadata files)")
            else:
                print(f"  ✓ {store_type}: Clean v3 store")
        elif verbose:
            print(f"  - {store_type}: Not a v3 store")

    return results


def find_all_experiments(experiment_configs_dir: Path) -> List[str]:
    """Find all experiment names from config files."""
    config_files = list(experiment_configs_dir.glob("ops*_config.yaml"))
    experiments = [cf.stem.replace("_config", "") for cf in config_files]
    return sorted(experiments)


def main():
    parser = argparse.ArgumentParser(
        description="Batch check Zarr v3 stores for corruption by v2 metadata"
    )
    parser.add_argument(
        "--experiment-configs-dir",
        type=Path,
        default=Path(BASE_PATH) / "configs/experiment_configs",
        help="Path to experiment_configs directory",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        type=str,
        help="Check only specific experiments (skip auto-detection)",
    )
    parser.add_argument(
        "--all-stores",
        action="store_true",
        help="Check all v3 store types (pheno, track, iss). Default is pheno only.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed checking information",
    )
    parser.add_argument(
        "--uncorrupt",
        action="store_true",
        help="Fix corrupted stores by removing v2 metadata files, then validate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --uncorrupt, show what would be removed without actually removing",
    )

    args = parser.parse_args()

    # Determine which experiments to check
    if args.experiments:
        experiments = args.experiments
        print(f"Checking specified experiments: {len(experiments)}")
    else:
        print(f"Scanning for experiments in: {args.experiment_configs_dir}")
        experiments = find_all_experiments(args.experiment_configs_dir)
        print(f"Found {len(experiments)} experiment configs")

    # Determine which stores to check
    if args.all_stores:
        store_types = list(STORE_MAP.keys())
        print(f"Checking all store types: {store_types}")
    else:
        store_types = ["pheno"]
        print(f"Checking phenotyping_v3.zarr only (use --all-stores for track/iss)")

    print(f"{'='*80}")
    if args.uncorrupt:
        if args.dry_run:
            print(f"DRY RUN - Checking corruption and showing what would be fixed")
        else:
            print(f"Checking and fixing Zarr v3 corruption (removing v2 metadata)")
    else:
        print(f"Checking for Zarr v3 corruption (v2 metadata in v3 stores)")
    print(f"{'='*80}\n")

    # Track summary statistics
    total_checked = 0
    total_corrupted = 0
    total_fixed = 0
    total_files_removed = 0
    corrupted_experiments = []
    stores_to_fix = []  # List of (experiment, store_type, store_path, result)

    # Check each experiment
    for experiment in tqdm(experiments, desc="Checking experiments"):
        if args.verbose:
            print(f"\n{experiment}:")

        results = check_experiment_stores(
            experiment,
            store_types=store_types,
            verbose=args.verbose
        )

        # Check if any stores are corrupted
        experiment_corrupted = False
        for store_type, result in results.items():
            if result.get("is_corrupted"):
                experiment_corrupted = True
                total_corrupted += 1
                if not args.verbose:
                    print(f"⚠ {experiment} - {store_type}: CORRUPTED ({len(result['v2_files'])} v2 files)")

                # Store for fixing if --uncorrupt is set
                if args.uncorrupt:
                    try:
                        dataset = OpsDataset(experiment)
                        store_key = STORE_MAP.get(store_type)
                        store_path = dataset.store_paths.get(store_key)
                        stores_to_fix.append((experiment, store_type, store_path, result))
                    except Exception as e:
                        print(f"  ✗ Error getting store path: {e}")

            elif result.get("is_v3") and not args.verbose:
                # Print clean stores too (when not verbose)
                print(f"✓ {experiment} - {store_type}: Clean v3 store")

        if experiment_corrupted:
            corrupted_experiments.append(experiment)

        if results:
            total_checked += len(results)

    # Fix corrupted stores if --uncorrupt is set
    if args.uncorrupt and stores_to_fix:
        print(f"\n{'='*80}")
        print(f"Fixing {len(stores_to_fix)} corrupted stores...")
        print(f"{'='*80}\n")

        for experiment, store_type, store_path, corruption_result in tqdm(stores_to_fix, desc="Fixing stores"):
            print(f"\n{experiment} - {store_type}:")

            fix_result = fix_zarr_v3_corruption(
                store_path,
                dry_run=args.dry_run,
                verbose=args.verbose
            )

            if fix_result["success"]:
                num_removed = len(fix_result["files_removed"])
                total_files_removed += num_removed

                if args.dry_run:
                    print(f"  Would remove {num_removed} v2 files")
                else:
                    print(f"  ✓ Removed {num_removed} v2 files")
                    total_fixed += 1

                    # Validate the fix
                    print(f"  Validating fix...")
                    validation = check_zarr_v3_corruption(store_path, verbose=False)
                    if validation["is_corrupted"]:
                        print(f"  ✗ VALIDATION FAILED - Still corrupted!")
                    else:
                        print(f"  ✓ Validated - Store is now clean")
            else:
                print(f"  ✗ Failed to fix")

    # Print summary
    print(f"\n{'='*80}")
    print(f"Summary")
    print(f"{'='*80}")
    print(f"Total experiments checked: {len(experiments)}")
    print(f"Total stores checked: {total_checked}")
    print(f"Corrupted stores found: {total_corrupted}")
    print(f"Experiments with corruption: {len(corrupted_experiments)}")

    if args.uncorrupt:
        print(f"\nFix results:")
        if args.dry_run:
            print(f"  Would fix: {len(stores_to_fix)} stores")
            print(f"  Would remove: {total_files_removed} v2 metadata files")
        else:
            print(f"  Successfully fixed: {total_fixed} stores")
            print(f"  Total v2 metadata files removed: {total_files_removed}")

    if corrupted_experiments and not args.uncorrupt:
        print(f"\nCorrupted experiments (use --uncorrupt to fix):")
        for exp in corrupted_experiments:
            print(f"  - {exp}")
    elif not corrupted_experiments:
        print(f"\n✓ No corrupted stores found!")
    elif args.uncorrupt and not args.dry_run:
        print(f"\n✓ All corrupted stores have been fixed!")

    print(f"{'='*80}")


if __name__ == "__main__":
    main()
