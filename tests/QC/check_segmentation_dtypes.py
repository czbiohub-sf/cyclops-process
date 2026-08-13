"""
Check segmentation dtype across all experiments.

This script checks the dtype of phenotyping segmentation arrays to identify
experiments that were created with int16 instead of int32, which causes overflow
when there are more than 32767 cells.

python tests/check_segmentation_dtypes.py -e 71
python tests/check_segmentation_dtypes.py -e 71 --stores pheno_assembled lc_20x_segmentation_cells
python tests/check_segmentation_dtypes.py  # check all experiments
"""

import sys
import os
from pathlib import Path
import yaml
from prettytable import PrettyTable
import re
from tqdm import tqdm
from iohub.ngff import open_ome_zarr
import dask.array as da

sys.path.insert(0, os.getcwd())

from ops_utils.data.experiment import OpsDataset
from ops_utils.data.filesystem import resolve_experiment_name


def check_segmentation_dtype(experiment: str, store_key: str = "pheno_assembled"):
    """
    Check the dtype of segmentation for a single experiment.

    Args:
        experiment: Experiment name
        store_key: Which store to check (default: pheno_assembled for stitched segmentation)

    Returns:
        dict with dtype info or None if store doesn't exist
    """
    try:
        dataset = OpsDataset(experiment)
        store_path = dataset.store_paths.get(store_key)

        if not store_path or not store_path.exists():
            return None

        # Open store and check first available position
        with open_ome_zarr(store_path, mode="r") as store:
            positions = [path for path, _ in store.positions()]
            if not positions:
                return None

            # Check first position
            first_pos = positions[0]

            # Try to access seg/0 for stitched stores, or just 0 for per-tile stores
            try:
                if "seg" in store[first_pos].zgroup:
                    seg_array = store[first_pos].zgroup["seg"]["0"]
                else:
                    seg_array = store[first_pos]["0"]
            except (KeyError, AttributeError):
                # Fall back to regular 0 array
                seg_array = store[first_pos]["0"]

            # Get dtype and shape info - use Dask slicing to sample a single chunk
            # Determine chunk size to sample
            chunks = seg_array.chunks if hasattr(seg_array, 'chunks') else None

            if seg_array.ndim > 2:
                # For multidimensional arrays, get a single 2D slice
                # Sample just the first chunk in each dimension
                if chunks and len(chunks) >= seg_array.ndim:
                    # Get first chunk size for last two dimensions (spatial)
                    # Handle case where chunks might be integers or tuples
                    chunk_h_dim = chunks[-2]
                    chunk_w_dim = chunks[-1]
                    chunk_h = chunk_h_dim[0] if isinstance(chunk_h_dim, tuple) else chunk_h_dim
                    chunk_w = chunk_w_dim[0] if isinstance(chunk_w_dim, tuple) else chunk_w_dim
                    chunk_h = min(chunk_h, seg_array.shape[-2])
                    chunk_w = min(chunk_w, seg_array.shape[-1])
                else:
                    chunk_h = min(512, seg_array.shape[-2])
                    chunk_w = min(512, seg_array.shape[-1])

                seg_sample = seg_array[0, 0, 0, :chunk_h, :chunk_w]
            else:
                # For 2D arrays, sample first chunk
                if chunks and len(chunks) >= 2:
                    chunk_h_dim = chunks[0]
                    chunk_w_dim = chunks[1]
                    chunk_h = chunk_h_dim[0] if isinstance(chunk_h_dim, tuple) else chunk_h_dim
                    chunk_w = chunk_w_dim[0] if isinstance(chunk_w_dim, tuple) else chunk_w_dim
                    chunk_h = min(chunk_h, seg_array.shape[0])
                    chunk_w = min(chunk_w, seg_array.shape[1])
                else:
                    chunk_h = min(512, seg_array.shape[0])
                    chunk_w = min(512, seg_array.shape[1])

                seg_sample = seg_array[:chunk_h, :chunk_w]

            # Compute only the sampled chunk
            seg_2d = seg_sample.compute() if hasattr(seg_sample, 'compute') else seg_sample

            max_val = int(seg_2d.max())
            min_val = int(seg_2d.min())
            unique_vals = len(set(seg_2d.flatten()))

            return {
                "experiment": experiment,
                "store_key": store_key,
                "dtype": str(seg_array.dtype),
                "shape": seg_array.shape,
                "max_value": max_val,
                "min_value": min_val,
                "sample_unique_vals": unique_vals,
                "positions_count": len(positions),
                "first_position": first_pos,
            }

    except Exception as e:
        return {
            "experiment": experiment,
            "store_key": store_key,
            "error": str(e),
        }


def check_all_experiments(store_keys=["pheno_assembled", "lc_20x_segmentation_cells", "lc_20x_segmentation", "iss_stitch", "iss_segmentation", "lc_5x_phase_2d_stitched", "lc_5x_segmentation"]):
    """
    Check segmentation dtype across all experiments.

    Args:
        store_keys: List of store keys to check
            Default stores:
            - pheno_assembled: Final assembled phenotyping with segmentation
            - lc_20x_segmentation_cells: Cell segmentation (membranes) for phenotyping - phenotyping_segmentation_cells.zarr
            - lc_20x_segmentation: Nuclear segmentation for phenotyping (stitched) - phenotyping_segmentation_stitched.zarr
            - iss_stitch: ISS stitched phase - bc_stitched.zarr
            - iss_segmentation: ISS segmentation - bc_segmentation.zarr
            - lc_5x_phase_2d_stitched: Tracking stitched phase - tracking_phase_2d_stitched.zarr
            - lc_5x_segmentation: Tracking segmentation (stitched) - tracking_segmentation_stitched.zarr
    """
    dummy_dataset = OpsDataset("dummy")
    config_root_dir = dummy_dataset.config_paths["exp_config_dir"]

    if not config_root_dir.is_dir():
        print(f"Configuration directory not found at {config_root_dir}. Aborting.")
        return

    config_files = sorted(list(config_root_dir.glob("*_config.yaml")))

    print("\n" + "=" * 120)
    print("SEGMENTATION DTYPE CHECK")
    print("=" * 120 + "\n")

    all_results = []
    skipped_experiments = []

    for config_path in tqdm(config_files, desc="Checking experiments"):
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                if not config or "experiment_name" not in config:
                    continue

            exp_name = config["experiment_name"]

            # Filter experiments: only include standard format opsXXXX_YYYYMMDD
            standard_format = re.match(r"^ops\d{4}_\d{8}$", exp_name)
            if not standard_format:
                skipped_experiments.append(exp_name)
                continue

            # Check each store type
            for store_key in store_keys:
                result = check_segmentation_dtype(exp_name, store_key)
                if result:
                    all_results.append(result)

        except Exception as e:
            print(f"\nError processing {config_path.stem}: {e}")
            continue

    # Display results
    if skipped_experiments:
        print(f"\nSkipped {len(skipped_experiments)} non-standard experiments\n")

    print(f"\nChecked {len(all_results)} stores across experiments\n")

    # Group results by dtype
    results_by_dtype = {}
    for result in all_results:
        if "error" in result:
            dtype_key = "ERROR"
        else:
            dtype_key = result["dtype"]

        if dtype_key not in results_by_dtype:
            results_by_dtype[dtype_key] = []
        results_by_dtype[dtype_key].append(result)

    # Print summary by dtype
    print("\n" + "=" * 120)
    print("SUMMARY BY DTYPE")
    print("=" * 120)

    summary_table = PrettyTable()
    summary_table.field_names = ["DType", "Count", "Store Types"]
    summary_table.align = "l"

    for dtype_key, results in sorted(results_by_dtype.items()):
        store_types = set(r.get("store_key", "unknown") for r in results)
        summary_table.add_row([
            dtype_key,
            len(results),
            ", ".join(sorted(store_types))
        ])

    print(summary_table)

    # Print detailed table for problematic dtypes (int16)
    if "int16" in results_by_dtype:
        print("\n" + "=" * 120)
        print("⚠️  EXPERIMENTS WITH INT16 SEGMENTATION (WILL OVERFLOW AT 32,767 CELLS)")
        print("=" * 120)

        int16_table = PrettyTable()
        int16_table.field_names = [
            "Experiment",
            "Store",
            "Max Value",
            "Min Value",
            "Shape",
            "Positions",
        ]
        int16_table.align = "l"
        int16_table.align["Max Value"] = "r"
        int16_table.align["Min Value"] = "r"
        int16_table.align["Positions"] = "r"

        for result in sorted(results_by_dtype["int16"], key=lambda x: x["experiment"]):
            if "error" not in result:
                int16_table.add_row([
                    result["experiment"],
                    result["store_key"],
                    f"{result['max_value']:,}",
                    f"{result['min_value']:,}",
                    f"{result['shape']}",
                    result['positions_count'],
                ])

        print(int16_table)
        print("\n⚠️  These experiments need to be re-segmented with dtype=int32 or uint32")
        print("⚠️  Max value of 32767 and negative min values indicate integer overflow")

    # Print full details for each dtype
    print("\n" + "=" * 120)
    print("DETAILED RESULTS")
    print("=" * 120)

    for dtype_key in sorted(results_by_dtype.keys()):
        results = results_by_dtype[dtype_key]

        print(f"\n\n--- {dtype_key} ({len(results)} stores) ---\n")

        detail_table = PrettyTable()
        if dtype_key == "ERROR":
            detail_table.field_names = ["Experiment", "Store", "Error"]
            detail_table.align = "l"
            for result in results[:20]:  # Show first 20 errors
                detail_table.add_row([
                    result["experiment"],
                    result["store_key"],
                    result.get("error", "Unknown")[:60]
                ])
        else:
            detail_table.field_names = [
                "Experiment",
                "Store",
                "DType",
                "Max Value",
                "Min Value",
                "Shape",
                "Positions",
                "First Position",
            ]
            detail_table.align = "l"
            detail_table.align["Max Value"] = "r"
            detail_table.align["Min Value"] = "r"
            detail_table.align["Positions"] = "r"

            for result in results[:20]:  # Show first 20 of each type
                detail_table.add_row([
                    result["experiment"],
                    result["store_key"],
                    result["dtype"],
                    f"{result['max_value']:,}",
                    f"{result['min_value']:,}",
                    f"{result['shape']}",
                    result['positions_count'],
                    result['first_position'],
                ])

        print(detail_table)

        if len(results) > 20:
            print(f"\n... and {len(results) - 20} more")

    # Save to CSV
    try:
        output_dir = Path(__file__).resolve().parent

        import pandas as pd

        # Flatten results for CSV
        csv_rows = []
        for result in all_results:
            if "error" in result:
                csv_rows.append({
                    "experiment": result["experiment"],
                    "store_key": result["store_key"],
                    "dtype": "ERROR",
                    "error": result["error"],
                })
            else:
                csv_rows.append({
                    "experiment": result["experiment"],
                    "store_key": result["store_key"],
                    "dtype": result["dtype"],
                    "max_value": result["max_value"],
                    "min_value": result["min_value"],
                    "shape": str(result["shape"]),
                    "positions_count": result["positions_count"],
                    "first_position": result["first_position"],
                })

        df = pd.DataFrame(csv_rows)
        csv_path = output_dir / "segmentation_dtype_check.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n\nSaved results to: {csv_path}")

    except Exception as e:
        print(f"\nWarning: Failed to save CSV: {e}")


def check_single_experiment(experiment: str, store_keys=["pheno_assembled", "lc_20x_segmentation_cells", "lc_20x_segmentation", "iss_stitch", "iss_segmentation", "lc_5x_phase_2d_stitched", "lc_5x_segmentation"]):
    """
    Check segmentation dtype for a single experiment.

    Args:
        experiment: Experiment name (e.g., 'ops0072_20241015')
        store_keys: List of store keys to check
            Default stores:
            - pheno_assembled: Final assembled phenotyping with segmentation
            - lc_20x_segmentation_cells: Cell segmentation (membranes) for phenotyping - phenotyping_segmentation_cells.zarr
            - lc_20x_segmentation: Nuclear segmentation for phenotyping (stitched) - phenotyping_segmentation_stitched.zarr
            - iss_stitch: ISS stitched phase - bc_stitched.zarr
            - iss_segmentation: ISS segmentation - bc_segmentation.zarr
            - lc_5x_phase_2d_stitched: Tracking stitched phase - tracking_phase_2d_stitched.zarr
            - lc_5x_segmentation: Tracking segmentation (stitched) - tracking_segmentation_stitched.zarr
    """
    print("\n" + "=" * 120)
    print(f"SEGMENTATION DTYPE CHECK - {experiment}")
    print("=" * 120 + "\n")

    results = []
    for store_key in store_keys:
        print(f"Checking {store_key}...", end=" ")
        result = check_segmentation_dtype(experiment, store_key)
        if result:
            results.append(result)
            if "error" in result:
                print(f"ERROR: {result['error']}")
            else:
                print(f"OK - dtype={result['dtype']}, max={result['max_value']:,}, shape={result['shape']}")
        else:
            print("Store not found")

    # Display detailed results
    if results:
        print("\n" + "=" * 120)
        print("DETAILED RESULTS")
        print("=" * 120 + "\n")

        detail_table = PrettyTable()
        detail_table.field_names = [
            "Store",
            "DType",
            "Max Value",
            "Min Value",
            "Shape",
            "Positions",
            "First Position",
            "Error",
        ]
        detail_table.align = "l"
        detail_table.align["Max Value"] = "r"
        detail_table.align["Min Value"] = "r"
        detail_table.align["Positions"] = "r"

        for result in results:
            if "error" in result:
                detail_table.add_row([
                    result["store_key"],
                    "ERROR",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    result["error"][:60],
                ])
            else:
                detail_table.add_row([
                    result["store_key"],
                    result["dtype"],
                    f"{result['max_value']:,}",
                    f"{result['min_value']:,}",
                    f"{result['shape']}",
                    result['positions_count'],
                    result['first_position'],
                    "-",
                ])

        print(detail_table)

        # Check for int16 issues
        int16_results = [r for r in results if r.get("dtype") == "int16"]
        if int16_results:
            print("\n⚠️  WARNING: Detected int16 segmentation dtype!")
            print("⚠️  This will overflow at 32,767 cells")
            print("⚠️  Consider re-segmenting with dtype=int32 or uint32")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check segmentation dtypes across experiments")
    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        help="Single experiment to check (e.g., 'ops0072_20241015' or just '0072')",
    )
    parser.add_argument(
        "--stores",
        nargs="+",
        default=["pheno_assembled", "lc_20x_segmentation_cells", "lc_20x_segmentation", "iss_stitch", "iss_segmentation", "lc_5x_phase_2d_stitched", "lc_5x_segmentation"],
        help="Store keys to check (default: pheno_assembled lc_20x_segmentation_cells lc_20x_segmentation iss_stitch iss_segmentation lc_5x_phase_2d_stitched lc_5x_segmentation)",
    )
    args = parser.parse_args()

    if args.experiment:
        # Resolve experiment name using the filesystem helper
        exp_name = resolve_experiment_name(args.experiment, verbose=True)
        check_single_experiment(exp_name, store_keys=args.stores)
    else:
        check_all_experiments(store_keys=args.stores)

    # To run: python check_segmentation_dtypes.py
    # To check single experiment: python check_segmentation_dtypes.py --experiment ops0072_20241015
    # To check with shorthand: python check_segmentation_dtypes.py -e 0072
    # To check specific stores: python check_segmentation_dtypes.py --stores pheno_assembled iss_segmentation
