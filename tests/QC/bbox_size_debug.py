"""
Debug script to analyze bounding box sizes across all experiments.

# usage
python tests/QC/bbox_size_debug.py
python tests/QC/bbox_size_debug.py --skip-low-count-exps --verbose-timing

This script:
1. Finds all experiments with linked_pheno_iss.csv files
2. Analyzes bbox sizes (both from segmentation and fallback 200px boxes)
3. Reports distribution statistics and anomalies
4. Identifies cells with abnormally large bboxes (>1000x1000 pixels)
"""

import sys
import os
from pathlib import Path
from collections import defaultdict
import yaml
from prettytable import PrettyTable
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
import ast
import time

sys.path.insert(0, os.getcwd())

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


from ops_utils.data.experiment import OpsDataset

# Low-count experiments to exclude when --skip-low-count-exps is used
LOW_COUNT_EXPERIMENTS = [
    "ops0028_20250417",
    "ops0011_20250205",
    "ops0012_20250206",
    "ops0023_20250317",
]


def parse_bbox(bbox_str):
    """Parse bbox string (tuple format) into (min_row, min_col, max_row, max_col)."""
    try:
        if pd.isna(bbox_str):
            return None
        # Handle tuple string format: "(min_row, min_col, max_row, max_col)"
        bbox = ast.literal_eval(bbox_str)
        if len(bbox) == 4:
            return tuple(bbox)
        return None
    except Exception:
        return None


def calculate_bbox_dimensions(bbox):
    """Calculate width and height from bbox tuple (min_row, min_col, max_row, max_col)."""
    if bbox is None:
        return None, None
    min_row, min_col, max_row, max_col = bbox
    height = max_row - min_row
    width = max_col - min_col
    return height, width


def analyze_experiment_bboxes(dataset: OpsDataset, config: dict, exp_name: str, verbose: bool = False):
    """
    Analyze bbox sizes for a single experiment.

    Returns dict with statistics or None if no linked files found.
    """
    timings = {}
    t_start = time.time()

    # Directly glob for linked CSV files - faster than checking wells
    t0 = time.time()
    linked_csvs = list(dataset.results.glob("*_linked_pheno_iss.csv"))
    timings['glob_files'] = time.time() - t0

    if not linked_csvs:
        return None

    t0 = time.time()
    all_dfs = []
    for linked_path in linked_csvs:
        well_short = linked_path.stem.replace("_linked_pheno_iss", "")

        try:
            # Only read the columns we need - much faster!
            df = pd.read_csv(linked_path, usecols=['bbox', 'segmentation_id'])
            df['well_short'] = well_short
            all_dfs.append(df)

        except Exception as e:
            print(f"  Warning: Error reading {linked_path}: {e}")
            continue

    if not all_dfs:
        return None

    timings['read_csvs'] = time.time() - t0

    # Concatenate all dataframes for vectorized operations
    t0 = time.time()
    combined_df = pd.concat(all_dfs, ignore_index=True)
    total_cells = len(combined_df)
    timings['concat_dfs'] = time.time() - t0

    # Count fallback vs valid segmentation
    t0 = time.time()
    fallback_count = int(combined_df["segmentation_id"].isna().sum())
    valid_seg_count = int(combined_df["segmentation_id"].notna().sum())
    timings['count_segmentation'] = time.time() - t0

    # Parse all bboxes at once using vectorized apply
    t0 = time.time()
    combined_df['bbox_parsed'] = combined_df['bbox'].apply(parse_bbox)
    timings['parse_bboxes'] = time.time() - t0

    # Filter out parse errors
    t0 = time.time()
    valid_bbox_df = combined_df[combined_df['bbox_parsed'].notna()].copy()
    parse_errors = total_cells - len(valid_bbox_df)

    if len(valid_bbox_df) == 0:
        return None
    timings['filter_valid'] = time.time() - t0

    # Calculate dimensions vectorized
    t0 = time.time()
    dims = valid_bbox_df['bbox_parsed'].apply(lambda b: calculate_bbox_dimensions(b))
    valid_bbox_df['height'] = dims.apply(lambda x: x[0])
    valid_bbox_df['width'] = dims.apply(lambda x: x[1])
    valid_bbox_df['area'] = valid_bbox_df['height'] * valid_bbox_df['width']
    timings['calculate_dimensions'] = time.time() - t0

    # Remove any NaN dimensions
    t0 = time.time()
    valid_bbox_df = valid_bbox_df.dropna(subset=['height', 'width'])

    if len(valid_bbox_df) == 0:
        return None
    timings['dropna'] = time.time() - t0

    # Find large bboxes
    t0 = time.time()
    large_bbox_mask = (valid_bbox_df['height'] > 1000) | (valid_bbox_df['width'] > 1000)
    large_bbox_df = valid_bbox_df[large_bbox_mask]
    timings['find_large_bboxes'] = time.time() - t0

    # Convert large bbox cells to list of dicts (only if needed for detailed reporting)
    t0 = time.time()
    large_bbox_cells = []
    if len(large_bbox_df) > 0:
        for idx, row in large_bbox_df.iterrows():
            large_bbox_cells.append({
                "well": row["well_short"],
                "row_idx": idx,
                "height": int(row["height"]),
                "width": int(row["width"]),
                "area": int(row["area"]),
                "segmentation_id": row["segmentation_id"],
                "is_fallback": pd.isna(row["segmentation_id"]),
                "bbox": row["bbox_parsed"],
            })
    timings['build_large_bbox_list'] = time.time() - t0

    # Calculate statistics using vectorized operations
    t0 = time.time()
    stats = {
        "exp_name": exp_name,
        "total_cells": total_cells,
        "valid_seg_count": valid_seg_count,
        "fallback_count": fallback_count,
        "parse_errors": parse_errors,
        "large_bbox_count": len(large_bbox_cells),
        "large_bbox_cells": large_bbox_cells,
        "height_mean": float(valid_bbox_df['height'].mean()),
        "height_median": float(valid_bbox_df['height'].median()),
        "height_std": float(valid_bbox_df['height'].std()),
        "height_min": float(valid_bbox_df['height'].min()),
        "height_max": float(valid_bbox_df['height'].max()),
        "height_95th": float(valid_bbox_df['height'].quantile(0.95)),
        "height_99th": float(valid_bbox_df['height'].quantile(0.99)),
        "width_mean": float(valid_bbox_df['width'].mean()),
        "width_median": float(valid_bbox_df['width'].median()),
        "width_std": float(valid_bbox_df['width'].std()),
        "width_min": float(valid_bbox_df['width'].min()),
        "width_max": float(valid_bbox_df['width'].max()),
        "width_95th": float(valid_bbox_df['width'].quantile(0.95)),
        "width_99th": float(valid_bbox_df['width'].quantile(0.99)),
        "area_mean": float(valid_bbox_df['area'].mean()),
        "area_median": float(valid_bbox_df['area'].median()),
        "area_std": float(valid_bbox_df['area'].std()),
        "area_min": float(valid_bbox_df['area'].min()),
        "area_max": float(valid_bbox_df['area'].max()),
        "area_95th": float(valid_bbox_df['area'].quantile(0.95)),
        "area_99th": float(valid_bbox_df['area'].quantile(0.99)),
    }
    timings['calculate_stats'] = time.time() - t0

    timings['total'] = time.time() - t_start

    if verbose:
        print(f"\n[TIMING] {exp_name} ({total_cells:,} cells):")
        for key, val in sorted(timings.items(), key=lambda x: -x[1]):
            pct = (val / timings['total'] * 100) if timings['total'] > 0 else 0
            print(f"  {key:25s}: {val:6.3f}s ({pct:5.1f}%)")

    return stats


def debug_bbox_sizes(skip_low_count_exps: bool = False, verbose_timing: bool = False):
    """
    Main function to analyze bbox sizes across all experiments.
    """
    dummy_dataset = OpsDataset("dummy")

    # Debug: print what we have
    print(f"DEBUG: dummy_dataset type: {type(dummy_dataset)}")
    print(f"DEBUG: has config_paths attr: {hasattr(dummy_dataset, 'config_paths')}")
    if hasattr(dummy_dataset, 'config_paths'):
        print(f"DEBUG: config_paths type: {type(dummy_dataset.config_paths)}")
        print(f"DEBUG: config_paths keys: {list(dummy_dataset.config_paths.keys())[:5]}")
        print(f"DEBUG: 'exp_config_dir' in config_paths: {'exp_config_dir' in dummy_dataset.config_paths}")

    config_root_dir = dummy_dataset.config_paths["exp_config_dir"]

    if not config_root_dir.is_dir():
        print(f"Configuration directory not found at {config_root_dir}. Aborting.")
        return

    config_files = sorted(list(config_root_dir.glob("*_config.yaml")))

    print("\n" + "=" * 100)
    print("BBOX SIZE DEBUG ANALYSIS")
    print("=" * 100 + "\n")

    all_stats = []
    skipped_experiments = []
    low_count_skipped = []
    no_data_experiments = []

    # Track aggregate timings
    exp_count = 0
    total_analyze_time = 0
    total_config_load_time = 0

    for config_path in tqdm(config_files, desc="Analyzing experiments", disable=verbose_timing):
        try:
            t0 = time.time()
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                if not config or "experiment_name" not in config:
                    continue

            exp_name = config["experiment_name"]

            # Skip low-count experiments if flag is set
            if skip_low_count_exps and exp_name in LOW_COUNT_EXPERIMENTS:
                low_count_skipped.append(exp_name)
                continue

            # Filter experiments: only include standard format opsXXXX_YYYYMMDD
            standard_format = re.match(r"^ops\d{4}_\d{8}$", exp_name)
            if not standard_format:
                skipped_experiments.append(exp_name)
                continue

            dataset = OpsDataset(exp_name)
            total_config_load_time += time.time() - t0

            # Analyze bboxes for this experiment
            t0 = time.time()
            stats = analyze_experiment_bboxes(dataset, config, exp_name, verbose=verbose_timing)
            analyze_time = time.time() - t0
            total_analyze_time += analyze_time

            if stats is None:
                no_data_experiments.append(exp_name)
                continue

            all_stats.append(stats)
            exp_count += 1

            if not verbose_timing:
                # Show simple progress update every 10 experiments
                if exp_count % 10 == 0:
                    avg_time = total_analyze_time / exp_count
                    print(f"  Processed {exp_count} experiments, avg time: {avg_time:.2f}s per experiment")

        except Exception as e:
            print(f"\nError processing {config_path.stem}: {e}")
            continue

    # Display summary statistics
    if low_count_skipped:
        print(f"\nExcluded {len(low_count_skipped)} low-count experiments")

    if skipped_experiments:
        print(f"\nSkipped {len(skipped_experiments)} non-standard experiments")

    if no_data_experiments:
        print(f"\nNo linked data found for {len(no_data_experiments)} experiments")

    print(f"\nAnalyzed {len(all_stats)} experiments with bbox data\n")

    # Create summary table
    table = PrettyTable()
    table.field_names = [
        "Experiment",
        "Total Cells",
        "Valid Seg",
        "Fallback",
        "Large BBox (>1000px)",
        "% Large",
        "Height (median)",
        "Width (median)",
        "Area (median)",
        "Height (max)",
        "Width (max)",
    ]
    table.align = "l"
    table.align["Total Cells"] = "r"
    table.align["Valid Seg"] = "r"
    table.align["Fallback"] = "r"
    table.align["Large BBox (>1000px)"] = "r"
    table.align["% Large"] = "r"

    # Sort by percentage of large bboxes descending
    all_stats_sorted = sorted(
        all_stats,
        key=lambda x: (x["large_bbox_count"] / x["total_cells"] * 100) if x["total_cells"] > 0 else 0,
        reverse=True
    )

    for stats in all_stats_sorted:
        pct_large = (stats["large_bbox_count"] / stats["total_cells"] * 100) if stats["total_cells"] > 0 else 0

        table.add_row([
            stats["exp_name"],
            f"{stats['total_cells']:,}",
            f"{stats['valid_seg_count']:,}",
            f"{stats['fallback_count']:,}",
            f"{stats['large_bbox_count']:,}",
            f"{pct_large:.2f}%",
            f"{stats.get('height_median', 0):.0f}" if "height_median" in stats else "N/A",
            f"{stats.get('width_median', 0):.0f}" if "width_median" in stats else "N/A",
            f"{stats.get('area_median', 0):.0f}" if "area_median" in stats else "N/A",
            f"{stats.get('height_max', 0):.0f}" if "height_max" in stats else "N/A",
            f"{stats.get('width_max', 0):.0f}" if "width_max" in stats else "N/A",
        ])

    print(table)
    print("\n" + "=" * 100)

    # Aggregate statistics across all experiments
    total_cells_all = sum(s["total_cells"] for s in all_stats)
    total_large_all = sum(s["large_bbox_count"] for s in all_stats)
    total_fallback_all = sum(s["fallback_count"] for s in all_stats)
    total_valid_seg_all = sum(s["valid_seg_count"] for s in all_stats)

    print("\nAGGREGATE STATISTICS (ALL EXPERIMENTS)")
    print("=" * 100)
    print(f"Total cells analyzed:              {total_cells_all:,}")
    print(f"Cells with valid segmentation:     {total_valid_seg_all:,} ({total_valid_seg_all/total_cells_all*100:.2f}%)")
    print(f"Cells with fallback bbox:          {total_fallback_all:,} ({total_fallback_all/total_cells_all*100:.2f}%)")
    print(f"Cells with large bbox (>1000px):   {total_large_all:,} ({total_large_all/total_cells_all*100:.2f}%)")

    # Calculate percentile statistics across all experiments
    all_heights = []
    all_widths = []
    all_areas = []

    for stats in all_stats:
        if "height_median" in stats:
            # We don't have raw data, but we can use the statistics we collected
            # For a proper distribution we'd need to recollect, but let's show what we have
            pass

    print("\n" + "=" * 100)

    # Identify experiments with anomalies
    print("\nEXPERIMENTS WITH ANOMALIES (>1% cells with large bboxes)")
    print("=" * 100)

    anomaly_table = PrettyTable()
    anomaly_table.field_names = [
        "Experiment",
        "Large BBox Count",
        "% of Total",
        "Max Height",
        "Max Width",
        "Max Area",
    ]
    anomaly_table.align = "l"

    for stats in all_stats_sorted:
        pct_large = (stats["large_bbox_count"] / stats["total_cells"] * 100) if stats["total_cells"] > 0 else 0

        if pct_large > 1.0 or stats["large_bbox_count"] > 100:
            anomaly_table.add_row([
                stats["exp_name"],
                f"{stats['large_bbox_count']:,}",
                f"{pct_large:.2f}%",
                f"{stats.get('height_max', 0):.0f}" if "height_max" in stats else "N/A",
                f"{stats.get('width_max', 0):.0f}" if "width_max" in stats else "N/A",
                f"{stats.get('area_max', 0):.0f}" if "area_max" in stats else "N/A",
            ])

    print(anomaly_table)
    print("\n" + "=" * 100)

    # Show detailed examples from worst offenders
    print("\nDETAILED EXAMPLES (Top 5 experiments with most large bboxes)")
    print("=" * 100)

    for stats in all_stats_sorted[:5]:
        if stats["large_bbox_count"] == 0:
            continue

        print(f"\n{stats['exp_name']}:")
        print(f"  Total cells: {stats['total_cells']:,}")
        print(f"  Large bbox cells: {stats['large_bbox_count']:,}")

        # Show first 10 examples
        for i, cell in enumerate(stats["large_bbox_cells"][:10]):
            seg_type = "FALLBACK" if cell["is_fallback"] else f"seg_id={cell['segmentation_id']}"
            print(f"    [{i+1}] Well={cell['well']}, Row={cell['row_idx']}, "
                  f"Size={cell['height']}x{cell['width']}px (area={cell['area']:,}px²), "
                  f"Type={seg_type}")

        if len(stats["large_bbox_cells"]) > 10:
            print(f"    ... and {len(stats['large_bbox_cells']) - 10} more")

    print("\n" + "=" * 100 + "\n")

    # Save detailed report
    try:
        output_dir = Path(__file__).resolve().parent

        # Save CSV with all statistics
        stats_df = pd.DataFrame([
            {
                "experiment": s["exp_name"],
                "total_cells": s["total_cells"],
                "valid_seg_count": s["valid_seg_count"],
                "fallback_count": s["fallback_count"],
                "large_bbox_count": s["large_bbox_count"],
                "pct_large_bbox": (s["large_bbox_count"] / s["total_cells"] * 100) if s["total_cells"] > 0 else 0,
                "height_mean": s.get("height_mean"),
                "height_median": s.get("height_median"),
                "height_std": s.get("height_std"),
                "height_max": s.get("height_max"),
                "height_95th": s.get("height_95th"),
                "height_99th": s.get("height_99th"),
                "width_mean": s.get("width_mean"),
                "width_median": s.get("width_median"),
                "width_std": s.get("width_std"),
                "width_max": s.get("width_max"),
                "width_95th": s.get("width_95th"),
                "width_99th": s.get("width_99th"),
                "area_mean": s.get("area_mean"),
                "area_median": s.get("area_median"),
                "area_std": s.get("area_std"),
                "area_max": s.get("area_max"),
                "area_95th": s.get("area_95th"),
                "area_99th": s.get("area_99th"),
            }
            for s in all_stats_sorted
        ])

        csv_path = output_dir / "bbox_debug_stats.csv"
        stats_df.to_csv(csv_path, index=False)
        print(f"Saved statistics to: {csv_path}")

        # Save detailed large bbox cells to separate CSV
        large_bbox_rows = []
        for stats in all_stats:
            for cell in stats["large_bbox_cells"]:
                large_bbox_rows.append({
                    "experiment": stats["exp_name"],
                    "well": cell["well"],
                    "row_idx": cell["row_idx"],
                    "height": cell["height"],
                    "width": cell["width"],
                    "area": cell["area"],
                    "segmentation_id": cell["segmentation_id"],
                    "is_fallback": cell["is_fallback"],
                    "bbox": str(cell["bbox"]),
                })

        if large_bbox_rows:
            large_bbox_df = pd.DataFrame(large_bbox_rows)
            large_csv_path = output_dir / "bbox_debug_large_cells.csv"
            large_bbox_df.to_csv(large_csv_path, index=False)
            print(f"Saved large bbox cells to: {large_csv_path}")

    except Exception as e:
        print(f"Warning: Failed to save debug reports: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug bbox sizes across all experiments")
    parser.add_argument(
        "--skip-low-count-exps",
        action="store_true",
        help="Skip low-count experiments (ops0028_20250417, ops0011_20250205, ops0012_20250206, ops0023_20250317)",
    )
    parser.add_argument(
        "--verbose-timing",
        action="store_true",
        help="Print detailed timing breakdown for each experiment",
    )
    args = parser.parse_args()

    debug_bbox_sizes(skip_low_count_exps=args.skip_low_count_exps, verbose_timing=args.verbose_timing)
    # To run: python tests/bbox_size_debug.py
    # To run with skip low-count: python tests/bbox_size_debug.py --skip-low-count-exps
    # To run with timing: python tests/bbox_size_debug.py --verbose-timing
