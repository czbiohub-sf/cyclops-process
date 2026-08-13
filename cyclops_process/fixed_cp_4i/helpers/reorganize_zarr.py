#!/usr/bin/env python3
"""Reorganize 4i zarr stores from flat integer positions to HCS well/grid layout.

Renames positions from `0/<int>/0` to `A/<well_num>/<XXXYYY>` using the
position map JSONs extracted from the NDTiff metadata.

Current:  round5.zarr/0/2398/0   (sequential integer, all in one "well")
Target:   round5.zarr/A/1/022002 (HCS: row A, well 1, grid position 022002)

Also updates the zarr plate metadata (.zattrs) to reflect the new structure.

Usage:
    # Reorganize all zarrs for all rounds
    python -m cyclops_process.fixed_cp_4i.helpers.reorganize_zarr

    # Specific rounds only
    python -m cyclops_process.fixed_cp_4i.helpers.reorganize_zarr --rounds 1 5

    # Dry run to see what would happen
    python -m cyclops_process.fixed_cp_4i.helpers.reorganize_zarr --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

from cyclops_process.fixed_cp_4i.configs.four_i_config import NUM_ROUNDS, get_default_output_dir


def _parse_label(label: str) -> tuple[str, int, str]:
    """Parse 'A1-Site_022002' -> (row='A', well=1, fov='022002')."""
    well_part, site_part = label.split("-Site_")
    row = well_part[0]  # 'A'
    well_num = int(well_part[1:])  # 1, 2, 3
    fov = site_part  # '022002'
    return row, well_num, fov


def reorganize_zarr_store(zarr_path: Path, pos_map: dict, dry_run: bool = False) -> None:
    """Reorganize a single zarr store in-place."""
    zarr_path = Path(zarr_path)
    if not zarr_path.exists():
        print(f"  SKIP: {zarr_path.name} (not found)")
        return

    old_root = zarr_path / "0"

    # Check if already reorganized: A/ exists AND 0/ is gone
    if (zarr_path / "A").exists() and not old_root.exists():
        print(f"  SKIP: {zarr_path.name} (already reorganized)")
        return

    if not old_root.exists():
        print(f"  SKIP: {zarr_path.name} (no '0' directory)")
        return

    print(f"  Reorganizing {zarr_path.name}...")

    # Clean up any leftover A/ dirs from a previous failed run
    a_dir = zarr_path / "A"
    if a_dir.exists():
        import shutil
        shutil.rmtree(a_dir)
        print(f"    Cleaned up leftover A/ from previous run")

    # Parse all positions and build the rename plan
    # old: zarr/0/<idx>  ->  new: zarr/A/<well_num>/<fov>
    renames = []
    wells_seen = set()
    for idx_str, label in pos_map.items():
        row, well_num, fov = _parse_label(label)
        old_dir = zarr_path / "0" / idx_str
        new_dir = zarr_path / row / str(well_num) / fov

        if not old_dir.exists():
            continue

        wells_seen.add((row, well_num))
        renames.append((old_dir, new_dir))

    print(f"    {len(renames)} positions -> wells: {sorted(wells_seen)}")

    if dry_run:
        print(f"    [DRY RUN] Would rename {len(renames)} position directories")
        return

    # Create well directories first
    for row, well_num in sorted(wells_seen):
        well_dir = zarr_path / row / str(well_num)
        well_dir.mkdir(parents=True, exist_ok=True)

    # Rename: 0/<idx> -> A/<well>/<fov>  (instant on same filesystem)
    moved = 0
    for old_dir, new_dir in renames:
        if new_dir.exists():
            print(f"    WARNING: {new_dir} already exists, skipping")
            continue
        old_dir.rename(new_dir)
        moved += 1

    print(f"    Moved {moved}/{len(renames)} positions")

    # Clean up old '0' row directory
    for leftover in old_root.iterdir():
        if leftover.is_file():
            leftover.unlink()
    if not any(old_root.iterdir()):
        old_root.rmdir()
        print(f"    Removed empty '0' directory")
    else:
        remaining = sum(1 for x in old_root.iterdir() if x.is_dir())
        print(f"    WARNING: {remaining} directories still in '0/'")


    # Fix nesting: collapse A/<well>/<fov>/0/ up into A/<well>/<fov>/
    # Original TIFFConverter created: well/<idx>/fov(0)/image(0)/.zarray
    # After rename: A/<well>/<fov>/0/0/.zarray  (extra "0" level)
    # Target:       A/<well>/<fov>/0/.zarray    (standard HCS position)
    #
    # Strategy: rename <fov>/0 -> <fov>/_tmp, delete <fov>'s old files,
    #           move _tmp contents into <fov>/, remove _tmp
    import shutil

    print(f"    Fixing nesting (collapsing extra 0/ level)...")
    fixed = 0
    for old_dir, new_dir in renames:
        fov_dir = new_dir  # e.g. A/1/022002
        inner_fov = fov_dir / "0"  # the extra "0" dir to collapse

        # Only fix if the extra nesting exists
        if not inner_fov.is_dir():
            continue
        if not (inner_fov / "0").exists():
            continue

        # Move inner_fov to temp
        tmp = fov_dir / "_collapse_tmp"
        inner_fov.rename(tmp)

        # Remove old .zattrs/.zgroup at fov level (stale well metadata)
        for f in fov_dir.iterdir():
            if f.is_file():
                f.unlink()

        # Move everything from tmp into fov_dir
        for item in tmp.iterdir():
            item.rename(fov_dir / item.name)
        tmp.rmdir()
        fixed += 1

    print(f"    Collapsed {fixed} positions")

    # Update plate metadata
    _update_plate_metadata(zarr_path, pos_map)
    print(f"    Updated plate metadata")


def _update_plate_metadata(zarr_path: Path, pos_map: dict) -> None:
    """Rewrite .zattrs with correct HCS plate metadata."""
    wells_info = {}  # (row, col) -> list of fov names
    for idx_str, label in pos_map.items():
        row, well_num, fov = _parse_label(label)
        key = (row, well_num)
        if key not in wells_info:
            wells_info[key] = []
        wells_info[key].append(fov)

    # Build OME-Zarr plate metadata
    rows = sorted(set(r for r, _ in wells_info.keys()))
    cols = sorted(set(c for _, c in wells_info.keys()))

    plate = {
        "version": "0.4",
        "acquisitions": [{"id": 0, "name": "4i"}],
        "rows": [{"name": r} for r in rows],
        "columns": [{"name": str(c)} for c in cols],
        "wells": [
            {
                "path": f"{r}/{c}",
                "rowIndex": rows.index(r),
                "columnIndex": cols.index(c),
            }
            for r, c in sorted(wells_info.keys())
        ],
    }

    zattrs_path = zarr_path / ".zattrs"
    with open(zattrs_path, "w") as f:
        json.dump({"plate": plate}, f, indent=2)

    # Write well-level .zattrs for each well (list of images/FOVs)
    for (row, well_num), fovs in wells_info.items():
        well_dir = zarr_path / row / str(well_num)
        well_zattrs = {
            "well": {
                "images": [{"path": fov, "acquisition": 0} for fov in sorted(fovs)],
                "version": "0.4",
            }
        }
        with open(well_dir / ".zattrs", "w") as f:
            json.dump(well_zattrs, f, indent=2)
        # Ensure .zgroup exists
        zgroup_path = well_dir / ".zgroup"
        if not zgroup_path.exists():
            with open(zgroup_path, "w") as f:
                json.dump({"zarr_format": 2}, f)

    # Ensure row-level .zgroup exists
    for row in rows:
        row_dir = zarr_path / row
        zgroup_path = row_dir / ".zgroup"
        if not zgroup_path.exists():
            with open(zgroup_path, "w") as f:
                json.dump({"zarr_format": 2}, f)


def fix_nesting(zarr_path: Path, pos_map: dict) -> None:
    """Collapse extra 0/ nesting on already-reorganized zarrs.

    Current:  A/1/022002/0/0/.zarray  (position has extra "0" from old FOV)
    Target:   A/1/022002/0/.zarray    (standard HCS: row/well/fov/image)

    For each position, moves contents of <fov>/0/ up into <fov>/ and
    replaces the stale well-level .zattrs with the position-level metadata.
    """
    import shutil

    zarr_path = Path(zarr_path)
    if not zarr_path.exists():
        print(f"  SKIP: {zarr_path.name} (not found)")
        return

    print(f"  Fixing nesting in {zarr_path.name}...")
    fixed = 0
    skipped = 0

    for idx_str, label in pos_map.items():
        row, well_num, fov = _parse_label(label)
        fov_dir = zarr_path / row / str(well_num) / fov

        if not fov_dir.exists():
            skipped += 1
            continue

        inner = fov_dir / "0"  # the extra "0" FOV dir to collapse
        if not inner.is_dir():
            skipped += 1
            continue

        # Check if nesting needs fixing: <fov>/0/0/.zarray exists (extra level)
        # If <fov>/0/.zarray exists instead, it's already collapsed — skip
        if (inner / ".zarray").exists():
            skipped += 1
            continue
        if not (inner / "0" / ".zarray").exists():
            skipped += 1
            continue

        # Move inner "0/" to a temp name
        tmp = fov_dir / "_tmp_collapse"
        if tmp.exists():
            import shutil
            shutil.rmtree(tmp)
        inner.rename(tmp)

        # Remove stale files at fov level (old well .zattrs, .zgroup)
        for f in fov_dir.iterdir():
            if f.is_file():
                f.unlink()

        # Move contents of tmp/ into fov_dir/
        for item in tmp.iterdir():
            item.rename(fov_dir / item.name)
        tmp.rmdir()
        fixed += 1

    print(f"    Collapsed {fixed} positions, skipped {skipped}")

    # Refresh plate metadata
    _update_plate_metadata(zarr_path, pos_map)
    print(f"    Updated plate metadata")


def main():
    parser = argparse.ArgumentParser(
        description="Reorganize 4i zarr stores to proper HCS well/grid layout",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing zarr stores (default: <experiment>/0-convert/4i/)",
    )
    parser.add_argument(
        "--rounds",
        nargs="+",
        type=int,
        default=list(range(1, NUM_ROUNDS + 1)),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate", action="store_true", help="Validate reorganized zarrs")
    parser.add_argument("--fix-nesting", action="store_true",
                        help="Collapse extra 0/ nesting on already-reorganized zarrs")

    args = parser.parse_args()
    input_dir = (
        Path(args.input_dir).expanduser().resolve()
        if args.input_dir
        else get_default_output_dir()
    )

    for rnd in args.rounds:
        map_path = input_dir / f"round{rnd}_position_map.json"
        if not map_path.exists():
            print(f"WARNING: No position map for round {rnd}: {map_path}")
            continue

        with open(map_path) as f:
            pos_map = json.load(f)

        print(f"\nRound {rnd}: {len(pos_map)} positions")

        for suffix in ["", "_max_proj", "_max_proj_flatfield"]:
            zarr_path = input_dir / f"round{rnd}{suffix}.zarr"
            if args.validate:
                validate_zarr_store(zarr_path, pos_map)
            elif args.fix_nesting:
                fix_nesting(zarr_path, pos_map)
            else:
                reorganize_zarr_store(zarr_path, pos_map, dry_run=args.dry_run)


def validate_zarr_store(zarr_path: Path, pos_map: dict) -> bool:
    """Validate a reorganized zarr store against the position map.

    Checks:
    1. No leftover '0/' directory
    2. All expected positions exist at the correct HCS path
    3. Each position has a zarr array (0/ subdir with .zarray)
    4. No unexpected files/dirs at the well level
    5. Plate metadata (.zattrs) is valid
    """
    zarr_path = Path(zarr_path)
    if not zarr_path.exists():
        print(f"  SKIP: {zarr_path.name} (not found)")
        return True

    print(f"  Validating {zarr_path.name}...")
    errors = []
    warnings = []

    # 1. No leftover '0/' directory
    if (zarr_path / "0").exists():
        remaining = sum(1 for x in (zarr_path / "0").iterdir() if x.is_dir())
        errors.append(f"Old '0/' directory still exists with {remaining} subdirs")

    # 2. Check all expected positions exist
    expected_wells = {}  # (row, well) -> [fov, ...]
    for idx_str, label in pos_map.items():
        row, well_num, fov = _parse_label(label)
        key = (row, well_num)
        if key not in expected_wells:
            expected_wells[key] = []
        expected_wells[key].append(fov)

    missing = 0
    no_array = 0
    total = 0
    for (row, well_num), fovs in expected_wells.items():
        well_dir = zarr_path / row / str(well_num)
        if not well_dir.exists():
            errors.append(f"Well {row}/{well_num} directory missing")
            missing += len(fovs)
            continue

        for fov in fovs:
            total += 1
            fov_dir = well_dir / fov
            if not fov_dir.exists():
                missing += 1
                continue

            # 3. Check for zarr array
            # After fix-nesting: <fov>/0/.zarray (correct)
            # Before fix-nesting: <fov>/0/0/.zarray (extra level)
            zarray = fov_dir / "0" / ".zarray"
            zarray_nested = fov_dir / "0" / "0" / ".zarray"
            if zarray.exists():
                pass  # correct
            elif zarray_nested.exists():
                no_array += 1  # count as "needs fix-nesting"
            else:
                no_array += 1

    # 4. Check for unexpected positions in wells
    unexpected = 0
    expected_fov_sets = {
        (row, well_num): set(fovs)
        for (row, well_num), fovs in expected_wells.items()
    }
    for (row, well_num) in expected_wells:
        well_dir = zarr_path / row / str(well_num)
        if not well_dir.exists():
            continue
        actual_fovs = set(
            d.name for d in well_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        extra = actual_fovs - expected_fov_sets[(row, well_num)]
        unexpected += len(extra)
        if extra:
            warnings.append(f"Well {row}/{well_num}: {len(extra)} unexpected positions")

    # 5. Check plate metadata
    zattrs_path = zarr_path / ".zattrs"
    if zattrs_path.exists():
        with open(zattrs_path) as f:
            attrs = json.load(f)
        plate = attrs.get("plate", {})
        meta_wells = len(plate.get("wells", []))
        meta_rows = len(plate.get("rows", []))
        meta_cols = len(plate.get("columns", []))
    else:
        errors.append("Missing .zattrs (plate metadata)")
        meta_wells = meta_rows = meta_cols = 0

    # Summary
    ok = len(errors) == 0 and missing == 0
    status = "OK" if ok else "FAILED"
    print(f"    [{status}] {total} positions, {len(expected_wells)} wells")
    print(f"    Positions: {total - missing} present, {missing} missing, {no_array} missing array")
    print(f"    Metadata: {meta_wells} wells, {meta_rows} rows, {meta_cols} cols")
    if unexpected:
        print(f"    Unexpected: {unexpected} extra positions")
    for e in errors:
        print(f"    ERROR: {e}")
    for w in warnings:
        print(f"    WARN: {w}")

    return ok


if __name__ == "__main__":
    main()
