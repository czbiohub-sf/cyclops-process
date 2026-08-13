"""Read-only audit layer for v3 pyramid stores.

Low-level readers (_read_*), audit checks (_audit_*), seg-label diagnosis,
the audit_v3_stores driver, and report printers. Imported by audit_fix.py
(which owns the write/fix side)."""
from pathlib import Path
from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.io.zarr_utils import (
    _iter_position_paths,
    _discover_last_position_per_well,
    detect_zarr_format,
    level_has_data,
)
from cyclops_utils.data.filesystem import resolve_experiment_name
from cyclops_process.processes.pyramids.build_drivers import _get_store_path


# Expected structure per store type (v3 zarr).
# Each entry: (label_name, min_pyramid_levels)
_EXPECTED_LABELS = {
    "pheno": {
        "nuclear_seg": 5,
        "cell_seg": 5,
        "iss_gene_image": 5,
        "iss_guide_image": 5,
        "grid_overlay": 5,
    },
    "iss": {
        "nuclear_seg": 5,
        "grid_overlay": 5,
    },
    "track": {
        "nuclear_seg": 5,
        "grid_overlay": 5,
    },
}
# Expected base image pyramid levels per store
_EXPECTED_IMAGE_LEVELS = 5
# OPS phenotyping native YX pixel size (µm/px). Pre-fix v2 → v3 assembly
# wrote 0.65 here, a 2× over-declaration. The Dragonfly raw acquisition is
# 0.325 µm/px and the assembled grid is at native resolution (the level-0
# array was never spatially binned), so 0.325 is the correct value to store
# at level 0 of every phenotyping_v3 multiscale.
PHENO_NATIVE_YX = 0.325
PHENO_BUGGY_YX = 0.65
# Track (5x) native YX is 4x pheno (20x): 0.325 * 4 = 1.3 µm/px. A regression
# left a window of stitched track stores with level-0 halved to 0.65 (and a
# malformed coarsest pyramid level), the same 0.65 buggy value pheno hits.
TRACK_NATIVE_YX = 1.3
# Canonical pyramid downsample factor (each level halves YX resolution).
PYRAMID_DOWNSAMPLE = 2.0
def _read_level0_yx_scale(pos_path: Path) -> tuple | None:
    """Read level-0 YX scale from a position's `zarr.json` (OME-Zarr v3) or
    `.zattrs` (v2). Returns (y, x) or None if not found / malformed."""
    levels = _read_all_levels_yx_scales(pos_path)
    return levels[0] if levels else None
def _read_all_levels_yx_scales(pos_path: Path) -> list[tuple] | None:
    """Read per-level YX scales from a position's metadata.

    Covers all OME-NGFF spots a Y/X scale can live in:
      1. ``multiscales[0].datasets[i].coordinateTransformations`` — per-level
         scale (the standard place, where 99% of stores keep it).
      2. ``multiscales[0].coordinateTransformations`` — top-level transform
         that applies to ALL datasets in the multiscales group; rarely used
         but valid per the OME-NGFF spec. If present, its YX is multiplied
         into each per-level scale.
      3. Per-level array's own ``.zattrs`` / ``zarr.json`` — not currently
         used by our pipeline so we skip it; readers should follow the
         multiscales[] datasets metadata, not array-level attrs.

    Returns ``[(y, x), …]`` ordered by level (i.e. coarsening as i grows),
    or None if not found / malformed.
    """
    import json
    for fname in ("zarr.json", ".zattrs"):
        p = pos_path / fname
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        attrs = d.get("attributes", d)
        ome = attrs.get("ome", attrs)
        ms = ome.get("multiscales") or attrs.get("multiscales")
        if not ms:
            continue
        try:
            # Optional top-level transform (applies to every dataset).
            top_yx = (1.0, 1.0)
            for ct in (ms[0].get("coordinateTransformations") or []):
                if ct.get("type") == "scale":
                    s = ct.get("scale", [])
                    if len(s) >= 2:
                        top_yx = (float(s[-2]), float(s[-1]))
                        break
            out = []
            for ds in ms[0]["datasets"]:
                s = ds["coordinateTransformations"][0]["scale"]
                out.append((float(s[-2]) * top_yx[0], float(s[-1]) * top_yx[1]))
            return out if out else None
        except (KeyError, IndexError, TypeError):
            continue
    return None
def _read_multiscale_level_paths(pos_path: Path) -> list[str] | None:
    """Return the list of dataset `path`s declared in a position's multiscales
    metadata (OME-Zarr v3 `zarr.json` or v2 `.zattrs`), or None if not found.

    napari (and other NGFF readers) discover pyramid levels from this list —
    NOT from on-disk level dirs. A store whose level arrays exist on disk but
    are absent from this list renders as having no/too-few levels.
    """
    import json
    for fname in ("zarr.json", ".zattrs"):
        p = pos_path / fname
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        attrs = d.get("attributes", d)
        ome = attrs.get("ome", attrs)
        ms = ome.get("multiscales") or attrs.get("multiscales")
        if not ms:
            continue
        try:
            return [str(ds["path"]) for ds in ms[0]["datasets"]]
        except (KeyError, IndexError, TypeError):
            continue
    return None
def _audit_yx_scale(
    store_path: Path,
    positions: list[str],
    native_yx: float = PHENO_NATIVE_YX,
    buggy_yx: float = PHENO_BUGGY_YX,
) -> dict:
    """Read YX scale at EVERY pyramid level across positions and classify.

    Verifies each level i has Y/X == ``native_yx * PYRAMID_DOWNSAMPLE**i`` (the
    canonical pyramid). Catches both level-0 regressions AND the malformed
    coarsest-level case where level 0 looks right but a later level was left
    at a stale value (or vice versa). Per-position scales are pulled by
    :func:`_read_all_levels_yx_scales`, which covers the standard
    multiscales[].datasets and the rarely-used top-level transform.

    Returns one of:
      {"status": "correct",  "observed": (y, x)}    — every level matches canonical
      {"status": "wrong",    "observed": (buggy, buggy)}  — every level uniformly buggy
      {"status": "malformed","per_position": {pos: [(y0,x0), ...], ...}}
                                                    — at least one level off canonical
      {"status": "mixed",    "per_position": {pos: (y, x), ...}}
      {"status": "unknown",  "observed": (y, x)}
      {"status": "skipped"}  — no positions / no metadata
    """
    if not positions:
        return {"status": "skipped"}
    per_pos_l0: dict[str, tuple] = {}
    malformed: dict[str, list] = {}
    for pos in positions:
        levels = _read_all_levels_yx_scales(store_path / pos)
        if levels is None:
            continue
        per_pos_l0[pos] = levels[0]
        # Verify every level is on the canonical curve.
        for i, (y, x) in enumerate(levels):
            expected = native_yx * (PYRAMID_DOWNSAMPLE ** i)
            if not (y == expected and x == expected):
                malformed[pos] = levels
                break
    if not per_pos_l0:
        return {"status": "skipped"}
    uniques = set(per_pos_l0.values())
    if len(uniques) > 1:
        return {"status": "mixed", "per_position": per_pos_l0,
                "malformed": malformed}
    only = next(iter(uniques))
    if not malformed and only == (native_yx, native_yx):
        return {"status": "correct", "observed": only}
    if malformed and only == (native_yx, native_yx):
        return {"status": "malformed", "observed": only,
                "per_position": malformed}
    if only == (buggy_yx, buggy_yx):
        return {"status": "wrong", "observed": only,
                "malformed": malformed}
    return {"status": "unknown", "observed": only, "malformed": malformed}
def _read_normalization_presence(pos_path: Path) -> tuple | None:
    """Return (has_top_level, has_custom_metadata) for a position's normalization.

    Reads the position-level `zarr.json` (v3) or `.zattrs` (v2). `has_top_level`
    is True when a non-empty top-level `normalization` exists (what viscy
    preprocess writes); `has_custom_metadata` is True when
    `custom_metadata.normalization` exists (the v3 schema convert_v3.py produced,
    which portal/napari/audit readers depend on). Returns None if no metadata
    file is found/parsed (position is then excluded from the audit).
    """
    import json
    for fname in ("zarr.json", ".zattrs"):
        p = pos_path / fname
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        attrs = d.get("attributes", d)
        has_top = bool(attrs.get("normalization"))
        has_cm = bool(attrs.get("custom_metadata", {}).get("normalization"))
        return (has_top, has_cm)
    return None
def _audit_normalization(store_path: Path, positions: list[str]) -> dict:
    """Classify normalization-metadata placement across positions.

    viscy preprocess writes a top-level `normalization` field; the v3 schema
    established by convert_v3.py also mirrors it at position-level
    `custom_metadata.normalization`, which portal/napari/audit readers depend
    on. This catches v3-native stores where that mirror is missing — analogous
    to the YX-scale check.

    Returns one of:
      {"status": "correct"}                  every position has custom_metadata.normalization
      {"status": "needs_mirror", "n": int}   top-level present but custom_metadata mirror missing (cheap in-place fix)
      {"status": "missing", "n": int}        normalization absent entirely on some positions (needs viscy_normalize)
      {"status": "skipped"}                  no positions / no readable metadata
    """
    if not positions:
        return {"status": "skipped"}
    presence = []
    for pos in positions:
        r = _read_normalization_presence(store_path / pos)
        if r is not None:
            presence.append(r)
    if not presence:
        return {"status": "skipped"}
    n_total = len(presence)
    n_cm = sum(1 for _top, cm in presence if cm)
    n_top_only = sum(1 for top, cm in presence if top and not cm)
    n_neither = sum(1 for top, cm in presence if not top and not cm)
    if n_cm == n_total:
        return {"status": "correct"}
    if n_neither == 0:
        # Every position has at least the top-level field; only the
        # custom_metadata mirror is missing → cheap metadata-only fix.
        return {"status": "needs_mirror", "n": n_top_only}
    return {"status": "missing", "n": n_neither}
def _audit_clims(pos_path: Path) -> dict:
    """Check contrast limits coverage for all image channels at a position.

    Clims are stored at the position-level zarr.json under
    ``attributes.clims_per_level.<level>.contrast_limits_per_channel`` (v3 zarr),
    or at ``.zattrs`` with the same keys (v2 zarr).

    Returns:
        {"complete": bool, "n_channels": int, "n_with_clims": int, "detail": str}
    """
    import json
    import dask.array as da

    result = {"complete": False, "n_channels": 0, "n_with_clims": 0, "detail": "no data"}

    level0_path = pos_path / "0"
    if not level0_path.exists():
        return result

    # Get number of channels from the array
    try:
        arr = da.from_zarr(str(pos_path.parent), component=f"{pos_path.name}/0")
        n_channels = arr.shape[1] if len(arr.shape) > 1 else 1
        result["n_channels"] = n_channels
    except Exception:
        result["detail"] = "cannot read array shape"
        return result

    # Look for clims in position-level metadata (v3: zarr.json, v2: .zattrs)
    clims_per_level = {}
    clims_data = None  # level-0 clims for validation
    for meta_file in ["zarr.json", ".zattrs"]:
        meta_path = pos_path / meta_file
        if not meta_path.exists():
            continue
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            attrs = meta if meta_file == ".zattrs" else meta.get("attributes", {})

            # v3 style: clims_per_level.<level>.contrast_limits_per_channel
            clims_per_level = attrs.get("clims_per_level", {})
            if clims_per_level:
                level0_clims = clims_per_level.get("0", {})
                if "contrast_limits_per_channel" in level0_clims:
                    clims_data = level0_clims["contrast_limits_per_channel"]
                    break

            # Flat style: contrast_limits_per_channel directly in attrs
            if "contrast_limits_per_channel" in attrs:
                clims_data = attrs["contrast_limits_per_channel"]
                break
        except Exception:
            pass

    if clims_data is None:
        result["detail"] = f"no clims ({n_channels} channels)"
        return result

    n_with_clims = len(clims_data) if isinstance(clims_data, (list, dict)) else 0
    result["n_with_clims"] = n_with_clims

    if n_with_clims < n_channels:
        result["detail"] = f"only {n_with_clims}/{n_channels} channels have clims"
        return result

    # Check that clims exist for ALL pyramid levels, not just level 0
    expected_levels = _EXPECTED_IMAGE_LEVELS
    levels_with_clims = [
        lvl for lvl in range(expected_levels)
        if str(lvl) in clims_per_level
        and "contrast_limits_per_channel" in clims_per_level.get(str(lvl), {})
    ]
    if len(levels_with_clims) < expected_levels:
        missing_levels = [l for l in range(expected_levels) if l not in levels_with_clims]
        result["detail"] = (
            f"clims only for levels {levels_with_clims}, "
            f"missing levels {missing_levels}"
        )
        return result

    # Validate clim values against actual data to detect stale/bad clims
    from cyclops_process.napari.dask.channel_clims import validate_clims

    stale_channels = validate_clims(pos_path, clims_data)
    if stale_channels:
        result["complete"] = False
        result["stale_channels"] = stale_channels
        result["detail"] = (
            f"{n_with_clims}/{n_channels} channels, "
            f"but channels {stale_channels} have stale/bad clims"
        )
    else:
        result["complete"] = True
        result["detail"] = f"{n_with_clims}/{n_channels} channels, {expected_levels} levels"

    return result
def _diagnose_seg_label_issue(
    label_name: str,
    label_path: Path,
    dataset,
    experiment: str,
    store_type: str,
    positions: list,
    n_levels: int,
    expected_levels: int,
    store_result: dict,
    fix_commands: list,
) -> None:
    """Diagnose and generate fix commands for nuclear_seg/cell_seg pyramid issues.

    Handles several failure modes:
    - v3 level 0 exists but has zero data (v2→v3 conversion incomplete)
    - v2 source has data → reconvert
    - v2 symlink is broken → re-attach + reconvert
    - Failed tiled upscale (a615332 regression) → re-upscale + reconvert
    - v2 source missing → symlink + reconvert
    - Pyramids missing but level 0 has data → reconvert + rebuild
    """
    level0_dir = label_path / "0"
    level0_exists_empty = (
        label_name == "nuclear_seg"
        and level_has_data(level0_dir, check_pixels=False)
        and not level_has_data(level0_dir, check_pixels=True, level_index=0)
    )

    if level0_exists_empty:
        # Level 0 exists but has zero pixels — need to fix the source data
        v2_store_path = _get_store_path(dataset, zarr_version=2, store_type=store_type)
        v2_has_label = False
        if v2_store_path and v2_store_path.exists():
            for well_key in positions[:1]:
                for candidate in [
                    v2_store_path / well_key / "0" / "labels" / label_name / "0",
                    v2_store_path / well_key / "0" / label_name / "0",
                    v2_store_path / well_key / "0" / "seg" / "0",
                ]:
                    if candidate.exists() and level_has_data(candidate, check_pixels=True, level_index=0):
                        v2_has_label = True
                        break

        if v2_has_label:
            store_result["missing"].append(
                f"labels/{label_name} source data is zeros — v2→v3 conversion incomplete"
            )
            fix_commands.append(
                f"uv run python -m cyclops_process.convert.v3_livecell "
                f"--experiment {experiment} --only-labels {label_name} "
                f"--mode {store_type} --source-zarr-version 3 --force"
            )
            return

        # v2 source also empty — for pheno nuclear_seg, re-upscale fixes all variants
        # (broken symlinks, failed tiled upscale, stale metadata). For other stores, symlink + reconvert.
        if label_name == "nuclear_seg" and store_type == "pheno":
            store_result["missing"].append(
                f"labels/{label_name} source data is empty — re-running upscale + reconvert"
            )
            fix_commands.append(
                f"uv run python -m cyclops_process.processes.run -e {experiment} "
                f"--rerun upscale_nuclear_segmentations --force"
            )
            fix_commands.append(
                f"uv run python -m cyclops_process.convert.v3_livecell "
                f"--experiment {experiment} --only-labels {label_name} "
                f"--mode {store_type} --source-zarr-version 3 --force"
            )
        else:
            store_result["missing"].append(
                f"labels/{label_name} source data is zeros — v2 source empty/missing"
            )
            fix_commands.append(
                f"uv run python -m cyclops_process.utils.batch.batch_symlink_nuclear_seg "
                f"--experiments {experiment} --symlink-target {store_type} --force "
                f"--zarr-version 3"
            )
            fix_commands.append(
                f"uv run python -m cyclops_process.convert.v3_livecell "
                f"--experiment {experiment} --only-labels {label_name} "
                f"--mode {store_type} --source-zarr-version 3 --force"
        )
    else:
        # Pyramids incomplete or missing entirely
        level0_dir = label_path / "0"
        level0_has_data = level0_dir.exists() and level_has_data(level0_dir, check_pixels=True, level_index=0)

        # For pheno nuclear_seg: if level 0 is missing, empty, or a group → re-upscale + reconvert
        if label_name == "nuclear_seg" and store_type == "pheno" and not level0_has_data:
            store_result["missing"].append(
                f"labels/{label_name} missing or empty — re-running upscale + reconvert"
            )
            fix_commands.append(
                f"uv run python -m cyclops_process.processes.run -e {experiment} "
                f"--rerun upscale_nuclear_segmentations --force"
            )
            fix_commands.append(
                f"uv run python -m cyclops_process.convert.v3_livecell "
                f"--experiment {experiment} --only-labels {label_name} "
                f"--mode {store_type} --source-zarr-version 3 --force"
            )
        elif level0_has_data:
            # Level 0 has data in v3 — but check if v2 source is also valid before building
            # (if v2 source is broken, pyramid build will just copy zeros)
            v2_store_path = _get_store_path(dataset, zarr_version=2, store_type=store_type)
            v2_source_valid = False
            if v2_store_path and v2_store_path.exists():
                for well_key in positions[:1]:
                    v2_label = v2_store_path / well_key / label_name / "0"
                    if v2_label.exists() and level_has_data(v2_label, check_pixels=True, level_index=0):
                        v2_source_valid = True

            if v2_source_valid:
                # v2 source has data — build pyramids in v2, then reconvert to v3
                store_result["missing"].append(
                    f"labels/{label_name} pyramids ({n_levels}/{expected_levels} levels)"
                )
                fix_commands.append(
                    f"uv run python -m cyclops_process.processes.pyramids.audit_fix "
                    f"--seg-pyramids --seg-types {label_name} "
                    f"--store {store_type} --zarr-version 2 --no-resume -e {experiment}"
                )
                fix_commands.append(
                    f"uv run python -m cyclops_process.convert.v3_livecell "
                    f"--experiment {experiment} --only-labels {label_name} "
                    f"--mode {store_type} --source-zarr-version 3 --force"
                )
            elif label_name == "nuclear_seg" and store_type == "pheno":
                # v2 source is broken — re-upscale + reconvert
                store_result["missing"].append(
                    f"labels/{label_name} v2 source empty — re-running upscale + reconvert"
                )
                fix_commands.append(
                    f"uv run python -m cyclops_process.processes.run -e {experiment} "
                    f"--rerun upscale_nuclear_segmentations --force"
                )
                fix_commands.append(
                    f"uv run python -m cyclops_process.convert.v3_livecell "
                    f"--experiment {experiment} --only-labels {label_name} "
                    f"--mode {store_type} --source-zarr-version 3 --force"
                )
            else:
                store_result["missing"].append(
                    f"labels/{label_name} pyramids missing, v2 source empty"
                )
        else:
            # Level 0 missing/empty for non-pheno-nuclear_seg — reconvert from v2
            store_result["missing"].append(
                f"labels/{label_name} level 0 missing or empty"
            )
            fix_commands.append(
                f"uv run python -m cyclops_process.convert.v3_livecell "
                f"--experiment {experiment} --only-labels {label_name} "
                f"--mode {store_type} --source-zarr-version 3 --force"
            )
def audit_v3_stores(
    experiment: str,
    verbose: bool = True,
    store_path_override: Path | None = None,
    store_type_override: str | None = None,
) -> dict:
    """Audit all v3 zarr stores for an experiment and report missing components.

    Checks for each store (pheno, iss, track):
      - Store exists
      - Wells present
      - Base image pyramids (levels 0-4)
      - Label layers with pyramids (nuclear_seg, cell_seg, iss overlays, grid)
      - Contrast limits (clims)

    Args:
        experiment: Experiment name or shorthand
        verbose: Print detailed report
        store_path_override: If set, audit this exact zarr path instead of resolving
            from the experiment config. Only the store type given by
            store_type_override (default "pheno") is audited; the other two are skipped.
        store_type_override: Which store type the override path represents
            ("pheno", "iss", or "track"). Defaults to "pheno".

    Returns:
        Dict with per-store audit results:
        {
            "pheno": {"exists": bool, "wells": [...], "missing": [...], "ok": [...]},
            "iss": {...},
            "track": {...},
            "fix_commands": [str, ...],  # CLI commands to fix issues
        }
    """
    dataset = OpsDataset(experiment)
    results = {}
    fix_commands = []

    if store_path_override is not None:
        store_types = [store_type_override or "pheno"]
    else:
        store_types = ["pheno", "iss", "track"]

    for store_type in store_types:
        if store_path_override is not None:
            store_path = Path(store_path_override)
        else:
            store_path = _get_store_path(dataset, zarr_version=3, store_type=store_type)
        store_result = {
            "exists": False,
            "path": str(store_path) if store_path else None,
            "wells": [],
            "ok": [],
            "missing": [],
            "warnings": [],
        }

        if not store_path or not store_path.exists():
            store_result["missing"].append("store does not exist")
            results[store_type] = store_result
            continue

        store_result["exists"] = True

        # Discover wells/positions
        positions = _iter_position_paths(store_path)
        if not positions:
            # Fallback for v3 zarr
            try:
                store_p = Path(store_path)
                positions = []
                for row_dir in sorted(store_p.iterdir()):
                    if row_dir.is_dir() and row_dir.name.isalpha():
                        for col_dir in sorted(row_dir.iterdir()):
                            if col_dir.is_dir() and col_dir.name.isdigit():
                                for fov_dir in sorted(col_dir.iterdir()):
                                    if fov_dir.is_dir() and fov_dir.name.isdigit():
                                        positions.append(
                                            f"{row_dir.name}/{col_dir.name}/{fov_dir.name}"
                                        )
            except Exception:
                pass

        # Filter to only wells configured for this experiment
        positions = _filter_to_configured_wells(experiment, positions)

        store_result["wells"] = positions

        if not positions:
            store_result["missing"].append("no positions found")
            results[store_type] = store_result
            continue

        # --- Base image pyramids: check ALL positions ---
        pos = positions[0]
        pos_path = store_path / pos

        import shutil
        failed_reshards = []
        positions_with_empty_levels = []  # (pos, level) pairs with missing/empty data

        for scan_pos in positions:
            scan_pos_path = store_path / scan_pos
            is_track_well2 = store_type == "track" and "/2/" in scan_pos

            for level in range(_EXPECTED_IMAGE_LEVELS):
                level_path = scan_pos_path / str(level)
                temp_path = scan_pos_path / f"{level}_resharding_temp"

                # Check for incomplete reshards
                if temp_path.exists():
                    print(f"    [DETECTED] Incomplete reshard at {scan_pos}/{level} (leftover temp)")
                    failed_reshards.append((scan_pos, level))
                elif level_path.exists() and not (level_path / "zarr.json").exists() and not (level_path / ".zarray").exists():
                    print(f"    [DETECTED] Incomplete reshard at {scan_pos}/{level} (missing zarr.json)")
                    failed_reshards.append((scan_pos, level))
                elif is_track_well2:
                    # Track well 2 (A/2) may have no t=0 data — skip all pixel checks
                    if level == 0:
                        print(f"    [SKIP] {scan_pos}/{level} — track well 2 (t=0 may be empty)")
                elif not level_has_data(level_path, check_pixels=True, level_index=level):
                    print(f"    [DETECTED] Empty/missing data at {scan_pos}/{level}")
                    positions_with_empty_levels.append((scan_pos, level))

        if failed_reshards:
            store_result["failed_reshards"] = failed_reshards

        # Determine which levels are fully OK across all positions
        all_bad = set(l for _, l in failed_reshards) | set(l for _, l in positions_with_empty_levels)
        image_levels_found = [l for l in range(_EXPECTED_IMAGE_LEVELS) if l not in all_bad]

        missing_levels = [l for l in range(_EXPECTED_IMAGE_LEVELS) if l not in image_levels_found]
        if missing_levels:
            store_result["missing"].append(
                f"base image pyramid levels {missing_levels}"
            )
            fix_commands.append(
                f"uv run python -m cyclops_process.processes.pyramids.audit_fix "
                f"--base-image --store {store_type} -e {experiment}"
            )
        elif failed_reshards:
            store_result["missing"].append(
                f"incomplete reshards: {failed_reshards}"
            )
        else:
            # Levels present — check per-timepoint coverage at level 1.
            # ISS cycle images are sparse (bright spots on dark background), so a single
            # center patch is unreliable. Sample a 4x4 grid across the full image extent;
            # a timepoint is considered missing only if ALL 16 samples are zero.
            missing_tps = []
            try:
                import dask.array as da
                import numpy as np
                arr = da.from_zarr(str(store_path), component=f"{pos}/1")
                if arr.ndim >= 5 and arr.shape[0] > 1:
                    H, W = int(arr.shape[-2]), int(arr.shape[-1])
                    ps = min(64, H, W)
                    # 4x4 grid of sample origins spaced evenly across the image
                    ys = [int(H * (i + 0.5) / 4) - ps // 2 for i in range(4)]
                    xs = [int(W * (j + 0.5) / 4) - ps // 2 for j in range(4)]
                    ys = [max(0, min(y, H - ps)) for y in ys]
                    xs = [max(0, min(x, W - ps)) for x in xs]
                    n_c = int(arr.shape[1])
                    for t in range(int(arr.shape[0])):
                        has_data = any(
                            np.count_nonzero(arr[t, c, 0, y:y+ps, x:x+ps].compute()) > 0
                            for y in ys for x in xs for c in range(n_c)
                        )
                        if not has_data:
                            missing_tps.append(t)
            except Exception:
                pass

            if missing_tps:
                store_result["missing"].append(
                    f"base image pyramids missing timepoints {missing_tps} (level 1 all-zero)"
                )
                fix_commands.append(
                    f"uv run python -m cyclops_process.processes.pyramids.audit_fix "
                    f"--base-image --store {store_type} --no-resume -e {experiment}"
                )
            else:
                store_result["ok"].append(
                    f"base image pyramids (levels 0-{_EXPECTED_IMAGE_LEVELS - 1})"
                )

        # --- Multiscale metadata completeness — check ALL positions ---
        # The pyramid check above only confirms level DIRS exist on disk; napari
        # discovers levels from the position multiscales `datasets` list. A build
        # that wrote the level arrays but never registered them in metadata (e.g.
        # an interrupted run) lists only level 0 — napari then skips the layer
        # ("No multiscale levels found"). Catch that mismatch here.
        incomplete_multiscales = []
        for scan_pos in positions:
            listed = _read_multiscale_level_paths(store_path / scan_pos)
            if listed is None:
                continue
            on_disk = [
                str(l) for l in range(_EXPECTED_IMAGE_LEVELS)
                if (store_path / scan_pos / str(l)).exists()
            ]
            # Flag when fewer levels are registered in metadata than exist on disk.
            if len(listed) < len(on_disk):
                incomplete_multiscales.append(
                    (scan_pos, len(listed), len(on_disk))
                )
        if incomplete_multiscales:
            ex_pos, n_listed, n_disk = incomplete_multiscales[0]
            store_result["missing"].append(
                f"multiscale metadata lists {n_listed} level(s) but {n_disk} exist "
                f"on disk (e.g. {ex_pos}); napari will skip these layers "
                f"[{len(incomplete_multiscales)} position(s)]"
            )
            fix_commands.append(
                f"uv run python -m cyclops_process.processes.pyramids.audit_fix "
                f"--base-image --store {store_type} --no-resume -e {experiment}"
            )
        else:
            store_result["ok"].append("multiscale metadata lists all levels")

        # --- Contrast limits (per-channel) — check ALL positions ---
        clims_missing_positions = []
        clims_n_channels = 0
        for _pos in positions:
            _pos_path = store_path / _pos
            _clims_status = _audit_clims(_pos_path)
            clims_n_channels = max(clims_n_channels, _clims_status["n_channels"])
            if not _clims_status["complete"]:
                clims_missing_positions.append((_pos, _clims_status["detail"]))

        if not clims_missing_positions:
            store_result["ok"].append(
                f"contrast limits ({clims_n_channels} channels)"
            )
        else:
            n_missing = len(clims_missing_positions)
            n_total = len(positions)
            detail = clims_missing_positions[0][1]
            store_result["missing"].append(
                f"contrast limits ({n_missing}/{n_total} positions missing: {detail})"
            )
            fix_commands.append(
                f"uv run python -m cyclops_process.processes.pyramids.audit_fix "
                f"--clims --store {store_type} -e {experiment}"
            )

        # --- Label layers: check ALL positions ---
        expected_labels = _EXPECTED_LABELS.get(store_type, {})

        for label_name, expected_levels in expected_labels.items():
            # Check label existence across all positions
            label_path = pos_path / "labels" / label_name  # first position (for fix command logic)
            label_exists_any = any(
                (store_path / p / "labels" / label_name).exists() for p in positions
            )

            if not label_exists_any:
                store_result["missing"].append(f"labels/{label_name}")
                # Determine fix command
                if label_name in ("iss_gene_image", "iss_guide_image"):
                    cmd = (
                        f"uv run python -m cyclops_process.processes.pyramids.audit_fix "
                        f"--iss -e {experiment}"
                    )
                    if cmd not in fix_commands:
                        fix_commands.append(cmd)
                elif label_name == "grid_overlay":
                    cmd = (
                        f"uv run python -m cyclops_process.processes.pyramids.audit_fix "
                        f"--build-grid --store {store_type} -e {experiment}"
                    )
                    fix_commands.append(cmd)
                elif label_name in ("nuclear_seg", "cell_seg"):
                    if store_type in ("iss", "track"):
                        # v2 source has been async-deleted post-convert, so we
                        # can't re-run with --source-zarr-version 2. Attach
                        # nuclear_seg as a top-level symlink onto the v3 store,
                        # then convert with --source-zarr-version 3 (in-place
                        # lift top-level → labels/).
                        symlink_cmd = (
                            f"uv run python -m cyclops_process.utils.batch.batch_symlink_nuclear_seg "
                            f"--experiments {experiment} --symlink-target {store_type} --force "
                            f"--zarr-version 3"
                        )
                        reconvert_cmd = (
                            f"uv run python -m cyclops_process.convert.v3_livecell "
                            f"--experiment {experiment} --only-labels {label_name} "
                            f"--mode {store_type} --source-zarr-version 3 --force"
                        )
                        store_result["missing"][-1] = (
                            f"labels/{label_name} (missing — needs symlink + convert for {store_type} store)"
                        )
                        store_result["missing"].append(f"  1. {symlink_cmd}")
                        store_result["missing"].append(f"  2. {reconvert_cmd}")
                        fix_commands.append(symlink_cmd)
                        fix_commands.append(reconvert_cmd)
                    else:
                        # Label missing from v3 store — reconvert from v2 source.
                        # --source-zarr-version 2 routes the v2 store as source
                        # (v3 store as dest). --source-zarr-version 3 expects
                        # the label already in v3 (top-level symlink lift) and
                        # silently no-ops with "No positions found" when it isn't.
                        reconvert_cmd = (
                            f"uv run python -m cyclops_process.convert.v3_livecell "
                            f"--experiment {experiment} --only-labels {label_name} "
                            f"--mode {store_type} --source-zarr-version 2 --force"
                        )
                        fix_commands.append(reconvert_cmd)
                continue

            # Check pyramid levels inside the label across ALL positions.
            # A level is only counted as OK if it has data in every position.
            # Skip pixel checks for rendered overlays (grid/iss) – they are
            # sparse RGBA images whose sampled patches are often all-zero,
            # causing false negatives.  Filesystem-only check is sufficient.
            _skip_pixel_labels = {"grid_overlay", "iss_gene_image", "iss_guide_image"}
            _do_pixel_check = label_name not in _skip_pixel_labels

            # Find max level across any position
            all_level_nums = set()
            for scan_pos in positions:
                scan_label = store_path / scan_pos / "labels" / label_name
                if scan_label.exists():
                    for d in scan_label.iterdir():
                        if d.is_dir() and d.name.isdigit():
                            all_level_nums.add(int(d.name))

            # A level is valid only if ALL positions have data at that level
            label_levels = []
            for lvl in sorted(all_level_nums):
                all_ok = True
                for scan_pos in positions:
                    is_track_well2 = store_type == "track" and "/2/" in scan_pos
                    if is_track_well2:
                        continue  # skip track well 2
                    lvl_path = store_path / scan_pos / "labels" / label_name / str(lvl)
                    if not level_has_data(lvl_path, check_pixels=_do_pixel_check, level_index=lvl):
                        print(f"    [DETECTED] Empty label data at {scan_pos}/labels/{label_name}/{lvl}")
                        all_ok = False
                        break
                if all_ok:
                    label_levels.append(str(lvl))

            n_levels = len(label_levels)

            if n_levels >= expected_levels:
                store_result["ok"].append(
                    f"labels/{label_name} ({n_levels} levels)"
                )
            else:
                if label_name in ("nuclear_seg", "cell_seg"):
                    _diagnose_seg_label_issue(
                        label_name, label_path, dataset, experiment,
                        store_type, positions, n_levels, expected_levels,
                        store_result, fix_commands,
                    )
                elif label_name in ("iss_gene_image", "iss_guide_image"):
                    store_result["missing"].append(
                        f"labels/{label_name} pyramids ({n_levels}/{expected_levels} levels)"
                    )
                    cmd = (
                        f"uv run python -m cyclops_process.processes.pyramids.audit_fix "
                        f"--iss -e {experiment}"
                    )
                    if cmd not in fix_commands:
                        fix_commands.append(cmd)
                elif label_name == "grid_overlay":
                    store_result["missing"].append(
                        f"labels/{label_name} pyramids ({n_levels}/{expected_levels} levels)"
                    )
                    fix_commands.append(
                        f"uv run python -m cyclops_process.processes.pyramids.audit_fix "
                        f"--build-grid --store {store_type} -e {experiment}"
                    )

        # --- Organelle labels (bonus: detect any extra _seg labels) ---
        labels_dir = pos_path / "labels"
        if labels_dir.exists():
            try:
                extra_labels = [
                    d.name
                    for d in labels_dir.iterdir()
                    if d.is_dir()
                    and not d.name.startswith(".")
                    and d.name.endswith("_seg")
                    and d.name not in ("seg", "nuclear_seg", "cell_seg")
                ]
                for org_label in sorted(extra_labels):
                    org_path = labels_dir / org_label
                    org_levels = [
                        d.name
                        for d in org_path.iterdir()
                        if d.is_dir() and d.name.isdigit() and level_has_data(d, check_pixels=True, level_index=int(d.name))
                    ]
                    if len(org_levels) >= _EXPECTED_IMAGE_LEVELS:
                        store_result["ok"].append(
                            f"labels/{org_label} ({len(org_levels)} levels)"
                        )
                    else:
                        store_result["missing"].append(
                            f"labels/{org_label} pyramids ({len(org_levels)}/{_EXPECTED_IMAGE_LEVELS} levels)"
                        )
                        fix_commands.append(
                            f"uv run python -m cyclops_process.processes.pyramids.audit_fix "
                            f"--organelle-pyramids --label-filter {org_label} -e {experiment}"
                        )
            except Exception:
                pass

        # --- YX-scale metadata audit (PHENO + TRACK) ---
        # A regression left level-0 YX over/under-declared at 0.65 µm/px in two
        # store types: pheno (native 0.325) and track (native 1.3). Both are
        # repairable in-place via --fix-yx-scale. ISS comes from an independent
        # upstream path — leave it alone.
        if store_type in ("pheno", "track"):
            _native_yx = PHENO_NATIVE_YX if store_type == "pheno" else TRACK_NATIVE_YX
            try:
                yx_status = _audit_yx_scale(
                    store_path, positions, native_yx=_native_yx
                )
                if yx_status["status"] == "wrong":
                    store_result["missing"].append(
                        f"YX scale metadata: declares {yx_status['observed']} "
                        f"but native is ({_native_yx}, {_native_yx})"
                    )
                    fix_commands.append(
                        f"uv run python -m cyclops_process.processes.pyramids.audit_fix "
                        f"--fix-yx-scale --store {store_type} -e {experiment}"
                    )
                elif yx_status["status"] == "correct":
                    store_result["ok"].append(
                        f"YX scale metadata ({_native_yx}, {_native_yx})"
                    )
                elif yx_status["status"] == "mixed":
                    store_result["warnings"].append(
                        f"YX scale mixed across positions: "
                        f"{yx_status['per_position']}"
                    )
                # status == "skipped" or "unknown": no entry
            except Exception as e:
                store_result["warnings"].append(f"YX scale audit failed: {e}")

        # --- Normalization metadata audit (PHENO only) ---
        # viscy preprocess writes a top-level `normalization` field; the v3
        # schema established by convert_v3.py also mirrors it under
        # position-level `custom_metadata.normalization`, which portal/napari/
        # audit readers depend on. Catch v3-native stores missing that mirror,
        # analogous to the YX-scale check above.
        if store_type == "pheno":
            try:
                norm_status = _audit_normalization(store_path, positions)
                if norm_status["status"] == "correct":
                    store_result["ok"].append("normalization metadata (custom_metadata)")
                elif norm_status["status"] == "needs_mirror":
                    store_result["missing"].append(
                        f"normalization metadata: top-level present but "
                        f"custom_metadata.normalization missing on "
                        f"{norm_status['n']} position(s)"
                    )
                    fix_commands.append(
                        f"uv run python -m cyclops_process.processes.pyramids.audit_fix "
                        f"--fix-normalization -e {experiment}"
                    )
                elif norm_status["status"] == "missing":
                    store_result["missing"].append(
                        f"normalization metadata absent on {norm_status['n']} "
                        f"position(s) — re-run the viscy_normalize step"
                    )
                # status == "skipped": no entry
            except Exception as e:
                store_result["warnings"].append(f"normalization audit failed: {e}")

        results[store_type] = store_result

    # Deduplicate fix commands while preserving order
    seen = set()
    unique_fixes = []
    for cmd in fix_commands:
        if cmd not in seen:
            seen.add(cmd)
            unique_fixes.append(cmd)
    results["fix_commands"] = unique_fixes

    # --- Fluorescence channel check ---
    results["fluor_warning"] = None
    pheno_result = results.get("pheno", {})
    if isinstance(pheno_result, dict) and pheno_result.get("exists"):
        try:
            from cyclops_process.utils.audit_v3_channels import check_missing_fluor_channels
            v3_path = _get_store_path(dataset, zarr_version=3, store_type="pheno")
            missing_fluor = check_missing_fluor_channels(experiment, Path(v3_path))
            if missing_fluor:
                warning = f"MISSING FLUORESCENT CHANNELS: {missing_fluor}"
                results["fluor_warning"] = warning
                pheno_result.setdefault("warnings", []).append(warning)
        except Exception:
            pass

    if verbose:
        _print_audit_report(experiment, results)

    return results
def _prompt_iss_rebuild(experiment: str):
    """Remind user to rebuild ISS overlays if link_calls_tracks was rerun."""
    cmd = (
        f"uv run python -m cyclops_process.processes.pyramids.audit_fix"
        f" --iss --no-resume --slurm -y -e {experiment}"
    )
    print(
        f"\n  NOTE: If you reran link_calls_tracks, rebuild ISS overlays to keep them up to date:"
        f"\n    {cmd}\n"
    )
def _print_audit_report(experiment: str, results: dict):
    """Pretty-print the audit report."""
    print(f"\n{'=' * 80}")
    print(f"  V3 STORE AUDIT: {experiment}")
    print(f"{'=' * 80}")

    for store_type in ["pheno", "iss", "track"]:
        r = results.get(store_type, {})
        if not r.get("exists"):
            print(f"\n  [{store_type.upper()}] NOT FOUND")
            continue

        n_ok = len(r.get("ok", []))
        n_miss = len(r.get("missing", []))
        wells = r.get("wells", [])
        status = "OK" if n_miss == 0 else f"{n_miss} MISSING"
        print(f"\n  [{store_type.upper()}] {status}  ({len(wells)} wells)")

        for item in r.get("ok", []):
            print(f"    \u2713 {item}")
        for item in r.get("missing", []):
            print(f"    \u2717 {item}")

    # Print fluorescence channel warning
    fluor_warning = results.get("fluor_warning")
    if fluor_warning:
        print(f"\n  ⚠⚠⚠  {fluor_warning}")

    fixes = results.get("fix_commands", [])
    if fixes:
        print(f"\n{'=' * 80}")
        print(f"  FIX COMMANDS ({len(fixes)}):")
        print(f"{'=' * 80}")
        print(f"  Run these commands to fix the issues:\n")
        for i, cmd in enumerate(fixes, 1):
            print(f"  {i}. {cmd}")
        print(f"\n  Or run all fixes automatically:")
        print(f"  python -m cyclops_process.processes.pyramids.audit_fix --fix -e {experiment}")
    else:
        print(f"\n  All stores complete!")

    print(f"{'=' * 80}\n")
