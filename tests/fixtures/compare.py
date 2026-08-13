"""Intelligent comparators for pipeline-stage outputs.

Used by real-data stage tests to validate that the output of a freshly-run
pipeline stage matches the reference cache to within an acceptable tolerance.

Comparators:
  - compare_ome_zarr: one OME-Zarr store vs another, structural + numerical
  - compare_ome_zarr_directory: all *.zarr stores in two parallel dirs
  - compare_csv: pandas-based, numeric within tolerance, strings exact
  - compare_yaml: structural recursive, numeric leaves within tolerance

Structural checks are always exact. Numerical data uses np.allclose with
caller-specified rtol/atol; per-stage tolerances are tracked in the test
files themselves rather than here, so this module stays stage-agnostic.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


def _deviation_stats(c: np.ndarray, r: np.ndarray, atol: float) -> dict[str, float]:
    """Compact stats summarizing how two arrays differ, for error messages."""
    diff = np.abs(c.astype(np.float64) - r.astype(np.float64))
    denom = np.abs(r.astype(np.float64)) + atol + 1e-30
    rel = diff / denom
    return {
        "max_abs": float(diff.max()),
        "max_rel": float(rel.max()),
        "mean_abs": float(diff.mean()),
        "frac_diff": float((diff > 0).mean()),
    }


def compare_ome_zarr(
    candidate: Path,
    reference: Path,
    *,
    rtol: float = 1e-5,
    atol: float = 0.0,
    check_data: bool = True,
    sample_positions: int | None = 5,
    sample_seed: int = 0,
    channels: list[str] | None = None,
    positions: list[str] | None = None,
) -> None:
    """Compare two OME-Zarr stores end-to-end, agnostic of zarr format version.

    If `channels` is given, only those named channels are compared (they must
    exist in BOTH stores) and the exact channel-list equality check is skipped.
    Use this when the candidate legitimately has MORE channels than the
    reference (e.g. a config added stains since the reference was generated) but
    the shared channels must still match. Shape is then compared with the
    channel axis excluded, and data is compared channel-by-channel.

    If `positions` is given, only those named positions are compared (they must
    exist in BOTH stores) and the exact position-list equality check is skipped.
    Use this for fan-out stages where one task writes only a subset of the
    store's positions (the rest are written by sibling tasks not run here): the
    listed positions are checked structurally and numerically, and the full
    position-list equality check that would flag the unwritten positions as
    "only in reference" is skipped. Composes with `channels`.

    The candidate may be zarr v3 / NGFF 0.5 while the reference is zarr v2 /
    NGFF 0.4 (or vice-versa). On-disk layout differences (chunk-store paths,
    .zarray vs zarr.json) are ignored; only logical content is compared via
    the iohub abstraction.

    Structural checks (always, on every position): position list, channel
    names, per-position shape, dtype, and coordinate scale.

    Numerical check (when `check_data`): per-position arrays compared via
    np.allclose with `rtol`/`atol`. By default a random sample of
    `sample_positions` positions per store is compared (deterministic via
    `sample_seed`); pass `sample_positions=None` to compare every position.

    Per-position chunk shape is intentionally NOT compared -- it is a storage
    choice that can legitimately differ between v2 and v3 writers without
    affecting logical content.

    Raises AssertionError with a focused error message on the first mismatch.
    """
    import random

    from iohub.ngff import open_ome_zarr

    candidate = Path(candidate)
    reference = Path(reference)

    with open_ome_zarr(candidate, mode="r") as cand, open_ome_zarr(
        reference, mode="r"
    ) as ref:
        cand_pos = sorted(p for p, _ in cand.positions())
        ref_pos = sorted(p for p, _ in ref.positions())
        if positions is None:
            if cand_pos != ref_pos:
                only_cand = sorted(set(cand_pos) - set(ref_pos))[:10]
                only_ref = sorted(set(ref_pos) - set(cand_pos))[:10]
                raise AssertionError(
                    f"position list mismatch\n"
                    f"  candidate: {candidate}\n"
                    f"  reference: {reference}\n"
                    f"  only in candidate (first 10): {only_cand}\n"
                    f"  only in reference (first 10): {only_ref}"
                )
            compare_pos = ref_pos
        else:
            missing_c = [p for p in positions if p not in cand_pos]
            missing_r = [p for p in positions if p not in ref_pos]
            if missing_c or missing_r:
                raise AssertionError(
                    f"requested positions not present\n"
                    f"  candidate: {candidate}\n"
                    f"  reference: {reference}\n"
                    f"  missing in candidate: {missing_c}\n"
                    f"  missing in reference: {missing_r}"
                )
            compare_pos = list(positions)

        cand_ch = list(getattr(cand, "channel_names", []) or [])
        ref_ch = list(getattr(ref, "channel_names", []) or [])
        # channel index maps; populated only when comparing a channel subset
        cand_ci: dict[str, int] = {}
        ref_ci: dict[str, int] = {}
        if channels is None:
            if cand_ch != ref_ch:
                raise AssertionError(
                    f"channel name mismatch at {candidate}\n"
                    f"  candidate: {cand_ch}\n"
                    f"  reference: {ref_ch}"
                )
        else:
            missing_c = [c for c in channels if c not in cand_ch]
            missing_r = [c for c in channels if c not in ref_ch]
            if missing_c or missing_r:
                raise AssertionError(
                    f"requested channels not present at {candidate}\n"
                    f"  requested: {channels}\n"
                    f"  missing in candidate: {missing_c}  (has {cand_ch})\n"
                    f"  missing in reference: {missing_r}  (has {ref_ch})"
                )
            cand_ci = {c: cand_ch.index(c) for c in channels}
            ref_ci = {c: ref_ch.index(c) for c in channels}

        # Structural check on every position; data check restricted to the
        # sampled subset so the per-test wall clock stays bounded even when
        # a store holds hundreds of positions.
        sampled: set[str]
        if check_data and sample_positions is not None and len(compare_pos) > sample_positions:
            rng = random.Random(sample_seed)
            sampled = set(rng.sample(compare_pos, sample_positions))
        else:
            sampled = set(compare_pos)

        for pos_name in compare_pos:
            cand_p = cand[pos_name]
            ref_p = ref[pos_name]
            cand_arr = cand_p.data
            ref_arr = ref_p.data

            # When comparing a channel subset the stores legitimately differ on
            # the channel axis (axis 1 of T,C,Z,Y,X); compare shape without it.
            cand_shape_cmp = (
                cand_arr.shape[:1] + cand_arr.shape[2:] if channels else cand_arr.shape
            )
            ref_shape_cmp = (
                ref_arr.shape[:1] + ref_arr.shape[2:] if channels else ref_arr.shape
            )
            if cand_shape_cmp != ref_shape_cmp:
                raise AssertionError(
                    f"shape mismatch at {candidate}::{pos_name}: "
                    f"{cand_arr.shape} vs {ref_arr.shape}"
                    + (f" (channel axis excluded; comparing {channels})" if channels else "")
                )
            if cand_arr.dtype != ref_arr.dtype:
                raise AssertionError(
                    f"dtype mismatch at {candidate}::{pos_name}: "
                    f"{cand_arr.dtype} vs {ref_arr.dtype}"
                )
            cand_scale = list(getattr(cand_p, "scale", []) or [])
            ref_scale = list(getattr(ref_p, "scale", []) or [])
            if cand_scale != ref_scale:
                raise AssertionError(
                    f"scale mismatch at {candidate}::{pos_name}: "
                    f"{cand_scale} vs {ref_scale}"
                )

            if check_data and pos_name in sampled:
                if channels:
                    # Compare each requested channel's slice independently.
                    pairs = [
                        (
                            ch,
                            np.asarray(cand_arr[:, cand_ci[ch]]),
                            np.asarray(ref_arr[:, ref_ci[ch]]),
                        )
                        for ch in channels
                    ]
                else:
                    pairs = [(None, np.asarray(cand_arr[:]), np.asarray(ref_arr[:]))]
                for ch, c, r in pairs:
                    if not np.allclose(c, r, rtol=rtol, atol=atol, equal_nan=True):
                        stats = _deviation_stats(c, r, atol)
                        where = f"{pos_name}" + (f" channel={ch!r}" if ch else "")
                        raise AssertionError(
                            f"data mismatch at {candidate}::{where} "
                            f"(rtol={rtol}, atol={atol})\n"
                            f"  max abs diff:   {stats['max_abs']:.6g}\n"
                            f"  max rel diff:   {stats['max_rel']:.6g}\n"
                            f"  mean abs diff:  {stats['mean_abs']:.6g}\n"
                            f"  fraction diff:  {stats['frac_diff']:.3%}"
                        )


def compare_ome_zarr_directory(
    candidate_dir: Path,
    reference_dir: Path,
    *,
    pattern: str = "*.zarr",
    rtol: float = 1e-5,
    atol: float = 0.0,
    check_data: bool = True,
    sample_positions: int | None = 5,
    sample_seed: int = 0,
) -> None:
    """Compare every OME-Zarr store matching `pattern` between two dirs.

    Both dirs must contain the same set of zarr-store names; symmetric
    differences raise AssertionError before any data is opened. Per-store
    structural checks run on every position; numerical checks honor
    `sample_positions` (see compare_ome_zarr).
    """
    candidate_dir = Path(candidate_dir)
    reference_dir = Path(reference_dir)

    cand_names = sorted(p.name for p in candidate_dir.glob(pattern))
    ref_names = sorted(p.name for p in reference_dir.glob(pattern))
    only_cand = sorted(set(cand_names) - set(ref_names))
    only_ref = sorted(set(ref_names) - set(cand_names))
    if only_cand or only_ref:
        raise AssertionError(
            f"zarr-store set differs\n"
            f"  candidate: {candidate_dir}\n"
            f"  reference: {reference_dir}\n"
            f"  only in candidate: {only_cand}\n"
            f"  only in reference: {only_ref}"
        )

    for name in cand_names:
        compare_ome_zarr(
            candidate_dir / name,
            reference_dir / name,
            rtol=rtol,
            atol=atol,
            check_data=check_data,
            sample_positions=sample_positions,
            sample_seed=sample_seed,
        )


def compare_npy(
    candidate: Path,
    reference: Path,
    *,
    rtol: float = 1e-5,
    atol: float = 0.0,
    sort_rows: bool = False,
) -> None:
    """Compare two .npy arrays via np.allclose.

    Shape and dtype-kind must match. With `sort_rows`, rows are lexicographically
    sorted before comparison -- use this for unordered point clouds (e.g.
    detected-spot coordinate tables) where the row ORDER is not meaningful but
    the SET of rows is. Requires a 2-D array; the row count must already match
    (a different number of points is a real difference, caught by the shape
    check).

    Raises AssertionError with deviation stats on the first mismatch.
    """
    cand = np.load(candidate)
    ref = np.load(reference)
    if cand.shape != ref.shape:
        raise AssertionError(
            f"shape mismatch at {candidate}: {cand.shape} vs {ref.shape}"
        )
    if cand.dtype.kind != ref.dtype.kind:
        raise AssertionError(
            f"dtype-kind mismatch at {candidate}: "
            f"{cand.dtype} vs {ref.dtype}"
        )
    if sort_rows:
        if cand.ndim < 2:
            raise ValueError("sort_rows requires a 2-D array")
        # lexsort's last key is primary; pass columns reversed so column 0 is
        # the primary sort key, giving a stable row-order-independent ordering.
        cand = cand[np.lexsort(cand.T[::-1])]
        ref = ref[np.lexsort(ref.T[::-1])]
    if not np.allclose(cand, ref, rtol=rtol, atol=atol, equal_nan=True):
        stats = _deviation_stats(cand, ref, atol)
        raise AssertionError(
            f"data mismatch at {candidate} (rtol={rtol}, atol={atol}"
            f"{', sorted rows' if sort_rows else ''})\n"
            f"  max abs diff:   {stats['max_abs']:.6g}\n"
            f"  max rel diff:   {stats['max_rel']:.6g}\n"
            f"  mean abs diff:  {stats['mean_abs']:.6g}\n"
            f"  fraction diff:  {stats['frac_diff']:.3%}"
        )


def compare_pyramid_levels(
    candidate: Path,
    reference: Path,
    position: str,
    n_levels: int,
    *,
    rtol: float = 1e-5,
    atol: float = 0.0,
) -> None:
    """Compare multiscale pyramid levels of one position between two stores.

    For fan-out stages that add downsampled pyramid levels in place to a single
    position (e.g. build_pyramids_position_job): asserts that levels ``0..n_levels-1``
    of `position` exist in both stores with matching shape, dtype, and data.

    Raises AssertionError on the first level that is missing or differs.
    """
    from iohub.ngff import open_ome_zarr

    candidate = Path(candidate)
    reference = Path(reference)
    with open_ome_zarr(candidate, mode="r") as cand, open_ome_zarr(
        reference, mode="r"
    ) as ref:
        cand_p = cand[position]
        ref_p = ref[position]
        for level in range(n_levels):
            key = str(level)
            try:
                cand_arr = np.asarray(cand_p[key][:])
            except KeyError:
                raise AssertionError(
                    f"pyramid level {key!r} missing in candidate "
                    f"{candidate}::{position}"
                )
            try:
                ref_arr = np.asarray(ref_p[key][:])
            except KeyError:
                raise AssertionError(
                    f"pyramid level {key!r} missing in reference "
                    f"{reference}::{position}"
                )
            if cand_arr.shape != ref_arr.shape:
                raise AssertionError(
                    f"level {key} shape mismatch at {candidate}::{position}: "
                    f"{cand_arr.shape} vs {ref_arr.shape}"
                )
            if not np.allclose(cand_arr, ref_arr, rtol=rtol, atol=atol, equal_nan=True):
                stats = _deviation_stats(cand_arr, ref_arr, atol)
                raise AssertionError(
                    f"level {key} data mismatch at {candidate}::{position} "
                    f"(rtol={rtol}, atol={atol})\n"
                    f"  max abs diff:   {stats['max_abs']:.6g}\n"
                    f"  max rel diff:   {stats['max_rel']:.6g}\n"
                    f"  mean abs diff:  {stats['mean_abs']:.6g}\n"
                    f"  fraction diff:  {stats['frac_diff']:.3%}"
                )


def compare_csv(
    candidate: Path,
    reference: Path,
    *,
    rtol: float = 1e-5,
    atol: float = 0.0,
    sort_rows: bool = False,
) -> None:
    """Compare two CSV files. Numeric columns within tolerance; non-numeric exact.

    Columns and row count must match exactly. By default row ordering must match;
    pass ``sort_rows=True`` to compare order-independently (both frames are sorted
    by all columns first). Use it for outputs whose ROW ORDER is not meaningful but
    whose SET of rows is -- e.g. base_calling's per-well reads.csv, which is
    assembled from parallel per-tile workers so the row order is not deterministic
    even though the reads themselves are (same code + same seeded input).
    """
    import pandas as pd

    cand_df = pd.read_csv(candidate)
    ref_df = pd.read_csv(reference)
    if list(cand_df.columns) != list(ref_df.columns):
        raise AssertionError(
            f"column mismatch at {candidate}\n"
            f"  candidate: {list(cand_df.columns)}\n"
            f"  reference: {list(ref_df.columns)}"
        )
    if len(cand_df) != len(ref_df):
        raise AssertionError(
            f"row count mismatch at {candidate}: "
            f"{len(cand_df)} vs {len(ref_df)}"
        )
    if sort_rows:
        # Canonicalize row order: sort by all columns, then drop the old index.
        cols = list(ref_df.columns)
        cand_df = cand_df.sort_values(by=cols, kind="mergesort").reset_index(drop=True)
        ref_df = ref_df.sort_values(by=cols, kind="mergesort").reset_index(drop=True)
    for col in ref_df.columns:
        if pd.api.types.is_numeric_dtype(ref_df[col]):
            if not np.allclose(
                cand_df[col].to_numpy(),
                ref_df[col].to_numpy(),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            ):
                diff = (cand_df[col] - ref_df[col]).abs()
                raise AssertionError(
                    f"column {col!r} differs at {candidate} "
                    f"(rtol={rtol}, atol={atol})\n"
                    f"  max abs diff: {diff.max():.6g} (row {int(diff.idxmax())})"
                )
        else:
            mask = cand_df[col].astype(str) != ref_df[col].astype(str)
            if mask.any():
                first = int(mask.idxmax())
                raise AssertionError(
                    f"column {col!r} differs at {candidate} (row {first}): "
                    f"{cand_df[col].iloc[first]!r} vs {ref_df[col].iloc[first]!r}"
                )


def compare_metrics_table(
    candidate: Path,
    reference: Path,
    *,
    rtol: float = 0.05,
    atol: float = 0.0,
    index_col: int = 0,
) -> None:
    """Compare a metrics table (rows = metric name, cols = wells) tolerantly.

    Intended for plate_stats.csv-style outputs whose metric SET legitimately
    evolves between pipeline versions. Lenient about coverage, strict about
    agreement:

    - Metrics (rows) or wells (columns) present in only ONE of the two tables
      are reported via ``warnings.warn`` (NOT a failure) -- so a candidate that
      computes a subset of the reference's metrics still passes, as long as the
      metrics it does compute agree.
    - Every metric/well value present in BOTH must agree within ``rtol``
      (default 5%); numeric within tolerance, non-numeric exact.

    Raises AssertionError only when shared values disagree beyond tolerance.
    """
    import warnings

    import pandas as pd

    cand = pd.read_csv(candidate, index_col=index_col)
    ref = pd.read_csv(reference, index_col=index_col)

    cand_idx, ref_idx = set(cand.index), set(ref.index)
    only_ref = [m for m in ref.index if m not in cand_idx]
    only_cand = [m for m in cand.index if m not in ref_idx]
    if only_ref:
        warnings.warn(
            f"{len(only_ref)}/{len(ref.index)} reference metric(s) MISSING from "
            f"candidate {Path(candidate).name}: {sorted(only_ref)}",
            stacklevel=2,
        )
    if only_cand:
        warnings.warn(
            f"{len(only_cand)} metric(s) in candidate not present in reference: "
            f"{sorted(only_cand)}",
            stacklevel=2,
        )

    cand_cols, ref_cols = set(cand.columns), set(ref.columns)
    only_ref_cols = [c for c in ref.columns if c not in cand_cols]
    only_cand_cols = [c for c in cand.columns if c not in ref_cols]
    if only_ref_cols or only_cand_cols:
        warnings.warn(
            f"well-column set differs in {Path(candidate).name}: "
            f"only-in-reference={only_ref_cols}, only-in-candidate={only_cand_cols}",
            stacklevel=2,
        )

    common_cols = [c for c in ref.columns if c in cand_cols]
    common_rows = [m for m in ref.index if m in cand_idx]
    failures = []
    for metric in common_rows:
        for col in common_cols:
            cv, rv = cand.at[metric, col], ref.at[metric, col]
            try:
                cvf, rvf = float(cv), float(rv)
            except (TypeError, ValueError):
                if str(cv) != str(rv):
                    failures.append((metric, col, cv, rv, "non-numeric"))
                continue
            if math.isnan(cvf) and math.isnan(rvf):
                continue
            if not math.isclose(cvf, rvf, rel_tol=rtol, abs_tol=atol):
                pct = (abs(cvf - rvf) / abs(rvf) * 100) if rvf else float("inf")
                failures.append((metric, col, cvf, rvf, f"{pct:.1f}%"))

    if failures:
        lines = "\n".join(
            f"  {m} [{c}]: {cv} vs {rv}  (off by {d})"
            for m, c, cv, rv, d in failures[:25]
        )
        raise AssertionError(
            f"{len(failures)} shared metric value(s) differ beyond "
            f"rtol={rtol:.0%} at {candidate}:\n{lines}"
            + (f"\n  ... (+{len(failures) - 25} more)" if len(failures) > 25 else "")
        )


def compare_yaml(
    candidate: Path,
    reference: Path,
    *,
    rtol: float = 1e-5,
    atol: float = 0.0,
    overrides: dict[str, dict[str, float]] | None = None,
) -> None:
    """Compare two YAML files structurally; numeric leaves within tolerance.

    ``overrides`` maps a TOP-LEVEL key to its own ``{"rtol": ..., "atol": ...}``
    so different sections can use different tolerances. This is needed for
    stitch_settings.yml, whose ``confidence`` block is bit-exact across runs but
    whose ``total_translation`` block carries a few-pixel additive jitter from
    the (CPU-fallback) registration solver; the two cannot share one tolerance
    without either masking confidence regressions or flagging translation noise.
    """
    import yaml

    with open(candidate) as f:
        cand = yaml.safe_load(f)
    with open(reference) as f:
        ref = yaml.safe_load(f)
    if overrides and isinstance(ref, dict) and isinstance(cand, dict):
        if set(cand.keys()) != set(ref.keys()):
            only_c = set(cand.keys()) - set(ref.keys())
            only_r = set(ref.keys()) - set(cand.keys())
            raise AssertionError(
                f"top-level keys differ at {candidate}::: "
                f"+{sorted(only_c)} -{sorted(only_r)}"
            )
        for k in ref:
            o = overrides.get(k, {})
            _compare_recursive(
                cand[k],
                ref[k],
                path=f"{candidate}::.{k}",
                rtol=o.get("rtol", rtol),
                atol=o.get("atol", atol),
            )
        return
    _compare_recursive(cand, ref, path=f"{candidate}::", rtol=rtol, atol=atol)


def _is_real_number(x: Any) -> bool:
    """True for int/float but not bool (bool is an int subclass we compare exactly)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _compare_recursive(c: Any, r: Any, path: str, rtol: float, atol: float) -> None:
    if type(c) is not type(r):
        if isinstance(c, (int, float)) and isinstance(r, (int, float)):
            pass  # allow int/float cross-comparison
        else:
            raise AssertionError(f"type mismatch at {path}: {type(c)} vs {type(r)}")
    if isinstance(r, dict):
        if set(c.keys()) != set(r.keys()):
            only_c = set(c.keys()) - set(r.keys())
            only_r = set(r.keys()) - set(c.keys())
            raise AssertionError(
                f"dict keys differ at {path}: +{sorted(only_c)} -{sorted(only_r)}"
            )
        for k in r:
            _compare_recursive(c[k], r[k], f"{path}.{k}", rtol, atol)
    elif isinstance(r, list):
        if len(c) != len(r):
            raise AssertionError(
                f"list length differs at {path}: {len(c)} vs {len(r)}"
            )
        for i, (ci, ri) in enumerate(zip(c, r)):
            _compare_recursive(ci, ri, f"{path}[{i}]", rtol, atol)
    elif _is_real_number(r) and _is_real_number(c):
        # Numeric leaves (int OR float) compare within tolerance. Treating ints
        # as exact-only was wrong: integer pixel translations carry solver
        # jitter that an `atol` override must be able to absorb. Under the
        # default rtol=1e-5/atol=0 this stays effectively exact for counts.
        if isinstance(r, float) and isinstance(c, float) and math.isnan(r) and math.isnan(c):
            return
        if not math.isclose(c, r, rel_tol=rtol, abs_tol=atol):
            raise AssertionError(
                f"number differs at {path}: {c} vs {r} "
                f"(rtol={rtol}, atol={atol})"
            )
    elif c != r:
        raise AssertionError(f"value differs at {path}: {c!r} vs {r!r}")
