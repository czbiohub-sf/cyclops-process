"""
Compare a candidate plate_stats.csv against the ops0094 reference, checking
that each metric matches within 0.5% relative tolerance.

Usage:
    python -m tests.QC.compare_plate_stats <candidate_csv> [--tolerance 0.005]
"""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
from prettytable import PrettyTable
from cyclops_process.paths import BASE_PATH


REFERENCE_PATH = Path(
    f"{BASE_PATH}/ops0094_20251217/3-assembly/ISS/mine/plate_stats.csv"
)

# Groups defined by 1-indexed row number in the reference CSV (row 1 is the
# header, metrics start at row 2). Ranges are inclusive on both ends.
GROUPS = [
    ("ISS quick-check", 2, 7),
    ("ISS base-calling", 2, 42),
    ("ISS spot detection", 10, 10),
    ("ISS stitching", 43, 47),
    ("Tracking stitching", 48, 52),
    ("Phenotyping stitching", 53, 57),
    ("Cross-modality overlap", 58, 61),
    ("Post-tracking", 62, 66),
    ("Phenotyping phase-reconstruction", 67, 78),
    ("Tracking phase-reconstruction", 79, 90),
    ("ISS drift correction", 91, 102),
    ("Tracking", 103, 127),
    ("Spatial coherence", 128, 143),
    ("Phenotyping cell-segmentation", 144, 166),
    ("Flatfield", 167, 173),
]


def load_plate_stats(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index.name = "metric"
    return df


def compare_value(ref: float, cand: float, tol: float) -> tuple[str, float | None]:
    """Return (status, relative_diff). Status is one of: pass, fail, both_nan,
    ref_nan, cand_nan, ref_zero_match, ref_zero_fail."""
    ref_nan = ref is None or (isinstance(ref, float) and math.isnan(ref))
    cand_nan = cand is None or (isinstance(cand, float) and math.isnan(cand))

    if ref_nan and cand_nan:
        return "both_nan", None
    if ref_nan:
        return "ref_nan", None
    if cand_nan:
        return "cand_nan", None

    if ref == 0:
        if cand == 0:
            return "ref_zero_match", 0.0
        return "ref_zero_fail", float("inf")

    rel = abs(cand - ref) / abs(ref)
    return ("pass" if rel <= tol else "fail"), rel


def compare(
    ref_df: pd.DataFrame, cand_df: pd.DataFrame, tol: float
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Compare every (metric, well) pair on the intersection of wells.
    Returns (results_df, wells_missing_in_candidate, wells_extra_in_candidate)."""
    common_wells = [w for w in ref_df.columns if w in cand_df.columns]
    missing = [w for w in ref_df.columns if w not in cand_df.columns]
    extra = [w for w in cand_df.columns if w not in ref_df.columns]

    rows = []
    for row_num, metric in enumerate(ref_df.index, start=2):
        for well in common_wells:
            ref = ref_df.at[metric, well]
            cand = cand_df.at[metric, well] if metric in cand_df.index else float("nan")
            status, rel = compare_value(ref, cand, tol)
            rows.append(
                {
                    "row": row_num,
                    "metric": metric,
                    "well": well,
                    "ref": ref,
                    "cand": cand,
                    "rel_diff": rel,
                    "status": status,
                }
            )
    return pd.DataFrame(rows), missing, extra


def is_fail(status: str) -> bool:
    return status in {"fail", "ref_zero_fail", "ref_nan", "cand_nan"}


def is_pass(status: str) -> bool:
    return status in {"pass", "both_nan", "ref_zero_match"}


def summary_table(results: pd.DataFrame, tol: float) -> PrettyTable:
    table = PrettyTable()
    table.field_names = ["Group", "Rows", "Total", "Pass", "Fail", "Skip", "Pass %"]
    table.align = "l"
    table.align["Pass %"] = "r"

    for name, start, end in GROUPS:
        sub = results[(results["row"] >= start) & (results["row"] <= end)]
        n = len(sub)
        n_pass = sub["status"].apply(is_pass).sum()
        n_fail = sub["status"].apply(is_fail).sum()
        n_skip = n - n_pass - n_fail
        pct = (100.0 * n_pass / n) if n else float("nan")
        table.add_row(
            [name, f"{start}-{end}", n, n_pass, n_fail, n_skip, f"{pct:.1f}%"]
        )

    n = len(results)
    n_pass = results["status"].apply(is_pass).sum()
    n_fail = results["status"].apply(is_fail).sum()
    n_skip = n - n_pass - n_fail
    pct = (100.0 * n_pass / n) if n else float("nan")
    table.add_row(["", "", "", "", "", "", ""])
    table.add_row(["OVERALL", "2-173", n, n_pass, n_fail, n_skip, f"{pct:.1f}%"])
    return table


def format_value(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "nan"
    if isinstance(v, float):
        if v == int(v):
            return f"{int(v):,}"
        return f"{v:,.4g}"
    return str(v)


def grouped_report(results: pd.DataFrame, tol: float, show_passes: bool) -> str:
    lines = []
    for name, start, end in GROUPS:
        sub = results[(results["row"] >= start) & (results["row"] <= end)]
        if sub.empty:
            continue
        failures = sub[sub["status"].apply(is_fail)]
        if failures.empty and not show_passes:
            lines.append(f"\n{name} (rows {start}-{end}): all {len(sub)} checks passed")
            continue

        lines.append(f"\n{name} (rows {start}-{end})")
        lines.append("-" * (len(name) + len(f" (rows {start}-{end})")))
        t = PrettyTable()
        t.field_names = ["row", "metric", "well", "ref", "candidate", "rel_diff", "status"]
        t.align = "l"
        t.align["rel_diff"] = "r"
        display = sub if show_passes else failures
        for _, r in display.iterrows():
            rel = r["rel_diff"]
            rel_str = "—" if rel is None else (
                "inf" if math.isinf(rel) else f"{rel * 100:.3f}%"
            )
            t.add_row(
                [
                    r["row"],
                    r["metric"],
                    r["well"],
                    format_value(r["ref"]),
                    format_value(r["cand"]),
                    rel_str,
                    r["status"],
                ]
            )
        lines.append(str(t))
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("candidate", type=Path, help="Path to candidate plate_stats.csv")
    p.add_argument(
        "--reference",
        type=Path,
        default=REFERENCE_PATH,
        help=f"Path to reference plate_stats.csv (default: {REFERENCE_PATH})",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.005,
        help="Relative tolerance (default: 0.005 = 0.5%%)",
    )
    p.add_argument(
        "--show-passes",
        action="store_true",
        help="Include passing rows in the detailed report (default: failures only)",
    )
    p.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional path to write the full per-(metric, well) results as CSV",
    )
    args = p.parse_args()

    if not args.reference.exists():
        sys.exit(f"Reference not found: {args.reference}")
    if not args.candidate.exists():
        sys.exit(f"Candidate not found: {args.candidate}")

    ref_df = load_plate_stats(args.reference)
    cand_df = load_plate_stats(args.candidate)

    cand_df = cand_df.reindex(ref_df.index)

    results, missing_wells, extra_wells = compare(ref_df, cand_df, args.tolerance)

    print("=" * 80)
    print(f"plate_stats.csv comparison @ tolerance ±{args.tolerance * 100:.2f}%")
    print(f"  reference: {args.reference}")
    print(f"  candidate: {args.candidate}")
    print(f"  reference wells: {list(ref_df.columns)}")
    print(f"  candidate wells: {list(cand_df.columns)}")
    if missing_wells:
        print(f"  WARNING: wells in reference but not candidate: {missing_wells}")
    if extra_wells:
        print(f"  note: wells in candidate but not reference (ignored): {extra_wells}")
    print("=" * 80)

    print("\n## Summary")
    print(summary_table(results, args.tolerance))

    print("\n## Detailed report (failures only)" if not args.show_passes
          else "\n## Detailed report (all rows)")
    print(grouped_report(results, args.tolerance, args.show_passes))

    if args.csv_out:
        results.to_csv(args.csv_out, index=False)
        print(f"\nFull results written to {args.csv_out}")

    n_fail = int(results["status"].apply(is_fail).sum())
    print(f"\n{'=' * 80}")
    if n_fail == 0:
        print(f"PASS: all {len(results)} (metric, well) checks within tolerance")
        sys.exit(0)
    else:
        print(f"FAIL: {n_fail} of {len(results)} (metric, well) checks exceeded tolerance")
        sys.exit(1)


if __name__ == "__main__":
    main()
