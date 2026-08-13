"""Dedupe ``*_linked_pheno_iss.csv`` files to one row per (well, tile_pheno,
segmentation_id) cell.

Per-cell classification (uses **distinct sgRNA count**, not gene_name —
matches the rule in ``cyclops_process.utils.cell_count_summary`` and the
portal-parquet dedupe in ``cyclops_utils.data_portal.dedupe_cell_data``):

  - **singlet (real gene)** — exactly 1 distinct sgRNA AND that sgRNA has a
    non-NaN ``gene_name`` (gene-targeting guide) → keep one row.
  - **singlet (NTC)** — exactly 1 distinct sgRNA AND ``gene_name`` is NaN
    (non-targeting control guide) → keep one row.
  - **multiplet** — 2+ distinct sgRNAs called for the cell → drop the cell
    entirely. This catches *every* ambiguous case: gene+gene, gene+NTC,
    NTC+NTC, same-gene-different-sgRNAs.

Why sgRNA, not gene_name? Each sgRNA is one CRISPR integration event. Two
distinct sgRNAs on one cell = two integrations = genuinely ambiguous KO,
regardless of whether one of them is an NTC barcode. The earlier
gene_name-based rule silently let gene+NTC mixed-call cells through as
singlets and under-counted multiplets by ~160k cells in paper-v1.

Usage::

    python -m cyclops_process.utils.dedupe_linked_pheno_iss \\
        --experiment ops0032_20250428 [--write-back]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from cyclops_process.paths import BASE_PATH

logger = logging.getLogger(__name__)

UNIQUE_KEY = ["well", "tile_pheno", "segmentation_id"]


def dedupe_linked_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Dedupe one experiment-well's linked_pheno_iss DataFrame.

    Assumes ``df`` already has a ``well`` column (derive from filename if not).
    Returns one row per kept cell — singletons (real gene or NTC); multiplets
    are dropped entirely.
    """
    df = df.copy().reset_index(drop=True)

    # Use `barcode` as the single guide column for multiplet detection.
    # Both CRISPR screens (where barcode is the 10mer alias for sgRNA, 1:1)
    # and custom-perturbation screens (where each barcode = one perturbation) carry it,
    # and verified byte-identical to sgRNA-based dedup across paper_v1 +
    # ops0094/0107/0124/0133 (5.4M cells, zero diff — 2026-06-09).
    if "barcode" not in df.columns:
        logger.warning(
            "dedupe_linked_csv: no `barcode` column found — skipping dedupe; "
            "returning %d rows unchanged", len(df),
        )
        return df

    # DROP fallback-bbox rows (segmentation_id is NaN). These come from
    # cell_bounding_boxes when the phenotyping point fell outside every
    # seg label — there's no real cell boundary, just a centroid window.
    # Excluding them avoids artificial multiplet collapse (when many
    # fallback rows share a well+tile would collapse to one synthetic
    # "...|nan" key) and keeps the per-experiment count principled.
    n_nan_seg = int(df["segmentation_id"].isna().sum())
    if n_nan_seg:
        logger.warning(
            "dedupe_linked_csv: dropping %d rows with NaN segmentation_id "
            "(fallback-bbox cells — no valid cell boundary)", n_nan_seg,
        )
        df = df.dropna(subset=["segmentation_id"]).reset_index(drop=True)

    df["_key"] = (
        df["well"].astype(str) + "|"
        + df["tile_pheno"].astype(str) + "|"
        + df["segmentation_id"].astype(str)
    )

    # Multiplet detection: cells with ≥2 distinct barcodes are ambiguous KOs
    # (CRISPR gene+gene, CRISPR gene+NTC, custom perturbation+perturbation, etc.)
    # and dropped entirely.
    with_guide = df.dropna(subset=["barcode"])
    n_distinct = with_guide.groupby("_key")["barcode"].nunique()
    singlet_keys = set(n_distinct[n_distinct == 1].index)

    # Keep one row per singlet cell. Within a library, barcode → gene_name
    # is 1:1, so every row of a singlet cell carries the same gene
    # attribution (NaN for NTC; non-NaN for real-gene). `keep="first"` is
    # therefore safe — no tie-break needed.
    out = df[df["_key"].isin(singlet_keys)].drop_duplicates("_key", keep="first")
    return out.drop(columns=["_key"])


def dedupe_experiment(
    experiment: str,
    write_back: bool = False,
    base_dir: Optional[Path] = None,
) -> dict:
    """Apply ``dedupe_linked_csv`` across every well of one experiment.

    ``write_back=False`` (default) is a dry-run — returns counts only.
    ``write_back=True`` overwrites each CSV in place after backing up to a
    sibling ``*.prededup.csv``.
    """
    base = base_dir or Path(f"{BASE_PATH}/{experiment}/3-assembly")
    stats = {
        "experiment": experiment,
        "files": 0,
        "raw_rows": 0,
        "kept_rows": 0,
        "singlet_real_gene_cells": 0,
        "singlet_ntc_cells": 0,
        "multiplet_cells_dropped": 0,
        "no_guide_cells": 0,
    }

    for csv_path in sorted(base.glob("*_linked_pheno_iss.csv")):
        df = pd.read_csv(csv_path)
        df["well"] = csv_path.name.split("_")[0]
        stats["files"] += 1
        stats["raw_rows"] += len(df)

        deduped = dedupe_linked_csv(df)
        stats["kept_rows"] += len(deduped)

        # Diagnostics (recompute from raw df so they're independent of dedupe_linked_csv)
        df_diag = df.copy()
        df_diag["_key"] = (
            df_diag["well"].astype(str) + "|"
            + df_diag["tile_pheno"].astype(str) + "|"
            + df_diag["segmentation_id"].astype(str)
        )
        all_keys = set(df_diag["_key"].unique())
        with_guide = df_diag.dropna(subset=["sgRNA"])
        n_sgrna = with_guide.groupby("_key")["sgRNA"].nunique()
        singlet_keys = set(n_sgrna[n_sgrna == 1].index)
        multi_keys   = set(n_sgrna[n_sgrna  > 1].index)
        no_guide     = all_keys - set(n_sgrna.index)

        # Of the singleton cells, how many have a real-gene call vs NTC?
        singleton_with_real_gene = (
            with_guide[with_guide["_key"].isin(singlet_keys)]
            .dropna(subset=["gene_name"])
        )
        real_gene_keys = set(singleton_with_real_gene["_key"])
        ntc_singleton_keys = singlet_keys - real_gene_keys

        stats["singlet_real_gene_cells"] += len(real_gene_keys)
        stats["singlet_ntc_cells"]       += len(ntc_singleton_keys)
        stats["multiplet_cells_dropped"] += len(multi_keys)
        stats["no_guide_cells"]          += len(no_guide)

        if write_back:
            backup = csv_path.with_suffix(".prededup.csv")
            if not backup.exists():
                csv_path.rename(backup)
            deduped.to_csv(csv_path, index=False)
            logger.info("  %s: %d → %d rows (backup: %s)",
                        csv_path.name, len(df), len(deduped), backup.name)
        else:
            logger.info("  %s: %d → %d rows (dry-run)",
                        csv_path.name, len(df), len(deduped))

    return stats


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--experiment", required=True,
                   help="Experiment id, e.g. ops0032_20250428")
    p.add_argument("--write-back", action="store_true",
                   help="Overwrite CSVs in place (default: dry-run; "
                        "creates *.prededup.csv backup)")
    p.add_argument("--base-dir", type=Path, default=None,
                   help="Override base dir (default: <base>/{experiment}/3-assembly)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    stats = dedupe_experiment(
        args.experiment,
        write_back=args.write_back,
        base_dir=args.base_dir,
    )
    print(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
