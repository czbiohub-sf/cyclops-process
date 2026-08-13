"""Tests for dedupe_linked_csv — the multiplet-dropping rule used both as a
standalone CLI and inside datasets.py:_process_well.

Run with: pytest cyclops_process/cyclops_process/utils/test_dedupe_linked_pheno_iss.py
"""
import pandas as pd
import pytest

from cyclops_process.utils.dedupe_linked_pheno_iss import dedupe_linked_csv


def _make_row(well, tile, seg, sgRNA, gene_name=None, **extra):
    """Construct a synthetic CRISPR row. `sgRNA` doubles as the `barcode`
    (10mer alias — they're 1:1 in real libraries). dedupe_linked_csv now
    keys multiplet detection on `barcode`, so we mirror both."""
    row = {
        "well": well,
        "tile_pheno": tile,
        "segmentation_id": seg,
        "sgRNA": sgRNA,
        "barcode": sgRNA,  # 1:1 in CRISPR libraries
        "gene_name": gene_name,
    }
    row.update(extra)
    return row


# A clean cell with 1 distinct sgRNA → kept as-is.
# Different sgRNA rows on the same cell → multiplet, drop entirely.
# Single NTC-only sgRNA → kept (gene_name will be NaN).
def test_singlet_real_gene_kept():
    df = pd.DataFrame([_make_row("A1", 5, 100, "sgABC", "GENE1")])
    out = dedupe_linked_csv(df)
    assert len(out) == 1
    assert out.iloc[0]["gene_name"] == "GENE1"


def test_singlet_ntc_kept():
    df = pd.DataFrame([_make_row("A1", 5, 100, "sgNTC1", None)])  # gene_name=NaN → NTC
    out = dedupe_linked_csv(df)
    assert len(out) == 1
    assert pd.isna(out.iloc[0]["gene_name"])


def test_multiplet_two_genes_dropped():
    # Same (well, tile, seg) has 2 distinct sgRNAs → multiplet
    df = pd.DataFrame([
        _make_row("A1", 5, 100, "sgABC", "GENE1"),
        _make_row("A1", 5, 100, "sgXYZ", "GENE2"),
    ])
    out = dedupe_linked_csv(df)
    assert len(out) == 0, "multiplet (gene+gene) must be dropped"


def test_multiplet_gene_plus_ntc_dropped():
    # Mixed gene+NTC on one cell — still 2 distinct sgRNAs → multiplet
    df = pd.DataFrame([
        _make_row("A1", 5, 100, "sgABC", "GENE1"),
        _make_row("A1", 5, 100, "sgNTC1", None),
    ])
    out = dedupe_linked_csv(df)
    assert len(out) == 0, "gene+NTC must drop (not silently survive as gene singlet)"


def test_same_sgrna_dedup_collapse():
    # Same sgRNA called twice on same cell — singlet (1 distinct sgRNA), keep 1 row
    df = pd.DataFrame([
        _make_row("A1", 5, 100, "sgABC", "GENE1"),
        _make_row("A1", 5, 100, "sgABC", "GENE1"),
    ])
    out = dedupe_linked_csv(df)
    assert len(out) == 1, "duplicate row of same sgRNA collapses to one"


def test_distinct_cells_independent():
    # Three different cells, each clean — all kept
    df = pd.DataFrame([
        _make_row("A1", 5, 100, "sgABC", "GENE1"),
        _make_row("A1", 5, 101, "sgXYZ", "GENE2"),
        _make_row("A1", 6, 100, "sgDEF", "GENE3"),
    ])
    out = dedupe_linked_csv(df)
    assert len(out) == 3
    assert set(out["gene_name"]) == {"GENE1", "GENE2", "GENE3"}


def test_mixed_scenario():
    # 1 singlet gene + 1 singlet NTC + 1 multiplet → 2 kept rows
    df = pd.DataFrame([
        _make_row("A1", 5, 100, "sgABC", "GENE1"),                     # singlet
        _make_row("A1", 5, 101, "sgNTC1", None),                        # NTC singlet
        _make_row("A1", 5, 102, "sgFOO", "GENE2"),                      # multiplet (with row below)
        _make_row("A1", 5, 102, "sgBAR", "GENE3"),                      # multiplet
    ])
    out = dedupe_linked_csv(df)
    assert len(out) == 2
    surviving_segs = sorted(out["segmentation_id"].tolist())
    assert surviving_segs == [100, 101]


def test_nan_segmentation_id_dropped():
    # Fallback-bbox cells (segmentation_id = NaN) come from
    # cell_bounding_boxes when the phenotyping point fell outside every seg
    # label. They have no real cell boundary, so we DROP them entirely from
    # the dedup output (logged via a warning). Regression guard for the
    # 2026-06-09 behavior change — previously these collapsed into a
    # synthetic "|A1|5|nan" multiplet and were dropped as a group.
    import numpy as np
    df = pd.DataFrame([
        _make_row("A1", 5, np.nan, "sgABC", "GENE1"),     # NaN-seg → drop
        _make_row("A1", 5, np.nan, "sgXYZ", "GENE2"),     # NaN-seg → drop
        _make_row("A1", 5, 100, "sgGOOD", "GENE3"),       # valid seg → keep
    ])
    out = dedupe_linked_csv(df)
    assert len(out) == 1, "NaN-seg rows must be dropped; only valid-seg cells kept"
    assert out.iloc[0]["gene_name"] == "GENE3"


def test_custom_perturbation_no_sgrna_column():
    # Some custom-perturbation experiments have no `sgRNA` column — they use
    # `barcode` (per gene_index_column_map in the library map). dedupe_linked_csv
    # keys multiplet detection on `barcode` for both modalities, so this should
    # work without any auto-detection branching, carrying a custom label column.
    rows = [
        {"well": "A1", "tile_pheno": 5, "segmentation_id": 100,
         "barcode": "GATTACAGTC", "perturbation": "P001"},
        {"well": "A1", "tile_pheno": 5, "segmentation_id": 101,
         "barcode": "AAACCCGGGT", "perturbation": "P002"},
        # Multiplet: cell 102 has 2 distinct barcodes
        {"well": "A1", "tile_pheno": 5, "segmentation_id": 102,
         "barcode": "TTTTTTTTTT", "perturbation": "P003"},
        {"well": "A1", "tile_pheno": 5, "segmentation_id": 102,
         "barcode": "AAAAAAAAAA", "perturbation": "P004"},
    ]
    df = pd.DataFrame(rows)
    out = dedupe_linked_csv(df)
    assert len(out) == 2, "2 singlets kept, multiplet (cell 102) dropped"
    assert set(out["segmentation_id"]) == {100, 101}
    assert set(out["perturbation"]) == {"P001", "P002"}


def test_no_barcode_column_returns_unchanged():
    # Defensive: no `barcode` column → log warning and return as-is.
    df = pd.DataFrame([
        {"well": "A1", "tile_pheno": 5, "segmentation_id": 100,
         "something_else": "foo"},
    ])
    out = dedupe_linked_csv(df)
    assert len(out) == 1
    assert list(out.columns) == list(df.columns)


def test_extra_columns_preserved():
    # Ensure dedupe doesn't strip unrelated columns (matters for _process_well
    # output which carries y_pheno/x_pheno/bbox/etc.)
    df = pd.DataFrame([
        _make_row("A1", 5, 100, "sgABC", "GENE1", barcode="GATTACA", y_pheno=12345),
    ])
    out = dedupe_linked_csv(df)
    assert "barcode" in out.columns
    assert "y_pheno" in out.columns
    assert out.iloc[0]["barcode"] == "GATTACA"
    assert out.iloc[0]["y_pheno"] == 12345


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
