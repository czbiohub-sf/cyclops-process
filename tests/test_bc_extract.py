"""Tests for cyclops_process.bc.extract (base-calling, pure numpy/pandas)."""

import numpy as np
import pandas as pd
import pytest

from cyclops_process.bc.extract import (
    _call_bases_fast,
    _quality,
    _transform_medians,
    call_reads,
)


class TestCallBasesFast:
    def test_argmax_to_bases(self):
        # shape: (N reads=2, cycles=3, channels=4); bases = "ACGT"
        values = np.array(
            [
                [  # read 0
                    [9, 1, 1, 1],  # A
                    [1, 1, 9, 1],  # G
                    [1, 1, 1, 9],  # T
                ],
                [  # read 1
                    [1, 9, 1, 1],  # C
                    [9, 1, 1, 1],  # A
                    [1, 9, 1, 1],  # C
                ],
            ]
        )
        out = _call_bases_fast(values, list("ACGT"))
        assert out == ["AGT", "CAC"]

    def test_shape_assertion(self):
        with pytest.raises(AssertionError):
            _call_bases_fast(np.zeros((2, 3)), ["A", "C"])

    def test_channels_must_match_bases(self):
        with pytest.raises(AssertionError):
            _call_bases_fast(np.zeros((1, 1, 4)), ["A", "C"])


class TestQuality:
    def test_pure_signal_is_max_quality(self):
        # One channel huge, rest zero -> Q clipped to 1.
        # _quality collapses the channels axis, so Q.shape == X.shape[:-1].
        X = np.array([[1000.0, 0.0, 0.0, 0.0]])
        Q = _quality(X)
        assert Q.shape == (1,)
        assert Q[0] == pytest.approx(1.0)

    def test_tied_top_two_is_zero_quality(self):
        # Top two channels equal -> log ratio = 1 -> Q = 1 - 1 = 0, clipped
        X = np.array([[100.0, 100.0, 0.0, 0.0]])
        Q = _quality(X)
        assert Q[0] == pytest.approx(0.0)

    def test_output_clipped_to_unit_interval(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1000, size=(20, 4))
        Q = _quality(X)
        assert Q.shape == (20,)
        assert (Q >= 0).all() and (Q <= 1).all()


class TestTransformMedians:
    def test_clean_signal_recovers_identity(self):
        # Build X where each row has exactly one bright channel, rotating
        # through all 4. With no cross-talk, the correction matrix should be
        # very close to identity and Y should match X (up to int rounding).
        X = np.tile(np.eye(4) * 100, (10, 1))  # 40 x 4
        Y, W = _transform_medians(X)
        np.testing.assert_allclose(W, np.eye(4), atol=1e-9)
        np.testing.assert_array_equal(Y, X.astype(int))

    def test_unseen_channel_uses_pseudo_median(self):
        # Channel 3 is never the max -> pseudo_median fallback inside
        # get_medians prevents the matrix from collapsing.
        X = np.array(
            [
                [100, 0, 0, 0],
                [0, 100, 0, 0],
                [0, 0, 100, 0],
            ],
            dtype=float,
        )
        Y, W = _transform_medians(X)
        assert W.shape == (4, 4)
        # No NaNs / infs from divide-by-zero
        assert np.isfinite(W).all()


class TestCleanUpBases:
    def test_sort_order(self):
        from cyclops_process.bc.extract import _clean_up_bases

        df = pd.DataFrame(
            {
                "cell": [2, 1, 1, 2],
                "read": [0, 1, 0, 1],
                "cycle": [0, 0, 1, 0],
                "channel": ["C", "A", "A", "A"],
                "intensity": [1, 2, 3, 4],
            }
        )
        out = _clean_up_bases(df).reset_index(drop=True)
        expected_order = sorted(zip(df["cell"], df["read"], df["cycle"], df["channel"]))
        assert (
            list(zip(out["cell"], out["read"], out["cycle"], out["channel"]))
            == expected_order
        )


class TestCallReads:
    @pytest.fixture
    def df_bases(self):
        # 2 cells, 1 read per cell, 2 cycles, 4 channels ("ACGT")
        # Cell 1 read 0: cycle 0 -> A, cycle 1 -> C  => "AC"
        # Cell 2 read 0: cycle 0 -> G, cycle 1 -> T  => "GT"
        rows = []
        intensities = {
            (1, 0, 0): {"A": 100, "C": 5, "G": 5, "T": 5},
            (1, 0, 1): {"A": 5, "C": 100, "G": 5, "T": 5},
            (2, 0, 0): {"A": 5, "C": 5, "G": 100, "T": 5},
            (2, 0, 1): {"A": 5, "C": 5, "G": 5, "T": 100},
        }
        for (cell, read, cycle), per_channel in intensities.items():
            for ch, val in per_channel.items():
                rows.append(
                    dict(
                        cell=cell,
                        read=read,
                        cycle=cycle,
                        channel=ch,
                        intensity=val,
                        i=cell * 10,
                        j=cell * 10,
                    )
                )
        return pd.DataFrame(rows)

    def test_returns_none_when_no_cell_reads(self, df_bases):
        # correction_only_in_cells=True requires at least one cell>0
        df_all_bg = df_bases.assign(cell=0)
        assert call_reads(df_all_bg) is None

    def test_returns_none_for_none_input(self):
        assert call_reads(None) is None

    def test_barcodes_called_correctly(self, df_bases):
        reads = call_reads(df_bases, correction_only_in_cells=True)
        assert reads is not None
        assert set(reads["cell"]) == {1, 2}
        # One row per (cell, read) pair
        assert len(reads) == 2
        barcode_by_cell = dict(zip(reads["cell"], reads["barcode"]))
        assert barcode_by_cell[1] == "AC"
        assert barcode_by_cell[2] == "GT"

    def test_drops_per_cycle_columns(self, df_bases):
        reads = call_reads(df_bases)
        for dropped in ("cycle", "channel", "intensity"):
            assert dropped not in reads.columns
        # Per-cycle quality columns remain
        assert {"Q_0", "Q_1", "Q_min"}.issubset(reads.columns)
