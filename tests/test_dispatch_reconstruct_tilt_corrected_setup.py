"""Tests for dispatch_cli command `reconstruct_tilt_corrected_setup`.

Maps to cyclops_process.processes.reconstruct_tilt_corrected
  .reconstruct_tilt_corrected_setup.

Function discovers positions in phase3d store, validates them, and prints
'well start end' lines for Nextflow fan-out. Tests verify chunk math and
skip_precheck.
"""

import numpy as np
import pytest

from ops_utils.data.experiment import OpsDataset
from ops_utils.io.zarr_precreate import create_hcs_store_fast


EXPERIMENT = "ops9999_20260101"


def _make_phase3d_store(store_path, positions):
    store_path.parent.mkdir(parents=True, exist_ok=True)
    create_hcs_store_fast(
        store_path=store_path,
        positions=positions,
        shape=(1, 1, 1, 4, 4),
        chunks=(1, 1, 1, 4, 4),
        dtype=np.float32,
        scale=(1, 1, 1, 1, 1),
        channel_names=["phase"],
    )


@pytest.fixture
def hermetic_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_FAST_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_CONFIGS_DIR", str(tmp_path / "configs"))
    (tmp_path / EXPERIMENT / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


class TestReconstructTiltCorrectedSetup:
    def test_prints_one_line_per_chunk_per_well_skip_precheck(
        self, hermetic_experiment, capsys, monkeypatch
    ):
        from cyclops_process.processes import reconstruct_tilt_corrected as mod

        dataset = OpsDataset(EXPERIMENT)
        # Pheno phase3d store. 2 wells × 3 positions each = 6 positions, all
        # under chunk_size=150 so each well emits a single chunk line.
        _make_phase3d_store(
            dataset.store_paths["lc_20x_phase"],
            positions=[
                "A/1/0", "A/1/1", "A/1/2",
                "A/2/0", "A/2/1", "A/2/2",
            ],
        )
        # Stub well discovery so we don't hit any filesystem dependency.
        monkeypatch.setattr(
            mod, "get_experiment_wells",
            lambda exp, prefix_only=True: ["A/1", "A/2"],
        )

        mod.reconstruct_tilt_corrected_setup(
            EXPERIMENT, process="pheno", skip_precheck=True,
        )

        out_lines = capsys.readouterr().out.strip().splitlines()
        # Each well's positions count is 3 -> 1 chunk -> "well 0 3".
        assert "A/1 0 3" in out_lines
        assert "A/2 0 3" in out_lines

    def test_chunk_math_splits_when_positions_exceed_chunk_size(
        self, hermetic_experiment, capsys, monkeypatch
    ):
        from cyclops_process.processes import reconstruct_tilt_corrected as mod

        dataset = OpsDataset(EXPERIMENT)
        # 151 positions in well A/1 → chunk_size=150 → 2 chunks.
        positions = [f"A/1/{i}" for i in range(151)]
        _make_phase3d_store(
            dataset.store_paths["lc_20x_phase"], positions=positions
        )
        monkeypatch.setattr(
            mod, "get_experiment_wells",
            lambda exp, prefix_only=True: ["A/1"],
        )

        mod.reconstruct_tilt_corrected_setup(
            EXPERIMENT, process="pheno", skip_precheck=True,
        )

        out_lines = capsys.readouterr().out.strip().splitlines()
        # Expect two chunks for A/1.
        assert "A/1 0 150" in out_lines
        assert "A/1 150 151" in out_lines
