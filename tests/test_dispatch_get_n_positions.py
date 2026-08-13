"""Tests for dispatch_cli command `get_n_positions`.

Maps to cyclops_process.processes.reconstruct_tilt_corrected.get_n_positions.

Function under test counts positions in the phase3d store whose path starts
with the given `well` prefix and prints the count. The store key is selected
by `process` ("pheno" -> lc_20x_phase, "track" -> lc_5x_phase).

Tests scope the experiment dir to tmp_path via OPS_OUTPUT_BASE_DIR and build
minimal HCS-OME-Zarr stores via the project's create_hcs_store_fast helper.
"""

import numpy as np
import pytest

from ops_utils.data.experiment import OpsDataset
from ops_utils.io.zarr_precreate import create_hcs_store_fast

from cyclops_process.processes.reconstruct_tilt_corrected import (
    _resolve_path,
    get_n_positions,
)


EXPERIMENT = "ops9999_20260101"


def _make_phase3d_store(store_path, positions):
    """Create a minimal HCS-OME-Zarr store at store_path with the given positions."""
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
    """Scope OpsDataset paths to tmp_path so we never touch /hpc."""
    monkeypatch.setenv("OPS_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_FAST_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_CONFIGS_DIR", str(tmp_path / "configs"))
    return tmp_path


class TestGetNPositionsPheno:
    """process='pheno' reads from lc_20x_phase."""

    def test_counts_positions_starting_with_well(
        self, hermetic_experiment, capsys
    ):
        dataset = OpsDataset(EXPERIMENT)
        _make_phase3d_store(
            dataset.store_paths["lc_20x_phase"],
            positions=["A/1/0", "A/1/1", "A/2/0", "B/3/0"],
        )

        get_n_positions(EXPERIMENT, "pheno", "A/1")

        assert capsys.readouterr().out.strip() == "2"

    def test_well_prefix_a_matches_all_a_positions(
        self, hermetic_experiment, capsys
    ):
        dataset = OpsDataset(EXPERIMENT)
        _make_phase3d_store(
            dataset.store_paths["lc_20x_phase"],
            positions=["A/1/0", "A/1/1", "A/2/0", "B/3/0"],
        )

        get_n_positions(EXPERIMENT, "pheno", "A")

        assert capsys.readouterr().out.strip() == "3"

    def test_no_matching_well_prints_zero(self, hermetic_experiment, capsys):
        dataset = OpsDataset(EXPERIMENT)
        _make_phase3d_store(
            dataset.store_paths["lc_20x_phase"],
            positions=["A/1/0", "A/2/0"],
        )

        get_n_positions(EXPERIMENT, "pheno", "C")

        assert capsys.readouterr().out.strip() == "0"

    def test_full_position_prefix_matches_exactly_one(
        self, hermetic_experiment, capsys
    ):
        dataset = OpsDataset(EXPERIMENT)
        _make_phase3d_store(
            dataset.store_paths["lc_20x_phase"],
            positions=["A/1/0", "A/1/1"],
        )

        get_n_positions(EXPERIMENT, "pheno", "A/1/0")

        assert capsys.readouterr().out.strip() == "1"


class TestGetNPositionsTrack:
    """process='track' reads from lc_5x_phase, not lc_20x_phase."""

    def test_uses_lc_5x_phase_store(self, hermetic_experiment, capsys):
        dataset = OpsDataset(EXPERIMENT)
        # Only the 5x store is populated — confirms 'track' does not read the 20x store.
        _make_phase3d_store(
            dataset.store_paths["lc_5x_phase"],
            positions=["A/1/0", "A/2/0"],
        )

        get_n_positions(EXPERIMENT, "track", "A")

        assert capsys.readouterr().out.strip() == "2"

    def test_pheno_and_track_stores_are_independent(
        self, hermetic_experiment, capsys
    ):
        dataset = OpsDataset(EXPERIMENT)
        _make_phase3d_store(
            dataset.store_paths["lc_20x_phase"],
            positions=["A/1/0"],
        )
        _make_phase3d_store(
            dataset.store_paths["lc_5x_phase"],
            positions=["A/1/0", "A/2/0", "A/3/0"],
        )

        get_n_positions(EXPERIMENT, "pheno", "A")
        pheno_out = capsys.readouterr().out.strip()

        get_n_positions(EXPERIMENT, "track", "A")
        track_out = capsys.readouterr().out.strip()

        assert pheno_out == "1"
        assert track_out == "3"


class TestGetNPositionsProcessArg:
    def test_process_none_defaults_to_pheno(self, hermetic_experiment, capsys):
        # process=None -> defaults to pheno -> reads lc_20x_phase.
        dataset = OpsDataset(EXPERIMENT)
        _make_phase3d_store(
            dataset.store_paths["lc_20x_phase"],
            positions=["A/1/0", "A/1/1"],
        )

        get_n_positions(EXPERIMENT, None, "A")

        assert capsys.readouterr().out.strip() == "2"

    def test_unknown_process_raises(self, hermetic_experiment):
        with pytest.raises(ValueError, match="Unknown process"):
            get_n_positions(EXPERIMENT, "not_a_real_process", "A")


class TestResolvePath:
    """_resolve_path: identity if path exists; fast_ops/ -> ops/ fallback otherwise."""

    def test_existing_path_returned_unchanged(self, tmp_path):
        p = tmp_path / "real"
        p.mkdir()
        assert _resolve_path(p) == p

    def test_nonexistent_non_fast_path_returned_as_is(self, tmp_path):
        # Path doesn't exist and isn't under fast_ops/ — function returns
        # the original path so callers raise a clear FileNotFoundError.
        p = tmp_path / "missing"
        assert _resolve_path(p) == p

    def test_fast_ops_falls_back_to_ops_when_alt_exists(
        self, tmp_path, monkeypatch
    ):
        # Simulate the production prefix swap by patching the module
        # constants to point at tmp_path subtrees.
        from cyclops_process.processes import reconstruct_tilt_corrected as mod

        fast_root = tmp_path / "fast"
        std_root = tmp_path / "std"
        std_root.mkdir()
        (std_root / "real").mkdir()
        # _FAST_PREFIX/_STD_PREFIX are matched as string prefixes, so they
        # must end in '/'.
        monkeypatch.setattr(mod, "_FAST_PREFIX", str(fast_root) + "/")
        monkeypatch.setattr(mod, "_STD_PREFIX", str(std_root) + "/")

        missing_fast = fast_root / "real"
        resolved = mod._resolve_path(missing_fast)
        assert resolved == std_root / "real"

    def test_fast_ops_returns_original_when_alt_also_missing(
        self, tmp_path, monkeypatch
    ):
        from cyclops_process.processes import reconstruct_tilt_corrected as mod

        fast_root = tmp_path / "fast"
        std_root = tmp_path / "std"
        std_root.mkdir()
        monkeypatch.setattr(mod, "_FAST_PREFIX", str(fast_root) + "/")
        monkeypatch.setattr(mod, "_STD_PREFIX", str(std_root) + "/")

        missing_fast = fast_root / "still_missing"
        # Neither fast nor std has the file — function returns the original.
        assert mod._resolve_path(missing_fast) == missing_fast

class TestGetNPositionsRealData:
    """End-to-end real-data test for the TRACK dispatch stage 'get_n_positions_track'.

    Stdout-driven: the stage prints the number of positions in the track phase3d
    store (lc_5x_phase) whose path starts with `well`; Nextflow captures that
    count to fan out per-position jobs. There is no output artifact to diff, so we
    submit the stage to SLURM, capture its stdout, and assert the printed count
    equals the count we compute independently from the seeded store.
    """

    @pytest.mark.real_data
    def test_get_n_positions_matches_reference(
        self, real_data_workdir, reference_cache
    ):
        from cyclops_process.processes.reconstruct_tilt_corrected import get_n_positions

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.stdout import run_and_capture
        from iohub import open_ome_zarr

        # process='track' reads store_paths['lc_5x_phase'] = tracking_phase.zarr.
        store_rel = "1-preprocess/live_imaging/reconstruction/tracking_phase.zarr"
        seed_from_cache(real_data_workdir, reference_cache, [store_rel])

        well = "A/1"
        seeded = real_data_workdir / REFERENCE_EXPERIMENT / store_rel
        with open_ome_zarr(seeded, mode="r") as store:
            expected = sum(1 for pos, _ in store.positions() if pos.startswith(well))
        assert expected > 0, "seeded track store has no positions under well A/1"

        log_dir = real_data_workdir / "submitit_logs" / "get_n_positions_track"
        out = run_and_capture(
            "get_n_positions_track",
            get_n_positions,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
            well=well,
        )
        assert out.strip() == str(expected), (
            f"stage printed {out.strip()!r}, expected {expected}"
        )


class TestGetNPositionsPhenoRealData:
    """End-to-end real-data test for dispatch stage 'get_n_positions_pheno'.

    Computes position count for pheno fan-out
    """

    @pytest.mark.real_data
    def test_get_n_positions_pheno_matches_reference(
        self, real_data_workdir, reference_cache
    ):
        from cyclops_process.processes.reconstruct_tilt_corrected import get_n_positions

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.stdout import run_and_capture
        from iohub import open_ome_zarr

        # process='pheno' reads store_paths['lc_20x_phase'] = phenotyping_phase.zarr.
        store_rel = "1-preprocess/live_imaging/reconstruction/phenotyping_phase.zarr"
        seed_from_cache(real_data_workdir, reference_cache, [store_rel])

        well = "A/1"
        seeded = real_data_workdir / REFERENCE_EXPERIMENT / store_rel
        with open_ome_zarr(seeded, mode="r") as store:
            expected = sum(1 for pos, _ in store.positions() if pos.startswith(well))
        assert expected > 0, "seeded pheno store has no positions under well A/1"

        log_dir = real_data_workdir / "submitit_logs" / "get_n_positions_pheno"
        out = run_and_capture(
            "get_n_positions_pheno",
            get_n_positions,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
            well=well,
        )
        assert out.strip() == str(expected), (
            f"stage printed {out.strip()!r}, expected {expected}"
        )
