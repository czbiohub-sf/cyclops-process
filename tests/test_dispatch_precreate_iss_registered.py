"""Tests for dispatch_cli command `precreate_iss_registered`.

Maps to cyclops_process.processes.auto_register.iss_cycle_register_orchestrator.precreate_iss_registered.

The public wrapper reads positions from the iss_stitch store, extracts unique
well column indices, and delegates to _precreate_registered_zarr. We test the
wrapper's well-discovery + delegation, plus one end-to-end test that lets the
inner helper run.
"""

import numpy as np
import pytest

from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.io.zarr_precreate import create_hcs_store_fast

from cyclops_process.processes.auto_register import iss_cycle_register_orchestrator as mod


EXPERIMENT = "ops9999_20260101"


def _make_iss_stitch_store(store_path, positions, channel_names=None, version="0.5"):
    store_path.parent.mkdir(parents=True, exist_ok=True)
    create_hcs_store_fast(
        store_path=store_path,
        positions=positions,
        shape=(1, 1, 1, 8, 8),
        chunks=(1, 1, 1, 8, 8),
        dtype=np.float32,
        scale=(1, 1, 1, 1, 1),
        channel_names=channel_names or ["c0", "c1"],
        version=version,
    )


@pytest.fixture
def hermetic_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_FAST_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_CONFIGS_DIR", str(tmp_path / "configs"))
    return tmp_path


class TestPrecreateIssRegisteredWellDiscovery:
    """Verify the well-list passed to _precreate_registered_zarr."""

    def test_unique_wells_extracted(self, hermetic_experiment, monkeypatch):
        dataset = OpsDataset(EXPERIMENT)
        _make_iss_stitch_store(
            dataset.store_paths["iss_stitch"],
            positions=["A/1/0", "A/2/0", "B/1/0"],
        )

        captured = {}

        def fake_precreate(experiment, wells, verbose=True):
            captured["experiment"] = experiment
            captured["wells"] = wells

        monkeypatch.setattr(mod, "_precreate_registered_zarr", fake_precreate)

        mod.precreate_iss_registered(EXPERIMENT)

        assert captured["experiment"] == EXPERIMENT
        assert captured["wells"] == ["A/1/0", "A/2/0", "B/1/0"]

    def test_well_list_is_sorted(self, hermetic_experiment, monkeypatch):
        dataset = OpsDataset(EXPERIMENT)
        # Insert wells in non-sorted order.
        _make_iss_stitch_store(
            dataset.store_paths["iss_stitch"],
            positions=["A/3/0", "A/1/0", "A/2/0"],
        )

        captured = {}
        monkeypatch.setattr(
            mod, "_precreate_registered_zarr",
            lambda exp, wells, verbose=True: captured.setdefault("wells", wells),
        )

        mod.precreate_iss_registered(EXPERIMENT)

        assert captured["wells"] == sorted(captured["wells"])
        assert captured["wells"] == ["A/1/0", "A/2/0", "A/3/0"]

    def test_multiple_fovs_per_well_deduplicated(
        self, hermetic_experiment, monkeypatch
    ):
        dataset = OpsDataset(EXPERIMENT)
        _make_iss_stitch_store(
            dataset.store_paths["iss_stitch"],
            positions=["A/1/0", "A/1/1", "A/2/0", "A/2/1"],
        )

        captured = {}
        monkeypatch.setattr(
            mod, "_precreate_registered_zarr",
            lambda exp, wells, verbose=True: captured.setdefault("wells", wells),
        )

        mod.precreate_iss_registered(EXPERIMENT)

        assert captured["wells"] == ["A/1/0", "A/2/0"]

    def test_single_well(self, hermetic_experiment, monkeypatch):
        dataset = OpsDataset(EXPERIMENT)
        _make_iss_stitch_store(
            dataset.store_paths["iss_stitch"],
            positions=["A/1/0"],
        )

        captured = {}
        monkeypatch.setattr(
            mod, "_precreate_registered_zarr",
            lambda exp, wells, verbose=True: captured.setdefault("wells", wells),
        )

        mod.precreate_iss_registered(EXPERIMENT)

        assert captured["wells"] == ["A/1/0"]


class TestPrecreateIssRegisteredEndToEnd:
    """Let the inner _precreate_registered_zarr run; verify output store shape."""

    def test_creates_registered_store_with_all_wells(
        self, hermetic_experiment
    ):
        from iohub.ngff import open_ome_zarr

        dataset = OpsDataset(EXPERIMENT)
        _make_iss_stitch_store(
            dataset.store_paths["iss_stitch"],
            positions=["A/1/0", "A/2/0", "A/3/0"],
            channel_names=["DAPI", "G", "T", "A", "C"],
        )

        mod.precreate_iss_registered(EXPERIMENT)

        registered_path = dataset.store_paths["iss_stitch_registered_v3"]
        assert registered_path.exists()

        with open_ome_zarr(registered_path, mode="r", version="0.5") as out:
            out_positions = sorted(pos for pos, _ in out.positions())
            assert out_positions == ["A/1/0", "A/2/0", "A/3/0"]
            assert list(out.channel_names) == ["DAPI", "G", "T", "A", "C"]

class TestPrecreateIssRegisteredRealData:
    """End-to-end real-data test for dispatch stage 'precreate_iss_registered'.

    Pre-creates v3 zarr store structure for registration
    """

    @pytest.mark.real_data
    def test_precreate_iss_registered_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.auto_register.iss_cycle_register_orchestrator import precreate_iss_registered

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_ome_zarr
        from fixtures.slurm import submit_stage

        # precreate reads bc_stitched.zarr (positions + per-position shape + channel
        # names) and writes the EMPTY v3 registered-store skeleton (real metadata +
        # empty arrays, no chunk data). Structural-only stage -> seed the input store
        # (symlink, read-through) and compare the skeleton's geometry/metadata to the
        # reference's (filled) v3 store: positions, channels, shape, dtype, scale all
        # match (precreate copies bc_stitched's shape and writes float32 = the v3
        # dtype); pixel data obviously differs (skeleton is empty) -> check_data=False.
        seed_from_cache(
            real_data_workdir,
            reference_cache,
            ['1-preprocess/in_situ_sequencing/stitch/bc_stitched.zarr'],
        )

        log_dir = real_data_workdir / "submitit_logs" / "precreate_iss_registered"
        submit_stage(
            "precreate_iss_registered",
            precreate_iss_registered,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
        )

        rel = "1-preprocess/in_situ_sequencing/register/bc_stitched_registered_v3.zarr"
        candidate = real_data_workdir / REFERENCE_EXPERIMENT / rel
        reference = reference_cache / rel
        compare_ome_zarr(candidate, reference, check_data=False)
