"""Tests for build_pyramids_setup."""

import pytest

EXPERIMENT = "ops9999_20260101"


@pytest.fixture
def hermetic_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_FAST_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_CONFIGS_DIR", str(tmp_path / "configs"))
    (tmp_path / EXPERIMENT / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


class TestBuildNapariPyramidsSetup:
    def test_emits_store_key_position_lines(
        self, hermetic_experiment, capsys, monkeypatch
    ):
        from cyclops_process.processes.pyramids import launcher as mod

        # Stub the store-discovery helper so we don't need real zarr stores.
        def fake_get_positions(experiment, store_key=None):
            if store_key == "pheno_assembled_v3":
                return None, ["A/1/0", "A/2/0"], None
            raise ValueError("not present")

        monkeypatch.setattr(
            mod, "get_positions_for_experiment", fake_get_positions
        )

        mod.build_pyramids_setup(
            experiment=EXPERIMENT, use_v3_stores=True
        )

        lines = capsys.readouterr().out.strip().splitlines()
        assert "PYRAMID_UNIT pheno_assembled_v3:A/1/0" in lines
        assert "PYRAMID_UNIT pheno_assembled_v3:A/2/0" in lines

    def test_no_stores_found_emits_nothing(
        self, hermetic_experiment, capsys, monkeypatch
    ):
        from cyclops_process.processes.pyramids import launcher as mod

        monkeypatch.setattr(
            mod, "get_positions_for_experiment",
            lambda exp, store_key=None: (_ for _ in ()).throw(ValueError("none")),
        )
        mod.build_pyramids_setup(experiment=EXPERIMENT)
        assert capsys.readouterr().out == ""

class TestBuildNapariPyramidsSetupRealData:
    """End-to-end real-data test for dispatch stage 'build_pyramids_setup'.

    Setup for napari pyramids; outputs position:channel items
    """

    @pytest.mark.real_data
    def test_build_pyramids_setup_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        import re

        from cyclops_process.processes.pyramids.launcher import build_pyramids_setup
        from ops_utils.io.zarr_utils import _iter_position_paths

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.stdout import run_and_capture

        # Stdout-driven fan-out: for each v3 store the stage prints "{store_key}:{pos}"
        # (positions discovered via iohub .positions(), so a symlinked .zarr seed is
        # traversed fine; a missing store -> [] -> skipped). Seed the three v3 stores
        # and assert the printed unit set equals the iohub-derived expected set.
        V3_STORES = {
            "pheno_assembled_v3": "3-assembly/phenotyping_v3.zarr",
            "lc_5x_phase_2d_stitched_v3": "1-preprocess/live_imaging/stitch/tracking_phase_2d_stitched_v3.zarr",
            "iss_stitch_registered_v3": "1-preprocess/in_situ_sequencing/register/bc_stitched_registered_v3.zarr",
        }
        seed_from_cache(real_data_workdir, reference_cache, list(V3_STORES.values()))

        expected = set()
        for key, rel in V3_STORES.items():
            for pos in _iter_position_paths(reference_cache / rel):
                expected.add(f"PYRAMID_UNIT {key}:{pos}")
        assert expected, "no positions found in any reference v3 store"

        log_dir = real_data_workdir / "submitit_logs" / "build_pyramids_setup"
        out = run_and_capture(
            "build_pyramids_setup",
            build_pyramids_setup,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
            use_v3_stores=True,
        )
        # Keep only the sentinel-tagged "PYRAMID_UNIT {store_key}:{position}" unit lines
        # (drop "Using store: ..." / "Found N positions" / any banner noise).
        unit_re = re.compile(r"^PYRAMID_UNIT (?:%s):A/\d+/\d+$" % "|".join(map(re.escape, V3_STORES)))
        printed = {ln.strip() for ln in out.splitlines() if unit_re.match(ln.strip())}
        assert printed == expected, f"printed {sorted(printed)} != expected {sorted(expected)}"
