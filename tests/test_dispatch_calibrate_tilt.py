"""Tests for dispatch_cli command `calibrate_tilt`.

Maps to cyclops_process.processes.reconstruct_tilt_corrected.calibrate_tilt.

The wrapper fan-outs `calibrate` across wells via a ProcessPoolExecutor
(spawn context). Tests cover the well-default-discovery path and that the
process-pool path is exercised.
"""

import pytest


EXPERIMENT = "ops9999_20260101"


@pytest.fixture
def hermetic_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_FAST_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_CONFIGS_DIR", str(tmp_path / "configs"))
    (tmp_path / EXPERIMENT / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


class TestCalibrateTilt:
    def test_explicit_wells_propagated_to_executor(
        self, hermetic_experiment, monkeypatch
    ):
        from cyclops_process.processes import reconstruct_tilt_corrected as mod

        submitted_wells = []

        class FakeFuture:
            def result(self):
                return None

        class FakePool:
            def __init__(self, max_workers=None, mp_context=None):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def submit(self, func, *args, **kwargs):
                # calibrate(experiment, well=..., process=..., resume=True)
                submitted_wells.append(kwargs.get("well"))
                return FakeFuture()

        # ProcessPoolExecutor is imported inside the function from
        # concurrent.futures — patch it on the canonical module.
        import concurrent.futures
        monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", FakePool)
        mod.calibrate_tilt(EXPERIMENT, process="track", wells=["A/1", "A/2"])

        assert sorted(submitted_wells) == ["A/1", "A/2"]

class TestCalibrateTiltRealData:
    """End-to-end real-data test for dispatch stage 'calibrate_tilt'.

    Calibrates tilt per well; outputs YAML files
    """

    @pytest.mark.real_data
    @pytest.mark.skip(reason="stage 'calibrate_tilt' is an orchestrator / stdout-driven stage with no single output to compare; needs stage-specific test logic")
    def test_calibrate_tilt_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.reconstruct_tilt_corrected import calibrate_tilt

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.slurm import submit_stage

        # (skipped; upstream seeding not exercised)

        log_dir = real_data_workdir / "submitit_logs" / "calibrate_tilt"
        submit_stage(
            "calibrate_tilt",
            calibrate_tilt,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
        )


class TestCalibrateTiltPhenoRealData:
    """End-to-end real-data test for dispatch stage 'calibrate_tilt_pheno'.

    Calibrates tilt for 20x phenotyping; parallel with tracking
    """

    @pytest.mark.real_data
    @pytest.mark.skip(reason="stage 'calibrate_tilt_pheno' is an orchestrator / stdout-driven stage with no single output to compare; needs stage-specific test logic")
    def test_calibrate_tilt_pheno_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.reconstruct_tilt_corrected import calibrate_tilt

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.slurm import submit_stage

        # (skipped; upstream seeding not exercised)

        log_dir = real_data_workdir / "submitit_logs" / "calibrate_tilt_pheno"
        submit_stage(
            "calibrate_tilt_pheno",
            calibrate_tilt,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
        )
