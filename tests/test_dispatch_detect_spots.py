"""Tests for detect_spots."""

import inspect

import pytest


class TestDetectSpotsSignature:
    def test_signature_takes_experiment_and_reproducible(self):
        from cyclops_process.processes import spots as mod
        sig = inspect.signature(mod.detect_spots)
        assert "experiment" in sig.parameters
        assert "reproducible" in sig.parameters
        # `reproducible` defaults to None (config-driven).
        assert sig.parameters["reproducible"].default is None

class TestDetectSpotsRealData:
    """End-to-end real-data test for dispatch stage 'detect_spots'.

    Detects fluorescent spots; outputs .npy file used by base_calling
    """

    @pytest.mark.real_data
    @pytest.mark.skip(reason="stage 'detect_spots' is an orchestrator / stdout-driven stage with no single output to compare; needs stage-specific test logic")
    def test_detect_spots_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.spots import detect_spots

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.slurm import submit_stage

        # (skipped; upstream seeding not exercised)

        log_dir = real_data_workdir / "submitit_logs" / "detect_spots"
        submit_stage(
            "detect_spots",
            detect_spots,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
        )
