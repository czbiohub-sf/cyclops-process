"""Tests for dispatch_cli command `reconstruct_tilt_corrected`.

Maps to cyclops_process.processes.reconstruct_tilt_corrected.reconstruct_tilt_corrected.

This is a thin orchestrator over submit_jobs with specific kwargs. Tests
verify the kwarg contract.
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


class TestReconstructTiltCorrected:
    def test_calls_submit_jobs_with_skip_calibration_and_default_chunk_size(
        self, hermetic_experiment, monkeypatch
    ):
        from cyclops_process.processes import reconstruct_tilt_corrected as mod

        captured = {}

        def fake_submit(experiment, **kwargs):
            captured["experiment"] = experiment
            captured.update(kwargs)

        monkeypatch.setattr(mod, "submit_jobs", fake_submit)

        mod.reconstruct_tilt_corrected(
            EXPERIMENT, process="track", wells=["A/1"], skip_precheck=True
        )

        assert captured["experiment"] == EXPERIMENT
        assert captured["wells"] == ["A/1"]
        assert captured["process"] == "track"
        assert captured["chunk_size"] == 150
        assert captured["wait_for_completion"] is True
        assert captured["skip_precheck"] is True
        assert captured["_skip_calibration"] is True
