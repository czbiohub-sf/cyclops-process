"""Tests for dispatch_cli command `correct_distortion`.

Maps to cyclops_process.processes.reconstruct.correct_distortion.

The function reads a source zarr, builds distortion coordinates, then
processes per-position. Tests cover the print line announcing the worker
count + process mode (the simplest verifiable behavior).
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


class TestCorrectDistortion:
    def test_prints_worker_count_and_process(
        self, hermetic_experiment, capsys, monkeypatch
    ):
        # The function fails when it tries to read a non-existent source zarr;
        # but only AFTER printing the worker-count line. We just verify the
        # log line shows up.
        from cyclops_process.processes import reconstruct as mod

        monkeypatch.setattr(mod, "get_optimal_workers", lambda **kw: 4)

        with pytest.raises(Exception):
            mod.correct_distortion(experiment=EXPERIMENT, process="lc_5x")

        out = capsys.readouterr().out
        assert "Correcting distortion" in out
        assert "process=lc_5x" in out
        assert "4 workers" in out

class TestCorrectDistortionRealData:
    """End-to-end real-data test for dispatch stage 'correct_distortion'.

    Corrects brightfield distortion for tracking
    """

    @pytest.mark.real_data
    def test_correct_distortion_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.reconstruct import correct_distortion

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_ome_zarr
        from fixtures.slurm import submit_stage

        seed_from_cache(
            real_data_workdir,
            reference_cache,
            ['0-convert/live_imaging/tracking_symlink.zarr'],
        )

        log_dir = real_data_workdir / "submitit_logs" / "correct_distortion"
        # Stage key in nextflow_ops_args.yaml is 'correct_distortion_bf'
        # (process=lc_5x BF reconstruction), not 'correct_distortion'.
        # The yaml python_kwargs set ngff_version='0.5', but the reference
        # cache stores are NGFF v0.4 — override so the stage reads them.
        submit_stage(
            "correct_distortion_bf",
            correct_distortion,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
            ngff_version=reference_ngff_version,
        )

        candidate = real_data_workdir / REFERENCE_EXPERIMENT / "1-preprocess/live_imaging/reconstruction/tracking_bf_corrected.zarr"
        reference = reference_cache / "1-preprocess/live_imaging/reconstruction/tracking_bf_corrected.zarr"
        compare_ome_zarr(candidate, reference, rtol=1e-5)

