"""Tests for virtual_staining_inference_setup."""

import pytest

EXPERIMENT = "ops9999_20260101"


@pytest.fixture
def hermetic_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_FAST_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_CONFIGS_DIR", str(tmp_path / "configs"))
    (tmp_path / EXPERIMENT / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


class TestVirtualStainingInferenceSetup:
    def test_invalid_dim_raises(self, hermetic_experiment):
        from cyclops_process.processes import virtual_staining as mod
        with pytest.raises(ValueError, match="dim must be one of"):
            mod.virtual_staining_inference_setup(
                experiment=EXPERIMENT, process="track", dim="4d"
            )

    def test_unknown_process_raises(self, hermetic_experiment):
        from cyclops_process.processes import virtual_staining as mod
        with pytest.raises(ValueError, match="process must be"):
            mod.virtual_staining_inference_setup(
                experiment=EXPERIMENT, process="bogus", dim="3d"
            )

class TestVirtualStainingInferenceSetupRealData:
    """End-to-end real-data test for dispatch stage 'virtual_staining_inference_setup'.

    Computes inference job params; stdout driven (nj, np)
    """

    @pytest.mark.real_data
    def test_virtual_staining_inference_setup_matches_reference(
        self, real_data_workdir, reference_cache
    ):
        """Stdout-driven fan-out setup (TRACK). vs_inference_setup_track runs with
        process='track', dim='2d', so it reads lc_5x_phase_2d_optimized, counts its
        position dirs with `find -mindepth 3 -maxdepth 3 -type d`, and prints
        "{num_array_jobs} {num_positions}" (num_array_jobs defaults to 1 for track)
        for Nextflow to flatMap. We seed that store as a real dir skeleton (the
        stage's `find -type d` won't traverse a .zarr symlink), count positions the
        same way, and assert the printed pair."""
        import subprocess

        from cyclops_process.processes.virtual_staining import (
            virtual_staining_inference_setup,
        )

        from conftest import REFERENCE_EXPERIMENT, seed_dir_skeleton
        from fixtures.stdout import run_and_capture

        store_rel = (
            "1-preprocess/live_imaging/reconstruction/tracking_phase_2d_optimized.zarr"
        )
        seeded = seed_dir_skeleton(real_data_workdir, reference_cache, store_rel)

        find = subprocess.run(
            ["find", str(seeded), "-mindepth", "3", "-maxdepth", "3", "-type", "d"],
            capture_output=True, text=True, check=True,
        )
        num_positions = len([ln for ln in find.stdout.strip().split("\n") if ln])
        assert num_positions > 0, "seeded track store has no position dirs"

        log_dir = real_data_workdir / "submitit_logs" / "vs_inference_setup_track"
        out = run_and_capture(
            "vs_inference_setup_track",
            virtual_staining_inference_setup,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
        )
        # track default num_array_jobs == 1.
        assert out == f"1 {num_positions}", f"stage printed {out!r}"


class TestVsInferenceSetupPhenoRealData:
    """End-to-end real-data test for dispatch stage 'vs_inference_setup_pheno'.

    Setup for pheno VS inference
    """

    @pytest.mark.real_data
    def test_vs_inference_setup_pheno_matches_reference(
        self, real_data_workdir, reference_cache
    ):
        """Stdout-driven fan-out setup (PHENO). vs_inference_setup_pheno runs with
        process='pheno', dim='3d', so it reads lc_20x_phase_3d_optimized and prints
        "{num_array_jobs} {num_positions}" (num_array_jobs defaults to 10 for
        pheno). Seeded as a real dir skeleton -- see the track variant."""
        import subprocess

        from cyclops_process.processes.virtual_staining import (
            virtual_staining_inference_setup,
        )

        from conftest import REFERENCE_EXPERIMENT, seed_dir_skeleton
        from fixtures.stdout import run_and_capture

        store_rel = (
            "1-preprocess/live_imaging/reconstruction/phenotyping_phase_optimized.zarr"
        )
        seeded = seed_dir_skeleton(real_data_workdir, reference_cache, store_rel)

        find = subprocess.run(
            ["find", str(seeded), "-mindepth", "3", "-maxdepth", "3", "-type", "d"],
            capture_output=True, text=True, check=True,
        )
        num_positions = len([ln for ln in find.stdout.strip().split("\n") if ln])
        assert num_positions > 0, "seeded pheno store has no position dirs"

        log_dir = real_data_workdir / "submitit_logs" / "vs_inference_setup_pheno"
        out = run_and_capture(
            "vs_inference_setup_pheno",
            virtual_staining_inference_setup,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
        )
        # pheno default num_array_jobs == 10.
        assert out == f"10 {num_positions}", f"stage printed {out!r}"
