"""Tests for virtual_staining_preprocess."""

import pytest


EXPERIMENT = "ops9999_20260101"


@pytest.fixture
def hermetic_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_FAST_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_CONFIGS_DIR", str(tmp_path / "configs"))
    (tmp_path / EXPERIMENT / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


class TestVirtualStainingPreprocess:
    def test_invalid_dim_raises(self, hermetic_experiment):
        from cyclops_process.processes import virtual_staining as mod
        with pytest.raises(ValueError, match="dim must be one of"):
            mod.virtual_staining_preprocess(
                experiment=EXPERIMENT, process="track", dim="4d"
            )

    def test_unknown_process_raises(self, hermetic_experiment):
        from cyclops_process.processes import virtual_staining as mod
        with pytest.raises(ValueError, match="Unknown process"):
            mod.virtual_staining_preprocess(
                experiment=EXPERIMENT, process="bogus", dim="3d"
            )

class TestVirtualStainingPreprocessRealData:
    """End-to-end real-data test for dispatch stage 'virtual_staining_preprocess'.

    Prepares data for VS inference; intermediate shards
    """

    @pytest.mark.real_data
    @pytest.mark.skip(
        reason="vs_preprocess writes intermediate VS shards that vs_inference_job "
        "overwrites in place; the cache's tracking_vs/ holds POST-inference shards, "
        "so there is no pre-inference reference to compare. Validate the final "
        "combined .zarr via vs_combine_* instead."
    )
    def test_virtual_staining_preprocess_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.virtual_staining import virtual_staining_preprocess

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_ome_zarr_directory
        from fixtures.slurm import submit_stage

        seed_from_cache(
            real_data_workdir,
            reference_cache,
            ['1-preprocess/live_imaging/reconstruction/tracking_phase_2d_optimized.zarr'],
        )

        log_dir = real_data_workdir / "submitit_logs" / "virtual_staining_preprocess"
        submit_stage(
            "virtual_staining_preprocess",
            virtual_staining_preprocess,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
        )

        candidate = real_data_workdir / REFERENCE_EXPERIMENT / "1-preprocess/live_imaging/virtual_staining/tracking_vs"
        reference = reference_cache / "1-preprocess/live_imaging/virtual_staining/tracking_vs"
        compare_ome_zarr_directory(
            candidate, reference, pattern="*.zarr", rtol=1e-5
        )



class TestVsPreprocessPhenoRealData:
    """End-to-end real-data test for dispatch stage 'vs_preprocess_pheno'.

    Preprocesses pheno for VS
    """

    @pytest.mark.real_data
    @pytest.mark.skip(
        reason="vs_preprocess writes intermediate VS shards that vs_inference_job "
        "overwrites in place; the cache's phenotyping_vs/ holds POST-inference shards, "
        "so there is no pre-inference reference to compare. Validate the final "
        "combined .zarr via vs_combine_* instead."
    )
    def test_vs_preprocess_pheno_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.virtual_staining import virtual_staining_preprocess

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_ome_zarr_directory
        from fixtures.slurm import submit_stage

        seed_from_cache(
            real_data_workdir,
            reference_cache,
            ['1-preprocess/live_imaging/virtual_staining/phenotyping_max_proj.zarr'],
        )

        log_dir = real_data_workdir / "submitit_logs" / "vs_preprocess_pheno"
        submit_stage(
            "vs_preprocess_pheno",
            virtual_staining_preprocess,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
        )

        candidate = real_data_workdir / REFERENCE_EXPERIMENT / "1-preprocess/live_imaging/virtual_staining/phenotyping_vs"
        reference = reference_cache / "1-preprocess/live_imaging/virtual_staining/phenotyping_vs"
        compare_ome_zarr_directory(
            candidate, reference, pattern="*.zarr", rtol=1e-5
        )

