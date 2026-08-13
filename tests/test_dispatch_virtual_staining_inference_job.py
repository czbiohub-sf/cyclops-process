"""Tests for virtual_staining_inference_job."""

import pytest

EXPERIMENT = "ops9999_20260101"


@pytest.fixture
def hermetic_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_FAST_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_CONFIGS_DIR", str(tmp_path / "configs"))
    (tmp_path / EXPERIMENT / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


class TestVirtualStainingInferenceJob:
    def test_invalid_dim_raises(self, hermetic_experiment):
        from cyclops_process.processes import virtual_staining as mod
        with pytest.raises(ValueError, match="dim must be one of"):
            mod.virtual_staining_inference_job(
                experiment=EXPERIMENT, process="track", dim="4d",
            )

    def test_unknown_process_raises(self, hermetic_experiment):
        from cyclops_process.processes import virtual_staining as mod
        with pytest.raises(ValueError, match="process must be"):
            mod.virtual_staining_inference_job(
                experiment=EXPERIMENT, process="bogus", dim="3d",
            )

    def test_empty_position_range_returns_early(
        self, hermetic_experiment, capsys
    ):
        # job_index=5, num_jobs=2, num_positions=10 -> start_pos=25 > 10
        from cyclops_process.processes import virtual_staining as mod
        mod.virtual_staining_inference_job(
            experiment=EXPERIMENT, process="track", dim="3d",
            job_index=5, num_jobs=2, num_positions=10,
        )
        out = capsys.readouterr().out
        assert "no positions to process" in out

class TestVirtualStainingInferenceJobRealData:
    """End-to-end real-data test for dispatch stage 'virtual_staining_inference_job'.

    Fan-out per job; inference; modifies shards
    """

    @pytest.mark.real_data
    @pytest.mark.skip(
        reason="vs_inference_job modifies VS shards IN PLACE and is fanned out per "
        "job_index; seeding the cache's post-inference tracking_vs/ and re-running "
        "would double-infer, and a single shard covers only some positions. Needs a "
        "pre-inference input (not separately cached) + position-subset compare."
    )
    def test_virtual_staining_inference_job_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.virtual_staining import virtual_staining_inference_job

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_ome_zarr_directory
        from fixtures.slurm import submit_stage

        seed_from_cache(
            real_data_workdir,
            reference_cache,
            ['1-preprocess/live_imaging/virtual_staining/tracking_vs'],
        )

        log_dir = real_data_workdir / "submitit_logs" / "virtual_staining_inference_job"
        submit_stage(
            "virtual_staining_inference_job",
            virtual_staining_inference_job,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
            job_id=0,
            n_jobs=1,
            n_positions=25,
        )

        candidate = real_data_workdir / REFERENCE_EXPERIMENT / "1-preprocess/live_imaging/virtual_staining/tracking_vs"
        reference = reference_cache / "1-preprocess/live_imaging/virtual_staining/tracking_vs"
        compare_ome_zarr_directory(
            candidate, reference, pattern="*.zarr", rtol=1e-5
        )



class TestVsInferenceJobPhenoRealData:
    """End-to-end real-data test for dispatch stage 'vs_inference_job_pheno'.

    Pheno VS inference fan-out
    """

    @pytest.mark.real_data
    @pytest.mark.skip(
        reason="vs_inference_job modifies VS shards IN PLACE and is fanned out per "
        "job_index; seeding the cache's post-inference phenotyping_vs/ and re-running "
        "would double-infer, and a single shard covers only some positions. Needs a "
        "pre-inference input (not separately cached) + position-subset compare."
    )
    def test_vs_inference_job_pheno_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.virtual_staining import virtual_staining_inference_job

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_ome_zarr_directory
        from fixtures.slurm import submit_stage

        seed_from_cache(
            real_data_workdir,
            reference_cache,
            ['1-preprocess/live_imaging/virtual_staining/phenotyping_vs'],
        )

        log_dir = real_data_workdir / "submitit_logs" / "vs_inference_job_pheno"
        submit_stage(
            "vs_inference_job_pheno",
            virtual_staining_inference_job,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
            job_id=0,
            n_jobs=1,
            n_positions=25,
        )

        candidate = real_data_workdir / REFERENCE_EXPERIMENT / "1-preprocess/live_imaging/virtual_staining/phenotyping_vs"
        reference = reference_cache / "1-preprocess/live_imaging/virtual_staining/phenotyping_vs"
        compare_ome_zarr_directory(
            candidate, reference, pattern="*.zarr", rtol=1e-5
        )

