"""Tests for build_pyramids_position_job."""

import pytest

EXPERIMENT = "ops9999_20260101"


@pytest.fixture
def hermetic_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_FAST_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_CONFIGS_DIR", str(tmp_path / "configs"))
    (tmp_path / EXPERIMENT / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


class TestBuildNapariPositionJob:
    def test_unknown_store_key_raises_valueerror(self, hermetic_experiment):
        from cyclops_process.processes.pyramids import launcher as mod

        with pytest.raises(ValueError, match="not found"):
            mod.build_pyramids_position_job(
                experiment=EXPERIMENT,
                store_key="nonexistent_store",
                position="A/1/0",
            )

class TestBuildNapariPositionJobRealData:
    """End-to-end real-data test for dispatch stage 'build_pyramids_position_job'.

    Fan-out per position+channel; builds pyramids
    """

    @pytest.mark.real_data
    @pytest.mark.skip(
        reason="fanned out per (store_key, position, channel); builds pyramid levels "
        "IN PLACE into phenotyping_v3.zarr for one position. A single shard touches "
        "only one position and only adds downscaled levels, so there is no clean "
        "whole-store reference to compare. Needs a full-fan-out run or a per-position "
        "pyramid-level check."
    )
    def test_build_pyramids_position_job_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.pyramids.launcher import build_pyramids_position_job

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_ome_zarr
        from fixtures.slurm import submit_stage

        seed_from_cache(
            real_data_workdir,
            reference_cache,
            ['3-assembly/phenotyping_v3.zarr'],
        )

        log_dir = real_data_workdir / "submitit_logs" / "build_pyramids_position_job"
        submit_stage(
            "build_pyramids_position_job",
            build_pyramids_position_job,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
            channel='DAPI',
            position='A/1/000000',
        )

        candidate = real_data_workdir / REFERENCE_EXPERIMENT / "3-assembly/phenotyping_v3.zarr"
        reference = reference_cache / "3-assembly/phenotyping_v3.zarr"
        compare_ome_zarr(candidate, reference, rtol=1e-5, check_data=False)

