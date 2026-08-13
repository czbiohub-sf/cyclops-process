import pytest

"""Tests for submit_registration_jobs."""

import inspect


class TestSubmitRegistrationJobsSignature:
    def test_signature_exposes_expected_kwargs(self):
        from cyclops_process.processes.auto_register import auto_register_orchestrator as mod
        sig = inspect.signature(mod.submit_registration_jobs)
        for name in (
            "experiment", "wells", "registration_types", "slurm_params",
            "dry_run", "wait_for_completion", "skip_prompt", "skip_track",
            "force",
        ):
            assert name in sig.parameters, f"missing kwarg: {name}"

    def test_default_registration_types_includes_iss_and_pheno(self):
        from cyclops_process.processes.auto_register import auto_register_orchestrator as mod
        sig = inspect.signature(mod.submit_registration_jobs)
        default_types = sig.parameters["registration_types"].default
        assert default_types == ["iss", "pheno"]

    def test_default_wells_is_one_two_three(self):
        from cyclops_process.processes.auto_register import auto_register_orchestrator as mod
        sig = inspect.signature(mod.submit_registration_jobs)
        assert sig.parameters["wells"].default == [1, 2, 3]

class TestSubmitRegistrationJobsRealData:
    """End-to-end real-data test for dispatch stage 'submit_registration_jobs'.

    Orchestrator: submits registration jobs to SLURM
    """

    @pytest.mark.real_data
    def test_submit_registration_jobs_matches_reference(
        self, real_data_workdir, reference_cache
    ):
        """Dry-run orchestrator: assert the plan it WOULD submit. The merged code
        guards on the v3 source stores existing before planning (else PipelineHalted),
        so seed the registered-ISS + assembled-pheno v3 stores from the reference
        (ops0161 has both). dry_run=True runs in-process and returns
        ``{"dry_run": True, "jobs": [...]}``."""
        from cyclops_process.processes.auto_register.auto_register_orchestrator import (
            submit_registration_jobs,
        )

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.stdout import assert_dry_run_plan

        seed_from_cache(
            real_data_workdir,
            reference_cache,
            [
                "1-preprocess/in_situ_sequencing/register/bc_stitched_registered_v3.zarr",
                "3-assembly/phenotyping_v3.zarr",
            ],
        )

        result = submit_registration_jobs(
            REFERENCE_EXPERIMENT,
            wells=[1],
            registration_types=["iss", "pheno"],
            dry_run=True,
            wait_for_completion=False,
            skip_prompt=True,
        )

        assert result["dry_run"] is True, "dry_run must not submit jobs"
        # iss + pheno for the single well -> 2 planned jobs.
        assert_dry_run_plan(
            result["jobs"],
            expect_jobs=2,
            contains=["iss_to_track_w1", "pheno_to_track_w1"],
        )
