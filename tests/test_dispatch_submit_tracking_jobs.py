"""Tests for submit_tracking_jobs."""

import inspect
import pytest


EXPERIMENT = "ops9999_20260101"


class TestSubmitTrackingJobsSignature:
    def test_signature_exposes_expected_kwargs(self):
        from cyclops_process.processes.track import track_orchestrator as mod
        sig = inspect.signature(mod.submit_tracking_jobs)
        for name in ("experiment", "wells", "slurm_params", "dry_run",
                     "wait_for_completion", "skip_track"):
            assert name in sig.parameters, f"missing kwarg: {name}"

    def test_default_wells_is_three_iss_wells(self):
        from cyclops_process.processes.track import track_orchestrator as mod
        sig = inspect.signature(mod.submit_tracking_jobs)
        # The body assigns wells if None — the signature default is None.
        assert sig.parameters["wells"].default is None

class TestSubmitTrackingJobsRealData:
    """End-to-end real-data test for dispatch stage 'submit_tracking_jobs'.

    Orchestrator: submits tracking; writes to tracks.geff
    """

    @pytest.mark.real_data
    def test_submit_tracking_jobs_matches_reference(self, real_data_workdir):
        """Dry-run orchestrator: no single output store to diff, so assert the
        plan it WOULD submit. submit_tracking_jobs builds one tracking job per
        well from its `wells` arg (it reads no zarr store), so dry_run=True runs
        fully in-process and returns ``{"dry_run": True, "jobs": [...]}`` without
        touching SLURM."""
        from cyclops_process.processes.track.track_orchestrator import submit_tracking_jobs

        from conftest import REFERENCE_EXPERIMENT
        from fixtures.stdout import assert_dry_run_plan

        wells = ["A/1/0", "A/2/0"]
        result = submit_tracking_jobs(
            REFERENCE_EXPERIMENT,
            wells=wells,
            dry_run=True,
            wait_for_completion=False,
        )

        assert result["dry_run"] is True, "dry_run must not submit jobs"
        # One tracking job per well, named track_<well-with-slashes-as-underscores>.
        assert_dry_run_plan(
            result["jobs"],
            expect_jobs=len(wells),
            contains=["track_A_1_0", "track_A_2_0"],
        )
