import pytest

"""Tests for submit_cell_segmentation_jobs."""

import inspect


class TestSubmitCellSegmentationJobsSignature:
    def test_signature_takes_experiment(self):
        from cyclops_process.processes.cell_seg import cell_segmentation_orchestrator as mod
        sig = inspect.signature(mod.submit_cell_segmentation_jobs)
        assert "experiment" in sig.parameters

class TestSubmitCellSegmentationJobsRealData:
    """End-to-end real-data test for dispatch stage 'submit_cell_segmentation_jobs'.

    Orchestrator: submits cell segmentation
    """

    @pytest.mark.real_data
    def test_submit_cell_segmentation_jobs_matches_reference(
        self, real_data_workdir, reference_cache
    ):
        """Dry-run plan assertion. The dry-run path is NOT self-contained:
        get_available_positions() opens phenotyping_v3.zarr (pheno_assembled_v3) and QCs
        its nuclei_prediction/membrane_prediction channels before planning. ops0161 has
        that v3 store with both channels (3 assembled positions A/{1,2,3}/0), so seed it
        and assert the dry-run plans successfully without submitting."""
        from cyclops_process.processes.cell_seg.cell_segmentation_orchestrator import (
            submit_cell_segmentation_jobs,
        )

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache

        seed_from_cache(
            real_data_workdir,
            reference_cache,
            ["3-assembly/phenotyping_v3.zarr"],
        )

        # ops0161 is a complete run, so its phenotyping_v3.zarr already has cell_seg —
        # force=True ignores the existing labels and plans all positions (skip_existing
        # off), so the QC + plan path is exercised instead of an empty "already done" plan.
        result = submit_cell_segmentation_jobs(
            REFERENCE_EXPERIMENT,
            dry_run=True,
            wait_for_completion=False,
            force=True,
            verbose=False,
        )

        assert result["dry_run"] is True, "dry_run must not submit jobs"
        # The inner submit_parallel_jobs dry-run returns {"dry_run": True, "jobs": [...]}.
        plan = result["result"]
        assert plan.get("dry_run") is True, f"unexpected dry-run result: {result}"
        # One cell-seg job per assembled position (A/1/0, A/2/0, A/3/0).
        assert len(plan["jobs"]) == 3, f"expected 3 cell-seg jobs, got {plan['jobs']}"

