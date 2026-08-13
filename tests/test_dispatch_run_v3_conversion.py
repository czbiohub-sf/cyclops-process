import pytest

"""Tests for run_v3_conversion."""

import inspect


class TestRunV3ConversionSignature:
    def test_signature_takes_experiment(self):
        from cyclops_process.convert import v3_livecell as mod
        sig = inspect.signature(mod.run_v3_conversion)
        assert "experiment" in sig.parameters

class TestRunV3ConversionRealData:
    """End-to-end real-data test for dispatch stage 'run_v3_conversion'.

    Orchestrator: v3 format conversion; job submission
    """

    @pytest.mark.real_data
    def test_run_v3_conversion_matches_reference(
        self, real_data_workdir, reference_cache, capsys
    ):
        """Dry-run orchestrator. mode='iss' reads the ISS registered store
        (iss_stitch_registered), enumerates its position+group combinations, and
        plans one ``convert_<pos>_base`` job + one ``convert_<pos>_nuclear_seg``
        job per position. dry_run=True runs in-process: it prints the plan and
        submits nothing (returns ``{'submitted': 0, 'failed': 0}``), so we assert
        on captured stdout. ISS reference has positions A/1/0, A/2/0, A/3/0."""
        from cyclops_process.convert.v3_livecell import run_v3_conversion

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.stdout import assert_dry_run_plan

        seed_from_cache(
            real_data_workdir,
            reference_cache,
            ["1-preprocess/in_situ_sequencing/register/bc_stitched_registered.zarr"],
        )

        result = run_v3_conversion(REFERENCE_EXPERIMENT, mode="iss", dry_run=True)
        out = capsys.readouterr().out

        # dry-run submits nothing through either the base or seg job batch.
        assert result == {"submitted": 0, "failed": 0}
        assert_dry_run_plan(
            out,
            contains=[
                "convert_A_1_0_base",
                "convert_A_1_0_nuclear_seg",
                "DRY RUN: No jobs submitted",
            ],
        )
