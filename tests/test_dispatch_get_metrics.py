import pytest

"""Tests for get_metrics.

The function orchestrates many sub-metric calls and requires a fully
populated dataset directory tree. Deep coverage is deferred; here we test
the signature contract.
"""

import inspect


class TestGetMetricsSignature:
    def test_signature_exposes_expected_kwargs(self):
        from cyclops_process.metrics import metrics as mod
        sig = inspect.signature(mod.get_metrics)
        for name in (
            "experiment", "method", "confidence_threshold",
            "iss_rounds", "failed_rounds_by_well", "force", "n_rounds",
        ):
            assert name in sig.parameters, f"missing kwarg: {name}"

    def test_default_method_is_mine(self):
        from cyclops_process.metrics import metrics as mod
        sig = inspect.signature(mod.get_metrics)
        assert sig.parameters["method"].default == "mine"

    def test_default_n_rounds_is_9(self):
        from cyclops_process.metrics import metrics as mod
        sig = inspect.signature(mod.get_metrics)
        assert sig.parameters["n_rounds"].default == 9

    def test_default_force_is_false(self):
        from cyclops_process.metrics import metrics as mod
        sig = inspect.signature(mod.get_metrics)
        assert sig.parameters["force"].default is False

class TestGetMetricsRealData:
    """End-to-end real-data test for dispatch stage 'get_metrics'.

    Generates comprehensive ISS metrics; outputs CSV + PNGs
    """

    @pytest.mark.real_data
    def test_get_metrics_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.metrics.metrics import get_metrics

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_metrics_table
        from fixtures.slurm import submit_stage

        seed_from_cache(
            real_data_workdir,
            reference_cache,
            [
                # cache stores reads per-well (A1_reads.csv, ...), not a single reads.csv
                '1-preprocess/in_situ_sequencing/base_calling/mine/A*_reads.csv',
                '3-assembly/ISS/mine/snr_stats_cache_mine.csv',
                # get_metrics also reads the ISS segmentation store
                '1-preprocess/in_situ_sequencing/segmentation/bc_segmentation.zarr',
            ],
        )

        log_dir = real_data_workdir / "submitit_logs" / "get_metrics"
        submit_stage(
            "get_metrics",
            get_metrics,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
        )

        candidate = real_data_workdir / REFERENCE_EXPERIMENT / "3-assembly/ISS/mine/plate_stats.csv"
        reference = reference_cache / "3-assembly/ISS/mine/plate_stats.csv"
        # plate_stats is a metrics table (rows=metric, cols=wells) whose metric
        # SET evolves between pipeline versions. Compare shared metrics within
        # +/-5%; warn (don't fail) on metrics present in only one side.
        compare_metrics_table(candidate, reference, rtol=0.05)

