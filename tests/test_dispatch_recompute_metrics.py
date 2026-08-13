"""Tests for dispatch_cli command `recompute_metrics`.

Maps to cyclops_process.metrics.metrics.recompute_metrics.

This is a trivial wrapper around get_metrics(..., force=True). Tests verify
arg propagation.
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


class TestRecomputeMetrics:
    def test_forwards_kwargs_with_force_true(self, hermetic_experiment, monkeypatch):
        from cyclops_process.metrics import metrics as mod

        captured = {}

        def fake_get_metrics(experiment, **kwargs):
            captured["experiment"] = experiment
            captured.update(kwargs)

        monkeypatch.setattr(mod, "get_metrics", fake_get_metrics)

        mod.recompute_metrics(
            experiment=EXPERIMENT,
            method="probabilistic",
            confidence_threshold=0.5,
            iss_rounds=[0, 1, 2],
            failed_rounds_by_well={"A/1/0": [3]},
        )

        assert captured["experiment"] == EXPERIMENT
        assert captured["method"] == "probabilistic"
        assert captured["confidence_threshold"] == 0.5
        assert captured["iss_rounds"] == [0, 1, 2]
        assert captured["failed_rounds_by_well"] == {"A/1/0": [3]}
        assert captured["force"] is True

class TestRecomputeMetricsRealData:
    """End-to-end real-data test for dispatch stage 'recompute_metrics'.

    Recomputes metrics with linked data; final summary
    """

    # recompute_metrics -> get_metrics(..., force=True) is a whole-experiment QC
    # *aggregator*, not a single pipeline stage. The reference plate_stats.csv
    # (3-assembly/ISS/mine/plate_stats.csv) has 172 metric rows assembled from
    # iss_stats.statistics(), which reads inputs spanning every processing phase:
    #   - base calling: per-well reads CSVs, detected-points .npy, frequency tables
    #   - segmentation/register: bc_segmentation.zarr, bc_stitched_registered.zarr
    #   - SNR + growth-effect + crosstalk caches (main_for_metrics)
    #   - stitch-confidence (iss/track/pheno), auto-register, z-offset/reconstruction
    #   - ISS cycle-drift, link stats, tracking-graph stats (.geff), spatial coherence,
    #     cell-segmentation shape stats, flatfield-correction stats
    # Each optional block is guarded by try/except, so a missing upstream silently
    # drops its rows -- and compare_csv requires the FULL set of 172 metric rows to
    # match exactly (no column/row-subset mode). Reproducing the reference therefore
    # requires seeding essentially the entire experiment output tree (an in-process
    # full-pipeline rebuild), which is outside the seed-a-few-inputs / run-one-stage
    # harness model. The single-CSV seed the generator emitted
    # ('3-assembly/linked_pheno_iss.csv', which does not even exist in the cache --
    # it is per-well A1_linked_pheno_iss.csv) covers a tiny fraction of the inputs.
    @pytest.mark.skip(
        reason="needs in-process upstream rebuild: get_metrics aggregates 172 "
        "plate_stats metrics from base-calling, segmentation, register, SNR, "
        "tracking-graph, link, drift, reconstruction, spatial-coherence, "
        "cell-seg and flatfield outputs across all phases; not a single stage"
    )
    @pytest.mark.real_data
    def test_recompute_metrics_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.metrics.metrics import recompute_metrics

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_csv
        from fixtures.slurm import submit_stage

        # Inputs read by get_metrics/statistics for plate_stats.csv. NOTE: this is
        # only the ISS base-calling subset; the full set spans many more phases
        # (see the skip reason above), which is why this test is skipped.
        seed_from_cache(
            real_data_workdir,
            reference_cache,
            [
                '1-preprocess/in_situ_sequencing/segmentation/bc_segmentation.zarr',
                '1-preprocess/in_situ_sequencing/register/bc_stitched_registered.zarr',
                '1-preprocess/in_situ_sequencing/base_calling/mine/A1_reads.csv',
                '1-preprocess/in_situ_sequencing/base_calling/mine/A2_reads.csv',
                '1-preprocess/in_situ_sequencing/base_calling/mine/A3_reads.csv',
                '1-preprocess/in_situ_sequencing/base_calling/A1_detected_points.npy',
                '1-preprocess/in_situ_sequencing/base_calling/A2_detected_points.npy',
                '1-preprocess/in_situ_sequencing/base_calling/A3_detected_points.npy',
            ],
        )

        log_dir = real_data_workdir / "submitit_logs" / "recompute_metrics"
        submit_stage(
            "recompute_metrics",
            recompute_metrics,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
        )

        candidate = real_data_workdir / REFERENCE_EXPERIMENT / "3-assembly/ISS/mine/plate_stats.csv"
        reference = reference_cache / "3-assembly/ISS/mine/plate_stats.csv"
        compare_csv(candidate, reference, rtol=1e-5)

