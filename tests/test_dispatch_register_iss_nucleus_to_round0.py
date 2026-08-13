"""Tests for dispatch_cli command `register_iss_nucleus_to_round0`.

Maps to job_register_nucleus_to_round0 in iss_cycle_register_orchestrator.

The wrapper composes registration params, calls the heavy
register_nucleus_to_round0, and writes metrics to JSON. Tests verify
delegation contract + metrics JSON path.
"""

import json
from pathlib import Path

import pytest


EXPERIMENT = "ops9999_20260101"


@pytest.fixture
def hermetic_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_FAST_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_CONFIGS_DIR", str(tmp_path / "configs"))
    (tmp_path / EXPERIMENT / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


class TestJobRegisterNucleusToRound0:
    def test_delegates_and_writes_metrics_json(
        self, hermetic_experiment, monkeypatch
    ):
        from cyclops_process.processes.auto_register import iss_cycle_register_orchestrator as mod
        from cyclops_utils.data.experiment import OpsDataset

        captured = {}

        def fake_helper(iss_zarr, seg_zarr, position, params,
                        transforms_dir, overlays_dir, dataset=None, verbose=True,
                        spots_round=0):
            captured["params"] = params
            captured["position"] = position
            captured["spots_round"] = spots_round
            return {"metrics": {"method": "pcc", "score": 0.9}, "yaml": "ok"}

        monkeypatch.setattr(mod, "register_nucleus_to_round0", fake_helper)

        mod.job_register_nucleus_to_round0(
            experiment=EXPERIMENT, well=3,
            spot_threshold=500, nucleus_threshold=250,
            transform_type="affine", max_distance=150,
            verbose=False,
        )

        assert captured["position"] == "A/3/0"
        assert captured["params"]["spot_threshold"] == 500
        assert captured["params"]["nucleus_threshold"] == 250
        assert captured["params"]["transform_type"] == "affine"
        assert captured["params"]["max_distance"] == 150
        assert captured["spots_round"] == 0  # wrapper forwards the anchor-round default

        ds = OpsDataset(EXPERIMENT)
        metrics_file = ds.preprocess_in_situ / "register/metrics/A3_nucleus_to_round0_metrics.json"
        assert metrics_file.exists()
        data = json.loads(metrics_file.read_text())
        assert data["method"] == "pcc"
        assert data["score"] == 0.9

    def test_skips_metrics_write_when_no_metrics_in_result(
        self, hermetic_experiment, monkeypatch
    ):
        from cyclops_process.processes.auto_register import iss_cycle_register_orchestrator as mod
        from cyclops_utils.data.experiment import OpsDataset

        monkeypatch.setattr(
            mod, "register_nucleus_to_round0",
            lambda *a, **kw: {"yaml": "ok"},  # no "metrics" key
        )

        mod.job_register_nucleus_to_round0(
            experiment=EXPERIMENT, well=1, verbose=False,
        )

        ds = OpsDataset(EXPERIMENT)
        metrics_file = ds.preprocess_in_situ / "register/metrics/A1_nucleus_to_round0_metrics.json"
        assert not metrics_file.exists()

class TestRegisterIssNucleusToRound0RealData:
    """End-to-end real-data test for dispatch stage 'register_iss_nucleus_to_round0'.

    Registers nucleus to round 0; fan-out per well
    """

    @pytest.mark.real_data
    @pytest.mark.skip(
        reason="Stage RUNS now (the raw-NDTiff unlock works: OpsDataset resolves "
        "iss_tif_dir to $OPS_INSTRUMENT_ROOT/ops0161_20260521 via the experiment "
        "config, and the DAPI-to-DAPI PCC reads it), but the result is NON-REPRODUCIBLE "
        "so an exact compare vs the single-sample reference is infeasible. The nucleus->"
        "round0 affine wobbles ~1px run-to-run: the Round-0 ch1-4 spot extraction "
        "recomputes (cache miss for this stage's key -- ~8s, 'Found 5000 spots') and "
        "subsamples a RANDOM subset (unseeded, same class as detect_spots), so the "
        "RANSAC matched-pair set -- and the affine -- changes each run. Observed: "
        "[1][1] scale 0.99998 vs 0.99994 across two runs; vs reference translation "
        "differs ~0.7-1.1px (dy/dx), scale ~6e-5. Unlike round_pair/seg_to_nucleus "
        "(deterministic: cached spots / pure PCC), this path's spot subsampling isn't "
        "seeded. To un-skip: a reproducible-mode reference (seeded subsampling), or a "
        "deliberately COARSE translation-tolerance compare (+/- a few px) if a weak "
        "regression guard is wanted. Body below runs end-to-end; seeds are correct."
    )
    def test_register_iss_nucleus_to_round0_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.auto_register.iss_cycle_register_orchestrator import job_register_nucleus_to_round0

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_yaml
        from fixtures.slurm import submit_stage

        # Seedable inputs (the cluster pattern): stitched ISS + segmentation stores
        # (symlink, read-through) and the spot/centroid caches so extraction reuses
        # the reference's exact points. The DAPI-to-DAPI PCC also reads the original
        # raw NDTiff at dataset.iss_tif_dir -- not seeded, but OpsDataset resolves it
        # from the experiment config (experiment_configs/ops0161_20260521_config.yaml
        # sets iss_tif_dir -> $OPS_INSTRUMENT_ROOT/ops0161_20260521, which
        # exists; the default bioe.ops.iss path does not).
        seed_from_cache(
            real_data_workdir,
            reference_cache,
            [
                '1-preprocess/in_situ_sequencing/stitch/bc_stitched.zarr',
                '1-preprocess/in_situ_sequencing/segmentation/bc_segmentation.zarr',
                '1-preprocess/in_situ_sequencing/register/iss_spot_cache',
                '1-preprocess/in_situ_sequencing/register/centroid_cache',
            ],
        )

        log_dir = real_data_workdir / "submitit_logs" / "register_iss_nucleus_to_round0"
        submit_stage(
            "register_iss_nucleus_to_round0",
            job_register_nucleus_to_round0,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
            well=1,
        )

        rel = "1-preprocess/in_situ_sequencing/register/transforms/A1/nucleus_to_round0.yml"
        candidate = real_data_workdir / REFERENCE_EXPERIMENT / rel
        reference = reference_cache / rel
        compare_yaml(candidate, reference, atol=1e-6)
