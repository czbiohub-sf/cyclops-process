"""Tests for dispatch_cli command `register_iss_round_pair`.

Maps to job_register_round_pair in iss_cycle_register_orchestrator.

The wrapper composes registration params, calls register_round_pair, and
writes metrics to a per-pair JSON file. Tests verify delegation + path.
"""

import json

import pytest


EXPERIMENT = "ops9999_20260101"


@pytest.fixture
def hermetic_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_FAST_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_CONFIGS_DIR", str(tmp_path / "configs"))
    (tmp_path / EXPERIMENT / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


class TestJobRegisterRoundPair:
    def test_delegates_with_round_args_and_writes_metrics(
        self, hermetic_experiment, monkeypatch
    ):
        from cyclops_process.processes.auto_register import iss_cycle_register_orchestrator as mod
        from ops_utils.data.experiment import OpsDataset

        captured = {}

        def fake_helper(iss_zarr, position, round_target, round_source,
                        params, transforms_dir, overlays_dir, verbose=True):
            captured["position"] = position
            captured["round_target"] = round_target
            captured["round_source"] = round_source
            captured["params"] = params
            return {"metrics": {"snr": 12.3}, "yaml": "ok"}

        monkeypatch.setattr(mod, "register_round_pair", fake_helper)

        mod.job_register_round_pair(
            experiment=EXPERIMENT, well=1,
            round_source=4, round_target=3,
            spot_threshold=600, transform_type="similarity",
            verbose=False,
        )

        # Args propagated.
        assert captured["position"] == "A/1/0"
        assert captured["round_source"] == 4
        assert captured["round_target"] == 3
        assert captured["params"]["spot_threshold"] == 600
        assert captured["params"]["transform_type"] == "similarity"

        # Metrics JSON written per-pair.
        ds = OpsDataset(EXPERIMENT)
        metrics_file = ds.preprocess_in_situ / "register/metrics/A1_round4_to_round3_metrics.json"
        assert metrics_file.exists()
        assert json.loads(metrics_file.read_text())["snr"] == 12.3

class TestRegisterIssRoundPairRealData:
    """End-to-end real-data test for dispatch stage 'register_iss_round_pair'.

    Fan-out per well and round pair; registers sequential rounds
    """

    @pytest.mark.real_data
    def test_register_iss_round_pair_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.auto_register.iss_cycle_register_orchestrator import job_register_round_pair

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_yaml
        from fixtures.slurm import submit_stage

        # The stage extracts ISS spots from bc_stitched.zarr (channels 1-4) at the
        # source + target rounds, then RANSAC-registers them (random_state=0).
        # Spot extraction caches per (well, round, threshold) under
        # register/iss_spot_cache; seeding that cache reuses the reference's exact
        # spot set (and skips reading the ~25G/position store), so with the seeded
        # RANSAC the resulting affine is reproducible. bc_stitched.zarr is seeded
        # too as a fallback if a cache key misses.
        seed_from_cache(
            real_data_workdir,
            reference_cache,
            [
                '1-preprocess/in_situ_sequencing/stitch/bc_stitched.zarr',
                '1-preprocess/in_situ_sequencing/register/iss_spot_cache',
            ],
        )

        log_dir = real_data_workdir / "submitit_logs" / "register_iss_round_pair"
        submit_stage(
            "register_iss_round_pair",
            job_register_round_pair,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
            round_source=1,
            round_target=0,
            well=1,
        )

        rel = "1-preprocess/in_situ_sequencing/register/transforms/A1/round1_to_round0.yml"
        candidate = real_data_workdir / REFERENCE_EXPERIMENT / rel
        reference = reference_cache / rel
        # With the spot cache seeded and RANSAC seeded (random_state=0), current
        # code reproduces ops0161's affine to <1e-9 (the registration algorithm is
        # unchanged across the lineage). atol=1e-6 guards float/platform noise while
        # still catching any real regression (sub-pixel and up) or structural change
        # (channels/interpolation, which compare exactly).
        compare_yaml(candidate, reference, atol=1e-6)
