import pytest

"""Tests for iss_snr_bimodal."""

import inspect


class TestIssSnrBimodalSignature:
    def test_signature_takes_experiment(self):
        from cyclops_process.metrics.plate_stats import iss_snr_bimodal as mod
        sig = inspect.signature(mod.iss_snr_bimodal)
        assert "experiment" in sig.parameters

class TestIssSnrBimodalRealData:
    """End-to-end real-data test for dispatch stage 'iss_snr_bimodal'.

    Computes SNR bimodal analysis; caches results
    """

    @pytest.mark.real_data
    @pytest.mark.skip(
        reason="No compare target in ops0161. iss_snr_bimodal loads a precomputed "
        "per-tile SNR CSV cache (load_snr_from_csv) or else recomputes SNR from "
        "store_paths['iss'], and writes bimodal CSVs + PNGs -- but the reference "
        "retains NONE of these (no *snr* or *bimodal* files under "
        "1-preprocess/in_situ_sequencing). So there's neither the input cache nor an "
        "output to diff against. Needs a reference run that writes the bimodal outputs "
        "(or a structural/count-tolerant check). (Prior 'symlink-store unreadable path' "
        "reason was the stale generic one; the real blocker is the missing artifacts.)"
    )
    def test_iss_snr_bimodal_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.metrics.plate_stats.iss_snr_bimodal import iss_snr_bimodal

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.slurm import submit_stage

        # (skipped; upstream seeding not exercised)

        log_dir = real_data_workdir / "submitit_logs" / "iss_snr_bimodal"
        submit_stage(
            "iss_snr_bimodal",
            iss_snr_bimodal,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
        )
