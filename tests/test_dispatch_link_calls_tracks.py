import pytest

"""Tests for link_calls_tracks."""

import inspect


class TestLinkCallsTracksSignature:
    def test_signature_exposes_expected_kwargs(self):
        from cyclops_process.data import datasets as mod
        sig = inspect.signature(mod.link_calls_tracks)
        for name in (
            "experiment", "wells", "method", "confidence_threshold",
            "iss_rounds", "failed_rounds_by_well", "n_jobs", "skip_track",
        ):
            assert name in sig.parameters, f"missing kwarg: {name}"

    def test_default_wells(self):
        from cyclops_process.data import datasets as mod
        sig = inspect.signature(mod.link_calls_tracks)
        assert sig.parameters["wells"].default == ["A/1/0", "A/2/0", "A/3/0"]

    def test_default_method(self):
        from cyclops_process.data import datasets as mod
        sig = inspect.signature(mod.link_calls_tracks)
        assert sig.parameters["method"].default == "mine"

    def test_default_confidence_threshold(self):
        from cyclops_process.data import datasets as mod
        sig = inspect.signature(mod.link_calls_tracks)
        assert sig.parameters["confidence_threshold"].default == 0.95

class TestLinkCallsTracksRealData:
    """End-to-end real-data test for dispatch stage 'link_calls_tracks'.

    Links ISS calls to tracking measurements
    """

    @pytest.mark.real_data
    def test_link_calls_tracks_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.data.datasets import link_calls_tracks

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_csv
        from fixtures.slurm import submit_stage

        seed_from_cache(
            real_data_workdir,
            reference_cache,
            [
                # cache stores reads per-well, not under a generic name
                '1-preprocess/in_situ_sequencing/base_calling/mine/A*_reads.csv',
                # link_calls_tracks reads tracks.geff AND the per-well
                # auto_register.yml sidecars; seed the whole 2-tracking dir.
                '2-tracking',
            ],
        )

        log_dir = real_data_workdir / "submitit_logs" / "link_calls_tracks"
        submit_stage(
            "link_calls_tracks",
            link_calls_tracks,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
        )

        # link_calls_tracks writes one linked CSV per well; compare a representative well.
        candidate = real_data_workdir / REFERENCE_EXPERIMENT / "3-assembly/A1_linked_pheno_iss.csv"
        reference = reference_cache / "3-assembly/A1_linked_pheno_iss.csv"
        compare_csv(candidate, reference, rtol=1e-5)

