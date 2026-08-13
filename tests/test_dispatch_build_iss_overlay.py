import pytest

"""Tests for build_iss_overlay."""

import inspect


class TestBuildIssOverlaySignature:
    def test_signature_takes_experiment(self):
        from cyclops_process.processes.pyramids import audit_fix as mod
        sig = inspect.signature(mod.build_iss_overlay)
        assert "experiment" in sig.parameters

class TestBuildIssOverlayRealData:
    """End-to-end real-data test for dispatch stage 'build_iss_overlay'.

    Builds ISS visualization pyramids
    """

    @pytest.mark.real_data
    @pytest.mark.skip(
        reason="no reference output to compare: 3-assembly/ISS/mine in the cache "
        "holds only CSVs/PNGs, no *.zarr overlay store. Also requires per-well "
        "reads.csv + linked_pheno_iss.csv inputs. Needs a regenerated reference that "
        "retains the overlay .zarr."
    )
    def test_build_iss_overlay_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.pyramids.audit_fix import build_iss_overlay

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_ome_zarr_directory
        from fixtures.slurm import submit_stage

        seed_from_cache(
            real_data_workdir,
            reference_cache,
            ['1-preprocess/in_situ_sequencing/base_calling/mine/reads.csv', '3-assembly/linked_pheno_iss.csv'],
        )

        log_dir = real_data_workdir / "submitit_logs" / "build_iss_overlay"
        submit_stage(
            "build_iss_overlay",
            build_iss_overlay,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
        )

        candidate = real_data_workdir / REFERENCE_EXPERIMENT / "3-assembly/ISS/mine"
        reference = reference_cache / "3-assembly/ISS/mine"
        compare_ome_zarr_directory(
            candidate, reference, pattern="*.zarr", rtol=1e-5
        )

