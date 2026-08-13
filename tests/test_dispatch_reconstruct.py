"""Tests for dispatch_cli command `reconstruct`.

Maps to cyclops_process.processes.reconstruct.reconstruct.

The function wraps waveorder CLI + per-subtile optimization. Tests cover
the assert-error branches when experiment=None and required paths absent.
"""

import pytest


class TestReconstructAssertions:
    def test_no_experiment_no_input_path_asserts(self):
        from cyclops_process.processes import reconstruct as mod

        with pytest.raises(AssertionError, match="Input path"):
            mod.reconstruct()

    def test_no_experiment_no_config_path_asserts(self):
        from cyclops_process.processes import reconstruct as mod

        with pytest.raises(AssertionError, match="Config path"):
            mod.reconstruct(input_path="/tmp/in")

    def test_no_experiment_no_output_path_asserts(self):
        from cyclops_process.processes import reconstruct as mod

        with pytest.raises(AssertionError, match="Output path"):
            mod.reconstruct(input_path="/tmp/in", config_path="/tmp/cfg.yml")

    def test_no_experiment_no_example_fov_asserts(self):
        from cyclops_process.processes import reconstruct as mod

        with pytest.raises(AssertionError, match="Example FOV"):
            mod.reconstruct(
                input_path="/tmp/in",
                config_path="/tmp/cfg.yml",
                output_path="/tmp/out",
            )

class TestReconstructRealData:
    """End-to-end real-data test for dispatch stage 'reconstruct'.

    Phase reconstruction from brightfield
    """

    @pytest.mark.real_data
    def test_reconstruct_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.reconstruct import reconstruct

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_ome_zarr
        from fixtures.slurm import submit_stage

        seed_from_cache(
            real_data_workdir,
            reference_cache,
            ['1-preprocess/live_imaging/reconstruction/tracking_bf_corrected.zarr'],
        )

        log_dir = real_data_workdir / "submitit_logs" / "reconstruct"
        # stage key is 'reconstruct_track' (the 'track' process variant) in
        # nextflow_ops_args.yaml; that yaml block also supplies process='track'.
        # ngff_version is forced to the v0.4 reference cache (yaml default is 0.5).
        submit_stage(
            "reconstruct_track",
            reconstruct,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
            ngff_version=reference_ngff_version,
        )

        candidate = real_data_workdir / REFERENCE_EXPERIMENT / "1-preprocess/live_imaging/reconstruction/tracking_phase.zarr"
        reference = reference_cache / "1-preprocess/live_imaging/reconstruction/tracking_phase.zarr"
        # Phase reconstruction is GPU/FFT float-nondeterministic and the phase
        # values hover near zero, so relative diffs blow up despite tiny absolute
        # deviations. The pheno sibling stage observed max abs diff ~1e-3 /
        # mean ~4e-5; cover that with an absolute floor and a modest rtol.
        compare_ome_zarr(candidate, reference, rtol=1e-2, atol=2e-3)



class TestReconstructPhenoRealData:
    """End-to-end real-data test for dispatch stage 'reconstruct_pheno'.

    Phase reconstruction for phenotyping
    """

    @pytest.mark.real_data
    def test_reconstruct_pheno_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.reconstruct import reconstruct

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_ome_zarr
        from fixtures.slurm import submit_stage

        seed_from_cache(
            real_data_workdir,
            reference_cache,
            ['0-convert/live_imaging/phenotyping_transform.zarr'],
        )

        log_dir = real_data_workdir / "submitit_logs" / "reconstruct_pheno"
        # 'reconstruct_pheno' yaml block supplies process='pheno'; force the v0.4
        # reference cache ngff_version (yaml default is 0.5).
        submit_stage(
            "reconstruct_pheno",
            reconstruct,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
            ngff_version=reference_ngff_version,
        )

        candidate = real_data_workdir / REFERENCE_EXPERIMENT / "1-preprocess/live_imaging/reconstruction/phenotyping_phase.zarr"
        reference = reference_cache / "1-preprocess/live_imaging/reconstruction/phenotyping_phase.zarr"
        # Phase reconstruction is GPU/FFT float-nondeterministic; observed
        # deviation in this stage was max abs ~0.00103 / mean abs ~4.4e-5 with a
        # huge relative diff only because phase values sit near zero (see
        # validation log A/1/013011). Cover the observed absolute spread with an
        # atol floor plus a modest rtol; this is legitimate float noise, not drift.
        compare_ome_zarr(candidate, reference, rtol=1e-2, atol=2e-3)

