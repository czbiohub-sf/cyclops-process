"""Tests for virtual_staining_combine_validate."""

import pytest

EXPERIMENT = "ops9999_20260101"


@pytest.fixture
def hermetic_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_FAST_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_CONFIGS_DIR", str(tmp_path / "configs"))
    (tmp_path / EXPERIMENT / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


class TestVirtualStainingCombineValidate:
    def test_invalid_dim_raises(self, hermetic_experiment):
        from cyclops_process.processes import virtual_staining as mod
        with pytest.raises(ValueError, match="dim must be one of"):
            mod.virtual_staining_combine_validate(
                experiment=EXPERIMENT, process="track", dim="4d",
            )

    def test_unknown_process_raises(self, hermetic_experiment):
        from cyclops_process.processes import virtual_staining as mod
        with pytest.raises(ValueError, match="process must be"):
            mod.virtual_staining_combine_validate(
                experiment=EXPERIMENT, process="bogus", dim="3d",
            )

    def test_track_dispatches_to_validate_combine_output(
        self, hermetic_experiment, monkeypatch
    ):
        from cyclops_process.processes import virtual_staining as mod
        from cyclops_utils.data.experiment import OpsDataset

        captured = {}
        monkeypatch.setattr(
            mod, "validate_combine_output",
            lambda output_store, n_samples: captured.update(
                {"store": output_store, "n_samples": n_samples}
            ),
        )
        mod.virtual_staining_combine_validate(
            experiment=EXPERIMENT, process="track", dim="3d", n_samples=5,
        )
        ds = OpsDataset(EXPERIMENT)
        assert captured["store"] == ds.store_paths["lc_5x_vs"]
        assert captured["n_samples"] == 5

class TestVirtualStainingCombineValidateRealData:
    """End-to-end real-data test for dispatch stage 'virtual_staining_combine_validate'.

    Validates combined VS output
    """

    @pytest.mark.real_data
    def test_virtual_staining_combine_validate_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.virtual_staining import virtual_staining_combine_validate

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_ome_zarr
        from fixtures.slurm import submit_stage

        seed_from_cache(
            real_data_workdir,
            reference_cache,
            ['1-preprocess/live_imaging/virtual_staining/tracking_vs.zarr'],
        )

        log_dir = real_data_workdir / "submitit_logs" / "vs_combine_validate_track"
        # The real yaml key for the tracking (lc_5x) validate stage is
        # 'vs_combine_validate_track'; its python_kwargs supply process=track, dim=2d.
        submit_stage(
            "vs_combine_validate_track",
            virtual_staining_combine_validate,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
        )

        candidate = real_data_workdir / REFERENCE_EXPERIMENT / "1-preprocess/live_imaging/virtual_staining/tracking_vs.zarr"
        reference = reference_cache / "1-preprocess/live_imaging/virtual_staining/tracking_vs.zarr"
        compare_ome_zarr(candidate, reference, rtol=1e-5)



class TestVsCombineValidatePhenoRealData:
    """End-to-end real-data test for dispatch stage 'vs_combine_validate_pheno'.

    Validates pheno VS
    """

    @pytest.mark.real_data
    def test_vs_combine_validate_pheno_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.virtual_staining import virtual_staining_combine_validate

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_ome_zarr
        from fixtures.slurm import submit_stage

        seed_from_cache(
            real_data_workdir,
            reference_cache,
            ['1-preprocess/live_imaging/virtual_staining/phenotyping_vs.zarr'],
        )

        log_dir = real_data_workdir / "submitit_logs" / "vs_combine_validate_pheno"
        submit_stage(
            "vs_combine_validate_pheno",
            virtual_staining_combine_validate,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
        )

        candidate = real_data_workdir / REFERENCE_EXPERIMENT / "1-preprocess/live_imaging/virtual_staining/phenotyping_vs.zarr"
        reference = reference_cache / "1-preprocess/live_imaging/virtual_staining/phenotyping_vs.zarr"
        compare_ome_zarr(candidate, reference, rtol=1e-5)

