"""Tests for dispatch_cli command `create_max_projection`.

Maps to cyclops_process.processes.assemble.create_max_projection.

Trivial wrapper around _create_max_projection. Tests verify arg propagation.
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


class TestCreateMaxProjection:
    def test_forwards_args_to_inner_helper(
        self, hermetic_experiment, monkeypatch
    ):
        from cyclops_process.processes import assemble as mod

        captured = {}

        def fake_impl(experiment, process, slices, projection):
            captured["experiment"] = experiment
            captured["process"] = process
            captured["slices"] = slices
            captured["projection"] = projection

        monkeypatch.setattr(mod, "_create_max_projection", fake_impl)

        mod.create_max_projection(
            experiment=EXPERIMENT,
            process="pheno-2d",
            slices=[2, 4],
            projection="mean",
        )

        assert captured == {
            "experiment": EXPERIMENT,
            "process": "pheno-2d",
            "slices": [2, 4],
            "projection": "mean",
        }

    def test_defaults_are_propagated(
        self, hermetic_experiment, monkeypatch
    ):
        from cyclops_process.processes import assemble as mod

        captured = {}
        monkeypatch.setattr(
            mod, "_create_max_projection",
            lambda exp, proc, sl, pj: captured.update(
                {"slices": sl, "projection": pj}
            ),
        )
        mod.create_max_projection(experiment=EXPERIMENT, process="track-2d")
        assert captured["slices"] == "all"
        assert captured["projection"] == "max"

class TestCreateMaxProjectionFluorRealData:
    """End-to-end real-data test for dispatch stage 'create_max_projection_fluor'.

    Max projection of fluorescence data
    """

    @pytest.mark.real_data
    @pytest.mark.skip(
        reason="process='lc_20x_fluor' writes the lc_20x_fluor_2d store "
        "(phenotyping_fluor_2d.zarr), which is NOT present in the reference cache. "
        "No reference output to compare against."
    )
    def test_create_max_projection_fluor_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.assemble import create_max_projection

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_ome_zarr
        from fixtures.slurm import submit_stage

        seed_from_cache(
            real_data_workdir,
            reference_cache,
            ['1-preprocess/live_imaging/reconstruction/phenotyping_phase_2d_optimized.zarr'],
        )

        log_dir = real_data_workdir / "submitit_logs" / "create_max_projection_fluor"
        submit_stage(
            "create_max_projection_fluor",
            create_max_projection,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
        )

        candidate = real_data_workdir / REFERENCE_EXPERIMENT / "1-preprocess/live_imaging/virtual_staining/phenotyping_max_proj.zarr"
        reference = reference_cache / "1-preprocess/live_imaging/virtual_staining/phenotyping_max_proj.zarr"
        compare_ome_zarr(candidate, reference, rtol=1e-5)



class TestCreateMaxProjectionLc20xRealData:
    """End-to-end real-data test for dispatch stage 'create_max_projection_lc20x'.

    Max projection of pheno VS output
    """

    @pytest.mark.real_data
    def test_create_max_projection_lc20x_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.assemble import create_max_projection

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_ome_zarr
        from fixtures.slurm import submit_stage

        seed_from_cache(
            real_data_workdir,
            reference_cache,
            ['1-preprocess/live_imaging/virtual_staining/phenotyping_vs.zarr'],
        )

        log_dir = real_data_workdir / "submitit_logs" / "create_max_projection_lc20x"
        submit_stage(
            "create_max_projection_lc20x",
            create_max_projection,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
        )

        candidate = real_data_workdir / REFERENCE_EXPERIMENT / "1-preprocess/live_imaging/virtual_staining/phenotyping_max_proj.zarr"
        reference = reference_cache / "1-preprocess/live_imaging/virtual_staining/phenotyping_max_proj.zarr"
        # Tolerance: the projection's VS input (phenotyping_vs) is produced by a
        # nondeterministic GPU model + resampling, so observed deviations vs the
        # cached reference are small in absolute terms (max ~56 on uint16). atol
        # covers that; tighten once a deterministically-regenerated reference exists.
        compare_ome_zarr(candidate, reference, rtol=1e-3, atol=64)

