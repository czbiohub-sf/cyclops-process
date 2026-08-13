"""Tests for dispatch_cli command `estimate_stitch_parameters`.

Maps to cyclops_process.processes.ops_stitch.estimate_stitch_parameters.

Tests cover arg validation and the per-process kwarg-propagation branches.
The heavy work (assemble.estimate_stitch) is monkeypatched.
"""

import pytest

from cyclops_process.processes import ops_stitch as mod


EXPERIMENT = "ops9999_20260101"


@pytest.fixture
def hermetic_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_FAST_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_CONFIGS_DIR", str(tmp_path / "configs"))
    (tmp_path / EXPERIMENT / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def stub_assemble(monkeypatch):
    captured = {}

    def fake_estimate_stitch(**kwargs):
        captured.update(kwargs)
        return {}  # mock shifts

    monkeypatch.setattr(mod.assemble, "estimate_stitch", fake_estimate_stitch)
    # Skip the post-estimation confidence check (reads a YAML file).
    monkeypatch.setattr(mod, "check_stitch_confidence", lambda *a, **kw: None)
    return captured


class TestArgValidation:
    def test_no_experiment_no_paths_raises(self):
        with pytest.raises(ValueError, match="input/output directories"):
            mod.estimate_stitch_parameters()

    def test_no_experiment_only_input_raises(self):
        with pytest.raises(ValueError, match="input/output directories"):
            mod.estimate_stitch_parameters(input_store_path="/tmp/foo")


class TestProcessRouting:
    def test_iss_routes_to_iss_drift_corrected(
        self, hermetic_experiment, stub_assemble
    ):
        from cyclops_utils.data.experiment import OpsDataset

        dataset = OpsDataset(EXPERIMENT)
        mod.estimate_stitch_parameters(experiment=EXPERIMENT, process="iss")
        assert stub_assemble["input_store_path"] == dataset.store_paths["iss_drift_corrected"]
        assert stub_assemble["output_config_path"] == dataset.config_paths["iss_stitch"]

    def test_track_routes_to_lc_5x_phase_2d_optimized_and_sets_orientation(
        self, hermetic_experiment, stub_assemble
    ):
        from cyclops_utils.data.experiment import OpsDataset

        dataset = OpsDataset(EXPERIMENT)
        mod.estimate_stitch_parameters(experiment=EXPERIMENT, process="track")
        assert stub_assemble["input_store_path"] == dataset.store_paths["lc_5x_phase_2d_optimized"]
        assert stub_assemble["fliplr"] is True
        assert stub_assemble["rot90"] == 1

    def test_pheno_routes_to_lc_20x_phase_2d_optimized_no_flips(
        self, hermetic_experiment, stub_assemble
    ):
        from cyclops_utils.data.experiment import OpsDataset

        dataset = OpsDataset(EXPERIMENT)
        mod.estimate_stitch_parameters(experiment=EXPERIMENT, process="pheno")
        assert stub_assemble["input_store_path"] == dataset.store_paths["lc_20x_phase_2d_optimized"]
        assert stub_assemble["flipud"] is False
        assert stub_assemble["fliplr"] is False
        assert stub_assemble["rot90"] == 0


class TestDebugSuffix:
    def test_debug_n_positions_appends_suffix_to_config_path(
        self, hermetic_experiment, stub_assemble
    ):
        mod.estimate_stitch_parameters(
            input_store_path="/tmp/in",
            output_config_path="/tmp/stitch.yml",
            debug_n_positions=4,
        )
        # Config path should have "_debug" inserted before the suffix.
        assert stub_assemble["output_config_path"].endswith("_debug.yml")

class TestEstimateStitchParametersRealData:
    """End-to-end real-data test for dispatch stage 'estimate_stitch_parameters'.

    Estimates tile registration shifts; outputs stitch config YAML
    """

    @pytest.mark.real_data
    def test_estimate_stitch_parameters_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.ops_stitch import estimate_stitch_parameters

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_yaml
        from fixtures.slurm import submit_stage

        # iss input is store_paths["iss_drift_corrected"].
        seed_from_cache(
            real_data_workdir,
            reference_cache,
            ['0-convert/in_situ_sequencing/bc_drift_corrected.zarr'],
        )

        log_dir = real_data_workdir / "submitit_logs" / "estimate_stitch_parameters"
        submit_stage(
            "estimate_stitch_parameters",
            estimate_stitch_parameters,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
            ngff_version=reference_ngff_version,
        )

        candidate = real_data_workdir / REFERENCE_EXPERIMENT / "1-preprocess/in_situ_sequencing/stitch/stitch_settings.yml"
        reference = reference_cache / "1-preprocess/in_situ_sequencing/stitch/stitch_settings.yml"
        # Structure (position/well keys, tile-pair graph) and `confidence` floats
        # are deterministic and compared exactly. `total_translation` (cumulative
        # tile positions, 0..25k px) carries a measured <=7px additive jitter from
        # the registration solver's CPU fallback, so allow a small absolute
        # tolerance there only; a real stitching regression is tens-hundreds of px.
        compare_yaml(
            candidate,
            reference,
            rtol=1e-5,
            overrides={"total_translation": {"atol": 10}},
        )



class TestEstimateStitchParametersTrackRealData:
    """End-to-end real-data test for dispatch stage 'estimate_stitch_parameters_track'.

    Estimates stitch for tracking VS data
    """

    @pytest.mark.real_data
    @pytest.mark.skip(
        reason="NO CURRENT-CODE REFERENCE for live_imaging stitch. The version= "
        "prod bug (Fix 0a) is fixed and the stage now runs, but the only "
        "live_imaging reference (kevin/full_pipeline) was produced by older code: "
        "feeding its own tracking_phase_2d_optimized.zarr back through the CURRENT "
        "estimate_stitch yields total_translation differing by >150px (not solver "
        "jitter) -> the stitch algorithm changed since full_pipeline. Un-skip once "
        "a current-code live_imaging reference exists (compare override already set)."
    )
    def test_estimate_stitch_parameters_track_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.ops_stitch import estimate_stitch_parameters

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_yaml
        from fixtures.slurm import submit_stage

        # track input is store_paths["lc_5x_phase_2d_optimized"] (the 2D-optimized
        # reconstruction), NOT the virtual-staining store.
        seed_from_cache(
            real_data_workdir,
            reference_cache,
            ['1-preprocess/live_imaging/reconstruction/tracking_phase_2d_optimized.zarr'],
        )

        log_dir = real_data_workdir / "submitit_logs" / "estimate_stitch_parameters_track"
        submit_stage(
            "estimate_stitch_parameters_track",
            estimate_stitch_parameters,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
            ngff_version=reference_ngff_version,
        )

        candidate = real_data_workdir / REFERENCE_EXPERIMENT / "1-preprocess/live_imaging/stitch/tracking_stitch_settings.yml"
        reference = reference_cache / "1-preprocess/live_imaging/stitch/tracking_stitch_settings.yml"
        # See iss variant: confidence/structure exact, total_translation carries
        # a few-px solver jitter -> small absolute tolerance on that section only.
        compare_yaml(
            candidate,
            reference,
            rtol=1e-5,
            overrides={"total_translation": {"atol": 10}},
        )



class TestEstimateStitchParametersPhenoRealData:
    """End-to-end real-data test for dispatch stage 'estimate_stitch_parameters_pheno'.

    Estimates stitch for pheno
    """

    @pytest.mark.real_data
    @pytest.mark.skip(
        reason="NO CURRENT-CODE REFERENCE for live_imaging stitch. The version= "
        "prod bug (Fix 0a) is fixed and the stage now runs, but the only "
        "live_imaging reference (kevin/full_pipeline) was produced by older code: "
        "feeding its own phenotyping_phase_2d_optimized.zarr back through the "
        "CURRENT estimate_stitch yields total_translation differing by >150px (e.g. "
        "44647 vs 44480), not solver jitter -> the stitch algorithm changed since "
        "full_pipeline. Un-skip once a current-code live_imaging reference exists "
        "(compare override already set)."
    )
    def test_estimate_stitch_parameters_pheno_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.ops_stitch import estimate_stitch_parameters

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_yaml
        from fixtures.slurm import submit_stage

        # pheno input is store_paths["lc_20x_phase_2d_optimized"] (the 2D-optimized
        # reconstruction), NOT the virtual-staining max-projection store.
        seed_from_cache(
            real_data_workdir,
            reference_cache,
            ['1-preprocess/live_imaging/reconstruction/phenotyping_phase_2d_optimized.zarr'],
        )

        log_dir = real_data_workdir / "submitit_logs" / "estimate_stitch_parameters_pheno"
        submit_stage(
            "estimate_stitch_parameters_pheno",
            estimate_stitch_parameters,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
            ngff_version=reference_ngff_version,
        )

        candidate = real_data_workdir / REFERENCE_EXPERIMENT / "1-preprocess/live_imaging/stitch/phenotyping_stitch_settings.yml"
        reference = reference_cache / "1-preprocess/live_imaging/stitch/phenotyping_stitch_settings.yml"
        # See iss variant: confidence/structure exact, total_translation carries
        # a few-px solver jitter -> small absolute tolerance on that section only.
        compare_yaml(
            candidate,
            reference,
            rtol=1e-5,
            overrides={"total_translation": {"atol": 10}},
        )

