"""Tests for dispatch_cli command `estimate_and_stitch`.

Maps to cyclops_process.processes.ops_stitch.estimate_and_stitch.

The full function does estimation + stitching across many branches. Tests
here cover the arg-validation early-error path; deep coverage is deferred.
"""

import pytest

from cyclops_process.processes import ops_stitch as mod


@pytest.fixture
def hermetic_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_FAST_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_CONFIGS_DIR", str(tmp_path / "configs"))
    (tmp_path / "ops9999_20260101" / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


class TestArgValidation:
    def test_no_experiment_no_paths_raises(self, hermetic_experiment):
        with pytest.raises(ValueError, match="input/output directories"):
            mod.estimate_and_stitch()

    def test_no_experiment_only_input_raises(self, hermetic_experiment):
        with pytest.raises(ValueError, match="input/output directories"):
            mod.estimate_and_stitch(input_store_path="/tmp/foo")

    def test_no_experiment_only_output_raises(self, hermetic_experiment):
        with pytest.raises(ValueError, match="input/output directories"):
            mod.estimate_and_stitch(output_store_path="/tmp/foo")

class TestEstimateAndStitchRealData:
    """End-to-end real-data test for dispatch stage 'estimate_and_stitch'.

    Estimates and applies registration; runs in parallel with segment_and_stitch
    """

    @pytest.mark.real_data
    def test_estimate_and_stitch_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.ops_stitch import estimate_and_stitch

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_ome_zarr
        from fixtures.slurm import submit_stage

        seed_from_cache(
            real_data_workdir,
            reference_cache,
            [
                '0-convert/in_situ_sequencing/bc_drift_corrected.zarr',
                '1-preprocess/in_situ_sequencing/stitch/stitch_settings.yml',
                # The merged estimate_and_stitch(iss) attaches nuclear_seg labels
                # inline (main's retired-symlink flow), reading iss_segmentation.
                '1-preprocess/in_situ_sequencing/segmentation/bc_segmentation.zarr',
            ],
        )

        log_dir = real_data_workdir / "submitit_logs" / "estimate_and_stitch"
        # zarr_version matches the reference's stitched-store format: ops0161 wrote
        # bc_stitched.zarr as v2/v0.4. (Merged estimate_and_stitch uses zarr_version,
        # not the old ngff_version; process must be explicit — defaults to None.)
        submit_stage(
            "estimate_and_stitch",
            estimate_and_stitch,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
            process="iss",
            zarr_version="0.4",
        )

        # estimate_and_stitch(process="iss") writes the stitched store iss_stitch
        # (stitch/bc_stitched.zarr) -- NOT register/bc_stitched_registered.zarr
        # (the manifest's output path was wrong; that store is produced by a later
        # registration stage). Compare the store this stage actually produces.
        candidate = real_data_workdir / REFERENCE_EXPERIMENT / "1-preprocess/in_situ_sequencing/stitch/bc_stitched.zarr"
        reference = reference_cache / "1-preprocess/in_situ_sequencing/stitch/bc_stitched.zarr"
        # Structural only: each position of a stitched store is a full-well canvas
        # (~26k x 26k), so loading whole arrays to np.allclose would OOM/hang the
        # login node. Structural checks (position list, canvas shape, dtype, scale)
        # still catch stitching errors; this matches segment_and_stitch's approach.
        compare_ome_zarr(candidate, reference, check_data=False)



class TestEstimateAndStitchTrackRealData:
    """End-to-end real-data test for dispatch stage 'estimate_and_stitch_track'.

    Stitches tracking VS data
    """

    @pytest.mark.real_data
    def test_estimate_and_stitch_track_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.ops_stitch import estimate_and_stitch

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_ome_zarr
        from fixtures.slurm import submit_stage

        seed_from_cache(
            real_data_workdir,
            reference_cache,
            [
                # estimate_and_stitch(track) reads the reconstructed tiles store,
                # not tracking_vs.zarr (the manifest upstream was wrong).
                '1-preprocess/live_imaging/reconstruction/tracking_phase_2d_optimized.zarr',
                '1-preprocess/live_imaging/stitch/tracking_stitch_settings.yml',
                # Merged track-2d branch materializes nuclear_seg into the v3 store
                # (main's retired-symlink flow), reading lc_5x_segmentation.
                '1-preprocess/live_imaging/segmentation/tracking_segmentation_stitched.zarr',
            ],
        )

        log_dir = real_data_workdir / "submitit_logs" / "estimate_and_stitch_track"
        # Output is a v3 store (tracking_phase_2d_stitched_v3.zarr) -> zarr_version=0.5;
        # process="track-2d"; merged signature uses zarr_version, not ngff_version.
        submit_stage(
            "estimate_and_stitch_track",
            estimate_and_stitch,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
            process="track-2d",
            zarr_version="0.5",
        )

        candidate = real_data_workdir / REFERENCE_EXPERIMENT / "1-preprocess/live_imaging/stitch/tracking_phase_2d_stitched_v3.zarr"
        reference = reference_cache / "1-preprocess/live_imaging/stitch/tracking_phase_2d_stitched_v3.zarr"
        # Structural only (divergent reference lineage + full-canvas data compare would
        # risk OOM on the test host); validates positions/shape/dtype/scale + that the
        # v3 store was produced. Strengthen to check_data on a current-code reference.
        compare_ome_zarr(candidate, reference, check_data=False)



class TestEstimateAndStitchPhenoRealData:
    """End-to-end real-data test for dispatch stage 'estimate_and_stitch_pheno'.

    Final stitching of phenotyping data
    """

    @pytest.mark.real_data
    def test_estimate_and_stitch_pheno_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.ops_stitch import estimate_and_stitch

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_ome_zarr
        from fixtures.slurm import submit_stage

        seed_from_cache(
            real_data_workdir,
            reference_cache,
            ['1-preprocess/live_imaging/stitch/phenotyping_tiles_unified.zarr'],
        )

        log_dir = real_data_workdir / "submitit_logs" / "estimate_and_stitch_pheno"
        submit_stage(
            "estimate_and_stitch_pheno",
            estimate_and_stitch,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
            ngff_version=reference_ngff_version,
        )

        candidate = real_data_workdir / REFERENCE_EXPERIMENT / "3-assembly/phenotyping_v3.zarr"
        reference = reference_cache / "3-assembly/phenotyping_v3.zarr"
        compare_ome_zarr(candidate, reference, rtol=1e-5)

