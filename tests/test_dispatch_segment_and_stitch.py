import pytest

"""Tests for segment_and_stitch."""

import inspect


class TestSegmentAndStitchSignature:
    def test_signature_takes_experiment(self):
        from cyclops_process.processes import segment as mod
        sig = inspect.signature(mod.segment_and_stitch)
        assert "experiment" in sig.parameters

class TestSegmentAndStitchRealData:
    """End-to-end real-data test for dispatch stage 'segment_and_stitch'.

    Segments nuclei and stitches using shift estimates
    """

    @pytest.mark.real_data
    def test_segment_and_stitch_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.segment import segment_and_stitch

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_ome_zarr
        from fixtures.slurm import submit_stage

        seed_from_cache(
            real_data_workdir,
            reference_cache,
            ['0-convert/in_situ_sequencing/bc_drift_corrected.zarr', '1-preprocess/in_situ_sequencing/stitch/stitch_settings.yml'],
        )

        log_dir = real_data_workdir / "submitit_logs" / "segment_and_stitch"
        submit_stage(
            "segment_and_stitch",
            segment_and_stitch,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
        )

        candidate = real_data_workdir / REFERENCE_EXPERIMENT / "1-preprocess/in_situ_sequencing/segmentation/bc_segmentation.zarr"
        reference = reference_cache / "1-preprocess/in_situ_sequencing/segmentation/bc_segmentation.zarr"
        # bc_segmentation.zarr is an int32 instance label map. Cellpose assigns
        # label ids in arbitrary order, so the integer values never match the
        # reference pixel-for-pixel (observed: max abs diff ~880k, ~21% of
        # pixels). Compare structure only (positions/channels/shape/dtype/scale);
        # label values are intentionally not compared.
        compare_ome_zarr(candidate, reference, check_data=False)



class TestSegmentAndStitchTrackRealData:
    """End-to-end real-data test for dispatch stage 'segment_and_stitch_track'.

    Segments tracking data; parallel with estimate_and_stitch_track
    """

    @pytest.mark.real_data
    def test_segment_and_stitch_track_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.segment import segment_and_stitch

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_ome_zarr
        from fixtures.slurm import submit_stage

        seed_from_cache(
            real_data_workdir,
            reference_cache,
            ['1-preprocess/live_imaging/virtual_staining/tracking_vs.zarr', '1-preprocess/live_imaging/stitch/tracking_stitch_settings.yml'],
        )

        log_dir = real_data_workdir / "submitit_logs" / "segment_and_stitch_track"
        submit_stage(
            "segment_and_stitch_track",
            segment_and_stitch,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
        )

        candidate = real_data_workdir / REFERENCE_EXPERIMENT / "1-preprocess/live_imaging/segmentation/tracking_segmentation_stitched.zarr"
        reference = reference_cache / "1-preprocess/live_imaging/segmentation/tracking_segmentation_stitched.zarr"
        # int32 instance label map (see base test). Label ids are assigned in
        # arbitrary order and do not match the reference pixel-for-pixel.
        # Compare structure only; label values are intentionally not compared.
        compare_ome_zarr(candidate, reference, check_data=False)



class TestSegmentAndStitchPhenoRealData:
    """End-to-end real-data test for dispatch stage 'segment_and_stitch_pheno'.

    Segments pheno data and stitches
    """

    @pytest.mark.real_data
    def test_segment_and_stitch_pheno_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.segment import segment_and_stitch

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_ome_zarr
        from fixtures.slurm import submit_stage

        seed_from_cache(
            real_data_workdir,
            reference_cache,
            ['1-preprocess/live_imaging/virtual_staining/phenotyping_max_proj.zarr', '1-preprocess/live_imaging/stitch/phenotyping_stitch_settings.yml'],
        )

        log_dir = real_data_workdir / "submitit_logs" / "segment_and_stitch_pheno"
        submit_stage(
            "segment_and_stitch_pheno",
            segment_and_stitch,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
        )

        candidate = real_data_workdir / REFERENCE_EXPERIMENT / "1-preprocess/live_imaging/segmentation/phenotyping_segmentation_stitched.zarr"
        reference = reference_cache / "1-preprocess/live_imaging/segmentation/phenotyping_segmentation_stitched.zarr"
        # int32 instance label map (see base test). Label ids are assigned in
        # arbitrary order and do not match the reference pixel-for-pixel.
        # Compare structure only; label values are intentionally not compared.
        compare_ome_zarr(candidate, reference, check_data=False)

