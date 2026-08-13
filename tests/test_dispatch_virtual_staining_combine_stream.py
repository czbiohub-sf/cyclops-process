"""Tests for virtual_staining_combine_stream."""

import pytest

EXPERIMENT = "ops9999_20260101"


@pytest.fixture
def hermetic_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_FAST_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_CONFIGS_DIR", str(tmp_path / "configs"))
    (tmp_path / EXPERIMENT / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


class TestVirtualStainingCombineStream:
    def test_invalid_dim_raises(self, hermetic_experiment):
        from cyclops_process.processes import virtual_staining as mod
        with pytest.raises(ValueError, match="dim must be one of"):
            mod.virtual_staining_combine_stream(
                experiment=EXPERIMENT, process="track", dim="4d",
            )

    def test_unknown_process_raises(self, hermetic_experiment):
        from cyclops_process.processes import virtual_staining as mod
        with pytest.raises(ValueError, match="process must be"):
            mod.virtual_staining_combine_stream(
                experiment=EXPERIMENT, process="bogus", dim="3d",
            )

class TestVirtualStainingCombineStreamRealData:
    """End-to-end real-data test for dispatch stage 'virtual_staining_combine_stream'.

    Fan-out per stream; combines inference into zarr
    """

    @pytest.mark.real_data
    @pytest.mark.skip(
        reason="ops0161's retained VS intermediate (tracking_vs/) is HCS format "
        "(gpu0.zarr), but vs_combine_stream/combine_single_batch_store processes only "
        "BATCH-format shards (predictions_{start}_{end}.zarr; it filters format!='hcs'), "
        "which ops0161 does not retain -> 0 batch stores. Not a seed gap, an input-format "
        "mismatch: this stage's path has no valid input in ops0161. Body below is "
        "otherwise re-targeted (process/store_index/dim, vs_combine_stream_track yaml key, "
        "seed reconstruction store, position-subset compare) for when batch shards exist."
    )
    def test_virtual_staining_combine_stream_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.virtual_staining import virtual_staining_combine_stream

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_ome_zarr
        from fixtures.slurm import submit_stage
        from iohub.ngff import open_ome_zarr

        seed_from_cache(
            real_data_workdir,
            reference_cache,
            [
                '1-preprocess/live_imaging/virtual_staining/tracking_vs',
                # combine reads the reconstruction store for the position list/geometry.
                '1-preprocess/live_imaging/reconstruction/tracking_phase_2d_optimized.zarr',
            ],
        )

        log_dir = real_data_workdir / "submitit_logs" / "virtual_staining_combine_stream"
        # One fan-out shard (store_index=0) combines only part of the store; SLURM config
        # is keyed vs_combine_stream_track. Signature: process + store_index (not stream_id);
        # track VS is 2d.
        submit_stage(
            "vs_combine_stream_track",
            virtual_staining_combine_stream,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
            process="track",
            dim="2d",
            store_index=0,
        )

        candidate = real_data_workdir / REFERENCE_EXPERIMENT / "1-preprocess/live_imaging/virtual_staining/tracking_vs.zarr"
        reference = reference_cache / "1-preprocess/live_imaging/virtual_staining/tracking_vs.zarr"
        # store_index=0 writes only its position batch; compare that subset against the
        # reference (combine streams the cached shards, so pixel-exact is expected).
        with open_ome_zarr(candidate) as c:
            batch_positions = [p for p, _ in c.positions()]
        assert batch_positions, "shard wrote no positions"
        compare_ome_zarr(candidate, reference, positions=batch_positions, rtol=1e-5)



class TestVsCombineStreamPhenoRealData:
    """End-to-end real-data test for dispatch stage 'vs_combine_stream_pheno'.

    Combines pheno VS inference
    """

    @pytest.mark.real_data
    @pytest.mark.skip(
        reason="vs_combine_stream is fanned out per store_index; a single shard "
        "combines only part of the store, so the candidate phenotyping_vs.zarr is "
        "partial vs the full reference (position-list mismatch). Needs a full "
        "multi-shard run or a position-subset compare."
    )
    def test_vs_combine_stream_pheno_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.virtual_staining import virtual_staining_combine_stream

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_ome_zarr
        from fixtures.slurm import submit_stage

        seed_from_cache(
            real_data_workdir,
            reference_cache,
            ['1-preprocess/live_imaging/virtual_staining/phenotyping_vs'],
        )

        log_dir = real_data_workdir / "submitit_logs" / "vs_combine_stream_pheno"
        submit_stage(
            "vs_combine_stream_pheno",
            virtual_staining_combine_stream,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
            stream_id=0,
        )

        candidate = real_data_workdir / REFERENCE_EXPERIMENT / "1-preprocess/live_imaging/virtual_staining/phenotyping_vs.zarr"
        reference = reference_cache / "1-preprocess/live_imaging/virtual_staining/phenotyping_vs.zarr"
        compare_ome_zarr(candidate, reference, rtol=1e-5)

