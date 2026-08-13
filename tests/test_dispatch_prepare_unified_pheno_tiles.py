"""Tests for dispatch_cli command `prepare_unified_pheno_tiles`.

Maps to cyclops_process.processes.register.prepare_unified_pheno_tiles.

The function orchestrates fluor-registration + unified-tile-store creation
over phase/fluor/VS stores. Tests cover the no-fluor-channels skip path of
the inner conditional; deep coverage is deferred.
"""

import numpy as np
import pytest

from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.io.zarr_precreate import create_hcs_store_fast


EXPERIMENT = "ops9999_20260101"


def _make_store(store_path, positions, channel_names):
    store_path.parent.mkdir(parents=True, exist_ok=True)
    create_hcs_store_fast(
        store_path=store_path,
        positions=positions,
        shape=(1, len(channel_names), 1, 4, 4),
        chunks=(1, 1, 1, 4, 4),
        dtype=np.float32,
        scale=(1, 1, 1, 1, 1),
        channel_names=channel_names,
    )


@pytest.fixture
def hermetic_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_FAST_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_CONFIGS_DIR", str(tmp_path / "configs"))
    (tmp_path / EXPERIMENT / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


class TestPrepareUnifiedPhenoTiles:
    def test_no_fluor_channels_skips_fluor_registration(
        self, hermetic_experiment, capsys
    ):
        from cyclops_process.processes import register as mod

        dataset = OpsDataset(EXPERIMENT)
        # lc_20x with no GFP/mCherry/Cy5 → fluor registration is skipped.
        _make_store(
            dataset.store_paths["lc_20x"],
            positions=["A/1/0"], channel_names=["BF"],
        )

        # Phase/VS stores don't exist; the function will raise later, but
        # only after printing the "no fluorescent channels" line.
        with pytest.raises(Exception):
            mod.prepare_unified_pheno_tiles(experiment=EXPERIMENT)

        out = capsys.readouterr().out
        assert "No fluorescent channels detected" in out

class TestPrepareUnifiedPhenoTilesRealData:
    """End-to-end real-data test for dispatch stage 'prepare_unified_pheno_tiles'.

    Builds unified tile store combining pheno phase and VS
    """

    @pytest.mark.real_data
    @pytest.mark.skip(
        reason="requires the lc_20x_fluor_2d_registered store "
        "(phenotyping_fluor_2d_registered.zarr), which is NOT present in the "
        "reference cache (the fluor branch outputs were not retained). Cannot run "
        "without that input."
    )
    def test_prepare_unified_pheno_tiles_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.register import prepare_unified_pheno_tiles

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_ome_zarr
        from fixtures.slurm import submit_stage

        seed_from_cache(
            real_data_workdir,
            reference_cache,
            ['1-preprocess/live_imaging/segmentation/phenotyping_segmentation_stitched.zarr'],
        )

        log_dir = real_data_workdir / "submitit_logs" / "prepare_unified_pheno_tiles"
        submit_stage(
            "prepare_unified_pheno_tiles",
            prepare_unified_pheno_tiles,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
        )

        candidate = real_data_workdir / REFERENCE_EXPERIMENT / "1-preprocess/live_imaging/stitch/phenotyping_tiles_unified.zarr"
        reference = reference_cache / "1-preprocess/live_imaging/stitch/phenotyping_tiles_unified.zarr"
        compare_ome_zarr(candidate, reference, rtol=1e-5)

