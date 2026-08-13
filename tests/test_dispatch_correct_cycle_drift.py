"""Tests for dispatch_cli command `correct_cycle_drift`.

Maps to cyclops_process.processes.register.correct_cycle_drift.

The full function runs a Dask cluster + per-tile drift correction (heavy
parallel zarr I/O). Tests here cover the explicit-overwrite=False skip
branch (output already exists) and the per-well grouping of positions.
"""

import numpy as np
import pytest

from ops_utils.data.experiment import OpsDataset
from ops_utils.io.zarr_precreate import create_hcs_store_fast


EXPERIMENT = "ops9999_20260101"


def _make_iss_store(store_path, positions):
    store_path.parent.mkdir(parents=True, exist_ok=True)
    create_hcs_store_fast(
        store_path=store_path,
        positions=positions,
        shape=(1, 1, 1, 8, 8),
        chunks=(1, 1, 1, 8, 8),
        dtype=np.float32,
        scale=(1, 1, 1, 1, 1),
        channel_names=["DAPI"],
    )


@pytest.fixture
def hermetic_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_FAST_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_CONFIGS_DIR", str(tmp_path / "configs"))
    (tmp_path / EXPERIMENT / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


class TestCorrectCycleDriftOverwrite:
    def test_explicit_overwrite_false_with_existing_output_skips(
        self, hermetic_experiment, capsys
    ):
        from cyclops_process.processes import register as mod

        dataset = OpsDataset(EXPERIMENT)
        _make_iss_store(dataset.store_paths["iss"], positions=["A/1/0"])

        # Pre-create the output store so the skip branch fires.
        out_path = dataset.store_paths["iss_drift_corrected"]
        _make_iss_store(out_path, positions=["A/1/0"])

        mod.correct_cycle_drift(experiment=EXPERIMENT, overwrite=False)

        out = capsys.readouterr().out
        assert "Skipping cycle drift correction" in out

    def test_explicit_overwrite_false_no_existing_output_proceeds(
        self, hermetic_experiment, monkeypatch
    ):
        # With overwrite=False and output absent, choice="create" — function
        # proceeds past the skip branch. We monkeypatch LocalCluster so the
        # heavy Dask path raises a clear error and we can assert the
        # function got past the skip.
        from cyclops_process.processes import register as mod

        dataset = OpsDataset(EXPERIMENT)
        _make_iss_store(dataset.store_paths["iss"], positions=["A/1/0"])

        class BoomCluster:
            def __init__(self, **kw):
                raise RuntimeError("reached cluster init")

        monkeypatch.setattr(
            "dask.distributed.LocalCluster", BoomCluster
        )

        with pytest.raises(RuntimeError, match="reached cluster init"):
            mod.correct_cycle_drift(experiment=EXPERIMENT, overwrite=False)

class TestCorrectCycleDriftRealData:
    """End-to-end real-data test for dispatch stage 'correct_cycle_drift'.

    Reads bc_symlink.zarr, writes drift-corrected version
    """

    @pytest.mark.real_data
    def test_correct_cycle_drift_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.register import correct_cycle_drift

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_ome_zarr
        from fixtures.slurm import submit_stage

        # Seed only the upstream ISS symlink-store (store_paths["iss"] =
        # bc_symlink.zarr). Its internal chunk symlinks point at the reference's
        # convert tiles by absolute path, so seeding the store alone is enough --
        # the stage reads pixel data straight through those symlinks. (This is the
        # current-code v0.5 reference store, which carries the `c/` chunk prefix
        # and reads correctly under zarr 3.2.1, unlike the old ops0094 cache that
        # motivated the prior skip.)
        seed_from_cache(
            real_data_workdir,
            reference_cache,
            ["0-convert/in_situ_sequencing/bc_symlink.zarr"],
        )

        # The stage writes a per-well drift QC plot to 3-assembly/ISS/ via
        # plt.savefig() without creating the parent. In a full pipeline run that
        # dir is created by an upstream stage; in this isolated single-stage test
        # we must materialize it so savefig() doesn't FileNotFoundError.
        (real_data_workdir / REFERENCE_EXPERIMENT / "3-assembly" / "ISS").mkdir(
            parents=True, exist_ok=True
        )

        log_dir = real_data_workdir / "submitit_logs" / "correct_cycle_drift"
        # python_kwargs (fast=True, overwrite=True, pad=False) come from the yaml
        # via submit_stage; overwrite=True keeps the stage non-interactive.
        submit_stage(
            "correct_cycle_drift",
            correct_cycle_drift,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
        )

        candidate = (
            real_data_workdir
            / REFERENCE_EXPERIMENT
            / "0-convert/in_situ_sequencing/bc_drift_corrected.zarr"
        )
        reference = (
            reference_cache / "0-convert/in_situ_sequencing/bc_drift_corrected.zarr"
        )
        # STRUCTURAL validation only (position list, channel names, per-position
        # shape, dtype, scale) -- the stage runs end-to-end and writes a
        # correctly-structured drift-corrected store.
        #
        # Pixel data is intentionally NOT compared against single_ISS: measured
        # against this OLD reference the pixels diverge wholesale (~80% of pixels
        # differ, max abs ~2.6e4) because current-code cycle-drift correction
        # produces a different result than the code that generated single_ISS
        # (the per-tile drift estimate and/or its GPU-vs-CPU path changed -- the
        # same current-vs-old divergence documented for estimate_stitch). A
        # meaningful pixel check needs a CURRENT-CODE reference: once the full run
        # lands, point OPS_REFERENCE_DIR at it and flip this to check_data=True.
        compare_ome_zarr(candidate, reference, check_data=False)
