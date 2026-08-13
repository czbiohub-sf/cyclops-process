"""Tests for virtual_staining_combine_setup."""

import pytest

EXPERIMENT = "ops9999_20260101"


@pytest.fixture
def hermetic_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_FAST_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_CONFIGS_DIR", str(tmp_path / "configs"))
    (tmp_path / EXPERIMENT / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


class TestVirtualStainingCombineSetup:
    def test_invalid_dim_raises(self, hermetic_experiment):
        from cyclops_process.processes import virtual_staining as mod
        with pytest.raises(ValueError, match="dim must be one of"):
            mod.virtual_staining_combine_setup(
                experiment=EXPERIMENT, process="track", dim="4d",
            )

    def test_unknown_process_raises(self, hermetic_experiment):
        from cyclops_process.processes import virtual_staining as mod
        with pytest.raises(ValueError, match="process must be"):
            mod.virtual_staining_combine_setup(
                experiment=EXPERIMENT, process="bogus", dim="3d",
            )

    def test_missing_intermediate_dir_raises_runtimeerror(
        self, hermetic_experiment
    ):
        from cyclops_process.processes import virtual_staining as mod
        with pytest.raises(RuntimeError, match="Intermediate path does not exist"):
            mod.virtual_staining_combine_setup(
                experiment=EXPERIMENT, process="track", dim="3d",
            )

class TestVsCombineTimepointPreservation:
    """Regression test for the VS-combine time-axis collapse
    (A. Hillsley's DRAFT_vs_combine_timepoint_fix).

    Batch-format VS inference flattens each (position, timepoint) into a separate
    batch sample, and ``create_combine_output_store`` allocates the output with a
    hardcoded ``T=1`` (``full_shape = (1,) + sample_shape``) while
    ``get_volume_shape_from_input`` drops the input's T axis entirely. For tracking
    — where each position is a genuine time series — this collapses T>1 -> T=1,
    which forces downstream segmentation / auto_register onto a single timepoint
    (the BoundsCheckError that Fix 0n only band-aided).

    The fix is architectural: route tracking through HCS-format inference
    (``viscy_multigpu_inference``, currently absent) so the combined output
    inherits the input's timepoint count. This test pins that invariant and is
    xfail until the HCS path lands; it flips to xpass when combine preserves T.
    """

    @pytest.mark.xfail(
        reason="batch-format VS combine hardcodes output T=1 (create_output_store) "
        "and get_volume_shape_from_input drops the input T axis; tracking needs "
        "HCS-format inference to preserve the time series. See A. Hillsley "
        "DRAFT_vs_combine_timepoint_fix / royerlab/ops_process#114 thread.",
        strict=False,
    )
    def test_combine_output_preserves_input_timepoints(self, tmp_path):
        import numpy as np
        import zarr
        from iohub.ngff import open_ome_zarr
        from cyclops_utils.io.zarr_precreate import create_hcs_store_fast
        from cyclops_process.utils.combine_batch_predictions import (
            create_combine_output_store,
        )

        n_timepoints = 3

        # Minimal tracking-style input: one position, a real time series (T=3).
        input_store = tmp_path / "tracking_input.zarr"
        create_hcs_store_fast(
            store_path=input_store,
            positions=["A/1/0"],
            shape=(n_timepoints, 1, 1, 4, 4),  # (T, C, Z, Y, X)
            chunks=(1, 1, 1, 4, 4),
            dtype=np.float32,
            scale=(1.0, 1.0, 1.0, 1.0, 1.0),
            channel_names=["phase"],
        )

        # Minimal batch-format prediction shard: predictions_{start}_{end}.zarr
        # holding a batch_NNNNN array (B, C, Z, Y, X) with 2 output channels.
        intermediate = tmp_path / "intermediate"
        intermediate.mkdir()
        pred = zarr.open_group(str(intermediate / "predictions_0_1.zarr"), mode="w")
        pred.create_array("batch_00000", shape=(1, 2, 1, 4, 4), dtype="float32")

        output_store = tmp_path / "tracking_vs.zarr"
        create_combine_output_store(
            intermediate_dir=intermediate,
            input_store=input_store,
            output_store=output_store,
            channel_names=["nuclei", "membrane"],
        )

        with open_ome_zarr(output_store, mode="r") as out:
            t_out = dict(out.positions())["A/1/0"]["0"].shape[0]

        assert t_out == n_timepoints, (
            f"combine collapsed the tracking time axis: output T={t_out}, "
            f"expected {n_timepoints} (the input's timepoint count)"
        )


class TestVirtualStainingCombineSetupRealData:
    """End-to-end real-data test for dispatch stage 'virtual_staining_combine_setup'.

    Counts combine stream items; stdout driven
    """

    @pytest.mark.real_data
    @pytest.mark.skip(reason="stage 'virtual_staining_combine_setup' is an orchestrator / stdout-driven stage with no single output to compare; needs stage-specific test logic")
    def test_virtual_staining_combine_setup_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.virtual_staining import virtual_staining_combine_setup

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.slurm import submit_stage

        # (skipped; upstream seeding not exercised)

        log_dir = real_data_workdir / "submitit_logs" / "virtual_staining_combine_setup"
        submit_stage(
            "virtual_staining_combine_setup",
            virtual_staining_combine_setup,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
        )


class TestVsCombineSetupPhenoRealData:
    """End-to-end real-data test for dispatch stage 'vs_combine_setup_pheno'.

    Setup for pheno VS combine
    """

    @pytest.mark.real_data
    @pytest.mark.skip(reason="stage 'vs_combine_setup_pheno' is an orchestrator / stdout-driven stage with no single output to compare; needs stage-specific test logic")
    def test_vs_combine_setup_pheno_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.virtual_staining import virtual_staining_combine_setup

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.slurm import submit_stage

        # (skipped; upstream seeding not exercised)

        log_dir = real_data_workdir / "submitit_logs" / "vs_combine_setup_pheno"
        submit_stage(
            "vs_combine_setup_pheno",
            virtual_staining_combine_setup,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
        )
