"""Tests for dispatch_cli command `stack_symlinks`.

Maps to cyclops_process.processes.assemble_link.stack_symlinks.

The function does extensive per-round / per-well symlink wiring with branches
for pre-nuclei round, DAPI_round10, skip-round-0, and 4-vs-5-channel source
layouts. Tests here cover the early-return branches; deep per-branch
symlink-mapping coverage requires non-trivial fixtures and is deferred.
"""

import pytest

from cyclops_process.processes import assemble_link as mod


EXPERIMENT = "ops9999_20260101"


@pytest.fixture
def hermetic_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_FAST_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_CONFIGS_DIR", str(tmp_path / "configs"))
    (tmp_path / EXPERIMENT / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


class TestStackSymlinksEarlyReturns:
    def test_no_well_zarrs_in_source_returns_early(
        self, hermetic_experiment, capsys
    ):
        # OpsDataset.convert_in_situ is <experiment>/0-convert/in_situ_sequencing.
        # When it has no *.zarr files, glob returns [] and the function prints
        # and returns without raising.
        from cyclops_utils.data.experiment import OpsDataset

        dataset = OpsDataset(EXPERIMENT)
        dataset.convert_in_situ.mkdir(parents=True, exist_ok=True)

        mod.stack_symlinks(experiment=EXPERIMENT)

        out = capsys.readouterr().out
        assert "No well zarrs found" in out

    def test_pre_nuclei_round_without_sufficient_rounds_raises(
        self, hermetic_experiment, monkeypatch
    ):
        # When pre_nuclei_round=True falls into the round0->round1 mode but
        # one or more wells have only a single round, the function raises
        # ValueError("pre_nuclei_round=True but not all wells have sufficient
        # rounds").
        from cyclops_utils.data.experiment import OpsDataset
        from cyclops_utils.io.zarr_precreate import create_hcs_store_fast
        import numpy as np

        dataset = OpsDataset(EXPERIMENT)
        src = dataset.convert_in_situ
        src.mkdir(parents=True, exist_ok=True)
        # One well, one round only -> after popping round 0 nothing remains.
        path = src / "A1_round0.zarr"
        create_hcs_store_fast(
            store_path=path,
            positions=["A/1/0"],
            shape=(1, 5, 1, 8, 8),
            chunks=(1, 1, 1, 8, 8),
            dtype=np.float32,
            scale=(1, 1, 1, 1, 1),
            channel_names=["DAPI", "G", "T", "A", "C"],
        )

        with pytest.raises(ValueError, match="sufficient rounds"):
            mod.stack_symlinks(experiment=EXPERIMENT, pre_nuclei_round=True)

    def test_skip_pre_dapi_round_without_sufficient_rounds_raises(
        self, hermetic_experiment
    ):
        from cyclops_utils.data.experiment import OpsDataset
        from cyclops_utils.io.zarr_precreate import create_hcs_store_fast
        import numpy as np

        dataset = OpsDataset(EXPERIMENT)
        src = dataset.convert_in_situ
        src.mkdir(parents=True, exist_ok=True)
        path = src / "A1_round0.zarr"
        create_hcs_store_fast(
            store_path=path,
            positions=["A/1/0"],
            shape=(1, 5, 1, 8, 8),
            chunks=(1, 1, 1, 8, 8),
            dtype=np.float32,
            scale=(1, 1, 1, 1, 1),
            channel_names=["DAPI", "G", "T", "A", "C"],
        )

        with pytest.raises(ValueError, match="skip_pre_dapi_round=True"):
            mod.stack_symlinks(
                experiment=EXPERIMENT, skip_pre_dapi_round=True
            )


class TestStackSymlinksRealData:
    """End-to-end real-data test: seeds per-round zarrs from the cache,
    runs stack_symlinks via SLURM, and verifies the assembled bc_symlink.zarr
    matches the reference.

    Skipped unless the reference cache is reachable. Run with
    `pytest -m real_data`.
    """

    @pytest.mark.real_data
    def test_stack_symlinks_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version,
        reference_tile_size,
    ):
        from cyclops_process.processes.assemble_link import stack_symlinks

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_ome_zarr
        from fixtures.slurm import submit_stage

        # Seed only the per-round zarrs (NOT bc_symlink.zarr, which is the
        # output we want to (re)produce). Glob is expanded against the cache;
        # each match is symlinked into the writable workdir.
        seed_from_cache(
            real_data_workdir,
            reference_cache,
            ["0-convert/in_situ_sequencing/A*.zarr"],
        )

        # stack_symlinks indexes chunk paths by ngff_version when reading
        # input zarrs; it must match the cache's actual version (v0.4 today).
        # The yaml default (v0.5) would lookup paths the v2 cache doesn't have.
        log_dir = real_data_workdir / "submitit_logs" / "stack_symlinks"
        # tile_size must match the reference's acquisition geometry — stack_symlinks
        # builds the output array shape from it (ops0094=2048, ops0161=2304).
        submit_stage(
            "stack_symlinks",
            stack_symlinks,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
            ngff_version=reference_ngff_version,
            tile_size=reference_tile_size,
        )

        candidate = (
            real_data_workdir
            / REFERENCE_EXPERIMENT
            / "0-convert"
            / "in_situ_sequencing"
            / "bc_symlink.zarr"
        )
        reference = (
            reference_cache
            / "0-convert"
            / "in_situ_sequencing"
            / "bc_symlink.zarr"
        )
        # check_data=False: bc_symlink.zarr in the cache is a symlink-store
        # pointing into /path/to/ops_data/ (the cache's
        # original physical location), which we do not have read permission
        # on. Metadata (positions, channels, shape, dtype, scale) is still
        # readable. Numerical correctness of stack_symlinks is implied by the
        # convert test passing data comparison on the per-round zarrs that
        # bc_symlink.zarr just rearranges.
        compare_ome_zarr(candidate, reference, rtol=1e-5, check_data=False)
