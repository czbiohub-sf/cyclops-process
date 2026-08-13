"""Tests for dispatch_cli command `link_tracking`.

Maps to cyclops_process.processes.assemble_link.link_tracking.

Tests focus on the highest-signal branches: source-path discovery (fast
partition vs dragonfly), and early-return when no source zarrs exist.
Deep behavior (the symlink mapping over tracking position lists) requires
extensive real zarr fixtures and is deferred.
"""

import pytest

from cyclops_process.processes import assemble_link as mod


EXPERIMENT = "ops9999_20260101"


@pytest.fixture
def hermetic_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_FAST_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_CONFIGS_DIR", str(tmp_path / "configs"))
    # @versioned_function writes a function_call_log lockfile under <experiment>/logs/.
    (tmp_path / EXPERIMENT / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


class TestLinkTrackingSourceDiscovery:
    def test_returns_early_when_no_tracking_zarrs_found(
        self, hermetic_experiment, capsys
    ):
        # Neither fast partition nor dragonfly has tracking_*.zarr.
        # The function should print and return without raising.
        mod.link_tracking(EXPERIMENT)
        out = capsys.readouterr().out
        assert "No tracking_*.zarr chunks found" in out

    def test_uses_fast_partition_when_it_has_tracking_zarrs(
        self, hermetic_experiment, capsys, monkeypatch
    ):
        # Make fast_convert exist with a tracking_X.zarr file so the function
        # selects it. We don't need a real zarr — we'll short-circuit before
        # heavy zarr work by monkeypatching open_ome_zarr to raise after the
        # source selection has been logged.
        fast_root = hermetic_experiment / EXPERIMENT / "0-convert" / "live_imaging" / "raw_convert"
        fast_root.mkdir(parents=True)
        (fast_root / "tracking_1.zarr").mkdir()

        # open_ome_zarr is used after source selection to read positions.
        # Make it raise so we abort early but after the "Using fast partition"
        # line has been printed.
        def boom(*args, **kwargs):
            raise RuntimeError("aborted after source selection")
        monkeypatch.setattr(mod, "open_ome_zarr", boom)

        with pytest.raises(RuntimeError, match="aborted"):
            mod.link_tracking(EXPERIMENT)
        out = capsys.readouterr().out
        assert "Using fast partition" in out
        assert str(fast_root) in out

    def test_uses_dragonfly_when_fast_partition_empty(
        self, hermetic_experiment, capsys
    ):
        # fast_convert exists but has no tracking_*.zarr files — code falls
        # back to dragonfly. dragonfly path won't exist under our isolated
        # setup, so glob returns [] and the function prints "Using dragonfly"
        # then "No tracking_*.zarr chunks found".
        fast_root = hermetic_experiment / EXPERIMENT / "0-convert" / "live_imaging" / "raw_convert"
        fast_root.mkdir(parents=True)
        # Empty: no tracking_*.zarr files inside.

        mod.link_tracking(EXPERIMENT)
        out = capsys.readouterr().out
        assert "Using dragonfly" in out
        assert "No tracking_*.zarr chunks found" in out

class TestLinkTrackingRealData:
    """End-to-end real-data test for dispatch stage 'link_tracking'.

    Links 5x tracking from external instrument
    """

    @pytest.mark.real_data
    def test_link_tracking_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.assemble_link import link_tracking

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_ome_zarr
        from fixtures.slurm import submit_stage

        # no upstream seeding needed for this stage

        log_dir = real_data_workdir / "submitit_logs" / "link_tracking"
        submit_stage(
            "link_tracking",
            link_tracking,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
        )

        candidate = real_data_workdir / REFERENCE_EXPERIMENT / "0-convert/live_imaging/tracking_symlink.zarr"
        reference = reference_cache / "0-convert/live_imaging/tracking_symlink.zarr"
        compare_ome_zarr(candidate, reference, rtol=1e-5)

