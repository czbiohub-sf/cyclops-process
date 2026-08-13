"""Tests for dispatch_cli command `link_phenotyping`.

Maps to cyclops_process.processes.assemble_link.link_phenotyping.

The function does substantial orchestration: source-partition discovery,
position-list read, channel_map validation, dest-store creation. Tests here
target the explicit error branch (missing channel_map) and source-discovery
logging. Deep behavior is deferred.
"""

import json

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


def _write_fast_position_list(fast_root, content):
    fast_root.mkdir(parents=True, exist_ok=True)
    (fast_root / "pheno_position_list.json").write_text(json.dumps(content))


class FakeDataset:
    """Minimal OpsDataset stand-in for branching tests.

    Only sets the attributes link_phenotyping actually reads before its first
    error branch fires.
    """

    def __init__(self, tmp_path, channel_map_data=None, has_fast_pheno_zarr=False):
        self.experiment_path_fast = tmp_path / EXPERIMENT
        self.lc_dragonfly_dir = tmp_path / "dragonfly_missing"  # never exists
        self.store_paths = {"lc_20x": tmp_path / EXPERIMENT / "lc_20x.zarr"}
        self.config_paths = {
            "lc_20x_position_list": tmp_path / "pheno_position_list.json"
        }
        self.channel_map_data = channel_map_data
        # Pre-write position list at the configured location so the function
        # can json.load it before hitting the channel-map check.
        self.config_paths["lc_20x_position_list"].write_text(
            json.dumps({"A1-Site_0": [0, 0], "A1-Site_1": [0, 1]})
        )
        if has_fast_pheno_zarr:
            fast_root = self.experiment_path_fast / "0-convert" / "live_imaging" / "raw_convert"
            fast_root.mkdir(parents=True)
            (fast_root / "phenotyping_well_A1_1.zarr").mkdir()


class TestLinkPhenotypingChannelMap:
    def test_missing_channel_map_raises_valueerror(
        self, hermetic_experiment, monkeypatch
    ):
        fake = FakeDataset(hermetic_experiment, channel_map_data={})
        monkeypatch.setattr(mod, "OpsDataset", lambda *a, **kw: fake)
        # Also stub _filter_positions_by_wells since FakeDataset may not satisfy
        # whatever it does internally.
        monkeypatch.setattr(
            mod, "_filter_positions_by_wells",
            lambda dataset, positions: positions,
        )

        with pytest.raises(ValueError, match="No channel_map found"):
            mod.link_phenotyping(experiment=EXPERIMENT)

    def test_none_channel_map_raises_valueerror(
        self, hermetic_experiment, monkeypatch
    ):
        # channel_map_data can be None — `ch_to_org = dataset.channel_map_data or {}`
        # yields {} and triggers the same ValueError.
        fake = FakeDataset(hermetic_experiment, channel_map_data=None)
        monkeypatch.setattr(mod, "OpsDataset", lambda *a, **kw: fake)
        monkeypatch.setattr(
            mod, "_filter_positions_by_wells",
            lambda dataset, positions: positions,
        )

        with pytest.raises(ValueError, match="No channel_map found"):
            mod.link_phenotyping(experiment=EXPERIMENT)


class TestLinkPhenotypingSourceDiscovery:
    def test_logs_fast_partition_when_pheno_zarrs_present(
        self, hermetic_experiment, capsys, monkeypatch
    ):
        # The source-detection log line precedes the channel_map check, so we
        # let the function fail at the channel_map check and inspect stdout.
        fake = FakeDataset(
            hermetic_experiment, channel_map_data=None,
            has_fast_pheno_zarr=True,
        )
        monkeypatch.setattr(mod, "OpsDataset", lambda *a, **kw: fake)
        monkeypatch.setattr(
            mod, "_filter_positions_by_wells",
            lambda dataset, positions: positions,
        )

        with pytest.raises(ValueError):
            mod.link_phenotyping(experiment=EXPERIMENT)
        out = capsys.readouterr().out
        assert "Using fast partition" in out

class TestLinkPhenotypingRealData:
    """End-to-end real-data test for dispatch stage 'link_phenotyping'.

    Links 20x phenotyping from external instrument
    """

    @pytest.mark.real_data
    def test_link_phenotyping_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.assemble_link import link_phenotyping

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_ome_zarr
        from fixtures.slurm import submit_stage

        # no upstream seeding needed for this stage

        log_dir = real_data_workdir / "submitit_logs" / "link_phenotyping"
        submit_stage(
            "link_phenotyping",
            link_phenotyping,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
        )

        candidate = real_data_workdir / REFERENCE_EXPERIMENT / "0-convert/live_imaging/phenotyping_transform.zarr"
        reference = reference_cache / "0-convert/live_imaging/phenotyping_transform.zarr"
        # The reference cache predates the additional CellPaint stains, so its
        # channels (['BF']) are a clean SUBSET of the candidate's 9 channels
        # (BF, CP1_*, CP2_*). Compare only the shared channel; the extra
        # candidate channels have no reference yet. (Verified channel_names:
        # candidate=['BF','CP1_nuclei_Hoechst',...,'CP2_ER_ConA'], reference=['BF'].)
        compare_ome_zarr(candidate, reference, rtol=1e-5, channels=["BF"])

