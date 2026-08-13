"""Tests for virtual_staining_combine_only."""

import pytest

EXPERIMENT = "ops9999_20260101"


@pytest.fixture
def hermetic_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_FAST_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_CONFIGS_DIR", str(tmp_path / "configs"))
    (tmp_path / EXPERIMENT / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


class TestVirtualStainingCombineOnly:
    def test_invalid_dim_raises(self, hermetic_experiment):
        from cyclops_process.processes import virtual_staining as mod
        with pytest.raises(ValueError, match="dim must be one of"):
            mod.virtual_staining_combine_only(
                experiment=EXPERIMENT, process="track", dim="4d",
            )

    def test_unknown_process_raises(self, hermetic_experiment):
        from cyclops_process.processes import virtual_staining as mod
        with pytest.raises(ValueError, match="process must be"):
            mod.virtual_staining_combine_only(
                experiment=EXPERIMENT, process="bogus", dim="3d",
            )

    def test_missing_intermediate_dir_raises_runtimeerror(
        self, hermetic_experiment
    ):
        from cyclops_process.processes import virtual_staining as mod
        with pytest.raises(RuntimeError, match="Intermediate path does not exist"):
            mod.virtual_staining_combine_only(
                experiment=EXPERIMENT, process="track", dim="3d",
            )
