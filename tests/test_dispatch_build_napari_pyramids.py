"""Tests for build_pyramids."""

import pytest

EXPERIMENT = "ops9999_20260101"


@pytest.fixture
def hermetic_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_FAST_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_CONFIGS_DIR", str(tmp_path / "configs"))
    (tmp_path / EXPERIMENT / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


class TestBuildNapariPyramids:
    def test_function_exists_and_callable(self):
        # Smoke test: the function should be importable and have the
        # expected signature.
        from cyclops_process.processes.pyramids import launcher as mod
        import inspect
        sig = inspect.signature(mod.build_pyramids)
        assert "experiment" in sig.parameters
        assert "wells" in sig.parameters
        assert "store_key" in sig.parameters
        assert "levels" in sig.parameters
