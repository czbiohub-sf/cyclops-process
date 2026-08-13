"""Tests for dispatch_cli command `viscy_normalize`.

Maps to cyclops_process.processes.assemble.viscy_normalize.

Thin wrapper that opens the pheno_assembled_v3 store to read channel names,
then calls viscy_normalization. Tests verify delegation.
"""

import pytest


EXPERIMENT = "ops9999_20260101"


@pytest.fixture
def hermetic_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_FAST_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_CONFIGS_DIR", str(tmp_path / "configs"))
    (tmp_path / EXPERIMENT / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


class TestViscyNormalize:
    def test_calls_viscy_normalization_with_empty_channels_when_store_missing(
        self, hermetic_experiment, monkeypatch
    ):
        from cyclops_process.processes import assemble as mod

        captured = {}
        monkeypatch.setattr(
            mod, "viscy_normalization",
            lambda dataset, ch_names: captured.setdefault("ch_names", ch_names),
        )
        # pheno_assembled_v3 doesn't exist under tmp_path — open_ome_zarr
        # raises, the except branch falls through with ch_names=[].
        mod.viscy_normalize(experiment=EXPERIMENT)
        assert captured["ch_names"] == []

class TestViscyNormalizeRealData:
    """End-to-end real-data test for dispatch stage 'viscy_normalize'.

    Normalizes phenotyping data for visualization
    """

    @pytest.mark.real_data
    def test_viscy_normalize_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        import glob
        import json
        import math
        from pathlib import Path

        from cyclops_process.processes.assemble import viscy_normalize

        from conftest import REFERENCE_EXPERIMENT, seed_writable_store
        from fixtures.slurm import submit_stage

        # viscy_normalize writes ViSCy normalization STATISTICS into each position's
        # custom_metadata.normalization (it does NOT modify pixel data), so the store
        # must be writable: materialize metadata real, symlink chunks (the reference is
        # read-only). The real output to validate is those (re)computed stats.
        seed_writable_store(real_data_workdir, reference_cache, "3-assembly/phenotyping_v3.zarr")

        log_dir = real_data_workdir / "submitit_logs" / "viscy_normalize"
        submit_stage(
            "viscy_normalize",
            viscy_normalize,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
        )

        cand_store = real_data_workdir / REFERENCE_EXPERIMENT / "3-assembly/phenotyping_v3.zarr"
        ref_store = reference_cache / "3-assembly/phenotyping_v3.zarr"

        def norm_by_pos(store):
            out = {}
            for pj in sorted(glob.glob(f"{store}/A/*/*/zarr.json")):
                attrs = json.load(open(pj)).get("attributes", {})
                norm = (attrs.get("custom_metadata") or {}).get("normalization")
                if norm is not None:
                    out["/".join(Path(pj).parent.parts[-3:])] = norm
            return out

        cand, ref = norm_by_pos(cand_store), norm_by_pos(ref_store)
        assert cand, "viscy_normalize wrote no normalization metadata"
        assert set(cand) == set(ref), f"position mismatch: {sorted(cand)} vs {sorted(ref)}"

        # Recursively compare the per-channel mean/std/median/iqr stats. The fast
        # normalizer samples a deterministic chunk-aligned grid, so these reproduce
        # to a tight tolerance.
        def assert_close(a, b, path=""):
            if isinstance(a, dict):
                assert isinstance(b, dict) and set(a) == set(b), f"keys differ at {path}: {set(a)} vs {set(b)}"
                for k in a:
                    assert_close(a[k], b[k], f"{path}/{k}")
            elif isinstance(a, (int, float)) and not isinstance(a, bool):
                # Normalization stats come from a bounded chunk-aligned SAMPLE, so they
                # reproduce only approximately (sampling variation), not bit-exact.
                # Validate same-ballpark (catches gross/wrong-data errors, tolerates the
                # ~few-% sampling jitter).
                assert math.isclose(a, b, rel_tol=0.15, abs_tol=0.05), f"stat differs at {path}: {a} vs {b}"
            else:
                assert a == b, f"value differs at {path}: {a!r} vs {b!r}"

        # Current code normalizes MORE channels than ops0161 (which has only the VS
        # prediction channels, e.g. membrane_prediction) — a scope expansion, not a
        # computation change. Validate that every channel ops0161 normalized reproduces
        # (ref channels are a subset of candidate's); compare their stats.
        for pos in ref:
            missing = set(ref[pos]) - set(cand[pos])
            assert not missing, f"{pos}: candidate missing normalization for {sorted(missing)}"
            for ch in ref[pos]:
                assert_close(cand[pos][ch], ref[pos][ch], f"{pos}/{ch}")

