"""Tests for dispatch_cli command `register_iss_seg_to_nucleus`.

Maps to cyclops_process.processes.auto_register.iss_cycle_register_orchestrator
  .job_register_segmentation_to_nucleus.

The wrapper builds paths, mkdir's transforms/overlays dirs, then delegates to
the heavy `register_segmentation_to_nucleus` helper. Tests verify that
delegation contract.
"""

from pathlib import Path

import pytest


EXPERIMENT = "ops9999_20260101"


@pytest.fixture
def hermetic_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_FAST_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPS_CONFIGS_DIR", str(tmp_path / "configs"))
    (tmp_path / EXPERIMENT / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


class TestJobRegisterSegmentationToNucleus:
    def test_calls_helper_with_correct_paths(
        self, hermetic_experiment, monkeypatch
    ):
        from cyclops_process.processes.auto_register import iss_cycle_register_orchestrator as mod
        from ops_utils.data.experiment import OpsDataset

        captured = {}

        def fake_helper(iss_zarr, seg_zarr, position, transforms_dir,
                        overlays_dir, verbose=True):
            captured["iss_zarr"] = iss_zarr
            captured["seg_zarr"] = seg_zarr
            captured["position"] = position
            captured["transforms_dir"] = transforms_dir
            captured["overlays_dir"] = overlays_dir
            return {"yaml_path": transforms_dir / "segmentation_to_nucleus.yaml"}

        monkeypatch.setattr(
            mod, "register_segmentation_to_nucleus", fake_helper
        )

        mod.job_register_segmentation_to_nucleus(
            experiment=EXPERIMENT, well=2, verbose=False
        )

        ds = OpsDataset(EXPERIMENT)
        assert captured["iss_zarr"] == ds.store_paths["iss_stitch"]
        assert captured["seg_zarr"] == ds.store_paths["iss_segmentation"]
        assert captured["position"] == "A/2/0"
        # transforms/overlays dirs are well-scoped under preprocess_in_situ/register/.
        assert captured["transforms_dir"] == ds.preprocess_in_situ / "register/transforms/A2"
        assert captured["overlays_dir"] == ds.preprocess_in_situ / "register/overlays/A2"
        # The dirs should have been created.
        assert captured["transforms_dir"].is_dir()
        assert captured["overlays_dir"].is_dir()

class TestRegisterIssSegToNucleusRealData:
    """End-to-end real-data test for dispatch stage 'register_iss_seg_to_nucleus'.

    Fan-out per well; registers segmentation to nucleus
    """

    @pytest.mark.real_data
    def test_register_iss_seg_to_nucleus_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version
    ):
        from cyclops_process.processes.auto_register.iss_cycle_register_orchestrator import job_register_segmentation_to_nucleus

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_yaml
        from fixtures.slurm import submit_stage

        # Deterministic stage: computes a phase-cross-correlation translation
        # between the segmentation mask and Round-0 DAPI (centered 2048 crop, no
        # randomness, no cache), reading only iss_stitch (bc_stitched.zarr) and
        # iss_segmentation (bc_segmentation.zarr) -- both seeded as symlinks
        # (read-through; only the center crop is loaded). Output is a
        # translation-only affine YAML.
        seed_from_cache(
            real_data_workdir,
            reference_cache,
            [
                '1-preprocess/in_situ_sequencing/stitch/bc_stitched.zarr',
                '1-preprocess/in_situ_sequencing/segmentation/bc_segmentation.zarr',
            ],
        )

        log_dir = real_data_workdir / "submitit_logs" / "register_iss_seg_to_nucleus"
        submit_stage(
            "register_iss_seg_to_nucleus",
            job_register_segmentation_to_nucleus,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
            well=1,
        )

        rel = "1-preprocess/in_situ_sequencing/register/transforms/A1/segmentation_to_nucleus.yaml"
        candidate = real_data_workdir / REFERENCE_EXPERIMENT / rel
        reference = reference_cache / rel
        # Deterministic PCC on identical crops -> reproduces the reference shift;
        # atol guards float/platform noise. Structural fields (channels/interp)
        # compare exactly.
        compare_yaml(candidate, reference, atol=1e-6)
