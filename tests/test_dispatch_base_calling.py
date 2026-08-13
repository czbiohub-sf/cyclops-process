import pytest

"""Tests for base_calling."""

import inspect


class TestBaseCallingSignature:
    def test_signature_exposes_expected_kwargs(self):
        from cyclops_process.processes import iss as mod
        sig = inspect.signature(mod.base_calling)
        for name in (
            "experiment", "method", "debug", "iss_rounds",
            "failed_rounds_by_well", "n_rounds",
        ):
            assert name in sig.parameters, f"missing kwarg: {name}"

    def test_default_method_is_mine(self):
        from cyclops_process.processes import iss as mod
        sig = inspect.signature(mod.base_calling)
        assert sig.parameters["method"].default == "mine"

    def test_default_n_rounds_is_9(self):
        from cyclops_process.processes import iss as mod
        sig = inspect.signature(mod.base_calling)
        assert sig.parameters["n_rounds"].default == 9

class TestRoundsAvailableGuard:
    """A store short a round must fail before any tile work, naming the store."""

    def test_missing_round_raises_with_store_and_rounds(self):
        from cyclops_process.processes.iss import _assert_rounds_available

        with pytest.raises(ValueError) as exc:
            _assert_rounds_available(
                list(range(10)), 9, "ops0179_20260729", "bc_stitched.zarr A/1/0",
            )
        msg = str(exc.value)
        assert "missing [9]" in msg
        assert "only 9 rounds" in msg
        assert "bc_stitched.zarr A/1/0" in msg

    def test_exact_and_shorter_round_lists_pass(self):
        from cyclops_process.processes.iss import _assert_rounds_available

        _assert_rounds_available(list(range(10)), 10, "exp", "store")
        _assert_rounds_available([0, 2, 4], 10, "exp", "store")


class TestBaseCallingRealData:
    """End-to-end real-data test for dispatch stage 'base_calling'.

    Assigns base identities from spots; outputs reads.csv
    """

    @pytest.mark.real_data
    def test_base_calling_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version,
        reference_n_rounds,
    ):
        from cyclops_process.processes.iss import base_calling

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_csv
        from fixtures.slurm import submit_stage

        # base_calling reads the v3 registered/stitched store + ISS segmentation +
        # the per-well detected-spot point clouds, looks up barcodes in the codebook
        # (resolved from OPS_CONFIGS_DIR/library/), and writes per-well reads CSVs.
        # Needs a CURRENT-CODE reference: it reads store_paths["iss_stitch_registered_v3"],
        # which only exists in a v3 run (point OPS_REFERENCE_DIR at the completed run).
        seed_from_cache(
            real_data_workdir,
            reference_cache,
            [
                "1-preprocess/in_situ_sequencing/register/bc_stitched_registered_v3.zarr",
                "1-preprocess/in_situ_sequencing/segmentation/bc_segmentation.zarr",
                "1-preprocess/in_situ_sequencing/base_calling/A*_detected_points.npy",
            ],
        )

        log_dir = real_data_workdir / "submitit_logs" / "base_calling"
        # n_rounds is a TOP-LEVEL pipeline param (ISS rounds excluding round 0) that
        # nextflow passes on the CLI -- it is NOT in base_calling's per-stage python_kwargs,
        # so submit_stage won't supply it and the signature default would mismatch the
        # reference's per-round Q columns. Derive it from the reference (ops0094=8,
        # ops0161=10) so the read CSVs line up. (method=mine comes from yaml.)
        submit_stage(
            "base_calling",
            base_calling,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
            n_rounds=reference_n_rounds,
        )

        # Per-well reads (base_calling/mine/<well>_reads.csv). Row order is not
        # deterministic (assembled from parallel per-tile workers), so compare the
        # SET of reads order-independently; values are exact (same code + input).
        reads_rel = "1-preprocess/in_situ_sequencing/base_calling/mine"
        cand_dir = real_data_workdir / REFERENCE_EXPERIMENT / reads_rel
        ref_dir = reference_cache / reads_rel
        ref_reads = sorted(ref_dir.glob("A*_reads.csv"))
        assert ref_reads, f"no reference reads CSVs under {ref_dir}"
        for ref_csv in ref_reads:
            compare_csv(cand_dir / ref_csv.name, ref_csv, sort_rows=True)
