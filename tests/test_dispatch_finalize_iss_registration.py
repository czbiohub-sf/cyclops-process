"""Tests for dispatch_cli command `finalize_iss_registration`.

Maps to job_finalize_registration in iss_cycle_register_orchestrator.

This function composes cumulative affines from many per-round YAML files;
testing it deeply requires a fully populated transforms_dir. Tests here
cover the explicit FileNotFoundError when nucleus_to_round0.yml is missing
and DAPI_round10 also absent.
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


class TestJobFinalizeRegistration:
    def test_missing_nucleus_yaml_without_dapi_round10_raises(
        self, hermetic_experiment, monkeypatch
    ):
        # check_for_dapi_round10 normally globs the convert dir — stub it to
        # return False so we reliably hit the FileNotFoundError branch.
        from cyclops_process.processes.auto_register import iss_cycle_register_orchestrator as mod

        monkeypatch.setattr(
            mod, "check_for_dapi_round10",
            lambda dataset, verbose=False: False,
        )

        with pytest.raises(FileNotFoundError, match="nucleus_to_round0.yml not found"):
            mod.job_finalize_registration(
                experiment=EXPERIMENT, well=1, verbose=False, n_rounds=9,
            )

class TestFinalizeIssRegistrationRealData:
    """End-to-end real-data test for dispatch stage 'finalize_iss_registration'.

    Finalizes registration; waits for all wells
    """

    @pytest.mark.real_data
    def test_finalize_iss_registration_matches_reference(
        self, real_data_workdir, reference_cache, reference_ngff_version, reference_n_rounds
    ):
        """HEAVY GPU test. finalize composes the per-well cumulative transforms
        (round{r}_to_round0_cumulative.yaml -- deterministic, the registration
        correctness artifact) then resamples on GPU into bc_stitched_registered_v3
        (~87G/well). Needs a GPU + OPS_TEST_TMP_BASE on a roomy FS.

        Seed ONLY the SEQUENTIAL input transforms (round{i}_to_round{i-1}.yml,
        nucleus_to_round0.yml, segmentation_to_nucleus.yaml) plus the stitched ISS
        and segmentation stores -- do NOT pre-seed the *_cumulative.yaml outputs,
        which the stage overwrites in place (a read-only symlink would write through
        to the reference). Unlike register_iss_nucleus_to_round0 this is NOT blocked
        on raw NDTiff: it reads the retained nucleus_to_round0.yml, and
        check_for_dapi_round10 globs the (unseeded) convert dir -> False.
        """
        from cyclops_process.processes.auto_register.iss_cycle_register_orchestrator import job_finalize_registration

        from conftest import REFERENCE_EXPERIMENT, seed_from_cache
        from fixtures.compare import compare_ome_zarr, compare_yaml
        from fixtures.slurm import submit_stage

        tdir = "1-preprocess/in_situ_sequencing/register/transforms/A1"
        seed_from_cache(
            real_data_workdir,
            reference_cache,
            [
                '1-preprocess/in_situ_sequencing/stitch/bc_stitched.zarr',
                '1-preprocess/in_situ_sequencing/segmentation/bc_segmentation.zarr',
                f'{tdir}/round*_to_round*.yml',          # sequential inputs (.yml) only
                f'{tdir}/nucleus_to_round0.yml',
                f'{tdir}/segmentation_to_nucleus.yaml',
            ],
        )

        log_dir = real_data_workdir / "submitit_logs" / "finalize_iss_registration"
        submit_stage(
            "finalize_iss_registration",
            job_finalize_registration,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
            well=1,
            n_rounds=reference_n_rounds,
        )

        # 1. Cumulative transforms: every *_cumulative.yaml the reference has must be
        #    written by the candidate and match (deterministic compose; sub-pixel
        #    shifts accumulate over the chain -> small atol). Covers rounds 0..n_rounds
        #    + nucleus + segmentation. The compose loop is inclusive (#114 fix), so the
        #    LAST sequencing round (round{n_rounds}) is composed too -- previously it
        #    was dropped; this now asserts it's present.
        cand_tdir = real_data_workdir / REFERENCE_EXPERIMENT / tdir
        ref_tdir = reference_cache / tdir
        ref_cumulatives = sorted(ref_tdir.glob("*_cumulative.yaml"))
        assert ref_cumulatives, "reference has no cumulative transforms to compare"
        for ref_yaml in ref_cumulatives:
            cand_yaml = cand_tdir / ref_yaml.name
            assert cand_yaml.exists(), f"stage did not write {ref_yaml.name}"
            compare_yaml(cand_yaml, ref_yaml, atol=1e-3)

        # 2. Final registered v3 store: structural only (divergent lineage + GPU/
        #    iterative resample -> pixels won't match; geometry/metadata do).
        rel = "1-preprocess/in_situ_sequencing/register/bc_stitched_registered_v3.zarr"
        candidate = real_data_workdir / REFERENCE_EXPERIMENT / rel
        reference = reference_cache / rel
        compare_ome_zarr(candidate, reference, positions=["A/1/0"], check_data=False)
