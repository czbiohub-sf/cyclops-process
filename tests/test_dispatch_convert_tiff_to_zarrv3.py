"""Tests for dispatch_cli command `convert_tiff_to_zarrv3`.

Maps to cyclops_process.processes.assemble_link.convert.

The function reads OME-TIFFs via TIFFConverter and writes per-round zarrs.
Tests target the explicit error/early branches (missing args, NotImplemented
process) plus DAPI_round10 discovery — leaving the heavy TIFF-conversion
parallel section mocked or untouched.
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


class TestConvertArgValidation:
    def test_missing_experiment_and_dirs_raises_valueerror(self):
        with pytest.raises(ValueError, match="experiment or input/output"):
            mod.convert()

    def test_missing_output_dir_with_no_experiment_raises_valueerror(self, tmp_path):
        with pytest.raises(ValueError, match="experiment or input/output"):
            mod.convert(input_dir=str(tmp_path))

    def test_missing_input_dir_with_no_experiment_raises_valueerror(self, tmp_path):
        with pytest.raises(ValueError, match="experiment or input/output"):
            mod.convert(output_dir=str(tmp_path))


class TestConvertProcessBranches:
    def test_lc_20x_process_raises_notimplemented(self, hermetic_experiment):
        with pytest.raises(NotImplementedError, match="5x conversion"):
            mod.convert(experiment=EXPERIMENT, process="lc_20x")


class TestConvertParallelOrchestration:
    def test_parallel_called_once_per_well(
        self, hermetic_experiment, monkeypatch, tmp_path
    ):
        # Build a fake iss_tif_dir with two "A*" well subdirs, no DAPI_round10.
        # Stub out Parallel/delayed and TIFFConverter so we just verify the
        # function dispatches one job per well and prints the result line.
        from cyclops_utils.data.experiment import OpsDataset

        dataset = OpsDataset(EXPERIMENT)
        iss_tif_dir = dataset.iss_tif_dir
        monkeypatch.setattr(
            type(dataset), "iss_tif_dir", iss_tif_dir, raising=False
        )
        # iss_tif_dir is a real $OPS_INSTRUMENT_ROOT/ path — redirect it onto
        # tmp_path by monkeypatching the attribute lookup on OpsDataset.
        fake_iss_dir = tmp_path / "iss_tif"
        fake_iss_dir.mkdir()
        (fake_iss_dir / "A1").mkdir()
        (fake_iss_dir / "A2").mkdir()
        (fake_iss_dir / "not_a_well").mkdir()  # filtered out (doesn't start with A* dir of letters+digits? actually code only checks startswith("A"))

        # Pre-create the output store parent.
        out_parent = dataset.store_paths["iss"].parent
        out_parent.mkdir(parents=True, exist_ok=True)

        # Monkeypatch the OpsDataset constructor to return a fake whose
        # iss_tif_dir / store_paths point at our tmp_path.
        original_ctor = mod.OpsDataset

        class FakeDataset:
            def __init__(self, experiment):
                real = original_ctor(experiment)
                self.__dict__.update(real.__dict__)
                self.iss_tif_dir = fake_iss_dir

        monkeypatch.setattr(mod, "OpsDataset", FakeDataset)

        # Stub out heavy parallel + TIFF conversion.
        called_with = []

        def fake_parallel(n_jobs):
            def runner(tasks):
                # tasks is a generator of delayed(_convert_single_well)(...)
                # each yields (func, args, kwargs) tuples per joblib.
                results = []
                for task in tasks:
                    # joblib's delayed wraps to (func, args, kwargs). We
                    # exercise it directly by extracting the captured args.
                    func, args, kwargs = task[0], task[1], task[2]
                    called_with.append(args)
                    results.append(f"Completed: {args[0].name}")
                return results
            return runner

        monkeypatch.setattr(mod, "Parallel", fake_parallel)
        # delayed should produce the (func, args, kwargs) shape we expect
        # above. joblib.delayed already returns such a callable wrapper, but
        # to keep this self-contained we replace it.
        monkeypatch.setattr(
            mod, "delayed",
            lambda func: (lambda *a, **kw: (func, a, kw)),
        )
        # Also stub TIFFConverter so importing/inspecting doesn't try to
        # touch real files.
        monkeypatch.setattr(mod, "TIFFConverter", lambda **kw: lambda: None)
        # Force a deterministic worker count.
        monkeypatch.setattr(mod, "get_optimal_workers", lambda **kw: 1)

        mod.convert(experiment=EXPERIMENT, process="iss")

        # Each call_args is (ird, op, overwrite_decision, ngff_version).
        well_names = sorted(a[0].name for a in called_with)
        assert "A1" in well_names
        assert "A2" in well_names
        # "not_a_well" did not start with "A"+letter so won't be filtered out
        # via the codepath; it does start with 'n', so it IS excluded by the
        # `name.startswith("A")` check.
        assert "not_a_well" not in well_names


class TestConvertRealData:
    """End-to-end real-data test: runs convert on the ops0094_20251217 TIFFs
    via SLURM and compares the produced zarrs against the reference cache.

    Skipped unless the reference cache is reachable. Run with
    `pytest -m real_data`.
    """

    @pytest.mark.real_data
    def test_convert_matches_reference(self, real_data_workdir, reference_cache):
        from cyclops_process.processes.assemble_link import convert

        from conftest import REFERENCE_EXPERIMENT
        from fixtures.compare import compare_ome_zarr_directory
        from fixtures.slurm import submit_stage

        log_dir = real_data_workdir / "submitit_logs" / "convert"
        submit_stage(
            "convert_tiff_to_zarrv3",
            convert,
            log_dir,
            experiment=REFERENCE_EXPERIMENT,
        )

        candidate = (
            real_data_workdir
            / REFERENCE_EXPERIMENT
            / "0-convert"
            / "in_situ_sequencing"
        )
        reference = reference_cache / "0-convert" / "in_situ_sequencing"
        compare_ome_zarr_directory(
            candidate,
            reference,
            pattern="A*.zarr",
            rtol=1e-5,
        )
