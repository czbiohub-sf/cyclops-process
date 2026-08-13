"""
Virtual staining pipeline (Viscy): preprocessing, inference via SLURM, and combine.

CLI (when run as __main__)
--------------------------
  python -m cyclops_process.processes.virtual_staining <command> [options]

Commands
--------
  preprocess     Run only preprocessing (normalization metadata generation). ~15-45 min.
  inference      Run only inference (SLURM array jobs) then combine. Blocks until complete.
  all            Run preprocess then inference for the given process (full pipeline).
  combine_only   Run only the combine job (merge intermediate tiles into final zarr).
                 Use when inference completed but the combine job failed or never ran.

Usage examples
--------------
  # Full pipeline (preprocess + inference) for pheno
  python -m cyclops_process.processes.virtual_staining all -e 117 -p pheno

  # Preprocess only (e.g. to regenerate normalization)
  python -m cyclops_process.processes.virtual_staining preprocess -e 117 -p pheno --force

  # Inference only (requires preprocess to have been run first)
  python -m cyclops_process.processes.virtual_staining inference -e 117 -p pheno

  # Rerun combine only (block until job completes)
  python -m cyclops_process.processes.virtual_staining combine_only -e 117 -p pheno --no-wait

  # Help
  python -m cyclops_process.processes.virtual_staining --help
  python -m cyclops_process.processes.virtual_staining all --help

Common options (preprocess, inference, all, combine_only)
----------------------------------------------------------
  -e, --experiment   Experiment name or shorthand (e.g. 117, ops0117). Resolved via configs.
  -p, --process      track (5x) or pheno (20x). Required.
  -d, --dim          2d or 3d (default: 3d).

Extra options
-------------
  preprocess:    --force  Force regeneration even if normalization exists.
                 --num-workers N  (default: 16)
  inference:     --num-array-jobs N  Override auto (e.g. pheno=10, track=1).
                 --gpus-per-job N  (default: 1)
  combine_only:  --no-wait  Submit combine job and exit without waiting.
"""
import subprocess
import sys
import yaml
from pathlib import Path
import shutil
import os
from ops_utils.data.experiment import OpsDataset


def _viscy_bin() -> str:
    """Resolve the `viscy` executable from the running Python's venv.

    SLURM child shells don't have the venv's bin on PATH, so bare-name
    `subprocess.run(["viscy", ...])` raises FileNotFoundError. Derive
    the absolute path from sys.executable, fall back to "viscy" only
    if the venv layout doesn't match.
    """
    candidate = Path(sys.executable).parent / "viscy"
    if candidate.is_file():
        return str(candidate)
    return "viscy"
from iohub import open_ome_zarr
from ops_utils.profiling.decorators import versioned_function
from ops_utils.data.filesystem import decide_overwrite_resume_skip, resolve_experiment_name
from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
from cyclops_process.utils.combine_batch_predictions import (
    combine_predictions,
    create_combine_output_store,
    combine_single_batch_store,
    validate_combine_output,
    discover_prediction_stores,
)
import json


@versioned_function("v1.0")
def virtual_staining(
    experiment: str, process: str, dim: str = "3d", debug: bool = False
) -> None:
    """
    Wrapper for the Viscy virtual staining CLI
    """

    dataset = OpsDataset(experiment)

    dim_norm = (dim or "").lower()
    if dim_norm not in {"3d", "2d"}:
        raise ValueError("dim must be one of {'3d','2d'}")

    print(f"Running virtual staining for {process} on {dim} phase reconstruction")

    if process == "track":
        # wrtie a normalization config yaml
        # Select 3D or 2D reconstruction for tracking.
        # For 2D, only use the first channel (exclude focus channel).
        if dim_norm == "3d":
            input_path = dataset.store_paths["lc_5x_phase_3d_optimized"]
            vs_config_path = dataset.config_paths["lc_5x_vs_config"]
            channel_names = ["Phase3D"]
        else:
            # 2D recon path
            input_path = dataset.store_paths["lc_5x_phase_2d_optimized"]
            # Use the 2D model config
            vs_config_path = dataset.config_paths["lc_5x_vs_config_2d"]
            channel_names = ["Phase2D"]  # first channel only

        normalization_config = {
            "data_path": f"{input_path}",
            "channel_names": channel_names,
            "num_workers": 32,
            "block_size": 32,
        }
        norm_config_path = dataset.config_paths["lc_5x_vs_norm"]
        # if dim_norm == "2d":
        #     vs_output_path = dataset.store_paths["lc_5x_vs_2d"]
        # else:
        #
        vs_output_path = dataset.store_paths[
            "lc_5x_vs"
        ]  # NOTE: 2D will go to orginal 3D store
        vs_intermediate_path = vs_output_path.with_suffix("")
        jobs_meta_path = dataset.config_paths["vs_jobs_track"]

    elif process == "pheno":
        # Select 3D or 2D reconstruction for phenotyping.
        # For 2D, only use the first channel (exclude focus channel).
        if dim_norm == "3d":
            input_path = dataset.store_paths["lc_20x_phase_3d_optimized"]
            vs_config_path = dataset.config_paths["lc_20x_vs_config"]
            channel_names = ["Phase3D"]
        else:
            # 2D recon path
            input_path = dataset.store_paths["lc_20x_phase_2d_optimized"]
            # Use the 2D model config
            vs_config_path = dataset.config_paths["lc_20x_vs_config_2d"]
            channel_names = ["Phase2D"]  # first channel only

        normalization_config = {
            "data_path": f"{input_path}",
            "channel_names": channel_names,
            "num_workers": 32,
            "block_size": 32,
        }
        norm_config_path = dataset.config_paths["lc_20x_vs_norm"]
        # if dim_norm == "2d":
        #     vs_output_path = dataset.store_paths["lc_20x_vs_2d"]
        # else:
        #     vs_output_path = dataset.store_paths["lc_20x_vs"]
        vs_output_path = dataset.store_paths[
            "lc_20x_vs"
        ]  # NOTE: 2D will go to orginal 3D store
        vs_intermediate_path = vs_output_path.with_suffix("")
        jobs_meta_path = dataset.config_paths["vs_jobs_pheno"]
    else:
        raise ValueError(f"Unknown process: {process}. Must be 'track' or 'pheno'")

    input_store = open_ome_zarr(input_path)
    positions = [a[0] for a in input_store.positions()]
    num_positions = len(positions)

    # Check if output paths exist and handle overwrite/resume/skip
    decision_output = decide_overwrite_resume_skip(
        vs_output_path, is_debug=debug, expected_positions=positions
    )
    
    # We do NOT use decide_overwrite_resume_skip for the intermediate path because:
    # 1. It enforces strict structural completeness (is_precreated_store), which fails for 
    #    a working directory containing a partial set of individual tile Zarrs.
    # 2. It auto-deletes "incomplete" stores, which wipes out our valid partial work.
    # Instead, we derive the intermediate handling from the output decision.

    if decision_output == "skip":
        print(f"Skipping virtual staining for {process} (user requested skip)")
        return

    # Handle overwrite for output store (treat "resume" as "overwrite" for VS)
    if (decision_output in ["overwrite", "resume"]) and vs_output_path.exists():
        _fast_remove_directory(vs_output_path, "output store")

    # Handle overwrite for intermediate directory (treat "resume" as "overwrite" for VS)
    if (
        decision_intermediate in ["overwrite", "resume"]
    ) and vs_intermediate_path.exists():
        _fast_remove_directory(vs_intermediate_path, "intermediate directory")

    # generate new normalization config yml
    if indices_to_run and (not norm_config_path.exists() or decision_output == "overwrite"):
        print(f"Generating normalization config at {norm_config_path}")
        with open(norm_config_path, "w") as f:
            yaml.dump(normalization_config, f)
        
        # run normalization
        print("Running normalization...")
        normalization_config["_norm_config_path"] = norm_config_path
        _run_fast_normalization(
            normalization_config["data_path"], normalization_config,
            normalization_config.get("num_workers", 16),
        )
    elif indices_to_run:
        print("Normalization config exists, skipping normalization.")
    else:
        print("All tiles complete, skipping normalization check.")

    job_id = None
    
    # Create log directory for this experiment
    log_dir = Path(f"slurm_logs/slurm_virtual_staining_logs/{experiment}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if not indices_to_run:
        print("All tiles appear complete. Skipping prediction jobs.")
    else:
        print(f"Submitting jobs for {len(indices_to_run)} missing/incomplete tiles.")
        
        # Format array string
        if len(indices_to_run) == num_positions:
            array_str = f"0-{num_positions - 1}"
        else:
            # Create concise array string (e.g. "1,3,5-10")
            indices_to_run.sort()
            
            # Helper to create ranges from indices
            ranges_tuples = []  # list of [start, end]
            start = indices_to_run[0]
            end = start
            for idx in indices_to_run[1:]:
                if idx == end + 1:
                    end = idx
                else:
                    ranges_tuples.append([start, end])
                    start = idx
                    end = idx
            ranges_tuples.append([start, end])

            def _to_slurm_str(rtuples):
                parts = []
                for s, e in rtuples:
                    if s == e:
                        parts.append(str(s))
                    else:
                        parts.append(f"{s}-{e}")
                return ",".join(parts)

            array_str = _to_slurm_str(ranges_tuples)
            
            # If the string is too long for Slurm/Shell, we simplify by merging 
            # small gaps (effectively running some completed tiles again) until it fits.
            # Conservative limit.
            MAX_ARG_LEN = 4000 
            
            if len(array_str) > MAX_ARG_LEN:
                print(f"Array string length {len(array_str)} exceeds limit. Merging gaps to simplify...")
                while len(array_str) > MAX_ARG_LEN and len(ranges_tuples) > 1:
                    min_gap = float('inf')
                    merge_idx = -1
                    for i in range(len(ranges_tuples) - 1):
                        gap = ranges_tuples[i+1][0] - ranges_tuples[i][1]
                        if gap < min_gap:
                            min_gap = gap
                            merge_idx = i
                    
                    if merge_idx != -1:
                        # Merge the two ranges: [s1, e1] and [s2, e2] -> [s1, e2]
                        s_new = ranges_tuples[merge_idx][0]
                        e_new = ranges_tuples[merge_idx+1][1]
                        ranges_tuples[merge_idx : merge_idx+2] = [[s_new, e_new]]
                        array_str = _to_slurm_str(ranges_tuples)
                    else:
                        break
                print(f"Simplified array string length: {len(array_str)}")

        # Log paths for array job: %A = array job ID, %a = array task ID
        # Each array job gets its own subfolder named by %A
        vs_log_out = log_dir / "%A" / f"vs_{process}_%A_%a.out"
        vs_log_err = log_dir / "%A" / f"vs_{process}_%A_%a.err"
        
        command_vs = [
            "sbatch",
            "--parsable",
            f"--array={array_str}%100",
            f"--output={vs_log_out}",
            f"--error={vs_log_err}",
            f"{dataset.config_paths['vs_helper']}",
            f"{input_path}",
            f"{vs_intermediate_path}",
            f"{vs_config_path}",
        ]
        result_vs = subprocess.run(command_vs, capture_output=True, text=True, check=True)
        job_id = result_vs.stdout.strip()
        print(f"Submitted VS array job: {job_id}")
        print(f"  Logs: {log_dir}/{job_id}/")

        # Persist Slurm array job IDs for the runner to monitor
        try:
            jobs_meta_path.parent.mkdir(parents=True, exist_ok=True)
            with open(jobs_meta_path, "w") as jf:
                yaml.safe_dump(
                    {
                        "array_job_id": job_id,
                        "num_positions": int(num_positions),
                        "input_store": str(input_path),
                        "output_store": str(vs_output_path),
                    },
                    jf,
                )
        except Exception:
            pass

    # Use afterany if we have jobs running, so "already exists" errors don't block the combine
    # afterany means: run after all array tasks finish, regardless of their exit codes
    dependency_type = "afterany" if job_id else None
    
    # Log paths for combine job: %j = job ID
    # Each combine job gets its own subfolder named by %j
    combine_log_out = log_dir / "%j" / f"combine_{process}_%j.out"
    combine_log_err = log_dir / "%j" / f"combine_{process}_%j.err"
    
    combine_command = [
        "sbatch",
        "--parsable",
        f"--output={combine_log_out}",
        f"--error={combine_log_err}",
    ]
    if dependency_type and job_id:
        combine_command.append(f"--dependency={dependency_type}:{job_id}")
        
    combine_command.extend([
        f"{dataset.config_paths['vs_combine_script']}",
        f"{vs_intermediate_path}",
        f"{input_path}",
        f"{vs_output_path}",
    ])
    
    result_combine = subprocess.run(
        combine_command, capture_output=True, text=True, check=True
    )
    combine_job_id = result_combine.stdout.strip()
    print(f"Submitted combine job (afterany): {combine_job_id}")
    print(f"  Logs: {log_dir}/{combine_job_id}/")

    # Append combine job id as well
    try:
        if jobs_meta_path.exists():
            with open(jobs_meta_path, "r") as jf:
                meta = yaml.safe_load(jf) or {}
            meta["combine_job_id"] = result_combine.stdout.strip()
            with open(jobs_meta_path, "w") as jf:
                yaml.safe_dump(meta, jf)
    except Exception:
        pass

    return


def _create_output_store_with_metadata(input_path: Path, output_path: Path) -> None:
    """
    Pre-create output zarr store with proper OME-Zarr metadata.

    This creates an empty store structure matching the input store,
    allowing viscy's HCSPredictionWriter to write directly to it.

    Args:
        input_path: Input zarr store path
        output_path: Output zarr store path to create
    """
    import zarr

    print(f"Creating output store: {output_path}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy plate-level metadata from input (v3: zarr.json, v2: .zattrs/.zgroup)
    for fname in ("zarr.json", ".zattrs", ".zgroup"):
        src = input_path / fname
        if src.exists():
            shutil.copy(src, output_path / fname)

    # Iterate through rows and wells using filesystem
    for row_dir in sorted(input_path.iterdir()):
        if not row_dir.is_dir() or row_dir.name.startswith('.'):
            continue

        row_name = row_dir.name
        row_path = output_path / row_name
        row_path.mkdir(exist_ok=True)

        # Copy row metadata
        for fname in ("zarr.json", ".zgroup"):
            src = input_path / row_name / fname
            if src.exists():
                shutil.copy(src, row_path / fname)

        # Create well directories
        for well_dir in sorted(row_dir.iterdir()):
            if not well_dir.is_dir() or well_dir.name.startswith('.'):
                continue

            well_name = well_dir.name
            well_path = row_path / well_name
            well_path.mkdir(exist_ok=True)

            # Copy well metadata
            for fname in ("zarr.json", ".zattrs", ".zgroup"):
                src = input_path / row_name / well_name / fname
                if src.exists():
                    shutil.copy(src, well_path / fname)

    print(f"Output store structure created successfully")


def _fast_remove_directory(path: Path, description: str = "directory") -> None:
    """
    Quickly remove a directory by moving it and spawning a background deletion job.

    This is much faster than shutil.rmtree() for large zarr stores.

    Args:
        path: Directory to remove
        description: Description for logging
    """
    from ops_utils.data.filesystem import async_delete_path

    print(f"Fast-removing {description}: {path}")
    trash = async_delete_path(path)
    if trash is not None:
        print(f"  Moved to {trash.name}; background deletion job spawned.")


def _check_normalization_completeness(norm_config_path: Path) -> tuple[bool, int, int]:
    """
    Check if normalization metadata exists for ALL positions.

    Viscy preprocess stores normalization stats in each FOV's .zattrs under
    the "normalization" key. We sample positions to verify completeness.

    Returns:
        (is_complete, num_with_norm, total_positions)
    """
    if not norm_config_path.exists():
        return False, 0, 0

    try:
        with open(norm_config_path, 'r') as f:
            config = yaml.safe_load(f)

        if not config or 'data_path' not in config:
            return False, 0, 0

        data_path = Path(config['data_path'])
        if not data_path.exists():
            return False, 0, 0

        # Collect all FOV directories
        fov_dirs = []
        for row_dir in data_path.iterdir():
            if not row_dir.is_dir() or row_dir.name.startswith('.'):
                continue
            for well_dir in row_dir.iterdir():
                if not well_dir.is_dir() or well_dir.name.startswith('.'):
                    continue
                for fov_dir in well_dir.iterdir():
                    if not fov_dir.is_dir() or fov_dir.name.startswith('.'):
                        continue
                    fov_dirs.append(fov_dir)

                    zarr_json_path = fov_dir / "zarr.json"
                    zattrs_path    = fov_dir / ".zattrs"
                    if zarr_json_path.exists():
                        meta_path, use_v3 = zarr_json_path, True
                    elif zattrs_path.exists():
                        meta_path, use_v3 = zattrs_path, False
                    else:
                        continue

                    try:
                        with open(meta_path, 'r') as f:
                            raw = json.load(f)
                        attrs = raw.get("attributes", raw) if use_v3 else raw
                        if "normalization" in attrs:
                            # v3 fast-path: store carries normalization metadata
                            # -> treat as complete. (Return the declared 3-tuple,
                            # not a bare bool, or the caller's unpack throws.)
                            n = len(fov_dirs)
                            return True, n, n
                    except Exception:
                        pass

        total = len(fov_dirs)
        # Sample up to 200 positions evenly spread across the store
        # (checking all can be slow for 7000+ positions)
        import random
        if total <= 200:
            sample = fov_dirs
        else:
            # Always include first, last, and evenly spaced positions
            step = max(1, total // 198)
            sample = fov_dirs[::step]
            # Ensure last position is included (often the one that's missing)
            if fov_dirs[-1] not in sample:
                sample.append(fov_dirs[-1])

        has_norm = 0
        for fov_dir in sample:
            zattrs_path = fov_dir / ".zattrs"
            if not zattrs_path.exists():
                continue
            try:
                with open(zattrs_path, 'r') as f:
                    attrs = json.load(f)
                if "normalization" in attrs:
                    has_norm += 1
            except Exception:
                pass

        # Extrapolate: if all sampled have it, assume complete
        # If any sampled are missing, it's incomplete
        is_complete = (has_norm == len(sample))
        # Scale has_norm to estimate total
        estimated_with_norm = int(has_norm / len(sample) * total) if sample else 0
        return is_complete, estimated_with_norm, total

    except Exception:
        pass

    return False, 0, 0


def _run_viscy_predict_batch(position_batch: list, output_dir: str, config_path: str, work_dir: str, batch_id: int) -> list:
    """
    Run viscy predict for a batch of positions (keeps model loaded).

    This processes multiple positions with a single model load, avoiding
    the overhead of loading/unloading the 8GB model for each position.

    Args:
        position_batch: List of (idx, position_path) tuples
        output_dir: Directory to save output zarrs
        config_path: Path to the viscy model config
        work_dir: Working directory for logs
        batch_id: Batch identifier for logging

    Returns:
        List of (idx, success) tuples
    """
    import subprocess
    from pathlib import Path
    import uuid

    results = []

    for position_idx, position_path in position_batch:
        # Create unique log directory for this position
        unique_id = str(uuid.uuid4())[:8]
        log_dir = Path(work_dir) / "logs" / f"position_{position_idx}_{unique_id}"
        log_dir.mkdir(parents=True, exist_ok=True)

        output_path = f"{output_dir}/{position_idx}.zarr"

        # Run viscy predict
        command = [
            _viscy_bin(), "predict",
            "--config", str(config_path),
            "--data.data_path", str(position_path),
            "--trainer.callbacks+=viscy.translation.predict_writer.HCSPredictionWriter",
            f"--trainer.callbacks.output_store={output_path}",
            f"--trainer.default_root_dir={log_dir}",
            "--trainer.logger=False",
        ]

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            results.append((position_idx, True))
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] viscy predict failed for position {position_idx}")
            print(f"Command: {' '.join(command)}")
            if e.stderr:
                print(f"STDERR:\n{e.stderr}")
            results.append((position_idx, False))
            raise

    return results


# ── Submitit wrappers for viscy inference (used by submit_parallel_jobs) ──────


_VISCY_PYTHON = sys.executable


@versioned_function("v1.0")
def _run_viscy_multigpu_inference(
    experiment: str,
    process: str,
    input_store: str,
    output_dir: str,
    config_path: str,
    num_positions: int,
    num_gpus: int,
    batch_size: int = 7,
    num_workers: int = 8,
) -> None:
    """
    Submitit wrapper: run multi-GPU batch inference for all positions.

    Calls the viscy_multigpu_inference.py script in the viscy conda env.
    """
    import subprocess
    from pathlib import Path

    script = str(
        Path(__file__).parent.parent.parent
        / "nextflow"
        / "bin"
        / "viscy_multigpu_inference.py"
    )

    command = [
        _VISCY_PYTHON,
        script,
        "--input-store", str(input_store),
        "--output-dir", str(output_dir),
        "--config", str(config_path),
        "--start-pos", "0",
        "--end-pos", str(num_positions),
        "--num-gpus", str(num_gpus),
        "--batch-size", str(batch_size),
        "--num-workers", str(num_workers),
    ]

    print(f"Running multi-GPU inference ({num_gpus} GPUs, {num_positions} positions)")
    result = subprocess.run(command, check=True)


@versioned_function("v1.0")
def _run_viscy_array_inference(
    experiment: str,
    process: str,
    input_store: str,
    output_dir: str,
    config_path: str,
    start_pos: int,
    end_pos: int,
    batch_size: int = 7,
    num_workers: int = 8,
) -> None:
    """
    Submitit wrapper: run batch inference for a range of positions.

    Calls the viscy_batch_inference.py script in the viscy conda env.
    Each submitit array task processes [start_pos, end_pos) positions.
    """
    import subprocess
    from pathlib import Path

    script = str(Path(__file__).parent / "viscy_batch_inference.py")

    num_positions = end_pos - start_pos
    if num_positions <= 0:
        print(f"No positions to process (start={start_pos} >= end={end_pos})")
        return

    command = [
        _VISCY_PYTHON,
        script,
        "--input-store", str(input_store),
        "--output-dir", str(output_dir),
        "--config", str(config_path),
        "--start-pos", str(start_pos),
        "--end-pos", str(end_pos),
        "--batch-size", str(batch_size),
        "--num-workers", str(num_workers),
    ]

    print(f"Running batch inference for positions {start_pos}-{end_pos} ({num_positions} positions)")
    result = subprocess.run(command, check=True)


def _run_viscy_predict_single_position(position_path: str, output_path: str, config_path: str, position_idx: int, work_dir: str) -> None:
    """
    Run viscy predict for a single position.

    This function mimics what predict_slurm.sh does for array jobs.

    Args:
        position_path: Path to the position in the input zarr store
        output_path: Path to save the output zarr for this position
        config_path: Path to the viscy model config
        position_idx: Position index (for logging)
        work_dir: Working directory (for creating unique log dirs)
    """
    import subprocess
    from pathlib import Path
    import os
    import uuid

    # Create unique log directory for this position to avoid conflicts
    # Add UUID to ensure uniqueness even if positions run at same second
    unique_id = str(uuid.uuid4())[:8]
    log_dir = Path(work_dir) / "logs" / f"position_{position_idx}_{unique_id}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Run viscy predict
    # Disable logger to prevent config file collision between parallel workers
    command = [
        _viscy_bin(), "predict",
        "--config", str(config_path),
        "--data.data_path", str(position_path),
        "--trainer.callbacks+=viscy.translation.predict_writer.HCSPredictionWriter",
        f"--trainer.callbacks.output_store={output_path}",
        f"--trainer.default_root_dir={log_dir}",
        "--trainer.logger=False",  # Disable logger to prevent config save conflicts
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        # Print stderr/stdout for debugging
        print(f"\n[ERROR] viscy predict failed for position {position_idx}")
        print(f"Command: {' '.join(command)}")
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            print(f"STDERR:\n{e.stderr}")
        raise
    return


def _run_fast_normalization(input_path, normalization_config, num_workers):
    """Run viscy preprocess as subprocess with timeout handling.

    The viscy subprocess hangs indefinitely on zarr store cleanup after
    completing all work. We just run it with subprocess.run and a generous
    timeout — if it times out, the work is already done.
    """
    import subprocess
    from pathlib import Path

    norm_config_path = normalization_config.get("_norm_config_path")
    if norm_config_path is None:
        import yaml
        norm_config_path = Path(input_path).parent / f".vs_norm_tmp_{Path(input_path).stem}.yml"
        with open(norm_config_path, "w") as f:
            yaml.dump({
                "data_path": str(input_path),
                "channel_names": normalization_config["channel_names"],
                "num_workers": num_workers,
                "block_size": normalization_config.get("block_size", 32),
            }, f)

    command = [_viscy_bin(), "preprocess", "-c", str(norm_config_path)]
    try:
        subprocess.run(command, check=True, timeout=1800)
    except subprocess.TimeoutExpired:
        print("  viscy preprocess timed out after 30 min")

    # viscy writes normalization to the top-level `normalization` field (its
    # own reader expects it there). The v3 schema established by convert_v3.py
    # also exposes it under position-level `custom_metadata.normalization`,
    # which portal/napari/audit readers depend on. Mirror it there so the
    # v3-native store complies without disturbing viscy's top-level copy.
    _mirror_normalization_to_custom_metadata(input_path)


def _mirror_normalization_to_custom_metadata(input_path):
    """Copy each position's top-level `normalization` into custom_metadata.

    Matches the layout convert_v3.py produced for v2->v3 stores. Direct
    re-assignment (not in-place mutation) is required so the write persists to
    the v3 ``zarr.json`` attributes.
    """
    from iohub import open_ome_zarr

    try:
        with open_ome_zarr(input_path, layout="hcs", mode="r+") as plate:
            mirrored = 0
            for _, pos in plate.positions():
                norm = pos.zattrs.get("normalization")
                if not norm:
                    continue
                custom = dict(pos.zattrs.get("custom_metadata", {}))
                custom["normalization"] = norm
                pos.zattrs["custom_metadata"] = custom
                mirrored += 1
        print(f"  mirrored normalization -> custom_metadata for {mirrored} position(s)")
    except Exception as e:
        print(f"  ⚠ Could not mirror normalization into custom_metadata: {e}")


@versioned_function("v1.0")
def virtual_staining_preprocess(
    experiment: str,
    process: str,
    dim: str = "3d",
    debug: bool = False,
    force: bool = False,
    num_workers: int = None,
) -> None:
    """
    Run only the preprocessing step (normalization metadata generation).

    This step analyzes the input data to compute normalization statistics
    and creates metadata used by the inference step. Takes ~15-45 minutes.

    Args:
        experiment: Experiment name
        process: 'track' or 'pheno'
        dim: '2d' or '3d'
        debug: Debug mode
        force: Force regeneration even if metadata exists
        num_workers: Number of workers for preprocessing (auto-detected if None)
    """
    if num_workers is None:
        from ops_utils.hpc.resource_manager import get_optimal_workers
        num_workers = get_optimal_workers(use_gpu=False, model_ram_gb=0.5, data_ram_gb=1.0)

    dataset = OpsDataset(experiment)

    dim_norm = (dim or "").lower()
    if dim_norm not in {"3d", "2d"}:
        raise ValueError("dim must be one of {'3d','2d'}")

    print(f"Running virtual staining PREPROCESSING for {process} on {dim} phase reconstruction")

    # Setup paths based on process and dimension
    if process == "track":
        if dim_norm == "3d":
            input_path = dataset.store_paths["lc_5x_phase_3d_optimized"]
            channel_names = ["Phase3D"]
        else:
            input_path = dataset.store_paths["lc_5x_phase_2d_optimized"]
            channel_names = ["Phase2D"]

        norm_config_path = dataset.config_paths["lc_5x_vs_norm"]

    elif process == "pheno":
        if dim_norm == "3d":
            input_path = dataset.store_paths["lc_20x_phase_3d_optimized"]
            channel_names = ["Phase3D"]
        else:
            input_path = dataset.store_paths["lc_20x_phase_2d_optimized"]
            channel_names = ["Phase2D"]

        norm_config_path = dataset.config_paths["lc_20x_vs_norm"]
    else:
        raise ValueError(f"Unknown process: {process}. Must be 'track' or 'pheno'")

    # Check if normalization already exists (unless force=True)
    is_complete, num_with_norm, total = _check_normalization_completeness(norm_config_path)
    if not force and is_complete:
        print(f"Normalization metadata already exists for all {total} positions at {norm_config_path}")
        print(f"Skipping preprocessing step. Use force=True to regenerate.")
        return

    # Generate normalization config
    normalization_config = {
        "data_path": f"{input_path}",
        "channel_names": channel_names,
        "num_workers": num_workers,
        "block_size": 32,
    }

    # Write normalization config
    norm_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(norm_config_path, "w") as f:
        yaml.dump(normalization_config, f)

    print(f"Running viscy preprocess with {num_workers} workers...")
    print(f"  Input: {input_path}")
    print(f"  Config: {norm_config_path}")

    # Check if profiling is enabled
    enable_profiling = os.environ.get('ENABLE_PROFILING', '0') == '1'

    # Run normalization (this is the time-consuming step ~15-45 min)
    import time
    start_time = time.perf_counter()

    normalization_config["_norm_config_path"] = norm_config_path
    _run_fast_normalization(input_path, normalization_config, num_workers)

    elapsed = time.perf_counter() - start_time
    print(f"Preprocessing complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Normalization metadata saved to {norm_config_path}")
    return


@versioned_function("v1.1")
def virtual_staining_inference(
    experiment: str,
    process: str,
    dim: str = "3d",
    debug: bool = False,
    num_workers: int = 8,  # DataLoader workers
    num_array_jobs: int = None,  # Auto-determined based on process
    gpus_per_job: int = 1  # GPUs per array job (1 is optimal with batch inference)
) -> None:
    """
    Run inference step via SLURM array jobs with batch inference.

    Uses optimized batch inference that:
    - Loads model once per job (not per position)
    - Uses async writing to overlap I/O with GPU compute
    - Achieves ~1.1s per position (vs ~3.9s with per-position CLI)

    Default array job counts:
    - pheno: 10 jobs (~700 positions each, ~13 min per job)
    - track: 1 job (~300 positions, ~5 min total)

    Args:
        experiment: Experiment name
        process: 'track' or 'pheno'
        dim: '2d' or '3d'
        debug: Debug mode
        num_workers: DataLoader workers (default: 8)
        num_array_jobs: Number of parallel array jobs (auto if None)
        gpus_per_job: GPUs per job (default: 1, optimal for batch inference)
    """
    dataset = OpsDataset(experiment)

    dim_norm = (dim or "").lower()
    if dim_norm not in {"3d", "2d"}:
        raise ValueError("dim must be one of {'3d','2d'}")

    print(f"Running virtual staining INFERENCE for {process} on {dim} phase reconstruction")

    # Setup paths based on process and dimension
    if process == "track":
        if dim_norm == "3d":
            input_path = dataset.store_paths["lc_5x_phase_3d_optimized"]
            vs_config_path = dataset.config_paths["lc_5x_vs_config"]
        else:
            input_path = dataset.store_paths["lc_5x_phase_2d_optimized"]
            vs_config_path = dataset.config_paths["lc_5x_vs_config_2d"]

        norm_config_path = dataset.config_paths["lc_5x_vs_norm"]
        vs_output_path = dataset.store_paths["lc_5x_vs"]
        vs_intermediate_path = vs_output_path.with_suffix("")

    elif process == "pheno":
        if dim_norm == "3d":
            input_path = dataset.store_paths["lc_20x_phase_3d_optimized"]
            vs_config_path = dataset.config_paths["lc_20x_vs_config"]
        else:
            input_path = dataset.store_paths["lc_20x_phase_2d_optimized"]
            vs_config_path = dataset.config_paths["lc_20x_vs_config_2d"]

        norm_config_path = dataset.config_paths["lc_20x_vs_norm"]
        vs_output_path = dataset.store_paths["lc_20x_vs"]
        vs_intermediate_path = vs_output_path.with_suffix("")
    else:
        raise ValueError(f"Unknown process: {process}. Must be 'track' or 'pheno'")

    # Check that preprocessing was run AND completed for all positions
    is_complete, num_with_norm, total = _check_normalization_completeness(norm_config_path)

    if not is_complete:
        if total == 0:
            raise RuntimeError(
                f"Normalization metadata not found at {norm_config_path}. "
                f"You must run virtual_staining_preprocess_{process} first!"
            )

        # Partial normalization — preprocess was interrupted
        print(f"\n⚠️  WARNING: Normalization metadata is INCOMPLETE!")
        print(f"  Only ~{num_with_norm}/{total} positions have normalization metadata.")
        print(f"  This means the preprocessing step was interrupted or timed out.")
        print(f"  Re-running preprocessing automatically to fill missing positions...\n")

        virtual_staining_preprocess(
            experiment=experiment, process=process, dim=dim,
            debug=debug, force=True,
        )

        # Verify it worked
        is_complete, num_with_norm, total = _check_normalization_completeness(norm_config_path)
        if not is_complete:
            raise RuntimeError(
                f"Normalization metadata is STILL incomplete after re-running preprocessing! "
                f"Only ~{num_with_norm}/{total} positions have normalization. "
                f"Check the preprocessing step for errors — the input store may have "
                f"corrupt or unreadable positions."
            )

    # Always overwrite VS output and intermediates to avoid stale partial data
    if vs_output_path.exists():
        _fast_remove_directory(vs_output_path, "final output store")
    if vs_intermediate_path.exists():
        _fast_remove_directory(vs_intermediate_path, "intermediate shards")

    # Get position count efficiently via filesystem
    import subprocess
    result = subprocess.run(
        ["find", str(input_path), "-mindepth", "3", "-maxdepth", "3", "-type", "d"],
        capture_output=True, text=True, check=True
    )
    num_positions = len(result.stdout.strip().split('\n'))
    print(f"Found {num_positions} positions via filesystem scan")

    # Auto-determine array job count based on process and position count
    if num_array_jobs is None:
        if process == "track":
            # Track has ~300 positions, single job is fine (~5 min)
            num_array_jobs = 1
        else:
            # Pheno has ~7000 positions, use 10 jobs (~700 each, ~13 min per job)
            num_array_jobs = 10

    # Create intermediate directory
    vs_intermediate_path.mkdir(parents=True, exist_ok=True)

    # Absolute log directory for all inference/combine logs
    log_dir = Path.cwd() / f"slurm_logs/slurm_virtual_staining_logs/{experiment}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Estimate runtime: ~1.1s per position with batch inference + async writing
    positions_per_job = (num_positions + num_array_jobs - 1) // num_array_jobs
    estimated_minutes_per_job = positions_per_job * 1.1 / 60

    # Adjust time estimate for multi-GPU
    if gpus_per_job > 1:
        # Multi-GPU scales at ~85% efficiency
        estimated_minutes_per_job = estimated_minutes_per_job / (gpus_per_job * 0.85)

    # Calculate time limit with 50% buffer + model loading overhead
    time_limit_minutes = int(estimated_minutes_per_job * 1.5) + 30

    print(f"\n--- Submitting {num_array_jobs} job(s) with {gpus_per_job} GPU(s) each ---")
    print(f"  Total GPUs: {num_array_jobs * gpus_per_job}")
    print(f"  Positions: {num_positions}")
    print(f"  Positions per job: ~{positions_per_job}")
    print(f"  Estimated time per job: ~{estimated_minutes_per_job:.0f} min")
    print(f"  Time limit: {time_limit_minutes} min")
    print(f"  Input: {input_path}")
    print(f"  Output: {vs_output_path}")

    # Build job list for submit_parallel_jobs
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

    jobs_to_submit = []

    if num_array_jobs == 1:
        # Single job processes all positions (optionally multi-GPU)
        jobs_to_submit.append({
            "name": f"vs_inference_{process}_all",
            "func": _run_viscy_multigpu_inference,
            "kwargs": {
                "experiment": experiment,
                "process": process,
                "input_store": str(input_path),
                "output_dir": str(vs_intermediate_path),
                "config_path": str(vs_config_path),
                "num_positions": num_positions,
                "num_gpus": gpus_per_job,
                "batch_size": 7,
                "num_workers": num_workers,
            },
            "metadata": {
                "experiment": experiment,
                "process": process,
                "positions": f"0-{num_positions}",
            },
        })
    else:
        # Multiple array jobs, each processing a range of positions
        for job_idx in range(num_array_jobs):
            start_pos = job_idx * positions_per_job
            end_pos = min(start_pos + positions_per_job, num_positions)
            if start_pos >= num_positions:
                break
            jobs_to_submit.append({
                "name": f"vs_inference_{process}_{job_idx} (pos {start_pos}-{end_pos})",
                "func": _run_viscy_array_inference,
                "kwargs": {
                    "experiment": experiment,
                    "process": process,
                    "input_store": str(input_path),
                    "output_dir": str(vs_intermediate_path),
                    "config_path": str(vs_config_path),
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "batch_size": 7,
                    "num_workers": num_workers,
                },
                "metadata": {
                    "experiment": experiment,
                    "process": process,
                    "positions": f"{start_pos}-{end_pos}",
                },
            })

    # Match original sbatch params: single-job scales CPU/mem by gpus_per_job,
    # multi-job uses fixed 16 CPUs / 64G (always 1 GPU per job in practice)
    if num_array_jobs == 1:
        job_cpus = 16 * gpus_per_job
        job_mem = f"{64 * gpus_per_job}G"
    else:
        job_cpus = 16
        job_mem = "64G"

    slurm_params = {
        "timeout_min": time_limit_minutes,
        "mem": job_mem,
        "cpus_per_task": job_cpus,
        "gpus_per_node": gpus_per_job,
        "slurm_partition": "gpu",
        "slurm_constraint": "[h200|h100|6000_blackwell]",
    }

    result = submit_parallel_jobs(
        jobs_to_submit=jobs_to_submit,
        experiment=f"{experiment}_vs_inference_{process}",
        slurm_params=slurm_params,
        log_dir=f"slurm_virtual_staining_logs/{experiment}/vs_inference",
        manifest_prefix=f"vs_inference_{process}",
        wait_for_completion=True,
    )

    if not result.get("all_completed"):
        failed = result.get("failed", [])
        raise RuntimeError(
            f"Virtual staining inference failed for {process}: "
            f"{len(failed)}/{len(jobs_to_submit)} jobs failed. Failed: {failed}"
        )

    print(f"\nInference complete. Intermediate output: {vs_intermediate_path}")
    return


# ── Nextflow-level inference fan-out (replaces virtual_staining_inference sbatch) ──────────


def virtual_staining_inference_setup(
    experiment: str,
    process: str,
    dim: str = "3d",
    num_array_jobs: int = None,
) -> None:
    """
    Nextflow fan-out Phase 1 for inference: count positions, create intermediate
    dir, print "{num_jobs} {num_positions}" to stdout for Nextflow to flatMap.
    """
    dataset = OpsDataset(experiment)
    dim_norm = (dim or "").lower()
    if dim_norm not in {"3d", "2d"}:
        raise ValueError("dim must be one of {'3d','2d'}")

    if process == "track":
        input_path = dataset.store_paths["lc_5x_phase_3d_optimized"] if dim_norm == "3d" else dataset.store_paths["lc_5x_phase_2d_optimized"]
        vs_output_path = dataset.store_paths["lc_5x_vs"]
    elif process == "pheno":
        input_path = dataset.store_paths["lc_20x_phase_3d_optimized"] if dim_norm == "3d" else dataset.store_paths["lc_20x_phase_2d_optimized"]
        vs_output_path = dataset.store_paths["lc_20x_vs"]
    else:
        raise ValueError(f"process must be 'track' or 'pheno', got {process!r}")

    vs_intermediate_path = vs_output_path.with_suffix("")
    vs_intermediate_path.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["find", str(input_path), "-mindepth", "3", "-maxdepth", "3", "-type", "d"],
        capture_output=True, text=True, check=True,
    )
    num_positions = len(result.stdout.strip().split('\n'))

    if num_array_jobs is None:
        num_array_jobs = 1 if process == "track" else 10

    print(f"{num_array_jobs} {num_positions}", end="")


def virtual_staining_inference_job(
    experiment: str,
    process: str,
    dim: str = "3d",
    job_index: int = 0,
    num_jobs: int = 1,
    num_positions: int = 0,
    batch_size: int = 7,
    num_workers: int = 8,
) -> None:
    """
    Nextflow fan-out Phase 2 for inference: run viscy batch inference for one
    job's position range. Called N times in parallel by Nextflow GPU processes.
    No sbatch — Nextflow submits this process to the GPU partition natively.
    """
    _VISCY_SCRIPT = Path(__file__).parent / "viscy_batch_inference.py"

    dataset = OpsDataset(experiment)
    dim_norm = (dim or "").lower()
    if dim_norm not in {"3d", "2d"}:
        raise ValueError("dim must be one of {'3d','2d'}")

    if process == "track":
        input_path     = dataset.store_paths["lc_5x_phase_3d_optimized"] if dim_norm == "3d" else dataset.store_paths["lc_5x_phase_2d_optimized"]
        vs_output_path = dataset.store_paths["lc_5x_vs"]
        vs_config_path = dataset.config_paths["lc_5x_vs_config"] if dim_norm == "3d" else dataset.config_paths["lc_5x_vs_config_2d"]
    elif process == "pheno":
        input_path     = dataset.store_paths["lc_20x_phase_3d_optimized"] if dim_norm == "3d" else dataset.store_paths["lc_20x_phase_2d_optimized"]
        vs_output_path = dataset.store_paths["lc_20x_vs"]
        vs_config_path = dataset.config_paths["lc_20x_vs_config"] if dim_norm == "3d" else dataset.config_paths["lc_20x_vs_config_2d"]
    else:
        raise ValueError(f"process must be 'track' or 'pheno', got {process!r}")

    vs_intermediate_path = vs_output_path.with_suffix("")

    positions_per_job = (num_positions + num_jobs - 1) // num_jobs
    start_pos = job_index * positions_per_job
    end_pos   = min(start_pos + positions_per_job, num_positions)

    if start_pos >= num_positions:
        print(f"Job {job_index}: no positions to process (start={start_pos} >= total={num_positions})")
        return

    subprocess.run(
        [
            _VISCY_PYTHON, str(_VISCY_SCRIPT),
            "--input-store", str(input_path),
            "--output-dir",  str(vs_intermediate_path),
            "--config",      str(vs_config_path),
            "--start-pos",   str(start_pos),
            "--end-pos",     str(end_pos),
            "--batch-size",  str(batch_size),
            "--num-workers", str(num_workers),
        ],
        check=True,
    )


def _run_combine_predictions(
    intermediate_dir: str,
    input_store: str,
    output_store: str,
    batch_size: int = 7,
    num_workers: int = 16,
) -> None:
    """
    Wrapper function for submitit to run combine_predictions (legacy single-job).

    This is called by SLURM via submitit. It imports and runs the
    combine_batch_predictions module. Used as fallback for HCS format stores.
    """
    from pathlib import Path
    from cyclops_process.utils.combine_batch_predictions import combine_predictions

    combine_predictions(
        intermediate_dir=Path(intermediate_dir),
        input_store=Path(input_store),
        output_store=Path(output_store),
        channel_names=['nuclei', 'membrane'],
        batch_size=batch_size,
        num_workers=num_workers,
    )


# ── Parallel combine: per-phase submitit wrappers ────────────────────────────


def _run_create_combine_store(
    intermediate_dir: str,
    input_store: str,
    output_store: str,
    batch_size: int = 7,
) -> None:
    """Submitit wrapper: create the pre-allocated output store (Phase 1)."""
    from pathlib import Path
    from cyclops_process.utils.combine_batch_predictions import create_combine_output_store

    create_combine_output_store(
        intermediate_dir=Path(intermediate_dir),
        input_store=Path(input_store),
        output_store=Path(output_store),
        channel_names=['nuclei', 'membrane'],
        batch_size=batch_size,
    )


def _run_combine_single_store(
    intermediate_dir: str,
    input_store: str,
    output_store: str,
    store_index: int,
    batch_size: int = 7,
    num_workers: int = 8,
) -> int:
    """Submitit wrapper: stream one prediction store into the output (Phase 2)."""
    from pathlib import Path
    from cyclops_process.utils.combine_batch_predictions import combine_single_batch_store

    return combine_single_batch_store(
        intermediate_dir=Path(intermediate_dir),
        input_store=Path(input_store),
        output_store=Path(output_store),
        store_index=store_index,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def _run_validate_combine(
    output_store: str,
    n_samples: int = 10,
) -> None:
    """Submitit wrapper: validate the combined output store (Phase 3)."""
    from pathlib import Path
    from cyclops_process.utils.combine_batch_predictions import validate_combine_output

    validate_combine_output(
        output_store=Path(output_store),
        n_samples=n_samples,
    )


def _submit_parallel_combine(
    experiment: str,
    process: str,
    vs_intermediate_path: Path,
    input_path: Path,
    vs_output_path: Path,
    log_dir: Path,
    batch_size: int = 7,
    num_workers: int = 8,
    wait: bool = True,
) -> None:
    """
    Run parallel combine: local setup, parallel SLURM streaming, local validation.

    1. Create output store locally (just filesystem metadata, ~2-3 min)
    2. Submit N parallel SLURM jobs to stream prediction stores (~3 min each)
    3. Validate output locally after all jobs complete

    Falls back to the legacy single-job combine if HCS format stores are detected.
    """
    from cyclops_process.utils.combine_batch_predictions import (
        create_combine_output_store,
        discover_prediction_stores,
        validate_combine_output,
    )
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

    # ── Discover prediction stores locally (fast - just directory listing) ──
    pred_stores = discover_prediction_stores(vs_intermediate_path)
    batch_stores = [ps for ps in pred_stores if ps.get('format') != 'hcs']
    hcs_stores = [ps for ps in pred_stores if ps.get('format') == 'hcs']

    if not pred_stores:
        raise ValueError(f"No prediction stores found in {vs_intermediate_path}")

    # HCS format → fall back to legacy single-job combine (uses fast mv)
    if hcs_stores and not batch_stores:
        print("HCS format detected — using legacy single-job combine (fast mv)...")
        result = submit_parallel_jobs(
            jobs_to_submit=[{
                "name": f"vs_combine_{process}",
                "func": _run_combine_predictions,
                "kwargs": {
                    "intermediate_dir": str(vs_intermediate_path),
                    "input_store": str(input_path),
                    "output_store": str(vs_output_path),
                    "batch_size": batch_size,
                    "num_workers": 16,
                },
                "metadata": {"experiment": experiment, "phase": "legacy_combine"},
            }],
            experiment=experiment,
            slurm_params={
                "timeout_min": 20,
                "mem": "32G",
                "cpus_per_task": 16,
                "slurm_partition": "cpu",
            },
            log_dir=f"slurm_virtual_staining_logs/{experiment}/vs_combine",
            manifest_prefix="vs_combine",
            wait_for_completion=wait,
        )
        if wait and not result.get("all_completed"):
            raise RuntimeError(f"Legacy combine job failed for {process}")
        return

    num_stores = len(batch_stores)
    print(f"\nParallel combine: {num_stores} batch stores found")
    print(f"  Step 1: Create output store (local)")
    print(f"  Step 2: Stream stores in parallel ({num_stores} SLURM jobs)")
    print(f"  Step 3: Validate output (local)\n")

    # ── Step 1: Create output store locally ──────────────────────────
    print(f"{'='*60}")
    print(f"Step 1: Creating output store (local)")
    print(f"{'='*60}")

    create_combine_output_store(
        intermediate_dir=vs_intermediate_path,
        input_store=input_path,
        output_store=vs_output_path,
        channel_names=['nuclei', 'membrane'],
        batch_size=batch_size,
    )

    # ── Step 2: Submit parallel SLURM jobs for streaming ─────────────
    print(f"\n{'='*60}")
    print(f"Step 2: Streaming {num_stores} stores in parallel (SLURM)")
    print(f"{'='*60}")

    stream_jobs = []
    for i in range(num_stores):
        store_name = batch_stores[i]['path'].name
        stream_jobs.append({
            "name": f"vs_combine_store{i}_{process} ({store_name})",
            "func": _run_combine_single_store,
            "kwargs": {
                "intermediate_dir": str(vs_intermediate_path),
                "input_store": str(input_path),
                "output_store": str(vs_output_path),
                "store_index": i,
                "batch_size": batch_size,
                "num_workers": num_workers,
            },
            "metadata": {
                "experiment": experiment,
                "store_index": i,
                "store_name": store_name,
                "phase": "stream",
            },
        })

    stream_result = submit_parallel_jobs(
        jobs_to_submit=stream_jobs,
        experiment=experiment,
        slurm_params={
            "timeout_min": 30,
            "mem": "64G",
            "cpus_per_task": 8,
            "slurm_partition": "cpu",
        },
        log_dir=f"slurm_virtual_staining_logs/{experiment}/vs_combine_stream",
        manifest_prefix="vs_combine_stream",
        wait_for_completion=wait,
    )

    if wait and not stream_result.get("all_completed"):
        failed = stream_result.get("failed", [])
        raise RuntimeError(
            f"{len(failed)}/{num_stores} combine streaming jobs failed for {process}. "
            f"Failed: {failed}"
        )

    # ── Step 3: Validate locally ─────────────────────────────────────
    if wait:
        print(f"\n{'='*60}")
        print(f"Step 3: Validating output (local)")
        print(f"{'='*60}")

        validate_combine_output(
            output_store=vs_output_path,
            n_samples=10,
        )

    print(f"\nParallel combine complete for {process}!")


def virtual_staining_combine_only(
    experiment: str,
    process: str,
    dim: str = "3d",
    wait: bool = True,
    batch_size: int = 7,
    num_workers: int = 64,
) -> None:
    """
    Run only the combine job to merge intermediate VS outputs into the final zarr.

    Use this when inference array jobs completed and wrote to the intermediate
    directory, but the combine job failed or never ran (e.g. final store has
    position structure but empty arrays). Does not re-run inference.

    Uses submitit for direct SLURM job submission instead of shell scripts.

    Args:
        experiment: Experiment name (e.g. ops0117_20250207)
        process: 'track' or 'pheno'
        dim: '2d' or '3d' (must match how inference was run)
        wait: If True, block until the combine job completes (default True)
        batch_size: Batch size used during inference (default: 7)
        num_workers: Number of parallel workers for combining (default: 16)
    """
    dataset = OpsDataset(experiment)
    dim_norm = (dim or "").lower()
    if dim_norm not in {"3d", "2d"}:
        raise ValueError("dim must be one of {'3d','2d'}")

    if process == "track":
        if dim_norm == "3d":
            input_path = dataset.store_paths["lc_5x_phase_3d_optimized"]
        else:
            input_path = dataset.store_paths["lc_5x_phase_2d_optimized"]
        vs_output_path = dataset.store_paths["lc_5x_vs"]
    elif process == "pheno":
        if dim_norm == "3d":
            input_path = dataset.store_paths["lc_20x_phase_3d_optimized"]
        else:
            input_path = dataset.store_paths["lc_20x_phase_2d_optimized"]
        vs_output_path = dataset.store_paths["lc_20x_vs"]
    else:
        raise ValueError(f"process must be 'track' or 'pheno', got {process!r}")

    vs_intermediate_path = vs_output_path.with_suffix("")
    if not vs_intermediate_path.exists():
        raise RuntimeError(
            f"Intermediate path does not exist: {vs_intermediate_path}. "
            "Run inference first so that intermediate tiles exist."
        )

    # Always overwrite final output to avoid stale partial data
    if vs_output_path.exists():
        _fast_remove_directory(vs_output_path, "final output store (combine_only)")

    # Absolute log directory
    log_dir = Path.cwd() / f"slurm_logs/slurm_virtual_staining_logs/{experiment}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Run parallel combine (3-phase: setup → parallel stream → validate)
    _submit_parallel_combine(
        experiment=experiment,
        process=process,
        vs_intermediate_path=vs_intermediate_path,
        input_path=input_path,
        vs_output_path=vs_output_path,
        log_dir=log_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        wait=wait,
    )

    return


# ── Nextflow-level parallel combine (replaces _submit_parallel_combine fan-out) ─────────────


def virtual_staining_combine_setup(
    experiment: str,
    process: str,
    dim: str = "3d",
    batch_size: int = 7,
) -> None:
    """
    Nextflow fan-out Phase 1: discover stores, create output skeleton, print N to stdout.

    Prints the number of batch stores as the sole stdout line so Nextflow captures
    it via `stdout` output and can flatMap it into per-store items.
    If HCS-format stores are detected (N=0), runs the legacy single-job combine
    inline and prints 0, allowing Nextflow to skip the stream fan-out entirely.
    """
    dataset = OpsDataset(experiment)
    dim_norm = (dim or "").lower()
    if dim_norm not in {"3d", "2d"}:
        raise ValueError("dim must be one of {'3d','2d'}")

    if process == "track":
        input_path = dataset.store_paths["lc_5x_phase_3d_optimized"] if dim_norm == "3d" else dataset.store_paths["lc_5x_phase_2d_optimized"]
        vs_output_path = dataset.store_paths["lc_5x_vs"]
    elif process == "pheno":
        input_path = dataset.store_paths["lc_20x_phase_3d_optimized"] if dim_norm == "3d" else dataset.store_paths["lc_20x_phase_2d_optimized"]
        vs_output_path = dataset.store_paths["lc_20x_vs"]
    else:
        raise ValueError(f"process must be 'track' or 'pheno', got {process!r}")

    vs_intermediate_path = vs_output_path.with_suffix("")
    if not vs_intermediate_path.exists():
        raise RuntimeError(f"Intermediate path does not exist: {vs_intermediate_path}")

    pred_stores  = discover_prediction_stores(vs_intermediate_path)
    batch_stores = [ps for ps in pred_stores if ps.get('format') != 'hcs']
    hcs_stores   = [ps for ps in pred_stores if ps.get('format') == 'hcs']

    if not pred_stores:
        raise ValueError(f"No prediction stores found in {vs_intermediate_path}")

    if hcs_stores and not batch_stores:
        _run_combine_predictions(
            intermediate_dir=str(vs_intermediate_path),
            input_store=str(input_path),
            output_store=str(vs_output_path),
            batch_size=batch_size,
            num_workers=16,
        )
        print(0, end="")
        return

    create_combine_output_store(
        intermediate_dir=vs_intermediate_path,
        input_store=input_path,
        output_store=vs_output_path,
        channel_names=['nuclei', 'membrane'],
        batch_size=batch_size,
    )
    # WARNING: this must remain the LAST stdout write. Nextflow captures the
    # entire stdout of this process and parses the last line as an integer
    # (n_jobs) via .readLines().last().toInteger() in iss.nf. Any print()
    # after this line — including inside create_combine_output_store — will
    # corrupt the output and cause a NumberFormatException in the pipeline.
    print(len(batch_stores), end="")


def virtual_staining_combine_stream(
    experiment: str,
    process: str,
    dim: str = "3d",
    store_index: int = 0,
    batch_size: int = 7,
    num_workers: int = 8,
) -> None:
    """Nextflow fan-out Phase 2: stream one prediction store into the output zarr."""
    dataset = OpsDataset(experiment)
    dim_norm = (dim or "").lower()
    if dim_norm not in {"3d", "2d"}:
        raise ValueError("dim must be one of {'3d','2d'}")

    if process == "track":
        input_path = dataset.store_paths["lc_5x_phase_3d_optimized"] if dim_norm == "3d" else dataset.store_paths["lc_5x_phase_2d_optimized"]
        vs_output_path = dataset.store_paths["lc_5x_vs"]
    elif process == "pheno":
        input_path = dataset.store_paths["lc_20x_phase_3d_optimized"] if dim_norm == "3d" else dataset.store_paths["lc_20x_phase_2d_optimized"]
        vs_output_path = dataset.store_paths["lc_20x_vs"]
    else:
        raise ValueError(f"process must be 'track' or 'pheno', got {process!r}")

    vs_intermediate_path = vs_output_path.with_suffix("")

    _run_combine_single_store(
        intermediate_dir=str(vs_intermediate_path),
        input_store=str(input_path),
        output_store=str(vs_output_path),
        store_index=store_index,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def virtual_staining_combine_validate(
    experiment: str,
    process: str,
    dim: str = "3d",
    n_samples: int = 10,
) -> None:
    """Nextflow fan-out Phase 3: validate the combined output store."""
    dataset = OpsDataset(experiment)
    dim_norm = (dim or "").lower()
    if dim_norm not in {"3d", "2d"}:
        raise ValueError("dim must be one of {'3d','2d'}")

    if process == "track":
        vs_output_path = dataset.store_paths["lc_5x_vs"]
    elif process == "pheno":
        vs_output_path = dataset.store_paths["lc_20x_vs"]
    else:
        raise ValueError(f"process must be 'track' or 'pheno', got {process!r}")

    validate_combine_output(output_store=vs_output_path, n_samples=n_samples)


def _add_common_experiment_args(parser):
    """Add -e/--experiment, -p/--process, -d/--dim to a subparser."""
    parser.add_argument(
        "--experiment",
        "-e",
        required=True,
        help="Experiment name or shorthand (e.g. 117, ops0117). Resolved via configs.",
    )
    parser.add_argument(
        "--process",
        "-p",
        required=True,
        choices=["track", "pheno"],
        help="Process: track (5x) or pheno (20x).",
    )
    parser.add_argument(
        "--dim",
        "-d",
        default="3d",
        choices=["2d", "3d"],
        help="Dimension (default: 3d).",
    )


def _cli():
    """Parse CLI and run the requested command. See module docstring for usage and examples."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Virtual staining pipeline: preprocess, inference, all, or combine_only.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # preprocess
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Run only preprocessing (normalization metadata generation).",
    )
    _add_common_experiment_args(preprocess_parser)
    preprocess_parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if normalization metadata exists.",
    )
    preprocess_parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        metavar="N",
        help="Number of workers for preprocessing (default: 16).",
    )
    preprocess_parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode.",
    )

    # inference
    inference_parser = subparsers.add_parser(
        "inference",
        help="Run inference (SLURM array jobs) then combine. Blocks until complete.",
    )
    _add_common_experiment_args(inference_parser)
    inference_parser.add_argument(
        "--num-array-jobs",
        type=int,
        default=None,
        metavar="N",
        help="Override auto (pheno=10, track=1). Default: auto.",
    )
    inference_parser.add_argument(
        "--gpus-per-job",
        type=int,
        default=1,
        metavar="N",
        help="GPUs per array job (default: 1).",
    )
    inference_parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        metavar="N",
        help="DataLoader workers (default: 8).",
    )
    inference_parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode.",
    )

    # all (preprocess then inference)
    all_parser = subparsers.add_parser(
        "all",
        help="Run preprocess then inference for the given process (full pipeline).",
    )
    _add_common_experiment_args(all_parser)
    all_parser.add_argument(
        "--force",
        action="store_true",
        help="Force preprocess regeneration even if normalization exists.",
    )
    all_parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=16,
        metavar="N",
        help="Preprocess num_workers (default: 16).",
    )
    all_parser.add_argument(
        "--num-array-jobs",
        type=int,
        default=None,
        metavar="N",
        help="Inference: override auto array job count. Default: auto.",
    )
    all_parser.add_argument(
        "--gpus-per-job",
        type=int,
        default=1,
        metavar="N",
        help="Inference: GPUs per job (default: 1).",
    )
    all_parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode.",
    )

    # combine_only
    combine_parser = subparsers.add_parser(
        "combine_only",
        help="Run only the combine job (e.g. after inference wrote intermediates but combine failed).",
    )
    _add_common_experiment_args(combine_parser)
    combine_parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit the combine job and exit without waiting.",
    )
    combine_parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=7,
        help="Batch size used during inference (default: 7).",
    )
    combine_parser.add_argument(
        "--num-workers",
        "-w",
        type=int,
        default=64,
        help="Number of parallel workers for combining (default: 64).",
    )

    # audit
    audit_parser = subparsers.add_parser(
        "audit",
        help="Audit VS output store for positions with wrong timepoint count or empty data.",
    )
    _add_common_experiment_args(audit_parser)

    args = parser.parse_args()
    experiment = resolve_experiment_name(
        args.experiment.strip(),
        allow_interactive=True,
    )

    if args.command == "preprocess":
        virtual_staining_preprocess(
            experiment=experiment,
            process=args.process,
            dim=args.dim,
            debug=getattr(args, "debug", False),
            force=getattr(args, "force", False),
            num_workers=args.num_workers,
        )
    elif args.command == "inference":
        virtual_staining_inference(
            experiment=experiment,
            process=args.process,
            dim=args.dim,
            debug=args.debug,
            num_workers=args.num_workers,
            num_array_jobs=args.num_array_jobs,
            gpus_per_job=args.gpus_per_job,
        )
    elif args.command == "all":
        virtual_staining_preprocess(
            experiment=experiment,
            process=args.process,
            dim=args.dim,
            debug=args.debug,
            force=args.force,
            num_workers=args.preprocess_workers,
        )
        virtual_staining_inference(
            experiment=experiment,
            process=args.process,
            dim=args.dim,
            debug=args.debug,
            num_array_jobs=args.num_array_jobs,
            gpus_per_job=args.gpus_per_job,
        )
    elif args.command == "combine_only":
        virtual_staining_combine_only(
            experiment=experiment,
            process=args.process,
            dim=args.dim,
            wait=not args.no_wait,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    elif args.command == "audit":
        audit_vs_output(
            experiment=experiment,
            process=args.process,
            dim=args.dim,
        )


def audit_vs_output(experiment: str, process: str, dim: str = "3d"):
    """Audit VS output store for positions with wrong T or empty timepoints."""
    import numpy as np
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor
    from collections import Counter
    from ops_utils.data.filesystem import get_experiment_wells

    dataset = OpsDataset(experiment)

    if process == "track":
        vs_path = dataset.store_paths["lc_5x_vs"]
        input_path = dataset.store_paths["lc_5x_phase_3d_optimized"] if dim == "3d" else dataset.store_paths["lc_5x_phase_2d_optimized"]
    elif process == "pheno":
        vs_path = dataset.store_paths["lc_20x_vs"]
        input_path = dataset.store_paths["lc_20x_phase_3d_optimized"] if dim == "3d" else dataset.store_paths["lc_20x_phase_2d_optimized"]
    else:
        raise ValueError(f"Unknown process: {process}")

    if not vs_path.exists():
        print(f"VS output store does not exist: {vs_path}")
        return None

    wells = get_experiment_wells(experiment, prefix_only=True)

    # Get expected T from input store
    expected_T = None
    if input_path.exists():
        with open_ome_zarr(input_path, mode="r") as inp:
            first_pos = [p for p, _ in inp.positions()][0]
            expected_T = inp[first_pos]["0"].shape[0]
        print(f"  Expected T={expected_T} from input store")

    with open_ome_zarr(vs_path, mode="r") as store:
        all_positions = [p for p, _ in store.positions()]

    problems = []  # (pos, issue, details)
    t_counts = Counter()

    for well in wells:
        well_positions = sorted(p for p in all_positions if p.startswith(well))
        if not well_positions:
            continue

        def _check(args):
            idx, pos = args
            with open_ome_zarr(vs_path / pos, layout="fov", mode="r") as fov:
                arr = fov["0"]
                T = arr.shape[0]
                Y, X = arr.shape[-2], arr.shape[-1]
                issues = []
                if expected_T and T != expected_T:
                    issues.append(f"wrong_T={T} (expected {expected_T})")
                for t in range(T):
                    crop = np.array(arr[t, 0, 0, Y//2-32:Y//2+32, X//2-32:X//2+32])
                    if np.all(crop == 0):
                        issues.append(f"empty_t={t}")
            return (pos, T, issues) if issues else (pos, T, None)

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(tqdm(
                executor.map(_check, enumerate(well_positions)),
                total=len(well_positions),
                desc=f"  Auditing {well} ({process})",
            ))

        for pos, T, issues in results:
            t_counts[T] += 1
            if issues:
                problems.append((pos, issues))

    # Summary
    print(f"\n{'='*60}")
    print(f"  VS AUDIT: {experiment} ({process})")
    print(f"  Store: {vs_path}")
    print(f"{'='*60}")
    print(f"\n  Timepoint distribution:")
    for t, count in sorted(t_counts.items()):
        marker = " <<<" if expected_T and t != expected_T else ""
        print(f"    T={t}: {count} positions{marker}")

    if not problems:
        print(f"\n  All positions OK.")
        return []

    print(f"\n  Found {len(problems)} positions with issues:\n")
    for pos, issues in problems[:20]:
        print(f"    {pos}: {', '.join(issues)}")
    if len(problems) > 20:
        print(f"    ... and {len(problems) - 20} more")

    return problems


if __name__ == "__main__":
    _cli()
