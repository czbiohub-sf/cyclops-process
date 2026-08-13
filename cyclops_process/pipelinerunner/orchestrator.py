import sys
import os
import submitit
import yaml
from pathlib import Path
from typing import List, Optional, Tuple

from cyclops_utils.data.experiment import OpsDataset
from cyclops_process.pipelinerunner.pipelinerunner import PipelineRunner
from cyclops_process.pipelinerunner.piperun_utils import _matches_selection, Deferred
from cyclops_utils.data.filesystem import (
    resolve_experiment_name,
    setup_experiment_directories,
)

from cyclops_process.processes import (
    segment,
    reconstruct,
    reconstruct_tilt_corrected,
    register,
    virtual_staining,
    assemble,
    ops_stitch,
    flatfield_correction,
    iss_merge,
)
from cyclops_process.processes.pyramids import launcher as build_pyramids
from cyclops_process.processes.track import track_orchestrator
from cyclops_process.processes.auto_register import iss_cycle_register_orchestrator, auto_register_orchestrator
from cyclops_process.processes.auto_register.channel_reg import submit_channel_registration_jobs
from cyclops_process.utils import project
from cyclops_process.convert import raw_to_zarr as convert_raw
from cyclops_process.convert.tiff_to_zarr import convert as tiff_convert
from cyclops_process.processes import assemble_link
from cyclops_process.processes import inference
from cyclops_utils.io.zarr_utils import has_fluorescence_channels_from_config
from cyclops_process.processes.cell_seg import cell_segmentation_orchestrator, nuclei_segmentation_orchestrator
from cyclops_process.data.datasets import link_calls_tracks
from cyclops_process.data.create_dataset import create_dataset
from cyclops_process.metrics import metrics
from cyclops_process.metrics.plate_stats import iss_snr_bimodal
from cyclops_process.metrics.plate_stats import optimize_failed_rounds_orchestrator
from cyclops_process.processes.pyramids.audit_fix import fix_v3_stores


def _get_config_root_dir() -> Path:
    """
    Returns the directory that contains experiment configuration files.
    """
    dummy_dataset = OpsDataset("dummy")
    return dummy_dataset.config_paths["exp_config_dir"]


def _iter_experiment_configs() -> List[Tuple[str, Path]]:
    """
    Lists available experiment config files and returns tuples of
    (experiment_name, config_path).

    Assumes config files are named as "<experiment_name>_config.yaml".
    """
    import os

    results: List[Tuple[str, Path]] = []
    # First check if there's a specific config file set via environment variable
    env_config_file = os.environ.get('OPS_EXP_CONFIG_FILE')
    if env_config_file and Path(env_config_file).exists():
        
        cfg_path = Path(env_config_file)
        stem = cfg_path.stem  # e.g., "ops0060_20250724_config"
        experiment_name = stem[:-7] if stem.endswith("_config") else stem
        results.append((experiment_name, cfg_path))
    # Then scan the config directory for additional configs
    config_root_dir = _get_config_root_dir()
    if config_root_dir.is_dir():
        for cfg in sorted(config_root_dir.glob("*_config.yaml")):
            stem = cfg.stem  # e.g., "ops0060_20250724_config"
            experiment_name = stem[:-7] if stem.endswith("_config") else stem
            # Avoid duplicates if the env config file is also in the config directory
            if not any(name == experiment_name for name, _ in results):
                results.append((experiment_name, cfg))
    return results


def _maybe_handle_manual_checkpoint(
    runner: PipelineRunner, exists: bool, message: str
) -> bool:
    """
    If `exists` is False, prompt the user with `message`.
    Returns True to continue pipeline, False if the user chose to quit.

    Handles the 'back' action by re-running the previously selected step when available.
    """
    if exists:
        return True

    action = runner.menu.prompt_checkpoint(message=message)[0]
    if action == "quit":
        return False
    if action == "back":
        prev_tuple = None
        if getattr(
            runner, "_history_index", None
        ) is not None and 0 <= runner._history_index < len(
            getattr(runner, "_completed_steps_history", [])
        ):
            prev_tuple = runner._completed_steps_history[runner._history_index]
        elif getattr(runner, "_last_completed_step_info", None):
            prev_tuple = runner._last_completed_step_info
        if prev_tuple:
            prev_func, prev_kwargs, prev_key = prev_tuple
            print(f"--- Re-running selected previous step: '{prev_key}' ---")
            runner._execute_step(prev_func, prev_kwargs)
    elif action == "skip":
        # User selected a step from the full list. Check if the target is a
        # previously completed step (behind the current position). If so,
        # re-run it directly and clear the target so the pipeline continues
        # normally. If the target is ahead, the normal skip-forward logic in
        # run() will handle it.
        target_key = runner.menu.get_target_log_key()
        if target_key:
            history = getattr(runner, "_completed_steps_history", [])
            for prev_func, prev_kwargs, prev_key in history:
                if _matches_selection(runner, prev_key, target_key):
                    print(f"--- Re-running selected previous step: '{prev_key}' ---")
                    runner._execute_step(prev_func, prev_kwargs)
                    runner.menu.set_target_log_key(None)
                    break

    # Mark that first-run prompt has been shown, so next incomplete step auto-runs
    runner._first_run_prompt_done = True
    return True


def resolve_experiment_config(
    user_input: str, allow_interactive: bool = True
) -> Optional[Path]:
    """
    Resolves a user-provided experiment identifier to a concrete config file path.

    - Uses resolve_experiment_name for consistent matching logic.
    - Returns the path to the config file or None if not found.

    Note: Interactive selection should be handled by resolve_experiment_name.
    This function just maps the resolved name to its config path.
    """
    # Resolve experiment name (interactive selection handled there if needed)
    resolved_name = resolve_experiment_name(
        user_input, allow_interactive=allow_interactive
    )

    # Get all experiment configs to find the matching path
    entries = _iter_experiment_configs()
    if not entries:
        print("No experiment configuration files were found.")
        return None

    # Find the config path for the resolved name
    for name, path in entries:
        if name == resolved_name:
            return path

    # If no exact match found, the experiment doesn't have a config file
    print(f"No configuration file found for '{resolved_name}'.")
    return None


def _run_experiment_safely(
    config_path: Path, rerun_steps: list = None, use_slurm_steps: bool = False, slurm_task_config_path: str = None, use_dag: bool = False, slurm_qos: str = None,
):
    try:
        print(f"--- Starting job for experiment: {config_path.name} ---")
        run_experiment_from_config(
            config_path, rerun_steps=rerun_steps, use_slurm_steps=use_slurm_steps, slurm_task_config_path=slurm_task_config_path, use_dag=use_dag, slurm_qos=slurm_qos,
        )
        print(f"--- Finished job for experiment: {config_path.name} ---")
    except Exception as e:
        import traceback

        print(f"\n{'!'*20} ERROR: Experiment Failed {'!'*20}")
        print(f"Config file: {config_path.name}")
        print(f"Error: {e}")
        traceback.print_exc()
        print(f"{'!'*50}\n")


def run_all_experiments_from_configs(
    rerun_steps: list = None,
    use_slurm_steps: bool = False,
    use_slurm_experiments: bool = False,
    use_local_parallel: bool = False,
    slurm_task_config_path: str = None,
    experiment_filter: list = None,
    use_dag: bool = False,
    slurm_qos: str = None,
):
    """
    Finds all experiment config files in the designated directory and runs the
    full pipeline for each one, either sequentially or in parallel.

    Args:
        experiment_filter: Optional list of experiment number substrings to filter (e.g., ["46", "47"])
    """
    dummy_dataset = OpsDataset("dummy")
    config_root_dir = dummy_dataset.config_paths["exp_config_dir"]
    if not config_root_dir.is_dir():
        print(f"Configuration directory not found at {config_root_dir}. Aborting.")
        return

    config_files = sorted(config_root_dir.glob("*_config.yaml"))
    if not config_files:
        print(
            f"No config files ending with '_config.yaml' found in {config_root_dir}. Nothing to run."
        )
        return

    # Apply experiment filter if provided
    if experiment_filter:
        from cyclops_utils.data.filesystem import resolve_experiment_name

        # Resolve each filter to canonical experiment name
        resolved_experiments = set()
        for f in experiment_filter:
            resolved = resolve_experiment_name(f, autoselect=True)
            resolved_experiments.add(resolved)

        def matches_filter(config_path):
            """Check if config matches resolved experiment names."""
            # Extract experiment name from config filename (remove _config.yaml suffix)
            name = config_path.stem
            if name.endswith("_config"):
                name = name[:-7]
            return name in resolved_experiments

        original_count = len(config_files)
        config_files = [c for c in config_files if matches_filter(c)]
        print(f"[Filter] Applied experiment filter: {experiment_filter}")
        print(f"[Filter] {original_count} -> {len(config_files)} experiments")

    if use_slurm_experiments:
        # Use step-level SLURM submission - each step gets its own job with proper resources
        print(
            f"Found {len(config_files)} experiments. Submitting each step as individual Slurm jobs with step-specific resource allocation..."
        )
        for config_path in config_files:
            print(f"\n{'='*20} Starting Experiment from: {config_path.name} {'='*20}")
            try:
                run_experiment_from_config(
                    config_path,
                    rerun_steps=rerun_steps,
                    use_slurm_steps=True,  # Use step-specific SLURM configs
                    slurm_task_config_path=slurm_task_config_path,
                    use_dag=use_dag,
                    slurm_qos=slurm_qos,
                )
                print(
                    f"\n{'='*20} Finished Experiment from: {config_path.name} {'='*20}"
                )
            except Exception as e:
                print(
                    f"An error occurred while running the pipeline for {config_path.name}: {e}"
                )
                print("Skipping to the next experiment.")
                continue

    elif use_slurm_steps:
        # Parallel SLURM submission - submit all experiment steps without waiting
        print(
            f"Found {len(config_files)} experiment configurations. Submitting all steps to Slurm in parallel..."
        )
        for config_path in config_files:
            print(f"\n{'='*20} Submitting Experiment: {config_path.name} {'='*20}")
            try:
                run_experiment_from_config(
                    config_path,
                    rerun_steps=rerun_steps,
                    use_slurm_steps=True,
                    slurm_task_config_path=slurm_task_config_path,
                    use_dag=use_dag,
                    slurm_qos=slurm_qos,
                )
            except Exception as e:
                print(
                    f"An error occurred while submitting pipeline for {config_path.name}: {e}"
                )
                print("Skipping to the next experiment.")
                continue
        print(f"\n{'='*20} All experiments submitted to Slurm. {'='*20}")
        print("Jobs are running in parallel. Check Slurm queue with 'squeue -u $USER'")

    elif use_local_parallel:
        from cyclops_utils.hpc.resource_manager import get_optimal_workers
        from joblib import Parallel, delayed

        num_workers = get_optimal_workers(use_gpu=False, verbose=False)
        print(
            f"Found {len(config_files)} experiment configurations. Starting local parallel batch processing with {num_workers} workers..."
        )

        Parallel(n_jobs=num_workers)(
            delayed(_run_experiment_safely)(
                config_path, rerun_steps=rerun_steps, use_slurm_steps=False, slurm_task_config_path=slurm_task_config_path, use_dag=use_dag, slurm_qos=slurm_qos,
            )
            for config_path in config_files
        )
        print(f"\n{'='*20} Finished all experiments. {'='*20}")

    else:
        print(
            f"Found {len(config_files)} experiment configurations. Starting sequential batch processing..."
        )
        for config_path in config_files:
            print(f"\n{'='*20} Starting Experiment from: {config_path.name} {'='*20}")
            try:
                run_experiment_from_config(
                    config_path,
                    rerun_steps=rerun_steps,
                    use_slurm_steps=False,
                    slurm_task_config_path=slurm_task_config_path,
                    use_dag=use_dag,
                    slurm_qos=slurm_qos,
                )
                print(
                    f"\n{'='*20} Finished Experiment from: {config_path.name} {'='*20}"
                )
            except Exception as e:
                print(
                    f"An error occurred while running the pipeline for {config_path.name}: {e}"
                )
                print("Skipping to the next experiment.")
                continue


def run_experiment_from_config(
    config_path: Path,
    rerun_steps: list = None,
    use_slurm_steps: bool = False,
    slurm_task_config_path: str = None,
    no_preflight: bool = False,
    use_dag: bool = False,
    slurm_qos: str = None,
    auto_run: bool = False,
):
    """
    Runs the full OPS pipeline for a single experiment based on a YAML config file.

    Default is sequential execution (topological order, one step at a time).
    Pass use_dag=True (--dag flag) for fully async DAG execution where
    independent branches run concurrently.

    Args:
        config_path: Path to experiment config YAML
        rerun_steps: List of step names to re-run
        use_slurm_steps: If True, submit steps as SLURM jobs
        slurm_task_config_path: Path to custom slurm_task_config.yaml file
        no_preflight: If True, skip file-existence preflight check and let
            each step's internal completion logic decide whether to run.
        use_dag: If True, use parallel DAG execution. Default is sequential.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}.")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        if not config:
            raise ValueError(f"Config file {config_path} is empty or invalid.")

    experiment = config["experiment_name"]

    setup_experiment_directories(experiment)

    # Initialize dataset with config to capture dataset-scoped values (e.g., channel_map)
    dataset = OpsDataset(experiment, config, slurm_task_config_path=slurm_task_config_path)

    runner = PipelineRunner(
        experiment,
        config,
        rerun_steps=rerun_steps,
        use_slurm=use_slurm_steps,
        dataset=dataset,
        slurm_qos=slurm_qos,
        auto_run=auto_run,
    )

    print(f"\n--- Running pipeline for experiment: {experiment} ({'DAG' if use_dag else 'sequential'}) ---")

    dag = _build_pipeline_dag(runner, config, dataset)

    if use_dag:
        dag.run(skip_preflight=no_preflight)
    else:
        _run_pipeline_sequential(runner, dag, config, dataset, experiment)
    print(f"--- Finished pipeline for experiment: {experiment} ---")


def _run_pipeline_sequential(runner, dag, config: dict, dataset, experiment: str):
    """Run the pipeline sequentially using runner.run() for each step.

    This is the original execution model: each step runs one at a time with
    the interactive table UI, skip/override prompts, and completion checking.

    Steps and their order come from the DAG (single source of truth).
    The DAG's topological order is walked linearly, calling runner.run()
    for each step so the full interactive sequential functionality is preserved:
    checkpoints, conditional steps, and skip/rerun prompts.
    """
    from cyclops_process.pipelinerunner.exceptions import PipelineHalted

    # Build a map: step_name -> list of checkpoints that gate it (via `before`)
    checkpoint_gates: dict[str, list] = {}
    for cp_name, cp_def in dag.checkpoints.items():
        for gated_step in cp_def.before:
            checkpoint_gates.setdefault(gated_step, []).append(cp_def)

    fired_checkpoints: set[str] = set()

    for step_name in dag._topological_order():
        step_def = dag.steps[step_name]

        # Skip steps whose condition was False at DAG-build time
        if not step_def.condition:
            continue

        # Handle any checkpoints that gate this step
        for cp_def in checkpoint_gates.get(step_name, []):
            if cp_def.name in fired_checkpoints:
                continue
            fired_checkpoints.add(cp_def.name)
            if cp_def.condition:
                if not _maybe_handle_manual_checkpoint(
                    runner, exists=False, message=cp_def.message,
                ):
                    return

        # Run the step through PipelineRunner (interactive table, completion check, etc.)
        try:
            runner.run(step_def.func, **step_def.params)
        except PipelineHalted as halt:
            print(f"\n--- Pipeline halted at step '{step_name}': {halt.reason} ---")
            return

    print(f"--- Finished pipeline for experiment: {experiment} ---")


def _build_pipeline_dag(runner, config: dict, dataset) -> "DAGRunner":
    """Build the full pipeline DAG with all steps registered declaratively.

    Dependencies come from slurm_task_config.yaml (single source of truth).
    Step names here must match the yaml keys exactly.

    Conditional steps (ISS, fluorescence) are omitted from the DAG entirely
    when data is not present — their downstream deps become auto-satisfied.
    """
    from cyclops_process.pipelinerunner.dag_runner import DAGRunner
    from cyclops_process.convert import v3_livecell as convert_v3_slurm

    dag = DAGRunner(runner)

    # ── ISS Processing ──────────────────────────────────────────────────
    # ISS steps are only registered if not in "run_iss_only" mode or if ISS data exists.
    # When ISS steps are absent, their downstream deps are treated as satisfied.
    has_iss = not config.get("skip_iss", False)
    run_iss_only = config.get("run_iss_only", False)

    iss_rounds = config.get("metrics_params", {}).get("iss_rounds", list(range(10)))
    failed_rounds_by_well = config.get("metrics_params", {}).get("failed_rounds_by_well")
    wells = config.get("wells_to_process", [])

    if has_iss:
        dag.add("convert_iss", tiff_convert,
                {"process": "iss", **config["convert_iss_params"]})
        dag.add("stack_symlinks", assemble_link.stack_symlinks,
                config["stack_symlinks_params"])
        dag.add("iss_snr_bimodal", iss_snr_bimodal.iss_snr_bimodal,
                config.get("metrics_snr_bimodal_params", {}))
        dag.add("correct_cycle_drift", register.correct_cycle_drift,
                config["correct_cycle_drift_params"])
        dag.add("estimate_stitch_parameters_iss", ops_stitch.estimate_stitch_parameters,
                {"process": "iss", **config["estimate_stitch_parameters_iss_params"]})
        dag.add("segment_and_stitch_iss", segment.segment_and_stitch,
                {"process": "iss", **config["segment_and_stitch_iss_params"]})
        dag.add("estimate_and_stitch_iss", ops_stitch.estimate_and_stitch,
                {"process": "iss", **config["estimate_and_stitch_iss_params"]})

        # Merged post-stitch step: warp + detect_spots + base_calling fused
        # per-well in shared memory, no NFS round-trip through
        # bc_stitched_registered.zarr. register_iss_cycles must skip its final
        # apply_transforms — the merge step now owns that work.
        register_iss_kwargs = {
            **config.get("register_iss_cycles_params", {}),
            "skip_apply_transforms": True,
        }
        dag.add("register_iss_cycles", iss_cycle_register_orchestrator.register_iss_cycles,
                register_iss_kwargs)
        dag.add("merge_spots_base_calling", iss_merge.merge_spots_base_calling,
                config.get("merge_spots_base_calling_params", {}))
        # ISS is independent of the track/pheno chains, so convert its stitched
        # store to v3 right away and async-delete the v2 source. Pheno/track
        # stitch v3-native (PR #96), so this is the only remaining convert.
        dag.add("convert_iss_to_v3", convert_v3_slurm.convert_iss_to_v3,
                config.get("convert_iss_to_v3_params", {}))
        # Optimize per-well failed rounds and rewrite ops_failed_rounds.yaml before
        # metrics; get_metrics re-reads that config at runtime.
        dag.add("optimize_failed_rounds",
                optimize_failed_rounds_orchestrator.optimize_failed_rounds,
                config.get("optimize_failed_rounds_params", {}))
        # failed_rounds is resolved at dispatch (not now): optimize_failed_rounds
        # rewrites the experiment config mid-run, so re-read it fresh when
        # get_metrics is scheduled, falling back to the startup value.
        def _fresh_failed_rounds(_ds=dataset, _fallback=failed_rounds_by_well):
            try:
                cfg = yaml.safe_load(open(_ds.config_paths["exp_config"])) or {}
                fr = cfg.get("metrics_params", {}).get("failed_rounds_by_well")
                return fr if fr is not None else _fallback
            except Exception:
                return _fallback
        dag.add("get_metrics", metrics.get_metrics,
                {"iss_rounds": iss_rounds,
                 "failed_rounds_by_well": Deferred(_fresh_failed_rounds)})

    if run_iss_only:
        # Only ISS steps — DAG stops here since no downstream registered
        return dag

    # ── Raw conversion ──────────────────────────────────────────────────
    # First live-cell step: dragonfly tiffs -> zarr on the fast partition.
    # Feeds both link_tracking and link_phenotyping (deps in slurm_task_config.yaml).
    dag.add("convert_raw", convert_raw.convert_raw,
            config.get("convert_raw_params", {}))

    # ── Tracking chain ──────────────────────────────────────────────────
    dag.add("link_tracking", assemble_link.link_tracking,
            config["link_tracking_params"])
    dag.add("correct_distortion", reconstruct.correct_distortion,
            config["correct_distortion_params"])
    dag.add("reconstruct_track", reconstruct.reconstruct,
            {"process": "track", **config["reconstruct_track_params"]})
    dag.add("calibrate_tilt_track", reconstruct_tilt_corrected.calibrate_tilt,
            {"process": "track", "wells": wells,
             **config.get("calibrate_tilt_track_params", {})})
    dag.add("reconstruct_tilt_corrected_track", reconstruct_tilt_corrected.reconstruct_tilt_corrected,
            {"process": "track", "wells": wells,
             **config.get("reconstruct_tilt_corrected_track_params", {})})
    dag.add("virtual_staining_preprocess_track", virtual_staining.virtual_staining_preprocess,
            {"process": "track", "dim": "2d",
             **config.get("virtual_staining_preprocess_track_params", {})})
    dag.add("virtual_staining_inference_track", virtual_staining.virtual_staining_inference,
            {"process": "track", "dim": "2d",
             **config.get("virtual_staining_inference_track_params", {})})
    dag.add("virtual_staining_combine_only_track", virtual_staining.virtual_staining_combine_only,
            {"process": "track", "dim": "2d",
             **config.get("virtual_staining_combine_only_track_params", {})})
    dag.add("estimate_stitch_parameters_track", ops_stitch.estimate_stitch_parameters,
            {"process": "track", **config.get("estimate_stitch_parameters_track_params", {})})
    dag.add("segment_and_stitch_track", segment.segment_and_stitch,
            {"process": "track", **config["segment_and_stitch_track_params"]})
    dag.add("estimate_and_stitch_track-2d", ops_stitch.estimate_and_stitch,
            {"process": "track-2d", **config.get("estimate_and_stitch_track-2d_params", {})})

    # ── Phenotyping chain ───────────────────────────────────────────────
    dag.add("link_phenotyping", assemble_link.link_phenotyping,
            config["link_phenotyping_params"])

    # Fluorescence max projection + flatfield (conditional, right after link)
    has_fluor = has_fluorescence_channels_from_config(config)
    dag.add("create_max_projection_lc_20x_fluor", project.create_max_projection,
            {"process": "lc_20x_fluor", "projection": "max",
             **config.get("create_max_projection_lc_20x_fluor_params", {})},
            condition=has_fluor)
    dag.add("correct_flatfield_fluor", flatfield_correction.correct_flatfield,
            {"process": "fluor", **config.get("correct_flatfield_fluor_params", {})},
            condition=has_fluor)

    dag.add("reconstruct_pheno", reconstruct.reconstruct,
            {"process": "pheno", **config["reconstruct_pheno_params"]})
    dag.add("calibrate_tilt_pheno", reconstruct_tilt_corrected.calibrate_tilt,
            {"process": "pheno", "wells": wells,
             **config.get("calibrate_tilt_pheno_params", {})})
    dag.add("reconstruct_tilt_corrected_pheno", reconstruct_tilt_corrected.reconstruct_tilt_corrected,
            {"process": "pheno", "wells": wells,
             **config.get("reconstruct_tilt_corrected_pheno_params", {})})
    dag.add("virtual_staining_preprocess_pheno", virtual_staining.virtual_staining_preprocess,
            {"process": "pheno", "dim": "3d",
             **config.get("virtual_staining_preprocess_pheno_params", {})})
    dag.add("virtual_staining_inference_pheno", virtual_staining.virtual_staining_inference,
            {"process": "pheno", "dim": "3d",
             **config.get("virtual_staining_inference_pheno_params", {})})
    dag.add("virtual_staining_combine_only_pheno", virtual_staining.virtual_staining_combine_only,
            {"process": "pheno", "dim": "3d",
             **config.get("virtual_staining_combine_only_pheno_params", {})})
    dag.add("create_max_projection_lc_20x", project.create_max_projection,
            {"process": "lc_20x", **config["create_max_projection_lc_20x_params"]})

    dag.add("estimate_stitch_parameters_pheno", ops_stitch.estimate_stitch_parameters,
            {"process": "pheno", **config.get("estimate_stitch_parameters_pheno_params", {})})
    # Nuclei segmentation is now native-20x (submit_nuclei_segmentation_jobs,
    # added below after v3 assembly); the 5x segment_and_stitch_pheno is retired.

    # ── Automatic fluor->Phase2D channel registration (before review) ───
    # Best-effort auto-registration + review renders, fanned out one SLURM job
    # per fluor channel. Runs before the review checkpoint so the user reviews
    # the auto result and either accepts it or quits to register manually.
    dag.add("submit_channel_registration_jobs", submit_channel_registration_jobs,
            {"channel_map": config.get("channel_map"),
             **config.get("auto_register_channels_params", {})},
            condition=has_fluor)

    # ── Post-channel-registration stitching and assembly ────────────────
    # (manual review checkpoint removed — pipeline keeps running through
    # auto-registered fluor channels like any other step)
    dag.add("prepare_unified_pheno_tiles", register.prepare_unified_pheno_tiles,
            config.get("prepare_unified_pheno_tiles_params", {}))
    dag.add("estimate_and_stitch_pheno-2d", ops_stitch.estimate_and_stitch,
            {"process": "pheno-2d", **config.get("estimate_and_stitch_pheno-2d_params", {})})
    dag.add("viscy_normalize", assemble.viscy_normalize,
            config.get("viscy_normalize_params", {}))
    # All stitched stores are v3 by this point (pheno/track v3-native, iss
    # converted right after merge), so build pyramids against the _v3 stores.
    dag.add("build_pyramids", build_pyramids.build_pyramids,
            {"use_v3_stores": True, **config.get("build_pyramids_params", {})})

    # ── Cell segmentation ───────────────────────────────────────────────
    dag.add("submit_cell_segmentation_jobs", cell_segmentation_orchestrator.submit_cell_segmentation_jobs,
            config.get("cell_segmentation_params", {}))

    # ── Nuclei segmentation (native 20x) ────────────────────────────────
    # Writes the `nuclear_seg` label into phenotyping_v3.zarr; must precede
    # registration/tracking, which read it.
    dag.add("submit_nuclei_segmentation_jobs", nuclei_segmentation_orchestrator.submit_nuclei_segmentation_jobs,
            config.get("nuclei_segmentation_params", {}))

    # ── Automatic registration and tracking ─────────────────────────────
    dag.add("submit_registration_jobs", auto_register_orchestrator.submit_registration_jobs,
            {"wells": wells,
             **config.get("auto_register_params", {})})
    dag.add("submit_tracking_jobs", track_orchestrator.submit_tracking_jobs,
            {"wells": wells, **config["track_params"]})

    # ── Zarr v3 conversion ──────────────────────────────────────────────
    # All three stores are v3 by this point: pheno/track stitch v3-native
    # (PR #96) and iss is converted right after merge_spots_base_calling by
    # convert_iss_to_v3. No end-of-pipeline conversion step is needed.

    # ── Link calls to tracks ────────────────────────────────────────────
    dag.add("link_calls_tracks", link_calls_tracks,
            {"wells": wells, **config["link_calls_tracks_params"]})

    # ── Fix v3 stores (audit + fix all missing pyramids, overlays, labels, clims) ─
    dag.add("fix_v3_stores", fix_v3_stores,
            {"auto_fix": True, **config.get("fix_v3_stores_params", {})})

    # ── Final QC metrics ────────────────────────────────────────────────
    dag.add("recompute_metrics", metrics.recompute_metrics,
            {"iss_rounds": iss_rounds,
             "failed_rounds_by_well": failed_rounds_by_well})

    # ── End-of-pipeline model inference (parallel to recompute_metrics) ──
    # OP branch (organelle seg → feature extraction), CellProfiler, and CellDINO.
    # Each reads phenotyping_v3.zarr; deps enforced via slurm_task_config.
    dag.add("organelle_segmentation", inference.organelle_segmentation,
            config.get("organelle_segmentation_params", {}))
    dag.add("op_feature_extraction", inference.op_feature_extraction,
            config.get("op_feature_extraction_params", {}))
    dag.add("cp_features", inference.cp_features,
            config.get("cp_features_params", {}))
    dag.add("celldino_inference", inference.celldino_inference,
            config.get("celldino_inference_params", {}))

    return dag
