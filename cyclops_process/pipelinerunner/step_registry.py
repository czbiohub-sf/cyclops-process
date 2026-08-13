"""
Central registry of ALL pipeline steps.

This is the single source of truth for step names and their corresponding functions.
Both the orchestrator and batch submission modes use this registry.

Step names come from OpsDataset.get_output_files_for_step() in experiment.py.
"""

from typing import Callable, Dict, Any
import importlib


def get_all_pipeline_steps() -> Dict[str, Dict[str, Any]]:
    """
    Returns ALL pipeline steps with their metadata.

    This is imported lazily to avoid circular dependencies and only imports
    functions when actually needed.

    Returns:
        Dict mapping step_name -> {
            "module": str,          # Module path
            "function": str,        # Function name
            "needs_wells": bool,    # Whether 'wells' param is required
            "needs_process": bool,  # Whether 'process' param is required
        }
    """
    return {
        # --- ISS Processing ---
        "convert_iss": {
            "module": "cyclops_process.convert.tiff_to_zarr",
            "function": "convert",
            "needs_wells": False,
            "needs_process": True,
            "process": "iss",
        },
        "stack_symlinks": {
            "module": "cyclops_process.processes.assemble_link",
            "function": "stack_symlinks",
            "needs_wells": False,
            "needs_process": False,
        },
        "iss_snr_bimodal": {
            "module": "cyclops_process.metrics.plate_stats.iss_snr_bimodal",
            "function": "iss_snr_bimodal",
            "needs_wells": False,
            "needs_process": False,
        },
        "correct_cycle_drift": {
            "module": "cyclops_process.processes.register",
            "function": "correct_cycle_drift",
            "needs_wells": False,
            "needs_process": False,
        },
        "estimate_stitch_parameters_iss": {
            "module": "cyclops_process.processes.ops_stitch",
            "function": "estimate_stitch_parameters",
            "needs_wells": False,
            "needs_process": True,
            "process": "iss",
        },
        "segment_and_stitch_iss": {
            "module": "cyclops_process.processes.segment",
            "function": "segment_and_stitch",
            "needs_wells": False,
            "needs_process": True,
            "process": "iss",
        },
        "estimate_and_stitch_iss": {
            "module": "cyclops_process.processes.ops_stitch",
            "function": "estimate_and_stitch",
            "needs_wells": False,
            "needs_process": True,
            "process": "iss",
        },
        "register_iss_cycles": {
            "module": "cyclops_process.processes.auto_register.iss_cycle_register_orchestrator",
            "function": "register_iss_cycles",
            "needs_wells": False,
            "needs_process": False,
        },
        "merge_spots_base_calling": {
            "module": "cyclops_process.processes.iss_merge",
            "function": "merge_spots_base_calling",
            "needs_wells": False,
            "needs_process": False,
        },
        "convert_iss_to_v3": {
            "module": "cyclops_process.convert.v3_livecell",
            "function": "convert_iss_to_v3",
            "needs_wells": False,
            "needs_process": False,
        },
        "optimize_failed_rounds": {
            "module": "cyclops_process.metrics.plate_stats.optimize_failed_rounds_orchestrator",
            "function": "optimize_failed_rounds",
            "needs_wells": False,
            "needs_process": False,
        },
        "get_metrics": {
            "module": "cyclops_process.metrics.metrics",
            "function": "get_metrics",
            "needs_wells": False,
            "needs_process": False,
        },
        # --- Raw conversion (feeds tracking + phenotyping chains) ---
        "convert_raw": {
            "module": "cyclops_process.convert.raw_to_zarr",
            "function": "convert_raw",
            "needs_wells": False,
            "needs_process": False,
        },
        # --- Live-cell Processing ---
        "link_phenotyping": {
            "module": "cyclops_process.processes.assemble_link",
            "function": "link_phenotyping",
            "needs_wells": False,
            "needs_process": False,
        },
        "link_tracking": {
            "module": "cyclops_process.processes.assemble_link",
            "function": "link_tracking",
            "needs_wells": False,
            "needs_process": False,
        },
        "correct_distortion": {
            "module": "cyclops_process.processes.reconstruct",
            "function": "correct_distortion",
            "needs_wells": False,
            "needs_process": False,
        },
        "reconstruct_track": {
            "module": "cyclops_process.processes.reconstruct",
            "function": "reconstruct",
            "needs_wells": False,
            "needs_process": True,
            "process": "track",
        },
        "calibrate_tilt_track": {
            "module": "cyclops_process.processes.reconstruct_tilt_corrected",
            "function": "calibrate_tilt",
            "needs_wells": True,
            "needs_process": True,
            "process": "track",
        },
        "reconstruct_tilt_corrected_track": {
            "module": "cyclops_process.processes.reconstruct_tilt_corrected",
            "function": "reconstruct_tilt_corrected",
            "needs_wells": True,
            "needs_process": True,
            "process": "track",
        },
        # --- Virtual Staining (Track / 5x 2D) ---
        "virtual_staining_preprocess_track": {
            "module": "cyclops_process.processes.virtual_staining",
            "function": "virtual_staining_preprocess",
            "needs_wells": False,
            "needs_process": True,
            "process": "track",
            "dim": "2d",
        },
        "virtual_staining_inference_track": {
            "module": "cyclops_process.processes.virtual_staining",
            "function": "virtual_staining_inference",
            "needs_wells": False,
            "needs_process": True,
            "process": "track",
            "dim": "2d",
        },
        "virtual_staining_combine_only_track": {
            "module": "cyclops_process.processes.virtual_staining",
            "function": "virtual_staining_combine_only",
            "needs_wells": False,
            "needs_process": True,
            "process": "track",
            "dim": "2d",
        },
        "estimate_stitch_parameters_track": {
            "module": "cyclops_process.processes.ops_stitch",
            "function": "estimate_stitch_parameters",
            "needs_wells": False,
            "needs_process": True,
            "process": "track",
        },
        "segment_and_stitch_track": {
            "module": "cyclops_process.processes.segment",
            "function": "segment_and_stitch",
            "needs_wells": False,
            "needs_process": True,
            "process": "track",
        },
        "estimate_and_stitch_track-2d": {
            "module": "cyclops_process.processes.ops_stitch",
            "function": "estimate_and_stitch",
            "needs_wells": False,
            "needs_process": True,
            "process": "track-2d",
        },
        # --- Phenotyping Reconstruction ---
        "reconstruct_pheno": {
            "module": "cyclops_process.processes.reconstruct",
            "function": "reconstruct",
            "needs_wells": False,
            "needs_process": True,
            "process": "pheno",
        },
        "calibrate_tilt_pheno": {
            "module": "cyclops_process.processes.reconstruct_tilt_corrected",
            "function": "calibrate_tilt",
            "needs_wells": True,
            "needs_process": True,
            "process": "pheno",
        },
        "reconstruct_tilt_corrected_pheno": {
            "module": "cyclops_process.processes.reconstruct_tilt_corrected",
            "function": "reconstruct_tilt_corrected",
            "needs_wells": True,
            "needs_process": True,
            "process": "pheno",
        },
        "create_max_projection_lc_20x_fluor": {
            "module": "cyclops_process.utils.project",
            "function": "create_max_projection",
            "needs_wells": False,
            "needs_process": True,
            "process": "lc_20x_fluor",
        },
        "correct_flatfield_fluor": {
            "module": "cyclops_process.processes.flatfield_correction",
            "function": "correct_flatfield",
            "needs_wells": False,
            "needs_process": True,
            "process": "fluor",
        },
        # --- Virtual Staining (Pheno / 20x 3D) ---
        "virtual_staining_preprocess_pheno": {
            "module": "cyclops_process.processes.virtual_staining",
            "function": "virtual_staining_preprocess",
            "needs_wells": False,
            "needs_process": True,
            "process": "pheno",
            "dim": "3d",
        },
        "virtual_staining_inference_pheno": {
            "module": "cyclops_process.processes.virtual_staining",
            "function": "virtual_staining_inference",
            "needs_wells": False,
            "needs_process": True,
            "process": "pheno",
            "dim": "3d",
        },
        "virtual_staining_combine_only_pheno": {
            "module": "cyclops_process.processes.virtual_staining",
            "function": "virtual_staining_combine_only",
            "needs_wells": False,
            "needs_process": True,
            "process": "pheno",
            "dim": "3d",
        },
        "create_max_projection_lc_20x": {
            "module": "cyclops_process.utils.project",
            "function": "create_max_projection",
            "needs_wells": False,
            "needs_process": True,
            "process": "lc_20x",
        },
        # --- Stitching & Assembly ---
        "estimate_stitch_parameters_pheno": {
            "module": "cyclops_process.processes.ops_stitch",
            "function": "estimate_stitch_parameters",
            "needs_wells": False,
            "needs_process": True,
            "process": "pheno",
        },
        # Native-20x nuclei seg (writes nuclear_seg in place); retires the 5x
        # segment_and_stitch_pheno. See nuclei_pass.py.
        "submit_nuclei_segmentation_jobs": {
            "module": "cyclops_process.processes.cell_seg.nuclei_segmentation_orchestrator",
            "function": "submit_nuclei_segmentation_jobs",
            "needs_wells": False,
            "needs_process": False,
        },
        "segment_and_stitch_pheno_cells": {
            "module": "cyclops_process.processes.segment",
            "function": "segment_and_stitch",
            "needs_wells": False,
            "needs_process": True,
            "process": "pheno_cells",
        },
        "submit_channel_registration_jobs": {
            "module": "cyclops_process.processes.auto_register.channel_reg",
            "function": "submit_channel_registration_jobs",
            "needs_wells": False,
            "needs_process": False,
        },
        "prepare_unified_pheno_tiles": {
            "module": "cyclops_process.processes.register",
            "function": "prepare_unified_pheno_tiles",
            "needs_wells": False,
            "needs_process": False,
        },
        "estimate_and_stitch_pheno-2d": {
            "module": "cyclops_process.processes.ops_stitch",
            "function": "estimate_and_stitch",
            "needs_wells": False,
            "needs_process": True,
            "process": "pheno-2d",
            # TEMPORARY: preserve pyramids/labels when re-stitching (remove after batch re-stitch)
            "restitch_base_only": True,
        },
        "viscy_normalize": {
            "module": "cyclops_process.processes.assemble",
            "function": "viscy_normalize",
            "needs_wells": False,
            "needs_process": False,
        },
        # --- Napari Pyramids ---
        "build_pyramids": {
            "module": "cyclops_process.processes.pyramids.launcher",
            "function": "build_pyramids",
            "needs_wells": False,
            "needs_process": False,
        },
        # --- Automatic Registration ---
        "submit_registration_jobs": {
            "module": "cyclops_process.processes.auto_register.auto_register_orchestrator",
            "function": "submit_registration_jobs",
            "needs_wells": True,
            "needs_process": False,
        },
        # --- Tracking ---
        "track_wells": {
            "module": "cyclops_process.processes.track.track_orchestrator",
            "function": "submit_tracking_jobs",
            "needs_wells": True,
            "needs_process": False,
        },
        "submit_tracking_jobs": {
            "module": "cyclops_process.processes.track.track_orchestrator",
            "function": "submit_tracking_jobs",
            "needs_wells": True,
            "needs_process": False,
        },
        # --- Cell Segmentation ---
        "submit_cell_segmentation_jobs": {
            "module": "cyclops_process.processes.cell_seg.cell_segmentation_orchestrator",
            "function": "submit_cell_segmentation_jobs",
            "needs_wells": False,
            "needs_process": False,
        },
        # --- Link Calls to Tracks ---
        "link_calls_tracks": {
            "module": "cyclops_process.data.datasets",
            "function": "link_calls_tracks",
            "needs_wells": True,
            "needs_process": False,
        },
        # --- ISS Overlay ---
        "build_iss_overlay": {
            "module": "cyclops_process.processes.pyramids.audit_fix",
            "function": "build_iss_overlay",
            "needs_wells": False,
            "needs_process": False,
        },
        # --- Final QC Metrics ---
        "recompute_metrics": {
            "module": "cyclops_process.metrics.metrics",
            "function": "recompute_metrics",
            "needs_wells": False,
            "needs_process": False,
        },
        # --- End-of-pipeline inference (parallel to recompute_metrics) ---
        "organelle_segmentation": {
            "module": "cyclops_process.processes.inference",
            "function": "organelle_segmentation",
            "needs_wells": False,
            "needs_process": False,
        },
        "op_feature_extraction": {
            "module": "cyclops_process.processes.inference",
            "function": "op_feature_extraction",
            "needs_wells": False,
            "needs_process": False,
        },
        "cp_features": {
            "module": "cyclops_process.processes.inference",
            "function": "cp_features",
            "needs_wells": False,
            "needs_process": False,
        },
        "celldino_inference": {
            "module": "cyclops_process.processes.inference",
            "function": "celldino_inference",
            "needs_wells": False,
            "needs_process": False,
        },
        # --- Feature Extraction (not in main orchestrator) ---
        "submit_organelle_segmentation_jobs": {
            "module": "organelle_profiler.organelle_seg.organelle_segmentation_slurm",
            "function": "submit_organelle_segmentation_jobs",
            "needs_wells": False,
            "needs_process": False,
        },
        "build_organelle_pyramids": {
            "module": "cyclops_process.processes.pyramids.launcher",
            "function": "build_organelle_pyramids_only",
            "needs_wells": False,
            "needs_process": False,
        },
        "extract_features_for_experiment": {
            "module": "organelle_profiler.feature_extraction.feature_extraction",
            "function": "extract_features_for_experiment",
            "needs_wells": False,
            "needs_process": False,
        },
    }


def get_step_function(step_name: str) -> Callable:
    """
    Get the callable function for a given step name.

    Args:
        step_name: Name of the step (e.g., "build_pyramids")

    Returns:
        The callable function for this step

    Raises:
        ValueError: If step_name is not found
    """
    steps = get_all_pipeline_steps()
    if step_name not in steps:
        available = ", ".join(sorted(steps.keys()))
        raise ValueError(f"Unknown step '{step_name}'. Available steps: {available}")

    step_info = steps[step_name]
    try:
        module = importlib.import_module(step_info["module"])
        return getattr(module, step_info["function"])
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Could not import {step_info['function']} from {step_info['module']}: {e}"
        )


def get_step_metadata(step_name: str) -> Dict[str, Any]:
    """
    Get metadata about what parameters a step needs.

    Args:
        step_name: Name of the step

    Returns:
        Dict with keys: needs_wells, needs_process, and optionally process (default for batch submission).
    """
    steps = get_all_pipeline_steps()
    if step_name not in steps:
        raise ValueError(f"Unknown step '{step_name}'")

    info = steps[step_name]
    result = {
        "needs_wells": info["needs_wells"],
        "needs_process": info["needs_process"],
    }
    if "process" in info:
        result["process"] = info["process"]
    # Pass through any extra keys (e.g., restitch_base_only)
    RESERVED_KEYS = {"module", "function", "needs_wells", "needs_process", "process"}
    for key, value in info.items():
        if key not in RESERVED_KEYS:
            result[key] = value
    return result


def list_all_steps() -> list[str]:
    """Get list of all available step names."""
    return sorted(get_all_pipeline_steps().keys())
