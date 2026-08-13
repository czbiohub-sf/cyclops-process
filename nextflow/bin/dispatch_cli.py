#!/usr/bin/env python
# CLI for invoking cyclops_process functions from Nextflow processes.
# CuPy-importing modules (stitch, segment, register, assemble, datasets) and
# heavy frameworks (torch/waveorder, dask, GPU) are loaded lazily — only after
# the command is known — so that forking steps (e.g. calibrate_tilt) never
# inherit a broken CUDA runtime state from the main process.
import argparse
import importlib
import inspect
import os
import sys
import typing
from dataclasses import dataclass, field
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# All cyclops_process imports are lazy — resolved only after the command is known.
# This prevents CuPy, torch, dask, and other heavy/GPU frameworks from being
# imported on every invocation regardless of which function is called.


@dataclass(frozen=True)
class ConfigKey:
    """Typed binding from a command to its experiment-config param block(s).

    Either a single ``default`` key applied to every process, or a per-process
    ``by_process`` map (e.g. iss / track / pheno). Modeled as a dataclass rather
    than a bare ``str | dict`` so the binding is typed, resolution and
    serialization live in one place, and it can't silently nest — the edge case
    guarded in ``__post_init__``: every value is a plain config key (str), never
    another dict. That keeps a very flexibly-read param set deterministic.
    """
    default: "str | None" = None
    by_process: "dict[str, str]" = field(default_factory=dict)

    def __post_init__(self):
        if self.default is not None and not isinstance(self.default, str):
            raise TypeError(
                f"ConfigKey.default must be a str key, got {type(self.default).__name__}"
            )
        for proc, key in self.by_process.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"ConfigKey.by_process[{proc!r}] must be a str key, "
                    f"got {type(key).__name__}"
                )

    def resolve(self, process: "str | None") -> "str | None":
        """The config key for this process: a per-process override if present,
        otherwise the default (None if neither)."""
        if process is not None and process in self.by_process:
            return self.by_process[process]
        return self.default

    def serialize(self) -> dict:
        """Deterministic, repeatable dump of the binding's contents (stable key
        order) — for logging/inspection of how a command points at config."""
        return {
            "default": self.default,
            "by_process": {p: self.by_process[p] for p in sorted(self.by_process)},
        }


@dataclass(frozen=True)
class Command:
    """A single dispatchable step. One entry per command is the sole source of
    truth for its module, callable, and experiment-config binding — there is no
    parallel dict keyed by the same command names to drift against.

    config_key (a ConfigKey) binds the step to the experiment-config block(s)
    whose contents are merged, under explicit CLI args, into the call — the
    ``**config[...]`` expansion in orchestrator.py::_build_pipeline_dag. None
    when the step takes no config-sourced params.
    """
    module: str
    func: str
    config_key: "ConfigKey | None" = None


COMMANDS = {
    # ── ISS ──────────────────────────────────────────────────────────────────
    'convert_tiff_to_zarrv3':          Command('cyclops_process.processes.assemble_link', 'convert', ConfigKey('convert_iss_params')),
    'stack_symlinks':                  Command('cyclops_process.processes.assemble_link', 'stack_symlinks', ConfigKey('stack_symlinks_params')),
    'correct_cycle_drift':             Command('cyclops_process.processes.register', 'correct_cycle_drift', ConfigKey('correct_cycle_drift_params')),
    'estimate_stitch_parameters':      Command('cyclops_process.processes.ops_stitch', 'estimate_stitch_parameters', ConfigKey(by_process={
        'iss':   'estimate_stitch_parameters_iss_params',
        'track': 'estimate_stitch_parameters_track_params',
        'pheno': 'estimate_stitch_parameters_pheno_params',
    })),
    'segment_and_stitch':              Command('cyclops_process.processes.segment', 'segment_and_stitch', ConfigKey(by_process={
        'iss':   'segment_and_stitch_iss_params',
        'track': 'segment_and_stitch_track_params',
        # pheno retired (native-20x nuclei seg replaces the 5x segment_and_stitch_pheno)
    })),
    'estimate_and_stitch':             Command('cyclops_process.processes.ops_stitch', 'estimate_and_stitch', ConfigKey(by_process={
        'iss':      'estimate_and_stitch_iss_params',
        'track-2d': 'estimate_and_stitch_track-2d_params',
        'pheno-2d': 'estimate_and_stitch_pheno-2d_params',
    })),
    'register_iss_seg_to_nucleus':     Command('cyclops_process.processes.auto_register.iss_cycle_register_orchestrator', 'job_register_segmentation_to_nucleus'),
    'register_iss_nucleus_to_round0':  Command('cyclops_process.processes.auto_register.iss_cycle_register_orchestrator', 'job_register_nucleus_to_round0'),
    'register_iss_round_pair':         Command('cyclops_process.processes.auto_register.iss_cycle_register_orchestrator', 'job_register_round_pair'),
    'finalize_iss_registration':       Command('cyclops_process.processes.auto_register.iss_cycle_register_orchestrator', 'job_finalize_registration'),
    'precreate_iss_registered':        Command('cyclops_process.processes.auto_register.iss_cycle_register_orchestrator', 'precreate_iss_registered'),
    'iss_snr_bimodal':                 Command('cyclops_process.metrics.plate_stats.iss_snr_bimodal', 'iss_snr_bimodal', ConfigKey('metrics_snr_bimodal_params')),
    'merge_spots_base_calling':        Command('cyclops_process.processes.iss_merge', 'merge_spots_base_calling_well'),
    'convert_iss_to_v3':               Command('cyclops_process.convert.v3_livecell', 'convert_iss_to_v3', ConfigKey('convert_iss_to_v3_params')),
    'optimize_failed_rounds':          Command('cyclops_process.metrics.plate_stats.optimize_failed_rounds_orchestrator', 'optimize_failed_rounds', ConfigKey('optimize_failed_rounds_params')),
    'get_metrics':                     Command('cyclops_process.metrics.metrics', 'get_metrics', ConfigKey('metrics_params')),
    # ── Pheno / tracking ─────────────────────────────────────────────────────
    'convert_raw':                     Command('cyclops_process.convert.raw_to_zarr', 'convert_raw', ConfigKey('convert_raw_params')),
    'link_phenotyping':                Command('cyclops_process.processes.assemble_link', 'link_phenotyping', ConfigKey('link_phenotyping_params')),
    'link_tracking':                   Command('cyclops_process.processes.assemble_link', 'link_tracking', ConfigKey('link_tracking_params')),
    'correct_flatfield':               Command('cyclops_process.processes.flatfield_correction', 'correct_flatfield', ConfigKey('correct_flatfield_fluor_params')),
    'channel_registration_setup':      Command('cyclops_process.processes.auto_register.channel_reg', 'channel_registration_setup'),
    'channel_registration_job':        Command('cyclops_process.processes.auto_register.channel_reg', 'channel_registration_job'),
    'nuclei_segmentation_setup':       Command('cyclops_process.processes.cell_seg.nuclei_segmentation_orchestrator', 'nuclei_segmentation_setup'),
    'nuclei_segmentation_job':         Command('cyclops_process.processes.cell_seg.nuclei_segmentation_orchestrator', 'nuclei_segmentation_job'),
    'correct_distortion':              Command('cyclops_process.processes.reconstruct', 'correct_distortion', ConfigKey('correct_distortion_params')),
    'reconstruct':                     Command('cyclops_process.processes.reconstruct', 'reconstruct', ConfigKey(by_process={
        'track': 'reconstruct_track_params',
        'pheno': 'reconstruct_pheno_params',
    })),
    'calibrate_tilt':                  Command('cyclops_process.processes.reconstruct_tilt_corrected', 'calibrate_tilt'),
    'get_n_positions':                 Command('cyclops_process.processes.reconstruct_tilt_corrected', 'get_n_positions'),
    'reconstruct_tilt_corrected':      Command('cyclops_process.processes.reconstruct_tilt_corrected', 'reconstruct_tilt_corrected'),
    'reconstruct_tilt_corrected_setup': Command('cyclops_process.processes.reconstruct_tilt_corrected', 'reconstruct_tilt_corrected_setup'),
    'reconstruct_tilt_corrected_job':  Command('cyclops_process.processes.reconstruct_tilt_corrected', 'reconstruct_tilt_corrected_job'),
    'virtual_staining_preprocess':       Command('cyclops_process.processes.virtual_staining', 'virtual_staining_preprocess'),
    'virtual_staining_inference':        Command('cyclops_process.processes.virtual_staining', 'virtual_staining_inference'),
    'virtual_staining_combine_only':     Command('cyclops_process.processes.virtual_staining', 'virtual_staining_combine_only'),
    'virtual_staining_combine_setup':    Command('cyclops_process.processes.virtual_staining', 'virtual_staining_combine_setup'),
    'virtual_staining_combine_stream':   Command('cyclops_process.processes.virtual_staining', 'virtual_staining_combine_stream'),
    'virtual_staining_combine_validate': Command('cyclops_process.processes.virtual_staining', 'virtual_staining_combine_validate'),
    'virtual_staining_inference_setup':  Command('cyclops_process.processes.virtual_staining', 'virtual_staining_inference_setup'),
    'virtual_staining_inference_job':    Command('cyclops_process.processes.virtual_staining', 'virtual_staining_inference_job'),
    'create_max_projection':           Command('cyclops_process.processes.assemble', 'create_max_projection', ConfigKey(by_process={
        'lc_20x':       'create_max_projection_lc_20x_params',
        'lc_20x_fluor': 'create_max_projection_lc_20x_fluor_params',
    })),
    'prepare_unified_pheno_tiles':     Command('cyclops_process.processes.register', 'prepare_unified_pheno_tiles', ConfigKey('prepare_unified_pheno_tiles_params')),
    'viscy_normalize':                 Command('cyclops_process.processes.assemble', 'viscy_normalize', ConfigKey('viscy_normalize_params')),
    'build_pyramids':           Command('cyclops_process.processes.pyramids.launcher', 'build_pyramids'),
    'build_pyramids_setup':     Command('cyclops_process.processes.pyramids.launcher', 'build_pyramids_setup'),
    'build_pyramids_position_job':       Command('cyclops_process.processes.pyramids.launcher', 'build_pyramids_position_job'),
    'submit_registration_jobs':        Command('cyclops_process.processes.auto_register.auto_register_orchestrator', 'submit_registration_jobs'),
    'submit_tracking_jobs':            Command('cyclops_process.processes.track.track_orchestrator', 'submit_tracking_jobs'),
    'run_v3_conversion':               Command('cyclops_process.convert.v3_livecell', 'run_v3_conversion'),
    'submit_cell_segmentation_jobs':   Command('cyclops_process.processes.cell_seg.cell_segmentation_orchestrator', 'submit_cell_segmentation_jobs'),
    'link_calls_tracks':               Command('cyclops_process.data.datasets', 'link_calls_tracks', ConfigKey('link_calls_tracks_params')),
    'fix_v3_stores':                   Command('cyclops_process.processes.pyramids.audit_fix', 'fix_v3_stores', ConfigKey('fix_v3_stores_params')),
    'recompute_metrics':               Command('cyclops_process.metrics.metrics', 'recompute_metrics', ConfigKey('metrics_params')),
}


def _load_config_params(command: Command, process, valid_params):
    """Per-experiment kwargs for this step, read from the experiment config
    (OPS_EXP_CONFIG_FILE) and filtered to the function signature. Returns {} when
    there's no binding, no config file, or no matching params — so a step with no
    config_key behaves exactly as before (CLI args + function defaults)."""
    key = command.config_key.resolve(process) if command.config_key else None
    if not key:
        return {}
    cfg_path = os.environ.get('OPS_EXP_CONFIG_FILE')
    if not cfg_path or not Path(cfg_path).exists():
        return {}
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return {}
    block = cfg.get(key) or {}
    if not isinstance(block, dict):
        return {}
    return {k: v for k, v in block.items()
            if k in valid_params and k != 'experiment'}


def _type_from_annotation(annotation):
    if annotation is inspect.Parameter.empty:
        return str
    origin = getattr(annotation, '__origin__', None)
    if origin is typing.Union:  # handles Optional[X]
        args = [a for a in annotation.__args__ if a is not type(None)]
        return _type_from_annotation(args[0]) if len(args) == 1 else str
    if origin is typing.Literal:
        return str
    if annotation is bool:
        return lambda x: x.lower() not in ('false', '0', 'no')
    if annotation in (int, float, str):
        return annotation
    return str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=list(COMMANDS.keys()))

    # Parse command, leave rest for function-specific args
    args, remaining = parser.parse_known_args()

    # Resolve the command's callable on demand (lazy import).
    command = COMMANDS[args.command]
    func = getattr(importlib.import_module(command.module), command.func)
    sig = inspect.signature(func)
    try:
        hints = typing.get_type_hints(func)
    except Exception:
        hints = {}

    func_parser = argparse.ArgumentParser()
    for param_name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
            continue
        type_fn = _type_from_annotation(hints.get(param_name, param.annotation))
        if param.default == inspect.Parameter.empty:
            func_parser.add_argument(f'--{param_name}', required=True, type=type_fn)
        else:
            # SUPPRESS so unpassed optionals stay absent — lets experiment-config
            # values fill them without argparse defaults clobbering the config.
            func_parser.add_argument(f'--{param_name}', default=argparse.SUPPRESS, type=type_fn)

    # Explicit CLI args (from nextflow_ops_args.yaml python_kwargs) take precedence
    # over per-experiment params pulled from the experiment config; anything set in
    # neither falls back to the function's own default.
    cli_args = vars(func_parser.parse_args(remaining))
    config_params = _load_config_params(command, cli_args.get('process'), set(sig.parameters))
    func(**{**config_params, **cli_args})


if __name__ == '__main__':
    main()
