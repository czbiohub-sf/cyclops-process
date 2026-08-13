"""
Utility functions for PipelineRunner: parsing, formatting, and Slurm metrics.

This module contains helper functions for:
- Memory/time/size parsing and formatting
- Slurm job statistics collection and display
- GPU allocation and metrics parsing
"""

import subprocess
import re
import math
from datetime import datetime
from pathlib import Path
from ops_utils.profiling.decorators import _get_git_commit_hash, _load_and_write_log
from ops_utils.data.experiment import OpsDataset

# Baseline well count the static slurm_task_config timeouts are tuned for.
_BASELINE_WELLS = 3


def scale_timeout_for_wells(slurm_params: dict, config: dict, log_key: str = "") -> dict:
    """Scale a step's timeout_min by well count.

    Configs are tuned for 3 wells; single-job steps take ~N/3x wall time with N
    wells. Only timeout scales — mem/cpu/gpu are bounded per-well/per-tile, not by
    total well count. Returns a new dict when scaled, else the original unchanged.
    """
    if not slurm_params or not slurm_params.get("timeout_min"):
        return slurm_params
    wells = config.get("wells_to_process", []) if isinstance(config, dict) else []
    n_wells = len(wells or [])
    if n_wells > _BASELINE_WELLS:
        mult = math.ceil(n_wells / _BASELINE_WELLS)
        scaled = {**slurm_params, "timeout_min": int(slurm_params["timeout_min"]) * mult}
        print(f"[well-scale] {log_key}: {n_wells} wells → timeout ×{mult} = {scaled['timeout_min']} min")
        return scaled
    return slurm_params


def _lookup_slurm_config_with_fallback(slurm_task_config: dict, log_key: str) -> dict:
    """
    Look up Slurm configuration for a log_key with fallback to base key.

    Tries in order:
    1. Exact match (e.g., track_well_A_1_0)
    2. Strip well tag (e.g., track_well_A_1_0 -> track_well)
    3. Strip method tag (e.g., base_calling_probabilistic -> base_calling)

    Args:
        slurm_task_config: Dictionary of Slurm configurations keyed by step name
        log_key: Generated log key that may include well and/or method suffixes

    Returns:
        Dictionary containing slurm_params and other config, or empty dict if no match
    """
    # 1. Try exact match first
    if log_key in slurm_task_config:
        return slurm_task_config[log_key]

    # 2. Try stripping well tag (pattern: _[A-Z]_\d+_\d+ at end)
    # Handles: track_well_A_1_0, segment_pheno_B_2_0, etc.
    well_pattern = re.compile(r"_[A-Z]_\d+(_\d+)?$")
    key_without_well = well_pattern.sub("", log_key)
    if key_without_well != log_key and key_without_well in slurm_task_config:
        return slurm_task_config[key_without_well]

    return {}


class Deferred:
    """A step parameter resolved at dispatch time instead of DAG-build time.

    `PipelineRunner._prepare_step_execution` calls `.resolve()` just before the
    step runs, so the value can reflect state produced by earlier steps in the
    same run (e.g. get_metrics reading failed_rounds_by_well from the experiment
    config after optimize_failed_rounds rewrote it). `_generate_log_key` ignores
    it (only inspects process/well), so an unresolved Deferred is harmless there.
    """

    def __init__(self, fn):
        self._fn = fn

    def resolve(self):
        return self._fn()


def _generate_log_key(self, func, **kwargs):
    """Generates a unique key aligned with the DAG step names."""
    func_name = func.__name__
    key_parts = [func_name]
    if "process" in kwargs:
        key_parts.append(str(kwargs["process"]))
    # Include well for per-well steps
    if "well" in kwargs:
        key_parts.append(str(kwargs["well"]).replace("/", "_"))
    return "_".join(key_parts)


def _matches_selection(self, log_key: str, selected: str | None) -> bool:
    """
    Return True if the concrete log_key matches a selected target. Supports
    exact matches (e.g., 'labels_to_contours') and prefix matches for:
    - Per-well variants (e.g., 'track_well' matches 'track_well_A_1_0')

    Does NOT match arbitrary process suffixes like _fluor or _2d. This prevents
    'create_max_projection_lc_20x' from matching 'create_max_projection_lc_20x_fluor'.
    """
    if not selected:
        return False
    if log_key == selected:
        return True


    # Check if log_key starts with selected + "_"
    prefix = str(selected) + "_"
    if not log_key.startswith(prefix):
        return False

    # Extract the suffix after the selected key
    suffix = log_key[len(prefix):]

    # Match well patterns: uppercase letter(s) + underscore + digits, optionally more
    # Examples: A_1, B_2_0, AA_4, A_1_0
    well_suffix_pattern = re.compile(r"^[A-Z]+_\d+(_\d+)?$")
    if well_suffix_pattern.match(suffix):
        return True


    return False


# Consolidated helpers
def _has_content(self, p: Path) -> bool:
    try:
        if not p.exists():
            return False
        if p.is_file():
            return p.stat().st_size > 0
        if p.is_dir():
            # Fast Zarr validation: presence of .zgroup or .zarray
            if p.suffix == ".zarr" or p.name.endswith(".zarr"):
                if (p / ".zgroup").exists() or (p / ".zarray").exists():
                    return True
                return False
            return next(p.iterdir(), None) is not None
        return False
    except Exception:
        return False


def _dataset_for_kwargs(self, kwargs: dict | None) -> OpsDataset:
    if kwargs and "method" in kwargs:
        return OpsDataset(
            self.experiment, self.config, method=str(kwargs.get("method"))
        )
    return self.dataset


def _get_output_files(
    self, func, kwargs: dict | None = None, config: dict | None = None
):
    """Return expected outputs for a step using canonical key (func + optional process + well).

    For per-well steps, include the well suffix so the dataset can check the specific well.
    """
    ds = self._dataset_for_kwargs(kwargs)
    try:
        func_name = func.__name__ if hasattr(func, "__name__") else str(func)
    except Exception:
        func_name = str(func)
    process = (kwargs or {}).get("process")
    well = (kwargs or {}).get("well")

    # Build the key: func_name + optional _process + optional _well
    base_key = func_name
    if process:
        base_key = f"{base_key}_{process}"
    if well:
        base_key = f"{base_key}_{str(well).replace('/', '_')}"

    return ds.get_output_files_for_step(base_key, config or self.config)


# Unified audit logging helpers
def _audit_log_start(self, log_key: str) -> None:
    log_entry = {
        log_key: {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "git_commit": _get_git_commit_hash(),
        }
    }
    _load_and_write_log(log_key, self.log_file_path, log_entry)


# Selection/history helpers
def _add_completed_history_entry(self, func, kwargs: dict, log_key: str) -> int:
    self._completed_steps_history.append((func, kwargs, log_key))
    number = self._next_menu_index
    self._selection_map[number] = (
        "history",
        len(self._completed_steps_history) - 1,
    )
    self._next_menu_index += 1
    self._last_completed_step_info = self._completed_steps_history[-1]
    return number


def _set_one_shot_target(self, key: str) -> None:
    self._target_log_key_to_run_once = str(key)
    # Keep skip-to-incomplete enabled so we fast-forward to the selected step
    self.skip_to_incomplete = True


def _clear_one_shot_target_if_matched(self, log_key: str) -> None:
    if self._target_log_key_to_run_once == log_key:
        self._target_log_key_to_run_once = None


def _audit_log_end(self, log_key: str, elapsed_seconds: float) -> None:
    log_entry = {
        log_key: {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "Ran in": f"{elapsed_seconds:.2f} seconds",
            "git_commit": _get_git_commit_hash(),
        }
    }
    _load_and_write_log(log_key, self.log_file_path, log_entry)


def _format_timeout_display(self, step_key: str) -> str:
    """
    Return formatted resource string from Slurm config for the given step key/log_key.
    Includes timeout and resources when available, e.g.: " (~15m, 8 CPU, 64G RAM, 1 GPU)".
    Returns empty string if nothing is defined.
    Uses fallback lookup to handle well and method tags.
    """
    try:
        cfg = _lookup_slurm_config_with_fallback(self.slurm_task_config, step_key)
        if not isinstance(cfg, dict):
            return ""
        sp = (
            cfg.get("slurm_params") if isinstance(cfg.get("slurm_params"), dict) else {}
        )

        parts: list[str] = []

        # Timeout
        tval = sp.get("timeout_min")
        if isinstance(tval, (int, float)):
            minutes = float(tval)
            if minutes > 60:
                hours = minutes / 60.0
                if abs(hours - round(hours)) < 1e-6:
                    parts.append(f"~{int(round(hours))}h")
                else:
                    parts.append(f"~{hours:.1f}h")
            else:
                parts.append(f"~{int(minutes)}m")

        # CPUs
        cval = sp.get("cpus_per_task")
        if isinstance(cval, (int, float)):
            parts.append(f"{int(cval)} CPU")

        # Memory
        mval = sp.get("mem")
        if isinstance(mval, str) and mval.strip():
            parts.append(f"{mval.strip()} RAM")

        # GPUs
        gval = sp.get("gpus_per_node")
        if isinstance(gval, (int, float)) and int(gval) > 0:
            parts.append(f"{int(gval)} GPU")

        return f" (" + ", ".join(parts) + ")" if parts else ""
    except Exception:
        return ""