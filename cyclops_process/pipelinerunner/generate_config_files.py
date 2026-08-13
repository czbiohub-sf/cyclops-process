import yaml
from pathlib import Path
import argparse
import sys
import os
from copy import deepcopy
import re
import shutil
from datetime import datetime


# how to run:
# python -m cyclops_process.pipelinerunner.generate_config_files

# Ensure the package is in the path
sys.path.insert(0, os.getcwd())
from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.data.filesystem import resolve_experiment_name
from cyclops_process.configs.sync_configs_to_hpc import sync_all_configs
from cyclops_utils.data.bad_experiments import (
    derive_experimental_design,
    derive_project,
    get_category,
)
from cyclops_process.paths import BASE_PATH


def _get_config_paths() -> dict:
    """Get paths to config files via OpsDataset."""
    dataset = OpsDataset("dummy")
    return {
        "channel_maps": dataset.channel_maps,
        "failed_rounds": dataset.failed_rounds,
        "library_map": dataset.library_map,
    }


# This dictionary serves as the default template for all generated config files.
DEFAULT_CONFIG = {
    "experiment_name": "placeholder",
    # Project this experiment belongs to (used by reporting/grouping).
    # Derived automatically from the library map when not set explicitly.
    # Recognized values are config-driven (per-experiment `project` field in ops_library_map.yaml).
    "project": None,
    # Imaging modality / experimental design. Derived automatically from the
    # channel map when not set. Recognized values: OPS, 4i, Cell_Painting, MERFISH.
    "experimental_design": None,
    # Cell line used in this experiment. Default A549; override per-experiment as needed.
    "cell_line": "A549",
    "wells_to_process": ["A/1/0", "A/2/0", "A/3/0"],
    "run_iss_only": False,
    "channel_map": {
        "BF": "Phase",
        "GFP": None,
        "mCherry": None,
    },
    # Channels kept in channel_map for labeling but excluded from live-cell
    # link_phenotyping (cell-painting CP*/4i panels). Set from ops_channel_maps.yaml.
    "fixed_channels": [],
    "metrics_snr_bimodal_params": {},
    "convert_iss_params": {},
    "stack_symlinks_params": {"pre_nuclei_round": False, "skip_pre_dapi_round": False},
    "correct_cycle_drift_params": {"pad": False, "fast": True},
    # In-place ISS distortion correction on the drift-corrected store
    "correct_distortion_iss_params": {},
    "estimate_stitch_parameters_iss_params": {"flipud": True, "fliplr": False, "rot90": 0},
    "segment_and_stitch_iss_params": {},
    "estimate_and_stitch_iss_params": {"flipud": True, "fliplr": False, "rot90": 0},
    "register_iss_cycles_params": {},
    "detect_spots_params": {},
    "base_calling_params": {
        "iss_rounds": list(range(0, 10)),
        "failed_rounds_by_well": None,  # See examples below
        # ========== Failed Rounds Examples ==========
        # DROPOUT MODE: Skip the round entirely (both physical cycle and logical barcode position)
        #   Example: {"A/1/0": [0, 3]}
        #   Result: Uses rounds [1,2,4,5,6,7,8,9] for both image reading and codebook matching
        #
        # OFFSET MODE: Shift to next physical cycle, but keep all logical positions
        #   Example: {"A/1/0": {"dropout": [], "offset": 5}}
        #   Result: Physical rounds [0,1,2,3,4,6,7,8,9,10], Logical rounds [0,1,2,3,4,5,6,7,8,9]
        #   (Round 5 failed, so read physical cycle 6 as logical position 5)
        #
        # COMBINED: Use both dropout and offset
        #   Example: {"A/1/0": {"dropout": [0], "offset": 5}}
        #   Result: Physical [1,2,3,4,6,7,8,9,10], Logical [1,2,3,4,5,6,7,8,9]
        #   (Skip position 0 entirely, shift position 5+ to next physical cycle)
    },

    # Library/codebook configuration (overridden per-experiment from ops_library_map.yaml)
    "codebook": "pool1_design.csv",
    "gene_index": "library/twist1k_pool_CERES.csv",
    "codebook_column_map": None,
    "gene_index_column_map": None,
    "gene_name_output_column": None,
    "iss_secondary_gene_column": None,

    "metrics_params": {
        "iss_rounds": list(range(0, 10)),
        "failed_rounds_by_well": None,  # See examples in base_calling_params
    },
    # Runs before get_metrics: optimizes per-well failed rounds and writes them
    # into ops_failed_rounds.yaml + this config (conservative: 1 round>3%,
    # 2 rounds>5%, removal capped at 2). Empty = default job params.
    "optimize_failed_rounds_params": {},
    # End-of-pipeline model inference (parallel to recompute_metrics). Each step
    # generates its per-experiment config under configs/inference_configs/<model>/v2/.
    "organelle_segmentation_params": {},
    "op_feature_extraction_params": {},
    "cp_features_params": {},
    "celldino_inference_params": {},
    # Raw conversion (dragonfly tiffs -> zarr); empty = convert all configured
    # wells. Pheno wells absent from wells_to_process are skipped automatically.
    "convert_raw_params": {},
    "link_phenotyping_params": {
        "phase_fliplr": True,
        "phase_flipud": False,
        "phase_rot90": 1,
        "gfp_fliplr": False,
        "gfp_flipud": False,
        "gfp_rot90": 0,
        "mCherry_fliplr": True,
        "mCherry_flipud": False,
        "mCherry_rot90": 0,
        "cy5_fliplr": False,
        "cy5_flipud": False,
        "cy5_rot90": 0,
    },
    "link_tracking_params": {},
    "correct_distortion_params": {},
    "reconstruct_track_params": {},
    "reconstruct_track-2d_params": {},
    "reconstruct_pheno_params": {},
    "reconstruct_pheno-2d_params": {},
    "estimate_stitch_parameters_track_params": {},
    "estimate_stitch_parameters_pheno_params": {},
    "estimate_and_stitch_track-2d_params": {},
    "estimate_and_stitch_pheno-2d_params": {},
    "virtual_staining_track_params": {},
    "virtual_staining_pheno_params": {},
    # "focus": project each FOV over its per-FOV in-focus range from the tilt
    # calibration (see create_max_projection). Use "all" to project every Z plane.
    "create_max_projection_lc_20x_params": {"slices": "focus"},
    "create_max_projection_lc_20x_fluor_params": {"slices": "all"},
    "correct_flatfield_fluor_params": {},
    "segment_and_stitch_track_params": {},
    "segment_and_stitch_pheno_params": {},
    "segment_and_stitch_pheno_cells_params": {
        "use_preprocess": True,
        "clahe_clip_limit": 0.01,
    },
    # New: mirror orientation-flip params for the unified tile preparation step
    "prepare_unified_pheno_tiles_params": {},
    # New: params for viscy normalization step (optional)
    "estimate_and_stitch_pheno-2d_params": {},
    "build_pyramids_params": {
        "levels": 5,
        "factor": 2,
        "grid_line_width": 1,
    },
    "viscy_normalize_params": {},
    "apply_registration_tracking_params": {},
    "stack_params": {},
    "labels_to_contours_params": {},
    "auto_register_params": {
        "skip_track": False,  # If True, register ISS and pheno directly without using tracking data
    },
    "track_params": {
        "skip_track": False,  # If True, skip tracking entirely
    },
    "link_calls_tracks_params": {
        "iss_rounds": list(range(0, 10)),
        "failed_rounds_by_well": None,  # See examples in base_calling_params
        "skip_track": False,  # If True, adjust for missing tracking timepoints
    },
    "create_dataset_params": {},
    # ISS-only v3 convert that runs right after merge_spots_base_calling and
    # async-deletes the v2 source. Pheno/track stitch v3-native, so no other
    # convert step is needed. Empty = sensible defaults (force=False, delete_v2=True).
    "convert_iss_to_v3_params": {},
    "fix_v3_stores_params": {},  # Audits and fixes all missing pyramids, overlays, labels, clims

}

# Experiment exclusion lists — loaded from shared bad_experiment.yaml
ISS_ONLY_EXPERIMENTS = get_category("iss_only")
DO_NOT_RUN_EXPERIMENTS = get_category("do_not_run")

# Experiments that are missing the 9th round of ISS (use 9 rounds instead of 10)
ISS_MISSING_NINTH_ROUND = [42]

# Experiments that need gfp_fliplr=False
GFP_FLIPLR_EXPS = [35, 36, 46, 63]

# Experiments with pre-nuclei round (explicit list for exp < 100; all exp > 99 are automatic)
PRE_NUCLEI_ROUND_EXPS = [78, 81, 83, 85, 89, 90, 91, 94, 96, 98]

# Experiments that explicitly do NOT have a pre-nuclei round (overrides the >99 auto-rule)
# 179: no DAPI-only pre-round; round 1 is the 5-channel (DAPI+SBS) round 0.
NO_PRE_NUCLEI_ROUND_EXPS = [138, 156, 173, 179]


# skip pre-dapi round for these experiments
SKIP_PRE_DAPI_ROUND_EXPS = [131, "ops0117_11_20_20260324", "ops0156_11_20_20260625", 139, 143, 146, 154]

# Experiments that skip tracking
SKIP_TRACK_EXPS = [106, 108, 149, 166]

# CROPseq library design — appended to experimental_design (on top of the
# auto-derived modalities like livecell_OPS / Cell_Painting / 4i).
CROPSEQ_EXPERIMENTS = [45, 63, 86, 103, 150]

# Cell Painting experiments not auto-detected from ops_channel_maps.yaml
# (i.e. lacking a ``cell_painting: enabled: true`` entry). Appended to
# experimental_design just like the auto-derived flag would be (the
# extend-loop dedupes, so listing an experiment that's also flagged in the
# channel map is harmless).
CELL_PAINTING_EXPERIMENTS = [72, 94]

# Experiment groups for batch config generation
EXPERIMENT_GROUPS = {
    "blnk_exps": [
        "ops0000_blnk_PLA_20251103",
        "ops0000_blnk_delrin_20251103",
        "ops0000_blnk_oracal_20251103",
        "ops0000_blnk_clrPETG_20251031",
        "ops0000_blnk_blkPETG_20251031",
        "ops0000_blnk_PCTG_20251031",
        "ops0000_blnk_curr_caps_20251031",
    ],
    "ops0079_materials": [
        "ops0079_rnd10_cardboard",
        "ops0079_black2pt0",
        "ops0079_black_oracal",
        "ops0079_blk_anodized",
        "ops0079_delrin",
        "ops0079_milled_acry",
        "ops0079_orig_cap",
        "ops0079_PETG",
        "ops0079_PLA",
        "ops0079_clr_PETG",
        "ops0079_PCTG",
    ],
    "ops0079_timepoints": [
        "ops0079_pr2_20h_20251030",
        "ops0079_pr2_41h_20251031",
        "ops0079_cleave_20h_20251030",
        "ops0079_cleave_41h_20251031",
        "ops0079_inc_20h_20251030",
        "ops0079_inc_41h_20251031",
    ],
}


def backup_config_files(config_root_dir):
    """Backs up the existing config directory to a new timestamped directory."""
    if config_root_dir.exists() and any(config_root_dir.iterdir()):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backups_dir = config_root_dir.parent / "backups"
        backups_dir.mkdir(parents=True, exist_ok=True)
        backup_dir = backups_dir / f"{config_root_dir.name}_backup_{timestamp}"

        try:
            shutil.copytree(config_root_dir, backup_dir)
            print(f"--- Successfully backed up existing configs to '{backup_dir}' ---")
        except Exception as e:
            print(f"--- ERROR: Could not back up config directory. Reason: {e} ---")
    else:
        print("--- No existing config directory to back up. ---")


# ============================================================================
# Channel Map Loading (from ops_channel_maps.yaml)
# ============================================================================

def _load_ops_channel_maps() -> dict:
    """Load per-experiment channel mappings from `ops_channel_maps.yaml`."""
    channel_maps_yaml = _get_config_paths()["channel_maps"]
    if not channel_maps_yaml.exists():
        return {}
    try:
        with open(channel_maps_yaml, "r") as f:
            data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"    -> ERROR: Failed to parse {channel_maps_yaml}: {e}")
        return {}


def _normalize_channel_key(channel_name: str) -> str:
    """Normalize raw channel names from YAML to config keys."""
    if not channel_name:
        return channel_name
    lower = str(channel_name).strip().lower()
    if lower in ("brightfield", "bf", "bright field"):
        return "BF"
    if lower == "gfp":
        return "GFP"
    if lower == "mcherry":
        return "mCherry"
    return channel_name


def _normalize_label_value(label_value):
    """Normalize label values coming from YAML."""
    if label_value is None:
        return None
    if isinstance(label_value, str):
        stripped = label_value.strip()
        if stripped == "":
            return None
        if "no label" in stripped.lower():
            return "no label"
        # Normalize "phase" capitalization
        if stripped.lower() == "phase":
            return "Phase"
        return stripped
    return label_value


def _build_channel_map_override(per_experiment_entry: list) -> dict:
    """Convert list entries from YAML into the config's channel_map dict."""
    channel_map = {}
    if not isinstance(per_experiment_entry, list):
        return channel_map
    for item in per_experiment_entry:
        if not isinstance(item, dict):
            continue
        raw_name = item.get("channel_name")
        norm_key = _normalize_channel_key(raw_name)
        label_value = _normalize_label_value(item.get("label"))
        channel_map[norm_key] = label_value
    return channel_map


def _build_fixed_channels(per_experiment_entry: list) -> list:
    """Names of channels flagged ``fixed: true`` (cell-painting / 4i panels).

    These stay in channel_map for labeling but are excluded from live-cell
    link_phenotyping, which only handles BF/GFP/mCherry/Cy5.
    """
    fixed = []
    if not isinstance(per_experiment_entry, list):
        return fixed
    for item in per_experiment_entry:
        if isinstance(item, dict) and item.get("fixed") and item.get("channel_name"):
            fixed.append(_normalize_channel_key(item["channel_name"]))
    return fixed


# ============================================================================
# Failed Rounds Loading (from ops_failed_rounds.yaml)
# ============================================================================

def _load_ops_failed_rounds() -> dict:
    """Load per-experiment failed rounds from `ops_failed_rounds.yaml`."""
    failed_rounds_yaml = _get_config_paths()["failed_rounds"]
    if not failed_rounds_yaml.exists():
        return {}
    try:
        with open(failed_rounds_yaml, "r") as f:
            data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"    -> ERROR: Failed to parse {failed_rounds_yaml}: {e}")
        return {}


def _get_exp_number(experiment_name: str) -> int | None:
    """Extract experiment number from experiment name."""
    match = re.search(r"ops(\d{4})", experiment_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _get_ops_key(experiment_name: str) -> str | None:
    """Extract ops key (e.g., 'ops0065') from experiment name."""
    match = re.search(r"ops(\d{4})", experiment_name, re.IGNORECASE)
    if match:
        return f"ops{match.group(1)}".lower()
    return None


# ============================================================================
# Library/Codebook Loading (from ops_library_map.yaml)
# ============================================================================

def _load_ops_library_map() -> dict:
    """Load per-experiment library/codebook mappings from `ops_library_map.yaml`."""
    library_map_yaml = _get_config_paths()["library_map"]
    if not library_map_yaml.exists():
        return {}
    try:
        with open(library_map_yaml, "r") as f:
            data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"    -> ERROR: Failed to parse {library_map_yaml}: {e}")
        return {}


def _apply_library_override(
    experiment_name: str, config: dict, ops_library_map: dict
) -> None:
    """Apply codebook/gene_index overrides from ops_library_map.yaml."""
    ops_key = _get_ops_key(experiment_name)

    defaults = ops_library_map.get("default", {})
    overrides = ops_library_map.get("overrides", {})

    # Start with defaults
    if defaults.get("codebook"):
        config["codebook"] = defaults["codebook"]
    if defaults.get("gene_index"):
        config["gene_index"] = defaults["gene_index"]

    # Apply per-experiment override (try ops key, e.g. "ops0138")
    exp_override = overrides.get(ops_key, {}) if ops_key else {}
    if exp_override:
        if "codebook" in exp_override:
            config["codebook"] = exp_override["codebook"]
            print(f"    -> NOTE: Setting codebook={exp_override['codebook']}")
        if "gene_index" in exp_override:
            config["gene_index"] = exp_override["gene_index"]
            print(f"    -> NOTE: Setting gene_index={exp_override['gene_index']}")
        if "codebook_column_map" in exp_override:
            config["codebook_column_map"] = exp_override["codebook_column_map"]
            print(f"    -> NOTE: Setting codebook_column_map={exp_override['codebook_column_map']}")
        if "gene_index_column_map" in exp_override:
            config["gene_index_column_map"] = exp_override["gene_index_column_map"]
            print(f"    -> NOTE: Setting gene_index_column_map={exp_override['gene_index_column_map']}")
        if "gene_name_output_column" in exp_override:
            config["gene_name_output_column"] = exp_override["gene_name_output_column"]
            print(f"    -> NOTE: Setting gene_name_output_column={exp_override['gene_name_output_column']}")
        if "iss_secondary_gene_column" in exp_override:
            config["iss_secondary_gene_column"] = exp_override["iss_secondary_gene_column"]
            print(f"    -> NOTE: Setting iss_secondary_gene_column={exp_override['iss_secondary_gene_column']}")


# ============================================================================
# Per-Experiment Config Application
# ============================================================================

def _apply_channel_map_override(
    experiment_name: str, config: dict, ops_channel_maps: dict
) -> None:
    """Apply channel map override from ops_channel_maps.yaml if available."""
    ops_key = _get_ops_key(experiment_name)

    # Try various key formats
    per_exp_entry = None
    for key in [ops_key, ops_key.upper() if ops_key else None, experiment_name, experiment_name.lower()]:
        if key and key in ops_channel_maps:
            per_exp_entry = ops_channel_maps[key]
            break

    if per_exp_entry:
        override_map = _build_channel_map_override(per_exp_entry)
        cleaned_override = {k: v for k, v in override_map.items() if v is not None}
        if cleaned_override:
            config["channel_map"] = cleaned_override
            # Format channel map for display
            channel_str = ", ".join(f"{k}: {v}" for k, v in cleaned_override.items())
            print(f"    -> NOTE: Setting channel_map: {{{channel_str}}}")
        else:
            print(f"    -> WARNING: No channel_map found for '{experiment_name}'")

        # Mark fixed-cell panel channels (CP/4i) so link_phenotyping skips them
        fixed_channels = [c for c in _build_fixed_channels(per_exp_entry) if c in cleaned_override]
        config["fixed_channels"] = fixed_channels
        if fixed_channels:
            print(f"    -> NOTE: Marking {len(fixed_channels)} fixed (non-live-cell) channel(s): {fixed_channels}")


def _apply_failed_rounds_override(
    experiment_name: str, config: dict, ops_failed_rounds: dict
) -> None:
    """Apply failed rounds and related overrides from ops_failed_rounds.yaml."""
    exp_config = ops_failed_rounds.get(experiment_name)
    if not exp_config:
        print(f"    -> WARNING: No entry in ops_failed_rounds.yaml. To optimize run:")
        print(f"       python -m cyclops_process.metrics.plate_stats.optimize_failed_rounds {experiment_name} --all-wells")
        return

    # Apply failed_rounds_by_well
    # Use deepcopy to avoid YAML anchor/alias references in output
    if "failed_rounds_by_well" in exp_config:
        failed_rounds = exp_config["failed_rounds_by_well"]
        config["base_calling_params"]["failed_rounds_by_well"] = deepcopy(failed_rounds)
        config["metrics_params"]["failed_rounds_by_well"] = deepcopy(failed_rounds)
        config["link_calls_tracks_params"]["failed_rounds_by_well"] = deepcopy(failed_rounds)
        # Always print the failed rounds per well
        has_dropouts = any(rounds for rounds in failed_rounds.values() if rounds)
        wells_str = ", ".join(f"{w}: {r}" for w, r in failed_rounds.items())
        if has_dropouts:
            print(f"    -> NOTE: failed_rounds_by_well: {{{wells_str}}}")
        else:
            print(f"    -> OK: failed_rounds_by_well: all wells clean (no dropouts)")

    # Apply wells_to_process
    if "wells_to_process" in exp_config:
        config["wells_to_process"] = exp_config["wells_to_process"]
        print(f"    -> NOTE: Setting wells_to_process={exp_config['wells_to_process']}")

    # Apply cell_line override (e.g., HeLa instead of default A549)
    if "cell_line" in exp_config:
        config["cell_line"] = exp_config["cell_line"]
        print(f"    -> NOTE: Setting cell_line={exp_config['cell_line']}")

    # Apply iss_rounds
    if "iss_rounds" in exp_config:
        iss_rounds = list(exp_config["iss_rounds"])
        config["base_calling_params"]["iss_rounds"] = iss_rounds
        config["metrics_params"]["iss_rounds"] = list(iss_rounds)
        config["link_calls_tracks_params"]["iss_rounds"] = list(iss_rounds)
        print(f"    -> NOTE: Setting iss_rounds={iss_rounds}")

    # Apply run_iss_only
    if "run_iss_only" in exp_config:
        config["run_iss_only"] = exp_config["run_iss_only"]
        if exp_config["run_iss_only"]:
            print(f"    -> NOTE: Setting run_iss_only=True")

    # Apply iss_tif_dir override (e.g., for experiments acquired on a different
    # instrument). Values in ops_failed_rounds.yaml are written against
    # $OPS_INSTRUMENT_ROOT, so expand env vars into a concrete path here — the
    # consumers (OpsDataset, tiff_to_zarr) do not expand them themselves.
    if "iss_tif_dir" in exp_config:
        resolved = os.path.expandvars(str(exp_config["iss_tif_dir"]))
        if "$" in resolved:
            raise RuntimeError(
                f"iss_tif_dir for this experiment references an unset environment "
                f"variable: {exp_config['iss_tif_dir']!r}. Set OPS_INSTRUMENT_ROOT "
                f"to the mount holding the raw acquisitions."
            )
        config["iss_tif_dir"] = resolved
        print(f"    -> NOTE: Setting iss_tif_dir={resolved}")

    # Apply ISS anchor round (e.g., 1 when round 0 has no spots). Top-level key
    # (like codebook_round_offset) — NOT under auto_register_params, which is
    # spread as kwargs into submit_registration_jobs and would reject it.
    if "register_anchor_round" in exp_config:
        config["anchor_round"] = exp_config["register_anchor_round"]
        print(f"    -> NOTE: Setting anchor_round={exp_config['register_anchor_round']}")

    # Apply codebook_round_offset (e.g., 10 for experiments using rounds 11-20 of a 20-round pool)
    if "codebook_round_offset" in exp_config:
        config["codebook_round_offset"] = exp_config["codebook_round_offset"]
        print(f"    -> NOTE: Setting codebook_round_offset={exp_config['codebook_round_offset']}")

    # Apply tile_size override (e.g., for cameras with non-standard sensor dimensions)
    if "tile_size" in exp_config:
        ts = list(exp_config["tile_size"])
        config["stack_symlinks_params"]["tile_size"] = ts
        config["estimate_stitch_parameters_iss_params"]["tile_size"] = ts
        config["estimate_and_stitch_iss_params"]["tile_size"] = ts
        print(f"    -> NOTE: Setting tile_size={ts} for stack_symlinks + stitch + segment params")

    # Apply ISS stitch orientation overrides (e.g., for new microscopes that don't need flipud)
    iss_stitch_params = ["estimate_stitch_parameters_iss_params", "estimate_and_stitch_iss_params", "segment_and_stitch_iss_params"]
    if "iss_stitch_flipud" in exp_config:
        for p in iss_stitch_params:
            config[p]["flipud"] = exp_config["iss_stitch_flipud"]
        print(f"    -> NOTE: Setting ISS stitch + segment flipud={exp_config['iss_stitch_flipud']}")
    if "iss_stitch_fliplr" in exp_config:
        for p in iss_stitch_params:
            config[p]["fliplr"] = exp_config["iss_stitch_fliplr"]
        print(f"    -> NOTE: Setting ISS stitch + segment fliplr={exp_config['iss_stitch_fliplr']}")
    if "iss_stitch_rot90" in exp_config:
        for p in iss_stitch_params:
            config[p]["rot90"] = exp_config["iss_stitch_rot90"]
        print(f"    -> NOTE: Setting ISS stitch + segment rot90={exp_config['iss_stitch_rot90']}")

    # Apply link_phenotyping_params overrides (e.g., cy5_fliplr, phase_rot90, etc.)
    if "link_phenotyping_params" in exp_config:
        pheno_overrides = exp_config["link_phenotyping_params"]
        for key, value in pheno_overrides.items():
            config["link_phenotyping_params"][key] = value
        params_str = ", ".join(f"{k}={v}" for k, v in pheno_overrides.items())
        print(f"    -> NOTE: Setting link_phenotyping_params: {{{params_str}}}")


def _apply_experiment_number_overrides(
    experiment_name: str, config: dict
) -> None:
    """Apply overrides based on experiment number (from hardcoded lists)."""
    exp_number = _get_exp_number(experiment_name)
    if exp_number is None:
        return

    # GFP fliplr for older experiments
    if exp_number <= 33 or exp_number in GFP_FLIPLR_EXPS:
        config["link_phenotyping_params"]["gfp_fliplr"] = False
        print(f"    -> NOTE: Setting gfp_fliplr=False for experiment '{experiment_name}'")

    # ISS missing 9th round (use 9 rounds instead of 10)
    if exp_number in ISS_MISSING_NINTH_ROUND:
        iss_rounds = list(range(0, 9))
        config["base_calling_params"]["iss_rounds"] = iss_rounds
        config["metrics_params"]["iss_rounds"] = list(iss_rounds)
        config["link_calls_tracks_params"]["iss_rounds"] = list(iss_rounds)
        print(f"    -> NOTE: Setting iss_rounds=[0-8] (missing 9th round)")

    # Skip tracking experiments
    if exp_number in SKIP_TRACK_EXPS:
        config["auto_register_params"]["skip_track"] = True
        config["track_params"]["skip_track"] = True
        config["link_calls_tracks_params"]["skip_track"] = True
        print(f"    -> NOTE: Setting skip_track=True")

    # ISS only experiments
    if exp_number in ISS_ONLY_EXPERIMENTS:
        config["run_iss_only"] = True
        print(f"    -> NOTE: Setting run_iss_only=True")

    # Pre-nuclei round experiments (explicit list OR exp_number > 99, unless excluded)
    if exp_number not in NO_PRE_NUCLEI_ROUND_EXPS and (exp_number in PRE_NUCLEI_ROUND_EXPS or exp_number > 99):
        config["stack_symlinks_params"]["pre_nuclei_round"] = True
        print(f"    -> NOTE: Setting pre_nuclei_round=True")

    # Experiments with a pre-DAPI round that should be skipped entirely (no DAPI copying)
    if exp_number in SKIP_PRE_DAPI_ROUND_EXPS or experiment_name in SKIP_PRE_DAPI_ROUND_EXPS:
        config["stack_symlinks_params"]["skip_pre_dapi_round"] = True
        config["stack_symlinks_params"]["pre_nuclei_round"] = False
        print(f"    -> NOTE: Setting skip_pre_dapi_round=True (pre_nuclei_round overridden to False)")


def _sync_iss_params(config: dict) -> None:
    """Sync iss_rounds and failed_rounds_by_well across all relevant params."""
    # Sync iss_rounds
    if "iss_rounds" in config.get("base_calling_params", {}):
        iss_rounds = config["base_calling_params"]["iss_rounds"]
        config["metrics_params"]["iss_rounds"] = list(iss_rounds)
        config["link_calls_tracks_params"]["iss_rounds"] = list(iss_rounds)

    # Sync failed_rounds_by_well
    if "failed_rounds_by_well" in config.get("base_calling_params", {}):
        failed_rounds = config["base_calling_params"]["failed_rounds_by_well"]
        if failed_rounds is not None:
            config["link_calls_tracks_params"]["failed_rounds_by_well"] = deepcopy(failed_rounds)


def _apply_per_experiment_overrides(
    experiment_name: str, base_config: dict, ops_channel_maps: dict, ops_failed_rounds: dict,
    ops_library_map: dict | None = None,
    print_header: bool = False
) -> dict:
    """Return a new config dict with per-experiment overrides applied.

    Args:
        experiment_name: Name of the experiment
        base_config: Base configuration template
        ops_channel_maps: Channel maps loaded from YAML
        ops_failed_rounds: Failed rounds loaded from YAML
        ops_library_map: Library/codebook map loaded from YAML
        print_header: If True, print the experiment name header before notes
    """
    config = deepcopy(base_config)
    config["experiment_name"] = experiment_name

    # Print experiment name first if requested
    if print_header:
        print(f"--> {experiment_name}")

    # Apply channel map from ops_channel_maps.yaml
    try:
        _apply_channel_map_override(experiment_name, config, ops_channel_maps)
    except Exception as e:
        print(f"    -> WARNING: Failed to apply channel_map override: {e}")

    # Apply failed rounds/wells/iss_rounds from ops_failed_rounds.yaml
    try:
        _apply_failed_rounds_override(experiment_name, config, ops_failed_rounds)
    except Exception as e:
        print(f"    -> WARNING: Failed to apply failed_rounds override: {e}")

    # Apply library/codebook overrides from ops_library_map.yaml
    if ops_library_map:
        try:
            _apply_library_override(experiment_name, config, ops_library_map)
        except Exception as e:
            print(f"    -> WARNING: Failed to apply library override: {e}")

    # Apply overrides based on experiment number (hardcoded lists)
    try:
        _apply_experiment_number_overrides(experiment_name, config)
    except Exception as e:
        print(f"    -> WARNING: Failed to apply experiment number overrides: {e}")

    # Sync iss_rounds and failed_rounds across params
    _sync_iss_params(config)

    # Auto-derive project tag if not explicitly set
    if not config.get("project"):
        try:
            config["project"] = derive_project(experiment_name)
            print(f"    -> NOTE: Setting project={config['project']}")
        except Exception as e:
            print(f"    -> WARNING: Failed to derive project: {e}")

    # Auto-derive experimental_design (imaging modality) if not explicitly set
    if not config.get("experimental_design"):
        try:
            config["experimental_design"] = derive_experimental_design(experiment_name)
            print(f"    -> NOTE: Setting experimental_design={config['experimental_design']}")
        except Exception as e:
            print(f"    -> WARNING: Failed to derive experimental_design: {e}")

    # Tag designs that aren't derivable from channel/library configs (CROPseq,
    # plus Cell_Painting experiments missing the channel-map flag).
    exp_number = _get_exp_number(experiment_name)
    extra_designs = []
    if exp_number in CROPSEQ_EXPERIMENTS:
        extra_designs.append("CROPseq")
    if exp_number in CELL_PAINTING_EXPERIMENTS:
        extra_designs.append("Cell_Painting")
    if extra_designs:
        designs = config.get("experimental_design") or []
        if isinstance(designs, str):
            designs = [designs]
        added = [d for d in extra_designs if d not in designs]
        if added:
            designs.extend(added)
            config["experimental_design"] = designs
            print(f"    -> NOTE: Adding {added} to experimental_design={designs}")

    # Print empty line after notes if header was printed
    if print_header:
        print()

    return config


# ============================================================================
# Config File Generation
# ============================================================================

def generate_config_for_experiment(
    experiment_name: str, ask_overwrite: bool = True
) -> bool:
    """Generate a single config file for a specific experiment.

    Returns True on success, False on skip/abort.
    """
    dataset = OpsDataset(experiment_name)
    config_path = dataset.config_paths["exp_config"]

    # Ensure parent dir exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    if config_path.exists() and ask_overwrite:
        resp = (
            input(
                f"Config for '{experiment_name}' already exists at {config_path}. Overwrite? [y/N]: "
            )
            .strip()
            .lower()
        )
        if resp not in ("y", "yes"):
            print(f"--> Skipping generation for '{experiment_name}' (user declined overwrite)")
            return False

    ops_channel_maps = _load_ops_channel_maps()
    ops_failed_rounds = _load_ops_failed_rounds()
    ops_library_map = _load_ops_library_map()

    new_config = _apply_per_experiment_overrides(
        experiment_name, DEFAULT_CONFIG, ops_channel_maps, ops_failed_rounds,
        ops_library_map=ops_library_map,
        print_header=True
    )

    try:
        with open(config_path, "w") as f:
            yaml.dump(new_config, f, sort_keys=False, default_flow_style=False)
        return True
    except Exception as e:
        print(f"--> ERROR: Could not write config for '{experiment_name}'. Reason: {e}")
        return False


def generate_config_files():
    """
    Finds all OPS experiments and generates a default configuration file for each one.
    This will backup and clear the existing config directory before generating new files.
    """
    base_path = Path(f"{BASE_PATH}/")
    all_ops_folders = sorted([p for p in base_path.glob("ops0*") if p.is_dir()])

    experiment_folders = []
    for p in all_ops_folders:
        exp_number = _get_exp_number(p.name)
        if exp_number is not None and exp_number >= 5:
            experiment_folders.append(p)

    if not experiment_folders:
        print("No experiment folders found with experiment number >= 5. Aborting.")
        return

    # Get the target directory from a dummy dataset object
    dummy_dataset = OpsDataset("dummy")
    config_root_dir = dummy_dataset.config_paths["exp_config_dir"]

    # Backup and clear existing configs before generating new ones
    backup_config_files(config_root_dir)
    if config_root_dir.exists():
        try:
            shutil.rmtree(config_root_dir)
        except Exception as e:
            print(f"--- ERROR: Could not clear config directory. Reason: {e} ---")
            return
    config_root_dir.mkdir(exist_ok=True)

    print(
        f"Found {len(experiment_folders)} experiments to process. "
        f"Generating config files in '{config_root_dir}'..."
    )

    generated_count = 0
    skipped_count = 0

    # Load config files once
    ops_channel_maps = _load_ops_channel_maps()
    ops_failed_rounds = _load_ops_failed_rounds()
    ops_library_map = _load_ops_library_map()

    for exp_folder in experiment_folders:
        experiment_name = exp_folder.name

        # Check if experiment should be skipped
        exp_number = _get_exp_number(experiment_name)
        if exp_number is not None and exp_number in DO_NOT_RUN_EXPERIMENTS:
            print(f"--> Skipping '{experiment_name}': Experiment is in DO_NOT_RUN_EXPERIMENTS list.")
            skipped_count += 1
            continue

        # Generate and write config
        try:
            new_config = _apply_per_experiment_overrides(
                experiment_name, DEFAULT_CONFIG, ops_channel_maps, ops_failed_rounds,
                ops_library_map=ops_library_map,
                print_header=True
            )
            config_path = OpsDataset(experiment_name).config_paths["exp_config"]
            with open(config_path, "w") as f:
                yaml.dump(new_config, f, sort_keys=False, default_flow_style=False)
            generated_count += 1
        except Exception as e:
            print(f"--> ERROR: Could not write config for '{experiment_name}'. Reason: {e}")

    print("\n--- Generation Complete ---")
    print(f"Generated: {generated_count} new config files.")
    print(f"Skipped:   {skipped_count} existing config files.")


if __name__ == "__main__":
    # Build list of available groups for help text
    available_groups = ", ".join(f"'{g}'" for g in EXPERIMENT_GROUPS.keys())

    parser = argparse.ArgumentParser(
        description="Generate default config files for all OPS experiments.",
        epilog=f"Available experiment groups: {available_groups}",
    )
    parser.add_argument(
        "-e",
        "--exp",
        "--experiment",
        dest="experiment",
        type=str,
        help=f"Generate config(s) for a single experiment or an experiment group. "
        f"Use experiment name (e.g., ops0030_20250422) or group name (e.g., blnk_exps).",
    )
    # Positional experiment name (first non-flag arg) fallback
    parser.add_argument(
        "experiment_positional",
        nargs="?",
        help="Experiment name or group name (positional). Used if -e/--experiment is not provided.",
    )
    args = parser.parse_args()

    # If no -e/--experiment provided, but a first positional arg exists, use it
    if not args.experiment and getattr(args, "experiment_positional", None):
        args.experiment = args.experiment_positional

    # Always sync local configs to HPC before generating
    sync_all_configs(prompt=True)

    if args.experiment:
        # Check if it's an experiment group
        if args.experiment in EXPERIMENT_GROUPS:
            group_name = args.experiment
            experiment_list = EXPERIMENT_GROUPS[group_name]
            print(f"\n=== Generating configs for experiment group '{group_name}' ===")
            print(f"Found {len(experiment_list)} experiments in group:")
            for exp in experiment_list:
                print(f"  - {exp}")
            print()

            generated_count = 0
            failed_count = 0

            for exp_name in experiment_list:
                try:
                    ok = generate_config_for_experiment(exp_name, ask_overwrite=False)
                    if ok:
                        generated_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    print(f"--> ERROR: Failed to generate config for '{exp_name}': {e}")
                    failed_count += 1

            print(f"\n--- Generation Complete (experiment group: {group_name}) ---")
            print(f"Successfully generated: {generated_count}/{len(experiment_list)}")
            if failed_count > 0:
                print(f"Failed: {failed_count}/{len(experiment_list)}")
        else:
            # Single-experiment mode
            resolved_name = resolve_experiment_name(args.experiment, allow_interactive=True)
            ok = generate_config_for_experiment(resolved_name, ask_overwrite=True)
            if ok:
                print("\n--- Generation Complete (single experiment) ---")
            else:
                print("\n--- No changes made (single experiment) ---")
    else:
        generate_config_files()

    # Usage examples:
    # Generate all configs:
    #   python cyclops_process/pipelinerunner/generate_config_files.py
    # Generate single experiment:
    #   python cyclops_process/pipelinerunner/generate_config_files.py ops0062_20250729
    #   python cyclops_process/pipelinerunner/generate_config_files.py --experiment ops0062_20250729
    # Generate experiment group:
    #   python cyclops_process/pipelinerunner/generate_config_files.py blnk_exps
    #   python cyclops_process/pipelinerunner/generate_config_files.py ops0079_materials
