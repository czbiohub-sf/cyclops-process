import sys
import os
import csv
from pathlib import Path
from datetime import datetime
import yaml
from prettytable import PrettyTable
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ops_utils.data.experiment import OpsDataset
from cyclops_process.processes import (
    segment,
    spots,
)
from cyclops_process.metrics import metrics
from ops_utils.io.zarr_utils import has_fluorescence_channels_from_config
from ops_utils.data.bad_experiments import (
    KNOWN_EXPERIMENTAL_DESIGNS,
    KNOWN_PROJECTS,
    count_codebook_perturbations,
    derive_experimental_design,
    derive_library,
    derive_project,
    get_category,
    get_date_cutoff,
    get_experiment_tag,
    get_paper_v1_experiments,
    get_paper_v2_experiments,
    get_reason,
    is_bad_channel,
    is_excluded,
    is_in_paper_v1,
    is_in_paper_v2,
)
from cyclops_process.paths import BASE_PATH

# No local source of truth here: we rely on OpsDataset.get_all_step_keys()

# Experiment exclusion lists — loaded from shared bad_experiment.yaml
POSITIVE_CONTROL_EXPERIMENTS = get_category("positive_control")
NEED_RESCUE_EXPERIMENTS = get_category("need_rescue")

# Cache configuration
CACHE_VERSION = "2.1"  # Increment when cache structure changes (v2.1: partition fallback in cache hash)
CACHE_FILE_NAME = ".pipeline_status_cache.json"

# Partition paths for old/new fallback logic
_NEW_BASE = f"{BASE_PATH}/"
_OLD_BASE = f"{BASE_PATH}/"


def _get_fallback_path(p: Path) -> Path | None:
    """If path is on the new partition, return equivalent old partition path."""
    ps = str(p)
    if ps.startswith(_NEW_BASE):
        return Path(ps.replace(_NEW_BASE, _OLD_BASE, 1))
    return None


def _get_cache_path() -> Path:
    """Get the path to the cache file."""
    return Path(__file__).resolve().parent / CACHE_FILE_NAME


def _load_cache() -> tuple[dict, bool]:
    """
    Load the cache from disk.

    Returns:
        tuple: (cache_dict, is_read_only)
        - cache_dict: The loaded cache or empty cache structure
        - is_read_only: True if cache file exists but we can't write to it
    """
    cache_path = _get_cache_path()

    if not cache_path.exists():
        # Check if we can write to the directory
        try:
            cache_path.touch()
            cache_path.unlink()
            return {"version": CACHE_VERSION, "experiments": {}}, False
        except (PermissionError, OSError):
            return {"version": CACHE_VERSION, "experiments": {}}, True

    # Check write access to existing file
    is_read_only = not os.access(cache_path, os.W_OK)

    try:
        with open(cache_path, "r") as f:
            cache = json.load(f)

        # Invalidate cache if version mismatch
        if cache.get("version") != CACHE_VERSION:
            if is_read_only:
                # Can't update cache, but still use what we can
                print(f"Warning: Cache version mismatch but no write access to update")
            return {"version": CACHE_VERSION, "experiments": {}}, is_read_only

        return cache, is_read_only
    except Exception as e:
        print(f"Warning: Failed to load cache: {e}")
        return {"version": CACHE_VERSION, "experiments": {}}, is_read_only


def _save_cache(cache: dict, is_read_only: bool = False):
    """
    Save the cache to disk.

    Args:
        cache: The cache dictionary to save
        is_read_only: If True, skip saving (no write access)
    """
    if is_read_only:
        return

    cache_path = _get_cache_path()
    cache["version"] = CACHE_VERSION
    cache["last_updated"] = datetime.now().isoformat()

    try:
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
    except (PermissionError, OSError) as e:
        print(f"Warning: Failed to save cache (no write access): {e}")
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")


def _compute_file_hash(file_path: Path, use_content: bool = False) -> str:
    """
    Compute a hash for a file based on content or metadata.
    Tries the given path first, then falls back to the old partition if missing.

    Args:
        file_path: Path to the file
        use_content: If True, hash file content (slower but accurate).
                     If False, use inode + size (faster, detects most changes).

    Returns:
        Hash string representing file state, or "missing" if file doesn't exist.
    """
    def _hash_single(p: Path) -> str | None:
        """Hash a single path. Returns None if file doesn't exist."""
        try:
            if not p.exists():
                return None
            stat = p.stat()
            if use_content:
                hasher = hashlib.md5()
                with open(p, 'rb') as f:
                    for chunk in iter(lambda: f.read(8192), b''):
                        hasher.update(chunk)
                return hasher.hexdigest()
            else:
                return f"{stat.st_ino}:{stat.st_size}"
        except Exception as e:
            return f"error:{e}"

    result = _hash_single(file_path)
    if result is not None:
        return result

    # Fallback to old partition
    fallback = _get_fallback_path(file_path)
    if fallback is not None:
        result = _hash_single(fallback)
        if result is not None:
            return result

    return "missing"


def _compute_experiment_hash(dataset: OpsDataset, config: dict, config_path: Path) -> str:
    """
    Compute a hash for an experiment based on key files that indicate state changes.

    The hash is based on:
    1. Config file content (MD5 of actual content)
    2. Key output files using inode + size:
       - plate_stats.csv (ISS results)
       - linked_pheno_iss.csv files (tracked cells)
       - Results directory structure

    This approach is more reliable than mtime-based hashing because:
    - Config content hash: Only changes when config actually changes (not just opened)
    - Inode + size: Detects file replacement and size changes without reading entire files
    """
    hash_inputs = []

    # 1. Config file content hash (MD5 of actual YAML content)
    # This ensures cache invalidation only when config actually changes
    try:
        hash_inputs.append(f"config:{_compute_file_hash(config_path, use_content=True)}")
    except Exception:
        hash_inputs.append("config:missing")

    # 2. ISS plate_stats.csv (indicates ISS completion)
    # Use inode+size for performance (file rarely modified in-place)
    try:
        stats_path = dataset.results / "ISS" / "mine" / "plate_stats.csv"
        hash_inputs.append(f"plate_stats:{_compute_file_hash(stats_path)}")
    except Exception:
        hash_inputs.append("plate_stats:error")

    # 3. Check for linked_pheno_iss.csv files (indicates tracking completion)
    # Sample first 6 wells for performance, use inode+size
    try:
        wells = config.get("wells_to_process", []) or dataset.infer_wells()
        linked_files_info = []
        for well in wells[:6]:  # Sample first 6 wells for performance
            parts = well.split("/")
            if len(parts) >= 2:
                well_short = f"{parts[0]}{parts[1]}"
                linked_path = dataset.results_fast / f"{well_short}_linked_pheno_iss.csv"
                file_hash = _compute_file_hash(linked_path)
                linked_files_info.append(f"{well_short}:{file_hash}")
        hash_inputs.append(f"linked:{','.join(linked_files_info)}")
    except Exception:
        hash_inputs.append("linked:error")

    # 4. Check key pipeline output directories
    # Count files as a proxy for completion state (try new partition, fallback to old)
    key_dirs = [
        dataset.results / "ISS",
        dataset.pheno_results if hasattr(dataset, 'pheno_results') else None,
    ]
    for d in key_dirs:
        if d is None:
            continue
        # Try the directory itself, fall back to old partition
        check_dir = d
        if not check_dir.exists():
            fallback_dir = _get_fallback_path(d)
            if fallback_dir is not None and fallback_dir.exists():
                check_dir = fallback_dir
        if check_dir.exists():
            try:
                file_count = len(list(check_dir.glob("*")))
                hash_inputs.append(f"{d.name}:{file_count}")
            except Exception:
                hash_inputs.append(f"{d.name}:error")

    # 5. Include run_iss_only flag as it affects pipeline steps
    hash_inputs.append(f"iss_only:{config.get('run_iss_only', False)}")

    # Compute final hash
    hash_string = "|".join(hash_inputs)
    return hashlib.md5(hash_string.encode()).hexdigest()


def _get_cached_experiment_data(
    cache: dict,
    exp_name: str,
    current_hash: str
) -> dict | None:
    """
    Get cached data for an experiment if the hash matches.

    Returns:
        Cached data dict if hash matches, None if cache miss or hash mismatch.
    """
    exp_cache = cache.get("experiments", {}).get(exp_name)
    if exp_cache and exp_cache.get("hash") == current_hash:
        return exp_cache.get("data")
    return None


def _update_experiment_cache(
    cache: dict,
    exp_name: str,
    exp_hash: str,
    data: dict
):
    """Update the cache with new experiment data."""
    if "experiments" not in cache:
        cache["experiments"] = {}

    cache["experiments"][exp_name] = {
        "hash": exp_hash,
        "data": data,
        "cached_at": datetime.now().isoformat(),
    }


class DummyGraphGeneratorRun:
    __qualname__ = "DummyGraphGeneratorRun"
    __name__ = "DummyGraphGeneratorRun"


# Explain the pipeline status report:
# The pipeline status report helps track progress across pipeline steps per experiment.
# It checks for expected output files (as defined in `cyclops_process.data.experiment.OpsDataset.get_output_files_for_step`).
# Steps with existing expected files are marked as verified-complete; steps with no expected files are tracked as skipped (names listed).
# The "Last Complete Step" reflects the most recent step verified by files (skipped steps do not advance completion).
# The "Next Step" is the first step that is incomplete (expected files missing). If none are missing, the pipeline is Complete.


def _normalize_well_token(w: str) -> str:
    return str(w).replace("/", "_")


def _get_color_for_progress(progress_pct: int) -> tuple[str, str]:
    """Return ANSI color codes for progress percentage.

    Returns:
        tuple: (color_code, reset_code)

    0-25%: Red
    26-50%: Orange/Yellow
    51-75%: Yellow
    76-100%: Green
    """
    # ANSI color codes
    RED = "\033[91m"
    ORANGE = "\033[93m"  # Yellow in most terminals
    YELLOW = "\033[33m"
    GREEN = "\033[92m"
    RESET = "\033[0m"

    if progress_pct <= 25:
        return RED, RESET
    elif progress_pct <= 50:
        return ORANGE, RESET
    elif progress_pct <= 75:
        return YELLOW, RESET
    else:
        return GREEN, RESET


def _colorize_progress(progress_pct: int) -> str:
    """Return color-coded progress percentage string based on value."""
    color, reset = _get_color_for_progress(progress_pct)
    return f"{color}{progress_pct}%{reset}"


def _colorize_step(step_str: str, progress_pct: int) -> str:
    """Return color-coded step string based on progress percentage."""
    color, reset = _get_color_for_progress(progress_pct)
    return f"{color}{step_str}{reset}"


def _check_manual_fluor_registration(dataset: OpsDataset, config: dict) -> bool:
    """Check if manual fluorescence registration files exist."""
    from ops_utils.io.zarr_utils import has_fluorescence_channels_from_config

    if not has_fluorescence_channels_from_config(config):
        return True  # N/A, consider as "complete"

    gfp_yaml = dataset.config_paths.get("lc_GFP_register")
    mch_yaml = dataset.config_paths.get("lc_mCherry_register")
    has_manual_fluor = bool(gfp_yaml and gfp_yaml.exists()) or bool(
        mch_yaml and mch_yaml.exists()
    )
    return has_manual_fluor


def _check_manual_iss_pheno_registration(
    dataset: OpsDataset, config: dict
) -> tuple[bool, int, int]:
    """Check if ISS→Pheno registration files exist for all wells.

    Returns:
        tuple: (all_present, count_present, total_wells)
    """
    wells = config.get("wells_to_process", []) or dataset.infer_wells()
    if not wells:
        return True, 0, 0

    count_present = 0
    for w in wells:
        path_yml = dataset.append_well("iss_seg_register", w)
        path_yaml = path_yml.with_suffix(".yaml")
        if path_yml.exists() or path_yaml.exists():
            count_present += 1

    return count_present == len(wells), count_present, len(wells)


def _emoji_for_step(step_key: str):
    """Return a category emoji for a given step key."""
    s = (step_key or "").lower()
    # Manual checkpoints
    if "manual" in s and "registration" in s:
        return "🔧"
    # Feature extraction
    if "morphology" in s or "organelle" in s or "feature" in s or "graph" in s:
        return "📊"
    # Tracking
    if (
        s == "stack"
        or "track_well" in s
        or "apply_registration_tracking" in s
        or "labels_to_contours" in s
        or "_track" in s
        or s.startswith("track_")
    ):
        return "👣"
    # Phenotyping
    if (
        "pheno" in s
        or "phenotyping" in s
        or "virtual_staining" in s
        or "segmentation_pheno_cells" in s
    ):
        return "🦠"
    # ISS
    if (
        "iss" in s
        or "detect_spots" in s
        or "base_calling" in s
        or "metrics" in s
        or "stack_symlinks" in s
    ):
        return "🧬"
    return ""


def _get_step_state_by_key(
    dataset: OpsDataset,
    log_key: str,
    config: dict,
    fast: bool = False,
    method: str | None = None,
):
    """
    Determine the state of a pipeline step based on expected output files.

    Returns a tuple: (state, log_key)
      - state: one of {"complete", "incomplete", "skipped"}
      - log_key: the normalized step key used across the report
    """
    dataset_for_check = (
        OpsDataset(dataset.experiment, config, method=str(method))
        if method
        else dataset
    )
    output_files = dataset_for_check.get_output_files_for_step(log_key, config)

    # If a step is not defined in get_output_files_for_step, or explicitly maps to
    # an empty list (no concrete file to check), treat it as "skipped" (not counted
    # towards last verified-complete step).
    if not output_files:  # None or empty list
        return "skipped", log_key

    # If we have one or more files, check existence (fast) or existence+content (default)
    def _check_path(p: Path) -> bool:
        """Check if a single path has content."""
        if fast:
            return p.exists()
        try:
            if not p.exists():
                return False
            if p.is_file():
                return p.stat().st_size > 0
            if p.is_dir():
                if p.suffix == ".zarr" or p.name.endswith(".zarr"):
                    return (p / ".zgroup").exists() or (p / ".zarray").exists()
                return next(p.iterdir(), None) is not None
            return False
        except Exception:
            return False

    def _has_content(p: Path) -> bool:
        """Check new partition first, fall back to old partition (icd.ops)."""
        if _check_path(p):
            return True
        fallback = _get_fallback_path(p)
        if fallback is not None:
            return _check_path(fallback)
        return False

    is_complete = all(_has_content(f) for f in output_files)
    # Append method suffix for display to distinguish variants when needed
    display_key = f"{log_key}_{method}" if method else log_key
    return ("complete" if is_complete else "incomplete"), display_key


def _format_number_ranges(numbers: list[int]) -> str:
    """
    Format a list of numbers into compact ranges without spaces.
    Example: [1,2,3,4,5,7,8,10] -> "1-5,7-8,10"
    """
    if not numbers:
        return ""

    sorted_nums = sorted(set(numbers))
    ranges = []
    start = sorted_nums[0]
    end = sorted_nums[0]

    for num in sorted_nums[1:]:
        if num == end + 1:
            end = num
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = num
            end = num

    # Add the last range
    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")

    return ",".join(ranges)


def _extract_exp_number(exp_id: str) -> int | None:
    """
    Extract experiment number from ID, handling both 'ops' and 'op' prefixes.
    Examples: ops0033 -> 33, op0072 -> 72, ops0108 -> 108
    """
    # Match both 'ops' and 'op' prefixes followed by digits
    match = re.match(r"op[s]?0*(\d+)", exp_id)
    if match:
        return int(match.group(1))
    return None


def _format_exp_ids_as_ranges(exp_ids: list[str], max_individual: int = 5) -> str:
    """
    Format experiment IDs as shorthand numbers, using ranges if more than max_individual.

    Args:
        exp_ids: List of experiment IDs (e.g., ['ops0033', 'ops0064', 'op0072'])
        max_individual: If more than this many experiments, use range format

    Returns:
        Formatted string like "33,64,72" or "3-113" for many experiments
    """
    if not exp_ids:
        return ""

    # Extract numbers from all experiment IDs
    numbers = []
    for exp_id in exp_ids:
        num = _extract_exp_number(exp_id)
        if num is not None:
            numbers.append(num)

    if not numbers:
        return ",".join(exp_ids)  # Fallback to original IDs

    # Sort and deduplicate
    numbers = sorted(set(numbers))

    # If few experiments, list them individually
    if len(numbers) <= max_individual:
        return ",".join(str(n) for n in numbers)

    # Otherwise, use range format
    return _format_number_ranges(numbers)


def _get_iss_total_cells(dataset: OpsDataset, experiment: str, debug: bool = False) -> tuple[int, int, int]:
    """
    Get total cells, cells with reads (spots), and cells with matched reads from ISS plate_stats.csv.
    Returns tuple of (total_cells, cells_with_reads, cells_with_matched_reads) summed across all wells.

    The distinction:
    - total_cells: All DAPI segmented cells
    - cells_with_reads: Cells that have spots/barcodes assigned to them (may not match codebook)
    - cells_with_matched_reads: Cells with reads that match the codebook (ISS matched)
    """
    try:
        # dataset.results_iss already points to 3-assembly/ISS/mine or /ISS/prob
        # We need to use the mine version specifically
        stats_path = dataset.results / "ISS" / "mine" / "plate_stats.csv"

        if debug:
            print(f"[ISS Cells] {experiment}: Checking {stats_path}")

        if not stats_path.exists():
            if debug:
                print(f"[ISS Cells] {experiment}: ❌ File not found")
            return None, None, None

        stats_df = pd.read_csv(stats_path, index_col=0)

        # Get total cells (num_cells row)
        total_cells = None
        if "num_cells" in stats_df.index:
            total_cells = int(stats_df.loc["num_cells"].sum())
            if debug:
                print(f"[ISS Cells] {experiment}: ✅ Found {total_cells:,} total cells")
        elif debug:
            print(f"[ISS Cells] {experiment}: ❌ 'num_cells' row not found")

        # Get cells with reads (spots assigned, may not match codebook)
        cells_with_reads = None
        if "cells_with_reads" in stats_df.index:
            cells_with_reads = int(stats_df.loc["cells_with_reads"].sum())
            if debug:
                print(f"[ISS Cells] {experiment}: ✅ Found {cells_with_reads:,} cells with reads")
        elif debug:
            print(f"[ISS Cells] {experiment}: ❌ 'cells_with_reads' row not found")

        # Get cells with matched reads (match to codebook)
        matched_cells = None
        if "cells_with_matched_reads" in stats_df.index:
            matched_cells = int(stats_df.loc["cells_with_matched_reads"].sum())
            if debug:
                print(f"[ISS Cells] {experiment}: ✅ Found {matched_cells:,} matched cells")
        elif debug:
            print(f"[ISS Cells] {experiment}: ❌ 'cells_with_matched_reads' row not found")

        return total_cells, cells_with_reads, matched_cells
    except Exception as e:
        if debug:
            print(f"[ISS Cells] {experiment}: ❌ Error: {e}")
        return None, None, None


def _extract_experiment_id(experiment_name: str) -> str:
    """
    Extract experiment ID from full experiment name.
    Example: 'ops0061_20250728' -> 'ops0061'
    """
    match = re.match(r"^(ops\d{4})_\d{8}$", experiment_name)
    return match.group(1) if match else None


def _load_channel_maps(config_path: Path = None) -> dict:
    """
    Load channel maps YAML and return mapping of experiment_id -> list of markers.
    Each marker is a dict with 'channel_name' and 'label'.
    """
    if config_path is None:
        config_path = Path(os.environ.get("OPS_CONFIGS_DIR", f"{BASE_PATH}/configs")) / "ops_channel_maps.yaml"

    if not config_path.exists():
        return {}

    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def _normalize_marker_label(label: str) -> str:
    """Normalize marker label for case-insensitive matching."""
    return label.strip().lower()


def _get_unique_markers_from_channel_maps(channel_maps: dict) -> dict:
    """
    Extract unique markers from channel maps.
    Returns dict mapping marker_label -> {"exp_ids": list, "channel_name": str}.
    Skips empty labels, 'no label', 'empty, no label', and all 'bleedthrough' variants.
    Only includes 'autofluorescence, no label' (as a real marker).
    Matching is case-insensitive (e.g., MAP1LC3B and MAP1LC3b are treated as the same).
    """
    marker_to_info = {}
    # Maps normalized label -> display label (preserves first-seen capitalization)
    normalized_to_display = {}

    for exp_id, channels in channel_maps.items():
        if not isinstance(channels, list):
            continue
        for channel in channels:
            label = channel.get("label", "")
            channel_name = channel.get("channel_name", "")
            # Skip empty labels
            if label == "":
                continue
            # Skip bad channels: non-informative labels (no label / bleedthrough /
            # autofluorescence) and explicit (experiment, channel) exclusions in
            # bad_experiment.yaml's bad_channels (e.g. ops0101 FeRhoNox — saturated).
            if is_bad_channel(exp_id, channel_name, label):
                continue

            # Normalize for case-insensitive matching
            normalized = _normalize_marker_label(label)

            # Use first-seen capitalization as display label
            if normalized not in normalized_to_display:
                normalized_to_display[normalized] = label

            display_label = normalized_to_display[normalized]

            if display_label not in marker_to_info:
                marker_to_info[display_label] = {"exp_ids": [], "channel_name": channel_name}
            # Dedupe: a marker that appears in multiple channels of the same
            # experiment (e.g. CP1_nuclei_Hoechst + CP2_nuclei_Hoechst in
            # ops0094) must only contribute that experiment once — otherwise
            # downstream cell-count aggregation double-counts the experiment.
            if exp_id not in marker_to_info[display_label]["exp_ids"]:
                marker_to_info[display_label]["exp_ids"].append(exp_id)

    return marker_to_info


def _get_channel_indicator(channel_name: str, marker_label: str = "") -> str:
    """
    Return a colored dot indicator based on channel name (and optional marker label).

    🟢 Green   = GFP
    🔴 Red     = mCherry
    🟠 Orange  = farred / Cy5
    ⚫ Gray    = BF / Phase (label-free)
    🟣 Purple  = Cell Painting channels (``CP1_*``, ``CP2_*``)
    🔵 Blue    = 4i channels (``4i_R*_*``)
    🟡 Yellow  = mScarlet reporter markers (on mCherry, but disambiguated from
                 regular mCherry via the marker label)
    """
    ch = channel_name.lower()
    label = (marker_label or "").lower()

    # Modality-specific overrides take precedence so reporter/CP/4i markers don't
    # collapse into the generic GFP/mCherry/Cy5 buckets.
    if "mscarlet" in label:
        return "🟡"
    if ch.startswith("cp1_") or ch.startswith("cp2_"):
        return "🟣"
    if ch.startswith("4i_"):
        return "🔵"

    if ch == "gfp":
        return "🟢"
    elif ch == "mcherry":
        return "🔴"
    elif ch in ("farred", "cy5"):
        return "🟠"
    elif ch in ("bf", "phase"):
        return "⚫"
    return ""


def _get_tracked_cell_stats(
    dataset: OpsDataset, config: dict, experiment: str, debug: bool = False
) -> tuple[int, float, int, int]:
    """
    Read each well's linked_pheno_iss.csv once and derive all tracked-cell stats.

    This replaces three separate full-file passes (total cells, mean cells/gene,
    segmentation breakdown) with a single read per well, restricted to the only
    columns those metrics need. On a shared network filesystem the CSV reads
    dominate runtime, so collapsing 3 reads → 1 is the main speedup.

    Returns tuple of (total_cells, mean_cells_per_gene, full_seg_cells, fallback_seg_cells):
    - total_cells: sum of rows across all well CSVs
    - mean_cells_per_gene: mean cell count per gene across all wells combined
    - full_seg_cells: cells with valid segmentation_id (not NaN)
    - fallback_seg_cells: cells with NaN segmentation_id (200px fallback bbox)
    Each component is None when there's no data for it (matches prior helpers).
    """
    gene_candidates = ("dep_map_gene_name", "Gene name", "gene_name")
    wanted = set(gene_candidates) | {"segmentation_id"}

    try:
        wells = config.get("wells_to_process", []) or dataset.infer_wells()
        total_cells = 0
        full_seg_count = 0
        fallback_count = 0
        gene_counts = None  # pandas Series: gene -> cell count, summed across wells
        any_file = False

        for well in wells:
            parts = well.split("/")
            if len(parts) < 2:
                continue
            well_short = f"{parts[0]}{parts[1]}"
            linked_path = dataset.results_fast / f"{well_short}_linked_pheno_iss.csv"
            if not linked_path.exists():
                continue
            try:
                df = pd.read_csv(linked_path, usecols=lambda c: c in wanted)
            except Exception:
                continue

            any_file = True
            total_cells += len(df)

            if "segmentation_id" in df.columns:
                fallback_count += int(df["segmentation_id"].isna().sum())
                full_seg_count += int(df["segmentation_id"].notna().sum())

            gene_col = next((c for c in gene_candidates if c in df.columns), None)
            if gene_col is not None:
                counts = df[gene_col].value_counts()  # drops NaN, like groupby().size()
                gene_counts = (
                    counts if gene_counts is None
                    else gene_counts.add(counts, fill_value=0)
                )

        if not any_file:
            return None, None, None, None

        mean_cpg = (
            float(np.round(gene_counts.mean(), 2))
            if gene_counts is not None and len(gene_counts) > 0
            else None
        )

        if debug:
            print(
                f"[Tracked Cells] {experiment}: total={total_cells:,}, "
                f"mean_cells/gene={mean_cpg}, full_seg={full_seg_count:,}, "
                f"fallback={fallback_count:,}"
            )

        return (
            total_cells if total_cells > 0 else None,
            mean_cpg,
            full_seg_count if full_seg_count > 0 else None,
            fallback_count if fallback_count > 0 else None,
        )
    except Exception as e:
        if debug:
            print(f"[Tracked Cells] {experiment}: ❌ Error: {e}")
        return None, None, None, None


def _generate_marker_analysis(
    results: list,
    restrict_to_used_markers: bool = False,
    marker_filter=None,
) -> list:
    """
    Generate marker analysis by aggregating cell counts across experiments for each unique marker.

    Excludes experiments in POSITIVE_CONTROL_EXPERIMENTS and NEED_RESCUE_EXPERIMENTS from aggregation.

    Args:
        results: List of experiment result dicts with experiment_id, total_cells, etc.
        restrict_to_used_markers: When True, drop markers that have no experiment in
            ``results``. Useful for modality-specific tables (per-project, Cell
            Painting, 4i) where listing every channel-map marker would be noisy.
        marker_filter: Optional callable ``(marker_label, channel_name) -> bool``
            applied before aggregation. Used to silo live-cell vs fixed-cell
            (Cell Painting / 4i) channels into separate tables even though the
            same experiment (e.g. ops0094) appears in both modality buckets.

    Returns:
        List of dicts with marker statistics, sorted by total_cells descending.
    """
    # Load channel maps
    channel_maps = _load_channel_maps()
    if not channel_maps:
        return []

    # Get unique markers and their info (exp_ids and channel_name)
    marker_to_info = _get_unique_markers_from_channel_maps(channel_maps)
    if not marker_to_info:
        return []

    if marker_filter is not None:
        marker_to_info = {
            m: info for m, info in marker_to_info.items()
            if marker_filter(m, info.get("channel_name", ""))
        }
        if not marker_to_info:
            return []

    # Build experiment_id -> result mapping for quick lookup
    # Exclude positive control and need rescue experiments from marker aggregation
    excluded_exp_names = set(POSITIVE_CONTROL_EXPERIMENTS + NEED_RESCUE_EXPERIMENTS)
    exp_id_to_results = {}
    for r in results:
        # Skip if this experiment is in exclusion lists
        if r.get("experiment") in excluded_exp_names:
            continue

        exp_id = r.get("experiment_id")
        if exp_id:
            if exp_id not in exp_id_to_results:
                exp_id_to_results[exp_id] = []
            exp_id_to_results[exp_id].append(r)

    # Optionally drop markers that don't appear in any of the included experiments.
    if restrict_to_used_markers:
        marker_to_info = {
            m: info for m, info in marker_to_info.items()
            if any(eid in exp_id_to_results for eid in info["exp_ids"])
        }
        if not marker_to_info:
            return []

    # Aggregate stats per marker
    marker_stats = []
    for marker, info in marker_to_info.items():
        exp_ids_from_map = info["exp_ids"]
        channel_name = info["channel_name"]
        total_cells = 0
        iss_matched_cells = 0
        tracked_total_cells = 0
        sum_mean_cpg = 0
        experiment_count = 0
        included_exp_ids = []  # Only exp_ids that are in the filtered results

        for exp_id in exp_ids_from_map:
            # Check if we have results for this experiment ID (after filtering)
            if exp_id in exp_id_to_results:
                for r in exp_id_to_results[exp_id]:
                    experiment_count += 1
                    included_exp_ids.append(exp_id)
                    if r.get("total_cells"):
                        total_cells += r["total_cells"]
                    if r.get("iss_matched_cells"):
                        iss_matched_cells += r["iss_matched_cells"]
                    if r.get("tracked_total_cells"):
                        tracked_total_cells += r["tracked_total_cells"]
                    if r.get("mean_cells_per_gene") is not None:
                        sum_mean_cpg += r["mean_cells_per_gene"]

        # Include all markers from channel map, even those without experiments in filtered results
        # This helps identify which markers are missing data
        marker_stats.append({
            "marker": marker,
            "channel_name": channel_name,
            "channel_indicator": _get_channel_indicator(channel_name, marker),
            "experiment_count": experiment_count,
            "experiment_ids": list(set(included_exp_ids)),  # Dedupe and use only included exp_ids
            "total_cells": total_cells if total_cells > 0 else None,
            "iss_matched_cells": iss_matched_cells if iss_matched_cells > 0 else None,
            "tracked_total_cells": tracked_total_cells if tracked_total_cells > 0 else None,
            "mean_cells_per_gene": sum_mean_cpg if sum_mean_cpg > 0 else None,
        })

    # Sort by mean_cells_per_gene (sum of mean cells/gene) descending (None values at end)
    marker_stats.sort(key=lambda x: (x["mean_cells_per_gene"] or 0), reverse=True)

    return marker_stats


def _generate_project_summary(results: list, marker_analysis_results: list = None) -> list:
    """Aggregate per-project stats across all experiments.

    Returns a list of dicts (one per project) with experiment count, distinct
    cell lines, summed cell totals, the number of unique markers contributed
    by experiments in that project, and (when ``marker_analysis_results`` is
    supplied) the average ``Mean Cells/Perturbation`` across those
    markers.
    """
    if not results:
        return []

    channel_maps = _load_channel_maps()
    marker_to_info = _get_unique_markers_from_channel_maps(channel_maps) if channel_maps else {}

    # Build marker → sum-of-mean cells/perturbation lookup from the precomputed
    # marker analysis (excluding 'phase' which has no per-perturbation meaning).
    marker_to_sum_cpg: dict[str, float] = {}
    if marker_analysis_results:
        for m in marker_analysis_results:
            label = m.get("marker")
            if not label or _normalize_marker_label(label) == "phase":
                continue
            if m.get("mean_cells_per_gene"):
                marker_to_sum_cpg[label] = m["mean_cells_per_gene"]

    project_to_exp_ids: dict[str, set] = {}
    project_to_codebooks: dict[str, set] = {}
    project_rows: dict[str, dict] = {}

    for r in results:
        # An experiment contributes to its canonical project AND, if it's in
        # the curated paper_v1 list, also to a paper_v1 overlay row.
        canonical = r.get("project") or "unknown"
        targets = [canonical]
        if r.get("in_paper_v1") and "paper_v1" not in targets:
            targets.append("paper_v1")

        for project in targets:
            row = project_rows.setdefault(project, {
                "project": project,
                "experiment_count": 0,
                "cell_lines": set(),
                "experimental_designs": set(),
                "libraries": set(),
                "total_cells": 0,
                "iss_matched_cells": 0,
                "tracked_total_cells": 0,
                "tracked_full_seg": 0,
            })
            row["experiment_count"] += 1
            cell_line = r.get("cell_line")
            if cell_line:
                row["cell_lines"].add(cell_line)
            library = r.get("library")
            if library:
                row["libraries"].add(library)
            designs = r.get("experimental_design") or []
            if isinstance(designs, str):
                designs = [designs]
            for d in designs:
                row["experimental_designs"].add(d)
            row["total_cells"] += r.get("total_cells") or 0
            row["iss_matched_cells"] += r.get("iss_matched_cells") or 0
            row["tracked_total_cells"] += r.get("tracked_total_cells") or 0
            row["tracked_full_seg"] += r.get("tracked_full_seg") or 0

            exp_id = r.get("experiment_id")
            if exp_id:
                project_to_exp_ids.setdefault(project, set()).add(exp_id)

            codebook = r.get("codebook")
            if codebook:
                project_to_codebooks.setdefault(project, set()).add(codebook)

    # Count unique markers per project from the channel-map index.
    for project, row in project_rows.items():
        exp_ids = project_to_exp_ids.get(project, set())
        markers = {
            marker for marker, info in marker_to_info.items()
            if any(eid in exp_ids for eid in info.get("exp_ids", []))
            and _normalize_marker_label(marker) != "phase"
        }
        row["unique_markers"] = len(markers)
        row["cell_lines"] = sorted(row["cell_lines"])
        row["libraries"] = sorted(row["libraries"])

        # Sum perturbations across the project's distinct codebooks.
        codebooks = project_to_codebooks.get(project, set())
        row["num_perturbations"] = sum(count_codebook_perturbations(c) for c in codebooks)

        # Average each marker's (Mean Cells/Perturbation) across this
        # project's markers — gives a per-marker yield metric for the project.
        marker_sums = [marker_to_sum_cpg[m] for m in markers if m in marker_to_sum_cpg]
        row["avg_sum_cells_per_perturbation"] = (
            sum(marker_sums) / len(marker_sums) if marker_sums else None
        )
        # Sort designs in canonical order (livecell_OPS first, then add-ons).
        row["experimental_designs"] = sorted(
            row["experimental_designs"],
            key=lambda d: (KNOWN_EXPERIMENTAL_DESIGNS.index(d)
                           if d in KNOWN_EXPERIMENTAL_DESIGNS else len(KNOWN_EXPERIMENTAL_DESIGNS)),
        )

    # Sort by total_cells descending so the heaviest projects lead.
    return sorted(project_rows.values(), key=lambda x: x["total_cells"], reverse=True)


def _save_funnel_graph(results: list, output_dir: Path):
    """
    Generate and save a line graph showing cell retention across pipeline stages for each experiment.

    X-axis: Pipeline stages (Total Cells, Cells with Reads, ISS Matched, Tracked, Full Seg)
    Y-axis: % of total cells retained
    Lines: One per experiment, colored by chronological date (viridis_r colormap)
    Labels: Shorthand experiment numbers at end of each line
    """
    # Filter to experiments with tracked data (complete funnel)
    tracked_exps = [r for r in results if r.get("tracked_total_cells") is not None and r.get("tracked_total_cells") > 0]

    if not tracked_exps:
        print("No experiments with complete funnel data to plot.")
        return None

    # Extract dates from experiment names for sorting and coloring
    exp_data = []
    for r in tracked_exps:
        exp_name = r["experiment"]
        # Extract date from ops0033_20250429 -> 20250429
        match = re.match(r"ops\d{4}_(\d{8})$", exp_name)
        if match:
            date_str = match.group(1)
            # Extract shorthand number: ops0033 -> 33
            num_match = re.match(r"ops0*(\d+)", exp_name)
            shorthand = num_match.group(1) if num_match else exp_name

            total = r.get("total_cells") or 0
            if total == 0:
                continue

            # Calculate percentages
            cells_with_reads = r.get("cells_with_reads") or 0
            iss_matched = r.get("iss_matched_cells") or 0
            tracked = r.get("tracked_total_cells") or 0
            full_seg = r.get("tracked_full_seg") or 0

            pct_total = 100.0
            pct_with_reads = (cells_with_reads / total * 100) if total > 0 else 0
            pct_iss = (iss_matched / total * 100) if total > 0 else 0
            pct_tracked = (tracked / total * 100) if total > 0 else 0
            pct_full_seg = (full_seg / total * 100) if total > 0 else 0

            exp_data.append({
                "experiment": exp_name,
                "shorthand": shorthand,
                "date": date_str,
                "percentages": [pct_total, pct_with_reads, pct_iss, pct_tracked, pct_full_seg],
            })

    if not exp_data:
        print("No valid experiment data for funnel graph.")
        return None

    # Sort by date for coloring
    exp_data.sort(key=lambda x: x["date"])

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Pipeline stage labels
    stages = ["Total\nCells", "Cells with\nReads", "ISS\nMatched", "Tracked", "Full\nSeg"]
    x_positions = list(range(len(stages)))

    # Get colormap (viridis_r: older = yellow, newer = purple)
    n_experiments = len(exp_data)
    colors = cm.viridis_r(np.linspace(0, 1, n_experiments))

    # Plot each experiment
    for idx, exp in enumerate(exp_data):
        color = colors[idx]
        y_values = exp["percentages"]

        # Most recent experiment (last in sorted list) gets thicker line
        is_most_recent = (idx == n_experiments - 1)
        line_width = 3.5 if is_most_recent else 1.5
        marker_size = 6 if is_most_recent else 4
        alpha = 1.0 if is_most_recent else 0.7

        # Plot line
        ax.plot(x_positions, y_values, marker='o', markersize=marker_size, linewidth=line_width,
                color=color, alpha=alpha, label=exp["shorthand"])

        # Add shorthand label at end of line (50% larger: 8 -> 12)
        ax.annotate(
            exp["shorthand"],
            xy=(x_positions[-1], y_values[-1]),
            xytext=(5, 0),
            textcoords='offset points',
            fontsize=12,
            color=color,
            va='center',
            ha='left',
            fontweight='bold' if is_most_recent else 'normal',
        )

    # Customize plot
    ax.set_xticks(x_positions)
    ax.set_xticklabels(stages, fontsize=10)
    ax.set_xlabel("Pipeline Stage", fontsize=12)
    ax.set_ylabel("% of Total Cells", fontsize=12)
    ax.set_title("Cell Retention Across Pipeline Stages by Experiment", fontsize=14)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add colorbar for date interpretation
    sm = plt.cm.ScalarMappable(cmap=cm.viridis_r, norm=plt.Normalize(0, n_experiments - 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label="Experiment (older → newer)", shrink=0.8)

    # Set colorbar ticks to show oldest and newest dates
    if exp_data:
        oldest_date = exp_data[0]["date"]
        newest_date = exp_data[-1]["date"]
        # Format dates: 20250429 -> 2025-04-29
        oldest_fmt = f"{oldest_date[:4]}-{oldest_date[4:6]}-{oldest_date[6:]}"
        newest_fmt = f"{newest_date[:4]}-{newest_date[4:6]}-{newest_date[6:]}"
        cbar.set_ticks([0, n_experiments - 1])
        cbar.set_ticklabels([oldest_fmt, newest_fmt])

    plt.tight_layout()

    # Save figure
    graph_path = output_dir / "cell_retention_funnel.png"
    plt.savefig(graph_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved funnel graph to: {graph_path}")
    return graph_path


def get_pipeline_status(experiment, config, dataset, fast: bool = False):
    """
    Checks all pipeline steps for completion status.
    Returns the last completed step, missing steps (incomplete but with later steps done),
    and the next truly incomplete step.
    """
    all_step_results = []  # List of (display_key, state) tuples in order
    skipped_steps = []  # list of log_keys that were skipped (no files defined)

    # Source of truth: dataset-declared canonical keys
    try:
        canonical_keys = list(dataset.get_all_step_keys())
    except Exception:
        canonical_keys = []

    # Truncate for ISS-only after get_metrics when requested
    if config.get("run_iss_only", False):
        try:
            cutoff = canonical_keys.index("get_metrics")
            canonical_keys = canonical_keys[: cutoff + 1]
        except ValueError:
            pass

    # Filter out fluorescence-only steps if no fluorescence channels
    has_fluor = has_fluorescence_channels_from_config(config)
    if not has_fluor:
        # Remove fluorescence-specific step
        canonical_keys = [
            k for k in canonical_keys if k != "create_max_projection_lc_20x_fluor"
        ]

    # Calculate total steps accounting for method variants, per-well expansion, and manual checkpoints
    total_steps = 0

    for key in canonical_keys:
        step_count = 0
        if key in ("track_well", "create_dataset"):
            wells = config.get("wells_to_process", []) or dataset.infer_wells()
            step_count = len(wells) if wells else 1
        elif key in ("base_calling", "get_metrics"):
            # Count method variants (mine, probabilistic)
            step_count = 2
        else:
            step_count = 1

        total_steps += step_count

        # Add manual checkpoints
        if key == "estimate_stitch_parameters_pheno" and has_fluor:
            total_steps += 1
        if key == "apply_registration_tracking":
            total_steps += 1

    # First pass: collect all step states
    for key in canonical_keys:
        # Skip fluorescence step if not applicable
        if key == "create_max_projection_lc_20x_fluor" and not has_fluor:
            continue

        # Check manual fluorescence registration checkpoint
        if key == "estimate_stitch_parameters_pheno" and has_fluor:
            manual_fluor_complete = _check_manual_fluor_registration(dataset, config)
            state = "complete" if manual_fluor_complete else "incomplete"
            all_step_results.append(("manual_fluor_registration", state))

        # Check manual ISS→Pheno registration checkpoint
        if key == "apply_registration_tracking":
            all_present, count_present, total_wells = (
                _check_manual_iss_pheno_registration(dataset, config)
            )
            if not all_present:
                display_key = f"manual_iss_pheno_registration ({count_present}/{total_wells} wells)"
                all_step_results.append((display_key, "incomplete"))
            else:
                all_step_results.append(("manual_iss_pheno_registration", "complete"))

        # Per-well expansion for steps that operate per well
        if key in ("track_well", "create_dataset"):
            wells = config.get("wells_to_process", []) or dataset.infer_wells()
            for w in wells:
                log_key = f"{key}_{_normalize_well_token(w)}"
                state, display_key = _get_step_state_by_key(
                    dataset, log_key, config, fast=fast
                )
                if state == "skipped":
                    skipped_steps.append(display_key)
                else:
                    all_step_results.append((display_key, state))
            continue

        # Method variants: base_calling and get_metrics
        if key in ("base_calling", "get_metrics"):
            for method in ("mine", "probabilistic"):
                state, display_key = _get_step_state_by_key(
                    dataset, key, config, fast=fast, method=method
                )
                if state == "skipped":
                    skipped_steps.append(display_key)
                else:
                    all_step_results.append((display_key, state))
            continue

        # Standard step
        state, display_key = _get_step_state_by_key(dataset, key, config, fast=fast)
        if state == "skipped":
            skipped_steps.append(display_key)
        else:
            all_step_results.append((display_key, state))

    # Second pass: analyze the collected states
    completed_count = sum(1 for _, state in all_step_results if state == "complete")
    last_completed_step = None
    missing_steps = []
    next_step = "Complete"

    # Find the last completed step
    for display_key, state in reversed(all_step_results):
        if state == "complete":
            last_completed_step = display_key
            break

    # Find missing steps (incomplete steps before the last completed step)
    # and the next incomplete step (first incomplete after last completed or first overall)
    found_last_complete = False
    first_incomplete_found = False

    for display_key, state in all_step_results:
        if state == "complete":
            if display_key == last_completed_step:
                found_last_complete = True
        elif state == "incomplete":
            if found_last_complete:
                # This is incomplete but we've passed the last complete - it's truly next
                if not first_incomplete_found:
                    next_step = display_key
                    first_incomplete_found = True
            else:
                # This is incomplete but there are completed steps after it - it's missing
                if last_completed_step:  # Only count as missing if we have completed steps
                    missing_steps.append(display_key)
                elif not first_incomplete_found:
                    # No completed steps yet, so this is the next step
                    next_step = display_key
                    first_incomplete_found = True

    return (
        next_step,
        last_completed_step,
        skipped_steps,
        missing_steps,
        completed_count,
        total_steps,
    )


def _get_progress_emoji(progress_pct: int) -> str:
    """Return emoji indicator for progress percentage in Markdown."""
    if progress_pct <= 25:
        return "🔴"  # Red
    elif progress_pct <= 50:
        return "🟠"  # Orange
    elif progress_pct <= 75:
        return "🟡"  # Yellow
    else:
        return "🟢"  # Green


def _save_csv_pipeline_status(
    csv_path: Path,
    results: list,
    missing_steps_legend: dict,
    skip_iss_only: bool = False,
):
    """Save pipeline status table as CSV."""
    # Configure columns based on skip_iss_only flag
    if skip_iss_only:
        fieldnames = [
            "Experiment",
            "Project",
            "Library",
            "# Perturbations",
            "Experimental Design",
            "Paper v1",
            "Cell Line",
            "Category",
            "Completed Steps",
            "Total Steps",
            "Progress %",
            "Total Cells",
            "Cells with Reads",
            "ISS Matched Cells",
            "Tracked Total Cells",
            "Tracked Full Seg",
            "Tracked Fallback Seg",
            "Mean Cells/Gene",
            "Last Complete Step",
            "Next Step",
            "Missing Steps",
            "Missing Steps (Detailed)",
        ]
    else:
        fieldnames = [
            "Experiment",
            "Project",
            "Library",
            "# Perturbations",
            "Experimental Design",
            "Paper v1",
            "Cell Line",
            "Category",
            "ISS Only",
            "Completed Steps",
            "Total Steps",
            "Progress %",
            "Total Cells",
            "Cells with Reads",
            "ISS Matched Cells",
            "Tracked Total Cells",
            "Tracked Full Seg",
            "Tracked Fallback Seg",
            "Mean Cells/Gene",
            "Last Complete Step",
            "Next Step",
            "Missing Steps",
            "Missing Steps (Detailed)",
        ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            # Format missing steps (numbers and detailed)
            missing_steps_nums = ""
            missing_steps_detailed = ""
            if r.get("missing_steps"):
                missing_nums = [missing_steps_legend[ms] for ms in r["missing_steps"]]
                missing_steps_nums = _format_number_ranges(missing_nums)
                missing_steps_detailed = "; ".join(r["missing_steps"])

            # Determine category display
            category = ""
            if r.get("experiment_category") == "positive_control":
                category = "Positive Control"
            elif r.get("experiment_category") == "need_rescue":
                category = "Need Rescue"
            else:
                exp_tag = get_experiment_tag(r["experiment"])
                project = r.get("project")
                if exp_tag and project:
                    category = f"{project} ({exp_tag})"
                elif exp_tag:
                    category = exp_tag

            designs = r.get("experimental_design") or []
            if isinstance(designs, str):
                designs = [designs]

            row = {
                "Experiment": r["experiment"],
                "Project": r.get("project") or "",
                "Library": r.get("library") or "",
                "# Perturbations": r.get("num_perturbations") or "",
                "Experimental Design": "; ".join(designs),
                "Paper v1": "Yes" if r.get("in_paper_v1") else "No",
                "Cell Line": r.get("cell_line") or "",
                "Category": category,
                "Completed Steps": r["completed_steps"],
                "Total Steps": r["total_steps"],
                "Progress %": r["progress_percent"],
                "Total Cells": r.get("total_cells") or "",
                "Cells with Reads": r.get("cells_with_reads") or "",
                "ISS Matched Cells": r.get("iss_matched_cells") or "",
                "Tracked Total Cells": r.get("tracked_total_cells") or "",
                "Tracked Full Seg": r.get("tracked_full_seg") or "",
                "Tracked Fallback Seg": r.get("tracked_fallback_seg") or "",
                "Mean Cells/Gene": r.get("mean_cells_per_gene") or "",
                "Last Complete Step": r["last_completed_step"] if r["last_completed_step"] else "Not Started",
                "Next Step": r["next_step"],
                "Missing Steps": missing_steps_nums,
                "Missing Steps (Detailed)": missing_steps_detailed,
            }

            if not skip_iss_only:
                row["ISS Only"] = "Yes" if r["iss_only"] else "No"

            writer.writerow(row)


def _save_csv_project_summary(csv_path: Path, project_summary: list):
    """Save the per-project summary as CSV."""
    fieldnames = [
        "Project",
        "Experiment Count",
        "Cell Lines",
        "Libraries",
        "# Perturbations",
        "Experimental Designs",
        "Unique Markers",
        "Total Cells",
        "ISS Matched Cells",
        "Tracked+Seg Cells",
        "Mean Cells/Perturbation",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in project_summary:
            avg_cpg = p.get("avg_sum_cells_per_perturbation")
            writer.writerow({
                "Project": p["project"],
                "Experiment Count": p["experiment_count"],
                "Cell Lines": "; ".join(p["cell_lines"]) if p["cell_lines"] else "",
                "Libraries": "; ".join(p["libraries"]) if p["libraries"] else "",
                "# Perturbations": p.get("num_perturbations") or "",
                "Experimental Designs": "; ".join(p["experimental_designs"]) if p["experimental_designs"] else "",
                "Unique Markers": p["unique_markers"],
                "Total Cells": p["total_cells"] or "",
                "ISS Matched Cells": p["iss_matched_cells"] or "",
                "Tracked+Seg Cells": p["tracked_full_seg"] or "",
                "Mean Cells/Perturbation": f"{avg_cpg:.2f}" if avg_cpg else "",
            })


def _save_csv_marker_analysis(csv_path: Path, marker_analysis_results: list):
    """Save marker analysis table as CSV."""
    fieldnames = [
        "#",
        "Channel",
        "Channel Indicator",
        "Marker",
        "Experiment Count",
        "Experiment IDs",
        "Total Cells",
        "ISS Matched Cells",
        "Tracked Cells",
        "Sum of Mean Cells/Gene",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, m in enumerate(marker_analysis_results, start=1):
            # Skip showing experiment list for Phase (too long); show placeholder
            if _normalize_marker_label(m.get("marker", "")) == "phase":
                exp_ids_str = "-"
            else:
                exp_ids_str = ", ".join(m.get("experiment_ids", []))

            writer.writerow({
                "#": idx,
                "Channel": m["channel_name"],
                "Channel Indicator": m.get("channel_indicator", ""),
                "Marker": m["marker"],
                "Experiment Count": m["experiment_count"],
                "Experiment IDs": exp_ids_str,
                "Total Cells": m.get("total_cells") or "",
                "ISS Matched Cells": m.get("iss_matched_cells") or "",
                "Tracked Cells": m.get("tracked_total_cells") or "",
                "Sum of Mean Cells/Gene": f"{m['mean_cells_per_gene']:.2f}" if m.get("mean_cells_per_gene") else "",
            })


def _save_markdown_report(
    md_path: Path,
    results: list,
    timestamp: str,
    fast: bool,
    manual_registration_count: int,
    skipped_experiments: list,
    missing_steps_legend: dict,
    iss_only_skipped: list = None,
    skip_iss_only: bool = False,
    marker_analysis_results: list = None,
    funnel_data: dict = None,
    bad_exps_skipped: list = None,
    project_summary: list = None,
    project_filter: str = None,
    marker_analyses: list = None,
):
    """Save a Markdown-formatted report with color indicators."""
    lines = [
        "# OPS Pipeline Status Report" + (" (FAST)" if fast else ""),
        f"\n**Generated:** {timestamp}\n",
    ]

    if bad_exps_skipped:
        lines.append(f"\n## Excluded Experiments ({len(bad_exps_skipped)})\n")
        lines.append("Experiments excluded via `bad_experiment.yaml`:\n\n")
        for exp_name, reason in bad_exps_skipped:
            lines.append(f"- `{exp_name}`: {reason}")
        lines.append("\n")

    if iss_only_skipped:
        lines.append(f"\n## Excluded ISS-Only Experiments ({len(iss_only_skipped)})\n")
        lines.append("Experiments excluded via `--skip-iss-only` flag:\n")
        lines.append(", ".join(f"`{exp}`" for exp in iss_only_skipped))
        lines.append("\n")

    if skipped_experiments:
        lines.append(f"\n## Skipped Experiments ({len(skipped_experiments)})\n")
        lines.append(
            "Non-standard experiment names (not matching `opsXXXX_YYYYMMDD` format):\n"
        )
        lines.append(", ".join(f"`{exp}`" for exp in skipped_experiments))
        lines.append("\n")

    lines.append("\n## Pipeline Progress\n")

    # Configure table headers based on skip_iss_only flag
    if skip_iss_only:
        lines.append(
            "| Experiment | Library | # Pert | Cell Line | Exp Design | Paper v1 | Step | Progress | Total Cells | ISS Matched Cells | Tracked Total Cells | Last Complete Step | Next Step | Missing Steps |"
        )
        lines.append(
            "|------------|---------|-------:|-----------|------------|----------|------|----------|-------------|-------------------|---------------------|-------------------|-----------|---------------|"
        )
    else:
        lines.append(
            "| Experiment | Library | # Pert | Cell Line | Exp Design | Paper v1 | ISS Only | Step | Progress | Total Cells | ISS Matched Cells | Tracked Total Cells | Last Complete Step | Next Step | Missing Steps |"
        )
        lines.append(
            "|------------|---------|-------:|-----------|------------|----------|----------|------|----------|-------------|-------------------|---------------------|-------------------|-----------|---------------|"
        )

    for r in results:
        exp = r["experiment"]
        
        # Add category tag to experiment name if applicable
        if r.get("experiment_category") == "positive_control":
            exp = f"{exp} `[+CTRL]`"
        elif r.get("experiment_category") == "need_rescue":
            exp = f"{exp} `[RESCUE]`"
        else:
            exp_tag = get_experiment_tag(r["experiment"])
            if exp_tag:
                exp = f"{exp} `[{exp_tag}]`"

        iss_only = "🟢 Yes" if r["iss_only"] else "🔵 No"
        step = f"{r['completed_steps']}/{r['total_steps']}"
        progress_pct = r["progress_percent"]
        progress_emoji = _get_progress_emoji(progress_pct)
        progress = f"{progress_emoji} {progress_pct}%"
        last_step = (
            r["last_completed_step"] if r["last_completed_step"] else "🔴 Not Started"
        )
        next_step = r["next_step"]

        # Add emoji to next step if complete
        if next_step == "Complete":
            next_step = "✅ Complete"
        else:
            step_emoji = _emoji_for_step(next_step)
            if step_emoji:
                next_step = f"{step_emoji} {next_step}"

        # Add emoji to last step
        if last_step != "🔴 Not Started":
            last_emoji = _emoji_for_step(last_step)
            if last_emoji:
                last_step = f"{last_emoji} {last_step}"

        # Format cell counts for markdown
        total_cells_md = (
            f"{r['total_cells']:,}" if r.get("total_cells") is not None else ""
        )
        iss_matched_cells_md = (
            f"{r['iss_matched_cells']:,}" if r.get("iss_matched_cells") is not None else ""
        )
        tracked_cells_md = (
            f"{r['tracked_total_cells']:,}"
            if r.get("tracked_total_cells") is not None
            else ""
        )

        # Format missing steps for markdown - use numbers matching the legend as ranges
        missing_steps_md = ""
        if r.get("missing_steps"):
            missing_nums = [missing_steps_legend[ms] for ms in r["missing_steps"]]
            missing_steps_md = _format_number_ranges(missing_nums)

        library_md = r.get("library") or ""
        num_pert = r.get("num_perturbations") or 0
        num_pert_md = f"{num_pert:,}" if num_pert else ""
        cell_line_md = r.get("cell_line") or ""
        designs_r = r.get("experimental_design") or []
        if isinstance(designs_r, str):
            designs_r = [designs_r]
        exp_design_md = ", ".join(designs_r)
        paper_v1_md = "🟢 Yes" if r.get("in_paper_v1") else "🔴 No"

        # Build table row based on skip_iss_only flag
        if skip_iss_only:
            lines.append(
                f"| {exp} | {library_md} | {num_pert_md} | {cell_line_md} | {exp_design_md} | {paper_v1_md} | {step} | {progress} | {total_cells_md} | {iss_matched_cells_md} | {tracked_cells_md} | {last_step} | {next_step} | {missing_steps_md} |"
            )
        else:
            lines.append(
                f"| {exp} | {library_md} | {num_pert_md} | {cell_line_md} | {exp_design_md} | {paper_v1_md} | {iss_only} | {step} | {progress} | {total_cells_md} | {iss_matched_cells_md} | {tracked_cells_md} | {last_step} | {next_step} | {missing_steps_md} |"
            )

    lines.append(f"\n## Summary\n")
    lines.append(
        f"🔧 **{manual_registration_count}** experiment(s) need manual ISS→Pheno registration\n"
    )

    lines.append("\n### Experiment Categories\n")
    lines.append("Some experiments are tagged with special categories:\n")
    lines.append("- **`[+CTRL]`** - Positive control experiments (included in per-experiment stats, excluded from marker analysis)")
    lines.append("- **`[RESCUE]`** - Experiments needing rescue (included in per-experiment stats, excluded from marker analysis)")
    lines.append("- **`[TAG]`** - Experiments carrying a per-project display tag from the library map (included in per-experiment stats, shown in their own marker table)")

    lines.append("\n### Progress Legend\n")
    lines.append("- 🔴 0-25% (Early stage)")
    lines.append("- 🟠 26-50% (Getting started)")
    lines.append("- 🟡 51-75% (Halfway)")
    lines.append("- 🟢 76-100% (Almost done/Complete)")

    # Add missing steps legend if there are any
    if missing_steps_legend:
        lines.append("\n### Missing Steps Legend\n")
        # Sort by number for consistent display
        for step_name, step_num in sorted(
            missing_steps_legend.items(), key=lambda x: x[1]
        ):
            emoji = _emoji_for_step(step_name)
            step_display = f"{emoji} {step_name}" if emoji else step_name
            lines.append(f"- **[{step_num}]** {step_display}")

    # Add cell loss funnel section
    if funnel_data:
        lines.append("\n## Cell Loss Funnel - Where cells are lost in the pipeline\n")

        total_cells = funnel_data.get("total_cells", 0)
        cells_with_reads = funnel_data.get("cells_with_reads", 0)
        iss_matched = funnel_data.get("iss_matched_cells", 0)
        tracked = funnel_data.get("tracked_total_cells", 0)
        full_seg = funnel_data.get("tracked_full_seg", 0)
        fallback = funnel_data.get("tracked_fallback_seg", 0)

        # Calculate percentages of total
        pct_with_reads = (cells_with_reads / total_cells * 100) if total_cells > 0 else 0
        pct_iss = (iss_matched / total_cells * 100) if total_cells > 0 else 0
        pct_tracked = (tracked / total_cells * 100) if total_cells > 0 else 0
        pct_full_seg = (full_seg / total_cells * 100) if total_cells > 0 else 0

        # Calculate losses (absolute numbers)
        loss_no_spots = total_cells - cells_with_reads
        loss_no_codebook = cells_with_reads - iss_matched
        loss_to_tracking = iss_matched - tracked
        loss_to_full_seg = tracked - full_seg

        # Retention percentages (relative to previous stage)
        ret_with_reads = (cells_with_reads / total_cells * 100) if total_cells > 0 else 0
        ret_iss = (iss_matched / cells_with_reads * 100) if cells_with_reads > 0 else 0
        ret_tracking = (tracked / iss_matched * 100) if iss_matched > 0 else 0
        ret_full_seg = (full_seg / tracked * 100) if tracked > 0 else 0

        # Loss percentages relative to PREVIOUS stage (not total)
        pct_loss_no_spots = 100 - ret_with_reads
        pct_loss_no_codebook = 100 - ret_iss
        pct_loss_tracking = 100 - ret_tracking
        pct_loss_full_seg = 100 - ret_full_seg

        lines.append("### Pipeline Stages\n")
        lines.append("| Stage | Cells | % of Total | Loss (from prev stage) | Loss Reason |")
        lines.append("|-------|------:|----------:|-----:|-------------|")
        lines.append(f"| 🧫 Total Cells (DAPI Segmented) | {total_cells:,} | 100.0% | - | - |")
        lines.append(f"| 📍 Cells with Reads | {cells_with_reads:,} | {pct_with_reads:.1f}% | {loss_no_spots:,} ({pct_loss_no_spots:.1f}%) | No spots/reads detected in cell |")
        lines.append(f"| 🧬 ISS Matched Cells | {iss_matched:,} | {pct_iss:.1f}% | {loss_no_codebook:,} ({pct_loss_no_codebook:.1f}%) | Barcode not in codebook |")
        lines.append(f"| 👣 Tracked Cells | {tracked:,} | {pct_tracked:.1f}% | {loss_to_tracking:,} ({pct_loss_tracking:.1f}%) | Not tracked across timepoints |")
        lines.append(f"| 🔬 Full Segmentation | {full_seg:,} | {pct_full_seg:.1f}% | {loss_to_full_seg:,} ({pct_loss_full_seg:.1f}%) | Fallback segmentation |")
        lines.append(f"| 📦 Fallback Segmentation | {fallback:,} | - | - | (subset of tracked) |")

        lines.append("\n### Stage-by-Stage Retention\n")
        lines.append("| Transition | Retention | Loss |")
        lines.append("|------------|----------:|-----:|")
        lines.append(f"| Total Cells → Cells with Reads | {ret_with_reads:.1f}% | {pct_loss_no_spots:.1f}% |")
        lines.append(f"| Cells with Reads → ISS Matched | {ret_iss:.1f}% | {pct_loss_no_codebook:.1f}% |")
        lines.append(f"| ISS Matched → Tracked | {ret_tracking:.1f}% | {pct_loss_tracking:.1f}% |")
        lines.append(f"| Tracked → Full Segmentation | {ret_full_seg:.1f}% | {pct_loss_full_seg:.1f}% |")

        lines.append(f"\n**Overall Yield:** {total_cells:,} → {full_seg:,} = **{pct_full_seg:.1f}%**\n")

    # Marker analysis sections — one per modality bucket. Falls back to the
    # legacy single-table input when ``marker_analyses`` is not supplied.
    bucket_inputs = marker_analyses
    if not bucket_inputs and marker_analysis_results:
        bucket_inputs = [("Cell Counts by Unique Marker", marker_analysis_results)]

    if bucket_inputs:
        for bucket_idx, (bucket_title, bucket_analysis) in enumerate(bucket_inputs):
            if not bucket_analysis:
                continue
            lines.append(f"\n## Marker Analysis - {bucket_title}\n")
            if bucket_idx == 0:
                lines.append("**Note:** Positive control experiments `[+CTRL]` and need rescue experiments `[RESCUE]` are excluded from this aggregation.\n")
            lines.append("| # | Ch | Marker | # Exps | Exps | Total Cells | ISS Matched Cells | Tracked Cells | Sum of Mean Cells/Gene |")
            lines.append("|--:|:--:|--------|-------:|------|-------------|-------------------|---------------|------------------------|")

            for idx, m in enumerate(bucket_analysis, start=1):
                ch_indicator = m.get("channel_indicator", "")
                marker = m["marker"]
                exp_count = m["experiment_count"]
                total_cells_m = f"{m['total_cells']:,}" if m.get("total_cells") else ""
                iss_matched_m = f"{m['iss_matched_cells']:,}" if m.get("iss_matched_cells") else ""
                tracked_m = f"{m['tracked_total_cells']:,}" if m.get("tracked_total_cells") else ""
                mean_cpg = f"{m['mean_cells_per_gene']:.2f}" if m.get("mean_cells_per_gene") else ""

                if _normalize_marker_label(marker) == "phase":
                    exps_str = "-"
                else:
                    exp_ids = m.get("experiment_ids", [])
                    exps_str = _format_exp_ids_as_ranges(exp_ids, max_individual=5)

                lines.append(f"| {idx} | {ch_indicator} | {marker} | {exp_count} | {exps_str} | {total_cells_m} | {iss_matched_m} | {tracked_m} | {mean_cpg} |")

    # Project summary section (always cross-project, even when --project filter is active)
    if project_summary:
        lines.append("\n## Project Summary - Cell Counts by Project\n")
        if project_filter:
            lines.append(f"**Note:** `--project {project_filter}` filters the per-experiment table above, but this summary always reflects every project.\n")
        lines.append("| Project | # Exps | Cell Lines | Libraries | # Perturbations | Experimental Designs | # Markers | Total Cells | ISS Matched Cells | Tracked+Seg Cells | Mean Cells/Perturbation |")
        lines.append("|---------|-------:|------------|-----------|----------------:|----------------------|----------:|------------:|------------------:|------------------:|------------------------:|")
        for p in project_summary:
            cell_lines = ", ".join(p["cell_lines"]) if p["cell_lines"] else ""
            libraries = ", ".join(p["libraries"]) if p["libraries"] else ""
            designs = ", ".join(p["experimental_designs"]) if p["experimental_designs"] else ""
            num_pert = p.get("num_perturbations") or 0
            num_pert_str = f"{num_pert:,}" if num_pert else ""
            total = f"{p['total_cells']:,}" if p["total_cells"] else ""
            iss = f"{p['iss_matched_cells']:,}" if p["iss_matched_cells"] else ""
            tracked_seg = f"{p['tracked_full_seg']:,}" if p["tracked_full_seg"] else ""
            avg_cpg = p.get("avg_sum_cells_per_perturbation")
            avg_cpg_str = f"{avg_cpg:.2f}" if avg_cpg else ""
            lines.append(
                f"| {p['project']} | {p['experiment_count']} | {cell_lines} | {libraries} | {num_pert_str} | {designs} | {p['unique_markers']} | {total} | {iss} | {tracked_seg} | {avg_cpg_str} |"
            )

    with open(md_path, "w") as f:
        f.write("\n".join(lines))


def report_pipeline_status(fast: bool = False, debug_cells: bool = False, skip_iss_only: bool = True, date_after: str = None, date_before: str = None, no_cache: bool = False, cache_verbose: bool = False, skip_bad_exps: bool = True, project_filter: str = None, save_reports: bool = False, workers: int = 16):
    """
    Generates a report on the current pipeline status for each experiment.

    Args:
        fast: If True, only check file existence (skip content validation)
        debug_cells: If True, print debug information for cell count retrieval
        skip_iss_only: If True (default), skip experiments with run_iss_only=True
        date_after: If provided, only include experiments with date >= this value (format: YYYYMMDD)
        date_before: If provided, only include experiments with date <= this value (format: YYYYMMDD)
        no_cache: If True, ignore cache and recompute all experiment data
        cache_verbose: If True, print detailed cache hit/miss information for each experiment
        skip_bad_exps: If True (default), skip experiments excluded by bad_experiment.yaml
        project_filter: If provided, only show experiments tagged with this project
                        (e.g. a project name). The project summary table still shows all projects.
    """
    dummy_dataset = OpsDataset("dummy")
    config_root_dir = dummy_dataset.config_paths["exp_config_dir"]

    if not config_root_dir.is_dir():
        print(f"Configuration directory not found at {config_root_dir}. Aborting.")
        return

    config_files = sorted(list(config_root_dir.glob("*_config.yaml")))
    table = PrettyTable()

    # Load cache
    if no_cache:
        cache = {"version": CACHE_VERSION, "experiments": {}}
        cache_read_only = True  # Don't save if --no-cache
    else:
        cache, cache_read_only = _load_cache()
        if cache_read_only:
            print("Note: Cache is read-only (no write access)")
    cache_hits = 0
    cache_misses = 0

    # Configure table columns based on skip_iss_only flag
    if skip_iss_only:
        table.field_names = [
            "Experiment",
            "Library",
            "# Pert",
            "Cell Line",
            "Exp Design",
            "Paper v1",
            "Step",
            "Progress %",
            "Total Cells",
            "ISS Matched Cells",
            "Tracked Total Cells",
            "Last Complete Step",
            "Next Step",
            "Missing Steps",
        ]
    else:
        table.field_names = [
            "Experiment",
            "Library",
            "# Pert",
            "Cell Line",
            "Exp Design",
            "Paper v1",
            "ISS Only",
            "Step",
            "Progress %",
            "Total Cells",
            "ISS Matched Cells",
            "Tracked Total Cells",
            "Last Complete Step",
            "Next Step",
            "Missing Steps",
        ]
    table.align = "l"

    # Collect structured results for YAML/Markdown outputs
    results = []
    skipped_experiments = []
    iss_only_skipped = []
    date_filtered_skipped = []
    bad_exps_skipped = []  # Experiments skipped by --skip-bad-exps
    manual_iss_pheno_registration_needed_count = 0
    # Global missing steps legend: maps step name to number
    missing_steps_legend = {}

    def _compute_experiment(config_path):
        """Heavy per-experiment I/O: config parse, hash, file checks, CSV reads.

        Runs in worker threads, so it only reads shared state (``cache``) and
        returns a record. All cache writes and shared-list mutations happen in
        the sequential bookkeeping pass below, keeping output deterministic.
        """
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            if not config or "experiment_name" not in config:
                return {"action": "continue"}

            exp_name = config["experiment_name"]

            # Filter experiments: only include standard format opsXXXX_YYYYMMDD
            # Must start with ops followed by 4 digits, then underscore, then 8 digits (date)
            standard_format = re.match(r"^ops\d{4}_(\d{8})$", exp_name)
            if not standard_format:
                return {"action": "skip_nonstandard", "exp_name": exp_name}

            # Filter by date if --date-after or --date-before is provided
            exp_date = standard_format.group(1)  # Extract YYYYMMDD from experiment name
            if date_after and exp_date < date_after:
                return {"action": "skip_date", "exp_name": exp_name}
            if date_before and exp_date > date_before:
                return {"action": "skip_date", "exp_name": exp_name}

            # Filter by --skip-bad-exps: date cutoff + all default exclusion categories
            # Use only the original 4 categories — positive_control and need_rescue
            # should still appear in the pipeline report.
            _REPORT_EXCLUDE = ("bad", "iss_only", "do_not_run", "non_standard")
            if skip_bad_exps and is_excluded(exp_name, categories=_REPORT_EXCLUDE):
                reason = get_reason(exp_name) or "excluded"
                return {"action": "skip_bad", "exp_name": exp_name, "reason": reason}

            dataset = OpsDataset(exp_name)

            iss_only = config.get("run_iss_only", False)

            # Skip ISS-only experiments if flag is set
            if skip_iss_only and iss_only:
                return {"action": "skip_iss_only", "exp_name": exp_name}

            # Check cache for this experiment
            exp_hash = _compute_experiment_hash(dataset, config, config_path)
            cached_data = _get_cached_experiment_data(cache, exp_name, exp_hash) if not no_cache else None

            base = {
                "action": "ok",
                "config_path": config_path,
                "config": config,
                "exp_name": exp_name,
                "iss_only": iss_only,
                "exp_hash": exp_hash,
            }

            if cached_data:
                return {**base, "cache_hit": True, "data": cached_data}

            # Compute fresh data
            (
                next_step,
                last_completed,
                skipped_list,
                missing_list,
                completed_count,
                total_steps,
            ) = get_pipeline_status(exp_name, config, dataset, fast=fast)

            # Get cell counts for new columns
            total_cells, cells_with_reads, iss_matched_cells = _get_iss_total_cells(dataset, exp_name, debug=debug_cells)
            # Single read per well covers total cells, mean cells/gene, and seg breakdown
            (
                tracked_total_cells,
                mean_cells_per_gene,
                tracked_full_seg,
                tracked_fallback_seg,
            ) = _get_tracked_cell_stats(dataset, config, exp_name, debug=debug_cells)
            # Extract experiment ID for marker mapping (e.g., ops0061_20250728 -> ops0061)
            experiment_id = _extract_experiment_id(exp_name)

            data = {
                "next_step": next_step,
                "last_completed_step": last_completed,
                "skipped_steps": skipped_list,
                "missing_steps": missing_list,
                "completed_steps": completed_count,
                "total_steps": total_steps,
                "total_cells": total_cells,
                "cells_with_reads": cells_with_reads,
                "iss_matched_cells": iss_matched_cells,
                "tracked_total_cells": tracked_total_cells,
                "tracked_full_seg": tracked_full_seg,
                "tracked_fallback_seg": tracked_fallback_seg,
                "mean_cells_per_gene": mean_cells_per_gene,
                "experiment_id": experiment_id,
            }
            return {**base, "cache_hit": False, "data": data}
        except Exception as e:
            return {"action": "error", "config_path": config_path, "error": e}

    # Phase 1: fan the heavy I/O out across threads (network-FS bound, so threads
    # help even in --fast mode). ``map`` preserves config order so the table and
    # cache bookkeeping below stay deterministic.
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        computed_records = list(
            tqdm(
                executor.map(_compute_experiment, config_files),
                total=len(config_files),
                desc="Checking experiments",
            )
        )

    # Phase 2: sequential bookkeeping — mutate shared lists/cache and build rows.
    for record in computed_records:
        action = record["action"]
        if action == "continue":
            continue
        if action == "skip_nonstandard":
            skipped_experiments.append(record["exp_name"])
            continue
        if action == "skip_date":
            date_filtered_skipped.append(record["exp_name"])
            continue
        if action == "skip_bad":
            bad_exps_skipped.append((record["exp_name"], record["reason"]))
            continue
        if action == "skip_iss_only":
            iss_only_skipped.append(record["exp_name"])
            continue
        if action == "error":
            exp_name_from_file = record["config_path"].stem.replace("_config", "")
            n_cols = len(table.field_names)
            error_row = [f"{exp_name_from_file}"] + [""] * (n_cols - 4) + [f"ERROR: {record['error']}", "", ""]
            table.add_row(error_row)
            continue

        try:
            config_path = record["config_path"]
            config = record["config"]
            exp_name = record["exp_name"]
            iss_only = record["iss_only"]
            exp_hash = record["exp_hash"]
            iss_only_display = "🟢 Yes" if iss_only else "🔵 No"

            if record["cache_hit"]:
                cache_hits += 1
                if cache_verbose:
                    print(f"  [CACHE HIT]  {exp_name} (hash: {exp_hash[:8]}...)")
            else:
                cache_misses += 1
                if cache_verbose:
                    old_cache = cache.get("experiments", {}).get(exp_name)
                    if old_cache is None:
                        reason = "not in cache"
                    else:
                        old_hash = old_cache.get("hash", "")
                        reason = f"hash changed: {old_hash[:8]}... → {exp_hash[:8]}..."
                    print(f"  [CACHE MISS] {exp_name} ({reason})")
                _update_experiment_cache(cache, exp_name, exp_hash, record["data"])

            data = record["data"]
            next_step = data["next_step"]
            last_completed = data["last_completed_step"]
            skipped_list = data["skipped_steps"]
            missing_list = data["missing_steps"]
            completed_count = data["completed_steps"]
            total_steps = data["total_steps"]
            total_cells = data["total_cells"]
            cells_with_reads = data["cells_with_reads"]
            iss_matched_cells = data["iss_matched_cells"]
            tracked_total_cells = data["tracked_total_cells"]
            tracked_full_seg = data["tracked_full_seg"]
            tracked_fallback_seg = data["tracked_fallback_seg"]
            mean_cells_per_gene = data["mean_cells_per_gene"]
            experiment_id = data["experiment_id"]

            # Calculate progress based on all pipeline steps
            progress_pct = round(
                (completed_count / total_steps * 100) if total_steps > 0 else 0
            )

            step_display = _colorize_step(
                f"{completed_count}/{total_steps}", progress_pct
            )
            progress_display = _colorize_progress(progress_pct)
            last_completed_display = last_completed if last_completed else "Not Started"
            skipped_display = ", ".join(skipped_list) if skipped_list else ""

            # If running ISS only, the pipeline ends at metrics. If we've reached
            # any "Complete" state already, keep it; otherwise infer completion
            # by checking that the next step is beyond metrics (handled above).
            if (
                iss_only
                and isinstance(last_completed, str)
                and last_completed.endswith("get_metrics")
            ):
                next_step = "Complete"

            # Add category emojis for visibility
            if next_step == "Complete":
                next_step_display = "✅ Complete"
            else:
                next_emoji = _emoji_for_step(next_step)
                next_step_display = (
                    f"{next_emoji} {next_step}" if next_emoji else next_step
                )

            if last_completed_display == "Not Started":
                last_completed_display = "🔴 Not Started"
            else:
                last_emoji = _emoji_for_step(last_completed_display)
                if last_emoji:
                    last_completed_display = f"{last_emoji} {last_completed_display}"

            # Format missing steps with numbers and build legend
            missing_display = ""
            missing_numbers = []
            if missing_list:
                for ms in missing_list:
                    # Add to legend if not already present
                    if ms not in missing_steps_legend:
                        missing_steps_legend[ms] = len(missing_steps_legend) + 1
                    missing_numbers.append(missing_steps_legend[ms])
                # Format as ranges (e.g., "1-5, 7-8, 10")
                missing_display = _format_number_ranges(missing_numbers)

            # Format cell counts for display
            total_cells_display = (
                f"{total_cells:,}" if total_cells is not None else ""
            )
            iss_matched_cells_display = (
                f"{iss_matched_cells:,}" if iss_matched_cells is not None else ""
            )
            tracked_cells_display = (
                f"{tracked_total_cells:,}" if tracked_total_cells is not None else ""
            )

            # Add category tag to experiment name if applicable
            exp_name_display = exp_name
            experiment_category = None
            if exp_name in POSITIVE_CONTROL_EXPERIMENTS:
                exp_name_display = f"{exp_name} [+CTRL]"
                experiment_category = "positive_control"
            elif exp_name in NEED_RESCUE_EXPERIMENTS:
                exp_name_display = f"{exp_name} [RESCUE]"
                experiment_category = "need_rescue"
            else:
                exp_tag = get_experiment_tag(exp_name)
                if exp_tag:
                    exp_name_display = f"{exp_name} [{exp_tag}]"

            # Resolve project tag and experimental design (config takes precedence,
            # fall back to derivation). The full results list always contains every
            # experiment so the cross-project summary stays accurate; only the
            # displayed table is filtered when --project is supplied.
            project_tag = config.get("project") or derive_project(exp_name)
            experimental_design = config.get("experimental_design")
            if not experimental_design:
                experimental_design = derive_experimental_design(exp_name)
            elif isinstance(experimental_design, str):
                # Tolerate older single-string configs by promoting to a list.
                experimental_design = [experimental_design]
            library = derive_library(exp_name)
            codebook_filename = config.get("codebook", "")
            num_perturbations = count_codebook_perturbations(codebook_filename)
            in_paper_v1 = is_in_paper_v1(exp_name)
            in_paper_v2 = is_in_paper_v2(exp_name)
            cell_line = config.get("cell_line") or "A549"
            # paper_v2 is an A549, non-Validation cohort: HELA and Validation-library
            # experiments are never part of it, even if a stale
            # good_experiment_list_v2.yml entry says otherwise.
            if str(cell_line).upper() == "HELA" or project_tag == "Validation":
                in_paper_v2 = False
            # paper_v1/paper_v2 are overlay projects — match either the canonical
            # project tag or membership in the corresponding curated list.
            if project_filter is None:
                display_in_table = True
            elif project_filter == "paper_v1":
                display_in_table = in_paper_v1
            elif project_filter == "paper_v2":
                display_in_table = in_paper_v2
            else:
                display_in_table = project_tag == project_filter

            num_pert_display = f"{num_perturbations:,}" if num_perturbations else ""
            cell_line_display = cell_line or ""
            exp_design_display = ", ".join(experimental_design) if experimental_design else ""
            paper_v1_display = "🟢 Yes" if in_paper_v1 else "🔴 No"

            # Build table row based on skip_iss_only flag
            if not display_in_table:
                pass
            elif skip_iss_only:
                table.add_row(
                    [
                        exp_name_display,
                        library or "",
                        num_pert_display,
                        cell_line_display,
                        exp_design_display,
                        paper_v1_display,
                        step_display,
                        progress_display,
                        total_cells_display,
                        iss_matched_cells_display,
                        tracked_cells_display,
                        last_completed_display,
                        next_step_display,
                        missing_display,
                    ]
                )
            else:
                table.add_row(
                    [
                        exp_name_display,
                        library or "",
                        num_pert_display,
                        cell_line_display,
                        exp_design_display,
                        paper_v1_display,
                        iss_only_display,
                        step_display,
                        progress_display,
                        total_cells_display,
                        iss_matched_cells_display,
                        tracked_cells_display,
                        last_completed_display,
                        next_step_display,
                        missing_display,
                    ]
                )

            # Track experiments that need manual ISS→Pheno registration
            if not iss_only:  # Only applies to full pipeline
                # If next step is the manual registration or it's the last step before it
                if next_step and "manual_iss_pheno_registration" in str(next_step):
                    manual_iss_pheno_registration_needed_count += 1

            # Append to structured results
            results.append(
                {
                    "experiment": exp_name,
                    "experiment_id": experiment_id,
                    "experiment_category": experiment_category,
                    "project": project_tag,
                    "experimental_design": experimental_design,
                    "library": library,
                    "codebook": codebook_filename,
                    "num_perturbations": num_perturbations,
                    "in_paper_v1": in_paper_v1,
                    "in_paper_v2": in_paper_v2,
                    "cell_line": cell_line,
                    "iss_only": bool(iss_only),
                    "completed_steps": completed_count,
                    "total_steps": total_steps,
                    "progress_percent": progress_pct,
                    "total_cells": total_cells,
                    "cells_with_reads": cells_with_reads,
                    "iss_matched_cells": iss_matched_cells,
                    "tracked_total_cells": tracked_total_cells,
                    "tracked_full_seg": tracked_full_seg,
                    "tracked_fallback_seg": tracked_fallback_seg,
                    "mean_cells_per_gene": mean_cells_per_gene,
                    "last_completed_step": last_completed if last_completed else None,
                    "next_step": next_step,
                    "missing_steps": missing_list,
                    "skipped_steps": skipped_list,
                }
            )

        except Exception as e:
            exp_name_from_file = config_path.stem.replace("_config", "")
            # Pad the error row to match the active column count: name + empties +
            # error in the "Last Complete Step" slot + 2 trailing empties.
            n_cols = len(table.field_names)
            error_row = [f"{exp_name_from_file}"] + [""] * (n_cols - 4) + [f"ERROR: {e}", "", ""]
            table.add_row(error_row)

    # Save cache after processing all experiments
    _save_cache(cache, cache_read_only)

    # Keep the full list for the cross-project summary; downstream code (per-experiment
    # table, funnel, marker analysis, paper-v1 summary, CSV/markdown rows) operates on
    # the filtered view so --project hides everything outside the chosen project.
    # paper_v1 is an overlay subset (curated good_experiment_list_v1.yml), so it
    # filters by membership in that list rather than the canonical project tag.
    all_results = results
    if project_filter == "paper_v1":
        results = [r for r in all_results if r.get("in_paper_v1")]
    elif project_filter == "paper_v2":
        results = [r for r in all_results if r.get("in_paper_v2")]
    elif project_filter:
        results = [r for r in all_results if r.get("project") == project_filter]

    print("\n" + "=" * 80)
    print("OPS Pipeline Status Report" + (" (FAST)" if fast else ""))
    if project_filter:
        print(f"Filtered to project: {project_filter} ({len(results)}/{len(all_results)} experiments)")

    # Display cache statistics
    total_processed = cache_hits + cache_misses
    if total_processed > 0:
        cache_pct = (cache_hits / total_processed * 100)
        print(f"Cache: {cache_hits} hits, {cache_misses} misses ({cache_pct:.1f}% hit rate)")
    print("=" * 80)

    # Display excluded ISS-only experiments if any
    if iss_only_skipped:
        print(f"\nExcluded {len(iss_only_skipped)} ISS-only experiments (--skip-iss-only):")
        print("  " + ", ".join(iss_only_skipped))
        print()

    # Display date-filtered experiments if any
    if date_filtered_skipped:
        date_range_msg = ""
        if date_after and date_before:
            date_range_msg = f"outside range {date_after}-{date_before}"
        elif date_after:
            date_range_msg = f"before {date_after}"
        elif date_before:
            date_range_msg = f"after {date_before}"
        print(f"\nExcluded {len(date_filtered_skipped)} experiments {date_range_msg} (--date-after/--date-before):")
        print("  " + ", ".join(date_filtered_skipped))
        print()

    # Display bad experiments skipped (--skip-bad-exps, default enabled)
    if bad_exps_skipped:
        print(f"\nExcluded {len(bad_exps_skipped)} experiments (via bad_experiment.yaml):")
        for exp_name, reason in bad_exps_skipped:
            print(f"    {exp_name}: {reason}")
        print()

    # Display skipped experiments if any
    if skipped_experiments:
        print(f"\nSkipped {len(skipped_experiments)} non-standard experiments:")
        print("  " + ", ".join(skipped_experiments))
        print()

    print(table)
    print("=" * 80)
    print(
        f"\n🔧 Summary: {manual_iss_pheno_registration_needed_count} experiment(s) need manual ISS→Pheno registration\n"
    )

    # Print missing steps legend if there are any
    if missing_steps_legend:
        print("Missing Steps Legend:")
        print("-" * 80)
        # Sort by number for consistent display
        for step_name, step_num in sorted(
            missing_steps_legend.items(), key=lambda x: x[1]
        ):
            emoji = _emoji_for_step(step_name)
            step_display = f"{emoji} {step_name}" if emoji else step_name
            print(f"  [{step_num}] {step_display}")
        print()

    # Calculate total cells across all experiments
    total_cells_sum = sum(r.get("total_cells") or 0 for r in results)
    total_cells_with_reads = sum(r.get("cells_with_reads") or 0 for r in results)
    total_iss_matched_cells = sum(r.get("iss_matched_cells") or 0 for r in results)
    total_tracked_cells = sum(r.get("tracked_total_cells") or 0 for r in results)
    total_tracked_full_seg = sum(r.get("tracked_full_seg") or 0 for r in results)
    total_tracked_fallback = sum(r.get("tracked_fallback_seg") or 0 for r in results)

    # Count experiments that contributed to each total
    total_cells_experiment_count = sum(1 for r in results if r.get("total_cells") is not None and r.get("total_cells") > 0)
    cells_with_reads_experiment_count = sum(1 for r in results if r.get("cells_with_reads") is not None and r.get("cells_with_reads") > 0)
    iss_matched_experiment_count = sum(1 for r in results if r.get("iss_matched_cells") is not None and r.get("iss_matched_cells") > 0)
    tracked_experiment_count = sum(1 for r in results if r.get("tracked_total_cells") is not None and r.get("tracked_total_cells") > 0)
    full_seg_experiment_count = sum(1 for r in results if r.get("tracked_full_seg") is not None and r.get("tracked_full_seg") > 0)

    # For the funnel: only include experiments that have been fully tracked
    # This gives a true picture of cell loss, not incomplete pipelines
    tracked_experiments = [r for r in results if r.get("tracked_total_cells") is not None and r.get("tracked_total_cells") > 0]
    funnel_total_cells = sum(r.get("total_cells") or 0 for r in tracked_experiments)
    funnel_cells_with_reads = sum(r.get("cells_with_reads") or 0 for r in tracked_experiments)
    funnel_iss_matched = sum(r.get("iss_matched_cells") or 0 for r in tracked_experiments)
    funnel_tracked = sum(r.get("tracked_total_cells") or 0 for r in tracked_experiments)
    funnel_full_seg = sum(r.get("tracked_full_seg") or 0 for r in tracked_experiments)
    funnel_fallback = sum(r.get("tracked_fallback_seg") or 0 for r in tracked_experiments)

    # Calculate percentages for the funnel
    pct_iss_matched = (total_iss_matched_cells / total_cells_sum * 100) if total_cells_sum > 0 else 0
    pct_tracked = (total_tracked_cells / total_cells_sum * 100) if total_cells_sum > 0 else 0
    pct_full_seg = (total_tracked_full_seg / total_cells_sum * 100) if total_cells_sum > 0 else 0

    # Display large text summary of total cells
    print("=" * 80)
    print("TOTAL CELLS SUMMARY")
    print("=" * 80)
    print(f"\n  🧫 Total Cells (dapi segmented):              {total_cells_sum:,} from {total_cells_experiment_count} experiments")
    print(f"  🧬 ISS Matched Cells (codebook match):        {total_iss_matched_cells:,} from {iss_matched_experiment_count} experiments")
    print(f"  👣 Total Tracked Cells (linked pheno-iss):    {total_tracked_cells:,} from {tracked_experiment_count} experiments")
    print(f"  🔬 Tracked Cells with Full Seg:               {total_tracked_full_seg:,} from {full_seg_experiment_count} experiments")
    print("\n" + "=" * 80 + "\n")

    # Paper v1 summary: use the curated good_experiment_list_v1.yml directly
    # rather than excluding by category, so the count matches the config.
    paper_v1_results = [r for r in results if r.get("in_paper_v1")]
    pv1_total = sum(r.get("total_cells") or 0 for r in paper_v1_results)
    pv1_iss = sum(r.get("iss_matched_cells") or 0 for r in paper_v1_results)
    pv1_tracked = sum(r.get("tracked_total_cells") or 0 for r in paper_v1_results)
    pv1_full_seg = sum(r.get("tracked_full_seg") or 0 for r in paper_v1_results)
    pv1_n_total = sum(1 for r in paper_v1_results if r.get("total_cells") is not None and r.get("total_cells") > 0)
    pv1_n_iss = sum(1 for r in paper_v1_results if r.get("iss_matched_cells") is not None and r.get("iss_matched_cells") > 0)
    pv1_n_tracked = sum(1 for r in paper_v1_results if r.get("tracked_total_cells") is not None and r.get("tracked_total_cells") > 0)
    pv1_n_full = sum(1 for r in paper_v1_results if r.get("tracked_full_seg") is not None and r.get("tracked_full_seg") > 0)
    pv1_count = len(paper_v1_results)

    print("=" * 80)
    print(f"TOTAL CELLS SUMMARY (paper v1 — {pv1_count} experiments from good_experiment_list_v1.yml)")
    print("=" * 80)
    print(f"\n  🧫 Total Cells (dapi segmented):              {pv1_total:,} from {pv1_n_total} experiments")
    print(f"  🧬 ISS Matched Cells (codebook match):        {pv1_iss:,} from {pv1_n_iss} experiments")
    print(f"  👣 Total Tracked Cells (linked pheno-iss):    {pv1_tracked:,} from {pv1_n_tracked} experiments")
    print(f"  🔬 Tracked Cells with Full Seg:               {pv1_full_seg:,} from {pv1_n_full} experiments")
    print("\n" + "=" * 80 + "\n")

    # Paper v2 summary: curated good_experiment_list_v2.yml (extends v1 to all
    # current curated experiments).
    paper_v2_results = [r for r in results if r.get("in_paper_v2")]
    pv2_total = sum(r.get("total_cells") or 0 for r in paper_v2_results)
    pv2_iss = sum(r.get("iss_matched_cells") or 0 for r in paper_v2_results)
    pv2_tracked = sum(r.get("tracked_total_cells") or 0 for r in paper_v2_results)
    pv2_full_seg = sum(r.get("tracked_full_seg") or 0 for r in paper_v2_results)
    pv2_n_total = sum(1 for r in paper_v2_results if r.get("total_cells"))
    pv2_n_iss = sum(1 for r in paper_v2_results if r.get("iss_matched_cells"))
    pv2_n_tracked = sum(1 for r in paper_v2_results if r.get("tracked_total_cells"))
    pv2_n_full = sum(1 for r in paper_v2_results if r.get("tracked_full_seg"))
    pv2_count = len(paper_v2_results)

    # Unique-marker coverage: how many distinct markers the v1 vs v2 experiment
    # sets contribute (via the channel maps), and how many v2 adds over v1.
    _channel_maps = _load_channel_maps()
    _marker_to_info = _get_unique_markers_from_channel_maps(_channel_maps) if _channel_maps else {}
    v1_ids = {_extract_experiment_id(n) for n in get_paper_v1_experiments()}
    v2_ids = {_extract_experiment_id(n) for n in get_paper_v2_experiments()}

    def _markers_for(ids: set) -> set:
        return {
            m for m, info in _marker_to_info.items()
            if _normalize_marker_label(m) != "phase"
            and any(eid in ids for eid in info.get("exp_ids", []))
        }

    v1_markers = _markers_for(v1_ids)
    v2_markers = _markers_for(v2_ids)
    new_markers = v2_markers - v1_markers

    print("=" * 80)
    print(f"TOTAL CELLS SUMMARY (paper v2 — {pv2_count} experiments from good_experiment_list_v2.yml)")
    print("=" * 80)
    print(f"\n  🧫 Total Cells (dapi segmented):              {pv2_total:,} from {pv2_n_total} experiments")
    print(f"  🧬 ISS Matched Cells (codebook match):        {pv2_iss:,} from {pv2_n_iss} experiments")
    print(f"  👣 Total Tracked Cells (linked pheno-iss):    {pv2_tracked:,} from {pv2_n_tracked} experiments")
    print(f"  🔬 Tracked Cells with Full Seg:               {pv2_full_seg:,} from {pv2_n_full} experiments")
    print(f"\n  🏷️  Unique markers (excl. Phase):             {len(v2_markers)} (v1: {len(v1_markers)}, +{len(new_markers)} new in v2)")
    if new_markers:
        print(f"      New markers: {', '.join(sorted(new_markers))}")
    print("\n" + "=" * 80 + "\n")

    # Display cell loss funnel/flowchart (using only fully tracked experiments)
    print("=" * 80)
    print(f"CELL LOSS FUNNEL - Where cells are lost ({tracked_experiment_count} fully tracked experiments)")
    print("=" * 80)

    # Calculate loss at each stage (absolute numbers) - using funnel_ prefixed vars
    loss_no_spots = funnel_total_cells - funnel_cells_with_reads  # cells with no spots assigned
    loss_no_codebook_match = funnel_cells_with_reads - funnel_iss_matched  # spots didn't match codebook
    loss_to_tracking = funnel_iss_matched - funnel_tracked
    loss_to_full_seg = funnel_tracked - funnel_full_seg

    # Percentages of total for the funnel
    funnel_pct_with_reads = (funnel_cells_with_reads / funnel_total_cells * 100) if funnel_total_cells > 0 else 0
    funnel_pct_iss = (funnel_iss_matched / funnel_total_cells * 100) if funnel_total_cells > 0 else 0
    funnel_pct_tracked = (funnel_tracked / funnel_total_cells * 100) if funnel_total_cells > 0 else 0
    funnel_pct_full_seg = (funnel_full_seg / funnel_total_cells * 100) if funnel_total_cells > 0 else 0

    # Retention percentages (relative to previous stage) - this is what matters for each transition
    retention_with_reads = (funnel_cells_with_reads / funnel_total_cells * 100) if funnel_total_cells > 0 else 0
    retention_iss = (funnel_iss_matched / funnel_cells_with_reads * 100) if funnel_cells_with_reads > 0 else 0
    retention_tracking = (funnel_tracked / funnel_iss_matched * 100) if funnel_iss_matched > 0 else 0
    retention_full_seg = (funnel_full_seg / funnel_tracked * 100) if funnel_tracked > 0 else 0

    # Loss percentages relative to PREVIOUS stage (not total)
    pct_loss_no_spots = 100 - retention_with_reads  # loss from total cells (no spots)
    pct_loss_no_codebook = 100 - retention_iss  # loss from cells with reads (no codebook match)
    pct_loss_to_tracking = 100 - retention_tracking  # loss from ISS matched
    pct_loss_to_full_seg = 100 - retention_full_seg  # loss from tracked

    # ASCII funnel visualization - consistent width bars with fill showing retention
    funnel_width = 60
    print("\n")

    # Helper to create a bar with filled portion and empty space
    def make_bar(fill_pct: float) -> str:
        filled = max(1, int(funnel_width * fill_pct / 100))
        empty = funnel_width - filled
        return "█" * filled + "░" * empty

    # Stage 1: Total Cells (100%)
    print(f"  🧫 Total Cells (DAPI Segmented)")
    print(f"  ┌{'─' * (funnel_width + 2)}┐")
    print(f"  │ {make_bar(100)} │")
    print(f"  │ {funnel_total_cells:>15,} cells (100.0%)            │")
    print(f"  └{'─' * (funnel_width + 2)}┘")
    print(f"           │")
    print(f"           │  ❌ Lost: {loss_no_spots:,} cells ({pct_loss_no_spots:.1f}% of stage) - No spots/reads detected in cell")
    print(f"           ▼")

    # Stage 2: Cells with Reads (spots assigned)
    print(f"  📍 Cells with Reads (spots assigned)")
    print(f"  ┌{'─' * (funnel_width + 2)}┐")
    print(f"  │ {make_bar(funnel_pct_with_reads)} │")
    print(f"  │ {funnel_cells_with_reads:>15,} cells ({funnel_pct_with_reads:.1f}% of total)        │")
    print(f"  └{'─' * (funnel_width + 2)}┘")
    print(f"           │")
    print(f"           │  ❌ Lost: {loss_no_codebook_match:,} cells ({pct_loss_no_codebook:.1f}% of stage) - Barcode not in codebook")
    print(f"           ▼")

    # Stage 3: ISS Matched Cells
    print(f"  🧬 ISS Matched Cells (codebook match)")
    print(f"  ┌{'─' * (funnel_width + 2)}┐")
    print(f"  │ {make_bar(funnel_pct_iss)} │")
    print(f"  │ {funnel_iss_matched:>15,} cells ({funnel_pct_iss:.1f}% of total)        │")
    print(f"  └{'─' * (funnel_width + 2)}┘")
    print(f"           │")
    print(f"           │  ❌ Lost: {loss_to_tracking:,} cells ({pct_loss_to_tracking:.1f}% of stage) - Not tracked across timepoints")
    print(f"           ▼")

    # Stage 4: Tracked Cells
    print(f"  👣 Tracked Cells (Linked Pheno-ISS)")
    print(f"  ┌{'─' * (funnel_width + 2)}┐")
    print(f"  │ {make_bar(funnel_pct_tracked)} │")
    print(f"  │ {funnel_tracked:>15,} cells ({funnel_pct_tracked:.1f}% of total)        │")
    print(f"  └{'─' * (funnel_width + 2)}┘")
    print(f"           │")
    print(f"           │  ❌ Lost: {loss_to_full_seg:,} cells ({pct_loss_to_full_seg:.1f}% of stage) - Fallback segmentation (no pheno seg)")
    print(f"           ▼")

    # Stage 5: Full Segmentation Cells
    print(f"  🔬 Tracked Cells with Full Seg")
    print(f"  ┌{'─' * (funnel_width + 2)}┐")
    print(f"  │ {make_bar(funnel_pct_full_seg)} │")
    print(f"  │ {funnel_full_seg:>15,} cells ({funnel_pct_full_seg:.1f}% of total)        │")
    print(f"  └{'─' * (funnel_width + 2)}┘")

    print("\n")

    # Summary table of losses
    print("  ┌─────────────────────────────────────────────────────────────────────────────────┐")
    print("  │                           STAGE-BY-STAGE RETENTION                              │")
    print("  ├─────────────────────────────────────────────────────────────────────────────────┤")
    print(f"  │  Total Cells → Cells with Reads:   {retention_with_reads:>6.1f}% retained ({pct_loss_no_spots:>5.1f}% lost)          │")
    print(f"  │  Cells with Reads → ISS Matched:   {retention_iss:>6.1f}% retained ({pct_loss_no_codebook:>5.1f}% lost)          │")
    print(f"  │  ISS Matched → Tracked:            {retention_tracking:>6.1f}% retained ({pct_loss_to_tracking:>5.1f}% lost)          │")
    print(f"  │  Tracked → Full Segmentation:      {retention_full_seg:>6.1f}% retained ({pct_loss_to_full_seg:>5.1f}% lost)          │")
    print("  ├─────────────────────────────────────────────────────────────────────────────────┤")
    # Calculate ISS-Pheno yield (from ISS matched to full seg)
    iss_to_full_seg_pct = (funnel_full_seg / funnel_iss_matched * 100) if funnel_iss_matched > 0 else 0
    print(f"  │  OVERALL: {funnel_total_cells:,} → {funnel_full_seg:,} = {funnel_pct_full_seg:.1f}% final yield              │")
    print(f"  │  OVERALL (ISS-Pheno): {funnel_iss_matched:,} → {funnel_full_seg:,} = {iss_to_full_seg_pct:.1f}% yield       │")
    print("  └─────────────────────────────────────────────────────────────────────────────────┘")

    print("\n" + "=" * 80 + "\n")

    # Generate and save funnel graph (only when --save-reports)
    if save_reports:
        try:
            output_dir = Path(__file__).resolve().parent
            _save_funnel_graph(results, output_dir)
        except Exception as e:
            print(f"Warning: Failed to save funnel graph: {e}")

    # Split results into marker buckets by modality. Experiments are routed by
    # `experimental_design` (live-cell vs Cell_Painting vs 4i) and by project
    # tag (the config `project` field). Critically, a single experiment can appear in multiple
    # buckets — e.g. ops0094 has both live-cell Phase AND Cell Painting add-on
    # channels, so it contributes to the regular Phase row AND the CP rows.
    # Per-bucket marker filtering (below) ensures live-cell markers only show
    # up in the regular table and CP/4i markers only in their own tables.
    DEFAULT_PROJECT = "40_marker"

    def _project(r):
        return r.get("project") or DEFAULT_PROJECT

    def _is_cell_painting(r):
        return "Cell_Painting" in (r.get("experimental_design") or [])

    def _is_4i(r):
        return "4i" in (r.get("experimental_design") or [])

    cell_painting_results = [r for r in results if _is_cell_painting(r)]
    four_i_results = [r for r in results if _is_4i(r)]
    # Live-cell bucket: default-project experiments (they have live-cell Phase).
    # Each non-default project (config `project` field) gets its own marker table.
    regular_results = [r for r in results if _project(r) == DEFAULT_PROJECT]
    extra_projects = sorted({_project(r) for r in results} - {DEFAULT_PROJECT})

    # Marker filters classify markers by channel so each bucket only surfaces
    # its own modality's markers.
    def _is_cp_channel(ch):
        c = (ch or "").lower()
        return c.startswith("cp1_") or c.startswith("cp2_")

    def _is_4i_channel(ch):
        return (ch or "").lower().startswith("4i_")

    def _live_cell_marker(label, ch):
        # Anything that isn't a CP/4i channel is treated as live-cell.
        return not (_is_cp_channel(ch) or _is_4i_channel(ch))

    marker_buckets = [
        ("Cell Counts by Unique Marker", regular_results, _live_cell_marker),
    ]
    for _proj in extra_projects:
        marker_buckets.append((
            f"{_proj} Markers",
            [r for r in results if _project(r) == _proj],
            _live_cell_marker,
        ))
    marker_buckets += [
        ("Cell Painting Markers", cell_painting_results, lambda label, ch: _is_cp_channel(ch)),
        ("4i Markers", four_i_results, lambda label, ch: _is_4i_channel(ch)),
    ]

    # Compute per-bucket marker analysis. Restrict markers in every bucket so
    # only markers actually used by that bucket's experiments appear — otherwise
    # the regular table would still list every per-project/CP/4i channel.
    marker_analyses: list[tuple[str, list]] = []
    for title, bucket, mfilter in marker_buckets:
        bucket_analysis = _generate_marker_analysis(
            bucket,
            restrict_to_used_markers=True,
            marker_filter=mfilter,
        )
        if bucket_analysis:
            marker_analyses.append((title, bucket_analysis))

    # Keep `marker_analysis_results` (the regular bucket) for downstream
    # consumers that expect a single primary table.
    marker_analysis_results = next(
        (a for t, a in marker_analyses if t == "Cell Counts by Unique Marker"),
        [],
    )

    def _print_marker_table(title: str, analysis: list) -> None:
        print("=" * 100)
        print(f"MARKER ANALYSIS - {title}")
        print("=" * 100)
        print("\nNote: Positive control experiments [+CTRL] and need rescue experiments [RESCUE]")
        print("      are excluded from this aggregation.\n")

        marker_table = PrettyTable()
        marker_table.field_names = [
            "#",
            "Ch",
            "Marker",
            "# Exps",
            "Exps",
            "Total Cells",
            "ISS Matched Cells",
            "Tracked Cells",
            "Sum of Mean Cells/Gene",
        ]
        marker_table.align = "l"
        for col in ["#", "# Exps", "Total Cells", "ISS Matched Cells", "Tracked Cells", "Sum of Mean Cells/Gene"]:
            marker_table.align[col] = "r"

        for idx, marker_data in enumerate(analysis, start=1):
            total_cells_str = f"{marker_data['total_cells']:,}" if marker_data['total_cells'] else ""
            iss_matched_str = f"{marker_data['iss_matched_cells']:,}" if marker_data['iss_matched_cells'] else ""
            tracked_str = f"{marker_data['tracked_total_cells']:,}" if marker_data['tracked_total_cells'] else ""
            mean_cpg_str = f"{marker_data['mean_cells_per_gene']:.2f}" if marker_data['mean_cells_per_gene'] else ""

            if _normalize_marker_label(marker_data["marker"]) == "phase":
                exps_str = "-"
            else:
                exp_ids = marker_data.get("experiment_ids", [])
                exps_str = _format_exp_ids_as_ranges(exp_ids, max_individual=5)

            marker_table.add_row([
                idx,
                marker_data["channel_indicator"],
                marker_data["marker"],
                marker_data["experiment_count"],
                exps_str,
                total_cells_str,
                iss_matched_str,
                tracked_str,
                mean_cpg_str,
            ])

        print(marker_table)
        print("=" * 100 + "\n")

    for title, analysis in marker_analyses:
        _print_marker_table(title, analysis)

    # Project summary — uses the unfiltered all_results so the cross-project
    # breakdown stays complete even when --project filters the rest of the report.
    # Always recompute a unified (cross-modality) marker analysis here: the per-
    # bucket marker tables above each restrict to one modality, but Mean Cells/
    # Perturbation needs every marker available so each project's average has
    # complete inputs.
    full_marker_analysis = _generate_marker_analysis(all_results)
    project_summary = _generate_project_summary(all_results, full_marker_analysis)
    if project_summary:
        print("=" * 100)
        print("PROJECT SUMMARY - Cell Counts by Project")
        print("=" * 100)
        if project_filter:
            print(f"\nNote: --project {project_filter} filters the per-experiment table above,")
            print("      but this summary always reflects every project.\n")

        project_table = PrettyTable()
        project_table.field_names = [
            "Project",
            "# Exps",
            "Cell Lines",
            "Libraries",
            "# Perturbations",
            "Experimental Designs",
            "# Markers",
            "Total Cells",
            "ISS Matched Cells",
            "Tracked+Seg Cells",
            "Mean Cells/Perturbation",
        ]
        project_table.align = "l"
        for col in ["# Exps", "# Perturbations", "# Markers", "Total Cells", "ISS Matched Cells", "Tracked+Seg Cells", "Mean Cells/Perturbation"]:
            project_table.align[col] = "r"

        for p in project_summary:
            avg_cpg = p.get("avg_sum_cells_per_perturbation")
            num_pert = p.get("num_perturbations") or 0
            project_table.add_row([
                p["project"],
                p["experiment_count"],
                ", ".join(p["cell_lines"]) if p["cell_lines"] else "",
                ", ".join(p["libraries"]) if p["libraries"] else "",
                f"{num_pert:,}" if num_pert else "",
                ", ".join(p["experimental_designs"]) if p["experimental_designs"] else "",
                p["unique_markers"],
                f"{p['total_cells']:,}" if p["total_cells"] else "",
                f"{p['iss_matched_cells']:,}" if p["iss_matched_cells"] else "",
                f"{p['tracked_full_seg']:,}" if p["tracked_full_seg"] else "",
                f"{avg_cpg:.2f}" if avg_cpg else "",
            ])

        print(project_table)
        print("=" * 100 + "\n")

    # Save reports to the pipelinerunner folder (only when --save-reports)
    if not save_reports:
        return
    try:
        output_dir = Path(__file__).resolve().parent
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save plain text version (no colors, for compatibility)
        header = [
            "=" * 80,
            f"OPS Pipeline Status{' (FAST)' if fast else ''}",
            f"Generated: {timestamp}",
            "=" * 80,
        ]
        table_text = table.get_string()
        text_content = "\n".join(
            header
            + [
                table_text,
                "=" * 80,
                f"\n🔧 Summary: {manual_iss_pheno_registration_needed_count} experiment(s) need manual ISS→Pheno registration\n",
            ]
        )
        txt_path = output_dir / "pipeline_status.txt"
        with open(txt_path, "w") as f:
            f.write(text_content)
        print(f"Saved plain text report to: {txt_path}")

        # Save Markdown version with color indicators
        md_path = output_dir / "pipeline_status.md"

        # Build funnel data dict for markdown report (using filtered experiments only)
        funnel_data = {
            "total_cells": funnel_total_cells,
            "cells_with_reads": funnel_cells_with_reads,
            "iss_matched_cells": funnel_iss_matched,
            "tracked_total_cells": funnel_tracked,
            "tracked_full_seg": funnel_full_seg,
            "tracked_fallback_seg": funnel_fallback,
            "experiment_count": len(tracked_experiments),
        }

        _save_markdown_report(
            md_path,
            results,
            timestamp,
            fast,
            manual_iss_pheno_registration_needed_count,
            skipped_experiments,
            missing_steps_legend,
            iss_only_skipped,
            skip_iss_only,
            marker_analysis_results,
            funnel_data,
            bad_exps_skipped,
            project_summary=project_summary,
            project_filter=project_filter,
            marker_analyses=marker_analyses,
        )
        print(f"Saved Markdown report to: {md_path}")

        # Save CSV reports
        csv_path = output_dir / "pipeline_status.csv"
        _save_csv_pipeline_status(csv_path, results, missing_steps_legend, skip_iss_only)
        print(f"Saved pipeline status CSV to: {csv_path}")

        # One CSV per modality bucket (regular, per-project, Cell Painting, 4i).
        bucket_to_filename = {
            "Cell Counts by Unique Marker": "marker_analysis.csv",
            "Cell Painting Markers": "marker_analysis_cell_painting.csv",
            "4i Markers": "marker_analysis_4i.csv",
        }
        for title, analysis in marker_analyses:
            filename = bucket_to_filename.get(title)
            if not filename and title.endswith(" Markers"):
                # per-project bucket -> marker_analysis_<project>.csv
                slug = title[: -len(" Markers")].strip().lower().replace(" ", "_")
                filename = f"marker_analysis_{slug}.csv"
            if not filename:
                continue
            marker_csv_path = output_dir / filename
            _save_csv_marker_analysis(marker_csv_path, analysis)
            print(f"Saved marker analysis CSV to: {marker_csv_path}")

        if project_summary:
            project_csv_path = output_dir / "project_summary.csv"
            _save_csv_project_summary(project_csv_path, project_summary)
            print(f"Saved project summary CSV to: {project_csv_path}")
    except Exception as e:
        print(f"Warning: Failed to save pipeline status reports: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate OPS pipeline status report")
    parser.add_argument(
        "-fast",
        "--fast",
        action="store_false",
        help="Fast mode: only check for path existence (skip content/Zarr checks)",
    )
    parser.add_argument(
        "--debug-cells",
        action="store_true",
        help="Print debug information for cell count retrieval",
    )
    parser.add_argument(
        "--skip-iss-only",
        action="store_true",
        default=True,
        help="Skip ISS-only experiments (run_iss_only=True) (default: enabled)",
    )
    parser.add_argument(
        "--no-skip-iss-only",
        action="store_false",
        dest="skip_iss_only",
        help="Include ISS-only experiments in the report",
    )
    parser.add_argument(
        "--date-after",
        type=str,
        default=None,
        help="Only include experiments with date >= this value (format: YYYYMMDD, e.g., 20251027)",
    )
    parser.add_argument(
        "--date-before",
        type=str,
        default=None,
        help="Only include experiments with date <= this value (format: YYYYMMDD, e.g., 20251027)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cache and recompute all experiment data fresh",
    )
    parser.add_argument(
        "--cache-verbose",
        action="store_true",
        help="Print detailed cache hit/miss information for each experiment",
    )
    parser.add_argument(
        "--skip-bad-exps",
        action="store_true",
        default=True,
        help="Skip experiments excluded by bad_experiment.yaml (default: enabled)",
    )
    parser.add_argument(
        "--no-skip-bad-exps",
        action="store_false",
        dest="skip_bad_exps",
        help="Include all experiments regardless of date",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help=(
            "Only show experiments tagged with this project (the config `project` field). "
            f"Baseline values: {', '.join(KNOWN_PROJECTS)} (plus any project declared in the library map). Match is case-insensitive. "
            "The Project Summary table always shows every project regardless of this filter."
        ),
    )
    parser.add_argument(
        "--save-reports",
        action="store_true",
        help="Save txt/markdown/CSV reports and funnel graph to the pipelinerunner folder (off by default)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of threads for per-experiment file checks/CSV reads (I/O-bound; default: 16)",
    )
    args = parser.parse_args()

    # Normalize --project to the canonical casing in KNOWN_PROJECTS so users can
    # type a project name in any case (e.g. 'validation' matches 'Validation'). Hyphens and
    # underscores are treated as equivalent so 'paper-v1' matches 'paper_v1'.
    project_filter = args.project
    if project_filter:
        def _norm(s):
            return s.lower().replace("-", "_")
        canonical = next(
            (p for p in KNOWN_PROJECTS if _norm(p) == _norm(project_filter)),
            None,
        )
        project_filter = canonical or project_filter

    report_pipeline_status(
        fast=args.fast,
        debug_cells=args.debug_cells,
        skip_iss_only=args.skip_iss_only,
        date_after=args.date_after,
        date_before=args.date_before,
        no_cache=args.no_cache,
        cache_verbose=args.cache_verbose,
        skip_bad_exps=args.skip_bad_exps,
        project_filter=project_filter,
        save_reports=args.save_reports,
        workers=args.workers,
    )
    # to run: python -m cyclops_process.pipelinerunner.report_pipeline_status
    # to run with debug: python -m cyclops_process.pipelinerunner.report_pipeline_status --debug-cells
    # to run without cache: python -m cyclops_process.pipelinerunner.report_pipeline_status --no-cache
    # to see cache details: python -m cyclops_process.pipelinerunner.report_pipeline_status --cache-verbose
    # to include all experiments (disable --skip-bad-exps): python -m cyclops_process.pipelinerunner.report_pipeline_status --no-skip-bad-exps
