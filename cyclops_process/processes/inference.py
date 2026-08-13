"""End-of-pipeline model inference steps, run in parallel with recompute_metrics.

Three independent branches on the pheno DAG:
  - Organelle Profiler (OP): submit_organelle_segmentation_jobs -> op_feature_extraction
  - CellProfiler (CP):       cp_features
  - CellDINO:                celldino_inference

CP and CellDINO are config-driven; each step generates its per-experiment YAML under
configs/inference_configs/<model>/v2/ at runtime (if absent) from the experiment's
channels/wells, then hands off to the model package's own SLURM driver. Heavy model
imports are done lazily inside each step so importing this module (and the orchestrator)
stays CPU/GPU-free.
"""

from pathlib import Path
from types import SimpleNamespace

import yaml

from cyclops_utils.data.experiment import OpsDataset
from cyclops_process.paths import BASE_PATH

# fast_ops is a symlink to icd.fast.ops; both resolve here.
INFERENCE_CONFIG_ROOT = Path(
    f"{BASE_PATH}/configs/inference_configs"
)


# ── config-derivation helpers ────────────────────────────────────────────────

def _experiment_config(experiment: str) -> dict:
    """Load the experiment's generated pipeline config (for channel_map/wells/project)."""
    cfg_path = OpsDataset(experiment).config_paths["exp_config"]
    if not cfg_path.exists():
        return {}
    try:
        return yaml.safe_load(cfg_path.read_text()) or {}
    except Exception:
        return {}


def _wells(experiment: str, config: dict | None) -> list[str]:
    """Wells to process, as 'A/1/0' position strings."""
    wells = (config or {}).get("wells_to_process")
    if wells:
        return list(wells)
    ds = OpsDataset(experiment)
    return sorted(f"{w}/0" for w in ds.infer_wells())


def _out_channels(config: dict | None) -> list[str]:
    """Real image channels for feature extraction: Phase2D base + fluor when present."""
    cmap = (config or {}).get("channel_map", {}) or {}
    channels = ["Phase2D"]
    # Order matches the checked-in v2 configs: mCherry before GFP.
    for fluor in ("mCherry", "GFP"):
        if cmap.get(fluor):
            channels.append(fluor)
    return channels


def _guide_col(config: dict | None) -> str:
    """Guide/perturbation column: the experiment's configured gene-name output
    column (from ops_library_map.yaml) if set, else the default `sgRNA`."""
    return (config or {}).get("gene_name_output_column") or "sgRNA"


def _use_reporter_names(config: dict | None) -> bool:
    """Whether to split output files by reporter name. Screens with a custom
    gene-name column (label-free / non-sgRNA perturbation names) skip reporter
    splits by default; override explicitly via config `use_reporter_names`."""
    cfg = config or {}
    return bool(cfg.get("use_reporter_names", not cfg.get("gene_name_output_column")))


def _cell_type(config: dict | None) -> str:
    return (config or {}).get("cell_line", "A549")


def _output_dir(experiment: str, subdir: str) -> str:
    """Assembly output dir for an experiment, resolved to the real icd.fast.ops path
    (not the icd.fast.ops symlink)."""
    return str((OpsDataset(experiment).results / subdir).resolve())


def _write_config(path: Path, cfg: dict, regenerate: bool = False) -> Path:
    """Write cfg to path (YAML) unless it already exists (unless regenerate)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not regenerate:
        print(f"[inference] reusing existing config: {path}")
        return path
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    print(f"[inference] wrote config: {path}")
    return path


# ── config builders (mirror the checked-in v2/ examples) ─────────────────────

def build_cp_config(experiment: str, config: dict | None) -> dict:
    return {
        "data_manager": {
            "experiments": {experiment: _wells(experiment, config)},
            "batch_size": 256,
            "data_split": [1, 0, 0],
            "out_channels": _out_channels(config),
            "initial_yx_patch_size": [256, 256],
            "final_yx_patch_size": [128, 128],
            "balanced_sampling": False,
        },
        "chunk_size": 100,
        "wait_for_completion": False,
        "output_dir": _output_dir(experiment, "cell-profiler"),
        "cell_type": _cell_type(config),
        # array job hardcodes gres gpu:1 -> must be a gpu partition
        "slurm_partition": "gpu",
        "processing": {
            "cell-profiler": True,
            "max_nan_features_per_cell": 0,
            "use_reporter_names": _use_reporter_names(config),
            "guide_col": _guide_col(config),
        },
    }


def build_celldino_config(experiment: str, config: dict | None) -> dict:
    return {
        "model_type": "cell_dino",
        "dataset_type": "basic",
        "data_manager": {
            "experiments": {experiment: _wells(experiment, config)},
            "batch_size": 256,
            "data_split": [0, 0, 1],
            "out_channels": _out_channels(config),
            "initial_yx_patch_size": [256, 256],
            "final_yx_patch_size": [160, 160],
            "cell_masks": False,
            "num_workers": 20,
        },
        "output_dir": _output_dir(experiment, "cell_dino_features_v2"),
        "cell_type": _cell_type(config),
        "processing": {
            "use_reporter_names": _use_reporter_names(config),
        },
        "slurm": {
            "partition": "gpu",
            "gres": "gpu:1",
            "cpus_per_task": 20,
            "mem": "36G",
            "time": "8:00:00",
            "constraint": "h100|h200|6000_blackwell",
        },
    }


# ── DAG step entrypoints (func(experiment, **params)) ────────────────────────

def organelle_segmentation(experiment, wait: bool = True, force: bool = False,
                           positions=None, channels=None, **_):
    """OP Stage 1: organelle segmentation (1 job per well x channel). Writes the
    organelle label arrays into phenotyping_v3.zarr."""
    from organelle_profiler.organelle_seg.organelle_segmentation_slurm import (
        submit_organelle_segmentation_jobs,
    )
    return submit_organelle_segmentation_jobs(
        experiment, positions=positions, channels=channels,
        wait_for_completion=wait, force=force,
    )


def op_feature_extraction(experiment, wait: bool = True, split: bool = True,
                          partition: str = None, force: bool = False,
                          dry_run: bool = False, **_):
    """OP Stage 2: organelle feature extraction (GPU -> CPU -> aggregate).

    Runs in --split mode by default (the CLI default): the GPU/CPU/aggregate phases
    are submitted as separate dependent SLURM jobs. Depends on
    submit_organelle_segmentation_jobs (Stage 1). The driver reads all other options
    via getattr defaults, so only the fields that diverge from those are set here;
    `split` must be explicit because getattr falls back to False, not the CLI's True.
    """
    from organelle_profiler.feature_extraction.feature_extraction_slurm import (
        submit_feature_extraction_jobs,
    )
    args = SimpleNamespace(
        experiment=experiment,
        split=split,
        resume_from=None,
        stop_after=None,
        aggregate_only=False,
        cells_csv=None,
        dry_run=dry_run,
        force=force,
        max_concurrent=None,
        no_wait=not wait,
        partition=partition,
    )
    return submit_feature_extraction_jobs(experiment, args)


def cp_features(experiment, wait: bool = True, regenerate_config: bool = False, **_):
    """CellProfiler feature extraction. Generates the per-experiment v2 config, then
    submits the self-chaining array -> concat -> anndata SLURM jobs."""
    from cyclops_model.models.cellprofiler.cp_features import cp_features_main

    config = _experiment_config(experiment)
    cfg_path = INFERENCE_CONFIG_ROOT / "cell-profiler" / "v2" / f"{experiment}_cp_v2.yml"
    _write_config(cfg_path, build_cp_config(experiment, config), regenerate_config)
    return cp_features_main(str(cfg_path), wait_for_completion=wait)


def celldino_inference(experiment, force: bool = False, regenerate_config: bool = False, **_):
    """CellDINO embedding inference. Generates the per-experiment v2 config, then submits
    per-(experiment, channel) SLURM jobs via the embedding batch driver."""
    from cyclops_model.features.batch_process_embeddings import batch_process_slurm

    config = _experiment_config(experiment)
    cfg_path = INFERENCE_CONFIG_ROOT / "cell-dino" / "v2" / f"{experiment}_v2.yml"
    _write_config(cfg_path, build_celldino_config(experiment, config), regenerate_config)
    return batch_process_slurm(config_path=str(cfg_path), force_reprocess=force)
