# cyclops_process

Python package implementing the OPS (Optical Pooled Screens) analysis pipeline: raw
acquisition → converted OME-Zarr → reconstruction, stitching, segmentation, tracking,
in-situ sequencing (ISS) base calling, and downstream feature extraction / QC.

For installation and how to run the pipeline, see the [top-level README](../README.md).
Entrypoint: `uv run python -m cyclops_process.processes.run --experiment <exp_name>`.

## Top-level modules

| Path | Purpose |
| --- | --- |
| `paths.py` | Single source of truth for the storage root (`BASE_PATH`, overridable with `OPS_BASE_PATH`). |

## Subdirectories

### [`processes/`](processes/README.md)
The pipeline steps themselves — each module is one stage that `processes/run.py` (the
CLI/DAG entrypoint) can execute standalone or as part of a full experiment run:
convert, `reconstruct*` (phase/label-free reconstruction, incl. subtile and
tilt-corrected variants), `flatfield_correction`, `ops_stitch`, `register`,
`segment`, `spots`, `iss` / `iss_merge` (base calling and read merging), `assemble`,
`virtual_staining`, and `inference`. Sub-packages group the more involved steps:

- `auto_register/` — automated cross-cycle / cross-channel registration (PCC, RANSAC,
  graph-based solving) plus its orchestrator and visualization; see
  `AUTO_REGISTER_README.md` and `REGISTRATION_LOGIC.md`.
- `cell_seg/` — nuclei and cell segmentation passes and their SLURM orchestrators.
- `track/` — cell tracking over timepoints and its orchestrator.
- `pyramids/` — multiscale OME-Zarr pyramid construction, resharding, contrast limits,
  overlays, and store auditing.

### [`pipelinerunner/`](pipelinerunner/README.md)
Orchestration layer above `processes/`: DAG execution (`dag_runner.py`), step-name →
callable resolution (`step_registry.py`), SLURM submission (`slurm_executor.py`),
per-experiment config generation, completion checking, and status reporting /
interactive monitoring of in-flight experiments. The step *dependency graph* itself is
not in Python — it lives in `configs/slurm_task_config.yaml`.

### [`convert/`](convert/README.md)
Raw acquisition → OME-Zarr conversion. Includes the v3 (Zarr v3) converters for fixed
and live-cell modalities, TIFF ingestion, warp handling, and metadata construction /
repair.

### `configs/`
Static, versioned configuration: channel maps (`ops_channel_maps.yaml`), per-modality
phase/reconstruction settings, registration settings per camera pair, normalization and
prediction configs for virtual staining, codebook/gene tables, segmentation parameters,
and an example experiment config. `slurm_task_config.yaml` does double duty as both the
pipeline dependency graph and the per-step SLURM resource spec.

### `data/`
Dataset construction and access helpers — building `OpsDataset`-backed tables from
stores, per-cell feature/measurement extraction, and linking ISS calls to tracks.

### `bc/`
Base-calling primitives: extracting per-spot intensities across cycles into the
tidy reads dataframe consumed by the ISS steps.

### [`metrics/`](metrics/README.md)
Quantitative QC. Top level holds generic metric implementations (reconstruction,
segmentation, tracking, SNR); `plate_stats/` holds the plate-wide ISS QC suite —
read matching, Hamming distances, confluency, entropy, flatfield, spatial coherence,
heatmaps, timing, and failed-round optimization.

### [`fixed_cp_4i/`](fixed_cp_4i/README.md)
Standalone numbered pipeline for the fixed-cell / 4i (iterative immunofluorescence)
modality: convert → project → flatfield → stitch → segment → register, plus linking
back to the live/ISS data. `configs/` holds its modality configs and `helpers/`
its parameter sweeps and one-off fixes.

### [`napari/`](napari/README.md)
Interactive napari viewers for inspecting results — Dask-backed store/grid viewers,
gene- and attention-based views, and embedding overlays.

### [`utils/`](utils/README.md)
Shared low-level helpers used across the pipeline: async HCS Zarr writing, projection,
normalization, reconstruction and segmentation utilities, waveorder helpers, prediction
combination, and store/metadata maintenance scripts. `batch/` holds batch variants.
