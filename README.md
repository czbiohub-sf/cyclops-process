# cyclops_process

Image-processing pipeline for Optical Pooled Screens (OPS) at CZ Biohub: raw microscope
acquisitions → stitched, segmented, registered OME-Zarr composites → per-cell sequencing
calls linked to tracked cell lineages.

Three co-acquired modalities are processed per experiment and merged into a single per-cell
record:

| Modality | Acquisition | Role |
| --- | --- | --- |
| **ISS** | multi-cycle 10× in situ sequencing | reads the perturbation barcode in each cell |
| **Phenotyping** | 20× volumetric brightfield + fluorescence | the morphological readout |
| **Tracking** | 5× single-plane brightfield timelapse | follows cells over time |

Each is converted to OME-Zarr, reconstructed, stitched into one composite per well,
segmented, and registered into a common frame. The end products are one Zarr v3 store per
modality per well (with multi-resolution pyramids and segmentation label groups) and a
per-well `linked_pheno_iss.csv` joining each tracked cell to its called gene/sgRNA.

Downstream single-cell feature extraction and analysis live in a separate submodule,
`cyclops_model` — this repo produces the images and calls that it consumes.

This repository accompanies the following preprint, and should be preserved and kept public
indefinitely:

> [A multimodal perturbation atlas defines the phenotypic resolution of cellular morphology](https://www.biorxiv.org/content/10.64898/2026.06.01.728087v1.abstract) — bioRxiv, 2026. doi:10.64898/2026.06.01.728087

## Data availability

The processed image datasets this pipeline produces are available for download through the
Biohub OPS Explorer portal:

> [OPS Explorer — perturbation atlas collection](https://biohub.ai/ops-explorer?collection=6a3f8b91-1c5e-4d3a-9b4c-f7e0a2d8b6f3)

---

## Installation

`cyclops_process` is **not a standalone package.** It is one submodule of the
[`czbiohub-sf/cyclops-monorepo`](https://github.com/czbiohub-sf/cyclops-monorepo) uv workspace and is
only supported as a piece of that larger project. Install the monorepo, not this repo on its
own:

```bash
git clone --recurse-submodules git@github.com:czbiohub-sf/cyclops-monorepo.git
cd cyclops-monorepo
uv sync
```

All commands are then run from the **monorepo root** (`cyclops-monorepo/`) with `uv run`, as in
every example below.

### Required configuration

The pipeline reads and writes under `$OPS_BASE_PATH`, which has **no default** — importing
`cyclops_process` raises a `RuntimeError` if it is unset, so a misconfigured run cannot write
into somebody else's storage:

```bash
export OPS_BASE_PATH="/path/to/ops_data"          # required
export OPS_INSTRUMENT_ROOT="/path/to/instrument"  # required by the conversion steps only
```

Optional overrides: `OPS_OUTPUT_BASE_DIR` (write outputs somewhere other than
`$OPS_BASE_PATH`), `OPS_CONFIGS_DIR` and `OPS_EXP_CONFIG_FILE` (a different configs directory
or a specific experiment config), `OPS_CUDA_SEARCH_DIRS` (extra CUDA toolkit roots for CuPy
kernel compilation), and `OPS_LOG_ROOTDIR` (central log mirror in operational mode).

### Dependencies

Declared dependencies and the optional extras live in [`pyproject.toml`](pyproject.toml);
Python 3.12 is required, plus CUDA 12.x for the GPU steps. Exact resolved versions for the
whole workspace are pinned in the monorepo's `uv.lock`, which is the authoritative
environment specification.

```bash
uv sync --extra viz       # napari overlays with geff/tracksdata
uv sync --extra tracking  # cell tracking (gurobipy, tracksdata, geff, hoct)
uv sync --extra model     # UMAP, HDBSCAN, plotly
uv sync --extra rapids    # GPU-accelerated analysis (cucim)
```

---

## Modules

Each module below has its own README with a per-file breakdown.

| Module | Purpose |
| --- | --- |
| [`processes/`](cyclops_process/processes/README.md) | **The pipeline steps themselves.** One module per stage — convert, reconstruct, flatfield, stitch, register, segment, spot detection, base calling, assemble, virtual staining, inference — plus sub-packages for the involved ones: `auto_register/` (cross-cycle and cross-modality registration), `cell_seg/`, `track/`, `pyramids/`. Also holds `run.py`, the CLI entry point. |
| [`pipelinerunner/`](cyclops_process/pipelinerunner/README.md) | **Orchestration above `processes/`.** Decides which steps run, in what order, locally or as SLURM jobs, and reports what has already completed. Completion is file-based; the step dependency graph lives in `configs/slurm_task_config.yaml`. |
| [`convert/`](cyclops_process/convert/README.md) | Raw acquisition → OME-Zarr: the Zarr v3 converters for fixed and live-cell modalities, TIFF ingestion, warp handling, and metadata construction/repair. |
| [`configs/`](cyclops_process/configs) | Static configuration — channel maps, per-modality reconstruction settings, registration settings per camera pair, codebook/gene tables, segmentation parameters, and `slurm_task_config.yaml`, which doubles as the pipeline DAG and the per-step SLURM resource spec. |
| [`metrics/`](cyclops_process/metrics/README.md) | Quantitative QC: generic metrics for reconstruction, segmentation, tracking and SNR, plus the plate-wide ISS suite in `plate_stats/` (read matching, Hamming distance, confluency, entropy, flatfield, spatial coherence, failed-round optimization). |
| [`data/`](cyclops_process/data) | Dataset construction and access — building tables from stores, per-cell measurement extraction, and linking ISS calls to tracks. |
| [`bc/`](cyclops_process/bc) | Base-calling primitives: per-spot intensity extraction across cycles into the tidy reads dataframe the ISS steps consume. |
| [`fixed_cp_4i/`](cyclops_process/fixed_cp_4i/README.md) | Standalone numbered pipeline for the fixed-cell / 4i (iterative immunofluorescence) modality, plus linking back to the live/ISS data. |
| [`napari/`](cyclops_process/napari/README.md) | Interactive viewers for inspecting results — Dask-backed store and grid viewers, gene- and attention-based views, embedding overlays. |
| [`utils/`](cyclops_process/utils/README.md) | Shared low-level helpers (async HCS Zarr writing, projection, normalization, reconstruction/segmentation/waveorder utilities) and maintenance scripts. |
| [`nextflow/`](nextflow/README.md) | Nextflow implementation of the ISS branch (`iss.nf` + `iss.config`) — an alternative to the Python orchestrator. |
| `paths.py` | The single storage root (`BASE_PATH`, from `$OPS_BASE_PATH`) and the lazily resolved instrument mount. |

---

## Running the pipeline

`processes/run.py` is the entry point for everything. It resolves an experiment name to its
`*_config.yaml`, builds the DAG, and executes it — sequentially, as a parallel DAG, or by
submitting each step to SLURM. Experiment names may be abbreviated (`ops0042` resolves to
`ops0042_20250520`).

```bash
# Full pipeline for one experiment
uv run python -m cyclops_process.processes.run --experiment ops0042_20250520

# Parallel DAG execution: independent branches run concurrently
uv run python -m cyclops_process.processes.run --experiment ops0042_20250520 --dag

# Every experiment that has a config
uv run python -m cyclops_process.processes.run --all
```

Steps are submitted to SLURM by default (`--slurm-steps`); pass `--local-steps` to run them
in-process instead. `--mode research` (the default) redirects outputs into a `/rerun/`
subdirectory so test runs cannot overwrite real data; `--mode operational` uses the real
output paths and mirrors logs centrally.

### Running individual steps

Name one or more steps with `--rerun`. This skips the interactive prompts and runs only what
you asked for:

```bash
# One step, one experiment
uv run python -m cyclops_process.processes.run -e ops0042_20250520 --rerun merge_spots_base_calling

# Several steps, executed in dependency order
uv run python -m cyclops_process.processes.run -e ops0042_20250520 --rerun base_calling get_metrics

# One step across every experiment, with live progress (fast batch path)
uv run python -m cyclops_process.processes.run --all --slurm-batch --rerun build_pyramids
uv run python -m cyclops_process.processes.run --all --slurm-batch --rerun build_pyramids --dry-run
```

Useful flags: `--force` (re-run even when outputs already exist), `--dry-run` (show
submissions without sending them), `--no-wait` (submit and return immediately),
`--local-parallel` (run all experiments concurrently on this node; requires `--rerun`),
`--slurm-tag` (tag submitted jobs with a QoS).

A step counts as complete when every output file it declares exists and has content, so
re-running a finished step is a no-op unless you pass `--force`. There are 56 registered
steps; to see the current graph and the step names:

```bash
uv run python -m cyclops_process.pipelinerunner.visualize_dag            # colored tree
uv run python -m cyclops_process.pipelinerunner.visualize_dag --mermaid  # Mermaid
```

Most step modules also expose their own CLI for one-off work outside the DAG — see the
per-module READMEs above.

### Supporting entry points

```bash
# Generate the per-experiment *_config.yaml files
uv run python -m cyclops_process.pipelinerunner.generate_config_files [<experiment|group>]

# Cross-experiment progress report
uv run python -m cyclops_process.pipelinerunner.report_pipeline_status [--save-reports]
```

---

## Tests

A CPU-only unit test suite lives under [`tests/`](tests/) and runs on every push and PR via
[`.github/workflows/tests.yml`](.github/workflows/tests.yml):

```bash
uv run pytest
```

It is intentionally scoped to pure-Python modules with no GPU dependency so it can run on a
stock CI runner. Heavier modules — `processes/segment.py`, `processes/iss.py`, anything
touching `cellpose`/`cupy`/`torch`/`viscy` — are not covered; test those by running the
relevant pipeline step on a real experiment. Tests marked `real_data` are deselected by
default and need `$OPS_REFERENCE_BASE` (a reference experiment cache) and
`$OPS_TEST_TMP_BASE` (a directory visible to compute nodes); they skip cleanly when unset.

After a pipeline run, [`tests/QC/compare_plate_stats.py`](tests/QC/compare_plate_stats.py)
compares the resulting `plate_stats.csv` against a reference and flags any metric drifting by
more than 0.5% relative — exit code 1 on failure, so it is suitable for CI gating.

```bash
uv run python tests/QC/compare_plate_stats.py <candidate-plate_stats.csv> --reference <ref.csv>
```

---

## Ownership and maintenance

**This repository is the result of work done at [biohub San Francisco](https://github.com/czbiohub-sf).**

This repository is owned by the [Leonetti group](https://biohub.org/leonetti/) at
[biohub San Francisco](https://github.com/czbiohub-sf).

Maintainers (see also [`.github/CODEOWNERS`](.github/CODEOWNERS)):

- Alexander Hillsley ([@ahillsley](https://github.com/ahillsley))
- Gav Sturm ([@gav-sturm](https://github.com/gav-sturm))

Please open an issue or pull request for questions, bugs, or contributions.

## License

BSD 3-Clause — see [`LICENSE`](LICENSE).
