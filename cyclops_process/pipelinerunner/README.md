# pipelinerunner

Orchestration layer above [`processes/`](../processes): it decides *which* pipeline
steps run, *in what order*, *where* (locally or as SLURM jobs), and reports what has
already completed. The steps themselves live in `processes/`; nothing here does image
processing. The user-facing entrypoint is `processes/run.py`, which parses the CLI and
hands off to `orchestrator.py`.

Two facts define how the layer behaves:

- **Completion is file-based.** A step is complete iff every path returned by
  `OpsDataset.get_output_files_for_step(<step_key>, config)` exists and has content.
  Logs are written for provenance/timing only, never consulted for completion.
- **Step dependencies live in `configs/slurm_task_config.yaml`**, not in Python. Each
  key in that YAML is a step name with a `dependencies:` list plus its `slurm_params`
  (cpus, mem, gpus, timeout). `dag_runner.py` reads it as the single source of truth,
  so the same file defines both the graph and the resources.

## Source modules

| File | Purpose |
| --- | --- |
| `orchestrator.py` | Top-level entry called by `processes/run.py`: resolves an experiment name to its `*_config.yaml`, builds the full pipeline DAG (`_build_pipeline_dag`, registering each step with its config params and conditional flags for missing ISS/fluorescence data), then runs it either sequentially over the topological order or via the async DAG runner. Also loops over all experiment configs for `--all`. |
| `pipelinerunner.py` | `PipelineRunner` — the execution mechanics for one step: log-key generation, completion check, interactive prompts, local vs. SLURM dispatch, output waiting, and audit logging. Also holds `batch_submit_steps_all_experiments()`, the fast `--slurm-batch` path that submits one step across many experiments without the full orchestrator. |
| `dag_runner.py` | `DAGRunner` — registers steps declaratively, loads their dependencies from `slurm_task_config.yaml`, and executes them. Supports sequential, rerun-subset, and fully async modes (each step fires as soon as its own deps finish), plus per-step state tracking, preflight scanning, retry/skip prompts on failure, blocking of downstream steps, and optional manual checkpoint nodes. |
| `step_registry.py` | Flat registry mapping every step name to its `{module, function, needs_wells, needs_process, process}` metadata, with `get_step_function()` / `get_step_metadata()` / `list_all_steps()`. Used by the batch-submission path (and any caller that needs to resolve a step name to a callable) — it carries no dependency information. |
| `slurm_executor.py` | `SlurmExecutor`: builds submitit jobs for a step (env setup, CUDA detection, GPU/CPU monitoring, log dirs under `slurm_logs/`), submits them, polls `sacct`/`squeue` for state, prints a submission table and resource statistics, and provides `wait_for_outputs()` / `wait_for_virtual_staining_jobs()` for steps that fan out their own inner jobs. |
| `completion_checker.py` | `CompletionChecker`: the file-existence logic behind "is this step done" — per-method and per-well variants, Zarr-aware content validation (`.zgroup`/`.zarray`/`zarr.json`), an `_ALWAYS_RUN` set for steps that intentionally redo work, and the 🟢/🔴/⚪ status dot used in menus. |
| `interactive_menu.py` | `InteractiveMenu`: all terminal prompts — the completed-step action prompt (`[s]kip / [y]es re-run / [n]o / [a]ll / [f]ull list / [q]uit`), the checkpoint prompt, and the numbered full step list with per-step status and expected runtime/resources. |
| `piperun_utils.py` | Shared helpers used by `PipelineRunner`: log-key generation and selection matching, SLURM config lookup with well/method-suffix fallback, timeout scaling by well count, resource-string formatting, audit-log writes, and `Deferred` (a step parameter resolved at dispatch time rather than DAG-build time). |
| `dag_display.py` | `DAGDisplay`: the Nextflow-style in-place ANSI progress table shown during DAG runs. Redraws on a background thread, redirects each step's stdout/stderr to a per-step log file, and falls back to one line per state change on non-TTY. |
| `visualize_dag.py` | Standalone viewer for the dependency graph: `parse_dag()` (also imported by `dag_runner.py`) turns `slurm_task_config.yaml` into `{step: [deps]}`, and the CLI renders it as a colored tree, a flat level table, or Mermaid. |
| `exceptions.py` | `PipelineHalted` — raised inside a step to tell the runner to stop the whole pipeline cleanly (no stack trace, no retry prompt). |
| `generate_config_files.py` | Generates the per-experiment `*_config.yaml` files from a `DEFAULT_CONFIG` template plus overrides from the shared channel maps, failed-rounds, and library-map YAMLs. Runs for a single experiment, a named experiment group, or all experiments (backing up the existing config dir first). |
| `report_pipeline_status.py` | Cross-experiment status report: walks every experiment config, resolves each step's expected outputs, and prints per-experiment progress, cell-count funnels, marker tables, and a project summary. Has a hash-based cache, thread-pooled I/O, date/project filters, and an opt-in `--save-reports` that writes status artifacts (txt/markdown/CSV plus a cell-retention PNG) into this directory. Those artifacts and the `.pipeline_status_cache.json` cache are regenerable output rather than source, and are not intended to be committed. |

## Pipeline shape

The DAG has two roots, `convert_iss` and `convert_raw`, which open three branches that
run concurrently and then merge:

- **ISS**: `stack_symlinks` → drift correction / SNR → stitch estimate → segment +
  stitch → `register_iss_cycles` → `merge_spots_base_calling` → `convert_iss_to_v3` →
  metrics.
- **Tracking**: `link_tracking` → `correct_distortion` → reconstruct (+ tilt
  calibration/correction) → virtual staining (preprocess → inference → combine) →
  stitch estimate → `segment_and_stitch_track` → `estimate_and_stitch_track-2d`.
- **Phenotyping**: `link_phenotyping` → reconstruct (+ tilt) and fluorescence
  projection/flatfield → virtual staining → channel registration →
  `prepare_unified_pheno_tiles` → `estimate_and_stitch_pheno-2d` → `viscy_normalize`.

The branches join at `build_pyramids`, then continue through cell/nuclei segmentation,
`submit_registration_jobs` → `submit_tracking_jobs` → `link_calls_tracks` →
`fix_v3_stores`, and finish with `recompute_metrics` running in parallel with the
inference steps (`organelle_segmentation` → `op_feature_extraction`, `cp_features`,
`celldino_inference`).

To see the current graph rather than this summary:

```bash
uv run python -m cyclops_process.pipelinerunner.visualize_dag           # colored tree
uv run python -m cyclops_process.pipelinerunner.visualize_dag --mermaid # Mermaid
```

## Usage

Everything below is driven through `processes/run.py`; see its module docstring for the
full list.

```bash
# Full pipeline for one experiment (partial names allowed)
uv run python -m cyclops_process.processes.run --experiment ops0042_20250520

# Same, but with parallel DAG execution (independent branches run concurrently)
uv run python -m cyclops_process.processes.run --experiment ops0042_20250520 --dag

# Every experiment that has a *_config.yaml
uv run python -m cyclops_process.processes.run --all

# Re-run specific steps only (non-interactive)
uv run python -m cyclops_process.processes.run -e ops0042_20250520 --rerun base_calling get_metrics

# Steps are submitted to SLURM by default (--slurm-steps); opt out with:
uv run python -m cyclops_process.processes.run -e ops0042_20250520 --local-steps

# Fast batch submission of one step across all experiments (requires --rerun)
uv run python -m cyclops_process.processes.run --all --slurm-batch --rerun build_pyramids
uv run python -m cyclops_process.processes.run --all --slurm-batch --rerun build_pyramids --dry-run
uv run python -m cyclops_process.processes.run --all --slurm-batch --rerun build_pyramids --force --no-wait
```

Supporting entrypoints in this directory are invoked directly:

```bash
uv run python -m cyclops_process.pipelinerunner.generate_config_files [<experiment|group>]
uv run python -m cyclops_process.pipelinerunner.report_pipeline_status [--save-reports] [--no-cache]
```
