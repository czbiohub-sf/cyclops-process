# utils

Shared low-level helpers used across the pipeline — async HCS Zarr writing, Z projection,
normalization statistics, reconstruction / segmentation / waveorder utilities, and
virtual-staining prediction combination. It also collects a handful of one-off
maintenance and migration scripts that operate over stores and metadata rather than being
imported by pipeline steps.

## Importable library helpers

Modules meant to be imported. Consumers listed where a pipeline step or orchestrator
imports them directly.

| File | Purpose | Used by |
| --- | --- | --- |
| `async_hcs_writer.py` | `AsyncHCSPredictionWriter` — viscy-style HCS OME-Zarr prediction writer that overlaps GPU compute with I/O via a thread pool. Optionally pre-creates all positions with the input store's `(T,C,Z,Y,X)` shape so concurrent writes never need to resize. | `nextflow/bin/viscy_multigpu_inference.py` |
| `combine_batch_predictions.py` | Reassembles the raw `batch_NNNNN` tensors written by batched viscy inference into a proper HCS OME-Zarr store, reproducing the inference position ordering (sorted row → well → position). Exposes the phases separately (`create_combine_output_store`, `combine_single_batch_store`, `validate_combine_output`) plus the all-in-one `combine_predictions`, and includes a v2→v3 plate-scaffold reconciliation for fast-mv'd stores. Also runnable standalone (see `combine_batch.sh`). | `processes/virtual_staining.py` |
| `fast_normalize.py` | Bounded-I/O replacement for viscy's `generate_normalization_metadata`. Samples a capped set of chunk-aligned blocks instead of reading full Y*X slices (which times out on ~100k×100k assembled canvases) and writes byte-compatible `normalization` metadata using viscy's own `get_val_stats` / `write_meta_field`. Has a `benchmark_sampling` CLI for measuring read amplification. | `processes/assemble.py` |
| `project.py` | `create_max_projection` — parallel Z projection (max or sum) of a store into a 2D store, for `lc_20x` / `lc_5x` virtual staining and `lc_20x_fluor` raw fluorescence. `slices` can be `"all"`, an explicit plane list, or `"focus"`/`"auto"`, which derives per-FOV in-focus ranges from the tilt-calibration `tilt_params_*.csv` z-offsets. | `processes/assemble.py`; registered directly as the `create_max_projection_lc_20x` / `create_max_projection_lc_20x_fluor` DAG steps in `pipelinerunner/step_registry.py` and `pipelinerunner/orchestrator.py` |
| `recon_utils.py` | Reconstruction support for the 2D-autofocus and subtile reconstruction steps: path/config setup for `pheno-2d` and `track-2d`, output store creation (`Phase2D` + `Focus3D` channels), subtile grid bounds, a locked on-disk transfer-function cache keyed on optics + z-offset + shape, `_infer_focus_index` (waveorder transverse-band focus finding), and subtile metadata CSV / report generation. | `processes/reconstruct.py`, `processes/reconstruct_subtile.py` |
| `segmentation_utils.py` | Torch label-image utilities adapted from InstanSeg: label relabeling (`torch_fastremap`), sparse one-hot encoding, sparse dual IoU, tile-to-tile label matching (`match_labels`, plus a numpy `match_labels_fast` for many-small-object tiles), and edge-label removal. | `processes/segment.py`, `processes/cell_seg/cell_segmentation.py` |
| `waveorder_utils.py` | Waveorder integration layer: Pydantic models for the phase reconstruction config (`ReconstructionConfig` and friends), `model_to_yaml` / `yaml_to_model`, lazy loading of `focus_from_transverse_band` (preferring a GPU-enabled build at `$OPS_GPU_WAVEORDER_PATH`), sub-pixel-focus capability detection, and transfer-function store validation. | `processes/reconstruct.py`, `processes/reconstruct_subtile.py`, `utils/recon_utils.py` |

## Maintenance / migration scripts

Operational scripts, not pipeline steps. All default to dry-run where they mutate data.

- **`audit_v3_channels.py`** — Compares each experiment's v3 pheno store channels against
  `configs/ops_channel_maps.yaml` and flags stores missing expected fluorescent channels
  (`GFP`, `mCherry`), also reporting the v2 metadata channel count vs the actual v2 array
  `C` dimension. `--fix` prints the rebuild/re-stitch/re-convert commands rather than
  running them. Its `check_missing_fluor_channels` is imported by
  `processes/pyramids/store_audit.py`.
  `uv run python -m cyclops_process.utils.audit_v3_channels [--fix]`
- **`batch_rm_vs_stores.py`** — Deletes the large, regenerable `virtual_staining/` stores
  for `track` and/or `pheno` in parallel, guarded on the downstream product existing
  (`lc_5x_segmentation` for track, `pheno_assembled_v3` for pheno). Uses
  `async_delete_path` (rename + background rm) to avoid blocking on NFS unlinks; dry-run
  unless `--execute`.
  `uv run python -m cyclops_process.utils.batch_rm_vs_stores [--process pheno] [--show-paths] [--execute]`
- **`dedupe_linked_pheno_iss.py`** — Dedupes `*_linked_pheno_iss.csv` down to one row per
  `(well, tile_pheno, segmentation_id)` cell: drops NaN-`segmentation_id` fallback-bbox
  rows, keeps singlets (exactly one distinct `barcode`), and drops multiplets entirely.
  `--write-back` overwrites in place after backing up to `*.prededup.csv`. `dedupe_linked_csv`
  is imported by `data/datasets.py`.
  `uv run python -m cyclops_process.utils.dedupe_linked_pheno_iss --experiment ops0032_20250428 [--write-back]`
- **`scan_fix_stray_v2_metadata.py`** — Walks every `*_v3` store and removes stray zarr-v2
  markers (`.zgroup`, `.zarray`, `.zattrs`, `.zmetadata`) that v2-era readers/writers leave
  behind and that break strict readers such as napari's. Only touches stores whose root has
  a `zarr.json`, records the content of every deleted file to a restore manifest, and can
  fan out one SLURM job per experiment.
  `uv run python -m cyclops_process.utils.scan_fix_stray_v2_metadata [-e 180] [--fix] [--slurm]`

## `batch/`

Batch-over-all-experiments variants of the above; both are also imported for their
per-experiment/per-well functions.

- **`batch_collect_drift_trajectories.py`** — Gathers ISS cycle registration drift across
  all experiments from the cumulative affine YAMLs, computes drift magnitude/direction
  statistics, and writes `drift_trajectory_full_data.csv` +
  `drift_trajectory_summary.csv` plus a large set of plots (per-round, per-well,
  drift-vs-ISS-matching correlation, nucleus-registration comparison, drift over time)
  under `{BASE_PATH}/ops_data_report/<date>/drift_trajectories/`. Its
  `collect_drift_data_for_well` is imported by `metrics/plate_stats/iss_register_metrics.py`.
  `uv run python -m cyclops_process.utils.batch.batch_collect_drift_trajectories --reference-exp ops0107_20251208`
- **`batch_symlink_nuclear_seg.py`** — Symlinks nuclear segmentation from the per-modality
  segmentation store into the `pheno`, `iss`, or `track` destination store (v3-native
  destinations by default, `--zarr-version 2` for legacy). Auto-detects experiments needing
  symlinks from the experiment-configs dir unless `--experiments` is given; no upscaling.
  Called in-process by `processes/pyramids/audit_fix.py` and emitted as a fix command by
  `processes/pyramids/store_audit.py`.
  `uv run python -m cyclops_process.utils.batch.batch_symlink_nuclear_seg --experiments ops0036_20250505 --symlink-target pheno --force`

## Shell scripts

- **`combine_batch.sh`** — SLURM wrapper (`sbatch combine_batch.sh <intermediate_dir>
  <input_store> <output_store>`, 16 CPUs / 64 GB / 2 h) around
  `cyclops_process.utils.combine_batch_predictions`.
- **`mirror_experiment.sh`** — Mirrors an experiment directory into a destination base
  (default `$OPS_OUTPUT_BASE_DIR/reruns`) as a symlink tree, so downstream steps can be
  re-run against the mirror without touching the source. `--steps 0,1,2,3` selects which
  stage directories to mirror, `--jobs N` sets the parallel symlink workers (auto-detected
  from `cyclops_utils.hpc.resource_manager` otherwise), `--dry-run` reports only. Prints the
  `OPS_OUTPUT_BASE_DIR` / `OPS_FAST_OUTPUT_BASE_DIR` exports needed to use the mirror.
