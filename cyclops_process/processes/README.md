# processes

The pipeline steps themselves: each module here implements one (or a few) stages of the
OPS pipeline — conversion hand-off, phase reconstruction, flatfield correction, stitching,
segmentation, registration, ISS base calling, tracking, pyramids, and end-of-pipeline
inference.

`processes/` holds the *work*; [`../pipelinerunner/`](../pipelinerunner/) holds the
*orchestration*. Step names are mapped to the functions below in
[`../pipelinerunner/step_registry.py`](../pipelinerunner/step_registry.py), the DAG edges
(dependencies) live in [`../configs/slurm_task_config.yaml`](../configs/slurm_task_config.yaml),
and `../pipelinerunner/orchestrator.py` wires them into a `DAGRunner`. `run.py` in this
directory is the CLI entrypoint for all of it.

## Running a step

`run.py` is a thin argparse wrapper around the orchestrator. Examples (from its docstring):

```bash
# Full pipeline for one experiment (partial names allowed)
uv run python -m cyclops_process.processes.run --experiment ops0042_20250520

# All experiments that have a *_config.yaml
uv run python -m cyclops_process.processes.run --all

# Re-run specific steps only
uv run python -m cyclops_process.processes.run -e ops0042_20250520 --rerun base_calling get_metrics

# Parallel DAG execution (independent ISS / track / pheno branches run concurrently)
uv run python -m cyclops_process.processes.run -e ops0042_20250520 --dag

# Batch-submit one step across all experiments, with live progress
uv run python -m cyclops_process.processes.run --all --slurm-batch --rerun build_pyramids [--force|--dry-run|--no-wait]
```

Steps are submitted to SLURM by default (`--slurm-steps`, resources from
`slurm_task_config.yaml`); pass `--local-steps` to run in-process. `--mode research`
(the default) redirects `OPS_OUTPUT_BASE_DIR` / `OPS_FAST_OUTPUT_BASE_DIR` into a
`rerun/` subdirectory so test runs cannot overwrite production data; `--mode operational`
uses real paths and dual-writes logs. Other useful flags: `--auto` (run every incomplete
step, no prompts), `--no-preflight`, `--experiments 46,47,52`, `--slurm-tag <qos>`.

Several modules also expose their own `argparse` CLI for standalone/debug runs, e.g.:

```bash
uv run python -m cyclops_process.processes.ops_stitch estimate_and_stitch --experiment ops0105_20260106 --process pheno-2d
uv run python -m cyclops_process.processes.reconstruct_tilt_corrected calibrate --experiment ops0105_20260106 --well A/1
uv run python -m cyclops_process.processes.segment segment_and_stitch --experiment ... --process iss
uv run python -m cyclops_process.processes.virtual_staining all -e 117 -p pheno
```

Modules with a `__main__`: `ops_stitch` (`estimate_and_stitch` / `estimate_stitch_parameters` /
`stitch`), `reconstruct` (`reconstruct` / `correct_distortion`), `reconstruct_tilt_corrected`
(`calibrate` / `reconstruct` / `submit` / `audit` / `summarize`), `register`
(`prepare_unified_pheno_tiles`, `build_unified_pheno_tiles_symlink`,
`build_phase_family_tiles_symlink`, `register_stitched_fluor_to_phase`,
`register_fluor_2d_tiles`), `segment` (`segment_and_stitch` / `segmentation` /
`upscale_nuclear_segmentations`), `virtual_staining`, `flatfield_correction`,
`viscy_batch_inference`, plus `pyramids/launcher.py`, `pyramids/audit_fix.py`, and the
`cell_seg/` / `track/` / `auto_register/` orchestrators and workers.

## Pipeline steps, in canonical order

Three branches run concurrently — ISS, tracking (5x), phenotyping (20x) — then converge at
`build_pyramids`. Steps marked *(elsewhere)* are registered in the DAG but implemented
outside this directory; they are listed to keep the sequence readable.

### ISS branch

| Step | Implementation | What it does |
| --- | --- | --- |
| `convert_iss` | *(elsewhere)* `convert/tiff_to_zarr.py` | Raw ISS OME-TIFF → one OME-Zarr store per sequencing round. |
| `stack_symlinks` | `assemble_link.stack_symlinks` | Assembles the per-round zarr stores into a single store with rounds as timepoints (symlinking chunks). Handles the optional pre-DAPI round (`pre_nuclei_round` / `skip_pre_dapi_round`). Writes the `iss` store. |
| `iss_snr_bimodal` | *(elsewhere)* `metrics/plate_stats/` | Per-round SNR / bimodality QC on the stacked store. |
| `correct_cycle_drift` | `register.correct_cycle_drift` | Estimates and applies a per-well mean XY drift between rounds. Reads the `iss` store, writes `iss_drift_corrected`. |
| `estimate_stitch_parameters_iss` | `ops_stitch.estimate_stitch_parameters` | Solves globally optimal tile positions from `iss_drift_corrected` and writes the stitch config YAML (`config_paths["iss_stitch"]`) — no pixels moved. |
| `segment_and_stitch_iss` | `segment.segment_and_stitch` | Two-phase: Cellpose-segments every tile in parallel (Dask), then stitches the label tiles per well using the estimated shifts, resolving labels across seams. Writes `bc_segmentation.zarr` (`iss_segmentation`). |
| `estimate_and_stitch_iss` | `ops_stitch.estimate_and_stitch` | Assembles the stitched ISS mosaic (`bc_stitched.zarr`) from the drift-corrected tiles and symlinks the segmentation labels into it. |
| `register_iss_cycles` | `auto_register/iss_cycle_register_orchestrator.register_iss_cycles` | Fans out per-well SLURM jobs that register every round to the anchor round (plus nucleus→round0, segmentation→nucleus) and compose cumulative affines. Writes `register/transforms/<well>/*.yml` and QA overlays; pre-creates the registered zarr. |
| `merge_spots_base_calling` | `iss_merge.merge_spots_base_calling` | Per-well job running `apply_iss_transforms` → `spots.detect_spots` → `iss.base_calling` back-to-back with the registered `(T,C,Z,Y,X)` array held in `shared_memory`, avoiding two NFS round-trips. Writes `bc_stitched_registered.zarr`, per-well spot `.npy`, and per-well reads CSVs. |
| `convert_iss_to_v3` | *(elsewhere)* `convert/v3_livecell.py` | Converts the ISS results into the Zarr v3 store layout. |
| `optimize_failed_rounds` | *(elsewhere)* `metrics/plate_stats/` | Searches for round subsets that maximise barcode matching, per well. |
| `get_metrics` | *(elsewhere)* `metrics/metrics.py` | ISS QC metrics for the branch. |

Supporting modules on this branch, called by the merge step rather than registered directly:
`spots.py` (Spotiflow spot detection on the anchor round — `detect_spots`, plus the
in-memory per-well variant and intensity normalization helpers) and `iss.py`
(`base_calling`: per-spot intensities across rounds → nucleotide calls; handles
non-contiguous / failed rounds and 9- vs 10-round experiments; writes the per-well
`reads` CSV).

### Raw conversion (feeds both live-cell branches)

| Step | Implementation | What it does |
| --- | --- | --- |
| `convert_raw` | *(elsewhere)* `convert/raw_to_zarr.py` | Raw live-cell acquisition → `raw_convert/` OME-Zarr stores. |

### Tracking branch (5x)

| Step | Implementation | What it does |
| --- | --- | --- |
| `link_tracking` | `assemble_link.link_tracking` | Reorganizes/renames the 5x `tracking_*.zarr` chunks into the `lc_5x` store with the FOV position names stitching requires (symlink-based). Refuses to run if the source `raw_convert` stores are not fully written. |
| `correct_distortion` | `reconstruct.correct_distortion` | Applies the pre-calibrated pin-cushion distortion correction (CuPy `map_coordinates` when a GPU is available) to the 5x brightfield tiles. |
| `reconstruct_track` | `reconstruct.reconstruct` | waveorder phase reconstruction of the 5x brightfield z-stacks (3D and/or 2D autofocus, optionally subtile-based via `reconstruct_subtile.py`). |
| `calibrate_tilt_track` | `reconstruct_tilt_corrected.calibrate_tilt` | Fits the per-run tilt model (zenith/azimuth/z-offset) by chain calibration along four cardinal spokes from the centre tile; one process per well. Writes `model.yaml` + `all_points.yaml`. |
| `reconstruct_tilt_corrected_track` | `reconstruct_tilt_corrected.reconstruct_tilt_corrected` | Applies the calibrated tilt model with per-subtile optimization, producing the tilt-corrected 3D and 2D phase stores. Runs its own internal SLURM scheduler with a work-stealing queue. |
| `virtual_staining_preprocess_track` | `virtual_staining.virtual_staining_preprocess` | Computes ViSCy normalization statistics/metadata over the input store (bounded sampling; ~15–45 min). |
| `virtual_staining_inference_track` | `virtual_staining.virtual_staining_inference` | Submits SLURM array jobs that run batched ViSCy inference (`viscy_batch_inference.py`) writing per-batch intermediates, then combines them. |
| `virtual_staining_combine_only_track` | `virtual_staining.virtual_staining_combine_only` | Combine-only rerun: merges the intermediate VS outputs into the final `tracking_vs.zarr` without re-running inference. |
| `estimate_stitch_parameters_track` | `ops_stitch.estimate_stitch_parameters` | Tile-position solve for the tilt-corrected 5x tiles → stitch config YAML. |
| `segment_and_stitch_track` | `segment.segment_and_stitch` | Cellpose segmentation + stitching of the 5x tiles (needs `tracking_vs.zarr`) → `tracking_segmentation_stitched.zarr`. |
| `estimate_and_stitch_track-2d` | `ops_stitch.estimate_and_stitch` | Assembles the stitched 2D tracking mosaic and attaches the segmentation labels as a symlink. |

### Phenotyping branch (20x)

| Step | Implementation | What it does |
| --- | --- | --- |
| `link_phenotyping` | `assemble_link.link_phenotyping` | Reorganizes/renames the 20x `phenotyping_well_*.zarr` chunks into the `lc_20x` store with stitching-compatible FOV names, applying the per-channel flip/rot90 orientation parameters. |
| `create_max_projection_lc_20x_fluor` | *(elsewhere)* `utils/project.py` | Max-projects the 20x fluorescence z-stacks to 2D. |
| `correct_flatfield_fluor` | `flatfield_correction.correct_flatfield` | Estimates per-channel illumination profiles from the images themselves (Gaussian-smoothed sampled mean, max-normalized so only dim regions are boosted; camera offset subtracted and re-added) and writes `phenotyping_fluor_2d_flatfield_corrected.zarr`, optionally saving profiles and debug comparisons. |
| `reconstruct_pheno` | `reconstruct.reconstruct` | waveorder phase reconstruction of the 20x brightfield z-stacks. |
| `calibrate_tilt_pheno` | `reconstruct_tilt_corrected.calibrate_tilt` | Same tilt calibration as the track branch, on 20x data. |
| `reconstruct_tilt_corrected_pheno` | `reconstruct_tilt_corrected.reconstruct_tilt_corrected` | Tilt-corrected 3D + 2D phase reconstruction, producing `phenotyping_phase_2d_optimized.zarr` and its 3D sibling. |
| `virtual_staining_preprocess_pheno` / `_inference_pheno` / `_combine_only_pheno` | `virtual_staining.*` | Same three ViSCy stages as the track branch, at 20x/3D. |
| `create_max_projection_lc_20x` | *(elsewhere)* `utils/project.py` | Max-projects the VS output to 2D. |
| `estimate_stitch_parameters_pheno` | `ops_stitch.estimate_stitch_parameters` | Tile-position solve for the tilt-corrected 20x tiles → stitch config YAML. |
| `submit_channel_registration_jobs` | `auto_register/channel_reg/channel_register.py` | Automatically estimates the fluorescence→Phase2D affine per channel (seeded scale/rotation + PCC on gradient images, cross-validated by NMI) and writes `3-assembly/lc_<ch>_register.yml` plus review PNGs and a QC CSV. |
| `prepare_unified_pheno_tiles` | `register.prepare_unified_pheno_tiles` | Applies the channel-registration affines to the fluor tiles (`register_fluor_2d_tiles`), then builds the unified Phase2D + Fluor2D + VS per-tile store by symlinking source chunks. |
| `estimate_and_stitch_pheno-2d` | `ops_stitch.estimate_and_stitch` | Assembles the unified tiles into the stitched phenotyping mosaic (`phenotyping_v3.zarr` / `pheno_assembled_v3`), forcing the correct level-0 YX scale. |
| `viscy_normalize` | `assemble.viscy_normalize` | Runs ViSCy preprocess (via the bounded-sampling `fast_normalize` helper) over the stitched phenotyping store, writing `normalization` metadata for all channels. |

### Convergence and downstream

| Step | Implementation | What it does |
| --- | --- | --- |
| `build_pyramids` | `pyramids/launcher.build_pyramids` | Waits on all three branches, then submits per-position/per-unit SLURM jobs that build multiscale pyramid levels, contrast limits, grid and ISS overlays in place in the v3 stores. |
| `submit_nuclei_segmentation_jobs` | `cell_seg/nuclei_segmentation_orchestrator.py` | Fans out native-20x nuclei segmentation (`cell_seg/nuclei_pass.py`) writing the `nuclear_seg` label in place. Retires the old 5x `segment_and_stitch_pheno`. |
| `submit_cell_segmentation_jobs` | `cell_seg/cell_segmentation_orchestrator.py` | Fans out per-position whole-cell Cellpose-SAM segmentation from the `membrane_prediction`/`nuclei_prediction` channels, writing the `cell_seg` label in place. |
| `submit_registration_jobs` | `auto_register/auto_register_orchestrator.submit_registration_jobs` | Fans out ISS→track and pheno→track (or →pheno) segmentation registration jobs, then runs check/refine passes. Writes the per-well `*_register.yml` affines, overlays, and `auto_register_metrics.csv`. |
| `submit_tracking_jobs` | `track/track_orchestrator.submit_tracking_jobs` | Gates on registration QC, then submits one `track.track_wells` job per well; each solves cell linkage over timepoints and writes a `tracking_geff` graph plus a completion marker. |
| `link_calls_tracks` | *(elsewhere)* `data/datasets.py` | Joins the ISS barcode calls to the tracked cells. |
| `fix_v3_stores` | `pyramids/audit_fix.fix_v3_stores` | Audits every v3 store for missing pyramids, labels, clims, overlays and wrong YX scale, and re-submits the specific builds needed to fix them. Registered directly by the orchestrator (not present in `step_registry.py`). |
| `recompute_metrics` | *(elsewhere)* `metrics/metrics.py` | Final QC metric pass. |

### End-of-pipeline inference (parallel to `recompute_metrics`)

All four live in `inference.py`, which generates each model's per-experiment YAML config
under `<BASE_PATH>/configs/inference_configs/<model>/v2/` from the experiment's
channels/wells and then hands off to that model package's own SLURM driver. Heavy model
imports are lazy, so importing this module (and the orchestrator) stays CPU/GPU-free.

- `organelle_segmentation` — Organelle Profiler stage 1; writes organelle label arrays into `phenotyping_v3.zarr`.
- `op_feature_extraction` — Organelle Profiler stage 2; GPU → CPU → aggregate as dependent SLURM jobs.
- `cp_features` — CellProfiler features (`ops_model.features.cp_features`); array → concat → AnnData chain.
- `celldino_inference` — CellDINO embedding inference (`ops_model.features.batch_process_embeddings`).

### Registry entries outside the main DAG

`step_registry.py` also exposes `segment_and_stitch_pheno_cells` (`segment.segment_and_stitch`
with `process="pheno_cells"`), `track_wells` (alias of `submit_tracking_jobs`),
`build_iss_overlay` (`pyramids/audit_fix.py`, re-exported from `pyramids/build_drivers.py`),
`build_organelle_pyramids` (`pyramids/launcher.build_organelle_pyramids_only`), and
`submit_organelle_segmentation_jobs` and `extract_features_for_experiment` (both in the
external `organelle_profiler` package — feature extraction does not live in this
directory).

## Top-level modules not covered above

| Path | Purpose |
| --- | --- |
| `run.py` | CLI entrypoint; parses flags, sets `OPS_MODE` / output-dir redirection / SLURM QoS, then calls the orchestrator. |
| `assemble.py` | `viscy_normalize` and a `create_max_projection` wrapper, plus `prepare_beads` (converts and reconstructs the 20x bead store and restacks it with orientation augmentations applied) — used for bead-based channel registration, not wired into the DAG. |
| `assemble_link.py` | The four link/stack steps above, plus a `convert` function (OME-TIFF → Zarr by experiment or by explicit input/output dirs) used by the tests; the DAG's `convert_iss` uses `convert/tiff_to_zarr.py` instead. |
| `reconstruct_subtile.py` | Subtile autofocus reconstruction: splits each tile into N overlapping subtiles, reconstructs each at its own focus, and blends them back together. Library only — called from `reconstruct.py` and `reconstruct_tilt_corrected.py`. |
| `viscy_batch_inference.py` | Standalone script run inside the viscy conda env by a SLURM array task: loads the model once, predicts a position range with async zarr writing. Invoked by `virtual_staining.py` and `nextflow/bin/run_viscy_array_task.sh`. |
| `register.py` | Besides `correct_cycle_drift` and `prepare_unified_pheno_tiles`: `register_fluor_2d_tiles` (apply per-channel affines to the 20x fluor tiles), tile-drift/stabilization model helpers, and drift plotting. |
| `segment.py` | Besides `segment_and_stitch`: standalone `segmentation` (per-FOV Cellpose, no stitching), `preprocess_pheno_cells`, `dilate_masks`, and `upscale_nuclear_segmentations` (5x/pseudo-5x nuclei labels → full 20x resolution, optionally symlinked into the destination store). |

## Sub-packages

### `auto_register/`
Automated cross-cycle and cross-modality registration. It has its own documentation —
see [`AUTO_REGISTER_README.md`](auto_register/AUTO_REGISTER_README.md),
[`REGISTRATION_LOGIC.md`](auto_register/REGISTRATION_LOGIC.md), and
[`iss_auto_register.md`](auto_register/iss_auto_register.md) — so the list below is only a
file index.

| File | Summary |
| --- | --- |
| `auto_register.py` | Core single-job engine: PCC pre-alignment → centroid extraction → KDTree/Hu-moment/graph matching → RANSAC affine → composition → validation overlays and metrics. Reads the segmentation and stitched intensity stores; writes `*_register.yml`, overlay PNGs, `auto_register_metrics.csv`. CLI subcommands `iss-track`, `pheno-track`, `all`, `compare`, `overlap-iss`, `overlap-pheno`. |
| `auto_register_orchestrator.py` | SLURM fan-out for ISS→track / pheno→track / cell-painting→pheno registration, plus the check/refine passes and combined-metrics aggregation. Pipeline entrypoint `submit_registration_jobs`. |
| `auto_register_pcc.py` | Phase-cross-correlation pre-alignment (binary / edges / distance transform / blurred variants, adaptive downsampling) with a JSON cache under `2-tracking/cache/pcc/`. |
| `auto_register_ransac.py` | Pure computation: KDTree nearest-neighbour point matching and RANSAC affine/similarity estimation from matched centroids. No I/O. |
| `auto_register_utils.py` | Shared helpers: segmentation/spot loading and path resolution, centroid/area extraction with spatial subsampling, Hu moments and kNN matching, affine 3x3↔4x4 conversion and YAML I/O, overlap/quality metrics, and the `.npy`/`.pkl` caches. |
| `auto_register_graph.py` | Graph-based match validation: k-NN neighbourhood graphs with rotation-invariant descriptors combined with Hu-moment distance, for cases where PCC pre-alignment is poor. No I/O. |
| `auto_register_visualization.py` | Overlay and QC image generation (PCC before/after, final alignment, detail grids, centroid and sampling overlays, manual-vs-auto comparisons), CuPy-accelerated with a NumPy fallback. |
| `iss_cycle_register.py` | ISS round-to-round registration in stitched well space: per-round-pair spot matching + RANSAC, nucleus→round0 and segmentation→nucleus, cumulative composition, transform application, QA overlays/metrics. Writes `register/transforms/`, `register/overlays/`, and `bc_stitched_registered.zarr`. Has a CLI. |
| `iss_cycle_register_orchestrator.py` | SLURM orchestration for the above: pre-creates the registered zarr, submits the per-well round-pair / nucleus / segmentation jobs (incl. optional `DAPI_round10`), monitors them, then runs the finalize job. Pipeline entrypoint `register_iss_cycles`. |
| `channel_reg/channel_register.py` | Fluorescence→Phase2D per-channel affine registration (seeded scale/rotation, PCC on gradient magnitude, NMI cross-validation). Writes `3-assembly/lc_<ch>_register.yml`, review PNGs, `channel_registration_qc.csv`. Entrypoints `submit_channel_registration_jobs`, `auto_register_channels`, and the Nextflow `channel_registration_setup` / `channel_registration_job`. |
| `channel_reg/__init__.py` | Re-export shim for the four `channel_register` entrypoints. |

### `cell_seg/`
Nuclei and whole-cell segmentation on the native-20x phenotyping store, plus their SLURM
fan-out drivers.

| File | Summary |
| --- | --- |
| `cell_segmentation.py` | Per-position whole-cell segmentation: reads the `membrane_prediction` + `nuclei_prediction` channels of `phenotyping_v3.zarr`, tiles with overlap, runs Cellpose-SAM per tile on GPU via Dask workers, merges tiles by IoU + union-find, reshards, builds the pyramid, and writes `labels/cell_seg`. Entry function `segment_single_position`; CLI supports preview/sweep and `--cell-paint` modes. |
| `cell_segmentation_orchestrator.py` | Discovers positions, skips those that already have the output label, and submits per-position (or chunked) SLURM jobs. Pipeline entrypoint `submit_cell_segmentation_jobs`. |
| `nuclei_pass.py` | Native-20x nuclei sibling of the above: segments the `nuclei_prediction` channel with Cellpose-SAM and writes `labels/nuclear_seg`, reusing the cell-seg tile grid / merge / pyramid helpers. Entry function `segment_nuclei_single_position`. |
| `nuclei_segmentation_orchestrator.py` | SLURM/Nextflow fan-out for `segment_nuclei_single_position`. Pipeline entrypoint `submit_nuclei_segmentation_jobs`, plus `nuclei_segmentation_setup` / `nuclei_segmentation_job` for Nextflow. Note: `nuclei_segmentation_setup` imports from a `cell_segmentation_slurm` module that no longer exists, so that path would raise `ModuleNotFoundError`. |
| `__init__.py` | Lazy (PEP 562) re-export of `segment_single_position`, so importing the package does not pull in CuPy and initialise CUDA in the parent process. |

### `track/`
Cell tracking over timepoints.

| File | Summary |
| --- | --- |
| `track.py` | Per-well tracking: assembles configs from three segmentation sources (v3 `nuclear_seg`, ISS segmentation, 5x segmentation), applies the registration affines, builds a `tracksdata` graph with parallel regionprops, and solves linkage with `hoct`'s ILP solver. Writes the well's `tracking_geff` graph plus a `<well>_tracking_complete.yaml` marker. Main function `track_wells`; has a CLI. |
| `track_orchestrator.py` | Checks registration QC from `auto_register_metrics.csv`, finds wells whose `tracking_geff` output is missing, and submits one `track_wells` job per well. Pipeline entrypoint `submit_tracking_jobs`. |
| `__init__.py` | Empty. |

### `pyramids/`
Multiscale OME-Zarr pyramid construction, overlays, contrast limits, resharding, and store
auditing — all done in place in existing stores.

| File | Summary |
| --- | --- |
| `launcher.py` | SLURM/local submission front-end: discovers stores and positions, enumerates build units, submits job arrays, stamps the canonical level-0 YX scale. Pipeline entrypoints `build_pyramids` and `build_organelle_pyramids_only`, plus `build_pyramids_local` and the Nextflow `build_pyramids_setup` / `build_pyramids_position_job`. Has a CLI. |
| `build_dask.py` | Core in-place pyramid engine: creates and fills multiscale levels for base images, `seg` / `nuclear_seg`, and organelle labels using dask/joblib and `downscale_local_mean`. Entrypoints `build_pyramid_in_place`, `build_seg_pyramid_only`, `build_organelle_seg_pyramids`, `build_base_image_pyramids`. |
| `workers.py` | The SLURM job bodies executed on compute nodes (one build unit, one seg unit, overlays for a position, one reshard level, all-components-for-one-position), shared by `launcher.py` and `audit_fix.py`. |
| `overlays.py` | Renders ISS gene/guide label overlays and grid overlays as RGBA layers tiled into the store, from the linked-results CSV and library map. Entrypoints `build_iss_overlay_in_place`, `build_grid_overlay_in_place`. |
| `clims.py` | Computes per-level contrast limits and writes them into each pyramid level's attributes. Entrypoint `build_clims_in_place`. |
| `reshard.py` | Converts unsharded Zarr v3 arrays (written that way so parallel tile writes are safe) into sharded arrays by tile-by-tile copy. No-op for v2. |
| `build_drivers.py` | One batch build driver per component (clims, ISS overlay, grid overlay, seg pyramids, organelle pyramids, base images, cell painting) that resolves store paths, filters to configured wells, and calls the engine. `build_iss_overlay` is the public one; the rest are the lower layer used by `audit_fix.py`. |
| `store_audit.py` | Read-only audit of v3 stores — metadata readers, YX-scale / normalization / clims checks, seg-label diagnosis — returning copy-paste `fix_commands`. Entrypoint `audit_v3_stores`; also holds the canonical scale constants. |
| `audit_fix.py` | Audit/fix orchestrator: wraps `store_audit` and `build_drivers`, applies metadata fixes, turns audit fix-commands into SLURM job specs, reshards afterwards, and scans experiment status. Pipeline entrypoints `fix_v3_stores` and `build_iss_overlay` (re-exported). Has a CLI. |
| `__init__.py` | Docstring only; exports nothing. |

## Shell scripts

- `make_dirs.sh` — one-off helper that `mkdir`s the numbered experiment directory skeleton
  (`0-convert`, `1-preprocess`, `2-tracking`, `3-assembly`) for the experiment named as its
  first argument, under `$OPS_BASE_PATH`. Superseded by the path handling in `OpsDataset`.
