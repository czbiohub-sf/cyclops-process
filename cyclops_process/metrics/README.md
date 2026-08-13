# metrics

Quantitative QC for OPS experiments. The top level holds generic metric implementations
(reconstruction, segmentation, tracking, SNR) plus the `get_metrics` driver;
`plate_stats/` holds the plate-wide in-situ sequencing (ISS) QC suite that produces
`plate_stats.csv` and its accompanying plots.

Most outputs land in the experiment's `results_iss` directory (`3-assembly/ISS/<method>/`),
resolved through `OpsDataset.metrics_paths` / `result_paths`.

## Top-level modules

| Path | Purpose |
| --- | --- |
| `metrics.py` | **Entrypoint.** `get_metrics` / `recompute_metrics` — the driver that runs the whole plate-stats suite for one experiment (frequency tables, read accuracy, confluency, cell size, heatmaps, Hamming distance, base fractions, SNR/crosstalk, growth effect, `statistics`, the histogram plots, then metrics-over-time) and prints the resulting stats table. `recompute_metrics` is the same thing with `force=True`, run again at the end of the pipeline. |
| `metrics_reconstruction.py` | **Entrypoint.** Turns the per-subtile reconstruction metadata CSV (`focus_index`, `z_focus_offset`) into mean/median/std subtile heatmaps and per-well spatial heatmaps (PNG) for a given process tag (`track-2d`, `pheno-2d`, or `both`). |
| `metrics_segmentation.py` | **Entrypoint.** Standalone Cellpose-vs-ground-truth segmentation evaluation: IoU-based recall/precision on hand-annotated tiles, with flow-threshold and model sweeps and side-by-side comparison canvases (PNGs). Not a pipeline step — it uses hard-coded annotation/position tables and a local minimal `OpsDataset` stub for path resolution. |
| `metrics_snr.py` | **Entrypoint.** Spot-based ISS SNR heatmaps: samples spots per tile via `plate_stats/metrics_iss_utils.process_well_snr_tiles` and emits overall / per-channel / per-channel-per-round SNR heatmaps, metric-vs-cycle plots, a crosstalk heatmap, and per-tile + summary CSVs. |
| `metrics_tracking.py` | **Entrypoint.** Cross-experiment tracking QC from `linked_pheno_iss.csv`: cells over time, cells per well, cells per gene, growth-effect correlation, ISS→tracking cell loss, plus per-experiment gene histograms and well/normalized-tracking/tracking-loss heatmaps. Writes `tracking_metrics_summary.csv` and many PNGs. |
| `top_gene_over_time.py` | **Entrypoint.** Finds the top gene by cell count per experiment (optionally for both base-calling methods) and plots that count across experiments over time (PNG). |

Invocation:

```bash
uv run python -m cyclops_process.metrics.metrics 94 --method mine [--force] [--function <name>]
uv run python -m cyclops_process.metrics.metrics_reconstruction --experiment ops0063_20250731 --tag track-2d
uv run python -m cyclops_process.metrics.metrics_segmentation [--sweep-flow | --sweep-models | --compare-tile ...]
uv run python -m cyclops_process.metrics.metrics_snr --experiment ops0092_20251027 --grid-size 30
uv run python -m cyclops_process.metrics.metrics_tracking ops0113_20251219 --verbose --force
uv run python -m cyclops_process.metrics.top_gene_over_time ops0113_20251219 [--compare]
```

`get_metrics` is also a registered pipeline step, so it can be run for every experiment with
`uv run python -m cyclops_process.processes.run --all --rerun get_metrics`.

(The `--compare-tile` examples inside `metrics_segmentation.py` refer to a
`cyclops_process.metrics.graphs.*` path that no longer exists; use the module path above.)

## `plate_stats/`

The ISS QC suite. Almost all of these are imported helpers called by `metrics.get_metrics`;
the handful with a CLI are marked **Entrypoint**.

### Aggregation and shared primitives

| Path | Purpose |
| --- | --- |
| `iss_stats.py` | `statistics()` — the aggregator. Walks every well, counts cells / reads / matched reads / cells-with-reads at each pipeline stage, pulls in the per-stage helper stats (registration, stitching, reconstruction, tracking, linking, spatial coherence, cell size, flatfield, entropy) and writes the transposed table to `plate_stats.csv`. |
| `match_reads.py` | Matches ISS read barcodes to codebook sgRNAs, and owns the failed-round bookkeeping (`dropout`, `shift`, and leading read-frame `offset` modes) used consistently by every other module. Returns dataframes; writes nothing. |
| `metrics_iss_utils.py` | Shared signal-quality primitives: spot detection on the fly, background masking, signal/background statistics, crosstalk-matrix estimation, and the SNR / SBR / LLD / Z′ calculations, plus per-tile and per-well drivers. Pure library. |
| `__init__.py` | Package marker only. |

### Read and base-calling quality

| Path | Purpose |
| --- | --- |
| `iss_metrics.py` | Base-composition (`count_bases`, should be ~25% per base), doublet counting, per-well and per-experiment frequency tables (`frequency_table.csv`), read accuracy vs. round, and the ISS rounds manifest (`iss_rounds_manifest.yaml`). Produces CSVs, a YAML manifest, and PNGs. |
| `iss_heatmaps.py` | Per-tile spatial heatmaps of read accuracy (or mean confidence for the probabilistic method) and of percent-cells-with-reads / cells-with-reads-per-cell. Returns the per-tile maps and saves PNGs. |
| `hamming_distance.py` | Distribution of the minimum Hamming distance from each read to the codebook for one well; saves a PNG. Only run for the `mine` base-calling method. |
| `iss_entropy.py` | Shannon-entropy statistics over matched barcodes (entropy-vs-count regression slope/R²/p, top-guide ratio) and Pearson correlation of the guide-frequency distribution against a cached reference. Returns dataclasses consumed by `iss_stats.py`. |
| `iss_histrogram.py` | **Entrypoint.** Cells-per-gene histogram, top-N genes and top-N guides by cell count (with NTC variants), and guide-entropy-vs-cell-count plots; also writes `frequency_table_filtered.csv` (gene-matched reads only). The CLI regenerates just that filtered table for one/many experiments, optionally fanning out over SLURM. |

### Image and signal quality

| Path | Purpose |
| --- | --- |
| `metrics_probs_ISS.py` | **Entrypoint.** Spot-based signal/background/crosstalk characterization per well: signal, background mean/noise, SNR, SBR, LLD and Z′ vs. cycle, plus an estimated crosstalk heatmap and matrix CSV. `main_for_metrics` is the hook `get_metrics` calls; the click CLI runs it standalone and writes a well-stats summary CSV. |
| `iss_snr_bimodal.py` | **Entrypoint.** Spot-free alternative to `metrics_snr.py`: fits a two-component Gaussian mixture to tile intensities to separate signal from background, then emits the same overall / per-channel / per-channel-per-round heatmaps, metric-vs-cycle plots, and per-tile + summary CSVs. Reuses cached CSVs unless `--force-recompute`. |
| `iss_flatfield_metrics.py` | Aggregates the per-tile flatfield QC CSV written by `correct_flatfield` to per-well means (CV and SNR before/after, profile range) for `plate_stats.csv`. Returns a dict. |

### Cell counts, morphology, spatial structure

| Path | Purpose |
| --- | --- |
| `iss_confluency.py` | Cell density (confluency) per tile across each well, counted in parallel from the ISS segmentation store; returns the per-well maps and saves a heatmap PNG. |
| `iss_cell_size.py` | **Entrypoint.** Per-cell bounding-box height/width/area/aspect ratio derived from the linked CSV (no segmentation zarr needed), plus a large-cell analysis. Writes a per-cell CSV per well, a summary CSV, and distribution plots. |
| `iss_spatial_coherence.py` | KDTree-based sister-cell (same-sgRNA) proximity metrics on pheno coordinates: `sister_count`, `sister_ratio`, `region_homogeneity` and a `miscall_score` flagging the spatial signature of barcode miscalls. Returns per-well summaries and caches `per_cell_spatial_coherence.csv`. |

### Upstream pipeline stages folded into plate_stats

| Path | Purpose |
| --- | --- |
| `plate_stats_reconstruction_metrics.py` | Per-well reconstruction QC — tilt parameters (`z_offset`, zenith, azimuth) from the tilt-corrected backend, falling back to `z_focus_offset` from the legacy subtile metadata CSV. Returns dataclasses/dicts. |
| `plate_stats_stitch_metrics.py` | Per-well stitch confidence statistics (mean/median/std/min/max, edge count) read from the stitch config, plus per-well confidence heatmap PNGs and a cached stats CSV. |
| `iss_register_metrics.py` | Registration QC from three sources: per-round ISS cycle registration metrics (inlier ratio, residuals), per-round and cumulative drift from the affine transforms, and auto-register (ISS→track, pheno→track) overlap/inlier metrics. Returns dataclasses/dicts. |
| `plate_stats_track_metrics.py` | Tracking and linking QC from the GEFF graphs and linked results: graph statistics, movement / per-track distance, doublets, daughter-barcode consistency, tracking loss, post-tracking cells per gene, and the link-pipeline progression from `link_metrics.csv`. Also produces the link-level cells-per-gene / top-genes / top-guides PNGs. |

### Screen-level and cross-experiment views

| Path | Purpose |
| --- | --- |
| `iss_growth_effect.py` | Correlates per-gene OPS cell counts against DepMap/CERES growth-effect scores, pooled and per well, with linear regression annotations. Saves two PNGs and caches the regression stats to `growth_effect_stats.csv`. |
| `iss_timing.py` | Parses the per-experiment timing log (YAML) and plots wall-clock minutes per pipeline step (PNG). |
| `metrics_over_time.py` | **Entrypoint.** Scans `plate_stats.csv` across all experiments and plots the evolution of every metric over time (one plot per metric, per-well points with a mean line), filed into category subdirectories; parallelised with joblib. |

### Failed-round optimization

| Path | Purpose |
| --- | --- |
| `optimize_failed_rounds.py` | **Entrypoint.** Searches dropout/shift combinations per well to find the configuration that maximises codebook match rate without inflating false positives, validating either by entropy statistics (default) or by correlation to a cached cross-experiment reference guide-frequency distribution (`--ops-median`). Writes per-well summary/detail CSVs and an optimized YAML config. |
| `optimize_failed_rounds_orchestrator.py` | **Entrypoint.** SLURM fan-out over experiment × well for the above, plus `--aggregate` to collect results, print a summary, write a failed-rounds report, and update the experiment / `ops_failed_rounds` configs. |

Invocation:

```bash
uv run python -m cyclops_process.metrics.plate_stats.iss_cell_size 94 [--force]
uv run python -m cyclops_process.metrics.plate_stats.iss_histrogram 94 [--all] [--slurm]
uv run python -m cyclops_process.metrics.plate_stats.iss_snr_bimodal --experiment ops0033_20250429
uv run python -m cyclops_process.metrics.plate_stats.metrics_probs_ISS --experiment ops0033_20250429 --wells all
uv run python -m cyclops_process.metrics.plate_stats.metrics_over_time ops0113_20251219
uv run python -m cyclops_process.metrics.plate_stats.optimize_failed_rounds ops0108_20251209 --well "A/3/0"
uv run python -m cyclops_process.metrics.plate_stats.optimize_failed_rounds_orchestrator --all
```

## Regression checking `plate_stats.csv`

`plate_stats.csv` is the canonical QC artifact of a run and is regression-tested against a
reference plate. See **Plate-stats regression check** in the
[ops_process README](../../README.md): `tests/QC/compare_plate_stats.py` compares a candidate
CSV against the ops0094 reference per (metric, well), groups failures by pipeline module, and
exits non-zero when any metric drifts by more than the tolerance (0.5% by default).
