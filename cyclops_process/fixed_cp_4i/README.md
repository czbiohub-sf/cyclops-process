# fixed_cp_4i

Standalone numbered pipeline for the fixed-cell modalities — cell painting (`cp`,
2 "parts") and 4i / iterative immunofluorescence (`4i`, 5 "rounds") — taking raw
acquisitions through convert → project/flatfield → stitch → segment → register →
warp into the v3 store, and finally linking each fixed cell back to the live
phenotyping and ISS data. Both modalities share one code path; every step script
takes `--modality {cp,4i}` and all differences (unit vocabulary, channel panels,
store/YAML naming, tile size) live in `configs/modality_config.py`.

## Orchestrator

`run_pipeline.py` is the source of truth for step order (`STEP_ORDER`). Each step
is itself a SLURM-submitting CLI that blocks until its jobs finish; the
orchestrator sequences them and halts on the first failure.

```
uv run python -m cyclops_process.fixed_cp_4i.run_pipeline -e ops0174                  # cp (default)
uv run python -m cyclops_process.fixed_cp_4i.run_pipeline -e ops0144 --modality 4i
uv run python -m cyclops_process.fixed_cp_4i.run_pipeline -e ops0174 --from convert_v3
uv run python -m cyclops_process.fixed_cp_4i.run_pipeline -e ops0174 --steps stitch segment_5x --dry-run
```

Per-experiment orientation (`flipud`/`fliplr`/`rot90`) and stitch settings
(registration channel, tile overlap, CLAHE) are recorded in the `PIPELINE_PARAMS`
dict at the top of `run_pipeline.py`, keyed by the `opsNNNN` token parsed out of
the experiment name. Experiments not listed fall back to `cp`, no orientation,
channel 0 / overlap 100 / CLAHE on.

## Stages, in run order

| # | Step (`run_pipeline` name) | Module | What it does |
| --- | --- | --- | --- |
| 00 | `convert` | `00_convert.py` | Discovers the per-unit raw acquisition dirs (CP: `round{N}_1` leaf dirs under the dragonfly "cellpainting" root, junk rejected, max-tiff leaf per round; 4i: hardcoded instrument dirs from `four_i_config`) and submits one resumable SLURM convert job per unit, writing `0-convert/<subdir>/<stem>{N}.zarr`. |
| 01 | `project` | `01_project.py` | Per-position max projection over Z, `(T,C,Z,Y,X)` → `(T,C,1,Y,X)`, and — in the same SLURM job — the flatfield correction. Produces `<stem>_max_proj.zarr` then `<stem>_max_proj_flatfield.zarr`. This is also the single place orientation is applied (per-tile `flipud`/`fliplr`/`rot90`), so everything downstream runs on already-oriented data. |
| 02 | *(not used by the orchestrator)* | `02_flatfield_correction.py` | Standalone per-unit flatfield correction of the max-projected store, one SLURM job per unit. Kept as a separate entrypoint; `run_pipeline` gets flatfield from step 01 instead. |
| 04 | `stitch` | `04_stitch.py` | Estimates tile shifts from overlaps and assembles a per-well mosaic on GPU, writing `<stem>_stitch.yaml` (the shifts step 05 reuses), `<stem>_stitched.zarr`, and a stitch-confidence plot; warns when a well's median edge confidence falls below `--min-confidence` (default 0.8). |
| 05 | `segment_5x` | `05_segment.py` | Cellpose nuclei segmentation of the flatfield max-projection (4x downsample, diameter 30) stitched with step 04's shift YAML, producing `<stem>_max_proj_flatfield_segmentation.zarr` — the registration input. A second subcommand, `upscale`, lifts 5x segmentations to 20x. |
| 06 | `register` | `06_register.py` | Auto-registers those 5x nuclear segs and writes chained register YAMLs. CP: part1 → pheno, partN → part(N−1), into `2-tracking/`. 4i: rounds 2-5 → round 1 (same-modality DAPI, PCC only) plus round 1 → pheno (cross-modality, with RANSAC), into `0-convert/4i/registration/`. Also has `--check-all` / `--refine-all` modes that score and refine existing YAMLs (IoU metrics + overlays). |
| — | `convert_v3` | `cyclops_process.convert.v3_fixed_cli` | Outside this directory: extends the pheno v3 store with the fixed-cell channels, warped in via the step-06 registration affines. |
| — | `segment_20x` | `cyclops_process.processes.cell_seg.{nuclei_segmentation_slurm,cell_segmentation_slurm}` | Outside this directory: 20x segmentation on the v3 store — one per-unit nuclear-seg job (`CP{N}`/`4i_R{N}_nuclear_seg`) plus one cell-seg job (`--cell-paint` or `--4i`), all launched concurrently. |
| — | `link` | `link_slurm.py` → `link.py` | Links fixed cells to pheno and ISS by nearest-neighbour matching of nuclear-seg centroids at pyramid level 2, merges in barcodes/genes, and writes `<well>_linked_pheno_iss_{cp,4i}.csv` plus an overlay preview PNG per well. |

Direct invocations for the in-directory steps:

```
uv run python -m cyclops_process.fixed_cp_4i.00_convert -e ops0174 --modality cp --parts 1 2
uv run python -m cyclops_process.fixed_cp_4i.01_project -e ops0174 --parts 1 2 --rot90 1
uv run python -m cyclops_process.fixed_cp_4i.02_flatfield_correction --experiment ops0094 --parts 1 2
uv run python -m cyclops_process.fixed_cp_4i.04_stitch --experiment ops0094 --parts 1 2 --channel 1 --overlap 75 --use-clahe
uv run python -m cyclops_process.fixed_cp_4i.05_segment segment -e ops0174 --parts 1 2
uv run python -m cyclops_process.fixed_cp_4i.06_register -e ops0094 --force
uv run python -m cyclops_process.fixed_cp_4i.06_register --modality 4i -e ops0144 --type cross-round
uv run python -m cyclops_process.fixed_cp_4i.link_slurm --experiment ops0094
uv run python -m cyclops_process.fixed_cp_4i.link --experiment ops0094 --wells A/1/0   # single-process
```

There is no `03_` module in this directory, and none appears in the repository
history for this path; no code references a step 03.

## `configs/`

| File | Purpose |
| --- | --- |
| `modality_config.py` | The `Modality` dataclass plus the `cp` / `4i` instances and `get_modality()`. Single source of truth for unit vocabulary (`part`/`round`), channel-name prefixes, default unit lists, cell-seg labels, convert subdirectory, register-YAML directory and name, full-res tile size (2048 for CP, 2304 for 4i), and the CP panel channel definitions (`CELL_PAINTING_CHANNELS`). Also provides the path helpers (`convert_dir`, `seg_store_path`, `register_dir`) used by every step. |
| `four_i_config.py` | Experiment-specific 4i config for the `20260318_4i_re-run` acquisition (`EXPERIMENT = "ops0144_20260406"`): instrument root, per-round acquisition dir + "final" subfolder, antibody pairs, and the `swap_488_647` flag that corrects rounds acquired in reverse channel order. Derives `FOUR_I_CHANNELS`, `FOUR_I_COLORS`, `NUM_ROUNDS`, and round-dir/channel-name helpers. |
| `__init__.py` | Empty package marker. |

## `helpers/`

| File | Purpose |
| --- | --- |
| `sweep_stitch_params.py` | Sweeps overlap × flip × rotation × registration channel with the fast estimate-only stitch on a small adjacent tile grid, then prints a table ranked by the worst non-empty well's median edge confidence. This is how you discover a new experiment's orientation/channel/overlap before recording them in `PIPELINE_PARAMS`. Runs combos locally in parallel, or one SLURM job each with `--slurm`. |
| `04b_cell_seg_sweep_4i.py` | Submits a 4i cell-segmentation channel sweep (`FOUR_I_SWEEP_CONFIGS`) on a single position of the unregistered 4i v3 store, writing each combination to its own `4i_sweep_<name>` label group for visual comparison in napari. |
| `extract_position_map.py` | Reads the MicroManager metadata header of each 4i round's first NDTiff file and writes `round{N}_position_map.json`, mapping zarr position index → original position label (e.g. `A1-Site_040052`). Metadata only, so it takes seconds rather than a reconvert. |
| `reorganize_zarr.py` | One-off fix for 4i convert output: rewrites flat integer positions (`0/<idx>/0`) into the HCS layout (`A/<well>/<XXXYYY>`) using those position-map JSONs, updating the plate `.zattrs`. Also has `--fix-nesting` (removes an extra array level) and `--validate` (checks positions/arrays/metadata against the expected well layout). |
| `__init__.py` | Empty package marker. |

## Usage notes

- **Orientation is set once.** `01_project` bakes `flipud`/`fliplr`/`rot90` into
  the projection; `04_stitch` and `05_segment` must then run with identity
  orientation. If you do pass orientation flags to 04 and 05 directly, they must
  match each other — step 05 reuses step 04's shift YAML, so a mismatch
  misplaces tiles.
- **Stitch before segment.** `05_segment` needs the `<stem>_stitch.yaml` written
  by `04_stitch`; `06_register` needs the segmentation stores written by
  `05_segment`.
- **Store naming chain** (per unit, under `0-convert/<cell_painting|4i>/`):
  `part1.zarr` → `part1_max_proj.zarr` → `part1_max_proj_flatfield.zarr` →
  `..._stitched.zarr` / `..._max_proj_flatfield_segmentation.zarr`. Steps that
  consume a projection prefer the `_flatfield` variant and fall back to plain
  `_max_proj`.
- **Resources.** Convert and flatfield are CPU-heavy (64 CPUs, 200-250 GB);
  stitch assembly and segmentation require GPU nodes (stitch assembly is
  CuPy-accelerated — on CPU it falls back to a far slower blend path). All
  SLURM parameters are the `SLURM_PARAMS` dicts at the top of each step module.
- **Reruns.** Steps skip units whose final output already exists; use `--force`
  (`--force-convert` via the orchestrator) to redo, and `--resume` on
  `00_convert` to finish a timed-out conversion by filling only missing
  positions. Most steps also accept `--dry-run` and `--local`.
- **Linking prerequisites.** `link_slurm.py` checks for the pheno v3 store with
  the primary nuclear-seg label and the ISS segmentation store, and skips wells
  missing them; wells default to `A/1/0 A/2/0 A/3/0`. Cached centroid parquets in
  `results_fast/<cp_links|four_i_links>/` make reruns much faster.
- Several module docstrings still show pre-move import paths (e.g.
  `cyclops_process.data.link_cell_painting`, `cyclops_process.fixed_cp_4i.sweep_stitch_params`).
  Use the actual module paths shown above.
