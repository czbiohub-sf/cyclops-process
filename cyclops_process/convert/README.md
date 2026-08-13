# convert

Conversion layer: raw acquisitions → OME-Zarr, and OME-Zarr v2 → v3 (including
warping fixed-cell channels into the live-cell phenotyping frame).

Two concerns live here and they are independent:

1. **Raw convert** — turn what the microscope wrote into an OME-Zarr store. One
   entrypoint per acquisition source (`tiff_to_zarr.py` for live-cell OME-TIFF,
   `raw_to_zarr.py` for Dragonfly raw TIFFs).
2. **v3 convert** — migrate/extend stores into Zarr v3. A shared engine
   (`v3_common.py`, `v3_metadata.py`, `v3_warp.py`) with one driver per modality
   (`v3_livecell.py`; `v3_fixed.py` + `v3_fixed_cli.py`), plus a metadata-only
   repair tool (`update_v3_metadata.py`).

## Files

| File | Purpose |
| --- | --- |
| `__init__.py` | Package docstring only; no re-exports. |
| `tiff_to_zarr.py` | OME-TIFF → OME-Zarr via iohub `TIFFConverter`, parallelized over wells with joblib. Library only — public API is the single `convert()`, used as the pipeline's `convert_iss` step and by `processes/assemble.py`. |
| `raw_to_zarr.py` | Dragonfly raw TIFFs → zarr on the fast partition, one SLURM job per store. HPC port of the Windows `convert.ps1`, adding experiment-name resolution, already-converted skipping, tiff-count/size prechecks, per-job progress, and separate tracking/pheno resource profiles. CLI + `convert_raw()`. |
| `v3_common.py` | Shared v2→v3 engine: store init, channel-based sharding, array copy, per-position group copy, seg-label writing, validation. Also owns the authoritative `OPS_NATIVE_YX_UM_PER_PX = 0.325` and auto-corrects the historical `0.65` over-declaration on the way through. |
| `v3_metadata.py` | Builds the metadata v3 stores carry: `channels_metadata` at plate level in `zarr.json`, and `segmentation_metadata` on each `labels/<name>` subgroup. Format matches `organelle_seg` for cross-store consistency. Also holds channel-type detection and `OVERLAY_METADATA`. |
| `v3_warp.py` | The single chunked-affine warp loop for fixed-cell conversion: per output chunk, map corners back through the affine, build a chunk-local affine, `scipy.ndimage.affine_transform`, write to tensorstore. Plus small affine helpers (load from YAML, compose, scale translation). Replaced three near-identical copies (cp images, 4i images, labels). |
| `v3_livecell.py` | Live-cell v3 driver and CLI. `--mode pheno\|track\|iss\|all` selects the store; supports `--all` batch discovery, `--local`, per-position/per-label reconversion, validation-only, resubmit-from-file, and dry runs. Exposes `convert_iss_to_v3()` (the `convert_iss_to_v3` pipeline step) and `cleanup_top_level_seg_symlinks()`. |
| `v3_fixed.py` | Fixed-cell v3 implementation for **both** cell painting and 4i, warping fixed channels/labels into the pheno 20× frame. Merger of the former `convert_v3_slurm_cp.py` + `convert_v3_slurm_4i.py`. Keeps a staged `main()` (4i-oriented: `--copy-only`, `--transforms-only`, `--labels-only`, `--pyramids-only`, `--seg-pyramids-only`, `--reshard-base`, `--preview`, `--full`) because the 4i orchestrator drives it as a subprocess. |
| `v3_fixed_cli.py` | The one CLI for the fixed-cell v3 step. Dispatches on `--modality cp\|4i` straight into `v3_fixed` backends — no second CLI layer. |
| `update_v3_metadata.py` | CLI to refresh metadata on existing v3 stores without reconverting. Merges rather than overwrites: updates convert-owned fields, preserves downstream `segmentation_metadata` written by other processes. Has `--audit` and `--dry-run` modes. |

## Entrypoints

```bash
# raw convert (fixed-cell / Dragonfly)
uv run python -m cyclops_process.convert.raw_to_zarr 146
uv run python -m cyclops_process.convert.raw_to_zarr OPS0146 --dry-run
uv run python -m cyclops_process.convert.raw_to_zarr 146 --only pheno

# live-cell v3
uv run python -m cyclops_process.convert.v3_livecell --experiment ops0033_20250429 --mode pheno
uv run python -m cyclops_process.convert.v3_livecell --all --mode pheno          # batch
uv run python -m cyclops_process.convert.v3_livecell -e ops0033_20250429 --local --mode track
uv run python -m cyclops_process.convert.v3_livecell -e ops0033_20250429 --validate-only --mode iss

# fixed-cell v3 (cell painting / 4i)
uv run python -m cyclops_process.convert.v3_fixed_cli -e ops0174 --modality cp --parts 1 2
uv run python -m cyclops_process.convert.v3_fixed_cli -e ops0174 --modality cp --mode cp_seg_only --parts 1 2
uv run python -m cyclops_process.convert.v3_fixed_cli -e ops0144 --modality 4i

# metadata-only refresh on existing v3 stores
uv run python -m cyclops_process.convert.update_v3_metadata -e 33 --mode pheno
uv run python -m cyclops_process.convert.update_v3_metadata -e 33 --audit
uv run python -m cyclops_process.convert.update_v3_metadata --all --mode pheno
```

`tiff_to_zarr.py`, `v3_common.py`, `v3_metadata.py` and `v3_warp.py` have no CLI —
they are libraries, called by the steps and drivers above. `v3_fixed.py` has a
`__main__`, but prefer `v3_fixed_cli.py`; the staged flags are for the 4i
orchestrator and for resuming a failed stage.

## How it's wired in

**Pipeline steps** (via `pipelinerunner/step_registry.py`):

| Step | Module | Function |
| --- | --- | --- |
| `convert_iss` | `tiff_to_zarr` | `convert` (`process="iss"`) |
| `convert_raw` | `raw_to_zarr` | `convert_raw` |
| `convert_iss_to_v3` | `v3_livecell` | `convert_iss_to_v3` |

**Imported as libraries** by code outside this package:

- `v3_common.write_seg_label_v3`, `.calculate_channel_based_shards` →
  `processes/ops_stitch.py`, `processes/pyramids/workers.py`
- `v3_metadata.build_channels_metadata`, `.build_label_metadata`, `.OVERLAY_METADATA` →
  `processes/ops_stitch.py`, `processes/cell_seg/cell_segmentation.py`,
  `processes/pyramids/overlays.py`
- `v3_livecell.cleanup_top_level_seg_symlinks`, `.BASE_SLURM_PARAMS` →
  `processes/pyramids/build_drivers.py`
- `tiff_to_zarr.convert` → `processes/assemble.py`, `pipelinerunner/orchestrator.py`
- `raw_to_zarr._convert_single` → `fixed_cp_4i/00_convert.py`

**Driven as a subprocess:** `fixed_cp_4i/run_pipeline.py` invokes
`python -m cyclops_process.convert.v3_fixed_cli` for its `convert_v3` stage.

`v3_warp.py` has no importers outside this package — it is `v3_fixed.py`'s
extracted helper. `update_v3_metadata.py` is hand-run only; nothing imports it
and it is not a pipeline step.

## Notes

- **Pixel size.** `v3_common.OPS_NATIVE_YX_UM_PER_PX = 0.325` is authoritative.
  Older v2 assembly wrote `0.65` (a 2× over-declaration that propagated into
  downstream feature values); v3 conversion detects and corrects it. Do not
  "fix" this back.
- **ISS is the only true v2→v3 conversion left.** Pheno and track now stitch
  v3-native, so with `source_zarr_version=3` their `v3_livecell` modes have
  source == dest and only lift symlinked seg data from top-level groups into the
  `labels/` group. `register_iss_cycles` still writes v2, so `--mode iss` does a
  real conversion (and then async-deletes the v2 source).
- **Two raw converters is deliberate**, not duplication: `tiff_to_zarr.py` is
  the iohub `TIFFConverter` path for live-cell OME-TIFF acquisitions;
  `raw_to_zarr.py` is the fixed-cell Dragonfly path with its own SLURM fan-out.
- **`tiff_to_zarr.convert()` only implements some of its documented processes.**
  `iss` and `20x_beads` work; `lc_20x` raises `NotImplementedError`, and
  `lc_5x` matches no branch (so it would fail rather than convert).
- **Ordering:** raw convert runs before the link/stitch steps that consume
  `raw_convert/`; fixed-cell v3 convert requires an existing pheno v3 store plus
  the registration affines, since it warps into that frame. In `v3_fixed`'s
  full 4i pipeline, stage 1 (init + copy pheno) must complete before the
  transform and label tracks run in parallel.
- **`v3_fixed.py` is large (~3.2k lines) by design** — it holds cp and 4i side
  by side so the shared topology stays visible. Only the warp loop was
  extracted (`v3_warp.py`); the affine *composition* differs per modality (cp
  composes then scales, 4i scales then composes) and stays in `v3_fixed.py`.
- **`update_v3_metadata.py` merges metadata deliberately.** It must not clobber
  `segmentation_metadata` owned by `organelle_seg` and other downstream writers
  — see `merge_label_metadata` / `has_complete_pipeline_metadata`. It lives here
  to sit next to the metadata builders
  it shares with `v3_metadata.py`, though it is a hand-run tool rather than part
  of the conversion path.
