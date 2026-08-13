# napari

Interactive viewers for inspecting OPS pipeline results: Dask-backed multiscale
viewers for stitched OME-Zarr stores (with grid / ISS / segmentation / track
overlays), gene- and attention-driven tiled cell galleries, and a browser-based
embedding atlas.

## Files

| Path | Purpose |
| --- | --- |
| `view_by_gene.py` | User-facing CLI (click). Pulls cells for one gene out of the assembled phenotyping store (`pheno_assembled_v3`) via `BaseDataset` bboxes and lays them out as a tiled napari grid, one row per guide by default, with an inverse-mask layer and hidden coordinate annotations. `--4i` swaps to the `4i_cell_seg` mask / `4i_bbox` column; `--attention` routes through to `view_by_gene_attention.py`. |
| `view_by_gene_attention.py` | Attention-atlas implementation, imported on demand by `view_by_gene.py --attention` (not run directly). Renders the PMA attention atlas in napari: top-N cells per marker for a gene (or CHAD complex) across experiments, one row per marker with NTC strips interleaved, all channels of each cell loadable as toggleable overlays. Reads the `pma_top_phase_cells_*` / `pma_top_fluorescent_cells_*` CSVs (parquet-cached via `df_cache`) and reuses the caption parser and mAP-based marker selection from `ops_model/.../attention/atlas/attention_atlas_shap.py`, which it puts on `sys.path` by relative path. |
| `nd_embed.py` | User-facing CLI wrapping [nd-embedding-atlas](https://github.com/czbiohub-sf/nd-embedding-atlas) — not napari. `prepare` merges UMAP coordinates into the experiment's `*_cell_features.h5ad` and writes a zarr cache; `serve` starts the browser viewer (default `localhost:5055`) with the phenotyping v3 store attached so points link back to cell crops; `launch` does both. |
| `chad_complex_descriptions.yml` | Static data, no code in this directory reads it: one-line functional descriptions of the CHAD protein complexes (`subu …` / `GOI …` keys matching the CHAD v5 hierarchy names), used as the italic subtitle on complex atlas pages. Its loader (`_load_chad_descriptions` in `attention_atlas_shap.py`) reads the copy sitting next to itself in `ops_model`, so this file is effectively a duplicate kept beside the viewer. |
| `dask/view_dask.py` | The main viewer and the biggest module here. Builds and/or views in-place multiscale pyramids for an experiment's stitched stores, resolving them through `OpsDataset` store keys per `--mode` (`pheno`, `track`, `iss`, `all`, `cell_paint`, `bf`). Handles well-separation offsets, per-level contrast limits, registration affines (`--mode all` aligns track/ISS onto pheno using the registration YAMLs), grid overlays, ISS gene/guide overlays, organelle/cell label layers, and geff tracks. Also exposes `view_inplace_pyramid_in_napari`, `view_registered_stores_in_one_viewer`, and `build_and_view` as a Python API. |
| `dask/view_store.py` | Small user-facing CLI that opens *any* `.zarr` store by path in napari via `view_inplace_pyramid_in_napari`, bypassing the experiment + store-key lookup — use it for custom store names that aren't registered in `OpsDataset`. |
| `dask/dask_utils.py` | Helpers for `view_dask.py`: listing numeric pyramid levels, pruning extra levels, resolving the stitch-config YAML for a mode, synthesizing grid-line masks, applying per-level contrast limits on zoom/level change, and small pyramid build/downsample utilities. |
| `dask/channel_clims.py` | Contrast-limit library. Matches each channel name to a `ChannelClimProfile` (fixed ranges for phase / membrane / nuclei predictions; percentile-based profiles for DAPI, MiSeq ISS bases, Cell Painting, fluorescent proteins), samples pixels from multiple windows, and scales limits per pyramid level. Also consumed outside this directory by `cyclops_process/processes/pyramids/clims.py`, which writes the results into the store. |
| `dask/geff_utils.py` | Loads cell tracks from a geff Zarr store with `tracksdata`, converts them to napari tracks format (with the lineage graph), and adds a `tracks:<pos>` layer without an affine so `view_dask.py` can apply the same transform pipeline as the image layers. |
| `dask/grid_exp.py` | Orphaned helper: draws a synthetic grid of well-image "stamps" and experiment/channel labels around the real data. Nothing imports it, and the `--show-exp-grids` path in `view_dask.py` is now an explicit no-op. |

## Entrypoints

`view_dask.py`, `view_store.py`, `view_by_gene.py`, and `nd_embed.py` are the
entrypoints; everything else (`view_by_gene_attention.py`, `dask/dask_utils.py`,
`dask/channel_clims.py`, `dask/geff_utils.py`, `dask/grid_exp.py`) is a helper
module.

```bash
# Build the in-place pyramid + contrast limits (do this once per store)
uv run python -m cyclops_process.napari.dask.view_dask -e ops0033_20250429 -m pheno -a build --num-levels 5 --with-grid

# View it (pheno / track / iss / all / bf / cell_paint), optionally one well
uv run python -m cyclops_process.napari.dask.view_dask -e ops0033_20250429 -m pheno -a view
uv run python -m cyclops_process.napari.dask.view_dask -e 33 -m all -a view --wells A/1

# Other build actions: build-grid, build-iss, build-clim, build-seg, verify-grid

# Open an arbitrary store by path
uv run python -m cyclops_process.napari.dask.view_store \
    /path/to/ops_data/ops0144_20260406/3-assembly/phenotyping_v3_4i_rerun.zarr --wells A/1

# Tiled gallery of cells for one gene
uv run python -m cyclops_process.napari.view_by_gene 12 RBPJ --crop_size 256 --well A1

# Attention atlas (see note below — must be run as a script, not with -m)
uv run python cyclops_process/napari/view_by_gene.py _ ABCE1 --attention --top 10

# Embedding atlas in the browser
uv run python -m cyclops_process.napari.nd_embed launch -e ops0094_20251217
uv run python -m cyclops_process.napari.nd_embed launch -e 94 --preview 10000   # subsampled
```

Experiment names accept the usual shorthand (`33`, `ops33` → `ops0033_20250429`),
and wells accept `A/1` or `A1`.

## Notes

- **Display.** These are GUI tools. On the HPC, launch a noVNC remote desktop and
  run them from a terminal inside it (or otherwise provide an X11/OpenGL-capable
  display); `nd_embed` instead serves HTTP on `localhost:5055` and opens a browser.
- **Extras.** Track overlays need the package `viz` extra (`tracksdata`, `geff`);
  `nd_embed` needs `nd-embedding-atlas` installed separately
  (`pip install git+https://github.com/czbiohub-sf/nd-embedding-atlas.git@main`);
  `view_dask.py` also imports `stitch.registration.register` (biahub) for
  registration transforms.
- **Build before viewing.** `view_dask.py --action view` does not build anything.
  It expects each position to already contain numeric level directories
  (`0`, `1`, …), per-level contrast limits in the array attrs, and — where
  applicable — sibling overlay arrays (`grid_overlay` / legacy `grid_edges` +
  `grid_ids` / `grid_props`, `iss_gene_image` / `iss_guide_image` or legacy
  `iss_points` / `iss_points_props`, and `labels/*_seg`). If the pyramid is
  missing it exits with "Pyramid not found. Run with `--action build` or `both`
  first." Both v2-style (arrays beside the position) and v3-style (arrays under
  `labels/`) overlay locations are handled.
- **`--attention` must be run as a script.** `view_by_gene.py` imports the
  attention module by bare name (`from view_by_gene_attention import …`), so
  `python -m cyclops_process.napari.view_by_gene … --attention` fails with
  `ModuleNotFoundError`. Run the file path instead
  (`uv run python cyclops_process/napari/view_by_gene.py …`) so the script's own
  directory lands on `sys.path`. All other modes work fine with `-m`.
- **`--mode cell_paint --action view` is broken.** It imports
  `cyclops_process.napari.dask.cell_paint`, which does not exist in the repo. The
  `cell_paint` *build* actions still work.
- **Attention-atlas caches.** Attention CSVs and per-(experiment, well) NTC pools
  are cached under `/path/to/ops_data/cache/` (override with
  `$OPS_CACHE_DIR`) so the one-time build cost is shared between users.
  `--top-markers 0` and `--preload-layers` both create hundreds of napari layers
  and make startup very slow.
- An earlier version of this README documented a `view_tile.py <experiment> <tile>`
  script; it no longer exists here — `dask/view_store.py` and
  `dask/view_dask.py --wells` cover that use case.
