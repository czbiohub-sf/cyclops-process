"""Attention-atlas mode for `view_by_gene.py`.

Renders the PMA attention atlas in napari: per-gene top-N cells per
marker, cross-experiment, with all channels of each cell loaded as
toggleable overlays. Mirrors the PDF attention atlas layout (Phase row
+ one row per fluor viz_channel, with NTC strips interleaved).

Imported on demand from `view_by_gene.view_by_gene_cli` when the user
passes `--attention`. Kept in a separate module so the bulk of the
attention-only machinery doesn't clutter `view_by_gene.py`.
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np
import napari

from ops_utils.data.experiment import OpsDataset
from ops_utils.data.bbox_utils import BaseDataset
from ops_utils.data.disk_cache import df_cache
from ops_utils.data.filesystem import resolve_experiment_name
from iohub import open_ome_zarr

# Reuse the canonical caption parser + mAP-based marker selection from
# the SHAP atlas pipeline. The atlas viewers have moved to
# ``ops_model/src/ops_model/models/attention/atlas/``; add that to
# sys.path so the import works regardless of cwd. We deliberately
# DON'T re-implement these locally — the parser in particular has subtle
# regex logic for caption section boundaries (channel names with
# em-dashes / semicolons) that we must not drift from.
_ATLAS_DIR = (
    Path(__file__).resolve().parents[3]
    / "ops_model" / "src" / "ops_model" / "models" / "attention" / "atlas"
)
sys.path.insert(0, str(_ATLAS_DIR))
from attention_atlas_shap import (  # noqa: E402  (path-dependent import)
    DEFAULT_MARKER_MAP_CSV,
    DEFAULT_CHAD_CONSISTENCY_CSV,
    DEFAULT_CHAD_COMPLEX_CONFIG,
    _load_chad_consistency_matrix,
    _load_marker_map,
    _parse_caption_per_channel,
    select_top_channels,
)
from cyclops_process.paths import BASE_PATH


DEFAULT_ATTENTION_PHASE_CSV = (
    f"{BASE_PATH}/models/alex_lin_attention/v3/attention_v3/"
    "pma_top_phase_cells_v3.csv"
)
DEFAULT_ATTENTION_FLUOR_CSV = (
    f"{BASE_PATH}/models/alex_lin_attention/v3/attention_v3/"
    "pma_top_fluorescent_cells_v3.csv"
)
DEFAULT_ATTENTION_CAPTIONS_CSV = (
    f"{BASE_PATH}/models/alex_lin_attention/top20_v4/"
    "ko_shap_captions.csv"
)
# Top-marker selection uses the SHAP-atlas's canonical mAP matrix
# (DEFAULT_MARKER_MAP_CSV) imported from `attention_atlas_shap`. CHAD
# (complex-level) selection uses DEFAULT_CHAD_CONSISTENCY_CSV. Both
# constants come from the upstream module so this script picks up any
# path/version changes there automatically — do not re-define here.
# Cell mask must cover at least this many pixels for a tile to count
# as "real" — catches stale seg IDs (mask totally absent) and cells
# where the seg has drifted off-FOV. Lowered from the attention-atlas
# default of 25 because the tiled napari view has VISIBLE gaps for
# every dropped tile, so we'd rather render a partial mask than show
# a black hole. 5 px still rejects truly-missing segs while preserving
# cells whose mask is just clipped by the bbox edge.
NTC_MIN_MASK_PIXELS = 5
# Two centroids within this many pheno pixels are treated as the same
# physical cell during NTC dedup (handles re-segmentations that give the
# same cell a different seg_id). Mirrors `attention_atlas.XY_DEDUP_TOL_PX`.
XY_DEDUP_TOL_PX = 10


def _load_attention_table(csv_path: str | Path, gene: str) -> pd.DataFrame:
    """Read an attention CSV (phase or fluor) and return rows for `gene`.

    First call builds the parquet via `df_cache`; subsequent calls
    hit the cache with a pushdown filter on `gene`. Cache invalidates
    whenever the source CSV mtime advances. Lives in the shared
    /path/to/ops_data/cache/ops_utils/ root so all users on
    the cluster share the one-time ~30 s build.
    """
    csv_path = Path(csv_path)
    df = df_cache(
        namespace="attention_csv",
        key=csv_path.name,
        builder=lambda: pd.read_csv(csv_path),
        source_path=csv_path,
        read_kwargs={"filters": [("gene", "==", str(gene))]},
    )
    if df is None:
        return pd.DataFrame()
    # Pyarrow filter pushdown is type-strict; if the cached `gene` column
    # is non-string, the filter silently returns 0 rows. Re-filter
    # in-memory as a guard — cheap on the small filtered df.
    if not df.empty and not (df["gene"].astype(str) == str(gene)).all():
        df = df[df["gene"].astype(str) == str(gene)].copy()
    return df



def _dedup_attention_cells(df: pd.DataFrame, xy_tol: int = XY_DEDUP_TOL_PX) -> pd.DataFrame:
    """Drop duplicate cells from a pool. A cell is identified by
    (experiment, well, segmentation) OR by (experiment, well, xy-bin) so
    a re-segmentation that gives the same cell a different seg_id still
    dedups. Mirrors `attention_atlas._dedup_cells`. Keeps first row so
    the higher-ranked copy is preserved.
    """
    if df is None or df.empty:
        return df
    subset = [c for c in ("experiment", "well", "segmentation") if c in df.columns]
    if subset:
        df = df.drop_duplicates(subset=subset, keep="first")
    if (
        xy_tol
        and {"x_pheno", "y_pheno", "experiment", "well"}.issubset(df.columns)
    ):
        df = df.copy()
        df["_xb"] = (df["x_pheno"].astype(float) / xy_tol).round().astype("Int64")
        df["_yb"] = (df["y_pheno"].astype(float) / xy_tol).round().astype("Int64")
        df = df.drop_duplicates(subset=["experiment", "well", "_xb", "_yb"], keep="first")
        df = df.drop(columns=["_xb", "_yb"])
    return df.reset_index(drop=True)



def view_by_gene_attention(
    gene,
    top_n=10,
    crop_size=400,
    mask_dilation=10,
    phase_csv=None,
    fluor_csv=None,
    captions_csv=None,
    marker_map_csv=None,
    chad_config=None,
    aggregation_level="gene",
    top_markers=3,
    preload_layers=False,
):
    """Attention-atlas mode: cross-experiment top-N cells per marker.

    Mirrors the PDF attention atlas in napari but with all channels of
    every cell loaded as toggleable overlays (the user can flip Phase
    on/off under the actin tiles, see nuclei_prediction over fluor, etc).

    Layout:
      - Row 0:   Phase top-N for `gene`
      - Rows 1+: One row per fluor viz_channel that has top-attention cells
                 for `gene`, top-N each.

    Cells span possibly many experiments. The channel layers are the
    UNION of channel names across all source zarrs; each tile fills only
    the channels its experiment has, leaving missing slots as zeros.

    The mask overlay (red inverse-dilation) and per-tile coordinate
    annotations match the existing view_by_gene UI.
    """
    import re
    import warnings
    import types as _types
    from concurrent.futures import ThreadPoolExecutor
    from scipy.ndimage import binary_dilation

    # Phase-level timer. Single-element list so closures can mutate it
    # (nonlocal would also work but is uglier across nested defs).
    _t = [time.perf_counter()]
    _t_total = time.perf_counter()

    def _phase(label: str) -> None:
        now = time.perf_counter()
        print(f"  [time] {label}: {now - _t[0]:.2f}s", flush=True)
        _t[0] = now

    phase_csv = phase_csv or DEFAULT_ATTENTION_PHASE_CSV
    fluor_csv = fluor_csv or DEFAULT_ATTENTION_FLUOR_CSV

    print(f"Loading attention CSVs:")
    print(f"  phase: {phase_csv}")
    print(f"  fluor: {fluor_csv}")
    # v3 attention CSVs are 10M+ rows each (~700 MB on disk). pandas's
    # default CSV parser takes 20-30 s to load them; we only ever filter
    # to a single gene afterwards. Cache a sidecar parquet next to each
    # CSV (mtime-keyed) and load with a pushdown gene filter — turns the
    # 20+ s cold load into ~0.5 s after the first run.
    phase_df = _load_attention_table(phase_csv, gene)
    fluor_df = _load_attention_table(fluor_csv, gene)
    _phase("attention CSV load (gene filter)")

    if phase_df.empty and fluor_df.empty:
        print(f"No attention rows for gene '{gene}' — check spelling against the CSVs.")
        return

    # v3 schema renamed `viz_channel` → `channel` and dropped `channel_rank`.
    # Alias `channel` so the rest of this function (and downstream caption
    # parsing) keeps using the v2 name.
    for df in (phase_df, fluor_df):
        if "channel" in df.columns and "viz_channel" not in df.columns:
            df.rename(columns={"channel": "viz_channel"}, inplace=True)

    # ── Top-N marker filter ────────────────────────────────────────────
    # Without this filter, the atlas renders every fluor channel that
    # has any top-attention cell for the gene (~50-60 channels for v3
    # cohort), which blows up napari layer creation. Use the same
    # mAP-based selection the SHAP atlas uses, and pick the SAME matrix
    # the SHAP atlas would for this aggregation level:
    #   gene-level    → mAP DISTINCTIVENESS  (gene_reporter_distinctiveness_raw.csv)
    #   complex-level → mAP CONSISTENCY      (complex_reporter_chad_consistency.csv)
    # Mirrors `attention_atlas_shap.build_shap_data` — distinctiveness
    # is never used at complex level (per upstream user spec).
    # `top_markers <= 0` opts out and renders all markers.
    if top_markers and top_markers > 0 and not fluor_df.empty:
        if aggregation_level == "complex":
            consistency_path = marker_map_csv or DEFAULT_CHAD_CONSISTENCY_CSV
            cfg_path = chad_config or DEFAULT_CHAD_COMPLEX_CONFIG
            try:
                map_df = _load_chad_consistency_matrix(consistency_path, cfg_path)
                map_label = "mAP CONSISTENCY"
                map_path_name = Path(consistency_path).name
            except SystemExit as e:
                # Upstream raises SystemExit with regen instructions when
                # the consistency CSV is missing — re-print the message
                # but keep the viewer open with all markers.
                print(f"  [top-markers] {e}", flush=True)
                map_df = None
                map_label = "mAP CONSISTENCY"
                map_path_name = Path(consistency_path).name
        else:
            map_path = marker_map_csv or DEFAULT_MARKER_MAP_CSV
            map_df = _load_marker_map(map_path)
            map_label = "mAP DISTINCTIVENESS"
            map_path_name = Path(map_path).name

        available = fluor_df["viz_channel"].astype(str).unique().tolist()
        # `select_top_channels` prepends "Phase" if present in `available`
        # — strip it, since Phase rows come from phase_df, not fluor_df.
        selected, _map_per_ch = select_top_channels(
            gene, available, map_df, top_k=top_markers,
        )
        keep = [c for c in selected if c.lower() != "phase"]
        if keep and map_df is not None and gene in map_df.index:
            print(f"Top {len(keep)} markers for {gene} by {map_label}: {keep}",
                  flush=True)
            fluor_df = fluor_df[fluor_df["viz_channel"].isin(keep)].copy()
        else:
            print(f"  [top-markers] no {map_label} ranking found for {gene} "
                  f"in {map_path_name} — rendering all "
                  f"{fluor_df['viz_channel'].nunique()} markers")

    # Build the row plan. Phase = top N. Fluor = one row per viz_channel,
    # restricted to rank_type=='top' so we mirror the atlas's KO-only
    # selection (the atlas pulls "bottom" rows separately for NTC strip).
    # KO rows first; NTC rows are interleaved AFTER we have the experiment
    # set (NTC sampling pulls from the same wells the KO cells came from).
    ko_rows: list[tuple[str, pd.DataFrame]] = []
    if not phase_df.empty:
        ko_rows.append(("Phase", phase_df.sort_values("rank").head(top_n)))
    if not fluor_df.empty:
        rt = fluor_df.get("rank_type")
        ftop = fluor_df if rt is None else fluor_df[rt == "top"]
        if "channel_rank" in ftop.columns:
            ftop = ftop.sort_values(["channel_rank", "rank"])
        for ch_name, gdf in ftop.groupby("viz_channel", sort=False):
            ko_rows.append((str(ch_name), gdf.sort_values("rank").head(top_n)))
    if not ko_rows:
        print("No tiles to render.")
        return

    print(f"KO rows: {[(lbl, len(df)) for lbl, df in ko_rows]}")

    # ── Open one OME-Zarr store per unique experiment ──────────────────
    # Parallelized — each open is ~0.5–2 s of metadata reads + config
    # loading; with 10+ experiments per gene, the sequential version
    # used to dominate cold-start time.
    all_cells = pd.concat([df for _, df in ko_rows], ignore_index=True)
    experiments = sorted(set(all_cells["experiment"].astype(str)))
    print(f"Opening {len(experiments)} experiment stores in parallel...")
    stores: dict[str, object] = {}
    exp_channel_names: dict[str, list[str]] = {}
    ops_datasets: dict[str, OpsDataset] = {}
    channel_map_per_exp: dict[str, dict] = {}

    def _open_one(exp: str):
        try:
            ds_obj = OpsDataset(resolve_experiment_name(exp))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", module="zarr")
                store = open_ome_zarr(ds_obj.store_paths["pheno_assembled_v3"], mode="r")
            return exp, ds_obj, store, None
        except Exception as e:
            return exp, None, None, str(e)

    with ThreadPoolExecutor(max_workers=min(len(experiments), 12)) as ex:
        for exp, ds_obj, store, err in ex.map(_open_one, experiments):
            if err is not None:
                print(f"  Skipping {exp}: {err}")
                continue
            stores[exp] = store
            exp_channel_names[exp] = list(store.channel_names)
            ops_datasets[exp] = ds_obj
            channel_map_per_exp[exp] = dict(getattr(ds_obj, "channel_map_data", {}) or {})

    if not stores:
        print("Failed to open any experiment store.")
        return
    _phase(f"open {len(stores)} OME-Zarr stores")

    # ── NTC pool cache (in-memory + on-disk parquet) ───────────────────
    # The original implementation re-filtered the full linked_results df
    # for every KO row and every (exp, well) it touched — with ~25 rows ×
    # ~15 well-overlap, that was ~375 redundant pandas filter ops on
    # 50k-row dfs. Now: each (exp, well) gets one cached NTC-only pool,
    # built once on cache miss and persisted to parquet. Subsequent runs
    # of *any* gene that touches the same well skip the linked_results
    # CSV entirely (parquet read ≈ 10–50 ms vs ~0.5–2 s + ~30 ms filter).
    KEEP_COLS = ["experiment", "well", "segmentation", "y_pheno", "x_pheno"]
    NTC_OPT_COLS = ["sgRNA", "barcode"]

    # Process-local memoization on top of the shared df_cache disk store
    # — avoids the parquet read on the second call for the same (exp,
    # well) within a single run (e.g. when many KO rows share a well).
    ntc_pool_mem: dict[tuple[str, str], pd.DataFrame | None] = {}

    def _build_ntc_pool(exp: str, well_2lvl: str) -> pd.DataFrame | None:
        """Read linked_results, filter to NTC + valid coords, return the
        per-well pool. Called by `df_cache` on cold misses only."""
        if exp not in ops_datasets:
            return None
        try:
            csv_path = ops_datasets[exp].append_well("linked_results", well_2lvl)
            if not csv_path.exists():
                return None
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  [ntc] linked_results read failed ({exp} {well_2lvl}): {e}")
            return None
        gene_col = "gene_name" if "gene_name" in df.columns else None
        seg_col = "segmentation_id" if "segmentation_id" in df.columns else None
        if gene_col is None or seg_col is None:
            return None
        ntc_mask = (
            df[gene_col].isna()
            | df[gene_col].astype(str).str.startswith("NTC")
        )
        sub = df[ntc_mask].copy()
        sub["segmentation"] = pd.to_numeric(sub[seg_col], errors="coerce")
        sub = sub[sub["segmentation"] > 0]
        for col in ("y_pheno", "x_pheno"):
            if col not in sub.columns:
                return None
            sub = sub[sub[col].notna()]
        if sub.empty:
            return None
        sub["experiment"] = exp
        sub["well"] = well_2lvl
        keep_cols = list(KEEP_COLS) + [c for c in NTC_OPT_COLS if c in sub.columns]
        return sub[keep_cols].reset_index(drop=True)

    def _ntc_source_path(exp: str, well_2lvl: str) -> Path | None:
        if exp not in ops_datasets:
            return None
        try:
            return ops_datasets[exp].append_well("linked_results", well_2lvl)
        except Exception:
            return None

    def _ntc_cache_path(exp: str, well_2lvl: str) -> Path:
        # Mirror df_cache's internal key normalization so the pre-warm
        # warm/cold counter can stat the same file df_cache will read.
        from ops_utils.data.disk_cache import _cache_path
        return _cache_path("ntc_pool", f"{exp}__{well_2lvl}", None)

    def _ntc_pool(exp: str, well_pheno: str) -> pd.DataFrame | None:
        """Cached NTC pool for one (exp, well). Disk-backed via df_cache,
        with an in-process memo on top so the second call in this run
        skips the parquet read entirely."""
        well_2lvl = well_pheno.replace("/0", "")
        mkey = (exp, well_2lvl)
        if mkey in ntc_pool_mem:
            return ntc_pool_mem[mkey]
        pool = df_cache(
            namespace="ntc_pool",
            key=f"{exp}__{well_2lvl}",
            builder=lambda: _build_ntc_pool(exp, well_2lvl),
            source_path=_ntc_source_path(exp, well_2lvl),
        )
        ntc_pool_mem[mkey] = pool
        return pool

    def _sample_ntc_row(ko_df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Sample n NTC cells from the same (exp, well) pairs as the
        given KO row's cells. Pulls each (exp, well) NTC pool from the
        shared cache (in-memory after pre-warm), concats, dedups across
        wells, then samples distributed across (exp, well) pairs.
        """
        ko = ko_df.copy()
        ko["__exp"] = ko["experiment"].astype(str)
        ko["__well_pheno"] = ko["well"].astype(str).where(
            ko["well"].astype(str).str.contains("/"),
            ko["well"].astype(str).map(
                lambda w: re.sub(r"^([A-Za-z]+)(\d+)$", r"\1/\2", str(w))
            ),
        ) + "/0"
        groups = list(ko.groupby(["__exp", "__well_pheno"]).groups.keys())

        pool_pieces = [
            p for p in (_ntc_pool(exp, wp) for exp, wp in groups)
            if p is not None and not p.empty
        ]
        if not pool_pieces:
            return pd.DataFrame()
        pool = pd.concat(pool_pieces, ignore_index=True)
        pool = _dedup_attention_cells(pool, xy_tol=XY_DEDUP_TOL_PX)
        if pool.empty:
            return pd.DataFrame()

        # Sample n cells distributed across (exp, well) groups, then top
        # up from any unsampled remainder so we always return n cells
        # whenever the pool has them. The previous version used floor-
        # divided per-group quotas (n // ngroups) which left visible
        # gaps in the atlas — e.g. n=10 with 4 wells yielded 8 cells.
        groups = list(pool.groupby(["experiment", "well"]))
        ngroups = max(1, len(groups))
        # Ceiling division: per_group quota covers any rounding loss.
        per_group = max(1, (n + ngroups - 1) // ngroups)
        rng_seed = abs(hash(("ntc", str(gene)))) % (2**31)
        out_pieces: list[pd.DataFrame] = []
        used_idx: set = set()
        for (_exp, _well), gdf in groups:
            k = min(per_group, len(gdf))
            if k <= 0:
                continue
            chosen = gdf.sample(k, random_state=rng_seed)
            out_pieces.append(chosen)
            used_idx |= set(chosen.index)
            rng_seed += 1
        sampled = (
            pd.concat(out_pieces) if out_pieces else pd.DataFrame()
        )
        # Top up from the unsampled remainder if balanced sampling
        # didn't reach n (small wells, ceiling overshoot trimming).
        if len(sampled) < n and len(pool) > len(used_idx):
            remaining = pool.drop(index=list(used_idx))
            extra_n = min(n - len(sampled), len(remaining))
            if extra_n > 0:
                extra = remaining.sample(extra_n, random_state=rng_seed)
                sampled = pd.concat([sampled, extra])
        sampled = sampled.head(n).reset_index(drop=True)
        sampled["rank"] = 0
        sampled["pma_attention"] = 0.0
        return sampled

    # Pre-warm the per-(exp, well) NTC pool cache in parallel. Disk-cache
    # hits return immediately; misses pay the linked_results CSV read +
    # filter cost once and persist the result. Either way, downstream
    # `_sample_ntc_row` calls become pure in-memory concat + sample.
    unique_pairs: set[tuple[str, str]] = set()
    for _label, ko_df in ko_rows:
        for _, row in ko_df.iterrows():
            exp = str(row["experiment"])
            well_str = str(row["well"])
            if "/" not in well_str:
                m = re.match(r"^([A-Za-z]+)(\d+)$", well_str)
                if m:
                    well_str = f"{m.group(1)}/{m.group(2)}"
            unique_pairs.add((exp, f"{well_str}/0"))
    if unique_pairs:
        # Count how many will hit the disk cache vs. need rebuilding so
        # the user can tell at a glance whether a slow run was a cold
        # cache or a structural problem.
        warm = sum(
            1 for exp, wp in unique_pairs
            if _ntc_cache_path(exp, wp.replace("/0", "")).exists()
        )
        cold = len(unique_pairs) - warm
        print(f"Pre-loading {len(unique_pairs)} NTC pools in parallel "
              f"(disk cache: {warm} warm / {cold} cold)...")
        with ThreadPoolExecutor(max_workers=min(len(unique_pairs), 16)) as ex:
            ex.map(lambda p: _ntc_pool(*p), unique_pairs)
    _phase(f"NTC pool pre-warm ({len(unique_pairs)} wells)")

    # Interleave: for each KO row, append its matching NTC row right after.
    rows: list[tuple[str, pd.DataFrame, str]] = []
    for label, ko_df in ko_rows:
        rows.append((label, ko_df, "ko"))
        ntc_df = _sample_ntc_row(ko_df, top_n)
        if len(ntc_df):
            rows.append((f"{label} (NTC)", ntc_df, "ntc"))
        else:
            print(f"  [ntc] no NTC cells found for row '{label}'")
    print(f"Rows to render (KO + NTC): {[(lbl, len(df), kind) for lbl, df, kind in rows]}")
    _phase(f"NTC sampling ({len(rows)} rows)")

    # Union of channel names across experiments. Order: phase channels
    # first (Phase / Phase2D / BF / VS), then everything else in
    # experiment-encounter order so the napari layer list is stable.
    PHASE_PRIORITY = {"Phase2D", "Phase", "BF", "VS", "Focus3D"}
    union_channels: list[str] = []
    seen: set[str] = set()
    for exp in experiments:
        for c in exp_channel_names.get(exp, []):
            if c in PHASE_PRIORITY and c not in seen:
                seen.add(c); union_channels.append(c)
    for exp in experiments:
        for c in exp_channel_names.get(exp, []):
            if c not in seen:
                seen.add(c); union_channels.append(c)
    n_channels = len(union_channels)
    print(f"Union of {n_channels} channels: {union_channels}")
    ch_global_index = {c: i for i, c in enumerate(union_channels)}

    # ── Build labels_df for BaseDataset ────────────────────────────────
    cell_records: list[dict] = []
    for r_idx, (label, df, kind) in enumerate(rows):
        for c_idx, (_, row) in enumerate(df.iterrows()):
            exp = str(row["experiment"])
            if exp not in stores:
                continue
            well_str = str(row["well"])
            if "/" not in well_str:
                m = re.match(r"^([A-Za-z]+)(\d+)$", well_str)
                if m:
                    well_str = f"{m.group(1)}/{m.group(2)}"
            seg_id = row.get("segmentation")
            if pd.isna(seg_id):
                continue
            cell_records.append({
                "store_key":       exp,
                "well":            f"{well_str}/0",
                "y_pheno":         int(row["y_pheno"]),
                "x_pheno":         int(row["x_pheno"]),
                "segmentation_id": int(seg_id),
                "row_idx":         r_idx,
                "col_idx":         c_idx,
                "row_label":       label,
                "row_kind":        kind,
                "experiment":      exp,
                "rank":            int(row.get("rank", 0)) if pd.notna(row.get("rank", 0)) else 0,
                "pma_attention":   float(row.get("pma_attention", 0.0)),
                "total_index":     len(cell_records),
            })
    if not cell_records:
        print("All candidate rows lacked a usable segmentation_id / store.")
        return
    labels_df = pd.DataFrame(cell_records)

    # BaseDataset reads `bbox` directly from labels_df (it doesn't auto-
    # compute from y_pheno/x_pheno). Build it here from the centroid + crop.
    half = crop_size // 2
    labels_df["bbox"] = [
        [int(r.y_pheno) - half, int(r.x_pheno) - half,
         int(r.y_pheno) + half, int(r.x_pheno) + half]
        for r in labels_df.itertuples()
    ]

    # BaseDataset needs gene_name for some internal label lookups; fill it.
    labels_df["gene_name"] = str(gene)

    base_dataset = BaseDataset(
        stores=stores,
        labels_df=labels_df,
        initial_yx_patch_size=(crop_size, crop_size),
        final_yx_patch_size=(crop_size, crop_size),
        out_channels="all",
        mask_cell=False,
    )

    # ── Per-row default marker resolution ──────────────────────────────
    # Computed BEFORE the cell load so we can drive lazy channel loading:
    # only the row's primary channel + a phase backdrop are read up
    # front; everything else is fetched on first visibility-toggle.
    PHASE_CHS = {"Phase2D", "Phase", "BF", "VS"}
    AUX_CHS = {"Focus3D", "nuclei_prediction", "membrane_prediction"}

    row_experiments: dict[int, set[str]] = {}
    for r_idx in range(len(rows)):
        row_experiments[r_idx] = set(
            labels_df.loc[labels_df["row_idx"] == r_idx, "store_key"].astype(str)
        )

    def _norm_ch(s: str) -> str:
        return re.sub(r"[\s\-_/]+", "", str(s)).lower()

    def _row_cell_experiments(r_idx: int) -> list[str]:
        """Per-cell experiment list for this row (preserves duplicates so
        the majority-vote primary-channel resolution weights by actual
        cell count, not unique-experiment count)."""
        return labels_df.loc[
            labels_df["row_idx"] == r_idx, "store_key"
        ].astype(str).tolist()

    def _resolve_primary_channels(row_label: str, kind: str, exps: set[str], r_idx: int) -> set[str]:
        """Return AT MOST ONE channel name to default-show for this row.
        Phase rows: most-common phase channel. Fluor rows: most-common
        zarr channel that resolves to the row's viz_channel via the
        cell's experiment channel_map (or normalized-name match for CP/4i
        where the marker name is baked into the zarr label). Returns
        set() if nothing resolves."""
        from collections import Counter
        present_per_exp = {e: exp_channel_names.get(e, []) for e in exps}
        present = set().union(*present_per_exp.values()) if present_per_exp else set()
        if not present:
            return set()
        cell_exps = _row_cell_experiments(r_idx)
        base_label = row_label.replace(" (NTC)", "").strip()

        if base_label.lower() == "phase":
            counts: Counter = Counter()
            for e in cell_exps:
                for c in present_per_exp.get(e, []):
                    if c in PHASE_CHS:
                        counts[c] += 1
            return {counts.most_common(1)[0][0]} if counts else set()

        viz_norm = _norm_ch(base_label)
        counts: Counter = Counter()
        for e in cell_exps:
            cmap = channel_map_per_exp.get(e, {})
            row_present = present_per_exp.get(e, [])
            for zarr_ch, bio in cmap.items():
                if bio and _norm_ch(bio) == viz_norm and zarr_ch in row_present:
                    counts[zarr_ch] += 1
            for ch in row_present:
                if _norm_ch(ch) == viz_norm:
                    counts[ch] += 1
        if counts:
            return {counts.most_common(1)[0][0]}

        counts = Counter()
        for e in cell_exps:
            for c in present_per_exp.get(e, []):
                if c not in PHASE_CHS and c not in AUX_CHS:
                    counts[c] += 1
        return {counts.most_common(1)[0][0]} if counts else set()

    primary_chs_per_row: dict[int, set[str]] = {
        r_idx: _resolve_primary_channels(label, kind, row_experiments[r_idx], r_idx)
        for r_idx, (label, _df, kind) in enumerate(rows)
    }
    for r_idx, (label, _df, kind) in enumerate(rows):
        print(f"  row {r_idx} '{label}' ({kind}): primary = "
              f"{sorted(primary_chs_per_row[r_idx])}")

    # ── Lazy-load channel plan ────────────────────────────────────────
    # At startup we only read each cell's primary channel(s) plus one
    # phase channel as a backdrop — drops the per-cell zarr reads from
    # ~14 channels to ~2. Other channels start as zero strips and
    # populate on first visibility-toggle in napari (see lazy callback
    # below). Cuts cold cell-load time from ~46 s → ~6-10 s.
    PHASE_PRIORITY_LIST = ("Phase2D", "Phase", "BF", "VS")

    def _startup_channels_for_cell(rec) -> list[str]:
        """Channels to read at startup for one cell — in the order they
        appear in the cell's source zarr (matches the data-tensor slice
        order returned by BaseDataset)."""
        row_idx = int(rec["row_idx"])
        primary = primary_chs_per_row.get(row_idx, set())
        exp_chs = exp_channel_names[rec["store_key"]]
        backdrop = next(
            (c for c in PHASE_PRIORITY_LIST if c in exp_chs),
            None,
        )
        wanted = set(primary)
        if backdrop:
            wanted.add(backdrop)
        return [c for c in exp_chs if c in wanted]

    cell_load_chs_startup: list[list[str]] = [
        _startup_channels_for_cell(labels_df.iloc[i])
        for i in range(len(labels_df))
    ]

    # Per-cell channel filter that BaseDataset._get_channels consults.
    # `mode="startup"` reads cell_load_chs_startup; `mode="single"` reads
    # exactly the channel named in `_active_filter["channel"]`. The
    # placement loop and the lazy-load callback flip between these two
    # modes under a lock so a lazy load doesn't perturb an in-flight
    # startup load.
    import threading
    _ds_lock = threading.Lock()
    _active_filter = {"mode": "startup", "channel": None}

    def _get_channels_filtered(self, ci):
        all_names = exp_channel_names[ci.store_key]
        if _active_filter["mode"] == "single":
            ch = _active_filter["channel"]
            if ch in all_names:
                return [ch], [all_names.index(ch)]
            return [], []
        wanted = cell_load_chs_startup[int(ci.total_index)]
        # Order indices by exp-zarr position so the data tensor's slice
        # order matches `wanted` exactly (caller iterates in this order).
        indices = [all_names.index(c) for c in wanted]
        return list(wanted), indices

    base_dataset._get_channels = _types.MethodType(_get_channels_filtered, base_dataset)

    # ── Caption lookup (optional left-gutter overlay) ──────────────────
    # Resolved before geometry so the gutter width can be set to 0 when
    # there's no caption to display (saves screen space). Pass --captions
    # = "" to force-disable.
    caption_per_row: dict[str, str] = {}
    if captions_csv != "":
        cap_path = captions_csv or DEFAULT_ATTENTION_CAPTIONS_CSV
        if os.path.exists(cap_path):
            try:
                cap_df = pd.read_csv(cap_path)
                hit = cap_df[cap_df["gene"].astype(str) == str(gene)]
                if len(hit):
                    caption_text = str(hit.iloc[0]["caption"])
                    # Caption sections are keyed on KO labels only
                    # (no "<ch> (NTC)" sections in the CSV).
                    parsed = _parse_caption_per_channel(
                        caption_text,
                        [lbl for lbl, _, kind in rows if kind == "ko"],
                    )
                    caption_per_row = {k: v for k, v in parsed.items() if v}
                    print(f"Loaded captions for {gene} from {cap_path} "
                          f"({len(caption_per_row)} sections)")
                else:
                    print(f"No caption row for {gene} in {cap_path}")
            except Exception as e:
                print(f"  Captions disabled — failed to read {cap_path}: {e}")
        else:
            print(f"  Captions disabled — file not found: {cap_path}")

    # ── Tile geometry ──────────────────────────────────────────────────
    n_rows = len(rows)
    n_cols = top_n
    label_spacing = 24
    cell_size_with_border = crop_size + 2
    row_height = cell_size_with_border + label_spacing
    # Left-side caption gutter — wide enough for ~45 chars at size=12 px.
    # 0 when no captions were loaded so the gutter doesn't waste space.
    gutter_w = 380 if caption_per_row else 0
    caption_wrap_chars = 42
    grid_h = n_rows * row_height
    grid_w = gutter_w + n_cols * cell_size_with_border

    # Per-(row_idx, channel_name) strips, allocated only for the
    # (row, channel) pairs that actually receive data at startup.
    # Replaces the previous monolithic `tiled_array` of shape
    # (n_channels, grid_h, grid_w) which, for a 114-row × top-3
    # gene with 30 union channels, would allocate 7+ GB of zeros
    # — most of them never written. Per-strip allocation drops that
    # to ~2 MB × ~120 strips ≈ 250 MB.
    row_ch_strips: dict[tuple[int, str], np.ndarray] = {}
    strip_shape = (cell_size_with_border, grid_w)

    def _get_strip(r_idx: int, ch_name: str) -> np.ndarray:
        key = (r_idx, ch_name)
        s = row_ch_strips.get(key)
        if s is None:
            s = np.zeros(strip_shape, dtype=np.float32)
            row_ch_strips[key] = s
        return s

    inverse_mask_array = np.ones((grid_h, grid_w), dtype=np.uint8)

    # ── Parallel cell load ─────────────────────────────────────────────
    # Threads (not processes): zarr / tensorstore reads release the GIL
    # so high concurrency is fine. Bumped cap from cpu_count to 64 since
    # each read is I/O-bound on the zarr filesystem; with 80+ cells × 14
    # channels we need more in-flight reads than the default cpu_count.
    n_workers = min(len(labels_df), 64)
    print(f"Loading {len(labels_df)} cells across {n_rows} rows × {n_cols} cols "
          f"using {n_workers} workers...")

    def _load_cell(i):
        try:
            batch = base_dataset[i]
            return i, batch["data"].numpy(), batch["mask"].numpy(), None
        except Exception as e:
            return i, None, None, repr(e)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        loaded = list(ex.map(_load_cell, range(len(labels_df))))
    _phase(f"load {len(labels_df)} cells (zarr block reads)")

    coord_points: list = []
    coord_texts:  list = []
    failures = 0
    first_err = None
    dropped_sparse = 0
    for i, data, mask, err in loaded:
        if data is None:
            failures += 1
            if first_err is None:
                first_err = (i, err)
            continue
        rec = labels_df.iloc[i]
        # Drop tiles whose seg mask is essentially absent (stale seg ID
        # / cell drifted off-FOV). Threshold is tuned low so partially
        # clipped masks still render rather than leaving black holes
        # in the atlas grid.
        cell_mask_check = mask[0].astype(bool)
        if int(cell_mask_check.sum()) < NTC_MIN_MASK_PIXELS:
            dropped_sparse += 1
            continue
        r = int(rec["row_idx"]); c = int(rec["col_idx"])
        exp = rec["store_key"]
        # Pad with 1px border, then place at union-channel positions.
        crop_b = np.pad(data, ((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=0)
        y_start = r * row_height + label_spacing
        x_start = gutter_w + c * cell_size_with_border
        # `data` only contains the channels we asked for at startup
        # (cell_load_chs_startup[i]) — iterate that, not the full exp
        # channel list, since the slice index along axis 0 corresponds
        # to the requested-channel order.
        loaded_chs = cell_load_chs_startup[i]
        for ch_local, ch_name in enumerate(loaded_chs):
            if ch_name not in ch_global_index:
                continue
            strip = _get_strip(r, ch_name)
            strip[:, x_start:x_start + cell_size_with_border] = crop_b[ch_local]
        # Inverse mask (carved by dilated cell outline)
        cell_mask = mask[0].astype(bool)
        dil = binary_dilation(cell_mask, iterations=mask_dilation) if mask_dilation > 0 else cell_mask
        inv = (~dil).astype(np.uint8)
        inv_pad = np.pad(inv, ((1, 1), (1, 1)), mode="constant", constant_values=1)
        inverse_mask_array[
            y_start:y_start + cell_size_with_border,
            x_start:x_start + cell_size_with_border,
        ] = inv_pad
        # Per-tile coord text
        well_label = str(rec["well"]).replace("/0", "").replace("/", "")
        coord_points.append([y_start + 4, x_start + 4])
        coord_texts.append(
            f"{exp} {well_label}\nrank={int(rec['rank'])}  "
            f"attn={rec['pma_attention']:.2e}"
        )

    if failures:
        print(f"  Warning: {failures} cells failed to load.")
        if first_err is not None:
            i, err = first_err
            rec = labels_df.iloc[i]
            print(f"    First error (cell {i}, exp={rec['store_key']}, "
                  f"well={rec['well']}, seg={rec['segmentation_id']}): {err}")
    if dropped_sparse:
        print(f"  Dropped {dropped_sparse} tiles with mask < "
              f"{NTC_MIN_MASK_PIXELS} px (stale seg / off-FOV cell).")

    # ── Napari assembly ────────────────────────────────────────────────
    print("Creating napari Viewer...", flush=True)
    viewer = napari.Viewer()
    _phase("napari Viewer init")

    color_dict = {
        "GFP": "green", "mCherry": "magenta",
        "Phase": "gray", "Phase2D": "gray", "BF": "gray", "VS": "gray",
        "Cy5": "cyan", "farred": "cyan",
        "Focus3D": "gray",
        "nuclei_prediction": "blue", "membrane_prediction": "magenta",
    }
    contrast_limits_dict = {"Phase2D": (-0.5, 0.8), "Focus3D": (-0.5, 0.8)}

    # Per-row × per-channel layers. Each layer covers ONE row (a thin
    # horizontal strip) so visibility is row-specific. The strip is
    # placed in the global canvas via napari `translate=(y_start, 0)`.
    #
    # Two channel sets per row decide what we render:
    #   row_loaded_chs[r]    — channels populated at startup (primary
    #                          + phase backdrop). Strip is non-zero.
    #   row_potential_chs[r] — every channel any cell in this row's
    #                          experiments has. We create layers for
    #                          all of these (initially zero strip if
    #                          not in row_loaded_chs) and attach a
    #                          lazy-load callback to populate them
    #                          on first visibility-toggle.
    row_loaded_chs: dict[int, set[str]] = {}
    row_potential_chs: dict[int, set[str]] = {}
    for cell_idx in range(len(labels_df)):
        rec = labels_df.iloc[cell_idx]
        r_idx = int(rec["row_idx"])
        row_loaded_chs.setdefault(r_idx, set()).update(cell_load_chs_startup[cell_idx])
        row_potential_chs.setdefault(r_idx, set()).update(
            exp_channel_names[rec["store_key"]]
        )

    def _load_row_channel(r_idx: int, ch_name: str) -> np.ndarray | None:
        """Read one channel for all cells in row r_idx and return the
        full row strip (ready to assign to layer.data). Runs on a
        background thread; serializes against startup loads via
        `_ds_lock` so they don't perturb each other's `_get_channels`
        return values."""
        row_cells = labels_df.index[labels_df["row_idx"] == r_idx].tolist()
        strip = np.zeros((cell_size_with_border, grid_w), dtype=np.float32)
        with _ds_lock:
            prev_mode = _active_filter["mode"]
            prev_channel = _active_filter["channel"]
            _active_filter["mode"] = "single"
            _active_filter["channel"] = ch_name
            try:
                for i in row_cells:
                    rec = labels_df.iloc[i]
                    if ch_name not in exp_channel_names[rec["store_key"]]:
                        continue
                    try:
                        batch = base_dataset[i]
                    except Exception as e:
                        print(f"  [lazy] cell {i} ({rec['store_key']} {rec['well']}): {e}")
                        continue
                    data = batch["data"]
                    if hasattr(data, "numpy"):
                        data = data.numpy()
                    if data.ndim < 3 or data.shape[0] == 0:
                        continue
                    c = int(rec["col_idx"])
                    x_start = gutter_w + c * cell_size_with_border
                    crop_b = np.pad(
                        data, ((0, 0), (1, 1), (1, 1)),
                        mode="constant", constant_values=0,
                    )
                    strip[:, x_start:x_start + cell_size_with_border] = crop_b[0]
            finally:
                _active_filter["mode"] = prev_mode
                _active_filter["channel"] = prev_channel
        return strip

    def _attach_lazy_loader(layer, r_idx: int, ch_idx: int, ch_name: str,
                            y_start: int, y_end: int) -> None:
        """First time `layer` becomes visible, kick off a background
        read of (r_idx, ch_name) and update `layer.data` when it lands.
        Idempotent — the closure flag prevents reloads."""
        from napari.qt.threading import thread_worker
        state = {"loaded": False, "loading": False, "name": layer.name}

        def _on_visible(event):
            visible = bool(getattr(event, "value", layer.visible))
            if not visible or state["loaded"] or state["loading"]:
                return
            state["loading"] = True
            layer.name = f"{state['name']} [loading…]"

            @thread_worker
            def _work():
                return _load_row_channel(r_idx, ch_name)

            def _on_done(strip_data):
                try:
                    if strip_data is None:
                        layer.name = f"{state['name']} [empty]"
                        return
                    row_ch_strips[(r_idx, ch_name)] = strip_data
                    layer.data = strip_data
                    state["loaded"] = True
                    layer.name = state["name"]
                finally:
                    state["loading"] = False

            worker = _work()
            worker.returned.connect(_on_done)
            worker.start()

        layer.events.visible.connect(_on_visible)

    # Layers that aren't loaded at startup are created with a 1×1
    # zero placeholder rather than a full (cell_size_with_border,
    # grid_w) zero strip. Saves 2 MB × ~400 placeholder layers
    # ≈ 800 MB of allocation + napari Image-layer init time. When
    # the lazy callback fires, `layer.data = full_strip` resizes
    # automatically.
    LAZY_PLACEHOLDER = np.zeros((1, 1), dtype=np.float32)

    # Bulk layer creation. With ~570 layers (every (row, channel) the
    # screen could possibly show), the dominant cost was Qt repainting
    # the layer-list widget after each insert and emitting per-layer
    # signal cascades. We freeze paints + suppress the layers-inserted
    # event for the duration of the loop and let napari catch up once.
    n_layers_added = 0
    n_lazy_attached = 0
    qt_window = None
    try:
        qt_window = viewer.window._qt_window
        qt_window.setUpdatesEnabled(False)
    except Exception:
        # Internal API; fall back gracefully if napari moved it.
        qt_window = None

    # Default fast path: only create layers for channels we actually
    # loaded at startup (primary + phase backdrop per row, ≈ 1-3 layers
    # per row). Pre-creating layers for the full union of channels
    # (~30 channels × 114 rows ≈ 570 layers) costs napari ~30-90 s of
    # Qt setup before the viewer opens. With `preload_layers=True`,
    # the slow path is restored and every (row, channel) gets a
    # placeholder layer with a lazy-load callback so the user can
    # toggle any channel visible without reopening the viewer.
    loop_t0 = time.perf_counter()
    progress_every = 25  # rows
    try:
        for r_idx, (label, _df, kind) in enumerate(rows):
            y_start = r_idx * row_height + label_spacing
            y_end = y_start + cell_size_with_border
            primary = primary_chs_per_row.get(r_idx, set())
            loaded_now = row_loaded_chs.get(r_idx, set())
            potential = row_potential_chs.get(r_idx, set())
            for ch_idx, ch_name in enumerate(union_channels):
                if ch_name not in potential:
                    # Channel doesn't exist in any of this row's cells'
                    # experiments — no point creating a layer.
                    continue
                if ch_name in loaded_now:
                    strip = row_ch_strips.get((r_idx, ch_name))
                    data = strip if strip is not None else LAZY_PLACEHOLDER
                elif preload_layers:
                    data = LAZY_PLACEHOLDER
                else:
                    # Fast path: skip channels we didn't load at startup.
                    # User won't be able to toggle them visible from the
                    # napari layer panel — pass --preload-layers if you
                    # need that.
                    continue
                visible = ch_name in primary
                # Pass gamma to add_image directly rather than setting it
                # after — saves one Qt property emit per layer, which
                # adds up at hundreds of layers.
                layer = viewer.add_image(
                    data,
                    name=f"[{label}] {ch_name}",
                    colormap=color_dict.get(ch_name, "gray"),
                    blending="additive",
                    visible=visible,
                    contrast_limits=contrast_limits_dict.get(ch_name, None),
                    translate=(y_start, 0),
                    gamma=0.75,
                )
                n_layers_added += 1
                if ch_name not in loaded_now:
                    _attach_lazy_loader(layer, r_idx, ch_idx, ch_name, y_start, y_end)
                    n_lazy_attached += 1
            if (r_idx + 1) % progress_every == 0:
                elapsed = time.perf_counter() - loop_t0
                print(f"  [layers] {r_idx + 1}/{n_rows} rows "
                      f"({n_layers_added} layers, {elapsed:.1f}s elapsed)",
                      flush=True)
    finally:
        if qt_window is not None:
            try:
                qt_window.setUpdatesEnabled(True)
            except Exception:
                pass

    print(f"Added {n_layers_added} per-row image layers "
          f"({n_lazy_attached} lazy-loaded on first toggle).", flush=True)
    _phase("napari layer creation")

    inv_layer = viewer.add_labels(
        inverse_mask_array,
        name=f"Inverse Mask (dilation={mask_dilation}px)",
        opacity=0.3,
    )
    inv_layer.color = {1: "red"}

    # Row labels in the spacer above each row's first tile (post-gutter).
    row_label_points = []
    row_label_texts = []
    for r_idx, (label, df, kind) in enumerate(rows):
        y_pos = r_idx * row_height + label_spacing // 2
        row_label_points.append([y_pos, gutter_w + 5])
        row_label_texts.append(f"{label}  (n={len(df)})")
    viewer.add_points(
        np.array(row_label_points),
        name="Row Labels",
        text={
            "string": row_label_texts, "color": "white",
            "size": 14, "anchor": "upper_left",
        },
        size=0, face_color="transparent",
    )

    # Captions in the left gutter — wrapped to fit, anchored to the row's
    # vertical center. One Points layer with multi-line strings (napari
    # text supports embedded \n). Toggleable so it can be hidden for
    # tile inspection. Captions only attach to KO rows; the matching NTC
    # row beneath shares the same biological context but no text.
    if caption_per_row:
        import textwrap
        cap_points = []
        cap_texts = []
        for r_idx, (label, _df, kind) in enumerate(rows):
            if kind != "ko":
                continue
            text = caption_per_row.get(label, "")
            if not text:
                continue
            wrapped = "\n".join(textwrap.wrap(text, width=caption_wrap_chars))
            y_top = r_idx * row_height + label_spacing
            cap_points.append([y_top, 8])
            cap_texts.append(wrapped)
        if cap_points:
            viewer.add_points(
                np.array(cap_points),
                name="Captions",
                text={
                    "string": cap_texts, "color": "#cfe1ff",
                    "size": 11, "anchor": "upper_left",
                },
                size=0, face_color="transparent",
                visible=True,
            )

    if coord_points:
        viewer.add_points(
            np.array(coord_points),
            name="Cell Coordinates",
            text={
                "string": coord_texts, "color": "yellow",
                "size": 9, "anchor": "upper_left",
            },
            size=0, face_color="transparent",
            visible=False,
        )

    viewer.title = f"Attention Atlas — {gene} (top {top_n} per marker)"
    print(f"Displaying {len(labels_df) - failures} cells across {n_rows} rows.")
    _phase("napari assembly")
    print(f"  [time] TOTAL setup: {time.perf_counter() - _t_total:.2f}s "
          f"(then napari blocks on UI)")
    napari.run()
