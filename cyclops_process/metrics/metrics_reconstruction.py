import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import time
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import argparse

from ops_utils.data.experiment import OpsDataset


def _compute_subtile_grids(
    subtile_results: dict,
) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    t0 = time.perf_counter()
    print("[MetricsRecon] Computing per-position subtile grids...")
    positions_order: list[str] = []
    focus_grids: list[np.ndarray] = []
    zoffset_grids: list[np.ndarray] = []

    for pos, res in tqdm(subtile_results.items(), desc="subtile grids"):
        n_sub = res.get("n_subtiles", 0)
        g = res.get("grid_size", int(np.sqrt(n_sub) or 0))
        if g <= 0:
            continue

        focus_grid = np.zeros((g, g), dtype=float)
        zoffset_grid = np.zeros((g, g), dtype=float)

        for info in res.get("subtile_metadata", []):
            sid = info.get("subtile_id")
            if sid is None:
                continue
            row, col = sid // g, sid % g
            # Prefer integer focus_index (now stored as int), fallback to rounded float
            try:
                if "focus_index" in info:
                    focus_grid[row, col] = float(int(info.get("focus_index", 0)))
                elif "focus_index_float" in info:
                    focus_grid[row, col] = float(
                        int(round(float(info.get("focus_index_float", 0.0))))
                    )
                else:
                    focus_grid[row, col] = 0.0
            except Exception:
                focus_grid[row, col] = 0.0
            zoffset_grid[row, col] = float(info.get("z_focus_offset", 0.0))

        positions_order.append(pos)
        focus_grids.append(focus_grid)
        zoffset_grids.append(zoffset_grid)

    print(
        f"[MetricsRecon] Subtile grids computed for {len(positions_order)} positions in {time.perf_counter() - t0:.1f}s"
    )
    return positions_order, focus_grids, zoffset_grids


def _format_tag(tag: str | None) -> str:
    tag_local = (tag or "unknown").replace(" ", "_")
    if tag_local.endswith("-2d"):
        tag_local = tag_local[:-3]
    return tag_local


def _ensure_recon_dir(dataset: OpsDataset) -> Path:
    recon_dir = dataset.results / "phase_recon"
    try:
        recon_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return recon_dir


def _save_stat_heatmap(
    dataset: OpsDataset,
    array: np.ndarray,
    experiment: str,
    tag: str | None,
    stat_name: str,
    metric_name: str,
    cmap: str,
) -> None:
    t0 = time.perf_counter()
    print(f"[MetricsRecon] Saving {stat_name} {metric_name} heatmap...")
    tag_local = _format_tag(tag)
    recon_dir = _ensure_recon_dir(dataset)
    try:
        out_png = recon_dir / f"{experiment}_{tag_local}_{stat_name}_{metric_name}.png"
        title = f"{experiment} [{tag_local}] - {stat_name.capitalize()} {('Z Offset' if metric_name=='zoffset' else 'Focus Index')}"
        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        im = ax.imshow(array, cmap=cmap)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(
            f"[MetricsRecon] {stat_name.capitalize()} {metric_name} heatmap saved in {time.perf_counter() - t0:.1f}s"
        )
    except Exception as _e:
        print(f"[SubtileRecon][WARN] Failed saving {stat_name} {metric_name} PNG: {_e}")


def _save_combined_all_wells_png(
    dataset: OpsDataset,
    experiment: str,
    positions_order: list[str],
    focus_grids: list[np.ndarray],
    zoffset_grids: list[np.ndarray],
    tag: str | None = None,
) -> None:
    t0 = time.perf_counter()
    print(
        "[MetricsRecon] Saving combined per-position heatmaps (focus and z-offset)..."
    )
    if not focus_grids:
        return
    tag_local = (tag or "unknown").replace(" ", "_")
    if tag_local.endswith("-2d"):
        tag_local = tag_local[:-3]
    recon_dir = dataset.results / "phase_recon"
    try:
        recon_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    num_cols = len(focus_grids)
    fig_all, axes_all = plt.subplots(
        2, num_cols, figsize=(4 * num_cols, 8), squeeze=False, constrained_layout=True
    )
    vmin_focus = min(np.min(g) for g in focus_grids)
    vmax_focus = max(np.max(g) for g in focus_grids)
    vmin_z = min(np.min(g) for g in zoffset_grids)
    vmax_z = max(np.max(g) for g in zoffset_grids)
    ims_focus = []
    ims_z = []
    for idx, (pos, fg, zg) in enumerate(
        zip(positions_order, focus_grids, zoffset_grids)
    ):
        axf = axes_all[0, idx]
        imf = axf.imshow(fg, cmap="viridis", vmin=vmin_focus, vmax=vmax_focus)
        axf.set_title(f"{pos}")
        axf.set_xticks([])
        axf.set_yticks([])
        ims_focus.append(imf)
        axz = axes_all[1, idx]
        imz = axz.imshow(zg, cmap="magma", vmin=vmin_z, vmax=vmax_z)
        axz.set_xticks([])
        axz.set_yticks([])
        ims_z.append(imz)
    # Use constrained layout-friendly colorbars attached to row axes
    fig_all.colorbar(
        ims_focus[0],
        ax=axes_all[0, :].ravel().tolist(),
        fraction=0.02,
        pad=0.02,
        label="Focus index",
    )
    fig_all.colorbar(
        ims_z[0],
        ax=axes_all[1, :].ravel().tolist(),
        fraction=0.02,
        pad=0.02,
        label="Z offset",
    )
    fig_all.suptitle(f"Subtile heatmaps per well - {experiment} [{tag_local}]")
    combined_png = (
        recon_dir / f"{experiment}_{tag_local}_subtile_heatmaps_all_wells.png"
    )
    fig_all.savefig(combined_png, dpi=200)
    plt.close(fig_all)
    print(
        f"[MetricsRecon] Combined per-position heatmaps saved in {time.perf_counter() - t0:.1f}s → {combined_png}"
    )


def _save_mean_heatmaps(
    dataset: OpsDataset,
    mean_focus: np.ndarray,
    mean_zoffset: np.ndarray,
    experiment: str,
    tag: str | None = None,
) -> None:
    _save_stat_heatmap(dataset, mean_focus, experiment, tag, "mean", "focus", "viridis")
    _save_stat_heatmap(
        dataset, mean_zoffset, experiment, tag, "mean", "zoffset", "magma"
    )


def _save_median_heatmaps(
    dataset: OpsDataset,
    median_focus: np.ndarray,
    median_zoffset: np.ndarray,
    experiment: str,
    tag: str | None = None,
) -> None:
    _save_stat_heatmap(
        dataset, median_focus, experiment, tag, "median", "focus", "viridis"
    )
    _save_stat_heatmap(
        dataset, median_zoffset, experiment, tag, "median", "zoffset", "magma"
    )


def _save_std_heatmaps(
    dataset: OpsDataset,
    std_focus: np.ndarray,
    std_zoffset: np.ndarray,
    experiment: str,
    tag: str | None = None,
) -> None:
    _save_stat_heatmap(dataset, std_focus, experiment, tag, "std", "focus", "viridis")
    _save_stat_heatmap(dataset, std_zoffset, experiment, tag, "std", "zoffset", "magma")


def _parse_tile_row_col(tile_str: str) -> tuple[int, int]:
    digits = "".join(ch for ch in tile_str if ch.isdigit())
    if not digits:
        return 0, 0
    if len(digits) >= 4 and len(digits) % 2 == 0:
        half = len(digits) // 2
        return int(digits[:half]), int(digits[half:])
    val = int(digits)
    return val, val


def _save_spatial_well_heatmaps(
    subtile_results: dict, dataset: OpsDataset, experiment: str, tag: str | None = None
) -> None:
    t0 = time.perf_counter()
    print("[MetricsRecon] Saving spatial per-well mosaics (tile means)...")
    tag_local = (tag or "unknown").replace(" ", "_")
    if tag_local.endswith("-2d"):
        tag_local = tag_local[:-3]
    recon_dir = dataset.results / "phase_recon"
    try:
        recon_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    # Group positions by well and compute per-well mosaics using TILE-LEVEL statistics
    well_to_entries_mean: dict[str, list[tuple[int, int, float, float]]] = {}
    well_to_entries_median: dict[str, list[tuple[int, int, float, float]]] = {}
    well_to_entries_std: dict[str, list[tuple[int, int, float, float]]] = {}

    for pos, res in tqdm(subtile_results.items(), desc="aggregate wells"):
        parts = Path(pos).parts
        if len(parts) < 3:
            continue
        well = f"{parts[0]}/{parts[1]}"
        tile_rc = _parse_tile_row_col(parts[2])

        # Compute per-tile means from subtile metadata
        focus_vals = []
        zoffset_vals = []
        for info in res.get("subtile_metadata", []):
            try:
                # Aggregate focus using integer indices (prefer focus_index)
                if "focus_index" in info and not np.isnan(
                    info.get("focus_index", np.nan)
                ):
                    focus_vals.append(float(int(info.get("focus_index", np.nan))))
                elif "focus_index_float" in info and not np.isnan(
                    info.get("focus_index_float", np.nan)
                ):
                    f = float(info.get("focus_index_float", np.nan))
                    focus_vals.append(float(int(round(f))))
                zoffset_vals.append(float(info.get("z_focus_offset", np.nan)))
            except Exception:
                pass
        if not focus_vals or not zoffset_vals:
            continue
        tile_focus_mean = float(np.nanmean(focus_vals))
        tile_zoffset_mean = float(np.nanmean(zoffset_vals))
        tile_focus_median = float(np.nanmedian(focus_vals))
        tile_zoffset_median = float(np.nanmedian(zoffset_vals))
        tile_focus_std = float(np.nanstd(focus_vals))
        tile_zoffset_std = float(np.nanstd(zoffset_vals))

        well_to_entries_mean.setdefault(well, []).append(
            (tile_rc[0], tile_rc[1], tile_focus_mean, tile_zoffset_mean)
        )
        well_to_entries_median.setdefault(well, []).append(
            (tile_rc[0], tile_rc[1], tile_focus_median, tile_zoffset_median)
        )
        well_to_entries_std.setdefault(well, []).append(
            (tile_rc[0], tile_rc[1], tile_focus_std, tile_zoffset_std)
        )

    # Determine global color scales from TILE means (separate normalizations)
    all_focus_means = []
    all_zoffset_means = []
    for entries in well_to_entries_mean.values():
        for _, _, fmean, zmean in entries:
            all_focus_means.append(fmean)
            all_zoffset_means.append(zmean)
    if not all_focus_means:
        return
    vmin_focus = float(np.nanmin(all_focus_means))
    vmax_focus = float(np.nanmax(all_focus_means))
    vmin_z = float(np.nanmin(all_zoffset_means))
    vmax_z = float(np.nanmax(all_zoffset_means))

    # Build and save per-well circular mosaics at TILE resolution (MEAN)
    compiled_entries: list[tuple[str, np.ndarray, np.ndarray]] = []
    for well in tqdm(sorted(well_to_entries_mean.keys()), desc="build mosaics [mean]"):
        entries = well_to_entries_mean[well]
        rows = [r for r, _, _, _ in entries]
        cols = [c for _, c, _, _ in entries]
        r0, c0 = min(rows), min(cols)
        r1, c1 = max(rows), max(cols)
        H = r1 - r0 + 1
        W = c1 - c0 + 1
        focus_mosaic = np.full((H, W), np.nan, dtype=float)
        zoffset_mosaic = np.full((H, W), np.nan, dtype=float)

        for r, c, fmean, zmean in entries:
            rr = r - r0
            cc = c - c0
            focus_mosaic[rr, cc] = fmean
            zoffset_mosaic[rr, cc] = zmean

        # Flip horizontally so the first column appears on the right (A1 at top-right)
        focus_mosaic = np.fliplr(focus_mosaic)
        zoffset_mosaic = np.fliplr(zoffset_mosaic)

        # Apply circular well mask on tile grid
        Yt, Xt = focus_mosaic.shape
        yy, xx = np.ogrid[:Yt, :Xt]
        cy, cx = (Yt - 1) / 2.0, (Xt - 1) / 2.0
        radius = min(cy, cx)
        circle_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
        focus_plot = np.where(circle_mask, focus_mosaic, np.nan)
        zoffset_plot = np.where(circle_mask, zoffset_mosaic, np.nan)

        compiled_entries.append((well, focus_plot, zoffset_plot))

    # Create a single compiled PNG with wells laid out horizontally (rows=[focus, z-offset], cols=wells)
    if compiled_entries:
        n_wells = len(compiled_entries)
        figC, axesC = plt.subplots(
            2, n_wells, figsize=(4 * n_wells, 8), squeeze=False, constrained_layout=True
        )
        for j, (well, fplot, zplot) in enumerate(
            tqdm(compiled_entries, desc="plot wells [mean]", leave=False)
        ):
            axf = axesC[0, j]
            imf = axf.imshow(fplot, cmap="viridis", vmin=vmin_focus, vmax=vmax_focus)
            axf.set_xticks([])
            axf.set_yticks([])
            axf.set_title(well)
            axz = axesC[1, j]
            imz = axz.imshow(zplot, cmap="magma", vmin=vmin_z, vmax=vmax_z)
            axz.set_xticks([])
            axz.set_yticks([])

        # Independent colorbars for each row with consistent scales
        sm_focus = cm.ScalarMappable(
            norm=Normalize(vmin=vmin_focus, vmax=vmax_focus), cmap="viridis"
        )
        sm_focus.set_array([])
        figC.colorbar(
            sm_focus,
            ax=axesC[0, :].ravel().tolist(),
            fraction=0.02,
            pad=0.02,
            label="Focus index (tile mean)",
        )

        sm_z = cm.ScalarMappable(norm=Normalize(vmin=vmin_z, vmax=vmax_z), cmap="magma")
        sm_z.set_array([])
        figC.colorbar(
            sm_z,
            ax=axesC[1, :].ravel().tolist(),
            fraction=0.02,
            pad=0.02,
            label="Z offset (tile mean)",
        )

        figC.suptitle(
            f"{experiment} [{tag_local}] - Tile MEAN mosaics (Focus index, Z offset)"
        )
        compiled_png = (
            recon_dir / f"{experiment}_{tag_local}_wells_compiled_tile_means.png"
        )
        figC.savefig(compiled_png, dpi=220)
        plt.close(figC)
        print(
            f"[MetricsRecon] Spatial mosaics saved in {time.perf_counter() - t0:.1f}s → {compiled_png}"
        )

    # MEDIAN mosaics
    all_focus_medians = []
    all_zoffset_medians = []
    for entries in well_to_entries_median.values():
        for _, _, fmed, zmed in entries:
            all_focus_medians.append(fmed)
            all_zoffset_medians.append(zmed)
    if all_focus_medians:
        vmin_focus_med = float(np.nanmin(all_focus_medians))
        vmax_focus_med = float(np.nanmax(all_focus_medians))
        vmin_z_med = float(np.nanmin(all_zoffset_medians))
        vmax_z_med = float(np.nanmax(all_zoffset_medians))

        compiled_entries_med: list[tuple[str, np.ndarray, np.ndarray]] = []
        for well in tqdm(
            sorted(well_to_entries_median.keys()), desc="build mosaics [median]"
        ):
            entries = well_to_entries_median[well]
            rows = [r for r, _, _, _ in entries]
            cols = [c for _, c, _, _ in entries]
            r0, c0 = min(rows), min(cols)
            r1, c1 = max(rows), max(cols)
            H = r1 - r0 + 1
            W = c1 - c0 + 1
            focus_mosaic = np.full((H, W), np.nan, dtype=float)
            zoffset_mosaic = np.full((H, W), np.nan, dtype=float)

            for r, c, fmed, zmed in entries:
                rr = r - r0
                cc = c - c0
                focus_mosaic[rr, cc] = fmed
                zoffset_mosaic[rr, cc] = zmed

            # Flip horizontally for orientation
            focus_mosaic = np.fliplr(focus_mosaic)
            zoffset_mosaic = np.fliplr(zoffset_mosaic)

            # Apply circular mask
            Yt, Xt = focus_mosaic.shape
            yy, xx = np.ogrid[:Yt, :Xt]
            cy, cx = (Yt - 1) / 2.0, (Xt - 1) / 2.0
            radius = min(cy, cx)
            circle_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
            focus_plot = np.where(circle_mask, focus_mosaic, np.nan)
            zoffset_plot = np.where(circle_mask, zoffset_mosaic, np.nan)

            compiled_entries_med.append((well, focus_plot, zoffset_plot))

        n_wells = len(compiled_entries_med)
        figCm, axesCm = plt.subplots(
            2, n_wells, figsize=(4 * n_wells, 8), squeeze=False, constrained_layout=True
        )
        for j, (well, fplot, zplot) in enumerate(
            tqdm(compiled_entries_med, desc="plot wells [median]", leave=False)
        ):
            axf = axesCm[0, j]
            imf = axf.imshow(
                fplot, cmap="viridis", vmin=vmin_focus_med, vmax=vmax_focus_med
            )
            axf.set_xticks([])
            axf.set_yticks([])
            axf.set_title(well)
            axz = axesCm[1, j]
            imz = axz.imshow(zplot, cmap="magma", vmin=vmin_z_med, vmax=vmax_z_med)
            axz.set_xticks([])
            axz.set_yticks([])

        sm_focus_med = cm.ScalarMappable(
            norm=Normalize(vmin=vmin_focus_med, vmax=vmax_focus_med), cmap="viridis"
        )
        sm_focus_med.set_array([])
        figCm.colorbar(
            sm_focus_med,
            ax=axesCm[0, :].ravel().tolist(),
            fraction=0.02,
            pad=0.02,
            label="Focus index (tile median)",
        )

        sm_z_med = cm.ScalarMappable(
            norm=Normalize(vmin=vmin_z_med, vmax=vmax_z_med), cmap="magma"
        )
        sm_z_med.set_array([])
        figCm.colorbar(
            sm_z_med,
            ax=axesCm[1, :].ravel().tolist(),
            fraction=0.02,
            pad=0.02,
            label="Z offset (tile median)",
        )

        figCm.suptitle(
            f"{experiment} [{tag_local}] - Tile MEDIAN mosaics (Focus index, Z offset)"
        )
        compiled_png_med = (
            recon_dir / f"{experiment}_{tag_local}_wells_compiled_tile_medians.png"
        )
        figCm.savefig(compiled_png_med, dpi=220)
        plt.close(figCm)
        print(f"[MetricsRecon] Spatial mosaics (median) saved → {compiled_png_med}")

    # STD mosaics
    all_focus_stds = []
    all_zoffset_stds = []
    for entries in well_to_entries_std.values():
        for _, _, fstd, zstd in entries:
            all_focus_stds.append(fstd)
            all_zoffset_stds.append(zstd)
    if all_focus_stds:
        vmin_focus_std = float(np.nanmin(all_focus_stds))
        vmax_focus_std = float(np.nanmax(all_focus_stds))
        vmin_z_std = float(np.nanmin(all_zoffset_stds))
        vmax_z_std = float(np.nanmax(all_zoffset_stds))

        compiled_entries_std: list[tuple[str, np.ndarray, np.ndarray]] = []
        for well in tqdm(
            sorted(well_to_entries_std.keys()), desc="build mosaics [std]"
        ):
            entries = well_to_entries_std[well]
            rows = [r for r, _, _, _ in entries]
            cols = [c for _, c, _, _ in entries]
            r0, c0 = min(rows), min(cols)
            r1, c1 = max(rows), max(cols)
            H = r1 - r0 + 1
            W = c1 - c0 + 1
            focus_mosaic = np.full((H, W), np.nan, dtype=float)
            zoffset_mosaic = np.full((H, W), np.nan, dtype=float)

            for r, c, fstd, zstd in entries:
                rr = r - r0
                cc = c - c0
                focus_mosaic[rr, cc] = fstd
                zoffset_mosaic[rr, cc] = zstd

            # Flip horizontally for orientation
            focus_mosaic = np.fliplr(focus_mosaic)
            zoffset_mosaic = np.fliplr(zoffset_mosaic)

            # Apply circular mask
            Yt, Xt = focus_mosaic.shape
            yy, xx = np.ogrid[:Yt, :Xt]
            cy, cx = (Yt - 1) / 2.0, (Xt - 1) / 2.0
            radius = min(cy, cx)
            circle_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
            focus_plot = np.where(circle_mask, focus_mosaic, np.nan)
            zoffset_plot = np.where(circle_mask, zoffset_mosaic, np.nan)

            compiled_entries_std.append((well, focus_plot, zoffset_plot))

        n_wells = len(compiled_entries_std)
        figCs, axesCs = plt.subplots(
            2, n_wells, figsize=(4 * n_wells, 8), squeeze=False, constrained_layout=True
        )
        for j, (well, fplot, zplot) in enumerate(
            tqdm(compiled_entries_std, desc="plot wells [std]", leave=False)
        ):
            axf = axesCs[0, j]
            imf = axf.imshow(
                fplot, cmap="viridis", vmin=vmin_focus_std, vmax=vmax_focus_std
            )
            axf.set_xticks([])
            axf.set_yticks([])
            axf.set_title(well)
            axz = axesCs[1, j]
            imz = axz.imshow(zplot, cmap="magma", vmin=vmin_z_std, vmax=vmax_z_std)
            axz.set_xticks([])
            axz.set_yticks([])

        sm_focus_std = cm.ScalarMappable(
            norm=Normalize(vmin=vmin_focus_std, vmax=vmax_focus_std), cmap="viridis"
        )
        sm_focus_std.set_array([])
        figCs.colorbar(
            sm_focus_std,
            ax=axesCs[0, :].ravel().tolist(),
            fraction=0.02,
            pad=0.02,
            label="Focus index (tile std)",
        )

        sm_z_std = cm.ScalarMappable(
            norm=Normalize(vmin=vmin_z_std, vmax=vmax_z_std), cmap="magma"
        )
        sm_z_std.set_array([])
        figCs.colorbar(
            sm_z_std,
            ax=axesCs[1, :].ravel().tolist(),
            fraction=0.02,
            pad=0.02,
            label="Z offset (tile std)",
        )

        figCs.suptitle(
            f"{experiment} [{tag_local}] - Tile Stdv mosaics (Focus index, Z offset)"
        )
        compiled_png_std = (
            recon_dir / f"{experiment}_{tag_local}_wells_compiled_tile_stds.png"
        )
        figCs.savefig(compiled_png_std, dpi=220)
        plt.close(figCs)
        print(f"[MetricsRecon] Spatial mosaics (std) saved → {compiled_png_std}")


def generate_subtile_heatmaps(
    subtile_results: dict, dataset: OpsDataset, experiment: str, tag: str | None = None
) -> None:
    # Only compute and save the global mean subtile heatmaps (average tile)
    try:
        _save_spatial_well_heatmaps(subtile_results, dataset, experiment, tag=tag)
        _positions_order, focus_grids, zoffset_grids = _compute_subtile_grids(
            subtile_results
        )
        if focus_grids:
            try:
                shapes_equal = (
                    len({g.shape for g in focus_grids}) == 1
                    and len({g.shape for g in zoffset_grids}) == 1
                )
                if not shapes_equal:
                    print(
                        "[SubtileRecon][WARN] Skipping mean heatmaps: subtile grid shapes differ across tiles"
                    )
                    return
                stack_focus = np.stack(focus_grids, axis=0)
                stack_z = np.stack(zoffset_grids, axis=0)
                mean_focus = np.nanmean(stack_focus, axis=0)
                mean_zoffset = np.nanmean(stack_z, axis=0)
                median_focus = np.nanmedian(stack_focus, axis=0)
                median_zoffset = np.nanmedian(stack_z, axis=0)
                std_focus = np.nanstd(stack_focus, axis=0)
                std_zoffset = np.nanstd(stack_z, axis=0)
                _save_mean_heatmaps(
                    dataset, mean_focus, mean_zoffset, experiment, tag=tag
                )
                _save_median_heatmaps(
                    dataset, median_focus, median_zoffset, experiment, tag=tag
                )
                _save_std_heatmaps(dataset, std_focus, std_zoffset, experiment, tag=tag)
            except Exception as _em:
                print(f"[SubtileRecon][WARN] Failed computing mean heatmaps: {_em}")
    except Exception as _e:
        print(f"[SubtileRecon][WARN] Failed generating mean heatmaps: {_e}")


def generate_subtile_heatmaps_from_csv(
    csv_path: str | Path, dataset: OpsDataset, experiment: str, tag: str | None = None
) -> None:
    """Load subtile metadata from CSV and generate all reconstruction heatmaps.

    Expects columns at least: ['position','subtile_id','focus_index','z_focus_offset'].
    Optional columns used if present: ['y_start','y_end','x_start','x_end','z_stack_size','base_offset','reconstruction_success'].
    """
    t0 = time.perf_counter()
    print(f"[MetricsRecon] Generating subtile heatmaps from CSV: {csv_path}")
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[SubtileRecon][WARN] CSV not found: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[SubtileRecon][WARN] Failed to read CSV {csv_path}: {e}")
        return

    required = {"position", "subtile_id", "focus_index", "z_focus_offset"}
    if not required.issubset(df.columns):
        print(
            f"[SubtileRecon][WARN] CSV missing required columns: {sorted(required - set(df.columns))}"
        )
        return

    # Build the in-memory structure expected by generate_subtile_heatmaps
    subtile_results: dict[str, dict] = {}
    for pos, g in df.groupby("position"):
        try:
            ids = g["subtile_id"].dropna().astype(int).tolist()
        except Exception:
            ids = []
        n_sub = len(ids)
        gsize = int(round(np.sqrt(n_sub))) if n_sub > 0 else 0

        meta = []
        for _, row in g.iterrows():
            rec = {
                "subtile_id": (
                    int(row.get("subtile_id", -1))
                    if not pd.isna(row.get("subtile_id", np.nan))
                    else -1
                ),
                "focus_index": (
                    float(row.get("focus_index", np.nan))
                    if "focus_index" in g.columns
                    else np.nan
                ),
                "z_focus_offset": float(row.get("z_focus_offset", np.nan)),
            }
            # Optionally include explicit integer and float focus indices if present in CSV
            if "focus_index_int" in g.columns and not pd.isna(
                row.get("focus_index_int", np.nan)
            ):
                try:
                    rec["focus_index_int"] = int(row.get("focus_index_int"))
                except Exception:
                    pass
            if "focus_index_float" in g.columns and not pd.isna(
                row.get("focus_index_float", np.nan)
            ):
                try:
                    rec["focus_index_float"] = float(row.get("focus_index_float"))
                except Exception:
                    pass
            # Optional extras
            for k in ("y_start", "y_end", "x_start", "x_end"):
                if k in g.columns and not pd.isna(row.get(k, np.nan)):
                    rec.setdefault("bounds", [None, None, None, None])
            if "bounds" in rec:
                rec["bounds"] = (
                    int(
                        row.get("y_start", 0)
                        if not pd.isna(row.get("y_start", np.nan))
                        else 0
                    ),
                    int(
                        row.get("y_end", 0)
                        if not pd.isna(row.get("y_end", np.nan))
                        else 0
                    ),
                    int(
                        row.get("x_start", 0)
                        if not pd.isna(row.get("x_start", np.nan))
                        else 0
                    ),
                    int(
                        row.get("x_end", 0)
                        if not pd.isna(row.get("x_end", np.nan))
                        else 0
                    ),
                )
            if "z_stack_size" in g.columns and not pd.isna(
                row.get("z_stack_size", np.nan)
            ):
                rec["z_stack_size"] = float(row.get("z_stack_size"))
            if "base_offset" in g.columns and not pd.isna(
                row.get("base_offset", np.nan)
            ):
                rec["base_offset"] = float(row.get("base_offset"))
            if "reconstruction_success" in g.columns:
                try:
                    rec["reconstruction_success"] = bool(
                        row.get("reconstruction_success")
                    )
                except Exception:
                    pass
            meta.append(rec)

        subtile_results[pos] = {
            "n_subtiles": n_sub,
            "grid_size": gsize,
            "subtile_metadata": meta,
        }

    generate_subtile_heatmaps(subtile_results, dataset, experiment, tag=tag)
    print(
        f"[MetricsRecon] CSV-based heatmap generation completed in {time.perf_counter() - t0:.1f}s"
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate subtile reconstruction metrics from CSV"
    )
    p.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment name (e.g., ops0060_20250724)",
    )
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to subtile metadata CSV; defaults to <1-preprocess/live_imaging/reconstruction>/<exp>_subtile_metadata.csv",
    )
    p.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Process tag to include in output filenames (e.g., pheno-2d, track-2d, both)",
    )
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    experiment = args.experiment
    dataset = OpsDataset(experiment)
    recon_dir = dataset.preprocess_live / "reconstruction"

    # Support running both process types in one call
    if str(args.tag).lower() == "both":
        for proc_tag in ("track-2d", "pheno-2d"):
            tag_local = proc_tag.replace(" ", "_")
            csv_path = recon_dir / f"{experiment}_{tag_local}_subtile_metadata.csv"
            generate_subtile_heatmaps_from_csv(
                csv_path, dataset, experiment, tag=proc_tag
            )
        return

    # Single run (optional explicit CSV path)
    if args.csv:
        csv_path = Path(args.csv)
    else:
        if args.tag:
            tag_local = str(args.tag).replace(" ", "_")
            csv_path = recon_dir / f"{experiment}_{tag_local}_subtile_metadata.csv"
        else:
            csv_path = recon_dir / f"{experiment}_subtile_metadata.csv"

    generate_subtile_heatmaps_from_csv(csv_path, dataset, experiment, tag=args.tag)


if __name__ == "__main__":
    main()
# usage: python -m cyclops_process.metrics.metrics_reconstruction --experiment ops0063_20250731 --tag track-2d
