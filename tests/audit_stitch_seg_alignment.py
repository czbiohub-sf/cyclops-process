#!/usr/bin/env python3
"""Audit stitch-vs-segmentation alignment across all experiments.

For each experiment, checks whether the stitch config (shifts YAML) was modified
AFTER the segmentation store was created. When this happens, the image was
stitched with newer shifts than the seg, causing a canvas-size mismatch.

Also compares the actual YX shapes of image level-0 vs nuclear_seg level-0
in the stitched stores to detect existing mismatches.

Usage:
    # Text audit (offsets / SHIFTS_AFTER_SEG)
    python tests/audit_stitch_seg_alignment.py
    python tests/audit_stitch_seg_alignment.py --experiment 31

    # Visual overlay: nuclear_seg drawn on the stitched image at center + cardinal
    # edges, for the worst-offset experiments (so alignment can be judged by eye).
    python tests/audit_stitch_seg_alignment.py overlay --top 10
    python tests/audit_stitch_seg_alignment.py overlay -e 138 --stores track
    python tests/audit_stitch_seg_alignment.py overlay --top 10 --slurm     # fan out one job/exp
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "ops_utils" / "src"))
from ops_utils.data.experiment import OpsDataset
from ops_utils.data.filesystem import resolve_experiment_name
from cyclops_process.paths import BASE_PATH


# (process_label, stitch_config_key, image_store_key, seg_store_key)
CHECKS = [
    ("track", "lc_5x_stitch", "lc_5x_phase_2d_stitched", "lc_5x_segmentation"),
    ("track_v3", "lc_5x_stitch", "lc_5x_phase_2d_stitched_v3", "lc_5x_segmentation"),
    ("iss", "iss_stitch", "iss_stitch", "iss_segmentation"),
    ("pheno", "lc_20x_stitch", "pheno_assembled", "lc_20x_segmentation"),
    ("pheno_v3", "lc_20x_stitch", "pheno_assembled_v3", "lc_20x_segmentation"),
]


def _mtime(p: Path) -> float | None:
    try:
        return p.stat().st_mtime
    except (OSError, FileNotFoundError):
        return None


def _get_yx_shape(store_path: Path, pos: str, subpath: str) -> tuple[int, int] | None:
    """Read YX shape from zarr array at store_path/pos/subpath."""
    arr_path = store_path / pos / subpath
    if not arr_path.exists():
        return None
    try:
        arr = zarr.open_array(str(arr_path), mode="r")
        return (arr.shape[-2], arr.shape[-1])
    except Exception:
        return None


def _find_first_position(store_path: Path) -> str | None:
    """Find first A/*/0* position in a zarr store."""
    for row_dir in sorted(store_path.glob("A")):
        if not row_dir.is_dir():
            continue
        for col_dir in sorted(row_dir.iterdir()):
            if not col_dir.is_dir():
                continue
            for fov_dir in sorted(col_dir.iterdir()):
                if not fov_dir.is_dir():
                    continue
                if (fov_dir / "0").exists() or (fov_dir / "0" / "zarr.json").exists():
                    return str(fov_dir.relative_to(store_path))
    return None


def audit_experiment(experiment: str) -> list[dict]:
    """Audit one experiment, return list of findings."""
    try:
        ds = OpsDataset(experiment)
    except Exception as e:
        return [{"experiment": experiment, "error": str(e)}]

    findings = []

    for label, cfg_key, img_key, seg_key in CHECKS:
        cfg_path = Path(ds.config_paths.get(cfg_key, ""))
        img_store = Path(ds.store_paths.get(img_key, ""))
        seg_store = Path(ds.store_paths.get(seg_key, ""))

        if not img_store.exists() or not seg_store.exists():
            continue

        finding = {
            "experiment": experiment,
            "process": label,
            "cfg": str(cfg_path),
            "img_store": str(img_store),
            "seg_store": str(seg_store),
        }

        # 1) Timestamp check: was config modified after seg store?
        cfg_mtime = _mtime(cfg_path)
        # Use the zarr metadata file as proxy for seg creation time
        seg_pos = _find_first_position(seg_store)
        seg_mtime = None
        if seg_pos:
            for meta_name in (".zarray", "zarr.json"):
                seg_mtime = _mtime(seg_store / seg_pos / "0" / meta_name)
                if seg_mtime:
                    break
            # Also check the data chunk dir
            if not seg_mtime:
                seg_mtime = _mtime(seg_store / seg_pos / "0" / "c")

        if cfg_mtime and seg_mtime:
            finding["cfg_mtime"] = cfg_mtime
            finding["seg_mtime"] = seg_mtime
            finding["shifts_after_seg"] = cfg_mtime > seg_mtime
        else:
            finding["shifts_after_seg"] = None

        # 2) Shape check: compare image vs seg YX in stitched store
        img_pos = _find_first_position(img_store)
        if img_pos:
            img_yx = _get_yx_shape(img_store, img_pos, "0")

            # Try both v2 (pos/nuclear_seg/0) and v3 (pos/labels/nuclear_seg/0)
            seg_yx = _get_yx_shape(img_store, img_pos, "nuclear_seg/0")
            if seg_yx is None:
                seg_yx = _get_yx_shape(img_store, img_pos, "labels/nuclear_seg/0")

            finding["img_yx"] = img_yx
            finding["seg_yx"] = seg_yx
            if img_yx and seg_yx:
                finding["yx_match"] = img_yx == seg_yx
                finding["yx_delta"] = (img_yx[0] - seg_yx[0], img_yx[1] - seg_yx[1])
            else:
                finding["yx_match"] = None

        findings.append(finding)

    return findings


def _run_report(experiment):
    """Text audit: print SHIFTS_AFTER_SEG / YX_MISMATCH for one or all experiments."""
    base = Path(BASE_PATH)

    if experiment:
        experiments = [resolve_experiment_name(experiment, autoselect=True)]
    else:
        experiments = sorted([
            d.name for d in base.iterdir()
            if d.is_dir() and d.name.startswith("ops")
        ])

    problems = []
    for exp in experiments:
        findings = audit_experiment(exp)
        for f in findings:
            if f.get("error"):
                continue

            is_problem = f.get("shifts_after_seg") or f.get("yx_match") is False
            if is_problem:
                problems.append(f)

            # Always print if single experiment, otherwise only problems
            if experiment or is_problem:
                status = []
                if f.get("shifts_after_seg"):
                    status.append("SHIFTS_AFTER_SEG")
                if f.get("yx_match") is False:
                    delta = f.get("yx_delta", ("?", "?"))
                    status.append(f"YX_MISMATCH(dY={delta[0]},dX={delta[1]})")
                if not status:
                    status.append("OK")

                print(f"{f['experiment']:30s} {f['process']:12s} {' | '.join(status)}")
                if f.get("img_yx") and f.get("seg_yx"):
                    print(f"{'':30s} {'':12s}   img={f['img_yx']}  seg={f['seg_yx']}")

    if not experiment:
        print(f"\n--- Summary: {len(problems)} problem(s) across {len(experiments)} experiments ---")


# ──────────────────────────────────────────────────────────────────────────
# Visual overlay (nuclear_seg over stitched image, for the worst-offset exps)
# ──────────────────────────────────────────────────────────────────────────

# modality -> ordered audit process labels (prefer v3, the store the viewer reads)
_MODALITY_LABELS = {
    "track": ["track_v3", "track"],
    "iss": ["iss"],
    "pheno": ["pheno_v3", "pheno"],
}

# Crop locations in normalized coords of the shared (min) canvas. The well is a
# disc inscribed in the rectangle, so the corners are empty; sample center + the
# four cardinal edges, where tissue exists and the offset is largest.
_LOCATIONS = [
    ("center", 0.50, 0.50),
    ("top", 0.07, 0.50),
    ("bottom", 0.93, 0.50),
    ("left", 0.50, 0.07),
    ("right", 0.50, 0.93),
]


def collect_modality_findings(experiment, stores):
    """Return {modality: finding} from audit_experiment, preferring v3 over v2."""
    by_label = {f.get("process"): f for f in audit_experiment(experiment) if not f.get("error")}
    out = {}
    for modality in stores:
        for label in _MODALITY_LABELS[modality]:
            f = by_label.get(label)
            if f and f.get("img_yx") and f.get("seg_yx"):
                out[modality] = f
                break
    return out


def worst_offset(findings):
    """Max |dY|,|dX| across a modality-findings dict (0 if none)."""
    best = 0
    for f in findings.values():
        dy, dx = f.get("yx_delta", (0, 0))
        best = max(best, abs(dy), abs(dx))
    return best


def _seg_subpath(pos_dir):
    """nuclear_seg array path for a stitched position (v3 then v2 layout)."""
    for sub in ("labels/nuclear_seg/0", "nuclear_seg/0"):
        if (pos_dir / sub).exists():
            return pos_dir / sub
    return None


def _read_yx_window(arr, y0, y1, x0, x1, ch=0, force_first=False):
    """Read a 2D YX window from (T,C,Z,Y,X)/(T,C,Y,X)/(Y,X). force_first => t0,c0,zmid."""
    nd = arr.ndim
    c = 0 if force_first else ch
    if nd == 5:
        crop = arr[0, c, arr.shape[2] // 2, y0:y1, x0:x1]
    elif nd == 4:
        crop = arr[0, c, y0:y1, x0:x1]
    elif nd == 3:
        crop = arr[arr.shape[0] // 2, y0:y1, x0:x1]
    else:
        crop = arr[y0:y1, x0:x1]
    return np.asarray(crop)


def render_experiment(experiment, findings, output_dir, rank=None, crop_size=512,
                      channel=0, read_workers=5):
    """Render one overlay figure: rows=stores, cols=crop locations (origin->edges)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["pdf.fonttype"] = 42
    from skimage.segmentation import find_boundaries

    modalities = list(findings.keys())
    if not modalities:
        return None

    fig, axes = plt.subplots(len(modalities), len(_LOCATIONS),
                             figsize=(3.2 * len(_LOCATIONS), 3.6 * len(modalities)),
                             squeeze=False)
    half = crop_size // 2
    for ri, modality in enumerate(modalities):
        f = findings[modality]
        img_store = Path(f["img_store"])
        pos = _find_first_position(img_store)
        dy, dx = f.get("yx_delta", (0, 0))

        img_arr = seg_arr = None
        if pos:
            pos_dir = img_store / pos
            try:
                img_arr = zarr.open_array(str(pos_dir / "0"), mode="r")
            except Exception:
                img_arr = None
            seg_path = _seg_subpath(pos_dir)
            if seg_path is not None:
                try:
                    seg_arr = zarr.open_array(str(seg_path), mode="r")
                except Exception:
                    seg_arr = None

        if img_arr is not None and seg_arr is not None:
            miny = min(img_arr.shape[-2], seg_arr.shape[-2])
            minx = min(img_arr.shape[-1], seg_arr.shape[-1])
        elif img_arr is not None:
            miny, minx = img_arr.shape[-2], img_arr.shape[-1]
        else:
            miny = minx = 0

        def _read_loc(ci_loc):
            ci, (_, fy, fx) = ci_loc
            if miny == 0:
                return ci, None, None
            cy, cx = int(fy * miny), int(fx * minx)
            y0, y1 = max(0, cy - half), min(miny, cy + half)
            x0, x1 = max(0, cx - half), min(minx, cx + half)
            if y1 - y0 < 4 or x1 - x0 < 4:
                return ci, None, None
            img_c = seg_c = None
            try:
                if img_arr is not None:
                    img_c = _read_yx_window(img_arr, y0, y1, x0, x1, ch=channel)
                if seg_arr is not None:
                    seg_c = _read_yx_window(seg_arr, y0, y1, x0, x1, force_first=True)
            except Exception:
                pass
            return ci, img_c, seg_c

        img_crops, seg_crops = {}, {}
        with ThreadPoolExecutor(max_workers=read_workers) as ex:
            for ci, img_c, seg_c in ex.map(_read_loc, list(enumerate(_LOCATIONS))):
                if img_c is not None:
                    img_crops[ci] = img_c
                if seg_c is not None:
                    seg_crops[ci] = seg_c

        if img_crops:
            allv = np.concatenate([c.ravel() for c in img_crops.values()])
            vmin, vmax = np.percentile(allv, [1, 99])
        else:
            vmin, vmax = 0, 1

        for ci, (loc_name, _, _) in enumerate(_LOCATIONS):
            ax = axes[ri, ci]
            ax.set_xticks([]); ax.set_yticks([])
            img_crop = img_crops.get(ci)
            seg_crop = seg_crops.get(ci)
            if img_crop is not None:
                disp = np.clip((img_crop.astype(np.float32) - vmin) / max(vmax - vmin, 1e-6), 0, 1)
                ax.imshow(disp, cmap="gray", vmin=0, vmax=1)
            if seg_crop is not None and seg_crop.max() > 0:
                lbl = seg_crop.astype(np.int32)
                mask = lbl > 0
                bound = find_boundaries(lbl, mode="outer")
                overlay = np.zeros((*lbl.shape, 4), dtype=np.float32)
                overlay[mask] = (1.0, 0.25, 0.25, 0.35)   # translucent red fill
                overlay[bound] = (1.0, 0.0, 0.0, 0.95)     # solid red boundary
                ax.imshow(overlay)
            if img_crop is None and seg_crop is None:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                        ha="center", va="center", color="red", fontsize=12)
            if ri == 0:
                ax.set_title(loc_name, fontsize=9)
            if ci == 0:
                bad = (abs(dy) >= 10 or abs(dx) >= 10)
                ax.set_ylabel(
                    f"{modality}\nimg={f['img_yx']}\nseg={f['seg_yx']}\nΔ=(dY={dy},dX={dx})",
                    fontsize=8, color="red" if bad else "black",
                    fontweight="bold" if bad else "normal",
                )

    rank_str = f"#{rank}  " if rank is not None else ""
    fig.suptitle(
        f"{rank_str}{experiment}  —  nuclear_seg (red) over stitched image  |  "
        f"worst Δ={worst_offset(findings)}px\ncrops: origin → edges (channel {channel})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    os.makedirs(output_dir, exist_ok=True)
    name = f"{rank:02d}_{experiment}.png" if rank is not None else f"{experiment}.png"
    out = os.path.join(output_dir, name)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def render_one_job(experiment, stores, output_dir, crop_size=512, channel=0,
                   rank=None, read_workers=5):
    """Self-contained render of one experiment (module-level => picklable for SLURM)."""
    findings = collect_modality_findings(experiment, stores)
    if not findings:
        return {"experiment": experiment, "status": "no_stores"}
    out = render_experiment(experiment, findings, output_dir, rank=rank,
                            crop_size=crop_size, channel=channel, read_workers=read_workers)
    return {"experiment": experiment, "png": out, "worst": worst_offset(findings)}


def _rank_experiments(stores, workers):
    """Scan all experiments (threaded, metadata-only) and rank by worst YX offset."""
    base = Path(BASE_PATH)
    experiments = sorted(d.name for d in base.iterdir() if d.is_dir() and d.name.startswith("ops"))
    print(f"Scanning {len(experiments)} experiments for stitch/seg offsets ({workers} threads)...")

    def _scan(exp):
        f = collect_modality_findings(exp, stores)
        return (worst_offset(f), exp, f) if f else None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        ranked = [r for r in ex.map(_scan, experiments) if r]
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked


def _run_overlay(args):
    output_dir = os.path.abspath(args.output_dir)

    if args.experiment:
        exp = resolve_experiment_name(args.experiment, autoselect=True)
        result = render_one_job(exp, args.stores, output_dir,
                                crop_size=args.crop_size, channel=args.channel,
                                read_workers=args.workers)
        print(f"Saved: {result.get('png')}" if result.get("png") else f"No overlayable stores for {exp}")
        return

    ranked = _rank_experiments(args.stores, args.workers)
    top = ranked[: args.top]
    print(f"\nTop {len(top)} worst-offset experiments:")
    print(f"  {'rank':>4}  {'experiment':<24} {'worstΔ(px)':>10}   per-store Δ(dY,dX)")
    for rank, (w, exp, findings) in enumerate(top, 1):
        deltas = "  ".join(f"{m}={f['yx_delta']}" for m, f in findings.items())
        print(f"  {rank:>4}  {exp:<24} {w:>10}   {deltas}")

    if args.slurm:
        from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
        jobs = [{
            "name": f"seg_overlay_{exp}",
            "func": render_one_job,
            "kwargs": {"experiment": exp, "stores": args.stores, "output_dir": output_dir,
                       "crop_size": args.crop_size, "channel": args.channel,
                       "rank": rank, "read_workers": args.workers},
            "metadata": {"experiment": exp, "worst_px": w},
        } for rank, (w, exp, findings) in enumerate(top, 1)]
        slurm_params = {"timeout_min": 20, "mem": "16G",
                        "cpus_per_task": args.workers, "slurm_partition": "cpu"}
        print(f"\nSubmitting {len(jobs)} SLURM render jobs -> {output_dir}/ ...")
        submit_parallel_jobs(
            jobs_to_submit=jobs, experiment="seg_overlay_audit", slurm_params=slurm_params,
            log_dir="slurm_seg_overlay_logs", manifest_prefix="seg_overlay",
            wait_for_completion=not args.no_wait,
        )
        return

    print(f"\nRendering overlays to {output_dir}/ ...")
    for rank, (w, exp, findings) in enumerate(top, 1):
        out = render_experiment(exp, findings, output_dir, rank=rank,
                                crop_size=args.crop_size, channel=args.channel,
                                read_workers=args.workers)
        print(f"  [{rank}/{len(top)}] {exp} -> {out}")


def main():
    parser = argparse.ArgumentParser(description="Audit stitch/seg alignment")
    parser.add_argument("--experiment", "-e", type=str, default=None,
                        help="Single experiment to audit (e.g. 31)")
    sub = parser.add_subparsers(dest="command")

    ov = sub.add_parser("overlay", help="Render nuclear_seg-over-image overlays for worst-offset exps")
    ov.add_argument("--top", type=int, default=10, help="Number of worst-offset experiments (default: 10)")
    ov.add_argument("-e", "--experiment", type=str, default=None, help="Single experiment, bypassing ranking")
    ov.add_argument("--stores", nargs="+", default=["track", "iss", "pheno"],
                    choices=["track", "iss", "pheno"], help="Stores to overlay (default: all)")
    ov.add_argument("--crop-size", type=int, default=512, help="Crop window size in px (default: 512)")
    ov.add_argument("--channel", type=int, default=0, help="Image channel index (default: 0)")
    ov.add_argument("--output-dir", "-o", default="seg_overlay_audit", help="Output directory")
    ov.add_argument("--workers", type=int, default=8, help="Threads for scan + crop reads (default: 8)")
    ov.add_argument("--slurm", action="store_true", help="Fan out rendering as one SLURM job per experiment")
    ov.add_argument("--no-wait", action="store_true", help="With --slurm, submit and return without waiting")

    args = parser.parse_args()

    if args.command == "overlay":
        _run_overlay(args)
    else:
        _run_report(args.experiment)


if __name__ == "__main__":
    main()
