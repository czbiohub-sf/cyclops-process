"""Estimate stitching parameters and stitch fixed-cell tiles via SLURM.

Step 04 in the unified fixed-cell pipeline (cell painting + 4i). Estimates tile
shifts from overlaps and assembles a stitched mosaic per well, writing the
<stem>_stitch.yaml shifts that step 05 (segment) reuses. Runs before segment so
the shifts exist. Submits one SLURM job per unit (CP "part" / 4i "round").

Usage:
    python -m cyclops_process.fixed_cp_4i.04_stitch --experiment ops0094 --parts 1 2 --channel 1 --overlap 75 --use-clahe --part1-fliplr
    python -m cyclops_process.fixed_cp_4i.04_stitch --modality 4i --experiment ops0144 --overlap 184 --local
    python -m cyclops_process.fixed_cp_4i.04_stitch --experiment ops0094 --parts 1 2 --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import sys
import os

sys.path.insert(0, os.getcwd())

from cyclops_process.processes.ops_stitch import estimate_and_stitch
from cyclops_process.fixed_cp_4i.configs.modality_config import get_modality
from ops_utils.data.experiment import OpsDataset
from ops_utils.data.filesystem import resolve_experiment_name
from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
import yaml
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Stitch assembly (stitch.assemble) is CuPy-accelerated: with a GPU present it
# uses _process_x_block_gpu; on CPU it falls back to a much slower EDT-blend path
# (hours for a 3-well, 2345-tile mosaic). Run on GPU like the pipeline's
# estimate_and_stitch_pheno-2d step so assembly takes ~20-30 min, not hours.
SLURM_PARAMS = {
    "gpus_per_node": 2,
    "slurm_constraint": "[a100_80|h100|h200|6000_blackwell]",
    "timeout_min": 60,
    "mem": "600GB",  # 3-well full-res mosaic assembly peaks high; 400G OOM-killed part1
    "cpus_per_task": 32,
    "slurm_partition": "gpu",
}


def _default_paths(input_store: Path, suffix: str = "") -> tuple[Path, Path, Path]:
    """Derive default config/output paths next to the input store.

    - Config YAML: <stem><suffix>_stitch.yaml
    - Output Zarr: <stem><suffix>_stitched.zarr
    - Confidence plot: <stem><suffix>_stitch_confidence.png
    """
    input_store = Path(input_store)
    cfg = input_store.with_name(f"{input_store.stem}{suffix}_stitch.yaml")
    out = input_store.with_name(f"{input_store.stem}{suffix}_stitched.zarr")
    plot = input_store.with_name(f"{input_store.stem}{suffix}_stitch_confidence.png")
    return cfg, out, plot


def _visualize_stitch_confidence(config_path: Path, output_plot_path: Path) -> None:
    """Generate and save a visualization of stitching confidence scores."""
    try:
        with open(config_path, "r") as file:
            config_dict = yaml.safe_load(file)
    except Exception as e:
        print(f"[WARNING] Could not read config for visualization: {e}")
        return

    if "confidence" not in config_dict:
        print("[WARNING] No confidence data found in config")
        return

    confidence_dict = config_dict["confidence"]

    num_wells = len(confidence_dict.keys())
    if num_wells == 0:
        print("[WARNING] No wells found in confidence data")
        return

    fig, ax = plt.subplots(1, num_wells, figsize=(min(26, 8 * num_wells), 6))
    if num_wells == 1:
        ax = [ax]

    cmap = cm.magma

    for i, well in enumerate(confidence_dict.keys()):
        well_conf = confidence_dict[well]

        ax[i].set_facecolor("black")
        ax[i].set_xticks([])
        ax[i].set_yticks([])

        for key, val in well_conf.items():
            ax[i].plot(
                [val[0][1], val[1][1]],
                [val[0][0], val[1][0]],
                color=cmap(float(val[2])),
                linewidth=8,
            )
        ax[i].set_title(well)
        ax[i].invert_yaxis()

    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Stitching Confidence")
    fig.suptitle(f"Stitching Confidence Map")
    plt.tight_layout()

    try:
        plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved confidence plot: {output_plot_path}")
        plt.close(fig)
    except Exception as e:
        print(f"[WARNING] Could not save confidence plot: {e}")
        plt.close(fig)


def _check_stitch_confidence(config_path: Path, min_confidence: float = 0.8) -> bool:
    """Summarize per-well stitch confidence and warn if below `min_confidence`.

    Returns True if every well's median edge confidence is >= min_confidence.
    """
    try:
        with open(config_path, "r") as f:
            confidence_dict = (yaml.safe_load(f) or {}).get("confidence", {})
    except Exception as e:
        print(f"[confidence] Could not read {config_path}: {e}")
        return False
    if not confidence_dict:
        print(f"[confidence] No confidence data in {config_path.name}")
        return False

    import statistics
    ok = True
    print(f"\n--- Stitch confidence ({config_path.name}, threshold {min_confidence}) ---")
    for well, edges in confidence_dict.items():
        vals = [float(v[2]) for v in edges.values()]
        if not vals:
            continue
        med = statistics.median(vals)
        frac_low = sum(1 for v in vals if v < min_confidence) / len(vals)
        flag = "OK " if med >= min_confidence else "LOW"
        print(f"  [{flag}] well {well}: median={med:.3f} min={min(vals):.3f} "
              f"({100*frac_low:.0f}% of {len(vals)} edges < {min_confidence})")
        if med < min_confidence:
            ok = False
    if not ok:
        print(f"  ⚠️  Median stitch confidence below {min_confidence} — inspect the confidence plot before trusting the mosaic.")
    return ok


def _stitch_store(
    input_store: Path,
    config_path: Path | None,
    output_store: Path | None,
    flipud: bool,
    fliplr: bool,
    rot90: int,
    debug_n_positions: int | None,
    debug_output_suffix: str,
    channel: int = 0,
    overlap: int = 150,
    use_clahe: bool = False,
    clahe_clip_limit: float = 0.02,
    confidence_plot_path: Path | None = None,
    min_confidence: float = 0.8,
) -> None:
    input_store = Path(input_store)
    if not input_store.exists():
        print(f"[ERROR] Input store not found: {input_store}")
        return

    suffix = debug_output_suffix if (debug_n_positions is not None and int(debug_n_positions) > 0) else ""
    cfg, out, plot = _default_paths(input_store, suffix=suffix)
    if config_path is not None:
        cfg = Path(config_path)
    if output_store is not None:
        out = Path(output_store)
    if confidence_plot_path is not None:
        plot = Path(confidence_plot_path)

    print("\n--- Stitching (estimate + assemble) ---")
    print(f"Input tiles store : {input_store}")
    print(f"Shifts YAML       : {cfg}")
    print(f"Output stitched   : {out}")
    print(f"Confidence plot   : {plot}")
    print(f"Transforms        : flipud={flipud}, fliplr={fliplr}, rot90={rot90}")
    print(f"Channel for registration: {channel}")
    print(f"Overlap           : {overlap} pixels")
    if use_clahe:
        print(f"CLAHE preprocessing: enabled (clip_limit={clahe_clip_limit})")
    if debug_n_positions is not None and int(debug_n_positions) > 0:
        print(
            f"Debug mode        : limiting to {debug_n_positions}x{debug_n_positions} grid = {debug_n_positions**2} positions, suffix='{debug_output_suffix}'"
        )

    estimate_and_stitch(
        experiment=None,
        process=None,
        input_store_path=input_store,
        output_config_path=cfg,
        output_store_path=out,
        flipud=flipud,
        fliplr=fliplr,
        rot90=int(rot90) % 4,
        debug_n_positions=debug_n_positions,
        debug_output_suffix=debug_output_suffix,
        channel=channel,
        overlap=overlap,
        use_clahe=use_clahe,
        clahe_clip_limit=clahe_clip_limit,
        verbose=True,
    )

    print("\n--- Generating confidence visualization ---")
    _visualize_stitch_confidence(cfg, plot)
    _check_stitch_confidence(cfg, min_confidence=min_confidence)


def run_stitch_part(
    store_path: str,
    flipud: bool = False,
    fliplr: bool = False,
    rot90: int = 0,
    channel: int = 0,
    overlap: int = 150,
    use_clahe: bool = False,
    clahe_clip_limit: float = 0.02,
    debug_n_positions: int | None = None,
    min_confidence: float = 0.8,
):
    """Run stitching for a single store. Called by SLURM or locally."""
    _stitch_store(
        input_store=Path(store_path),
        config_path=None,
        output_store=None,
        flipud=flipud,
        fliplr=fliplr,
        rot90=rot90,
        debug_n_positions=debug_n_positions,
        debug_output_suffix="_debug",
        channel=channel,
        overlap=overlap,
        use_clahe=use_clahe,
        clahe_clip_limit=clahe_clip_limit,
        min_confidence=min_confidence,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Stitch fixed-cell (CP/4i) tiles via SLURM (one job per unit)",
    )
    parser.add_argument("--modality", choices=["cp", "4i"], default="cp", help="Imaging modality (default: cp)")
    parser.add_argument("--experiment", required=True, help="Experiment name (e.g., ops0094)")
    parser.add_argument("--parts", nargs="+", type=int, default=None, help="Units to stitch (default: modality default parts/rounds)")
    parser.add_argument("--channel", type=int, default=0, help="Channel for registration (default: 0)")
    parser.add_argument("--overlap", type=int, default=150, help="Tile overlap in pixels (default: 150)")
    parser.add_argument("--rot90", type=int, default=0, help="Rotate tiles by 90 degrees k times (default: 0)")
    parser.add_argument("--flipud", action="store_true", help="Flip all parts vertically")
    parser.add_argument("--fliplr", action="store_true", help="Flip all parts horizontally")
    parser.add_argument("--part1-flipud", action="store_true", help="Flip part 1 vertically only")
    parser.add_argument("--part1-fliplr", action="store_true", help="Flip part 1 horizontally only")
    parser.add_argument("--part2-flipud", action="store_true", help="Flip part 2 vertically only")
    parser.add_argument("--part2-fliplr", action="store_true", help="Flip part 2 horizontally only")
    parser.add_argument("--use-clahe", action="store_true", help="CLAHE preprocessing for registration")
    parser.add_argument("--clahe-clip-limit", type=float, default=0.02, help="CLAHE clip limit (default: 0.02)")
    parser.add_argument("--debug-n-positions", type=int, default=None, help="Debug: stitch NxN grid only")
    parser.add_argument("--min-confidence", type=float, default=0.8, help="Warn if a well's median stitch confidence is below this (default: 0.8)")
    parser.add_argument("--dry-run", action="store_true", help="Print SLURM plan without submitting")
    parser.add_argument("--local", action="store_true", help="Run locally instead of via SLURM")

    args = parser.parse_args(argv)
    args.experiment = resolve_experiment_name(args.experiment, autoselect=True)

    m = get_modality(args.modality)
    units = args.parts if args.parts is not None else m.default_units

    dataset = OpsDataset(args.experiment)
    convert_dir = m.convert_dir(args.experiment)

    jobs = []
    for part in units:
        # Look for flatfield-corrected store first
        store_path = convert_dir / f"{m.unit_stem(part)}_max_proj_flatfield.zarr"
        if not store_path.exists():
            store_path = convert_dir / f"{m.unit_stem(part)}_max_proj.zarr"
        if not store_path.exists():
            print(f"WARNING: No store found for {m.unit_word} {part} in {convert_dir}, skipping")
            continue

        # Resolve per-part flips: global flags OR part-specific flags
        flipud = args.flipud or (part == 1 and args.part1_flipud) or (part == 2 and args.part2_flipud)
        fliplr = args.fliplr or (part == 1 and args.part1_fliplr) or (part == 2 and args.part2_fliplr)

        jobs.append({
            "name": f"stitch_{m.name}_{m.unit_stem(part)}",
            "func": run_stitch_part,
            "kwargs": {
                "store_path": str(store_path),
                "flipud": flipud,
                "fliplr": fliplr,
                "rot90": args.rot90,
                "channel": args.channel,
                "overlap": args.overlap,
                "use_clahe": args.use_clahe,
                "clahe_clip_limit": args.clahe_clip_limit,
                "debug_n_positions": args.debug_n_positions,
                "min_confidence": args.min_confidence,
            },
            "metadata": {"type": f"stitch_{m.name}", "part": part},
        })

    if not jobs:
        print("No parts to process.")
        return

    if args.local:
        for job in jobs:
            print(f"\n{'='*60}")
            job["func"](**job["kwargs"])
    else:
        result = submit_parallel_jobs(
            jobs_to_submit=jobs,
            experiment=args.experiment,
            slurm_params=SLURM_PARAMS,
            log_dir=dataset.experiment_path / "slurm_logs" / f"stitch_{m.name}",
            manifest_prefix=f"stitch_{m.name}",
            dry_run=args.dry_run,
        )
        # Propagate failure so the orchestrator halts instead of running segment
        # on a partial/corrupt stitch (a part can OOM even when others succeed).
        if isinstance(result, dict) and result.get("failed"):
            print(f"\n❌ {len(result['failed'])} stitch job(s) failed: {result['failed']}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
