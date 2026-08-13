"""
Per-position SLURM submitter for the native-20x nuclei segmentation pass.

Mirrors the structure of `cell_segmentation_orchestrator.submit_cell_segmentation_jobs`
but invokes `nuclei_pass.segment_nuclei_single_position` per position. Wired
into the pipeline via `submit_nuclei_segmentation_jobs` in `step_registry.py`.

Replaces the legacy `segment_and_stitch_pheno` step (which segments at 5x):
this pass runs at native 20x with diameter=150 (H100-bench sweet spot), gives
biology-equivalent masks at ±1.1% count + mean IoU 0.83, and runs ~70% faster
per physical area.

Usage
-----
    # Via orchestrator (DAG):
    python run.py --slurm-steps --dag --rerun submit_nuclei_segmentation_jobs

    # Direct, single experiment:
    python -m cyclops_process.processes.cell_seg.nuclei_segmentation_orchestrator \\
        --experiment ops0094_20251217 --position A/1/0
"""

from __future__ import annotations

# IMPORTANT: nuclei_pass needs to be imported FIRST so its module-top
# `os.environ.pop("CUDA_VISIBLE_DEVICES")` runs before anything else in this
# process touches cupy/torch. See nuclei_pass.py header for the rationale.
from cyclops_process.processes.cell_seg.nuclei_pass import (
    segment_nuclei_single_position,
    DEFAULT_NUCLEI_DIAMETER,
    DEFAULT_NUCLEI_LABEL,
    DEFAULT_TILE_SIZE,
    DEFAULT_OVERLAP,
    DEFAULT_FLOW_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_NUCLEI_CHANNEL,
)

import argparse
import os
from pathlib import Path

from ops_utils.data.experiment import OpsDataset
from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs


def _segment_nuclei_position_chunk(positions: list[str], **per_pos_kwargs) -> list:
    """Run ``segment_nuclei_single_position`` sequentially for each position
    in the chunk. Returns the list of per-position results.

    Used by ``submit_nuclei_segmentation_jobs`` when ``chunk_size > 1`` so a
    single SLURM task processes multiple FOVs and amortises the per-task
    setup (slurm scheduling, worker import, model load) across them.
    """
    results = []
    for pos in positions:
        results.append(
            segment_nuclei_single_position(position=pos, **per_pos_kwargs)
        )
    return results


def submit_nuclei_segmentation_jobs(
    experiment: str,
    positions: list[str] = None,
    slurm_params: dict = None,
    dry_run: bool = False,
    wait_for_completion: bool = True,
    verbose: bool = True,
    force: bool = False,
    chunk_size: int = 1,
    # Segmentation parameters
    tile_size: int = DEFAULT_TILE_SIZE,
    tile_overlap: int = DEFAULT_OVERLAP,
    diameter: float = DEFAULT_NUCLEI_DIAMETER,
    flow_threshold: float = DEFAULT_FLOW_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    nuclei_channel_name: str = DEFAULT_NUCLEI_CHANNEL,
    output_label_name: str = DEFAULT_NUCLEI_LABEL,
    store_override: str | None = None,
) -> dict:
    """Submit per-position SLURM jobs for the native-20x nuclei pass.

    Parameters mirror `submit_cell_segmentation_jobs`. Defaults match the
    H100 bench sweet spot (diameter=150, 4096² tiles, 512 overlap).
    """
    # Position discovery — reuse the cell_seg path picker but check for OUR
    # output label so we skip positions already nuclei-segmented (not cells).
    from cyclops_process.processes.cell_seg.cell_segmentation_orchestrator import (
        get_available_positions,
    )
    info = get_available_positions(
        experiment, skip_existing=not force, output_label_name=output_label_name,
    )

    pos_list = positions if positions else info["positions"]
    all_positions = info["positions"] + info["skipped_positions"]
    invalid = [p for p in pos_list if p not in all_positions]
    if invalid:
        print(f"  Warning: invalid positions ignored: {invalid}")
        pos_list = [p for p in pos_list if p in all_positions]

    if not pos_list:
        skipped = info.get("skipped_positions", [])
        if skipped:
            print(
                f"All positions already have '{output_label_name}' for {experiment} "
                f"({len(skipped)} positions). Use --force to overwrite."
            )
            return {"success": True, "skipped": True, "n_skipped": len(skipped)}
        raise RuntimeError(
            f"Nuclei segmentation: no positions to process for {experiment}"
        )

    # SLURM resources — lighter than cell seg because the nuclei pass uses
    # less VRAM per worker (~2 GB vs ~13 GB) and is GPU-cheaper per tile.
    default_slurm = {
        "timeout_min": 30,            # nuclei pass benchmarked 7m40s on 4 H100s
        "mem": "256G",                # 44 GB canvas + workers + headroom
        "cpus_per_task": 32,
        "gpus_per_node": 2,           # match cell_seg's allocation; can drop to 1 later
        "slurm_partition": "gpu",
        "slurm_constraint": "[h100|h200]",
    }
    if slurm_params:
        default_slurm.update(slurm_params)

    # Build job list — one SLURM task per chunk of `chunk_size` positions.
    # chunk_size=1 preserves the original "one job per FOV" shape; bigger
    # chunk_size amortises the per-task setup tax across multiple FOVs.
    chunk_size = max(1, int(chunk_size))
    chunks = [pos_list[i:i + chunk_size]
              for i in range(0, len(pos_list), chunk_size)]
    common_kwargs = dict(
        experiment=experiment,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        diameter=diameter,
        flow_threshold=flow_threshold,
        iou_threshold=iou_threshold,
        nuclei_channel_name=nuclei_channel_name,
        output_label_name=output_label_name,
        use_parallel=True,
        store_override=store_override,
        preview_full=False,
    )

    jobs = []
    for chunk in chunks:
        first_safe = chunk[0].replace("/", "_")
        last_safe = chunk[-1].replace("/", "_")
        if len(chunk) == 1:
            name = f"nucseg_{first_safe}"
            func = segment_nuclei_single_position
            kwargs = {"position": chunk[0], **common_kwargs}
        else:
            name = f"nucseg_{first_safe}_to_{last_safe}_n{len(chunk)}"
            func = _segment_nuclei_position_chunk
            kwargs = {"positions": chunk, **common_kwargs}
        jobs.append({
            "name": name,
            "func": func,
            "kwargs": kwargs,
            "metadata": {
                "experiment": experiment,
                "positions": chunk,
                "output_label": output_label_name,
            },
            "slurm_params": default_slurm,
        })

    print(f"\n{'='*60}")
    print(f"Nuclei (20x) Segmentation Batch")
    print(f"{'='*60}")
    print(f"Experiment:    {experiment}")
    print(f"Output label:  {output_label_name}")
    print(f"Positions:     {len(pos_list)}")
    if len(pos_list) <= 6:
        for p in pos_list:
            print(f"  - {p}")
    else:
        for p in pos_list[:3]:
            print(f"  - {p}")
        print(f"  ... and {len(pos_list) - 3} more")
    print(f"Diameter:      {diameter}")
    print(f"Tile/overlap:  {tile_size}/{tile_overlap}")
    print(f"GPUs/job:      {default_slurm['gpus_per_node']}  "
          f"({default_slurm['slurm_constraint']})")
    print(f"{'='*60}\n")

    result = submit_parallel_jobs(
        jobs_to_submit=jobs,
        experiment=f"{experiment}_nuclei_segmentation",
        slurm_params=default_slurm,
        log_dir=f"slurm_nuclei_seg_logs/{experiment}",
        manifest_prefix="nuclear_seg",
        step_name='submit_nuclei_segmentation_jobs',
        dry_run=dry_run,
        wait_for_completion=False,
        verbose=verbose,
        post_completion_callback=None,
    )

    if wait_for_completion and not dry_run:
        from ops_utils.hpc.slurm_batch_utils import wait_for_multiple_job_arrays
        if result.get("success") and "submitted_jobs" in result:
            wait_for_multiple_job_arrays(
                job_arrays=[{
                    "submitted_jobs": result["submitted_jobs"],
                    "base_job_id": result["base_job_id"],
                    "label": f"Nuclei Segmentation ({result['base_job_id']})",
                    "slurm_params": default_slurm,
                }],
                experiment=experiment,
                verbose=verbose,
            )

    return result


# ── Nextflow fan-out (setup → per-position job) ──────────────────────────────
# These replace the phantom-slurm `submit_nuclei_segmentation_jobs` wrapper when
# driven from Nextflow: the setup task enumerates positions and the job task
# segments ONE position with no nested slurm (README "Porting Functionality" 3a).
# The submitit path above is kept for the PipelineRunner / direct CLI use.

_NUCSEG_SENTINEL = "NUCSEG_POS "


def nuclei_segmentation_setup(
    experiment: str,
    force: bool = False,
    output_label_name: str = DEFAULT_NUCLEI_LABEL,
) -> None:
    """Nextflow fan-out setup: print each position needing nuclei segmentation,
    one per line prefixed with the ``NUCSEG_POS`` sentinel so the caller can
    parse them out of any stdout noise (import logs, banners). Prints nothing
    when every position already has the label — the workflow then no-ops."""
    from cyclops_process.processes.cell_seg.cell_segmentation_slurm import (
        get_available_positions,
    )
    info = get_available_positions(
        experiment, skip_existing=not force, output_label_name=output_label_name,
    )
    for pos in info["positions"]:
        print(f"{_NUCSEG_SENTINEL}{pos}")


def nuclei_segmentation_job(
    experiment: str,
    position: str,
    tile_size: int = DEFAULT_TILE_SIZE,
    tile_overlap: int = DEFAULT_OVERLAP,
    diameter: float = DEFAULT_NUCLEI_DIAMETER,
    flow_threshold: float = DEFAULT_FLOW_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    nuclei_channel_name: str = DEFAULT_NUCLEI_CHANNEL,
    output_label_name: str = DEFAULT_NUCLEI_LABEL,
    store_override: str | None = None,
):
    """Nextflow fan-out worker: segment nuclei for ONE position, no nested slurm.
    Thin wrapper over ``segment_nuclei_single_position`` with the batch defaults;
    Nextflow owns the per-position parallelism."""
    return segment_nuclei_single_position(
        experiment=experiment,
        position=position,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        diameter=diameter,
        flow_threshold=flow_threshold,
        iou_threshold=iou_threshold,
        nuclei_channel_name=nuclei_channel_name,
        output_label_name=output_label_name,
        use_parallel=True,
        store_override=store_override,
        preview_full=False,
    )


def main():
    p = argparse.ArgumentParser(description="20x nuclei segmentation — SLURM batch")
    p.add_argument("--experiment", required=True)
    p.add_argument("--position", action="append", default=None,
                   help="Repeatable; specific position(s) to process.")
    p.add_argument("--diameter", type=float, default=DEFAULT_NUCLEI_DIAMETER)
    p.add_argument("--flow-threshold", type=float, default=DEFAULT_FLOW_THRESHOLD,
                   help="Cellpose flow_threshold (lower=stricter; default 0.7)")
    p.add_argument("--output-label", default=DEFAULT_NUCLEI_LABEL)
    p.add_argument("--nuclei-channel", default=DEFAULT_NUCLEI_CHANNEL,
                   help=f"Channel name to segment (default {DEFAULT_NUCLEI_CHANNEL}); "
                        "e.g. CP1_nuclei_Hoechst to segment a cell-painting part in the v3 store")
    p.add_argument("--store", default=None, help="Override source zarr path")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing label if present")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-wait", action="store_true", help="Don't wait for completion")
    args = p.parse_args()

    res = submit_nuclei_segmentation_jobs(
        experiment=args.experiment,
        positions=args.position,
        diameter=args.diameter,
        flow_threshold=args.flow_threshold,
        output_label_name=args.output_label,
        nuclei_channel_name=args.nuclei_channel,
        store_override=args.store,
        force=args.force,
        dry_run=args.dry_run,
        wait_for_completion=not args.no_wait,
    )
    if not res.get("success"):
        print(f"FAILED: {res.get('error', res)}")
        raise SystemExit(1)
    print("DONE.")


if __name__ == "__main__":
    main()
