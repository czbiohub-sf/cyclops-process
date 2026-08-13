"""
SLURM batch submission for cell segmentation.

Submits segmentation jobs for all positions as parallel SLURM jobs.
Each job processes one position with GPU-accelerated tile segmentation.

Usage:
------
# Process ALL experiments that have phenotyping_v3.zarr but no cell_seg yet
python -m cyclops_process.processes.cell_seg.cell_segmentation_orchestrator --all

# Force reprocess ALL experiments (even those with existing cell_seg)
python -m cyclops_process.processes.cell_seg.cell_segmentation_orchestrator --all --force

# Process specific experiments (supports shorthand like "103" or "ops103")
python -m cyclops_process.processes.cell_seg.cell_segmentation_orchestrator -e 103 69

# List available positions for an experiment
python -m cyclops_process.processes.cell_seg.cell_segmentation_orchestrator \
    --experiment ops0033_20250429 --list-positions

# Submit segmentation for all positions in a single experiment
python -m cyclops_process.processes.cell_seg.cell_segmentation_orchestrator \
    --experiment ops0033_20250429

# Submit segmentation for specific positions
python -m cyclops_process.processes.cell_seg.cell_segmentation_orchestrator  --experiment 58 --positions A/1/0

# Preview what would be submitted (dry run)
python -m cyclops_process.processes.cell_seg.cell_segmentation_orchestrator \
    --experiment ops0033_20250429 --dry-run

# Submit without waiting for completion
python -m cyclops_process.processes.cell_seg.cell_segmentation_orchestrator \
    --experiment ops0033_20250429 --no-wait

# Force reprocess even if cell_seg label already exists
python -m cyclops_process.processes.cell_seg.cell_segmentation_orchestrator -e 33 --force

Cell Painting Mode:
-------------------
For experiments with cell_painting configuration (e.g., ops0094), use --cell-paint
to segment cells using the actin channel (CP1_f_actin_Phalloidin) instead of
virtual staining membrane_prediction. Results are stored to 'cp_cell_seg' label.

# Cell painting segmentation for a single experiment
python -m cyclops_process.processes.cell_seg.cell_segmentation_orchestrator \
    --experiment ops0094 --cell-paint

# Cell painting segmentation for all experiments with --all
python -m cyclops_process.processes.cell_seg.cell_segmentation_orchestrator \
    --all --cell-paint

# Custom output label name
python -m cyclops_process.processes.cell_seg.cell_segmentation_orchestrator \
    --experiment ops0094 --cell-paint --output-label my_custom_label

Preview Mode:
-------------
Preview mode runs segmentation LOCALLY (no SLURM) on a small 2x2 tile grid
to test segmentation and stitching. Use the main cell_segmentation.py for preview:

python -m cyclops_process.processes.cell_seg.cell_segmentation \
    --experiment ops0033_20250429 --position A/1/0 --preview
"""

import argparse
import sys
from pathlib import Path
import os

# Add project root to path
sys.path.insert(0, os.getcwd())

from cyclops_process.processes.cell_seg.cell_segmentation import (
    segment_single_position,
    DEFAULT_TILE_SIZE,
    DEFAULT_OVERLAP,
    DEFAULT_DIAMETER,
    DEFAULT_FLOW_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_NUCLEI_CHANNEL,
    DEFAULT_MEMBRANE_CHANNEL,
    _get_channel_indices,
)
from cyclops_utils.hpc.slurm_batch_utils import (
    submit_parallel_jobs,
    detect_experiments_needing_processing,
)
from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.data.filesystem import resolve_experiment_name

# =============================================================================
# Cell Painting Mode Configuration
# =============================================================================

# Default channels for cell painting segmentation
# Uses actin-only for CLAHE preprocessing (best results from sweep testing)
CP_NUCLEI_CHANNEL = "CP1_f_actin_Phalloidin"  # Actin channel (same as membrane for actin-only mode)
CP_MEMBRANE_CHANNEL = "CP1_f_actin_Phalloidin"  # Actin channel for cell boundaries
CP_OUTPUT_LABEL = "cp_cell_seg"  # Output label group for cell painting segmentation

# Default channels for 4i segmentation
# Uses b-catenin (round 4 cell-junction marker) for cell boundaries + DAPI for nuclei.
# NOTE: Before the R1/R3/R4/R5 488/647 slot-order correction, this pointed at
# "4i_R4_p21" — but that label was on the 488-captured slot, which actually
# contained the b-catenin (mouse-488) image. That's what the sweep picked for
# cell-boundary quality, so post-correction we point at "4i_R4_b-catenin" to
# keep using the same underlying imagery (now correctly labeled).
FOUR_I_NUCLEI_CHANNEL = "4i_R4_nuclei_DAPI"
FOUR_I_MEMBRANE_CHANNEL = "4i_R4_b-catenin"
FOUR_I_OUTPUT_LABEL = "4i_cell_seg"


# =============================================================================
# QC Check functions
# =============================================================================


def check_channels_have_data(
    experiment: str,
    position: str = None,
    nuclei_channel: str = DEFAULT_NUCLEI_CHANNEL,
    membrane_channel: str = DEFAULT_MEMBRANE_CHANNEL,
    sample_size: int = 1024,
    min_nonzero_fraction: float = 0.001,
) -> dict:
    """
    Check if nuclei/membrane prediction channels have actual data (not blank).

    Samples multiple regions from within the central 75% of a position to check
    if they contain non-zero values. If ANY region has blank membrane data, the
    QC check fails.

    Args:
        experiment: Experiment name
        position: Specific position to check (e.g., "A/1/0"). If None, uses first position.
        nuclei_channel: Name of nuclei channel (default: "nuclei_prediction")
        membrane_channel: Name of membrane channel (default: "membrane_prediction")
        sample_size: Size of region to sample (default: 1024x1024)
        min_nonzero_fraction: Minimum fraction of non-zero pixels to pass QC (default: 0.1%)

    Returns:
        dict with keys:
            - passed: bool, True if QC passed (ALL sampled regions have data)
            - nuclei_ok: bool, True if all regions have nuclei data
            - membrane_ok: bool, True if all regions have membrane data
            - nuclei_nonzero_frac: float, average fraction of non-zero pixels in nuclei
            - membrane_nonzero_frac: float, average fraction of non-zero pixels in membrane
            - message: str, human-readable status message
            - position_checked: str, which position was checked
            - regions_checked: int, number of regions sampled
            - failed_regions: int, number of regions that failed QC
    """
    import numpy as np
    from iohub import open_ome_zarr

    dataset = OpsDataset(experiment)
    store_path = dataset.store_paths.get("pheno_assembled_v3")

    if store_path is None or not store_path.exists():
        return {
            "passed": False,
            "nuclei_ok": False,
            "membrane_ok": False,
            "nuclei_nonzero_frac": 0.0,
            "membrane_nonzero_frac": 0.0,
            "message": f"phenotyping_v3.zarr not found at {store_path}",
            "position_checked": None,
            "regions_checked": 0,
            "failed_regions": 0,
        }

    try:
        # Get channel indices
        channel_names, nuclei_idx, membrane_idx = _get_channel_indices(
            store_path, nuclei_channel, membrane_channel
        )
    except ValueError as e:
        return {
            "passed": False,
            "nuclei_ok": False,
            "membrane_ok": False,
            "nuclei_nonzero_frac": 0.0,
            "membrane_nonzero_frac": 0.0,
            "message": str(e),
            "position_checked": None,
            "regions_checked": 0,
            "failed_regions": 0,
        }

    # Open store and sample multiple regions from specified or first position
    with open_ome_zarr(str(store_path), mode="r") as ds:
        all_positions = [p for p, _ in ds.positions()]
        if not all_positions:
            return {
                "passed": False,
                "nuclei_ok": False,
                "membrane_ok": False,
                "nuclei_nonzero_frac": 0.0,
                "membrane_nonzero_frac": 0.0,
                "message": "No positions found in store",
                "position_checked": None,
                "regions_checked": 0,
                "failed_regions": 0,
            }

        # Use specified position or default to first
        if position is None:
            position = all_positions[0]
        elif position not in all_positions:
            return {
                "passed": False,
                "nuclei_ok": False,
                "membrane_ok": False,
                "nuclei_nonzero_frac": 0.0,
                "membrane_nonzero_frac": 0.0,
                "message": f"Position '{position}' not found in store",
                "position_checked": position,
                "regions_checked": 0,
                "failed_regions": 0,
            }

        pos_data = ds[position]["0"]
        shape = pos_data.shape  # (T, C, Z, Y, X)
        _, _, _, h, w = shape

        # Define the central 75% region
        margin_y = int(h * 0.125)  # 12.5% margin on each side = 75% center
        margin_x = int(w * 0.125)
        center_y_min = margin_y
        center_y_max = h - margin_y
        center_x_min = margin_x
        center_x_max = w - margin_x

        # Sample 5 regions within the central 75%: center and 4 quadrant centers
        half_sample = sample_size // 2
        center_h = (center_y_min + center_y_max) // 2
        center_w = (center_x_min + center_x_max) // 2
        quad_h = (center_y_max - center_y_min) // 4
        quad_w = (center_x_max - center_x_min) // 4

        regions = [
            (center_h, center_w),  # center
            (center_h - quad_h, center_w - quad_w),  # upper-left quadrant
            (center_h - quad_h, center_w + quad_w),  # upper-right quadrant
            (center_h + quad_h, center_w - quad_w),  # lower-left quadrant
            (center_h + quad_h, center_w + quad_w),  # lower-right quadrant
        ]

        nuclei_fracs = []
        membrane_fracs = []
        n_failed = 0
        n_checked = 0

        for y_center, x_center in regions:
            y_start = max(0, y_center - half_sample)
            x_start = max(0, x_center - half_sample)
            y_end = min(h, y_start + sample_size)
            x_end = min(w, x_start + sample_size)

            # Skip if region is too small
            if (y_end - y_start) < sample_size // 2 or (x_end - x_start) < sample_size // 2:
                continue

            n_checked += 1

            # Read nuclei channel sample
            nuclei_sample = np.asarray(
                pos_data[0, nuclei_idx, 0, y_start:y_end, x_start:x_end]
            )
            nuclei_nonzero = np.count_nonzero(nuclei_sample)
            nuclei_total = nuclei_sample.size
            nuclei_frac = nuclei_nonzero / nuclei_total if nuclei_total > 0 else 0.0

            # Read membrane channel sample
            membrane_sample = np.asarray(
                pos_data[0, membrane_idx, 0, y_start:y_end, x_start:x_end]
            )
            membrane_nonzero = np.count_nonzero(membrane_sample)
            membrane_total = membrane_sample.size
            membrane_frac = membrane_nonzero / membrane_total if membrane_total > 0 else 0.0

            nuclei_fracs.append(nuclei_frac)
            membrane_fracs.append(membrane_frac)

            # If this region has blank membrane or nuclei, count it as failed
            if membrane_frac < min_nonzero_fraction or nuclei_frac < min_nonzero_fraction:
                n_failed += 1

    # Calculate averages
    avg_nuclei_frac = sum(nuclei_fracs) / len(nuclei_fracs) if nuclei_fracs else 0.0
    avg_membrane_frac = sum(membrane_fracs) / len(membrane_fracs) if membrane_fracs else 0.0

    # FAIL if ANY region has blank membrane OR nuclei
    nuclei_ok = all(f >= min_nonzero_fraction for f in nuclei_fracs)
    membrane_ok = all(f >= min_nonzero_fraction for f in membrane_fracs)

    # Both membrane and nuclei are required
    passed = membrane_ok and nuclei_ok

    if passed:
        message = (
            f"QC passed ({n_checked} regions): "
            f"membrane={avg_membrane_frac:.2%}, nuclei={avg_nuclei_frac:.2%}"
        )
    else:
        failed_channels = []
        if not membrane_ok:
            failed_channels.append("membrane")
        if not nuclei_ok:
            failed_channels.append("nuclei")
        message = (
            f"QC FAILED: {n_failed}/{n_checked} regions "
            f"have blank {' and '.join(failed_channels)} channel(s)"
        )

    return {
        "passed": passed,
        "nuclei_ok": nuclei_ok,
        "membrane_ok": membrane_ok,
        "nuclei_nonzero_frac": avg_nuclei_frac,
        "membrane_nonzero_frac": avg_membrane_frac,
        "message": message,
        "position_checked": position,
        "regions_checked": n_checked,
        "failed_regions": n_failed,
    }


# =============================================================================
# Detection functions for --all mode
# =============================================================================


def _check_cell_seg_input(dataset: OpsDataset) -> bool:
    """Check if experiment has phenotyping_v3.zarr (input for cell segmentation)."""
    store_path = dataset.store_paths.get("pheno_assembled_v3")
    return store_path is not None and store_path.exists()


def _get_cell_seg_outputs(dataset: OpsDataset, wells) -> list[Path]:
    """
    Get expected cell_seg output paths.

    Returns paths to cell_seg labels in phenotyping_v3.zarr.
    The caller checks .exists() to determine completion status.
    """
    store_path = dataset.store_paths.get("pheno_assembled_v3")
    if not store_path or not store_path.exists():
        return []

    # Check for cell_seg in first position's labels
    try:
        import zarr
        store = zarr.open(str(store_path), mode="r")

        # Find first position
        for row in sorted(store_path.iterdir()):
            if row.is_dir() and row.name.isalpha():
                for col in sorted(row.iterdir()):
                    if col.is_dir() and col.name.isdigit():
                        for fov in sorted(col.iterdir()):
                            if fov.is_dir() and fov.name.isdigit():
                                pos = f"{row.name}/{col.name}/{fov.name}"
                                cell_seg_path = store_path / pos / "labels" / "cell_seg"
                                return [cell_seg_path]
    except Exception:
        pass

    return []


def detect_experiments_for_cell_seg(
    force: bool = False,
    verbose: bool = True,
) -> tuple[list[tuple], list[tuple]]:
    """
    Detect experiments needing cell segmentation.

    Args:
        force: If True, include all experiments with input (ignore existing outputs)
        verbose: Print progress

    Returns:
        (experiments_to_process, experiments_completed)
        Each item is (experiment_name, n_done, n_total, metadata)
    """
    # Fallback full-unit wells; per-experiment fan-out uses wells_to_process later.
    default_wells = ["A/1/0", "A/2/0", "A/3/0"]
    return detect_experiments_needing_processing(
        input_checker=_check_cell_seg_input,
        output_checker=_get_cell_seg_outputs,
        wells=default_wells,
        force=force,
        verbose=verbose,
    )


def get_available_positions(
    experiment: str,
    skip_existing: bool = True,
    output_label_name: str = "cell_seg",
) -> dict:
    """
    Get available positions for cell segmentation.

    Parameters
    ----------
    experiment : str
        Experiment name
    skip_existing : bool
        If True, skip positions that already have the output label
    output_label_name : str
        Name of output label to check for (default: "cell_seg", use "cp_cell_seg" for cell painting)

    Returns
    -------
    dict with keys:
        - positions: list of position paths
        - skipped_positions: list of positions with existing label
        - source_path: Path to phenotyping_v3.zarr
    """
    dataset = OpsDataset(experiment)
    source_path = dataset.store_paths.get("pheno_assembled_v3")

    if source_path is None or not source_path.exists():
        raise FileNotFoundError(
            f"phenotyping_v3.zarr not found for {experiment}. "
            f"Expected at: {dataset.store_paths.get('pheno_assembled_v3')}"
        )

    # Get all positions from zarr store, filtered to experiment wells
    from iohub import open_ome_zarr
    from cyclops_utils.data.filesystem import get_experiment_wells

    exp_wells = get_experiment_wells(experiment, prefix_only=True)

    with open_ome_zarr(str(source_path), mode="r") as ds:
        all_positions = [
            p for p, _ in ds.positions()
            if any(p.startswith(w + "/") for w in exp_wells)
        ]

    if not skip_existing:
        return {
            "positions": all_positions,
            "skipped_positions": [],
            "source_path": source_path,
        }

    # Check which positions already have the output label
    import zarr

    positions_to_process = []
    skipped_positions = []

    store = zarr.open(str(source_path), mode="r")
    for pos in all_positions:
        if pos not in store:
            continue

        # Check if output label exists
        labels_group = store[pos].get("labels", {})
        if output_label_name in labels_group:
            skipped_positions.append(pos)
        else:
            positions_to_process.append(pos)

    return {
        "positions": positions_to_process,
        "skipped_positions": skipped_positions,
        "source_path": source_path,
    }


def _segment_cell_position_chunk(positions: list[str], **per_pos_kwargs) -> list:
    """Run ``segment_single_position`` sequentially for each position in the
    chunk. Used by ``submit_cell_segmentation_jobs`` when ``chunk_size > 1``
    so one SLURM task processes multiple FOVs and amortises the per-task
    setup (slurm scheduling, worker import, model load) across them.
    """
    results = []
    for pos in positions:
        results.append(
            segment_single_position(position=pos, **per_pos_kwargs)
        )
    return results


def submit_cell_segmentation_jobs(
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
    diameter: float = DEFAULT_DIAMETER,
    flow_threshold: float = DEFAULT_FLOW_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    nuclei_channel: str = DEFAULT_NUCLEI_CHANNEL,
    membrane_channel: str = DEFAULT_MEMBRANE_CHANNEL,
    use_clahe: bool = True,
    output_label_name: str = "cell_seg",
) -> dict:
    """
    Submit parallel SLURM jobs for cell segmentation.

    Parameters
    ----------
    experiment : str
        Experiment name (e.g., "ops0033_20250429")
    positions : list[str]
        Positions to process (default: all positions in store)
    slurm_params : dict
        SLURM parameters (timeout_min, mem, cpus_per_task, gpus_per_node, etc.)
    dry_run : bool
        If True, print what would be submitted without actually submitting
    wait_for_completion : bool
        If True, wait for all jobs to complete before returning
    verbose : bool
        Print detailed progress
    force : bool
        If True, process positions even if cell_seg already exists
    output_label_name : str
        Name for output label group (default: "cell_seg", use "cp_cell_seg" for cell painting)

    Returns
    -------
    dict
        Job submission results with job IDs, metadata, and completion status
    """
    # Get available positions
    try:
        info = get_available_positions(
            experiment, skip_existing=not force, output_label_name=output_label_name
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"Cell segmentation failed: {e}") from e

    # Use provided positions or defaults
    pos_list = positions if positions else info["positions"]

    # Validate positions
    all_positions = info["positions"] + info["skipped_positions"]
    invalid_positions = [p for p in pos_list if p not in all_positions]
    if invalid_positions:
        print(f"Warning: Invalid positions ignored: {invalid_positions}")
        pos_list = [p for p in pos_list if p in all_positions]

    if not pos_list:
        skipped = info.get("skipped_positions", [])
        if skipped:
            print(f"\n{'='*60}")
            print(f"All positions already have cell_seg for {experiment}")
            print(f"{'='*60}")
            print(f"\nPositions with existing cell_seg:")
            for pos in skipped[:10]:
                print(f"  - {pos}")
            if len(skipped) > 10:
                print(f"  ... and {len(skipped) - 10} more")
            print()

            # Prompt for overwrite
            try:
                response = input("Would you like to overwrite existing segmentations? [y/N]: ").strip().lower()
                if response in ['y', 'yes']:
                    info = get_available_positions(
                        experiment, skip_existing=False, output_label_name=output_label_name
                    )
                    pos_list = info["positions"]
                    print(f"\nProceeding to overwrite {len(pos_list)} position(s)...")
                else:
                    print("\nNo positions to process. Exiting.")
                    return {"success": True, "skipped": True, "message": "All segmentations already exist"}
            except (KeyboardInterrupt, EOFError):
                print("\n\nCancelled by user.")
                return {"success": False, "error": "Cancelled by user"}
        else:
            raise RuntimeError(
                f"Cell segmentation failed for {experiment}: no valid positions to process"
            )

    # QC Check: Verify prediction channels have data for EACH position to be processed
    print(f"Running QC check on prediction channels...")
    failed_qc_positions = []
    for pos in pos_list:
        qc_result = check_channels_have_data(
            experiment=experiment,
            position=pos,
            nuclei_channel=nuclei_channel,
            membrane_channel=membrane_channel,
        )
        if verbose:
            print(f"  Position checked: {pos}")
            print(f"  Regions sampled: {qc_result['regions_checked']}")
            print(f"  Membrane nonzero (avg): {qc_result['membrane_nonzero_frac']:.2%}")
            print(f"  Nuclei nonzero (avg): {qc_result['nuclei_nonzero_frac']:.2%}")
            if qc_result['failed_regions'] > 0:
                print(f"  Failed regions: {qc_result['failed_regions']}")
        if not qc_result["passed"]:
            failed_qc_positions.append(pos)

    if failed_qc_positions:
        print(f"\n{'='*60}")
        print(f"QC CHECK FAILED for {experiment}")
        print(f"{'='*60}")
        print(f"\nPosition checked: {qc_result['position_checked']}")
        print(f"{qc_result['message']}")
        print(f"\nThis likely means virtual staining predictions are missing or failed.")
        print(f"Please check the phenotyping_v3.zarr store and ensure membrane_prediction")
        print(f"channel has valid data before running cell segmentation.\n")
        raise RuntimeError(
            f"Cell segmentation QC failed for {experiment}: "
            f"{len(failed_qc_positions)} position(s) have blank channels: "
            f"{', '.join(failed_qc_positions)}"
        )

    print(f"  {qc_result['message']}")

    # Define SLURM parameters for GPU-accelerated cell segmentation
    # Based on segment_and_stitch_pheno_cells config but optimized for per-position jobs:
    # - Original: 400 min for all positions with 2 GPUs
    # - Per-position segmentation: ~133 min (400/3)
    # - Resharding: ~10 min (convert unsharded → sharded format)
    # - Pyramid building: ~20 min (5 levels)
    # - Total: ~165 min with buffer
    # - Using 2 GPUs to match the original config for tile-level parallelism
    default_slurm_params = {
        "timeout_min": 60,  # ~133 min segmentation + 30 min resharding/pyramids + buffer
        "mem": "400G",  # 30 workers + 44GB canvas + 30GB overlap cache + fragmentation
        "cpus_per_task": 32,  # CPUs for Dask workers
        "gpus_per_node": 2,  # 2 GPUs for tile-level parallelism (matching original)
        "slurm_partition": "gpu",
        "slurm_constraint": "[h100|h200]",  # MultiGPUCluster distributes across GPUs
    }

    if slurm_params:
        default_slurm_params.update(slurm_params)

    # Prepare job list — one SLURM task per chunk of `chunk_size` positions.
    # chunk_size=1 preserves the original "one job per FOV" shape; bigger
    # chunk_size amortises per-task setup across multiple FOVs.
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
        nuclei_channel_name=nuclei_channel,
        membrane_channel_name=membrane_channel,
        use_clahe=use_clahe,
        debug_only=False,
        use_parallel=True,
        output_label_name=output_label_name,
    )

    jobs_to_submit = []
    for chunk in chunks:
        first_safe = chunk[0].replace("/", "_")
        last_safe = chunk[-1].replace("/", "_")
        if len(chunk) == 1:
            name = f"cellseg_{first_safe}"
            func = segment_single_position
            kwargs = {"position": chunk[0], **common_kwargs}
        else:
            name = f"cellseg_{first_safe}_to_{last_safe}_n{len(chunk)}"
            func = _segment_cell_position_chunk
            kwargs = {"positions": chunk, **common_kwargs}
        jobs_to_submit.append({
            "name": name,
            "func": func,
            "kwargs": kwargs,
            "metadata": {
                "experiment": experiment,
                "positions": chunk,
                "output_label": output_label_name,
            },
            "slurm_params": default_slurm_params,
        })

    if not jobs_to_submit:
        raise RuntimeError(
            f"Cell segmentation failed for {experiment}: no jobs to submit"
        )

    # Print summary
    print(f"\n{'='*60}")
    print(f"Cell Segmentation Batch Submission")
    print(f"{'='*60}")
    print(f"Experiment: {experiment}")
    print(f"Output label: {output_label_name}")
    print(f"Positions to process: {len(pos_list)}")
    if len(pos_list) <= 10:
        for pos in pos_list:
            print(f"  - {pos}")
    else:
        for pos in pos_list[:5]:
            print(f"  - {pos}")
        print(f"  ... and {len(pos_list) - 5} more")
    print(f"\nSegmentation parameters:")
    print(f"  Nuclei channel: {nuclei_channel}")
    print(f"  Membrane channel: {membrane_channel}")
    print(f"  Tile size: {tile_size}, Overlap: {tile_overlap}")
    print(f"  Cellpose: d={diameter}, ft={flow_threshold}")
    print(f"  IoU threshold: {iou_threshold}")
    print(f"  CLAHE: {'enabled' if use_clahe else 'disabled'}")
    print(f"\nSLURM resources per job:")
    print(f"  Timeout: {default_slurm_params['timeout_min']} min")
    print(f"  Memory: {default_slurm_params['mem']}")
    print(f"  CPUs: {default_slurm_params['cpus_per_task']}")
    print(f"  GPUs: {default_slurm_params['gpus_per_node']}")
    print(f"  Partition: {default_slurm_params['slurm_partition']}")
    if default_slurm_params.get('slurm_constraint'):
        print(f"  GPU constraint: {default_slurm_params['slurm_constraint']}")
    print(f"{'='*60}\n")

    # Submit jobs
    print(f"Submitting {len(jobs_to_submit)} cell segmentation jobs...")
    result = submit_parallel_jobs(
        jobs_to_submit=jobs_to_submit,
        experiment=f"{experiment}_cell_segmentation",
        slurm_params=default_slurm_params,
        log_dir=f"slurm_cell_seg_logs/{experiment}",
        step_name="submit_cell_segmentation_jobs",
        manifest_prefix="cell_seg",
        dry_run=dry_run,
        wait_for_completion=False,  # Don't wait yet
        verbose=verbose,
        post_completion_callback=None,
    )

    # If user wants to wait, wait for job array completion
    if wait_for_completion and not dry_run:
        from cyclops_utils.hpc.slurm_batch_utils import wait_for_multiple_job_arrays

        job_arrays = []
        if result.get("success") and "submitted_jobs" in result:
            job_arrays.append({
                "submitted_jobs": result["submitted_jobs"],
                "base_job_id": result["base_job_id"],
                "label": f"Cell Segmentation ({result['base_job_id']})",
                "slurm_params": default_slurm_params,
            })

        if job_arrays:
            wait_result = wait_for_multiple_job_arrays(
                job_arrays=job_arrays,
                experiment=experiment,
                verbose=verbose,
            )

            failed = wait_result.get("failed", [])
            if failed:
                failed_names = [f[0] if isinstance(f, tuple) else str(f) for f in failed]
                raise RuntimeError(
                    f"Cell segmentation failed for {len(failed)}/{len(wait_result.get('completed', [])) + len(failed)} jobs:\n"
                    + "\n".join(f"  - {n}" for n in failed_names)
                )

            return {
                "success": True,
                "result": result,
                "completed": wait_result.get("completed", []),
                "failed": [],
                "all_completed": True,
            }

    return {
        "success": result.get("success", False),
        "result": result,
        "dry_run": dry_run,
    }


def main():
    """CLI entry point for SLURM batch cell segmentation submission."""
    parser = argparse.ArgumentParser(
        description="Submit cell segmentation jobs to SLURM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process ALL experiments needing cell segmentation
  python -m cyclops_process.processes.cell_seg.cell_segmentation_orchestrator --all

  # Force reprocess ALL experiments (even with existing cell_seg)
  python -m cyclops_process.processes.cell_seg.cell_segmentation_orchestrator --all --force

  # Process specific experiments (supports shorthand like "103")
  python -m cyclops_process.processes.cell_seg.cell_segmentation_orchestrator -e 103 69

  # Process a single experiment
  python -m cyclops_process.processes.cell_seg.cell_segmentation_orchestrator -e ops0033_20250429

  # Cell painting mode: use actin channel for cell boundaries (e.g., ops0094)
  python -m cyclops_process.processes.cell_seg.cell_segmentation_orchestrator -e 94 --cell-paint

  # Cell painting mode with custom output label
  python -m cyclops_process.processes.cell_seg.cell_segmentation_orchestrator -e 94 --cell-paint --output-label actin_cell_seg
        """,
    )

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--all",
        action="store_true",
        help="Process ALL experiments with phenotyping_v3.zarr that need cell_seg",
    )
    mode_group.add_argument(
        "--experiment", "-e",
        type=str,
        nargs="+",
        default=None,
        help="Experiment name(s) to process (supports shorthand like '103' or 'ops103')",
    )

    parser.add_argument(
        "--positions", "-p",
        type=str,
        nargs="+",
        default=None,
        help="Positions to process (e.g., A/1/0 A/2/0). Default: all positions. Only valid with single experiment.",
    )

    parser.add_argument(
        "--list-positions",
        action="store_true",
        help="List available positions for the experiment, then exit",
    )

    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-run even if cell_seg already exists",
    )

    # SLURM parameters
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="SLURM timeout in minutes (default: 120 = segmentation + resharding + pyramids)",
    )

    parser.add_argument(
        "--mem",
        type=str,
        default="400G",
        help="SLURM memory allocation (default: 400G for 30 workers + canvas + overlap cache)",
    )

    parser.add_argument(
        "--cpus",
        type=int,
        default=32,
        help="SLURM CPUs per task (default: 32, matching segment_and_stitch_pheno_cells)",
    )

    parser.add_argument(
        "--gpus",
        type=int,
        default=2,
        help="SLURM GPUs per node (default: 2, matching segment_and_stitch_pheno_cells)",
    )

    parser.add_argument(
        "--gpu-constraint",
        type=str,
        default="[h100|h200]",
        help="SLURM GPU constraint (default: [h100|h200], matching segment_and_stitch_pheno_cells). Use 'none' to disable.",
    )

    parser.add_argument(
        "--partition",
        type=str,
        default="gpu",
        help="SLURM partition (default: gpu)",
    )

    # Segmentation parameters
    parser.add_argument(
        "--tile-size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        help=f"Tile size in pixels (default: {DEFAULT_TILE_SIZE})",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help=f"Tile overlap in pixels (default: {DEFAULT_OVERLAP})",
    )

    parser.add_argument(
        "--diameter",
        type=float,
        default=DEFAULT_DIAMETER,
        help=f"Cellpose diameter (default: {DEFAULT_DIAMETER})",
    )

    parser.add_argument(
        "--flow-threshold",
        type=float,
        default=DEFAULT_FLOW_THRESHOLD,
        help=f"Cellpose flow threshold (default: {DEFAULT_FLOW_THRESHOLD})",
    )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help=f"IoU threshold for merging (default: {DEFAULT_IOU_THRESHOLD})",
    )

    parser.add_argument(
        "--nuclei-channel",
        type=str,
        default=DEFAULT_NUCLEI_CHANNEL,
        help=f"Nuclei channel name (default: {DEFAULT_NUCLEI_CHANNEL})",
    )

    parser.add_argument(
        "--membrane-channel",
        type=str,
        default=DEFAULT_MEMBRANE_CHANNEL,
        help=f"Membrane channel name (default: {DEFAULT_MEMBRANE_CHANNEL})",
    )

    parser.add_argument(
        "--no-clahe",
        action="store_true",
        help="Disable CLAHE preprocessing",
    )

    # Cell painting mode
    parser.add_argument(
        "--cell-paint",
        action="store_true",
        help=(
            "Cell painting mode: segment cells using actin channel only (CP1_f_actin_Phalloidin) "
            "instead of virtual staining membrane_prediction. Uses actin for both CLAHE inputs "
            "(no nuclei mixing). Output stored to 'cp_cell_seg' label. "
            "Use this for experiments with cell_painting config (e.g., ops0094)."
        ),
    )

    # 4i mode
    parser.add_argument(
        "--four-i",
        action="store_true",
        help=(
            "4i mode: segment cells using DAPI (round 4) + b-catenin (round 4 cell-junction marker) "
            "from the 4i v3 store. Output stored to '4i_cell_seg' label. "
            "Use this for experiments with 4i config (e.g., ops0144)."
        ),
    )

    parser.add_argument(
        "--output-label",
        type=str,
        default=None,
        help=(
            "Custom output label name (default: 'cell_seg', or 'cp_cell_seg' with --cell-paint). "
            "Overrides the default label name for the segmentation output."
        ),
    )

    # Job control
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be submitted without actually submitting",
    )

    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit jobs and return immediately without waiting for completion",
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce verbosity",
    )

    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt (auto-confirm submission)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.all and not args.experiment:
        parser.print_help()
        print("\nError: Must specify either --all or --experiment/-e")
        sys.exit(1)

    if args.positions and args.all:
        print("Error: --positions is not valid with --all mode")
        sys.exit(1)

    if args.positions and args.experiment and len(args.experiment) > 1:
        print("Error: --positions is only valid with a single experiment")
        sys.exit(1)

    # Build SLURM parameters
    slurm_params = {
        "timeout_min": args.timeout,
        "mem": args.mem,
        "cpus_per_task": args.cpus,
        "gpus_per_node": args.gpus,
        "slurm_partition": args.partition,
    }
    if args.gpu_constraint and args.gpu_constraint.lower() != "none":
        slurm_params["slurm_constraint"] = args.gpu_constraint

    # ==========================================================================
    # Cell Painting Mode: Override channel and output label settings
    # ==========================================================================
    if args.cell_paint:
        # Use cell painting channels (actin-only mode)
        nuclei_channel = CP_NUCLEI_CHANNEL
        membrane_channel = CP_MEMBRANE_CHANNEL
        output_label = args.output_label if args.output_label else CP_OUTPUT_LABEL
        print(f"\n{'='*60}")
        print("Cell Painting Mode Enabled (actin-only)")
        print(f"{'='*60}")
        print(f"  CLAHE input: {membrane_channel} (actin-only, no nuclei mixing)")
        print(f"  Output label: {output_label}")
        print(f"{'='*60}\n")
    elif args.four_i:
        # Use 4i channels: DAPI (round 1) + b-catenin (round 4 membrane marker)
        nuclei_channel = FOUR_I_NUCLEI_CHANNEL
        membrane_channel = FOUR_I_MEMBRANE_CHANNEL
        output_label = args.output_label if args.output_label else FOUR_I_OUTPUT_LABEL
        print(f"\n{'='*60}")
        print("4i Mode Enabled")
        print(f"{'='*60}")
        print(f"  Nuclei channel: {nuclei_channel}")
        print(f"  Membrane channel: {membrane_channel} (b-catenin cell-junction marker)")
        print(f"  Output label: {output_label}")
        print(f"{'='*60}\n")
    else:
        # Standard virtual staining mode
        nuclei_channel = args.nuclei_channel
        membrane_channel = args.membrane_channel
        output_label = args.output_label if args.output_label else "cell_seg"

    # ==========================================================================
    # --all mode: detect and process all experiments needing cell segmentation
    # ==========================================================================
    if args.all:
        print(f"\n{'='*60}")
        print("Detecting experiments needing cell segmentation...")
        print(f"{'='*60}\n")

        experiments_to_process, experiments_completed = detect_experiments_for_cell_seg(
            force=args.force,
            verbose=not args.quiet,
        )

        if not experiments_to_process:
            print(f"\nAll experiments are complete! No cell segmentation jobs needed.\n")
            if experiments_completed:
                print(f"Completed experiments: {len(experiments_completed)}")
                for exp, _, _, _ in experiments_completed[:10]:
                    print(f"  - {exp}")
                if len(experiments_completed) > 10:
                    print(f"  ... and {len(experiments_completed) - 10} more")
            sys.exit(0)

        # Print summary of experiments to process
        print(f"\n{'='*60}")
        print(f"Cell Segmentation Batch Submission: {len(experiments_to_process)} experiments")
        print(f"{'='*60}\n")

        print("Experiments to process:")
        for exp, n_done, n_total, _ in experiments_to_process:
            status = f"({n_done}/{n_total} positions done)" if n_total > 0 else ""
            print(f"  - {exp} {status}")

        if experiments_completed:
            print(f"\nAlready completed ({len(experiments_completed)} experiments):")
            for exp, _, _, _ in experiments_completed[:10]:
                print(f"  - {exp}")
            if len(experiments_completed) > 10:
                print(f"  ... and {len(experiments_completed) - 10} more")

        # Count total positions and run QC check on ALL positions BEFORE confirmation
        print(f"\n{'='*60}")
        print("Running QC checks on prediction channels...")
        print(f"{'='*60}\n")

        total_positions = 0
        experiment_position_counts = {}
        experiment_positions = {}  # Store positions per experiment
        failed_qc_positions = []  # List of (exp, pos) tuples that failed QC

        for exp, _, _, _ in experiments_to_process:
            try:
                info = get_available_positions(
                    exp, skip_existing=not args.force, output_label_name=output_label
                )
                positions = info["positions"]
                pos_count = len(positions)
                experiment_position_counts[exp] = pos_count
                experiment_positions[exp] = positions
                total_positions += pos_count

                # QC check each position in this experiment
                print(f"  {exp} ({pos_count} positions):")
                for pos in positions:
                    qc_result = check_channels_have_data(
                        experiment=exp,
                        position=pos,
                        nuclei_channel=nuclei_channel,
                        membrane_channel=membrane_channel,
                    )
                    if qc_result["passed"]:
                        if not args.quiet:
                            print(f"    [PASS] {pos}: membrane={qc_result['membrane_nonzero_frac']:.1%}")
                    else:
                        failed_qc_positions.append((exp, pos, qc_result))
                        print(f"    [FAIL] {pos}: membrane={qc_result['membrane_nonzero_frac']:.2%}")

            except Exception as e:
                print(f"  {exp}: Error getting positions - {e}")
                experiment_position_counts[exp] = 0
                experiment_positions[exp] = []

        # Report QC failures
        if failed_qc_positions:
            print(f"\n{'='*60}")
            print(f"QC FAILED for {len(failed_qc_positions)} position(s)")
            print(f"{'='*60}")
            print("\nPositions with blank prediction channels:")
            for exp, pos, qc in failed_qc_positions:
                print(f"  - {exp} / {pos}: membrane={qc['membrane_nonzero_frac']:.4%}")
            print("\nThese positions will be SKIPPED.")
            print("Please ensure virtual staining predictions are complete before running cell seg.\n")

            # Remove failed positions from the lists
            for exp, pos, _ in failed_qc_positions:
                if exp in experiment_positions and pos in experiment_positions[exp]:
                    experiment_positions[exp].remove(pos)
                    experiment_position_counts[exp] -= 1
                    total_positions -= 1

        # Check if any positions remain
        if total_positions == 0:
            print("\nNo positions passed QC check. Exiting.\n")
            sys.exit(1)

        n_passed = total_positions
        n_failed = len(failed_qc_positions)
        print(f"\nQC Summary: {n_passed} positions passed, {n_failed} positions failed")

        print(f"\nTotal positions to process: {total_positions}")
        print(f"\nSLURM Resources (per position):")
        print(f"  Timeout: {slurm_params['timeout_min']} min")
        print(f"  Memory: {slurm_params['mem']}")
        print(f"  CPUs: {slurm_params['cpus_per_task']}")
        print(f"  GPUs: {slurm_params['gpus_per_node']}")
        print(f"  Partition: {slurm_params['slurm_partition']}")
        if slurm_params.get('slurm_constraint'):
            print(f"  GPU constraint: {slurm_params['slurm_constraint']}")
        print(f"\n{'='*60}\n")

        # Confirmation prompt
        if not args.yes:
            try:
                response = input(f"Submit {total_positions} jobs for {len(experiments_to_process)} experiments? [y/N]: ").strip().lower()
                if response not in ['y', 'yes']:
                    print("\nCancelled. No jobs submitted.\n")
                    sys.exit(0)
            except (KeyboardInterrupt, EOFError):
                print("\n\nCancelled. No jobs submitted.\n")
                sys.exit(0)
            print()
        else:
            print("Proceeding with submission (--yes flag provided)...\n")

        # Gather ALL jobs from ALL experiments into a single job array
        # Use experiment_positions which already has QC-passed positions only
        all_jobs = []
        for exp, _, _, _ in experiments_to_process:
            positions = experiment_positions.get(exp, [])
            if not positions:
                continue

            for pos in positions:
                pos_safe = pos.replace("/", "_")
                job_name = f"cellseg_{exp}_{pos_safe}"

                all_jobs.append({
                    "name": job_name,
                    "func": segment_single_position,
                    "kwargs": {
                        "experiment": exp,
                        "position": pos,
                        "tile_size": args.tile_size,
                        "tile_overlap": args.overlap,
                        "diameter": args.diameter,
                        "flow_threshold": args.flow_threshold,
                        "iou_threshold": args.iou_threshold,
                        "nuclei_channel_name": nuclei_channel,
                        "membrane_channel_name": membrane_channel,
                        "use_clahe": not args.no_clahe,
                        "debug_only": False,
                        "use_parallel": True,
                        "output_label_name": output_label,
                    },
                    "metadata": {
                        "experiment": exp,
                        "position": pos,
                        "output_label": output_label,
                    },
                })

        if not all_jobs:
            print("No jobs to submit!")
            sys.exit(1)

        print(f"Submitting {len(all_jobs)} jobs as a single SLURM array...")

        # Submit all jobs as a single array
        result = submit_parallel_jobs(
            jobs_to_submit=all_jobs,
            experiment="batch_cell_segmentation",
            slurm_params=slurm_params,
            log_dir="slurm_cell_seg_logs/all/%j",
            manifest_prefix="cell_seg_batch",
            dry_run=args.dry_run,
            wait_for_completion=not args.no_wait,
            verbose=not args.quiet,
        )

        # Save experiment-to-job mapping in the all/ directory
        if result.get("success") and not args.dry_run:
            from collections import defaultdict
            from pathlib import Path
            import yaml

            base_job_id = result.get("base_job_id")
            jobs_list = result.get("jobs", [])

            # Build experiment -> job IDs mapping
            exp_to_jobs = defaultdict(list)
            for job in jobs_list:
                exp = job.get("experiment", "unknown")
                job_id = job.get("job_id", job.get("array_index", "?"))
                exp_to_jobs[exp].append(job_id)

            # Save mapping to all/ directory
            manifest_dir = Path("slurm_logs/slurm_cell_seg_logs/all")
            manifest_dir.mkdir(parents=True, exist_ok=True)
            manifest_file = manifest_dir / f"experiment_mapping_{base_job_id}.yaml"

            mapping_data = {
                "slurm_array_id": base_job_id,
                "total_jobs": len(jobs_list),
                "experiments": {exp: {"job_ids": ids, "count": len(ids)} for exp, ids in sorted(exp_to_jobs.items())},
            }

            with open(manifest_file, "w") as f:
                yaml.dump(mapping_data, f, default_flow_style=False, sort_keys=False)

            print(f"\nExperiment mapping saved: {manifest_file}")

        # Exit with appropriate code
        if args.dry_run:
            sys.exit(0)
        elif result.get("success"):
            if result.get("all_completed") is not None:
                sys.exit(0 if result.get("all_completed") else 1)
            else:
                sys.exit(0)
        else:
            sys.exit(1)

    # ==========================================================================
    # --experiment mode: process specific experiment(s)
    # ==========================================================================
    # Resolve experiment names
    resolved_experiments = []
    for exp_input in args.experiment:
        resolved = resolve_experiment_name(exp_input, allow_interactive=True)
        if resolved is None:
            print(f"Error: Could not resolve experiment '{exp_input}'")
            sys.exit(1)
        resolved_experiments.append(resolved)

    # Handle --list-positions mode (single experiment only)
    if args.list_positions:
        if len(resolved_experiments) > 1:
            print("Error: --list-positions only works with a single experiment")
            sys.exit(1)

        try:
            info = get_available_positions(resolved_experiments[0], skip_existing=False)
            print(f"\nExperiment: {resolved_experiments[0]}")
            print(f"Source: {info['source_path']}")
            print(f"\nPositions ({len(info['positions'])}):")
            for pos in info['positions']:
                print(f"  {pos}")
            print()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        sys.exit(0)

    # Single experiment with optional positions
    if len(resolved_experiments) == 1:
        result = submit_cell_segmentation_jobs(
            experiment=resolved_experiments[0],
            positions=args.positions,
            slurm_params=slurm_params,
            dry_run=args.dry_run,
            wait_for_completion=not args.no_wait,
            verbose=not args.quiet,
            force=args.force,
            tile_size=args.tile_size,
            tile_overlap=args.overlap,
            diameter=args.diameter,
            flow_threshold=args.flow_threshold,
            iou_threshold=args.iou_threshold,
            nuclei_channel=nuclei_channel,
            membrane_channel=membrane_channel,
            use_clahe=not args.no_clahe,
            output_label_name=output_label,
        )

        # Exit with appropriate code
        if result.get("dry_run"):
            sys.exit(0)
        elif result.get("success"):
            if result.get("all_completed") is not None:
                sys.exit(0 if result.get("all_completed") else 1)
            else:
                sys.exit(0)
        else:
            sys.exit(1)

    # Multiple experiments: process each
    print(f"\n{'='*60}")
    print(f"Cell Segmentation: {len(resolved_experiments)} experiments")
    print(f"{'='*60}\n")

    for exp in resolved_experiments:
        print(f"  - {exp}")

    # Count total positions
    total_positions = 0
    experiment_position_counts = {}
    for exp in resolved_experiments:
        try:
            info = get_available_positions(
                exp, skip_existing=not args.force, output_label_name=output_label
            )
            pos_count = len(info["positions"])
            experiment_position_counts[exp] = pos_count
            total_positions += pos_count
        except Exception as e:
            print(f"  Warning: Could not get positions for {exp}: {e}")
            experiment_position_counts[exp] = 0

    print(f"\nTotal positions to process: {total_positions}")
    print(f"\n{'='*60}\n")

    # Confirmation prompt
    if not args.yes:
        try:
            response = input(f"Submit {total_positions} jobs for {len(resolved_experiments)} experiments? [y/N]: ").strip().lower()
            if response not in ['y', 'yes']:
                print("\nCancelled. No jobs submitted.\n")
                sys.exit(0)
        except (KeyboardInterrupt, EOFError):
            print("\n\nCancelled. No jobs submitted.\n")
            sys.exit(0)
        print()

    # Submit jobs for each experiment
    all_results = []
    for exp in resolved_experiments:
        print(f"\n[{exp}] Submitting {experiment_position_counts.get(exp, '?')} positions...")

        result = submit_cell_segmentation_jobs(
            experiment=exp,
            positions=None,
            slurm_params=slurm_params,
            dry_run=args.dry_run,
            wait_for_completion=False,
            verbose=not args.quiet,
            force=args.force,
            tile_size=args.tile_size,
            tile_overlap=args.overlap,
            diameter=args.diameter,
            flow_threshold=args.flow_threshold,
            iou_threshold=args.iou_threshold,
            nuclei_channel=nuclei_channel,
            membrane_channel=membrane_channel,
            use_clahe=not args.no_clahe,
            output_label_name=output_label,
        )
        all_results.append((exp, result))

    # Summary
    print(f"\n{'='*60}")
    print("Submission Summary")
    print(f"{'='*60}")
    successful = [(exp, r) for exp, r in all_results if r.get("success")]
    failed = [(exp, r) for exp, r in all_results if not r.get("success")]
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    if failed:
        for exp, r in failed:
            print(f"    - {exp}: {r.get('error', 'Unknown error')}")
    print(f"{'='*60}\n")

    # Optionally wait for all jobs
    if not args.no_wait and not args.dry_run and successful:
        from cyclops_utils.hpc.slurm_batch_utils import wait_for_multiple_job_arrays

        job_arrays = []
        for exp, result in successful:
            if "submitted_jobs" in result.get("result", {}):
                job_arrays.append({
                    "submitted_jobs": result["result"]["submitted_jobs"],
                    "base_job_id": result["result"]["base_job_id"],
                    "label": f"Cell Seg {exp} ({result['result']['base_job_id']})",
                    "slurm_params": slurm_params,
                })

        if job_arrays:
            print("Waiting for all jobs to complete...")
            wait_result = wait_for_multiple_job_arrays(
                job_arrays=job_arrays,
                experiment="batch_cell_segmentation",
                verbose=not args.quiet,
            )
            sys.exit(0 if len(wait_result.get("failed", [])) == 0 else 1)

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
