"""
Cell Segmentation Module
========================

Provides cell segmentation using Cellpose-SAM with hybrid IoU-based stitching.

Key features:
- Operates on stitched phenotyping_v3.zarr (not tile store)
- Configurable tile size and overlap
- Hybrid stitching: IoU-based merging (avoids both edge-cell loss and over-merging)
- Preview mode for quick testing
- GPU parallelization using Dask workers (same as segment.py)
- SLURM batch submission for multi-position processing

Usage:
    from cyclops_process.processes.cell_seg import segment_single_position

    # Preview mode (sequential, single GPU)
    result = segment_single_position(
        experiment="ops0033_20250429",
        position="A/1/0",
        debug_only=True,  # Preview mode (always sequential)
    )

    # Full position with parallel processing (auto-detects available GPUs)
    result = segment_single_position(
        experiment="ops0033_20250429",
        position="A/1/0",
        use_parallel=True,  # Enable Dask parallelization (default)
    )

SLURM Batch Submission:
    # Submit all positions to SLURM
    python -m cyclops_process.processes.cell_seg.cell_segmentation_orchestrator \\
        --experiment ops0033_20250429

    # Submit specific positions
    python -m cyclops_process.processes.cell_seg.cell_segmentation_orchestrator \\
        --experiment ops0033_20250429 --positions A/1/0 A/2/0
"""

# Lazy export — importing `cell_segmentation` transitively loads cupy +
# stitch.tile, which initializes CUDA in the parent process. With CUDA
# bound to GPU 0 in the parent, Dask workers fork()'d from it all bind
# to GPU 0 regardless of CUDA_VISIBLE_DEVICES. Switching to PEP 562
# lazy lookup so sibling modules (e.g. `nuclei_pass`) can be imported
# without paying that cost. The public API is unchanged: callers can
# still `from cyclops_process.processes.cell_seg import segment_single_position`.
__all__ = ["segment_single_position"]


def __getattr__(name):
    if name == "segment_single_position":
        from .cell_segmentation import segment_single_position
        return segment_single_position
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
