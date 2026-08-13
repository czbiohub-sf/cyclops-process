"""
SLURM batch submission for ISS round-to-round registration.

Submits parallel jobs for:
- Nucleus → Round 0 registration (1 job)
- Round-to-round registration (9 jobs: R1→R0, R2→R1, ..., R9→R8)
- Visualization and metrics generation (1 job)

Usage:
------
# Submit all registration jobs for a single well
python -m cyclops_process.processes.auto_register.iss_cycle_register_orchestrator --experiment ops0032_20250428 --well 1

# Submit for all wells
python -m cyclops_process.processes.auto_register.iss_cycle_register_orchestrator --experiment ops0032_20250428 --well all

# Custom SLURM parameters
python -m cyclops_process.processes.auto_register.iss_cycle_register_orchestrator \
    --experiment ops0032_20250428 --well 1 \
    --timeout 10 --mem 16GB --cpus 16

# Dry run to see what would be submitted
python -m cyclops_process.processes.auto_register.iss_cycle_register_orchestrator \
    --experiment ops0032_20250428 --well 1 --dry-run
"""

import argparse
import re
import sys
import os
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, os.getcwd())

from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
from cyclops_utils.data.filesystem import resolve_experiment_name, parse_well
from cyclops_utils.hpc.slurm_utils import check_jobs_complete
from cyclops_utils.profiling.decorators import versioned_function
from iohub.ngff import open_ome_zarr
from cyclops_utils.io.zarr_utils import ensure_position_array

# Import registration functions
from cyclops_process.processes.auto_register.iss_cycle_register import (
    register_segmentation_to_nucleus,
    register_nucleus_to_round0,
    register_round_pair,
    DEFAULT_ISS_PARAMS,
    create_drift_trajectory_plot,
    create_all_rounds_overlay,
    save_affine_to_yaml,
    affine_3x3_to_4x4_zyx,
    apply_iss_transforms,
)

# Experiments with only 9 physical ISS rounds (indices 0-8) instead of the default 10
EXPERIMENTS_WITH_9_ROUNDS = [42]

def _get_n_physical_rounds(dataset, well, experiment: str = "", verbose: bool = False) -> int:
    """Number of physical ISS rounds actually stacked for this experiment.

    Nothing is hardcoded per-experiment. Primary source is the stitched store's T
    dimension (ground truth — e.g. 11 for a no-incorporation round 0 + extra end
    round layout). If the store can't be read, fall back to len(iss_rounds) from
    the experiment config.

    Reads T from the position array's plain zarr metadata (.zarray) — NOT via
    iohub.open_ome_zarr, so it stays a ~10ms metadata read (no NGFF plate parse).
    """
    import zarr
    store_err = None
    try:
        iss_zarr = dataset.store_paths["iss_stitch"]
        grp = zarr.open(str(iss_zarr), mode="r")
        row, col = parse_well(well)
        arr = grp[f"{row}/{col}/0"]
        if isinstance(arr, zarr.Group):  # HCS: position holds a multiscale group
            arr = arr["0"] if "0" in arr else arr[list(arr.array_keys())[0]]
        n_rounds = int(arr.shape[0])  # T
        if verbose:
            print(f"  Physical ISS rounds (from store T): {n_rounds} (0-{n_rounds-1})")
        return n_rounds
    except Exception as e:
        store_err = e

    # Fallback: len(iss_rounds) from the experiment config
    try:
        import yaml
        with open(dataset.config_paths["exp_config"], "r") as f:
            cfg = yaml.safe_load(f) or {}
        iss_rounds = (cfg.get("base_calling_params") or {}).get("iss_rounds")
        if iss_rounds:
            n_rounds = len(iss_rounds)
            if verbose:
                print(f"  Physical ISS rounds (from config iss_rounds): {n_rounds} "
                      f"(store unreadable: {store_err})")
            return n_rounds
    except Exception as cfg_err:
        store_err = f"{store_err}; config read failed: {cfg_err}"

    if verbose:
        print(f"  WARNING: could not determine round count ({store_err}); defaulting to 10")
    return 10


def _set_reproducible_seed(seed: int = 42):
    """Seed every randomness source registration touches.

    Call at the start of each job_* function when reproducible=True.
    Threading must also be pinned to 1 (OMP_NUM_THREADS=1 etc) for BLAS
    floating-point ordering to be stable across runs — those env vars
    have to be set before numpy/BLAS imports, so the caller's sbatch
    is responsible for that side. We assert here as a tripwire.
    """
    import os, random

    np.random.seed(seed)
    random.seed(seed)
    try:
        import cupy as _cp
        _cp.random.seed(seed)
    except Exception:
        # CuPy may import but fail to access a GPU on CPU-only nodes
        # (cudaErrorNoDevice). Either way, no GPU randomness to seed.
        pass
    # Tripwire — these only take effect if set before BLAS imports, so
    # we don't try to set them here. We just warn the user if they're
    # not pinned to 1.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        v = os.environ.get(var, "")
        if v != "1":
            print(f"[reproducible] WARNING: {var}={v!r} (expected '1' for "
                  "bit-reproducible BLAS — set in sbatch before launching python)")


def _precreate_registered_zarr(experiment: str, wells: list, verbose: bool = True):
    """Pre-create the registered zarr store with all well positions.

    ``wells`` is a list of "row/col/fov" units so wells across different plate
    rows never collide. Must run before finalization jobs are submitted so that
    concurrent jobs only append data (no metadata race conditions).
    """
    dataset = OpsDataset(experiment)
    iss_zarr = dataset.store_paths["iss_stitch"]
    registered_zarr = dataset.store_paths["iss_stitch_registered_v3"]

    # Read shape and channel info from the input zarr
    with open_ome_zarr(iss_zarr, mode="r", version="0.5") as store_in:
        data = store_in[wells[0]].data
        T, C, Z, Y, X = data.shape
        channel_names = store_in.channel_names

    if verbose:
        print(f"\nPre-creating registered zarr with {len(wells)} well positions...")
        print(f"  Shape: T={T}, C={C}, Z={Z}, Y={Y}, X={X}")
        print(f"  Path: {registered_zarr}")

    # Write OME-NGFF v0.4 (zarr v2); convert_iss_to_v3 (after merge) migrates this
    # to a proper v3 store (pyramids/metadata). base_calling auto-selects the
    # TensorStore driver by store format, and detect_spots reads via iohub.
    store_mode = "a" if registered_zarr.exists() else "w"
    with open_ome_zarr(registered_zarr, layout="hcs", mode=store_mode, channel_names=channel_names, version="0.4") as store_out:
        for well in wells:
            # Read shape per-well: wells differ by a few pixels in Y/X due to stitching boundary rounding
            with open_ome_zarr(iss_zarr, mode="r", version="0.5") as store_in:
                T, C, Z, Y, X = store_in[well].data.shape
            ensure_position_array(
                store_out,
                well,
                shape=(T, C, Z, Y, X),
                chunk_size=(1, 1, 1, 4096, 4096),
                dtype=np.float32,
                scale=[1, 1, 1, 1, 1],
            )
            if verbose:
                print(f"  Created position: {well} (Y={Y}, X={X})")

    if verbose:
        print(f"  Done.\n")

def precreate_iss_registered(experiment: str) -> None:
    """Public entry point for Nextflow: discover wells from iss_stitch zarr and pre-create
    bc_stitched_registered.zarr with all positions before the per-well finalize fan-out.
    """
    dataset = OpsDataset(experiment)
    iss_zarr = dataset.store_paths["iss_stitch"]

    with open_ome_zarr(iss_zarr, mode="r", version="0.5") as store:
        wells = sorted({
            "/".join(pos.split("/")[:2]) + "/0" for pos, _ in store.positions()
        })

    _precreate_registered_zarr(experiment, wells)


def _create_dapi10_spots_overlay(
    dapi10_stitched: Path,
    iss_zarr: Path,
    position: str,
    affine_dapi10_to_r9: np.ndarray,
    overlays_dir: Path,
    verbose: bool = False,
):
    """Create diagnostic overlay: DAPI_round10 spots → Round 9 spots (reusing iss_cycle_register visualization code)."""
    from cyclops_process.processes.auto_register.iss_cycle_register import create_iss_registration_overlay_custom_zarrs

    output_path = overlays_dir / "dapi10_spots_to_round9_spots.png"

    create_iss_registration_overlay_custom_zarrs(
        source_zarr=dapi10_stitched,
        target_zarr=iss_zarr,
        position=position,
        source_t_idx=0,  # DAPI_round10 is single-round
        target_t_idx=9,  # Round 9
        affine_3x3=affine_dapi10_to_r9,
        output_path=output_path,
        spot_channels=[1, 2, 3, 4],
    )

    if verbose:
        print(f"    ✓ Saved DAPI10→R9 spots overlay: {output_path.name}")


def check_for_dapi_round10(
    dataset: "OpsDataset",
    verbose: bool = True,
) -> bool:
    """
    Check if DAPI_round10 zarrs exist for this experiment.

    For experiments like ops0078 and ops0081 where DAPI was not imaged in Round 0,
    a separate DAPI_round10 acquisition contains nucleus + spots (same as Round 9).
    This will be registered to Round 9 spots during finalization.

    Parameters
    ----------
    dataset : OpsDataset
        Dataset object
    verbose : bool
        Print progress messages

    Returns
    -------
    bool
        True if DAPI_round10 zarrs exist
    """
    convert_dir = dataset.convert_in_situ
    dapi_round10_zarrs = list(convert_dir.glob("DAPI_round10*.zarr"))
    has_dapi_round10 = len(dapi_round10_zarrs) > 0

    if has_dapi_round10 and verbose:
        print(f"\n{'='*60}")
        print(f"DAPI_round10 detected ({len(dapi_round10_zarrs)} wells)")
        print(f"{'='*60}")
        print(f"DAPI_round10 nucleus will be registered to Round 9 spots")
        print(f"during finalization (after round-to-round registration).\n")

    return has_dapi_round10


def register_dapi_round10_to_round9(
    dataset: "OpsDataset",
    well: int,
    transforms_dir: Path,
    overlays_dir: Path,
    affines_cumulative: dict,
    verbose: bool = True,
) -> dict:
    """
    Register DAPI_round10 nucleus to Round 0 position via Round 9.

    Simplified approach mirroring register_round_pair:
    1. Extract spots from DAPI_round10 and Round 9 using fast subsampled extraction
    2. Apply manual pre-alignment affine to source spots
    3. Graph-based matching + RANSAC (same as round-to-round)
    4. Chain with cumulative Round 9 → Round 0 transform
    5. Save combined transform as nucleus_to_round0.yml

    Parameters
    ----------
    dataset : OpsDataset
        Dataset object
    well : int
        Well number
    transforms_dir : Path
        Directory for transform YAMLs
    overlays_dir : Path
        Directory for visualization overlays
    affines_cumulative : dict
        Dictionary of cumulative transforms (Round i → Round 0)
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Registration result with affine and paths
    """
    import time
    import re
    from cyclops_process.processes.auto_register.auto_register_utils import extract_spots_from_intensity_subsampled
    from cyclops_process.processes.auto_register.auto_register_ransac import estimate_affine_ransac
    from cyclops_process.processes.auto_register.auto_register_graph import match_cells_by_graph_consistency

    t_start = time.time()

    row, col = parse_well(well)
    well_token = f"{row}{col}"
    if verbose:
        print(f"\n{'='*60}")
        print(f"Registering DAPI_round10 → Round 9 for Well {well_token}")
        print(f"{'='*60}\n")

    # Find stitched DAPI_round10 zarr for this well
    stitch_dir = dataset.store_paths["iss_stitch"].parent
    position = f"{row}/{col}/0"

    dapi_round10_zarrs = list(stitch_dir.glob("DAPI_round10*_stitched.zarr"))
    dapi_round10_stitched = None
    for zarr_path in dapi_round10_zarrs:
        m = re.search(rf'{row}(\d+)', zarr_path.name)
        if m and int(m.group(1)) == col:
            dapi_round10_stitched = zarr_path
            break

    if dapi_round10_stitched is None:
        raise FileNotFoundError(
            f"Stitched DAPI_round10 zarr not found for well {well_token}. "
            f"Expected pattern: DAPI_round10_{well_token}_*_stitched.zarr in {stitch_dir}"
        )

    iss_zarr = dataset.store_paths["iss_stitch"]

    if verbose:
        print(f"  DAPI_round10: {dapi_round10_stitched.name}")
        print(f"  ISS stitch: {iss_zarr.name}")

    # Get manual pre-alignment affine (from biahub GUI registration)
    # Manual affines are well-specific since plate movement affects each well differently
    experiment_name = dataset.experiment
    well_num = int(position.split('/')[1])  # Extract well number from position "A/2/0"

    manual_affines = {
        ("ops0078_20250923", 1): np.array([
            [1.00004411e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, 1.00002408e+00, -6.32816646e-03, -8.87044922e+02],
            [0.00000000e+00, 6.32816646e-03, 1.00002408e+00, -7.25321899e+02],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ]),
        ("ops0078_20250923", 2): np.array([
            [9.99937296e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, 9.99917269e-01, -6.30051317e-03, -6.73531982e+02],
            [0.00000000e+00, 6.30051317e-03, 9.99917328e-01, -7.23516113e+02],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ]),
        ("ops0078_20250923", 3): np.array([
            [1.00003159e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, 1.00001276e+00, -6.14447333e-03, -4.66936188e+02],
            [0.00000000e+00, 6.14447333e-03, 1.00001264e+00, -7.20014954e+02],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ]),
        ("ops0081_20250924", 1): np.array([
            [9.99888182e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, 9.99888062e-01, 3.96397838e-04, -1.05083679e+03],
            [0.00000000e+00, -3.96397838e-04, 9.99888122e-01, -3.66904449e+02],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ]),
        ("ops0081_20250924", 2): np.array([
            [9.99981403e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, 9.99981165e-01, 6.27114787e-04, -6.69753113e+02],
            [0.00000000e+00, -6.27114787e-04, 9.99981225e-01, -3.62576569e+02],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ]),
        ("ops0081_20250924", 3): np.array([
            [9.99980330e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, 9.99979973e-01, 5.99339430e-04, -2.93168457e+02],
            [0.00000000e+00, -5.99339430e-04, 9.99980032e-01, -3.69565338e+02],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ]),
    }

    affine_key = (experiment_name, well_num)
    if affine_key not in manual_affines:
        raise ValueError(f"No manual affine configured for {experiment_name} well {well_num}")

    # Extract 2D affine from 4x4 biahub matrix
    # biahub saves inverse transforms (for ndi.affine_transform backward mapping)
    # For point coordinate transform with matrix multiplication, we need to invert
    manual_affine_4x4 = manual_affines[affine_key]
    manual_affine_4x4_inv = np.linalg.inv(manual_affine_4x4)
    manual_affine_fwd_3x3 = np.eye(3)
    manual_affine_fwd_3x3[:2, :2] = manual_affine_4x4_inv[1:3, 1:3]  # rotation/scale
    manual_affine_fwd_3x3[:2, 2] = manual_affine_4x4_inv[1:3, 3]      # translation (dy, dx)

    if verbose:
        print(f"\n  Step 0: Applying manual affine for {experiment_name} well {well_num}")
        print(f"    Manual shift (inverted, forward): dy={manual_affine_fwd_3x3[0, 2]:.1f}, dx={manual_affine_fwd_3x3[1, 2]:.1f}")

    # Use same parameters as register_round_pair
    params = DEFAULT_ISS_PARAMS.copy()
    params["spot_threshold"] = 400
    params["transform_type"] = "similarity"
    # Use standard search radius - manual affine provides good pre-alignment
    params["max_match_distance"] = 100.0

    # Extract spots from DAPI_round10 (source) using fast subsampled method
    if verbose:
        print(f"\n  Extracting spots from DAPI_round10 (source)...")
    t_extract_src = time.time()
    source_spots = extract_spots_from_intensity_subsampled(
        dapi_round10_stitched,
        position,
        t_idx=0,  # DAPI_round10 is single-round
        channel_indices=[1, 2, 3, 4],
        threshold=params["spot_threshold"],
        min_distance=params["spot_min_distance"],
        bins_to_select=params["subsample_bins_to_select"],
        grid_size=params["subsample_grid_size"],
        max_spots_per_bin=params["max_spots_per_bin"],
        cache_subdir="in_situ_sequencing/register/dapi10_spot_cache",
    )
    dt_extract_src = time.time() - t_extract_src
    if verbose:
        print(f"    Found {len(source_spots)} spots ({dt_extract_src:.2f}s)")
        if len(source_spots) > 0:
            print(f"    Source centroid: y={source_spots[:, 0].mean():.0f}, x={source_spots[:, 1].mean():.0f}")
            print(f"    Source range: y=[{source_spots[:, 0].min():.0f}, {source_spots[:, 0].max():.0f}], x=[{source_spots[:, 1].min():.0f}, {source_spots[:, 1].max():.0f}]")

    # Extract spots from Round 9 (target)
    if verbose:
        print(f"  Extracting spots from Round 9 (target)...")
    t_extract_tgt = time.time()
    target_spots = extract_spots_from_intensity_subsampled(
        iss_zarr,
        position,
        t_idx=9,  # Round 9
        channel_indices=[1, 2, 3, 4],
        threshold=params["spot_threshold"],
        min_distance=params["spot_min_distance"],
        bins_to_select=params["subsample_bins_to_select"],
        grid_size=params["subsample_grid_size"],
        max_spots_per_bin=params["max_spots_per_bin"],
        cache_subdir="in_situ_sequencing/register/iss_spot_cache",
    )
    dt_extract_tgt = time.time() - t_extract_tgt
    if verbose:
        print(f"    Found {len(target_spots)} spots ({dt_extract_tgt:.2f}s)")
        if len(target_spots) > 0:
            print(f"    Target centroid: y={target_spots[:, 0].mean():.0f}, x={target_spots[:, 1].mean():.0f}")
            print(f"    Target range: y=[{target_spots[:, 0].min():.0f}, {target_spots[:, 0].max():.0f}], x=[{target_spots[:, 1].min():.0f}, {target_spots[:, 1].max():.0f}]")

    # Show offset BEFORE manual alignment
    if verbose and len(source_spots) > 0 and len(target_spots) > 0:
        dy_before = source_spots[:, 0].mean() - target_spots[:, 0].mean()
        dx_before = source_spots[:, 1].mean() - target_spots[:, 1].mean()
        print(f"\n  BEFORE manual: offset dy={dy_before:.1f}, dx={dx_before:.1f}")

    # Apply manual affine pre-alignment to source spots
    source_spots_homog = np.column_stack([source_spots, np.ones(len(source_spots))])
    source_spots_aligned = (manual_affine_fwd_3x3 @ source_spots_homog.T).T[:, :2]

    if verbose:
        # Show offset after manual alignment
        dy_offset = source_spots_aligned[:, 0].mean() - target_spots[:, 0].mean()
        dx_offset = source_spots_aligned[:, 1].mean() - target_spots[:, 1].mean()
        print(f"  AFTER manual: offset dy={dy_offset:.1f}, dx={dx_offset:.1f}")
        if abs(dy_offset) > params["max_match_distance"] or abs(dx_offset) > params["max_match_distance"]:
            print(f"  ⚠️  WARNING: Offset exceeds search radius ({params['max_match_distance']}px)!")

    # Graph-based matching (same as register_round_pair)
    if verbose:
        print(f"\n  Graph-based matching...")
        print(f"    Source: {len(source_spots_aligned)} spots, Target: {len(target_spots)} spots")
        print(f"    Search radius: {params['max_match_distance']}px")
    t_match = time.time()

    # Create dummy Hu moments (spots don't have shape)
    source_hu = np.zeros((len(source_spots_aligned), 7))
    target_hu = np.zeros((len(target_spots), 7))

    # Weights: spatial only (same as register_round_pair)
    weights = {
        "hu": 0.0,
        "neighbor_hu": 0.0,
        "edge_length": 0.5,
        "angular_spacing": 0.4,
        "clustering": 0.1,
    }

    source_idx, target_idx, distances, _, _ = match_cells_by_graph_consistency(
        source_spots_aligned,
        target_spots,
        source_hu,
        target_hu,
        search_radius=params["max_match_distance"],
        k_neighbors=params["graph_k_neighbors"],
        top_k_candidates=params["graph_top_k_candidates"],
        weights=weights,
        max_score_threshold=100,
        min_matches_per_cell=0,
        min_total_matches=10,
        cache_dir=None,
        verbose=verbose,
    )

    dt_match = time.time() - t_match
    if verbose:
        print(f"    Found {len(source_idx)} matches ({dt_match:.2f}s)")

    # RANSAC affine estimation (same as register_round_pair)
    min_matches_required = params["min_samples"]
    if len(source_idx) < min_matches_required:
        if verbose:
            print(f"    WARNING: Insufficient matches ({len(source_idx)} < {min_matches_required})")
            print(f"    Using manual affine only (no RANSAC refinement)")
        ransac_affine_3x3 = np.eye(3)
        metrics = {"n_matches": len(source_idx), "n_inliers": 0, "inlier_ratio": 0.0}
    else:
        if verbose:
            print(f"  RANSAC affine estimation...")
        t_ransac = time.time()
        ransac_affine_3x3, inliers, metrics = estimate_affine_ransac(
            source_spots_aligned[source_idx],
            target_spots[target_idx],
            params["min_samples"],
            params["residual_threshold"],
            params["max_trials"],
            params["stop_probability"],
            params["transform_type"],
        )
        dt_ransac = time.time() - t_ransac

        if verbose:
            print(f"    RANSAC: {metrics['n_inliers']}/{metrics['n_matches']} inliers ({dt_ransac:.2f}s)")
            print(f"    Inlier ratio: {metrics['inlier_ratio']:.2%}")
            print(f"    Residual: {metrics['residual_mean']:.2f} ± {metrics['residual_std']:.2f} px")

    # Compose transforms: T_final = T_ransac @ T_manual
    # (RANSAC was fit on manually-aligned points)
    affine_dapi10_to_r9 = ransac_affine_3x3 @ manual_affine_fwd_3x3

    # Extract final transform parameters
    final_trans = affine_dapi10_to_r9[:2, 2]
    if verbose:
        print(f"\n  Final DAPI10→R9 transform:")
        print(f"    Translation: dy={final_trans[0]:.2f}, dx={final_trans[1]:.2f}")

    # Generate validation overlays
    if verbose:
        print(f"\n  Generating validation overlays...")
    try:
        from PIL import Image, ImageDraw, ImageFont
        import zarr

        # Get image dimensions from ISS zarr
        iss_store = zarr.open(str(iss_zarr), mode='r')
        iss_data = iss_store[position]
        if isinstance(iss_data, zarr.Group) and '0' in iss_data:
            iss_data = iss_data['0']
        full_img_shape = iss_data.shape[-2:]  # (Y, X)

        # Downsample 8x for visualization
        ds_viz = 8
        img_shape = (full_img_shape[0] // ds_viz, full_img_shape[1] // ds_viz)

        # Calculate offsets for labels
        dy_before = source_spots[:, 0].mean() - target_spots[:, 0].mean()
        dx_before = source_spots[:, 1].mean() - target_spots[:, 1].mean()
        dy_after_manual = source_spots_aligned[:, 0].mean() - target_spots[:, 0].mean()
        dx_after_manual = source_spots_aligned[:, 1].mean() - target_spots[:, 1].mean()

        # Apply final transform to original source spots for "after RANSAC" overlay
        source_spots_final_homog = np.column_stack([source_spots, np.ones(len(source_spots))])
        source_spots_final = (affine_dapi10_to_r9 @ source_spots_final_homog.T).T[:, :2]
        dy_after_ransac = source_spots_final[:, 0].mean() - target_spots[:, 0].mean()
        dx_after_ransac = source_spots_final[:, 1].mean() - target_spots[:, 1].mean()

        # Create spot masks for overlay - use larger markers (5x5) so they're visible
        def create_spot_mask(spots, img_shape, ds, marker_radius=2):
            mask = np.zeros(img_shape, dtype=np.uint8)
            for spot in spots:  # Use all spots (already subsampled to ~5000)
                y, x = int(spot[0] / ds), int(spot[1] / ds)
                # Draw a small filled circle for visibility
                for dy in range(-marker_radius, marker_radius + 1):
                    for dx in range(-marker_radius, marker_radius + 1):
                        if dy*dy + dx*dx <= marker_radius*marker_radius:
                            yy, xx = y + dy, x + dx
                            if 0 <= yy < img_shape[0] and 0 <= xx < img_shape[1]:
                                mask[yy, xx] = 255
            return mask

        # Overlay 1: Before vs After Manual alignment
        source_mask_before = create_spot_mask(source_spots, img_shape, ds_viz)
        source_mask_manual = create_spot_mask(source_spots_aligned, img_shape, ds_viz)
        target_mask = create_spot_mask(target_spots, img_shape, ds_viz)

        before = np.stack([target_mask, source_mask_before, np.zeros_like(target_mask)], axis=-1)
        after_manual = np.stack([target_mask, source_mask_manual, np.zeros_like(target_mask)], axis=-1)
        combined1 = np.hstack([before, after_manual])
        combined1_pil = Image.fromarray(combined1)

        draw1 = ImageDraw.Draw(combined1_pil)
        try:
            font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 40)
        except:
            font = ImageFont.load_default()
        draw1.text((10, 10), f"BEFORE (dy={dy_before:.0f}, dx={dx_before:.0f})", fill=(255, 255, 0), font=font)
        draw1.text((img_shape[1] + 10, 10), f"AFTER MANUAL (dy={dy_after_manual:.0f}, dx={dx_after_manual:.0f})", fill=(255, 255, 0), font=font)
        combined1_pil.save(overlays_dir / "01_dapi10_manual_alignment.png")

        # Overlay 2: After Manual vs After RANSAC
        source_mask_final = create_spot_mask(source_spots_final, img_shape, ds_viz)
        after_ransac = np.stack([target_mask, source_mask_final, np.zeros_like(target_mask)], axis=-1)
        combined2 = np.hstack([after_manual, after_ransac])
        combined2_pil = Image.fromarray(combined2)

        draw2 = ImageDraw.Draw(combined2_pil)
        draw2.text((10, 10), f"MANUAL (dy={dy_after_manual:.0f}, dx={dx_after_manual:.0f})", fill=(255, 255, 0), font=font)
        draw2.text((img_shape[1] + 10, 10), f"MANUAL+RANSAC (dy={dy_after_ransac:.0f}, dx={dx_after_ransac:.0f})", fill=(255, 255, 0), font=font)
        combined2_pil.save(overlays_dir / "02_dapi10_ransac_refinement.png")

        if verbose:
            print(f"    ✓ Saved: 01_dapi10_manual_alignment.png")
            print(f"    ✓ Saved: 02_dapi10_ransac_refinement.png")

    except Exception as e:
        if verbose:
            print(f"    ⚠️  Overlay generation failed: {e}")
            import traceback
            traceback.print_exc()

    # Chain with Round 9 → Round 0 cumulative transform
    if 9 not in affines_cumulative:
        raise ValueError("Round 9 cumulative transform not found. Run round-to-round registration first.")

    # Compute R9→R0 by removing the R0→Seg portion from R9→Seg
    # affines_cumulative[9] = R9→Seg, affines_cumulative[0] = R0→Seg
    # R9→R0 = R9→Seg @ Seg→R0 = cumulative[9] @ inv(cumulative[0])
    affine_r0_to_seg = affines_cumulative[0]
    affine_seg_to_r0 = np.linalg.inv(affine_r0_to_seg)
    affine_r9_to_seg = affines_cumulative[9]
    affine_r9_to_r0 = affine_r9_to_seg @ affine_seg_to_r0

    # Compose DAPI10→R9 with R9→R0 to get DAPI10→R0
    affine_dapi10_to_r0 = affine_r9_to_r0 @ affine_dapi10_to_r9

    if verbose:
        r9_trans = affine_r9_to_r0[:2, 2]
        total_trans = affine_dapi10_to_r0[:2, 2]
        print(f"\n  [DEBUG] R9→Seg cumulative: dy={affine_r9_to_seg[0,2]:.2f}, dx={affine_r9_to_seg[1,2]:.2f}")
        print(f"  [DEBUG] R0→Seg cumulative: dy={affine_r0_to_seg[0,2]:.2f}, dx={affine_r0_to_seg[1,2]:.2f}")
        print(f"  [DEBUG] Computed R9→R0: dy={r9_trans[0]:.2f}, dx={r9_trans[1]:.2f}")
        print(f"  Total DAPI10→Round 0: dy={total_trans[0]:.2f}, dx={total_trans[1]:.2f}")

    # Save as nucleus_to_round0.yml (biahub convention: save inverse)
    # Computed: DAPI10→R0 (forward), Saved: R0→DAPI10 (inverse)
    affine_4x4 = affine_3x3_to_4x4_zyx(affine_dapi10_to_r0)
    affine_4x4_inv = np.linalg.inv(affine_4x4)
    output_yaml = transforms_dir / "nucleus_to_round0.yml"
    save_affine_to_yaml(affine_4x4_inv, output_yaml)

    dt_total = time.time() - t_start
    if verbose:
        print(f"\n  [DEBUG] Saved transform (R0→DAPI10 inverse):")
        print(f"    YX translation: [{affine_4x4_inv[1,3]:.2f}, {affine_4x4_inv[2,3]:.2f}]")
        print(f"  ✓ Saved: {output_yaml.name} ({dt_total:.1f}s total)")
        print(f"{'='*60}\n")

    return {
        "success": True,
        "affine_3x3": affine_dapi10_to_r0,
        "affine_dapi10_to_r9": affine_dapi10_to_r9,
        "affine_r9_to_r0": affine_r9_to_r0,
        "yaml_path": output_yaml,
        "metrics": metrics,
    }


@versioned_function("v1.0")
def job_register_nucleus_to_round0(
    experiment: str,
    well: int,
    spot_threshold: float = 400,
    nucleus_threshold: float = 200,
    transform_type: str = "similarity",
    max_distance: float = 200,
    verbose: bool = True,
    reproducible: bool = False,
    spots_round: int = 0,
):
    """Job function: Register nucleus → anchor spots round (default Round 0)."""
    import json

    if reproducible:
        _set_reproducible_seed()

    dataset = OpsDataset(experiment)
    row, col = parse_well(well)
    well_token = f"{row}{col}"
    position = f"{row}/{col}/0"

    iss_zarr = dataset.store_paths["iss_stitch"]
    seg_zarr = dataset.store_paths["iss_segmentation"]

    register_root = dataset.preprocess_in_situ / "register"
    transforms_dir = register_root / f"transforms/{well_token}"
    overlays_dir = register_root / f"overlays/{well_token}"
    metrics_dir = register_root / "metrics"

    transforms_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Setup parameters
    params = DEFAULT_ISS_PARAMS.copy()
    params["spot_threshold"] = spot_threshold
    params["nucleus_threshold"] = nucleus_threshold
    params["transform_type"] = transform_type
    params["max_distance"] = max_distance

    result = register_nucleus_to_round0(
        iss_zarr, seg_zarr, position, params,
        transforms_dir, overlays_dir, dataset=dataset, verbose=verbose,
        spots_round=spots_round,
    )

    # Save metrics to JSON for later aggregation
    if "metrics" in result and result["metrics"]:
        metrics_file = metrics_dir / f"{well_token}_nucleus_to_round0_metrics.json"
        # Convert numpy types to Python native for JSON serialization
        import math

        def _json_safe(v):
            if isinstance(v, (np.floating, np.integer)):
                v = float(v)
            if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
                return str(v)
            return v
        serializable_metrics = {k: _json_safe(v) for k, v in result["metrics"].items()}
        with open(metrics_file, "w") as f:
            json.dump(serializable_metrics, f, indent=2)
        if verbose:
            print(f"  Saved metrics: {metrics_file.name}")

    if verbose:
        method = result.get("metrics", {}).get("method", "unknown")
        print(f"\n✓ Nucleus → Round 0 registration complete (method: {method})")
        print(f"  Affine: {result.get('yaml', result.get('transform_yaml', 'identity'))}")
        if "bimodal" in method:
            print(f"  NOTE: Bimodal tiles detected — noise tiles filtered before PCC averaging")

    return result


@versioned_function("v1.0")
def job_register_segmentation_to_nucleus(experiment: str, well, verbose: bool = True, reproducible: bool = False):
    """SLURM job wrapper for segmentation → nucleus registration."""
    if reproducible:
        _set_reproducible_seed()
    from cyclops_utils.data.experiment import OpsDataset

    ds = OpsDataset(experiment)
    row, col = parse_well(well)
    well_token = f"{row}{col}"
    position = f"{row}/{col}/0"

    iss_zarr = ds.store_paths["iss_stitch"]
    seg_zarr = ds.store_paths["iss_segmentation"]

    register_root = ds.preprocess_in_situ / "register"
    transforms_dir = register_root / f"transforms/{well_token}"
    overlays_dir = register_root / f"overlays/{well_token}"
    transforms_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    result = register_segmentation_to_nucleus(
        iss_zarr,
        seg_zarr,
        position, transforms_dir, overlays_dir, verbose=verbose
    )

    if verbose:
        print(f"✓ Segmentation→nucleus complete: {result['yaml_path'].name}")

    return result


@versioned_function("v1.0")
def job_register_round_pair(
    experiment: str,
    well: int,
    round_source: int,
    round_target: int,
    spot_threshold: float = 400,
    transform_type: str = "similarity",
    verbose: bool = True,
    reproducible: bool = False,
):
    """Job function: Register a single round pair (source → target)."""
    import json

    if reproducible:
        _set_reproducible_seed()

    dataset = OpsDataset(experiment)
    row, col = parse_well(well)
    well_token = f"{row}{col}"
    position = f"{row}/{col}/0"

    iss_zarr = dataset.store_paths["iss_stitch"]

    register_root = dataset.preprocess_in_situ / "register"
    transforms_dir = register_root / f"transforms/{well_token}"
    overlays_dir = register_root / f"overlays/{well_token}"
    metrics_dir = register_root / "metrics"

    transforms_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Setup parameters
    params = DEFAULT_ISS_PARAMS.copy()
    params["spot_threshold"] = spot_threshold
    params["transform_type"] = transform_type

    result = register_round_pair(
        iss_zarr, position, round_target, round_source,
        params, transforms_dir, overlays_dir, verbose=verbose
    )

    # Save metrics to JSON for later aggregation
    if "metrics" in result and result["metrics"]:
        metrics_file = metrics_dir / f"{well_token}_round{round_source}_to_round{round_target}_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(result["metrics"], f, indent=2)
        if verbose:
            print(f"  Saved metrics: {metrics_file.name}")

    if verbose:
        print(f"\n✓ Round {round_source} → Round {round_target} registration complete")
        print(f"  Affine: {result.get('yaml', result.get('transform_yaml', 'identity'))}")

    return result


def job_finalize_registration(
    experiment: str,
    well: int,
    verbose: bool = True,
    skip_apply_transforms: bool = False,
    reproducible: bool = False,
    n_rounds: int = 9,
    anchor_round: int = 0,
):
    """Job function: Compose cumulative transforms and generate final visualizations.

    When ``skip_apply_transforms=True`` we still compose the per-round cumulative
    transform YAMLs (the merge step needs them) but skip the
    ``apply_iss_transforms`` call that writes ``bc_stitched_registered.zarr``.
    The merge step (``merge_spots_base_calling``) will warp the array in-memory
    instead.
    """
    import time
    t_total_start = time.time()

    if reproducible:
        _set_reproducible_seed()

    dataset = OpsDataset(experiment)
    row, col = parse_well(well)
    well_token = f"{row}{col}"
    position = f"{row}/{col}/0"

    iss_zarr = dataset.store_paths["iss_stitch"]
    seg_zarr = dataset.store_paths["iss_segmentation"]

    register_root = dataset.preprocess_in_situ / "register"
    transforms_dir = register_root / f"transforms/{well_token}"
    overlays_dir = register_root / f"overlays/{well_token}"
    metrics_dir = register_root / "metrics"

    metrics_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'='*80}")
        print(f"Finalizing ISS Registration: {experiment} Well {well_token}")
        print(f"{'='*80}\n")

    # Load all sequential affines
    import yaml
    t_load_start = time.time()

    sequential_affines = {}

    # Load segmentation→nucleus affine (new first step)
    seg_yaml = transforms_dir / "segmentation_to_nucleus.yaml"
    if seg_yaml.exists():
        with open(seg_yaml) as f:
            seg_data = yaml.safe_load(f)
            # YAML stores inverse (nucleus→seg), so load and invert to get seg→nucleus
            affine_4x4_inv = np.array(seg_data["affine_transform_zyx"])
            affine_4x4 = np.linalg.inv(affine_4x4_inv)
            # Convert 4x4 back to 3x3
            affine_3x3 = np.eye(3)
            affine_3x3[:2, :2] = affine_4x4[1:3, 1:3]
            affine_3x3[:2, 2] = affine_4x4[1:3, 3]
            sequential_affines[-2] = affine_3x3  # Index -2 for segmentation
            if verbose:
                print(f"  Loaded segmentation_to_nucleus: dy={affine_3x3[0,2]:.2f}, dx={affine_3x3[1,2]:.2f}")
    else:
        if verbose:
            print(f"  No segmentation_to_nucleus.yaml found - skipping seg alignment")

    # Load nucleus affine (if it exists - may be skipped for experiments without pre-DAPI round)
    # IMPORTANT: Skip loading if DAPI_round10 will be registered (it will compute fresh nucleus transform)
    has_dapi_round10 = check_for_dapi_round10(dataset, verbose=False)
    nucleus_yaml = transforms_dir / "nucleus_to_round0.yml"

    if nucleus_yaml.exists() and not has_dapi_round10:
        with open(nucleus_yaml) as f:
            nucleus_data = yaml.safe_load(f)
            # YAML stores R0→DAPI10 (inverse, same convention as round-to-round)
            # Use directly as spots→nucleus (which IS R0→DAPI10 semantically)
            affine_4x4 = np.array(nucleus_data["affine_transform_zyx"])
            if verbose:
                print(f"\n  [DEBUG] Loaded nucleus_to_round0.yml (R0→DAPI10):")
                print(f"    YX translation: [{affine_4x4[1,3]:.2f}, {affine_4x4[2,3]:.2f}]")
            # Convert 4x4 back to 3x3: extract YX rotation and translation
            affine_3x3 = np.eye(3)
            affine_3x3[:2, :2] = affine_4x4[1:3, 1:3]  # YX rotation
            affine_3x3[:2, 2] = affine_4x4[1:3, 3]     # YX translation
            sequential_affines[-1] = affine_3x3
            if verbose:
                print(f"  Loaded nucleus_to_round0: dy={affine_3x3[0,2]:.2f}, dx={affine_3x3[1,2]:.2f}")
    elif has_dapi_round10:
        # Will compute fresh nucleus transform from DAPI_round10 registration
        if verbose:
            print(f"  Skipping nucleus_to_round0.yml (DAPI_round10 will compute fresh transform)")
    else:
        raise FileNotFoundError(
            f"nucleus_to_round0.yml not found at {nucleus_yaml}. "
            f"Spots-to-nucleus registration is required — re-run register_iss_cycles."
        )

    # Number of physical ISS rounds from the stitched store's T dimension (cheap
    # metadata read) — 11 for ops0173, so round 10 gets a cumulative transform.
    n_rounds = _get_n_physical_rounds(dataset, well, experiment, verbose=verbose)

    # Load round-to-round affines. With an anchor round > 0 the leading rounds
    # (e.g. a no-incorporation round 0) have no spots, so no round{i}_to_round{i-1}
    # pairs exist for i <= anchor_round — start the chain at anchor_round+1.
    for i in range(anchor_round + 1, n_rounds):
        yaml_file = transforms_dir / f"round{i}_to_round{i-1}.yml"
        if yaml_file.exists():
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
                # Load the inverse affine and invert it back to get forward transform
                affine_4x4_inv = np.array(data["affine_transform_zyx"])
                affine_4x4 = np.linalg.inv(affine_4x4_inv)
                # Convert 4x4 back to 3x3: extract YX rotation and translation
                affine_3x3 = np.eye(3)
                affine_3x3[:2, :2] = affine_4x4[1:3, 1:3]  # YX rotation
                affine_3x3[:2, 2] = affine_4x4[1:3, 3]     # YX translation
                sequential_affines[i] = affine_3x3
                if verbose:
                    print(f"  Loaded round{i}_to_round{i-1}: dy={affine_3x3[0,2]:.2f}, dx={affine_3x3[1,2]:.2f}")
        else:
            if verbose:
                print(f"  WARNING: Missing {yaml_file.name}")

    t_load_elapsed = time.time() - t_load_start
    if verbose:
        print(f"\nLoaded {len(sequential_affines)} sequential transforms ({t_load_elapsed:.2f}s)")

    # Validate all round-to-round transforms are present
    expected_rounds = set(range(anchor_round + 1, n_rounds))
    loaded_rounds = set(i for i in range(anchor_round + 1, n_rounds) if i in sequential_affines)
    missing_rounds = expected_rounds - loaded_rounds
    if missing_rounds:
        missing_files = [f"round{i}_to_round{i-1}.yml" for i in sorted(missing_rounds)]
        raise ValueError(
            f"Missing {len(missing_rounds)} round-to-round transforms in {transforms_dir}: "
            f"{missing_files}. "
            f"Registration jobs may have failed upstream - check registration logs."
        )

    # Compose cumulative transforms (all rounds → segmentation anchor)
    t_compose_start = time.time()
    # Chain: segmentation → nucleus → spots → rounds

    affines_cumulative = {}

    # 1. Establish Anchor Chain
    # T_nuc_to_seg (Nucleus -> Segmentation)
    if -2 in sequential_affines:
        # Loaded transform is Seg->Nucleus (because YAML stored Nuc->Seg and loading inverted it)
        # Wait, user says "surely -8.8 (can confirm visually)".
        # In 24929640_0_log.out:
        #   Loaded segmentation_to_nucleus: dy=-8.80, dx=2.20
        #   Nucleus→Segmentation: dy=8.80, dx=-2.20  (This is result of INVERSION)
        # User says "its positive 8.8 in the cumulative affine but it surely -8.8"
        # So we should NOT invert here if we want to keep -8.8.
        #
        # Let's use the loaded transform directly (assuming loading logic is now consistent with saving)
        affine_nuc_to_seg = sequential_affines[-2]
        affines_cumulative[-2] = np.eye(3)  # Segmentation is anchor
        if verbose:
            print(f"\nAnchor: Segmentation")
            print(f"  Nucleus→Segmentation: dy={affine_nuc_to_seg[0, 2]:.2f}, dx={affine_nuc_to_seg[1, 2]:.2f}")
    else:
        affine_nuc_to_seg = np.eye(3)
        if verbose:
            print(f"\nAnchor: Nucleus (Segmentation not available)")

    # T_spots_to_nuc (Spots -> Nucleus)
    if -1 in sequential_affines:
        # sequential_affines[-1] is already Spots->Nucleus (loaded from YAML at line 224)
        # Use it directly - no inversion needed
        affine_spots_to_nuc = sequential_affines[-1]
        if verbose:
            print(f"  Spots→Nucleus: dy={affine_spots_to_nuc[0, 2]:.2f}, dx={affine_spots_to_nuc[1, 2]:.2f}")
    else:
        affine_spots_to_nuc = np.eye(3)

    # Store Cumulative Transforms
    # Key -1: Nucleus (Round 0, ch0) -> Anchor
    affines_cumulative[-1] = affine_nuc_to_seg

    # Key anchor_round: anchor spots round (ch1-4) -> Anchor
    # nucleus_to_round0.yml holds spots(anchor_round)→nucleus, so:
    # T_anchorspots_to_anchor = T_nuc_to_seg @ T_spots_to_nuc
    affines_cumulative[anchor_round] = affine_nuc_to_seg @ affine_spots_to_nuc

    if verbose:
        t0 = affines_cumulative[anchor_round]
        print(f"  Cumulative Round {anchor_round} (Spots)→Anchor: dy={t0[0, 2]:.2f}, dx={t0[1, 2]:.2f}")

    # Rounds (anchor+1)..(n_rounds-1): compose sequential transforms from the anchor
    for i in range(anchor_round + 1, n_rounds):
        if i not in sequential_affines:
            if verbose:
                print(f"  WARNING: Missing transform for Round {i}")
            continue

        # Compose: Round i → ... → anchor spots → Anchor
        # T_i_to_anchor = T_i-1_to_anchor @ T_i_to_i-1
        # sequential_affines[i] is T_i_to_i-1
        affines_cumulative[i] = affines_cumulative[i - 1] @ sequential_affines[i]

    # Leading rounds before the anchor (e.g. a no-incorporation round 0) carry no
    # usable spots and are excluded from base calling; alias them to the anchor so
    # downstream array-warping has a valid (if unused) transform per round.
    for r in range(0, anchor_round):
        affines_cumulative[r] = affines_cumulative[anchor_round].copy()

    # Handle DAPI_round10 registration if present
    has_dapi_round10 = check_for_dapi_round10(dataset, verbose=False)
    if has_dapi_round10 and anchor_round != 0:
        # DAPI_round10's nucleus rebasing assumes round 0 is the spots anchor;
        # combining it with anchor_round>0 is not supported.
        if verbose:
            print(f"\n  WARNING: anchor_round={anchor_round} with DAPI_round10 is not "
                  f"supported — skipping DAPI_round10 rebasing.")
        has_dapi_round10 = False
    if has_dapi_round10:
        try:
            if verbose:
                print(f"\nProcessing DAPI_round10 registration...")

            result = register_dapi_round10_to_round9(
                dataset=dataset,
                well=well,
                transforms_dir=transforms_dir,
                overlays_dir=overlays_dir,
                affines_cumulative=affines_cumulative,
                verbose=verbose,
            )

            # Nucleus is FIXED at DAPI10 position (segmentation space)
            # Nucleus cumulative should be IDENTITY (nucleus doesn't move)
            # affines_cumulative[-1] represents where nucleus is in segmentation space
            affines_cumulative[-1] = np.eye(3)  # Nucleus is the anchor (identity)

            # Now update Round 0 to include the transform TO the nucleus
            # result["affine_3x3"] = DAPI10→R0, but we need R0→DAPI10 (inverse)
            affine_dapi10_to_r0 = result["affine_3x3"]
            affine_r0_to_dapi10 = np.linalg.inv(affine_dapi10_to_r0)

            # Update R0: Spots need to move TO nucleus position
            # R0→Seg = R0→DAPI10 (since DAPI10 = Seg)
            affines_cumulative[0] = affine_r0_to_dapi10

            # CRITICAL: Rebuild cumulative chain for rounds 1-(n_rounds-1) using new R0 base
            for i in range(1, n_rounds): # beta/nextflow: n_rounds + 1 (?)
                if i in sequential_affines:
                    # Compose: Round i → Round 0 → Anchor
                    affines_cumulative[i] = affines_cumulative[i - 1] @ sequential_affines[i]

            if verbose:
                print(f"  ✓ DAPI_round10 nucleus registered successfully")
                print(f"  Nucleus position (identity): dy={affines_cumulative[-1][0, 2]:.2f}, dx={affines_cumulative[-1][1, 2]:.2f}")
                print(f"  Round 0 Spots→Nucleus: dy={affines_cumulative[0][0, 2]:.2f}, dx={affines_cumulative[0][1, 2]:.2f}")
                print(f"  Round 9→Anchor: dy={affines_cumulative[9][0, 2]:.2f}, dx={affines_cumulative[9][1, 2]:.2f}")
        except Exception as e:
            if verbose:
                print(f"\n  ✗ DAPI_round10 registration failed!")
                print(f"  Error type: {type(e).__name__}")
                print(f"  Error message: {e}")
                print(f"\n  Full traceback:")
                import traceback
                traceback.print_exc()
                print(f"\n  Continuing finalization without DAPI_round10 registration...")
            # Don't fail the entire finalization if DAPI_round10 fails
            pass

    # Save cumulative transforms (biahub convention: save inverse)
    for r, affine_3x3 in affines_cumulative.items():
        affine_4x4 = affine_3x3_to_4x4_zyx(affine_3x3)
        affine_4x4_inv = np.linalg.inv(affine_4x4)

        if r == -2:
            output_yaml = transforms_dir / "segmentation_to_round0_cumulative.yaml"
        elif r == -1:
            output_yaml = transforms_dir / "nucleus_to_round0_cumulative.yaml"
        elif r == 0:
            output_yaml = transforms_dir / "round0_to_round0_cumulative.yaml"
        else:
            output_yaml = transforms_dir / f"round{r}_to_round0_cumulative.yaml"

        save_affine_to_yaml(affine_4x4_inv, output_yaml)

    t_compose_elapsed = time.time() - t_compose_start
    if verbose:
        anchor_name = "segmentation" if -2 in affines_cumulative else ("nucleus" if -1 in affines_cumulative else "Round 0")
        print(f"\n✓ Saved {len(affines_cumulative)} cumulative transforms ({t_compose_elapsed:.2f}s)")
        print(f"\nCumulative translations (relative to {anchor_name} anchor):")
        if -2 in affines_cumulative:
            print(f"    Segmentation: dy=0.0, dx=0.0 (anchor)")
        if -1 in affines_cumulative:
            cumul_trans = affines_cumulative[-1][:2, 2]
            print(f"    Nucleus: dy={cumul_trans[0]:.1f}, dx={cumul_trans[1]:.1f}")
        for i in range(n_rounds):
            if i in affines_cumulative:
                cumul_trans = affines_cumulative[i][:2, 2]  # dy, dx from 3x3 matrix
                print(f"    Round {i}: dy={cumul_trans[0]:.1f}, dx={cumul_trans[1]:.1f}")

        # Debug: Check if Round 0 is actually NOT identity
        if 0 in affines_cumulative:
            is_identity = np.allclose(affines_cumulative[0], np.eye(3), atol=1e-6)
            print(f"\n  DEBUG: Round 0 is_identity = {is_identity}")
            print(f"  DEBUG: Round 0 affine:\n{affines_cumulative[0]}")

    # Run visualizations in parallel (thread-safe: uses matplotlib Figure OO API, not pyplot)
    # while transforms run in the main thread
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    drift_plot_path = overlays_dir / "drift_trajectory.png"
    all_rounds_overlay_path = overlays_dir / "all_rounds_overlay.png"
    all_rounds_nucleus_overlay_path = overlays_dir / "all_rounds_overlay_with_dapi.png"
    final_overlay_path = overlays_dir / "final_registration_with_segmentation.png"

    viz_tasks = [
        ("drift_plot", lambda: create_drift_trajectory_plot(affines_cumulative, drift_plot_path)),
        ("all_rounds_overlay", lambda: create_all_rounds_overlay(
            iss_zarr, position, affines_cumulative, all_rounds_overlay_path,
            crop_size=500, n_crops=6)),
        ("all_rounds_nucleus", lambda: create_all_rounds_overlay(
            iss_zarr, position, affines_cumulative, all_rounds_nucleus_overlay_path,
            crop_size=500, n_crops=6, include_nucleus=True)),
        ("final_3d", lambda: create_all_rounds_overlay(
            iss_zarr, position, affines_cumulative, final_overlay_path,
            crop_size=500, n_crops=6, include_nucleus=True,
            include_segmentation=True, seg_zarr_path=seg_zarr)),
    ]

    def run_all_visualizations():
        t_viz_start = time.time()
        if verbose:
            print(f"\n[Visualizations] Starting in parallel with transforms...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(fn): name for name, fn in viz_tasks}
            for future in as_completed(futures):
                viz_name = futures[future]
                try:
                    future.result()
                    if verbose:
                        print(f"  [Visualizations] ✓ Saved: {viz_name}")
                except Exception as e:
                    if verbose:
                        print(f"  [Visualizations] WARNING: {viz_name} failed: {e}")
        t_viz_elapsed = time.time() - t_viz_start
        if verbose:
            print(f"\n  [Visualizations] Complete ({t_viz_elapsed:.2f}s)")

    # Start visualizations in background thread
    t_parallel_start = time.time()
    viz_thread = threading.Thread(target=run_all_visualizations, daemon=False)
    viz_thread.start()

    # Apply transforms in main thread (runs concurrently with visualizations)
    t_apply_start = time.time()
    if verbose:
        print(f"\n[Transforms] Applying transforms to create registered zarr...")

    registered_zarr = dataset.store_paths["iss_stitch_registered_v3"]

    # Calculate required padding based on max shift magnitude
    max_shift = 0
    for affine in affines_cumulative.values():
        shift_y = abs(affine[0, 2])
        shift_x = abs(affine[1, 2])
        max_shift = max(max_shift, shift_y, shift_x)

    # Padding must cover max shift + safety margin (20%)
    required_padding = int(np.ceil(max_shift * 1.2))
    required_padding = max(required_padding, 256)  # Minimum 256px

    if verbose and max_shift > 256:
        print(f"  Large shifts detected: max={max_shift:.0f}px")
        print(f"  Using padding={required_padding}px (default=256px)")

    if skip_apply_transforms:
        if verbose:
            print(f"  [Transforms] skip_apply_transforms=True — leaving registered zarr "
                  f"to be produced by the downstream merge step")
        t_apply_elapsed = 0.0
    else:
        try:
            apply_iss_transforms(
                iss_zarr,
                position,
                affines_cumulative,
                registered_zarr,
                padding=required_padding,
                verbose=verbose,
            )
            t_apply_elapsed = time.time() - t_apply_start
            if verbose:
                print(f"  [Transforms] ✓ Saved: {registered_zarr} ({t_apply_elapsed:.2f}s)")
        except Exception as e:
            print(f"  [Transforms] ERROR: Registered zarr creation failed: {e}")
            import traceback
            traceback.print_exc()
            raise  # Don't swallow - let submitit know the job failed

    # Wait for visualizations to complete
    if verbose:
        print(f"\n[Main] Waiting for visualizations to complete...")
    viz_thread.join()

    t_parallel_elapsed = time.time() - t_parallel_start
    if verbose:
        print(f"\n[Main] Both visualizations and transforms complete ({t_parallel_elapsed:.2f}s total)")

    # Ensure variable exists even if all_rounds_overlay failed
    if not all_rounds_overlay_path.exists():
        all_rounds_overlay_path = None

    # Symlink segmentation into registered zarr (only when the zarr exists)
    t_symlink_start = time.time()
    if not skip_apply_transforms and registered_zarr is not None and seg_zarr is not None:
        try:
            if verbose:
                print(f"\nSymlinking segmentation labels into registered zarr...")

            # Import the symlink function from ops_stitch
            from cyclops_process.processes.ops_stitch import _attach_seg_labels_symlink

            _attach_seg_labels_symlink(
                seg_source_store=str(seg_zarr),
                assembled_store=str(registered_zarr),
                label_name="nuclear_seg"
            )

            t_symlink_elapsed = time.time() - t_symlink_start
            if verbose:
                print(f"  ✓ Segmentation symlinks created ({t_symlink_elapsed:.2f}s)")
        except Exception as e:
            if verbose:
                print(f"  WARNING: Segmentation symlinking failed: {e}")
                import traceback
                traceback.print_exc()

    # Aggregate metrics from individual JSON files into CSV
    _aggregate_metrics_to_csv(metrics_dir, well, n_rounds=n_rounds, verbose=verbose)

    t_total_elapsed = time.time() - t_total_start
    if verbose:
        print(f"\n{'='*80}")
        print(f"Finalization Complete - Total time: {t_total_elapsed:.2f}s ({t_total_elapsed/60:.1f}m)")
        print(f"{'='*80}\n")

    return {
        "affines_cumulative": affines_cumulative,
        "drift_plot": drift_plot_path,
        "all_rounds_overlay": all_rounds_overlay_path,
        "registered_zarr": registered_zarr,
    }


def _aggregate_metrics_to_csv(metrics_dir: Path, well, n_rounds: int = 10, verbose: bool = True):
    """
    Aggregate metrics from individual JSON files into a single CSV.

    Collects all *_metrics.json files for a well and saves to
    registration_metrics_<row><col>.csv in the same format as the non-SLURM version.
    """
    import json
    import pandas as pd

    row, col = parse_well(well)
    well_token = f"{row}{col}"
    rows = []

    # Nucleus metrics
    nucleus_file = metrics_dir / f"{well_token}_nucleus_to_round0_metrics.json"
    if nucleus_file.exists():
        with open(nucleus_file) as f:
            nuc_metrics = json.load(f)
        rows.append({
            "round_pair": "nucleus_to_round0",
            "n_matches": nuc_metrics.get("n_matches", 0),
            "n_inliers": nuc_metrics.get("n_inliers", 0),
            "inlier_ratio": nuc_metrics.get("inlier_ratio", 0.0),
            "residual_mean": nuc_metrics.get("residual_mean", 0.0),
            "residual_std": nuc_metrics.get("residual_std", 0.0),
            "residual_max": nuc_metrics.get("residual_max", 0.0),
        })

    # Round-to-round metrics
    for i in range(1, n_rounds):
        round_file = metrics_dir / f"{well_token}_round{i}_to_round{i-1}_metrics.json"
        if round_file.exists():
            with open(round_file) as f:
                rnd_metrics = json.load(f)
            rows.append({
                "round_pair": f"round{i}_to_round{i-1}",
                "n_matches": rnd_metrics.get("n_matches", 0),
                "n_inliers": rnd_metrics.get("n_inliers", 0),
                "inlier_ratio": rnd_metrics.get("inlier_ratio", 0.0),
                "residual_mean": rnd_metrics.get("residual_mean", 0.0),
                "residual_std": rnd_metrics.get("residual_std", 0.0),
                "residual_max": rnd_metrics.get("residual_max", 0.0),
            })

    if rows:
        df = pd.DataFrame(rows)
        output_path = metrics_dir / f"registration_metrics_{well_token}.csv"
        df.to_csv(output_path, index=False)
        if verbose:
            print(f"\n  Saved aggregated metrics: {output_path}")
            print(f"    {len(rows)} registration steps recorded")
    elif verbose:
        print(f"\n  No metrics JSON files found to aggregate for well {well_token}")


def submit_iss_registration_jobs(
    experiment: str,
    well: int,
    spot_threshold: float = 400,
    nucleus_threshold: float = 200,
    transform_type: str = "similarity",
    slurm_params: dict = None,
    finalize_slurm_params: dict = None,
    dry_run: bool = False,
    wait_for_completion: bool = True,
    verbose: bool = True,
    skip_prompt: bool = False,
    skip_apply_transforms: bool = False,
    reproducible: bool = False,
) -> dict:
    """
    Submit parallel SLURM jobs for ISS round-to-round registration.

    Creates 11-12 jobs (depending on pre_nuclei_round and number of rounds):
    - 1 segmentation → nucleus job
    - 1 nucleus → Round 0 job (only if pre_nuclei_round=True)
    - 8-9 round-to-round jobs (9 for 10-round experiments, 8 for 9-round experiments)
    - 1 finalization job (compose cumulative transforms + visualizations)

    Parameters
    ----------
    experiment : str
        Experiment name
    well : int
        Well number (1, 2, or 3)
    spot_threshold : float
        Spot intensity threshold
    nucleus_threshold : float
        Nucleus intensity threshold
    transform_type : str
        Transform type: "similarity", "affine", or "euclidean"
    slurm_params : dict
        SLURM parameters (timeout_min, mem, cpus_per_task, etc.)
    dry_run : bool
        Print what would be submitted without submitting
    wait_for_completion : bool
        Wait for all jobs to complete
    verbose : bool
        Print detailed progress

    Returns
    -------
    dict
        Job submission results
    """
    # Default SLURM parameters for registration jobs
    if slurm_params is None:
        # Get the cyclops_process directory for PYTHONPATH (3 levels up from this file)
        cyclops_process_dir = str(Path(__file__).parents[3])  # Go up to cyclops_process directory
        slurm_params = {
            "timeout_min": 10,
            "mem": "32GB",
            "cpus_per_task": 16,
            "slurm_partition": "cpu",
            "slurm_srun_args": ["--cpu-bind=none"],  # Disable CPU binding to avoid non-contiguous CPU errors
            "slurm_setup": [f"export PYTHONPATH={cyclops_process_dir}:$PYTHONPATH"],  # Ensure cyclops_process is importable
        }

    # Default SLURM parameters for finalization job (GPU + more time for transform application)
    if finalize_slurm_params is None:
        # Get the cyclops_process directory for PYTHONPATH (3 levels up from this file)
        cyclops_process_dir = str(Path(__file__).parents[3])  # Go up to cyclops_process directory
        finalize_slurm_params = {
            "timeout_min": 60,  # Increased from 45 for safety with parallel writes
            "mem": "64GB",
            "cpus_per_task": 16,
            "slurm_partition": "gpu",
            "slurm_gres": "gpu:1",
            "slurm_srun_args": ["--cpu-bind=none"],  # Disable CPU binding to avoid non-contiguous CPU errors
            "slurm_setup": [f"export PYTHONPATH={cyclops_process_dir}:$PYTHONPATH"],  # Ensure cyclops_process is importable
        }

    # Check if experiment has pre_nuclei_round (determines if nucleus registration is needed)
    dataset = OpsDataset(experiment)
    row, col = parse_well(well)
    well_token = f"{row}{col}"  # SLURM-safe label (no slashes) for job/manifest names

    # Load config from YAML file
    import yaml
    exp_config_path = dataset.config_paths["exp_config"]
    if exp_config_path.exists():
        with open(exp_config_path, "r") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    skip_nucleus_registration = False  # Always run nucleus registration

    # Anchor spots round: when round 0 has no usable spots (e.g. a no-incorporation
    # cycle that only carries DAPI), anchor the spots-registration chain at a later
    # round. Read from top-level config anchor_round (default 0 = round 0).
    anchor_round = int(config.get("anchor_round", 0) or 0)
    if verbose and anchor_round != 0:
        print(f"  Anchor spots round = {anchor_round} (round 0 spots skipped; "
              f"nucleus registers to round {anchor_round}, pairs start at round {anchor_round+1})")

    # Number of physical ISS rounds from the stitched store's T dimension (cheap
    # metadata read) — 11 for ops0173, so round 10's pair (R10->R9) gets submitted.
    n_rounds = _get_n_physical_rounds(dataset, well, experiment, verbose=verbose)

    # Note: DAPI_round10 manual registration is now handled in register_all_wells()
    # to prompt once for all wells instead of per-well

    # Build job list
    jobs_to_submit = []
    job_offset = 0  # Tracks index shift if nucleus job is skipped

    # Job -1: Segmentation → Nucleus (always runs first)
    jobs_to_submit.append({
        "name": f"w{well_token}_seg_to_nuc",
        "func": job_register_segmentation_to_nucleus,
        "kwargs": {
            "experiment": experiment,
            "well": well,
            "verbose": verbose,
            "reproducible": reproducible,
        },
        "metadata": {
            "type": "segmentation",
            "well": well,
        },
    })

    # Job 0: Nucleus → Round 0 (only if pre_nuclei_round=True)
    if not skip_nucleus_registration:
        jobs_to_submit.append({
            "name": f"w{well_token}_nucleus_to_r0",
            "func": job_register_nucleus_to_round0,
            "kwargs": {
                "experiment": experiment,
                "well": well,
                "spot_threshold": spot_threshold,
                "nucleus_threshold": nucleus_threshold,
                "transform_type": transform_type,
                "verbose": verbose,
                "reproducible": reproducible,
                "spots_round": anchor_round,
            },
            "metadata": {
                "type": "nucleus",
                "well": well,
            },
        })
    else:
        job_offset = -1  # Shift all job indices down by 1

    # Jobs (anchor+1)..(N-1): Round-to-round (R(a+1)→Ra, ..., R(N-1)→R(N-2))
    # Default anchor=0 → R1→R0, R2→R1, ... (unchanged). With anchor>0 the
    # spot-less leading rounds are skipped (no round-pair anchored at them).
    for i in range(anchor_round + 1, n_rounds):
        jobs_to_submit.append({
            "name": f"w{well_token}_r{i}_to_r{i-1}",
            "func": job_register_round_pair,
            "kwargs": {
                "experiment": experiment,
                "well": well,
                "round_source": i,
                "round_target": i - 1,
                "spot_threshold": spot_threshold,
                "transform_type": transform_type,
                "verbose": verbose,
                "reproducible": reproducible,
            },
            "metadata": {
                "type": "round_pair",
                "well": well,
                "round_source": i,
                "round_target": i - 1,
            },
        })

    # Job N: Finalization (compose cumulative + visualizations)
    # Dependencies: all previous registration jobs
    n_registration_jobs = len(jobs_to_submit)
    dependencies = list(range(n_registration_jobs))  # All registration jobs

    jobs_to_submit.append({
        "name": f"w{well_token}_finalize",
        "func": job_finalize_registration,
        "kwargs": {
            "experiment": experiment,
            "well": well,
            "verbose": verbose,
            "skip_apply_transforms": skip_apply_transforms,
            "reproducible": reproducible,
            "anchor_round": anchor_round,
        },
        "metadata": {
            "type": "finalize",
            "well": well,
        },
        "dependencies": dependencies,
    })

    # Store config for caller (used for multi-well prompt)
    result_metadata = {
        "experiment": experiment,
        "well": well,
        "skip_nucleus_registration": skip_nucleus_registration,
        "n_registration_jobs": n_registration_jobs,
        "config": config,
    }

    # Return metadata without submitting if skip_prompt=True (caller will prompt once for all wells)
    if skip_prompt:
        return {
            "success": True,
            "metadata": result_metadata,
            "jobs_to_submit": jobs_to_submit,
            "n_registration_jobs": n_registration_jobs,
            "finalize_job_spec": jobs_to_submit[n_registration_jobs],
            "finalize_slurm_params": finalize_slurm_params,
            "skip_prompt": True,
        }

    # Define registration_jobs before using it
    registration_jobs = jobs_to_submit[:n_registration_jobs]  # All registration jobs (9 or 10)

    # Single-well mode: prompt and submit
    if not dry_run:
        print(f"\n{'='*60}")
        print(f"Ready to submit {n_registration_jobs} registration jobs + 1 finalization job:")
        for i, job in enumerate(registration_jobs, 1):
            print(f"  {i}. {job['name']}")
        print(f"  {n_registration_jobs + 1}. finalize_iss_w{well_token} (GPU, runs after registration)")

        if skip_nucleus_registration:
            print(f"\nNote: Nucleus registration skipped (Round 0 nucleus & spots aligned)")

        print(f"\nSLURM Resources (registration):")
        print(f"  {slurm_params['timeout_min']}min | {slurm_params['mem']} | {slurm_params['cpus_per_task']} CPUs | {slurm_params['slurm_partition']}")
        print(f"SLURM Resources (finalization):")
        print(f"  {finalize_slurm_params['timeout_min']}min | {finalize_slurm_params['mem']} | {finalize_slurm_params['cpus_per_task']} CPUs | GPU: {finalize_slurm_params.get('slurm_gres', 'none')}")
        print(f"{'='*60}\n")

        response = input("Proceed with submission? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("\nSubmission cancelled by user.")
            return {"success": False, "error": "Cancelled by user"}
        print()

    # Submit registration jobs with standard CPU resources

    result = submit_parallel_jobs(
        jobs_to_submit=registration_jobs,
        experiment=experiment,
        slurm_params=slurm_params,
        log_dir=f"slurm_iss_register_logs/%j",
        manifest_prefix=f"iss_register_w{well_token}_reg",
        step_name="iss_register_cycles",
        dry_run=dry_run,
        wait_for_completion=False,  # Don't wait - caller will handle finalization
        verbose=verbose,
    )

    if dry_run:
        return result

    # Extract job IDs from result (submit_parallel_jobs returns base_job_id and jobs list)
    if result.get("base_job_id") and result.get("jobs"):
        base_id = result["base_job_id"]
        num_jobs = len(result["jobs"])
        result["job_ids"] = [f"{base_id}_{i}" for i in range(num_jobs)]
    else:
        result["job_ids"] = []

    # Store finalization job info for later submission (after registration completes)
    result["finalize_job_spec"] = jobs_to_submit[n_registration_jobs]
    result["finalize_slurm_params"] = finalize_slurm_params
    result["n_registration_jobs"] = n_registration_jobs
    result["metadata"] = result_metadata

    if verbose:
        job_ids = result.get('job_ids', [])
        print(f"\n✓ Submitted {len(job_ids)} registration jobs (finalization will auto-submit after completion)")

        # Print scancel command for easy cancellation
        if job_ids and result.get("base_job_id"):
            print(f"  To cancel: scancel {result['base_job_id']}")
        print()

    return result


def check_well_completion(dataset, well, verbose: bool = False) -> dict:
    """
    Check which registration outputs already exist for a well.

    Returns
    -------
    dict
        {
            "well_complete": bool,  # True if registered zarr exists with data
            "nucleus_complete": bool,  # True if nucleus affine exists
            "rounds_complete": list[int],  # List of round numbers with existing affines
        }
    """
    row, col = parse_well(well)
    well_token = f"{row}{col}"
    register_root = dataset.preprocess_in_situ / "register"
    transforms_dir = register_root / f"transforms/{well_token}"

    # Check if registered zarr position exists and contains non-zero data
    registered_zarr = dataset.store_paths.get("iss_stitch_registered")
    position = f"{row}/{col}/0"
    well_complete = False

    if verbose:
        print(f"\n  Checking Well {well_token} completion:")
        print(f"    Registered zarr path: {registered_zarr}")
        print(f"    Position: {position}")

    if registered_zarr and registered_zarr.exists():
        import zarr
        try:
            store = zarr.open(str(registered_zarr), mode='r')

            if verbose:
                print(f"    Zarr opened successfully")
                print(f"    Available keys: {list(store.keys())[:5]}...")  # Show first 5

            if position in store:
                if verbose:
                    print(f"    Position {position} found in zarr")

                # Access the actual data array
                data = store[position]

                # Handle HCS zarr format (Group containing '0' array)
                if isinstance(data, zarr.Group):
                    if verbose:
                        print(f"    Data is a zarr Group, accessing '0' array...")
                    if '0' in data:
                        data = data['0']
                    else:
                        if verbose:
                            print(f"    No '0' array found in Group")
                        data = None

                if data is None or not hasattr(data, 'shape'):
                    if verbose:
                        print(f"    Could not access data array")
                else:
                    if verbose:
                        print(f"    Data shape: {data.shape}")

                    if len(data.shape) >= 3:  # Should be (rounds, channels, z, y, x) or similar
                        # Get metadata for last round without loading it
                        round_idx = 9 if data.shape[0] > 9 else data.shape[0] - 1

                        if verbose:
                            print(f"    Checking round R{round_idx}")

                        # Get chunk shape from zarr metadata
                        chunks = data.chunks if hasattr(data, 'chunks') else None

                        if verbose and chunks:
                            print(f"    Full data chunks: {chunks}")

                        has_nonzero = False

                        # Sample 3 random chunks directly from the zarr array
                        for i in range(3):
                            # Determine chunk coordinates based on shape
                            if len(data.shape) == 5:  # (rounds, c, z, y, x)
                                c = np.random.randint(0, data.shape[1]) if data.shape[1] > 1 else 0
                                z = np.random.randint(0, data.shape[2]) if data.shape[2] > 1 else 0

                                # Calculate number of chunks in each dimension
                                n_chunks_y = max(1, data.shape[3] // chunks[3]) if chunks else 1
                                n_chunks_x = max(1, data.shape[4] // chunks[4]) if chunks else 1

                                y_chunk_idx = np.random.randint(0, n_chunks_y)
                                x_chunk_idx = np.random.randint(0, n_chunks_x)

                                y_start = y_chunk_idx * chunks[3] if chunks else 0
                                x_start = x_chunk_idx * chunks[4] if chunks else 0
                                y_end = min(y_start + (chunks[3] if chunks else 1024), data.shape[3])
                                x_end = min(x_start + (chunks[4] if chunks else 1024), data.shape[4])

                                # Load only this single chunk
                                chunk = data[round_idx, c, z, y_start:y_end, x_start:x_end]

                            elif len(data.shape) == 4:  # (rounds, c, y, x)
                                c = np.random.randint(0, data.shape[1]) if data.shape[1] > 1 else 0

                                n_chunks_y = max(1, data.shape[2] // chunks[2]) if chunks else 1
                                n_chunks_x = max(1, data.shape[3] // chunks[3]) if chunks else 1

                                y_chunk_idx = np.random.randint(0, n_chunks_y)
                                x_chunk_idx = np.random.randint(0, n_chunks_x)

                                y_start = y_chunk_idx * chunks[2] if chunks else 0
                                x_start = x_chunk_idx * chunks[3] if chunks else 0
                                y_end = min(y_start + (chunks[2] if chunks else 1024), data.shape[2])
                                x_end = min(x_start + (chunks[3] if chunks else 1024), data.shape[3])

                                # Load only this single chunk
                                chunk = data[round_idx, c, y_start:y_end, x_start:x_end]

                            elif len(data.shape) == 3:  # (rounds, y, x)
                                n_chunks_y = max(1, data.shape[1] // chunks[1]) if chunks else 1
                                n_chunks_x = max(1, data.shape[2] // chunks[2]) if chunks else 1

                                y_chunk_idx = np.random.randint(0, n_chunks_y)
                                x_chunk_idx = np.random.randint(0, n_chunks_x)

                                y_start = y_chunk_idx * chunks[1] if chunks else 0
                                x_start = x_chunk_idx * chunks[2] if chunks else 0
                                y_end = min(y_start + (chunks[1] if chunks else 1024), data.shape[1])
                                x_end = min(x_start + (chunks[2] if chunks else 1024), data.shape[2])

                                # Load only this single chunk
                                chunk = data[round_idx, y_start:y_end, x_start:x_end]
                            else:
                                if verbose:
                                    print(f"    Unknown shape format: {data.shape}")
                                continue

                            chunk_max = np.max(chunk)
                            chunk_nonzero = np.count_nonzero(chunk)

                            if verbose:
                                print(f"    Chunk {i+1} {chunk.shape}: max={chunk_max}, nonzero={chunk_nonzero}")

                            if chunk_max > 0:
                                has_nonzero = True
                                break

                        well_complete = has_nonzero

                        if verbose:
                            print(f"    → Well complete: {well_complete}")
                    else:
                        if verbose:
                            print(f"    Shape too small: {data.shape}")
            else:
                if verbose:
                    print(f"    Position {position} NOT found in zarr")
        except Exception as e:
            # If we can't verify data, assume incomplete
            if verbose:
                import traceback
                print(f"    Error checking zarr: {e}")
                print(f"    Full traceback:")
                traceback.print_exc()
            pass
    elif verbose:
        if registered_zarr:
            print(f"    Registered zarr does not exist at: {registered_zarr}")
        else:
            print(f"    Registered zarr path not configured")

    # Check segmentation affine
    segmentation_yaml = transforms_dir / "segmentation_to_nucleus.yaml"
    segmentation_complete = segmentation_yaml.exists()

    # Check nucleus affine
    nucleus_yaml = transforms_dir / "nucleus_to_round0.yml"
    nucleus_complete = nucleus_yaml.exists()

    # Check round-to-round affines
    n_rounds = _get_n_physical_rounds(dataset, well, getattr(dataset, "experiment", ""))
    rounds_complete = []
    for i in range(1, n_rounds):
        yaml_file = transforms_dir / f"round{i}_to_round{i-1}.yml"
        if yaml_file.exists():
            rounds_complete.append(i)

    return {
        "well_complete": well_complete,
        "segmentation_complete": segmentation_complete,
        "nucleus_complete": nucleus_complete,
        "rounds_complete": rounds_complete,
    }


def filter_jobs_by_completion(
    dataset,
    well: int,
    jobs_to_submit: list,
    skip_nucleus: bool,
    force: bool = False,
    verbose: bool = False,
) -> tuple[list, dict]:
    """
    Filter jobs based on existing outputs.

    Returns
    -------
    tuple[list, dict]
        (filtered_jobs, completion_status)
    """
    if force:
        return jobs_to_submit, {"well_complete": False, "skipped": 0}

    completion = check_well_completion(dataset, well, verbose=verbose)

    # If well is complete (zarr exists with data), skip entirely
    if completion["well_complete"]:
        return [], completion

    # Filter individual registration jobs (but always include finalize if zarr doesn't exist)
    filtered_jobs = []
    skipped_count = 0

    for job in jobs_to_submit:
        job_type = job.get("metadata", {}).get("type")

        if job_type == "segmentation":
            if completion["segmentation_complete"]:
                skipped_count += 1
                continue
        elif job_type == "nucleus":
            if completion["nucleus_complete"] or skip_nucleus:
                skipped_count += 1
                continue
        elif job_type == "round_pair":
            round_source = job.get("metadata", {}).get("round_source")
            if round_source in completion["rounds_complete"]:
                skipped_count += 1
                continue
        elif job_type == "finalize":
            # Always include finalize job if we get here (zarr doesn't exist with data)
            # This handles the case where affines exist but zarr is incomplete/missing
            pass

        filtered_jobs.append(job)

    completion["skipped"] = skipped_count

    # If all registration jobs were skipped but finalize exists, it means we need to run finalization
    # Check if we only have the finalize job left
    if len(filtered_jobs) == 1 and filtered_jobs[0].get("metadata", {}).get("type") == "finalize":
        if verbose:
            print(f"    → All registration complete, but zarr incomplete - will run finalization only")

    return filtered_jobs, completion




def submit_finalization_job(experiment, well, result, verbose=True):
    """Submit finalization job for a well."""
    finalize_spec = result["finalize_job_spec"]
    finalize_params = result["finalize_slurm_params"]
    row, col = parse_well(well)

    finalize_result = submit_parallel_jobs(
        jobs_to_submit=[finalize_spec],
        experiment=experiment,
        slurm_params=finalize_params,
        log_dir=f"slurm_iss_register_logs/%j",
        manifest_prefix=f"iss_register_w{row}{col}_finalize",
        step_name="iss_register_finalize",
        dry_run=False,
        wait_for_completion=False,
        verbose=verbose,
    )

    # Extract job IDs
    if finalize_result.get("base_job_id"):
        base_id = finalize_result["base_job_id"]
        num_jobs = len(finalize_result.get("jobs", []))

        # For single jobs or array jobs submitted as batch, use base_id only
        # (sacct recognizes base_id but not base_id_0 format)
        if num_jobs == 1:
            return [str(base_id)]
        else:
            # For actual array jobs (multiple elements), use array format
            job_ids = [f"{base_id}_{i}" for i in range(num_jobs)]
            return job_ids
    else:
        return []


def monitor_and_finalize_wells(
    experiment,
    wells,
    all_results,
    slurm_params,
    all_job_specs=None,
    verbose=True,
):
    """
    Monitor registration jobs with retry, then submit finalization when ALL succeed.

    This function:
    1. Monitors all registration jobs until completion
    2. Retries any failed jobs once (using monitor_jobs_with_retry)
    3. Only submits finalization jobs when ALL registrations succeed
    4. Raises RuntimeError if registration fails after retry

    Parameters
    ----------
    experiment : str
        Experiment name
    wells : list[int]
        List of well numbers
    all_results : list[dict]
        Results from job submission with job_ids, finalize_job_spec, etc.
    slurm_params : dict
        SLURM parameters for registration jobs
    all_job_specs : list[dict], optional
        Original job specifications (needed for retries). If not provided, retries disabled.
    verbose : bool
        Print progress

    Returns
    -------
    dict
        {well: [finalization_job_ids]}

    Raises
    ------
    RuntimeError
        If registration jobs fail after retry attempt
    """
    from datetime import timedelta
    from cyclops_utils.hpc.slurm_utils import monitor_jobs_with_retry

    finalization_job_ids = {well: [] for well in wells}

    # Collect all registration job IDs and specs
    all_reg_job_ids = []
    all_reg_job_specs = []
    job_id_to_well_idx = {}  # Map job_id to (well_idx, local_idx)

    for well_idx, well in enumerate(wells):
        result = all_results[well_idx]
        job_ids = result.get("job_ids", [])

        for local_idx, job_id in enumerate(job_ids):
            all_reg_job_ids.append(job_id)
            job_id_to_well_idx[job_id] = (well_idx, local_idx)

            # Get the job spec if available
            if all_job_specs and well_idx < len(all_job_specs):
                spec = all_job_specs[well_idx]
                reg_jobs = spec.get("jobs_to_submit", [])[:spec.get("n_registration_jobs", 0)]
                if local_idx < len(reg_jobs):
                    all_reg_job_specs.append(reg_jobs[local_idx])
                else:
                    all_reg_job_specs.append({"name": f"job_{job_id}"})
            else:
                all_reg_job_specs.append({"name": f"job_{job_id}"})

    total_reg_jobs = len(all_reg_job_ids)

    # === Phase 1: Monitor registration jobs with retry ===
    if total_reg_jobs > 0:
        # Use the generic retry-enabled monitor
        monitor_result = monitor_jobs_with_retry(
            job_ids=all_reg_job_ids,
            job_specs=all_reg_job_specs,
            experiment=experiment,
            slurm_params=slurm_params,
            submit_fn=submit_parallel_jobs,
            poll_interval=10,
            max_retries=1,  # Single retry
            verbose=verbose,
            phase_name="registration jobs",
        )

        # If we get here, all registration jobs succeeded
        if verbose and monitor_result.get("retry_count", 0) > 0:
            print(f"  (completed after {monitor_result['retry_count']} retry)")

    # === Phase 2: Submit finalization jobs for ALL wells ===
    all_finalize_jobs = []
    well_finalize_mapping = []

    for well_idx, well in enumerate(wells):
        result = all_results[well_idx]
        finalize_spec = result["finalize_job_spec"]
        start_idx = len(all_finalize_jobs)
        all_finalize_jobs.append(finalize_spec)
        well_finalize_mapping.append({
            "well": well,
            "start_idx": start_idx,
            "end_idx": start_idx + 1,
        })

    if not all_finalize_jobs:
        if verbose:
            print(f"\n✓ No finalization jobs to submit.\n")
        return finalization_job_ids

    # Pre-create the registered zarr with all well positions to avoid race conditions
    # when finalization jobs run concurrently. Skip it when finalize doesn't write
    # the store (skip_apply_transforms=True, merge pipeline) — otherwise the empty
    # v3 store would make convert_iss_to_v3 skip, and merge owns the write anyway.
    _skip_apply = bool(
        all_results[0].get("finalize_job_spec", {}).get("kwargs", {}).get("skip_apply_transforms", False)
    ) if all_results else False
    if not _skip_apply:
        _precreate_registered_zarr(experiment, wells, verbose=verbose)

    if verbose:
        print(f"\nSubmitting {len(all_finalize_jobs)} finalization jobs...")

    finalize_result = submit_parallel_jobs(
        jobs_to_submit=all_finalize_jobs,
        experiment=experiment,
        slurm_params=all_results[0]["finalize_slurm_params"],
        log_dir=f"slurm_iss_register_logs/%j",
        manifest_prefix=f"iss_register_all_wells_finalize",
        step_name="iss_register_all_wells_finalize",
        dry_run=False,
        wait_for_completion=False,
        verbose=verbose,
    )

    # Extract finalization job IDs
    all_finalize_job_ids = []
    if finalize_result.get("base_job_id") and finalize_result.get("jobs"):
        base_id = finalize_result["base_job_id"]
        num_jobs = len(finalize_result["jobs"])

        if num_jobs == 1:
            all_finalize_job_ids = [str(base_id)]
        else:
            all_finalize_job_ids = [f"{base_id}_{i}" for i in range(num_jobs)]

        if verbose:
            print(f"  → Submitted: {base_id} ({num_jobs} jobs)\n")

        # Map job IDs back to wells
        for mapping in well_finalize_mapping:
            well = mapping["well"]
            finalization_job_ids[well] = all_finalize_job_ids[mapping["start_idx"]:mapping["end_idx"]]

    # === Phase 3: Monitor finalization jobs (no retry - these should succeed) ===
    if all_finalize_job_ids:
        # Build finalize specs for potential error messages
        finalize_specs = [{"name": f"finalize_w{str(m['well']).replace('/', '')}"} for m in well_finalize_mapping]

        finalize_monitor = monitor_jobs_with_retry(
            job_ids=all_finalize_job_ids,
            job_specs=finalize_specs,
            experiment=experiment,
            slurm_params=all_results[0]["finalize_slurm_params"],
            submit_fn=submit_parallel_jobs,
            poll_interval=10,
            max_retries=0,  # No retry for finalization
            verbose=verbose,
            phase_name="finalization jobs",
        )

        if verbose:
            print(f"\n✓ All {len(wells)} wells completed successfully "
                  f"({total_reg_jobs} registration + {len(all_finalize_job_ids)} finalization jobs)\n")

    return finalization_job_ids


def submit_and_monitor_wells(
    experiment: str,
    wells: list[int],
    all_job_specs: list[dict],
    slurm_params: dict,
    wait_for_completion: bool = True,
    verbose: bool = True,
) -> list[dict]:
    """
    Submit registration jobs for all wells in a single SLURM array and optionally monitor/finalize.

    Parameters
    ----------
    experiment : str
        Experiment name
    wells : list[int]
        List of well numbers
    all_job_specs : list[dict]
        Job specifications from submit_iss_registration_jobs with skip_prompt=True
    slurm_params : dict
        SLURM parameters for registration jobs
    wait_for_completion : bool
        Wait for jobs to complete and submit finalization
    verbose : bool
        Print progress

    Returns
    -------
    list[dict]
        Results for each well
    """
    all_results = []

    # Collect all registration jobs across all wells into one batch
    all_registration_jobs = []
    well_job_mapping = []  # Track which jobs belong to which well
    results_by_index = [None] * len(wells)  # Preserve 1:1 alignment with wells

    for well_idx, well_num in enumerate(wells):
        spec = all_job_specs[well_idx]
        registration_jobs = spec["jobs_to_submit"][:spec["n_registration_jobs"]]

        if len(registration_jobs) > 0:
            # Track the range of jobs for this well
            start_idx = len(all_registration_jobs)
            all_registration_jobs.extend(registration_jobs)
            end_idx = len(all_registration_jobs)
            well_job_mapping.append({
                "well": well_num,
                "well_idx": well_idx,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "spec": spec,
            })
        else:
            # Finalize-only case: create placeholder result (no registration job_ids)
            results_by_index[well_idx] = {
                "success": True,
                "job_ids": [],
                "finalize_job_spec": spec["finalize_job_spec"],
                "finalize_slurm_params": spec["finalize_slurm_params"],
                "n_registration_jobs": spec["n_registration_jobs"],
                "metadata": spec["metadata"],
            }

    # Submit all registration jobs as a single SLURM array
    if all_registration_jobs:
        if verbose:
            print(f"\nSubmitting {len(all_registration_jobs)} registration jobs across {len(wells)} wells as single SLURM array...")

        result = submit_parallel_jobs(
            jobs_to_submit=all_registration_jobs,
            experiment=experiment,
            slurm_params=slurm_params,
            log_dir=f"slurm_iss_register_logs/%j",
            manifest_prefix=f"iss_register_all_wells",
            step_name="iss_register_all_wells",
            dry_run=False,
            wait_for_completion=False,
            verbose=verbose,
        )

        # Extract job IDs and map to wells
        if result.get("base_job_id") and result.get("jobs"):
            base_id = result["base_job_id"]
            num_jobs = len(result["jobs"])
            all_job_ids = [f"{base_id}_{i}" for i in range(num_jobs)]

            if verbose:
                print(f"  → Submitted: {base_id} ({num_jobs} jobs)\n")

            # Map job IDs back to each well and place in aligned results
            for mapping in well_job_mapping:
                well_idx = mapping["well_idx"]
                results_by_index[well_idx] = {
                    "success": True,
                    "job_ids": all_job_ids[mapping["start_idx"]:mapping["end_idx"]],
                    "finalize_job_spec": mapping["spec"]["finalize_job_spec"],
                    "finalize_slurm_params": mapping["spec"]["finalize_slurm_params"],
                    "n_registration_jobs": mapping["spec"]["n_registration_jobs"],
                    "metadata": mapping["spec"]["metadata"],
                }
        else:
            # No registration jobs submitted
            for mapping in well_job_mapping:
                well_idx = mapping["well_idx"]
                results_by_index[well_idx] = {
                    "success": True,
                    "job_ids": [],
                    "finalize_job_spec": mapping["spec"]["finalize_job_spec"],
                    "finalize_slurm_params": mapping["spec"]["finalize_slurm_params"],
                    "n_registration_jobs": mapping["spec"]["n_registration_jobs"],
                    "metadata": mapping["spec"]["metadata"],
                }
    else:
        # No registration jobs at all
        for well_idx, well_num in enumerate(wells):
            spec = all_job_specs[well_idx]
            results_by_index[well_idx] = {
                "success": True,
                "job_ids": [],
                "finalize_job_spec": spec["finalize_job_spec"],
                "finalize_slurm_params": spec["finalize_slurm_params"],
                "n_registration_jobs": spec["n_registration_jobs"],
                "metadata": spec["metadata"],
            }

    # Build aligned results list in original wells order
    all_results = results_by_index

    print()

    # Monitor and finalize if requested
    finalization_job_ids = {}
    if wait_for_completion:
        finalization_job_ids = monitor_and_finalize_wells(
            experiment,
            wells,
            all_results,
            slurm_params=slurm_params,
            all_job_specs=all_job_specs,
            verbose=verbose,
        )

    # Store finalization info in results
    for well_idx, well in enumerate(wells):
        all_results[well_idx]["finalization_job_ids"] = finalization_job_ids.get(well, [])

    return all_results


def _ensure_iss_overlays_symlink(dataset) -> None:
    """Expose preprocess_in_situ/register/overlays under 3-assembly/ISS/registration_overlays
    so review artifacts live alongside the rest of the assembly outputs. Idempotent."""
    target = dataset.preprocess_in_situ / "register" / "overlays"
    link = dataset.results / "ISS" / "registration_overlays"
    target.mkdir(parents=True, exist_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        try:
            if link.resolve() == target.resolve():
                return
        except Exception:
            pass
        try:
            link.unlink()
        except OSError as e:
            print(f"[iss_register] could not replace stale symlink {link}: {e}")
            return
    elif link.exists():
        print(f"[iss_register] {link} exists as a real path; leaving it alone")
        return
    try:
        link.symlink_to(target, target_is_directory=True)
        print(f"[iss_register] symlinked {link} -> {target}")
    except OSError as e:
        print(f"[iss_register] failed to create symlink {link} -> {target}: {e}")


@versioned_function(version="1.0")
def register_iss_cycles(
    experiment: str,
    spot_threshold: float = 400,
    nucleus_threshold: float = 200,
    transform_type: str = "similarity",
    slurm_params: dict = None,
    finalize_slurm_params: dict = None,
    wait_for_completion: bool = True,
    force: bool = False,
    verbose: bool = True,
    skip_apply_transforms: bool = False,
    reproducible: bool = False,
) -> dict:
    """
    Wrapper function to register all wells for an experiment.

    Reads wells_to_process from experiment config and submits registration
    jobs for each well. Can be called from orchestrator or CLI.

    Parameters
    ----------
    experiment : str
        Experiment name
    spot_threshold : float
        Spot intensity threshold
    nucleus_threshold : float
        Nucleus intensity threshold
    transform_type : str
        Transform type: "similarity", "affine", or "euclidean"
    slurm_params : dict
        SLURM parameters for registration jobs
    finalize_slurm_params : dict
        SLURM parameters for finalization jobs
    wait_for_completion : bool
        Wait for all jobs to complete
    force : bool
        Force resubmission even if outputs exist
    verbose : bool
        Print detailed progress

    Returns
    -------
    dict
        Combined results from all wells
    """
    import yaml
    from cyclops_utils.data.experiment import OpsDataset

    # Load experiment config to get wells
    dataset = OpsDataset(experiment)
    _ensure_iss_overlays_symlink(dataset)
    exp_config_path = dataset.config_paths["exp_config"]

    if exp_config_path.exists():
        with open(exp_config_path, "r") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    cfg_wells = config.get("wells_to_process", ["A/1/0", "A/2/0", "A/3/0"])

    # Keep each well as a full row/col unit so wells in different rows don't collide.
    well_units = []
    for w in cfg_wells:
        r, c = parse_well(w)
        well_units.append(f"{r}/{c}/0")

    if verbose:
        print(f"\n=== ISS Cycle Registration for {experiment} ===")
        print(f"Wells to process: {well_units}")
        print()

    # Always run nucleus registration (spots-to-nucleus alignment is critical)
    skip_nucleus_registration = False

    # Check for DAPI_round10 (will be handled in finalization)
    has_dapi_round10 = check_for_dapi_round10(dataset, verbose=verbose)
    if has_dapi_round10:
        # Skip normal nucleus registration - DAPI_round10 will be handled differently
        skip_nucleus_registration = True

    # When reproducible, BLAS must be single-threaded inside the SLURM job too —
    # parallel reductions in OpenBLAS/MKL produce run-to-run float32 noise that
    # breaks bit-reproducibility of registration optimization.
    repro_setup = []
    if reproducible:
        repro_setup = [
            "export OMP_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            "export OPENBLAS_NUM_THREADS=1",
            "export NUMEXPR_NUM_THREADS=1",
        ]

    # Default SLURM parameters if not provided
    if slurm_params is None:
        # Get the cyclops_process directory for PYTHONPATH (3 levels up from this file)
        cyclops_process_dir = str(Path(__file__).parents[3])  # Go up to cyclops_process directory
        slurm_params = {
            "timeout_min": 10,
            "mem": "32GB",
            "cpus_per_task": 16,
            "slurm_partition": "cpu",
            "slurm_srun_args": ["--cpu-bind=none"],  # Disable CPU binding to avoid non-contiguous CPU errors
            "slurm_setup": [f"export PYTHONPATH={cyclops_process_dir}:$PYTHONPATH"] + repro_setup,
        }
    elif reproducible:
        slurm_params = {**slurm_params, "slurm_setup": list(slurm_params.get("slurm_setup", [])) + repro_setup}

    if finalize_slurm_params is None:
        # Get the cyclops_process directory for PYTHONPATH (3 levels up from this file)
        cyclops_process_dir = str(Path(__file__).parents[3])  # Go up to cyclops_process directory
        finalize_slurm_params = {
            "timeout_min": 80,
            "mem": "64GB",
            "cpus_per_task": 16,
            "slurm_partition": "gpu",
            "slurm_gres": "gpu:1",
            "slurm_srun_args": ["--cpu-bind=none"],  # Disable CPU binding to avoid non-contiguous CPU errors
            "slurm_setup": [f"export PYTHONPATH={cyclops_process_dir}:$PYTHONPATH"] + repro_setup,
        }
    elif reproducible:
        finalize_slurm_params = {**finalize_slurm_params, "slurm_setup": list(finalize_slurm_params.get("slurm_setup", [])) + repro_setup}

    # Collect job specs for all wells
    all_job_specs = []
    wells_to_process = []
    for well_unit in well_units:
        if verbose:
            print(f"--- Processing Well {well_unit} ---")

        spec = submit_iss_registration_jobs(
            experiment=experiment,
            well=well_unit,
            spot_threshold=spot_threshold,
            nucleus_threshold=nucleus_threshold,
            transform_type=transform_type,
            slurm_params=slurm_params,
            finalize_slurm_params=finalize_slurm_params,
            dry_run=False,
            wait_for_completion=False,
            verbose=verbose,
            skip_prompt=True,  # Don't submit yet, just collect specs
            skip_apply_transforms=skip_apply_transforms,
            reproducible=reproducible,
        )

        # Filter jobs based on completion status (skip jobs with existing outputs).
        # When skip_apply_transforms=True the registered zarr never gets written,
        # so the standard "well_complete = registered zarr exists" check would
        # never trigger; bypass the early-exit path entirely when skipping.
        filtered_jobs, completion = filter_jobs_by_completion(
            dataset, well_unit, spec["jobs_to_submit"], skip_nucleus_registration, force, verbose=verbose
        )

        if completion["well_complete"] and not force and not skip_apply_transforms:
            if verbose:
                print(f"  → Well {well_unit}: Already complete (registered zarr exists)\n")
            continue

        if not filtered_jobs:
            if verbose:
                print(f"  → Well {well_unit}: All registration complete, skipping\n")
            continue

        # Update spec with filtered jobs
        spec["jobs_to_submit"] = filtered_jobs
        spec["n_registration_jobs"] = len([j for j in filtered_jobs if j.get("metadata", {}).get("type") != "finalize"])

        all_job_specs.append(spec)
        wells_to_process.append(well_unit)

    # If no wells need processing, return early
    if not all_job_specs:
        if verbose:
            print(f"\n✓ All wells already complete!")
            if not force:
                print(f"  Use force=True to resubmit anyway\n")
        return {
            "success": True,
            "experiment": experiment,
            "wells": well_units,
            "results": [],
        }

    # Submit and monitor all wells
    all_results = submit_and_monitor_wells(
        experiment=experiment,
        wells=wells_to_process,
        all_job_specs=all_job_specs,
        slurm_params=slurm_params,
        wait_for_completion=wait_for_completion,
        verbose=verbose,
    )

    if verbose:
        print(f"\n=== Registration Complete for {experiment} ===")
        print(f"Processed {len(wells_to_process)} wells")

    return {
        "success": all(r.get("success", False) for r in all_results),
        "experiment": experiment,
        "wells": wells_to_process,
        "results": all_results,
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Submit ISS round-to-round registration jobs to SLURM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        required=True,
        help="Experiment name (e.g., ops0032_20250428)",
    )

    parser.add_argument(
        "--well",
        "-w",
        type=str,
        required=True,
        help="Well number (1, 2, 3, or 'all')",
    )

    parser.add_argument(
        "--spot-threshold",
        type=float,
        default=400,
        help="Spot intensity threshold (default: 400)",
    )

    parser.add_argument(
        "--nucleus-threshold",
        type=float,
        default=200,
        help="Nucleus intensity threshold (default: 200)",
    )

    parser.add_argument(
        "--transform-type",
        type=str,
        default="similarity",
        choices=["affine", "similarity", "euclidean"],
        help="Transform type (default: similarity)",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="SLURM timeout in minutes (default: 10)",
    )

    parser.add_argument(
        "--mem",
        type=str,
        default="16GB",
        help="SLURM memory allocation (default: 16GB)",
    )

    parser.add_argument(
        "--cpus",
        type=int,
        default=16,
        help="SLURM CPUs per task (default: 16)",
    )

    parser.add_argument(
        "--partition",
        type=str,
        default="cpu",
        help="SLURM partition (default: cpu)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be submitted without submitting",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Reduce verbosity",
    )

    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit jobs and return immediately without waiting",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force resubmission even if outputs exist",
    )

    args = parser.parse_args()

    # Resolve experiment name
    experiment = resolve_experiment_name(
        args.experiment,
        verbose=True,
        allow_interactive=True
    )

    # Parse wells
    if args.well.lower() == "all":
        wells = [1, 2, 3]
    else:
        wells = [int(args.well)]

    # Build SLURM parameters
    slurm_params = {
        "timeout_min": args.timeout,
        "mem": args.mem,
        "cpus_per_task": args.cpus,
        "slurm_partition": args.partition,
    }

    # Multi-well mode: collect metadata first, then show single prompt
    if len(wells) > 1:
        # Load dataset for completion checking
        dataset = OpsDataset(experiment)

        # Collect job specs and check completion status
        all_job_specs = []
        wells_to_process = []
        completion_summary = {}

        for well in wells:
            spec = submit_iss_registration_jobs(
                experiment=experiment,
                well=well,
                spot_threshold=args.spot_threshold,
                nucleus_threshold=args.nucleus_threshold,
                transform_type=args.transform_type,
                slurm_params=slurm_params,
                dry_run=args.dry_run,
                wait_for_completion=False,
                verbose=not args.quiet,
                skip_prompt=True,  # Don't prompt yet
            )

            # Check completion and filter jobs (with debug output)
            skip_nucleus = spec["metadata"]["skip_nucleus_registration"]

            # Add debug output for completion check
            if not args.quiet:
                print(f"\nChecking Well A{well}...")

            filtered_jobs, completion = filter_jobs_by_completion(
                dataset, well, spec["jobs_to_submit"], skip_nucleus, args.force, verbose=not args.quiet
            )

            completion_summary[well] = completion

            if completion["well_complete"]:
                if not args.quiet:
                    print(f"  → Well A{well}: Already complete (registered zarr exists with data)")
                continue

            # Check if we have any jobs to submit
            n_reg_jobs = len([j for j in filtered_jobs if j.get("metadata", {}).get("type") != "finalize"])
            n_finalize_jobs = len([j for j in filtered_jobs if j.get("metadata", {}).get("type") == "finalize"])

            if not filtered_jobs:
                if not args.quiet:
                    print(f"  Well A{well}: All jobs complete, skipping")
                continue

            # Special case: if only finalization job remains (all registration YAMLs exist but zarr incomplete)
            if n_reg_jobs == 0 and n_finalize_jobs > 0:
                if not args.quiet:
                    print(f"  Well A{well}: Registration complete, will run finalization only")

            # Update spec with filtered jobs
            spec["jobs_to_submit"] = filtered_jobs
            spec["n_registration_jobs"] = n_reg_jobs

            all_job_specs.append(spec)
            wells_to_process.append(well)

        # If no wells need processing, exit
        if not wells_to_process:
            print(f"\n✓ All wells already complete!")
            print(f"  Use --force to resubmit anyway")
            sys.exit(0)

        # Update wells list to only include wells that need processing
        wells = wells_to_process

        # Show unified prompt for all wells
        if not args.dry_run:
            metadata = all_job_specs[0]["metadata"]  # All wells share same config
            config = metadata["config"]
            has_pre_dapi = config.get("stack_symlinks_params", {}).get("pre_nuclei_round", False)
            skip_nucleus = metadata["skip_nucleus_registration"]
            n_reg_jobs = metadata["n_registration_jobs"]

            print(f"\n{'='*60}")
            print(f"ISS Registration Pipeline for {experiment}")
            print(f"{'='*60}")

            print(f"\nWells to process: {wells}")

            # Show completion summary if any jobs were skipped
            if not args.force:
                total_skipped = sum(completion_summary.get(w, {}).get("skipped", 0) for w in wells)
                if total_skipped > 0:
                    print(f"\nCompletion status (--force to override):")
                    for well in wells:
                        comp = completion_summary[well]
                        if comp.get("skipped", 0) > 0:
                            print(f"  Well A{well}: {comp['skipped']} jobs already complete")

            print(f"\nExperiment Configuration:")
            print(f"  Pre-DAPI round: {has_pre_dapi}")
            if skip_nucleus:
                print(f"  → Nucleus registration: SKIPPED (DAPI_round10 handled in finalization)")
            else:
                print(f"  → Nucleus registration: ENABLED")
                print(f"     (Will align Round 0 nucleus → spots)")

            # Calculate actual job counts from filtered specs
            total_reg_jobs = sum(spec["n_registration_jobs"] for spec in all_job_specs)
            total_finalize_jobs = len(wells)

            print(f"\nJobs to submit:")
            print(f"  - {total_reg_jobs} registration jobs (all wells, parallel)")
            print(f"  - {total_finalize_jobs} finalization jobs (per well, after registration)")

            if not args.force:
                original_total = len(wells) * n_reg_jobs
                total_skipped = original_total - total_reg_jobs
                if total_skipped > 0:
                    print(f"  - {total_skipped} jobs skipped (already complete)")

            print(f"\nSLURM Resources (registration jobs):")
            print(f"  Timeout: {slurm_params['timeout_min']} min")
            print(f"  Memory: {slurm_params['mem']}")
            print(f"  CPUs: {slurm_params['cpus_per_task']}")
            print(f"  Partition: {slurm_params['slurm_partition']}")

            finalize_params = all_job_specs[0]["finalize_slurm_params"]
            print(f"\nSLURM Resources (finalization jobs):")
            print(f"  Timeout: {finalize_params['timeout_min']} min")
            print(f"  Memory: {finalize_params['mem']}")
            print(f"  CPUs: {finalize_params['cpus_per_task']}")
            print(f"  Partition: {finalize_params['slurm_partition']}")
            print(f"  GPU: {finalize_params.get('slurm_gres', 'none')}")
            print(f"{'='*60}\n")

            response = input("Proceed with submission? [y/N]: ").strip().lower()
            if response not in ['y', 'yes']:
                print("\nSubmission cancelled by user.")
                sys.exit(0)
            print()

        # Now submit all wells using the shared helper
        print(f"Submitting jobs...")
        all_results = submit_and_monitor_wells(
            experiment=experiment,
            wells=wells,
            all_job_specs=all_job_specs,
            slurm_params=slurm_params,
            wait_for_completion=not args.no_wait,
            verbose=not args.quiet,
        )

    else:
        # Single-well mode: use original function with prompt
        all_results = []
        result = submit_iss_registration_jobs(
            experiment=experiment,
            well=wells[0],
            spot_threshold=args.spot_threshold,
            nucleus_threshold=args.nucleus_threshold,
            transform_type=args.transform_type,
            slurm_params=slurm_params,
            dry_run=args.dry_run,
            wait_for_completion=False,
            verbose=not args.quiet,
        )
        all_results.append(result)

    # Dry run - exit early
    if args.dry_run:
        sys.exit(0)

    # Summary
    print(f"\n{'='*60}")
    print(f"All registration jobs submitted")
    print(f"  Total wells: {len(wells)}")

    # Count total jobs properly (handle both individual job IDs and array job IDs)
    total_reg_jobs = 0
    for r in all_results:
        job_ids = r.get("job_ids", [])
        if job_ids:
            # Check if this is an array job (format: "12345_0")
            first_job = job_ids[0]
            if "_" in first_job:
                # Array job - count based on array indices
                total_reg_jobs += len(job_ids)
            else:
                # Individual jobs
                total_reg_jobs += len(job_ids)

    print(f"  Total registration jobs: {total_reg_jobs}")

    # Print scancel command for easy cancellation
    base_job_ids = []
    for r in all_results:
        if r.get("base_job_id"):
            base_job_ids.append(str(r["base_job_id"]))

    if base_job_ids:
        scancel_cmd = f"scancel {' '.join(base_job_ids)}"
        print(f"\n  To cancel all jobs, run:")
        print(f"    {scancel_cmd}")

    print(f"{'='*60}\n")

    # Monitoring and finalization already handled by submit_and_monitor_wells
    # Check if any jobs failed (results include finalization_job_ids after monitoring)
    if not args.no_wait:
        # Collect all finalization job IDs
        all_finalize_ids = []
        for r in all_results:
            all_finalize_ids.extend(r.get("finalization_job_ids", []))

        # Check if any failed
        if all_finalize_ids:
            try:
                status = check_jobs_complete(all_finalize_ids)
                failed = len(status["failed_jobs"]) > 0
                sys.exit(1 if failed else 0)
            except:
                sys.exit(1)
        elif not all(r.get("success", False) for r in all_results):
            # No finalization jobs but registration failed
            sys.exit(1)

    # Success if we get here
    sys.exit(0)


if __name__ == "__main__":
    main()
