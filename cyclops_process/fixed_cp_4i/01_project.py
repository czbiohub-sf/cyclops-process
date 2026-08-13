"""Max Z-projection for fixed-cell stores (cell painting + 4i).

Step 01 in the unified pipeline. Creates per-position max projections over Z
for all channels, collapsing (T,C,Z,Y,X) to (T,C,1,Y,X). Modality-aware via
``--modality {cp,4i}``; unit vocabulary and convert dir come from the config.

Usage:
    python -m cyclops_process.fixed_cp_4i.01_project --experiment ops0174 --parts 1 2
    python -m cyclops_process.fixed_cp_4i.01_project -e ops0174 --modality 4i --parts 1 3 5
    python -m cyclops_process.fixed_cp_4i.01_project -e ops0174 --parts 1 --dry-run
    python -m cyclops_process.fixed_cp_4i.01_project /path/to/part1.zarr --local  # manual mode
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import zarr
from joblib import Parallel, delayed
from tqdm import tqdm

from iohub import open_ome_zarr

import sys
import os

sys.path.insert(
    0, os.getcwd()
)

from ops_utils.hpc.resource_manager import get_optimal_workers
from ops_utils.data.filesystem import ensure_output_path
from ops_utils.io.zarr_utils import _discover_positions
from ops_utils.io.zarr_precreate import create_hcs_store_fast
from ops_utils.data.experiment import OpsDataset
from ops_utils.data.filesystem import resolve_experiment_name
from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
from cyclops_process.fixed_cp_4i.configs.modality_config import get_modality
from ops_utils.data.image_utils import augment_tile

# Max-Z projection loads whole positions into memory — heavy, so submit to SLURM
# (like steps 02/05) rather than running on the login node.
SLURM_PARAMS = {
    "timeout_min": 240,
    "mem": "250GB",  # sized for the flatfield pass folded into this step
    "cpus_per_task": 64,
    "slurm_partition": "cpu",
}


def _max_project_store(source_path: Path, num_workers: int | None = None, overwrite: bool | None = None,
                       flipud: bool = False, fliplr: bool = False, rot90: int = 0) -> Path:
    """Create a per-position max projection over Z for all channels and timepoints.

    - Input store is assumed OME-Zarr in HCS layout with positions like A/<well>/<tile>.
    - Output store is created next to input with suffix "_max_proj.zarr".
    - Shape transformation: (T, C, Z, Y, X) → (T, C, 1, Y, X) by max over axis=2.
    - Orientation (flipud/fliplr/rot90) is applied per-tile here so downstream
      stitch/segment operate on already-oriented data (this is why there is no
      separate flip step). rot90 by an odd count swaps the Y,X output dims.
    """
    rot90 = int(rot90) % 4
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Input store not found: {source_path}")

    out_path = source_path.with_name(f"{source_path.stem}_max_proj.zarr")

    # Prepare source metadata and position list
    with open_ome_zarr(source_path, mode="r") as ds:
        try:
            channel_names = ds.channel_names
        except Exception:
            channel_names = None
        # Use fast glob-based discovery (balanced) when available; fallback to generic
        try:
            positions = _discover_positions(source_path)
        except Exception:
            positions = [p for p, _ in ds.positions()]
        if not positions:
            raise ValueError(f"No positions found in {source_path}")
        first_pos = ds[positions[0]]
        input_shape = first_pos.data.shape  # (T, C, Z, Y, X)
        dtype = first_pos.data.dtype
        print(f"Input shape: {input_shape}")
        print(f"Dtype: {dtype}")
        try:
            scale = first_pos.scale
            # Fix: Override incorrect XY pixel size from microscope metadata
            # 20x OPS imaging uses 0.65 μm/px (not 0.325 from binned metadata)
            if len(scale) == 5 and abs(scale[-1] - 0.325) < 0.01:
                print(f"⚠️  Correcting incorrect pixel size: {scale[-2:]}")
                scale = scale[:-2] + [0.65, 0.65]
                print(f"   → Using correct 20x scale: {scale[-2:]}")
        except Exception:
            scale = [1.0, 1.0, 1.0, 0.65, 0.65]  # Default to correct 20x scale
        print(f"Scale: {scale}")
        # Output: Z collapsed to 1 plane. An odd rot90 swaps the Y,X dims, so the
        # pre-created store (and chunks/scale) must reflect the oriented geometry.
        out_y, out_x = input_shape[3], input_shape[4]
        chunks = first_pos.data.chunks
        if rot90 % 2 == 1:
            out_y, out_x = out_x, out_y
            chunks = chunks[:3] + (chunks[4], chunks[3])
            scale = list(scale[:3]) + [scale[4], scale[3]]
            print(f"rot90={rot90} (odd): swapped Y,X -> out ({out_y}x{out_x})")
        out_shape = (input_shape[0], input_shape[1], 1, out_y, out_x)
        if flipud or fliplr or rot90:
            print(f"Orientation baked into projection: flipud={flipud} fliplr={fliplr} rot90={rot90}")
        print(f"Chunks: {chunks}")
    # Initialize output store (prompt only in interactive/manual mode)
    if not ensure_output_path(out_path, prompt_user=(overwrite is None), overwrite=overwrite):
        print(f"Skipping projection (existing output retained at {out_path}).")
        return out_path

    # Use fast zarr precreation instead of slow iohub method
    print(f"Creating output store with {len(positions)} positions using fast precreate...")
    create_hcs_store_fast(
        store_path=out_path,
        positions=positions,
        shape=out_shape,
        chunks=chunks,
        dtype=dtype,
        scale=tuple(scale),
        channel_names=channel_names if channel_names else [f"ch{i}" for i in range(out_shape[1])],
    )

    # Worker to project a single position (+ apply orientation per T,C tile)
    def _project_pos(pos: str) -> None:
        with open_ome_zarr(source_path, mode="r") as src:
            arr = src[pos].data[:]  # (T,C,Z,Y,X) - load into memory
            proj = np.max(arr, axis=2)  # (T,C,Y,X)
        if flipud or fliplr or rot90:
            T, C = proj.shape[0], proj.shape[1]
            oriented = [[augment_tile(proj[t, c], flipud=flipud, fliplr=fliplr, rot90=rot90)
                         for c in range(C)] for t in range(T)]
            proj = np.asarray(oriented)  # (T,C,Y',X')
        out_arr = np.expand_dims(proj, axis=2)  # (T,C,1,Y',X')
        # Write directly to zarr (faster than iohub for bulk writes)
        out_zarr = zarr.open(str(out_path / pos / "0"), mode="r+")
        out_zarr[:] = out_arr

    # Parallel processing per position
    if num_workers is None:
        num_workers = get_optimal_workers(use_gpu=False)
    print(
        f"Projecting (max-Z) {len(positions)} positions from {source_path.name} using {num_workers} worker(s)"
    )
    Parallel(n_jobs=num_workers)(
        delayed(_project_pos)(pos) for pos in tqdm(positions, desc="Max projecting")
    )

    print(f"Max projection complete: {out_path}")
    return out_path


def run_project_store(store_path: str, num_workers: int | None = None, overwrite: bool | None = None,
                      flipud: bool = False, fliplr: bool = False, rot90: int = 0,
                      experiment: str | None = None, flatfield: bool = True,
                      sigma: int = 75, camera_offset: float = 100.0) -> str:
    """Project a store (+ orientation), then flatfield-correct it — one SLURM job.

    Projection and flatfield were separate steps; folding them avoids a second
    per-store job + store round-trip. Produces <stem>_max_proj.zarr then
    <stem>_max_proj_flatfield.zarr (what stitch/segment consume). Module-level so
    it's picklable for SLURM.
    """
    store_path = Path(store_path)
    out = _max_project_store(store_path, num_workers=num_workers, overwrite=overwrite,
                             flipud=flipud, fliplr=fliplr, rot90=rot90)
    if not flatfield:
        return f"Done (projection only): {out}"

    # Flatfield-correct the projection in the same job.
    from cyclops_process.processes.flatfield_correction import correct_flatfield
    from ops_utils.data.experiment import OpsDataset
    corrected = out.with_name(out.stem.replace("_max_proj", "_max_proj_flatfield") + ".zarr")
    output_dir = None
    if experiment:
        output_dir = (OpsDataset(experiment).experiment_path / "3-assembly"
                      / "illumination_correction" / store_path.stem)
    print(f"\nFlatfield: {out.name} -> {corrected.name}")
    correct_flatfield(
        experiment=experiment, num_workers=num_workers, sigma=sigma, camera_offset=camera_offset,
        source_path_override=out, corrected_path_override=corrected, output_dir_override=output_dir,
    )
    return f"Done: {corrected}"


def _resolve_stores(experiment: str, units: list[int], modality: str) -> list[Path]:
    """Map experiment units to their <stem>.zarr convert stores for the modality."""
    m = get_modality(modality)
    convert_dir = m.convert_dir(experiment)
    stores = []
    for unit in units:
        sp = convert_dir / f"{m.unit_stem(unit)}.zarr"
        if sp.exists():
            stores.append(sp)
        else:
            print(f"WARNING: no store found for {m.unit_word} {unit}: {sp}, skipping")
    return stores


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Max-project Z for fixed-cell stores (per position, all channels), via SLURM"
    )
    parser.add_argument("stores", nargs="*", help="Explicit input store paths (manual mode)")
    parser.add_argument("--modality", choices=["cp", "4i"], default="cp", help="Imaging modality (default: cp)")
    parser.add_argument("--experiment", "-e", default=None, help="Experiment name/shorthand; projects per-unit zarrs under 0-convert/<modality>")
    parser.add_argument("--parts", nargs="+", type=int, default=None, help="Units to project (default: modality default units)")
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers (default: auto)")
    parser.add_argument("--force", action="store_true", help="Reproject even if _max_proj.zarr exists")
    parser.add_argument("--flipud", action="store_true", help="Flip each tile vertically during projection")
    parser.add_argument("--fliplr", action="store_true", help="Flip each tile horizontally during projection")
    parser.add_argument("--rot90", type=int, default=0, help="Rotate each tile 90deg k times during projection (odd swaps Y,X)")
    parser.add_argument("--no-flatfield", action="store_true", help="Projection only; skip the folded-in flatfield correction")
    parser.add_argument("--dry-run", action="store_true", help="Print SLURM plan without submitting")
    parser.add_argument("--local", action="store_true", help="Run inline instead of submitting to SLURM")
    args = parser.parse_args(argv)

    m = get_modality(args.modality)
    units = args.parts if args.parts is not None else m.default_units

    # Resolve the list of input stores (experiment mode preferred).
    if args.experiment:
        experiment = resolve_experiment_name(args.experiment, autoselect=True)
        stores = _resolve_stores(experiment, units, args.modality)
    elif args.stores:
        experiment = None
        stores = [Path(s) for s in args.stores]
    else:
        parser.error("provide --experiment or explicit store path(s)")

    # Flatfield is folded in for experiment mode; manual explicit-store mode
    # (no experiment) does projection only.
    do_flatfield = (experiment is not None) and (not args.no_flatfield)

    # Build jobs, skipping stores already done (unless --force). The final output
    # is the flatfield-corrected projection when flatfield is on.
    jobs = []
    for sp in stores:
        proj = sp.with_name(f"{sp.stem}_max_proj.zarr")
        final = proj.with_name(proj.stem + "_flatfield.zarr") if do_flatfield else proj
        if final.exists() and not args.force:
            print(f"SKIP: {final.name} exists (use --force to redo)")
            continue
        jobs.append({
            "name": f"project_{m.name}_{sp.stem}",
            "func": run_project_store,
            "kwargs": {"store_path": str(sp), "num_workers": args.workers, "overwrite": True,
                       "flipud": args.flipud, "fliplr": args.fliplr, "rot90": args.rot90,
                       "experiment": experiment, "flatfield": do_flatfield},
            "metadata": {"type": f"project_{m.name}", "store": sp.stem},
        })

    if not jobs:
        print("Nothing to project (all outputs exist).")
        return

    if args.dry_run:
        print(f"Would submit {len(jobs)} job(s): " + ", ".join(j["name"] for j in jobs))
        return

    # Manual mode without an experiment runs inline unless SLURM is available.
    if args.local:
        for job in jobs:
            print(f"\n{'='*60}\nRunning {job['name']} locally...")
            kw = dict(job["kwargs"])
            kw["overwrite"] = None if experiment is None else True
            run_project_store(**kw)
        return

    if experiment is None:
        parser.error("SLURM submission needs --experiment (for logging/paths); use --local for manual paths")

    dataset = OpsDataset(experiment)
    submit_parallel_jobs(
        jobs_to_submit=jobs,
        experiment=experiment,
        slurm_params=SLURM_PARAMS,
        log_dir=dataset.experiment_path / "slurm_logs" / f"project_{m.name}",
        manifest_prefix=f"project_{m.name}",
        dry_run=False,
    )


if __name__ == "__main__":
    main()
