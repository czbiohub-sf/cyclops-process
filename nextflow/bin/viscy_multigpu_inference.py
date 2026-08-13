#!/usr/bin/env python
"""
Multi-GPU batch inference for viscy virtual staining.

Spawns multiple subprocesses, each assigned to a different GPU,
to process positions in parallel. Uses async HCS writer to maintain
proper FOV structure while overlapping I/O with GPU compute.

Usage:
    python viscy_multigpu_inference.py \
        --input-store /path/to/input.zarr \
        --output-dir /path/to/output \
        --config /path/to/predict.yml \
        --start-pos 0 \
        --end-pos 1000 \
        --num-gpus 4

    # Or run as a single-GPU worker (called by main script):
    python viscy_multigpu_inference.py \
        --worker \
        --gpu-id 0 \
        --input-store ... \
        --output-dir ... \
        --config ... \
        --start-pos 0 \
        --end-pos 500
"""

import argparse
import json
import os
import sys
import time
import subprocess
from pathlib import Path


def run_single_gpu_worker(
    gpu_id: int,
    input_store: Path,
    output_dir: Path,
    config_path: Path,
    start_pos: int,
    end_pos: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
):
    """
    Run inference on a single GPU. Called when --worker flag is set.
    Uses AsyncHCSPredictionWriter to maintain FOV structure with async I/O.
    """
    # Set CUDA device BEFORE importing torch
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    import torch
    import numpy as np
    import yaml
    import shutil

    from viscy.data.hcs import HCSDataModule
    from viscy.translation.engine import VSUNet
    from iohub.ngff import open_ome_zarr

    # iohub compat shim: upgraded iohub renamed ImageArray.name -> .path.
    # viscy's HCS dataloader still reads img.name to tag samples by position
    # (viscy/data/hcs.py), so alias it back to the full array path.
    from iohub.ngff import ImageArray as _ImageArray
    if not hasattr(_ImageArray, "name"):
        _ImageArray.name = property(lambda self: self.path)

    # Add cyclops_process to path so async_hcs_writer can be imported from
    # the viscy Python env (which doesn't have cyclops_process installed)
    import pathlib
    _ops_process_dir = str(pathlib.Path(__file__).resolve().parents[2])
    if _ops_process_dir not in sys.path:
        sys.path.insert(0, _ops_process_dir)
    from cyclops_process.utils.async_hcs_writer import AsyncHCSPredictionWriter

    worker_start = time.perf_counter()

    print(f"[GPU {gpu_id}] Starting worker for positions {start_pos}-{end_pos}", flush=True)
    print(f"[GPU {gpu_id}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"[GPU {gpu_id}] torch.cuda.device_count()={torch.cuda.device_count()}", flush=True)

    num_positions = end_pos - start_pos
    if num_positions <= 0:
        print(f"[GPU {gpu_id}] No positions to process", flush=True)
        return

    # Create subset store
    subset_store = output_dir / f"_subset_gpu{gpu_id}_{start_pos}_{end_pos}"

    try:
        print(f"[GPU {gpu_id}] Creating subset store...", flush=True)
        _create_subset_store(input_store, subset_store, start_pos, end_pos)

        # Load config
        with open(config_path) as f:
            config = yaml.safe_load(f)

        ckpt_path = Path(config['ckpt_path'])

        # Load model
        print(f"[GPU {gpu_id}] Loading model...", flush=True)
        model_config = config['model']['init_args']
        model = VSUNet(**model_config)
        checkpoint = torch.load(ckpt_path, map_location='cuda:0', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.cuda().eval()
        print(f"[GPU {gpu_id}] Model loaded", flush=True)

        # Setup data module
        source_channel = config.get('data', {}).get('init_args', {}).get('source_channel', 'Phase3D')
        target_channel = config.get('data', {}).get('init_args', {}).get('target_channel', ['nuclei', 'membrane'])
        z_window_size = config.get('data', {}).get('init_args', {}).get('z_window_size', 1)

        dm = HCSDataModule(
            data_path=subset_store,
            source_channel=source_channel,
            target_channel=target_channel,
            z_window_size=z_window_size,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=True,
        )
        dm.prepare_data()
        dm.setup(stage='predict')
        dataloader = dm.predict_dataloader()

        print(f"[GPU {gpu_id}] Dataset: {len(dm.predict_dataset)} items, {len(dataloader)} batches", flush=True)

        # Get scale metadata from input store
        dataset_scale = None
        with open_ome_zarr(input_store, mode='r') as plate:
            for _, pos in plate.positions():
                dataset_scale = pos.scale
                break

        # Calculate z_padding for 2.5D
        z_padding = z_window_size // 2

        # Warmup
        print(f"[GPU {gpu_id}] Warmup...", flush=True)
        warmup_iter = iter(dataloader)
        for _ in range(min(2, len(dataloader))):
            try:
                batch = next(warmup_iter)
                with torch.no_grad():
                    x = batch['source'].cuda()
                    _ = model(x)
            except StopIteration:
                break
        torch.cuda.synchronize()

        # Recreate dataloader
        dm.setup(stage='predict')
        dataloader = dm.predict_dataloader()

        # Create output store path for this GPU
        output_store_path = output_dir / f"gpu{gpu_id}.zarr"

        # Run inference with async HCS writing
        print(f"[GPU {gpu_id}] Running inference with async HCS writing...", flush=True)
        inference_start = time.perf_counter()
        total_batches = len(dataloader)
        last_print_time = inference_start

        with AsyncHCSPredictionWriter(
            output_store=str(output_store_path),
            channel_names=target_channel if isinstance(target_channel, list) else [target_channel],
            z_padding=z_padding,
            num_workers=8,  # Parallel writer threads
            dataset_scale=dataset_scale,
            input_store=str(subset_store),
        ) as writer:
            for i, batch in enumerate(dataloader):
                with torch.no_grad():
                    x = batch['source'].cuda()
                    output = model(x)
                torch.cuda.synchronize()

                # Queue for async writing (maintains FOV structure)
                writer.write_batch(batch, output)

                # Print progress every 10 seconds
                current_time = time.perf_counter()
                elapsed = current_time - inference_start
                if current_time - last_print_time >= 10 or i == total_batches - 1:
                    pct = 100 * (i + 1) / total_batches
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    remaining = (total_batches - i - 1) / rate if rate > 0 else 0
                    print(f"[GPU {gpu_id}] {i+1}/{total_batches} batches ({pct:.1f}%) - {rate:.1f} batch/s - ETA {remaining:.0f}s", flush=True)
                    last_print_time = current_time

            # Wait for all pending writes
            writer.wait_pending()

        inference_time = time.perf_counter() - inference_start
        total_time = time.perf_counter() - worker_start

        print(f"[GPU {gpu_id}] COMPLETE: {num_positions} positions in {total_time:.1f}s ({num_positions/inference_time:.1f} pos/sec)", flush=True)

    except Exception as e:
        import traceback
        print(f"[GPU {gpu_id}] ERROR: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Cleanup subset store
        if subset_store.exists():
            shutil.rmtree(subset_store)


def _create_subset_store(input_store: Path, output_store: Path, start_pos: int, end_pos: int):
    """Create a subset store with positions in the specified range."""
    import shutil

    if output_store.exists():
        shutil.rmtree(output_store)

    output_store.mkdir(parents=True, exist_ok=True)

    # Detect zarr format
    is_v3 = (input_store / 'zarr.json').exists()

    # Copy plate-level metadata
    if is_v3:
        src = input_store / 'zarr.json'
        if src.exists():
            shutil.copy(src, output_store / 'zarr.json')
    else:
        for meta_file in ['.zattrs', '.zgroup']:
            src = input_store / meta_file
            if src.exists():
                shutil.copy(src, output_store / meta_file)

    # Collect all positions
    all_positions = []
    for row_dir in sorted(input_store.iterdir()):
        if not row_dir.is_dir() or row_dir.name.startswith('.'):
            continue
        for well_dir in sorted(row_dir.iterdir()):
            if not well_dir.is_dir() or well_dir.name.startswith('.'):
                continue
            for pos_dir in sorted(well_dir.iterdir()):
                if not pos_dir.is_dir() or pos_dir.name.startswith('.'):
                    continue
                all_positions.append((row_dir.name, well_dir.name, pos_dir.name, pos_dir))

    # Select positions in range
    selected = all_positions[start_pos:end_pos]

    # Build subset store with symlinks
    included_wells = set()
    for row_name, well_name, pos_name, pos_path in selected:
        row_path = output_store / row_name
        row_path.mkdir(exist_ok=True)

        if is_v3:
            row_meta = input_store / row_name / 'zarr.json'
            if row_meta.exists() and not (row_path / 'zarr.json').exists():
                shutil.copy(row_meta, row_path / 'zarr.json')
        else:
            row_meta = input_store / row_name / '.zgroup'
            if row_meta.exists() and not (row_path / '.zgroup').exists():
                shutil.copy(row_meta, row_path / '.zgroup')

        well_path = row_path / well_name
        well_path.mkdir(exist_ok=True)

        if is_v3:
            well_json = input_store / row_name / well_name / 'zarr.json'
            if well_json.exists() and not (well_path / 'zarr.json').exists():
                shutil.copy(well_json, well_path / 'zarr.json')
        else:
            well_zgroup = input_store / row_name / well_name / '.zgroup'
            if well_zgroup.exists() and not (well_path / '.zgroup').exists():
                shutil.copy(well_zgroup, well_path / '.zgroup')

        pos_link = well_path / pos_name
        if not pos_link.exists():
            pos_link.symlink_to(pos_path)

        included_wells.add((row_name, well_name))

    # Rebuild plate metadata with only included wells
    if is_v3:
        plate_json = input_store / 'zarr.json'
        if plate_json.exists():
            with open(plate_json) as f:
                plate_meta = json.load(f)
            ome = plate_meta.get('attributes', {}).get('ome', {})
            if 'plate' in ome and 'wells' in ome['plate']:
                ome['plate']['wells'] = [
                    w for w in ome['plate']['wells']
                    if (w['path'].split('/')[0], w['path'].split('/')[1]) in included_wells
                ]
            with open(output_store / 'zarr.json', 'w') as f:
                json.dump(plate_meta, f, indent=2)
    else:
        plate_zattrs = input_store / '.zattrs'
        if plate_zattrs.exists():
            with open(plate_zattrs) as f:
                plate_meta = json.load(f)
            if 'plate' in plate_meta and 'wells' in plate_meta['plate']:
                plate_meta['plate']['wells'] = [
                    w for w in plate_meta['plate']['wells']
                    if (w['path'].split('/')[0], w['path'].split('/')[1]) in included_wells
                ]
            with open(output_store / '.zattrs', 'w') as f:
                json.dump(plate_meta, f, indent=2)

    # Rebuild well metadata for each well
    for row_name, well_name in included_wells:
        included_positions = set(
            p[2] for p in selected
            if p[0] == row_name and p[1] == well_name
        )
        if is_v3:
            well_json_src = input_store / row_name / well_name / 'zarr.json'
            well_json_dst = output_store / row_name / well_name / 'zarr.json'
            if well_json_src.exists():
                with open(well_json_src) as f:
                    well_meta = json.load(f)
                ome = well_meta.get('attributes', {}).get('ome', {})
                if 'well' in ome and 'images' in ome['well']:
                    ome['well']['images'] = [
                        img for img in ome['well']['images']
                        if img['path'] in included_positions
                    ]
                with open(well_json_dst, 'w') as f:
                    json.dump(well_meta, f, indent=2)
        else:
            well_zattrs_src = input_store / row_name / well_name / '.zattrs'
            well_zattrs_dst = output_store / row_name / well_name / '.zattrs'
            if well_zattrs_src.exists():
                with open(well_zattrs_src) as f:
                    well_meta = json.load(f)
                if 'well' in well_meta and 'images' in well_meta['well']:
                    well_meta['well']['images'] = [
                        img for img in well_meta['well']['images']
                        if img['path'] in included_positions
                    ]
                with open(well_zattrs_dst, 'w') as f:
                    json.dump(well_meta, f, indent=2)


def _count_positions_in_store(store_path: Path, expected_z: int = 0) -> int:
    """Count completed positions in an HCS zarr store (row/well/pos hierarchy).

    If expected_z > 0, only counts positions whose array has the full Z depth.
    Partially-written positions (from a cancelled run) are not counted.
    """
    count = 0
    partial = 0
    if not store_path.exists():
        return 0
    for row_dir in store_path.iterdir():
        if not row_dir.is_dir() or row_dir.name.startswith('.'):
            continue
        for well_dir in row_dir.iterdir():
            if not well_dir.is_dir() or well_dir.name.startswith('.'):
                continue
            for pos_dir in well_dir.iterdir():
                if not pos_dir.is_dir() or pos_dir.name.startswith('.'):
                    continue
                if expected_z > 0:
                    arr_dir = pos_dir / "0"
                    zarray_path = arr_dir / "zarr.json" if (arr_dir / "zarr.json").exists() else arr_dir / ".zarray"
                    if not zarray_path.exists():
                        partial += 1
                        continue
                    try:
                        meta = json.loads(zarray_path.read_text())
                        z = meta["shape"][2]
                        if z < expected_z:
                            partial += 1
                            continue
                    except (json.JSONDecodeError, KeyError, IndexError):
                        partial += 1
                        continue
                count += 1
    if partial > 0:
        print(f"  Found {partial} partially-written positions in {store_path.name} "
              f"(Z < {expected_z}), will re-process them")
    return count


def run_multigpu_coordinator(
    input_store: Path,
    output_dir: Path,
    config_path: Path,
    start_pos: int,
    end_pos: int,
    num_gpus: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    resume: bool = False,
):
    """
    Coordinate multi-GPU inference by spawning subprocess workers.
    """
    total_positions = end_pos - start_pos
    positions_per_gpu = (total_positions + num_gpus - 1) // num_gpus

    print("="*70)
    print("Multi-GPU Viscy Batch Inference (HCS Output)")
    print("="*70)
    print(f"  Total positions: {total_positions} ({start_pos} to {end_pos})")
    print(f"  GPUs: {num_gpus}")
    print(f"  Positions per GPU: ~{positions_per_gpu}")
    print(f"  Input: {input_store}")
    print(f"  Output: {output_dir}")
    if resume:
        print(f"  Mode: RESUME (skipping completed positions)")
    print("="*70)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get expected Z depth from input store for resume validation
    expected_z = 0
    if resume:
        for row_dir in sorted(input_store.iterdir()):
            if not row_dir.is_dir() or row_dir.name.startswith('.'):
                continue
            for well_dir in sorted(row_dir.iterdir()):
                if not well_dir.is_dir() or well_dir.name.startswith('.'):
                    continue
                for pos_dir in sorted(well_dir.iterdir()):
                    if not pos_dir.is_dir() or pos_dir.name.startswith('.'):
                        continue
                    zarray_path = pos_dir / "0" / ".zarray"
                    if zarray_path.exists():
                        try:
                            meta = json.loads(zarray_path.read_text())
                            expected_z = meta["shape"][2]
                            print(f"  Expected Z depth from input: {expected_z}")
                        except (json.JSONDecodeError, KeyError, IndexError):
                            pass
                    break
                break
            break

    # Calculate per-GPU position ranges, adjusting for resume
    gpu_ranges = []
    for gpu_id in range(num_gpus):
        gpu_start = start_pos + gpu_id * positions_per_gpu
        gpu_end = min(gpu_start + positions_per_gpu, end_pos)
        if gpu_start >= end_pos:
            continue

        actual_start = gpu_start
        if resume:
            store_path = output_dir / f"gpu{gpu_id}.zarr"
            completed = _count_positions_in_store(store_path, expected_z=expected_z)
            if completed > 0:
                actual_start = gpu_start + completed
                remaining = gpu_end - actual_start
                print(f"\n  GPU {gpu_id}: {completed}/{gpu_end - gpu_start} positions done, "
                      f"{remaining} remaining (positions {actual_start}-{gpu_end})")
                if actual_start >= gpu_end:
                    print(f"  GPU {gpu_id}: All positions complete, skipping")
                    continue

        gpu_ranges.append((gpu_id, actual_start, gpu_end))

    if not gpu_ranges:
        print("\nAll positions already complete!")
        return

    # Clean up stale subset stores from previous failed runs
    if resume:
        import shutil
        for old_subset in output_dir.glob('_subset_gpu*'):
            if old_subset.is_dir():
                print(f"  Cleaning up stale subset: {old_subset.name}")
                shutil.rmtree(old_subset)

    # Spawn worker subprocesses
    processes = []
    for gpu_id, gpu_start, gpu_end in gpu_ranges:
        print(f"\nSpawning worker for GPU {gpu_id}: positions {gpu_start}-{gpu_end}")

        cmd = [
            sys.executable,  # inherit parent interpreter (uv .venv), not a stale conda env
            str(Path(__file__)),
            '--worker',
            '--gpu-id', str(gpu_id),
            '--input-store', str(input_store),
            '--output-dir', str(output_dir),
            '--config', str(config_path),
            '--start-pos', str(gpu_start),
            '--end-pos', str(gpu_end),
            '--batch-size', str(batch_size),
            '--num-workers', str(num_workers),
            '--prefetch-factor', str(prefetch_factor),
        ]

        # Start subprocess with unbuffered output
        worker_env = os.environ.copy()
        worker_env['PYTHONUNBUFFERED'] = '1'
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=worker_env,
        )
        processes.append((gpu_id, p, gpu_end - gpu_start))

    print(f"\n{len(processes)} workers started, waiting for completion...\n")

    # Stream output from all workers
    start_time = time.perf_counter()

    import select
    active_processes = {p.stdout.fileno(): (gpu_id, p, n_pos) for gpu_id, p, n_pos in processes}

    while active_processes:
        # Wait for output from any process
        readable, _, _ = select.select(list(active_processes.keys()), [], [], 1.0)

        for fd in readable:
            gpu_id, p, n_pos = active_processes[fd]
            line = p.stdout.readline()
            if line:
                print(line, end='', flush=True)
            elif p.poll() is not None:
                # Process finished
                del active_processes[fd]

        # Check for finished processes that didn't have output
        for fd in list(active_processes.keys()):
            gpu_id, p, n_pos = active_processes[fd]
            if p.poll() is not None and fd not in readable:
                # Drain remaining output
                for line in p.stdout:
                    print(line, end='', flush=True)
                del active_processes[fd]

    total_time = time.perf_counter() - start_time

    # Check exit codes
    all_success = True
    for gpu_id, p, n_pos in processes:
        if p.returncode != 0:
            print(f"[GPU {gpu_id}] Worker failed with exit code {p.returncode}")
            all_success = False

    # Summary
    print("\n" + "="*70)
    print("MULTI-GPU INFERENCE COMPLETE")
    print("="*70)
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Positions: {total_positions}")
    print(f"  Throughput: {total_positions / total_time:.1f} positions/sec")
    print(f"  Output stores: {output_dir}/gpu*.zarr")
    print("="*70)

    if not all_success:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Multi-GPU viscy batch inference')
    parser.add_argument('--worker', action='store_true', help='Run as worker (internal use)')
    parser.add_argument('--gpu-id', type=int, help='GPU ID for worker')
    parser.add_argument('--input-store', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--start-pos', type=int, required=True)
    parser.add_argument('--end-pos', type=int, required=True)
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=7)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--prefetch-factor', type=int, default=4)
    parser.add_argument('--resume', action='store_true',
                        help='Resume from partial results (skip completed positions)')

    args = parser.parse_args()

    if args.worker:
        # Run as single-GPU worker
        run_single_gpu_worker(
            gpu_id=args.gpu_id,
            input_store=args.input_store,
            output_dir=args.output_dir,
            config_path=args.config,
            start_pos=args.start_pos,
            end_pos=args.end_pos,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
        )
    else:
        # Run as coordinator
        run_multigpu_coordinator(
            input_store=args.input_store,
            output_dir=args.output_dir,
            config_path=args.config,
            start_pos=args.start_pos,
            end_pos=args.end_pos,
            num_gpus=args.num_gpus,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            resume=args.resume,
        )


if __name__ == '__main__':
    main()
