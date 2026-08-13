#!/usr/bin/env python
"""
Batch inference script for viscy virtual staining.

Loads model once, processes multiple positions with async writing.
Designed to be called from SLURM array jobs.

Usage:
    python viscy_batch_inference.py \
        --input-store /path/to/input.zarr \
        --output-dir /path/to/output \
        --config /path/to/predict.yml \
        --start-pos 0 \
        --end-pos 100 \
        [--batch-size 7] \
        [--num-workers 8]
"""

import argparse
import json
import time
import sys
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # optional GPU dep; used only in GPU runtime paths
    torch = None
import numpy as np
import yaml
import zarr

from ops_utils.io.async_zarr_writer import AsyncZarrWriter

# --- iohub>=0.3.7 compatibility shim (Fix 0k) ---------------------------------
# iohub 0.3.7 (pulled in for the ISS branch's NGFF-v0.4 channel-name parsing)
# dropped `ImageArray.name`, which the pinned viscy still reads in its HCS
# dataloader (`viscy/data/hcs.py`: `(img.name, t, z)`). Here that sample tag is
# vestigial -- predictions are written keyed by BATCH ORDER (`batch_{i:05d}`),
# never by name -- so restoring `.name` as the in-store path is safe and only
# prevents the AttributeError. Applied in the main process before the DataLoader
# forks its workers (Linux fork inherits the patch).
from iohub.ngff import ImageArray as _ImageArray
if not hasattr(_ImageArray, "name"):
    _ImageArray.name = property(lambda self: self.path)


def create_subset_store(input_store: Path, output_store: Path, start_pos: int, end_pos: int) -> list:
    """
    Create a subset store with positions in the specified range.

    Returns list of position paths that were included.
    """
    import shutil

    print(f"Creating subset store for positions {start_pos} to {end_pos}...")

    # Remove existing subset store
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
    print(f"  Selected {len(selected)} positions from {len(all_positions)} total")

    # Build subset store with symlinks
    position_info = []
    for row_name, well_name, pos_name, pos_path in selected:
        # Create row directory
        row_path = output_store / row_name
        row_path.mkdir(exist_ok=True)

        # Copy row metadata
        if is_v3:
            row_meta = input_store / row_name / 'zarr.json'
            if row_meta.exists() and not (row_path / 'zarr.json').exists():
                shutil.copy(row_meta, row_path / 'zarr.json')
        else:
            row_meta = input_store / row_name / '.zgroup'
            if row_meta.exists() and not (row_path / '.zgroup').exists():
                shutil.copy(row_meta, row_path / '.zgroup')

        # Create well directory
        well_path = row_path / well_name
        well_path.mkdir(exist_ok=True)

        # Copy well group metadata (images list rebuilt below)
        if is_v3:
            well_json = input_store / row_name / well_name / 'zarr.json'
            if well_json.exists() and not (well_path / 'zarr.json').exists():
                shutil.copy(well_json, well_path / 'zarr.json')
        else:
            well_zgroup = input_store / row_name / well_name / '.zgroup'
            if well_zgroup.exists() and not (well_path / '.zgroup').exists():
                shutil.copy(well_zgroup, well_path / '.zgroup')

        # Symlink position directory
        pos_link = well_path / pos_name
        if not pos_link.exists():
            pos_link.symlink_to(pos_path)

        position_info.append({
            'row': row_name,
            'well': well_name,
            'position': pos_name,
            'path': pos_path,
        })

    # Rebuild plate metadata with only included wells
    included_wells = set()
    for p in position_info:
        included_wells.add((p['row'], p['well']))

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
            p['position'] for p in position_info
            if p['row'] == row_name and p['well'] == well_name
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

    print(f"  Subset store created at {output_store}")
    return position_info


def run_batch_inference(
    input_store: Path,
    output_dir: Path,
    config_path: Path,
    start_pos: int,
    end_pos: int,
    batch_size: int = 7,
    num_workers: int = 8,
    prefetch_factor: int = 4,
):
    """
    Run batch inference on positions in range [start_pos, end_pos).

    Uses async writing to overlap I/O with GPU compute.
    """
    from viscy.data.hcs import HCSDataModule
    from viscy.translation.engine import VSUNet

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Get checkpoint path from config
    ckpt_path = Path(config['ckpt_path'])

    print(f"\n{'='*70}")
    print(f"Viscy Batch Inference")
    print(f"  Positions: {start_pos} to {end_pos}")
    print(f"  Input: {input_store}")
    print(f"  Output: {output_dir}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Batch size: {batch_size}")
    print(f"  Workers: {num_workers}")
    print(f"{'='*70}\n")

    # Create temp subset store
    subset_store = output_dir / f"_subset_{start_pos}_{end_pos}"
    position_info = create_subset_store(input_store, subset_store, start_pos, end_pos)

    if len(position_info) == 0:
        print("No positions to process!")
        return

    # Load model
    print("\nLoading model...")
    model_config = config['model']['init_args']
    model = VSUNet(**model_config)
    checkpoint = torch.load(ckpt_path, map_location='cuda:0', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda().eval()
    print("Model loaded")

    # Setup data module
    print("\nSetting up data module...")
    source_channel = config.get('data', {}).get('init_args', {}).get('source_channel', 'Phase3D')
    target_channel = config.get('data', {}).get('init_args', {}).get('target_channel', ['nuclei', 'membrane'])

    dm = HCSDataModule(
        data_path=subset_store,
        source_channel=source_channel,
        target_channel=target_channel,
        z_window_size=1,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
    )
    dm.prepare_data()
    dm.setup(stage='predict')
    dataloader = dm.predict_dataloader()

    print(f"Dataset size: {len(dm.predict_dataset)} items")
    print(f"Dataloader batches: {len(dataloader)}")

    # Get output channels from model config
    out_channels = model_config.get('out_channels', 2)
    output_channel_names = target_channel if isinstance(target_channel, list) else [target_channel]

    # Create output store
    output_store_path = output_dir / f"predictions_{start_pos}_{end_pos}.zarr"
    print(f"\nOutput store: {output_store_path}")

    # Warmup
    print("\nWarmup (2 batches)...")
    warmup_iter = iter(dataloader)
    for _ in range(2):
        try:
            batch = next(warmup_iter)
            with torch.no_grad():
                x = batch['source'].cuda()
                _ = model(x)
        except StopIteration:
            break
    torch.cuda.synchronize()

    # Run inference with async writing
    print(f"\nRunning inference with async writing...")

    # Recreate dataloader after warmup consumed some batches
    dm.setup(stage='predict')
    dataloader = dm.predict_dataloader()

    with AsyncZarrWriter(output_store_path, max_queue_size=4, mode='w') as writer:
        batch_times = []
        total_start = time.perf_counter()

        for i, batch in enumerate(dataloader):
            batch_start = time.perf_counter()

            # GPU inference
            with torch.no_grad():
                x = batch['source'].cuda()
                output = model(x)
            torch.cuda.synchronize()

            # Async write (D2H + queue)
            output_cpu = output.cpu().numpy()

            writer.write(
                f'batch_{i:05d}',
                output_cpu,
                chunks=output_cpu.shape,
                dtype=np.float32,
            )

            batch_time = time.perf_counter() - batch_start
            batch_times.append(batch_time)

            if i < 5 or i % 50 == 0 or i == len(dataloader) - 1:
                print(f"  Batch {i:4d}/{len(dataloader)}: {batch_time:.3f}s")

    total_time = time.perf_counter() - total_start

    # Cleanup subset store
    import shutil
    shutil.rmtree(subset_store)

    # Summary
    print(f"\n{'='*70}")
    print(f"INFERENCE COMPLETE")
    print(f"  Positions: {len(position_info)}")
    print(f"  Batches: {len(batch_times)}")
    print(f"  Avg batch time: {sum(batch_times)/len(batch_times):.3f}s")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Throughput: {len(position_info) / total_time:.1f} positions/sec")
    print(f"  Output: {output_store_path}")
    print(f"{'='*70}")

    return output_store_path


def main():
    parser = argparse.ArgumentParser(description='Batch viscy inference with async writing')
    parser.add_argument('--input-store', type=Path, required=True, help='Input zarr store')
    parser.add_argument('--output-dir', type=Path, required=True, help='Output directory')
    parser.add_argument('--config', type=Path, required=True, help='Viscy predict config')
    parser.add_argument('--start-pos', type=int, required=True, help='Start position index')
    parser.add_argument('--end-pos', type=int, required=True, help='End position index (exclusive)')
    parser.add_argument('--batch-size', type=int, default=7, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=8, help='DataLoader workers')
    parser.add_argument('--prefetch-factor', type=int, default=4, help='Prefetch factor')

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_batch_inference(
        input_store=args.input_store,
        output_dir=args.output_dir,
        config_path=args.config,
        start_pos=args.start_pos,
        end_pos=args.end_pos,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )


if __name__ == '__main__':
    main()
