"""
This script checks for metadata consistency across a random sample of positions
in an OME-Zarr store.

It is designed to be much faster than reading all metadata, as it only inspects
a small, random subset of positions. This is useful for verifying the assumption
that all positions in a dataset have identical metadata (shape, chunks, dtype, etc.)
without iterating over thousands of files.

If the sampled positions are all consistent, it prints the common metadata,
giving you confidence to use a faster, assumption-based approach for processing
the entire dataset.

Usage:
$ python tests/check_zarr_consistency.py /path/to/your/data.zarr
$ python tests/check_zarr_consistency.py /path/to/your/data.zarr --samples 20
"""
import click
import iohub.ngff
import yaml
import random
from pathlib import Path
import sys

def get_single_position_metadata(position_path: Path):
    """
    Reads the metadata for a single position directly from its path.
    """
    metadata = {}
    # We open the position directly, which is faster than opening the whole dataset
    with iohub.ngff.open_ome_zarr(position_path, mode='r') as pos:
        # The key metadata is in the '0' image array
        img_array = pos['0']
        metadata['shape'] = img_array.shape
        metadata['chunks'] = img_array.chunks
        metadata['dtype'] = str(img_array.dtype)
        metadata['transform'] = pos.zattrs.get('multiscales', [{}])[0].get('datasets', [{}])[0].get('coordinateTransformations')
    return metadata

@click.command()
@click.argument('zarr_path', type=click.Path(exists=True, dir_okay=True, readable=True))
@click.option('--samples', default=15, help='Number of random positions to check.', show_default=True)
def check_zarr_consistency(zarr_path, samples):
    """
    Checks if a random sample of positions in a Zarr store have identical metadata.
    """
    zarr_path = Path(zarr_path)
    print(f"--- Checking metadata consistency for: {zarr_path} ---")
    print(f"--- Sampling {samples} random positions ---")

    try:
        # Efficiently find all position directories using glob.
        # This avoids the slow metadata scan of `dataset.positions()`.
        # Assumes a standard HCS layout of Row/Column/FOV.
        position_paths = sorted(list(zarr_path.glob('*/*/*')))
        # Filter for directories that look like positions (contain a '0' array)
        position_paths = [p for p in position_paths if (p / '0').is_dir()]

        if not position_paths:
            print("Error: No positions found. Check the Zarr path and structure.", file=sys.stderr)
            return

        print(f"Found {len(position_paths)} total positions.")

        if len(position_paths) <= samples:
            print("Number of positions is less than or equal to sample size. Checking all positions.")
            sampled_paths = position_paths
        else:
            sampled_paths = random.sample(position_paths, samples)

        # Get metadata for the first sampled position to use as a reference
        print(f"Reading metadata for reference position: {sampled_paths[0].relative_to(zarr_path.parent)}")
        reference_meta = get_single_position_metadata(sampled_paths[0])
        
        all_match = True
        # Compare the rest of the samples to the reference
        for i in range(1, len(sampled_paths)):
            path = sampled_paths[i]
            print(f"Checking metadata for position: {path.relative_to(zarr_path.parent)}")
            current_meta = get_single_position_metadata(path)
            if current_meta != reference_meta:
                all_match = False
                print("\n---!!! METADATA MISMATCH FOUND !!!---")
                print(f"Position: {path.name}")
                print("\nReference Metadata:")
                print(yaml.dump(reference_meta))
                print("\nMismatching Metadata:")
                print(yaml.dump(current_meta))
                break # Stop after the first mismatch

        print("\n--- Verification Result ---")
        if all_match:
            print("✅ SUCCESS: All randomly sampled positions have identical metadata.")
            print("\nYou can likely speed up processing by assuming this metadata is consistent for all positions.")
            print("\nCommon Metadata:")
            print(yaml.dump(reference_meta))
        else:
            print("❌ FAILURE: Metadata inconsistency was found among the sampled positions.")

    except Exception as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
        print("Please ensure the path points to a valid OME-Zarr store with an HCS layout.", file=sys.stderr)

if __name__ == '__main__':
    check_zarr_consistency() 