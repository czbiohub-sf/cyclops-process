"""
This script serves two purposes for inspecting OME-Zarr stores.

1.  **Inspect a Single Zarr Store:**
    When given a path to a single Zarr store (`.zarr` directory), it prints key
    metadata such as channel names, Zarr attributes, and information about the
    first few positions (shape, chunks, data type, etc.). It also saves this
    metadata to a corresponding `.txt` file in the same directory as the script.

    Command-line example:
    $ python tests/read_zarr_meta.py /path/to/your/single_store.zarr

2.  **Compare Multiple Zarr Stores in a Directory:**
    When given a path to a directory, it discovers all Zarr stores within that
    directory. It then performs a pairwise comparison of their metadata to check
    if they are identical. It prints a summary of which pairs match and which do not.

    Command-line example:
    $ python tests/read_zarr_meta.py /path/to/your/directory/
    $ python tests/read_zarr_meta.py /path/to/ops_data/ops0042_20250520/1-preprocess/live_imaging/virtual_staining/phenotyping_max_proj.zarr
"""
import click
import iohub.ngff
import yaml
import os
from pathlib import Path
import sys
import itertools
from tqdm import tqdm
import random
from deepdiff import DeepDiff


def get_zarr_metadata(zarr_path, sample_n=None, positions_to_process_names=None):
    """
    Extracts key metadata from a Zarr store for comparison.
    - If positions_to_process_names is provided, it will process only those positions.
    - Otherwise, if sample_n is provided, it will sample randomly.
    - Otherwise, it will process all positions.
    """
    metadata = {}
    with iohub.ngff.open_ome_zarr(zarr_path, mode='r') as dataset:
        metadata['channel_names'] = dataset.channel_names
        metadata['zattrs'] = dataset.zattrs
        
        all_positions_list = list(dataset.positions())
        all_positions_dict = dict(all_positions_list)
        metadata['position_count'] = len(all_positions_list)
        
        if positions_to_process_names is not None:
            # Use the exact list of positions provided for comparison
            if not all(name in all_positions_dict for name in positions_to_process_names):
                raise ValueError(f"One or more provided positions not found in {zarr_path}")
            positions_to_process = [(name, all_positions_dict[name]) for name in positions_to_process_names]
            desc = f"Reading {len(positions_to_process)} specific positions from {Path(zarr_path).name}"
        elif sample_n and len(all_positions_list) > sample_n:
            # Randomly sample from all positions (for single file inspection)
            positions_to_process = random.sample(all_positions_list, sample_n)
            desc = f"Reading {sample_n} random positions"
        else:
            # Process all positions
            positions_to_process = all_positions_list
            desc = "Reading all positions"

        # To keep comparison stable, sort positions by name
        sorted_positions = sorted(positions_to_process, key=lambda x: x[0])
        
        pos_meta = {}
        for pos_name, pos_object in tqdm(sorted_positions, desc=desc, unit="pos"):
            pos_meta[pos_name] = {
                'shape': pos_object.data.shape,
                'chunks': pos_object.data.chunks,
                'dtype': str(pos_object.data.dtype), # Use string for comparison
                'scale': pos_object.scale,
                'translation': getattr(pos_object, 'translation', None)
            }
        metadata['positions'] = pos_meta
    return metadata

def compare_zarr_stores(path1, path2, sample_n=None):
    """
    Compares the metadata of two Zarr stores using the same random sample of positions.
    Returns a tuple: (is_match, differences_string)
    """
    try:
        # Get the full list of positions from the first store to define the sampling universe
        with iohub.ngff.open_ome_zarr(path1, mode='r') as ds1:
            all_pos_names_ds1 = [p[0] for p in ds1.positions()]

        # Create one definitive sample of position names
        if sample_n and len(all_pos_names_ds1) > sample_n:
            sampled_pos_names = random.sample(all_pos_names_ds1, sample_n)
        else:
            sampled_pos_names = all_pos_names_ds1
            
        # Get metadata for the *exact same sample* of positions from both stores
        meta1 = get_zarr_metadata(path1, positions_to_process_names=sampled_pos_names)
        meta2 = get_zarr_metadata(path2, positions_to_process_names=sampled_pos_names)
        
        # We don't need to compare the total position count, just the sampled metadata
        if 'position_count' in meta1: del meta1['position_count']
        if 'position_count' in meta2: del meta2['position_count']
        
        diff = DeepDiff(meta1, meta2, ignore_order=True)
        
        if not diff:
            return (True, "MATCH")
        else:
            pretty_diff = yaml.dump(diff, indent=2)
            return (False, f"MISMATCH:\n{pretty_diff}")

    except Exception as e:
        print(f"Error comparing {path1} and {path2}: {e}", file=sys.stderr)
        return (False, f"ERROR: {e}")

def print_zarr_metadata(zarr_path):
    """
    Opens an OME-Zarr file, prints its metadata, and saves it to a file.
    """
    output_lines = []
    
    try:
        # Determine output file path
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        zarr_name = Path(zarr_path).name
        output_filename = f"{zarr_name}_metadata.txt"
        output_filepath = script_dir / output_filename
        
        with iohub.ngff.open_ome_zarr(zarr_path, mode='r') as dataset:
            output_lines.append(f"--- Metadata for: {zarr_path} ---")

            if dataset.channel_names:
                output_lines.append("\n# Channels:")
                output_lines.append(str(dataset.channel_names))

            output_lines.append("\n# Top-level Zarr Attributes (zattrs):")
            output_lines.append(yaml.dump(dataset.zattrs))

            output_lines.append("\n# Positions:")
            position_list = list(dataset.positions())
            if not position_list:
                output_lines.append("No positions found in this dataset.")
            else:
                output_lines.append(f"Found {len(position_list)} positions.")
                # Display only the first 5 for brevity
                sorted_positions = sorted(position_list, key=lambda x: x[0])
                for i, (pos_name, pos_object) in enumerate(sorted_positions):
                    if i < 5:
                        output_lines.append(f"\n## Position: {pos_name}")
                        output_lines.append(f"  - Shape: {pos_object.data.shape}")
                        output_lines.append(f"  - Chunks: {pos_object.data.chunks}")
                        output_lines.append(f"  - Data Type: {pos_object.data.dtype}")
                        output_lines.append(f"  - Scale Transform: {pos_object.scale}")
                        if hasattr(pos_object, 'translation') and pos_object.translation:
                             output_lines.append(f"  - Translation Transform: {pos_object.translation}")
                    elif i == 5:
                        output_lines.append("\n... and more. Displaying first 5 positions only.")
                        break
        
        final_output = "\n".join(output_lines)
        
        # Write to file
        with open(output_filepath, 'w') as f:
            f.write(final_output)
            
        # Print to console as well
        print(final_output)
        print(f"\n--- Metadata also saved to: {output_filepath} ---")

    except AttributeError as e:
        print(f"A metadata attribute was not found. This can happen with non-standard Zarr files. Error: {e}", file=sys.stderr)
        final_output = "\n".join(output_lines)
        print("--- Partial Data Collected ---")
        print(final_output)

    except Exception as e:
        print(f"An error occurred while reading {zarr_path}: {e}", file=sys.stderr)

@click.command()
@click.argument('path', type=click.Path(exists=True, readable=True))
@click.option('--samples', 'sample_n', default=15, help='Number of random positions to subsample for comparison.', show_default=True)
def main(path, sample_n):
    """
    Reads metadata from a single OME-Zarr store or compares all Zarr stores in a directory.
    """
    target_path = Path(path)

    # First, check if the path provided is itself a Zarr store (a dir ending in .zarr)
    if target_path.is_dir() and str(target_path).endswith('.zarr'):
        # It's a single Zarr store
        print_zarr_metadata(target_path)

    # If not, check if it's a directory that we should search inside
    elif target_path.is_dir():
        # It's a directory containing multiple Zarr stores, find and compare them
        print(f"--- Comparing Zarr stores in directory: {target_path} ---")
        print(f"--- Subsampling {sample_n} positions from each store for speed ---")
        
        # Define an output file for the detailed report in the target directory
        report_filename = f"comparison_report_{target_path.name}.txt"
        report_filepath = target_path / report_filename

        with open(report_filepath, 'w') as report_file:
            report_file.write(f"--- Comparison Report for Zarr stores in: {target_path} ---\n")
            report_file.write(f"--- Using a sample size of {sample_n} positions per store ---\n\n")

            zarr_stores = sorted(list(target_path.glob('*.zarr')))
            
            if len(zarr_stores) < 2:
                message = "Found fewer than two .zarr directories to compare."
                print(message)
                report_file.write(message)
                return
                
            print(f"Found {len(zarr_stores)} stores: {[p.name for p in zarr_stores]}")
            
            # Generate all unique pairs for comparison
            comparison_pairs = list(itertools.combinations(zarr_stores, 2))
            
            results = []
            print("\n--- Comparison Results ---")
            # Header for the console results table
            print(f"{'File 1':<30} | {'File 2':<30} | {'Result'}")
            print(f"-"*30 + " | " + "-"*30 + " | " + "-"*8)

            for path1, path2 in tqdm(comparison_pairs, desc="Comparing stores"):
                match, reason = compare_zarr_stores(path1, path2, sample_n=sample_n)
                results.append({'file1': path1.name, 'file2': path2.name, 'match': match, 'reason': reason})
                status = "MATCH" if match else "NO MATCH"
                print(f"{path1.name:<30} | {path2.name:<30} | {status}")
            
            # Write detailed discrepancies to the report file
            report_file.write("\n--- Summary & Discrepancies ---\n")
            any_mismatch = False
            for res in results:
                if not res['match']:
                    any_mismatch = True
                    report_file.write(f"\nDiscrepancy between {res['file1']} and {res['file2']}:\n")
                    report_file.write(f"{res['reason']}\n")
                    report_file.write("-" * 70 + "\n")

            if not any_mismatch:
                report_file.write("All compared pairs matched successfully.\n")
        
        print(f"\n--- Detailed report saved to: {report_filepath} ---")

    else:
        print(f"Error: Path '{target_path}' is not a file or a directory as expected.", file=sys.stderr)

if __name__ == '__main__':
    main() 