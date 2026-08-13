#!/bin/bash
#SBATCH --job-name=vs_combine
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --partition=cpu

# Combine batch predictions into HCS OME-Zarr store
#
# This script handles the output format from viscy_batch_inference.py which writes:
#   intermediate_dir/predictions_{start}_{end}.zarr/batch_NNNNN/
#
# And combines them into a proper HCS OME-Zarr structure:
#   output_store/Row/Well/Position/0
#
# Usage: sbatch combine_batch.sh <intermediate_dir> <input_store> <output_store>
#
# Arguments:
#   intermediate_dir: Directory containing predictions_X_Y.zarr stores
#   input_store:      Original input zarr (for position ordering and metadata)
#   output_store:     Output HCS OME-Zarr store path

set -e

INTERMEDIATE_DIR=$1
INPUT_STORE=$2
OUTPUT_STORE=$3

if [ -z "$INTERMEDIATE_DIR" ] || [ -z "$INPUT_STORE" ] || [ -z "$OUTPUT_STORE" ]; then
    echo "Usage: $0 <intermediate_dir> <input_store> <output_store>"
    echo ""
    echo "Arguments:"
    echo "  intermediate_dir  Directory containing predictions_X_Y.zarr stores"
    echo "  input_store       Original input zarr (for position ordering)"
    echo "  output_store      Output HCS OME-Zarr store path"
    exit 1
fi

echo "============================================================"
echo "VS Combine Batch Predictions"
echo "============================================================"
echo "Intermediate: $INTERMEDIATE_DIR"
echo "Input store:  $INPUT_STORE"
echo "Output store: $OUTPUT_STORE"
echo "Node:         $(hostname)"
echo "Time:         $(date)"
echo "============================================================"

# cyclops_process should be installed via pip install -e in the conda env
python -m cyclops_process.utils.combine_batch_predictions \
    --intermediate-dir "$INTERMEDIATE_DIR" \
    --input-store "$INPUT_STORE" \
    --output-store "$OUTPUT_STORE" \
    --batch-size 7 \
    --num-workers 16

echo ""
echo "============================================================"
echo "Combine job completed successfully at $(date)"
echo "============================================================"
