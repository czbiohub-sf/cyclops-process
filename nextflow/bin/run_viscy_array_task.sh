#!/bin/bash
#SBATCH --job-name=viscy_batch
#SBATCH --output=logs/viscy_batch_%A_%a.out
#SBATCH --error=logs/viscy_batch_%A_%a.err

# Array job script for parallel viscy batch inference
# Each task processes a range of positions using batch inference with async writing

set -e

INPUT_STORE=$1
OUTPUT_DIR=$2
CONFIG=$3
GPUS_PER_TASK=$4
TOTAL_POSITIONS=$5
NUM_ARRAY_TASKS=$6

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Calculate position range for this task
POSITIONS_PER_TASK=$(( (TOTAL_POSITIONS + NUM_ARRAY_TASKS - 1) / NUM_ARRAY_TASKS ))
START_POS=$(( SLURM_ARRAY_TASK_ID * POSITIONS_PER_TASK ))
END_POS=$(( START_POS + POSITIONS_PER_TASK ))

# Don't go past total
if [ $END_POS -gt $TOTAL_POSITIONS ]; then
    END_POS=$TOTAL_POSITIONS
fi

# Skip if no positions for this task
if [ $START_POS -ge $TOTAL_POSITIONS ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID}: No positions to process (start=$START_POS >= total=$TOTAL_POSITIONS)"
    exit 0
fi

echo "================================================================"
echo "Viscy Batch Inference - Array Task"
echo "================================================================"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID} / $((NUM_ARRAY_TASKS - 1))"
echo "Position range: ${START_POS} to ${END_POS} (of ${TOTAL_POSITIONS} total)"
echo "Positions this task: $((END_POS - START_POS))"
echo "GPUs: ${GPUS_PER_TASK}"
echo "Node: $(hostname)"
echo "GPU(s): $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ')"
echo "================================================================"

# Create logs directory
mkdir -p "${OUTPUT_DIR}/logs"

# Run batch inference
"${OPS_PYTHON:-python}" \
    "$(dirname "$(readlink -f "$0")")/../../cyclops_process/processes/viscy_batch_inference.py" \
    --input-store "$INPUT_STORE" \
    --output-dir "$OUTPUT_DIR" \
    --config "$CONFIG" \
    --start-pos "$START_POS" \
    --end-pos "$END_POS" \
    --batch-size 7 \
    --num-workers 8

echo ""
echo "================================================================"
echo "Array task ${SLURM_ARRAY_TASK_ID} completed successfully"
echo "================================================================"
