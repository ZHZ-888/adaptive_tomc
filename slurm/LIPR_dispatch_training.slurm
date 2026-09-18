#!/bin/bash
#SBATCH --job-name=tsg_training
#SBATCH --array=0-8
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --output=/nesi/nobackup/uoa04480/train-logs/tsg_%j.out
#SBATCH --error=/nesi/nobackup/uoa04480/train-logs/tsg_%j.err

set -euo pipefail

# Ensure the script runs from the project root directory
cd "$SLURM_SUBMIT_DIR"

# ================================
# Hyperparameter Configuration
# ================================
LRS=(0.0001 0.0005 0.001)
TRAIN_INTERVALS=(16 32 64)

TASK_ID=${SLURM_ARRAY_TASK_ID}
LR_IDX=$((TASK_ID / 3))
INTERVAL_IDX=$((TASK_ID % 3))

LR=${LRS[$LR_IDX]}
TRAIN_INTERVAL=${TRAIN_INTERVALS[$INTERVAL_IDX]}

SEARCH_ROOT="/nesi/nobackup/uoa04480/TSG_Search_${SLURM_ARRAY_JOB_ID}"
export RUN_DIR="${SEARCH_ROOT}/task_${TASK_ID}_LR${LR}_I${TRAIN_INTERVAL}"

mkdir -p "$RUN_DIR"
CONFIG_FILE="$RUN_DIR/training_config.csv"

echo "task_id,learning_rate,train_interval,hidden_dims,simulation_time" \
  > "$CONFIG_FILE"

echo "$TASK_ID,$LR,$TRAIN_INTERVAL,16x16,6000" \
  >> "$CONFIG_FILE"

echo "Array task: $TASK_ID"
echo "Learning rate: $LR"
echo "Train interval: $TRAIN_INTERVAL"
echo "All TSG logs and models will be saved to: $RUN_DIR"

# ================================
# Load Environment
# ================================
module load Python/3.11.3-gimkl-2022a
source /nesi/project/uoa04480/myenv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ================================
# Run shared TSG multi-scenario training
# ================================
stdbuf -oL python -u run.py \
  --algo lipr_dispatch_train \
  --lr "$LR" \
  --train_interval "$TRAIN_INTERVAL" \
  --st 30000