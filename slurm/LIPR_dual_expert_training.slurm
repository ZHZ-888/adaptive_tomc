#!/bin/bash
#SBATCH --job-name=mpgc_training            # Job name for tracking
#SBATCH --time=7:00:00                     # Increased time limit for longer simulation
#SBATCH --cpus-per-task=1                   # CPU cores per task for Torch/SUMO
#SBATCH --mem=4G                            # Memory allocation for Replay Buffer
#SBATCH --output=/nesi/nobackup/uoa04480/train-logs/out_%A_%a.txt # Standard output log
#SBATCH --error=/nesi/nobackup/uoa04480/train-logs/err_%A_%a.txt # Error log
#SBATCH --array=0-17
# NOTE: 18 tasks = 2 agents * 3 learning rates * 3 update intervals

set -euo pipefail

# Ensure the script runs from the project root directory
cd "$SLURM_SUBMIT_DIR"

# ================================
# HPC Storage Configuration
# ================================
export RUN_DIR="/nesi/nobackup/uoa04480/RL_Training_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$RUN_DIR"
mkdir -p train-logs
echo "All heavy logs and models will be saved to: $RUN_DIR"

# ================================
# Load Environment
# ================================
module load Python/3.11.3-gimkl-2022a
source /nesi/project/uoa04480/myenv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ================================
# Parameter Mapping (Grid Search)
# ================================
LR_VALS=(0.0001 0.0005 0.001)               # Candidate learning rates
I_VALS=(16 32 64)                           # Candidate update intervals
AGENT_LIST=("CA" "SA")                      # Fixed array name for indexing

TASK_ID=${SLURM_ARRAY_TASK_ID}

# 1. Determine agent type (tasks 0-8: CA; tasks 9-17: SA)
agent_idx=$(( TASK_ID / 9 ))
AGENT_TYPE=${AGENT_LIST[$agent_idx]}

# 2. Fixed traffic settings for both experts
AV_P=0.1
R_FR=0
M_FR=1000

# 3. Determine hyperparameters (three LRs within each interval)
rem=$(( TASK_ID % 9 ))
ui_idx=$(( rem / 3 ))
lr_idx=$(( rem % 3 ))

# FIX: Assign values from the arrays
UPDATE_INTERVAL=${I_VALS[$ui_idx]}
LR=${LR_VALS[$lr_idx]}

echo "------------------------------------------------"
echo "JOB START: Array ID $TASK_ID"
echo "CONFIG: Agent=$AGENT_TYPE | AV_P=$AV_P | R_FR=$R_FR | M_FR=$M_FR | LR=$LR | Interval=$UPDATE_INTERVAL"
echo "------------------------------------------------"

# ================================
# Execution
# ================================
stdbuf -oL python -u run.py \
  --algo lipr_dual_expert_train \
  --train_agent "$AGENT_TYPE" \
  --lr "$LR" \
  --seed 21 \
  --train_interval "$UPDATE_INTERVAL" \
  --hidden_layer 64 64 \
  --st 6000000 \
  --av_p "$AV_P" \
  --r_fr "$R_FR" \
  --m_fr "$M_FR"
