#!/bin/bash
#SBATCH --job-name=mpgc_ml_array          # Job name shown in the queue
#SBATCH --time=00:20:00                  # Walltime limit (hh:mm:ss)
#SBATCH --cpus-per-task=1                # Number of CPU cores per task
#SBATCH --mem=2G                         # Memory per task
#SBATCH --output=output-logs/out_%A_%a.txt   # Standard output log
#SBATCH --error=error-logs/err_%A_%a.txt    # Standard error log
# NOTE: Submit with --array=0-29  (2 demand settings * 3 AV_Ps * 5 seeds = 30 jobs)
# NOTE: Submit with --array=0-60  (2 demand settings * 3 AV_Ps * 10 seeds = 60 jobs)

# ==============================================================================
# USAGE EXAMPLES:
# 1. Default submission (uses values defined in this script):
#    sbatch --array=0-29 slurm/collect_RF_arrival_time_data.slurm
# ==============================================================================

set -euo pipefail

# Ensure relative paths are resolved from the submission directory
cd "$SLURM_SUBMIT_DIR"


# ================================
# Load Python environment
# ================================
module load Python/3.11.3-gimkl-2022a
source /nesi/project/uoa04480/myenv/bin/activate

# Create log directories if not exist
mkdir -p output-logs error-logs

# ================================
# Trajectory (XML) output directory
# Use nobackup due to large file size
# Read by Python via environment variable TRAJ_DIR
# ================================
export TRAJ_DIR="/nesi/nobackup/uoa04480/traj"
mkdir -p "$TRAJ_DIR"
echo "TRAJ_DIR=$TRAJ_DIR"

# ================================
# Experiment parameter grid
# ================================
AV_P_VALS=(0.1 0.2 0.3)
# SEEDS=(21 22 23 24 25)
SEEDS=(11 12 13 14 15 16 17 18 19 20)
M_FR_VALS=(1500 0)
R_FR_VALS=(0 800)

NAV=${#AV_P_VALS[@]}
NS=${#SEEDS[@]}
NF=${#M_FR_VALS[@]}
NT=$((NF * NAV * NS))

# Check SLURM array task ID
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "Error: SLURM_ARRAY_TASK_ID is not set. Submit with --array=0-$((NT-1))"
  exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID}

# Map array task ID to (demand, av_p, seed)
flow_idx=$(( TASK_ID / (NAV * NS) ))
remaining=$(( TASK_ID % (NAV * NS) ))
av_idx=$(( remaining / NS ))
s_idx=$(( remaining % NS ))

M_FR=${M_FR_VALS[$flow_idx]}
R_FR=${R_FR_VALS[$flow_idx]}
AV_P=${AV_P_VALS[$av_idx]}
SEED=${SEEDS[$s_idx]}

echo "TASK_ID=$TASK_ID | M_FR=$M_FR | R_FR=$R_FR | AV_P=$AV_P | Seed=$SEED"

# ================================
# Per-task CSV output (safe for parallel jobs)
# Each task writes to its own CSV file
# ================================
OUT_CSV="data/raw/tmp/rf_data_mfr${M_FR}_rfr${R_FR}_avp${AV_P}_seed${SEED}.csv"
mkdir -p data/raw/tmp

# ================================
# Run mpgc_multi_lane_collect_RF_at_data multi-lane simulation
# ================================
python run.py \
  --algo mpgc_multi_lane_collect_RF_at_data \
  --av_p "$AV_P" \
  --seed "$SEED" \
  --m_fr "$M_FR" \
  --r_fr "$R_FR" \
  --out_csv "$OUT_CSV"