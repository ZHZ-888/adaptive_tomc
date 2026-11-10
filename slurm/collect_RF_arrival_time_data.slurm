#!/bin/bash
#SBATCH --job-name=rf_at_sampling         # Job name shown in the queue
#SBATCH --time=00:20:00                   # Walltime limit (hh:mm:ss)
#SBATCH --cpus-per-task=1                # Number of CPU cores per task
#SBATCH --mem=2G                         # Memory per task
#SBATCH --array=0-89
#SBATCH --output=output-logs/rf_at_%A_%a.out
#SBATCH --error=error-logs/rf_at_%A_%a.err

# ==============================================================================
# USAGE EXAMPLES:
# 1. Default submission (3 AV penetrations * 30 seeds = 90 jobs):
#    sbatch slurm/collect_RF_arrival_time_data.slurm
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
SEEDS=($(seq 21 50))
M_FR=1500
R_FR=0
ST=9000

NAV=${#AV_P_VALS[@]}
NS=${#SEEDS[@]}
NT=$((NAV * NS))

# Check SLURM array task ID
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "Error: SLURM_ARRAY_TASK_ID is not set. Submit with --array=0-$((NT-1))"
  exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID}

# Map array task ID to (av_p, seed)
av_idx=$(( TASK_ID / NS ))
s_idx=$(( TASK_ID % NS ))

AV_P=${AV_P_VALS[$av_idx]}
SEED=${SEEDS[$s_idx]}

echo "TASK_ID=$TASK_ID | M_FR=$M_FR | R_FR=$R_FR | AV_P=$AV_P | Seed=$SEED | ST=$ST"

# ================================
# Per-task CSV output (safe for parallel jobs)
# Each task writes to its own CSV file
# ================================
OUT_CSV="data/raw/tmp/rf_data_mfr${M_FR}_rfr${R_FR}_avp${AV_P}_seed${SEED}.csv"
mkdir -p data/raw/tmp

# ================================
# Run action-aware arrival-time data collection
# ================================
python run.py \
  --algo collect_RF_at_data \
  --av_p "$AV_P" \
  --seed "$SEED" \
  --m_fr "$M_FR" \
  --r_fr "$R_FR" \
  --st "$ST" \
  --jam-mode-core sampling \
  --out_csv "$OUT_CSV"
