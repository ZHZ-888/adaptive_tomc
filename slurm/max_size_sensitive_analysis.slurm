#!/bin/bash
#SBATCH --job-name=mpgc_maxsize_sa
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=output-logs/maxsize_out_%A_%a.txt
#SBATCH --error=error-logs/maxsize_err_%A_%a.txt

# Use:
#   sbatch --array=0-79 slurm/max_size_sensitive_analysis.slurm

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# ================================
# Environment
# ================================
module load Python/3.11.3-gimkl-2022a
source /nesi/project/uoa04480/myenv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p output-logs error-logs data/raw/tmp

# ================================
# Trajectory output directory
# ================================
export TRAJ_DIR="/nesi/nobackup/uoa04480/traj"
mkdir -p "$TRAJ_DIR"
echo "TRAJ_DIR=$TRAJ_DIR"

# ================================
# Fixed traffic settings for max-size sensitivity analysis
# ================================
AV_P=0.1
R_FR=800
M_FR=1500
TSG_MODE=predict

# ================================
# Sweep dimensions
# ================================
MAX_TEAM_SIZES=(7 8 9 10 11 12 13 14)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

NMAX=${#MAX_TEAM_SIZES[@]}
NS=${#SEEDS[@]}
NT=$((NMAX * NS))

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "Error: SLURM_ARRAY_TASK_ID is not set. Submit with --array=0-$((NT-1))"
  exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID}
max_idx=$(( TASK_ID / NS ))
seed_idx=$(( TASK_ID % NS ))

MAX_TEAM_SIZE=${MAX_TEAM_SIZES[$max_idx]}
SEED=${SEEDS[$seed_idx]}

echo "TASK_ID=$TASK_ID | MAX_TEAM_SIZE=$MAX_TEAM_SIZE | SEED=$SEED | AV_P=$AV_P | R_FR=$R_FR | M_FR=$M_FR"

# ================================
# Per-task CSV output
# ================================
OUT_CSV="data/raw/tmp/maxsize_sa_mts${MAX_TEAM_SIZE}_seed${SEED}_task${TASK_ID}.csv"

# ================================
# Run simulation
# ================================
python run.py \
  --algo mpgc_multi_lane \
  --av_p "$AV_P" \
  --r_fr "$R_FR" \
  --m_fr "$M_FR" \
  --seed "$SEED" \
  --tsg_mode "$TSG_MODE" \
  --max_team_size "$MAX_TEAM_SIZE" \
  --out_csv "$OUT_CSV"

echo "Finished TASK_ID=$TASK_ID"
