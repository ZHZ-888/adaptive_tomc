#!/bin/bash
#SBATCH --job-name=rf_state_data_collection
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --array=0-120
#SBATCH --output=output-logs/out_%A_%a.txt
#SBATCH --error=error-logs/err_%A_%a.txt
# 120 jobs: 3 AV penetration rates × 40 random seeds

set -euo pipefail

# Ensure relative paths are resolved from the submission directory
cd "$SLURM_SUBMIT_DIR"

# ================================
# Load Python environment
# ================================
module load Python/3.11.3-gimkl-2022a
source /nesi/project/uoa04480/myenv/bin/activate

# Ensure deterministic CPU behaviour
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONHASHSEED=1

# Create output directories
mkdir -p output-logs error-logs
mkdir -p data/features/tmp

# ================================
# Fixed experiment settings
# ================================
M_FR=1500
ST=1500

# ================================
# Experiment parameter grid
# ================================
AV_P_VALS=(0.1 0.2 0.3)
SEEDS=(21 22 23 24 25 26 27 28 29 30 \
       31 32 33 34 35 36 37 38 39 40 \
       41 42 43 44 45 46 47 48 49 50 \
       51 52 53 54 55 56 57 58 59 60)
# SEEDS=(51 52 53 54 55 56 57 58 59 60)

NAV=${#AV_P_VALS[@]}
NS=${#SEEDS[@]}
NT=$((NAV * NS))

# Check SLURM array task ID
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "Error: SLURM_ARRAY_TASK_ID is not set."
  echo "Submit with: sbatch slurm/collect_RF_state_data.slurm"
  exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID}

# Map task ID to (av_p, seed)
av_idx=$(( TASK_ID / NS ))
s_idx=$(( TASK_ID % NS ))

AV_P=${AV_P_VALS[$av_idx]}
SEED=${SEEDS[$s_idx]}

echo "TASK_ID=$TASK_ID | AV_P=$AV_P | SEED=$SEED"

# ================================
# Output CSV for this task
# ================================
OUT_CSV="data/features/tmp/df_RF_state_data_av${AV_P}_seed${SEED}.csv"

# ================================
# Run RF data collection
# ================================
python run.py \
  --algo mpgc_multi_lane_collect_RF_state_data \
  --av_p "$AV_P" \
  --m_fr "$M_FR" \
  --seed "$SEED" \
  --st "$ST" \
  --out_csv "$OUT_CSV"