#!/bin/bash
#SBATCH --job-name=pf_ablation
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=output-logs/pf_ablation_%A_%a.out
#SBATCH --error=error-logs/pf_ablation_%A_%a.err

# Submit with:
# sbatch --array=0-149 slurm/pf_ablation.slurm

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# ================================
# Python environment
# ================================
module load Python/3.11.3-gimkl-2022a
source /nesi/project/uoa04480/myenv/bin/activate

# Deterministic behaviour
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONHASHSEED=1

# ================================
# Log directories
# ================================
mkdir -p output-logs
mkdir -p error-logs

# ================================
# Trajectory output directory
# ================================
export TRAJ_DIR="/nesi/nobackup/uoa04480/traj/pf_ablation/${SLURM_ARRAY_JOB_ID}"
mkdir -p "$TRAJ_DIR"

echo "TRAJ_DIR=$TRAJ_DIR"

# ================================
# Fixed experiment settings
# ================================
R_FR=0
M_FR=1000
ST=1500
MAX_TEAM_SIZE=12
TSG_MODE=predict

# ================================
# Parameter sweep
# ================================
FC_MODES=(dla_only dla_tsc dla_tsc_lhr dla_tsc_lhr_ce full)
AV_P_VALS=(0.1 0.2 0.3)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

NM=${#FC_MODES[@]}
NAV=${#AV_P_VALS[@]}
NS=${#SEEDS[@]}
NT=$((NM * NAV * NS))

# ================================
# Check task id
# ================================
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set."
    echo "Submit with: sbatch --array=0-$((NT-1)) slurm/pf_ablation.slurm"
    exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID}

# ================================
# Map task id -> (fc_mode, av_p, seed)
# ================================
mode_idx=$(( TASK_ID / (NAV * NS) ))
rem=$(( TASK_ID % (NAV * NS) ))
av_idx=$(( rem / NS ))
s_idx=$(( rem % NS ))

FC_MODE=${FC_MODES[$mode_idx]}
AV_P=${AV_P_VALS[$av_idx]}
SEED=${SEEDS[$s_idx]}

echo "=================================="
echo "TASK_ID : $TASK_ID"
echo "FC_MODE : $FC_MODE"
echo "TSG_MODE: $TSG_MODE"
echo "AV_P    : $AV_P"
echo "SEED    : $SEED"
echo "R_FR    : $R_FR"
echo "M_FR    : $M_FR"
echo "ST      : $ST"
echo "MAX_SIZE: $MAX_TEAM_SIZE"
echo "=================================="

# ================================
# Output CSV
# ================================
mkdir -p data/raw/pf_ablation

OUT_CSV="data/raw/pf_ablation/pf_ablation_${FC_MODE}_av${AV_P}_seed${SEED}.csv"
# ================================
# Run simulation
# ================================
python run.py \
    --algo mpgc_multi_lane \
    --av_p "$AV_P" \
    --r_fr "$R_FR" \
    --m_fr "$M_FR" \
    --seed "$SEED" \
    --st "$ST" \
    --max_team_size "$MAX_TEAM_SIZE" \
    --tsg_mode "$TSG_MODE" \
    --fc_mode "$FC_MODE" \
    --no-lc \
    --out_csv "$OUT_CSV"

echo "Finished TASK_ID=$TASK_ID"
