#!/bin/bash
#SBATCH --job-name=pf_mc_impact_exp
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=output-logs/pf_mc_impact_exp_%A_%a.out
#SBATCH --error=error-logs/pf_mc_impact_exp_%A_%a.err

# Submit with:
# sbatch --array=0-130 slurm/pf_mc_impact_exp.slurm

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# ================================
# Python environment
# ================================
module load Python/3.11.3-gimkl-2022a
source /nesi/project/uoa04480/myenv/bin/activate

# Deterministic behaviour
export PYTHONHASHSEED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ================================
# Log directories
# ================================
mkdir -p output-logs
mkdir -p error-logs

# ================================
# Trajectory output directory
# ================================
export TRAJ_DIR="/nesi/nobackup/uoa04480/traj/pf_mc_impact_exp/${SLURM_ARRAY_JOB_ID}"
mkdir -p "$TRAJ_DIR"

echo "TRAJ_DIR=$TRAJ_DIR"

# ================================
# Fixed experiment settings
# ================================
M_FR=1500
AV_P=0.1
ST=1500
MAX_TEAM_SIZE=12
TSG_MODE=predict

# ================================
# Parameter sweep
# ================================
RAMP_DEMANDS=(200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400)
SEEDS=(0 1 2 3 4 5 6 7 8 9)
FC_MODES=(dla_only)
R_PLATOON_PS=(0.7)

NG=${#FC_MODES[@]}
NR=${#RAMP_DEMANDS[@]}
NS=${#SEEDS[@]}
NT=$((NG * NR * NS))

# ================================
# Check task id
# ================================
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set."
    echo "Submit with: sbatch --array=0-$((NT-1)) slurm/pf_mc_impact_exp.slurm"
    exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID}

# ================================
# Map task id -> (group, ramp demand, seed)
# ================================
group_idx=$(( TASK_ID / (NR * NS) ))
remaining=$(( TASK_ID % (NR * NS) ))
r_idx=$(( remaining / NS ))
s_idx=$(( remaining % NS ))

FC_MODE=${FC_MODES[$group_idx]}
R_PLATOON_P=${R_PLATOON_PS[$group_idx]}
R_FR=${RAMP_DEMANDS[$r_idx]}
SEED=${SEEDS[$s_idx]}

echo "=================================="
echo "TASK_ID : $TASK_ID"
echo "FC_MODE : $FC_MODE"
echo "R_PLATOON_P: $R_PLATOON_P"
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
mkdir -p data/raw/pf_mc_impact_exp

OUT_CSV="data/raw/pf_mc_impact_exp/pf_mc_${FC_MODE}_rp${R_PLATOON_P}_av${AV_P}_m${M_FR}_r${R_FR}_seed${SEED}.csv"

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
    --r_platoon_p "$R_PLATOON_P" \
    --lc \
    --out_csv "$OUT_CSV"

echo "Finished TASK_ID=$TASK_ID"