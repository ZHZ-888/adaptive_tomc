#!/bin/bash
#SBATCH --job-name=dispatch_eval
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --array=0-89
#SBATCH --output=/nesi/nobackup/uoa04480/train-logs/dispatch_eval_%A_%a.out
#SBATCH --error=/nesi/nobackup/uoa04480/train-logs/dispatch_eval_%A_%a.err

# Example:
# sbatch --export=ALL,TRAIN_ROOT=/nesi/nobackup/uoa04480/TSG_Search_<job_id> \
#   slurm/LIPR_dispatch_evaluation.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

: "${TRAIN_ROOT:?Set TRAIN_ROOT to TSG_Search_<training_job_id>}"
RESULT_ROOT="${RESULT_ROOT:-/nesi/nobackup/uoa04480/LIPR_Dispatch_Eval_${SLURM_ARRAY_JOB_ID}}"
mkdir -p "$RESULT_ROOT/results" "$RESULT_ROOT/logs"

module load Python/3.11.3-gimkl-2022a
source /nesi/project/uoa04480/myenv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

LR_VALS=(0.0001 0.0005 0.001)
INTERVAL_VALS=(16 32 64)
AV_P_VALS=(0.1 0.1 0.1 0.1 0.1 0.2 0.2 0.2 0.3 0.3)
SEED_VALS=(31 32 33 34 35 36 37 38 39 40)
N_SCENARIOS=${#SEED_VALS[@]}

MODEL_ID=$((SLURM_ARRAY_TASK_ID / N_SCENARIOS))
SCENARIO_ID=$((SLURM_ARRAY_TASK_ID % N_SCENARIOS))
LR_IDX=$((MODEL_ID / 3))
INTERVAL_IDX=$((MODEL_ID % 3))

LR=${LR_VALS[$LR_IDX]}
TRAIN_INTERVAL=${INTERVAL_VALS[$INTERVAL_IDX]}
AV_P=${AV_P_VALS[$SCENARIO_ID]}
SEED=${SEED_VALS[$SCENARIO_ID]}
MODEL_DIR="task_${MODEL_ID}_LR${LR}_I${TRAIN_INTERVAL}"
DISPATCH_MODEL="$TRAIN_ROOT/$MODEL_DIR/task_self_gate_v2_latest.pt"

if [[ ! -f "$DISPATCH_MODEL" ]]; then
  echo "Dispatch model not found: $DISPATCH_MODEL" >&2
  exit 1
fi

export RUN_DIR="$RESULT_ROOT/logs/model_${MODEL_ID}_scenario_${SCENARIO_ID}_seed_${SEED}"
mkdir -p "$RUN_DIR"
OUT_CSV="$RESULT_ROOT/results/model_${MODEL_ID}_LR${LR}_I${TRAIN_INTERVAL}_avp${AV_P}_seed${SEED}.csv"

echo "Model ID: $MODEL_ID"
echo "Learning rate: $LR"
echo "Train interval: $TRAIN_INTERVAL"
echo "AV penetration: $AV_P"
echo "Validation seed: $SEED"
echo "Dispatch model: $DISPATCH_MODEL"
echo "Result: $OUT_CSV"

python -u run.py \
  --algo lipr_dispatch_eval \
  --dispatch_model_path "$DISPATCH_MODEL" \
  --model_id "$MODEL_ID" \
  --lr "$LR" \
  --train_interval "$TRAIN_INTERVAL" \
  --seed "$SEED" \
  --av_p "$AV_P" \
  --m_fr 1500 \
  --st 1500 \
  --max_team_size 12 \
  --out_csv "$OUT_CSV"
