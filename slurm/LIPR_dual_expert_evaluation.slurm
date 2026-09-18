#!/bin/bash
#SBATCH --job-name=lipr_pair_eval
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --array=0-89
#SBATCH --output=train-logs/lipr_eval_%A_%a.out
#SBATCH --error=train-logs/lipr_eval_%A_%a.err

# NOTE: sbatch \
#  --array=0-4 \
#  --export=ALL,TRAIN_ROOT=/nesi/nobackup/uoa04480/RL_Training_<训练JOB_ID> \
#  slurm/LIPR_dual_expert_evaluation.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

: "${TRAIN_ROOT:?Set TRAIN_ROOT to RL_Training_<job_id>}"
RESULT_ROOT="${RESULT_ROOT:-/nesi/nobackup/uoa04480/LIPR_Eval_${SLURM_ARRAY_JOB_ID}}"
mkdir -p "$RESULT_ROOT/results" "$RESULT_ROOT/logs" train-logs

module load Python/3.11.3-gimkl-2022a
source /nesi/project/uoa04480/myenv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

LR_VALS=(0.0001 0.0005 0.001)
I_VALS=(16 32 64)
SEEDS=(10 11 12 13 14 15 16 17 18 19)
N_SEEDS=${#SEEDS[@]}

PAIR_IDX=$((SLURM_ARRAY_TASK_ID / N_SEEDS))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % N_SEEDS))
INTERVAL_IDX=$((PAIR_IDX / 3))
LR_IDX=$((PAIR_IDX % 3))

LR=${LR_VALS[$LR_IDX]}
INTERVAL=${I_VALS[$INTERVAL_IDX]}
SEED=${SEEDS[$SEED_IDX]}
CE_TASK=$PAIR_IDX
SE_TASK=$((PAIR_IDX + 9))

# Prefer final models.
CE_MODEL=$(find "$TRAIN_ROOT" -type f \
  -path "*_task${CE_TASK}/models/final_CA.pt" -print -quit)

SE_MODEL=$(find "$TRAIN_ROOT" -type f \
  -path "*_task${SE_TASK}/models/final_SA.pt" -print -quit)

# If final_CA.pt is unavailable, select the checkpoint with the
# lexicographically largest timestamp in its filename.
if [[ -z "$CE_MODEL" ]]; then
  CE_MODEL=$(find "$TRAIN_ROOT" -type f \
    -path "*_task${CE_TASK}/models/free_insert_score_model_*.pt" \
    | sort | tail -n 1)
fi

# If final_SA.pt is unavailable, select the largest SE checkpoint.
if [[ -z "$SE_MODEL" ]]; then
  SE_MODEL=$(find "$TRAIN_ROOT" -type f \
    -path "*_task${SE_TASK}/models/sa_*.pt" \
    | sort | tail -n 1)
fi

if [[ -z "$CE_MODEL" || -z "$SE_MODEL" ]]; then
  echo "Missing model: CE task $CE_TASK => $CE_MODEL; SE task $SE_TASK => $SE_MODEL" >&2
  exit 1
fi

echo "Selected CE model: $CE_MODEL"
echo "Selected SE model: $SE_MODEL"

export RUN_DIR="$RESULT_ROOT/logs/pair_${PAIR_IDX}_seed_${SEED}"
mkdir -p "$RUN_DIR"
OUT_CSV="$RESULT_ROOT/results/pair_${PAIR_IDX}_LR${LR}_I${INTERVAL}_seed${SEED}.csv"

python -u run.py \
  --algo lipr_dual_expert_eval \
  --ce_model_path "$CE_MODEL" \
  --se_model_path "$SE_MODEL" \
  --pair_id "$PAIR_IDX" \
  --lr "$LR" \
  --train_interval "$INTERVAL" \
  --seed "$SEED" \
  --av_p 0.1 \
  --m_fr 1000 \
  --st 1500 \
  --hidden_layer 64 64 \
  --out_csv "$OUT_CSV"
