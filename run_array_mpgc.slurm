#!/bin/bash
#SBATCH --job-name=mpgc_array         # Job name shown in the queue
#SBATCH --time=04:00:00               # Walltime limit (hh:mm:ss)
#SBATCH --cpus-per-task=1             # Number of CPU cores per task
#SBATCH --mem=4G                      # Memory per task
# Do not fix --array here; set it when submitting with sbatch
#SBATCH --output=output-logs/out_%A_%a.txt   # Standard output log
#SBATCH --error=error-logs/err_%A_%a.txt    # Standard error log

# ===== Fixed algorithm module (MPGC) =====
MODULE="mpgc_250817"   # Change to the correct Python file name if needed

# ===== Required argument: EXP (1|2|3|4) =====
if [ -z "$EXP" ]; then
  echo "Error: EXP not set. Use: sbatch --export=EXP=<1|2|3|4> --array=... run_array_mpgc.slurm"
  exit 1
fi

# Informational only: expected number of tasks for each experiment
case "$EXP" in
  1) N=300 ;;   # Experiment 1: ideal
  2) N=200 ;;   # Experiment 2: acc_hv
  3) N=300 ;;   # Experiment 3: platoon_hv
  4) N=300 ;;   # Experiment 4: v2x_disturbance
  5) N=520 ;;   # Experiment 5: find best mpc_interval
  *) echo "Unknown EXP=$EXP (use 1|2|3|4)"; exit 1 ;;
esac

# ===== Load Python environment =====
module load Python/3.11.3-gimkl-2022a
source /nesi/project/uoa04480/myenv/bin/activate

# Ensure log directories exist
mkdir -p output-logs error-logs

# Set trajectory output dir (env var read by mpgc_250817.py)
export TRAJ_DIR="/nesi/nobackup/uoa04480/traj"
mkdir -p "$TRAJ_DIR"
echo "TRAJ_DIR=$TRAJ_DIR"

# Check that SLURM_ARRAY_TASK_ID is set
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  echo "Error: SLURM_ARRAY_TASK_ID is empty. Submit with --array=0-$((N-1)) (optionally add %concurrency)."
  exit 1
fi

echo "Running: MODULE=${MODULE} | EXP=${EXP} | TASK_ID=${SLURM_ARRAY_TASK_ID} (valid 0..$((N-1)))"

# ===== Run the Python script =====
python run_one_mpgc.py "${MODULE}" "${EXP}" "${SLURM_ARRAY_TASK_ID}"
