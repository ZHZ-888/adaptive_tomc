#!/bin/bash
#SBATCH --job-name=mpgc_ml_simulation       # Job name shown in the queue
#SBATCH --time=00:20:00                     # Walltime limit (hh:mm:ss)
#SBATCH --cpus-per-task=1                   # Number of CPU cores per task
#SBATCH --mem=4G                            # Memory per task
#SBATCH --output=output-logs/out_%A_%a.txt  # Standard output log
#SBATCH --error=error-logs/err_%A_%a.txt    # Standard error log
# NOTE: Submit with --array=0-9  (1 AV_Ps * 10 Seeds * 1 RAMPS = 10 jobs)

set -euo pipefail

# Ensure relative paths are resolved from the submission directory
cd "$SLURM_SUBMIT_DIR"

# ================================
# Load Python environment
# ================================
module load Python/3.11.3-gimkl-2022a
source /nesi/project/uoa04480/myenv/bin/activate

# Ensure RL deterministic
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

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
MAINLINE=1500                                # Mainline demand (veh/h)
AV_P_VALS=(0.1)                     # AV penetration
SEEDS=(0 1 2 3 4 5 6 7 8 9)                      # Random seeds
RAMPS=(600) # Ramp demands

NAV=${#AV_P_VALS[@]}                        # Number of AV Settings (3)
NS=${#SEEDS[@]}                             # Number of Seeds (10)
NR=${#RAMPS[@]}                             # Number of Ramp Demands (13)
NT=$((NAV * NS * NR))                       # Total Tasks (390)

# Check SLURM array task ID
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "Error: SLURM_ARRAY_TASK_ID is not set. Submit with --array=0-$((NT-1))"
  exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID}

# Map array task ID to (av_p, seed, ramp)
av_idx=$(( TASK_ID / (NS * NR) ))
r_idx=$(( (TASK_ID / NS) % NR ))
s_idx=$(( TASK_ID % NS ))

AV_P=${AV_P_VALS[$av_idx]}
RAMP=${RAMPS[$r_idx]}
SEED=${SEEDS[$s_idx]}

# Fix Python string/set hash order for reproducible runs.
export PYTHONHASHSEED="$SEED"

echo "TASK_ID=$TASK_ID | AV_P=$AV_P | RAMP=$RAMP | MAINLINE=$MAINLINE | SEED=$SEED | (Indices: av=$av_idx, ramp=$r_idx, seed=$s_idx)"

# ================================
# Per-task XML output (safe for parallel jobs)
# Each task writes to its own XML file
# ================================
OUT_CSV="data/raw/tmp/mpgc_multi_lane_task_${TASK_ID}.csv"
mkdir -p data/raw/tmp

# ================================
# Run mpgc_multi_lane simulation
# ================================
python run.py \
  --algo mpgc_multi_lane \
  --av_p "$AV_P" \
  --r_fr "$RAMP" \
  --m_fr "$MAINLINE" \
  --seed "$SEED" \
  --st 1500 \
  --out_csv "$OUT_CSV"
