#!/bin/bash
#SBATCH --job-name=RL_array                # Job name shown in the queue
#SBATCH --time=04:00:00                   # Walltime limit (hh:mm:ss)
#SBATCH --cpus-per-task=1                 # Number of CPU cores per task
#SBATCH --mem=4G                          # Memory per task
#SBATCH --array=0-8                       # 9 array jobs (task IDs: 0..8)
#SBATCH --output=output-logs/out_%A_%a.txt   # Standard output log
#SBATCH --error=error-logs/err_%A_%a.txt     # Standard error log

# ===== Load Python environment =====
module load Python/3.11.3-gimkl-2022a
source /nesi/project/uoa04480/myenv/bin/activate

# Ensure log directories exist
mkdir -p output-logs error-logs

# ===== Parameter sets (total 9 combinations: 3 AV rates × 3 ramp flows) =====
# These lists correspond to task IDs 0..8
av_p_list=(0.1 0.1 0.1   0.2 0.2 0.2   0.3 0.3 0.3)
r_fr_list=(810 720 630   810 720 630   810 720 630)

# Fixed parameters
seed=21
st=6000

# ===== Select parameters based on SLURM array task ID =====
ID=${SLURM_ARRAY_TASK_ID}
av_p=${av_p_list[$ID]}
r_fr=${r_fr_list[$ID]}

echo "Running task $ID with parameters:"
echo "  av_p = $av_p"
echo "  r_fr = $r_fr"
echo "  seed = $seed"
echo "  st   = $st"

# ===== Run the simulation =====
python run_one_RLdata.py \
    --av_p $av_p \
    --r_fr $r_fr \
    --seed $seed \
    --st $st \
    --nogui
