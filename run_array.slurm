#!/bin/bash
#SBATCH --job-name=rm_array
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-99
#SBATCH --output=output-logs/out_%A_%a.txt
#SBATCH --error=error-logs/err_%A_%a.txt

# Check if MODULE variable is provided
if [ -z "$MODULE" ]; then
    echo "Error: MODULE variable not set. Use: sbatch --export=MODULE=<module_name> run_array.slurm"
    exit 1
fi

module load Python/3.11.3-gimkl-2022a
source /nesi/project/uoa04480/myenv/bin/activate

echo "Running module: $MODULE, task ID: ${SLURM_ARRAY_TASK_ID}"
python run_one.py $MODULE ${SLURM_ARRAY_TASK_ID}
