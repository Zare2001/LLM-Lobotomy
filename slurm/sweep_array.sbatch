#!/bin/bash
# ============================================================
# PARALLEL sweep using Slurm array jobs
#
# Splits 1486 configs across 32 independent GPU jobs.
# Each job loads the model ONCE and processes its batch.
#
# With 32 tasks on H100s:
#   1486 / 32 ≈ 47 configs per task × ~2 min each ≈ 1.5 hours
#
# Usage:
#   sbatch slurm/sweep_array.sbatch
#
# After all jobs finish:
#   python slurm/merge_results.py --input-dir results/scores --output results --n-layers 54
# ============================================================

#SBATCH --job-name=lobotomy-sweep
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --time=05:00:00
#SBATCH --array=0-31
#SBATCH --output=slurm/logs/sweep_%A_%a.out
#SBATCH --error=slurm/logs/sweep_%A_%a.err

# >>> SET YOUR BUDGET ACCOUNT <<<
##SBATCH --account=your_project_account

module purge
module load 2025
module load NCCL/2.26.6-GCCcore-14.2.0-CUDA-12.8.0

PROJECT_DIR="$HOME/lobotomy"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

mkdir -p slurm/logs results/scores

TASK_ID=$SLURM_ARRAY_TASK_ID
N_TASKS=$SLURM_ARRAY_TASK_COUNT

echo "=== Array task $TASK_ID / $N_TASKS on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python slurm/hpc_runner.py \
    --model utter-project/EuroLLM-22B-Instruct-2512 \
    --output "$PROJECT_DIR/results/scores/sweep_task_${TASK_ID}.csv" \
    --data-dir "$PROJECT_DIR/data" \
    --task-id "$TASK_ID" \
    --n-tasks "$N_TASKS" \
    --multilingual

echo "Task $TASK_ID finished at $(date)"
