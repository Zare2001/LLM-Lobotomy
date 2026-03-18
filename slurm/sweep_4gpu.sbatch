#!/bin/bash
# ============================================================
# 4-GPU single-node sweep
#
# Runs 4 independent sweep batches on 4 GPUs within one node.
# 1486 configs / 4 GPUs × ~2 min each ≈ 12 hours
#
# Usage:
#   sbatch slurm/sweep_4gpu.sbatch
# ============================================================

#SBATCH --job-name=lobotomy-4gpu
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/sweep_4gpu_%j.out
#SBATCH --error=slurm/logs/sweep_4gpu_%j.err

# >>> SET YOUR BUDGET ACCOUNT <<<
##SBATCH --account=your_project_account

module purge
module load 2025
module load NCCL/2.26.6-GCCcore-14.2.0-CUDA-12.8.0

PROJECT_DIR="$HOME/lobotomy"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

mkdir -p slurm/logs results/scores

echo "=== 4-GPU sweep on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

for GPU_ID in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python slurm/hpc_runner.py \
        --model utter-project/EuroLLM-22B-Instruct-2512 \
        --output "$PROJECT_DIR/results/scores/sweep_gpu_${GPU_ID}.csv" \
        --data-dir "$PROJECT_DIR/data" \
        --task-id "$GPU_ID" \
        --n-tasks 4 \
        --multilingual &
done

wait
echo "All 4 GPUs finished at $(date)"

python slurm/merge_results.py \
    --input-dir "$PROJECT_DIR/results/scores" \
    --output "$PROJECT_DIR/results" \
    --n-layers 54

echo "Done."
