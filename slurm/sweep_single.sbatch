#!/bin/bash
# ============================================================
# Single-GPU full sweep on Snellius
# Runs ALL 1486 configs sequentially on one GPU.
# Estimated: ~50 hours (within the 120h max)
#
# Usage:
#   sbatch slurm/sweep_single.sbatch
# ============================================================

#SBATCH --job-name=lobotomy-sweep
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --time=72:00:00
#SBATCH --output=slurm/logs/sweep_%j.out
#SBATCH --error=slurm/logs/sweep_%j.err

module purge
module load 2025
module load NCCL/2.26.6-GCCcore-14.2.0-CUDA-12.8.0
export HF_HOME="/scratch-shared/$USER/huggingface"
PROJECT_DIR="$HOME/lobotomy"
MODEL="utter-project/EuroLLM-22B-Instruct-2512"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

mkdir -p slurm/logs

echo "Starting full sweep on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python run_lobotomy.py sweep \
    --model "$MODEL" \
    --output "$PROJECT_DIR/results" \
    --data-dir "$PROJECT_DIR/data" \
    --4bit \
    --multilingual

echo "Sweep finished at $(date)"

echo "Generating heatmaps..."
python run_lobotomy.py heatmap \
    --input "$PROJECT_DIR/results/sweep.csv" \
    --output "$PROJECT_DIR/results/heatmaps" \
    --n-layers 54

echo "Done."
