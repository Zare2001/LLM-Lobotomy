#!/bin/bash
# ============================================================
# Snellius Environment Setup for LLM Lobotomy
# Run this ONCE on a login node (or interactive session).
#
# Assumes uv is already installed and you have a .venv at
# $HOME/lobotomy/.venv created via: uv venv --python=3.12
# ============================================================

PROJECT_DIR="$HOME/lobotomy"
MODEL_DIR="$PROJECT_DIR/models/EuroLLM-22B-Instruct-2512"
MODEL_NAME="utter-project/EuroLLM-22B-Instruct-2512"

echo "=== LLM Lobotomy — Snellius Setup ==="
echo "Project: $PROJECT_DIR"
echo ""

# --- 1. Load modules ---
module purge
module load 2025
module load NCCL/2.26.6-GCCcore-14.2.0-CUDA-12.8.0

# --- 2. Activate venv ---
source "$PROJECT_DIR/.venv/bin/activate"

# --- 3. Install dependencies with uv ---
echo "Installing Python dependencies..."
uv pip install -r "$PROJECT_DIR/requirements.txt"

# --- 4. Download model ---
mkdir -p "$MODEL_DIR"
echo "Downloading model (~45 GB)..."
uv pip install huggingface-hub
huggingface-cli download "$MODEL_NAME" \
    --local-dir "$MODEL_DIR" \
    --local-dir-use-symlinks False

# --- 5. Create output directories ---
mkdir -p "$PROJECT_DIR/results/scores" "$PROJECT_DIR/results/heatmaps" "$PROJECT_DIR/slurm/logs"

echo ""
echo "=== Setup complete ==="
echo "Model:   $MODEL_DIR"
echo "Results: $PROJECT_DIR/results/"
echo ""
echo "Next: edit slurm/*.sbatch to set --account, then sbatch"
