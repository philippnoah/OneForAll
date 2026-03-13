#!/bin/bash
#SBATCH --job-name=ofa_midterm_pseudo
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=08:00:00

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ofa

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs

PYTHONWARNINGS=ignore python run_cdm.py \
    task_names midterm_retweet_pseudo_node \
    llm_name precomputed \
    emb_dim 256 \
    num_layers 6 \
    dropout 0.15 \
    lr 0.001 \
    batch_size 16 \
    eval_batch_size 32 \
    num_epochs 50 \
    num_workers 8 \
    exp_name midterm_pseudo \
    offline_log False
