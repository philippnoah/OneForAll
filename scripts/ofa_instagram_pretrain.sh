#!/bin/bash
#SBATCH --job-name=ofa_ig_pretrain
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ofa
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

cd /home1/eibl/gfm/OneForAll
mkdir -p logs

PYTHONWARNINGS=ignore python -u run_cdm.py \
    task_names instagram_mention_node \
    llm_name precomputed \
    emb_dim 256 \
    num_layers 6 \
    dropout 0.15 \
    lr 0.001 \
    batch_size 16 \
    eval_batch_size 32 \
    num_epochs 50 \
    num_workers 8 \
    exp_name instagram_pretrain \
    offline_log False
