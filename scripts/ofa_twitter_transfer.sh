#!/bin/bash
# Transfer evaluation: Instagram-pretrained OFA model → Twitter repdem
#
# Usage:
#   bash scripts/ofa_twitter_transfer.sh <checkpoint_path> [finetune|eval]
#
#   checkpoint_path: Lightning .ckpt saved by ofa_instagram_pretrain
#   mode:
#     finetune  – load checkpoint, fine-tune on Twitter repdem (default)
#     eval      – load checkpoint, zero-shot eval only (no training)
#
# Example:
#   bash scripts/ofa_twitter_transfer.sh \
#       lightning_logs/version_0/checkpoints/epoch=49-step=1000.ckpt \
#       eval

CKPT="${1:?Usage: $0 <checkpoint_path> [finetune|eval]}"
MODE="${2:-finetune}"

if [ "$MODE" = "eval" ]; then
    EPOCHS=1
    EVAL_ONLY="True"
    PREFIX="ofa_twitter_repdem_zeroshot"
else
    EPOCHS=30
    EVAL_ONLY="False"
    PREFIX="ofa_twitter_repdem_finetune"
fi

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${PREFIX}
#SBATCH --output=logs/${PREFIX}_%j.out
#SBATCH --error=logs/${PREFIX}_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ofa
export LD_PRELOAD=\$CONDA_PREFIX/lib/libstdc++.so.6

cd /home1/eibl/gfm/OneForAll
mkdir -p logs

PYTHONWARNINGS=ignore python -u run_cdm.py \
    task_names twitter_retweet_repdem_node \
    llm_name precomputed \
    load_checkpoint ${CKPT} \
    eval_only ${EVAL_ONLY} \
    emb_dim 256 \
    num_layers 6 \
    dropout 0.15 \
    lr 0.0005 \
    batch_size 16 \
    eval_batch_size 32 \
    num_epochs ${EPOCHS} \
    num_workers 8 \
    exp_name ${PREFIX} \
    offline_log False
EOF
