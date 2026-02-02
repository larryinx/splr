#!/bin/bash
#SBATCH --job-name=splr_pretrain_think
#SBATCH --account=<account-name>
#SBATCH --time=11:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=800G
#SBATCH --gres=gpu:h100:4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --array=0-9

# ── Array task mapping ──────────────────────────────────────────
#  0  recurrent      splr_trm_qwen
#  1  recurrent      splr_trm_qwen_zl
#  2  recurrent      splr_trm_gpt
#  3  recurrent      splr_trm_gpt_zl
#  4  autoregressive splr_trm_qwen
#  5  autoregressive splr_trm_qwen_zl
#  6  autoregressive splr_trm_qwen_dis
#  7  autoregressive splr_trm_gpt
#  8  autoregressive splr_trm_gpt_zl
#  9  autoregressive splr_trm_gpt_dis
# ────────────────────────────────────────────────────────────────

MAX_REASONING_STEPS=2

ARCH_CONFIGS=(
    splr_trm_qwen
    splr_trm_qwen_zl
    splr_trm_gpt
    splr_trm_gpt_zl
    splr_trm_qwen
    splr_trm_qwen_zl
    splr_trm_qwen_dis
    splr_trm_gpt
    splr_trm_gpt_zl
    splr_trm_gpt_dis
)

INPUT_MODES=(
    recurrent
    recurrent
    recurrent
    recurrent
    autoregressive
    autoregressive
    autoregressive
    autoregressive
    autoregressive
    autoregressive
)

TOKENIZER_PATHS=(
# TODO: add tokenizer paths
)

ARCH=${ARCH_CONFIGS[$SLURM_ARRAY_TASK_ID]}
MODE=${INPUT_MODES[$SLURM_ARRAY_TASK_ID]}
TOKENIZER_PATH=${TOKENIZER_PATHS[$SLURM_ARRAY_TASK_ID]}

OUTPUT_DIR=./results/experiments/pretrain_think/${ARCH}_${MODE}_${MAX_REASONING_STEPS}

echo "=== Array task $SLURM_ARRAY_TASK_ID: arch=$ARCH  input_mode=$MODE ==="



export TOKENIZERS_PARALLELISM=false

NUM_GPUS=4
MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID))

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    main.py \
    --config-name cfg_pretrain \
    arch=$ARCH \
    input_mode=$MODE \
    tokenizer_path=$TOKENIZER_PATH \
    output_dir=$OUTPUT_DIR \
    run_name=${ARCH}_${MODE}_${MAX_REASONING_STEPS}

echo "Training completed!"
