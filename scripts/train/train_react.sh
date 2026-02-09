#!/bin/bash
#SBATCH --job-name=splr_pretrain_react
#SBATCH --account=<account-name>
#SBATCH --time=11:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=400G
#SBATCH --gres=gpu:h100:2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --array=0-1

# ── Array task mapping ──────────────────────────────────────────
#  0  recurrent      splr_trm_gpt
#  1  recurrent      splr_trm_gpt
# ────────────────────────────────────────────────────────────────

MAX_REASONING_STEPS=<max_reasoning_steps>

ARCH_CONFIGS=(
    splr_trm_gpt
    splr_trm_gpt
)

INPUT_MODES=(
    recurrent
    recurrent
)

TOKENIZER_PATHS=(
# TODO: add tokenizer paths
)

ARCH=${ARCH_CONFIGS[$SLURM_ARRAY_TASK_ID]}
MODE=${INPUT_MODES[$SLURM_ARRAY_TASK_ID]}
TOKENIZER_PATH=${TOKENIZER_PATHS[$SLURM_ARRAY_TASK_ID]}

OUTPUT_DIR=./results/experiments/pretrain_react/${ARCH}_${MODE}_${MAX_REASONING_STEPS}

echo "=== Array task $SLURM_ARRAY_TASK_ID: arch=$ARCH  input_mode=$MODE ==="



export TOKENIZERS_PARALLELISM=false

NUM_GPUS=2
MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID))

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    main.py \
    --config-name cfg_pretrain_tool \
    arch=$ARCH \
    input_mode=$MODE \
    tokenizer_path=$TOKENIZER_PATH \
    output_dir=$OUTPUT_DIR \
    run_name=${ARCH}_${MODE}_${MAX_REASONING_STEPS}

echo "Training completed!"
