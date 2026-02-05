#!/bin/bash
#SBATCH --job-name=pretrain_gpt2
#SBATCH --account=rrg-pynie
#SBATCH --time=04:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=400G
#SBATCH --gres=gpu:h100:2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

TRAIN_FILE="$PROJECT_DIR/datasets/gsm8k/react/train.txt"
VALID_FILE="$PROJECT_DIR/datasets/gsm8k/react/valid.txt"
TEST_FILE="$PROJECT_DIR/datasets/gsm8k/react/test.txt"

OUTPUT_DIR="$PROJECT_DIR/results/pretrain_gpt2_react"

# ── Combine validation files ─────────────────────────────────────
VAL_COMBINED="$PROJECT_DIR/datasets/gsm8k/react/val_combined.txt"
cat "$VALID_FILE" "$TEST_FILE" > "$VAL_COMBINED"
echo "Combined validation file: $(wc -l < "$VAL_COMBINED") lines"

# ── Training config ──────────────────────────────────────────────
NUM_GPUS=2
MASTER_PORT=${MASTER_PORT:-29500}

export TOKENIZERS_PARALLELISM=false

module load cuda/12.6

mkdir -p "$OUTPUT_DIR" logs

# ── Launch training ──────────────────────────────────────────────
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    "$SCRIPT_DIR/run_clm.py" \
    --model_type gpt2 \
    --tokenizer_name openai-community/gpt2 \
    --train_file "$TRAIN_FILE" \
    --validation_file "$VAL_COMBINED" \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-4 \
    --weight_decay 0.1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --lr_scheduler_type cosine \
    --warmup_steps 1000 \
    --num_train_epochs 100 \
    --block_size 1024 \
    --preprocessing_num_workers 16 \
    --bf16 \
    --logging_steps 50 \
    --eval_strategy steps \
    --eval_steps 2000 \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 7 \
    --output_dir "$OUTPUT_DIR" \
    --overwrite_output_dir \
    --report_to wandb \
    --run_name pretrain_gpt2_react \
    "$@"

echo "Training completed!"
