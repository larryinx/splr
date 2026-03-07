#!/bin/bash
#SBATCH --job-name=pretrain_gpt2_init_emb
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

# ── Dataset variant: react | wo_cot | cot (default) ─────────────
VARIANT="${1:-cot}"
shift 2>/dev/null || true

case "$VARIANT" in
    react|wo_cot)
        DATA_SUFFIX="/$VARIANT"
        NAME_SUFFIX="_${VARIANT}"
        ;;
    cot)
        DATA_SUFFIX=""
        NAME_SUFFIX=""
        ;;
    *)
        echo "Usage: $0 [react|wo_cot|cot] [extra args...]"
        exit 1
        ;;
esac

# ── Paths ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATA_DIR="$PROJECT_DIR/datasets/gsm8k${DATA_SUFFIX}"
TRAIN_FILE="$DATA_DIR/train.txt"
VALID_FILE="$DATA_DIR/valid.txt"
TEST_FILE="$DATA_DIR/test.txt"

OUTPUT_DIR="$PROJECT_DIR/results/pretrain_gpt2${NAME_SUFFIX}_init_emb"

# ── Pretrained model to take embeddings from ─────────────────────
INIT_EMB_FROM="openai-community/gpt2"

# ── Combine validation files ─────────────────────────────────────
VAL_COMBINED="$DATA_DIR/val_combined.txt"
cat "$VALID_FILE" "$TEST_FILE" > "$VAL_COMBINED"
echo "Variant: $VARIANT | Data dir: $DATA_DIR"
echo "Combined validation file: $(wc -l < "$VAL_COMBINED") lines"

# ── Training config ──────────────────────────────────────────────
NUM_GPUS=2
# MASTER_PORT=${MASTER_PORT:-29500}
MASTER_PORT=29600

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=2,3

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
    --init_embeddings_from "$INIT_EMB_FROM" \
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
    --run_name "pretrain_gpt2${NAME_SUFFIX}_init_emb" \
    "$@"

echo "Training completed!"
