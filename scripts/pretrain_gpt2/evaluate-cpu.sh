#!/bin/bash
set -euo pipefail

# ── Dataset variant: wo_cot | cot (default) ──────────────────────
VARIANT="${1:-cot}"
shift 2>/dev/null || true
CHECKPOINT_STEP="${1:-6000}"
shift 2>/dev/null || true

case "$VARIANT" in
    wo_cot) NAME_SUFFIX="_wo_cot" ;;
    cot)    NAME_SUFFIX="" ;;
    *)  echo "Usage: $0 [wo_cot|cot] [checkpoint_step] [extra args...]"; exit 1 ;;
esac

# ── Paths ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

RESULT_DIR="$PROJECT_DIR/results/pretrain_gpt2${NAME_SUFFIX}"
MODEL_PATH="$RESULT_DIR/checkpoint-$CHECKPOINT_STEP"
OUTPUT_DIR="$RESULT_DIR/eval-$CHECKPOINT_STEP"

export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUTPUT_DIR"

# ── Run evaluation ───────────────────────────────────────────────
python "$SCRIPT_DIR/eval.py" \
    --model_path "$MODEL_PATH" \
    --tokenizer_name openai-community/gpt2 \
    --benchmarks \
        "$PROJECT_DIR/datasets/gsm8k.json" \
        "$PROJECT_DIR/datasets/gsm-hard.json" \
        "$PROJECT_DIR/datasets/MultiArith.json" \
        "$PROJECT_DIR/datasets/SVAMP.json" \
    --batch_size 32 \
    --max_new_tokens 256 \
    --bf16 \
    --output_file "$OUTPUT_DIR/eval_results.json" \
    --cpu \
    "$@"

echo "Evaluation completed!"
