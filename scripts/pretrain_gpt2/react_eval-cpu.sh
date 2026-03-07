#!/bin/bash
set -euo pipefail

# ── React-only evaluation ─────────────────────────────────────────
CHECKPOINT_STEP="${1:-6000}"
shift 2>/dev/null || true

# ── Paths ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

RESULT_DIR="$PROJECT_DIR/results/pretrain_gpt2_react"
MODEL_PATH="$RESULT_DIR/checkpoint-$CHECKPOINT_STEP"
OUTPUT_DIR="$RESULT_DIR/eval-$CHECKPOINT_STEP"

export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUTPUT_DIR"

# ── Run evaluation ───────────────────────────────────────────────
python "$SCRIPT_DIR/react_eval.py" \
    --model_path "$MODEL_PATH" \
    --tokenizer_name openai-community/gpt2 \
    --benchmarks \
        "$PROJECT_DIR/datasets/gsm8k.json" \
        "$PROJECT_DIR/datasets/gsm-hard.json" \
        "$PROJECT_DIR/datasets/MultiArith.json" \
        "$PROJECT_DIR/datasets/SVAMP.json" \
    --batch_size 32 \
    --max_turns 12 \
    --max_new_tokens 128 \
    --cpu \
    --output_file "$OUTPUT_DIR/eval_results.json" \
    "$@"

echo "Evaluation completed!"
