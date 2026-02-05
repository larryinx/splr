#!/bin/bash
set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

CHECKPOINT_STEP=10000

MODEL_PATH="${1:-$PROJECT_DIR/results/pretrain_gpt2_react/checkpoint-$CHECKPOINT_STEP}"
OUTPUT_DIR="$PROJECT_DIR/results/pretrain_gpt2_react/eval-$CHECKPOINT_STEP"

export TOKENIZERS_PARALLELISM=false

module load cuda/12.6
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
    --batch_size 256 \
    --max_turns 12 \
    --max_new_tokens 128 \
    --bf16 \
    --output_file "$OUTPUT_DIR/eval_results.json" \
    "$@"

echo "Evaluation completed!"
