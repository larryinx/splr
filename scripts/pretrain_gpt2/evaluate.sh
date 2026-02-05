#!/bin/bash
set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_PATH="${1:-$PROJECT_DIR/results/pretrain_gpt2_wo_cot/checkpoint-$CHECKPOINT_STEP}"
OUTPUT_DIR="$PROJECT_DIR/results/pretrain_gpt2_wo_cot/eval-$CHECKPOINT_STEP"

export TOKENIZERS_PARALLELISM=false

# module load cuda/12.6

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
    "$@"

echo "Evaluation completed!"
