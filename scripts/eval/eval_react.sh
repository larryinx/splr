#!/bin/bash
# Evaluation script for the SPLR model (ReAct / tool-use mode).

# -------------------Basic Configuration-------------------
export TOKENIZERS_PARALLELISM=false

NUM_GPUS=2
MASTER_PORT=29500
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TOKENIZER_PATH="$PROJECT_DIR/src/splr/core/tokenizers/gpt2"

# ---------------(To be set) Evaluation Configuration---------------

# Set the following parameters before running.
ARCH=splr_trm_gpt
MODE=recurrent
MAX_REASONING_STEPS=12
OUTPUT_DIR="$PROJECT_DIR/results/experiments/pretrain_react/${ARCH}_${MODE}_${MAX_REASONING_STEPS}"
LOAD_CHECKPOINT=$OUTPUT_DIR/checkpoints/step_<step_number>.pt
EVAL_RESULTS_DIR="$PROJECT_DIR/results/experiments/eval_results_react/${ARCH}_${MODE}_${MAX_REASONING_STEPS}"

# -------------------------------------------------------------


torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    main.py \
    --config-name cfg_eval_tool \
    arch=$ARCH \
    input_mode=$MODE \
    tokenizer_path=$TOKENIZER_PATH \
    project_name=splr_react_eval \
    run_name=${ARCH}_${MODE}_${MAX_REASONING_STEPS} \
    load_checkpoint=$LOAD_CHECKPOINT \
    eval_results_dir=$EVAL_RESULTS_DIR

echo "Evaluation completed!"
