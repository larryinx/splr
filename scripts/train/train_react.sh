#!/bin/bash
# This is the script to train the first curriculum learning step of the SPLR model.

# -------------------Basic Configuration-------------------
export TOKENIZERS_PARALLELISM=false

NUM_GPUS=2
MASTER_PORT=29500
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TOKENIZER_PATH="$PROJECT_DIR/src/splr/core/tokenizers/gpt2"

ARCH=splr_trm_gpt
MODE=recurrent
OUTPUT_DIR="$PROJECT_DIR/results/experiments/pretrain_react/${ARCH}_${MODE}_${MAX_REASONING_STEPS}"

# ---------------(To be set) Training Configuration---------------

# Please set the MAX_REASONING_STEPS, LOAD_CHECKPOINT, and NUM_EPOCHS for the following curriculum learning steps.
# Then pass those as arguments to the training script.
MAX_REASONING_STEPS=2
LOAD_CHECKPOINT=$OUTPUT_DIR/checkpoints/step_<step_number>.pt
NUM_EPOCHS=100

# -------------------------------------------------------------

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
    run_name=${ARCH}_${MODE}_${MAX_REASONING_STEPS} \
    arch.max_reasoning_steps=$MAX_REASONING_STEPS \
    # load_checkpoint=$LOAD_CHECKPOINT \
    # num_epochs=$NUM_EPOCHS

echo "Training completed!"
