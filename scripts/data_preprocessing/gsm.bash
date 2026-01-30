#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Create data directory if it doesn't exist
mkdir -p ./data/input/gsm8k

# Remove metadata if exists
if [ -f ./data/input/gsm8k/multi_normal/metadata.json ]; then
  rm ./data/input/gsm8k/multi_normal/metadata.json
fi

# Download and process GSM8K dataset for Internalize CoT

wget https://media.githubusercontent.com/media/da03/Internalize_CoT_Step_by_Step/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k/train.txt -O ./data/input/gsm8k/train.txt
wget https://raw.githubusercontent.com/da03/Internalize_CoT_Step_by_Step/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k/valid.txt -O ./data/input/gsm8k/valid.txt
wget https://raw.githubusercontent.com/da03/Internalize_CoT_Step_by_Step/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k/test.txt -O ./data/input/gsm8k/test.txt

for split in train valid test; do
  python -m src.recursive.data_generator --dataset gsm8k --split ${split}
  rm ./data/input/gsm8k/${split}.txt
done