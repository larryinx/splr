# Step-wise Persistent Latent Reasoner

This is the repository of the paper "**Learning Multi-step Reasoning via Persistent Latent State Propagation**" (under review), a lightweight multi-step latent reasoner that propagates persistent hidden states across reasoning steps, replacing long chain-of-thought token traces with compact hierarchical latent dynamics.

## Environment Setup

This repository now relies on [uv](https://github.com/astral-sh/uv) for dependency and virtual-environment management and targets Python 3.11.

1. Install `uv` (one-line installers are listed in the uv README) if you do not already have it.
2. Run `./prepare-env.sh [python_version]` (defaults to `3.11`). The script writes the requested version to `.python-version` and runs `uv sync --dev` to create or update `.venv`.
3. Activate the environment for interactive work with `source .venv/bin/activate` (or use `uv run ...` to execute commands without activating).

## Dataset Preparation

**Generating dataset for GSM8k:**

```bash
# Download the GSM8k-Aug dataset
bash scripts/data_preprocessing/gsm.bash
```

Additional evaluation benchmarks are already included in `datasets/`: MultiArith, SVAMP, GSM-Hard.

## Training Scripts

SPLR supports two interaction modes: **Think** (reasoning-only) and **ReAct** (reasoning with external observations). Training uses curriculum learning -- set `MAX_REASONING_STEPS`, and optionally `LOAD_CHECKPOINT` and `NUM_EPOCHS`, for each curriculum stage.

```bash
# Think (reasoning-only)
bash scripts/train/train_think.sh

# ReAct (with tool use)
bash scripts/train/train_react.sh
```

## Evaluation Scripts

To evaluate a trained checkpoint, set the following parameters before running the eval script: `ARCH`, `MODE`, `MAX_REASONING_STEPS`, and `LOAD_CHECKPOINT`. Then run:

```bash
# Think evaluation
bash scripts/eval/eval_think.sh

# ReAct evaluation
bash scripts/eval/eval_react.sh
```

By default, evaluation propagates the full latent state across reasoning steps. To study the effect of inference-time latent disruption, those settings can be configured in the evaluation script by setting `ARCH` to `splr_trm_gpt_zl` or `splr_trm_gpt_dis`.
