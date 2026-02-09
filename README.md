# Recursive-Internal

This is an beta version of the SPLR project README.md file. We will update the official version later.

## Development setup

This repository now relies on [uv](https://github.com/astral-sh/uv) for dependency and virtual-environment management and targets Python 3.11.

1. Install `uv` (one-line installers are listed in the uv README) if you do not already have it.
2. Run `./prepare-env.sh [python_version]` (defaults to `3.11`). The script writes the requested version to `.python-version` and runs `uv sync --dev` to create or update `.venv`.
3. Activate the environment for interactive work with `source .venv/bin/activate` (or use `uv run ...` to execute commands without activating).


## Generating datasets

**Generating dataset for GSM8k:**

```bash
# Download the GSM8k-Aug dataset
bash scripts/data_preprocessing/gsm.bash
```

## Training

To train the models:
```bash
bash scripts/train/train_react.sh
bash scripts/train/train_think.sh
```