"""Check maximum question token lengths across datasets for GPT2 and Qwen tokenizers."""

import json
from pathlib import Path

from transformers import AutoTokenizer

DATASETS_DIR = Path(__file__).resolve().parents[2] / "datasets"
TOKENIZERS_DIR = Path(__file__).resolve().parents[2] / "src" / "splr" / "core" / "tokenizers"

TOKENIZER_PATHS = {
    "gpt2": TOKENIZERS_DIR / "gpt2",
    "qwen": TOKENIZERS_DIR / "qwen",
}


def main():
    tokenizers = {
        name: AutoTokenizer.from_pretrained(str(path))
        for name, path in TOKENIZER_PATHS.items()
    }

    json_files = sorted(DATASETS_DIR.glob("*.json"))
    print(f"Found {len(json_files)} JSON files in {DATASETS_DIR}\n")
    print(f"{'Dataset':<20} {'Samples':>8} {'GPT2 Max':>10} {'Qwen Max':>10}")
    print("-" * 52)

    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)

        questions = [sample["question"] for sample in data]
        max_lengths = {}
        for name, tok in tokenizers.items():
            lengths = [len(tok.encode(q)) for q in questions]
            max_lengths[name] = max(lengths)

        print(
            f"{json_file.stem:<20} {len(questions):>8} "
            f"{max_lengths['gpt2']:>10} {max_lengths['qwen']:>10}"
        )


if __name__ == "__main__":
    main()
