import json
import sys
from pathlib import Path


def convert_sample(sample: dict) -> str:
    """Convert a single JSON sample to multi-line react format."""
    question = sample["question"]
    steps = sample["steps"]
    answer = sample["answer"]

    # Split each step at the first '=' into (before=, after)
    # e.g. "<<16-3=13>>" -> ("<<16-3=", "13>>")
    parts = []
    for step in steps:
        idx = step.index("=")
        before = step[: idx + 1]  # includes '='
        after = step[idx + 1 :]   # includes '>>'
        parts.append((before, after))

    lines = []
    # First line: question || first_step_before
    lines.append(f"{question}||{parts[0][0]}")
    # Middle lines: prev_after + next_before
    for i in range(1, len(parts)):
        lines.append(f"{parts[i - 1][1]} {parts[i][0]}")
    # Last line: last_after #### answer
    lines.append(f"{parts[-1][1]} #### {answer}")

    return "\n".join(lines)


def process_file(input_path: str, output_path: str):
    with open(input_path) as f:
        data = json.load(f)

    with open(output_path, "w") as f:
        for i, sample in enumerate(data):
            if i > 0:
                f.write("\n")
            f.write(convert_sample(sample))
        f.write("\n")


def main():
    project_dir = Path(__file__).resolve().parents[2]
    src_dir = project_dir / "datasets" / "gsm8k" / "tool_normalized"
    dst_dir = project_dir / "datasets" / "gsm8k" / "react"
    dst_dir.mkdir(parents=True, exist_ok=True)

    for name in ["train", "valid", "test"]:
        src = src_dir / f"{name}.json"
        dst = dst_dir / f"{name}.txt"
        if not src.exists():
            print(f"Skipping {src} (not found)", file=sys.stderr)
            continue
        process_file(str(src), str(dst))

        with open(src) as f:
            n_samples = len(json.load(f))
        with open(dst) as f:
            n_lines = sum(1 for _ in f)
        print(f"{name}: {n_samples} samples, {n_lines} lines -> {dst}")


if __name__ == "__main__":
    main()
