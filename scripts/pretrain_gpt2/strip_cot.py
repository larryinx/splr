import os
import re
import sys
from pathlib import Path


def strip_cot(line: str) -> str:
    """Remove everything between || and #### (inclusive of ||)."""
    return re.sub(r"\|\|.*?####", " ####", line)


def process_file(input_path: str, output_path: str):
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            fout.write(strip_cot(line))


def main():
    project_dir = Path(__file__).resolve().parents[2]
    src_dir = project_dir / "datasets" / "gsm8k"
    dst_dir = src_dir / "wo_cot"
    dst_dir.mkdir(parents=True, exist_ok=True)

    for name in ["train.txt", "valid.txt", "test.txt"]:
        src = src_dir / name
        dst = dst_dir / name
        if not src.exists():
            print(f"Skipping {src} (not found)", file=sys.stderr)
            continue
        process_file(str(src), str(dst))
        with open(dst) as f:
            n = sum(1 for _ in f)
        print(f"{name}: {n} lines -> {dst}")


if __name__ == "__main__":
    main()
