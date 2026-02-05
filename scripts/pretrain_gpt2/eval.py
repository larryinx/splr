import argparse
import json
import logging
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 on math benchmarks")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pretrained model checkpoint directory",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Tokenizer name or path (defaults to model_path)",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        required=True,
        help="Paths to benchmark JSON files",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional path to save detailed results as JSON",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 precision",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference (ignores --bf16)",
    )
    return parser.parse_args()


def extract_answer(text: str) -> str | None:
    """Extract the final numeric answer after '####' from generated text."""
    match = re.search(r"####\s*([^\n]+)", text)
    if match:
        return match.group(1).strip()
    return None


def normalize_number(s: str) -> float | None:
    """Normalize a numeric string for comparison."""
    if s is None:
        return None
    # Remove commas, dollar signs, percent signs, trailing periods
    s = s.replace(",", "").replace("$", "").replace("%", "").strip().rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def answers_match(predicted: str | None, gold: str) -> bool:
    """Compare predicted and gold answers numerically."""
    pred_num = normalize_number(predicted)
    gold_num = normalize_number(gold)
    if pred_num is None or gold_num is None:
        return False
    # Compare with small tolerance for floating point
    if gold_num == 0:
        return abs(pred_num) < 1e-6
    return abs(pred_num - gold_num) / max(abs(gold_num), 1e-12) < 1e-6


def load_benchmark(path: str) -> list[dict]:
    """Load a benchmark JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    """Generate completions for a batch of prompts."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
    ).to(model.device)

    prompt_lengths = inputs["attention_mask"].sum(dim=1)

    with torch.no_grad():
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        outputs = model.generate(**inputs, **gen_kwargs)

    completions = []
    for i, output in enumerate(outputs):
        # Decode only the generated part (after the prompt)
        generated_ids = output[prompt_lengths[i] :]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        completions.append(text)

    return completions


def evaluate_benchmark(
    model,
    tokenizer,
    benchmark_path: str,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
) -> dict:
    """Evaluate model on a single benchmark. Returns results dict."""
    data = load_benchmark(benchmark_path)
    name = Path(benchmark_path).stem

    # Build prompts: "question||"
    prompts = [item["question"] + "||" for item in data]
    gold_answers = [item["answer"] for item in data]

    # Generate in batches
    all_completions = []
    for i in tqdm(range(0, len(prompts), batch_size), desc=name):
        batch_prompts = prompts[i : i + batch_size]
        completions = generate_batch(
            model, tokenizer, batch_prompts, max_new_tokens, temperature
        )
        all_completions.extend(completions)

    # Extract answers and compare
    correct = 0
    details = []
    for i, (completion, gold) in enumerate(zip(all_completions, gold_answers)):
        predicted = extract_answer(completion)
        is_correct = answers_match(predicted, gold)
        if is_correct:
            correct += 1

        details.append(
            {
                "question": data[i]["question"],
                "gold_answer": gold,
                "predicted_answer": predicted,
                "correct": is_correct,
                "generation": completion,
            }
        )

    accuracy = correct / len(data) if data else 0.0
    return {
        "benchmark": name,
        "path": benchmark_path,
        "total": len(data),
        "correct": correct,
        "accuracy": accuracy,
        "details": details,
    }


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    # Load model and tokenizer
    tokenizer_name = args.tokenizer_name or args.model_path
    logger.info(f"Loading tokenizer from {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    logger.info(f"Loading model from {args.model_path}")
    if args.cpu:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float32,
        )
    else:
        dtype = torch.bfloat16 if args.bf16 else None
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            device_map="auto",
        )
    model.eval()

    # Evaluate each benchmark
    all_results = []
    for bench_path in args.benchmarks:
        logger.info(f"Evaluating on {bench_path}")
        result = evaluate_benchmark(
            model,
            tokenizer,
            bench_path,
            args.batch_size,
            args.max_new_tokens,
            args.temperature,
        )
        all_results.append(result)

        print(f"\n{'=' * 60}")
        print(f"  {result['benchmark']}")
        print(f"  Accuracy: {result['correct']}/{result['total']} = {result['accuracy']:.4f}")
        print(f"{'=' * 60}")

    # Summary
    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    for r in all_results:
        print(f"  {r['benchmark']:20s}  {r['correct']:4d}/{r['total']:<4d}  {r['accuracy']:.4f}")
    print(f"{'=' * 60}\n")

    # Save detailed results
    if args.output_file:
        output = {r["benchmark"]: r for r in all_results}
        with open(args.output_file, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Detailed results saved to {args.output_file}")


if __name__ == "__main__":
    main()
