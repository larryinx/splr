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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_formula(expr: str) -> float | None:
    """Safely evaluate an arithmetic expression."""
    expr = expr.strip()
    if not expr:
        return None
    # Only allow digits, operators, parentheses, dots, spaces
    if not re.match(r"^[\d+\-*/().  ]+$", expr):
        return None
    # Reject exponentiation (**) — can hang eval() on adversarial inputs
    if '**' in expr:
        return None
    try:
        result = eval(expr, {"__builtins__": {}}, {})
        if isinstance(result, (int, float)):
            return float(result)
        return None
    except Exception:
        return None


def format_result(value: float) -> str:
    """Format a numeric result to match training data."""
    if value == int(value):
        return str(int(value))
    return f"{value:.10f}".rstrip("0").rstrip(".")


def extract_answer(text: str) -> str | None:
    """Extract the answer string after ####."""
    match = re.search(r"####\s*(.+)", text)
    return match.group(1).strip() if match else None


def normalize_number(s: str) -> float | None:
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").replace("%", "").strip().rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def answers_match(predicted: str | None, gold: str) -> bool:
    pred_num = normalize_number(predicted)
    gold_num = normalize_number(gold)
    if pred_num is None or gold_num is None:
        return False
    if gold_num == 0:
        return abs(pred_num) < 1e-6
    return abs(pred_num - gold_num) / max(abs(gold_num), 1e-12) < 1e-6


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    stop_token_ids: list[int],
) -> list[str]:
    """Generate one turn for a batch of prompts, stopping at newline or EOS."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Reserve room for generation so total length stays within position embeddings
    max_input_length = tokenizer.model_max_length - max_new_tokens

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    ).to(model.device)

    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=stop_token_ids,
            pad_token_id=tokenizer.pad_token_id,
        )

    completions = []
    for i, output in enumerate(outputs):
        gen_ids = output[input_length:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        completions.append(text.rstrip())

    return completions


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_benchmark(
    model,
    tokenizer,
    benchmark_path: str,
    batch_size: int,
    max_turns: int,
    max_new_tokens: int,
    stop_token_ids: list[int],
) -> dict:
    with open(benchmark_path) as f:
        data = json.load(f)

    name = Path(benchmark_path).stem
    questions = [item["question"] for item in data]
    gold_answers = [item["answer"] for item in data]
    n = len(questions)

    # State for every sample
    accumulated = [f"{q}||" for q in questions]
    stopped = [False] * n
    predicted = [None] * n
    step_traces = [[] for _ in range(n)]  # per-sample list of step dicts
    # Max input tokens before generation would require truncation
    max_input_tokens = tokenizer.model_max_length - max_new_tokens

    for turn in range(max_turns):
        # Stop samples whose context already exceeds the model's capacity
        for i in range(n):
            if not stopped[i]:
                n_tokens = len(tokenizer.encode(accumulated[i], add_special_tokens=False))
                if n_tokens >= max_input_tokens:
                    step_traces[i].append({
                        "turn": turn + 1,
                        "generated": "",
                        "action": "stopped_context_too_long",
                        "n_tokens": n_tokens,
                    })
                    stopped[i] = True

        active = [i for i in range(n) if not stopped[i]]
        if not active:
            break

        # Process active samples in sub-batches
        for batch_start in range(0, len(active), batch_size):
            batch_idx = active[batch_start : batch_start + batch_size]
            batch_prompts = [accumulated[i] for i in batch_idx]

            generated = generate_batch(
                model, tokenizer, batch_prompts, max_new_tokens, stop_token_ids
            )

            for j, idx in enumerate(batch_idx):
                text = generated[j]

                # 1) Check for final answer
                if "####" in text:
                    predicted[idx] = extract_answer(text)
                    step_traces[idx].append({
                        "turn": turn + 1,
                        "generated": text,
                        "action": "final_answer",
                        "answer": predicted[idx],
                    })
                    stopped[idx] = True
                    continue

                # 2) Find formula <<expr=
                match = re.search(r"<<(.+?)=", text)
                if match is None:
                    step_traces[idx].append({
                        "turn": turn + 1,
                        "generated": text,
                        "action": "stopped_no_formula",
                    })
                    stopped[idx] = True
                    continue

                result = compute_formula(match.group(1))
                if result is None:
                    step_traces[idx].append({
                        "turn": turn + 1,
                        "generated": text,
                        "action": "stopped_invalid_formula",
                        "formula": match.group(1),
                    })
                    stopped[idx] = True
                    continue

                # Track latest result as potential answer
                predicted[idx] = format_result(result)
                step_traces[idx].append({
                    "turn": turn + 1,
                    "generated": text,
                    "action": "tool_call",
                    "formula": match.group(1),
                    "result": format_result(result),
                })
                # Append: generated text + \n + EOS + result>>
                # No trailing space — the model will generate " <<" as one BPE pre-token
                accumulated[idx] += text + "\n" + tokenizer.eos_token + format_result(result) + ">>"

        logger.info(
            f"  {name}: turn {turn + 1}/{max_turns} done, "
            f"{sum(stopped)}/{n} stopped"
        )

    # Score
    correct = 0
    details = []
    for i in range(n):
        is_correct = answers_match(predicted[i], gold_answers[i])
        if is_correct:
            correct += 1
        details.append(
            {
                "question": questions[i],
                "gold_answer": gold_answers[i],
                "predicted_answer": predicted[i],
                "correct": is_correct,
                "steps": step_traces[i],
                "final_context": accumulated[i],
            }
        )

    accuracy = correct / n if n else 0.0
    return {
        "benchmark": name,
        "path": benchmark_path,
        "total": n,
        "correct": correct,
        "accuracy": accuracy,
        "details": details,
    }



def parse_args():
    p = argparse.ArgumentParser(description="ReAct multi-turn eval for GPT-2")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--tokenizer_name", type=str, default=None)
    p.add_argument(
        "--benchmarks", type=str, nargs="+", required=True,
        help="Paths to benchmark JSON files",
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_turns", type=int, default=12)
    p.add_argument("--max_new_tokens", type=int, default=128,
                    help="Max tokens to generate per turn")
    p.add_argument("--output_file", type=str, default=None)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--cpu", action="store_true",
                    help="Force CPU inference (ignores --bf16)")
    return p.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    tokenizer_name = args.tokenizer_name or args.model_path
    logger.info(f"Loading tokenizer from {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    logger.info(f"Loading model from {args.model_path}")
    if args.cpu:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.float32,
        )
    else:
        dtype = torch.bfloat16 if args.bf16 else None
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=dtype, device_map="auto",
        )
    model.eval()

    # Stop generation at newline or EOS
    newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]
    stop_token_ids = [tokenizer.eos_token_id, newline_id]

    all_results = []
    for bench_path in args.benchmarks:
        logger.info(f"Evaluating on {bench_path}")
        result = evaluate_benchmark(
            model, tokenizer, bench_path,
            args.batch_size, args.max_turns, args.max_new_tokens,
            stop_token_ids,
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

    if args.output_file:
        output = {r["benchmark"]: r for r in all_results}
        with open(args.output_file, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Detailed results saved to {args.output_file}")


if __name__ == "__main__":
    main()
