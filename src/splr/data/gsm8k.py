# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
import re
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------------------

def process_sample(sample: Dict[str, Any]) -> tuple[Dict[str, Any], bool, str]:
    """
    Process a single sample: filter out trivial equation steps (no operators).
    Only validates that the last step's result matches the answer.

    Returns:
        Tuple of (processed_sample, should_skip, error_message)
    """
    try:
        if not sample.get('steps'):
            return sample, True, "Empty steps"

        question = sample['question']
        answer = sample['answer']

        # Keep only steps whose left side contains an operator
        new_steps = []
        for step in sample['steps']:
            match = re.search(r'<<(.+?)>>', step)
            if not match:
                continue
            equation = match.group(1)
            if '=' not in equation:
                continue
            left_side = equation.split('=', 1)[0]
            if any(op in left_side for op in '+-*/()'):
                new_steps.append(step)

        if not new_steps:
            return sample, True, "No valid steps after filtering"

        # Check last step result matches answer
        last_match = re.search(r'<<.+?=(.+?)>>', new_steps[-1])
        if last_match:
            last_result = last_match.group(1).strip()
            try:
                if abs(float(last_result) - float(answer)) > 1e-9:
                    return sample, True, "Last step result doesn't match answer"
            except ValueError:
                if last_result != answer.strip():
                    return sample, True, "Last step result doesn't match answer"

        return {
            'question': question,
            'steps': new_steps,
            'answer': answer,
            'task_id': 0,
        }, False, ""

    except Exception as e:
        return sample, True, str(e)


def generate_think_data(data: list[Dict[str, Any]], split: str) -> Dict[str, Any]:
    """
    Generate think-normalized version of the dataset.
    Output to ./datasets/gsm8k/think_normalized/{split}.json
    No metadata. All task_id = 0.
    """
    output_dir = Path("./datasets/gsm8k/think_normalized")
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_data = []
    stats = {
        'total': len(data),
        'processed': 0,
        'skipped': 0,
        'max_steps_length': 0,
        'steps_length_distribution': {},
    }

    for sample in tqdm(data, desc=f"Processing {split}", unit="sample"):
        processed, should_skip, error = process_sample(sample)
        if should_skip:
            stats['skipped'] += 1
            continue

        steps_length = len(processed['steps'])
        if steps_length > stats['max_steps_length']:
            stats['max_steps_length'] = steps_length

        length_key = str(steps_length)
        stats['steps_length_distribution'][length_key] = stats['steps_length_distribution'].get(length_key, 0) + 1

        processed_data.append(processed)
        stats['processed'] += 1

    output_path = output_dir / f"{split}.json"
    with open(output_path, 'w') as f:
        json.dump(processed_data, f, indent=2)

    print(f"\n=== Think-Normalized Processing Statistics ===")
    print(f"Total samples: {stats['total']}")
    print(f"Processed: {stats['processed']} ({stats['processed']/stats['total']*100:.2f}%)")
    print(f"Skipped: {stats['skipped']} ({stats['skipped']/stats['total']*100:.2f}%)")
    print(f"Max steps length: {stats['max_steps_length']}")

    if stats['steps_length_distribution']:
        print(f"\nSteps length distribution:")
        sorted_lengths = sorted(stats['steps_length_distribution'].items(), key=lambda x: int(x[0]))
        for length, count in sorted_lengths:
            percentage = count / stats['processed'] * 100 if stats['processed'] > 0 else 0
            print(f"  {length} step(s): {count:6d} samples ({percentage:5.2f}%)")

    print(f"\nSaved to: {output_path}")
    return stats


def generate_data(args):
    """
    Convert icot text data to JSON format and generate think-normalized version.
    """
    from transformers import AutoTokenizer

    print("Loading tokenizer for token counting...")
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    with open(f"./datasets/gsm8k/{args.split}.txt") as f:
        data = f.readlines()

    token_stats = {
        "question": {"samples": [], "total": 0},
        "steps": {"samples": [], "total": 0},
        "answer": {"samples": [], "total": 0},
        "all": {"samples": [], "total": 0}
    }

    data = [
        {
            "question": d.split("||")[0],
            "steps": d.split("||")[1].split("##")[0].strip().split(" "),
            "answer": d.split("##")[-1].strip(),
        }
        for d in data
    ]

    print(f"Counting tokens for {len(data)} samples...")
    for sample in data:
        question = sample["question"]
        steps = " ".join(sample["steps"])
        answer = sample["answer"]
        all_text = f"{question} {steps} {answer}"

        question_tokens = len(tokenizer.encode(question))
        steps_tokens = len(tokenizer.encode(steps))
        answer_tokens = len(tokenizer.encode(answer))
        all_tokens = len(tokenizer.encode(all_text))

        token_stats["question"]["samples"].append(question_tokens)
        token_stats["question"]["total"] += question_tokens
        token_stats["steps"]["samples"].append(steps_tokens)
        token_stats["steps"]["total"] += steps_tokens
        token_stats["answer"]["samples"].append(answer_tokens)
        token_stats["answer"]["total"] += answer_tokens
        token_stats["all"]["samples"].append(all_tokens)
        token_stats["all"]["total"] += all_tokens

    print("\n=== Token Statistics ===")
    for field_name, stats in token_stats.items():
        samples = stats["samples"]
        if not samples:
            continue
        print(f"\n{field_name.upper()}:")
        print(f"  Min: {min(samples)} tokens")
        print(f"  Max: {max(samples)} tokens")
        print(f"  Average: {stats['total'] // len(samples)} tokens")
        print(f"  Total: {stats['total']} tokens")

    json.dump(data, open(f"./datasets/gsm8k/{args.split}.json", "w"))

    print(f"\n{'='*80}")
    print("Generating think-normalized version...")
    print(f"{'='*80}")
    generate_think_data(data, args.split)


# ---------------------------------------------------------------------------
# Dataset loading — <think> style
# ---------------------------------------------------------------------------

def _extract_step_content(step: str) -> str:
    """Extract content from a step, stripping << and >> markers."""
    match = re.search(r"<<(.+?)>>", step)
    if match:
        return match.group(1)
    return step


def load_dataset_accum(file_paths, tokenizer, config):
    """
    Load and preprocess GSM8K dataset for multi-step reasoning with accumulated context.
    Each turn's input includes the full conversation history.
    Labels use <think>step_content</think> format.
    """
    import torch

    if isinstance(file_paths, str):
        file_paths = [file_paths]

    print(f"Loading GSM8K think ACCUMULATED dataset from {len(file_paths)} file(s)")

    max_reasoning_steps = getattr(config, 'max_reasoning_steps', 12)

    task_emb_ndim = getattr(config, 'task_emb_ndim', 0)
    task_emb_len_cfg = getattr(config, 'task_emb_len', 0)

    if task_emb_ndim > 0:
        task_emb_len = -(task_emb_ndim // -config.hidden_size) if task_emb_len_cfg == 0 else task_emb_len_cfg
        input_max_len = config.max_position_embeddings - task_emb_len
        print(f"Task embeddings enabled: task_emb_len={task_emb_len}, input_max_len={input_max_len}")
    else:
        task_emb_len = 0
        input_max_len = config.max_position_embeddings
        print(f"Task embeddings disabled: input_max_len={input_max_len}")

    print(f"Max reasoning steps: {max_reasoning_steps}")

    all_data = []
    for file_path in file_paths:
        print(f"  Loading {file_path}...")
        with open(file_path, 'r') as f:
            all_data.extend(json.load(f))

    print(f"Loaded {len(all_data)} total samples")

    processed = []
    skipped_count = 0

    for i, sample in enumerate(tqdm(all_data, desc="Tokenizing samples")):
        try:
            question = sample["question"]
            steps = sample.get("steps", [])

            if not steps:
                skipped_count += 1
                continue

            inputs_list = []
            labels_list = []
            task_ids_list = []

            accumulated_text = question

            for step_idx, step in enumerate(steps):
                if step_idx >= max_reasoning_steps:
                    break

                step_content = _extract_step_content(step)

                # Build input for this turn
                if step_idx == 0:
                    input_text = question
                else:
                    input_text = accumulated_text

                # Tokenize input
                input_tokens = tokenizer.encode(
                    input_text,
                    max_length=input_max_len,
                    truncation=True,
                    padding="max_length"
                )

                # Tokenize input text alone to know how many tokens to mask
                input_text_tokens = tokenizer.encode(
                    input_text,
                    add_special_tokens=False,
                    truncation=False
                )

                # Build label: input + <think>step_content</think>
                if step_idx == len(steps) - 1:
                    label_text = input_text + tokenizer.eos_token + f"<think>{step_content}</think>"
                else:
                    label_text = f"{input_text}<think>{step_content}</think>"

                label_tokens = tokenizer.encode(
                    label_text,
                    add_special_tokens=False,
                    padding="max_length",
                    max_length=input_max_len,
                )

                label_tokens_tensor = torch.tensor(label_tokens, dtype=torch.long)
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

                # Mask the input portion
                input_token_len = len(input_text_tokens)
                for j in range(min(input_token_len, len(label_tokens))):
                    label_tokens_tensor[j] = pad_token_id

                if config.max_position_embeddings > input_max_len:
                    label_tokens_tensor = torch.cat([
                        torch.full((config.max_position_embeddings - input_max_len,), pad_token_id, dtype=torch.long),
                        label_tokens_tensor
                    ])

                inputs_list.append(torch.tensor(input_tokens, dtype=torch.long))
                labels_list.append(label_tokens_tensor)
                task_ids_list.append(torch.tensor(0, dtype=torch.long))

                # Accumulate: input + <think>step_content</think>
                accumulated_text = f"{input_text}<think>{step_content}</think>"

            # Pad remaining turns
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            pad_input = torch.full((input_max_len,), pad_token_id, dtype=torch.long)
            pad_label = torch.full((config.max_position_embeddings,), pad_token_id, dtype=torch.long)
            pad_task_id = torch.tensor(-1, dtype=torch.long)

            for _ in range(len(inputs_list), max_reasoning_steps):
                inputs_list.append(pad_input)
                labels_list.append(pad_label)
                task_ids_list.append(pad_task_id)

            processed.append({
                "inputs": torch.stack(inputs_list),
                "labels": torch.stack(labels_list),
                "task_identifiers": torch.stack(task_ids_list),
            })

        except Exception as e:
            print(f"Warning: Error processing sample {i}: {e}")
            skipped_count += 1
            continue

    print(f"Successfully processed {len(processed)} samples from GSM8K think ACCUMULATED dataset")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} samples due to errors or missing data")

    return processed


def load_dataset(file_paths, tokenizer, config):
    """
    Load and preprocess GSM8K dataset for multi-step reasoning.
    Each turn's input is the question plus only the previous step.
    Labels use <think>step_content</think> format.
    """
    import torch

    if isinstance(file_paths, str):
        file_paths = [file_paths]

    print(f"Loading GSM8K think dataset from {len(file_paths)} file(s)")

    max_reasoning_steps = getattr(config, 'max_reasoning_steps', 12)

    task_emb_ndim = getattr(config, 'task_emb_ndim', 0)
    task_emb_len_cfg = getattr(config, 'task_emb_len', 0)

    if task_emb_ndim > 0:
        task_emb_len = -(task_emb_ndim // -config.hidden_size) if task_emb_len_cfg == 0 else task_emb_len_cfg
        input_max_len = config.max_position_embeddings - task_emb_len
        print(f"Task embeddings enabled: task_emb_len={task_emb_len}, input_max_len={input_max_len}")
    else:
        task_emb_len = 0
        input_max_len = config.max_position_embeddings
        print(f"Task embeddings disabled: input_max_len={input_max_len}")

    print(f"Max reasoning steps: {max_reasoning_steps}")

    all_data = []
    for file_path in file_paths:
        print(f"  Loading {file_path}...")
        with open(file_path, 'r') as f:
            all_data.extend(json.load(f))

    print(f"Loaded {len(all_data)} total samples")

    processed = []
    skipped_count = 0

    for i, sample in enumerate(tqdm(all_data, desc="Tokenizing samples")):
        try:
            question = sample["question"]
            steps = sample.get("steps", [])

            if not steps:
                skipped_count += 1
                continue

            inputs_list = []
            labels_list = []
            task_ids_list = []

            prev_step_content = None

            for step_idx, step in enumerate(steps):
                if step_idx >= max_reasoning_steps:
                    break

                step_content = _extract_step_content(step)

                # Build input for this turn
                if step_idx == 0:
                    input_text = question
                else:
                    input_text = f"{question}<think>{prev_step_content}</think>"

                # Tokenize input
                input_tokens = tokenizer.encode(
                    input_text,
                    max_length=input_max_len,
                    truncation=True,
                    padding="max_length"
                )

                input_text_tokens = tokenizer.encode(
                    input_text,
                    add_special_tokens=False,
                    truncation=False
                )

                # Build label: input + <think>step_content</think>
                if step_idx == len(steps) - 1:
                    label_text = input_text + tokenizer.eos_token + f"<think>{step_content}</think>"
                else:
                    label_text = f"{input_text}<think>{step_content}</think>"

                label_tokens = tokenizer.encode(
                    label_text,
                    add_special_tokens=False,
                    padding="max_length",
                    max_length=input_max_len,
                )

                label_tokens_tensor = torch.tensor(label_tokens, dtype=torch.long)
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

                # Mask the input portion
                input_token_len = len(input_text_tokens)
                for j in range(min(input_token_len, len(label_tokens))):
                    label_tokens_tensor[j] = pad_token_id

                if config.max_position_embeddings > input_max_len:
                    label_tokens_tensor = torch.cat([
                        torch.full((config.max_position_embeddings - input_max_len,), pad_token_id, dtype=torch.long),
                        label_tokens_tensor
                    ])

                inputs_list.append(torch.tensor(input_tokens, dtype=torch.long))
                labels_list.append(label_tokens_tensor)
                task_ids_list.append(torch.tensor(0, dtype=torch.long))

                prev_step_content = step_content

            # Pad remaining turns
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            pad_input = torch.full((input_max_len,), pad_token_id, dtype=torch.long)
            pad_label = torch.full((config.max_position_embeddings,), pad_token_id, dtype=torch.long)
            pad_task_id = torch.tensor(-1, dtype=torch.long)

            for _ in range(len(inputs_list), max_reasoning_steps):
                inputs_list.append(pad_input)
                labels_list.append(pad_label)
                task_ids_list.append(pad_task_id)

            processed.append({
                "inputs": torch.stack(inputs_list),
                "labels": torch.stack(labels_list),
                "task_identifiers": torch.stack(task_ids_list),
            })

        except Exception as e:
            print(f"Warning: Error processing sample {i}: {e}")
            skipped_count += 1
            continue

    print(f"Successfully processed {len(processed)} samples from GSM8K think dataset")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} samples due to errors or missing data")

    return processed


if __name__ == "__main__":
    from transformers import AutoTokenizer

    class Config:
        max_position_embeddings = 256
        hidden_size = 1024
        task_emb_ndim = 1024
        task_emb_len = 1
        max_reasoning_steps = 6

    tokenizer_path = ""

    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    file_paths = [
        "./datasets/gsm8k/think_normalized/valid.json"
    ]

    print(f"\n{'='*80}")
    print("Testing load_dataset_accum (accumulated, <think> style)")
    print(f"{'='*80}")
    processed_multi = load_dataset_accum(file_paths, tokenizer, Config)

    if processed_multi:
        print(f"\n=== Sample 0 ===")
        print(f"Inputs shape: {processed_multi[0]['inputs'].shape}")
        print(f"Labels shape: {processed_multi[0]['labels'].shape}")
        print(f"Task IDs shape: {processed_multi[0]['task_identifiers'].shape}")

        for turn_idx in range(Config.max_reasoning_steps):
            task_id = processed_multi[0]['task_identifiers'][turn_idx].item()
            if task_id == -1:
                print(f"\nTurn {turn_idx}: [PADDING]")
                break
            else:
                print(f"\nTurn {turn_idx} (Task ID: {task_id}):")
                print(f"  Input: {tokenizer.decode(processed_multi[0]['inputs'][turn_idx])}")
                print(f"  Label: {tokenizer.decode(processed_multi[0]['labels'][turn_idx])}")

        print(f"\n=== Sample 1 ===")
        print(f"Number of actual steps: {(processed_multi[1]['task_identifiers'] != -1).sum().item()}")
        for turn_idx in range(min(4, Config.max_reasoning_steps)):
            task_id = processed_multi[1]['task_identifiers'][turn_idx].item()
            if task_id == -1:
                print(f"\nTurn {turn_idx}: [PADDING]")
                break
            else:
                print(f"\nTurn {turn_idx} (Task ID: {task_id}):")
                print(f"  Input: {tokenizer.decode(processed_multi[1]['inputs'][turn_idx])}")
                print(f"  Label: {tokenizer.decode(processed_multi[1]['labels'][turn_idx])}")
    else:
        print("No samples processed")
