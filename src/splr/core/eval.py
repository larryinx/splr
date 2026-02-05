import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pydantic
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizer
from tqdm import tqdm
import wandb

from .models import SPLRModel, SPLRConfig
from .samplers.splr import SPLRSampler
from splr.tasks.tool import (
    postprocess_tool_output,
    validate_and_compute_formula,
    compare_answers as compare_answers_tool,
    extract_last_tool_response,
)
from splr.tasks.think import (
    postprocess_think_output,
    compare_answers as compare_answers_think,
    extract_last_think_response,
)


class EvalConfig(pydantic.BaseModel):
    """Configuration for a single evaluation run."""
    model_config = pydantic.ConfigDict(extra='allow')

    name: str
    enable_halt: bool = True
    halt_max_steps: int = 16
    max_reasoning_steps: Optional[int] = None
    eval_datasets: List[str] = []
    eval_mode: str = "tool"  # "tool" or "think"
    batch_size: int = 512


def load_eval_config(path: str) -> EvalConfig:
    """Load an EvalConfig from a YAML file."""
    import yaml
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return EvalConfig(**data)


def preprocess_input(
    questions: List[str],
    tokenizer: PreTrainedTokenizer,
    input_max_len: int,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Tokenize questions with right padding.

    Args:
        questions: List of question strings
        tokenizer: Tokenizer instance
        input_max_len: Maximum input length (excluding task embeddings)

    Returns:
        Tuple of (input_ids, actual_lengths).
        input_ids: (batch, input_max_len) tensor, right-padded.
        actual_lengths: List of actual token lengths per sample.
    """
    batch_size = len(questions)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    input_ids = torch.full((batch_size, input_max_len), pad_token_id, dtype=torch.long)
    actual_lengths = []

    for i, q in enumerate(questions):
        tokens = tokenizer.encode(q, add_special_tokens=False, truncation=True, max_length=input_max_len)
        length = len(tokens)
        input_ids[i, :length] = torch.tensor(tokens, dtype=torch.long)
        actual_lengths.append(length)

    return input_ids, actual_lengths


def build_next_input(
    questions: List[str],
    accumulated_texts: List[str],
    formulas: List[Optional[str]],
    results: List[Optional[float]],
    stopped: List[bool],
    input_mode: str,
    tokenizer: PreTrainedTokenizer,
    max_len: int,
) -> Tuple[torch.Tensor, List[int], List[str]]:
    """
    Build the next turn's input for each sample.

    For samples that are stopped, the input is unchanged (padded).

    Args:
        questions: Original question strings
        accumulated_texts: Current accumulated conversation texts per sample
        formulas: Formula text extracted from this turn (or None)
        results: Computed results from this turn (or None)
        stopped: Whether each sample has stopped
        input_mode: "recurrent" or "autoregressive"
        tokenizer: Tokenizer instance
        max_len: Maximum input length

    Returns:
        Tuple of (input_ids, actual_lengths, new_accumulated_texts).
    """
    new_texts = []
    for i in range(len(questions)):
        if stopped[i] or formulas[i] is None:
            new_texts.append(accumulated_texts[i])
            continue

        formula = formulas[i]
        result = results[i]
        result_str = _format_result(result) if result is not None else "error"

        tool_suffix = f"<tool_call>{formula}</tool_call><tool_response>{result_str}</tool_response>"

        if input_mode == "recurrent":
            # recurrent: question + latest tool_call/response
            new_text = f"{questions[i]}{tool_suffix}"
        else:
            # autoregressive: accumulated + tool_call/response
            new_text = f"{accumulated_texts[i]}{tool_suffix}"

        new_texts.append(new_text)

    input_ids, actual_lengths = preprocess_input(new_texts, tokenizer, max_len)
    return input_ids, actual_lengths, new_texts


def _format_result(value: float) -> str:
    """Format a float result for tool_response text."""
    if value == int(value):
        return str(int(value))
    return f"{value:.10f}".rstrip('0').rstrip('.')


def build_next_think_input(
    questions: List[str],
    accumulated_texts: List[str],
    think_contents: List[Optional[str]],
    stopped: List[bool],
    input_mode: str,
    tokenizer: PreTrainedTokenizer,
    max_len: int,
) -> Tuple[torch.Tensor, List[int], List[str]]:
    """
    Build the next turn's input for think-mode evaluation.

    Think mode uses <think>content</think> tags where content is the
    full equation (e.g. "2+3=5").

    Args:
        questions: Original question strings
        accumulated_texts: Current accumulated conversation texts per sample
        think_contents: Think content extracted from this turn (or None)
        stopped: Whether each sample has stopped
        input_mode: "recurrent" or "autoregressive"
        tokenizer: Tokenizer instance
        max_len: Maximum input length

    Returns:
        Tuple of (input_ids, actual_lengths, new_accumulated_texts).
    """
    new_texts = []
    for i in range(len(questions)):
        if stopped[i] or think_contents[i] is None:
            new_texts.append(accumulated_texts[i])
            continue

        think_suffix = f"<think>{think_contents[i]}</think>"

        if input_mode == "recurrent":
            new_text = f"{questions[i]}{think_suffix}"
        else:
            new_text = f"{accumulated_texts[i]}{think_suffix}"

        new_texts.append(new_text)

    input_ids, actual_lengths = preprocess_input(new_texts, tokenizer, max_len)
    return input_ids, actual_lengths, new_texts


class SPLREvaluator:
    """
    Benchmark evaluator for SPLR models.

    Runs multi-step tool-based evaluation using the SPLRSampler,
    with support for distributed evaluation across GPUs.
    """

    def __init__(
        self,
        model: SPLRModel,
        tokenizer: PreTrainedTokenizer,
        model_config: SPLRConfig,
        device: str,
        rank: int = -1,
        world_size: int = 1,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == -1 or rank == 0)

        # Create sampler
        self.sampler = SPLRSampler(
            model=model,
            tokenizer=tokenizer,
            config=model_config,
        )

        # Calculate input_max_len (same formula as data loading)
        task_emb_ndim = getattr(model_config, 'task_emb_ndim', 0)
        task_emb_len_cfg = getattr(model_config, 'task_emb_len', 0)
        if task_emb_ndim > 0:
            task_emb_len = -(task_emb_ndim // -model_config.hidden_size) if task_emb_len_cfg == 0 else task_emb_len_cfg
            self.input_max_len = model_config.max_position_embeddings - task_emb_len
        else:
            self.input_max_len = model_config.max_position_embeddings

    def evaluate(
        self,
        eval_configs: List[EvalConfig],
        global_step: int,
        input_mode: str = "recurrent",
    ) -> Dict:
        """
        Run all evaluation configs and log results.

        Args:
            eval_configs: List of EvalConfig objects
            global_step: Current training step (for logging)
            input_mode: "recurrent" or "autoregressive"

        Returns:
            Dict of all evaluation metrics
        """
        # Barrier at start of evaluation to ensure all ranks enter together
        if self.rank != -1:
            dist.barrier()

        all_metrics = {}
        for eval_config in eval_configs:
            if eval_config.eval_mode not in ("tool", "think"):
                if self.is_main_process:
                    print(f"Skipping eval config '{eval_config.name}': mode '{eval_config.eval_mode}' not implemented")
                continue

            config_metrics = self.evaluate_config(eval_config, global_step, input_mode)
            all_metrics[eval_config.name] = config_metrics

        # Barrier at end of evaluation to synchronize before returning to training
        if self.rank != -1:
            dist.barrier()

        return all_metrics

    def evaluate_config(
        self,
        eval_config: EvalConfig,
        global_step: int,
        input_mode: str = "recurrent",
    ) -> Dict:
        """
        Run one eval config on all its datasets.

        Args:
            eval_config: EvalConfig object
            global_step: Current training step
            input_mode: "recurrent" or "autoregressive"

        Returns:
            Dict of metrics per dataset
        """
        config_metrics = {}
        prefix = f"eval_{eval_config.name}"

        for dataset_path in eval_config.eval_datasets:
            # Synchronize file existence check across all ranks to avoid NCCL desync
            # on distributed filesystems (NFS) where os.path.exists() can return
            # different results due to metadata caching inconsistencies
            if self.rank != -1:
                # Rank 0 checks file existence and broadcasts to all ranks
                exists_tensor = torch.tensor(
                    [1 if os.path.exists(dataset_path) else 0],
                    dtype=torch.int32,
                    device=self.device
                )
                dist.broadcast(exists_tensor, src=0)
                file_exists = exists_tensor.item() == 1
                
                # Barrier to ensure all ranks are synchronized before proceeding
                dist.barrier()
            else:
                file_exists = os.path.exists(dataset_path)

            if not file_exists:
                if self.is_main_process:
                    print(f"  Skipping missing dataset: {dataset_path}")
                continue

            dataset_name = Path(dataset_path).stem
            metrics = self.evaluate_dataset(dataset_path, eval_config, input_mode)
            config_metrics[dataset_name] = metrics

            # Log to wandb
            if self.is_main_process:
                log_dict = {
                    f"{prefix}/{dataset_name}/accuracy": metrics["accuracy"],
                    f"{prefix}/{dataset_name}/correct": metrics["correct"],
                    f"{prefix}/{dataset_name}/total": metrics["total"],
                }
                wandb.log(log_dict, step=global_step)
                print(f"  {prefix}/{dataset_name}: accuracy={metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")

        return config_metrics

    def evaluate_dataset(
        self,
        dataset_path: str,
        eval_config: EvalConfig,
        input_mode: str = "recurrent",
    ) -> Dict[str, float]:
        """
        Evaluate one dataset.

        Loads JSON, batches samples, runs multi-step eval, gathers
        metrics across GPUs.

        Args:
            dataset_path: Path to the dataset JSON file
            eval_config: EvalConfig object
            input_mode: "recurrent" or "autoregressive"

        Returns:
            Dict with accuracy, total, correct
        """
        # Load dataset
        with open(dataset_path, 'r') as f:
            data = json.load(f)

        questions = [sample["question"] for sample in data]
        # Extract expected answers — handle different dataset formats
        answers = []
        for sample in data:
            if "answer" in sample:
                answers.append(str(sample["answer"]))
            elif "steps" in sample and len(sample["steps"]) > 0:
                # Try to extract answer from last step
                import re
                last_step = sample["steps"][-1]
                match = re.search(r'=(.+?)>>', last_step)
                if match:
                    answers.append(match.group(1).strip())
                else:
                    answers.append("")
            else:
                answers.append("")

        # Create indices and use DistributedSampler if multi-GPU
        indices = list(range(len(questions)))

        if self.rank != -1:
            sampler = DistributedSampler(
                indices,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
            )
            local_indices = list(sampler)
        else:
            local_indices = indices

        # Batch evaluation
        batch_size = eval_config.batch_size
        correct_count = 0
        total_count = 0

        self.model.eval()

        for batch_start in range(0, len(local_indices), batch_size):
            batch_indices = local_indices[batch_start:batch_start + batch_size]
            batch_questions = [questions[i] for i in batch_indices]
            batch_answers = [answers[i] for i in batch_indices]

            if eval_config.eval_mode == "think":
                batch_correct, batch_total, _ = self.evaluate_batch_think(
                    batch_questions,
                    batch_answers,
                    eval_config,
                    input_mode,
                )
            else:
                batch_correct, batch_total, _ = self.evaluate_batch(
                    batch_questions,
                    batch_answers,
                    eval_config,
                    input_mode,
                )
            correct_count += batch_correct
            total_count += batch_total

        # Gather counts across GPUs
        if self.rank != -1:
            correct_tensor = torch.tensor(correct_count, dtype=torch.float32, device=self.device)
            total_tensor = torch.tensor(total_count, dtype=torch.float32, device=self.device)
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
            correct_count = int(correct_tensor.item())
            total_count = int(total_tensor.item())

        accuracy = correct_count / max(total_count, 1)
        return {
            "accuracy": accuracy,
            "total": total_count,
            "correct": correct_count,
        }

    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        eval_config: EvalConfig,
        input_mode: str = "recurrent",
    ) -> Tuple[int, int, List[Dict]]:
        """
        Evaluate one batch of questions.

        Multi-step loop:
        1. Preprocess input
        2. Run sampler
        3. Postprocess output (extract tool_call, compute formula)
        4. Build next input with tool_response
        5. Repeat until EOS or max_reasoning_steps

        Args:
            questions: List of question strings
            answers: List of expected answer strings
            eval_config: EvalConfig object
            input_mode: "recurrent" or "autoregressive"

        Returns:
            Tuple of (correct_count, total_count, results_list)
        """
        batch_size = len(questions)
        device = self.device

        # Initialize
        accumulated_texts = list(questions)
        stopped = [False] * batch_size
        final_answers = [None] * batch_size
        generate_carry = None

        # Resolve max_reasoning_steps: eval config override > model config
        max_reasoning_steps = eval_config.max_reasoning_steps if eval_config.max_reasoning_steps is not None else self.model_config.max_reasoning_steps

        # Preprocess initial input
        input_ids, actual_lengths = preprocess_input(
            questions, self.tokenizer, self.input_max_len
        )
        input_ids = input_ids.to(device)

        for step_idx in range(max_reasoning_steps):
            # Run sampler
            sampler_output = self.sampler.sample(
                input_ids=input_ids,
                generate_carry=generate_carry,
                enable_halt=eval_config.enable_halt,
                halt_max_steps=eval_config.halt_max_steps,
                max_reasoning_steps=max_reasoning_steps,
            )

            output_ids = sampler_output.output_ids
            generate_carry = sampler_output.generate_carry

            # Postprocess: extract tool calls and compute results
            step_results = postprocess_tool_output(
                output_ids, actual_lengths, self.tokenizer
            )

            # Update state for each sample
            formulas = []
            results = []
            for i, (formula_text, computed_result, is_done) in enumerate(step_results):
                if stopped[i]:
                    formulas.append(None)
                    results.append(None)
                    continue

                if is_done:
                    stopped[i] = True
                    # Extract final answer from accumulated text
                    # last_response = extract_last_tool_response(accumulated_texts[i])
                    # if last_response is not None:
                    #     final_answers[i] = last_response
                    final_answers[i] = computed_result
                    formulas.append(formula_text)
                    results.append(computed_result)
                elif computed_result is not None:
                    formulas.append(formula_text)
                    results.append(computed_result)
                    # Track the latest result as potential final answer
                    final_answers[i] = _format_result(computed_result)
                else:
                    # Formula was invalid — stop this sample
                    # stopped[i] = True
                    # last_response = extract_last_tool_response(accumulated_texts[i])
                    # if last_response is not None:
                    #     final_answers[i] = last_response
                    formulas.append(formula_text)
                    results.append(None)

            # Check if all stopped
            if all(stopped):
                break

            # Check if generate_carry is None (all globally halted)
            if generate_carry is None:
                # All samples completed their reasoning steps
                for i in range(batch_size):
                    if not stopped[i]:
                        stopped[i] = True
                        last_response = extract_last_tool_response(accumulated_texts[i])
                        if last_response is not None:
                            final_answers[i] = last_response
                break

            # Build next input
            input_ids, actual_lengths, accumulated_texts = build_next_input(
                questions=questions,
                accumulated_texts=accumulated_texts,
                formulas=formulas,
                results=results,
                stopped=stopped,
                input_mode=input_mode,
                tokenizer=self.tokenizer,
                max_len=self.input_max_len,
            )
            input_ids = input_ids.to(device)

        # Compare final answers
        correct_count = 0
        results_list = []
        for i in range(batch_size):
            is_correct = compare_answers_tool(final_answers[i], answers[i])
            if is_correct:
                correct_count += 1
            results_list.append({
                "question": questions[i],
                "expected": answers[i],
                "predicted": final_answers[i],
                "correct": is_correct,
            })

        return correct_count, batch_size, results_list

    def evaluate_batch_think(
        self,
        questions: List[str],
        answers: List[str],
        eval_config: EvalConfig,
        input_mode: str = "recurrent",
    ) -> Tuple[int, int, List[Dict]]:
        """
        Evaluate one batch of questions in think mode.

        Think mode: model outputs <think>equation</think> where equation
        contains both formula and result (e.g. "2+3=5"). The result is
        extracted from after the = sign and used as the final answer.

        Multi-step loop:
        1. Preprocess input
        2. Run sampler
        3. Postprocess output (extract <think> content)
        4. Build next input with <think>content</think>
        5. Repeat until EOS or max_reasoning_steps

        Args:
            questions: List of question strings
            answers: List of expected answer strings
            eval_config: EvalConfig object
            input_mode: "recurrent" or "autoregressive"

        Returns:
            Tuple of (correct_count, total_count, results_list)
        """
        batch_size = len(questions)
        device = self.device

        # Initialize
        accumulated_texts = list(questions)
        stopped = [False] * batch_size
        final_answers = [None] * batch_size
        generate_carry = None

        # Resolve max_reasoning_steps: eval config override > model config
        max_reasoning_steps = eval_config.max_reasoning_steps if eval_config.max_reasoning_steps is not None else self.model_config.max_reasoning_steps

        # Preprocess initial input
        input_ids, actual_lengths = preprocess_input(
            questions, self.tokenizer, self.input_max_len
        )
        input_ids = input_ids.to(device)

        for step_idx in range(max_reasoning_steps):
            # Run sampler
            sampler_output = self.sampler.sample(
                input_ids=input_ids,
                generate_carry=generate_carry,
                enable_halt=eval_config.enable_halt,
                halt_max_steps=eval_config.halt_max_steps,
                max_reasoning_steps=max_reasoning_steps,
            )

            output_ids = sampler_output.output_ids
            generate_carry = sampler_output.generate_carry

            # Postprocess: extract <think> content
            step_results = postprocess_think_output(
                output_ids, actual_lengths, self.tokenizer
            )

            # Update state for each sample
            think_contents = []
            for i, (think_content, result_str, is_done) in enumerate(step_results):
                if stopped[i]:
                    think_contents.append(None)
                    continue

                if is_done:
                    stopped[i] = True
                    final_answers[i] = result_str
                    think_contents.append(think_content)
                elif think_content is not None:
                    think_contents.append(think_content)
                    # Track the latest result as potential final answer
                    final_answers[i] = result_str
                else:
                    # No think content found
                    think_contents.append(None)

            # Check if all stopped
            if all(stopped):
                break

            # Check if generate_carry is None (all globally halted)
            if generate_carry is None:
                for i in range(batch_size):
                    if not stopped[i]:
                        stopped[i] = True
                        last_response = extract_last_think_response(accumulated_texts[i])
                        if last_response is not None:
                            final_answers[i] = last_response
                break

            # Build next input
            input_ids, actual_lengths, accumulated_texts = build_next_think_input(
                questions=questions,
                accumulated_texts=accumulated_texts,
                think_contents=think_contents,
                stopped=stopped,
                input_mode=input_mode,
                tokenizer=self.tokenizer,
                max_len=self.input_max_len,
            )
            input_ids = input_ids.to(device)

        # Compare final answers
        correct_count = 0
        results_list = []
        for i in range(batch_size):
            is_correct = compare_answers_think(final_answers[i], answers[i])
            if is_correct:
                correct_count += 1
            results_list.append({
                "question": questions[i],
                "expected": answers[i],
                "predicted": final_answers[i],
                "correct": is_correct,
            })

        return correct_count, batch_size, results_list
