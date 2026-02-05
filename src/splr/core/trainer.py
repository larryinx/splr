from pathlib import Path
from typing import Dict, Any, List, Union, Optional, cast
from dataclasses import dataclass
import os
import time
import random
import json
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from transformers import AutoTokenizer
import wandb
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import pydantic

from .models import SPLRModel, SPLRConfig
from .models.modeling_splr import SPLRCarry
from .ema import EMAHelper

class ArchConfig(pydantic.BaseModel):
    """Architecture configuration."""
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class TrainConfig(pydantic.BaseModel):
    """Training configuration."""
    model_config = pydantic.ConfigDict(extra='allow')

    # Architecture
    arch: ArchConfig
    # Data
    dataset_name: str
    train_file: Union[str, List[str]]
    val_file: Union[None, str, List[str]] = None
    tokenizer_path: str
    vocab_size: Union[int, None] = None  # If specified, use this instead of len(tokenizer)

    # Input mode
    input_mode: str = "recurrent"  # "recurrent" or "autoregressive"

    # Training
    number_of_workers: int = 16
    output_dir: str = "./results/splr_model"
    load_checkpoint: Union[str, None] = None
    global_batch_size: int = 4
    num_epochs: int = 3
    lr: float = 5e-5
    lr_min_ratio: float = 1.0
    lr_warmup_steps: int = 1000
    task_emb_lr: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    enable_halt_loss: bool = True,
    q_halt_weight: float = 0.5,

    grad_clip_norm: float = 1.0

    log_interval: int = 10
    save_interval: int = 1000
    val_interval: int = 2000  # Validation (loss-based) every N steps
    eval_interval: int = 4000  # Benchmark evaluation (accuracy-based) every N steps
    checkpoint_every_eval: bool = True

    # Eval-only mode
    eval_only: bool = False

    # Benchmark evaluation configs (list of YAML file paths)
    eval_configs: List[str] = []

    # EMA
    ema: bool = False
    ema_rate: float = 0.999
    save_ema: bool = False  # Save EMA state dict in checkpoint
    load_checkpoint_from_ema: bool = False  # Load from EMA state when resuming

    # Distributed training
    use_ddp: bool = False
    local_rank: int = -1

    # Wandb
    project_name: str = None
    run_name: str = None

    # Misc
    seed: int = 0


@dataclass
class TrainState:
    """Training state."""
    model: nn.Module
    carry: SPLRCarry


def create_optimizers(config: TrainConfig, splr_config: SPLRConfig, model: SPLRModel, world_size: int):
    if not config.freeze_weights:
        if config.optimizer == "AdamW":
            from torch.optim import AdamW
            model_optimizer = cast(Any, AdamW)(
                model.parameters(),
                lr=0,
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        elif config.optimizer == "AdamATan2":
            try:
                from .optimizers import AdamATan2
                model_optimizer = cast(Any, AdamATan2)(
                    model.parameters(),
                    lr=0,  # Needs to be set by scheduler
                    weight_decay=config.weight_decay,
                    betas=(config.beta1, config.beta2)
                )
            except ImportError:
                raise RuntimeError(
                    "adam_atan2 package is required for training optimizers but was not found. "
                )
        else:
            raise ValueError(f"Invalid optimizer: {config.optimizer}")

        if splr_config.task_emb_ndim > 0:
            from .optimizers import CastedSparseEmbeddingSignSGD_Distributed
            optimizers = [
                CastedSparseEmbeddingSignSGD_Distributed(
                    model.task_emb.buffers(),
                    lr=0,
                    weight_decay=config.weight_decay,
                    world_size=world_size
                ),
                model_optimizer
            ]
            optimizer_lrs = [
                config.task_emb_lr,
                config.lr
            ]
        else:
            optimizers = [
                model_optimizer
            ]
            optimizer_lrs = [
                config.lr
            ]
    else:
        from .optimizers import CastedSparseEmbeddingSignSGD_Distributed
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.task_emb.buffers(),
                lr=0,
                weight_decay=config.weight_decay,
                world_size=world_size
            ),
        ]
        optimizer_lrs = [
            config.task_emb_lr
        ]

    return optimizers, optimizer_lrs


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    """Cosine learning rate schedule with warmup.

    Args:
        current_step: Current training step
        base_lr: Base learning rate
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        min_ratio: Minimum LR as fraction of base_lr
        num_cycles: Number of cosine cycles

    Returns:
        Current learning rate
    """
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))



class SPLRTrainer:
    """Trainer for SPLR with stateful carry and ACT-style halting."""

    def __init__(
        self,
        config: TrainConfig,
        model: SPLRModel,
        tokenizer: AutoTokenizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizers: List[torch.optim.Optimizer],
        optimizer_lrs: List[float],
        device: str = "cuda",
        rank: int = -1,
        world_size: int = 1,
        initial_global_step: int = 0,
        initial_epoch: int = 0,
        evaluator: Optional[Any] = None,
    ):
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == -1 or rank == 0)

        # Move model to device first
        self.model = model.to(device)

        # Wrap with DDP if using distributed training
        if rank != -1:
            self.model = DDP(
                self.model,
                device_ids=[rank],
                output_device=rank,
            )

        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizers = optimizers
        self.optimizer_lrs = optimizer_lrs

        self.global_step = initial_global_step
        self.epoch = initial_epoch
        self.carry = None  # Persistent carry state across epochs (per-GPU)

        # Benchmark evaluator (SPLREvaluator instance, or None)
        self.evaluator = evaluator

        # Initialize EMA if enabled
        self.ema_helper = None
        if config.ema:
            if self.is_main_process:
                print(f'Setting up EMA with rate={config.ema_rate}')
            self.ema_helper = EMAHelper(mu=config.ema_rate)
            self.ema_helper.register(self.get_model())

    def get_model(self):
        """Get the underlying model (unwrap DDP if needed)."""
        return self.model.module if isinstance(self.model, DDP) else self.model

    def aggregate_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Aggregate losses across all GPUs for logging.

        Args:
            metrics: Dict of metrics from current GPU

        Returns:
            aggregated_metrics: Dict of metrics averaged across all GPUs
        """
        if self.rank == -1:
            return metrics

        # return metrics

        aggregated = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                tensor = torch.tensor(value, device=self.device)
                dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
                aggregated[key] = tensor.item()
            else:
                # Skip non-numeric values (like loss_type string)
                aggregated[key] = value

        return aggregated

    def compute_loss(
        self,
        state: TrainState,
        batch: Dict[str, torch.Tensor],
    ):
        """Compute the loss for the model."""
        outputs = state.model(carry=state.carry, batch=batch)

        logits = outputs.logits
        q_halt_logits = outputs.q_halt_logits
        carry = outputs.carry
        new_current_data = carry.current_data

        loss_fct = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id,
            reduction='mean'
        )
        ce_loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            new_current_data["labels"].view(-1)
        )

        # Calculate halt loss based on prediction correctness
        if self.config.enable_halt_loss:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)  # (batch, seq_len)
                labels = new_current_data["labels"]  # (batch, seq_len)

                # Create mask for non-padded tokens
                mask = (labels != self.tokenizer.pad_token_id)  # (batch, seq_len)

                correct = (preds == labels) & mask  # (batch, seq_len)
                num_valid_tokens = mask.sum(dim=1).float()  # (batch,)
                num_correct = correct.sum(dim=1).float()  # (batch,)
                sample_accuracy = (num_correct == num_valid_tokens).float()  # (batch,) - exact match

                halt_target = sample_accuracy

            # BCEWithLogitsLoss with q_halt_logits
            halt_loss_fct = nn.BCEWithLogitsLoss(reduction='mean')
            q_halt_logits_squeezed = q_halt_logits.squeeze(-1) if q_halt_logits.dim() > 1 else q_halt_logits
            halt_loss = halt_loss_fct(q_halt_logits_squeezed, halt_target)

            # Combine losses
            loss = ce_loss + self.config.q_halt_weight * halt_loss
            valid_metrics = carry.halted & (num_valid_tokens > 0)

            if valid_metrics.sum() > 0:
                accuracy = sample_accuracy[valid_metrics].mean().item()
                token_accuracy = (num_correct[valid_metrics].sum() / (num_valid_tokens[valid_metrics].sum() + 1e-8)).item()
                q_halt_accuracy = ((q_halt_logits_squeezed > 0) == halt_target).float().mean().item()
            else:
                accuracy = 0.0
                token_accuracy = 0.0
                q_halt_accuracy = 1.0

            metrics = {
                "loss_type": "cross_entropy",
                "lm_loss": ce_loss.item(),
                "q_halt_loss": halt_loss.item(),
                "accuracy": accuracy,
                "token_accuracy": token_accuracy,
                "q_halt_accuracy": q_halt_accuracy,
                "steps": carry.steps[valid_metrics].float().mean().item(),
            }
        else:
            loss = ce_loss
            # FIXME: fix accuracy metrics
            metrics = {
                "loss_type": "cross_entropy",
                "lm_loss": ce_loss.item(),
                "q_halt_loss": 0.0,
                "accuracy": 0.0,
                "token_accuracy": 0.0,
                "q_halt_accuracy": 0.0,
                "steps": 0.0,
            }
        
        state.carry = carry

        return loss, metrics, state


    def train_batch(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Train on one batch with ACT-style halting.

        Args:
            batch: Dict with 'inputs' and 'labels'

        Returns:
            metrics: Dict of training metrics
        """
        # Move batch to device
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

        # Initialize carry if None
        if self.carry is None:
            self.carry = self.get_model().initial_carry(batch)

        state = TrainState(model=self.model, carry=self.carry)
        # Forward pass
        loss, outputs, state = self.compute_loss(state=state, batch=batch)
        self.carry = state.carry

        # Backward
        loss.backward()

        # Optimizer step
        lr_this_step = None
        for optim, base_lr in zip(self.optimizers, self.optimizer_lrs):
            lr_this_step = cosine_schedule_with_warmup_lr_lambda(
                self.global_step,
                base_lr=base_lr,
                num_warmup_steps=round(self.config.lr_warmup_steps),
                num_training_steps=self.config.num_epochs * len(self.train_dataloader),
                min_ratio=self.config.lr_min_ratio
            )

            for param_group in optim.param_groups:
                param_group['lr'] = lr_this_step

            optim.step()
            optim.zero_grad()

        # Update EMA if enabled
        if self.ema_helper is not None:
            self.ema_helper.update(self.get_model())

        self.global_step += 1

        # Metrics already computed in model forward
        metrics = {
            "loss": loss.item(),
            "loss_type": outputs.get("loss_type", "unknown"),
            "lm_loss": outputs.get("lm_loss", 0.0),
            "q_halt_loss": outputs.get("q_halt_loss", 0.0),
            "accuracy": outputs.get("accuracy", 0.0),
            "token_accuracy": outputs.get("token_accuracy", 0.0),
            "q_halt_accuracy": outputs.get("q_halt_accuracy", 0.0),
            "steps": outputs.get("steps", 0.0),
        }

        return metrics

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()

        epoch_metrics = {
            "loss": 0.0,
            "lm_loss": 0.0,
            "q_halt_loss": 0.0,
            "accuracy": 0.0,
            "token_accuracy": 0.0,
            "q_halt_accuracy": 0.0,
            "steps": 0.0,
        }
        num_batches = 0

        # Only show progress bar on main process
        if self.is_main_process:
            pbar = tqdm(total=len(self.train_dataloader), desc=f"Epoch {self.epoch}")
        else:
            pbar = None

        for batch_idx, batch in enumerate(self.train_dataloader):
            metrics = self.train_batch(batch)

            # Accumulate local metrics
            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key] += metrics[key]
            num_batches += 1

            # Aggregate metrics across GPUs (must be called by ALL processes)
            should_log = self.global_step % self.config.log_interval == 0
            if should_log:
                aggregated_metrics = self.aggregate_metrics(metrics)

            # Only log on main process
            if should_log and self.is_main_process:
                wandb.log({f"train/{k}": v for k, v in aggregated_metrics.items()}, step=self.global_step)

                # Update progress bar
                if pbar is not None:
                    pbar.set_postfix({
                        "loss": f"{aggregated_metrics['loss']:.4f}",
                        "lm_loss": f"{aggregated_metrics.get('lm_loss', 0.0):.4f}",
                        "q_halt_loss": f"{aggregated_metrics.get('q_halt_loss', 0.0):.4f}",
                        "accuracy": f"{aggregated_metrics.get('accuracy', 0.0):.4f}",
                        "token_accuracy": f"{aggregated_metrics.get('token_accuracy', 0.0):.4f}",
                        "q_halt_accuracy": f"{aggregated_metrics.get('q_halt_accuracy', 0.0):.4f}",
                    })
            elif pbar is not None:
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "lm_loss": f"{metrics.get('lm_loss', 0.0):.4f}",
                    "q_halt_loss": f"{metrics.get('q_halt_loss', 0.0):.4f}",
                    "accuracy": f"{metrics.get('accuracy', 0.0):.4f}",
                    "token_accuracy": f"{metrics.get('token_accuracy', 0.0):.4f}",
                    "q_halt_accuracy": f"{metrics.get('q_halt_accuracy', 0.0):.4f}",
                })

            # Update progress bar
            if pbar is not None:
                pbar.update(1)

            # Save checkpoint
            if self.is_main_process and self.global_step % self.config.save_interval == 0:
                self.save_checkpoint(f"step_{self.global_step}")

            # Step-based validation (loss-based)
            if self.val_dataloader is not None and self.global_step % self.config.val_interval == 0 and self.global_step > 0:
                val_metrics = self.validate()
                if self.is_main_process:
                    wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=self.global_step)
                # Resume training mode
                self.model.train()

            # Step-based benchmark evaluation (accuracy-based)
            if self.evaluator is not None and self.global_step % self.config.eval_interval == 0 and self.global_step > 0:
                self._run_benchmark_eval()
                # Resume training mode
                self.model.train()

        # Close progress bar
        if pbar is not None:
            pbar.close()

        # Average metrics across batches
        if num_batches > 0:
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches

        return epoch_metrics

    def _run_benchmark_eval(self):
        """Run benchmark evaluation using the evaluator with EMA model if available."""
        from .eval import EvalConfig, load_eval_config

        # Load eval configs
        eval_configs = []
        for config_path in self.config.eval_configs:
            try:
                eval_configs.append(load_eval_config(config_path))
            except Exception as e:
                if self.is_main_process:
                    print(f"Warning: Failed to load eval config {config_path}: {e}")

        if not eval_configs:
            return

        # Use EMA model for evaluation if available
        original_model = None
        if self.ema_helper is not None:
            original_model = self.model
            ema_model = self.ema_helper.ema_copy(self.get_model())
            self.model = ema_model
            # Update the evaluator's model reference
            self.evaluator.model = ema_model
            self.evaluator.sampler.model = ema_model

        if self.is_main_process:
            print(f"\nRunning benchmark evaluation at step {self.global_step}...")

        self.evaluator.evaluate(
            eval_configs=eval_configs,
            global_step=self.global_step,
            input_mode=self.config.input_mode,
        )

        # Restore original model
        if original_model is not None:
            self.model = original_model
            model = self.get_model()
            self.evaluator.model = model
            self.evaluator.sampler.model = model


    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set (loss-based, supports both single-step and multi-step models)."""
        # Use EMA model for validation if enabled
        original_model = None
        if self.ema_helper is not None:
            # Save original model
            original_model = self.model
            # Create EMA copy (no DDP wrapping — validation has no gradient
            # sync, and DDP buffer broadcasts would desync ranks when the
            # turn loop breaks early on data-dependent carry.global_halted)
            ema_model = self.ema_helper.ema_copy(self.get_model())
            self.model = ema_model

        self.model.eval()

        val_metrics = {
            "loss": 0.0,
            "lm_loss": 0.0,
            "q_halt_loss": 0.0,
            "accuracy": 0.0,
            "token_accuracy": 0.0,
            "q_halt_accuracy": 0.0,
            "steps": 0.0,
        }

        if self.is_main_process:
            pbar = tqdm(total=len(self.val_dataloader), desc=f"Validating step {self.global_step}")
        else:
            pbar = None

        # Track number of turns processed (for averaging)
        num_turns_processed = 0

        for batch_idx, batch in enumerate(self.val_dataloader):
            # Move to device
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            # Initialize carry
            carry = self.get_model().initial_carry(batch)

            # Check if this is a multi-step model (has global_halted attribute)
            is_multi_step = hasattr(carry, 'global_halted')

            if is_multi_step:
                # Multi-step evaluation: loop through turns
                max_reasoning_steps = self.get_model().config.max_reasoning_steps

                for turn_idx in range(max_reasoning_steps):
                    # Run inference steps until all samples halt locally for this turn
                    inference_steps = 0
                    while True:
                        outputs = self.model(carry=carry, batch=batch)
                        carry = outputs.carry
                        inference_steps += 1
                        carry.halted = torch.zeros_like(carry.halted, dtype=torch.bool)
                        carry.global_halted = torch.zeros_like(carry.global_halted, dtype=torch.bool)

                        if inference_steps == self.get_model().config.halt_max_steps - 1:
                            state = TrainState(model=self.model, carry=carry)
                            loss, outputs, state = self.compute_loss(state=state, batch=batch)
                            carry = state.carry
                            carry.halted = torch.ones_like(carry.halted, dtype=torch.bool)
                            break

                    # Accumulate metrics for this turn
                    for key in val_metrics:
                        if key == "loss":
                            val_metrics[key] += loss.item()
                        else:
                            val_metrics[key] += outputs.get(key, 0.0)
                    num_turns_processed += 1

                    if pbar is not None:
                        pbar.set_postfix({
                            "turn": f"{turn_idx}/{max_reasoning_steps}",
                            "loss": f"{loss.item():.4f}",
                            "accuracy": f"{outputs.get('accuracy', 0.0):.4f}",
                        })

                    # Check if all samples are globally halted
                    if carry.global_halted.all():
                        break

                if pbar is not None:
                    pbar.update(1)

        # Average across turns (for multi-step) or batches (for single-step)
        for key in val_metrics:
            val_metrics[key] /= max(num_turns_processed, 1)

        # Aggregate metrics across GPUs
        val_metrics = self.aggregate_metrics(val_metrics)

        if pbar is not None:
            pbar.close()

        # Restore original model if using EMA
        if original_model is not None:
            self.model = original_model

        return val_metrics

    def train(self, num_epochs: int):
        """Full training loop."""
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch

            # Set epoch for DistributedSampler (ensures different shuffle each epoch)
            if self.rank != -1 and hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(epoch)

            # Train epoch
            train_metrics = self.train_epoch()

            # Log epoch-level training metrics (only on main process)
            if self.is_main_process:
                wandb.log({f"train_epoch/{k}": v for k, v in train_metrics.items()}, step=self.global_step)

    def save_checkpoint(self, name: str):
        """Save model checkpoint (only on main process)."""
        if not self.is_main_process:
            return

        checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{name}.pt"

        # Unwrap DDP model before saving
        model_to_save = self.get_model()

        # Prepare checkpoint dict with regular model state
        checkpoint_dict = {
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": [opt.state_dict() for opt in self.optimizers],
            "global_step": self.global_step,
            "epoch": self.epoch,
        }

        # Add EMA state if enabled
        if self.config.save_ema and self.ema_helper is not None:
            ema_model = self.ema_helper.ema_copy(model_to_save)
            checkpoint_dict["ema_state_dict"] = ema_model.state_dict()

        torch.save(checkpoint_dict, checkpoint_path)

        if self.config.save_ema and self.ema_helper is not None:
            print(f"Checkpoint saved (with EMA): {checkpoint_path}")
        else:
            print(f"Checkpoint saved: {checkpoint_path}")
