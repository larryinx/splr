from pathlib import Path
from typing import Dict, Any, List, Union, cast
from dataclasses import dataclass
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from transformers import AutoTokenizer
import wandb
import hydra
from omegaconf import DictConfig

from splr.data import load_dataset
from splr.core.trainer import SPLRTrainer, ArchConfig, TrainConfig, create_optimizers
from splr.core.models import SPLRModel, SPLRConfig



def collate_fn(batch):
    """Collate function for DataLoader."""
    inputs = torch.stack([item["inputs"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    # Handle task_identifiers if present
    if "task_identifiers" in batch[0]:
        task_identifiers = torch.stack([item["task_identifiers"] for item in batch])
        return {
            "inputs": inputs,
            "labels": labels,
            "task_identifiers": task_identifiers
        }

    return {
        "inputs": inputs,
        "labels": labels
    }

def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = -1
        world_size = -1
        local_rank = -1

    if rank != -1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        print(f"Initialized DDP: rank={rank}, world_size={world_size}, local_rank={local_rank}")

    return rank, world_size, local_rank

def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def create_evaluator(model, tokenizer, splr_config, device, rank, world_size):
    """Create an SPLREvaluator instance."""
    from splr.core.eval import SPLREvaluator
    return SPLREvaluator(
        model=model,
        tokenizer=tokenizer,
        model_config=splr_config,
        device=device,
        rank=rank,
        world_size=world_size,
    )


@hydra.main(config_path="./configs", config_name="cfg_train_splr", version_base=None)
def main(hydra_config: DictConfig):
    """Main training function with Hydra config."""

    # Parse config
    config = TrainConfig(**hydra_config)

    # Initialize distributed training if enabled
    rank, world_size, local_rank = -1, 1, -1
    if config.use_ddp:
        rank, world_size, local_rank = setup_distributed()
        config.local_rank = local_rank

    is_main_process = (rank == -1 or rank == 0)

    # Set seed (with rank offset for reproducibility)
    seed = config.seed + rank if rank != -1 else config.seed
    torch.random.manual_seed(seed)

    # Initialize wandb
    if is_main_process:
        if config.project_name is None:
            config.project_name = "splr"
        if config.run_name is None:
            config.run_name = f"splr_{config.arch.name}"

        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=config.model_dump()
        )

    # Load tokenizer
    if is_main_process:
        print(f"Loading tokenizer from tokenizer_path: {config.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)

    # Create model config from arch config
    # Use config.vocab_size if specified, otherwise use tokenizer vocab_size
    vocab_size = config.vocab_size if config.vocab_size is not None else len(tokenizer)

    model_config_dict = dict(
        **config.arch.__pydantic_extra__,
        vocab_size=vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        batch_size=config.global_batch_size // world_size,
    )


    # Determine device first
    if config.use_ddp:
        device = f"cuda:{local_rank}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model on the target device
    if is_main_process:
        print("Initializing SPLR model...")

    with torch.device(device):
        model = SPLRModel(SPLRConfig(**model_config_dict))

    if is_main_process:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(model)

    # Load checkpoint if specified
    initial_global_step = 0
    initial_epoch = 0
    checkpoint_optimizer_states = None
    if config.load_checkpoint is not None and config.load_checkpoint.strip() != "":
        if is_main_process:
            print(f"\nLoading checkpoint from {config.load_checkpoint}")

        checkpoint = torch.load(config.load_checkpoint, map_location=device)

        # Load model state dict (from EMA or regular)
        if config.load_checkpoint_from_ema and "ema_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["ema_state_dict"])
            if is_main_process:
                print("Loaded model from EMA state dict")
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
            if is_main_process and config.load_checkpoint_from_ema:
                print("Warning: load_checkpoint_from_ema=True but no ema_state_dict found, loading from regular model_state_dict")

        # Get training state
        initial_global_step = checkpoint.get("global_step", 0)
        initial_epoch = checkpoint.get("epoch", 0)
        checkpoint_optimizer_states = checkpoint.get("optimizer_state_dict", None)

        if is_main_process:
            print(f"Resuming from step {initial_global_step}, epoch {initial_epoch}")

    # Broadcast parameters from rank 0 to ensure all ranks start with same weights
    if config.use_ddp:
        with torch.no_grad():
            for param in list(model.parameters()) + list(model.buffers()):
                dist.broadcast(param, src=0)

    splr_config = SPLRConfig(**model_config_dict)

    # --- Eval-only mode ---
    if config.eval_only:
        if is_main_process:
            print("\n=== EVAL-ONLY MODE ===")

        evaluator = create_evaluator(model, tokenizer, splr_config, device, rank, world_size)

        # Load eval configs
        from splr.core.eval import load_eval_config
        eval_configs = []
        for config_path in config.eval_configs:
            try:
                eval_configs.append(load_eval_config(config_path))
            except Exception as e:
                if is_main_process:
                    print(f"Warning: Failed to load eval config {config_path}: {e}")

        if not eval_configs:
            if is_main_process:
                print("No eval configs loaded. Exiting.")
        else:
            # Resolve eval_results_dir
            save_eval_results = getattr(config, 'save_eval_results', False)
            eval_results_dir = getattr(config, 'eval_results_dir', None)
            if eval_results_dir is None and save_eval_results:
                eval_results_dir = str(Path(config.output_dir) / "eval_results") if hasattr(config, 'output_dir') else "./results/eval_results"

            evaluator.evaluate(
                eval_configs=eval_configs,
                global_step=initial_global_step,
                input_mode=config.input_mode,
                save_eval_results=save_eval_results,
                eval_results_dir=eval_results_dir,
            )

        if is_main_process:
            print("\nEvaluation completed!")
            wandb.finish()

        if config.use_ddp:
            cleanup_distributed()
        return

    # --- Training mode ---

    # Load datasets
    # Convert train_file to list if it's a string
    train_files = config.train_file if isinstance(config.train_file, list) else [config.train_file]
    train_dataset = load_dataset(config.dataset_name, train_files, tokenizer, splr_config, input_mode=config.input_mode)

    val_dataset = None
    if config.val_file:
        # Convert val_file to list if it's a string
        val_files = config.val_file if isinstance(config.val_file, list) else [config.val_file]
        val_dataset = load_dataset(config.dataset_name, val_files, tokenizer, splr_config, input_mode=config.input_mode)

    # Create dataloaders with DistributedSampler if using DDP
    if config.use_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config.seed
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        ) if val_dataset else None
    else:
        train_sampler = None
        val_sampler = None

    if config.use_ddp:
        num_workers = max(1, config.number_of_workers // world_size)
        per_gpu_batch_size = config.global_batch_size // world_size
    else:
        num_workers = config.number_of_workers
        per_gpu_batch_size = config.global_batch_size

    prefetch_factor = 8 if num_workers > 0 else None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=per_gpu_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=prefetch_factor,
        drop_last=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=per_gpu_batch_size,
        sampler=val_sampler,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=prefetch_factor,
        drop_last=False
    ) if val_dataset else None

    # Create optimizers
    optimizers, optimizer_lrs = create_optimizers(config, splr_config, model, world_size)

    # Load optimizer states if checkpoint was loaded
    if checkpoint_optimizer_states is not None:
        if is_main_process:
            print("Loading optimizer states from checkpoint")

        # Update world_size in optimizer state if it differs from current world_size
        for i, optimizer_state in enumerate(checkpoint_optimizer_states):
            # Check if this optimizer has world_size in its param_groups
            for param_group in optimizer_state.get('param_groups', []):
                if 'world_size' in param_group:
                    checkpoint_world_size = param_group['world_size']
                    if checkpoint_world_size != world_size:
                        if is_main_process:
                            print(f"  Updating optimizer {i} world_size from {checkpoint_world_size} to {world_size}")
                        param_group['world_size'] = world_size

            optimizers[i].load_state_dict(optimizer_state)

    # Create benchmark evaluator if eval_configs are specified
    evaluator = None
    if config.eval_configs:
        evaluator = create_evaluator(model, tokenizer, splr_config, device, rank, world_size)

    # Create trainer
    trainer = SPLRTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        device=device,
        rank=rank,
        world_size=world_size,
        initial_global_step=initial_global_step,
        initial_epoch=initial_epoch,
        evaluator=evaluator,
    )

    # Train
    if is_main_process:
        print(f"\nStarting training for {config.num_epochs} epochs...")
    trainer.train(num_epochs=config.num_epochs)

    if is_main_process:
        print("\nTraining completed!")
        wandb.finish()

    # Cleanup distributed training
    if config.use_ddp:
        cleanup_distributed()


if __name__ == "__main__":
    main()
