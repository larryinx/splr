

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from ..models import SPLRModel, SPLRConfig
from ..models.modeling_splr import SPLRCarry
from transformers import PreTrainedTokenizer


@dataclass
class SPLRSamplerOutput:
    output_ids: torch.Tensor
    generate_carry: "SPLRCarry | None" = None


class SPLRSampler():
    def __init__(
        self,
        model: SPLRModel,
        tokenizer: PreTrainedTokenizer,
        config: SPLRConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config


    @torch.no_grad()
    def sample(
        self,
        input_ids: torch.Tensor,
        task_identifiers: torch.Tensor = None,
        generate_carry: SPLRCarry = None,
        enable_halt: Optional[bool] = None,
        halt_max_steps: Optional[int] = None,
        max_reasoning_steps: Optional[int] = None,
        **kwargs,
    ):
        self.model.eval()

        # Use runtime overrides or fall back to config defaults
        _enable_halt = enable_halt if enable_halt is not None else getattr(self.config, 'enable_halt', True)
        _halt_max_steps = halt_max_steps if halt_max_steps is not None else self.config.halt_max_steps
        _max_reasoning_steps = max_reasoning_steps if max_reasoning_steps is not None else self.config.max_reasoning_steps
        _pad_token_id = self.config.pad_token_id
        _seq_len = self.config.max_position_embeddings

        # Input should be (batch, input_max_len) for current turn
        assert input_ids.dim() == 2, f"input_ids should be (batch, input_max_len), got shape {input_ids.shape}"
        batch_size, input_max_len = input_ids.shape
        device = input_ids.device

        # Handle task_identifiers
        if task_identifiers is None:
            task_identifiers = torch.zeros(batch_size, dtype=torch.long, device=device)
        assert task_identifiers.dim() == 1, f"task_identifiers should be (batch,), got shape {task_identifiers.shape}"

        # Initialize or update carry
        if generate_carry is None or generate_carry.global_halted.all():
            # First call: Create multi_step_data with all turns (mostly padding)
            # inputs: (batch, max_turns, input_max_len)
            # labels: (batch, max_turns, seq_len) - full length for task embeddings
            multi_turn_input_ids = torch.full(
                (batch_size, _max_reasoning_steps, input_max_len),
                _pad_token_id,
                dtype=input_ids.dtype,
                device=device
            )
            multi_turn_input_ids[:, 0, :] = input_ids

            # Labels should be full seq_len (will be left-padded)
            multi_turn_label_ids = torch.full(
                (batch_size, _max_reasoning_steps, _seq_len),
                _pad_token_id,
                dtype=input_ids.dtype,
                device=device
            )
            # For generation, labels are dummy - just use padded version of inputs
            # Left-pad the input_ids to seq_len
            multi_turn_label_ids[:, 0, -input_max_len:] = input_ids

            multi_turn_task_ids = torch.full(
                (batch_size, _max_reasoning_steps),
                -1,
                dtype=torch.long,
                device=device
            )
            multi_turn_task_ids[:, 0] = task_identifiers

            batch = {
                "inputs": multi_turn_input_ids,
                "labels": multi_turn_label_ids,
                "task_identifiers": multi_turn_task_ids
            }

            carry = self.model.initial_carry(batch)
        else:
            carry = generate_carry

            # Get current turn index for each sample
            current_turns = carry.global_steps

            # Update multi_step_data with new inputs at current turn
            for i in range(batch_size):
                turn_idx = current_turns[i].item()
                if turn_idx < _max_reasoning_steps:
                    # Update inputs
                    carry.multi_step_data["inputs"][i, turn_idx, :] = input_ids[i]

                    # Update labels (left-pad to seq_len)
                    carry.multi_step_data["labels"][i, turn_idx, :] = _pad_token_id
                    carry.multi_step_data["labels"][i, turn_idx, -input_max_len:] = input_ids[i]

                    # Update task_identifiers
                    carry.multi_step_data["task_identifiers"][i, turn_idx] = task_identifiers[i]

            batch = carry.multi_step_data

        output_z_H = carry.inner_carry.z_H.clone()
        output_updated = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(_halt_max_steps):
            batch_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward
            outputs = self.model(carry=carry, batch=batch_on_device)
            carry = outputs.carry

            # Extract z_H and halt from carry
            z_H = carry.inner_carry.z_H

            if _enable_halt:
                halt = carry.halted
                steps = carry.steps

                is_last_step = steps >= _halt_max_steps
                halt = halt | is_last_step

                # Update output_z_H where not updated yet and halt is True
                update_mask = (~output_updated) & halt
                output_z_H = torch.where(
                    update_mask.view(-1, 1, 1),
                    z_H,
                    output_z_H
                )
                output_updated = output_updated | halt
                carry.halted = torch.zeros(batch_size, dtype=torch.bool, device=device)
            else:
                output_z_H = z_H
                carry.halted = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # After halt_max_steps, set halted to True and increment global_steps
        carry.halted = torch.ones(batch_size, dtype=torch.bool, device=device)
        carry.global_steps = carry.global_steps + 1

        # Check if we've exceeded max_turns
        global_halted = carry.global_steps >= _max_reasoning_steps
        carry.global_halted = global_halted

        z_H = output_z_H

        # Slice out task embeddings if present
        task_emb_len = self.model.inner.task_emb_len if hasattr(self.model.inner, 'task_emb_len') else 0

        z_H = z_H[:, task_emb_len:] if task_emb_len > 0 else z_H

        logits = self.model.output_head(z_H)
        output_ids = logits.argmax(dim=-1)

        # Reset generate_carry to None if all samples have completed
        if carry.global_halted.all():
            generate_carry = None
        else:
            generate_carry = carry

        return SPLRSamplerOutput(
            output_ids=output_ids,
            generate_carry=generate_carry
        )
