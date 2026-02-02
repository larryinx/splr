from typing import Dict, Tuple, Union, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from .configuration_splr import SPLRConfig


@dataclass
class SPLRInnerCarry:
    """Inner carry state for recursive reasoning."""
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class SPLRCarry:
    inner_carry: SPLRInnerCarry

    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]
    
    global_steps: torch.Tensor
    global_halted: torch.Tensor
    multi_step_data: Dict[str, torch.Tensor]


@dataclass
class SPLRModelOutput:
    logits: torch.Tensor
    q_halt_logits: torch.Tensor
    carry: SPLRCarry


CosSin = Tuple[torch.Tensor, torch.Tensor]

def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    """Truncated normal initialization."""
    # NOTE: PyTorch nn.init.trunc_normal_ is not mathematically correct, the std dev is not actually the std dev of initialized tensor
    # This function is a PyTorch version of jax truncated normal init (default init method in flax)
    # https://github.com/jax-ml/jax/blob/main/jax/_src/random.py#L807-L848
    # https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L162-L199

    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor

def _find_multiple(a, b):
    """Find smallest multiple of b that is >= a."""
    return (-(a // -b)) * b


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    """
    RMS normalization (from TRM).

    Args:
        hidden_states: Input tensor
        variance_epsilon: Small constant for numerical stability

    Returns:
        Normalized tensor
    """
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)

    return hidden_states.to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input (for RoPE)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor (B, S, H, D)
        k: Key tensor (B, S, H, D)
        cos: Cosine values (S, D)
        sin: Sine values (S, D)

    Returns:
        q_embed, k_embed: Rotated tensors
    """
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    # Expand cos/sin for broadcasting: (S, D) -> (1, S, 1, D)
    # This matches TRM's unsqueeze(-2) which adds dim at position -2 (num_heads)
    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float = 10000.0,
        device=None
    ):
        super().__init__()

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Concatenate for full rotation
        emb = torch.cat((freqs, freqs), dim=-1)

        # Use nn.Buffer (TRM uses this, not register_buffer)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self) -> CosSin:
        """Returns cached cos and sin values."""
        return self.cos_cached, self.sin_cached


class CastedLinear(nn.Module):
    """Linear layer with dynamic dtype casting."""

    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()

        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward with dtype casting."""
        return F.linear(
            input,
            self.weight.to(input.dtype),
            bias=self.bias.to(input.dtype) if self.bias is not None else None
        )


class CastedEmbedding(nn.Module):
    """Embedding layer with dynamic dtype casting."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        init_std: float,
        cast_to: torch.dtype
    ):
        super().__init__()
        self.cast_to = cast_to

        # Truncated normal init
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class SwiGLU(nn.Module):
    """SwiGLU activation."""

    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()

        # Round to nearest multiple of 256 for efficiency (matching TRM)
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class Attention(nn.Module):
    """Multi-head attention with optional causal masking."""

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        num_key_value_heads: int,
        causal: bool = False
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        # QKV projection
        self.qkv_proj = CastedLinear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=False
        )

        # Output projection
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(
        self,
        cos_sin: Optional[CosSin],
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass (matching TRM signature).

        Args:
            cos_sin: Optional RoPE (cos, sin) tuple
            hidden_states: (batch, max_position_embeddings, hidden_size)

        Returns:
            output: (batch, max_position_embeddings, hidden_size)
        """
        batch_size, max_position_embeddings, _ = hidden_states.shape

        # QKV projection
        qkv = self.qkv_proj(hidden_states)

        # Split head
        qkv = qkv.view(batch_size, max_position_embeddings, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Flash attention
        query, key, value = map(lambda t: einops.rearrange(t, 'B S H D -> B H S D'), (query, key, value))
        attn_output = F.scaled_dot_product_attention(query=query, key=key, value=value, is_causal=self.causal)
        attn_output = einops.rearrange(attn_output, 'B H S D -> B S H D')
        attn_output = attn_output.view(batch_size, max_position_embeddings, self.output_size)

        return self.o_proj(attn_output)


class SPLRBlock(nn.Module):
    """
    Single transformer block.

    Consists of:
    - Multi-head attention (or MLP for transposed mode)
    - SwiGLU FFN
    - RMS norm (post-norm)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Self-attention
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_attention_heads,
            num_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            causal=False
        )

        # FFN
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.intermediate_size / config.hidden_size
        )

        self.norm_eps = config.rms_norm_eps

    def forward(
        self,
        cos_sin: Optional[CosSin],
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Post-norm transformer block.

        Args:
            cos_sin: RoPE (cos, sin)
            hidden_states: (batch, max_position_embeddings, hidden_size)

        Returns:
            output: (batch, max_position_embeddings, hidden_size)
        """
        # Self-attention with post-norm
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps
        )

        # FFN with post-norm
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)

        return hidden_states


class CastedSparseEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, batch_size: int, init_std: float, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Truncated LeCun normal init
        self.weights = nn.Buffer(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std), persistent=True
        )

        # Local weights and IDs
        self.local_weights = nn.Buffer(torch.zeros(batch_size, embedding_dim, requires_grad=True), persistent=False)
        self.local_ids = nn.Buffer(torch.zeros(batch_size, dtype=torch.int32), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training:
            # Test mode, no gradient
            return self.weights[inputs].to(self.cast_to)

        # Training mode, fill task embedding from weights
        with torch.no_grad():
            self.local_weights.copy_(self.weights[inputs])
            self.local_ids.copy_(inputs)

        return self.local_weights.to(self.cast_to)


class SPLRReasoningModule(nn.Module):
    """SPLR-style recursive reasoning with z_H and z_L states."""

    def __init__(self, config: SPLRConfig, layers: int):
        super().__init__()
        self.config = config

        # Forward dtype
        self.forward_dtype = getattr(torch, config.forward_dtype)
        self.layers = nn.ModuleList([
            SPLRBlock(config) for _ in range(layers)
        ])

        # Initial states as buffers
        self.H_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(config.hidden_size, dtype=self.forward_dtype),
                std=1
            ),
            persistent=True
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(config.hidden_size, dtype=self.forward_dtype),
                std=1
            ),
            persistent=True
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        One L-level reasoning step with input injection.

        Args:
            hidden_states: Current hidden state to refine (batch, max_position_embeddings, hidden)
            input_injection: Input to inject (batch, max_position_embeddings, hidden)
            **kwargs: Additional args like cos_sin from parent

        Returns:
            refined_state: Updated hidden state (batch, max_position_embeddings, hidden)
        """
        # Add input injection
        hidden_states = hidden_states + input_injection

        # Apply L-level layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                **kwargs
            )

        return hidden_states



class SPLRInner(nn.Module):
    """Inner SPLR model for step-wise persistent latent reasoning."""

    def __init__(self, config: SPLRConfig):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)

        # Embedding scale
        self.embed_scale = math.sqrt(config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        # Token embedding
        self.embed_tokens = CastedEmbedding(
            config.vocab_size,
            config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype
        )

        # Load embeddings from HuggingFace model if specified 
        # TODO: (Should be manually disabled when loading from checkpoint)
        if config.load_embedding is not None:
            from transformers import AutoModelForCausalLM

            print(f"Loading embeddings from HuggingFace model: {config.load_embedding}")
            hf_model = AutoModelForCausalLM.from_pretrained(config.load_embedding)

            # Check hidden size
            hf_hidden_size = hf_model.config.hidden_size
            if hf_hidden_size != config.hidden_size:
                raise ValueError(
                    f"HuggingFace model hidden_size ({hf_hidden_size}) does not match "
                    f"config.hidden_size ({config.hidden_size})"
                )

            # Load embeddings from the HuggingFace model
            if config.load_embedding == "Qwen/Qwen3-0.6B":
                hf_embeddings = hf_model.model.embed_tokens.weight
            elif config.load_embedding == "openai-community/gpt2":
                hf_embeddings = hf_model.transformer.wte.weight
            else:
                raise ValueError(
                    f"Unsupported HuggingFace model: {config.load_embedding}"
                )

            with torch.no_grad():
                target_vocab_size = self.embed_tokens.embedding_weight.shape[0]
                hf_vocab_size = hf_embeddings.shape[0]

                if hf_vocab_size >= target_vocab_size:
                    # Pretrained vocab covers all tokens, just truncate
                    self.embed_tokens.embedding_weight.copy_(hf_embeddings[:target_vocab_size])
                else:
                    # Pretrained vocab is smaller (e.g. added special tokens)
                    # Copy pretrained embeddings, init new tokens with mean embedding
                    self.embed_tokens.embedding_weight[:hf_vocab_size].copy_(hf_embeddings)
                    mean_embedding = hf_embeddings.mean(dim=0)
                    self.embed_tokens.embedding_weight[hf_vocab_size:].copy_(
                        mean_embedding.unsqueeze(0).expand(target_vocab_size - hf_vocab_size, -1)
                    )
                    print(f"Initialized {target_vocab_size - hf_vocab_size} new token embeddings with mean embedding")

            print(f"Successfully loaded embeddings from {config.load_embedding}")
            del hf_model  # Free memory

        # Task embeddings
        self.task_emb_len = -(config.task_emb_ndim // -config.hidden_size) if config.task_emb_len == 0 else config.task_emb_len  # ceil div
        if config.task_emb_ndim > 0:
            self.task_emb = CastedSparseEmbedding(
                config.num_task_identifiers,
                config.task_emb_ndim,
                batch_size=config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype
            )

        # Q-head for halting
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)

        # Position embeddings (RoPE)
        if hasattr(config, 'rope_theta') and config.rope_theta is not None:
            self.rotary_emb = RotaryEmbedding(
                dim=config.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta
            )

        # Reasoning module (L-level)
        self.L_level = SPLRReasoningModule(config, config.L_layers)
        if config.hierarchical_reasoning:
            if config.H_layers == 0:
                raise ValueError("H_layers must be greater than 0 for hierarchical reasoning")
            self.H_level = SPLRReasoningModule(config, config.H_layers)

        # Q-head special init
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _input_embeddings(self, input: torch.Tensor, task_identifiers: torch.Tensor = None) -> torch.Tensor:
        """
        Token embedding with scaling.

        Args:
            input: (batch, max_position_embeddings) token IDs
            task_identifiers: (batch,) task IDs (optional)

        Returns:
            embedding: (batch, max_position_embeddings, hidden_size) token embeddings
        """
        batch_size, _ = input.shape

        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Task embeddings
        if self.config.task_emb_ndim > 0 and task_identifiers is not None:
            task_embedding = self.task_emb(task_identifiers)

            # Pad to match hidden_size * task_emb_len
            pad_count = self.task_emb_len * self.config.hidden_size - task_embedding.shape[-1]
            if pad_count > 0:
                task_embedding = F.pad(task_embedding, (0, pad_count))

            # Reshape and concatenate
            task_embedding = task_embedding.view(batch_size, self.task_emb_len, self.config.hidden_size)
            embedding = torch.cat((task_embedding, embedding), dim=-2)

        # Don't scale when using pretrained embeddings (they're already in the right scale)
        # This ensures consistency between training and evaluation
        if self.config.load_embedding is not None:
            return embedding
        else:
            # Scale randomly initialized embeddings
            return self.embed_scale * embedding


    def forward(
        self,
        carry: SPLRInnerCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[SPLRInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through inner model.

        Args:
            carry: Inner carry state
            batch: Dict with 'inputs' (token IDs), 'task_identifiers' (task IDs), and 'labels'

        Returns:
            new_carry: Updated carry state
            logits: Output logits (batch, max_position_embeddings, vocab_size)
            q_logits: (q_halt_logits, q_continue_logits)
        """
        # Get RoPE embeddings if available
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        task_identifiers = batch.get("task_identifiers", None)
        input_embeddings = self._input_embeddings(batch["inputs"], task_identifiers)

        # Forward iterations
        z_H, z_L = carry.z_H, carry.z_L

        if self.config.hierarchical_reasoning:
            H_level = self.H_level
        else:
            H_level = self.L_level

        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(
                        hidden_states=z_L,
                        input_injection=z_H + input_embeddings,
                        **seq_info
                    )
                z_H = H_level(
                    hidden_states=z_H,
                    input_injection=z_L,
                    **seq_info
                )

        # Final H-cycle with grad
        if self.config.L_cycles > self.config.L_grad_cycles:
            with torch.no_grad():
                for _L_step in range(self.config.L_cycles - self.config.L_grad_cycles):
                    z_L = self.L_level(
                        hidden_states=z_L,
                        input_injection=z_H + input_embeddings,
                        **seq_info
                    )
        for _L_step in range(self.config.L_grad_cycles):
            z_L = self.L_level(
                hidden_states=z_L,
                input_injection=z_H + input_embeddings,
                **seq_info
            )
        z_H = H_level(
            hidden_states=z_H,
            input_injection=z_L,
            **seq_info
        )

        # Q-head for halting (use first position of z_H)
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)  # (batch, 2)

        # New carry (detach for next iteration)
        new_carry = SPLRInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())

        return new_carry, z_H, (q_logits[..., 0], q_logits[..., 1])



class SPLRModel(nn.Module):
    def __init__(self, config: SPLRConfig):
        super().__init__()
        self.config = config
        self.inner = SPLRInner(self.config)
        self.output_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.output_head.weight = self.inner.embed_tokens.embedding_weight
        # NOTE: Assume the embeddings of the small model are tied
        elif self.config.load_embedding is not None:
            with torch.no_grad():
                self.output_head.weight.copy_(self.inner.embed_tokens.embedding_weight)
    
    @property
    def task_emb(self):
        return self.inner.task_emb
    
    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> SPLRCarry:
        """
        Initialize carry state for a batch with multi-step data.

        Args:
            batch: Dict with multi-step tensors
                   - 'inputs': (batch, max_reasoning_steps, max_position_embeddings)
                   - 'labels': (batch, max_reasoning_steps, max_position_embeddings)
                   - 'task_identifiers': (batch, max_reasoning_steps)

        Returns:
            carry: Initial SPLRCarry state
        """
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device

        return SPLRCarry(
            inner_carry=self.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),  # Start halted
            current_data={k: v[:, 0, ...].clone() for k, v in batch.items()},  # Initialize with turn 0
            global_steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            global_halted=torch.ones((batch_size,), dtype=torch.bool, device=device),  # Start globally halted
            multi_step_data={k: v.clone() for k, v in batch.items()}
        )
    
    def empty_carry(self, batch_size: int) -> SPLRInnerCarry:
        """Create empty carry state on the same device as the model."""
        # Use H_init's device to determine where to create the carry
        device = self.inner.L_level.H_init.device
        return SPLRInnerCarry(
            z_H=torch.empty(
                batch_size,
                self.config.max_position_embeddings,
                self.config.hidden_size,
                dtype=self.inner.forward_dtype,
                device=device
            ),
            z_L=torch.empty(
                batch_size,
                self.config.max_position_embeddings,
                self.config.hidden_size,
                dtype=self.inner.forward_dtype,
                device=device
            ),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: SPLRInnerCarry, reset_H_only: bool = False) -> SPLRInnerCarry:
        """
        Reset carry state for halted sequences.

        Args:
            reset_flag: (batch,) boolean tensor indicating which sequences to reset
            carry: Current inner carry state

        Returns:
            new_carry: Updated carry state
        """
        # Ensure H_init and L_init are on the same device as carry
        device = carry.z_H.device
        H_init = self.inner.L_level.H_init.to(device)
        L_init = self.inner.L_level.L_init.to(device)

        if reset_H_only:
            z_L = carry.z_L
        else:
            z_L = torch.where(
                reset_flag.view(-1, 1, 1),
                L_init,
                carry.z_L
            )

        return SPLRInnerCarry(
            z_H=torch.where(
                reset_flag.view(-1, 1, 1),
                H_init,
                carry.z_H
            ),
            z_L=z_L,
        )
    
    def forward(
        self,
        carry: SPLRCarry,
        batch: Dict[str, torch.Tensor],
        compute_logits: bool = True,
    ) -> SPLRModelOutput:
        """
        Forward pass with multi-step reasoning halting logic.

        Args:
            carry: Current multi-step reasoning carry state
            batch: Dict with multi-step data
                   - 'inputs': (batch, max_reasoning_steps, max_position_embeddings)
                   - 'labels': (batch, max_reasoning_steps, max_position_embeddings)
                   - 'task_identifiers': (batch, max_reasoning_steps)

        Returns:
            outputs: SPLRModelOutput
        """
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device

        # Update multi_step_data if globally halted
        new_multi_step_data = {}
        for k in batch.keys():
            new_multi_step_data[k] = torch.where(
                carry.global_halted.view(-1, *([1] * (batch[k].ndim - 1))),
                batch[k],
                carry.multi_step_data[k]
            )

        # Reset inner carry if globally halted or locally halted
        if self.config.enable_inter_latent:
            new_inner_carry = self.reset_carry(carry.global_halted, carry.inner_carry)
            if self.config.inter_reasoning_only:
                new_inner_carry = self.reset_carry(carry.halted, new_inner_carry, reset_H_only=True)
        else:
            new_inner_carry = self.reset_carry(carry.halted, carry.inner_carry)

        # Reset global_steps if globally halted
        new_global_steps = torch.where(
            carry.global_halted,
            torch.zeros_like(carry.global_steps),
            carry.global_steps
        )

        # Clear global_halted flag
        new_global_halted = torch.zeros_like(carry.global_halted, dtype=torch.bool)

        # Update current_data from multi_step_data if halted
        # Extract the turn at global_steps index for each sample
        batch_indices = torch.arange(batch_size, device=device)

        new_current_data = {}
        for k in batch.keys():
            current_turn_data = new_multi_step_data[k][batch_indices, new_global_steps.clamp(max=self.config.max_reasoning_steps - 1)]

            # Update only for halted samples
            new_current_data[k] = torch.where(
                carry.halted.view(-1, *([1] * (current_turn_data.ndim - 1))),
                current_turn_data,
                carry.current_data[k]
            )

        # Reset local steps if halted
        new_steps = torch.where(carry.halted, torch.zeros_like(carry.steps), carry.steps)

        # Forward through reasoning backbone
        new_inner_carry, z_H, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)
        if compute_logits:
            logits = self.output_head(z_H)
        else:
            logits = z_H.new_empty(0)

        # Local halting logic
        with torch.no_grad():
            # Increment step counter
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps

            # Determine if halted
            if self.training:
                halted = is_last_step
            else:
                halted = torch.zeros_like(is_last_step, dtype=torch.bool)

            # Halt signal
            if self.training:
                halted = halted | (q_halt_logits > 0)
            else:
                halted = halted | (q_halt_logits > 0.5)

            # Exploration
            if self.training:
                min_halt_steps = (torch.rand_like(q_halt_logits.float()) < self.config.halt_exploration_prob).long() * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

        # If locally halted, increment global_steps and check for global halt
        with torch.no_grad():
            # Increment global_steps for halted samples
            updated_global_steps = torch.where(halted, new_global_steps + 1, new_global_steps)

            # Check if exceeded max_reasoning_steps
            is_out_of_turns = updated_global_steps >= self.config.max_reasoning_steps

            # Check if next turn is padding (task_id == -1)
            is_padding = torch.zeros_like(halted, dtype=torch.bool)

            # For samples that halted and not exceeded max_reasoning_steps, check if next turn is padding
            for i in range(batch_size):
                if halted[i] and not is_out_of_turns[i]:
                    next_turn_idx = updated_global_steps[i].item()
                    is_padding[i] = (new_multi_step_data["task_identifiers"][i, next_turn_idx] == -1)

            # Set global_halted for samples that are out of turns or padding
            new_global_halted = halted & (is_out_of_turns | is_padding)
            new_global_steps = updated_global_steps

        

        # print("new_steps:")
        # print(new_steps)
        # print("halted:")
        # print(halted)
        # print("new_global_steps:")
        # print(new_global_steps)
        # print("new_global_halted:")
        # print(new_global_halted)

        # Create final carry
        final_carry = SPLRCarry(
            inner_carry=new_inner_carry,
            steps=new_steps,
            halted=halted,
            current_data=new_current_data,
            global_steps=new_global_steps,
            global_halted=new_global_halted,
            multi_step_data=new_multi_step_data
        )

        return SPLRModelOutput(
            logits=logits,
            q_halt_logits=q_halt_logits,
            carry=final_carry
        )