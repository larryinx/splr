from re import T

from transformers.configuration_utils import PretrainedConfig


class SPLRConfig(PretrainedConfig):
    """
    Configuration class for SPLR (Step-wise Persistent Latent Reasoner).
    """

    model_type = "splr"

    def __init__(
        self,
        # Vocabulary and embeddings
        vocab_size: int = 32000,
        hidden_size: int = 512,
        batch_size: int = 1024,
        max_position_embeddings: int = 4096,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 8,
        intermediate_size: int = 2048,
        tie_word_embeddings: bool = True,

        # Reasoning Module configuration
        hierarchical_reasoning: bool = False,
        L_cycles: int = 6,
        H_cycles: int = 3,
        L_layers: int = 2,
        H_layers: int = 0,
        L_grad_cycles: int = 6,
        H_grad_cycles: int = 1,
        halt_max_steps: int = 16,
        halt_exploration_prob: float = 0.1,

        # Multi-step reasoning
        max_reasoning_steps: int = 10,

        # Task embeddings
        task_emb_len: int = 0,
        task_emb_ndim: int = 0,
        num_task_identifiers: int = 0,

        # Inter-step reasoning
        enable_inter_latent: bool = True,
        inter_reasoning_only: bool = False,

        # Embedding loading
        load_embedding: str = None,

        # Attention
        attention_bias: bool = False,
        attention_dropout: float = 0.0,

        # Position encoding
        rope_theta: float = 10000.0,

        # Normalization
        rms_norm_eps: float = 1e-6,

        # Forward dtype
        forward_dtype: str = "bfloat16",

        # ACT options
        no_ACT_continue: bool = True,

        # Special tokens
        pad_token_id: int = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,

        **kwargs,
    ):
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # Vocabulary and embeddings
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.intermediate_size = intermediate_size
        self.tie_word_embeddings = tie_word_embeddings

        # Reasoning Module configuration
        self.hierarchical_reasoning = hierarchical_reasoning
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.L_layers = L_layers
        self.H_layers = H_layers
        self.L_grad_cycles = L_grad_cycles
        self.H_grad_cycles = H_grad_cycles
        self.halt_max_steps = halt_max_steps
        self.halt_exploration_prob = halt_exploration_prob

        # Multi-step reasoning
        self.max_reasoning_steps = max_reasoning_steps

        # Task embeddings
        self.task_emb_len = task_emb_len
        self.task_emb_ndim = task_emb_ndim
        self.num_task_identifiers = num_task_identifiers

        # Inter-step reasoning
        self.enable_inter_latent = enable_inter_latent
        self.inter_reasoning_only = inter_reasoning_only

        # Embedding loading
        self.load_embedding = load_embedding

        # Attention
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # Position encoding
        self.rope_theta = rope_theta

        # Normalization
        self.rms_norm_eps = rms_norm_eps

        # Forward dtype
        self.forward_dtype = forward_dtype

        # ACT options
        self.no_ACT_continue = no_ACT_continue

        # Derived parameters
        # Ensure head_dim * num_heads == hidden_size by adjusting if needed
        self.head_dim = hidden_size // num_attention_heads

        # Validate that dimensions are compatible
        if self.head_dim * num_attention_heads != hidden_size:
            # Adjust hidden_size to be divisible by num_heads
            old_hidden = hidden_size
            self.hidden_size = self.head_dim * num_attention_heads
            import warnings
            warnings.warn(
                f"hidden_size ({old_hidden}) is not divisible by num_attention_heads ({num_attention_heads}). "
                f"Adjusting hidden_size to {self.hidden_size}"
            )

    def to_dict(self):
        """Override to include all custom parameters."""
        output = super().to_dict()
        return output
