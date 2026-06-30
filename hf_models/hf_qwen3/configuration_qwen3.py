# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# This file may have been modified by Bytedance Ltd. and/or its affiliates
# ("Bytedance's Modifications"). All Bytedance's Modifications are Copyright
# 2026 Bytedance Ltd. and/or its affiliates.

"""Qwen3 model configuration"""

from transformers.configuration_utils import PretrainedConfig, layer_type_validation
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging


logger = logging.get_logger(__name__)


class Qwen3Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3Model`]. It is used to instantiate a
    Qwen3 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen3-8B [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Qwen3 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Qwen3Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `32`.
        head_dim (`int`, *optional*, defaults to 128):
            The attention head dimension.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        max_window_layers (`int`, *optional*, defaults to 28):
            The number of layers using full attention. The first `max_window_layers` layers will use full attention, while any
            additional layer afterwards will use SWA (Sliding Window Attention).
        layer_types (`list`, *optional*):
            Attention pattern for each layer.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import Qwen3Model, Qwen3Config

    >>> # Initializing a Qwen3 style configuration
    >>> configuration = Qwen3Config()

    >>> # Initializing a model from the Qwen3-8B style configuration
    >>> model = Qwen3Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen3"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `Qwen3`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        layer_types=None,
        attention_dropout=0.0,
        full_attention_layers=None,
        # TTT-related parameters
        ttt_layers=[0, 6, 12, 18, 24, 30],
        ttt_mode=True,
        ttt_proj=True,
        ttt_key_norm=False,
        ttt_lr=0.3,
        ttt_chunk=8192,
        ttt_target="hidden_states",
        ttt_write_rule="outer",
        ttt_nlms_lambda=1.0,
        ttt_nlms_detach_state=False,
        ttt_nlms_decay=0.0,
        ttt_lr_warmup_steps=0,
        ttt_lr_warmup_init=0.0,
        ttt_compress_window=0,
        ttt_aux_loss_weight=0.0,
        ttt_aux_target="next_input_embed",
        ttt_aux_future_chunks=1,
        ttt_aux_loss_type="jepa",
        ttt_jepa_loss_exp=1.0,
        ttt_jepa_reg_coeff=0.0,
        ttt_jepa_reg_eps=0.0001,
        ttt_train_only=False,
        ttt_param_lr_multiplier=1.0,
        ttt_param_weight_decay=None,
        ttt_monitor_sample_dim=64,
        ttt_monitor_sample_tokens=1,
        ttt_monitor_output_delta_target="mlp",
        ttt_monitor_logit_sample_tokens=1,
        ttt_monitor_logit_sample_dim=0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)
        self.full_attention_layers = list(full_attention_layers) if full_attention_layers else []
        for i in self.full_attention_layers:
            if not (0 <= i < self.num_hidden_layers):
                raise ValueError(
                    f"full_attention_layers contains index {i} out of range [0, {self.num_hidden_layers})"
                )

        # TTT settings
        self.ttt_layers = ttt_layers
        self.ttt_mode = ttt_mode
        self.ttt_proj = ttt_proj
        self.ttt_key_norm = bool(ttt_key_norm)
        self.ttt_lr = ttt_lr
        self.ttt_chunk = ttt_chunk
        self.ttt_write_rule = str(ttt_write_rule)
        if self.ttt_write_rule not in {"outer", "nlms"}:
            raise ValueError("ttt_write_rule must be one of {'outer', 'nlms'}")
        self.ttt_nlms_lambda = float(ttt_nlms_lambda)
        self.ttt_nlms_detach_state = bool(ttt_nlms_detach_state)
        # Decay gate for the NLMS fast-weight state: S <- (1 - decay) * S + dW.
        # Bounds ||S|| (~||dW||/decay) to break the runaway readout-residual
        # feedback. decay=0.0 is bit-identical to pure NLMS (unbounded accumulation).
        self.ttt_nlms_decay = float(ttt_nlms_decay)
        if not (0.0 <= self.ttt_nlms_decay < 1.0):
            raise ValueError("ttt_nlms_decay must be in [0.0, 1.0)")
        # Linear warmup for the TTT write-rule lr (ttt_lr): protects the NLMS
        # cold-start (projections not yet converging residuals) from divergence.
        # warmup_steps=0 → no warmup (ttt_lr_effective == ttt_lr, bit-identical).
        self.ttt_lr_warmup_steps = int(ttt_lr_warmup_steps)
        if self.ttt_lr_warmup_steps < 0:
            raise ValueError("ttt_lr_warmup_steps must be >= 0")
        self.ttt_lr_warmup_init = float(ttt_lr_warmup_init)
        if self.ttt_lr_warmup_init < 0:
            raise ValueError("ttt_lr_warmup_init must be >= 0")
        self.ttt_target = ttt_target
        if self.ttt_target not in {"hidden_states", "input_embed"}:
            raise ValueError("ttt_target must be one of {'hidden_states', 'input_embed'}")
        self.ttt_compress_window = ttt_compress_window
        self.ttt_aux_loss_weight = float(ttt_aux_loss_weight)
        if self.ttt_aux_loss_weight < 0:
            raise ValueError(f"ttt_aux_loss_weight must be >= 0, got {self.ttt_aux_loss_weight}")
        self.ttt_aux_target = str(ttt_aux_target)
        if self.ttt_aux_target not in {"next_input_embed", "future_chunk_hidden"}:
            raise ValueError(
                "ttt_aux_target must be one of {'next_input_embed', 'future_chunk_hidden'}, "
                f"got {self.ttt_aux_target!r}"
            )
        self.ttt_aux_future_chunks = int(ttt_aux_future_chunks)
        if self.ttt_aux_future_chunks < 1:
            raise ValueError(f"ttt_aux_future_chunks must be >= 1, got {self.ttt_aux_future_chunks}")
        self.ttt_aux_loss_type = str(ttt_aux_loss_type)
        if self.ttt_aux_loss_type not in {"jepa", "cosine"}:
            raise ValueError(
                "ttt_aux_loss_type must be one of {'jepa', 'cosine'}, got "
                f"{self.ttt_aux_loss_type!r}"
            )
        self.ttt_jepa_loss_exp = float(ttt_jepa_loss_exp)
        if self.ttt_jepa_loss_exp <= 0:
            raise ValueError(f"ttt_jepa_loss_exp must be > 0, got {self.ttt_jepa_loss_exp}")
        self.ttt_jepa_reg_coeff = float(ttt_jepa_reg_coeff)
        if self.ttt_jepa_reg_coeff < 0:
            raise ValueError(f"ttt_jepa_reg_coeff must be >= 0, got {self.ttt_jepa_reg_coeff}")
        self.ttt_jepa_reg_eps = float(ttt_jepa_reg_eps)
        if self.ttt_jepa_reg_eps < 0:
            raise ValueError(f"ttt_jepa_reg_eps must be >= 0, got {self.ttt_jepa_reg_eps}")
        self.ttt_train_only = bool(ttt_train_only)
        self.ttt_param_lr_multiplier = float(ttt_param_lr_multiplier)
        if self.ttt_param_lr_multiplier <= 0:
            raise ValueError(f"ttt_param_lr_multiplier must be > 0, got {self.ttt_param_lr_multiplier}")
        self.ttt_param_weight_decay = None if ttt_param_weight_decay is None else float(ttt_param_weight_decay)
        if self.ttt_param_weight_decay is not None and self.ttt_param_weight_decay < 0:
            raise ValueError(f"ttt_param_weight_decay must be >= 0, got {self.ttt_param_weight_decay}")
        self.ttt_monitor_sample_dim = int(ttt_monitor_sample_dim)
        if self.ttt_monitor_sample_dim < 0:
            raise ValueError(f"ttt_monitor_sample_dim must be >= 0, got {self.ttt_monitor_sample_dim}")
        self.ttt_monitor_sample_tokens = int(ttt_monitor_sample_tokens)
        if self.ttt_monitor_sample_tokens < 0:
            raise ValueError(f"ttt_monitor_sample_tokens must be >= 0, got {self.ttt_monitor_sample_tokens}")
        self.ttt_monitor_output_delta_target = str(ttt_monitor_output_delta_target)
        if self.ttt_monitor_output_delta_target not in {"mlp", "logits"}:
            raise ValueError(
                "ttt_monitor_output_delta_target must be one of {'mlp', 'logits'}, "
                f"got {self.ttt_monitor_output_delta_target!r}"
            )
        self.ttt_monitor_logit_sample_tokens = int(ttt_monitor_logit_sample_tokens)
        if self.ttt_monitor_logit_sample_tokens < 0:
            raise ValueError(
                f"ttt_monitor_logit_sample_tokens must be >= 0, got {self.ttt_monitor_logit_sample_tokens}"
            )
        self.ttt_monitor_logit_sample_dim = int(ttt_monitor_logit_sample_dim)
        if self.ttt_monitor_logit_sample_dim < 0:
            raise ValueError(f"ttt_monitor_logit_sample_dim must be >= 0, got {self.ttt_monitor_logit_sample_dim}")

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["Qwen3Config"]
