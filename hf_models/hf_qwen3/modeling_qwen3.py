# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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

from typing import Callable, Optional, Union

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import check_model_inputs as _check_model_inputs
from in_place_ttt.transformers_compat import resolve_check_model_inputs
from .configuration_qwen3 import Qwen3Config

# TTT: additional imports
from einops import rearrange, repeat
from in_place_ttt.ttt_aux.loss import compute_ttt_aux_loss

try:
    from opt_einsum import contract
except ModuleNotFoundError:
    contract = torch.einsum

check_model_inputs = resolve_check_model_inputs(_check_model_inputs)


@use_kernel_forward_from_hub("RMSNorm")
class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3MLP(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):  # TTT: added layer_idx
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        # TTT: setup
        self.layer_idx = -1 if layer_idx is None else layer_idx
        if getattr(config, "ttt_mode", False) and self.layer_idx in getattr(config, "ttt_layers", []):
            self.ttt_chunk = getattr(config, "ttt_chunk", 8192)
            if getattr(config, "ttt_proj", True):
                self.ttt_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            else:
                self.ttt_proj = None
            self.ttt_lr = getattr(config, "ttt_lr", 0.3)
            # effective write-rule lr; defaults to target ttt_lr. The training loop
            # overwrites this each step during warmup (set_ttt_lr_effective). eval /
            # inference keep the target value.
            self.ttt_lr_effective = self.ttt_lr
            self.ttt_write_rule = getattr(config, "ttt_write_rule", "outer")
            self.ttt_nlms_lambda = float(getattr(config, "ttt_nlms_lambda", 1.0))
            self.ttt_nlms_detach_state = bool(getattr(config, "ttt_nlms_detach_state", False))
            self.ttt_nlms_decay = float(getattr(config, "ttt_nlms_decay", 0.0))
            self.ttt_conv = nn.Conv1d(
                self.hidden_size, self.hidden_size, kernel_size=5, padding=2,
                groups=self.hidden_size, bias=False,
            )
            # TTT: optional key normalization. Normalizes the gate key/query h
            # (= SiLU(gate)*up, intermediate_size dims) so the un-normalized
            # h_t . h_c' gate cannot be dominated by a few massive-norm tokens.
            # Only the delta (fast-weight) path consumes the normalized key; the
            # base MLP path keeps the original h (see forward).
            if getattr(config, "ttt_key_norm", False):
                self.ttt_key_norm = Qwen3RMSNorm(self.intermediate_size, eps=config.rms_norm_eps)

    # TTT: new method
    def padding(self, x):
        if not hasattr(self, "ttt_chunk"):
            return x
        if x.shape[1] % self.ttt_chunk != 0:
            padding_embeddings = torch.zeros(
                [x.shape[0], self.ttt_chunk - x.shape[1] % self.ttt_chunk, x.shape[2]],
                device=x.device, dtype=x.dtype,
            )
            x = torch.cat([x, padding_embeddings], dim=1)
        return rearrange(x, "b (t c) d -> b t c d", c=self.ttt_chunk)

    def compute_v_hat_for_aux(self, inputs_embeds: torch.Tensor) -> Optional[torch.Tensor]:
        if not hasattr(self, "ttt_conv"):
            return None

        seq_len = inputs_embeds.shape[1]
        t_padded = self.padding(inputs_embeds)
        bs, chunk_num, chunk_size, _ = t_padded.shape
        t_conv = (
            self.ttt_conv(t_padded.transpose(-1, -2).reshape(bs * chunk_num, -1, chunk_size))
            .transpose(-1, -2)
            .reshape(bs, chunk_num, chunk_size, -1)
        )
        if self.ttt_proj is not None:
            v_hat = contract("b t c d, d e -> b t c e", t_conv, self.ttt_proj.weight)
        else:
            v_hat = t_conv
        return rearrange(v_hat, "b t c d -> b (t c) d")[:, :seq_len, :]

    def _record_ttt_monitor_stats(
        self,
        delta_down_proj: torch.Tensor,
        h_padded: torch.Tensor,
        down_proj: torch.Tensor,
    ) -> None:
        if delta_down_proj.shape[1] == 0:
            self._last_ttt_monitor_stats = None
            return

        sample_dim = min(int(getattr(self.config, "ttt_monitor_sample_dim", 64)), delta_down_proj.shape[-2])
        sample_tokens = min(int(getattr(self.config, "ttt_monitor_sample_tokens", 1)), h_padded.shape[-2])
        if sample_dim <= 0 or sample_tokens <= 0:
            self._last_ttt_monitor_stats = None
            return

        with torch.no_grad():
            eps = torch.tensor(1e-12, device=down_proj.device, dtype=torch.float32)
            base_weight_sample = self.down_proj.weight[:sample_dim].detach().float()
            base_weight_norm = base_weight_sample.norm().clamp_min(eps)

            delta_sample = delta_down_proj[:, :, :sample_dim, :].detach().float()
            delta_weight_ratio = delta_sample.norm(dim=(-2, -1)).mean() / base_weight_norm
            delta_weight_cumsum_ratio = delta_sample.cumsum(dim=1).norm(dim=(-2, -1)).mean() / base_weight_norm

            h_sample = h_padded[:, :, :sample_tokens, :].detach().float()
            output_sample = down_proj[:, :, :sample_tokens, :sample_dim].detach().float()
            base_output_sample = contract("d h, b t c h -> b t c d", base_weight_sample, h_sample)
            output_delta_ratio = (output_sample - base_output_sample).norm() / output_sample.norm().clamp_min(eps)

            self._last_ttt_monitor_stats = {
                "delta_weight_sample_ratio": delta_weight_ratio.detach(),
                "delta_weight_cumsum_sample_ratio": delta_weight_cumsum_ratio.detach(),
                "output_delta_sample_ratio": output_delta_ratio.detach(),
            }

    def _record_ttt_future_chunk_aux(
        self,
        prediction_states: torch.Tensor,
        target_states: torch.Tensor,
        seq_len: int,
    ) -> None:
        self._last_ttt_aux_prediction = None
        self._last_ttt_aux_target = None

        if (
            not self.training
            or float(getattr(self.config, "ttt_aux_loss_weight", 0.0)) <= 0.0
            or getattr(self.config, "ttt_aux_target", "next_input_embed") != "future_chunk_hidden"
        ):
            return

        future_chunks = int(getattr(self.config, "ttt_aux_future_chunks", 1))
        if future_chunks < 1 or prediction_states.shape[1] <= future_chunks:
            return

        _, chunk_num, chunk_size, _ = prediction_states.shape
        valid_tokens = torch.arange(
            chunk_num * chunk_size,
            device=prediction_states.device,
        ) < seq_len
        valid_tokens = valid_tokens.view(1, chunk_num, chunk_size, 1)
        valid_tokens = valid_tokens.to(dtype=prediction_states.dtype)
        denom = valid_tokens.sum(dim=2).clamp_min(1.0)

        prediction_pool = (prediction_states * valid_tokens).sum(dim=2) / denom
        target_pool = (target_states.detach() * valid_tokens).sum(dim=2) / denom
        self._last_ttt_aux_prediction = prediction_pool[:, :-future_chunks, :]
        self._last_ttt_aux_target = target_pool[:, future_chunks:, :]

    def forward(self, x, t: Optional[torch.Tensor] = None):  # TTT: added t param
        h = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        # TTT: branch on whether this is a TTT layer with target states
        if t is None or not hasattr(self, "ttt_conv"):
            self._last_ttt_aux_prediction = None
            self._last_ttt_aux_target = None
            return self.down_proj(h)
        # TTT path
        target_padded = self.padding(t)
        h_padded = self.padding(h)
        bs, chunk_num, chunk_size, _ = target_padded.shape
        t_conv = (
            self.ttt_conv(target_padded.transpose(-1, -2).reshape(bs * chunk_num, -1, chunk_size))
            .transpose(-1, -2)
            .reshape(bs, chunk_num, chunk_size, -1)
        )
        if self.ttt_proj is not None:
            prediction_states = contract("b t c d, d e -> b t c e", t_conv, self.ttt_proj.weight)
        else:
            prediction_states = t_conv
        if getattr(self, "ttt_write_rule", "outer") == "nlms":
            # Block residual write (per-key NLMS), chunk-serial loop.
            # base path uses original h (W0); delta path reads sum_{j<t} of NLMS updates.
            # delta_down_proj keeps per-chunk dW for the monitor stats (same meaning as outer).
            W0 = self.down_proj.weight                       # [d, h_dim]
            S = torch.zeros(bs, W0.shape[0], W0.shape[1], device=h.device, dtype=torch.float32)
            # truncated BPTT (opt-in): detach the *history* before each chunk so the
            # readout/write see prior state as a constant, but the current chunk's
            # own write dW_i stays differentiable. This bounds the cross-chunk
            # gradient chain (which explodes over many chunks with real-scale weights)
            # WITHOUT severing the value path's gradient. Forward values are unchanged.
            detach_state = getattr(self, "ttt_nlms_detach_state", False)
            outs = []
            per_chunk_dw = []
            for i in range(chunk_num):
                Ki = h_padded[:, i].float()                  # [b, c, h_dim]
                Vi = prediction_states[:, i].float()         # [b, c, d]
                S_hist = S.detach() if detach_state else S   # write history (constant under detach)
                base_i = contract("d h, b c h -> b c d", W0.float(), Ki)
                # readout reads the *live* S (carries the previous chunk's differentiable
                # dW under detach); write residual reads the detached history. Numerically
                # S == S_hist (same values), so forward is unchanged in either mode; only
                # the backward graph differs. This gives 1-step truncated BPTT: each dW_i
                # earns gradient via the NEXT chunk's readout (chain length = 1 chunk, no
                # 16-step unroll → no explosion), yet ttt_proj/ttt_conv stay trainable even
                # with a frozen backbone (ttt_train_only).
                delta_i = contract("b d h, b c h -> b c d", S, Ki)
                outs.append((base_i + delta_i).to(h.dtype))
                # per-key residual write (chunk-start S for all keys in this chunk)
                pred_i = contract("b c h, b d h -> b c d", Ki, S_hist)   # Ki @ S^T
                resid_i = Vi - pred_i                                # [b, c, d]
                denom_i = self.ttt_nlms_lambda + (Ki * Ki).sum(dim=-1, keepdim=True)  # [b, c, 1]
                resid_i = resid_i / denom_i
                dW_i = contract("b c d, b c h -> b d h", resid_i, Ki) * self.ttt_lr_effective   # [b, d, h_dim]
                # average the C per-key rank-1 writes over the chunk: applying all C
                # writes simultaneously against a stale chunk-start S otherwise scales
                # the combined update ~sqrt(C), causing S to diverge over many chunks.
                dW_i = dW_i / Ki.shape[1]
                per_chunk_dw.append(dW_i)
                # accumulate with decay gate: (1-α)·history + current write.
                # decay=0 → pure NLMS (S = S_hist + dW_i, bit-identical to before);
                # decay>0 → ‖S‖ bounded (~‖dW‖/α), breaks runaway feedback, and the
                # cross-chunk gradient chain decays as (1-α)^k (no detach needed).
                S = (1.0 - self.ttt_nlms_decay) * S_hist + dW_i
            down_proj = torch.stack(outs, dim=1)             # [b, chunk_num, c, d]
            delta_down_proj = torch.stack(per_chunk_dw, dim=1).to(h.dtype)  # [b, chunk_num, d, h_dim]
            self._record_ttt_future_chunk_aux(prediction_states, target_padded, x.shape[1])
            self._record_ttt_monitor_stats(delta_down_proj, h_padded, down_proj)
            return rearrange(down_proj, "b t c d -> b (t c) d")[:, : x.shape[1], :]
        if not hasattr(self, "ttt_key_norm"):
            # Original fused path — kept verbatim so that disabling key-norm is
            # bit-for-bit identical to the pre-change behavior (the base W0 and
            # the delta share one cumsum and one contract over the same h).
            d_down_proj = contract(
                "b t c h, b t c d -> b t d h",
                h_padded[:, :-1], prediction_states[:, :-1],
            )
            delta_down_proj = d_down_proj * self.ttt_lr_effective
            d_down_proj = torch.cat(
                [repeat(self.down_proj.weight, "d h -> b 1 d h", b=bs), delta_down_proj],
                dim=1,
            )
            d_down_proj_sum = d_down_proj.cumsum(dim=1)
            down_proj = contract("b t d h, b t c h -> b t c d", d_down_proj_sum, h_padded)
        else:
            # Key-norm path: the gate key/query that builds and reads the fast
            # weights uses the normalized h, while the frozen base projection
            # W0 keeps the original h — otherwise normalizing h would also
            # rewrite the pretrained MLP output and break it.
            h_norm_padded = self.padding(self.ttt_key_norm(h))
            d_down_proj = contract(
                "b t c h, b t c d -> b t d h",
                h_norm_padded[:, :-1], prediction_states[:, :-1],
            )
            delta_down_proj = d_down_proj * self.ttt_lr_effective
            # cumsum over deltas only (W0 handled separately); a leading zero
            # block preserves the same causal offset: chunk t reads sum_{j<t}.
            delta_w_sum = torch.cat(
                [torch.zeros_like(delta_down_proj[:, :1]), delta_down_proj],
                dim=1,
            ).cumsum(dim=1)
            base_out = contract("d h, b t c h -> b t c d", self.down_proj.weight, h_padded)
            delta_out = contract("b t d h, b t c h -> b t c d", delta_w_sum, h_norm_padded)
            down_proj = base_out + delta_out
        self._record_ttt_future_chunk_aux(prediction_states, target_padded, x.shape[1])
        self._record_ttt_monitor_stats(delta_down_proj, h_padded, down_proj)
        return rearrange(down_proj, "b t c d -> b (t c) d")[:, : x.shape[1], :]


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3MLP(config, layer_idx=layer_idx)  # TTT: pass layer_idx
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]
        # TTT: check if this is a TTT layer
        self.is_ttt_layer = getattr(config, "ttt_mode", False) and layer_idx in getattr(config, "ttt_layers", [])

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        target_states: Optional[torch.Tensor] = None,
        ttt_aux_only: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        if ttt_aux_only:
            return self.mlp.compute_v_hat_for_aux(hidden_states)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if target_states is None and self.is_ttt_layer:
            target_states = hidden_states
        hidden_states = self.mlp(hidden_states, t=target_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class Qwen3PreTrainedModel(PreTrainedModel):
    config: Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen3DecoderLayer,
        "attentions": Qwen3Attention,
    }

    # TTT: custom weight init for continual pretraining.
    # Non-TTT weights are loaded from pretrained checkpoint; only TTT modules need init.
    # Handles DTensor for FSDP2 distributed training.
    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)

        if isinstance(module, nn.Linear):
            if module.weight.device.type == "meta":
                return
            # Square matrix (ttt_proj is hidden_size x hidden_size): diagonal init
            if module.weight.shape[0] == module.weight.shape[1]:
                diag_size = module.weight.shape[0]

                weight_data = module.weight.data
                if hasattr(weight_data, '_local_tensor'):
                    # DTensor: operate on local shard
                    import torch.distributed as dist
                    local_tensor = weight_data._local_tensor
                    local_tensor.zero_()

                    local_rows = local_tensor.shape[0]
                    num_cols = local_tensor.shape[1]
                    rank = dist.get_rank()
                    start_row = rank * local_rows

                    g = torch.Generator(device=local_tensor.device)
                    g.manual_seed(42)
                    all_diag_values = torch.randn(diag_size, generator=g, device=local_tensor.device, dtype=local_tensor.dtype) * std

                    local_row_indices = torch.arange(local_rows, device=local_tensor.device)
                    global_col_indices = start_row + local_row_indices

                    valid_mask = global_col_indices < num_cols
                    local_row_indices = local_row_indices[valid_mask]
                    global_col_indices = global_col_indices[valid_mask]

                    if len(local_row_indices) > 0:
                        local_tensor[local_row_indices, global_col_indices] = all_diag_values[global_col_indices]
                else:
                    weight_data.zero_()
                    diag_values = torch.randn(diag_size, device=weight_data.device, dtype=weight_data.dtype) * std
                    indices = torch.arange(diag_size, device=weight_data.device)
                    weight_data[indices, indices] = diag_values
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            # TTT conv: zero init
            module.weight.data.zero_()
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif "RMSNorm" in module.__class__.__name__:
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.fill_(1.0)


class Qwen3RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Qwen3Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@auto_docstring
class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config):
        ttt_compress_window = getattr(config, "ttt_compress_window", 0)
        if ttt_compress_window > 0:
            config.use_sliding_window = True
            config.sliding_window = ttt_compress_window
            full_attn_idx = set(getattr(config, "full_attention_layers", []) or [])
            config.layer_types = [
                "full_attention" if i in full_attn_idx else "sliding_attention"
                for i in range(config.num_hidden_layers)
            ]

        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.ttt_layers = getattr(config, "ttt_layers", [])
        self.ttt_mode = getattr(config, "ttt_mode", False)
        self.ttt_target = getattr(config, "ttt_target", "hidden_states")

        # Initialize weights and apply final processing
        self.post_init()

    def _resolve_ttt_target_states(
        self,
        decoder_layer: Qwen3DecoderLayer,
        inputs_embeds: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if not self.ttt_mode or not decoder_layer.is_ttt_layer:
            return None
        if self.ttt_target == "input_embed":
            return inputs_embeds
        return None

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        ttt_aux_loss_weight = float(getattr(self.config, "ttt_aux_loss_weight", 0.0))
        ttt_aux_target = getattr(self.config, "ttt_aux_target", "next_input_embed")
        collect_future_chunk_aux = self.training and ttt_aux_loss_weight > 0.0 and self.ttt_mode and (
            ttt_aux_target == "future_chunk_hidden"
        )
        future_chunk_predictions = []
        future_chunk_targets = []

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            collect_layer_aux = collect_future_chunk_aux and decoder_layer.is_ttt_layer
            saved_ckpt = getattr(decoder_layer, "gradient_checkpointing", False)
            if collect_layer_aux:
                decoder_layer.gradient_checkpointing = False
            try:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    target_states=self._resolve_ttt_target_states(decoder_layer, inputs_embeds),
                    **kwargs,
                )
            finally:
                if collect_layer_aux:
                    decoder_layer.gradient_checkpointing = saved_ckpt

            if collect_layer_aux:
                prediction = getattr(decoder_layer.mlp, "_last_ttt_aux_prediction", None)
                target = getattr(decoder_layer.mlp, "_last_ttt_aux_target", None)
                if prediction is not None and target is not None:
                    future_chunk_predictions.append(prediction)
                    future_chunk_targets.append(target)

        hidden_states = self.norm(hidden_states)
        ttt_aux_loss = None
        if collect_future_chunk_aux and future_chunk_predictions:
            ttt_aux_loss = compute_ttt_aux_loss(
                future_chunk_predictions,
                future_chunk_targets,
                loss_type=getattr(self.config, "ttt_aux_loss_type", "jepa"),
                loss_exp=float(getattr(self.config, "ttt_jepa_loss_exp", 1.0)),
                reg_coeff=float(getattr(self.config, "ttt_jepa_reg_coeff", 0.0)),
                reg_eps=float(getattr(self.config, "ttt_jepa_reg_eps", 0.0001)),
            )
        elif (
            self.training
            and ttt_aux_loss_weight > 0.0
            and self.ttt_mode
            and self.ttt_target == "input_embed"
            and ttt_aux_target == "next_input_embed"
        ):
            seq_len = inputs_embeds.shape[1]
            if seq_len > 1:
                emb_for_aux = inputs_embeds.detach()
                target = emb_for_aux[:, 1:, :]
                predictions = []
                targets = []
                for decoder_layer in self.layers[: self.config.num_hidden_layers]:
                    if not decoder_layer.is_ttt_layer:
                        continue
                    saved_ckpt = getattr(decoder_layer, "gradient_checkpointing", False)
                    decoder_layer.gradient_checkpointing = False
                    try:
                        v_hat = decoder_layer(emb_for_aux, ttt_aux_only=True)
                    finally:
                        decoder_layer.gradient_checkpointing = saved_ckpt
                    if v_hat is None:
                        continue
                    predictions.append(v_hat[:, : seq_len - 1, :])
                    targets.append(target)
                if predictions:
                    ttt_aux_loss = compute_ttt_aux_loss(
                        predictions,
                        targets,
                        loss_type=getattr(self.config, "ttt_aux_loss_type", "jepa"),
                        loss_exp=float(getattr(self.config, "ttt_jepa_loss_exp", 1.0)),
                        reg_coeff=float(getattr(self.config, "ttt_jepa_reg_coeff", 0.0)),
                        reg_eps=float(getattr(self.config, "ttt_jepa_reg_eps", 0.0001)),
                    )

        out = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )
        out.ttt_aux_loss = ttt_aux_loss
        self._last_ttt_aux_loss = ttt_aux_loss
        return out


@auto_docstring
class Qwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        out = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        out.ttt_aux_loss = getattr(outputs, "ttt_aux_loss", None)
        return out


class Qwen3ForSequenceClassification(GenericForSequenceClassification, Qwen3PreTrainedModel):
    pass


class Qwen3ForTokenClassification(GenericForTokenClassification, Qwen3PreTrainedModel):
    pass


class Qwen3ForQuestionAnswering(GenericForQuestionAnswering, Qwen3PreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


__all__ = [
    "Qwen3ForCausalLM",
    "Qwen3ForQuestionAnswering",
    "Qwen3PreTrainedModel",
    "Qwen3Model",
    "Qwen3ForSequenceClassification",
    "Qwen3ForTokenClassification",
]
