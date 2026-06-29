"""Regression tests for the TTT key-normalization path (Qwen3 training model).

Two guarantees:
1. With ttt_key_norm disabled, the new forward is bit-for-bit identical to the
   original fused path (the dual-path split must not perturb the baseline).
2. With ttt_key_norm enabled, only the delta (fast-weight) path changes — the
   base MLP projection W0 @ h stays identical, and the output is finite.
"""

import torch

from hf_models.hf_qwen3.configuration_qwen3 import Qwen3Config
from hf_models.hf_qwen3.modeling_qwen3 import Qwen3MLP
from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config as InferenceQwen3Config
from inference_model.hf_qwen3.modeling_qwen3 import Qwen3MLP as InferenceQwen3MLP


def _tiny_mlp_config(**overrides):
    kwargs = dict(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=64,
        ttt_layers=[0],
        ttt_mode=True,
        ttt_proj=True,
        ttt_lr=0.5,
        ttt_chunk=2,
        ttt_target="input_embed",
    )
    kwargs.update(overrides)
    return Qwen3Config(**kwargs)


def _build_mlp(*, key_norm: bool, seed: int = 0) -> Qwen3MLP:
    torch.manual_seed(seed)
    config = _tiny_mlp_config(ttt_key_norm=key_norm)
    mlp = Qwen3MLP(config, layer_idx=0)
    # Give the TTT modules non-trivial weights so the delta path is exercised
    # (ttt_conv is zero-init and ttt_proj is diag-init by default).
    torch.manual_seed(seed + 1)
    with torch.no_grad():
        mlp.ttt_conv.weight.normal_(std=0.1)
        if mlp.ttt_proj is not None:
            mlp.ttt_proj.weight.normal_(std=0.1)
    return mlp


def _tiny_inference_mlp_config(**overrides):
    kwargs = dict(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=64,
        ttt_layers=[0],
        ttt_mode=True,
        ttt_proj=True,
        ttt_lr=0.5,
        ttt_chunk=2,
        ttt_target="input_embed",
    )
    kwargs.update(overrides)
    return InferenceQwen3Config(**kwargs)


def _build_inference_mlp(*, key_norm: bool, seed: int = 0) -> InferenceQwen3MLP:
    torch.manual_seed(seed)
    config = _tiny_inference_mlp_config(ttt_key_norm=key_norm)
    mlp = InferenceQwen3MLP(config, layer_idx=0)
    torch.manual_seed(seed + 1)
    with torch.no_grad():
        mlp.ttt_conv.weight.normal_(std=0.1)
        if mlp.ttt_proj is not None:
            mlp.ttt_proj.weight.normal_(std=0.1)
    return mlp


def test_key_norm_off_bitwise_equals_baseline():
    """ttt_key_norm=False must hit the verbatim original fused path."""
    mlp = _build_mlp(key_norm=False)
    assert not hasattr(mlp, "ttt_key_norm")

    torch.manual_seed(42)
    x = torch.randn(1, 6, 8)   # seq_len 6 > ttt_chunk 2 -> TTT path active
    t = torch.randn(1, 6, 8)

    out = mlp(x, t=t)
    assert out.shape == (1, 6, 8)
    assert torch.isfinite(out).all()

    # Recompute the original fused path independently and require bit equality.
    from einops import rearrange, repeat
    try:
        from opt_einsum import contract
    except ModuleNotFoundError:
        contract = torch.einsum

    h = mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)
    target_padded = mlp.padding(t)
    h_padded = mlp.padding(h)
    bs, chunk_num, chunk_size, _ = target_padded.shape
    t_conv = (
        mlp.ttt_conv(target_padded.transpose(-1, -2).reshape(bs * chunk_num, -1, chunk_size))
        .transpose(-1, -2)
        .reshape(bs, chunk_num, chunk_size, -1)
    )
    prediction_states = contract("b t c d, d e -> b t c e", t_conv, mlp.ttt_proj.weight)
    d_down_proj = contract("b t c h, b t c d -> b t d h", h_padded[:, :-1], prediction_states[:, :-1])
    delta_down_proj = d_down_proj * mlp.ttt_lr
    d_down_proj = torch.cat(
        [repeat(mlp.down_proj.weight, "d h -> b 1 d h", b=bs), delta_down_proj], dim=1
    )
    d_down_proj_sum = d_down_proj.cumsum(dim=1)
    down_proj = contract("b t d h, b t c h -> b t c d", d_down_proj_sum, h_padded)
    expected = rearrange(down_proj, "b t c d -> b (t c) d")[:, : x.shape[1], :]

    assert torch.equal(out, expected), "key_norm=False path must be bit-identical to baseline"


def test_key_norm_on_preserves_base_changes_delta():
    """ttt_key_norm=True: base path (W0 @ h) unchanged; delta path differs; finite."""
    mlp_off = _build_mlp(key_norm=False, seed=0)
    mlp_on = _build_mlp(key_norm=True, seed=0)
    # Copy the shared (non-key-norm) weights so the only difference is key-norm.
    mlp_on.load_state_dict(mlp_off.state_dict(), strict=False)
    assert hasattr(mlp_on, "ttt_key_norm")
    # RMSNorm weight is initialized to 1.0 by default — but variance
    # normalization still changes h, so the delta path must differ.

    torch.manual_seed(42)
    x = torch.randn(1, 6, 8)
    t = torch.randn(1, 6, 8)

    out_off = mlp_off(x, t=t)
    out_on = mlp_on(x, t=t)

    assert torch.isfinite(out_on).all()
    # The two outputs must differ (key-norm changed the delta path).
    assert not torch.allclose(out_on, out_off, atol=1e-6), "key-norm must change the delta readout"

    # The base component (W0 @ h, no delta) must be identical between the two:
    # take the first chunk's first token, where sum_{j<0} delta = 0, so output
    # is purely base. With ttt_chunk=2, token 0 sees no prior chunk delta.
    h = mlp_off.act_fn(mlp_off.gate_proj(x)) * mlp_off.up_proj(x)
    base_token0 = (h[:, :1] @ mlp_off.down_proj.weight.t())
    assert torch.allclose(out_off[:, :1], base_token0, atol=1e-5)
    assert torch.allclose(out_on[:, :1], base_token0, atol=1e-5), (
        "first-token base output must be unaffected by key-norm (no delta yet)"
    )


def test_inference_key_norm_on_preserves_base_changes_delta():
    """Inference MLP must honor ttt_key_norm the same way as training."""
    mlp_off = _build_inference_mlp(key_norm=False, seed=0)
    mlp_on = _build_inference_mlp(key_norm=True, seed=0)
    mlp_on.load_state_dict(mlp_off.state_dict(), strict=False)

    assert not hasattr(mlp_off, "ttt_key_norm")
    assert hasattr(mlp_on, "ttt_key_norm")

    torch.manual_seed(42)
    x = torch.randn(1, 6, 8)
    t = torch.randn(1, 6, 8)

    out_off, final_w_off = mlp_off(x, t=t)
    out_on, final_w_on = mlp_on(x, t=t)

    assert torch.isfinite(out_on).all()
    assert torch.isfinite(final_w_on).all()
    assert not torch.allclose(out_on, out_off, atol=1e-6), "key-norm must change inference TTT output"
    assert not torch.allclose(final_w_on, final_w_off, atol=1e-6), "key-norm must change inference fast weights"

    h = mlp_off.act_fn(mlp_off.gate_proj(x)) * mlp_off.up_proj(x)
    base_token0 = h[:, :1] @ mlp_off.down_proj.weight.t()
    assert torch.allclose(out_off[:, :1], base_token0, atol=1e-5)
    assert torch.allclose(out_on[:, :1], base_token0, atol=1e-5)
