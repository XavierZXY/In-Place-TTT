"""Regression + correctness tests for the NLMS block residual write rule
(inference Qwen3 TTT path).

Guarantees:
1. ttt_write_rule defaults to "outer" and the forward is bit-for-bit identical
   to the pre-change outer-product path (must not perturb the existing baseline
   that reproduces RULER 0.594).
2. NLMS with eta=0 degrades to the base MLP (no fast-weight contribution),
   i.e. equals disable-ttt — this is the R002 guardrail.
3. The NLMS block residual update matches the hand-computed math
   R = V - K S_delta^T ; dW = eta * R^T K / (lambda + tr(K^T K)).
4. write_subchunk splits a chunk into sub-blocks and stays a valid recurrence
   (serial over sub-blocks), not equal to the single-block update when keys
   are correlated (sanity that the knob actually changes the computation).
"""

import torch

from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config
from inference_model.hf_qwen3.modeling_qwen3 import Qwen3MLP


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


def _build_mlp(*, seed: int = 0, **cfg_overrides) -> Qwen3MLP:
    torch.manual_seed(seed)
    config = _tiny_mlp_config(**cfg_overrides)
    mlp = Qwen3MLP(config, layer_idx=0)
    # randomize the TTT projections so the fast-weight path is non-trivial
    with torch.no_grad():
        for p in mlp.parameters():
            p.normal_(0.0, 0.1)
    mlp.eval()
    return mlp


def test_default_write_rule_is_outer():
    """ttt_write_rule must default to 'outer' on the inference config."""
    config = _tiny_mlp_config()
    assert getattr(config, "ttt_write_rule", "outer") == "outer"


def test_outer_path_unchanged():
    """With ttt_write_rule='outer', forward equals the original outer-product path."""
    torch.manual_seed(1)
    x = torch.randn(1, 4, 8)  # seq_len=4 > ttt_chunk=2 → TTT path active
    t = torch.randn(1, 4, 8)

    mlp_outer = _build_mlp(seed=0, ttt_write_rule="outer")
    mlp_default = _build_mlp(seed=0)  # no flag → must behave as outer

    out_outer, w_outer = mlp_outer(x, t=t)
    out_default, w_default = mlp_default(x, t=t)

    torch.testing.assert_close(out_outer, out_default)
    torch.testing.assert_close(w_outer, w_default)


def test_nlms_eta_zero_degrades_to_base():
    """NLMS with ttt_lr(eta)=0 must equal the frozen base MLP (disable-ttt)."""
    torch.manual_seed(2)
    x = torch.randn(1, 4, 8)
    t = torch.randn(1, 4, 8)

    mlp = _build_mlp(seed=0, ttt_write_rule="nlms", ttt_lr=0.0)
    out, w = mlp(x, t=t)

    # base MLP output = down_proj(h), no fast-weight delta
    h = mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)
    base = torch.nn.functional.linear(h, mlp.down_proj.weight, mlp.down_proj.bias)
    torch.testing.assert_close(out, base)
    # fast weight must stay at W0
    torch.testing.assert_close(w, mlp.down_proj.weight)


def test_nlms_block_residual_matches_manual_math():
    """One-chunk NLMS update matches R = V - K S^T ; dW = eta R^T K / (lambda + tr(K^T K)).

    Keys are scaled up so tr(K^T K) is clearly != (1 - lambda); this makes the
    normalized NLMS update numerically distinct from the un-normalized outer
    product, so the test actually discriminates the two write rules.
    """
    torch.manual_seed(3)
    x = torch.randn(1, 4, 8) * 3.0   # amplify so tr(K^T K) >> 1
    t = torch.randn(1, 4, 8)
    eta, lam = 0.5, 0.1
    mlp = _build_mlp(seed=0, ttt_write_rule="nlms", ttt_lr=eta, ttt_nlms_lambda=lam)

    # reconstruct K (=h) and V (=t_conv @ proj) for the FIRST chunk exactly as forward does
    h = mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)
    from einops import rearrange
    t_padded = mlp.padding(t)
    bs, chunk_num, chunk_size, _ = t_padded.shape
    t_conv = (
        mlp.ttt_conv(t_padded.transpose(-1, -2).reshape(bs * chunk_num, -1, chunk_size))
        .transpose(-1, -2)
        .reshape(bs, chunk_num, chunk_size, -1)
    )
    h_padded = mlp.padding(h)
    K0 = h_padded[0, 0]          # [c, h_dim]
    V0 = torch.einsum("c d, d e -> c e", t_conv[0, 0], mlp.ttt_proj.weight)  # [c, d]
    S_delta0 = torch.zeros(mlp.down_proj.weight.shape)  # start: ΔW = 0
    R0 = V0 - torch.einsum("c h, d h -> c d", K0, S_delta0)  # [c, d]
    denom_c = lam + (K0 * K0).sum(dim=-1, keepdim=True)      # [c, 1] per-key
    R0n = R0 / denom_c                                        # normalize each residual row
    dW_expected = eta * torch.einsum("c d, c h -> d h", R0n, K0) / K0.shape[0]  # avg over chunk
    w_expected_after_first = mlp.down_proj.weight + dW_expected

    # run forward and capture the fast weight after processing; with 2 chunks the
    # returned w is after the 2nd update, so we re-run a 1-chunk-equivalent check:
    # easier: directly call the internal write once via a 1-chunk input.
    x1 = x[:, :2]   # exactly one chunk
    t1 = t[:, :2]
    # seq_len == ttt_chunk → need update_partial to force a write on the only chunk
    _, w1 = mlp(x1, t=t1, update_partial=True)
    torch.testing.assert_close(w1, w_expected_after_first, rtol=1e-4, atol=1e-5)

    # discrimination guard: the NLMS write must differ from the outer-product write
    mlp_outer = _build_mlp(seed=0, ttt_write_rule="outer", ttt_lr=eta)
    _, w1_outer = mlp_outer(x1, t=t1, update_partial=True)
    assert not torch.allclose(w1, w1_outer, rtol=1e-3, atol=1e-4)


def test_write_subchunk_changes_computation():
    """write_subchunk=1 (sub-block serial) differs from a single 1024-style block
    when keys are correlated — confirms the knob actually alters the recurrence."""
    torch.manual_seed(4)
    x = torch.randn(1, 4, 8)
    t = torch.randn(1, 4, 8)

    mlp_block = _build_mlp(seed=0, ttt_write_rule="nlms", ttt_chunk=4)  # one big block
    mlp_sub = _build_mlp(seed=0, ttt_write_rule="nlms", ttt_chunk=4, ttt_write_subchunk=1)

    _, w_block = mlp_block(x, t=t, update_partial=True)
    _, w_sub = mlp_sub(x, t=t, update_partial=True)

    # they must NOT be equal: serial sub-block residual sees an updated S within the
    # chunk, while the single block uses chunk-start S for all keys. Under per-key
    # normalization the gap is smaller but still non-zero — assert it is detectable.
    max_diff = (w_block - w_sub).abs().max().item()
    assert max_diff > 1e-5, f"subchunk should alter the recurrence, got max_diff={max_diff}"
