"""ttt_lr warmup: pure factor fn + config flags + setter + backward compat."""
import pytest
import torch

from hf_models.hf_qwen3.configuration_qwen3 import Qwen3Config
from hf_models.hf_qwen3.modeling_qwen3 import Qwen3MLP


def _cfg(**ov):
    kw = dict(
        vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4, max_position_embeddings=64,
        ttt_layers=[0], ttt_mode=True, ttt_proj=True, ttt_lr=0.5, ttt_chunk=2,
        ttt_target="input_embed",
    )
    kw.update(ov)
    return Qwen3Config(**kw)


def test_config_warmup_defaults_and_validation():
    # 默认:无 warmup
    c = _cfg()
    assert c.ttt_lr_warmup_steps == 0
    assert c.ttt_lr_warmup_init == 0.0
    # 合法值
    c2 = _cfg(ttt_lr_warmup_steps=200, ttt_lr_warmup_init=1e-3)
    assert c2.ttt_lr_warmup_steps == 200
    assert c2.ttt_lr_warmup_init == 1e-3
    # 非法值
    with pytest.raises(ValueError):
        _cfg(ttt_lr_warmup_steps=-1)
    with pytest.raises(ValueError):
        _cfg(ttt_lr_warmup_init=-0.5)


def test_mlp_ttt_lr_effective_defaults_to_ttt_lr():
    mlp = Qwen3MLP(_cfg(ttt_lr=0.5), layer_idx=0)
    assert mlp.ttt_lr_effective == mlp.ttt_lr == 0.5


def _randomize(mlp):
    with torch.no_grad():
        for p in mlp.parameters():
            p.normal_(0.0, 0.1)
    return mlp.eval()


def test_forward_unchanged_when_effective_equals_target():
    """ttt_lr_effective 默认 == ttt_lr,forward 必须与改动前一致(用 outer 与 nlms 各验)。"""
    for rule in ("outer", "nlms"):
        torch.manual_seed(7)
        mlp = _randomize(Qwen3MLP(_cfg(ttt_write_rule=rule, ttt_lr=0.5, ttt_chunk=2), layer_idx=0))
        x = torch.randn(1, 4, 8); t = torch.randn(1, 4, 8)
        with torch.no_grad():
            out_default = mlp(x, t=t)
        mlp.ttt_lr_effective = mlp.ttt_lr
        with torch.no_grad():
            out_explicit = mlp(x, t=t)
        torch.testing.assert_close(out_default, out_explicit)


def test_forward_scales_with_effective_lr():
    """改 ttt_lr_effective 必须改变 NLMS 写入(证明 forward 真的用了它而非 ttt_lr)。"""
    torch.manual_seed(7)
    mlp = _randomize(Qwen3MLP(_cfg(ttt_write_rule="nlms", ttt_lr=0.5, ttt_chunk=2), layer_idx=0))
    x = torch.randn(1, 4, 8); t = torch.randn(1, 4, 8)
    with torch.no_grad():
        out_full = mlp(x, t=t)
        mlp.ttt_lr_effective = 0.0
        out_zero = mlp(x, t=t)
    assert not torch.allclose(out_full, out_zero), "forward must respond to ttt_lr_effective"
