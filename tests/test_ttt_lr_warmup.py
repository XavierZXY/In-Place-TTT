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
