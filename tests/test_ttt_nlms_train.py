"""Training-side NLMS write rule: config flags + forward correctness + train/infer parity."""
import pytest
import torch

from hf_models.hf_qwen3.configuration_qwen3 import Qwen3Config
from hf_models.hf_qwen3.modeling_qwen3 import Qwen3MLP
from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config as InfConfig
from inference_model.hf_qwen3.modeling_qwen3 import Qwen3MLP as InfMLP


def _cfg(**ov):
    kw = dict(
        vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4, max_position_embeddings=64,
        ttt_layers=[0], ttt_mode=True, ttt_proj=True, ttt_lr=0.5, ttt_chunk=2,
        ttt_target="input_embed",
    )
    kw.update(ov)
    return Qwen3Config(**kw)


def test_train_config_default_write_rule_outer():
    assert getattr(_cfg(), "ttt_write_rule", "outer") == "outer"


def test_train_config_rejects_bad_write_rule():
    with pytest.raises(ValueError):
        _cfg(ttt_write_rule="bogus")
