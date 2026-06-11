import torch

from hf_models.hf_qwen3.configuration_qwen3 import Qwen3Config
from hf_models.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3Model
from in_place_ttt.ttt_aux.training import compute_ttt_logits_delta_sample_ratio


def _tiny_qwen3_config(**overrides):
    kwargs = dict(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=3,
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
        ttt_compress_window=4,
        full_attention_layers=[1],
        ttt_aux_loss_weight=0.1,
    )
    kwargs.update(overrides)
    return Qwen3Config(**kwargs)


def test_ttt_compress_window_overrides_qwen3_layer_types_before_layer_construction():
    config = _tiny_qwen3_config()

    model = Qwen3Model(config)

    assert model.config.use_sliding_window is True
    assert model.config.sliding_window == 4
    assert model.config.layer_types == ["sliding_attention", "full_attention", "sliding_attention"]
    assert model.layers[0].self_attn.sliding_window == 4
    assert model.layers[1].self_attn.sliding_window is None
    assert model.layers[2].self_attn.sliding_window == 4


def test_qwen3_model_exposes_ttt_aux_loss_when_training_with_input_embed_target():
    torch.manual_seed(0)
    config = _tiny_qwen3_config(num_hidden_layers=1, full_attention_layers=[])
    model = Qwen3Model(config)
    model.train()

    outputs = model(input_ids=torch.tensor([[1, 2, 3, 4]]), use_cache=False)

    assert outputs.ttt_aux_loss is not None
    assert outputs.ttt_aux_loss.requires_grad
    assert torch.isfinite(outputs.ttt_aux_loss)
    assert model._last_ttt_aux_loss is outputs.ttt_aux_loss


def test_qwen3_model_exposes_future_chunk_hidden_ttt_aux_loss():
    torch.manual_seed(0)
    config = _tiny_qwen3_config(
        num_hidden_layers=1,
        full_attention_layers=[],
        ttt_target="hidden_states",
        ttt_aux_target="future_chunk_hidden",
        ttt_aux_future_chunks=1,
    )
    model = Qwen3Model(config)
    model.train()

    outputs = model(input_ids=torch.tensor([[1, 2, 3, 4, 5, 6]]), use_cache=False)

    assert outputs.ttt_aux_loss is not None
    assert outputs.ttt_aux_loss.requires_grad
    assert torch.isfinite(outputs.ttt_aux_loss)
    assert model._last_ttt_aux_loss is outputs.ttt_aux_loss


def test_qwen3_causal_lm_keeps_ttt_aux_loss_separate_from_main_loss():
    torch.manual_seed(0)
    config = _tiny_qwen3_config(num_hidden_layers=1, full_attention_layers=[])
    model = Qwen3ForCausalLM(config)
    model.train()
    input_ids = torch.tensor([[1, 2, 3, 4]])

    outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)

    assert outputs.loss is not None
    assert outputs.ttt_aux_loss is not None
    assert outputs.ttt_aux_loss is model.model._last_ttt_aux_loss
    assert outputs.ttt_aux_loss is not outputs.loss


def test_qwen3_logits_delta_monitor_compares_ttt_on_and_off():
    torch.manual_seed(0)
    config = _tiny_qwen3_config(
        num_hidden_layers=1,
        full_attention_layers=[],
        ttt_aux_loss_weight=0.0,
    )
    model = Qwen3ForCausalLM(config)
    model.train()
    mlp = model.model.layers[0].mlp
    mlp.ttt_conv.weight.data.fill_(0.1)
    mlp.ttt_proj.weight.data.copy_(torch.eye(config.hidden_size))

    ratio = compute_ttt_logits_delta_sample_ratio(
        model,
        {"input_ids": torch.tensor([[1, 2, 3, 4]])},
        sample_tokens=2,
        sample_dim=16,
    )

    assert ratio is not None
    assert torch.isfinite(ratio)
    assert ratio > 0
    assert model.training
    assert model.model.layers[0].is_ttt_layer
    assert not hasattr(mlp, "_last_ttt_monitor_stats")
