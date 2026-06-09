import torch

from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config
from inference_model.hf_qwen3.modeling_qwen3 import Qwen3MLP, Qwen3Model


def _tiny_inference_qwen3_config(**overrides):
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
        ttt_proj=False,
        ttt_lr=0.5,
        ttt_chunk=4,
        ttt_target="input_embed",
    )
    kwargs.update(overrides)
    return Qwen3Config(**kwargs)


def test_qwen3_inference_mlp_can_update_partial_chunk_weight():
    torch.manual_seed(0)
    config = _tiny_inference_qwen3_config()
    mlp = Qwen3MLP(config, layer_idx=0)

    with torch.no_grad():
        mlp.ttt_conv.weight.zero_()
        mlp.ttt_conv.weight[:, 0, 2] = 1.0

    x = torch.randn(1, 6, config.hidden_size)
    t = torch.randn(1, 6, config.hidden_size)
    _, full_chunk_only_w = mlp(x, t, update_partial=False)
    _, partial_w = mlp(x, t, update_partial=True)

    assert not torch.allclose(full_chunk_only_w, partial_w)


def test_qwen3_prefill_partial_update_clears_cached_tail():
    torch.manual_seed(0)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])

    model_without_flush = Qwen3Model(_tiny_inference_qwen3_config(ttt_prefill_update_partial=False))
    outputs_without_flush = model_without_flush(input_ids=input_ids, use_cache=True)
    tail_without_flush, _, _ = outputs_without_flush.past_key_values.ttt_states[0]
    assert tail_without_flush.shape[1] == 2

    model_with_flush = Qwen3Model(_tiny_inference_qwen3_config(ttt_prefill_update_partial=True))
    outputs_with_flush = model_with_flush(input_ids=input_ids, use_cache=True)
    tail_with_flush, target_tail_with_flush, _ = outputs_with_flush.past_key_values.ttt_states[0]
    assert tail_with_flush is None
    assert target_tail_with_flush is None


def test_qwen3_prefill_partial_update_handles_short_prefill():
    torch.manual_seed(0)
    input_ids = torch.tensor([[1, 2, 3]])

    model = Qwen3Model(_tiny_inference_qwen3_config(ttt_prefill_update_partial=True))
    outputs = model(input_ids=input_ids, use_cache=True)

    hidden_tail, target_tail, present_w = outputs.past_key_values.ttt_states[0]
    assert hidden_tail is None
    assert target_tail is None
    assert present_w is not None
