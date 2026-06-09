from types import SimpleNamespace

import torch

from tasks.halo_hidden_alignment import compute_hidden_alignment_loss, resolve_hidden_align_layers


def test_resolve_hidden_align_layers_defaults_to_converted_layers():
    config = SimpleNamespace(
        num_hidden_layers=6,
        layer_types=[
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
        ],
        ttt_layers=[2],
    )

    assert resolve_hidden_align_layers(config, None, skip_ttt_layers=True) == [1, 4, 5]


def test_hidden_alignment_mse_ignores_padding_tokens():
    student = {1: torch.tensor([[[1.0, 3.0], [100.0, 100.0]]])}
    teacher = {1: torch.zeros(1, 2, 2)}
    attention_mask = torch.tensor([[1, 0]])

    loss = compute_hidden_alignment_loss(
        student,
        teacher,
        attention_mask=attention_mask,
        loss_fn="mse",
    )

    assert torch.allclose(loss, torch.tensor(5.0))
