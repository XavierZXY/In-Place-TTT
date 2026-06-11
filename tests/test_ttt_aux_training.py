import torch
from torch import nn

from in_place_ttt.ttt_aux.training import (
    accumulate_ttt_aux_grads,
    build_ttt_optimizer_param_groups,
    pop_ttt_monitor_stats,
)


class _FakeMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.ttt_conv = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.ttt_proj = nn.Linear(1, 1, bias=False)
        self.down_proj = nn.Linear(1, 1, bias=False)


class _FakeLayer(nn.Module):
    def __init__(self, is_ttt_layer=True):
        super().__init__()
        self.is_ttt_layer = is_ttt_layer
        self.mlp = _FakeMlp()


class _FakeInnerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(True), _FakeLayer(False)])


class _FakeCausalLm(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _FakeInnerModel()


def test_accumulate_ttt_aux_grads_updates_only_ttt_aux_params():
    model = _FakeCausalLm()
    ttt_layer = model.model.layers[0]
    non_ttt_layer = model.model.layers[1]

    aux_loss = (
        ttt_layer.mlp.ttt_conv.weight.sum()
        + 2.0 * ttt_layer.mlp.ttt_proj.weight.sum()
        + 3.0 * non_ttt_layer.mlp.ttt_conv.weight.sum()
        + 4.0 * ttt_layer.mlp.down_proj.weight.sum()
    )

    scaled = accumulate_ttt_aux_grads(model, aux_loss, aux_loss_weight=0.5, loss_scale=0.25)

    assert scaled is not None
    assert torch.allclose(scaled, aux_loss.detach() * 0.125)
    assert torch.allclose(ttt_layer.mlp.ttt_conv.weight.grad, torch.full_like(ttt_layer.mlp.ttt_conv.weight, 0.125))
    assert torch.allclose(ttt_layer.mlp.ttt_proj.weight.grad, torch.full_like(ttt_layer.mlp.ttt_proj.weight, 0.25))
    assert non_ttt_layer.mlp.ttt_conv.weight.grad is None
    assert ttt_layer.mlp.down_proj.weight.grad is None


def test_build_ttt_optimizer_param_groups_is_opt_in():
    model = _FakeCausalLm()

    assert build_ttt_optimizer_param_groups(model, base_lr=1e-4, base_weight_decay=0.1) is None


def test_build_ttt_optimizer_param_groups_splits_only_ttt_aux_params():
    model = _FakeCausalLm()
    ttt_layer = model.model.layers[0]
    non_ttt_layer = model.model.layers[1]

    groups = build_ttt_optimizer_param_groups(
        model,
        base_lr=1e-4,
        base_weight_decay=0.1,
        lr_multiplier=3.0,
        weight_decay=0.0,
    )

    assert groups is not None
    assert len(groups) == 2
    assert groups[0]["lr"] == 1e-4
    assert groups[0]["weight_decay"] == 0.1
    assert abs(groups[1]["lr"] - 3e-4) < 1e-12
    assert groups[1]["weight_decay"] == 0.0

    ttt_param_ids = {id(param) for param in groups[1]["params"]}
    assert id(ttt_layer.mlp.ttt_conv.weight) in ttt_param_ids
    assert id(ttt_layer.mlp.ttt_proj.weight) in ttt_param_ids
    assert id(non_ttt_layer.mlp.ttt_conv.weight) not in ttt_param_ids
    assert id(ttt_layer.mlp.down_proj.weight) not in ttt_param_ids


def test_pop_ttt_monitor_stats_averages_ttt_layers_and_clears_values():
    model = _FakeCausalLm()
    ttt_mlp = model.model.layers[0].mlp
    non_ttt_mlp = model.model.layers[1].mlp
    ttt_mlp._last_ttt_monitor_stats = {
        "delta_weight_sample_ratio": torch.tensor(0.2),
        "output_delta_sample_ratio": torch.tensor(0.4),
    }
    non_ttt_mlp._last_ttt_monitor_stats = {
        "delta_weight_sample_ratio": torch.tensor(10.0),
        "output_delta_sample_ratio": torch.tensor(10.0),
    }

    stats = pop_ttt_monitor_stats(model)

    assert torch.allclose(stats["delta_weight_sample_ratio"], torch.tensor(0.2))
    assert torch.allclose(stats["output_delta_sample_ratio"], torch.tensor(0.4))
    assert ttt_mlp._last_ttt_monitor_stats is None
