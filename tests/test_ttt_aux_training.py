import torch
from torch import nn

from in_place_ttt.ttt_aux.training import accumulate_ttt_aux_grads


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
