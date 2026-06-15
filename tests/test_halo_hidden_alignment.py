from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from tasks.halo_hidden_alignment import (
    HiddenAlignmentOrchestrator,
    compute_hidden_alignment_loss,
    resolve_hidden_align_layers,
)


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


def test_hidden_alignment_nmse_is_scale_invariant():
    student = {1: torch.tensor([[[2.0, 4.0], [100.0, 100.0]]])}
    teacher = {1: torch.tensor([[[1.0, 2.0], [0.0, 0.0]]])}
    attention_mask = torch.tensor([[1, 0]])

    loss = compute_hidden_alignment_loss(
        student,
        teacher,
        attention_mask=attention_mask,
        loss_fn="nmse",
    )
    # ||s - t||^2 / ||t||^2 = (1 + 4) / (1 + 4) = 1, regardless of scale.
    assert torch.allclose(loss, torch.tensor(1.0), atol=1e-5)

    scaled_loss = compute_hidden_alignment_loss(
        {1: student[1] * 10.0},
        {1: teacher[1] * 10.0},
        attention_mask=attention_mask,
        loss_fn="nmse",
    )
    assert torch.allclose(scaled_loss, loss, atol=1e-5)


class _DummyBody(nn.Module):
    def __init__(self, layer_scales, layer_cls=nn.Linear):
        super().__init__()
        layers = []
        for scale in layer_scales:
            linear = layer_cls(4, 4, bias=False)
            with torch.no_grad():
                linear.weight.copy_(torch.eye(4) * scale)
            layers.append(linear)
        self.layers = nn.ModuleList(layers)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        hidden = input_ids
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden, hidden_states=None)


class _DummyLM(nn.Module):
    def __init__(self, layer_scales, layer_cls=nn.Linear):
        super().__init__()
        self.model = _DummyBody(layer_scales, layer_cls=layer_cls)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)


def test_orchestrator_teacher_input_isolates_upstream_errors():
    torch.manual_seed(0)
    inputs = torch.randn(1, 3, 4)
    teacher = _DummyLM([1.0, 1.0])
    # Layer 0 deviates from the teacher, layer 1 is identical.
    student = _DummyLM([2.0, 1.0])

    student_mode = HiddenAlignmentOrchestrator(
        student=student, teacher=teacher, layer_idxs=[0, 1], loss_fn="nmse", align_input="student"
    ).forward(input_ids=inputs)
    teacher_mode = HiddenAlignmentOrchestrator(
        student=student, teacher=teacher, layer_idxs=[0, 1], loss_fn="nmse", align_input="teacher"
    ).forward(input_ids=inputs)

    # Layer 0's own error is visible in both modes.
    assert student_mode.layer_losses[0].item() > 0.1
    assert teacher_mode.layer_losses[0].item() > 0.1
    # In student mode layer 0's error propagates into layer 1; with teacher
    # inputs layer 1 (identical weights) must show zero local error.
    assert student_mode.layer_losses[1].item() > 0.1
    assert teacher_mode.layer_losses[1].item() < 1e-10


class _CheckpointedLinear(nn.Linear):
    """Mimics HF GradientCheckpointingLayer: module hooks run INSIDE the
    checkpointed function, so backward recompute re-fires them and must see
    the same inputs as the original forward."""

    def __call__(self, *args, **kwargs):
        if torch.is_grad_enabled():
            return torch_checkpoint(super().__call__, *args, use_reentrant=False, **kwargs)
        return super().__call__(*args, **kwargs)


def test_teacher_input_mode_survives_gradient_checkpointing():
    torch.manual_seed(0)
    inputs = torch.randn(1, 3, 4)
    teacher = _DummyLM([1.0, 1.0])
    student = _DummyLM([2.0, 1.0], layer_cls=_CheckpointedLinear)

    orchestrator = HiddenAlignmentOrchestrator(
        student=student, teacher=teacher, layer_idxs=[0, 1], loss_fn="nmse", align_input="teacher"
    )
    output = orchestrator.forward(input_ids=inputs)
    # Backward triggers checkpoint recompute, which re-runs the pre-hooks;
    # they must still serve the same teacher inputs (no CheckpointError).
    output.loss.backward()

    assert student.model.layers[0].weight.grad.norm().item() > 0.01
    assert student.model.layers[1].weight.grad.norm().item() < 1e-8

    # Under no_grad the overrides are dropped right after the forward, so a
    # plain student call afterwards is NOT teacher-forced.
    with torch.no_grad():
        orchestrator.forward(input_ids=inputs)
        assert not orchestrator._student_input_overrides
        plain = student(input_ids=inputs).last_hidden_state
        assert torch.allclose(plain, inputs * 2.0)
