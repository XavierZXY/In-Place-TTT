"""Hidden-state alignment helpers for HALO-style warmup training."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import Any, Iterator, Optional

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class HiddenAlignmentOutput:
    loss: torch.Tensor
    loss_hidden: torch.Tensor
    layer_count: int


def _model_layers(model: nn.Module) -> nn.ModuleList:
    body = getattr(model, "model", None)
    layers = getattr(body, "layers", None)
    if layers is None:
        raise AttributeError("hidden alignment expects a CausalLM model with model.layers")
    return layers


def resolve_hidden_align_layers(
    config: Any,
    requested_layers: Optional[list[int] | str],
    *,
    skip_ttt_layers: bool = False,
) -> list[int]:
    num_layers = int(getattr(config, "num_hidden_layers"))

    if requested_layers in (None, "", "converted", "auto"):
        layer_types = list(getattr(config, "layer_types", ["full_attention"] * num_layers))
        layers = [idx for idx, layer_type in enumerate(layer_types) if layer_type != "full_attention"]
    elif requested_layers == "all":
        layers = list(range(num_layers))
    elif isinstance(requested_layers, str):
        layers = [int(item.strip()) for item in requested_layers.split(",") if item.strip()]
    else:
        layers = [int(layer_idx) for layer_idx in requested_layers]

    if skip_ttt_layers:
        ttt_layers = {int(layer_idx) for layer_idx in getattr(config, "ttt_layers", []) or []}
        layers = [layer_idx for layer_idx in layers if layer_idx not in ttt_layers]

    invalid = [layer_idx for layer_idx in layers if layer_idx < 0 or layer_idx >= num_layers]
    if invalid:
        raise ValueError(f"hidden_align_layers contains invalid layer indexes: {invalid}")
    if not layers:
        raise ValueError("hidden_align_layers resolved to an empty layer list")
    return layers


def configure_hidden_alignment_trainable_params(
    model: nn.Module,
    layer_idxs: list[int],
    *,
    train_scope: str = "layers",
) -> int:
    if train_scope not in {"all", "layers", "self_attn"}:
        raise ValueError("hidden_align_train_scope must be one of {'all', 'layers', 'self_attn'}")

    if train_scope == "all":
        for param in model.parameters():
            param.requires_grad_(True)
    else:
        for param in model.parameters():
            param.requires_grad_(False)

        layers = _model_layers(model)
        for layer_idx in layer_idxs:
            module = layers[layer_idx] if train_scope == "layers" else layers[layer_idx].self_attn
            module.requires_grad_(True)

    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _halo_hidden_align_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    cache_position=None,
    logits_to_keep=0,
    skip_logits=None,
    return_dict=None,
    halo_hidden_align_return_hidden=False,
    **kwargs,
):
    if halo_hidden_align_return_hidden:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            cache_position=cache_position,
            **kwargs,
        )
        out = CausalLMOutputWithPast(
            loss=None,
            logits=outputs.last_hidden_state,
            past_key_values=None,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=None,
        )
        out.ttt_aux_loss = getattr(outputs, "ttt_aux_loss", None)
        return out

    return self._halo_hidden_align_original_forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
        logits_to_keep=logits_to_keep,
        return_dict=return_dict,
        **kwargs,
    )


def patch_model_for_hidden_alignment(model: nn.Module) -> None:
    if hasattr(model, "_halo_hidden_align_original_forward"):
        return
    model._halo_hidden_align_original_forward = model.forward
    model.forward = MethodType(_halo_hidden_align_forward, model)


@contextmanager
def _capture_layer_outputs(
    model: nn.Module,
    layer_idxs: list[int],
    *,
    detach: bool,
) -> Iterator[dict[int, torch.Tensor]]:
    captures: dict[int, torch.Tensor] = {}
    handles = []

    def build_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captures[layer_idx] = hidden.detach() if detach else hidden

        return hook

    layers = _model_layers(model)
    for layer_idx in layer_idxs:
        handles.append(layers[layer_idx].register_forward_hook(build_hook(layer_idx)))

    try:
        yield captures
    finally:
        for handle in handles:
            handle.remove()


def _valid_token_mask(attention_mask: Optional[torch.Tensor], reference: torch.Tensor) -> Optional[torch.Tensor]:
    if attention_mask is None or attention_mask.ndim != 2:
        return None
    if tuple(attention_mask.shape) != tuple(reference.shape[:2]):
        return None
    return attention_mask.to(device=reference.device).bool()


def compute_hidden_alignment_loss(
    student_states: dict[int, torch.Tensor],
    teacher_states: dict[int, torch.Tensor],
    *,
    attention_mask: Optional[torch.Tensor] = None,
    loss_fn: str = "mse",
) -> torch.Tensor:
    if loss_fn not in {"mse", "l2norm"}:
        raise ValueError("hidden_align_loss_fn must be one of {'mse', 'l2norm'}")
    if set(student_states) != set(teacher_states):
        raise ValueError("student and teacher captured different hidden-alignment layers")

    losses = []
    for layer_idx in sorted(student_states):
        student = student_states[layer_idx]
        teacher = teacher_states[layer_idx].to(device=student.device, dtype=student.dtype)
        if student.shape != teacher.shape:
            raise ValueError(
                f"hidden state shape mismatch at layer {layer_idx}: "
                f"student={tuple(student.shape)}, teacher={tuple(teacher.shape)}"
            )

        valid_mask = _valid_token_mask(attention_mask, student)
        diff = student.float() - teacher.float()
        if loss_fn == "mse":
            per_dim_loss = diff.pow(2)
            if valid_mask is None:
                losses.append(per_dim_loss.mean())
            else:
                denom = valid_mask.sum().clamp_min(1).to(torch.float32) * per_dim_loss.shape[-1]
                losses.append((per_dim_loss * valid_mask.unsqueeze(-1)).sum() / denom)
        else:
            per_token_loss = torch.linalg.vector_norm(diff, dim=-1) * (diff.shape[-1] ** -0.5)
            if valid_mask is None:
                losses.append(per_token_loss.mean())
            else:
                denom = valid_mask.sum().clamp_min(1).to(torch.float32)
                losses.append((per_token_loss * valid_mask).sum() / denom)

    return torch.stack(losses).mean()


class HiddenAlignmentOrchestrator:
    """Runs a frozen full-attention teacher and aligns selected student layer outputs."""

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        *,
        layer_idxs: list[int],
        loss_fn: str = "mse",
    ):
        if not layer_idxs:
            raise ValueError("layer_idxs must contain at least one layer")

        self.student = student
        self.teacher = teacher
        self.layer_idxs = [int(layer_idx) for layer_idx in layer_idxs]
        self.loss_fn = loss_fn

        patch_model_for_hidden_alignment(self.student)
        patch_model_for_hidden_alignment(self.teacher)
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        self.teacher.eval()

    @property
    def config(self):
        return self.student.config

    def train(self, mode: bool = True):
        self.student.train(mode)
        self.teacher.eval()
        return self

    def eval(self):
        return self.train(False)

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs) -> HiddenAlignmentOutput:
        for key in (
            "labels",
            "use_cache",
            "output_hidden_states",
            "halo_hidden_align_return_hidden",
        ):
            kwargs.pop(key, None)

        with torch.no_grad(), _capture_layer_outputs(self.teacher, self.layer_idxs, detach=True) as teacher_states:
            self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                halo_hidden_align_return_hidden=True,
                **kwargs,
            )

        with _capture_layer_outputs(self.student, self.layer_idxs, detach=False) as student_states:
            self.student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                halo_hidden_align_return_hidden=True,
                **kwargs,
            )

        missing = [layer_idx for layer_idx in self.layer_idxs if layer_idx not in student_states or layer_idx not in teacher_states]
        if missing:
            raise RuntimeError(f"failed to capture hidden states for layers: {missing}")

        loss = compute_hidden_alignment_loss(
            student_states,
            teacher_states,
            attention_mask=attention_mask,
            loss_fn=self.loss_fn,
        )
        return HiddenAlignmentOutput(
            loss=loss,
            loss_hidden=loss.detach(),
            layer_count=len(self.layer_idxs),
        )
