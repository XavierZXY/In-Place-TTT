"""HALO-style teacher-student KD helpers for the local VeOmni trainer.

The implementation follows HALO Stage 2 semantics: a frozen full-attention
teacher supervises the hybrid student with KL divergence. For long contexts, it
computes the LM-head logits in token chunks instead of materializing full
``[batch, seq, vocab]`` tensors.
"""

from dataclasses import dataclass
from types import MethodType
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class HaloKDOutput:
    loss: torch.Tensor
    loss_ce: torch.Tensor
    loss_kl: torch.Tensor


def _full_weight(weight: torch.Tensor) -> torch.Tensor:
    if hasattr(weight, "full_tensor"):
        return weight.full_tensor()
    return weight


def _chunked_ce_kl_loss(
    student_hidden: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_hidden: Optional[torch.Tensor],
    teacher_weight: Optional[torch.Tensor],
    labels: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    alpha_ce: float,
    alpha_kl: float,
    temperature: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ce_sum = student_hidden.new_tensor(0.0, dtype=torch.float32)
    kl_sum = student_hidden.new_tensor(0.0, dtype=torch.float32)
    ce_denom = student_hidden.new_tensor(1.0, dtype=torch.float32)
    temp = float(temperature)
    chunk_size = int(chunk_size)

    if alpha_ce > 0.0:
        if labels is None:
            raise ValueError("labels are required when alpha_ce > 0")
        ce_student_flat = student_hidden[:, :-1, :].contiguous().view(-1, student_hidden.shape[-1])
        ce_label_flat = labels[:, 1:].contiguous().view(-1)
        ce_valid = ce_label_flat.ne(-100)
        ce_denom = ce_valid.sum().clamp_min(1).to(torch.float32)

        for start in range(0, ce_student_flat.shape[0], chunk_size):
            end = min(start + chunk_size, ce_student_flat.shape[0])
            labels_chunk = ce_label_flat[start:end]
            valid_chunk = ce_valid[start:end]
            if not torch.any(valid_chunk):
                continue

            student_logits = F.linear(ce_student_flat[start:end], student_weight).float()
            ce_sum = ce_sum + F.cross_entropy(
                student_logits,
                labels_chunk,
                ignore_index=-100,
                reduction="sum",
            )

    if alpha_kl > 0.0:
        if teacher_hidden is None or teacher_weight is None:
            raise ValueError("teacher_hidden and teacher_weight are required when alpha_kl > 0")

        kl_student_flat = student_hidden.contiguous().view(-1, student_hidden.shape[-1])
        kl_teacher_flat = teacher_hidden.contiguous().view(-1, teacher_hidden.shape[-1])
        if attention_mask is None:
            kl_valid = torch.ones(kl_student_flat.shape[0], device=kl_student_flat.device, dtype=torch.bool)
        else:
            kl_valid = attention_mask.contiguous().view(-1).to(device=kl_student_flat.device).bool()
        kl_denom = kl_valid.sum().clamp_min(1).to(torch.float32)

        for start in range(0, kl_student_flat.shape[0], chunk_size):
            end = min(start + chunk_size, kl_student_flat.shape[0])
            valid_chunk = kl_valid[start:end]
            if not torch.any(valid_chunk):
                continue

            student_logits = F.linear(kl_student_flat[start:end], student_weight).float()
            with torch.no_grad():
                teacher_logits = F.linear(
                    kl_teacher_flat[start:end].to(kl_student_flat.dtype),
                    teacher_weight.to(kl_student_flat.dtype),
                ).float()
                teacher_log_prob = F.log_softmax(teacher_logits[valid_chunk] / temp, dim=-1)
            student_log_prob = F.log_softmax(student_logits[valid_chunk] / temp, dim=-1)
            kl_sum = kl_sum + F.kl_div(
                student_log_prob,
                teacher_log_prob,
                log_target=True,
                reduction="sum",
            ) * (temp * temp)
    else:
        kl_denom = ce_denom

    loss_ce = ce_sum / ce_denom
    loss_kl = kl_sum / kl_denom
    loss = float(alpha_ce) * loss_ce + float(alpha_kl) * loss_kl
    return loss, loss_ce.detach(), loss_kl.detach()


def _halo_kd_forward(
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
    halo_kd_return_hidden=False,
    halo_kd_teacher_hidden=None,
    halo_kd_teacher_lm_weight=None,
    halo_kd_alpha_ce=0.0,
    halo_kd_alpha_kl=0.0,
    halo_kd_temperature=1.0,
    halo_kd_chunk_size=128,
    **kwargs,
):
    if halo_kd_return_hidden:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )
        out = CausalLMOutputWithPast(
            loss=None,
            logits=outputs.last_hidden_state,
            past_key_values=None,
            hidden_states=outputs.hidden_states,
            attentions=None,
        )
        out.ttt_aux_loss = getattr(outputs, "ttt_aux_loss", None)
        return out

    if halo_kd_teacher_hidden is None and halo_kd_alpha_kl <= 0.0:
        return self._halo_kd_original_forward(
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
            skip_logits=skip_logits,
            return_dict=return_dict,
            **kwargs,
        )

    if labels is None and halo_kd_alpha_ce > 0.0:
        raise ValueError("labels are required when alpha_ce > 0")

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
        **kwargs,
    )

    loss, loss_ce, loss_kl = _chunked_ce_kl_loss(
        student_hidden=outputs.last_hidden_state,
        student_weight=self.lm_head.weight,
        teacher_hidden=halo_kd_teacher_hidden,
        teacher_weight=halo_kd_teacher_lm_weight,
        labels=labels,
        attention_mask=attention_mask,
        alpha_ce=float(halo_kd_alpha_ce),
        alpha_kl=float(halo_kd_alpha_kl),
        temperature=float(halo_kd_temperature),
        chunk_size=int(halo_kd_chunk_size),
    )

    out = CausalLMOutputWithPast(
        loss=loss,
        logits=None,
        past_key_values=None,
        hidden_states=outputs.hidden_states,
        attentions=None,
    )
    out.loss_ce = loss_ce
    out.loss_kl = loss_kl
    out.ttt_aux_loss = getattr(outputs, "ttt_aux_loss", None)
    return out


def patch_model_for_halo_kd(model: nn.Module) -> None:
    if hasattr(model, "_halo_kd_original_forward"):
        return
    model._halo_kd_original_forward = model.forward
    model.forward = MethodType(_halo_kd_forward, model)


class HaloKDOrchestrator:
    """Plain orchestrator holding FSDP-managed student and teacher models."""

    def __init__(
        self,
        student: nn.Module,
        teacher: Optional[nn.Module],
        alpha_ce: float = 0.0,
        alpha_kl: float = 1.0,
        temperature: float = 1.0,
        chunk_size: int = 128,
    ):
        if alpha_ce <= 0.0 and alpha_kl <= 0.0:
            raise ValueError("At least one of alpha_ce or alpha_kl must be positive")
        if alpha_kl > 0.0 and teacher is None:
            raise ValueError("teacher is required when alpha_kl > 0")

        self.student = student
        self.teacher = teacher
        self.alpha_ce = float(alpha_ce)
        self.alpha_kl = float(alpha_kl)
        self.temperature = float(temperature)
        self.chunk_size = int(chunk_size)

        patch_model_for_halo_kd(self.student)
        if self.teacher is not None:
            patch_model_for_halo_kd(self.teacher)
            for param in self.teacher.parameters():
                param.requires_grad_(False)
            self.teacher.eval()

    @property
    def config(self):
        return self.student.config

    def train(self, mode: bool = True):
        self.student.train(mode)
        if self.teacher is not None:
            self.teacher.eval()
        return self

    def eval(self):
        return self.train(False)

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        for key in (
            "use_cache",
            "output_hidden_states",
            "labels",
            "halo_kd_return_hidden",
            "halo_kd_teacher_hidden",
            "halo_kd_teacher_lm_weight",
        ):
            kwargs.pop(key, None)

        teacher_hidden = None
        teacher_weight = None
        if self.alpha_kl > 0.0:
            with torch.no_grad():
                teacher_out = self.teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=False,
                    halo_kd_return_hidden=True,
                    **kwargs,
                )
            teacher_hidden = teacher_out.logits.detach()
            teacher_weight = _full_weight(self.teacher.lm_head.weight).detach()
            del teacher_out

        student_out = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            output_hidden_states=False,
            halo_kd_teacher_hidden=teacher_hidden,
            halo_kd_teacher_lm_weight=teacher_weight,
            halo_kd_alpha_ce=self.alpha_ce,
            halo_kd_alpha_kl=self.alpha_kl,
            halo_kd_temperature=self.temperature,
            halo_kd_chunk_size=self.chunk_size,
            **kwargs,
        )

        return HaloKDOutput(
            loss=student_out.loss,
            loss_ce=getattr(student_out, "loss_ce", student_out.loss.detach()),
            loss_kl=getattr(student_out, "loss_kl", student_out.loss.detach()),
        )
