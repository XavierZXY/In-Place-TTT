from collections.abc import Sequence

import torch
import torch.nn.functional as F


def _as_tensor_list(value: torch.Tensor | Sequence[torch.Tensor], name: str) -> list[torch.Tensor]:
    if torch.is_tensor(value):
        return [value]
    tensors = list(value)
    if not tensors:
        raise ValueError(f"{name} must contain at least one tensor")
    if not all(torch.is_tensor(tensor) for tensor in tensors):
        raise TypeError(f"{name} must be a tensor or a sequence of tensors")
    return tensors


def _validate_pairs(predictions: list[torch.Tensor], targets: list[torch.Tensor]) -> None:
    if len(predictions) != len(targets):
        raise ValueError(
            f"predictions and targets must have the same length, got {len(predictions)} and {len(targets)}"
        )
    for idx, (prediction, target) in enumerate(zip(predictions, targets)):
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction and target at index {idx} must have the same shape, "
                f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
            )
        if prediction.ndim < 2:
            raise ValueError(f"prediction at index {idx} must have at least batch and token dimensions")


def compute_ttt_aux_loss(
    predictions: torch.Tensor | Sequence[torch.Tensor],
    targets: torch.Tensor | Sequence[torch.Tensor],
    *,
    loss_type: str = "jepa",
    loss_exp: float = 1.0,
    reg_coeff: float = 0.0,
    reg_eps: float = 0.0001,
) -> torch.Tensor:
    """Compute auxiliary supervision for TTT V-hat predictions."""
    prediction_list = _as_tensor_list(predictions, "predictions")
    target_list = _as_tensor_list(targets, "targets")
    _validate_pairs(prediction_list, target_list)

    if loss_type == "jepa":
        return _compute_jepa_loss(prediction_list, target_list, loss_exp, reg_coeff, reg_eps)
    if loss_type == "cosine":
        return _compute_cosine_loss(prediction_list, target_list)
    raise ValueError(f"unknown TTT aux loss_type {loss_type!r}; expected 'jepa' or 'cosine'")


def _compute_jepa_loss(
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
    loss_exp: float,
    reg_coeff: float,
    reg_eps: float,
) -> torch.Tensor:
    if loss_exp <= 0.0:
        raise ValueError(f"loss_exp must be positive, got {loss_exp}")
    if reg_coeff < 0.0:
        raise ValueError(f"reg_coeff must be non-negative, got {reg_coeff}")
    if reg_eps < 0.0:
        raise ValueError(f"reg_eps must be non-negative, got {reg_eps}")

    pred_loss = predictions[0].new_zeros((), dtype=torch.float32)
    pstd = predictions[0].new_zeros(predictions[0].shape[0], predictions[0].shape[-1], dtype=torch.float32)
    for prediction, target in zip(predictions, targets):
        prediction_fp32 = prediction.float()
        target_fp32 = target.detach().float()
        pred_loss = pred_loss + torch.mean(torch.abs(prediction_fp32 - target_fp32).pow(loss_exp)) / loss_exp
        pstd = pstd + torch.sqrt(prediction_fp32.var(dim=1, unbiased=False) + reg_eps)

    pred_loss = pred_loss / len(predictions)
    if reg_coeff == 0.0:
        return pred_loss

    pstd = pstd / len(predictions)
    reg_loss = torch.mean(F.relu(1.0 - pstd))
    return pred_loss + reg_coeff * reg_loss


def _compute_cosine_loss(
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
) -> torch.Tensor:
    loss = predictions[0].new_zeros((), dtype=torch.float32)
    for prediction, target in zip(predictions, targets):
        prediction_fp32 = prediction.float()
        target_fp32 = target.detach().float()
        loss = loss - (F.normalize(prediction_fp32, dim=-1) * F.normalize(target_fp32, dim=-1)).sum(dim=-1).mean()
    return loss / len(predictions)
