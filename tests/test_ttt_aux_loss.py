import pytest
import torch
import torch.nn.functional as F

from in_place_ttt.ttt_aux.loss import compute_ttt_aux_loss


def test_jepa_prediction_loss_matches_reference_formula():
    pred = torch.tensor([[[1.0, -2.0], [3.0, -4.0]]])
    target = torch.tensor([[[0.5, -1.0], [1.0, -1.5]]])

    loss = compute_ttt_aux_loss(pred, target, loss_type="jepa", loss_exp=1.0)

    expected = torch.mean(torch.abs(pred - target)) / 1.0
    assert torch.allclose(loss, expected)


def test_jepa_reg_loss_matches_predictor_std_hinge():
    pred = torch.tensor(
        [
            [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        ]
    )
    target = pred.clone()

    loss = compute_ttt_aux_loss(
        pred,
        target,
        loss_type="jepa",
        loss_exp=1.0,
        reg_coeff=0.25,
        reg_eps=0.0001,
    )

    pred_loss = torch.mean(torch.abs(pred - target))
    pstd = torch.sqrt(pred.var(dim=1, unbiased=False) + 0.0001)
    reg_loss = torch.mean(F.relu(1.0 - pstd))
    assert torch.allclose(loss, pred_loss + 0.25 * reg_loss)


def test_cosine_aux_loss_remains_available_for_compatibility():
    pred = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    target = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])

    loss = compute_ttt_aux_loss(pred, target, loss_type="cosine")

    expected = -(F.normalize(pred, dim=-1) * F.normalize(target, dim=-1)).sum(dim=-1).mean()
    assert torch.allclose(loss, expected)


def test_jepa_loss_rejects_non_positive_loss_exp():
    pred = torch.zeros(1, 2, 3)
    target = torch.zeros(1, 2, 3)

    with pytest.raises(ValueError, match="loss_exp"):
        compute_ttt_aux_loss(pred, target, loss_type="jepa", loss_exp=0.0)
