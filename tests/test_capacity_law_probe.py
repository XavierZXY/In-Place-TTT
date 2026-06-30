"""Correctness tests for the capacity-law probe metrics (R006).

These are pure functions over a captured key matrix h ∈ [seq, d]:
- effective_rank: spectral entropy of the key Gram; full for orthonormal keys,
  ~1 for rank-1 (all keys collinear).
- write_collision: mean fraction of a new key's energy already spanned by
  earlier keys; ~0 for orthogonal keys, ~1 for collinear keys.
"""

import math
import torch

from eval.diagnostics.ttt_signal_probe import effective_rank, write_collision


def test_effective_rank_orthonormal_is_full():
    h = torch.eye(8)  # 8 orthonormal keys in 8-dim
    er = effective_rank(h)
    assert er > 7.5, f"orthonormal keys should give near-full effective rank, got {er}"


def test_effective_rank_collinear_is_one():
    base = torch.randn(8)
    h = torch.stack([base * (i + 1.0) for i in range(8)])  # all collinear
    er = effective_rank(h)
    assert er < 1.2, f"collinear keys should give effective rank ~1, got {er}"


def test_write_collision_orthogonal_is_low():
    h = torch.eye(8)
    wc = write_collision(h)
    assert wc < 0.1, f"orthogonal keys should have near-zero collision, got {wc}"


def test_write_collision_collinear_is_high():
    base = torch.randn(8)
    h = torch.stack([base * (i + 1.0) for i in range(8)])
    wc = write_collision(h)
    assert wc > 0.9, f"collinear keys should have near-total collision, got {wc}"


def test_nlms_output_delta_eta_zero_is_zero():
    """η=0 时 NLMS 不写,output_delta 必须为 0(读出退化到 base)。"""
    import torch
    from eval.diagnostics.ttt_signal_probe import nlms_output_delta_for_etas
    torch.manual_seed(0)
    K = torch.randn(2, 4, 8)
    V = torch.randn(2, 4, 8)
    W0 = torch.randn(8, 8)
    out = nlms_output_delta_for_etas(K, V, W0, etas=[0.0, 1.0], lam=1.0)
    assert abs(out[0.0]) < 1e-6
    assert out[1.0] > 0.0   # η=1 有非零写入
