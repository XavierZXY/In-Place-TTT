"""Position-bucketed per-token LM loss metrics for SWA-conversion evals.

Average CE over a long sequence is dominated by locally-predictable tokens and
dilutes the long-range damage caused by replacing full attention with SWA.
Bucketing the student-teacher loss gap by absolute token position exposes it:
positions below the SWA window form a natural control group (SWA equals full
attention there), while the far buckets isolate the capability being repaired.
"""

from typing import Sequence

import torch
import torch.nn.functional as F


def per_token_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the shifted next-token CE per position.

    Returns ``(losses, valid)`` of shape ``[batch, seq_len - 1]``; column ``j``
    is the loss of predicting the token at absolute position ``j + 1``. Logits
    are upcast to float32 chunk-by-chunk to bound peak memory.
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    bsz, seq_len, vocab = shift_logits.shape
    flat_logits = shift_logits.reshape(-1, vocab)
    flat_labels = shift_labels.reshape(-1)

    losses = torch.empty(flat_labels.shape, dtype=torch.float32, device=flat_labels.device)
    for start in range(0, flat_logits.shape[0], chunk_size):
        end = min(start + chunk_size, flat_logits.shape[0])
        losses[start:end] = F.cross_entropy(
            flat_logits[start:end].float(),
            flat_labels[start:end],
            ignore_index=ignore_index,
            reduction="none",
        )

    valid = flat_labels.ne(ignore_index)
    return losses.view(bsz, seq_len), valid.view(bsz, seq_len)


class PositionBucketMeter:
    """Accumulates per-token loss sums bucketed by absolute token position."""

    def __init__(self, boundaries: Sequence[int]):
        self.boundaries = sorted(int(b) for b in boundaries)
        self._lows = [0] + self.boundaries
        self._highs = self.boundaries + [None]
        self.loss_sums = [0.0] * len(self._lows)
        self.token_counts = [0.0] * len(self._lows)

    @property
    def labels(self) -> list[str]:
        return [f"{lo}-{hi}" if hi is not None else f"{lo}+" for lo, hi in zip(self._lows, self._highs)]

    def add(self, per_token_loss: torch.Tensor, valid_mask: torch.Tensor) -> None:
        seq_len = per_token_loss.shape[1]
        # Column j predicts the token at absolute position j + 1.
        positions = torch.arange(1, seq_len + 1, device=per_token_loss.device).unsqueeze(0)
        for idx, (lo, hi) in enumerate(zip(self._lows, self._highs)):
            mask = valid_mask & (positions >= lo)
            if hi is not None:
                mask = mask & (positions < hi)
            self.loss_sums[idx] += float((per_token_loss * mask).sum().item())
            self.token_counts[idx] += float(mask.sum().item())

    def reduced_means(self, all_reduce_fn, group) -> dict[str, float]:
        """All-reduce sums/counts across ranks and return per-bucket means.

        Collective call — must be invoked on every rank. Buckets with zero
        tokens globally are omitted.
        """
        sums = all_reduce_fn(self.loss_sums + self.token_counts, op="sum", group=group)
        n = len(self.loss_sums)
        loss_sums, token_counts = sums[:n], sums[n:]
        return {
            label: loss_sum / count
            for label, loss_sum, count in zip(self.labels, loss_sums, token_counts)
            if count > 0
        }
