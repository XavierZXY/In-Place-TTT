import torch
import torch.nn.functional as F

from tasks.position_lm_metrics import PositionBucketMeter, per_token_lm_loss


def test_per_token_lm_loss_matches_unchunked_reference():
    torch.manual_seed(0)
    batch, seq, vocab = 2, 9, 11
    logits = torch.randn(batch, seq, vocab)
    labels = torch.randint(0, vocab, (batch, seq))
    labels[0, :4] = -100

    losses, valid = per_token_lm_loss(logits, labels, chunk_size=3)

    ref = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, vocab).float(),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(batch, seq - 1)
    assert torch.allclose(losses, ref, atol=1e-6)
    assert torch.equal(valid, labels[:, 1:].ne(-100))


def test_position_bucket_meter_buckets_by_predicted_position():
    meter = PositionBucketMeter([2, 4])
    assert meter.labels == ["0-2", "2-4", "4+"]

    # 1 sample, 5 predictions covering positions 1..5; losses equal position value.
    per_token_loss = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    valid = torch.tensor([[True, True, True, True, False]])
    meter.add(per_token_loss, valid)

    # bucket 0-2: position 1 -> loss 1; bucket 2-4: positions 2,3 -> 2,3; 4+: position 4 -> 4.
    identity_reduce = lambda data, op=None, group=None: data
    means = meter.reduced_means(identity_reduce, group=None)
    assert means["0-2"] == 1.0
    assert means["2-4"] == 2.5
    assert means["4+"] == 4.0


def test_position_bucket_meter_omits_empty_buckets():
    meter = PositionBucketMeter([100])
    per_token_loss = torch.tensor([[1.0, 2.0]])
    valid = torch.tensor([[True, True]])
    meter.add(per_token_loss, valid)

    identity_reduce = lambda data, op=None, group=None: data
    means = meter.reduced_means(identity_reduce, group=None)
    assert "0-100" in means
    assert "100+" not in means
