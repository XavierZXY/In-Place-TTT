import torch
import torch.nn.functional as F

from tasks.halo_kd_distillation import _chunked_ce_kl_loss


def _make_inputs(seed=0, batch=2, seq=10, hidden=8, vocab=13):
    torch.manual_seed(seed)
    student_hidden = torch.randn(batch, seq, hidden, requires_grad=True)
    student_weight = torch.randn(vocab, hidden, requires_grad=True)
    teacher_hidden = torch.randn(batch, seq, hidden)
    teacher_weight = torch.randn(vocab, hidden)
    labels = torch.randint(0, vocab, (batch, seq))
    labels[0, :3] = -100
    attention_mask = torch.ones(batch, seq, dtype=torch.long)
    attention_mask[1, -2:] = 0
    return student_hidden, student_weight, teacher_hidden, teacher_weight, labels, attention_mask


def _reference_kl(student_hidden, student_weight, teacher_hidden, teacher_weight, attention_mask, temp):
    student_logits = F.linear(student_hidden, student_weight).float()
    teacher_logits = F.linear(teacher_hidden, teacher_weight).float()
    valid = attention_mask.reshape(-1).bool()
    student_log_prob = F.log_softmax(student_logits.reshape(-1, student_logits.shape[-1])[valid] / temp, dim=-1)
    teacher_log_prob = F.log_softmax(teacher_logits.reshape(-1, teacher_logits.shape[-1])[valid] / temp, dim=-1)
    kl = F.kl_div(student_log_prob, teacher_log_prob, log_target=True, reduction="sum") * (temp * temp)
    return kl / valid.sum()


def test_chunked_kl_matches_full_reference_with_grad():
    student_hidden, student_weight, teacher_hidden, teacher_weight, labels, attention_mask = _make_inputs()

    loss, _, loss_kl = _chunked_ce_kl_loss(
        student_hidden=student_hidden,
        student_weight=student_weight,
        teacher_hidden=teacher_hidden,
        teacher_weight=teacher_weight,
        labels=labels,
        attention_mask=attention_mask,
        alpha_ce=0.0,
        alpha_kl=1.0,
        temperature=2.0,
        chunk_size=3,
    )
    loss.backward()
    chunked_grad = student_hidden.grad.clone()

    student_hidden.grad = None
    ref_loss = _reference_kl(student_hidden, student_weight, teacher_hidden, teacher_weight, attention_mask, 2.0)
    ref_loss.backward()

    assert torch.allclose(loss, ref_loss, atol=1e-5)
    assert torch.allclose(loss_kl, ref_loss, atol=1e-5)
    assert torch.allclose(chunked_grad, student_hidden.grad, atol=1e-5)


def test_chunked_ce_matches_full_reference_with_grad():
    student_hidden, student_weight, _, _, labels, attention_mask = _make_inputs(seed=1)

    loss, loss_ce, _ = _chunked_ce_kl_loss(
        student_hidden=student_hidden,
        student_weight=student_weight,
        teacher_hidden=None,
        teacher_weight=None,
        labels=labels,
        attention_mask=attention_mask,
        alpha_ce=1.0,
        alpha_kl=0.0,
        temperature=1.0,
        chunk_size=3,
    )
    loss.backward()
    chunked_grad = student_hidden.grad.clone()

    student_hidden.grad = None
    logits = F.linear(student_hidden[:, :-1, :], student_weight).float()
    ref_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="sum",
    ) / labels[:, 1:].ne(-100).sum()
    ref_loss.backward()

    assert torch.allclose(loss, ref_loss, atol=1e-5)
    assert torch.allclose(loss_ce, ref_loss, atol=1e-5)
    assert torch.allclose(chunked_grad, student_hidden.grad, atol=1e-5)


def test_chunked_loss_works_under_no_grad():
    student_hidden, student_weight, teacher_hidden, teacher_weight, labels, attention_mask = _make_inputs(seed=2)

    with torch.no_grad():
        loss, _, _ = _chunked_ce_kl_loss(
            student_hidden=student_hidden,
            student_weight=student_weight,
            teacher_hidden=teacher_hidden,
            teacher_weight=teacher_weight,
            labels=labels,
            attention_mask=attention_mask,
            alpha_ce=0.0,
            alpha_kl=1.0,
            temperature=1.0,
            chunk_size=4,
        )
    assert torch.isfinite(loss)
    assert not loss.requires_grad
