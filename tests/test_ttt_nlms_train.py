"""Training-side NLMS write rule: config flags + forward correctness + train/infer parity."""
import pytest
import torch

from hf_models.hf_qwen3.configuration_qwen3 import Qwen3Config
from hf_models.hf_qwen3.modeling_qwen3 import Qwen3MLP
from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config as InfConfig
from inference_model.hf_qwen3.modeling_qwen3 import Qwen3MLP as InfMLP


def _cfg(**ov):
    kw = dict(
        vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4, max_position_embeddings=64,
        ttt_layers=[0], ttt_mode=True, ttt_proj=True, ttt_lr=0.5, ttt_chunk=2,
        ttt_target="input_embed",
    )
    kw.update(ov)
    return Qwen3Config(**kw)


def test_train_config_default_write_rule_outer():
    assert getattr(_cfg(), "ttt_write_rule", "outer") == "outer"


def test_train_config_rejects_bad_write_rule():
    with pytest.raises(ValueError):
        _cfg(ttt_write_rule="bogus")


def test_train_mlp_reads_write_rule():
    torch.manual_seed(0)
    mlp = Qwen3MLP(_cfg(ttt_write_rule="nlms", ttt_nlms_lambda=2.0), layer_idx=0)
    assert mlp.ttt_write_rule == "nlms"
    assert mlp.ttt_nlms_lambda == 2.0


def _randomize(mlp):
    with torch.no_grad():
        for p in mlp.parameters():
            p.normal_(0.0, 0.1)
    return mlp.eval()


def test_train_nlms_matches_inference_nlms():
    """训练侧 chunk-loop NLMS 必须与推理侧逐-chunk NLMS 在同权重同输入下数值一致。"""
    torch.manual_seed(0)
    train_cfg = _cfg(ttt_write_rule="nlms", ttt_lr=0.5, ttt_nlms_lambda=1.0, ttt_chunk=2)
    inf_cfg = InfConfig(
        vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4, max_position_embeddings=64,
        ttt_layers=[0], ttt_mode=True, ttt_proj=True, ttt_lr=0.5, ttt_chunk=2,
        ttt_target="input_embed", ttt_write_rule="nlms", ttt_nlms_lambda=1.0,
    )
    torch.manual_seed(1)
    train_mlp = _randomize(Qwen3MLP(train_cfg, layer_idx=0))
    torch.manual_seed(1)
    inf_mlp = _randomize(InfMLP(inf_cfg, layer_idx=0))

    x = torch.randn(1, 4, 8)   # seq=4 = 2 chunks of size 2
    t = torch.randn(1, 4, 8)
    with torch.no_grad():
        train_out = train_mlp(x, t=t)              # [1,4,8]
        inf_out, _ = inf_mlp(x, t=t)               # [1,4,8]
    torch.testing.assert_close(train_out, inf_out, rtol=1e-4, atol=1e-5)


def test_train_nlms_eta_zero_is_base():
    """训练侧 NLMS η=0 必须等于 base MLP(无 fast-weight delta)。"""
    torch.manual_seed(2)
    mlp = _randomize(Qwen3MLP(_cfg(ttt_write_rule="nlms", ttt_lr=0.0, ttt_chunk=2), layer_idx=0))
    x = torch.randn(1, 4, 8)
    t = torch.randn(1, 4, 8)
    with torch.no_grad():
        out = mlp(x, t=t)
        h = mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)
        base = torch.nn.functional.linear(h, mlp.down_proj.weight, mlp.down_proj.bias)
    torch.testing.assert_close(out, base, rtol=1e-4, atol=1e-5)


def test_train_outer_flag_unchanged():
    """显式 ttt_write_rule=outer 必须与默认(无 nlms)的 cumsum 路径 bit 级一致。"""
    torch.manual_seed(3)
    x = torch.randn(1, 4, 8)
    t = torch.randn(1, 4, 8)
    torch.manual_seed(5)
    mlp_default = _randomize(Qwen3MLP(_cfg(ttt_chunk=2), layer_idx=0))           # no flag → outer
    torch.manual_seed(5)
    mlp_outer = _randomize(Qwen3MLP(_cfg(ttt_write_rule="outer", ttt_chunk=2), layer_idx=0))
    with torch.no_grad():
        torch.testing.assert_close(mlp_default(x, t=t), mlp_outer(x, t=t))

def test_nlms_detach_state_flag_declared():
    """ttt_nlms_detach_state 必须在 config 声明(否则被 HF 静默丢弃)。"""
    assert _cfg(ttt_nlms_detach_state=True).ttt_nlms_detach_state is True
    assert _cfg().ttt_nlms_detach_state is False  # default


def test_nlms_detach_state_forward_identical():
    """detach 只改 backward,forward 数值必须与非 detach 一致。"""
    torch.manual_seed(9)
    x = torch.randn(1, 8, 8); t = torch.randn(1, 8, 8)
    torch.manual_seed(9)
    mlp_full = _randomize(Qwen3MLP(_cfg(ttt_write_rule="nlms", ttt_lr=0.5, ttt_chunk=2), layer_idx=0))
    torch.manual_seed(9)
    mlp_det = _randomize(Qwen3MLP(_cfg(ttt_write_rule="nlms", ttt_lr=0.5, ttt_chunk=2,
                                       ttt_nlms_detach_state=True), layer_idx=0))
    with torch.no_grad():
        torch.testing.assert_close(mlp_full(x, t=t), mlp_det(x, t=t))


def test_nlms_detach_state_changes_gradient():
    """detach 改变梯度图:full BPTT 与 1-step truncated BPTT 的 ttt_proj 梯度均非零,
    但数值不同(detach 截断了更早历史的梯度贡献)。"""
    torch.manual_seed(9)
    x = torch.randn(1, 8, 8); t = torch.randn(1, 8, 8)
    torch.manual_seed(9)
    m_full = _randomize(Qwen3MLP(_cfg(ttt_write_rule="nlms", ttt_lr=0.5, ttt_chunk=2), layer_idx=0))
    m_full(x, t=t).float().pow(2).mean().backward()
    g_full = m_full.ttt_proj.weight.grad
    assert g_full is not None and g_full.norm() > 0, "full BPTT must train ttt_proj"
    torch.manual_seed(9)
    m_det = _randomize(Qwen3MLP(_cfg(ttt_write_rule="nlms", ttt_lr=0.5, ttt_chunk=2,
                                     ttt_nlms_detach_state=True), layer_idx=0))
    m_det(x, t=t).float().pow(2).mean().backward()
    g_det = m_det.ttt_proj.weight.grad
    assert g_det is not None and g_det.norm() > 0, "1-step truncated BPTT must still train ttt_proj"
    for p_ in m_det.parameters():
        if p_.grad is not None:
            assert torch.isfinite(p_.grad).all()
    # 截断后梯度应与完整 BPTT 不同(否则 detach 无效)
    assert not torch.allclose(g_full, g_det), "detach must change the gradient"


def _freeze_backbone(mlp):
    """模拟 ttt_train_only:仅 ttt_conv/ttt_proj 可训练,其余冻结。"""
    for n, p in mlp.named_parameters():
        p.requires_grad = ("ttt_conv" in n) or ("ttt_proj" in n)
    return mlp


def test_nlms_detach_state_trains_under_frozen_backbone():
    """回归:matched-CPT 真实条件(冻结 backbone + detach)下,loss 必须可微且
    ttt_proj/ttt_conv 梯度非零。否则 loss.backward() 崩 'does not require grad'。"""
    torch.manual_seed(9)
    x = torch.randn(1, 8, 8); t = torch.randn(1, 8, 8)
    torch.manual_seed(9)
    mlp = _randomize(Qwen3MLP(_cfg(ttt_write_rule="nlms", ttt_lr=0.03, ttt_chunk=2,
                                   ttt_nlms_detach_state=True), layer_idx=0))
    _freeze_backbone(mlp)
    loss = mlp(x, t=t).float().pow(2).mean()
    assert loss.requires_grad, "frozen-backbone NLMS+detach loss must require grad"
    loss.backward()
    for name in ("ttt_proj", "ttt_conv"):
        g = getattr(mlp, name).weight.grad
        assert g is not None and g.norm() > 0, f"{name} must receive nonzero grad under detach"
        assert torch.isfinite(g).all()


def test_nlms_decay_flag_declared_and_validated():
    """ttt_nlms_decay 必须两侧 config 同名声明(防 HF 静默丢弃),默认 0.0,范围 [0,1)。"""
    from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config as InfConfig
    # 默认 0.0
    assert _cfg().ttt_nlms_decay == 0.0
    assert InfConfig(vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                     num_attention_heads=2, num_key_value_heads=1, head_dim=4,
                     max_position_embeddings=64).ttt_nlms_decay == 0.0
    # 合法值
    assert _cfg(ttt_nlms_decay=0.1).ttt_nlms_decay == 0.1
    # 非法值抛错
    with pytest.raises(ValueError):
        _cfg(ttt_nlms_decay=1.0)
    with pytest.raises(ValueError):
        _cfg(ttt_nlms_decay=-0.1)


def test_train_mlp_reads_decay():
    mlp = Qwen3MLP(_cfg(ttt_write_rule="nlms", ttt_nlms_decay=0.1), layer_idx=0)
    assert mlp.ttt_nlms_decay == 0.1
    # 默认 0.0
    assert Qwen3MLP(_cfg(ttt_write_rule="nlms"), layer_idx=0).ttt_nlms_decay == 0.0
