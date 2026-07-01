# NLMS per-key 归一化 + 零训练 η 标定 + matched CPT 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 TTT 写规则改成 per-key 归一化的 NLMS 残差写(消除 cross-talk),在训练侧实现并经零训练 η 标定后做 matched CPT,验证能否把 RULER 0.285 拉回甚至超过关-TTT 上界 0.458。

**Architecture:** per-key NLMS `dw_c = η·r_c⊗k_c/(λ+‖k_c‖²)`,r_c=v_c−S·k_c。推理侧已有 NLMS 分支(修分母粒度即可);训练侧用 chunk 间 Python loop(16k=16 步)替代 cumsum,base/delta 路径分离复用 key_norm 双路结构。零新增可训练参数。

**Tech Stack:** PyTorch, einops/opt_einsum `contract`, HF Qwen3 自定义 modeling,pytest,`.venv/bin/python`。

**来源 spec:** `docs/superpowers/specs/2026-06-30-nlms-perkey-matched-cpt-design.md`

---

## File Structure

- `inference_model/hf_qwen3/modeling_qwen3.py` — 修 `_write_block` 的 NLMS 分母为 per-key(改动 1)
- `tests/test_ttt_nlms_write.py` — 更新 math 测试为 per-key 分母
- `hf_models/hf_qwen3/configuration_qwen3.py` — 训练侧 config 声明 `ttt_write_rule`/`ttt_nlms_lambda`(改动 2a)
- `hf_models/hf_qwen3/modeling_qwen3.py` — 训练侧 `Qwen3MLP.forward` 加 NLMS chunk-loop 分支(改动 2b)
- `tests/test_ttt_nlms_train.py` — 新建,训练侧 NLMS 正确性 + 训练/推理一致性
- `eval/diagnostics/ttt_signal_probe.py` — 加离线 NLMS output_delta 重算(阶段 A)

---

## Task 1: 推理侧 NLMS 分母改为 per-key

**Files:**
- Modify: `inference_model/hf_qwen3/modeling_qwen3.py:206-215`
- Test: `tests/test_ttt_nlms_write.py`

- [ ] **Step 1: 更新 math 测试为 per-key 分母**

替换 `tests/test_ttt_nlms_write.py` 中 `test_nlms_block_residual_matches_manual_math` 的期望计算块。找到这段:

```python
    S_delta0 = torch.zeros(mlp.down_proj.weight.shape)  # start: ΔW = 0
    R0 = V0 - torch.einsum("c h, d h -> c d", K0, S_delta0)  # [c, d]
    dW_expected = eta * torch.einsum("c d, c h -> d h", R0, K0) / (lam + (K0 * K0).sum())
    w_expected_after_first = mlp.down_proj.weight + dW_expected
```

替换为(per-key 分母:每个 token 除以自己的 λ+‖k‖²):

```python
    S_delta0 = torch.zeros(mlp.down_proj.weight.shape)  # start: ΔW = 0
    R0 = V0 - torch.einsum("c h, d h -> c d", K0, S_delta0)  # [c, d]
    denom_c = lam + (K0 * K0).sum(dim=-1, keepdim=True)      # [c, 1] per-key
    R0n = R0 / denom_c                                        # normalize each residual row
    dW_expected = eta * torch.einsum("c d, c h -> d h", R0n, K0)
    w_expected_after_first = mlp.down_proj.weight + dW_expected
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_ttt_nlms_write.py::test_nlms_block_residual_matches_manual_math -x -q`
Expected: FAIL（当前实现用整-chunk 标量分母,与 per-key 期望不符）

- [ ] **Step 3: 修改 `_write_block` 的 nlms 分支为 per-key**

`inference_model/hf_qwen3/modeling_qwen3.py:206-215`,把当前:

```python
    def _write_block(self, current_w, key, value, write_rule):
        if write_rule == "nlms":
            delta = current_w - self.down_proj.weight  # [d, h_dim]
            pred = contract("c h, d h -> c d", key, delta)  # key @ delta^T
            residual = value - pred  # [c, d]
            denom = self.ttt_nlms_lambda + (key * key).sum()
            dw = contract("c d, c h -> d h", residual, key) * (self.ttt_lr / denom)
        else:  # outer
            dw = contract("c h, c d -> d h", key, value) * self.ttt_lr
        return current_w + dw
```

替换为(分母 per-key,形状 [c,1],残差逐行归一化后再外积累加):

```python
    def _write_block(self, current_w, key, value, write_rule):
        if write_rule == "nlms":
            delta = current_w - self.down_proj.weight  # [d, h_dim]
            pred = contract("c h, d h -> c d", key, delta)  # key @ delta^T
            residual = value - pred  # [c, d]
            denom = self.ttt_nlms_lambda + (key * key).sum(dim=-1, keepdim=True)  # [c, 1] per-key
            residual = residual / denom
            dw = contract("c d, c h -> d h", residual, key) * self.ttt_lr
        else:  # outer
            dw = contract("c h, c d -> d h", key, value) * self.ttt_lr
        return current_w + dw
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_ttt_nlms_write.py -q`
Expected: PASS（5 个测试全过；outer 不变、η=0 退化、per-key math、subchunk 仍成立）

- [ ] **Step 5: 提交**

```bash
git add inference_model/hf_qwen3/modeling_qwen3.py tests/test_ttt_nlms_write.py
git commit -m "fix: NLMS inference write uses per-key normalization (lambda+||k||^2)"
```

---

## Task 2: 训练侧 config 声明 write_rule flag

**Files:**
- Modify: `hf_models/hf_qwen3/configuration_qwen3.py:186-192`（构造参数）和 `:261-263`（赋值）
- Test: `tests/test_ttt_nlms_train.py`（新建）

- [ ] **Step 1: 写失败测试（config 默认值与校验）**

新建 `tests/test_ttt_nlms_train.py`，写入:

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest tests/test_ttt_nlms_train.py::test_train_config_rejects_bad_write_rule -x -q`
Expected: FAIL（训练侧 config 尚未声明 `ttt_write_rule`，不会 raise）

- [ ] **Step 3: 在训练侧 config 加 flag**

`hf_models/hf_qwen3/configuration_qwen3.py`，在构造函数参数 `:191` 的 `ttt_target="hidden_states",` 之后插入:

```python
        ttt_write_rule="outer",
        ttt_nlms_lambda=1.0,
```

在赋值区 `:262` 的 `self.ttt_chunk = ttt_chunk` 之后插入:

```python
        self.ttt_write_rule = str(ttt_write_rule)
        if self.ttt_write_rule not in {"outer", "nlms"}:
            raise ValueError("ttt_write_rule must be one of {'outer', 'nlms'}")
        self.ttt_nlms_lambda = float(ttt_nlms_lambda)
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/python -m pytest tests/test_ttt_nlms_train.py -q`
Expected: PASS（2 个 config 测试通过）

- [ ] **Step 5: 提交**

```bash
git add hf_models/hf_qwen3/configuration_qwen3.py tests/test_ttt_nlms_train.py
git commit -m "feat: declare ttt_write_rule/ttt_nlms_lambda on training Qwen3Config"
```

---

## Task 3: 训练侧 MLP 读 write_rule flag

**Files:**
- Modify: `hf_models/hf_qwen3/modeling_qwen3.py:98`（`__init__` 读 flag）

- [ ] **Step 1: 写失败测试（MLP 实例暴露 write_rule 属性）**

在 `tests/test_ttt_nlms_train.py` 追加:

```python
def test_train_mlp_reads_write_rule():
    torch.manual_seed(0)
    mlp = Qwen3MLP(_cfg(ttt_write_rule="nlms", ttt_nlms_lambda=2.0), layer_idx=0)
    assert mlp.ttt_write_rule == "nlms"
    assert mlp.ttt_nlms_lambda == 2.0
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest tests/test_ttt_nlms_train.py::test_train_mlp_reads_write_rule -x -q`
Expected: FAIL（`AttributeError: ... has no attribute 'ttt_write_rule'`）

- [ ] **Step 3: 在训练侧 MLP `__init__` 读 flag**

`hf_models/hf_qwen3/modeling_qwen3.py:98`，把:

```python
            self.ttt_lr = getattr(config, "ttt_lr", 0.3)
```

改为:

```python
            self.ttt_lr = getattr(config, "ttt_lr", 0.3)
            self.ttt_write_rule = getattr(config, "ttt_write_rule", "outer")
            self.ttt_nlms_lambda = float(getattr(config, "ttt_nlms_lambda", 1.0))
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/python -m pytest tests/test_ttt_nlms_train.py -q`
Expected: PASS（3 个测试通过）

- [ ] **Step 5: 提交**

```bash
git add hf_models/hf_qwen3/modeling_qwen3.py tests/test_ttt_nlms_train.py
git commit -m "feat: training MLP reads ttt_write_rule/ttt_nlms_lambda"
```

---

## Task 4: 训练侧 NLMS chunk-loop forward 分支

**Files:**
- Modify: `hf_models/hf_qwen3/modeling_qwen3.py:231-245`（在 outer cumsum 分支前加 nlms 分支）
- Test: `tests/test_ttt_nlms_train.py`

**背景:** 训练侧 outer 在 `:231-245` 用 cumsum 并行(`d_down_proj_sum = d_down_proj.cumsum(dim=1)`)。NLMS 因残差依赖累积 S 不能 cumsum，用 chunk 间 Python loop。读出与 outer 一致地用 `sum_{j<t}`（chunk t 读其之前所有 chunk 的累积 ΔW），base 路径用原始 h。

- [ ] **Step 1: 写失败测试（训练侧 NLMS = 推理侧 NLMS，同输入数值一致）**

在 `tests/test_ttt_nlms_train.py` 追加:

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest tests/test_ttt_nlms_train.py::test_train_nlms_matches_inference_nlms -x -q`
Expected: FAIL（训练侧无 nlms 分支，走 outer cumsum，数值不等于推理 NLMS）

- [ ] **Step 3: 在训练侧 forward 加 nlms 分支**

`hf_models/hf_qwen3/modeling_qwen3.py`,定位 `:231` 的 `if not hasattr(self, "ttt_key_norm"):`。在它**之前**插入 nlms 分支(NLMS 优先于 outer/key_norm 路径):

```python
        if getattr(self, "ttt_write_rule", "outer") == "nlms":
            # Block residual write (per-key NLMS), chunk-serial loop.
            # base path uses original h (W0); delta path reads sum_{j<t} of NLMS updates.
            # delta_down_proj is kept for the monitor stats (same meaning as outer: per-chunk dW).
            W0 = self.down_proj.weight                       # [d, h_dim]
            S = torch.zeros(bs, W0.shape[0], W0.shape[1], device=h.device, dtype=torch.float32)  # ΔW per batch
            outs = []
            per_chunk_dw = []
            for i in range(chunk_num):
                Ki = h_padded[:, i].float()                  # [b, c, h_dim]
                Vi = prediction_states[:, i].float()         # [b, c, d]
                base_i = contract("d h, b c h -> b c d", W0.float(), Ki)
                delta_i = contract("b d h, b c h -> b c d", S, Ki)
                outs.append((base_i + delta_i).to(h.dtype))
                # per-key residual write
                pred_i = contract("b c h, b d h -> b c d", Ki, S)   # Ki @ S^T
                resid_i = Vi - pred_i                                # [b, c, d]
                denom_i = self.ttt_nlms_lambda + (Ki * Ki).sum(dim=-1, keepdim=True)  # [b, c, 1]
                resid_i = resid_i / denom_i
                dW_i = contract("b c d, b c h -> b d h", resid_i, Ki) * self.ttt_lr   # [b, d, h_dim]
                per_chunk_dw.append(dW_i)
                S = S + dW_i
            down_proj = torch.stack(outs, dim=1)             # [b, chunk_num, c, d]
            delta_down_proj = torch.stack(per_chunk_dw, dim=1).to(h.dtype)  # [b, chunk_num, d, h_dim]
            self._record_ttt_future_chunk_aux(prediction_states, target_padded, x.shape[1])
            self._record_ttt_monitor_stats(delta_down_proj, h_padded, down_proj)
            return rearrange(down_proj, "b t c d -> b (t c) d")[:, : x.shape[1], :]
```

注意:此分支在 `prediction_states` 与 `target_padded`/`h_padded` 计算之后(`:230` 之后),所以这些变量已就绪。`bs`/`chunk_num` 已在 `:221` 定义。

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/python -m pytest tests/test_ttt_nlms_train.py -q`
Expected: PASS（5 个测试：3 config/init + 2 NLMS forward）

- [ ] **Step 5: 运行训练侧回归(确认 outer/key_norm 路径未受影响)**

Run: `.venv/bin/python -m pytest tests/test_ttt_key_norm.py tests/test_qwen3_swa_ttt_aux.py -q`
Expected: PASS（既有训练侧测试不破）

- [ ] **Step 6: 提交**

```bash
git add hf_models/hf_qwen3/modeling_qwen3.py tests/test_ttt_nlms_train.py
git commit -m "feat: training-side per-key NLMS write rule (chunk-serial loop)"
```

---

## Task 5: 训练侧 outer 分支 bit 级不变 sanity

**Files:**
- Test: `tests/test_ttt_nlms_train.py`

- [ ] **Step 1: 写测试(write_rule=outer 与缺省 outer cumsum 完全一致)**

在 `tests/test_ttt_nlms_train.py` 追加:

```python
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
```

- [ ] **Step 2: 运行确认通过**

Run: `.venv/bin/python -m pytest tests/test_ttt_nlms_train.py::test_train_outer_flag_unchanged -q`
Expected: PASS（outer 分支走原 cumsum，未被 nlms 分支影响）

- [ ] **Step 3: 提交**

```bash
git add tests/test_ttt_nlms_train.py
git commit -m "test: training outer write rule is bit-identical to cumsum baseline"
```

---

## Task 6: 探针离线 NLMS output_delta 重算(阶段 A η 标定)

**Files:**
- Modify: `eval/diagnostics/ttt_signal_probe.py`（加 `nlms_output_delta` 函数 + report 字段 + CLI `--nlms-etas`）
- Test: `tests/test_capacity_law_probe.py`

- [ ] **Step 1: 写失败测试(离线 NLMS output_delta 函数)**

在 `tests/test_capacity_law_probe.py` 追加:

```python
from eval.diagnostics.ttt_signal_probe import nlms_output_delta_for_etas


def test_nlms_output_delta_eta_zero_is_zero():
    """η=0 时 NLMS 不写,output_delta 必须为 0(读出退化到 base)。"""
    import torch
    torch.manual_seed(0)
    # K,V per chunk: [chunk_num, c, dim]; W0: [d, h_dim]
    K = torch.randn(2, 4, 8)
    V = torch.randn(2, 4, 8)
    W0 = torch.randn(8, 8)
    out = nlms_output_delta_for_etas(K, V, W0, etas=[0.0, 1.0], lam=1.0)
    assert abs(out[0.0]) < 1e-6
    assert out[1.0] > 0.0   # η=1 有非零写入
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest tests/test_capacity_law_probe.py::test_nlms_output_delta_eta_zero_is_zero -x -q`
Expected: FAIL（`ImportError: cannot import name 'nlms_output_delta_for_etas'`）

- [ ] **Step 3: 实现离线重算函数**

在 `eval/diagnostics/ttt_signal_probe.py` 的 `probe_capacity_law` 函数(`:255`)**之前**插入:

```python
def nlms_output_delta_for_etas(K, V, W0, etas, lam=1.0):
    """Offline per-key NLMS readout: relative output_delta vs base, per eta.

    K: [chunk_num, c, h_dim] keys (= h) per chunk
    V: [chunk_num, c, d]     values (= ttt_proj(ttt_conv(t))) per chunk
    W0: [d, h_dim]           base down_proj weight
    Returns {eta: mean ||delta_out|| / ||out|| over chunks}.
    """
    import torch
    Kf = K.float(); Vf = V.float(); W0f = W0.float()
    chunk_num = Kf.shape[0]
    out = {}
    for eta in etas:
        S = torch.zeros_like(W0f)              # [d, h_dim]
        num = 0.0; den = 0.0
        for i in range(chunk_num):
            Ki = Kf[i]; Vi = Vf[i]             # [c, h_dim], [c, d]
            base_i = torch.einsum("d h, c h -> c d", W0f, Ki)
            delta_i = torch.einsum("c h, d h -> c d", Ki, S)
            out_i = base_i + delta_i
            num += float((out_i - base_i).norm())
            den += float(out_i.norm().clamp_min(1e-12))
            # per-key residual write
            pred_i = torch.einsum("c h, d h -> c d", Ki, S)
            resid_i = (Vi - pred_i) / (lam + (Ki * Ki).sum(dim=-1, keepdim=True))
            S = S + eta * torch.einsum("c d, c h -> d h", resid_i, Ki)
        out[eta] = num / max(den, 1e-12)
    return out
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/python -m pytest tests/test_capacity_law_probe.py -q`
Expected: PASS（既有 4 + 新 1 = 5 个通过）

- [ ] **Step 5: 提交**

```bash
git add eval/diagnostics/ttt_signal_probe.py tests/test_capacity_law_probe.py
git commit -m "feat: offline per-key NLMS output_delta sweep for eta calibration"
```

---

## Task 7: 阶段 A 运行 — 零训练 η 标定

**Files:** 无代码改动，运行 + 记录。

- [ ] **Step 1: 准备脚本读取捕获的 K/V 并 sweep η**

新建 `eval/diagnostics/run_nlms_eta_sweep.py`:

```python
"""Sweep eta for offline per-key NLMS using a captured RULER forward.
Reuses TTTKeyCapture to grab h(K) and recomputes V=ttt_proj(ttt_conv(t)).
"""
import argparse, json, sys
from pathlib import Path
import torch
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
import hf_models  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval.diagnostics.ttt_signal_probe import nlms_output_delta_for_etas, load_samples

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", required=True); ap.add_argument("--length", type=int, default=16384)
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--etas", type=float, nargs="+", default=[1, 3, 10, 30, 100])
    ap.add_argument("--out", default="results/nlms_eta_sweep.json")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    model = AutoModelForCausalLM.from_pretrained(a.ckpt, torch_dtype=torch.bfloat16).cuda().eval()
    inner = getattr(model, "model", model)
    # find TTT layers + their chunk
    ttt_mlps = [(l.self_attn.layer_idx if hasattr(l,'self_attn') else i, l.mlp)
                for i, l in enumerate(inner.layers) if hasattr(l.mlp, "ttt_conv")]
    samples = load_samples(a.data, a.num_samples)
    agg = {e: [] for e in a.etas}
    for text in samples:
        ids = tok(text, return_tensors="pt", truncation=True, max_length=a.length).input_ids.cuda()
        if ids.shape[1] < 2: continue
        with torch.no_grad():
            inner(input_ids=ids, use_cache=False)  # populate, but we recompute K/V directly below
            for li, mlp in ttt_mlps:
                # recompute h(K) from last layer input is non-trivial; instead use the
                # forward's own padded h/V via a capture — here we approximate by
                # re-running mlp internals is out of scope; we read monitor-free path.
                pass
    # NOTE: detailed K/V capture wired in Step 2.
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"etas": a.etas, "note": "see step2"}, indent=2))

if __name__ == "__main__":
    main()
```

实际 K/V 捕获用 Task 6 的 `nlms_output_delta_for_etas` + 一个 forward-pre-hook 抓 `h_padded`/`prediction_states`(与 `ttt_signal_probe.py` 的 `TTTKeyCapture` 同法)。**实现细节在执行时参照 `ttt_signal_probe.py:73-114` 的 hook 模式补全。**

- [ ] **Step 2: 运行 sweep**

Run:
```bash
CKPT=outputs/qwen3-1.7b-stage3-cpt-swa3-full1-strict-c1024-16k/checkpoints/global_step_8000/hf_ckpt
RULER=/zouxiangyu/codes/TTT/In-Place-TTT-v0/eval_scripts/ruler
OUT=eval/exp_analysis/ruler_results_nlms_exp
CUDA_VISIBLE_DEVICES=1 .venv/bin/python eval/diagnostics/run_nlms_eta_sweep.py \
  --ckpt "$CKPT" --tokenizer /zouxiangyu/models/Qwen/Qwen3-1.7B \
  --data "$OUT/probe_data_16k.jsonl" --length 16384 --num-samples 20 \
  --etas 1 3 10 30 100 --out "$OUT/nlms_eta_sweep.json"
```
Expected: 输出每个 η 的 output_delta 均值。

- [ ] **Step 3: 选 η**

读取 `$OUT/nlms_eta_sweep.json`，选 output_delta 落在 **[0.2, 0.4]** 的 η（1–2 个）。记录到 `refine-logs/EXPERIMENT_RESULTS_2026-06-29.md` 的「M3 准备」小节。

判据：若无任何 η 落区间 → 近似版可能不成立，停下与用户讨论是否上严格序贯/DeltaNet form。

- [ ] **Step 4: 提交**

```bash
git add eval/diagnostics/run_nlms_eta_sweep.py refine-logs/EXPERIMENT_RESULTS_2026-06-29.md
git commit -m "feat: stage-A NLMS eta calibration sweep + selected eta"
```

---

## Task 8: 阶段 B — matched CPT 三臂训练 + eval

**Files:**
- Create: `scripts/train/longsft/qwen3-1.7b/run_stage3_nlms_matched.sh`（薄 wrapper，沿用 stage3 config，仅切 write_rule/eta）

**前提:** Task 7 选定 η。三臂从同一 stage2 ckpt 起、冻 backbone、同 tokens/optimizer/步数。

- [ ] **Step 1: 写 wrapper 脚本**

新建 `scripts/train/longsft/qwen3-1.7b/run_stage3_nlms_matched.sh`:

```bash
#!/bin/bash
# Matched CPT arm: stage3 strict + selectable TTT write rule (outer/nlms/keynorm).
# Usage: WRITE_RULE=nlms TTT_LR=<eta> bash run_stage3_nlms_matched.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WRITE_RULE="${WRITE_RULE:-outer}"
TTT_LR_VALUE="${TTT_LR:-1.0}"
NLMS_LAMBDA="${NLMS_LAMBDA:-1.0}"
export EXP_NAME="${EXP_NAME:-qwen3-1.7b-stage3-matched-${WRITE_RULE}-lr${TTT_LR_VALUE}}"
export WANDB_NAME="${WANDB_NAME:-$EXP_NAME}"
export MASTER_PORT="${MASTER_PORT:-12375}"
export MODEL_FOUNDATION_JSON="{\"ttt_write_rule\": \"${WRITE_RULE}\", \"ttt_lr\": ${TTT_LR_VALUE}, \"ttt_nlms_lambda\": ${NLMS_LAMBDA}, \"ttt_train_only\": true}"
exec bash "$SCRIPT_DIR/run_stage3_cpt_swa3_full1_strict.sh" "$@"
```

- [ ] **Step 2: 启动三臂(从同一 stage2 ckpt，同预算)**

Run（η 用 Task 7 选定值，下例占位 `<ETA>`）:
```bash
cd /zouxiangyu/codes/Learning/In-Place-TTT
WRITE_RULE=outer  CUDA_VISIBLE_DEVICES=1,2,3,4 bash scripts/train/longsft/qwen3-1.7b/run_stage3_nlms_matched.sh &
WRITE_RULE=nlms TTT_LR=<ETA> CUDA_VISIBLE_DEVICES=5,6,7,0 EXP_NAME=qwen3-1.7b-stage3-matched-nlms MASTER_PORT=12376 bash scripts/train/longsft/qwen3-1.7b/run_stage3_nlms_matched.sh &
```
keynorm 臂可串行随后跑（GPU 复用）。**注意校验 stage3 config 的 ttt_layers/ttt_chunk 与 R001 ckpt 一致。**

- [ ] **Step 3: eval 三臂 RULER 16k 全任务 + 4k/8k 护栏**

每臂转 HF ckpt 后:
```bash
RULER=/zouxiangyu/codes/TTT/In-Place-TTT-v0/eval_scripts/ruler
OUT=eval/exp_analysis/ruler_results_nlms_exp
CUDA_VISIBLE_DEVICES=1 .venv/bin/python eval/eval_scripts/eval_ruler.py \
  --model_path <ARM_HF_CKPT> --abbr <ARM>-matched \
  --lengths 4096 8192 16384 --n_per_task 50 --max_seq 32768 \
  --ruler_root "$RULER" --out_root "$OUT" --max_new_tokens 1024
```

- [ ] **Step 4: 判定**

对比三臂 overall + 检索子任务:
- NLMS-CPT 相对 outer-CPT(0.285)是否显著提升？理想超过关-TTT 上界 0.458？
- keynorm-CPT 不应达到 NLMS 水平。
- 4k/8k 护栏不退化。
记录到 `refine-logs/EXPERIMENT_RESULTS_2026-06-29.md`。用 `/result-to-claim` 评判结论。

- [ ] **Step 5: 提交**

```bash
git add scripts/train/longsft/qwen3-1.7b/run_stage3_nlms_matched.sh refine-logs/EXPERIMENT_RESULTS_2026-06-29.md
git commit -m "feat: matched CPT three-arm (outer/nlms/keynorm) script + results"
```

---

## Self-Review

**Spec coverage:**
- 核心修正 per-key 归一化 → Task 1（推理）+ Task 4（训练）✓
- 改动1 推理分母 → Task 1 ✓
- 改动2 训练 NLMS 分支 + config flag → Task 2/3/4 ✓
- 改动3 白名单无需改 → 已在 spec 说明，无 task（正确）✓
- 阶段 A 零训练 η 标定 → Task 6（函数）+ Task 7（运行）✓
- 阶段 B matched CPT → Task 8 ✓
- 护栏(η=0 退化/outer 不变/训推一致/短上下文) → Task 1 Step4、Task 4 Step1、Task 5、Task 8 Step3 ✓

**Placeholder scan:** Task 7 Step1 脚本标注「实现细节执行时参照 hook 模式补全」——这是唯一一处非完整代码,因为 K/V 捕获需复用 `TTTKeyCapture` 的运行时 hook(已给出参照行号 `:73-114`),属可接受的执行时补全;其余步骤代码完整。

**Type consistency:** `ttt_write_rule`/`ttt_nlms_lambda` 在训练/推理 config 同名;`nlms_output_delta_for_etas(K,V,W0,etas,lam)` 签名在 Task 6 定义、Task 7 调用一致;per-key 分母 `λ+‖k_c‖²`(`sum(dim=-1,keepdim=True)`)在 Task 1/4/6 写法一致。

**风险提示:** Task 4 的训练侧 NLMS 用 chunk-loop + FP32 S,反向传播会展开 16 步计算图——需在 Task 8 启动前确认显存/速度可接受(可先单 step 试跑)。
