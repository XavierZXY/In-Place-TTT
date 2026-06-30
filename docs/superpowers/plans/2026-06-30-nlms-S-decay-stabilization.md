# NLMS 累积 S 衰减门控稳定化(方案 C)实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给训练侧和推理侧的 per-key NLMS 累积 S 加全局常量衰减门控 `S ← (1−α)·S + dW`,使 ‖S‖ 有界、打断 runaway 正反馈,让 matched CPT 能稳定训练。

**Architecture:** 单一标量超参 `ttt_nlms_decay`,两侧对称改动:config 声明 flag + MLP 读 flag + forward/`_write_block` 的 S 累加点改为衰减累加。`α=0` 逐位退化为现状(向后兼容)。零新增可训练参数。

**Tech Stack:** PyTorch, einops/opt_einsum `contract`, HF Qwen3 自定义 modeling, pytest, `.venv/bin/python`。

**来源 spec:** `docs/superpowers/specs/2026-06-30-nlms-S-decay-stabilization-design.md`

---

## File Structure

- `inference_model/hf_qwen3/configuration_qwen3.py` — 推理侧 config 声明 `ttt_nlms_decay`(Task 1)
- `hf_models/hf_qwen3/configuration_qwen3.py` — 训练侧 config 声明 `ttt_nlms_decay`(Task 1)
- `inference_model/hf_qwen3/modeling_qwen3.py` — 推理侧 MLP 读 flag(Task 2)+ `_write_block` 衰减累加(Task 4)
- `hf_models/hf_qwen3/modeling_qwen3.py` — 训练侧 MLP 读 flag(Task 2)+ forward 衰减累加(Task 3)
- `tests/test_ttt_nlms_train.py` — config 校验、有界性、训推一致、梯度非消失(Task 1/3/5)
- `tests/test_ttt_nlms_write.py` — 推理侧 α=0 退化(Task 4)
- `scripts/train/longsft/qwen3-1.7b/run_stage3_nlms_matched.sh` — NLMS_DECAY env 透传(Task 6)

---

## Task 1: 两侧 config 声明 ttt_nlms_decay flag

**Files:**
- Modify: `inference_model/hf_qwen3/configuration_qwen3.py:195`(构造参数)和 `:267`(赋值校验后)
- Modify: `hf_models/hf_qwen3/configuration_qwen3.py:194`(构造参数)和 `:270`(赋值后)
- Test: `tests/test_ttt_nlms_train.py`

- [ ] **Step 1: 写失败测试(config 默认值 + 校验)**

在 `tests/test_ttt_nlms_train.py` 末尾追加:

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest tests/test_ttt_nlms_train.py::test_nlms_decay_flag_declared_and_validated -x -q`
Expected: FAIL（`AttributeError: ... has no attribute 'ttt_nlms_decay'`，或默认值不存在）

- [ ] **Step 3: 训练侧 config 加 flag**

`hf_models/hf_qwen3/configuration_qwen3.py:194`，在 `ttt_nlms_detach_state=False,` 之后插入:

```python
        ttt_nlms_decay=0.0,
```

在 `:270` 的 `self.ttt_nlms_detach_state = bool(ttt_nlms_detach_state)` 之后插入:

```python
        # Decay gate for the NLMS fast-weight state: S <- (1 - decay) * S + dW.
        # Bounds ||S|| (~||dW||/decay) to break the runaway readout-residual
        # feedback. decay=0.0 is bit-identical to pure NLMS (unbounded accumulation).
        self.ttt_nlms_decay = float(ttt_nlms_decay)
        if not (0.0 <= self.ttt_nlms_decay < 1.0):
            raise ValueError("ttt_nlms_decay must be in [0.0, 1.0)")
```

- [ ] **Step 4: 推理侧 config 加 flag**

`inference_model/hf_qwen3/configuration_qwen3.py:195`，在 `ttt_write_subchunk=0,` 之后插入:

```python
        ttt_nlms_decay=0.0,
```

在 `:267` 的 `raise ValueError("ttt_write_subchunk must be >= 0")` 之后插入:

```python
        # Decay gate for the NLMS fast-weight state: ΔW <- (1 - decay) * ΔW + dW.
        # Bounds the accumulated delta to break runaway feedback; 0.0 = pure NLMS.
        self.ttt_nlms_decay = float(ttt_nlms_decay)
        if not (0.0 <= self.ttt_nlms_decay < 1.0):
            raise ValueError("ttt_nlms_decay must be in [0.0, 1.0)")
```

- [ ] **Step 5: 运行确认通过**

Run: `.venv/bin/python -m pytest tests/test_ttt_nlms_train.py::test_nlms_decay_flag_declared_and_validated -q`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add inference_model/hf_qwen3/configuration_qwen3.py hf_models/hf_qwen3/configuration_qwen3.py tests/test_ttt_nlms_train.py
git commit -m "feat: declare ttt_nlms_decay flag on both Qwen3Config (default 0.0)"
```

---

## Task 2: 两侧 MLP 读 ttt_nlms_decay flag

**Files:**
- Modify: `hf_models/hf_qwen3/modeling_qwen3.py:101`（训练侧 `__init__`）
- Modify: `inference_model/hf_qwen3/modeling_qwen3.py:110`（推理侧 `__init__`）
- Test: `tests/test_ttt_nlms_train.py`

- [ ] **Step 1: 写失败测试(MLP 实例暴露 decay 属性)**

在 `tests/test_ttt_nlms_train.py` 末尾追加:

```python
def test_train_mlp_reads_decay():
    mlp = Qwen3MLP(_cfg(ttt_write_rule="nlms", ttt_nlms_decay=0.1), layer_idx=0)
    assert mlp.ttt_nlms_decay == 0.1
    # 默认 0.0
    assert Qwen3MLP(_cfg(ttt_write_rule="nlms"), layer_idx=0).ttt_nlms_decay == 0.0
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest tests/test_ttt_nlms_train.py::test_train_mlp_reads_decay -x -q`
Expected: FAIL（`AttributeError: ... has no attribute 'ttt_nlms_decay'`）

- [ ] **Step 3: 训练侧 MLP 读 flag**

`hf_models/hf_qwen3/modeling_qwen3.py:101`，在 `self.ttt_nlms_detach_state = bool(getattr(config, "ttt_nlms_detach_state", False))` 之后插入:

```python
            self.ttt_nlms_decay = float(getattr(config, "ttt_nlms_decay", 0.0))
```

- [ ] **Step 4: 推理侧 MLP 读 flag**

`inference_model/hf_qwen3/modeling_qwen3.py:110`，在 `self.ttt_nlms_lambda = float(getattr(config, "ttt_nlms_lambda", 1.0))` 之后插入:

```python
            self.ttt_nlms_decay = float(getattr(config, "ttt_nlms_decay", 0.0))
```

- [ ] **Step 5: 运行确认通过**

Run: `.venv/bin/python -m pytest tests/test_ttt_nlms_train.py::test_train_mlp_reads_decay -q`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add hf_models/hf_qwen3/modeling_qwen3.py inference_model/hf_qwen3/modeling_qwen3.py tests/test_ttt_nlms_train.py
git commit -m "feat: both MLPs read ttt_nlms_decay (default 0.0)"
```

---

## Task 3: 训练侧 forward 衰减累加 + 有界性测试

**Files:**
- Modify: `hf_models/hf_qwen3/modeling_qwen3.py:274`（nlms 分支 chunk-loop 末的 S 累加）
- Test: `tests/test_ttt_nlms_train.py`

**背景:** 当前 `:274` 是 `S = S_hist + dW_i`(纯累加)。改为 `S = (1 - decay) * S_hist + dW_i`。`decay=0` 时逐位不变。

- [ ] **Step 1: 写失败测试(有界性 + α=0 退化)**

在 `tests/test_ttt_nlms_train.py` 末尾追加:

```python
def _s_norms_over_chunks(decay, eta=0.5, n_chunks=16, scale=5.0):
    """跑一个放大尺度的多 chunk forward，返回每 chunk 后 ‖S‖（用 monitor 的 cumsum delta 近似）。
    直接构造 MLP 并 hook 累积 delta_down_proj 的 cumsum norm。"""
    torch.manual_seed(7)
    cfg = _cfg(ttt_write_rule="nlms", ttt_lr=eta, ttt_chunk=2, ttt_nlms_decay=decay)
    mlp = _randomize(Qwen3MLP(cfg, layer_idx=0))
    x = torch.randn(1, n_chunks * 2, 8) * scale
    t = torch.randn(1, n_chunks * 2, 8) * scale
    with torch.no_grad():
        mlp(x, t=t)
    # delta_down_proj: [b, chunk_num, d, h_dim]; ‖S after chunk i‖ = ‖sum_{j<=i} (1-α)^(i-j) dW_j‖
    # 用 monitor 暴露的 per-chunk dW 重建 S 序列
    stats = mlp._last_ttt_monitor_stats
    return stats  # 仅用于确认 forward 跑通；‖S‖ 序列在 Step 3 用直接重算验证


def test_train_nlms_decay_bounds_S():
    """decay>0 时 ‖S‖ 收敛（末 4 chunk 相邻增幅 < 5%）；decay=0 时持续增长（末 > 5× 首）。
    直接复现 forward 的 S 递推以观测 ‖S‖ 序列。"""
    import torch
    from opt_einsum import contract

    def s_norm_seq(decay, eta=0.5, n_chunks=16, csize=8, dim=8, lam=1.0, scale=5.0):
        torch.manual_seed(7)
        W0 = torch.randn(dim, dim) * 0.05
        S = torch.zeros(dim, dim)
        norms = []
        for i in range(n_chunks):
            K = torch.randn(csize, dim) * scale
            V = torch.randn(csize, dim) * scale
            pred = K @ S.T
            resid = (V - pred) / (lam + (K * K).sum(-1, keepdim=True))
            dW = eta * (resid.T @ K) / csize
            S = (1 - decay) * S + dW
            norms.append(float(S.norm()))
        return norms

    n_decay = s_norm_seq(decay=0.1)
    n_pure = s_norm_seq(decay=0.0)
    # decay>0: 收敛（末 4 chunk 相邻增幅 < 5%）
    last4 = n_decay[-4:]
    for a, b in zip(last4, last4[1:]):
        assert (b - a) / max(a, 1e-9) < 0.05, f"decay should plateau, got {last4}"
    # decay=0: 持续增长
    assert n_pure[-1] > 5 * n_pure[0], f"pure NLMS should grow unbounded, got {n_pure[0]} -> {n_pure[-1]}"


def test_train_nlms_decay_zero_is_pure_nlms():
    """ttt_nlms_decay=0.0 的 forward 必须与无 decay flag 的纯 NLMS 逐位相同。"""
    torch.manual_seed(11)
    x = torch.randn(1, 8, 8); t = torch.randn(1, 8, 8)
    torch.manual_seed(13)
    mlp_pure = _randomize(Qwen3MLP(_cfg(ttt_write_rule="nlms", ttt_lr=0.5, ttt_chunk=2), layer_idx=0))
    torch.manual_seed(13)
    mlp_d0 = _randomize(Qwen3MLP(_cfg(ttt_write_rule="nlms", ttt_lr=0.5, ttt_chunk=2,
                                      ttt_nlms_decay=0.0), layer_idx=0))
    with torch.no_grad():
        torch.testing.assert_close(mlp_pure(x, t=t), mlp_d0(x, t=t))
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest tests/test_ttt_nlms_train.py::test_train_nlms_decay_bounds_S tests/test_ttt_nlms_train.py::test_train_nlms_decay_zero_is_pure_nlms -x -q`
Expected: `test_train_nlms_decay_zero_is_pure_nlms` PASS（decay=0 默认已等价，但 mlp 尚未应用 decay 所以仍 pass）；`test_train_nlms_decay_bounds_S` 是纯数值复现测试，独立于 mlp 实现，应直接 PASS。

注：本任务的核心测试 `test_train_nlms_decay_bounds_S` 验证的是 decay 递推数学，不依赖 mlp 改动；真正守护 mlp 实现的是下一步的 forward 改动后 `test_train_nlms_decay_zero_is_pure_nlms` 仍需 PASS + Task 5 的训推一致测试。先确认这两个测试在改动前的状态。

- [ ] **Step 3: 训练侧 forward 应用 decay**

`hf_models/hf_qwen3/modeling_qwen3.py:274`，把:

```python
                # accumulate: history (detached if enabled) + current differentiable write
                S = S_hist + dW_i
```

改为:

```python
                # accumulate with decay gate: (1-α)·history + current write.
                # decay=0 → pure NLMS (S = S_hist + dW_i, bit-identical to before);
                # decay>0 → ‖S‖ bounded (~‖dW‖/α), breaks runaway feedback, and the
                # cross-chunk gradient chain decays as (1-α)^k (no detach needed).
                S = (1.0 - self.ttt_nlms_decay) * S_hist + dW_i
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/python -m pytest tests/test_ttt_nlms_train.py::test_train_nlms_decay_bounds_S tests/test_ttt_nlms_train.py::test_train_nlms_decay_zero_is_pure_nlms -q`
Expected: PASS（decay=0 逐位等价 + 有界性数学成立）

- [ ] **Step 5: 运行训练侧 NLMS 全回归**

Run: `.venv/bin/python -m pytest tests/test_ttt_nlms_train.py -q`
Expected: PASS（所有既有 NLMS 训练测试不破，因 decay 默认 0.0）

- [ ] **Step 6: 提交**

```bash
git add hf_models/hf_qwen3/modeling_qwen3.py tests/test_ttt_nlms_train.py
git commit -m "feat: training-side NLMS decay gate S<-(1-a)S+dW (a=0 bit-identical)"
```

---

## Task 4: 推理侧 _write_block 衰减累加 + α=0 退化测试

**Files:**
- Modify: `inference_model/hf_qwen3/modeling_qwen3.py:218`（`_write_block` nlms 分支返回）
- Test: `tests/test_ttt_nlms_write.py`

**背景:** 推理侧用 `current_w = W0 + ΔW`。decay 只能作用在 ΔW（= current_w − W0），不衰减 W0。当前 `:218` 是 `return current_w + dw`，但这是 outer/nlms 共用的返回。需在 nlms 分支内对 dw 累加前先衰减 ΔW。

- [ ] **Step 1: 写失败测试(推理侧 α=0 退化 + decay 数学)**

在 `tests/test_ttt_nlms_write.py` 末尾追加:

```python
def test_inference_nlms_decay_zero_is_pure_nlms():
    """推理侧 ttt_nlms_decay=0.0 的 _write_block 必须与无 decay 的纯 NLMS 逐位相同。"""
    import torch
    from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config
    from inference_model.hf_qwen3.modeling_qwen3 import Qwen3MLP
    torch.manual_seed(0)
    def mk(decay):
        cfg = Qwen3Config(
            vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=4, max_position_embeddings=64,
            ttt_layers=[0], ttt_mode=True, ttt_proj=True, ttt_lr=0.5, ttt_chunk=2,
            ttt_target="input_embed", ttt_write_rule="nlms", ttt_nlms_lambda=1.0,
            ttt_nlms_decay=decay,
        )
        torch.manual_seed(1)
        mlp = Qwen3MLP(cfg, layer_idx=0)
        with torch.no_grad():
            for p in mlp.parameters():
                p.normal_(0.0, 0.1)
        return mlp.eval()
    mlp_pure = mk(0.0)
    key = torch.randn(4, mlp_pure.down_proj.weight.shape[1])
    value = torch.randn(4, mlp_pure.down_proj.weight.shape[0])
    w0 = mlp_pure.down_proj.weight.clone()
    # 两次连续写入，确认 decay=0 与纯累加一致
    w_pure = mlp_pure._apply_ttt_write(w0.clone(), key, value)
    w_pure = mlp_pure._apply_ttt_write(w_pure, key, value)
    mlp_d0 = mk(0.0)
    w_d0 = mlp_d0._apply_ttt_write(w0.clone(), key, value)
    w_d0 = mlp_d0._apply_ttt_write(w_d0, key, value)
    torch.testing.assert_close(w_pure, w_d0)


def test_inference_nlms_decay_shrinks_delta():
    """decay>0 时第二次写入前 ΔW 应被 (1-α) 缩小：连续写同样 (k,v)，
    decay>0 的累积 ΔW 必须小于 decay=0 的累积 ΔW。"""
    import torch
    from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config
    from inference_model.hf_qwen3.modeling_qwen3 import Qwen3MLP
    def mk(decay):
        cfg = Qwen3Config(
            vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=4, max_position_embeddings=64,
            ttt_layers=[0], ttt_mode=True, ttt_proj=True, ttt_lr=0.5, ttt_chunk=2,
            ttt_target="input_embed", ttt_write_rule="nlms", ttt_nlms_lambda=1.0,
            ttt_nlms_decay=decay,
        )
        torch.manual_seed(1)
        mlp = Qwen3MLP(cfg, layer_idx=0)
        with torch.no_grad():
            for p in mlp.parameters():
                p.normal_(0.0, 0.1)
        return mlp.eval()
    torch.manual_seed(2)
    key = torch.randn(4, 8); value = torch.randn(4, 16)
    w0 = mk(0.0).down_proj.weight.clone()
    mlp0, mlpd = mk(0.0), mk(0.3)
    w0_acc = mlp0._apply_ttt_write(mlp0._apply_ttt_write(w0.clone(), key, value), key, value)
    wd_acc = mlpd._apply_ttt_write(mlpd._apply_ttt_write(w0.clone(), key, value), key, value)
    d0 = (w0_acc - w0).norm()
    dd = (wd_acc - w0).norm()
    assert dd < d0, f"decay should shrink accumulated delta: decay0={d0}, decay0.3={dd}"
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest tests/test_ttt_nlms_write.py::test_inference_nlms_decay_shrinks_delta -x -q`
Expected: FAIL（decay 尚未应用，decay=0.3 的 ΔW 与 decay=0 相同，`dd < d0` 不成立）

- [ ] **Step 3: 推理侧 _write_block 应用 decay**

`inference_model/hf_qwen3/modeling_qwen3.py:206-218`，把当前:

```python
    def _write_block(self, current_w, key, value, write_rule):
        if write_rule == "nlms":
            delta = current_w - self.down_proj.weight  # [d, h_dim]
            pred = contract("c h, d h -> c d", key, delta)  # key @ delta^T
            residual = value - pred  # [c, d]
            denom = self.ttt_nlms_lambda + (key * key).sum(dim=-1, keepdim=True)  # [c, 1] per-key
            residual = residual / denom
            dw = contract("c d, c h -> d h", residual, key) * self.ttt_lr
            # average per-key writes over the chunk (matches training; bounds S growth)
            dw = dw / key.shape[0]
        else:  # outer
            dw = contract("c h, c d -> d h", key, value) * self.ttt_lr
        return current_w + dw
```

替换为:

```python
    def _write_block(self, current_w, key, value, write_rule):
        if write_rule == "nlms":
            delta = current_w - self.down_proj.weight  # [d, h_dim]
            pred = contract("c h, d h -> c d", key, delta)  # key @ delta^T
            residual = value - pred  # [c, d]
            denom = self.ttt_nlms_lambda + (key * key).sum(dim=-1, keepdim=True)  # [c, 1] per-key
            residual = residual / denom
            dw = contract("c d, c h -> d h", residual, key) * self.ttt_lr
            # average per-key writes over the chunk (matches training; bounds S growth)
            dw = dw / key.shape[0]
            # decay gate: ΔW <- (1-α)·ΔW + dw (decay only on the delta, not W0).
            # decay=0 → current_w + dw (bit-identical pure NLMS).
            decay = getattr(self, "ttt_nlms_decay", 0.0)
            return self.down_proj.weight + (1.0 - decay) * delta + dw
        else:  # outer
            dw = contract("c h, c d -> d h", key, value) * self.ttt_lr
        return current_w + dw
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/python -m pytest tests/test_ttt_nlms_write.py -q`
Expected: PASS（既有推理 NLMS 测试 + 新 α=0 退化 + decay 缩小 ΔW 全过）

- [ ] **Step 5: 提交**

```bash
git add inference_model/hf_qwen3/modeling_qwen3.py tests/test_ttt_nlms_write.py
git commit -m "feat: inference-side NLMS decay gate on delta (a=0 bit-identical)"
```

---

## Task 5: 训推一致性(decay>0)+ 梯度非消失

**Files:**
- Test: `tests/test_ttt_nlms_train.py`

**背景:** 现有 `test_train_nlms_matches_inference_nlms` 只测 decay=0。需加 decay>0 的训推一致,以及冻结 backbone 下 decay 梯度非消失(对比 detach)。

- [ ] **Step 1: 写测试(decay>0 训推一致 + 梯度非消失)**

在 `tests/test_ttt_nlms_train.py` 末尾追加:

```python
def test_train_nlms_decay_matches_inference():
    """decay>0 时训练侧 chunk-loop NLMS 必须与推理侧逐 chunk NLMS 数值一致。"""
    from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config as InfConfig
    from inference_model.hf_qwen3.modeling_qwen3 import Qwen3MLP as InfMLP
    torch.manual_seed(0)
    train_cfg = _cfg(ttt_write_rule="nlms", ttt_lr=0.5, ttt_nlms_lambda=1.0, ttt_chunk=2,
                     ttt_nlms_decay=0.2)
    inf_cfg = InfConfig(
        vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4, max_position_embeddings=64,
        ttt_layers=[0], ttt_mode=True, ttt_proj=True, ttt_lr=0.5, ttt_chunk=2,
        ttt_target="input_embed", ttt_write_rule="nlms", ttt_nlms_lambda=1.0, ttt_nlms_decay=0.2,
    )
    torch.manual_seed(1)
    train_mlp = _randomize(Qwen3MLP(train_cfg, layer_idx=0))
    torch.manual_seed(1)
    inf_mlp = _randomize(InfMLP(inf_cfg, layer_idx=0))
    x = torch.randn(1, 4, 8); t = torch.randn(1, 4, 8)
    with torch.no_grad():
        train_out = train_mlp(x, t=t)
        inf_out, _ = inf_mlp(x, t=t)
    torch.testing.assert_close(train_out, inf_out, rtol=1e-4, atol=1e-5)


def test_train_nlms_decay_trains_under_frozen_backbone():
    """decay>0(不开 detach)+ 冻结 backbone:loss 可微且 ttt_proj/ttt_conv 梯度非零有限。
    对比 detach 的梯度消失（~1e-12），decay 的几何收敛应保留可见梯度。"""
    torch.manual_seed(9)
    x = torch.randn(1, 8, 8); t = torch.randn(1, 8, 8)
    torch.manual_seed(9)
    mlp = _randomize(Qwen3MLP(_cfg(ttt_write_rule="nlms", ttt_lr=0.5, ttt_chunk=2,
                                   ttt_nlms_decay=0.1), layer_idx=0))
    _freeze_backbone(mlp)
    loss = mlp(x, t=t).float().pow(2).mean()
    assert loss.requires_grad, "decay NLMS loss must require grad under frozen backbone"
    loss.backward()
    for name in ("ttt_proj", "ttt_conv"):
        g = getattr(mlp, name).weight.grad
        assert g is not None and g.norm() > 0, f"{name} must receive nonzero grad with decay"
        assert torch.isfinite(g).all()
```

- [ ] **Step 2: 运行确认通过**

Run: `.venv/bin/python -m pytest tests/test_ttt_nlms_train.py::test_train_nlms_decay_matches_inference tests/test_ttt_nlms_train.py::test_train_nlms_decay_trains_under_frozen_backbone -q`
Expected: PASS（Task 3+4 的两侧 decay 实现已对称，训推一致；decay 不 detach，梯度非零）

- [ ] **Step 3: 全 NLMS 回归**

Run: `.venv/bin/python -m pytest tests/test_ttt_nlms_train.py tests/test_ttt_nlms_write.py -q`
Expected: PASS（全部通过）

- [ ] **Step 4: 提交**

```bash
git add tests/test_ttt_nlms_train.py
git commit -m "test: NLMS decay train/infer parity + nonzero grad under frozen backbone"
```

---

## Task 6: matched CPT wrapper 透传 NLMS_DECAY

**Files:**
- Modify: `scripts/train/longsft/qwen3-1.7b/run_stage3_nlms_matched.sh:18-22`（env）和 `:37`（nlms 分支 JSON）

- [ ] **Step 1: 加 NLMS_DECAY env**

`scripts/train/longsft/qwen3-1.7b/run_stage3_nlms_matched.sh:22`，在 `NLMS_DETACH_VALUE="${NLMS_DETACH:-true}"` 之后插入:

```bash
# decay gate alpha for NLMS state: S<-(1-a)S+dW, bounds ||S|| to break runaway
# feedback. Default 0.1 for the nlms arm; 0.0 reproduces pure (diverging) NLMS.
NLMS_DECAY_VALUE="${NLMS_DECAY:-0.1}"
```

- [ ] **Step 2: nlms 分支 JSON 加 decay 字段**

`scripts/train/longsft/qwen3-1.7b/run_stage3_nlms_matched.sh:37`，把 nlms 分支的:

```bash
      export MODEL_FOUNDATION_JSON="{\"ttt_write_rule\": \"nlms\", \"ttt_lr\": ${TTT_LR_VALUE}, \"ttt_nlms_lambda\": ${NLMS_LAMBDA}, \"ttt_nlms_detach_state\": ${NLMS_DETACH_VALUE}, \"ttt_train_only\": ${TTT_TRAIN_ONLY_VALUE}}"
```

改为:

```bash
      export MODEL_FOUNDATION_JSON="{\"ttt_write_rule\": \"nlms\", \"ttt_lr\": ${TTT_LR_VALUE}, \"ttt_nlms_lambda\": ${NLMS_LAMBDA}, \"ttt_nlms_detach_state\": ${NLMS_DETACH_VALUE}, \"ttt_nlms_decay\": ${NLMS_DECAY_VALUE}, \"ttt_train_only\": ${TTT_TRAIN_ONLY_VALUE}}"
```

- [ ] **Step 3: 校验脚本语法**

Run: `bash -n scripts/train/longsft/qwen3-1.7b/run_stage3_nlms_matched.sh`
Expected: 无输出（语法正确）

- [ ] **Step 4: 提交**

```bash
git add scripts/train/longsft/qwen3-1.7b/run_stage3_nlms_matched.sh
git commit -m "feat: matched-CPT wrapper passes NLMS_DECAY (default 0.1) to nlms arm"
```

---

## Task 7: 真冒烟 sweep (η, α) 验证

**Files:** 无代码改动,运行 + 记录。

**前提:** Task 1-6 全部完成。需 2 张空闲 GPU(<15GB 占用)。stage2 ckpt: gs11000(HF 已转换)。

- [ ] **Step 1: 跑首组冒烟 η=5, α=0.1**

确认 2 张卡空闲(`nvidia-smi`),然后:

```bash
cd /zouxiangyu/codes/Learning/In-Place-TTT
LOG=refine-logs/smoke/nlms_decay_eta5_a0.1.log
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=<GPU_A>,<GPU_B> \
WRITE_RULE=nlms TTT_LR=5.0 NLMS_DECAY=0.1 NLMS_DETACH=false \
STAGE2_GLOBAL_STEP=11000 MAX_STEPS=2 SAVE_STEPS=0 \
EXP_NAME=qwen3-1.7b-stage3-nlms-decay-smoke MASTER_PORT=12390 WANDB_MODE=offline \
nohup bash scripts/train/longsft/qwen3-1.7b/run_stage3_nlms_matched.sh > "$LOG" 2>&1 &
```

等约 4 分钟,读 `./logs/log-qwen3-1.7b-stage3-nlms-decay-smoke_node0_*.txt` 的 `Epoch 1/1` 行。

- [ ] **Step 2: 判定首组三判据**

- `ttt_dw` 有界（非天文数字，对比 decay=0/η=5 的 2.58e36）
- `ttt_do` 跳出 bf16 噪声地板（> 1.66e-3，理想落 [0.2,0.4]）
- loss 不发散（对比 decay=0/η=5 的 5.27→7.07↑）

跑完停进程 + `rm -rf outputs/qwen3-1.7b-stage3-nlms-decay-smoke`。

- [ ] **Step 3: 按首组结果 sweep 其余 (η,α)**

若首组三判据通过 → 记录为候选。若 ttt_do 仍埋噪声地板 → 提 η（10, 30）；若仍发散 → 提 α（0.3）。
网格:η∈{1,5,(10,30 视需要)}、α∈{0.05,0.1,0.3}。每组同样 2 步,记录 ttt_dw/ttt_do/loss。

- [ ] **Step 4: 记录结论到实验日志**

把 (η,α) sweep 表格 + 选定的"有界+可学"组合写入 `refine-logs/EXPERIMENT_RESULTS_2026-06-29.md`。判据：存在 (η,α) 使三判据同时满足 → C 成立，交完整 matched CPT；若网格无任一组合 → C 不足，记录并停下与用户讨论严格序贯/DeltaNet form。

- [ ] **Step 5: 提交**

```bash
git add refine-logs/EXPERIMENT_RESULTS_2026-06-29.md
git commit -m "feat: NLMS decay-gate smoke sweep (eta,alpha) + selected stable+learnable config"
```

---

## Self-Review

**Spec coverage:**
- 核心修正 decay gate `S←(1−α)S+dW` → Task 3(训练)+ Task 4(推理)✓
- 改动 1 训练 forward → Task 3 ✓
- 改动 2 推理 _write_block(ΔW 衰减,不碰 W0)→ Task 4 ✓
- 改动 3 两侧 config flag + 校验 [0,1) → Task 1 ✓
- 改动 4 wrapper NLMS_DECAY env → Task 6 ✓
- 改动 5 白名单无需改 → spec 已说明,无 task(正确)✓
- 测试 1 α=0 退化 → Task 3 Step1(训练)+ Task 4 Step1(推理)✓
- 测试 2 config 校验 → Task 1 Step1 ✓
- 测试 3 有界性 → Task 3 Step1 ✓
- 测试 4 训推一致 → Task 5 Step1 ✓
- 测试 5 梯度非消失 → Task 5 Step1 ✓
- 验证路径 真冒烟 sweep → Task 7 ✓
- 护栏 outer 不变/短上下文/α=0 退化 → α=0 退化在 Task3/4 守门,outer 分支未碰(Task4 仅改 nlms 分支)✓

**Placeholder scan:** 无 TBD/TODO。Task 7 的 `<GPU_A>,<GPU_B>` 是运行时按 `nvidia-smi` 填的真实卡号(资源依赖,执行时定),已在 Step1 说明确认空闲卡——属可接受的运行时参数,非代码占位。

**Type consistency:** `ttt_nlms_decay` 在两侧 config/MLP 同名;decay 应用公式训练侧 `(1-decay)*S_hist+dW_i`、推理侧 `W0+(1-decay)*delta+dw` 语义一致(均对 ΔW 衰减);`_freeze_backbone` 在 Task 5 复用 Task(已有于 test 文件,commit d712cd7 引入)。校验范围 `[0.0,1.0)` 两侧一致。
