# ttt_lr warmup(NLMS 冷启动稳定化)实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 TTT 写规则系数 ttt_lr 在训练前期从极小值线性 warmup 到目标值,保护 NLMS 冷启动不发散。

**Architecture:** 无状态纯 global_step 函数 `ttt_lr_warmup_factor` + 训练循环每步 setter `set_ttt_lr_effective`(复用 `_iter_decoder_layers` 遍历 TTT 层)注入 `mlp.ttt_lr_effective`,forward 三个乘点改用该动态值。`warmup_steps=0` 时 `ttt_lr_effective ≡ ttt_lr`,bit 级向后兼容。

**Tech Stack:** PyTorch, HF Qwen3 自定义 modeling, FSDP, pytest, `.venv/bin/python`。

**来源 spec:** `docs/superpowers/specs/2026-06-30-ttt-lr-warmup-design.md`

---

## File Structure

- `hf_models/hf_qwen3/configuration_qwen3.py` — 训练侧 config 声明 `ttt_lr_warmup_steps`/`ttt_lr_warmup_init`(Task 1)
- `hf_models/hf_qwen3/modeling_qwen3.py` — MLP `__init__` 加 `ttt_lr_effective`(Task 2)+ forward 三乘点改用(Task 3)
- `in_place_ttt/ttt_aux/training.py` — `ttt_lr_warmup_factor` 纯函数 + `set_ttt_lr_effective` setter(Task 4)
- `tasks/train_torch.py` — 训练循环每步调用 setter(Task 5)
- `tests/test_ttt_lr_warmup.py` — warmup 函数 + config + setter + 向后兼容(Task 1/2/4)
- `scripts/train/longsft/qwen3-1.7b/run_stage3_nlms_matched.sh` — env 透传(Task 6)

---

## Task 1: config 声明 warmup flag

**Files:**
- Modify: `hf_models/hf_qwen3/configuration_qwen3.py:195`(构造参数)和 `:277`（赋值校验后）
- Test: `tests/test_ttt_lr_warmup.py`（新建）

- [ ] **Step 1: 写失败测试(config 默认值 + 校验)**

新建 `tests/test_ttt_lr_warmup.py`，写入:

```python
"""ttt_lr warmup: pure factor fn + config flags + setter + backward compat."""
import pytest
import torch

from hf_models.hf_qwen3.configuration_qwen3 import Qwen3Config
from hf_models.hf_qwen3.modeling_qwen3 import Qwen3MLP


def _cfg(**ov):
    kw = dict(
        vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4, max_position_embeddings=64,
        ttt_layers=[0], ttt_mode=True, ttt_proj=True, ttt_lr=0.5, ttt_chunk=2,
        ttt_target="input_embed",
    )
    kw.update(ov)
    return Qwen3Config(**kw)


def test_config_warmup_defaults_and_validation():
    # 默认:无 warmup
    c = _cfg()
    assert c.ttt_lr_warmup_steps == 0
    assert c.ttt_lr_warmup_init == 0.0
    # 合法值
    c2 = _cfg(ttt_lr_warmup_steps=200, ttt_lr_warmup_init=1e-3)
    assert c2.ttt_lr_warmup_steps == 200
    assert c2.ttt_lr_warmup_init == 1e-3
    # 非法值
    with pytest.raises(ValueError):
        _cfg(ttt_lr_warmup_steps=-1)
    with pytest.raises(ValueError):
        _cfg(ttt_lr_warmup_init=-0.5)
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest tests/test_ttt_lr_warmup.py::test_config_warmup_defaults_and_validation -x -q`
Expected: FAIL（`AttributeError: ... has no attribute 'ttt_lr_warmup_steps'`）

- [ ] **Step 3: config 加 flag**

`hf_models/hf_qwen3/configuration_qwen3.py:195`，在 `ttt_nlms_decay=0.0,` 之后插入:

```python
        ttt_lr_warmup_steps=0,
        ttt_lr_warmup_init=0.0,
```

在 `:277` 的 `raise ValueError("ttt_nlms_decay must be in [0.0, 1.0)")` 之后插入:

```python
        # Linear warmup for the TTT write-rule lr (ttt_lr): protects the NLMS
        # cold-start (projections not yet converging residuals) from divergence.
        # warmup_steps=0 → no warmup (ttt_lr_effective == ttt_lr, bit-identical).
        self.ttt_lr_warmup_steps = int(ttt_lr_warmup_steps)
        if self.ttt_lr_warmup_steps < 0:
            raise ValueError("ttt_lr_warmup_steps must be >= 0")
        self.ttt_lr_warmup_init = float(ttt_lr_warmup_init)
        if self.ttt_lr_warmup_init < 0:
            raise ValueError("ttt_lr_warmup_init must be >= 0")
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/python -m pytest tests/test_ttt_lr_warmup.py::test_config_warmup_defaults_and_validation -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add hf_models/hf_qwen3/configuration_qwen3.py tests/test_ttt_lr_warmup.py
git commit -m "feat: declare ttt_lr_warmup_steps/ttt_lr_warmup_init on training config"
```

---

## Task 2: MLP 初始化 ttt_lr_effective

**Files:**
- Modify: `hf_models/hf_qwen3/modeling_qwen3.py:98`（`__init__`）
- Test: `tests/test_ttt_lr_warmup.py`

- [ ] **Step 1: 写失败测试(MLP 默认 ttt_lr_effective == ttt_lr)**

在 `tests/test_ttt_lr_warmup.py` 末尾追加:

```python
def test_mlp_ttt_lr_effective_defaults_to_ttt_lr():
    mlp = Qwen3MLP(_cfg(ttt_lr=0.5), layer_idx=0)
    assert mlp.ttt_lr_effective == mlp.ttt_lr == 0.5
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest tests/test_ttt_lr_warmup.py::test_mlp_ttt_lr_effective_defaults_to_ttt_lr -x -q`
Expected: FAIL（`AttributeError: ... has no attribute 'ttt_lr_effective'`）

- [ ] **Step 3: MLP `__init__` 加 ttt_lr_effective**

`hf_models/hf_qwen3/modeling_qwen3.py:98`，把:

```python
            self.ttt_lr = getattr(config, "ttt_lr", 0.3)
```

改为:

```python
            self.ttt_lr = getattr(config, "ttt_lr", 0.3)
            # effective write-rule lr; defaults to target ttt_lr. The training loop
            # overwrites this each step during warmup (set_ttt_lr_effective). eval /
            # inference keep the target value.
            self.ttt_lr_effective = self.ttt_lr
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/python -m pytest tests/test_ttt_lr_warmup.py::test_mlp_ttt_lr_effective_defaults_to_ttt_lr -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add hf_models/hf_qwen3/modeling_qwen3.py tests/test_ttt_lr_warmup.py
git commit -m "feat: MLP exposes ttt_lr_effective (defaults to ttt_lr)"
```

---

## Task 3: forward 三个乘点改用 ttt_lr_effective

**Files:**
- Modify: `hf_models/hf_qwen3/modeling_qwen3.py:268`（NLMS）、`:292`/`:309`（outer/key_norm）
- Test: `tests/test_ttt_lr_warmup.py`

**背景:** 三个乘点当前用 `self.ttt_lr`。改用 `self.ttt_lr_effective`。因默认相等,无 warmup 时 bit 级不变。

- [ ] **Step 1: 写测试(默认相等时 forward bit 级不变)**

在 `tests/test_ttt_lr_warmup.py` 末尾追加:

```python
def _randomize(mlp):
    with torch.no_grad():
        for p in mlp.parameters():
            p.normal_(0.0, 0.1)
    return mlp.eval()


def test_forward_unchanged_when_effective_equals_target():
    """ttt_lr_effective 默认 == ttt_lr,forward 必须与改动前一致(用 outer 与 nlms 各验)。"""
    for rule in ("outer", "nlms"):
        torch.manual_seed(7)
        mlp = _randomize(Qwen3MLP(_cfg(ttt_write_rule=rule, ttt_lr=0.5, ttt_chunk=2), layer_idx=0))
        x = torch.randn(1, 4, 8); t = torch.randn(1, 4, 8)
        with torch.no_grad():
            out_default = mlp(x, t=t)
        # 显式把 effective 设成与 ttt_lr 相同,结果必须一致
        mlp.ttt_lr_effective = mlp.ttt_lr
        with torch.no_grad():
            out_explicit = mlp(x, t=t)
        torch.testing.assert_close(out_default, out_explicit)


def test_forward_scales_with_effective_lr():
    """改 ttt_lr_effective 必须改变 NLMS 写入(证明 forward 真的用了它而非 ttt_lr)。"""
    torch.manual_seed(7)
    mlp = _randomize(Qwen3MLP(_cfg(ttt_write_rule="nlms", ttt_lr=0.5, ttt_chunk=2), layer_idx=0))
    x = torch.randn(1, 4, 8); t = torch.randn(1, 4, 8)
    with torch.no_grad():
        out_full = mlp(x, t=t)
        mlp.ttt_lr_effective = 0.0   # 写入归零 → 退化到 base
        out_zero = mlp(x, t=t)
    assert not torch.allclose(out_full, out_zero), "forward must respond to ttt_lr_effective"
```

- [ ] **Step 2: 运行确认(test_forward_scales 应失败:forward 还在用 ttt_lr)**

Run: `.venv/bin/python -m pytest tests/test_ttt_lr_warmup.py::test_forward_scales_with_effective_lr -x -q`
Expected: FAIL（forward 用 self.ttt_lr,改 ttt_lr_effective 无效,out_zero == out_full）

- [ ] **Step 3: 三个乘点改用 ttt_lr_effective**

`hf_models/hf_qwen3/modeling_qwen3.py:268`，把:

```python
                dW_i = contract("b c d, b c h -> b d h", resid_i, Ki) * self.ttt_lr   # [b, d, h_dim]
```

改为:

```python
                dW_i = contract("b c d, b c h -> b d h", resid_i, Ki) * self.ttt_lr_effective   # [b, d, h_dim]
```

`:292`，把:

```python
            delta_down_proj = d_down_proj * self.ttt_lr
```

改为:

```python
            delta_down_proj = d_down_proj * self.ttt_lr_effective
```

`:309`（另一处同样的 `delta_down_proj = d_down_proj * self.ttt_lr`），改为:

```python
            delta_down_proj = d_down_proj * self.ttt_lr_effective
```

注:`:292` 和 `:309` 文本相同,用 Read 确认各自上下文后逐个替换(不要 replace_all,以免漏改或误改)。

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/python -m pytest tests/test_ttt_lr_warmup.py -q`
Expected: PASS（forward 现在用 effective:默认相等不变 + 改 effective 生效）

- [ ] **Step 5: 全 NLMS 回归(确认未破既有)**

Run: `.venv/bin/python -m pytest tests/test_ttt_nlms_train.py tests/test_ttt_nlms_write.py -q`
Expected: PASS（23 个既有测试不破,因 ttt_lr_effective 默认 == ttt_lr）

- [ ] **Step 6: 提交**

```bash
git add hf_models/hf_qwen3/modeling_qwen3.py tests/test_ttt_lr_warmup.py
git commit -m "feat: forward uses ttt_lr_effective (defaults to ttt_lr, bit-identical)"
```

---

## Task 4: warmup 纯函数 + setter

**Files:**
- Modify: `in_place_ttt/ttt_aux/training.py`（加 `ttt_lr_warmup_factor` + `set_ttt_lr_effective`,在 `_iter_decoder_layers` 之后)
- Test: `tests/test_ttt_lr_warmup.py`

- [ ] **Step 1: 写失败测试(warmup 函数数学 + setter 注入)**

在 `tests/test_ttt_lr_warmup.py` 末尾追加:

```python
from in_place_ttt.ttt_aux.training import ttt_lr_warmup_factor, set_ttt_lr_effective


def test_warmup_factor_math():
    # warmup_steps=0 → 恒 target
    assert ttt_lr_warmup_factor(0, 0, 0.0, 3.0) == 3.0
    assert ttt_lr_warmup_factor(100, 0, 0.0, 3.0) == 3.0
    # step=0 → init
    assert ttt_lr_warmup_factor(0, 200, 1e-3, 3.0) == pytest.approx(1e-3)
    # 中点 → (init+target)/2
    assert ttt_lr_warmup_factor(100, 200, 1.0, 3.0) == pytest.approx(2.0)
    # step >= warmup_steps → target
    assert ttt_lr_warmup_factor(200, 200, 1e-3, 3.0) == pytest.approx(3.0)
    assert ttt_lr_warmup_factor(500, 200, 1e-3, 3.0) == pytest.approx(3.0)
    # 单调递增、夹在 [init, target]
    vals = [ttt_lr_warmup_factor(s, 200, 0.1, 3.0) for s in range(0, 220, 20)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    assert all(0.1 <= v <= 3.0 + 1e-9 for v in vals)


def test_set_ttt_lr_effective_injects_into_ttt_layers():
    """setter 必须把值写进每个 TTT MLP 的 ttt_lr_effective。用单层 TTT 模型验证。"""
    from hf_models.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(_cfg(ttt_lr=0.5))
    set_ttt_lr_effective(model, 0.123)
    inner = getattr(model, "model", model)
    found = False
    for layer in inner.layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "ttt_lr_effective"):
            assert mlp.ttt_lr_effective == pytest.approx(0.123)
            found = True
    assert found, "at least one TTT MLP must have been updated"
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest tests/test_ttt_lr_warmup.py::test_warmup_factor_math -x -q`
Expected: FAIL（`ImportError: cannot import name 'ttt_lr_warmup_factor'`）

- [ ] **Step 3: 实现 warmup 函数 + setter**

`in_place_ttt/ttt_aux/training.py`，在 `_iter_decoder_layers` 函数（`:15` 起）之后插入:

```python
def ttt_lr_warmup_factor(step, warmup_steps, init, target):
    """Linear warmup of the TTT write-rule lr. Stateless function of global_step.

    step < warmup_steps: linearly interpolate init -> target.
    step >= warmup_steps (or warmup_steps <= 0): target.
    """
    if warmup_steps <= 0:
        return float(target)
    frac = min(max(int(step), 0) / float(warmup_steps), 1.0)
    return float(init) + (float(target) - float(init)) * frac


def set_ttt_lr_effective(model, lr):
    """Set ttt_lr_effective on every TTT MLP. Called each training step for warmup."""
    lr = float(lr)
    for layer in _iter_decoder_layers(model):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "ttt_lr_effective"):
            mlp.ttt_lr_effective = lr
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/python -m pytest tests/test_ttt_lr_warmup.py -q`
Expected: PASS（warmup 函数数学 + setter 注入 + 前面所有测试）

- [ ] **Step 5: 提交**

```bash
git add in_place_ttt/ttt_aux/training.py tests/test_ttt_lr_warmup.py
git commit -m "feat: ttt_lr_warmup_factor (linear) + set_ttt_lr_effective setter"
```

---

## Task 5: 训练循环每步调用 setter

**Files:**
- Modify: `tasks/train_torch.py:40-45`（import）和 `:665`（global_step+=1 之后）

**背景:** `model_config = model.config`(:471)已存在。target = `model_config.ttt_lr`,warmup 参数同从 model_config 读。setter 每步在 micro_batch 循环前调一次。

- [ ] **Step 1: import warmup 函数**

`tasks/train_torch.py:40-45` 的 import 块（`from in_place_ttt.ttt_aux.training import (`），在 `configure_ttt_only_trainable_params as _configure_ttt_only_trainable_params,` 那一行之后加:

```python
    ttt_lr_warmup_factor as _ttt_lr_warmup_factor,
    set_ttt_lr_effective as _set_ttt_lr_effective,
```

- [ ] **Step 2: 训练循环每步调用 setter**

`tasks/train_torch.py:665`，把:

```python
            global_step += 1

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)
```

改为:

```python
            global_step += 1

            # ttt_lr warmup: set the effective write-rule lr from the current step.
            # Stateless (pure fn of global_step) → resume-safe. No-op when
            # ttt_lr_warmup_steps == 0 (effective stays at target ttt_lr).
            _ttt_warmup_steps = int(getattr(model_config, "ttt_lr_warmup_steps", 0))
            if _ttt_warmup_steps > 0:
                _eff_ttt_lr = _ttt_lr_warmup_factor(
                    global_step,
                    _ttt_warmup_steps,
                    float(getattr(model_config, "ttt_lr_warmup_init", 0.0)),
                    float(getattr(model_config, "ttt_lr", 0.3)),
                )
                _set_ttt_lr_effective(model, _eff_ttt_lr)

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)
```

- [ ] **Step 3: 语法 + import 校验**

Run: `.venv/bin/python -c "import ast; ast.parse(open('tasks/train_torch.py').read()); print('syntax OK')"`
Expected: `syntax OK`

Run: `.venv/bin/python -c "from in_place_ttt.ttt_aux.training import ttt_lr_warmup_factor, set_ttt_lr_effective; print('import OK')"`
Expected: `import OK`

- [ ] **Step 4: 提交**

```bash
git add tasks/train_torch.py
git commit -m "feat: training loop applies ttt_lr warmup each step (no-op when steps=0)"
```

---

## Task 6: wrapper env 透传

**Files:**
- Modify: `scripts/train/longsft/qwen3-1.7b/run_stage3_nlms_matched.sh`（env + nlms 分支 JSON）

- [ ] **Step 1: 加 warmup env**

`scripts/train/longsft/qwen3-1.7b/run_stage3_nlms_matched.sh`，在 `NLMS_DECAY_VALUE="${NLMS_DECAY:-0.1}"` 之后插入:

```bash
# ttt_lr warmup for NLMS cold-start: ttt_lr ramps init->target over N steps.
# 0 steps = no warmup. Defaults protect the nlms cold-start.
TTT_LR_WARMUP_STEPS_VALUE="${TTT_LR_WARMUP_STEPS:-200}"
TTT_LR_WARMUP_INIT_VALUE="${TTT_LR_WARMUP_INIT:-0.001}"
```

- [ ] **Step 2: nlms 分支 JSON 加 warmup 字段**

把 nlms 分支的 `export MODEL_FOUNDATION_JSON=...` 那一行（含 `\"ttt_nlms_decay\": ${NLMS_DECAY_VALUE}`），在 `${NLMS_DECAY_VALUE},` 之后、`\"ttt_train_only\"` 之前插入 warmup 字段。完整替换为:

```bash
      export MODEL_FOUNDATION_JSON="{\"ttt_write_rule\": \"nlms\", \"ttt_lr\": ${TTT_LR_VALUE}, \"ttt_nlms_lambda\": ${NLMS_LAMBDA}, \"ttt_nlms_detach_state\": ${NLMS_DETACH_VALUE}, \"ttt_nlms_decay\": ${NLMS_DECAY_VALUE}, \"ttt_lr_warmup_steps\": ${TTT_LR_WARMUP_STEPS_VALUE}, \"ttt_lr_warmup_init\": ${TTT_LR_WARMUP_INIT_VALUE}, \"ttt_train_only\": ${TTT_TRAIN_ONLY_VALUE}}"
```

- [ ] **Step 3: 语法校验**

Run: `bash -n scripts/train/longsft/qwen3-1.7b/run_stage3_nlms_matched.sh`
Expected: 无输出（语法正确）

- [ ] **Step 4: 提交**

```bash
git add scripts/train/longsft/qwen3-1.7b/run_stage3_nlms_matched.sh
git commit -m "feat: wrapper passes ttt_lr_warmup_steps/init to nlms arm (default 200/1e-3)"
```

---

## Task 7: 短验证(4k / ~300 步)

**Files:** 无代码改动,运行 + 记录。

**前提:** Task 1-6 完成。2 张空闲 GPU。stage2 ckpt gs11000。

- [ ] **Step 1: 跑短验证冒烟**

确认 2 张卡空闲(`nvidia-smi`),然后:

```bash
cd /zouxiangyu/codes/Learning/In-Place-TTT
LOG=refine-logs/smoke/nlms_warmup_4k_300step.log
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=<A>,<B> \
WRITE_RULE=nlms TTT_LR=3.0 NLMS_DETACH=false NLMS_DECAY=0.0 \
TTT_LR_WARMUP_STEPS=200 TTT_LR_WARMUP_INIT=0.001 \
STAGE2_GLOBAL_STEP=11000 MAX_STEPS=300 SAVE_STEPS=0 MAX_SEQ_LEN=4096 \
EXP_NAME=qwen3-1.7b-stage3-nlms-warmup-val MASTER_PORT=12395 WANDB_MODE=offline \
nohup bash scripts/train/longsft/qwen3-1.7b/run_stage3_nlms_matched.sh > "$LOG" 2>&1 &
```

注:`MAX_SEQ_LEN=4096` 把长度从 64k 降到 4k(短验证,省 GPU);若 wrapper/template 不透传 MAX_SEQ_LEN,执行时在底层脚本确认 env 名(参考 run_pretrain_template.sh 的 `MAX_SEQ_LEN` → `--data.max_seq_len`,已确认存在)。

- [ ] **Step 2: 监控三判据(读训练日志的 Epoch 行)**

读 `./logs/log-qwen3-1.7b-stage3-nlms-warmup-val_node0_*.txt`,看 ttt_dw / ttt_do / loss 随 step 变化:
- **判据 1**:冷启动前期(step < 200,warmup 中)`ttt_dw` 有界,不出现 2.58e36。
- **判据 2**:warmup 结束后(step > 200)`ttt_do` > 1.66e-3(跳出 bf16 噪声地板)。
- **判据 3**:loss 平稳下降,不发散(不出现持续上升/NaN)。

- [ ] **Step 3: 记录结论**

把三判据观测 + ttt_dw/ttt_do/loss 关键 step 的值写入 `refine-logs/EXPERIMENT_RESULTS_2026-06-29.md`。
- 三判据满足 → 冷启动方案成立,记录"可上全量 64k stage3 matched 双臂"。
- 否则 → 记录失败模式,回路线讨论(叠加 decay / chunk 内序贯 / 共训)。

跑完停进程 + `rm -rf outputs/qwen3-1.7b-stage3-nlms-warmup-val`。

- [ ] **Step 4: 提交**

```bash
git add refine-logs/EXPERIMENT_RESULTS_2026-06-29.md
git commit -m "feat: ttt_lr warmup short validation (4k/300step) results"
```

---

## Self-Review

**Spec coverage:**
- 改动 1 config 声明 + 校验 → Task 1 ✓
- 改动 2 MLP ttt_lr_effective → Task 2 ✓
- 改动 3 forward 三乘点 → Task 3 ✓
- 改动 4 warmup_fn + setter → Task 4 ✓
- 改动 5 训练循环每步调用 → Task 5 ✓
- 改动 6 wrapper env → Task 6 ✓
- 测试 1 warmup 函数数学 → Task 4 Step1 ✓
- 测试 2 config 校验 → Task 1 Step1 ✓
- 测试 3 MLP 默认相等 → Task 2 Step1 ✓
- 测试 4 setter 注入 → Task 4 Step1 ✓
- 测试 5 向后兼容(warmup=0 bit级) → Task 3 Step1(默认相等)+ Step5(全回归)✓
- 短验证三判据 → Task 7 ✓
- 护栏 warmup=0 退化/outer不受影响/resume → warmup=0 由 Task3 Step1+5 守门;resume 是纯函数(Task5 注释说明,无状态故无需额外测试)✓

**Placeholder scan:** 无 TBD/TODO。Task 7 的 `<A>,<B>` 是运行时按 nvidia-smi 填的真实卡号(资源依赖),Step1 已说明确认空闲卡——可接受的运行时参数。

**Type consistency:** `ttt_lr_effective`(属性名)、`ttt_lr_warmup_factor(step, warmup_steps, init, target)`(签名)、`set_ttt_lr_effective(model, lr)`(签名)、config 字段 `ttt_lr_warmup_steps`/`ttt_lr_warmup_init` 在所有 Task 一致。train_torch 读 `model_config.ttt_lr`(target)/`.ttt_lr_warmup_steps`/`.ttt_lr_warmup_init`,与 config 声明一致。
