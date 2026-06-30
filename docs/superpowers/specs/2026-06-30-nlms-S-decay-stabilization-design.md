# 设计:NLMS 累积 S 衰减门控稳定化(方案 C)

**日期**: 2026-06-30
**来源**: 接续 M3 阶段 B — D 方向(提 η)证伪后转向 C
**前置依据**: refine-logs/EXPERIMENT_RESULTS_2026-06-29.md(D sweep 死结分析)、commit d712cd7(detach 修复)

## 背景与问题

M3 阶段 B 要在训练侧实现 per-key NLMS 写规则,跑 matched CPT 验证能否消除 cross-talk、把 RULER 0.285 拉回甚至超过关-TTT 上界 0.458。但训练遇到双重死结:

1. **detach 梯度消失**(commit d712cd7 已修 backward 断开,但仍学不动):1-step truncated BPTT 把梯度截到 ~1e-12。
2. **D 方向(提 η)证伪**(完整 sweep η∈{0.03,0.3,1.0,5.0}):
   - η≤1.0:`ttt_dw` 随 η 线性放大但绝对量级始终可忽略;`ttt_do=1.66e-3` 四次逐位相同 = 真实写入埋在 bf16 噪声地板下,学不动。
   - η=5.0:`ttt_dw=2.58e36` 爆炸,loss 5.27→7.07 上升。
   - 悬崖在 (1.0, 5.0],跨 46 个数量级,**无"既可学又稳定"的 η**。

**根因**:`resid = V − K·Sᵀ`。此 ckpt 的 ttt_proj/ttt_conv 在 **outer 动力学**下训练,NLMS 读出 `K·Sᵀ` 无法预测 V → S 增长时残差被**放大而非收缩** → dW 放大 → S 再放大,**runaway 正反馈**。这是 forward 侧 S 累加的结构性发散,detach(只切 backward)无效,η 量级也压不住。

## 核心修正:全局常量衰减门控(decay gate)

对累积 S 的每 chunk 更新从纯累加改为几何衰减:

```
S  ←  (1−α)·S + dW_i        # α ∈ [0, 1)，全局常量超参
```

**为什么这治本**:
- **有界性**:稳态 `‖S‖ ≲ ‖dW‖/α`,α>0 即收敛,数学上保证 forward 不爆炸 → 打断 runaway 正反馈。
- **梯度链几何收敛**:跨 chunk 贡献按 `(1−α)^k` 衰减,完整 BPTT 也不爆,且不像 detach 那样把梯度截到 0 → 同时治掉 D 暴露的梯度消失。
- **退化一致**:`α=0` 时 `S ← S + dW_i`,逐位回到当前纯 NLMS 实现 → 天然向后兼容。
- **零新增可训练参数**:α 是标量超参,保住 NLMS matched 同参数量的干净对比优点。

**设计决策**(brainstorming 已确认):
- S 限幅机制 = **decay gate**(而非 readout RMSNorm:后者 S 仍无界、且引入 RMSNorm 权重破坏 matched 同参数量)。
- α 形态 = **全局常量超参**(而非 per-layer 可学 / 输入相关动态:后两者引入新参数,违背最小改动)。
- decay 与 detach 关系 = **decay 代替 detach,默认完整 BPTT**(decay 已让梯度链几何收敛,无需暴力截断;detach flag 保留作可选对照)。
- 验证路径 = **直接真冒烟 sweep (η, α)**(decay 有界性 toy 已能证明,真冒烟看真实尺度行为最可信)。

## 代码改动(两侧对称)

训推必须语义一致,由现有 `test_train_nlms_matches_inference_nlms` 守护。

### 改动 1 — 训练侧 forward 衰减累加
文件:`hf_models/hf_qwen3/modeling_qwen3.py:274`(nlms 分支 chunk-loop 末)。
当前:`S = S_hist + dW_i`。
改为:`S = (1 - self.ttt_nlms_decay) * S_hist + dW_i`。
其余(base/delta 双路、readout 读 live S、per-key 归一化、per-chunk 平均)全部不变。

### 改动 2 — 推理侧 _write_block 衰减累加
文件:`inference_model/hf_qwen3/modeling_qwen3.py:218`(`_write_block` nlms 分支返回)。
当前:`return current_w + dw`。
改为:`return self.down_proj.weight + (1 - self.ttt_nlms_decay) * (current_w - self.down_proj.weight) + dw`。
**关键**:推理用 `current_w = W0 + ΔW`,decay 只作用在 ΔW(= current_w − W0)上,不衰减 W0 本身。这是两侧唯一不对称的写法,但语义与训练侧对 S 的 `(1−α)·S + dW` 完全一致。
注意:outer 分支返回不变(`current_w + dw`),decay 仅作用于 nlms。

### 改动 3 — config flag(两侧同名声明)
文件:`hf_models/hf_qwen3/configuration_qwen3.py` + `inference_model/hf_qwen3/configuration_qwen3.py`。
新增构造参数与赋值:
```python
ttt_nlms_decay = 0.0          # 默认 0.0 = 纯 NLMS（向后兼容，bit 级不变）
self.ttt_nlms_decay = float(ttt_nlms_decay)
if not (0.0 <= self.ttt_nlms_decay < 1.0):
    raise ValueError("ttt_nlms_decay must be in [0.0, 1.0)")
```
两侧 MLP `__init__` 读 flag:`self.ttt_nlms_decay = float(getattr(config, "ttt_nlms_decay", 0.0))`。

### 改动 4 — matched CPT wrapper 透传
文件:`scripts/train/longsft/qwen3-1.7b/run_stage3_nlms_matched.sh`。
nlms 分支的 `MODEL_FOUNDATION_JSON` 加 `"ttt_nlms_decay": ${NLMS_DECAY_VALUE}`,env `NLMS_DECAY="${NLMS_DECAY:-0.1}"`。

### 改动 5 — ttt_train_only 白名单
**无需改**。decay 是标量超参,不引入新可训练参数,保持 NLMS 零新增参数优点。

## 测试计划(TDD)

新增/扩展到 `tests/test_ttt_nlms_train.py` + `tests/test_ttt_nlms_write.py`:

1. **α=0 退化**:`ttt_nlms_decay=0.0` 时训练/推理 forward 与当前纯 NLMS 逐位相同(向后兼容守门)。
2. **config 校验**:`decay=1.0` 和 `decay<0` 抛 `ValueError`;默认 `0.0`;两侧 config 同名声明(防 HF 静默丢弃)。
3. **有界性**:构造 16 chunk + 放大尺度输入,`α=0.1` 下 `‖S‖` 收敛(末 4 个 chunk 的 `‖S‖` 相邻增幅 < 5%,即接近稳态平台),`α=0` 下 `‖S‖` 末 chunk > 首 chunk 的 5 倍(持续增长)→ 对比证明 decay 限幅生效。
4. **训推一致**:`α=0.1` 下训练侧 chunk-loop 与推理侧逐 chunk 数值一致(扩展现有 `test_train_nlms_matches_inference_nlms`)。
5. **梯度非消失**:`α=0.1` + 冻结 backbone,ttt_proj/ttt_conv 梯度非零有限(对比 detach 的梯度消失)。

## 验证路径(直接真冒烟 sweep)

- 2-GPU,`MAX_STEPS=2`,`SAVE_STEPS=0`,sweep (η, α) 小网格:η∈{1, 5}、α∈{0.05, 0.1, 0.3}。
- 三判据:
  1. `ttt_dw` 有界(非天文数字,对比 η=5.0/decay=0 的 2.58e36)。
  2. `ttt_do` 跳出 bf16 噪声地板(>1.66e-3,理想落 [0.2,0.4])。
  3. loss 不发散(对比 η=5.0/decay=0 的 5.27→7.07↑)。
- 选出"有界 + 可学"的 (η, α),交后续完整 matched CPT(阶段 B 三臂)。

## 护栏

- outer 路径 bit 级不变(decay 仅在 nlms 分支)。
- 短上下文 4k/8k 不退化(decay 在少 chunk 下影响小)。
- α=0 全链路退化为现状(所有现有 NLMS/outer 测试不破)。

## 单元边界

- **写规则更新**:训练/推理各一处 S 累加点,纯函数式改动,可独立测有界性与一致性。
- **decay 限幅**:单一标量 α 作用于单一累加表达式,职责单一,无副作用。
- **验证**:真冒烟 read-only 观测 ttt_dw/ttt_do/loss,不改模型结构。

## 风险

- α 与 η 的联合标定:真冒烟 sweep 若 (η,α) 网格无任一组合同时"有界+可学",说明 decay 不足以补偿 off-distribution → 需考虑严格逐-token 序贯 / DeltaNet chunkwise form(下一轮设计)。
- decay 改变 NLMS 递推语义(S 不再纯残差累加,偏向 gated/decay 形态,近 DeltaNet/Mamba):与"NLMS 消除 cross-talk"motivation 仍一致(归一化+衰减都抑制串扰),但论文叙述需明确这是 "decayed NLMS" 而非纯 NLMS。
- 长上下文(32k=32 chunk)下 decay 的有界性更强,但也可能过度遗忘早期 key → matched CPT 后需检查长程检索是否受损。
