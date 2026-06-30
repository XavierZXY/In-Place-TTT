# 设计:ttt_lr warmup(NLMS 冷启动稳定化)

**日期**: 2026-06-30
**来源**: M3 路线选项 2 — stage3 开端用 NLMS 重训,需冷启动稳定化
**前置依据**: refine-logs/EXPERIMENT_RESULTS_2026-06-29.md(D/A/C 证伪 + 路线决策)

## 背景与问题

选项 2(stage3 开端用 NLMS 重训)让 ttt_conv/ttt_proj 从出生就在 NLMS 动力学下学习,使残差自然收缩。但**冷启动鸡生蛋**:训练最初几十步 projection 还没学会让 `K·Sᵀ` 预测 V,残差不收缩 → S 发散(= C sweep 的 ttt_dw=2.58e36 状态),梯度爆炸会在 projection 学到东西前把它打挂。

**对策**:ttt_lr warmup。`ttt_lr` 从极小值线性爬到目标值,冷启动写入幅度 `dW ∝ ttt_lr` 被压到不发散,撑到 projection 适应后再加大。当前 `ttt_lr` 是 MLP `__init__` 读的静态标量(modeling:98),forward 直接乘(:268/:292/:309),无任何调度机制——需新建。

## 设计决策(brainstorming 已确认)
- **曲线形状** = 线性 warmup(与 backbone lr warmup 同构,最简可预测)。
- **注入机制** = 训练循环每步 setter(复用现有 `_iter_decoder_layers` 遍历 TTT 层设 `mlp.ttt_lr_effective`),不动 forward 签名,FSDP 友好。
- **参数化** = 绝对步数 + 起始值(`ttt_lr_warmup_steps` / `ttt_lr_warmup_init`,目标值复用现有 `ttt_lr`)。跨不同 max_steps 不需重算。
- **resume** = 纯 global_step 函数(无额外状态),resume 后 global_step 从 ckpt 恢复,warmup 自动接对位置。

## 架构与数据流

无状态,纯 global_step 函数:
```
训练循环每步(train_torch.py, global_step 递增后、model() 之前):
  eff_lr = ttt_lr_warmup_factor(global_step, warmup_steps, init, target)
  set_ttt_lr_effective(model, eff_lr)        # 遍历 TTT 层注入
       ↓ 每个 TTT MLP.ttt_lr_effective = eff_lr
  forward 用 self.ttt_lr_effective 乘 dW(而非 self.ttt_lr)
```

线性 warmup 函数:
```
factor(step) = init + (target - init) * min(step / warmup_steps, 1.0)   # warmup_steps>0
factor(step) = target                                                    # warmup_steps<=0
```

三个职责单一组件:
1. `ttt_lr_warmup_factor`(纯函数,独立单测,不碰模型)
2. `set_ttt_lr_effective(model, lr)`(遍历注入,复用 `_iter_decoder_layers`)
3. forward 读 `ttt_lr_effective`(默认 = `ttt_lr`,无 warmup 时 bit 级不变)

**关键不变量**:
- warmup 只作用训练;eval/推理用目标 `ttt_lr`(`ttt_lr_effective` 默认 fallback 到 `ttt_lr`)。
- `warmup_steps=0` 时 `ttt_lr_effective ≡ ttt_lr`,全链路 bit 级不变(向后兼容)。
- resume 后 global_step 恢复,warmup_fn 自动接对位置,无额外状态。

## 代码改动点

### 改动 1 — config 声明(训练侧 hf_models/.../configuration_qwen3.py)
新增构造参数 + 校验:
```python
ttt_lr_warmup_steps=0,        # 0 = 无 warmup（向后兼容）
ttt_lr_warmup_init=0.0,       # warmup 起始值
self.ttt_lr_warmup_steps = int(ttt_lr_warmup_steps)
if self.ttt_lr_warmup_steps < 0:
    raise ValueError("ttt_lr_warmup_steps must be >= 0")
self.ttt_lr_warmup_init = float(ttt_lr_warmup_init)
if self.ttt_lr_warmup_init < 0:
    raise ValueError("ttt_lr_warmup_init must be >= 0")
```
推理侧 config 无需改(warmup 是训练期概念)。

### 改动 2 — MLP 初始化 ttt_lr_effective(modeling_qwen3.py:98)
```python
self.ttt_lr = getattr(config, "ttt_lr", 0.3)
self.ttt_lr_effective = self.ttt_lr   # 默认=目标值；训练循环每步覆写
```

### 改动 3 — forward 三个乘点改用 ttt_lr_effective(:268 NLMS、:292/:309 outer/key_norm)
```python
... * self.ttt_lr_effective   # 而非 self.ttt_lr
```
默认 `ttt_lr_effective == ttt_lr`，无 warmup 时 bit 级不变。

### 改动 4 — warmup 函数 + setter(in_place_ttt/ttt_aux/training.py，复用 _iter_decoder_layers)
```python
def ttt_lr_warmup_factor(step, warmup_steps, init, target):
    if warmup_steps <= 0:
        return target
    frac = min(max(step, 0) / warmup_steps, 1.0)
    return init + (target - init) * frac

def set_ttt_lr_effective(model, lr):
    for layer in _iter_decoder_layers(model):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "ttt_lr_effective"):
            mlp.ttt_lr_effective = float(lr)
```

### 改动 5 — 训练循环每步调用(tasks/train_torch.py，global_step+=1 后、model() 前)
```python
eff_lr = ttt_lr_warmup_factor(global_step, warmup_steps, warmup_init, target_ttt_lr)
set_ttt_lr_effective(model, eff_lr)
```
三者从 `model_config`(train_torch.py:471 `model_config = model.config`,build_foundation_model 后已存在)读:`model_config.ttt_lr`(target)、`model_config.ttt_lr_warmup_steps`、`model_config.ttt_lr_warmup_init`。注:foundation JSON 有 fail-fast 校验(:475 检测 silently dropped keys),故改动 1 的 config 声明是硬前提,新字段未声明会直接报错。

### 改动 6 — wrapper env 透传(run_stage3_nlms_matched.sh)
nlms 分支 MODEL_FOUNDATION_JSON 加 `ttt_lr_warmup_steps`/`ttt_lr_warmup_init`，env `TTT_LR_WARMUP_STEPS`/`TTT_LR_WARMUP_INIT`。

## 测试计划(TDD)

新建 `tests/test_ttt_lr_warmup.py`:
1. **warmup 函数数学**:step=0→init；step>=warmup_steps→target；中点→(init+target)/2；warmup_steps=0→恒 target；单调递增夹紧 [init,target]。
2. **config 声明 + 校验**:默认 warmup_steps=0/init=0.0；负值抛 ValueError。
3. **MLP 默认 ttt_lr_effective == ttt_lr**:无 setter 时 forward bit 级不变。
4. **setter 注入**:set 后每个 TTT MLP.ttt_lr_effective==v；非 TTT 层不受影响。
5. **向后兼容**:warmup_steps=0 时训练侧 NLMS/outer forward 与改动前逐位相同(守 23 既有测试不破)。

## 短验证(全量重训前的廉价闸门)
- 从 stage2 ckpt 起，NLMS 写规则，**4k 长度**、**~300 步**、2-GPU。
- warmup：`ttt_lr_warmup_steps=200`、`ttt_lr_warmup_init=1e-3`、目标 `ttt_lr=3`(或先小目标如 1)。
- **三判据**:① 冷启动前期 ttt_dw 有界(不爆 2.58e36)② warmup 结束后 ttt_do 跳出 bf16 噪声地板(>1.66e-3)③ loss 平稳下降不发散。
- **判定**:三判据满足 → 冷启动方案成立，上全量 64k stage3 matched 双臂；否则记录并回路线讨论(subchunk/共训)。

## 护栏
- warmup_steps=0 全链路 bit 级退化为现状。
- outer 路径不受影响(三乘点对 outer 也用 effective，但 outer 不配 warmup 时 effective=target=ttt_lr)。
- resume：纯 global_step 函数自动接对位置。

## 单元边界
- warmup_fn：纯函数,无副作用,独立测。
- setter：单一职责注入,复用现有遍历器。
- forward：只把乘数从 ttt_lr 换 ttt_lr_effective,语义单点改动。

## 风险
- 短验证若 warmup 单独不够稳(projection 适应慢),备选叠加 C 的中等 decay(代码已留存,默认 0.0)。
- warmup_steps 取值需经验:太短保护不足,太长拖慢学习——短验证用 200 探,全量按比例放大。
