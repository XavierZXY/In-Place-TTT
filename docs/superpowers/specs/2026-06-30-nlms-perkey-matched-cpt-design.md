# 设计:NLMS per-key 归一化修正 + 零训练 η 标定 + matched CPT

**日期**: 2026-06-30
**来源**: 接续 R001/R002/R006 实验(cross-talk 病因确证,r=−0.726),推进 M3
**前置依据**: refine-logs/FINAL_PROPOSAL.md, EXPERIMENT_RESULTS_2026-06-29.md

## 背景与问题
R006 证实 cross-talk 是 RULER 失败主因(write_collision 与 score r=−0.726;outer TTT 在此 ckpt 净害 −0.17)。NLMS 残差写规则旨在消除 cross-talk。但动手前诊断发现当前推理侧 NLMS 实现有**数值缺陷**:

- 当前 `denom = λ + tr(KᵀK)`(整 chunk 1024 个 key 的总能量)≈ **5.2e6**(由 R006 的 token_norm median≈71 估算:1024×71²)。
- 后果:同 η 下 NLMS 写入幅度被压成 outer 的 ~1/5e6,几乎不写 → 退化到 disable-ttt(0.458)。
- 这与 [[ttt-key-normalization-direction-verdict]] 的教训一致(归一化后需重标 η),但正确解法是**修正归一化粒度**,不是盲目放大 η。

## 核心修正:per-key 归一化(标准 NLMS)
```
对 chunk c 内每个 key k_c(用 chunk-start 的 S):
  r_c  = v_c − S·k_c                         # 残差 [d]
  dw_c = η · r_c ⊗ k_c / (λ + ‖k_c‖²)        # per-key 分母 ≈ 5000
  S   ← S + Σ_c dw_c                          # chunk 内批量累加
```
分母回到 ~5000 量级,η 回到 O(1–10),数值稳定。

**实现路线**:chunk 间串行 + chunk 内批量近似——S 跨 chunk 串行更新,chunk 内 1024 个 key 共用同一个 chunk-start S(忽略 chunk 内序贯)。这是 per-key NLMS 的近似版,最务实;若零训练验证有效再考虑严格逐 token 序贯或 DeltaNet chunkwise 并行 form。

## 三处代码改动

### 改动 1 — 推理侧分母修正
文件:`inference_model/hf_qwen3/modeling_qwen3.py` 的 `_write_block`。
当前:`denom = λ + (key*key).sum()`(整 chunk 标量)。
改为 per-key:`denom_c = λ + ‖k_c‖²`(形状 [c,1],每 token 一个),残差 R 逐行除以 denom_c 再外积累加。约 3 行改动。

### 改动 2 — 训练侧 NLMS 分支
文件:`hf_models/hf_qwen3/modeling_qwen3.py` 的 `Qwen3MLP.forward`。
当前 outer 用 `cumsum`(行 244)全并行;NLMS 因 `R=V−S·K` 依赖累积 S 不能 cumsum,改为 chunk 间 Python loop(训练 16k = 16 chunk,16 步,可接受):
```
S = 0
for chunk i in range(chunk_num):
    R_i        = V_i − K_i·Sᵀ              # chunk-start S
    base_out_i = K_i·W0ᵀ                   # base 路径用原始 h(路径分离)
    delta_out_i= K_i·Sᵀ                    # delta 路径读出用 chunk-start S
    out_i      = base_out_i + delta_out_i
    S         += η·(R_i/denom_i)ᵀ·K_i      # 更新
```
**关键:base/delta 路径分离**——W0 永远吃原始 h,只 delta 路径走 NLMS,复用现有 key_norm 分支已验证的双路结构,避免破坏预训练 MLP。
新增 config flag `ttt_write_rule`/`ttt_nlms_lambda`/`ttt_write_subchunk`,**训练侧 config 也要声明**(避免 HF 静默丢弃,见 [[ttt-config-kwargs-silently-dropped]])。
训练/推理必须共享同一递推语义,S 用 FP32 accumulator。

### 改动 3 — ttt_train_only 白名单
**无需改**。NLMS 不引入新可训练参数(η/λ 是标量超参),ttt_conv/ttt_proj 已在白名单(`in_place_ttt/ttt_aux/training.py:180`)。这是 per-key NLMS 的优点:零新增参数,matched CPT 与 outer 完全同参数量,比较干净。

## 实验设计

### 阶段 A — 零训练 η 标定(先做,便宜,<2 GPU-h)
在 `eval/diagnostics/ttt_signal_probe.py` 加离线 NLMS 重算路径:用捕获的 per-chunk (K,V),按 per-key NLMS 递推重算 S,算 output_delta(复用现有 monitor 公式)。
- **sweep η ∈ {1, 3, 10, 30, 100}**(λ=1),长度 4k/16k/32k,看:
  - output_delta 是否落回 **[0.2, 0.4]** 健康写入区
  - write_collision 是否随 NLMS 下降
- **产出**:选 1–2 个让 output_delta 落健康区的 η,交阶段 B。
- **判据**:若无任何 η 能同时压 collision + 保持 output_delta 有效 → 近似版不成立,回退严格序贯/DeltaNet form。
- **护栏**:零训练负结果**不能否定方法**(off-distribution),只用于定 η 范围。

### 阶段 B — matched CPT(真正判定)
三臂同预算:`outer+CPT` / `NLMS+CPT`(阶段 A 选的 η) / `keynorm+CPT`(负对照)。
- 同一 stage2 ckpt 起,冻 backbone(ttt_train_only),只训 ttt_conv/ttt_proj。
- **同 tokens/optimizer/步数/lr**(沿用 stage3 yaml:gbs32, lr5e-5, constant)。
- eval RULER 16k 全任务 + 4k/8k 护栏。
- **判据**:NLMS-CPT 相对 outer-CPT(0.285)显著提升,理想**超过关-TTT 上界 0.458**;keynorm 不应达到 NLMS 水平。
- 成本:3 臂 × ~1 GPU-day(可并行)。

## 验证与护栏
- η=0 必须退化到 disable-ttt(已验证)。
- 短上下文 4k/8k 不退化。
- 训练侧 outer 分支必须 bit 级复现现状(实现 sanity,类比 R001)。
- outer/NLMS 训练侧数值与推理侧一致性检查。

## 测试计划(TDD)
- 训练侧 NLMS forward:η=0 退化为 base、per-key 分母数学正确、outer 分支不变(bit 级)、与推理侧同输入数值一致。
- 推理侧 per-key 分母修正:更新 `tests/test_ttt_nlms_write.py` 的 math 测试为 per-key 分母。

## 单元边界
- 写规则:训练/推理各一个 `_write_block`(NLMS/outer 分支),纯函数,可独立测。
- η 标定:探针离线重算,read-only,不碰模型权重。
- CPT:复用现有 stage3 训练管线,只切 config 的 write_rule。

## 风险
- chunk 内批量近似 vs 严格 per-key 的偏差:阶段 A 验证近似版是否够;不够则升级。
- 训练侧 Python loop 吞吐:16 chunk 可接受,但更长上下文(32k=32步)需评估。
- η 标定区间:阶段 A 若 output_delta 对 η 极敏感,需细化 sweep。
