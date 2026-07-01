# Research Proposal: Residual-Write In-Place TTT (NLMS) — 用块残差写规则减少 MLP fast-weight 记忆的 key 覆盖,并建立可预测 RULER 失败的容量定律

> 状态: research-refine 2 轮后 7.55/10 (REVISE→已整合全部 P0 修复定稿)。可进 /experiment-plan。

## Problem Anchor (immutable)
- **Bottom-line**: Qwen3-1.7B In-Place TTT 长上下文 RULER 检索质量不足(16k overall 0.594 vs base 上界 0.852;disable-ttt 0.173,说明 TTT 在贡献检索但质量不够)。
- **Bottleneck**: write rule `S += ttt_lr·KᵀV` 无界外积累加 → 多 chunk 关联线性叠加进同一 fast-weight,cross-talk / score dilution,检索熵近均匀。
- **Non-goals**: 不改 base MLP 预训练;不改 SWA/full 布局;非生成/repetition(已证伪);非 attention query update(qTTT 已占)。
- **Constraints**: ≤8 GPU;优先零/少训练;复用探针 `eval/diagnostics/ttt_signal_probe.py`;训练(`hf_models`)/推理(`inference_model`)数学一致。
- **Success**: (a) block residual write 在 RULER 检索子任务可测改善或干净证伪;(b) 容量指标预测 per-sample 失败 AUC 显著>0.5;(c) 短上下文 4k/8k 不退化。

## Method Thesis
把 In-Place TTT 的 chunk 写规则从纯外积 `S += η·KᵀV` 改成 **block residual write (NLMS)**:
```
R_c = V_c − K_c·Sᵀ                      # chunk-start S 下,当前记忆对本 chunk key 的预测误差
S  ← S + η·(R_cᵀ·K_c)/(λ + tr(K_cᵀK_c)) # 只写残差
O_t = h_t@W₀ᵀ + h_t@Sᵀ                   # base/delta 路径分离不变
```
每个 chunk 只写"当前 fast-weight 尚未解释的残差",在**有限容量关联记忆下减少 key 方向的覆盖/稀释**。最小机制:替换一处 chunk 更新;无新可训练模块;key/value/布局不变。

## Contribution Focus
- **Dominant (论文主张,按 R2-P0 调整)**: 不是"发明新 update rule",而是 **"预训练 Transformer MLP fast-weight 记忆的容量失败定律 + 最小 residual intervention"**——诊断 In-Place TTT 为何在长上下文检索崩,并用最小写规则改动验证/修复。
- **Supporting**: in-place-TTT 专属容量定律(write collision / effective rank / read entropy)预测 RULER 失败 + 当写规则筛选器。
- **Non-contributions**: 不声称新 feature map / gate / attention 改动 / 发明 delta-rule(归功 DeltaNet/Gated DeltaNet/MIRAS)。

## Complexity Budget
- Frozen: backbone, ttt_conv, ttt_proj, key=h, value=v̂, SWA/full 布局。
- New trainable: 0(诊断阶段)。仅 2 标量 η, λ + 1 个 granularity 旋钮 write_subchunk。
- Excluded(默认不开): decay, keynorm, 独立检索模块, token gate, diagonal-RLS(仅 scalar NLMS 正信号后才考虑), low-rank approx。

## Core Mechanism
- 每 chunk c: K_c∈R^{C×d_h}, V_c∈R^{C×d}, S∈R^{d×d_h}(=ΣΔW 当前值)。
- 残差 `R_c = V_c − K_c·Sᵀ`(整块一次算,用 chunk-start S)。
- 更新 `S ← S + η·(R_cᵀ·K_c)/(λ + tr(K_cᵀK_c))`。
- **write granularity 旋钮(R2-A)**: write_subchunk∈{1024,512,256,128} 把 chunk 切成子块串行写,缓解 chunk 内 token 共享 chunk-start S 导致的 intra-chunk dilution。**先押 256**。256≫1024 ⇒ cross-talk 也在 chunk 内;1024 已有效 ⇒ 用 1024 主结果更优雅。这是同一 NLMS 的 scan granularity,非方法漂移。
- **一致性**: chunk(子块)间串行递推,训练/推理共享同一递推,S 用 FP32 accumulator,读出 BF16。**不再是 cumsum**——相对现状实现的主要工程改动点。

## Supporting (capacity law, 3 指标)
- write collision: 新(子)chunk key 在已写 key 子空间投影占比。
- effective rank: 累积 key 矩阵谱熵。
- read entropy: 门控行 softmax 熵(复用 Probe B)。
- (condition number 入 appendix,数值不稳。)
- 用途: per-sample 预测 correct/incorrect(AUC + length scaling);先证明预测力,**暂不做 selector**(防膨胀成第二条主线)。

## Validation (faithful / offline 分离 + go-no-go 逻辑修正)
**Claim 1 (dominant): block residual write 减少 key 覆盖、提升长上下文检索**
- **1a Faithful no-training forward(唯一报 RULER)**: `inference_model` 加 `ttt_write_rule={outer,nlms,keynorm}` + `write_subchunk` flag,**整模型重跑 forward**,每层用当前激活在线算 k/v/S/readout。RULER 16k 检索子集(NIAH multikey_2/3, multiquery, vt)。
  - baselines: outer(现状)/ NLMS-1024 / NLMS-256 / keynorm(负对照)。
  - 护栏: 4k/8k 不退化;报 η=0/disable-delta guardrail 确认收益非数值缩放副作用。
  - **impl sanity 前置**: 先证明新 recurrence path 能复现 outer rule,数值≈原 checkpoint,再开 NLMS。
- **1b Offline counterfactual probe(只报诊断,不报 RULER)**: `ttt_signal_probe.py` 离线重放 current vs NLMS 的 S,报 read entropy/collision/local recon error/counterfactual delta。

**Claim 2 (supporting): 容量指标预测失败**
- offline instrument 现状 run,3 指标预测 per-sample 对错,AUC + 长度相关。expected: ≥1 指标(collision/eff-rank)AUC>0.6,随长度恶化吻合掉点。

## go/no-go 逻辑(R2-P0 关键修正)
- **zero-training(1a)只是 drop-in 兼容性测试**: checkpoint 的 ttt_proj/ttt_conv/value 在 outer-product 动力学下训练,换 NLMS 是 off-distribution。**正结果强,负结果不致命**。
- **真正判定 = Phase 2 matched CPT**: `outer+CPT` vs `NLMS+CPT`,冻 backbone,同 tokens/optimizer/数据/预算。这才是 architecture potential comparison。
- zero-training 负但 capacity 指标强相关 → 不立即放弃,先判断是否 train/infer mismatch。

## Failure Modes
- faithful forward 实现复杂(改 inference 递推)→ impl sanity 前置。
- chunk-serial recurrence 影响吞吐/缓存 → 先小样本/少层验证。
- 短上下文退化 → 护栏 4k/8k。
- 容量指标无关 → 有价值 negative result,转 value/objective 方向(idea ⑦ Causal-Effect Value Targets)。

## Novelty
closest: Gated DeltaNet / DeltaNet(delta-rule on linear-attn state)、MIRAS。delta: residual write 落到 **pretrained Transformer MLP down_proj fast-weight**(In-Place TTT 独有接口)+ 零训练可注入诊断 + in-place-TTT 专属容量定律。非 gate/feature-map/attention。

## Compute & Timeline
- 1a faithful forward: RULER 子集 × {outer,NLMS-1024,NLMS-256,keynorm} × 几长度,<8 GPU-h。
- 1b/Claim2 offline: 探针几十样本,<2 GPU-h。
- Phase 2(条件触发)matched CPT: outer-CPT vs NLMS-CPT,~2 GPU-day。
- Go/no-go 2-3 天。

## Experiment Handoff Inputs
- Must-prove: block residual write(matched CPT 下)> outer;容量指标预测失败。
- Must-run ablations: write_subchunk sweep;η/λ(dev only);短上下文护栏;keynorm 负对照;η=0 guardrail;逐层 vs 全层。
- Critical: RULER 16k 检索子集 + 4k/8k 护栏;trim score / read entropy / collision / AUC。
- Highest-risk: (1) cross-talk 是主因;(2) train/infer write-rule mismatch 掩盖真实潜力(故需 matched CPT)。
