# Experiment Plan — Residual-Write In-Place TTT (NLMS)

**Problem**: Qwen3-1.7B In-Place TTT 长上下文 RULER 检索质量不足(16k overall 0.594 vs base 0.852;disable-ttt 0.173)
**Method Thesis**: 把 chunk 写规则从外积累加 `S += η·KᵀV` 改成 block residual write `R=V−KSᵀ; S←S+η·RᵀK/(λ+tr(KᵀK))`,只写未被解释的残差,减少有限容量关联记忆下的 key 覆盖/score dilution
**Date**: 2026-06-29
**来源**: refine-logs/FINAL_PROPOSAL.md(research-refine 2 轮 7.55/10)

## Claim Map
| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|-----------------|-----------------------------|---------------|
| **C1 (dominant)** block residual write 在**同预算训练下**优于外积写,提升长上下文检索 | 直接证明 write-rule 是 bottleneck,且改动有效 | matched CPT(outer-CPT vs NLMS-CPT)在 RULER 检索子集显著提升,短上下文不退化 | B1, B2, B4 |
| **C2 (supporting)** 容量/碰撞指标可预测 RULER 失败 | 把"cross-talk 是病因"从假说变可测,且当写规则筛选器 | ≥1 指标 per-sample AUC>0.6 且随长度恶化吻合掉点曲线 | B3 |
| **Anti-claim 要排除** 收益只来自数值缩放/抑制 TTT,而非 residual 结构 | 防 keynorm 式假阳/假阴 | η=0 与 disable-delta guardrail;keynorm 负对照不应复现 NLMS 的收益 | B1, B2 |

## Paper Storyline
- **Main paper 必须证明**: C1(matched CPT 下 NLMS>outer)+ C2(容量定律预测失败)+ anti-claim 排除。
- **Appendix 支持**: write granularity 完整 sweep、diagonal-RLS variant、逐层启用消融、condition number 指标。
- **Intentionally cut**: 多 backbone 泛化、其他长上下文 benchmark(LongBench 等)——非本文核心 claim,留作 future work。

## Experiment Blocks

### Block 0: 实现 sanity(write-rule 基础设施) — MUST-RUN
- **Claim tested**: 无(基础设施正确性,是一切结论的前提)。
- **Why**: R2-P0 要求——新 recurrence path 必须能复现 outer rule,否则 NLMS vs outer 比较不可信。
- **改动点**: 
  - 推理侧 `inference_model/hf_qwen3`: config 加 `ttt_write_rule∈{outer,nlms}` + `ttt_write_subchunk`(声明在 configuration_qwen3.py 避免静默丢弃);`modeling_qwen3.py:56/181` 的 `current_w = current_w + dw` 处分支。
  - 训练侧 `hf_models/hf_qwen3`: 对应 `modeling_qwen3.py:244` 的 cumsum 处加 NLMS 串行递推分支。
- **Setup**: ttt_write_rule=outer 时数值必须与现状 checkpoint 推理结果 bit/近似一致。
- **Success criterion**: outer 路径复现现状 RULER 16k≈0.594(±噪声);NLMS 路径在 η=0 时退化为 base(disable-delta)。
- **Failure interpretation**: 不一致 → 递推/精度/顺序实现有 bug,先修再继续。
- **Target**: 无(内部 gate);写进 appendix 实现细节。
- **Priority**: MUST-RUN(阻塞全部后续)。

### Block 1: Main anchor — matched CPT 下 NLMS vs outer(报 RULER) — MUST-RUN
- **Claim tested**: C1 + anti-claim。
- **Why**: 唯一能 faithful 报 RULER 的判定实验(R2-P0:zero-training 是 off-distribution,负结果不致命;matched CPT 才是判定)。
- **Dataset/split/task**: RULER 16k 检索子集 NIAH multikey_2/3 + multiquery + vt(score dilution 最重);护栏 4k/8k。n_per_task=50。
- **Compared systems**(≤3 family):
  1. outer + CPT(强基线,= 现状路线重训)
  2. NLMS + CPT(write_subchunk=256,先验最优)
  3. keynorm + CPT(负对照,已知失败,不应复现 NLMS 收益)
- **Metrics**: trim 后 RULER score(决定性);read entropy / write collision(辅助机理);4k/8k 护栏分。
- **Setup**: 冻 backbone(ttt_train_only),只训 ttt_conv/ttt_proj;**同 tokens/optimizer/数据/步数/lr**(沿用 stage3 yaml:gbs32, lr5e-5, constant);从同一 stage2 ckpt 起。DEFAULT_SEEDS=3(若预算紧 → 2)。
- **Success criterion**: NLMS-CPT 相对 outer-CPT,multikey/multiquery 子集平均 ≥+3pt,且 4k/8k 不退化(±1pt 内);keynorm 不应达到 NLMS 水平。
- **Failure interpretation**: NLMS≈outer → write rule 不是 bottleneck,转 value/objective 方向(idea ⑦);NLMS<outer → residual 擦掉了有用信号,需 diagonal-RLS 或重标 η。
- **Target**: 主表 Table 1。
- **Priority**: MUST-RUN。

### Block 2: drop-in zero-training 兼容性测试(faithful forward) — MUST-RUN(廉价前置)
- **Claim tested**: C1 的廉价预判(非判定)。
- **Why**: <8 GPU-h 拿早期信号决定是否值得 Block1 的 CPT 投入;但因 off-distribution,只作 go/no-go 参考。
- **Dataset**: 同 Block1 子集 + 长度。
- **Compared systems**: outer(现状 ckpt)/ NLMS-1024 / NLMS-256 / keynorm 负对照(全部在现状 ckpt 上换 write rule,不训练)。
- **Metrics**: trim score + read entropy;η=0 guardrail。
- **Setup**: Block0 的 faithful forward flag,整模型重跑,不训练。
- **Success criterion**: NLMS 任一 granularity 相对 outer 不显著掉(理想小涨)→ 绿灯进 Block1;大幅掉 → 先查 train/infer mismatch,不立即放弃。
- **Failure interpretation**: 见 success;关键是 negative 不否定方法(R2-B)。
- **Target**: appendix(drop-in 兼容性)。
- **Priority**: MUST-RUN(Block1 前置闸门)。

### Block 3: 容量定律预测 RULER 失败(offline probe) — MUST-RUN
- **Claim tested**: C2。
- **Why**: 独立支撑"cross-talk 是病因",且不依赖任何训练。
- **Dataset**: 现状 ckpt 在 RULER 4k/16k/32k 全任务,per-sample。
- **Compared systems**: 3 指标(write collision / effective rank / read entropy)单独 + 组合逻辑回归。
- **Metrics**: 预测 correct/incorrect 的 AUC;指标随长度 vs 掉点曲线相关。
- **Setup**: `eval/diagnostics/ttt_signal_probe.py` 扩展,只 offline,不报 RULER(只报这些诊断指标 + 与已有 eval 的 per-sample 对错对齐)。
- **Success criterion**: ≥1 指标 AUC>0.6,组合>0.7,且长度相关显著。
- **Failure interpretation**: 全部 AUC≈0.5 → cross-talk 非主因,削弱 C1 动机,转向 value/objective。
- **Target**: 主表/图 Figure 2(容量定律)。
- **Priority**: MUST-RUN。

### Block 4: write granularity sweep + diagonal-RLS(消融) — NICE-TO-HAVE
- **Claim tested**: C1 的机理细化(intra-chunk dilution 是否存在)+ modernization。
- **Why**: R2-A——256≫1024 说明 cross-talk 也在 chunk 内;diagonal-RLS 仅在 scalar NLMS 正信号后做。
- **Compared systems**: NLMS write_subchunk∈{1024,512,256,128}(CPT 或 drop-in);+ diagonal-RLS variant。
- **Metrics**: RULER 子集 score vs granularity 曲线。
- **Setup**: 同 Block1,只变 granularity。
- **Success criterion**: 找到 score-vs-granularity 拐点;diagonal-RLS 若再涨则入正文,否则 appendix。
- **Target**: appendix Table。
- **Priority**: NICE-TO-HAVE(Block1 正信号后才做)。

## Run Order and Milestones
| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0 sanity | Block0 基础设施 | outer 复现 + η=0 退化 | outer≈0.594 & η=0≈disable-ttt 才继续 | <2 GPU-h | 递推实现 bug(中) |
| M1 drop-in | Block2 廉价预判 | outer/NLMS-1024/256/keynorm faithful forward | NLMS 不大幅掉 → 进 M2 | <8 GPU-h | off-distribution 掉分(低危,不否定) |
| M2 capacity | Block3 容量定律 | offline probe 3 长度 | ≥1 AUC>0.6 → C2 成立 | <2 GPU-h | 指标无关(中,则转向) |
| M3 main | Block1 matched CPT | outer-CPT / NLMS-256-CPT / keynorm-CPT × seeds | NLMS≥outer+3pt & 护栏不退 → C1 成立 | ~2-3 GPU-day | NLMS≈outer(核心风险) |
| M4 polish | Block4 granularity + diagonal-RLS | subchunk sweep | 拐点/增益决定正文 vs appendix | ~1-2 GPU-day | 边际收益小 |

## Compute and Data Budget
- 总估算: M0-M2 <12 GPU-h;M3 ~2-3 GPU-day×seeds;M4 ~1-2 GPU-day。must-run(M0-M3)主导。
- 数据: 复用 ultrafineweb-ruler-v2-sft CPT 数据 + 现成 RULER eval 数据,无新标注。
- 最大瓶颈: M3 matched CPT 的多 seed 训练。

## Risks and Mitigations
- **核心风险 NLMS≈outer(write rule 非 bottleneck)**: M2 容量定律先独立验证病因;若 M2 强而 M3 弱 → train/infer mismatch 或 η 未标定,先 sweep η/granularity 再判。
- **train/infer mismatch 掩盖潜力**: 故 M1 只作参考,M3 matched CPT 判定。
- **递推实现破坏吞吐/数值**: M0 强制 outer 复现 + FP32 S accumulator。
- **容量指标数值不稳(condition number)**: 入 appendix,主用 collision/eff-rank/entropy。

## Final Checklist
- [x] 主表覆盖(Table1 = Block1 matched CPT)
- [x] novelty 隔离(keynorm 负对照 + anti-claim η=0 guardrail)
- [x] simplicity 辩护(write granularity 是唯一旋钮;diagonal-RLS 仅条件触发)
- [x] frontier 必要性(delta-rule 用对,strict RLS 降级为可选 variant)
- [x] must-run(M0-M3)与 nice-to-have(M4)分离
