# Round 2 Review (gpt-5.4 xhigh)

Overall: **7.55/10, REVISE**(锚点保住,方法明显更聚焦)。最大剩余风险: chunk 内 dilution + outer-train/NLMS-infer 分布错配。

## Scores
| Dim | Score |
|---|---|
| Problem Fidelity | 8.5 |
| Method Specificity | 8.0 |
| Contribution Quality | 7.0 |
| Frontier Leverage | 7.5 |
| Feasibility | 7.0 |
| Validation Focus | 8.0 |
| Venue Readiness | 6.0 |

## P0
1. **Venue/novelty**: 论文主张从"新 update rule"改为"**pretrained MLP fast-weight 的容量失败定律 + 最小 residual intervention**"(否则被评为"把 DeltaNet 放进 In-Place TTT")。
2. **go/no-go 逻辑修正**: zero-training negative **不足以否定方法**,因 checkpoint 是 outer-product 训练的。Phase 1 只是 drop-in 兼容性测试;**Phase 2 必须 matched CPT: outer-CPT vs NLMS-CPT 同预算**才是真正判定。

## 问题 A 裁决(chunk 内 dilution)
block rule 减少跨 chunk 误差,但 chunk 内 1024 token 共享 chunk-start S → 仍以 batch outer-product superpose,高相干 key 下 intra-chunk dilution 仍在。
→ 把 **write granularity 当唯一复杂度旋钮**: sweep write_subchunk∈{1024,512,256,128},**先押 256**。256 显著好于 1024 ⇒ cross-talk 也在 chunk 内;1024 已有效 ⇒ 方法更优雅。不是漂移,是同一 NLMS 的 scan granularity。

## 问题 B 裁决(train/infer mismatch,致命 subtlety)
faithful forward 是干净的 **inference-time intervention comparison**(可报 RULER),但**不是干净的 architecture potential comparison**——ttt_proj/ttt_conv/value 在 outer-product 下训练,NLMS 改变 memory semantics(读前 subtract S·k 可能把训练时学到依赖 additive accumulation 的有用信号当残差去掉)。
控制:
1. 先 impl sanity: 新 recurrence path 复现 outer rule,数值≈原 checkpoint。
2. zero-training NLMS 只声明 drop-in test(正结果强,负结果不致命)。
3. Phase 2 必须 matched CPT: outer+CPT vs NLMS+CPT,冻 backbone,同 tokens/optimizer/数据。
4. η,λ 只在 dev subset 选,RULER test 不反复调。
5. 报 η=0 / disable-delta guardrail,确认收益非数值缩放副作用。

## Simplification
- Phase 1 只跑 outer / NLMS-1024 / NLMS-256 / keynorm-negctrl。nlms_rls 入 Phase 1b/appendix。
- capacity law 只留 collision / eff-rank / read-entropy,先证明预测 failure,别急着做 selector。

## Modernization
- diagonal preconditioned NLMS 作为可选升级,**仅在 scalar NLMS 有正信号后**做(否则削弱最小机制)。

## Drift: NONE(警惕 capacity law 膨胀成另一条主线,它应服务 NLMS)

## Remaining Action Items
1. impl faithful inference flag + 证明 outer 可复现原结果。
2. 跑 NLMS-1024/256 小子集,先看 NIAH multikey/multiquery/VT。
3. offline probe 只报 local metrics。
4. zero-training 正 → matched CPT。
5. zero-training 负但 capacity 强相关 → 先判断是否 train/infer mismatch,别立即放弃。
