# Refinement Report

**Problem**: Qwen3-1.7B In-Place TTT 长上下文 RULER 检索质量不足(0.594 vs base 0.852)
**Initial Approach**: 残差最小二乘写规则(①)+ 容量/碰撞失败预测诊断(⑫)
**Date**: 2026-06-29
**Rounds**: 2 / 5
**Final Score**: 7.55 / 10
**Final Verdict**: REVISE(整合全部 P0 定稿)

## Output Files
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Final proposal: `refine-logs/FINAL_PROPOSAL.md`
- Score history: `refine-logs/score-history.md`

## Score Evolution
| Round | PF | MS | CQ | FL | Feas | VF | VR | Overall | Verdict |
|-------|----|----|----|----|------|----|----|---------|---------|
| 1 | 8 | 6 | 7 | 7 | 6 | 6 | 5 | 6.65 | REVISE |
| 2 | 8.5 | 8.0 | 7.0 | 7.5 | 7.0 | 8.0 | 6.0 | 7.55 | REVISE |

## Final Proposal Snapshot
- block residual write (NLMS) on MLP down_proj fast-weight: `R=V−KSᵀ; S←S+η·RᵀK/(λ+tr(KᵀK))`
- 论文主张 = 预训练 MLP fast-weight 的容量失败定律 + 最小 residual intervention(非"发明 update rule")
- write granularity 是唯一复杂度旋钮(sweep 1024/512/256/128,先押 256)
- zero-training = drop-in 兼容性测试(负结果不致命);真正判定 = matched CPT(outer-CPT vs NLMS-CPT)
- supporting: 3 指标容量定律(collision/eff-rank/read-entropy)预测 RULER 失败

## Method Evolution Highlights
1. 验证拆成 faithful forward(报 RULER)+ offline probe(只报诊断)——杜绝"offline 重放声称 RULER 提升"的漂移。
2. go-no-go 逻辑修正: zero-training 负不能否定方法 → matched CPT 才是判定。
3. write granularity 旋钮治 intra-chunk dilution。

## Pushback / Drift Log
| Round | Reviewer Said | Author Response | Outcome |
|-------|---------------|-----------------|---------|
| 1 | offline 重放可报 RULER? | 我主动抛出 cross-layer feedback 顾虑,reviewer 确认不 faithful | 接受,拆 faithful/offline |
| 2 | chunk 内是否仍 dilution | 接受,加 write granularity 旋钮 | 接受 |
| 2 | train/infer mismatch | 接受,go-no-go 改 matched CPT | 接受 |

## Remaining Weaknesses
- Venue novelty 偏早期(6.0): delta-rule prior(DeltaNet/Gated DeltaNet)近,需第一批 matched-CPT 实测结果 + 容量定律证据支撑才像 paper。
- 未刷到 9: 因 venue readiness 本质要靠实验结果,不靠纸面;继续刷分会变成过度包装。

## Next Steps
- READY-enough to implement: 建议 → /experiment-plan 出详细 roadmap,或直接 brainstorming→实现 faithful inference flag。
- 先做 impl sanity(新 path 复现 outer)→ NLMS-1024/256 drop-in → 若正则 matched CPT。

## Raw Reviewer Responses
见 thread 019f12e1-3b69-71d0-a56b-b91fefde53b4(round-1-review.md / round-2-review.md 已摘要)。
