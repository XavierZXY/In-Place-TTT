# Review Summary

**Problem**: Qwen3-1.7B In-Place TTT 长上下文 RULER 检索质量不足(0.594 vs base 0.852)
**Initial Approach**: 残差最小二乘写规则(idea ①)+ 容量/碰撞失败预测诊断(idea ⑫)
**Date**: 2026-06-29
**Rounds**: 2 / 5
**Final Score**: 7.55 / 10
**Final Verdict**: REVISE(已整合全部 P0,定稿;未强行刷到 9)

## Problem Anchor
见 FINAL_PROPOSAL.md(全程未漂移)。

## Round-by-Round Resolution Log
| Round | Main Concerns | Simplified/Modernized | Solved? | Remaining Risk |
|-------|---------------|------------------------|---------|----------------|
| 1 | "RLS"表述过强;chunk 内更新语义矛盾;offline 重放不 faithful 于端到端;claim 过强 | 改名 NLMS;锁定 block residual write;验证拆 faithful/offline;claim 弱化 | yes | train/infer mismatch 尚未处理 |
| 2 | chunk 内 dilution;outer-train/NLMS-infer 分布错配;novelty 仍可能被评增量 | write granularity 旋钮;go-no-go 改为 matched CPT 判定;论文主张改为"容量定律+最小 intervention" | partial | venue novelty 仍偏早期(6.0) |

## Overall Evolution
- 方法从模糊"残差写规则"收敛为精确 **block NLMS write on MLP down_proj fast-weight**。
- 最大认知修正: zero-training 负结果**不能**否定方法(checkpoint 是 outer 训练的),真正判定靠 matched CPT。
- 复杂度旋钮收敛为唯一一个: write granularity。

## Final Status
- Anchor: preserved
- Focus: tight(单一主方法 + 一个 supporting 诊断)
- Modernity: appropriately frontier-aware(delta-rule 用对,strict RLS 正确降级)
- 最强部分: 最小机制 + 零训练可诊断 + 容量定律可证伪
- 剩余弱点: venue novelty 偏早期(delta-rule prior 近),需第一批 matched-CPT 结果支撑
