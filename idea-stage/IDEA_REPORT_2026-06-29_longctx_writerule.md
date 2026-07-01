# Research Idea Report — In-Place TTT 长上下文检索 (write-rule / value-target / capacity 方向)

**Direction**: 提升 Qwen3-1.7B In-Place TTT 的长上下文(RULER)检索质量,对症 score dilution
**Generated**: 2026-06-29
**Reviewer**: gpt-5.4 (xhigh) via Codex MCP, thread 019f12e1-3b69-71d0-a56b-b91fefde53b4
**Pipeline**: 12 ideas generated → ranked by novelty×feasibility (pilots not yet run)
**注**: 本文件独立于 2026-06-26 的多模态 TTT 报告(IDEA_REPORT.md),主题不同。

## 关键定位
所有 idea 刻意避开 banlist 的正面战场(feature map φ / gate / qTTT query-update),
聚焦 **write rule / value target / memory capacity** —— 这是 in-place-TTT-in-MLP 结构独有、
attention-based 工作(Hedgehog/qTTT/GDWM/SeerAttention/Mamba2/Titans)无法 claim 的领地。

## 现状基线
- RULER 16k overall: 最强 run 0.594, base 上界 0.852, disable-ttt 0.173
- 真问题: 检索/答案质量, score dilution (needle 被 distractor 淹没), 非生成崩塌
- 当前 TTT write rule: ΔW += k⊗v 无界外积累加, key=h 未归一化, 检索熵近均匀
- 已否决: keynorm(纯归一化,0.04-0.13); 已查新弱新颖: φ feature map, TTT-as-gate

## 12 个 idea(Codex novelty×feasibility 排序)

### Tier 1 — 推荐优先(诊断包 + 可能涨分)
1. **Residual Least-Squares TTT** (新方法+诊断, MED, 6/10)
   ΔW += η·(v - ΔW·k)⊗k / (λ+||k||²),在线残差回归/delta rule 替代纯外积累加。
   假设: cross-talk 是主因,写"未被解释的残差"减少旧事实被背景覆盖。
   MVE: 零训练替换 write rule + 归一化分母,跑 RULER NIAH/VT 子集,<2 GPU-h。
   最近 prior: Gated DeltaNet。差异: 不是 gate/feature map,是 MLP fast-weight 的在线最小二乘写规则。

12. **Capacity Law / Failure Predictor** (理论+诊断, LOW, 6/10)
   建预测 RULER 失败的容量指标: key mutual coherence / effective rank / condition number /
   read entropy / write collision。MVE: 不改模型,只 instrument,按样本预测对错做 AUC + length scaling。
   差异: Hedgehog 诊断 linear-attn feature map; 这里是 in-place MLP fast-weight 的容量/碰撞定律。
   价值: negative result 也能写成有用论文,且能筛选其他 write rule。

2. **Future-Query Supervised TTT Objective** (新 objective, MED, 6.5/10)
   训练时 prefix 写 memory → 给未来 query → 直接优化 delta path 对答案的可恢复性,而非只 NTP。
   假设: NTP-aligned objective ≠ long-context retrieval objective。
   MVE: 合成 KV/RULER, 冻 backbone 只训 ttt_proj/ttt_conv, prefix 写完后 query loss 监督答案 logit。
   差异: qTTT 更新 attention Q; 这里训 MLP fast-weight 写入要服务未来查询。

### Tier 2 — 便宜可快验
4. **Centered/Anti-Background Memory** (诊断+方法, LOW-MED, 5.5/10)
   只写相对背景的协方差 Σ(k-μk)⊗(v-μv) 或维护负背景 memory。零训练 ablation 快验 dilution 来源。

10. **Downstream-Useful Value Subspace** (诊断+方法, MED, 6.5/10)
   限制 TTT delta 输出只注入下游 full-attn 真正读取的 hidden subspace。比 key-side φ 更少被做。

3. **Collision-Aware Write Rule** (新方法+诊断, MED, 5.5/10)
   写前估计新 key 与已有 key 子空间干涉,按 orthogonal residual / Gram leverage 调整写入。

### Tier 3 — 高新颖高风险
7. **Causal-Effect Value Targets** (新 objective, HIGH, 7.5/10)
   memory 存"历史 span 对未来答案的因果影响向量"(counterfactual deletion delta),非 raw activation。
5. **Error-Correcting Multi-Bank Memory** (新方法, HIGH, 7/10) — CountSketch 式冗余写入纠碰撞。
11. **Self-Indexing Latent Sentinels** (新方法, HIGH, 7/10) — chunk 写虚拟 index entry 做层内目录。
(dict). **Streaming Low-Rank Dictionary TTT** (新方法+理论, MED, 6.5/10) — online SVD/FD 维护 rank-r 字典。

### Tier 4 — 其他
8. **Layer Role Specialization** (MED-HIGH, 6/10) — 不同 TTT 层分工(身份/关系/答案残差)。
9. **Deterministic Multi-Timescale ΔW** (LOW-MED, 5.5/10) — recent/mid/far 多时间尺度 bank。

## Codex 的下注
先做 **1 + 12 + 2 的两周诊断包**(残差写规则 + 容量定律 + 未来查询目标);
若 collision/capacity 指标能解释 0.59 的失败,再把 **7 或 Causal-value** 做成真正的 paper contribution。

## CC 补充判断
- idea 1 和 12 共享同一套 instrumentation(都要测 memory 碰撞/容量),打包做边际成本低。
- idea 1 是唯一"零训练就能拿 go/no-go"的,且直接改 write rule —— ROI 最高,先做。
- 与已纠正记忆一致: 真问题是检索质量,这批 idea 全部对症,不再走 repetition/decay 弯路。

## Next Steps
- [ ] 选定 1-3 个 idea 进 pilot (idea 1 零训练优先)
- [ ] 选定后 → /research-refine 细化 → brainstorming 出 spec → 实现
