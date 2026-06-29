# Research Idea Report — TTT / Fast-Weights / 在线关联记忆 在多模态任务上的应用

**Direction**: 把 Test-Time Training（fast-weights / 在线关联记忆）的核心机制迁移到多模态任务（视频生成 / 多视角空间 / 世界模型 / 交错 VLM），重点攻「内环目标（inner-loop target）」这一本质空白。
**Generated**: 2026-06-26
**起点论文**: In-Place TTT（NTP/预测对齐目标 + 复用 W_down + chunk-parallel）、δ-mem（8×8 在线状态 + delta rule + 低秩修正 attention）、HALO（attention→RNN 蒸馏 warm-start）
**本地论文库**: `/zouxiangyu/codes/Learning/In-Place-TTT/papers/related-work/`（14 篇）
**Ideas evaluated**: 9 生成 → 4 保留 + 2 降级 → 3 推荐主线（GPT-5.4 xhigh 评审）

---

## Landscape Summary

当前多模态 TTT 工作（One-Minute Video TTT、Spatial-TTT、CustomTTT、Pathwise Test-Time Correction、世界模型 next-state prediction）几乎**清一色沿用 reconstruction 重构目标**作为 fast-weights 的内环监督。而 In-Place TTT 在语言上已经从理论（induction-head 下 LM-aligned target 抬升正确 next-token logit，重构目标不能）和实验两方面证明：**预测/未来对齐目标本质优于重构目标**。这条结论从未被严格搬到多模态自回归生成。

核心问题因此是一个**第一性问题**：连续视频 token / 多视角帧 / 交错图文流的「NTP analog」到底是什么？是 next-frame？next-latent？innovation（预测残差）？还是 horizon-weighted 预测统计量？没有人回答过，而 In-Place TTT 暗示这个 target 的选择是决定性的、而非调参。

统一理论侧（Test-time Regression、Miras）已把序列层统一为「在线回归求解器」，但这些框架是在 token 空间、完全可观测假设下建立的。多模态是**连续、部分可观测**的，最优内环 target 是否仍是重构、还是 predictive sufficient statistic（≈ Bayes filter 的 innovation），是一个有理论增量的开放问题。

---

## Recommended Ideas（GPT-5.4 残酷筛选后保留，按下注优先级排序）

### ⭐ Idea 1（主赌注）: Innovation-Aligned TTT for Continuous Video
- **Hypothesis**: 把视频 TTT 内环目标从「重建当前 latent」改成「预测未来 latent 的 innovation / 去噪残差」，能让 fast-weights 把容量集中在局部动力学与长程一致性所需的 state 上，而不是浪费在高方差低价值的静态外观/纹理。
- **机制原因**: 视频中高方差的是背景纹理（nuisance），重构会优先编码它；innovation 才对应 velocity、occlusion、identity 等 causal 慢变量。
- **本质新意**: 首次明确把连续视频的「NTP analog」定义为**未来条件残差/速度项**，而非下一帧像素。
- **MVE**: 300M 级 latent causal video transformer，SSv2 64×64×48f 预训练，测 96f rollout；固定同一 fast-weight 载体，**四组头对头**：`No-TTT / Recon-TTT / Naive-next-latent-TTT / Innovation-TTT`。指标 FVD、frame-order accuracy、identity consistency。
- **Go 判据**: ≥2 个数据设置上，相对 recon-TTT，FVD ≥10% 改善 **或** 长程一致性 ≥5pt，且 3 seed 不翻车，**并且同时赢过 naive next-latent**（这是 novelty 关键）。
- **Risk**: MEDIUM。最大失败模式：innovation 只学到光流级短期运动，带不来全局语义一致性；或优势仅来自更强 teacher forcing 而非 state update。
- **撞车风险**: In-Place TTT（token 版）、One-Minute Video TTT、Pathwise Test-Time Correction。**护城河**：必须固定 carrier 只改 target，且证明胜过 naive predictive baseline，否则被判为换皮。
- **Contribution**: 新方法 + 假设检验。**先做 autoregressive latent video，不要一上来做 diffusion video**（否则被 Pathwise 直接对标）。

### ⭐ Idea 5+4（去风险核心，必做）: 诊断 + 最小理论
**5 — Why Reconstruction Fails: Nuisance-Capture Diagnostic**
- **Hypothesis**: recon-target TTT 会优先占满 fast-weights 容量去编码 nuisance（背景/纹理/相机抖动）；predictive target 才集中在 causal variables。
- **MVE**: 50–100M 视频 transformer + 合成因子化数据（显式分离 appearance-nuisance / dynamics / slow-identity）；probe / CCA / 谱分析，测 nuisance-R²、dynamics-R²、future NLL。
- **Go**: recon-TTT 的 nuisance probe 更高但 future NLL 更差，predictive-TTT 相反。
- **Risk**: LOW。**9 个里最能承载负结果**——即使发现「recon 并不更偏 nuisance」本身也是重要结果。这是整个项目的去风险地基。

**4 — Predictive Sufficiency Theorem（缩小版，支持线，不单独押注）**
- 在 latent Markov / 线性-高斯 / 部分可观测系统下，刻画最优 inner-loop target，证明其等价于 predictive sufficient statistic（≈ Bayes filter innovation），并明确 **reconstruction 仅在近静态/可逆观测的窄条件下最优**。
- **护城河**: 不能是 token induction-head 换皮，必须把 fast-weight 梯度更新与 predictive sufficient statistic 直接连起来，落在连续/部分可观测设定。
- **Risk**: MEDIUM-HIGH。失败模式：假设过强，落不到真实模型。**只做支持线，不把 6 个月主赌注压在纯理论。**

### Idea 2（次级扩展，仅当主线成立才开）: Epipolar-Predictive TTT for Streaming Multi-View
- **Hypothesis**: 多视角/空间流里，用「预测未见相机位姿下的 latent / depth-warp residual」替代重建当前视图，迫使 fast-weights 存 scene-centric 3D state（几何/遮挡/可见性）而非视角特有纹理。
- **MVE**: 150M posed multi-view transformer，CO3D / ScanNet 流式；冻结 backbone 只开 fast-weights；`recon-TTT vs unseen-view-TTT`。指标 pose-conditioned next-view NLL、LPIPS、object-pose retrieval。
- **Go**: 真实 posed 数据上，相对 recon-TTT，LPIPS ≥8% 改善或 retrieval/spatial-QA ≥4–5pt，且优势在遮挡/大视角变化场景更明显。
- **Risk**: MEDIUM。失败模式：与标准 per-scene novel-view finetuning 一比 TTT 独特性消失；只在 synthetic/精确 pose 成立。
- **撞车风险**: Spatial-TTT、整套 novel-view prediction 文献。**护城河**：必须证明学到 view-invariant scene state，而非更强 self-supervision。承载「geometry-aligned predictive sufficiency」原则才有独立价值。

---

## Eliminated / Downgraded Ideas（存档，避免重复踩坑）

| Idea | 处置 | 原因 |
|------|------|------|
| #3 Binding-Impact TTT for Interleaved VLM | **砍** | binding target 定义太虚，负结果永远可被解释成 proxy 选坏；易退化成 logit distillation / grounding supervision |
| #7 Prefix-Scan Predictive TTT for 300k-token | **砍** | chunk-parallel 已是 TTT Done Right/In-Place 的工程套路，搬到视频本末倒置；负结果不可发，正结果只是更快难上顶会 |
| #8 Counterfactual-Action TTT for World Models | **砍** | 与标准 world-model action-conditioned next-state prediction 撞车严重，confound 太多，易被判为「在线微调 dynamics model」 |
| #6 Fisher-Guided Fast-Weight Grafting | **降级附录** | 概念近 PEFT 模块选择/参数显著性，非核心研究问题；只比固定写 W_down 略好则无意义 |
| #9 Multi-Horizon Slow-State TTT | **降级为 #1 的 ablation** | 与 #1 太近，不独立；但「一步预测天然错配慢变量」若被证实可作为 #1 的关键消融 |

---

## 6 个月研究路线图（≤8×A100，单人/小团队）

三线关系：**诊断线去风险 → 理论线给原则 → 方法线出顶会论文**。

| 阶段 | 时间 | 工作 | Go/No-Go |
|------|------|------|----------|
| 1 | 第 1 月 | 复现最小 recon-TTT baseline；搭 toy 因子化 benchmark；锁定 fast-weight carrier | — |
| 2 | 第 2 月 | 完成 probes + 最小理论推导（线 5+4）；**第一次 go/no-go** | recon vs predictive 在 matched compute 下无稳定差异 → 收缩为纯 empirical/diagnostic paper |
| 3 | 第 3–4 月 | 全力做方法线 1（#1），少数据强对照，不贪规模 | #1 只在单数据/单 seed/单 horizon 好看，或仅短期画质提升 → 收缩为「video TTT inner-target 经验研究」 |
| 4 | 第 5 月 | **仅当 #1 明显成立**才开方法线 2（#2）；否则本月补诊断 + 更强 baseline | #2 只在 synthetic 成立 / 与标准 novel-view self-sup 比 TTT 特性消失 → 仅作主论文边界实验 |
| 5 | 第 6 月 | 包装：最佳=单篇总论文（视频主结果 + 空间泛化 + 诊断/理论支撑）；次优=锋利视频论文 + 空间作附录 | — |

**最终下注**：主赌注 #1；去风险核心 #5+#4；次级扩展 #2。**明确不做** #3/#7/#8，#6 附录，#9 仅 ablation。

---

## Next Steps
- [ ] （可选）让 GPT-5.4 把保留的 3 条线写成 rebuttal 准备表：`核心 claim / 必须证据 / 致命弱点 / 预期 reviewer attack / 预先防守实验`
- [ ] `/novelty-check` 深查 Idea 1（重点对比 In-Place TTT、One-Minute Video TTT、Pathwise Correction）
- [ ] `/research-refine` 细化 Idea 1 的方法设计（innovation target 的精确数学形式）
- [ ] 复现最小 recon-TTT video baseline，搭因子化 toy benchmark（路线图阶段 1）
