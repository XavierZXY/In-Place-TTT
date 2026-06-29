# Novelty Check Report — Innovation-Aligned TTT for Continuous Video

**Date**: 2026-06-26
**Reviewer**: GPT-5.4 (xhigh) via Codex MCP
**Verdict**: ⚠️ **PROCEED-WITH-CAUTION** | Novelty **5/10**

---

## Proposed Method
在视频自回归/流式模型的 TTT 层中，把 fast-weights 的内环目标从主流的 reconstruction（回归当前 latent，MAE 式）改成 predictive/innovation（回归未来 latent 的预测残差/速度/去噪残差）。机制假设：重构浪费容量在静态外观 nuisance，innovation 迫使 fast-weights 编码 velocity/occlusion/identity 等 causal 慢变量。

---

## Core Claims — 逐项判定

| # | Claim | 新颖性 | 最接近的已有工作 |
|---|-------|--------|-----------------|
| (a) | 视频/连续模态把 TTT 内环目标从 recon 改成 predictive | **MEDIUM** | In-Place TTT —— 上位命题「reconstruction misaligned, predictive better」已被它讲过，你只是**视频实例化** |
| (b) | innovation/residual/velocity 作为内环回归目标（区别于 naive next-latent） | **LOW** | Stochastic Latent Residual Video Prediction —— residual/velocity 作视频 latent dynamics 参数化很老 |
| (c) | 「recon 浪费容量在 nuisance，innovation 抓 causal 慢变量」的机制+诊断 | **LOW** | "Prediction over Reconstruction" world-model 文献 + NextLat 的 belief-state 已说过该机制 |

---

## ⚠️ 最致命的数学观察（决定生死的 crux）

**朴素定义下，innovation 目标 ≡ next-latent 目标的 reparameterization。**

设 frozen base 预测 `b_t = stopgrad(g(x≤t))`，innovation `e_{t+1} = z_{t+1} − b_t`，让 fast-weights 回归残差 `r_W(k_t)`：

```
L_innov(W) = ‖r_W(k_t) − e_{t+1}‖²
           = ‖r_W(k_t) − (z_{t+1} − b_t)‖²
           = ‖b_t + r_W(k_t) − z_{t+1}‖²        ← 带 frozen additive skip 的 next-latent MSE
```

对 W 的梯度完全一样 → **reviewer 会直接说「innovation 只是 next-latent 残差化，不是新目标」。这一刀能杀死整个 novelty。**

### 唯一能守住「innovation ≠ next-latent」的三条出路（必须采纳至少一条）
1. **Whitened innovation**：`‖Σ_t^{−1/2}(r_W − e_{t+1})‖²` —— 加白化/协方差归一，不再等价于 raw next-latent MSE。
2. **明确把贡献定义为 capacity allocation / parameterization constraint**：frozen base 免费负责 easy/static 部分，fast-weights 只写 residual（容量受限下的最优分配），而非 target semantics。
3. **加正交/独立性/uncertainty weighting 约束**，让 residual 真正只承载 surprise。

> ⚠️ 还有一个滑点：`innovation / residual / velocity / denoising residual` 是**四个不同的东西**，必须钉死一个数学定义，否则 novelty statement 会显得很滑、被审稿人抓。

---

## 最危险的单一 prior：In-Place TTT

reviewer 最容易写出的致命句：**「核心 insight 已有——reconstruction 对 fast-weights 不对齐、future/task-aligned target 更好；你只是搬到视频。」**

（比 Spatial-TTT 更危险——Spatial-TTT 的 spatial-predictive 是 architecture/branch bias，不是 target swap；In-Place TTT 直接占了「inner-loop target alignment」这层概念地基。）

**可辩护前提**：把主张缩到很窄——「在连续视频中，**raw next-latent 仍然不对**，真正该写进 fast-weights 的是 **(whitened) innovation**，且必须在固定 carrier 下同时打败 reconstruction 和 raw next-latent」。

---

## 最优 Positioning（最大化可守新颖性）

❌ 不要卖：「first predictive TTT for video」/「prediction beats reconstruction」（这两层都已有强近邻）

✅ 应该卖：
- **"Target design for fast-weight memory in continuous video"**
- **"Under a fixed video TTT carrier, conditional/whitened innovation is a better WRITE target than reconstruction AND raw next-latent."**

> 红线：若实验只能证明 `predictive > reconstruction` 而证明不了 `innovation > next-latent`，novelty 守不住，会被压成「In-Place TTT 的视频移植版 + 一个 residual trick」。

---

## 必需的补强实验（让它 bulletproof 的唯一关键结果）

**一个带地面真值因子分解的线性高斯/合成视频实验，证明并验证「容量受限 fast-weights 的最优写入统计量是 whitened innovation」。**

最低配置：
- 视频 = `静态外观 nuisance a` + `动态隐状态 s_t`，含 occlusion / reappearance。
- 固定同一 fast-weight carrier，同算力、同参数量。
- **直接 probe fast-weights**：测它编码了多少 `a`、多少 `s_t`、多少 occlusion state。
- 期望看到：`Recon-TTT` 主要写 `a` → `Naive-next-latent-TTT` 写 `a+s_t` → `Innovation-TTT` 明显更偏 `s_t`/occlusion，且长程 rollout 更好。

没有这个结果，最近 prior 会把整篇压成一句：「predictive-target TTT 换个模态，再做了个 residual trick。」

---

## 对原 IDEA_REPORT 的影响（修正）

1. **Idea 1 的护城河必须升级**：从「innovation 赢 recon + naive next-latent」升级为「**whitened/conditional innovation**，并配套因子分解 probe 实验证明 capacity allocation 机制」。否则数学上等价于 next-latent，novelty 崩塌。
2. **Idea 5（诊断）从"加分项"升级为"必需项"**：它正是上面 GPT-5.4 要求的 bulletproof 实验，应作为论文第一张主图，而非可选支撑。
3. **Idea 4（理论）的定位收窄**：理论目标应精确锁定为「证明容量受限 fast-weights 的最优写入统计量 = whitened innovation / predictive sufficient statistic」，而非泛泛的 predictive sufficiency。

---

## Next Steps
- [ ] 用 `/research-refine` 把 innovation target 的**精确数学定义**钉死（建议直接采纳 whitened innovation + capacity-allocation framing）
- [ ] 设计因子分解 toy video benchmark（static appearance / dynamic state / occlusion 可控分离）+ fast-weight probe 协议
- [ ] （可选）`/research-review` 让外部再审一轮 whitened-innovation 的理论可证明性
