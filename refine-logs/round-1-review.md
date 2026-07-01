# Round 1 Review (gpt-5.4 xhigh)

Overall: **6.65/10, REVISE**. 最接近威胁是 DeltaNet/Gated DeltaNet/MIRAS 残差状态更新族(不是 qTTT/GDWM)。可辩护 delta = 落到 In-Place TTT 的 MLP down_proj fast weight + zero-training 诊断。

## Scores
| Dim | Score | Note |
|---|---|---|
| Problem Fidelity | 8 | 锚点清楚,不漂 |
| Method Specificity | 6 | 公式具体但 "RLS/least-squares" 表述过强;chunk 内更新语义不清 |
| Contribution Quality | 7 | 小机制干净可 falsify;但 delta-rule prior 很近 |
| Frontier Leverage | 7 | 用了 delta-rule,但须正面承认 Gated DeltaNet/MIRAS |
| Feasibility | 6 | sequential residual scan 破坏 cumsum 路径;offline probe 不能声称端到端 RULER |
| Validation Focus | 6 | offline recompute → RULER score 不够 faithful |
| Venue Readiness | 5 | 需先收敛成准确 update rule + faithful intervention |

## P0 修复
1. **命名**: 公式是 normalized LMS / delta rule, 不是严格 RLS。主方法改名 **Residual-Write / NLMS-TTT**;要 claim least-squares 则加 diagonal/low-rank P variant 作为 modernized RLS。
2. **chunk 内语义矛盾**: "chunk-serial scan + intra-chunk parallel" 自相矛盾。明确选 **block residual write**: `R=V−SK`, `S←S+η·R·Kᵀ/(λ+tr(KKᵀ))`。别同时声称 token-sequential 和 intra-chunk parallel。
3. **Feasibility**: 先在 inference_model 加 no-training flag 跑端到端 forward;hf_models 只在 positive 后同步。S 用 FP32 accumulator,读出 BF16。
4. **Validation faithfulness**: Phase 1 拆成 (a) local counterfactual probe(offline 重放,只报 entropy/collision/local recon error)和 (b) **faithful no-training forward**(每层用当前激活在线重算 k/v/S/readout,才能报 RULER score)。

## faithful 性裁决(关键)
- offline 顺序重放 captured k/v = 给定这些写入事件、换写规则的真实 S_t/readout,**faithful 于"固定写入下换规则"**。
- **不 faithful 于端到端模型替换**: 改 L 层 readout → 改后续层 hidden/attention KV/MLP intermediate/后续 TTT 层 k/v → 对未来 token 还经 causal attention cache 传播。
- 可 offline 报: read entropy/collision/local recon/counterfactual delta。**不可** offline 报: 真实 RULER accuracy(除非 rerun full forward 在线重算)。
- 例外: 只改最后一个 TTT 层、从该层输入重算到 lm head,offline 可近似 exact。all-layer 不行。

## P1
- claim "eliminates cross-talk" 过强 → 改 "reduces key-direction overwrite under finite-capacity associative memory",用 capacity metrics 证明何时有效/无效。

## Simplification
- 只留一个主方法 block residual write;砍 token-sequential/low-rank/capacity selector/Phase2 同时上。
- capacity law 瘦身到 3 指标: write collision, effective rank, read entropy。condition number 入 appendix。
- baseline 瘦身: current / residual / keynorm-negctrl。

## Modernization
- 最强升级: diagonal-RLS residual write(仍小,比 scalar 分母更对抗高频方向)。
- 正面承认 Gated DeltaNet prior;新意表述为 pretrained Transformer MLP fast-weight intervention,不是 delta-rule 本身。

## Drift: NONE(但若 Phase1 只做 offline probe 却宣称 RULER 提升 = 漂移)

<details><summary>raw</summary>见 thread 019f12e1-3b69-71d0-a56b-b91fefde53b4 round2 reply</summary></details>
