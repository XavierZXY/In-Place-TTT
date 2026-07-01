# Research Proposal: Residual-Write In-Place TTT — 用在线残差写规则消除 fast-weight 记忆的 cross-talk,并建立可预测 RULER 失败的容量定律

## Problem Anchor (immutable)
- **Bottom-line problem**: Qwen3-1.7B 上 In-Place TTT 的长上下文 RULER 检索质量不足(16k overall 0.594 vs base 上界 0.852;disable-ttt 仅 0.173,说明 TTT 在贡献检索但质量不够)。需要让 TTT fast-weight 记忆在长上下文下检索更准。
- **Must-solve bottleneck**: 当前 write rule `ΔW += ttt_lr·Σ_c(h_c⊗v̂_c)` 是无界外积累加。多个 chunk 的关联被线性叠加进同一个 ΔW,导致 **cross-talk / score dilution**:新写入稀释旧 needle,检索熵接近均匀。
- **Non-goals**: 不改 base MLP 预训练行为;不改 SWA/full 布局;不追求生成阶段问题(已证伪 repetition collapse 不是病因);不做 attention 侧 query update(qTTT 已占)。
- **Constraints**: ≤8 GPU;优先零训练/极少训练;已有探针 `eval/diagnostics/ttt_signal_probe.py` 可复用;改动需训练侧(`hf_models`)与推理侧(`inference_model`)数学一致。
- **Success condition**: (a) 零训练替换 write rule 后,RULER NIAH/VT 子集相对纯外积有可测改善(或明确证伪 cross-talk 假说);(b) 一组容量/碰撞指标能显著预测 per-sample RULER 失败(AUC 明显 >0.5),从而为后续写规则改动提供可信筛选器;(c) 短上下文(4k/8k)不退化。

## Technical Gap
当前 In-Place TTT 的 fast-weight 是 Hebbian 外积的无界前缀和。文献(Fast Weight Programmer 容量上限、Gated DeltaNet 的 delta-rule)指出纯外积关联记忆有两个结构性缺陷:(1)**容量有限**——当写入的 key 数远超有效秩,关联互相干涉;(2)**无擦除**——新写入只会叠加,不会修正已存储的错误/冗余。RULER 的 score dilution 正是这种 cross-talk 的行为表现:needle 的关联被海量 distractor 的关联线性淹没。

naive 修复不足:(i)加大 ttt_lr 只放大幅度不改干涉结构;(ii)keynorm 纯归一化已实测有害(0.04–0.13);(iii)加 decay 压幅度会同时擦掉远处 needle(score dilution 与"远"无必然关系,needle 可能在任意位置)。缺的是**写入时就扣除已被记忆解释的部分**——即 delta-rule / 在线最小二乘。

## Method Thesis
- **One-sentence thesis**: 把 TTT 的写规则从纯外积累加 `ΔW += k⊗v` 改成在线残差写入 `ΔW += η·(v − ΔW·k)⊗k/(λ+‖k‖²)`——只写"当前记忆尚未解释的残差",从机制上消除 cross-talk。
- **Why smallest adequate**: 这是对单行更新规则的局部替换,不加新可训练模块、不改 key/value 路径、不改布局。是直击 cross-talk 的最小干预。
- **Why timely**: delta-rule / test-time-regression 是 2025 序列建模的统一视角(Mamba2/Gated DeltaNet/MIRAS),但它们都在标准 linear-attention state 上;**把残差写规则落到 in-place-TTT-in-MLP 的 down_proj fast-weight 上是未被占据的领地**。

## Contribution Focus
- **Dominant contribution**: 残差/最小二乘写规则用于 in-place MLP fast-weight,零训练即可注入,消除 cross-talk。
- **Supporting contribution**: 一组 in-place TTT memory 的**容量/碰撞定律**(key mutual coherence、effective rank、condition number、read entropy、write collision),能预测 RULER 失败并作为写规则筛选器。
- **Explicit non-contributions**: 不声称新 feature map;不声称新 gate;不声称 attention 改动。

## Proposed Method
### Complexity Budget
- **Frozen / reused**: backbone、ttt_conv、ttt_proj、key=h、value=v̂、SWA/full 布局全部不变。
- **New trainable components**: 0(零训练诊断阶段)。残差写规则只引入两个标量超参 η、λ。
- **Tempting additions intentionally excluded**: decay 门控、key 归一化、独立检索模块、token 选择 gate。

### System Overview
零训练诊断分两条并行线,共享同一套 instrumentation:
```
现状 write:  S_t = S_{t-1} + Δ_t,   Δ_t = ttt_lr·(k_c ⊗ v̂_c)   [纯外积累加]
残差 write:  S_t = S_{t-1} + η·(v̂_c − S_{t-1}·k_c) ⊗ k_c / (λ+‖k_c‖²)   [delta-rule]
                 ↑ 两者都离线可重算:探针已能捕获 per-chunk 的 k_c, v̂_c
诊断:对同一批 RULER 样本,用现状 vs 残差两种 S_t 重算读出 O_t → 末位 logits + 容量指标
```

### Core Mechanism (残差写规则)
- **Input/output**: 输入 per-chunk 的 key `k_c`(=h)与 value `v̂_c`;输出更新后的 fast-weight `S_t`。
- **更新式**: `S_t = S_{t-1} + η·(v̂_c − S_{t-1} k_c) ⊗ k_c / (λ + ‖k_c‖²)`。残差项 `(v̂_c − S_{t-1}k_c)` 是"当前记忆对这个 key 的预测误差",只写误差部分。
- **训练/推理一致性**: 现状的 `cumsum`(`hf_models/.../modeling_qwen3.py:244`)是闭式前缀和;残差写规则**不是简单 cumsum**(S_{t-1} 出现在更新里,有序贯依赖),需改成 chunk 间串行扫描(chunk 内仍可并行)。训练侧与推理侧(`inference_model/.../modeling_qwen3.py:181` 的循环)要用同一递推,保证 bit 级一致。
- **Why main novelty**: 把 delta-rule 落到 in-place MLP fast-weight,且证明它治的是 RULER 的 cross-talk。

### Supporting Component (容量/碰撞定律)
- **指标**(每 TTT 层、每长度、每样本):
  - key mutual coherence: `max_{i≠j} |k_i·k_j|/(‖k_i‖‖k_j‖)`(采样 chunk 内 token)
  - effective rank of 累积 key 矩阵(谱熵)
  - condition number / write collision: 新 key 在已有 key 子空间的投影占比
  - read entropy: 门控行 softmax 熵(复用 Probe B)
- **用途**: 对每个 RULER 样本预测 correct/incorrect,算 AUC + length scaling;并对比"现状 write vs 残差 write"下这些指标是否改善。

### Modern Primitive Usage
- delta-rule / 在线最小二乘(test-time regression 视角)作为写规则。角色:write-rule replacement,不是新模块。比 decay/normalize 更对症,因为它显式建模"已存内容"。

### Integration
- 诊断阶段:全部在 `eval/diagnostics/ttt_signal_probe.py` 内离线重算(探针已能 hook、捕获 per-chunk delta/h、只过 inner model 避免 64k OOM)。**不碰训练代码**。
- 若诊断为正,再落地到 `hf_models`(训练)+ `inference_model`(推理)的 write rule,走训练验证。

### Training Plan
- 阶段 1(本提案核心):零训练。离线重算残差 write 的读出,对比 RULER 子集得分 + 容量指标。
- 阶段 2(仅当阶段 1 正信号):冻 backbone,只用残差 write rule 继续 CPT(沿用现有 ttt_train_only),small step 验证真实涨分。

### Failure Modes and Diagnostics
- **残差写需要矩阵-向量积 S_{t-1}k_c,离线重算成本高**:用低秩/采样近似 S,先在 2-4 个 TTT 层验证。
- **残差写在短上下文退化(护栏)**:4k/8k 上残差 vs 现状读出末位熵/检索不应显著变差。
- **容量指标与失败不相关(supporting 失败)**:本身是有价值的 negative result(说明 cross-talk 不是主因,需转向 value/objective 方向 idea ⑦)。

### Novelty and Elegance Argument
最近 prior:Gated DeltaNet(delta-rule 在 linear-attn state)。差异:本提案在 **in-place-TTT-in-MLP 的 down_proj fast-weight** 上做残差写,且**零训练即可注入并诊断**,并配套 in-place TTT 专属的容量定律。不是 gate、不是 feature map、不是 attention 改动——避开全部 banlist 正面战场。

## Claim-Driven Validation Sketch
### Claim 1 (dominant): 残差写规则消除 cross-talk,提升长上下文检索
- **Minimal experiment**: 零训练,在探针里对同批 RULER 16k 样本(NIAH multikey_2/3、multiquery、vt 这些最依赖检索的子任务)用现状 vs 残差 write 重算读出,比 trim 后 score。
- **Baselines/ablations**: 现状外积(γ=1 等价)、残差写(sweep η∈{0.5,1,2}, λ)、外加 keynorm(已知失败,作负对照)。
- **Metric**: RULER 子任务 score、read entropy、key coherence。
- **Expected evidence**: 残差写在 multikey/multiquery 上 score 上升且 read entropy 下降(更 spiky),短上下文不退化。

### Claim 2 (supporting): 容量/碰撞指标可预测 RULER 失败
- **Minimal experiment**: 不改模型,instrument 现状 run,按样本算指标 → 预测 correct/incorrect。
- **Baselines/ablations**: 单指标 vs 组合;跨长度 4k/16k/32k。
- **Metric**: 预测 AUC、与长度的相关。
- **Expected evidence**: 至少一个指标(预期 write collision / effective rank)AUC 显著 >0.6,且随长度恶化与掉点曲线吻合。

## Experiment Handoff Inputs
- **Must-prove claims**: 残差写 > 外积写(检索子任务);容量指标可预测失败。
- **Must-run ablations**: η/λ sweep;短上下文护栏;keynorm 负对照;逐层 vs 全层启用。
- **Critical datasets/metrics**: RULER 16k(NIAH multikey/multiquery/vt 子集)+ 4k/8k 护栏;trim 后 score、read entropy、collision。
- **Highest-risk assumptions**: (1) cross-talk 是主因(Claim 1 验);(2) 离线残差重算能忠实近似真实序贯 write。

## Compute & Timeline Estimate
- 阶段 1 零训练诊断: <8 GPU-h(探针 forward + 离线重算,几十样本 × 几长度)。
- 阶段 2(条件触发)small-step CPT: 数百 step,~1 GPU-day。
- Timeline: 诊断 2-3 天出 go/no-go。
