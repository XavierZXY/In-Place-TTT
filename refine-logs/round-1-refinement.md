# Round 1 Refinement

## Problem Anchor (verbatim)
- **Bottom-line**: Qwen3-1.7B In-Place TTT 长上下文 RULER 检索质量不足(16k 0.594 vs base 0.852;disable-ttt 0.173)。
- **Bottleneck**: write rule `ΔW += ttt_lr·Σ_c(h_c⊗v̂_c)` 无界外积累加 → cross-talk / score dilution,检索熵近均匀。
- **Non-goals**: 不改 base MLP 预训练;不改 SWA/full 布局;非生成/repetition;非 attention query update。
- **Constraints**: ≤8 GPU;优先零/少训练;复用探针;训练/推理数学一致。
- **Success**: (a) 残差写规则给 RULER 检索子任务可测改善或干净证伪;(b) 容量指标预测失败 AUC>>0.5;(c) 短上下文不退化。

## Anchor Check
- 原 bottleneck = 外积累加的 cross-talk。修订后方法(block residual write)仍直击它。
- 拒绝的漂移:reviewer 没要求漂移;但我自我纠正——**不能用 offline 单遍重放声称 RULER 提升**,否则从"修模型 retrieval"漂成"固定激活上的 memory algebra 分析"。

## Simplicity Check
- Dominant contribution 收敛为单一:**block residual write rule**(NLMS 形式)。
- 删除:token-sequential 表述、low-rank approx(降为可选)、Phase2 与诊断并列(改为严格条件触发)。
- capacity law 瘦身到 3 指标(collision / effective rank / read entropy)。
- 仍是最小机制:一行 update rule 替换,无新可训练模块。

## Changes Made
### 1. 命名与数学:RLS → NLMS / block residual write
- Reviewer said: 公式是 normalized LMS/delta rule,不是严格 RLS(RLS 需维护 inverse covariance P)。
- Action: 主方法更名 **NLMS-TTT (Normalized-LMS residual write)**;严格 least-squares 降为可选 **diagonal-RLS variant**(modernized,维护对角 P)。
- Impact: claim 更诚实,不会被审稿抓表述硬伤。

### 2. 明确 block residual write,消除"chunk 内并行"矛盾
- Reviewer said: "chunk-serial + intra-chunk parallel" 自相矛盾。
- Action: 锁定 **block residual write**:每 chunk 用 chunk-start 的 S 计算残差 `R = V − S·Kᵀ`(整块),`S ← S + η·R·K / (λ + tr(KᵀK))`。chunk 间串行,chunk 内是一次矩阵运算(块更新,不是 token 递归)。
- Impact: 语义明确,可实现,与现有 chunk 结构兼容。

### 3. 验证拆成 faithful forward + offline probe(最关键修复)
- Reviewer said: offline 单遍重放 captured k/v 不 faithful 于端到端(改 L 层 memory 改 L+1 层 k/v),不能报 RULER。
- Action: Phase 1 拆两条:
  - **1a Faithful no-training forward**:在 `inference_model` 加 `ttt_write_rule` flag,**整模型重跑 forward**,每层用当前激活在线算 k/v/S/readout。**只有这个能报 RULER score**。
  - **1b Offline counterfactual probe**:在 `ttt_signal_probe.py` 离线重放,**只报** read entropy / collision / local recon error / counterfactual delta(诊断,不报 RULER)。
- Impact: 杜绝漂移;RULER 结论 faithful。

### 4. claim 弱化 + 正面承认 prior
- Reviewer said: "eliminates cross-talk" 过强;须承认 Gated DeltaNet/MIRAS。
- Action: claim 改 "**reduces key-direction overwrite under finite-capacity associative memory**";novelty 表述为 **pretrained Transformer MLP fast-weight 上的 residual-write intervention + zero-training 诊断**,delta-rule 本身归功于 DeltaNet/Gated DeltaNet。
- Impact: 守住可辩护的 delta。

## Revised Proposal

### Method Thesis
把 In-Place TTT 的 chunk 写规则从纯外积 `S += η·KᵀV` 改成 **block residual write (NLMS)**:`R = V − S·Kᵀ; S ← S + η·R·K/(λ+tr(KᵀK))` —— 每个 chunk 只写"当前 fast-weight 尚未解释的残差",在有限容量关联记忆下**减少 key 方向的覆盖/稀释**。最小机制:替换一处 chunk 更新;无新可训练模块;key/value/布局不变。可辩护 delta:落到 pretrained Transformer 的 **MLP down_proj fast-weight**(In-Place TTT 独有接口)+ 零训练可注入诊断。

### Contribution Focus
- **Dominant**: MLP fast-weight 的 block residual write rule(NLMS),零训练注入,减少 key 覆盖。
- **Supporting**: in-place TTT 专属容量定律(write collision / effective rank / read entropy)预测 RULER 失败 + 当写规则筛选器。
- **Non-contributions**: 不声称新 feature map / gate / attention 改动 / 发明 delta-rule。

### Complexity Budget
- Frozen: backbone, ttt_conv, ttt_proj, key=h, value=v̂, 布局。
- New trainable: 0(诊断阶段)。仅 2 标量 η, λ。可选 diagonal-RLS 加一个对角 P(仍无矩阵逆)。
- Excluded: decay, keynorm, 独立检索模块, token gate, low-rank approx(默认不开)。

### Core Mechanism (block residual write)
- 每 chunk c:K_c=[k]∈R^{C×d_h}, V_c=[v̂]∈R^{C×d}, S∈R^{d×d_h}(= ΣΔW 的当前值)。
- 残差:`R_c = V_c − K_c·Sᵀ`(chunk-start S,整块一次算)。
- 更新:`S ← S + η·(R_cᵀ·K_c)/(λ + tr(K_cᵀK_c))`。
- 读出:`O_t = h_t@W₀ᵀ + h_t@Sᵀ`(base/delta 路径分离不变)。
- **一致性**:chunk 间串行递推,训练侧(`hf_models`)与推理侧(`inference_model`)共享同一递推,S 用 FP32 accumulator,读出 BF16。**不再是 cumsum**——这是相对现状实现的主要工程改动点。
- 可选 **diagonal-RLS variant**:维护对角 P,更新 `S += η·R·(P⊙K)/(...)`,更抗高频方向。

### Supporting (capacity law, 瘦身到 3 指标)
- write collision:新 chunk key 在已写 key 子空间的投影占比。
- effective rank:累积 key 矩阵谱熵。
- read entropy:门控行 softmax 熵(复用 Probe B)。
- condition number 入 appendix(数值不稳)。
- 用途:per-sample 预测 correct/incorrect(AUC + length scaling);对比 current vs residual write。

### Validation (拆 faithful / offline)
**Claim 1 (dominant): block residual write 减少 key 覆盖、提升长上下文检索**
- **1a Faithful forward(报 RULER)**:`inference_model` 加 `ttt_write_rule={outer,nlms,nlms_rls,keynorm}` flag,整模型重跑 RULER 16k(NIAH multikey_2/3, multiquery, vt 子集)。这是**唯一**报 RULER score 的实验。
- baseline: outer(现状)/ nlms (sweep η∈{0.5,1,2}, λ)/ nlms_rls / keynorm(负对照)。
- metric: trim 后 score;护栏 4k/8k 不退化。
- expected: nlms 在 multikey/multiquery 提升 + 短上下文不退化。

**Claim 2 (supporting): 容量指标预测失败**
- offline instrument 现状 run,3 指标预测 per-sample 对错,AUC + 长度相关。
- expected: ≥1 指标(collision/eff-rank)AUC>0.6,随长度恶化吻合掉点。

**1b Offline probe(只报诊断)**:离线重放 current vs nlms 的 S,报 read entropy / collision / local recon error / counterfactual delta。**不报 RULER**。

### Failure Modes
- faithful forward 实现复杂(改 inference 递推)→ 先在 inference 加 flag、小样本验证数值与现状 outer 路径一致(η=0 或 nlms 退化检查),再开 nlms。
- 短上下文退化(护栏)→ 4k/8k 必须不退化。
- 容量指标无关 → 有价值 negative result,转 value/objective 方向(idea ⑦)。

### Novelty
closest: Gated DeltaNet / DeltaNet(delta-rule on linear-attn state)、MIRAS。delta:residual write 落到 **pretrained Transformer MLP down_proj fast-weight**,零训练可注入 + 诊断,配 in-place-TTT 专属容量定律。非 gate/feature-map/attention。

### Compute
- 1a faithful forward: RULER 子集 × 4 write rules × 几长度,<8 GPU-h。
- 1b/Claim2 offline: 探针几十样本,<2 GPU-h。
- Phase 2(条件触发)small-step CPT with nlms:~1 GPU-day。
- Go/no-go 2-3 天。
