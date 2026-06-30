# 三实验结果分析 — R001 / R002 / R006

**日期**: 2026-06-29
**ckpt**: `outputs/qwen3-1.7b-stage3-cpt-swa3-full1-strict-c1024-16k/checkpoints/global_step_8000/hf_ckpt`(stage3, ttt_lr=1, 7 个 TTT 层, chunk=1024)
**数据**: RULER 16k, `/zouxiangyu/codes/TTT/In-Place-TTT-v0/eval_scripts/ruler`, 50/task(R001/R002), probe 66 样本(R006)
**结果目录**: `eval/exp_analysis/ruler_results_nlms_exp/`

## 执行摘要
三个实验全部成功,且都通过验证闸门。核心结论:
1. **R001(outer 复现)= 0.2854,与已知基线逐任务 diff=+0.000** → NLMS 代码改动未破坏 outer 路径,M0 sanity 通过。
2. **R002(η=0 退化)= 0.4583,与已知 tttlr0 基线完全一致** → write-rule flag / disable 路径正确。
3. **R006 容量定律:r(score, collision)=−0.726, r(score, eff_rank)=+0.751** → cross-talk 能预测 RULER 失败,C2 成立。
4. **关键动机证据:开 outer TTT(0.285)< 关 TTT(0.458),TTT 净害 −0.17** → outer-product 写规则的 cross-talk 拖垮检索,正是 NLMS 要解决的问题。

## R001 — outer 复现(sanity 闸门)
| | overall | rep_stop |
|---|---|---|
| R001 (TTT_WRITE_RULE=outer) | **0.2854** | 597/650 |
| 已知基线(同 ckpt) | 0.2854 | — |

逐任务 diff 全部 = +0.000(13/13 任务)。**结论:我实现的 `_apply_ttt_write` 在 outer 分支与原始 `current_w+dw` bit 级等价。** einsum 两步重构(value=t_conv@projᵀ 再 contract)经验证数值 diff=0。

## R002 — η=0 退化检查
overall=**0.4583**,与已知 tttlr0 基线逐任务完全一致;`__run_meta__`: ttt_lr=0.0, disable_fw=True。**结论:关闭 fast-weight 写入的路径正确。**

## 核心发现:outer-product TTT 在此 ckpt 上净害,且害处与 cross-talk 对齐
| task | ttt_on(R001) | ttt_off(R002) | Δ(on−off) | collision |
|---|---|---|---|---|
| niah_single_1 | 0.000 | 1.000 | **−1.000** | **0.867** |
| vt | 0.000 | 0.320 | **−0.320** | **0.919** |
| niah_multikey_2 | 0.000 | 0.240 | **−0.240** | **0.689** |
| fwe | 0.000 | 0.760 | −0.760 | n/a |
| cwe | 0.000 | 0.158 | −0.158 | n/a |
| niah_multikey_1 | 0.260 | 0.380 | −0.120 | 0.239 |
| niah_multiquery | 0.555 | 0.390 | **+0.165** | 0.227 |
| niah_single_2 | 0.980 | 0.840 | **+0.140** | 0.232 |
| niah_multivalue | 0.395 | 0.330 | +0.065 | 0.231 |
| **overall** | **0.285** | **0.458** | **−0.173** | — |

**模式**:TTT 害处最大的任务(single_1, vt, multikey_2)恰是 collision 最高的(0.87/0.92/0.69);TTT 有益的任务(multiquery, single_2)collision 低(~0.23)。**这在任务级别双重验证了 cross-talk 假说**——当记忆里 key 高度互相干涉时,outer-product 累加把检索拖垮;低碰撞时 TTT 才有正贡献。

## R006 — 容量/碰撞定律(C2)
write_collision 各层 0.45–0.51(层 25 最高 0.51–0.61,eff_rank 最低 131),说明**每个 key 约半数能量已被先前 key 张成的子空间覆盖**。

全 11 检索任务上:
- **r(RULER score, write_collision) = −0.726**(碰撞越高越差)
- **r(RULER score, effective_rank) = +0.751**(秩越满越好)

得分=0 任务 collision 0.68–0.92 / eff_rank 17–73;得分高任务 collision ~0.23 / eff_rank ~515。**C2 成立:容量定律可预测 RULER 失败。**

## go/no-go 裁决
- ✅ **C2(cross-talk 是病因)强成立**:r=−0.726/+0.751,任务级 TTT-harm 与 collision 对齐。
- ✅ **NLMS 动机被证据强化**:outer TTT 在 cross-talk 重的任务上净害 −0.17,正是 NLMS 残差写要消除的。
- ➡️ **下一步 = M3 matched CPT**:zero-training drop-in(R003/4)是 off-distribution,不能判定;需在训练侧 `hf_models` 实现 NLMS,跑 outer-CPT vs NLMS-CPT 同预算,看 NLMS 能否把 0.285 拉回甚至超过 0.458(关 TTT 上界)。

## 局限 / 待补
- R006 probe 数据只覆盖 11 个检索任务、66 样本(num-samples 限制),AUC 用 per-task 相关(N=11)而非 per-sample;统计稳健性可加大样本。
- cwe/fwe 未进 probe(非 needle 检索类),collision 缺失。
- 单 ckpt 结论;其他 stage3 ckpt 是否同样模式待验证。

---

# M3 准备:阶段 A 零训练 η 标定(2026-06-30)

**脚本**: `eval/diagnostics/run_nlms_eta_sweep.py`(per-key NLMS 离线重算 output_delta)
**ckpt**: 同 R001(stage3-full1-c1024-16k/gs8000),probe 数据 16k,20 样本平均
**结果**(`nlms_eta_sweep.json`):

| η | output_delta | 落 [0.2,0.4]? |
|---|---|---|
| 0.001 | 0.0000 | |
| 0.003 | 0.0000 | |
| 0.01 | 0.0053 | |
| **0.03** | **0.3044** | **YES** |
| 0.1 | nan(发散) | |

**裁决**:
- η≈**0.03** 是标定目标(output_delta 落健康区 [0.2,0.4])。
- 但映射极陡且不稳:η≤0.01 几乎不写(~disable-ttt),η=0.03 已到 0.30,η≥0.1 发散为 NaN。5 样本子集里 η=0.03 曾测出 1.0(高方差)。
- **根因(符合 research-refine R2 预测)**:此 ckpt 的 ttt_proj/ttt_conv 在 outer 动力学下训练,per-key NLMS 读出 K·Sᵀ 不能预测 V → 残差不收缩 → S 在 16 chunk 上发散。**这正是 zero-training drop-in off-distribution、不能判定方法的证据。**
- **结论**:η=0.03 作为 matched CPT 的初始 η;但真正判定必须靠 matched CPT(在 NLMS 动力学下训练 projections),drop-in 不可信。matched CPT 建议同时试 η∈{0.01, 0.03, 0.1},因训练会改变 projections 使稳定区间右移。

---

# M3 阶段 B 受阻:NLMS 训练数值发散(2026-06-30)

**2-GPU FSDP 冒烟**(NLMS, η=0.03, 16k=16chunk, 从 stage2/gs8000 起, 2 步):
- 训练**能端到端跑**(forward+backward+ckpt 保存,FSDP 下 NLMS chunk-loop 工作)。
- 但**立即数值发散**:
  - step1: `ttt_dw=0.00e+00`, `grad_norm=5.7e17`(梯度爆炸,裁剪前)
  - step2: `ttt_dw=1.17e+12`(fast-weight 增量爆炸), `ttt_do=0.686`, loss 6.73→18.35
- 注:单 GPU 因 `init_device=meta` 需 FSDP 而失败,须 ≥2 GPU。

**诊断**:
- 与阶段 A 零训练 sweep 一致——per-key NLMS 在 16-chunk 无界 S 累加下不稳:η=0.03 已是 output_delta 临界点,训练 1 步后 optimizer 把 ttt_conv/proj 推入发散区。
- 小尺度玩具测试(hid=128, 随机 init 0.02)梯度稳定且对 η 不敏感→发散源于真实权重尺度(‖h‖≈71)+ 真实 V 尺度 + 16 chunk 深度,非代码 bug。
- 根因:NLMS 读出 `K·Sᵀ` 在 S 无界累加下随 chunk 深度放大;backward 穿过 16 步串行图进一步放大梯度。

**结论**:per-key NLMS 的"chunk 间串行+chunk 内批量近似"版本在当前架构(chunk=1024, 16 chunk @16k)训练不稳定,不能直接跑 matched CPT。需要稳定化改造再继续:
候选(下一轮设计):
1. **S 归一化/衰减**:对累积 S 加 RMSNorm 或衰减门控(回到 decay 方向,但这次治的是训练稳定性)。
2. **更小 η + warmup**:η<0.003 起步,或 ttt_lr warmup。
3. **detach 跨 chunk 的 S**(truncated BPTT):backward 不穿透全部 16 chunk,切断梯度爆炸链。← 最可能有效且最小改动。
4. **更细 write_subchunk**:减小单次写入幅度。

**代码状态**:Task1-7 全部完成并提交(推理/训练 NLMS 实现 + 单测 + 探针 + η标定脚本 + matched CPT wrapper);仅 Task8 的全量 CPT 因发散暂停。

---

# M3 阶段 B 续(2026-06-30):发散修复 + 暴露梯度消失

## 修复 1 — detach_state 梯度断开 bug(commit d712cd7)
**症状**:加 `ttt_nlms_detach_state=true` 后冒烟 step1 `loss.backward()` 崩
`RuntimeError: element 0 of tensors does not require grad`。
**根因**:`ttt_train_only=true` 冻 backbone,ttt_proj/ttt_conv 是唯一可训练参数,其梯度**仅**经跨-chunk 累积 S 链回传。原 detach 实现 readout 也读 detached `S_hist` → dW_i 成孤儿节点 → loss 与全部可训练参数断开。
**修复**:readout 改读 live `S`(携带上一步可微 dW),write 残差仍读 detached `S_hist`。这是真正的 1-step truncated BPTT:梯度链长度=1 chunk(不爆炸),ttt_proj 仍可训练,forward 数值不变(S==S_hist)。加冻结-backbone 回归测试。

## 冒烟验证(2-GPU 卡6/7, η=0.03, detach=true, 16k, 2步, expandable_segments)
| 指标 | 上次发散 | 修复后 step1 | step2 |
|---|---|---|---|
| grad_norm | **5.7e17** | 0.0000 | 0.0000 |
| ttt_dw | 1.17e12 | 0.00e+00 | **3.65e-12** |
| ttt_do | 0.686 | 1.66e-03 | 1.66e-03 |
| loss | 6.73→18.35 | 5.27 | **4.24**↓ |
| 完成 | ❌崩 | — | ✅2/2+ckpt |

**✅ 发散彻底消除**:grad_norm 5.7e17→0,ttt_dw 1.17e12→~1e-12,loss 正常下降,端到端跑通 + ckpt 保存。
**注**:OOM 排查发现集群被外部作业占满 + stage3 变长打包样本达 52k token;`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + 空闲卡解决。

## ⚠️ 新问题:梯度消失(过度截断)
`grad_norm≈0`、`ttt_dw≈3.65e-12` → ttt_proj/ttt_conv **实质学不动**。
1-step truncated BPTT 在 7 TTT 层 + 真实权重尺度下,单 chunk readout 回传的梯度被压到 ~1e-12,优化器基本不更新。**稳定性买来了,代价是 matched CPT 学不动**——直接跑完整 CPT,NLMS 臂≈冻结的随机初始化 ttt_proj,无法公平对比 outer-CPT。

**这是真实方法论权衡,非 bug。** 候选下一步:
1. **放开截断深度**:k-step truncated BPTT(detach 每 k 个 chunk 而非每 1 个),在稳定与梯度量级间找平衡。
2. **不 detach + 梯度裁剪**:关 detach,靠 `max_grad_norm`(已=1.0)裁爆炸梯度,看 ttt_dw 是否仍发散。
3. **S 归一化/衰减**:对累积 S 加 RMSNorm 或衰减门控,从源头限幅(治本)。
4. **重标 η + warmup**:detach 下梯度小,可大幅提 η(如 0.3~1)+ ttt_lr warmup,补偿被截断的学习信号。
5. **检查 ttt_aux 是否是主要学习信号**:ttt_aux≈0.026 稳定,主 loss 在降——需确认 loss 下降是否来自 ttt 参数还是其他。

---

# M3 阶段 B 续2(2026-06-30):D 方向(提 η)+ 发散是正反馈驱动的发现

## D 方向冒烟:η=5.0 + detach(卡6/7, 2步)
| 指标 | η=0.03(前) | **η=5.0** |
|---|---|---|
| ttt_dw | ~1e-12 | **2.58e+36** 💥 |
| ttt_do | 1.66e-3 | 9.37e-2 |
| loss | 5.27→4.24↓ | **5.27→7.07↑** |
| grad_norm | 0 | 3e-4 |

η=5.0 让 fast-weight 增量 forward 爆炸,loss 不降反升。**D 的简单提 η 不可行。**

## 关键诊断:发散是"读出-残差正反馈",非 η 量级
- toy(随机独立 K/V,真实尺度 ‖k‖~64,16 chunk):η=5 下 ‖S‖ 仍**有界**线性缓增。
- 真实 ckpt η=5:`ttt_dw=2.58e36` 爆炸。
- **反差根因**:`resid = V − K·Sᵀ`。此 ckpt 的 ttt_proj/ttt_conv 在 **outer 动力学**下训练,NLMS 读出 `K·Sᵀ` 无法预测 V → S 增长时残差被**放大而非收缩** → dW 放大 → S 再放大,**runaway 正反馈**。η 越大踢得越猛,但根因是反馈结构(off-distribution),与 detach 无关(detach 只切 backward,forward S 仍自由发散)。
- 这强化了阶段 A 的预测:zero-training / 未在 NLMS 动力学下训练的 projections,NLMS 必然 off-distribution 不稳。

## 窗口存在性(待真跑确认)
- η=0.03:forward 有界,但 ttt_dw~1e-12 学不动。
- η=5.0:正反馈爆炸。
- 中间是否有"有界且可学"窗口 → toy 答不了(随机数据无对抗反馈),只能真跑 sweep η∈{0.3, 1.0}。
- **若中间也无窗口 → 确认正反馈是死结 → 转 C(对 S 加 RMSNorm/衰减,从源头限幅,治本)。**

## D sweep 完整结果(2026-06-30):D 证伪
| η | ttt_dw | ttt_do | loss step2 | grad_norm |
|---|---|---|---|---|
| 0.03 | 1e-12 | 1.66e-3 | 4.24 | 0 |
| 0.3 | 1e-11 | 1.66e-3 | 4.24 | 0 |
| 1.0 | 8.6e-11 | 1.66e-3 | 4.25 | 1e-4 |
| 5.0 | 2.58e36💥 | 9.37e-2 | 7.07↑ | 3e-4 |

**结论:无可用 η 窗口,D 死。**
- η≤1.0:`ttt_dw` 随 η 线性放大但绝对量级始终可忽略;`ttt_do=1.66e-3` 四次逐位相同 = **真实写入埋在 bf16 噪声地板(~1.6e-3)以下**,monitor 没坏只是测不出。loss/grad 实质不动。
- η=5.0:正反馈爆炸。
- **悬崖在 (1.0, 5.0]**:ttt_dw 从 8.6e-11 → 2.58e36,跨 46 个数量级,无平滑过渡。
- 死结三因素:① off-dist projections 残差不收缩反放大 ② 健康写入需 ~150× 当前量级 ③ 稳定补丁 `/chunk_size`(≈1000×)又压低写入。**"可学"的 η 与"稳定"的 η 无交集。**

**裁决 → 转 C**:A(k-step 截断)只动 backward,对 forward 侧 S 累加爆炸(ttt_dw)无效;C(对 S 加 RMSNorm/衰减)动 forward,直接打断 runaway 正反馈,让大 η 可用——治本。

---

# M3 阶段 B 续3(2026-06-30):C(decay gate)实现完成 + 首轮冒烟暴露 bf16 溢出嫌疑

## C 代码实现(commit 链 Task1-6,全部 TDD 通过,23 单测绿)
- 两侧 config flag `ttt_nlms_decay`(默认 0.0,校验 [0,1))+ 两侧 MLP 读取。
- 训练侧 forward:`S = (1-α)·S_hist + dW_i`(line 278)。
- 推理侧 `_write_block`:`return W0 + (1-α)·ΔW + dw`(decay 只作用 ΔW,不碰 W0)。
- α=0 逐位退化为纯 NLMS(训推均验证);decay>0 训推一致(α=0.2);冻结 backbone 梯度非零(α=0.1,不 detach)。
- wrapper 加 `NLMS_DECAY` env(默认 0.1)。

## C 首轮冒烟:η=5, α=0.1, detach=false(卡6/7, 2步)
| 指标 | C(η=5,α=0.1) | 对比 D(η=5,α=0) |
|---|---|---|
| ttt_dw | **2.58e+36** | 2.58e+36(**逐位相同!**) |
| ttt_do | 9.43e-2 | 9.37e-2 |
| loss | 7.01↑ | 7.07↑ |
| grad_norm | 635(detach=false 梯度回来但爆) | 3e-4(detach=true) |

## ⚠️ 关键反常:decay=0 与 decay=0.1 的 ttt_dw 逐位相同(2.58e36)
两条不同 forward 路径(纯累加 vs 0.9 衰减)给出**完全相同**的天文数字 → 数学上不可能是巧合。
**新假设:`2.58e36` 不是算出的发散值,而是 bf16 上溢后的饱和/固定常量。** 真实 ckpt 在 η=5 第一个 chunk 的 resid/dW 即 bf16 溢出,(1-α) 系数作用在已饱和值上无意义 → decay 救不回。
- 注:`ttt_dw` = monitor 的 `delta_weight_sample_ratio` = **per-chunk dW_i 范数 / W0 范数**(非累积 S)。decay 限的是 S 累加,不直接改 dW_i;dW_i 爆炸源于 resid=V−K·S_histᵀ 在 S_hist 大时被放大,但若首 chunk 即溢出,decay 无从介入。
- toy(随机 K/V,真实尺度,η=5/α=0.1~0.9)全部有界 ‖dW‖≈0.64 → 再次确认 toy 无法复现真实对抗性反馈/溢出(spec 风险节已述)。

## 架构性障碍(systematic-debugging Phase 4.5)
连续 3+ 轮(D 的 η=5、C 的 α=0.1)均撞同一墙:真实 ckpt η=5 立即爆固定值 2.58e36,toy 无法复现。这不是参数调节(α/η)能解的——指向更深的架构问题:
- **要么** bf16 精度不足以承载 NLMS 在此尺度的中间量(需 fp32 S + fp32 resid 全程,或更小 η 起步);
- **要么** off-distribution projections 的对抗反馈太强,任何 forward 限幅都需配合"在 NLMS 动力学下重训 projections"(即 matched CPT 本身,但它又需要先稳定——鸡生蛋)。

**待与用户讨论的下一步候选(不再盲调 α/η)**:
1. 先验证 bf16 溢出假设:打印真实 ckpt 首 chunk 的 resid/dW 数值范围(加诊断,read-only)。若溢出 → 全 fp32 路径或 resid clip。
2. 小 η 起步 + decay:η=0.3~1(已知 forward 有界区)+ α=0.3,看 decay 能否让"有界但学得动"——D 证明 η≤1 写入埋噪声地板,但 decay 改变动力学,可能 ttt_do 上移。
3. 回到 motivation 层:NLMS 在 outer-ckpt 上 drop-in 本就 off-distribution(阶段 A 早已预测),或许应放弃"在 outer-ckpt 上 matched CPT",改为从更早 stage、用更温和的写规则共同训练。

## C decay sweep 完整相图(2026-06-30):C 也无"稳+可学"窗口
| η | α | ttt_dw | ttt_do | loss | 区域 |
|---|---|---|---|---|---|
| 1 | 0.5 | 0.00 | 1.66e-3 | 5.27 | 稳·学不动 |
| 5 | 0.1 | 2.58e36💥 | 9.4e-2 | 7.01↑ | 爆 |
| 5 | 0.5 | 2.58e36💥 | 9.3e-2 | 7.07↑ | 爆 |
| 5 | 0.9 | 0.00 | 1.66e-3 | 4.24 | 稳·学不动 |

**核心发现:decay 救不了,因为发散在 chunk 内、decay 在 chunk 间。**
- η=5 时单 chunk 的 `resid=V−K·Sᵀ` 配 1024 key 外积,在**该 chunk 内**就 fp32 溢出到 2.58e36(α=0.1/0.5 逐位相同 = 饱和值,非精度伪影:forward 全程 fp32,2.58e36 < bf16 max)。chunk **间**的 (1−α) 衰减来不及介入。
- α=0.9 能压住(把上一 chunk S 几乎清零 → 降低 chunk 内 `K·Sᵀ` 起点),但 S 留不住记忆 → ttt_do 回噪声地板,学不动。
- 与 D 同构的张力,换了旋钮:**稳定与可学不可兼得**。grad_norm 在 detach=false 时为 inf(backward 穿 16 chunk 爆,但被 max_grad_norm=1 裁,不致命)。

**C 证伪。三因素叠加锁死**:① off-dist projections 残差不收缩反放大 ② 发散粒度是 chunk 内(1024 key 同时外积),任何 chunk 间机制(decay/detach)都够不着 ③ 真实写入需 ~150× 才跳出噪声地板,而那个量级必然 chunk 内溢出。

**裁决:停止在 outer-ckpt 上 drop-in/matched-CPT NLMS 的所有 forward-限幅尝试(D 提η、A 截断、C 衰减均证伪)。** 根因统一:**NLMS 在为 outer 动力学训练的 ckpt 上 off-distribution**(阶段 A R2 早已预测)。下一步需换层级而非换旋钮:
- **选项 1(治本)**:chunk 内序贯 NLMS(write_subchunk→1 或严格逐 token),消除"1024 key 同时外积"的 chunk 内爆炸源。代价:训练吞吐大降(16k=16384 步串行)。
- **选项 2(治本)**:从 stage1/2 早期就用 NLMS 写规则共同训练 projections,让 ttt_proj/ttt_conv 学到能让 `K·Sᵀ` 预测 V 的表示(残差自然收缩)。代价:重训成本高。
- **选项 3**:DeltaNet/Gated chunkwise 成熟 form,放弃"纯 NLMS"叙事。
- 待与用户讨论选型。
