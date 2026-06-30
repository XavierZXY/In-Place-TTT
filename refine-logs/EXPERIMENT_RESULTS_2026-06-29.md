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
