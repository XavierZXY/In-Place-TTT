# v0 相对当前 base 的模型架构与训练方法改动说明

对比对象：

- base：`/zouxiangyu/codes/Learning/In-Place-TTT`
- 修改后版本：`/zouxiangyu/codes/TTT/In-Place-TTT-v0`

本文只分析 `v0` 相对当前 base 做了哪些模型架构、前向逻辑、训练方法和推理路径改动；不讨论 shell 脚本、日志、实验结果、评估数据、YAML 配置取值等外围差异。配置类会纳入分析，因为它们决定模型可用结构开关。

## 一句话结论

`v0` 是在 base 的简洁 In-Place TTT 实现上扩展出的实验版。base 主要是：

```text
Qwen3/Llama + MLP down_proj In-Place TTT + 标准训练循环
```

`v0` 进一步加入了：

```text
SWA 压缩/anchor full-attention + 多种 TTT 算法 + Spatial/LoRA/SGM 变体
+ TTT aux loss + 蒸馏训练 + 动态 window/chunk/anchor 退火
+ 更复杂的在线推理 TTT cache
```

因此，你理解的“加了 SWA 机制和 TTT 机制的结合”是成立的，但更精确地说：`v0` 在 base 的 MLP In-Place TTT 上，扩展了多种 SWA/TTT hybrid 结构和训练调度，而不是只加了一个简单 SWA 开关。

## 代码规模变化

关键文件规模对比：

| 模块 | base | `v0` | `v0` 改动含义 |
| --- | ---: | ---: | --- |
| `hf_models/hf_qwen3/modeling_qwen3.py` | 661 行 | 2612 行 | Qwen3 从基础 In-Place TTT 扩展为多算法实验平台 |
| `hf_models/hf_llama/modeling_llama.py` | 701 行 | 807 行 | Llama 增加 SGM/LoRA 算法分支 |
| `tasks/train_torch.py` | 636 行 | 984 行 | 训练循环加入 aux loss、退火、best/last checkpoint 管理 |
| `inference_model/hf_qwen3/modeling_qwen3.py` | 674 行 | 1811 行 | Qwen3 推理加入多种 TTT state、KV 裁剪和 attention/o_proj TTT |
| `inference_model/hf_llama3/modeling_llama.py` | 658 行 | 915 行 | Llama 推理加入 SGM/LoRA state |

`v0` 新增了几个 base 没有的关键模块：

- `lora_ttt.py`：LoRA-TTT / DeltaNet 风格低秩快权重实现。
- `in_place_ttt/ttt_aux/loss.py`：TTT V-hat 辅助损失。
- `tasks/distillation.py`：teacher/student 蒸馏编排。
- `hf_models/*/parallel_scan.py`：SGM、LoRA、gated scan 等递归聚合需要的并行 scan。
- `inference_model/*/parallel_scan.py`：推理侧对应 scan 工具。

这些新增文件说明 `v0` 不是小修补，而是围绕长上下文、SWA 压缩和 TTT 变体做了系统性实验扩展。

## base 的出发点

base 已经具备基础 TTT：

- 通过 `hf_models/__init__.py` 注册本地 Qwen3/Llama 模型。
- `ttt_mode` 控制是否开启 TTT。
- `ttt_layers` 控制哪些层启用 TTT。
- `ttt_chunk` 控制 chunk 大小。
- `ttt_lr` 控制快权重更新强度。
- `ttt_target` 支持 `hidden_states` 和 `input_embed`。
- Qwen3 仍保留标准 HuggingFace/Qwen3 的 `layer_types` / `sliding_window` 注意力机制。

base 的 Qwen3/Llama TTT 插入点都在 MLP：

```text
hidden_states
  -> MLP: h = SiLU(gate_proj(x)) * up_proj(x)
  -> 用 target 构造每个 chunk 的 ΔW
  -> cumsum 得到逐 chunk 演化的 down_proj fast weight
  -> output = W_chunk @ h_chunk
```

也就是说，base 已有的是“MLP down_proj 的 In-Place TTT”。

## v0 对 Qwen3 模型架构的改动

### 1. 增加 `ttt_algorithm` 多算法分发

base 的 `Qwen3MLP` 只有一条基础 TTT 路径：

```text
if layer in ttt_layers:
    使用 down_proj In-Place TTT
else:
    标准 MLP
```

`v0` 改成算法分发：

```text
ttt_algorithm == "in_place" -> 原始 full-rank down_proj TTT
ttt_algorithm == "sgm"      -> 低秩 Subspace Gradient Memory
ttt_algorithm == "lora"     -> LoRA-TTT / DeltaNet 风格快权重
ttt_algorithm == "spatial"  -> LaCT / Spatial-TTT 风格分支
```

这使 `v0` 可以用同一套 `ttt_mode + ttt_layers + ttt_target` 门控，切换不同 TTT 结构。

### 2. 扩展 In-Place TTT 的 V-hat pipeline

base 的 V-hat 只支持：

```text
target -> depthwise Conv1d(k=5) -> optional ttt_proj
```

`v0` 增加 `ttt_v_pipeline`：

- `linear`：保持 base 的 Conv1d + optional projection。
- `v3`：multi-scale depthwise conv `{5,17,65}` + pointwise conv + SiLU + SwiGLU。

`v3` 的目的不是改变 Transformer 主干，而是增强 TTT target/V-hat 的生成能力，让 V-hat 能看到多尺度局部上下文。

### 3. 增加 In-Place TTT update-rule 变体

base 的更新规则基本是：

```text
ΔW = V_hat outer h
W_i = W_down + cumsum(ttt_lr * ΔW)
```

`v0` 通过 `ttt_in_place_loss` 增加多种更新形式：

- `dot`：base 的原始 Hebbian outer product。
- `mse`：用 `V_hat - W_down @ h` 做 error feedback。
- `mse_norm`：在 `mse` 基础上加入 key normalization。
- `mse_running`：预测时使用随 chunk 演化的 running W。
- `deltanet`：running W + DeltaNet 风格 key normalization。

这部分把 base 的“纯外积更新”扩展成了更接近在线学习/DeltaNet 的更新族。

### 4. 增加 Muon / Newton-Schulz 正交化

`v0` 加入 `ttt_muon`：

```text
per-chunk ΔW -> Newton-Schulz 近似正交化 -> 再进入 cumsum
```

目标是调整每个 chunk 的更新方向，削弱奇异值尺度影响，使更新更像谱归一化后的方向更新。

base 没有这一层。

### 5. 增加 chunk decay

base 对 chunk delta 做普通累加：

```text
W_i = W_0 + Σ ΔW_j
```

`v0` 增加 `ttt_chunk_decay_mode`：

- `none`：base 行为。
- `hype_log`：按 chunk index 做 log 衰减。
- `hype_sqrt`：按 chunk index 做 sqrt 衰减。
- `recurrent_exp`：递归指数衰减，类似带遗忘的 recurrent state。

这使 TTT state 不再必须平等累计所有历史 chunk，可以控制长上下文下旧信息的影响。

### 6. 增加 gated accumulation

base 使用 `cumsum` 累计 ΔW。

`v0` 增加 `in_place_use_gate`：

```text
S_t = gate_t * S_{t-1} + lr * ΔW_t
```

gate 是数据相关的，用于替代固定 cumsum。初始化上尽量保持接近 cumsum，以便 warm start。

### 7. 增加 TTT RoPE 和 QK Norm

base 的 TTT 更新没有额外 RoPE/QK Norm。

`v0` 加入：

- `ttt_use_rope`
- `ttt_rope_target`
- `ttt_use_qk_norm`

可选把 RoPE 加到 TTT 的 h/k 侧、V-hat 侧或两侧，并对 TTT 的 h 侧加 RMSNorm。其动机是让 TTT 的 recurrent/cumsum 风格状态获得显式位置信号，并改善更新稳定性。

### 8. 增加 SGM

`v0` 的 SGM 是低秩状态记忆：

```text
h -> sgm_V -> low-rank key
target -> Conv1d -> sgm_U/projection -> low-rank value
delta -> gated/scan/cumsum aggregation
adapt_out -> sgm_U -> hidden
```

它不是 full-rank 更新 `down_proj`，而是在 rank-r 子空间里维护记忆状态，减少状态维度并提供 scan/gate 变体。

base 没有 SGM。

### 9. 增加 LoRA-TTT

`v0` 新增 `lora_ttt.py` 并在 Qwen3/Llama 中接入 `ttt_algorithm="lora"`。

主要能力：

- 低秩 fast weight。
- delta rule / natural gradient / standard GD / token delta / chunk delta 等 causal mode。
- momentum。
- surprise gating。
- per-rank alpha / HiPPO 风格初始化。
- 推理侧维护 LoRA-TTT state。

这条路径把 TTT 从 full-rank `down_proj` 快权重，扩展成更轻量的低秩 adapter 式在线更新。

### 10. 增加 Spatial-TTT / LaCT 风格分支

这是 `v0` 中与 SWA 结合最强的一部分。

base 的 TTT 只在 MLP `down_proj` 中发生。

`v0` 加入 `ttt_algorithm="spatial"`，支持 attention-attached 版本：

```text
SWA attention pre-o_proj output
  + Spatial-TTT branch output
  -> shared attention.o_proj
```

这意味着 TTT 分支可以直接与 SWA attention 并联，然后共享 `o_proj` 输出。相比 base 的 MLP-only TTT，这更接近“同一 attention block 内 SWA + TTT 的融合”。

`v0` 也保留 MLP-attached spatial ablation，但默认设计更偏向 attention-attached。

### 11. 增加 full-attention anchor 与 `ttt_compress_window`

base 依赖原始 Qwen3 配置里的 `layer_types` 和 `sliding_window`。

`v0` 在 `Qwen3Model.__init__` 中加入：

```text
if ttt_compress_window > 0:
    config.use_sliding_window = True
    config.sliding_window = ttt_compress_window
    非 full_attention_layers -> sliding_attention
    full_attention_layers -> full_attention
```

因此 `v0` 可以表达：

```text
多数层：SWA 压缩窗口
少数层：full-attention anchor
部分层：TTT
```

这就是 `v0` 中 SWA 和 TTT 结合的核心结构之一：SWA 负责局部窗口压缩，anchor 层保留全局路径，TTT 层提供跨 chunk 的可学习状态。

### 12. 增加 full-attention anchor 的 NoPE/位置策略

`v0` 增加 `attn_use_rope`，用于控制 full-attention anchor 层是否使用 RoPE。

设计意图是区分：

- SWA 层：保留 RoPE，因为窗口内相对位置稳定。
- full-attention anchor 层：可选 NoPE，避免长距离 RoPE 外推问题。
- TTT 层：由 `ttt_use_rope` 单独控制。

base 没有这套分层位置编码策略。

## v0 对 Llama 模型架构的改动

Llama 侧改动比 Qwen3 少，但方向一致。

base Llama 只支持 MLP `down_proj` In-Place TTT。

`v0` 增加：

- `ttt_algorithm` 分发。
- `sgm`。
- `lora`。
- 对应的 LoRA-TTT 参数。

Llama 没有 Qwen3 的 `layer_types` / SWA 体系，因此 `v0` 的 Llama 改动主要是扩展 TTT 算法，而不是实现 SWA hybrid。

## v0 对配置系统的改动

`v0` 的 `configuration_qwen3.py` 相对 base 增加了大量结构开关。

### SWA / anchor / schedule 配置

新增：

- `full_attention_layers`
- `ttt_compress_window`
- `rev_anneal_start_window`
- `rev_anneal_end_window`
- `rev_anneal_start_chunk`
- `rev_anneal_end_chunk`
- `rev_anneal_steps`
- `anchor_anneal_start_step`
- `anchor_anneal_end_step`
- `anchor_anneal_final_layers`

这些字段让模型和训练循环能动态表达：

```text
纯 SWA -> 逐步增大 window/chunk -> 逐步打开 full-attention anchor
```

### TTT 算法配置

新增：

- `ttt_algorithm`
- `ttt_v_pipeline`
- `ttt_v3_kernels`
- `ttt_muon`
- `ttt_in_place_loss`
- `ttt_in_place_mse_gate_init`
- `in_place_use_gate`
- `in_place_gate_bottleneck`
- `in_place_gate_init_bias`
- `ttt_chunk_decay_mode`
- `ttt_chunk_decay_param`

### SGM 配置

新增：

- `sgm_rank`
- `sgm_use_gate`
- `sgm_gate_init_bias`
- `sgm_aggregation`
- `sgm_alpha`

### Spatial-TTT 配置

新增：

- `ttt_spatial_chunk`
- `ttt_spatial_use_muon`
- `ttt_spatial_attach`
- `ttt_spatial_num_heads`
- `ttt_spatial_inter_multi`
- `ttt_spatial_qkv_silu`
- `ttt_spatial_no_v_silu`
- `ttt_spatial_use_momentum`
- `ttt_spatial_lr_dim`
- `ttt_spatial_base_lr`
- `ttt_spatial_learnable_scale`
- `ttt_spatial_fp32_states`
- `ttt_spatial_fw_init_gain`

### LoRA-TTT 配置

新增：

- `ttt_rank`
- `ttt_alpha_init`
- `ttt_clip_threshold`
- `ttt_use_delta_rule`
- `ttt_gram_eps`
- `ttt_use_momentum`
- `ttt_momentum_beta1`
- `ttt_momentum_beta2`
- `ttt_surprise_threshold`
- `ttt_surprise_metric`
- `ttt_surprise_gate`
- `ttt_teach_signal_weight`
- `ttt_causal_mode`
- `ttt_alpha_per_rank`
- `ttt_alpha_hippo_init`
- `ttt_alpha_hippo_min`
- `ttt_alpha_hippo_max`

### TTT aux / RoPE / QK Norm 配置

新增：

- `ttt_aux_loss_weight`
- `ttt_aux_loss_type`
- `ttt_jepa_loss_exp`
- `ttt_jepa_reg_coeff`
- `ttt_jepa_reg_eps`
- `ttt_use_rope`
- `ttt_rope_target`
- `ttt_use_qk_norm`
- `attn_use_rope`

这些字段说明 `v0` 不只是结构变化，也在探索 TTT 的监督信号和位置编码策略。

## v0 对训练方法的改动

### 1. 增加 TTT auxiliary loss

base 训练只对主 LM loss 做 backward：

```text
loss = model_outputs.loss * token_scale
loss.backward()
```

`v0` 增加 TTT aux loss：

- `in_place_ttt/ttt_aux/loss.py` 计算 V-hat 辅助监督。
- Qwen3 模型在 forward 中保存 `_last_ttt_aux_loss`。
- `hf_models/hf_qwen3/__init__.py` 包装 Liger `lce_forward`，把 `ttt_aux_loss` 挂到输出对象。
- `tasks/train_torch.py` 中 `_accumulate_ttt_aux_grads` 手动对 `ttt_conv` / `ttt_proj` 累积 aux gradient。

训练时变成：

```text
主 LM loss backward
+ 单独计算/累积 TTT aux loss 对 TTT 模块的梯度
```

这样可以让 V-hat pipeline 得到额外监督，而不是完全依赖 LM loss 间接训练。

### 2. 增加 reverse anneal

`v0` 支持训练中动态改变：

```text
sliding_window: rev_anneal_start_window -> rev_anneal_end_window
ttt_chunk:      rev_anneal_start_chunk  -> rev_anneal_end_chunk
```

每个 step 前执行：

```text
_apply_reverse_anneal(model, schedule, global_step, state)
```

它会同步更新：

- config 的 `sliding_window`
- 每个 Qwen3Attention 的 `sliding_window`
- 每个 TTT MLP 的 `ttt_chunk`

base 没有训练中动态调整 window/chunk 的机制。

### 3. 增加 anchor anneal

`v0` 支持训练中逐步启用 full-attention anchor：

```text
step < start:       没有 anchor，接近 pure SWA
start ~ end:        按顺序逐步激活 anchor
step >= end:        所有 final anchors 激活
```

它会动态修改：

- `model.config.layer_types`
- `decoder_layer.attention_type`
- `decoder_layer.self_attn.sliding_window`
- `model.has_sliding_layers`

base 的 `layer_types` 是初始化后的固定结构，不会训练中被改变。

### 4. 增加 best/last checkpoint 管理

base 的 checkpoint 行为是到 `save_steps` / `save_epochs` 就保存。

`v0` 增加：

- eval loss 改善时保存 best。
- 周期性保存 last。
- 删除旧 best/last，控制 checkpoint 数量。
- 训练结束保存 final last。
- 导出 HF 权重时优先使用 best checkpoint。

这使 `v0` 更像实验训练脚本，而不是简单的持续训练脚本。

### 5. 增加更复杂的 eval 流程

base 新增了简洁的 `_run_eval_loss` 和 `should_run_eval` 控制。

`v0` 的 eval 更服务于 best checkpoint：

- eval loss 参与 best checkpoint 选择。
- eval 指标进入 wandb 和本地日志。
- eval 路径和 dataloader 构建更直接地挂在训练脚本内。

注意：`v0` 的 eval 还包含跨 rank StopIteration 对齐逻辑，避免某些 iterable eval dataset 下 rank 间 collective 不一致导致 NCCL 卡死。

### 6. 增加蒸馏训练编排

`v0` 新增 `tasks/distillation.py`，支持：

- shared-backbone teacher。
- separate teacher/student。
- teacher 模式下禁用 TTT，把 SWA 替换/调整为 teacher 行为。
- CE + KL + hidden-state MSE 组合损失。
- 跳过 TTT 层 hidden state 对齐，避免压制 TTT 本身要学习的差异。

base 没有蒸馏训练模块。

### 7. wandb 依赖更宽松

base 直接 `import wandb`。

`v0` 改成 optional import：

```text
try:
    import wandb
except ImportError:
    wandb = None
```

所以 `v0` 在不安装 wandb 的环境里也可以启动训练，只要不开启 wandb logging。

## v0 对推理路径的改动

base 推理侧只维护基础 TTT state：

```text
TTTDynamicCache.ttt_states[layer] = (past_h, past_t, past_w)
```

用于 batch size 1 的 online down_proj fast-weight 更新。

`v0` 推理侧扩展了很多：

### 1. 增加 TTT module placement

`v0` 支持 `_resolve_ttt_role`：

```text
ttt_module == "down_proj"  -> MLP down_proj TTT
ttt_module == "o_proj"     -> attention o_proj TTT
ttt_module == "interleave" -> ttt_layers 中交替使用 down_proj / o_proj
```

base 只支持 down_proj TTT。

### 2. 增加 attention o_proj TTT

`v0` 的 inference Qwen3Attention 增加 `_forward_o_proj_ttt`，可以对 attention 的 `o_proj.weight` 做在线快权重更新。

这让 TTT 不再局限于 MLP，而可以直接作用在 attention 输出投影上。

### 3. 增加 SGM/LoRA 推理 state

`v0` 的推理 MLP 支持：

- `_forward_sgm`
- `_forward_lora`
- LoRA-TTT 的 momentum / low-rank state
- SGM 的低秩 scan state

base 没有这些推理路径。

### 4. 增加 Spatial-TTT 推理 state

`v0` 的 `TTTDynamicCache` 增加 `ttt_spatial_states`，保存 Spatial-TTT 的快权重和 pending token。

这对应训练侧 attention-attached Spatial-TTT。

### 5. 增加 KV cache window 裁剪

`v0` 的推理 cache 支持 `ttt_compress_window`：

```text
SWA 层：KV cache 裁剪到 window
full-attention anchor 层：保留完整 KV cache
```

这个逻辑是 SWA + anchor hybrid 在推理阶段成立的关键。没有它，SWA 层会无限积累 KV；如果错误裁剪 anchor 层，又会破坏 full-attention anchor 的全局语义。

base 没有这套 per-layer KV 裁剪。

## SWA 与 TTT 结合在 v0 中具体体现在哪里

`v0` 的 SWA/TTT 结合不是单点改动，而是多个层面的组合：

### 模型结构层

```text
多数层使用 sliding_attention / ttt_compress_window
少数层使用 full_attention anchor
TTT 层在 MLP down_proj、attention o_proj 或 Spatial-TTT branch 中提供跨 chunk 状态
```

### attention 内并联层

Spatial-TTT attention-attached 路径：

```text
SWA attention pre-o_proj output
+ TTT branch pre-o_proj output
-> shared o_proj
```

这是最直接的 SWA + TTT 并联融合。

### 训练调度层

`v0` 可以先训练更强压缩/更小窗口设置，再逐步：

```text
增大 sliding_window
调整 ttt_chunk
打开 full-attention anchors
```

这让模型从局部压缩逐步过渡到 hybrid 长上下文结构。

### 推理状态层

`v0` 同时维护：

- SWA 层裁剪后的 KV cache。
- full-attention anchor 层完整 KV cache。
- TTT 层的 fast weight state。
- Spatial-TTT pending chunk state。

这保证训练时的 hybrid 假设在 autoregressive decoding 中也有对应实现。

## 风险和复杂度

`v0` 的改动提升了研究表达力，但也显著增加复杂度。

主要风险：

- 配置组合空间大，错误组合可能静默改变实验含义。
- 训练循环每步动态修改模型属性，调试成本更高。
- aux loss 使用手动梯度累积，需要确保和 gradient checkpoint/FSDP 行为一致。
- 推理 cache 状态复杂，容易出现 prefill/decode 行为不一致。
- 多算法共存导致初始化、保存加载和参数冻结策略更难验证。

base 的优势是 KISS：路径短、行为稳定、容易复现。`v0` 的优势是研究能力强：可以系统探索 SWA/TTT/anchor/aux/distill 的组合。

## 建议阅读顺序

如果目标是理解 `v0` 做了什么，建议按下面顺序看：

1. `hf_models/hf_qwen3/configuration_qwen3.py`：先看所有新增结构开关。
2. `hf_models/hf_qwen3/modeling_qwen3.py`：重点看 `Qwen3MLP`、`Qwen3SpatialTTTBranch`、`Qwen3Attention`、`Qwen3Model.__init__`。
3. `tasks/train_torch.py`：重点看 aux grad、reverse anneal、anchor anneal、best checkpoint。
4. `tasks/distillation.py`：理解 teacher/student 和 TTT 禁用逻辑。
5. `inference_model/hf_qwen3/modeling_qwen3.py`：理解 TTTDynamicCache、KV 裁剪、o_proj TTT、Spatial state。
6. `lora_ttt.py`：如果关心 LoRA-TTT / DeltaNet 风格低秩更新，再单独深入。

## 最终判断

`v0` 相对当前 base 的核心改动可以归纳为：

```text
1. 把单一路径 MLP In-Place TTT 扩展为多算法 TTT 实验平台。
2. 把普通 Qwen3 SWA 支持扩展为 ttt_compress_window + full-attention anchors 的 hybrid 长上下文结构。
3. 加入 Spatial-TTT，使 TTT 可以与 SWA attention 在 pre-o_proj 处并联融合。
4. 加入 TTT aux loss、蒸馏、reverse anneal、anchor anneal，强化训练策略。
5. 加入更复杂的在线推理 cache，使 SWA 裁剪、anchor 全局注意力和 TTT fast weight 能同时工作。
```

所以，`v0` 的主线不是简单“加了 SWA”或简单“加了 TTT”，而是把 base 的 MLP In-Place TTT 扩展成一套围绕长上下文压缩的 SWA + anchor + TTT hybrid 实验系统。
