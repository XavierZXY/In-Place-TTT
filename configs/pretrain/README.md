# Pretrain Config 说明

本目录存放继续预训练、长上下文训练、TTT aux、SWA/Full hybrid、halo 对齐和蒸馏相关配置。训练入口通常是：

```bash
bash train.sh tasks/train_torch.py configs/pretrain/<config>.yaml
```

`scripts/train/longsft/*.sh` 是历史路径名，当前启动脚本已经按通用模板组织；数据路径、数据类型和序列长度都可以通过环境变量覆盖，不要求实验绑定某个固定数据集。

## 配置文件速览

| 配置 | 主要用途 |
| --- | --- |
| `qwen3_longct.yaml` | Qwen3 长上下文 plaintext 继续预训练，默认 64K，上游推荐基础配置。 |
| `llama3_longct.yaml` | LLaMA-3.1 长上下文 plaintext 继续预训练模板，需要替换本地模型路径。 |
| `qwen3_4b_smoke.yaml` | Qwen3 小规模 smoke test，短序列、少量样本，用于快速验证训练链路。 |
| `qwen3_code_32k.yaml` | Qwen3 32K 代码继续预训练配置，使用 plaintext/mapping eval 数据。 |
| `qwen3_swa1024_ttt_aux.yaml` | Qwen3 SWA(1024) + TTT aux 基础配置。 |
| `qwen3_swa1024_ttt_aux_low_lr.yaml` | Qwen3 SWA(1024) + TTT aux 低学习率配置，可复用于 coding 或其它数据。 |
| `qwen3_swa3_full1_v0anchor_ttt_aux_c1024.yaml` | Qwen3 v0-anchor Full:SWA = 1:3 + TTT aux，TTT chunk 1024。 |
| `qwen3_swa3_full1_v0anchor_ttt_aux_c2048.yaml` | Qwen3 v0-anchor Full:SWA = 1:3 + TTT aux，TTT chunk 2048。 |
| `qwen3_swa3_full1_strict_ttt_aux.yaml` | SWA:Full = 3:1 周期布局；`0 full, 1/2/3 SWA` 循环，TTT 放在每组第一个 SWA 层。 |
| `qwen3_swa3_full1_v0anchor_halo_hidden_align.yaml` | halo hidden alignment 训练配置，aux loss 默认关闭。 |
| `qwen3_swa3_full1_v0anchor_halo_kd.yaml` | halo KD 蒸馏训练配置，aux loss 默认关闭。 |
| `qwen3-1.7b/stage1_hidden_align_swa3_full1_strict.yaml` | Qwen3-1.7B 严格 Full:SWA = 1:3 的 hidden-state 对齐阶段。 |
| `qwen3-1.7b/stage2_kd_swa3_full1_strict.yaml` | Qwen3-1.7B 严格 Full:SWA = 1:3 的 KD 对齐阶段。 |
| `qwen3-1.7b/stage3_cpt_swa3_full1_strict.yaml` | Qwen3-1.7B 严格 Full:SWA = 1:3 的 CPT 阶段。 |

## 核心字段

### `model`

- `model_path` / `tokenizer_path`：本地基础模型和 tokenizer 路径。
- `foundation`：传入自定义 HF config 的扩展字段，是 TTT/SWA/aux/monitor 的主要入口。
- `foundation.ttt_mode`：是否启用 TTT 层。
- `foundation.ttt_layers`：启用 TTT 的 decoder layer 下标。
- `foundation.ttt_proj`：是否使用 `ttt_proj` 生成 V-hat。
- `foundation.ttt_lr`：fast-weight 更新强度。
- `foundation.ttt_chunk`：TTT chunk 大小，直接影响 fast-weight 更新粒度和显存/速度。
- `foundation.ttt_target`：TTT target 来源，常见值为 `input_embed` 或 `hidden_states`。
- `foundation.ttt_compress_window`：大于 0 时启用 SWA 压缩窗口。
- `foundation.full_attention_layers`：在 SWA 压缩下保留 full attention 的 anchor 层。
- `foundation.ttt_aux_loss_weight`：TTT aux loss 权重，设为 0 表示关闭 aux loss。

### `data`

- `train_path` / `eval_path`：训练和验证数据路径。
- `data_type`：常见值为 `plaintext`、`conversation`、`pretokenized`。
- `datasets_type`：常见值为 `iterable` 或 `mapping`。
- `eval_datasets_type`：验证集类型；例如 `qwen3_code_32k.yaml` 使用 `mapping`。
- `max_seq_len`：训练序列长度，是长上下文实验最关键的数据侧约束。
- `text_keys`：输入文本字段名。
- `drop_last`：长上下文训练通常保持 `true`，避免 batch shape 不稳定。

### `train`

- `global_batch_size` / `micro_batch_size`：全局 batch 和 micro-batch。
- `max_steps` / `num_train_epochs`：训练步数和 epoch 控制。
- `eval_steps` / `eval_batches`：validation 触发频率和每次验证 batch 数。
- `save_steps` / `save_epochs`：checkpoint 保存频率。
- `enable_gradient_checkpointing`：长上下文训练通常开启以降低显存。
- `use_wandb`：是否写入 wandb 指标。

## TTT 监控指标

训练循环目前会收集 TTT layer 内部的 fast-weight 监控指标：

- `training/ttt_delta_weight_sample_ratio`：单个 chunk 的 `delta_down_proj` 采样范数，相对基础 `down_proj.weight` 采样范数的比例。
- `training/ttt_delta_weight_cumsum_sample_ratio`：沿 chunk 累积后的 fast-weight 采样范数比例。
- `training/ttt_output_delta_sample_ratio`：默认 `mlp` 模式下，TTT MLP 输出相对原始 `down_proj` 输出的采样变化比例。

采样控制字段：

```yaml
model:
  foundation:
    ttt_monitor_sample_dim: 64
    ttt_monitor_sample_tokens: 1
```

`ttt_monitor_sample_dim` 控制输出维度采样数，`ttt_monitor_sample_tokens` 控制 token 采样数。把任一字段设为 0 会关闭对应层内监控。

## Logits 级 TTT 影响指标

为了观察“最终 logits 受 TTT 开关影响的强度”，新增了 validation 阶段的 logits 级指标：

```yaml
model:
  foundation:
    ttt_monitor_output_delta_target: logits
    ttt_monitor_logit_sample_tokens: 1
    ttt_monitor_logit_sample_dim: 0
```

开启后，在 validation 中额外计算：

```text
||logits_ttt_on - logits_ttt_off|| / ||logits_ttt_on||
```

并上报为：

```text
eval/ttt_output_delta_sample_ratio
```

实现约定：

- 该指标只在 validation 中计算，不在训练 micro-batch 中计算，避免训练主路径每步额外跑两次 forward。
- `ttt_monitor_logit_sample_tokens` 表示取最后 N 个 token 的 logits。
- `ttt_monitor_logit_sample_dim=0` 表示使用全 vocab；设为正数时只取前 N 个 vocab 维度，用于降低 validation 开销。
- 只有配置了 `eval_path`、`eval_steps > 0`、`eval_batches > 0` 时才会触发该指标。
- 开启 `logits` 模式后，`training/ttt_output_delta_sample_ratio` 不再写入；最终 logits 影响强度以 `eval/ttt_output_delta_sample_ratio` 为准。

通用 shell 模板支持环境变量覆盖：

```bash
TTT_MONITOR_OUTPUT_DELTA_TARGET=logits
TTT_MONITOR_LOGIT_SAMPLE_TOKENS=1
TTT_MONITOR_LOGIT_SAMPLE_DIM=4096
```

## 配置选择建议

- 快速验证代码链路：优先使用 `qwen3_4b_smoke.yaml`。
- 复现基础长上下文 TTT：优先使用 `qwen3_longct.yaml` 或 `llama3_longct.yaml`。
- SWA(1024) + TTT aux：优先使用 `qwen3_swa1024_ttt_aux.yaml`。
- 需要 SWA/anchor 结构：使用 `qwen3_swa3_full1_v0anchor_ttt_aux_c1024.yaml` 或 `qwen3_swa3_full1_v0anchor_ttt_aux_c2048.yaml`。
- 需要严格周期布局：使用 `qwen3_swa3_full1_strict_ttt_aux.yaml`，SWA:Full = 3:1，full 层为 `0,4,8,...`，TTT 层为每组第一个 SWA 层 `1,5,9,...`。
- 只想观察最终 logits 影响强度：在目标配置中加入 `ttt_monitor_output_delta_target: logits`，并保证 validation 配置有效。
- Qwen3-1.7B 三阶段中，stage1/stage2 只训练 SWA/Full 结构，`ttt_mode: false` 且 `ttt_layers: []`；stage3 CPT 才启用 TTT。三阶段都显式设置 `lr_min`，cosine 退火不会降到 0。

## 开发注意事项

- `model.foundation` 中的字段必须在对应 config class 中定义，否则训练脚本会 fail fast，避免字段被 HF 静默丢弃。
- `ttt_chunk`、`ttt_compress_window`、`max_seq_len` 三者要一起看：chunk 影响 TTT 更新频率，compress window 影响 SWA 可见范围，max seq len 决定训练上下文长度。
- 不同实验配置中的绝对路径是本地环境路径，迁移机器时需要替换 `model_path`、`tokenizer_path`、`train_path`、`eval_path` 和 `output_dir`。
- 新增配置优先复用已有 YAML 的字段组合；只有实验目标明确需要时，再增加新的 `foundation` 字段。
