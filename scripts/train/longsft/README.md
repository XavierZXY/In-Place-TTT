# 训练启动脚本说明

本目录是历史路径，当前脚本已经按“配置驱动 + 数据可替换”的方式组织。数据集语义应放在 YAML 或环境变量里，启动脚本只保留架构、任务入口和少量实验默认值。

核心原则：

- `run_pretrain_template.sh` 是推荐的通用启动入口。
- 其它 `run_*.sh` 只作为薄 wrapper，负责设置默认 `CONFIG`、`TASK_SCRIPT`、`EXP_NAME`、`MASTER_PORT` 和少量实验变量。
- 不建议在 wrapper 中重复 torchrun、日志、分布式环境和 CLI override 逻辑。

## 推荐入口

通用模板：

```bash
bash "scripts/train/longsft/run_pretrain_template.sh"
```

常用架构 wrapper：

| 脚本 | 默认用途 |
| --- | --- |
| `run_swa1024_ttt_aux.sh` | Qwen3 SWA(1024) + TTT aux。 |
| `run_swa3_full1_v0anchor_ttt_aux_c1024.sh` | Qwen3 v0-anchor Full:SWA = 1:3 + TTT aux，chunk 1024。 |
| `run_swa3_full1_v0anchor_ttt_aux_c2048.sh` | Qwen3 v0-anchor Full:SWA = 1:3 + TTT aux，chunk 2048。 |
| `run_swa3_full1_v0anchor_hidden_chunk_aux.sh` | Qwen3 Full:SWA = 1:3 hybrid + hidden-state TTT + future-chunk hidden aux。 |
| `run_swa3_full1_v0anchor_halo_hidden_align.sh` | HALO hidden alignment 入口。 |
| `run_swa3_full1_v0anchor_halo_kd.sh` | HALO KD 入口。 |
| `run_swa3_full1_v0anchor_window_anneal.sh` | 多阶段 SWA window annealing。 |
| `qwen3-1.7b/run_stage1_hidden_align_swa3_full1_strict.sh` | Qwen3-1.7B strict hidden-state 对齐阶段。 |
| `qwen3-1.7b/run_stage2_kd_swa3_full1_strict.sh` | Qwen3-1.7B strict KD 对齐阶段。 |
| `qwen3-1.7b/run_stage3_cpt_swa3_full1_strict.sh` | Qwen3-1.7B strict CPT 阶段。 |

Qwen3-1.7B 三阶段脚本默认按顺序衔接：stage2 从 stage1 的 `global_step_8000` 加载，stage3 从 stage2 的 `global_step_8000` 加载。stage1/stage2 不实例化 TTT 模块，只保留 SWA/Full 布局；stage3 CPT 才启用 TTT。实际 checkpoint 不同时，覆盖 `LOAD_CHECKPOINT_PATH`、`STAGE1_GLOBAL_STEP` 或 `STAGE2_GLOBAL_STEP`。

所有 wrapper 都会调用 `run_pretrain_template.sh`，所以通用环境变量和额外 CLI 参数行为一致。

## 通用变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `CONFIG` | wrapper 指定；模板默认 `configs/pretrain/qwen3_swa3_full1_v0anchor_ttt_aux_c2048.yaml` | YAML 配置路径。 |
| `TASK_SCRIPT` | `tasks/train_torch.py` | 训练入口；HALO wrapper 会覆盖为对应 task。 |
| `EXP_NAME` | `CONFIG` 文件名去后缀 | 影响默认 `WANDB_NAME`、`OUTPUT_DIR` 和日志前缀。 |
| `OUTPUT_DIR` | `outputs/$EXP_NAME` | 训练输出目录。 |
| `WANDB_PROJECT` | `in-place-ttt` | W&B project。 |
| `WANDB_NAME` | `$EXP_NAME` | W&B run name。 |
| `USE_WANDB` | 空 | 覆盖 `--train.use_wandb`。 |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | 可见 GPU。 |
| `NPROC_PER_NODE` | `MLP_GPU` 或可见 GPU 数 | `torchrun --nproc_per_node`。 |
| `NNODES` / `NODE_RANK` | `MLP_WORKER_NUM` / `MLP_ROLE_INDEX` | 多机训练参数。 |
| `MASTER_ADDR` / `MASTER_PORT` | `MLP_WORKER_0_HOST` / `MLP_WORKER_0_PORT` | rendezvous 地址和端口。 |
| `TORCHRUN` | `.venv/bin/torchrun` 或 PATH 中的 `torchrun` | torchrun 可执行文件。 |
| `LOG_DIR` / `LOG_FILE` | `./logs` / 自动生成 | 日志输出。 |
| `TRACE` | `1` | 设为 `0` 可关闭 `set -x`。 |

## 数据与训练覆盖

| 变量 | 覆盖到的参数 |
| --- | --- |
| `TRAIN_PATH` | `--data.train_path` |
| `EVAL_PATH` | `--data.eval_path` |
| `DATALOADER_TYPE` | `--data.dataloader_type` |
| `DATASETS_TYPE` | `--data.datasets_type` |
| `EVAL_DATASETS_TYPE` | `--data.eval_datasets_type` |
| `DATA_TYPE` | `--data.data_type` |
| `CHAT_TEMPLATE` | `--data.chat_template` |
| `TEXT_KEYS` | `--data.text_keys` |
| `TRAIN_SIZE` | `--data.train_size` |
| `MAX_SEQ_LEN` | `--data.max_seq_len` |
| `MAX_STEPS` | `--train.max_steps` |
| `SAVE_STEPS` | `--train.save_steps` |
| `EVAL_STEPS` | `--train.eval_steps` |
| `EVAL_BATCHES` | `--train.eval_batches` |
| `GLOBAL_BATCH_SIZE` | `--train.global_batch_size` |
| `MICRO_BATCH_SIZE` | `--train.micro_batch_size` |
| `LR` | `--train.lr` |
| `LR_MIN` | `--train.lr_min` |
| `LR_WARMUP_RATIO` | `--train.lr_warmup_ratio` |
| `LR_DECAY_STYLE` | `--train.lr_decay_style` |
| `LR_DECAY_RATIO` | `--train.lr_decay_ratio` |
| `WEIGHT_DECAY` | `--train.weight_decay` |
| `MAX_GRAD_NORM` | `--train.max_grad_norm` |
| `LOAD_CHECKPOINT_PATH` | `--train.load_checkpoint_path`；可显式设为空字符串禁用 resume。 |
| `HF_MODEL_DIR` | 同时覆盖 `model_path`、`config_path`、`tokenizer_path`。 |
| `MODEL_PATH` | `--model.model_path` |
| `MODEL_CONFIG_PATH` | `--model.config_path` |
| `TOKENIZER_PATH` | `--model.tokenizer_path` |

## Foundation 覆盖

这些变量会组装成一次 `--model.foundation`：

| 变量 | key |
| --- | --- |
| `TTT_CHUNK` | `ttt_chunk` |
| `TTT_COMPRESS_WINDOW` | `ttt_compress_window` |
| `TTT_LR` | `ttt_lr` |
| `TTT_AUX_LOSS_WEIGHT` | `ttt_aux_loss_weight` |
| `TTT_TARGET` | `ttt_target` |
| `TTT_AUX_TARGET` | `ttt_aux_target` |
| `TTT_AUX_FUTURE_CHUNKS` | `ttt_aux_future_chunks` |
| `TTT_AUX_LOSS_TYPE` | `ttt_aux_loss_type` |
| `TTT_JEPA_LOSS_EXP` | `ttt_jepa_loss_exp` |
| `TTT_JEPA_REG_COEFF` | `ttt_jepa_reg_coeff` |
| `TTT_MONITOR_SAMPLE_DIM` | `ttt_monitor_sample_dim` |
| `TTT_MONITOR_SAMPLE_TOKENS` | `ttt_monitor_sample_tokens` |
| `TTT_MONITOR_OUTPUT_DELTA_TARGET` | `ttt_monitor_output_delta_target` |
| `TTT_MONITOR_LOGIT_SAMPLE_TOKENS` | `ttt_monitor_logit_sample_tokens` |
| `TTT_MONITOR_LOGIT_SAMPLE_DIM` | `ttt_monitor_logit_sample_dim` |
| `TTT_PARAM_LR_MULTIPLIER` | `ttt_param_lr_multiplier` |
| `TTT_PARAM_WEIGHT_DECAY` | `ttt_param_weight_decay` |
| `TTT_TRAIN_ONLY` | `ttt_train_only` |
| `TTT_MODE` | `ttt_mode` |
| `TTT_LAYERS` | `ttt_layers` |
| `FULL_ATTENTION_LAYERS` | `full_attention_layers` |
| `DISTILL_TEACHER_PATH` | `distill_teacher_path` |
| `HIDDEN_ALIGN_TEACHER_PATH` | `hidden_align_teacher_path` |
| `HIDDEN_ALIGN_LOSS_FN` | `hidden_align_loss_fn` |
| `HIDDEN_ALIGN_LAYERS` | `hidden_align_layers` |
| `HIDDEN_ALIGN_SKIP_TTT_LAYERS` | `hidden_align_skip_ttt_layers` |
| `HIDDEN_ALIGN_TRAIN_SCOPE` | `hidden_align_train_scope` |

`MODEL_FOUNDATION_JSON` 可以传入完整 dict，优先级高于上述变量：

```bash
MODEL_FOUNDATION_JSON='{"ttt_compress_window": 2048, "ttt_chunk": 1024}' \
bash "scripts/train/longsft/run_pretrain_template.sh"
```

不要同时使用 `MODEL_FOUNDATION_JSON` 和额外追加的 `--model.foundation`，训练入口只会预处理第一个 `--model.foundation`。

## 示例

切换数据但复用同一架构：

```bash
CONFIG="configs/pretrain/qwen3_swa3_full1_strict_ttt_aux.yaml" \
EXP_NAME="qwen3-swa3-full1-mydata" \
TRAIN_PATH="/path/to/train.jsonl" \
EVAL_PATH="/path/to/val.jsonl" \
DATA_TYPE="conversation" \
TEXT_KEYS="messages" \
MAX_SEQ_LEN="32768" \
bash "scripts/train/longsft/run_pretrain_template.sh"
```

只改 TTT chunk/window：

```bash
EXP_NAME="qwen3-swa4096-chunk1024" \
TTT_COMPRESS_WINDOW="4096" \
TTT_CHUNK="1024" \
bash "scripts/train/longsft/run_pretrain_template.sh"
```

开启 validation-only logits 级 TTT 影响指标：

```bash
TTT_MONITOR_OUTPUT_DELTA_TARGET="logits" \
TTT_MONITOR_LOGIT_SAMPLE_TOKENS="1" \
TTT_MONITOR_LOGIT_SAMPLE_DIM="4096" \
bash "scripts/train/longsft/run_pretrain_template.sh"
```

HALO KD teacher 覆盖：

```bash
TEACHER_MODEL_PATH="/path/to/full_attention_teacher" \
bash "scripts/train/longsft/run_swa3_full1_v0anchor_halo_kd.sh"
```

没有模板变量的字段直接追加 CLI 参数：

```bash
bash "scripts/train/longsft/run_pretrain_template.sh" \
  --train.lr 3.0e-5 \
  --train.save_steps 500 \
  --train.eval_steps 50
```
