# LongSFT 启动脚本配置指南

本目录的固定实验脚本用于复现实验；新增实验建议使用 `run_longsft_template.sh`，通过环境变量覆盖配置，避免反复编辑脚本。

## 当前脚本

`run_swa_full_1to3_ttt_aux.sh` 当前固定使用：

- `CONFIG=configs/pretrain/qwen3_longsft_swa_full_1to3_ttt_aux.yaml`
- `WANDB_PROJECT=${WANDB_PROJECT:-in-place-ttt}`
- `WANDB_NAME=${WANDB_NAME:-longsft-swa-anchor-ttt-aux-swa4096-chunk2048-32k}`
- 单机默认 `MASTER_PORT=12361`，GPU 数默认来自 `MLP_GPU`，否则为 `8`

它已经支持这些环境变量覆盖：

| 变量 | 覆盖到的训练参数 |
| --- | --- |
| `WANDB_PROJECT` | `--train.wandb_project` |
| `WANDB_NAME` | `--train.wandb_name` |
| `OUTPUT_DIR` | `--train.output_dir` |
| `TRAIN_PATH` | `--data.train_path` |
| `EVAL_PATH` | `--data.eval_path` |
| `MAX_SEQ_LEN` | `--data.max_seq_len` |

脚本末尾的 `"$@"` 会把额外命令行参数继续传给 `tasks/train_torch.py`。没有内置环境变量的字段可以直接追加，例如：

```bash
bash "scripts/train/longsft/run_swa_full_1to3_ttt_aux.sh" \
  --train.max_steps 2000 \
  --train.lr 3.0e-5
```

## 推荐模板

通用模板路径：

```bash
bash "scripts/train/longsft/run_longsft_template.sh"
```

常用变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `CONFIG` | `configs/pretrain/qwen3_longsft_swa_full_1to3_ttt_aux.yaml` | YAML 配置路径 |
| `TASK_SCRIPT` | `tasks/train_torch.py` | 训练入口，KD 实验可改为 `tasks/train_torch_halo_kd.py`，隐状态对齐可改为 `tasks/train_torch_halo_hidden_align.py` |
| `EXP_NAME` | `CONFIG` 文件名去后缀 | 影响默认 `WANDB_NAME`、`OUTPUT_DIR`、日志前缀 |
| `OUTPUT_DIR` | `outputs/$EXP_NAME` | 训练输出目录 |
| `WANDB_PROJECT` | `in-place-ttt` | W&B project |
| `WANDB_NAME` | `$EXP_NAME` | W&B run name |
| `USE_WANDB` | 空 | 设置为 `false` 可覆盖 `--train.use_wandb false` |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | 可见 GPU |
| `NPROC_PER_NODE` | `MLP_GPU` 或可见 GPU 数 | `torchrun --nproc_per_node` |
| `NNODES` / `NODE_RANK` | `MLP_WORKER_NUM` / `MLP_ROLE_INDEX` | 多机训练参数 |
| `MASTER_ADDR` / `MASTER_PORT` | `MLP_WORKER_0_HOST` / `MLP_WORKER_0_PORT` | rendezvous 地址和端口 |
| `LOG_DIR` / `LOG_FILE` | `./logs` / 自动生成 | 日志输出 |

训练与数据覆盖：

| 变量 | 覆盖到的训练参数 |
| --- | --- |
| `TRAIN_PATH` | `--data.train_path` |
| `EVAL_PATH` | `--data.eval_path` |
| `MAX_SEQ_LEN` | `--data.max_seq_len` |
| `MAX_STEPS` | `--train.max_steps` |
| `SAVE_STEPS` | `--train.save_steps` |
| `EVAL_STEPS` | `--train.eval_steps` |
| `EVAL_BATCHES` | `--train.eval_batches` |
| `GLOBAL_BATCH_SIZE` | `--train.global_batch_size` |
| `MICRO_BATCH_SIZE` | `--train.micro_batch_size` |
| `LR` | `--train.lr` |
| `LOAD_CHECKPOINT_PATH` | `--train.load_checkpoint_path` |
| `HF_MODEL_DIR` | 同时覆盖 `--model.model_path`、`--model.config_path`、`--model.tokenizer_path` |
| `MODEL_PATH` | `--model.model_path` |
| `MODEL_CONFIG_PATH` | `--model.config_path` |
| `TOKENIZER_PATH` | `--model.tokenizer_path` |

TTT / foundation 覆盖：

| 变量 | 写入 `--model.foundation` 的 key |
| --- | --- |
| `TTT_CHUNK` | `ttt_chunk` |
| `TTT_COMPRESS_WINDOW` | `ttt_compress_window` |
| `TTT_LR` | `ttt_lr` |
| `TTT_AUX_LOSS_WEIGHT` | `ttt_aux_loss_weight` |
| `TTT_PARAM_LR_MULTIPLIER` | `ttt_param_lr_multiplier` |
| `TTT_PARAM_WEIGHT_DECAY` | `ttt_param_weight_decay` |
| `TTT_MODE` | `ttt_mode` |
| `TTT_LAYERS` | `ttt_layers` |
| `FULL_ATTENTION_LAYERS` | `full_attention_layers` |
| `MODEL_FOUNDATION_JSON` | 完整自定义 dict，优先级高于上述 TTT 变量 |

`MODEL_FOUNDATION_JSON` 会在 `tasks/train_torch.py` 中和 YAML 的 `model.foundation` 做 `dict.update()`，适合只覆盖少数字段。

## 示例

只改输出目录和 W&B 名称：

```bash
EXP_NAME="longsft-swa4096-chunk2048-debug" \
OUTPUT_DIR="outputs/longsft-swa4096-chunk2048-debug" \
WANDB_NAME="longsft-swa4096-chunk2048-debug" \
bash "scripts/train/longsft/run_longsft_template.sh"
```

切换 config、数据和序列长度：

```bash
CONFIG="configs/pretrain/qwen3_longsft_swa_full_1to3_ttt_aux.yaml" \
EXP_NAME="longsft-64k-swa-ttt" \
TRAIN_PATH="/path/to/train.jsonl" \
EVAL_PATH="/path/to/val.jsonl" \
MAX_SEQ_LEN="65536" \
OUTPUT_DIR="outputs/longsft-64k-swa-ttt" \
bash "scripts/train/longsft/run_longsft_template.sh"
```

覆盖 TTT chunk/window：

```bash
EXP_NAME="longsft-swa-w4096-c1024" \
TTT_COMPRESS_WINDOW="4096" \
TTT_CHUNK="1024" \
bash "scripts/train/longsft/run_longsft_template.sh"
```

传入完整 foundation dict：

```bash
MODEL_FOUNDATION_JSON='{"ttt_compress_window": 2048, "ttt_chunk": 1024, "ttt_aux_loss_weight": 0.05}' \
EXP_NAME="longsft-swa-w2048-c1024-aux005" \
bash "scripts/train/longsft/run_longsft_template.sh"
```

增强 TTT 模块训练强度：

```bash
EXP_NAME="longsft-swa-ttt-lr3x-nowd" \
TTT_PARAM_LR_MULTIPLIER="3.0" \
TTT_PARAM_WEIGHT_DECAY="0.0" \
TTT_AUX_LOSS_WEIGHT="0.2" \
bash "scripts/train/longsft/run_longsft_template.sh"
```

从已有 HF checkpoint 继续训练：

```bash
HF_MODEL_DIR="/path/to/hf_checkpoint" \
EXP_NAME="longsft-continue-from-hf" \
OUTPUT_DIR="outputs/longsft-continue-from-hf" \
LOAD_CHECKPOINT_PATH="" \
bash "scripts/train/longsft/run_longsft_template.sh"
```

HALO Stage-1 风格隐状态对齐：

```bash
TEACHER_MODEL_PATH="/path/to/full_attention_teacher" \
OUTPUT_DIR="outputs/longsft-hidden-align" \
bash "scripts/train/longsft/run_swa_full_1to3_halo_hidden_align.sh"
```

没有模板变量的字段直接追加 CLI 参数：

```bash
bash "scripts/train/longsft/run_longsft_template.sh" \
  --train.lr 3.0e-5 \
  --train.save_steps 500 \
  --train.eval_steps 50
```

不要同时使用 `MODEL_FOUNDATION_JSON` 和额外追加的 `--model.foundation`，因为训练入口只会预处理第一个 `--model.foundation`。
