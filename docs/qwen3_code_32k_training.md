# Qwen3 Code 32K 训练记录

本文记录 `qwen3-4b-code-32k` 这次 32K 代码继续预训练的启动方式、数据规模、训练流程、关键代码和产物状态。

## 训练结论

上一次训练已成功完成。

验证依据：

- 训练日志：`logs/train_qwen3_code_32k_20260523_013049.log`
- 日志尾部显示训练进度完成到 `5000/5000`。
- 最终 DCP checkpoint 保存成功：`outputs/qwen3-4b-code-32k/checkpoints/global_step_5000`
- 最终 HuggingFace checkpoint 导出成功：`outputs/qwen3-4b-code-32k/checkpoints/global_step_5000/hf_ckpt`

当前 checkpoint 目录中保留：

- `global_step_500`
- `global_step_1500`
- `global_step_5000`

说明：当前目录未保留全部每 500 step 的中间 checkpoint，但最终 `global_step_5000` 的 DCP 与 HF 产物完整。

## 训练启动命令

实际训练使用下面命令启动：

```bash
CUDA_VISIBLE_DEVICES="2,3,4,5" \
NPROC_PER_NODE=4 \
HF_HOME="/zouxiangyu/codes/Learning/In-Place-TTT/data/hf_home" \
uvx uv==0.9.8 run bash "train.sh" \
  "tasks/train_torch.py" \
  "configs/pretrain/qwen3_code_32k.yaml" \
  --data.train_path "data/code_32k_20b/train_plaintext.jsonl" \
  --data.eval_path "data/code_32k_20b/eval_plaintext.jsonl" \
  --train.output_dir "outputs/qwen3-4b-code-32k" \
  2>&1 | tee -a "logs/train_qwen3_code_32k_20260523_013049.log"
```

复现要点：

- 必须通过 `uvx uv==0.9.8 run ...` 进入项目依赖环境。
- 直接执行 `bash train.sh ...` 可能使用系统 Python 或系统 `torchrun`，导致自定义模块如 `hf_models` 无法导入。
- 当前配置默认使用 split 后的 `train_plaintext.jsonl` 和 `eval_plaintext.jsonl`；启动前需要先运行 held-out split 脚本生成这两个文件。
- 当前要求不做 60 秒级轮询检查；如外部监控脚本需要定时检查训练状态，检查间隔应使用 1 小时，例如 `sleep 3600`。

## 训练数据

数据文件：

- `data/code_32k_20b/plaintext.jsonl`
- `data/code_32k_20b/train_plaintext.jsonl`
- `data/code_32k_20b/eval_plaintext.jsonl`
- `data/code_32k_20b/manifest.json`

数据规模来自 `manifest.json`：

| 指标 | 数值 |
| --- | ---: |
| 目标 token | 20,000,000,000 |
| 实际 token | 20,000,000,364 |
| 实际规模 | 20.000000364B token |
| 样本数 | 13,549,989 |
| 扫描行数 | 56,571,299 |
| 写入行数 | 13,549,989 |
| 字符数 | 83,214,723,487 |

语言 token 分布：

| 语言族 | token |
| --- | ---: |
| Java / JVM | 7,903,074,556 |
| TypeScript / JavaScript | 6,471,464,484 |
| Python | 3,529,545,892 |
| Go | 1,888,627,494 |
| Rust | 207,287,938 |

许可证 token 分布：

| 许可证 | token |
| --- | ---: |
| Apache-2.0 | 9,900,296,062 |
| MIT | 7,985,250,592 |
| BSD-3-Clause | 1,694,312,270 |
| BSD-2-Clause | 333,955,440 |
| ISC | 86,186,000 |

## 数据构建流程

入口脚本：

- `scripts/build_github_code_20b_data.py`

数据源：

- `codeparrot/github-code`

核心策略：

- 使用 streaming 方式读取 HuggingFace dataset，避免一次性落盘全量源数据。
- 默认 tokenizer 使用 `/zouxiangyu/models/Qwen/Qwen3-4B`，token 统计与训练模型保持一致。
- 过滤语言：Python、JavaScript、TypeScript、Go、Rust、Java。
- 过滤许可证：MIT、Apache-2.0、BSD-3-Clause、BSD-2-Clause、ISC。
- 过滤常见无效目录：`.git`、`.venv`、`node_modules`、`dist`、`build`、cache 目录等。
- 支持 `--resume`，根据 `manifest.json` 跳过已处理行。
- 周期性 flush、fsync 并更新 manifest，降低长时间构建中断后的恢复成本。

每行 JSONL 的训练字段：

```json
{
  "content_split": "<repo name=\"...\" task=\"file_completion\">\n<file path=\"...\">\n...\n</file>\n</repo>",
  "metadata": {
    "repo_name": "...",
    "path": "...",
    "language": "...",
    "license": "...",
    "tokens": 123
  }
}
```

训练实际读取 `content_split` 字段。

### Held-out eval 切分

周期性 eval loss 使用预先切出的 held-out JSONL，避免直接在训练样本上评估。切分脚本：

```bash
uvx uv==0.9.8 run python "scripts/split_code_32k_eval.py" \
  --input "data/code_32k_20b/plaintext.jsonl" \
  --train-output "data/code_32k_20b/train_plaintext.jsonl" \
  --eval-output "data/code_32k_20b/eval_plaintext.jsonl" \
  --manifest "data/code_32k_20b/split_manifest.json" \
  --eval-examples 20000 \
  --eval-stride 500
```

脚本按固定 stride 抽取 eval 行，并将这些行从 `train_plaintext.jsonl` 中排除。默认不覆盖已有输出；如需重建，需要显式追加 `--overwrite`。

## 训练配置摘要

配置文件：

- `configs/pretrain/qwen3_code_32k.yaml`

核心参数：

| 配置项 | 值 |
| --- | --- |
| base model | `/zouxiangyu/models/Qwen/Qwen3-4B` |
| tokenizer | `/zouxiangyu/models/Qwen/Qwen3-4B` |
| max sequence length | `32768` |
| data type | `plaintext` |
| text key | `content_split` |
| eval path | `data/code_32k_20b/eval_plaintext.jsonl` |
| eval steps | `500` |
| eval batches | `8` |
| max steps | `5000` |
| global batch size | `64` |
| micro batch size | `1` |
| parallel mode | `fsdp2` |
| optimizer | `adamw` |
| learning rate | `5.0e-6` |
| warmup ratio | `0.02` |
| lr decay | cosine, decay ratio `0.90` |
| weight decay | `0.1` |
| max grad norm | `1.0` |
| mixed precision | enabled |
| gradient checkpointing | enabled |
| full shard | enabled |
| checkpoint manager | `dcp` |
| save steps | `500` |
| save HF weights | enabled |

TTT 相关参数：

| 配置项 | 值 |
| --- | --- |
| `ttt_mode` | `true` |
| `ttt_layers` | `[0, 6, 12, 18, 24, 30, 35]` |
| `ttt_proj` | `true` |
| `ttt_lr` | `3` |
| `ttt_chunk` | `1024` |

## 训练流程

整体链路：

1. `train.sh` 设置运行环境变量，并根据 `CUDA_VISIBLE_DEVICES` 推断 `NPROC_PER_NODE`。
2. `train.sh` 调用 `torchrun`，单机训练时追加 `--standalone`。
3. `tasks/train_torch.py` 设置 `MODELING_BACKEND=hf` 并导入 `hf_models`，注册自定义模型实现。
4. 使用 VeOmni 构建 tokenizer、dataset 和 dataloader。
5. plaintext 数据通过 `process_pretrain_example` 读取 `content_split` 字段并 tokenize。
6. `build_foundation_model` 按配置加载 Qwen3-4B 和 TTT 参数。
7. `build_parallelize_model` 启用 FSDP2、mixed precision、gradient checkpointing、full shard。
8. 构建 AdamW optimizer 和 cosine learning-rate scheduler。
9. 训练循环按 micro batch 执行 forward、backward、梯度裁剪、optimizer step、scheduler step 和 wandb log。
10. 每 `eval_steps=500` 在 held-out eval 数据上跑 `eval_batches=8` 个 batch，记录 `eval/loss`。
11. 每 `save_steps=500` 保存 DCP checkpoint。
12. 训练结束后，如果 `save_hf_weights=true`，从 DCP checkpoint 转出 HuggingFace `safetensors` 权重。

## 关键代码

`train.sh`

- 设置 `TOKENIZERS_PARALLELISM=false`。
- 设置 `TORCH_NCCL_AVOID_RECORD_STREAMS=1`。
- 设置默认 `WANDB_BASE_URL=https://wandb.glm.ai`。
- 根据 GPU / NPU 可见设备推断并行进程数。
- 单机时使用 `torchrun --standalone`。
- 将训练输出同时写入 `log.txt`。

`tasks/train_torch.py`

- 第 28 行设置 `MODELING_BACKEND=hf`。
- 第 36 行导入 `hf_models`，确保自定义模型注册生效。
- 定义 `EvalDataArguments` 和 `EvalTrainingArguments`，为训练配置扩展 `eval_path`、`eval_datasets_type`、`eval_steps` 和 `eval_batches`。
- 构建训练 dataloader 后，如果 eval 配置启用，则构建独立 eval dataloader。
- 训练中通过 `_run_eval_loss()` 执行 `model.eval()` + `torch.no_grad()` 的小样本评估，并将结果写入 `eval/loss`。
- 保存 DCP checkpoint，训练结束后导出 HuggingFace checkpoint。

`tasks/eval_control.py`

- `should_run_eval()` 负责判断当前 global step 是否需要触发 eval。

`scripts/split_code_32k_eval.py`

- 从原始 `plaintext.jsonl` 中按固定 stride 切出 held-out eval 文件。
- 同时生成排除 eval 行的 train 文件，保证训练和 eval 不重叠。
- 写出 `split_manifest.json`，记录输入、输出、stride、eval 行数和 train 行数。

`scripts/build_github_code_20b_data.py`

- 定义默认数据源、语言和许可证范围。
- `accept_row()` 负责过滤语言、许可证、长度、路径和内容。
- `build_content()` 将单文件代码包装成 repo/file completion 格式。
- JSONL 输出字段是 `content_split` 与 `metadata`。
- `write_manifest()` 持续记录 token、样本、跳过原因、语言分布和许可证分布。

`configs/pretrain/qwen3_code_32k.yaml`

- 固化模型、tokenizer、32K 上下文、TTT 参数、FSDP2 参数、优化器参数和保存策略。
- 数据路径可被启动命令覆盖。

## 训练产物

最终 HF checkpoint 目录：

```text
outputs/qwen3-4b-code-32k/checkpoints/global_step_5000/hf_ckpt
```

当前文件：

```text
added_tokens.json
chat_template.jinja
config.json
merges.txt
model-00001-of-00002.safetensors
model-00002-of-00002.safetensors
model.safetensors.index.json
special_tokens_map.json
tokenizer_config.json
tokenizer.json
vocab.json
```

两个权重分片大小：

- `model-00001-of-00002.safetensors`: 4,969,939,984 bytes
- `model-00002-of-00002.safetensors`: 3,944,885,704 bytes

## 后续复现检查

复现或继续训练前建议检查：

```bash
tail -n 80 "logs/train_qwen3_code_32k_20260523_013049.log"
find "outputs/qwen3-4b-code-32k/checkpoints/global_step_5000/hf_ckpt" -maxdepth 1 -type f -printf "%f %s bytes\n" | sort
sed -n '1,220p' "data/code_32k_20b/manifest.json"
```

如果要继续从 checkpoint 恢复训练，应优先使用 DCP checkpoint：

```text
outputs/qwen3-4b-code-32k/checkpoints/global_step_5000
```

如果要做推理、评测或权重分发，应优先使用 HF checkpoint：

```text
outputs/qwen3-4b-code-32k/checkpoints/global_step_5000/hf_ckpt
```
