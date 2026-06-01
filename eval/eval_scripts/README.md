# 本地 RULER Eval 流程

这个目录提供一个不依赖 OpenCompass 的 RULER 评测流程，参考了
`/zouxiangyu/codes/TTT/In-Place-TTT-v0/eval_scripts`，但路径和入口已改成当前仓库可直接使用的形式。

## 文件职责

| 文件 | 作用 |
| --- | --- |
| `eval_ruler.py` | 加载 HF checkpoint，读取本地 RULER 数据，逐样本生成并打分。 |
| `run_ruler.sh` | Shell 包装器，统一处理 GPU、日志、常用参数和环境变量。 |
| `convert_dcp_to_hf.sh` | 将 VeOmni DCP checkpoint 转成 HuggingFace checkpoint。 |

当前仓库原有的 `eval.sh` 仍是 OpenCompass 流程；本目录是更接近 v0 参考实现的本地 HF 评测流程。

## 0. 准备 RULER 数据

评测脚本默认读取：

```bash
eval/eval_scripts/ruler/<length>/data/<task>/validation.jsonl
```

如果已经有参考仓库的数据，可以直接指定路径：

```bash
RULER_ROOT="/zouxiangyu/codes/TTT/In-Place-TTT-v0/eval_scripts/ruler" \
bash eval/eval_scripts/run_ruler.sh /path/to/hf_ckpt my-run 0 10
```

也可以建立软链接：

```bash
ln -s /zouxiangyu/codes/TTT/In-Place-TTT-v0/eval_scripts/ruler eval/eval_scripts/ruler
```

## 1. 转换 checkpoint

如果训练输出已经是 HF 格式，可以跳过这步。如果是 VeOmni DCP checkpoint：

```bash
bash eval/eval_scripts/convert_dcp_to_hf.sh /path/to/model_root 4000
```

默认读取：

```text
/path/to/model_root/checkpoints/global_step_4000/
```

默认写入：

```text
/path/to/model_root/checkpoints/global_step_4000/hf_ckpt/
```

可选环境变量：

```bash
MODEL_ASSETS_DIR=/path/to/base_or_assets \
SAVE_DIR=/path/to/save_hf_ckpt \
SHARD_SIZE=5000000000 \
bash eval/eval_scripts/convert_dcp_to_hf.sh /path/to/model_root 4000
```

如果目标目录已存在，脚本会跳过；显式传 `--force` 才会覆盖。

## 2. 运行 RULER

最小 smoke run：

```bash
RULER_ROOT="/zouxiangyu/codes/TTT/In-Place-TTT-v0/eval_scripts/ruler" \
LENGTHS="4096" \
bash eval/eval_scripts/run_ruler.sh /path/to/hf_ckpt smoke-qwen3 0 1
```

常规 4K/8K/16K：

```bash
RULER_ROOT="/zouxiangyu/codes/TTT/In-Place-TTT-v0/eval_scripts/ruler" \
LENGTHS="4096 8192 16384" \
MAX_SEQ=32768 \
bash eval/eval_scripts/run_ruler.sh /path/to/hf_ckpt qwen3-step4000 0 50
```

参数含义：

| 参数 | 说明 |
| --- | --- |
| `/path/to/hf_ckpt` | HF checkpoint 目录，必须包含 `config.json`、tokenizer 和权重文件。 |
| `qwen3-step4000` | 结果目录和日志使用的短名称。 |
| `0` | 使用的 GPU id，会写入 `CUDA_VISIBLE_DEVICES`。 |
| `50` | 每个 RULER task 取多少条样本。 |

## 3. 常用环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PYTHON` | 优先 `.venv/bin/python`，否则 `python3` | 指定 Python 解释器。 |
| `RULER_ROOT` | `eval/eval_scripts/ruler` | RULER 数据目录。 |
| `LENGTHS` | `4096 8192 16384` | 要评测的上下文长度。 |
| `MAX_SEQ` | `32768` | tokenizer 输入截断上限加生成上限。 |
| `MAX_NEW_TOKENS` | `1024` | 每条样本最大生成 token 数。 |
| `DTYPE` | `bfloat16` | 模型加载 dtype，可设 `auto`、`float16`、`float32`。 |
| `ATTN_IMPLEMENTATION` | `sdpa` | 传给 `from_pretrained` 的 attention 实现。 |
| `EOS_IDS` | 自动推断 | 手动指定 EOS token id，例如 `151643 151645`。 |
| `EXP_ROOT` | `eval/exp_analysis` | 实验分析根目录，`OUT_ROOT` 和 `LOG_ROOT` 默认基于它展开。 |
| `OUT_ROOT` | `eval/exp_analysis/ruler_results` | 结果输出根目录。 |
| `LOG_ROOT` | `eval/exp_analysis/logs` | 统一日志根目录。 |
| `LOG_DIR` | `eval/exp_analysis/logs/ruler/<abbr>` | 当前 RULER 实验日志目录。 |
| `LOG_FILE` | `eval/exp_analysis/logs/ruler/<abbr>/run.log` | 当前运行的完整日志文件。 |

Thinking / Chat 模型常用参数：

```bash
CHAT_TEMPLATE=1 \
STRIP_THINK=1 \
MAX_NEW_TOKENS=4096 \
STOP_STRINGS=$'\n\nQuestion:;\n\nAnswer:' \
bash eval/eval_scripts/run_ruler.sh /path/to/hf_ckpt sft-no-think 0 50
```

说明：

- `CHAT_TEMPLATE=1` 会调用 tokenizer 自带 chat template。
- `STRIP_THINK=1` 会在打分前去掉 `<think>...</think>`。
- `STOP_STRINGS` 使用分号分隔多个停止字符串。

## 4. 输出结构

```text
eval/exp_analysis/
  logs/
    ruler/
      <abbr>/
        run.log
  ruler_results/<abbr>/
    len_4096_samples.jsonl
    len_4096_summary.json
    len_8192_samples.jsonl
    len_8192_summary.json
    len_16384_samples.jsonl
    len_16384_summary.json
    summary_all_lengths.json
```

旧版本直接落在 `eval/exp_analysis/ruler_<abbr>.log` 的日志不会自动迁移；新运行统一写入 `logs/`。

`*_samples.jsonl` 保存逐样本的生成、答案和分数；`*_summary.json` 保存每个 task 的平均分和整体分数。
如果中断后重跑，同一个输出目录会按已写入的样本数继续。

## 5. 打分规则

| Task | 规则 |
| --- | --- |
| `niah_single_*`、`niah_multikey_*` | 忽略大小写的答案 substring 命中。 |
| `qa_1`、`qa_2` | 忽略大小写的答案 substring 命中。 |
| `niah_multiquery`、`niah_multivalue`、`vt`、`cwe`、`fwe` | gold list overlap，使用词边界匹配降低误命中。 |

生成使用 greedy decoding：`do_sample=False`、`num_beams=1`。

## 6. 实现要点

- 模型加载走当前仓库的 `inference_model/`，支持 Qwen3 和 LLaMA 的 TTT-aware forward/cache。
- 默认 batch size 为 1；可通过 `BATCH_SIZE` 或 `--batch_size` 启用批量 greedy decode。
- 支持 repetition-loop early stop，并会裁掉重复尾部，避免长文本生成陷入重复。
- 不写死 conda 路径；默认优先使用当前仓库 `.venv/bin/python`。
