#!/bin/bash
set -e

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
NPROC_PER_NODE=8 \
HF_HOME="/zouxiangyu/codes/Learning/In-Place-TTT/data/hf_home" \
bash "train.sh" \
  "tasks/train_torch.py" \
  "configs/pretrain/qwen3_code_32k.yaml" \
  --data.train_path "data/code_32k_20b/train_plaintext.jsonl" \
  --data.eval_path "data/code_32k_20b/eval_plaintext.jsonl" \
  --train.output_dir "outputs/qwen3-4b-code-32k" \
  2>&1 | tee -a "logs/train_qwen3_code_32k_20260528_001.log"