#!/bin/bash
# Full:SWA attention = 1:3 hybrid + In-Place TTT on SWA layers only.

set -x
set -o pipefail

CONFIG="configs/pretrain/qwen3_longsft_swa_full_1to3_ttt_aux.yaml"
WANDB_PROJECT="${WANDB_PROJECT:-in-place-ttt}"
WANDB_NAME="${WANDB_NAME:-longsft-full-swa-1to3-swa-ttt-aux-amp}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

NPROC_PER_NODE="${MLP_GPU:-8}"
NNODES="${MLP_WORKER_NUM:-1}"
NODE_RANK="${MLP_ROLE_INDEX:-0}"
MASTER_ADDR="${MLP_WORKER_0_HOST:-localhost}"
MASTER_PORT="${MLP_WORKER_0_PORT:-12361}"

if [[ "$NNODES" == "1" ]]; then
  additional_args="--standalone"
fi

LOG_FILE="./logs/log-longsft-full-swa-1to3-swa-ttt-aux-amp_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).txt"

torchrun \
  --nproc_per_node "$NPROC_PER_NODE" \
  --nnodes "$NNODES" \
  --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  $additional_args tasks/train_torch.py "$CONFIG" \
  --train.wandb_project "$WANDB_PROJECT" \
  --train.wandb_name "$WANDB_NAME" \
  "$@" 2>&1 | tee "$LOG_FILE"
