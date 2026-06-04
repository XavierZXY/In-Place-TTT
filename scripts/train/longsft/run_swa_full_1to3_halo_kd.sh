#!/bin/bash
# HALO Stage-2 style KD for Full:SWA = 1:3 hybrid + TTT on SWA layers.

set -x
set -o pipefail

CONFIG="configs/pretrain/qwen3_longsft_swa_full_1to3_halo_kd.yaml"
WANDB_PROJECT="${WANDB_PROJECT:-in-place-ttt}"
WANDB_NAME="${WANDB_NAME:-longsft-swa-full-halo-1to3-kd-swa4096-chunk1024-4k}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

TORCHRUN="${TORCHRUN:-$REPO_ROOT/.venv/bin/torchrun}"
if [[ ! -x "$TORCHRUN" ]]; then
  TORCHRUN="$(command -v torchrun)"
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-${MLP_GPU:-8}}"
NNODES="${MLP_WORKER_NUM:-1}"
NODE_RANK="${MLP_ROLE_INDEX:-0}"
MASTER_ADDR="${MLP_WORKER_0_HOST:-localhost}"
MASTER_PORT="${MLP_WORKER_0_PORT:-12363}"

if [[ "$NNODES" == "1" ]]; then
  additional_args="--standalone"
fi

runtime_args=(
  --train.wandb_project "$WANDB_PROJECT"
  --train.wandb_name "$WANDB_NAME"
)

if [[ -n "${OUTPUT_DIR:-}" ]]; then
  runtime_args+=(--train.output_dir "$OUTPUT_DIR")
fi
if [[ -n "${TRAIN_PATH:-}" ]]; then
  runtime_args+=(--data.train_path "$TRAIN_PATH")
fi
if [[ -n "${EVAL_PATH:-}" ]]; then
  runtime_args+=(--data.eval_path "$EVAL_PATH")
fi
if [[ -n "${MAX_SEQ_LEN:-}" ]]; then
  runtime_args+=(--data.max_seq_len "$MAX_SEQ_LEN")
fi
if [[ -n "${MAX_STEPS:-}" ]]; then
  runtime_args+=(--train.max_steps "$MAX_STEPS")
fi
if [[ -n "${TEACHER_MODEL_PATH:-}" ]]; then
  runtime_args+=(--model.foundation "{\"distill_teacher_path\": \"${TEACHER_MODEL_PATH}\"}")
fi

mkdir -p "./logs"
LOG_FILE="./logs/log-longsft-swa-full-1to3-halo-kd_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).txt"

"$TORCHRUN" \
  --nproc_per_node "$NPROC_PER_NODE" \
  --nnodes "$NNODES" \
  --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  $additional_args tasks/train_torch_halo_kd.py "$CONFIG" \
  "${runtime_args[@]}" \
  "$@" 2>&1 | tee "$LOG_FILE"
