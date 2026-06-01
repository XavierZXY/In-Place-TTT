#!/bin/bash
# Progressive SWA window annealing for Full:SWA = 1:3 + TTT on SWA layers.

set -x
set -e
set -o pipefail

CONFIG="configs/pretrain/qwen3_longsft_swa_full_1to3_ttt_aux.yaml"
WANDB_PROJECT="${WANDB_PROJECT:-in-place-ttt}"
WANDB_NAME="${WANDB_NAME:-longsft-full-swa-1to3-swa-ttt-progressive-window-anneal}"
TOTAL_STEPS="${TOTAL_STEPS:-4000}"
WINDOW_SCHEDULE="${WINDOW_SCHEDULE:-4096,2048,1024}"
STAGE_STOP_STEPS="${STAGE_STOP_STEPS:-400,800}"
OUTPUT_BASE="${OUTPUT_BASE:-/zouxiangyu/codes/Learning/In-Place-TTT/outputs/longsft-full-swa-1to3-swa-ttt-progressive-window-anneal}"

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

IFS=',' read -r -a WINDOWS <<< "$WINDOW_SCHEDULE"
IFS=',' read -r -a STOP_STEPS <<< "$STAGE_STOP_STEPS"

if (( ${#WINDOWS[@]} < 2 )); then
  echo "WINDOW_SCHEDULE must contain at least two comma-separated window sizes." >&2
  exit 1
fi

if (( ${#STOP_STEPS[@]} != ${#WINDOWS[@]} - 1 )); then
  echo "STAGE_STOP_STEPS must contain exactly one fewer entries than WINDOW_SCHEDULE." >&2
  echo "WINDOW_SCHEDULE=$WINDOW_SCHEDULE" >&2
  echo "STAGE_STOP_STEPS=$STAGE_STOP_STEPS" >&2
  exit 1
fi

mkdir -p "./logs"

run_stage() {
  local stage_name="$1"
  local window_size="$2"
  local output_dir="$3"
  local log_file="$4"
  shift 4

  torchrun \
    --nproc_per_node "$NPROC_PER_NODE" \
    --nnodes "$NNODES" \
    --node_rank "$NODE_RANK" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    $additional_args tasks/train_torch.py "$CONFIG" \
    --model.foundation "{\"ttt_compress_window\": ${window_size}}" \
    --train.output_dir "$output_dir" \
    --train.max_steps "$TOTAL_STEPS" \
    --train.wandb_project "$WANDB_PROJECT" \
    --train.wandb_name "${WANDB_NAME}-${stage_name}-w${window_size}" \
    "$@" 2>&1 | tee "$log_file"
}

previous_ckpt=""
last_stage_index=$((${#WINDOWS[@]} - 1))

for stage_index in "${!WINDOWS[@]}"; do
  stage_num=$((stage_index + 1))
  window_size="${WINDOWS[$stage_index]}"
  stage_name="stage${stage_num}"
  output_dir="${OUTPUT_BASE}/${stage_name}-w${window_size}"
  log_file="./logs/log-${WANDB_NAME}-${stage_name}-w${window_size}_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).txt"

  load_args=(--train.load_checkpoint_path "auto")
  if [[ -n "$previous_ckpt" ]]; then
    load_args=(--train.load_checkpoint_path "$previous_ckpt")
  fi

  stop_args=()
  if (( stage_index < last_stage_index )); then
    stop_step="${STOP_STEPS[$stage_index]}"
    stop_args=(--train.stage_stop_steps "$stop_step")
  fi

  run_stage \
    "$stage_name" \
    "$window_size" \
    "$output_dir" \
    "$log_file" \
    "${load_args[@]}" \
    "${stop_args[@]}" \
    "$@"

  if (( stage_index < last_stage_index )); then
    previous_ckpt="${output_dir}/checkpoints/global_step_${stop_step}"
    if [[ ! -d "$previous_ckpt" ]]; then
      echo "Stage checkpoint not found: $previous_ckpt" >&2
      exit 1
    fi
  fi
done
