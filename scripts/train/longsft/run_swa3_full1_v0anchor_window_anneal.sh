#!/bin/bash
# Progressive SWA window annealing for Full:SWA = 1:3 + TTT on SWA layers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CONFIG="${CONFIG:-configs/pretrain/qwen3_swa3_full1_v0anchor_ttt_aux_c2048.yaml}"
EXP_NAME="${EXP_NAME:-qwen3-swa3-full1-v0anchor-window-anneal}"
WANDB_PROJECT="${WANDB_PROJECT:-in-place-ttt}"
WANDB_NAME="${WANDB_NAME:-$EXP_NAME}"
TOTAL_STEPS="${TOTAL_STEPS:-4000}"
WINDOW_SCHEDULE="${WINDOW_SCHEDULE:-4096,2048,1024}"
STAGE_STOP_STEPS="${STAGE_STOP_STEPS:-400,800}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs/$EXP_NAME}"
MASTER_PORT="${MASTER_PORT:-12361}"
NODE_RANK="${NODE_RANK:-${MLP_ROLE_INDEX:-0}}"

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
  local load_checkpoint_path="$5"
  shift 5

  CONFIG="$CONFIG" \
  EXP_NAME="${EXP_NAME}-${stage_name}-w${window_size}" \
  OUTPUT_DIR="$output_dir" \
  WANDB_PROJECT="$WANDB_PROJECT" \
  WANDB_NAME="${WANDB_NAME}-${stage_name}-w${window_size}" \
  LOG_FILE="$log_file" \
  MASTER_PORT="$MASTER_PORT" \
  MAX_STEPS="$TOTAL_STEPS" \
  LOAD_CHECKPOINT_PATH="$load_checkpoint_path" \
  TTT_COMPRESS_WINDOW="$window_size" \
    bash "$SCRIPT_DIR/run_pretrain_template.sh" "$@"
}

previous_ckpt=""
last_stage_index=$((${#WINDOWS[@]} - 1))

for stage_index in "${!WINDOWS[@]}"; do
  stage_num=$((stage_index + 1))
  window_size="${WINDOWS[$stage_index]}"
  stage_name="stage${stage_num}"
  output_dir="${OUTPUT_BASE}/${stage_name}-w${window_size}"
  log_file="./logs/log-${WANDB_NAME}-${stage_name}-w${window_size}_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).txt"

  load_checkpoint_path="auto"
  if [[ -n "$previous_ckpt" ]]; then
    load_checkpoint_path="$previous_ckpt"
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
    "$load_checkpoint_path" \
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
