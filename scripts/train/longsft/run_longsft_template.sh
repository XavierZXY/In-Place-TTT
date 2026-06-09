#!/bin/bash
# Generic LongSFT launcher template.

set -euo pipefail
if [[ "${TRACE:-1}" == "1" ]]; then
  set -x
fi

CONFIG="${CONFIG:-configs/pretrain/qwen3_longsft_swa_full_1to3_ttt_aux.yaml}"
TASK_SCRIPT="${TASK_SCRIPT:-tasks/train_torch.py}"

config_base="${CONFIG##*/}"
config_stem="${config_base%.*}"
EXP_NAME="${EXP_NAME:-$config_stem}"

WANDB_PROJECT="${WANDB_PROJECT:-in-place-ttt}"
WANDB_NAME="${WANDB_NAME:-$EXP_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/$EXP_NAME}"
LOG_DIR="${LOG_DIR:-./logs}"
LOG_PREFIX="${LOG_PREFIX:-$EXP_NAME}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

TORCHRUN="${TORCHRUN:-$REPO_ROOT/.venv/bin/torchrun}"
if [[ ! -x "$TORCHRUN" ]]; then
  TORCHRUN="$(command -v torchrun || true)"
fi
if [[ -z "$TORCHRUN" ]]; then
  echo "torchrun not found. Set TORCHRUN=/path/to/torchrun or activate the training environment." >&2
  exit 1
fi

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if [[ -n "${MLP_GPU:-}" ]]; then
    NPROC_PER_NODE="$MLP_GPU"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a visible_devices <<< "$CUDA_VISIBLE_DEVICES"
    NPROC_PER_NODE="${#visible_devices[@]}"
  else
    NPROC_PER_NODE="8"
  fi
fi

NNODES="${NNODES:-${MLP_WORKER_NUM:-1}}"
NODE_RANK="${NODE_RANK:-${MLP_ROLE_INDEX:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-localhost}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-12361}}"

additional_args=()
if [[ "$NNODES" == "1" ]]; then
  additional_args+=(--standalone)
fi

runtime_args=(
  --train.output_dir "$OUTPUT_DIR"
  --train.wandb_project "$WANDB_PROJECT"
  --train.wandb_name "$WANDB_NAME"
)

append_runtime_arg() {
  local value="$1"
  local key="$2"
  if [[ -n "$value" ]]; then
    runtime_args+=("$key" "$value")
  fi
}

append_runtime_arg "${USE_WANDB:-}" --train.use_wandb
append_runtime_arg "${TRAIN_PATH:-}" --data.train_path
append_runtime_arg "${EVAL_PATH:-}" --data.eval_path
append_runtime_arg "${MAX_SEQ_LEN:-}" --data.max_seq_len
append_runtime_arg "${MAX_STEPS:-}" --train.max_steps
append_runtime_arg "${SAVE_STEPS:-}" --train.save_steps
append_runtime_arg "${EVAL_STEPS:-}" --train.eval_steps
append_runtime_arg "${EVAL_BATCHES:-}" --train.eval_batches
append_runtime_arg "${GLOBAL_BATCH_SIZE:-}" --train.global_batch_size
append_runtime_arg "${MICRO_BATCH_SIZE:-}" --train.micro_batch_size
append_runtime_arg "${LR:-}" --train.lr
if [[ "${LOAD_CHECKPOINT_PATH+x}" == "x" ]]; then
  runtime_args+=(--train.load_checkpoint_path "$LOAD_CHECKPOINT_PATH")
fi

if [[ -n "${HF_MODEL_DIR:-}" ]]; then
  MODEL_PATH="${MODEL_PATH:-$HF_MODEL_DIR}"
  MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH:-$HF_MODEL_DIR}"
  TOKENIZER_PATH="${TOKENIZER_PATH:-$HF_MODEL_DIR}"
fi
append_runtime_arg "${MODEL_PATH:-}" --model.model_path
append_runtime_arg "${MODEL_CONFIG_PATH:-}" --model.config_path
append_runtime_arg "${TOKENIZER_PATH:-}" --model.tokenizer_path

foundation_items=()
append_foundation_raw() {
  local key="$1"
  local value="$2"
  if [[ -n "$value" ]]; then
    foundation_items+=("\"$key\": $value")
  fi
}

append_foundation_raw "ttt_chunk" "${TTT_CHUNK:-}"
append_foundation_raw "ttt_compress_window" "${TTT_COMPRESS_WINDOW:-}"
append_foundation_raw "ttt_lr" "${TTT_LR:-}"
append_foundation_raw "ttt_aux_loss_weight" "${TTT_AUX_LOSS_WEIGHT:-}"
append_foundation_raw "ttt_param_lr_multiplier" "${TTT_PARAM_LR_MULTIPLIER:-}"
append_foundation_raw "ttt_param_weight_decay" "${TTT_PARAM_WEIGHT_DECAY:-}"
append_foundation_raw "ttt_mode" "${TTT_MODE:-}"
append_foundation_raw "ttt_layers" "${TTT_LAYERS:-}"
append_foundation_raw "full_attention_layers" "${FULL_ATTENTION_LAYERS:-}"

foundation_override="${MODEL_FOUNDATION_JSON:-}"
if [[ -z "$foundation_override" && "${#foundation_items[@]}" -gt 0 ]]; then
  foundation_override="{"
  for item in "${foundation_items[@]}"; do
    if [[ "$foundation_override" != "{" ]]; then
      foundation_override+=", "
    fi
    foundation_override+="$item"
  done
  foundation_override+="}"
fi
if [[ -n "$foundation_override" ]]; then
  runtime_args+=(--model.foundation "$foundation_override")
fi

mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/log-${LOG_PREFIX}_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).txt}"

printf 'CONFIG=%s\nOUTPUT_DIR=%s\nWANDB_PROJECT=%s\nWANDB_NAME=%s\nLOG_FILE=%s\n' \
  "$CONFIG" "$OUTPUT_DIR" "$WANDB_PROJECT" "$WANDB_NAME" "$LOG_FILE"

"$TORCHRUN" \
  --nproc_per_node "$NPROC_PER_NODE" \
  --nnodes "$NNODES" \
  --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  "${additional_args[@]}" "$TASK_SCRIPT" "$CONFIG" \
  "${runtime_args[@]}" \
  "$@" 2>&1 | tee "$LOG_FILE"
