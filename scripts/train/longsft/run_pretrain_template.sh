#!/bin/bash
# Generic pretrain launcher template. Data-specific wrappers should only set defaults.

set -euo pipefail
if [[ "${TRACE:-1}" == "1" ]]; then
  set -x
fi

CONFIG="${CONFIG:-configs/pretrain/qwen3_swa3_full1_v0anchor_ttt_aux_c2048.yaml}"
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
append_runtime_arg "${EVAL_DATASETS_TYPE:-}" --data.eval_datasets_type
append_runtime_arg "${DATALOADER_TYPE:-}" --data.dataloader_type
append_runtime_arg "${DATASETS_TYPE:-}" --data.datasets_type
append_runtime_arg "${DATA_TYPE:-}" --data.data_type
append_runtime_arg "${CHAT_TEMPLATE:-}" --data.chat_template
append_runtime_arg "${MAX_SEQ_LEN:-}" --data.max_seq_len
append_runtime_arg "${TEXT_KEYS:-}" --data.text_keys
append_runtime_arg "${TRAIN_SIZE:-}" --data.train_size
append_runtime_arg "${MAX_STEPS:-}" --train.max_steps
append_runtime_arg "${SAVE_STEPS:-}" --train.save_steps
append_runtime_arg "${EVAL_STEPS:-}" --train.eval_steps
append_runtime_arg "${EVAL_BATCHES:-}" --train.eval_batches
append_runtime_arg "${GLOBAL_BATCH_SIZE:-}" --train.global_batch_size
append_runtime_arg "${MICRO_BATCH_SIZE:-}" --train.micro_batch_size
append_runtime_arg "${LR:-}" --train.lr
append_runtime_arg "${LR_MIN:-}" --train.lr_min
append_runtime_arg "${LR_WARMUP_RATIO:-}" --train.lr_warmup_ratio
append_runtime_arg "${LR_DECAY_STYLE:-}" --train.lr_decay_style
append_runtime_arg "${LR_DECAY_RATIO:-}" --train.lr_decay_ratio
append_runtime_arg "${WEIGHT_DECAY:-}" --train.weight_decay
append_runtime_arg "${MAX_GRAD_NORM:-}" --train.max_grad_norm
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

append_foundation_string() {
  local key="$1"
  local value="$2"
  if [[ -n "$value" ]]; then
    foundation_items+=("\"$key\": \"$value\"")
  fi
}

append_foundation_raw "ttt_chunk" "${TTT_CHUNK:-}"
append_foundation_raw "ttt_compress_window" "${TTT_COMPRESS_WINDOW:-}"
append_foundation_raw "ttt_lr" "${TTT_LR:-}"
append_foundation_raw "ttt_aux_loss_weight" "${TTT_AUX_LOSS_WEIGHT:-}"
append_foundation_string "ttt_target" "${TTT_TARGET:-}"
append_foundation_string "ttt_aux_target" "${TTT_AUX_TARGET:-}"
append_foundation_raw "ttt_aux_future_chunks" "${TTT_AUX_FUTURE_CHUNKS:-}"
append_foundation_string "ttt_aux_loss_type" "${TTT_AUX_LOSS_TYPE:-}"
append_foundation_raw "ttt_jepa_loss_exp" "${TTT_JEPA_LOSS_EXP:-}"
append_foundation_raw "ttt_jepa_reg_coeff" "${TTT_JEPA_REG_COEFF:-}"
append_foundation_raw "ttt_monitor_sample_dim" "${TTT_MONITOR_SAMPLE_DIM:-}"
append_foundation_raw "ttt_monitor_sample_tokens" "${TTT_MONITOR_SAMPLE_TOKENS:-}"
append_foundation_string "ttt_monitor_output_delta_target" "${TTT_MONITOR_OUTPUT_DELTA_TARGET:-}"
append_foundation_raw "ttt_monitor_logit_sample_tokens" "${TTT_MONITOR_LOGIT_SAMPLE_TOKENS:-}"
append_foundation_raw "ttt_monitor_logit_sample_dim" "${TTT_MONITOR_LOGIT_SAMPLE_DIM:-}"
append_foundation_raw "ttt_param_lr_multiplier" "${TTT_PARAM_LR_MULTIPLIER:-}"
append_foundation_raw "ttt_param_weight_decay" "${TTT_PARAM_WEIGHT_DECAY:-}"
append_foundation_raw "ttt_train_only" "${TTT_TRAIN_ONLY:-}"
append_foundation_raw "ttt_mode" "${TTT_MODE:-}"
append_foundation_raw "ttt_layers" "${TTT_LAYERS:-}"
append_foundation_raw "full_attention_layers" "${FULL_ATTENTION_LAYERS:-}"
append_foundation_string "distill_teacher_path" "${DISTILL_TEACHER_PATH:-}"
append_foundation_string "hidden_align_teacher_path" "${HIDDEN_ALIGN_TEACHER_PATH:-}"
append_foundation_string "hidden_align_loss_fn" "${HIDDEN_ALIGN_LOSS_FN:-}"
append_foundation_raw "hidden_align_layers" "${HIDDEN_ALIGN_LAYERS:-}"
append_foundation_raw "hidden_align_skip_ttt_layers" "${HIDDEN_ALIGN_SKIP_TTT_LAYERS:-}"
append_foundation_string "hidden_align_train_scope" "${HIDDEN_ALIGN_TRAIN_SCOPE:-}"

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
