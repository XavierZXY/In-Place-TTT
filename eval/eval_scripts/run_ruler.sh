#!/usr/bin/env bash
# Local RULER wrapper for HuggingFace checkpoints.
#
# Usage:
#   bash eval/eval_scripts/run_ruler.sh <hf_ckpt_dir> <abbr> [gpu_id] [n_per_task]

set -euo pipefail

MODEL_PATH="${1:?Usage: bash eval/eval_scripts/run_ruler.sh <hf_ckpt_dir> <abbr> [gpu_id] [n_per_task]}"
ABBR="${2:?Usage: bash eval/eval_scripts/run_ruler.sh <hf_ckpt_dir> <abbr> [gpu_id] [n_per_task]}"
GPU="${3:-0}"
N_PER_TASK="${4:-50}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${EVAL_ROOT}/.." && pwd)"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON="${PROJECT_ROOT}/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

LENGTHS="${LENGTHS:-4096 8192 16384}"
MAX_SEQ="${MAX_SEQ:-32768}"
RULER_ROOT="${RULER_ROOT:-${EVAL_ROOT}/eval_scripts/ruler}"
EXP_ROOT="${EXP_ROOT:-${EVAL_ROOT}/exp_analysis}"
OUT_ROOT="${OUT_ROOT:-${EXP_ROOT}/ruler_results}"
LOG_ROOT="${LOG_ROOT:-${EXP_ROOT}/logs}"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/ruler/${ABBR}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/run.log}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
STRIP_THINK="${STRIP_THINK:-0}"
STOP_STRINGS="${STOP_STRINGS:-}"
ASSISTANT_PREFILL="${ASSISTANT_PREFILL:-}"
EOS_IDS="${EOS_IDS:-}"
DISABLE_REPETITION_STOP="${DISABLE_REPETITION_STOP:-0}"
REPEAT_STOP_MIN_NEW_TOKENS="${REPEAT_STOP_MIN_NEW_TOKENS:-96}"
REPEAT_STOP_MIN_NGRAM="${REPEAT_STOP_MIN_NGRAM:-8}"
REPEAT_STOP_MAX_NGRAM="${REPEAT_STOP_MAX_NGRAM:-128}"
REPEAT_STOP_REPEATS="${REPEAT_STOP_REPEATS:-3}"
REPEAT_STOP_DIVERSITY_WINDOW="${REPEAT_STOP_DIVERSITY_WINDOW:-160}"
REPEAT_STOP_DIVERSITY_UNIQUE_RATIO="${REPEAT_STOP_DIVERSITY_UNIQUE_RATIO:-0.22}"
REPEAT_STOP_DIVERSITY_TOP_RATIO="${REPEAT_STOP_DIVERSITY_TOP_RATIO:-0.18}"

[[ -d "${MODEL_PATH}" ]] || { echo "ERROR: model dir not found: ${MODEL_PATH}" >&2; exit 1; }
[[ -d "${RULER_ROOT}" ]] || { echo "ERROR: RULER data dir not found: ${RULER_ROOT}" >&2; exit 1; }

read -r -a LENGTH_ARGS <<< "${LENGTHS}"

EOS_FLAG=()
if [[ -n "${EOS_IDS}" ]]; then
  read -r -a EOS_ARGS <<< "${EOS_IDS}"
  EOS_FLAG=(--eos_token_ids "${EOS_ARGS[@]}")
fi

CHAT_FLAG=()
if [[ -n "${CHAT_TEMPLATE}" ]]; then
  CHAT_FLAG=(--chat_template "${CHAT_TEMPLATE}")
fi

STRIP_FLAG=()
if [[ "${STRIP_THINK}" == "1" ]]; then
  STRIP_FLAG=(--strip_think)
fi

STOP_FLAG=()
if [[ -n "${STOP_STRINGS}" ]]; then
  STOP_ARGS=()
  STOP_WORK="${STOP_STRINGS};"
  while [[ "${STOP_WORK}" == *";"* ]]; do
    STOP_ITEM="${STOP_WORK%%;*}"
    STOP_WORK="${STOP_WORK#*;}"
    if [[ -n "${STOP_ITEM}" ]]; then
      STOP_ARGS+=("${STOP_ITEM}")
    fi
  done
  STOP_FLAG=(--stop_strings "${STOP_ARGS[@]}")
fi

PREFILL_FLAG=()
if [[ -n "${ASSISTANT_PREFILL}" ]]; then
  PREFILL_FLAG=(--assistant_prefill "${ASSISTANT_PREFILL}")
fi

REPEAT_FLAG=()
if [[ "${DISABLE_REPETITION_STOP}" == "1" ]]; then
  REPEAT_FLAG=(--disable_repetition_stop)
fi

mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname "${LOG_FILE}")"

echo "[$(date)] running local RULER eval"
echo "  model_path      : ${MODEL_PATH}"
echo "  abbr            : ${ABBR}"
echo "  gpu             : ${GPU}"
echo "  python          : ${PYTHON}"
echo "  ruler_root      : ${RULER_ROOT}"
echo "  exp_root        : ${EXP_ROOT}"
echo "  out_root        : ${OUT_ROOT}"
echo "  log_root        : ${LOG_ROOT}"
echo "  log_dir         : ${LOG_DIR}"
echo "  n_per_task      : ${N_PER_TASK}"
echo "  lengths         : ${LENGTHS}"
echo "  max_seq         : ${MAX_SEQ}"
echo "  max_new_tokens  : ${MAX_NEW_TOKENS}"
echo "  batch_size      : ${BATCH_SIZE}"
echo "  dtype           : ${DTYPE}"
echo "  attn_impl       : ${ATTN_IMPLEMENTATION}"
echo "  chat_template   : ${CHAT_TEMPLATE:-none}"
echo "  strip_think     : ${STRIP_THINK}"
echo "  stop_strings    : ${STOP_STRINGS:-none}"
echo "  eos_ids         : ${EOS_IDS:-auto}"
echo "  repetition_stop : $([[ "${DISABLE_REPETITION_STOP}" == "1" ]] && echo disabled || echo enabled)"
echo "  log_file        : ${LOG_FILE}"

cd "${PROJECT_ROOT}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${SCRIPT_DIR}/eval_ruler.py" \
  --model_path "${MODEL_PATH}" \
  --abbr "${ABBR}" \
  --lengths "${LENGTH_ARGS[@]}" \
  --n_per_task "${N_PER_TASK}" \
  --max_seq "${MAX_SEQ}" \
  --ruler_root "${RULER_ROOT}" \
  --out_root "${OUT_ROOT}" \
  --dtype "${DTYPE}" \
  --attn_implementation "${ATTN_IMPLEMENTATION}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --batch_size "${BATCH_SIZE}" \
  --repeat_stop_min_new_tokens "${REPEAT_STOP_MIN_NEW_TOKENS}" \
  --repeat_stop_min_ngram "${REPEAT_STOP_MIN_NGRAM}" \
  --repeat_stop_max_ngram "${REPEAT_STOP_MAX_NGRAM}" \
  --repeat_stop_repeats "${REPEAT_STOP_REPEATS}" \
  --repeat_stop_diversity_window "${REPEAT_STOP_DIVERSITY_WINDOW}" \
  --repeat_stop_diversity_unique_ratio "${REPEAT_STOP_DIVERSITY_UNIQUE_RATIO}" \
  --repeat_stop_diversity_top_ratio "${REPEAT_STOP_DIVERSITY_TOP_RATIO}" \
  "${EOS_FLAG[@]}" \
  "${CHAT_FLAG[@]}" \
  "${STRIP_FLAG[@]}" \
  "${STOP_FLAG[@]}" \
  "${PREFILL_FLAG[@]}" \
  "${REPEAT_FLAG[@]}" \
  2>&1 | tee "${LOG_FILE}"

echo "[$(date)] done. results -> ${OUT_ROOT}/${ABBR}/"
