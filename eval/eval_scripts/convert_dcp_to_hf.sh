#!/usr/bin/env bash
# Convert a VeOmni DCP checkpoint to HuggingFace format for local eval.
#
# Usage:
#   bash eval/eval_scripts/convert_dcp_to_hf.sh <model_root_dir> <step> [--force]

set -euo pipefail

MODEL_ROOT="${1:?Usage: bash eval/eval_scripts/convert_dcp_to_hf.sh <model_root_dir> <step> [--force]}"
STEP="${2:?Usage: bash eval/eval_scripts/convert_dcp_to_hf.sh <model_root_dir> <step> [--force]}"
FORCE="${3:-}"

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

LOAD_DIR="${MODEL_ROOT}/checkpoints/global_step_${STEP}"
SAVE_DIR="${SAVE_DIR:-${LOAD_DIR}/hf_ckpt}"
MODEL_ASSETS_DIR="${MODEL_ASSETS_DIR:-${MODEL_ROOT}/model_assets}"
SHARD_SIZE="${SHARD_SIZE:-2000000000}"

[[ -d "${LOAD_DIR}" ]] || { echo "ERROR: checkpoint dir not found: ${LOAD_DIR}" >&2; exit 1; }

if [[ -d "${SAVE_DIR}" && -n "$(find "${SAVE_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  if [[ "${FORCE}" != "--force" ]]; then
    echo "HF checkpoint already exists: ${SAVE_DIR}" >&2
    echo "Pass --force to overwrite it." >&2
    exit 0
  fi
  rm -rf "${SAVE_DIR}"
fi

ASSET_ARGS=()
if [[ -d "${MODEL_ASSETS_DIR}" ]]; then
  ASSET_ARGS=(--model-assets-dir "${MODEL_ASSETS_DIR}")
fi

echo "[$(date)] converting DCP checkpoint to HF"
echo "  python          : ${PYTHON}"
echo "  load_dir        : ${LOAD_DIR}"
echo "  save_dir        : ${SAVE_DIR}"
echo "  model_assets_dir: $([[ -d "${MODEL_ASSETS_DIR}" ]] && echo "${MODEL_ASSETS_DIR}" || echo "none")"
echo "  shard_size      : ${SHARD_SIZE}"

cd "${PROJECT_ROOT}"

"${PYTHON}" "scripts/merge_dcp_to_hf.py" \
  --load-dir "${LOAD_DIR}" \
  --save-dir "${SAVE_DIR}" \
  --shard-size "${SHARD_SIZE}" \
  "${ASSET_ARGS[@]}"

echo "[$(date)] done. HF checkpoint -> ${SAVE_DIR}"
