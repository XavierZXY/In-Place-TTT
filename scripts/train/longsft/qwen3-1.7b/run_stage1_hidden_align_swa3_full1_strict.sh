#!/bin/bash
# Qwen3-1.7B Stage 1 hidden-state alignment for strict Full:SWA = 1:3.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export TASK_SCRIPT="${TASK_SCRIPT:-tasks/train_torch_halo_hidden_align.py}"
export CONFIG="${CONFIG:-configs/pretrain/qwen3-1.7b/stage1_hidden_align_swa3_full1_strict.yaml}"
export EXP_NAME="${EXP_NAME:-qwen3-1.7b-stage1-hidden-align-swa3-full1-strict-512}"
export WANDB_NAME="${WANDB_NAME:-$EXP_NAME}"
export LOG_PREFIX="${LOG_PREFIX:-$EXP_NAME}"
export MASTER_PORT="${MASTER_PORT:-12371}"

if [[ -n "${TEACHER_MODEL_PATH:-}" && -z "${HIDDEN_ALIGN_TEACHER_PATH:-}" ]]; then
  export HIDDEN_ALIGN_TEACHER_PATH="$TEACHER_MODEL_PATH"
fi

bash "$SCRIPT_DIR/../run_pretrain_template.sh" "$@"
