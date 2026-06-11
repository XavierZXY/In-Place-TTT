#!/bin/bash
# Qwen3-1.7B Stage 2 KD alignment for strict Full:SWA = 1:3.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export TASK_SCRIPT="${TASK_SCRIPT:-tasks/train_torch_halo_kd.py}"
export CONFIG="${CONFIG:-configs/pretrain/qwen3-1.7b/stage2_kd_swa3_full1_strict.yaml}"
export EXP_NAME="${EXP_NAME:-qwen3-1.7b-stage2-kd-swa3-full1-strict-4k}"
export WANDB_NAME="${WANDB_NAME:-$EXP_NAME}"
export LOG_PREFIX="${LOG_PREFIX:-$EXP_NAME}"
export MASTER_PORT="${MASTER_PORT:-12372}"
STAGE1_EXP_NAME="${STAGE1_EXP_NAME:-qwen3-1.7b-stage1-hidden-align-swa3-full1-strict-512}"
STAGE1_GLOBAL_STEP="${STAGE1_GLOBAL_STEP:-8000}"
export LOAD_CHECKPOINT_PATH="${LOAD_CHECKPOINT_PATH:-outputs/${STAGE1_EXP_NAME}/checkpoints/global_step_${STAGE1_GLOBAL_STEP}}"

if [[ -n "${TEACHER_MODEL_PATH:-}" && -z "${DISTILL_TEACHER_PATH:-}" ]]; then
  export DISTILL_TEACHER_PATH="$TEACHER_MODEL_PATH"
fi

bash "$SCRIPT_DIR/../run_pretrain_template.sh" "$@"
