#!/bin/bash
# HALO Stage-2 style KD for Full:SWA = 1:3 hybrid + TTT on SWA layers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export TASK_SCRIPT="${TASK_SCRIPT:-tasks/train_torch_halo_kd.py}"
export CONFIG="${CONFIG:-configs/pretrain/qwen3_swa3_full1_v0anchor_halo_kd.yaml}"
export EXP_NAME="${EXP_NAME:-qwen3-swa3-full1-v0anchor-halo-kd-c1024-4k}"
export WANDB_NAME="${WANDB_NAME:-$EXP_NAME}"
export LOG_PREFIX="${LOG_PREFIX:-$EXP_NAME}"
export MASTER_PORT="${MASTER_PORT:-12363}"

if [[ -n "${TEACHER_MODEL_PATH:-}" && -z "${DISTILL_TEACHER_PATH:-}" ]]; then
  export DISTILL_TEACHER_PATH="$TEACHER_MODEL_PATH"
fi

bash "$SCRIPT_DIR/run_pretrain_template.sh" "$@"
