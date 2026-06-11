#!/bin/bash
# Qwen3-1.7B Stage 3 CPT for strict Full:SWA = 1:3 hybrid + In-Place TTT aux.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export CONFIG="${CONFIG:-configs/pretrain/qwen3-1.7b/stage3_cpt_swa3_full1_strict.yaml}"
export EXP_NAME="${EXP_NAME:-qwen3-1.7b-stage3-cpt-swa3-full1-strict-c2048-32k}"
export WANDB_NAME="${WANDB_NAME:-$EXP_NAME}"
export LOG_PREFIX="${LOG_PREFIX:-$EXP_NAME}"
export MASTER_PORT="${MASTER_PORT:-12373}"
STAGE2_EXP_NAME="${STAGE2_EXP_NAME:-qwen3-1.7b-stage2-kd-swa3-full1-strict-4k}"
STAGE2_GLOBAL_STEP="${STAGE2_GLOBAL_STEP:-8000}"
export LOAD_CHECKPOINT_PATH="${LOAD_CHECKPOINT_PATH:-outputs/${STAGE2_EXP_NAME}/checkpoints/global_step_${STAGE2_GLOBAL_STEP}}"

bash "$SCRIPT_DIR/../run_pretrain_template.sh" "$@"
