#!/bin/bash
# Qwen3-1.7B Stage 3 CPT variant: strict Full:SWA + In-Place TTT KeyNorm.
#
# This wrapper intentionally delegates initialization and launch details to the
# base strict Stage-3 script. It only provides safe experiment defaults and the
# foundation overrides validated by the keynorm smoke runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export EXP_NAME="${EXP_NAME:-qwen3-1.7b-stage3-cpt-swa3-full0-strict-c1024-keynorm-lr0p05}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/$EXP_NAME}"
export WANDB_NAME="${WANDB_NAME:-$EXP_NAME}"
export LOG_PREFIX="${LOG_PREFIX:-$EXP_NAME}"
export MASTER_PORT="${MASTER_PORT:-12374}"

if [[ -z "${MODEL_FOUNDATION_JSON:-}" ]]; then
  TTT_KEY_NORM_VALUE="${TTT_KEY_NORM:-true}"
  TTT_LR_VALUE="${TTT_LR:-0.05}"
  TTT_AUX_LOSS_WEIGHT_VALUE="${TTT_AUX_LOSS_WEIGHT:-0.0}"
  TTT_TRAIN_ONLY_VALUE="${TTT_TRAIN_ONLY:-false}"

  export MODEL_FOUNDATION_JSON="{\"ttt_key_norm\": ${TTT_KEY_NORM_VALUE}, \"ttt_lr\": ${TTT_LR_VALUE}, \"ttt_aux_loss_weight\": ${TTT_AUX_LOSS_WEIGHT_VALUE}, \"ttt_train_only\": ${TTT_TRAIN_ONLY_VALUE}}"
fi

exec bash "$SCRIPT_DIR/run_stage3_cpt_swa3_full1_strict.sh" "$@"
