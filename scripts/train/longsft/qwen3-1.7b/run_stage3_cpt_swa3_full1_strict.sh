#!/bin/bash
# Qwen3-1.7B Stage 3 CPT for strict Full:SWA = 1:3 hybrid + In-Place TTT.
#
# Stage transition: stage-2 DCP checkpoint is converted to HF format once and
# used as the student's initial weights (model_path). load_checkpoint_path
# stays "auto" and only resumes stage-3's own checkpoints.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

export CONFIG="${CONFIG:-configs/pretrain/qwen3-1.7b/stage3_cpt_swa3_full1_strict.yaml}"
export WANDB_PROJECT="${WANDB_PROJECT:-in-place-ttt-mini}"
export EXP_NAME="${EXP_NAME:-qwen3-1.7b-stage3-cpt-swa3-full1-strict-c1024-16k}"
export WANDB_NAME="${WANDB_NAME:-$EXP_NAME}"
export LOG_PREFIX="${LOG_PREFIX:-$EXP_NAME}"
export MASTER_PORT="${MASTER_PORT:-12373}"

STAGE2_EXP_NAME="${STAGE2_EXP_NAME:-qwen3-1.7b-stage2-kd-swa3-full1-strict-4096}"
STAGE2_OUTPUT_DIR="$REPO_ROOT/outputs/${STAGE2_EXP_NAME}"
# Default to the latest stage-2 checkpoint.
if [[ -z "${STAGE2_GLOBAL_STEP:-}" ]]; then
  STAGE2_GLOBAL_STEP="$(ls -d "$STAGE2_OUTPUT_DIR"/checkpoints/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1)"
  if [[ -z "$STAGE2_GLOBAL_STEP" ]]; then
    echo "No stage-2 checkpoints found under $STAGE2_OUTPUT_DIR/checkpoints" >&2
    echo "Set STAGE2_EXP_NAME / STAGE2_GLOBAL_STEP or STAGE2_HF_CKPT." >&2
    exit 1
  fi
fi
STAGE2_DCP_CKPT="$STAGE2_OUTPUT_DIR/checkpoints/global_step_${STAGE2_GLOBAL_STEP}"
STAGE2_HF_CKPT="${STAGE2_HF_CKPT:-$STAGE2_DCP_CKPT/hf_ckpt}"

if [[ ! -f "$STAGE2_HF_CKPT/config.json" ]]; then
  if [[ ! -d "$STAGE2_DCP_CKPT" ]]; then
    echo "Stage-2 DCP checkpoint not found: $STAGE2_DCP_CKPT" >&2
    exit 1
  fi
  echo "Converting stage-2 DCP checkpoint to HF format: $STAGE2_HF_CKPT"
  "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/merge_dcp_to_hf.py" \
    --load-dir "$STAGE2_DCP_CKPT" \
    --save-dir "$STAGE2_HF_CKPT" \
    --model-assets-dir "$STAGE2_OUTPUT_DIR/model_assets"
fi
export MODEL_PATH="${MODEL_PATH:-$STAGE2_HF_CKPT}"

bash "$SCRIPT_DIR/../run_pretrain_template.sh" "$@"
