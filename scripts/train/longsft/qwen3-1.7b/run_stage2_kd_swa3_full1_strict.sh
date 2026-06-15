#!/bin/bash
# Qwen3-1.7B Stage 2 KD alignment for strict Full:SWA = 1:3.
#
# By default, stage-1 DCP checkpoint is converted to HF format once and used as
# the student's initial weights (model_path). Set STAGE2_INIT_FROM=base to skip
# stage 1 and initialize the student directly from BASE_MODEL_PATH.
# load_checkpoint_path stays "auto" and only resumes stage-2's own checkpoints.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

export TASK_SCRIPT="${TASK_SCRIPT:-tasks/train_torch_halo_kd.py}"
export CONFIG="${CONFIG:-configs/pretrain/qwen3-1.7b/stage2_kd_swa3_full1_strict.yaml}"
export WANDB_PROJECT="${WANDB_PROJECT:-in-place-ttt-mini}"
export EXP_NAME="${EXP_NAME:-qwen3-1.7b-stage2-kd-swa-full0-strict-4096}"
export WANDB_NAME="${WANDB_NAME:-$EXP_NAME}"
export LOG_PREFIX="${LOG_PREFIX:-$EXP_NAME}"
export MASTER_PORT="${MASTER_PORT:-12372}"

STAGE2_INIT_FROM="${STAGE2_INIT_FROM:-stage1}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/zouxiangyu/models/Qwen/Qwen3-1.7B}"

case "$STAGE2_INIT_FROM" in
  stage1)
    STAGE1_EXP_NAME="${STAGE1_EXP_NAME:-qwen3-1.7b-stage1-hidden-align-swa3-full1-strict-4096}"
    STAGE1_OUTPUT_DIR="$REPO_ROOT/outputs/${STAGE1_EXP_NAME}"
    # Default to the latest stage-1 checkpoint (stage 1 is data-bound and may stop
    # before max_steps, so the final step number is not fixed).
    if [[ -z "${STAGE1_GLOBAL_STEP:-}" ]]; then
      STAGE1_GLOBAL_STEP="$(ls -d "$STAGE1_OUTPUT_DIR"/checkpoints/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1)"
      if [[ -z "$STAGE1_GLOBAL_STEP" ]]; then
        echo "No stage-1 checkpoints found under $STAGE1_OUTPUT_DIR/checkpoints" >&2
        echo "Set STAGE1_EXP_NAME / STAGE1_GLOBAL_STEP or STAGE1_HF_CKPT." >&2
        exit 1
      fi
    fi
    STAGE1_DCP_CKPT="$STAGE1_OUTPUT_DIR/checkpoints/global_step_${STAGE1_GLOBAL_STEP}"
    STAGE1_HF_CKPT="${STAGE1_HF_CKPT:-$STAGE1_DCP_CKPT/hf_ckpt}"

    if [[ ! -f "$STAGE1_HF_CKPT/config.json" ]]; then
      if [[ ! -d "$STAGE1_DCP_CKPT" ]]; then
        echo "Stage-1 DCP checkpoint not found: $STAGE1_DCP_CKPT" >&2
        echo "Set STAGE1_EXP_NAME / STAGE1_GLOBAL_STEP or STAGE1_HF_CKPT." >&2
        exit 1
      fi
      echo "Converting stage-1 DCP checkpoint to HF format: $STAGE1_HF_CKPT"
      "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/merge_dcp_to_hf.py" \
        --load-dir "$STAGE1_DCP_CKPT" \
        --save-dir "$STAGE1_HF_CKPT" \
        --model-assets-dir "$STAGE1_OUTPUT_DIR/model_assets"
    fi
    export MODEL_PATH="${MODEL_PATH:-$STAGE1_HF_CKPT}"
    ;;
  base)
    export MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH}"
    ;;
  *)
    echo "Unsupported STAGE2_INIT_FROM: $STAGE2_INIT_FROM" >&2
    echo "Use STAGE2_INIT_FROM=stage1 or STAGE2_INIT_FROM=base." >&2
    exit 1
    ;;
esac

if [[ -n "${TEACHER_MODEL_PATH:-}" && -z "${DISTILL_TEACHER_PATH:-}" ]]; then
  export DISTILL_TEACHER_PATH="$TEACHER_MODEL_PATH"
fi

bash "$SCRIPT_DIR/../run_pretrain_template.sh" "$@"
