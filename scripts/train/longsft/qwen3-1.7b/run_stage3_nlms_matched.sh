#!/bin/bash
# Matched CPT arm: stage3 strict + selectable TTT write rule (outer/nlms/keynorm).
# All arms share the base stage3 config (layout/data/optimizer); only the write
# rule + ttt_lr (+ nlms lambda) differ. MODEL_FOUNDATION_JSON MERGES into the YAML
# foundation (train_torch.py:388), so layout fields (ttt_layers/chunk/target) are
# preserved; we only override the write-rule delta.
#
# Usage:
#   WRITE_RULE=outer  TTT_LR=3     bash run_stage3_nlms_matched.sh
#   WRITE_RULE=nlms   TTT_LR=0.03  bash run_stage3_nlms_matched.sh
#   WRITE_RULE=keynorm TTT_LR=0.05 bash run_stage3_nlms_matched.sh   # negative control
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

WRITE_RULE="${WRITE_RULE:-outer}"
TTT_LR_VALUE="${TTT_LR:-3}"
NLMS_LAMBDA="${NLMS_LAMBDA:-1.0}"
# truncated BPTT: detach cross-chunk history so backward does not unroll all
# chunks (the gradient-explosion fix). On by default for the nlms arm; set
# NLMS_DETACH=false to reproduce the un-truncated (diverging) baseline.
NLMS_DETACH_VALUE="${NLMS_DETACH:-true}"
# decay gate alpha for NLMS state: S<-(1-a)S+dW, bounds ||S|| to break runaway
# feedback. Default 0.1 for the nlms arm; 0.0 reproduces pure (diverging) NLMS.
NLMS_DECAY_VALUE="${NLMS_DECAY:-0.1}"
TTT_TRAIN_ONLY_VALUE="${TTT_TRAIN_ONLY:-true}"

export EXP_NAME="${EXP_NAME:-qwen3-1.7b-stage3-matched-${WRITE_RULE}-lr${TTT_LR_VALUE}}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/$EXP_NAME}"
export WANDB_NAME="${WANDB_NAME:-$EXP_NAME}"
export LOG_PREFIX="${LOG_PREFIX:-$EXP_NAME}"
export MASTER_PORT="${MASTER_PORT:-12375}"

if [[ -z "${MODEL_FOUNDATION_JSON:-}" ]]; then
  case "$WRITE_RULE" in
    outer)
      export MODEL_FOUNDATION_JSON="{\"ttt_write_rule\": \"outer\", \"ttt_lr\": ${TTT_LR_VALUE}, \"ttt_train_only\": ${TTT_TRAIN_ONLY_VALUE}}"
      ;;
    nlms)
      export MODEL_FOUNDATION_JSON="{\"ttt_write_rule\": \"nlms\", \"ttt_lr\": ${TTT_LR_VALUE}, \"ttt_nlms_lambda\": ${NLMS_LAMBDA}, \"ttt_nlms_detach_state\": ${NLMS_DETACH_VALUE}, \"ttt_nlms_decay\": ${NLMS_DECAY_VALUE}, \"ttt_train_only\": ${TTT_TRAIN_ONLY_VALUE}}"
      ;;
    keynorm)
      export MODEL_FOUNDATION_JSON="{\"ttt_key_norm\": true, \"ttt_lr\": ${TTT_LR_VALUE}, \"ttt_train_only\": ${TTT_TRAIN_ONLY_VALUE}}"
      ;;
    *)
      echo "Unsupported WRITE_RULE: $WRITE_RULE (use outer|nlms|keynorm)" >&2
      exit 1
      ;;
  esac
fi

exec bash "$SCRIPT_DIR/run_stage3_cpt_swa3_full1_strict.sh" "$@"
