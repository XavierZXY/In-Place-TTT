#!/bin/bash
# V0-aligned Full:SWA = 1:3 hybrid + hidden-state TTT + future-chunk hidden aux.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export CONFIG="${CONFIG:-configs/pretrain/qwen3_swa3_full1_v0anchor_ttt_aux_c1024.yaml}"
export EXP_NAME="${EXP_NAME:-qwen3-swa3-full1-v0anchor-hidden-chunk-aux-c1024-32k}"
export WANDB_NAME="${WANDB_NAME:-$EXP_NAME}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/$EXP_NAME}"
export LOG_PREFIX="${LOG_PREFIX:-$EXP_NAME}"
export MASTER_PORT="${MASTER_PORT:-12363}"

export TTT_TARGET="${TTT_TARGET:-hidden_states}"
export TTT_AUX_TARGET="${TTT_AUX_TARGET:-future_chunk_hidden}"
export TTT_AUX_FUTURE_CHUNKS="${TTT_AUX_FUTURE_CHUNKS:-1}"
export TTT_AUX_LOSS_WEIGHT="${TTT_AUX_LOSS_WEIGHT:-0.05}"
export TTT_AUX_LOSS_TYPE="${TTT_AUX_LOSS_TYPE:-jepa}"
export TTT_JEPA_LOSS_EXP="${TTT_JEPA_LOSS_EXP:-2.0}"

export TTT_CHUNK="${TTT_CHUNK:-1024}"
export TTT_COMPRESS_WINDOW="${TTT_COMPRESS_WINDOW:-4096}"
export TTT_LR="${TTT_LR:-5}"
export TTT_PARAM_LR_MULTIPLIER="${TTT_PARAM_LR_MULTIPLIER:-5.0}"
export TTT_PARAM_WEIGHT_DECAY="${TTT_PARAM_WEIGHT_DECAY:-0.0}"

export LR_DECAY_STYLE="${LR_DECAY_STYLE:-constant}"
export LR_DECAY_RATIO="${LR_DECAY_RATIO:-1.0}"

bash "$SCRIPT_DIR/run_pretrain_template.sh" "$@"
