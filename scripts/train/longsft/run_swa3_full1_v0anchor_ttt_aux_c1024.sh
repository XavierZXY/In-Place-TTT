#!/bin/bash
# V0-aligned Full:SWA = 1:3 hybrid + In-Place TTT aux.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export CONFIG="${CONFIG:-configs/pretrain/qwen3_swa3_full1_v0anchor_ttt_aux_c1024.yaml}"
export EXP_NAME="${EXP_NAME:-qwen3-swa3-full1-v0anchor-ttt-aux-c1024-32k}"
export WANDB_NAME="${WANDB_NAME:-$EXP_NAME}"
export LOG_PREFIX="${LOG_PREFIX:-$EXP_NAME}"
export MASTER_PORT="${MASTER_PORT:-12362}"

bash "$SCRIPT_DIR/run_pretrain_template.sh" "$@"
