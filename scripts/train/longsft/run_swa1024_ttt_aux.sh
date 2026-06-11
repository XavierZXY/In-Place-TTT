#!/bin/bash
# SWA(1024) + In-Place TTT + JEPA-style aux loss alpha=0.1.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export CONFIG="${CONFIG:-configs/pretrain/qwen3_swa1024_ttt_aux.yaml}"
export EXP_NAME="${EXP_NAME:-qwen3-swa1024-ttt-aux-c256-24k}"
export WANDB_NAME="${WANDB_NAME:-$EXP_NAME}"
export LOG_PREFIX="${LOG_PREFIX:-$EXP_NAME}"
export MASTER_PORT="${MASTER_PORT:-12361}"

bash "$SCRIPT_DIR/run_pretrain_template.sh" "$@"
