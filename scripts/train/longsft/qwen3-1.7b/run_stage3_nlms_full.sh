#!/bin/bash
# =============================================================================
# Qwen3-1.7B Stage-3 NLMS full retrain pipeline (single arm, 8-GPU).
#
# GOAL: validate that per-key NLMS (with ttt_lr warmup cold-start) stays stable
#   across a FULL stage-3 CPT run (10000 steps @ 64k) and that the write signal
#   (ttt_do) keeps rising — i.e. projections trained UNDER NLMS dynamics make the
#   residual converge, unlike drop-in on an outer-trained ckpt (which diverged).
#
# This is the "option 2" route: introduce NLMS at the stage-3 start (where the
# ttt_conv/ttt_proj projections are born) so they learn the NLMS dynamics from
# step 0. ttt_lr warmup (init->target over N steps) protects the cold-start.
#
# Three phases (each can be run independently via PHASE=train|merge|eval):
#   1. train  — full stage-3 CPT with NLMS write rule + ttt_lr warmup
#   2. merge  — convert the final DCP checkpoint to HF format
#   3. eval   — RULER eval (4k/8k/16k) vs baselines (outer-CPT 0.285, off-TTT 0.458)
#
# Usage:
#   bash run_stage3_nlms_full.sh                 # all phases: train -> merge -> eval
#   PHASE=train bash run_stage3_nlms_full.sh     # train only
#   PHASE=eval  bash run_stage3_nlms_full.sh     # eval only (after train+merge)
#
# Overridable env (with defaults):
#   GPUS=0,1,2,3,4,5,6,7   MAX_STEPS=10000   SAVE_STEPS=3000
#   TTT_LR=3   TTT_LR_WARMUP_STEPS=1000   TTT_LR_WARMUP_INIT=0.001
#   EVAL_LENGTHS="4096 8192 16384"   EVAL_N_PER_TASK=50
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

# ---- knobs -----------------------------------------------------------------
PHASE="${PHASE:-all}"                       # all | train | merge | eval
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"             # 8-GPU single arm

# Experiment identity
EXP_NAME="${EXP_NAME:-qwen3-1.7b-stage3-nlms-full-w1000}"
export EXP_NAME
OUTPUT_DIR="$REPO_ROOT/outputs/$EXP_NAME"

# Training scale (defaults follow stage3 config)
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-3000}"

# NLMS write rule + cold-start stabilization
TTT_LR="${TTT_LR:-3}"
TTT_LR_WARMUP_STEPS="${TTT_LR_WARMUP_STEPS:-1000}"
TTT_LR_WARMUP_INIT="${TTT_LR_WARMUP_INIT:-0.001}"
NLMS_DECAY="${NLMS_DECAY:-0.0}"             # warmup alone proved sufficient in 4k/300 smoke
NLMS_DETACH="${NLMS_DETACH:-false}"         # full BPTT (warmup keeps grads bounded)

# stage-2 init checkpoint
STAGE2_EXP_NAME="${STAGE2_EXP_NAME:-qwen3-1.7b-stage2-kd-swa-full0-strict-8192}"
STAGE2_GLOBAL_STEP="${STAGE2_GLOBAL_STEP:-11000}"

# Eval
RULER_ROOT="${RULER_ROOT:-/zouxiangyu/codes/TTT/In-Place-TTT-v0/eval_scripts/ruler}"
TOKENIZER="${TOKENIZER:-/zouxiangyu/models/Qwen/Qwen3-1.7B}"
EVAL_LENGTHS="${EVAL_LENGTHS:-4096 8192 16384}"
EVAL_N_PER_TASK="${EVAL_N_PER_TASK:-50}"
EVAL_MAX_SEQ="${EVAL_MAX_SEQ:-32768}"
EVAL_OUT_ROOT="${EVAL_OUT_ROOT:-$REPO_ROOT/eval/exp_analysis/ruler_results_nlms_full}"
EVAL_GPU="${EVAL_GPU:-0}"                   # single GPU for eval

PY="$REPO_ROOT/.venv/bin/python"

echo "=============================================================="
echo " Stage-3 NLMS full retrain pipeline"
echo "   EXP_NAME            = $EXP_NAME"
echo "   PHASE               = $PHASE"
echo "   GPUS                = $GPUS"
echo "   MAX_STEPS/SAVE      = $MAX_STEPS / $SAVE_STEPS"
echo "   TTT_LR (target)     = $TTT_LR"
echo "   warmup steps/init   = $TTT_LR_WARMUP_STEPS / $TTT_LR_WARMUP_INIT"
echo "   nlms decay/detach   = $NLMS_DECAY / $NLMS_DETACH"
echo "   stage2 init         = $STAGE2_EXP_NAME @ gs$STAGE2_GLOBAL_STEP"
echo "   OUTPUT_DIR          = $OUTPUT_DIR"
echo "=============================================================="

# ---- phase 1: train --------------------------------------------------------
run_train() {
  echo "[phase:train] launching full NLMS stage-3 CPT ..."
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  CUDA_VISIBLE_DEVICES="$GPUS" \
  WRITE_RULE=nlms \
  TTT_LR="$TTT_LR" \
  NLMS_DECAY="$NLMS_DECAY" \
  NLMS_DETACH="$NLMS_DETACH" \
  TTT_LR_WARMUP_STEPS="$TTT_LR_WARMUP_STEPS" \
  TTT_LR_WARMUP_INIT="$TTT_LR_WARMUP_INIT" \
  STAGE2_EXP_NAME="$STAGE2_EXP_NAME" \
  STAGE2_GLOBAL_STEP="$STAGE2_GLOBAL_STEP" \
  MAX_STEPS="$MAX_STEPS" \
  SAVE_STEPS="$SAVE_STEPS" \
  EXP_NAME="$EXP_NAME" \
  MASTER_PORT="${MASTER_PORT:-12377}" \
  bash "$SCRIPT_DIR/run_stage3_nlms_matched.sh"
  echo "[phase:train] done."
}

# ---- phase 2: merge DCP -> HF ----------------------------------------------
# Resolve the latest stage-3 checkpoint into globals (LAST_STEP, DCP_CKPT, HF_CKPT).
# NOTE: call directly (NOT in a $(...) subshell) so the global assignments persist.
LAST_STEP=""
DCP_CKPT=""
HF_CKPT=""
resolve_hf_ckpt() {
  LAST_STEP="$(ls -d "$OUTPUT_DIR"/checkpoints/global_step_* 2>/dev/null \
    | sed 's/.*global_step_//' | sort -n | tail -1)"
  if [[ -z "$LAST_STEP" ]]; then
    echo "[resolve] no stage-3 checkpoints under $OUTPUT_DIR/checkpoints" >&2
    return 1
  fi
  DCP_CKPT="$OUTPUT_DIR/checkpoints/global_step_${LAST_STEP}"
  HF_CKPT="$DCP_CKPT/hf_ckpt"
}

run_merge() {
  echo "[phase:merge] resolving latest checkpoint ..."
  if ! resolve_hf_ckpt; then
    echo "[phase:merge] no checkpoint to merge — run training first." >&2
    exit 1
  fi
  echo "[phase:merge] latest = global_step_${LAST_STEP}"
  if [[ -f "$HF_CKPT/config.json" ]]; then
    echo "[phase:merge] HF ckpt already exists: $HF_CKPT (skip)"
    return 0
  fi
  echo "[phase:merge] converting $DCP_CKPT -> $HF_CKPT"
  "$PY" "$REPO_ROOT/scripts/merge_dcp_to_hf.py" \
    --load-dir "$DCP_CKPT" \
    --save-dir "$HF_CKPT" \
    --model-assets-dir "$OUTPUT_DIR/model_assets"
  echo "[phase:merge] done: $HF_CKPT"
}

# ---- phase 3: eval ---------------------------------------------------------
run_eval() {
  echo "[phase:eval] resolving HF ckpt ..."
  if ! resolve_hf_ckpt; then
    echo "[phase:eval] no checkpoint found — run training (and merge) first." >&2
    exit 1
  fi
  if [[ ! -f "$HF_CKPT/config.json" ]]; then
    echo "[phase:eval] HF ckpt missing ($HF_CKPT); run merge first." >&2
    exit 1
  fi
  echo "[phase:eval] RULER eval @ lengths: $EVAL_LENGTHS"
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
  "$PY" "$REPO_ROOT/eval/eval_scripts/eval_ruler.py" \
    --model_path "$HF_CKPT" \
    --abbr "${EXP_NAME}-gs${LAST_STEP}" \
    --lengths $EVAL_LENGTHS \
    --n_per_task "$EVAL_N_PER_TASK" \
    --max_seq "$EVAL_MAX_SEQ" \
    --ruler_root "$RULER_ROOT" \
    --out_root "$EVAL_OUT_ROOT" \
    --max_new_tokens 1024
  echo "[phase:eval] done. Results under $EVAL_OUT_ROOT"
  echo "[phase:eval] compare overall vs baselines: outer-CPT=0.285, off-TTT=0.458"
}

# ---- dispatch --------------------------------------------------------------
case "$PHASE" in
  train) run_train ;;
  merge) run_merge ;;
  eval)  run_eval ;;
  all)
    run_train
    run_merge
    run_eval
    ;;
  *)
    echo "Unknown PHASE: $PHASE (use all|train|merge|eval)" >&2
    exit 1
    ;;
esac

echo "[pipeline] PHASE=$PHASE complete."
