# TTT KeyNorm Safe Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Qwen3 inference honor `ttt_key_norm`, then validate an isolated Stage3 smoke run without touching existing training outputs.

**Architecture:** Keep training-side behavior unchanged. Add the same optional key-normalized fast-weight path to `inference_model/hf_qwen3/modeling_qwen3.py`, guarded by `ttt_key_norm=false` by default. Run all training and eval artifacts under a new experiment name.

**Tech Stack:** PyTorch, HuggingFace-style Qwen3 model files, pytest, VeOmni DCP checkpoints, local RULER eval.

---

### Task 1: Inference KeyNorm Regression Test

**Files:**
- Modify: `tests/test_ttt_key_norm.py`

- [ ] Add an inference-side test that constructs `inference_model.hf_qwen3.modeling_qwen3.Qwen3MLP` with `ttt_key_norm=True` and asserts the module owns `ttt_key_norm`.
- [ ] Add a behavior test comparing keynorm off/on with shared weights. The first token must match the base projection, and later output must differ when keynorm is enabled.
- [ ] Run `".venv/bin/python" -m pytest "tests/test_ttt_key_norm.py" -q` and verify the new inference test fails before implementation.

### Task 2: Inference KeyNorm Implementation

**Files:**
- Modify: `inference_model/hf_qwen3/modeling_qwen3.py`

- [ ] Add optional `self.ttt_key_norm = Qwen3RMSNorm(self.intermediate_size, eps=config.rms_norm_eps)` under the existing TTT-layer setup.
- [ ] In inference `forward`, keep the existing path verbatim when keynorm is disabled.
- [ ] When keynorm is enabled, use normalized MLP key `h_norm` for fast-weight read and write, while preserving the base down projection over original `h`.
- [ ] Run `".venv/bin/python" -m pytest "tests/test_ttt_key_norm.py" -q` and verify all keynorm tests pass.

### Task 3: Isolated Stage3 Smoke

**Files:**
- No repo edits.
- New output: `outputs/qwen3-1.7b-stage3-cpt-swa3-full1-strict-c1024-keynorm-safe-v1`

- [ ] Confirm the output directory does not already exist.
- [ ] Launch the existing Stage3 wrapper with only environment overrides: `MAX_STEPS=20`, `SAVE_STEPS=20`, `EVAL_STEPS=0`, `MAX_SEQ_LEN=8192`, `USE_WANDB=false`, and `MODEL_FOUNDATION_JSON='{"ttt_key_norm": true, "ttt_lr": 0.3, "ttt_aux_loss_weight": 0.0, "ttt_train_only": true}'`.
- [ ] Watch the process to completion and record whether checkpoint `global_step_20` exists.

### Task 4: Convert and RULER Smoke

**Files:**
- No repo edits.
- New eval output: `eval/exp_analysis/ruler_results/qwen3-1.7b-stage3-cpt-swa3-full1-strict-c1024-keynorm-safe-v1-gs20-ruler-smoke`

- [ ] Convert only the new checkpoint with `eval/eval_scripts/convert_dcp_to_hf.sh`.
- [ ] Run a minimal local RULER smoke at length 4096, `n_per_task=5`.
- [ ] Read `summary_all_lengths.json` and report the overall score and any runtime issue.

### Self-Review

- The plan does not modify existing checkpoint directories.
- The implementation is guarded by an existing config default of `ttt_key_norm=false`.
- The smoke run uses a unique `EXP_NAME`, `OUTPUT_DIR`, `LOG_PREFIX`, and RULER `abbr`.
- No git commit is planned because repository instructions forbid commits unless explicitly requested.
