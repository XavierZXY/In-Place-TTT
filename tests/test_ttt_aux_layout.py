import os
import subprocess
from pathlib import Path

from in_place_ttt.ttt_aux.loss import compute_ttt_aux_loss
from in_place_ttt.ttt_aux.training import accumulate_ttt_aux_grads


def read_script(*parts):
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root.joinpath(*parts)
    assert path.is_file()
    return path.read_text(encoding="utf-8")


def repo_path(*parts):
    return Path(__file__).resolve().parents[1].joinpath(*parts)


def test_ttt_aux_helpers_live_under_in_place_ttt_package():
    assert compute_ttt_aux_loss.__module__ == "in_place_ttt.ttt_aux.loss"
    assert accumulate_ttt_aux_grads.__module__ == "in_place_ttt.ttt_aux.training"


def test_pretrain_template_owns_common_torchrun_logic():
    text = read_script("scripts", "train", "longsft", "run_pretrain_template.sh")

    assert 'CONFIG="${CONFIG:-configs/pretrain/qwen3_swa3_full1_v0anchor_ttt_aux_c2048.yaml}"' in text
    assert 'TASK_SCRIPT="${TASK_SCRIPT:-tasks/train_torch.py}"' in text
    assert '--train.wandb_project "$WANDB_PROJECT"' in text
    assert '--train.wandb_name "$WANDB_NAME"' in text
    assert 'append_runtime_arg "${TRAIN_PATH:-}" --data.train_path' in text
    assert 'append_runtime_arg "${EVAL_PATH:-}" --data.eval_path' in text
    assert 'append_runtime_arg "${LR_MIN:-}" --train.lr_min' in text
    assert 'append_foundation_string "ttt_monitor_output_delta_target"' in text
    assert 'REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"' in text


def test_swa1024_aux_launcher_is_thin_wrapper():
    text = read_script("scripts", "train", "longsft", "run_swa1024_ttt_aux.sh")

    assert 'export CONFIG="${CONFIG:-configs/pretrain/qwen3_swa1024_ttt_aux.yaml}"' in text
    assert 'export EXP_NAME="${EXP_NAME:-qwen3-swa1024-ttt-aux-c256-24k}"' in text
    assert 'bash "$SCRIPT_DIR/run_pretrain_template.sh" "$@"' in text


def test_swa3_full1_v0anchor_launchers_use_architecture_names():
    c1024_text = read_script(
        "scripts", "train", "longsft", "run_swa3_full1_v0anchor_ttt_aux_c1024.sh"
    )
    c2048_text = read_script(
        "scripts", "train", "longsft", "run_swa3_full1_v0anchor_ttt_aux_c2048.sh"
    )

    assert 'export CONFIG="${CONFIG:-configs/pretrain/qwen3_swa3_full1_v0anchor_ttt_aux_c1024.yaml}"' in c1024_text
    assert 'export EXP_NAME="${EXP_NAME:-qwen3-swa3-full1-v0anchor-ttt-aux-c1024-32k}"' in c1024_text
    assert 'bash "$SCRIPT_DIR/run_pretrain_template.sh" "$@"' in c1024_text

    assert 'export CONFIG="${CONFIG:-configs/pretrain/qwen3_swa3_full1_v0anchor_ttt_aux_c2048.yaml}"' in c2048_text
    assert 'export EXP_NAME="${EXP_NAME:-qwen3-swa3-full1-v0anchor-ttt-aux-c2048-32k}"' in c2048_text
    assert 'bash "$SCRIPT_DIR/run_pretrain_template.sh" "$@"' in c2048_text


def test_swa3_full1_v0anchor_window_anneal_launcher_uses_hybrid_config():
    text = read_script("scripts", "train", "longsft", "run_swa3_full1_v0anchor_window_anneal.sh")

    assert 'CONFIG="${CONFIG:-configs/pretrain/qwen3_swa3_full1_v0anchor_ttt_aux_c2048.yaml}"' in text
    assert 'EXP_NAME="${EXP_NAME:-qwen3-swa3-full1-v0anchor-window-anneal}"' in text
    assert 'WINDOW_SCHEDULE="${WINDOW_SCHEDULE:-4096,2048,1024}"' in text
    assert 'STAGE_STOP_STEPS="${STAGE_STOP_STEPS:-400,800}"' in text
    assert 'TTT_COMPRESS_WINDOW="$window_size"' in text
    assert 'bash "$SCRIPT_DIR/run_pretrain_template.sh" "$@"' in text


def test_halo_launchers_use_task_specific_entrypoints():
    hidden_align_text = read_script(
        "scripts", "train", "longsft", "run_swa3_full1_v0anchor_halo_hidden_align.sh"
    )
    kd_text = read_script("scripts", "train", "longsft", "run_swa3_full1_v0anchor_halo_kd.sh")

    assert 'export TASK_SCRIPT="${TASK_SCRIPT:-tasks/train_torch_halo_hidden_align.py}"' in hidden_align_text
    assert 'export EXP_NAME="${EXP_NAME:-qwen3-swa3-full1-v0anchor-halo-hidden-align-c1024-512}"' in hidden_align_text
    assert 'export HIDDEN_ALIGN_TEACHER_PATH="$TEACHER_MODEL_PATH"' in hidden_align_text

    assert 'export TASK_SCRIPT="${TASK_SCRIPT:-tasks/train_torch_halo_kd.py}"' in kd_text
    assert 'export EXP_NAME="${EXP_NAME:-qwen3-swa3-full1-v0anchor-halo-kd-c1024-4k}"' in kd_text
    assert 'export DISTILL_TEACHER_PATH="$TEACHER_MODEL_PATH"' in kd_text


def test_qwen3_1_7b_stage_launchers_use_child_configs_and_parent_template():
    stage1_text = read_script(
        "scripts", "train", "longsft", "qwen3-1.7b", "run_stage1_hidden_align_swa3_full1_strict.sh"
    )
    stage2_text = read_script(
        "scripts", "train", "longsft", "qwen3-1.7b", "run_stage2_kd_swa3_full1_strict.sh"
    )
    stage3_text = read_script(
        "scripts", "train", "longsft", "qwen3-1.7b", "run_stage3_cpt_swa3_full1_strict.sh"
    )

    assert 'export TASK_SCRIPT="${TASK_SCRIPT:-tasks/train_torch_halo_hidden_align.py}"' in stage1_text
    assert 'export CONFIG="${CONFIG:-configs/pretrain/qwen3-1.7b/stage1_hidden_align_swa3_full1_strict.yaml}"' in stage1_text
    assert 'export HIDDEN_ALIGN_TEACHER_PATH="$TEACHER_MODEL_PATH"' in stage1_text

    assert 'export TASK_SCRIPT="${TASK_SCRIPT:-tasks/train_torch_halo_kd.py}"' in stage2_text
    assert 'export CONFIG="${CONFIG:-configs/pretrain/qwen3-1.7b/stage2_kd_swa3_full1_strict.yaml}"' in stage2_text
    assert 'export DISTILL_TEACHER_PATH="$TEACHER_MODEL_PATH"' in stage2_text
    assert 'STAGE2_INIT_FROM="${STAGE2_INIT_FROM:-stage1}"' in stage2_text
    assert 'BASE_MODEL_PATH="${BASE_MODEL_PATH:-/zouxiangyu/models/Qwen/Qwen3-1.7B}"' in stage2_text
    assert 'export MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH}"' in stage2_text
    assert 'Unsupported STAGE2_INIT_FROM' in stage2_text
    assert 'STAGE1_EXP_NAME="${STAGE1_EXP_NAME:-qwen3-1.7b-stage1-hidden-align-swa3-full1-strict-4096}"' in stage2_text
    assert 'global_step_${STAGE1_GLOBAL_STEP}' in stage2_text
    # Stage transitions go through HF-converted weights, not resume-style loading.
    assert "merge_dcp_to_hf.py" in stage2_text
    assert "LOAD_CHECKPOINT_PATH" not in stage2_text

    assert 'export CONFIG="${CONFIG:-configs/pretrain/qwen3-1.7b/stage3_cpt_swa3_full1_strict.yaml}"' in stage3_text
    assert 'export EXP_NAME="${EXP_NAME:-qwen3-1.7b-stage3-cpt-swa3-full1-strict-c1024-16k}"' in stage3_text
    assert 'STAGE2_EXP_NAME="${STAGE2_EXP_NAME:-qwen3-1.7b-stage2-kd-swa3-full1-strict-4096}"' in stage3_text
    assert 'global_step_${STAGE2_GLOBAL_STEP}' in stage3_text
    assert "merge_dcp_to_hf.py" in stage3_text
    assert "LOAD_CHECKPOINT_PATH" not in stage3_text

    assert 'bash "$SCRIPT_DIR/../run_pretrain_template.sh" "$@"' in stage1_text
    assert 'bash "$SCRIPT_DIR/../run_pretrain_template.sh" "$@"' in stage2_text
    assert 'bash "$SCRIPT_DIR/../run_pretrain_template.sh" "$@"' in stage3_text


def test_qwen3_1_7b_stage2_launcher_can_initialize_from_base_model(tmp_path):
    script = repo_path(
        "scripts", "train", "longsft", "qwen3-1.7b", "run_stage2_kd_swa3_full1_strict.sh"
    )
    env = os.environ.copy()
    for key in (
        "DISTILL_TEACHER_PATH",
        "EXP_NAME",
        "LOAD_CHECKPOINT_PATH",
        "MODEL_PATH",
        "OUTPUT_DIR",
        "STAGE1_GLOBAL_STEP",
        "STAGE1_HF_CKPT",
        "TEACHER_MODEL_PATH",
        "WANDB_NAME",
    ):
        env.pop(key, None)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "LOG_FILE": str(tmp_path / "stage2-base-dryrun.log"),
            "STAGE2_INIT_FROM": "base",
            "TORCHRUN": "/bin/echo",
            "TRACE": "0",
        }
    )

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo_path(),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "No stage-1 checkpoints found" not in result.stderr
    assert "--model.model_path /zouxiangyu/models/Qwen/Qwen3-1.7B" in result.stdout
