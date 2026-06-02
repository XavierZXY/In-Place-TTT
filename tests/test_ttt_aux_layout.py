from pathlib import Path

from in_place_ttt.ttt_aux.loss import compute_ttt_aux_loss
from in_place_ttt.ttt_aux.training import accumulate_ttt_aux_grads


def test_ttt_aux_helpers_live_under_in_place_ttt_package():
    assert compute_ttt_aux_loss.__module__ == "in_place_ttt.ttt_aux.loss"
    assert accumulate_ttt_aux_grads.__module__ == "in_place_ttt.ttt_aux.training"


def test_longsft_aux_launcher_lives_under_scripts_train_longsft():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = repo_root / "scripts" / "train" / "longsft" / "run_full_swa_ttt_aux.sh"
    old_launcher = repo_root / "run_longsft_full_swa_ttt_aux.sh"

    assert launcher.is_file()
    assert not old_launcher.exists()

    text = launcher.read_text(encoding="utf-8")
    assert 'CONFIG="configs/pretrain/qwen3_longsft_full_swa_ttt_aux.yaml"' in text
    assert 'WANDB_PROJECT="${WANDB_PROJECT:-in-place-ttt}"' in text
    assert 'WANDB_NAME="${WANDB_NAME:-longsft-full-swa-ttt-aux-amp}"' in text
    assert '--train.wandb_project "$WANDB_PROJECT"' in text
    assert '--train.wandb_name "$WANDB_NAME"' in text
    assert 'REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"' in text


def test_longsft_swa_full_1to3_launcher_points_to_hybrid_config():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = repo_root / "scripts" / "train" / "longsft" / "run_swa_full_1to3_ttt_aux.sh"

    assert launcher.is_file()

    text = launcher.read_text(encoding="utf-8")
    assert 'CONFIG="configs/pretrain/qwen3_longsft_swa_full_1to3_ttt_aux.yaml"' in text
    assert 'WANDB_PROJECT="${WANDB_PROJECT:-in-place-ttt}"' in text
    assert 'WANDB_NAME="${WANDB_NAME:-longsft-v0-swa-anchor-ttt-aux-swa4096-chunk1024-64k}"' in text
    assert '--train.wandb_project "$WANDB_PROJECT"' in text
    assert '--train.wandb_name "$WANDB_NAME"' in text
    assert 'runtime_args+=(--data.train_path "$TRAIN_PATH")' in text
    assert 'runtime_args+=(--data.eval_path "$EVAL_PATH")' in text
    assert 'REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"' in text


def test_longsft_swa_window_anneal_launcher_keeps_full_swa_1to3_config():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = repo_root / "scripts" / "train" / "longsft" / "run_full_swa_1to3_ttt_aux_swa_window_anneal.sh"

    assert launcher.is_file()

    text = launcher.read_text(encoding="utf-8")
    assert 'CONFIG="configs/pretrain/qwen3_longsft_swa_full_1to3_ttt_aux.yaml"' in text
    assert 'WINDOW_SCHEDULE="${WINDOW_SCHEDULE:-4096,2048,1024}"' in text
    assert 'STAGE_STOP_STEPS="${STAGE_STOP_STEPS:-400,800}"' in text
    assert 'IFS=\',\' read -r -a WINDOWS <<< "$WINDOW_SCHEDULE"' in text
    assert 'IFS=\',\' read -r -a STOP_STEPS <<< "$STAGE_STOP_STEPS"' in text
    assert 'stop_args=(--train.stage_stop_steps "$stop_step")' in text
    assert '--model.foundation "{\\"ttt_compress_window\\": ${window_size}}"' in text


def test_longsft_v0_swa_anchor_launcher_points_to_v0_aligned_config():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = repo_root / "scripts" / "train" / "longsft" / "run_v0_swa_anchor_ttt_aux.sh"

    assert launcher.is_file()

    text = launcher.read_text(encoding="utf-8")
    assert 'CONFIG="configs/pretrain/qwen3_longsft_v0_swa_anchor_ttt_aux.yaml"' in text
    assert 'WANDB_PROJECT="${WANDB_PROJECT:-in-place-ttt}"' in text
    assert 'WANDB_NAME="${WANDB_NAME:-longsft-v0-swa-anchor-ttt-aux-swa4096-chunk1024-64k}"' in text
    assert '--train.wandb_project "$WANDB_PROJECT"' in text
    assert '--train.wandb_name "$WANDB_NAME"' in text
    assert 'runtime_args+=(--data.train_path "$TRAIN_PATH")' in text
    assert 'runtime_args+=(--data.eval_path "$EVAL_PATH")' in text
    assert 'REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"' in text
