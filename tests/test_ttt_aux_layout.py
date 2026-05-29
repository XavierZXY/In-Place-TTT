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
    assert 'REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"' in text
