import yaml


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_swa1024_ttt_aux_config_matches_conversation_schema():
    config = load_config("configs/pretrain/qwen3_swa1024_ttt_aux.yaml")

    data = config["data"]
    assert data["data_type"] == "conversation"
    assert data["text_keys"] == "messages"
    assert data["chat_template"] == "chatml"

    train = config["train"]
    assert train["wandb_project"] == "in-place-ttt"
    assert train["wandb_name"] == "qwen3-swa1024-ttt-aux-c256-24k"


def test_swa3_full1_v0anchor_c2048_config_matches_layout():
    config = load_config("configs/pretrain/qwen3_swa3_full1_v0anchor_ttt_aux_c2048.yaml")

    foundation = config["model"]["foundation"]
    full_layers = [0, 1, 7, 14, 15, 22, 24, 33, 34]
    ttt_layers = [3, 6, 12, 18, 25, 30]

    assert foundation["full_attention_layers"] == full_layers
    assert foundation["ttt_layers"] == ttt_layers
    assert set(foundation["ttt_layers"]).isdisjoint(foundation["full_attention_layers"])
    assert foundation["ttt_compress_window"] == 4096
    assert foundation["ttt_chunk"] == 2048

    data = config["data"]
    assert data["data_type"] == "conversation"
    assert data["train_path"] == "/zouxiangyu/data_ttt/long-sft/train.jsonl"
    assert data["eval_path"] == "/zouxiangyu/data_ttt/long-sft/val.jsonl"
    assert data["max_seq_len"] == 32768

    train = config["train"]
    assert train["wandb_project"] == "in-place-ttt"
    assert train["wandb_name"] == "qwen3-swa3-full1-v0anchor-ttt-aux-c2048-32k"


def test_swa3_full1_v0anchor_c1024_config_matches_layout():
    config = load_config("configs/pretrain/qwen3_swa3_full1_v0anchor_ttt_aux_c1024.yaml")

    foundation = config["model"]["foundation"]
    full_layers = [0, 1, 7, 14, 15, 22, 24, 33, 34]
    ttt_layers = [3, 6, 12, 18, 25, 30]

    assert foundation["full_attention_layers"] == full_layers
    assert foundation["ttt_layers"] == ttt_layers
    assert set(ttt_layers).isdisjoint(full_layers)
    assert foundation["ttt_compress_window"] == 4096
    assert foundation["ttt_chunk"] == 1024

    data = config["data"]
    assert data["data_type"] == "conversation"
    assert data["train_path"] == "/zouxiangyu/data_ttt/long-sft/train.jsonl"
    assert data["eval_path"] == "/zouxiangyu/data_ttt/long-sft/val.jsonl"
    assert data["max_seq_len"] == 32768

    train = config["train"]
    assert train["wandb_project"] == "in-place-ttt"
    assert train["wandb_name"] == "qwen3-swa3-full1-v0anchor-ttt-aux-c1024-32k"


def test_swa3_full1_strict_config_keeps_periodic_layout():
    config = load_config("configs/pretrain/qwen3_swa3_full1_strict_ttt_aux.yaml")

    foundation = config["model"]["foundation"]
    assert foundation["full_attention_layers"] == [0, 4, 8, 12, 16, 20, 24, 28, 32]
    assert foundation["ttt_layers"] == [1, 5, 9, 13, 17, 21, 25, 29, 33]
    assert set(foundation["ttt_layers"]).isdisjoint(foundation["full_attention_layers"])
    assert foundation["ttt_chunk"] == 2048
    assert foundation["ttt_compress_window"] == 4096

    train = config["train"]
    assert train["wandb_name"] == "qwen3-swa3-full1-strict-ttt-aux-c2048-32k"


def assert_qwen3_1_7b_strict_layout(config):
    foundation = config["model"]["foundation"]
    assert foundation["full_attention_layers"] == [0, 4, 8, 12, 16, 20, 24]
    assert max(foundation["full_attention_layers"]) < 28
    assert foundation["ttt_compress_window"] > 0


def test_qwen3_1_7b_stage1_hidden_align_config_keeps_28_layer_layout():
    config = load_config("configs/pretrain/qwen3-1.7b/stage1_hidden_align_swa3_full1_strict.yaml")

    model = config["model"]
    assert model["model_path"] == "/zouxiangyu/models/Qwen/Qwen3-1.7B"
    assert model["tokenizer_path"] == "/zouxiangyu/models/Qwen/Qwen3-1.7B"
    assert_qwen3_1_7b_strict_layout(config)

    foundation = model["foundation"]
    assert foundation["ttt_mode"] is False
    assert foundation["ttt_layers"] == []
    assert foundation["ttt_aux_loss_weight"] == 0.0
    assert foundation["hidden_align_teacher_path"] == "/zouxiangyu/models/Qwen/Qwen3-1.7B"
    assert foundation["hidden_align_loss_fn"] == "nmse"
    assert foundation["hidden_align_input"] == "teacher"
    # Student RoPE must match the frozen teacher (original Qwen3-1.7B values)
    # so the alignment loss only reflects the SWA layout difference.
    assert foundation["rope_theta"] == 1000000.0
    assert foundation["max_position_embeddings"] == 40960

    # Sequences must be longer than the SWA window, otherwise sliding
    # attention degenerates into full attention and there is nothing to align.
    assert config["data"]["max_seq_len"] == 4096
    assert config["data"]["max_seq_len"] > foundation["ttt_compress_window"]
    assert 0 < config["train"]["lr_min"] <= config["train"]["lr"]
    assert config["train"]["wandb_name"] == "qwen3-1.7b-stage1-hidden-align-swa3-full1-strict-4096"


def test_qwen3_1_7b_stage2_kd_config_keeps_28_layer_layout():
    config = load_config("configs/pretrain/qwen3-1.7b/stage2_kd_swa3_full1_strict.yaml")

    model = config["model"]
    # Stage 2 initializes from the stage-1 aligned weights (HF-converted).
    assert model["model_path"].endswith("/hf_ckpt")
    assert "stage1-hidden-align" in model["model_path"]
    assert model["tokenizer_path"] == "/zouxiangyu/models/Qwen/Qwen3-1.7B"
    assert_qwen3_1_7b_strict_layout(config)

    foundation = model["foundation"]
    assert foundation["ttt_mode"] is False
    assert foundation["ttt_layers"] == []
    assert foundation["ttt_aux_loss_weight"] == 0.0
    assert foundation["distill_teacher_path"] == "/zouxiangyu/models/Qwen/Qwen3-1.7B"
    assert foundation["distill_alpha_kl"] == 1.0
    # Same RoPE as the teacher and stage 1; same SWA window as stage 1.
    assert foundation["rope_theta"] == 1000000.0
    assert foundation["max_position_embeddings"] == 40960
    assert foundation["ttt_compress_window"] == 1024

    assert config["data"]["max_seq_len"] == 4096
    assert config["data"]["max_seq_len"] > foundation["ttt_compress_window"]
    assert 0 < config["train"]["lr_min"] <= config["train"]["lr"]
    assert config["train"]["wandb_name"] == "qwen3-1.7b-stage2-kd-swa3-full1-strict-4096"


def test_qwen3_1_7b_stage3_cpt_config_keeps_28_layer_layout():
    config = load_config("configs/pretrain/qwen3-1.7b/stage3_cpt_swa3_full1_strict.yaml")

    model = config["model"]
    # Stage 3 initializes from the stage-2 KD weights (HF-converted).
    assert model["model_path"].endswith("/hf_ckpt")
    assert "stage2-kd" in model["model_path"]
    assert model["tokenizer_path"] == "/zouxiangyu/models/Qwen/Qwen3-1.7B"
    assert_qwen3_1_7b_strict_layout(config)
    foundation = model["foundation"]
    assert foundation["ttt_mode"] is True
    assert foundation["ttt_layers"] == [1, 5, 9, 13, 17, 21, 25]
    assert max(foundation["ttt_layers"]) < 28
    assert set(foundation["ttt_layers"]).isdisjoint(foundation["full_attention_layers"])
    # Same RoPE and SWA window as stages 1/2 — the frozen backbone cannot
    # compensate for an architecture change.
    assert foundation["rope_theta"] == 1000000.0
    assert foundation["max_position_embeddings"] == 40960
    assert foundation["ttt_compress_window"] == 1024
    assert foundation["ttt_chunk"] == foundation["ttt_compress_window"]
    # Stage 3 trains only the TTT modules.
    assert foundation["ttt_train_only"] is True

    assert config["data"]["max_seq_len"] > foundation["ttt_compress_window"]
    # Eval extrapolates beyond the training length.
    assert config["data"]["eval_max_seq_len"] > config["data"]["max_seq_len"]

    train = config["train"]
    assert 0 < train["lr_min"] <= train["lr"]
    assert train["wandb_name"] == "qwen3-1.7b-stage3-cpt-swa3-full1-strict-c1024-16k"
