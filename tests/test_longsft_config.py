import yaml


def test_longsft_aux_config_matches_messages_dataset_schema():
    with open("configs/pretrain/qwen3_longsft_full_swa_ttt_aux.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data = config["data"]
    assert data["data_type"] == "conversation"
    assert data["text_keys"] == "messages"
    assert data["chat_template"] == "chatml"

    train = config["train"]
    assert train["wandb_project"] == "in-place-ttt"
    assert train["wandb_name"] == "longsft-full-swa-ttt-aux-amp"


def test_longsft_swa_full_1to3_config_matches_v0_anchor_layout():
    with open("configs/pretrain/qwen3_longsft_swa_full_1to3_ttt_aux.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    foundation = config["model"]["foundation"]
    full_layers = [0, 1, 7, 14, 15, 22, 24, 33, 34]
    ttt_layers = [3, 6, 12, 18, 25, 30]

    assert foundation["full_attention_layers"] == full_layers
    assert foundation["ttt_layers"] == ttt_layers
    assert set(foundation["ttt_layers"]).isdisjoint(foundation["full_attention_layers"])
    assert foundation["ttt_compress_window"] == 4096
    assert foundation["ttt_chunk"] == 1024

    data = config["data"]
    assert data["train_path"] == "/workspace/post_training_data_cloud/weilai/datasets/long-sft-mix-16k-64k/train.jsonl"
    assert data["eval_path"] == "/workspace/post_training_data_cloud/weilai/datasets/long-sft-mix-16k-64k/val.jsonl"
    assert data["max_seq_len"] == 65536

    train = config["train"]
    assert train["wandb_project"] == "in-place-ttt"
    assert train["wandb_name"] == "longsft-v0-swa-anchor-ttt-aux-swa4096-chunk1024-64k"


def test_longsft_v0_swa_anchor_config_matches_v0_layout_and_data():
    with open("configs/pretrain/qwen3_longsft_v0_swa_anchor_ttt_aux.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    foundation = config["model"]["foundation"]
    full_layers = [0, 1, 7, 14, 15, 22, 24, 33, 34]
    ttt_layers = [3, 6, 12, 18, 25, 30]

    assert foundation["full_attention_layers"] == full_layers
    assert foundation["ttt_layers"] == ttt_layers
    assert set(ttt_layers).isdisjoint(full_layers)
    assert foundation["ttt_compress_window"] == 4096
    assert foundation["ttt_chunk"] == 1024

    data = config["data"]
    assert data["train_path"] == "/workspace/post_training_data_cloud/weilai/datasets/long-sft-mix-16k-64k/train.jsonl"
    assert data["eval_path"] == "/workspace/post_training_data_cloud/weilai/datasets/long-sft-mix-16k-64k/val.jsonl"
    assert data["max_seq_len"] == 65536

    train = config["train"]
    assert train["wandb_project"] == "in-place-ttt"
    assert train["wandb_name"] == "longsft-v0-swa-anchor-ttt-aux-swa4096-chunk1024-64k"
