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


def test_longsft_swa_full_1to3_config_restricts_ttt_to_swa_layers():
    with open("configs/pretrain/qwen3_longsft_swa_full_1to3_ttt_aux.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    foundation = config["model"]["foundation"]
    full_layers = list(range(0, 36, 4))
    swa_layers = [idx for idx in range(36) if idx not in full_layers]

    assert foundation["ttt_compress_window"] == 1024
    assert foundation["ttt_layers"] == swa_layers
    assert foundation["full_attention_layers"] == full_layers
    assert set(foundation["ttt_layers"]).isdisjoint(foundation["full_attention_layers"])

    train = config["train"]
    assert train["wandb_project"] == "in-place-ttt"
    assert train["wandb_name"] == "longsft-full-swa-1to3-swa-ttt-aux-amp"

