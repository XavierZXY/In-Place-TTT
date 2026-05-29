import yaml


def test_longsft_aux_config_matches_messages_dataset_schema():
    with open("configs/pretrain/qwen3_longsft_full_swa_ttt_aux.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data = config["data"]
    assert data["data_type"] == "conversation"
    assert data["text_keys"] == "messages"
    assert data["chat_template"] == "chatml"
