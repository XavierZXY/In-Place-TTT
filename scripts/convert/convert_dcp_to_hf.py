#!/usr/bin/env python3
"""Convert a VeOmni DCP model checkpoint to HuggingFace safetensors."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True, help="DCP checkpoint directory, e.g. global_step_8000.")
    parser.add_argument("--config", required=True, help="Training yaml used to build the student config.")
    parser.add_argument("--output-dir", required=True, help="Destination HuggingFace checkpoint directory.")
    parser.add_argument("--save-dtype", default="bfloat16", help="Output weight dtype passed to save_model_weights.")
    parser.add_argument("--shard-size", type=int, default=5_000_000_000, help="Max shard size in bytes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"DCP checkpoint does not exist: {checkpoint_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config does not exist: {config_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    os.environ.setdefault("MODELING_BACKEND", "hf")

    # Registers local TTT Qwen/Llama configs with HuggingFace AutoConfig.
    import hf_models  # noqa: F401
    from veomni.checkpoint import ckpt_to_state_dict
    from veomni.models import build_tokenizer, save_model_weights
    from veomni.models.auto import build_config

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_args = config["model"]
    foundation = dict(model_args.get("foundation", {}))
    for key in list(foundation):
        if key.startswith("distill_"):
            foundation.pop(key)

    model_config = build_config(
        model_args.get("config_path") or model_args["model_path"],
        **foundation,
    )
    tokenizer = build_tokenizer(model_args.get("tokenizer_path") or model_args.get("config_path") or model_args["model_path"])

    print(f"Loading DCP model state from {checkpoint_path}", flush=True)
    model_state_dict = ckpt_to_state_dict(str(checkpoint_path), ckpt_manager="dcp")
    print(f"Loaded {len(model_state_dict)} tensors. Saving HF checkpoint to {output_dir}", flush=True)

    save_model_weights(
        str(output_dir),
        model_state_dict,
        save_dtype=args.save_dtype,
        shard_size=args.shard_size,
        safe_serialization=True,
        model_assets=[model_config, tokenizer],
    )
    print(f"Saved HuggingFace checkpoint to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
