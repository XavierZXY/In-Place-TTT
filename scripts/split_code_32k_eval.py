#!/usr/bin/env python3
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class SplitStats:
    input_path: str
    train_output: str
    eval_output: str
    eval_examples: int
    eval_stride: int
    rows_read: int = 0
    train_rows: int = 0
    eval_rows: int = 0


def _ensure_writable(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)


def _is_eval_row(row_number: int, eval_rows: int, eval_examples: int, eval_stride: int) -> bool:
    return eval_rows < eval_examples and row_number % eval_stride == 0


def split_jsonl(
    input_path: Path,
    train_output: Path,
    eval_output: Path,
    manifest_path: Path,
    eval_examples: int,
    eval_stride: int,
    overwrite: bool,
) -> SplitStats:
    if eval_examples <= 0:
        raise ValueError("eval_examples must be positive.")
    if eval_stride <= 0:
        raise ValueError("eval_stride must be positive.")
    if input_path == train_output or input_path == eval_output:
        raise ValueError("input_path must be different from output paths.")

    _ensure_writable(train_output, overwrite)
    _ensure_writable(eval_output, overwrite)
    _ensure_writable(manifest_path, overwrite)

    stats = SplitStats(
        input_path=str(input_path),
        train_output=str(train_output),
        eval_output=str(eval_output),
        eval_examples=eval_examples,
        eval_stride=eval_stride,
    )
    with input_path.open(encoding="utf-8") as source:
        with train_output.open("w", encoding="utf-8") as train_handle:
            with eval_output.open("w", encoding="utf-8") as eval_handle:
                for row_number, line in enumerate(source, start=1):
                    if not line.strip():
                        continue
                    stats.rows_read += 1
                    if _is_eval_row(row_number, stats.eval_rows, eval_examples, eval_stride):
                        eval_handle.write(line)
                        stats.eval_rows += 1
                    else:
                        train_handle.write(line)
                        stats.train_rows += 1

    manifest_path.write_text(json.dumps(asdict(stats), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a plaintext JSONL file into train and held-out eval JSONL files.")
    parser.add_argument("--input", type=Path, default=Path("data/code_32k_20b/plaintext.jsonl"))
    parser.add_argument("--train-output", type=Path, default=Path("data/code_32k_20b/train_plaintext.jsonl"))
    parser.add_argument("--eval-output", type=Path, default=Path("data/code_32k_20b/eval_plaintext.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/code_32k_20b/split_manifest.json"))
    parser.add_argument("--eval-examples", type=int, default=20_000)
    parser.add_argument("--eval-stride", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = split_jsonl(
        input_path=args.input,
        train_output=args.train_output,
        eval_output=args.eval_output,
        manifest_path=args.manifest,
        eval_examples=args.eval_examples,
        eval_stride=args.eval_stride,
        overwrite=args.overwrite,
    )
    print(
        f"read={stats.rows_read} train={stats.train_rows} eval={stats.eval_rows} "
        f"train_output={stats.train_output} eval_output={stats.eval_output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
