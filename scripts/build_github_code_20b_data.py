#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import collections
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from transformers import AutoTokenizer


DEFAULT_TARGET_TOKENS = 20_000_000_000
DEFAULT_DATASET = "codeparrot/github-code"
DEFAULT_LANGUAGES = ("Python", "JavaScript", "TypeScript", "Go", "Rust", "Java")
DEFAULT_LICENSES = ("mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc")
SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}


@dataclass
class BuildStats:
    target_tokens: int
    total_tokens: int = 0
    total_examples: int = 0
    rows_seen: int = 0
    rows_written: int = 0
    total_chars: int = 0
    started_at: float = 0.0
    updated_at: float = 0.0
    source_config_index: int = 0
    source_config_name: str = ""
    source_config_rows_seen: int = 0


def normalize_language(value: Any) -> str:
    text = str(value or "").strip()
    return {"GO": "Go", "JS": "JavaScript", "TS": "TypeScript"}.get(text.upper(), text)


def normalize_license(value: Any) -> str:
    return str(value or "").strip().lower()


def language_family(language: str) -> str:
    normalized = normalize_language(language)
    if normalized == "Python":
        return "python"
    if normalized in {"JavaScript", "TypeScript"}:
        return "typescript_javascript"
    if normalized == "Go":
        return "go"
    if normalized == "Rust":
        return "rust"
    if normalized == "Java":
        return "jvm"
    return "other"


def dataset_language_name(language: str) -> str:
    normalized = normalize_language(language)
    return "GO" if normalized == "Go" else normalized


def dataset_config_names(languages: Iterable[str], licenses: Iterable[str]) -> list[str]:
    return [f"{dataset_language_name(language)}-{license_}" for language in languages for license_ in licenses]


def is_skipped_path(path: str) -> bool:
    return any(part in SKIP_DIRS for part in Path(path).parts)


def accept_row(
    row: dict[str, Any],
    allowed_languages: set[str],
    allowed_licenses: set[str],
    min_chars: int,
    max_chars: int,
) -> str:
    repo = row.get("repo_name")
    path = row.get("path")
    code = row.get("code") or row.get("content")
    if not isinstance(repo, str) or not isinstance(path, str) or not isinstance(code, str):
        return "missing_field"
    if is_skipped_path(path):
        return "skipped_dir"
    if "\x00" in code or not code.strip():
        return "binary_or_empty"
    if len(code) < min_chars:
        return "below_min_chars"
    if len(code) > max_chars:
        return "above_max_chars"
    if normalize_language(row.get("language")) not in allowed_languages:
        return "language"
    if allowed_licenses and normalize_license(row.get("license")) not in allowed_licenses:
        return "license"
    return "accepted"


def build_content(row: dict[str, Any]) -> str:
    repo = row["repo_name"]
    path = row["path"]
    code = (row.get("code") or row.get("content")).rstrip()
    return f'<repo name="{repo}" task="file_completion">\n<file path="{path}">\n{code}\n</file>\n</repo>'


def build_episode(row: dict[str, Any], token_count: int) -> str:
    content = build_content(row)
    metadata = {
        "repo": row["repo_name"],
        "path": row["path"],
        "kind": "file_completion",
        "chars": len(content),
        "tokens": token_count,
        "files": 1,
        "source": DEFAULT_DATASET,
        "language": normalize_language(row.get("language")),
        "language_family": language_family(str(row.get("language"))),
        "license": normalize_license(row.get("license")),
    }
    return json.dumps({"content_split": content, "metadata": metadata}, ensure_ascii=False)


def load_existing_state(
    manifest_path: Path,
    target_tokens: int,
) -> tuple[BuildStats, collections.Counter[str], collections.Counter[str], collections.Counter[str]]:
    if not manifest_path.exists():
        return (
            BuildStats(target_tokens=target_tokens, started_at=time.time()),
            collections.Counter(),
            collections.Counter(),
            collections.Counter(),
        )
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    stats = BuildStats(target_tokens=target_tokens)
    for key in asdict(stats):
        if key in data and key != "target_tokens":
            setattr(stats, key, data[key])
    if "source_config_rows_seen" not in data and stats.rows_seen:
        stats.source_config_rows_seen = stats.rows_seen
    if "source_config_name" not in data:
        stats.source_config_name = "all-all"
    if not stats.started_at:
        stats.started_at = time.time()
    return (
        stats,
        collections.Counter(data.get("skipped_counts", {})),
        collections.Counter(data.get("language_counts", {})),
        collections.Counter(data.get("license_counts", {})),
    )


def write_manifest(
    manifest_path: Path,
    stats: BuildStats,
    skipped_counts: collections.Counter[str],
    language_counts: collections.Counter[str],
    license_counts: collections.Counter[str],
) -> None:
    stats.updated_at = time.time()
    payload = asdict(stats) | {
        "tokens_B": stats.total_tokens / 1_000_000_000,
        "target_tokens_B": stats.target_tokens / 1_000_000_000,
        "progress": stats.total_tokens / stats.target_tokens if stats.target_tokens else 0,
        "skipped_counts": dict(skipped_counts),
        "language_counts": dict(language_counts),
        "license_counts": dict(license_counts),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stream_dataset_config(dataset: str, config_name: str) -> Iterable[dict[str, Any]]:
    if config_name == "all-all":
        yield from load_dataset(
            dataset,
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        return
    yield from load_dataset(
        dataset,
        name=config_name,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream public GitHub code into VeOmni plaintext JSONL.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--tokenizer", default="/zouxiangyu/models/Qwen/Qwen3-4B")
    parser.add_argument("--output", type=Path, default=Path("data/code_32k_20b/plaintext.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/code_32k_20b/manifest.json"))
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    parser.add_argument("--languages", default=",".join(DEFAULT_LANGUAGES))
    parser.add_argument("--licenses", default=",".join(DEFAULT_LICENSES))
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=120_000)
    parser.add_argument("--report-every", type=int, default=1_000)
    parser.add_argument("--max-rows", type=int, default=0, help="Debug limit for rows seen before stopping.")
    parser.add_argument("--resume", action="store_true", help="Skip rows already recorded in the manifest.")
    parser.add_argument(
        "--use-dataset-configs",
        action="store_true",
        help="Iterate language-license builder configs. This avoids post-filtering but scans the source once per config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allowed_languages = set(parse_csv(args.languages))
    requested_languages = parse_csv(args.languages)
    requested_licenses = tuple(license_.lower() for license_ in parse_csv(args.licenses))
    allowed_licenses = set(requested_licenses)
    configs = dataset_config_names(requested_languages, requested_licenses) if args.use_dataset_configs else ["all-all"]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    if args.resume:
        stats, skipped_counts, language_counts, license_counts = load_existing_state(args.manifest, args.target_tokens)
    else:
        stats = BuildStats(target_tokens=args.target_tokens, started_at=time.time())
        skipped_counts = collections.Counter()
        language_counts = collections.Counter()
        license_counts = collections.Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.resume and args.output.exists() else "w"
    resume_config_index = stats.source_config_index if args.resume else -1
    resume_config_rows_seen = stats.source_config_rows_seen if args.resume else 0
    with args.output.open(mode, encoding="utf-8") as handle:
        for config_index, config_name in enumerate(configs):
            if args.resume and config_index < resume_config_index:
                continue
            stats.source_config_index = config_index
            stats.source_config_name = config_name
            for config_row_index, row in enumerate(stream_dataset_config(args.dataset, config_name), start=1):
                if args.resume and config_index == resume_config_index and config_row_index <= resume_config_rows_seen:
                    continue
                stats.rows_seen += 1
                stats.source_config_rows_seen = config_row_index
                reason = accept_row(row, allowed_languages, allowed_licenses, args.min_chars, args.max_chars)
                if reason != "accepted":
                    skipped_counts[reason] += 1
                else:
                    content = build_content(row)
                    token_count = len(tokenizer.encode(content, add_special_tokens=False))
                    handle.write(build_episode(row, token_count) + "\n")
                    stats.total_tokens += token_count
                    stats.total_examples += 1
                    stats.rows_written += 1
                    stats.total_chars += len(content)
                    language_counts[language_family(str(row.get("language")))] += token_count
                    license_counts[normalize_license(row.get("license"))] += token_count

                if stats.rows_seen % args.report_every == 0:
                    handle.flush()
                    os.fsync(handle.fileno())
                    write_manifest(args.manifest, stats, skipped_counts, language_counts, license_counts)
                    print(
                        f"config={config_name} config_rows_seen={stats.source_config_rows_seen} "
                        f"rows_seen={stats.rows_seen} examples={stats.total_examples} "
                        f"tokens={stats.total_tokens} progress={stats.total_tokens / stats.target_tokens:.6%}",
                        flush=True,
                    )
                if stats.total_tokens >= args.target_tokens:
                    break
                if args.max_rows and stats.rows_seen >= args.max_rows:
                    break
            if stats.total_tokens >= args.target_tokens or (args.max_rows and stats.rows_seen >= args.max_rows):
                break
            stats.source_config_index = config_index + 1
            stats.source_config_name = ""
            stats.source_config_rows_seen = 0
            handle.flush()
            os.fsync(handle.fileno())
            write_manifest(args.manifest, stats, skipped_counts, language_counts, license_counts)

        handle.flush()
        os.fsync(handle.fileno())
        write_manifest(args.manifest, stats, skipped_counts, language_counts, license_counts)
        print(
            f"Wrote {stats.total_examples} examples and {stats.total_tokens} tokens "
            f"({stats.total_tokens / 1_000_000_000:.6f}B) to {args.output}",
            flush=True,
        )


if __name__ == "__main__":
    main()
