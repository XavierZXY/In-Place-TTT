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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


DEFAULT_SUFFIXES = {
    ".py",
    ".pyi",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".mjs",
    ".cjs",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".md",
}

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


@dataclass(frozen=True)
class Episode:
    content: str
    repo: str
    kind: str
    chars: int
    files: int
    source: str = "local_repo"
    language_family: str = "mixed"
    length_bucket: str = "unknown"


@dataclass(frozen=True)
class FileRecord:
    repo: str
    path: str
    content: str
    source: str


def _is_test_path(path: Path) -> bool:
    lowered_parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    return (
        "test" in lowered_parts
        or "tests" in lowered_parts
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name.endswith(".test.ts")
        or name.endswith(".test.tsx")
        or name.endswith(".spec.ts")
        or name.endswith(".spec.tsx")
    )


def _file_priority(path: Path) -> tuple[int, str]:
    name = path.name.lower()
    if name.startswith("readme"):
        return (0, str(path))
    if name in {"pyproject.toml", "package.json", "tsconfig.json", "setup.py", "requirements.txt"}:
        return (1, str(path))
    if _is_test_path(path):
        return (3, str(path))
    return (2, str(path))


def classify_language_family(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".py", ".pyi"}:
        return "python"
    if suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
        return "typescript_javascript"
    if suffix == ".go":
        return "go"
    if suffix == ".rs":
        return "rust"
    if suffix in {".java", ".kt", ".kts", ".scala"}:
        return "jvm"
    if suffix in {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp"}:
        return "cpp"
    if suffix in {".md", ".rst", ".txt"}:
        return "docs"
    if suffix in {".json", ".toml", ".yaml", ".yml"}:
        return "config"
    return "other"


def dominant_language_family(paths: Iterable[str | Path]) -> str:
    counts: collections.Counter[str] = collections.Counter()
    for path in paths:
        family = classify_language_family(path)
        if family not in {"docs", "config", "other"}:
            counts[family] += 1
    if counts:
        return counts.most_common(1)[0][0]
    families = [classify_language_family(path) for path in paths]
    return families[0] if families else "mixed"


def length_bucket(chars: int) -> str:
    if chars < 8_000:
        return "lt_8k_chars"
    if chars < 16_000:
        return "8k_16k_chars"
    if chars < 32_000:
        return "16k_32k_chars"
    return "gt_32k_chars"


def iter_repo_files(repo_dir: Path, include_suffixes: set[str]) -> list[Path]:
    files: list[Path] = []
    for path in repo_dir.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(repo_dir).parts):
            continue
        if path.suffix.lower() not in include_suffixes:
            continue
        files.append(path.relative_to(repo_dir))
    return sorted(files, key=_file_priority)


def _read_text(path: Path, max_file_chars: int) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None
    except OSError:
        return None
    if "\x00" in text:
        return None
    return text[:max_file_chars]


def _append_file_block(parts: list[str], repo_dir: Path, rel_path: Path, max_file_chars: int) -> bool:
    text = _read_text(repo_dir / rel_path, max_file_chars)
    if text is None or not text.strip():
        return False
    parts.append(f'<file path="{rel_path.as_posix()}">\n{text.rstrip()}\n</file>')
    return True


def _truncate_parts(parts: list[str], max_chars: int) -> str:
    output: list[str] = []
    used = 0
    for part in parts:
        added = len(part) + 2
        if output and used + added > max_chars:
            break
        output.append(part)
        used += added
    return "\n\n".join(output)


def build_repo_episode(
    repo_dir: Path,
    include_suffixes: set[str] | None = None,
    max_chars: int = 120_000,
    max_file_chars: int = 40_000,
) -> Episode | None:
    include_suffixes = include_suffixes or DEFAULT_SUFFIXES
    rel_files = iter_repo_files(repo_dir, include_suffixes)
    parts = [f'<repo name="{repo_dir.name}" task="repo_completion">']
    files = 0
    for rel_path in rel_files:
        if _append_file_block(parts, repo_dir, rel_path, max_file_chars):
            files += 1
    parts.append("</repo>")
    if files == 0:
        return None
    content = _truncate_parts(parts, max_chars)
    return Episode(
        content=content,
        repo=repo_dir.name,
        kind="repo_completion",
        chars=len(content),
        files=files,
        language_family=dominant_language_family(str(path) for path in rel_files),
        length_bucket=length_bucket(len(content)),
    )


def build_test_conditioned_episode(
    repo_dir: Path,
    include_suffixes: set[str] | None = None,
    max_chars: int = 120_000,
    max_file_chars: int = 40_000,
) -> Episode | None:
    include_suffixes = include_suffixes or DEFAULT_SUFFIXES
    rel_files = iter_repo_files(repo_dir, include_suffixes)
    test_files = [path for path in rel_files if _is_test_path(path)]
    target_files = [path for path in rel_files if path not in test_files and path.suffix.lower() in {".py", ".ts", ".tsx", ".js", ".jsx"}]
    if not test_files or not target_files:
        return None

    target = target_files[0]
    parts = [f'<repo name="{repo_dir.name}" task="test_conditioned_implementation">', "<repo_context>"]
    files = 0
    for rel_path in rel_files:
        if rel_path == target or _is_test_path(rel_path):
            continue
        if _append_file_block(parts, repo_dir, rel_path, max_file_chars):
            files += 1
    parts.append("</repo_context>")
    parts.append("<tests>")
    for rel_path in test_files:
        if _append_file_block(parts, repo_dir, rel_path, max_file_chars):
            files += 1
    parts.append("</tests>")
    target_text = _read_text(repo_dir / target, max_file_chars)
    if target_text is None or not target_text.strip():
        return None
    parts.append(f'<target_file path="{target.as_posix()}">\n{target_text.rstrip()}\n</target_file>')
    parts.append("</repo>")
    files += 1
    content = _truncate_parts(parts, max_chars)
    return Episode(
        content=content,
        repo=repo_dir.name,
        kind="test_conditioned_implementation",
        chars=len(content),
        files=files,
        language_family=dominant_language_family(str(path) for path in rel_files),
        length_bucket=length_bucket(len(content)),
    )


def iter_repos(root: Path) -> Iterable[Path]:
    for path in sorted(root.iterdir()):
        if path.is_dir() and path.name not in SKIP_DIRS:
            yield path


def _first_present(row: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def _is_usable_text(text: str, max_chars_per_file: int) -> bool:
    return bool(text.strip()) and "\x00" not in text and len(text) <= max_chars_per_file


def _is_skipped_record_path(path: str) -> bool:
    return any(part in SKIP_DIRS for part in Path(path).parts)


def file_record_from_row(row: dict[str, Any], source_name: str) -> FileRecord | None:
    repo = _first_present(row, ("repo_name", "repository", "max_stars_repo_name", "repo"))
    path = _first_present(row, ("path", "file_path", "filepath", "relative_path"))
    content = _first_present(row, ("content", "text", "blob", "code"))
    if not isinstance(repo, str) or not isinstance(path, str) or not isinstance(content, str):
        return None
    return FileRecord(repo=repo, path=path, content=content, source=source_name)


def records_from_jsonl(path: Path, source_name: str) -> Iterable[FileRecord]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = file_record_from_row(json.loads(line), source_name)
            if record is not None:
                yield record


def records_from_parquet(path: Path, source_name: str) -> Iterable[FileRecord]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Reading Parquet sources requires pyarrow.") from exc

    table = pq.read_table(path)
    for row in table.to_pylist():
        record = file_record_from_row(row, source_name)
        if record is not None:
            yield record


def records_from_source(source: dict[str, Any]) -> Iterable[FileRecord]:
    path = Path(source["path"]).expanduser()
    if not path.exists():
        return
    source_name = str(source.get("name", path.stem))
    source_format = str(source.get("format", path.suffix.lstrip(".") or "jsonl")).lower()
    if source_format in {"jsonl", "json"}:
        yield from records_from_jsonl(path, source_name)
        return
    if source_format == "parquet":
        yield from records_from_parquet(path, source_name)
        return
    raise ValueError(f"Unsupported file record source format: {source_format}")


def group_records_by_repo(records: Iterable[FileRecord]) -> dict[str, list[FileRecord]]:
    grouped: dict[str, list[FileRecord]] = {}
    for record in records:
        grouped.setdefault(record.repo, []).append(record)
    return grouped


def _append_record_block(parts: list[str], record: FileRecord) -> None:
    parts.append(f'<file path="{record.path}">\n{record.content.rstrip()}\n</file>')


def build_repo_episode_from_records(
    repo: str,
    records: Iterable[FileRecord],
    kind: str = "repo_completion",
    max_chars: int = 120_000,
) -> Episode:
    ordered_records = sorted(records, key=lambda record: _file_priority(Path(record.path)))
    parts = [f'<repo name="{repo}" task="{kind}">']
    for record in ordered_records:
        _append_record_block(parts, record)
    parts.append("</repo>")
    content = _truncate_parts(parts, max_chars)
    return Episode(
        content=content,
        repo=repo,
        kind=kind,
        chars=len(content),
        files=len(ordered_records),
        source=ordered_records[0].source if ordered_records else "unknown",
        language_family=dominant_language_family(record.path for record in ordered_records),
        length_bucket=length_bucket(len(content)),
    )


def iter_valid_records(
    records: Iterable[FileRecord],
    include_suffixes: set[str],
    max_chars_per_file: int,
    skipped_counts: collections.Counter[str],
) -> Iterable[FileRecord]:
    for record in records:
        if _is_skipped_record_path(record.path):
            skipped_counts["skipped_dir"] += 1
            continue
        if not _is_usable_text(record.content, max_chars_per_file):
            skipped_counts["binary_or_empty"] += 1
            continue
        if Path(record.path).suffix.lower() not in include_suffixes:
            skipped_counts["unsupported_suffix"] += 1
            continue
        yield record


def build_issue_patch_episode(
    row: dict[str, Any],
    source_name: str,
    max_chars: int = 120_000,
    max_chars_per_file: int = 40_000,
) -> Episode | None:
    repo = _first_present(row, ("repo_name", "repository", "repo", "max_stars_repo_name"))
    issue = _first_present(row, ("issue", "problem_statement", "instruction", "prompt"))
    patch = _first_present(row, ("patch", "diff", "solution_patch"))
    base_files = _first_present(row, ("base_files", "files", "context_files")) or {}
    if not isinstance(repo, str) or not isinstance(issue, str) or not isinstance(patch, str):
        return None
    if not isinstance(base_files, dict):
        return None

    file_records: list[FileRecord] = []
    for path, content in base_files.items():
        if isinstance(path, str) and isinstance(content, str) and _is_usable_text(content, max_chars_per_file):
            file_records.append(FileRecord(repo=repo, path=path, content=content, source=source_name))

    parts = [f'<repo name="{repo}" task="issue_patch">', f"<issue>\n{issue.rstrip()}\n</issue>", "<repo_context>"]
    for record in sorted(file_records, key=lambda record: _file_priority(Path(record.path))):
        _append_record_block(parts, record)
    parts.extend(["</repo_context>", f"<patch>\n{patch.rstrip()}\n</patch>", "</repo>"])
    content = _truncate_parts(parts, max_chars)
    return Episode(
        content=content,
        repo=repo,
        kind="issue_patch",
        chars=len(content),
        files=len(file_records),
        source=source_name,
        language_family=dominant_language_family(record.path for record in file_records),
        length_bucket=length_bucket(len(content)),
    )


def issue_patch_rows_from_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def issue_patch_rows_from_source(source: dict[str, Any]) -> Iterable[dict[str, Any]]:
    path = Path(source["path"]).expanduser()
    if not path.exists():
        return
    source_format = str(source.get("format", path.suffix.lstrip(".") or "jsonl")).lower()
    if source_format in {"jsonl", "json"}:
        yield from issue_patch_rows_from_jsonl(path)
        return
    if source_format == "parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Reading Parquet sources requires pyarrow.") from exc
        yield from pq.read_table(path).to_pylist()
        return
    raise ValueError(f"Unsupported issue patch source format: {source_format}")


def write_manifest(
    episodes: Iterable[Episode],
    skipped_counts: collections.Counter[str],
    manifest_path: Path,
) -> None:
    episode_list = list(episodes)
    stats = {
        "total_examples": len(episode_list),
        "kind_counts": dict(collections.Counter(episode.kind for episode in episode_list)),
        "language_counts": dict(collections.Counter(episode.language_family for episode in episode_list)),
        "bucket_counts": dict(collections.Counter(episode.length_bucket for episode in episode_list)),
        "source_counts": dict(collections.Counter(episode.source for episode in episode_list)),
        "skipped_counts": dict(skipped_counts),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(episodes: Iterable[Episode], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            row = {
                "content_split": episode.content,
                "metadata": asdict(episode) | {"content": None},
            }
            del row["metadata"]["content"]
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_from_config(config: dict) -> list[Episode]:
    output = config["output"]
    max_chars = int(output.get("max_chars_per_example", 120_000))
    max_file_chars = int(output.get("max_chars_per_file", 40_000))
    quality = config.get("quality", {})
    min_files = int(quality.get("min_files", 1))
    min_chars = int(quality.get("min_chars", 1))
    include_suffixes = set(config.get("include_suffixes", DEFAULT_SUFFIXES))
    episodes: list[Episode] = []
    skipped_counts: collections.Counter[str] = collections.Counter()

    for source in config.get("local_repo_sources", []):
        root = Path(source["root"]).expanduser()
        if not root.exists():
            continue
        kinds = set(source.get("kinds", ["repo_completion"]))
        limit = int(source.get("limit", 0))
        for repo in iter_repos(root):
            if "repo_completion" in kinds:
                episode = build_repo_episode(repo, include_suffixes, max_chars, max_file_chars)
                if episode is not None:
                    episodes.append(episode)
            if "test_conditioned_implementation" in kinds:
                episode = build_test_conditioned_episode(repo, include_suffixes, max_chars, max_file_chars)
                if episode is not None:
                    episodes.append(episode)
            if limit and len(episodes) >= limit:
                break

    for source in config.get("file_record_sources", []):
        if not source.get("enabled", True):
            continue
        records = iter_valid_records(records_from_source(source), include_suffixes, max_file_chars, skipped_counts)
        for repo, repo_records in group_records_by_repo(records).items():
            if len(repo_records) < min_files:
                skipped_counts["below_min_files"] += 1
                continue
            if sum(len(record.content) for record in repo_records) < min_chars:
                skipped_counts["below_min_chars"] += 1
                continue
            episodes.append(
                build_repo_episode_from_records(
                    repo,
                    repo_records,
                    kind=str(source.get("kind", "repo_completion")),
                    max_chars=max_chars,
                )
            )

    for source in config.get("issue_patch_sources", []):
        if not source.get("enabled", True):
            continue
        source_name = str(source.get("name", Path(source["path"]).stem))
        for row in issue_patch_rows_from_source(source):
            episode = build_issue_patch_episode(row, source_name, max_chars, max_file_chars)
            if episode is None:
                skipped_counts["invalid_issue_patch_row"] += 1
                continue
            episodes.append(episode)

    manifest_path = output.get("manifest_path")
    if manifest_path:
        write_manifest(episodes, skipped_counts, Path(manifest_path).expanduser())

    return episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build VeOmni plaintext JSONL episodes for 32k code continual training.")
    parser.add_argument("--config", type=Path, required=True, help="YAML data config.")
    parser.add_argument("--output", type=Path, default=None, help="Override output JSONL path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_path = args.output or Path(config["output"]["path"])
    episodes = build_from_config(config)
    count = write_jsonl(episodes, output_path)
    print(f"Wrote {count} examples to {output_path}")


if __name__ == "__main__":
    main()
