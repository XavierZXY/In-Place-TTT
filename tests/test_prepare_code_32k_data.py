import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_code_32k_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_code_32k_data", MODULE_PATH)
prepare_code_32k_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_code_32k_data)


def test_build_repo_episode_orders_context_before_tests(tmp_path):
    repo = tmp_path / "sample"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "README.md").write_text("# Sample\n", encoding="utf-8")
    (repo / "src" / "impl.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (repo / "tests" / "test_impl.py").write_text("from src.impl import add\nassert add(1, 2) == 3\n", encoding="utf-8")

    episode = prepare_code_32k_data.build_repo_episode(
        repo,
        include_suffixes={".md", ".py"},
        max_chars=10_000,
    )

    assert episode is not None
    assert episode.kind == "repo_completion"
    assert episode.content.index('<file path="README.md">') < episode.content.index('<file path="src/impl.py">')
    assert episode.content.index('<file path="src/impl.py">') < episode.content.index(
        '<file path="tests/test_impl.py">'
    )
    assert '<repo name="sample"' in episode.content


def test_build_test_conditioned_episode_places_tests_before_target(tmp_path):
    repo = tmp_path / "sample"
    (repo / "pkg").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "pkg" / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (repo / "tests" / "test_calc.py").write_text("from pkg.calc import add\nassert add(1, 2) == 3\n", encoding="utf-8")

    episode = prepare_code_32k_data.build_test_conditioned_episode(
        repo,
        include_suffixes={".py"},
        max_chars=10_000,
    )

    assert episode is not None
    assert episode.kind == "test_conditioned_implementation"
    assert episode.content.index("<tests>") < episode.content.index('<target_file path="pkg/calc.py">')
    assert "test_calc.py" in episode.content
    assert "calc.py" in episode.content


def test_write_jsonl_uses_content_split_field(tmp_path):
    output = tmp_path / "plaintext.jsonl"
    episodes = [
        prepare_code_32k_data.Episode(
            content='<repo name="sample"></repo>',
            repo="sample",
            kind="repo_completion",
            chars=27,
            files=0,
        )
    ]

    prepare_code_32k_data.write_jsonl(episodes, output)

    row = json.loads(output.read_text(encoding="utf-8").strip())
    assert row["content_split"] == '<repo name="sample"></repo>'
    assert row["metadata"]["repo"] == "sample"
    assert row["metadata"]["kind"] == "repo_completion"


def test_file_record_jsonl_sources_group_open_data_exports_into_repo_episodes(tmp_path):
    source = tmp_path / "stack_export.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "max_stars_repo_name": "org/project",
                        "path": "README.md",
                        "content": "# Project\n",
                    }
                ),
                json.dumps(
                    {
                        "repository": "org/project",
                        "file_path": "src/main.py",
                        "text": "def main():\n    return 1\n",
                    }
                ),
                json.dumps(
                    {
                        "repo_name": "org/project",
                        "path": "data/blob.bin",
                        "content": "abc\x00def",
                    }
                ),
                json.dumps(
                    {
                        "repo_name": "org/too-small",
                        "path": "src/only.py",
                        "content": "x = 1\n",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"

    episodes = prepare_code_32k_data.build_from_config(
        {
            "output": {
                "path": str(tmp_path / "plaintext.jsonl"),
                "manifest_path": str(manifest),
                "max_chars_per_example": 10_000,
                "max_chars_per_file": 1_000,
            },
            "quality": {
                "min_files": 2,
                "min_chars": 20,
            },
            "file_record_sources": [
                {
                    "name": "stack_v2_local_export",
                    "path": str(source),
                    "format": "jsonl",
                    "kind": "repo_completion",
                }
            ],
        }
    )

    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.repo == "org/project"
    assert episode.kind == "repo_completion"
    assert episode.source == "stack_v2_local_export"
    assert episode.language_family == "python"
    assert episode.length_bucket == "lt_8k_chars"
    assert '<file path="README.md">' in episode.content
    assert '<file path="src/main.py">' in episode.content
    assert "blob.bin" not in episode.content

    stats = json.loads(manifest.read_text(encoding="utf-8"))
    assert stats["total_examples"] == 1
    assert stats["kind_counts"] == {"repo_completion": 1}
    assert stats["language_counts"] == {"python": 1}
    assert stats["bucket_counts"] == {"lt_8k_chars": 1}
    assert stats["source_counts"] == {"stack_v2_local_export": 1}
    assert stats["skipped_counts"]["binary_or_empty"] == 1
    assert stats["skipped_counts"]["below_min_files"] == 1


def test_issue_patch_jsonl_sources_build_patch_style_episodes(tmp_path):
    source = tmp_path / "issues.jsonl"
    source.write_text(
        json.dumps(
            {
                "repo_name": "org/project",
                "issue": "Implement add().",
                "base_files": {
                    "pkg/calc.py": "def add(a, b):\n    pass\n",
                    "tests/test_calc.py": "from pkg.calc import add\nassert add(1, 2) == 3\n",
                },
                "patch": "diff --git a/pkg/calc.py b/pkg/calc.py\n+    return a + b\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    episodes = prepare_code_32k_data.build_from_config(
        {
            "output": {
                "path": str(tmp_path / "plaintext.jsonl"),
                "max_chars_per_example": 10_000,
                "max_chars_per_file": 1_000,
            },
            "issue_patch_sources": [
                {
                    "name": "starcoderdata_patch_export",
                    "path": str(source),
                    "format": "jsonl",
                    "kind": "issue_patch",
                }
            ],
        }
    )

    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.repo == "org/project"
    assert episode.kind == "issue_patch"
    assert episode.source == "starcoderdata_patch_export"
    assert episode.language_family == "python"
    assert "<issue>\nImplement add().\n</issue>" in episode.content
    assert '<file path="tests/test_calc.py">' in episode.content
    assert "<patch>\ndiff --git" in episode.content


def test_file_record_sources_skip_vendor_directories(tmp_path):
    source = tmp_path / "github_code.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "repo_name": "org/web",
                        "path": "package.json",
                        "content": '{"scripts": {"test": "vitest"}}',
                    }
                ),
                json.dumps(
                    {
                        "repo_name": "org/web",
                        "path": "src/app.ts",
                        "content": "export const app = 1;\n",
                    }
                ),
                json.dumps(
                    {
                        "repo_name": "org/web",
                        "path": "node_modules/pkg/index.js",
                        "content": "module.exports = 1;\n",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    episodes = prepare_code_32k_data.build_from_config(
        {
            "output": {
                "path": str(tmp_path / "plaintext.jsonl"),
                "max_chars_per_example": 10_000,
                "max_chars_per_file": 1_000,
            },
            "quality": {
                "min_files": 2,
                "min_chars": 1,
            },
            "include_suffixes": [".json", ".ts", ".js"],
            "file_record_sources": [
                {
                    "name": "github_code_local_export",
                    "path": str(source),
                    "format": "jsonl",
                    "kind": "repo_completion",
                }
            ],
        }
    )

    assert len(episodes) == 1
    assert '<file path="src/app.ts">' in episodes[0].content
    assert "node_modules" not in episodes[0].content
