import importlib.util
import json
from argparse import Namespace
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_github_code_20b_data.py"
SPEC = importlib.util.spec_from_file_location("build_github_code_20b_data", MODULE_PATH)
build_github_code_20b_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_github_code_20b_data)


def test_row_filter_rejects_vendor_and_unsupported_records():
    allowed_languages = {"Python", "JavaScript"}
    allowed_licenses = {"mit", "apache-2.0"}

    accepted = build_github_code_20b_data.accept_row(
        {
            "repo_name": "org/repo",
            "path": "src/app.py",
            "language": "Python",
            "license": "MIT",
            "code": "print('ok')\n",
        },
        allowed_languages=allowed_languages,
        allowed_licenses=allowed_licenses,
        min_chars=1,
        max_chars=1_000,
    )
    assert accepted == "accepted"

    assert (
        build_github_code_20b_data.accept_row(
            {
                "repo_name": "org/repo",
                "path": "node_modules/pkg/index.js",
                "language": "JavaScript",
                "license": "mit",
                "code": "module.exports = 1;\n",
            },
            allowed_languages=allowed_languages,
            allowed_licenses=allowed_licenses,
            min_chars=1,
            max_chars=1_000,
        )
        == "skipped_dir"
    )
    assert (
        build_github_code_20b_data.accept_row(
            {
                "repo_name": "org/repo",
                "path": "src/app.py",
                "language": "Python",
                "license": "gpl-2.0",
                "code": "print('ok')\n",
            },
            allowed_languages=allowed_languages,
            allowed_licenses=allowed_licenses,
            min_chars=1,
            max_chars=1_000,
        )
        == "license"
    )


def test_build_episode_uses_content_split_and_metadata():
    row = {
        "repo_name": "org/repo",
        "path": "src/app.py",
        "language": "Python",
        "license": "mit",
        "code": "def main():\n    return 1\n",
    }

    episode = build_github_code_20b_data.build_episode(row, token_count=12)
    parsed = json.loads(episode)

    assert parsed["content_split"].startswith('<repo name="org/repo" task="file_completion">')
    assert '<file path="src/app.py">' in parsed["content_split"]
    assert parsed["metadata"]["repo"] == "org/repo"
    assert parsed["metadata"]["kind"] == "file_completion"
    assert parsed["metadata"]["tokens"] == 12
    assert parsed["metadata"]["language_family"] == "python"


def test_dataset_config_names_use_github_code_builder_names():
    configs = build_github_code_20b_data.dataset_config_names(
        ["Python", "Go"],
        ["mit", "apache-2.0"],
    )

    assert configs == [
        "Python-mit",
        "Python-apache-2.0",
        "GO-mit",
        "GO-apache-2.0",
    ]


def test_main_flushes_final_manifest_before_output_handle_closes(tmp_path, monkeypatch):
    output = tmp_path / "plaintext.jsonl"
    manifest = tmp_path / "manifest.json"

    monkeypatch.setattr(
        build_github_code_20b_data,
        "parse_args",
        lambda: Namespace(
            dataset="local",
            tokenizer="local",
            output=output,
            manifest=manifest,
            target_tokens=2,
            languages="Python",
            licenses="mit",
            min_chars=1,
            max_chars=1_000,
            report_every=100,
            max_rows=0,
            resume=False,
            use_dataset_configs=False,
        ),
    )
    monkeypatch.setattr(
        build_github_code_20b_data.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: type("Tokenizer", (), {"encode": lambda self, text, add_special_tokens=False: [1, 2]})(),
    )
    monkeypatch.setattr(
        build_github_code_20b_data,
        "stream_dataset_config",
        lambda *args, **kwargs: iter(
            [
                {
                    "repo_name": "org/repo",
                    "path": "src/app.py",
                    "language": "Python",
                    "license": "mit",
                    "code": "print('ok')\n",
                }
            ]
        ),
    )

    build_github_code_20b_data.main()

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["total_tokens"] == 2
    assert output.read_text(encoding="utf-8").count("\n") == 1
