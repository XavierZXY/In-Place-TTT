# Code 32k Data Selection

This project should use a 32k coding-focused continual pretraining set before attempting a 128k stage.

The recommended first dataset is `code_32k`, a VeOmni plaintext JSONL file with one field:

```json
{"content_split": "..."}
```

## Target Mix

- Python repo-level completion: 35%
- TypeScript/JavaScript repo-level completion: 20%
- Go/Rust/Java repo-level completion: 15%
- Test-conditioned implementation: 20%
- Issue or commit patch style: 10%

## Episode Boundaries

Each example should represent one coherent code episode. Good boundaries are:

- one repository snapshot
- one test-conditioned implementation episode
- one issue/commit patch episode

Do not concatenate unrelated repositories into one `content_split`. In-Place TTT resets fast weights at document boundaries, so unrelated content in the same example teaches the wrong memory behavior.

## Length Buckets

Use a single `max_seq_len=32768` training configuration. Internally sample by approximate token length:

- 4k-8k: 10%
- 8k-16k: 30%
- 16k-32k: 60%

The 4k/8k/16k/32k/64k/128k split is primarily an evaluation sweep, not a reason to create separate training datasets for every length.

## Open Dataset Inputs

The preparation script is an adapter for existing open datasets or local mirrors. It does not synthesize training data by itself.

For The Stack v2 / StarCoderData style file exports, write local JSONL or Parquet rows with these canonical fields:

```json
{"repo_name": "org/repo", "path": "src/main.py", "content": "..."}
```

The adapter also accepts common aliases:

- repository name: `repo_name`, `repository`, `max_stars_repo_name`, `repo`
- file path: `path`, `file_path`, `filepath`, `relative_path`
- file text: `content`, `text`, `blob`, `code`

For issue or patch-style data, write local JSONL rows like:

```json
{"repo_name": "org/repo", "issue": "...", "base_files": {"pkg/a.py": "..."}, "patch": "diff --git ..."}
```

Use `configs/data/code_32k_mix.yaml` to point at these local files through `file_record_sources` and `issue_patch_sources`. The checked-in examples are disabled until the paths are replaced with real local exports.

## Build Command

Mirror or filter source repositories under `data/raw/repos`, then run:

```bash
uvx uv==0.9.8 run python scripts/prepare_code_32k_data.py \
  --config configs/data/code_32k_mix.yaml
```

Use the output path as:

```bash
--data.train_path data/code_32k/plaintext.jsonl
```

The script also writes `data/code_32k/manifest.json`, which records example counts by task kind, language family, length bucket, source, and skipped records.
