import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "split_code_32k_eval.py"
SPEC = importlib.util.spec_from_file_location("split_code_32k_eval", MODULE_PATH)
split_code_32k_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(split_code_32k_eval)


def test_split_jsonl_writes_eval_rows_and_excludes_them_from_train(tmp_path):
    source = tmp_path / "plaintext.jsonl"
    train_output = tmp_path / "train_plaintext.jsonl"
    eval_output = tmp_path / "eval_plaintext.jsonl"
    manifest = tmp_path / "split_manifest.json"
    rows = [{"content_split": f"sample-{idx}", "metadata": {"idx": idx}} for idx in range(1, 11)]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    stats = split_code_32k_eval.split_jsonl(
        input_path=source,
        train_output=train_output,
        eval_output=eval_output,
        manifest_path=manifest,
        eval_examples=3,
        eval_stride=2,
        overwrite=False,
    )

    eval_rows = [json.loads(line) for line in eval_output.read_text(encoding="utf-8").splitlines()]
    train_rows = [json.loads(line) for line in train_output.read_text(encoding="utf-8").splitlines()]

    assert [row["metadata"]["idx"] for row in eval_rows] == [2, 4, 6]
    assert [row["metadata"]["idx"] for row in train_rows] == [1, 3, 5, 7, 8, 9, 10]
    assert stats.rows_read == 10
    assert stats.eval_rows == 3
    assert stats.train_rows == 7

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["input_path"] == str(source)
    assert payload["eval_output"] == str(eval_output)
    assert payload["train_output"] == str(train_output)
    assert payload["eval_examples"] == 3
    assert payload["eval_stride"] == 2
