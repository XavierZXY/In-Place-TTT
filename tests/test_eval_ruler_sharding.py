import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).resolve().parents[1] / "eval" / "eval_scripts" / "eval_ruler.py"
SPEC = importlib.util.spec_from_file_location("eval_ruler", MODULE_PATH)
eval_ruler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(eval_ruler)


def test_select_shard_uses_stable_global_sample_indices_without_mutating_input():
    rows = [{"task": f"task-{index}"} for index in range(10)]

    shard = eval_ruler.select_shard(rows, num_shards=3, shard_id=1)

    assert [row["sample_index"] for row in shard] == [1, 4, 7]
    assert [row["task"] for row in shard] == ["task-1", "task-4", "task-7"]
    assert all("sample_index" not in row for row in rows)


def test_apply_ttt_runtime_overrides_can_disable_ttt_mode():
    config = SimpleNamespace(ttt_mode=True, ttt_layers=[4, 10], ttt_lr=3)

    eval_ruler.apply_ttt_runtime_overrides(
        config,
        disable_ttt=True,
        disable_ttt_fast_weights=False,
        ttt_prefill_update_partial=False,
        ttt_prefill_partial_min_tokens=None,
    )

    assert config.ttt_mode is False
    assert config.ttt_layers == []
    assert config.ttt_lr == 0.0
    assert config.disable_ttt is True
    assert config.disable_ttt_fast_weights is True


def test_merge_shard_outputs_writes_ordered_samples_and_combined_summary(tmp_path):
    rows_by_shard = {
        0: [
            {"sample_index": 0, "task": "niah_single_1", "score": 1.0, "gen": "a", "answer": ["a"]},
            {"sample_index": 2, "task": "qa_1", "score": 0.0, "gen": "x", "answer": ["b"]},
        ],
        1: [
            {"sample_index": 1, "task": "niah_single_1", "score": 0.0, "gen": "x", "answer": ["a"]},
        ],
    }
    for shard_id, rows in rows_by_shard.items():
        with (tmp_path / f"len_16384_shard_{shard_id:02d}_samples.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    merged = eval_ruler.merge_shard_outputs(tmp_path, lengths=[16384], num_shards=2)

    sample_path = tmp_path / "len_16384_samples.jsonl"
    summary_path = tmp_path / "len_16384_summary.json"
    all_summary_path = tmp_path / "summary_all_lengths.json"

    merged_rows = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines()]
    assert [row["sample_index"] for row in merged_rows] == [0, 1, 2]

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["niah_single_1"] == {"n": 2, "score": 0.5}
    assert summary["qa_1"] == {"n": 1, "score": 0.0}
    assert summary["__overall__"] == {"n": 3, "score": 1.0 / 3.0}
    assert summary["__merged_from_shards__"] == {"num_shards": 2}

    assert json.loads(all_summary_path.read_text(encoding="utf-8")) == {"16384": summary}
    assert merged == {16384: summary}
