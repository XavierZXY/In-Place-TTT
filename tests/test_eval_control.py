import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tasks" / "eval_control.py"
SPEC = importlib.util.spec_from_file_location("eval_control", MODULE_PATH)
eval_control = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(eval_control)


def test_should_run_eval_requires_path_positive_interval_and_positive_batches():
    assert not eval_control.should_run_eval(eval_path=None, eval_steps=500, eval_batches=8, global_step=500)
    assert not eval_control.should_run_eval(eval_path="data/eval.jsonl", eval_steps=0, eval_batches=8, global_step=500)
    assert not eval_control.should_run_eval(eval_path="data/eval.jsonl", eval_steps=500, eval_batches=0, global_step=500)
    assert not eval_control.should_run_eval(eval_path="data/eval.jsonl", eval_steps=500, eval_batches=8, global_step=499)

    assert eval_control.should_run_eval(eval_path="data/eval.jsonl", eval_steps=500, eval_batches=8, global_step=500)
