def should_run_eval(eval_path: str | None, eval_steps: int, eval_batches: int, global_step: int) -> bool:
    return bool(eval_path) and eval_steps > 0 and eval_batches > 0 and global_step > 0 and global_step % eval_steps == 0
