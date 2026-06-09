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

"""Self-contained RULER evaluator for local In-Place-TTT HF checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList, StopStringCriteria


EVAL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EVAL_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

import inference_model  # noqa: E402,F401


TASK_NAMES = [
    "cwe",
    "fwe",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multiquery",
    "niah_multivalue",
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "qa_1",
    "qa_2",
    "vt",
]

QWEN3_EOS_TOKEN_IDS = [151643, 151645]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _word_in(needle: str, haystack: str) -> bool:
    if not needle:
        return False
    if not needle[0].isalnum() or not needle[-1].isalnum():
        return needle in haystack
    return bool(re.search(r"\b" + re.escape(needle) + r"\b", haystack))


def score_contains(generation: str, answers: list[str]) -> float:
    normalized_generation = _norm(generation)
    return float(any(_norm(answer) in normalized_generation for answer in answers))


def score_list_overlap(generation: str, answers: list[str]) -> float:
    if not answers:
        return 0.0
    normalized_generation = _norm(generation)
    return sum(_word_in(_norm(answer), normalized_generation) for answer in answers) / len(answers)


SCORERS = {
    "niah_single_1": score_contains,
    "niah_single_2": score_contains,
    "niah_single_3": score_contains,
    "niah_multikey_1": score_contains,
    "niah_multikey_2": score_contains,
    "niah_multikey_3": score_contains,
    "niah_multiquery": score_list_overlap,
    "niah_multivalue": score_list_overlap,
    "qa_1": score_contains,
    "qa_2": score_contains,
    "vt": score_list_overlap,
    "cwe": score_list_overlap,
    "fwe": score_list_overlap,
}

FULL_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think_block(generation: str) -> str:
    cleaned = FULL_THINK_RE.sub("", generation).strip()
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[1].strip()
    return cleaned


def _find_repeated_suffix(
    token_ids: list[int],
    min_ngram: int = 8,
    max_ngram: int = 128,
    min_repeats: int = 3,
) -> dict[str, int | str] | None:
    token_count = len(token_ids)
    max_ngram = min(max_ngram, token_count // max(min_repeats, 1))
    for ngram_size in range(max_ngram, min_ngram - 1, -1):
        suffix = token_ids[token_count - ngram_size : token_count]
        repeats = 1
        pos = token_count - ngram_size
        while pos - ngram_size >= 0 and token_ids[pos - ngram_size : pos] == suffix:
            repeats += 1
            pos -= ngram_size
        if repeats >= min_repeats:
            return {
                "reason": "exact_ngram",
                "ngram_tokens": ngram_size,
                "repeats": repeats,
                "cut_idx": pos + ngram_size,
            }
    return None


class RepetitionStoppingCriteria(StoppingCriteria):
    """Stop obvious greedy-decoding repetition loops."""

    def __init__(
        self,
        prompt_len: int,
        min_new_tokens: int = 96,
        min_ngram: int = 8,
        max_ngram: int = 128,
        repeats: int = 3,
        diversity_window: int = 160,
        diversity_unique_ratio: float = 0.22,
        diversity_top_ratio: float = 0.18,
    ) -> None:
        self.prompt_len = int(prompt_len)
        self.min_new_tokens = int(min_new_tokens)
        self.min_ngram = int(min_ngram)
        self.max_ngram = int(max_ngram)
        self.repeats = int(repeats)
        self.diversity_window = int(diversity_window)
        self.diversity_unique_ratio = float(diversity_unique_ratio)
        self.diversity_top_ratio = float(diversity_top_ratio)
        self.triggered = False
        self.reason: str | None = None
        self.cut_new_tokens: int | None = None
        self.details: dict[str, Any] = {}
        self._triggered: list[bool] = []
        self._reasons: list[str | None] = []
        self._cut_new_tokens: list[int | None] = []
        self._details: list[dict[str, Any]] = []

    def _ensure_batch_size(self, batch_size: int) -> None:
        missing = batch_size - len(self._triggered)
        if missing <= 0:
            return
        self._triggered.extend([False] * missing)
        self._reasons.extend([None] * missing)
        self._cut_new_tokens.extend([None] * missing)
        self._details.extend({} for _ in range(missing))

    def _mark_triggered(self, index: int, reason: str, cut_new_tokens: int, details: dict[str, Any]) -> None:
        self._triggered[index] = True
        self._reasons[index] = reason
        self._cut_new_tokens[index] = cut_new_tokens
        self._details[index] = details
        self.triggered = True
        self.reason = reason
        self.cut_new_tokens = cut_new_tokens
        self.details = details

    def meta_for(self, index: int, generated_token_count: int) -> dict[str, Any] | None:
        if index >= len(self._triggered) or not self._triggered[index]:
            return None
        cut_idx = self._cut_new_tokens[index]
        if cut_idx is None:
            return None
        cut_idx = max(0, min(int(cut_idx), generated_token_count))
        return {
            "reason": self._reasons[index],
            "generated_tokens_before_trim": generated_token_count,
            "generated_tokens_after_trim": cut_idx,
            **self._details[index],
        }

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> torch.BoolTensor:
        batch_size = input_ids.shape[0]
        self._ensure_batch_size(batch_size)
        is_done = torch.zeros(batch_size, device=input_ids.device, dtype=torch.bool)

        for index in range(batch_size):
            if self._triggered[index]:
                is_done[index] = True
                continue

            generated_ids = input_ids[index, self.prompt_len :].tolist()
            generated_len = len(generated_ids)
            if generated_len < self.min_new_tokens:
                continue

            match = _find_repeated_suffix(
                generated_ids,
                min_ngram=self.min_ngram,
                max_ngram=self.max_ngram,
                min_repeats=self.repeats,
            )
            if match is not None:
                self._mark_triggered(
                    index=index,
                    reason=str(match["reason"]),
                    cut_new_tokens=int(match["cut_idx"]),
                    details={
                        "ngram_tokens": int(match["ngram_tokens"]),
                        "repeats": int(match["repeats"]),
                    },
                )
                is_done[index] = True
                continue

            if self.diversity_window > 0 and generated_len >= max(self.min_new_tokens, self.diversity_window):
                window = generated_ids[-self.diversity_window :]
                counts = Counter(window)
                unique_ratio = len(counts) / self.diversity_window
                top_ratio = counts.most_common(1)[0][1] / self.diversity_window
                if unique_ratio <= self.diversity_unique_ratio and top_ratio >= self.diversity_top_ratio:
                    self._mark_triggered(
                        index=index,
                        reason="low_diversity",
                        cut_new_tokens=generated_len - self.diversity_window,
                        details={
                            "window": self.diversity_window,
                            "unique_ratio": unique_ratio,
                            "top_ratio": top_ratio,
                        },
                    )
                    is_done[index] = True

        return is_done


def _trim_repetition_tail(
    new_tokens: torch.LongTensor,
    repetition_criteria: RepetitionStoppingCriteria | None,
    min_ngram: int,
    max_ngram: int,
    batch_index: int = 0,
) -> tuple[torch.LongTensor, dict[str, Any] | None]:
    token_ids = new_tokens.tolist()

    if repetition_criteria is not None:
        meta = repetition_criteria.meta_for(batch_index, len(token_ids))
        if meta is not None:
            cut_idx = int(meta["generated_tokens_after_trim"])
            if cut_idx < len(token_ids):
                return new_tokens[:cut_idx], meta

    match = _find_repeated_suffix(token_ids, min_ngram=min_ngram, max_ngram=max_ngram, min_repeats=2)
    if match is not None and int(match["cut_idx"]) < len(token_ids):
        cut_idx = int(match["cut_idx"])
        return new_tokens[:cut_idx], {
            "reason": "post_exact_ngram",
            "generated_tokens_before_trim": len(token_ids),
            "generated_tokens_after_trim": cut_idx,
            "ngram_tokens": int(match["ngram_tokens"]),
            "repeats": int(match["repeats"]),
        }

    return new_tokens, None


def _torch_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    dtype = getattr(torch, name, None)
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {name}")
    return dtype


def _model_class(config: Any) -> type[torch.nn.Module]:
    if config.model_type == "qwen3":
        from inference_model.hf_qwen3 import Qwen3ForCausalLM

        return Qwen3ForCausalLM
    if config.model_type == "llama":
        from inference_model.hf_llama3 import LlamaForCausalLM

        return LlamaForCausalLM
    raise ValueError(f"Unsupported model_type={config.model_type!r}; expected qwen3 or llama")


def _resolve_eos_token_ids(config: Any, tokenizer: Any, explicit_ids: list[int]) -> list[int]:
    if explicit_ids:
        return explicit_ids

    eos_ids = set()
    config_eos = getattr(config, "eos_token_id", None)
    if isinstance(config_eos, int):
        eos_ids.add(config_eos)
    elif isinstance(config_eos, list):
        eos_ids.update(int(item) for item in config_eos)

    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))

    if getattr(config, "model_type", None) == "qwen3":
        eos_ids.update(QWEN3_EOS_TOKEN_IDS)

    return sorted(eos_ids)


def load_model(
    model_path: str,
    dtype_name: str,
    attn_implementation: str,
    device: str,
    disable_ttt_fast_weights: bool,
    ttt_prefill_update_partial: bool,
    ttt_prefill_partial_min_tokens: int | None,
) -> tuple[torch.nn.Module, Any, Any]:
    print(f"[load] model_path={model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if disable_ttt_fast_weights:
        original_ttt_lr = getattr(config, "ttt_lr", None)
        config.ttt_lr = 0.0
        config.disable_ttt_fast_weights = True
        print(f"[load] disable_ttt_fast_weights=True original_ttt_lr={original_ttt_lr} effective_ttt_lr=0.0")
    if ttt_prefill_update_partial:
        config.ttt_prefill_update_partial = True
        if ttt_prefill_partial_min_tokens is not None:
            if ttt_prefill_partial_min_tokens < 1:
                raise ValueError(
                    "ttt_prefill_partial_min_tokens must be >= 1, "
                    f"got {ttt_prefill_partial_min_tokens}"
                )
            config.ttt_prefill_partial_min_tokens = int(ttt_prefill_partial_min_tokens)
        print(
            "[load] ttt_prefill_update_partial=True "
            f"min_tokens={getattr(config, 'ttt_prefill_partial_min_tokens', 1)}"
        )
    model_cls = _model_class(config)

    model = model_cls.from_pretrained(
        model_path,
        config=config,
        torch_dtype=_torch_dtype(dtype_name),
        attn_implementation=attn_implementation,
    ).eval()
    if device:
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(
        "[load] done "
        f"model_type={config.model_type} "
        f"ttt_mode={getattr(config, 'ttt_mode', None)} "
        f"ttt_layers={getattr(config, 'ttt_layers', None)} "
        f"device={device or 'from_pretrained'} "
        f"dtype={dtype_name}"
    )
    return model, tokenizer, config


def maybe_wrap_chat(tokenizer: Any, prompt: str, chat_template: str, assistant_prefill: str) -> str:
    if not chat_template:
        return prompt + assistant_prefill
    if not tokenizer.chat_template:
        raise RuntimeError("chat_template is enabled, but the tokenizer has no chat template")
    wrapped = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return wrapped + assistant_prefill


def _strip_trailing_pad_tokens(token_ids: torch.LongTensor, pad_token_id: int | None) -> torch.LongTensor:
    if pad_token_id is None:
        return token_ids
    end = len(token_ids)
    while end > 0 and int(token_ids[end - 1]) == int(pad_token_id):
        end -= 1
    return token_ids[:end]


def generate_one(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    max_seq: int,
    eos_token_ids: list[int],
    chat_template: str,
    stop_string_criteria: StopStringCriteria | None,
    assistant_prefill: str,
    repetition_stop: bool,
    repeat_stop_min_new_tokens: int,
    repeat_stop_min_ngram: int,
    repeat_stop_max_ngram: int,
    repeat_stop_repeats: int,
    repeat_stop_diversity_window: int,
    repeat_stop_diversity_unique_ratio: float,
    repeat_stop_diversity_top_ratio: float,
    device: str,
) -> tuple[str, dict[str, Any] | None]:
    prompt = maybe_wrap_chat(tokenizer, prompt, chat_template, assistant_prefill)
    max_input_len = max_seq - max_new_tokens
    if max_input_len <= 0:
        raise ValueError(f"max_seq ({max_seq}) must be larger than max_new_tokens ({max_new_tokens})")

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_len,
        add_special_tokens=False,
    )
    if device:
        inputs = inputs.to(device)

    criteria = []
    if stop_string_criteria is not None:
        criteria.append(stop_string_criteria)

    repetition_criteria = None
    if repetition_stop:
        repetition_criteria = RepetitionStoppingCriteria(
            prompt_len=inputs.input_ids.shape[1],
            min_new_tokens=repeat_stop_min_new_tokens,
            min_ngram=repeat_stop_min_ngram,
            max_ngram=repeat_stop_max_ngram,
            repeats=repeat_stop_repeats,
            diversity_window=repeat_stop_diversity_window,
            diversity_unique_ratio=repeat_stop_diversity_unique_ratio,
            diversity_top_ratio=repeat_stop_diversity_top_ratio,
        )
        criteria.append(repetition_criteria)

    stopping_criteria = StoppingCriteriaList(criteria) if criteria else None

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_ids or tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )

    new_tokens = output_ids[0, inputs.input_ids.shape[1] :]
    new_tokens, repetition_meta = _trim_repetition_tail(
        new_tokens,
        repetition_criteria=repetition_criteria,
        min_ngram=repeat_stop_min_ngram,
        max_ngram=repeat_stop_max_ngram,
    )
    return tokenizer.decode(new_tokens, skip_special_tokens=True), repetition_meta


def generate_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
    max_seq: int,
    eos_token_ids: list[int],
    chat_template: str,
    stop_string_criteria: StopStringCriteria | None,
    assistant_prefill: str,
    repetition_stop: bool,
    repeat_stop_min_new_tokens: int,
    repeat_stop_min_ngram: int,
    repeat_stop_max_ngram: int,
    repeat_stop_repeats: int,
    repeat_stop_diversity_window: int,
    repeat_stop_diversity_unique_ratio: float,
    repeat_stop_diversity_top_ratio: float,
    device: str,
) -> list[tuple[str, dict[str, Any] | None]]:
    if not prompts:
        return []

    wrapped_prompts = [maybe_wrap_chat(tokenizer, prompt, chat_template, assistant_prefill) for prompt in prompts]
    max_input_len = max_seq - max_new_tokens
    if max_input_len <= 0:
        raise ValueError(f"max_seq ({max_seq}) must be larger than max_new_tokens ({max_new_tokens})")

    inputs = tokenizer(
        wrapped_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_len,
        add_special_tokens=False,
    )
    if device:
        inputs = inputs.to(device)

    criteria = []
    if stop_string_criteria is not None:
        criteria.append(stop_string_criteria)

    repetition_criteria = None
    if repetition_stop:
        repetition_criteria = RepetitionStoppingCriteria(
            prompt_len=inputs.input_ids.shape[1],
            min_new_tokens=repeat_stop_min_new_tokens,
            min_ngram=repeat_stop_min_ngram,
            max_ngram=repeat_stop_max_ngram,
            repeats=repeat_stop_repeats,
            diversity_window=repeat_stop_diversity_window,
            diversity_unique_ratio=repeat_stop_diversity_unique_ratio,
            diversity_top_ratio=repeat_stop_diversity_top_ratio,
        )
        criteria.append(repetition_criteria)

    stopping_criteria = StoppingCriteriaList(criteria) if criteria else None

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_ids or tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )

    prompt_width = inputs.input_ids.shape[1]
    generations = []
    for batch_index in range(len(prompts)):
        new_tokens = output_ids[batch_index, prompt_width:]
        if len(prompts) > 1:
            new_tokens = _strip_trailing_pad_tokens(new_tokens, tokenizer.pad_token_id)
        new_tokens, repetition_meta = _trim_repetition_tail(
            new_tokens,
            repetition_criteria=repetition_criteria,
            min_ngram=repeat_stop_min_ngram,
            max_ngram=repeat_stop_max_ngram,
            batch_index=batch_index,
        )
        generations.append((tokenizer.decode(new_tokens, skip_special_tokens=True), repetition_meta))
    return generations


def _load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Reading parquet RULER data requires pyarrow. Install pyarrow or use JSONL data.") from exc
    return pq.read_table(path).to_pylist()


def _ordered_tasks(tasks: list[str]) -> list[str]:
    known = [task for task in TASK_NAMES if task in tasks]
    extra = sorted(task for task in tasks if task not in TASK_NAMES)
    return known + extra


def _load_samples(ruler_root: Path, length: int, n_per_task: int, default_max_new_tokens: int) -> list[dict[str, Any]]:
    parquet_path = ruler_root / str(length) / "test-00000-of-00001.parquet"
    jsonl_dir = ruler_root / str(length) / "data"
    samples: list[dict[str, Any]] = []

    if parquet_path.exists():
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in _load_parquet_rows(parquet_path):
            by_task[row["task"]].append(row)

        for task in _ordered_tasks(list(by_task.keys())):
            for row in by_task[task][:n_per_task]:
                samples.append(
                    {
                        "task": task,
                        "prompt": row["context"] + "\n" + row["question"] + "\n" + row["answer_prefix"],
                        "answer": list(row["answer"]),
                        "max_new_tokens": int(row.get("max_new_tokens", default_max_new_tokens)),
                    }
                )
        return samples

    if jsonl_dir.exists():
        for task in TASK_NAMES:
            path = jsonl_dir / task / "validation.jsonl"
            if not path.exists():
                continue
            taken = 0
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if taken >= n_per_task:
                        break
                    row = json.loads(line)
                    samples.append(
                        {
                            "task": task,
                            "prompt": row["input"],
                            "answer": list(row["outputs"]),
                            "max_new_tokens": int(row.get("max_new_tokens", default_max_new_tokens)),
                        }
                    )
                    taken += 1
        if samples:
            return samples

    raise FileNotFoundError(f"No RULER data found at {parquet_path} or {jsonl_dir}")


def _load_existing_rows(samples_path: Path, expected_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not samples_path.exists():
        return []

    rows = []
    found_corrupt_tail = False
    with samples_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                found_corrupt_tail = True
                break

    if found_corrupt_tail or len(rows) > len(expected_rows):
        rows = rows[: len(expected_rows)]
        with samples_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    for index, row in enumerate(rows):
        if row.get("task") != expected_rows[index].get("task"):
            samples_path.unlink()
            return []

    return rows


def eval_length(
    model: torch.nn.Module,
    tokenizer: Any,
    length: int,
    n_per_task: int,
    max_seq: int,
    out_dir: Path,
    ruler_root: Path,
    eos_token_ids: list[int],
    chat_template: str,
    max_new_tokens: int,
    strip_think: bool,
    stop_strings: list[str],
    assistant_prefill: str,
    repetition_stop: bool,
    repeat_stop_min_new_tokens: int,
    repeat_stop_min_ngram: int,
    repeat_stop_max_ngram: int,
    repeat_stop_repeats: int,
    repeat_stop_diversity_window: int,
    repeat_stop_diversity_unique_ratio: float,
    repeat_stop_diversity_top_ratio: float,
    device: str,
    batch_size: int,
    run_meta: dict[str, Any],
) -> dict[str, Any]:
    sample_rows = _load_samples(ruler_root, length, n_per_task, max_new_tokens)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        by_task[row["task"]].append(row)

    stop_string_criteria = StopStringCriteria(tokenizer=tokenizer, stop_strings=stop_strings) if stop_strings else None

    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / f"len_{length}_samples.jsonl"
    summary_path = out_dir / f"len_{length}_summary.json"

    existing_rows = _load_existing_rows(samples_path, sample_rows)
    done_count = len(existing_rows)
    if done_count:
        print(f"[eval-{length}] resume {done_count}/{len(sample_rows)} samples")
    else:
        print(
            f"[eval-{length}] samples={len(sample_rows)} n_per_task={n_per_task} "
            f"chat_template={chat_template or 'none'} max_new_tokens={max_new_tokens} "
            f"strip_think={strip_think} stop_strings={stop_strings or 'none'} "
            f"repetition_stop={repetition_stop} batch_size={batch_size}"
        )

    per_task_scores: dict[str, list[float]] = defaultdict(list)
    repetition_stop_count = 0
    for row in existing_rows:
        per_task_scores[row["task"]].append(float(row["score"]))
        if row.get("repetition_stop"):
            repetition_stop_count += 1

    start_time = time.time()
    if done_count < len(sample_rows):
        with samples_path.open("a", encoding="utf-8") as f:
            progress = tqdm(desc=f"len={length}", initial=done_count, total=len(sample_rows))
            row_index = done_count
            while row_index < len(sample_rows):
                first_row = sample_rows[row_index]
                current_max_new_tokens = int(first_row["max_new_tokens"])
                current_batch = [first_row]
                row_index += 1
                while (
                    batch_size > 1
                    and row_index < len(sample_rows)
                    and len(current_batch) < batch_size
                    and int(sample_rows[row_index]["max_new_tokens"]) == current_max_new_tokens
                ):
                    current_batch.append(sample_rows[row_index])
                    row_index += 1

                if len(current_batch) == 1:
                    generations = [
                        generate_one(
                            model=model,
                            tokenizer=tokenizer,
                            prompt=current_batch[0]["prompt"],
                            max_new_tokens=current_max_new_tokens,
                            max_seq=max_seq,
                            eos_token_ids=eos_token_ids,
                            chat_template=chat_template,
                            stop_string_criteria=stop_string_criteria,
                            assistant_prefill=assistant_prefill,
                            repetition_stop=repetition_stop,
                            repeat_stop_min_new_tokens=repeat_stop_min_new_tokens,
                            repeat_stop_min_ngram=repeat_stop_min_ngram,
                            repeat_stop_max_ngram=repeat_stop_max_ngram,
                            repeat_stop_repeats=repeat_stop_repeats,
                            repeat_stop_diversity_window=repeat_stop_diversity_window,
                            repeat_stop_diversity_unique_ratio=repeat_stop_diversity_unique_ratio,
                            repeat_stop_diversity_top_ratio=repeat_stop_diversity_top_ratio,
                            device=device,
                        )
                    ]
                else:
                    generations = generate_batch(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=[row["prompt"] for row in current_batch],
                        max_new_tokens=current_max_new_tokens,
                        max_seq=max_seq,
                        eos_token_ids=eos_token_ids,
                        chat_template=chat_template,
                        stop_string_criteria=stop_string_criteria,
                        assistant_prefill=assistant_prefill,
                        repetition_stop=repetition_stop,
                        repeat_stop_min_new_tokens=repeat_stop_min_new_tokens,
                        repeat_stop_min_ngram=repeat_stop_min_ngram,
                        repeat_stop_max_ngram=repeat_stop_max_ngram,
                        repeat_stop_repeats=repeat_stop_repeats,
                        repeat_stop_diversity_window=repeat_stop_diversity_window,
                        repeat_stop_diversity_unique_ratio=repeat_stop_diversity_unique_ratio,
                        repeat_stop_diversity_top_ratio=repeat_stop_diversity_top_ratio,
                        device=device,
                    )

                for row, (generation_raw, repetition_meta) in zip(current_batch, generations, strict=True):
                    generation_for_scoring = strip_think_block(generation_raw) if strip_think else generation_raw
                    scorer = SCORERS.get(row["task"], score_contains)
                    score = scorer(generation_for_scoring, row["answer"])
                    per_task_scores[row["task"]].append(score)

                    output_row = {
                        "task": row["task"],
                        "score": score,
                        "gen": generation_raw,
                        "answer": row["answer"],
                    }
                    if repetition_meta:
                        output_row["repetition_stop"] = repetition_meta
                        repetition_stop_count += 1
                    if strip_think and generation_for_scoring != generation_raw:
                        output_row["gen_for_scoring"] = generation_for_scoring

                    f.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except OSError:
                        pass
                    progress.update(1)
            progress.close()

    summary: dict[str, Any] = {
        task: {"n": len(scores), "score": sum(scores) / len(scores)}
        for task, scores in sorted(per_task_scores.items())
        if scores
    }
    total = sum(len(scores) for scores in per_task_scores.values())
    overall = sum(score for scores in per_task_scores.values() for score in scores) / max(total, 1)
    summary["__overall__"] = {"n": total, "score": overall}
    summary["__elapsed_s__"] = time.time() - start_time
    summary["__eval_version__"] = "local_ruler_v1"
    summary["__chat_template__"] = chat_template or None
    summary["__eos_token_ids__"] = eos_token_ids
    summary["__max_new_tokens__"] = max_new_tokens
    summary["__strip_think__"] = strip_think
    summary["__stop_strings__"] = stop_strings
    summary["__assistant_prefill__"] = assistant_prefill
    summary["__batch_size__"] = batch_size
    summary["__run_meta__"] = run_meta
    summary["__repetition_stop__"] = {
        "enabled": repetition_stop,
        "triggered": repetition_stop_count,
        "min_new_tokens": repeat_stop_min_new_tokens,
        "min_ngram": repeat_stop_min_ngram,
        "max_ngram": repeat_stop_max_ngram,
        "repeats": repeat_stop_repeats,
        "diversity_window": repeat_stop_diversity_window,
        "diversity_unique_ratio": repeat_stop_diversity_unique_ratio,
        "diversity_top_ratio": repeat_stop_diversity_top_ratio,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[eval-{length}] overall={overall:.3f} elapsed={summary['__elapsed_s__']:.0f}s samples={total}")
    for task in sorted(by_task):
        task_summary = summary.get(task)
        if task_summary:
            print(f"  {task:<22}: {task_summary['score']:.3f} (n={task_summary['n']})")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local HuggingFace RULER evaluation.")
    parser.add_argument("--model_path", required=True, help="HF checkpoint directory.")
    parser.add_argument("--abbr", required=True, help="Short name used for output directory and logs.")
    parser.add_argument("--lengths", nargs="+", type=int, default=[4096, 8192, 16384])
    parser.add_argument("--n_per_task", type=int, default=50)
    parser.add_argument("--max_seq", type=int, default=32768)
    parser.add_argument("--ruler_root", default=str(EVAL_ROOT / "eval_scripts/ruler"))
    parser.add_argument("--out_root", default=str(EVAL_ROOT / "exp_analysis/ruler_results"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--chat_template", default="", help="Any non-empty value enables tokenizer chat template.")
    parser.add_argument("--eos_token_ids", nargs="+", type=int, default=[])
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--strip_think", action="store_true")
    parser.add_argument("--stop_strings", nargs="*", default=[])
    parser.add_argument("--assistant_prefill", default="")
    parser.add_argument(
        "--disable_ttt_fast_weights",
        action="store_true",
        help="Keep TTT layers enabled but set ttt_lr=0 to disable online fast-weight updates.",
    )
    parser.add_argument(
        "--ttt_prefill_update_partial",
        action="store_true",
        help="Update TTT fast weights with the final incomplete prefill chunk before decoding.",
    )
    parser.add_argument(
        "--ttt_prefill_partial_min_tokens",
        type=int,
        default=None,
        help="Minimum tail tokens required by --ttt_prefill_update_partial.",
    )
    parser.add_argument("--disable_repetition_stop", action="store_true")
    parser.add_argument("--repeat_stop_min_new_tokens", type=int, default=96)
    parser.add_argument("--repeat_stop_min_ngram", type=int, default=8)
    parser.add_argument("--repeat_stop_max_ngram", type=int, default=128)
    parser.add_argument("--repeat_stop_repeats", type=int, default=3)
    parser.add_argument("--repeat_stop_diversity_window", type=int, default=160)
    parser.add_argument("--repeat_stop_diversity_unique_ratio", type=float, default=0.22)
    parser.add_argument("--repeat_stop_diversity_top_ratio", type=float, default=0.18)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_root) / args.abbr
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, config = load_model(
        model_path=args.model_path,
        dtype_name=args.dtype,
        attn_implementation=args.attn_implementation,
        device=args.device,
        disable_ttt_fast_weights=args.disable_ttt_fast_weights,
        ttt_prefill_update_partial=args.ttt_prefill_update_partial,
        ttt_prefill_partial_min_tokens=args.ttt_prefill_partial_min_tokens,
    )
    eos_token_ids = _resolve_eos_token_ids(config, tokenizer, args.eos_token_ids)
    run_meta = {
        "ttt_mode": getattr(config, "ttt_mode", None),
        "ttt_layers": getattr(config, "ttt_layers", None),
        "ttt_lr": getattr(config, "ttt_lr", None),
        "ttt_chunk": getattr(config, "ttt_chunk", None),
        "ttt_target": getattr(config, "ttt_target", None),
        "ttt_prefill_update_partial": getattr(config, "ttt_prefill_update_partial", None),
        "ttt_prefill_partial_min_tokens": getattr(config, "ttt_prefill_partial_min_tokens", None),
        "disable_ttt_fast_weights": args.disable_ttt_fast_weights,
    }

    print(f"[main] output={out_dir}")
    if args.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {args.batch_size}")
    print(f"[main] lengths={args.lengths} eos_token_ids={eos_token_ids} max_seq={args.max_seq} batch_size={args.batch_size}")

    all_summaries = {}
    for length in args.lengths:
        all_summaries[length] = eval_length(
            model=model,
            tokenizer=tokenizer,
            length=length,
            n_per_task=args.n_per_task,
            max_seq=args.max_seq,
            out_dir=out_dir,
            ruler_root=Path(args.ruler_root),
            eos_token_ids=eos_token_ids,
            chat_template=args.chat_template,
            max_new_tokens=args.max_new_tokens,
            strip_think=args.strip_think,
            stop_strings=args.stop_strings,
            assistant_prefill=args.assistant_prefill,
            repetition_stop=not args.disable_repetition_stop,
            repeat_stop_min_new_tokens=args.repeat_stop_min_new_tokens,
            repeat_stop_min_ngram=args.repeat_stop_min_ngram,
            repeat_stop_max_ngram=args.repeat_stop_max_ngram,
            repeat_stop_repeats=args.repeat_stop_repeats,
            repeat_stop_diversity_window=args.repeat_stop_diversity_window,
            repeat_stop_diversity_unique_ratio=args.repeat_stop_diversity_unique_ratio,
            repeat_stop_diversity_top_ratio=args.repeat_stop_diversity_top_ratio,
            device=args.device,
            batch_size=args.batch_size,
            run_meta=run_meta,
        )

    summary_path = out_dir / "summary_all_lengths.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    print(f"[main] saved {summary_path}")


if __name__ == "__main__":
    main()
