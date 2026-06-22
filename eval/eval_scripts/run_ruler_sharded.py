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

"""Launch sharded local RULER eval jobs over multiple GPUs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = EVAL_ROOT.parent


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path")
    parser.add_argument("abbr")
    parser.add_argument("--lengths", default=os.getenv("LENGTHS", "4096 8192 16384"))
    parser.add_argument("--gpus", default=os.getenv("GPUS", "0 1 2 3 4 5 6 7"))
    parser.add_argument("--num_shards", type=int, default=int(os.getenv("NUM_SHARDS", "16")))
    parser.add_argument("--n_per_task", type=int, default=int(os.getenv("N_PER_TASK", "100")))
    parser.add_argument("--max_seq", type=int, default=int(os.getenv("MAX_SEQ", "32768")))
    parser.add_argument("--max_new_tokens", type=int, default=int(os.getenv("MAX_NEW_TOKENS", "1024")))
    parser.add_argument("--batch_size", type=int, default=int(os.getenv("BATCH_SIZE", "1")))
    parser.add_argument("--ruler_root", default=os.getenv("RULER_ROOT", str(SCRIPT_DIR / "ruler")))
    parser.add_argument("--out_root", default=os.getenv("OUT_ROOT", str(EVAL_ROOT / "exp_analysis/ruler_results")))
    parser.add_argument("--log_root", default=os.getenv("LOG_ROOT", str(EVAL_ROOT / "exp_analysis/logs/ruler")))
    parser.add_argument("--dtype", default=os.getenv("DTYPE", "bfloat16"))
    parser.add_argument("--attn_implementation", default=os.getenv("ATTN_IMPLEMENTATION", "sdpa"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    python = Path(os.getenv("PYTHON", str(PROJECT_ROOT / ".venv/bin/python")))
    eval_script = SCRIPT_DIR / "eval_ruler.py"
    lengths = parse_int_list(args.lengths)
    gpu_ids = parse_int_list(args.gpus)

    log_dir = Path(args.log_root) / args.abbr
    out_dir = Path(args.out_root) / args.abbr
    log_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"model_path={args.model_path}", flush=True)
    print(f"abbr={args.abbr}", flush=True)
    print(f"lengths={' '.join(map(str, lengths))}", flush=True)
    print(f"gpus={' '.join(map(str, gpu_ids))}", flush=True)
    print(f"num_shards={args.num_shards}", flush=True)
    print(f"n_per_task={args.n_per_task}", flush=True)
    print(f"max_seq={args.max_seq}", flush=True)
    print(f"ruler_root={args.ruler_root}", flush=True)
    print(f"out_dir={out_dir}", flush=True)
    print(f"log_dir={log_dir}", flush=True)

    processes: list[tuple[int, subprocess.Popen[bytes], Path]] = []
    for shard_id in range(args.num_shards):
        gpu_id = gpu_ids[shard_id % len(gpu_ids)]
        log_path = log_dir / f"shard_{shard_id:02d}.log"
        cmd = [
            str(python),
            str(eval_script),
            "--model_path",
            args.model_path,
            "--abbr",
            args.abbr,
            "--lengths",
            *map(str, lengths),
            "--n_per_task",
            str(args.n_per_task),
            "--max_seq",
            str(args.max_seq),
            "--ruler_root",
            args.ruler_root,
            "--out_root",
            args.out_root,
            "--dtype",
            args.dtype,
            "--attn_implementation",
            args.attn_implementation,
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--batch_size",
            str(args.batch_size),
            "--num_shards",
            str(args.num_shards),
            "--shard_id",
            str(shard_id),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        with log_path.open("wb") as log_file:
            process = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        print(f"started shard={shard_id:02d}/{args.num_shards} gpu={gpu_id} pid={process.pid} log={log_path}", flush=True)
        processes.append((shard_id, process, log_path))

    failed = []
    for shard_id, process, log_path in processes:
        return_code = process.wait()
        print(f"finished shard={shard_id:02d} return_code={return_code}", flush=True)
        if return_code != 0:
            failed.append((shard_id, return_code, log_path))

    if failed:
        for shard_id, return_code, log_path in failed:
            print(f"failed shard={shard_id:02d} return_code={return_code} log={log_path}", file=sys.stderr, flush=True)
        return 1

    merge_log = log_dir / "merge.log"
    merge_cmd = [
        str(python),
        str(eval_script),
        "--merge_shards_only",
        "--abbr",
        args.abbr,
        "--lengths",
        *map(str, lengths),
        "--num_shards",
        str(args.num_shards),
        "--out_root",
        args.out_root,
    ]
    with merge_log.open("wb") as log_file:
        merge_result = subprocess.run(merge_cmd, cwd=PROJECT_ROOT, stdout=log_file, stderr=subprocess.STDOUT)
    print(f"merge return_code={merge_result.returncode} log={merge_log}", flush=True)
    return merge_result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
