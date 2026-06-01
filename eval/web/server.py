#!/usr/bin/env python3
"""Serve the RULER results dashboard with a dynamic local JSON API."""

from __future__ import annotations

import argparse
import json
import mimetypes
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


WEB_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = WEB_ROOT.parent / "exp_analysis" / "ruler_results"
SUMMARY_FILENAMES = ("summary_all_lengths.json", "summary_all_lenghts.json")


def _score_node(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    score = value.get("score")
    if not isinstance(score, (int, float)):
        return None
    result: dict[str, Any] = {"score": float(score)}
    n = value.get("n")
    if isinstance(n, int):
        result["n"] = n
    return result


def _first_meta_value(length_nodes: list[dict[str, Any]], key: str) -> Any:
    for node in length_nodes:
        if key in node:
            return node[key]
    return None


def _infer_method(name: str, has_chat_template: bool) -> str:
    lower = name.lower()
    if "smoke" in lower:
        return "smoke"
    if lower.endswith("-chat") or lower.endswith("-chattpl") or has_chat_template:
        return "chat"
    if lower.endswith("-base") or lower.endswith("-nochattpl") or "-base-" in lower:
        return "base"
    return "other"


def _find_summary_file(result_dir: Path) -> Path | None:
    for filename in SUMMARY_FILENAMES:
        path = result_dir / filename
        if path.is_file():
            return path
    return None


def scan_results(results_root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    if not results_root.exists():
        return {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "resultsRoot": str(results_root),
            "records": records,
            "skipped": [{"name": str(results_root), "reason": "results root does not exist"}],
        }

    for result_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        summary_path = _find_summary_file(result_dir)
        if summary_path is None:
            continue

        try:
            with summary_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            skipped.append({"name": result_dir.name, "reason": str(exc)})
            continue

        if not isinstance(raw, dict):
            skipped.append({"name": result_dir.name, "reason": "summary root is not an object"})
            continue

        summaries: dict[str, Any] = {}
        length_nodes: list[dict[str, Any]] = []
        task_names: set[str] = set()

        for length_key, length_value in raw.items():
            if not str(length_key).isdigit() or not isinstance(length_value, dict):
                continue

            length = str(int(length_key))
            length_nodes.append(length_value)
            tasks: dict[str, Any] = {}

            for task_name, task_value in length_value.items():
                if task_name.startswith("__"):
                    continue
                task_score = _score_node(task_value)
                if task_score is None:
                    continue
                tasks[task_name] = task_score
                task_names.add(task_name)

            summaries[length] = {
                "tasks": dict(sorted(tasks.items())),
                "overall": _score_node(length_value.get("__overall__")),
                "elapsed_s": length_value.get("__elapsed_s__"),
                "batch_size": length_value.get("__batch_size__"),
                "max_new_tokens": length_value.get("__max_new_tokens__"),
                "eval_version": length_value.get("__eval_version__"),
                "repetition_stop": length_value.get("__repetition_stop__"),
            }

        if not summaries:
            skipped.append({"name": result_dir.name, "reason": "no numeric length entries"})
            continue

        chat_template = _first_meta_value(length_nodes, "__chat_template__")
        has_chat_template = chat_template not in (None, "", False)
        is_smoke = "smoke" in result_dir.name.lower()
        method = _infer_method(result_dir.name, has_chat_template)

        try:
            summary_file = str(summary_path.relative_to(results_root.parent.parent))
        except ValueError:
            summary_file = str(summary_path)

        records.append(
            {
                "id": result_dir.name,
                "name": result_dir.name,
                "method": method,
                "isSmoke": is_smoke,
                "hasChatTemplate": has_chat_template,
                "chatTemplate": chat_template,
                "lengths": sorted((int(length) for length in summaries.keys())),
                "tasks": sorted(task_names),
                "summaries": summaries,
                "summaryFile": summary_file,
                "modifiedAt": datetime.fromtimestamp(summary_path.stat().st_mtime, timezone.utc).isoformat(),
            }
        )

    all_lengths = sorted({length for record in records for length in record["lengths"]})
    all_tasks = sorted({task for record in records for task in record["tasks"]})

    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "resultsRoot": str(results_root),
        "summaryFilenames": list(SUMMARY_FILENAMES),
        "records": records,
        "lengths": all_lengths,
        "tasks": all_tasks,
        "skipped": skipped,
    }


class DashboardHandler(SimpleHTTPRequestHandler):
    results_root = DEFAULT_RESULTS_ROOT

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/ruler-results":
            self._send_json(scan_results(self.results_root))
            return
        if parsed.path == "/api/health":
            self._send_json({"ok": True})
            return
        super().do_GET()

    def translate_path(self, path: str) -> str:
        parsed_path = urlparse(path).path
        decoded_path = unquote(parsed_path).lstrip("/")
        if not decoded_path:
            decoded_path = "index.html"
        candidate = (WEB_ROOT / decoded_path).resolve()
        try:
            candidate.relative_to(WEB_ROOT)
        except ValueError:
            return str(WEB_ROOT / "index.html")
        return str(candidate)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def guess_type(self, path: str) -> str:
        if path.endswith(".js"):
            return "application/javascript"
        return mimetypes.guess_type(path)[0] or "application/octet-stream"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the RULER results dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address.")
    parser.add_argument("--port", type=int, default=8765, help="Bind port.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Directory containing RULER result subdirectories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DashboardHandler.results_root = args.results_root.resolve()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Serving RULER dashboard at http://{args.host}:{args.port}/")
    print(f"Scanning results from {DashboardHandler.results_root}")
    server.serve_forever()


if __name__ == "__main__":
    main()
