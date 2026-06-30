"""Stage-A: sweep eta for offline per-key NLMS using a captured RULER forward.

Hooks each TTT MLP to capture (x, t), reconstructs per-chunk K=pad(h) and
V=pad(ttt_proj(ttt_conv(t))) exactly as the training forward does, then calls
nlms_output_delta_for_etas to report output_delta vs eta per layer/length.

Read-only: never mutates the model. Runs the inner model only (no lm_head) to
avoid OOM at long context.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import hf_models  # noqa: E402,F401  (registers TTT-Qwen3)
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from einops import rearrange  # noqa: E402

from eval.diagnostics.ttt_signal_probe import nlms_output_delta_for_etas, load_samples  # noqa: E402


class TTTKVCapture:
    """Pre-hook capturing per-chunk K=pad(h) and V=pad(ttt_proj(ttt_conv(t)))."""

    def __init__(self, model):
        self.handles = []
        self.kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        inner = getattr(model, "model", model)
        for layer in getattr(inner, "layers", []):
            mlp = getattr(layer, "mlp", None)
            if mlp is None or not hasattr(mlp, "ttt_conv"):
                continue
            self.handles.append(
                mlp.register_forward_pre_hook(self._hook(mlp.layer_idx, mlp), with_kwargs=True)
            )

    def _hook(self, layer_idx, mlp):
        def hook(_m, args, kwargs):
            x = args[0] if args else kwargs.get("x")
            t = kwargs.get("t", args[1] if len(args) > 1 else None)
            if x is None or t is None:
                return None
            with torch.no_grad():
                h = mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)     # [b,s,inter]
                t_padded = mlp.padding(t)                             # [b,nc,c,d_in]
                bs, nc, cs, _ = t_padded.shape
                t_conv = (
                    mlp.ttt_conv(t_padded.transpose(-1, -2).reshape(bs * nc, -1, cs))
                    .transpose(-1, -2).reshape(bs, nc, cs, -1)
                )
                if mlp.ttt_proj is not None:
                    V = torch.einsum("b t c d, d e -> b t c e", t_conv, mlp.ttt_proj.weight)
                else:
                    V = t_conv
                K = mlp.padding(h)                                    # [b,nc,c,inter]
            # store chunk-major for sample 0 (batch=1 in probe)
            self.kv[layer_idx] = (K[0].detach(), V[0].detach())       # [nc,c,inter], [nc,c,d]
            return None
        return hook

    def clear(self):
        self.kv.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--length", type=int, default=16384)
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--etas", type=float, nargs="+", default=[1, 3, 10, 30, 100])
    ap.add_argument("--out", default="results/nlms_eta_sweep.json")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    model = AutoModelForCausalLM.from_pretrained(a.ckpt, torch_dtype=torch.bfloat16).to(a.device).eval()
    inner = getattr(model, "model", model)
    cap = TTTKVCapture(model)
    samples = load_samples(a.data, a.num_samples)
    if not samples:
        raise SystemExit(f"no usable samples in {a.data}")

    # accumulate output_delta per (layer, eta)
    agg: dict[int, dict[float, list]] = {}
    n_used = 0
    for text in samples:
        ids = tok(text, return_tensors="pt", truncation=True, max_length=a.length).input_ids.to(a.device)
        if ids.shape[1] < 2:
            continue
        cap.clear()
        with torch.no_grad():
            inner(input_ids=ids, use_cache=False)
        for li, (K, V) in cap.kv.items():
            W0 = dict(model.named_parameters())  # fetch down_proj for this layer
            # find the mlp to get its W0
            mlp = inner.layers[li].mlp
            res = nlms_output_delta_for_etas(K, V, mlp.down_proj.weight, etas=a.etas, lam=1.0)
            agg.setdefault(li, {e: [] for e in a.etas})
            for e, v in res.items():
                agg[li][e].append(v)
        n_used += 1

    cap.remove()
    # mean over samples, then over layers
    by_layer = {li: {e: sum(vs) / len(vs) for e, vs in d.items()} for li, d in sorted(agg.items())}
    by_eta = {}
    for e in a.etas:
        vals = [by_layer[li][e] for li in by_layer]
        by_eta[e] = sum(vals) / len(vals) if vals else float("nan")

    report = {"ckpt": a.ckpt, "length": a.length, "n_samples": n_used,
              "etas": a.etas, "output_delta_by_eta": by_eta, "by_layer": by_layer}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(f"\n=== NLMS eta sweep (len={a.length}, n={n_used}) ===")
    print(f"{'eta':>8}{'output_delta':>14}{'in[0.2,0.4]?':>14}")
    for e in a.etas:
        od = by_eta[e]
        flag = "YES" if 0.2 <= od <= 0.4 else ""
        print(f"{e:>8}{od:>14.4f}{flag:>14}")
    print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    main()
