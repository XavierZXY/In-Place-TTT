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

"""Zero-training diagnostic gate for In-Place TTT long-context degradation.

This script does NOT train anything. It runs forward passes on an existing HF
checkpoint to decide *which* of three competing root-cause hypotheses for
long-context RULER degradation is real, before committing GPU hours to a fix:

  (1) under-writing   — the fast weights barely change (delta too small)
  (2) gate saturation — the un-normalized 6144-d gate ``h_t . h_c'`` is
                         dominated by a few massive-norm tokens
  (3) unbounded accum — readout noise grows with sqrt(T*C) (decay needed)

It emits three probes, each mapped to one hypothesis:

  Probe A (h-norm distribution): per-token / per-channel norm stats of the TTT
     key ``h = SiLU(gate)*up``. Heavy-tailed -> hypothesis (2) plausible.
  Probe B (offline key-norm injection): re-run the TTT readout with an RMSNorm
     spliced onto the key path, WITHOUT training, and measure how much the gate
     selectivity (argmax retrieval / entropy) changes. No change -> key-norm is
     a no-op here, hypothesis (2) refuted.
  Probe C (delta strength vs length): recompute ``output_delta`` /
     ``delta_weight_cumsum`` at several context lengths. Flat & tiny -> (1)
     under-writing; growing with length -> (3) unbounded accumulation.

Decision rule (printed at the end): if Probe A shows no heavy tail AND Probe B
shows no retrieval change -> do NOT pursue key normalization; investigate
under-writing (ttt_lr / value depth) instead.

Usage:
    python eval/diagnostics/ttt_signal_probe.py \
        --ckpt outputs/.../global_step_20000/hf_ckpt \
        --tokenizer /zouxiangyu/models/Qwen/Qwen3-1.7B \
        --data /workspace/.../val.jsonl \
        --lengths 4096 16384 32768 65536 \
        --num-samples 8 \
        --out results/ttt_diag
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Training-side model: full chunk-wise forward + monitor hooks live here.
import hf_models  # noqa: E402,F401  (registers TTT-Qwen3 with AutoModel)
from transformers import AutoModelForCausalLM  # noqa: E402


# --------------------------------------------------------------------------- #
# Capture: hook every TTT MLP to grab the key ``h`` it builds, per forward.
# --------------------------------------------------------------------------- #
class TTTKeyCapture:
    """Forward pre-hook on TTT MLPs to recompute and stash the key ``h``.

    The MLP does not expose ``h`` as an output, so we recompute it from the
    layer input with the same ops the forward uses (gate/up/act). This is a
    read-only probe: it never mutates the module or its outputs.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.handles = []
        self.captured: dict[int, torch.Tensor] = {}
        self._register(model)

    def _register(self, model: torch.nn.Module) -> None:
        inner = getattr(model, "model", model)
        for layer in getattr(inner, "layers", []):
            mlp = getattr(layer, "mlp", None)
            if mlp is None or not hasattr(mlp, "ttt_conv"):
                continue
            self.handles.append(
                mlp.register_forward_pre_hook(self._build_hook(mlp.layer_idx, mlp), with_kwargs=True)
            )

    def _build_hook(self, layer_idx: int, mlp: torch.nn.Module):
        def hook(_module, args, kwargs):
            x = args[0] if args else kwargs.get("x")
            if x is None:
                return None
            with torch.no_grad():
                h = mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)  # [b, s, intermediate]
            self.captured[layer_idx] = h.detach()
            return None

        return hook

    def clear(self) -> None:
        self.captured.clear()

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


# --------------------------------------------------------------------------- #
# Probe A: h-norm distribution (hypothesis 2 — gate saturation by massive norm)
# --------------------------------------------------------------------------- #
def probe_h_norm(captured: dict[int, torch.Tensor]) -> dict[int, dict[str, float]]:
    stats: dict[int, dict[str, float]] = {}
    for layer_idx, h in captured.items():
        hf = h.float().reshape(-1, h.shape[-1])  # [N_tokens, d]
        token_norm = hf.norm(dim=-1)  # per-token L2 over channels
        # per-channel max-abs across tokens: detects a few huge channels
        chan_max = hf.abs().amax(dim=0)
        chan_median = hf.abs().median(dim=0).values.clamp_min(1e-9)
        # "massive channel" = channel whose max-abs >> typical channel
        massive_ratio = (chan_max / chan_max.median().clamp_min(1e-9))
        stats[layer_idx] = {
            "token_norm_median": token_norm.median().item(),
            "token_norm_p99": token_norm.quantile(0.99).item(),
            "token_norm_max": token_norm.max().item(),
            # tail ratio > ~3 suggests a few tokens dominate the gate
            "token_norm_p99_over_median": (token_norm.quantile(0.99) / token_norm.median().clamp_min(1e-9)).item(),
            "token_norm_max_over_median": (token_norm.max() / token_norm.median().clamp_min(1e-9)).item(),
            "channel_massive_max_over_median": massive_ratio.max().item(),
            "channel_outlier_frac_gt10x": (massive_ratio > 10.0).float().mean().item(),
        }
    return stats


# --------------------------------------------------------------------------- #
# Probe B: offline key-norm injection (does normalizing the key change which
# token the gate retrieves? if not, key-norm is a no-op -> hypothesis 2 refuted)
# --------------------------------------------------------------------------- #
def probe_keynorm_effect(captured: dict[int, torch.Tensor], eps: float = 1e-6) -> dict[int, dict[str, float]]:
    """Compare the within-sequence gate matrix G = h @ h^T before/after RMSNorm.

    We do NOT need the value V_hat to answer "does the gate's *selection*
    change": that is fully determined by the gate matrix. We measure, for each
    query token, whether the argmax key (who it retrieves most) and the softmax
    entropy of its gate row shift after normalization.
    """
    out: dict[int, dict[str, float]] = {}
    for layer_idx, h in captured.items():
        hf = h.float()[0]  # [s, d]  (single sample)
        s = hf.shape[0]
        if s < 2:
            continue
        # cap sequence to keep the s*s gate matrix affordable
        cap = min(s, 4096)
        hf = hf[:cap]
        # RMSNorm with weight=1 (identity-init): pure variance normalization
        h_norm = hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + eps)

        g_raw = hf @ hf.t()  # [cap, cap]
        g_norm = h_norm @ h_norm.t()
        # causal: each query only attends to keys <= itself
        causal = torch.tril(torch.ones_like(g_raw, dtype=torch.bool))
        neg = torch.finfo(g_raw.dtype).min
        g_raw = g_raw.masked_fill(~causal, neg)
        g_norm = g_norm.masked_fill(~causal, neg)

        argmax_raw = g_raw.argmax(dim=-1)
        argmax_norm = g_norm.argmax(dim=-1)
        argmax_shift = (argmax_raw != argmax_norm).float().mean().item()

        # softmax entropy per query row (selectivity: low entropy = spiky)
        p_raw = torch.softmax(g_raw, dim=-1)
        p_norm = torch.softmax(g_norm, dim=-1)
        ent_raw = -(p_raw.clamp_min(1e-12).log() * p_raw).sum(-1).mean().item()
        ent_norm = -(p_norm.clamp_min(1e-12).log() * p_norm).sum(-1).mean().item()

        out[layer_idx] = {
            "seq_used": float(cap),
            "argmax_retrieval_shift_frac": argmax_shift,
            "gate_entropy_raw": ent_raw,
            "gate_entropy_norm": ent_norm,
            "gate_entropy_delta": ent_norm - ent_raw,
        }
    return out


# --------------------------------------------------------------------------- #
# Probe D: capacity / collision law (R006 — predict RULER failure)
#   These are the metrics the experiment plan tracks to test whether memory
#   cross-talk predicts retrieval failure.
# --------------------------------------------------------------------------- #
def effective_rank(h: torch.Tensor, eps: float = 1e-12) -> float:
    """Spectral entropy (exp of Shannon entropy of normalized singular values^2).

    h: [seq, d] key matrix. Full for orthonormal keys, ~1 for collinear keys.
    """
    hf = h.float()
    # singular values of the key matrix; s^2 are the eigenvalues of the Gram.
    s = torch.linalg.svdvals(hf)
    p = (s * s)
    total = p.sum().clamp_min(eps)
    p = (p / total).clamp_min(eps)
    entropy = -(p * p.log()).sum()
    return float(entropy.exp())


def write_collision(h: torch.Tensor, eps: float = 1e-12) -> float:
    """Mean fraction of each key's energy already spanned by the earlier keys.

    For key i, project onto the span of keys[:i] and measure ||proj||^2/||k_i||^2.
    ~0 for orthogonal keys, ~1 for collinear keys. Causal: key i only sees k<i.

    Implemented with an incremental orthonormal basis (modified Gram-Schmidt):
    collision_i = 1 - ||k_i - sum_b (k_i . b) b||^2 / ||k_i||^2, where {b} is the
    orthonormal basis of span(keys[:i]). This is exact and O(seq * d * rank),
    vastly faster than a per-key lstsq loop.
    """
    hf = h.float()
    n = hf.shape[0]
    if n < 2:
        return 0.0
    d = hf.shape[1]
    basis = torch.zeros(0, d, dtype=hf.dtype, device=hf.device)  # [rank, d] orthonormal rows
    collisions: list[float] = []
    for i in range(n):
        ki = hf[i]
        norm2 = (ki * ki).sum()
        if norm2 <= eps:
            new_dir = ki  # degenerate; nothing to add
        else:
            if basis.shape[0] > 0:
                coeffs = basis @ ki                       # [rank]
                proj = coeffs @ basis                     # [d]
                residual = ki - proj
                if i >= 1:
                    proj_energy = (proj * proj).sum() / norm2
                    collisions.append(float(proj_energy.clamp(0.0, 1.0)))
            else:
                residual = ki
            # extend the basis with the (normalized) residual direction
            r_norm = residual.norm()
            if r_norm > 1e-6:
                basis = torch.cat([basis, (residual / r_norm).unsqueeze(0)], dim=0)
    return sum(collisions) / len(collisions) if collisions else 0.0


def nlms_output_delta_for_etas(K, V, W0, etas, lam=1.0):
    """Offline per-key NLMS readout: relative output_delta vs base, per eta.

    K: [chunk_num, c, h_dim] keys (= h) per chunk
    V: [chunk_num, c, d]     values (= ttt_proj(ttt_conv(t))) per chunk
    W0: [d, h_dim]           base down_proj weight
    Returns {eta: mean ||delta_out|| / ||out|| over chunks}.
    """
    import torch
    Kf = K.float(); Vf = V.float(); W0f = W0.float()
    chunk_num = Kf.shape[0]
    out = {}
    for eta in etas:
        S = torch.zeros_like(W0f)              # [d, h_dim]
        num = 0.0; den = 0.0
        for i in range(chunk_num):
            Ki = Kf[i]; Vi = Vf[i]             # [c, h_dim], [c, d]
            base_i = torch.einsum("d h, c h -> c d", W0f, Ki)
            delta_i = torch.einsum("c h, d h -> c d", Ki, S)
            out_i = base_i + delta_i
            num += float((out_i - base_i).norm())
            den += float(out_i.norm().clamp_min(1e-12))
            # per-key residual write
            pred_i = torch.einsum("c h, d h -> c d", Ki, S)
            resid_i = (Vi - pred_i) / (lam + (Ki * Ki).sum(dim=-1, keepdim=True))
            S = S + eta * torch.einsum("c d, c h -> d h", resid_i, Ki)
        out[eta] = num / max(den, 1e-12)
    return out


def probe_capacity_law(
    captured: dict[int, torch.Tensor], rank_cap: int = 2048, collision_cap: int = 256
) -> dict[int, dict[str, float]]:
    """Per-TTT-layer capacity metrics from the captured key h (single sample).

    effective_rank uses a fast SVD (cap rank_cap rows). write_collision runs an
    O(seq^2 * d) Gram-Schmidt loop, so it is computed on an evenly-strided
    subsample of collision_cap keys — a representative slice that preserves the
    cross-talk structure while staying bounded for 16k/32k sequences.
    """
    out: dict[int, dict[str, float]] = {}
    for layer_idx, h in captured.items():
        hf = h.float()[0]               # [seq, d]
        seq = hf.shape[0]
        if seq < 2:
            continue
        rank_slice = hf[:rank_cap]
        if seq > collision_cap:
            idx = torch.linspace(0, seq - 1, collision_cap).long()
            coll_slice = hf[idx]
        else:
            coll_slice = hf
        out[layer_idx] = {
            "effective_rank": effective_rank(rank_slice),
            "write_collision": write_collision(coll_slice),
        }
    return out


# --------------------------------------------------------------------------- #
# Probe C: delta strength vs context length (hypothesis 1 vs 3)
# Reads the model's own monitor stats already wired into the forward.
# --------------------------------------------------------------------------- #
def pop_monitor_stats(model: torch.nn.Module) -> dict[int, dict[str, float]]:
    inner = getattr(model, "model", model)
    result: dict[int, dict[str, float]] = {}
    for layer in getattr(inner, "layers", []):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        stats = getattr(mlp, "_last_ttt_monitor_stats", None)
        if not stats:
            continue
        result[mlp.layer_idx] = {k: float(v) for k, v in stats.items() if torch.is_tensor(v)}
    return result


# --------------------------------------------------------------------------- #
# Data + model
# --------------------------------------------------------------------------- #
def load_samples(data_path: str, num_samples: int) -> list[str]:
    texts: list[str] = []
    with open(data_path) as f:
        for line in f:
            if len(texts) >= num_samples:
                break
            obj = json.loads(line)
            msgs = obj.get("messages", [])
            # Concatenate the user turn(s); the needle lives in the user content.
            text = "\n".join(m.get("content", "") for m in msgs if m.get("role") == "user")
            if text.strip():
                texts.append(text)
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="HF checkpoint dir (hf_ckpt)")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data", required=True, help="jsonl with RULER-style messages")
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 16384, 32768, 65536])
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="results/ttt_diag")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype=dtype)
    model.to(args.device).eval()
    # NOTE: _record_ttt_monitor_stats (modeling_qwen3.py:236) is called
    # unconditionally inside the TTT path — it is NOT gated on training mode — so
    # eval() suffices and avoids dropout / the future-chunk-aux branch (:180).
    # The TTT path (and thus a non-zero output_delta) only triggers when
    # seq_len > ttt_chunk, so --lengths must exceed the checkpoint's ttt_chunk.

    capture = TTTKeyCapture(model)
    samples = load_samples(args.data, args.num_samples)
    if not samples:
        raise SystemExit(f"no usable samples in {args.data}")

    report: dict[str, dict] = {"ckpt": args.ckpt, "by_length": {}}

    for length in args.lengths:
        agg_hnorm: dict[int, list[dict[str, float]]] = {}
        agg_keyeff: dict[int, list[dict[str, float]]] = {}
        agg_monitor: dict[int, list[dict[str, float]]] = {}
        agg_capacity: dict[int, list[dict[str, float]]] = {}
        per_sample_capacity: list[dict[str, Any]] = []

        for sample_idx, text in enumerate(samples):
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=length)
            input_ids = ids["input_ids"].to(args.device)
            if input_ids.shape[1] < 2:
                continue
            capture.clear()
            with torch.no_grad():
                # Call the inner model (skip lm_head): the monitor stats and the
                # captured key ``h`` are produced inside the decoder layers, so we
                # never need the [b, seq, vocab] logits — computing them OOMs at 64k.
                inner = getattr(model, "model", model)
                inner(input_ids=input_ids, use_cache=False)

            for li, st in probe_h_norm(capture.captured).items():
                agg_hnorm.setdefault(li, []).append(st)
            for li, st in probe_keynorm_effect(capture.captured).items():
                agg_keyeff.setdefault(li, []).append(st)
            for li, st in pop_monitor_stats(model).items():
                agg_monitor.setdefault(li, []).append(st)
            cap_stats = probe_capacity_law(capture.captured)
            for li, st in cap_stats.items():
                agg_capacity.setdefault(li, []).append(st)
            # per-sample record (for joining with RULER correctness → AUC downstream)
            per_sample_capacity.append({"sample_index": sample_idx, "by_layer": cap_stats})

        def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
            if not rows:
                return {}
            keys = rows[0].keys()
            return {k: sum(r[k] for r in rows) / len(rows) for k in keys}

        report["by_length"][str(length)] = {
            "probe_a_h_norm": {li: _mean(rows) for li, rows in sorted(agg_hnorm.items())},
            "probe_b_keynorm_effect": {li: _mean(rows) for li, rows in sorted(agg_keyeff.items())},
            "probe_c_delta_strength": {li: _mean(rows) for li, rows in sorted(agg_monitor.items())},
            "probe_d_capacity_law": {li: _mean(rows) for li, rows in sorted(agg_capacity.items())},
            "probe_d_per_sample": per_sample_capacity,
        }
        print(f"[length={length}] processed {len(samples)} samples")

    capture.remove()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ttt_signal_probe.json"
    out_path.write_text(json.dumps(report, indent=2))

    _print_verdict(report)
    print(f"\nFull report written to {out_path}")


def _print_verdict(report: dict) -> None:
    """Apply the decision rule across lengths and print a routed conclusion."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC VERDICT (zero-training gate)")
    print("=" * 70)

    lengths = sorted(report["by_length"], key=int)
    if not lengths:
        print("no data")
        return

    def avg_over_layers(length: str, probe: str, field: str) -> float | None:
        layers = report["by_length"][length][probe]
        vals = [v[field] for v in layers.values() if field in v]
        return sum(vals) / len(vals) if vals else None

    # Hypothesis 2: heavy-tailed h-norm + key-norm changes retrieval
    long = lengths[-1]
    tail = avg_over_layers(long, "probe_a_h_norm", "token_norm_max_over_median")
    chan_outlier = avg_over_layers(long, "probe_a_h_norm", "channel_outlier_frac_gt10x")
    retrieval_shift = avg_over_layers(long, "probe_b_keynorm_effect", "argmax_retrieval_shift_frac")

    print(f"\n[Probe A] @len={long}: token_norm max/median = {tail!r}, "
          f"channel outlier frac(>10x) = {chan_outlier!r}")
    print(f"[Probe B] @len={long}: key-norm retrieval argmax shift = {retrieval_shift!r}")

    # Hypothesis 1 vs 3: delta strength flat (under-write) vs growing (unbounded)
    short, longest = lengths[0], lengths[-1]
    od_short = avg_over_layers(short, "probe_c_delta_strength", "output_delta_sample_ratio")
    od_long = avg_over_layers(longest, "probe_c_delta_strength", "output_delta_sample_ratio")
    cw_short = avg_over_layers(short, "probe_c_delta_strength", "delta_weight_cumsum_sample_ratio")
    cw_long = avg_over_layers(longest, "probe_c_delta_strength", "delta_weight_cumsum_sample_ratio")
    print(f"[Probe C] output_delta: {short}->{od_short!r}, {longest}->{od_long!r}")
    print(f"[Probe C] delta_weight_cumsum: {short}->{cw_short!r}, {longest}->{cw_long!r}")

    print("\n--- routed conclusion ---")
    h2_alive = (tail is not None and tail > 3.0) or (chan_outlier is not None and chan_outlier > 0.0)
    h2_actionable = retrieval_shift is not None and retrieval_shift > 0.05
    if h2_alive and h2_actionable:
        print("H2 (gate saturation) SUPPORTED: h-norm is heavy-tailed AND key-norm")
        print("   shifts retrieval -> key normalization is worth a 2-arm training test.")
    elif not h2_alive or (retrieval_shift is not None and retrieval_shift <= 0.05):
        print("H2 (gate saturation) WEAK/REFUTED: key-norm barely changes retrieval.")
        print("   -> Do NOT pursue key normalization first. Investigate under-writing:")
        print("      check if output_delta/delta_weight are tiny and ttt_lr too small.")

    if cw_long is not None and cw_short is not None:
        if cw_long > 1.5 * cw_short:
            print("H3 (unbounded accumulation) PLAUSIBLE: delta_weight grows with length")
            print("   -> consider a decay/forgetting gate (GLA/Mamba2 style), not key-norm.")
        elif cw_long is not None and cw_long < 0.02:
            print("H1 (under-writing) PLAUSIBLE: cumulative delta_weight is tiny (<2%)")
            print("   -> strengthen writing (raise ttt_lr / deepen value), not normalize.")
    print("=" * 70)


if __name__ == "__main__":
    main()
