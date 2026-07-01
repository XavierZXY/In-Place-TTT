# Novelty Check Trace — shared learnable φ for in-place TTT retrieval

Date: 2026-06-29
Reviewer: gpt-5.4 (xhigh) via Codex MCP
Thread: 019f12e1-3b69-71d0-a56b-b91fefde53b4

## Proposed method
Learnable feature map φ producing key/query for in-place TTT fast-weight memory
(replacing direct use of MLP h), shared across all TTT layers, jointly trained,
base/delta path separated. Goal: fix high-entropy retrieval hurting RULER.

## Verdict
Novelty 3-4/10. PROCEED WITH CAUTION — do as engineering ablation, not standalone paper.

## Closest prior work
- TTT-KVB "Secretly Linear Attention" (2602.21204) — closest idea prior; TTT≈learned linear attn.
- Hedgehog (2402.04347) — learnable feature map to recover spiky softmax-like linear attn.
- LoLCATs (ICLR2025) — trained feature maps, jointly; linearize pretrained LLM.
- In-Place TTT (ICLR2026, 2604.06169) — the base method this repo implements.
- qTTT (2512.13898) — query-only TTT for long-context retrieval; PARTIALLY SCOOPS motivation.
- CLA/LCKV/LISA/LAWCAT — cross-layer Q/K projection sharing.

## Key risks
- "strengthen query/key retrieval to fix long-context" narrative scooped by qTTT.
- learnable φ for linear attn covered by Hedgehog/LoLCATs.
- shared φ across layers is CLA-style, not new.

## Reframing to be publishable
- Diagnosis paper: prove In-Place TTT failure = feature collision/high-entropy retrieval,
  give predictive metrics (key mutual coherence, effective rank, retrieval entropy vs RULER acc).
- New memory rule: φ + delta-rule/erase-write/collision-aware write gate (overcapacity of outer-product memory).
- Theory: shared φ = common address space across layers + layer-local specialized value memory.
- Strong ablations: per-layer vs shared vs fixed-random vs Hedgehog vs Taylor/Based φ, dim sweep, entropy reg.
