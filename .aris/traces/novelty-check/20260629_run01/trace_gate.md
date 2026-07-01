# Novelty Check Trace — TTT-as-GATE direction

Date: 2026-06-29
Reviewer: gpt-5.4 (xhigh) via Codex MCP
Thread: 019f12e1-3b69-71d0-a56b-b91fefde53b4 (run02, follow-up after φ direction)

## Proposed (pivot): TTT fast-weight memory AS a multiplicative gate, not additive content
- G1 channel gate: g_t=σ(memory readout) ⊙ hidden
- G2 token selection: TTT memory selects which history tokens reach full-attn layers (score dilution)
- G3 cross-layer memory routing

## Verdict
- Novelty rank: G3 > G1 > G2
- Push-RULER / feasibility rank: G2 > G1 > G3
- Best novelty-to-risk-to-effort: G2-lite (TTT-memory-sourced SOFT token selector / attention bias, NOT hard drop)
- Best variant novelty: 4-5/10. PROCEED WITH CAUTION.
- GATE direction novelty > φ direction novelty. But still crowded.

## Closest prior (kill-shots a reviewer would cite)
- GDWM (2601.12906) — gates which chunk to memorize into LoRA; very close.
- SeerAttention-R — trainable gate into attention, trains only gate, near-lossless sparse. ~ G2.
- Token Sparse Attention (ICML2026, 2602.03216), Double-P top-p (2602.05191) — learned token select.
- GLA (2312.06635), Mamba2 (2405.21060), Gated DeltaNet — input-dependent gates already.
- Titans (2501.00663) — test-time memorization with gating/surprise.
- qTTT (2512.13898) — score-dilution framing of long-context failure.

## The distinction that must be defended
"gate from test-time-learned memory" is NOT enough alone (Mamba2 gate is already input-dependent,
Titans already gates fast-weight writes). Must prove STATE-dependence: fix current token, vary history/
fast-weights → gate changes significantly AND raises target-token attention mass / passkey accuracy.

## Reframe to be publishable
"TTT fast weights as a context-adaptive CONTROL plane, not a content memory" + diagnosis experiments
showing gate raises needle-token survival/attention mass.
