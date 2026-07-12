# Spec 11 — Architecture hill-climb: 2026-07 open-weight review

- **Spec ID:** 11_arch_hillclimb
- **Worker tier:** 🟪 Opus — every task here changes the forward pass or the KV/state layout of a model
  that must stay causal (T6.1's hard-won invariant). A silent regression here is exactly the class of bug
  Stage 6 spent itself finding (no causal mask, J-Space leaking the future).
- **Dependencies:** 04_model_and_configs (`model_1b.py`, `TransformerBlock1B`, `YaRNScaledRoPE`), the
  causality test suite from T6.1 (28 tests — every new attention/state variant must pass the same
  "exactly 0.0 logit change at positions < t" test), Stage 9 (scale ladder, open risk #1).
- **Consumers:** Stage 9 (base1b VRAM trim decision), Stage 10 (longer context needs cheaper KV per token).

## Why this exists

This project previously had this content confused with a *different* solo project
(`vector-hoops`, a small ~527K-parameter tabular basketball-stats model). A 2026-07-11 review of six
open-weight LLMs (source: `share.google/ApbX6CzAGagVjbpGY` — Zaya1-8B, VibeThinker-3B, DeepSeek V4 Flash,
Qwen 3.6, Gemma 4 / DiffusionGemma) got written into that project's public-facing site as if it applied to
its 17-tower residual MTNN, which has no sequence, no context length, and no KV-cache. It has been reverted
there (`vector-hoops` commit reverting `c2e5717`). This is where that review actually belongs: `AvaModel1B`
is a real causal transformer (GQA + SwiGLU + YaRN RoPE, `is_causal=True` SDPA) with a real, linearly-growing
KV-cache and a real VRAM ceiling — **open risk #1 in `TODOS.md`: base1b is 1409M params, 20% over the 1.17B
spec, 8.4GB before activations against ~11.6GB usable.** Four of the six reviewed models attack that exact
problem (KV-cache size) from a different angle each. Nothing here is adopted until it passes the same
causality + numerics gates as every other change in this codebase — these are **candidates to falsify**,
not claims to ship.

**All benchmark numbers below are the source vendors' own published cards, not ava-agi measurements.**
Every task's acceptance criterion is a number this repo measures itself, on nano or mini, before any base1b
decision leans on it.

## Candidates, mapped to concrete tasks

### T11.1 — Compressed-latent attention → cheaper KV per token (Zaya1-8B)
Zaya1's approach: project Q/K/V into a shared low-rank latent, mix sequence with a depthwise conv instead
of full attention over the latent, then decode — vendor reports ~8× KV-cache reduction. For `AvaModel1B`
this is an alternative `TransformerBlock1B.attn` path, not a replacement for RoPE or GQA.
*accept:* implement as `attn_mode="compressed_latent"` behind the existing GQA path (config-gated like
`gradient_checkpointing`); causality test suite (T6.1) passes unchanged; on nano, measure actual KV memory
at a fixed context length before/after — report the real ratio, do not assume 8×.

### T11.2 — Gated DeltaNet fixed-state branch → the base1b VRAM answer (Qwen 3.6)
Qwen 3.6's hybrid: most layers run gated DeltaNet (linear-attention with a fixed-size recurrent state that
does not grow with context — delta-rule update + gating decay), a minority stay full-attention for
precision. This is the most directly relevant candidate to open risk #1 and to the base1b trim decision
(`TODOS.md` T9.3: "drop `n_fusion_layers` 28→24, or narrow the workspaces") — a fixed-state layer is a third
option that doesn't shrink capacity. It also has a real conceptual cousin already in this codebase:
J-Space's **chunk-recurrent** broadcast (T6.1's causality fix) already keeps a bounded state across chunks
instead of mean-pooling the whole sequence. `T11.2` should read `ava/j_space_module.py`'s chunk-recurrent
implementation before designing the DeltaNet layer — they may share a state-passing interface.
*accept:* a `DeltaNetBlock` swappable for `TransformerBlock1B` at a config-gated subset of layers; the
causality test suite passes; a needle-in-haystack run (`evals/needle.py`, already YaRN-tested to 2048) shows
state size independent of context length at 2×, 4×, 8× the native window; report actual peak VRAM at
base1b's target context vs the current GQA path — this number, not the vendor's, decides T9.3.

### T11.3 — Sparse/compressed KV hybrid at long context (DeepSeek V4 Flash)
DeepSeek's Flash variant claims 10% of V3.2's KV at 1M context via a compressed-sparse KV path with disk
streaming for the coldest entries. Lower priority than T11.2 (base1b's context target is far short of 1M),
but the "stream cold KV to disk" idea composes with this project's existing disk-is-the-constraint posture
(`PLAN.md`: 28.5GB free, single drive). Track as a future task once T10 (continuous supply) and T11.2 both
land; do not build until base1b's actual context target is set.

### T11.4 — MatFormer nesting → one checkpoint spans the scale ladder (Gemma 4 E2B/E4B)
Gemma 4's MatFormer trains a smaller model as a structural subset of a larger one (shared prefix of
layers/widths), so one download serves multiple deploy targets. This project already has a scale ladder
(nano 13.8M → mini 171.3M → base1b 1409M) trained as three **independent** runs. A MatFormer-style nesting
would let mini's weights be a literal slice of base1b's, which changes T9.2→T9.3's GO/NO-GO logic (mini
would stop being a separate training run and become a checkpoint of base1b's early width). This is a
training-curriculum redesign, not a small patch — write it up as its own spec (`12_matformer_ladder.md`)
before touching `ava/train.py`'s phase manager. Do not start on the current mini run (T9.2, already ~32k
tokens in); target base1b only.

### T11.5 — Per-layer embeddings + discrete diffusion decoding — explicitly out of scope
The review's other two ideas (per-layer token embedding tables for phone-class deploy; DiffusionGemma's
discrete-diffusion decoder) target inference-hardware and decoding-paradigm problems this project doesn't
have: `ava-agi` has no phone-deploy target (`specs/07_serving_deployment.md` is Docker/GPU-server), and a
bidirectional diffusion decoder is a different training objective from the causal LM this repo just spent
Stage 6 making causal — swapping it in is not a hill-climb step, it's a different project. Not tracked as a
task; recorded here only so a future review doesn't re-propose it without reading this line.

## Ordering

T11.2 before T11.1 before T11.3 — T11.2 answers the open VRAM risk directly and has the clearest internal
analog (J-Space chunk-recurrence) to reuse; T11.1 is the next-cheapest KV win; T11.3 only matters once a
long-context target exists. T11.4 is independent and gated on writing `12_matformer_ladder.md` first. T11.5
is a non-task, kept for the record.
