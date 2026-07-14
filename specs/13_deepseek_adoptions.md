# 13 — DeepSeek-lineage adoptions for Ava

Triage of the efficiency stack (per user brief, 2026-07-13) against Ava's
reality: 171M `mini` mid-run on a 12GB laptop GPU, `nano` as the cheap
experiment rung, `base1b` as the target. Rule: nothing lands mid-run on
`mini` (checkpoint compatibility); everything enters via nano first.

## Adopted now (shipped)

**Muon optimizer** (`ava/optim.py`, `optimizer.name: "muon"`), the
orthogonalized-momentum method (Jordan et al. 2024) with the **Moonlight
scaling recipe** (arXiv 2502.16982, adversarially verified 2026-07-13):
updates rescaled by `0.2·√max(A,B)` to match AdamW's update RMS + decoupled
weight decay, after which Muon reuses the AdamW-tuned LR directly — one WSD
schedule drives both optimizers. Hidden matrices get Muon (one momentum
buffer — half of AdamW's optimizer memory, ~0.6GB back on mini-class runs);
embeddings/heads/norms/decay-logits keep AdamW (the split Jordan, Moonlight,
and Essential AI all land on independently). Validated efficiency, honestly
sized: **1.35x token-efficiency at 124M** (NanoGPT speedrun) and **~25%
compute reduction at 1.5B**; the circulated "~2x" figure is a vendor
scaling-law self-report refuted 0-3 under independent benchmarking
(Stanford "Fantastic Pretraining Optimizers") — do not plan around it.
Gate to flip it on: nano A/B (same data/seed, adamw vs muon) — adopt for
the mini successor if muon reaches adamw's step-3000 loss in ≤0.8x steps.
**First datapoint (2026-07-14, scripts/muon_ab_ocr.py, CPU pilot scale —
d=128, byte vocab, OCR task): AdamW won.** Final loss 0.1325 vs Muon's
0.1804 at identical init/data/LR; Muon never reached AdamW's bar in 300
steps. Published Muon gains live at 124M+ (768-dim) — Ava's small-matrix
J-space regime may sit outside them. The gate stands, the prior is now
skeptical: do NOT enable Muon for any real run until the nano-preset GPU
A/B clears it.

## Next in line (ordered, each gated on the previous)

1. **GRPO post-training** (DeepSeekMath). Critic-free RL: sample k
   completions per prompt, advantage = groupwise z-score of rewards. Fits
   12GB precisely because there is NO value network. Ava has the missing
   piece already: `ava/datagen/math_gen.py` / `logic.py` COMPUTE ground
   truths, so rewards are exact verifiers, not reward models. Plan: new
   `ava/rl/grpo.py` driving `serve_engine` generation on the chat/math
   branch after P5; k=8, seq 512, verifiable-answer reward + format reward.
2. **Self-verification shaping** (DeepSeekMath V2 recipe). Add a
   check-then-answer format to GRPO prompts; reward bonus when the model's
   self-check verdict agrees with the verifier, penalty when it emits a
   final answer its own check flagged. Pure reward-shaping on top of (1).
3. **MLA / compressed KV** (FlashMLA territory). Irrelevant to training at
   seq≤4096 (no KV cache in fwd/bwd), decisive for base1b's 131k-ctx
   serving. Adopt at base1b design time as a config-gated attention class;
   do NOT retrofit mini. The existing `attention/compressed_conv.py` and
   `sparse_compressed.py` stubs are the natural home.
4. **mHC (manifold-constrained hyper-connections)**. Replace residual adds
   with constrained learned mixing across layer streams. Genuinely
   promising for base1b stability, but it is an architecture change:
   prototype as `--hyper-connections` on nano only, compare loss curves at
   equal tokens, then decide for base1b. Not before (1)-(3) pay rent.
5. **Reasoning distillation**. The repo already points at local teachers
   (`OLLAMA_GUIDE.md`, qwen3:32b; `on_policy_distill.py`). The DeepSeek-R1
   recipe (teacher thinking-tokens → small-model SFT) maps to a post-P5
   branch: capture teacher traces OFFLINE into RAW shards (datagen's
   no-network rule applies to generators, not to a collector-side teacher
   source), then SFT the chat branch. Needs a `teacher_traces` source spec
   + provenance/decontamination pass before it touches the manifest.

## Explicitly matched to work already in this repo

- **Janus-style decoupled encoders**: the pxpipe optical arm
  (specs/12) already separates the understanding encoder (page-patch →
  VisionEncoder) from any future generation path — same decoupling logic.
- **Synthetic:real data curricula**: Ava's datagen-heavy phase mixes are
  the same philosophy (the P2 collector fix restored it); the 1:1
  aesthetic-stabilization idea becomes relevant only if a generative
  image head is ever added.

## Non-goals at this scale

MoE routing (the 4-workspace J-space router IS Ava's expert-routing bet),
1M-token contexts on a 12GB card, and 8-bit optimizer swaps mid-run
(resume-state incompatibility; revisit at the next cold start).
