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
there (`vector-hoops` commit reverting `c2e5717`). This is where that review actually belongs: `DottieModel1B`
is a real causal transformer (GQA + SwiGLU + YaRN RoPE, `is_causal=True` SDPA) with a real, linearly-growing
KV-cache and a real VRAM ceiling — **open risk #1 in `TODOS.md`: base1b is 1409M params, 20% over the 1.17B
spec, 8.4GB before activations against ~11.6GB usable.** Four of the six reviewed models attack that exact
problem (KV-cache size) from a different angle each. Nothing here is adopted until it passes the same
causality + numerics gates as every other change in this codebase — these are **candidates to falsify**,
not claims to ship.

**All benchmark numbers below are the source vendors' own published cards, not dottie-agi measurements.**
Every task's acceptance criterion is a number this repo measures itself, on nano or mini, before any base1b
decision leans on it.

## Candidates, mapped to concrete tasks

### T11.1 — Compressed-latent attention → cheaper KV per token (Zaya1-8B)
Zaya1's approach: project Q/K/V into a shared low-rank latent, mix sequence with a depthwise conv instead
of full attention over the latent, then decode — vendor reports ~8× KV-cache reduction. For `DottieModel1B`
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
instead of mean-pooling the whole sequence. `T11.2` should read root `multi_jspace_module.py`'s
`MultiJSpace.forward` chunk-recurrent loop (chunked `broadcast_from` prefix-state → `read` fold, default
`chunk_size=128`) before designing the DeltaNet layer — they may share a state-passing interface. (The
root `j_space_module.py` is the older single-workspace fallback and has no chunking; `dottie/j_space_module.py`
does not exist.)
*accept:* a `DeltaNetBlock` swappable for `TransformerBlock1B` at a config-gated subset of layers; the
causality test suite passes; a needle-in-haystack run (`evals/needle.py`, already YaRN-tested to 2048) shows
state size independent of context length at 2×, 4×, 8× the native window; report actual peak VRAM at
base1b's target context vs the current GQA path — this number, not the vendor's, decides T9.3.

**Status (2026-07-11):** implemented in `model_1b.py`, gated behind `deltanet_layers` (unset by default —
zero effect on the existing model, checked by a regression test). 32/32 `tests/test_model.py` pass,
including 4 new: block-level causality, state-size invariance across L=16→128, full-model causality with a
DeltaNet layer mixed into `fusion_layers`, and the default-off regression guard. Analytic KV-cache-vs-fixed-
state comparison using base1b's actual config (not the vendor's numbers), 3-DeltaNet:1-full-attn split
(21/7 of 28 fusion layers): 2.3x smaller at L=2048, rising to 3.95x at L=131072 (7.52GB → 1.90GB). Still
open: live `torch.cuda.max_memory_allocated` and the needle-in-haystack run — both deliberately deferred
rather than take GPU time from the in-progress mini run (T9.2). Not wired into `DottieConfig`/`configs/*.yaml`
yet; that's the adoption step, gated on the live numbers above, not a decision to make speculatively.

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
before touching `dottie/train.py`'s phase manager. Do not start on the current mini run (T9.2, already ~32k
tokens in); target base1b only.

### T11.5 — Per-layer embeddings + discrete diffusion decoding — explicitly out of scope
The review's other two ideas (per-layer token embedding tables for phone-class deploy; DiffusionGemma's
discrete-diffusion decoder) target inference-hardware and decoding-paradigm problems this project doesn't
have: `dottie-agi` has no phone-deploy target (`specs/07_serving_deployment.md` is Docker/GPU-server), and a
bidirectional diffusion decoder is a different training objective from the causal LM this repo just spent
Stage 6 making causal — swapping it in is not a hill-climb step, it's a different project. Not tracked as a
task; recorded here only so a future review doesn't re-propose it without reading this line.

### T11.6 — Markovian recursive trace aggregation → bounded-context multi-sample reasoning (Zaya1, "1b")
Zaya1 also runs k=4 parallel reasoning traces at inference and folds their tail ends into a bounded
256-token aggregation context (entropy-gated, τ=0.7) instead of concatenating all k traces — longer
effective reasoning without paying full self-consistency's token cost. Unlike T11.1-T11.4 this is **not**
a forward-pass/KV-layout change: it's a decode-time strategy over `dottie/serve_engine.py`'s `generate()`,
which today is single-sample greedy/temperature sampling (`ServeEngine.generate`, no multi-trace path
exists). It therefore does not need the T6.1 causality gate — nothing about `DottieModel1B.forward` changes.
It does have a real conceptual cousin already in this codebase: `MultiJSpace`'s chunk-recurrent broadcast
(`multi_jspace_module.py`, `chunk_size` default 128) already folds a stream into a bounded, non-growing
state instead of pooling everything (root `multi_jspace_module.py`, not `dottie/j_space_module.py` — see the
T11.2 correction above) — the same "bounded state over a stream" idea T11.2 points at, except Zaya folds
**parallel samples**, not sequential chunks.
*accept:* implement as an opt-in `generate(..., k_traces=1)` path in `ServeEngine`; k=1 must be
byte-identical to current output (negative control); at k=4, measure wall-clock and token cost against
plain k=1 and against a naive concatenate-all-k baseline before claiming any quality win — nothing here is
validated until `eval_harness.py` shows an accuracy delta on a reasoning probe, not assumed from the
vendor's number.
*priority:* below T11.1-T11.4 — needs a serve path at a scale where 4x sampling cost is affordable
(mini+), and "7400 tokens correct... 91.9% AIME 2025" is Zaya1's own card, not measured here.

### T11.7 — Parametric compression-coverage → the Math branch's fine-tune recipe (VibeThinker-3B)
VibeThinker-3B's result is a training-recipe claim, not an attention/KV mechanism: a small (1.5B-3B)
verifiable-reasoning (math/code) base gets a 2-stage SFT + MaxEnt-guided RL + offline self-distill
post-train and reportedly lands near frontier math benchmarks for a ~$7,800 post-train budget. This maps
to `T9.5` (branch fine-tunes) and specifically the **Math branch** already defined in
`configs/base1b.yaml` (`freeze: [system1, planner]`, fine-tune `[system2, critic, router]`, data
`math_formal/proofs/synthetic_math`, `lr: 8e-5`) — not to `model_1b.py`. The repo's actual post-train file
for this, `sft_sota_2025.py`, is currently a 2-line placeholder (`print(...)`), so there is no existing
recipe to compare against; adopting VibeThinker's 3-stage recipe would mean writing that file for real.
*accept:* blocked on T9.3 (GO/NO-GO), same as T9.5 itself — do not build a training recipe for a branch
that may not exist yet. When T9.5 starts, treat 2-stage SFT + MaxEnt RL + self-distill as one candidate
recipe for the Math branch, measured against plain SFT on the same frozen eval snapshot (T10.6), not
adopted by vendor-card default.

### T11.8 — Inkling blatant architecture wins: relative pos, short conv, MoE sig-router, Muon, effort conditioning, encoder-free multimodal, calibration via proper scoring

Source: Thinking Machines Inkling 975B total / 41B active, Small 276B total / 12B active, 1M ctx, 45T tokens text+images+audio+video, GB300 NVL72 train. Open weights original + NVFP4 for Blackwell.

**Steal 1 — Position: Relative > RoPE for 1M extrapolation**
Inkling: relative positional embedding (Shaw et al 2018, Music Transformer Huang 2018) interleaved sliding-window:global 5:1 ratio, 8 KV heads, performs better + extrapolates better than RoPE.
*Our mapping:* Current YaRN Scaled RoPE 10k->1M in `model_1b.py` works but shows YaRN brittleness at 131k. T11.8.1 candidate `pos_mode="relative"` behind config gate. Causality test same as T6.1. Accept: needle @2x,4x,8x window shows lower drift than RoPE; report actual bpb at 32k,131k,256k.

**Steal 2 — Short convolutions after k/v + residual branches**
Inkling applies short convs at two points — after key and value projections in each attn layer, and on attention and MLP residual branch outputs before they rejoin main residual stream. Stabilizes long ctx + helps local pattern.
*Our mapping:* Add `short_conv_kv=True` optional depthwise conv1d k=3 after k/v proj (group = n_kv_heads). Add `residual_conv=True` conv on branch outputs. Must stay causal (causal conv padding). Accept: loss delta measured, no causal leak (0.0 logit change test).

**Steal 3 — MoE: 256 routed + 2 shared, 6 active, sigmoid router, aux-loss-free bias, joint norm**
Each MoE layer 256 routed experts + 2 shared, 6 routed active per token, sigmoid router with auxiliary-loss-free load-balancing bias (DeepSeek-V3 style), scores of selected routed + shared normalized jointly weighting combined outputs.
*Our mapping:* Current J-Spaces (S1 32 hl=8, S2 64 hl=300, Critic 16 hl=30, Planner 32 hl=150) are dense workspaces with OroJaR orth regularization. Treat each J-Space as routed expert pool with shared experts as System0 commons. Implement `moe_mode="inkling"` with `n_routed=64 scaled down for 1B` (not 256, that is 975B), `n_shared=2`, `k=2-4`, sigmoid router + bias update `b_i += lr * (target - actual load)` no aux loss. Shared expert always on = router + critic fallback. Accept: load balance hist, expert utilization >0.7, no drop in frontier rubric, peak VRAM reported.

**Steal 4 — Hybrid optimizer: Muon for large matrices + Adam for others, wd ∝ lr²**
Inkling: Muon for large matrix weights, Adam for other params, hyperparam schedules inspired by modular manifolds, weight decay coupled to square of learning rate keeping overall weight size stable across horizons (Kosson 2023, Defazio 2025).
*Our mapping:* Our WSD 736k (92% stable) + decay-free WSM merging infinite continuation is already stable. Add `optim="muon_adam"` option: Muon for `qkv,o,gate,up,down` matrices >2D, AdamW for embeddings, norms, router biases, J-Space controllers. Implement `wd = base_wd * (lr / base_lr)^2` coupling. Accept: weight norm curve flat across 736k, no NaN with Peri-LN+QK-Norm, compare grad noise scale.

**Steal 5 — Effort conditioning 0.2-0.99 via system message + per-token cost**
Inkling specifies effort level on different samples by changing system message and adjusting per-token cost, causing model to use different amount of tokens in different rollouts and learn ability to control thinking effort. Sweep 0.2→0.99 traces performance vs mean generated tokens Terminal Bench 2.1/HLE/IFBench, 1/3 tokens to match Nemotron 3 Ultra. Chain-of-thought emergent compression verbose grammatical → telegraphic concise, efficiency alone drove compression (similar to Cognition SWE-1.7).
*Our mapping:* Direct to neuroscience:
- S1 Fast 32 hl=8 = effort 0.2-0.4: Kahneman System1 + basal ganglia habits + Theater of Mind τ=0.7 top-k=8 competition, automatic, low tokens, parallel k=4 traces folded into bounded 256-token aggregation (T11.6)
- S2 Slow 64 hl=300 = effort 0.8-0.99: PFC deliberative + Global Workspace Theory + Dehaene capacity law, hl=300 long horizon, expensive but high score.
- Half-life curves: activation strength = exp(-ln2 * t / hl) => S1 decays fast (hl=8), S2 slow (hl=300), Critic fast (hl=30), Planner medium (hl=150)
- Train with `effort_conditioning=True`: sample effort ~ Uniform(0.2,0.99), prepend system message "effort={effort:.2f}", adjust loss: per-token cost λ = 0.01 * effort, so high effort penalized for verbosity? Actually low effort penalize tokens: cost = (1-effort)*N_tokens. Implement in `dottie/train.py`.
- Inference: `generate(effort=0.3)` for S1, `effort=0.99` for S2, report token vs score curve per eval_frontier_rubric.

**Steal 6 — Encoder-free multimodal: dMel spectrograms + 40x40 patches hMLP**
Audio as dMel spectrograms (dMel: Speech Tokenization made Simple, Richard He Bai 2024), images as patches 40x40 via 4-layer hMLP (Three things everyone should know about Vision Transformers, Touvron 2022), both transformed via lightweight embedding layer processed jointly with text tokens. No encoder. Python tool for zoom/crop seamlessly integrating visual reasoning + code.
*Our mapping:* For Dottie future plus `04_Tennis_DINOv3` and passive lab multimodal generators. Add `multimodal_mode="inkling_encoderfree"`: audio->dMel via torch STFT + mel 128 bins -> linear proj, vision->unfold 40x40 patches -> 4-layer MLP with LayerNorm (hMLP). Joint sequence: [audio_emb][vision_patch_emb][text_emb]. For current 1B keep text-only but leave hook `dottie/audio/` and `dottie/vision/` to use same embedding interface. Benefit: 2MB distilled ConvNeXt-Tiny compatible with ONNX WASM free-tier, no heavy CLIP/Whisper encoder download. Accept: multimodal eval not in scope yet, but code path gated and unit tested with dummy mel + patch.

**Steal 7 — Epistemics: calibration via proper scoring rules + abstention-aware + censorship non-compliance**
Inkling trained for calibration, instruction following, resistance to censorship. Calibration with RL against proper scoring rules on large corpus of resolved real-world questions. Forecasting requires integrating multiple sources into calibrated probability, core skill for trustworthy model — trained ForecastBench Brier Index, Prophet Arena Brier. Instruction following via RL with two automated graders: rubric grader (checklist) + claims grader (verifies each factual claim via agentic web search, not solely own knowledge), improves helpfulness + reduces hallucination not trading one for other. Targeted datasets short-form factual QA with abstention-aware rewards: answering only pays off when likely right, optimal policy answer when confident otherwise "I don't know" or hedged best guess. Some prompts encourage/forbid hedging teaching user preference forced guess vs calibrated non-answer. Finally trained to answer directly on topics subject to censorship (Propaganda and Censorship Eval — Cognition), strong non-compliance.
*Our mapping:* Direct to eval_frontier_rubric.py and Critic J-Space:
- Critic 16 hl=30 = amygdala + insula safety eval, target FORTRESS adversarial 78%, benign 95.9%, StrongREJECT 98.6% as reference, not ceiling
- Planner 32 hl=150 = hippocampus episodic + PFC planning, temporal credit assignment for forecasting
- Implement rubric grader + claims grader dual reward (see CURRICULUM_V2_INKLING.md) and abstention-aware rewards (say IDK gets 0.4 vs hallucinating 0.0)
- Calibration: add Brier score tracking on forecast tasks (ForecastBench style), proper scoring rule RL

**Steal 8 — RL at scale 30M rollouts log-linear + SFT bootstrap from open-weights**
Bootstrap SFT on synthetic data generated by open-weights including Kimi K2.5 small fraction compute, majority large-scale RL async synthetic + human envs, reward on held-out aggregate AIME/HLE/GPQA improves log-linear over 30M rollouts.
*Our mapping:* Our 45T scaled down to 15T curriculum: phases 0-5 but hybrid optimization plan (TBD in CURRICULUM_V2). Keep WSD 736k stable, but post-train with synthetic from open-weights (Qwen3-32B local via Ollama, Kimi style) for bootstrap, then RL with same log-linear tracking.

**Acceptance for T11.8 as a whole:**
- Each sub-steal behind config gate, default-off, causality suite 32/32 pass.
- New docs `docs/INKLING_STEALS.md` mapping to 4 J-Spaces + neuroscience advantage.
- `CURRICULUM_V2_INKLING.md` 6-phase with effort conditioning.
- `eval_frontier_rubric.py` dual grader + abstention + effort sweep implemented and `frontier_eval_results.json` includes effort_curve.
- No work resources, solo personal project footer in all new docs.

## Ordering

T11.2 before T11.1 before T11.3 — T11.2 answers the open VRAM risk directly and has the clearest internal
analog (J-Space chunk-recurrence) to reuse; T11.1 is the next-cheapest KV win; T11.3 only matters once a
long-context target exists. T11.4 is independent and gated on writing `12_matformer_ladder.md` first. T11.5
is a non-task, kept for the record. T11.6 and T11.7 are not forward-pass work and don't compete with
T11.1-T11.4 for the same causality-gated review slot: T11.6 waits on a serve path worth the k=4 cost,
T11.7 waits on T9.3/T9.5 same as the rest of branch fine-tuning. T11.8 runs in parallel with T11.2-T11.7 —
it is curriculum + eval + config-gated arch variants, not blocking base1b trim decision, but provides the
perfect training loop upgrades that make T11.2's fixed-state + T11.1's compressed latent actually stable at
1M (relative pos, Muon, wd∝lr², Peri-LN+QK-Norm anti entropic collapse).


