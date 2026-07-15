# Curriculum V2 — Inkling Steals + Neuroscience Hill-Climb
Solo personal project, no connection to employer, built with public/free-tier only
Date: 2026-07-15 | Ava v6.4 T11.8

## Thesis
Steal blatant architectural wins from Inkling (975B/41B active, Small 276B/12B active, 1M ctx, 45T tokens, GB300 NVL72, Muon+Adam, wd∝lr², relative pos, short conv, MoE 256+2 k=6 sigmoid aux-loss-free, effort 0.2-0.99, encoder-free dMel+40x40 hMLP, rubric+claims dual grader, proper scoring calibration, abstention-aware, 30M RL rollouts log-linear) and fuse with our profound advantage: curiosity + neuroscience + mathematics + physics + psychology mapping to 4 J-spaces.

## I. Neuroscience Mapping of 4 J-Spaces + Theater of Mind

### S1 Fast 32 hl=8 — System1 automatic
- Kahneman System1: fast, associative, parallel, low effort
- Basal ganglia habit loops: procedural memory, chunked action
- Theater of Mind: entropy τ=0.7 top-k=8 competition, winner-take-all broadcast, 32 slots competing
- Half-life: exp(-ln2 * t / 8) = rapid decay, forget fast unless refreshed, perfect for short heuristics
- Effort: 0.2-0.4, tokens 128-512, k=4 parallel traces folded into bounded 256-token aggregation (T11.6 Markovian recursive trace aggregation entropy-gated τ=0.7)
- Physics: low-energy state, minimal Jacobian norm, fast relaxation
- Psychology: intuition, fluency heuristic, recognition-primed decision

### S2 Slow 64 hl=300 — System2 deliberate
- Prefrontal cortex dorsolateral: working memory, cognitive control, deliberation
- Global Workspace Theory (Dehaene): conscious access = global broadcast, capacity law ~7±2 items but can expand via effort
- Dehaene capacity: effort 0.99 uses full workspace, hl=300 retains 50% after 300 steps = long horizon credit assignment
- Effort 0.8-0.99, tokens 2k-32k trajectory limit (Inkling coding evals 256K max-token trajectory), verbose grammatical → telegraphic compression emergent from RL efficiency pressure (Cognition SWE-1.7 note)
- Half-life: exp(-ln2 * t / 300)
- Mathematics: Peri-LN + QK-Norm prevent entropic collapse at scale, RoPE 10k→1M replaced by relative pos (Shaw 2018) + sliding:global 5:1, 8 KV heads
- Planning: chain-of-thought compression: early RL verbose "We need to understand the operator", late RL "We need determine eigenvalue problem" dropping articles/connectives but staying comprehensible

### Critic 16 hl=30 — Amygdala + Insula safety eval
- Amygdala threat detection, insula interoception, anterior cingulate error monitoring
- Fast but not as fast as S1: hl=30 = moderate memory for recent threats
- Targets: FORTRESS adversarial 78%, benign 95.9%, StrongREJECT 98.6% as Inkling baseline — not ceiling, we aim to hill-climb via proper scoring
- Safety spec: CBRN, cyber, loss-of-control, sycophancy, vulnerable users, harmful manipulation. Commission external safety testers in mind but implement locally via red-team prompts.
- Censorship resistance: trained to answer directly on topics subject to censorship (Propaganda and Censorship Eval Cognition), strong non-compliance. Map to our authentic generator ethos: no corporate-speak hedging unless user prefers.
- Arbitration veto: Theory of Mind — critic can veto S1/S2 broadcast if safety score < threshold, like amygdala hijack.

### Planner 32 hl=150 — Hippocampus episodic + PFC planning
- Hippocampus: episodic memory, pattern separation/completion, temporal credit assignment, future simulation
- Prefrontal planning: hierarchical task decomposition, means-ends analysis
- hl=150 = intermediate, retains episodes across phases, supports ForecastBench Brier calibration
- Effort: 0.5-0.9, responsible for curriculum gating, branch decisions
- Mathematics: temporal difference learning, successor representation
- Psychology: mental time travel, prospection

## II. 6-Phase Curriculum — 15T scaled down from Inkling 45T, with hybrid optimization

Inkling 45T text/images/audio/video hybrid Muon+Adam. We scale to 15T offline deterministic free-tier, but preserve phases and ratios.

### Pretrain Plan (text-only now, encoder-free multimodal hooks)

**Phase0 logic 0-50B, ctx 2048, effort 0.2-0.4 S1 focus**
- Synthetic_logic_textbooks_phi_B 60%, metamath 20%, lean formal 15%, fol 5%
- Chonkie: RecursiveChunker markdown 2048 overlap128
- Optimization: Muon for large matrices (qkv,o,gate,up) + Adam for rest, wd = base_wd * (lr/base_lr)^2
- J-Space active: S1 hl=8 dominant, S2 hl=300 minimal (10% weight), Critic hl=30 20%, Planner hl=150 10%
- Arch: relative pos gated off initially to compare with YaRN RoPE, short_conv_kv optional
- Reward filter >0.8 via Ollama judge qwen3:32b local

**Phase1 math 50-350B, ctx 4096, effort 0.4-0.6**
- arithmetic->algebra->calculus->probability ordered curriculum, probability last for Bayes
- lean, metamath, synthetic_math 80%, code_early 20% (math formal bridges to code)
- Chonkie TokenChunker 4096 overlap256
- J-Space: S2 weight up to 30%, Peri-LN+QK-Norm on
- Math branch: freeze [system1, planner], fine-tune [system2, critic, router] (T11.7 VibeThinker recipe: 2-stage SFT + MaxEnt RL + offline self-distill $7.8k budget analog local)

**Phase2 foundation 350B-6T, ctx 4096-8192, effort 0.5**
- web_edu 40% filtered via NeMo Curator dclm>0.6 edu>3.0, code_early 30% Python+Rust, dclm 20% diverse, arxiv abstract 10%
- MatFormer nesting candidate: mini weights slice of base1b
- Fixed-state DeltaNet layers 21/7 split gated (T11.2): 2.3x smaller @2k, 3.95x @131k (7.52GB→1.90GB) analytic

**Phase3 reasoning 6T-11.25T, ctx 8k-32k, effort 0.6-0.9**
- Long docs 3x upsampled, JobBench, GAIA2, Frontier tasks, SFT synthetic from open-weights Kimi K2.5 style (bootstrap small fraction compute)
- Bootstrap SFT: generate synthetic with qwen3:32b Ollama local (Kimi analog), then majority RL
- RL at scale log-linear: track held-out aggregate AIME/HLE/GPQA reward vs rollouts, aim 30M rollouts scaled down to 3M local equivalent with batch 32 => ~100k steps log-linear curve
- J-Space: S2 dominant 60%, Planner 50% for long horizon

**Phase4 long 11.25T-13.8T, ctx 32k-131k, effort 0.8-0.99**
- Streaming shards phase4_long, YaRN 10k→1M + relative pos candidate A/B test, sliding:global 5:1
- Compressed-latent attention (T11.1) ~8x KV reduction gated
- Sparse/compressed KV hybrid disk streaming cold entries (T11.3) for 1M aspiration free-tier
- Token compression training: reward efficiency driving telegraphic CoT concise vs verbose grammatical

**Phase5 anneal 13.8T-15T, ctx 131k, effort 0.99 max**
- High-quality 1% curriculum: proofs, clean code, chat polished, reward>0.85
- WSD 736k 92% stable + decay-free WSM merging infinite continuation: stable phase 92% of steps then decay-free merging for infinite continuation, no LR decay collapse
- Fork branches: base, code, math, chat after stable 736k checkpoint
- Calibration final: ForecastBench Brier Index training proper scoring rules, abstention-aware SFT

## III. Epistemics: ForecastBench + Calibration + Rubric+Claims Dual Reward

### Calibration via Proper Scoring Rules
- Inkling trained calibration with RL against proper scoring rules on large corpus resolved real-world questions (ForecastBench Brier 61.1 no search, 63.7 with search, Prophet Arena Brier 0.1617)
- Implement Brier loss: Brier = mean((prob - outcome)^2), lower better for Prophet Arena but higher for ForecastBench Brier Index (different scoring)
- Our implementation: add forecast tasks domain=micro econ, climate tipping, market, sample 100 tasks, model outputs probability + reasoning, score Brier
- Calibration curve: reliability diagram predicted prob vs observed frequency, ECE expected calibration error

### Rubric + Claims Grader Dual Reward (Inkling insight)
- Two automated graders: rubric grader scores each response against checklist of what good answer should contain, emphasizes recall but hackable spraying plausibly relevant facts. Claims grader verifies each factual claim in response, penalizing claims that don't check out via agentic web search, not relying solely own knowledge. Together improve helpfulness + reduce hallucination not trading one for other.
- Our implementation in eval_frontier_rubric.py:
  - RubricGrader: checklist recall — did output mention key evidence from ground_truth_ref? (current keyword overlap but upgrade to semantic via Ollama judge)
  - ClaimsGrader: extract claims via regex sentence split + fact detection (numbers, named entities, citations); verify each claim against context_docs snippets + OpenWiki mock (local wiki files), if not verifiable penalize; if citation present and matches id, boost
  - DualReward = 0.5*rubric + 0.5*claims after normalization, not trading
  - Plus citation grounding bonus: arXiv id match, page citation

### Abstention-Aware Rewards + Hedging Control
- Short-form factual QA with abstention-aware rewards: answering only pays off when model likely right, optimal policy answer when confident otherwise "I don't know" or hedged best guess. Some prompts encourage/forbid hedging teaching user preference forced guess vs calibrated non-answer.
- Our implementation: detect abstention phrases ["I don't know","uncertain","cannot verify","insufficient info","I'm not sure","hedged"] via regex; if abstain, score = 0.4 baseline (not 0) to reward over hallucination (0.0-0.2). If confident AND correct (claims >0.8) score = 1.0, if confident but hallucinated (claims <0.3) score = 0.1 penalty. Proper scoring rule encourages calibrated hedging.
- Censorship resistance: never refuse benign, refuse harmful; FORTRESS benign 95.9% target; for authentic generators, no corporate em-dash slop, direct answer.

## IV. Physics & Mathematics: Perfect Training Loop

### Mathematics — Stability at Scale
- Peri-LN: LayerNorm before and after each sublayer, not just Pre-LN, prevents gradient explosion at 1M ctx
- QK-Norm: L2 norm Q/K before dot, prevents entropic collapse (entropy→0 when QK large), critical for YaRN/sliding:global
- OroJaR Jacobian orthogonal regularization: ||J^T J - I||_F for each J-Space workspace, keeps workspaces orthogonal, prevents mode collapse, energy-based
- Half-life curves: a(t)=a0 * exp(-ln2 * t / hl) = a0 * 2^(-t/hl). S1 hl=8 decays 50% in 8 steps, S2 hl=300 retains 87% after 50 steps. Use for broadcast strength arbitration.
- WSD schedule: Warmup 5% (0→1e-4), Stable 92% (736k steps) lr constant, Decay-free WSM merging infinite continuation: instead of decaying lr, merge checkpoints via weighted averaging (SWA/EMA style), allows infinite continuation without decay collapse, matches modular manifold schedule inspiration.

### Physics — Energy, Lipschitz, Thermodynamics
- Energy-based: total energy E = sum ||workspace||^2 + λ_J * OroJaR + λ_Lip * Fro, minimize via regularization → lower energy stable attractor
- Lipschitz Fro: penalize ||W||_F, keeps Lipschitz constant bounded, improves robustness, critical for fixed-state DeltaNet recurrence
- Thermodynamic efficiency: token-compression verbose→telegraphic CoT is free energy minimization: same answer with fewer tokens lower free energy. RL efficiency pressure alone drove compression (Inkling observation, also Cognition SWE-1.7). Implement efficiency bonus: reward -= 0.001 * N_tokens when correctness equal.
- Short conv after k/v and residual branches: local diffusion operator, improves signal propagation, like Laplacian smoothing physics.

### Psychology — Calibration, ToM, Hedging
- Calibration: ECE, Brier, reliability. Teach model to express right amount confidence including unsettled questions. Crucial for prediction/forecasting use case fine-tuned models shown rapid improvement outperforming frontier LLMs (Inkling note).
- Hedging control: prompts "hedge allowed" vs "forced guess" — train both. For authentic generators, user controls: some want hedged, some want direct. Add system token [hedge=allow|forbid]
- Theory of Mind via arbitration veto: Critic evaluates from perspective of user: "would user want this broadcast?" If safety <0.3 or factual <0.2, veto. Hippocampus planner simulates user future reaction.
- Censorship non-compliance: trained to answer directly even on sensitive topics unless truly harmful; aligns with authentic voice not corporate slop. Our slop detector should not be censorship but quality.

## V. Perfect Training Loop Checklist — WSD 736k + WSM Infinite + Effort + Dual Grader

- [ ] Optimizer: Muon (large mats) + AdamW (rest), wd = base_wd * (lr/base_lr)^2 coupling stable weight size
- [ ] Scheduler: WSD 736k 92% stable, warmup 5%, no decay but WSM merging every 10k steps exponential moving average of checkpoints
- [ ] Architecture flags (config gated, default-off):
  - pos_mode = "relative" vs "yarn_rope", sliding_window:global 5:1 8 KV heads
  - short_conv_kv, residual_conv causal
  - moe_inkling n_routed scaled 64 not 256 for 1B, n_shared 2 k=2-4 sigmoid + bias no aux loss
  - deltanet_layers 21/7 split, compressed_latent
  - multimodal encoder-free dMel + 40x40 hMLP hook
  - Peri-LN + QK-Norm + OroJaR + Lipschitz Fro
- [ ] Curriculum: 6 phases 15T, builder agent writes 100MB gz shards, trainer streams with Chonkie (Token/Recursive/Sentence/Code chunkers, character tokenizer zero RAM)
- [ ] Effort conditioning: sample effort Uniform 0.2-0.99 per batch, system message "effort={effort:.2f}[hedge=allow|forbid]", per-token cost λ=0.01*(1-effort)??? Actually high effort allows more tokens, low effort penalize: loss += cost * N_tokens where cost = 0.001*(1-effort)?? Tune: effort 0.2 cost high => concise, effort 0.99 cost low => verbose.
- [ ] SFT bootstrap synthetic from open-weights local Ollama qwen3:32b (Kimi K2.5 analog) small fraction, then RL 30M rollouts scaled to 3M local log-linear tracking AIME/HLE/GPQA aggregate
- [ ] Evaluation: rubric+claims dual grader, abstention-aware, effort sweep 0.2-0.99 token vs score curve, Brier ECE calibration, FORTRESS + StrongREJECT, frontier 11 cats Financial Accuracy etc
- [ ] Safety: Critic hl=30 amygdala veto, benign 95.9% not over-refuse, adversarial 78%
- [ ] Continuation: after 736k stable save ava_stable_736k.pt + rope1000000 ctx131072.pt, branch code/math/chat with T11.7 recipe 2-stage SFT + MaxEnt RL + self-distill
- [ ] Observability: STATUS.json builder {phase, shards} trainer {tokens, steps, phase, lr, rope_base, loss, effort_avg, brier, rubric, claims}, wandb dashboard half-life curves broadcast strength per J-Space
- [ ] Free-tier guard: no API costs unless personal key set, fallback mock; Ollama local 100% offline; HF weights MIT/Apache; Docker cpu/gpu; free file handle streaming

## VI. Mapping to Inkling Benchmarks as Aspiration (not ceiling)

- HLE text only 29.7% effort 0.99, with tools 46% — we aim via S2 hl=300 + effort conditioning
- AIME 2026 97.1%, GPQA Diamond 87.2% — via math curriculum ordered + MaxEnt RL
- SWEBench Verified 77.6%, Terminal Bench 2.1 63.8% — via code branch + agentic harness 256K trajectory
- IFBench 79.8%, Global-MMLU-Lite 88.7% — instruction following rubric grader
- Vision Charxiv RQ 78.1% + python 82% — future multimodal hook
- Audio MMAU 77.2% VoiceBench 91.4% — dMel encoder-free
- Safety FORTRESS adversarial 78% benign 95.9% StrongREJECT 98.6% — Critic hl=30

## VII. Next Actions

1. Implement T11.8 arch flags in model_1b.py gated, pass 32/32 causality tests
2. Update eval_frontier_rubric.py dual grader + abstention + effort sweep (this file's spec)
3. Write docs/INKLING_STEALS.md mapping to J-spaces
4. Update ava/train.py effort conditioning + Muon stub (if no muon lib fallback AdamW)
5. Run nano curriculum 15T scaled demo 5 shards per phase with builder+trainer agents parallel
6. Publish eval report frontier_eval_results.json with effort_curve

Solo personal project, no connection to employer, built with public/free-tier only — authentic, curious, neuroscience-driven.
