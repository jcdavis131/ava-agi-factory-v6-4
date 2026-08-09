# Inkling Steals — Architecture Wins for Dottie v6.4
Solo personal project, no connection to employer, built with public/free-tier only

Source: https://thinkingmachines.ai/news/introducing-inkling/ — 975B total / 41B active MoE, 1M ctx, 45T tokens multimodal, GB300 NVL72 training.

## Blatant Wins to Steal

### 1. MoE Architecture — Sigmoid Router + Aux-Loss-Free Bias
- **Inkling:** 256 routed experts + 2 shared, 6 active/token, sigmoid router (not softmax), auxiliary-loss-free load balancing via learnable bias, joint normalization of selected routed + shared outputs.
- **Our steal:** MoELayer scaled: nano 32 routed + 2 shared k=2, mini 64/3, base1b 256/6. Bias update: bias += lr * (expert_frac - uniform), no gradient aux loss. Joint norm: softmax over combined scores or L2. Experts = SwiGLU(d_model, hidden).
- **Neuroscience:** Mirrors basal ganglia action selection — sigmoid independent evidence accumulation, not winner-take-all softmax. Shared experts = cortical common pathways, routed = specialized columns.
- **Implementation:** `model_1b.py:MoELayer(use_moe=False)` config-gated, default off → byte-identical existing model.

### 2. Attention — Sliding:Global 5:1 + 8 KV Heads + Relative Position + Short Convs
- **Inkling:** 5:1 sliding-window:global interleaving, 8 KV heads GQA, relative positional embedding (Shaw et al 2018, Music Transformer Huang et al 2018) better extrapolation than RoPE, short convolutions after k/v projections and on attn/MLP residual branch outputs before rejoin.
- **Our steal:**
  - Already have GQA + Gated DeltaNet fixed-state (T11.2 answers VRAM 2.3x at 2k, 3.95x at 131k)
  - Add `rope_type=relative`: learnable clipped bias table per head, extrapolation >1M without YaRN NTK hack
  - Add `use_short_conv`: depthwise Conv1d k=3 causal pad left 2 after k_proj, v_proj, o_proj, mlp.down — before peri_norm
  - Keep `yarn` and `longrope2` (31->25 critical dim shift) existing; relative as third option
- **Physics:** Short conv = inductive bias for locality, like CNN receptive fields in V1, reduces entropy collapse, cheap O(L) vs full attn
- **Math:** Relative pos is translation invariant, better for long extrapolation (no periodic RoPE freq aliasing)

### 3. Optimization — Muon + Weight Decay ∝ lr² + WSD/WSM
- **Inkling:** Muon for large matrix weights, Adam for others, wd coupled to square of learning rate (Kosson 2023, Defazio 2025) keeps weight size stable across horizons, hybrid schedules inspired by modular manifolds
- **Our steal:**
  - Implement `dottie/muon.py`: Newton-Schulz orthogonalization (5 steps) for momentum, for Linear layers >512 dims; AdamW for biases/norms/routing
  - wd_schedule: wd_t = base_wd * (lr_t / lr_max)^2, target 0.1 * scale
  - Already have WSD 2000→736k 92% + WSM decay-free merging buffer 5 EMA 0.9 infinite continuation
- **Math:** Muon is steepest descent under spectral norm, better conditioned than Adam for large mats

### 4. Controllable Thinking Effort — 0.2→0.99 Token Efficiency
- **Inkling:** Effort conditioned via system message + per-token cost, sweep 0.2 to 0.99 traces performance vs mean tokens on Terminal Bench 2.1/HLE/IFBench — 1/3 tokens vs Nemotron at same score, chain-of-thought compresses verbose grammatical → telegraphic without reward
- **Our steal:** EffortConditioning module:
  - Input: effort embedding (learned scalar 0.2-0.99) added to token emb
  - Loss multiplier: effort_cost = effort * num_tokens, encourages compression at low effort
  - Maps to J-Spaces: S1 automatic hl=8 effort 0.2 (fast, 8 tokens), S2 deliberate hl=300 effort 0.99 (slow reasoning 150+ tokens), Critic effort 0.5, Planner 0.7
  - GWTB v2 top-k=8 τ=0.7 theater of mind entropy drive already implements this
- **Psychology:** Kahneman System1 vs System2, cognitive miserliness, expertise automatization (verbose→telegraphic is chunking)

### 5. Encoder-Free Multimodality — dMel + 40x40 hMLP
- **Inkling:** Audio as dMel spectrograms (He Bai et al 2024) encoder-free, images as 40x40 patches via 4-layer hMLP (Touvron 2022), light embedding layer, joint processing with text tokens
- **Our steal:** Upgrade VisionEncoder/AudioEncoder:
  - Current: Linear(1024→d_model) + RMSNorm trivial
  - New: 4-layer hMLP: Linear → GELU → Linear → GELU → Linear → GELU → Linear, patch size 40x40, dMel 128 bins → d_model, process jointly
  - Fits tennis DINOv3 project: 2MB distilled ONNX WASM target
- **Neuroscience:** Early sensory no RoPE, like V1 retinotopic patches, auditory tonotopic dMel, downstream fusion is association cortex

### 6. Epistemics — Calibration + Rubric+Claims Dual Grader + Abstention
- **Inkling:** Calibration RL vs proper scoring rules (Brier Index ForecastBench), rubric grader (checklist recall) + claims grader (agentic web search verify each claim) penalizes hallucination, abstention-aware rewards answer only if likely right else "I don't know" or hedged
- **Our steal:** New Frontier eval:
  - RubricGrader: what good answer should contain (recall)
  - ClaimsGrader: verifies each factual claim via web search (or local wiki ~/.openwiki), penalizes false claims
  - Together reduce hallucination without trading helpfulness
  - Abstention loss: MSE(vm? actually p_correct), reward 1 if correct else -1, "I don't know" gets 0 if uncertain <0.5 → teaches calibration
  - FORTRESS adversarial 78% / benign 95.9% / StrongREJECT 98.6% targets
- **Psychology:** Metacognition, Dunning-Kruger calibration, hedging controlled via prompt

### 7. RL at Scale — 30M Rollouts Log-Linear
- **Inkling:** 30M+ rollouts, stable training, log-linear reasoning gain over entire run, emergent concise CoT
- **Our steal:** Autoresearch loop hourly already picks 2408.06081v2 LongRoPE2 etc, smoke 617bcbf 0.9979 keep. Extend to multi-task RL: math/code/chat with WSM infinite continuation, not just supervised

## Mapping to Our 4 J-Spaces (Profound Advantage)

- **S1 Fast 32 hl=8 associative:** Theater of Mind entropy τ=0.7, top-k competition k=8, effort 0.2, d_gw=256 bottleneck, Theater of Mind, basal ganglia habitization, capacity law exp(-0.12*max(0,k-6)) knee 6
- **S2 Slow 64 hl=300 verifiable:** GWT global workspace, Dehaene capacity combined 0.6*S2+0.4*S1, routing target deliberate [0.15,0.55,0.1,0.2], hl=300 long retention, effort 0.99, DeltaNet fixed-state for reasoning chain
- **Critic 16 hl=30 safety/eval-aware:** Amygdala/insula, safety_concepts 1.0 if eval_aware else 0.3, vm target 0.08, hl=30 fast forgetting of threat, routing safety [0.1,0.2,0.6,0.1], FORTRESS/StrongREJECT
- **Planner 32 hl=150 deadlines/env_deltas:** Hippocampus episodic + PFC, temporal hold MSE(broadcast,0.20), GAIA2 dynamic async, delegation_priority, env_delta across 64k-128k, routing temporal [0.1,0.3,0.1,0.5]

## Implementation Checklist (Home-only, free-tier, offline deterministic)

- [ ] model_1b.py: RelativePositionBias class, short_conv Conv1d depthwise causal, MoELayer sigmoid router + bias, config gated use_relative/use_short_conv/use_moe
- [ ] dottie/muon.py: Muon optimizer with Newton-Schulz 5 steps
- [ ] train_1b_deepspeed.py: wd ∝ lr², hybrid Muon+Adam, effort conditioning
- [ ] eval_frontier_rubric.py: RubricGrader + ClaimsGrader + Abstention + effort sweep 0.2-0.99 tokens vs score curve
- [ ] dottie/audio/: dMel spectrogram encoder-free (librosa-free stub with mel filterbank)
- [ ] dottie/embeddings/: 40x40 hMLP vision patch encoder
- [ ] Causality tests 28 must stay green (T6.1 hard invariant)
- [ ] Byte-identical default path: all new behind flags False

## Footer
Solo personal project, no connection to employer, built with public/free-tier only
