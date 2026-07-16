# Program Inkling-Small — 12B active / 276B total scaled for RTX 4090 Q4
Solo personal project, no connection to employer, built with public/free-tier only

Source: https://thinkingmachines.ai/news/introducing-inkling/ — Small preview 276B total / 12B active, 1M ctx, 45T tokens, GB300 NVL72. Open weights original + NVFP4 for Blackwell.

## Goal
Hill-climb perfect curriculum + architecture using neuroscience advantage, with Inkling-Small as local judge replacing qwen3:32b Ollama judge, running Q4 on RTX 4090 24GB (fits ~12GB Q4 KV + weights). RTX 4080 16GB runs with offload / layer-drop variant.

## Why Inkling-Small as judge
- qwen3:32b current judge is 20GB Q4, good but not MoE calibrated
- Inkling-Small 12B active (total 276B) Q4 ~ 8-10GB active weights + 276B total stored in NVMe, active 12B per token fits 4090
- Better rubric+claims dual grading, abstention-aware, calibration Brier, effort sweep native support
- Encoder-free multimodal dMel + 40x40 hMLP for audio/vision future

## Setup (Alienware RTX 4080/4090)

1. Install Inkling-Small Q4: `ollama pull inkling-small:q4` or HF download + `llama.cpp --quant q4_k_m`
2. Verify VRAM: 4090 should stay <22GB peak, 4080 <15.5GB with offload to CPU for inactive experts
3. Existing Dottie nano 13.8M / mini 171M unaffected; this program trains Dottie 1B with Inkling-inspired flags + uses Small as judge, not training Small itself (future: distill from Small)
4. Hardware profile: ada-24gb-plus batch64 BF16 SDPA for base1b, ada-16gb batch32 for nano/mini, same as base track

## Architecture flags (all config-gated default-off byte-identical)

From docs/INKLING_STEALS.md T11.8:
- `use_relative=True` — Shaw 2018 relative bias 128 max, 5:1 sliding:global, 8 KV heads, extrapolates >1M better than YaRN 10k->1M
- `use_short_conv=True` — depthwise Conv1d k=3 causal after k/v and on o_proj + mlp residual outputs, identity init
- `use_moe=True` scaled: nano 32 routed +2 shared k=2, mini 64/3, base1b 256/6, sigmoid router aux-loss-free bias, joint norm softmax over routed+shared
- `use_effort=True` — EffortConditioning 0.2-0.99 via system message + per-token cost, S1 Fast hl=8 0.2-0.4, S2 Slow hl=300 0.8-0.99, Critic 16 hl=30, Planner 32 hl=150
- Optimizer: Muon hybrid large mats Newton-Schulz 5 steps + Adam rest, wd ∝ lr² wd_t = base* (lr/lr_max)² Kosson 2023/Defazio 2025
- Encoder-free hooks: dottie/audio/dmel.py STFT 128 mel bins, dottie/embeddings/hmlp.py 40x40 patch 4-layer MLP, joint sequence text+audio+vision, gated off now
- Peri-LN + QK-Norm anti entropic collapse for 1M ctx

## Training loop upgrades (T11.8)

- WSD 2k warmup → 736k stable 92% → WSM decay-free merging infinite continuation (5 EMA 0.9)
- Effort sampling Uniform(0.2,0.99) per batch, system message `effort={effort:.2f}`, loss bonus `loss += 0.001*(1-effort)*N_tokens` encourages verbosity compression at low effort (telegraphic CoT)
- SFT bootstrap from synthetic via Inkling-Small (replacing Kimi K2.5 analog): small fraction compute
- RL at scale 30M rollouts log-linear aggregated AIME/HLE/GPQA, scaled local 3M equivalent batch32

## Curriculum 6-phase 15T scaled from 45T (CURRICULUM_V2_INKLING.md)

- Phase0 logic 0-50B ctx 2048 effort 0.2-0.4 S1 hl=8 dominant synthetic_logic phi-B 60% + metamath 20% + lean 15% + fol 5%
- Phase1 math 50-350B ctx 4096 effort 0.4-0.6 arithmetic→algebra→calculus→probability, 2-stage SFT + MaxEnt RL + offline self-distill
- Phase2 foundation 350B-6T ctx 4096-8192 effort 0.5 web_edu 40% NeMo dclm>0.6 edu>3.0 + code 30% + dclm 20% + arxiv 10%
- Phase3 reasoning 6T-11.25T ctx 8k-32k effort 0.6-0.9 JobBench + GAIA2 + Frontier, bootstrap SFT from Small, RL majority
- Phase4 long 11.25T-13.8T ctx 32k-131k effort 0.8-0.99 streaming shards phase4_long, YaRN vs relative A/B, sliding:global 5:1, compressed-latent ~8x, sparse KV disk
- Phase5 anneal 13.8T-15T ctx 131k effort 0.99 max high-quality 1% + calibration ForecastBench Brier + abstention-aware

## Evaluation升级 — Dual grader + effort sweep replaces single mock judge

From eval_frontier_rubric.py v2:
- RubricGrader recall: checklist what good answer should contain, ground_truth_ref keywords
- ClaimsGrader precision: extract factual claims via regex, verify against context_docs + ~/.openwiki local wiki files, penalize hallucination if no citation or not in docs, boost if arXiv id match
- DualReward = 0.5*rubric +0.5*claims
- Abstention-aware: detect ["I don't know","uncertain","cannot verify"...] -> score 0.4 baseline vs hallucination 0.0-0.2, correct confident 1.0, hallucinated confident 0.1, proper scoring teaches calibration
- Brier proxy: |prob - outcome|² via rubric confidence estimation, ForecastBench style
- Effort sweep 0.2→0.99 tokens vs score curve: mock_model_output_effort telegraphic at 0.2 (60 tokens) vs verbose reasoning at 0.99 (300+ tokens), same correctness, demonstrates Inkling compression win
- Judge selection: --judge inkling uses Inkling-Small if available, fallback qwen3:32b, fallback mock; effort_sweep flag

## Experimentation loop — RTX offload compatible

Use same autonomous loop as program-base.md but with Dottie presets:

```
LOOP:
1. Pick preset nano/mini/base1b + flags combo (relative / short_conv / moe / effort)
2. git commit -m "inkling-steal: {flag}"
3. uv run python dottie/train.py --preset nano --max-steps 1000 --effort-conditioning (smoke)
4. Evaluate: python eval_frontier_rubric.py --judge inkling --effort_sweep --domain finance,bio,code
5. Log: results.tsv val_bpb + frontier_eval_results.json dual=... brier=... tokens curve
6. If val_bpb improved (lower) + dual up + brier down → keep, else revert
7. After kept: append to bb-offload/results/results.jsonl for dashboard + scout rtx releases
```

Timeout: 30 min per smoke (vs 5 min for tiny autoresearch), since Dottie 1B larger.

VRAM targets: nano <8GB, mini <14GB, base1b <22GB on 4090, <15.5GB on 4080 with ada-16gb profile (activation checkpoint True, batch 32).

## BigBang integration

- Queue task: `scout rtx queue add --program inkling-small --preset nano --flags use_relative,use_short_conv`
- Artifact offload: bb-offload/queue.json + bb-offload/results.jsonl + GitHub release scout-rtx
- Dashboard: rtx-offload-dashboard polls api.github.com/repos/jcdavis131/scout-rtx/releases every 60s, imports TSV/JSONL assets dedup commit_sha

## Deliverables checklist

- [x] model_1b.py: RelativePositionBias, MoELayer sigmoid + bias, short_conv causal, EffortConditioning, flags gated default-off
- [x] dottie/muon.py: Muon hybrid Newton-Schulz + Adam + wd coupling + effort classes
- [x] dottie/train.py: effort sampling Uniform 0.2-0.99 + wd coupling lr² + hybrid optimizer selection
- [x] eval_frontier_rubric.py: RubricGrader + ClaimsGrader dual + abstention + Brier + effort_sweep + --judge inkling
- [ ] encoders: dottie/audio/dmel.py + dottie/embeddings/hmlp.py 40x40 stub (next)
- [ ] This doc + CURRICULUM_V2_INKLING.md + INKLING_STEALS.md + spec 11_arch_hillclimb T11.8
- [ ] GitHub release v0.6.0-dottie-0716 with results.tsv + frontier_eval_results.json including effort curve

## Solo disclaimer
Solo personal project, no connection to employer, built with public/free-tier only

