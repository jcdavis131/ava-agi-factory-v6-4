# Ava v6.4 — Antidoom + Compression Curriculum Assessment

> **Solo personal project, no connection to employer, built with public/free-tier only**  
> **HOME directory only** — `~/workspace/ava-agi-factory-v6-4` · Free-tier: HF datasets streaming, R2/Workers/Supabase if needed, ONNX WASM, public pip only  
> Date: 2026-07-15 · Assessment for: https://github.com/Liquid4All/antidoom + compression algorithms

## 0. Executive Summary

Ava is healthy in **continuous data-gather mode**. Builder is shipping 10M-token shards on the new fast path, GDrive sync is live, weekly trainer correctly BLOCKED_NO_GPU in Hatch VM (expected — real training runs on Alienware / local). Frontier eval mock scores 0.49-0.69 point to weak domains macro/materials/climate/bio/finance — consistent with Phase0 still being only Logic.

**antidoom** is a narrow, high-ROI post-training fix for "doom loops" — repetitive `Wait / So / But / Alternatively` spirals that emerge when heavy synthetic reasoning overtrains a few discourse tokens + low-temp sampling + self-reinforcing context. Ava's Phase 3 Reasoning (6T-11.25T, 55% math_reasoning + synthetic textbooks) is exactly the recipe that creates this.

**Compression** should be injected as 3 parallel tracks:
- **(a) Knowledge curriculum**: teach Ava Shannon entropy, Huffman, LZ77/78/LZW, Arithmetic Coding, BWT, ANS — verified by recomputing — improves LM-is-compression reasoning.
- **(b) Model serving compression**: quantization (GPTQ/AWQ/SmoothQuant/AutoRound/SpinQuant/QuIP), KV-cache quantization, Zarr chunked storage — required to fit base1b 1409M params (8.4GB bf16+grads+AdamW8bit before activations) into Alienware 4080 12GB at 128k context.
- **(c) Semantic / neural compression**: Z-token compressor/decompressor, Training LLMs over Neurally Compressed Text, DeepMind LMC work — direct mapping to Ava's J-Space: 144 workspace slots (32+64+16+32) compressing 2k-131k context at 20% broadcast target.

All three are HOME-safe, MIT/Apache public repos, no work IP.

---

## 1. Current Ava Health Check (July 15 2026)

### 1.1 Builder / Lake

From `STATUS.json` (live):

- **current_phase**: `phase0_logic` (Phase 0 0-50B, Phi Method B synthetic logic 60% + Metamath 20% + Lean 15% + FOL 5%)
- **writes_this_phase**: 1,926,589 docs · **total_shards**: 110 · **lake_gb**: tiny (0.000018 — local view, GDrive is source of truth)
- **Last expansion** `2026-07-15T10:04:18Z`: 10,000,063 tokens, 32,084 docs, 1 shard `packed_20260715_100412_00060_6385.jsonl.gz`, qual_filtered 1,379 (alpha>0.6 reward>0.8)
- **Detailed** `07:58Z`: 500,030 tokens, 5,058 docs, dup_filtered 9,720 (simhash th=3 + md5), qual_filtered 8,528 — this is the new 500K HatchVM path with daily_expanded 59 files, global_manifest_lines 930k
- **Previous** `05:59Z`: 10,000,285 tokens, 32,097 docs, 2 shards
- **GDrive upload**: `Ava-Datasets-Expansion` folder `19tqzjB-ofqKmx1w6S4qLNB_jAEa6s3ve` — uploaded 2 new shards with content-addressable sha12 dedup, personal-safe guard passed
- **Efficiency**: 500K run in 27s, disk 88% (100G total, 12G used) — healthy

**Interpretation**: The 4h cron `ava-data-gather-4h` and `ava-data-gather-daily` are both healthy. Migration from 10M shards → 500K shards + simhash dedup improved quality. `phase_progress: 1.0` is misleading — that's per-builder-run, not curriculum-wide. Real need: Pacer (T10.1) to hold lead buffer.

### 1.2 Trainer / WSD / YaRN

- **weekly_training.status**: `BLOCKED_NO_GPU` — expected. Hatch VM has no `nvidia-smi` / docker / ollama. `next_action_host`: `./scripts/local_train.sh torchrun --nproc_per_node=1 train_1b_deepspeed.py --preset mini --deepspeed deepspeed_zero3_bf16.json --tokens_total 2500000000 --resume`
- **WSD target**: warmup 2k, stable 736k (92% to 13.8T), decay to 2e-5 — matches `specs/05_training.md`
- **YaRN schedule**: 10k (2k/4k ctx) → 50k (8k) → 100k (16k) → 500k (32k) → 1M (64k/128k) with YaRN 2.0-4.0, attn_factor 0.1*ln(scale)+1
- **Real training intended**: Alienware host per `specs/08_alienware_runbook.md`

### 1.3 Eval

- `frontier_eval_results.json` mock (ollama judge): macro 0.499, materials 0.502, climate 0.549, bio 0.621, finance 0.625, code 0.634, law 0.692 — all mock, need real eval after Phase2 foundation grows. Weak domains identified by `data/discovery/` daily cron already driving dataset discovery.
- `branch_eval_results_real.json`: 5 canonical J-Space tests (Spider→Ant, France→China, Soccer→Rugby, Spanish→French, Safety blackmail) — currently FAIL at nano scale (verbalizable_mass 0.001 vs target 0.065, auto_cos vs deliberate_cos delta 1e-8) — expected at 0 tokens trained; will improve after S2 hl 300-400 learning.

### 1.4 Cron Health

All healthy:
- `ava-dataset-discovery-daily` 10:00 UTC — generates weak_domain needs
- `ava-eval-distill-daily` 09:00 — runs mock frontier eval
- `ava-data-gather-4h` / `ava-data-gather-daily` interval 4h — live
- `ava-training-monitor` interval 30m
- `ava-training-weekly` Sun 03:00

### 1.5 Open TODOs Relevant

- T10.1 Pacer setpoint controller, T10.2 Infinite-generator governor, T10.3 Bounded-memory streaming, T10.5 as-of watermark (reproducibility), T10.6 frozen eval snapshots
- T11.1 Compressed-latent attention (Zaya1-style), T11.2 Gated DeltaNet (done, 32/32 tests), T11.3 Sparse/compressed KV + disk streaming
- Stage 12 Workflow generators done (jobbench/gaia2) — 203 passed

Gap: **no compression textbook + no anti-loop defense**. Need to add.

---

## 2. Deep Dive: antidoom https://github.com/Liquid4All/antidoom

### 2.1 What it does

- Generates and trains **targeted preference data for reducing repetition loops (doom loops)**
- Not full SFT/DPO — **Final Token Preference Optimization (FTPO)**: single-token preference at loop-start.

Flow:
1. Sample completions from base checkpoint at low temp (e.g., 0.01)
2. Scan for inner repetition: detect where a span repeats (naive string → token boundary refinement)
3. Mark **first loop-starting token as rejected**
4. Sample filtered alternative next tokens at same position as **chosen** (plausible, coherent)
5. Row = {context_prefix ending before rejected_token, rejected_decoded, multi_chosen_decoded, metadata}
6. Regularize rejected/chosen distribution: `rejected_regularisation_strength 0.3`, `chosen_regularisation_strength 0.5` — shave overrepresented Wait/So/But
7. Train LoRA adapter (rank 128-256 recommended) with FTPO loss + MSE tether to reference model (lambda_mse, tau_mse_target)

Dataset: `LiquidAI/antidoom-mix-v1.0` — prompt-only ShareGPT mixture, **no gold answers**.

Hyperparams starting point from README:
- `target_pairs` 15k-20k, `max_train_examples` ≤70% of generated (e.g., 12k from 15-20k)
- `learning_rate` 1e-5 to 2e-5 for ~12k samples, early_stopping `chosen_win` 0.4 (stop when chosen beating rejected 40%, >0.5 overtrains)

### 2.2 Why doom loops happen (their doc)

- **Overtrained tokens**: `Wait`, `So`, `But`, `Alternatively` become over-attractive after heavy synthetic reasoning
- **Self-reinforcing context**: once short sequence appears, prior context makes it more likely → probability →1
- **Low-temp sampling**: temp 0 keeps selecting highest-prob continuation, no escape

This is exactly Ava Phase 3: synthetic reasoning 55%, logic textbooks 60% Phase0 — high risk.

### 2.3 Why relevant to Ava

- Blueprint already expects reasoning chain length 7,400 tokens (per LOCAL_LLMS_2026_SOTA.md Zaya1 note) beating GPT-5.5 — long chains → loops
- J-Space routing: deliberate S2 (hl 300) handles System2 — if it loops, Critic (hl30) should catch, but Critic currently has verbalizable_mass only 0.08 — needs anti-loop training signal
- Current eval: no doom loop metric — we need one
- Persona risk: Phase5 anneal + chat branch (safety_blackmail_leverage 20%) — loops in safety scenarios look like stalling, flagged by eval

### 2.4 Where in curriculum

| Phase | Injection | How |
|-------|-----------|-----|
| Phase 3 Reasoning | Data generation | Add synthetic generator that deliberately creates then breaks loops -> negative examples; also filter training docs with loop detector |
| Phase 4 Long | Serving | KV-cache growth magnifies loops; need detection in `ava/serve_engine.py` generate loop (abort + resample at higher temp, like antidoom temp sweep) |
| Phase 5 Anneal | Post-training | **Primary**: Run antidoom pipeline on `ava_stable_736k.pt` to produce LoRA adapter `ava_branch_chat_antidoom_lora`, merge |
| Branches | Branch eval | Math branch especially — synthetic math R1 15% + lean_mathlib 20% → overtrained "Therefore" |
| Chat branch | Final polish | After chat fine-tune, second antidoom pass on chat distribution (prompts from frontier weak domains) |

Best order: Add eval metric first → then collect data at Phase5 → train LoRA → merge.

### 2.5 Technical fit HOME-only

- antidoom requires PyTorch + vLLM + GPU. Hatch VM no GPU, but Alienware does.
- We can implement **lightweight detector without vLLM** for dataset cleaning: same logic as `detect_loop` using token repeat detection (suffix array / Lempel-Ziv style) — free-tier, public pip only (no vllm needed for cleaning)
- Full FTPO training: reuse `peft` LoRA + custom loss (already in `OPEN_SOURCE_TOOLCHAIN.md` PEFT section). Can port their FTPO loss: logit difference loss + MSE tether.

File changes proposed:
- `ava/datagen/compression.py` contains loop-detector helper as subset (repetition detection shared with compression)
- `scripts/antidoom_integration.py` wrapper: `generate -> detect -> build FTPO jsonl -> train LoRA`
- `evals/doom_loop.py` metric: % generations containing loop (>2 repeats of span length >=10 tokens within 200-token window at temp 0)
- `ava/serve_engine.py`: add `DoomLoopBreaker` — if 3 consecutive tokens probability >0.95 and n-gram repeats, resample with temp 0.7 once, log to `serve_audit.jsonl`

License: MIT, public — OK per AGENTS.md solo disclaimer.

---

## 3. Deep Dive: Compression Algorithms

### 3a. Knowledge Curriculum (Teach Ava to compress)

Why: DeepMind paper "Language Modeling Is Compression" — Chinchilla 70B compresses ImageNet patches 43.4% (vs PNG 58.5%) and LibriSpeech 16.4% (vs FLAC 30.3%) with NO vision/audio training. Training loss = negative log likelihood = bits. **Better compressor = better world model.**

What to teach (verifiable by recomputing in generator):

1. **Shannon Entropy / Information Theory** — Phase0 Logic
   - Definition H = -sum p log2 p
   - Examples: compute entropy of {"a":0.5, "b":0.25, "c":0.25} → 1.5 bits
   - Kraft inequality, prefix codes
   - Generator: random distribution, compute entropy with python `math`, produce textbook paragraph + question + answer (answer recomputed, not templated)

2. **Huffman Coding** — Phase1 Math
   - Build tree, assign codes, compute avg length vs entropy
   - Example docs: "Given frequencies..., build Huffman tree, encode 'abac'..." with step-by-step merging
   - Verification: independent Huffman implementation in generator vs student answer

3. **LZ77 / LZ78 / LZW** — Phase2 Foundation (code early)
   - Sliding window matching, dictionary building
   - Example: compress "ababa..." with tuples (offset,length,next)
   - Code: small Python functions for compress/decompress, doctests executed via safe exec (same as existing `code_gen.py`)

4. **Arithmetic Coding** — Phase1/2
   - Interval narrowing with probabilities, shows optimal vs Huffman
   - Links to LLM as arithmetic coder: LLM predicts prob → arithmetic code

5. **BWT / MTF / ANS** — Phase3 Reasoning
   - Burrows-Wheeler Transform + Move-to-Front, Asymmetric Numeral Systems (used in Zstd)
   - Long reasoning chains computing BWT matrix rotation

Integration with existing:
- `ava/datagen/logic.py` already has truth-table walkthroughs → add entropy walkthroughs same template 25%
- `ava/datagen/math_gen.py` has staged arithmetic→probability → insert compression math after probability (fits order)
- `ava/datagen/code_gen.py` already execs code → add compression code family

Benefits: improves reasoning, gives model ability to reason about its own tokenization compression ratio (currently 3.28 chars/token measured), directly evaluates via compression_chars_per_token metric.

### 3b. Model Serving Compression (Fit base1b on 4080)

Problem: base1b 1409M params — calculation from TODOS.md Stage 11:
- bf16 weights: 1409M *2 = 2.8GB
- grads: 2.8GB
- AdamW8bit optimizer: ~1.4GB
- Total 8.4GB before activations
- Activations at L=2048: ~? 28 fusion layers * seq * d_model * layers ≈ 2-3GB → 11.6GB budget tight
- At L=131072: KV-cache dominates: 7.52GB → 1.90GB with DeltaNet 3:1 split

Solutions in open source toolchain (all already listed in OPEN_SOURCE_TOOLCHAIN.md but need wiring):

- **LLM Compressor (vLLM)** https://github.com/vllm-project/llm-compressor — GPTQ, AWQ, SmoothQuant, AutoRound, FP8/INT8 KV cache quantization
  - W8A8 int8: weight+activation, W4A16 for memory
  - For Ava: use W4A16 GPTQ for 4x memory (2.8GB→0.7GB weights) on server side after training
- **LightCompress** — EMNLP 2024 & AAAI 2026 toolkit, supports Llama, Mistral, Qwen, quantization + sparsity + mixed-precision
  - Mixed-precision: keep Critic (safety) at bf16, compress S1 fast.
- **AWQ / SpinQuant / QuIP** — rotation-based quantization preserves outliers (important for J-Space broadcast vectors)
- **KV-cache quantization**: FP8 KV, NVFP4 for long context Phase4 — reduces 7.5GB → ~1GB
- **DeltaNet fixed-state** already done T11.2: 21 layers DeltaNet (fixed 1.05MB each) + 7 full attn = 22MB vs 7.5GB growing cache → 3.95x saving at 131k

Where in curriculum:

- Phase4_long: enable `--attn sparse_compressed --kv-quant fp8` in training (not just serving) — train with quantized KV to be robust
- Phase5_anneal: QAT (quantization-aware) annealing: last 10% steps with simulated quantization
- Branch serving: code/math/chat branches as 4-bit merged LoRAs

File changes:

- `configs/base1b.yaml` add `compression: {kv_quant: fp8, weight_quant: w4a16, use_deltanet: [0,1,2,...21 indices]}`
- `ava/attention/compressed_conv.py` exists (Zaya1 8x) — wire via `AvaConfig(attn_mode=compressed_conv)`
- `specs/04_model_and_configs.md` update param math with compression numbers
- `specs/11_arch_hillclimb.md` T11.3 promotion to T11.1 priority

### 3c. Semantic / Neural Compression (Maps to J-Space)

This is the deepest fit.

Ava J-Space is already a **semantic compressor**:
- Input: 2048-131k tokens
- Bottleneck: 144 slots total (S1 32 hl8 automatic, S2 64 hl300 deliberate, Critic 16 hl30, Planner 32 hl150) — each slot is latent vector
- Broadcast: 20% norm → deliberate compression to 20% of fused norm
- Router: per-task_type targets [auto 0.6,0.15,0.1,0.15] etc — learned routing is compression
- Losses: reportability, broadcast, selectivity, modulation, inter-space MI cos 0.45 — all enforce that compressed workspace is verbalizable and useful

Papers relevant:

- **Large Language Model as Token Compressor and Decompressor (Z-token)** — train 3 LoRA adapters on same backbone: compressor (NL→Z), decompressor (Z→NL), inferencer (Z→Z). Length regularizer `(K/|X| - 1/r)^2` controls compression ratio r. Codebook usage + commitment regularizers. Directly maps to Ava: S2 is compressor, broadcast is Z, Planner is inferencer.

- **Training LLMs over Neurally Compressed Text** — M1 compresses raw bytes via arithmetic coding, M2 trains over compressed bitstream chunked into tokens. Finding: AC-compressed text not readily learnable even with unigram M1 — suggests need for learnable Z-tokens not pure arithmetic coding. Implication for Ava: don't use pure gzip for training, use learned slots.

- **Language Modeling is Compression (DeepMind)** — Chinchilla as compressor → evaluation metric: compress held-out sets (enwik9, ImageNet patches, LibriSpeech) and measure compression rate. Can add as eval harness: measure Ava's cross-entropy on enwik9 as compression rate (bits per byte).

Integration:

| Paper idea | Ava mapping | Implementation |
|---|---|---|
| Z-token compressor adapter Δφ, decompressor Δθ, inferencer Δψ | S1=fast compressor hl8, S2=slow compressor hl300, Planner=inferencer hl150, Critic=safety verifier of decompressed | Create `ava/compression/z_token.py` — LoRA adapters sharing backbone, train on reconstruction + continuation tasks |
| Length regularizer controlling K/|X| | Broadcast strength loss already does 20% — can add explicit length penalty (slots used / seq_len - target)^2 | Extend `multi_jspace_module.py` with `compression_ratio_loss` |
| Contextual regularity of Z-tokens, repeated Z in semantically related contexts | Concept vector in `concept_token()` — France/Peking etc — measure if same slot fires for semantically related paraphrases | Add metric in `evals/probes.py`: Z-slot cosine similarity across paraphrases |
| Training over neurally compressed text | Train mini over Chonkie-compressed shards but also over Z-bottlenecked representation (autoencoding) | In `streaming_data.py` Phase Chonkie config, add `neural_compress: True` option that feeds through current Ava checkpoint to compress then trains decompressor |

Curriculum injection:

- Phase 0-1: Textbook about LM is compression (Shannon)
- Phase 3: Add synthetic dataset where task is compress paragraph to 20-token summary then reconstruct — exactly Z-token training objective, verifies reportability loss
- Phase 4 Long: Long docs 50% (>16k) — add task_type `compression` where input 32k, target is 144-slot summary (literal J-Space slots) + reconstruction metric — tests needle retrieval but compressed
- Eval: Add `compression` eval set: reconstruct Wikipedia article from its Z — measured BLEU / reconstruction CE

---

## 4. Concrete Integration Plan

### 4.1 File Map (HOME-only, public pip)

**New generators:**

- `ava/datagen/compression.py` — `CompressionGenerator` (phases 0,1,2,3,5)
  - Families: shannon 25% (entropy calc, Kraft), huffman 20% (tree build), lz77 20% (sliding window tuples), arithmetic 15% (interval narrowing), bwt_ans 10% (BWT matrix), z_token 10% (compress-to-20% task)
  - Each doc: `text` is textbook walkthrough + problem + solution, with solution **computed by Python** (Huffman via heapq, LZ via naive search, entropy via -sum p log2 p), byte-deterministic via `random.Random(seed)`, not global RNG
  - Schema matches `ava/datagen/base.py`: `text, task_type, concept, phase, source`
  - `task_type`: shannon/huffman/entropy → deliberate, lz/bwt code → deliberate, z_token → automatic + deliberate mix (routing test)
  - Phases: p0 small (entropy), p1 med (Huffman), p2 15MB (LZ code), p3 12MB (BWT + Z compression reasoning 6000+ chars → long), p5 (anneal high-quality verified proofs of Kraft inequality)
  - Acceptance: `python -m ava.datagen.compression --seed 1234 --out /tmp/comp --mb 5` → byte-deterministic, `tests/test_datagen.py -k compression` green

**Antidoom integration:**

- `ava/evals/doom_loop.py` — detector: scan completion for repeated span detection (min span 10 tokens, appears 3x within 200-token window, or probability >0.9 loop) + metrics
- `scripts/antidoom_integration.py` — lightweight port:
  - `python scripts/antidoom_integration.py generate --ckpt ava_stable_736k.pt --prompts data/discovery/prompts.jsonl --temp 0.01 --out runs/antidoom/ftpo.jsonl` → samples 15k prompts, detects loops using detector, builds FTPO rows
  - `train` subcommand: PEFT LoRA r=128 alpha=128 target_modules [q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head] lr 1e-5, MSE tether lambda_mse 10, tau 0.1, early stop chosen_win 0.4
  - No vLLM dep for generate (uses `ava/serve_engine.py` generate), optional vLLM flag

**Model compression wiring:**

- `ava/attention/compressed_conv.py` — already exists (Zaya1 8x) — add test + wire via `configs/nano.yaml` `model.attn_mode: compressed_conv`
- `ava/attention/sparse_compressed.py` — new (DeepSeek V4 Flash style 10% KV) — hybrid full:compressed 1:3
- `ava/compression/quantization.py` — wrapper for llm-compressor: load HF converted model, apply GPTQ W4A16, save

**Spec updates:**

- `specs/02_data_generation.md` — add B6 section for CompressionGenerator, mirror B1-B5 style, include MB targets, schema, verification method (recompute Huffman tree, recompute LZ tuples)
- `specs/04_model_and_configs.md` — add compression block with VRAM math updated with W4 + KV FP8 numbers
- `specs/05_training.md` — note doom loop avoidance via Chonkie overlap preserving reasoning chains + antidoom as Phase5 anneal
- `specs/06_evaluation.md` — add doom_loop + compression reconstruction eval
- `specs/10_continuous_supply.md` — pacer consider compression textbook lead buffer (edu>=4.5)
- `CURRICULUM_LOOP_PLAN.md` — Table:

```
Phase0: + entropy (Shannon) via compression shannon family 60% logic + 10% shannon
Phase1: + prefix codes (Huffman, Kraft) via huffman family
Phase2: + LZ code via code_gen extension + compression lz family
Phase3: + BWT/ANS + Z-token compress/reconstruct tasks (long)
Phase4: + KV cache compressed training, semantic compression metrics, Z-slot paraphrase cos
Phase5: + Anneal with verified compression proofs + antidoom LoRA FT
```

**Toolchain:**

- `OPEN_SOURCE_TOOLCHAIN.md` — add:
  - **Antidoom** https://github.com/Liquid4All/antidoom — FTPO LoRA for doom loops, MIT
  - **LLM Compressor (vLLM)** — already there but promote + add usage example for Ava
  - **LightCompress** https://github.com/ModelTC/LightCompress — EMNLP 2024 & AAAI 2026 comprehensive quantization/sparsity
  - **Z-Token** arxiv — LoRA compressor/decompressor/inferencer paradigm

**Tasks doc:**

- `tasks/plan-antidoom-compression.md` — detailed breakdown with acceptance commands, tiers, deps, risk table

### 4.2 Acceptance Commands

```bash
# Compression generator
python -m ava.datagen.compression --seed 1234 --out /tmp/comp --mb 1
python -m ava.datagen.compression --seed 1234 --out /tmp/comp2 --mb 1
diff <(sha256sum /tmp/comp/*.jsonl) <(sha256sum /tmp/comp2/*.jsonl) # empty

# Datagen suite
pytest tests/test_datagen.py -k compression -q  # green

# Doom loop eval
python -m ava.evals.doom_loop --ckpt checkpoints/nano/latest.pt --prompts evals/probe_items/doom_prompts.jsonl --temp 0

# Full datagen small
python scripts/gen_all_data.py --seed 1234 --tiny --include compression

# Antidoom dry-run (no GPU)
python scripts/antidoom_integration.py generate --dry-run --prompts 100

# Model compression dry-run
AVA_SMOKE_DRY_RUN=1 python scripts/bench_pipeline.py # ensure tok/s ratio still >3
```

### 4.3 Effort Estimates

- CompressionGenerator: 1-2 days (templates + verified Huffman/LZ impl) — Sonnet tier
- Doom loop eval + serve breaker: 0.5 day
- antidoom_integration.py port: 1 day (FTPO loss from paper)
- Attention wiring + configs: 0.5 day
- Spec/docs updates: 0.5 day
- Total: ~3-4 days solo, can run parallel in subagents

---

## 5. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Synthetic reasoning overtraining → doom loops | antidoom LoRA after stable + temp resample breaker in serve_engine |
| base1b VRAM 1409M doesn't fit 4080 at 128k | DeltaNet 21/7 + FP8 KV + W4A16 weights → 11.6GB target, measured via `torch.cuda.max_memory_allocated` at T11.2 |
| Compression textbook may leak eval (eval has enwik9 compression metric) | decontaminate.py 13-gram check, frozen eval snapshots T10.6 |
| Chonkie + neural compression recursion (compress compressed shard → blow up) | Guard: neural_compress only on raw docs > chunk_size, not on already compressed shards |
| Work separation breach (AGENTS.md absolute) | All new code uses public repos only, no work systems, disclaimer footer mandatory |
| GDrive upload bloat with compression datasets | 500K shards rotating, high-water eviction T10.9 already sheds over-supplied phases |

---

## 6. Suggested Roadmap Order

1. **Immediate (today)**: Land this assessment doc, update OPEN_SOURCE_TOOLCHAIN.md with antidoom + lightCompress links
2. **T12.6** (next): `ava/datagen/compression.py` + spec 02 B6 — adds 30MB verified synthetic (fits existing pipeline, no GPU needed, unblocks Phase0-2 quality)
3. **T7.6**: `evals/doom_loop.py` + `evals/compression_reconstruct.py` — measure baseline before training
4. **T8.6**: Add DoomLoopBreaker to `ava/serve_engine.py` — cheap inference-time fix, no retrain
5. **T11.1 + T11.3** revived: Wire `compressed_conv` and `sparse_compressed` attention via config flags, keep default-off regression guard (same pattern as DeltaNet T11.2)
6. **T9.5 / Phase5**: After `ava_stable_736k.pt` exists, run full antidoom generate+train on Alienware: `uv run antidoom -c configs/antidoom.yaml -r runs/antidoom1 --temp 0.01 --model-name <hf_converted>` OR our port `scripts/antidoom_integration.py`
7. **T9.6**: QAT anneal: last 10% Phase5 steps with KV FP8 + W4 simulation, evaluate compression reconstruction BLEU + frontier macro

All tasks HOME-only, solo, free-tier.

---

## 7. Solo Project Disclaimer

All additions are **Solo personal project, no connection to employer, built with public/free-tier only**. No proprietary data, no work systems, no internal models. Repos referenced (Liquid4All/antidoom MIT, vllm-project/llm-compressor Apache 2.0, ModelTC/LightCompress Apache 2.0, Z-token arxiv public) are public open-source, permissible per AGENTS.md.

## 8. References

- Antidoom: https://github.com/Liquid4All/antidoom
- Dataset: LiquidAI/antidoom-mix-v1.0
- Paper ideas: Antislop (single-token preference), LLM as Token Compressor and Decompressor (Z-token, budget-aware length regularizer), Training LLMs over Neurally Compressed Text (M1→M2), Language Modeling is Compression (DeepMind ICLR 2024, Chinchilla 70B 12x, ImageNet 43.4% vs PNG 58.5%)
- Existing Ava files: `ava/attention/compressed_conv.py` (Zaya1 8x), `specs/11_arch_hillclimb.md` T11.1-T11.3, `OPEN_SOURCE_TOOLCHAIN.md` 1370 lines, `CURRICULUM_LOOP_PLAN.md`, `STATUS.json` today's expansions

---
*Kitty Scout — curiouser about compression* 🐾
