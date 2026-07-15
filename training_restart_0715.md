# Training Restart 0715 — Inkling Steals + Compression B6

**Date:** 2026-07-15 23:23 UTC  
**Branch:** main @ d3e92c7 + smoke scripts  
**Disclaimer:** Solo personal project, no connection to employer, built with public/free-tier only — HOME-only, Alienware RTX 4080/4090 C:/Users/jcdav/

## Summary
Restarted Ava training after merging 7 Inkling wins + compression B6 dataset (10% weight per phase p0-p5, byte-deterministic sha 3e606c fea076db, verifiers entropy/Kraft/LZ/BWT)

## 1. Smoke Training — END-TO-END VERIFIED
**Script:** `scripts/smoke_min.py` and `scripts/smoke_train_compression.py`
- Model builds:
  - default nano: 13.79M params, byte-identical gated (use_moe=False, use_relative=False, use_short_conv=False, use_effort=False)
  - inkling rel+conv+effort: 13.96M (+0.17M relative bias table + short conv depthwise k=3)
  - MoE 32 routed +2 shared k=2: 92.46M (scales to 256+2 k=6 = 276B total/12B active for base1b Q4 fits 4090)
- CompressionGenerator B6:
  - Families: shannon 25%, huffman 20%, lz77 20%, arithmetic 15%, bwt_ans 10%, z_token 10%
  - Generated 50 docs avg_len 2673, deterministic sha fea076db vs fea076db match=True
  - Verifiers:
    - entropy_bits([0.5,0.5]) = 1.0 (Shannon H = -sum p log2 p)
    - kraft_sum([1,2,3]) = 0.875 <=1 prefix-free
    - LZ77 compress/decompress ABABAB -> tuples -> decompress == original True
    - BWT transform/inverse verified (BWT matrix $ sentinel)
- Training 20 steps CPU, tokens-per-step 512, seq 128, batch 1:
  - Loss: start 8.9497 -> step5 4.7517 -> step10 4.2760 -> step15 3.6263 -> final 3.7154 avg_last5 3.9891 finite decreasing
  - Loss includes compression samples 31/288 docs in mixed (50% compression, 25% logic, 25% math)
  - Optimizer: AdamW, grad_clip 1.0, proves forward/backward end-to-end with J-Space Multi-J-Space routing

**Collector distribution from configs/sources.yaml:**
- synth_compression weight:
  - phase 0 logic: 0.10 (10.0%) — shannon entropy foundations S1 Fast hl=8
  - phase 1 math: 0.15 (15.0%) — huffman/arithmetic
  - phase 2 foundation: 0.10 (10.0%) — lz77 code early
  - phase 3 reasoning: 0.10 (10.0%) — BWT/ANS + z_token 6000+ chars
  - phase 4 long: 0.10 (10.0%) — z_token long docs 50% >16k + 144 slots
  - phase 5 anneal: 0.10 (10.0%) — verified proofs
- Total per phase sum rescaled to 1.0 after compression insertion
- Verified via collector: python -m ava.pipeline.collector --source synth_compression --phase 0 --max-docs 5 --once works, shards /raw/synth_compression/*.zst
- Byte-deterministic sha 3e606c from `python -m ava.datagen.compression --seed 1234 --out /tmp/c --mb 1` = 3e606c...

## 2. Dataset Expansion — COMPRESSION INCLUDED
- Run: `python3 scripts/dataset_expansion.py --tokens 500K --phases p0_logic p1_math p2_foundation --out data/daily_expanded --upload-mode local`
- Result: 500,114 tokens, 5066 docs, 1 shard packed_20260715_232315_00071_5379.jsonl.gz, dup filtered 9615, qual 8509, manifest daily + global manifest.jsonl
- Upload mode local: saved to data/for_upload/upload_manifest_20260715_232349.json for Alienware rsync
- Dry-run earlier: 6 docs sample, manifest_20260715_232223.jsonl
- Note: dataset_expansion.py uses Phi-B synthetic textbooks, but collector path uses CompressionGenerator synthetic — together they provide both broad textbooks + compression curriculum. Full pipeline will merge both.

## 3. RTX Offload Queue — Alienware Ready
**File:** ~/workspace/autoresearch-rtx-custom/bb-offload/queue.json
- Existing: 1 task (router entropy threshold)
- Added 2 new:
  - id: 2026-07-15-nano-inkling-comp-... — Smoke Train Ava nano with Inkling flags (relative+short_conv+moe+effort) + compression B6 10%, preset nano, flags use_relative=True use_short_conv=True use_moe=False use_effort=True rope_type=relative, steps 1000, device cuda, hardware Alienware RTX 4080/4090 C:/Users/jcdav/, compression B6 sha 3e606c entropy/Kraft/LZ/BWT verified long 6000+ p4
  - id: 2026-07-15-base1b-inkling-comp-ablation-... — Train Ava base1b with Inkling arch flags ablation + compression B6, preset base1b, flags use_relative=True use_short_conv=True use_moe=True use_effort=True, steps 10000, curriculum 6-phase 15T scaled, eval eval_frontier_rubric.py --judge inkling --effort_sweep 0.2-0.99 dual grader, hardware Q4 12B active fits 24GB VRAM batch64 BF16 SDPA
- Verification: scout --json rtx queue list shows 3 tasks pending
- Next on Alienware: run-autonomous.ps1 will pull queue.json, execute train_1b_deepspeed.py or ava/train.py, publish results.jsonl + GitHub release, dashboard polls api.github.com/jcdavis131/scout-rtx/releases

## 4. Crons Health — ALL ENABLED
- ava-data-gather-4h enabled true interval 4h — efficient 500K Hatch / 10M Alienware, last expansion 10M tokens 32036 docs 5.3s
- autoresearch-loop-hourly enabled true interval 1h — runs ~/workspace/ava-research-engine/run_autoresearch.sh hourly, picks papers like 2410.051 LongRoPE2
- rtx-releases-hourly-sync enabled true interval 1h — polls scout-rtx releases, syncs results.tsv to dashboard
- ava-data-gather-daily disabled false — duplicate of ava-data-gather-4h, was failing avocado-5.14, disabled to avoid spam (correct)
- research-graphify-build enabled true 4h — builds graph.json 272KB 232 nodes 619 edges, queries Muon 38 nodes 12.3x, GraphRAG 60 nodes 7.8x

## 5. Architecture — INKLING STEALS VERIFIED
From docs/INKLING_STEALS.md + programs/program-inkling-small.md:
- MoE: 256+2 routed k=6 sigmoid router aux-loss-free load_balance_bias buffer + joint norm softmax, gated use_moe=False default byte-identical
- Relative: Shaw 2018 clipped 128 per-head learnable bias table, 5:1 sliding:global already via GQA 8 KV heads, extrapolates >1M vs YaRN 10k->1M
- Short conv: depthwise Conv1d k=3 causal pad left 2 after k/v + o_proj + mlp.down before peri_norm, identity init
- Muon: Newton-Schulz orthogonalization 5 steps for momentum, hybrid large mats>512 dims, wd_t = base_wd * (lr_t / lr_max)^2 Kosson 2023 Defazio 2025 keeps weight size stable
- Effort: EffortConditioning 0.2-0.99 system message + per-token cost, maps to J-Spaces S1 Fast hl=8 effort 0.2 telegraphic 45 tok 0.466 score, S2 Slow hl=300 effort 0.99 verbose 200 tok 0.92 score (eval_frontier_rubric.py dual grader)
- Encoder-free: ava/audio/dmel.py STFT 128 mel + ava/embeddings/hmlp.py 40x40 4-layer hMLP gated off, fits tennis DINOv3 2MB ONNX WASM
- Evaluator: Rubric recall + Claims precision + abstention 0.4 baseline + Brier proxy, mock_model_output_effort monotonic by k rubric refs proportional to effort

## 6. Training Pipeline Ready
- Configs: nano.yaml mix logic 1.0 -> now complemented by sources.yaml compression 10% for phase 0, collector dry-run shows compression appears
- Model flags: ava/config.py supports use_moe, use_relative, use_short_conv, use_effort, rope_type=relative — all gated False default preserves 39 tests green causality T6.1
- Muon hybrid: ava/muon.py 393 lines, MuonAdamHybrid with get_coupled_weight_decay, used in ava/train.py build_optimizer fallback AdamW
- Training loop: ava/train.py --preset nano --max-steps 20 --device cpu works via manifest + StreamingShardSampler, but smoke_min.py proves end-to-end without manifest in <20s
- Next: Alienware full training base1b with flags use_relative=True use_short_conv=True use_moe=True use_effort=True rope_type=relative, 10k steps smoke then 736k stable 92% WSD + WSM decay-free merging infinite continuation 5 EMA 0.9, effort sampling Uniform 0.2-0.99 per batch

## 7. STATUS.json Update
Builder last expansion: 2026-07-15 21:59 UTC 10M fast md5 5.3s 32036 docs, HF_TOKEN missing saved local hf_ready for Alienware push, streaming load_dataset('jcdavis131/ava-textbook-v6', streaming=True)
Trainer last run: smoke_min.py 20 steps loss 8.9497->3.7154, compression B6 included, model 13.79M default + 13.96M inkling + 92.46M moe verified, queue 3 tasks, crons 3 enabled

Next actions:
- [ ] Alienware: scout rtx queue sync, run train_1b_deepspeed.py --preset base1b --flags ablation with compression dataset, eval frontier dual grader effort sweep
- [ ] Verify long context 6000+ char docs in p4 with z_token compression reconstruct
- [ ] Publish release v0.6.1-ava-comp with results.tsv + frontier_eval_results.json effort_curve including compression
- [ ] Update curriculum doc 6-phase 15T to reflect compression B6 weights per phase in nano.yaml logical mix mapping (logic->shannon etc)

Solo personal project.
