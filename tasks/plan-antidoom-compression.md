# Plan: Antidoom + Compression Integration — Dottie v6.4

> Solo personal project, no connection to employer, built with public/free-tier only  
> Tier breakdown: 🟦 Sonnet · 🟪 Opus · 👷 foreman/human

## Goal
Integrate antidoom loop fix and 3-bucket compression (knowledge, serving, neural) into 6-phase curriculum without breaking continuous supply.

## Tasks

### T13.1 🟦 Compression textbook generator `dottie/datagen/compression.py`
- **What**: B6 generator per spec 02 style. Families: shannon 25%, huffman 20%, lz77 20%, arithmetic 15%, bwt_ans 10%, z_token 10%
- **Verifiable**: 
  - entropy = -sum p log2 p recomputed
  - Huffman tree via heapq, avg_len >= entropy, kraft sum <=1
  - LZ77 tuples verified by decompress == original
  - Arithmetic interval contains symbol
- **Schema**: text, task_type, concept, phase, source ; deterministic via private Random(seed)
- **Phases**: p0 shannon, p1 huffman+kraft, p2 lz code, p3 bwt+z long docs 6000+, p5 proofs
- **Accept**: `python -m dottie.datagen.compression --seed 1234 --out /tmp/c --mb 1` → byte-deterministic, `pytest tests/test_datagen.py -k compression`
- **Files**: `dottie/datagen/compression.py`, `dottie/datagen/__init__.py` register, `configs/sources.yaml` add `synth_compression` weights p0 5% p1 10% p2 10% p3 10% p4 5% p5 10% rescaled.

### T13.2 🟦 Doom loop eval `evals/doom_loop.py` + `evals/compression_recon.py`
- **doom_loop**: detector min_span 10 tokens appears >=3 in 200 window, or P(token)>0.95 repeated. Metric: % generations with loop at temp 0, temp 0.7
- **compression**: enwik9 slice 2048b → bits per byte via Dottie CE, also reconstruct task: comrpess paragraph to 144-slot summary then ROUGE/BLEU
- **Accept**: `python -m dottie.evals.doom_loop --help` exits 0, smoke test on 20 prompts no crash.

### T13.3 🟪 Serve breaker + antidoom integration script
- `dottie/serve_engine.py`: add `class DoomLoopBreaker` with sliding window n-gram tracker, if repeat detected, resample once at temp 0.7, log to serve_audit.jsonl
- `scripts/antidoom_integration.py`: generate FTPO jsonl from checkpoint sampling, build rows {context_before, rejected, chosen[]}, train PEFT LoRA r=128 lr 1e-5 with MSE tether lambda_mse 10.0 tau 0.1
- **No vLLM dep** for Hatch VM dry-run; flag `--use-vllm` for Alienware.
- **Accept**: dry-run `python scripts/antidoom_integration.py generate --dry-run --n-prompts 10` creates valid jsonl.

### T13.4 🟪 Model compression wiring
- Wire existing `dottie/attention/compressed_conv.py` via config: `model.attn_mode` enum `full, compressed_conv, gated_deltanet, sparse_compressed`
- Create `dottie/attention/sparse_compressed.py` (DeepSeek Flash 10% KV)
- Update `configs/base1b.yaml`: `compression: {kv_quant: fp8, weight_quant: w4a16}`
- Update `specs/04_model_and_configs.md` VRAM math with compressed numbers:
  - 1409M bf16 2.8GB → W4 0.7GB weights
  - KV at 131k 7.52GB → FP8 3.76GB → + DeltaNet 1.9GB → 2.4GB total
- **Accept**: `pytest tests/test_model.py -k compression` + default-off regression (attn_mode=full unchanged).

### T13.5 🟦 Docs + toolchain
- Update `specs/02_data_generation.md` B6, `specs/06_evaluation.md`, `OPEN_SOURCE_TOOLCHAIN.md` add antidoom, llm-compressor, LightCompress, Z-token papers
- Update `CURRICULUM_LOOP_PLAN.md` phase table
- Update `TODOS.md` Stage 13 new section
- **Accept**: docs build no broken links.

### T13.6 👷 Alienware full run (blocked on T9.3)
- After stable checkpoint: `uv run antidoom -c configs/antidoom.yaml -r runs/antidoom1 --temp 0.01 --model-name hf_model` OR our script
- Evaluate doom loop % before/after, chosen_win ~0.4 target
- Merge LoRA, evaluate frontier macro lift.

## Dependencies
T13.1 no deps, can start now (free-tier, no GPU). T13.2 after T13.1. T13.3 after T13.2. T13.4 parallel to T13.1. T13.5 after.

## Risks
- Over-suppressing Wait/So etc creates new loops — mitigated by regularisation strengths.
- Compression textbook may increase duplicate rate — dedup simhash th3 already handles.
- Quantization may degrade reportability mass — need eval gate >0.02

## Acceptance overall
- New generator green + 30MB
- Doom loop metric exposed in /report
- Serve breaker logs loops broken
- Default attn_mode=full unchanged behavior
- Docs updated with disclaimer
