# TODOS — Continuous Pipeline

Foreman updates after every dispatch and verification. A task is `done` only when the foreman has
**run its acceptance command and seen it pass** — never on a worker's word.

Tiers: 🟦 Sonnet (mechanical) · 🟪 Opus (correctness-critical) · 👷 foreman/human

---

## Stage 0 — Host prep ✅
- [x] **T0.1** Identify the 1.8GB GPU process — **NOT stray**: it is `vector-hoops/pipeline/sweep_v5.py --epochs 40 --seeds 7,13,21 --device cuda --resume`, a live sweep. Left running. nano/mini (~5-7GB) coexist; base1b needs it gone.
- [x] **T0.2** `docker builder prune -a` → **reclaimed 25.06GB** (est. was 14GB). Volumes untouched.
- [x] **T0.3** Re-measure. C: shows 26.8GB free; Docker freed 25GB *inside* its 45GB ext4 VHDX, which it reuses before growing. Effective headroom ≈ 50GB. `image prune` not needed.
- [x] **T0.4** `.wslconfig` written (`memory=10GB`, `processors=24`, `swap=8GB`, `sparseVhd=true`). ⚠️ Needs `wsl --shutdown` + Docker Desktop restart to apply — **deferred, requires your OK** (stops all distros).
- [x] **T0.5** Volumes `ava_{raw,packed,ckpt,state,reports}` created **and chowned to uid 1000** (docker creates them root-owned; the manifest could not create its DB).

## Stage 1 — Docker infrastructure ✅
- [x] **T1.1** 🟦 `docker/Dockerfile.cpu` — *accept:* streams 2 TinyStories rows in-container ✅
- [x] **T1.2** 🟦 `docker/Dockerfile.gpu` — based on `python:3.11-slim` + cu124 wheels (not `nvidia/cuda:*-runtime`: the wheels vendor their own CUDA libs, saving ~5GB). *accept:* `torch.cuda.is_available()` + bf16 matmul on the 4080 ✅
- [x] **T1.3** 🟦 `docker-compose.yml` (collector×4, curator×6, trainer, server, janitor; named volumes; GPU reservations) — *accept:* `docker compose config` valid ✅
- [x] **T1.4** 🟦 `Makefile`, `.dockerignore`, `.env.example`, `.gitattributes` (CRLF breaks make recipes and shebangs in Linux containers) ✅

## Stage 2 — Manifest + shard flow ✅
- [x] **T2.1** 🟪 `ava/pipeline/manifest.py` — WAL SQLite, `BEGIN IMMEDIATE` atomic claims, leases + requeue, state-machine guards, tokenizer freeze gate, resumable cursors, structural val/test protection
- [x] **T2.2** 🟪 `ava/pipeline/flow.py` — backpressure predicates, `DATA_STARVED`, phase prefetch
- [x] **T2.3** 🟦 `configs/pipeline.yaml` — watermarks sized for this host
- [x] *accept:* **28 tests** — 12 threads + 4 processes over 1000 shards: zero double-claims, zero lost shards. **Negative control:** downgrading to `BEGIN DEFERRED` makes it fail, so the test is not vacuous ✅

## Stage 3 — Collector ✅
- [x] **T3.1** 🟪 `ava/pipeline/collector.py` — HF streaming, backoff+jitter, resumable cursors, 256MB zstd shards, atomic publish (`.tmp`→fsync→`os.replace`→register), backpressure-aware. *accept:* 15 offline tests ✅ + **live**: streamed 500 TinyStories docs, restarted, resumed at cursor 500 with no duplicate doc_ids ✅
- [x] **T3.3** 🟦 `configs/sources.yaml` — 11 sources, all verified `200` + `gated:false` against the HF API. `bigcode/the-stack-smol` excluded (`gated:"auto"`). Per-phase weights sum to 1.0 (asserted)
- [x] **T3.2** 🟦 `ava/datagen/*` — logic / math / encyclopedia / code / chat_safety. 35 tests that check **content**: an independent proof checker re-derives each natural-deduction proof; every math answer is recomputed; every code snippet is re-exec'd; spider→8 / ant→6 / France→Paris never contradicted with ≥40 paraphrases each. Byte-deterministic.
- [x] **T3.4** 👷 **Reconciled.** Deleted the collector's inline toy generators (a stub logic doc, a hardcoded refusal) in favour of `ava/datagen`. Three bugs surfaced: the registry was **relabelling** generator output (needle docs were phase-1 arithmetic stamped phase-4); needle actually lives in `EncyclopediaGenerator`, not `MathGenerator`; and cursors keyed by source alone would make a multi-phase generator resume into the wrong subsequence (now per `(source, phase)`). `synth_code` had no registry entry at all. *accept:* cpu 97 / gpu 56 ✅ + live shard written and inspected ✅

## Stage 4 — Curator ✅
- [x] **T4.1** 🟪 `clean.py` — normalize / is_english / Gopher heuristics / edu_score / PII scrub (conservative: leaves `0xDEADBEEF`, bare digit runs)
- [x] **T4.2** 🟪 `dedup.py` — sha256 exact + MinHash LSH (9×13 bands @ 0.8) in its own WAL DB; `add_if_new` is check-and-insert in one `BEGIN IMMEDIATE` for cross-replica safety
- [x] **T4.3** 🟪 `decontaminate.py` — 13-gram + short-phrase floor (≥5 words). **Both directions tested**: every eval prompt is removed; "Spiders possess eight legs." is kept
- [x] **T4.4** 🟪 `split.py` — `bucket(sha1(doc_id))`, order-invariant, rerun-stable
- [x] **T4.5** 🟪 `pack.py` — uint16 `.bin` + `.idx.json`, vocab≤65535 asserted, frozen-tokenizer gate
- [x] **T4.6** 🟦 `curator.py` service loop; SIGTERM-graceful; `fail()` never crashes the container
- [x] *accept:* 19 curator tests, **62/62 suite** ✅
- [x] **Deviation accepted:** `complete()` **before** deleting raw. Spec said the reverse; the worker was right — deleting first then crashing would requeue a row whose raw file is gone, losing data. Worst case now is an inert orphaned file.

## Stage 5 — Tokenizer bootstrap + throughput gate
- [ ] **T5.1** 🟦 Collect a 2GB stratified bootstrap sample (collector `--bootstrap-sample`)
- [ ] **T5.2** 🟦 `ava/tokenizer.py` — byte-level BPE 32k (nano 8k) + specials; freeze + sha256 into manifest
- [ ] **T5.3** 👷 Freeze gate live-check: curator refuses to pack against a mismatched tokenizer hash *(already enforced in manifest; needs an end-to-end assertion)*
- [ ] **T5.4** 🟦 `scripts/bench_pipeline.py` — *accept:* curation tok/s ≥ 3× trainer tok/s

## Stage 6 — Model + trainer 🟡
- [x] **T6.1** 🟪 Model fixes — **the big one.** *accept:* 28 tests ✅
  - [x] Causal mask (SDPA). Bare transformer stack now measures **exactly 0.0** logit change at positions < t
  - [x] **J-Space was non-causal**: it mean-pooled the whole sequence and broadcast it everywhere (measured leak ~0.20). Now **chunk-recurrent** — broadcast into chunk *c* comes only from chunks < *c*
  - [x] `rotate_half` half-split (was interleaved, disagreeing with cos/sin → garbage rotation)
  - [x] `_prev_workspaces` detach + batch guard (backward-through-freed-graph on step 2)
  - [x] `JacobianLens.top_concepts` implemented (was dead → `verbalizable_mass` constant 0.06)
  - [x] Verbalizer tied to lm_head (was allocating 2×[V,D] per workspace, discarding one)
  - [x] **Initialization** wired: init loss 196 → **9.07** vs ln(8192)=9.011. Overfits one batch to 0.05/30 steps
  - [x] GQA + SwiGLU + gradient checkpointing (config-gated, causality-tested)
  - [x] Param counts corrected: nano 13.8M, mini 171.3M (was 270M — `tie_verbalizer` must stay true), base1b **1409M** (spec said 1.17B)
- [x] **T6.2** 🟪 `ava/jlosses.py` — combined objective with blueprint weights
- [ ] **T6.3** 🟪 `ava/data.py` — `StreamingShardSampler`: claims PACKED shards, mixes by curriculum weight, `task_type`-pure batches, blocks with `DATA_STARVED` not a crash
- [ ] **T6.4** 🟪 `ava/train.py` — WSD, phase manager + RoPE transitions, grad-accum, bf16, AdamW8bit, ckpt/resume bit-exact, `metrics.jsonl`, `--branch chat --init` real `load_state_dict`
- [ ] **T6.5** 🟦 `ava/pipeline/janitor.py` — watermarks, delete CONSUMED (never val/test), ckpt rotation

## Stage 7 — Real evaluation harness
- [ ] **T7.1** 🟪 `evals/perplexity.py` — val (in-training) / test (milestones only)
- [ ] **T7.2** 🟪 `evals/probes.py` — exact-match greedy; **no PASS bars inherited from the 14M synthetic assumptions**
- [ ] **T7.3** 🟪 `evals/jspace_tests.py` + `interventions.py` — the 5 canonical tests as real forward-hook measurements on live workspaces, concept vectors from **real tokenizer ids**
- [ ] **T7.4** 🟪 `evals/needle.py` — native ctx + eval-time YaRN
- [ ] **T7.5** 🟪 `evals/run_harness.py` → `reports/eval_real.json`. *accept:* `tests/test_no_mock.py` fails if any mock literal (`0.82`, `0.983`, `0.91`) appears unconditionally

## Stage 8 — Live serving
- [ ] **T8.1** 🟪 `ava/serve_engine.py` — real `generate` / `inspect` / `intervene` (+ `runs/serve_audit.jsonl`)
- [ ] **T8.2** 🟪 `server.py` — fix `from typing import Optional` (import-time `NameError`), pydantic-v2 `Field(alias="from")`, wire to engine, keep the 403 gate, add `/health` `/generate` `/report`
- [ ] **T8.3** 🟪 Hot-reload `ckpt/latest` — experiment against the model *while it trains*
- [ ] **T8.4** 🟦 `scripts/make_report.py` → self-contained `reports/index.html` (no CDN)
- [ ] **T8.5** 🟦 `scripts/smoke_live.sh`

## Stage 9 — Scale ladder
- [ ] **T9.1** 👷 nano smoke: all five services, ~10 min. Gate = *the loop works*
- [ ] **T9.2** 👷 mini (171M, ~2.5B tokens, 3–5 days). Watch `hl_est → target`, `route_probs` separating by `task_type`, val PPL ↓. Serve throughout
- [ ] **T9.3** 👷 **GO/NO-GO** for base1b, on mini's `reports/eval_real.json`. Also decide the base1b trim: 1409M × (bf16 weights + grads + AdamW8bit) = 8.4GB before activations, against ~11.6GB. Options: drop `n_fusion_layers` 28→24 (−92M), or narrow the workspaces
- [ ] **T9.4** 👷 base1b milestones M1 2B → M2 10B → M3 30B+
- [ ] **T9.5** 👷 Branch fine-tunes (code/math/chat) from any stable checkpoint

## Docs
- [x] `PLAN.md`, `TODOS.md`, `ORCHESTRATION.md` rewritten for the continuous pipeline
- [ ] `specs/` refresh — `specs/04` is still accurate; `specs/08` param math needs the J-Space correction

---

## Open risks
1. **base1b VRAM.** 1409M is 20% over spec. Not yet proven to fit. Decided at T9.3.
2. **`trust_remote_code` sources.** `proof-pile-2` and `github-code` fetch a loader script from HF at runtime; an upstream change can break collection mid-run.
3. **Only `tinystories` (HF) and `synth_logic` (synthetic) were live-run.** The other 5 HF sources are API-verified but not yet pulled. `fineweb-edu`'s `score` field name is taken from the spec, not observed.
4. **Decontamination coupling.** `evals/eval_sets.py` and `ava/datagen/encyclopedia.py` must keep their phrasings distinct; verbatim matching is what separates prompt-form from fact.
5. **`.wslconfig` not applied** — needs `wsl --shutdown`.
