# TODOS — Ava v6.4 End-to-End Pilot
> Live tracker. The **foreman updates this file after every dispatch, verification, and phase transition**, and commits it with each status change.
> Status values: `todo` → `dispatched` → `review` → **`done`** (acceptance command passed) / `blocked(reason)`
> Tier: 🟦 Sonnet (mechanical) · 🟪 Opus (complex) · 👷 foreman-executed

Legend for checkboxes: `[ ]` todo/dispatched/review · `[x]` done (acceptance verified by foreman)

---

## Phase P0 — Environment & scaffolding · spec `specs/01_environment.md`
- [ ] **A1** 🟦 `scripts/setup_env.sh` + pinned CPU deps installed — *accept:* `bash scripts/setup_env.sh && python -c "import torch, tokenizers, fastapi; print(torch.__version__)"` · status: `todo`
- [ ] **A2** 🟦 `ava/__init__.py` + `ava/config.py` (AvaConfig, `load(preset)`, `--count-params` CLI) + `configs/nano.yaml`, `nano_quick.yaml` — *accept:* `python -m ava.config --preset nano --count-params` prints 13–16M · status: `todo` · deps: A1
- [ ] **A3** 🟦 `Makefile`, `pytest.ini`, `.gitignore` additions, `ava/datagen/base.py` ABC — *accept:* `make -n` lists all targets; `pytest --collect-only` clean · status: `todo` · deps: A1

## Phase P1 — Synthetic data generators (4 parallel) · spec `specs/02_data_generation.md`
- [ ] **B1** 🟦 `ava/datagen/logic.py` (P0 corpus ≥30MB: truth tables, valid ND proofs, syllogisms, FOL, critique pairs) — *accept:* size + double-run sha256 identical + `pytest tests/test_datagen.py -k logic` · status: `todo` · deps: A3
- [ ] **B2** 🟦 `ava/datagen/math_gen.py` (P1+P3 ≥40MB: staged arithmetic→probability, CoT, temporal workflow logs) — *accept:* same pattern, `-k math` · status: `todo` · deps: A3
- [ ] **B3** 🟦 `ava/datagen/encyclopedia.py` + `code_gen.py` (P2 ≥50MB: canonical fact corpus [spider/ant, France/China, soccer/rugby, Spanish/French] + exec-verified Python) — *accept:* same pattern + canonical-entity coverage check · status: `todo` · deps: A3
- [ ] **B4** 🟦 `ava/datagen/chat_safety.py` (≥20MB: dialogues, safety/refusal + benign twins, delegation/temporal, counterfactual) — *accept:* same pattern, `-k chat` · status: `todo` · deps: A3
- [ ] **B5** 🟦 `scripts/gen_all_data.py --seed 1234` (runs all four) — *accept:* full corpus regenerated, manifest with per-file sha256 written · status: `todo` · deps: B1–B4

## Phase P1' — Model bug fixes + parameterization · spec `specs/04_model_and_configs.md`
- [ ] **D1** 🟪 Surgical fixes in `model_1b.py`: causal mask (SDPA), rotate_half layout, vision-fusion precedence, `_prev_workspaces` detach + `use_memory` gate, size parameterization, shared per-forward RoPE — *accept:* `pytest tests/test_model.py` green <60s · status: `todo` · deps: A2
- [ ] **D2** 🟪 Fixes in `multi_jspace_module.py`: `JacobianLens.top_concepts`, verbalizer tied to lm_head, batch-size guard, configurable slots/hl/heads — *accept:* top_concepts returns real ids, mass ∈ (0,1) input-dependent · status: `todo` · deps: A2 (same worker as D1)

## Phase P2 — Tokenizer · spec `specs/03_tokenizer.md`
- [ ] **C1** 🟦 `ava/tokenizer.py` + BPE-8192 artifact `data/nano/tokenizer/ava_nano_bpe.json` — *accept:* 1k-doc round-trip exact; ≥3.0 chars/token heldout; `pytest tests/test_tokenizer.py` · status: `todo` · deps: B1–B4 (partial ok)

## Phase P3 — Packing pipeline · spec `specs/05_training.md` §packing
- [ ] **E1** 🟦 `ava/data.py` + `scripts/build_dataset.py` (per-phase uint16 memmaps + idx sidecars + 200k heldout/phase; task_type-pure batches) — *accept:* per-phase token counts ±10% of budget; `pytest tests/test_data.py` · status: `todo` · deps: C1

## Phase P4 — Trainer + J-losses · spec `specs/05_training.md`
- [ ] **F1** 🟪 `ava/jlosses.py` (combined loss exactly per blueprint weights; reuses `MultiJSpaceLosses`) — *accept:* unit test: all loss terms finite, nonzero, correct weighting · status: `todo` · deps: D1, D2
- [ ] **F2** 🟪 `ava/train.py` (WSD, phase manager + RoPE transitions, ckpt/resume, JSONL metrics, `--branch chat --init` real state_dict load + freeze) — *accept:* `pytest tests/test_train_smoke.py`: 50-step loss strictly ↓, kill@30 + `--resume` identical step-50 loss ±1e-4 · status: `todo` · deps: F1 (E1 for real data; stub tensors ok before)

## Phase P5 — Bench + budget lock
- [ ] **G1** 🟦 `scripts/bench_throughput.py` → `runs/bench.json`; budget rule `clamp(tok_s×6h, 15M, 40M)` picks nano vs nano_quick — *accept:* projected base-run ≤12h · status: `todo` · deps: E1, F2
- [ ] **G2** 👷 `scripts/smoke_e2e.sh` full rehearsal (~5 min) — *accept:* exits 0: tiny-train → mini-eval → server boot → curls → teardown · status: `todo` · deps: G1, J1 (server skeleton)

## Phase P6 — Nano training run (foreman-monitored background)
- [ ] **H1** 👷 Base run `python -m ava.train --preset nano --run runs/base` (bg, poll metrics.jsonl, `--resume` on crash) — *accept:* `ava_nano_stable.pt` (step 3369) + `ava_nano_final.pt`; smoothed loss ↓; no NaNs · status: `todo` · deps: G2
- [ ] **H2** 👷 Chat branch `--preset branch_chat --init runs/base/ava_nano_stable.pt` — *accept:* `runs/chat/ava_nano_chat.pt`; log proves stable ckpt hash loaded; frozen spaces unchanged (param-hash check) · status: `todo` · deps: H1

## Phase P7 — Real eval harness (build during P6) · spec `specs/06_evaluation.md`
- [ ] **I1** 🟪 `evals/perplexity.py`, `evals/probes.py`, `evals/jspace_tests.py` (5 canonical tests as real hook-based measurements), `evals/needle.py`, `evals/run_harness.py` — *accept:* runs on smoke ckpt without error; anti-mock grep clean · status: `todo` · deps: D1, D2 (not H)
- [ ] **I2** 👷 Run harness on base + chat finals → `reports/branch_eval_results_real.json` + `REPORT_REAL.md` — *accept:* completes <20 min; all values measured; PASS/FAIL/MEASURED table present · status: `todo` · deps: H1, H2, I1

## Phase P8 — Serving (build during P6) · spec `specs/07_serving_deployment.md`
- [ ] **J1** 🟪 `ava/serve_engine.py` + `server.py` fixes (Optional import, pydantic v2, real backend for all endpoints + new `/health`, `/generate`, `/report`) — *accept:* boots with smoke ckpt; endpoints return input-dependent data; intervene 403-gated · status: `todo` · deps: D1, D2
- [ ] **J2** 🟦 `scripts/make_report.py` → self-contained `reports/index.html` (no CDN) — *accept:* renders all metric series from a sample metrics.jsonl; file works offline · status: `todo` · deps: A3
- [ ] **J3** 🟦 `scripts/smoke_live.sh` curl suite — *accept:* all checks scripted per spec · status: `todo` · deps: J1
- [ ] **J4** 🟦 `Dockerfile` (CPU + CUDA-variant build-arg) + `run.sh` self-host package — *accept:* `docker build` succeeds (or documented dry-run if docker unavailable in container) · status: `todo` · deps: J1

## Phase P9 — Conversion & release · spec `specs/09_conversion_release.md`
- [ ] **K1** 🟦 `scripts/convert_checkpoint.py` → `export/ava-nano/` (safetensors + honest config + tokenizer + modeling files) — *accept:* reload-equivalence: logits match original atol 1e-5 on 10 prompts · status: `todo` · deps: H2

## Phase P10 — LIVE DEPLOY
- [ ] **L1** 👷 Container live: `AVA_CKPT=runs/chat/ava_nano_chat.pt uvicorn server:app --host 0.0.0.0 --port 8000` + `bash scripts/smoke_live.sh` — *accept:* every smoke check green · status: `todo` · deps: H2, I2, J1–J3
- [ ] **L2** 🟦 Vercel static dashboard from `reports/` — *accept:* public URL serves index.html + eval JSON · status: `todo` · deps: I2, J2
- [ ] **L3** 👷 Final results summary appended to README ("Nano pilot results") — *accept:* real numbers, links to reports · status: `todo` · deps: L1

## Phase P11 — Alienware GPU handoff · spec `specs/08_alienware_runbook.md`
- [ ] **M1** 🟦 Runbook complete + `configs/mini.yaml` + `configs/base1b.yaml` — *accept:* foreman review: WSL2 steps, VRAM/throughput math, milestone schedule, ops section all present · status: `todo` · deps: A2
- [ ] **M2** 👷 USER: execute runbook on Alienware — nano sanity → mini (GO/NO-GO) → base1b milestones M1 2B / M2 10B / M3 30B+ · status: `todo` · deps: M1, L1

---

## Foreman log
| When (UTC) | Event |
|---|---|
| 2026-07-09 | Plan approved; specs authored; tracker initialized. All tasks `todo`. |
