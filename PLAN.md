# Ava AGI Factory v6.4 — End-to-End Execution Plan
### Data → Tokenizer → Train → Eval → Deploy-Live, foreman + worker-bee orchestration

**Status: READY FOR EXECUTION** · Tracker: [`TODOS.md`](TODOS.md) · Protocol: [`ORCHESTRATION.md`](ORCHESTRATION.md) · Specs: [`specs/`](specs/)

---

## 1. Why this plan exists

The v6.4 repo is a **design blueprint, not a working trainer**. An audit (2026-07-09) found:

**Real, reusable code**
- `model_1b.py` — actual `nn.Module` architecture (3-regime transformer + YaRN RoPE + QK-Norm), *with bugs* (no causal mask, broken rotary pairing, autograd leak — full list in `specs/04_model_and_configs.md`)
- `multi_jspace_module.py` — Multi-J-Space (S1/S2/Critic/Planner slot-attention workspaces, Router, Arbitration) + `MultiJSpaceLosses`
- WSD/RoPE/branch schedules in `train_1b_deepspeed.py`; curriculum manifests `dolma_config.yaml`, `nemo_curator_pipeline.yaml`

**Mock / must be built**
- No tokenizer exists anywhere ("ava-tokenizer" is referenced, never defined; vocab inconsistent 128000 vs 32000)
- Training loop = 5 steps of `loss=torch.tensor(1.0)` writing *text files* as "checkpoints"; DeepSpeed never initialized
- Eval harness returns hardcoded PASS; `server.py` returns canned JSON and crashes on import; `convert_to_hf.py` writes a fake config; wandb logging is print-only; zero tests

This plan turns the blueprint into a real system in which **every stage is genuinely executed**: real synthetic data, real BPE tokenizer, real training with the Multi-J-Space losses, real WSD stable checkpoint + branch fine-tune, real measured evals, and a live deployed server backed by real inference.

## 2. Hard constraints (verified)

| Constraint | Consequence |
|---|---|
| Build container: 4 CPU cores, 15 GB RAM, **no GPU**, ephemeral | container trains the **nano** pilot only; everything committed; artifacts regenerable from seeds |
| huggingface.co **blocked** by proxy; PyPI open | 100% locally-generated synthetic data (blueprint's Phase 0 is 60% synthetic by design); `tokenizers` lib trains BPE locally |
| wandb assumed blocked in container | JSONL metrics + self-contained HTML dashboard; wandb-offline optional on the GPU machine |
| Real GPU: **Alienware m16, RTX 4080 Laptop 12 GB, Windows + WSL2** | bf16 + 8-bit AdamW + gradient checkpointing + SDPA-flash; no DeepSpeed dependency |

## 3. The scale ladder

Validation is incremental — each rung gates the next:

| Rung | Params | Where | Tokens | Wall-clock | Purpose |
|---|---|---|---|---|---|
| **smoke** | ~2M | container CPU | ~1M | ~5 min | `scripts/smoke_e2e.sh` rehearsal gate before any long run |
| **nano** | ~14M | container CPU | 30M (fallback 15M) | 5–12 h background | proves every stage real; produces the **live-deploy checkpoint** |
| **mini** | ~160M | RTX 4080 12GB | ~2.5B | ~3–5 days | GPU-scale validation of curriculum + J-losses; GO/NO-GO gate for base1b |
| **base1b** | ~1.0–1.2B | RTX 4080 12GB | milestones: 2B → 10B → 30B+ | ~9–12 days / 1B tokens | the target. Vocab-32k **tied** embeddings (trims blueprint's accidental ~2.9B); stop-anytime WSD stable checkpoints; code/math/chat branches fork from any stable ckpt |

Honest math: Chinchilla-optimal for 1B ≈ 20B tokens ≈ ~6 months on this GPU. Hence milestones with decision gates, not one heroic run. Full arithmetic and VRAM budgets: `specs/08_alienware_runbook.md`.

**Nano config (the container pilot):** d_model 256, 4 heads × 64, layers 2 text / 6 fusion / 2 reasoning, J-slots 32/64/16/32 (unchanged from blueprint — slot counts are d_model-independent), half-life targets scaled 8/60/30/50, vocab 8192 BPE, verbalizer tied to lm_head → **≈14M params**. Curriculum compressed to 6 phases (P0 logic 5M → P5 anneal 3M, seq 256→1024, RoPE 10k→32k NTK 1.2), step = 8,192 tokens → ~3,662 steps. WSD: warmup 110 → stable 1e-3 → **stable checkpoint at step 3,369 (92%)** → cosine to 1e-4. Chat branch: 3M tokens from stable ckpt, `system1`+`system2` frozen.

## 4. Phase graph

```
P0 scaffold (Sonnet) ─┬─→ P1 datagen B1–B4 (4 × Sonnet, parallel) ─→ P2 tokenizer (Sonnet) ─→ P3 packing (Sonnet) ─┐
                      └─→ P1' model fixes (Opus) ─────────────────────→ P4 trainer + J-losses (Opus) ─→ P5 bench ──┤
                                                                                                                   ▼
                            P6 NANO TRAIN  (foreman-monitored background run; stable ckpt @92% → chat branch)
                            ├─→ P7 eval harness   (Opus — built DURING P6, run after)
                            ├─→ P8 serving        (Opus engine + Sonnet report/Docker/smoke — built DURING P6)
                            └─→ P9 convert+release (Sonnet) ─→ P10 LIVE DEPLOY + smoke_live.sh ─→ P11 Alienware handoff
```

Maximum parallelism: after P0, the four data generators, the model-fix task, and doc/serving skeletons all proceed concurrently. P7/P8 are built while P6 trains, so wall-clock is dominated by the training run itself.

## 5. Deliverables per phase (detail in specs/)

| Phase | Spec | Key outputs | Done-gate (foreman runs) |
|---|---|---|---|
| P0 | `specs/01_environment.md` | `scripts/setup_env.sh`, `ava/config.py`, configs, Makefile, pytest scaffold | import check + `--count-params` ≈ 14M |
| P1 | `specs/02_data_generation.md` | `ava/datagen/{logic,math_gen,encyclopedia,code_gen,chat_safety}.py`, ≥140MB raw JSONL | double-run sha256 identical; `pytest -k datagen` |
| P1' | `specs/04_model_and_configs.md` | surgical fixes to `model_1b.py` + `multi_jspace_module.py`, `tests/test_model.py` | causality/rotary/resume tests green < 60 s |
| P2 | `specs/03_tokenizer.md` | `ava/tokenizer.py`, BPE-8192 artifact | round-trip 1k docs; ≥3.0 chars/token |
| P3 | `specs/05_training.md` §packing | `ava/data.py`, per-phase uint16 memmaps + heldout | phase token counts ±10% of budget |
| P4 | `specs/05_training.md` | `ava/train.py`, `ava/jlosses.py` | 50-step loss ↓; kill+`--resume` bit-exact |
| P5 | `specs/05_training.md` §bench | `runs/bench.json`, budget lock | projected ≤ 12 h else auto nano_quick |
| P6 | — (foreman op) | `ava_nano_stable.pt`, `ava_nano_final.pt`, `ava_nano_chat.pt` | heldout PPL ≪ step-200 PPL; no NaNs |
| P7 | `specs/06_evaluation.md` | `evals/*`, real 5-test J-Space harness | measured results, anti-mock grep clean |
| P8 | `specs/07_serving_deployment.md` | real `server.py` backend, `reports/index.html`, Dockerfile | `scripts/smoke_live.sh` all green |
| P9 | `specs/09_conversion_release.md` | `export/ava-nano/` safetensors | reload-equivalence atol 1e-5 |
| P10 | `specs/07_serving_deployment.md` §deploy | live uvicorn :8000 · Vercel static dashboard · self-host package | curl suite + public URL |
| P11 | `specs/08_alienware_runbook.md` | WSL2 runbook, `configs/mini.yaml`, `configs/base1b.yaml` | user executes on Alienware |

## 6. Orchestration model

This session = **foreman**. Workers are dispatched per `ORCHESTRATION.md`:
- **Sonnet workers** — mechanical, well-specified tasks (scaffolding, generators, tokenizer, packing, bench, HTML report, Docker, docs)
- **Opus workers** — complex/correctness-critical tasks (model bug fixes, trainer + J-losses, eval harness, serve engine)
- Foreman never marks a task done on a worker's word: it **runs the spec's acceptance command(s)** first, then updates `TODOS.md` and commits.
- Long runs (P6) execute via `nohup … --resume`-able background process; foreman polls `runs/*/metrics.jsonl` and restarts on crash.
- `.claude/workflows/ava-build.js` encodes the build fan-out (P0–P5) as an executable workflow for one-command dispatch.

## 7. Deployment ("test live") — three targets

1. **Container live test (the gate):** `AVA_CKPT=runs/chat/ava_nano_chat.pt uvicorn server:app --host 0.0.0.0 --port 8000` — real `/generate`, `/jspace/inspect`, gated `/jspace/intervene` with audit log, real eval JSON, WS stream. Verified by `scripts/smoke_live.sh`.
2. **Vercel static dashboard:** `reports/` (self-contained HTML, no CDN) deployed for a persistent public URL — training curves, J-Space metrics, eval table.
3. **Self-host package:** Dockerfile (CPU + CUDA variants) + `run.sh` so the Alienware can serve any checkpoint it trains.

## 8. Success criteria

- All pytest suites green (causality, rotary, determinism, round-trip, resume-equivalence)
- Nano run: smoothed lm_loss monotonically decreasing; final heldout PPL substantially below early-run PPL; three real `torch.load`-able checkpoints
- Eval report contains **only measured numbers** (anti-mock grep enforced)
- Live server: every `smoke_live.sh` check passes; inspect output is input-dependent; intervention gating enforced (403 without env flag)
- Alienware runbook validated at least through environment-check steps

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| CPU throughput below estimate → nano run > 12 h | P5 bench gates budget; auto-fallback to `nano_quick` (15M tokens) |
| J-losses destabilize tiny model | j_weight schedule 0.08→0.15 per blueprint; per-loss finiteness asserts; loss-spike alarm in foreman monitoring |
| 14M params too weak for canonical eval flips | eval bars set honestly (report MEASURED values; PASS bars are nano-scaled); mini rung re-runs same harness at 160M |
| Container dies mid-run | checkpoints every 250 steps + `--resume` bit-exact; code always committed; data regenerable from seed 1234 |
| 1B on a 12GB laptop GPU is months at Chinchilla scale | milestone schedule (2B/10B/30B+) with WSD stable ckpts = stop-anytime value; mini rung is the GO/NO-GO gate |
