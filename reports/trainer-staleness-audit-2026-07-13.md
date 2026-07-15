# Trainer Staleness / Crash-Loop Audit — 2026-07-13

Cross-repo review of `ava-agi-factory-v6-4` (this repo, live), `ava-open-harness`,
`ava-skills`, and `cursor-agent-skills`, run against the LIVE mini pipeline
(step ~3780, P2 foundation, RTX 4080 Laptop 12GB, Docker Desktop/WSL2 on a
Windows 11 laptop). Method: 10 parallel deep-readers + in-container forensics
(shard scans, sqlite manifest queries, py-spy, Windows event-log correlation),
each critical/high finding adversarially re-verified against the code, then
fixes implemented behind regression tests.

## Executive summary

The trainer was never deadlocked on data. The "Trainer Stale" banner conflated
four separate problems:

1. **Crash loop (42 process starts / 35 resumes in 3.2 days):** recurring
   `RuntimeError: CUDA error: unknown error` / `CUBLAS_INTERNAL_ERROR`, raised
   from `backward()` (and once from the router forward — asynchronous CUDA
   errors surface at the next sync point, so the call site varies). Root
   condition: the GPU sits at **97% VRAM (11,910/12,282 MiB)**; ~2GB of that is
   a per-micro-batch fp32 cross-entropy transient. Crash-timeline forensics:
   45/47 restarts happened with the host awake — this is memory-pressure/driver
   instability, not sleep. Fixed via chunked+checkpointed CE (bit-exact),
   dropping retained logits across the optimizer step, and
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
2. **False staleness alarms:** after every restart the adaptive threshold
   collapsed to its 180s floor (fewer than 2 step rows in the new run) while a
   P2 recovery legitimately takes ~15 min to its first step event; staleness
   was also measured from the last *step* rather than the last trainer event.
   Fixed: cadence falls back to pre-restart history, liveness counts any
   trainer event, and a distinct "Trainer recovering" mode shows between
   resume and first step.
3. **Real multi-hour gaps were the host, not the code:** overnight the laptop
   ran on battery with the GPU power-throttled to ~780MHz/22W (~100 tok/s →
   a 10-step metrics interval stretches to ~7h), plus Modern Standby freezes
   of the WSL2 VM and one dirty shutdown (Kernel-Power 41). **Operational fix
   (user action): keep the laptop on AC power with High Performance profile
   while training** — no code can compensate for a 22W GPU cap.
4. **Data-plane defects silently degraded the run** (found while auditing the
   "1.5B runway"):
   - **Collector mixture collapse:** the smooth weighted round-robin was
     rebuilt from scratch every loop iteration, so it always picked the
     max-weight source. P2's configured 30/20/15/15/10/10 diet shipped as
     **100% fineweb_edu / task_type=automatic**. (This is also why `report`,
     `selectivity` and friends had nothing to do: no tagged/deliberate data.)
   - **Data treadmill:** the sampler's task-type round-robin drew
     `len(TASK_TYPES)`=4 times per shard regardless of types present, so
     single-type shards (the P2 norm) were trained **4x per claim**. Only
     ~133M unique tokens back the ~983M `tokens_done`.
   - **Shard stranding:** every claim increments `attempts`; the trainer held
     shards ~4.5h against a **900s default lease** (pipeline.yaml's
     `leases: train_seconds: 3600` was never parsed), so lease expiry +
     crash-restarts ratcheted good PACKED shards to the `attempts>=3` cap —
     unclaimable forever, yet still counted by `tokens_ready` (~122M tokens of
     phantom runway at audit time).
   - **Phase-transition leak:** the old phase's partially consumed shard
     stayed in `sampler._held` and fed the new phase's stream.
   - **P3 OOM time bomb:** `micro_batch_for()` ignored seq length; at P3
     (seq 2048) activations would have doubled on a GPU already at 97%.

Also fixed: the J-space half-life freeze — `decay_factor()` clamped at 0.99,
capping half-life at 68.97 tokens, below System2's target (300, needs
d=0.99769) and Planner's (150, 0.99539); both initialized above the ceiling and
sat there with zero gradient, freezing `half_life` loss at exactly 6.767e-05
(= 0.8·(0.99769−0.99)² + 0.7·(0.99539−0.99)²) from step 1. Ceiling now 0.9999.

## Repo verdicts

- **ava-open-harness** is a standalone eval harness (~1000 LOC). It does NOT
  manage `/packed` and holds no locks the trainer can contend on; the packed
  runway is produced entirely by `ava/pipeline/{collector,curator,pack}.py` in
  this repo. One notable finding (not fixed here): its "real" eval mode
  fabricates scores (`harness/evals/perplexity.py:20-22`) despite README
  claims — treat harness numbers as mock until wired to a real forward pass.
- **ava-skills / cursor-agent-skills** are dev-time skill specs; they are NOT
  tokenized into training shards. The P2 `tool_use` diet slice comes from
  `ava/datagen/react_tools.py`. No file locking or shard contention exists in
  either repo. (Legacy `train_1b_deepspeed.py` references `skills.loader` but
  that trainer is superseded by `ava/train.py`.)
- **Live shard scan** (stratified, across phases, in-container): tokenizer
  vocab, token-id ranges (< 32000), idx/bin consistency and doc bounds all
  clean — the "malformed token block boundary" hypothesis is empirically dead.

## Fixes shipped (all behind regression tests; 258 tests green)

| Area | File | Change |
|---|---|---|
| CE transient (~2GB/µbatch) | `ava/jlosses.py` | chunked+checkpointed cross-entropy; proven bit-exact (loss diff 0.0, grad diff 0.0) |
| Retained logits across step | `ava/train.py` | detach metric snapshot, `del out/parts` inside accum loop |
| Allocator fragmentation | `docker-compose.yml` | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| P3+ deterministic OOM | `ava/train.py` | `MAX_MICRO_TOKENS=8192` cap in `micro_batch_for` (P2 unchanged; P3 mb=4, P4/5 mb=2) |
| Crash observability | `ava/train.py` | `trainer_crash` event logged to metrics before the process dies |
| Data treadmill (4x epochs) | `ava/data.py` | each present task type yielded exactly once per shard claim |
| Lease bugs | `manifest.py`, `flow.py`, `data.py` | per-stage lease (train 3600s wired from pipeline.yaml), 5-min renewal during consumption, tolerant release |
| Stranded shards | `manifest.py`, `data.py` | `tokens_ready` counts only claimable shards; `rescue_stranded()` resets attempts on PACKED (not FAILED) at sampler startup |
| Phase-transition leak | `ava/train.py` | `release_held()` at phase boundary |
| Mixture collapse | `ava/pipeline/collector.py` | RR state persists across iterations; rebuilt only when (phase, weights) change |
| Poison-row crash | `ava/data.py`, `curator.py` | load failures `fail()` the shard instead of killing the trainer; empty-train-split rows retired via `mark_deleted` |
| False stale alarms | `ava/pipeline_status.py` | cadence fallback across restarts, liveness from any trainer event, "recovering" mode, `restarts_window` (model_built count) |
| Half-life freeze | `multi_jspace_module.py` | decay clamp ceiling 0.99 → 0.9999 |
| Dead config | `configs/mini.yaml` | `compile: false` (torch.compile was never wired; `true` only skewed perf expectations) |

## Verification standard

- 258/258 tests pass (pre-existing httpx/starlette `TestClient(app=...)`
  incompatibility in `test_server_endpoints.py` reproduces identically on
  clean HEAD; unrelated).
- Chunked CE: loss and gradient bit-identical to the original on
  mb=8 × seq=1024 × vocab=32000.
- Collector RR: weights 3/2/1 over 18 draws now pick exactly 9/6/3 (was 18/0/0).
- Post-deploy success metric: trainer resumes and steps continuously; VRAM
  headroom > 1.5GB (was 370MB); no `trainer_crash` events at the prior ~1/4h
  rate; `tokens_ready` reflects claimable data; dashboard shows
  training/recovering correctly between metric events.

## Deliberately not changed (candidates for next hill-climb cycles)

- torch.compile wiring (perf, riskier on WSL2 — needs a canary run).
- gradient checkpointing at seq≥2048 (revisit with measured post-fix VRAM).
- Harness "real"-mode fabricated scores (eval integrity, separate PR).
- Backfilling the P2 mixture: shards already packed are 100% fineweb_edu; the
  collector fix applies to new collection only. Consider demand-boosting the
  missing task types for the remainder of P2.
- Host power ops: AC power + High Performance while training; consider
  disabling Modern Standby during long runs (`powercfg /change standby-timeout-ac 0`).
