# ORCHESTRATION — Foreman / Worker-Bee Protocol
> How the execution session (foreman) dispatches, verifies, and tracks worker agents to run [`PLAN.md`](PLAN.md). Task states live in [`TODOS.md`](TODOS.md).

## Roles

- **Foreman** (the interactive session, highest-capability model): decomposes work, dispatches workers, **runs every acceptance command itself before marking done**, updates `TODOS.md`, commits after each verified task, monitors long-running training, makes GO/NO-GO calls at gates.
- **Worker bees** (subagents, chosen per task tier):
  - 🟦 **Sonnet tier** — mechanical, fully-specified tasks: scaffolding, data generators, tokenizer, packing, bench script, HTML report, Docker, docs. Cheap, parallel, disposable.
  - 🟪 **Opus tier** — correctness-critical tasks: model bug fixes (D1/D2), trainer + J-losses (F1/F2), eval harness (I1), serve engine (J1). Fewer, deeper.
- **User** — executes `specs/08_alienware_runbook.md` on the GPU machine (P11) and makes the mini→base1b GO/NO-GO decision.

## Dispatch loop (per task)

1. **Select** the next unblocked task(s) from `TODOS.md` (respect `deps:`; maximize parallel lanes — see graph in PLAN.md §4).
2. **Dispatch** one worker per task. The worker prompt MUST contain: the task ID, the full text of (or pointer to) its `specs/*.md` section, the repo path, the acceptance command(s), and the instruction *"do not commit; do not touch files outside your deliverable list; report the exact commands you ran and their output."*
3. **Mark** the task `dispatched` in `TODOS.md`.
4. **Verify** on worker return: foreman runs the acceptance command(s) from the spec **itself** (never trusts the worker's claim). Green → `done`, check the box, append Foreman-log row. Red → return the failure output to the *same* worker (continue its context) for one repair round; second failure → `blocked(reason)` and escalate tier (Sonnet→Opus) or split the task.
5. **Commit** after every verified task or coherent batch: `git add -A && git commit -m "<task-ids>: <what>"`. Push at phase boundaries.
6. Parallel dispatch: independent tasks are sent as one batch of concurrent workers (B1–B4 together; D1/D2 alongside B*).

## Worker rules (include verbatim in every worker prompt)

- Implement exactly the deliverable files listed in your spec section; touching blueprint files is forbidden unless the spec explicitly says "surgical in-place fix".
- No network calls in produced code (HF hub and wandb are blocked in this container). PyPI installs only via `scripts/setup_env.sh`.
- Determinism: any randomness must be seeded from the config/CLI; same seed ⇒ byte-identical output.
- Every deliverable ships with its tests; your task is not done until the spec's acceptance command passes locally.
- Return: files created/modified, commands run + outputs, deviations from spec (with reason), open concerns. Raw facts, no cheerleading.

## Long-run supervision (P6 nano train; same pattern for GPU runs)

- Launch: `nohup python -m ava.train --preset nano --run runs/base > runs/base/train.log 2>&1 &`
- Poll `runs/base/metrics.jsonl` on a timer (scheduled wake-ups, not busy-waiting): check ① new lines appearing, ② smoothed `lm_loss` trending down, ③ no `NaN/inf`, ④ `tok_s` ≥ 60% of bench, ⑤ RSS < 12 GB.
- Crash/stall → relaunch with `--resume` (bit-exact by design; verified by test F2). Loss spike >3× trailing median → pause, inspect phase transition (RoPE/seq-len change is the usual suspect), decide: resume / rollback to last ckpt with lower j_weight.
- Milestones: log stable-ckpt save (step 3369), phase boundaries, and ETA to `TODOS.md` Foreman log.

## Escalation & decision gates

| Gate | Decision rule |
|---|---|
| P5 bench | projected base run > 12 h → switch preset to `nano_quick` (15M tokens), note in log |
| P6 health | loss non-decreasing over 500 steps → halt, dispatch Opus debug worker with metrics excerpt |
| P7 evals | canonical tests may legitimately FAIL at 14M — report measured values; only *pipeline* errors block |
| P10 live | any `smoke_live.sh` failure blocks deploy; fix-forward via J1 worker |
| P11 mini → base1b | user decides on mini eval report: J-losses learning (hl_est → targets, routing differentiates task_types) + probes clearly above nano |

## Executable build workflow

`.claude/workflows/ava-build.js` encodes P0–P5 as a Workflow script (fan-out with acceptance-gated stages) for one-command dispatch: run it via the Workflow tool with `{scriptPath: ".claude/workflows/ava-build.js"}`. P6+ (training, deploy) stays foreman-driven — long background processes and live servers don't belong inside workflow agents.

## Model-tier map (summary)

| Tasks | Tier | Rationale |
|---|---|---|
| A1–A3, B1–B5, C1, E1, G1, J2–J4, K1, L2, M1 | 🟦 Sonnet | fully specified, mechanical, testable |
| D1, D2, F1, F2, I1, J1 | 🟪 Opus | correctness-critical: silent bugs here poison everything downstream |
| G2, H1, H2, I2, L1, L3, M2 | 👷 foreman/user | verification gates, long-run ops, live deploy |
