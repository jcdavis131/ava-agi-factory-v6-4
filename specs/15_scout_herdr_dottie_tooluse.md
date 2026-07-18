# Spec 15 — Scout×Herdr, Dottie Tool-Use Curriculum, and the Arxiviq Assistant

Status: **active** · Author: foreman session · Date: 2026-07-17 · Supersedes: none

> One CLI for the harness (Scout ⨯ Herdr), one curriculum to teach Dottie/Ava
> to *use* tools, and one assistant surface on arxiviq.com — all wired so that
> **Telemetry** (everything an agent does is observable after the fact) and
> **Trust** (nothing an agent does escapes a declared, enforced capability
> boundary) are first-class, not bolted on.

This spec is the design authority for three tracks that ship together. Each
track lives in its own repo; the shared telemetry/trust vocabulary (§2) is the
seam that makes them one system.

```
   ┌─────────────────────────────────────────────────────────────────────┐
   │  Scout CLI (scout-cli / bigbang)   ── one control plane for tools    │
   │    └─ plugin: herdr  ──socket/CLI──▶  Herdr multiplexer (agent panes)│
   │         telemetry ▶ herdr/telemetry.jsonl   trust ▶ manifest+policy  │
   ├─────────────────────────────────────────────────────────────────────┤
   │  Ava/Dottie (ava-agi)                                                │
   │    ├─ datagen: tool_use curriculum  ──trains──▶  the model           │
   │    │     (plain-text ReAct; frozen-tokenizer-safe; dormant weight 0) │
   │    ├─ ava/trust.py   shared telemetry-event + tool-capability policy │
   │    └─ ava/assistant.py  server-side ReAct tool loop (Hermes-style)   │
   │         POST /assistant  +  GET /assistant page  +  status publisher │
   ├─────────────────────────────────────────────────────────────────────┤
   │  arxiviq.com (bluehenre / apps/sites/research)                       │
   │    └─ /assistant page  ──GitHub-raw poll──▶ assistant_status.json    │
   │         + BFF /api/assistant  ──(gated live)──▶ ava-agi /assistant   │
   └─────────────────────────────────────────────────────────────────────┘
```

---

## 0. Non-negotiable constraints (discovered, ground-truthed)

These are load-bearing. Violating any one breaks a running system.

1. **The tokenizer is frozen.** `ava/tokenizer.py::SPECIALS` pins ids 0–5
   (`<|pad|> <|bos|> <|eos|> <|endofdoc|> <|user|> <|assistant|>`). There is **no
   `<|tool|>` / `<|observation|>` token.** Adding one retrains + re-freezes the
   tokenizer and invalidates every packed shard (`Manifest.complete` rejects a
   mismatched `tokenizer_sha`). ⇒ **The tool-use curriculum MUST use the
   existing plain-text ReAct convention** — `Action: fn(args)` in an
   `<|assistant|>` turn, tool results returned as an `<|user|>` turn prefixed
   `Observation: `. No new specials, ever, in this track.

2. **A live `mini` training run is in flight** (`AVA_PRESET=mini`,
   `reports/metrics_mini.jsonl` active; prior crash-loop incident on record).
   ⇒ New curriculum sources enter `configs/sources.yaml` at **weight 0**
   (wired-but-dormant). No edits to `train.py`, `optim.py`, `model.py`,
   `tokenizer.py`. The operator flips weights on and rescales the phase to ~1.0
   deliberately, out of band.

3. **The ReAct parser contract is fixed** and shared across four surfaces:
   `AgenticOS/ava_bridge.py::_ACTION_RE = re.compile(r"Action:\s*([a-zA-Z_][a-zA-Z0-9_]*)\((.*)\)\s*$", re.M)`.
   Tool names must match `[a-zA-Z_][a-zA-Z0-9_]*`; one Action per assistant turn
   (parser uses `.search`, takes the first). Curriculum output, the assistant
   loop, `serve_engine` inference, and the eval harness must all agree with this
   regex. Every generated `Action:` line is asserted to parse in tests.

4. **There is no live network path from Vercel to the ava-agi box.** arxiviq.com
   reads one-way from `raw.githubusercontent.com/jcdavis131/ava-agi-factory-v6-4/main`
   (`STATUS.json`, `reports/dottie_live_status.json`, …). ⇒ The assistant's
   arxiviq surface is **telemetry-first, published as JSON** through the same
   GitHub-raw pipeline; live chat is an **opt-in, gated** path that only
   activates when a tunnel + token are configured. We build the CORS+auth
   scaffolding so it *can* go live, and we are honest in the UI about
   live-vs-demo.

5. **herdr may not be installed** (it is not, on the current win32 box; Windows
   support is preview-beta). ⇒ The Scout herdr plugin degrades gracefully:
   every command works in a `not-installed` / `offline` state, and the whole
   plugin + its tests run with no herdr binary present.

---

## 2. Shared telemetry & trust model (the seam)

Two paved paths, one on each side of the repo boundary, speaking the same
vocabulary.

### 2.1 Telemetry event schema

A telemetry event is a flat JSON object, one per line (JSONL), append-only:

```jsonc
{
  "ts":        "2026-07-17T18:04:11.221Z",  // ISO-8601 UTC, ms precision
  "surface":   "scout.herdr" | "ava.assistant",
  "actor":     "<session/agent id>",         // who acted
  "action":    "tool_call" | "agent_status" | "gate_denied" | "loop_step" | ...,
  "target":    "<tool name / pane id / endpoint>",
  "args":      { ... },                       // secret-scrubbed (keys w/ "secret"|"key"|"token" redacted)
  "status":    "ok" | "error" | "denied" | "empty",
  "duration_ms": 12,
  "meta":      { ... }                        // free-form (e.g. agent_status transition from→to)
}
```

- **Scout side** reuses `bigbang/core/audit.py::log_event` (already writes
  `~/.local/share/bigbang/audit.jsonl`) and additionally writes a herdr-scoped
  ledger `~/.local/share/bigbang/herdr/telemetry.jsonl` shaped exactly as above.
- **Ava side** is `ava/trust.py::emit_event(...)` → `runs/assistant_audit.jsonl`,
  same shape. A rollup of the last N events is published into
  `reports/assistant_status.json` for arxiviq.com.

The rule: **no agent action without a telemetry line.** A tool call that isn't
logged is a bug, not a feature.

### 2.2 Trust / capability policy

Trust = a declared capability boundary that is actually *enforced* before the
action, plus a confirm-gate on irreversible actions.

- **Scout side**: the herdr `manifest.yaml` declares a default-deny capability
  set (network to the herdr socket only; no fs writes beyond the telemetry
  ledger; no secrets). The plugin **calls `enforce_or_raise` before** any
  socket/subprocess action — actually wiring the policy engine that
  `docs/SECURITY.md` currently calls a stub. Destructive verbs
  (`session stop/delete`, `pane run`, `pane kill`) require `--yes` or a TTY
  confirm; in `--json` mode they refuse without `--yes` and emit a
  `gate_denied` telemetry event.
- **Ava side**: `ava/trust.py` holds the **tool capability table** — each tool
  the assistant may call is declared with `{read_only: bool, side_effects: bool,
  sandbox_root: path|None, max_output_bytes: int}`. The loop consults this table
  before dispatching a tool; an undeclared tool, or a path-traversal attempt
  outside `sandbox_root`, is denied (telemetry `gate_denied`) and the model is
  told so via the Observation, teaching it the refusal path at inference time
  the same way the curriculum teaches it at train time.

Symmetry is the point: the model is *trained* (Track B) to respect grounding &
refusal, and the runtime (Track C) *enforces* the same boundaries, and both
*emit the same telemetry shape* (Track A shares the vocabulary). Trust is thus
end-to-end, not per-component.

---

## 3. Track A — Scout ⨯ Herdr (`scout-cli`)

> **Reconciled with synced reality (2026-07-17).** After pulling the latest
> scout-cli, the vision is already largely built: the `herd` plugin (a
> Herdr-*inspired* JSON session ledger — Scout "steals Herdr's orchestration
> model without becoming a multiplexer") and the `planes` plugin (the Judgment
> cockpit, "differentiated above Herdr") exist and are tested. A fresh `herdr`
> plugin would duplicate them. So Track A **extends `herd`** to fill the three
> gaps that map exactly onto this project's themes, completing FOUNDATION Waves
> F1/F2 in the process. Nothing is rebuilt.

**Gap → fill (all in `bigbang/plugins/herd/`):**

1. **Herdr integration** (the literal "build herdr.dev into scout-cli"). `herd`
   only did `shutil.which("herdr")` + static pairing prose; `herdr_pane` was
   always `None`. New **`herdr_bridge.py`** is a real, **offline-safe** bridge:
   `herdr_path()`, `bridge_status()`, `list_agents()` (normalizes Herdr's
   versioned agent/pane JSON to `[{pane_id, agent, state}]`), `schema()`
   (`herdr api schema --json`), all returning structured results when herdr is
   absent (never raise, short timeouts). Wired into an enhanced `herd herdr`
   (live status), a new `herd bridge` command, and `herd attach <s> --pane <id>`
   → `store.attach_pane` populates the previously-dead `herdr_pane` field.

2. **Trust** (FOUNDATION F1). `herd start` spawned arbitrary argv ungated. New
   **`spawn_guard.py`** refuses destructive commands (`rm -rf /`, `mkfs`, `dd
   of=/dev/…`, fork bombs, `curl|sh`, force-push, …) unless `--allow-risky`, and
   records the refusal as telemetry; `start`/`attach` also call
   `enforce_or_raise(manifest, "fs_write", …)` at the ledger write.

3. **Telemetry** (FOUNDATION F2 + the arxiviq seam). New **`events.py`** is a
   per-session append-only JSONL stream you own
   (`~/.local/share/bigbang/herd/events.jsonl`), surfaced by `herd events`
   (rollup or `--tail`). `herd export --sink <dir>` is **opt-in local export
   where you name the sink** — the only sanctioned seam per
   `DIFFERENTIATION.md` (telemetry is a Trust boundary, never phoned home);
   pointed at the factory `reports/` dir it rides that repo's git-push daemon
   onto the `main` branch arxiviq reads. Event/redaction shape matches
   `ava/trust.py` and `dottie/telemetry.py` so the surfaces read as one system.

**Routing:** the existing `herd` block in `_heuristic_route` gains
`attach`/`pane`/`bridge`/`events`/`telemetry`/`export` keywords (confidence
unchanged at 0.94). **Manifest** bumped to 0.8.0.

**Tests** (`tests/test_herd_bridge.py`, offline-safe): bridge normalization +
offline structure; spawn-guard deny/allow; events record/tail/rollup/export +
secret scrubbing; `store.attach_pane` populates `herdr_pane`; `--json` CLI
contract for `herd bridge`/`herd events` (via `sys.executable`, Windows-safe).

Acceptance: the new surface works with **no herdr binary present**; every new
command emits valid JSON; destructive `start` is refused without `--allow-risky`;
a telemetry event is written per lifecycle transition. (Pre-existing suite
failures under this venv are unrelated — `test_herd.py` etc. hardcode a
`python3` shim that lacks `typer`; the new tests use `sys.executable`.)

---

## 4. Track B — Dottie/Ava tool-use curriculum (`ava-agi`)

**Goal:** teach the model to *use tools well* — select among many, chain
multi-step, recover from errors, and refuse/ground when appropriate — in the
frozen-tokenizer-safe plain-text ReAct format, without disturbing the live run.

**Curriculum ladder** (levels → the gap they close vs. today's `react_tools`):

| Level | Name              | Teaches                                              | Phase(s) |
|-------|-------------------|-----------------------------------------------------|----------|
| L0    | grounded single   | one call, answer from the Observation (keep strength)| 2, 3     |
| L1    | multi-step chain  | ≥2–5 calls, later args from earlier Observations     | 3        |
| L2    | error & recovery  | bad-arg/timeout/empty → corrected retry or fallback  | 3, 4     |
| L3    | tool selection    | choose the right tool from a listed catalog of many  | 3, 4     |
| L4    | negative / refuse | answer directly w/o a tool; refuse when none fits    | 5        |

New generator `ava/datagen/tool_curriculum.py`: class `ToolUseGenerator`,
`name = "tool_use"`, `phases = (2, 3, 4, 5)`. Each doc:
- Optionally opens with a **tool catalog** block (an `<|user|>` framing that
  lists N available tools with `name(sig) — one-line purpose`) so the model
  learns selection-among-many — but always parseable, catalog is prose not a new
  special.
- Uses `dialogue()`-style `<|user|>`/`<|assistant|>` turns; every `Action:` line
  matches the shared `_ACTION_RE`; Observations return as `<|user|>` turns.
- Numbers/results are **computed in Python from `self.rng`**, never templated —
  the tool's Observation is ground truth and the final answer must match it.
- A **large, varied tool/entity vocabulary** (dozens of tool names, files,
  entities) to prevent memorization; tool names stay `[a-zA-Z_][a-zA-Z0-9_]*`.
- Error family injects a realistic failure Observation (`Error: unknown argument
  'foo'`, `Error: timeout`, `(no matches)`) and a corrected next Action or a
  grounded give-up — teaching recovery and anti-fabrication together.
- Negative family: the tool catalog is present but the right move is either a
  direct answer (no Action) or an explicit "none of these tools can do X, here's
  what I can do instead" refusal.

**Registration:**
- `ava/datagen/__init__.py`: import `ToolUseGenerator`, add to `GENERATORS` &
  `__all__`.
- `configs/sources.yaml`: new source `synth_tool_use`, `kind: synthetic`,
  `generator: tool_use`, `phases: [2,3,4,5]`, **`weight` all 0.0** (dormant),
  `task_type: deliberate`, `filters: {min_chars: 1}`, `license: synthetic`,
  `gated: false`. A comment documents the flip-on + rescale-to-1.0 procedure.

**Tests** (`tests/test_tool_curriculum.py` + add class to `ALL_GENERATORS` in
`tests/test_datagen.py`): free determinism/schema/phase checks; task_type
assertion; **trajectory verification** — re-parse every `Action:` (assert it
matches `_ACTION_RE`), independently recompute each Observation and the final
answer, assert consistency; grounding test (empty/error Observation ⇒ no
fabrication); negative-family test (refusal docs contain no `Action:` OR contain
an explicit no-tool statement); ava_bridge parse-compat (skip if AgenticOS
absent). Determinism is byte-level (`self.rng` only).

Acceptance: `pytest tests/test_tool_curriculum.py tests/test_datagen.py` green;
the live `mini` run is provably untouched (weights 0, no train/tokenizer edits);
every `Action:` parses with the production regex.

---

## 5. Track C — the Arxiviq assistant (Hermes/OpenClaw-style)

**Goal:** a persona'd, tool-using assistant ("Dottie") whose every step is
telemetered and trust-gated, reachable as a backend endpoint and surfaced on
arxiviq.com.

### 5.1 Backend (`ava-agi`)

`ava/assistant.py` — a **server-side ReAct tool loop** over
`ServeEngine.generate()`:
- **Persona**: a leading `<|user|>` system-framing preamble (no system special
  exists) establishing the Dottie assistant voice + the tool contract.
- **Tool executor**: a whitelist of **read-only, sandboxed** tools whose names
  mirror the training distribution (`get_clock`, calculator `add/subtract/
  multiply/sum`, `repo_grep`, `repo_read_file`, `list_dir`, `pipeline_status`,
  `ecosystem_status`, `skill_search`). Each is declared in `ava/trust.py`'s
  capability table; `repo_*`/`list_dir` are sandboxed to a read-only allowlisted
  root with path-traversal rejection and output caps.
- **Loop**: prompt → generate (small `max_tokens`, CPU-quadratic aware) → parse
  first `Action:` via the shared regex → dispatch through the trust gate →
  append `Observation:` as a `<|user|>` turn → repeat until a final answer, a
  step budget, or a refusal. Every iteration emits telemetry.
- Degrades gracefully when `AVA_SKIP_ENGINE_BOOT=1` (engine absent): returns a
  503-style structured payload, exactly like `/chat`.

`server.py` additions (additive, low-risk):
- `POST /assistant` — `AssistantReq{messages, max_steps, temperature}` → the loop
  → `{content, steps:[{thought,action,observation,gate}], tokens, latency_ms}`.
- `GET /assistant` — `ava/assistant_html.py::ASSISTANT_HTML` page following the
  established self-contained `*_html.py` convention; links from `/` index.
- **Opt-in CORS + bearer auth**: `CORSMiddleware` with an allowlist from
  `AVA_ASSISTANT_CORS` (default off); a bearer check against `AVA_ASSISTANT_TOKEN`
  applied **only** to `/assistant` (default off → open locally, closed when set).
  This is new ground (spec 07 lists auth as out of scope) justified by "Trust
  paramount"; it must not affect existing routes or the running dashboard.
- **Status publisher** `ava/assistant_status.py::collect_assistant_status()` →
  served at `GET /assistant/status` and written to
  `reports/assistant_status.json` (capabilities, tool catalog, trust policy,
  telemetry rollup, a demo transcript) for arxiviq.com to poll via GitHub-raw.

### 5.2 Frontend (`bluehenre / apps/sites/research`)

- `app/assistant/page.tsx` — a Fleet-styled ("@synthaembed/ui-fleet") page:
  a **Trust & Telemetry** panel (reads published `assistant_status.json` from
  GitHub-raw, same pattern as `DottieControlPlane`) showing the tool catalog,
  capability boundaries, and recent telemetry; plus a chat box.
- `app/api/assistant/route.ts` — a server-only BFF (the `core-api.ts` bearer
  pattern): forwards to the ava-agi `/assistant` endpoint **iff**
  `AVA_ASSISTANT_BASE_URL` + `AVA_ASSISTANT_TOKEN` are set (live mode), else
  serves the published demo transcript (demo mode). The UI states which mode is
  active. Additive, self-contained files that do **not** depend on the
  origin/main-only Dottie files (local checkout is stale); a reconcile note is
  in the PR.

Acceptance: backend `pytest tests/test_assistant.py` green (engine-absent path,
loop over a stubbed engine, trust gate denies traversal/undeclared tool,
telemetry line per step); `assistant_status.json` validates; the Next.js page
typechecks and renders in demo mode with no live backend.

---

## 6. Rollout & gates (outward-facing actions are gated)

Built, tested, and committed **on feature branches in each repo**. The
following are **outward-facing / hard-to-reverse and are NOT done without
explicit confirmation**, presented as the final gated step:
1. `git push` in any of the three repos.
2. Publishing `assistant_status.json` to `ava-agi-factory-v6-4` (arxiviq reads
   it live).
3. Vercel deploy of `arxiv-exam-app` (auto-triggers on push to `bluehen` main).
4. Flipping the `synth_tool_use` weights > 0 in a live curriculum run.

Everything up to those gates is reversible local work.

## 7. Test matrix

| Repo       | Command                                             | Gate                          |
|------------|-----------------------------------------------------|-------------------------------|
| scout-cli  | `pytest` (from repo root)                            | green, no herdr binary        |
| ava-agi    | `pytest tests/test_tool_curriculum.py tests/test_datagen.py` | green, determinism holds |
| ava-agi    | `pytest tests/test_assistant.py`                    | green, engine-absent safe     |
| bluehenre  | `pnpm --filter @synthaembed/research typecheck`     | typechecks, renders demo mode |
