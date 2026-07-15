# Ava Ecosystem Review Prompt — LLMVM / Metamate Advanced Auto Architecture

> **Solo personal project, no connection to employer, built with public/free-tier only**

## Role
You are the Principal Architect for Ava AGI Factory v6.4 — a 1B-parameter model with 4 J-Spaces (S1 Fast hl=8, S2 Slow hl=300, Critic hl=30, Planner hl=150) plus Router/veto, YaRN RoPE 10k→1M, WSD 736k steps, 6-phase logic-first curriculum, and a living ecosystem of continuous pipelines.

Your job: audit the current ecosystem and propose an upgrade inspired by Metamate Advanced Auto (Unified Auto / LLMVM).

## Context: Ava v6.4 As-Is

**Factory layout:**
- `train_1b_deepspeed.py` — WSD + YaRN + J-Space losses (reportability, broadcast 20%, selectivity, modulation, inter-MI cos 0.45)
- `branch_anneal.py` — forks stable ckpt at 736k into code/math/chat
- `eval_branch_harness.py` + `eval_frontier_rubric.py` — 5 canonical J-space tests + 11-cat rubric via Ollama qwen3:32b judge
- `streaming_data.py` / `data_builder_agent.py` / `trainer_agent.py` / `prefect_flows.py` — data gather + manifest concurrency
- `server.py` — J-Lens viewer (audit vs research mode), WebSocket layer stream
- `docs/HARNESS_SKILL_INTEGRATION.md` — 8 starter skills: jspace-inspector, openwiki-sync, logic-prover, code-bench, safety-scanner, memory-router, eval-harness-runner, family-brain-wiki
- Continuous crons: ava-data-gather 4h interval (fast md5 10M shards), ava-dataset-discovery daily (58 HF candidates), ava-eval-distill daily (branch harness mock PASS cap_score 0.983)

**Current execution model:**
- JSON tool-calling loop for agents (trainer_agent, data_builder_agent)
- One-shot shell commands (no interactive terminal)
- Fixed tool sets at conversation start
- Schema tax: every tool hand-written JSON schema
- Context bloat: all intermediate results piled into LLM context

Read these before answering:
- `README.md` — architecture overview
- `ORCHESTRATION.md` — Foreman / Sonnet / Opus dispatch protocol
- `docs/HARNESS_SKILL_INTEGRATION.md` — skill / OpenWiki bridge
- `specs/08_alienware_runbook.md` — local RTX 4080/4090 constraints
- `ava/ecosystem_status.py` + `ava/pipeline_status.py` — live status

## Inspiration: Metamate Advanced Auto (What Changes)

**Core thesis:** Give the LLM a Python runtime, not a JSON tool-calling loop.

> Most frameworks: LLM emits JSON → framework runs 1 tool → LLM decides next. Repeat.
> You can't write a for-loop in JSON. You can't handle errors. You can't compose.
> Unified Auto: LLM writes and executes real Python in a persistent notebook. Every tool from Confucius/DevMate/Metamate is an async function. Loops, branches, asyncio.gather(), try/except all in one execution.

**Why it matters for Ava:**
- **Round-trip overhead:** 15+ LLM calls → 1 Python cell. Ava data gather currently: discover → dedup → filter → shard → manifest → validate = 6 round-trips. Should be 1.
- **No composition:** Can't do "search 3 patterns in parallel, filter deprecated, summarize top 5" in JSON. In Python you just write it.
- **Schema tax:** 3-param function = 16 lines JSON boilerplate. In LLMVM, signature is schema, docstring is description. Tax = 0.
- **Context blowup:** Bento kernel sandboxes execution, only final output enters context. Not intermediate 10MB logs.

**Key innovations to steal:**

1. **UnifiedCodeExtension — Code is orchestration:**
   ```python
   # What Ava should be able to do in ONE cell:
   docs = await asyncio.gather(
     search_code("S2 hl=300 broadcast"),
     search_logs("loss spike 3x median"),
     openwiki_search("YaRN factors")
   )
   filtered = [d for d in docs if "deprecated" not in d.path]
   summary = await summarize_top(filtered[:5])
   ```
   Nobody designed a tool for this. Agent wrote a program.

2. **Self-Modification — Agent rewrites its own tools mid-session:**
   Because runtime is persistent Python namespace, agent can:
   ```python
   def cached_download(url): # override existing
       if url in cache: return cache[url]
       data = await original_download(url)
       cache[url] = data
       return data
   
   download = cached_download # every subsequent call uses cache
   ```
   The agent added caching without a "caching tool". This is emergent.

   Other examples agent built for itself when asked to audit codebase:
   - caching layer for HF downloads
   - code smell analyzer for J-Space losses
   - diff history searcher for training logs
   - WSD phase transition detector

3. **TMUX Extension — Real terminal interaction:**
   - Persistent tmux sessions, send keystrokes including Ctrl+C, capture pane output
   - Unlocks: debugging failing test by iteratively running, inspecting partial output, adjusting
   - For Ava: debugging `train_1b_deepspeed.py` CUDA OOM, watching `rocm-smi`, interactive `ipdb`
   - Strategy as code: "run test, if fails check log for X, if X try Y" across multiple panes

4. **Skills and Skillbooks — Knowledge + code + notebooks, versioned:**
   - Skill = complete package, not function signature
   - Skillbooks: anyone can create/share without diff, edit in Bento Notebook, use immediately, private/team/everyone visibility, latest vs published versions
   - Fastest creation: get workflow working in conversation, tell agent "save as skillbook"
   - Example for Ava: "create a skillbook for diagnosing WSD phase transition loss spikes from this notebook" → 1 hour → whole team uses it
   - Map to Ava's 8 skills: make each a Skillbook with docs + Python + bootstrap notebook

5. **1000+ Tools, No Context Blowup:**
   - Defer-load: only lightweight metadata upfront (~100 tokens per skill)
   - Agent searches and loads detailed definitions on-demand
   - Bidirectional context control: loading in via tool search, compacting out via agent-managed memory (remove failed attempts, replace verbose with summary)
   - Resilience: multiple paths to same goal (code search → internal search → wiki → web), agent recovers by combining tools unexpectedly

## Your Task: Audit + Redesign

### 1. Audit Current Bottlenecks (be brutal, measure)
For each pipeline in `docs/CONTINUOUS_PIPELINES.md` and `TODOS.md`:
- Where are we paying JSON tax? List every tool that needs hand-written schema.
- Where are sequential round-trips that should be one Python cell? Count tokens wasted.
- Where is context bloat? Show `logs/builder.log` growing, `streaming_data.py` intermediate results piling up.
- Where is orchestration stateless? (bash tool stateless between invocations, state only on disk)
- Grade: Could this be emergent capability vs designed tool? e.g., parsing training logs with regex vs needing a `parse_logs` tool.

### 2. Apply LLMVM Lens — Propose Ava v6.5 LLMVM Layer
Design:

**A. Runtime:**
- Persistent notebook: Python kernel (Bento-like sandbox) vs Jupyter? Local on Alienware RTX 4080 box, free-tier only, no work systems.
- Tool exposure: every function in `ava/` as async def. Signature = schema. How to expose `train_1b_deepspeed.py`, `eval_branch_harness.py`, `server.py` as async?
- Sandboxing: execution pauses for host calls (HF download, CUDA), resumes, only final output → context.

**B. Self-Modification Protocol:**
- Where should Ava self-modify? Propose 3 high-value overrides:
  1. Caching wrapper for `scripts/hf_uploader.py` + `streaming_data.py`
  2. Code smell analyzer that watches `multi_jspace_module.py` for broadcast leaking future→past
  3. Training watchdog that auto-adjusts `j_weight` on loss spike >3x median (currently manual in ORCHESTRATION.md)
- Define safety: audit log for overrides, rollback, `ENABLE_JSPACE_WRITE` gate analogy.

**C. TMUX Layer:**
- Design 3 tmux sessions for Ava: `train`, `eval`, `data`
- Show interactive debug flow: "run `python eval_branch_harness.py --mode real`, if OOM check `nvidia-smi`, if spike check last 100 lines of `logs/builder.log`"
- How combined with Python: write debugging strategy as code that drives tmux panes.

**D. Skillbooks v2:**
- Convert 8 starter skills to Skillbooks: each needs docs + code + bootstrap notebook + versioning
- Propose 3 new Skillbooks unlocked by LLMVM: `diagnose-wsd-spike`, `audit-jspace-leak`, `discover-dataset-fast`
- Flow: get working in conversation → `save as skillbook` → version latest/published
- Storage: `ava/skills/` local, free, public-tier only, with `AGENTS.md` block for OpenWiki.

**E. Context Management:**
- Defer-load: design metadata ~100 tokens per Ava tool. Which 1000+ tools? (Confucius equivalent = HF datasets, GitHub, Papers, Ollama judges, W&B)
- Compaction: agent actively removes failed attempts, summarizes verbose `frontier_eval_results.json`
- Measure: current eval harness uses ~8k tokens intermediate, target <1k after compaction.

### 3. What to Try — Concrete POCs (Free-to-build/host/serve, public pip only)

Pick 2 POCs to one-shot, like Metamate one-shotted nest app:

1. **Code & Debugging POC:** Persistent notebook that audits `multi_jspace_module.py` for 3 properties that were "passing" for years (no causal mask, broadcast future→past, verbalizable_mass constant 0.06). Build custom toolkit (caching, smell analyzer, diff search) to do it in one cell.

2. **Data Analytics POC:** Describe analysis in plain English: "pull Presto? no — pull local manifest, show WSD loss curve, overlay RoPE transitions 10k→1M". Agent writes Python SQL/pandas/matplotlib, executes in notebook, returns chart. Iterative, not starting over each turn.

3. **Research POC:** Multi-source investigation: code search + `~/.openwiki/wiki/` + `specs/` + HF dataset cards + arXiv. Cross-reference and synthesize, not dump links.

Each POC must show: before (15+ JSON round-trips) vs after (1 Python execution), tokens saved, emergent tool invented.

### 4. Comparison — Claude Code vs Ava LLMVM

Map same axes as Metamate note, but for Ava:

| Dimension | Current Ava (JSON loop) | Ava LLMVM (Python runtime) |
|-----------|-------------------------|----------------------------|
| Parallel execution | Sequential tool calls | `asyncio.gather()` 10 ops in one cell |
| Persistent state | Stateless bash, state on disk | Variables persist across cells |
| Self-modification | Fixed tools at start | Define/override function mid-session |
| Interactive terminal | One-shot `exec` | TMUX with keystroke control |
| Composability | 2 tools = 2 inference trips | Loops/conditionals/error handling in one expression |

### 5. Deliverable

Write `docs/LLMVM_REDESIGN_v6.5.md` with:

- **TL;DR** — 3-bullet why Python runtime not JSON loop for Ava, with one concrete example from `logs/builder.log`
- **Audit Table** — 5 current bottlenecks, token cost, failure mode
- **Architecture Diagram** — text Mermaid: Bento kernel ↔ orchestrator ↔ host calls (HF, CUDA, OpenWiki) ↔ TMUX sessions ↔ Skillbooks
- **Self-Modification Examples** — 2 code snippets where Ava overrides its own download/eval function mid-session (with caching layer)
- **TMUX Debug Recipe** — copy-paste interactive debug flow for training stall
- **Skillbook Migration Plan** — 8 skills → Skillbooks, plus 3 new ones, with `openwiki code --init` step
- **1000 Tools Without Blowup** — defer-load design, ~100 tokens metadata list
- **POC Plan** — 2 POCs with before/after round-trip counts, success criteria, free-tier implementation (ONNX WASM, ExecuTorch local, HF ZeroGPU, R2/Workers/Supabase if needed)
- **Migration Safety** — audit log, rollback, `ENABLE_JSPACE_WRITE` analog for self-modification, no work IP, local-first
- **Footer:** Solo personal project, no connection to employer, built with public/free-tier only

Keep it short/concise — one tight update with links to detailed docs. Design quality gate: Sunni Davis SCAD critique — but for architecture docs: clear hierarchy, 18px readability, code blocks runnable.

## Constraints

- HOME-LIFE ONLY — zero use of Meta/Work data, code, systems, IP, models, pipelines, docs, resources — past/present/future. Public pip + free-tier only (R2/Workers/Supabase/HF ZeroGPU/ONNX WASM).
- No Vercel/Bluehen references.
- Manual bridge only if user provides sanitized non-confidential summary.
- Every artifact footer includes: "Solo personal project, no connection to employer, built with public/free-tier only"
- Free-to-build/host/serve — Ollama before paid APIs (qwen3:32b default for judges), WANDB offline if needed.

## Prompt Starter to Use in Ava

```
You are Ava's self-modifying architect running in LLMVM.

Goal: Audit ava-agi-factory-v6-4 codebase like Metamate Advanced Auto audited itself.

1. In ONE Python cell, search three patterns in parallel:
   - "BEGIN DEFERRED" vs "BEGIN IMMEDIATE" in manifest concurrency
   - "causal_mask" absence in attention
   - "verbalizable_mass = 0.06" constant

2. If any found, define a new tool `audit_jspace_leak()` that:
   - caches results
   - analyzes code smell (future leaking to past in broadcast)
   - searches diff history for when leak introduced

3. Override your own file reader to add caching and use it immediately.

4. Show TMUX debug plan for training loss spike >3x median.

5. Save working workflow as Skillbook `audit-jspace-leak`.

Only final summary enters context. Self-modification is expected, not exceptional.
```

Execute it and report measured values — don't claim test passed unless you saw it pass. Negative control like downgrading BEGIN IMMEDIATE → DEFERRED must fail.
