# CodeAct / LLM-VM plan — code as the model's action substrate (spec 13)

Date: 2026-07-17
Source: MAI-Thinking-1 agentic SWE + tool-use findings → `docs/RL_INTEGRATION.md`
Contract: `specs/13_codeact.md` (T13C.1–T13C.6)
Builds on: `specs/12_rl_training.md` (CodeAct is an agentic mode of the GRPO loop, not a parallel system)
Status: **sandbox + datagen halves unblocked now; RL halves inherit spec 12's T9.3/T9.5 block**

## Objective

Make Ava *think in code*: the model's actions are executable Python run in a persistent, sandboxed
LLM-VM with tools bound as callables; observations are real stdout/return values. Turn "leverage
tools to execute workflows we care about" into a trainable, verifiable objective instead of narrated
ReAct text that never runs.

## Phase order (do not skip)

1. **Sandbox (T13C.1) — ✅ DONE (2026-07-17)** — `ava/rl/codeact_sandbox.py`, multi-turn persistent
   namespace via a long-lived worker subprocess, per-step wall cap (setsid+killpg), POSIX resource
   caps, guarded open/blocked socket+fork, importable-or-source tools with call accounting,
   deterministic replay. `tests/test_codeact_sandbox.py` 14/14. Extends `code_gen.run_sandboxed`.
2. **Datagen (T13C.2, GPU-free)** — `ava/datagen/codeact.py`, executable trajectories with
   Python-computed answers, grounding-over-syntax bias from `react_tools.py`.
3. **Eval (T13C.3, GPU-free plumbing)** — harness CodeAct eval, exec-verified success rate,
   anti-mock; feeds `test_no_mock.py`.
4. **RL terms (T13C.4, gated on T9.3/T9.5)** — add `R_exec`/`R_codeuse` to spec 12's `rl_return`;
   reuse the discipline system + difficulty-scaled length penalty unchanged.
5. **Consolidate + serve (T13C.5)** — CodeAct traces into MOPD; `ServeEngine` code-act loop with
   FINAL-only user output + trace capture for memory-mint.
6. **EG gate (T13C.6)** — promote only on a 2-rung EG win vs the non-CodeAct agentic baseline.

## Gates

| Gate | Math | Target |
|------|------|--------|
| G1 isolation | fork bomb / socket open / out-of-scratch write | killed/blocked + reported, host unharmed |
| G2 determinism | replay (seed, tools, program) | byte-identical Observations |
| G3 datagen honesty | re-exec emitted trajectories | 100% reach labeled answer; no answer leakage |
| G4 eval sensitivity | break a tool binding | success rate drops measurably |
| G5 no narrated-code hack | non-executing code vs running code | penalized; `R_exec` secondary to `R_task` |
| G6 promotion | `eg_trend(nano, mini)` vs non-CodeAct agentic baseline | `promote` |

## Open questions (log answers here)

- Tool-binding surface: start with `react_tools` tools (calculator, get_clock, read/cite) — which
  subset is dense enough to train on without becoming a memorized API?
- Step cap vs task difficulty: a fixed cap penalizes hard multi-step tasks; scale the cap with the
  same historical pass-rate difficulty signal as the length penalty?
- Redundant-call detection: reuse scout-cli RFT `reward_components.redundant_steps` semantics over
  sandbox tool-call logs, or define fresh? Keep the definition versioned (spec 12 naming guard).
- Safety set for the interpreter: which "refuse to run this" cases belong in the Critic-scoped
  set before CodeAct serving is exposed?
