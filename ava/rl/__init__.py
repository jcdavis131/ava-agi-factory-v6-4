# Solo personal project, no connection to employer, built with public/free-tier only
"""ava.rl — reinforcement-learning + agentic-execution substrate (specs 12 & 13).

GPU-free, tested building blocks landed:
  * `codeact_sandbox`       — the LLM-VM (T13C.1): subprocess-isolated persistent-namespace interpreter.
  * `codeact_rewards`       — R_exec / R_codeuse / R_len / codeact_return (T13C.4 reward terms).
  * `grpo`                  — GRPO-lite discipline mechanics (T12R.2 / T13C.4): group advantages,
                              entropy-thermostat controller, outer ratio clip, trace-bank recovery.
  * `codeact_loop`          — pluggable-policy decode/serving loop (T13C.5): emit→sandbox→observe→FINAL.
  * `codeact_consolidation` — MOPD trace-pool prep (T13C.5): verified-only, stratified.
  * `codeact_eg_gate`       — EG-gated rollout (T13C.6): success→error transform + eg_trend verdict.

Gated on branch fine-tunes (T9.3/T9.5) + GPU (BLOCKED_NO_GPU): the torch GRPO optimizer step
(`grpo.GRPOOptimizerStep`), the real-model policy (`codeact_loop.ModelPolicy`), and the MOPD
distillation run (`codeact_consolidation.mopd_consolidation_run`) — each refuses rather than faking.
"""
