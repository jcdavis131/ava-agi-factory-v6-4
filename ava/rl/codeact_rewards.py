# Solo personal project, no connection to employer, built with public/free-tier only
"""CodeAct return terms (spec 13 T13C.4) — pure reward-component functions over execution logs.

These consume the `Observation`s that the T13C.1 `Sandbox` produces for a trajectory and compute
the CodeAct-specific components that extend spec 12's `rl_return`:

  R_exec    — fraction of emitted code blocks that executed without an uncaught error. Penalizes
              "narrated"/hallucinated code that doesn't run. MUST be secondary to R_task (guarded by
              the weight and by the `codeact_return` blend below).
  R_codeuse — rewards independent tool calls that advance the task, penalizes redundant/duplicated
              calls (identical tool+args in consecutive steps) — the MAI tool-use finding. The
              redundancy definition mirrors scout-cli RFT `reward_components.redundant_steps`.
  R_len     — difficulty-scaled length penalty carried from spec 12: easy families (high historical
              pass rate) get a severe penalty (snap to a terse program), hard families a relaxed one.

These are GPU-free and fully testable. The GRPO loop that *consumes* them (T13C.4 wiring into
`ava/rl/grpo.py`) stays blocked on branch fine-tunes (T9.3/T9.5); this module is the reusable,
verified building block it will call. Naming per the spec-12 guard: `rl_return`/`R_*`, metrics
namespaced `rl.codeact.*`; `reward` remains the data-quality filter score elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ava.rl.codeact_sandbox import Observation


def r_exec(observations: Sequence[Observation]) -> float:
    """Fraction of blocks that executed cleanly, in [0, 1]. Empty → 0.0 (nothing ran)."""
    if not observations:
        return 0.0
    return sum(1 for o in observations if o.ok) / len(observations)


def _flatten_calls(observations: Sequence[Observation]) -> List[Tuple]:
    """Per-step tool calls flattened to hashable (tool, args, kwargs) tuples, in execution order."""
    calls: List[Tuple] = []
    for o in observations:
        for c in o.tool_calls:
            args = tuple(c.get("args", []) or [])
            kwargs = tuple(sorted((c.get("kwargs") or {}).items()))
            calls.append((c.get("tool"), args, kwargs))
    return calls


def redundant_calls(observations: Sequence[Observation]) -> int:
    """Count of consecutive identical (tool, args, kwargs) calls — the redundancy signal."""
    calls = _flatten_calls(observations)
    return sum(1 for a, b in zip(calls, calls[1:]) if a == b)


def r_codeuse(observations: Sequence[Observation]) -> float:
    """Tool-use quality in [-1, 1]. No tool use → 0.0 (neutral). 0 redundancy → 1.0; all-redundant
    → −1.0. Independent (non-duplicated) calls are rewarded by *not* being penalized."""
    calls = _flatten_calls(observations)
    if not calls:
        return 0.0
    redundant = sum(1 for a, b in zip(calls, calls[1:]) if a == b)
    return max(-1.0, 1.0 - 2.0 * redundant / len(calls))


def r_len(token_count: int, family_pass_rate: float, *, scale: float = 512.0) -> float:
    """Difficulty-scaled length penalty (≤ 0), carried from spec 12.

    The penalty *weight* is proportional to the family's historical pass rate: an easy family
    (pass_rate → 1) is penalized severely for length (snap to a terse program); a hard family
    (pass_rate → 0) is barely penalized (budget for a deep derivation). `scale` is the token count
    at which an easy family incurs a unit penalty — a training knob, not a measured constant.
    """
    pass_rate = min(1.0, max(0.0, family_pass_rate))
    return -pass_rate * (max(0, token_count) / scale)


@dataclass(frozen=True)
class ReturnWeights:
    """Blend weights. w_task dominates so R_exec can never outrank solving the task
    (guards the 'many trivial valid statements, wrong answer' hack)."""

    w_task: float = 1.0
    w_exec: float = 0.2
    w_codeuse: float = 0.2
    w_len: float = 0.1


def codeact_return(
    r_task: float,
    observations: Sequence[Observation],
    token_count: int,
    family_pass_rate: float,
    weights: ReturnWeights = ReturnWeights(),
) -> float:
    """Blend the CodeAct components into a scalar `rl_return` contribution.

    r_task is the verified-answer signal (0/1 or graded) owned by T12R.1 and passed in here.
    R_exec is deliberately weighted below R_task so a correct terse solution outranks lots of
    trivially-valid statements that don't solve the task.
    """
    return (
        weights.w_task * r_task
        + weights.w_exec * r_exec(observations)
        + weights.w_codeuse * r_codeuse(observations)
        + weights.w_len * r_len(token_count, family_pass_rate)
    )
