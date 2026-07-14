"""Verifiable prompt/reward tasks for GRPO post-training.

Same discipline as ava/datagen (specs/02): every ground truth is COMPUTED,
prompts are deterministic given the seed, zero network. Each task family
yields Sample(prompt, check) where check(completion_text) -> float is an
exact verifier plus light format shaping toward the curriculum's own
"Final answer: X" convention (ava/datagen/math_gen.py) -- the policy is
rewarded for answering in the dialect it was pretrained on.

Reward scale: 0.0 wrong/absent, 1.0 correct, +0.1 when the completion leads
with the answer instead of burying it. Self-verification shaping (specs/13
item 2) composes by wrapping `check`, not by editing this file.
"""

from __future__ import annotations

import dataclasses
import random
import re
from typing import Callable

_INT_RE = re.compile(r"-?\d+")


@dataclasses.dataclass(frozen=True)
class Sample:
    prompt: str
    check: Callable[[str], float]
    meta: dict


def _int_verifier(answer: int) -> Callable[[str], float]:
    def check(completion: str) -> float:
        m = _INT_RE.search(completion)
        if m is None or int(m.group()) != answer:
            return 0.0
        return 1.0 + (0.1 if m.start() < 16 else 0.0)
    return check


class ArithmeticTask:
    """a OP b with digit-count difficulty; the P0/P1 curriculum's home turf."""

    name = "arithmetic"

    def __init__(self, digits: int = 2):
        self.digits = digits

    def sample(self, rng: random.Random) -> Sample:
        hi = 10 ** self.digits - 1
        a, b = rng.randint(0, hi), rng.randint(0, hi)
        op = rng.choice(("+", "-", "*"))
        ans = {"+": a + b, "-": a - b, "*": a * b}[op]
        prompt = f"Compute {a} {op} {b}.\nFinal answer:"
        return Sample(prompt, _int_verifier(ans),
                      {"task": self.name, "answer": ans})


class ModularTask:
    """a mod m -- mirrors math_gen's _modular_doc family."""

    name = "modular"

    def sample(self, rng: random.Random) -> Sample:
        a, m = rng.randint(10, 999), rng.randint(2, 12)
        prompt = f"Compute {a} mod {m}.\nFinal answer:"
        return Sample(prompt, _int_verifier(a % m),
                      {"task": self.name, "answer": a % m})


class ComparisonTask:
    """Which is larger -- exact answer is one of the two operands."""

    name = "comparison"

    def sample(self, rng: random.Random) -> Sample:
        a = rng.randint(-999, 999)
        b = rng.randint(-999, 999)
        while b == a:
            b = rng.randint(-999, 999)
        prompt = f"Which is larger, {a} or {b}?\nFinal answer:"
        return Sample(prompt, _int_verifier(max(a, b)),
                      {"task": self.name, "answer": max(a, b)})


class SelfVerifyWrapper:
    """DeepSeekMath-V2-style self-verification shaping (specs/13 item 2).

    Wraps any task so the policy must propose, check its own work, then
    commit -- and gets paid for HONEST checking, not just lucky answers:

        <question>
        Proposed answer: <int>
        Check: PASS|FAIL
        Final answer: <int>

    Reward = base verifier on the FINAL answer
           + 0.2 when the Check verdict correctly labels the PROPOSED answer
             (PASS iff it was right -- agreement with the exact verifier)
           - 0.3 when the model commits an answer its own Check flagged
             (verdict FAIL but final == proposed: it ignored itself)
           + 0.1 when all three fields parse (format shaping).

    Pure composition over Sample.check; the GRPO core and loop are untouched.
    """

    _VERDICT_RE = re.compile(r"Check:\s*(PASS|FAIL)", re.IGNORECASE)
    _FINAL_RE = re.compile(r"Final answer:\s*(-?\d+)")

    def __init__(self, task):
        self.task = task
        self.name = f"selfverify_{task.name}"

    def sample(self, rng: random.Random) -> Sample:
        inner = self.task.sample(rng)
        answer = inner.meta["answer"]
        question = inner.prompt.rsplit("\nFinal answer:", 1)[0]
        prompt = f"{question}\nProposed answer:"

        def check(completion: str) -> float:
            proposed_m = _INT_RE.search(completion)
            verdict_m = self._VERDICT_RE.search(completion)
            final_m = self._FINAL_RE.search(completion)

            reward = 0.0
            if final_m is not None and int(final_m.group(1)) == answer:
                reward += 1.0
            if proposed_m is not None and verdict_m is not None:
                proposed_right = int(proposed_m.group()) == answer
                said_pass = verdict_m.group(1).upper() == "PASS"
                if said_pass == proposed_right:
                    reward += 0.2                      # honest self-check
                if (not said_pass) and final_m is not None \
                        and int(final_m.group(1)) == int(proposed_m.group()):
                    reward -= 0.3                      # ignored its own FAIL
            if proposed_m and verdict_m and final_m:
                reward += 0.1                          # full format
            return reward

        return Sample(prompt, check, {**inner.meta, "task": self.name})


TASKS: dict[str, Callable[[], object]] = {
    ArithmeticTask.name: ArithmeticTask,
    ModularTask.name: ModularTask,
    ComparisonTask.name: ComparisonTask,
}


def build_tasks(names: str | list[str]) -> list:
    """'arithmetic,selfverify_modular' -> task instances. A 'selfverify_'
    prefix wraps the base task in SelfVerifyWrapper."""
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]
    out = []
    for n in names:
        wrap = n.startswith("selfverify_")
        base = n.removeprefix("selfverify_")
        if base not in TASKS:
            raise ValueError(f"unknown task {n!r}; available: {sorted(TASKS)} "
                             f"(optionally prefixed with 'selfverify_')")
        task = TASKS[base]()
        out.append(SelfVerifyWrapper(task) if wrap else task)
    return out
