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


TASKS: dict[str, Callable[[], object]] = {
    ArithmeticTask.name: ArithmeticTask,
    ModularTask.name: ModularTask,
    ComparisonTask.name: ComparisonTask,
}


def build_tasks(names: str | list[str]) -> list:
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]
    unknown = [n for n in names if n not in TASKS]
    if unknown:
        raise ValueError(f"unknown tasks {unknown}; available: {sorted(TASKS)}")
    return [TASKS[n]() for n in names]
