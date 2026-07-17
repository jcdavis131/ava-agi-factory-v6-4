# Solo personal project, no connection to employer, built with public/free-tier only
"""CodeAct eval (spec 13 T13C.3) — exec-verified success rate over a frozen held-out set.

The eval's **scoring engine is real**: it runs emitted code blocks through the T13C.1 `Sandbox`
and checks the final value equals the trajectory's gold answer. That engine is GPU-free and
fully tested here.

The **real-model path** (`run_codeact_eval`) needs the CodeAct decode loop — emit code → run in
the sandbox → feed the Observation back → iterate — which is the `ServeEngine` code-act loop
(T13C.5), not yet wired. So real mode **fails honestly** (returns an error record) rather than
fabricating a capability number, matching this repo's anti-mock discipline. Until then,
`simulate_policy_eval` exercises the real scoring engine with a *clearly-labeled synthetic policy*
(a plumbing check that varies by seed and is sensitive to a broken tool binding), satisfying the
T13C.3 acceptance criteria without a model.
"""

from __future__ import annotations

import random
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ava.datagen.codeact import Trajectory, iter_trajectories
from ava.rl.codeact_sandbox import Sandbox

CODEACT_EVAL_SEED = 20_240_717   # frozen held-out set seed (comparable across milestones)
CODEACT_BAR = 0.70


def held_out(n: int) -> List[Trajectory]:
    """Frozen held-out trajectory set — same seed every call so scores are comparable."""
    return list(iter_trajectories(seed=CODEACT_EVAL_SEED, n=n))


def score_emission(blocks: Sequence[str], gold_answer: str,
                   tool_sources: Optional[Dict[str, str]] = None, *, max_steps: int = 8) -> bool:
    """Run emitted code through the REAL sandbox; True iff the final value == the gold answer.

    A block may error mid-trajectory (the recover family's first block does); success is judged on
    the last value the trajectory produces. This is the load-bearing scoring primitive."""
    with Sandbox(tool_sources=tool_sources or {}, max_steps=max_steps) as vm:
        last_value = None
        for block in blocks:
            obs = vm.step(block)
            if obs.value is not None:
                last_value = obs.value
    return last_value == gold_answer


def _corrupt(blocks: List[str]) -> List[str]:
    """Deterministically break a trajectory so it no longer reaches the gold answer (drop the final
    computation, emit a wrong constant). Used only by the synthetic-policy plumbing check."""
    return list(blocks[:-1]) + ["0  # corrupted emission"]


def simulate_policy_eval(n: int = 20, accuracy: float = 0.75, seed: int = CODEACT_EVAL_SEED,
                         tool_binding_ok: bool = True) -> Dict[str, Any]:
    """Plumbing check: a seeded synthetic 'policy' emits correct-or-corrupted code per trajectory;
    every score comes from REAL sandbox execution. NOT a model-capability measurement — it exists to
    prove the eval harness works, varies by seed, and is sensitive to a broken tool binding."""
    trajs = held_out(n)
    rng = random.Random(seed)
    successes = 0
    for traj in trajs:
        emit_correct = rng.random() < accuracy
        blocks = traj.blocks if emit_correct else _corrupt(traj.blocks)
        tool_sources = traj.tool_sources if tool_binding_ok else {}
        successes += int(score_emission(blocks, traj.answer, tool_sources))
    rate = successes / max(1, len(trajs))
    return {
        "test": "codeact", "mode": "simulation",
        "measured": {"success_rate": round(rate, 4), "n": len(trajs),
                     "tool_binding_ok": tool_binding_ok},
        "pass": rate >= CODEACT_BAR,
        "bar": f"success_rate>={CODEACT_BAR}",
        "note": "synthetic-policy plumbing check — NOT model capability",
    }


def run_codeact_eval(model: Any, tokenizer: Any, preset: str = "nano",
                     device: str = "cpu", n: int = 20) -> Dict[str, Any]:
    """Real-model CodeAct eval. Honest-fail until the ServeEngine code-act decode loop (T13C.5)
    exists — the scoring engine (`score_emission`) is live and tested; the model-driving loop that
    would feed it the model's emissions is not wired, so this never fabricates a success rate."""
    return {
        "test": "codeact", "measured": None, "pass": False,
        "bar": f"success_rate>={CODEACT_BAR}",
        "error": "real mode not implemented: the CodeAct decode loop (emit code -> sandbox -> feed "
                 "Observation back -> iterate) is T13C.5 (ServeEngine code-act). The scoring engine "
                 "score_emission() is live; simulate_policy_eval() exercises it without a model.",
    }
