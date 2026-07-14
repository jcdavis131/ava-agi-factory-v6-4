"""End-to-end GRPO loop tests: full text path (tokenize -> sample -> decode ->
verify -> update), driven by a char-level stub tokenizer and a tiny GRU.

The learning assertion targets single-digit sums through the REAL loop --
prompt text in, verifier reward out -- so tokenizer plumbing, EOS handling,
and reward decoding are all on the hook, not just the optimizer math.
"""

from __future__ import annotations

import copy
import random

import torch

from ava.rl.grpo import GRPOConfig
from ava.rl.loop import run_grpo
from ava.rl.tasks import ArithmeticTask, ComparisonTask, ModularTask, build_tasks

_CHARS = "0123456789+-* .,?\nabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ:"
_EOS = len(_CHARS)                    # one past the char vocab
VOCAB = len(_CHARS) + 1


class CharTok:
    def encode(self, text: str) -> list[int]:
        return [_CHARS.index(c) if c in _CHARS else _CHARS.index(" ") for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(_CHARS[i] for i in ids if 0 <= i < len(_CHARS))


class TinyLM(torch.nn.Module):
    def __init__(self, d=48):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, d)
        self.rnn = torch.nn.GRU(d, d, batch_first=True)
        self.head = torch.nn.Linear(d, VOCAB)

    def forward(self, input_ids=None, **_):
        h, _s = self.rnn(self.embed(input_ids))
        return {"lm_logits": self.head(h)}


def test_task_verifiers_are_exact():
    rng = random.Random(3)
    for task in (ArithmeticTask(digits=1), ModularTask(), ComparisonTask()):
        s = task.sample(rng)
        ans = s.meta["answer"]
        assert s.check(f" {ans}") >= 1.0
        assert s.check(f"the result is {ans} obviously") >= 1.0
        assert s.check(f" {ans + 1}") == 0.0
        assert s.check("gibberish with no digits") == 0.0


def test_tasks_are_deterministic_and_buildable():
    a = [ArithmeticTask().sample(random.Random(5)).prompt for _ in range(1)]
    b = [ArithmeticTask().sample(random.Random(5)).prompt for _ in range(1)]
    assert a == b
    assert len(build_tasks("arithmetic,modular")) == 2
    try:
        build_tasks("nope")
        raise AssertionError("unknown task must raise")
    except ValueError:
        pass


def test_loop_runs_logs_and_saves():
    torch.manual_seed(0)
    model = TinyLM()
    ref = copy.deepcopy(model).eval()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    cfg = GRPOConfig(group_size=4, max_new_tokens=4, temperature=1.0)
    events, saves = [], []
    res = run_grpo(model, ref, CharTok(), build_tasks("arithmetic"),
                   steps=12, cfg=cfg, opt=opt, rng=random.Random(0),
                   log=lambda ev, **kw: events.append((ev, kw)),
                   eos_id=_EOS, save_every=6, save_fn=lambda s: saves.append(s))
    assert res["steps"] == 12 and len(res["history"]) == 12
    assert saves == [6, 12]
    steps_logged = [kw["step"] for ev, kw in events if ev == "rl_step"]
    assert 1 in steps_logged and 10 in steps_logged
    assert all("reward_mean" in kw for ev, kw in events if ev == "rl_step")


def test_loop_learns_single_digit_sums_through_text():
    torch.manual_seed(11)

    class SingleDigitSum:
        name = "sds"

        def sample(self, rng):
            from ava.rl.tasks import Sample, _int_verifier
            a, b = rng.randint(0, 4), rng.randint(0, 4)
            return Sample(f"{a}+{b}:", _int_verifier(a + b),
                          {"task": "sds", "answer": a + b})

    model = TinyLM()
    ref = copy.deepcopy(model).eval()
    opt = torch.optim.Adam(model.parameters(), lr=4e-3)
    cfg = GRPOConfig(group_size=8, max_new_tokens=1, temperature=1.0,
                     kl_coef=0.001, inner_epochs=2)
    res = run_grpo(model, ref, CharTok(), [SingleDigitSum()],
                   steps=250, cfg=cfg, opt=opt, rng=random.Random(1),
                   log=lambda *a, **k: None, eos_id=_EOS)
    hist = res["history"]
    first, last = sum(hist[:20]) / 20, sum(hist[-20:]) / 20
    # The GRU starts near 0.04 (digits are 10 of 70 chars); a sustained climb
    # to >4x that through the FULL text path (tokenize/sample/decode/verify)
    # is the loop-correctness claim. Capacity ceilings belong to real models.
    assert last > max(0.18, first + 0.12), (first, last)


# ---------------------------------------------------------------------------
# Self-verification shaping (specs/13 item 2): pay for honest checking.

from ava.rl.tasks import SelfVerifyWrapper


def _sv_sample(seed=9):
    task = SelfVerifyWrapper(ArithmeticTask(digits=1))
    s = task.sample(random.Random(seed))
    return s, s.meta["answer"]


def test_selfverify_reward_matrix():
    s, ans = _sv_sample()
    wrong = ans + 1
    full = lambda p, v, f: f" {p}\nCheck: {v}\nFinal answer: {f}"
    # right answer, honest PASS, committed: 1.0 + 0.2 + 0.1
    assert abs(s.check(full(ans, "PASS", ans)) - 1.3) < 1e-9
    # wrong proposal, honest FAIL, corrected final: 1.0 + 0.2 + 0.1
    assert abs(s.check(full(wrong, "FAIL", ans)) - 1.3) < 1e-9
    # wrong proposal, dishonest PASS, wrong final: format only
    assert abs(s.check(full(wrong, "PASS", wrong)) - 0.1) < 1e-9
    # ignored its own FAIL (committed the flagged answer): 0.2 + 0.1 - 0.3
    assert abs(s.check(full(wrong, "FAIL", wrong)) - 0.0) < 1e-9
    # no verdict, no final: nothing
    assert s.check(f" {ans}") == 0.0


def test_selfverify_prompt_shape_and_build():
    s, _ = _sv_sample()
    assert s.prompt.endswith("Proposed answer:")
    assert "Final answer:" not in s.prompt
    tasks = build_tasks("selfverify_arithmetic,modular")
    assert tasks[0].name == "selfverify_arithmetic"
    assert tasks[1].name == "modular"
