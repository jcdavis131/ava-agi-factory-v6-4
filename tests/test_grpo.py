"""GRPO core tests: the learning claim is tested, not just the plumbing.

The toy task: prompts are (a, b, PLUS) over a digit vocabulary; the correct
completion's first token is (a+b) mod 10. Exact verifier, single-token
credit -- if GRPO's group-relative advantages work, accuracy must climb far
above the 10% random baseline within a few hundred updates on a tiny model.
"""

from __future__ import annotations

import copy

import torch

from ava.rl.grpo import (
    GRPOConfig,
    arithmetic_reward,
    completion_logprobs,
    group_advantages,
    grpo_loss,
    grpo_step,
    sample_group,
)

VOCAB = 12                       # digits 0-9, PLUS=10, EOS=11
PLUS = 10


class TinyLM(torch.nn.Module):
    """Smallest thing with AvaModel1B's forward contract."""

    def __init__(self, d=32):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, d)
        self.rnn = torch.nn.GRU(d, d, batch_first=True)
        self.head = torch.nn.Linear(d, VOCAB)

    def forward(self, input_ids=None, **_):
        h, _s = self.rnn(self.embed(input_ids))
        return {"lm_logits": self.head(h)}


def _prompts(rng: torch.Generator, n: int) -> torch.Tensor:
    a = torch.randint(0, 10, (n,), generator=rng)
    b = torch.randint(0, 10, (n,), generator=rng)
    return torch.stack([a, b, torch.full_like(a, PLUS)], dim=1)


def _reward(prompt_row: torch.Tensor, full_row: torch.Tensor) -> float:
    want = int(prompt_row[0] + prompt_row[1]) % 10
    return 1.0 if int(full_row[3]) == want else 0.0


def test_group_advantages_zscore_and_degenerate_groups():
    r = torch.tensor([[1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    adv = group_advantages(r)
    assert abs(float(adv[0].mean())) < 1e-6
    assert float(adv[0].std()) > 0
    assert torch.all(adv[1] == 0), "all-same-reward group must carry no signal"


def test_grpo_loss_direction_and_kl_nonnegative():
    cfg = GRPOConfig()
    logp_old = torch.zeros(4)
    logp_ref = torch.zeros(4)
    up = torch.full((4,), 0.1, requires_grad=True)
    adv = torch.tensor([1.0, 1.0, -1.0, -1.0])
    loss, stats = grpo_loss(up, logp_old, logp_ref, adv, cfg)
    assert stats["kl"] >= 0
    loss.backward()
    # positive-advantage rows must be pushed UP (negative gradient on logp)
    assert up.grad[0] < 0 and up.grad[2] > 0


def test_sample_group_layout_and_logprobs_shape():
    torch.manual_seed(0)
    m = TinyLM()
    g = torch.Generator().manual_seed(0)
    prompts = _prompts(g, 3)
    cfg = GRPOConfig(group_size=4, max_new_tokens=2)
    ids = sample_group(m, prompts, cfg)
    assert ids.shape[0] == 12 and ids.shape[1] <= 5
    assert torch.equal(ids[0, :3], prompts[0]) and torch.equal(ids[3, :3], prompts[0])
    assert torch.equal(ids[4, :3], prompts[1])
    lp = completion_logprobs(m, ids, prompt_len=3)
    assert lp.shape == (12,) and torch.all(lp <= 0)


def test_grpo_learns_modular_addition():
    torch.manual_seed(7)
    model = TinyLM()
    ref = copy.deepcopy(model).eval()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    cfg = GRPOConfig(group_size=8, max_new_tokens=1, temperature=1.0,
                     kl_coef=0.001, inner_epochs=2)
    g = torch.Generator().manual_seed(7)

    # Assert on the quantity GRPO optimizes: mean sampled reward. (Argmax
    # accuracy of the tiny GRU lags the sampled policy and oscillates --
    # asserting on it tested the toy model's capacity, not the optimizer.)
    history = [grpo_step(model, ref, opt, _prompts(g, 8), _reward, cfg)["reward_mean"]
               for _ in range(300)]
    first, last20 = sum(history[:20]) / 20, sum(history[-20:]) / 20
    # random-guess reward is 0.10; demand a decisive, sustained climb
    assert last20 > max(0.35, first + 0.20), (first, last20)


def test_arithmetic_reward_verifier():
    assert arithmetic_reward("2+3", "5 is the answer", 5) == 1.1
    assert arithmetic_reward("2+3", "the answer is probably 5", 5) == 1.0
    assert arithmetic_reward("2+3", "4", 5) == 0.0
    assert arithmetic_reward("2+3", "no digits here", 5) == 0.0
