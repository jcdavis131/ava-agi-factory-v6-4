"""GRPO: Group Relative Policy Optimization (DeepSeekMath, arXiv 2402.03300).

The post-P5 RL recipe for Ava's chat/math branches, chosen because it is the
one that fits a 12GB card: advantages are computed RELATIVE TO THE GROUP of k
samples from the same prompt (z-scored rewards), so there is no critic/value
network at all -- the second model PPO would keep resident simply does not
exist. The only extra memory over SFT is a frozen reference policy for the KL
term, and at Ava scale that is a few hundred MB.

Ava's structural advantage here: the synthetic curriculum COMPUTES its ground
truths (specs/02 hard rule), so rewards come from exact verifiers, not a
learned reward model. `arithmetic_reward` below is the template; math_gen /
logic verifiers plug into the same `reward_fn(prompt, completion) -> float`
interface.

This module is deliberately engine-agnostic: `sample_group` only needs a
callable returning {"lm_logits": [B, T, V]} (AvaModel1B's contract), so the
same core drives a toy model in tests today and serve_engine's sampler when
the post-P5 branch lands. Self-verification shaping (specs/13 item 2) enters
purely through reward_fn composition -- no changes here.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Callable, Mapping

import torch
import torch.nn.functional as F


@dataclasses.dataclass(frozen=True)
class GRPOConfig:
    group_size: int = 8          # k samples per prompt
    clip_eps: float = 0.2        # PPO-style ratio clip, kept by GRPO
    kl_coef: float = 0.02        # penalty vs the frozen reference policy
    temperature: float = 1.0
    max_new_tokens: int = 64
    inner_epochs: int = 2        # reuse each sampled batch this many times


# ---------------------------------------------------------------------------
# sampling


@torch.no_grad()
def sample_group(model, prompt_ids: torch.Tensor, cfg: GRPOConfig,
                 eos_id: int | None = None) -> torch.Tensor:
    """Sample k completions for each prompt. [B, P] -> [B*k, P+<=max_new].

    Plain temperature sampling; groups are laid out prompt-major
    (rows i*k..(i+1)*k-1 belong to prompt i), which is what
    `group_advantages` assumes.
    """
    B, P = prompt_ids.shape
    k = cfg.group_size
    ids = prompt_ids.repeat_interleave(k, dim=0)
    finished = torch.zeros(B * k, dtype=torch.bool, device=ids.device)
    for _ in range(cfg.max_new_tokens):
        logits = model(input_ids=ids)["lm_logits"][:, -1]
        probs = F.softmax(logits / max(cfg.temperature, 1e-6), dim=-1)
        nxt = torch.multinomial(probs, 1).squeeze(-1)
        if eos_id is not None:
            nxt = torch.where(finished, torch.full_like(nxt, eos_id), nxt)
            finished |= nxt == eos_id
        ids = torch.cat([ids, nxt[:, None]], dim=1)
        if eos_id is not None and bool(finished.all()):
            break
    return ids


def completion_logprobs(model, ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Sum of token log-probs over the completion region. [N, T] -> [N]."""
    logits = model(input_ids=ids)["lm_logits"]
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    tgt = ids[:, 1:]
    tok_lp = logp.gather(-1, tgt[..., None]).squeeze(-1)     # [N, T-1]
    return tok_lp[:, prompt_len - 1:].sum(dim=-1)


# ---------------------------------------------------------------------------
# the GRPO objective


def group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Z-score rewards within each group. [B, k] -> [B, k].

    A group with identical rewards (all right or all wrong) carries no
    learning signal; its std is ~0, so the clamp makes its advantages exactly
    0 rather than exploding -- degenerate groups are silently skipped, which
    is the behavior the DeepSeekMath paper relies on.
    """
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True)
    adv = (rewards - mean) / std.clamp_min(1e-4)
    return torch.where(std < 1e-6, torch.zeros_like(adv), adv)


def grpo_loss(logp_new: torch.Tensor, logp_old: torch.Tensor,
              logp_ref: torch.Tensor, advantages: torch.Tensor,
              cfg: GRPOConfig) -> tuple[torch.Tensor, dict]:
    """Clipped policy-gradient loss + k3 KL penalty. All inputs [N]."""
    ratio = torch.exp(logp_new - logp_old)
    clipped = ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
    pg = -torch.min(ratio * advantages, clipped * advantages).mean()
    # Schulman's k3 estimator: unbiased, always >= 0.
    log_r = logp_ref - logp_new
    kl = (torch.exp(log_r) - log_r - 1.0).mean()
    loss = pg + cfg.kl_coef * kl
    return loss, {"pg": float(pg), "kl": float(kl),
                  "ratio_mean": float(ratio.mean())}


def grpo_step(model, ref_model, opt, prompt_ids: torch.Tensor,
              reward_fn: Callable[[torch.Tensor, torch.Tensor], float],
              cfg: GRPOConfig, eos_id: int | None = None) -> Mapping:
    """One full GRPO update: sample -> score -> z-advantage -> clipped PG.

    reward_fn(prompt_row, full_row) -> float, called per sampled sequence.
    Returns metrics including mean reward (THE hill-climb number).
    """
    B, P = prompt_ids.shape
    k = cfg.group_size
    ids = sample_group(model, prompt_ids, cfg, eos_id=eos_id)

    rewards = torch.tensor(
        [reward_fn(prompt_ids[i // k], ids[i]) for i in range(B * k)],
        dtype=torch.float32, device=ids.device).reshape(B, k)
    adv = group_advantages(rewards).reshape(-1)

    with torch.no_grad():
        logp_old = completion_logprobs(model, ids, P)
        logp_ref = completion_logprobs(ref_model, ids, P)

    stats: dict = {}
    for _ in range(cfg.inner_epochs):
        logp_new = completion_logprobs(model, ids, P)
        loss, stats = grpo_loss(logp_new, logp_old, logp_ref, adv, cfg)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {"reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "loss": float(loss), **stats}


# ---------------------------------------------------------------------------
# verifier rewards (exact, computed -- no reward model)


_INT_RE = re.compile(r"-?\d+")


def arithmetic_reward(question: str, completion_text: str, answer: int) -> float:
    """1.0 if the first integer in the completion equals the computed answer,
    +0.1 format bonus when the answer appears in the first 8 characters.
    The template for math_gen/logic verifiers: exactness over vibes."""
    m = _INT_RE.search(completion_text)
    if m is None or int(m.group()) != answer:
        return 0.0
    return 1.0 + (0.1 if m.start() < 8 else 0.0)
