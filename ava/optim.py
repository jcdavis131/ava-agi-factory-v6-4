"""Muon: momentum-orthogonalized optimizer for the hidden matrices.

The recipe DeepSeek-lineage labs use to cut optimizer cost (Jordan et al.
2024, github.com/KellerJordan/Muon; scaled to frontier size by Moonlight/
Kimi K2): SGD-momentum whose update is orthogonalized with a Newton-Schulz
iteration before being applied. Two properties matter at Ava's scale:

* **Half the optimizer memory of AdamW for matrices.** One momentum buffer
  per matrix instead of Adam's (m, v) pair -- on mini that is ~0.6GB back on
  a 12GB GPU that crash-looped at 97% VRAM.
* **Fewer steps to a given loss** (reported ~1.3-2x on small transformers);
  at 262k tokens/step and ~86s/step, step-efficiency IS wall-clock.

Muon applies ONLY to 2D+ hidden weights. Embeddings, tied heads, norms,
biases, and the J-space decay logits keep AdamW -- orthogonalizing those is
known to hurt. `build_hybrid` wires the split and presents a single
optimizer-shaped object (step / zero_grad / state_dict / param_groups) so
ava/train.py's checkpoint and LR plumbing does not care.

LR coupling: the WSD schedule in the train loop writes group["lr"] every
step. Muon wants a larger LR than AdamW (rule of thumb ~0.02 vs 6e-4), so
each group carries "lr_scale" and the loop multiplies: lr * lr_scale.
Not a config default swap: presets opt in with optimizer.name: "muon".
"""

from __future__ import annotations

import torch

_NS_COEFFS = (3.4445, -4.7750, 2.0315)


@torch.no_grad()
def newton_schulz_orthogonalize(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate UV^T (the orthogonal factor of g's SVD) via the quintic
    Newton-Schulz iteration of the Muon reference implementation. Runs in
    bf16 on CUDA for speed; exact orthogonality is not required -- the
    iteration only has to crush the spread of singular values."""
    assert g.ndim == 2
    a, b, c = _NS_COEFFS
    x = g.to(torch.bfloat16 if g.is_cuda else torch.float32)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.mT
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        s = x @ x.mT
        x = a * x + (b * s + c * (s @ s)) @ x
    if transposed:
        x = x.mT
    return x.to(g.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5,
                 weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay,
                        lr_scale=1.0)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            # group["lr"] is the FINAL lr: when the train loop drives the WSD
            # schedule it already folds lr_scale in (train.py); standalone use
            # just takes the constructor lr. lr_scale is metadata, not a
            # factor here -- multiplying twice was the first bug this file had.
            lr = group["lr"]
            mom, nesterov = group["momentum"], group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)
                upd = g.add(buf, alpha=mom) if nesterov else buf
                flat = upd.reshape(upd.shape[0], -1)
                ortho = newton_schulz_orthogonalize(flat, group["ns_steps"])
                # Match RMS across shapes (Muon repo's scale rule).
                scale = max(1.0, flat.shape[0] / flat.shape[1]) ** 0.5
                if group["weight_decay"]:
                    p.mul_(1.0 - lr * group["weight_decay"])
                p.add_(ortho.view_as(p), alpha=-lr * scale)
        return loss


class HybridOptimizer:
    """Muon for hidden matrices + AdamW for everything else, presented as one
    optimizer: the interface subset ava/train.py actually uses."""

    def __init__(self, muon: Muon, adamw: torch.optim.AdamW):
        self.muon = muon
        self.adamw = adamw

    @property
    def param_groups(self):
        return list(self.muon.param_groups) + list(self.adamw.param_groups)

    def step(self):
        self.muon.step()
        self.adamw.step()

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, sd):
        self.muon.load_state_dict(sd["muon"])
        self.adamw.load_state_dict(sd["adamw"])


_ADAMW_NAME_MARKERS = ("embed", "lm_head", "verbalizer", "decay_logit")


def is_muon_param(name: str, p: torch.nn.Parameter) -> bool:
    """Hidden matrices only: 2D+, not embeddings/heads/logits/norms."""
    if p.ndim < 2:
        return False
    return not any(m in name for m in _ADAMW_NAME_MARKERS)


def build_hybrid(model: torch.nn.Module, *, adamw_lr: float,
                 betas: tuple[float, float], weight_decay: float,
                 muon_lr: float = 0.02, momentum: float = 0.95,
                 ns_steps: int = 5) -> HybridOptimizer:
    muon_params, decay, no_decay = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if is_muon_param(n, p):
            muon_params.append(p)
        elif p.ndim < 2 or "decay_logit" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    # lr_scale carries Muon's LR ratio through the trainer's per-step
    # `group["lr"] = wsd_lr * group.get("lr_scale", 1)` assignment, so the
    # WSD shape (warmup/stable/decay) applies to both optimizers.
    muon = Muon(muon_params, lr=muon_lr, momentum=momentum, ns_steps=ns_steps)
    muon.param_groups[0]["lr_scale"] = muon_lr / max(adamw_lr, 1e-12)
    adamw = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=adamw_lr, betas=betas)
    return HybridOptimizer(muon, adamw)
