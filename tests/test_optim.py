"""Muon / HybridOptimizer tests: routing, WSD coupling, convergence, resume."""

from __future__ import annotations

import torch

from ava.optim import (
    HybridOptimizer,
    Muon,
    build_hybrid,
    is_muon_param,
    newton_schulz_orthogonalize,
)


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(11, 8)
        self.w1 = torch.nn.Linear(8, 16, bias=True)
        self.w2 = torch.nn.Linear(16, 8, bias=False)
        self.norm = torch.nn.LayerNorm(8)
        self.lm_head = torch.nn.Linear(8, 11, bias=False)
        self.decay_logit = torch.nn.Parameter(torch.zeros(2, 2))

    def forward(self, ids):
        return self.lm_head(self.norm(self.w2(torch.relu(self.w1(self.embed(ids))))))


def test_param_routing():
    m = _Tiny()
    muon = {n for n, p in m.named_parameters() if is_muon_param(n, p)}
    assert muon == {"w1.weight", "w2.weight"}, muon
    # embeddings/head/norms/biases/decay logits all stay AdamW
    assert "embed.weight" not in muon and "lm_head.weight" not in muon
    assert "decay_logit" not in muon


def test_newton_schulz_tightens_singular_values():
    g = torch.randn(32, 64)
    o = newton_schulz_orthogonalize(g, steps=5).float()
    s_in = torch.linalg.svdvals(g)
    s_out = torch.linalg.svdvals(o)
    # The quintic NS iteration lands singular values in ~[0.7, 1.2] by design
    # (speed over exactness); assert that band plus a real spread reduction.
    assert (s_in.max() / s_in.min()) > 2 * (s_out.max() / s_out.min()), \
        "orthogonalization must substantially tighten the singular-value spread"
    assert 0.5 < s_out.min() and s_out.max() < 1.4
    assert o.shape == g.shape


def test_wsd_lr_coupling_matches_train_loop():
    """Moonlight RMS matching (arXiv 2502.16982): Muon reuses AdamW's lr
    directly, so both optimizers see the SAME schedule value."""
    m = _Tiny()
    opt = build_hybrid(m, adamw_lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
    wsd = 3e-4                                        # mid-schedule value
    for g in opt.param_groups:                        # exact train.py line
        g["lr"] = wsd * g.get("lr_scale", 1.0)
    assert abs(opt.muon.param_groups[0]["lr"] - wsd) < 1e-12
    assert abs(opt.adamw.param_groups[0]["lr"] - wsd) < 1e-12


def test_muon_update_rms_matches_adamw_range():
    """The 0.2*sqrt(max(m,n)) rescale must land update RMS near 0.2*lr --
    the property that makes AdamW's tuned lr transferable."""
    torch.manual_seed(4)
    w = torch.nn.Parameter(torch.zeros(64, 128))
    opt = Muon([w], lr=1.0, momentum=0.0, nesterov=False, weight_decay=0.0)
    w.grad = torch.randn(64, 128)
    before = w.detach().clone()
    opt.step()
    delta = (w.detach() - before)
    rms = delta.pow(2).mean().sqrt().item()
    assert 0.1 < rms < 0.3, rms                       # ~0.2 by construction


def test_hybrid_converges_and_beats_zero_progress():
    torch.manual_seed(0)
    m = _Tiny()
    opt = build_hybrid(m, adamw_lr=1e-2, betas=(0.9, 0.95), weight_decay=0.0)
    ids = torch.randint(0, 11, (64,))
    tgt = torch.randint(0, 11, (64,))
    first = None
    for _ in range(60):
        loss = torch.nn.functional.cross_entropy(m(ids), tgt)
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.7, (first, loss.item())


def test_state_dict_roundtrip():
    torch.manual_seed(1)
    m = _Tiny()
    opt = build_hybrid(m, adamw_lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0)
    ids = torch.randint(0, 11, (8,))
    loss = m(ids).sum()
    loss.backward()
    opt.step()
    sd = opt.state_dict()
    assert set(sd.keys()) == {"muon", "adamw"}

    m2 = _Tiny()
    opt2 = build_hybrid(m2, adamw_lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0)
    opt2.load_state_dict(sd)                          # the resume path
    assert len(opt2.muon.state_dict()["state"]) == len(opt.muon.state_dict()["state"])


def test_muon_standalone_updates_weights():
    torch.manual_seed(2)
    w = torch.nn.Parameter(torch.randn(16, 8))
    opt = Muon([w], lr=0.02)
    before = w.detach().clone()
    (w.sum() ** 2).backward()
    opt.step()
    assert not torch.allclose(before, w)
