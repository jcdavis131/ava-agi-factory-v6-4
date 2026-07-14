"""Muon vs AdamW A/B on the OCR-decompression task (specs/13 gate, CPU).

Same seed, same data, same tiny-but-real AvaModel1B as the vision pilot; the
ONLY difference between arms is build_hybrid (Moonlight-recipe Muon on
hidden matrices, AdamW elsewhere, same LR) vs plain AdamW. Reported: loss
curves and steps-to-target -- the specs/13 adoption gate is muon reaching
adamw's final loss in <=0.8x the steps. CPU-scale caveat: a 400-step byte-
level pilot is directional evidence for the gate, not the gate itself (that
runs at nano preset scale on GPU); Newton-Schulz per step also costs more
wall-clock per step on CPU than it does on CUDA.

Usage:  python scripts/muon_ab_ocr.py [--steps 400]
Writes: reports/muon_ab_ocr.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ava.optim import build_hybrid


def _build(seed: int):
    from model_1b import AvaModel1B
    torch.manual_seed(seed)
    return AvaModel1B(
        vocab_size=258, d_model=128, n_text=2, n_fusion=2, n_reason=1,
        n_heads=4, head_dim=32, mlp="swiglu", mlp_ratio=2.0,
        tie_lm_head=True, tie_verbalizer=True, multimodal=True,
        multi_jspace_enabled=True,
        jspace_slots={"system1": 8, "system2": 8, "critic": 4, "planner": 8},
        jspace_half_life={"system1": 8, "system2": 300, "critic": 30, "planner": 150},
        rope_base=10000,
    )


def _run_arm(name: str, opt_factory, pairs, steps: int, log) -> list[float]:
    from ava.vision_ocr import ocr_decompress_loss
    model = _build(seed=7)                    # identical init across arms
    opt = opt_factory(model)
    losses = []
    t0 = time.time()
    for step in range(1, steps + 1):
        opt.zero_grad()
        acc = 0.0
        for k in range(4):
            ids, patches = pairs[(step * 4 + k) % len(pairs)]
            loss = ocr_decompress_loss(model, ids, patches) / 4
            loss.backward()
            acc += float(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(acc)
        if step % 50 == 0:
            log("ab_step", arm=name, step=step, loss=round(acc, 4),
                s_per_step=round((time.time() - t0) / step, 2))
    return losses


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    # data identical across arms
    from scripts.nano_vision_pilot import _pairs  # reuse the pair builder
    from ava.datagen.wiki_gen import WikiGenerator
    pairs = _pairs(WikiGenerator(seed=7), 48, 192)

    reports = Path("reports"); reports.mkdir(exist_ok=True)
    mfile = open(reports / "muon_ab_ocr.jsonl", "a", buffering=1)

    def log(event, **kw):
        line = json.dumps({"ts": time.time(), "event": event, **kw})
        print(line, flush=True)
        mfile.write(line + "\n")

    log("ab_start", steps=args.steps, lr=args.lr, pairs=len(pairs))

    adamw = _run_arm("adamw", lambda m: torch.optim.AdamW(m.parameters(), lr=args.lr),
                     pairs, args.steps, log)
    muon = _run_arm("muon", lambda m: build_hybrid(m, adamw_lr=args.lr,
                                                   betas=(0.9, 0.95),
                                                   weight_decay=0.1),
                    pairs, args.steps, log)

    def tail_mean(xs, k=25):
        return sum(xs[-k:]) / min(k, len(xs))

    target = tail_mean(adamw)                 # adamw's final loss = the bar
    steps_to_target = next((i + 1 for i, v in enumerate(muon) if v <= target),
                           None)
    ratio = (steps_to_target / args.steps) if steps_to_target else None
    log("ab_done", adamw_final=round(tail_mean(adamw), 4),
        muon_final=round(tail_mean(muon), 4),
        muon_steps_to_adamw_final=steps_to_target,
        step_ratio=round(ratio, 3) if ratio else None,
        gate="muon <= 0.8x steps (specs/13; directional at CPU pilot scale)")
    mfile.close()
    print(f"\nA/B: adamw final {tail_mean(adamw):.4f} | muon final "
          f"{tail_mean(muon):.4f} | muon hit adamw's final at step "
          f"{steps_to_target} of {args.steps} (ratio "
          f"{ratio if ratio else 'n/a'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
