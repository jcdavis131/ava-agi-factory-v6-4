"""Nano vision pilot: does the vision-prefix path learn to READ pages?

The smoke rung for specs/12's training arm, runnable on CPU while the mini
run owns the GPU. A tiny-but-real AvaModel1B (multimodal=True, J-space on)
trains with ava/vision_ocr.ocr_decompress_loss on (rendered page, text)
pairs derived from WikiGenerator atlases -- the DeepSeek-OCR decompression
objective end to end, no stubs.

The decisive metric is the HELD-OUT CONDITIONAL GAP: cross-entropy on unseen
pages WITH the page image prefix vs WITHOUT it (images=None). A model that
exploits pixels must beat its own unconditional self; byte-level tokens make
the copy circuit as direct as possible. The full specs/12 gate (>=90% char
accuracy at 4x + comprehension probe) belongs to the GPU run with the real
BPE tokenizer -- this pilot proves the mechanism and the plumbing.

Usage:  python scripts/nano_vision_pilot.py [--steps 400] [--seq 192]
Writes: reports/nano_vision_pilot.jsonl, runs/nano_vision_pilot/pilot.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ava.datagen.wiki_gen import WikiGenerator
from ava.pipeline.pxpipe import render_to_patches
from ava.vision_ocr import comprehension_probe, ocr_decompress_loss
from model_1b import AvaModel1B

EOD = 256
VOCAB = 258  # 256 bytes + EOD + pad


class ByteTok:
    """Self-contained byte tokenizer: the pilot must not depend on volume
    files, and byte-level targets make optical decoding maximally direct."""

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8", errors="replace"))

    def decode(self, ids) -> str:
        return bytes(i for i in ids if 0 <= i < 256).decode("utf-8", errors="replace")


def _pairs(gen: WikiGenerator, n: int, seq: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """(input_ids [1, seq], patches [N, 1024]) pairs: a page slice and ITS
    OWN rendering. Same-text pairing is the OCR objective's whole point."""
    tok = ByteTok()
    out = []
    for d in gen.generate(10**9):
        if d["phase"] != "p2":
            continue
        text = d["text"][:seq]                       # one short page slice
        ids = tok.encode(text)[:seq]
        patches = render_to_patches(text, max_pages=1)
        if len(ids) < 32 or patches.shape[0] == 0:
            continue
        out.append((torch.tensor([ids], dtype=torch.long),
                    torch.from_numpy(patches)))
        if len(out) >= n:
            return out
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seq", type=int, default=192)
    ap.add_argument("--batch-pairs", type=int, default=4, help="pairs per step")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model = AvaModel1B(
        vocab_size=VOCAB, d_model=128, n_text=2, n_fusion=2, n_reason=1,
        n_heads=4, head_dim=32, mlp="swiglu", mlp_ratio=2.0,
        tie_lm_head=True, tie_verbalizer=True, multimodal=True,
        multi_jspace_enabled=True,
        jspace_slots={"system1": 8, "system2": 8, "critic": 4, "planner": 8},
        jspace_half_life={"system1": 8, "system2": 300, "critic": 30, "planner": 150},
        rope_base=10000,
    )
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train = _pairs(WikiGenerator(seed=args.seed), 48, args.seq)
    held = _pairs(WikiGenerator(seed=args.seed + 1), 12, args.seq)

    reports = Path("reports"); reports.mkdir(exist_ok=True)
    mfile = open(reports / "nano_vision_pilot.jsonl", "a", buffering=1)

    def log(event, **kw):
        line = json.dumps({"ts": time.time(), "event": event, **kw})
        print(line, flush=True)
        mfile.write(line + "\n")

    @torch.no_grad()
    def heldout_gap() -> tuple[float, float]:
        model.eval()
        with_img, without = [], []
        for ids, patches in held:
            with_img.append(float(ocr_decompress_loss(model, ids, patches)))
            logits = model(input_ids=ids)["lm_logits"]
            ce = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, VOCAB).float(), ids[:, 1:].reshape(-1))
            without.append(float(ce))
        model.train()
        return sum(with_img) / len(with_img), sum(without) / len(without)

    log("pilot_start", params=n_params, train_pairs=len(train),
        held_pairs=len(held), steps=args.steps, seq=args.seq,
        vision_tokens_per_pair=int(train[0][1].shape[0]))
    w0, wo0 = heldout_gap()
    log("heldout", step=0, loss_with_image=round(w0, 4),
        loss_without=round(wo0, 4), gap=round(wo0 - w0, 4))

    model.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        loss_acc = 0.0
        opt.zero_grad()
        for k in range(args.batch_pairs):
            ids, patches = train[(step * args.batch_pairs + k) % len(train)]
            loss = ocr_decompress_loss(model, ids, patches) / args.batch_pairs
            loss.backward()
            loss_acc += float(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0 or step == 1:
            log("pilot_step", step=step, ocr_loss=round(loss_acc, 4),
                s_per_step=round((time.time() - t0) / step, 2))

    w1, wo1 = heldout_gap()
    log("heldout", step=args.steps, loss_with_image=round(w1, 4),
        loss_without=round(wo1, 4), gap=round(wo1 - w1, 4))

    # comprehension probe: executes the full greedy-decode path; the SCORE
    # gate belongs to the GPU run -- a minutes-long CPU pilot reads pixels
    # (the gap) long before it answers questions.
    atlas = next(d["text"] for d in WikiGenerator(seed=99).generate(10**9)
                 if d["phase"] == "p2")
    probe = comprehension_probe(model, ByteTok(), atlas, max_questions=3)
    log("probe", **{k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in probe.items()})

    out = Path("runs/nano_vision_pilot"); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "vocab": VOCAB,
                "params": n_params, "steps": args.steps}, out / "pilot.pt")

    verdict = "PASS" if (wo1 - w1) > 0.2 and w1 < w0 else "FAIL"
    log("pilot_done", verdict=verdict,
        gap_before=round(wo0 - w0, 4), gap_after=round(wo1 - w1, 4),
        ocr_loss_first_to_last=f"{w0:.3f}->{w1:.3f}")
    mfile.close()
    print(f"\nVERDICT: {verdict} -- held-out conditional gap "
          f"{wo0 - w0:+.3f} -> {wo1 - w1:+.3f} nats "
          f"(image-conditioned CE {w0:.3f} -> {w1:.3f})")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
