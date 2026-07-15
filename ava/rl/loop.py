"""GRPO post-training loop: checkpoint in, RL rounds, checkpoint + metrics out.

The post-P5 driver (specs/13): fork a pretrained Ava checkpoint, sample k
completions per verifiable prompt, push the policy toward exact-verifier
rewards with ava/rl/grpo.py. No critic, no reward model; the only second
model in memory is the FROZEN reference for the KL leash.

Design constraints this file honors:

* **One prompt per grpo_step.** AvaModel1B.forward has no attention mask, so
  mixed-length prompt batches would need padding it cannot see. One prompt x
  group_size samples sidesteps masking entirely and is the memory shape a
  12GB card wants anyway. Rounds aggregate several prompts' metrics.
* **Engine-agnostic core.** Anything with encode/decode and the
  {"lm_logits"} forward contract runs -- the tests drive a char-stub
  tokenizer and a GRU; production runs AvaTokenizer + AvaModel1B via CLI:
    python -m ava.rl.loop --preset mini --ckpt /ckpt/stable_p5.pt \
        --tasks arithmetic,modular --steps 500 --out /ckpt/rl
* **Same observability as pretraining.** Events append to
  reports/rl_{preset}.jsonl (rl_step / rl_checkpoint / rl_done), so the
  dashboard's readers can chart reward the way they chart loss.

Do not point this at the LIVE run's GPU while the pretrainer is stepping.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import torch

from ava.rl.grpo import GRPOConfig, grpo_step
from ava.rl.tasks import build_tasks


def run_grpo(model, ref_model, tok, tasks, *, steps: int, cfg: GRPOConfig,
             opt, rng: random.Random, log, device: str = "cpu",
             eos_id: int | None = None, save_every: int = 0,
             save_fn=None) -> dict:
    """The loop body, separated from CLI/IO so tests can drive it whole."""
    model.train()
    history: list[float] = []
    for step in range(1, steps + 1):
        task = tasks[(step - 1) % len(tasks)]
        sample = task.sample(rng)
        prompt_ids = torch.tensor([tok.encode(sample.prompt)],
                                  dtype=torch.long, device=device)
        p_len = prompt_ids.shape[1]

        def reward_fn(_prompt_row, full_row, _s=sample, _p=p_len):
            completion = tok.decode(full_row[_p:].tolist())
            return _s.check(completion)

        stats = grpo_step(model, ref_model, opt, prompt_ids, reward_fn,
                          cfg, eos_id=eos_id)
        history.append(stats["reward_mean"])
        if step % 10 == 0 or step == 1:
            window = history[-20:]
            log("rl_step", step=step, task=sample.meta.get("task"),
                reward_mean=round(stats["reward_mean"], 4),
                reward_window=round(sum(window) / len(window), 4),
                kl=round(stats["kl"], 5), pg=round(stats["pg"], 5),
                ratio_mean=round(stats["ratio_mean"], 4))
        if save_every and save_fn and step % save_every == 0:
            save_fn(step)
    final = sum(history[-20:]) / max(1, len(history[-20:]))
    return {"steps": steps, "reward_final_window": final, "history": history}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ava.rl.loop")
    ap.add_argument("--preset", default=os.environ.get("AVA_PRESET", "nano"))
    ap.add_argument("--ckpt", required=True, help=".pt to fork the policy from")
    ap.add_argument("--out", default="/ckpt/rl")
    ap.add_argument("--reports", default=os.environ.get("AVA_REPORTS_DIR", "/reports"))
    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    ap.add_argument("--tasks", default="arithmetic,modular,comparison")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl-coef", type=float, default=0.02)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(argv)

    from ava.config import AvaConfig
    from ava.model import build_model
    from ava.tokenizer import ENDOFDOC_ID, AvaTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = AvaConfig.load(args.preset)
    tok = AvaTokenizer.load(os.environ.get("AVA_TOKENIZER"))

    model = build_model(cfg).to(device)
    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(blob["model"] if "model" in blob else blob)
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    gcfg = GRPOConfig(group_size=args.group_size,
                      max_new_tokens=args.max_new_tokens,
                      temperature=args.temperature, kl_coef=args.kl_coef)

    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    mfile = open(reports / f"rl_{args.preset}.jsonl", "a", buffering=1)

    def log(event, **kw):
        line = json.dumps({"ts": time.time(), "event": event,
                           "preset": args.preset, **kw})
        print(line, flush=True)
        mfile.write(line + "\n")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def save_fn(step):
        p = out / f"rl_step_{step}.pt"
        tmp = p.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "step": step,
                    "preset": args.preset, "rl": "grpo",
                    "forked_from": str(args.ckpt)}, tmp)
        os.replace(tmp, p)
        log("rl_checkpoint", path=str(p), step=step)

    log("rl_start", ckpt=args.ckpt, tasks=args.tasks, steps=args.steps,
        group_size=args.group_size, lr=args.lr, kl_coef=args.kl_coef,
        device=device)
    result = run_grpo(model, ref, tok, build_tasks(args.tasks),
                      steps=args.steps, cfg=gcfg, opt=opt,
                      rng=random.Random(args.seed), log=log, device=device,
                      eos_id=ENDOFDOC_ID, save_every=args.save_every,
                      save_fn=save_fn)
    save_fn(args.steps)
    log("rl_done", reward_final_window=round(result["reward_final_window"], 4))
    mfile.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
