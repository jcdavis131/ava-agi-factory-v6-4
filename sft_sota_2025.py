"""sft_sota_2025.py — chat/tool-use branch SFT data prep for Ava.

Was a 2-line stub (`print("SFT 2025: iw-SFT importance weighted, instruction
tuning for code/math/chat branches")`). This is Phase 6 of
~/.claude/plans/tender-tinkering-sketch.md: the model-side lever for the
coding-agent stack — generates and packs the training data a future chat/
tool-use branch fine-tune needs, so `ava/train.py --branch chat` has real
data to consume instead of nothing.

Data sources, combined:
  * ava/datagen/chat_safety.py — existing chat/safety/delegation corpus.
  * ava/datagen/react_tools.py — new (Phase 6) synthetic ReAct tool-use
    corpus, weighted toward grounding/anti-hallucination per the project's
    north star.
  * agent-eval/results/*.json via agent-eval/scripts/export_sft_corpus.py's
    output (--distilled) — real, verified-successful transcripts from
    whichever brain is currently serving as the harness's tool-calling
    baseline, once any exist (none do yet as of 2026-07-12 — qwen2.5:1.5b's
    baseline run was 0/6; see agent-eval/hillclimb-log.md).

Deliberately does NOT launch training. This script is pure CPU/disk work —
generation, tokenization, packing, and registering shards in an ISOLATED
manifest DB (never the live pipeline's /state/manifest.db) — and is safe to
run alongside a live GPU training job. Actually training loads a model onto
GPU, which the RAM-pressure incident this session hit (documented in
agent-eval/README.md) showed needs to be done deliberately, with checked
headroom, not as a side effect of a data-prep script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ava.datagen.chat_safety import ChatSafetyGenerator
from ava.datagen.react_tools import ReactToolsGenerator
from ava.pipeline.manifest import PACKED, Manifest
from ava.pipeline.pack import load_tokenizer, pack_docs, write_shard


def _load_distilled_docs(distilled_jsonl: str | Path) -> list[dict]:
    """Reshape agent-eval's distilled export (task_id/category/source/text)
    into the doc schema pack_docs() expects. phase is fixed at p5 (anneal) —
    this is chat-branch-only data, not curriculum-phased pretraining data,
    so the usual phase-progression semantics don't apply; p5 is just where
    chat_safety.py's own data already lives."""
    docs = []
    for i, line in enumerate(Path(distilled_jsonl).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        d = json.loads(line)
        docs.append({
            "doc_id": f"distilled:{d['task_id']}:{i}",
            "text": d["text"],
            "task_type": "deliberate",
            "concept": d.get("category", "distilled"),
            "phase": "p5",
            "source": d["source"],
        })
    return docs


def prepare_branch_data(
    out_dir: str,
    db_path: str,
    tokenizer_path: str | None = None,
    target_mb_per_generator: float = 2.0,
    seed: int = 1234,
    distilled_jsonl: str | Path | None = None,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lt = load_tokenizer(tokenizer_path)

    generators = [ChatSafetyGenerator(seed=seed), ReactToolsGenerator(seed=seed)]
    all_docs: list[dict] = []
    for gen in generators:
        docs = list(gen.generate(int(target_mb_per_generator * (1024 ** 2))))
        all_docs.extend(docs)
        print(f"  {gen.name}: {len(docs)} docs")

    if distilled_jsonl and Path(distilled_jsonl).is_file():
        distilled = _load_distilled_docs(distilled_jsonl)
        all_docs.extend(distilled)
        print(f"  distilled: {len(distilled)} docs")

    arr, idx = pack_docs(all_docs, lt)
    bin_path = out / "chat_branch_0000.bin"
    write_shard(arr, idx, bin_path)
    print(f"Packed {idx['tokens']} tokens, {len(idx['docs'])} docs -> {bin_path}")

    with Manifest(db_path=db_path) as m:
        m.add_shard(
            "sft_chat_branch_0000", source="sft_sota_2025", phase=5,
            path=str(bin_path), split="train", bytes_=int(arr.nbytes),
            docs=len(idx["docs"]), sha256=idx["tokenizer_sha"], state=PACKED,
        )
    print(f"Registered in isolated manifest at {db_path} (NOT the live pipeline's manifest.db)")

    return {
        "tokens": idx["tokens"], "docs": len(idx["docs"]),
        "bin_path": str(bin_path), "db_path": db_path,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/sft_branch/packed")
    ap.add_argument("--db", default="runs/sft_branch/manifest.db")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--target-mb", type=float, default=2.0, help="per generator")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--distilled", default=None,
        help="path to agent-eval's exported distilled_react.jsonl, if any real successes exist yet",
    )
    args = ap.parse_args()

    stats = prepare_branch_data(
        args.out, args.db, tokenizer_path=args.tokenizer,
        target_mb_per_generator=args.target_mb, seed=args.seed,
        distilled_jsonl=args.distilled,
    )
    print()
    print(f"Data prep done: {stats['tokens']} tokens, {stats['docs']} docs.")
    print()
    print("This script does NOT launch training. Launching loads a model onto GPU and")
    print("should be a deliberate step with confirmed headroom, not a script side effect")
    print("-- especially while another training run may already be using the GPU.")
    print()
    print("Also unresolved before that step can run: locate a valid --init checkpoint")
    print("(the earlier nano chat fork was pruned by the janitor's retention policy per")
    print("TODOS.md T9.1; no runs/base/*.pt is present in this checkout -- checkpoints")
    print("live in the compose services' bind-mounted volumes, not the bare repo).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
