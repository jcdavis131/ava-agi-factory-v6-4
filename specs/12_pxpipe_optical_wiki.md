# 12 — pxpipe optical wikis: screenshots as curriculum

Status: data plane SHIPPED (generator + renderer + tests); training arm
SPECIFIED here, gated on the next run (mini's checkpoints are text-only and
`multimodal: false` — enabling vision mid-run would invalidate them).

## What and why

Three research threads, combined for this project:

1. **LLM wikis** (Karpathy, gist 442a6bf5): a persistent, interlinked
   markdown wiki — entity pages, concept pages, `index.md`, append-only
   `log.md`, `[[cross-links]]` — as a compounding knowledge artifact.
2. **pxpipe** (github.com/teamchong/pxpipe): render dense text as PNG pages
   so a multimodal model reads the same content in fewer tokens (~18 chars
   per vision token with a 5x8 monospace font).
3. **DeepSeek-OCR, "Contexts Optical Compression"** (arXiv 2510.18234): a
   text decoder can recover text from page images at ~97% precision under
   ~10x optical compression — vision tokens are a cheaper carrier for bulk
   text, and OCR-style decompression is a trainable objective.

The curriculum consequence: train the model to READ RENDERED PAGES of its own
wiki corpus — text conditioned on the page image (optical decompression) —
so long contexts can later be fed as cheap vision tokens, and grounding
(pointing at lines/boxes) becomes available as a native skill.

## What shipped (this repo, text pipeline unchanged)

- `ava/datagen/wiki_gen.py` — `WikiGenerator` (`generator: wiki`), a
  procedural star-system atlas in the Karpathy wiki shape. Every number is
  computed (Kepler `P = a^1.5`; equilibrium `T = 278·L^0.25/√a`; snow-line
  classification; concept-page aggregates), so the same fact recurs
  consistently across star page, planet page, concept page, and index —
  a factual graph, not paraphrase soup. Emits per wiki: entity pages +
  concept pages + index.md + log.md (p2, automatic), 2 query-with-citation
  docs (p3, deliberate), and a whole-wiki book (p4, automatic).
- `ava/pipeline/pxpipe.py` — deterministic text→image: 512×512 grayscale
  pages, vendored public-domain 8×8 font (`_font8x8.py`; PIL deliberately
  avoided — its default font changes across versions, breaking datagen's
  byte-identical contract). 32×32 patches → 1024-dim vectors, exactly
  `VisionEncoder`'s input dim. `render_to_patches(..., crop=True)` drops
  blank trailing patch-rows so sparse pages don't waste vision tokens.
- `configs/sources.yaml` — `synth_wiki_px` at p2 0.05 / p3 0.05 / p4 0.10
  (shaved from fineweb_edu p2/p4 and open_web_math p3; sums stay 1.0).
- Tests: `tests/test_pxpipe.py` — render determinism, patch geometry/order,
  crop behavior, wiki schema validity, link resolution, Kepler consistency.

**Key design call: images are never stored.** Rendering is a pure function
of text, so the trainer re-renders on demand from packed text windows. No
sidecar files, no new shard format, no curator changes, no extra disk.

## Measured compression (this renderer, this corpus)

| unit | chars | vision tokens (cropped) | ratio vs BPE (≈4 chars/tok) |
|---|---|---|---|
| entity page | 429 | 80 | 1.34x |
| wiki book | 2,778 | 496 | 1.40x |
| **packed 1024-token window** | ~4,096 | 256 | **4.0x** |

The training objective therefore operates on **packed windows** (dense by
construction), not raw docs. 4x is the honest unlearned baseline at this
patch geometry; DeepSeek-OCR's ~10x needs a learned conv compressor on top
(roadmap below).

## Training arm (next run / nano pilot — NOT mini mid-run)

Current `model_1b.forward` vision path is a stub (`x + v.mean(...)`) — a
single pooled vector added to every position cannot carry a page. Required
change (small, ~30 lines):

1. `VisionEncoder` output `[B, N, d]` becomes a PREFIX sequence:
   `x = cat([vis_tokens, embed(input_ids)])`, causal mask intact (text
   attends to all vision tokens; RoPE positions offset by N).
2. **Objective `ocr_decompress`**: sample a packed window (existing sampler),
   render via `render_to_patches` (CPU, ~ms), feed patches as prefix, LM
   loss on the text tokens only. This is the DeepSeek-OCR recipe at nano
   scale, and it reuses the entire existing loss/J-space stack unchanged.
3. Curriculum: a `p2v` interleave (e.g. 10% of steps run the optical
   objective on the same mixture) or a dedicated post-P5 vision phase for
   the mini/base1b successor; nano first as the smoke rung
   (`multimodal: true`, d_model 256 — VisionEncoder is `Linear(1024, d)`).
4. Eval gate: held-out page → greedy decode → char accuracy; targets:
   ≥90% at 4x (unlearned patches) before investing in the compressor.

## Roadmap after the OCR rung

- **Learned compressor**: small conv stack over the page (stride 2-4) to cut
  256 patches/page toward 64 (→ ~16x) — this is where DeepSeek's 10x lives.
- **Visual primitives** (pointing/boxes): `render_pages` already returns
  per-line bounding boxes. Generate QA docs whose answers cite a line box
  ("the answer is on line 12: [y0,x0,w]"), training point-and-read behavior
  — the DeepSeek-style grounding the user asked for, with zero extra
  rendering machinery.
- **Real pxpipe interop**: if the TypeScript renderer's page geometry is
  ever preferred (1568×728, Spleen 5×8), only `GLYPH/PAGE_SIDE/COLS/ROWS`
  constants change; the patch contract (1024-dim) is the stable interface.
