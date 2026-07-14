# Deep research: validating the 2026-07-13 pipeline upgrades

Method: 5-angle web sweep → 15 sources fetched → falsifiable claims
extracted → each claim adversarially verified by 3 independent voters
against primary sources (≥2/3 refutations kill a claim). 105 agents, 595
tool calls. Full transcript: workflow `wf_c6ad8bd3-db5`.

## Headline: what survived, what didn't

Only two of five angles produced claims that survived 3-vote adversarial
verification: **optical text compression** and **Muon**. The other three
(CUDA/WSL2 stability lore, GRPO best practices, synthetic-wiki corpora)
yielded zero surviving claims — the web evidence is thin, anecdotal, or
vendor-only. Note carefully what that means: our LOCAL findings on those
fronts (expandable_segments failing on this WSL2 driver, battery throttling,
GRPO mechanics proven in our own tests) rest on our own empirical
measurements, which remain valid; they simply have no independently
verified external corroboration yet.

## Verified findings and what we did about them

### Optical compression (angle 2)

| Finding | Confidence | Action taken |
|---|---|---|
| DeepSeek-OCR: ~97% OCR precision <10x compression, ~60% at 20x (independently re-measured 59.1%) | high | Cited correctly in specs/12; no change |
| Glyph (arXiv 2510.17800): only **3-4x** sustainable when downstream task accuracy must hold | medium | Our 4x packed-window target sits exactly at this envelope — kept; noted in specs/12 |
| VTCBench (arXiv 2512.15649): OCR precision ≠ comprehension; retrieval degrades with context even at 2x (97.2%→81.3%, 1k→32k) | high | **Eval gate amended**: comprehension probe required, not char accuracy alone |
| pxpipe production telemetry: ~3.1 chars/image-token on dense traffic (~68% savings), ceiling ~18.3; silent single-glyph confabulations on hex strings; readability anti-correlated with density (100% retrieval @22pt → 17% @16pt → fail @12pt on fixed canvas) | medium | specs/12 now forbids routing byte-exact content (ids, hashes, code literals) through the optical path |

### Muon (angle 3)

| Finding | Confidence | Action taken |
|---|---|---|
| Real gains, honestly sized: 1.35x token-efficiency @124M (NanoGPT speedrun), ~25% compute @1.5B; **the "~2x" claim was REFUTED 0-3** (vendor self-report; disputed by Stanford's "Fantastic Pretraining Optimizers") | high | specs/13 numbers corrected; A/B gate relaxed 0.75x→0.8x |
| Muon on 2D hidden matrices only; AdamW for embeddings/LM head/1D+norms (Jordan, Moonlight, Essential AI independently agree) | high | Our split already matched — confirmed |
| Canonical hyperparameters: 5 Newton-Schulz steps in bf16, quintic coeffs (3.4445, -4.7750, 2.0315), singular values land [0.7, 1.3], Nesterov ~0.95 | high | Our implementation already matched — confirmed (incl. our test's [0.5, 1.4] band) |
| **Moonlight recipe (arXiv 2502.16982): rescale updates by 0.2·√max(A,B) + weight decay → Muon directly reuses the AdamW-tuned LR** | high | **`ava/optim.py` UPGRADED**: replaced the Keller-Jordan shape scale + 33x lr_scale hack with RMS matching; one WSD schedule now drives both optimizers; new RMS-property test pins update RMS ≈ 0.2·lr |
| Muon holds data efficiency far beyond the critical batch size (validated to 4B params / 16M-token batches, arXiv 2505.02222) | medium | Noted — relevant to base1b batch sizing |

## Implications summary

1. The pxpipe optical arm's conservative 4x target and packed-window
   objective are now externally validated as the right envelope; the eval
   gate gained a comprehension probe and a byte-exact-content exclusion.
2. The Muon implementation was upgraded from "popular recipe" to the
   verified Moonlight recipe — materially better (no second LR to tune)
   and honestly documented (1.35x, not 2x).
3. Claims we could not externally validate stay flagged as
   locally-evidenced-only in the hillclimb log; decisions based on them
   (expandable_segments revert, battery-power operations) remain justified
   by our own reproducible measurements on this machine.

Sources (primary, quote-verified): arXiv 2510.18234 (DeepSeek-OCR),
arXiv 2511.15244 (C3 re-measurement), arXiv 2510.17800 + github.com/thu-coai/Glyph,
arXiv 2512.15649 + github.com/Moenupa/VTCBench, github.com/teamchong/pxpipe,
arXiv 2502.16982 + github.com/MoonshotAI/Moonlight, arXiv 2505.02222
(Essential AI), Keller Jordan's Muon repo/NanoGPT speedrun records.
