# Deferred: port LongRoPE2 / Peri-LN / attention sinks onto the fixed model_1b.py

## Why this file exists

`master` and `claude/model-training-workflow-plan-n5vep5` diverged at `f508569`
and both rewrote `model_1b.py` / `multi_jspace_module.py` independently. Merging
`claude/model-training-workflow-plan-n5vep5` into `master` (this merge) kept the
**branch's** `model_1b.py` and dropped **master's** architecture additions,
because master's version still has the exact bug the branch's version fixed —
see below. Nothing was deleted permanently; it's recoverable from git history
and listed here so it doesn't get lost.

## What was kept (branch's model_1b.py)

Real, documented bug fixes, exercised by `tests/test_model.py` /
`tests/test_jlosses.py` (43 passed at merge time):

- Causal masking via `F.scaled_dot_product_attention(..., is_causal=True)`
  (previously: no mask at all — the model could attend to the future).
- `rotate_half()` fixed to match `get_cos_sin()`'s half-split
  (`cat((freqs, freqs))`) layout. Master's version still pairs dims via
  `x[..., ::2]` / `x[..., 1::2]` while its own `get_cos_sin()` builds the
  half-split layout — those two disagree, so **master's RoPE rotation is
  currently wrong**. Any hill-climb numbers measured against master's
  `model_1b.py` (LongRoPE2 non-uniform factors, Peri-LN, etc.) were computed
  on top of that broken rotation and should be treated as unverified until
  re-run on a corrected base.
- Detached cross-step `_prev_workspaces` cache (was crashing backward pass 2).
- Tied `lm_head`, and heads/head_dim/layer counts driven by `d_model` instead
  of hardcoded 16x128.
- Adds GQA (`n_kv_heads`) + SwiGLU + gradient checkpointing, config-gated.

## What was dropped from this merge (master's model_1b.py / multi_jspace_module.py)

Find the pre-merge content at `master`'s tip as of this merge:
`git show 67a5499:model_1b.py` (and `67a5499:scripts/prepare_longrope2.py`,
`67a5499:multi_jspace_module.py` for the OroJaR helpers — note OroJaR itself
*was* kept, see below).

- **LongRoPE2**: `longrope2_factors()`, non-uniform per-dim RoPE factors,
  `critical_dim_shift` 31→25, resonance mitigation. `scripts/prepare_longrope2.py`
  depended on this and was removed from the merge (dangling import) — recover
  it from `master` alongside the port.
- **Peri-LN**: QK-L2-norm (RMSNorm) + output-LN after attn and after FFN.
- **4 learnable attention sinks** (Xiao et al. 2023), `[H,4,D]` KV, always-attended.
- `rope_type="longrope2"` / `"yarn"` switch and the `use_peri_ln` / `n_sinks`
  constructor args on `AvaModel1B`.

## What was kept from master despite the conflict (no port needed — already merged)

`multi_jspace_module.py`'s conflict was purely additive (master added new
methods, branch's side was empty — no competing logic), so these are already
in the merged tree:

- `estimate_jacobian_fro_norm`, `orojar_orthogonal_loss`,
  `orojar_comprehensive_loss` (OroJaR: Jacobian Orthogonal Regularization +
  Lipschitz Fro norm). Not yet wired into `ava/jlosses.py`'s objective —
  wiring them in (or confirming they should stay opt-in) is a separate task
  from the RoPE port.

## To do the port

1. Re-derive `longrope2_factors()` / Peri-LN / attention-sink code from
   `master`'s `model_1b.py` (`git show 67a5499:model_1b.py`), applied on top
   of the *current* (fixed) `model_1b.py` rather than replacing it.
2. Fix master's `rotate_half()`/`get_cos_sin()` mismatch as part of the port —
   don't reintroduce it.
3. Re-run whatever produced master's hill-climb numbers
   ("`GWTB v2 d_gw=256 k=8 selective 0.3514 phi 4.55x cap 0.82 collapse 1.0`",
   see `tasks/hillclimb-log.md` and commit `5a8c2f6`) against the corrected
   rotation before trusting them.
4. Restore `scripts/prepare_longrope2.py` once `longrope2_factors` exists again.
5. Update `docs/LOCAL_LLMS_2026_SOTA.md`, which currently states "Ava v6.4
   already has ... LongRoPE2 non-uniform 31→25 ... Peri-LN, 4 sinks" — true of
   `master` pre-merge, not true of the merged tree until this port lands.
