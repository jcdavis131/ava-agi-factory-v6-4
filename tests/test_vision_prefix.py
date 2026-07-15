"""Vision-prefix path (specs/12_pxpipe_optical_wiki.md, training arm item 1).

The old vision path was a pooled-mean stub -- `x = x + v.mean(dim=1)` -- one
vector added to every text position cannot carry a page, so nothing trained
through it could decompress one. The replacement prepends the [B, N, d] patch
tokens to the embedded text and lets the causal mask do the reading (text
follows the prefix, so every text position attends to all N vision tokens),
then slices the prefix back off before J-space / reasoning / lm_head so every
caller-visible shape stays [B, L_text, ...].

The non-negotiable property: a live training run depends on images=None being
byte-identical to the pre-prefix model. That gets the strictest check here
(torch.equal, not allclose).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model_1b import AvaModel1B

VOCAB = 128
# 7 is deliberately unrelated to any model dimension so a shape mix-up between
# the prefix length and heads/slots/seq-len cannot cancel out silently.
N_PATCHES = 7


def _model(multimodal: bool = True, seed: int = 0) -> AvaModel1B:
    """Small enough to run every test on CPU in milliseconds."""
    torch.manual_seed(seed)
    return AvaModel1B(
        vocab_size=VOCAB, d_model=32, n_text=1, n_fusion=1, n_reason=1,
        n_heads=2, head_dim=16, multimodal=multimodal, multi_jspace_enabled=True,
        jspace_slots={"system1": 4, "system2": 4, "critic": 4, "planner": 4},
    )


def _ids(B: int = 2, L: int = 11, seed: int = 1) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randint(0, VOCAB, (B, L))


def _images(B: int = 2, N: int = N_PATCHES, seed: int = 2) -> torch.Tensor:
    # render_to_patches (ava/pipeline/pxpipe.py) emits [N, 1024] float32 in
    # [0, 1]; rand matches that contract exactly.
    torch.manual_seed(seed)
    return torch.rand(B, N, 1024)


# ---------------------------------------------------------------------------
# 1. The prefix must be invisible in every caller-facing shape.

def test_prefix_leaves_all_caller_visible_shapes_text_sized():
    """lm_logits / fused / route_probs must be sized by the TEXT length alone:
    the trainer's shifted-CE, the J-losses, and the router targets all index by
    text position and would silently misalign if N leaked through."""
    m = _model().eval()
    ids = _ids()
    B, L = ids.shape
    with torch.no_grad():
        out = m(input_ids=ids, images=_images(B=B))
        text_only = m(input_ids=ids)

    assert out["lm_logits"].shape == (B, L, VOCAB)
    assert out["fused"].shape == text_only["fused"].shape
    assert out["jspace"]["route_probs"].shape == text_only["jspace"]["route_probs"].shape


# ---------------------------------------------------------------------------
# 2. images=None must be exactly the model the live run is training.

def test_images_none_is_byte_identical_to_multimodal_false_twin():
    """A multimodal=False twin carrying the same weights must produce
    bit-identical logits: proof the images=None path never consults the
    multimodal machinery (no prefix, no extra RoPE positions, no slicing)."""
    mm = _model(multimodal=True)
    txt = _model(multimodal=False, seed=99)  # different init on purpose; weights come from mm
    result = txt.load_state_dict(mm.state_dict(), strict=False)
    assert not result.missing_keys, "twin must receive every weight it owns"
    assert all(k.startswith(("vision_enc.", "audio_enc.")) for k in result.unexpected_keys), \
        "only the encoders may differ between the twins"

    mm.eval()
    txt.eval()
    ids = _ids()
    with torch.no_grad():
        a = mm(input_ids=ids)["lm_logits"]
        b = txt(input_ids=ids)["lm_logits"]
    assert torch.equal(a, b), "images=None must be byte-identical to the text-only model"


# ---------------------------------------------------------------------------
# 3. The prefix must actually be read, not just carried along.

def test_different_images_change_logits_for_identical_text():
    m = _model().eval()
    ids = _ids(B=1)
    zeros = torch.zeros(1, N_PATCHES, 1024)
    rand = _images(B=1)
    with torch.no_grad():
        out_zeros = m(input_ids=ids, images=zeros)["lm_logits"]
        out_rand = m(input_ids=ids, images=rand)["lm_logits"]
        out_none = m(input_ids=ids)["lm_logits"]

    assert not torch.allclose(out_zeros, out_rand, atol=1e-5), \
        "prefix CONTENT must reach the logits (text attends to the vision tokens)"
    assert not torch.allclose(out_rand, out_none, atol=1e-5), \
        "prefix PRESENCE must reach the logits"


# ---------------------------------------------------------------------------
# 4. The ocr_decompress objective needs gradient to reach the projection.

def test_lm_loss_backward_reaches_vision_projection():
    """The pooled-mean stub also passed gradient, but through a useless
    bottleneck; the property that matters for training arm item 2 is that a
    plain shifted-CE on TEXT logits still back-propagates into the (sliced-off)
    vision prefix via the attention keys/values."""
    torch.manual_seed(3)
    m = _model().train()
    ids = _ids(B=2, L=9)
    out = m(input_ids=ids, images=_images(B=2))
    loss = torch.nn.functional.cross_entropy(
        out["lm_logits"][:, :-1].reshape(-1, VOCAB).float(), ids[:, 1:].reshape(-1)
    )
    loss.backward()
    g = m.vision_enc.proj.weight.grad
    assert g is not None, "no grad reached vision_enc.proj -- the prefix is detached from the loss"
    assert float(g.abs().sum()) > 0.0, "grad on vision_enc.proj is identically zero"


# ---------------------------------------------------------------------------
# 5. multimodal=False is the live run's config: images must be inert there.

def test_multimodal_false_ignores_images_entirely():
    m = _model(multimodal=False).eval()
    assert m.vision_enc is None
    ids = _ids()
    with torch.no_grad():
        with_imgs = m(input_ids=ids, images=_images())["lm_logits"]
        without = m(input_ids=ids)["lm_logits"]
    assert torch.equal(with_imgs, without), \
        "multimodal=False must make images a no-op, bit for bit"


# ---------------------------------------------------------------------------
# 6. Same headline property as tests/test_model.py, now with a prefix present:
#    the prefix precedes all text (so it MAY inform every position), but text
#    must still not see its own future.

def test_text_causality_holds_with_vision_prefix():
    m = _model().eval()
    ids = _ids(B=1, L=10)
    imgs = _images(B=1)
    with torch.no_grad():
        base = m(input_ids=ids, images=imgs)["lm_logits"]

    t = 5
    p = ids.clone()
    p[0, t] = (p[0, t] + 1) % VOCAB
    with torch.no_grad():
        after = m(input_ids=p, images=imgs)["lm_logits"]

    torch.testing.assert_close(base[:, :t], after[:, :t], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(base[:, t], after[:, t], atol=1e-5), \
        "perturbation must propagate forward, or the test is vacuous"
