"""Tests for ava/vision_ocr.py -- the OCR-decompression loss, the render
wrapper, and the specs/12 comprehension gate.

We drive everything with a stub model that honors the vision-prefix contract
(forward(images, input_ids, task_type) -> {"lm_logits": [B, L, V]}) instead of
model_1b: another agent is editing model_1b concurrently, and the objective is
supposed to depend only on the contract, not on the real network. The stub
biases logits by the image mean in a vocab-dependent way, so tests can verify
the image actually changes the loss (a bias uniform across the vocab would be
invisible to cross-entropy, which is what we want to guard against).
"""

import numpy as np
import torch

from ava.pipeline import pxpipe
from ava.datagen.wiki_gen import WikiGenerator
from ava.vision_ocr import (
    ocr_decompress_loss,
    render_window,
    comprehension_probe,
)

VOCAB = 256


class StubModel(torch.nn.Module):
    """Minimal contract-honoring model. lm_logits[b,t] = W[input_ids[b,t]]
    plus, when images are present, image_mean * per-vocab bias -- a
    NON-uniform shift, so it perturbs the softmax (and thus the CE)."""

    def __init__(self, vocab: int = VOCAB, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(vocab, vocab, generator=g)
        self.vbias = torch.randn(vocab, generator=g)

    def forward(self, images=None, audio=None, input_ids=None, task_type="deliberate"):
        logits = self.W[input_ids]                       # [B, L, V]
        if images is not None:
            logits = logits + images.mean() * self.vbias
        return {"lm_logits": logits}


class ByteTok:
    """Byte-level tokenizer stub: encode/decode over 0..255."""

    def encode(self, s: str) -> list[int]:
        return [b for b in s.encode("utf-8", "replace")]

    def decode(self, ids: list[int]) -> str:
        return bytes(int(i) % 256 for i in ids).decode("utf-8", "replace")


# ---------------------------------------------------------------------------
# ocr_decompress_loss
# ---------------------------------------------------------------------------

def test_loss_is_finite_scalar():
    model = StubModel()
    input_ids = torch.randint(0, VOCAB, (2, 10))
    patches = torch.zeros(pxpipe.PATCHES_PER_PAGE, pxpipe.PATCH_DIM)
    loss = ocr_decompress_loss(model, input_ids, patches)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_image_changes_loss():
    """Zero-image and random-image forwards must yield different losses:
    the vision prefix has to actually reach the logits."""
    model = StubModel()
    input_ids = torch.randint(0, VOCAB, (3, 12))
    n, d = pxpipe.PATCHES_PER_PAGE, pxpipe.PATCH_DIM

    zero = ocr_decompress_loss(model, input_ids, torch.zeros(n, d))
    g = torch.Generator().manual_seed(1)
    rand = ocr_decompress_loss(model, input_ids, torch.rand(n, d, generator=g))

    assert torch.isfinite(zero) and torch.isfinite(rand)
    assert abs(zero.item() - rand.item()) > 1e-4


def test_loss_accepts_batched_and_unbatched_patches():
    model = StubModel()
    input_ids = torch.randint(0, VOCAB, (2, 8))
    n, d = 32, pxpipe.PATCH_DIM

    unbatched = ocr_decompress_loss(model, input_ids, torch.rand(n, d))
    batched = ocr_decompress_loss(model, input_ids, torch.rand(2, n, d))
    assert torch.isfinite(unbatched) and torch.isfinite(batched)


def test_loss_accepts_1d_input_ids_and_numpy_patches():
    model = StubModel()
    input_ids = torch.randint(0, VOCAB, (16,))          # [L] -> promoted to [1, L]
    patches = np.random.default_rng(0).random(
        (24, pxpipe.PATCH_DIM)).astype(np.float32)
    loss = ocr_decompress_loss(model, input_ids, patches)
    assert loss.dim() == 0 and torch.isfinite(loss)


# ---------------------------------------------------------------------------
# render_window
# ---------------------------------------------------------------------------

def test_render_window_shape_and_range():
    patches = render_window("hello world\n" * 20)
    assert patches.ndim == 2
    assert patches.shape[1] == pxpipe.PATCH_DIM
    assert patches.shape[0] % pxpipe.GRID == 0          # whole patch-rows
    assert patches.dtype == np.float32
    assert patches.min() >= 0.0 and patches.max() <= 1.0


def test_render_window_matches_pxpipe():
    text = "the quick brown fox jumps over the lazy dog"
    assert np.array_equal(render_window(text), pxpipe.render_to_patches(text))


def test_render_window_max_pages_caps_output():
    dense = ("x" * 64 + "\n") * 400                     # many pages of full lines
    one = render_window(dense, max_pages=1)
    two = render_window(dense, max_pages=2)
    assert one.shape[0] <= pxpipe.PATCHES_PER_PAGE
    assert two.shape[0] > one.shape[0]


# ---------------------------------------------------------------------------
# comprehension_probe
# ---------------------------------------------------------------------------

def _wiki_p2_atlas() -> str:
    gen = WikiGenerator(seed=7)
    for doc in gen.generate(40_000):
        if doc["phase"] == "p2":
            return doc["text"]
    raise AssertionError("WikiGenerator produced no p2 atlas doc")


def test_probe_finds_facts_and_scores_in_range():
    atlas = _wiki_p2_atlas()
    result = comprehension_probe(StubModel(), ByteTok(), atlas, max_questions=4)
    assert result["n"] >= 1                             # at least one fact question
    assert 0.0 <= result["score"] <= 1.0


def test_probe_respects_max_questions():
    atlas = _wiki_p2_atlas()
    result = comprehension_probe(StubModel(), ByteTok(), atlas, max_questions=2)
    assert result["n"] <= 2


def test_probe_empty_text_is_zero():
    result = comprehension_probe(StubModel(), ByteTok(), "no facts here at all")
    assert result == {"score": 0.0, "n": 0}


def test_parsed_facts_bind_to_planet_headings():
    """A perfect-oracle tok that always emits the right number scores 1.0 --
    proves the fact/answer parsing is internally consistent."""
    from ava import vision_ocr

    atlas = _wiki_p2_atlas()
    facts = vision_ocr._parse_facts(atlas)
    assert facts, "expected computed facts in a wiki atlas"
    # every fact's name must be an actual heading in the atlas
    headings = {m.group(1).strip() for m in vision_ocr._HEADING.finditer(atlas)}
    for f in facts:
        assert f["name"] in headings
        assert f["field"] in {"orbit", "period", "equilibrium temperature", "moons"}


class OracleModel(torch.nn.Module):
    """Emits the answer's bytes in order (cycling), so the greedy-decoded
    continuation spells the answer and containment scoring fires. Exercises the
    scoring path end-to-end without a trained network."""

    def __init__(self, answer: str):
        super().__init__()
        self.answer_bytes = [b for b in answer.encode("utf-8")]
        self.step = 0

    def forward(self, images=None, audio=None, input_ids=None, task_type="deliberate"):
        B, L = input_ids.shape
        logits = torch.zeros(B, L, VOCAB)
        nxt = self.answer_bytes[self.step % len(self.answer_bytes)]
        logits[:, -1, nxt] = 10.0
        self.step += 1
        return {"lm_logits": logits}


def test_probe_scoring_counts_hits():
    """With a model rigged to emit the exact answer, containment scoring fires."""
    from ava import vision_ocr

    atlas = _wiki_p2_atlas()
    facts = vision_ocr._parse_facts(atlas)[:1]
    assert facts
    value = facts[0]["value"]
    result = comprehension_probe(OracleModel(value), ByteTok(), atlas, max_questions=1)
    assert result["n"] == 1
    assert result["score"] == 1.0
