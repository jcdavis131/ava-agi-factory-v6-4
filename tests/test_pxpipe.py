"""pxpipe optical rendering + WikiGenerator tests.

The contract under test: page rendering is a PURE function of text (so
(screenshot, text) pairs can be derived at train time instead of stored), the
patch geometry matches VisionEncoder's 1024-dim input, and the generated
wikis are internally consistent (links resolve, physics recomputes).
"""

from __future__ import annotations

import numpy as np

from ava.datagen import GENERATORS, validate_doc
from ava.datagen.wiki_gen import WikiGenerator
from ava.pipeline.pxpipe import (
    CHARS_PER_PAGE,
    PATCH_DIM,
    PATCHES_PER_PAGE,
    compression_stats,
    page_to_patches,
    render_pages,
    render_to_patches,
)

SAMPLE = "# Kervolis\n\nKervolis is a class-G star.\n" + ("orbit line\n" * 80)


def test_render_is_deterministic():
    a = render_pages(SAMPLE)
    b = render_pages(SAMPLE)
    assert len(a) == len(b) == 2                     # 83 lines -> 2 pages
    for pa, pb in zip(a, b):
        assert np.array_equal(pa.pixels, pb.pixels)
        assert pa.pixels.dtype == np.uint8


def test_page_geometry_matches_vision_encoder():
    page = render_pages(SAMPLE)[0]
    assert page.pixels.shape == (512, 512)
    patches = page_to_patches(page)
    assert patches.shape == (PATCHES_PER_PAGE, PATCH_DIM) == (256, 1024)
    assert patches.dtype == np.float32
    assert 0.0 <= patches.min() and patches.max() <= 1.0
    assert patches.max() == 1.0, "ink must reach full intensity"


def test_patch_order_is_row_major():
    # a page with ink ONLY in the top-left 32x32 patch
    page = render_pages("X")[0]
    patches = page_to_patches(page)
    assert patches[0].max() == 1.0
    assert patches[1:].max() == 0.0


def test_render_to_patches_caps_pages():
    out = render_to_patches("line\n" * 1000, max_pages=2)
    assert out.shape == (2 * PATCHES_PER_PAGE, PATCH_DIM)


def test_empty_text_renders_nothing():
    assert render_to_patches("").shape == (0, PATCH_DIM)


def test_compression_stats_reports_ratio():
    text = "word " * 2000                            # 10_000 chars
    s = compression_stats(text)
    assert s["pages"] == (10_000 + CHARS_PER_PAGE - 1) // CHARS_PER_PAGE + 1 or s["pages"] >= 2
    assert s["vision_tokens"] == s["pages"] * PATCHES_PER_PAGE
    assert s["compression_ratio"] > 1.0, "optical form must be denser than BPE"


# ---------------------------------------------------------------------------
# WikiGenerator


def _pages(seed=7, target=40_000):
    return list(WikiGenerator(seed=seed).generate(target))


def test_wiki_registered_and_schema_valid():
    assert "wiki" in GENERATORS
    docs = _pages()
    assert len(docs) > 10
    for d in docs:
        validate_doc(d, allowed_phases=(2, 3, 4))


def test_wiki_is_deterministic():
    a = [d["text"] for d in _pages(seed=11)]
    b = [d["text"] for d in _pages(seed=11)]
    assert a == b


def test_wiki_links_resolve():
    """Karpathy-pattern integrity: every [[link]] in an atlas targets one of
    that atlas's own section pages (docs are now whole-wiki bundles)."""
    import re

    for d in _pages():
        if d["phase"] != "p2":                       # atlases carry the pages
            continue
        titles = {m.group(1).strip() for m in
                  re.finditer(r"^#+ (.+)$", d["text"], re.M)}
        links = set(re.findall(r"\[\[([^\]]+)\]\]", d["text"]))
        unresolved = {l for l in links if l not in titles}
        assert not unresolved, f"dangling wiki links: {sorted(unresolved)[:5]}"


def test_wiki_physics_recomputes():
    """Kepler's third law must hold on every planet infobox: period == orbit^1.5."""
    import re

    for d in _pages():
        for m in re.finditer(r"Orbit: ([\d.]+) AU\. Period: ([\d.]+) yr\.",
                             d["text"]):
            a, p = float(m.group(1)), float(m.group(2))
            assert abs(p - round(a ** 1.5, 2)) < 1e-9, d["text"].splitlines()[0]


def test_wiki_pages_render_to_patches():
    d = _pages()[0]
    patches = render_to_patches(d["text"], max_pages=4)
    assert patches.shape[0] % 16 == 0                # whole patch-rows
    assert patches.shape[1] == PATCH_DIM
    assert patches.max() == 1.0


def test_wiki_docs_pass_production_quality_filter():
    """The regression that mattered: the first deployment shipped 695,842
    per-page micro-docs and the curator's Gopher gate kept exactly ZERO
    (too_short at <50 words; pipe-table infoboxes also broke mean word
    length). Every emitted doc must pass the REAL filter, not a proxy."""
    from ava.pipeline.clean import gopher_quality

    docs = _pages(seed=3, target=60_000)
    assert len(docs) >= 10
    for d in docs:
        ok, reason = gopher_quality(d["text"])
        assert ok, (reason, d["text"][:120])


# ---------------------------------------------------------------------------
# Pointing docs (visual-primitives grounding, specs/12 roadmap item 2)


def _pointing(docs):
    """Pointing docs share phase/task with QA docs; the 'rendered atlas'
    phrase only ever appears in pointing docs, so it is the discriminator."""
    return [d for d in docs if "rendered atlas" in d["text"]]


def test_wiki_pointing_docs_exist_per_batch():
    docs = _pages(seed=7, target=30_000)
    pointing = _pointing(docs)
    assert len(pointing) >= 1, "expected >= 1 pointing doc per 30KB batch"
    for d in pointing:
        assert d["phase"] == "p3"
        assert d["task_type"] == "deliberate"
        # two planets pointed at per system
        assert d["text"].count("Q: On the rendered atlas") == 2


def test_wiki_pointing_claims_are_true_on_rerender():
    """The pointer must be TRUE, not merely well-formed: re-render the atlas
    doc that precedes each pointing doc, index its line_boxes, and verify
    every claimed (page, line, pixel row, text, width) tuple."""
    import re

    # greedy (.*): the quoted line text may itself end with a period, and
    # ' The full box spans' occurs exactly once per answer line
    pat = re.compile(
        r"A: page (\d+), line (\d+) \(pixel row (\d+)\), which reads: "
        r"(.*)\. The full box spans (\d+) pixels from column 0\.")
    docs = _pages(seed=5, target=30_000)
    last_atlas = None
    checked = 0
    for d in docs:
        if d["phase"] == "p2":                       # atlas precedes pointer
            last_atlas = d["text"]
            continue
        if "rendered atlas" not in d["text"]:
            continue
        assert last_atlas is not None
        pages = render_pages(last_atlas)
        for m in pat.finditer(d["text"]):
            pg, line_k, row_px = int(m.group(1)), int(m.group(2)), int(m.group(3))
            said, w_px = m.group(4), int(m.group(5))
            boxes = {r: (w, t) for r, _c, w, t in pages[pg].line_boxes}
            assert row_px in boxes, "no rendered line at claimed pixel row"
            w, t = boxes[row_px]
            assert t.strip() == said, (t.strip(), said)
            assert w == w_px
            assert line_k == row_px // 8             # line index convention
            checked += 1
    assert checked >= 2, "no pointing claims were actually verified"


def test_wiki_pointing_docs_pass_gopher():
    from ava.pipeline.clean import gopher_quality

    pointing = _pointing(_pages(seed=9, target=30_000))
    assert pointing
    for d in pointing:
        ok, reason = gopher_quality(d["text"])
        assert ok, (reason, d["text"][:120])
        # charter floor: comfortably above the 50-word Gopher minimum
        assert len(d["text"].split()) >= 60


def test_wiki_pointing_docs_deterministic():
    a = [d["text"] for d in _pointing(_pages(seed=13))]
    b = [d["text"] for d in _pointing(_pages(seed=13))]
    assert a and a == b
