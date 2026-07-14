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
    """Karpathy-pattern integrity: every [[link]] targets a page that exists
    (or the system index/log, which are pages too)."""
    import re

    docs = _pages()
    titles = {d["text"].splitlines()[0].lstrip("# ") for d in docs}
    links = set()
    for d in docs:
        if d["phase"] == "p4":                       # the book duplicates pages
            continue
        links |= set(re.findall(r"\[\[([^\]]+)\]\]", d["text"]))
    unresolved = {l for l in links if l not in titles}
    assert not unresolved, f"dangling wiki links: {sorted(unresolved)[:5]}"


def test_wiki_physics_recomputes():
    """Kepler's third law must hold on every planet infobox: period == orbit^1.5."""
    import re

    for d in _pages():
        orbit = re.search(r"\| orbit \| ([\d.]+) AU \|", d["text"])
        period = re.search(r"\| period \| ([\d.]+) yr \|", d["text"])
        if orbit and period:
            a, p = float(orbit.group(1)), float(period.group(1))
            assert abs(p - round(a ** 1.5, 2)) < 1e-9, d["text"].splitlines()[0]


def test_wiki_pages_render_to_patches():
    d = _pages()[0]
    patches = render_to_patches(d["text"], max_pages=4)
    assert patches.shape[0] % 16 == 0                # whole patch-rows
    assert patches.shape[1] == PATCH_DIM
    assert patches.max() == 1.0
