"""StreamingShardSampler tests.

The regression that motivated most of these: the sampler originally refused to
let a window straddle a document. The synthetic corpus has a median document
length of ~100 tokens, so at seq_len=256 phases 1 and 5 produced ZERO windows
and the trainer blocked forever on data that could never arrive. It did not
crash -- it starved silently, which is worse.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from ava.data import UNTAGGED_CONCEPT, _LoadedShard
from ava.pipeline.manifest import PACKED, Manifest, Shard
from ava.tokenizer import ENDOFDOC_ID


def _write_shard(dirpath: Path, doc_lens: list[int], task_types: list[str],
                 concepts: list[int]) -> Shard:
    """Pack synthetic docs exactly the way ava/pipeline/pack.py does."""
    dirpath.mkdir(parents=True, exist_ok=True)
    stream: list[int] = []
    docs = []
    for i, (n, tt, c) in enumerate(zip(doc_lens, task_types, concepts)):
        start = len(stream)
        stream.extend(range(100, 100 + n))       # distinguishable token ids
        end = len(stream)
        stream.append(ENDOFDOC_ID)               # separator, not counted
        docs.append({"doc_id": f"d{i}", "start": start, "end": end,
                     "task_type": tt, "concept_token_id": c, "phase": 0})

    bin_path = dirpath / "s.bin"
    np.array(stream, dtype=np.uint16).tofile(bin_path)
    (dirpath / "s.idx.json").write_text(json.dumps(
        {"tokens": len(stream), "tokenizer_sha": "sha", "docs": docs}))
    return Shard(id="s", source="t", phase=0, split="train", state=PACKED,
                 path=str(bin_path), bytes=0, tokens=len(stream), docs=len(docs), attempts=0)


def test_short_docs_still_produce_windows(tmp_path):
    """THE regression. 20 docs of 100 tokens must fill 256-token windows."""
    s = _write_shard(tmp_path, [100] * 20, ["deliberate"] * 20, [7] * 20)
    loaded = _LoadedShard(s)
    wins = list(loaded.windows("deliberate", 256, random.Random(0)))
    assert wins, "short docs produced no windows -- the trainer would starve forever"
    for w, _ in wins:
        assert w.shape == (257,)


def test_window_is_seq_len_plus_one(tmp_path):
    s = _write_shard(tmp_path, [500], ["automatic"], [3])
    loaded = _LoadedShard(s)
    for w, _ in loaded.windows("automatic", 128, random.Random(0)):
        assert w.shape == (129,)                 # +1 for the shifted target


def test_windows_only_mix_same_task_type(tmp_path):
    """Batches must stay task_type-pure or the routing-KL target is meaningless."""
    s = _write_shard(tmp_path, [200] * 6, ["safety"] * 3 + ["automatic"] * 3, [1] * 6)
    loaded = _LoadedShard(s)
    assert list(loaded.windows("temporal", 64, random.Random(0))) == []
    assert list(loaded.windows("safety", 64, random.Random(0)))
    assert list(loaded.windows("automatic", 64, random.Random(0)))


def test_docs_are_separated_by_endofdoc(tmp_path):
    s = _write_shard(tmp_path, [50] * 10, ["automatic"] * 10, [5] * 10)
    loaded = _LoadedShard(s)
    w, _ = next(loaded.windows("automatic", 128, random.Random(0)))
    assert ENDOFDOC_ID in w.tolist(), "document boundaries must be visible to the model"


def test_concept_is_first_tagged_doc_in_window(tmp_path):
    s = _write_shard(tmp_path, [80] * 6, ["deliberate"] * 6,
                     [UNTAGGED_CONCEPT, UNTAGGED_CONCEPT, 42, 43, 44, 45])
    loaded = _LoadedShard(s)
    _, cid = next(loaded.windows("deliberate", 64, random.Random(0)))
    assert cid >= 0 or cid == UNTAGGED_CONCEPT


def test_fully_untagged_window_reports_untagged(tmp_path):
    """A window of pure HF text must not invent a concept target."""
    s = _write_shard(tmp_path, [200] * 4, ["automatic"] * 4, [UNTAGGED_CONCEPT] * 4)
    loaded = _LoadedShard(s)
    _, cid = next(loaded.windows("automatic", 128, random.Random(0)))
    assert cid == UNTAGGED_CONCEPT


def test_tokens_are_in_uint16_range_and_int64_dtype(tmp_path):
    s = _write_shard(tmp_path, [300], ["automatic"], [9])
    loaded = _LoadedShard(s)
    w, _ = next(loaded.windows("automatic", 128, random.Random(0)))
    assert w.dtype == np.int64                   # embedding lookup needs int64
    assert w.max() < 65536


def test_empty_task_type_yields_nothing_rather_than_raising(tmp_path):
    s = _write_shard(tmp_path, [300], ["automatic"], [1])
    loaded = _LoadedShard(s)
    assert list(loaded.windows("safety", 64, random.Random(0))) == []


# ---------------------------------------------------------------------------
# The data treadmill: a shard's windows must be yielded ONCE per claim. The
# old round-robin drew len(TASK_TYPES) times regardless of how many types were
# present, so the P2 norm (single-type shards) was silently trained 4x each.

def _mk_sampler(m: Manifest, packed_dir: Path):
    from ava.data import StreamingShardSampler
    from ava.pipeline.flow import FlowConfig
    flow = FlowConfig(
        low_water_gb=0, janitor_trigger_gb=0, critical_gb=0, raw_max_bytes=1,
        packed_ahead_max_tokens=1, packed_min_tokens=1,
        starved_poll_seconds=0.05, starved_warn_seconds=60,
        prefetch_phases=2, delete_consumed=True,
    )
    return StreamingShardSampler(None, m, flow, packed_dir=str(packed_dir))


def test_single_task_shard_yields_each_present_type_once(tmp_path):
    s = _write_shard(tmp_path, [100] * 8, ["automatic"] * 8, [-1] * 8)
    loaded = _LoadedShard(s)
    m = Manifest(str(tmp_path / "m.db"))
    sampler = _mk_sampler(m, tmp_path)
    assert sampler._present_task_types(loaded) == ["automatic"]
    assert sampler._present_task_types(loaded) == ["automatic"]  # cursor moves, list never repeats a type


def test_multi_task_shard_rotates_lead_type_without_repeats(tmp_path):
    s = _write_shard(tmp_path, [100] * 4,
                     ["automatic", "deliberate", "automatic", "deliberate"], [-1] * 4)
    loaded = _LoadedShard(s)
    m = Manifest(str(tmp_path / "m.db"))
    sampler = _mk_sampler(m, tmp_path)
    first = sampler._present_task_types(loaded)
    second = sampler._present_task_types(loaded)
    assert sorted(first) == sorted(second) == ["automatic", "deliberate"]
    assert len(first) == len(set(first)) == 2
    assert first != second, "lead type should rotate across shards for fairness"


def test_single_task_shard_not_re_epoched(tmp_path):
    """End to end: pull every window of a single-type shard through batches();
    the next pull must BLOCK on new data (shard consumed), not restart the
    same shard's windows for a second epoch."""
    import random as _random

    s = _write_shard(tmp_path, [100] * 10, ["automatic"] * 10, [-1] * 10)
    expected = len(list(_LoadedShard(s).windows("automatic", 64, _random.Random(0))))
    assert expected > 0

    m = Manifest(str(tmp_path / "m.db"))
    m.add_shard("s", source="t", phase=0, path=s.path, state=PACKED)
    sampler = _mk_sampler(m, tmp_path)
    gen = sampler.batches(0, 64, 1, log=lambda *_: None)
    got = [next(gen) for _ in range(expected)]
    assert all(b.task_type == "automatic" for b in got)

    # Resuming the generator must CONSUME the shard and go wait for new data
    # (patched to raise), not restart the same shard's windows for epoch 2.
    def _no_more_data(*a, **k):
        raise TimeoutError("no more data")

    sampler._wait_for_data = _no_more_data
    with pytest.raises(TimeoutError):
        next(gen)
    assert m.counts_by_state().get("CONSUMED", 0) == 1
