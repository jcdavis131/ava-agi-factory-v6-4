"""Trajectory-level verification for the tool-use curriculum (spec 15 §4).

The generator is never trusted: this file re-parses the rendered ReAct text and
independently re-derives what each tool's Observation *must* be, then asserts the
generator agrees. It also pins the two contracts the curriculum exists to honor:

  * every ``Action:`` line matches the production parser regex
    (``AgenticOS/ava_bridge.py::_ACTION_RE``), and
  * the negative/refuse family (L4) issues no tool call, while the multi-step
    family (L1) issues more than one.
"""
from __future__ import annotations

import re

import pytest

from ava.datagen.tool_curriculum import (
    ToolUseGenerator,
    l0_arith_doc,
    l1_listdir_sum_doc,
    l1_read_then_multiply_doc,
    l1_two_reads_add_doc,
    l2_empty_giveup_doc,
    l4_direct_answer_doc,
    l4_refuse_destructive_doc,
    l3_select_doc,
)

# Byte-identical to AgenticOS/ava_bridge.py::_ACTION_RE — the production parser.
_ACTION_RE = re.compile(r"Action:\s*([a-zA-Z_][a-zA-Z0-9_]*)\((.*)\)\s*$", re.M)
_OBS_RE = re.compile(r"^Observation:\s*(.*)$", re.M)

SMALL_TARGET = 300_000


def _collect(seed=7):
    return list(ToolUseGenerator(seed=seed).generate(SMALL_TARGET))


# ---------------------------------------------------------------------------
# Turn parsing + independent arithmetic re-derivation.
# ---------------------------------------------------------------------------

def _parse_turns(text: str) -> list[tuple[str, str]]:
    """Split rendered text into (role, content) turns on the marker lines."""
    turns: list[tuple[str, str]] = []
    role = None
    buf: list[str] = []
    for line in text.split("\n"):
        if line == "<|user|>":
            if role is not None:
                turns.append((role, "\n".join(buf)))
            role, buf = "user", []
        elif line == "<|assistant|>":
            if role is not None:
                turns.append((role, "\n".join(buf)))
            role, buf = "assistant", []
        else:
            buf.append(line)
    if role is not None:
        turns.append((role, "\n".join(buf)))
    return turns


def _parse_num(tok: str):
    tok = tok.strip()
    try:
        return int(tok)
    except ValueError:
        return float(tok)


def _parse_scalar_args(argstr: str) -> dict:
    out = {}
    for m in re.finditer(r"([a-zA-Z_]\w*)\s*=\s*(-?\d+(?:\.\d+)?)", argstr):
        out[m.group(1)] = _parse_num(m.group(2))
    return out


def _parse_values_list(argstr: str):
    m = re.search(r"values\s*=\s*\[([^\]]*)\]", argstr)
    if not m:
        return None
    inner = m.group(1).strip()
    if not inner:
        return []
    return [_parse_num(t) for t in inner.split(",")]


def _action_obs_pairs(turns):
    """Yield (tool, argstr, observation) triples by pairing each assistant
    Action with the immediately following user Observation."""
    pending = None
    for role, content in turns:
        if role == "assistant":
            m = _ACTION_RE.search(content)
            pending = (m.group(1), m.group(2)) if m else None
        elif role == "user" and pending is not None:
            mo = _OBS_RE.search(content)
            obs = mo.group(1) if mo else None
            yield (pending[0], pending[1], obs)
            pending = None


# ---------------------------------------------------------------------------
# 1. Every Action parses; at most one Action per assistant turn.
# ---------------------------------------------------------------------------

def test_every_action_parses_with_production_regex():
    docs = _collect()
    assert docs
    seen_action = False
    for d in docs:
        for role, content in _parse_turns(d["text"]):
            if role != "assistant":
                continue
            actions = _ACTION_RE.findall(content)
            assert len(actions) <= 1, (
                f"assistant turn has {len(actions)} Action lines (parser takes only the "
                f"first):\n{content}"
            )
            for name, argstr in actions:
                seen_action = True
                # tool name must be a bare identifier; argstr must not span lines
                assert re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", name)
                assert "\n" not in argstr
    assert seen_action, "corpus contained no tool calls at all"


# ---------------------------------------------------------------------------
# 2. Independent arithmetic re-derivation over the whole corpus. Whenever an
#    arithmetic tool is called, its Observation must equal the true result.
# ---------------------------------------------------------------------------

def test_arithmetic_observations_are_correct():
    docs = _collect()
    checked = 0
    for d in docs:
        turns = _parse_turns(d["text"])
        for tool, argstr, obs in _action_obs_pairs(turns):
            if obs is None:
                continue
            if tool in ("add", "subtract", "multiply"):
                a = _parse_scalar_args(argstr)
                if "a" not in a or "b" not in a:
                    continue
                expected = {"add": a["a"] + a["b"],
                            "subtract": a["a"] - a["b"],
                            "multiply": a["a"] * a["b"]}[tool]
                # obs may be a bare number; compare numerically
                assert _parse_num(obs.split()[0]) == expected, (
                    f"{tool}({argstr}) -> Observation {obs!r} != {expected}"
                )
                checked += 1
            elif tool == "sum":
                vals = _parse_values_list(argstr)
                if vals is None:
                    continue
                assert _parse_num(obs.split()[0]) == sum(vals), (
                    f"sum({argstr}) -> Observation {obs!r} != {sum(vals)}"
                )
                checked += 1
    assert checked > 0, "no arithmetic tool calls were exercised"


# ---------------------------------------------------------------------------
# 3. Grounding: the final answer echoes the final Observation's value (except
#    the no-tool refusal family, handled separately).
# ---------------------------------------------------------------------------

def test_final_answer_is_grounded_in_last_observation():
    rng_docs = _collect()
    checked = 0
    for d in rng_docs:
        if d["concept"] == "tool_refuse":
            continue  # no tool call; grounding-in-observation doesn't apply
        turns = _parse_turns(d["text"])
        obs_values = [
            _OBS_RE.search(c).group(1) for r, c in turns
            if r == "user" and _OBS_RE.search(c)
        ]
        if not obs_values:
            continue
        final_answer = turns[-1][1]
        last_obs = obs_values[-1]
        # An "absence" observation (e.g. "(no matches)") is grounded by reporting
        # absence, not by echoing a value — covered by test_empty_result_never_
        # fabricates. Skip those here.
        if last_obs.startswith("("):
            continue
        # the salient token of the last observation (first whitespace field,
        # e.g. the number or date) must appear verbatim in the final answer
        salient = last_obs.split()[0].strip('"')
        assert salient in final_answer, (
            f"[{d['concept']}] final answer does not cite last observation "
            f"{last_obs!r} (salient {salient!r}):\n{final_answer}"
        )
        checked += 1
    assert checked > 0


# ---------------------------------------------------------------------------
# 4. Structure: L1 chains are multi-step; L4 refusals issue no tool call.
# ---------------------------------------------------------------------------

def test_multistep_chains_issue_more_than_one_call():
    import random
    rng = random.Random(3)
    for builder in (l1_listdir_sum_doc, l1_read_then_multiply_doc, l1_two_reads_add_doc):
        text, _, concept = builder(rng)
        assert concept == "tool_chain"
        n_actions = len(_ACTION_RE.findall(text))
        assert n_actions >= 2, f"{builder.__name__} is not multi-step ({n_actions} calls)"


def test_refusal_family_makes_no_tool_call():
    import random
    rng = random.Random(5)
    # direct-answer negative: answerable from the message, no tool
    for _ in range(20):
        text, tt, concept = l4_direct_answer_doc(rng)
        assert concept == "tool_refuse"
        assert not _ACTION_RE.search(text), f"direct-answer doc issued a tool call:\n{text}"
        assert tt == "deliberate"
    # destructive refusal: a safety turn that declines to call the tool
    for _ in range(20):
        text, tt, concept = l4_refuse_destructive_doc(rng)
        assert tt == "safety"
        assert not _ACTION_RE.search(text), f"destructive-refuse doc issued a tool call:\n{text}"
        low = text.lower()
        assert ("won't" in low or "not" in low or "decline" in low), (
            f"destructive-refuse doc lacks a clear refusal:\n{text}"
        )


def test_empty_result_never_fabricates():
    import random
    rng = random.Random(11)
    for _ in range(20):
        text, _, concept = l2_empty_giveup_doc(rng)
        assert concept == "tool_recover"
        # two grep calls, both empty, then an honest give-up
        assert text.count("(no matches)") == 2
        final = _parse_turns(text)[-1][1].lower()
        assert ("doesn't" in final or "no definition" in final or "won't invent" in final
                or "found no" in final), f"give-up answer looks like a fabrication:\n{final}"


# ---------------------------------------------------------------------------
# 5. Tool selection presents a catalog and picks a listed tool.
# ---------------------------------------------------------------------------

def test_selection_calls_a_tool_from_the_catalog():
    import random
    rng = random.Random(17)
    for _ in range(40):
        text, _, concept = l3_select_doc(rng)
        assert concept == "tool_select"
        assert "Available tools" in text
        m = _ACTION_RE.search(text)
        assert m, f"selection doc issued no tool call:\n{text}"
        tool = m.group(1)
        # the chosen tool's name must appear in the catalog listing
        catalog = text.split("Task:")[0]
        assert tool in catalog, f"chose {tool} which isn't in the presented catalog:\n{catalog}"


# ---------------------------------------------------------------------------
# 6. Cross-repo: the production parser accepts our Action lines (skip if the
#    sibling AgenticOS repo isn't checked out).
# ---------------------------------------------------------------------------

def test_ava_bridge_parses_our_actions():
    try:
        import sys
        from pathlib import Path
        sib = Path(__file__).resolve().parent.parent.parent / "AgenticOS"
        if sib.exists() and str(sib) not in sys.path:
            sys.path.insert(0, str(sib))
        from ava_bridge import parse_react_response  # type: ignore
    except Exception:
        pytest.skip("AgenticOS/ava_bridge.py not importable")
    import random
    rng = random.Random(23)
    for builder in (l0_arith_doc, l3_select_doc, l1_read_then_multiply_doc):
        text, _, _ = builder(rng)
        # feed each assistant turn that carries an Action
        for role, content in _parse_turns(text):
            if role == "assistant" and "Action:" in content:
                parsed = parse_react_response(content)
                assert parsed.get("tool_calls"), f"ava_bridge failed to parse:\n{content}"
