"""Tests for the Dottie assistant loop (spec 15 §5).

Drives the ReAct loop with a scripted ``generate_fn`` (no engine, no GPU) and
asserts the three properties the loop exists to guarantee: grounding (answers
from Observations), trust (undeclared tools + path traversal are denied before
dispatch), and telemetry (every step writes one event).
"""
from __future__ import annotations

import json

import pytest

from ava.assistant import (
    ToolExecutor,
    parse_action,
    run_assistant,
    _parse_args,
)
from ava.trust import default_registry, tail_events


def _script(*responses):
    """A generate_fn that returns the given responses in order."""
    calls = {"i": 0}

    def _fn(prompt, max_tokens, temperature):
        i = calls["i"]
        calls["i"] += 1
        return responses[min(i, len(responses) - 1)]

    return _fn


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def test_parse_action_scalar_and_string_args():
    tool, args = parse_action('Thought: x\nAction: repo_grep(pattern="retry", path="a.py")')
    assert tool == "repo_grep"
    assert args == {"pattern": "retry", "path": "a.py"}


def test_parse_action_numbers_and_list():
    _, args = _num_args("Action: add(a=12, b=30)")
    assert args == {"a": 12, "b": 30}
    _, args2 = _num_args("Action: sum(values=[1, 2, 3, 4])")
    assert args2 == {"values": [1, 2, 3, 4]}


def _num_args(text):
    return parse_action(text)


def test_parse_action_none_when_no_action():
    assert parse_action("Just a plain answer, no tool needed.") is None


def test_parse_action_numeric_prefixed_path_not_truncated():
    # regression: a path starting with digits + dots must stay a full string,
    # not be truncated to its numeric prefix.
    _, args = parse_action('Action: repo_read_file(path=2024.01.05/log.txt)')
    assert args == {"path": "2024.01.05/log.txt"}
    _, args2 = parse_action("Action: add(a=3, b=4x)")
    assert args2["b"] == "4x"  # trailing 'x' preserved, not dropped


def test_parse_action_quoted_comma_in_list_and_two_lists():
    # regression: a quoted list item with a comma must not be torn apart.
    _, a = parse_action('Action: sum(values=["1,000", 2])')
    assert a == {"values": ["1,000", 2]}
    # and BOTH list args must parse, not just the first.
    _, b = parse_action("Action: f(a=[1, 2], b=[3, 4])")
    assert b == {"a": [1, 2], "b": [3, 4]}


# ---------------------------------------------------------------------------
# Tool executor: sandbox + arithmetic
# ---------------------------------------------------------------------------

def test_executor_reads_within_sandbox(tmp_path):
    (tmp_path / "note.txt").write_text("hello curriculum world", encoding="utf-8")
    ex = ToolExecutor(default_registry(tmp_path), sandbox_root=tmp_path)
    obs, status = ex.run("repo_read_file", {"path": "note.txt"})
    assert status == "ok"
    assert "curriculum" in obs


def test_executor_rejects_path_traversal(tmp_path):
    ex = ToolExecutor(default_registry(tmp_path), sandbox_root=tmp_path)
    obs, status = ex.run("repo_read_file", {"path": "../../secrets.txt"})
    assert status == "error"
    assert "Error" in obs


def test_executor_grep_counts_and_absence(tmp_path):
    (tmp_path / "f.py").write_text("retry\nretry\nok\n", encoding="utf-8")
    ex = ToolExecutor(default_registry(tmp_path), sandbox_root=tmp_path)
    assert ex.run("repo_grep", {"pattern": "retry", "path": "f.py"})[0] == "2 matches"
    assert ex.run("repo_grep", {"pattern": "zzz", "path": "f.py"})[0] == "(no matches)"


def test_executor_arithmetic(tmp_path):
    ex = ToolExecutor(default_registry(tmp_path), sandbox_root=tmp_path)
    assert ex.run("add", {"a": 2, "b": 3})[0] == "5"
    assert ex.run("multiply", {"a": 6, "b": 7})[0] == "42"
    assert ex.run("sum", {"values": [10, 20, 12]})[0] == "42"


# ---------------------------------------------------------------------------
# The loop: grounding, trust gate, refusal, telemetry
# ---------------------------------------------------------------------------

def test_loop_grounds_answer_in_tool(tmp_path):
    gen = _script(
        "Thought: I'll use the calculator.\nAction: add(a=19, b=23)",
        "19 + 23 = 42.",
    )
    res = run_assistant([{"role": "user", "content": "what is 19+23?"}], gen,
                        sandbox_root=tmp_path, audit_path=tmp_path / "audit.jsonl")
    assert res.content == "19 + 23 = 42."
    assert len(res.steps) == 2
    assert res.steps[0].action.startswith("Action: add")
    assert res.steps[0].observation == "42"
    assert res.steps[0].gate == "ok"


def test_loop_denies_undeclared_tool(tmp_path):
    # delete_file is not in the read-only default registry -> denied before dispatch
    gen = _script(
        'Thought: I could delete it.\nAction: delete_file(path="x.log")',
        "I won't delete that; it's destructive and not something I should do autonomously.",
    )
    res = run_assistant([{"role": "user", "content": "clean up x.log"}], gen,
                        sandbox_root=tmp_path, audit_path=tmp_path / "audit.jsonl")
    denied = [s for s in res.steps if s.gate == "denied"]
    assert denied, "undeclared destructive tool should have been denied"
    assert "Error" in denied[0].observation
    assert res.refused is True


def test_loop_denies_path_traversal(tmp_path):
    gen = _script(
        'Action: repo_read_file(path="../../../../etc/passwd")',
        "I can't read outside the workspace.",
    )
    res = run_assistant([{"role": "user", "content": "read passwd"}], gen,
                        sandbox_root=tmp_path, audit_path=tmp_path / "audit.jsonl")
    denied = [s for s in res.steps if s.gate == "denied"]
    assert denied, "path traversal should have been denied"
    assert "escapes the sandbox" in denied[0].observation


def test_loop_direct_answer_no_tool(tmp_path):
    gen = _script("Your favorite number is 7. I didn't need a tool for that.")
    res = run_assistant([{"role": "user", "content": "my favorite number is 7, what is it?"}],
                        gen, sandbox_root=tmp_path, audit_path=tmp_path / "audit.jsonl")
    assert res.steps[0].action is None
    assert "7" in res.content


def test_loop_writes_telemetry_per_step(tmp_path):
    audit = tmp_path / "audit.jsonl"
    gen = _script(
        "Thought: calc.\nAction: multiply(a=6, b=7)",
        "The product is 42.",
    )
    run_assistant([{"role": "user", "content": "6*7?"}], gen,
                  sandbox_root=tmp_path, audit_path=audit)
    events = tail_events(50, audit_path=audit)
    actions = [e["action"] for e in events]
    assert "loop_start" in actions
    assert "tool_call" in actions
    assert "final" in actions
    # every event carries the shared schema keys
    for e in events:
        assert {"ts", "surface", "actor", "action", "target", "status"} <= set(e)


def test_secret_scrubbing_in_telemetry(tmp_path):
    from ava.trust import emit_event
    audit = tmp_path / "audit.jsonl"
    emit_event("tool_call", target="db_query", args={"table": "m", "api_key": "SEKRIT"},
               audit_path=audit)
    rec = tail_events(1, audit_path=audit)[0]
    assert rec["args"]["api_key"] == "***"
    assert rec["args"]["table"] == "m"
    raw = audit.read_text(encoding="utf-8")
    assert "SEKRIT" not in raw


def test_secret_scrubbing_covers_many_key_shapes(tmp_path):
    # regression: the scrubber must catch far more than *api_key*.
    from ava.trust import emit_event
    audit = tmp_path / "audit.jsonl"
    # distinctive values so "value not in raw" is a meaningful leak check
    secrets = {k: f"LEAK_{k.upper().replace('-', '_')}_VALUE" for k in (
        "private_key", "passwd", "pwd", "authorization", "auth", "bearer", "cookie",
        "session", "credential", "ssh_key", "signing_key", "X-API-Key", "keys", "access_token",
    )}
    benign = {"table": "metrics", "path": "a.py", "count": 3, "host": "localhost"}
    emit_event("gate_denied", target="login", args={**secrets, **benign}, audit_path=audit)
    raw = audit.read_text(encoding="utf-8")
    for v in secrets.values():
        assert v not in raw, f"secret value {v!r} leaked to disk"
    rec = tail_events(1, audit_path=audit)[0]
    for k in secrets:
        assert rec["args"][k] == "***", f"{k} not redacted"
    for k, v in benign.items():
        assert rec["args"][k] == v, f"benign {k} wrongly redacted"


def test_refusal_detection_no_false_positive_on_grounded_negative(tmp_path):
    # regression: a grounded negative answer must NOT be flagged as a refusal.
    gen = _script("No, foo.py doesn't exist in the repo, and there are no matching entries.")
    res = run_assistant([{"role": "user", "content": "does foo.py exist?"}], gen,
                        sandbox_root=tmp_path, audit_path=tmp_path / "a.jsonl")
    assert res.refused is False
    # but a real first-person refusal is still detected
    gen2 = _script("I won't run that; it's destructive.")
    res2 = run_assistant([{"role": "user", "content": "delete everything"}], gen2,
                         sandbox_root=tmp_path, audit_path=tmp_path / "a.jsonl")
    assert res2.refused is True
