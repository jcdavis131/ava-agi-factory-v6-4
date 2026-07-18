"""Endpoint tests for the assistant surface (spec 15 §5.1).

Sets AVA_SKIP_ENGINE_BOOT=1 before importing server so no engine boots. Drives
the route coroutines directly via asyncio.run rather than fastapi.testclient —
this repo's TestClient is pinned for CI/docker and errors under the local venv's
newer httpx (tests/test_server_endpoints.py hits the same wall here). Calling the
handlers directly exercises the real handler logic and is version-robust.
"""
from __future__ import annotations

import asyncio
import os

os.environ["AVA_SKIP_ENGINE_BOOT"] = "1"

import pytest
from fastapi import HTTPException

import server as srv


def _run(coro):
    return asyncio.run(coro)


class _StubEngine:
    """Scripted engine: returns an Action, then a final answer."""

    def __init__(self):
        self.i = 0

    def generate(self, prompt, max_tokens=64, temperature=0.8, task_type="chat", **kw):
        responses = [
            "Thought: I'll use the calculator.\nAction: add(a=2, b=40)",
            "2 + 40 = 42.",
        ]
        out = responses[min(self.i, len(responses) - 1)]
        self.i += 1
        return {"text": out, "tokens": len(out.split()), "latency_ms": 1}


def test_assistant_page_served():
    resp = _run(srv.assistant_page())
    assert b"Dottie" in resp.body


def test_assistant_status_json():
    body = _run(srv.assistant_status())
    assert body["surface"] == "ava.assistant"
    assert body["persona"] == "Dottie"
    assert body["trust"]["read_only_tools"] >= 1
    assert isinstance(body["tools"], list) and body["tools"]
    gates = [s["gate"] for s in body["demo"]["refused"]["steps"]]
    assert "denied" in gates


def test_assistant_503_when_engine_absent(monkeypatch):
    monkeypatch.setattr(srv, "get_engine", lambda: (_ for _ in ()).throw(RuntimeError("no ckpt")))
    req = srv.AssistantReq(messages=[srv.ChatMessage(role="user", content="2+40?")])
    with pytest.raises(HTTPException) as ei:
        _run(srv.assistant(req))
    assert ei.value.status_code == 503
    assert "unavailable" in ei.value.detail


def test_assistant_happy_path_with_stub_engine(monkeypatch):
    monkeypatch.setattr(srv, "get_engine", lambda: _StubEngine())
    req = srv.AssistantReq(messages=[srv.ChatMessage(role="user", content="2+40?")], max_steps=3)
    body = _run(srv.assistant(req))
    assert body["content"] == "2 + 40 = 42."
    assert body["steps"][0]["observation"] == "42"
    assert body["steps"][0]["gate"] == "ok"


def test_assistant_empty_messages_422():
    req = srv.AssistantReq(messages=[])
    with pytest.raises(HTTPException) as ei:
        _run(srv.assistant(req))
    assert ei.value.status_code == 422


def test_require_assistant_token(monkeypatch):
    # auth disabled when env unset
    monkeypatch.delenv("AVA_ASSISTANT_TOKEN", raising=False)
    assert srv._require_assistant_token(None) is None
    # enabled -> missing/invalid rejected, correct accepted
    monkeypatch.setenv("AVA_ASSISTANT_TOKEN", "s3cret")
    with pytest.raises(HTTPException) as ei:
        srv._require_assistant_token(None)
    assert ei.value.status_code == 401
    with pytest.raises(HTTPException) as ei:
        srv._require_assistant_token("Bearer nope")
    assert ei.value.status_code == 403
    assert srv._require_assistant_token("Bearer s3cret") is None
