"""Pipeline status helpers for the live dashboard (read-only, cheap)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from ava.pipeline.flow import free_gb
from ava.pipeline.manifest import Manifest

_STATES = (
    "RAW",
    "CLAIMED_CURATE",
    "PACKED",
    "CLAIMED_TRAIN",
    "CONSUMED",
    "DELETED",
    "FAILED",
)


def _reports_dir() -> Path:
    return Path(os.environ.get("AVA_REPORTS_DIR", "/reports"))


def _state_db() -> str:
    return os.environ.get("AVA_STATE_DB", "/state/manifest.db")


def _ckpt_dir() -> Path:
    return Path(os.environ.get("AVA_CKPT_DIR", "/ckpt"))


def _tail_jsonl(path: Path, n: int = 120) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        # Read last ~256KB then take last n lines — cheap for growing files.
        size = path.stat().st_size
        with open(path, "rb") as f:
            f.seek(max(0, size - 262_144))
            raw = f.read().decode("utf-8", errors="replace")
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        out: list[dict[str, Any]] = []
        for ln in lines[-n:]:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
        return out
    except OSError:
        return []


def collect_status(preset: str | None = None) -> dict[str, Any]:
    preset = preset or os.environ.get("AVA_PRESET", "nano")
    db = _state_db()
    reports = _reports_dir()
    ckpt = _ckpt_dir()

    by_state = {s: 0 for s in _STATES}
    tokens_by_phase: dict[str, int] = {str(p): 0 for p in range(6)}
    raw_bytes = 0
    tok_sha = None
    total = 0
    manifest_ok = True
    manifest_err = None
    try:
        with Manifest(db) as m:
            counts = m.counts_by_state()
            for k, v in counts.items():
                by_state[k] = int(v)
            total = sum(by_state.values())
            raw_bytes = int(m.raw_bytes())
            tok_sha = (m.tokenizer_sha() or "")[:12] or None
            for p in range(6):
                tokens_by_phase[str(p)] = int(m.tokens_ready(p))
    except Exception as e:  # noqa: BLE001 — dashboard must never 500 the server
        manifest_ok = False
        manifest_err = str(e)

    metrics_path = reports / f"metrics_{preset}.jsonl"
    metrics = _tail_jsonl(metrics_path, 150)
    last = metrics[-1] if metrics else None

    latest_ptr = ckpt / "latest"
    latest_target = None
    if latest_ptr.is_file():
        try:
            latest_target = latest_ptr.read_text(encoding="utf-8").strip()
        except OSError:
            latest_target = None

    ckpt_files = []
    if ckpt.is_dir():
        try:
            for p in sorted(ckpt.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)[:8]:
                ckpt_files.append({
                    "name": p.name,
                    "mb": round(p.stat().st_size / (1024 * 1024), 1),
                    "mtime": int(p.stat().st_mtime),
                })
        except OSError:
            pass

    series = {
        "step": [],
        "lm_loss": [],
        "tok_s": [],
        "phase": [],
    }
    for row in metrics:
        if row.get("event") not in (None, "step"):
            # Keep step metrics; skip boot/starved noise for the curve.
            if row.get("event") != "step":
                continue
        step = row.get("step")
        loss = row.get("lm_loss", row.get("lm", row.get("total")))
        if step is None or loss is None:
            continue
        series["step"].append(step)
        series["lm_loss"].append(loss)
        series["tok_s"].append(row.get("tok_s"))
        series["phase"].append(row.get("phase"))

    last_step = None
    for row in reversed(metrics):
        if row.get("event") == "step" or ("lm" in row or "lm_loss" in row):
            last_step = row
            break
    if last_step is None and metrics:
        last_step = metrics[-1]
    # Normalize display keys for the UI
    if last_step and "lm_loss" not in last_step and "lm" in last_step:
        last_step = {**last_step, "lm_loss": last_step["lm"]}

    disk = None
    try:
        disk = round(free_gb("/"), 2)
    except Exception:  # noqa: BLE001
        disk = None

    starved = False
    for row in metrics[-20:]:
        if row.get("event") == "data_starved":
            starved = True
    if last_step and last_step.get("event") == "data_starved":
        starved = True
    # If we have a recent real step after starved events, clear the flag.
    if last_step and last_step.get("event") == "step":
        starved = False
        # unless a starved event is newer
        last_ts = float(last_step.get("ts") or 0)
        for row in metrics[-20:]:
            if row.get("event") == "data_starved" and float(row.get("ts") or 0) > last_ts:
                starved = True
                break

    return {
        "ts": time.time(),
        "preset": preset,
        "manifest": {
            "ok": manifest_ok,
            "error": manifest_err,
            "db": db,
            "total_shards": total,
            "by_state": by_state,
            "raw_bytes": raw_bytes,
            "raw_gb": round(raw_bytes / (1024 ** 3), 3),
            "tokenizer_sha": tok_sha,
            "tokens_ready_by_phase": tokens_by_phase,
        },
        "disk_free_gb": disk,
        "ckpt": {
            "latest_pointer": latest_target,
            "files": ckpt_files,
        },
        "trainer": {
            "metrics_path": str(metrics_path),
            "n_points": len(metrics),
            "last": last_step,
            "series": series,
            "data_starved": starved,
        },
        "eval": {
            "json_exists": (reports / "branch_eval_results_real.json").is_file()
            or Path("/app/reports/branch_eval_results_real.json").is_file(),
            "report_html": (reports / "index.html").is_file()
            or Path("/app/reports/index.html").is_file(),
        },
    }
