"""Unit tests for dashboard status helpers (no Docker / no live DB)."""

from __future__ import annotations

from ava.pipeline_status import _SERIES_FIELDS, current_run_series, full_run_series


def test_current_run_series_drops_pre_restart_history():
    metrics = [
        {"event": "step", "step": 450, "lm": 0.12, "tok_s": 12000, "phase": 0},
        {"event": "step", "step": 460, "lm": 0.11, "tok_s": 12100, "phase": 0},
        {"event": "step", "step": 470, "lm": 0.10, "tok_s": 12000, "phase": 0},
        # CUDA restart — step resets
        {"event": "step", "step": 1, "lm": 10.5, "tok_s": None, "phase": 0},
        {"event": "step", "step": 10, "lm": 8.0, "tok_s": 11000, "phase": 0},
        {"event": "step", "step": 100, "lm": 0.30, "tok_s": 11200, "phase": 0},
    ]
    series = current_run_series(metrics)
    assert series["step"] == [1, 10, 100]
    assert series["lm_loss"][0] == 10.5
    assert series["lm_loss"][-1] == 0.30
    assert len(series["tok_s"]) == 3


def test_current_run_series_empty():
    assert current_run_series([]) == {
        "step": [],
        "lm_loss": [],
        "phase": [],
        "total": [],
        **{k: [] for k in _SERIES_FIELDS},
    }


def test_current_run_series_carries_aux_loss_and_optimizer_fields():
    metrics = [
        {"event": "step", "step": 1, "lm": 9.0, "total": 9.4, "phase": 0,
         "grad_norm": 0.8, "lr": 1e-4, "report": 0.2, "broadcast": 0.1,
         "selectivity": 0.05, "modulation": 0.03, "half_life": 0.02,
         "inter_mi": 0.01, "routing": 0.09,
         "verbalizable_mass": 0.06, "broadcast_strength": 0.2},
    ]
    series = current_run_series(metrics)
    assert series["grad_norm"] == [0.8]
    assert series["lr"] == [1e-4]
    assert series["routing"] == [0.09]
    assert series["verbalizable_mass"] == [0.06]


def test_current_run_series_ignores_non_step_events():
    metrics = [
        {"event": "model_built", "preset": "mini"},
        {"event": "step", "step": 1, "lm": 9.0, "phase": 0},
        {"event": "checkpoint", "step": 100, "path": "/ckpt/step_100.pt"},
        {"event": "step", "step": 100, "lm": 0.3, "tok_s": 10000, "phase": 0},
    ]
    series = current_run_series(metrics)
    assert series["step"] == [1, 100]


def test_full_run_series_keeps_pre_restart_history_and_flags_restarts():
    """Unlike current_run_series, full_run_series must NOT drop the segment
    before a restart — that's the whole point of the "loss landscape doesn't
    show the full timeline" fix. cum_step keeps counting up across the
    restart instead of jumping backward (raw step resets to 1), so a chart
    plotting cum_step never has to draw a line that jumps backward."""
    metrics = [
        {"event": "step", "step": 450, "lm": 0.12, "ts": 100.0, "phase": 0},
        {"event": "step", "step": 460, "lm": 0.11, "ts": 110.0, "phase": 0},
        # CUDA restart — step resets
        {"event": "step", "step": 1, "lm": 10.5, "ts": 200.0, "phase": 0},
        {"event": "step", "step": 10, "lm": 8.0, "ts": 210.0, "phase": 0},
    ]
    result = full_run_series(metrics)
    assert result["series"]["step"] == [450, 460, 1, 10]
    assert result["series"]["ts"] == [100.0, 110.0, 200.0, 210.0]
    assert result["series"]["cum_step"] == [450, 460, 461, 470]
    assert result["series"]["cum_step"] == sorted(result["series"]["cum_step"]), "cum_step must be non-decreasing"
    assert result["restarts"] == [{"cum_step": 461, "ts": 200.0}]


def test_full_run_series_empty():
    result = full_run_series([])
    assert result["series"]["step"] == []
    assert result["series"]["cum_step"] == []
    assert result["series"]["ts"] == []
    assert result["restarts"] == []


def test_full_run_series_downsamples_but_keeps_latest_point():
    metrics = [
        {"event": "step", "step": i, "lm": 1.0, "ts": float(i)}
        for i in range(1, 2001)
    ]
    result = full_run_series(metrics)
    n = len(result["series"]["step"])
    assert n <= 610, f"expected downsampling to roughly _FULL_SERIES_MAX_POINTS, got {n}"
    assert result["series"]["step"][-1] == 2000, "must always keep the most recent point"


# ---------------------------------------------------------------------------
# Staleness: the false-alarm regressions behind 'Trainer Stale' during every
# post-crash recovery window.

from ava.pipeline_status import _mode, _stale_threshold_s


def test_stale_threshold_falls_back_to_pre_restart_cadence():
    """Fresh run (<2 step rows) must inherit the pre-restart cadence, not
    collapse to the 180s floor while a P2 recovery takes ~15 min."""
    all_rows = [
        {"event": "step", "step": 3690, "ts": 1000.0, "tokens": 967_311_360,
         "tok_s": 3000, "lm": 4.4},
        {"event": "step", "step": 3700, "ts": 1874.0, "tokens": 969_932_800,
         "tok_s": 3000, "lm": 4.4},
        {"event": "model_built", "ts": 1900.0},
        {"event": "resumed", "ts": 1910.0, "step": 3700},
    ]
    # tokens delta 2_621_440 at 3000 tok/s => ~874s expected, x2.5 => ~2185s
    assert _stale_threshold_s([], all_rows=all_rows) > 2000
    # without history the floor still applies
    assert _stale_threshold_s([]) == 180.0


def test_mode_recovering_between_resume_and_first_step():
    mode = _mode(last_step={"event": "step", "step": 3700}, starved=False,
                 age_s=120.0, stale_after_s=2185.0, gates=[], recovering=True)
    assert mode["id"] == "recovering"


def test_mode_stale_wins_over_recovering_when_age_exceeds_threshold():
    mode = _mode(last_step={"event": "step", "step": 3700}, starved=False,
                 age_s=5000.0, stale_after_s=2185.0, gates=[], recovering=True)
    assert mode["id"] == "stale"


def test_mode_training_when_stepping_normally():
    mode = _mode(last_step={"event": "step", "step": 3700}, starved=False,
                 age_s=300.0, stale_after_s=2185.0, gates=[], recovering=False)
    assert mode["id"] == "training"


# ---------------------------------------------------------------------------
# Battery/power throttling detection: 14.5h of the run's 'silent gaps' were a
# 17-22W GPU cap on battery, invisible until tok/s is compared to its median.

from ava.pipeline_status import _throttle_state


def _steps(toks):
    return [{"event": "step", "step": 10 * (i + 1), "phase": 2, "tok_s": t,
             "lm": 4.4, "ts": float(i)} for i, t in enumerate(toks)]


def test_throttle_detected_on_collapse():
    metrics = _steps([3000, 3100, 2900, 3050, 480])
    throttled, detail = _throttle_state(metrics)
    assert throttled and "battery" in detail


def test_no_throttle_at_normal_speed():
    assert _throttle_state(_steps([3000, 3100, 2900, 3050, 2950])) == (False, "")


def test_no_throttle_without_history():
    assert _throttle_state(_steps([3000, 400]))[0] is False  # <3 history rows


def test_throttle_ignores_other_phase_cadence():
    # phase transition halves tok/s legitimately: history is same-phase only
    metrics = _steps([9000, 9100, 9200, 9050])
    for r in metrics:
        r["phase"] = 1
    metrics += [{"event": "step", "step": 60, "phase": 2, "tok_s": 3000,
                 "lm": 4.9, "ts": 99.0}]
    assert _throttle_state(metrics)[0] is False


def test_mode_throttled_wins_over_training():
    mode = _mode(last_step={"event": "step", "step": 3700}, starved=False,
                 age_s=300.0, stale_after_s=2185.0, gates=[],
                 throttled=True, throttle_detail="tok/s 480 is 16% of median")
    assert mode["id"] == "throttled"
