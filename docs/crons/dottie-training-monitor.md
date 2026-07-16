---
id: dottie-training-monitor
enabled: true
mode: task
schedule:
  kind: interval
  timezone: UTC
  at: 2026-07-16T00:00:00
  every: 30m
metadata:
  created_by: continuous_system
---
# Dottie Training Monitor — 30m with Telemetry
> Solo personal project, no connection to employer, built with public/free-tier only — HOME only

Purpose: Poll Dottie training logs every 30m, parse loss, steps, eval scores, log to telemetry for Control Dash.

**Logging:**
- `dottie/telemetry.py` → `reports/dottie_telemetry.jsonl` mode `training_monitor`
- File: `logs/cron-dottie-training_monitor.log`

**Steps:**
1. `cd ~/workspace/ava-agi-factory-v6-4`
2. `python3 scripts/dottie_continuous_loop.py --mode monitor >> logs/cron-dottie-training_monitor.log 2>&1`

**Footer:** Solo personal project, no connection to employer, built with public/free-tier only
