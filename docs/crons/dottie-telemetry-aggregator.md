---
id: dottie-telemetry-aggregator
enabled: true
mode: task
schedule:
  kind: interval
  timezone: UTC
  at: 2026-07-16T12:00:00
  every: 1h
metadata:
  created_by: continuous_system
  note: HOME only
---
# Dottie Telemetry Aggregator — Hourly for Control Dash
> Solo personal project, no connection to employer, built with public/free-tier only — HOME only

Purpose: Aggregate `reports/dottie_telemetry.jsonl` into `reports/dottie_live_status.json` for online Dottie Control Plane dashboard.

**Logging:**
- Reads `reports/dottie_telemetry.jsonl` last 1000 events
- Writes `reports/dottie_live_status.json` summary + latest_per_mode + totals + health
- Own telemetry: `log_event("telemetry_aggregator", updated=..., modes=..., duration_s=...)`
- File: `logs/cron-dottie-telemetry_aggregator.log`

**Steps:**
1. `cd ~/workspace/ava-agi-factory-v6-4`
2. `python3 scripts/dottie_continuous_loop.py --mode aggregate`

Structure of `dottie_live_status.json`:
- updated (ISO)
- disclaimer
- last_expansion: timestamp 2026-07-16T15:56:01.603296+00:00, tokens 500034, docs 5045, shards ["packed_20260716_155535_00081_6671.jsonl.gz"], manifest manifest_20260716_155535.jsonl, sha12 d8cb5a396dbf, gdrive_folder_id 19tqzjB-ofqKmx1w6S4qLNB_jAEa6s3ve, total_shards 74
- totals_last_1000, latest_per_mode, by_mode_counts, recent_events, system_health

**Dash:** Fetches https://raw.githubusercontent.com/jcdavis131/ava-agi-factory-v6-4/main/reports/dottie_live_status.json + local fallback, and STATUS.json, plus telemetry jsonl tail.

**Footer:** Solo personal project, no connection to employer, built with public/free-tier only
