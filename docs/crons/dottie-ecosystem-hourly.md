---
id: dottie-ecosystem-hourly
enabled: true
mode: task
schedule:
  kind: interval
  timezone: UTC
  at: 2026-07-16T12:00:00
  every: 1h
metadata:
  created_by: continuous_system
  note: HOME only, solo project
---
# Dottie Ecosystem Hourly — Rotate Shards + Skillbooks + OpenWiki
> Solo personal project, no connection to employer, built with public/free-tier only — HOME only

Purpose: Keep Dottie codebase/ecosystem moving hourly — rotate shards, update skillbooks, curate OpenWiki → S2 Slow hl300, log to telemetry for dash.

**Logging:**
- `dottie/telemetry.py` → `reports/dottie_telemetry.jsonl` mode `ecosystem`
- File: `logs/cron-dottie-ecosystem.log`
- Dashboard: skillbook count, free_gb, openwiki_adapter status

**Steps:**
1. `cd ~/workspace/ava-agi-factory-v6-4`
2. `mkdir -p logs reports`
3. Run ecosystem loop:
   ```
   python3 scripts/dottie_continuous_loop.py --mode ecosystem >> logs/cron-dottie-ecosystem.log 2>&1
   ```
Internally via `dottie/ecosystem_updater.py`:
- `sync_openwiki()` → check ~/.openwiki/wiki
- `rotate_shards()` → if disk >80% janitor evict CONSUMED shards keep last 2 days
- `update_skillbooks()` → validate 11 skillbooks expected
- `check_docs_links()` → validate docs links

**Telemetry:** `log_event("ecosystem", free_gb=..., skillbooks=11, openwiki_adapter=...)` → aggregated to `reports/dottie_live_status.json`

**Footer:** Solo personal project, no connection to employer, built with public/free-tier only
