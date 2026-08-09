---
id: dottie-wiki-ingest-2h
enabled: true
mode: task
schedule:
  kind: interval
  timezone: UTC
  at: 2026-07-16T08:00:00
  every: 2h
metadata:
  created_by: openwiki_sync
  note: HOME only
---
# Dottie Wiki Ingest — Every 2h
> Solo personal project, no connection to employer, built with public/free-tier only — HOME only

Purpose: Ingest OpenWiki ~1k Wikipedia packs into Dottie dataset every 2h, rotating quality.

**Logging:**
- `dottie/telemetry.py` → `reports/dottie_telemetry.jsonl` mode `wiki_ingest`
- File: `logs/cron-dottie-wiki_ingest.log`

**Steps:**
- `python3 scripts/dottie_continuous_loop.py --mode data --tokens 500000` includes p3 wikipedia phase mapped to OpenWiki adapter
- Adapter: ~/.openwiki/wiki/*.md → dataset format if present else skip
- Deduplicate via simhash th3+md5

**Footer:** Solo personal project, no connection to employer, built with public/free-tier only
