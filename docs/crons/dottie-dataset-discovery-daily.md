---
id: dottie-dataset-discovery-daily
enabled: true
mode: task
schedule:
  kind: daily
  timezone: UTC
  time: 14:00:00
metadata:
  created_by: continuous_system
---
# Dottie Dataset Discovery — Daily 14:00 UTC with Telemetry
> Solo personal project, no connection to employer, built with public/free-tier only — HOME only

Renamed from dataset-discovery-daily → dottie-dataset-discovery-daily.

Purpose: Discover HF datasets for weak domains based on eval results.

**Logging:**
- `dottie/telemetry.py` → `reports/dottie_telemetry.jsonl` mode `dataset_discovery`
- File: `logs/cron-dottie-dataset_discovery.log`

**Steps:**
- Parse weak domains cap_score<0.9 from eval results
- Searches HF API free license MIT/Apache2/CC0
- Writes candidates json + download sh for Alienware, no massive download in VM

Old generic dataset-discovery-daily disabled to avoid duplicate.

**Footer:** Solo personal project, no connection to employer, built with public/free-tier only
