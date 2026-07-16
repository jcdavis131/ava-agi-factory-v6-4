---
id: dottie-eval-distill-daily
enabled: true
mode: task
schedule:
  kind: daily
  timezone: UTC
  time: 09:00:00
metadata:
  created_by: continuous_system
---
# Dottie Eval + Distill — Daily 09:00 UTC with Telemetry
> Solo personal project, no connection to employer, built with public/free-tier only — HOME only

Renamed from ava-eval-distill-daily → Dottie.

Purpose: Nightly eval of Dottie branches + frontier rubric with Ollama judge qwen3:32b, logs to dashboard.

**Logging:**
- `dottie/telemetry.py` → `reports/dottie_telemetry.jsonl` mode `eval_distill`
- File: `logs/cron-dottie-eval_distill.log`

**Steps:**
- `python3 scripts/dottie_continuous_loop.py --mode eval --branch all --eval-mode mock`
- frontier rubric Ollama qwen3:32b → frontier_eval_results.json effort_curve 0.8
- log via log_eval(branch, score, mode)

**Footer:** Solo personal project, no connection to employer, built with public/free-tier only
