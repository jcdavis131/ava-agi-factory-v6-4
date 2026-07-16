---
id: dottie-training-weekly
enabled: true
mode: task
schedule:
  kind: weekly
  timezone: UTC
  time: 03:00:00
  dow: [Sun]
metadata:
  created_by: continuous_system
---
# Dottie Training — Weekly Sunday 03:00 UTC with Telemetry
> Solo personal project, no connection to employer, built with public/free-tier only — HOME only

Renamed from ava-training-weekly → Dottie.

Purpose: Weekly incremental training nano/mini/base1b WSD + YaRN on Alienware, mock in Hatch VM.

**Logging:**
- `dottie/telemetry.py` → `reports/dottie_telemetry.jsonl` mode `train`
- File: `logs/cron-dottie-training-weekly.log`

**Steps:**
- `python3 scripts/dottie_continuous_loop.py --mode train --preset mini --tokens-total 2500000000 --resume --steps 1000`
- Hatch VM mock: dottie_mini_mock.pt loss 3.2
- Alienware real: torchrun deepspeed_zero3_bf16 mini WSD 736k stable 92% YaRN RoPE 1M checkpoint

**Footer:** Solo personal project, no connection to employer, built with public/free-tier only
