---
id: ava-data-gather-4h
enabled: true
mode: task
schedule:
  kind: interval
  timezone: UTC
  at: 2026-07-11T08:00:00
  every: 4h
metadata:
  created_by: dataset_expansion
  note: HOME persona only, solo project, no employer resources
---
# Ava Dataset Expansion — Every 4h (Hatch VM Efficient Downstream)
# Solo personal project, no connection to employer, built with public/free-tier only. HOME only.

Purpose: Continuously expand training data more frequently than daily. Runs 10M tokens per invocation = 60M/day = 1.8B/month, incremental, deduplicated, efficient for downstream Alienware RTX 4090.

Steps:
1. cd ~/workspace/ava-agi-factory-v6-4
2. Check disk: df -h, ensure <80% usage, else rotate old shards to data/for_upload/ and clean data/daily_expanded/ keeping last 2 days.
3. Run expansion:
   python scripts/dataset_expansion.py --tokens 10M --phases p0_logic p1_math p2_foundation p3_code --out data/daily_expanded --upload-mode local
   - Uses simhash dedup threshold 3 + md5 + quality filter (alpha_ratio >0.6, reward >0.8)
   - Writes content-addressable shards: packed_{timestamp}_{idx}_{rand}.jsonl.gz with sha256 first12 in manifest
   - Append-only global data/manifest.jsonl with sha256_full, tokens_est, source, phase, timestamp, version
4. Check Drive guard before ANY upload:
   python scripts/gdrive_uploader.py --check
   - If WORK DRIVE DETECTED (owners @meta.com, health.json files), ABORT upload, save to data/for_upload/ only
   - Log warning: "WORK DRIVE DETECTED — Home/Work separation violation risk, not uploading Home data to work Drive. Please connect personal Drive jcdavis131@gmail.com or use R2"
5. If personal Drive connected AND safe:
   python scripts/gdrive_uploader.py --upload data/daily_expanded/ --folder Ava-Datasets-Expansion --dry-run (first)
   Then real: --upload data/daily_expanded/ --folder Ava-Datasets-Expansion
   - Batch 2 workers, dedup by sha12 in filename, retry 3x backoff
6. Else if R2 creds present (CLOUDFLARE_R2_*):
   python scripts/dataset_expansion.py --tokens 10M --upload-mode r2
   - Uploads to s3://ava-datasets/expansion/ with content-addressable keys
7. Else local fallback:
   - Copy manifest to data/for_upload/upload_manifest_{ts}.json
   - Write your_files/ava-agi/runs/expansion-{date}.log with tokens, shards, disk usage
   - For Alienware: rsync -avz data/daily_expanded/ your-alienware:~/ava-agi-factory-v6-4/data/daily_expanded/
8. Update STATUS.json: builder.last_expansion tokens, shards, timestamp

Efficiency:
- Chunked 50MB shards gzipped, incremental manifest push, content-addressable filename = sha256 first12 avoids re-upload
- Simhash dedup O(1) with 15k window, keeps Hatch VM RAM low
- Public pip only: no torch needed, uses hashlib, gzip, json

Compliance:
- HOME only, never touch 03_Meta_Work_ISOLATED, never upload Home data to work Drive camd@meta.com
- Disclaimer footer on all logs

Downstream usage (Alienware):
  ./scripts/local_train.sh python streaming_data.py --use_expanded data/daily_expanded/ --pack --seq 2048
  torchrun train_1b_deepspeed.py --preset mini --data_manifest data/manifest.jsonl --tokens_total 2B
