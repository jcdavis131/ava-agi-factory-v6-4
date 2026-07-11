# Continuous Pipelines — Ava + Dumb Models
Solo personal project, no connection to employer, built with public/free-tier only

## Architecture

```
[Gather] -> [Curate/Clean] -> [Tokenize/Pack] -> [Train] -> [Distill MOPD] -> [Eval] -> [Deploy]
   ^                           ^                  ^           ^            ^         ^
   | cron 08:00 UTC            | cron daily       | weekly Sun 03:00 | daily 09:00 | vector daily 12:00 UTC
   | Prefect retry 3x          | streaming_data   | deepspeed      | Ollama judge | Vercel webhook
```

### Prefect Server (Free Self-Hosted)
- `pip install prefect==3.4.0`
- `prefect server start --port 4200`  # UI at http://localhost:4200
- All flows use `OLAMA_HOST=http://host.docker.internal:11434` to talk to host Ollama (qwen3:32b)
- No YAML, no DSL — just @flow/@task decorators, retries, version tracking

### Hatch Cron Jobs (Created)

| ID | Schedule | What |
|---|---|---|
| `ava-data-gather-daily` | daily 08:00 UTC | logic_textbook_pipeline p0-p3 100M tokens, tokenizer build, pack shards |
| `ava-eval-distill-daily` | daily 09:00 UTC | branch harness + frontier rubric via Ollama + distill dry-run |
| `vector-dumb-models-daily` | daily 12:00 UTC (06:00 CDT) | nflverse/StatsBomb/ESPN ingest -> per-100 z-score -> PCA embeddings -> Vercel deploy |
| `ava-training-weekly` | weekly Sun 03:00 UTC | torchrun nano/mini/base1b incremental WSD 736k stable 92% |

Plus existing: cash-drag-watcher 02:00, fidelity screenshot Mon 09:00 CT

### For your Alienware (RTX 4090 24GB + Docker + Ollama)

Local crontab to add (`crontab -e`):

```cron
# Ava data - 2am Central daily
0 2 * * * cd ~/ava-agi-factory-v6-4 && ./scripts/local_train.sh python logic_textbook_pipeline.py --phases all --out data/daily/raw --tokens 100M >> logs/cron-data.log 2>&1

# Ava train - Sun 3am Central weekly incremental
0 3 * * 0 cd ~/ava-agi-factory-v6-4 && ./scripts/local_train.sh torchrun --nproc_per_node=1 train_1b_deepspeed.py --preset mini --tokens_total 2500000000 --resume-if-exists >> logs/cron-train.log 2>&1

# Eva eval - 3am Central daily
0 3 * * * cd ~/ava-agi-factory-v6-4 && OLLAMA_HOST=http://host.docker.internal:11434 OLLAMA_MODEL=qwen3:32b python eval_frontier_rubric.py --domain all --judge ollama >> logs/cron-eval.log 2>&1

# Vector models - 6am Central daily
0 6 * * * cd ~/ava-agi-factory-v6-4 && python prefect_flows.py --run vector --leagues all >> logs/cron-vector.log 2>&1
```

### Prefect Deployment Commands (Alternative to cron)

```bash
# Inside Docker
prefect deployment build prefect_flows.py:ava_full_pipeline -n ava-daily --cron "0 8 * * *" --pool default-agent-pool -q ava-queue
prefect deployment build prefect_flows.py:daily_vector_flow -n vector-daily --cron "0 12 * * *" --pool default-agent-pool -q vector-queue
prefect agent start -q ava-queue
prefect agent start -q vector-queue
```

### Distillation Pipeline (New - MOPD from HF blog)

Implements https://huggingface.co/blog/sergiopaniego/distillation-2026 :

1. Train 3 separate RL experts (same size 1.17B) per domain: code, math, agentic/chat - each YaRN 1M RoPE
2. Student (unified) generates own rollouts (on-policy)
3. For each token, teachers grade: reverse KL KL(p_student || p_teacher) dense signal
4. Cost ~1/10 GPU hours vs RL (Qwen3 claim)
5. Optional privileged hint (Cursor Composer 2.5) and earlier-teacher continual learning (Thinking Machines)

Run:
```
./scripts/distill.sh torchrun --nproc_per_node=1 on_policy_distill.py --mode mopd --teachers ckpts/code_expert.pt ckpts/math_expert.pt ckpts/chat_expert.pt --student-ckpt ava_stable_736k.pt --tokens_total 100M --preserve-router
```

### Dumb Models Details

**Vector Hoops** (12,392 player-seasons 1996-2026):
- Daily ingest nflverse/basketball-ref via @task retries 5x60s on 500
- Compute per-100 z-scored within season, PCA(3) 8 archetypes
- Build flow SVG truthful topology

**Vector Gridiron** (fantasy MAE 4.268 R2 0.39):
- nflverse no tracking, usage/snaps/weather/Vegas
- Next-game task daily

**Vector Pitch** (633 WC tournaments):
- StatsBomb per-90 tournament z-scored

All deploy via Vercel webhook if VERCEL_DEPLOY_HOOK_URL set.

### Observability

- Hatch: activity feed /spaces_actions
- Prefect: http://localhost:4200 dashboard - flow runs, retries, logs, version
- Logs: your_files/ava-agi/runs/, logs/builder.log, your_files/vector-daily/

### Solo Disclaimer
All pipelines: public pip only (torch, transformers, prefect), free-tier R2/Workers/Supabase/HF ZeroGPU, ONNX WASM, local Ollama qwen3:32b. No work data/code/systems. Footer: "Solo personal project, no connection to employer, built with public/free-tier only"
