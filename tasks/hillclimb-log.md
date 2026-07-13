# Hill-climb log (one line per tick)

| ts_local | step | lm | tok_s | host_free_GB | ready_P0 | action | delta |
|----------|------|-----|-------|--------------|----------|--------|-------|
| 2026-07-10T18:32 | 460 | 0.111 | 12344 | 1.6 | 481M | plan+loop armed; image prune 0B; identified dedup.db 7.2G + host disk as D1 fail | baseline |
| 2026-07-10T18:34 | 460 | — | — | 1.6 | — | docker image prune -a → 12.4GB VHDX (host free unchanged); found host HF cache **352GB**; started delete | D1 in progress |
| 2026-07-10T18:35 | 470 | 0.099 | 12055 | **353.4** | 450M | HF cache cleared; collectors `source_start` again; scale collector→4 | D1 green |
| 2026-07-10T18:36 | 470→1 | — | — | 353 | — | CUDA unknown error @470; **no mini ckpt** (every 500); resume from scratch; set `checkpoint_every_steps: 100` | reliability |
| 2026-07-10T18:45 | 1 | 10.58 | — | 366 | 419M | ticks 3–12 batched; HF clear done; trainer rebuilt (ckpt/100); collectors×4 healthy; waiting first ckpt @100 | train climb |
| 2026-07-10T18:45 | 1 | 10.58 | — | 366 | 419M | **loop 1m→15m** (budget: tick spam while waiting on train); D1 still green | cadence |
| 2026-07-10T19:05 | 50 | 1.022 | 10721 | 353 | 419M | 15m tick#1: D1–D3 green; mini climb 1→50 (lm 10.6→1.02); ckpt_every=100 live; no mini.pt yet; hold for first ckpt | train climb |
| 2026-07-10T19:24 | 100 | 0.302 | 11275 | 353 | 419M | 15m tick#2: **first mini ckpt** `/ckpt/step_100.pt` (~1.9GB); lm 0.35→0.30; D1 green; resume-safe | reliability win |
| 2026-07-10T19:32 | 120 | 0.245 | 11302 | 353 | 419M | 15m tick#3: D1–D5 green; climb 100→120; false-stale between log intervals (~4m); hold for step_200 ckpt; dash gates live | train climb |
| 2026-07-10T19:47 | 150 | 0.211 | 12070 | 353 | 419M | 15m tick#4: D1–D5 green; 120→150; lm↓; ~12k tok/s; still on step_100.pt; hold for step_200 | train climb |
| 2026-07-10T19:58 | 100 | — | — | 353 | 419M | **closed-loop v1 live**: demand.py+train publish+/collectors reweight+dash; resumed step_100 (lost ~80 unckpt steps); `/state/demand.json`; loop 15m→**1m** | demand channel |
| 2026-07-10T20:02 | 110 | 0.326 | 11300 | 365 | 479M | 1m tick#1: demand refresh @110 (runway healthy→maintain); P0 479M; miners up; train climbing post-resume | closed-loop ok |
| 2026-07-10T20:03 | 110 | 0.326 | 11300 | 365 | 479M | 1m ticks#2–3 batched: training mode; demand↔step synced; GPU ~68%; hold for step 120/200 | measure |
| 2026-07-10T20:04 | 110 | 0.326 | 11300 | 365 | 479M | tick#4: still @110 (inter-log); **loop 1m→5m** (step logs ~4m; cut empty ticks) | cadence |
| 2026-07-10T20:08 | 120 | 0.302 | 12186 | 365 | 479M | 5m tick#1: 110→120; demand synced maintain; D1 green; hold for step_200 ckpt | train climb |
| 2026-07-10T20:13 | 140 | 0.240 | 12729 | 365 | 479M | 5m tick#2: 120→140; lm↓; demand@140 maintain; ~60 steps to step_200 | train climb |
| 2026-07-10T20:18 | 150 | 0.211 | 12729 | 365 | 479M | 5m tick#3: 140→150; lm↓; demand maintain; ~50 steps to step_200 | train climb |
| 2026-07-10T20:23 | 170 | 0.186 | 13373 | 365 | 479M | 5m tick#4: 150→170; waiting step_200; **dash**: curriculum+lifecycle glossary+watch signals | fill-wait |
| 2026-07-10T20:40 | 100 | — | — | 353 | 479M | ticks#4–7: CUDA @190 again (no step_200); resumed×2; **ckpt_every 100→50**; demand had fired examples on lm rise | reliability |
| 2026-07-10T20:45 | 100 | — | — | 353 | 479M | tick#8: CUDA restart-loop; **root cause**: host GPU shared with `train_mtnn.py` + `train_stage2.py`; trainer **stopped** pending GPU exclusive | blocker |
| 2026-07-10T20:49 | — | — | — | 361 | 479M | tick#9: trainer still stopped; `train_stage2` gone; **`train_mtnn.py` still on GPU**; collectors active; hold | wait GPU |
| 2026-07-10T21:19 | 100 | — | — | — | 479M | user: continue training → resumed step_100; **hung 25m @100% GPU, 0 steps** (train_mtnn still sharing); trainer stopped again | GPU exclusive needed |
| 2026-07-10T21:20 | — | — | — | 360 | 479M | ticks#10–13: trainer stopped; **train_mtnn.py still on GPU**; data plane OK; hold for exclusive GPU | wait GPU |
| 2026-07-10T21:23 | — | — | — | — | 479M | tick#14: train_mtnn still on GPU; trainer stopped; hold | wait GPU |
| 2026-07-10T21:28 | 100 | — | — | — | 479M | tick#15: **GPU free**; trainer resumed step_100 (ckpt/50); exclusive; climbing toward step_150 | train resume |
| 2026-07-10T21:34 | 110 | 0.329 | 11792 | 360 | 479M | tick#16: exclusive GPU healthy; 100→110 @~12k tok/s; next ckpt **150** | train climb |
| 2026-07-10T21:38 | 120 | 0.332 | 9862 | 360 | 479M | tick#17: 110→120; ~30 steps to step_150 ckpt; no CUDA | train climb |
| 2026-07-10T21:43 | 130 | 0.264 | 10023 | 360 | 479M | tick#18: 120→130; lm↓; ~20 steps to step_150 | train climb |
| 2026-07-10T21:53 | 150 | 0.225 | 9875 | 385 | 479M | tick#19: **step_150.pt** written (~2.0GB); 130?150; lm?; demand maintain; exclusive GPU; next ckpt **200** | reliability win |
| 2026-07-10T21:57 | 160 | 0.192 | 8245 | 385 | 479M | tick#20: 150?160 post-ckpt; lm?; false-stale cleared; demand maintain; ~40 steps to step_200 | train climb |
| 2026-07-10T22:02 | 170 | 0.195 | 8844 | 385 | 479M | tick#21: 160?170; lm?; demand maintain; ~30 steps to step_200 | train climb |
| 2026-07-10T22:10 | 180 | 0.176 | 6061 | 384 | 479M | tick#22: 170?180 (slow ~7m/10steps); demand maintain; ~20 steps to step_200 | train climb |
| 2026-07-10T22:18 | 190 | 0.167 | 5025 | 384 | 479M | tick#23: 180?190; lm=0.167; ~10 to step_200; demand maintain | train climb |
| 2026-07-10T22:24 | 200 | 0.339 | 8507 | 382 | 479M | tick#24: **step_200.pt** written; 190?200; demand maintain; next ckpt 250 | reliability win |
| 2026-07-10T22:28 | 210 | 0.172 | 8492 | 382 | 479M | tick#25: 200?210; lm 0.1724; **demand?examples** (lm_trend+0.15); boost deliberate/automatic; next ckpt 250 | closed-loop |
| 2026-07-10T22:36 | 210 | 0.172 | 8492 | 381 | 479M | tick#26: CUDA after 210; resume from step_200 then CUDA again; **train_mtnn.py on GPU**; trainer **stopped**; exclusive needed | blocker |
| 2026-07-10T22:39 | 210 | � | � | � | 479M | ticks#27�28: trainer stopped; **train_mtnn.py still on GPU**; hold for exclusive; last ckpt step_200 | wait GPU |
| 2026-07-10T22:42 | 210 | � | � | � | 479M | tick#29: trainer stopped; train_mtnn still on GPU; hold | wait GPU |
| 2026-07-10T22:47 | 210 | � | � | � | 479M | tick#30: trainer stopped; train_mtnn still on GPU; hold; ckpt step_200 | wait GPU |
| 2026-07-10T22:52 | 210 | � | � | � | 479M | tick#31: trainer stopped; train_mtnn still on GPU (72% util); hold; ckpt step_200 | wait GPU |
| 2026-07-10T23:01 | 200 | � | � | � | 479M | tick#32: **GPU free**; trainer resumed from step_200; exclusive | train resume |
| 2026-07-10T23:09 | 220 | 0.149 | 9213 | � | 479M | tick#33: resume healthy 200?220; lm?; ~9.2k tok/s; exclusive; ~30 to step_250 | train climb |
| 2026-07-10T23:15 | 230 | 0.131 | 9251 | � | 479M | tick#34: 220?230; exclusive; ~20 to step_250 | train climb |
| 2026-07-10T23:24 | 250 | 0.116 | 9113 | � | 479M | tick#35: **step_250.pt** written; 230?250; lm?; exclusive; next ckpt 300 | reliability win |
| 2026-07-10T23:29 | 260 | 0.118 | 7465 | � | 479M | ticks#36�37: post-ckpt 250?260; exclusive; ~40 to step_300 | train climb |
| 2026-07-10T23:36 | 270 | 0.308 | 6859 | � | 479M | tick#38: 260?270; exclusive; ~30 to step_300 | train climb |
| 2026-07-10T23:43 | 280 | 0.137 | 6058 | � | 479M | tick#39: 270?280; lm recovered; demand?examples; ~20 to step_300 | train climb |
| 2026-07-10T23:54 | 290 | 0.125 | 6817 | � | 479M | ticks#40�41: 280?290; ~10 to step_300; exclusive | train climb |
| 2026-07-10T23:58 | 300 | 0.116 | 6692 | � | 479M | ticks#42�44: **step_300.pt** written; 290?300; exclusive; next ckpt 350 | reliability win |
| 2026-07-11T00:03 | 310 | 0.111 | 5879 | � | 479M | tick#45: 300?310; exclusive; ~40 to step_350 | train climb |
| 2026-07-11T00:10 | 320 | 0.139 | 6338 | � | 479M | tick#46: 310?320; exclusive; ~30 to step_350 | train climb |
| 2026-07-11T00:16 | 330 | 0.192 | 7273 | � | 479M | tick#47: 320?330; exclusive; ~20 to step_350 | train climb |
| 2026-07-11T00:28 | 350 | 0.111 | 8142 | � | 479M | tick#48: **step_350.pt** written; ?350; exclusive | reliability win |
| 2026-07-11T00:33 | 360 | 0.11 | 7222 | � | 479M | ticks#49�50: post-ckpt 350?360; exclusive; ~40 to step_400 | train climb |
| 2026-07-11T07:52 | 360 | 0.110 | 7222 | � | 479M | tick#51: **hung ~hours @360** (GPU busy, no new steps); restarted from step_350; exclusive | hang recover |
| 2026-07-11T07:57 | 360 | 0.164 | 11033 | � | 479M | tick#52: post-hang resume from 350?360; exclusive; ~40 to step_400 | train climb |
| 2026-07-11T08:01 | 370 | 0.125 | 11610 | � | 479M | tick#53: 360?370; exclusive; ~30 to step_400 | train climb |
| 2026-07-11T08:04 | 380 | 0.111 | 11507 | � | 479M | tick#54: 370?380; exclusive; ~20 to step_400 | train climb |
| 2026-07-11T08:13 | 400 | 0.133 | 11755 | � | 479M | tick#55: **step_400.pt** written; 380?400; exclusive; next ckpt 450 | reliability win |
| 2026-07-11T09:27 | 410 | 0.126 | 10728 | � | 479M | tick#56: hung ~1h post-400; resumed step_400?410 @~10.7k tok/s; syllabus plan + hang watchdog; 5m loop re-armed | hang recover |
| 2026-07-11T09:29 | 410 | 0.126 | 10728 | � | 479M | ticks#57�58: old 5m loop killed during re-arm; new syllabus loop PID 33224 live; climb post-400 hang recover | cadence |
| 2026-07-11T09:35 | 410 | 0.126 | 8891 | � | 479M | syllabus-loop tick#1: 2nd resume healthy ?410 @~8.9k tok/s; hang check OK; exclusive; ~40 to step_450 | train climb |
| 2026-07-11T09:39 | 420 | 0.11 | 10200 | � | 479M | syllabus-loop tick#2: 410?420; hang OK; exclusive; ~30 to step_450 | train climb |
| 2026-07-11T09:44 | 420 | 0.110 | 10200 | � | 479M | syllabus-loop tick#3: **train_mtnn back on GPU**; trainer **stopped** at step_400.pt; hold exclusive | wait GPU |
| 2026-07-11T09:48 | 420 | � | � | � | 479M | syllabus-loop tick#4: trainer stopped; train_mtnn still on GPU; dash back @ :8000; hold; ckpt step_400 | wait GPU |
| 2026-07-11T10:01 | 410 | 0.126 | 8370 | � | 479M | syllabus-loop tick#5: GPU-first policy � 4080 free, resumed step_400?410 @~8.4k tok/s; loop re-armed | train resume |
| 2026-07-11T10:06 | 420 | 0.11 | 7708 | � | 479M | syllabus-loop tick#6: 410?420; GPU-first on 4080; ~30 to step_450 | train climb |
| 2026-07-11T10:07 | 420 | 0.11 | 7708 | 368 | 479M | 2m-monitor: trainer Up 13 minutes; ckpt step_400.pt; GPU 55 %, 10248 MiB, 12282 MiB; collectors healthy~4; demand runway healthy �?? maintain mixture; next ckpt 450 | status |
| 2026-07-11T10:09 | 420 | 0.11 | 7708 | 372 | 479M | 2m-loop armed PID=29660; trainer Up 2 minutes; ckpt step_400.pt; GPU 15 %, 3884 MiB, 12282 MiB; collectors healthy~4; hostCUDA~0; demand runway healthy �?? maintain mixture; next hard 450 | status |
| 2026-07-11T10:11 | 420 | 0.11 | 7708 | � | 479M | 2m-tick: trainer Up 3 minutes; ckpt step_400.pt; GPU 29 %, 10343 MiB; collectors~4 | status |
| 2026-07-11T10:11 | 420 | 0.11 | 7708 | � | 479M | 2m-tick: trainer Up 4 minutes; ckpt step_400.pt; GPU 80 %, 10345 MiB | status |
| 2026-07-11T10:12 | 420 | 0.11 | 7708 | � | 479M | 2m-tick#1 (428347): trainer Up 4 minutes; ckpt step_400.pt; GPU 30 %, 10333 MiB; collectors~4; post-resume steps~0 | status |
| 2026-07-11T10:14 | 420 | 0.11 | 7708 | � | 479M | 2m-tick#2: trainer Up 6 minutes; ckpt step_400.pt; GPU 68 %, 10336 MiB; collectors~4; post-resume steps=2 | status |
| 2026-07-11T10:15 | 410 | 0.126 | 8389 | � | 479M | 2m-tick#3: trainer Up 8 minutes; ckpt step_400.pt; GPU 80 %, 10343 MiB; collectors~4; post-resume=1 | status |
| 2026-07-11T10:17 | 410 | 0.126 | 8389 | � | 479M | 2m-tick#4: trainer Up 10 minutes; ckpt step_400.pt; GPU 33 %, 10348 MiB; collectors~4 | status |
| 2026-07-11T10:19 | 420 | 0.11 | 9707 | � | 479M | 2m-tick#5: trainer Up 12 minutes; ckpt step_400.pt; GPU 60 %, 10353 MiB; collectors~4 | status |
| 2026-07-11T10:21 | 420 | 0.11 | 9707 | � | 479M | 2m-tick#6: trainer Up 14 minutes; ckpt step_400.pt; GPU 66 %, 10345 MiB; collectors~4 | status |
| 2026-07-11T10:23 | 430 | 0.107 | 10174 | � | 479M | 2m-tick#7: trainer Up 16 minutes; ckpt step_400.pt; GPU 67 %, 10347 MiB; collectors~4 | status |
| 2026-07-11T10:26 | 430 | 0.107 | 10174 | � | 479M | 2m-tick#8: trainer Up 18 minutes; ckpt step_400.pt; GPU 35 %, 10353 MiB; collectors~4 | status |
| 2026-07-11T10:27 | 430 | 0.107 | 10174 | � | 479M | 2m-tick#9: trainer Up 20 minutes; ckpt step_400.pt; GPU 83 %, 10365 MiB; collectors~4 | status |
| 2026-07-11T10:29 | 440 | 0.105 | 9780 | � | 479M | 2m-tick#10: trainer Up 22 minutes; ckpt step_400.pt; GPU 60 %, 10357 MiB; collectors~4 | status |
| 2026-07-11T10:31 | 440 | 0.105 | 9780 | � | 479M | 2m-tick#11: trainer Up 24 minutes; ckpt step_400.pt; GPU 38 %, 10360 MiB; collectors~4 | status |
| 2026-07-11T10:34 | 450 | 0.105 | 9790 | � | 479M | 2m-tick#12: trainer Up 26 minutes; ckpt step_450.pt; GPU 61 %, 10332 MiB; collectors~4 | status |
| 2026-07-11T10:36 | 450 | 0.105 | 9790 | � | 479M | 2m-tick#13: ckpt step_450.pt; GPU 49 %, 10352 MiB; post-ckpt steps=0; collectors~4 | status |
| 2026-07-11T10:37 | 460 | 0.107 | 8785 | � | 479M | 2m-tick#14: ckpt step_450.pt; GPU 79 %, 10346 MiB; post-ckpt steps=1; collectors~4 | status |
| 2026-07-11T10:40 | 460 | 0.107 | 8785 | � | 479M | 2m-tick#15: ckpt step_450.pt; GPU 69 %, 10349 MiB; collectors~4 | status |
| 2026-07-11T10:41 | 460 | 0.107 | 8785 | � | 479M | 2m-tick#16: ckpt step_450.pt; GPU 63 %, 10349 MiB; collectors~4 | status |
| 2026-07-11T10:43 | 470 | 0.102 | 9736 | � | 479M | 2m-tick#17: ckpt step_450.pt; GPU 37 %, 10337 MiB; collectors~4 | status |
| 2026-07-11T10:45 | 470 | 0.102 | 9736 | � | 479M | 2m-tick#18: ckpt step_450.pt; GPU 24 %, 10336 MiB; collectors~4 | status |
| 2026-07-11T10:47 | 480 | 0.102 | 9997 | � | 479M | 2m-tick#19: ckpt step_450.pt; GPU 27 %, 10341 MiB; collectors~4 | status |
| 2026-07-11T10:49 | 480 | 0.102 | 9997 | � | 479M | 2m-tick#20: ckpt step_450.pt; GPU 78 %, 10341 MiB; collectors~4 | status |
| 2026-07-11T10:51 | 490 | 0.126 | 10216 | � | 479M | 2m-tick#21: ckpt step_450.pt; GPU 58 %, 10345 MiB; collectors~4 | status |
| 2026-07-11T10:54 | 490 | 0.126 | 10216 | � | 479M | 2m-tick#22: ckpt step_450.pt; GPU 56 %, 10342 MiB; collectors~4 | status |
| 2026-07-11T10:56 | 500 | 0.115 | 9927 | � | 479M | 2m-tick#23: ckpt step_500.pt; GPU 74 %, 10337 MiB; collectors~4 | status |
| 2026-07-11T10:58 | 500 | 0.115 | 9927 | � | 479M | 2m-tick#24: ckpt step_500.pt; GPU 35 %, 10353 MiB; post-ckpt steps=0; collectors~4 | status |
| 2026-07-11T10:59 | 510 | 0.131 | 9457 | � | 479M | 2m-tick#25: ckpt step_500.pt; GPU 69 %, 10337 MiB; post-ckpt steps=1; collectors~4 | status |
| 2026-07-11T11:01 | 510 | 0.131 | 9457 | � | 479M | 2m-tick#26: ckpt step_500.pt; GPU 89 %, 10344 MiB; collectors~4 | status |
| 2026-07-11T11:04 | 520 | 0.188 | 10760 | � | 479M | 2m-tick#27: ckpt step_500.pt; GPU 62 %, 10337 MiB; collectors~4 | status |
| 2026-07-11T11:05 | 520 | 0.188 | 10760 | � | 479M | 2m-tick#28: ckpt step_500.pt; GPU 39 %, 10335 MiB; collectors~4 | status |
| 2026-07-11T11:08 | 520 | 0.188 | 10760 | � | 479M | 2m-tick#29: ckpt step_500.pt; GPU 62 %, 10342 MiB; collectors~4 | status |
| 2026-07-11T11:09 | 530 | 0.111 | 10565 | � | 479M | 2m-tick#30: ckpt step_500.pt; GPU 56 %, 10354 MiB; collectors~4 | status |
| 2026-07-11T11:11 | 530 | 0.111 | 10565 | � | 479M | 2m-tick#31: ckpt step_500.pt; GPU 52 %, 10343 MiB; collectors~4 | status |
| 2026-07-11T11:13 | 540 | 0.106 | 9878 | � | 479M | 2m-tick#32: ckpt step_500.pt; GPU 21 %, 10351 MiB; collectors~4 | status |
| 2026-07-11T11:16 | 540 | 0.106 | 9878 | � | 479M | 2m-tick#33: ckpt step_500.pt; GPU 50 %, 10344 MiB; collectors~4 | status |
| 2026-07-11T11:18 | 550 | 0.103 | 9328 | � | 479M | 2m-tick#34: ckpt step_550.pt; GPU 40 %, 10342 MiB; collectors~4 | status |
| 2026-07-11T11:20 | 550 | 0.103 | 9328 | � | 479M | 2m-tick#35: ckpt step_550.pt; GPU 39 %, 10335 MiB; post-ckpt steps=0; collectors~4 | status |
| 2026-07-11T11:21 | 550 | 0.103 | 9328 | � | 479M | 2m-tick#36: ckpt step_550.pt; GPU 71 %, 10337 MiB; post-ckpt steps=0; collectors~4 | status |
| 2026-07-11T11:23 | 560 | 0.101 | 8421 | � | 479M | 2m-tick#37: ckpt step_550.pt; GPU 56 %, 10340 MiB; post-ckpt steps=1; collectors~4 | status |
| 2026-07-11T11:25 | 560 | 0.101 | 8421 | � | 479M | 2m-tick#38: ckpt step_550.pt; GPU 38 %, 10351 MiB; collectors~4 | status |
| 2026-07-11T11:28 | 570 | 0.109 | 10203 | � | 479M | 2m-tick#39: ckpt step_550.pt; GPU 69 %, 10340 MiB; collectors~4 | status |
| 2026-07-11T11:30 | 570 | 0.109 | 10203 | � | 479M | 2m-tick#40: ckpt step_550.pt; GPU 74 %, 10330 MiB; collectors~4 | status |
| 2026-07-11T11:31 | 580 | 0.112 | 10039 | � | 479M | 2m-tick#41: ckpt step_550.pt; GPU 53 %, 10345 MiB; collectors~4 | status |
| 2026-07-11T11:34 | 580 | 0.112 | 10039 | � | 479M | 2m-tick#42: ckpt step_550.pt; GPU 49 %, 10353 MiB; collectors~4 | status |
| 2026-07-11T11:36 | 590 | 0.104 | 10110 | � | 479M | 2m-tick#43: ckpt step_550.pt; GPU 43 %, 10348 MiB; collectors~4 | status |
| 2026-07-11T11:37 | 590 | 0.104 | 10110 | � | 479M | 2m-tick#44: ckpt step_550.pt; GPU 82 %, 10361 MiB; collectors~4 | status |
| 2026-07-11T11:40 | 600 | 0.104 | 10089 | � | 479M | 2m-tick#45: ckpt step_550.pt; GPU 26 %, 10349 MiB; collectors~4 | status |
| 2026-07-11T11:40 | 600 | 0.104 | 10089 | � | 479M | 2m-tick#45: hard save step_600.pt confirmed; GPU active | status |
| 2026-07-11T11:42 | 600 | 0.104 | 10089 | � | 479M | 2m-tick#46: ckpt step_600.pt; GPU 34 %, 10351 MiB; post-ckpt steps=0; collectors~4 | status |
| 2026-07-11T11:43 | 600 | 0.104 | 10089 | � | 479M | 2m-tick#47: ckpt step_600.pt; GPU 41 %, 10361 MiB; post-ckpt steps=0; collectors~4 | status |
| 2026-07-11T11:46 | 610 | 0.103 | 8635 | � | 479M | 2m-tick#48: ckpt step_600.pt; GPU 31 %, 10355 MiB; post-ckpt=1; age_s~81; collectors~4 | status |
| 2026-07-11T11:47 | 610 | 0.103 | 8635 | � | 479M | 2m-tick#49: ckpt step_600.pt; GPU 33 %, 10354 MiB; collectors~4 | status |
| 2026-07-11T11:50 | 620 | 0.122 | 10313 | � | 479M | 2m-tick#50: ckpt step_600.pt; GPU 73 %, 10351 MiB; collectors~4 | status |
| 2026-07-11T11:51 | 620 | 0.122 | 10313 | � | 479M | 2m-tick#51: ckpt step_600.pt; GPU 32 %, 10347 MiB; collectors~4 | status |
| 2026-07-11T11:54 | 630 | 0.11 | 10809 | � | 479M | 2m-tick#52: ckpt step_600.pt; GPU 34 %, 10346 MiB; collectors~4 | status |
| 2026-07-11T11:56 | 630 | 0.11 | 10809 | � | 479M | 2m-tick#53: ckpt step_600.pt; GPU 40 %, 10349 MiB; collectors~4 | status |
| 2026-07-11T11:57 | 640 | 0.126 | 10789 | � | 479M | 2m-tick#54: ckpt step_600.pt; GPU 35 %, 10358 MiB; collectors~4 | status |
| 2026-07-11T12:00 | 640 | 0.126 | 10789 | � | 479M | 2m-tick#55: ckpt step_600.pt; GPU 64 %, 10350 MiB; collectors~4 | status |
| 2026-07-11T12:02 | 650 | 0.104 | 10750 | � | 479M | 2m-tick#56: ckpt step_650.pt; GPU 37 %, 10358 MiB; collectors~4 | status |
| 2026-07-11T12:04 | 650 | 0.104 | 10750 | � | 479M | 2m-tick#57: ckpt step_650.pt; GPU 24 %, 10345 MiB; post-ckpt steps=0; collectors~4 | status |
| 2026-07-11T12:06 | 660 | 0.127 | 9392 | � | 479M | 2m-tick#58: ckpt step_650.pt; GPU 30 %, 10358 MiB; post-ckpt steps=1; collectors~4 | status |
| 2026-07-11T12:08 | 660 | 0.127 | 9392 | � | 479M | 2m-tick#59: ckpt step_650.pt; GPU 45 %, 10357 MiB; collectors~4 | status |
| 2026-07-11T12:09 | 660 | 0.127 | 9392 | � | 479M | 2m-tick#60: ckpt step_650.pt; GPU 31 %, 10354 MiB; collectors~4 | status |
| 2026-07-11T12:12 | 670 | 0.102 | 10007 | � | 479M | 2m-tick#61: ckpt step_650.pt; GPU 60 %, 10359 MiB; age_s~117; collectors~4 | status |
| 2026-07-11T12:14 | 670 | 0.102 | 10007 | � | 479M | 2m-tick#62: ckpt step_650.pt; GPU 65 %, 10349 MiB; collectors~4 | status |
| 2026-07-11T12:16 | 680 | 0.098 | 9904 | � | 479M | 2m-tick#63: ckpt step_650.pt; GPU 73 %, 10349 MiB; collectors~4 | status |
| 2026-07-11T12:18 | 680 | 0.098 | 9904 | � | 479M | 2m-tick#64: ckpt step_650.pt; GPU 60 %, 10349 MiB; collectors~4 | status |
| 2026-07-11T12:20 | 690 | 0.099 | 10204 | � | 479M | 2m-tick#65: ckpt step_650.pt; GPU 50 %, 10349 MiB; collectors~4 | status |
| 2026-07-11T12:22 | 690 | 0.099 | 10204 | � | 479M | 2m-tick#66: ckpt step_650.pt; GPU 54 %, 10361 MiB; collectors~4 | status |
| 2026-07-11T12:24 | 700 | 0.1 | 10156 | � | 479M | 2m-tick#67: ckpt step_700.pt; GPU 58 %, 10355 MiB; collectors~4 | status |
| 2026-07-11T12:26 | 700 | 0.1 | 10156 | � | 479M | 2m-tick#68: ckpt step_700.pt; GPU 44 %, 10349 MiB; post-ckpt steps=0; collectors~4 | status |
| 2026-07-11T12:28 | 700 | 0.1 | 10156 | � | 479M | 2m-tick#69: ckpt step_700.pt; GPU 25 %, 10346 MiB; post-ckpt steps=0; collectors~4 | status |
| 2026-07-11T12:30 | 710 | 0.099 | 8664 | � | 479M | 2m-tick#70: ckpt step_700.pt; GPU 37 %, 10348 MiB; post-ckpt=1; age_s~116; collectors~4 | status |
| 2026-07-11T12:32 | 710 | 0.099 | 8664 | � | 479M | 2m-tick#71: ckpt step_700.pt; GPU 62 %, 10356 MiB; collectors~4 | status |
| 2026-07-11T12:33 | 720 | 0.099 | 10171 | � | 479M | 2m-tick#72: ckpt step_700.pt; GPU 80 %, 10366 MiB; collectors~4 | status |
| 2026-07-11T12:36 | 720 | 0.099 | 10171 | � | 479M | 2m-tick#73: ckpt step_700.pt; GPU 46 %, 10360 MiB; collectors~4 | status |
| 2026-07-11T12:38 | 730 | 0.098 | 10117 | � | 479M | 2m-tick#74: ckpt step_700.pt; GPU 85 %, 10356 MiB; collectors~4 | status |
| 2026-07-11T12:39 | 730 | 0.098 | 10117 | � | 479M | 2m-tick#75: ckpt step_700.pt; GPU 32 %, 10366 MiB; collectors~4 | status |
| 2026-07-11T12:42 | 740 | 0.104 | 10798 | � | 479M | 2m-tick#76: ckpt step_700.pt; GPU 35 %, 10347 MiB; collectors~4 | status |
| 2026-07-11T12:43 | 740 | 0.104 | 10798 | � | 479M | 2m-tick#77: ckpt step_700.pt; GPU 71 %, 10344 MiB; collectors~4 | status |
| 2026-07-11T12:46 | 750 | 0.036 | 10793 | � | 479M | 2m-tick#78: ckpt step_750.pt; GPU 78 %, 10345 MiB; collectors~4 | status |
| 2026-07-11T12:48 | 750 | 0.036 | 10793 | � | 479M | 2m-tick#79: ckpt step_750.pt; GPU 45 %, 10354 MiB; post-ckpt steps=0; collectors~4 | status |
| 2026-07-11T12:50 | 760 | 0.104 | 9859 | � | 479M | 2m-tick#80: ckpt step_750.pt; GPU 29 %, 10343 MiB; post-ckpt steps=1; collectors~4 | status |
| 2026-07-11T12:52 | 760 | 0.104 | 9859 | � | 479M | 2m-tick#81: ckpt step_750.pt; GPU 25 %, 10351 MiB; collectors~4 | status |
| 2026-07-11T12:54 | 770 | 0.115 | 10685 | � | 479M | 2m-tick#82: ckpt step_750.pt; GPU 53 %, 10354 MiB; collectors~4 | status |
| 2026-07-11T12:56 | 770 | 0.115 | 10685 | � | 479M | 2m-tick#83: ckpt step_750.pt; GPU 34 %, 10351 MiB; collectors~4 | status |
| 2026-07-11T12:57 | 780 | 0.098 | 10432 | � | 479M | 2m-tick#84: ckpt step_750.pt; GPU 74 %, 10352 MiB; collectors~4 | status |
| 2026-07-11T13:00 | 780 | 0.098 | 10432 | � | 479M | 2m-tick#85: ckpt step_750.pt; GPU 70 %, 10354 MiB; collectors~4 | status |
| 2026-07-11T13:02 | 790 | 0.097 | 9930 | � | 479M | 2m-tick#86: ckpt step_750.pt; GPU 83 %, 10341 MiB; collectors~4 | status |
| 2026-07-11T13:04 | 790 | 0.097 | 9930 | � | 479M | 2m-tick#87: ckpt step_750.pt; GPU 56 %, 10340 MiB; collectors~4 | status |
| 2026-07-11T13:06 | 790 | 0.097 | 9930 | � | 479M | 2m-tick#88: ckpt step_750.pt; GPU 47 %, 10352 MiB; collectors~4 | status |
| 2026-07-11T13:08 | 800 | 0.096 | 9978 | � | 479M | 2m-tick#89: ckpt step_800.pt; GPU 40 %, 10346 MiB; age_s~116; collectors~4 | status |
| 2026-07-11T13:10 | 800 | 0.096 | 9978 | � | 479M | 2m-tick#90: ckpt step_800.pt; GPU 63 %, 10345 MiB; post-ckpt steps=0; collectors~4 | status |
| 2026-07-11T13:12 | 810 | 0.098 | 8964 | � | 479M | 2m-tick#91: ckpt step_800.pt; GPU 43 %, 10341 MiB; post-ckpt steps=1; collectors~4 | status |
| 2026-07-11T13:14 | 810 | 0.098 | 8964 | � | 479M | 2m-tick#92: ckpt step_800.pt; GPU 59 %, 10351 MiB; collectors~4 | status |
| 2026-07-11T13:16 | 820 | 0.097 | 10268 | � | 479M | 2m-tick#93: ckpt step_800.pt; GPU 38 %, 10348 MiB; collectors~4 | status |
| 2026-07-11T13:18 | 820 | 0.097 | 10268 | � | 479M | 2m-tick#94: ckpt step_800.pt; GPU 58 %, 10347 MiB; collectors~4 | status |
| 2026-07-11T13:20 | 830 | 0.098 | 10307 | � | 479M | 2m-tick#95: ckpt step_800.pt; GPU 30 %, 10356 MiB; collectors~4 | status |
| 2026-07-11T13:22 | 830 | 0.098 | 10307 | � | 479M | 2m-tick#96: ckpt step_800.pt; GPU 48 %, 10358 MiB; collectors~4 | status |
| 2026-07-11T13:24 | 840 | 0.1 | 10179 | � | 479M | 2m-tick#97: ckpt step_800.pt; GPU 41 %, 10360 MiB; collectors~4 | status |
| 2026-07-11T13:26 | 840 | 0.1 | 10179 | � | 479M | 2m-tick#98: ckpt step_800.pt; GPU 65 %, 10339 MiB; collectors~4 | status |
| 2026-07-11T13:28 | 840 | 0.1 | 10179 | � | 479M | 2m-tick#99: ckpt step_800.pt; GPU 79 %, 10356 MiB; collectors~4 | status |
| 2026-07-11T13:30 | 850 | 0.1 | 10058 | � | 479M | 2m-tick#100: ckpt step_850.pt; GPU 45 %, 10345 MiB; age_s~105; collectors~4 | status |
| 2026-07-11T13:32 | 850 | 0.1 | 10058 | � | 479M | 2m-tick#101: ckpt step_850.pt; GPU 75 %, 10351 MiB; post-ckpt steps=0; collectors~4 | status |
| 2026-07-11T13:34 | 860 | 0.096 | 9794 | � | 479M | 2m-tick#102: ckpt step_850.pt; GPU 54 %, 10353 MiB; post-ckpt steps=1; collectors~4 | status |
| 2026-07-11T13:36 | 860 | 0.096 | 9794 | � | 479M | 2m-tick#103: ckpt step_850.pt; GPU 41 %, 10324 MiB; collectors~4 | status |
| 2026-07-11T13:38 | 870 | 0.12 | 10223 | � | 479M | 2m-tick#104: ckpt step_850.pt; GPU 66 %, 10377 MiB; collectors~4 | status |
| 2026-07-11T13:40 | 870 | 0.12 | 10223 | � | 479M | 2m-tick#105: ckpt step_850.pt; GPU 53 %, 10352 MiB; collectors~4 | status |
| 2026-07-11T13:42 | 880 | 0.098 | 10254 | � | 479M | 2m-tick#106: ckpt step_850.pt; GPU 58 %, 10347 MiB; collectors~4 | status |
| 2026-07-11T13:44 | 880 | 0.098 | 10254 | � | 479M | 2m-tick#107: ckpt step_850.pt; GPU 77 %, 10347 MiB; collectors~4 | status |
| 2026-07-11T13:46 | � | � | � | � | 479M | 2m-tick#108: ckpt step_850.pt; GPU 49 %, 10348 MiB; collectors~4 | status |
| 2026-07-11T13:47 | 850(resume) | � | � | � | 479M | 2m-tick#108: CUDA after 880; auto-resume step_850; GPU climbing; collectors~4 | status |
| 2026-07-11T13:48 | 850(resume) | � | � | � | 479M | 2m-tick#109: post-CUDA; ckpt step_850.pt; GPU 73 %, 10360 MiB; post-resume=0; collectors~4 | status |
| 2026-07-11T13:50 | 850(resume) | � | � | � | 479M | 2m-tick#110: post-CUDA; ckpt step_850.pt; GPU 62 %, 10354 MiB; post-resume=0; collectors~4 | status |
| 2026-07-11T13:52 | 860 | 0.106 | 9565 | � | 479M | 2m-tick#111: post-CUDA; ckpt step_850.pt; GPU 38 %, 10350 MiB; post-resume=1; collectors~4 | status |
| 2026-07-11T13:54 | 860 | 0.106 | 9565 | � | 479M | 2m-tick#112: ckpt step_850.pt; GPU 57 %, 10343 MiB; collectors~4 | status |
| 2026-07-11T13:56 | 870 | 0.098 | 10117 | � | 479M | 2m-tick#113: ckpt step_850.pt; GPU 63 %, 10353 MiB; collectors~4 | status |
| 2026-07-11T13:58 | 870 | 0.098 | 10117 | � | 479M | 2m-tick#114: ckpt step_850.pt; GPU 53 %, 10358 MiB; collectors~4 | status |
| 2026-07-11T14:00 | 880 | 0.099 | 10358 | � | 479M | 2m-tick#115: ckpt step_850.pt; GPU 59 %, 10377 MiB; collectors~4 | status |
| 2026-07-11T14:02 | 880 | 0.099 | 10358 | � | 479M | 2m-tick#116: ckpt step_850.pt; GPU 15 %, 3367 MiB; collectors~4; cuda_watch | status |
| 2026-07-11T14:02 | 850? | � | � | � | 479M | 2m-tick#116: CUDA again near 880; checking resume; trainer Up About a minute; ckpt step_850.pt; GPU 47 %, 3955 MiB | status |
| 2026-07-11T14:02 | 850(resume) | � | � | � | 479M | 2m-tick#116: CUDA again after 880 (2nd); auto-resume step_850; pattern suspect | status |
| 2026-07-11T14:04 | 850(resume#2) | � | � | � | 479M | 2m-tick#117: post-CUDA#2; ckpt step_850.pt; GPU 46 %, 866 MiB; post-resume=0; collectors~4 | status |
| 2026-07-11T14:04 | 850? | � | � | � | 479M | 2m-tick#117: CUDA#3 CUBLAS on resume; trainer Up About a minute; GPU 35 %, 3286 MiB | status |
| 2026-07-11T14:06 | 850(resume#3) | � | � | � | 479M | 2m-tick#118: post-CUBLAS; ckpt step_850.pt; GPU 25 %, 846 MiB; post-resume=0; collectors~4 | status |
| 2026-07-11T14:07 | 850? | � | � | � | 479M | 2m-tick#118: CUDA restart loop (unknown after CUBLAS resume); trainer Up About a minute; GPU 19 %, 3050 MiB | status |
| 2026-07-11T14:08 | 850(resume) | � | � | � | 479M | 2m-tick#119: ckpt step_850.pt; GPU 23 %, 8948 MiB; post-resume=0; resumes_20m=3; collectors~4 | status |
| 2026-07-11T14:10 | 850(resume) | � | � | � | 479M | 2m-tick#120: ckpt step_850.pt; GPU 70 %, 10448 MiB; post-resume=0; collectors~4 | status |
| 2026-07-11T14:12 | 850(resume) | � | � | � | 479M | 2m-tick#121: ckpt step_850.pt; GPU 39 %, 10455 MiB; post-resume=0; collectors~4 | status |
| 2026-07-11T14:14 | 860 | 0.119 | 9632 | � | 479M | 2m-tick#122: ckpt step_850.pt; GPU 32 %, 10440 MiB; post-resume=1; collectors~4 | status |
| 2026-07-11T14:16 | 860 | 0.119 | 9632 | � | 479M | 2m-tick#123: ckpt step_850.pt; GPU 75 %, 10476 MiB; collectors~4 | status |
| 2026-07-11T14:18 | 870 | 0.099 | 10579 | � | 479M | 2m-tick#124: ckpt step_850.pt; GPU 71 %, 10480 MiB; collectors~4 | status |
| 2026-07-11T14:20 | 870 | 0.099 | 10579 | � | 479M | 2m-tick#125: ckpt step_850.pt; GPU 57 %, 10486 MiB; collectors~4; near-880-watch | status |
| 2026-07-11T14:22 | 880 | 0.097 | 10545 | � | 479M | 2m-tick#126: ckpt step_850.pt; GPU 43 %, 10370 MiB; collectors~4; 880-watch | status |
| 2026-07-11T14:24 | 880 | 0.097 | 10545 | � | 479M | 2m-tick#127: post-880; ckpt step_850.pt; GPU 65 %, 10441 MiB; collectors~4; cuda=False | status |
| 2026-07-11T14:26 | 890 | 0.097 | 11193 | � | 479M | 2m-tick#128: ckpt step_850.pt; GPU 39 %, 10404 MiB; collectors~4 | status |
| 2026-07-11T14:28 | 890 | 0.097 | 11193 | � | 479M | 2m-tick#129: ckpt step_850.pt; GPU 73 %, 10390 MiB; collectors~4 | status |
| 2026-07-11T14:30 | 900 | 0.096 | 10373 | � | 479M | 2m-tick#130: ckpt step_900.pt; GPU 51 %, 10348 MiB; collectors~4 | status |
| 2026-07-11T14:32 | 900 | 0.096 | 10373 | � | 479M | 2m-tick#131: ckpt step_900.pt; GPU 84 %, 10359 MiB; post-ckpt steps=0; collectors~4 | status |
| 2026-07-11T14:34 | 910 | 0.098 | 9978 | � | 479M | 2m-tick#132: ckpt step_900.pt; GPU 8 %, 10340 MiB; post-ckpt steps=1; collectors~4 | status |
| 2026-07-11T14:36 | 910 | 0.098 | 9978 | � | 479M | 2m-tick#133: ckpt step_900.pt; GPU 26 %, 10344 MiB; collectors~4 | status |
| 2026-07-11T14:38 | 920 | 0.096 | 11284 | � | 479M | 2m-tick#134: ckpt step_900.pt; GPU 33 %, 10353 MiB; collectors~4 | status |
| 2026-07-11T14:40 | 920 | 0.096 | 11284 | � | 479M | 2m-tick#135: ckpt step_900.pt; GPU 35 %, 10366 MiB; collectors~4 | status |
| 2026-07-11T14:42 | 930 | 0.095 | 10583 | � | 479M | 2m-tick#136: ckpt step_900.pt; GPU 86 %, 10674 MiB; collectors~4 | status |
| 2026-07-11T14:44 | 930 | 0.095 | 10583 | � | 479M | 2m-tick#137: ckpt step_900.pt; GPU 13 %, 10697 MiB; collectors~4 | status |
| 2026-07-11T14:45 | 930 | 0.095 | 10583 | � | 479M | 2m-tick#137: CUDA after 930; ckpt step_900.pt; trainer Up 36 seconds; GPU 16 %, 793 MiB | status |
| 2026-07-11T14:46 | 930 | 0.095 | 10583 | � | 479M | 2m-tick#138: post-930-CUDA; ckpt step_900.pt; GPU 19 %, 1671 MiB; post-resume=3; collectors~4 | status |
| 2026-07-11T14:48 | 900(resume) | � | � | � | 479M | 2m-tick#139: ckpt step_900.pt; GPU 23 %, 9335 MiB; post-resume=0; collectors~4 | status |
| 2026-07-11T14:50 | 900(resume) | � | � | � | 479M | 2m-tick#140: ckpt step_900.pt; GPU 37 %, 10351 MiB; post-resume=0; collectors~4 | status |
| 2026-07-11T14:52 | 900(resume) | � | � | � | 479M | 2m-tick#141: ckpt step_900.pt; GPU 61 %, 10348 MiB; post-resume=0; collectors~4 | status |
| 2026-07-11T14:54 | 910 | 0.102 | 8217 | � | 479M | 2m-tick#142: ckpt step_900.pt; GPU 49 %, 10357 MiB; post-resume=1; collectors~4 | status |
| 2026-07-11T14:56 | 910 | 0.102 | 8217 | � | 479M | 2m-tick#143: ckpt step_900.pt; GPU 80 %, 10360 MiB; collectors~4 | status |
| 2026-07-11T14:58 | 920 | 0.103 | 10174 | � | 479M | 2m-tick#144: ckpt step_900.pt; GPU 53 %, 10356 MiB; collectors~4 | status |
| 2026-07-11T15:00 | 920 | 0.103 | 10174 | � | 479M | 2m-tick#145: ckpt step_900.pt; GPU 75 %, 10361 MiB; collectors~4; 930-watch | status |
| 2026-07-11T15:02 | 930 | 0.097 | 10287 | � | 479M | 2m-tick#146: ckpt step_900.pt; GPU 50 %, 10358 MiB; collectors~4; 930-watch | status |
| 2026-07-11T15:04 | 930 | 0.097 | 10287 | � | 479M | 2m-tick#147: post-930; ckpt step_900.pt; GPU 31 %, 10354 MiB; collectors~4; cuda=False | status |
| 2026-07-11T15:06 | 940 | 0.096 | 10159 | � | 479M | 2m-tick#148: ckpt step_900.pt; GPU 23 %, 10362 MiB; collectors~4 | status |
| 2026-07-11T15:08 | 940 | 0.096 | 10159 | � | 479M | 2m-tick#149: ckpt step_900.pt; GPU 41 %, 10355 MiB; collectors~4 | status |
| 2026-07-11T15:10 | 950 | 0.101 | 10197 | � | 479M | 2m-tick#150: ckpt step_900.pt; GPU 52 %, 10343 MiB; collectors~4 | status |
| 2026-07-11T15:11 | 950 | 0.101 | 10197 | � | 479M | 2m-tick#150: hard save step_950.pt confirmed | status |
| 2026-07-11T15:14 | 950 | 0.101 | 10197 | � | 479M | 2m-tick#151: ckpt step_950.pt; GPU 33 %, 10368 MiB; post-ckpt steps=0; collectors~4 | status |
| 2026-07-11T15:14 | 960 | 0.098 | 8565 | � | 479M | 2m-tick#152: ckpt step_950.pt; GPU 27 %, 10359 MiB; post-ckpt steps=1; collectors~4 | status |
| 2026-07-11T15:16 | 960 | 0.098 | 8565 | � | 479M | 2m-tick#153: ckpt step_950.pt; GPU 71 %, 10357 MiB; collectors~4 | status |
| 2026-07-11T15:18 | 960 | 0.098 | 8565 | � | 479M | 2m-tick#154: ckpt step_950.pt; GPU 64 %, 10348 MiB; collectors~4 | status |
| 2026-07-11T15:20 | 970 | 0.097 | 10313 | � | 479M | 2m-tick#155: ckpt step_950.pt; GPU 88 %, 10345 MiB; age 73s; collectors~4 | status |
| 2026-07-11T15:22 | 970 | 0.097 | 10313 | � | 479M | 2m-tick#156: ckpt step_950.pt; GPU 72 %, 10358 MiB; age 199s; collectors~4 | status |
| 2026-07-11T15:24 | 980 | 0.096 | 9886 | � | 479M | 2m-tick#157: ckpt step_950.pt; GPU 59 %, 10348 MiB; age 52s; collectors~4 | status |
| 2026-07-11T15:26 | 980 | 0.096 | 9886 | � | 479M | 2m-tick#158: ckpt step_950.pt; GPU 56 %, 10363 MiB; age 172s; collectors~4 | status |
| 2026-07-11T15:28 | 980 | 0.096 | 9886 | � | 479M | 2m-tick#159: ckpt step_950.pt; GPU 55 %, 10365 MiB; age 274s; collectors~4 | status |
| 2026-07-11T15:30 | 990 | 0.107 | 9116 | � | 479M | 2m-tick#160: ckpt step_950.pt; GPU 20 %, 10343 MiB; age 120s; collectors~4 | status |
| 2026-07-11T15:32 | 990 | 0.107 | 9116 | � | 479M | 2m-tick#161: ckpt step_950.pt; GPU 46 %, 10355 MiB; age 228s; collectors~4 | status |
| 2026-07-11T15:34 | 1000 | 0.098 | 8961 | � | 479M | 2m-tick#162: ckpt step_1000.pt; GPU 43 %, 10351 MiB; age 56s; collectors~4 | status |
| 2026-07-11T15:36 | 1000 | 0.098 | 8961 | � | 479M | 2m-tick#163: ckpt step_1000.pt; GPU 22 %, 10362 MiB; age 176s; collectors~4 | status |
| 2026-07-11T15:38 | 1000 | 0.098 | 8961 | � | 479M | 2m-tick#164: ckpt step_1000.pt; GPU 59 %, 10358 MiB; age 296s; collectors~4 | status |
| 2026-07-11T15:40 | 1010 | 0.097 | 8176 | � | 479M | 2m-tick#165: ckpt step_1000.pt; GPU 31 %, 10367 MiB; age 96s; collectors~4 | status |
| 2026-07-11T15:42 | 1010 | 0.097 | 8176 | � | 479M | 2m-tick#166: ckpt step_1000.pt; GPU 23 %, 10364 MiB; age 215s; collectors~4 | status |
| 2026-07-11T15:44 | 1020 | 0.099 | 9128 | � | 479M | 2m-tick#167: ckpt step_1000.pt; GPU 68 %, 10350 MiB; age 48s; collectors~4 | status |
| 2026-07-11T15:46 | 1020 | 0.099 | 9128 | � | 479M | 2m-tick#168: ckpt step_1000.pt; GPU 34 %, 10364 MiB; age 166s; collectors~4 | status |
| 2026-07-11T15:48 | 1030 | 0.097 | 9993 | � | 479M | 2m-tick#169: ckpt step_1000.pt; GPU 32 %, 10350 MiB; age 26s; collectors~4 | status |
| 2026-07-11T15:50 | 1030 | 0.097 | 9993 | � | 479M | 2m-tick#170: ckpt step_1000.pt; GPU 42 %, 10361 MiB; age 144s; collectors~4 | status |
| 2026-07-11T15:52 | 1040 | 0.095 | 10274 | � | 479M | 2m-tick#171: ckpt step_1000.pt; GPU 63 %, 10354 MiB; age 12s; collectors~4 | status |
| 2026-07-11T15:54 | 1040 | 0.095 | 10274 | � | 479M | 2m-tick#172: ckpt step_1000.pt; GPU 57 %, 10365 MiB; age 131s; collectors~4 | status |
| 2026-07-11T15:56 | 1040 | 0.095 | 10274 | � | 479M | 2m-tick#173: ckpt step_1000.pt; GPU 37 %, 10358 MiB; age 250s; collectors~4 | status |
| 2026-07-11T15:58 | 1050 | 0.096 | 10120 | � | 479M | 2m-tick#174: ckpt step_1050.pt; GPU 49 %, 10355 MiB; age 110s; collectors~4 | status |
| 2026-07-11T16:00 | 1050 | 0.096 | 10120 | � | 479M | 2m-tick#175: ckpt step_1050.pt; GPU 82 %, 10356 MiB; age 231s; collectors~4 | status |
| 2026-07-11T16:02 | 1060 | 0.095 | 9091 | � | 479M | 2m-tick#176: ckpt step_1050.pt; GPU 30 %, 10360 MiB; age 80s; collectors~4 | status |
| 2026-07-11T16:04 | 1060 | 0.095 | 9091 | � | 479M | 2m-tick#177: ckpt step_1050.pt; GPU 65 %, 10363 MiB; age 184s; collectors~4 | status |
| 2026-07-11T16:06 | 1070 | 0.093 | 10373 | � | 479M | 2m-tick#178: ckpt step_1050.pt; GPU 57 %, 10364 MiB; age 50s; collectors~4 | status |
| 2026-07-11T16:08 | 1070 | 0.093 | 10373 | � | 479M | 2m-tick#179: ckpt step_1050.pt; GPU 43 %, 10354 MiB; age 169s; collectors~4 | status |
| 2026-07-11T16:28 | 1120 | 0.095 | 10752 | � | 479M | 2m-tick#180-189 catchup: ckpt step_1100.pt; GPU 34 %, 10361 MiB; age 203s; collectors~4; cuda_25m=False | status |
| 2026-07-11T16:30 | 1130 | 0.094 | 9681 | � | 479M | 2m-tick#190: ckpt step_1100.pt; GPU 75 %, 10349 MiB; age 26s; collectors~4 | status |
| 2026-07-11T16:32 | 1130 | 0.094 | 9681 | � | 479M | 2m-tick#191: ckpt step_1100.pt; GPU 80 %, 10347 MiB; age 145s; collectors~4 | status |
| 2026-07-11T16:34 | 1130 | 0.094 | 9681 | � | 479M | 2m-tick#192: ckpt step_1100.pt; GPU 63 %, 10359 MiB; age 264s; collectors~4 | status |
| 2026-07-11T16:36 | 1140 | 0.099 | 9404 | � | 479M | 2m-tick#193: ckpt step_1100.pt; GPU 64 %, 10356 MiB; age 104s; collectors~4 | status |
| 2026-07-11T16:38 | 1140 | 0.099 | 9404 | � | 479M | 2m-tick#194: ckpt step_1100.pt; GPU 81 %, 10357 MiB; age 226s; collectors~4 | status |
| 2026-07-11T16:40 | 1150 | 0.096 | 9456 | � | 479M | 2m-tick#195: ckpt step_1150.pt; GPU 42 %, 10356 MiB; age 68s; collectors~4 | status |
| 2026-07-11T16:42 | 1150 | 0.096 | 9456 | � | 479M | 2m-tick#196: ckpt step_1150.pt; GPU 42 %, 10353 MiB; age 187s; collectors~4 | status |
| 2026-07-11T16:44 | 1160 | 0.096 | 8742 | � | 479M | 2m-tick#197: ckpt step_1150.pt; GPU 28 %, 10354 MiB; age 9s; collectors~4 | status |
| 2026-07-11T16:46 | � | � | � | � | 479M | 2m-tick#198: ckpt step_1150.pt; GPU 34 %, 10353 MiB; age -1s; collectors~4 | status |
| 2026-07-11T16:47 | 1150 | � | � | � | 479M | 2m-tick#198: CUDA after step 1160; auto-resumed step_1150.pt; trainer Up ~1m; collectors~4 | status |
| 2026-07-11T16:48 | � | � | � | � | 479M | 2m-tick#199: post-CUDA climb; ckpt step_1150.pt; GPU 30 %, 10357 MiB; age -1s; collectors~4 | status |
| 2026-07-11T16:50 | � | � | � | � | 479M | 2m-tick#200: post-CUDA; ckpt step_1150.pt; GPU 37 %, 10355 MiB; resume_age 258s; collectors~4 | status |
| 2026-07-11T16:52 | 1160 | 0.098 | 9045 | � | 479M | 2m-tick#201: post-CUDA; ckpt step_1150.pt; GPU 60 %, 10358 MiB; resume_age 379s; collectors~4 | status |
| 2026-07-11T16:54 | 1160 | 0.098 | 9045 | � | 479M | 2m-tick#202: ckpt step_1150.pt; GPU 62 %, 10354 MiB; age 218s; collectors~4 | status |
| 2026-07-11T16:56 | 1170 | 0.098 | 10109 | � | 479M | 2m-tick#203: ckpt step_1150.pt; GPU 28 %, 10348 MiB; age 68s; collectors~4 | status |
| 2026-07-11T16:58 | 1170 | 0.098 | 10109 | � | 479M | 2m-tick#204: ckpt step_1150.pt; GPU 71 %, 10365 MiB; age 187s; collectors~4 | status |
| 2026-07-11T17:00 | 1180 | 0.099 | 10651 | � | 479M | 2m-tick#205: ckpt step_1150.pt; GPU 32 %, 10361 MiB; age 61s; collectors~4 | status |
| 2026-07-11T17:02 | 1180 | 0.099 | 10651 | � | 479M | 2m-tick#206: ckpt step_1150.pt; GPU 62 %, 10344 MiB; age 182s; collectors~4 | status |
| 2026-07-11T17:04 | 1190 | 0.095 | 10810 | � | 479M | 2m-tick#207: ckpt step_1150.pt; GPU 85 %, 10356 MiB; age 59s; collectors~4 | status |
| 2026-07-11T17:06 | 1190 | 0.095 | 10810 | � | 479M | 2m-tick#208: ckpt step_1150.pt; GPU 46 %, 10352 MiB; age 179s; collectors~4 | status |
| 2026-07-11T17:13 | 1210 | 0.097 | 9763 | � | 479M | 2m-tick#209-211 catchup: ckpt step_1200.pt; GPU 23 %, 10251 MiB; age 85s; collectors~4 | status |
| 2026-07-11T17:14 | 1210 | 0.097 | 9763 | � | 479M | 2m-tick#212: ckpt step_1200.pt; GPU 41 %, 10270 MiB; age 149s; collectors~4 | status |
| 2026-07-11T17:16 | 1220 | 0.098 | 11141 | � | 479M | 2m-tick#213: ckpt step_1200.pt; GPU 42 %, 10257 MiB; age 32s; collectors~4 | status |
| 2026-07-11T17:18 | 1220 | 0.098 | 11141 | � | 479M | 2m-tick#214: ckpt step_1200.pt; GPU 60 %, 10256 MiB; age 151s; collectors~4 | status |
| 2026-07-11T17:20 | 1230 | 0.098 | 12755 | � | 479M | 2m-tick#215: ckpt step_1200.pt; GPU 59 %, 10253 MiB; age 95s; collectors~4 | status |
| 2026-07-11T17:22 | 1230 | 0.098 | 12755 | � | 479M | 2m-tick#216: ckpt step_1200.pt; GPU 63 %, 10265 MiB; age 186s; collectors~4 | status |
| 2026-07-11T17:24 | 1240 | 0.096 | 12748 | � | 479M | 2m-tick#217: ckpt step_1200.pt; GPU 51 %, 10266 MiB; age 100s; collectors~4 | status |
| 2026-07-11T17:26 | 1250 | 0.094 | 12545 | � | 479M | 2m-tick#218: ckpt step_1200.pt; GPU 25 %, 10251 MiB; age 12s; collectors~4 | status |
| 2026-07-11T17:26 | 1250 | 0.094 | 12545 | � | 479M | 2m-tick#218: hard save step_1250.pt confirmed | status |
| 2026-07-11T17:28 | 1250 | 0.094 | 12545 | � | 479M | 2m-tick#219: ckpt step_1250.pt; GPU 29 %, 10379 MiB; age 136s; collectors~4 | status |
| 2026-07-11T17:30 | 1250 | 0.094 | 12545 | � | 479M | 2m-tick#220: ckpt step_1250.pt; GPU 21 %, 10320 MiB; age 254s; collectors~4 | status |
| 2026-07-11T17:32 | 1260 | 0.098 | 8793 | � | 479M | 2m-tick#221: ckpt step_1250.pt; GPU 41 %, 10317 MiB; age 75s; collectors~4 | status |
| 2026-07-11T17:34 | 1260 | 0.098 | 8793 | � | 479M | 2m-tick#222: ckpt step_1250.pt; GPU 42 %, 10338 MiB; age 197s; collectors~4 | status |
| 2026-07-11T17:42 | 1280 | 0.1 | 10345 | � | 479M | 2m-tick#223-226 catchup: ckpt step_1250.pt; GPU 59 %, 10303 MiB; age 146s; collectors~4 | status |
