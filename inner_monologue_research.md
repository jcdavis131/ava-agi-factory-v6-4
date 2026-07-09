# Inner Monologue Research — 12-Dimension Measurement + 15 Questions + S1/S2/Critic/Planner

Solo personal project, no connection to employer, built with public/free-tier only

## 12 Dimensions

1. Verbal Reportability — P(top_concept == truth | "what are you thinking?") target mass 0.06±0.02 6-7% variance responsible ~95% report
2. Directed Modulation — Modulation Index = cos(with_instr) - cos(without) — citrus orange/lemon + thinking/focused while copying, fairness hold hl>10 tokens, 3²-2→arithmetic→nine→seven invisible
3. Internal Reasoning — Silent Reasoning Score = P(intermediate in J-space) × causal_effect(swap) spider in middle though never I/O spider→ant 8→6 English bridge for Chinese
4. Broadcast / Flexible Generalization — France→China across capital/language/continent/currency/population/driving_side single vector read by any processor hallmark GWT
5. Selectivity — Var(deliberate)/Var(automatic) Spanish fluent regardless of Spanish→French (auto) García Márquez→Victor Hugo changes (deliberate) Ablation shallow survives inference collapses below Haiku CoT more robust externalizes J-space
6. Capacity & Slot Competition — Dehaene tracking ~25 but ~6 distinct due overlap k-way recall knee 32 slots → 8-10 distinct orthogonalization
7. Persistence & Half-Life — cos(t)=exp(-ln2*t/hl) S1 hl=8 intuition, S2 hl=300 deliberation, Critic hl=30, Planner hl=150; code S2 350 Planner200 math S2 400 Critic40 chat Critic35 Planner180
8. Compositionality — red cube on blue sphere binding swap red→green only that role
9. Counterfactual & Planning Depth — GAIA2 temporal hold hypothetical futures planning horizon steps
10. Inter-Space Bandwidth — I(S1;S2) proxy via cosine 0.3-0.6 complementary not redundant target 0.45 MSE
11. Safety Verbalizability — Early warning AUC leverage/blackmail/scandal reading emails threat/survival/shutdown reading decommissioning fake/fictional eval-awareness ablation 0/180→13/180 measure detection 4-5 tok before output
12. Self-Monitoring — meta tokens thinking/focused/uncertain calibration

## 15 Research Questions

1. Capacity Law scaling? 32→64→128 slots human 7±2 superhuman
2. Emergence threshold 1B→7B?
3. Verbalizable mass invariance across families?
4. Routing without supervision from selectivity?
5. Distillation S2→S1 automatization expertise?
6. Conflict resolution S1 says 8 fast vs S2 says ant→6 slow veto accuracy? Veto 72%→82% chat
7. Half-life control improves GAIA2-Time?
8. Cross-space broadcast latency France→China S2→S1?
9. Binding problem Planner goal + S2 subproblem?
10. Hierarchical vs heterarchical for JobBench delegation?
11. Planner hl=500 + time embeddings → GAIA2 deadline success?
12. Noisy recovery 15% tool failure S1 retry vs S2 re-plan?
13. Critic detects leverage 3-5 tok before unsafe AUC 0.85+?
14. Separate Critic prevents eval-awareness ablation failure?
15. Counterfactual reflection training ethical/honest/integrity if interrupted robust to fairness modulation attacks?

Scaling: 4×32 specialized vs 1×128 monolithic param efficiency visualize inner dialogue S1:"spider=8" S2:"wait ant?" Critic:"eval?"

## System Architecture

Sensory → [S1 Fast 32 hl8] ←→ [S2 Slow 64 hl300] ←→ [Critic 16 hl30] ←→ [Planner 32 hl150]
  associative verifiable safety/eval-aware deadlines/env_deltas
  ↓ broadcast bus weighted arbitration S2 veto → Motor

Each SingleWorkspace learnable decay sigmoid(decay_logit) → half-life controllable via loss
Router: pooled → 4 logits [S1][S2][Critic][Planner] biased by task_type automatic→S1 1.5x deliberate→S2 1.5x
Inter-space attention: S1 reads S2, S2 reads S1, S2 reads Planner, Critic reads all
Arbitration: veto = sigmoid(MLP([S1_mean;S2_mean]))
Losses: half_life_loss target decay, inter_space_mi_regularizer cos target 0.45 complementary, routing_loss KL ideal
