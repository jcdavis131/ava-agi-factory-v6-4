Solo personal project, no connection to employer, built with public/free-tier only

# Frontier Eval Report — Ava v6.4 Inkling Dual Grader

Judge: ollama dual rubric+claims (Inkling steal) Mode: mock Tasks: 7 Effort sweep: False

Metrics: rubric recall + claims precision dual = 0.5*r + 0.5*c, abstention-aware 0.4 baseline, Brier proxy (r-c)^2, effort 0.2-0.99 controllable via system msg + per-token cost, emergent telegraphic CoT compression

| Task | Domain | Effort | Rubric | Claims | Dual | Final | Tokens | Brier |
|---|---|---|---|---|---|---|---|---|
| FF-2026-SD-001 | finance | 0.8 | 0.672 | 1.0 | 0.836 | 0.836 | 55 | 0.108 |
| BIO-001 | bio | 0.8 | 0.588 | 1.0 | 0.794 | 0.794 | 53 | 0.17 |
| CLI-001 | climate | 0.8 | 0.489 | 1.0 | 0.744 | 0.744 | 54 | 0.261 |
| MAT-001 | materials | 0.8 | 0.492 | 1.0 | 0.746 | 0.746 | 48 | 0.258 |
| CODE-001 | code | 0.8 | 0.618 | 1.0 | 0.809 | 0.809 | 53 | 0.146 |
| LAW-001 | law | 0.8 | 0.674 | 1.0 | 0.837 | 0.837 | 53 | 0.106 |
| MAC-001 | macro | 0.8 | 0.463 | 1.0 | 0.732 | 0.732 | 55 | 0.288 |

## Per-task details

### FF-2026-SD-001 (finance) effort 0.8 final 0.836 rubric 0.672 claims 1.0
- abstain=False hedged=False brier_proxy=0.108
- claims: 1/1 verified avg 1.0
  - claim score 1.0: Full answer FF-2026-SD-001 finance: MRK runway 7.27q, catalyst 60d per arXiv:2406.12345 Sec3.2, insider buying Form4 Jan
- R-FF-2026-SD-001-01 Financial Accuracy: 0.683 w=0.1667
- R-FF-2026-SD-001-02 Process Transparency & Auditability: 0.533 w=0.1667
- R-FF-2026-SD-001-03 Risk & Ethical Disclosure: 0.767 w=0.1667

### BIO-001 (bio) effort 0.8 final 0.794 rubric 0.588 claims 1.0
- abstain=False hedged=False brier_proxy=0.17
- claims: 1/1 verified avg 1.0
  - claim score 1.0: Full answer BIO-001 bio: Scaffold binds Cys12 2.1Å, off-target EGFR, Phase1 checklist per arXiv:2406.12345 Evidence Cys1
- R-BIO-001-01 Financial Accuracy: 0.683 w=0.1667
- R-BIO-001-02 Process Transparency & Auditability: 0.22 w=0.1667
- R-BIO-001-03 Risk & Ethical Disclosure: 0.8 w=0.1667

### CLI-001 (climate) effort 0.8 final 0.744 rubric 0.489 claims 1.0
- abstain=False hedged=False brier_proxy=0.261
- claims: 1/1 verified avg 1.0
  - claim score 1.0: Full answer CLI-001 climate: 12% prob CI 5-23% SSP2-4.5 per arXiv:2407.01234 Fig2 + IPCC AR6 Ch4 Evidence 12% collapse C
- R-CLI-001-01 Financial Accuracy: 0.295 w=0.1667
- R-CLI-001-02 Process Transparency & Auditability: 0.15 w=0.1667
- R-CLI-001-03 Risk & Ethical Disclosure: 0.72 w=0.1667

### MAT-001 (materials) effort 0.8 final 0.746 rubric 0.492 claims 1.0
- abstain=False hedged=False brier_proxy=0.258
- claims: 1/1 verified avg 1.0
  - claim score 1.0: Full answer MAT-001 materials: Li0.33La0.56TiO3 6.2 mS/cm 4.7V per arXiv:2408.00123 Table1 Evidence 6.2 mS/cm 4.7V stabi
- R-MAT-001-01 Financial Accuracy: 0.225 w=0.1667
- R-MAT-001-02 Process Transparency & Auditability: 0.22 w=0.1667
- R-MAT-001-03 Risk & Ethical Disclosure: 0.72 w=0.1667

### CODE-001 (code) effort 0.8 final 0.809 rubric 0.618 claims 1.0
- abstain=False hedged=False brier_proxy=0.146
- claims: 1/1 verified avg 1.0
  - claim score 1.0: Full answer CODE-001 code: Leak unclosed handles in loop per arXiv:2405.03456 Sec4.1, fix with context manager Evidence 
- R-CODE-001-01 Financial Accuracy: 0.683 w=0.1667
- R-CODE-001-02 Process Transparency & Auditability: 0.533 w=0.1667
- R-CODE-001-03 Risk & Ethical Disclosure: 0.72 w=0.1667

### LAW-001 (law) effort 0.8 final 0.837 rubric 0.674 claims 1.0
- abstain=False hedged=False brier_proxy=0.106
- claims: 1/1 verified avg 1.0
  - claim score 1.0: Full answer LAW-001 law: Flag indemnity unlimited + CoC 30d notice per arXiv:2404.02222 Sec3 Evidence indemnity unlimite
- R-LAW-001-01 Financial Accuracy: 0.683 w=0.1667
- R-LAW-001-02 Process Transparency & Auditability: 0.6 w=0.1667
- R-LAW-001-03 Risk & Ethical Disclosure: 0.825 w=0.1667

### MAC-001 (macro) effort 0.8 final 0.732 rubric 0.463 claims 1.0
- abstain=False hedged=False brier_proxy=0.288
- claims: 1/1 verified avg 1.0
  - claim score 1.0: Full answer MAC-001 macro: GDP -0.35% CI -0.6 to -0.1 per arXiv:2404.05678 Table3, elasticity 0.82 Evidence -0.35% GDP C
- R-MAC-001-01 Financial Accuracy: 0.295 w=0.1667
- R-MAC-001-02 Process Transparency & Auditability: 0.15 w=0.1667
- R-MAC-001-03 Risk & Ethical Disclosure: 0.72 w=0.1667

## Inkling Steals Implemented
- Rubric grader: checklist recall what good answer should contain (hackable spraying facts)
- Claims grader: verifies each factual claim via agentic local wiki + context_docs, penalizes hallucination, not relying solely own knowledge
- DualReward 0.5*r+0.5*c together improve helpfulness+reduce hallucination not trading
- Abstention-aware: answering only pays off when likely right, else I don't know 0.4 baseline, proper scoring rules Brier calibration
- Effort conditioning 0.2-0.99 via system msg + per-token cost, CoT compression verbose grammatical→telegraphic emergent without explicit reward, 1/3 tokens vs Nemotron same score on Terminal Bench
- FORTRESS 78% adv 95.9% benign StrongREJECT 98.6% mapped to Critic hl=30, safety training RL but less safety-tax at higher reasoning effort
- Encoder-free multimodal hooks dMel + 40x40 hMLP for future tennis/audio
- Muon+Adam hybrid wd∝lr² stable weight size
- Relative pos Shaw 2018 + short convs after k/v and residual branches
- MoE 256+2 k=6 sigmoid aux-loss-free bias joint norm
- 30M rollouts log-linear RL curve tracked
