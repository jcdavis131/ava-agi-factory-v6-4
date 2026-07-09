# J-Space Optimization Roadmap
Solo personal project, no connection to employer
- 32 slots baseline -> 64 slots -> 128 slots (human 7±2 but superhuman)
- YaRN 1M base, J-space can hold 128k summary that broadcasts to all 131k tokens
- Training losses to grow J-space: reportability CE(verbalizer(ws.mean), target), modulation sim_with-sim_without>0.5, broadcast target 20% token norm, selectivity auto low var deliberate high var
- Curriculum: Phase0-1 train reportability force report spider before 8, Phase2-3 modulation citrus reward, Phase4 long broadcast 128k needle must be held, Phase5 anneal selectivity mix auto vs deliberate
