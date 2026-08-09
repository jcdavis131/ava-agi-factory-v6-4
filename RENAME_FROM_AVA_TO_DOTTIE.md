# Rename: Ava → Dottie

> Solo personal project, no connection to employer, built with public/free-tier only

**Date:** 2026-07-16

Ava was a placeholder name for the AGI Factory. Renamed to **Dottie**.

### Changes
- Package `ava/` → `dottie/` (379 files touched)
- Classes: `AvaModel1B` → `DottieModel1B`, `AvaConfig` → `DottieConfig`, `AvaTokenizer` → `DottieTokenizer`, `AvaStreamingDataset` → `DottieStreamingDataset`
- Checkpoints: `ava_stable_736k.pt` → `dottie_stable_736k.pt` (symlinks kept for compat: `ava_*.pt -> dottie_*.pt`)
- Artifacts: `ava_nano_stable.pt` → `dottie_nano_stable.pt`, `ava_nano_bpe.json` → `dottie_nano_bpe.json`
- Env: `AVA_CKPT` → `DOTTIE_CKPT` (fallback to legacy), `AVA_CONFIG_DIR` → `DOTTIE_CONFIG_DIR`
- Flows: `ava_data_gen_flow` → `dottie_data_gen_flow`, etc.
- Registry: `create_ava_registry()` → `create_dottie_registry()` with alias
- Docs: all references updated, specs point to `dottie/` paths
- GitHub repo: `https://github.com/jcdavis131/ava-agi-factory-v6-4` → TODO rename to `dottie-agi-factory` (remote still old for now)

### Compat Shims
- `ava/` directory kept as shim that re-exports `dottie/` — `from ava.config import AvaConfig` still works (alias to DottieConfig)
- Checkpoint symlinks preserve old scripts
- Env var fallbacks for AVA_*
- Tokenizer symlinks `data/.../dottie_*.json -> ava_*.json`

### Online Frontend
- New: Dottie Control Plane (web artifact) to check progress from anywhere — displays STATUS.json, WSD+YaRN chart, eval results, LLMVM metrics

Solo personal project, no connection to employer, built with public/free-tier only
