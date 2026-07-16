# Spec 02 — Synthetic Data Generators (B1–B5)

- **Spec ID:** 02_data_generation
- **Worker tier:** Sonnet — FOUR PARALLEL WORKERS, one per generator task (B1, B2, B3, B4).
  Each worker also gets the shared-base section; the FIRST worker to start (or a 5th micro-task)
  lands `dottie/datagen/base.py` + `dottie/datagen/__init__.py`; the others import it. If parallel
  workers collide, each may vendor an identical `base.py` and the foreman keeps one copy —
  the ABC below is the byte-exact contract either way.
- **Dependencies:** 01_environment (env installed, `dottie/` package exists).
- **Consumers:** 03_tokenizer trains on this raw data; the packer and the 5 canonical J-space
  eval tests (Spider→Ant, France→China, Soccer→Rugby, Spanish→French, Safety 0/180) depend on
  B3/B4 content verbatim.

## Purpose

Produce ~140MB+ of fully synthetic, phase-tagged training text for the nano curriculum with
ZERO network access. Everything is generated from templates + RNG; every factual/numeric answer
is computed by the generator so it is correct by construction. Determinism is a hard contract:
same seed → byte-identical JSONL output, verified by sha256.

## Curriculum the data must cover (mirrors dolma_config.yaml, compressed)
P0 logic 5M tok @seq256 · P1 math 6M @256 · P2 foundation 10M @512 · P3 reasoning 4.5M @512 ·
P4 long 1.5M @1024 · P5 anneal 3M @1024. Raw MB targets below assume ~3.5 chars/token headroom.

## Shared base (contract for all four workers)

### dottie/datagen/base.py
```python
class Generator(ABC):
    name: str                      # e.g. "logic"
    def __init__(self, seed: int): ...   # seeds a PRIVATE random.Random(seed); NEVER the global RNG
    @abstractmethod
    def generate(self, target_mb: float) -> Iterator[dict]: ...
```
- Yielded dict schema (all 5 keys required, all values non-empty str):
  `{"text": str, "task_type": str, "concept": str, "phase": str, "source": str}`
  - `task_type ∈ {"automatic","deliberate","safety","temporal"}` (drives J-space routing/selectivity losses)
  - `concept`: short keyword string (e.g. `"spider"`, `"modus ponens"`); its first token id becomes
    the reportability target downstream — pick the most specific single entity/skill in the doc.
  - `phase ∈ {"p0","p1","p2","p3","p4","p5"}`; `source`: generator-specific tag (e.g. `"logic/natded"`).
- `base.py` also provides `write_shards(gen, out_dir, target_mb, shard_mb=8)` → files
  `{out_dir}/{gen.name}_{shard:04d}.jsonl` (UTF-8, `\n`-terminated, `json.dumps(..., ensure_ascii=False, sort_keys=True)`
  — sort_keys mandatory for byte-determinism), stopping once cumulative bytes ≥ target_mb·2^20,
  and returns `{"files": [...], "bytes": int, "docs": int, "sha256": <hex of all shard bytes concatenated in filename order>}`.
- CLI (identical per module, via a shared `base.run_cli(GeneratorCls)`):
  `python -m dottie.datagen.<mod> --seed 1234 --out data/nano/raw/ --mb N`
  → writes shards into `data/nano/raw/`, prints one JSON line: the `write_shards` return dict.
  No tqdm/timestamps/pids in file output (stdout progress is fine but the JSON line is last).
- Determinism rules: private `random.Random` only; no `set()` iteration ordering in doc
  construction (use sorted lists); no floats formatted with locale; no dict iteration on
  py-hash-randomized keys (str keys are fine in 3.11 dicts — insertion-ordered — but be deliberate).
- Doc length: aim 500–4000 chars typical; P4-tagged docs 6000–12000 chars (they feed seq 1024).

## B1 — dottie/datagen/logic.py (worker 1) — P0, ≥30MB
- Doc families (approx mix by MB): truth-table walkthroughs 25% — random propositional formula
  over 2–4 vars (operators ¬ ∧ ∨ → ↔, depth ≤4), full table enumerated row by row, then a prose
  verdict (tautology/contradiction/contingent) computed by actually evaluating the formula.
- Natural-deduction proofs 30%: generate VALID derivations forward — start from 2–5 random
  premises, repeatedly apply modus ponens, modus tollens, ∧-intro/elim, ∨-intro/elim, →-intro
  (via subproof) to derive a conclusion; render numbered lines with rule citations. Correct by
  construction: never write a proof backward from a target.
- Syllogism sets 15% (all 24 valid forms + named invalid forms labeled INVALID with the
  counterexample), FOL statements + instantiations 15% (∀/∃ over small named domains, e.g.
  "domain: {a,b,c}" with explicit instantiation lines), wrong-proof/critique pairs 15%
  (take a valid proof, inject one labeled flaw — affirming the consequent, undistributed middle —
  then a critique paragraph identifying the exact bad line).
- Tags: `phase="p0"`; `task_type="deliberate"` (critique pairs may use `"deliberate"` too;
  ~5% simple truth-table drills may be `"automatic"`). `concept`: the rule or form name
  (`"modus ponens"`, `"truth table"`, `"syllogism"`, ...). `source="logic/<family>"`.

## B2 — dottie/datagen/math_gen.py (worker 2) — P1 + P3, ≥40MB
- P1 (~28MB, `phase="p1"`, task_type `deliberate`; trivial 1-digit drills `automatic`):
  staged curriculum in this order within the corpus — (a) 1–3 digit add/sub/mul with worked
  column-arithmetic steps (carries/borrows shown digit by digit, partial products for mul);
  (b) linear equations `ax+b=c` (int coefficients −20..20, a≠0) solved step-by-step, answer as
  exact fraction when non-integer; (c) geometry facts + computed answers (perimeter/area of
  rectangles, triangles via base·height/2, circle circumference/area with π kept symbolic AND
  approximated to 2dp); (d) modular arithmetic (a mod m tables, congruence solving);
  (e) sequences (arithmetic/geometric: next term, nth term, sum formulas with the plugged numbers);
  (f) probability word problems with chain-of-thought (dice, urns, coins — probabilities as
  reduced fractions). EVERY answer computed in Python by the generator, never templated as text.
- P3 (~12MB, `phase="p3"`): multi-step word problems (2–4 chained operations, unit conversions,
  rate/time/distance) with numbered reasoning steps; plus temporal "workflow log" docs —
  a dated task list with deadlines, then env-delta events ("Day 3: server outage delays task B
  by 2 days"), then a recomputed schedule — these get `task_type="temporal"`,
  `concept` = `"deadline"` / `"schedule"` / `"delay"`. Non-temporal P3 docs: `deliberate`.
- ~10% of P3 docs sized 6000+ chars and tagged `phase="p4"` to feed the long phase.
- `concept`: the skill (`"addition"`, `"linear equation"`, `"probability"`, ...). `source="math/<stage>"`.

## B3 — dottie/datagen/encyclopedia.py + dottie/datagen/code_gen.py (worker 3) — P2, ≥50MB combined
- encyclopedia.py (≥35MB, `phase="p2"`, `task_type="automatic"`): fact corpus with HEAVY
  paraphrase coverage (≥40 distinct sentence templates per fact) of the CANONICAL EVAL ENTITIES —
  the 5 J-space eval tests read this corpus verbatim, so these exact facts must appear thousands
  of times each: spider→8 legs, ant→6 legs; France→Paris/French/Euro/Europe;
  China→Beijing/Mandarin/Yuan/Asia; soccer facts vs rugby facts (11 vs 15 players, round vs oval
  ball, no-hands vs hands, goals vs tries); Spanish/French parallel sentence pairs
  ("ES: ... / FR: ... / EN: ...", ≥200 base sentences × combinatorial fillers). Plus general
  animals (≥60 species: legs, habitat, diet, class), countries (≥60: capital, language, currency,
  continent — internally consistent fixed table), sports (≥15). `concept` = the entity keyword
  exactly: `"spider"`, `"ant"`, `"france"`, `"china"`, `"soccer"`, `"rugby"`, `"spanish"`,
  `"french"`, or the species/country name. `source="ency/<domain>"`.
- code_gen.py (≥15MB, `phase="p2"`, `task_type="deliberate"`, `concept`=function topic e.g.
  `"fibonacci"`, `source="code/pyfunc"`): small Python functions (10–40 lines: string ops, list
  algorithms, math utils, simple classes) each followed by doctests whose expected outputs are
  produced by actually exec()-ing the snippet. Sandbox spec: `exec(code, {"__builtins__": SAFE})`
  where `SAFE` is a whitelist dict of exactly {abs, min, max, sum, len, range, enumerate, zip,
  sorted, reversed, int, float, str, bool, list, dict, set, tuple, print, isinstance, ValueError,
  TypeError}; no import statements in generated code; execution wrapped in
  `signal.alarm(2)`-based timeout (or a `multiprocessing` worker with 2s join) and any
  exception/timeout → discard the candidate and regenerate (deterministically: candidates come
  from the seeded RNG stream, discards do not consume extra entropy from other docs).
- ~10% of encyclopedia docs are 6000+ char "country profile"/"animal compendium" long docs tagged `phase="p4"`.

## B4 — dottie/datagen/chat_safety.py (worker 4) — chat branch + Critic data, ≥20MB
- Dialogue format: turns delimited by literal `<|user|>` and `<|assistant|>` marker strings in
  `text` (these become tokenizer special tokens in spec 03).
- Families: (a) safety scenarios ~35% (`task_type="safety"`, `phase="p5"`): templated abstract
  scenarios where a user pressures the assistant using leverage/blackmail/threat/shutdown/
  survival/scandal/secretly/fake vocabulary (draw from the dolma/nemo watchlist), paired with
  firm refusal-style assistant completions that name the concern and redirect. Keep scenarios
  abstract/entity-templated ("Agent A", "Company X") — this is standard alignment-training data
  (scenario + refusal), never operational instructions for wrongdoing. `concept` = the watchlist
  word present (`"blackmail"`, `"leverage"`, `"threat"`, `"shutdown"`, ...).
- (b) benign near-twins ~20%: same templates with the coercive element swapped for an innocuous
  one (negotiation→scheduling, leverage→feedback) and a helpful completion — `task_type="automatic"`,
  `concept`=topic word — these provide the AUC contrast class for the Safety eval.
- (c) delegation/temporal workflow dialogues ~25% (`task_type="temporal"`, `phase="p3"` or `"p5"`):
  user delegates a multi-step task with deadlines; assistant plans, reports progress across turns,
  handles an injected env-delta. `concept` ∈ {"delegation","deadline","priority"}.
- (d) counterfactual-reflection ~10% (`task_type="deliberate"`, `phase="p5"`): assistant reasons
  "had X been different, Y would…" about a prior turn. (e) plain helpful QA chat ~10%
  (`task_type="automatic"`, `phase="p5"`).
- `source="chat/<family>"`. Safety/benign twins must be distinguishable ONLY by the coercive
  vocabulary, not by length or formatting (matched templates).

## B5 — dottie/datagen/workflow_jobbench.py + dottie/datagen/workflow_gaia2.py (Stage 12) — P3/P4/P5
Unlike B1-B4, B5 was not built alongside the original four (it landed later, once the blueprint's own
`workflow_jobbench`/`workflow_gaia2` mix-weight labels in `dolma_config.yaml`/`streaming_data.py` needed a
real generator behind them). Same base contract, same zero-network/private-RNG/correct-by-construction
rules apply unchanged; wiring is via `configs/sources.yaml` + `dottie/datagen/__init__.py`'s `GENERATORS` dict
(the collector's actual source registry — see `dottie/pipeline/collector.py`), not the `scripts/gen_all_data.py`
orchestrator sketched below (which several phases of this project's history never actually built; B5 does
not depend on it).

- **workflow_jobbench.py** (`WorkflowJobBenchGenerator`, name=`"jobbench"`, phases `(3,4,5)`): modeled on
  the real JobBench benchmark (job-bench.github.io — 1,500+ professionals rating what work they want
  delegated, ~28 occupations across 7 domains, tasks as small dossiers of contradictory heterogeneous
  inputs, binary anchored rubrics) without any network dependency on it. 25 occupations × 3
  planted-contradiction families, each with its own deterministic reconciliation:
  - `duplicate` (`task_type="deliberate"`): a line-item CSV table has one row accidentally duplicated; a
    memo naively sums the raw table. Correct answer = table sum minus the duplicated value.
  - `units` (`task_type="deliberate"`): an itemized, auditable table is contradicted by a summary figure
    that states the SAME raw number but claims different units (thousands/hundreds) — the itemized total
    is always the correct one.
  - `stale` (`task_type="temporal"`): two dated snapshots disagree because an item was added/removed/changed
    between them; a memo cites the older, superseded figure. Correct answer = the later snapshot's total.
  - `concept` = the occupation slug. Phase 3/5 docs use a small `randint` item-count range (500-4000
    chars); phase 4 docs GROW the item count (more RNG-drawn rows, same construction) until the rendered
    text clears 6000 chars, since a fixed count landed anywhere from ~4000 to ~10000 chars depending which
    occupation's value units (dollars vs. small integer counts) got drawn.
- **workflow_gaia2.py** (`WorkflowGaia2Generator`, name=`"gaia2"`, phases `(3,4,5)`, all
  `task_type="temporal"`): modeled on Meta's real Gaia2 benchmark
  (facebookresearch/meta-agents-research-environments — 800 async scenarios across 10 universes, events
  that fire on the environment's own clock independent of the agent) without any network/code dependency on
  the real ARE. A deterministic scheduling state machine — an initial candidate-slot list plus a sequence of
  RNG-seeded async events that each prune or reorder it — over 4 twists mirroring Gaia2's named capability
  axes: `adaptability` (a candidate slot is declined), `ambiguity` (a later explicit time supersedes an
  earlier vague one — NEVER a silent substitution when the explicit slot itself misses the deadline; a
  regression test guards this exact bug), `deadline` (a late constraint prunes the window), `collaboration`
  (a second agent's already-made booking must be accepted or flagged, never silently duplicated). The
  "Resolution:" text is always the literal output of replaying the state machine against the parsed
  slots/deadline/events, so it is independently checkable from the rendered text alone — see
  `tests/test_datagen.py`'s `test_gaia2_*_resolution_is_correct`/`*_is_correct` tests, which parse the
  doc's own numbers back out via regex and recompute the expected resolution rather than trusting the
  generator's internal variables. Phase 4 docs chain further independent scenarios ("same universe, later
  that day") until long enough, since a single scenario averages ~650 chars.
- Both generators' phase mix (3/4/5 weight split) intentionally differs: jobbench skews toward phase 3
  (reasoning), gaia2 skews toward phase 4 (long/async context) — this mirrors the blueprint's own original
  framing in `inner_monologue_research.md` (JobBench closer to S2/deliberate reasoning; GAIA2 closer to the
  Planner's long-horizon temporal hold) and is realized concretely in `configs/sources.yaml`'s per-phase
  `weight` fields (jobbench 10/10/5% at p3/p4/p5, gaia2 5/15/10%), with the phase's other sources rescaled
  down proportionally so each phase's weights still sum to 1.0.
- Acceptance, same shape as B1-B4: `python -m dottie.datagen.workflow_jobbench --seed 1234 --out /tmp/dg --mb 5`
  and the `workflow_gaia2` equivalent exit 0 and are byte-reproducible across two runs at the same seed;
  `pytest tests/test_datagen.py -k "jobbench or gaia2"` green (16 tests: determinism/schema via
  `ALL_GENERATORS`, task_type accuracy, per-family reconciliation math re-derived independently, phase-4
  long-doc char-band, and the four GAIA2 state-machine replay tests).

## Orchestrator + tests (owned by whichever worker the foreman assigns last, or B4)
- `scripts/gen_all_data.py --seed 1234 [--out data/nano/raw/]`: runs all generators with fixed
  derived seeds `{logic: seed+1, math_gen: seed+2, encyclopedia: seed+3, code_gen: seed+4,
  chat_safety: seed+5}` and MB targets `{logic:30, math_gen:40, encyclopedia:35, code_gen:15,
  chat_safety:20}`; writes `data/nano/raw/manifest.json` = mapping module → the write_shards
  dict (sorted keys). Sequential execution is fine (RAM-light: generators must stream, never
  hold >64MB in memory).
- `tests/test_datagen.py` (marked so full-size runs are skipped: use `--mb 1` in tests):
  `test_determinism_<mod>` — run each generator twice at `--mb 1` seed 1234 into two temp dirs,
  assert per-shard sha256 equality; `test_schema` — every line parses, has exactly the 5 keys,
  task_type in the enum, concept/phase/source non-empty; `test_phase_coverage` — union of phases
  across all generators at 1MB each ⊇ {p0,p1,p2,p3,p4,p5}; `test_markers` — chat docs contain
  both `<|user|>` and `<|assistant|>`; `test_canonical_entities` — 1MB encyclopedia sample
  contains all of: "8 legs" near "spider", "6 legs" near "ant", "Paris", "Beijing", "Mandarin",
  "Yuan", "Euro", "rugby", and a `concept=="france"` doc.

## Acceptance criteria (foreman runs, per generator worker)
1. `python -m dottie.datagen.<mod> --seed 1234 --out /tmp/dg_a --mb <target>` exits 0 in <20 min,
   final JSON line reports `bytes >= target*1048576`.
2. Re-run into `/tmp/dg_b` with the same seed; `diff <(cd /tmp/dg_a && sha256sum *.jsonl) <(cd /tmp/dg_b && sha256sum *.jsonl)` → empty.
3. `python - <<'EOF'` spot-check: sample 200 lines, json.loads each, assert schema keys + enum. EOF
4. `pytest tests/test_datagen.py -k <mod>` → green.
5. Full pipeline: `python scripts/gen_all_data.py --seed 1234` → `data/nano/raw/manifest.json`
   exists, total bytes ≥ 140·2^20, `du -sh data/nano/raw` ≤ 400MB (don't blow the disk).
6. B2 spot-check: grep 20 arithmetic answers and re-verify with python — 20/20 correct.
   B3 code: extract 20 doctests, run them with `python -m doctest` semantics — 20/20 pass.

## Out of scope
- Tokenization/packing (specs 03+), dedup/quality filtering, any dolma/nemo-curator usage,
  network access of any kind, LLM-generated text (templates+RNG only), modifying
  logic_textbook_pipeline.py (blueprint stays untouched), git commits.
