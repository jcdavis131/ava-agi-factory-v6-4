"""Self-contained live pipeline dashboard HTML (no CDN)."""

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Ava Pipeline</title>
<style>
:root {
  --bg: #f4f2ec;
  --ink: #1a1a18;
  --muted: #5c5a52;
  --line: #d4d0c4;
  --card: #fffcf5;
  --ok: #2f6b3a;
  --warn: #9a6b12;
  --bad: #8b2e2e;
  --accent: #1e4d6b;
  --bar: #c8c2b4;
  --bar-ok: #5a8f64;
  --bar-warn: #c4a04a;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
  background: var(--bg);
  color: var(--ink);
  min-height: 100vh;
}
header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding: 1.25rem 1.5rem;
  border-bottom: 1px solid var(--line);
  background: var(--card);
}
header h1 {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 600;
  letter-spacing: -0.02em;
}
header .meta { color: var(--muted); font-size: 0.85rem; font-variant-numeric: tabular-nums; }
main {
  display: grid;
  grid-template-columns: 1.15fr 1fr;
  gap: 1rem;
  padding: 1rem 1.5rem 2rem;
}
@media (max-width: 960px) { main { grid-template-columns: 1fr; } }
.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 2px;
  padding: 1rem 1.1rem;
}
.card h2 {
  margin: 0 0 0.75rem;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  font-weight: 600;
}
.span2 { grid-column: 1 / -1; }
.row { display: flex; gap: 0.75rem; flex-wrap: wrap; }
.stat {
  flex: 1 1 7rem;
  min-width: 6.5rem;
  padding: 0.6rem 0.7rem;
  border: 1px solid var(--line);
  background: var(--bg);
}
.stat .k { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
.stat .v { font-size: 1.35rem; font-variant-numeric: tabular-nums; font-weight: 600; margin-top: 0.15rem; }
.stat .v.sm { font-size: 1rem; }
.stat .sub { font-size: 0.7rem; color: var(--muted); margin-top: 0.2rem; }
.pill {
  display: inline-block;
  padding: 0.15rem 0.5rem;
  font-size: 0.75rem;
  border: 1px solid var(--line);
  font-variant-numeric: tabular-nums;
}
.pill.ok { color: var(--ok); border-color: #9cbc9f; background: #e8f0e9; }
.pill.warn { color: var(--warn); border-color: #d4b56a; background: #f7efd8; }
.pill.bad { color: var(--bad); border-color: #c99; background: #f5e8e8; }
.mode {
  display: flex;
  gap: 1rem;
  align-items: flex-start;
  padding: 0.85rem 1rem;
  border: 1px solid var(--line);
  background: var(--bg);
  margin-bottom: 0.85rem;
}
.mode .title { font-weight: 600; font-size: 1.05rem; }
.mode .detail { color: var(--muted); font-size: 0.85rem; margin-top: 0.2rem; }
.gates { display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.45rem; }
@media (max-width: 960px) { .gates { grid-template-columns: repeat(2, 1fr); } }
.gate {
  border: 1px solid var(--line);
  padding: 0.55rem 0.6rem;
  background: var(--bg);
}
.gate.ok { border-color: #9cbc9f; }
.gate.bad { border-color: #c99; background: #faf2f2; }
.gate .id { font-size: 0.65rem; color: var(--muted); letter-spacing: 0.06em; text-transform: uppercase; }
.gate .name { font-size: 0.8rem; font-weight: 600; margin: 0.15rem 0; }
.gate .val { font-size: 0.85rem; font-variant-numeric: tabular-nums; }
.gate .tgt { font-size: 0.7rem; color: var(--muted); margin-top: 0.15rem; }
.bar-row { display: grid; grid-template-columns: 7.5rem 1fr 3.5rem; gap: 0.5rem; align-items: center; margin: 0.35rem 0; font-size: 0.85rem; font-variant-numeric: tabular-nums; }
.bar-track { height: 10px; background: var(--bar); position: relative; }
.bar-fill { height: 100%; background: var(--accent); }
.funnel { display: grid; grid-template-columns: repeat(6, 1fr); gap: 0.4rem; margin-bottom: 0.75rem; }
@media (max-width: 960px) { .funnel { grid-template-columns: repeat(3, 1fr); } }
.funnel .cell {
  border: 1px solid var(--line);
  padding: 0.45rem;
  text-align: center;
  background: var(--bg);
}
.funnel .lab { font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
.funnel .num { font-size: 1.05rem; font-weight: 600; font-variant-numeric: tabular-nums; }
.glossary { margin-top: 0.75rem; display: grid; gap: 0.35rem; }
.glossary .g-row {
  display: grid;
  grid-template-columns: 8.5rem 3rem 1fr;
  gap: 0.5rem;
  align-items: start;
  font-size: 0.8rem;
  padding: 0.35rem 0.4rem;
  border-bottom: 1px solid var(--line);
}
.glossary .g-state { font-weight: 600; font-variant-numeric: tabular-nums; }
.glossary .g-n { font-variant-numeric: tabular-nums; color: var(--muted); }
.glossary .g-help { color: var(--muted); }
.curric {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 0.4rem;
  margin-top: 0.5rem;
}
@media (max-width: 960px) { .curric { grid-template-columns: repeat(2, 1fr); } }
.curric .cphase {
  border: 1px solid var(--line);
  padding: 0.55rem 0.5rem;
  background: var(--bg);
  min-height: 5.5rem;
}
.curric .cphase.active {
  border-color: var(--accent);
  box-shadow: inset 0 0 0 1px var(--accent);
  background: #eef4f8;
}
.curric .cphase.done { opacity: 0.55; }
.curric .cname { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
.curric .ctitle { font-weight: 600; font-size: 0.9rem; margin: 0.15rem 0; }
.curric .cmeta { font-size: 0.72rem; color: var(--muted); font-variant-numeric: tabular-nums; }
.curric .cmix { font-size: 0.68rem; color: var(--muted); margin-top: 0.35rem; line-height: 1.35; }
.curric .cfill { margin-top: 0.4rem; height: 6px; background: var(--bar); }
.curric .cfill > span { display: block; height: 100%; background: var(--accent); }
.hints { margin: 0.6rem 0 0; padding: 0; list-style: none; }
.hints li {
  font-size: 0.82rem;
  padding: 0.35rem 0.5rem;
  border-left: 3px solid var(--accent);
  background: var(--bg);
  margin: 0.25rem 0;
  color: var(--ink);
}
.hints li.warn { border-left-color: var(--warn); }
.phase-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 0.4rem; }
.phase {
  border: 1px solid var(--line);
  padding: 0.5rem;
  text-align: center;
  background: var(--bg);
  position: relative;
}
.phase.trainer { border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); }
.phase.target:not(.trainer) { border-color: var(--warn); }
.phase .p { font-size: 0.7rem; color: var(--muted); }
.phase .n { font-size: 0.95rem; font-variant-numeric: tabular-nums; font-weight: 600; }
.phase .fill {
  margin-top: 0.35rem;
  height: 6px;
  background: var(--bar);
}
.phase .fill > span { display: block; height: 100%; background: var(--bar-ok); }
.phase.thin .fill > span { background: var(--bar-warn); }
.phase .tag {
  font-size: 0.6rem;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-top: 0.25rem;
  min-height: 0.85rem;
}
svg.chart { width: 100%; height: 180px; display: block; background: var(--bg); border: 1px solid var(--line); }
.legend { display: flex; gap: 1rem; font-size: 0.75rem; color: var(--muted); margin-top: 0.4rem; flex-wrap: wrap; }
.legend i { display: inline-block; width: 12px; height: 3px; vertical-align: middle; margin-right: 4px; }
.muted { color: var(--muted); font-size: 0.85rem; }
.alert {
  margin: 0 0 0.75rem;
  padding: 0.55rem 0.7rem;
  border: 1px solid #d4b56a;
  background: #f7efd8;
  color: var(--warn);
  font-size: 0.85rem;
}
.alert.bad { border-color: #c99; background: #f5e8e8; color: var(--bad); }
table { width: 100%; border-collapse: collapse; font-size: 0.85rem; font-variant-numeric: tabular-nums; }
td, th { text-align: left; padding: 0.35rem 0.4rem; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 500; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; }
.probe { display: flex; gap: 0.5rem; margin-top: 0.5rem; }
.probe input {
  flex: 1;
  padding: 0.5rem 0.6rem;
  border: 1px solid var(--line);
  background: var(--bg);
  font: inherit;
}
.probe button {
  padding: 0.5rem 0.9rem;
  border: 1px solid var(--accent);
  background: var(--accent);
  color: #fff;
  font: inherit;
  cursor: pointer;
}
pre.out {
  margin: 0.6rem 0 0;
  padding: 0.6rem;
  background: var(--bg);
  border: 1px solid var(--line);
  font-size: 0.75rem;
  max-height: 180px;
  overflow: auto;
  white-space: pre-wrap;
}
.kv { display: grid; grid-template-columns: 7.5rem 1fr; gap: 0.25rem 0.6rem; font-size: 0.85rem; font-variant-numeric: tabular-nums; }
.kv .k { color: var(--muted); }
a { color: var(--accent); }
</style>
</head>
<body>
<header>
  <h1>Ava pipeline</h1>
  <div class="meta">
    <span id="preset">—</span>
    · <span id="clock">—</span>
    · poll 3s
    · <a href="/pipeline/status">json</a>
    · <a href="/health">/health</a>
    · <a href="/jspace/viewer">viewer</a>
    · <a href="/report">report</a>
  </div>
</header>
<main>
  <section class="card span2">
    <h2>Operator view</h2>
    <div id="modeBanner"></div>
    <div class="gates" id="gates"></div>
    <div class="row" id="topStats" style="margin-top:0.85rem"></div>
  </section>

  <section class="card span2">
    <h2>Curriculum — where we are</h2>
    <div id="curriculumHero"></div>
    <div class="curric" id="curriculum"></div>
    <p class="muted" id="curriculumCaption" style="margin:0.75rem 0 0">Six-phase logic-first curriculum. Active phase is outlined.</p>
  </section>

  <section class="card span2">
    <h2>Closed-loop demand (train → miners)</h2>
    <div id="demandPanel"></div>
    <p class="muted" style="margin:0.75rem 0 0">Only collectors fetch outside data. Trainer publishes expand / curate / examples; miners reweight sources.</p>
  </section>

  <section class="card">
    <h2>Shard lifecycle</h2>
    <div id="prepAlerts"></div>
    <div class="funnel" id="funnel"></div>
    <div class="glossary" id="lifecycleGlossary"></div>
  </section>

  <section class="card">
    <h2>Packed runway by phase</h2>
    <div class="phase-grid" id="phases"></div>
    <p class="muted" id="runwayCaption" style="margin:0.75rem 0 0">Fill = tokens vs packed_min. Outlined = trainer phase; amber = collector target.</p>
  </section>

  <section class="card span2">
    <h2>Training — current run</h2>
    <div id="trainAlerts"></div>
    <div class="row" id="trainStats"></div>
    <svg class="chart" id="lossChart" viewBox="0 0 800 180" preserveAspectRatio="none" style="margin-top:0.75rem"></svg>
    <div class="legend">
      <span><i style="background:#1e4d6b"></i>lm_loss</span>
      <span><i style="background:#2f6b3a"></i>tok/s (scaled)</span>
      <span class="muted">chart resets on trainer restart</span>
    </div>
    <div class="kv" id="trainDetail" style="margin-top:0.75rem"></div>
    <h2 style="margin-top:1rem">Watch signals</h2>
    <div class="row" id="watchStats"></div>
    <ul class="hints" id="watchHints"></ul>
    <p class="muted" id="trainCaption">Source: metrics jsonl</p>
  </section>

  <section class="card">
    <h2>Checkpoints</h2>
    <table>
      <thead><tr><th>File</th><th>MB</th><th>Age</th></tr></thead>
      <tbody id="ckpts"></tbody>
    </table>
  </section>

  <section class="card">
    <h2>Live inspect</h2>
    <p class="muted">Forward pass on the hot-reloaded checkpoint (needs engine boot). Mass / route should change with input.</p>
    <div class="probe">
      <input id="probeText" value="The number of legs on the animal that spins webs is"/>
      <button id="probeBtn" type="button">Inspect</button>
    </div>
    <pre class="out" id="probeOut">—</pre>
  </section>
</main>
<script>
const STATES = ["RAW","CLAIMED_CURATE","PACKED","CLAIMED_TRAIN","CONSUMED","FAILED","DELETED"];
let lastPayload = null;

function fmt(n) {
  if (n == null || Number.isNaN(n)) return "—";
  if (Math.abs(n) >= 1e9) return (n/1e9).toFixed(2) + "B";
  if (Math.abs(n) >= 1e6) return (n/1e6).toFixed(2) + "M";
  if (Math.abs(n) >= 1e3) return (n/1e3).toFixed(1) + "k";
  return String(n);
}
function fmtTs(t) {
  try { return new Date(t * 1000).toLocaleTimeString(); } catch { return "—"; }
}
function fmtAge(s) {
  if (s == null) return "—";
  if (s < 60) return Math.round(s) + "s";
  if (s < 3600) return Math.round(s/60) + "m";
  return (s/3600).toFixed(1) + "h";
}
function pillClass(ok, warn) {
  if (ok === false) return "bad";
  if (warn) return "warn";
  return "ok";
}

function renderMode(d) {
  const m = d.mode || {};
  const id = m.id || "unknown";
  const cls = id === "training" ? "ok" : (id === "data_prep" ? "warn" : "bad");
  document.getElementById("modeBanner").innerHTML = `
    <div class="mode">
      <div>
        <div class="title"><span class="pill ${cls}">${m.label || id}</span></div>
        <div class="detail">${m.detail || ""}</div>
      </div>
    </div>`;
}

function renderGates(d) {
  const gates = (d.flow && d.flow.gates) || [];
  document.getElementById("gates").innerHTML = gates.map(g => `
    <div class="gate ${g.ok ? "ok" : "bad"}">
      <div class="id">${g.id}</div>
      <div class="name">${g.name}</div>
      <div class="val">${g.value || "—"}</div>
      <div class="tgt">${g.target || ""}</div>
    </div>`).join("");
}

function renderTop(d) {
  const m = d.manifest || {};
  const tr = d.trainer || {};
  const last = tr.last || {};
  const flow = d.flow || {};
  const disk = d.disk || {};
  const starved = !!tr.data_starved;
  const pause = flow.collector_pause || {};
  const health = starved ? "bad" : (tr.stale ? "warn" : (m.ok ? "ok" : "warn"));
  const healthLabel = starved ? "DATA_STARVED" : (tr.stale ? "STALE" : (m.ok ? "healthy" : "manifest?"));
  const step = last.step != null ? last.step : "—";
  const loss = last.lm_loss != null ? Number(last.lm_loss).toFixed(3) : "—";
  const toks = last.tok_s != null ? Math.round(last.tok_s) : "—";
  const free = disk.free_gb != null ? disk.free_gb : d.disk_free_gb;
  const freeCls = disk.below_low_water ? "bad" : "ok";
  document.getElementById("preset").textContent = d.preset || "—";
  document.getElementById("clock").textContent = fmtTs(d.ts);
  document.getElementById("topStats").innerHTML = `
    <div class="stat"><div class="k">Loop</div><div class="v sm"><span class="pill ${health}">${healthLabel}</span></div>
      <div class="sub">${flow.data_detail || ""}</div></div>
    <div class="stat"><div class="k">Host free</div><div class="v sm"><span class="pill ${freeCls}">${free != null ? free + " GB" : "—"}</span></div>
      <div class="sub">probe ${disk.probe || "—"} · low ${disk.low_water_gb ?? "—"}</div></div>
    <div class="stat"><div class="k">Collectors</div><div class="v sm"><span class="pill ${pause.paused ? "warn" : "ok"}">${pause.paused ? "paused" : "running"}</span></div>
      <div class="sub">${pause.reason || "feeding target phase"}</div></div>
    <div class="stat"><div class="k">Trainer phase</div><div class="v">${flow.trainer_phase != null ? flow.trainer_phase : "—"}</div>
      <div class="sub">${(d.watch && d.watch.phase_progress && d.watch.phase_progress.short) || "target P"+(flow.target_phase != null ? flow.target_phase : "—")}</div></div>
    <div class="stat"><div class="k">Step</div><div class="v">${step}</div>
      <div class="sub">age ${fmtAge(tr.age_s)}</div></div>
    <div class="stat"><div class="k">lm loss</div><div class="v sm">${loss}</div>
      <div class="sub">${toks} tok/s</div></div>
    <div class="stat"><div class="k">Raw backlog</div><div class="v sm">${m.raw_gb != null ? m.raw_gb + " GB" : "—"}</div>
      <div class="sub">${Math.round((m.raw_fill || 0)*100)}% of max ${m.raw_max_gb ?? "—"} GB</div></div>
    <div class="stat"><div class="k">Ckpt</div><div class="v sm">${(d.ckpt && d.ckpt.latest_pointer) || "—"}</div>
      <div class="sub">tok ${m.tokenizer_sha || "—"}</div></div>
  `;
}

function mixStr(mix) {
  if (!mix) return "—";
  return Object.entries(mix).sort((a,b) => b[1]-a[1])
    .map(([k,v]) => `${k} ${(v*100).toFixed(0)}%`).join(" · ");
}

function renderCurriculum(d) {
  const cur = d.curriculum;
  const flow = d.flow || {};
  const watch = d.watch || {};
  const hero = document.getElementById("curriculumHero");
  const grid = document.getElementById("curriculum");
  if (!cur || !cur.phases) {
    hero.innerHTML = `<p class="muted">Curriculum preset not loaded.</p>`;
    grid.innerHTML = "";
    return;
  }
  const pp = watch.phase_progress || {};
  const rp = watch.run_progress || {};
  const active = flow.trainer_phase != null ? flow.trainer_phase : pp.phase;
  const activePh = cur.phases.find(p => p.index === active) || cur.phases[0];
  const pctPhase = pp.frac != null ? (pp.frac * 100).toFixed(1) : "—";
  const pctRun = rp.frac != null ? (rp.frac * 100).toFixed(2) : "—";
  hero.innerHTML = `
    <div class="mode">
      <div>
        <div class="title">Now training:
          <span class="pill ok">P${active} · ${activePh.short || activePh.name}</span>
        </div>
        <div class="detail">
          seq ${activePh.seq} · rope ${activePh.rope_base} · mix ${mixStr(activePh.mix)}
          · phase progress ${pctPhase}% (${fmt(pp.tokens_in_phase)} / ${fmt(pp.phase_tokens)})
          · run ${pctRun}% of ${fmt(cur.tokens_total)} tokens
          · next ckpt in ${watch.steps_to_ckpt != null ? watch.steps_to_ckpt : "—"} steps
        </div>
      </div>
    </div>`;
  grid.innerHTML = cur.phases.map(p => {
    const isActive = p.index === active;
    const isDone = active != null && p.index < active;
    const frac = (isActive && pp.frac != null) ? pp.frac
      : (isDone ? 1 : 0);
    const cls = ["cphase", isActive ? "active" : "", isDone ? "done" : ""].filter(Boolean).join(" ");
    return `<div class="${cls}">
      <div class="cname">P${p.index}</div>
      <div class="ctitle">${p.short || p.name}</div>
      <div class="cmeta">${fmt(p.tokens)} tok · seq ${p.seq}</div>
      <div class="cfill"><span style="width:${Math.round(frac*100)}%"></span></div>
      <div class="cmix">${mixStr(p.mix)}</div>
    </div>`;
  }).join("");
  document.getElementById("curriculumCaption").textContent =
    `Preset ${d.preset} · ${fmt(cur.tokens_per_step)} tok/step · ckpt every ${cur.checkpoint_every_steps} · lr ${cur.lr_max}→${cur.lr_min}`;
}

function renderPrep(d) {
  const flow = d.flow || {};
  const pause = flow.collector_pause || {};
  const m = d.manifest || {};
  const funnel = m.funnel || {};
  const life = d.lifecycle || {};
  const help = life.help || {};
  const order = life.order || ["RAW","CLAIMED_CURATE","PACKED","CLAIMED_TRAIN","CONSUMED","FAILED","DELETED"];
  const by = m.by_state || {};
  const alerts = [];
  if (pause.paused) alerts.push(`<div class="alert">Collectors paused: ${pause.reason || "unknown"}</div>`);
  if (d.disk && d.disk.below_low_water) alerts.push(`<div class="alert bad">Host disk below low-water (${d.disk.free_gb} GB &lt; ${d.disk.low_water_gb} GB)</div>`);
  if ((funnel.failed || 0) > 0) alerts.push(`<div class="alert">FAILED shards: ${funnel.failed} — check curator logs</div>`);
  document.getElementById("prepAlerts").innerHTML = alerts.join("");

  const cells = [
    ["RAW", funnel.raw],
    ["Curating", funnel.curating],
    ["Packed", funnel.packed],
    ["Training", funnel.training],
    ["Consumed", funnel.consumed],
    ["Failed", funnel.failed],
  ];
  document.getElementById("funnel").innerHTML = cells.map(([lab, n]) =>
    `<div class="cell"><div class="lab">${lab}</div><div class="num">${fmt(n || 0)}</div></div>`
  ).join("");

  document.getElementById("lifecycleGlossary").innerHTML = order.map(s => {
    const n = by[s] || 0;
    return `<div class="g-row"><div class="g-state">${s}</div><div class="g-n">${n}</div><div class="g-help">${help[s] || ""}</div></div>`;
  }).join("");
}

function renderPhases(d) {
  const flow = d.flow || {};
  const runway = flow.phase_runway || [];
  const cur = d.curriculum;
  const minTok = flow.packed_min_tokens || 0;
  const nameOf = (p) => {
    if (!cur || !cur.phases) return `P${p}`;
    const ph = cur.phases.find(x => x.index === p);
    return ph ? `P${p} ${ph.short}` : `P${p}`;
  };
  if (!runway.length) {
    const t = (d.manifest && d.manifest.tokens_ready_by_phase) || {};
    document.getElementById("phases").innerHTML = [0,1,2,3,4,5].map(p => {
      const n = t[String(p)] || 0;
      return `<div class="phase"><div class="p">${nameOf(p)}</div><div class="n">${fmt(n)}</div></div>`;
    }).join("");
    return;
  }
  document.getElementById("phases").innerHTML = runway.map(r => {
    const cls = [
      "phase",
      r.ok ? "" : "thin",
      r.is_trainer ? "trainer" : "",
      r.is_target ? "target" : "",
    ].filter(Boolean).join(" ");
    const tag = r.is_trainer ? "trainer" : (r.is_target ? "collect" : "");
    const pct = Math.round((r.fill || 0) * 100);
    return `<div class="${cls}">
      <div class="p">${nameOf(r.phase)}</div>
      <div class="n">${fmt(r.tokens)}</div>
      <div class="fill"><span style="width:${pct}%"></span></div>
      <div class="tag">${tag}</div>
    </div>`;
  }).join("");
  document.getElementById("runwayCaption").textContent =
    `Fill = tokens / packed_min (${fmt(minTok)}). Outlined = trainer phase; amber = collector target.`;
}

function renderWatch(d) {
  const w = d.watch || {};
  const last = (d.trainer && d.trainer.last) || {};
  const hl = last.hl_est || {};
  const dom = w.dominant_route || {};
  document.getElementById("watchStats").innerHTML = `
    <div class="stat"><div class="k">Dominant route</div><div class="v sm">${dom.name || "—"} ${dom.p != null ? (dom.p*100).toFixed(0)+"%" : ""}</div>
      <div class="sub">entropy ${w.route_entropy != null ? w.route_entropy : "—"} bits</div></div>
    <div class="stat"><div class="k">J-aux share</div><div class="v sm">${w.j_aux_share != null ? (w.j_aux_share*100).toFixed(0)+"%" : "—"}</div>
      <div class="sub">of total loss</div></div>
    <div class="stat"><div class="k">Δ lm (log)</div><div class="v sm">${w.lm_delta_10 != null ? (w.lm_delta_10 > 0 ? "+" : "")+w.lm_delta_10 : "—"}</div>
      <div class="sub">vs prior step log</div></div>
    <div class="stat"><div class="k">grad</div><div class="v sm">${w.grad_vs_clip != null ? w.grad_vs_clip : "—"}</div>
      <div class="sub">clip target ~1.0</div></div>
    <div class="stat"><div class="k">mass</div><div class="v sm">${last.verbalizable_mass != null ? Number(last.verbalizable_mass).toFixed(3) : "—"}</div>
      <div class="sub">broadcast ${last.broadcast_strength != null ? Number(last.broadcast_strength).toFixed(3) : "—"}</div></div>
    <div class="stat"><div class="k">half-lives</div><div class="v sm">${hl.system1 != null ? Math.round(hl.system1)+"/"+Math.round(hl.system2||0) : "—"}</div>
      <div class="sub">S1/S2 · C ${hl.critic != null ? Math.round(hl.critic) : "—"} · P ${hl.planner != null ? Math.round(hl.planner) : "—"}</div></div>
  `;
  const hints = w.hints || [];
  document.getElementById("watchHints").innerHTML = hints.map(h => {
    const warn = /rising|collapsed|high|low|diluted|FAILED|starv/i.test(h);
    return `<li class="${warn ? "warn" : ""}">${h}</li>`;
  }).join("");
}

function poly(xs, ys, w, h, pad) {
  if (!xs.length) return "";
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const yvals = ys.filter(v => v != null);
  if (!yvals.length) return "";
  const ymin = Math.min(...yvals), ymax = Math.max(...yvals);
  const dx = Math.max(1e-9, xmax - xmin);
  const dy = Math.max(1e-9, ymax - ymin);
  const pts = [];
  for (let i = 0; i < xs.length; i++) {
    if (ys[i] == null) continue;
    const x = pad + (w - 2*pad) * ((xs[i] - xmin) / dx);
    const y = h - pad - (h - 2*pad) * ((ys[i] - ymin) / dy);
    pts.push(x.toFixed(1) + "," + y.toFixed(1));
  }
  return pts.join(" ");
}

function renderTrain(d) {
  const tr = d.trainer || {};
  const last = tr.last || {};
  const alerts = [];
  if (tr.data_starved) alerts.push(`<div class="alert bad">DATA_STARVED — no packed tokens for trainer phase</div>`);
  if (tr.stale) alerts.push(`<div class="alert">Trainer stale (${fmtAge(tr.age_s)} since last step)</div>`);
  document.getElementById("trainAlerts").innerHTML = alerts.join("");

  const route = last.route_probs || [];
  const routeStr = route.length
    ? route.map((p,i) => `R${i}:${(Number(p)*100).toFixed(0)}%`).join(" ")
    : "—";
  document.getElementById("trainStats").innerHTML = `
    <div class="stat"><div class="k">Step</div><div class="v">${last.step ?? "—"}</div></div>
    <div class="stat"><div class="k">Tokens seen</div><div class="v sm">${fmt(last.tokens)}</div></div>
    <div class="stat"><div class="k">lm / total</div><div class="v sm">${last.lm_loss != null ? Number(last.lm_loss).toFixed(3) : "—"} / ${last.total != null ? Number(last.total).toFixed(3) : "—"}</div></div>
    <div class="stat"><div class="k">tok/s</div><div class="v sm">${last.tok_s != null ? Math.round(last.tok_s) : "—"}</div></div>
    <div class="stat"><div class="k">lr</div><div class="v sm">${last.lr != null ? Number(last.lr).toExponential(1) : "—"}</div></div>
    <div class="stat"><div class="k">grad</div><div class="v sm">${last.grad_norm != null ? Number(last.grad_norm).toFixed(2) : "—"}</div></div>
  `;
  document.getElementById("trainDetail").innerHTML = `
    <div class="k">broadcast</div><div>${last.broadcast != null ? Number(last.broadcast).toFixed(4) : "—"}</div>
    <div class="k">report</div><div>${last.report != null ? Number(last.report).toFixed(3) : "—"}</div>
    <div class="k">routing</div><div>${last.routing != null ? Number(last.routing).toFixed(4) : "—"}</div>
    <div class="k">mass</div><div>${last.verbalizable_mass != null ? Number(last.verbalizable_mass).toFixed(3) : "—"}</div>
    <div class="k">routes</div><div>${routeStr}</div>
    <div class="k">phase name</div><div>${last.phase != null ? "P"+last.phase : "—"}</div>
  `;

  const s = tr.series || {};
  const steps = [], losses = [], stepTok = [], toks = [];
  for (let i = 0; i < (s.step || []).length; i++) {
    if (s.lm_loss && s.lm_loss[i] != null && s.step[i] != null) {
      steps.push(s.step[i]);
      losses.push(s.lm_loss[i]);
    }
    if (s.tok_s && s.tok_s[i] != null && s.step[i] != null) {
      stepTok.push(s.step[i]);
      toks.push(s.tok_s[i]);
    }
  }
  const svg = document.getElementById("lossChart");
  const w = 800, h = 180, pad = 14;
  let html = "";
  if (steps.length >= 2) {
    html += `<polyline fill="none" stroke="#1e4d6b" stroke-width="2" points="${poly(steps, losses, w, h, pad)}"/>`;
  }
  if (stepTok.length >= 2 && losses.length) {
    const tmin = Math.min(...toks), tmax = Math.max(...toks);
    const lmin = Math.min(...losses), lmax = Math.max(...losses);
    const scaled = toks.map(t => lmin + (lmax - lmin) * ((t - tmin) / Math.max(1e-9, tmax - tmin)));
    html += `<polyline fill="none" stroke="#2f6b3a" stroke-width="1.5" stroke-dasharray="4 3" points="${poly(stepTok, scaled, w, h, pad)}"/>`;
  }
  if (!html) {
    html = `<text x="20" y="90" fill="#5c5a52" font-size="14">No steps in current run yet — data prep or waiting for trainer</text>`;
  }
  svg.innerHTML = html;
  document.getElementById("trainCaption").textContent =
    `Source: ${tr.metrics_path || "metrics"} · ${steps.length} points in current run · ${tr.n_points || 0} recent jsonl lines`;
}

function renderCkpts(d) {
  const files = (d.ckpt && d.ckpt.files) || [];
  const latest = d.ckpt && d.ckpt.latest_pointer;
  document.getElementById("ckpts").innerHTML = files.length
    ? files.map(f => {
        const mark = f.name === latest ? " ← latest" : "";
        return `<tr><td>${f.name}${mark}</td><td>${f.mb}</td><td>${fmtAge(f.age_s)}</td></tr>`;
      }).join("")
    : `<tr><td colspan="3" class="muted">No .pt files</td></tr>`;
}

function renderDemand(d) {
  const dem = d.demand;
  const el = document.getElementById("demandPanel");
  if (!dem) {
    el.innerHTML = `<div class="alert">No demand.json yet — waiting for trainer heartbeat to publish.</div>`;
    return;
  }
  const phases = dem.phases || [];
  const boost = dem.boost_task_types || {};
  const boostStr = Object.keys(boost).length
    ? Object.entries(boost).map(([k,v]) => `${k}×${v}`).join(", ")
    : "—";
  const reasonStr = (dem.reasons || []).join(" · ") || "—";
  el.innerHTML = `
    <div class="row">
      <div class="stat"><div class="k">Demand step</div><div class="v">${dem.step ?? "—"}</div>
        <div class="sub">age ${fmtAge(dem.age_s)} · phase P${dem.trainer_phase ?? "—"}</div></div>
      <div class="stat"><div class="k">Curate stricter</div><div class="v sm"><span class="pill ${dem.curate_stricter ? "warn" : "ok"}">${dem.curate_stricter ? "yes" : "no"}</span></div></div>
      <div class="stat"><div class="k">Task boosts</div><div class="v sm">${boostStr}</div></div>
      <div class="stat"><div class="k">Reasons</div><div class="v sm">${reasonStr}</div></div>
    </div>
    <div class="phase-grid" style="margin-top:0.75rem">
      ${[0,1,2,3,4,5].map(p => {
        const row = phases.find(x => x.phase === p) || {};
        const acts = (row.actions || []).join(",") || "—";
        const eff = row.effort != null ? (row.effort*100).toFixed(0)+"%" : "0%";
        return `<div class="phase ${p === dem.trainer_phase ? "trainer" : ""}">
          <div class="p">P${p}</div>
          <div class="n">${eff}</div>
          <div class="tag">${acts}</div>
        </div>`;
      }).join("")}
    </div>`;
}

async function refresh() {
  try {
    const r = await fetch("/pipeline/status");
    const d = await r.json();
    lastPayload = d;
    renderMode(d);
    renderGates(d);
    renderTop(d);
    renderCurriculum(d);
    renderDemand(d);
    renderPrep(d);
    renderPhases(d);
    renderTrain(d);
    renderWatch(d);
    renderCkpts(d);
  } catch (e) {
    document.getElementById("clock").textContent = "fetch failed";
  }
}

document.getElementById("probeBtn").onclick = async () => {
  const text = document.getElementById("probeText").value;
  document.getElementById("probeOut").textContent = "…";
  try {
    const r = await fetch("/jspace/inspect", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text}),
    });
    if (!r.ok) {
      const err = await r.text();
      document.getElementById("probeOut").textContent = `HTTP ${r.status}: ${err.slice(0, 400)}\n(Engine may be skipped during training — AVA_SKIP_ENGINE_BOOT=1)`;
      return;
    }
    const j = await r.json();
    const spaces = j.spaces || j.per_space || {};
    const route = j.route_probs || j.routing || {};
    const mass = j.verbalizable_mass ?? j.mass;
    document.getElementById("probeOut").textContent = JSON.stringify({
      verbalizable_mass: mass,
      route_probs: route,
      spaces: Object.fromEntries(Object.entries(spaces).map(([k,v]) => [k, {
        mass: v.verbalizable_mass ?? v.mass,
        top: (v.top_concepts || v.top || []).slice?.(0,5) || v.top,
      }])),
    }, null, 2);
  } catch (e) {
    document.getElementById("probeOut").textContent = String(e);
  }
};

refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>
"""
