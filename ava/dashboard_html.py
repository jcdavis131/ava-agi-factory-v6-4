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
  grid-template-columns: 1.2fr 1fr;
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
.bar-row { display: grid; grid-template-columns: 7.5rem 1fr 3.5rem; gap: 0.5rem; align-items: center; margin: 0.35rem 0; font-size: 0.85rem; font-variant-numeric: tabular-nums; }
.bar-track { height: 10px; background: var(--bar); position: relative; }
.bar-fill { height: 100%; background: var(--accent); }
.phase-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 0.4rem; }
.phase {
  border: 1px solid var(--line);
  padding: 0.5rem;
  text-align: center;
  background: var(--bg);
}
.phase .p { font-size: 0.7rem; color: var(--muted); }
.phase .n { font-size: 0.95rem; font-variant-numeric: tabular-nums; font-weight: 600; }
svg.chart { width: 100%; height: 160px; display: block; background: var(--bg); border: 1px solid var(--line); }
.legend { display: flex; gap: 1rem; font-size: 0.75rem; color: var(--muted); margin-top: 0.4rem; }
.legend i { display: inline-block; width: 12px; height: 3px; vertical-align: middle; margin-right: 4px; }
.muted { color: var(--muted); font-size: 0.85rem; }
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
    · <a href="/health">/health</a>
    · <a href="/jspace/viewer">viewer</a>
    · <a href="/report">report</a>
  </div>
</header>
<main>
  <section class="card span2">
    <h2>Status</h2>
    <div class="row" id="topStats"></div>
  </section>

  <section class="card">
    <h2>Shard lifecycle</h2>
    <div id="shardBars"></div>
  </section>

  <section class="card">
    <h2>Packed tokens ready by phase</h2>
    <div class="phase-grid" id="phases"></div>
    <p class="muted" style="margin:0.75rem 0 0">Trainer needs phase-current runway. Zero here with an active phase ⇒ DATA_STARVED.</p>
  </section>

  <section class="card span2">
    <h2>Training curve</h2>
    <svg class="chart" id="lossChart" viewBox="0 0 800 160" preserveAspectRatio="none"></svg>
    <div class="legend">
      <span><i style="background:#1e4d6b"></i>lm_loss</span>
      <span><i style="background:#2f6b3a"></i>tok/s (scaled)</span>
    </div>
    <p class="muted" id="trainCaption">Source: /reports/metrics_{preset}.jsonl</p>
  </section>

  <section class="card">
    <h2>Checkpoints</h2>
    <table>
      <thead><tr><th>File</th><th>MB</th></tr></thead>
      <tbody id="ckpts"></tbody>
    </table>
  </section>

  <section class="card">
    <h2>Live inspect</h2>
    <p class="muted">Forward pass on the hot-reloaded checkpoint. Mass / route should change with input.</p>
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
  if (Math.abs(n) >= 1e6) return (n/1e6).toFixed(2) + "M";
  if (Math.abs(n) >= 1e3) return (n/1e3).toFixed(1) + "k";
  return String(n);
}
function fmtTs(t) {
  try { return new Date(t * 1000).toLocaleTimeString(); } catch { return "—"; }
}

function renderTop(d) {
  const m = d.manifest || {};
  const tr = d.trainer || {};
  const last = tr.last || {};
  const starved = !!tr.data_starved;
  const health = starved ? "bad" : (m.ok ? "ok" : "warn");
  const healthLabel = starved ? "DATA_STARVED" : (m.ok ? "running" : "manifest?");
  const step = last.step != null ? last.step : "—";
  const loss = last.lm_loss != null ? Number(last.lm_loss).toFixed(3) : "—";
  const toks = last.tok_s != null ? Math.round(last.tok_s) : "—";
  const phase = last.phase != null ? last.phase : "—";
  document.getElementById("preset").textContent = d.preset || "—";
  document.getElementById("clock").textContent = fmtTs(d.ts);
  document.getElementById("topStats").innerHTML = `
    <div class="stat"><div class="k">Loop</div><div class="v sm"><span class="pill ${health}">${healthLabel}</span></div></div>
    <div class="stat"><div class="k">Shards</div><div class="v">${fmt(m.total_shards)}</div></div>
    <div class="stat"><div class="k">Raw on disk</div><div class="v sm">${m.raw_gb != null ? m.raw_gb + " GB" : "—"}</div></div>
    <div class="stat"><div class="k">Free disk</div><div class="v sm">${d.disk_free_gb != null ? d.disk_free_gb + " GB" : "—"}</div></div>
    <div class="stat"><div class="k">Step</div><div class="v">${step}</div></div>
    <div class="stat"><div class="k">Phase</div><div class="v">${phase}</div></div>
    <div class="stat"><div class="k">lm loss</div><div class="v sm">${loss}</div></div>
    <div class="stat"><div class="k">tok/s</div><div class="v sm">${toks}</div></div>
    <div class="stat"><div class="k">ckpt</div><div class="v sm">${(d.ckpt && d.ckpt.latest_pointer) || "—"}</div></div>
    <div class="stat"><div class="k">tokenizer</div><div class="v sm">${m.tokenizer_sha || "—"}</div></div>
  `;
}

function renderShards(d) {
  const by = (d.manifest && d.manifest.by_state) || {};
  const max = Math.max(1, ...STATES.map(s => by[s] || 0));
  document.getElementById("shardBars").innerHTML = STATES.map(s => {
    const v = by[s] || 0;
    const pct = (100 * v / max).toFixed(1);
    return `<div class="bar-row"><div>${s}</div><div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div><div>${v}</div></div>`;
  }).join("");
}

function renderPhases(d) {
  const t = (d.manifest && d.manifest.tokens_ready_by_phase) || {};
  document.getElementById("phases").innerHTML = [0,1,2,3,4,5].map(p => {
    const n = t[String(p)] || 0;
    return `<div class="phase"><div class="p">P${p}</div><div class="n">${fmt(n)}</div></div>`;
  }).join("");
}

function poly(xs, ys, w, h, pad) {
  if (!xs.length) return "";
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ys.filter(v => v != null)), ymax = Math.max(...ys.filter(v => v != null));
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

function renderChart(d) {
  const s = (d.trainer && d.trainer.series) || {};
  const steps = (s.step || []).filter((v,i) => s.lm_loss && s.lm_loss[i] != null);
  const losses = (s.lm_loss || []).filter(v => v != null);
  const toks = [];
  const stepTok = [];
  for (let i = 0; i < (s.step||[]).length; i++) {
    if (s.tok_s && s.tok_s[i] != null && s.step[i] != null) {
      stepTok.push(s.step[i]);
      toks.push(s.tok_s[i]);
    }
  }
  const svg = document.getElementById("lossChart");
  const w = 800, h = 160, pad = 12;
  let html = "";
  if (steps.length >= 2 && losses.length >= 2) {
    html += `<polyline fill="none" stroke="#1e4d6b" stroke-width="2" points="${poly(steps, losses, w, h, pad)}"/>`;
  }
  if (stepTok.length >= 2) {
    // scale tok/s into loss-ish visual band for overlay
    const tmin = Math.min(...toks), tmax = Math.max(...toks);
    const lmin = losses.length ? Math.min(...losses) : 0;
    const lmax = losses.length ? Math.max(...losses) : 1;
    const scaled = toks.map(t => lmin + (lmax - lmin) * ((t - tmin) / Math.max(1e-9, tmax - tmin)));
    html += `<polyline fill="none" stroke="#2f6b3a" stroke-width="1.5" stroke-dasharray="4 3" points="${poly(stepTok, scaled, w, h, pad)}"/>`;
  }
  if (!html) {
    html = `<text x="20" y="80" fill="#5c5a52" font-size="14">No lm_loss points yet (waiting for trainer steps)</text>`;
  }
  svg.innerHTML = html;
  const n = d.trainer && d.trainer.n_points;
  document.getElementById("trainCaption").textContent =
    `Source: ${(d.trainer && d.trainer.metrics_path) || "metrics"} · ${n || 0} recent lines · loss solid, tok/s dashed (normalized)`;
}

function renderCkpts(d) {
  const files = (d.ckpt && d.ckpt.files) || [];
  document.getElementById("ckpts").innerHTML = files.length
    ? files.map(f => `<tr><td>${f.name}</td><td>${f.mb}</td></tr>`).join("")
    : `<tr><td colspan="2" class="muted">No .pt files</td></tr>`;
}

async function refresh() {
  try {
    const r = await fetch("/pipeline/status");
    const d = await r.json();
    lastPayload = d;
    renderTop(d);
    renderShards(d);
    renderPhases(d);
    renderChart(d);
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
