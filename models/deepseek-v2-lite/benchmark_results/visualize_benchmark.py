#!/usr/bin/env python3
"""
visualize_benchmarks.py — inferpd PD Disagg benchmark charts
Usage: python visualize_benchmarks.py /path/to/benchmark_results
Output: benchmark_report.html
"""
import json, sys
from pathlib import Path

BASE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent

def find_dir(exact, glob_pat):
    hits = sorted(BASE.glob(exact), reverse=True)
    if hits: return hits[0]
    hits = sorted(BASE.glob(glob_pat), reverse=True)
    return hits[0] if hits else None

P50_DIR = find_dir("p50_sweep_20260315_113629", "p50_sweep_*")
P95_DIR = find_dir("p95_stress_20260315_120241", "p95_stress_*")

def load(d):
    if d is None or not d.exists(): return []
    rows = []
    for p in sorted(d.glob("*.json")):
        if "warmup" in p.name: continue
        try: rows.append(json.loads(p.read_text(encoding="utf-8", errors="replace")))
        except: pass
    return sorted(rows, key=lambda r: r.get("request_rate", 0))

def ms_to_s(d, key): return round(d.get(key, 0) / 1000, 3)
def get_ms(d, key):  return round(d.get(key, 0), 2)

p50 = load(P50_DIR)
p95 = load(P95_DIR)

if not p50 and not p95:
    print(f"[ERROR] No data found under {BASE}")
    sys.exit(1)

# Extract series
def series(rows):
    return {
        "x":         [r.get("request_rate") for r in rows],
        "ttft_p50":  [ms_to_s(r, "p50_ttft_ms") for r in rows],
        "ttft_p95":  [ms_to_s(r, "p95_ttft_ms") for r in rows],
        "tpot_p50":  [get_ms(r, "p50_tpot_ms")  for r in rows],
        "e2e_p50":   [ms_to_s(r, "p50_e2el_ms") for r in rows],
        "e2e_p95":   [ms_to_s(r, "p95_e2el_ms") for r in rows],
        "completed": [r.get("completed", 0)      for r in rows],
        "failed":    [r.get("failed", 0)          for r in rows],
        "input_tok": rows[0].get("total_input_tokens", 0) // max(rows[0].get("completed",1),1) if rows else 0,
        "output_tok":rows[0].get("total_output_tokens",0) // max(rows[0].get("completed",1),1) if rows else 0,
    }

s50 = series(p50)
s95 = series(p95)

import json as J
x50 = J.dumps(s50["x"])
x95 = J.dumps(s95["x"])

# All x values combined for shared axis ticks
all_x = sorted(set(s50["x"] + s95["x"]))
xall  = J.dumps(all_x)

# Pad shorter series with null for aligned x-axis
def pad(vals, xs, all_xs):
    lookup = dict(zip(xs, vals))
    return J.dumps([lookup.get(x) for x in all_xs])

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>inferpd PD Disagg — Benchmark Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=DM+Mono&display=swap" rel="stylesheet">
<style>
:root {{
  --bg:#0b0d12; --card:#111318; --border:#1d2130;
  --text:#dde3f0; --muted:#566080;
  --blue:#4f8ef7; --amber:#f5a623;
}}
* {{ box-sizing:border-box; margin:0; padding:0 }}
body {{ background:var(--bg); color:var(--text); font-family:'Space Grotesk',sans-serif;
       max-width:1000px; margin:0 auto; padding:36px 24px 56px }}

header {{ margin-bottom:36px }}
.eyebrow {{ font-family:'DM Mono',monospace; font-size:11px; letter-spacing:.12em;
            text-transform:uppercase; color:var(--muted); margin-bottom:10px }}
h1 {{ font-size:26px; font-weight:700; line-height:1.2; margin-bottom:5px }}
h1 span {{ color:var(--blue) }}
.sub {{ font-family:'DM Mono',monospace; font-size:11px; color:var(--muted) }}

.grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:16px }}
.grid-3 {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; margin-bottom:16px }}

.panel {{ background:var(--card); border:1px solid var(--border); border-radius:10px;
          padding:18px 20px 14px }}
.panel h2 {{ font-size:12px; font-weight:600; margin-bottom:3px }}
.panel .psub {{ font-family:'DM Mono',monospace; font-size:10px; color:var(--muted);
                margin-bottom:14px; line-height:1.5 }}
.chart-wrap {{ position:relative; height:200px }}

/* legend */
.leg {{ display:flex; gap:16px; margin-top:10px }}
.leg-item {{ display:flex; align-items:center; gap:6px;
             font-family:'DM Mono',monospace; font-size:10px; color:var(--muted) }}
.leg-dot {{ width:8px; height:8px; border-radius:50%; flex-shrink:0 }}

/* info strip */
.info-strip {{ background:var(--card); border:1px solid var(--border); border-radius:10px;
               padding:16px 20px; margin-top:20px;
               display:grid; grid-template-columns:1fr 1fr; gap:16px }}
.info-col h3 {{ font-size:12px; font-weight:600; margin-bottom:8px }}
.info-row {{ display:flex; justify-content:space-between;
             font-family:'DM Mono',monospace; font-size:11px;
             padding:5px 0; border-bottom:1px solid var(--border); color:var(--muted) }}
.info-row:last-child {{ border-bottom:none }}
.info-row span {{ color:var(--text) }}
</style>
</head>
<body>

<header>
  <p class="eyebrow">inferpd · DeepSeek-V2-Lite · 2× A100 SXM 80GB · Phase 1 PoC</p>
  <h1>PD Disaggregated Prefill — <span>Benchmark Results</span></h1>
  <p class="sub">NixlConnector · vLLM 0.16.0 · BF16 · 1 prefiller (GPU 0) : 1 decoder (GPU 1) · Poisson arrivals · temp=0</p>
</header>

<!-- Row 1: TTFT + E2E -->
<div class="grid">
  <div class="panel">
    <h2>Time to First Token (TTFT P50)</h2>
    <p class="psub">P50 percentile · seconds · lower is better</p>
    <div class="chart-wrap"><canvas id="ttft"></canvas></div>
    <div class="leg">
      <div class="leg-item"><div class="leg-dot" style="background:var(--blue)"></div>P50 profile</div>
      <div class="leg-item"><div class="leg-dot" style="background:var(--amber)"></div>P95 profile</div>
    </div>
  </div>
  <div class="panel">
    <h2>End-to-End Latency (E2E P50)</h2>
    <p class="psub">P50 percentile · seconds · lower is better</p>
    <div class="chart-wrap"><canvas id="e2e"></canvas></div>
    <div class="leg">
      <div class="leg-item"><div class="leg-dot" style="background:var(--blue)"></div>P50 profile</div>
      <div class="leg-item"><div class="leg-dot" style="background:var(--amber)"></div>P95 profile</div>
    </div>
  </div>
</div>

<!-- Row 2: TPOT + Failed -->
<div class="grid">
  <div class="panel">
    <h2>Time per Output Token (TPOT P50)</h2>
    <p class="psub">P50 percentile · milliseconds · measures decode speed per GPU</p>
    <div class="chart-wrap"><canvas id="tpot"></canvas></div>
    <div class="leg">
      <div class="leg-item"><div class="leg-dot" style="background:var(--blue)"></div>P50 profile</div>
      <div class="leg-item"><div class="leg-dot" style="background:var(--amber)"></div>P95 profile</div>
    </div>
  </div>
  <div class="panel">
    <h2>Successful vs Failed Requests</h2>
    <p class="psub">Stacked · failures = proxy 120s timeout exceeded (queue overflow)</p>
    <div class="chart-wrap"><canvas id="failed"></canvas></div>
    <div class="leg">
      <div class="leg-item"><div class="leg-dot" style="background:var(--blue)"></div>P50 success</div>
      <div class="leg-item"><div class="leg-dot" style="background:var(--amber)"></div>P95 success</div>
      <div class="leg-item"><div class="leg-dot" style="background:#f04f59"></div>Failed</div>
    </div>
  </div>
</div>

<!-- Info strip -->
<div class="info-strip">
  <div class="info-col">
    <h3>P50 Profile — typical traffic</h3>
    <div class="info-row">Input tokens<span>7,870</span></div>
    <div class="info-row">Output tokens<span>252</span></div>
    <div class="info-row">Shared prefix<span>6,000 tok (76%)</span></div>
    <div class="info-row">Prompts per step<span>500</span></div>
    <div class="info-row">QPS steps tested<span>1, 3, 5</span></div>
    <div class="info-row">Prefiller capacity<span>~2 req/s</span></div>
  </div>
  <div class="info-col">
    <h3>P95 Profile — long-context stress</h3>
    <div class="info-row">Input tokens<span>15,840</span></div>
    <div class="info-row">Output tokens<span>561</span></div>
    <div class="info-row">Shared prefix<span>12,000 tok (76%)</span></div>
    <div class="info-row">Prompts per step<span>200</span></div>
    <div class="info-row">QPS steps tested<span>1, 3, 5</span></div>
    <div class="info-row">Prefiller capacity<span>~0.75 req/s</span></div>
  </div>
</div>

<script>
Chart.defaults.color = '#566080';
Chart.defaults.borderColor = '#1d2130';
Chart.defaults.font.family = "'DM Mono', monospace";
Chart.defaults.font.size = 11;

const BLUE  = '#4f8ef7';
const AMBER = '#f5a623';
const RED   = '#f04f59';

const baseOpts = {{
  responsive: true,
  maintainAspectRatio: false,
  plugins: {{
    legend: {{ display: false }},
    tooltip: {{
      backgroundColor: '#111318',
      borderColor: '#1d2130',
      borderWidth: 1,
    }},
  }},
  scales: {{
    x: {{
      grid: {{ color: '#181b24' }},
      title: {{ display: true, text: 'req/s (configured)', font: {{ size:10 }} }},
      ticks: {{ stepSize: 1 }},
    }},
    y: {{
      grid: {{ color: '#181b24' }},
      beginAtZero: true,
    }},
  }},
}};

const X = {xall};

// ── TTFT ─────────────────────────────────────────────────────────────────────
new Chart(document.getElementById('ttft'), {{
  type: 'line',
  data: {{
    labels: X,
    datasets: [
      {{ label:'P50 profile', data:{pad(s50["ttft_p50"], s50["x"], all_x)},
         borderColor:BLUE, backgroundColor:'rgba(79,142,247,.08)',
         borderWidth:2.5, pointRadius:5, tension:.15, fill:false }},
      {{ label:'P95 profile', data:{pad(s95["ttft_p50"], s95["x"], all_x)},
         borderColor:AMBER, backgroundColor:'rgba(245,166,35,.08)',
         borderWidth:2.5, pointRadius:5, tension:.15, fill:false }},
    ],
  }},
  options: {{ ...baseOpts,
    scales: {{ ...baseOpts.scales,
      y: {{ ...baseOpts.scales.y, title:{{ display:true, text:'seconds' }} }} }} }},
}});

// ── E2E ───────────────────────────────────────────────────────────────────────
new Chart(document.getElementById('e2e'), {{
  type: 'line',
  data: {{
    labels: X,
    datasets: [
      {{ label:'P50 profile', data:{pad(s50["e2e_p50"], s50["x"], all_x)},
         borderColor:BLUE, borderWidth:2.5, pointRadius:5, tension:.15, fill:false }},
      {{ label:'P95 profile', data:{pad(s95["e2e_p50"], s95["x"], all_x)},
         borderColor:AMBER, borderWidth:2.5, pointRadius:5, tension:.15, fill:false }},
    ],
  }},
  options: {{ ...baseOpts,
    scales: {{ ...baseOpts.scales,
      y: {{ ...baseOpts.scales.y, title:{{ display:true, text:'seconds' }} }} }} }},
}});

// ── TPOT ──────────────────────────────────────────────────────────────────────
new Chart(document.getElementById('tpot'), {{
  type: 'line',
  data: {{
    labels: X,
    datasets: [
      {{ label:'P50 profile', data:{pad(s50["tpot_p50"], s50["x"], all_x)},
         borderColor:BLUE, borderWidth:2.5, pointRadius:5, tension:.15, fill:false }},
      {{ label:'P95 profile', data:{pad([get_ms(r,"p50_tpot_ms") for r in p95], s95["x"], all_x)},
         borderColor:AMBER, borderWidth:2.5, pointRadius:5, tension:.15, fill:false }},
    ],
  }},
  options: {{ ...baseOpts,
    scales: {{ ...baseOpts.scales,
      y: {{ ...baseOpts.scales.y, title:{{ display:true, text:'milliseconds' }} }} }} }},
}});

// ── Failed stacked bar ────────────────────────────────────────────────────────
new Chart(document.getElementById('failed'), {{
  type: 'bar',
  data: {{
    labels: X,
    datasets: [
      {{ label:'P50 success', data:{pad(s50["completed"], s50["x"], all_x)},
         backgroundColor:'rgba(79,142,247,.65)', borderColor:BLUE, borderWidth:1, borderRadius:3, stack:'p50' }},
      {{ label:'P50 failed',  data:{pad(s50["failed"], s50["x"], all_x)},
         backgroundColor:'rgba(240,79,89,.75)',  borderColor:RED,  borderWidth:1, borderRadius:3, stack:'p50' }},
      {{ label:'P95 success', data:{pad(s95["completed"], s95["x"], all_x)},
         backgroundColor:'rgba(245,166,35,.65)', borderColor:AMBER,borderWidth:1, borderRadius:3, stack:'p95' }},
      {{ label:'P95 failed',  data:{pad(s95["failed"], s95["x"], all_x)},
         backgroundColor:'rgba(240,79,89,.75)',  borderColor:RED,  borderWidth:1, borderRadius:3, stack:'p95' }},
    ],
  }},
  options: {{ ...baseOpts,
    scales: {{ ...baseOpts.scales,
      y: {{ ...baseOpts.scales.y, title:{{ display:true, text:'requests' }} }} }} }},
}});
</script>
</body>
</html>"""

out = BASE / "benchmark_report.html"
out.write_text(html, encoding="utf-8")
print(f"[OK] Wrote {out}")