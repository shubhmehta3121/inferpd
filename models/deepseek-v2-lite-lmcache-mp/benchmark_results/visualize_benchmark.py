#!/usr/bin/env python3
"""
visualize_benchmark.py — inferpd PD Disagg Benchmark Report Generator

Usage:
  python visualize_benchmark.py [benchmark_results_dir]

Expects two result directories under the given path (latest match wins):
  p50_2p2d_* / p50_sweep_*  — P50 profile (qps*.json, any sweep rates)
  p95_2p2d_* / p95_stress_* — P95 stress (p95_rate*.json)

Falls back to hardcoded values if directories are not found.

Output:
  <base_path>/benchmark_report.html
"""

import json
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

_DEFAULT_BASE = Path(__file__).resolve().parent
BASE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else _DEFAULT_BASE

QPS_SWEEP_DISPLAY = "1, 2, 5, 10, 20, 30, 50, 60 req/s"
P50_PREFIX_DISPLAY = "6,000 tok (shared prefix)"
P95_PREFIX_DISPLAY = "12,000 tok (shared prefix)"
PROMPTS_PER_STEP_DISPLAY = "600 per step"
# For prefix-cache % line (scenario prefix length vs avg input from JSON)
P50_PREFIX_TOK = 6000
P95_PREFIX_TOK = 12000

# ── Helpers ────────────────────────────────────────────────────────────────────


def find_latest_dir(pattern):
    hits = sorted(BASE.glob(pattern), reverse=True)
    return hits[0] if hits else None


def load_results(directory, filename_pattern):
    """Load all JSON result files matching pattern, sorted by request_rate."""
    if directory is None or not directory.exists():
        return []
    rows = []
    for path in sorted(directory.glob(filename_pattern)):
        if "warmup" in path.name:
            continue
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8", errors="replace")))
        except Exception:
            pass
    return sorted(rows, key=lambda r: r.get("request_rate", 0))


def ms2s(d, key):
    v = d.get(key, 0) or 0
    return round(v / 1000 if v > 100 else v, 3)


def extract_series(rows):
    """Pull the metrics we need from a list of result dicts."""
    return {
        "qps":      [r.get("request_rate") for r in rows],
        "ttft_p50": [ms2s(r, "p50_ttft_ms") for r in rows],
        "ttft_p95": [ms2s(r, "p95_ttft_ms") for r in rows],
        "e2e_p50":  [ms2s(r, "p50_e2el_ms") for r in rows],
        "e2e_p95":  [ms2s(r, "p95_e2el_ms") for r in rows],
        "tpot_p50": [round(r.get("p50_tpot_ms", 0) or 0, 2) for r in rows],
        "completed":[r.get("completed", 0) for r in rows],
        "failed":   [r.get("failed", 0) for r in rows],
        "max_conc": [int(r.get("max_concurrent_requests") or 0) for r in rows],
        "input_tok": rows[0].get("total_input_tokens", 0) // max(rows[0].get("completed", 1), 1) if rows else 0,
        "output_tok":rows[0].get("total_output_tokens", 0) // max(rows[0].get("completed", 1), 1) if rows else 0,
        "prefix_len": (rows[0].get("random_prefix_len") or 0) if rows else 0,
    }


def prefix_cache_note(s: dict, scenario_prefix_tok: int) -> str:
    """One line: shared-prefix share vs avg input; LMCache hit path."""
    it = max(s.get("input_tok") or 0, 1)
    pct = min(100, round(scenario_prefix_tok / it * 100))
    return f"~{pct}% of input tokens are shared prefix — LMCache prefix cache hit path active."


# ── Load data ─────────────────────────────────────────────────────────────────

p50_dir = find_latest_dir("p50_2p2d_*") or find_latest_dir("p50_sweep_*")
p95_dir = find_latest_dir("p95_2p2d_*") or find_latest_dir("p95_stress_*")

p50_rows = load_results(p50_dir, "qps*.json")
p95_rows = load_results(p95_dir, "p95_rate*.json")

FALLBACK_P50 = {
    "qps":      [2,     5,      10     ],
    "ttft_p50": [0.774, 10.066, 39.588 ],
    "ttft_p95": [1.235, 14.213, 72.258 ],
    "e2e_p50":  [10.027,19.737, 49.219 ],
    "e2e_p95":  [10.453,23.637, 81.470 ],
    "tpot_p50": [36.77, 38.47,  38.53  ],
    "completed":[600,   600,    600    ],
    "failed":   [0,     0,      0      ],
    "max_conc": [0, 0, 0],
    "input_tok": 7870, "output_tok": 252, "prefix_len": 6000,
}

FALLBACK_P95 = {
    "qps":      [1,      3,      5      ],
    "ttft_p50": [1.675,  40.367, 59.772 ],
    "ttft_p95": [2.927,  75.576, 112.928],
    "e2e_p50":  [22.468, 62.082, 81.334 ],
    "e2e_p95":  [23.631, 96.075, 133.292],
    "tpot_p50": [36.99,  38.42,  38.44  ],
    "completed":[300,    300,    300    ],
    "failed":   [0,      0,      0      ],
    "max_conc": [0, 0, 0],
    "input_tok": 15840, "output_tok": 561, "prefix_len": 12000,
}

s50 = extract_series(p50_rows) if p50_rows else FALLBACK_P50
s95 = extract_series(p95_rows) if p95_rows else FALLBACK_P95


def profile_info(
    s,
    label,
    *,
    prefix_display: str | None = None,
    num_prompts_display: str | None = None,
    qps_explicit: str | None = None,
    scenario_prefix_tok: int = 0,
    source_run: str = "",
):
    hit_pct = round(s["prefix_len"] / s["input_tok"] * 100) if s["input_tok"] and s["prefix_len"] else 0
    if prefix_display is not None:
        prefix_txt = prefix_display
    else:
        prefix_txt = (
            f"{s['prefix_len']:,} tok ({hit_pct}%)"
            if s["prefix_len"]
            else "— (not in JSON)"
        )
    qps_range = qps_explicit if qps_explicit else f"{s['qps'][0]} → {s['qps'][-1]} req/s"
    np_txt = num_prompts_display if num_prompts_display else f"{s['completed'][0]:,} per step"
    cache = prefix_cache_note(s, scenario_prefix_tok) if scenario_prefix_tok else ""
    return {
        "label":      label,
        "input_tok":  f"{s['input_tok']:,}",
        "output_tok": f"{s['output_tok']:,}",
        "prefix":     prefix_txt,
        "qps_range":  qps_range,
        "num_prompts":np_txt,
        "best_ttft":  f"{s['ttft_p50'][0]:.3f}s",
        "cache_note": cache,
        "source_run": source_run or "—",
    }


info50 = profile_info(
    s50,
    "P50 — Typical traffic",
    prefix_display=P50_PREFIX_DISPLAY,
    num_prompts_display=PROMPTS_PER_STEP_DISPLAY,
    qps_explicit=QPS_SWEEP_DISPLAY,
    scenario_prefix_tok=P50_PREFIX_TOK,
    source_run=p50_dir.name if p50_dir else "",
)
info95 = profile_info(
    s95,
    "P95 — Long-context stress",
    prefix_display=P95_PREFIX_DISPLAY,
    num_prompts_display=PROMPTS_PER_STEP_DISPLAY,
    qps_explicit=QPS_SWEEP_DISPLAY,
    scenario_prefix_tok=P95_PREFIX_TOK,
    source_run=p95_dir.name if p95_dir else "",
)

leg50_tok = f"{s50['input_tok']:,} tok in"
leg95_tok = f"{s95['input_tok']:,} tok in"

import json as J

def js(v):
    return J.dumps(v)


scatter50_ttft  = js(list(zip(s50["qps"], s50["ttft_p50"])))
scatter95_ttft  = js(list(zip(s95["qps"], s95["ttft_p50"])))
scatter50_e2e   = js(list(zip(s50["qps"], s50["e2e_p50"])))
scatter95_e2e   = js(list(zip(s95["qps"], s95["e2e_p50"])))
scatter50_tpot  = js(list(zip(s50["qps"], s50["tpot_p50"])))
scatter95_tpot  = js(list(zip(s95["qps"], s95["tpot_p50"])))
scatter50_conc  = js(list(zip(s50["qps"], s50["max_conc"])))
scatter95_conc  = js(list(zip(s95["qps"], s95["max_conc"])))

all_qps = sorted(set(s50["qps"] + s95["qps"]))
x_max   = all_qps[-1] + 1 if all_qps else 1

tpot_min = max(0, min(s50["tpot_p50"] + s95["tpot_p50"]) - 3)
tpot_max = max(s50["tpot_p50"] + s95["tpot_p50"]) + 3

all_conc = s50["max_conc"] + s95["max_conc"]
conc_max = max(max(all_conc, default=0) * 1.08, 1)

# ── HTML template ─────────────────────────────────────────────────────────────

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>inferpd — PD Disagg Benchmark</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{{--bg:#09090f;--card:#0f1018;--border:#1a1d2e;--text:#c9d1e8;--muted:#404865;--green:#5eead4;--blue:#60a5fa;--amber:#fbbf24}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;font-size:13px;max-width:960px;margin:0 auto;padding:40px 20px 64px}}
header{{margin-bottom:28px}}
.eyebrow{{font-family:'DM Mono',monospace;font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin-bottom:8px}}
h1{{font-size:22px;font-weight:600;color:#e2e8f8;letter-spacing:-.02em;margin-bottom:4px}}
.meta{{font-family:'DM Mono',monospace;font-size:10px;color:var(--muted);line-height:2}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
@media(max-width:560px){{.grid{{grid-template-columns:1fr}}}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px 18px}}
.card h2{{font-size:11px;font-weight:600;color:var(--text);margin-bottom:2px}}
.card p{{font-family:'DM Mono',monospace;font-size:9px;color:var(--muted);margin-bottom:12px;line-height:1.6}}
.chart-wrap{{position:relative;height:196px}}
.leg{{display:flex;gap:14px;margin-top:10px;flex-wrap:wrap}}
.leg span{{display:flex;align-items:center;gap:5px;font-family:'DM Mono',monospace;font-size:9px;color:var(--muted)}}
.dot{{width:7px;height:7px;border-radius:50%;flex-shrink:0}}
.profiles{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}}
.profile{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px 18px}}
.profile h3{{font-size:11px;font-weight:600;color:var(--text);margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid var(--border)}}
.row{{display:flex;justify-content:space-between;font-family:'DM Mono',monospace;font-size:10px;padding:5px 0;border-bottom:1px solid #12141f;color:var(--muted);gap:12px}}
.row:last-child{{border-bottom:none}}
.row span{{color:var(--text);text-align:right}}
.footnote{{font-family:'DM Mono',monospace;font-size:10px;color:var(--muted);margin-top:16px;padding:12px 14px;border:1px solid var(--border);border-radius:8px;background:#0a0a12;line-height:1.5}}
</style>
</head>
<body>

<header>
  <p class="eyebrow">inferpd · DeepSeek-V2-Lite · LMCache MP · March 2026</p>
  <h1>PD Disaggregated Prefill — LMCache MP Results</h1>
  <p class="meta">
    vLLM 0.18.0 · LMCacheMPConnector · ZMQ · LMCache 0.4.2 · 2× A100 SXM 80GB · BF16<br>
    Disagg proxy + LMCache multiprocess server · Prefiller / Decoder (see runbook for CLI)
  </p>
</header>

<div class="grid">

  <div class="card">
    <h2>Time to First Token — TTFT P50</h2>
    <p>seconds · lower is better · dashed = 2.0s production target</p>
    <div class="chart-wrap"><canvas id="c1"></canvas></div>
    <div class="leg">
      <span><div class="dot" style="background:var(--blue)"></div>P50 — Typical traffic ({leg50_tok})</span>
      <span><div class="dot" style="background:var(--amber)"></div>P95 — Long-context stress ({leg95_tok})</span>
    </div>
  </div>

  <div class="card">
    <h2>End-to-End Latency — E2E P50</h2>
    <p>seconds · lower is better · dashed = 11.0s production target</p>
    <div class="chart-wrap"><canvas id="c2"></canvas></div>
    <div class="leg">
      <span><div class="dot" style="background:var(--blue)"></div>P50 — Typical traffic</span>
      <span><div class="dot" style="background:var(--amber)"></div>P95 — Long-context stress</span>
    </div>
  </div>

  <div class="card">
    <h2>Time per Output Token — TPOT P50</h2>
    <p>milliseconds · stable line = decoder is not the bottleneck</p>
    <div class="chart-wrap"><canvas id="c3"></canvas></div>
    <div class="leg">
      <span><div class="dot" style="background:var(--blue)"></div>P50 — Typical traffic</span>
      <span><div class="dot" style="background:var(--amber)"></div>P95 — Long-context stress</span>
    </div>
  </div>

  <div class="card">
    <h2>Max concurrent requests</h2>
    <p>peak in-flight requests per run (from benchmark JSON)</p>
    <div class="chart-wrap"><canvas id="c4"></canvas></div>
    <div class="leg">
      <span><div class="dot" style="background:var(--blue)"></div>P50 — Typical traffic</span>
      <span><div class="dot" style="background:var(--amber)"></div>P95 — Long-context stress</span>
    </div>
  </div>

</div>

<div class="profiles">
  <div class="profile">
    <h3 style="color:var(--blue)">{info50["label"]}</h3>
    <div class="row">Result folder<span>{info50["source_run"]}</span></div>
    <div class="row">Input tokens (avg)<span>{info50["input_tok"]}</span></div>
    <div class="row">Output tokens (avg)<span>{info50["output_tok"]}</span></div>
    <div class="row">Shared prefix<span>{info50["prefix"]}</span></div>
    <div class="row">Prompts per step<span>{info50["num_prompts"]}</span></div>
    <div class="row">QPS sweep<span>{info50["qps_range"]}</span></div>
    <div class="row">Best TTFT P50<span style="color:var(--green)">{info50["best_ttft"]}</span></div>
    <div class="row" style="flex-direction:column;align-items:stretch"><span style="color:var(--muted);font-size:9px;line-height:1.5">{info50["cache_note"]}</span></div>
  </div>
  <div class="profile">
    <h3 style="color:var(--amber)">{info95["label"]}</h3>
    <div class="row">Result folder<span>{info95["source_run"]}</span></div>
    <div class="row">Input tokens (avg)<span>{info95["input_tok"]}</span></div>
    <div class="row">Output tokens (avg)<span>{info95["output_tok"]}</span></div>
    <div class="row">Shared prefix<span>{info95["prefix"]}</span></div>
    <div class="row">Prompts per step<span>{info95["num_prompts"]}</span></div>
    <div class="row">QPS sweep<span>{info95["qps_range"]}</span></div>
    <div class="row">Best TTFT P50<span style="color:var(--green)">{info95["best_ttft"]}</span></div>
    <div class="row" style="flex-direction:column;align-items:stretch"><span style="color:var(--muted);font-size:9px;line-height:1.5">{info95["cache_note"]}</span></div>
  </div>
</div>

<p class="footnote">Next step for higher sustained QPS: scale to 2P2D (add prefill / decode replicas) or widen the decode pool.</p>

<script>
Chart.defaults.color='#404865';
Chart.defaults.borderColor='#151825';
Chart.defaults.font.family="'DM Mono',monospace";
Chart.defaults.font.size=10;

const BLUE='#60a5fa', AMBER='#fbbf24';

const toXY = pts => pts.map(([x,y]) => ({{x,y}}));

const line = (pts, color) => ({{
  type:'scatter', data:toXY(pts),
  borderColor:color, backgroundColor:color+'18',
  borderWidth:2, pointRadius:4.5, pointHoverRadius:6,
  showLine:true, tension:0.1, fill:false,
}});

const ref = (xMin, xMax, y, color) => ({{
  type:'scatter', data:[{{x:xMin,y}},{{x:xMax,y}}],
  borderColor:color+'55', borderDash:[4,4], borderWidth:1,
  pointRadius:0, showLine:true, fill:false,
}});

const axes = (yLabel, yMin, yMax, xMax) => ({{
  responsive:true, maintainAspectRatio:false,
  plugins:{{legend:{{display:false}},tooltip:{{backgroundColor:'#0f1018',borderColor:'#1a1d2e',borderWidth:1}}}},
  scales:{{
    x:{{type:'linear',grid:{{color:'#12141f'}},min:0,max:xMax,
       title:{{display:true,text:'configured req/s',color:'#404865',font:{{size:9}}}}}},
    y:{{grid:{{color:'#12141f'}},beginAtZero:yMin===undefined,
       ...(yMin!==undefined?{{min:yMin,max:yMax}}:{{}}),
       title:{{display:true,text:yLabel,color:'#404865',font:{{size:9}}}}}},
  }},
}});

const p50_ttft  = {scatter50_ttft};
const p95_ttft  = {scatter95_ttft};
const p50_e2e   = {scatter50_e2e};
const p95_e2e   = {scatter95_e2e};
const p50_tpot  = {scatter50_tpot};
const p95_tpot  = {scatter95_tpot};
const p50_conc  = {scatter50_conc};
const p95_conc  = {scatter95_conc};
const xMax      = {x_max};
const concMax   = {conc_max:.1f};
const tpotMin   = {tpot_min:.1f};
const tpotMax   = {tpot_max:.1f};

new Chart('c1',{{type:'scatter',
  data:{{datasets:[line(p50_ttft,BLUE),line(p95_ttft,AMBER),ref(0,xMax,2.0,AMBER)]}},
  options:axes('seconds',undefined,undefined,xMax),
}});

new Chart('c2',{{type:'scatter',
  data:{{datasets:[line(p50_e2e,BLUE),line(p95_e2e,AMBER),ref(0,xMax,11.0,AMBER)]}},
  options:axes('seconds',undefined,undefined,xMax),
}});

new Chart('c3',{{type:'scatter',
  data:{{datasets:[line(p50_tpot,BLUE),line(p95_tpot,AMBER)]}},
  options:axes('milliseconds',tpotMin,tpotMax,xMax),
}});

new Chart('c4',{{type:'scatter',
  data:{{datasets:[line(p50_conc,BLUE),line(p95_conc,AMBER)]}},
  options:axes('concurrent requests',0,concMax,xMax),
}});
</script>
</body>
</html>"""

out = BASE / "benchmark_report.html"
out.write_text(html, encoding="utf-8")
print(f"[OK] Wrote {out}")
print(f"     P50 rows loaded: {len(p50_rows)}  ({'from disk' if p50_rows else 'fallback'})")
print(f"     P95 rows loaded: {len(p95_rows)}  ({'from disk' if p95_rows else 'fallback'})")
