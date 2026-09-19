#!/usr/bin/env python3
"""
visualize_benchmark.py — inferpd PD Disagg Benchmark Report Generator

Usage:
  python visualize_benchmark.py /workspace/benchmark_results

Expects two result directories under the given path:
  p50_2p2d_*   — P50 profile results (qps2, qps5, qps10 JSON files)
  p95_2p2d_*   — P95 profile results (p95_rate1, p95_rate3, p95_rate5 JSON files)

Falls back to hardcoded values if directories are not found.

Output:
  <base_path>/benchmark_report.html
"""

import json
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

BASE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(r"models\deepseek-v2-lite-2p2d\benchmark_results")

PROD_TARGETS = {
    "ttft_p50_s": 2.0,
    "ttft_p95_s": 3.5,
    "e2e_p50_s": 11.0,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

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
        # grab config info from first row
        "input_tok": rows[0].get("total_input_tokens", 0) // max(rows[0].get("completed", 1), 1) if rows else 0,
        "output_tok":rows[0].get("total_output_tokens", 0) // max(rows[0].get("completed", 1), 1) if rows else 0,
        "prefix_len":rows[0].get("random_prefix_len", 0) if rows else 0,
    }


# ── Load data ─────────────────────────────────────────────────────────────────

p50_dir = find_latest_dir("p50_2p2d_*") or find_latest_dir("p50_sweep_*")
p95_dir = find_latest_dir("p95_2p2d_*") or find_latest_dir("p95_stress_*")

p50_rows = load_results(p50_dir, "qps*.json")
p95_rows = load_results(p95_dir, "p95_rate*.json")

# ── Fallback to hardcoded values from the final confirmed run ─────────────────
# These match the uploaded JSON files exactly:
#   p50: qps2/qps5/qps10 from p50_2p2d_20260318_030940
#   p95: rate1/rate3/rate5 from p95_2p2d_20260318_024801

FALLBACK_P50 = {
    "qps":      [2,     5,      10     ],
    "ttft_p50": [0.774, 10.066, 39.588 ],
    "ttft_p95": [1.235, 14.213, 72.258 ],
    "e2e_p50":  [10.027,19.737, 49.219 ],
    "e2e_p95":  [10.453,23.637, 81.470 ],
    "tpot_p50": [36.77, 38.47,  38.53  ],
    "completed":[600,   600,    600    ],
    "failed":   [0,     0,      0      ],
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
    "input_tok": 15840, "output_tok": 561, "prefix_len": 12000,
}

s50 = extract_series(p50_rows) if p50_rows else FALLBACK_P50
s95 = extract_series(p95_rows) if p95_rows else FALLBACK_P95

# ── Derive profile info strings ───────────────────────────────────────────────

def profile_info(s, label):
    hit_pct = round(s["prefix_len"] / s["input_tok"] * 100) if s["input_tok"] else 0
    qps_range = f"{s['qps'][0]} → {s['qps'][-1]} req/s"
    ceiling = max(s["qps"])  # beyond which TTFT climbs sharply
    return {
        "label":      label,
        "input_tok":  f"{s['input_tok']:,}",
        "output_tok": f"{s['output_tok']:,}",
        "prefix":     f"{s['prefix_len']:,} tok ({hit_pct}%)",
        "qps_range":  qps_range,
        "num_prompts":f"{s['completed'][0]:,} per step",
        "best_ttft":  f"{s['ttft_p50'][0]:.3f}s",
        "ceiling":    f"~{max(r for r,c in zip(s['qps'], s['completed']) if c > 0 and s['failed'][s['qps'].index(r)] == 0)} req/s (0 failures)",
    }

info50 = profile_info(s50, "P50 — typical traffic")
info95 = profile_info(s95, "P95 — long-context stress")

# ── Build bar chart labels and data ───────────────────────────────────────────

bar_labels  = ([f"P50 {q}r/s" for q in s50["qps"]] +
               [f"P95 {q}r/s" for q in s95["qps"]])
bar_success = s50["completed"] + s95["completed"]
bar_failed  = s50["failed"]    + s95["failed"]
bar_colors  = (["'#60a5fa99'" ] * len(s50["qps"]) +
               ["'#fbbf2499'" ] * len(s95["qps"]))
bar_borders = (["'#60a5fa'"]   * len(s50["qps"]) +
               ["'#fbbf24'"]   * len(s95["qps"]))

# Convert lists to JS arrays for template interpolation
import json as J

def js(v): return J.dumps(v)

scatter50_ttft  = js(list(zip(s50["qps"], s50["ttft_p50"])))
scatter95_ttft  = js(list(zip(s95["qps"], s95["ttft_p50"])))
scatter50_e2e   = js(list(zip(s50["qps"], s50["e2e_p50"])))
scatter95_e2e   = js(list(zip(s95["qps"], s95["e2e_p50"])))
scatter50_tpot  = js(list(zip(s50["qps"], s50["tpot_p50"])))
scatter95_tpot  = js(list(zip(s95["qps"], s95["tpot_p50"])))

all_qps = sorted(set(s50["qps"] + s95["qps"]))
x_max   = all_qps[-1] + 1

tpot_min = max(0, min(s50["tpot_p50"] + s95["tpot_p50"]) - 3)
tpot_max = max(s50["tpot_p50"] + s95["tpot_p50"]) + 3

# ── HTML template ─────────────────────────────────────────────────────────────

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>inferpd — PD Disagg Benchmark</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{{--bg:#09090f;--card:#0f1018;--border:#1a1d2e;--text:#c9d1e8;--muted:#404865;--green:#5eead4;--blue:#60a5fa;--amber:#fbbf24;--red:#fb7185}}
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
.leg{{display:flex;gap:14px;margin-top:10px}}
.leg span{{display:flex;align-items:center;gap:5px;font-family:'DM Mono',monospace;font-size:9px;color:var(--muted)}}
.dot{{width:7px;height:7px;border-radius:50%;flex-shrink:0}}
.profiles{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}}
.profile{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px 18px}}
.profile h3{{font-size:11px;font-weight:600;color:var(--text);margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid var(--border)}}
.row{{display:flex;justify-content:space-between;font-family:'DM Mono',monospace;font-size:10px;padding:5px 0;border-bottom:1px solid #12141f;color:var(--muted)}}
.row:last-child{{border-bottom:none}}
.row span{{color:var(--text)}}
</style>
</head>
<body>

<header>
  <p class="eyebrow">inferpd · DeepSeek-V2-Lite · March 2026</p>
  <h1>PD Disaggregated Prefill — 2P + 2D Final Results</h1>
  <p class="meta">
    vLLM 0.16.0 · NixlConnector · 4× A100 SXM 80GB · BF16<br>
    Prefiller: --max-num-batched-tokens 32768 --max-model-len 32768 &nbsp;|&nbsp; Decoder: --max-model-len 32768
  </p>
</header>

<div class="grid">

  <div class="card">
    <h2>Time to First Token — TTFT P50</h2>
    <p>seconds · lower is better · dashed = 2.0s production target</p>
    <div class="chart-wrap"><canvas id="c1"></canvas></div>
    <div class="leg">
      <span><div class="dot" style="background:var(--blue)"></div>P50 profile (7870 tok in)</span>
      <span><div class="dot" style="background:var(--amber)"></div>P95 profile (15840 tok in)</span>
    </div>
  </div>

  <div class="card">
    <h2>End-to-End Latency — E2E P50</h2>
    <p>seconds · lower is better · dashed = 11.0s production target</p>
    <div class="chart-wrap"><canvas id="c2"></canvas></div>
    <div class="leg">
      <span><div class="dot" style="background:var(--blue)"></div>P50 profile</span>
      <span><div class="dot" style="background:var(--amber)"></div>P95 profile</span>
    </div>
  </div>

  <div class="card">
    <h2>Time per Output Token — TPOT P50</h2>
    <p>milliseconds · stable line = decoder is not the bottleneck</p>
    <div class="chart-wrap"><canvas id="c3"></canvas></div>
    <div class="leg">
      <span><div class="dot" style="background:var(--blue)"></div>P50 profile</span>
      <span><div class="dot" style="background:var(--amber)"></div>P95 profile</span>
    </div>
  </div>

  <div class="card">
    <h2>Successful vs Failed Requests</h2>
    <p>failures start when configured QPS exceeds system ceiling</p>
    <div class="chart-wrap"><canvas id="c4"></canvas></div>
    <div class="leg">
      <span><div class="dot" style="background:var(--blue)"></div>P50 success</span>
      <span><div class="dot" style="background:var(--amber)"></div>P95 success</span>
      <span><div class="dot" style="background:var(--red)"></div>failed</span>
    </div>
  </div>

</div>

<!-- Profile info -->
<div class="profiles">
  <div class="profile">
    <h3 style="color:var(--blue)">{info50["label"]}</h3>
    <div class="row">Input tokens<span>{info50["input_tok"]}</span></div>
    <div class="row">Output tokens<span>{info50["output_tok"]}</span></div>
    <div class="row">Shared prefix<span>{info50["prefix"]}</span></div>
    <div class="row">Prompts per step<span>{info50["num_prompts"]}</span></div>
    <div class="row">QPS sweep<span>{info50["qps_range"]}</span></div>
    <div class="row">Best TTFT P50<span style="color:var(--green)">{info50["best_ttft"]} 🎯</span></div>
    <div class="row">Zero-failure ceiling<span style="color:var(--green)">{info50["ceiling"]}</span></div>
  </div>
  <div class="profile">
    <h3 style="color:var(--amber)">{info95["label"]}</h3>
    <div class="row">Input tokens<span>{info95["input_tok"]}</span></div>
    <div class="row">Output tokens<span>{info95["output_tok"]}</span></div>
    <div class="row">Shared prefix<span>{info95["prefix"]}</span></div>
    <div class="row">Prompts per step<span>{info95["num_prompts"]}</span></div>
    <div class="row">QPS sweep<span>{info95["qps_range"]}</span></div>
    <div class="row">Best TTFT P50<span style="color:var(--green)">{info95["best_ttft"]} ✅</span></div>
    <div class="row">Zero-failure ceiling<span style="color:var(--green)">{info95["ceiling"]}</span></div>
  </div>
</div>

<script>
Chart.defaults.color='#404865';
Chart.defaults.borderColor='#151825';
Chart.defaults.font.family="'DM Mono',monospace";
Chart.defaults.font.size=10;

const BLUE='#60a5fa', AMBER='#fbbf24', RED='#fb7185';

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
const xMax      = {x_max};
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

new Chart('c4',{{type:'bar',
  data:{{
    labels:{js(bar_labels)},
    datasets:[
      {{label:'success',data:{js(bar_success)},
        backgroundColor:[{",".join(bar_colors)}],
        borderColor:[{",".join(bar_borders)}],
        borderWidth:1,borderRadius:3}},
      {{label:'failed',data:{js(bar_failed)},
        backgroundColor:RED+'99',borderColor:RED,borderWidth:1,borderRadius:3}},
    ],
  }},
  options:{{
    responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{display:false}},tooltip:{{backgroundColor:'#0f1018',borderColor:'#1a1d2e',borderWidth:1}}}},
    scales:{{
      x:{{grid:{{color:'#12141f'}},ticks:{{font:{{size:9}}}}}},
      y:{{grid:{{color:'#12141f'}},beginAtZero:true,title:{{display:true,text:'requests',color:'#404865',font:{{size:9}}}}}},
    }},
  }},
}});
</script>
</body>
</html>"""

# ── Write output ──────────────────────────────────────────────────────────────

out = BASE / "benchmark_report.html"
out.write_text(html, encoding="utf-8")
print(f"[OK] Wrote {out}")
print(f"     P50 rows loaded: {len(p50_rows)}  ({'from disk' if p50_rows else 'fallback'})")
print(f"     P95 rows loaded: {len(p95_rows)}  ({'from disk' if p95_rows else 'fallback'})")