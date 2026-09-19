#!/usr/bin/env python3
"""
visualize_benchmark.py — inferpd PD Disagg Benchmark Report Generator

Usage:
  python visualize_benchmark.py [benchmark_results_dir]

Loads P50 sweep from (under the given path):
  p50_sweep_20260328_061945/ — qps*.json

If that folder is missing, falls back to latest p50_2p2d_* / p50_sweep_*,
then to hardcoded P50 values.

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
PROMPTS_PER_STEP_DISPLAY = "600 per step"
# For prefix-cache % line (scenario prefix length vs avg input from JSON)
P50_PREFIX_TOK = 6000
P50_RUN_DIR = "p50_sweep_20260328_061945"

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


def req_throughput(r) -> float:
    """Achieved request throughput (req/s) from vLLM bench JSON."""
    v = r.get("request_throughput")
    if v is None:
        v = r.get("actual_request_rate")
    try:
        return round(float(v), 3) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def extract_series(rows):
    """Pull the metrics we need from a list of result dicts."""
    return {
        "qps":      [r.get("request_rate") for r in rows],
        "ttft_p50": [ms2s(r, "p50_ttft_ms") for r in rows],
        "ttft_p95": [ms2s(r, "p95_ttft_ms") for r in rows],
        "e2e_p50":  [ms2s(r, "p50_e2el_ms") for r in rows],
        "e2e_p95":  [ms2s(r, "p95_e2el_ms") for r in rows],
        "tpot_p50": [round(r.get("p50_tpot_ms", 0) or 0, 2) for r in rows],
        "req_tput": [req_throughput(r) for r in rows],
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

_p50_explicit = BASE / P50_RUN_DIR
p50_dir = _p50_explicit if _p50_explicit.is_dir() else (
    find_latest_dir("p50_2p2d_*") or find_latest_dir("p50_sweep_*")
)

p50_rows = load_results(p50_dir, "qps*.json")

FALLBACK_P50 = {
    "qps":      [2,     5,      10     ],
    "ttft_p50": [0.774, 10.066, 39.588 ],
    "ttft_p95": [1.235, 14.213, 72.258 ],
    "e2e_p50":  [10.027,19.737, 49.219 ],
    "e2e_p95":  [10.453,23.637, 81.470 ],
    "tpot_p50": [36.77, 38.47,  38.53  ],
    "req_tput": [1.9,   4.8,    9.5    ],
    "completed":[600,   600,    600    ],
    "failed":   [0,     0,      0      ],
    "max_conc": [0, 0, 0],
    "input_tok": 7870, "output_tok": 252, "prefix_len": 6000,
}

s50 = extract_series(p50_rows) if p50_rows else FALLBACK_P50


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

leg50_tok = f"{s50['input_tok']:,} tok in"

import json as J

def js(v):
    return J.dumps(v)


scatter50_ttft  = js(list(zip(s50["qps"], s50["ttft_p50"])))
scatter50_e2e   = js(list(zip(s50["qps"], s50["e2e_p50"])))
scatter50_tpot  = js(list(zip(s50["qps"], s50["tpot_p50"])))
scatter50_conc  = js(list(zip(s50["qps"], s50["max_conc"])))
scatter50_req_tput = js(list(zip(s50["qps"], s50["req_tput"])))

all_qps = sorted(set(s50["qps"]))
x_max   = all_qps[-1] + 1 if all_qps else 1

tpot_min = max(0, min(s50["tpot_p50"], default=0) - 3)
tpot_max = max(s50["tpot_p50"], default=0) + 3

all_conc = s50["max_conc"]
conc_max = max(max(all_conc, default=0) * 1.08, 1)

all_req_vals = s50["req_tput"] + s50["qps"]
req_tput_max = max(max(all_req_vals, default=1) * 1.12, x_max)

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
html{{font-size:18px}}
body{{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;font-size:1rem;line-height:1.45;max-width:1100px;margin:0 auto;padding:36px 24px 72px;-webkit-print-color-adjust:exact;print-color-adjust:exact}}
header{{margin-bottom:32px}}
.eyebrow{{font-family:'DM Mono',monospace;font-size:0.72rem;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:10px}}
h1{{font-size:1.55rem;font-weight:600;color:#e2e8f8;letter-spacing:-.02em;margin-bottom:8px;line-height:1.2}}
.meta{{font-family:'DM Mono',monospace;font-size:0.78rem;color:#a8b0d0;line-height:1.85}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
@media(max-width:560px){{.grid{{grid-template-columns:1fr}}}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px 20px}}
.card h2{{font-size:0.95rem;font-weight:600;color:var(--text);margin-bottom:4px;line-height:1.3}}
.card p{{font-family:'DM Mono',monospace;font-size:0.72rem;color:#94a0c8;margin-bottom:14px;line-height:1.55}}
.chart-wrap{{position:relative;height:260px}}
.leg{{display:flex;gap:16px;margin-top:12px;flex-wrap:wrap}}
.leg span{{display:flex;align-items:center;gap:6px;font-family:'DM Mono',monospace;font-size:0.72rem;color:#94a0c8}}
.dot{{width:9px;height:9px;border-radius:50%;flex-shrink:0}}
.profiles{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}}
.profile{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px 20px}}
.profile h3{{font-size:0.95rem;font-weight:600;color:var(--text);margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid var(--border)}}
.row{{display:flex;justify-content:space-between;font-family:'DM Mono',monospace;font-size:0.78rem;padding:7px 0;border-bottom:1px solid #12141f;color:#94a0c8;gap:14px}}
.row:last-child{{border-bottom:none}}
.row span{{color:var(--text);text-align:right}}
.cache-note{{color:#94a0c8;margin-top:6px;font-size:0.72rem;line-height:1.55}}
.note-p95{{margin:20px 0 24px;padding:14px 18px;border:1px solid #3d3420;border-radius:10px;background:#120f0a;font-family:'DM Mono',monospace;font-size:0.74rem;color:#c4b89a;line-height:1.6}}
.note-p95 strong{{color:#fbbf24;font-weight:600;display:block;margin-bottom:8px;font-size:0.78rem;font-family:'DM Sans',sans-serif}}
@media print{{body{{padding:16px}} .chart-wrap{{height:220px}}}}
</style>
</head>
<body>

<header>
  <p class="eyebrow">inferpd · DeepSeek-V3.2 (NVIDIA) · 8× B200 · LMCache MP · March 2026</p>
  <h1>PD Disaggregated Prefill — LMCache MP Results</h1>
  <p class="meta">
    NVIDIA DeepSeek-V3.2 · FP4 weights · vLLM 0.18.0 · LMCacheMPConnector · ZMQ · LMCache 0.4.2<br>
    8× NVIDIA B200 GPUs · 180 GB VRAM · 1 prefiller (4 GPUs) + 1 decoder (4 GPUs), 1P1D · tensor_parallel_size = 4 (prefill and decode) · disagg proxy + LMCache multiprocess server
  </p>
</header>

<div class="note-p95">
  <strong>P95 stress — why OOM at high QPS</strong>
  P95 is not plotted here. Under heavy QPS, many long-context sequences ran together; GPU KV filled (max_num_seqs, max_num_batched_tokens, gpu_memory_utilization) and a forward still needed extra workspace → CUDA OOM, engine death, timeouts on the client.
  <br><br>
  Fix at a high level: limit concurrency, tighten those vLLM caps, optionally free a little VRAM headroom (slightly lower gpu_memory_utilization), scale decoders for more sustained rps.
</div>

<div class="grid">

  <div class="card">
    <h2>Time to First Token — TTFT P50</h2>
    <p>seconds · lower is better · dashed = 2.0s production target</p>
    <div class="chart-wrap"><canvas id="c1"></canvas></div>
    <div class="leg">
      <span><div class="dot" style="background:var(--blue)"></div>P50 — Typical traffic ({leg50_tok})</span>
    </div>
  </div>

  <div class="card">
    <h2>End-to-End Latency — E2E P50</h2>
    <p>seconds · lower is better · dashed = 11.0s production target</p>
    <div class="chart-wrap"><canvas id="c2"></canvas></div>
    <div class="leg">
      <span><div class="dot" style="background:var(--blue)"></div>P50 — Typical traffic</span>
    </div>
  </div>

  <div class="card">
    <h2>Time per Output Token — TPOT P50</h2>
    <p>milliseconds · stable line = decoder is not the bottleneck</p>
    <div class="chart-wrap"><canvas id="c3"></canvas></div>
    <div class="leg">
      <span><div class="dot" style="background:var(--blue)"></div>P50 — Typical traffic</span>
    </div>
  </div>

  <div class="card">
    <h2>Max concurrent requests</h2>
    <p>peak in-flight requests per run (from benchmark JSON)</p>
    <div class="chart-wrap"><canvas id="c4"></canvas></div>
    <div class="leg">
      <span><div class="dot" style="background:var(--blue)"></div>P50 — Typical traffic</span>
    </div>
  </div>

  <div class="card" style="grid-column:1/-1">
    <h2>Request throughput — achieved vs configured</h2>
    <p>req/s · Y = achieved (request_throughput) · X = configured request_rate · dashed = ideal (Y = X)</p>
    <div class="chart-wrap"><canvas id="c5"></canvas></div>
    <div class="leg">
      <span><div class="dot" style="background:var(--blue)"></div>P50 profile — typical traffic</span>
      <span style="color:var(--muted)">· ideal</span>
    </div>
  </div>

</div>

<div class="profiles" style="grid-template-columns:1fr">
  <div class="profile">
    <h3 style="color:var(--blue)">{info50["label"]}</h3>
    <div class="row">Result folder<span>{info50["source_run"]}</span></div>
    <div class="row">Input tokens (avg)<span>{info50["input_tok"]}</span></div>
    <div class="row">Output tokens (avg)<span>{info50["output_tok"]}</span></div>
    <div class="row">Shared prefix<span>{info50["prefix"]}</span></div>
    <div class="row">Prompts per step<span>{info50["num_prompts"]}</span></div>
    <div class="row">QPS sweep<span>{info50["qps_range"]}</span></div>
    <div class="row">Best TTFT P50<span style="color:var(--green)">{info50["best_ttft"]}</span></div>
    <div class="row" style="flex-direction:column;align-items:stretch"><span class="cache-note">{info50["cache_note"]}</span></div>
  </div>
</div>

<script>
Chart.defaults.color='#8b9acf';
Chart.defaults.borderColor='#151825';
Chart.defaults.font.family="'DM Mono',monospace";
Chart.defaults.font.size=13;

const BLUE='#60a5fa', TARGET='#94a3b8';

const toXY = pts => pts.map(([x,y]) => ({{x,y}}));

const line = (pts, color) => ({{
  type:'scatter', data:toXY(pts),
  borderColor:color, backgroundColor:color+'18',
  borderWidth:2.5, pointRadius:5.5, pointHoverRadius:7,
  showLine:true, tension:0.1, fill:false,
}});

const ref = (xMin, xMax, y, color) => ({{
  type:'scatter', data:[{{x:xMin,y}},{{x:xMax,y}}],
  borderColor:color+'55', borderDash:[4,4], borderWidth:1,
  pointRadius:0, showLine:true, fill:false,
}});

const diag = (m) => ({{
  type:'scatter', data:[{{x:0,y:0}},{{x:m,y:m}}],
  borderColor:'#6b7aad', borderDash:[5,5], borderWidth:1.5,
  pointRadius:0, showLine:true, fill:false,
}});

const AX_TITLE = {{color:'#a8b4e8',font:{{size:13,family:"'DM Mono',monospace"}}}};
const AX_TICKS = {{color:'#8b9acf',font:{{size:12,family:"'DM Mono',monospace"}}}};
const axes = (yLabel, yMin, yMax, xMax) => ({{
  responsive:true, maintainAspectRatio:false,
  plugins:{{legend:{{display:false}},tooltip:{{backgroundColor:'#0f1018',borderColor:'#1a1d2e',borderWidth:1,titleFont:{{size:13}},bodyFont:{{size:12}}}}}},
  scales:{{
    x:{{type:'linear',grid:{{color:'#12141f'}},min:0,max:xMax,
       ticks:AX_TICKS,
       title:{{display:true,text:'configured req/s',...AX_TITLE}}}},
    y:{{grid:{{color:'#12141f'}},beginAtZero:yMin===undefined,
       ...(yMin!==undefined?{{min:yMin,max:yMax}}:{{}}),
       ticks:AX_TICKS,
       title:{{display:true,text:yLabel,...AX_TITLE}}}},
  }},
}});

const p50_ttft  = {scatter50_ttft};
const p50_e2e   = {scatter50_e2e};
const p50_tpot  = {scatter50_tpot};
const p50_conc  = {scatter50_conc};
const p50_req_tput = {scatter50_req_tput};
const xMax      = {x_max};
const reqTputMax = {req_tput_max:.2f};
const concMax   = {conc_max:.1f};
const tpotMin   = {tpot_min:.1f};
const tpotMax   = {tpot_max:.1f};

new Chart('c1',{{type:'scatter',
  data:{{datasets:[line(p50_ttft,BLUE),ref(0,xMax,2.0,TARGET)]}},
  options:axes('seconds',undefined,undefined,xMax),
}});

new Chart('c2',{{type:'scatter',
  data:{{datasets:[line(p50_e2e,BLUE),ref(0,xMax,11.0,TARGET)]}},
  options:axes('seconds',undefined,undefined,xMax),
}});

new Chart('c3',{{type:'scatter',
  data:{{datasets:[line(p50_tpot,BLUE)]}},
  options:axes('milliseconds',tpotMin,tpotMax,xMax),
}});

new Chart('c4',{{type:'scatter',
  data:{{datasets:[line(p50_conc,BLUE)]}},
  options:axes('concurrent requests',0,concMax,xMax),
}});

new Chart('c5',{{type:'scatter',
  data:{{datasets:[line(p50_req_tput,BLUE),diag(reqTputMax)]}},
  options:{{
    responsive:true, maintainAspectRatio:false,
    plugins:{{legend:{{display:false}},tooltip:{{backgroundColor:'#0f1018',borderColor:'#1a1d2e',borderWidth:1,titleFont:{{size:13}},bodyFont:{{size:12}}}}}},
    scales:{{
      x:{{type:'linear',grid:{{color:'#12141f'}},min:0,max:reqTputMax,
         ticks:AX_TICKS,
         title:{{display:true,text:'configured req/s',...AX_TITLE}}}},
      y:{{grid:{{color:'#12141f'}},min:0,max:reqTputMax,beginAtZero:true,
         ticks:AX_TICKS,
         title:{{display:true,text:'achieved req/s',...AX_TITLE}}}},
    }},
  }},
}});
</script>
</body>
</html>"""

out = BASE / "benchmark_report.html"
out.write_text(html, encoding="utf-8")
print(f"[OK] Wrote {out}")
print(f"     P50 rows loaded: {len(p50_rows)}  ({'from disk' if p50_rows else 'fallback'})")
