#!/bin/bash

set -euo pipefail
source "$(dirname "$0")/../.env"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="/workspace/benchmark_results/p50_sweep_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

PREFIX_LEN=6000

COMMON_ARGS=(
  --backend vllm
  --host localhost
  --port "$PROXY_PORT"
  --model "$MODEL_NAME"
  --dataset-name random
  --temperature 0
  --random-input-len 1870
  --random-output-len 252
  --random-prefix-len 6000
  --seed 2003
  --percentile-metrics ttft,tpot,itl,e2el
  --metric-percentiles 50,90,95,99
  --save-result
  --save-detailed
  --plot-timeline
)

echo "============================================================"
echo " BENCHMARK: P50 Traffic Profile + Peak QPS Sweep"
echo " Target:      http://localhost:$PROXY_PORT"
echo " Input len:   7870 tokens (workload P50)"
echo " Output len:  252  tokens (workload P50)"
echo " Prefix len:  $PREFIX_LEN tokens (simulates 80% cache hit)"
echo " Unique len:  $((7870 - PREFIX_LEN)) tokens per request"
# echo " QPS sweep:   10 → 20 → 30 → 40 → 50 → 60 req/s"
echo " QPS sweep:   1 → 2 → 5 → 10 → 20 → 30 → 50 → 60 req/s"
echo " Per step:    600 prompts"
echo " Results:     $RESULTS_DIR"
echo " Artifacts:   *.json (detailed per-request), *.timeline.html"
echo "============================================================"


echo ""
echo "[ WARMUP ] Priming prefix cache (100 prompts @ 5 req/s)..."
echo "  This populates KV cache for the $PREFIX_LEN-token shared prefix."
echo "  Warmup results are NOT saved — they're just cache priming."

vllm bench serve \
  "${COMMON_ARGS[@]}" \
  --num-prompts 100 \
  --request-rate 5 \
  --result-dir "$RESULTS_DIR" \
  --result-filename "warmup_discard.json" \
  2>&1 | tee "$RESULTS_DIR/warmup.log" | grep -E "(TTFT|Throughput|Finished|Error)" || true

# Discard warmup result (JSON + vLLM timeline HTML next to it)
rm -f "$RESULTS_DIR/warmup_discard.json" "$RESULTS_DIR/warmup_discard.timeline.html"

echo ""
echo "[ WARMUP DONE ] Waiting 30s for KV cache to settle..."
sleep 30

# ---- PHASE 2: QPS SWEEP -------------------------------------
echo ""
echo "[ QPS SWEEP ] Starting..."

set +e  # Don't exit on metrics extraction failure — always run all rate steps
# for RATE in 10 20 30 40 50 60; do
# for RATE in 5 10 15 20; do
for RATE in 1 2 5 10 20 30 50 60; do
  echo ""
  echo "──────────────────────────────────────────────────────"
  echo " QPS: $RATE req/s | Prompts: 600 | Duration: ~$((600 / RATE))s"
  echo "──────────────────────────────────────────────────────"

  RESULT_FILE="qps${RATE}_${TIMESTAMP}.json"

  vllm bench serve \
    "${COMMON_ARGS[@]}" \
    --num-prompts 600 \
    --request-rate "$RATE" \
    --result-dir "$RESULTS_DIR" \
    --result-filename "$RESULT_FILE" \
    2>&1 | tee -a "$RESULTS_DIR/sweep.log"

  # ---- Extract key metrics from result ----
  RESULT_PATH="$RESULTS_DIR/$RESULT_FILE"

  TTFT_P50=$(python3 -c "
import json, sys
try:
    d = json.load(open('$RESULT_PATH'))
    v = d.get('ttft_percentile_50', 0) or 0
    print(v / 1000 if v > 100 else v)
except: print(9999)
" 2>/dev/null)

  TTFT_P95=$(python3 -c "
import json, sys
try:
    d = json.load(open('$RESULT_PATH'))
    v = d.get('ttft_percentile_95', 0) or 0
    print(v / 1000 if v > 100 else v)
except: print(9999)
" 2>/dev/null)

  E2E_P50=$(python3 -c "
import json, sys
try:
    d = json.load(open('$RESULT_PATH'))
    v = d.get('e2el_percentile_50', 0) or 0
    print(v / 1000 if v > 100 else v)
except: print(9999)
" 2>/dev/null)

  THROUGHPUT=$(python3 -c "
import json, sys
try:
    d = json.load(open('$RESULT_PATH'))
    print(d.get('output_throughput', 0) or 0)
except: print(0)
" 2>/dev/null)

  echo ""
  echo "  Results @ ${RATE} QPS:"
  echo "    TTFT P50:   ${TTFT_P50}s  (target: <2.0s | goal: <0.5s)"
  echo "    TTFT P95:   ${TTFT_P95}s  (target: <3.5s | goal: <1.5s)"
  echo "    E2E  P50:   ${E2E_P50}s   (target: <11s)"
  echo "    Throughput: ${THROUGHPUT} tok/s"

  if [ "$RATE" -lt 60 ]; then
    echo ""
    echo "  Cooling down 30s before next rate step..."
    sleep 30
  fi
done
set -e  # Restore exit-on-error for summary phase

# ---- PHASE 3: SUMMARY TABLE ---------------------------------
echo ""
echo "============================================================"
echo " QPS SWEEP SUMMARY — P50 Profile (7870 in / 252 out)"
echo " Prefix: ${PREFIX_LEN} tokens (80% cache hit simulation)"
echo "============================================================"
echo ""
python3 -c "
import json, glob, os

targets = {'ttft_p50': 2.0, 'ttft_p95': 3.5, 'e2e_p50': 11.0}
goals   = {'ttft_p50': 0.5, 'ttft_p95': 1.5}

files = sorted(glob.glob('$RESULTS_DIR/qps*.json'))
if not files:
    print('  No results found.')
    exit()

print(f\"  {'QPS':>5} | {'TTFT P50':>10} | {'TTFT P95':>10} | {'E2E P50':>9} | {'tok/s':>7} | Status\")
print(f\"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*9}-+-{'-'*7}-+--------\")

for path in files:
    try:
        d = json.load(open(path))
        rate = d.get('request_rate', '?')

        def get_ms(key):
            v = d.get(key, 0) or 0
            return v / 1000 if v > 100 else v

        t50  = get_ms('ttft_percentile_50')
        t95  = get_ms('ttft_percentile_95')
        e50  = get_ms('e2el_percentile_50')
        tput = d.get('output_throughput', 0) or 0

        def fmt(v, target=None, goal=None):
            if goal and v <= goal:   return f'{v:.3f}s 🎯'
            if target and v <= target: return f'{v:.3f}s ✅'
            if target and v > target:  return f'{v:.3f}s ❌'
            return f'{v:.3f}s'

        status = '✅ PASS' if t95 < targets['ttft_p95'] else '❌ Over'
        print(f\"  {rate:>5} | {fmt(t50, targets['ttft_p50'], goals['ttft_p50']):>10} | {fmt(t95, targets['ttft_p95'], goals['ttft_p95']):>10} | {fmt(e50, targets['e2e_p50']):>9} | {tput:>6.0f}t | {status}\")
    except Exception as e:
        print(f'  Error reading {os.path.basename(path)}: {e}')
"

echo ""
echo "============================================================"
echo " Results saved to: $RESULTS_DIR"
echo " Per-request JSON: qps*.json (includes errors[], ttfts, itls when detailed)"
echo " Timeline HTML:    qps*.timeline.html (open in browser)"
echo " Full analysis:    python analyze_results.py $RESULTS_DIR"
echo "============================================================"
