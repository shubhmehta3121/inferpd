#!/bin/bash
# ============================================================
# run_benchmark_p95.sh
# P95 Long-Context Stress Test
#
# workload production profile (P95):
#   Input:        15840 tokens
#   Output:       561   tokens
#   Prefix cache: 80% hit rate → simulated via --random-prefix-len 12000
#   Rate:         1 → 2 → 3 → 5 req/s
#                 (long prompts are compute-heavy on the prefiller;
#                  PD disagg should keep TTFT low even here)
#
# What this tests:
#   At P95 input length, a non-disaggregated system would have
#   enormous TTFT because prefill of 15840 tokens blocks decode.
#   With PD disaggregation, the prefiller processes KV and the
#   decoder starts generating immediately from cached KV.
#   This is where you should see the biggest TTFT improvement.
#
#   With 12000-token shared prefix, after warmup the prefiller
#   only needs to compute KV for the remaining ~3840 unique tokens.
#   Prefix cache hit = massive TTFT reduction.
#
# Key fixes vs old file:
#   - Removed --no-stream  (streaming = correct TTFT measurement)
#   - Added --random-prefix-len 12000 (80% prefix hit simulation)
#   - Warmup phase before measuring
#   - Num prompts scaled: 150 per rate step (good P95 sample size)
# ============================================================

set -euo pipefail
source "$(dirname "$0")/../.env"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="/workspace/benchmark_results/p95_stress_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# ------------------------------------------------------------
# Shared prefix for P95 profile.
# Total P95 input = 15840 tokens. 12000 = 75.8% of input.
# Remainder = 3840 unique tokens per request.
# After warmup, prefiller only computes new KV for 3840 tokens.
# ------------------------------------------------------------
PREFIX_LEN=12000

COMMON_ARGS=(
  --backend vllm
  --host localhost
  --port "$PROXY_PORT"
  --model "$MODEL_NAME"
  --dataset-name random
  --temperature 0
  --random-input-len 15840
  --random-output-len 561
  --random-prefix-len "$PREFIX_LEN"
  --seed 1132
  --percentile-metrics ttft,tpot,itl,e2el
  --metric-percentiles 50,90,95,99
  --save-result
)

echo "============================================================"
echo " BENCHMARK: P95 Long-Context Stress Test"
echo " Target:      http://localhost:$PROXY_PORT"
echo " Input len:   15840 tokens (workload P95)"
echo " Output len:  561   tokens (workload P95)"
echo " Prefix len:  $PREFIX_LEN tokens (simulates 80% cache hit)"
echo " Unique len:  $((15840 - PREFIX_LEN)) tokens per request"
echo " Rate steps:  1 → 3 → 5 req/s"
echo " Per step:    200 prompts"
echo " Results:     $RESULTS_DIR"
echo ""
echo " Note: This is your biggest test of PD disagg value."
echo "   Without PD: prefill of 15840 tokens blocks the decoder."
echo "   With PD + prefix cache: decoder gets KV immediately;"
echo "   prefiller only recomputes the ~3840 unique tokens."
echo "============================================================"

# ---- PHASE 1: WARMUP ----------------------------------------
# The warmup is especially important for P95 because the KV for
# 12000 prefix tokens is large. Prime it before measuring.
# 50 prompts at 1 req/s = 50 seconds — slow, but necessary since
# each prompt is expensive (15840 tokens) and we need the cache
# to actually be populated on the prefiller GPU before we test.
# ------------------------------------------------------------
echo ""
echo "[ WARMUP ] Priming prefix cache for long-context prompts..."
echo "  50 prompts @ 1 req/s. This takes ~50s — do not skip."
echo "  The 12000-token prefix KV must be cached before measuring."

vllm bench serve \
  "${COMMON_ARGS[@]}" \
  --num-prompts 50 \
  --request-rate 1 \
  --result-dir "$RESULTS_DIR" \
  --result-filename "warmup_discard.json" \
  2>&1 | tee "$RESULTS_DIR/warmup.log" | grep -E "(TTFT|Throughput|Finished|Error)" || true

rm -f "$RESULTS_DIR/warmup_discard.json"

echo ""
echo "[ WARMUP DONE ] Waiting 30 for system to settle..."
sleep 30

# ---- PHASE 2: RATE SWEEP ------------------------------------
echo ""
echo "[ RATE SWEEP ] Starting P95 stress test..."

set +e  # Don't exit on metrics extraction failure — always run all rate steps
for RATE in 1 3 5; do
  DURATION=$((200 / RATE))
  echo ""
  echo "──────────────────────────────────────────────────────"
  echo " Rate: $RATE req/s | Prompts: 200 | Duration: ~${DURATION}s"
  echo "──────────────────────────────────────────────────────"

  RESULT_FILE="p95_rate${RATE}_${TIMESTAMP}.json"

  vllm bench serve \
    "${COMMON_ARGS[@]}" \
    --num-prompts 200 \
    --request-rate "$RATE" \
    --result-dir "$RESULTS_DIR" \
    --result-filename "$RESULT_FILE" \
    2>&1 | tee -a "$RESULTS_DIR/p95_stress.log"

  # ---- Extract metrics ----
  RESULT_PATH="$RESULTS_DIR/$RESULT_FILE"

  python3 -c "
import json, sys
try:
    d = json.load(open('$RESULT_PATH'))
    def v(v): return (v or 0) / 1000 if (v or 0) > 100 else (v or 0)
    def get(k, alt): return v(d.get(k) or d.get(alt) or 0)
    t50 = get('p50_ttft_ms', 'ttft_percentile_50')
    t95 = get('p95_ttft_ms', 'ttft_percentile_95')
    t99 = get('p99_ttft_ms', 'ttft_percentile_99')
    e50 = get('p50_e2el_ms', 'e2el_percentile_50')
    e95 = get('p95_e2el_ms', 'e2el_percentile_95')
    itl = get('p50_itl_ms', 'itl_percentile_50')
    tput = d.get('output_throughput', 0) or 0

    targets = {'ttft_p50': 2.0, 'ttft_p95': 3.5, 'e2e_p50': 11.0, 'e2e_p95': 25.0}
    goals   = {'ttft_p50': 0.5, 'ttft_p95': 1.5}

    def tag(val, key):
        t, g = targets.get(key), goals.get(key)
        if g and val <= g: return '🎯 GOAL'
        if t and val <= t: return '✅ PASS'
        if t and val > t:  return f'❌ ({val/t:.1f}x over)'
        return ''

    print()
    print(f'  Results @ $RATE req/s (P95 profile, 15840 tokens):')
    print(f'    TTFT P50:   {t50:.3f}s  {tag(t50, \"ttft_p50\")}')
    print(f'    TTFT P95:   {t95:.3f}s  {tag(t95, \"ttft_p95\")}')
    print(f'    TTFT P99:   {t99:.3f}s')
    print(f'    E2E  P50:   {e50:.3f}s  {tag(e50, \"e2e_p50\")}')
    print(f'    E2E  P95:   {e95:.3f}s  {tag(e95, \"e2e_p95\")}')
    itl_str = f'{itl*1000:.1f}ms  (~{1/itl:.0f} tok/s decode)' if itl and itl > 0 else 'N/A'
    print(f'    ITL  P50:   {itl_str}')
    print(f'    Throughput: {tput:.0f} tok/s')
except Exception as e:
    print(f'  Error reading result: {e}')
sys.exit(0)
" 2>/dev/null || true

  if [ "$RATE" -lt 5 ]; then
    echo ""
    echo "  Cooling down 45s (long-context — GPU needs more recovery time)..."
    sleep 45
  fi
done
set -e  # Restore exit-on-error for summary phase

# ---- PHASE 3: SUMMARY TABLE ---------------------------------
echo ""
echo "============================================================"
echo " STRESS TEST SUMMARY — P95 Profile (15840 in / 561 out)"
echo " Prefix: ${PREFIX_LEN} tokens (80% cache hit simulation)"
echo "============================================================"
echo ""
python3 -c "
import json, glob, os

targets = {'ttft_p50': 2.0, 'ttft_p95': 3.5, 'e2e_p50': 11.0, 'e2e_p95': 25.0}
goals   = {'ttft_p50': 0.5, 'ttft_p95': 1.5}

files = sorted(glob.glob('$RESULTS_DIR/p95_rate*.json'))
if not files:
    print('  No results found.')
    exit()

print(f\"  {'Rate':>6} | {'TTFT P50':>10} | {'TTFT P95':>10} | {'E2E P50':>9} | {'E2E P95':>9} | {'tok/s':>7}\")
print(f\"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*9}-+-{'-'*9}-+-{'-'*7}\")

for path in files:
    try:
        d = json.load(open(path))
        rate = d.get('request_rate', '?')

        def get_s(key):
            v = d.get(key, 0) or 0
            return v / 1000 if v > 100 else v

        t50  = get_s('ttft_percentile_50')
        t95  = get_s('ttft_percentile_95')
        e50  = get_s('e2el_percentile_50')
        e95  = get_s('e2el_percentile_95')
        tput = d.get('output_throughput', 0) or 0

        def fmt(v, target=None, goal=None):
            s = f'{v:.3f}s'
            if goal and v <= goal:    return s + ' 🎯'
            if target and v <= target: return s + ' ✅'
            if target and v > target:  return s + ' ❌'
            return s

        print(f\"  {rate:>5}/s | {fmt(t50, targets['ttft_p50'], goals['ttft_p50']):>10} | {fmt(t95, targets['ttft_p95'], goals['ttft_p95']):>10} | {fmt(e50, targets['e2e_p50']):>9} | {fmt(e95, targets['e2e_p95']):>9} | {tput:>6.0f}t\")
    except Exception as e:
        print(f'  Error: {e}')
"

echo ""
echo "  Interpreting results:"
echo "  - TTFT should be dramatically lower than P95 with non-disagg setup"
echo "    because the decoder skips recomputing the 12000-token prefix KV."
echo "  - If TTFT is still high, check prefiller logs for KV transfer errors:"
echo "    grep 'nixl\|lmcache\|transfer' /workspace/logs/prefiller/*.log"
echo ""
echo "============================================================"
echo " Results saved to: $RESULTS_DIR"
echo " Full analysis:    python analyze_results.py $RESULTS_DIR"
echo "============================================================"
