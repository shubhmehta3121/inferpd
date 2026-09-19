#!/bin/bash
# ============================================================
# run_benchmark_p95.sh — P95 Long-Context Stress Test  (2P + 2D FINAL)
#
# FINAL OPTIMAL CONFIG:
#   Prefiller: max_num_batched_tokens=32768  max_model_len=32768
#   Decoder:   max_model_len=32768  (default batched_tokens=2048)
#
# workload P95 production profile:
#   Input:  15840 tokens | Prefix: 12000 tokens | Output: 561 tokens
#   Unique tokens per request: ~8745 (44.8% benchmark hit rate)
#   Prefill batch at 32768: 32768/8745 = ~3.7 → 3 requests per step
#
# QPS sweep: 1 → 3 → 5 req/s  (long-context stress test)
#   At 1P+1D the ceiling was ~0.75 req/s.
#   With 2P+2D we expect ~1.5–2.0 req/s ceiling.
#   Testing at 1/3/5 will show:
#     1 QPS  → should be clean (below ceiling)
#     3 QPS  → near/at ceiling — expect some TTFT climb
#     5 QPS  → above ceiling — shows how it degrades
#
# Per step: 300 prompts
# ============================================================
set -euo pipefail
source "$(dirname "$0")/../.env"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="/workspace/benchmark_results/p95_2p2d_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

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
echo " BENCHMARK: P95 Stress Test — 2P + 2D (FINAL CONFIG)"
echo " Prefiller: batch=32768  model=32768"
echo " Decoder:   batch=2048   model=32768"
echo " Input:     15840 tokens | Prefix: $PREFIX_LEN | Output: 561"
echo " Unique:    $((15840 - PREFIX_LEN)) tokens per request"
echo " QPS sweep:  1 → 3 → 5 req/s  (long-context stress test)"
echo " Per step:    300 prompts"
echo " Results:   $RESULTS_DIR"
echo ""
echo " This is the core PD disagg value test."
echo " 1P+1D @ 1 QPS gave TTFT P50 = 24s (prefill blocks decode)."
echo " 2P+2D should show dramatically lower TTFT here."
echo "============================================================"

# ---- WARMUP ----
# P95 warmup is critical — 12000-token prefix KV must be cached
# on both prefillers before measuring. 50 prompts at 1 req/s.
echo ""
echo "[ WARMUP ] 50 prompts @ 1 req/s (priming 12000-tok prefix cache)..."
echo "  Do not skip — cold cache gives 5-10x worse TTFT."

vllm bench serve \
  "${COMMON_ARGS[@]}" \
  --num-prompts 50 \
  --request-rate 1 \
  --result-dir "$RESULTS_DIR" \
  --result-filename "warmup_discard.json" \
  2>&1 | tee "$RESULTS_DIR/warmup.log" | grep -E "(TTFT|Throughput|Finished|Error)" || true

rm -f "$RESULTS_DIR/warmup_discard.json"

echo ""
echo "[ WARMUP DONE ] Waiting 45s for GPU memory to settle..."
sleep 45

# ---- QPS SWEEP ----
echo ""
echo "[ RATE SWEEP ] Starting P95 stress test..."

set +e
for RATE in 1 3 5; do
  DURATION=$((300 / RATE))
  echo ""
  echo "──────────────────────────────────────────────────────"
  echo " Rate: $RATE req/s | Prompts: 300 | Duration: ~${DURATION}s"
  echo "──────────────────────────────────────────────────────"

  RESULT_FILE="p95_rate${RATE}_${TIMESTAMP}.json"

  vllm bench serve \
    "${COMMON_ARGS[@]}" \
    --num-prompts 300 \
    --request-rate "$RATE" \
    --result-dir "$RESULTS_DIR" \
    --result-filename "$RESULT_FILE" \
    2>&1 | tee -a "$RESULTS_DIR/p95_stress.log"

  RESULT_PATH="$RESULTS_DIR/$RESULT_FILE"

  if [ "$RATE" -lt 3 ]; then
    echo ""
    echo "  Cooling down 45s (long-context recovery)..."
    sleep 45
  fi
done
set -e

echo ""
echo " Results: $RESULTS_DIR"
echo " Analysis: python analyze_results.py $RESULTS_DIR"
echo "============================================================"