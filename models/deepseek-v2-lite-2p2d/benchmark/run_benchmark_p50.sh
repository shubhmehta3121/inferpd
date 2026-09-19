#!/bin/bash
# ============================================================
# run_benchmark_p50.sh — P50 QPS Sweep (2P + 2D validation)
#
# workload P50 profile:
#   Input:  7870 tokens | Output: 252 tokens | Prefix: 6000 tokens
#
# QPS sweep: 2 → 5 → 10
#   At 1P+1D the ceiling was ~5 req/s.
#   With 2P+2D we expect ~15 req/s ceiling.
#   Testing at 2/5/10 will show:
#     2 QPS  → should be clean (below ceiling)
#     5 QPS  → near/at ceiling — expect some TTFT climb
#    10 QPS  → above ceiling — shows how it degrades
#
# Per step: 600 prompts
# ============================================================
set -euo pipefail
source "$(dirname "$0")/../.env"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="/workspace/benchmark_results/p50_2p2d_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

PREFIX_LEN=6000

COMMON_ARGS=(
  --backend vllm
  --host localhost
  --port "$PROXY_PORT"
  --model "$MODEL_NAME"
  --dataset-name random
  --temperature 0
  --random-input-len 7870
  --random-output-len 252
  --random-prefix-len "$PREFIX_LEN"
  --seed 7
  --percentile-metrics ttft,tpot,itl,e2el
  --metric-percentiles 50,90,95,99
  --save-result
)

echo "============================================================"
echo " BENCHMARK: P50 QPS Sweep — 2P + 2D"
echo " QPS sweep:   2 → 5 → 10"
echo " Per step:    600 prompts"
echo " Results:     $RESULTS_DIR"
echo "============================================================"

# ---- WARMUP ----
echo ""
echo "[ WARMUP ] 100 prompts @ 5 req/s..."
vllm bench serve \
  "${COMMON_ARGS[@]}" \
  --num-prompts 100 \
  --request-rate 5 \
  --result-dir "$RESULTS_DIR" \
  --result-filename "warmup_discard.json" \
  2>&1 | tee "$RESULTS_DIR/warmup.log" | grep -E "(TTFT|Throughput|Finished|Error)" || true

rm -f "$RESULTS_DIR/warmup_discard.json"

echo ""
echo "[ WARMUP DONE ] Waiting 30s..."
sleep 30

# ---- QPS SWEEP ----
echo ""
echo "[ QPS SWEEP ] Starting..."

set +e
for RATE in 2 5 10; do
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

  if [ "$RATE" -lt 10 ]; then
    echo ""
    echo "  Cooling down 30s..."
    sleep 30
  fi
done
set -e

echo ""
echo " Results: $RESULTS_DIR"
echo "============================================================"