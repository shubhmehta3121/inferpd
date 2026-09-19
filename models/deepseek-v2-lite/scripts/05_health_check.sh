#!/bin/bash
# ============================================================
# 05_health_check.sh — Verify all services are running
# Run this after launching proxy, decoder, and prefiller
# (in that order — see README Step 5)
# ============================================================

source "$(dirname "$0")/../.env"

PASS=0
FAIL=0

check_service() {
  local name=$1
  local url=$2
  local response

  response=$(curl -sf --max-time 20 "$url" 2>/dev/null)
  if [ $? -eq 0 ]; then
    echo "  ✅ $name — OK ($url)"
    ((PASS++))
  else
    echo "  ❌ $name — UNREACHABLE ($url)"
    ((FAIL++))
  fi
}

echo ""
echo "============================================================"
echo " Health Check — DeepSeek V2-Lite PD Stack"
echo "============================================================"

echo ""
echo "[ vLLM Services ]"
check_service "Prefiller (8100)" "http://localhost:$PREFILLER_PORT/health"
check_service "Decoder   (8200)" "http://localhost:$DECODER_PORT/health"
check_service "Proxy     (9000)" "http://localhost:$PROXY_PORT/health"

echo ""
echo "[ NIXL Side Channel — decoder must be listening on 5560 ]"
DECODER_NIXL_PORT="${DECODER_NIXL_PORT:-5560}"
if ss -tlnp 2>/dev/null | grep -q ":$DECODER_NIXL_PORT "; then
  echo "  ✅ NIXL side channel ($DECODER_NIXL_PORT) — LISTENING"
  ((PASS++))
else
  echo "  ❌ NIXL side channel ($DECODER_NIXL_PORT) — NOT LISTENING (decoder NIXL not ready)"
  ((FAIL++))
fi

echo ""
echo "[ Monitoring ]"
check_service "Prometheus (9090)" "http://localhost:$PROMETHEUS_PORT/-/healthy"
check_service "Grafana    (3000)" "http://localhost:$GRAFANA_PORT/api/health"

echo ""
echo "[ GPU Status ]"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
  --format=csv,noheader,nounits | while IFS=',' read -r idx name util mem_used mem_total temp; do
  echo "  GPU $idx: $name | Util: ${util}% | VRAM: ${mem_used}/${mem_total} MB | Temp: ${temp}°C"
done

echo ""
echo "[ Quick Inference Test ]"
echo "  Testing proxy endpoint (single non-streaming request)..."
RESPONSE=$(curl -sf --max-time 60 \
  -H "Content-Type: application/json" \
  -d '{"model": "/workspace/models/deepseek-v2-lite", "prompt": "Hello, say hi back.", "max_tokens": 10, "stream": false}' \
  "http://localhost:$PROXY_PORT/v1/completions" 2>/dev/null)

if [ $? -eq 0 ]; then
  echo "  ✅ Inference test PASSED"
  echo "  Response snippet: $(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['text'][:80])" 2>/dev/null || echo "$RESPONSE" | head -c 200)"
else
  echo "  ❌ Inference test FAILED — check proxy/prefiller/decoder logs"
fi

echo ""
echo "============================================================"
echo " Summary: $PASS passed, $FAIL failed"
if [ $FAIL -eq 0 ]; then
  echo " All services healthy! Ready for benchmarking."
  echo " Run: cd benchmark && bash run_benchmark_baseline.sh"
else
  echo " Some services are down. Check logs in /workspace/logs/"
  echo ""
  echo " Tip: verify start order was proxy → decoder → prefiller"
  echo "   tail -50 /workspace/logs/proxy/proxy_*.log"
  echo "   tail -50 /workspace/logs/decoder/decoder_*.log"
  echo "   tail -50 /workspace/logs/prefiller/prefiller_*.log"
fi
echo "============================================================"
echo ""
