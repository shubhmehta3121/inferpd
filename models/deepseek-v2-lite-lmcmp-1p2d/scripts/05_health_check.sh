#!/bin/bash
# ============================================================
# 05_health_check.sh — 1P + 2D
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
echo " Health Check — DeepSeek V2-Lite (LMCache MP, 1P+2D)"
echo "============================================================"

echo ""
echo "[ vLLM + proxy ]"
check_service "Prefiller" "http://localhost:$PREFILLER_PORT/health"

if [ -n "${DECODER_PORTS+x}" ] && [ "${#DECODER_PORTS[@]}" -gt 0 ]; then
  for i in "${!DECODER_PORTS[@]}"; do
    check_service "Decoder $i" "http://localhost:${DECODER_PORTS[$i]}/health"
  done
else
  check_service "Decoder" "http://localhost:${DECODER_PORT:-8200}/health"
fi

check_service "Proxy HTTP" "http://localhost:$PROXY_PORT/health"

echo ""
echo "[ LMCache MP ZMQ listener ]"
MP_PORT="${LMCACHE_MP_PORT:-6000}"
if ss -tlnp 2>/dev/null | grep -q ":$MP_PORT "; then
  echo "  ✅ LMCache MP (TCP :$MP_PORT) — LISTENING"
  ((PASS++))
else
  echo "  ❌ LMCache MP (:$MP_PORT) — NOT LISTENING"
  ((FAIL++))
fi

echo ""
echo "[ Telemetry (proxy thread) ]"
TPORT="${TELEMETRY_PORT:-5768}"
if ss -tlnp 2>/dev/null | grep -q ":$TPORT "; then
  echo "  ✅ Telemetry (:$TPORT) — LISTENING"
  ((PASS++))
else
  echo "  ❌ Telemetry (:$TPORT) — NOT LISTENING"
  ((FAIL++))
fi

echo ""
echo "[ Monitoring ]"
check_service "Prometheus" "http://localhost:$PROMETHEUS_PORT/-/healthy"
check_service "Grafana" "http://localhost:$GRAFANA_PORT/api/health"

echo ""
echo "[ GPU ]"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
  --format=csv,noheader,nounits | while IFS=',' read -r idx name util mem_used mem_total temp; do
  echo "  GPU $idx: $name | Util: ${util}% | VRAM: ${mem_used}/${mem_total} MB | Temp: ${temp}°C"
done

echo ""
echo "[ Inference smoke ]"
RESPONSE=$(curl -sf --max-time 120 \
  -H "Content-Type: application/json" \
  -d '{"model": "'"$MODEL_PATH"'", "prompt": "Hi.", "max_tokens": 5, "stream": false}' \
  "http://localhost:$PROXY_PORT/v1/completions" 2>/dev/null || true)

if [ -n "$RESPONSE" ]; then
  echo "  ✅ Proxy /v1/completions responded"
  ((PASS++))
else
  echo "  ❌ Proxy inference failed"
  ((FAIL++))
fi

echo ""
echo "============================================================"
echo " Summary: $PASS passed, $FAIL failed"
echo " Start order: 07 (LMCache MP) → 04 (proxy) → 03 (decoder 0 & 1) → 02 (prefiller)"
echo "============================================================"
echo ""
