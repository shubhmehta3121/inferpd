#!/bin/bash
# ============================================================
# 05_health_check.sh — 1P + 3D (DeepSeek-V3.2-NVFP4)
# Two-node: set DISAGG_HEALTH_PREFILLER_HOST and
# DISAGG_HEALTH_DECODER_TARGET_CSV (host:port,host:port,...)
# ============================================================

source "$(dirname "$0")/../.env"

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
if [ "${NUM_GPUS:-0}" -lt 8 ]; then
  echo "[WARN] Expected ≥8 GPUs on this host for the 8-GPU slice; saw ${NUM_GPUS:-0}."
fi

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
echo " Health Check — DeepSeek-V3.2-NVFP4 (LMCache MP, 1P+3D)"
echo "============================================================"

PREF_H="${DISAGG_HEALTH_PREFILLER_HOST:-localhost}"
check_service "Prefiller" "http://${PREF_H}:${PREFILLER_PORT}/health"

NUM_D="${NUM_DECODERS:-3}"
BASE_PORT="${DECODER_PORT:-8200}"
TARGET_CSV="${DISAGG_HEALTH_DECODER_TARGET_CSV:-}"

if [ -n "$TARGET_CSV" ]; then
  IFS=',' read -ra DEC_TARGETS <<< "$TARGET_CSV"
  if [ "${#DEC_TARGETS[@]}" -ne "$NUM_D" ]; then
    echo "  ⚠️  DISAGG_HEALTH_DECODER_TARGET_CSV has ${#DEC_TARGETS[@]} entries; NUM_DECODERS=$NUM_D"
  fi
  for ((i = 0; i < ${#DEC_TARGETS[@]}; i++)); do
    check_service "Decoder $i" "http://${DEC_TARGETS[i]}/health"
  done
else
  for ((i = 0; i < NUM_D; i++)); do
    check_service "Decoder $i" "http://localhost:$((BASE_PORT + i))/health"
  done
fi

PROXY_H="${DISAGG_HEALTH_PROXY_HOST:-localhost}"
check_service "Proxy HTTP" "http://${PROXY_H}:${PROXY_PORT}/health"

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
check_service "Prometheus" "http://localhost:${PROMETHEUS_PORT:-9090}/-/healthy"
check_service "Grafana" "http://localhost:${GRAFANA_PORT:-3000}/api/health"

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
  -d '{"model": "'"$MODEL_NAME"'", "prompt": "Hi.", "max_tokens": 5, "stream": false}' \
  "http://${PROXY_H}:${PROXY_PORT}/v1/completions" 2>/dev/null || true)

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
echo " Start order: 07 (LMCache MP) → 04 (proxy) → 03 ×3 (decoders) → 02 (prefiller)"
echo "============================================================"
echo ""
