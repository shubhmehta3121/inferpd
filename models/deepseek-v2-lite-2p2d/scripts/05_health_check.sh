#!/bin/bash
# ============================================================
# 05_health_check.sh — Verify all 5 services are running (2P+2D)
# ============================================================
source "$(dirname "$0")/../.env"

PASS=0
FAIL=0

check() {
  local name=$1 url=$2
  if curl -sf --max-time 10 "$url" > /dev/null 2>&1; then
    echo "  ✅ $name — OK"
    ((PASS++))
  else
    echo "  ❌ $name — UNREACHABLE ($url)"
    ((FAIL++))
  fi
}

check_nixl_port() {
  local name=$1 port=$2
  if ss -tlnp 2>/dev/null | grep -q ":$port "; then
    echo "  ✅ $name (port $port) — LISTENING"
    ((PASS++))
  else
    echo "  ❌ $name (port $port) — NOT LISTENING"
    ((FAIL++))
  fi
}

echo ""
echo "============================================================"
echo " Health Check — 2P + 2D PD Stack"
echo "============================================================"

echo ""
echo "[ Proxy ]"
check "Proxy  (port $PROXY_PORT)" "http://localhost:$PROXY_PORT/health"

echo ""
echo "[ Prefillers ]"
for i in "${!PREFILLER_PORTS[@]}"; do
  PORT=${PREFILLER_PORTS[$i]}
  GPU=${PREFILLER_GPUS[$i]}
  check "Prefiller $i  GPU $GPU  port $PORT" "http://localhost:$PORT/health"
done

echo ""
echo "[ Decoders ]"
for i in "${!DECODER_PORTS[@]}"; do
  PORT=${DECODER_PORTS[$i]}
  GPU=${DECODER_GPUS[$i]}
  check "Decoder $i  GPU $GPU  port $PORT" "http://localhost:$PORT/health"
done

echo ""
echo "[ NIXL Side Channels (decoders must be listening) ]"
for i in "${!NIXL_PORTS_DECODER[@]}"; do
  check_nixl_port "Decoder $i NIXL" "${NIXL_PORTS_DECODER[$i]}"
done

echo ""
echo "[ GPU Status ]"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits | while IFS=',' read -r idx name util mem_used mem_total; do
  echo "  GPU $idx: $name | Util: ${util}% | VRAM: ${mem_used}/${mem_total} MB"
done

echo ""
echo "[ Quick Inference Test — single request through proxy ]"
RESP=$(curl -sf --max-time 60 \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$MODEL_NAME\", \"prompt\": \"Hello, say hi.\", \"max_tokens\": 5, \"stream\": false}" \
  "http://localhost:$PROXY_PORT/v1/completions" 2>/dev/null)

if [ $? -eq 0 ]; then
  TEXT=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'][:80])" 2>/dev/null || echo "$RESP")
  echo "  ✅ Inference OK — response: $TEXT"
  ((PASS++))
else
  echo "  ❌ Inference FAILED"
  ((FAIL++))
fi

echo ""
echo "============================================================"
echo " Summary: $PASS passed, $FAIL failed"
if [ $FAIL -eq 0 ]; then
  echo " All services healthy — ready to benchmark."
else
  echo " Some services are down. Check logs:"
  echo "   tmux attach -t pd   (then Ctrl-B + 0-4 for each process)"
fi
echo "============================================================"