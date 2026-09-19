# DeepSeek-V3 Deployment

Two configurations available. Start with **v1** to validate the setup and collect baseline metrics. Move to **v2** when you want lower TTFT and independent prefill/decode scaling.

---

## v1 — Single Node (this folder)

| Property | Value |
|---|---|
| **Nodes** | 1 |
| **GPUs** | 8× H200 (1 machine) |
| **Cost** | ~$30 / hr |
| **Parallelism** | Tensor Parallel TP=8 across all 8 GPUs |
| **Model precision** | BF16 (no quantisation) |
| **KV connector** | `LMCacheConnectorV1` — `kv_both` (single instance handles prefill + decode) |
| **KV cache — GPU** | vLLM Automatic Prefix Caching (APC) — hot blocks stay in GPU VRAM |
| **KV cache — CPU** | LMCache spills overflow blocks to CPU RAM |
| **Compression** | **CacheGen** — compresses KV blocks before writing to CPU RAM (~3-5× smaller) |
| **NIXL** | Installed but not used as a transport (no inter-node transfer needed) |
| **Model** | Stable: `deepseek-ai/DeepSeek-V3.2` (not experimental) |
| **Disaggregated prefill** | ❌ Not possible — model needs all 8 GPUs; none left to split |

### Files
```
deepseek-v3.2/  (this folder)
  .env                    ← secrets (fill HF_TOKEN before deploying)
  00-env.sh               ← source this first on the server
  01-install.sh           ← run once to install all dependencies
  02-start-server.sh      ← starts the vLLM server
  lmcache_config.yaml     ← LMCache config (CPU cache + CacheGen)
  prometheus.yml          ← Prometheus scrape config
  docker-compose.yml      ← monitoring stack (Prometheus + Grafana + exporters)
  benchmark.py            ← load test matching production traffic
  system_validation.py    ← pre-flight GPU / package check
  RUNBOOK.md              ← step-by-step deployment guide
```

### Startup order
```bash
source 00-env.sh
python system_validation.py
docker compose up -d

tmux new -s vllm
source 00-env.sh && source $PYTHON_ENV_PATH/bin/activate
bash 02-start-server.sh 2>&1 | tee $VLLM_LOGS_DIR/server.log
# Wait for: "Uvicorn running on http://0.0.0.0:8000"

curl http://localhost:8000/v1/models
python benchmark.py --url http://localhost:8000 --qps 30 --num-requests 300
```

---

## v2 — Disaggregated Prefill, Two Nodes (`v2/` folder)

| Property | Value |
|---|---|
| **Nodes** | 2 separate machines |
| **GPUs** | 8× H200 per node = 16 GPUs total |
| **Cost** | ~$60 / hr |
| **Parallelism** | TP=8 on EACH node independently |
| **Model precision** | BF16 (no quantisation) |
| **KV connector** | `LMCacheConnectorV1` — `kv_producer` on Node A, `kv_consumer` on Node B |
| **KV cache — CPU** | LMCache on prefiller node only (hot prefix store) |
| **Compression** | **CacheGen** — on the prefiller, compresses blocks in the CPU prefix cache |
| **NIXL** | ✅ Active — transfers KV blocks from prefiller → decoder over network (UCX) |
| **Disaggregated prefill** | ✅ Full P/D disaggregation via LMCache + NIXL |

### Why two nodes?
DeepSeek-V3 in BF16 needs ~700 GB VRAM. A single 8xH200 node has 1200 GB total — all of it is needed for one model copy. You can't split 4+4 (4 GPUs = 600 GB, too small for the model). Each disaggregated instance therefore needs its own 8-GPU node.

### Files (`v2/`)
```
v2/
  env.sh                  ← source this on both nodes (extends ../00-env.sh)
  lmcache-prefiller.yaml  ← LMCache config for Node A (sender + CPU cache + CacheGen)
  lmcache-decoder.yaml    ← LMCache config for Node B (receiver via NIXL)
  03-start-prefiller.sh   ← run on Node A
  04-start-decoder.sh     ← run on Node B (after prefiller is up)
  05-start-proxy.sh       ← run on either node (after both vLLM instances are up)
  proxy_server.py         ← FastAPI proxy: routes prefill → decode
```

### Things to fill in before deploying v2
1. `v2/env.sh` — set `PREFILLER_IP` and `DECODER_IP`
2. `v2/lmcache-decoder.yaml` — set `pd_peer_host` to `PREFILLER_IP`
3. `.env` — `HF_TOKEN` (same on both nodes)

---

## Component summary

| Component | v1 (single node) | v2 (disaggregated) |
|---|---|---|
| vLLM | TP=8, kv_both | TP=8 each, kv_producer / kv_consumer |
| LMCache | CPU spill + prefix cache | CPU prefix cache (prefiller only) |
| CacheGen | ✅ CPU cache compression | ✅ CPU cache compression (prefiller) |
| NIXL | ❌ not used as transport | ✅ inter-node KV transfer |
| Proxy | ❌ clients hit vLLM directly | ✅ routes prefill→decode |
| Cost | ~$30/hr | ~$60/hr |
| TTFT | Good | Better (no decode interference) |
