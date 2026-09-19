# DeepSeek-V2-Lite PD Disaggregation — Complete Deployment Guide

## inferpd | Phase 1: Single Node, 1 Prefill : 1 Decode

---

## 1. GPU Selection (RunPod)

**Rent: 2x A100 SXM** at **$1.49/hr each = $2.98/hr total**

Why A100 SXM specifically:

- **80GB VRAM each**: V2-Lite BF16 uses ~31GB → 49GB left for KV cache blocks
- **NVLink (SXM form factor)**: NIXL uses `cuda_ipc` transport over NVLink for KV transfer
(~~900 GB/s) vs PCIe (~~32 GB/s). This is critical — without NVLink, KV transfer becomes a bottleneck.
- **High availability (8 max)**: Actually rentable, unlike H100 NVL (Low)
- **117GB RAM, 16 vCPU**: Enough for LMCache buffers and proxy server

Do NOT use:

- RTX 4090 (24GB — too small for BF16, needs FP8 which adds complexity)
- A100 PCIe (no NVLink — slower NIXL transfers)
- H100 NVL (Low/Unavailable on RunPod, $3.07/hr)
- H100 SXM (Great but overkill at $2.69/hr for V2-Lite testing)

RunPod pod settings:

- Container disk: 60 GB
- Volume disk:   150 GB (mounted at /workspace)
- Template:      PyTorch 2.8.0 + CUDA 12.8
- Pod type:      Secure Cloud (NOT Serverless)

---

## 2. Software Versions


| Component   | Version | Notes                                          |
| ----------- | ------- | ---------------------------------------------- |
| Python      | 3.11    | Comes with RunPod template                     |
| PyTorch     | 2.8.0   | Comes with RunPod template                     |
| CUDA        | 12.8    | Comes with RunPod template                     |
| vLLM        | 0.16.0  | Tested working with V2-Lite + LMCache          |
| LMCache     | Local   | `uv pip install -e ./LMCache` (PD fix)         |
| NIXL        | latest  | `uv pip install nixl`                          |
| DeepGEMM    | latest  | Required for MLA kernels                       |
| UV          | latest  | Fast Python package installer (auto-installed) |
| Model dtype | BF16    | Native HF weights, fits in 80GB easily         |


### LMCache Patch (PD Decoder Eviction Fix)

This repo ships a **local LMCache copy** (`LMCache/`) with a temporary fix for the PD decoder eviction bug that caused "Failed to allocate memory object, retrying..." and decoder OOM during benchmarking.

**What the fix does:**

1. **cache_engine.py** (retrieve loop): Swaps order — `ref_count_down()` before `remove()`. Original order had lookup pinning (`ref_count==2`) when `remove()` ran, so PD backend never deleted; buffer leaked.
2. **pd_backend.py** (`remove()`): Calls `mem_obj.ref_count_down()` before `del` when `ref_count==1`, so the allocator reclaims the block. Original had no release → memory stayed allocated.

**Install:** `00_setup.sh` uses `uv pip install -e ./LMCache --no-build-isolation` (editable install from local copy). Run setup from project root so `./LMCache` exists.

**Rollback:** If upstream LMCache fixes the bug, switch to PyPI:

```bash
uv pip uninstall lmcache
uv pip install lmcache
```

In `scripts/00_setup.sh`, change `uv pip install -e ./LMCache --no-build-isolation` back to `uv pip install lmcache`.

**Files modified (search for `[TEMPORARY FIX]`):**

- `LMCache/lmcache/v1/cache_engine.py` — ref_count_down before remove
- `LMCache/lmcache/v1/storage_backend/pd_backend.py` — ref_count_down before del

### Why BF16 over FP8?

- A100 SXM has 80GB. V2-Lite BF16 = ~31GB. You have 49GB for KV cache.
- BF16 is the native precision from HuggingFace — zero precision loss.
- FP8 on A100 is NOT natively supported in hardware (FP8 is a Hopper/H100 feature).
vLLM simulates FP8 on A100 via software, which can be slower not faster.
- Only switch to FP8 if you later need to fit on smaller GPUs.

---

## 3. Architecture Overview

```
Client Request (streaming)
        │
        ▼
┌────────────────────────────────┐
│     Proxy Server (Port 9000)   │
│   disagg_proxy_server.py       │
│   Prometheus metrics at /metrics│
└──────────┬─────────────────────┘
           │
     ┌─────┴────────────────────┐
     │                          │
     ▼  Phase 1: Prefill        ▼  Phase 2: Decode
┌──────────────┐          ┌──────────────────┐
│  Prefiller   │  NIXL/   │    Decoder       │
│  GPU 0       │ NVLink   │    GPU 1         │
│  Port 8100   │──KV────►│    Port 8200     │
│  kv_producer │  cache   │    kv_consumer   │
│  LMCache     │          │    LMCache       │
│  (sender)    │          │    (receiver)    │
└──────────────┘          └──────────────────┘
                                   │
                          Streaming tokens
                                   ▼
                               Client
```

Flow:

1. Proxy receives request, generates a unique request ID, sends to Prefiller with
  max_tokens=1 + X-Request-Id header + kv_transfer_params (includes disagg_spec
   with decoder's NIXL ports: receiver_init_port, receiver_alloc_port)
2. Prefiller runs prefill, LMCache transfers KV tensors to Decoder via NIXL (NVLink)
3. LMCache notifies Proxy Telemetry (port 7500): "KV transfer complete for request X"
4. Proxy unblocks (was waiting for this signal) and forwards request to Decoder
5. Decoder skips recomputation (KV is already there), generates tokens, streams back

### Design notes

- **Telemetry is required**: The proxy blocks until LMCache POSTs to `/api/v1/telemetry`
that the KV transfer is done. Without it, the proxy would forward to the decoder
before KV cache arrives → decode would fail or recompute. Do not remove telemetry.
- **Request ID**: Proxy generates it via `uuid.uuid4()[:16]`, sends as `X-Request-Id`.
vLLM wraps it as `chatcmpl-{id}-{seq}`. LMCache telemetry sends that back; proxy
extracts the id to unblock the matching pending request.
- **Why only decoder init/alloc port (not prefiller)?** Prefiller is the KV *sender* —
it connects TO the decoder. Decoder is the *receiver* — it binds on 7300/7400.
We tell the prefiller "send KV here" (decoder's ports). The prefiller does not
advertise its own ports; it initiates the NIXL connection as the client.

---

## 4. Step-by-Step Deployment

### Step 0: Spin up RunPod Pod

1. Go to RunPod → Pods → Deploy a Pod → GPU Pod
2. Select **A100 SXM** → click **Customize Deployment**
3. Set: GPU Count = 2, Container Disk = 60GB, Volume = 150GB, mount at /workspace
4. Template: PyTorch 2.8.0 + CUDA 12.8
5. Expose ports: 8100, 8200, 9000, 9090, 3000
6. Click Deploy → wait for pod to start → open terminal

### Step 1: Copy project files to pod

```bash
# From your local machine, use RunPod's file upload or scp:
scp -r deepseek-v2-lite/ root@<pod-ip>:/workspace/
scp -i ~/.ssh/<your-key> -P <ssh-port> -r models/deepseek-v2-lite root@<pod-ip>:/workspace/deepseek-v2-lite

# Faster transfer (tar excluding LMCache, then clone + patch):
tar czf - -C models --exclude='deepseek-v2-lite/LMCache' deepseek-v2-lite | ssh -i ~/.ssh/<your-key> -p <ssh-port> root@<pod-ip> "cd /workspace && tar xzf - --no-same-owner"
ssh -i ~/.ssh/<your-key> -p <ssh-port> root@<pod-ip> "cd /workspace/deepseek-v2-lite && git clone https://github.com/LMCache/LMCache.git LMCache"
scp -i ~/.ssh/<your-key> -P <ssh-port> models/deepseek-v2-lite/LMCache/lmcache/v1/cache_engine.py root@<pod-ip>:/workspace/deepseek-v2-lite/LMCache/lmcache/v1/
scp -i ~/.ssh/<your-key> -P <ssh-port> models/deepseek-v2-lite/LMCache/lmcache/v1/storage_backend/pd_backend.py root@<pod-ip>:/workspace/deepseek-v2-lite/LMCache/lmcache/v1/storage_backend/

# Or clone from your git repo:
cd /workspace && git clone <your-repo-url> deepseek-v2-lite
cd deepseek-v2-lite
```

### Step 2: Configure environment

```bash
# Edit .env and add your HuggingFace token
nano .env
# Change: HF_TOKEN=hf_YOUR_TOKEN_HERE
# To:     HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

### Step 3: Run setup (installs all packages)

```bash
chmod +x scripts/*.sh benchmark/*.sh
bash scripts/00_setup.sh
```

### Step 4: Download the model (~31GB, saves to /workspace/models)

```bash
bash scripts/01_download_model.sh
# Takes ~10-15 minutes. Only needed once — model persists on volume disk.
```

### Step 5: Launch the stack — ORDER MATTERS

**Start order: Proxy → Decoder → Prefiller**

The proxy must be up before the prefiller boots, because LMCache (inside the
prefiller) tries to connect to the telemetry endpoint on port 7500 at startup.
The decoder must be up before the prefiller so its NIXL sockets (7300/7400) are
ready when the prefiller starts making connections.

```bash
# Open tmux
tmux new -s pd

# ── Window 1: Proxy (FIRST) ──────────────────────────────────
# Ctrl+B C  (or just in first window)
bash scripts/04_launch_proxy.sh
# Wait for BOTH of these lines before continuing:
#   "LMCache telemetry server listening on 0.0.0.0:7500"
#   "Application startup complete."  (the HTTP proxy on 9000)

# ── Window 2: Decoder (SECOND) ──────────────────────────────
# Ctrl+B C
bash scripts/03_launch_decoder.sh
# Wait for: "vLLM application startup complete"
# You should also see LMCache bind NIXL ports 7300 and 7400 in the log

# ── Window 3: Prefiller (LAST) ──────────────────────────────
# Ctrl+B C
bash scripts/02_launch_prefiller.sh
# Wait for: "vLLM application startup complete"
# In the proxy log you should see:
#   "LMCache telemetry server listening..."  (already up)
# When first request comes in you'll see:
#   "[<id>] Prefill HTTP done in X.XXXs"
#   "Telemetry [request_store_finished]: chatcmpl-..."
#   "[<id>] KV transfer confirmed in X.XXXms"

# ── Window 4: Monitoring (optional) ────────────────────────
# Ctrl+B C
bash scripts/06_launch_monitoring.sh
```

### Step 6: Verify everything is running

```bash
bash scripts/05_health_check.sh
```

Expected output:

```
  ✅ Prefiller (8100) — OK
  ✅ Decoder   (8200) — OK
  ✅ Proxy     (9000) — OK
  ✅ Inference test PASSED
```

### Step 7: Run benchmarks

```bash
cd benchmark

# Start with baseline (P50 production-like workload)
bash run_benchmark_baseline.sh

# Then long-context stress test (P95)
bash run_benchmark_p95.sh

# Find maximum QPS
bash run_benchmark_peak_qps.sh

# Analyze results vs target production profile
python analyze_results.py /workspace/benchmark_results/
```

---

## 5. Production-like workload


| Metric      | Current Prod | Target (with PD disagg) |
| ----------- | ------------ | ----------------------- |
| TTFT P50    | ~2.0s        | < 0.5s 🎯               |
| TTFT P95    | ~3.5s        | < 1.5s 🎯               |
| E2E P50     | ~11s         | < 6s                    |
| E2E P95     | ~25s         | < 15s                   |
| Peak QPS    | ~60          | > 60 (same or better)   |
| Input P50   | 7870 tokens  | benchmark target        |
| Input P95   | 15840 tokens | benchmark target        |
| Prefix hits | 80%          | maintained              |


---

## 6. Port Map Reference


| Service           | Port | Purpose                                                |
| ----------------- | ---- | ------------------------------------------------------ |
| Proxy HTTP        | 9000 | **Send all client requests here**                      |
| LMCache Telemetry | 7500 | LMCache posts "KV done" callbacks here (pd_proxy_port) |
| NIXL init         | 7300 | Decoder binds here; prefiller connects to transfer KV  |
| NIXL alloc        | 7400 | Decoder binds here; NIXL memory negotiation            |
| Prefiller vLLM    | 8100 | OpenAI-compatible API (internal, proxy routes to it)   |
| Decoder vLLM      | 8200 | OpenAI-compatible API (internal, proxy routes to it)   |
| Prometheus        | 9090 | Metrics scraping                                       |
| Grafana           | 3000 | Dashboard (admin/admin)                                |


Port wiring — every value must be consistent across these three files:


| Port | .env var           | prefiller YAML | decoder YAML       | proxy CLI arg        |
| ---- | ------------------ | -------------- | ------------------ | -------------------- |
| 7500 | PROXY_NIXL_PORT    | pd_proxy_port  | —                  | --telemetry-port     |
| 7300 | DECODER_INIT_PORT  | —              | pd_peer_init_port  | --decoder-init-port  |
| 7400 | DECODER_ALLOC_PORT | —              | pd_peer_alloc_port | --decoder-alloc-port |


---

## 7. Monitoring & Grafana

1. Access Grafana at [http://localhost:3000](http://localhost:3000) (forward port in RunPod settings)
2. Login: admin / admin
3. Go to Dashboards → Import → Upload JSON file
4. Select: `monitoring/grafana_dashboard.json`

Key panels to watch:

- TTFT P50/P95 — target <2s / <3.5s for V3.2 eventual parity
- GPU KV Cache % — keep below 85% to avoid OOM
- Active Requests — watch for queue buildup
- Prefill Latency — how fast the prefiller is processing input

---

## 8. Troubleshooting

### vLLM fails to start with MLA error

```bash
# Add --enforce-eager if CUDA graphs fail
# Add --disable-custom-all-reduce for MoE stability
```

### NIXL connection refused / KV transfer fails / disagg_spec=null error

```bash
# 1. Verify start order was proxy → decoder → prefiller
#    (LMCache tries to reach port 7500 at prefiller boot; if nothing is there,
#     disagg_spec stays null and the prefiller crashes with AttributeError)

# 2. Confirm telemetry endpoint is up before starting prefiller
curl http://localhost:7500/health   # must return {"status":"ok"}

# 3. Confirm decoder NIXL sockets are bound before starting prefiller
ss -tlnp | grep -E "7300|7400"     # must show LISTEN entries

# 4. Verify UCX transport is set
echo $UCX_TLS   # should be: cuda_ipc,cuda_copy,tcp

# 5. In the proxy log you should see this sequence per request:
#      "[<id>] Prefill HTTP done in X.XXXs"
#      "Telemetry [request_store_finished]: chatcmpl-..."
#      "[<id>] KV transfer confirmed in X.XXXms"
#    If you see "[<id>] KV telemetry timeout" it means the telemetry
#    callback never arrived — check prefiller's lmcache-prefiller-config.yaml
#    pd_proxy_port matches --telemetry-port in 04_launch_proxy.sh (both 7500)
```

### OOM during benchmarking

```bash
# Reduce gpu_memory_utilization in .env
# Reduce max_num_seqs
# Reduce max_model_len
# Verify nvidia-smi shows enough free VRAM before starting
nvidia-smi
```

### Slow TTFT

```bash
# Prefix caching is intentionally disabled for PD disaggregation.
# TTFT gains come from the KV transfer: prefiller does the heavy prefill
# work and ships KV tensors to the decoder via NVLink at ~900 GB/s.
# Check KV wait time in proxy logs:
grep "KV transfer confirmed" /workspace/logs/proxy/proxy_*.log | tail -20
# Should be single-digit milliseconds for NVLink transfers.
```

### LMCache disconnected/errors

```bash
# Try without LMCache first using pure NixlConnector:
# Change kv_connector from "LMCacheConnectorV1" to "NixlConnector"
# in scripts/02_launch_prefiller.sh and 03_launch_decoder.sh
```

---

## 9. Scaling to 1:3 (P:D ratio) — Phase 2

Once Phase 1 is stable, scaling to 1 prefill : 3 decode means:

- Rent a node with 4x A100 SXM
- GPU 0: Prefiller
- GPU 1, 2, 3: 3x Decoder instances
- Update proxy to round-robin across 3 decoder ports
- This is optimal when prefix cache hit rate > 70% (you have 80% → perfect)

