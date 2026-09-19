# DeepSeek V2 Lite — 1P1D + LMCache MP

Stack: **LMCache multiprocess ZMQ server** + **vLLM `LMCacheMPConnector`** + **disagg proxy** (from LMCache `examples/disagg_prefill_mp`). **Not** Nixl.

| Component | Version |
|-----------|---------|
| vLLM | 0.18.0 |
| LMCache | 0.4.2 |
| PyTorch | 2.8.0 / CUDA 12.8 (RunPod template) |

**Ports:** LMCache MP **6000** (ZMQ) · Telemetry **5768** · Proxy **9000** · Prefiller **8100** · Decoder **8200**

---

## 1. Copy repo to RunPod (from your laptop)

**This folder only** (recommended — smaller upload):

```bash
scp -r -P <SSH_PORT> -i ~/.ssh/<your-key> \
  ./deepseek-v2-lite-lmcache-mp \
  root@<PUBLIC_IP>:/workspace/
```

Run from the directory **above** `deepseek-v2-lite-lmcache-mp` (or use the full path to this folder on your machine).

**Whole git repo** (slower, larger):

```bash
scp -r -P <SSH_PORT> -i ~/.ssh/<your-key> \
  /path/to/inferpd \
  root@<PUBLIC_IP>:/workspace/
```

Then on the pod use `cd /workspace/inferpd/models/deepseek-v2-lite-lmcache-mp` in the steps below.

---

## 2. SSH into the pod

```bash
ssh root@<PUBLIC_IP> -p <SSH_PORT> -i ~/.ssh/<your-key>
```

---

## 3. One-time setup (on the pod)

```bash
cd /workspace/deepseek-v2-lite-lmcache-mp

# Optional: chmod +x scripts/*.sh
bash scripts/00_setup.sh
source /workspace/.venv/bin/activate
```

Edit **`.env`**: set **`HF_TOKEN`** (and **`MODEL_PATH` / `MODEL_NAME`** if your model is not under `/workspace/models/deepseek-v2-lite`).

Download model (if not already on volume):

```bash
bash scripts/01_download_model.sh
```

---

## 4. Start services (order matters)

Use **four** terminals (or **tmux** panes) — same `cd` and `source /workspace/.venv/bin/activate` in each.

| Step | Script | What |
|------|--------|------|
| 1 | `bash scripts/07_launch_lmcache_mp_server.sh` | LMCache MP server (blocks; keep running) |
| 2 | `bash scripts/04_launch_proxy.sh` | HTTP **9000** + telemetry **5768** |
| 3 | `bash scripts/03_launch_decoder.sh` | Decoder on GPU 1 · **8200** |
| 4 | `bash scripts/02_launch_prefiller.sh` | Prefiller on GPU 0 · **8100** |

Wait until each step listens before starting the next (decoder before prefiller is fine).

---

## 5. Verify

```bash
cd /workspace/deepseek-v2-lite-lmcache-mp
source /workspace/.venv/bin/activate
bash scripts/05_health_check.sh
```

Clients call the **proxy** only: `http://<pod-ip>:9000/v1/completions` (or chat completions).

---

## 6. Optional: monitoring

```bash
bash scripts/06_launch_monitoring.sh
```

(Prometheus/Grafana — only if that script matches your layout.)

---

## 7. Benchmark

```bash
cd benchmark
bash run_benchmark_p50.sh
```

Ensure **`PROXY_PORT`** in `.env` matches what clients use.

---

## 8. Troubleshooting

- **`ss -tlnp | grep 6000`** — LMCache MP should listen before starting vLLM.
- **`ss -tlnp | grep 5768`** — telemetry thread (started with proxy).
- Prefiller needs **`LMCACHE_REQUEST_TELEMETRY_*`** (set in `02_launch_prefiller.sh`) so the proxy unblocks after KV store.
- If **`BLOCK_SIZE=64`** fails on vLLM for this model, set **`BLOCK_SIZE=32`** in `.env` and restart prefiller/decoder.

---

## 9. Hardware

**2× A100 80GB** on one node is enough for this config. **`docker-compose.yml`** in this repo is **outdated** (old Nixl stack); use the shell scripts above.
