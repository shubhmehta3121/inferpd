# DeepSeek V2 Lite — 1P + 2D + LMCache MP

Stack: **LMCache multiprocess ZMQ server** + **vLLM `LMCacheMPConnector`** + **disagg proxy** (LMCache `disagg_prefill_mp` pattern). **Not** NIXL.

This folder is the **one prefiller, two decoders** variant: double decode throughput vs `deepseek-v2-lite-lmcache-mp` (1P1D) while keeping a **single** prefiller and **one** shared LMCache MP process.

| Component | Version |
|-----------|---------|
| vLLM | 0.18.0 |
| LMCache | 0.4.2 |
| PyTorch | 2.8.0 / CUDA 12.8 (RunPod template) |

**Ports:** LMCache MP **6000** (ZMQ) · Telemetry **5768** · Proxy **9000** · Prefiller **8100** · Decoders **8200**, **8201**

**GPUs (default in `.env`):** Prefiller **GPU 0** · Decoder **GPU 1** · Decoder **GPU 2**

---

## 1. Copy to RunPod

```bash
scp -r -P <SSH_PORT> -i ~/.ssh/<your-key> \
  ./deepseek-v2-lite-lmcmp-1p2d \
  root@<PUBLIC_IP>:/workspace/
```

On the pod: `cd /workspace/deepseek-v2-lite-lmcmp-1p2d`.

---

## 2. SSH into the pod

```bash
ssh root@<PUBLIC_IP> -p <SSH_PORT> -i ~/.ssh/<your-key>
```

---

## 3. One-time setup

```bash
cd /workspace/deepseek-v2-lite-lmcmp-1p2d

bash scripts/00_setup.sh
source /workspace/.venv/bin/activate
```

Edit **`.env`**: set **`HF_TOKEN`**, confirm **`DECODER_GPUS`**, **`DECODER_PORTS`**, and **`NUM_DECODERS`** match your hardware.

Download the model if needed:

```bash
bash scripts/01_download_model.sh
```

---

## 4. Start services (order matters)

Use **five** terminals (or **tmux** panes) — same `cd` and `source /workspace/.venv/bin/activate` where needed.

| Step | Script | What |
|------|--------|------|
| 1 | `bash scripts/07_launch_lmcache_mp_server.sh` | LMCache MP ZMQ server (blocks) |
| 2 | `bash scripts/04_launch_proxy.sh` | HTTP **9000** + telemetry **5768** (`--num-decoders 2`) |
| 3 | `bash scripts/03_launch_decoder.sh 0` | Decoder 0 · **8200** · GPU from `DECODER_GPUS[0]` |
| 4 | `bash scripts/03_launch_decoder.sh 1` | Decoder 1 · **8201** · GPU from `DECODER_GPUS[1]` |
| 5 | `bash scripts/02_launch_prefiller.sh` | Prefiller · **8100** · `PREFILLER_GPU` |

Wait until listeners come up before the next step (decoder processes before prefiller is fine).

**Proxy routing:** The proxy advances one global index per request and picks **prefiller** and **decoder** by round-robin on each pool. With **one** prefiller and **two** decoders, every request uses that prefiller; decoders alternate **8200 → 8201 → 8200 → …**.

---

## 5. Verify

```bash
cd /workspace/deepseek-v2-lite-lmcmp-1p2d
source /workspace/.venv/bin/activate
bash scripts/05_health_check.sh
```

Clients use the **proxy** only: `http://<pod-ip>:9000/v1/completions` or `/v1/chat/completions`.

---

## 6. Optional: monitoring

```bash
bash scripts/06_launch_monitoring.sh
```

Prometheus scrapes **both** decoder metrics ports (**8200** and **8201**) in `configs/prometheus.yml`.

---

## 7. Benchmark

```bash
cd benchmark
bash run_benchmark_p50.sh
```

Ensure **`PROXY_PORT`** in `.env` matches the benchmark.

---

## 8. Troubleshooting

- **`ss -tlnp | grep 6000`** — LMCache MP must listen before vLLM.
- **`ss -tlnp | grep 5768`** — telemetry thread (started with proxy).
- Prefiller needs **`LMCACHE_REQUEST_TELEMETRY_*`** in `02_launch_prefiller.sh` so the proxy unblocks after KV store.
- If **`BLOCK_SIZE=64`** fails, set **`BLOCK_SIZE=32`** in `.env` and restart prefiller and decoders.
- If LMCache MP becomes CPU-bound with two decoders, raise **`LMCACHE_MP_MAX_WORKERS`** in `.env` and restart **07**.

---

## 9. Hardware

**3× A100 80GB** on one node (one prefiller + two decoders). **`docker-compose.yml`** here is **legacy** (old NIXL-oriented example); prefer the shell scripts above.

---

## 10. Relation to other folders

| Folder | Layout |
|--------|--------|
| `deepseek-v2-lite-lmcache-mp` | 1P1D baseline |
| **This folder** | 1P2D + LMCache MP |
| `deepseek-v2-lite-2p2d` | 2P2D + **NIXL** (different connector and proxy CLI — do not mix) |
