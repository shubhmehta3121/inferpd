# Deployment Runbook — DeepSeek-V3 on 8× H200 (v1, single node)

Follow these steps in order. Every command is copy-pasteable.

---

## Your RunPod values (fill these after each pod start)

Ports change every time you restart the pod. Get these from the RunPod UI → your pod → "Connect" / "Direct TCP".

| Item | Value (example) | Use for |
|------|-----------------|--------|
| **Public IP** | `<PUBLIC_IP>` | SCP host, SSH host |
| **SSH port** | `<SSH_PORT>` e.g. `19592` | SCP `-P`, SSH `-p` (maps to internal :22) |
| **Pod ID** | `<POD_ID>` e.g. `abc123xyz-64411989` | Optional: RunPod HTTP proxy URL |

**We use Direct TCP only** — it supports SCP, SFTP, and port forwarding. Proxied SSH is not used.

---

## Phase 0 — Do this on your laptop BEFORE paying for the instance

### 0.1 Fill in secrets

Open `models/deepseek-v3.2/.env` and set your HuggingFace token:
```
HF_TOKEN=hf_YOUR_REAL_TOKEN_HERE
VLLM_API_KEY=         # leave blank for now
```

### 0.2 Confirm the model repo name

Stable model is `deepseek-ai/DeepSeek-V3.2` (not -Exp).  
If the repo name changes, update `MODEL_REPO=` in `models/deepseek-v3.2/00-env.sh`.

### 0.3 Adjust CPU cache size

In `models/deepseek-v3.2/lmcache_config.yaml`, set `max_local_cpu_size` to `(server RAM GB) - 200`.  
- 512 GB RAM → `312`  
- 1600 GB RAM → `1400`  
You can check after SSH with: `free -g | awk '/^Mem:/ {print $2}'`

---

## Phase 1 — Provision the instance

**What to select:**
- GPU: **8× H200 SXM** (SXM has NVLink; PCIe does not)
- OS image: **Ubuntu 24.04** with **NVIDIA driver ≥ 580**
- Storage: **Container disk 40–60 GB** + **Volume/Network disk 1.5–2 TB**

### RunPod-specific

Once your Pod starts, open the pod in the RunPod UI and note:

- **Direct TCP** → **Public IP** and **SSH port** (the one mapping to internal :22). Use these for all SCP and SSH commands below.
- Optionally **Pod ID** and expose ports **8000** / **3000** if you want the HTTP proxy URLs (`https://<POD_ID>-8000.proxy.runpod.net`).

---

## Phase 2 — Copy files to the server

**Run on your laptop.** Replace `<SSH_PORT>` and `<PUBLIC_IP>` with the values from the RunPod UI (Direct TCP). SCP uses the same port as SSH.

```bash
# Bash / WSL (from inferpd repo root):
scp -i ~/.ssh/<your-key> -P <SSH_PORT> -r models/deepseek-v3.2 root@<PUBLIC_IP>:/workspace/deepseek-v3.2
```

```powershell
# Windows PowerShell:
scp -i $env:USERPROFILE\.ssh\<your-key> -P <SSH_PORT> -r models/deepseek-v3.2 root@<PUBLIC_IP>:/workspace/deepseek-v3.2
```

**If SCP is not available** — zip the `models/deepseek-v3.2` folder locally, upload via JupyterLab (port 8888), then SSH in and `unzip deepseek-v3.2.zip` (extracts to `deepseek-v3.2/`).

---

## Phase 3 — First SSH session

**Run on your laptop.** Replace `<SSH_PORT>` and `<PUBLIC_IP>` with your pod’s Direct TCP values.

```bash
ssh root@<PUBLIC_IP> -p <SSH_PORT> -i ~/.ssh/<your-key>
```

### 3.0 Install nano (if not present)

RunPod images often don’t include nano. Install it so you can edit files:

```bash
sudo apt-get update -y
sudo apt-get install -y nano
```

### 3.1 Verify GPUs

```bash
nvidia-smi
# Must show 8× H200. If "No devices found" — reboot or contact provider.
```

### 3.2 Validate environment

```bash
cd /workspace/deepseek-v3.2
source 00-env.sh
python3 system_validation.py
```

Look for: `✅ Hopper` on all 8 GPUs, `✅ NVLink detected`, `✅ Sufficient GPU memory`.  
`NOT INSTALLED` packages are fine at this stage.

### 3.3 Install dependencies

```bash
bash 01-install.sh
```

Expected runtime: **15–30 minutes**. At the end you should see `All OK`.

### 3.4 Re-source env

```bash
source 00-env.sh
source $PYTHON_ENV_PATH/bin/activate
```

---

## Phase 4 — Start the monitoring stack

If Docker is not installed:
```bash
sudo apt-get install -y docker.io docker-compose-plugin
sudo usermod -aG docker $USER && newgrp docker
```

```bash
cd /workspace/deepseek-v3.2
docker compose up -d
docker compose ps   # all 4 containers should show "Up"
```

---

## Phase 5 — Start the vLLM server

Use tmux so the server keeps running if your SSH drops.

```bash
tmux new -s vllm
source /workspace/deepseek-v3.2/00-env.sh
source $PYTHON_ENV_PATH/bin/activate
cd /workspace/deepseek-v3.2
bash 02-start-server.sh 2>&1 | tee $VLLM_LOGS_DIR/server.log
```

**Detach:** `Ctrl+B` then `D`  
**Reattach later:** `tmux attach -t vllm`

### What you will see in the logs

| Log line | What it means |
|---|---|
| `Downloading model-...` | Model downloading (~5–10 min first time) |
| `DeepGemm(fp8_gemm_nt) warmup ...` | GPU kernel warmup (5–10 min, normal) |
| `Capturing CUDA graphs ...` | CUDA graph compilation (3–5 min, first run only) |
| `All ranks ready` | Workers connected, almost up |
| `Uvicorn running on http://0.0.0.0:8000` | ✅ **Server is ready** |

> **First start total: ~5–10 min** (model cached after first run).  
> **Do NOT send requests during warmup.**

---

## Phase 6 — Smoke test

### From inside the Pod

```bash
# New tmux window or SSH session
source /workspace/deepseek-v3.2/00-env.sh

# If VLLM_API_KEY is set in models/deepseek-v3.2/.env, vLLM requires a Bearer token.
curl -H "Authorization: Bearer $VLLM_API_KEY" http://localhost:8000/health
curl -H "Authorization: Bearer $VLLM_API_KEY" http://localhost:8000/v1/models

curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V3.2",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 20,
    "temperature": 0
  }' | python3 -m json.tool
```

### From your laptop (RunPod HTTP proxy)

If you exposed port **8000** in the RunPod UI, use your `<POD_ID>`:
```bash
curl https://<POD_ID>-8000.proxy.runpod.net/health
curl https://<POD_ID>-8000.proxy.runpod.net/v1/models
```

---

## Phase 7 — Set up Grafana dashboard

### Option A — RunPod HTTP proxy

1. Expose port **3000** in RunPod UI.
2. Open `https://<POD_ID>-3000.proxy.runpod.net` in your browser (replace `<POD_ID>` with your pod ID).
3. Login: `admin` / `admin` (change the password).

### Option B — SSH port forwarding (Direct TCP)

Run on your laptop; replace `<PUBLIC_IP>` and `<SSH_PORT>`:

```bash
# Grafana on localhost:3000
ssh -i ~/.ssh/<your-key> -L 3000:localhost:3000 root@<PUBLIC_IP> -p <SSH_PORT> -N &

# vLLM API on localhost:8000
ssh -i ~/.ssh/<your-key> -L 8000:localhost:8000 root@<PUBLIC_IP> -p <SSH_PORT> -N &
```

Then open `http://localhost:3000`.

### 7.3 Add Prometheus data source

1. Left sidebar → **Connections → Data sources → Add new**
2. Select **Prometheus**
3. URL: `http://localhost:9090`
4. Click **Save & test**

### 7.4 Import the vLLM dashboard

1. Left sidebar → **Dashboards → Import**
2. Upload `llm-labs-deepseekv3.2/deepseek grafana.json`
3. Set `$model_name` to `deepseek-ai/DeepSeek-V3.2`
4. Click **Import**

---

## Phase 8 — Run the benchmark

```bash
# From inside the Pod
cd /workspace/deepseek-v3.2
source 00-env.sh && source $PYTHON_ENV_PATH/bin/activate

# benchmark.py automatically uses env VLLM_API_KEY (if set) for Authorization.
python benchmark.py --url http://localhost:8000 --qps 30 --num-requests 300
# If TTFT P50 < 2s and P95 < 3.5s → ramp up:
python benchmark.py --url http://localhost:8000 --qps 60 --num-requests 600
```

```bash
# From your laptop (RunPod proxied URL; replace <POD_ID>):
python benchmark.py --url https://<POD_ID>-8000.proxy.runpod.net --qps 30 --num-requests 300
```

**Target SLAs:**

| Metric | Target |
|---|---|
| TTFT P50 | ≤ 2.0 s |
| TTFT P95 | ≤ 3.5 s |
| E2E P50 | ≤ 11 s |
| E2E P95 | ≤ 25 s |

Results saved to `benchmark_results.json`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| **Error 803** (unsupported display driver / CUDA driver) | Ensure `LD_LIBRARY_PATH` is set in `00-env.sh` and you `source 00-env.sh` before starting vLLM. |
| DeepGEMM MoE crashes on H200 | Set `VLLM_USE_DEEP_GEMM=0` in `00-env.sh` (keeps MQA, disables MoE). |
| `TCPStore broken pipe` spam | A worker OOM'd. Lower `GPU_MEMORY_UTILIZATION` to `0.85` in `00-env.sh`. |
| `CUDA not available` | `nvidia-smi` returns nothing — reboot instance or contact provider. |
| Stuck at GEMM warmup >20 min | Normal on first run. Wait. Do NOT send requests during warmup. |
| OOM during model load | Another process using GPU. Run `sudo fuser -v /dev/nvidia*` and kill it. |
| `Module not found: lmcache` | `source $PYTHON_ENV_PATH/bin/activate` was not run. |
| Grafana shows no data | dcgm-exporter needs NVIDIA Container Toolkit: `sudo apt-get install -y nvidia-container-toolkit && sudo systemctl restart docker && docker compose up -d` |

---

## Stopping everything cleanly

```bash
tmux kill-session -t vllm        # stop vLLM
cd /workspace/deepseek-v3.2
docker compose down               # stop monitoring

# Then release/terminate the instance in the provider dashboard to stop billing.
```

---

## What to run where (RunPod, Direct TCP)

Use `<PUBLIC_IP>` and `<SSH_PORT>` from the RunPod UI (Direct TCP). Ports change after each pod restart.

| Step | Where | Command |
|------|-------|---------|
| 0. Edit secrets | **Laptop** | Edit `models/deepseek-v3.2/.env` (HF_TOKEN), `models/deepseek-v3.2/lmcache_config.yaml` (max_local_cpu_size), `models/deepseek-v3.2/00-env.sh` (model repo if needed) |
| 1. Copy files | **Laptop** | `scp -i ~/.ssh/<your-key> -P <SSH_PORT> -r models/deepseek-v3.2 root@<PUBLIC_IP>:/workspace/deepseek-v3.2` |
| 2. SSH in | **Laptop** | `ssh root@<PUBLIC_IP> -p <SSH_PORT> -i ~/.ssh/<your-key>` |
| 3. Install nano | **Pod (SSH)** | `sudo apt-get update -y && sudo apt-get install -y nano` |
| 4. Validate | **Pod (SSH)** | `cd /workspace/deepseek-v3.2` then `source 00-env.sh && python3 system_validation.py` |
| 5. Install | **Pod (SSH)** | `bash 01-install.sh` (~15–30 min) |
| 6. Monitoring | **Pod (SSH)** | `cd /workspace/deepseek-v3.2 && docker compose up -d` |
| 7. Start vLLM | **Pod (SSH, tmux)** | `tmux new -s vllm` then `source 00-env.sh && source $PYTHON_ENV_PATH/bin/activate && bash 02-start-server.sh` |
| 8. Smoke test | **Pod** or **Laptop** | In Pod: `curl http://localhost:8000/health`. Laptop: `curl https://<POD_ID>-8000.proxy.runpod.net/health` (after exposing 8000) |
| 9. Benchmark | **Pod** or **Laptop** | `python benchmark.py --url http://localhost:8000 --qps 30 --num-requests 300` (Pod) or `--url https://<POD_ID>-8000.proxy.runpod.net` (laptop) |
| 10. Grafana | **Laptop** | `https://<POD_ID>-3000.proxy.runpod.net` or SSH port forward: `ssh -L 3000:localhost:3000 root@<PUBLIC_IP> -p <SSH_PORT> -i ~/.ssh/<your-key> -N` then open `http://localhost:3000` |

**Expose in RunPod UI (optional):** ports **8000** (vLLM) and **3000** (Grafana) if you want proxy URLs.

---

## Quick-reference

| File | When to use |
|---|---|
| `.env` | Fill in once on laptop before upload |
| `00-env.sh` | `source` at the start of every SSH session |
| `01-install.sh` | Once, after first SSH |
| `system_validation.py` | After install, before starting server |
| `02-start-server.sh` | Every time you start the server |
| `docker-compose.yml` | `docker compose up -d` once per instance lifetime |
| `benchmark.py` | After server is up and responding |
