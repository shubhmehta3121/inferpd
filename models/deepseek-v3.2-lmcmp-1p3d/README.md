# DeepSeek-V3.2-NVFP4 — 1P + 3D + LMCache MP

**Deploy this directory:** always `scp` **`models/deepseek-v3.2-lmcmp-1p3d`** to your GPU host(s) and `cd /workspace/deepseek-v3.2-lmcmp-1p3d`. The sibling [`../deepseek-v3.2-lmcmp-1p1d`](../deepseek-v3.2-lmcmp-1p1d) folder is legacy **8-GPU / one-decoder** only — it does not carry the canonical runbook.

Stack: **LMCache multiprocess ZMQ server** + **vLLM `LMCacheMPConnector`** + **disagg proxy** (LMCache `disagg_prefill_mp` pattern). **Not** NIXL.

**Model:** [nvidia/DeepSeek-V3.2-NVFP4](https://huggingface.co/nvidia/DeepSeek-V3.2-NVFP4) (quantized on Hub). `MODEL_PATH` must point at a full local copy (see `scripts/01_download_model.sh`).

**Vast.ai:** **Public IP and SSH port change** when you stop/start or recreate a rental. Copy **Direct ssh** / **Proxy ssh** / `scp` lines from the **live** instance page; the commands below are **examples** for ids **<INSTANCE_A>** / **<INSTANCE_B>**.

---

## Instance A & Instance B — full runbook (two × 8× B200)

**Mapping (your Vast rentals)**

| Label | Vast id | **Public IP** (direct `ssh` / `scp` host) | Role |
|-------|---------|-------------------------------------------|------|
| **Instance A** | <INSTANCE_A> | **`<NODE_A_IP>`** | Prefiller + decoder **0** |
| **Instance B** | <INSTANCE_B> | **`<NODE_B_IP>`** | LMCache MP + proxy + decoders **1–2** |

**SSH / `scp` private key:** **`~/.ssh/<your-key>`** (Git Bash on Windows). Every command below uses **`-i ~/.ssh/<your-key>`**.

### `scp` — copy the **entire** `deepseek-v3.2-lmcmp-1p3d` folder to **both** instances (public IPs)

Run from the **inferpd** repo root. **Ports** (`-P` for `scp`, `-p` for `ssh`) must match the **Direct ssh** line on each instance in the Vast UI — examples use **A = <SSH_PORT_A>**, **B = <SSH_PORT_B>**.

**Instance A — public `<NODE_A_IP>`**

```bash
scp -i ~/.ssh/<your-key> -o IdentitiesOnly=yes -P <SSH_PORT_A> -r \
  models/deepseek-v3.2-lmcmp-1p3d \
  root@<NODE_A_IP>:/workspace/
```

**Instance B — public `<NODE_B_IP>`**

```bash
scp -i ~/.ssh/<your-key> -o IdentitiesOnly=yes -P <SSH_PORT_B> -r \
  models/deepseek-v3.2-lmcmp-1p3d \
  root@<NODE_B_IP>:/workspace/
```

If your shell is already in **`models/`**:

```bash
# Instance A
scp -i ~/.ssh/<your-key> -o IdentitiesOnly=yes -P <SSH_PORT_A> -r \
  deepseek-v3.2-lmcmp-1p3d \
  root@<NODE_A_IP>:/workspace/

# Instance B
scp -i ~/.ssh/<your-key> -o IdentitiesOnly=yes -P <SSH_PORT_B> -r \
  deepseek-v3.2-lmcmp-1p3d \
  root@<NODE_B_IP>:/workspace/
```

You need this **same** directory tree on **A** and on **B** before setup and model download on each machine.

**If `Permission denied (publickey)`:** `chmod 600 ~/.ssh/<your-key>`; add **`<your-key>.pub`** under Vast **Account → SSH keys**; try **proxy** SSH instead of direct; `ssh -o IdentitiesOnly=yes -i ~/.ssh/<your-key> …`; `ssh -v …` to see which key is tried; **restart** the instance from the Vast UI.

**0) From your laptop — open shells**

**Instance A — direct** (public **<NODE_A_IP>**)

```bash
ssh -i ~/.ssh/<your-key> -p <SSH_PORT_A> root@<NODE_A_IP> -L 9000:localhost:9000
```

**Instance A — proxy jump** (if direct fails)

```bash
ssh -i ~/.ssh/<your-key> -p <SSH_PORT_A_PROXY> root@ssh7.vast.ai -L 9000:localhost:9000
```

**Instance B — direct** (public **<NODE_B_IP>** — proxy + LMCache MP + benchmarks)

```bash
ssh -i ~/.ssh/<your-key> -p <SSH_PORT_B> root@<NODE_B_IP> -L 9000:localhost:9000
```

**Instance B — proxy jump**

```bash
ssh -i ~/.ssh/<your-key> -p <SSH_PORT_B_PROXY> root@ssh9.vast.ai -L 9000:localhost:9000
```

Use **`-L 9000:localhost:9000`** to reach the disagg **proxy** from your browser on `http://127.0.0.1:9000`. Add **`-L 8080:localhost:8080`** if you also use Jupyter on 8080.

You need **at least one terminal on A** and **several on B** (or tmux on B) for parallel processes.

**Roles**

| Instance | GPUs | What runs here |
|----------|------|----------------|
| **A** | 8× B200 | **Prefiller** (0–3) + **decoder 0** (4–7), ports **8100** / **8200** |
| **B** | 8× B200 | **LMCache MP** + **proxy** + **decoder 1** (0–3) + **decoder 2** (4–7), ports **6000** / **9000** / **5768** / **8201** / **8202** |

**Env files (IPs already filled for these two rentals)**

- Repo files **[`.env.a`](.env.a)** and **[`.env.b`](.env.b)** match **A** and **B** above (`NODE_*` / `DISAGG_*` use **<NODE_A_IP>** and **<NODE_B_IP>**).
- On **A**: `cp .env.a .env` then set **`HF_TOKEN`** (never commit a real token).
- On **B**: `cp .env.b .env` then set **`HF_TOKEN`**.
- **Decoders:** [`.env.b`](.env.b) defines **`DECODER_PORT_D0` / `DECODER_PORT_D1`** and **`DECODER_GPUS_D0` / `DECODER_GPUS_D1`** (ports **8201** / **8202**, GPU splits **0–3** / **4–7**). On **B**, run **`bash scripts/03_launch_decoder.sh 0`** and **`bash scripts/03_launch_decoder.sh 1`** in separate terminals (after `cp .env.b .env`). On **A**, [`.env.a`](.env.a) keeps **`DECODER_PORT`** + **`DECODER_GPUS`** — run **`bash scripts/03_launch_decoder.sh`** with **no** argument (decoder **0** on **8200**).

If cross-node ZMQ or telemetry fails, open the Vast instance page and swap those addresses for the **private** IPs shown there (keep the same ports).

`NODE_P_PRIVATE_IP` = **A**, `NODE_D_PRIVATE_IP` = **B** (used by `08_preflight_two_node.sh`).

**Firewall**

- **A → B:** **6000** (LMCache ZMQ), **5768** (telemetry)
- **B → A:** **8100** (prefiller), **8200** (decoder on A)

**1) Repo on both** (after **`scp` above** — duplicate also in **§ Copy to server**)

```bash
cd /workspace/deepseek-v3.2-lmcmp-1p3d
```

**2) `.env` on each** — `cp .env.a .env` on **A**, `cp .env.b .env` on **B**, add **`HF_TOKEN`** on both.

**3) One-time setup on A and on B (both instances)**

Scripts load `.env` themselves; you activate the venv for interactive work:

```bash
bash scripts/00_setup.sh
source /workspace/.venv/bin/activate
```

**4) Model weights on A and on B (both instances)**

Either run on **both**:

```bash
bash scripts/01_download_model.sh
```

or download on A and **rsync** `MODEL_PATH` to B so both see the same files.

**5) Start services — Instance B first**

On **B**, one process per terminal, `cd` + `source /workspace/.venv/bin/activate` each time:

| Step | Command |
|------|---------|
| B – LMCache MP | `bash scripts/07_launch_lmcache_mp_server.sh` |
| B – optional checks | `bash scripts/08_preflight_two_node.sh` |
| B – proxy + telemetry | `bash scripts/04_launch_proxy.sh` |
| B – decoder 1 | `bash scripts/03_launch_decoder.sh 0` (uses `DECODER_PORT_D0` / `DECODER_GPUS_D0` from `.env`) |
| B – decoder 2 | `bash scripts/03_launch_decoder.sh 1` (uses `DECODER_PORT_D1` / `DECODER_GPUS_D1`) |

**6) Start services — Instance A**

On **A**, `cd` + `source /workspace/.venv/bin/activate`:

| Step | Command |
|------|---------|
| A – decoder 0 | `bash scripts/03_launch_decoder.sh` (uses `DECODER_PORT` / `DECODER_GPUS` from `.env.a` → `.env`) |
| A – prefiller | `bash scripts/02_launch_prefiller.sh` |

**7) Health check — on Instance B**

```bash
cd /workspace/deepseek-v3.2-lmcmp-1p3d
source /workspace/.venv/bin/activate
bash scripts/05_health_check.sh
```

(`.env.b` sets `DISAGG_HEALTH_*` for these IPs once copied to `.env` on B.)

**8) Benchmarks — where to run `run_benchmark_p50.sh` / `run_benchmark_p95.sh`**

Run them on **Instance B** (same machine as the **proxy** on port **9000**):

```bash
cd /workspace/deepseek-v3.2-lmcmp-1p3d
source /workspace/.venv/bin/activate
cd benchmark
bash run_benchmark_p50.sh
# or
bash run_benchmark_p95.sh
```

The scripts use `--host localhost --port "$PROXY_PORT"` (see `benchmark/*.sh`), so the load generator must see the proxy at **127.0.0.1:9000** on **B**.

**Optional — laptop + SSH tunnel:** use the **Instance B** SSH line above with `-L 9000:localhost:9000`; **easiest load test** is still to run `run_benchmark_*.sh` **on B** in SSH.

**9) Client API**

- From laptop: SSH to **B** with `-L 9000:127.0.0.1:9000`, then `http://127.0.0.1:9000/v1/...`
- Exposing **9000** on the public internet is not recommended without auth.

---

| Component | Version |
|-----------|---------|
| vLLM | 0.18.0 |
| LMCache | 0.4.2 |
| PyTorch | 2.8.0 / CUDA 12.8 (typical RunPod template) |

---

## Prefill vs decode (what each piece does)

| Role | Process | GPUs (two-node layout) | HTTP port | What it does |
|------|---------|------------------------|-----------|----------------|
| **Prefill** | One vLLM **prefiller** | Node **P**: 4 GPUs (TP+EP) | **8100** | Runs the disagg **prefill** pass (`max_tokens=1`): builds prompt KV and **stores** it in **LMCache MP** over ZMQ. Sends **telemetry** when the store finishes so the proxy can continue. |
| **Decode** | Three vLLM **decoder** replicas | Node **P**: 4 GPUs → decoder **0**; Node **D**: 4+4 GPUs → decoders **1** and **2** | **8200**, **8201**, **8202** | Each holds the same weights; **loads** KV from LMCache and runs **full** token generation for client requests. The proxy **round-robins** across the three decoders. |
| **LMCache MP** | `lmcache.v1.multiprocess.server` | CPU/RAM on **Node D** | **6000** (TCP/ZMQ) | Central KV cache service **shared by all four vLLM engines**. Must bind so **Node P can reach it** (default `LMCACHE_MP_BIND_HOST=0.0.0.0`). |
| **Proxy** | `proxy/disagg_proxy_server.py` | CPU on **Node D** | **9000** (+ telemetry **5768**) | Single OpenAI-compatible API: prefill → wait for telemetry → forward streaming/non-streaming to the selected decoder. |

**Traffic paths**

- **HTTP:** Client → proxy → prefiller / decoders (only these use REST).
- **ZMQ:** Prefiller + every decoder ↔ **LMCache MP** (`LMCACHE_MP_CLIENT_HOST` + `LMCACHE_MP_PORT`).
- **Telemetry:** Prefiller → `http://<telemetry-host>:5768/api/v1/telemetry` (usually Node **D**).

---

## Ports (defaults)

| Port | Service |
|------|---------|
| 6000 | LMCache MP (ZMQ) |
| 5768 | Telemetry (thread inside proxy) |
| 9000 | Disagg proxy (client entry) |
| 8100 | Prefiller vLLM |
| 8200–8202 | Decoder vLLM replicas |

---

## Processes vs scripts (what each thing is)

| Script | OS process you get | Blocking? | Typical node |
|--------|-------------------|-----------|--------------|
| `00_setup.sh` | Installs deps / venv (one-time) | Yes, until done | Either |
| `01_download_model.sh` | `huggingface-cli download` → disk | Yes, long | Each node needs weights |
| `02_launch_prefiller.sh` | **vLLM prefiller** (OpenAI API on `PREFILLER_PORT`) | Yes (foreground) | **P** |
| `03_launch_decoder.sh` | **One vLLM decoder** — **A:** no args (`DECODER_PORT` / `DECODER_GPUS` in `.env`). **B:** `0` or `1` → `DECODER_PORT_D0|D1` / `DECODER_GPUS_D0|D1` in `.env.b` | Yes (foreground) | **P** and/or **D** |
| `04_launch_proxy.sh` | **Python** `disagg_proxy_server.py` + **telemetry** thread (HTTP + FastAPI) | Yes (foreground) | **D** |
| `05_health_check.sh` | `curl` + `nvidia-smi` (no daemon) | Yes, then exits | Run where you have HTTP reachability (often **D**) |
| `06_launch_monitoring.sh` | Prometheus + Grafana in background | Starts and returns | Usually **D** |
| `07_launch_lmcache_mp_server.sh` | **LMCache MP** ZMQ server (`python -m lmcache.v1.multiprocess.server`) | Yes (foreground) | **D** |
| `08_preflight_two_node.sh` | `nc` port checks only (no GPUs) | Yes, then exits | Either (needs `NODE_*_PRIVATE_IP` in `.env`) |

**Four long-running “services” for inference:** **07** (LMCache MP) → **04** (proxy + telemetry) → **03** ×3 (decoders) → **02** (prefiller). Each of those stays in its own terminal (or tmux pane) until you stop it with Ctrl+C.

---

## How to start, one step at a time (two-node)

Same flow as **§ Instance A & Instance B** above (**A** = prefiller + dec0, **B** = MP + proxy + dec1–2). Legacy names: **Node P = A**, **Node D = B**.

Use **separate SSH sessions** to **A** and **B**. On each: `cd /workspace/deepseek-v3.2-lmcmp-1p3d` and `source /workspace/.venv/bin/activate` before vLLM/proxy commands.

**Step A — Node D (LMCache must be up before any vLLM)**

1. **Terminal D1:** `bash scripts/07_launch_lmcache_mp_server.sh`  
   Wait until logs show the multiprocess server listening on `LMCACHE_MP_PORT` (default 6000).

2. **Optional — either node:** `bash scripts/08_preflight_two_node.sh`  
   Confirms TCP from P→D (MP + telemetry) and D→P (prefiller + decoder0). Fix cloud firewall if anything fails.

3. **Terminal D2:** `bash scripts/04_launch_proxy.sh`  
   Brings up **9000** (proxy) and **5768** (telemetry). Prefiller on P will POST telemetry here.

**Step B — Node D (two decoder replicas)**

4. **Terminal D3:** `bash scripts/03_launch_decoder.sh 0` (requires `.env.b` copied to `.env` with `DECODER_PORT_D0` / `DECODER_GPUS_D0`)
5. **Terminal D4:** `bash scripts/03_launch_decoder.sh 1` (`DECODER_PORT_D1` / `DECODER_GPUS_D1`)

**Step C — Node P (decoder 0, then prefiller)**

6. **Terminal P1:** `bash scripts/03_launch_decoder.sh` (`.env.a` → `.env`: `DECODER_PORT=8200`, `DECODER_GPUS=4,5,6,7`)
7. **Terminal P2:** `bash scripts/02_launch_prefiller.sh`

**Step D — verify**

8. On a shell that can reach the proxy (usually **D**): `bash scripts/05_health_check.sh`  
   Set `DISAGG_HEALTH_*` in `.env` if prefiller/decoders are on different IPs than `localhost`.

**Step E — optional metrics**

9. `bash scripts/06_launch_monitoring.sh` (often on **D**; use `PROMETHEUS_CONFIG_FILE` for two-node scrape targets).

Clients call **`http://<Node-D>:9000`** (or SSH `-L 9000:127.0.0.1:9000` to Node D).

---

## Two-node layout (2 × 8× B200) — recommended production shape

| Node | Processes | Local GPU map | You run |
|------|-----------|---------------|---------|
| **Node P** (prefill + decode 0) | Prefiller, decoder **0** | `PREFILLER_GPUS=0,1,2,3` · `DECODER_GPUS=4,5,6,7` · `DECODER_PORT=8200` in `.env.a` | `02` + `03` (no args) |
| **Node D** (MP + decode 1–2 + proxy) | LMCache MP, decoder **1**, decoder **2**, proxy | `.env.b`: `DECODER_PORT_D0`/`_D1`, `DECODER_GPUS_D0`/`_D1` → `03_launch_decoder.sh 0` then `… 1` | `07` + `04` + `03` ×2 |

**Firewall (private / VPC IPs)**

- **P → D:** `LMCACHE_MP_PORT` (e.g. 6000), `TELEMETRY_PORT` (5768)
- **D → P:** `PREFILLER_PORT` (8100), `DECODER_PORT` (8200)

### `.env` for Node P / Node D (generic)

For the **570 + 578** pair, use **[`.env.a`](.env.a)** / **[`.env.b`](.env.b)** as in **§ Instance A & Instance B** (copy to `.env` on each machine). Other rentals: start from [`env.instance-a.example`](env.instance-a.example) / [`env.instance-b.example`](env.instance-b.example) and edit IPs.

### Start order (short list)

Same as **§ How to start, one step at a time**: **07** → **08** (optional) → **04** → **03**×2 on **D** → **03** on **P** → **02** on **P**. Prefiller must start **after** LMCache MP and the telemetry listener (proxy) exist. You may start decoders on P before or after decoders on D; all three decoders should be healthy before heavy load.

### SSH from your laptop

See **§ Instance A & Instance B** for **Instance B** direct and proxy commands (use **`-L 9000:localhost:9000`** for the disagg proxy).

---

## Single-node 16 GPU (optional)

Same repo: one machine, `LMCACHE_MP_CLIENT_HOST=tcp://127.0.0.1`, leave `DISAGG_PROXY_*` unset (scripts use localhost + ports 8200–8202). Run `07 → 04 → 03` three times with three different decoder port/GPU pairs (e.g. per-terminal `DECODER_PORT` / `DECODER_GPUS` exports, or **`03_launch_decoder.sh` `0` / `1`** if `.env` defines **`DECODER_PORT_D0|D1`** / **`DECODER_GPUS_D0|D1`** plus a no-arg third with **`DECODER_PORT` / `DECODER_GPUS`**), then `02`.

---

## Copy to server

Same **`scp`** commands as **§ Instance A & Instance B** at the top: key **`~/.ssh/<your-key>`**, **Instance A public `<NODE_A_IP>`**, **Instance B public `<NODE_B_IP>`**, full recursive copy of **`models/deepseek-v3.2-lmcmp-1p3d`** → **`/workspace/`** on **each** host. **`scp` uses `-P`** (capital); **`ssh` uses `-p`** (lowercase). If direct `scp` fails, use the **proxy** host/port from the Vast UI (same as working `ssh`).

On **A**: `cp .env.a .env`. On **B**: `cp .env.b .env`. Set **`HF_TOKEN`** on both. Single-node 16 GPU: `cp .env.example .env` and edit.

---

## One-time setup

```bash
cd /workspace/deepseek-v3.2-lmcmp-1p3d
bash scripts/00_setup.sh
source /workspace/.venv/bin/activate
```

Download quantized weights (requires `HF_TOKEN` in `.env`):

```bash
bash scripts/01_download_model.sh
```

---

## Verify

```bash
source /workspace/.venv/bin/activate
bash scripts/05_health_check.sh
```

---

## Preflight (two-node, before burning GPU time)

`08_preflight_two_node.sh` only runs **`nc`** checks between `NODE_P_PRIVATE_IP` and `NODE_D_PRIVATE_IP` (no vLLM). Run it after both VMs are up and **07** is listening, or earlier to verify firewall rules.

```bash
bash scripts/08_preflight_two_node.sh
```

---

## Benchmark

**Run on Instance B** (proxy must be `localhost:$PROXY_PORT` for the bundled scripts):

```bash
cd /workspace/deepseek-v3.2-lmcmp-1p3d
source /workspace/.venv/bin/activate
cd benchmark
bash run_benchmark_p50.sh
# or: bash run_benchmark_p95.sh
```

Uses `PROXY_PORT` from `.env` on **B**. Sweep: **15, 30, 60** req/s, **300** prompts per step (plus warmup).

---

## Monitoring

```bash
bash scripts/06_launch_monitoring.sh
```

Override config for two-node scraping:

```bash
export PROMETHEUS_CONFIG_FILE=/workspace/deepseek-v3.2-lmcmp-1p3d/configs/prometheus.two-node.example.yml
bash scripts/06_launch_monitoring.sh
```

(`prometheus.two-node.example.yml` is pre-filled for **<NODE_A_IP>** / **<NODE_B_IP>**; edit if your IPs change.)

---

## Troubleshooting

- **`Permission denied (publickey)`** — see the **SSH key** paragraph in **§ Instance A & Instance B** (chmod, Vast keys, proxy SSH, `IdentitiesOnly=yes`, `ssh -v`, restart instance, new key).
- **`scp` fails** — fix `ssh` first; match **`-P`** and host to the live Vast page.
- **`ss -tlnp | grep 6000`** — LMCache MP must listen on `0.0.0.0` (or the IP Node P uses) **before** vLLM.
- **`ss -tlnp | grep 5768`** — telemetry thread (started with proxy on Node D).
- **`DECODER_PORT` is unset** / **`03_launch_decoder.sh` exits on B** — run **`cp .env.b .env`** on **B** so **`DECODER_PORT_D0` / `DECODER_PORT_D1`** and **`DECODER_GPUS_D0` / `DECODER_GPUS_D1`** are loaded; use **`bash scripts/03_launch_decoder.sh 0`** and **`… 1`**, not bare **`03_launch_decoder.sh`** without **`DECODER_PORT`** in `.env`.
- If **`BLOCK_SIZE=64`** OOMs, set **`BLOCK_SIZE=32`** in `.env` and restart prefiller + decoders.
- Raise **`LMCACHE_MP_MAX_WORKERS`** if MP CPU saturates (four vLLM engines share one MP server).

---

## Relation to other folders

| Folder | Layout |
|--------|--------|
| `deepseek-v3.2-lmcmp-1p1d` | Legacy 8-GPU 1P1D scripts only — **runbook is here (1p3d)** |
| **This folder** | Canonical deploy + 1 prefiller + **3** decoders (16 GPU), cross-node ready |
| `deepseek-v2-lite-2p2d` | 2P2D + **NIXL** — different stack |

`docker-compose.yml` is **legacy** (NIXL-oriented); use the shell scripts.
