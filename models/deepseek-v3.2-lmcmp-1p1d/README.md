# DeepSeek-V3.2-NVFP4 — 1P + 1D (legacy 8-GPU folder)

This path is the **single-machine, one-decoder** layout: one prefiller + one decoder (`NUM_DECODERS=1`, TP=4 each on **8 GPUs**).

**Do not use this README for Vast deployment.** All **SSH, `scp`, two-node wiring, `.env.a` / `.env.b`, benchmarks, monitoring, and troubleshooting** live in:

**[`../deepseek-v3.2-lmcmp-1p3d/README.md`](../deepseek-v3.2-lmcmp-1p3d/README.md)**

On servers, copy **`models/deepseek-v3.2-lmcmp-1p3d`** to `/workspace` and follow that file. The 1p3d tree supports **1P+3D** (and optional single-node 16 GPU); use `.env.example` or the instance env templates there.

Stack (same as 1p3d): LMCache multiprocess ZMQ + vLLM `LMCacheMPConnector` + disagg proxy — **not** NIXL. Typical ports: MP **6000**, telemetry **5768**, proxy **9000**, prefiller **8100**, decoder **8200**.

| Other folder | Role |
|--------------|------|
| **`deepseek-v3.2-lmcmp-1p3d`** | Canonical runbook + cross-node 1P+3D |
| `deepseek-v2-lite-lmcache-mp` | Smaller model 1P1D baseline |
| `deepseek-v2-lite-2p2d` | 2P2D + NIXL — different stack |
