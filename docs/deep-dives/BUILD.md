# Build: how the stack was assembled

Nine rented-GPU experiments from a first vLLM bring-up to a working disaggregated DeepSeek path. This is the engineering record. Deploy steps live in each model folder, not here.

**Related:** [RESULTS.md](./RESULTS.md) · [INSIGHTS.md](./INSIGHTS.md) · [site diagram](https://shubhmehta3121.github.io/inferpd/#solution)

---

## Shorthand

| Term | Meaning |
|------|---------|
| **Prefill** | Process the full prompt in parallel. Compute-bound. Builds the KV cache. |
| **Decode** | Generate tokens one at a time. Memory-bandwidth-bound. Reads the KV cache. |
| **1P2D** | 1 GPU (or GPU group) does prefill, 2 do decode. `1P1D`, `2P2D`, `1P3D` follow the same pattern. |
| **KV cache** | Stored keys and values from attention, reused so decode does not recompute the prompt. |
| **Prefix hit** | A new request shares a prompt prefix (system prompt, history) already in cache, so most of prefill can be skipped. |
| **NIXL** | NVIDIA Inference Transfer Library. GPU-to-GPU KV handoff for one request. |
| **LMCache MP** | LMCache multiprocess server. One shared KV store in host RAM. Every engine talks to it over ZMQ. |
| **Disagg proxy** | OpenAI-compatible front door. Sends a request to prefill, waits until KV is stored, then routes to a decoder. |

---

## Workload this was built for

Production-like **streaming chat**, not a synthetic short-prompt bench:

| Dimension | Ballpark |
|-----------|----------|
| Input length | P50 ~7.9k tokens, P95 ~15.8k |
| Output length | P50 ~252 tokens, P95 ~561 |
| Load | Peak ~60 QPS |
| Prefix reuse | ~80% cache hit rate |
| Latency that mattered | Time to first token (TTFT). Production ballpark ~2s P50 / ~3.5s P95. End-to-end was secondary. |
| Cost posture | Stay near official DeepSeek API pricing (about 2× as a ceiling), not “unlimited GPU.” |

That ~80% prefix hit rate is the design constraint. A handoff that only moves KV **for the current request** does not reuse work across chats that share a system prompt. A **shared store** does.

---

## Three ways KV moved

Public LMCache examples often show NIXL or a simple producer/consumer. This lab compared three paths on the same class of traffic.

### A. Classic LMCache P/D (abandoned)

- **Connector:** `LMCacheConnectorV1` (prefiller pushes KV into the decoder per request).
- **What broke:** Decode-side eviction in the PD backend. Blocks were not released. Under load the decoder filled VRAM until OOM, even though small smoke tests looked fine.
- **Outcome:** Stopped as the primary path.

### B. NIXL P/D (stable, not shared)

- **Connector:** `NixlConnector` plus LMCache where needed.
- **How it moves KV:** Prefiller is sender, decoder is receiver. On one node this uses NVLink / `cuda_ipc`. A telemetry port tells the proxy when the handoff is safe.
- **What worked:** Reliable 1P1D on 2× A100, then 2P2D scale-out.
- **What it is not:** A cross-request L1. Each request still copies KV privately. Repeat prefixes redo work.

### C. Shared L1 via LMCache MP (best fit)

- **Connector:** `LMCacheMPConnector` on every engine, plus a central MP server.
- **How it moves KV:**
  1. Prefiller stores prefix KV in the MP server (or hits an existing prefix).
  2. Prefiller signals telemetry so the proxy does not start decode until the store is visible.
  3. Decoders retrieve from the **same** server. LRU eviction keeps RAM bounded.
- **Why it matched the workload:** ~80% of requests can skip full prefill if the prefix already lives in one store that every decoder can read.

```
Client  →  disagg proxy :9000
              │
              ├─ prefill  →  prefiller :8100  ──writes──►
              └─ decode   →  decoders :8200+  ◄──reads───  LMCache MP :6000  (shared L1 in host RAM)
```

---

## Default port map (LMCache MP stacks)

| Port | Service |
|------|---------|
| 6000 | LMCache MP (ZMQ) |
| 5768 | Telemetry (proxy waits on KV commit) |
| 9000 | Disagg proxy |
| 8100 | Prefiller |
| 8200+ | Decoder(s) |

NIXL-centric V2-Lite stacks use telemetry **7500** and NIXL ports **7300 / 7400**. See `models/deepseek-v2-lite/README.md`.

Typical pins on the MP path: **vLLM 0.18.0**, **LMCache 0.4.2**, PyTorch **2.8 / CUDA 12.8**.

---

## Nine phases (order they were actually run)

### 1. DeepSeek V3.2 BF16, 8× H200

- **Folder:** `models/deepseek-v3.2/`
- **Setup:** TP=8, full-width BF16, LMCache on one node. No GPUs left to split prefill and decode.
- **Lesson:** The production-class model runs. You cannot iterate a P/D design on this footprint.

### 2. Dual-node BF16 P/D (theory)

- **Folder:** `models/deepseek-v3.2/v2/`
- **Setup:** Two nodes × 8× H200, producer/consumer roles, NIXL between nodes. ~$60/hr class.
- **Lesson:** Architecturally correct for true P/D at full width. Economically wrong for daily learning.

### 3. V2-Lite, classic LMCache P/D

- **Folder:** `models/archive/deepseek-v2-lite_lmcache/`
- **Setup:** Smaller model so failures are cheap. Stock LMCache producer/consumer, no MP server.
- **Lesson:** Decode eviction OOM under concurrency. Do not deploy from `archive/`.

### 4. V2-Lite, NIXL 1P1D

- **Folder:** `models/deepseek-v2-lite/`
- **Setup:** 2× A100 80GB SXM, NVLink. 1 prefiller + 1 decoder. Proxy boot order: proxy → decoder → prefiller.
- **Lesson:** First **stable** disaggregated reference. Proved P/D on hardware you can afford to break.

### 5. V2-Lite, NIXL 2P2D

- **Folder:** `models/deepseek-v2-lite-2p2d/`
- **Setup:** Two prefillers + two decoders.
- **Lesson:** More replicas raise throughput. Still no shared prefix cache across requests.

### 6. V2-Lite, LMCache MP 1P1D

- **Folder:** `models/deepseek-v2-lite-lmcache-mp/`
- **Setup:** Same 2× A100. KV no longer rides NIXL. One ZMQ store, `LMCacheMPConnector` on both engines.
- **Lesson:** Shared L1 is the mental model that matches prefix-heavy chat.

### 7. V2-Lite, LMCache MP 1P2D (headline)

- **Folder:** `models/deepseek-v2-lite-lmcmp-1p2d/`
- **Setup:** 3× A100. One prefiller, two decoders, one MP server. Proxy round-robins decode.
- **Hypothesis:** At ~80% prefix hits, decode is hotter than greenfield prefill, so extra decode GPUs help more than extra prefiller GPUs.
- **Lesson:** Validated. This is the best V2-Lite result. Numbers in [RESULTS.md](./RESULTS.md).

### 8. V3.2 NVFP4, LMCache MP 1P1D, 8× B200

- **Folder:** `models/deepseek-v3.2-lmcmp-1p1d/`
- **Setup:** NVIDIA [DeepSeek-V3.2-NVFP4](https://huggingface.co/nvidia/DeepSeek-V3.2-NVFP4). About half the footprint of naive BF16, so 4 GPUs prefill + 4 GPUs decode (TP=4 each) fit on one 8-GPU node. Same MP + proxy port story.
- **Lesson:** The **big-model path works end to end**. It did not reproduce the V2-Lite latency curve. Sub-second TTFT only held to low QPS.

### 9. V3.2 NVFP4, 1P3D cross-node

- **Folder:** `models/deepseek-v3.2-lmcmp-1p3d/`
- **Setup:** One prefiller + three decoders (16 GPUs). Two separate 8× B200 rentals. Runbook, env templates, and preflight scripts are complete.
- **Blocker:** No private network between those rentals. ZMQ over public TCP is possible in theory and a mess in practice (latency, firewalls, ops). Inventory on the other provider was also empty that week.
- **Lesson:** Software was ready. Placement was not. That is not an architecture failure.

---

## What worked vs what did not

| Item | Result |
|------|--------|
| Classic LMCache P/D (pre-MP) | Failed: decode-side eviction → OOM |
| NIXL 1P1D on V2-Lite | Worked: reference P/D |
| NIXL 2P2D on V2-Lite | Worked: more throughput, still no shared L1 |
| LMCache MP 1P1D on V2-Lite | Worked: shared prefix KV |
| LMCache MP 1P2D on V2-Lite | Worked well: best match for prefix-heavy traffic |
| BF16 V3.2 on 8× H200 (single node) | Ran, no real P/D, too rigid to iterate |
| NVFP4 V3.2 + MP 1P1D on 8× B200 | Worked: practical large-model slice |
| NVFP4 V3.2 1P3D on two 8× B200 | Runbook ready, blocked on provider networking |
| Full BF16 V3.2 with 8+8 top-end GPUs | Coherent on paper, later-phase economics |

---

## Trade-offs

| Choice | Upside | Downside |
|--------|--------|----------|
| BF16 full V3.2 | Closest to “the real model” | Two full 8-GPU nodes for P/D. Slow, expensive iteration. |
| NVFP4 V3.2 | Fits 4+4 on one 8× B200 | Quantization vs BF16 |
| NIXL P/D | Fast on-node NVLink handoff | Moving parts. Not a shared cross-request cache. |
| LMCache MP | Shared L1. Fits prefix-heavy traffic. | RAM bound, eviction, MP CPU. Cross-node is WAN-sensitive. |
| 1P2D / 1P3D | Matches cache-hit economics | More GPUs, harder to place on the spot market |
| Cloud spot | Cheap enough to learn | No guaranteed private fabric between two random rentals |

---

## Canonical folders

| Use this when | Folder |
|---------------|--------|
| Two-node 1P3D runbook (V3.2 NVFP4) | `models/deepseek-v3.2-lmcmp-1p3d/` |
| Single-node 8-GPU 1P1D (V3.2 NVFP4) | `models/deepseek-v3.2-lmcmp-1p1d/` |
| BF16 V3.2 single / dual-node reference | `models/deepseek-v3.2/` (+ `v2/`) |
| MP baseline, small model | `models/deepseek-v2-lite-lmcache-mp/` |
| MP 1P2D (headline V2-Lite) | `models/deepseek-v2-lite-lmcmp-1p2d/` |
| NIXL 2P2D | `models/deepseek-v2-lite-2p2d/` |
| NIXL 1P1D | `models/deepseek-v2-lite/` |

Further reading in-repo: per-folder runbooks under `models/`, plus [`docs/learning/LEARNING_ROADMAP.md`](../learning/LEARNING_ROADMAP.md).
