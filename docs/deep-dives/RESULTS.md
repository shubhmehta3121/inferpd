# Results: asked vs delivered

Sweeps used production-like load: ~8k-token inputs, ~80% shared prefixes, streaming.

**Compare within a model, not across them.** Phases 4–7 are DeepSeek V2-Lite on A100s. Phase 8 is DeepSeek V3.2 NVFP4 on B200s. A good V2-Lite number is not a V3.2 claim.

**Related:** [BUILD.md](./BUILD.md) · [site chart](https://shubhmehta3121.github.io/inferpd/#results)

---

## How to read a row

- **Asked QPS** is the load the benchmark requested.
- **Delivered** is what the stack actually sustained.
- **TTFT P50 / P95** is time to first token (median / tail). Production ballpark: ~2s P50, ~3.5s P95.
- A phase marked **success** means the topology ran and produced a clean sweep. It does **not** mean it beat production latency.

Phase 4 is the clearest example of that distinction: a working P/D reference whose TTFT collapses past a few req/s. Useful as a starting point. Not an answer.

---

## Headline stack (phase 7)

**DeepSeek V2-Lite · LMCache MP · 1 prefiller + 2 decoders · 3× A100**

Shared central KV store. Extra decode GPUs because prefix-heavy traffic is decode-hot once the prefix is cached.

| Asked QPS | Delivered | TTFT P50 | TTFT P95 | vs ~2s production P50 |
|-----------|-----------|----------|----------|------------------------|
| 10 | 8.47 req/s | **0.52s** | 0.93s | ~3.9× faster |
| 60 | ~11.21 req/s | 16.8s | — | Saturated. Ceiling is ~11 req/s, not 60. |

\* One 1P2D box. The stall is a GPU ceiling on those 3 cards. More QPS is more replicas at about 1:2 to 1:3 prefill:decode (the NIXL runbooks list 1:3 as the next step), e.g. 3 prefiller + 6 decoder GPUs, with throughput scaling roughly with replica count.

What this proves: a shared L1 is the **right shape** for prefix-heavy chat, validated cheaply end to end.

What this does not prove: production capacity on a V3.2-class model, or 60 QPS.

TTFT dropped because most requests skip full prefill. End-to-end stay decode-bound (about 13.3s P50 at the headline point vs an ~11s production ballpark). That was an accepted trade: streaming chat cared about first token more than total completion time.

---

## Snapshot at 10 asked QPS

Where a phase had no 10 QPS point, the nearest measured step is used and noted in [BUILD.md](./BUILD.md).

| Phase | Stack | TTFT P50 | vs ~2s prod |
|------:|-------|----------|-------------|
| 4 | V2-Lite · NIXL · 1P1D | 62.9s | ~31× slower |
| 5 | V2-Lite · NIXL · 2P2D | 39.6s | ~20× slower |
| 6 | V2-Lite · LMCache MP · 1P1D | 14.3s | ~7× slower |
| 7 | V2-Lite · LMCache MP · 1P2D | **0.52s** | ~4× faster |
| 8 | V3.2 NVFP4 · LMCache MP · 1P1D | 43.5s | ~22× slower |
| 9 | V3.2 NVFP4 · LMCache MP · 1P3D | — | No sweep. Provider networking. |

The jump from phase 6 to 7 is the important one on V2-Lite: same shared store, extra decode GPUs, TTFT falls onto the right side of the production line.

---

## Production-class model (phase 8)

Same architecture on **V3.2 NVFP4**, **8× B200**, 4 GPUs prefill + 4 GPUs decode.

- The path runs end to end (proxy, MP store, prefiller, decoder).
- Sub-second TTFT held only to about **2 req/s**, then degraded sharply past about **5**.
- At 10 asked QPS, TTFT P50 was **43.5s**. That is not the V2-Lite curve.

Reading this honestly: the lab showed the wiring transfers to the large model. It did not show that 3× A100 V2-Lite latency appears automatically on Blackwell + NVFP4. Closing that gap wanted 1P3D scale-out, which never got a private network.

---

## Data

- Machine-readable sweeps: [`site/data/results.json`](../../site/data/results.json)
- Per-model HTML reports: `models/*/benchmark_results/`
- Rebuild JSON: `python scripts/build_results_json.py`

These are single-operator measurements on rented GPUs, not vendor-certified results.
