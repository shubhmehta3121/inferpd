# Costs: GPU spend and why most of it was wasted on the wrong size

Spot rentals on **RunPod** and **Vast.ai**. Invoice totals are the ground truth. Per-phase rows below are **models**, not receipts.

**Related:** [BUILD.md](./BUILD.md) · [INSIGHTS.md](./INSIGHTS.md)

---

## Totals

| | USD |
|---|-----|
| **Total GPU spend** | ~$1,205 |
| RunPod | ~$535 |
| Vast.ai | ~$670 |

\* $600 of this was reimbursed. The rest was out of pocket.

Rates that shaped decisions:

| Hardware | Typical spot rate | What it was good for |
|----------|-------------------|----------------------|
| 2× A100 SXM | ~$3/hr | Daily iteration, NIXL and LMCache MP on V2-Lite |
| 3× A100 | ~$4.50/hr | Headline 1P2D sweep |
| 8× H200 | ~$30/hr | First V3.2 BF16 bring-up |
| Dual 8× H200 | ~$60/hr | Correct full-width P/D topology, too expensive to debug on |
| 8× B200 | listing-dependent, tens of $/hr | NVFP4 large-model path |

---

## The actual lesson

Roughly **80% of the money** went to the two big-GPU phases (H200 BF16, B200 NVFP4).

Roughly **80% of the learning** happened on the cheap A100 rigs.

A dual-node 8× H200 cluster is the right shape for production-class BF16 P/D and the wrong shape for “I do not know how LMCache eviction works yet.” Failed deploys on A100 cost an hour. Failed deploys on H200 cost a day of budget and a night of confusion.

That is why the sequence in [BUILD.md](./BUILD.md) looks “backwards” if you only care about the final model: start huge, get punished, drop to V2-Lite, then bring the working pattern back to V3.2 NVFP4.

---

## Per-phase estimates (upper bound)

GPU-hours × list rate, rounded. Pods were often killed early. Spot ran under list. **Sum of this table is higher than $1,205.** Use it only to see which phase was expensive and why.

| Phase | What | GPUs | Est. hours | Est. $ | Outcome |
|------:|------|------|------------|--------|---------|
| 1 | V3.2 BF16, TP=8 | 8× H200 | 12 | ~$360 | Model ran, no P/D on one node |
| 2 | Dual-node BF16 P/D | 16× H200 | 3 | ~$180 | Correct, too expensive to iterate |
| 3 | V2-Lite classic LMCache P/D | 2× A100 | 20 | ~$60 | Decode OOM |
| 4 | V2-Lite NIXL 1P1D | 2× A100 | 35 | ~$105 | First stable P/D |
| 5 | V2-Lite NIXL 2P2D | 2–4× A100 | 25 | ~$150 | Throughput up, no shared L1 |
| 6 | V2-Lite LMCache MP 1P1D | 2× A100 | 18 | ~$54 | Shared store works |
| 7 | V2-Lite LMCache MP 1P2D | 3× A100 | 18 | ~$81 | Headline result |
| 8 | V3.2 NVFP4 1P1D | 8× B200 | 12 | ~$130 | Large-model path works |
| 9 | V3.2 NVFP4 1P3D | 16× B200 | 15 | billed as part of Vast total | Runbook ready, networking blocked |

---

## Why cost was a design input

The R&D target was self-hosting DeepSeek-class inference without blowing past roughly **2× official API pricing**, while improving TTFT on prefix-heavy streaming chat.

That constraint, not taste, drove the pivots:

1. Dual-node BF16 H200 is the “real” topology and a terrible classroom.
2. V2-Lite on A100 is where connectors, proxies, and eviction bugs get isolated.
3. NVFP4 on B200 is how a V3.2-class model fits on one 8-GPU node with KV headroom.

Spot GPUs were the right market for this: cheap enough to retry, unreliable enough that you should not bet a two-node private fabric on two random listings. Phase 9 is that lesson in one line.
