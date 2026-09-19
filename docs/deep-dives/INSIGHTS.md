# Insights: what to do again

Notes from learning the stack from zero and then burning GPU time on it. Not a vendor pitch. Not a client case study.

**Related:** [BUILD.md](./BUILD.md) · [COSTS.md](./COSTS.md) · [RESOURCES.md](./RESOURCES.md)

---

## Three rules

1. **Small first.** Prove the architecture on DeepSeek V2-Lite and 2× A100 before V3.2 on B200. Same vLLM APIs, same disagg patterns, a fraction of the cost.
2. **Move fast.** Experiments per week beat a perfect plan on a $60/hr cluster. A failed A100 hour teaches more than a stalled H200 week.
3. **Shrink it.** When stuck, drop to one request, one GPU role, one connector path. Fix that. Add QPS and GPUs back one layer at a time.

In one line: prove it on the smallest model and cheapest hardware, iterate quickly, scale only what already works.

---

## What actually worked

### Start with the smaller model

Do not begin on the biggest checkpoint. V2-Lite on 2× A100 (~$3/hr) is where P/D, NIXL, LMCache MP, and the shared L1 were learned. Only after that stack cleared a TTFT bar did the runbook move to V3.2 NVFP4 on B200.

The early H200 bring-up was still useful. It answered “does the large model even load?” It was the wrong classroom for “why is the decoder OOM-ing?”

### Speed of execution is the scarce resource

The bottleneck was cycle time, not a missing paper. Dual-node 8× H200 was architecturally correct and economically wrong for daily learning. Breakthroughs happened on A100 rentals where a bad deploy cost an hour.

Optimize for experiments per week, not perfection per experiment.

### When stuck, make the problem tiny

Do not debug at production scale. Isolate:

- one request
- one role (prefill **or** decode)
- one KV path (NIXL **or** MP, not both at once)
- then add load

That is how the LMCache eviction OOM got isolated. That is why NIXL 1P1D existed as a reference **before** shared L1.

### Abandon paths that do not converge

Classic `LMCacheConnectorV1` P/D looked fine in smoke tests and OOM’d under real concurrency. Dual-node BF16 was too expensive to iterate. Both were stopped.

Sunk cost on a rental hour is smaller than sunk cost on a week of confusion.

### Match architecture to the traffic

~80% prefix cache hits mean cross-request KV reuse is not optional.

- **NIXL** is a good **per-request** GPU handoff.
- **Prefix-heavy chat** needed a **shared L1** (central ZMQ + `LMCacheMPConnector`).
- Extra **decode** capacity (1P2D) mattered once prefixes were cached.

The workload picked the topology. The topology did not pick the workload.

### Provider limits are not architecture failures

1P3D across two 8× B200 rentals was blocked by missing private networking, not by the runbook. Capacity was also missing on the other provider that week.

Document the constraint. Ship what you measured. Do not rewrite the stack to apologize for someone else’s VPC.

### TTFT and end-to-end are different levers

Shared L1 drops TTFT because prefill is mostly skipped. End-to-end stays decode-bound. If the product cares about total completion time more than first token, this architecture is the wrong lever. Be explicit about which number you are optimizing.

### Learn in public-ish notes, deploy from folders

The concept curriculum (attention → KV cache → PagedAttention → MoE/MLA → P/D) happened in chats and papers **before** most of the GPU bill. The production pieces (NIXL ports, MP ZMQ, telemetry proxy) only made sense **after** something failed on a box.

Keep a topic roadmap. Rent cheap GPUs as soon as the vocabulary exists. Do not wait to “finish theory.”

---

## Failure → what you actually learn

| If this happens | You are really learning |
|-----------------|-------------------------|
| OOM on long prompts | `gpu_memory_utilization`, block size, KV math |
| Prefix cache barely hits | Cache keys, prompt structure, whether you even have a shared store |
| Scaling is sublinear | NCCL, comms, EP vs TP |
| Disagg adds latency | Proxy wait, telemetry, transfer path |
| Cost explodes | Smaller model, quantization, spot vs on-demand, stop using 16× H200 as a classroom |
| Cross-node never comes up | Placement and networking, not your Python |

---

## Learning method (from zero)

No prior transformer, KV cache, or GPU-serving background. The path that worked:

1. Write a 12-topic roadmap (see [RESOURCES.md](./RESOURCES.md)).
2. Get orientation from AI chat (DeepSeek, then ChatGPT / Claude for a second angle).
3. Switch to a paper or a video when the chat explanation is mush.
4. Rent 2× A100 and break the real stack.

Foundations first (RNN → attention → decoder-only → KV cache) were not optional. Skipping them made every vLLM error look like black magic. Staying in foundations after the vocabulary existed was also a stall: the eviction bug only appears under load.

---

## What I would not do again

- Start daily iteration on the production-class model and full-width BF16 P/D.
- Treat a passing 1-request smoke test as “the connector works.”
- Optimize NIXL scale-out hoping it will magically share prefixes across chats.
- Blame the design when two spot machines cannot see each other on a private fabric.
- Hide the bad sweeps. Phase 4’s 62.9s TTFT is part of why phase 7 is believable.
