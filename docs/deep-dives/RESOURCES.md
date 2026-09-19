# Resources: how the stack was learned

Self-taught path used on this lab. AI chat for orientation, papers and videos when a second angle was needed, cheap GPUs as soon as the words meant something.

Private tutoring transcripts are **not** in this repo. This file is the public curriculum and bibliography.

**Related:** [INSIGHTS.md](./INSIGHTS.md) · [BUILD.md](./BUILD.md) · longer extract: [`docs/learning/LEARNING_ROADMAP.md`](../learning/LEARNING_ROADMAP.md)

---

## Method

1. Build a topic list **in order** (the 12 steps below). Do not shuffle “MoE” in front of “KV cache.”
2. Ask an AI to walk the current step with a formula and a tiny example.
3. If the explanation is still mush, open a paper or a video from the matching section.
4. Rent 2× A100 and hit the real error (OOM, port conflict, hang). That is the next lesson.

Tools used for step 2: [DeepSeek Chat](https://chat.deepseek.com), [ChatGPT](https://chatgpt.com), [Claude](https://claude.ai). None of them replace `models/*/README.md` when you are actually deploying.

---

## Curriculum (order matters)

```
Tier 0  Foundations     RNN → attention → masking → GPT decoder
Tier 1  Inference core  KV cache → prefill vs decode → prefix caching
Tier 2  vLLM memory     PagedAttention → block tables → copy-on-write
Tier 3  DeepSeek model  MoE → MLA → DSA → NVFP4
Tier 4  Scale-out       TP → DP → EP → disaggregated P/D
Tier 5  This lab        NIXL → LMCache MP → proxy → benchmarks
```

### Tier 0 — Foundations

You cannot debug a decoder OOM if “query / key / value” is still a slogan.

| Topic | Intuition |
|-------|-----------|
| RNN | Sequential hidden state, vanishing gradients: why attention exists |
| Self-attention | `Attention(Q,K,V) = softmax(QKᵀ / √d_k) V`. Q/K/V come from the **current sequence**, not a dictionary. |
| Causal mask | Position *i* attends only to ≤ *i* |
| Decoder-only GPT | Stack of masked attention + FFN. Generate one token, append, repeat. |

**Read / watch**

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) (Jay Alammar)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- [The Transformer Family](https://lilianweng.github.io/posts/2020-04-07-the-transformer-family/) (Lil’Log)
- [3Blue1Brown: Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M)
- [3Blue1Brown: Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc)
- [IBM Technology: What are Transformers?](https://www.youtube.com/watch?v=ZXiruGOCn9s)
- [StatQuest: RNNs, clearly explained](https://www.youtube.com/watch?v=AsNTP8Kwu80)
- [CodeBasics: Transformers explained](https://www.youtube.com/watch?v=ZhAz268Hdpw)
- [CodeBasics: Self attention](https://www.youtube.com/watch?v=SO2-3YS6e-k)
- [ByteByteGo: Transformers step-by-step](https://www.youtube.com/watch?v=avjX3QrYkls)
- [Discover AI: Q, K and V](https://www.youtube.com/watch?v=PFczJ6NR5rY)

### Tier 1 — Inference core (12 steps)

From the original learning-path plan. Keep this order.

| Step | Topic | Why it comes next |
|-----:|-------|-------------------|
| 0 | Transformer refresh | Prerequisite |
| 1 | **KV cache** | Decode only computes the new token’s K/V |
| 2 | **Prefill vs decode** | Explains TTFT vs throughput |
| 3 | **Prefix caching** | Shared system prompts. The ~80% hit rate. |
| 4 | **MLA** | DeepSeek compresses K/V |
| 5 | **MoE** | Huge total params, few active per token |
| 6 | FFN / SwiGLU | What MoE is a pile of |
| 7 | Sparse attention / DSA | Top-k tokens vs full O(L²) |
| 8 | Parallelism overview | TP, DP, PP, EP |
| 9 | Tensor parallelism | Split weights to fit the model |
| 10 | Data parallelism | Split requests |
| 11 | Expert parallelism | Split MoE experts |
| 12 | **Disaggregated P/D** | Separate GPU pools, transfer KV |

Bonus: continuous batching (Orca), PagedAttention, cost vs API pricing.

**Read**

- [How KV cache size is calculated](https://finbarr.io/how-is-kv-cache-size-calculated/) (Finbarr Timbers)
- [Hugging Face: generation with cache](https://huggingface.co/docs/transformers/main/en/cache)
- [vLLM: understanding LLM inference](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [Databricks: LLM inference performance](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)

### Tier 2 — vLLM memory

- GPU memory is one address space. “Layers” are software.
- **Block table** maps logical token chunks → physical blocks (OS virtual memory, but for KV).
- Shared prefix → same physical blocks. Sequences diverge → **copy-on-write**.
- Prefill and decode are **phases**, not two memory pools.

**Read / watch**

- [vLLM paper (PagedAttention)](https://arxiv.org/abs/2309.06180)
- [Anyscale: Fast LLM serving with vLLM and PagedAttention](https://www.youtube.com/watch?v=5ZlavKF_98U)
- [vLLM: automatic prefix caching](https://docs.vllm.ai/en/latest/design/prefix_caching.html)
- [SGLang](https://github.com/sgl-project/sglang) (RadixAttention as an alternate mental model)

### Tier 3 — DeepSeek-specific

| Idea | Why it showed up in this lab |
|------|------------------------------|
| MoE | ~685B-class total, ~37B active per token |
| MLA | Compress KV so long context does not explode memory |
| DSA | Sparse attention path on V3.2 |
| NVFP4 | Quantized V3.2 that fits 8× B200 with KV headroom |
| TP=8 | Stable first bring-up on H200/B200 |

**Read**

- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) (MLA)
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437) (MoE)
- [DeepSeek-V3.2-Exp model card](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp)
- [NVIDIA DeepSeek-V3.2-NVFP4](https://huggingface.co/nvidia/DeepSeek-V3.2-NVFP4)

### Tier 4 — Scale and disaggregation

- Prefill nodes: compute-bound, build KV.
- Decode nodes: memory-bandwidth-bound, stream tokens.
- KV has to **move** (NIXL GPU↔GPU, or a central MP store).
- A **proxy** must wait until KV is committed before decode, or you race.

**Read**

- [vLLM: disaggregated prefilling](https://docs.vllm.ai/en/latest/design/prefill_decode.html)
- [MoonCake](https://arxiv.org/abs/2407.00079)
- [Orca (continuous batching)](https://www.usenix.org/conference/osdi22/presentation/yu)
- [NIXL](https://github.com/NVIDIA/nixl)
- [LMCache](https://github.com/LMCache/LMCache)
- [Megatron-LM](https://arxiv.org/abs/1909.08053) (tensor parallelism)
- [GShard](https://arxiv.org/abs/2006.16668) (expert parallelism)

### Tier 5 — What this repo added on GPUs

Not in the early curriculum. Appeared after something broke.

| Path | Connector | Result |
|------|-----------|--------|
| Classic LMCache P/D | `LMCacheConnectorV1` | Failed: decode eviction OOM |
| NIXL P/D | `NixlConnector` | Worked: no shared cross-request L1 |
| LMCache MP | `LMCacheMPConnector` + ZMQ :6000 | Worked: best for ~80% prefix hits |

**In this repo**

- Per-topology runbooks under `models/`
- Study notes: [`docs/learning/LEARNING_ROADMAP.md`](../learning/LEARNING_ROADMAP.md)

---

## Formula cheat sheet

```
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) · V

q_i = x_i · W_Q
k_i = x_i · W_K
v_i = x_i · W_V

Prefill:  all prompt tokens in parallel → fill KV
Decode:   one new token per step → append KV

KV memory ≈ 2 × hidden × bytes × seq_len × layers   (MLA reduces this)

PagedAttention: logical chunks --block_table--> physical GPU blocks
Prefix cache:   hash(prefix) → reuse blocks; copy-on-write on divergence
```
