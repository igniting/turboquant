---
title: "Where This Fits in the Inference Stack"
weight: 16
part: "Part VI — Practical & Closing"
---

![Where This Fits in the Inference Stack](/img/s16-inference-stack.webp)

Modern LLM inference is converging toward a **layered compression stack** where different algorithms handle different memory domains. Each layer is independent and their savings compound.

---

## The Three Layers of Compression

```
┌─────────────────────────────────────────────────────────┐
│                    LLM INFERENCE STACK                   │
│                                                         │
│  Layer 1: WEIGHT QUANTIZATION                           │
│    What:    Model parameters (static, loaded once)      │
│    Methods: GPTQ, AWQ, GGUF, bitsandbytes               │
│    Typical: 4-bit (4× compression)                      │
│    Status:  Mature. Widely deployed.                    │
│                                                         │
│  Layer 2: ACTIVATION QUANTIZATION                       │
│    What:    Intermediate computation values              │
│    Methods: FP8 (H100), NVFP4 (Blackwell)              │
│    Typical: 8-bit or 4-bit (2-4× throughput gain)       │
│    Status:  Hardware-native on latest GPUs.             │
│                                                         │
│  Layer 3: KV CACHE QUANTIZATION     ← TurboQuant       │
│    What:    Cached attention state (dynamic, grows)     │
│    Methods: TurboQuant, KIVI, PolarQuant                │
│    Typical: 3-4 bits (4-5× compression)                 │
│    Status:  Emerging. Community implementations live.   │
│                                                         │
│  Each layer compresses a different memory pool.         │
│  The savings are additive, not multiplicative —         │
│  but together they dramatically shrink total footprint. │
└─────────────────────────────────────────────────────────┘
```

> **These layers are complementary, not competing.** You can quantize weights with AWQ, activations with FP8, AND the KV cache with TurboQuant. Each compounds the other's savings.

---

## Why KV Cache Was the Last Frontier

```
Weights:       Static     + Offline OK     + Large    → Solved first
Activations:   Ephemeral  + Hardware-native + Medium   → Solved by hardware
KV Cache:      Dynamic    + Must be online  + Largest  → Solved last (TurboQuant)
```

TurboQuant cracks the hardest layer -- the one that's dynamic, must be online, and grows without bound.

---

## Quality vs. Compression: Community Benchmarks

Measured on wikitext-2, Llama family, M5 Max 128GB. Lower perplexity delta is better.

| Format | Bits/val | Compression | PPL delta vs q8_0 |
|--------|----------|-------------|-------------------|
| f16 (baseline) | 16.0 | 1.0× | −0.16% |
| q8_0 | 8.5 | 1.9× | 0% (reference) |
| **turbo4** | **4.25** | **3.8×** | **+0.23%** |
| q4_0 | 4.5 | 3.6× | +0.52% |
| turbo3 | 3.5 | 4.6–5.1× | +1.06% |
| turbo2 | 2.5 | 6.4× | +6.48% |

> **turbo4 beats q4_0 in quality at higher compression.** This is a key result: the mathematically-grounded rotation closes the gap that naive quantization leaves open.

---

## What Changes When KV Cache Is 4x Smaller

```
Memory:
  └── Same GPU serves 4× more concurrent users
  └── Same GPU supports 4× longer context
  └── Models that needed 2 GPUs now fit on 1

Speed:
  └── 4× less data read from HBM per generation step
  └── Memory-bound generation becomes faster
  └── Attention kernel speedup of 3-5× measured on H100 (isolated microbenchmark)
      — does not translate directly to total throughput; attention is ~25% of decode

Cost:
  └── 4× fewer GPUs for the same workload
  └── Cloud inference cost per token drops significantly

Capability:
  └── 128K context becomes practical on consumer hardware
  └── 104B model at 128K context fits on a single MacBook (M5 Max)
  └── Long-document processing (books, codebases) becomes routine
  └── Multi-turn agent conversations can maintain longer memory
```

> **TurboQuant doesn't just save memory -- it shifts what's economically and technically feasible.**

---

## Decode Throughput vs. Time to First Token

TurboQuant's benefits are concentrated in the **decode** phase — the right framing depends on context length.

**For short to medium inputs (<32K tokens):**

- Prefill is **compute-bound**: GPUs run matrix multiplications at near-peak utilization. HBM reads are not the bottleneck. TurboQuant has minimal impact on TTFT here.
- Decode is **memory-bandwidth-bound**: for each new token, the entire KV cache must be read from HBM. TurboQuant directly cuts this per-step cost.

**For long inputs (>64K tokens):**

- Prefill attention becomes **memory-bandwidth bound** for its attention layers — the same regime FlashAttention was designed for. At these context lengths, reducing KV cache size with TurboQuant also reduces prefill HBM traffic, and **TTFT does improve**, typically by 20–40% at 128K+ context.
- Decode benefits are even larger: 4× less KV data per step, at a context length where each step is expensive.

```
Context < 32K:   TurboQuant → decode throughput ↑, TTFT unchanged
Context > 64K:   TurboQuant → decode throughput ↑↑, TTFT also improves (~20-40%)

Design implication:
  For short-context chatbots:       FlashAttention for TTFT, TurboQuant for throughput
  For long-context (docs, agents):  TurboQuant improves both
  For batch processing:             TurboQuant is the highest-leverage single change
```

> **The "TurboQuant doesn't help TTFT" rule of thumb holds at short context. At the long contexts where it matters most, TTFT improves too.**

---

## Production Deployment: Tensor Parallelism

At serving scale, models run across multiple GPUs using **tensor parallelism (TP)**. Each GPU holds a shard of the attention weight matrices and a corresponding slice of the KV cache — at TP=8, each GPU stores 1/8 of the total KV vectors.

TurboQuant must be applied per-shard:

```
TP=8 deployment, Llama-70B:
  8 GPUs, each holding 10 attention heads (of 80 total in GQA config)
  Each GPU quantizes its own 10-head KV slice independently
  Scale factors are per-block, per-head, per-GPU → no cross-GPU communication needed

Correct: quantize the shard on the owning GPU before storing
Wrong:   quantize globally then shard → scale factors span GPU boundaries
```

The good news: TurboQuant's per-block independence means quantization is naturally embarrassingly parallel across shards. Each GPU quantizes its slice without coordination. The scale factors are local to each shard and don't need to be shared.

> **TP-aware implementation is required for production use.** Existing community implementations (turboquant+ for llama.cpp) handle single-GPU cases correctly; multi-GPU serving stacks (vLLM, TGI) will need explicit TP integration.

---

## Disaggregated Serving — Where TurboQuant Has Untapped Leverage

Modern high-throughput inference increasingly uses **disaggregated prefill/decode**: prefill runs on compute-optimized nodes, decode runs on memory-optimized nodes, and the KV cache is transferred over the network.

At 128K context with a 70B model, a single request's KV cache can exceed 50 GB at full precision — a significant network transfer on every new session.

```
Without TurboQuant:
  Prefill node → [50 GB KV transfer over NVLink / InfiniBand] → Decode node

With TurboQuant:
  Prefill node → [quantize to 3.5 bits on-the-fly]
             → [~11 GB transfer]
             → Decode node (reads directly from compressed cache)
```

A 4-5× reduction in KV transfer size translates directly into lower inter-node latency and higher effective throughput per network link. This applies equally to **KV migration** during load rebalancing, where caches are moved between decode nodes. Disaggregated systems like Mooncake and SGLang's disaggregated mode are well-positioned to benefit.

---

## Compatibility Summary

| Technology | Compatible with TurboQuant? | Notes |
|---|:---:|---|
| PagedAttention (vLLM) | ✅ Yes | Orthogonal — manages allocation, not size |
| FlashAttention | ✅ Yes | Orthogonal — speeds prefill compute |
| Prefix caching | ✅ Yes | Rotation is deterministic per model load |
| Speculative decoding | ✅ Yes | Acceptance rates unaffected at 3+ bits |
| Continuous batching | ✅ Yes | Smaller KV raises the memory watermark |
| GQA / MQA models | ✅ Yes | Compresses the already-reduced KV heads |
| Disaggregated serving | ✅ Yes | Reduces cross-node KV transfer size |
| Tensor parallelism | ✅ Yes | Per-shard, no cross-GPU coordination |
| Pure SSM models (Mamba) | ❌ N/A | No KV cache to compress |
| TensorRT-LLM | ⏳ Pending | Not yet integrated |
