---
title: "Closing — The Three-Step Summary"
weight: 17
part: "Part VI — Practical & Closing"
---

![Closing: The Three-Step Summary](/img/s17-closing.webp)

We've covered a lot of ground. Let's bring it back to the essentials.

---

## The Problem in One Sentence

Every token you generate stores a Key and Value vector at every layer of the model. At scale -- millions of users, 128K context windows, 70B parameters -- this fills your GPU faster than the model weights do.

---

## The Insight in Three Steps

**Step 1 — Rotate**

Apply a random rotation matrix to each KV vector. This redistributes whatever "spiky" structure the raw vectors had into a smooth, near-Gaussian distribution. The rotation preserves inner products perfectly (rotations are orthogonal transformations), so attention accuracy is unchanged before quantization.

**Step 2 — Quantize**

Apply Lloyd-Max scalar quantization to the rotated, near-Gaussian vector. Because the distribution is now Gaussian, the codebook is optimal. Theory guarantees you're within 2.72× of the information-theoretic limit — closer than any other practical algorithm.

**Step 3 — Correct (when needed)**

If you're below 3 bits, add a 1-bit QJL sketch of the residual to eliminate the systematic bias that the MSE quantizer introduces. Above 3 bits, the bias is small enough that softmax tolerates it, and the correction step's variance costs more than it saves.

```
b ≤ 2:   Rotate → Quantize (b-1 bits) → QJL residual
b ≥ 3:   Rotate → Quantize (b bits)
```

---

## The Results in Three Numbers

- **3.5 bits** — the bit-width at which TurboQuant achieves zero measurable quality loss on LongBench across six task types
- **4.6×** — the compression ratio at 3.5 bits
- **0.997** — TurboQuant's Needle-in-a-Haystack recall at 128K context, identical to the uncompressed baseline

---

## Who Should Care

**If you run inference today:**
TurboQuant in the turboquant+ library (CUDA, ROCm, Metal, CPU) works on Llama, Qwen, Mistral, Gemma, Phi, DeepSeek variants, and vision-language models. The practical recipe: turbo4 for quality-first, turbo3 for maximum throughput. Keys at 4 bits, Values at 2 bits, boundary layers protected at higher precision.

**If you're building serving infrastructure:**
The three orthogonal wins are now available together: PagedAttention for allocation, FlashAttention for prefill, TurboQuant for decode KV bandwidth. Use all three. In disaggregated serving, TurboQuant also compresses inter-node KV transfers 4-5×, which can be the tightest bottleneck at long context.

**If you're evaluating SSM/Mamba-style alternatives:**
SSMs trade long-range retrieval fidelity for the elimination of KV cache growth. For workloads where exact retrieval matters — agents, RAG, legal/medical documents — attention with TurboQuant compression remains the more reliable architecture. For workloads where throughput matters and fine-grained retrieval does not, SSMs are a legitimate alternative. The choice is workload-dependent, not a blanket win for either side.

**If you're training new models:**
Consider MLA (DeepSeek's architectural approach) during pretraining. MLA achieves sub-0.1× KV cache compression without any inference-time quantization, at the cost of a different training-time architecture. TurboQuant and MLA address the same problem at different stages: TurboQuant is the best compression for models that exist today; MLA is the architectural choice for models being trained tomorrow.

---

## The Competitive Landscape — Why the Timing Matters

TurboQuant was published at ICLR 2026 with the paper and blog post but no official code. The community filled that gap within weeks.

The reason the timing matters: **Google has implementation advantages that no one else does.**

Google can integrate TurboQuant directly into XLA and TPU firmware — fused into the attention kernel at the hardware level, running on custom silicon they manufacture. The algorithm's rotation and quantization steps are small, regular computations that map extremely well to TPU systolic arrays. An XLA-native implementation could achieve lower latency and higher throughput than any GPU-based implementation, because the quantization overhead essentially disappears into the prefill and decode pipelines.

The community implementations on CUDA, ROCm, and Metal are genuine and production-ready. But they are running on hardware designed for different workloads, adding quantization as an external step. Google's advantage is the ability to co-design the algorithm's execution with the hardware it runs on.

```
Community (CUDA/ROCm/Metal):    Algorithm + hardware, separately optimized
                                 ~3-5× KV bandwidth improvement
                                 Quantization has measurable kernel overhead

Google (TPU + XLA + firmware):  Algorithm + hardware, co-designed
                                 Potential for near-zero quantization overhead
                                 Already have the serving infrastructure to deploy at scale
```

This gap is time-bounded. TPU-specific implementations will take time to productionize. The 6–18 month window before that happens is the window for everyone else to close the gap on their own hardware.

> **The algorithm is public. The math is the same for everyone. The race now is implementation quality, hardware efficiency, and integration depth — and that race has already started.**

---

## The One-Slide Version

```
Problem:   KV cache grows without bound. At 128K context, it exceeds the model weights.
           Every generation step reads the entire cache from HBM.

Algorithm: Rotate each vector → near-Gaussian distribution.
           Quantize → within 2.72× of information-theoretic optimum.
           At 3+ bits, skip the bias correction.

Result:    3.5 bits per coordinate. Zero quality loss. 4.6× compression.
           104B model. 128K context. One MacBook.

What's next: Tensor-parallel integration. vLLM native support.
             Disaggregated KV transfer compression.
             And Google racing to productionize this on TPUs.
```
