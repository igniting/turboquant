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
│    Methods: FP8 (H100/H200/B200), NVFP4 (Blackwell)    │
│    Typical: 8-bit or 4-bit (2-4× throughput gain)       │
│    Status:  Hardware-native on latest GPUs.             │
│                                                         │
│  Layer 3: KV CACHE QUANTIZATION     ← TurboQuant       │
│    What:    Cached attention state (dynamic, grows)     │
│    Methods: FP8 (free, 2×), TurboQuant, KIVI, PolarQ   │
│    Typical: 3-4 bits (4-5× compression over FP16)       │
│    Status:  FP8 deployed; TurboQuant: community live.   │
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
KV Cache:      Dynamic    + Must be online  + Largest  → FP8 solved 2×; TurboQuant solves 4-5×
```

---

## Quality vs. Compression: The Full Comparison

The honest comparison includes FP8 KV — the hardware-native baseline that ships today — alongside TurboQuant. Measured on wikitext-2, Llama family, M5 Max 128GB. Lower perplexity delta is better.

| Format | Bits/val | Compression | PPL delta vs FP16 | Overhead |
|--------|----------|-------------|-------------------|---------|
| fp16 (baseline) | 16.0 | 1.0× | 0% | none |
| **fp8 KV** | **8.0** | **2.0×** | **~0%** | **zero — hardware native** |
| q8_0 | 8.5 | 1.9× | +0.16% | software |
| **turbo4** | **4.125** | **3.9×** | **+0.23%** | **rotation + table lookup** |
| q4_0 | 4.5 | 3.6× | +0.52% | software |
| turbo3 | 3.125 | 5.1× | +1.06% | rotation + table lookup |
| turbo2 | 2.125 | 7.5× | +6.48% | rotation + table lookup |

Reading this table honestly:
- **FP8 KV is the right first step.** It is free on H100/H200/B200, already deployed, and quality-neutral. Use it if 2× compression is sufficient.
- **TurboQuant picks up where FP8 stops.** Going from 8-bit to 4-bit adds ~0.23% perplexity for another ~2× compression. This is the tradeoff you're evaluating when choosing TurboQuant.
- **turbo4 beats q4_0** at lower effective bit-width, confirming the rotation's value beyond simple rounding.

---

## What Changes When KV Cache Is 4× Smaller (vs FP16)

```
Memory:
  └── Same GPU serves 4× more concurrent users  (vs FP16; 2× more vs FP8)
  └── Same GPU supports 4× longer context
  └── Models that needed 2 GPUs now fit on 1

Speed:
  └── 4× less data read from HBM per generation step (vs FP16)
  └── Memory-bound generation becomes faster
  └── Attention kernel speedup of 3-5× measured in microbenchmarks on H100
      (attention is ~25% of decode; end-to-end improvement is proportionally smaller)

Cost:
  └── 4× fewer GPUs for the same workload (vs FP16)
  └── Cloud inference cost per token drops significantly

Capability:
  └── 128K context becomes practical on consumer hardware
  └── 104B model at 128K context fits on a single MacBook (M5 Max)
  └── Long-document processing (books, codebases) becomes routine
```

---

## Decode Throughput vs. Time to First Token

TurboQuant's benefits are concentrated in the **decode** phase — the right framing depends on context length.

**For short to medium inputs (<32K tokens):**
- Prefill is **compute-bound**: GPUs run matrix multiplications at near-peak utilization. TurboQuant has minimal impact on TTFT here.
- Decode is **memory-bandwidth-bound**: for each new token, the entire KV cache must be read from HBM. TurboQuant directly cuts this per-step cost.

**For long inputs (>64K tokens):**
- Prefill attention becomes **memory-bandwidth bound** for its attention layers — same regime FlashAttention was designed for. At these context lengths, TurboQuant also reduces prefill HBM traffic and **TTFT improves**, typically 20–40% at 128K+ context.

```
Context < 32K:   TurboQuant → decode throughput ↑, TTFT unchanged
Context > 64K:   TurboQuant → decode throughput ↑↑, TTFT also improves (~20-40%)
```

---

## Production Deployment: Tensor Parallelism

At serving scale, models run across multiple GPUs using **tensor parallelism (TP)**. Each GPU holds a shard of the attention weight matrices and a corresponding slice of the KV cache — at TP=8, each GPU stores 1/8 of the total KV vectors.

TurboQuant must be applied per-shard. Scale factors are local to each shard and don't require cross-GPU communication. The per-block independence makes this naturally embarrassingly parallel.

```
Correct: quantize the local KV shard on the owning GPU before writing to HBM
Wrong:   quantize globally then shard → scale factors span GPU boundaries
```

> **Existing community implementations handle single-GPU cases. Multi-GPU serving stacks (vLLM, TGI) need explicit TP-aware integration.**

---

## TensorRT-LLM: Why Integration Is a Kernel Problem

TensorRT-LLM is listed as "not yet integrated" — not because of resource priorities, but because of kernel architecture. TensorRT-LLM uses **fused attention kernels** (paged attention, chunked attention, flash-decoding variants) that handle KV reads and dot products inside a single CUDA kernel, without materializing intermediate values.

To support TurboQuant, dequantization must happen *inside* this kernel — in registers, while K/V tiles are loaded from HBM — not as a separate pre-processing pass. This requires modifying the inner loop of every fused attention kernel variant separately. It is substantial kernel engineering work, not a library wrapper.

```
Standard fused kernel:  K/V read from HBM (FP16) → attention scores
TurboQuant-fused:       K/V read from HBM (INT4) → dequantize in registers → attention scores

Required change: modify tile-load + compute path of every kernel variant
Scope: chunked-attention, paged-attention, flash-decoding — separately
```

When TensorRT-LLM integration arrives, it will be kernel-level and will likely be faster than the current community implementations that apply rotation and quantization as separate passes.

---

## Blackwell FP4: The Natural Convergence Point

Blackwell (B200, B100) introduces native **NVFP4** tensor core instructions — hardware-accelerated 4-bit floating-point arithmetic. For activation quantization, this is already being used. For KV cache, it represents a future convergence path.

TurboQuant currently uses INT4 integer codebook indices (table lookup dequantization). NVFP4 uses E2M1 floating-point format (hardware-accelerated). These are different representations. But the direction is clear:

```
Today:
  TurboQuant INT4 + custom CUDA kernel → ~4× KV compression, software overhead

Future (Blackwell-native path):
  TurboQuant centroids mapped to NVFP4 values → hardware-accelerated dequant
  → Same ~4× KV compression, near-zero overhead on B200+
```

Whether TurboQuant's Lloyd-Max centroids map cleanly to NVFP4's E2M1 values (or require retraining with FP4-aware codebooks) is an open engineering question. The algorithmic groundwork is done. The hardware is available. This is the most likely path to zero-overhead 4-bit KV compression on Blackwell.

---

## Disaggregated Serving — Where TurboQuant Has Untapped Leverage

Disaggregated prefill/decode separates prefill nodes from decode nodes, with KV cache transferred over the network. At 128K context with a 70B model, uncompressed KV exceeds 50 GB per request.

```
Without TurboQuant:  ~50 GB over InfiniBand (200 Gb/s) = ~2 seconds per request
With TurboQuant:     ~11 GB = ~0.45 seconds per request

NVSwitch (NVL72):    14.4 TB/s fabric — transfer time is <1 ms either way
```

The benefit is largest for **InfiniBand-connected disaggregated deployments** (the common cloud architecture), not for NVL72 rack-scale systems where NVSwitch bandwidth makes KV transfer negligible. Know your interconnect.

---

## The Production Recipe: W4A8KV4

The current state-of-the-art for H100/H200 inference combines three compression layers:

```
W4A8KV4 recipe:
  W4:  4-bit weight quantization (GPTQ or AWQ)
       → 4× weight memory reduction, faster weight loads
  A8:  FP8 activation quantization (hardware-native on H100+)
       → 2× compute throughput improvement at near-zero quality cost
  KV4: 4-bit KV cache (FP8 for free 2×; TurboQuant for the next 2×)
       → 4× KV memory and bandwidth reduction

Combined effect on Llama 3.1 70B serving:
  Weights:   140 GB → 35 GB    (W4)
  KV cache:  16 GB/user → 4 GB/user @ 128K context  (KV4)
  Throughput: A8 adds ~1.8× compute ops/sec on H100

System result: ~4× more concurrent users per GPU, ~2× faster decode, 128K context practical
```

If you are building a production inference stack today, this is the three-line config to evaluate first. FP8 activations require no application changes on H100. W4 via GPTQ/AWQ is widely available for major model families. KV4 via TurboQuant requires the turboquant+ library or a patched llama.cpp.

---

## Compatibility Summary

| Technology | Compatible with TurboQuant? | Notes |
|---|:---:|---|
| PagedAttention (vLLM) | ✅ Yes | Orthogonal — manages allocation, not size |
| FlashAttention | ✅ Yes | Orthogonal — speeds prefill compute |
| FP8 KV cache | ✅ Complementary | FP8 gives 2× free; TurboQuant gives the next 2× |
| Prefix caching | ✅ Yes | Rotation is deterministic per model load |
| Speculative decoding | ✅ Yes | Acceptance rates unaffected at 3+ bits |
| Continuous batching | ✅ Yes | Smaller KV raises the memory watermark |
| GQA / MQA models | ✅ Yes | Compresses the already-reduced KV heads |
| Disaggregated serving | ✅ Yes | Cuts InfiniBand KV transfer; less impact on NVSwitch |
| Tensor parallelism | ✅ Yes | Per-shard; no cross-GPU coordination needed |
| Blackwell NVFP4 path | 🔬 Research | INT4 → FP4 format alignment is open question |
| Pure SSM models (Mamba) | ❌ N/A | No KV cache to compress |
| TensorRT-LLM | ⏳ Kernel work | Requires fused attention kernel modification |
