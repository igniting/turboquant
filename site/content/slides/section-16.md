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
│    Status:  Emerging. TurboQuant is leading.            │
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

## What Changes When KV Cache Is 4x Smaller

```
Memory:
  └── Same GPU serves 4× more concurrent users
  └── Same GPU supports 4× longer context
  └── Models that needed 2 GPUs now fit on 1

Speed:
  └── 4× less data read from HBM per generation step
  └── Memory-bound generation becomes faster
  └── Up to 8× attention speedup reported on H100

Cost:
  └── 4× fewer GPUs for the same workload
  └── Cloud inference cost per token drops significantly

Capability:
  └── 128K context becomes practical on consumer hardware
  └── Long-document processing (books, codebases) becomes routine
  └── Multi-turn agent conversations can maintain longer memory
```

> **TurboQuant doesn't just save memory -- it shifts what's economically and technically feasible.**
