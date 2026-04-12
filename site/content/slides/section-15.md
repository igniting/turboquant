---
title: "What the Paper Doesn't Tell You"
weight: 15
part: "Part VI — Practical & Closing"
---

![What the Paper Doesn't Tell You](/img/s15-paper-doesnt-tell.webp)

Every paper tells a clean story. But when engineers try to implement the algorithm on real models and real hardware, they discover things the paper didn't mention. Here are the key findings from community implementations across llama.cpp, vLLM, PyTorch, MLX, and Rust -- validated since the March 2026 public release.

---

## Finding 1: Keys and Values Are Not Equal

The paper treats Keys and Values symmetrically. Real LLMs have dramatically different K and V distributions:

```
K norm / V norm ratios across models:

  GPT-2 family:       K/V ratio < 10×       → uniform bits work
  Phi-2, Qwen-3B:     K/V ratio 10-60×      → Keys need more bits
  Qwen-1.5B, 7B:      K/V ratio > 100×      → significant asymmetry
```

Since quantization error scales with $\|v\|^2$, if Keys have 50x larger norms, Key quantization error dominates by 2500x.

> **If you implement TurboQuant, don't use the same bit-width for Keys and Values.** Profile your model's K/V magnitude ratios first. Community recommendation: Keys at 3-4 bits, Values at 2 bits.

The turboquant+ community has independently validated three findings about this asymmetry across M1/M2/M3/M5 Mac, RTX 3080 Ti/3090/4090/5090, and AMD 6800 XT/9070 XT:

1. **V compression is essentially free.** Compressing the Value cache all the way to 2-bit has near-zero measurable effect on attention quality when Key precision is maintained.
2. **All quality degradation comes from Key compression.** Keys control attention routing through softmax -- compress K aggressively and quality drops; compress V aggressively and it barely matters. This makes asymmetric configs practical: `-ctk q8_0 -ctv turbo2`.
3. **Boundary layers are disproportionately sensitive.** Protecting the first 2 and last 2 transformer layers at higher precision recovers 37-91% of any quality gap, at minimal memory cost.

---

## Finding 2: QJL (Stage 2) May Hurt More Than It Helps

This is the most surprising finding. Multiple independent implementers found that TurboQuant_mse (MSE only) outperforms TurboQuant_prod (MSE + QJL) in practice:

```
llama.cpp community:
  At 3-bit on GPT-2 (head_dim=64):
  TurboQuant_prod → 300% perplexity increase (!!)
  TurboQuant_mse  → near-baseline perplexity
```

Why? At 3+ bits, the MSE bias is small (~3%). But QJL's variance gets **amplified exponentially by softmax**. The cure is worse than the disease. The turboquant+ implementation uses PolarQuant with Walsh-Hadamard rotation instead:

```
b ≤ 2:   Use TurboQuant_prod (bias is large, QJL correction needed)
b ≥ 3:   Use PolarQuant + WHT (simpler, consistently better in practice)
```

> **The theoretically optimal variant isn't always the practically optimal variant.** Softmax's nonlinear amplification of variance is something the paper's linear analysis doesn't fully capture.

---

## Finding 3: 3-4 Bits Is the Universal Sweet Spot

Across all community experiments -- different models, hardware, and frameworks -- the consensus:

```
2 bits:   Works for Values. Too aggressive for Keys. Quality degrades.
3 bits:   Good quality. ~5.1× compression (block_size=128: 5.12×).
3.5 bits: Quality-neutral for all tested models and tasks. ~4.6×.
4 bits:   Essentially perfect. ~4×. Beats q4_0 in quality at higher compression.
5+ bits:  Overkill. Not worth the memory cost.
```

---

## Finding 4: Rotation Implementation Matters

The paper uses dense random matrices (QR decomposition). In practice, **Walsh-Hadamard Transforms** work just as well and are dramatically faster:

```
Dense random rotation:
  Storage: d × d matrix (64 KB for d=128)
  Cost:    O(d²) per vector

Walsh-Hadamard Transform:
  Storage: One random sign vector (16 bytes)
  Cost:    O(d log d) per vector
```

The llama.cpp implementation uses this approach and achieves **speed parity with q8_0** prefill (2747 vs 2694 tok/s) -- the rotation overhead is negligible. Rotation Gaussianization has been validated on real Qwen3 KV tensors (kurtosis 900 → 2.9).

---

## Finding 5: It Works on Vision-Language Models Too

A community implementation tested on **Molmo2-8B** processing video -- a 12-second clip produces ~11,000 visual tokens:

```
11,000 visual tokens → 1.6 GB KV cache on a 24 GB GPU

With TurboQuant at 4 bits:
  KV cache compressed to ~430 MB (3.76× compression)
  Model correctly identifies all characters in a 23-min Seinfeld episode
  Throughput: 24 tokens/second
```

Visual tokens are 10x more numerous than text tokens. The fact that TurboQuant handles them without quality loss confirms it's genuinely model-agnostic.

---

## Finding 6: Extreme Scale Results

The community has stress-tested far beyond the paper's experiments:

```
Llama-70B Q4_K_M @ 48K context:   turbo3, PPL 4.019 -- quality-neutral
Command-R+ 104B Q4_K_M @ 128K:    turbo3, PPL 4.024, 74 GB peak memory
                                   Running on a single MacBook (M5 Max 128GB)
Apple M5 Max, Qwen3.5-35B MoE:    144 tokens/sec via Swift MLX
```

> **A 104B model at 128K context on a MacBook.** This was not possible before TurboQuant.

---

## The Current Landscape

| Framework | Status |
|---|---|
| Google (official) | Blog + paper public as of March 2026. No official code released yet. |
| **turboquant+** | **Active community fork (6.1k ★). turbo2/3/4 formats. CUDA, ROCm, Metal, CPU. 30+ testers across 12+ GPU/Mac configs. Validated to 104B at 128K context.** |
| llama.cpp (main) | Discussion #20969 open (125 comments). Incremental patches being upstreamed. Not yet merged as a whole. |
| mlx-swift-lm | Swift MLX for Apple Silicon. Full turbo2/3/4 support. 144 tok/s on Qwen3.5-35B on M5 Max. |
| ik_llama.cpp | Working implementation submitted and reviewed (issue #1509). |
| vLLM | Plugin available. Feature request open for native support. |
| LM Studio | Feature request open (#1719). |
| PyTorch/Triton | Multiple reference implementations. Experimental. |
| TensorRT-LLM | Not available. |

> **Working implementations are available today** for llama.cpp (CUDA, Metal, ROCm, CPU) and Apple Silicon (Swift MLX). Official framework integrations (vLLM native, TensorRT-LLM) are still in progress.
