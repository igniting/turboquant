---
title: "What the Paper Doesn't Tell You"
weight: 15
part: "Part VI — Practical & Closing"
---

![What the Paper Doesn't Tell You](/img/s15-paper-doesnt-tell.webp)

Every paper tells a clean story. But when engineers try to implement the algorithm on real models and real hardware, they discover things the paper didn't mention. Here are the key findings from community implementations in llama.cpp, vLLM, PyTorch, Triton, and Rust.

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

---

## Finding 2: QJL (Stage 2) May Hurt More Than It Helps

This is the most surprising finding. Multiple independent implementers found that TurboQuant_mse (MSE only) outperforms TurboQuant_prod (MSE + QJL) in practice:

```
llama.cpp community:
  At 3-bit on GPT-2 (head_dim=64):
  TurboQuant_prod → 300% perplexity increase (!!)
  TurboQuant_mse  → near-baseline perplexity
```

Why? At 3+ bits, the MSE bias is small (~3%). But QJL's variance gets **amplified exponentially by softmax**. The cure is worse than the disease.

```
b ≤ 2:   Use TurboQuant_prod (bias is large, QJL correction needed)
b ≥ 3:   Try TurboQuant_mse first (simpler, often better in practice)
```

> **The theoretically optimal variant isn't always the practically optimal variant.** Softmax's nonlinear amplification of variance is something the paper's linear analysis doesn't fully capture.

---

## Finding 3: 3-4 Bits Is the Universal Sweet Spot

Across all community experiments -- different models, hardware, and frameworks -- the consensus:

```
2 bits:   Works for Values. Too aggressive for Keys. Quality degrades.
3 bits:   Good quality. ~5.3× compression.
3.5 bits: Quality-neutral for all tested models and tasks. ~4.6×.
4 bits:   Essentially perfect. ~4×. Diminishing returns vs 3.5.
5+ bits:  Overkill. Not worth the memory cost.
```

---

## Finding 4: Rotation Implementation Matters

The paper uses dense random matrices (QR decomposition). In practice, **randomized Hadamard transforms** work just as well and are faster:

```
Dense random rotation:
  Storage: d × d matrix (64 KB for d=128)
  Cost:    O(d²) per vector

Randomized Hadamard Transform:
  Storage: One random sign vector (16 bytes)
  Cost:    O(d log d) per vector
```

The llama.cpp implementation uses this approach and achieves **speed parity with q8_0** -- the rotation overhead is negligible.

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

## The Current Landscape

| Framework | Status |
|---|---|
| Google (official) | No code released. Expected Q2 2026. |
| llama.cpp | Community impl with Metal support. Working on Apple Silicon. |
| vLLM | Plugin available. Feature request open for native support. |
| PyTorch/Triton | Multiple reference implementations. Experimental. |
| Rust | Standalone library with PolarQuant + QJL. |
| TensorRT-LLM | Not available. Expected post-Google release. |

> **Production-ready framework support is ~2-3 months away.** For prototyping, the llama.cpp and PyTorch implementations are functional today.
