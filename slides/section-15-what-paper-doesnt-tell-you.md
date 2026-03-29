# Section 15: What the Paper Doesn't Tell You — Lessons from Community Implementations

**Duration:** 5 minutes  
**Goal:** Bridge the gap between the paper's clean theory and messy production reality. This section gives the audience practical knowledge they can't get from reading the paper alone, and establishes your credibility as someone who has investigated beyond the published results.

---

## Why This Section Exists

> "Every paper tells a clean story. The experiments work, the bounds are tight, the results are compelling. But when engineers try to implement the algorithm on real models and real hardware, they discover things the paper didn't mention. I've been tracking the community implementations of TurboQuant — in llama.cpp, vLLM, PyTorch, Triton, and Rust — and there are several findings worth sharing."

---

## Finding 1: Keys and Values Are Not Equal

The paper treats Keys and Values symmetrically — same bit-width, same algorithm. But real LLMs have dramatically different K and V distributions.

```
Typical magnitude ratios (K norm / V norm) across models:

  GPT-2 family:       K/V ratio < 10×       → uniform bits work
  Phi-2, Qwen-3B:     K/V ratio 10-60×      → Keys need more bits
  Qwen-1.5B, 7B:      K/V ratio > 100×      → significant asymmetry
```

Why does this matter? Quantization error scales with the squared norm of the original vector. If Keys have 50× larger norms than Values, the Key quantization error dominates by 2500×.

```
Quantization error ∝ ‖vector‖² × (distortion rate)

If ‖K‖ = 50 × ‖V‖:
  Key error   ∝ 2500 × (distortion rate)
  Value error ∝    1 × (distortion rate)
```

**Practical recommendation from community experiments:**

```
Conservative:  Keys at 4 bits, Values at 2 bits
Balanced:      Keys at 3 bits, Values at 2 bits
Aggressive:    Keys at 3 bits, Values at 2 bits + sparse V optimization
```

The paper's 2.5-bit and 3.5-bit configurations partially address this through outlier channels, but the K/V asymmetry is a first-order effect the paper doesn't discuss explicitly.

> **If you implement TurboQuant, don't use the same bit-width for Keys and Values.** Profile your model's K/V magnitude ratios first.

---

## Finding 2: QJL (Stage 2) May Hurt More Than It Helps

This is the most surprising finding from the community.

The paper's theory says TurboQuant_prod (MSE + QJL) is superior to TurboQuant_mse (MSE only) because it's unbiased. But multiple independent implementers have found the opposite in practice:

```
llama.cpp community (Discussion #20969):
  "QJL adds variance that softmax amplifies.
   Low variance (MSE) beats unbiasedness (Prod)."

  At 3-bit on GPT-2 (head_dim=64):
  TurboQuant_prod → 300% perplexity increase (!!)
  TurboQuant_mse  → near-baseline perplexity
```

Why does this happen?

```
Theory:  Bias is bad because it distorts attention rankings.
Reality: At 3+ bits, the bias is small (~3%).
         But QJL's variance gets amplified EXPONENTIALLY by softmax.
         The cure is worse than the disease.

         The variance from QJL on a residual of norm 0.17 may
         cause more rank inversions (via softmax) than the
         3% systematic bias it was designed to fix.
```

The effect is strongest at **small head dimensions** (d=64) where the 1/d variance reduction isn't large enough, and at **higher bit-widths** (b≥3) where the residual bias is already small.

**Practical recommendation:**

```
b ≤ 2:   Use TurboQuant_prod (bias is large, QJL correction needed)
b ≥ 3:   Try TurboQuant_mse first (simpler, often better in practice)
          Benchmark both on your specific model and task
```

> **The theoretically optimal variant isn't always the practically optimal variant.** Softmax's nonlinear amplification of variance is something the paper's linear analysis doesn't fully capture.

---

## Finding 3: 3-4 Bits Is the Universal Sweet Spot

Across all community experiments — different models (GPT-2, Qwen, Llama, Mistral, Phi), different hardware (NVIDIA, Apple Silicon), different frameworks (PyTorch, llama.cpp, vLLM) — the consensus converges:

```
2 bits:   Works for Values. Too aggressive for Keys on most models.
          Quality degradation visible on hard tasks.

3 bits:   Good quality. The paper's sweet spot.
          Compression ~5.3× from float16.

3.5 bits: Quality-neutral for all tested models and tasks.
          Compression ~4.6× from float16.

4 bits:   Essentially perfect. Hard to distinguish from float16.
          Compression ~4× from float16.
          Diminishing returns vs 3.5 bits.

5+ bits:  Overkill. The improvement over 4 bits is negligible.
          Not worth the memory cost.
```

The theory predicted this: each additional bit reduces error by 4×, with rapidly diminishing returns. Practice confirms it perfectly.

---

## Finding 4: Rotation Implementation Matters

The paper describes rotation using a dense random matrix generated via QR decomposition. In practice, community implementations have found that **randomized Hadamard transforms** work just as well and are significantly faster:

```
Dense random rotation (QR decomposition):
  Storage: d × d matrix (128×128 = 64 KB)
  Cost:    O(d²) per vector
  Generation: Expensive (QR decomposition)

Randomized Hadamard Transform:
  Storage: One random sign vector (128 bits = 16 bytes)
  Cost:    O(d log d) per vector
  Generation: Trivial (random signs + deterministic Hadamard)
```

The Hadamard transform is a structured rotation that achieves the same distributional properties (Beta/Gaussian coordinates, near-independence) with less compute and virtually no storage.

```
Hadamard rotation:
  1. Multiply each coordinate by a random ±1  (the sign vector)
  2. Apply the Walsh-Hadamard transform       (recursive butterfly operations)

Both steps are fully vectorizable and cache-friendly on GPUs.
```

The llama.cpp implementation (turboquant_plus) uses this approach and achieves **speed parity with q8_0** — meaning the rotation overhead is negligible compared to the attention computation itself.

---

## Finding 5: It Works on Vision-Language Models Too

The paper tests only on text-only models. But a community implementation tested TurboQuant on **Molmo2-8B** processing video input — where a 12-second video clip produces ~11,000 visual tokens.

```
11,000 visual tokens → 1.6 GB KV cache on a 24 GB GPU

With TurboQuant at 4 bits:
  KV cache compressed to ~430 MB
  Compression ratio: 3.76×
  Model correctly identifies all characters in a 23-min Seinfeld episode
  Throughput: 24 tokens/second
```

Visual tokens are 10× more numerous than text tokens and flow through all the same attention layers. The fact that TurboQuant handles them without quality loss confirms it's genuinely model-agnostic — not just "works on the models the authors tested."

---

## The Current Landscape

A quick snapshot of where implementations stand as of today:

```
┌───────────────────┬────────────────────────────────────────────┐
│ Framework         │ Status                                     │
├───────────────────┼────────────────────────────────────────────┤
│ Google (official) │ No code released. Expected Q2 2026.        │
│ llama.cpp         │ Community impl with Metal support.         │
│                   │ Working end-to-end on Apple Silicon.       │
│                   │ Not merged to main branch yet.             │
│ vLLM              │ Plugin available. Feature request open.    │
│                   │ Not native yet.                            │
│ PyTorch/Triton    │ Multiple reference implementations.        │
│                   │ Triton kernels for GPU. Experimental.      │
│ Rust              │ Standalone library with PolarQuant + QJL.  │
│ TensorRT-LLM      │ Not available. Expected post-Google release│
└───────────────────┴────────────────────────────────────────────┘
```

> **Production-ready framework support is ~2-3 months away.** But if you want to prototype, the llama.cpp and PyTorch implementations are functional today.

---

## Speaker Notes

- **This section is what separates a paper-reader from a practitioner.** The audience will remember these findings because they're the kind of thing you only learn by looking beyond the paper. Presenting them establishes credibility.
- **Finding 2 (QJL hurting in practice) is the most surprising.** Spend the most time here. The tension between theory (unbiased is always better) and practice (variance through softmax is worse than small bias) is a genuinely interesting insight that sparks discussion.
- **Finding 1 (K/V asymmetry) is the most actionable.** If anyone in the audience goes home and implements TurboQuant, this is the first thing they need to know. "Don't use the same bits for Keys and Values."
- **Finding 4 (Hadamard transform) is for the systems-minded engineers.** It shows that the paper's theoretical framework admits very efficient implementations. O(d log d) instead of O(d²) matters at scale.
- **The implementation landscape table** sets realistic expectations. "This is coming, but it's not plug-and-play today." Engineers appreciate honesty about maturity levels.
- **Don't present these findings as criticisms of the paper.** Frame them as "the paper establishes the theory; the community is discovering the engineering." Both are necessary. The theory tells you what's possible; the engineering tells you what works best in practice.
- **Possible audience questions:**
  - "How did you find all these community implementations?" — GitHub, the llama.cpp discussions, Hacker News, arXiv comments, and ML Twitter/X. The paper dropped in April 2025 and the community moved fast.
  - "Should I wait for the official Google implementation?" — For production, probably yes. For prototyping and understanding, the community implementations are fine. The algorithm is simple enough that correctness is verifiable.
  - "Is there a risk Google patents this?" — The paper is published at ICLR, the math is in the public domain, and the QJL predecessor is already open. Patent risk seems low, but I'm not a lawyer.
- **Transition to Section 16:** "Let's zoom out and place TurboQuant in the broader picture of how LLM inference is being optimized."
