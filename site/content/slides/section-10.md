---
title: "The Bias Problem — When Minimizing MSE Isn't Enough"
weight: 10
part: "Part IV — The Algorithm"
---

![The Bias Problem](/img/s10-bias-problem.webp)

We just showed that TurboQuant's MSE quantizer is near-optimal -- within 2.7x of the best possible reconstruction. But there's a subtle, critical flaw: **the MSE-optimal quantizer is biased for inner products.**

---

## What "Biased" Means

The quantizer is **randomized** -- the rotation matrix is random, so the quantized output is a random variable. When the inner product estimator is "biased," it means:

$$E[\langle q, \tilde{k} \rangle] \neq \langle q, k \rangle$$

The *average* estimated inner product is NOT equal to the *true* inner product. This isn't random noise that averages out -- it's a **systematic error in one direction**.

> **Bias is a systematic error that doesn't average away.** No matter how many samples you take, the average is still wrong.

---

## Why MSE Quantizers Shrink Vectors

At 1 bit with a Gaussian distribution $N(0, 1/d)$, the Lloyd-Max codebook has two centroids at $\pm\sqrt{2/\pi d} \approx \pm 0.080$ for $d=128$.

The quantizer takes the sign of each rotated coordinate, then replaces it with $\pm 0.080$. What happens to the vector's length?

```
Original rotated vector y:  each coordinate ~ N(0, 1/d)
  → Expected length: ||y||² = d × (1/d) = 1    (unit vector)

Quantized vector:  each coordinate = ±0.080
  → Length: ||ỹ||² = d × (2/πd) = 2/π ≈ 0.637

  → ||ỹ|| = √(2/π) ≈ 0.798
```

The reconstructed vector has length **0.798 instead of 1.0**. It's been shrunk.

```
True inner product:        <q, k> = 0.50
Expected quantized IP:     E[<q, k~>] ≈ (2/π) × <q, k> = 0.637 × 0.50 = 0.318

Bias:  0.318 vs 0.50 → 36% underestimate!
```

> **At 1 bit, every inner product is underestimated by ~36%.** At higher bit-widths, the bias shrinks but remains meaningful:

```
Bit-width    Bias factor        Inner product preserved
   1         2/π   ≈ 0.637     ~64% of true value
   2         ~0.85             ~85%
   3         ~0.97             ~97%
   4         ~0.993            ~99.3%
```

---

## Why Uniform Bias Isn't the Problem

You might think: "If every inner product is scaled by 0.637, then softmax will normalize things out -- the relative ranking of attention scores is preserved."

This is partially true for **uniform** bias (same scale factor applied to all scores). But the bias is not uniform:

```
Token A with true inner product 0.80  →  quantized: E[ĨP] = 0.637 × 0.80 = 0.510
Token B with true inner product 0.20  →  quantized: E[ĨP] = 0.637 × 0.20 = 0.127
Token C with true inner product 0.00  →  quantized: E[ĨP] = 0.637 × 0.00 = 0.000
```

In softmax, the gaps between scores matter more than the absolute values. Softmax of [0.80, 0.20, 0.00] vs softmax of [0.51, 0.13, 0.00] gives different distributions -- A gets a larger share in the first case, C gets relatively more weight in the second.

Additionally, when different attention heads have different magnitudes of Key vectors (which they do, especially across layers), the bias scale factor is effectively different per head. This creates **non-uniform shrinkage** that distorts which tokens the model attends to.

---

## The Solution: An Unbiased Estimator

We need a quantization scheme where:

$$E[\langle q, \tilde{k} \rangle] = \langle q, k \rangle$$

The expected quantized inner product equals the true inner product. Exactly.

This requires a fundamentally different approach -- not just better centroids, but a different kind of quantizer. Section 11 introduces the **Quantized Johnson-Lindenstrauss (QJL)** transform, which achieves unbiased estimation by design.

---

## A Note Before Moving Forward

The two-stage solution (MSE quantizer + QJL bias correction) is elegant and theoretically sound. In controlled experiments on embedding vectors, it outperforms the MSE-only approach at low bit-widths. Sections 11–13 follow this theory to its conclusion.

However, Section 15 covers a finding that emerged after the paper's public release: in practice, on real LLM KV caches, the QJL correction stage **hurts more than it helps at 3+ bits**. The reason relates to softmax amplification — QJL's variance interacts with softmax in ways the theory doesn't fully model. This doesn't invalidate the theoretical analysis; it shows that the ideal operating regime differs between embedding search (where QJL wins) and LLM attention (where it doesn't at higher bit-widths).

> **Keep this in mind through Sections 11–13. The theory is correct. The practical tradeoffs are richer.**
