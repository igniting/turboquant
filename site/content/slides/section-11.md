---
title: "The Two-Stage Fix — QJL on the Residual"
weight: 11
part: "Part IV — The Algorithm"
---

![The Two-Stage Fix](/img/s11-two-stage.webp)

> Use most of your bit budget to get close (MSE quantizer), then spend 1 bit per coordinate to correct the bias on whatever error remains.

---

## Stage 1: Get Close with MSE Quantizer

For a target bit-width of $b$, run the MSE quantizer with **(b-1) bits** instead of $b$:

```
Input: vector k (unit norm, d-dimensional)

Stage 1: TurboQuant_mse with (b-1) bits
  → Rotate: y = Π · k
  → Quantize each coordinate to (b-1) bits
  → Dequantize to get reconstruction k̃_mse

k̃_mse is close to k, but biased for inner products.
```

The remaining 1 bit per coordinate is saved for Stage 2.

---

## The Residual: What Stage 1 Got Wrong

The **residual** is the error left over after Stage 1:

$$r = k - \tilde{k}_{\text{mse}}$$

Since Stage 1 was a good MSE quantizer, the residual is **small**. For $b = 4$ (Stage 1 uses 3 bits): $\|r\| \approx 0.17$ -- only ~17% of the original vector's energy.

The inner product error from Stage 1 comes entirely from the residual:

$$\langle q, k \rangle - \langle q, \tilde{k}_{\text{mse}} \rangle = \langle q, r \rangle$$

> **The problem reduces to: estimate the inner product with a small residual vector, using just 1 bit per coordinate.**

---

## Stage 2: QJL — 1-Bit Unbiased Inner Products

The **Quantized Johnson-Lindenstrauss (QJL)** transform produces a 1-bit-per-coordinate sketch that gives **unbiased** inner product estimates:

```
Input: residual vector r (d-dimensional)

Step 1: Multiply by a random Gaussian matrix S (d × d)
        z = S · r

Step 2: Take the sign of each entry
        qjl = sign(z)    → d values, each ±1 → 1 bit per coordinate

Step 3: Store qjl and ||r|| (the residual's norm)
```

The intuition: if $r$ and $q$ point in similar directions, most random projections of $r$ will have the same sign as the corresponding projections of $q$. The fraction of sign agreements is a noisy but **unbiased** estimate of their alignment.

> **QJL trades precision for unbiasedness.** Each bit carries very little information, but what it carries is correct on average.

---

## Why the Combination Is Unbiased

```
E[<q, k̃>] = E[<q, k̃_mse + k̃_qjl>]
           = E[<q, k̃_mse>] + E[<q, k̃_qjl>]
           = E[<q, k̃_mse>] + <q, r>           (QJL is unbiased)
           = E[<q, k̃_mse>] + <q, k - k̃_mse>
           = E[<q, k̃_mse>] + <q, k> - E[<q, k̃_mse>]
           = <q, k>  ✓
```

The bias from Stage 1 is **exactly cancelled** by QJL's unbiased estimate of the residual's inner product.

> **It doesn't matter that Stage 1 is biased. It only matters that Stage 1 makes the residual small (low MSE), and Stage 2 handles the residual without bias.**

---

## Why Not Just Use QJL Alone?

Because QJL has **variance** proportional to the input's squared norm:

```
Variance with Stage 1:    (π/2d) × 0.03 × ||q||²   (residual is small)
Variance without Stage 1: (π/2d) × 1.0 × ||q||²   (full vector)
```

Without Stage 1, QJL's variance is 33x higher (for 3-bit target). The residual's small norm is what makes QJL's variance acceptable.

---

## Storage Requirements

```
TurboQuant_prod at b bits per coordinate:

  Stage 1 component:   (b-1) bits per coordinate
  Stage 2 component:   1 bit per coordinate + 1 scalar (||r||)

  Total per vector: b bits per coordinate + 1 scalar
  Overhead from scalar ||r||: 16 bits / 128 coordinates = 0.125 bits per coordinate

  Effective cost: b + 0.125 bits per coordinate ≈ b bits  (overhead is small)
```

---

## The Practical Catch — When QJL Backfires

In the controlled experiments above, QJL clearly outperforms the MSE-only approach. But community implementations across llama.cpp, vLLM, and MLX revealed a surprising finding for production LLM inference:

**At 3+ bits, skipping the QJL stage consistently yields better perplexity.**

The culprit is softmax amplification. At 3-bit precision, the MSE bias has already shrunk to ~3% of the inner product. But QJL's residual estimation introduces variance that, when exponentiated by softmax, creates larger fluctuations than the original bias would have. The bias is small and systematic (tolerable); the QJL variance is unpredictable (harmful).

```
Empirical recommendation from turboquant+ community:

  b ≤ 2 bits:  Use two-stage TurboQuant_prod
               → Bias is 20-36%, QJL correction is essential
               → Variance is less harmful at this bit-width

  b ≥ 3 bits:  Use TurboQuant_mse (MSE stage only)
               → Bias has shrunk to ~3%, acceptable for softmax
               → Skipping QJL eliminates its variance cost
```

> **The theoretical optimum and the practical optimum diverge at 3+ bits.** This doesn't invalidate the theory — it reveals that LLM attention is a stricter environment than embedding search, where the full two-stage approach remains the right choice.
