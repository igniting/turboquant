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
   2                ≈ 0.88      ~88% of true value
   3                ≈ 0.97      ~97% of true value
   4                ≈ 0.993     ~99.3% of true value
```

---

## Why Bias Is Worse Than Noise for Attention

You might think: if all inner products are scaled by the same factor, softmax will compensate -- the relative ranking stays the same.

This would be true if the bias were a **uniform multiplicative factor**. But it's not. The bias depends on the **magnitude of the true inner product**:

```
When true IP ≈ 0.01:  bias ≈ 0.001  (small)
When true IP ≈ 0.06:  bias ≈ 0.005
When true IP ≈ 0.10:  bias ≈ 0.010
When true IP ≈ 0.17:  bias ≈ 0.020  (larger)
```

**Higher true inner products get biased more.** The tokens that *should* receive the most attention are the ones whose scores are reduced the most. This **compresses the score distribution** -- high scores pulled down more than low scores. Through softmax, this makes attention more uniform, weakening the model's ability to focus.

> **Bias doesn't just add noise -- it systematically blunts the attention mechanism's ability to discriminate between relevant and irrelevant tokens.**

---

## What We Need Instead

```
Current state (TurboQuant_mse):
  ✓ Near-optimal MSE
  ✓ Online, GPU-friendly
  ✗ Biased inner products

What we need (TurboQuant_prod):
  ✓ Near-optimal MSE (or close)
  ✓ Online, GPU-friendly
  ✓ UNBIASED inner products     ← the missing piece
```

The next section shows how TurboQuant achieves this with an elegant two-stage approach: use the MSE quantizer to get close, then apply a 1-bit correction to remove the bias.

*(A preview for the skeptical reader: in Section 15 we'll see that community implementations have found the two-stage fix isn't always worth it at 3+ bits -- the QJL correction's variance can hurt more than the small residual bias. But understanding the theory is essential for knowing when each variant is appropriate.)*
