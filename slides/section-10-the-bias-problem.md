# Section 10: The Bias Problem — When Minimizing MSE Isn't Enough

**Duration:** 5 minutes  
**Goal:** Reveal the twist that makes the paper more than just "rotate and quantize." The audience should understand: what bias means for inner products, why MSE-optimal quantizers systematically shrink vectors, why this distorts attention through softmax, and why a different approach is needed for inner product preservation.

---

## The Twist

> "We just showed that TurboQuant's MSE quantizer is near-optimal — within 2.7× of the best possible reconstruction. You might think we're done. We're not. There's a subtle but critical flaw: **the MSE-optimal quantizer is biased for inner products.**"

---

## What "Biased" Means

Recall that the quantizer is **randomized** — the rotation matrix Π is random, so the quantized output is a random variable. When we say the inner product estimator is "biased," we mean:

```
E[⟨q, k̃⟩]  ≠  ⟨q, k⟩

The AVERAGE estimated inner product (over random rotations)
is NOT equal to the TRUE inner product.
```

This is different from random noise. Random noise would sometimes overestimate and sometimes underestimate, averaging out to the correct value. Bias means the estimates are **systematically wrong in one direction**.

An analogy:

```
Unbiased estimator:
  True value: 100
  Estimates: 95, 108, 97, 103, 99, 101, ...
  Average: ≈ 100  ✓  (noise cancels out)

Biased estimator:
  True value: 100
  Estimates: 78, 82, 75, 80, 77, 81, ...
  Average: ≈ 79   ✗  (systematically too low)
```

> **Bias is a systematic error that doesn't average away.** No matter how many samples you take, the average is still wrong.

---

## Why MSE Quantizers Shrink Vectors

Let's see the bias in action with the simplest case: 1-bit quantization.

At 1 bit with a Gaussian distribution N(0, 1/d), the Lloyd-Max optimal codebook has two centroids:

```
c₁ = -√(2/πd)  ≈ -0.080   (for d = 128)
c₂ = +√(2/πd)  ≈ +0.080   (for d = 128)
```

The quantizer is essentially: **take the sign of each rotated coordinate, then replace it with ±0.080.**

Now think about what happens to the vector's length:

```
Original rotated vector y:  each coordinate ≈ N(0, 1/d)
  → Expected length: ‖y‖² = Σ yⱼ² ≈ d × (1/d) = 1    ✓  (unit vector)

Quantized vector ỹ:  each coordinate = ±0.080
  → Length: ‖ỹ‖² = Σ (0.080)² = d × (2/πd) = 2/π ≈ 0.637

  → ‖ỹ‖ = √(2/π) ≈ 0.798
```

The reconstructed vector has length **0.798 instead of 1.0**. It's been **shrunk by a factor of 2/π ≈ 0.637** in squared norm.

This shrinkage directly affects inner products:

```
True inner product:        ⟨q, k⟩ = some value, say 0.50

Expected quantized IP:     E[⟨q, k̃⟩] ≈ (2/π) × ⟨q, k⟩
                                      = 0.637 × 0.50
                                      = 0.318

Bias:  0.318 vs 0.50 → 36% underestimate!
```

> **At 1 bit, every inner product is underestimated by ~36%.** The quantizer systematically makes vectors shorter, which makes all dot products smaller.

---

## Why Does Shrinkage Happen?

The geometric intuition is straightforward.

Quantization snaps each coordinate to one of a few fixed values. These fixed values are chosen to minimize MSE — which means they're positioned at the **centroid (average)** of each bucket, not at the extremes.

```
Bucket for positive values:
  Contains all values from 0 to ∞ (approximately)
  Centroid (average) = √(2/πd) ≈ 0.080

But some values in this bucket were:
  0.15  → snapped to 0.080  (moved toward zero)
  0.20  → snapped to 0.080  (moved toward zero)
  0.05  → snapped to 0.080  (moved away from zero)
  0.01  → snapped to 0.080  (moved away from zero)
```

On average, values are moved **toward the centroid**, which is closer to zero than the typical value in the bucket. This "regression to the mean" makes the vector shorter overall.

At higher bit-widths, the buckets are narrower, so the shrinkage is smaller:

```
Bit-width    Bias factor        Inner product preserved
   1         2/π   ≈ 0.637     ~64% of true value
   2                ≈ 0.88      ~88% of true value
   3                ≈ 0.97      ~97% of true value
   4                ≈ 0.993     ~99.3% of true value
```

> The bias shrinks with more bits, but at 2-3 bits it's still large enough to matter for attention.

---

## Why Bias Is Worse Than Noise for Attention

You might think: "If all inner products are scaled by the same factor, softmax will compensate — the relative ranking stays the same."

This would be true if the bias were a **uniform multiplicative factor** across all inner products. But it's not. The bias depends on the **magnitude of the true inner product**.

From Figure 2 in the paper: for TurboQuant_mse at 2 bits, the bias is:

```
When true IP ≈ 0.01:  bias ≈ 0.001  (small absolute bias)
When true IP ≈ 0.06:  bias ≈ 0.005
When true IP ≈ 0.10:  bias ≈ 0.010
When true IP ≈ 0.17:  bias ≈ 0.020  (larger absolute bias)
```

**Higher true inner products get biased more.** This means the tokens that *should* receive the most attention (highest IP with the query) are the ones whose scores are reduced the most.

The effect on attention:

```
True scores:        Token A: 0.85    Token B: 0.30
                    (A should dominate)

After uniform bias: Token A: 0.54    Token B: 0.19
                    (A still dominates — ranking preserved ✓)

After IP-dependent  Token A: 0.54    Token B: 0.22
bias:               (A's lead reduced — ranking may flip ✗)
```

The non-uniform bias **compresses the score distribution** — high scores are pulled down more than low scores. Through softmax, this makes the attention distribution more uniform, weakening the model's ability to focus on the most relevant tokens.

> **Bias doesn't just add noise — it systematically blunts the attention mechanism's ability to discriminate between relevant and irrelevant tokens.**

---

## What We Need Instead

The MSE quantizer gives us near-optimal reconstruction but biased inner products. For KV cache quantization, we need:

```
Current state (TurboQuant_mse):
  ✓ Near-optimal MSE
  ✓ Online, GPU-friendly
  ✗ Biased inner products

What we need (TurboQuant_prod):
  ✓ Near-optimal MSE (or close)
  ✓ Online, GPU-friendly
  ✓ UNBIASED inner products     ← this is the missing piece
```

Unbiased means:

```
E[⟨q, k̃⟩] = ⟨q, k⟩    for ANY q and k
```

On average, the estimated inner product equals the true inner product. There's still variance (random noise), but no systematic error.

> "The next section shows how TurboQuant achieves this with an elegant two-stage approach: use the MSE quantizer to get close, then apply a 1-bit correction to remove the bias."

---

## Speaker Notes

- **Open with "we're not done"** to reset the audience's attention. After Section 9's satisfying conclusion ("near-optimal!"), this section introduces conflict. The story has a second act.
- **The 1-bit bias calculation** is the most concrete demonstration. Walk through the math: centroids at ±0.080, quantized vector length = √(2/π) ≈ 0.798, inner products scaled by 2/π ≈ 0.637. The "36% underestimate" should land hard.
- **The non-uniform bias is the key insight.** Spend time on this. Many audience members will immediately think "but softmax handles uniform scaling!" — and they're right. The dangerous part is that the bias varies with the true inner product value. This distorts relative rankings, not just absolute values.
- **Don't over-explain.** The bias problem is a setup for the solution (Section 11). Present the problem clearly, establish that it's serious, and move to the fix. Don't dwell on mathematical details of exactly how the bias varies.
- **The analogy of "regression to the mean"** helps engineers understand why quantization shrinks vectors. Each coordinate gets pulled toward its bucket's centroid, which is closer to zero than the coordinate's true value. This is a familiar statistical concept.
- **Possible audience questions:**
  - "At 4 bits the bias is only 0.7% — why bother fixing it?" — For many practical applications, 4-bit MSE-only is fine. The fix matters most at 2-3 bits where compression is highest. Community implementations have actually found MSE-only works better than the two-stage approach in some settings.
  - "Can't you just scale up the reconstructed vectors to fix the bias?" — You could compensate for the average bias, but the bias varies per inner product and per vector. A global scale factor doesn't fix the non-uniform part.
  - "Is bias or variance worse for attention?" — This is a great question and actually debated. The paper prioritizes unbiasedness, but community experiments suggest that at 3+ bits, variance from QJL (Stage 2) can hurt more than the small residual bias. We'll discuss this in Section 15.
- **Transition to Section 11:** "So we need to remove the bias without adding too many extra bits. The solution is beautifully simple: quantize the error, not just the vector."
