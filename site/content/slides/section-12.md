---
title: "Empirical Validation — Theory Meets Practice"
weight: 12
part: "Part V — Does It Actually Work?"
---

![Empirical Validation](/img/s12-empirical.webp)

We've spent considerable time building up the theory. Beautiful math -- but does it actually work on real data?

---

## Setup

The paper validates on the **DBpedia Entities dataset** -- 100,000 real-world text embeddings produced by OpenAI's embedding model, in 1536 dimensions. These aren't synthetic vectors -- they're the kind you'd actually store in a vector database. They use 1,000 separate query vectors to measure inner product accuracy.

---

## Result 1: Distortion Matches the Bounds

Measured MSE sits **right between the upper and lower bounds** at every bit-width from 1 to 5. The theory doesn't just give the right order of magnitude -- it gives the right constants.

> **The theory is tight.** What the math predicts is what you get in practice.

---

## Result 2: Unbiasedness Is Confirmed

The distribution of inner product errors across all 100M vector pairs:

- **TurboQuant_prod**: error distribution centered at **zero** -- symmetric, confirming unbiasedness at every bit-width
- **TurboQuant_mse**: error distribution **shifted right** -- systematically positive (inner products underestimated), confirming the bias predicted in Section 10. The shift decreases with higher bit-widths, exactly as theory predicts.

---

## Result 3: Bias Depends on Inner Product Magnitude

At 2-bit quantization, grouping vector pairs by their true inner product:

```
TurboQuant_prod:
  Avg IP = 0.01  → error centered at 0
  Avg IP = 0.06  → error centered at 0
  Avg IP = 0.10  → error centered at 0       ← unbiased regardless of IP
  Avg IP = 0.17  → error centered at 0

TurboQuant_mse:
  Avg IP = 0.01  → error centered near 0
  Avg IP = 0.06  → error shifted slightly right
  Avg IP = 0.10  → error shifted more right   ← bias grows with
  Avg IP = 0.17  → error shifted even more       true inner product
```

This confirms the non-uniform bias from Section 10: TurboQuant_mse's bias grows with the true IP value, while TurboQuant_prod's error is independent of it.

---

## The Bottom Line

```
Theory said:                              Experiments confirm:
─────────────                             ─────────────────────
MSE ≤ 2.72 × 4⁻ᵇ                         Measured MSE falls between bounds  ✓
Inner product estimator is unbiased       Error centered at zero             ✓
MSE quantizer has bias ∝ true IP          Bias grows with IP magnitude       ✓
Each bit reduces error by ~4×             Consistent 4× reduction observed   ✓
```

The theory isn't aspirational -- it's descriptive. Every prediction matches the measurement.

Now let's see what this means for actual LLM performance -- not just vector-level metrics, but end-to-end generation quality.
