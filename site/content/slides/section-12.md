---
title: "Empirical Validation — Theory Meets Practice"
weight: 12
part: "Part V — Does It Actually Work?"
---

![Empirical Validation](/img/s12-empirical.webp)

We've built up the theory across several sections. Does it hold up on real data? The paper validates on the **DBpedia Entities dataset** -- 100,000 real-world text embeddings from OpenAI's embedding model (1536 dimensions), with 1,000 separate query vectors for inner product accuracy.

---

## Every Theoretical Prediction Is Confirmed

The paper tests four specific predictions, and all four match experiment:

| What the theory predicted | What the experiments showed |
|---|---|
| MSE $\leq 2.72 \times 4^{-b}$ | Measured MSE falls between upper and lower bounds at every bit-width (1 through 5). The constants match, not just the order of magnitude. |
| Each bit reduces error by 4x | Consistent ~4x reduction observed at each step |
| TurboQuant_prod is unbiased | Inner product error distribution is symmetric and centered at zero |
| TurboQuant_mse has bias that grows with true IP | Error shifts rightward (underestimates), and the shift is larger for higher true inner products -- exactly the non-uniform bias from Section 10 |

---

## MSE-Only vs Two-Stage: The Bias Is Visible

The clearest result: when you plot the inner product error distribution for both variants, TurboQuant_prod gives a symmetric bell curve centered at zero (unbiased), while TurboQuant_mse gives a bell curve shifted to the right (biased -- systematically underestimates inner products).

At 2-bit quantization, the bias pattern is stark:

```
TurboQuant_prod:                    TurboQuant_mse:

  All IP magnitudes → error          Low IPs  → small bias
  centered at zero                   High IPs → large bias
  (unbiased ✓)                       (non-uniform bias ✗)
```

At higher bit-widths (3-4 bits), the MSE variant's bias shrinks and becomes harder to distinguish from noise -- foreshadowing the practical finding in Section 15 that MSE-only is often sufficient.

> **The theory isn't aspirational -- it's descriptive.** Every prediction matches the measurement. But vector-level metrics are just a proxy. The real test is: does the model still give correct answers?
