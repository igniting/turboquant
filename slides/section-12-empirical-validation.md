# Section 12: Empirical Validation — Theory Meets Practice

**Duration:** 3 minutes  
**Goal:** Show that the theoretical bounds from Sections 9-11 aren't just math — they predict real-world behavior with high accuracy. Quick, visual, confidence-building.

---

## The Question

> "We've spent the last 25 minutes building up the theory: rotation makes coordinates predictable, Lloyd-Max gives optimal codebooks, the two-stage approach removes bias, and the distortion is within 2.7× of the information-theoretic limit. Beautiful math. But does it actually work on real data?"

---

## Setup

The paper validates on the **DBpedia Entities dataset** — 100,000 real-world text embeddings produced by OpenAI's embedding model, encoded in 1536 dimensions. These aren't synthetic vectors designed to make TurboQuant look good — they're the kind of vectors you'd actually store in a vector database or encounter as KV cache entries.

They also use 1,000 separate query vectors to measure inner product accuracy.

---

## Result 1: Distortion Matches the Bounds

Figure 3 from the paper plots measured MSE and inner product error against the theoretical upper and lower bounds across bit-widths 1 through 5.

```
MSE (log scale)

  10⁻¹ │╲
       │  ╲  ← Upper bound: √(3π)/2 × 4⁻ᵇ
       │    ╲
       │  ●   ╲  ← TurboQuant_mse (measured)
  10⁻² │    ●   ╲
       │      ╲   ╲
       │        ●   ╲
       │          ╲   ╲  ← Lower bound: 4⁻ᵇ
  10⁻³ │            ●   ╲
       │              ╲   ╲
       │                ●   ╲
       └──────────────────────
         1    2    3    4    5
              Bit-width (b)
```

The measured values (●) sit **right between the upper and lower bounds** at every bit-width. The theory doesn't just give the right order of magnitude — it gives the right constants.

> **The theory is tight.** What the math predicts is what you get in practice. No surprises, no hidden constants.

---

## Result 2: Unbiasedness Is Confirmed

Figure 1 from the paper shows the distribution of inner product errors across all 100,000 × 1,000 vector pairs.

```
TurboQuant_prod:                    TurboQuant_mse:

  Frequency                           Frequency
    │      ╱╲                           │        ╱╲
    │     ╱  ╲                          │       ╱  ╲
    │    ╱    ╲                         │      ╱    ╲
    │   ╱      ╲                        │     ╱      ╲
    │  ╱        ╲                       │    ╱        ╲
    └──────────────                     └──────────────
     -0.1  0  +0.1                      -0.1  0  +0.1
         ↑                                    ↑
    Centered at ZERO                   Shifted to the RIGHT
    (unbiased ✓)                       (biased — underestimates IPs)
```

Two clear observations:

1. **TurboQuant_prod is centered at zero** — the inner product errors are symmetric around zero, confirming unbiasedness at every bit-width tested.

2. **TurboQuant_mse is shifted right** — the errors are systematically positive (meaning the estimated inner products are smaller than the true values), confirming the bias we predicted in Section 10. The shift decreases with higher bit-widths, exactly as the theory predicts.

---

## Result 3: Bias Depends on Inner Product Magnitude

Figure 2 shows a subtler result. At 2-bit quantization, they group vector pairs by their true average inner product and plot the error distribution for each group.

```
TurboQuant_prod:
  Avg IP = 0.01  → error centered at 0
  Avg IP = 0.06  → error centered at 0
  Avg IP = 0.10  → error centered at 0       ← variance constant
  Avg IP = 0.17  → error centered at 0          regardless of IP value

TurboQuant_mse:
  Avg IP = 0.01  → error centered near 0
  Avg IP = 0.06  → error shifted slightly right
  Avg IP = 0.10  → error shifted more right   ← bias grows with
  Avg IP = 0.17  → error shifted even more       true inner product
```

This confirms the non-uniform bias we discussed in Section 10: TurboQuant_mse's bias grows with the true inner product value, while TurboQuant_prod's error is independent of the true value.

> **For TurboQuant_prod, the error is pure noise — no systematic pattern. For TurboQuant_mse, the error is noise plus a systematic shift that grows with the inner product. This is exactly what the theory predicted.**

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

The theory isn't aspirational — it's descriptive. Every prediction matches the measurement.

> "Now let's see what this means for actual LLM performance — not just vector-level metrics, but end-to-end generation quality."

---

## Speaker Notes

- **Keep this section fast.** 3 minutes max. The audience has been in theory-land for 25 minutes. They want to see results, and the theoretical validation is a bridge to the KV cache experiments, not a destination.
- **The key visual is Figure 3** — measured distortion between upper and lower bounds. If you're making slides, this is worth reproducing or screenshotting from the paper. The tight fit between theory and experiment is visually compelling.
- **The bias visualization** (centered vs shifted histograms) is something you can sketch quickly. Two bell curves — one centered at zero, one shifted right. Label them prod and mse. Done.
- **Don't recite all the numbers.** The point is "theory matches practice." Show one or two data points that confirm this and move on.
- **The transition to Section 13 is important.** The audience should feel: "Okay, the vector-level math works. But does the model still produce good outputs?" That's the real test.
- **Possible audience questions:**
  - "These are embedding vectors, not actual KV cache vectors. Is that a fair test?" — Fair concern. The KV cache experiments in Sections 13-14 test on actual model inference. The embedding experiment validates the mathematical properties; the KV cache experiments validate end-to-end quality.
  - "What dataset would make TurboQuant fail?" — Since TurboQuant is data-oblivious (the guarantees hold for worst-case inputs), there's no adversarial dataset that breaks it. The bounds hold for any data. That said, practical quality depends on the model and task, which is why the KV cache experiments matter.
- **Transition to Section 13:** "Vector-level metrics are clean, but what we really care about is: does the model still give correct answers when its KV cache is compressed 4-5×?"
