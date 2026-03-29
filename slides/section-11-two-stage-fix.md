# Section 11: The Two-Stage Fix — MSE Quantizer + QJL on the Residual

**Duration:** 5 minutes  
**Goal:** Show how TurboQuant achieves unbiased inner product estimation by composing the MSE quantizer with a 1-bit error corrector. The audience should understand: what the residual is, how QJL works at an intuitive level, why composing the two stages produces an unbiased estimator, and how the bit budget works out.

---

## The Idea in One Sentence

> "Use most of your bit budget to get close (MSE quantizer), then spend 1 bit per coordinate to correct the bias on whatever error remains."

---

## Stage 1: Get Close with MSE Quantizer

We already have TurboQuant_mse from Sections 8-9. For a target bit-width of b, we run the MSE quantizer with **(b-1) bits** instead of b:

```
Input: vector k (unit norm, d-dimensional)

Stage 1: TurboQuant_mse with (b-1) bits
  → Rotate: y = Π · k
  → Quantize each coordinate to (b-1) bits
  → Dequantize to get reconstruction k̃_mse

k̃_mse is close to k, but biased for inner products.
```

Why (b-1) and not b? Because we're saving 1 bit per coordinate for Stage 2.

---

## The Residual: What Stage 1 Got Wrong

The **residual** is the error left over after Stage 1:

```
r = k - k̃_mse
```

This is a d-dimensional vector representing everything the MSE quantizer couldn't capture. Since Stage 1 was a good MSE quantizer, the residual is **small**:

```
‖r‖² = ‖k - k̃_mse‖² = MSE of (b-1)-bit quantizer

For b = 4 (Stage 1 uses 3 bits):
  ‖r‖² ≈ 0.030    →    ‖r‖ ≈ 0.17

The residual has only ~17% of the original vector's energy.
```

Now here's the key observation. The inner product error from Stage 1 comes entirely from the residual:

```
⟨q, k⟩ - ⟨q, k̃_mse⟩ = ⟨q, k - k̃_mse⟩ = ⟨q, r⟩
```

If we could estimate ⟨q, r⟩ without bias, we could add that correction to ⟨q, k̃_mse⟩ and get an unbiased estimate of the true ⟨q, k⟩.

> **The problem reduces to: estimate the inner product with a small residual vector, using just 1 bit per coordinate.**

---

## Stage 2: QJL — 1-Bit Unbiased Inner Products

This is where the **Quantized Johnson-Lindenstrauss (QJL)** transform comes in. QJL is a separate algorithm (published at AAAI 2025 by the same research group) that does one thing:

> **Given any vector, produce a 1-bit-per-coordinate sketch that gives unbiased inner product estimates.**

### How QJL Works

```
Input: residual vector r (d-dimensional)

Step 1: Multiply by a random Gaussian matrix S (d × d)
        z = S · r

Step 2: Take the sign of each entry
        qjl = sign(z)    → d values, each ±1 → 1 bit per coordinate

Step 3: Store qjl and ‖r‖₂ (the residual's norm)
```

To estimate an inner product ⟨q, r⟩ from the sketch:

```
Estimate of ⟨q, r⟩  =  ‖r‖ × √(π/2)/d × ⟨Sᵀ · qjl, q⟩
```

### Why Is This Unbiased?

The mathematical proof is in the paper, but the intuition is:

```
Each entry of z = S · r is a random projection of r.
Taking the sign preserves the DIRECTION of each projection.

The sign of a random projection of r, correlated with the
same random projection of q, gives an unbiased estimate
of the angle between r and q.

The angle between two vectors determines their inner product.
```

Think of it like this: if r and q point in similar directions, most random projections of r will have the same sign as the corresponding projections of q. If they point in opposite directions, the signs will mostly disagree. The fraction of agreements is a noisy but unbiased estimate of their alignment.

> **QJL trades precision for unbiasedness.** Each bit carries very little information, but what it carries is correct on average.

---

## Putting the Two Stages Together

The complete TurboQuant_prod algorithm:

```
QUANTIZE (bit-width b):
  ┌────────────────────────────────────────────────────────┐
  │                                                        │
  │  Input: vector k                                       │
  │                                                        │
  │  Stage 1: TurboQuant_mse with (b-1) bits               │
  │    → idx = MSE quantize(k)                             │
  │    → k̃_mse = MSE dequantize(idx)                      │
  │                                                        │
  │  Compute residual:                                     │
  │    → r = k - k̃_mse                                    │
  │    → γ = ‖r‖₂                                         │
  │                                                        │
  │  Stage 2: QJL on residual (1 bit per coordinate)       │
  │    → qjl = sign(S · r)                                │
  │                                                        │
  │  Store: (idx, qjl, γ)                                  │
  │         (b-1)d bits + d bits + 16 bits                 │
  │         ≈ bd total bits per vector                      │
  │                                                        │
  └────────────────────────────────────────────────────────┘

DEQUANTIZE:
  ┌────────────────────────────────────────────────────────┐
  │                                                        │
  │  Input: (idx, qjl, γ)                                  │
  │                                                        │
  │  Stage 1 reconstruction:                               │
  │    → k̃_mse = MSE dequantize(idx)                      │
  │                                                        │
  │  Stage 2 reconstruction:                               │
  │    → k̃_qjl = γ × √(π/2)/d × Sᵀ · qjl                │
  │                                                        │
  │  Combined:                                             │
  │    → k̃ = k̃_mse + k̃_qjl                               │
  │                                                        │
  └────────────────────────────────────────────────────────┘
```

---

## Why the Combination Is Unbiased

The proof is surprisingly clean. Let's trace through the expectation:

```
E[⟨q, k̃⟩] = E[⟨q, k̃_mse + k̃_qjl⟩]
           = E[⟨q, k̃_mse⟩] + E[⟨q, k̃_qjl⟩]
                                    ↑
                        QJL is unbiased for ⟨q, r⟩ (Lemma 4)
           = E[⟨q, k̃_mse⟩] + ⟨q, r⟩
           = E[⟨q, k̃_mse⟩] + ⟨q, k - k̃_mse⟩
           = E[⟨q, k̃_mse⟩] + ⟨q, k⟩ - E[⟨q, k̃_mse⟩]
           = ⟨q, k⟩  ✓
```

The bias from Stage 1 is **exactly cancelled** by QJL's unbiased estimate of the residual's inner product. The MSE quantizer's bias doesn't matter because QJL compensates for whatever error Stage 1 introduces.

> **It doesn't matter that Stage 1 is biased. It only matters that Stage 1 makes the residual small (low MSE), and Stage 2 handles the residual without bias.**

---

## The Variance: Why Stage 1 Still Matters

If QJL is unbiased by itself, why not just use QJL alone (skip Stage 1 entirely)?

Because QJL has **variance**, and that variance is proportional to the input vector's squared norm:

```
Var[QJL estimate of ⟨q, r⟩] ≤ (π/2d) × ‖r‖² × ‖q‖²
```

If you apply QJL directly to k (with ‖k‖ = 1), the variance is π/(2d). Not terrible, but not great.

If you first reduce k to a small residual r (with ‖r‖² ≈ 0.03 at 3-bit Stage 1), the variance drops to:

```
Variance with Stage 1:   (π/2d) × 0.03 × ‖q‖²
Variance without Stage 1: (π/2d) × 1.0  × ‖q‖²

Ratio: 0.03 / 1.0 = 33× less variance with Stage 1!
```

> **Stage 1 (MSE quantizer) makes the residual small. Stage 2 (QJL) handles the residual without bias. Together, you get low variance AND zero bias.**

This is why it's a **two-stage** algorithm, not just QJL alone.

---

## The JPEG Analogy

For engineers, the closest familiar parallel is **JPEG compression**:

```
JPEG:
  Stage 1: DCT + coarse quantization     → captures the broad structure
  Stage 2: Fine detail encoding           → corrects the high-frequency errors

TurboQuant:
  Stage 1: Rotation + Lloyd-Max (b-1 bits) → captures the vector's structure
  Stage 2: QJL on residual (1 bit)         → corrects the inner product bias
```

In both cases:
- Stage 1 does the heavy lifting (most of the compression, most of the quality)
- Stage 2 adds a lightweight correction for a specific property Stage 1 doesn't guarantee
- The total cost is barely more than Stage 1 alone

---

## The Complete Distortion Guarantee

Theorem 2 from the paper: for b bits total, TurboQuant_prod achieves:

```
Unbiasedness:   E[⟨q, k̃⟩] = ⟨q, k⟩                     ✓

Inner product distortion:

                    √(3π)    ‖q‖²     1
    Dₚᵣₒ  ≤  ────── × ───── × ────
                 2       d       4ᵇ
```

Compared to the lower bound of (1/d) × (1/4^b), TurboQuant_prod is again within a constant factor of 2.7×.

In concrete terms:

```
┌───────────┬────────────────────────────┐
│ Bit-width │ Inner product distortion   │
│    b      │ (×‖q‖²)                   │
├───────────┼────────────────────────────┤
│    1      │  1.57 / d                  │
│    2      │  0.56 / d                  │
│    3      │  0.18 / d                  │
│    4      │  0.047 / d                 │
└───────────┴────────────────────────────┘

For d = 128:
  b = 3:  distortion ≈ 0.0014 × ‖q‖²    (tiny!)
  b = 4:  distortion ≈ 0.00037 × ‖q‖²   (negligible)
```

> The 1/d factor is key — higher dimensions give **better** inner product estimates, which is exactly the regime where KV caches operate (d = 64 to 256 per head).

---

## Summary of the Full Algorithm

```
TurboQuant — the complete picture:

  ONE-TIME SETUP:
    1. Generate random rotation matrix Π
    2. Generate random Gaussian matrix S (for QJL)
    3. Precompute Lloyd-Max codebook for bit-width (b-1)

  QUANTIZE (per vector):
    1. Rotate:    y = Π · k
    2. Quantize:  idx = nearest centroids for each coordinate of y  [(b-1) bits]
    3. Residual:  r = k - dequantize(idx)
    4. QJL:       qjl = sign(S · r), store ‖r‖                    [1 bit + scalar]

  DEQUANTIZE (per vector):
    1. Lookup:    ỹ = centroids from idx
    2. Unrotate:  k̃_mse = Πᵀ · ỹ
    3. QJL:       k̃_qjl = ‖r‖ × √(π/2)/d × Sᵀ · qjl
    4. Combine:   k̃ = k̃_mse + k̃_qjl

  PROPERTIES:
    ✓ Unbiased inner products
    ✓ Near-optimal distortion (within 2.7×)
    ✓ Online (no data dependence)
    ✓ GPU-friendly (matrix multiplies + table lookups)
    ✓ b bits per coordinate total
```

---

## Speaker Notes

- **"Quantize the error, not just the vector"** — this framing immediately makes the two-stage approach feel natural. Don't present it as a complicated composition; present it as the obvious strategy of fixing what Stage 1 got wrong.
- **The residual is the bridge.** Make sure the audience sees that r = k - k̃_mse is the link between the two stages. Stage 1 minimizes ‖r‖. Stage 2 handles ⟨q, r⟩ without bias. These are complementary goals.
- **The unbiasedness proof** is worth showing — it's only 6 lines and the cancellation is satisfying. The moment where E[⟨q, k̃_mse⟩] appears with both + and - signs and cancels out is a genuine "aha."
- **The "why not just QJL alone?" question** anticipates a sharp audience member. The answer (variance scales with ‖r‖², and Stage 1 makes ‖r‖² small) is elegant and shows why both stages are necessary.
- **The JPEG analogy** works well for this audience. Most engineers have some intuition about how JPEG works (coarse structure + fine details). Map it explicitly: DCT ↔ rotation, coarse quantization ↔ Lloyd-Max, fine details ↔ QJL.
- **Don't dwell on the QJL internals.** "Multiply by a random matrix, take signs" is sufficient. The audience needs to know that QJL exists and is unbiased, not how the proof works. If someone asks, point them to the QJL paper.
- **The 1/d factor** in the distortion bound deserves one sentence: "Higher dimensions actually help — each coordinate contributes less noise, so the average is more accurate. d=128 is plenty for the inner product estimates to be excellent."
- **Possible audience questions:**
  - "Doesn't the random matrix S for QJL cost a lot of memory?" — S is d × d, so 128 × 128 × 4 bytes = 64 KB. Negligible. Or use a structured random matrix (random Hadamard) for even less.
  - "Two matrix multiplies per quantize/dequantize — isn't that expensive?" — It's O(d²) per vector, but d = 128 is small. On a GPU, this is a tiny matmul that takes microseconds. The paper shows negligible runtime overhead.
  - "What about the norm ‖r‖ — doesn't storing that cost extra bits?" — Yes, it's a single float16 per vector (16 bits). For d=128, that's 16/128 = 0.125 extra bits per coordinate. Negligible.
  - "In practice, should I use TurboQuant_mse or TurboQuant_prod?" — For KV cache at 3+ bits, community experiments suggest MSE-only often works as well or better, because QJL's variance gets amplified by softmax. The prod variant is theoretically cleaner but the mse variant is simpler and sometimes better in practice. This is discussed in Section 15.
- **Transition to Section 12:** "That's the complete algorithm. Now let's see if the theory holds up — do the experiments actually show what the math predicts?"
