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
Variance without Stage 1: (π/2d) × 1.0  × ||q||²   (full vector)

Ratio: 33× less variance with Stage 1!
```

> **Stage 1 makes the residual small. Stage 2 handles it without bias. Together: low variance AND zero bias.**

---

## The JPEG Analogy

```
JPEG:
  Stage 1: DCT + coarse quantization     → captures the broad structure
  Stage 2: Fine detail encoding           → corrects high-frequency errors

TurboQuant:
  Stage 1: Rotation + Lloyd-Max (b-1 bits) → captures the vector's structure
  Stage 2: QJL on residual (1 bit)         → corrects the inner product bias
```

---

## The Complete Distortion Guarantee

Theorem 2: for $b$ bits total, TurboQuant_prod achieves:

- **Unbiasedness:** $E[\langle q, \tilde{k} \rangle] = \langle q, k \rangle$
- **Distortion:** within 2.7x of the information-theoretic lower bound

For $d = 128$ at $b = 3$: inner product distortion $\approx 0.0014 \times \|q\|^2$ -- tiny.

The $1/d$ factor means **higher dimensions give better estimates** -- exactly the regime where KV caches operate ($d = 64$ to $256$ per head).

---

## The Full Algorithm Summary

```
ONE-TIME SETUP:
  1. Generate random rotation matrix Π
  2. Generate random Gaussian matrix S (for QJL)
  3. Precompute Lloyd-Max codebook for bit-width (b-1)

QUANTIZE (per vector):
  1. Rotate:    y = Π · k
  2. Quantize:  idx = nearest centroids [(b-1) bits]
  3. Residual:  r = k - dequantize(idx)
  4. QJL:       qjl = sign(S · r), store ||r||    [1 bit + scalar]

DEQUANTIZE (per vector):
  1. Lookup + Unrotate:  k̃_mse = Πᵀ · centroids(idx)
  2. QJL correction:     k̃_qjl = ||r|| × √(π/2)/d × Sᵀ · qjl
  3. Combine:            k̃ = k̃_mse + k̃_qjl

PROPERTIES:
  ✓ Unbiased inner products
  ✓ Near-optimal distortion (within 2.7×)
  ✓ Online, GPU-friendly
  ✓ b bits per coordinate total
```
