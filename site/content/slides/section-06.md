---
title: "Scalar Quantization 101"
weight: 6
part: "Part III — Quantization Fundamentals"
---

![Scalar Quantization 101](/img/s06-scalar-quant.webp)

Before we get to TurboQuant, we need to understand the building block it uses: scalar quantization. This is the foundation. If you get this, TurboQuant will feel almost obvious.

Scalar quantization is the simplest form of compression: take a single real number, and replace it with the nearest value from a small, fixed set of allowed values.

---

## 1-Bit Quantization: The Simplest Case

You have a number that can be anything between -1 and 1. You're allowed to store exactly **1 bit**. That gives you $2^1 = 2$ possible stored values.

You need to choose:
1. **Two representative values** (centroids): $c_1$ and $c_2$
2. **A boundary** between them: any input below the boundary maps to $c_1$, above maps to $c_2$

The most obvious choice:

```
Boundary: 0
Centroid 1: -0.5    (represents all negative values)
Centroid 2: +0.5    (represents all positive values)

    -----------------+------------------
    -1     -0.5      0      0.5      1
           c1     boundary    c2

Input:  0.37  -> positive -> store bit "1" -> reconstruct as  0.5
Input: -0.82  -> negative -> store bit "0" -> reconstruct as -0.5
Input:  0.03  -> positive -> store bit "1" -> reconstruct as  0.5   <- big error!
```

That last one hurts. The value 0.03 is very close to zero, but it gets snapped to 0.5. The error is 0.47.

Is there a better choice of centroids? **It depends on the distribution of your data.**

---

## Why Distribution Matters

### Scenario A: Uniform Distribution
Values are equally likely to be anywhere between -1 and 1. For uniform data, equal-width buckets are actually optimal.

### Scenario B: Gaussian Distribution
Most values cluster near zero, with few values near the extremes.

```
Probability
  |           /\
  |          /  \
  |         /    \
  |        /      \
  |      /          \
  |   /                \
  +--/--------------------\------
    -1        0            1
```

For Gaussian data, equal-width buckets are wasteful. Most values live near zero, so you want centroids **closer to zero** where the data is dense.

For a Gaussian $N(0, \sigma^2)$, the optimal 1-bit centroids are at $\pm 0.798\sigma$ (not $\pm 0.5\sigma$).

> **Key insight: The optimal quantizer depends on the probability distribution of the input.** If you know the distribution, you can design a quantizer that minimizes error for that specific distribution.

---

## Scaling to More Bits

With $b$ bits, you get $2^b$ buckets:

```
b = 1:   2 buckets    (2 centroids, 1 boundary)
b = 2:   4 buckets    (4 centroids, 3 boundaries)
b = 3:   8 buckets    (8 centroids, 7 boundaries)
b = 4:  16 buckets    (16 centroids, 15 boundaries)
```

The set of all centroids is called the **codebook**. Quantization stores only the bucket index ($b$ bits per value). Dequantization looks up the centroid from the codebook.

```
Quantize:    input value  ->  find nearest centroid  ->  store bucket index (b bits)
Dequantize:  bucket index ->  look up centroid       ->  return centroid value
```

> **Concentrate your limited precision where the data is, not where it isn't.** Narrower buckets near the peak (where most data lives), wider buckets in the tails (where data is rare).

---

## The Lloyd-Max Algorithm: Finding Optimal Centroids

Given a known probability distribution, how do you find the optimal centroids and boundaries? This is exactly a **1-dimensional k-means problem**.

The algorithm (published independently by Lloyd in 1957/1982 and Max in 1960):

```
Initialize: place 2^b centroids somehow (e.g., equally spaced)

Repeat until convergence:
  1. ASSIGN: Set each boundary to the midpoint of adjacent centroids
     boundary_i = (centroid_i + centroid_{i+1}) / 2

  2. UPDATE: Set each centroid to the expected value of the data in its bucket
     centroid_i = E[x | x in bucket_i]
```

### A Concrete Example

Optimal 2-bit (4 centroids) quantizer for a standard Gaussian $N(0,1)$:

```
Iteration 0 (initial): centroids at [-1.5, -0.5, 0.5, 1.5]
Iteration 1:           centroids at [-1.51, -0.45, 0.45, 1.51]
Iteration 2:           centroids at [-1.51, -0.453, 0.453, 1.51]  (converged)

Optimal 2-bit codebook for N(0,1): {-1.51, -0.453, 0.453, 1.51}
```

The beauty: **once you know the distribution, you solve this optimization once and store the codebook forever.** At runtime, quantization is just "find the nearest centroid" -- a simple comparison.

---

## Codebooks Are Just Lookup Tables

For an engineer, the codebook is nothing more than a **lookup table**.

```python
# 3-bit codebook for Gaussian N(0,1) -- 8 centroids
CODEBOOK = [-2.15, -1.34, -0.76, -0.25, 0.25, 0.76, 1.34, 2.15]

def quantize(value):
    """Find nearest centroid, return its 3-bit index."""
    best_idx = 0
    best_dist = abs(value - CODEBOOK[0])
    for i in range(1, 8):
        dist = abs(value - CODEBOOK[i])
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx  # 3-bit integer (0-7)

def dequantize(index):
    """Look up centroid from index."""
    return CODEBOOK[index]
```

For $b \leq 4$ (the practical range for KV cache), the codebook has at most 16 entries. The entire quantize-dequantize operation is trivially fast.

---

## The Catch: You Need to Know the Distribution

Everything above works beautifully **if you know the probability distribution** of the values you're quantizing. Lloyd-Max gives you the optimal codebook for any known distribution.

But in the KV cache problem:
- Key and Value vectors come from model activations
- Their distribution depends on the model, the layer, the head, and the input
- You can't precompute statistics because the data arrives online

This seems like a dead end. You need to know the distribution to build the codebook, but you don't know the distribution because the data is dynamic and unpredictable.

> **This is the problem that TurboQuant's random rotation solves.** Instead of adapting the codebook to the data's distribution (offline, expensive), it transforms the data so that it always has a known, fixed distribution -- regardless of the input. Then a single precomputed codebook works for everything.

Here's a preview:

```
1. Random rotation transforms any vector into one with a KNOWN distribution
   (Beta distribution, which looks Gaussian in high dimensions)

2. Lloyd-Max gives the OPTIMAL codebook for that distribution
   (solved once, stored forever)

3. At runtime: rotate the vector, then do a table lookup per coordinate
   (trivially fast, fully parallelizable)
```

But first, one more piece of background.
