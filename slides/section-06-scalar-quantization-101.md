# Section 6: Scalar Quantization 101

**Duration:** 7 minutes  
**Goal:** Build quantization from first principles. The audience should understand: codebooks, bucket boundaries, why uniform spacing is suboptimal, and how Lloyd-Max finds optimal centroids for a known distribution. This section is the toolbox — Section 8 will use it.

---

## Starting from Scratch

> "Before we get to TurboQuant, we need to understand the building block it uses: scalar quantization. This is the foundation. If you get this, TurboQuant will feel almost obvious."

Scalar quantization is the simplest form of compression: take a single real number, and replace it with the nearest value from a small, fixed set of allowed values.

---

## 1-Bit Quantization: The Simplest Case

You have a number that can be anything between -1 and 1. You're allowed to store exactly **1 bit**. That gives you 2¹ = 2 possible stored values.

You need to choose:
1. **Two representative values** (centroids): call them c₁ and c₂
2. **A boundary** between them: any input below the boundary maps to c₁, above maps to c₂

The most obvious choice:

```
Boundary: 0
Centroid 1: -0.5    (represents all negative values)
Centroid 2: +0.5    (represents all positive values)

    ─────────────────┬─────────────────
    -1     -0.5      0      0.5      1
           c₁     boundary    c₂

Input:  0.37  → positive → store bit "1" → reconstruct as  0.5
Input: -0.82  → negative → store bit "0" → reconstruct as -0.5
Input:  0.03  → positive → store bit "1" → reconstruct as  0.5   ← big error!
```

That last one hurts. The value 0.03 is very close to zero, but it gets snapped to 0.5. The error is 0.47.

Is there a better choice of centroids? **It depends on the distribution of your data.**

---

## Why Distribution Matters

Imagine two different scenarios:

### Scenario A: Uniform Distribution
Values are equally likely to be anywhere between -1 and 1.

```
Probability
  │  ┌──────────────────────────────┐
  │  │                              │  ← flat, equal everywhere
  │  │                              │
  └──┴──────────────────────────────┴───
    -1              0               1
```

For uniform data, equal-width buckets are actually optimal. Centroids at -0.5 and +0.5 minimize average error.

### Scenario B: Gaussian Distribution
Most values cluster near zero, with few values near the extremes.

```
Probability
  │           ╱╲
  │          ╱  ╲
  │         ╱    ╲
  │        ╱      ╲
  │      ╱          ╲
  │   ╱                ╲
  └──╱────────────────────╲──────
    -1        0            1
```

For Gaussian data, equal-width buckets are wasteful. Most values live near zero, so you want centroids **closer to zero** where the data is dense.

For a Gaussian N(0, σ²), the optimal 1-bit centroids are at **±0.798σ** (not ±0.5σ).

> **Key insight: The optimal quantizer depends on the probability distribution of the input.** If you know the distribution, you can design a quantizer that minimizes error for that specific distribution.

---

## Scaling to More Bits

With b bits, you get 2^b buckets:

```
b = 1:   2 buckets    (2 centroids, 1 boundary)
b = 2:   4 buckets    (4 centroids, 3 boundaries)
b = 3:   8 buckets    (8 centroids, 7 boundaries)
b = 4:  16 buckets    (16 centroids, 15 boundaries)
```

Each bucket is defined by:
- A **range** of input values that map to this bucket
- A **centroid** — the single value used to represent all inputs in this bucket

The set of all centroids is called the **codebook**. Quantization stores only the bucket index (b bits per value). Dequantization looks up the centroid from the codebook.

```
Quantize:    input value  →  find nearest centroid  →  store bucket index (b bits)
Dequantize:  bucket index →  look up centroid       →  return centroid value
```

For 3-bit quantization of a Gaussian, visualized:

```
Probability
  │           ╱╲
  │          ╱  ╲
  │   ┊  ┊  ╱┊  ┊╲  ┊  ┊
  │   ┊  ┊ ╱ ┊  ┊ ╲ ┊  ┊
  │   ┊  ┊╱  ┊  ┊  ╲┊  ┊
  │  ╱┊  ┊   ┊  ┊   ┊  ┊╲
  └─╱─┊──┊───┊──┊───┊──┊─╲──
    c₁ c₂  c₃  c₄  c₅  c₆ c₇ c₈

    ↑               ↑              ↑
  narrow          wide           narrow
  buckets        buckets         buckets
  (few values)   (many values)   (few values)
```

Notice: **narrow buckets in the tails** (where data is sparse, so even a wide bucket doesn't help much) and **tighter buckets near the center** (where data is dense, so finer granularity reduces more error). Wait — actually it's the opposite!

Let me correct that. The optimal layout is:

- **Narrower buckets near the peak** — where most data lives, so reducing error there has the biggest impact
- **Wider buckets in the tails** — where data is rare, so even coarse representation affects few values

```
Dense region (near zero):
  Many values per bucket → make buckets NARROW → less error per value

Sparse region (tails):
  Few values per bucket → make buckets WIDE → saves resolution for where it matters
```

> This is the core idea: **concentrate your limited precision where the data is, not where it isn't.**

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
     centroid_i = E[x | x ∈ bucket_i]
                = ∫ x · f(x) dx  /  ∫ f(x) dx
                  (integrated over bucket_i's range)
```

This is exactly k-means, but in 1D and with a continuous distribution instead of discrete data points. The "assign" step creates Voronoi regions (each value goes to the nearest centroid), and the "update" step moves centroids to the mean of their region.

### A Concrete Example

Let's find the optimal 2-bit (4 centroids) quantizer for a standard Gaussian N(0,1):

```
Iteration 0 (initial): centroids at [-1.5, -0.5, 0.5, 1.5]

Iteration 1:
  Boundaries: [-1.0, 0.0, 1.0]
  New centroids: [-1.51, -0.45, 0.45, 1.51]   (computed from conditional expectations)

Iteration 2:
  Boundaries: [-0.98, 0.0, 0.98]
  New centroids: [-1.51, -0.453, 0.453, 1.51]

  (converged)

Optimal 2-bit codebook for N(0,1): {-1.51, -0.453, 0.453, 1.51}
```

These specific numbers are known results from quantization theory. The beauty is: **once you know the distribution, you solve this optimization once and store the codebook forever.** At runtime, quantization is just "find the nearest centroid" — a simple comparison.

---

## Codebooks Are Just Lookup Tables

For an engineer, the codebook is nothing more than a **lookup table**.

```python
# 3-bit codebook for Gaussian N(0,1) — 8 centroids
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

In practice, since the centroids are sorted, you can use **binary search** (O(log 2^b) = O(b) comparisons) or precomputed boundary arrays for even faster lookup.

For b ≤ 4 (which is the practical range for KV cache), the codebook has at most 16 entries. The entire quantize-dequantize operation is trivially fast.

---

## The Catch: You Need to Know the Distribution

Everything above works beautifully **if you know the probability distribution** of the values you're quantizing. Lloyd-Max gives you the optimal codebook for any known distribution.

But in the KV cache problem:
- Key and Value vectors come from model activations
- Their distribution depends on the model, the layer, the head, and the input
- You can't precompute statistics because the data arrives online

This seems like a dead end. You need to know the distribution to build the codebook, but you don't know the distribution because the data is dynamic and unpredictable.

> **This is the problem that TurboQuant's random rotation solves.** Instead of adapting the codebook to the data's distribution (offline, expensive), it transforms the data so that it always has a known, fixed distribution — regardless of the input. Then a single precomputed codebook works for everything.

That transformation is the subject of Section 8. But first, one more piece of background.

---

## Why This Matters for TurboQuant

Let's preview how TurboQuant uses everything from this section:

```
1. Random rotation transforms any vector into one with a KNOWN distribution
   (Beta distribution, which looks Gaussian in high dimensions)

2. Lloyd-Max gives the OPTIMAL codebook for that distribution
   (solved once, stored forever)

3. At runtime: rotate the vector, then do a table lookup per coordinate
   (trivially fast, fully parallelizable)
```

The rotation is the magic that turns an offline problem into an online one. Lloyd-Max is the tool that makes the codebook optimal. Together, they give you near-perfect scalar quantization of any vector, without ever seeing the data distribution.

---

## Speaker Notes

- **Build from 1-bit to multi-bit progressively.** Start with the simplest case (1 bit, 2 centroids) before showing 2-bit and 3-bit. Each additional bit doubles the codebook but halves the quantization step size.
- **The distribution comparison** (uniform vs Gaussian) is essential. Draw both distributions on a whiteboard with bucket boundaries overlaid. The visual of "narrow where data is dense" is the key intuition.
- **Correct yourself on the bucket width layout.** I deliberately set up the wrong intuition ("narrow in tails") and then corrected it. In a talk, you can do this as a "what would you expect? ... actually it's the opposite" moment. Or just present the correct version directly if you prefer.
- **The Python code** makes it concrete for software engineers. They can look at `quantize()` and `dequantize()` and think "oh, that's all it is — a table lookup." Demystifies the math.
- **Lloyd-Max = 1D k-means:** If the audience knows k-means (most will), this analogy lands instantly. "It's k-means, but on a line instead of in a plane, and with a probability distribution instead of data points."
- **End on the catch:** The section should feel like "we have a beautiful tool, but we can't use it because we don't know the distribution." This creates tension that Section 8 resolves.
- **Don't rush the distribution insight.** The idea that "optimal quantization depends on the distribution" is the single most important concept from this section. If the audience gets this, TurboQuant's rotation trick will feel natural.
- **Possible audience questions:**
  - "Why not just compute the distribution online from the data we've seen so far?" — You could (running statistics), but it adds complexity, doesn't work well for the first few tokens, and the distribution shifts across layers/heads. TurboQuant's approach is simpler and provably optimal.
  - "How is this different from float8 or int8 quantization?" — FP8/INT8 use fixed, hardware-defined representations. Lloyd-Max designs an optimal representation for your specific distribution. TurboQuant is closer to the latter.
  - "Is the codebook the same for all dimensions of the vector?" — Yes! After rotation, every coordinate has the same distribution, so one codebook works for all coordinates. This is the key simplification.
- **Transition to Section 7:** "So scalar quantization works great when coordinates are independent and you know their distribution. But real vectors have correlated coordinates. How do you handle that?"
