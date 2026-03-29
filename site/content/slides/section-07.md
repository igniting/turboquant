---
title: "From Scalar to Vector Quantization — The Independence Question"
weight: 7
part: "Part III — Quantization Fundamentals"
---

![From Scalar to Vector Quantization](/img/s07-vector-quant.webp)

In Section 6, we learned how to optimally quantize a single number. The obvious approach for a vector is: **quantize each coordinate independently**. But is it optimal?

---

## When Coordinates Are Correlated

In real data, vector coordinates are rarely independent. Consider word embeddings:

```
Hypothetical pattern in embedding space:
  When coordinate 3 is positive, coordinate 7 tends to be negative.
  When coordinate 3 is negative, coordinate 7 tends to be positive.
```

Visually, if you plot coordinate 3 vs coordinate 7, instead of a circular cloud (independent), you see an elongated ellipse (correlated):

```
Independent coordinates:           Correlated coordinates:

  coord 7                            coord 7
    |    . . . .                       |       .  .
    |  . . . . . .                     |     .  .
    |  . . . . . .                     |   .  .
    |  . . . . . .                     |  . .
    |  . . . . . .                     | . .
    |    . . . .                       |. .
    +---------------                   +---------------
         coord 3                            coord 3
```

When coordinates are correlated, a **joint** quantizer can exploit this -- designing a grid that follows the shape of the data instead of wasting precision on empty corners.

> **Correlated coordinates mean scalar quantization leaves quality on the table.** A joint quantizer that understands the correlation can do better.

---

## Why Joint Vector Quantization Is Impossible

The problem is combinatorial explosion. For typical KV cache parameters ($d = 128$, $b = 3$):

```
Scalar codebook:  2^3        =          8 entries     <- trivial
Joint codebook:   2^(3*128)  = 2^384   entries        <- more than atoms in the universe
```

You can't store $2^{384}$ entries. You can't search through them. Joint vector quantization over the full vector is physically impossible.

---

## Product Quantization: The Practical Compromise

**Product Quantization (PQ)** is the standard compromise used in vector databases: **split the vector into small groups** and quantize each group jointly.

```
128-dim vector, split into 16 groups of 8 dimensions each:

  [x1 x2 ... x8 | x9 x10 ... x16 | ... | x121 ... x128]
   --group 1---   ---group 2----         ---group 16---

Each group: quantized jointly using a codebook of 2^b entries
```

PQ captures correlations *within* each group while ignoring correlations *between* groups. But it has a fatal flaw for KV cache: **building the codebook requires k-means clustering on your data**.

```
PQ codebook construction:
  1. Collect a large dataset of vectors
  2. Split each vector into groups
  3. Run k-means (with 256 clusters) on each group
  4. 37-494 seconds of compute (from the TurboQuant paper)
```

PQ is an **offline** method. It needs to see your data before it can quantize. That rules it out for the KV cache.

---

## The Key Question

We're stuck between two extremes:

```
Scalar quantization:  Fast, online, GPU-friendly
                      But SUBOPTIMAL when coordinates are correlated

Joint quantization:   Optimal (captures all correlations)
                      But IMPOSSIBLE (codebook too large)

Product quantization: Reasonable compromise
                      But OFFLINE (needs k-means preprocessing)
```

Here's the key question TurboQuant answers:

> **What if you could transform the vector so that its coordinates become independent?**

If the coordinates are genuinely independent, then scalar quantization per coordinate **is** joint quantization. There's nothing to gain from considering coordinates together because they carry no information about each other.

The reason joint quantization beats scalar quantization is that coordinate correlations let you "borrow" information from one coordinate to quantize another more precisely. If there are no correlations -- if knowing coordinate 3 tells you nothing about coordinate 7 -- then there's nothing to borrow. Scalar quantization is already optimal.

---

## Independence: The Formal Idea

Two coordinates $x_i$ and $x_j$ are **independent** if knowing the value of one gives you no information about the other:

$$P(x_i = a \text{ AND } x_j = b) = P(x_i = a) \times P(x_j = b) \quad \text{for all } a, b$$

> **If all coordinates are independent and identically distributed, then the optimal vector quantizer is simply the optimal scalar quantizer applied to each coordinate.**

This is the theoretical foundation of TurboQuant. The entire algorithm is built to create this condition.

---

## The Roadmap

```
What we know:
  ✓ Scalar quantization is fast, online, and GPU-friendly (Section 6)
  ✓ It's optimal IF coordinates are independent (this section)
  ✓ Lloyd-Max gives the optimal scalar codebook IF you know the distribution (Section 6)

What we need:
  -> A transformation that makes coordinates independent     <- Section 8
  -> A transformation that gives them a KNOWN distribution   <- Section 8
  -> These must work for ANY input vector (worst case)       <- Section 8
```

It turns out that one single operation -- a random rotation -- gives us both independence and a known distribution simultaneously. That's the core insight of TurboQuant.
