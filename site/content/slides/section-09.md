---
title: "The Distortion Guarantee — How Good Is This?"
weight: 9
part: "Part IV — The Algorithm"
---

![The Distortion Guarantee](/img/s09-distortion.webp)

We've seen the algorithm: rotate, quantize, done. But how good is it? Could there be a much better algorithm out there?

The paper answers this definitively -- with both an **upper bound** (how well TurboQuant does) and a **lower bound** (the best anything could ever do).

---

## The Upper Bound: TurboQuant's Guarantee

Theorem 1 from the paper states that for $b$ bits per coordinate, TurboQuant's MSE distortion is at most:

$$\text{MSE} \leq \frac{\sqrt{3\pi}}{2} \times \frac{1}{4^b} \approx 2.72 \times \frac{1}{4^b}$$

What does this mean in practice?

| Bit-width ($b$) | MSE (upper bound) | What this means |
|:---:|:---:|---|
| 1 | 0.36 | 36% of the vector's energy is lost -- rough, but usable |
| 2 | 0.117 | ~12% energy loss -- decent |
| 3 | 0.030 | ~3% energy loss -- very good |
| 4 | 0.009 | <1% energy loss -- nearly perfect |
| 5 | 0.002 | Negligible loss |

The key pattern: **every additional bit reduces error by exactly 4x.**

```
b=1 -> b=2:   0.36  / 0.117 ~ 3.1x  reduction
b=2 -> b=3:   0.117 / 0.030 ~ 3.9x  reduction
b=3 -> b=4:   0.030 / 0.009 ~ 3.3x  reduction
```

This is the $1/4^b$ term at work. It means there are **sharply diminishing returns** -- going from 1 to 2 bits is a massive quality jump, while going from 4 to 5 bits gives you almost nothing. This is why the practical sweet spot is 3-4 bits.

---

## The Lower Bound: The Laws of Physics

The paper also proves (Theorem 3) that **no algorithm in the universe** -- no matter how clever, how slow, how much memory it uses -- can achieve MSE better than:

$$\text{MSE} \geq \frac{1}{4^b}$$

This is an **information-theoretic lower bound**. It comes from Shannon's source coding theory -- the same mathematics that governs how much you can compress any signal. It's a law of nature, like the speed of light. No engineering trick can beat it.

```
With b bits per coordinate, you have 2^b possible reconstruction values.
The unit sphere has "volume" that must be covered by these 2^(b*d) total codewords.
Each codeword "covers" a region of the sphere.
The average volume per region determines the minimum average distance from
any point to its nearest codeword.

Shannon showed this distance is at least 1/4^b.
No codebook design can do better.
```

---

## The Gap: How Close Is TurboQuant to Perfect?

| Bit-width | Lower bound (best possible) | TurboQuant (actual) | Distance from perfection |
|:---:|:---:|:---:|:---:|
| $b = 1$ | 0.25 | 0.36 | 1.44x |
| $b = 2$ | 0.0625 | 0.117 | 1.87x |
| $b = 3$ | 0.0156 | 0.030 | 1.92x |
| $b = 4$ | 0.0039 | 0.009 | 2.31x |
| $b \to \infty$ | $1/4^b$ | $2.72/4^b$ | 2.72x |

At worst, TurboQuant is **2.72x away from the theoretical optimum**. At low bit-widths (the most practically relevant), it's even closer -- only 1.44x at 1 bit.

```
MSE (log scale)
  |
  |  \
  |    \  <- TurboQuant (upper bound)
  |      \
  |        \
  |    \     \
  |      \     \
  |        \     \
  |          \     \
  |            \     \  <- Lower bound (best possible)
  |              \     \
  +------------------------
    1    2    3    4    5
         Bit-width (b)

  The two lines are nearly parallel on a log scale.
  The gap is a constant factor ~ 2.7x.
```

---

## What 2.7x Means in Practice

**The gap is a constant factor, not a function of bit-width.** Many quantization methods have distortion that degrades relative to the optimum as you change parameters. TurboQuant maintains a constant 2.7x gap (or better) everywhere.

**No existing online method comes close.** Uniform scalar quantization (without rotation) has distortion that's exponentially worse than optimal at low bit-widths. The rotation is what closes the gap.

**The gap could theoretically be closed** -- but it would require joint vector quantization (computationally impossible) or data-dependent optimization (violating the online constraint). The 2.7x factor is the price of being online and practical.

**For KV cache, the MSE gap doesn't even matter much.** What matters is downstream model quality, and the experiments show zero quality loss at 3.5 bits.

---

## The Exponential Improvement Over Alternatives

The $1/4^b$ scaling means distortion decreases **exponentially** with bit-width:

| Method | Distortion scaling |
|---|---|
| Uniform scalar (no rotation) | $\sim 1/2^b$ (exponential, but slower base) |
| Random rounding | $\sim 1/b$ (polynomial -- much worse) |
| TurboQuant | $\sim 1/4^b$ (exponential, optimal base) |
| Lower bound | $= 1/4^b$ (can't do better) |

The difference between $1/2^b$ and $1/4^b$ is enormous:

```
b = 4:   1/2^4 = 0.0625      vs    1/4^4 = 0.0039    (16x difference!)
b = 8:   1/2^8 = 0.0039      vs    1/4^8 = 0.0000015 (2500x difference!)
```

TurboQuant achieves the right **base of the exponent** (4, not 2).

---

## The Bottom Line

| Bit-width ($b$) | Compression ratio | MSE ($\leq$) | Quality impact (KV cache) |
|:---:|:---:|:---:|---|
| 1 | 16x | 0.36 | Too lossy for most use cases |
| 2 | 8x | 0.117 | Usable with tricks |
| 3 | 5.3x | 0.030 | Good quality |
| 3.5 | 4.6x | ~0.018 | Quality neutral |
| 4 | 4x | 0.009 | Nearly perfect |

At 3.5 bits, you compress the KV cache by 4.6x with zero quality loss on every benchmark the paper tested. And the theory tells us we're within 2.7x of the best any algorithm could ever achieve.

> **This isn't a heuristic -- it's a provably near-optimal solution.**
