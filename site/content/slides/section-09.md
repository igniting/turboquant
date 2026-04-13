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

## Why Bias Matters More Than MSE — The Softmax Effect

The distortion bounds above measure how accurately individual vectors are reconstructed. But attention doesn't use raw inner products directly -- it pipes them through **softmax**:

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

Softmax is an exponential function. A small additive error $\varepsilon$ in an attention score produces an output error proportional to $e^x(e^\varepsilon - 1)$. For large scores, this amplification can be dramatic.

This has two practical consequences:

1. **Zero-mean random noise** tends to cancel out after softmax, because positive and negative errors balance across tokens.
2. **Systematic bias** gets amplified in a consistent direction. Non-uniform bias -- where some scores are underestimated more than others -- distorts the attention distribution even after averaging.

> **This is why the bias correction in Section 10 is not just a mathematical nicety.** It's the difference between softmax routing the model correctly and subtly miscounting which tokens matter. The MSE bound above only tells half the story; Sections 10-11 address the other half.

---

## The Lower Bound: The Laws of Physics

The paper also proves (Theorem 3) a **lower bound for the problem class studied**: no online, distribution-agnostic algorithm can achieve MSE better than:

$$\text{MSE} \geq \frac{1}{4^b}$$

This bound derives from a covering-number argument on the unit sphere — not from Shannon's channel capacity theorem. The key constraint is "online and distribution-agnostic": the algorithm sees each vector once, with no prior knowledge of the data distribution. Rate-distortion-optimal vector quantizers for Gaussian sources can beat 1/4^b by a large constant factor, but they require offline calibration — which is exactly what Requirement 5 from Section 5 rules out for KV cache compression.

```
With b bits per coordinate, you have 2^b possible reconstruction values.
The unit sphere has "volume" that must be covered by these 2^(b*d) total codewords.
Each codeword "covers" a region of the sphere.
The average volume per region determines the minimum average distance from
a random unit vector to its nearest codeword.

This minimum distance is 1/4^b -- a geometric fact, not an engineering shortcoming.
```

---

## The Gap: How Close Is TurboQuant to Optimal?

Putting the bounds together:

$$\\underbrace{\frac{1}{4^b}}_{\text{lower bound}} \leq \text{MSE} \leq \underbrace{\frac{2.72}{4^b}}_{\text{TurboQuant}}$$

The gap is a factor of **2.72** -- the constant $\sqrt{3\pi}/2$. TurboQuant is within 2.72x of what any algorithm could ever achieve.

```
Imagine the optimal algorithm is standing at the finish line.
TurboQuant is standing 2.72 body-lengths behind it.
Every other practical algorithm is further back.
The finish line is mathematically unreachable by anyone.
```

This constant arises from using Lloyd-Max scalar quantization applied to a Gaussian distribution. The Gaussian approximation isn't exact -- real vectors after rotation are close to Gaussian but not perfectly so -- which is why the theoretical bound leaves a 2.72x gap. In practice, the measured MSE is typically tighter than the bound.

> **TurboQuant is near-optimal by a mathematical proof, not just empirical comparison.** The ceiling on improvement for any algorithm is 2.72x better, and that ceiling is unreachable in practice.

---

## What "Near-Optimal" Actually Means Here

One important nuance: the lower bound applies within a specific problem class. It is the best that any **scalar quantization, distribution-agnostic, online** algorithm can do.

"Online" is the binding constraint: TurboQuant quantizes each KV vector as it is generated, with no look-ahead and no statistics from future tokens. This eliminates approaches like Product Quantization (PQ) or VQ-VAE-style neural codebooks, which need a data sample to learn their codebooks.

Approaches that step outside this class can achieve better compression:

- **Product Quantization** — trains a codebook offline on sampled data, achieving better compression at the cost of an offline calibration step. Impractical for online KV cache quantization.
- **MLA (architectural)** — bakes compression into the model during training. Not quantization at all; it changes what gets cached. No bit-budget constraint; no online inference required. But requires retraining from scratch.

> **The near-optimality guarantee means no one can beat TurboQuant with a better online scalar quantizer.** It does not mean no one can beat it by changing the rules — using offline calibration, or redesigning the model architecture. Those are different games.

---

## Convergence of the Two Approaches

The long-term inference stack probably includes both:

```
New models:         Train with MLA → architectural KV compression, no quantization needed
Existing GQA models: Deploy with TurboQuant → near-optimal online compression of current architecture

Neither makes the other irrelevant.
TurboQuant is the best compression for the models that exist today.
MLA is the design choice for models being trained tomorrow.
```
