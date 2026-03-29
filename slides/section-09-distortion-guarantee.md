# Section 9: The Distortion Guarantee — How Good Is This?

**Duration:** 5 minutes  
**Goal:** Translate the paper's theoretical bounds into plain language. The audience should understand: what the upper and lower bounds mean, that TurboQuant is provably within 2.7× of the best any algorithm could ever achieve, and that each additional bit cuts error by 4×. This section builds confidence that the algorithm isn't just clever — it's near-optimal.

---

## The Claim

> "We've seen the algorithm: rotate, quantize, done. But how good is it? Could there be a much better algorithm out there? The paper answers this definitively — with both an upper bound (how well TurboQuant does) and a lower bound (the best anything could ever do)."

---

## The Upper Bound: TurboQuant's Guarantee

Theorem 1 from the paper states that for b bits per coordinate, TurboQuant's MSE distortion is at most:

```
                    √(3π)     1
    MSE  ≤  ────── × ────
                2       4ᵇ

         ≈  2.72 × (1/4ᵇ)
```

What does this mean in practice? Let's compute:

```
┌───────────┬──────────────┬──────────────────────────────────┐
│ Bit-width │ MSE (upper   │ What this means                  │
│    b      │ bound)       │                                  │
├───────────┼──────────────┼──────────────────────────────────┤
│    1      │   0.36       │ 36% of the vector's energy is    │
│           │              │ lost — rough, but usable         │
│           │              │                                  │
│    2      │   0.117      │ ~12% energy loss — decent        │
│           │              │                                  │
│    3      │   0.030      │ ~3% energy loss — very good      │
│           │              │                                  │
│    4      │   0.009      │ <1% energy loss — nearly perfect │
│           │              │                                  │
│    5      │   0.002      │ Negligible loss                  │
└───────────┴──────────────┴──────────────────────────────────┘
```

The key pattern: **every additional bit reduces error by exactly 4×.**

```
b=1 → b=2:   0.36  / 0.117 ≈ 3.1×  reduction
b=2 → b=3:   0.117 / 0.030 ≈ 3.9×  reduction
b=3 → b=4:   0.030 / 0.009 ≈ 3.3×  reduction
```

Roughly 4× per bit. This is the 1/4^b term at work. It means there are **sharply diminishing returns** — going from 1 to 2 bits is a massive quality jump, while going from 4 to 5 bits gives you almost nothing. This is why the practical sweet spot is 3-4 bits.

---

## The Lower Bound: The Laws of Physics

Now for the remarkable part. The paper also proves (Theorem 3) that **no algorithm in the universe** — no matter how clever, how slow, how much memory it uses — can achieve MSE better than:

```
                  1
    MSE  ≥  ────
               4ᵇ
```

This is an **information-theoretic lower bound**. It comes from Shannon's source coding theory — the same mathematics that governs how much you can compress any signal. It's a law of nature, like the speed of light. No engineering trick can beat it.

Where does it come from? The intuition:

```
With b bits per coordinate, you have 2^b possible reconstruction values.
The unit sphere has "volume" that must be covered by these 2^(b×d) total codewords.
Each codeword "covers" a region of the sphere.
The average volume per region determines the average distance from
any point to its nearest codeword.

Shannon showed this average distance is at least 1/4^b.
No codebook design can do better.
```

---

## The Gap: How Close Is TurboQuant to Perfect?

Comparing the upper and lower bounds:

```
                                               TurboQuant's
              Lower bound     TurboQuant       distance from
Bit-width     (best possible) (actual)         perfection
─────────     ──────────────  ─────────        ──────────────
b = 1         0.25            0.36             1.44×
b = 2         0.0625          0.117            1.87×
b = 3         0.0156          0.030            1.92×
b = 4         0.0039          0.009            2.31×
b → ∞         1/4ᵇ            2.72/4ᵇ         2.72×
```

At worst, TurboQuant is **2.72× away from the theoretical optimum**. At low bit-widths (which are the most practically relevant), it's even closer — only 1.44× at 1 bit.

Visually, if you plot both bounds on a log scale:

```
MSE (log scale)
  │
  │  ╲
  │    ╲  ← TurboQuant (upper bound)
  │      ╲
  │        ╲
  │    ╲     ╲
  │      ╲     ╲
  │        ╲     ╲
  │          ╲     ╲
  │            ╲     ╲  ← Lower bound (best possible)
  │              ╲     ╲
  └────────────────────────
    1    2    3    4    5
         Bit-width (b)

  The two lines are nearly parallel on a log scale.
  The gap is a constant factor ≈ 2.7×.
```

The actual experimental results (Figure 3 in the paper) show measured distortion falling right between these two bounds, confirming the theory matches practice.

---

## What 2.7× Means in Practice

> "2.7× from optimal — is that good?"

To put this in perspective:

**It means the gap is a constant factor, not a function of bit-width.** Many quantization methods have distortion that degrades relative to the optimum as you change parameters. TurboQuant maintains a constant 2.7× gap (or better) everywhere.

**No existing online method comes close.** Uniform scalar quantization (without rotation) has distortion that's exponentially worse than optimal at low bit-widths. The rotation is what closes the gap.

**The gap could theoretically be closed** — but it would require joint vector quantization (computationally impossible, as we discussed in Section 7) or data-dependent optimization (violating the online constraint). The 2.7× factor is the price of being online and practical.

**For KV cache, the MSE gap doesn't even matter.** What matters is downstream model quality, and the experiments show zero quality loss at 3.5 bits. The theoretical gap overstates the practical impact because attention is robust to small distortions that preserve relative rankings.

---

## The Exponential Improvement Over Alternatives

One more way to appreciate TurboQuant's distortion rate. The 1/4^b scaling means distortion decreases **exponentially** with bit-width. Compare this to simpler methods:

```
Method                      Distortion scaling
──────                      ──────────────────
Uniform scalar (no rotation)    ~ 1/2^b       (exponential, but slower base)
Random rounding                 ~ 1/b          (polynomial — much worse)
TurboQuant                      ~ 1/4^b       (exponential, optimal base)
Lower bound                     = 1/4^b       (can't do better)
```

The difference between 1/2^b and 1/4^b is enormous at higher bit-widths:

```
b = 4:   1/2^4 = 0.0625      vs    1/4^4 = 0.0039    (16× difference!)
b = 8:   1/2^8 = 0.0039      vs    1/4^8 = 0.0000015 (2500× difference!)
```

TurboQuant achieves the right **base of the exponent** (4, not 2), which is what the paper means by "exponential improvement over existing methods in terms of bit-width dependence."

---

## Summary Table

```
┌────────────────────────────────────────────────────────────┐
│  Bit-width   Compression   MSE        Quality impact       │
│     b        ratio (×)     (≤)        (KV cache)           │
├────────────────────────────────────────────────────────────┤
│     1          16×         0.36       Too lossy for most   │
│     2           8×         0.117      Usable with tricks   │
│     3          5.3×        0.030      Good quality         │
│   3.5          4.6×        ~0.018     Quality neutral (!)  │
│     4           4×         0.009      Nearly perfect       │
└────────────────────────────────────────────────────────────┘

The compression ratio is 16/b (from float16 to b bits per value).
```

> "At 3.5 bits, you compress the KV cache by 4.6× with zero quality loss on every benchmark the paper tested. And the theory tells us we're within 2.7× of the best any algorithm could ever achieve. This isn't a heuristic — it's a provably near-optimal solution."

---

## Speaker Notes

- **Lead with the table.** The concrete MSE values for b = 1, 2, 3, 4 are more memorable than the formula. Show the table first, then explain where the numbers come from.
- **"Every additional bit cuts error by 4×"** is the one sentence to make stick. It's simple, surprising, and explains why 3-4 bits is the sweet spot (3 bits is already very good, 5 bits is overkill).
- **The lower bound explanation** should feel like a law of nature, not a math proof. "Shannon showed this in the 1940s — it's as fundamental as the speed of light." Engineers respect physical limits.
- **The gap table** (1.44× at b=1, up to 2.72× asymptotically) is impressive. Highlight that at low bit-widths — exactly where you care most — TurboQuant is closest to optimal.
- **Don't prove either theorem.** The proofs are in the paper for anyone who wants them. For the talk, state the results and explain what they mean. The audience trusts peer-reviewed mathematics; they don't need to verify it live.
- **The comparison to 1/2^b scaling** is worth one sentence if time permits: "Without the rotation trick, you'd need twice as many bits to get the same quality. The rotation is what gives you the optimal 1/4^b base."
- **This is a short section by design.** The results are powerful but the exposition is straightforward. Don't pad it. State the bounds, show the numbers, move on to the bias problem.
- **Possible audience questions:**
  - "Can you close the 2.7× gap?" — Only by giving up the online property (data-dependent optimization) or using joint quantization (computationally impossible). The gap is the cost of being practical.
  - "Do real experiments match the bounds?" — Yes, Figure 3 in the paper shows measured distortion falling between the upper and lower bounds at every bit-width. Theory matches practice.
  - "Why does each bit give 4× improvement instead of 2×?" — Each bit doubles the number of codebook entries, but in the MSE metric, the improvement scales as (number of entries)². Doubling entries → 4× better MSE. This comes from the 1/4^b = 1/(2^b)² relationship.
- **Transition to Section 10:** "So the MSE story is clean: rotate, quantize, near-optimal. But there's a twist. MSE-optimal quantizers have a hidden flaw when it comes to inner products — they're biased. Let me show you what that means."
