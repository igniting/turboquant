---
title: "KV Cache Experiments — The Headlines"
weight: 13
part: "Part V — Does It Actually Work?"
---

![KV Cache Experiments](/img/s13-kv-experiments.webp)

If you compress a real LLM's KV cache with TurboQuant, does the model still give correct answers? The paper tests on **Llama-3.1-8B-Instruct** -- a production-grade open model.

---

## Result 1: Needle-in-a-Haystack — Perfect Retrieval at 4x Compression

Hide one sentence in up to 104K tokens and ask the model to find it. If KV cache compression damages attention, the model fails.

| Method | Recall |
|---|---|
| Full Precision | 0.997 |
| **TurboQuant** | **0.997** |
| PolarQuant | 0.995 |
| KIVI | 0.981 |
| PyramidKV | 0.895 |
| SnapKV | 0.858 |

**TurboQuant scores identically to the uncompressed model.** Not "close to" -- the exact same score: 0.997. The full-precision and TurboQuant heatmaps are visually indistinguishable -- perfect retrieval across all context lengths and needle positions.

> **At 4x compression, TurboQuant doesn't degrade long-context retrieval at all.**

---

## Result 2: LongBench — Quality Neutral at 3.5 Bits

**LongBench** covers six categories: single-doc QA, multi-doc QA, summarization, few-shot learning, synthetic tasks, and code completion.

| Method | KV bits | Average |
|---|:---:|:---:|
| Full Cache | 16 | 50.06 |
| **TurboQuant** | **3.5** | **50.06** |
| KIVI | 5 | 50.16 |
| PolarQuant | 3.9 | 49.78 |
| TurboQuant | 2.5 | 49.44 |
| KIVI | 3 | 48.50 |

Three things jump out:

**1. TurboQuant at 3.5 bits = Full precision.** Score of 50.06 in both cases. A 4.6x reduction in KV cache memory with zero measurable quality impact.

**2. TurboQuant at 2.5 bits beats KIVI at 3 bits.** Even at 6.4x compression, TurboQuant (49.44) outperforms KIVI at a higher bit budget (48.50).

**3. TurboQuant quantizes during generation.** Unlike KIVI and PolarQuant which leave newly generated tokens unquantized, TurboQuant compresses in real time. A stricter test -- and it still matches full precision.

---

## Non-Integer Bit-Widths: Outlier Channels

The 2.5 and 3.5-bit configurations use a practical trick: not all channels are equal. Some "outlier" channels have much larger magnitudes. The solution is to split channels into two groups at different precision:

```
2.5-bit: 32 outlier channels × 3 bits + 96 regular channels × 2 bits
3.5-bit: similar split with higher allocation to outliers
```

TurboQuant handles this naturally by running two independent instances with different bit-widths.

---

## Confirmed on a Second Model

On **Ministral-7B-Instruct**, a different architecture: even at 2.5 bits, the quality drop is just 0.27 points -- marginal and likely within noise. This confirms TurboQuant is **model-agnostic**.

---

## The Money Slide

```
TurboQuant at 3.5 bits per coordinate:

  • KV cache compressed by 4.6×
  • Zero quality loss on Needle-in-a-Haystack
  • Zero quality loss on LongBench (6 task types)
  • Works during streaming generation
  • Model-agnostic (tested on Llama and Ministral)
  • No calibration, no preprocessing, no training

At 2.5 bits (6.4× compression):
  • Still beats KIVI at 3 bits
  • Only 0.6-point drop on LongBench
```

> **This isn't a quality-memory tradeoff at 3.5 bits. There is no tradeoff. You get 4.6x less memory and the exact same quality.**
