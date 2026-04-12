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

## Community Benchmarks: turbo3 and turbo4 vs Standard Quantization

After the March 2026 release, the turboquant+ community ran head-to-head perplexity comparisons against llama.cpp's standard quantization formats on Llama-3.1-8B across a 32K-token text corpus:

| Method | Bits/weight | PPL (↓ better) | vs q4_0 |
|---|:---:|:---:|:---:|
| f16 (baseline) | 16 | 6.12 | — |
| q5_0 | 5 | 6.21 | — |
| q4_0 | 4 | 6.34 | baseline |
| **turbo4** (K=4b, V=2b) | **~3.8** | **6.31** | **+0.05% better** |
| turbo3 (K=3b, V=2b) | ~2.8 | 6.41 | +1.1% |
| q3_K_M | 3.35 | 6.58 | +3.8% |

**turbo4 beats q4_0 on perplexity while using fewer bits.** This is the key result for practitioners: if you're already running q4_0 quantized models, switching to turbo4 gets you a smaller cache with equal or better quality.

---

## Real-World Landmark: 104B Parameters at 128K Context on a MacBook

The most striking community benchmark: a 104B parameter model (Llama 3.3 70B + 34B expert layers) running at 128K context length on a single **MacBook Pro M5 Max** with 128 GB unified memory.

```
Hardware:    MacBook Pro M5 Max, 128 GB RAM
Model:       104B parameters (Llama 3.3 70B + 34B expert layers)
Context:     128,000 tokens
KV config:   turbo3 (K=3b, V=2b, boundary layers at 4b)

KV cache:    ~16 GB   (vs ~128 GB at full FP16)
Throughput:  ~4 tok/s
Quality:     No measurable degradation on HumanEval and MMLU subsets
```

Running a 104B model at 128K context on consumer hardware was not feasible before TurboQuant. The 8× KV cache reduction (from GQA + TurboQuant combined) is what makes it fit in 128 GB.

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
  • Model-agnostic (tested on Llama, Ministral, Qwen, Phi, DeepSeek)
  • turbo4 beats q4_0 at lower bit-width
  • 104B model at 128K context on consumer hardware
```
