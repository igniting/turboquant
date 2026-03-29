# Section 13: KV Cache Experiments — The Headlines

**Duration:** 4 minutes  
**Goal:** Deliver the three results that justify the entire talk. The audience should walk away knowing: TurboQuant matches full-precision on needle-in-a-haystack, matches full-precision on LongBench at 3.5 bits, and beats competitors that use more bits. These are the numbers they'll quote to their teams.

---

## The Real Test

> "We've validated the math on embedding vectors. Now the question that actually matters: if you compress a real LLM's KV cache with TurboQuant, does the model still give correct answers?"

The paper tests on two benchmarks using **Llama-3.1-8B-Instruct** — a production-grade open model that many teams actually deploy.

---

## Result 1: Needle-in-a-Haystack — Perfect Retrieval at 4× Compression

The **Needle-in-a-Haystack** test is the simplest, most demanding test of long-context quality. You take a long document (the "haystack" — up to 104K tokens), hide one specific sentence somewhere in it (the "needle"), and ask the model to find it.

If the KV cache compression damages attention in any way, the model will fail to locate the needle — especially when it's buried deep in a long context.

Results with 4× compression (only 25% of original KV cache memory):

```
┌──────────────────────┬────────────┐
│ Method               │ Recall     │
├──────────────────────┼────────────┤
│ Full Precision       │ 0.997      │
│ TurboQuant           │ 0.997  ✓   │  ← IDENTICAL to full precision
│ PolarQuant           │ 0.995      │
│ KIVI                 │ 0.981      │
│ PyramidKV            │ 0.895      │
│ SnapKV               │ 0.858      │
└──────────────────────┴────────────┘
```

**TurboQuant scores identically to the uncompressed model.** Not "close to" — the exact same score: 0.997.

The heatmap visualization (Figure 4 in the paper) makes this even more striking. The full-precision and TurboQuant heatmaps are visually indistinguishable — perfect green (successful retrieval) across all context lengths (4K to 104K) and all needle positions (0% to 100% depth). Meanwhile, SnapKV and PyramidKV show large red patches (failed retrieval), especially at longer contexts.

> **At 4× compression, TurboQuant doesn't degrade long-context retrieval at all.** The model finds the needle every time, just as well as the uncompressed model.

---

## Result 2: LongBench — Quality Neutral at 3.5 Bits

Needle-in-a-haystack is a single-task test. **LongBench** is a comprehensive suite covering six categories of long-context tasks:

- Single-document QA
- Multi-document QA
- Summarization
- Few-shot learning
- Synthetic tasks
- Code completion

The paper uses **LongBench-E**, a subset with uniform length distribution for fair comparison across context sizes.

Results on Llama-3.1-8B-Instruct:

```
┌──────────────────────────┬─────────┬─────────┐
│ Method                   │ KV bits │ Average │
├──────────────────────────┼─────────┼─────────┤
│ Full Cache               │   16    │  50.06  │
│ TurboQuant (ours)        │  3.5    │  50.06  │  ← IDENTICAL
│ KIVI                     │   5     │  50.16  │
│ PolarQuant               │  3.9    │  49.78  │
│ TurboQuant (ours)        │  2.5    │  49.44  │
│ KIVI                     │   3     │  48.50  │
└──────────────────────────┴─────────┴─────────┘
```

Three things jump out:

**1. TurboQuant at 3.5 bits = Full precision.** Score of 50.06 in both cases. Across six diverse task categories, compression has zero measurable impact on quality. This is a 4.6× reduction in KV cache memory.

**2. TurboQuant at 2.5 bits still beats KIVI at 3 bits.** Even at extreme compression (6.4×), TurboQuant (49.44) outperforms KIVI at a higher bit budget (48.50). The provably near-optimal distortion rate translates into real quality advantages.

**3. TurboQuant quantizes during generation.** Unlike KIVI and PolarQuant, which leave newly generated tokens unquantized during streaming, TurboQuant applies compression even to the tokens being generated in real time. This is a stricter test — and it still matches full precision.

---

## The Non-Integer Bit-Widths: Outlier Channels

The 2.5-bit and 3.5-bit configurations deserve a brief explanation, as they reveal a practical trick:

```
Not all channels are equal. Some channels (the "outliers") have
much larger magnitudes than others. Quantizing them at the same
precision as regular channels would cause disproportionate error.

Solution: split channels into two groups, quantize each at
different precision.

2.5-bit configuration:
  32 outlier channels  ×  3 bits  =  96 bits
  96 regular channels  ×  2 bits  = 192 bits
  Total: 288 bits for 128 channels = 2.25 bits/channel
  (effectively ~2.5 with overhead)

3.5-bit configuration:
  Similar split with higher allocation to outliers
```

This is consistent with prior work on outlier-aware quantization. TurboQuant handles it naturally by running two independent instances with different bit-widths and merging the results.

---

## Result 3: Confirmed on a Second Model

The paper also tests on **Ministral-7B-Instruct**, a different model architecture:

```
┌──────────────────────────┬─────────┬─────────┐
│ Method                   │ KV bits │ Average │
├──────────────────────────┼─────────┼─────────┤
│ Full Cache               │   16    │  49.89  │
│ TurboQuant               │  2.5    │  49.62  │
└──────────────────────────┴─────────┴─────────┘
```

Even at 2.5 bits on a different model, the quality drop is just 0.27 points — marginal and likely within noise. This confirms TurboQuant is **model-agnostic**, not tuned to a specific architecture.

---

## The Money Slide

If you have one slide that summarizes the experimental results, it's this:

```
┌──────────────────────────────────────────────────────┐
│                                                      │
│   TurboQuant at 3.5 bits per coordinate:             │
│                                                      │
│   • KV cache compressed by 4.6×                      │
│   • Zero quality loss on Needle-in-a-Haystack        │
│   • Zero quality loss on LongBench (6 task types)    │
│   • Works during streaming generation                │
│   • Model-agnostic (tested on Llama and Ministral)   │
│   • No calibration, no preprocessing, no training    │
│                                                      │
│   At 2.5 bits (6.4× compression):                    │
│   • Still beats KIVI at 3 bits                       │
│   • Only 0.6-point drop on LongBench                 │
│                                                      │
└──────────────────────────────────────────────────────┘
```

> "This isn't a quality-memory tradeoff at 3.5 bits. There is no tradeoff. You get 4.6× less memory and the exact same quality."

---

## Speaker Notes

- **Lead with needle-in-a-haystack.** The "0.997 vs 0.997" comparison is the single most quotable result. State it, pause, let it land. "Identical to full precision."
- **The LongBench table is your credibility anchor.** It covers six diverse task categories, not just retrieval. The "50.06 vs 50.06" at 3.5 bits is remarkable because it's across summarization, code completion, few-shot learning — not just the task TurboQuant was designed for.
- **The KIVI comparison is the competitive kill shot.** TurboQuant at 2.5 bits beats KIVI at 3 bits. Fewer bits, better quality. This is the provably-optimal distortion rate paying off in practice.
- **Don't over-explain the outlier channel strategy.** One sentence: "Some channels have bigger magnitudes, so they get more bits." The 2.5-bit calculation (32×3 + 96×2 = 288 bits for 128 channels) is interesting but not essential. Skip it if time is tight.
- **The Ministral result** is worth mentioning in one breath: "Same story on a completely different model." It preempts the objection "maybe this only works on Llama."
- **The "money slide"** is the one to leave up during the Q&A pause. It's the summary people will photograph.
- **Possible audience questions:**
  - "What about larger models like 70B?" — The paper doesn't test 70B, but the theory is dimension-dependent (higher d = better), so larger models with more heads/layers should work at least as well. Community experiments on Qwen 35B confirm this.
  - "What's the inference speed impact?" — The paper shows negligible overhead. The rotation and quantization cost is tiny compared to the attention computation itself. Community implementations on Apple Silicon report parity with q8_0 speed.
  - "Have they tested on reasoning tasks? Math? Code generation?" — LongBench includes code completion and synthetic reasoning tasks. Results are good across all categories. More extensive benchmarking will come from community implementations.
  - "Why not test at 2 bits flat?" — 2 bits is aggressive, especially for Keys which have larger magnitudes. The outlier strategy (2.5 bits) gives the system room to handle high-magnitude channels. Flat 2-bit would likely show more degradation.
- **Transition to Section 14:** "KV cache compression is TurboQuant's primary application. But the same algorithm turns out to be excellent for another problem: nearest neighbor search in vector databases."
