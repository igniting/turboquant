---
title: "Nearest Neighbor Search — The Bonus Application"
weight: 14
part: "Part V — Does It Actually Work?"
---

![Nearest Neighbor Search](/img/s14-nearest-neighbor.webp)

TurboQuant is fundamentally a vector quantization algorithm -- and vectors show up everywhere. One of the most important use cases beyond KV cache is **nearest neighbor search in vector databases**.

---

## The Vector Search Problem

If you're building a RAG pipeline, search engine, or recommendation system, you have millions of vectors and need to find the most similar ones to a query. This is the same problem as KV cache quantization: compress vectors while preserving inner products.

---

## The Indexing Time Result

Time to quantize (index) 100,000 vectors:

| Method | d = 200 | d = 1536 | d = 3072 |
|---|:---:|:---:|:---:|
| Product Quantization | 37 sec | 240 sec | 494 sec |
| RabitQ | 597 sec | 2268 sec | 3957 sec |
| **TurboQuant** | **0.0007 s** | **0.0013 s** | **0.0021 s** |

```
Product Quantization:  240 seconds
TurboQuant:            0.0013 seconds

That's a 185,000× speedup.
```

PQ needs k-means clustering. RabitQ needs preprocessing. TurboQuant needs **nothing** -- the codebook is precomputed from the mathematical distribution, not from data.

> **TurboQuant reduces indexing time to virtually zero.** Vectors can be indexed the moment they arrive.

---

## The Recall Result

| | Top-1 | Top-4 | Top-16 | Top-64 |
|---|:---:|:---:|:---:|:---:|
| TurboQuant 4b | 0.98 | 0.99 | 1.00 | 1.00 |
| PQ 4-bit | 0.93 | 0.97 | 0.99 | 1.00 |
| RabitQ 4-bit | 0.96 | 0.98 | 0.99 | 1.00 |
| TurboQuant 2b | 0.88 | 0.94 | 0.98 | 1.00 |
| PQ 2-bit | 0.85 | 0.92 | 0.96 | 0.99 |

**TurboQuant beats or matches both PQ and RabitQ at every bit-width and every k** -- with zero preprocessing, while PQ had the advantage of training on the evaluation data.

---

## What This Means for Engineers

```
Before TurboQuant:
  1. Collect representative data
  2. Run k-means (minutes to hours)
  3. Build codebooks
  4. Quantize the database
  5. If data distribution shifts → repeat from step 1

With TurboQuant:
  1. Quantize each vector as it arrives (microseconds)
  2. There is no step 2.
```

This enables **streaming indexing** (add vectors in real time), **no distribution drift** (codebook never goes stale), and **simpler infrastructure** (no k-means pipeline, no codebook versioning, no reindexing jobs).
