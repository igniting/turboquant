# Section 14: Nearest Neighbor Search — The Bonus Application

**Duration:** 3 minutes  
**Goal:** Show that TurboQuant isn't a one-trick pony. For anyone building vector databases or RAG pipelines, the zero-indexing-time result is a game-changer. Keep it fast — this is a bonus, not the main event.

---

## Beyond KV Cache

> "Everything so far has been about KV cache compression. But TurboQuant is fundamentally a vector quantization algorithm — and vectors show up everywhere. One of the most important use cases is nearest neighbor search in vector databases."

---

## The Vector Search Problem

If you're building a RAG (Retrieval-Augmented Generation) pipeline, search engine, or recommendation system, you have millions of vectors in a database and you need to find the ones most similar to a query vector.

```
Database: 1,000,000 vectors (each 1536-dim, from OpenAI embeddings)
Query:    "What is the capital of France?"  → query vector q

Task:     Find the k vectors in the database with highest ⟨q, vᵢ⟩
```

At full precision, this database takes ~12 GB of memory. Scanning all vectors for every query is slow. The standard solution: **compress the vectors**, then search in the compressed space.

This is exactly the same problem as KV cache quantization: compress vectors while preserving inner products.

---

## The Indexing Time Result

The most striking result is Table 2 from the paper — time to quantize (index) 100,000 vectors:

```
┌────────────────────────┬──────────┬──────────┬──────────┐
│ Method                 │ d = 200  │ d = 1536 │ d = 3072 │
├────────────────────────┼──────────┼──────────┼──────────┤
│ Product Quantization   │  37 sec  │  240 sec │  494 sec │
│ RabitQ                 │ 597 sec  │ 2268 sec │ 3957 sec │
│ TurboQuant             │ 0.0007 s │ 0.0013 s │ 0.0021 s │
└────────────────────────┴──────────┴──────────┴──────────┘
```

Read those numbers again:

```
Product Quantization:  240 seconds
TurboQuant:            0.0013 seconds

That's a 185,000× speedup.
```

Why? PQ needs to run **k-means clustering** on the entire dataset to build codebooks. RabitQ also needs preprocessing. TurboQuant needs **nothing** — the codebook is precomputed from the mathematical distribution, not from the data.

```
PQ indexing:         collect data → run k-means → build codebooks → quantize
TurboQuant indexing: quantize

There's literally nothing to do except rotate and look up centroids.
```

> **TurboQuant reduces indexing time to virtually zero.** Vectors can be indexed the moment they arrive — no batch processing, no retraining when the distribution changes.

---

## The Recall Result

Speed means nothing if quality suffers. Figure 5 from the paper shows recall@1@k — the probability that the true nearest neighbor appears in the top-k results from the quantized index.

```
Dataset: OpenAI3 embeddings, d = 1536

              Top-1    Top-4    Top-16    Top-64
              ─────    ─────    ──────    ──────
TurboQuant 4b  0.98     0.99     1.00      1.00
PQ 4-bit       0.93     0.97     0.99      1.00
RabitQ 4-bit   0.96     0.98     0.99      1.00

TurboQuant 2b  0.88     0.94     0.98      1.00
PQ 2-bit       0.85     0.92     0.96      0.99
RabitQ 2-bit   0.87     0.93     0.97      0.99
```

**TurboQuant beats or matches both PQ and RabitQ at every bit-width and every k.** And it does this with zero preprocessing, while PQ had the advantage of training on the evaluation data itself.

The pattern holds across all three datasets tested: GloVe (d=200), OpenAI3 (d=1536), and OpenAI3 (d=3072). Higher dimensions actually help TurboQuant more, because the near-independence property of rotated coordinates strengthens with dimension.

---

## What This Means for Engineers

If you're building or maintaining a vector search system:

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

This enables:

- **Streaming indexing:** Add vectors to the index in real time, no batch processing
- **No distribution drift:** The codebook is data-independent, so it never goes stale
- **Simpler infrastructure:** No k-means training pipeline, no codebook versioning, no reindexing jobs
- **Same or better recall** than methods that require all that infrastructure

---

## Speaker Notes

- **This section is a bonus, not the main event.** 3 minutes, tops. The KV cache story is the talk's core. This section exists for the audience members who work on search or RAG — and to show TurboQuant's generality.
- **The 185,000× speedup** is your headline number. Say it, pause, move on. The number is so extreme that it doesn't need elaboration.
- **The recall table matters because it answers the obvious objection:** "Sure it's fast, but does it sacrifice quality?" No. It's actually better.
- **Don't explain PQ or RabitQ in detail.** The audience doesn't need to know how they work — only that they're the established baselines and TurboQuant beats them.
- **The "streaming indexing" implication** will resonate with engineers who've dealt with reindexing pipelines. "No batch processing, no retraining" is an operational win, not just a technical one.
- **Possible audience questions:**
  - "How does this compare to HNSW or other graph-based ANN methods?" — TurboQuant handles the quantization/compression layer. It's complementary to graph-based indexing methods like HNSW, which handle the search traversal. You can combine both: HNSW for graph traversal + TurboQuant for distance computation on compressed vectors.
  - "Would this work with Pinecone/Qdrant/pgvector?" — In principle, yes. TurboQuant could replace the PQ codebook in any vector database that uses product quantization. Integration would require framework-level changes, but the algorithm is compatible.
  - "At d=200 (GloVe), does the near-independence still hold?" — The recall is lower at d=200 than at d=1536 or 3072, which is consistent with weaker near-independence at lower dimensions. But TurboQuant still beats PQ and RabitQ even there.
- **Transition to Section 15:** "We've seen what the paper shows. Now let's talk about what the paper doesn't show — the practical lessons from the community implementations that have already started building on TurboQuant."
