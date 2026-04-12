---
title: "The Memory Wall — Why You Should Care"
weight: 1
part: "Part I — Setting the Stage"
---

![KV Cache: The Real Memory Monster](/img/s01-memory-monster.webp)

Running Llama 3.1 8B with 512 concurrent users at 32K context requires roughly **2 TB of memory** -- just for the KV cache. That's 25+ H100 GPUs doing nothing but storing past tokens. This section explains exactly *where* that memory goes, and how a [paper from Google Research](https://arxiv.org/abs/2504.19874) makes most of it disappear -- without losing a single correct answer.

---

## The Cost Nobody Talks About

When people think about the memory cost of running an LLM, they think about the **model weights** -- the billions of parameters that define the model. And yes, those are big:

| Model | Parameters | Weight Memory (FP16) |
|-------|-----------|---------------------|
| Llama 3.1 8B | 8 billion | ~16 GB |
| Llama 3.1 70B | 70 billion | ~140 GB |
| GPT-4 class | ~1.8 trillion (rumored) | ~3.6 TB |

But here's the thing most engineers don't realize: **the model weights are fixed**. You load them once. They don't grow. The thing that *grows* -- the thing that will actually crash your GPU -- is something called the **KV cache**.

---

## What Is the KV Cache?

Every time an LLM processes or generates a token, it stores a "memory" of that token so it doesn't have to reprocess the entire conversation from scratch. That stored memory is the KV cache.

We'll explain *how* this works in detail in the next section. For now, just understand this:

> **The KV cache grows linearly with the number of tokens in the conversation.**

Every new token adds a fixed amount of memory. And it adds that memory **at every layer** of the model, and **at every attention head** within each layer.

---

## The Brutal Math

Let's do the math for **Llama 3.1 8B**. Modern LLMs use two different attention variants, and the KV cache size differs substantially between them.

**Simplified MHA (Multi-Head Attention) — for illustration:**

```
KV cache per token = 2 (key + value)
                   × 32 (layers)
                   × 32 (attention heads)  ← full MHA
                   × 128 (dimension per head)
                   × 2 bytes (float16)
                   = 524,288 bytes ≈ 0.5 MB per token
```

**Actual Llama 3.1 8B uses GQA (Grouped Query Attention) — 8 KV heads, not 32:**

```
KV cache per token = 2 (key + value)
                   × 32 (layers)
                   × 8 (KV heads, shared across 4 query groups)
                   × 128 (dimension per head)
                   × 2 bytes (float16)
                   = 131,072 bytes ≈ 0.125 MB per token
```

GQA reduces the KV cache 4x versus MHA. But even at 0.125 MB per token, scale tells the rest of the story:

| Context Length | MHA (simplified) | GQA (actual Llama 3.1) | What This Means |
|---------------|-----------------|------------------------|-----------------|
| 1K tokens | ~0.5 GB | **~0.125 GB** | A short chat |
| 8K tokens | ~4 GB | **~1 GB** | A long conversation -- tight on a consumer GPU |
| 32K tokens | ~16 GB | **~4 GB** | A document -- equals the model weights |
| 128K tokens | ~64 GB | **~16 GB** | A book -- fills one H100 just for the cache |

> **Key Insight:** Even with GQA's 4x reduction, at 128K context the KV cache still equals the entire model. And that's for *one user*. GQA reduced the constant; it didn't change the linear growth.

---

## The Server Room Perspective

Now think about this from a production serving perspective (GQA — actual Llama 3.1 8B numbers):

| Scenario | KV Cache Memory |
|----------|----------------|
| 1 user, 32K context | ~4 GB |
| 10 concurrent users, 32K each | ~40 GB |
| 100 concurrent users, 32K each | ~400 GB |
| 512 concurrent users, 32K each | ~2 TB |

An H100 GPU has 80 GB of HBM. With 100 concurrent GQA-equipped users at 32K context, you need 5 H100s just for the KV cache, before loading a single model weight. The 4x GQA win disappears quickly once user count climbs.

This is why:

- **Your local LLM chokes on long documents.** The context window exists but the memory doesn't.
- **Cloud inference is expensive.** You're paying for GPUs that are mostly storing cached past tokens, not doing useful computation.
- **"Just add more context" isn't free.** Every doubling of context length doubles the KV cache, which means doubling the GPU memory.
- **Concurrent users are the real killer.** Each user gets their own KV cache. Scaling users scales memory linearly.

---

## The Bandwidth Problem

It's not just about fitting in memory. Even when the KV cache *does* fit, it creates a **bandwidth bottleneck**.

During token generation, for every single new token, the model must:

1. Read the **entire KV cache** from GPU HBM (high-bandwidth memory)
2. Compute attention scores against all cached tokens
3. Write the new token's key and value back to the cache

This read-everything-every-step pattern means KV cache memory bandwidth, not compute, is the bottleneck during generation. This is why LLM generation feels slow -- the GPU is waiting for data, not computing.

> **Smaller KV cache = less data to read each step = faster generation.**

Compression doesn't just save memory -- it speeds up inference.

---

## The Punchline

So we have a problem:

- The KV cache is often **larger than the model itself**
- It grows **linearly** with context length and number of users
- It creates a **bandwidth bottleneck** that slows down generation
- And it can't be compressed with traditional methods because **it's generated on the fly** -- you can't preprocess it

What if you could compress every vector in the KV cache from 16 bits per coordinate down to 3-4 bits -- a 4-5x reduction -- with **zero loss in model quality**?

That's TurboQuant.

And the beautiful part: it does this using just three operations -- a rotation, a table lookup, and a 1-bit error correction -- all backed by a mathematical proof that you can't do much better.

Let's understand how.
