---
title: "Why the KV Cache Exists"
weight: 4
part: "Part II — The KV Cache Problem"
---

![Why the KV Cache Exists](/img/s04-kv-cache.webp)

In Section 3, we established that attention requires computing dot products between the current token's Query and the Keys of **all previous tokens**. Now let's see what happens during generation without any caching.

---

## The Recomputation Problem

Let's generate the sentence "The cat sat on the mat" one token at a time, and watch the work pile up.

### Step 1: Generate "cat"

```
Input: ["The"]

Work: Run "The" through all 32 layers
      - Compute Q, K, V for "The" at each layer
      - No previous tokens to attend to (first token)
      - Produce next token prediction -> "cat"

Total forward passes through the model: 1 token x 32 layers
```

### Step 2: Generate "sat"

```
Input: ["The", "cat"]

Work: Run BOTH tokens through all 32 layers
      - Compute Q, K, V for "The" AGAIN (we already did this!)
      - Compute Q, K, V for "cat" at each layer
      - "cat" attends to "The" using dot products
      - Produce next token prediction -> "sat"

Total forward passes: 2 tokens x 32 layers
```

### Step 3: Generate "on"

```
Input: ["The", "cat", "sat"]

Work: Run ALL THREE tokens through all 32 layers
      - Compute Q, K, V for "The" AGAIN
      - Compute Q, K, V for "cat" AGAIN
      - Compute Q, K, V for "sat" at each layer
      - Produce next token prediction -> "on"

Total forward passes: 3 tokens x 32 layers
```

See the pattern? At step n, you're recomputing Q, K, V for all n tokens -- even though the first (n-1) tokens haven't changed. By step 1000, you're doing 1000x the work of step 1.

```
Total work without caching = 1 + 2 + 3 + ... + n = n(n+1)/2 ~ n²/2
```

For a 4000-token response, that's ~8 million forward passes instead of 4000. **A 2000x waste.**

---

## The Fix: Cache the Keys and Values

The insight is simple: **the Keys and Values for past tokens don't change**. Once you've computed K and V for token "The" at layer 5, those vectors are the same regardless of what comes after. Only the Query changes (because each new token has a different query).

So the fix is:

1. When you process a token, compute its Q, K, V
2. Use Q for the current attention computation
3. **Store K and V** in a cache for future steps
4. Next step, only compute Q, K, V for the **new** token
5. Look up all the cached K's and V's from previous tokens
6. Compute attention using the new Q against all cached K's

```
Step 1: Process "The"
  - Compute Q1, K1, V1
  - Store K1, V1 in cache
  - Predict -> "cat"

Step 2: Process "cat" (only the new token!)
  - Compute Q2, K2, V2
  - Store K2, V2 in cache
  - Attention: Q2 . [K1, K2]^T -> scores -> weighted sum of [V1, V2]
  - Predict -> "sat"

Step 3: Process "sat" (only the new token!)
  - Compute Q3, K3, V3
  - Store K3, V3 in cache
  - Attention: Q3 . [K1, K2, K3]^T -> scores -> weighted sum of [V1, V2, V3]
  - Predict -> "on"
```

Total work with caching: n forward passes (one per token), each processing just 1 token. **Linear instead of quadratic.**


> **The KV cache trades memory for compute.** You're storing all those K and V vectors so you never have to recompute them. The cost is memory -- which grows linearly with sequence length.

---

## Visualizing the Cache Growth

Here's what the KV cache looks like as tokens accumulate, for a single layer with a single attention head:

```
After token 1 ("The"):
  Keys:   [ K1 ]                    Values: [ V1 ]

After token 2 ("cat"):
  Keys:   [ K1 | K2 ]              Values: [ V1 | V2 ]

After token 3 ("sat"):
  Keys:   [ K1 | K2 | K3 ]        Values: [ V1 | V2 | V3 ]

After token 100:
  Keys:   [ K1 | K2 | K3 | ... | K100 ]
  Values: [ V1 | V2 | V3 | ... | V100 ]
```

Each K and V is a vector of 128 floats (for Llama 8B). And this is just **one head in one layer**. The full picture:

```
Layer 1:
  Head 1:  Keys: [K1 ... Kn]  Values: [V1 ... Vn]
  Head 2:  Keys: [K1 ... Kn]  Values: [V1 ... Vn]
  ...
  Head 32: Keys: [K1 ... Kn]  Values: [V1 ... Vn]

Layer 2 through Layer 32: (same structure)
```

That's **32 layers x 32 heads x 2 (K and V) = 2048 separate vectors** stored per token.

---

## The Exact Memory Math

Let's build the formula piece by piece:

```
KV cache memory per token:

  2               <- Key + Value
  x n_layers      <- one cache per layer        (32 for Llama 8B)
  x n_heads       <- one cache per head          (32 for Llama 8B)
  x d_head        <- dimension of each vector   (128 for Llama 8B)
  x bytes_per_val <- storage per number          (2 for float16)
```

Plugging in for **Llama 3.1 8B**:

```
Per token:  2 x 32 x 32 x 128 x 2 = 524,288 bytes ~ 0.5 MB
```

For larger models, it's even worse:

| Model | Layers | Heads | d_head | Per Token | At 32K |
|-------|--------|-------|--------|-----------|--------|
| Llama 8B | 32 | 32 | 128 | 0.5 MB | 16 GB |
| Llama 70B | 80 | 64 | 128 | 2.5 MB | 80 GB |
| GPT-4 class | ~120 | ~96 | ~128 | ~5.6 MB | ~180 GB |

> **At 32K context, Llama 70B's KV cache alone fills an entire H100 GPU (80 GB).** There's no room left for the model weights, activations, or anything else.

---

## The Generation Bottleneck

At **every single generated token**, the GPU must:

```
1. Compute Q, K, V for this token           <- small (one token)
2. Append K, V to the cache                 <- cache grows by one entry
3. Load ALL cached Keys from GPU memory     <- THIS IS THE BOTTLENECK
4. Compute dot product of Q with every K    <- scales with cache size
5. Softmax over all scores                  <- scales with cache size
6. Load ALL cached Values from GPU memory   <- ANOTHER BOTTLENECK
7. Weighted sum of Values                   <- scales with cache size
8. Repeat for all 32 heads x 32 layers
```

Steps 3 and 6 are the killers. For a 16 GB cache, that's 16 GB of memory reads **per token generated**.

On an H100 with ~3.35 TB/s memory bandwidth:

```
Time to read 16 GB cache = 16 / 3350 ~ 4.8 ms per token
-> Max ~208 tokens/second (ignoring all other work)
```

The actual compute needed (the dot products themselves) takes a fraction of a millisecond. **The GPU is spending 90%+ of its time waiting for memory, not computing.**

> **A smaller KV cache means less data to read, which means faster generation.** TurboQuant's 4-5x compression directly translates to faster inference, not just lower memory usage.

---

## The Compression Opportunity

```
Current state:
  Each value in the KV cache: 16 bits (float16)
  Per token: 0.5 MB (Llama 8B)
  At 32K: 16 GB

If we could compress to 4 bits per value (4x compression):
  Per token: 0.125 MB
  At 32K: 4 GB                <- fits comfortably on any modern GPU

If we could compress to 3 bits per value (~5.3x compression):
  Per token: 0.094 MB
  At 32K: 3 GB                <- room for much longer contexts
```

The question is: **can you compress from 16 bits to 3-4 bits without breaking the attention mechanism?**

Naive approaches (just rounding to fewer bits) break in subtle ways. The next section explains why.
