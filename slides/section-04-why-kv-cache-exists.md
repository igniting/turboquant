# Section 4: Why the KV Cache Exists

**Duration:** 7 minutes  
**Goal:** Walk through autoregressive generation step-by-step to make the audience *feel* why caching is necessary, then build the exact memory math so they understand where every byte goes.

---

## The Recomputation Problem

In Section 3, we established that attention requires computing dot products between the current token's Query and the Keys of **all previous tokens**. Now let's see what happens during generation without any caching.

> "Let's generate the sentence 'The cat sat on the mat' one token at a time, and watch the work pile up."

### Step 1: Generate "cat"

```
Input: ["The"]

Work: Run "The" through all 32 layers
      - Compute Q, K, V for "The" at each layer
      - No previous tokens to attend to (first token)
      - Produce next token prediction → "cat"

Total forward passes through the model: 1 token × 32 layers
```

### Step 2: Generate "sat"

```
Input: ["The", "cat"]

Work: Run BOTH tokens through all 32 layers
      - Compute Q, K, V for "The" AGAIN (we already did this!)
      - Compute Q, K, V for "cat" at each layer
      - "cat" attends to "The" using dot products
      - Produce next token prediction → "sat"

Total forward passes: 2 tokens × 32 layers
```

### Step 3: Generate "on"

```
Input: ["The", "cat", "sat"]

Work: Run ALL THREE tokens through all 32 layers
      - Compute Q, K, V for "The" AGAIN
      - Compute Q, K, V for "cat" AGAIN
      - Compute Q, K, V for "sat" at each layer
      - Each token attends to all previous tokens
      - Produce next token prediction → "on"

Total forward passes: 3 tokens × 32 layers
```

See the pattern? At step n, you're recomputing Q, K, V for all n tokens — even though the first (n-1) tokens haven't changed. By step 1000, you're doing 1000× the work of step 1.

```
Total work without caching = 1 + 2 + 3 + ... + n = n(n+1)/2 ≈ n²/2
```

For a 4000-token response, that's ~8 million forward passes instead of 4000. **A 2000× waste.**

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
  - Compute Q₁, K₁, V₁
  - Store K₁, V₁ in cache
  - Predict → "cat"

Step 2: Process "cat" (only the new token!)
  - Compute Q₂, K₂, V₂
  - Store K₂, V₂ in cache
  - Attention: Q₂ · [K₁, K₂]ᵀ → scores → weighted sum of [V₁, V₂]
  - Predict → "sat"

Step 3: Process "sat" (only the new token!)
  - Compute Q₃, K₃, V₃
  - Store K₃, V₃ in cache
  - Attention: Q₃ · [K₁, K₂, K₃]ᵀ → scores → weighted sum of [V₁, V₂, V₃]
  - Predict → "on"
```

Total work with caching: n forward passes (one per token), each processing just 1 token. **Linear instead of quadratic.**

> **The KV cache trades memory for compute.** You're storing all those K and V vectors so you never have to recompute them. The cost is memory — which grows linearly with sequence length.

---

## Visualizing the Cache Growth

Here's what the KV cache looks like as tokens accumulate, for a single layer with a single attention head:

```
After token 1 ("The"):
  Keys:   [ K₁ ]                    Values: [ V₁ ]

After token 2 ("cat"):
  Keys:   [ K₁ | K₂ ]              Values: [ V₁ | V₂ ]

After token 3 ("sat"):
  Keys:   [ K₁ | K₂ | K₃ ]        Values: [ V₁ | V₂ | V₃ ]

After token 100:
  Keys:   [ K₁ | K₂ | K₃ | ... | K₁₀₀ ]
  Values: [ V₁ | V₂ | V₃ | ... | V₁₀₀ ]
```

Each K and V is a vector of 128 floats (for Llama 8B). And remember — this is just **one head in one layer**. The full picture:

```
Layer 1:
  Head 1:  Keys: [K₁ ... Kₙ]  Values: [V₁ ... Vₙ]     ← 128 × n × 2 vectors
  Head 2:  Keys: [K₁ ... Kₙ]  Values: [V₁ ... Vₙ]
  ...
  Head 32: Keys: [K₁ ... Kₙ]  Values: [V₁ ... Vₙ]

Layer 2:
  Head 1:  Keys: [K₁ ... Kₙ]  Values: [V₁ ... Vₙ]
  ...
  Head 32: Keys: [K₁ ... Kₙ]  Values: [V₁ ... Vₙ]

...

Layer 32:
  Head 1:  Keys: [K₁ ... Kₙ]  Values: [V₁ ... Vₙ]
  ...
  Head 32: Keys: [K₁ ... Kₙ]  Values: [V₁ ... Vₙ]
```

That's **32 layers × 32 heads × 2 (K and V) = 2048 separate vectors** stored per token.

---

## The Exact Memory Math

Let's build the formula piece by piece:

```
KV cache memory per token:

  2               ← Key + Value
  × n_layers      ← one cache per layer        (32 for Llama 8B)
  × n_heads       ← one cache per head          (32 for Llama 8B)
  × d_head        ← dimension of each vector   (128 for Llama 8B)
  × bytes_per_val ← storage per number          (2 for float16)
```

Plugging in for **Llama 3.1 8B**:

```
Per token:  2 × 32 × 32 × 128 × 2 = 524,288 bytes ≈ 0.5 MB
```

Now scale:

```
  1,000 tokens:   0.5 GB     (a short conversation)
  4,000 tokens:   2.0 GB     (a long chat)
  8,000 tokens:   4.0 GB     (a document summary)
  32,000 tokens:  16.0 GB    (= model weight size!)
  128,000 tokens: 64.0 GB    (exceeds a single H100)
```

For larger models, it's even worse:

| Model | Layers | Heads | d_head | Per Token | At 32K |
|-------|--------|-------|--------|-----------|--------|
| Llama 8B | 32 | 32 | 128 | 0.5 MB | 16 GB |
| Llama 70B | 80 | 64 | 128 | 2.5 MB | 80 GB |
| GPT-4 class | ~120 | ~96 | ~128 | ~5.6 MB | ~180 GB |

> **At 32K context, Llama 70B's KV cache alone fills an entire H100 GPU (80 GB).** There's no room left for the model weights, activations, or anything else.

---

## What Happens During Generation

It's worth emphasizing what happens at **every single generated token**:

```
For each new token:
  1. Compute Q, K, V for this token           ← small (one token)
  2. Append K, V to the cache                 ← cache grows by one entry
  3. Load ALL cached Keys from GPU memory     ← THIS IS THE BOTTLENECK
  4. Compute dot product of Q with every K    ← scales with cache size
  5. Softmax over all scores                  ← scales with cache size
  6. Load ALL cached Values from GPU memory   ← ANOTHER BOTTLENECK
  7. Weighted sum of Values                   ← scales with cache size
  8. Repeat for all 32 heads × 32 layers
```

Steps 3 and 6 are the killers. At every token, the GPU must **read the entire KV cache** from HBM (high-bandwidth memory) to SRAM (on-chip fast memory). For a 16 GB cache, that's 16 GB of memory reads **per token generated**.

On an H100 with ~3.35 TB/s memory bandwidth:

```
Time to read 16 GB cache = 16 / 3350 ≈ 4.8 ms per token
→ Max ~208 tokens/second (ignoring all other work)
```

Compare to the actual compute needed (the dot products themselves): that takes a fraction of a millisecond. **The GPU is spending 90%+ of its time waiting for memory, not computing.**

> **A smaller KV cache means less data to read, which means faster generation.** TurboQuant's 4-5× compression directly translates to faster inference, not just lower memory usage.

---

## The Compression Opportunity

Here's where we stand:

```
Current state:
  Each value in the KV cache: 16 bits (float16)
  Per token: 0.5 MB (Llama 8B)
  At 32K: 16 GB

If we could compress to 4 bits per value (4× compression):
  Per token: 0.125 MB
  At 32K: 4 GB                ← fits comfortably on any modern GPU

If we could compress to 3 bits per value (~5.3× compression):
  Per token: 0.094 MB
  At 32K: 3 GB                ← room for much longer contexts
```

The question is: **can you compress from 16 bits to 3-4 bits without breaking the attention mechanism?**

Naive approaches (just rounding to fewer bits) break in subtle ways. The next section explains why.

---

## Speaker Notes

- **The step-by-step generation walkthrough is essential.** Walk through steps 1, 2, and 3 slowly. The "AGAIN" emphasis (recomputing past tokens) should make the audience wince. That visceral reaction is what makes the caching motivation stick.
- **n² vs n:** If your audience is software engineers, the quadratic-to-linear improvement will resonate immediately. It's the same instinct as "don't put a query inside a for loop."
- **Draw the cache growth live** if you have a whiteboard. Start with one row, add a column for each token. Then stack 32 heads. Then stack 32 layers. Let the scale build visually.
- **The memory bandwidth calculation** (4.8 ms per token at 32K context) is a powerful moment. It reframes compression from "save memory" to "go faster" — which is often the more compelling argument for production engineers.
- **Don't introduce compression techniques yet.** This section ends with the question, not the answer. Section 5 explains why naive compression fails, and Sections 6+ build toward TurboQuant.
- **The per-model table** (8B, 70B, GPT-4 class) hits differently for different audience members. Some run 8B locally, some deploy 70B in production. The table ensures everyone sees a number that feels personally relevant.
- **Possible audience questions:**
  - "Why not just keep fewer tokens? Drop old ones?" — That's token eviction/pruning (SnapKV, H2O, etc.). It works but you lose information. TurboQuant keeps all tokens, just compressed.
  - "Can you use GQA/MQA to reduce heads?" — Yes, and modern models do (Llama uses GQA with 8 KV heads instead of 32). But even with GQA, the cache is still the bottleneck at long contexts. TurboQuant compresses whatever heads remain.
  - "What about offloading to CPU RAM?" — Possible but slow. CPU-GPU transfer bandwidth is ~32 GB/s vs 3350 GB/s HBM bandwidth. Offloading adds 100× latency per memory access.
- **Transition to Section 5:** "So we know the KV cache is massive, we know it's the bottleneck, and we know that compressing it would save both memory and time. But you can't just naively round numbers to fewer bits. Let me show you why."
