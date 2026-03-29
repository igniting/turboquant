# Section 3: Attention — The Heart of the Transformer

**Duration:** 10 minutes  
**Goal:** The audience should walk out of this section understanding: what Query, Key, and Value vectors are, why attention is a dot product, what softmax does, and why preserving inner products is the entire game for KV cache compression.

---

## The Intuition: Attention as a Question

Start with a sentence:

> **"The cat sat on the mat because it was tired"**

When the model is processing the word **"tired"**, it needs to figure out: *what does "it" refer to?* Is "it" the cat, the mat, or something else?

Attention is the mechanism that answers this question. It lets each token "look back" at all previous tokens and decide how much to care about each one.

In this example, the attention mechanism should assign:
- **High attention** to "cat" (because "it" = "cat", and "tired" describes the cat)
- **Medium attention** to "sat" (related action)
- **Low attention** to "on", "the", "mat" (structurally present but not semantically important)

> **Attention = a learned relevance score between every pair of tokens.**

But how does the model compute these relevance scores? That's where Query, Key, and Value come in.

---

## The Three Vectors: Query, Key, Value

At each layer, every token gets transformed into **three separate vectors**:

| Vector | Analogy | What It Represents |
|--------|---------|-------------------|
| **Query (Q)** | A search query | "What am I looking for?" |
| **Key (K)** | A document title/tag | "What do I contain?" |
| **Value (V)** | The document content | "What information do I carry?" |

Think of it like a search engine:
- The **Query** is what you type into the search bar
- The **Key** is the title/metadata that gets matched against your query  
- The **Value** is the actual content you get back from the matching documents

Each of these is a vector — a list of numbers. For Llama 3.1 8B, each Q, K, and V vector has **128 dimensions** (per attention head).

```
Token "tired" at Layer 5, Head 3:
  Query: [0.23, -0.11, 0.87, ..., 0.45]    ← 128 numbers
  Key:   [-0.34, 0.56, 0.12, ..., -0.78]   ← 128 numbers
  Value: [0.67, 0.03, -0.41, ..., 0.19]    ← 128 numbers
```

These three vectors are produced by multiplying the token's embedding by three different **learned weight matrices** — WQ, WK, WV. These matrices are part of the model's parameters (the billions of numbers you download).

---

## The Dot Product: Measuring Relevance

Here's where the magic happens. To compute how relevant token A is to token B, the model takes the **dot product** (also called **inner product**) of A's Query with B's Key.

### What Is a Dot Product?

For engineers who need a refresher: the dot product of two vectors is simply "multiply corresponding elements, then sum."

```
Query of "tired":  [0.2,  0.5, -0.3, 0.8]
Key of "cat":      [0.7,  0.4,  0.1, 0.6]
                    ───────────────────────
Dot product:       (0.2×0.7) + (0.5×0.4) + (-0.3×0.1) + (0.8×0.6)
                 = 0.14 + 0.20 + (-0.03) + 0.48
                 = 0.79  ← HIGH score → "cat" is relevant to "tired"
```

```
Query of "tired":  [0.2,  0.5, -0.3, 0.8]
Key of "on":       [-0.1, 0.1,  0.6, -0.2]
                    ───────────────────────
Dot product:       (0.2×-0.1) + (0.5×0.1) + (-0.3×0.6) + (0.8×-0.2)
                 = -0.02 + 0.05 + (-0.18) + (-0.16)
                 = -0.31  ← LOW score → "on" is not relevant to "tired"
```

> **High dot product = vectors point in similar directions = tokens are relevant to each other.**

The model computes this dot product between the current token's Query and the Key of **every previous token**. For a sequence of 1000 tokens, that's 1000 dot products per head per layer.

---

## Softmax: Turning Scores into Probabilities

The raw dot products are just numbers — they can be negative, they can be large. We need to convert them into a **probability distribution**: a set of non-negative numbers that sum to 1.

That's what **softmax** does.

```
Raw attention scores (dot products):

  "The"  "cat"  "sat"  "on"  "the"  "mat"  "because"  "it"  "was"
  -0.1    0.79   0.45  -0.31  -0.15   0.02    0.11    0.55   0.38

After softmax:

  "The"  "cat"  "sat"  "on"  "the"  "mat"  "because"  "it"  "was"
  0.04    0.18   0.13  0.03   0.04   0.05    0.06    0.14   0.12

  (these now sum to ~1.0)
```

Softmax does two important things:

1. **Makes everything positive** — even negative scores become small positive numbers
2. **Amplifies differences** — the highest score gets disproportionately more weight

That second property is critical and will matter later. Softmax is **exponential** — it amplifies small differences in scores into large differences in attention weights. A score of 0.79 vs 0.45 (a difference of 0.34) becomes an attention weight of 0.18 vs 0.13 (a ratio of 1.4×). If the scores were 7.9 vs 4.5, softmax would make the ratio much more extreme.

> **This is why inner product accuracy matters so much.** Even a small systematic error in dot products gets amplified by softmax, distorting which tokens the model pays attention to.

---

## Values: Gathering Information

Now we have attention weights — a probability distribution over all previous tokens telling us how much to "care" about each one. The final step is to use these weights to gather information.

Each previous token also has a **Value vector** — the actual information content that token carries. We take a **weighted sum** of all the Value vectors using our attention weights:

```
Output = 0.04 × V("The") + 0.18 × V("cat") + 0.13 × V("sat") + 0.03 × V("on") + ...
```

The result is a new vector that's a blend of information from all previous tokens, weighted by relevance. The token "tired" now carries a representation that's enriched with information about the cat (high weight) and much less about the mat (low weight).

This is the complete attention computation:

```
                    ┌─────────────────────────────────────┐
                    │         ATTENTION (one head)         │
                    │                                     │
Current token ──→ Query                                   │
                    │                                     │
                    ├──→ Dot product with each Key ────→ Scores
                    │         ↑                           │
                    │     K₁  K₂  K₃ ... Kₙ              │
                    │     (from KV cache)         Softmax │
                    │                                ↓    │
                    │                           Weights   │
                    │                                ↓    │
                    │     V₁  V₂  V₃ ... Vₙ    Weighted  │
                    │     (from KV cache)   ──→   Sum ──→ Output
                    │                                     │
                    └─────────────────────────────────────┘
```

---

## The Formula (One Slide, Don't Dwell)

For completeness, the attention formula is:

```
Attention(Q, K, V) = softmax(Q · Kᵀ / √d) × V
```

Where:
- **Q · Kᵀ** = matrix of all dot products between queries and keys
- **√d** = a scaling factor (d = dimension of each head, e.g., 128) to keep dot products from getting too large
- **softmax** = applied row-wise to get attention weights
- **× V** = weighted sum of value vectors

> You don't need to memorize this formula. The intuition is: **dot product → softmax → weighted sum**. That's the whole thing.

---

## Multi-Head Attention: Multiple Perspectives

One more detail: the model doesn't run attention once per layer — it runs it **multiple times in parallel**. These parallel instances are called **attention heads**.

Each head has its own set of Q, K, V weight matrices, so each head learns to look for **different types of relationships**:

- **Head 1** might learn to track syntactic dependencies (subject-verb agreement)
- **Head 2** might learn to track coreference ("it" → "cat")
- **Head 3** might learn to track positional patterns (adjacent words)
- **Head 17** might learn something we can't easily interpret

Llama 3.1 8B has **32 heads per layer**. Each head independently produces Q, K, V vectors of dimension 128, runs its own attention computation, and produces its own output. These outputs are concatenated and mixed to form the layer's final output.

```
Layer 5:
  Head 1:  Q₁(128-dim), K₁(128-dim), V₁(128-dim) → Output₁
  Head 2:  Q₂(128-dim), K₂(128-dim), V₂(128-dim) → Output₂
  ...
  Head 32: Q₃₂(128-dim), K₃₂(128-dim), V₃₂(128-dim) → Output₃₂

  Final output = Concat(Output₁, Output₂, ..., Output₃₂) × Wₒ
               = 32 × 128 = 4096-dim vector (same as input)
```

> **Why this matters for KV cache:** Each head stores its own Keys and Values independently. 32 heads × 32 layers = 1024 separate KV caches, each growing with every token. That's where the memory math from Section 1 comes from.

---

## The Takeaway for the Rest of This Talk

Three things to carry forward:

### 1. Attention is fundamentally about dot products (inner products)

The relevance score between any two tokens is a dot product of their Query and Key vectors. If we mess up these dot products, we mess up which tokens the model attends to, which means we mess up the model's outputs.

### 2. Softmax amplifies errors

Small errors in dot products get exponentially amplified by softmax. A compression scheme that introduces even a small **systematic bias** in inner products will distort the attention distribution — potentially changing which tokens get the most attention.

### 3. The KV cache stores Keys and Values — not Queries

Queries are computed fresh for the current token. Keys and Values from past tokens are what get cached and reused. This is why TurboQuant compresses K and V vectors specifically.

> **The entire challenge of KV cache compression, in one sentence:**  
> Compress Key and Value vectors to fewer bits, while preserving their dot products with Query vectors accurately enough that softmax produces the same attention distribution.

---

## Speaker Notes

- **The dot product examples are your anchor.** Work through at least one of them step-by-step with actual numbers. Engineers trust concrete computations, not hand-wavy analogies.
- **The search engine analogy** (Query = search query, Key = document title, Value = document content) works well for this audience. Return to it if you see confused faces.
- **Don't over-explain softmax.** "It turns scores into probabilities and amplifies differences" is sufficient. The exponential amplification point is the important one — it's what makes bias in inner products dangerous.
- **The ASCII diagram of the attention flow** is worth drawing live. Trace the path: current token → Query → dot product with cached Keys → softmax → weighted sum of cached Values → output.
- **Multi-head attention:** Don't spend more than 60 seconds on this. The audience needs to know that multiple heads exist (to understand the memory math) but doesn't need to understand why multiple heads are beneficial.
- **Foreshadow TurboQuant's challenge:** The sentence "compress K and V vectors while preserving dot products" is the setup for everything that follows. Repeat it. Make sure it lands.
- **Possible audience questions:**
  - "Why three vectors? Why not just compare embeddings directly?" — The learned Q/K/V projections let the model learn *what to look for* vs *what to advertise* vs *what to deliver* as separate concepts. Direct embedding comparison would be much less expressive.
  - "What are the actual numbers in these vectors? What do they mean?" — They don't have human-interpretable meaning. They're learned representations in a high-dimensional space. Think of them as coordinates in a "meaning space."
  - "How big are the weight matrices WQ, WK, WV?" — For Llama 8B, each is 4096×128 per head, times 32 heads = 4096×4096 total. Three of them per layer, 32 layers. That's a substantial fraction of the model's total parameters.
- **Transition to Section 4:** "Now you understand what's in the KV cache and why it matters. Let's look at exactly why it grows so fast and what happens if you try to naively compress it."
