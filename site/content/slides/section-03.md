---
title: "Attention — The Heart of the Transformer"
weight: 3
part: "Part I — Setting the Stage"
---

![Attention: learned relevance score between tokens](/img/s03-attention-heart.webp)

Consider this sentence:

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

Each of these is a vector -- a list of numbers. For Llama 3.1 8B, each Q, K, and V vector has **128 dimensions** (per attention head).

```
Token "tired" at Layer 5, Head 3:
  Query: [0.23, -0.11, 0.87, ..., 0.45]    <- 128 numbers
  Key:   [-0.34, 0.56, 0.12, ..., -0.78]   <- 128 numbers
  Value: [0.67, 0.03, -0.41, ..., 0.19]    <- 128 numbers
```

These three vectors are produced by multiplying the token's embedding by three different **learned weight matrices** -- $W_Q$, $W_K$, $W_V$. These matrices are part of the model's parameters (the billions of numbers you download).

---

## The Dot Product: Measuring Relevance

To compute how relevant token A is to token B, the model takes the **dot product** (also called **inner product**) of A's Query with B's Key.

### What Is a Dot Product?

The dot product of two vectors is simply "multiply corresponding elements, then sum."

```
Query of "tired":  [0.2,  0.5, -0.3, 0.8]
Key of "cat":      [0.7,  0.4,  0.1, 0.6]
                    -----------------------
Dot product:       (0.2*0.7) + (0.5*0.4) + (-0.3*0.1) + (0.8*0.6)
                 = 0.14 + 0.20 + (-0.03) + 0.48
                 = 0.79  <- HIGH score -> "cat" is relevant to "tired"
```

```
Query of "tired":  [0.2,  0.5, -0.3, 0.8]
Key of "on":       [-0.1, 0.1,  0.6, -0.2]
                    -----------------------
Dot product:       (0.2*-0.1) + (0.5*0.1) + (-0.3*0.6) + (0.8*-0.2)
                 = -0.02 + 0.05 + (-0.18) + (-0.16)
                 = -0.31  <- LOW score -> "on" is not relevant to "tired"
```

> **High dot product = vectors point in similar directions = tokens are relevant to each other.**

The model computes this dot product between the current token's Query and the Key of **every previous token**. For a sequence of 1000 tokens, that's 1000 dot products per head per layer.

---

## Softmax: Turning Scores into Probabilities

The raw dot products are just numbers -- they can be negative, they can be large. We need to convert them into a **probability distribution**: a set of non-negative numbers that sum to 1.

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

1. **Makes everything positive** -- even negative scores become small positive numbers
2. **Amplifies differences** -- the highest score gets disproportionately more weight

That second property is critical and will matter later. Softmax is **exponential** -- it amplifies small differences in scores into large differences in attention weights.

> **This is why inner product accuracy matters so much.** Even a small systematic error in dot products gets amplified by softmax, distorting which tokens the model pays attention to.

---

## Values: Gathering Information

Now we have attention weights -- a probability distribution over all previous tokens telling us how much to "care" about each one. The final step is to use these weights to gather information.

Each previous token also has a **Value vector** -- the actual information content that token carries. We take a **weighted sum** of all the Value vectors using our attention weights:

```
Output = 0.04 * V("The") + 0.18 * V("cat") + 0.13 * V("sat") + 0.03 * V("on") + ...
```

The result is a new vector that's a blend of information from all previous tokens, weighted by relevance.

This is the complete attention computation:

```
                    +-------------------------------------+
                    |         ATTENTION (one head)         |
                    |                                     |
Current token --> Query                                   |
                    |                                     |
                    +---> Dot product with each Key ---> Scores
                    |         ^                           |
                    |     K1  K2  K3 ... Kn               |
                    |     (from KV cache)         Softmax |
                    |                                |    |
                    |                           Weights   |
                    |                                |    |
                    |     V1  V2  V3 ... Vn    Weighted   |
                    |     (from KV cache)   --> Sum  --> Output
                    |                                     |
                    +-------------------------------------+
```

---

## The Formula

For completeness, the attention formula is:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d}}\right) \times V$$

Where:
- $Q \cdot K^T$ = matrix of all dot products between queries and keys
- $\sqrt{d}$ = a scaling factor ($d$ = dimension of each head, e.g., 128) to keep dot products from getting too large
- **softmax** = applied row-wise to get attention weights
- $\times V$ = weighted sum of value vectors

> The intuition is: **dot product -> softmax -> weighted sum**. That's the whole thing.

---

## Multi-Head Attention: Multiple Perspectives

The model doesn't run attention once per layer -- it runs it **multiple times in parallel**. These parallel instances are called **attention heads**.

Each head has its own set of Q, K, V weight matrices, so each head learns to look for **different types of relationships**:

- **Head 1** might learn to track syntactic dependencies (subject-verb agreement)
- **Head 2** might learn to track coreference ("it" -> "cat")
- **Head 3** might learn to track positional patterns (adjacent words)

Llama 3.1 8B has **32 heads per layer**. Each head independently produces Q, K, V vectors of dimension 128, runs its own attention computation, and produces its own output.

```
Layer 5:
  Head 1:  Q1(128-dim), K1(128-dim), V1(128-dim) -> Output1
  Head 2:  Q2(128-dim), K2(128-dim), V2(128-dim) -> Output2
  ...
  Head 32: Q32(128-dim), K32(128-dim), V32(128-dim) -> Output32

  Final output = Concat(Output1, ..., Output32) * Wo
               = 32 * 128 = 4096-dim vector (same as input)
```

> **Why this matters for KV cache:** Each head stores its own Keys and Values independently. 32 heads x 32 layers = 1024 separate KV caches, each growing with every token. That's where the memory math from Section 1 comes from.

---

## The Takeaway

Three things to carry forward:

### 1. Attention is fundamentally about dot products (inner products)

The relevance score between any two tokens is a dot product of their Query and Key vectors. If we mess up these dot products, we mess up which tokens the model attends to.

### 2. Softmax amplifies errors

Small errors in dot products get exponentially amplified by softmax. A compression scheme that introduces even a small **systematic bias** in inner products will distort the attention distribution.

### 3. The KV cache stores Keys and Values -- not Queries

Queries are computed fresh for the current token. Keys and Values from past tokens are what get cached and reused. This is why TurboQuant compresses K and V vectors specifically.

> **The entire challenge of KV cache compression, in one sentence:**
> Compress Key and Value vectors to fewer bits, while preserving their dot products with Query vectors accurately enough that softmax produces the same attention distribution.
