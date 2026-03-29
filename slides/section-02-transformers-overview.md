# Section 2: Transformers — The 10,000 Foot View

**Duration:** 5 minutes  
**Goal:** Give the audience just enough understanding of transformer architecture to follow the rest of the talk. No more, no less. They need to understand: tokens → vectors → layers → next token.

---

## Start with What They Know

> "You've all used ChatGPT, Claude, or Gemini. You type a message, and the model generates a response one word at a time. Let's look at what's actually happening under the hood — at the level that matters for understanding the memory problem."

---

## Step 1: Everything Is a Token

The first thing to understand is that LLMs don't see words — they see **tokens**.

A token is a chunk of text, roughly a word or a word fragment. The model has a fixed vocabulary of tokens (typically 32K-128K unique tokens), and every input gets split into a sequence of them.

```
Input:  "The cat sat on the mat"
Tokens: ["The", " cat", " sat", " on", " the", " mat"]
         Token 1  Token 2  Token 3  Token 4  Token 5  Token 6
```

Some practical rules of thumb:
- 1 token ≈ 4 characters in English
- 1 token ≈ 0.75 words
- A typical page of text ≈ 500 tokens

> **Why this matters for our story:** The KV cache stores one entry per token. More tokens = bigger cache. That's the linear growth from Section 1.

---

## Step 2: Tokens Become Vectors

Each token gets converted into a **vector** — a list of numbers that represents the meaning of that token in a high-dimensional space.

```
"cat" → [0.12, -0.45, 0.78, 0.03, ..., -0.31]   (4096 numbers for Llama 8B)
"dog" → [0.14, -0.42, 0.76, 0.05, ..., -0.28]   (similar to "cat"!)
"tax" → [-0.67, 0.23, -0.11, 0.89, ..., 0.44]   (very different)
```

The key intuition: **similar meanings → nearby vectors**. "Cat" and "dog" are close in this vector space. "Cat" and "tax" are far apart.

This initial conversion is called **embedding**, and the resulting vector is the token's starting representation. From here on, everything the model does is manipulating these vectors.

---

## Step 3: The Vector Flows Through Layers

A transformer is a **stack of identical processing layers** — typically 32 for an 8B model, 80 for a 70B model.

```
Token embedding (4096-dim vector)
        │
        ▼
  ┌─────────────┐
  │   Layer 1    │  ← Each layer refines the vector's meaning
  └─────┬───────┘
        ▼
  ┌─────────────┐
  │   Layer 2    │
  └─────┬───────┘
        ▼
       ...
        ▼
  ┌─────────────┐
  │   Layer 32   │
  └─────┬───────┘
        ▼
  Final vector → Next token prediction
```

Each layer does two things:

1. **Attention:** "Look at all the previous tokens and figure out which ones are relevant to me right now." (This is where the KV cache lives — we'll go deep in the next section.)

2. **Feed-forward network:** "Now think about what I've gathered and update my representation." (This is a standard neural network — no caching involved.)

The vector goes in, gets refined, comes out. Then into the next layer, gets refined again. By the end, the vector has been enriched with contextual understanding from all previous tokens across all layers.

---

## Step 4: Generation Is One Token at a Time

Here's the crucial part: **the model generates text one token at a time**.

```
Step 1: Input "The cat"      → Model predicts " sat"
Step 2: Input "The cat sat"  → Model predicts " on"
Step 3: Input "The cat sat on" → Model predicts " the"
...
```

At each step, the model:
1. Takes all the tokens so far
2. Runs them through all 32 layers
3. Produces a probability distribution over the next token
4. Picks one (sampling)
5. Appends it and repeats

This is called **autoregressive generation** — each new token depends on all previous tokens.

> **The efficiency problem:** At step 100, the model needs to process all 100 tokens through all 32 layers. At step 1000, all 1000 tokens. At step 10000, all 10000 tokens. Recomputing everything from scratch at each step would be disastrously slow.

This is why the KV cache exists — to avoid recomputing past tokens. Instead of rerunning the entire sequence through every layer, you cache the intermediate results (Keys and Values) and only compute the new token's contribution.

---

## The Mental Model You Need

Here's the simplified picture to carry forward:

```
┌──────────────────────────────────────────────────┐
│                  TRANSFORMER                      │
│                                                   │
│  Tokens → Vectors → [Layer 1] → [Layer 2] → ...  │
│                        │           │              │
│                     KV Cache    KV Cache          │
│                    (Layer 1)   (Layer 2)          │
│                                                   │
│  ... → [Layer 32] → Final Vector → Next Token     │
│            │                                      │
│         KV Cache                                  │
│        (Layer 32)                                 │
└──────────────────────────────────────────────────┘
```

Every layer has its own KV cache. Every token adds to every layer's cache. That's the multiplicative effect from Section 1.

You don't need to understand backpropagation, gradient descent, or training to follow the rest of this talk. You just need:

- **Tokens** are the units of text
- **Vectors** are how the model represents them (lists of numbers)
- **Layers** process vectors sequentially, each one refining the representation
- **Attention** (inside each layer) is how one token "looks at" previous tokens
- **The KV cache** stores past tokens' intermediate results to avoid recomputation
- **Generation** is autoregressive — one token at a time, each depending on all previous ones

The next section zooms into the **attention** mechanism — the one component that makes the KV cache necessary, and the one whose accuracy TurboQuant must preserve.

---

## Speaker Notes

- **Keep this section brisk.** 5 minutes max. The audience doesn't need to understand transformer internals deeply — they need just enough scaffolding to follow the attention and KV cache explanation in Sections 3 and 4.
- **Use the ASCII diagrams.** Draw the layer stack on a whiteboard if available. The visual of "tokens flowing through layers" is the single most important mental model to establish.
- **Don't mention Query/Key/Value here.** Save that for Section 3. Here, attention is just "look at previous tokens."
- **The autoregressive generation explanation is critical.** This is what motivates the KV cache. Make sure the audience understands *why* you'd want to cache — because the alternative (recomputing everything) is impossibly slow.
- **Possible audience questions:**
  - "What's the difference between parameters and tokens?" — Parameters are the model's learned weights (fixed after training). Tokens are the input/output text (different every conversation).
  - "Why 32 layers? Why not more or fewer?" — More layers = more capacity to learn complex patterns, but also more compute and memory. It's a design tradeoff.
  - "What's the difference between attention and the feed-forward network?" — Attention looks *across* tokens (horizontal). Feed-forward processes each token *independently* (vertical). You need both.
- **Transition to Section 3:** "Now let's zoom into the one component that creates all this memory pressure — the attention mechanism."
