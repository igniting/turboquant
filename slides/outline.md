Here's the expanded 90-minute version with timing. The extra time lets you build proper foundations and go deeper where it matters.

---

**Talk Title:** *"TurboQuant: How Google Compresses LLM Memory 6× Without Losing a Single Answer"*

**Format:** 90 minutes total — ~75 min content + 15 min Q&A (with 2-3 natural pause points for questions during the talk)

---

## Part I: Setting the Stage (20 min)

---

**Section 1: The Memory Wall — Why You Should Care [5 min]**

Open with a concrete cost calculation that hits home: a model like Llama-3.1-8B serving 512 concurrent users at 32K context needs ~512 GB just for the KV cache — that's four H100 GPUs worth of memory doing nothing but storing past tokens. This is why your inference bill is so high, why context windows have practical limits even when models theoretically support 128K tokens, and why "just add more GPUs" is not a sustainable answer. Frame the rest of the talk as the journey to understanding one elegant algorithm that compresses this memory by 5-6× with zero accuracy loss and zero training cost. End with the punchline upfront: "By the end of this talk, you'll understand every step of how this works, and you'll be able to explain it to your team on Monday."

---

**Section 2: Transformers — The 10,000 Foot View [5 min]**

Don't assume the audience understands transformers beyond the name. Start with the simplest framing: a transformer is a function that takes a sequence of tokens and predicts the next one, doing this one token at a time during generation. The model is a stack of identical "layers" (typically 32-80 of them), and each layer has two parts: an attention block ("look at previous tokens and decide what's relevant") and a feed-forward block ("think about what you just gathered"). Everything in the model is vectors — a token gets converted to a vector (embedding), flows through layers as a vector, and the final vector gets decoded back into a token probability. Set up the key mental model: generation is sequential and autoregressive — you produce one token, append it, and repeat — which is why caching becomes essential.

---

**Section 3: Attention — The Heart of the Transformer [10 min]**

This section needs to be thorough since the audience knows the term but not the mechanics. Start with the intuition: attention answers "when predicting the next word, how much should I care about each previous word?" For each token, the model computes three vectors: Query ("what am I looking for?"), Key ("what do I represent?"), and Value ("what information do I carry"). The attention score between the current Query and a past Key is a dot product — a single number measuring alignment. Walk through a concrete example: in "The cat sat on the mat because it was tired", when processing "tired", the Query for "tired" has a high dot product with the Key for "cat" (they're semantically linked) and a low dot product with the Key for "on" (irrelevant). These raw scores go through softmax (turning them into a probability distribution that sums to 1), then you take a weighted sum of the Value vectors using these probabilities. Show the matrix form briefly: Attention(Q, K, V) = softmax(QK^T / √d) × V — but emphasize that the intuition matters more than the formula. End with multi-head attention: the model runs this process multiple times in parallel with different learned projections (typically 32 heads), each "head" looking for different types of relationships.

**[Pause for questions — 2 min]**

---

## Part II: The KV Cache Problem (15 min)

---

**Section 4: Why the KV Cache Exists [7 min]**

Walk through generation step by step. When you generate token 1, you compute Q₁, K₁, V₁. When you generate token 2, you need to attend to token 1, so you need K₁ and V₁ again — plus you compute Q₂, K₂, V₂. By token n, you need all of K₁...Kₙ₋₁ and V₁...Vₙ₋₁ to compute attention. Without caching, you'd recompute every Key and Value from scratch at every step — this means running the entire model on the full sequence for each new token, which is absurdly wasteful. The KV cache stores all previously computed Key and Value vectors so you only compute new ones for the current token. Draw the memory math on a whiteboard/slide: for each token, across each layer and each head, you store one Key vector and one Value vector, each of dimension d_head (typically 128), in float16 (2 bytes). So total cache = 2 × n_layers × n_heads × d_head × 2 bytes × seq_length. For Llama-3.1-8B: 2 × 32 × 32 × 128 × 2 bytes × seq_length = ~0.5 MB per token. At 32K context, that's ~16 GB — just for the cache of one user.

---

**Section 5: Why Compressing the KV Cache Is Hard [8 min]**

The naive approach — just round everything to fewer bits — fails in subtle ways. Walk through why with a worked example: take two 4-dimensional vectors, compute their true inner product, then round each coordinate to 2-bit precision and recompute. Show that the error isn't just small noise — it can change the relative ordering of attention scores, which softmax then amplifies exponentially. Explain the two flavors of error the paper cares about: MSE (how close is the reconstructed vector to the original?) and inner product error (how close is the reconstructed dot product to the true one?). MSE seems like the obvious metric, but the paper's key insight is that minimizing MSE doesn't automatically minimize inner product error — you can have low MSE but systematically biased inner products. Also explain why the KV cache is uniquely hard to compress: unlike model weights (static, can be compressed offline with calibration data), the KV cache is dynamic — new vectors arrive with every generated token, and you can't afford to run an expensive compression step each time. This rules out methods like GPTQ/AWQ that need calibration. You need an "online" quantizer: one that can compress any vector instantly, without seeing any other data.

**[Pause for questions — 3 min]**

---

## Part III: Quantization Fundamentals (15 min)

---

**Section 6: Scalar Quantization 101 [7 min]**

Build from first principles — most of the audience won't know how quantization works. Start with the simplest case: you have a number between -1 and 1, and you can only store 1 bit. Your only options are two "representative values" (centroids). If you pick -0.5 and +0.5, then any number < 0 maps to -0.5 and any number ≥ 0 maps to +0.5. Your "codebook" is {-0.5, +0.5} and your "quantization map" is just the sign. With 2 bits you get 4 buckets, with 3 bits you get 8 buckets, and so on. The key design question: where should you place the bucket boundaries and centroids to minimize average error? If your data is uniformly distributed, equal spacing is optimal. But if your data clusters around certain values (like a Gaussian — most values near zero, few at extremes), you want more buckets where the data is dense and fewer where it's sparse. This is exactly the Lloyd-Max algorithm (1960s): iteratively adjust bucket boundaries and centroids to minimize MSE, essentially k-means in 1D. Show a visual: Gaussian distribution with uniform spacing vs optimal spacing — the optimal one has tighter buckets near the peak and wider buckets in the tails.

---

**Section 7: From Scalar to Vector Quantization — The Independence Question [8 min]**

For a d-dimensional vector, the simplest approach is to quantize each coordinate independently using a scalar quantizer. But this ignores correlations between coordinates — if coordinate 3 is high, coordinate 7 tends to be low, and a joint quantizer could exploit this. The optimal approach (joint vector quantization) considers all coordinates together, but the codebook size explodes: for b bits per coordinate in d dimensions, you'd need 2^(b×d) entries — for d=128 and b=3, that's 2^384 entries, more than atoms in the universe. Product Quantization (PQ) compromises: split the vector into small subgroups (say 4-8 dimensions each), run k-means within each subgroup. This captures some correlation but requires expensive offline training. TurboQuant's radical insight: what if you could transform the vector so that coordinates become genuinely independent? Then scalar quantization per coordinate wouldn't sacrifice anything — you'd get near-optimal vector quantization for free. This is exactly what the random rotation does, and why it's the key breakthrough.

---

## Part IV: TurboQuant — The Algorithm (25 min)

---

**Section 8: The Random Rotation — Making Any Vector Predictable [10 min]**

This is the most important section of the talk — spend time here building deep intuition. Start with a 2D example on a whiteboard: take a vector (0.99, 0.01) on the unit circle — almost all its energy is in coordinate 1. A scalar quantizer tuned for "typical" values would do terribly on this vector. Now rotate it by a random angle — say 37 degrees — and you get roughly (0.79, 0.60). The energy is spread across both coordinates. The original vector was adversarial; the rotated one is "typical." Scale this to high dimensions: multiply by a random rotation matrix Π (generated via QR decomposition of a Gaussian matrix). After rotation, something remarkable happens — each coordinate of Π·x follows a known Beta distribution, regardless of what x was. In high dimensions (d=128 is enough), this Beta distribution looks almost exactly like a Gaussian N(0, 1/d). Even more importantly, different coordinates become nearly independent. This is a deep result from high-dimensional probability: on a high-dimensional sphere, any two coordinates of a random point carry almost no information about each other.

Use an analogy: imagine spinning a globe and picking a random point. The latitude and longitude of that point are nearly independent — knowing the latitude tells you almost nothing about the longitude. Same principle in higher dimensions, but even stronger.

Why this matters: since every coordinate has the same known distribution and coordinates are nearly independent, the optimal strategy is to design one scalar quantizer for that distribution and apply it to every coordinate. The per-coordinate quantizer is found by solving a 1D k-means problem (Eq. 4 in the paper) using Lloyd-Max, and you do this once offline for each bit-width you care about. At runtime, quantization is: (1) one matrix multiply (rotation), (2) one table lookup per coordinate. Both operations are perfectly vectorizable on GPUs.

Walk through the concrete codebook values: at 1 bit in high-d, the centroids are ±√(2/πd) ≈ ±0.063 for d=128. At 2 bits, four centroids: ±0.453/√d and ±1.51/√d. These are precomputed constants.

---

**Section 9: The Distortion Guarantee — How Good Is This? [5 min]**

State Theorem 1 in plain language: for b bits per coordinate, the MSE is at most (√3π/2) × (1/4^b). Walk through what this means concretely — each additional bit reduces error by 4×. Show the table: 1 bit → 0.36 MSE, 2 bits → 0.117, 3 bits → 0.03, 4 bits → 0.009. Then state the lower bound (Theorem 3): no algorithm in the universe can achieve MSE better than 1/4^b. So TurboQuant is within a factor of √3π/2 ≈ 2.7 of perfection. At 1 bit, it's only 1.45× off from the theoretical optimum. Explain the lower bound intuition without full proof: Shannon showed in the 1950s that compressing a random signal to B bits must lose at least a certain amount of information — this is a fundamental law of information theory, like the speed of light for physics. Yao's minimax principle then says: if no deterministic algorithm can handle random inputs well, no randomized algorithm can handle worst-case inputs well either.

---

**Section 10: The Bias Problem — When Minimizing MSE Isn't Enough [5 min]**

Now reveal the twist: the MSE-optimal quantizer from Section 8 is biased for inner products. Walk through why with a simple example. At 1 bit, the quantizer is essentially Qmse(x) = sign(Π·x), and the dequantized vector has norm √(2/π) ≈ 0.80 instead of 1. So every inner product gets multiplied by roughly 2/π ≈ 0.64 — a 36% underestimate. This isn't random noise — it's a systematic shrinkage. Why it matters: even though softmax is invariant to uniform scaling of all scores (adding a constant doesn't change the ranking), the bias here is not uniform. Show Figure 2 from the paper: the bias increases with the true inner product value. High-similarity pairs get underestimated more than low-similarity pairs. This distorts the attention distribution, pushing it toward uniform attention (ignoring the signal). At higher bit-widths the bias shrinks, but at 2-3 bits it's still meaningful enough to degrade model quality.

---

**Section 11: The Two-Stage Fix — QJL on the Residual [5 min]**

Introduce the solution as a two-stage pipeline. Stage 1: apply the MSE-optimal quantizer from Section 8 using (b-1) bits — this gives a good but biased approximation x̃. Stage 2: compute the residual r = x - x̃ (this is small because Stage 1 was good), then apply a 1-bit quantizer called QJL to the residual. QJL works by multiplying the residual by a random Gaussian matrix S and storing just the sign of each entry — literally 1 bit per dimension. The reconstruction adds a scaled version of S^T × signs back to x̃.

Why does this fix the bias? QJL has a proven mathematical property: its inner product estimate is unbiased (E[estimate] = true value). Since the MSE part gives a close approximation and QJL gives an unbiased correction for the residual, the combined estimate is unbiased overall. The variance of the correction is proportional to ∥r∥² (the residual norm squared), which is already small thanks to Stage 1 being a good MSE quantizer. Total bit budget: (b-1) + 1 = b bits per coordinate.

Use the analogy: Stage 1 is like a GPS giving you a position accurate to 10 meters (biased but close). Stage 2 is like a compass correction that removes the systematic bias using just 1 extra bit of information per dimension. Together, you get an unbiased position estimate that's almost as precise as the GPS alone.

**[Pause for questions — 3 min]**

---

## Part V: Does It Actually Work? (10 min)

---

**Section 12: Empirical Validation — Theory Meets Practice [3 min]**

Show Figure 3 from the paper: actual MSE and inner product errors plotted against the theoretical upper and lower bounds across bit-widths 1-5. The measured values sit right between the bounds, confirming the theory isn't just math — it predicts real-world behavior. Show Figure 1: the error distribution for TurboQuant_prod is centered at zero (unbiased) at every bit-width, while TurboQuant_mse shows visible rightward shift (bias) that shrinks with more bits. This is exactly what the theory predicted. Quick note on the dataset: these experiments used 100K vectors from DBpedia encoded with OpenAI embeddings at d=1536 — real-world, not synthetic.

---

**Section 13: KV Cache Experiments — The Headlines [4 min]**

Three headline results, each deserving its own slide. First, Needle-in-a-Haystack (Figure 4): the model must find a hidden sentence in up to 104K tokens. TurboQuant at 4× compression scores identically to the uncompressed model (0.997). SnapKV (0.858), PyramidKV (0.895), and KIVI (0.981) all show degradation, especially at longer contexts. Second, LongBench end-to-end (Table 1): TurboQuant at 3.5 bits matches full-precision quality exactly (50.06 average score). Even at 2.5 bits (a 6.4× compression!), it only drops to 49.44 — still beating KIVI at 3 bits. Note the practical detail: the 2.5-bit and 3.5-bit configurations come from giving more bits to outlier channels (32 channels at 3 bits + 96 channels at 2 bits = 2.5 average). Third, speed: unlike KIVI and PolarQuant which leave generated tokens unquantized, TurboQuant quantizes during streaming generation — it's truly online.

---

**Section 14: Nearest Neighbor Search — The Bonus Application [3 min]**

TurboQuant isn't just for KV cache — it applies anywhere you need compressed vector similarity search. Show Table 2: indexing time for 100K vectors at d=1536 is 0.001 seconds for TurboQuant vs 240 seconds for Product Quantization vs 2268 seconds for RabitQ. That's a 200,000× speedup in indexing. Show Figure 5: despite zero preprocessing, TurboQuant's recall equals or beats PQ at every bit-width and dimension tested. For anyone building vector databases or RAG pipelines, this means you can index vectors as they arrive in real-time with no k-means training phase, no codebook management, and better search quality than existing methods.

---

## Part VI: Practical Implications and Closing (10 min)

---

**Section 15: What the Paper Doesn't Tell You — Lessons from Community Implementations [5 min]**

This section bridges theory to practice and adds credibility by showing you've looked beyond the paper. Google hasn't released official code yet (expected Q2 2026), but community implementations have surfaced several important findings. First, Keys and Values are not equal: real LLMs have dramatically different K vs V magnitude distributions — Keys often have 10-100× larger norms. Since quantization error scales with norm², Keys need more bits than Values. The paper doesn't discuss this asymmetry. Second, the QJL stage (Stage 2) may actually hurt in practice: the variance QJL adds gets amplified by softmax, and at 3+ bits the MSE-only bias is small enough that the cure is worse than the disease. Several implementers report better quality with TurboQuant_mse alone. Third, 3-4 bits is the practical sweet spot: 2-bit works for Values but is too aggressive for Keys on most model families, while 4-bit gives diminishing returns over 3-bit. Fourth, the approach works on vision-language models too — community tests on Molmo2-8B with video input (11K+ visual tokens) show 3.76× compression with correct outputs.

---

**Section 16: Where This Fits in the Inference Stack [3 min]**

Position TurboQuant in the broader compression landscape. Model weights are compressed offline (GPTQ, AWQ, GGUF) — this is mature and widely deployed. Activations are compressed with hardware-native formats (FP8 on H100, NVFP4 on Blackwell GPUs). KV cache is the remaining frontier, and TurboQuant is the most promising approach because it's the only method that combines online operation, formal optimality guarantees, and GPU-friendly computation. These layers stack: you can quantize weights with AWQ, activations with FP8, AND the KV cache with TurboQuant. Each compounds the other's savings. Current integration status: llama.cpp has working community implementations including Apple Silicon Metal support, vLLM has a plugin and an open feature request for native support, and production-ready framework support is expected by mid-2026.

---

**Section 17: Closing — The Three-Step Summary [2 min]**

End with the simplest possible summary that the audience can repeat to colleagues. TurboQuant does three things: Rotate (make any vector's coordinates predictable and independent), Quantize (use a precomputed optimal codebook for 2-4 bits per coordinate), Correct (apply 1-bit QJL to the residual to remove inner product bias). It's online (no training data needed), fast (one matrix multiply + table lookup), and provably near-optimal (within 2.7× of the information-theoretic limit). The practical takeaway: at 3.5 bits per coordinate, you get 4.5× memory savings with literally zero quality loss on every benchmark tested. This changes the economics of LLM serving — expect it to become standard infrastructure within a year.

---

**Q&A [15 min]**

Prepare for likely questions: "How does this compare to FP8 quantization?" (different target — FP8 is for activations/weights, TurboQuant is specifically for KV cache), "Can I use this today?" (community implementations exist but no production-ready framework support yet), "Does this work with any model?" (yes — it's model-agnostic and data-oblivious), "What about the random rotation matrix — isn't that expensive to store?" (it's generated from a seed, not stored explicitly; or use a fast Walsh-Hadamard transform which is O(d log d) and needs no storage).

---

**Timing Summary:**

| Section | Duration |
|---|---|
| Part I: Setting the Stage (Sec 1-3) | 20 min |
| Q&A pause | 2 min |
| Part II: The KV Cache Problem (Sec 4-5) | 15 min |
| Q&A pause | 3 min |
| Part III: Quantization Fundamentals (Sec 6-7) | 15 min |
| Part IV: TurboQuant Algorithm (Sec 8-11) | 25 min |
| Q&A pause | 3 min |
| Part V: Results (Sec 12-14) | 10 min |
| Part VI: Practical + Closing (Sec 15-17) | 10 min |
| Final Q&A | 7 min |
| **Total** | **110 min buffer → 90 min delivery** |

The buffer accounts for the fact that sections always run shorter in delivery than in planning — you'll naturally trim as you read the room. If you're running long, Sections 7, 9, and 14 are the easiest to compress.
