---
  title: "Closing — The Three-Step Summary"
  weight: 17
  part: "Part VI — Practical & Closing"
  ---

  ![Closing — The Three-Step Summary](/img/s17-closing.webp)

  We started with a question: why does your GPU run out of memory when you give an LLM a long document? The answer was the KV cache -- a memory structure that grows with every token, often exceeding the model weights themselves.

  Then we asked: can you compress it? And we spent time understanding why that's hard -- you need to preserve inner products, not just vectors; softmax amplifies small errors; and the compression must work online, without ever seeing the data in advance.

  TurboQuant answers all of this with three operations.

  ---

  ## Three Operations

  ```
  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │                      TurboQuant                         │
  │                                                         │
  │    1.  ROTATE      Walsh-Hadamard Transform makes any   │
  │                    vector's coordinates predictable     │
  │                    and independent.                     │
  │                                                         │
  │    2.  QUANTIZE    A precomputed codebook snaps         │
  │                    each coordinate to 2-4 bits.         │
  │                    Near-optimal MSE.                    │
  │                                                         │
  │    3.  CORRECT     1-bit QJL on the residual            │
  │                    removes inner product bias.          │
  │                    (In practice: PolarQuant at 3+ bits) │
  │                                                         │
  │  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
  │                                                         │
  │  Online:       No training, no calibration, no data.    │
  │  Fast:         One transform + table lookup.            │
  │  Optimal:      Within 2.7× of the theoretical limit.   │
  │  Practical:    Zero quality loss at 3.5 bits (4.6×).    │
  │                                                         │
  └─────────────────────────────────────────────────────────┘
  ```

  ---

  ## Three Numbers to Remember

  **1.** Each additional bit reduces quantization error by **4x**.

  **2.** At 3.5 bits, KV cache compression is **quality-neutral** on every benchmark tested -- 4.6x smaller, identical outputs.

  **3.** TurboQuant is provably within **2.7x** of the best any algorithm could ever achieve. It's not a heuristic -- it's near-optimal by mathematical proof.

  ---

  ## The Bigger Picture

  TurboQuant is one algorithm, but it represents a broader shift. The era of solving AI infrastructure problems by buying more GPUs is ending. The next wave of progress comes from algorithms -- from understanding the mathematical structure of data and exploiting it.

  Shannon proved in the 1940s that there are fundamental limits to compression. Zador extended this to vectors in the 1960s. Lloyd and Max designed optimal scalar quantizers in the 1950s and 60s. The Johnson-Lindenstrauss lemma dates to 1984.

  TurboQuant combines ideas that are **40 to 80 years old** -- random rotation, optimal scalar quantization, information-theoretic lower bounds -- and applies them to a problem that didn't exist three years ago. That's the power of understanding fundamentals.

  The community has validated this: a 104B model running at 128K context on a single MacBook. Not a research demo -- a working implementation used by engineers today.

  ---

  ## Read the Paper, Try the Code

  - **The paper:** [TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (Google Research, ICLR 2026)
  - **turboquant+:** [github.com/TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) — the main community implementation (6.1k ★), CUDA/ROCm/Metal/CPU, validated to 104B
  - **llama.cpp fork:** [github.com/TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) — drop-in fork, use `--cache-type-k turbo3 --cache-type-v turbo3`
  - **Community discussion:** [llama.cpp discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969) (125 comments, 319 replies)
  - **Apple Silicon (Swift):** [github.com/ekryski/mlx-swift-lm](https://github.com/ekryski/mlx-swift-lm) — 144 tok/s on Qwen3.5-35B on M5 Max

  The algorithm is ready. The implementations are here. The only question is when it becomes standard infrastructure -- and the answer looks like: soon.
  