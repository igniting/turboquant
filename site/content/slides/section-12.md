---
title: "Empirical Validation — Theory Meets Practice"
weight: 12
part: "Part V — Does It Actually Work?"
---

![Empirical Validation](/img/s12-empirical.webp)

We've built up the theory across several sections. Does it hold up on real data? The paper validates on the **DBpedia Entities dataset** -- 100,000 real-world text embeddings from OpenAI's embedding model (1536 dimensions), with 1,000 separate query vectors for inner product accuracy.

> **Dimensionality note:** The paper validates at d=1536 (OpenAI embedding vectors). KV attention heads in Llama, Mistral, and Gemma use d=128 — a 12× difference. The WHT's Gaussianization quality improves with d, so d=1536 is the "easy" case where theory and experiment converge most cleanly. The community experiments below fill the gap: they validate the algorithm at d=128, the harder and more practically relevant case.

---

## Every Theoretical Prediction Is Confirmed

The paper tests four specific predictions, and all four match experiment:

| What the theory predicted | What the experiments showed |
|---|---|
| MSE $\leq 2.72 \times 4^{-b}$ | Measured MSE falls between upper and lower bounds at every bit-width (1 through 5). The constants match, not just the order of magnitude. |
| Each bit reduces error by 4x | Consistent ~4x reduction observed at each step |
| TurboQuant_prod is unbiased | Inner product error distribution is symmetric and centered at zero |
| TurboQuant_mse has bias that grows with true IP | Error shifts rightward (underestimates), and the shift is larger for higher true inner products -- exactly the non-uniform bias from Section 10 |

---

## MSE-Only vs Two-Stage: The Bias Is Visible

The clearest result: when you plot the inner product error distribution for both variants, TurboQuant_prod gives a symmetric bell curve centered at zero (unbiased), while TurboQuant_mse gives a bell curve shifted to the right (biased -- systematically underestimates inner products).

At 2-bit quantization, the bias pattern is stark:

```
TurboQuant_prod:                    TurboQuant_mse:

  All IP magnitudes → error          Low IPs  → small bias
  centered at zero                   High IPs → large bias
  (unbiased ✓)                       (non-uniform bias ✗)
```

At higher bit-widths (3-4 bits), the MSE variant's bias shrinks and becomes harder to distinguish from noise -- foreshadowing the practical finding in Section 15 that MSE-only is often sufficient.

> **The theory isn't aspirational -- it's descriptive.** Every prediction matches the measurement. But vector-level metrics are just a proxy. The real test is: does the model still give correct answers?

---

## Post-Publication Validation: The Community Confirms at d=128

After the paper's March 2026 public release, independent developers validated TurboQuant's claims across a much broader set of conditions than the original paper covered — critically including d=128, the actual attention head dimension for most deployed models.

| Validation axis | Paper (d=1536) | Community (d=128) |
|---|---|---|
| Embedding models | OpenAI text-embedding-3 | sentence-transformers, Cohere, GTE, BGE |
| LLM families | — | Llama, Mistral, Qwen, Phi, DeepSeek |
| Hardware | V100/A100 GPU | M1–M5 Mac, RTX 3080–5090, AMD 6800XT/9070XT |
| Context lengths | — | Up to 128K on consumer hardware |
| Inference engines | Standalone PyTorch | llama.cpp, vLLM, MLX, Rust |

Key community findings that extend the paper:

1. **The 4× error reduction per bit holds at d=128 across all tested model families** -- the information-theoretic argument is model-agnostic and holds even at the lower dimensionality where Gaussianization is less complete.
2. **WHT performs identically to Gaussian rotation at d=128** in all tested configurations, at ~18× lower computational cost. The kurtosis of Qwen3 KV tensors dropped from 900 to ~2.9 after WHT (target Gaussian kurtosis: 3.0).
3. **The 3.5-bit sweet spot** confirmed at d=128: 3 bits Keys + 2 bits Values (non-integer average) matches full-precision quality across six LongBench task categories.
4. **V compression is nearly free at d=128** across all tested models: compressing Values to 2 bits with Keys at 3-4 bits produces no measurable quality loss. Keys dominate the error budget because of their larger norms.

Note on model family taxonomy: the paper validates on Llama-3.1-8B and Ministral-7B. The community added Qwen 2.5, Phi-3, and DeepSeek as architecturally distinct families. Mistral-7B and Mistral-Nemo share the same GQA architecture and should be counted as one family; the community's contribution is Qwen, Phi, and DeepSeek (which uses MLA), where the attention head structure genuinely differs.

> **What the paper proved mathematically at d=1536, the community validated empirically at d=128 across 30+ hardware configurations and 10+ model families within weeks of release.**
