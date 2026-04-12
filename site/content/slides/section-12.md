---
title: "Empirical Validation — Theory Meets Practice"
weight: 12
part: "Part V — Does It Actually Work?"
---

![Empirical Validation](/img/s12-empirical.webp)

We've built up the theory across several sections. Does it hold up on real data? The paper validates on the **DBpedia Entities dataset** -- 100,000 real-world text embeddings from OpenAI's embedding model (1536 dimensions), with 1,000 separate query vectors for inner product accuracy.

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

## Post-Publication Validation: The Community Confirms

After the paper's March 2026 public release, independent developers validated TurboQuant's claims across a much broader set of conditions than the original paper covered:

| Validation axis | Paper covered | Community added |
|---|---|---|
| Embedding models | OpenAI embeddings (1536-dim) | sentence-transformers, Cohere, GTE, BGE |
| LLM families | Llama-3.1-8B, Ministral-7B | Qwen 2.5, Phi-3, DeepSeek, Mistral-Nemo |
| Hardware | V100/A100 GPU | M1/M2/M3/M5 Mac, RTX 3080-5090, AMD 6800XT/9070XT |
| Context lengths | Up to 104K | 128K+ on consumer hardware |
| Inference engines | Standalone PyTorch | llama.cpp, vLLM, MLX, Rust |

Key community findings that extend the paper:

1. **The 4x error reduction per bit holds across all tested model families** -- the information-theoretic argument is truly model-agnostic.
2. **WHT performs identically to Gaussian rotation** in all tested configurations (confirming Section 8's analysis), at ~18× lower computational cost.
3. **The 3.5-bit sweet spot** confirmed across models: at 3 bits Keys + 2 bits Values (non-integer average), quality matches full precision while delivering ~5× compression.
4. **V compression is nearly free** across all tested models: compressing Values to 2 bits with Keys at 3-4 bits produces no measurable quality loss, while compressing Keys aggressively degrades quality significantly.

> **What the paper proved mathematically, the community validated empirically across 30+ hardware configurations and 10+ model families within weeks of release.**
