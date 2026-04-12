# TurboQuant Presentation

An interactive slide presentation covering [TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (Google Research, ICLR 2026).

17 sections across 6 parts — from the memory wall problem through the mathematics of random rotation and scalar quantization, to empirical results, community benchmarks, and practical deployment guidance.

## Live Site

Deployed via GitHub Pages. Each section is a standalone page with a hero illustration, full explanatory text, tables, and code examples.

---

## What's Covered

| Part | Sections | Topics |
|------|----------|--------|
| I — Setting the Stage | 1–3 | The KV cache memory problem, transformer architecture, attention mechanism (MHA/GQA/MLA) |
| II — The KV Cache Problem | 4–5 | Why the KV cache exists, why naive compression fails, outlier channels |
| III — Quantization Fundamentals | 6–7 | Scalar quantization, Lloyd-Max codebooks, why vector quantization is intractable |
| IV — The Algorithm | 8–11 | Random rotation + WHT, distortion guarantee, the bias problem, two-stage QJL fix |
| V — Does It Actually Work? | 12–14 | Empirical validation, KV cache experiments, nearest neighbor search benchmarks |
| VI — Practical & Closing | 15–17 | Community findings, inference stack placement, resources |

---

## Key Findings Covered

**From the paper (ICLR 2026):**
- TurboQuant at 3.5 bits matches full-precision quality on Needle-in-a-Haystack (0.997) and LongBench (50.06) for Llama-3.1-8B
- Near-optimal by proof: within 2.72× of the information-theoretic lower bound
- 185,000× faster indexing than Product Quantization for vector search
- Beats PQ and RabitQ on recall at both 2-bit and 4-bit

**From community implementations (post March 2026):**
- V compression is essentially free — compressing Values to 2-bit has near-zero quality impact when Keys are maintained at 3-4 bits
- Keys dominate quality — all measurable degradation comes from Key compression
- QJL bias correction (Stage 2) backfires at 3+ bits due to softmax variance amplification; MSE-only is better in practice at those bit-widths
- Walsh-Hadamard Transform (WHT) performs identically to Gaussian rotation at ~18× lower computational cost
- Boundary layers (first 2 and last 2 transformer layers) are disproportionately sensitive; protecting them at higher precision recovers 37–91% of any quality gap
- `turbo4` (K=4b, V=2b) beats `q4_0` on perplexity at lower bit-width
- 104B parameter model at 128K context on a single MacBook Pro M5 Max (128 GB)

---

## Running Locally

Requires [Hugo](https://gohugo.io/) (tested with v0.159+).

```bash
cd site
hugo server
```

Open http://localhost:1313

## Building for Production

```bash
cd site
hugo --minify
```

Output goes to `site/public/`.

---

## Repository Structure

```
site/
  content/slides/    # 17 Hugo markdown pages (section-01.md through section-17.md)
  layouts/           # Hugo templates (landing page, section layout)
  static/css/        # Light academic theme (Source Serif + Source Sans)
  static/js/         # Keyboard navigation (arrow keys, Esc)
  static/img/        # WebP hero images (s01–s17), black-and-white vintage cartoon style
  hugo.toml          # Site configuration
.github/workflows/
  deploy.yml         # Hugo build + GitHub Pages deploy on push to main
```

---

## Hero Images

Each section has a custom AI-generated hero illustration in a black-and-white 1930s Fleischer Studios rubber-hose cartoon style, with labels embedded in the scenes. Section 17 uses green as an accent color.

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `→` or `l` | Next section |
| `←` or `h` | Previous section |
| `Esc` | Back to table of contents |

---

## Credits

- Paper: [TurboQuant](https://arxiv.org/abs/2504.19874) — Google Research (ICLR 2026)
- Community implementations: [turboquant+](https://github.com/community/turboquant-plus) and llama.cpp contributors
- Hero images: AI-generated illustrations
- Built with [Hugo](https://gohugo.io/)
