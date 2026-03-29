# TurboQuant: KV Cache Compression for LLMs

A static website presenting [TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate](https://arxiv.org/abs/2504.19874) by Google Research.

The presentation walks through the complete algorithm in 17 sections across 6 parts — from the memory wall problem through the mathematics of random rotation and scalar quantization, to empirical results and practical deployment guidance.

## Running locally

Requires [Hugo](https://gohugo.io/) (tested with v0.159+).

```bash
cd site
hugo server
```

Open http://localhost:1313

## Building for production

```bash
cd site
hugo --minify
```

Output goes to `site/public/`.

## Structure

```
slides/              # Original speaker-note-style source material
site/
  content/slides/    # 17 Hugo content pages (reader-facing)
  layouts/           # Hugo templates (landing page, slide layout)
  static/css/        # Light academic theme (Source Serif + Source Sans)
  static/js/         # Keyboard navigation (arrow keys, Esc)
  static/img/        # Compressed WebP hero images per section
  hugo.toml          # Site configuration
```

## Keyboard shortcuts (on slide pages)

| Key | Action |
|-----|--------|
| `→` or `l` | Next section |
| `←` or `h` | Previous section |
| `Esc` | Back to table of contents |

## Credits

- Paper: [TurboQuant](https://arxiv.org/abs/2504.19874) by Google Research
- Hero images: AI-generated illustrations per section
- Built with [Hugo](https://gohugo.io/)
