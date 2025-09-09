
# ICASSP Supplementary Materials — Demo Site Template

This repository is a ready-to-publish GitHub Pages site for supplementary materials (audio demos, interactive HTML plots) accompanying an ICASSP/IEEE paper.

## Quick Start

1. **Create a GitHub repo** named e.g. `icassp-2025-demo`.
2. Upload all files from this folder to the repo root (or push via git).
3. In **Settings → Pages**, set **Source** to `Deploy from a branch`, choose branch `main` (or `master`), and folder `/ (root)`.
4. Wait for Pages to build; your site will be available at: `https://<your-username>.github.io/<repo-name>/`.

## Add Your Files

- Put `.wav` files in `audio/`.
- Put interactive `.html` plots (e.g., Plotly exports) in `plots/`.
- Edit `data/manifest.json` to list items to display on the site.

### `data/manifest.json` Format

```jsonc
{
  "project": {
    "title": "Explainable Audio Codec — Supplementary Materials",
    "authors": "A. Taub, Y. Adi, et al.",
    "conference": "ICASSP 2025",
    "paper_link": "https://arxiv.org/abs/XXXX.XXXXX",
    "code_link": "https://github.com/<you>/<repo>",
    "doi_link": "https://zenodo.org/record/XXXXX"  // optional
  },
  "sections": [
    {
      "title": "Speech — Pitch Manipulation",
      "description": "VCTK samples across semitone shifts Δ.",
      "items": [
        {
          "type": "audio",
          "title": "Speaker p225 — Δ = +4",
          "file": "audio/vctk_p225_plus4.wav",
          "caption": "Encodec latents manipulated to +4 semitones."
        },
        {
          "type": "plot",
          "title": "3D PCA Trajectory",
          "file": "plots/pca_trajectory_p225.html",
          "caption": "Interactive PCA (3D) of latent trajectories."
        }
      ]
    },
    {
      "title": "Music — Instruments",
      "description": "Jamendo/GTZAN instrument clustering demos.",
      "items": [
        {
          "type": "audio",
          "title": "Violin solo",
          "file": "audio/violin_demo.wav",
          "caption": "Cluster center example."
        }
      ]
    }
  ]
}
```

> **Note:** Files are referenced with paths relative to the site root. If you place files in subfolders, include the subfolder in `file` (e.g., `audio/myfolder/sample.wav`).

## ICASSP/Citation Text

In your paper’s intro or footnote, add:
> “Additional audio samples and interactive visualizations are available on our project page: https://<your-username>.github.io/<repo-name>/.”

If you archive on Zenodo/OSF, also add a **Data Availability** note with the DOI.

## Local Preview

You can open `index.html` directly in a browser for a quick check. For full parity with GitHub Pages, serve with a local HTTP server:
```bash
python3 -m http.server 8000
# then visit http://localhost:8000
```

## Accessibility & Browser Support

- Audio players use native HTML5 `<audio controls>`.
- Plotly HTML exports are loaded via `<iframe>` so the interactive JS runs client-side.
- Large files: prefer 16‑bit PCM WAV or compressed OGG for preview; provide ZIP links when needed.

## License

MIT (see `LICENSE`).
