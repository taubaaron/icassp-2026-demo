
# ICASSP Supplementary Materials — GitHub Pages Bundle

Production-ready static site for demos (audio, interactive HTML) accompanying an ICASSP/IEEE paper.

## Publish in 60 seconds
1. Create a new GitHub repo (e.g., `icassp-2025-supp`).
2. Upload everything in this folder to the repo root.
3. **Settings → Pages → Build and deployment**: `Deploy from a branch` → branch `main` → folder `/ (root)`.
4. Open the public URL shown by GitHub Pages.

## Add your content
- Put `.wav` files in `audio/` and interactive `.html` (Plotly, etc.) in `plots/`.
- Edit `data/manifest.json` to control sections and items (title, caption, relative file path).
- Optional: Put paper image/logo in `img/` and update `index.html` header if desired.

## Cite / link from your paper
Add a sentence like:  
> Additional audio samples and interactive visualizations are available on our project page: https://<username>.github.io/<repo-name>/

If you archive on **Zenodo/OSF**, add the DOI link in `manifest.json` and include a “Data Availability” note in the paper.

## Advanced
- Custom domain? Add `CNAME` with your domain and configure DNS.  
- Analytics: drop your script in `assets/analytics.js` (left blank by default).  
- Accessibility: alt texts/captions in `manifest.json` are surfaced next to controls.

MIT licensed.
