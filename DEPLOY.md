# Deploying the dashboard

The Streamlit dashboard is a pure results viewer — it reads the JSONL files, needs
no Ollama and no API key — so it deploys anywhere Streamlit runs. A free live link is
the single highest-value thing you can add for a portfolio: recruiters click links,
they don't clone repos.

## Streamlit Community Cloud (free, ~3 minutes)

1. Push this repo to GitHub (public). See the push steps in the project README.
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
3. **Create app → Deploy a public app from GitHub**, and set:
   - **Repository:** `NeelMaddu268/llm-redteam`
   - **Branch:** `main`
   - **Main file path:** `dashboard/app.py`
   - **(Advanced settings) Python version:** 3.11 or newer (the code uses `StrEnum`)
4. **Deploy.** It installs `requirements.txt` and starts the app.

`results/` is gitignored, so on the cloud the dashboard automatically falls back to the
committed [`sample_results/`](sample_results/) datasets (the three-way model comparison,
the defenses run, and the denylist-obfuscation run) — so the live demo has real data to
show without you committing your local runs.

After it deploys you get a URL like `https://your-app.streamlit.app`. Put that link in:

- the GitHub repo's **About → Website** field,
- the top of the README,
- your résumé / LinkedIn.

## Notes

- **Update the GitHub link** in `dashboard/app.py` (the sidebar `view on github` href is
  a placeholder) and the README once you know your repo URL.
- To refresh the demo data, drop a new `*.jsonl` into `sample_results/` and push.
- Running locally still uses your real `results/` first; `sample_results/` is only the
  fallback when `results/` is empty.
