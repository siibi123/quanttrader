# QuantTrader

Paper-only autonomous desk. Streamlit UI over `core/` + `quant/` + `ai/`.

## Streamlit Cloud secrets

Runtime files (`runtime/broker.json`, `audit.jsonl`, …) are **deleted on every Cloud reboot**. Persistence is a private GitHub Gist.

Add this to **App settings → Secrets** (TOML):

```toml
GITHUB_TOKEN = "ghp_your_token_here"
```

Optional (only if you already have a gist):

```toml
GIST_ID = "the_gist_id"
```

### How to make the token (gist scope only)

1. Open [github.com/settings/tokens](https://github.com/settings/tokens)
2. **Generate new token (classic)**
3. Note: `quanttrader gist`
4. Expiration: 90 days or no expiration
5. Tick **gist** only — leave every other scope off
6. Generate, copy the token, paste it into Streamlit Secrets as `GITHUB_TOKEN`
7. Reboot the Cloud app once so it hydrates

The TRADES tab shows **Last saved to GitHub · … UTC** after the first fill.

Local runs: copy `.env.example` → `.env`. No token means local `runtime/` only.

## Run tests

```
python tests/test_core.py
```
