# Web GUI

A small **FastAPI + uvicorn** web GUI over the intelligence flow. The whole page
is a single sentence:

> **What happened to `[asset ▾]` within the last `[window ▾]` `?`**

Pick an asset and a window from the two dropdowns and press the animated **`?`**
button. It fires **two** requests at once and stacks the results:

1. **`price-event`** — confirms the move against real CoinGecko data (placeholder
   *"Confirming move…"*).
2. **`ask … --no-synth`** — retrieves the time-and-asset-windowed evidence behind
   the move (placeholder *"Fetching intel…"*), shown below the price box with each
   article's **relevance / accuracy score**.

It's a thin HTTP layer over [`detect_event()`](../crypto_intel/prices.py) and
[`retrieve_context()`](../crypto_intel/pipeline.py) plus one static page.

> ### No synthesis in the GUI — by design
> The GUI shows **retrieved evidence** (the `ask --no-synth` path): the ranked
> source articles, not a written answer. The full, citation-grounded **summary**
> of *why* the asset moved — the `ask` command, which calls Claude to synthesize a
> cited explanation — is **CLI-only**. It is intentionally *not* exposed over the
> web because it spends Anthropic API credits per request. Consequently the
> deployed web service needs **no `ANTHROPIC_API_KEY`**. See
> [DEPLOYMENT.md](DEPLOYMENT.md).

The web stack lives behind an optional extra so the core demo stays CLI-only.

## Install & run locally

```bash
pip install -e .[web]                    # adds fastapi + uvicorn on top of core

crypto-intel serve                       # http://127.0.0.1:8000
crypto-intel serve --host 0.0.0.0 --port 9000
crypto-intel serve --reload              # dev auto-reload
```

Equivalently, without the CLI wrapper:

```bash
uvicorn crypto_intel.web.app:app --port 8000
```

Then open **http://127.0.0.1:8000**. The asset dropdown is populated from
`assets.yaml`; the window dropdown offers **1 hour / 3 hours / 1 day / 3 days /
1 week**. **Defaults: BTC, 1 day (24h).**

> **Evidence depends on what's ingested.** Retrieval is filtered to
> `[now − window, now]`, so if the store hasn't been ingested recently the
> **Intel** box may be empty for short windows (it says so, and suggests a wider
> window). Run `crypto-intel ingest` first, or widen to 1 week.

## Windows & granularity

The window dropdown maps to hours `{1, 3, 24, 72, 168}`. The API bounds `hours`
to **1–720 (30 days)** and rejects anything outside that (HTTP 422). The bound
tracks CoinGecko's sampling granularity so the steepest-move sub-window stays
meaningful; the price response reports which was used:

| Window | CoinGecko granularity |
|---|---|
| ≤ 24h | 5-minutely |
| 1–90 days | hourly |
| > 90 days | daily (not reachable — above the 720h cap) |

## API endpoints

Four JSON/HTML endpoints (same-origin, no auth):

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Serves the single-page UI |
| `GET` | `/api/assets` | `{ "assets": [...tickers], "default": "BTC" }` for the dropdown |
| `GET` | `/api/price-event?asset=&hours=` | Runs `detect_event`; returns the event JSON |
| `GET` | `/api/evidence?question=&asset=&hours=` | Retrieves ranked evidence with scores. **When `DATABASE_URL` is set it reads from the shared pgvector DB (`ask-db`)**; otherwise from the local store. |

**`GET /api/price-event?asset=ETH&hours=168`:**

```json
{
  "asset": "ETH", "coingecko_id": "ethereum",
  "window_start": "2026-07-21T14:28:00+00:00", "window_end": "2026-07-28T14:28:00+00:00",
  "price_start": 1930.70, "price_end": 1872.62,
  "pct_change": -3.01, "max_drawdown_pct": 4.92,
  "move_start": "2026-07-22T17:00:00+00:00", "move_end": "2026-07-25T09:00:00+00:00",
  "direction": "down", "hours": 168, "granularity": "hourly"
}
```

**`GET /api/evidence?question=What+happened+to+ETH...&asset=ETH&hours=168`:**

```json
{
  "asset": "ETH", "hours": 168,
  "question": "What happened to ETH within the last 1 week?",
  "chunks": [
    {
      "source_name": "CryptoPotato",
      "url": "https://cryptopotato.com/...",
      "published_at": "2026-07-24T11:54:00+00:00",
      "snippet": "…ETH's Next Move? Earlier this week…",
      "score": 0.784, "semantic_score": 0.49, "bm25_score": 3.9,
      "assets": ["ETH", "BTC"]
    }
  ],
  "notes": []
}
```

`score` is the combined relevance/accuracy score (blended semantic similarity +
BM25); `semantic_score` and `bm25_score` are the components. When nothing matches
the window, `chunks` is empty and `notes` explains why.

**Error responses:**

| Status | When |
|---|---|
| `422` | `hours` outside 1–720, or `question` missing on `/api/evidence` |
| `400` | unknown asset, or too little price data in the window (`/api/price-event`) |
| `502` | upstream CoinGecko failure, or evidence retrieval failure |

Interactive API docs are available at `/docs` (FastAPI/Swagger).

## Frontend

The single page is split into `crypto_intel/web/static/`:

- `index.html` — structure (an editorial / broadsheet layout, set in **Charter**)
- `styles.css` — styling (paper/ink theme, light + dark)
- `app.js` — the two-call logic (price + evidence); calls `/api/*` **same-origin**
  with relative paths (`/api/assets`, `/api/price-event`, `/api/evidence`)
- `assets/` — the WhatHappenedCrypto logos (masthead + favicon)

FastAPI serves `/` → `index.html` and mounts the rest at `/static` (so the page
references `/static/styles.css`, `/static/app.js`, `/static/assets/…`).

Because the frontend uses **relative** `/api/*` paths, it can be hosted two ways:

- **All-in-one:** the FastAPI service serves both the static page and the API from
  one origin (local `serve`, or a single Railway service).
- **Split:** the static page on **Netlify**, the API on **Railway**, with a Netlify
  proxy rewriting `/api/*` → Railway so the relative paths still resolve (no CORS,
  no code change). This is the hosted topology — see [DEPLOYMENT.md](DEPLOYMENT.md).

## Get the full summary from the CLI

The GUI shows **retrieved** evidence only. For the full, citation-grounded
explanation of *why* the asset moved, use the CLI `ask` command:

```bash
crypto-intel ask "What happened to ETH within the last 1 week?"
```

The synthesis step needs an `ANTHROPIC_API_KEY`. See the
[README](../README.md) for the CLI quickstart.

## Security

The GUI is a **read-only** tool that queries a public API and a vector store. Run
locally it binds `127.0.0.1` by default — fine as-is. Do **not** expose it publicly
without adding rate-limiting and hardening. In the hosted deployment the API is
read-only, unauthenticated, and fronted by Netlify; keep an eye on CoinGecko
rate limits since the endpoint is public.

## Testing

Endpoint behavior is covered offline by `tests/test_web.py` using FastAPI's
`TestClient` with a mocked CoinGecko client and a stubbed retriever (no network,
no model download) — it skips cleanly if the `web` extra isn't installed.

```bash
pytest tests/test_web.py
```
