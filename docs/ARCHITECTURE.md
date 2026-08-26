# Architecture & design

The design reference for the Crypto Market Intelligence Assistant: what it is, the
pipeline, the data model, module responsibilities, and the build history. For
usage see the [README](../README.md); for the web GUI see [GUI.md](GUI.md); for
deploying the hosted stack see [DEPLOYMENT.md](DEPLOYMENT.md).

## What this is about

Traders and analysts see a sharp price move and immediately ask *why*. The answer
is usually scattered across a crypto-news article, an exchange announcement, or an
SEC/CFTC press release published in the same window. Manually hunting for it is
slow and easy to get wrong.

This is a small RAG system that does two things a generic "chat over documents"
bot does not:

1. **Grounds the question in a real market event.** When asked about a move, it
   pulls actual price data, confirms the magnitude and — importantly — pins down
   *when* the move happened. That timestamp window drives retrieval, so the system
   reads the news that was published *around the move*, not just anything topically
   similar.
2. **Answers with citations.** Every claim in the synthesized explanation maps back
   to a specific source document (source name, URL, timestamp), so the answer is
   auditable rather than a vibe.

Intended user: an analyst or trader who wants a fast, sourced first-draft
explanation of a move. It is a decision-support tool, **not** investment advice.

**Honest framing:** this is a *demo*, not a production data platform. It ingests a
modest, configurable set of public feeds on demand and stores them locally (or in
a shared pgvector DB for the hosted deployment). Coverage is deliberately narrow
and reproducible.

## Scope

**In scope**

- On-demand ingestion from **free, public** sources: RSS feeds for crypto news,
  exchange announcements, and SEC/CFTC press releases; CoinMarketCap headlines
  (optional, API-keyed); Reddit (optional, opt-in).
- Normalization of every source into one `Document` schema, with automatic asset
  (ticker) tagging and deduplication.
- Local chunking + embedding + a persistent vector store (embedded Chroma, or
  Postgres/pgvector).
- Real price data and **price-event detection** (confirm the move, find the window)
  via CoinGecko's free API.
- Time-windowed, asset-filtered retrieval combining semantic similarity with an
  optional keyword (BM25) rerank.
- Answer synthesis with the Anthropic Claude API, returning a grounded explanation
  with numbered inline citations.
- A `typer` CLI: `ingest`, `stats`, `price-event`, `ask`, `eval`, `prune`, `ask-db`,
  the S9 forecasting commands `train` / `predict`, the S10 `warehouse` group, and
  the S11 `monitor` command (+ `train --track`).
- An optional **FastAPI web GUI** over `price-event` + evidence retrieval (no
  synthesis — see [GUI.md](GUI.md)).
- A `pytest` suite (with fixtures + a synthetic price series) so the core logic
  runs and is testable offline.

**Out of scope**

- Real-time streaming / websockets / continuous background ingestion (ingestion is
  a manual command, or a scheduled cron in the hosted deployment).
- Auth. The web GUI is read-only and unauthenticated.
- Trading, order routing, or portfolio features. No buy/sell logic, no financial
  advice.
- Historical backfill or a research-grade corpus. Retrieval quality is bounded by
  whatever was ingested in the recent lookback window.
- Multi-language sources (English only).

## Tech stack

Python-first. Versions are known-good minimums; exact versions are pinned in
`pyproject.toml`.

| Concern | Choice | Why |
|---|---|---|
| Language / runtime | Python 3.11+ (developed on 3.14) | Good typing + async ergonomics. |
| Packaging / env | `pip` + venv (or `uv`) | Single `pyproject.toml`; core install is torch-free. |
| CLI | `typer>=0.12` | Clean subcommands, typed args, good `--help`. |
| Config / secrets | `pydantic-settings>=2.0` | Typed settings from env + `.env`, validated at startup. |
| HTTP | `httpx>=0.27` | Modern client for CoinGecko / CMC calls; timeouts + retries. |
| RSS | `feedparser>=6.0` | Robust parsing of messy real-world feeds. |
| Reddit (optional) | `praw>=7.7` | Mature Reddit API wrapper; free script-app credentials. |
| Embeddings (default) | `chromadb`-bundled **ONNX** `all-MiniLM-L6-v2` (384-dim) | Local, free, deterministic, **no torch**. |
| Embeddings (optional) | `sentence-transformers>=3.0` (`[st]` extra) | Same model via torch; opt-in. |
| Vector store (default) | `chromadb>=0.5` (persistent, local) | Embedded, zero infra, metadata filters (asset, time, source). |
| Vector store (optional) | Postgres + `pgvector` (`[pg]` extra) | Shared DB for the hosted/Supabase deployment. |
| Keyword rerank | `rank-bm25>=0.2` | Cheap lexical signal to complement dense retrieval. |
| Prices | CoinGecko free API (via `httpx`) | Free, no key for basic endpoints; hourly/5-minutely market data. |
| LLM synthesis | `anthropic` SDK, default `claude-sonnet-5` | Strong grounded-synthesis quality. `claude-haiku-4-5-20251001` configurable for cheaper runs. |
| Web GUI (optional) | `fastapi>=0.110` + `uvicorn>=0.29` (`[web]` extra) | Thin HTTP layer over price-event + evidence retrieval. |
| Config data | `PyYAML>=6.0` | Human-editable `feeds.yaml` / `assets.yaml`. |
| Testing | `pytest>=8.0` (`[dev]` extra) | Unit tests for the offline-testable core. |

> **As-built note.** The original design named Reddit + sentence-transformers as
> primary. In practice the keyless **RSS feeds are the content workhorse**,
> CoinMarketCap was brought forward as the default optional connector, Reddit is
> opt-in, and the **ONNX** embedding backend is the default (torch won't install
> under this repo's long Windows path). See [As-built notes](#as-built-notes).

## Pipeline architecture

```
        ┌────────────────────────────────────────────────────────────┐
        │                       Data sources                          │
        │  News RSS · Exchange RSS · SEC/CFTC RSS ·                    │
        │  CoinMarketCap (optional, API) · Reddit (optional, API)     │
        └───────────────────────────┬────────────────────────────────┘
                                     │  ingest/*  connectors (per-source)
                                     ▼
        ┌────────────────────────────────────────────────────────────┐
        │  normalize.py → Document{source,url,ts,author,text,assets}  │
        │  · asset tagging (assets.yaml aliases)  · dedup (id hash)   │
        └───────────────────────────┬────────────────────────────────┘
                                     │  chunking.py + embeddings.py
                                     ▼
        ┌────────────────────────────────────────────────────────────┐
        │  store.py → get_store(): Chroma (default) | pgvector        │
        │  chunk text + embedding + metadata{source,assets,ts,url}   │
        └───────────────────────────┬────────────────────────────────┘
                                     ▲                     │
   prices.py: CoinGecko OHLC        │                     │  retrieve.py
   → PriceEvent{asset,window,pct} ──┘                     │  · parse question (asset, time)
                            (window drives the time filter)│  · metadata filter: asset ∈ assets
                                                           │    AND ts ∈ [window]
                                                           │  · semantic top-N  → BM25 rerank
                                                           ▼
        ┌────────────────────────────────────────────────────────────┐
        │  context bundle: PriceEvent + top-k chunks (+ provenance)   │
        └───────────────────────────┬────────────────────────────────┘
                                     │  synthesize.py (Claude)
                                     ▼
        ┌────────────────────────────────────────────────────────────┐
        │  Answer{answer_text, citations[], event, retrieved_ids}     │
        │  numbered inline citations [1],[2] → source url + snippet   │
        └────────────────────────────────────────────────────────────┘
```

Orchestration entrypoints in `pipeline.py`:

- `ingest_all(...)` — runs the selected connectors, normalizes, chunks, embeds,
  upserts (`--if-empty` skips if the store is already populated).
- `retrieve_context(...)` — detects the price event (optional), retrieves the
  time-windowed context; the shared retrieval path.
- `ask(...)` — `retrieve_context` + synthesis into a cited `Answer`.
- `ask_db(...)` — retrieval forced against the **pgvector** backend (the hosted
  GUI's read path), independent of `STORE_BACKEND`.
- `prune_all(...)` — rolling-window retention (deletes old chunks + JSONL docs).

## Data model

```python
# models.py — pydantic models (all timestamps are timezone-aware UTC)

class Document(BaseModel):
    id: str                     # sha1(source_name + url + published_at), stable dedup key
    source: SourceType          # "reddit"|"news"|"exchange"|"regulator"|"cmc"
    source_name: str            # "r/ethereum" | "CoinDesk" | "Kraken" | "SEC" | ...
    url: str
    title: str | None
    text: str                   # cleaned body used for chunking
    author: str | None
    published_at: datetime      # UTC
    assets: list[str]           # tagged tickers, e.g. ["ETH", "BTC"]
    metadata: dict = {}         # source-specific extras (score, flair, feed, ...)

class Chunk(BaseModel):
    id: str                     # f"{doc_id}:{chunk_index}"
    doc_id: str
    chunk_index: int
    text: str
    # stored with denormalized metadata for filtering:
    #   source, source_name, url, published_at (epoch int), assets (+ has_<TICKER> flags)

class PriceEvent(BaseModel):
    asset: str                  # "ETH"
    coingecko_id: str           # "ethereum"
    window_start: datetime      # UTC — start of the analysed window
    window_end: datetime        # UTC — "now" (or requested end)
    price_start: float
    price_end: float
    pct_change: float           # signed, over the window
    max_drawdown_pct: float     # largest adverse intra-window move
    move_start: datetime        # start of the steepest adverse leg (drives retrieval)
    move_end: datetime          # end of the steepest adverse leg
    direction: Literal["up", "down", "flat"]

class Citation(BaseModel):
    n: int                      # citation number used in answer_text ([1], [2], ...)
    doc_id: str
    source_name: str
    url: str
    published_at: datetime
    snippet: str

class Answer(BaseModel):
    question: str
    event: PriceEvent | None
    answer_text: str            # contains inline [n] markers
    citations: list[Citation]
    retrieved_chunk_ids: list[str]
    notes: list[str] = []       # e.g. "thin evidence", "no news in window"
```

`crypto_intel/data/assets.yaml` maps each ticker to its CoinGecko id + tagging
aliases:

```yaml
ETH:  {coingecko_id: ethereum, aliases: [eth, ether, ethereum]}
BTC:  {coingecko_id: bitcoin,  aliases: [btc, bitcoin, xbt]}
SOL:  {coingecko_id: solana,   aliases: [sol, solana]}
# ... extend as needed
```

## Module responsibilities

- `crypto_intel/cli.py` — `typer` app; wires subcommands (`ingest`, `stats`,
  `price-event`, `ask`, `eval`, `prune`, `ask-db`, `serve`, `train`, `predict`,
  `warehouse`, `monitor`) to `pipeline.py`, `forecast/`, `warehouse/`, and `mlops/`.
- `crypto_intel/config.py` — `pydantic-settings` object: API keys, paths, model
  name, embedding/store backend, defaults (lookback, k). Loads `.env`.
- `crypto_intel/models.py` — the pydantic models above; `SourceType` enum.
- `crypto_intel/ingest/base.py` — `Connector` protocol: `fetch(lookback_hours) ->
  Iterable[RawItem]`; shared `RawItem` dataclass.
- `crypto_intel/ingest/rss.py` — `RSSConnector`: generic feed reader; one instance
  per feed entry in `feeds.yaml` (covers news, exchange, regulator).
- `crypto_intel/ingest/reddit.py` — `RedditConnector`: pulls new posts/comments
  from configured subreddits via PRAW (opt-in).
- `crypto_intel/ingest/coinmarketcap.py` — `CMCConnector`: CMC news/headlines if
  `CMC_API_KEY` present; no-op with a logged notice otherwise.
- `crypto_intel/ingest/registry.py` — builds the active connector list from
  `feeds.yaml` + config/flags.
- `crypto_intel/normalize.py` — `RawItem` → `Document`; text cleaning, asset
  tagging via `assets.yaml`, dedup by `id`.
- `crypto_intel/chunking.py` — splits `Document.text` into overlapping chunks
  (word-count based, ~220 tokens / 40 overlap).
- `crypto_intel/embeddings.py` — `Embedder` protocol + backends: `OnnxEmbedder`
  (default, chromadb-bundled MiniLM) and `SentenceTransformerEmbedder` (`[st]`).
- `crypto_intel/store.py` — backend-neutral seam: `QueryFilter`, `Store` protocol,
  `VectorStore` (Chroma), `get_store()` factory (`chroma` | `pgvector`).
- `crypto_intel/pgstore.py` — `PgVectorStore`: Postgres/pgvector backend
  (Supabase-compatible); `vector(384)` column, cosine `<=>` search, `init_schema()`.
- `crypto_intel/prices.py` — CoinGecko client (timeout, one retry, on-disk cache)
  + `detect_event(asset, window)` returning a `PriceEvent`.
- `crypto_intel/retrieve.py` — `parse_question` (asset + time window + optional
  claimed %), builds the store filter, semantic query, optional BM25 rerank,
  returns top-k with provenance.
- `crypto_intel/synthesize.py` — builds the Claude prompt from `PriceEvent` +
  numbered chunks; calls the API; parses `[n]` markers into `Citation`s.
- `crypto_intel/pipeline.py` — `ingest_all`, `retrieve_context`, `ask`, `ask_db`,
  `prune_all` orchestration + the JSONL document store helpers.
- `crypto_intel/evaluate.py` — the `eval` scorecard (retrieval + citation coverage).
- `crypto_intel/forecast/` — the S9 forecasting subsystem (optional extras):
  `features.py` (pure feature engineering + supervised windowing), `models.py`
  (the `Forecaster` zoo + bundle save/load), `dataset.py` (history fetch, CSV
  loader, optional news features), `train.py` (build → split → compare → persist),
  `predict.py` (bundle → `VolForecast`). Wired to `train` / `predict` in `cli.py`.
- `crypto_intel/warehouse/` — the S10 warehouse (optional extras): `duck.py`
  (`DuckWarehouse` — load prices/docs, SQL gridding + rolling-feature and news
  aggregation, reads consumed by `train --source warehouse`) and `bq.py` (optional
  BigQuery loader). Wired to `warehouse build` / `warehouse stats` in `cli.py`.
- `crypto_intel/mlops/` — the S11 MLOps loop (optional extras): `tracking.py`
  (MLflow run logging + `pyfunc` model registry, wired to `train --track`) and
  `monitor.py` (bundle reference snapshot + Evidently drift report, wired to
  `monitor`). Serving lives in `web/app.py` (`/api/forecast`, `/monitoring`); CI in
  `.github/workflows/`.
- `crypto_intel/web/app.py` — FastAPI app for the optional GUI + S11 serving
  (`/api/forecast`, `/monitoring`) (see [GUI.md](GUI.md)).
- `crypto_intel/data/feeds.yaml`, `assets.yaml` — configuration (feed list +
  ticker/CoinGecko-id map).

## Repo layout

```
crypto-intel/
├── pyproject.toml
├── README.md
├── .env.example
├── Dockerfile · Procfile · nixpacks.toml · netlify.toml   # deployment
├── .github/workflows/                # S11: ci.yml (lint+test matrix) · train.yml (retrain)
├── crypto_intel/
│   ├── cli.py · config.py · models.py
│   ├── normalize.py · chunking.py · embeddings.py
│   ├── store.py · pgstore.py · prices.py
│   ├── retrieve.py · synthesize.py · pipeline.py · evaluate.py
│   ├── forecast/                     # S9: volatility/regime forecasting (optional extras)
│   │   ├── features.py · models.py · dataset.py · train.py · predict.py
│   ├── warehouse/                    # S10: DuckDB/BigQuery SQL feature pipeline (optional)
│   │   ├── duck.py · bq.py
│   ├── mlops/                        # S11: MLflow tracking/registry + Evidently monitoring (optional)
│   │   ├── tracking.py · monitor.py
│   ├── ingest/
│   │   ├── base.py · registry.py · rss.py · reddit.py · coinmarketcap.py
│   ├── web/
│   │   ├── app.py
│   │   └── static/  (index.html, styles.css, app.js, assets/)
│   └── data/
│       ├── feeds.yaml
│       └── assets.yaml
├── deploy/
│   ├── schema.sql · retention.sql · start.sh
├── docs/
│   ├── ARCHITECTURE.md · GUI.md · DEPLOYMENT.md
├── data/store/                    # gitignored: Chroma db + documents.jsonl + price cache
└── tests/
    ├── conftest.py + test_*.py    # normalize, chunking, prices, retrieve, synthesize, store, pgstore, web, ...
```

## Build history (phases)

The project was built in self-contained increments. All are complete.

| Phase | What it delivered |
|---|---|
| **S1** | Scaffold, typed config, pydantic models, working `stats` on an empty store. |
| **S2** | Ingestion + normalization (RSS + Reddit), asset tagging, dedup, JSONL document store; fail-soft connectors. |
| **S3** | Chunking, MiniLM embeddings, persistent Chroma; per-source counts in `stats`; idempotent re-ingest. |
| **S4** | CoinGecko client (timeout, retry, on-disk cache) + `detect_event`; `price-event` command; synthetic-series unit tests. |
| **S5** | `parse_question`, asset+time-windowed retrieval, BM25 rerank, `ask --no-synth`. |
| **S6** | Grounded synthesis with inline `[n]` citations; full `ask` with a **Sources** list. |
| **S7** | Optional CMC connector, rounded-out test suite, `ingest --all`, clean-clone README. |
| **S8** | `eval` scorecard — retrieval hit-rate + citation coverage over a committed case set. |
| **S9** | **Volatility / risk-regime forecasting** (`train` / `predict`): a `forecast/` subsystem — feature engineering, a model zoo (persistence baseline · scikit-learn · XGBoost/LightGBM · PyTorch LSTM) compared on a temporal split by skill-vs-baseline, and model-bundle persistence. Volatility only — never a price/trade call. See [ML_ROADMAP.md](ML_ROADMAP.md). |
| **S10** | **Warehouse-backed feature pipeline** (`warehouse build` / `stats`): a `warehouse/` subsystem landing prices + doc metadata in **DuckDB** with SQL feature engineering (hourly gridding, rolling window functions, news aggregation); `train --source warehouse` consumes the SQL output. Optional **BigQuery** free-tier loader (`[bq]` extra). See [ML_ROADMAP.md](ML_ROADMAP.md). |
| **S11** | **MLOps loop**: MLflow experiment tracking + model registry (`train --track`, SQLite backend), FastAPI serving (`/api/forecast`, `/monitoring`), GitHub Actions CI + retrain workflows, and Evidently drift monitoring (`monitor`). See [ML_ROADMAP.md](ML_ROADMAP.md). |
| **S12** | **RAG↔forecast stitch**: `ask` weaves the current volatility regime into the synthesis prompt as market-state context (a regime banner + `Answer.market_state`), keeping citations and the not-advice guardrail. The two subsystems now read as one product. See [ML_ROADMAP.md](ML_ROADMAP.md). |

The **ML expansion (S9–S12) is complete** — see [ML_ROADMAP.md](ML_ROADMAP.md).

Beyond the original roadmap, the project also gained: a pluggable **pgvector**
store backend, `ask-db`, rolling-window **retention** (`prune` / `pg_cron`), a
**FastAPI web GUI**, and a hosted deployment (Docker + Railway + Supabase +
Netlify). See [DEPLOYMENT.md](DEPLOYMENT.md).

## As-built notes

Where the implementation intentionally diverges from the original spec:

- **Reddit → CoinMarketCap as the default optional content connector.** Reddit's
  OAuth app setup proved impractical, so CMC was brought forward (originally S7)
  and Reddit made opt-in. In practice the keyless **RSS feeds carry the news
  load**, since CMC's content endpoint is paid-tier.
- **`onnx` is the default embedding backend, not sentence-transformers.** `torch`
  won't `pip install` under this repo's long Windows path (`WinError 206`), and
  enabling system-wide long paths is out of scope. chromadb's bundled ONNX MiniLM
  is the same model with zero extra dependencies, so `pip install -e .` works
  out-of-the-box; sentence-transformers moved to the optional `[st]` extra.
- **The web GUI does retrieval only, no synthesis.** The hosted deployment
  deliberately omits the Claude call — the full cited answer stays in the CLI. See
  [GUI.md](GUI.md) and [DEPLOYMENT.md](DEPLOYMENT.md).

## Guardrail

This tool explains market moves from public sources with citations. It must **not**
emit buy/sell/hold recommendations, price targets, or personalized financial
advice. Synthesis is kept descriptive ("reports attribute the move to X [1], Y
[2]") and `ask` output includes a "not investment advice" line.

The **ML layer (S9–S12) holds the same wall**: it forecasts *volatility / risk
regime only*, never price direction; the S12 regime woven into `ask` is background
market-state context (a volatility estimate, not a source and not a prediction —
never cited, never turned into a call), and every `predict` / `ask` output keeps
the not-investment-advice line.
