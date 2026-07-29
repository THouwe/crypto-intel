# Deployment

How to run the hosted stack: the **static web GUI on Netlify**, the **FastAPI API
on Railway**, and the **shared vector store on Supabase (Postgres/pgvector)**. The
CLI needs none of this — it runs against a local Chroma store out of the box (see
the [README](../README.md)). This guide is for standing up the web deployment.

> **The deployed GUI does retrieval only — no LLM synthesis.** It answers "what
> happened to `[asset]`" with a confirmed price move + ranked source evidence. It
> never calls Claude, so **the hosted API needs no `ANTHROPIC_API_KEY`**. The full
> cited `ask` summary stays in the CLI. See [GUI.md](GUI.md).

## Topology

```
        Browser
           │
           ▼
   ┌───────────────┐     /api/*  (Netlify proxy rewrite, status=200)
   │    Netlify     │ ─────────────────────────────────────┐
   │  static site   │                                       ▼
   │  (index.html,  │                              ┌──────────────────┐
   │   app.js, css) │                              │     Railway      │
   └───────────────┘                              │  FastAPI (uvicorn)│
     no synthesis                                 │  /api/price-event │
                                                  │  /api/evidence    │
                                   ┌── CoinGecko ─┤  /api/assets      │
                                   │  (prices)    └────────┬─────────┘
                                   ▼                       │ pgvector <=>
                            api.coingecko.com     ┌────────▼─────────┐
                                                  │    Supabase       │
   Railway cron service ─── ingest ─────────────► │  Postgres +       │
   (3-hourly RSS refresh)                         │  pgvector(384)    │
                                                  └──────────────────┘
```

Two Railway services share **one image/repo**, differing only in the command:

- **Query API** — serves `/api/*` (and can serve the static page too). Binds `$PORT`.
- **Cron ingest** — a scheduled `crypto-intel ingest` that refreshes the shared DB.

## Deployment files in this repo

| File | Role |
|---|---|
| [`netlify.toml`](../netlify.toml) | Netlify: publish `crypto_intel/web/static/`, proxy `/api/*` → Railway, SPA fallback. |
| [`Procfile`](../Procfile) | Railway (Nixpacks): `web: uvicorn crypto_intel.web.app:app --host 0.0.0.0 --port $PORT`. |
| [`nixpacks.toml`](../nixpacks.toml) | Railway (Nixpacks): install the `[web,pg]` extras (default `pip install .` skips them). |
| [`Dockerfile`](../Dockerfile) | Alternative to Nixpacks: self-contained image with the ONNX model baked in. |
| [`deploy/schema.sql`](../deploy/schema.sql) | Supabase: `vector` extension + `chunks` table + indexes. |
| [`deploy/retention.sql`](../deploy/retention.sql) | Supabase: `pg_cron` rolling-window DELETE (DB-native retention). |
| [`deploy/start.sh`](../deploy/start.sh) | Service entrypoint: one-time initial populate (`--if-empty`) then exec the server. |

## 1 — Provision the Supabase store

Run [`deploy/schema.sql`](../deploy/schema.sql) once in the Supabase SQL Editor to
create the `vector` extension, the `chunks` table (`vector(384)`), and indexes.
(The app also self-creates these on first use, but running it explicitly is
cleaner for a fresh project.)

Grab the Postgres connection string (Supabase → Project Settings → Database) — it
becomes `DATABASE_URL`.

## 2 — Deploy the API to Railway

**Option A — Nixpacks (uses `Procfile` + `nixpacks.toml`).** Point Railway at the
repo. Nixpacks detects Python, runs `pip install -e '.[web,pg]'` (from
`nixpacks.toml`), and starts the `web:` process from the `Procfile`:

```
web: uvicorn crypto_intel.web.app:app --host 0.0.0.0 --port $PORT
```

> The built-in `crypto-intel serve` binds `127.0.0.1:8000` and is for local use —
> **do not** use it on Railway. The Procfile calls `uvicorn` directly so it binds
> `0.0.0.0` and honors Railway's injected `$PORT`.

**Option B — Docker (uses `Dockerfile`).** The image bakes the ONNX MiniLM model at
build time (no ~79 MB download / cold-start on the first request) and installs no
torch. Its default `CMD` runs [`deploy/start.sh`](../deploy/start.sh) (one-time
populate, then uvicorn on `$PORT`). Set Railway's builder to Dockerfile.

**Required env on the API service:**

| Variable | Value | Why |
|---|---|---|
| `STORE_BACKEND` | `pgvector` | Selects the Postgres backend. |
| `DATABASE_URL` | your Supabase DSN | Where to read/write chunks. |

`ANTHROPIC_API_KEY` is **not** needed (no synthesis). `EMBED_BACKEND=onnx` is the
default and needs nothing extra — chromadb's bundled ONNX MiniLM embeds the query.

## 3 — Recurring ingest (cron service)

Add a second Railway service from the same repo/image whose command is the
periodic refresh (it writes to the same Supabase DB):

```bash
crypto-intel ingest --sources news,exchange,regulator --lookback-hours 3
```

Schedule it every ~3 hours. It needs the **same** `STORE_BACKEND=pgvector` +
`DATABASE_URL`. The initial 7-day populate is handled once by
[`deploy/start.sh`](../deploy/start.sh) on the API service (`--if-empty` makes it a
no-op after the store is populated).

> **Wiring gotcha (important).** There are two independent switches:
> - `STORE_BACKEND` drives the **CLI/ingest** path via `get_store()`.
> - `DATABASE_URL` drives the **web read** path (`get_retriever` → `ask_db`, which
>   forces pgvector regardless of `STORE_BACKEND`).
>
> So on **any box that ingests**, set **both** `STORE_BACKEND=pgvector` *and*
> `DATABASE_URL`. If ingest runs with the default `STORE_BACKEND=chroma` but
> `DATABASE_URL` set, it silently writes to a local Chroma dir while the web app
> reads an **empty** pgvector — the most likely misconfiguration. Setting
> `STORE_BACKEND=pgvector` without `DATABASE_URL` raises a clear `RuntimeError`.

## 4 — Deploy the frontend to Netlify

The frontend ([`crypto_intel/web/static/`](../crypto_intel/web/static/)) is a
static page that fetches `/api/*` with **relative** paths. [`netlify.toml`](../netlify.toml)
publishes it and proxies the API to Railway so those paths resolve with **no CORS
and no code change**:

```toml
[build]
  publish = "crypto_intel/web/static"
  command = ""

[[redirects]]
  from = "/api/*"
  to = "https://YOUR-RAILWAY-APP.up.railway.app/api/:splat"
  status = 200        # transparent rewrite — browser stays same-origin
  force = true

[[redirects]]
  from = "/*"
  to = "/index.html"  # SPA fallback (after the /api rule, so it never shadows it)
  status = 200
```

**Edit before deploying:** replace `YOUR-RAILWAY-APP.up.railway.app` with your
Railway API hostname (no trailing slash; keep `/api/:splat`). Netlify does not
interpolate env vars into `netlify.toml` redirects, so the backend URL is
hardcoded here. If the Netlify UI has a build command set, clear it — this file's
empty `command` means "just publish".

**Alternative to the proxy:** inject an `API_BASE` into `app.js` and enable CORS on
the FastAPI side. The proxy is preferred — one origin, no CORS.

## 5 — Retention (rolling window)

Keep only the recent window (matches the 168h initial populate). Use **one** of:

- **App-level:** `crypto-intel prune --keep-days 7` (deletes old chunks + JSONL
  docs; works for both backends). Run it after ingest or on its own schedule.
- **DB-native:** schedule [`deploy/retention.sql`](../deploy/retention.sql) with
  `pg_cron` on Supabase (hourly `DELETE`).

Don't run both.

## All-in-one alternative (no Netlify)

Since the FastAPI app already serves its own static files at `/` and `/static`, a
single Railway service can host **both** the page and the API — one origin, no
proxy, no CORS. Deploy just the API service (steps 1–3) and skip Netlify; visit the
Railway URL directly. Use the split (Netlify + Railway) only if you specifically
want the frontend on Netlify's CDN.

## Docker quick reference

```bash
docker build -t crypto-intel .

# Query API (default CMD): initial populate (once) then uvicorn
docker run -p 8000:8000 \
  -e STORE_BACKEND=pgvector -e DATABASE_URL="postgresql://…supabase…:5432/postgres" \
  crypto-intel

# Recurring ingest (override the command — the 3-hourly cron service)
docker run \
  -e STORE_BACKEND=pgvector -e DATABASE_URL="postgresql://…" \
  crypto-intel crypto-intel ingest --sources news,exchange,regulator --lookback-hours 3
```

One image serves both services; only the command differs. The API binds Railway's
`$PORT` (defaults to 8000).
