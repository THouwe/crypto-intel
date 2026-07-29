"""FastAPI app exposing the `price-event` flow to a small web GUI.

Scope: **price-event only** — confirm a move against real CoinGecko data. It is a
thin HTTP layer over :func:`crypto_intel.prices.detect_event` (which already
returns a JSON-serializable :class:`PriceEvent`), plus a static single-page UI.

Not part of the core CLI demo — install with ``pip install -e .[web]`` and run
``crypto-intel serve`` (or ``uvicorn crypto_intel.web.app:app``).
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import get_settings
from ..normalize import load_assets
from ..prices import CoinGeckoClient, PriceError, detect_event

logger = logging.getLogger(__name__)

# --hours bounds. Cap tracks CoinGecko granularity: <=24h is 5-minutely, 1-90d is
# hourly, >90d drops to daily. 720h (30 days) keeps the steepest-move window
# meaningful while allowing a month-long view.
HOURS_MIN = 1
HOURS_MAX = 720
DEFAULT_ASSET = "BTC"
DEFAULT_HOURS = 24

_STATIC = Path(__file__).parent / "static"

app = FastAPI(
    title="Crypto Market Intelligence — price-event",
    version="0.1.0",
    description="Confirm a crypto price move against real market data.",
)

# Serve the split static assets (styles.css, app.js, assets/logos) at /static.
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


_SNIPPET_CHARS = 260


def get_price_client() -> CoinGeckoClient:
    """Provide the CoinGecko client (overridable in tests via dependency_overrides)."""
    return CoinGeckoClient(get_settings())


def get_retriever():
    """Provide the evidence retriever (overridable in tests via dependency_overrides).

    Returns a callable ``(question, asset, hours) -> RetrievalContext``. When
    ``DATABASE_URL`` is set it retrieves from the shared **pgvector DB** (the
    ``ask-db`` path — the deployed API + Supabase); otherwise it falls back to the
    configured local store so the GUI still runs in local dev. Price detection is
    skipped (``/api/price-event`` covers prices), and the retrieval window
    [now - hours, now] matches what the CLI ``ask`` would use.
    """

    def _retrieve(question: str, asset: str, hours: int):
        settings = get_settings()
        if settings.database_url:
            from ..pipeline import ask_db  # deferred (optional pg deps)

            return ask_db(
                question, settings=settings, asset_override=asset, hours_override=hours
            )
        from ..pipeline import retrieve_context  # deferred (heavy import chain)

        return retrieve_context(
            question,
            settings=settings,
            asset_override=asset,
            hours_override=hours,
            detect_prices=False,
        )

    return _retrieve


def _chunk_dict(chunk) -> dict:
    text = " ".join(chunk.text.split())
    snippet = text[:_SNIPPET_CHARS] + ("…" if len(text) > _SNIPPET_CHARS else "")
    return {
        "source_name": chunk.source_name,
        "url": chunk.url,
        "published_at": chunk.published_at.isoformat(),
        "snippet": snippet,
        "score": round(chunk.score, 3),
        "semantic_score": round(chunk.semantic_score, 3),
        "bm25_score": round(chunk.bm25_score, 2),
        "assets": chunk.assets,
    }


def _granularity(hours: int) -> str:
    """Human-readable CoinGecko sampling granularity for a window length."""
    if hours <= 24:
        return "5-minutely"
    if hours <= 90 * 24:
        return "hourly"
    return "daily"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/api/assets")
def api_assets() -> dict:
    """Tickers available for the dropdown (from assets.yaml)."""
    assets = load_assets(get_settings())
    return {"assets": sorted(assets.keys()), "default": DEFAULT_ASSET}


@app.get("/api/price-event")
def api_price_event(
    asset: str = Query(DEFAULT_ASSET, min_length=1, max_length=16),
    hours: int = Query(DEFAULT_HOURS, ge=HOURS_MIN, le=HOURS_MAX),
    client: CoinGeckoClient = Depends(get_price_client),
) -> JSONResponse:
    """Detect and return the price event for ``asset`` over ``hours``.

    ``hours`` is bounded to [1, 720] by FastAPI (422 outside). Unknown assets or
    sparse data raise :class:`PriceError` → 400; upstream failures → 502.
    """
    settings = get_settings()
    try:
        event = detect_event(asset, hours, settings=settings, client=client)
    except PriceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # network / unexpected upstream failure
        logger.warning("price-event failed for %s/%sh: %s", asset, hours, exc)
        raise HTTPException(status_code=502, detail="Upstream price data unavailable.")

    payload = event.model_dump(mode="json")
    payload["hours"] = hours
    payload["granularity"] = _granularity(hours)
    return JSONResponse(payload)


@app.get("/api/evidence")
def api_evidence(
    question: str = Query(..., min_length=1, max_length=300),
    asset: str = Query(DEFAULT_ASSET, min_length=1, max_length=16),
    hours: int = Query(DEFAULT_HOURS, ge=HOURS_MIN, le=HOURS_MAX),
    retriever=Depends(get_retriever),
) -> JSONResponse:
    """Retrieve time-and-asset-windowed evidence (the ``ask --no-synth`` path).

    Returns ranked chunks with their relevance (accuracy) scores. No LLM call.
    """
    try:
        ctx = retriever(question, asset, hours)
    except Exception as exc:  # store/embedding/unexpected failure
        logger.warning("evidence retrieval failed for %s/%sh: %s", asset, hours, exc)
        raise HTTPException(status_code=502, detail="Evidence retrieval failed.")

    return JSONResponse(
        {
            "asset": asset.upper(),
            "hours": hours,
            "question": question,
            "chunks": [_chunk_dict(c) for c in ctx.chunks],
            "notes": ctx.notes,
        }
    )
