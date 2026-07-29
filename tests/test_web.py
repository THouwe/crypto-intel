"""Tests for the FastAPI price-event GUI (offline, mocked CoinGecko)."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

pytest.importorskip("fastapi")  # web extra is optional
from fastapi.testclient import TestClient  # noqa: E402

from crypto_intel.config import Settings  # noqa: E402
from crypto_intel.pipeline import RetrievalContext  # noqa: E402
from crypto_intel.prices import CoinGeckoClient  # noqa: E402
from crypto_intel.retrieve import ParsedQuestion, RetrievedChunk  # noqa: E402
from crypto_intel.web.app import app, get_price_client, get_retriever  # noqa: E402

BASE = datetime(2026, 7, 20, 0, 0, tzinfo=timezone.utc)


def _price_payload(prices):
    ms0 = int(BASE.timestamp() * 1000)
    return {"prices": [[ms0 + i * 3600_000, p] for i, p in enumerate(prices)]}


@pytest.fixture
def client(tmp_path):
    """TestClient with a CoinGecko client that serves a canned down-move series."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_price_payload([100, 110, 104, 94]))

    def override():
        return CoinGeckoClient(
            Settings(price_cache_path=tmp_path / "pc"),
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

    def _chunk(doc_id, source_name, url, score, sem, bm):
        return RetrievedChunk(
            chunk_id=f"{doc_id}:0",
            doc_id=doc_id,
            text="ETH slid on ETF outflows after the SEC filing, analysts say.",
            source="news",
            source_name=source_name,
            url=url,
            published_at=BASE,
            assets=["ETH"],
            semantic_score=sem,
            bm25_score=bm,
            score=score,
        )

    def retriever_override():
        def _retrieve(question, asset, hours):
            parsed = ParsedQuestion(question=question, asset=asset.upper(), hours=hours)
            chunks = [
                _chunk("a", "BeInCrypto", "https://beincrypto.com/eth", 0.75, 0.49, 3.9),
                _chunk("b", "CryptoPotato", "https://cryptopotato.com/eth", 0.49, 0.40, 2.3),
            ]
            return RetrievalContext(parsed=parsed, event=None, chunks=chunks, notes=[])

        return _retrieve

    app.dependency_overrides[get_price_client] = override
    app.dependency_overrides[get_retriever] = retriever_override
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def empty_client():
    """TestClient whose retriever returns no evidence (with a note)."""

    def retriever_override():
        def _retrieve(question, asset, hours):
            parsed = ParsedQuestion(question=question, asset=asset.upper(), hours=hours)
            return RetrievalContext(parsed=parsed, event=None, chunks=[], notes=["No evidence in the window."])

        return _retrieve

    app.dependency_overrides[get_retriever] = retriever_override
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "What happened to" in r.text


def test_assets_endpoint(client):
    r = client.get("/api/assets")
    assert r.status_code == 200
    body = r.json()
    assert body["default"] == "BTC"
    assert "BTC" in body["assets"] and "ETH" in body["assets"]


def test_price_event_ok(client):
    r = client.get("/api/price-event", params={"asset": "ETH", "hours": 24})
    assert r.status_code == 200
    body = r.json()
    assert body["asset"] == "ETH"
    assert body["coingecko_id"] == "ethereum"
    assert body["direction"] == "down"  # 100 -> 94
    assert body["hours"] == 24
    assert body["granularity"] == "5-minutely"


def test_price_event_defaults_to_btc_24h(client):
    r = client.get("/api/price-event")  # no params
    assert r.status_code == 200
    body = r.json()
    assert body["asset"] == "BTC" and body["hours"] == 24


def test_granularity_hourly_for_week(client):
    r = client.get("/api/price-event", params={"asset": "ETH", "hours": 168})
    assert r.json()["granularity"] == "hourly"


def test_hours_above_cap_is_422(client):
    r = client.get("/api/price-event", params={"asset": "ETH", "hours": 1000})
    assert r.status_code == 422


def test_hours_below_min_is_422(client):
    r = client.get("/api/price-event", params={"asset": "ETH", "hours": 0})
    assert r.status_code == 422


def test_unknown_asset_is_400(client):
    r = client.get("/api/price-event", params={"asset": "NOTACOIN", "hours": 24})
    assert r.status_code == 400
    assert "Unknown asset" in r.json()["detail"]


# --- /api/evidence ---------------------------------------------------------- #

def test_evidence_returns_ranked_chunks_with_scores(client):
    r = client.get(
        "/api/evidence",
        params={"question": "What happened to ETH within the last 1 day?", "asset": "ETH", "hours": 24},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["asset"] == "ETH" and body["hours"] == 24
    assert len(body["chunks"]) == 2
    top = body["chunks"][0]
    assert top["source_name"] == "BeInCrypto"
    assert top["score"] == 0.75  # accuracy score surfaced
    assert "semantic_score" in top and "bm25_score" in top
    assert top["url"].startswith("https://")
    assert top["snippet"]


def test_evidence_empty_returns_notes(empty_client):
    r = empty_client.get(
        "/api/evidence",
        params={"question": "What happened to XYZ within the last 1 hour?", "asset": "BTC", "hours": 1},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["chunks"] == []
    assert body["notes"] and "No evidence" in body["notes"][0]


def test_evidence_requires_question(client):
    # `question` is required.
    r = client.get("/api/evidence", params={"asset": "ETH", "hours": 24})
    assert r.status_code == 422


def test_evidence_hours_cap(client):
    r = client.get(
        "/api/evidence",
        params={"question": "q", "asset": "ETH", "hours": 1000},
    )
    assert r.status_code == 422
