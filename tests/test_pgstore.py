"""Tests for the pgvector store's pure SQL/mapping helpers (no DB needed).

A live-DB round trip is available but skipped unless CRYPTO_INTEL_TEST_DATABASE_URL
is set (and the `pg` extra is installed).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from crypto_intel.pgstore import EMBED_DIM, PgVectorStore, filter_sql, row_to_hit
from crypto_intel.store import QueryFilter

BASE = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)


# --- filter_sql (pure) ------------------------------------------------------ #

def test_filter_sql_all_conditions():
    end = BASE + timedelta(hours=24)
    where, params = filter_sql(
        QueryFilter(asset="eth", window_start=BASE, window_end=end, sources={"news", "regulator"})
    )
    assert where.startswith(" WHERE ")
    assert "%(asset)s = ANY(assets)" in where
    assert "published_at >= %(ws)s" in where
    assert "published_at <= %(we)s" in where
    assert "source = ANY(%(sources)s)" in where
    assert params["asset"] == "ETH"  # uppercased
    assert params["ws"] == BASE and params["we"] == end
    assert params["sources"] == ["news", "regulator"]  # sorted


def test_filter_sql_asset_only():
    where, params = filter_sql(QueryFilter(asset="BTC"))
    assert where == " WHERE %(asset)s = ANY(assets)"
    assert params == {"asset": "BTC"}


def test_filter_sql_empty():
    assert filter_sql(QueryFilter()) == ("", {})
    assert filter_sql(None) == ("", {})


# --- row_to_hit (pure) ------------------------------------------------------ #

def test_row_to_hit_shapes_like_chroma():
    row = {
        "id": "d1:0",
        "text": "Ethereum fell on ETF outflows",
        "distance": 0.234,
        "doc_id": "d1",
        "source": "news",
        "source_name": "CoinDesk",
        "url": "https://x/1",
        "published_at": int(BASE.timestamp()),
        "assets": ["ETH", "BTC"],
    }
    hit = row_to_hit(row)
    assert hit["id"] == "d1:0"
    assert hit["text"].startswith("Ethereum")
    assert hit["distance"] == pytest.approx(0.234)
    md = hit["metadata"]
    assert md["doc_id"] == "d1"
    assert md["published_at"] == int(BASE.timestamp())
    assert md["assets"] == "ETH,BTC"  # CSV, matching the Chroma hit shape


def test_row_to_hit_empty_assets():
    row = {
        "id": "d2:0", "text": "t", "distance": 1.0, "doc_id": "d2",
        "source": "news", "source_name": "X", "url": "u",
        "published_at": int(BASE.timestamp()), "assets": None,
    }
    assert row_to_hit(row)["metadata"]["assets"] == ""


def test_embed_dim_matches_minilm():
    assert EMBED_DIM == 384


def test_pgstore_query_without_dsn_raises():
    from crypto_intel.config import Settings

    store = PgVectorStore(Settings(store_backend="pgvector", database_url=None))
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        store.count()


# --- live-DB integration (opt-in) ------------------------------------------- #

_DB_URL = os.environ.get("CRYPTO_INTEL_TEST_DATABASE_URL")


@pytest.mark.skipif(not _DB_URL, reason="set CRYPTO_INTEL_TEST_DATABASE_URL to run")
def test_pgstore_roundtrip_live():
    pytest.importorskip("psycopg")
    pytest.importorskip("pgvector")
    from crypto_intel.config import Settings
    from crypto_intel.models import Chunk, SourceType

    store = PgVectorStore(Settings(store_backend="pgvector", database_url=_DB_URL))
    store.init_schema()

    chunk = Chunk(
        id="itest:0",
        doc_id="itest",
        chunk_index=0,
        text="Ethereum ETF outflows accelerated after the SEC filing.",
        source=SourceType.news,
        source_name="CoinDesk",
        url="https://example.com/itest",
        published_at=BASE,
        assets=["ETH"],
    )
    emb = [0.01] * EMBED_DIM
    store.upsert_chunks([chunk], [emb])
    store.upsert_chunks([chunk], [emb])  # idempotent (ON CONFLICT DO NOTHING)

    hits = store.query(emb, QueryFilter(asset="ETH", window_start=BASE - timedelta(hours=1), window_end=BASE + timedelta(hours=1)), n_results=5)
    assert any(h["id"] == "itest:0" for h in hits)

    # cleanup
    store.prune(BASE + timedelta(days=3650))
