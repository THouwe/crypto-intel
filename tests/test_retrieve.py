"""Tests for question parsing, BM25 rerank ordering, where-filter, retrieval."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from crypto_intel.config import Settings
from crypto_intel.retrieve import (
    ParsedQuestion,
    RetrievedChunk,
    build_filter,
    parse_claimed_pct,
    parse_question,
    parse_window_hours,
    rerank_candidates,
    retrieve,
)
from crypto_intel.store import QueryFilter

BASE = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)


# --- question parsing ------------------------------------------------------- #

def test_parse_asset_from_alias():
    p = parse_question("Why did Ethereum drop today?", Settings())
    assert p.asset == "ETH"


def test_parse_asset_ticker_and_primary_when_multiple():
    # BTC appears before ETH -> BTC is primary.
    p = parse_question("Is BTC dragging ETH down?", Settings())
    assert p.asset == "BTC"


def test_parse_asset_override_wins():
    p = parse_question("Why did Ethereum move?", Settings(), asset_override="sol")
    assert p.asset == "SOL"


def test_parse_no_asset():
    p = parse_question("Why is the market down?", Settings())
    assert p.asset is None


def test_parse_window_phrases():
    assert parse_window_hours("why did eth drop today") == 24
    assert parse_window_hours("what happened yesterday") == 48
    assert parse_window_hours("why is sol up this week") == 168
    assert parse_window_hours("move over the last 12 hours") == 12
    assert parse_window_hours("over the past 3 days") == 72
    assert parse_window_hours("in the last 2 weeks") == 336
    assert parse_window_hours("no time phrase here") is None


def test_parse_hours_override_beats_phrase():
    p = parse_question("why did eth drop today", Settings(), hours_override=6)
    assert p.hours == 6 and p.window_explicit is True


def test_parse_default_window_when_absent():
    p = parse_question("why did eth move", Settings(default_lookback_hours=48))
    assert p.hours == 48 and p.window_explicit is False


def test_parse_claimed_pct():
    assert parse_claimed_pct("Why did ETH drop 6% today?") == 6.0
    assert parse_claimed_pct("moved 12.5% up") == 12.5
    assert parse_claimed_pct("no percent here") is None


# --- BM25 rerank ------------------------------------------------------------ #

def _chunk(doc_id: str, text: str, semantic: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{doc_id}:0",
        doc_id=doc_id,
        text=text,
        source="news",
        source_name="Test",
        url=f"https://x/{doc_id}",
        published_at=BASE,
        assets=["ETH"],
        semantic_score=semantic,
    )


def test_bm25_rerank_promotes_lexical_match():
    query = "ethereum etf outflows sec"
    # 'a' has the weaker semantic score but is a strong lexical match; with the
    # 0.5/0.5 blend it should outrank the semantically-close but off-topic 'b'.
    # Distractor docs give BM25 a real corpus so its IDF can discriminate (with
    # only 2 docs, a term in one of them has IDF ~ 0).
    a = _chunk("a", "Ethereum ETF outflows accelerated after the SEC filing.", semantic=0.60)
    b = _chunk("b", "A general market commentary about prices and sentiment.", semantic=0.66)
    fillers = [
        _chunk("f1", "Bitcoin miners discuss hardware and energy costs.", semantic=0.30),
        _chunk("f2", "Solana network throughput and validator uptime report.", semantic=0.28),
        _chunk("f3", "Stablecoin liquidity across decentralized exchanges.", semantic=0.25),
    ]
    ranked = rerank_candidates(query, [b, *fillers, a])
    assert ranked[0].doc_id == "a"
    assert ranked[0].bm25_score > ranked[1].bm25_score


def test_bm25_rerank_pure_semantic_when_alpha_one():
    query = "unrelated tokens zzz"
    a = _chunk("a", "no query words at all here", semantic=0.9)
    b = _chunk("b", "also nothing relevant", semantic=0.2)
    ranked = rerank_candidates(query, [b, a], alpha=1.0)
    assert [c.doc_id for c in ranked] == ["a", "b"]


def test_rerank_empty():
    assert rerank_candidates("q", []) == []


# --- build_filter (backend-neutral) ----------------------------------------- #

def test_build_filter_captures_asset_window_sources():
    start, end = BASE, BASE + timedelta(hours=24)
    f = build_filter("eth", start, end, sources={"news", "regulator"})
    assert isinstance(f, QueryFilter)
    assert f.asset == "ETH"  # uppercased
    assert f.window_start == start and f.window_end == end
    assert f.sources == {"news", "regulator"}


def test_build_filter_empty():
    f = build_filter(None, None, None, None)
    assert f.asset is None and f.window_start is None and f.sources is None


# --- retrieve() with fakes -------------------------------------------------- #

class _FakeEmbedder:
    model_name = "fake"
    dim = 3

    def encode(self, texts):
        return [[0.0, 0.0, 0.0] for _ in texts]


class _FakeStore:
    """Returns canned hits; records the `QueryFilter` it was queried with."""

    def __init__(self, hits):
        self._hits = hits
        self.last_filter = None

    def count(self):
        return len(self._hits)

    def query(self, embedding, filter=None, n_results=10):
        self.last_filter = filter
        return self._hits[:n_results]


def _hit(doc_id, text, distance):
    return {
        "id": f"{doc_id}:0",
        "text": text,
        "distance": distance,
        "metadata": {
            "doc_id": doc_id,
            "source": "news",
            "source_name": "Test",
            "url": f"https://x/{doc_id}",
            "published_at": int(BASE.timestamp()),
            "assets": "ETH",
        },
    }


def test_retrieve_dedups_by_doc_and_limits_k():
    hits = [
        _hit("d1", "ethereum etf outflows sec filing", 0.1),
        _hit("d1", "ethereum etf outflows continued", 0.15),  # same doc -> deduped
        _hit("d2", "solana unrelated content", 0.2),
        _hit("d3", "ethereum staking news", 0.25),
    ]
    store = _FakeStore(hits)
    out = retrieve(
        "ethereum etf outflows",
        asset="ETH",
        window_start=BASE - timedelta(hours=1),
        window_end=BASE + timedelta(hours=1),
        k=2,
        settings=Settings(),
        store=store,
        embedder=_FakeEmbedder(),
    )
    assert len(out) == 2
    doc_ids = [c.doc_id for c in out]
    assert len(set(doc_ids)) == 2  # deduped
    assert "d1" in doc_ids


def test_retrieve_builds_asset_and_time_filter():
    store = _FakeStore([_hit("d1", "eth news", 0.1)])
    end = BASE + timedelta(hours=24)
    retrieve(
        "why did eth drop",
        asset="ETH",
        window_start=BASE,
        window_end=end,
        k=3,
        settings=Settings(),
        store=store,
        embedder=_FakeEmbedder(),
    )
    assert store.last_filter is not None
    assert store.last_filter.asset == "ETH"
    assert store.last_filter.window_start == BASE
    assert store.last_filter.window_end == end


def test_retrieve_empty_store_returns_empty():
    out = retrieve(
        "q",
        asset="ETH",
        window_start=BASE,
        window_end=BASE + timedelta(hours=24),
        k=3,
        settings=Settings(),
        store=_FakeStore([]),
        embedder=_FakeEmbedder(),
    )
    assert out == []
