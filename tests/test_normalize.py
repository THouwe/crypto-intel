"""Tests for text cleaning, asset tagging, and dedup-id stability."""

from __future__ import annotations

from datetime import datetime, timezone

from crypto_intel.ingest.base import RawItem
from crypto_intel.models import SourceType
from crypto_intel.normalize import (
    clean_text,
    load_assets,
    make_doc_id,
    normalize,
    tag_assets,
)

ASSETS = {
    "ETH": {"coingecko_id": "ethereum", "aliases": ["eth", "ether", "ethereum"]},
    "BTC": {"coingecko_id": "bitcoin", "aliases": ["btc", "bitcoin", "xbt"]},
    "SOL": {"coingecko_id": "solana", "aliases": ["sol", "solana"]},
}


def test_clean_text_strips_html_and_collapses_whitespace():
    raw = "<p>Ether &amp;   friends</p>\n\n  moved   sharply</p>"
    assert clean_text(raw) == "Ether & friends moved sharply"


def test_clean_text_handles_none_and_empty():
    assert clean_text(None) == ""
    assert clean_text("") == ""


def test_tag_assets_matches_aliases_case_insensitively():
    text = "Ethereum slid while Bitcoin held; analysts eye BTC support."
    assert tag_assets(text, ASSETS) == ["BTC", "ETH"]


def test_tag_assets_uses_word_boundaries():
    # "sol" appears only inside "resolve" / "solid" — must NOT tag SOL.
    text = "The team will resolve the issue with a solid patch."
    assert tag_assets(text, ASSETS) == []


def test_tag_assets_ticker_symbol_alone_matches():
    assert tag_assets("SOL rallied today", ASSETS) == ["SOL"]


def test_tag_assets_empty_text():
    assert tag_assets("", ASSETS) == []


def test_load_assets_from_shipped_yaml():
    assets = load_assets()
    assert "ETH" in assets and assets["ETH"]["coingecko_id"] == "ethereum"


def test_make_doc_id_is_stable_and_deterministic():
    ts = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
    a = make_doc_id("CoinDesk", "https://x/1", ts)
    b = make_doc_id("CoinDesk", "https://x/1", ts)
    assert a == b and len(a) == 40


def test_make_doc_id_differs_on_any_field():
    ts = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
    base = make_doc_id("CoinDesk", "https://x/1", ts)
    assert base != make_doc_id("Decrypt", "https://x/1", ts)
    assert base != make_doc_id("CoinDesk", "https://x/2", ts)
    assert base != make_doc_id(
        "CoinDesk", "https://x/1", ts.replace(hour=13)
    )


def test_make_doc_id_normalizes_timezone():
    # Same instant expressed in different zones -> same id.
    utc = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
    from datetime import timedelta

    other = datetime(2026, 7, 20, 14, 0, tzinfo=timezone(timedelta(hours=2)))
    assert make_doc_id("SEC", "u", utc) == make_doc_id("SEC", "u", other)


def test_normalize_produces_tagged_document():
    item = RawItem(
        source=SourceType.news,
        source_name="CoinDesk",
        url="https://coindesk.com/eth-drop",
        text="<b>Ethereum</b> dropped 6% amid ETF outflows.",
        published_at=datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc),
        title="ETH slides",
    )
    doc = normalize(item, ASSETS)
    assert doc.source is SourceType.news
    assert doc.text == "Ethereum dropped 6% amid ETF outflows."
    assert doc.assets == ["ETH"]
    assert doc.published_at.tzinfo is timezone.utc
    assert len(doc.id) == 40


def test_normalize_makes_naive_datetime_utc():
    item = RawItem(
        source=SourceType.news,
        source_name="X",
        url="u",
        text="btc",
        published_at=datetime(2026, 7, 20, 12, 0),  # naive
    )
    doc = normalize(item, ASSETS)
    assert doc.published_at.tzinfo is not None
