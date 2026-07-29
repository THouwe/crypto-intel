"""Tests for word-count chunking."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from crypto_intel.chunking import chunk_document, chunk_text
from crypto_intel.models import Document, SourceType


def test_empty_text_yields_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_short_text_is_single_chunk():
    out = chunk_text("alpha beta gamma", chunk_size=220, overlap=40)
    assert out == ["alpha beta gamma"]


def test_long_text_splits_with_overlap():
    words = [f"w{i}" for i in range(500)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=220, overlap=40)

    # step = 180 -> windows starting at 0, 180, 360 => 3 chunks
    assert len(chunks) == 3
    assert chunks[0].split()[0] == "w0"
    assert chunks[0].split()[-1] == "w219"
    # overlap: first 40 words of chunk 2 equal last 40 of chunk 1
    assert chunks[1].split()[:40] == chunks[0].split()[-40:]
    # last chunk reaches the end
    assert chunks[-1].split()[-1] == "w499"


def test_every_word_is_covered():
    words = [f"w{i}" for i in range(1000)]
    chunks = chunk_text(" ".join(words), chunk_size=100, overlap=20)
    covered = set()
    for c in chunks:
        covered.update(c.split())
    assert covered == set(words)


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=10, overlap=10)


def _doc(text: str) -> Document:
    return Document(
        id="abc123",
        source=SourceType.news,
        source_name="CoinDesk",
        url="https://x/1",
        text=text,
        published_at=datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc),
        assets=["ETH", "BTC"],
    )


def test_chunk_document_carries_metadata_and_stable_ids():
    words = " ".join(f"w{i}" for i in range(300))
    chunks = chunk_document(_doc(words), chunk_size=220, overlap=40)
    assert len(chunks) == 2
    assert [c.id for c in chunks] == ["abc123:0", "abc123:1"]
    for i, c in enumerate(chunks):
        assert c.doc_id == "abc123"
        assert c.chunk_index == i
        assert c.source is SourceType.news
        assert c.assets == ["ETH", "BTC"]
        assert c.url == "https://x/1"
