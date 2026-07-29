"""Tests for retention pruning (JSONL docs + Chroma store + prune_all)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from crypto_intel.models import Chunk, Document, SourceType
from crypto_intel.pipeline import (
    append_documents,
    iter_documents,
    prune_all,
    prune_documents,
)
from crypto_intel.store import VectorStore

NOW = datetime.now(timezone.utc)


def _doc(doc_id: str, days_old: float) -> Document:
    return Document(
        id=doc_id,
        source=SourceType.news,
        source_name="CoinDesk",
        url=f"https://x/{doc_id}",
        text="Ethereum moved.",
        published_at=NOW - timedelta(days=days_old),
        assets=["ETH"],
    )


def _chunk(doc_id: str, days_old: float) -> Chunk:
    return Chunk(
        id=f"{doc_id}:0",
        doc_id=doc_id,
        chunk_index=0,
        text="Ethereum moved.",
        source=SourceType.news,
        source_name="CoinDesk",
        url=f"https://x/{doc_id}",
        published_at=NOW - timedelta(days=days_old),
        assets=["ETH"],
    )


# --- prune_documents (pure JSONL) ------------------------------------------- #

def test_prune_documents_removes_old_keeps_recent(tmp_path):
    path = tmp_path / "documents.jsonl"
    append_documents(path, [_doc("old", 10), _doc("recent", 1), _doc("edge", 3)])

    removed = prune_documents(path, NOW - timedelta(days=7))
    assert removed == 1  # only "old" (10 days) is beyond the 7-day cutoff

    remaining = {d.id for d in iter_documents(path)}
    assert remaining == {"recent", "edge"}


def test_prune_documents_noop_when_all_recent(tmp_path):
    path = tmp_path / "documents.jsonl"
    append_documents(path, [_doc("a", 1), _doc("b", 2)])
    assert prune_documents(path, NOW - timedelta(days=7)) == 0
    assert len(list(iter_documents(path))) == 2


def test_prune_documents_missing_file(tmp_path):
    assert prune_documents(tmp_path / "nope.jsonl", NOW) == 0


# --- Chroma store prune ----------------------------------------------------- #

def test_vectorstore_prune(tmp_settings):
    store = VectorStore(tmp_settings)
    embeddings = [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]
    store.upsert_chunks([_chunk("old", 10), _chunk("recent", 1)], embeddings)
    assert store.count() == 2

    removed = store.prune(NOW - timedelta(days=7))
    assert removed == 1
    assert store.count() == 1


def test_vectorstore_prune_empty(tmp_settings):
    assert VectorStore(tmp_settings).prune(NOW) == 0


# --- prune_all orchestration ------------------------------------------------ #

def test_prune_all_prunes_store_and_docs(tmp_settings):
    # Seed JSONL docs and matching store chunks (old + recent).
    append_documents(tmp_settings.documents_file, [_doc("old", 10), _doc("recent", 1)])
    store = VectorStore(tmp_settings)
    store.upsert_chunks(
        [_chunk("old", 10), _chunk("recent", 1)], [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]
    )

    result = prune_all(keep_days=7, settings=tmp_settings)
    assert result.chunks_removed == 1
    assert result.documents_removed == 1
    assert {d.id for d in iter_documents(tmp_settings.documents_file)} == {"recent"}
    assert store.count() == 1
