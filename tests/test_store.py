"""Tests for the Chroma store + the embed/upsert path (idempotency)."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from crypto_intel.ingest.base import RawItem
from crypto_intel.models import SourceType
from crypto_intel.pipeline import ingest_all
from crypto_intel.store import (
    QueryFilter,
    VectorStore,
    chunk_metadata,
    get_store,
    to_chroma_where,
)
from crypto_intel.chunking import chunk_document
from crypto_intel.models import Document


class DeterministicEmbedder:
    """Offline stub: hashes each text into a fixed-width unit-ish vector.

    No network, no model download — good enough to exercise the store path.
    """

    model_name = "deterministic-test"
    _dim = 16

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vec = [(h[i % len(h)] / 255.0) for i in range(self._dim)]
            out.append(vec)
        return out


def _item(url, ts, text="Ethereum fell while Bitcoin held", source_name="CoinDesk"):
    return RawItem(
        source=SourceType.news,
        source_name=source_name,
        url=url,
        text=text,
        published_at=ts,
    )


def test_get_embedder_selects_backend():
    from crypto_intel.config import Settings
    from crypto_intel.embeddings import (
        OnnxEmbedder,
        SentenceTransformerEmbedder,
        get_embedder,
    )

    # Selection is lazy — constructing the embedder must not load any model.
    assert isinstance(
        get_embedder(Settings(embed_backend="onnx")), OnnxEmbedder
    )
    assert isinstance(
        get_embedder(Settings(embed_backend="sentence-transformers")),
        SentenceTransformerEmbedder,
    )


def test_to_chroma_where_combines_conditions():
    start = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
    end = datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc)
    where = to_chroma_where(
        QueryFilter(asset="ETH", window_start=start, window_end=end, sources={"news", "regulator"})
    )
    conds = where["$and"]
    assert {"has_ETH": True} in conds
    assert {"source": {"$in": ["news", "regulator"]}} in conds
    assert {"published_at": {"$gte": int(start.timestamp())}} in conds
    assert {"published_at": {"$lte": int(end.timestamp())}} in conds


def test_to_chroma_where_single_condition_not_wrapped():
    assert to_chroma_where(QueryFilter(asset="eth")) == {"has_ETH": True}


def test_to_chroma_where_none():
    assert to_chroma_where(QueryFilter()) is None
    assert to_chroma_where(None) is None


def test_get_store_selects_backend():
    from crypto_intel.config import Settings

    # Chroma is the default; construction must stay lazy (no DB connection).
    assert isinstance(get_store(Settings(store_backend="chroma")), VectorStore)

    from crypto_intel.pgstore import PgVectorStore

    store = get_store(Settings(store_backend="pgvector", database_url="postgresql://x/y"))
    assert isinstance(store, PgVectorStore)


def test_chunk_metadata_is_chroma_safe():
    doc = Document(
        id="d1",
        source=SourceType.news,
        source_name="CoinDesk",
        url="https://x/1",
        text="Ethereum and Bitcoin",
        published_at=datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc),
        assets=["ETH", "BTC"],
    )
    md = chunk_metadata(chunk_document(doc)[0])
    # All values must be scalars (str/int/float/bool) for Chroma.
    assert all(isinstance(v, (str, int, float, bool)) for v in md.values())
    assert isinstance(md["published_at"], int)
    assert md["source"] == "news"
    assert md["assets"] == "ETH,BTC"
    assert md["has_ETH"] is True and md["has_BTC"] is True


def test_ingest_embeds_and_upserts_chunks(tmp_settings):
    ts = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
    long_text = " ".join(["Ethereum dropped on ETF outflows"] * 80)  # ~400 words
    conn = _FixedConnector([_item("https://x/1", ts, text=long_text)])

    result = ingest_all(
        connectors=[conn],
        lookback_hours=48,
        settings=tmp_settings,
        embedder=DeterministicEmbedder(),
    )
    assert result.added == 1
    assert result.chunks_added >= 2  # long doc -> multiple chunks

    store = VectorStore(tmp_settings)
    stats = store.stats()
    assert stats.chunk_count == result.chunks_added
    assert stats.per_source_chunks == {"news": result.chunks_added}


def test_reingest_does_not_duplicate_chunks(tmp_settings):
    ts = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
    long_text = " ".join(["Ethereum dropped on ETF outflows"] * 80)
    conn = _FixedConnector([_item("https://x/1", ts, text=long_text)])
    emb = DeterministicEmbedder()

    first = ingest_all(connectors=[conn], settings=tmp_settings, embedder=emb)
    count_after_first = VectorStore(tmp_settings).count()
    assert count_after_first == first.chunks_added

    # Re-ingest identical content: doc-level dedup means no new chunks.
    second = ingest_all(connectors=[conn], settings=tmp_settings, embedder=emb)
    assert second.added == 0
    assert second.chunks_added == 0
    assert VectorStore(tmp_settings).count() == count_after_first


def test_reingest_after_prune_repopulates_store(tmp_settings):
    """Regression: retention deletes chunks; the next ingest must add them back.

    The vector store — not the JSONL archive — is the dedup authority, so a doc
    whose chunks were pruned is treated as new again and re-embedded, even though
    it is still recorded in the JSONL archive (as happens under DB-native
    ``pg_cron`` retention, which prunes the store but not the archive).
    """
    old_ts = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    long_text = " ".join(["Ethereum dropped on ETF outflows"] * 80)
    conn = _FixedConnector([_item("https://x/1", old_ts, text=long_text)])
    emb = DeterministicEmbedder()

    first = ingest_all(connectors=[conn], settings=tmp_settings, embedder=emb)
    store = VectorStore(tmp_settings)
    assert first.chunks_added > 0
    assert store.count() == first.chunks_added

    # Simulate rolling-window retention deleting the aged chunks from the store,
    # leaving the JSONL archive untouched.
    removed = store.prune(datetime(2026, 7, 1, tzinfo=timezone.utc))
    assert removed == first.chunks_added
    assert VectorStore(tmp_settings).count() == 0

    # Re-ingest identical content: the store is empty, so the doc is re-embedded.
    second = ingest_all(connectors=[conn], settings=tmp_settings, embedder=emb)
    assert second.added == 1
    assert second.chunks_added == first.chunks_added
    assert VectorStore(tmp_settings).count() == first.chunks_added


def test_existing_doc_ids_reports_only_stored_docs(tmp_settings):
    from crypto_intel.pipeline import load_existing_ids

    ts = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
    long_text = " ".join(["Ethereum dropped on ETF outflows"] * 80)
    conn = _FixedConnector([_item("https://x/1", ts, text=long_text)])
    ingest_all(connectors=[conn], settings=tmp_settings, embedder=DeterministicEmbedder())

    stored = load_existing_ids(tmp_settings.documents_file)  # the one doc's id
    assert len(stored) == 1

    store = VectorStore(tmp_settings)
    present = store.existing_doc_ids(list(stored) + ["missing-doc-id"])
    assert present == stored
    # Empty input is a cheap no-op.
    assert store.existing_doc_ids([]) == set()


class _FixedConnector:
    name = "fixed"
    source_type = SourceType.news

    def __init__(self, items):
        self._items = items

    def fetch(self, lookback_hours):
        return list(self._items)
