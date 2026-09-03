"""Vector store abstraction + persistent Chroma implementation.

The store is kept behind a small backend-neutral seam so the pipeline and
retrieval don't depend on any one engine:

- :class:`QueryFilter` — a backend-neutral filter (asset, time window, sources).
- :class:`Store` — the protocol every backend implements (upsert / query / count / stats).
- :class:`VectorStore` — the default embedded Chroma implementation.
- :func:`get_store` — factory selecting the backend from ``settings.store_backend``
  (``chroma`` default, or ``pgvector`` → :class:`crypto_intel.pgstore.PgVectorStore`).

Each backend translates :class:`QueryFilter` into its own query language, and all
of them return hits in the same ``{id, text, metadata, distance}`` shape so
:mod:`crypto_intel.retrieve` is engine-agnostic.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from .config import Settings, get_settings
from .models import Chunk

# Metadata key prefix for per-asset boolean flags (enables Chroma `where`
# filtering by ticker, since Chroma metadata cannot store lists).
ASSET_FLAG_PREFIX = "has_"


@dataclass
class StoreStats:
    """Aggregate view of what's in the vector store."""

    chunk_count: int
    per_source_chunks: dict[str, int]


@dataclass
class QueryFilter:
    """Backend-neutral retrieval filter: asset flag, time window, source subset."""

    asset: str | None = None
    window_start: datetime | None = None
    window_end: datetime | None = None
    sources: set[str] | None = None


def to_chroma_where(f: QueryFilter | None) -> dict | None:
    """Translate a :class:`QueryFilter` into a Chroma ``where`` dict (or None)."""
    if f is None:
        return None
    conditions: list[dict] = []
    if f.asset:
        conditions.append({f"{ASSET_FLAG_PREFIX}{f.asset.upper()}": True})
    if f.window_start is not None and f.window_end is not None:
        # Chroma allows only one operator per field expression, so the range is
        # expressed as two conditions (combined via $and below).
        conditions.append({"published_at": {"$gte": int(f.window_start.timestamp())}})
        conditions.append({"published_at": {"$lte": int(f.window_end.timestamp())}})
    if f.sources:
        conditions.append({"source": {"$in": sorted(f.sources)}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


@runtime_checkable
class Store(Protocol):
    """Interface every vector-store backend implements."""

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        ...

    def query(
        self, embedding: list[float], filter: QueryFilter | None = None, n_results: int = 10
    ) -> list[dict]:
        ...

    def prune(self, older_than: datetime) -> int:
        ...

    def existing_doc_ids(self, doc_ids: Iterable[str]) -> set[str]:
        ...

    def count(self) -> int:
        ...

    def stats(self) -> StoreStats:
        ...


def chunk_metadata(chunk: Chunk) -> dict:
    """Build Chroma-safe metadata (scalars only) for a chunk.

    `published_at` is stored as an epoch int for range filtering; `assets` is
    kept as a CSV string for provenance, plus one boolean flag per ticker
    (`has_ETH: True`) so retrieval can filter by asset membership.
    """
    md: dict[str, object] = {
        "doc_id": chunk.doc_id,
        "chunk_index": chunk.chunk_index,
        "source": chunk.source.value,
        "source_name": chunk.source_name,
        "url": chunk.url,
        "published_at": int(chunk.published_at.timestamp()),
        "assets": ",".join(chunk.assets),
    }
    for ticker in chunk.assets:
        md[f"{ASSET_FLAG_PREFIX}{ticker}"] = True
    return md


class VectorStore:
    """Thin wrapper around a persistent Chroma collection."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = None
        self._collection = None

    # -- lazy Chroma handles (import is deferred so `--help` stays fast) --------

    @property
    def collection(self):
        if self._collection is None:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            chroma_dir = self.settings.chroma_dir
            chroma_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.settings.chroma_collection,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # -- write APIs ------------------------------------------------------------

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """Upsert chunks + precomputed embeddings. Idempotent by chunk id.

        Passing embeddings explicitly means Chroma never invokes its own
        embedding function, so no model download happens inside the store.
        """
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")

        self.collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[chunk_metadata(c) for c in chunks],
        )
        return len(chunks)

    # -- read APIs -------------------------------------------------------------

    def query(
        self,
        embedding: list[float],
        filter: QueryFilter | None = None,
        n_results: int = 10,
    ) -> list[dict]:
        """Semantic query with an optional :class:`QueryFilter`.

        Returns a list of hits, each ``{id, text, metadata, distance}``, ordered
        by ascending cosine distance (closest first). Empty if the store is empty.
        """
        if self.count() == 0:
            return []
        res = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=to_chroma_where(filter),
            include=["documents", "metadatas", "distances"],
        )
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        return [
            {"id": i, "text": d, "metadata": m or {}, "distance": dist}
            for i, d, m, dist in zip(ids, docs, metas, dists)
        ]

    def prune(self, older_than: datetime) -> int:
        """Delete chunks published before ``older_than``. Returns count removed."""
        if self.count() == 0:
            return 0
        cutoff = int(older_than.timestamp())
        got = self.collection.get(where={"published_at": {"$lt": cutoff}})
        ids = got.get("ids") or []
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def existing_doc_ids(self, doc_ids: Iterable[str]) -> set[str]:
        """Return which of ``doc_ids`` already have chunks in the store.

        Ingest dedups against this (the vector store is the source of truth), so
        a document whose chunks were pruned by retention is treated as new and
        re-embedded on the next run. See :func:`crypto_intel.pipeline.ingest_all`.
        """
        ids = list({d for d in doc_ids if d})
        if not ids or self.count() == 0:
            return set()
        got = self.collection.get(where={"doc_id": {"$in": ids}}, include=["metadatas"])
        return {
            str(md["doc_id"])
            for md in (got.get("metadatas") or [])
            if md and md.get("doc_id") is not None
        }

    def count(self) -> int:
        """Total number of chunks stored."""
        return self.collection.count()

    def stats(self) -> StoreStats:
        """Return chunk counts, broken down by source."""
        total = self.count()
        per_source: Counter[str] = Counter()
        if total:
            got = self.collection.get(include=["metadatas"])
            for md in got.get("metadatas") or []:
                per_source[str((md or {}).get("source", "unknown"))] += 1
        return StoreStats(chunk_count=total, per_source_chunks=dict(per_source))


# Backends selectable via settings.store_backend.
_PG_BACKENDS = {"pgvector", "pg", "postgres", "postgresql", "supabase"}


def get_store(settings: Settings | None = None) -> Store:
    """Return the configured vector store (``chroma`` default, or ``pgvector``)."""
    settings = settings or get_settings()
    backend = (settings.store_backend or "chroma").lower()
    if backend in _PG_BACKENDS:
        from .pgstore import PgVectorStore  # deferred (optional psycopg/pgvector deps)

        return PgVectorStore(settings)
    return VectorStore(settings)
