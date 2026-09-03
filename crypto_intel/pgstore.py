"""Postgres/pgvector vector store — a drop-in :class:`crypto_intel.store.Store`.

Mirrors the Chroma :class:`~crypto_intel.store.VectorStore` interface so it swaps
in via ``STORE_BACKEND=pgvector`` + ``DATABASE_URL`` (Supabase works out of the
box). Chunks live in one ``chunks`` table with a ``vector(384)`` embedding column;
retrieval is a cosine-distance (`<=>`) nearest-neighbour scan with the same
asset/time/source filter the Chroma path uses.

The SQL-building helpers (:func:`filter_sql`, :func:`row_to_hit`) are pure and
unit-tested offline; ``psycopg`` / ``pgvector`` are imported lazily so the core
package and its test suite don't require them (install the ``pg`` extra to use
this backend).
"""

from __future__ import annotations

import logging
from datetime import datetime

from .config import Settings, get_settings
from .models import Chunk
from .store import QueryFilter, StoreStats

logger = logging.getLogger(__name__)

EMBED_DIM = 384  # all-MiniLM-L6-v2

SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    id            text PRIMARY KEY,
    doc_id        text NOT NULL,
    chunk_index   integer NOT NULL,
    source        text NOT NULL,
    source_name   text NOT NULL,
    url           text NOT NULL,
    published_at  timestamptz NOT NULL,
    assets        text[] NOT NULL DEFAULT '{{}}',
    text          text NOT NULL,
    embedding     vector({EMBED_DIM}) NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_published_at_idx ON chunks (published_at);
CREATE INDEX IF NOT EXISTS chunks_assets_idx ON chunks USING gin (assets);
"""


# --------------------------------------------------------------------------- #
# Pure helpers (no DB) — unit-tested offline                                  #
# --------------------------------------------------------------------------- #

def filter_sql(f: QueryFilter | None) -> tuple[str, dict]:
    """Build the ``WHERE`` clause + named params for a :class:`QueryFilter`.

    Returns ``("", {})`` when there is nothing to filter. Uses psycopg-style
    ``%(name)s`` placeholders.
    """
    clauses: list[str] = []
    params: dict[str, object] = {}
    if f is not None:
        if f.asset:
            clauses.append("%(asset)s = ANY(assets)")
            params["asset"] = f.asset.upper()
        if f.window_start is not None:
            clauses.append("published_at >= %(ws)s")
            params["ws"] = f.window_start
        if f.window_end is not None:
            clauses.append("published_at <= %(we)s")
            params["we"] = f.window_end
        if f.sources:
            clauses.append("source = ANY(%(sources)s)")
            params["sources"] = sorted(f.sources)

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


def _to_vector(embedding: list[float]):
    """Wrap an embedding so psycopg adapts it to a pgvector ``vector`` (not a
    Postgres float array — a plain list would send ``double precision[]``)."""
    from pgvector import Vector

    return Vector(embedding)


def row_to_hit(row: dict) -> dict:
    """Map a DB row into the backend-neutral hit shape used by retrieval."""
    return {
        "id": row["id"],
        "text": row["text"],
        "distance": float(row["distance"]),
        "metadata": {
            "doc_id": row["doc_id"],
            "source": row["source"],
            "source_name": row["source_name"],
            "url": row["url"],
            "published_at": int(row["published_at"]),
            "assets": ",".join(row.get("assets") or []),
        },
    }


# --------------------------------------------------------------------------- #
# Store implementation                                                        #
# --------------------------------------------------------------------------- #

class PgVectorStore:
    """Postgres/pgvector-backed store (Supabase-compatible)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._schema_ready = False

    # -- connection ------------------------------------------------------------

    def _connect(self, register: bool = True):
        dsn = self.settings.database_url
        if not dsn:
            raise RuntimeError(
                "DATABASE_URL is not set; pgvector backend needs a Postgres DSN."
            )
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "pgvector backend requires the `pg` extra: pip install -e .[pg]"
            ) from exc
        conn = psycopg.connect(dsn)
        # register_vector needs the `vector` type to already exist, so skip it
        # while bootstrapping the schema (register=False) on a fresh database.
        if register:
            register_vector(conn)
        return conn

    def init_schema(self) -> None:
        """Create the extension, table, and indexes (idempotent)."""
        with self._connect(register=False) as conn, conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        self._schema_ready = True

    def _ensure_schema(self) -> None:
        if not self._schema_ready:
            self.init_schema()

    # -- write -----------------------------------------------------------------

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """Insert chunks + embeddings, skipping ids that already exist.

        ``ON CONFLICT (id) DO NOTHING`` makes re-ingest idempotent and safe under
        concurrent cron runs.
        """
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        self._ensure_schema()

        rows = [
            (
                c.id,
                c.doc_id,
                c.chunk_index,
                c.source.value,
                c.source_name,
                c.url,
                c.published_at,
                list(c.assets),
                c.text,
                _to_vector(emb),
            )
            for c, emb in zip(chunks, embeddings)
        ]
        sql = (
            "INSERT INTO chunks (id, doc_id, chunk_index, source, source_name, url, "
            "published_at, assets, text, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (id) DO NOTHING"
        )
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(sql, rows)
        return len(chunks)

    def prune(self, older_than: datetime) -> int:
        """Delete chunks published before ``older_than`` (rolling-window retention)."""
        self._ensure_schema()
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE published_at < %s", (older_than,))
            return cur.rowcount

    def existing_doc_ids(self, doc_ids) -> set[str]:
        """Return which of ``doc_ids`` already have chunks in the store.

        Ingest dedups against this (the vector store is the source of truth), so
        a document whose chunks were pruned by retention is treated as new and
        re-embedded on the next run. See :func:`crypto_intel.pipeline.ingest_all`.
        """
        ids = list({d for d in doc_ids if d})
        if not ids:
            return set()
        self._ensure_schema()
        with self._connect(register=False) as conn, conn.cursor() as cur:
            cur.execute("SELECT DISTINCT doc_id FROM chunks WHERE doc_id = ANY(%s)", (ids,))
            return {row[0] for row in cur.fetchall()}

    # -- read ------------------------------------------------------------------

    def query(
        self,
        embedding: list[float],
        filter: QueryFilter | None = None,
        n_results: int = 10,
    ) -> list[dict]:
        """Cosine nearest-neighbour search with the given filter."""
        self._ensure_schema()
        where_sql, params = filter_sql(filter)
        params["vec"] = _to_vector(embedding)
        params["n"] = n_results
        sql = (
            "SELECT id, text, doc_id, source, source_name, url, "
            "extract(epoch FROM published_at)::bigint AS published_at, assets, "
            "(embedding <=> %(vec)s) AS distance "
            f"FROM chunks{where_sql} "
            "ORDER BY embedding <=> %(vec)s LIMIT %(n)s"
        )
        from psycopg.rows import dict_row

        with self._connect() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [row_to_hit(r) for r in rows]

    def count(self) -> int:
        self._ensure_schema()
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM chunks")
            return int(cur.fetchone()[0])

    def stats(self) -> StoreStats:
        self._ensure_schema()
        per_source: dict[str, int] = {}
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT source, count(*) FROM chunks GROUP BY source")
            for source, n in cur.fetchall():
                per_source[str(source)] = int(n)
        return StoreStats(chunk_count=sum(per_source.values()), per_source_chunks=per_source)
