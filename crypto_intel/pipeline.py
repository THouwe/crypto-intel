"""Pipeline orchestration.

S2 provides :func:`ingest_all` (fetch -> normalize -> dedup against the vector
store -> embed/upsert -> archive to JSONL) plus the small JSONL document-store
helpers it needs. `ask(...)` arrives in later phases.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from .chunking import chunk_document
from .config import Settings, get_settings
from .embeddings import Embedder, get_embedder
from .ingest.base import Connector
from .ingest.registry import build_connectors
from .models import Answer, Chunk, Document, PriceEvent
from .normalize import load_assets, normalize
from .retrieve import ParsedQuestion, RetrievedChunk, parse_question, retrieve
from .store import Store, get_store

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# JSONL document store                                                        #
# --------------------------------------------------------------------------- #

def iter_documents(path: Path) -> Iterable[Document]:
    """Yield every :class:`Document` persisted in ``path`` (empty if missing)."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield Document.model_validate_json(line)
            except Exception:  # skip corrupt lines rather than crash
                continue


def load_existing_ids(path: Path) -> set[str]:
    """Return the set of document ids already persisted in ``path``."""
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def append_documents(path: Path, docs: list[Document]) -> None:
    """Append documents to the JSONL store (creating parent dirs as needed)."""
    if not docs:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(doc.model_dump_json())
            fh.write("\n")


def prune_documents(path: Path, older_than: datetime) -> int:
    """Drop documents published before ``older_than`` from the JSONL store.

    Rewrites the file atomically (temp file + replace). Returns count removed.
    """
    if not path.exists():
        return 0
    kept: list[Document] = []
    removed = 0
    for doc in iter_documents(path):
        if doc.published_at >= older_than:
            kept.append(doc)
        else:
            removed += 1
    if removed == 0:
        return 0
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for doc in kept:
            fh.write(doc.model_dump_json())
            fh.write("\n")
    tmp.replace(path)
    return removed


def read_documents_summary(
    path: Path,
) -> tuple[int, dict[str, int], tuple[str | None, str | None]]:
    """Return (total, per_source_counts, (min_ts_iso, max_ts_iso)) for the store."""
    total = 0
    per_source: Counter[str] = Counter()
    min_ts: datetime | None = None
    max_ts: datetime | None = None
    for doc in iter_documents(path):
        total += 1
        per_source[doc.source.value] += 1
        ts = doc.published_at
        min_ts = ts if min_ts is None or ts < min_ts else min_ts
        max_ts = ts if max_ts is None or ts > max_ts else max_ts

    def fmt(d: datetime | None) -> str | None:
        return d.astimezone(timezone.utc).isoformat() if d else None

    return total, dict(per_source), (fmt(min_ts), fmt(max_ts))


# --------------------------------------------------------------------------- #
# Ingestion                                                                   #
# --------------------------------------------------------------------------- #

@dataclass
class IngestResult:
    """Summary of one ingest run."""

    fetched: int = 0
    added: int = 0
    duplicates: int = 0
    chunks_added: int = 0
    per_source_added: dict[str, int] = field(default_factory=dict)
    connectors: int = 0
    errors: int = 0
    skipped: bool = False  # True when skipped because the store was already populated


def ingest_all(
    sources: set[str] | None = None,
    lookback_hours: int | None = None,
    settings: Settings | None = None,
    connectors: list[Connector] | None = None,
    embed: bool = True,
    embedder: Embedder | None = None,
    store: Store | None = None,
    skip_if_populated: bool = False,
) -> IngestResult:
    """Fetch from the selected connectors, normalize, dedup, persist, embed.

    ``connectors``/``embedder``/``store`` can be injected for testing; otherwise
    they're built from config. Never raises for a single bad connector —
    failures are counted and logged (fail soft). New documents are chunked,
    embedded, and upserted into the vector store, then appended to the JSONL
    archive. Chunk ids are deterministic, so the upsert is idempotent (no
    duplicate chunks).

    **Deduplication is keyed on the vector store, not the JSONL archive.** A
    fetched document is (re-)embedded whenever its chunks are *not* already in
    the store. This is what lets retention re-populate the DB: after a rolling
    window deletes old chunks (``deploy/retention.sql`` or ``prune_all``), the
    same feed items are re-ingested cleanly instead of being skipped as
    "already seen". The JSONL archive can validly diverge from the store (e.g.
    DB-native ``pg_cron`` retention prunes the store but not the JSONL), so it is
    NOT used to gate embedding — it is deduped only against its own ids when
    appending, to keep the archive free of duplicate lines. When ``embed`` is
    False (no store write), dedup falls back to the JSONL archive so that path
    stays idempotent.

    ``skip_if_populated=True`` makes this a no-op when the store already has
    chunks — used for the one-time initial populate on deploy (so restarts don't
    re-ingest).
    """
    settings = settings or get_settings()
    lookback_hours = lookback_hours or settings.default_lookback_hours

    # The vector store is the dedup authority; build it once and reuse it for the
    # populated check, the existence lookup, and the upsert.
    if embed and store is None:
        store = get_store(settings)

    if skip_if_populated:
        populated_store = store if store is not None else get_store(settings)
        existing = populated_store.count()
        if existing > 0:
            logger.info("Store already has %d chunk(s); skipping initial ingest.", existing)
            return IngestResult(skipped=True)

    assets_cfg = load_assets(settings)
    if connectors is None:
        connectors = build_connectors(sources, settings)

    doc_path = settings.documents_file
    result = IngestResult(connectors=len(connectors))
    per_source: Counter[str] = Counter()

    # First pass: fetch + normalize, deduping within this batch only.
    candidates: list[Document] = []
    seen_batch: set[str] = set()
    for connector in connectors:
        try:
            items = list(connector.fetch(lookback_hours))
        except Exception as exc:  # defensive: connectors should fail soft internally
            logger.warning("Connector %s failed: %s", getattr(connector, "name", "?"), exc)
            result.errors += 1
            continue

        for item in items:
            result.fetched += 1
            doc = normalize(item, assets_cfg)
            if doc.id in seen_batch:
                result.duplicates += 1
                continue
            seen_batch.add(doc.id)
            candidates.append(doc)

    # Second pass: drop candidates whose chunks the store already holds. When
    # embedding, ask the store (the source of truth); otherwise fall back to the
    # JSONL archive so the non-embedding path remains idempotent.
    if embed:
        already = store.existing_doc_ids([d.id for d in candidates]) if candidates else set()
    else:
        already = load_existing_ids(doc_path)
    new_docs = [d for d in candidates if d.id not in already]
    result.duplicates += len(candidates) - len(new_docs)
    for doc in new_docs:
        per_source[doc.source.value] += 1

    # Embed + upsert BEFORE persisting to JSONL. If embedding fails (e.g. the
    # backend isn't installed), nothing is committed, so a retry reprocesses the
    # same docs cleanly instead of leaving JSONL rows with no chunks. The upsert
    # is idempotent (deterministic chunk ids), so a crash after upsert but before
    # append is also safe — the retry overwrites the same chunks.
    if embed and new_docs:
        result.chunks_added = _embed_and_store(new_docs, settings, embedder, store)

    # Append to the JSONL archive, deduped against its own ids so a doc that was
    # pruned from the store but still recorded here doesn't pile up a second line.
    archived = load_existing_ids(doc_path)
    append_documents(doc_path, [d for d in new_docs if d.id not in archived])
    result.added = len(new_docs)
    result.per_source_added = dict(per_source)

    return result


def _embed_and_store(
    docs: list[Document],
    settings: Settings,
    embedder: Embedder | None,
    store: Store | None,
) -> int:
    """Chunk ``docs``, embed the chunks, and upsert them into the vector store."""
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(
            chunk_document(doc, settings.chunk_size, settings.chunk_overlap)
        )
    if not chunks:
        return 0

    embedder = embedder or get_embedder(settings)
    store = store or get_store(settings)
    vectors = embedder.encode([c.text for c in chunks])
    store.upsert_chunks(chunks, vectors)
    logger.info("Embedded + upserted %d chunk(s)", len(chunks))
    return len(chunks)


# --------------------------------------------------------------------------- #
# Retrieval orchestration (S5)                                                #
# --------------------------------------------------------------------------- #

@dataclass
class RetrievalContext:
    """Everything the answer step needs: parse, price event, evidence, notes."""

    parsed: ParsedQuestion
    event: PriceEvent | None
    chunks: list[RetrievedChunk]
    notes: list[str] = field(default_factory=list)


def retrieve_context(
    question: str,
    *,
    settings: Settings | None = None,
    asset_override: str | None = None,
    hours_override: int | None = None,
    k: int | None = None,
    sources: set[str] | None = None,
    use_bm25: bool = True,
    detect_prices: bool = True,
    store: Store | None = None,
) -> RetrievalContext:
    """Parse the question, confirm the move, and retrieve time-windowed evidence.

    The price event's analysed window drives the retrieval time filter. If price
    detection fails (or is disabled), retrieval falls back to the parsed window
    ``[now - hours, now]`` and a note is recorded. Shared by ``ask`` (S6).

    ``store`` can be injected to retrieve from a specific backend (e.g. the
    pgvector DB via :func:`ask_db`); otherwise the configured store is used.
    """
    settings = settings or get_settings()
    k = k or settings.default_k
    parsed = parse_question(question, settings, asset_override, hours_override)
    notes: list[str] = []

    if not parsed.asset:
        notes.append("Could not identify an asset in the question; pass --asset.")
        return RetrievalContext(parsed, None, [], notes)

    window_end = datetime.now(timezone.utc)
    window_start = window_end - timedelta(hours=parsed.hours)
    event: PriceEvent | None = None

    if detect_prices:
        from .prices import PriceError, detect_event  # deferred (needs network)

        try:
            event = detect_event(parsed.asset, parsed.hours, settings=settings)
            window_start, window_end = event.window_start, event.window_end
        except PriceError as exc:
            notes.append(f"Price event unavailable ({exc}); retrieving by parsed window.")

    chunks = retrieve(
        question,
        asset=parsed.asset,
        window_start=window_start,
        window_end=window_end,
        k=k,
        settings=settings,
        store=store,
        sources=sources,
        use_bm25=use_bm25,
    )
    if not chunks:
        notes.append(
            "No evidence in the asset+time window — try a wider --hours or ingest more sources."
        )
    return RetrievalContext(parsed, event, chunks, notes)


def ask_db(
    question: str,
    *,
    settings: Settings | None = None,
    asset_override: str | None = None,
    hours_override: int | None = None,
    k: int | None = None,
    sources: set[str] | None = None,
    use_bm25: bool = True,
    store: Store | None = None,
) -> RetrievalContext:
    """Retrieve time-and-asset-windowed evidence from the pgvector DB (no LLM).

    Forces the Postgres/pgvector backend (needs ``DATABASE_URL`` + the ``pg``
    extra), independent of ``STORE_BACKEND`` — this is the shared-DB retrieval the
    web GUI uses. Price detection is skipped (the GUI confirms the move
    separately). ``store`` can be injected for tests.
    """
    settings = settings or get_settings()
    if store is None:
        from .pgstore import PgVectorStore  # deferred (optional pg deps)

        store = PgVectorStore(settings)
    return retrieve_context(
        question,
        settings=settings,
        asset_override=asset_override,
        hours_override=hours_override,
        k=k,
        sources=sources,
        use_bm25=use_bm25,
        detect_prices=False,
        store=store,
    )


def _maybe_forecast(event, settings, client=None):
    """Best-effort volatility forecast for the event's asset (S12).

    Returns a ``VolForecast`` when a model is trained and the forecast extras +
    data are available; ``None`` otherwise. Never raises — the regime is a bonus
    on top of the cited answer, and must never break ``ask``.
    """
    if event is None:
        return None
    try:
        from .forecast.predict import predict as run_predict

        return run_predict(event.asset, settings=settings, client=client)
    except Exception as exc:  # no model / deps / data → silently skip
        logger.debug("Regime forecast unavailable for %s: %s", event.asset, exc)
        return None


def _forecast_context(fc) -> str:
    """A one-line market-state string for the synthesis prompt (background only)."""
    skill = f", skill {fc.skill_vs_baseline:+.2f} vs baseline" if fc.skill_vs_baseline is not None else ""
    return (
        f"Current volatility regime for {fc.asset}: {fc.regime.upper()} "
        f"(next-{fc.horizon_hours}h realized-vol forecast {fc.predicted_vol_annualized:.0%} "
        f"annualized, model {fc.model_name}{skill}). Background on how turbulent "
        f"conditions are — a volatility estimate, not a price prediction."
    )


def ask(
    question: str,
    *,
    settings: Settings | None = None,
    asset_override: str | None = None,
    hours_override: int | None = None,
    k: int | None = None,
    sources: set[str] | None = None,
    use_bm25: bool = True,
    model: str | None = None,
    with_regime: bool = True,
) -> Answer:
    """Full ``ask`` flow: retrieve time-windowed evidence, then synthesize a
    grounded, cited :class:`Answer`. Notes from retrieval are merged in.

    When ``with_regime`` and a forecast model is trained for the asset, the current
    volatility regime (S12) is woven into the prompt as market-state context and
    attached to ``Answer.market_state`` — background only, never a price/trade call.
    """
    settings = settings or get_settings()
    ctx = retrieve_context(
        question,
        settings=settings,
        asset_override=asset_override,
        hours_override=hours_override,
        k=k,
        sources=sources,
        use_bm25=use_bm25,
    )
    from .synthesize import synthesize  # deferred (optional anthropic dep)

    forecast = _maybe_forecast(ctx.event, settings) if with_regime else None
    fc_context = _forecast_context(forecast) if forecast else None

    answer = synthesize(
        question, ctx.event, ctx.chunks, settings=settings, model=model,
        forecast_context=fc_context,
    )
    answer.notes = ctx.notes + answer.notes
    if forecast is not None:
        answer.market_state = {
            "regime": forecast.regime,
            "model_name": forecast.model_name,
            "skill_vs_baseline": forecast.skill_vs_baseline,
            "predicted_vol_annualized": forecast.predicted_vol_annualized,
        }
    return answer


# --------------------------------------------------------------------------- #
# Retention (rolling window)                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class PruneResult:
    """Summary of a retention prune."""

    cutoff: datetime
    chunks_removed: int
    documents_removed: int


def prune_all(keep_days: float, settings: Settings | None = None) -> PruneResult:
    """Delete stored content older than ``keep_days`` (vector store + JSONL).

    The rolling-window retention for the periodic-ingest deployment: run it after
    each ingest (or on its own schedule) to keep only the last ``keep_days``.
    """
    settings = settings or get_settings()
    cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days)
    store = get_store(settings)
    chunks_removed = store.prune(cutoff)
    documents_removed = prune_documents(settings.documents_file, cutoff)
    return PruneResult(
        cutoff=cutoff, chunks_removed=chunks_removed, documents_removed=documents_removed
    )
