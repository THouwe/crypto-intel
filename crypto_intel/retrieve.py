"""Question parsing + time-and-asset-filtered retrieval with BM25 rerank.

Flow:
1. :func:`parse_question` pulls the asset (via ``assets.yaml`` aliases), a time
   window (from phrases like "today" / "last 24h" / "this week"), and any claimed
   percentage ("dropped 6%") out of the natural-language question.
2. :func:`retrieve` builds a Chroma ``where`` filter (asset flag AND
   ``published_at`` within the window, optional source subset), runs a semantic
   top-N query, optionally reranks the candidates with BM25 over the query terms,
   dedups by ``doc_id``, and returns the top-k with full provenance.

The pure pieces (parsing, rerank) have no network/store dependency so they're
unit-testable offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .config import Settings, get_settings
from .embeddings import Embedder, get_embedder
from .normalize import detect_primary_asset, load_assets
from .store import QueryFilter, Store, get_store


@dataclass
class ParsedQuestion:
    """Structured view of a natural-language question."""

    question: str
    asset: str | None
    hours: int
    claimed_pct: float | None = None
    window_explicit: bool = False  # True if a time phrase/override set the window


@dataclass
class RetrievedChunk:
    """A retrieved chunk with provenance and scoring detail."""

    chunk_id: str
    doc_id: str
    text: str
    source: str
    source_name: str
    url: str
    published_at: datetime
    assets: list[str] = field(default_factory=list)
    semantic_score: float = 0.0  # 1 - cosine distance
    bm25_score: float = 0.0
    score: float = 0.0  # final combined ranking score


# --------------------------------------------------------------------------- #
# Question parsing                                                            #
# --------------------------------------------------------------------------- #

_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_NUM_HOURS_RE = re.compile(r"(?:last|past|previous)?\s*(\d+)\s*(?:hours?|hrs?|h)\b", re.IGNORECASE)
_NUM_DAYS_RE = re.compile(r"(?:last|past|previous)?\s*(\d+)\s*(?:days?|d)\b", re.IGNORECASE)
_NUM_WEEKS_RE = re.compile(r"(?:last|past|previous)?\s*(\d+)\s*(?:weeks?|wks?|w)\b", re.IGNORECASE)


def parse_window_hours(text: str) -> int | None:
    """Map a time phrase in ``text`` to a number of hours, or None if absent."""
    t = text.lower()

    m = _NUM_HOURS_RE.search(t)
    if m:
        return int(m.group(1))
    m = _NUM_DAYS_RE.search(t)
    if m:
        return int(m.group(1)) * 24
    m = _NUM_WEEKS_RE.search(t)
    if m:
        return int(m.group(1)) * 168

    # Named phrases (checked after explicit numbers).
    if "yesterday" in t:
        return 48
    if "today" in t or "24h" in t or "past day" in t or "last day" in t:
        return 24
    if "this week" in t or "past week" in t or "last week" in t or "7d" in t or "weekly" in t:
        return 168
    if "this month" in t or "past month" in t or "last month" in t or "30d" in t:
        return 720
    return None


def parse_claimed_pct(text: str) -> float | None:
    """Extract a claimed percentage magnitude (e.g. '6%' -> 6.0)."""
    m = _PCT_RE.search(text)
    return float(m.group(1)) if m else None


def parse_question(
    question: str,
    settings: Settings | None = None,
    asset_override: str | None = None,
    hours_override: int | None = None,
) -> ParsedQuestion:
    """Parse asset, time window, and claimed percentage from a question."""
    settings = settings or get_settings()
    assets_cfg = load_assets(settings)

    asset = (asset_override or detect_primary_asset(question, assets_cfg) or None)
    if asset:
        asset = asset.upper()

    parsed_hours = parse_window_hours(question)
    if hours_override is not None:
        hours, explicit = hours_override, True
    elif parsed_hours is not None:
        hours, explicit = parsed_hours, True
    else:
        hours, explicit = settings.default_lookback_hours, False

    return ParsedQuestion(
        question=question,
        asset=asset,
        hours=hours,
        claimed_pct=parse_claimed_pct(question),
        window_explicit=explicit,
    )


# --------------------------------------------------------------------------- #
# BM25 rerank (pure)                                                          #
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def bm25_scores(query: str, texts: list[str]) -> list[float]:
    """Raw BM25 relevance scores of ``texts`` against ``query``."""
    if not texts:
        return []
    from rank_bm25 import BM25Okapi

    tokenized = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    return list(bm25.get_scores(_tokenize(query)))


def rerank_candidates(
    query: str,
    candidates: list[RetrievedChunk],
    alpha: float = 0.5,
) -> list[RetrievedChunk]:
    """Rerank by a blend of semantic similarity + max-normalized BM25 (desc).

    Semantic cosine similarity (``1 - distance``) is already on an absolute
    scale, so it's used directly; BM25 is scaled by the candidate-set max into
    [0, 1] so the two signals are comparable. ``alpha`` weights semantic,
    ``1 - alpha`` weights BM25. Mutates each candidate's ``bm25_score`` /
    ``score`` and returns them sorted.
    """
    if not candidates:
        return []
    raw_bm25 = bm25_scores(query, [c.text for c in candidates])
    max_bm = max(raw_bm25) if raw_bm25 else 0.0
    for cand, bm in zip(candidates, raw_bm25):
        cand.bm25_score = bm
        bm_norm = (bm / max_bm) if max_bm > 0 else 0.0
        cand.score = alpha * cand.semantic_score + (1.0 - alpha) * bm_norm
    return sorted(candidates, key=lambda c: (c.score, c.semantic_score), reverse=True)


def _dedup_by_doc(candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Keep the best-ranked chunk per ``doc_id`` (order preserved)."""
    seen: set[str] = set()
    out: list[RetrievedChunk] = []
    for cand in candidates:
        if cand.doc_id in seen:
            continue
        seen.add(cand.doc_id)
        out.append(cand)
    return out


# --------------------------------------------------------------------------- #
# Filter + retrieval                                                          #
# --------------------------------------------------------------------------- #

def build_filter(
    asset: str | None,
    window_start: datetime | None,
    window_end: datetime | None,
    sources: set[str] | None = None,
) -> QueryFilter:
    """Build a backend-neutral :class:`QueryFilter` for a query.

    Each store translates it to its own query language (Chroma ``where`` dict or
    a Postgres ``WHERE`` clause), so retrieval stays engine-agnostic.
    """
    return QueryFilter(
        asset=asset.upper() if asset else None,
        window_start=window_start,
        window_end=window_end,
        sources=sources or None,
    )


def _hit_to_chunk(hit: dict) -> RetrievedChunk:
    md = hit.get("metadata") or {}
    ts = md.get("published_at")
    published_at = (
        datetime.fromtimestamp(int(ts), tz=timezone.utc)
        if ts is not None
        else datetime.now(timezone.utc)
    )
    assets_csv = md.get("assets") or ""
    distance = hit.get("distance")
    semantic = 1.0 - float(distance) if distance is not None else 0.0
    return RetrievedChunk(
        chunk_id=hit.get("id", ""),
        doc_id=str(md.get("doc_id", "")),
        text=hit.get("text") or "",
        source=str(md.get("source", "")),
        source_name=str(md.get("source_name", "")),
        url=str(md.get("url", "")),
        published_at=published_at,
        assets=[a for a in assets_csv.split(",") if a],
        semantic_score=semantic,
    )


def retrieve(
    query: str,
    *,
    asset: str | None,
    window_start: datetime | None,
    window_end: datetime | None,
    k: int,
    settings: Settings | None = None,
    store: Store | None = None,
    embedder: Embedder | None = None,
    sources: set[str] | None = None,
    use_bm25: bool = True,
    overfetch: int = 4,
) -> list[RetrievedChunk]:
    """Retrieve the top-k time-and-asset-filtered chunks for ``query``."""
    settings = settings or get_settings()
    store = store or get_store(settings)
    if store.count() == 0:
        return []

    embedder = embedder or get_embedder(settings)
    query_filter = build_filter(asset, window_start, window_end, sources)
    n = max(k * overfetch, 20)
    query_vec = embedder.encode([query])[0]
    hits = store.query(query_vec, query_filter, n)

    candidates = [_hit_to_chunk(h) for h in hits]
    if use_bm25 and candidates:
        candidates = rerank_candidates(query, candidates)
    else:
        for cand in candidates:
            cand.score = cand.semantic_score
        candidates.sort(key=lambda c: c.score, reverse=True)

    return _dedup_by_doc(candidates)[:k]
