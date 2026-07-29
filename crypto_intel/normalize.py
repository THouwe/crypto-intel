"""RawItem -> Document: text cleaning, asset tagging, and stable dedup ids.

Asset tagging is alias-based (from ``assets.yaml``): a ticker is tagged when any
of its aliases (or the ticker symbol itself) appears as a whole word in the
title+body. The dedup id hashes source_name + url + published_at, so re-ingesting
the same item is a no-op.
"""

from __future__ import annotations

import hashlib
import html
import re
from datetime import datetime, timezone
from functools import lru_cache

import yaml

from .config import Settings, get_settings
from .ingest.base import RawItem
from .models import Document

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


@lru_cache(maxsize=4)
def _load_assets_cached(path_str: str) -> dict[str, dict]:
    with open(path_str, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_assets(settings: Settings | None = None) -> dict[str, dict]:
    """Load ``assets.yaml`` as {TICKER: {coingecko_id, aliases}}."""
    settings = settings or get_settings()
    return _load_assets_cached(str(settings.assets_file))


def clean_text(raw: str | None) -> str:
    """Strip HTML tags/entities and collapse whitespace."""
    if not raw:
        return ""
    text = _TAG_RE.sub(" ", raw)
    text = html.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def _alias_patterns(assets_cfg: dict[str, dict]) -> dict[str, re.Pattern]:
    """Compile one whole-word alias regex per ticker (cached by identity)."""
    patterns: dict[str, re.Pattern] = {}
    for ticker, cfg in assets_cfg.items():
        aliases = set(a.lower() for a in (cfg or {}).get("aliases", []))
        aliases.add(ticker.lower())
        # Longest first so multi-word aliases match cleanly; escape each.
        escaped = sorted((re.escape(a) for a in aliases if a), key=len, reverse=True)
        patterns[ticker] = re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)
    return patterns


def tag_assets(text: str, assets_cfg: dict[str, dict]) -> list[str]:
    """Return the sorted list of tickers mentioned in ``text``."""
    if not text:
        return []
    patterns = _alias_patterns(assets_cfg)
    found = [ticker for ticker, pat in patterns.items() if pat.search(text)]
    return sorted(set(found))


def detect_primary_asset(text: str, assets_cfg: dict[str, dict]) -> str | None:
    """Return the ticker whose alias appears earliest in ``text`` (or None)."""
    if not text:
        return None
    patterns = _alias_patterns(assets_cfg)
    best_ticker: str | None = None
    best_pos = len(text) + 1
    for ticker, pat in patterns.items():
        m = pat.search(text)
        if m and m.start() < best_pos:
            best_pos, best_ticker = m.start(), ticker
    return best_ticker


def make_doc_id(source_name: str, url: str, published_at: datetime) -> str:
    """Stable dedup key: sha1(source_name + url + published_at ISO)."""
    ts = published_at.astimezone(timezone.utc).isoformat()
    raw = f"{source_name}\x1f{url}\x1f{ts}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def normalize(item: RawItem, assets_cfg: dict[str, dict]) -> Document:
    """Convert a :class:`RawItem` into a normalized :class:`Document`."""
    published_at = _ensure_utc(item.published_at)
    title = clean_text(item.title) if item.title else None
    body = clean_text(item.text)
    # Tag against title + body so headline-only tickers are captured.
    tag_source = f"{title or ''} {body}"
    assets = tag_assets(tag_source, assets_cfg)

    return Document(
        id=make_doc_id(item.source_name, item.url, published_at),
        source=item.source,
        source_name=item.source_name,
        url=item.url,
        title=title,
        text=body,
        author=item.author,
        published_at=published_at,
        assets=assets,
        metadata=item.metadata or {},
    )
