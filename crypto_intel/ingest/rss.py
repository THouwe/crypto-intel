"""Generic RSS/Atom connector.

One instance per feed entry in ``feeds.yaml``. Covers news, exchange, and
regulator sources — they differ only in `source_type` and `name`. Uses
``feedparser``, which fetches and tolerates messy real-world feeds; network and
parse failures surface as an empty result plus a logged warning (fail soft).
"""

from __future__ import annotations

import calendar
import logging
from datetime import datetime, timedelta, timezone
from typing import Iterable

import feedparser

from ..models import SourceType
from .base import RawItem

logger = logging.getLogger(__name__)

# Many feeds reject the default python-feedparser agent; present a browser-ish UA.
_FEED_UA = "crypto-intel/0.1 (+https://github.com/) RSS reader"


def _entry_datetime(entry) -> datetime | None:
    """Extract a timezone-aware UTC datetime from a feed entry, or None."""
    for key in ("published_parsed", "updated_parsed"):
        struct = entry.get(key)
        if struct:
            # struct is a time.struct_time in UTC; calendar.timegm treats it as such.
            return datetime.fromtimestamp(calendar.timegm(struct), tz=timezone.utc)
    return None


def _entry_text(entry) -> str:
    """Best-effort body text from an entry (content > summary > title)."""
    content = entry.get("content")
    if content:
        # feedparser gives a list of {value, type, ...}
        parts = [c.get("value", "") for c in content if c.get("value")]
        if parts:
            return "\n".join(parts)
    return entry.get("summary") or entry.get("title") or ""


class RSSConnector:
    """Reads a single RSS/Atom feed."""

    def __init__(self, name: str, url: str, source_type: SourceType) -> None:
        self.name = name
        self.url = url
        self.source_type = source_type

    def fetch(self, lookback_hours: int) -> Iterable[RawItem]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        try:
            parsed = feedparser.parse(self.url, agent=_FEED_UA)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Feed %s (%s) failed to parse: %s", self.name, self.url, exc)
            return

        if getattr(parsed, "bozo", False) and not parsed.entries:
            logger.warning(
                "Feed %s (%s) unreachable or malformed: %s",
                self.name,
                self.url,
                getattr(parsed, "bozo_exception", "unknown error"),
            )
            return

        kept = 0
        for entry in parsed.entries:
            published_at = _entry_datetime(entry)
            if published_at is None:
                # Skip undated items so the dedup id (which hashes the timestamp)
                # stays stable across runs.
                continue
            if published_at < cutoff:
                continue

            url = entry.get("link") or entry.get("id") or ""
            if not url:
                continue

            yield RawItem(
                source=self.source_type,
                source_name=self.name,
                url=url,
                text=_entry_text(entry),
                published_at=published_at,
                title=entry.get("title"),
                author=entry.get("author"),
                metadata={"feed": self.url},
            )
            kept += 1

        logger.info("Feed %s: %d item(s) within %dh window", self.name, kept, lookback_hours)
