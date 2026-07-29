"""Builds the active connector list from ``feeds.yaml`` + requested sources.

`--sources` accepts a subset of {news, reddit, exchange, regulator, cmc}. RSS
source types (news/exchange/regulator) map straight onto :class:`SourceType`;
reddit and cmc are handled specially. Passing ``None`` means "all configured".
"""

from __future__ import annotations

import logging

import yaml

from ..config import Settings, get_settings
from ..models import SourceType
from ..normalize import load_assets
from .base import Connector
from .coinmarketcap import CMCConnector
from .reddit import RedditConnector
from .rss import RSSConnector

logger = logging.getLogger(__name__)

# Source keys the user can request via --sources.
VALID_SOURCES = {"news", "cmc", "exchange", "regulator", "reddit"}
_RSS_TYPES = ("news", "exchange", "regulator")

# Reddit needs OAuth "script" credentials that are awkward to obtain, so it is
# opt-in only (`--sources reddit`) rather than part of the default set. CMC is
# the default community/news content source in its place.
_DEFAULT_EXCLUDES = {"reddit"}


def load_feeds(settings: Settings | None = None) -> dict:
    """Load and parse ``feeds.yaml``."""
    settings = settings or get_settings()
    with settings.feeds_file.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def parse_sources(raw: str | None) -> set[str] | None:
    """Parse a comma-separated `--sources` string into a validated set.

    Returns ``None`` for "all". Unknown tokens are dropped with a warning.
    """
    if not raw:
        return None
    requested = {tok.strip().lower() for tok in raw.split(",") if tok.strip()}
    unknown = requested - VALID_SOURCES
    if unknown:
        logger.warning("Ignoring unknown source(s): %s", ", ".join(sorted(unknown)))
    valid = requested & VALID_SOURCES
    return valid or None


def _wanted(source: str, sources: set[str] | None) -> bool:
    """Whether ``source`` is selected. None means "default set" (excludes reddit)."""
    if sources is None:
        return source not in _DEFAULT_EXCLUDES
    return source in sources


def build_connectors(
    sources: set[str] | None,
    settings: Settings | None = None,
) -> list[Connector]:
    """Construct the connectors selected by ``sources`` (None = default set).

    The default set is every configured source except Reddit (opt-in). CMC and
    Reddit are key-gated and fail soft at fetch time when credentials are absent.
    """
    settings = settings or get_settings()
    feeds = load_feeds(settings)
    connectors: list[Connector] = []

    rss = feeds.get("rss", {}) or {}
    for stype_name in _RSS_TYPES:
        if not _wanted(stype_name, sources):
            continue
        for entry in rss.get(stype_name, []) or []:
            name, url = entry.get("name"), entry.get("url")
            if not (name and url):
                continue
            connectors.append(RSSConnector(name, url, SourceType(stype_name)))

    if _wanted("cmc", sources):
        cmc_cfg = feeds.get("coinmarketcap", {}) or {}
        symbols = list(load_assets(settings).keys())
        connectors.append(
            CMCConnector(
                settings,
                symbols=symbols,
                limit=int(cmc_cfg.get("limit", 100)),
                news_type=str(cmc_cfg.get("news_type", "all")),
            )
        )

    if _wanted("reddit", sources):
        subs = (feeds.get("reddit", {}) or {}).get("subreddits", []) or []
        if subs:
            connectors.append(RedditConnector(subs, settings))

    if not connectors:
        logger.warning("No connectors selected for sources=%s", sources)
    return connectors
