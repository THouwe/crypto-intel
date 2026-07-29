"""CoinMarketCap content connector (news / headlines / Alexandria articles).

Fetches from the CMC Pro API ``GET /v1/content/latest`` endpoint, which returns
a paginated feed of news headlines and Alexandria articles, each tagged with the
related crypto assets. This replaces Reddit as the default community/news
content source.

Key-gated + fail soft:
- No ``CMC_API_KEY`` -> logs a notice and yields nothing.
- ``/v1/content/latest`` is a **paid-plan** endpoint. A free "Basic" key gets an
  HTTP 401/403 (or a plan error_code); the connector logs a clear notice and
  yields nothing rather than crashing the ingest run.

Docs: https://coinmarketcap.com/api/documentation/v1/  (Content -> Content Latest)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Iterable

import httpx

from ..config import Settings, get_settings
from ..models import SourceType
from .base import RawItem

logger = logging.getLogger(__name__)

BASE_URL = "https://pro-api.coinmarketcap.com"
CONTENT_LATEST = "/v1/content/latest"

# CMC status.error_code values that indicate the plan lacks endpoint access.
_PLAN_ERROR_CODES = {1002, 1006}


def _parse_dt(raw: str | None) -> datetime | None:
    """Parse a CMC ISO-8601 timestamp (…Z) into a UTC datetime, or None."""
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class CMCConnector:
    """Reads recent crypto content from CoinMarketCap's content feed."""

    source_type = SourceType.cmc

    def __init__(
        self,
        settings: Settings | None = None,
        symbols: list[str] | None = None,
        limit: int = 100,
        news_type: str = "all",
        client: httpx.Client | None = None,
    ) -> None:
        self.name = "coinmarketcap"
        self.settings = settings or get_settings()
        self.symbols = symbols or None
        self.limit = limit
        self.news_type = news_type
        self._client = client  # injectable for tests

    def fetch(self, lookback_hours: int) -> Iterable[RawItem]:
        key = self.settings.cmc_api_key
        if not key:
            logger.warning("CMC_API_KEY not set; skipping CoinMarketCap ingestion.")
            return

        cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        params: dict[str, object] = {"limit": self.limit, "news_type": self.news_type}
        if self.symbols:
            params["symbol"] = ",".join(self.symbols)
        headers = {"X-CMC_PRO_API_KEY": key, "Accept": "application/json"}

        client = self._client or httpx.Client(timeout=20.0)
        try:
            resp = client.get(BASE_URL + CONTENT_LATEST, params=params, headers=headers)
        except httpx.HTTPError as exc:
            logger.warning("CMC request failed: %s", exc)
            return
        finally:
            if self._client is None:
                client.close()

        if resp.status_code in (401, 403):
            logger.warning(
                "CMC rejected the key (HTTP %s). The /v1/content/latest endpoint is a "
                "paid-plan feature — your CMC plan may not include it. Skipping.",
                resp.status_code,
            )
            return

        try:
            payload = resp.json()
        except ValueError:
            logger.warning("CMC returned a non-JSON response (HTTP %s). Skipping.", resp.status_code)
            return

        status = payload.get("status") or {}
        error_code = status.get("error_code")
        if error_code:
            note = ""
            if error_code in _PLAN_ERROR_CODES:
                note = " (your plan likely doesn't include the content endpoint)"
            logger.warning(
                "CMC API error %s: %s%s. Skipping.",
                error_code,
                status.get("error_message"),
                note,
            )
            return

        yield from self._to_items(payload.get("data") or [], cutoff, lookback_hours)

    def _to_items(self, data: list[dict], cutoff: datetime, lookback_hours: int) -> Iterable[RawItem]:
        kept = 0
        for item in data:
            released = _parse_dt(item.get("released_at") or item.get("created_at"))
            if released is None or released < cutoff:
                continue
            url = item.get("source_url") or ""
            if not url:
                continue
            title = item.get("title") or ""
            subtitle = item.get("subtitle") or ""
            text = f"{title}\n\n{subtitle}".strip()

            yield RawItem(
                source=SourceType.cmc,
                source_name=item.get("source_name") or "CoinMarketCap",
                url=url,
                text=text,
                published_at=released,
                title=title or None,
                author=None,
                metadata={
                    "cmc_type": item.get("type"),
                    "assets": item.get("assets"),
                    "cover": item.get("cover"),
                },
            )
            kept += 1
        logger.info("CMC: %d content item(s) within %dh window", kept, lookback_hours)
