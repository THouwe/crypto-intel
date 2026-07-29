"""CoinGecko price client + price-event detection.

Two concerns, kept separate so the analysis is testable without network:

- :class:`CoinGeckoClient` fetches a price series for a window from CoinGecko's
  free ``market_chart/range`` endpoint, with a timeout, one retry, and a simple
  on-disk response cache (to stay gentle on the free tier).
- :func:`build_event` is a pure function turning a price series into a
  :class:`PriceEvent` (window %, max drawdown, and the steepest-move sub-window
  that drives retrieval). :func:`detect_event` wires fetch + analysis together.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta, timezone

import httpx

from .config import Settings, get_settings
from .models import PriceEvent
from .normalize import load_assets

logger = logging.getLogger(__name__)

# (timestamp, price) sample.
PricePoint = tuple[datetime, float]

MARKET_CHART_RANGE = "/coins/{id}/market_chart/range"


class PriceError(RuntimeError):
    """Raised when a price series can't be fetched or is too sparse to analyse."""


def _coingecko_id(asset: str, settings: Settings) -> str:
    assets_cfg = load_assets(settings)
    entry = assets_cfg.get(asset.upper())
    if not entry or not entry.get("coingecko_id"):
        raise PriceError(
            f"Unknown asset {asset!r}. Add it to assets.yaml with a coingecko_id."
        )
    return entry["coingecko_id"]


class CoinGeckoClient:
    """Minimal CoinGecko client: windowed price series, cached + retried."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client  # injectable for tests

    # -- on-disk cache ---------------------------------------------------------

    def _cache_path(self, key: str):
        cache_dir = self.settings.price_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{key}.json"

    def _cache_get(self, key: str) -> list[PricePoint] | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        age = time.time() - blob.get("fetched_at", 0)
        if age > self.settings.price_cache_ttl_seconds:
            return None
        return [
            (datetime.fromtimestamp(ts, tz=timezone.utc), float(price))
            for ts, price in blob.get("series", [])
        ]

    def _cache_put(self, key: str, series: list[PricePoint]) -> None:
        path = self._cache_path(key)
        blob = {
            "fetched_at": time.time(),
            "series": [(ts.timestamp(), price) for ts, price in series],
        }
        try:
            path.write_text(json.dumps(blob), encoding="utf-8")
        except OSError as exc:  # caching is best-effort
            logger.debug("Could not write price cache %s: %s", path, exc)

    # -- fetch -----------------------------------------------------------------

    def market_chart_range(
        self, coingecko_id: str, start: datetime, end: datetime
    ) -> list[PricePoint]:
        """Return [(ts, price_usd)] for ``coingecko_id`` within [start, end]."""
        # Bucket the window to the minute so near-identical calls share a cache hit.
        from_ts = int(start.timestamp())
        to_ts = int(end.timestamp())
        key = hashlib.sha1(
            f"{coingecko_id}|{from_ts // 60}|{to_ts // 60}".encode()
        ).hexdigest()

        cached = self._cache_get(key)
        if cached is not None:
            logger.debug("Price cache hit for %s", coingecko_id)
            return cached

        url = self.settings.coingecko_base_url + MARKET_CHART_RANGE.format(id=coingecko_id)
        params = {"vs_currency": "usd", "from": from_ts, "to": to_ts}
        headers = {"Accept": "application/json"}
        if self.settings.coingecko_api_key:
            headers["x-cg-demo-api-key"] = self.settings.coingecko_api_key

        payload = self._get_json(url, params, headers)
        series = [
            (datetime.fromtimestamp(ms / 1000, tz=timezone.utc), float(price))
            for ms, price in payload.get("prices", [])
        ]
        if series:
            self._cache_put(key, series)
        return series

    def _get_json(self, url: str, params: dict, headers: dict) -> dict:
        """GET with a timeout and a single retry on transient failures."""
        client = self._client or httpx.Client(timeout=20.0)
        try:
            last_exc: Exception | None = None
            for attempt in (1, 2):
                try:
                    resp = client.get(url, params=params, headers=headers)
                except httpx.HTTPError as exc:
                    last_exc = exc
                    logger.warning("CoinGecko request error (attempt %d): %s", attempt, exc)
                else:
                    if resp.status_code == 200:
                        try:
                            return resp.json()
                        except ValueError as exc:
                            raise PriceError("CoinGecko returned invalid JSON") from exc
                    if resp.status_code in (429, 500, 502, 503, 504) and attempt == 1:
                        logger.warning(
                            "CoinGecko HTTP %s; retrying once ...", resp.status_code
                        )
                    else:
                        raise PriceError(
                            f"CoinGecko HTTP {resp.status_code}: {resp.text[:200]}"
                        )
                if attempt == 1:
                    time.sleep(1.0)
            raise PriceError(f"CoinGecko request failed: {last_exc}")
        finally:
            if self._client is None:
                client.close()


# --------------------------------------------------------------------------- #
# Pure analysis                                                               #
# --------------------------------------------------------------------------- #

def _max_drawdown_pct(series: list[PricePoint]) -> float:
    """Largest peak-to-trough decline over the series, as a positive percent."""
    peak = series[0][1]
    worst = 0.0
    for _, price in series:
        if price > peak:
            peak = price
        if peak > 0:
            dd = (price - peak) / peak * 100.0
            worst = min(worst, dd)
    return abs(worst)


def _steepest_leg(
    series: list[PricePoint], direction: str
) -> tuple[datetime, datetime]:
    """Endpoints of the steepest contiguous move in the net direction.

    For a down/flat move: the max peak->trough drawdown leg.
    For an up move: the max trough->peak run-up leg.
    """
    first_t = series[0][0]
    last_t = series[-1][0]
    start, end = first_t, last_t

    if direction == "up":
        best = 0.0
        min_p, min_t = series[0][1], series[0][0]
        for ts, price in series:
            if price < min_p:
                min_p, min_t = price, ts
            if min_p > 0:
                run_up = (price - min_p) / min_p * 100.0
                if run_up > best:
                    best, start, end = run_up, min_t, ts
    else:
        best = 0.0
        max_p, max_t = series[0][1], series[0][0]
        for ts, price in series:
            if price > max_p:
                max_p, max_t = price, ts
            if max_p > 0:
                draw = (price - max_p) / max_p * 100.0
                if draw < best:
                    best, start, end = draw, max_t, ts
    return start, end


def build_event(
    asset: str,
    coingecko_id: str,
    series: list[PricePoint],
    window_start: datetime,
    window_end: datetime,
    flat_threshold_pct: float = 1.0,
) -> PriceEvent:
    """Turn a price series into a :class:`PriceEvent` (pure, no network)."""
    if len(series) < 2:
        raise PriceError(
            f"Not enough price data for {asset} in the window "
            f"({len(series)} point(s)); try a wider --hours."
        )
    series = sorted(series, key=lambda p: p[0])
    price_start = series[0][1]
    price_end = series[-1][1]
    if price_start <= 0:
        raise PriceError(f"Invalid starting price for {asset}: {price_start}")

    pct_change = (price_end - price_start) / price_start * 100.0
    if pct_change <= -flat_threshold_pct:
        direction = "down"
    elif pct_change >= flat_threshold_pct:
        direction = "up"
    else:
        direction = "flat"

    move_start, move_end = _steepest_leg(series, direction)

    return PriceEvent(
        asset=asset.upper(),
        coingecko_id=coingecko_id,
        window_start=window_start,
        window_end=window_end,
        price_start=price_start,
        price_end=price_end,
        pct_change=pct_change,
        max_drawdown_pct=_max_drawdown_pct(series),
        move_start=move_start,
        move_end=move_end,
        direction=direction,
    )


def detect_event(
    asset: str,
    hours: int,
    settings: Settings | None = None,
    client: CoinGeckoClient | None = None,
    window_end: datetime | None = None,
) -> PriceEvent:
    """Fetch the price window for ``asset`` and detect the move."""
    settings = settings or get_settings()
    coingecko_id = _coingecko_id(asset, settings)
    window_end = window_end or datetime.now(timezone.utc)
    window_start = window_end - timedelta(hours=hours)

    client = client or CoinGeckoClient(settings)
    series = client.market_chart_range(coingecko_id, window_start, window_end)
    return build_event(
        asset,
        coingecko_id,
        series,
        window_start,
        window_end,
        flat_threshold_pct=settings.flat_threshold_pct,
    )
