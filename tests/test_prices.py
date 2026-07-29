"""Tests for price-event detection (synthetic series) + the CoinGecko client."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest

from crypto_intel.config import Settings
from crypto_intel.prices import (
    CoinGeckoClient,
    PriceError,
    build_event,
    detect_event,
)

BASE = datetime(2026, 7, 20, 0, 0, tzinfo=timezone.utc)


def _series(prices: list[float], step_minutes: int = 60):
    """Build a [(ts, price)] series at fixed intervals from BASE."""
    return [(BASE + timedelta(minutes=step_minutes * i), p) for i, p in enumerate(prices)]


def test_build_event_detects_known_drop():
    # Rises to a peak at index 3 (110), then falls to a trough at index 6 (94).
    prices = [100, 105, 108, 110, 104, 98, 94, 96]
    series = _series(prices)
    event = build_event("ETH", "ethereum", series, BASE, series[-1][0])

    assert event.direction == "down"
    # net change 100 -> 96 = -4%
    assert event.pct_change == pytest.approx(-4.0, abs=1e-6)
    # max drawdown is peak 110 -> trough 94 = -14.545%
    assert event.max_drawdown_pct == pytest.approx((110 - 94) / 110 * 100, abs=1e-6)
    # steepest adverse leg runs from the peak (idx 3) to the trough (idx 6)
    assert event.move_start == series[3][0]
    assert event.move_end == series[6][0]


def test_build_event_detects_up_move_and_runup_leg():
    # Dips to a trough at index 2 (95), then climbs to a peak at index 5 (120).
    prices = [100, 97, 95, 105, 112, 120]
    series = _series(prices)
    event = build_event("SOL", "solana", series, BASE, series[-1][0])

    assert event.direction == "up"
    assert event.pct_change == pytest.approx(20.0, abs=1e-6)
    # steepest run-up from trough (idx 2) to peak (idx 5)
    assert event.move_start == series[2][0]
    assert event.move_end == series[5][0]


def test_build_event_flat_within_threshold():
    series = _series([100, 100.2, 99.9, 100.4])
    event = build_event("BTC", "bitcoin", series, BASE, series[-1][0], flat_threshold_pct=1.0)
    assert event.direction == "flat"


def test_build_event_requires_two_points():
    with pytest.raises(PriceError):
        build_event("ETH", "ethereum", _series([100]), BASE, BASE)


def test_build_event_sorts_unordered_series():
    series = _series([100, 108, 110, 90])
    shuffled = [series[2], series[0], series[3], series[1]]
    event = build_event("ETH", "ethereum", shuffled, BASE, series[-1][0])
    assert event.price_start == 100
    assert event.price_end == 90


# --- CoinGecko client ------------------------------------------------------- #

def _price_payload(points_ms_price):
    return {"prices": points_ms_price}


def test_client_parses_and_caches(tmp_settings):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            200,
            json=_price_payload(
                [[int(BASE.timestamp() * 1000), 100.0], [int(BASE.timestamp() * 1000) + 3600_000, 96.0]]
            ),
        )

    client = CoinGeckoClient(
        tmp_settings, client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    start, end = BASE, BASE + timedelta(hours=1)

    first = client.market_chart_range("ethereum", start, end)
    assert len(first) == 2 and first[0][1] == 100.0
    assert calls["n"] == 1

    # Second identical call is served from the on-disk cache (no new HTTP call).
    second = client.market_chart_range("ethereum", start, end)
    assert second == first
    assert calls["n"] == 1


def test_client_retries_once_then_succeeds(tmp_settings, monkeypatch):
    monkeypatch.setattr("crypto_intel.prices.time.sleep", lambda *_: None)
    seq = iter([httpx.Response(503), httpx.Response(200, json=_price_payload(
        [[int(BASE.timestamp() * 1000), 100.0], [int(BASE.timestamp() * 1000) + 3600_000, 96.0]]
    ))])
    client = CoinGeckoClient(
        tmp_settings,
        client=httpx.Client(transport=httpx.MockTransport(lambda r: next(seq))),
    )
    out = client.market_chart_range("ethereum", BASE, BASE + timedelta(hours=1))
    assert len(out) == 2


def test_client_raises_priceerror_on_hard_failure(tmp_settings):
    client = CoinGeckoClient(
        tmp_settings,
        client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404))),
    )
    with pytest.raises(PriceError):
        client.market_chart_range("ethereum", BASE, BASE + timedelta(hours=1))


def test_detect_event_end_to_end_with_injected_client(tmp_settings):
    ms0 = int(BASE.timestamp() * 1000)
    payload = _price_payload(
        [[ms0 + i * 3600_000, p] for i, p in enumerate([100, 110, 104, 94])]
    )
    client = CoinGeckoClient(
        tmp_settings,
        client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))),
    )
    event = detect_event("ETH", hours=24, settings=tmp_settings, client=client, window_end=BASE + timedelta(hours=24))
    assert event.asset == "ETH"
    assert event.coingecko_id == "ethereum"
    assert event.direction == "down"


def test_detect_event_unknown_asset_raises(tmp_settings):
    with pytest.raises(PriceError):
        detect_event("NOTACOIN", hours=24, settings=tmp_settings)
