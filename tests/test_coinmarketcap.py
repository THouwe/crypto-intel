"""Tests for the CoinMarketCap content connector + registry wiring."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest

from crypto_intel.config import Settings
from crypto_intel.ingest.coinmarketcap import CMCConnector
from crypto_intel.ingest.registry import build_connectors
from crypto_intel.ingest.reddit import RedditConnector
from crypto_intel.models import SourceType


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _content_payload(items, error_code=0, error_message=None):
    return {
        "status": {"error_code": error_code, "error_message": error_message},
        "data": items,
    }


def _recent_iso(hours_ago: float) -> str:
    dt = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def test_cmc_skips_without_key(caplog):
    settings = Settings(cmc_api_key=None)
    conn = CMCConnector(settings, client=_client(lambda r: httpx.Response(200)))
    with caplog.at_level("WARNING"):
        items = list(conn.fetch(48))
    assert items == []
    assert "CMC_API_KEY not set" in caplog.text


def test_cmc_parses_and_windows_content():
    payload = _content_payload(
        [
            {
                "title": "Ethereum slides on ETF outflows",
                "subtitle": "Analysts weigh in",
                "source_name": "CoinDesk",
                "source_url": "https://coindesk.com/eth",
                "released_at": _recent_iso(2),
                "type": "news",
                "assets": ["ETH"],
            },
            {   # too old — outside the 48h window
                "title": "Old news",
                "source_url": "https://x/old",
                "released_at": _recent_iso(200),
            },
            {   # missing url — skipped
                "title": "No link",
                "released_at": _recent_iso(1),
            },
        ]
    )
    conn = CMCConnector(
        Settings(cmc_api_key="key"),
        client=_client(lambda r: httpx.Response(200, json=payload)),
    )
    items = list(conn.fetch(48))

    assert len(items) == 1
    it = items[0]
    assert it.source is SourceType.cmc
    assert it.source_name == "CoinDesk"
    assert it.url == "https://coindesk.com/eth"
    assert "Ethereum slides" in it.text and "Analysts weigh in" in it.text
    assert it.metadata["assets"] == ["ETH"]


def test_cmc_sends_auth_header_and_symbol_filter():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["key"] = request.headers.get("X-CMC_PRO_API_KEY")
        captured["url"] = str(request.url)
        return httpx.Response(200, json=_content_payload([]))

    conn = CMCConnector(
        Settings(cmc_api_key="secret-key"),
        symbols=["ETH", "BTC"],
        client=_client(handler),
    )
    list(conn.fetch(48))
    assert captured["key"] == "secret-key"
    assert "symbol=ETH%2CBTC" in captured["url"] or "symbol=ETH,BTC" in captured["url"]


@pytest.mark.parametrize("status_code", [401, 403])
def test_cmc_fails_soft_on_auth_error(status_code, caplog):
    conn = CMCConnector(
        Settings(cmc_api_key="free-key"),
        client=_client(lambda r: httpx.Response(status_code)),
    )
    with caplog.at_level("WARNING"):
        items = list(conn.fetch(48))
    assert items == []
    assert "paid-plan" in caplog.text


def test_cmc_fails_soft_on_plan_error_code(caplog):
    payload = _content_payload([], error_code=1006, error_message="Not authorized")
    conn = CMCConnector(
        Settings(cmc_api_key="free-key"),
        client=_client(lambda r: httpx.Response(200, json=payload)),
    )
    with caplog.at_level("WARNING"):
        items = list(conn.fetch(48))
    assert items == []
    assert "plan likely doesn't include" in caplog.text


def test_cmc_fails_soft_on_network_error():
    def handler(request):
        raise httpx.ConnectError("no network")

    conn = CMCConnector(Settings(cmc_api_key="key"), client=_client(handler))
    assert list(conn.fetch(48)) == []  # no raise


# --- registry wiring -------------------------------------------------------- #

def test_default_set_includes_cmc_excludes_reddit(tmp_settings):
    conns = build_connectors(None, settings=tmp_settings)
    types = [type(c) for c in conns]
    assert CMCConnector in types
    assert RedditConnector not in types


def test_reddit_is_opt_in(tmp_settings):
    conns = build_connectors({"reddit"}, settings=tmp_settings)
    assert any(isinstance(c, RedditConnector) for c in conns)
    assert not any(isinstance(c, CMCConnector) for c in conns)


def test_cmc_connector_seeded_with_asset_symbols(tmp_settings):
    conns = build_connectors({"cmc"}, settings=tmp_settings)
    cmc = next(c for c in conns if isinstance(c, CMCConnector))
    assert "ETH" in cmc.symbols and "BTC" in cmc.symbols


def test_all_sources_includes_reddit_and_cmc(tmp_settings):
    # Mirrors the CLI `--all` path: the explicit full source set builds every
    # connector, including opt-in Reddit and key-gated CMC.
    from crypto_intel.ingest.registry import VALID_SOURCES

    conns = build_connectors(set(VALID_SOURCES), settings=tmp_settings)
    types = [type(c) for c in conns]
    assert RedditConnector in types
    assert CMCConnector in types
