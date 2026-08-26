"""Offline tests for the DuckDB warehouse + BigQuery row mappers (S10)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from crypto_intel.models import Document, SourceType

UTC = timezone.utc


def _series(n=200, seed=0):
    rng = np.random.default_rng(seed)
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    price = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
    return [(t0 + timedelta(hours=i), float(price[i])) for i in range(n)]


def _doc(ts, asset="ETH", source=SourceType.news, name="CoinDesk", i=0):
    return Document(
        id=f"d{i}", source=source, source_name=name, url=f"https://x/{i}",
        title="t", text="body", author=None, published_at=ts, assets=[asset],
    )


# --- DuckDB warehouse ------------------------------------------------------- #

def test_load_and_grid_roundtrip():
    duckdb = pytest.importorskip("duckdb")  # noqa: F841
    from crypto_intel.warehouse.duck import DuckWarehouse

    series = _series(200)
    with DuckWarehouse(":memory:") as wh:
        n = wh.load_prices("ETH", series)
        assert n == 200
        grid = wh.read_price_grid("ETH")
        # Already hourly → same count, sorted, tz-aware UTC.
        assert len(grid) == 200
        assert grid[0][0].tzinfo is not None
        assert grid == sorted(grid, key=lambda p: p[0])
        assert abs(grid[0][1] - series[0][1]) < 1e-6


def test_build_features_sql_window_functions():
    pytest.importorskip("duckdb")
    from crypto_intel.warehouse.duck import DuckWarehouse

    with DuckWarehouse(":memory:") as wh:
        wh.load_prices("ETH", _series(100))
        rows = wh.build_features("ETH", rolling_hours=12)
        assert rows == 100
        # The materialized SQL feature table is queryable and non-degenerate.
        rv = wh.con.execute(
            "SELECT rolling_rv FROM price_features WHERE asset='ETH' "
            "AND rolling_rv IS NOT NULL ORDER BY h"
        ).fetchall()
        assert len(rv) > 0
        assert all(v[0] >= 0 for v in rv)


def test_news_counts_by_hour_aggregation():
    pytest.importorskip("duckdb")
    from crypto_intel.warehouse.duck import DuckWarehouse, warehouse_news_fn

    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    docs = [
        _doc(t0 + timedelta(hours=1), i=0),
        _doc(t0 + timedelta(hours=1, minutes=30), i=1),  # same hour bucket
        _doc(t0 + timedelta(hours=5), source=SourceType.regulator, name="SEC", i=2),
    ]
    with DuckWarehouse(":memory:") as wh:
        assert wh.load_documents("ETH", docs) == 3
        counts = wh.news_counts_by_hour("ETH")
        # Two docs collapse into one hourly bucket; regulator counted separately.
        assert sum(c for c, _ in counts.values()) == 3
        assert sum(r for _, r in counts.values()) == 1

        fn = warehouse_news_fn(counts)
        feats = fn(t0, t0 + timedelta(hours=6))
        assert feats["news_count"] == 3.0
        assert feats["regulator_count"] == 1.0


def test_load_documents_filters_by_asset():
    pytest.importorskip("duckdb")
    from crypto_intel.warehouse.duck import DuckWarehouse

    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    docs = [_doc(t0, asset="ETH", i=0), _doc(t0, asset="BTC", i=1)]
    with DuckWarehouse(":memory:") as wh:
        assert wh.load_documents("ETH", docs) == 1


def test_stats_reports_window():
    pytest.importorskip("duckdb")
    from crypto_intel.warehouse.duck import DuckWarehouse

    with DuckWarehouse(":memory:") as wh:
        wh.load_prices("ETH", _series(50))
        s = wh.stats("ETH")
        assert s["prices"] == 50 and s["from"] and s["to"]


def test_warehouse_series_trains_a_model():
    """End-to-end: SQL-gridded series out of DuckDB feeds the S9 pipeline."""
    pytest.importorskip("duckdb")
    from crypto_intel.warehouse.duck import DuckWarehouse
    from crypto_intel.forecast.dataset import prepare_dataset
    from crypto_intel.forecast.train import train_models

    with DuckWarehouse(":memory:") as wh:
        wh.load_prices("ETH", _series(600, seed=3))
        series = wh.read_price_grid("ETH")

    X, y, seqs, names, idx, _, _ = prepare_dataset(series, 72, 24, 6)
    report = train_models(
        X, y, seqs, names, idx, asset="ETH",
        lookback=72, horizon=24, stride=6, models=["baseline", "sklearn"],
    )
    assert report["best"] in ("baseline", "sklearn")


# --- BigQuery row mappers (pure; no network) -------------------------------- #

def test_bq_price_rows_pure():
    from crypto_intel.warehouse.bq import price_rows

    rows = price_rows("eth", _series(3))
    assert len(rows) == 3
    assert rows[0]["asset"] == "ETH"
    assert isinstance(rows[0]["price"], float)
    assert rows[0]["ts"].endswith("+00:00")


def test_bq_document_rows_filter_and_shape():
    from crypto_intel.warehouse.bq import document_rows

    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    docs = [_doc(t0, asset="ETH", i=0), _doc(t0, asset="BTC", i=1)]
    rows = document_rows("ETH", docs)
    assert len(rows) == 1
    assert rows[0]["source"] == "news"
    assert set(rows[0]) == {"asset", "ts", "source", "source_name", "doc_id"}
