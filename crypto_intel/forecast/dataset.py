"""Dataset assembly for forecasting: history fetch, CSV loader, news features.

This is the *only* forecast module that touches the network or disk stores. It
turns a price history (from CoinGecko or an offline CSV) into the gridded arrays
:func:`crypto_intel.forecast.features.build_supervised` consumes, and optionally
supplies the news-feature closure.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from ..config import Settings, get_settings
from . import features as F

UTC = timezone.utc


# --------------------------------------------------------------------------- #
# Price history                                                               #
# --------------------------------------------------------------------------- #

def fetch_price_history(
    asset: str,
    days: int,
    settings: Settings | None = None,
    client=None,
    end: datetime | None = None,
) -> list[tuple[datetime, float]]:
    """Fetch ``days`` of price history for ``asset`` from CoinGecko.

    Reuses the RAG core's :class:`crypto_intel.prices.CoinGeckoClient` (timeout,
    retry, on-disk cache). Free-tier granularity: hourly within ~90 days, daily
    beyond — the returned series is later resampled onto an hourly grid.
    """
    from ..prices import CoinGeckoClient, _coingecko_id

    settings = settings or get_settings()
    client = client or CoinGeckoClient(settings)
    coingecko_id = _coingecko_id(asset, settings)
    end = end or datetime.now(UTC)
    start = end - timedelta(days=days)
    return client.market_chart_range(coingecko_id, start, end)


def load_series_csv(path: Path) -> list[tuple[datetime, float]]:
    """Load a ``timestamp,price`` CSV into a price series.

    ``timestamp`` may be an ISO-8601 string or an epoch (seconds). This is the
    reproducible / offline training + test path (no CoinGecko, no key).
    """
    path = Path(path)
    out: list[tuple[datetime, float]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row or len(row) < 2:
                continue
            ts_raw, price_raw = row[0].strip(), row[1].strip()
            if ts_raw.lower() in ("timestamp", "time", "date", "ts"):
                continue  # header
            try:
                price = float(price_raw)
            except ValueError:
                continue
            out.append((_parse_ts(ts_raw), price))
    return out


def _parse_ts(raw: str) -> datetime:
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        dt = datetime.fromtimestamp(float(raw), tz=UTC)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Assembly                                                                     #
# --------------------------------------------------------------------------- #

def prepare_dataset(
    series: list[tuple[datetime, float]],
    lookback: int,
    horizon: int,
    stride: int,
    news_fn=None,
):
    """Resample a raw series to hourly then build the supervised windows.

    Returns the tuple from :func:`features.build_supervised`
    ``(X, y, seqs, feature_names, index_ts)`` plus the gridded ``(times, prices)``.
    """
    times, prices = F.resample_hourly(series)
    X, y, seqs, names, idx = F.build_supervised(
        times, prices, lookback, horizon, stride, news_fn=news_fn
    )
    return X, y, seqs, names, idx, times, prices


# --------------------------------------------------------------------------- #
# Optional news features                                                      #
# --------------------------------------------------------------------------- #

def news_feature_fn(asset: str, settings: Settings | None = None):
    """Build a ``news_fn(start, end) -> dict`` from the persisted document store.

    Reads ``documents.jsonl`` once (via the pipeline's :func:`iter_documents`) and
    returns a closure counting asset-tagged docs in each window. Returns ``None``
    when there are no documents, so the caller falls back to a tabular-only model.
    """
    from ..pipeline import iter_documents

    settings = settings or get_settings()
    asset_u = asset.upper()
    docs = []
    for doc in iter_documents(settings.documents_file):
        if asset_u in [a.upper() for a in doc.assets]:
            pub = doc.published_at
            pub = pub if pub.tzinfo else pub.replace(tzinfo=UTC)
            is_reg = getattr(doc.source, "value", str(doc.source)) == "regulator"
            docs.append((pub, is_reg))
    if not docs:
        return None
    docs.sort(key=lambda d: d[0])
    times_arr = np.array([d[0].timestamp() for d in docs], dtype=float)
    reg_arr = np.array([1.0 if d[1] else 0.0 for d in docs], dtype=float)

    def _fn(start: datetime, end: datetime) -> dict:
        s, e = start.timestamp(), end.timestamp()
        lo = np.searchsorted(times_arr, s, side="left")
        hi = np.searchsorted(times_arr, e, side="right")
        in_window = slice(lo, hi)
        count = hi - lo
        # "recent" = last horizon fraction of the window is approximated by the
        # latest quarter of its span; kept simple and monotone.
        recent_cut = e - (e - s) * 0.25
        recent_lo = np.searchsorted(times_arr, recent_cut, side="left")
        count_recent = hi - max(recent_lo, lo)
        last_ts = times_arr[hi - 1] if count > 0 else None
        recency_hours = (e - last_ts) / 3600.0 if last_ts is not None else 0.0
        return {
            "news_count": float(count),
            "news_count_recent": float(count_recent),
            "news_recency_hours": float(recency_hours),
            "regulator_count": float(reg_arr[in_window].sum()),
        }

    return _fn
