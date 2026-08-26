"""Pure feature engineering for volatility forecasting.

No network, no heavy ML deps — numpy only. Everything here is deterministic and
offline-testable, mirroring the ``prices.build_event`` split in the RAG core.

Conventions (see ``docs/ML_ROADMAP.md`` § 4.1):

- Work on a regular **hourly** grid. :func:`resample_hourly` turns an irregular
  ``[(ts, price), ...]`` series into one.
- ``r_t = ln(p_t / p_{t-1})`` are hourly log returns.
- **Realized volatility** over a set of returns is ``RV = sqrt(sum(r_i**2))``.
- The supervised target ``y`` at an anchor ``t`` is the RV over the *next* horizon
  window ``(t, t+H]``; features use *only* prices in the lookback ``(t-L, t]``.
  The strict separation is what keeps the problem leakage-free.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

# (timestamp, price) sample — matches crypto_intel.prices.PricePoint.
PricePoint = tuple[datetime, float]

HOUR = timedelta(hours=1)

# Stable tabular feature order. `build_supervised` emits columns in exactly this
# order (optionally followed by NEWS_FEATURE_NAMES when a news_fn is supplied).
FEATURE_NAMES: list[str] = [
    "rv_lookback",
    "rv_recent",
    "vol_of_vol",
    "mean_return",
    "std_return",
    "downside_vol",
    "mean_abs_return",
    "return_skew",
    "return_kurt",
    "acf1",
    "max_drawdown_pct",
    "momentum",
    "range_pct",
    "last_return",
]

NEWS_FEATURE_NAMES: list[str] = [
    "news_count",
    "news_count_recent",
    "news_recency_hours",
    "regulator_count",
]

# Hours in a (crypto) trading year — markets run 24/7.
_HOURS_PER_YEAR = 24 * 365


# --------------------------------------------------------------------------- #
# Core series math                                                            #
# --------------------------------------------------------------------------- #

def log_returns(prices: np.ndarray) -> np.ndarray:
    """Hourly log returns ``ln(p_t / p_{t-1})`` for a 1-D price array."""
    prices = np.asarray(prices, dtype=float)
    if prices.size < 2:
        return np.empty(0, dtype=float)
    return np.diff(np.log(prices))


def realized_vol(returns: np.ndarray) -> float:
    """Realized volatility ``sqrt(sum(r**2))`` over a set of returns."""
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return 0.0
    return float(np.sqrt(np.sum(returns**2)))


def annualize_vol(rv: float, horizon_hours: int) -> float:
    """Scale an ``horizon_hours``-window RV to an annualized figure."""
    if horizon_hours <= 0:
        return 0.0
    return float(rv) * float(np.sqrt(_HOURS_PER_YEAR / horizon_hours))


def max_drawdown_pct(prices: np.ndarray) -> float:
    """Largest peak-to-trough decline over the array, as a positive percent.

    Mirrors :func:`crypto_intel.prices._max_drawdown_pct` but on a numpy array.
    """
    prices = np.asarray(prices, dtype=float)
    if prices.size == 0:
        return 0.0
    peak = prices[0]
    worst = 0.0
    for p in prices:
        if p > peak:
            peak = p
        if peak > 0:
            worst = min(worst, (p - peak) / peak * 100.0)
    return abs(worst)


def _acf1(returns: np.ndarray) -> float:
    """Lag-1 autocorrelation of returns; 0 when undefined (const / <2 points)."""
    r = np.asarray(returns, dtype=float)
    if r.size < 3:
        return 0.0
    a, b = r[:-1], r[1:]
    sa, sb = a.std(), b.std()
    if sa == 0.0 or sb == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _skew(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return 0.0
    s = x.std()
    if s == 0.0:
        return 0.0
    return float(np.mean(((x - x.mean()) / s) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis (0 for a normal)."""
    x = np.asarray(x, dtype=float)
    if x.size < 4:
        return 0.0
    s = x.std()
    if s == 0.0:
        return 0.0
    return float(np.mean(((x - x.mean()) / s) ** 4) - 3.0)


def _vol_of_vol(returns: np.ndarray, sub: int = 6) -> float:
    """Std of realized vols computed over consecutive ``sub``-length sub-windows."""
    r = np.asarray(returns, dtype=float)
    if r.size < 2 * sub:
        return 0.0
    n_sub = r.size // sub
    rvs = [realized_vol(r[i * sub:(i + 1) * sub]) for i in range(n_sub)]
    return float(np.std(rvs)) if len(rvs) > 1 else 0.0


# --------------------------------------------------------------------------- #
# Windowing                                                                   #
# --------------------------------------------------------------------------- #

def resample_hourly(series: list[PricePoint]) -> tuple[np.ndarray, np.ndarray]:
    """Resample an irregular ``[(ts, price)]`` series onto a regular hourly grid.

    Returns ``(times, prices)`` where ``times`` is an object array of tz-aware
    ``datetime`` on the hour and ``prices`` is a float array, linearly
    interpolated. A series already on an hourly grid passes through unchanged.
    """
    if not series:
        return np.empty(0, dtype=object), np.empty(0, dtype=float)
    ordered = sorted(series, key=lambda p: p[0])
    t0 = ordered[0][0].astimezone(timezone.utc).replace(
        minute=0, second=0, microsecond=0
    )
    t1 = ordered[-1][0].astimezone(timezone.utc)
    src_epoch = np.array([p[0].timestamp() for p in ordered], dtype=float)
    src_price = np.array([p[1] for p in ordered], dtype=float)

    n = int((t1 - t0) // HOUR) + 1
    grid_times = np.empty(n, dtype=object)
    grid_epoch = np.empty(n, dtype=float)
    for i in range(n):
        gt = t0 + i * HOUR
        grid_times[i] = gt
        grid_epoch[i] = gt.timestamp()
    grid_price = np.interp(grid_epoch, src_epoch, src_price)
    return grid_times, grid_price


def window_features(
    prices: np.ndarray, horizon_hours: int, news: dict | None = None
) -> dict[str, float]:
    """Tabular feature dict for one lookback window of hourly prices.

    ``prices`` is the lookback window (``L+1`` prices → ``L`` returns). ``news``,
    if given, supplies the exogenous :data:`NEWS_FEATURE_NAMES` (defaulting to 0).
    """
    prices = np.asarray(prices, dtype=float)
    r = log_returns(prices)
    recent = r[-horizon_hours:] if horizon_hours > 0 else r
    neg = r[r < 0]
    p_first, p_last = (prices[0], prices[-1]) if prices.size else (0.0, 0.0)
    p_mean = float(prices.mean()) if prices.size else 0.0

    feats: dict[str, float] = {
        "rv_lookback": realized_vol(r),
        "rv_recent": realized_vol(recent),
        "vol_of_vol": _vol_of_vol(r),
        "mean_return": float(r.mean()) if r.size else 0.0,
        "std_return": float(r.std()) if r.size else 0.0,
        "downside_vol": float(neg.std()) if neg.size else 0.0,
        "mean_abs_return": float(np.mean(np.abs(r))) if r.size else 0.0,
        "return_skew": _skew(r),
        "return_kurt": _kurtosis(r),
        "acf1": _acf1(r),
        "max_drawdown_pct": max_drawdown_pct(prices),
        "momentum": (p_last / p_first - 1.0) if p_first > 0 else 0.0,
        "range_pct": (
            (float(prices.max()) - float(prices.min())) / p_mean * 100.0
            if p_mean > 0 else 0.0
        ),
        "last_return": float(r[-1]) if r.size else 0.0,
    }
    if news is not None:
        for name in NEWS_FEATURE_NAMES:
            feats[name] = float(news.get(name, 0.0))
    return feats


def feature_vector(feats: dict[str, float], names: list[str]) -> np.ndarray:
    """Order a feature dict into a vector following ``names``."""
    return np.array([feats.get(n, 0.0) for n in names], dtype=float)


def build_supervised(
    times: np.ndarray,
    prices: np.ndarray,
    lookback: int,
    horizon: int,
    stride: int,
    news_fn=None,
):
    """Slide lookback/horizon windows over an hourly grid into a supervised set.

    Parameters mirror ``docs/ML_ROADMAP.md`` § 4.1. ``times``/``prices`` must be a
    regular hourly grid (see :func:`resample_hourly`). ``news_fn``, if given, is
    called ``news_fn(start, end) -> dict`` for each window's ``(t-L, t]`` span.

    Returns ``(X, y, seqs, feature_names, index_ts)``:

    - ``X``          — ``(n, n_features)`` tabular features,
    - ``y``          — ``(n,)`` next-window realized vol (the target),
    - ``seqs``       — ``(n, lookback)`` raw return sequences (for the LSTM),
    - ``feature_names`` — column order of ``X``,
    - ``index_ts``   — ``(n,)`` anchor timestamps (``as_of`` = end of lookback).

    Anchors ``i`` run from ``lookback`` to ``len-1-horizon`` in steps of ``stride``,
    so every window has a full lookback before and a full horizon after — no leak.
    """
    prices = np.asarray(prices, dtype=float)
    n_total = prices.size
    feature_names = list(FEATURE_NAMES) + (
        list(NEWS_FEATURE_NAMES) if news_fn is not None else []
    )

    X_rows: list[np.ndarray] = []
    y_vals: list[float] = []
    seq_rows: list[np.ndarray] = []
    idx_ts: list = []

    last_anchor = n_total - 1 - horizon
    for i in range(lookback, last_anchor + 1, stride):
        look_prices = prices[i - lookback:i + 1]          # L+1 prices, ≤ t
        fut_returns = log_returns(prices[i:i + horizon + 1])  # H returns, > t
        news = None
        if news_fn is not None:
            news = news_fn(times[i - lookback], times[i])
        feats = window_features(look_prices, horizon, news=news)
        X_rows.append(feature_vector(feats, feature_names))
        y_vals.append(realized_vol(fut_returns))
        seq_rows.append(log_returns(look_prices))          # length == lookback
        idx_ts.append(times[i])

    X = np.array(X_rows, dtype=float) if X_rows else np.empty((0, len(feature_names)))
    y = np.array(y_vals, dtype=float)
    seqs = np.array(seq_rows, dtype=float) if seq_rows else np.empty((0, lookback))
    index_ts = np.array(idx_ts, dtype=object)
    return X, y, seqs, feature_names, index_ts


def regime_thresholds(y: np.ndarray, low_pct: float, high_pct: float) -> dict:
    """Percentile cut points that bucket vol into calm / normal / turbulent."""
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return {"calm_max": 0.0, "turbulent_min": 0.0}
    return {
        "calm_max": float(np.percentile(y, low_pct)),
        "turbulent_min": float(np.percentile(y, high_pct)),
    }


def classify_regime(vol: float, thresholds: dict) -> str:
    """Map a volatility value to a regime label using stored thresholds."""
    if vol <= thresholds.get("calm_max", 0.0):
        return "calm"
    if vol >= thresholds.get("turbulent_min", float("inf")):
        return "turbulent"
    return "normal"
