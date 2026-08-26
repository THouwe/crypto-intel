"""Offline tests for the pure feature-engineering core (S9)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from crypto_intel.forecast import features as F

UTC = timezone.utc


def _synthetic_series(n=400, seed=0):
    """A deterministic hourly series: trend + sinusoid + noise, all positive."""
    rng = np.random.default_rng(seed)
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    idx = np.arange(n)
    base = 100.0 + 0.02 * idx + 5.0 * np.sin(idx / 12.0)
    price = base * (1.0 + 0.01 * rng.standard_normal(n))
    times = np.array([t0 + timedelta(hours=int(i)) for i in idx], dtype=object)
    return times, price.astype(float)


# --- core math -------------------------------------------------------------- #

def test_log_returns_basic():
    prices = np.array([100.0, 110.0, 121.0])
    r = F.log_returns(prices)
    assert r.shape == (2,)
    assert math.isclose(r[0], math.log(1.1), rel_tol=1e-12)
    assert math.isclose(r[1], math.log(1.1), rel_tol=1e-12)


def test_log_returns_too_short():
    assert F.log_returns(np.array([100.0])).size == 0
    assert F.log_returns(np.array([])).size == 0


def test_realized_vol_matches_manual():
    r = np.array([0.01, -0.02, 0.015])
    assert math.isclose(F.realized_vol(r), math.sqrt(0.01**2 + 0.02**2 + 0.015**2))
    assert F.realized_vol(np.array([])) == 0.0


def test_realized_vol_zero_for_flat_series():
    assert F.realized_vol(F.log_returns(np.full(20, 50.0))) == 0.0


def test_annualize_scales_up():
    assert F.annualize_vol(0.1, 24) > 0.1
    assert F.annualize_vol(0.1, 0) == 0.0


def test_max_drawdown_pct():
    # 100 -> 80 is a 20% drawdown even though it recovers after.
    prices = np.array([100.0, 120.0, 96.0, 130.0])
    assert math.isclose(F.max_drawdown_pct(prices), 20.0, rel_tol=1e-9)
    assert F.max_drawdown_pct(np.full(5, 10.0)) == 0.0


# --- window features -------------------------------------------------------- #

def test_window_features_keys_and_types():
    _, prices = _synthetic_series(120)
    feats = F.window_features(prices[:73], horizon_hours=24)
    for name in F.FEATURE_NAMES:
        assert name in feats
        assert isinstance(feats[name], float)
        assert not math.isnan(feats[name])


def test_window_features_news_optional():
    _, prices = _synthetic_series(120)
    without = F.window_features(prices[:73], 24)
    assert not any(n in without for n in F.NEWS_FEATURE_NAMES)
    with_news = F.window_features(prices[:73], 24, news={"news_count": 3})
    assert with_news["news_count"] == 3.0
    assert with_news["regulator_count"] == 0.0  # default fill


def test_flat_window_is_finite_and_zero_vol():
    feats = F.window_features(np.full(73, 42.0), 24)
    assert feats["rv_lookback"] == 0.0
    assert feats["std_return"] == 0.0
    assert all(math.isfinite(v) for v in feats.values())


# --- supervised windowing --------------------------------------------------- #

def test_build_supervised_shapes():
    times, prices = _synthetic_series(400)
    L, H, S = 72, 24, 6
    X, y, seqs, names, idx = F.build_supervised(times, prices, L, H, S)
    n = X.shape[0]
    assert n > 0
    assert y.shape == (n,)
    assert seqs.shape == (n, L)          # one return per lookback hour
    assert X.shape == (n, len(names))
    assert names == F.FEATURE_NAMES      # no news_fn → tabular only
    assert idx.shape == (n,)


def test_build_supervised_anchor_count():
    times, prices = _synthetic_series(400)
    L, H, S = 72, 24, 6
    X, *_ = F.build_supervised(times, prices, L, H, S)
    expected = len(range(L, (prices.size - 1 - H) + 1, S))
    assert X.shape[0] == expected


def test_build_supervised_news_columns():
    times, prices = _synthetic_series(300)
    calls = []

    def news_fn(start, end):
        calls.append((start, end))
        return {"news_count": 1.0, "regulator_count": 2.0}

    X, y, seqs, names, idx = F.build_supervised(times, prices, 72, 24, 6, news_fn)
    assert names == F.FEATURE_NAMES + F.NEWS_FEATURE_NAMES
    assert X.shape[1] == len(names)
    assert len(calls) == X.shape[0]
    # regulator_count column is the last one and equals 2 everywhere.
    assert np.allclose(X[:, names.index("regulator_count")], 2.0)


def test_no_label_leakage():
    """The target must depend only on the future; mutating post-anchor prices
    changes y but not the anchor's features."""
    times, prices = _synthetic_series(400)
    L, H, S = 72, 24, 6
    X0, y0, _, _, idx0 = F.build_supervised(times, prices, L, H, S)

    # Pick a middle anchor and perturb prices strictly after it.
    a = X0.shape[0] // 2
    anchor_i = list(range(L, (prices.size - 1 - H) + 1, S))[a]
    perturbed = prices.copy()
    perturbed[anchor_i + 1:] *= 1.5   # only the future moves

    X1, y1, _, _, idx1 = F.build_supervised(times, perturbed, L, H, S)

    # Features at the anchor are unchanged (they use only prices <= anchor)...
    assert np.allclose(X0[a], X1[a])
    # ...but the target at the anchor did change (it uses future prices).
    assert not math.isclose(y0[a], y1[a], rel_tol=1e-9)


def test_regime_thresholds_and_classify():
    y = np.linspace(0.0, 1.0, 101)
    th = F.regime_thresholds(y, 33.0, 66.0)
    assert th["calm_max"] < th["turbulent_min"]
    assert F.classify_regime(0.0, th) == "calm"
    assert F.classify_regime(1.0, th) == "turbulent"
    assert F.classify_regime(0.5, th) == "normal"


# --- resampling ------------------------------------------------------------- #

def test_resample_hourly_regular_passthrough():
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    series = [(t0 + timedelta(hours=i), 100.0 + i) for i in range(5)]
    times, prices = F.resample_hourly(series)
    assert prices.shape == (5,)
    assert math.isclose(prices[0], 100.0)
    assert math.isclose(prices[-1], 104.0)


def test_resample_hourly_interpolates_gap():
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    # 100 at h0, 104 at h4 — the missing hours should interpolate linearly.
    series = [(t0, 100.0), (t0 + timedelta(hours=4), 104.0)]
    times, prices = F.resample_hourly(series)
    assert prices.shape == (5,)
    assert math.isclose(prices[2], 102.0, rel_tol=1e-9)


def test_resample_empty():
    times, prices = F.resample_hourly([])
    assert times.size == 0 and prices.size == 0
