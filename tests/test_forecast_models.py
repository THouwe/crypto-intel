"""Offline tests for the forecaster zoo, persistence, and training (S9)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from crypto_intel.forecast import features as F
from crypto_intel.forecast import models as M
from crypto_intel.forecast.train import temporal_split, train_models

UTC = timezone.utc


def _synthetic_series(n=500, seed=0):
    rng = np.random.default_rng(seed)
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    idx = np.arange(n)
    # Regime-switching vol so there is a learnable structure in the target.
    vol = np.where((idx // 50) % 2 == 0, 0.004, 0.02)
    steps = rng.standard_normal(n) * vol
    price = 100.0 * np.exp(np.cumsum(steps))
    times = np.array([t0 + timedelta(hours=int(i)) for i in idx], dtype=object)
    return times, price.astype(float)


# --- baseline --------------------------------------------------------------- #

def test_baseline_echoes_rv_lookback():
    names = list(F.FEATURE_NAMES)
    b = M.PersistenceBaseline(names)
    X = np.zeros((3, len(names)))
    X[:, names.index("rv_lookback")] = [0.1, 0.2, 0.3]
    assert np.allclose(b.predict(X, None), [0.1, 0.2, 0.3])


# --- sklearn learns + round-trips ------------------------------------------ #

def test_sklearn_learns_signal(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((400, 4))
    y = np.abs(0.6 * X[:, 0] - 0.4 * X[:, 1]) + 0.02  # positive, learnable
    tr, va, te = temporal_split(400)

    fc = M.SklearnForecaster(seed=0).fit(X[tr], None, y[tr])
    pred = fc.predict(X[te], None)
    ss_res = np.sum((y[te] - pred) ** 2)
    ss_tot = np.sum((y[te] - y[te].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    assert r2 > 0.3  # clearly better than predicting the mean

    # save/load round-trip yields identical predictions.
    d = tmp_path / "sk"
    d.mkdir()
    fc.save(d)
    fc2 = M.SklearnForecaster.load(d)
    assert np.allclose(fc.predict(X[te], None), fc2.predict(X[te], None))


def test_predictions_are_nonnegative():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((200, 4))
    y = np.abs(rng.standard_normal(200)) * 0.01
    fc = M.SklearnForecaster().fit(X[:160], None, y[:160])
    assert np.all(fc.predict(X[160:], None) >= 0.0)


# --- optional backends ------------------------------------------------------ #

def test_xgboost_fits_and_reports_importances(tmp_path):
    pytest.importorskip("xgboost")
    rng = np.random.default_rng(3)
    X = rng.standard_normal((300, 5))
    y = np.abs(0.5 * X[:, 2]) + 0.01
    fc = M.GBMForecaster(backend="xgboost").fit(X[:240], None, y[:240])
    pred = fc.predict(X[240:], None)
    assert pred.shape == (60,) and np.all(np.isfinite(pred))
    imp = fc.feature_importances([f"f{i}" for i in range(5)])
    assert len(imp) == 5

    d = tmp_path / "xgb"
    d.mkdir()
    fc.save(d)
    fc2 = M.GBMForecaster.load(d)
    assert np.allclose(fc.predict(X[240:], None), fc2.predict(X[240:], None))


def test_lstm_fits_and_round_trips(tmp_path):
    pytest.importorskip("torch")
    rng = np.random.default_rng(4)
    seqs = rng.standard_normal((120, 48)) * 0.01
    y = np.sqrt(np.sum(seqs**2, axis=1))  # RV of the sequence — very learnable
    fc = M.SequenceForecaster(seed=0).fit(None, seqs[:100], y[:100], max_epochs=60)
    pred = fc.predict(None, seqs[100:])
    assert pred.shape == (20,) and np.all(np.isfinite(pred)) and np.all(pred >= 0)

    d = tmp_path / "lstm"
    d.mkdir()
    fc.save(d)
    fc2 = M.SequenceForecaster.load(d)
    assert np.allclose(fc.predict(None, seqs[100:]), fc2.predict(None, seqs[100:]), atol=1e-5)


# --- end-to-end training ---------------------------------------------------- #

def test_train_models_end_to_end_and_bundle(tmp_path):
    times, prices = _synthetic_series(600)
    X, y, seqs, names, idx = F.build_supervised(times, prices, 72, 24, 6)
    report = train_models(
        X, y, seqs, names, idx,
        asset="ETH", lookback=72, horizon=24, stride=6,
        models=["baseline", "sklearn"],  # fast, no optional backends
        models_dir=tmp_path,
    )
    assert report["best"] in ("baseline", "sklearn")
    assert "sklearn" in report["models"]
    for split in ("val", "test"):
        for metric in ("mae", "rmse", "r2", "skill", "regime_f1"):
            assert metric in report["models"]["sklearn"][split]

    # The winning bundle is persisted and reloads.
    fc, meta = M.load_bundle(report["bundle_dir"])
    assert meta["asset"] == "ETH"
    assert meta["model_name"] == report["best"]
    assert "regime_thresholds" in meta


def test_train_models_skips_missing_backend(tmp_path, monkeypatch):
    times, prices = _synthetic_series(600)
    X, y, seqs, names, idx = F.build_supervised(times, prices, 72, 24, 6)

    # Force the sklearn model to look "not installed" and confirm the run survives.
    import crypto_intel.forecast.models as models_mod

    def boom(*a, **k):
        raise ImportError("pretend sklearn missing")

    monkeypatch.setattr(models_mod.SklearnForecaster, "fit", boom)
    report = train_models(
        X, y, seqs, names, idx,
        asset="ETH", lookback=72, horizon=24, stride=6,
        models=["baseline", "sklearn"],
    )
    assert "sklearn" in report["skipped"]
    assert report["best"] == "baseline"


def test_temporal_split_is_ordered():
    tr, va, te = temporal_split(100)
    assert tr.stop == 70 and va.start == 70 and va.stop == 85 and te.start == 85
