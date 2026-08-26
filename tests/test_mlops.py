"""Offline tests for the S11 MLOps loop: reference, tracking, monitoring, serving."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from crypto_intel.config import Settings
from crypto_intel.forecast import features as F
from crypto_intel.forecast.train import train_models

UTC = timezone.utc


def _series(n=700, seed=0):
    rng = np.random.default_rng(seed)
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    idx = np.arange(n)
    vol = np.where((idx // 40) % 2 == 0, 0.004, 0.02)
    price = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * vol))
    return [(t0 + timedelta(hours=i), float(price[i])) for i in range(n)]


def _train_bundle(models_dir, asset="ETH", models=("baseline", "sklearn"), seed=0):
    series = _series(700, seed=seed)
    times, prices = F.resample_hourly(series)
    X, y, seqs, names, idx = F.build_supervised(times, prices, 72, 24, 6)
    report = train_models(
        X, y, seqs, names, idx, asset=asset,
        lookback=72, horizon=24, stride=6, models=list(models),
        models_dir=models_dir,
    )
    return report


# --- reference snapshot ----------------------------------------------------- #

def test_reference_saved_by_training_and_roundtrips(tmp_path):
    from crypto_intel.mlops.monitor import load_reference

    report = _train_bundle(tmp_path)
    ref_X, names = load_reference(report["bundle_dir"])
    assert ref_X.ndim == 2
    assert names == report["feature_names"]
    assert ref_X.shape[0] > 0


# --- MLflow tracking + registry --------------------------------------------- #

def test_track_training_logs_run_and_registers(tmp_path):
    pytest.importorskip("mlflow")
    import mlflow
    from crypto_intel.mlops.tracking import track_training

    report = _train_bundle(tmp_path / "models", models=("baseline", "sklearn"))
    settings = Settings(
        models_path=tmp_path / "models",
        mlflow_tracking_uri=str(tmp_path / "mlruns"),
        mlflow_experiment="test-vol",
        mlflow_registry_prefix="test-vol",
    )
    info = track_training(report, report["bundle_dir"], settings)
    assert info is not None
    assert info["run_id"]

    client = mlflow.tracking.MlflowClient(tracking_uri=info["tracking_uri"])
    run = client.get_run(info["run_id"])
    # Params + at least one metric were logged.
    assert run.data.params["asset"] == "ETH"
    assert any(k.endswith(".test.skill") for k in run.data.metrics)

    if report["best"] in ("baseline", "sklearn"):  # tabular → registered
        assert info["registered"] is True
        assert client.get_registered_model(info["model_name"]) is not None


# --- Evidently drift report ------------------------------------------------- #

def test_build_drift_report_writes_html(tmp_path):
    pytest.importorskip("evidently")
    pytest.importorskip("pandas")
    from crypto_intel.mlops.monitor import build_drift_report

    rng = np.random.default_rng(0)
    names = ["a", "b", "c"]
    ref = rng.normal(0, 1, (200, 3))
    cur = rng.normal(0.6, 1, (200, 3))  # shifted → should register drift
    out = tmp_path / "ETH.html"
    summary = build_drift_report(ref, cur, names, out)
    assert out.exists() and out.stat().st_size > 0
    assert summary["n_features"] == 3


def test_compute_current_features_matches_geometry(tmp_path):
    from crypto_intel.mlops.monitor import compute_current_features

    meta = {"lookback_hours": 72, "horizon_hours": 24, "stride_hours": 6}
    X, names = compute_current_features(_series(400), meta)
    assert X.shape[1] == len(names) == len(F.FEATURE_NAMES)
    assert X.shape[0] > 0


# --- FastAPI serving -------------------------------------------------------- #

class _StubClient:
    """A CoinGecko client whose market_chart_range returns a synthetic series."""

    def market_chart_range(self, coingecko_id, start, end):
        n = int((end - start).total_seconds() // 3600) + 1
        rng = np.random.default_rng(1)
        price = 100.0 * np.exp(np.cumsum(rng.standard_normal(max(n, 2)) * 0.01))
        return [(start + timedelta(hours=i), float(price[i])) for i in range(max(n, 2))]


@pytest.fixture
def web_client(tmp_path, monkeypatch):
    fastapi = pytest.importorskip("fastapi")  # noqa: F841
    from fastapi.testclient import TestClient
    from crypto_intel.web import app as webapp

    # A trained bundle in an isolated models dir, and settings pointing at it.
    _train_bundle(tmp_path / "models", models=("baseline", "sklearn"))
    settings = Settings(
        models_path=tmp_path / "models",
        monitoring_path=tmp_path / "monitoring",
    )
    monkeypatch.setattr(webapp, "get_settings", lambda: settings)
    webapp.app.dependency_overrides[webapp.get_price_client] = lambda: _StubClient()
    client = TestClient(webapp.app)
    yield client, settings
    webapp.app.dependency_overrides.clear()


def test_api_forecast_returns_volforecast(web_client):
    client, _ = web_client
    resp = client.get("/api/forecast", params={"asset": "ETH"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["asset"] == "ETH"
    assert body["regime"] in ("calm", "normal", "turbulent")
    assert "predicted_vol" in body
    assert any("investment advice" in n.lower() for n in body["notes"])


def test_api_forecast_404_when_untrained(web_client):
    client, _ = web_client
    resp = client.get("/api/forecast", params={"asset": "SOL"})  # no bundle
    assert resp.status_code == 404


def test_monitoring_404_then_200(web_client):
    client, settings = web_client
    assert client.get("/monitoring", params={"asset": "ETH"}).status_code == 404
    # Drop a report where the endpoint looks for it.
    out = settings.resolve_path(settings.monitoring_path) / "ETH.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("<html>drift</html>", encoding="utf-8")
    ok = client.get("/monitoring", params={"asset": "ETH"})
    assert ok.status_code == 200 and "drift" in ok.text
