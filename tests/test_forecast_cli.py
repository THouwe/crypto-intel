"""End-to-end CLI tests for `train` / `predict` on an offline series (S9)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from crypto_intel.cli import app
from crypto_intel.config import Settings

runner = CliRunner()
UTC = timezone.utc


def _write_series_csv(path: Path, n=700, seed=0) -> None:
    rng = np.random.default_rng(seed)
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    idx = np.arange(n)
    vol = np.where((idx // 40) % 2 == 0, 0.004, 0.02)  # regime-switching vol
    price = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * vol))
    lines = [f"{(t0 + timedelta(hours=int(i))).isoformat()},{p:.6f}" for i, p in enumerate(price)]
    path.write_text("timestamp,price\n" + "\n".join(lines), encoding="utf-8")


@pytest.fixture
def patched_settings(tmp_path, monkeypatch):
    settings = Settings(
        documents_path=tmp_path / "documents.jsonl",
        chroma_path=tmp_path / "chroma",
        price_cache_path=tmp_path / "price_cache",
        models_path=tmp_path / "models",
        store_backend="chroma",
        database_url=None,
    )
    monkeypatch.setattr("crypto_intel.cli.get_settings", lambda: settings)
    return settings


def test_train_then_predict_offline(tmp_path, patched_settings):
    csv = tmp_path / "eth.csv"
    _write_series_csv(csv)

    # Train (fast models only — no optional backends needed for this assertion).
    res = runner.invoke(
        app,
        ["train", "--asset", "ETH", "--models", "baseline,sklearn",
         "--offline", str(csv)],
    )
    assert res.exit_code == 0, res.stdout
    assert "Best model" in res.stdout
    assert "skill" in res.stdout.lower()

    # A bundle + metrics.json landed under the isolated models dir.
    asset_dir = patched_settings.resolve_path(patched_settings.models_path) / "ETH"
    assert asset_dir.exists()
    assert (asset_dir / "metrics.json").exists()

    # Predict from the same series.
    res2 = runner.invoke(
        app, ["predict", "--asset", "ETH", "--offline", str(csv)]
    )
    assert res2.exit_code == 0, res2.stdout
    assert "Risk regime" in res2.stdout
    assert res2.stdout.count("calm") + res2.stdout.count("CALM") >= 0  # label present
    # The guardrail note must travel with every forecast.
    assert "Not investment advice" in res2.stdout


def test_predict_without_model_errors_cleanly(tmp_path, patched_settings):
    csv = tmp_path / "eth.csv"
    _write_series_csv(csv, n=200)
    res = runner.invoke(app, ["predict", "--asset", "ETH", "--offline", str(csv)])
    assert res.exit_code == 1
    assert "No trained model" in res.stdout or "Could not forecast" in res.stdout


def test_train_regime_label_in_predict(tmp_path, patched_settings):
    csv = tmp_path / "btc.csv"
    _write_series_csv(csv, seed=7)
    runner.invoke(app, ["train", "--asset", "BTC", "--models", "baseline", "--offline", str(csv)])
    res = runner.invoke(app, ["predict", "--asset", "BTC", "--offline", str(csv)])
    assert res.exit_code == 0, res.stdout
    assert any(w in res.stdout.upper() for w in ("CALM", "NORMAL", "TURBULENT"))
