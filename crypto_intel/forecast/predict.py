"""Inference: a trained bundle + the latest data → a :class:`VolForecast` (S9).

The forecast is a next-window realized-volatility / risk-regime estimate. It is
**not** a price-direction or trade call (see ``docs/ML_ROADMAP.md`` § 1.1); the
not-investment-advice note travels on every returned object.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..config import Settings, get_settings
from ..models import VolForecast
from . import features as F
from .dataset import fetch_price_history, prepare_dataset
from .models import GBMForecaster, load_bundle

UTC = timezone.utc

NOT_ADVICE = "Not investment advice. Forecasts volatility/regime, not price direction."


def bundle_dir_for(asset: str, model: str | None, settings: Settings) -> Path:
    """Resolve the bundle directory for an asset (+ optional explicit model)."""
    base = settings.resolve_path(settings.models_path) / asset.upper()
    if model:
        return base / model
    if not base.exists():
        raise FileNotFoundError(
            f"No trained model for {asset.upper()} under {base}. Run "
            f"`crypto-intel train --asset {asset.upper()}` first."
        )
    # Single trained model → use it; otherwise the caller must disambiguate.
    subdirs = [p for p in base.iterdir() if (p / "metadata.json").exists()]
    if not subdirs:
        raise FileNotFoundError(f"No trained model bundle found under {base}.")
    if len(subdirs) > 1:
        names = ", ".join(sorted(p.name for p in subdirs))
        raise ValueError(
            f"Multiple models for {asset.upper()} ({names}); pass --model to pick one."
        )
    return subdirs[0]


def predict(
    asset: str,
    model: str | None = None,
    settings: Settings | None = None,
    client=None,
    as_of: datetime | None = None,
    series: list[tuple[datetime, float]] | None = None,
) -> VolForecast:
    """Produce a :class:`VolForecast` for ``asset`` from a trained bundle.

    ``series`` (an offline price series) short-circuits the CoinGecko fetch — the
    path the CLI/tests use for deterministic runs.
    """
    settings = settings or get_settings()
    fc, meta = load_bundle(bundle_dir_for(asset, model, settings))

    lookback = int(meta["lookback_hours"])
    horizon = int(meta["horizon_hours"])
    feature_names = meta["feature_names"]
    thresholds = meta["regime_thresholds"]
    uses_news = any(n in feature_names for n in F.NEWS_FEATURE_NAMES)

    if series is None:
        days = max(int(np.ceil(lookback / 24)) + 2, 5)
        series = fetch_price_history(asset, days, settings, client, end=as_of)

    times, prices = F.resample_hourly(series)
    if prices.size < lookback + 1:
        raise ValueError(
            f"Need at least {lookback + 1} hourly points for {asset.upper()}; "
            f"got {prices.size}. Widen the history window."
        )

    # The most recent full lookback window.
    look_prices = prices[-(lookback + 1):]
    as_of_ts = times[-1]
    news = None
    if uses_news:
        from .dataset import news_feature_fn

        fn = news_feature_fn(asset, settings)
        if fn is not None:
            news = fn(times[-(lookback + 1)], as_of_ts)

    feats = F.window_features(look_prices, horizon, news=news)
    x = F.feature_vector(feats, feature_names).reshape(1, -1)
    seq = F.log_returns(look_prices).reshape(1, -1)

    pred_vol = float(fc.predict(x, seq)[0])
    regime = F.classify_regime(pred_vol, thresholds)

    drivers = {}
    if isinstance(fc, GBMForecaster):
        imp = fc.feature_importances(feature_names)
        drivers = dict(sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:5])

    return VolForecast(
        asset=asset.upper(),
        as_of=as_of_ts if isinstance(as_of_ts, datetime) else datetime.now(UTC),
        lookback_hours=lookback,
        horizon_hours=horizon,
        predicted_vol=pred_vol,
        predicted_vol_annualized=F.annualize_vol(pred_vol, horizon),
        regime=regime,
        regime_thresholds=thresholds,
        model_name=meta["model_name"],
        skill_vs_baseline=meta.get("skill_vs_baseline"),
        trained_at=_parse_dt(meta.get("trained_at")),
        drivers=drivers,
        notes=[NOT_ADVICE],
    )


def _parse_dt(raw):
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
