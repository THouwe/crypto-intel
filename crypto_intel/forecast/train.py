"""Training orchestration: build → temporal split → fit → compare → persist (S9).

Pure-ish: the only side effects are (optionally) persisting the winning bundle and
returning a metrics dict. The dataset is passed in as arrays, so this whole path
runs offline in tests with a synthetic series (no network, no CoinGecko).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import features as F
from .models import GBMForecaster, make_forecaster, save_bundle

DEFAULT_MODELS = ["baseline", "sklearn", "xgboost", "lstm"]


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #

def _mae(y, p):
    return float(np.mean(np.abs(y - p)))


def _rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _r2(y, p):
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _qlike(y, p):
    """QLIKE loss, a volatility-appropriate score (lower is better)."""
    y = np.clip(y, 1e-9, None)
    p = np.clip(p, 1e-9, None)
    return float(np.mean(np.log(p**2) + (y**2) / (p**2)))


def _regime_f1(y_true, p, thresholds) -> float:
    """Macro-F1 of regime labels derived by bucketing true vs predicted vol."""
    labels = ["calm", "normal", "turbulent"]
    yt = [F.classify_regime(v, thresholds) for v in y_true]
    yp = [F.classify_regime(v, thresholds) for v in p]
    f1s = []
    for lab in labels:
        tp = sum(1 for a, b in zip(yt, yp) if a == lab and b == lab)
        fp = sum(1 for a, b in zip(yt, yp) if b == lab and a != lab)
        fn = sum(1 for a, b in zip(yt, yp) if a == lab and b != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s))


def _evaluate(y, p, baseline_mse, thresholds) -> dict:
    mse = float(np.mean((y - p) ** 2))
    skill = 1.0 - mse / baseline_mse if baseline_mse > 0 else 0.0
    return {
        "mae": _mae(y, p),
        "rmse": _rmse(y, p),
        "r2": _r2(y, p),
        "qlike": _qlike(y, p),
        "skill": float(skill),
        "regime_f1": _regime_f1(y, p, thresholds),
    }


# --------------------------------------------------------------------------- #
# Temporal split                                                              #
# --------------------------------------------------------------------------- #

def temporal_split(n: int, train=0.70, val=0.15):
    """Chronological index slices — never shuffled (time series)."""
    n_tr = int(n * train)
    n_va = int(n * val)
    return slice(0, n_tr), slice(n_tr, n_tr + n_va), slice(n_tr + n_va, n)


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #

def train_models(
    X: np.ndarray,
    y: np.ndarray,
    seqs: np.ndarray,
    feature_names: list[str],
    index_ts: np.ndarray,
    *,
    asset: str,
    lookback: int,
    horizon: int,
    stride: int,
    models: list[str] | None = None,
    low_pct: float = 33.0,
    high_pct: float = 66.0,
    seed: int = 0,
    models_dir: Path | None = None,
    git_sha: str | None = None,
) -> dict:
    """Fit + compare the requested models on a temporal split.

    Returns a report dict ``{asset, n, split, thresholds, models: {name: metrics},
    best, skipped}`` and, when ``models_dir`` is given, persists the best model's
    bundle to ``models_dir/<asset>/<best>/``.
    """
    models = models or list(DEFAULT_MODELS)
    n = X.shape[0]
    if n < 20:
        raise ValueError(
            f"Too few samples ({n}) to train — widen history or shorten lookback."
        )
    tr, va, te = temporal_split(n)

    # Regime thresholds are fit on the TRAIN targets only (no test leakage).
    thresholds = F.regime_thresholds(y[tr], low_pct, high_pct)

    # Baseline reference MSE on val + test, for skill scores.
    base = make_forecaster("baseline", feature_names).fit(X[tr], seqs[tr], y[tr])
    base_val_pred = base.predict(X[va], seqs[va])
    base_te_pred = base.predict(X[te], seqs[te])
    base_val_mse = float(np.mean((y[va] - base_val_pred) ** 2)) or 1e-12
    base_te_mse = float(np.mean((y[te] - base_te_pred) ** 2)) or 1e-12

    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    importances: dict[str, dict] = {}
    skipped: dict[str, str] = {}

    for name in models:
        try:
            fc = make_forecaster(name, feature_names, seed=seed)
            fc.fit(X[tr], seqs[tr], y[tr])
        except ImportError as exc:  # optional backend not installed
            skipped[name] = f"backend not installed ({exc})"
            continue
        except Exception as exc:  # keep the run alive; report the failure
            skipped[name] = f"training failed: {exc}"
            continue

        val_metrics = _evaluate(y[va], fc.predict(X[va], seqs[va]), base_val_mse, thresholds)
        test_metrics = _evaluate(y[te], fc.predict(X[te], seqs[te]), base_te_mse, thresholds)
        results[name] = {"val": val_metrics, "test": test_metrics}
        fitted[name] = fc
        if isinstance(fc, GBMForecaster):
            importances[name] = fc.feature_importances(feature_names)

    if not results:
        raise RuntimeError(f"No models could be trained. Skipped: {skipped}")

    # Select by validation skill (highest). Baseline's own skill is ~0 by design.
    best = max(results, key=lambda m: results[m]["val"]["skill"])

    report = {
        "asset": asset,
        "n_samples": n,
        "n_features": X.shape[1],
        "feature_names": feature_names,
        "split": {"train": tr.stop, "val": va.stop - va.start, "test": n - te.start},
        "lookback_hours": lookback,
        "horizon_hours": horizon,
        "stride_hours": stride,
        "regime_thresholds": thresholds,
        "models": results,
        "importances": importances,
        "best": best,
        "skipped": skipped,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    if models_dir is not None:
        bundle_dir = Path(models_dir) / asset.upper() / best
        meta = {
            "model_name": fitted[best].name,
            "asset": asset.upper(),
            "lookback_hours": lookback,
            "horizon_hours": horizon,
            "stride_hours": stride,
            "feature_names": feature_names,
            "regime_thresholds": thresholds,
            "metrics": results[best],
            "skill_vs_baseline": results[best]["test"]["skill"],
            "seed": seed,
            "git_sha": git_sha,
            "trained_at": report["trained_at"],
            "train_window": {
                "from": str(index_ts[0]) if index_ts.size else None,
                "to": str(index_ts[-1]) if index_ts.size else None,
            },
        }
        save_bundle(fitted[best], meta, bundle_dir)
        # Snapshot the training feature distribution as the drift reference (S11).
        from ..mlops.monitor import save_reference
        save_reference(bundle_dir, X[tr], feature_names)
        report["bundle_dir"] = str(bundle_dir)

    return report
