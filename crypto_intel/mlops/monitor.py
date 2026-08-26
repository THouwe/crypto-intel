"""Drift monitoring with Evidently (phase S11).

Two light, dependency-free helpers (:func:`save_reference` / :func:`load_reference`)
snapshot the training feature distribution into a model bundle — ``train`` calls
``save_reference`` so monitoring has a reference to compare against later.

:func:`build_drift_report` (Evidently, imported lazily) compares a *current*
feature matrix against that reference and writes a static HTML report — the
clickable proof of production-ML hygiene served at ``/monitoring``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

REFERENCE_FILE = "reference_features.csv"


# --------------------------------------------------------------------------- #
# Reference snapshot (no heavy deps — usable from train)                      #
# --------------------------------------------------------------------------- #

def save_reference(bundle_dir: Path, X: np.ndarray, feature_names: list[str]) -> Path:
    """Persist the training feature matrix as the drift reference (CSV)."""
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    path = bundle_dir / REFERENCE_FILE
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(feature_names)
        for row in np.asarray(X, dtype=float):
            w.writerow([f"{v:.10g}" for v in row])
    return path


def load_reference(bundle_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Load the reference feature matrix + column names written by training."""
    path = Path(bundle_dir) / REFERENCE_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No drift reference at {path}. Retrain with a version that writes it."
        )
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        names = next(reader)
        rows = [[float(v) for v in row] for row in reader if row]
    return np.array(rows, dtype=float), names


# --------------------------------------------------------------------------- #
# Current features (reuses the S9 windower)                                   #
# --------------------------------------------------------------------------- #

def compute_current_features(series, meta: dict, news_fn=None) -> tuple[np.ndarray, list[str]]:
    """Build a current feature matrix over ``series`` using the bundle's geometry."""
    from ..forecast.dataset import prepare_dataset

    L = int(meta["lookback_hours"])
    H = int(meta["horizon_hours"])
    S = int(meta["stride_hours"])
    X, _, _, names, _, _, _ = prepare_dataset(series, L, H, S, news_fn=news_fn)
    return X, names


# --------------------------------------------------------------------------- #
# Evidently drift report (lazy import)                                        #
# --------------------------------------------------------------------------- #

def build_drift_report(
    reference_X: np.ndarray,
    current_X: np.ndarray,
    feature_names: list[str],
    out_html: Path,
) -> dict:
    """Write an Evidently data-drift HTML report; return a small summary dict.

    Summary: ``{n_features, dataset_drift, drift_share, out_html}``. Raises
    ``ImportError`` (with an install hint) when the ``[monitor]`` extra is absent.
    """
    try:
        import pandas as pd
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Monitoring needs the [monitor] extra: pip install -e .[monitor]"
        ) from exc

    ref_df = pd.DataFrame(np.asarray(reference_X, dtype=float), columns=feature_names)
    cur_df = pd.DataFrame(np.asarray(current_X, dtype=float), columns=feature_names)
    data_def = DataDefinition(numerical_columns=list(feature_names))
    ref_ds = Dataset.from_pandas(ref_df, data_definition=data_def)
    cur_ds = Dataset.from_pandas(cur_df, data_definition=data_def)

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(current_data=cur_ds, reference_data=ref_ds)

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(out_html))

    return {
        "n_features": len(feature_names),
        "drift_summary": _summarize(snapshot),
        "out_html": str(out_html),
    }


def _summarize(snapshot) -> dict:
    """Extract drifted-column count + share from a snapshot (version-tolerant)."""
    try:
        d = snapshot.dict()
    except Exception:  # pragma: no cover
        return {}
    for metric in d.get("metrics", []):
        name = str(metric.get("metric_name") or metric.get("metric_id") or "")
        val = metric.get("value")
        if name.startswith("DriftedColumnsCount") and isinstance(val, dict):
            return {
                "drifted_columns": int(val.get("count", 0)),
                "drift_share": round(float(val.get("share", 0.0)), 3),
            }
    return {}
