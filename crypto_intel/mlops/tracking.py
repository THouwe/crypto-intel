"""MLflow experiment tracking + model registry around ``train`` (phase S11).

:func:`track_training` logs a training run's params, per-model metrics, and the
winning model **bundle** as artifacts, then registers a servable ``pyfunc`` model
version (``<prefix>-<ASSET>``) in the MLflow Model Registry.

Zero-infra by default: a local ``./mlruns`` file store (``MLFLOW_TRACKING_URI``).
The import is lazy and the whole thing is best-effort — a tracking hiccup logs a
warning and never fails the training run.

Serving note: the ``pyfunc`` wrapper serves **tabular** forecasters
(baseline / sklearn / xgboost / lightgbm). A sequence (LSTM) best model is logged
and its metrics tracked, but not registered as a tabular pyfunc — it needs the raw
return sequence, not the feature row. This boundary is intentional for S11.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_TABULAR = {"baseline", "sklearn", "xgboost", "lightgbm"}


def _resolve_tracking_uri(uri: str, repo_root: Path) -> str:
    """Resolve to a usable tracking URI.

    MLflow 3.x deprecated the plain file store and the model registry needs a
    database backend, so a local **SQLite** DB is the zero-infra default: a
    ``sqlite:///`` URI is made absolute under the repo, a bare path becomes a
    SQLite DB inside it, and real URIs (http, postgresql, ...) pass through.
    """
    if uri.startswith("sqlite:///"):
        p = Path(uri[len("sqlite:///"):])
        if not p.is_absolute():
            p = repo_root / p
        p.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{p.as_posix()}"
    if "://" in uri:
        return uri
    p = Path(uri)
    if not p.is_absolute():
        p = repo_root / p
    p.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(p / 'mlflow.db').as_posix()}"


def _artifact_location(tracking_uri: str) -> str | None:
    """A local ``mlartifacts`` dir beside the SQLite DB (None for remote URIs)."""
    if tracking_uri.startswith("sqlite:///"):
        art = Path(tracking_uri[len("sqlite:///"):]).parent / "mlartifacts"
        art.mkdir(parents=True, exist_ok=True)
        return art.as_uri()
    return None


def _set_experiment(mlflow, name: str, artifact_location: str | None) -> None:
    """Select the experiment, creating it with a local artifact dir if new."""
    client = mlflow.tracking.MlflowClient()
    if client.get_experiment_by_name(name) is None:
        client.create_experiment(name, artifact_location=artifact_location)
    mlflow.set_experiment(name)


def _bundle_pyfunc():
    """Construct the pyfunc wrapper class (needs mlflow at definition time)."""
    import mlflow

    class _BundlePyfunc(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            from ..forecast.models import load_bundle

            self._fc, self._meta = load_bundle(context.artifacts["bundle"])
            self._names = self._meta["feature_names"]

        def predict(self, context, model_input):
            import numpy as np

            cols = [c for c in self._names if c in getattr(model_input, "columns", [])]
            X = model_input[cols].to_numpy(dtype=float)
            return self._fc.predict(X, None)

    return _BundlePyfunc


def track_training(report: dict, bundle_dir: str | Path, settings) -> dict | None:
    """Log a completed training ``report`` to MLflow and register the best model.

    Returns a small dict ``{run_id, experiment, model_name, registered}`` or
    ``None`` if MLflow is unavailable / tracking failed (never raises).
    """
    try:
        import mlflow
    except ImportError:
        logger.warning("MLflow not installed; skipping tracking (pip install -e .[mlops]).")
        return None

    try:
        from ..config import _REPO_ROOT  # type: ignore
    except Exception:  # pragma: no cover
        _REPO_ROOT = Path.cwd()

    try:
        uri = _resolve_tracking_uri(settings.mlflow_tracking_uri, _REPO_ROOT)
        mlflow.set_tracking_uri(uri)
        _set_experiment(mlflow, settings.mlflow_experiment, _artifact_location(uri))

        asset = report["asset"]
        best = report["best"]
        model_name = f"{settings.mlflow_registry_prefix}-{asset.upper()}"

        with mlflow.start_run(run_name=f"{asset.upper()}-{best}") as run:
            mlflow.log_params(
                {
                    "asset": asset,
                    "lookback_hours": report["lookback_hours"],
                    "horizon_hours": report["horizon_hours"],
                    "stride_hours": report["stride_hours"],
                    "n_samples": report["n_samples"],
                    "n_features": report["n_features"],
                    "models": ",".join(report["models"].keys()),
                    "best_model": best,
                }
            )
            for name, res in report["models"].items():
                for split in ("val", "test"):
                    for metric, value in res[split].items():
                        mlflow.log_metric(f"{name}.{split}.{metric}", float(value))
            mlflow.set_tag("regime_thresholds", str(report.get("regime_thresholds")))

            bundle_dir = Path(bundle_dir)
            if bundle_dir.exists():
                mlflow.log_artifacts(str(bundle_dir), artifact_path="bundle")
            metrics_json = bundle_dir.parent / "metrics.json"
            if metrics_json.exists():
                mlflow.log_artifact(str(metrics_json))

            registered = False
            if best in _TABULAR and bundle_dir.exists():
                try:
                    _log_pyfunc(mlflow, bundle_dir, model_name)
                    registered = True
                except Exception as exc:  # registry is best-effort
                    logger.warning("Model registration failed: %s", exc)
            else:
                mlflow.set_tag("registered", "false")
                mlflow.set_tag("register_skip_reason", f"non-tabular best model ({best})")

            return {
                "run_id": run.info.run_id,
                "experiment": settings.mlflow_experiment,
                "model_name": model_name if registered else None,
                "registered": registered,
                "tracking_uri": uri,
            }
    except Exception as exc:  # never fail training because of tracking
        logger.warning("MLflow tracking failed: %s", exc)
        return None


def _log_pyfunc(mlflow, bundle_dir: Path, model_name: str) -> None:
    """Log + register the bundle as a servable pyfunc model (version-tolerant)."""
    pyfunc_cls = _bundle_pyfunc()
    kwargs = dict(
        python_model=pyfunc_cls(),
        artifacts={"bundle": str(bundle_dir)},
        registered_model_name=model_name,
    )
    try:  # mlflow 3.x prefers name=; 2.x uses artifact_path=
        mlflow.pyfunc.log_model(name="model", **kwargs)
    except TypeError:
        mlflow.pyfunc.log_model(artifact_path="model", **kwargs)
