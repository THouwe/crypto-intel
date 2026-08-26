"""The forecaster zoo + model-bundle persistence (S9).

A unified :class:`Forecaster` interface over four backends of increasing power:

- :class:`PersistenceBaseline` — the honest yardstick (``ŷ = rv_lookback``),
- :class:`SklearnForecaster`   — ``HistGradientBoostingRegressor`` (no extra dep),
- :class:`GBMForecaster`       — XGBoost / LightGBM (``[gbm]`` extra),
- :class:`SequenceForecaster`  — a small PyTorch LSTM (``[dl]`` extra).

Every model is trained and predicts in **log1p(vol)** space for positivity and
stability, exposing ``predict`` back in raw realized-vol units. Heavy backends
(sklearn/xgboost/lightgbm/torch/joblib) are imported lazily inside the methods
that need them, so importing this module costs only numpy and the RAG core stays
torch-free.

Uniform call shape: ``fit(X, seqs, y)`` / ``predict(X, seqs)`` — tabular models
read ``X``, the sequence model reads ``seqs``; the orchestrator always passes both.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES

_EPS = 1e-12


def _to_log(y: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(np.asarray(y, dtype=float), 0.0, None))


def _from_log(z: np.ndarray) -> np.ndarray:
    return np.clip(np.expm1(np.asarray(z, dtype=float)), 0.0, None)


# --------------------------------------------------------------------------- #
# Base                                                                        #
# --------------------------------------------------------------------------- #

class Forecaster:
    """Common interface. Subclasses set ``name`` and ``input_kind``."""

    name: str = "base"
    input_kind: str = "tabular"  # "tabular" | "sequence"

    def fit(self, X: np.ndarray, seqs: np.ndarray, y: np.ndarray) -> "Forecaster":
        raise NotImplementedError

    def predict(self, X: np.ndarray, seqs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self, d: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, d: Path) -> "Forecaster":
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Persistence baseline                                                        #
# --------------------------------------------------------------------------- #

class PersistenceBaseline(Forecaster):
    """Predict next-window vol = current lookback realized vol. No fitting."""

    name = "baseline"
    input_kind = "tabular"

    def __init__(self, feature_names: list[str] | None = None) -> None:
        self.feature_names = feature_names or list(FEATURE_NAMES)
        self._idx = self.feature_names.index("rv_lookback")

    def fit(self, X, seqs, y):  # nothing to learn
        return self

    def predict(self, X, seqs):
        return np.asarray(X, dtype=float)[:, self._idx]

    def save(self, d: Path) -> None:
        (d / "model_config.json").write_text(
            json.dumps({"feature_names": self.feature_names}), encoding="utf-8"
        )

    @classmethod
    def load(cls, d: Path) -> "PersistenceBaseline":
        cfg = json.loads((d / "model_config.json").read_text(encoding="utf-8"))
        return cls(cfg["feature_names"])


# --------------------------------------------------------------------------- #
# scikit-learn                                                                #
# --------------------------------------------------------------------------- #

class SklearnForecaster(Forecaster):
    """StandardScaler → HistGradientBoostingRegressor, in log-vol space."""

    name = "sklearn"
    input_kind = "tabular"

    def __init__(self, pipeline=None, seed: int = 0) -> None:
        self._pipe = pipeline
        self.seed = seed

    def _build(self):
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "gbt",
                    HistGradientBoostingRegressor(
                        max_iter=300,
                        learning_rate=0.05,
                        max_depth=3,
                        l2_regularization=1.0,
                        random_state=self.seed,
                    ),
                ),
            ]
        )

    def fit(self, X, seqs, y):
        self._pipe = self._build()
        self._pipe.fit(np.asarray(X, dtype=float), _to_log(y))
        return self

    def predict(self, X, seqs):
        return _from_log(self._pipe.predict(np.asarray(X, dtype=float)))

    def save(self, d: Path) -> None:
        import joblib

        joblib.dump(self._pipe, d / "model.joblib")

    @classmethod
    def load(cls, d: Path) -> "SklearnForecaster":
        import joblib

        obj = cls()
        obj._pipe = joblib.load(d / "model.joblib")
        return obj


# --------------------------------------------------------------------------- #
# Gradient-boosted trees (XGBoost / LightGBM)                                 #
# --------------------------------------------------------------------------- #

class GBMForecaster(Forecaster):
    """XGBoost (default) or LightGBM regressor, in log-vol space."""

    input_kind = "tabular"

    def __init__(self, backend: str = "xgboost", model=None, seed: int = 0) -> None:
        self.backend = backend
        self.name = backend
        self._model = model
        self.seed = seed

    def _build(self):
        if self.backend == "xgboost":
            from xgboost import XGBRegressor

            return XGBRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=self.seed,
                n_jobs=0,
            )
        if self.backend == "lightgbm":
            from lightgbm import LGBMRegressor

            return LGBMRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=3,
                num_leaves=15,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=self.seed,
                n_jobs=1,
                verbose=-1,
            )
        raise ValueError(f"Unknown GBM backend {self.backend!r}")

    def fit(self, X, seqs, y):
        self._model = self._build()
        self._model.fit(np.asarray(X, dtype=float), _to_log(y))
        return self

    def predict(self, X, seqs):
        return _from_log(self._model.predict(np.asarray(X, dtype=float)))

    def feature_importances(self, feature_names: list[str]) -> dict:
        imp = getattr(self._model, "feature_importances_", None)
        if imp is None:
            return {}
        return {n: float(v) for n, v in zip(feature_names, imp)}

    def save(self, d: Path) -> None:
        import joblib

        joblib.dump(self._model, d / "model.joblib")
        (d / "model_config.json").write_text(
            json.dumps({"backend": self.backend}), encoding="utf-8"
        )

    @classmethod
    def load(cls, d: Path) -> "GBMForecaster":
        import joblib

        cfg = json.loads((d / "model_config.json").read_text(encoding="utf-8"))
        obj = cls(backend=cfg["backend"])
        obj._model = joblib.load(d / "model.joblib")
        return obj


# --------------------------------------------------------------------------- #
# PyTorch sequence model (LSTM)                                               #
# --------------------------------------------------------------------------- #

class SequenceForecaster(Forecaster):
    """A small LSTM over the lookback return sequence, in log-vol space.

    Input channels are the standardized return and its square (a vol proxy). Kept
    tiny (1 layer, 32 hidden) — the datasets here are small (Risk R2).
    """

    name = "lstm"
    input_kind = "sequence"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self._net = None
        self._mu = 0.0
        self._sd = 1.0
        self.hidden = 32
        self._epochs = 0

    # -- tensor prep -------------------------------------------------------- #
    def _standardize(self, seqs: np.ndarray) -> np.ndarray:
        r = (np.asarray(seqs, dtype=float) - self._mu) / (self._sd + _EPS)
        return np.stack([r, r**2], axis=-1)  # (n, L, 2)

    def _make_net(self, hidden: int):
        import torch.nn as nn

        class _LSTMReg(nn.Module):
            def __init__(self, hidden):
                super().__init__()
                self.lstm = nn.LSTM(input_size=2, hidden_size=hidden, batch_first=True)
                self.head = nn.Sequential(
                    nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
                )

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        return _LSTMReg(hidden)

    def fit(self, X, seqs, y, val_frac: float = 0.15, max_epochs: int = 200):
        import torch

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        seqs = np.asarray(seqs, dtype=float)
        self._mu = float(seqs.mean())
        self._sd = float(seqs.std()) or 1.0

        feats = self._standardize(seqs)
        target = _to_log(y)
        n = feats.shape[0]
        n_val = max(1, int(n * val_frac))
        # temporal split — last slice is validation (no shuffle).
        tr = slice(0, n - n_val)
        va = slice(n - n_val, n)

        Xt = torch.tensor(feats, dtype=torch.float32)
        yt = torch.tensor(target, dtype=torch.float32)
        self._net = self._make_net(self.hidden)
        opt = torch.optim.Adam(self._net.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = torch.nn.MSELoss()

        best_val = float("inf")
        best_state = None
        patience, bad = 20, 0
        for epoch in range(max_epochs):
            self._net.train()
            opt.zero_grad()
            pred = self._net(Xt[tr])
            loss = loss_fn(pred, yt[tr])
            loss.backward()
            opt.step()

            self._net.eval()
            with torch.no_grad():
                vloss = float(loss_fn(self._net(Xt[va]), yt[va]))
            if vloss < best_val - 1e-6:
                best_val, bad = vloss, 0
                best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
            self._epochs = epoch + 1
        if best_state is not None:
            self._net.load_state_dict(best_state)
        return self

    def predict(self, X, seqs):
        import torch

        self._net.eval()
        feats = torch.tensor(self._standardize(seqs), dtype=torch.float32)
        with torch.no_grad():
            z = self._net(feats).numpy()
        return _from_log(z)

    def save(self, d: Path) -> None:
        import torch

        torch.save(self._net.state_dict(), d / "model.pt")
        (d / "model_config.json").write_text(
            json.dumps({"hidden": self.hidden, "mu": self._mu, "sd": self._sd}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, d: Path) -> "SequenceForecaster":
        import torch

        cfg = json.loads((d / "model_config.json").read_text(encoding="utf-8"))
        obj = cls()
        obj.hidden = cfg["hidden"]
        obj._mu, obj._sd = cfg["mu"], cfg["sd"]
        obj._net = obj._make_net(obj.hidden)
        obj._net.load_state_dict(torch.load(d / "model.pt"))
        obj._net.eval()
        return obj


# --------------------------------------------------------------------------- #
# Registry + bundle persistence                                               #
# --------------------------------------------------------------------------- #

REGISTRY = {
    "baseline": PersistenceBaseline,
    "sklearn": SklearnForecaster,
    "xgboost": GBMForecaster,
    "lightgbm": GBMForecaster,
    "lstm": SequenceForecaster,
}


def make_forecaster(name: str, feature_names: list[str] | None = None,
                    seed: int = 0) -> Forecaster:
    """Construct a fresh forecaster by name."""
    if name == "baseline":
        return PersistenceBaseline(feature_names)
    if name == "sklearn":
        return SklearnForecaster(seed=seed)
    if name in ("xgboost", "lightgbm"):
        return GBMForecaster(backend=name, seed=seed)
    if name == "lstm":
        return SequenceForecaster(seed=seed)
    raise ValueError(f"Unknown model {name!r}. Known: {sorted(REGISTRY)}")


def save_bundle(forecaster: Forecaster, metadata: dict, directory: Path) -> Path:
    """Persist a trained forecaster + its ``metadata.json`` to ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    forecaster.save(directory)
    meta = dict(metadata)
    meta.setdefault("model_name", forecaster.name)
    (directory / "metadata.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    return directory


def load_bundle(directory: Path) -> tuple[Forecaster, dict]:
    """Load a forecaster + metadata written by :func:`save_bundle`."""
    directory = Path(directory)
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    cls = REGISTRY[meta["model_name"]]
    return cls.load(directory), meta
