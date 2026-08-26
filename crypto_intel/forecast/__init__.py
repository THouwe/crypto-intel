"""Volatility / risk-regime forecasting subsystem (phase S9).

Forecasts **next-window realized volatility** and a derived **risk regime**
(``calm`` / ``normal`` / ``turbulent``) for an asset — never price direction (see
``docs/ML_ROADMAP.md`` § 1.1). The forecast feeds the RAG explainer with market
state; it does not emit trade signals or investment advice.

Submodules:

- :mod:`crypto_intel.forecast.features`  — pure feature engineering (no network,
  numpy-only): grid resampling, log returns, realized vol, the supervised windower.
- :mod:`crypto_intel.forecast.models`    — the unified ``Forecaster`` zoo
  (baseline, sklearn, xgboost/lightgbm, torch LSTM) + bundle save/load.
- :mod:`crypto_intel.forecast.dataset`   — history fetch + optional news features.
- :mod:`crypto_intel.forecast.train`     — build → split → fit → compare → persist.
- :mod:`crypto_intel.forecast.predict`   — load a bundle → ``VolForecast``.
"""
