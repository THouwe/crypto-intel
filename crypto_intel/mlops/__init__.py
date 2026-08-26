"""MLOps loop (phase S11): experiment tracking, registry, monitoring.

- :mod:`crypto_intel.mlops.tracking` — MLflow experiment tracking + model
  registry around ``train`` (``[mlops]`` extra).
- :mod:`crypto_intel.mlops.monitor`  — Evidently feature/prediction **drift**
  reports (``[monitor]`` extra), plus the light reference-feature snapshot that
  ``train`` writes into each model bundle.

Both backends are optional and imported lazily, so the forecast core and the RAG
CLI keep working (and installing) without them.
"""
