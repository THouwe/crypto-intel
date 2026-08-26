"""Warehouse-backed feature pipeline (phase S10).

Lands price history and document metadata in a columnar warehouse and expresses
the *scale-y* feature engineering — hourly gridding, rolling window functions, and
news aggregation — in **SQL**, so ``train --source warehouse`` consumes
warehouse output rather than in-process numpy.

- :mod:`crypto_intel.warehouse.duck` — embedded **DuckDB** backend (the default;
  zero infra, ``[warehouse]`` extra).
- :mod:`crypto_intel.warehouse.bq`   — optional **BigQuery** free-tier loader
  (``[bq]`` extra) so the "cloud warehouse" story is literally true.
"""
