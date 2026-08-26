"""Optional BigQuery loader (phase S10, ``[bq]`` extra).

Mirrors the DuckDB schema into a BigQuery **free-tier** dataset so the
"cloud warehouse" story is literally true. The client import is lazy and guarded
— local runs and CI default to DuckDB and never touch GCP.

The row-mapping helpers (:func:`price_rows`, :func:`document_rows`) are pure and
unit-tested; the network load path (:func:`load_to_bigquery`) needs
``GOOGLE_APPLICATION_CREDENTIALS`` and a project, and is exercised manually.
"""

from __future__ import annotations

from datetime import datetime, timezone

UTC = timezone.utc

PRICE_SCHEMA = [("asset", "STRING"), ("ts", "TIMESTAMP"), ("price", "FLOAT64")]
DOCUMENT_SCHEMA = [
    ("asset", "STRING"), ("ts", "TIMESTAMP"), ("source", "STRING"),
    ("source_name", "STRING"), ("doc_id", "STRING"),
]


def _iso_utc(dt: datetime) -> str:
    dt = dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()


def price_rows(asset: str, series: list[tuple[datetime, float]]) -> list[dict]:
    """Map a price series to BigQuery JSON rows (pure)."""
    a = asset.upper()
    return [{"asset": a, "ts": _iso_utc(ts), "price": float(p)} for ts, p in series]


def document_rows(asset: str, documents) -> list[dict]:
    """Map asset-tagged documents to BigQuery JSON rows (pure)."""
    a = asset.upper()
    rows = []
    for doc in documents:
        if a not in [x.upper() for x in getattr(doc, "assets", [])]:
            continue
        rows.append(
            {
                "asset": a,
                "ts": _iso_utc(doc.published_at),
                "source": getattr(doc.source, "value", str(doc.source)),
                "source_name": doc.source_name,
                "doc_id": doc.id,
            }
        )
    return rows


def _bq_schema(fields: list[tuple[str, str]]):
    from google.cloud import bigquery

    return [bigquery.SchemaField(name, kind) for name, kind in fields]


def load_to_bigquery(
    project: str,
    dataset: str,
    asset: str,
    series: list[tuple[datetime, float]],
    documents=None,
    location: str = "US",
) -> dict:
    """Create the dataset/tables if needed and load ``asset``'s rows.

    Requires the ``[bq]`` extra and GCP credentials. Returns a small summary dict.
    """
    try:
        from google.cloud import bigquery
    except ImportError as exc:  # pragma: no cover - exercised only with the extra
        raise ImportError(
            "BigQuery support needs the [bq] extra: pip install -e .[bq]"
        ) from exc

    client = bigquery.Client(project=project)
    ds_ref = bigquery.Dataset(f"{project}.{dataset}")
    ds_ref.location = location
    client.create_dataset(ds_ref, exists_ok=True)

    def _load(table: str, rows: list[dict], schema):
        table_id = f"{project}.{dataset}.{table}"
        client.create_table(
            bigquery.Table(table_id, schema=_bq_schema(schema)), exists_ok=True
        )
        if rows:
            errors = client.insert_rows_json(table_id, rows)
            if errors:
                raise RuntimeError(f"BigQuery insert errors for {table}: {errors}")
        return len(rows)

    n_prices = _load("prices", price_rows(asset, series), PRICE_SCHEMA)
    n_docs = _load("documents", document_rows(asset, documents or []), DOCUMENT_SCHEMA)
    return {"dataset": f"{project}.{dataset}", "prices": n_prices, "documents": n_docs}
