"""DuckDB warehouse: land prices + doc metadata, engineer features in SQL (S10).

The SQL here does the work a warehouse is for:

- **gridding** an irregular price series to hourly buckets (``time_bucket`` +
  ``last(... ORDER BY ...)``),
- **rolling features** via window functions (log return, rolling realized vol,
  rolling mean return) materialized into a queryable ``price_features`` table,
- **news aggregation** — per-asset, per-hour document counts (``GROUP BY``) that
  feed the forecaster's exogenous features.

``train`` reads the SQL-gridded price series and the SQL-aggregated news counts
back out, so training genuinely depends on warehouse output. All of it runs
offline against a file or ``:memory:`` database — no network.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

UTC = timezone.utc


def _to_naive_utc(dt: datetime) -> datetime:
    dt = dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).replace(tzinfo=None)


def _attach_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


class DuckWarehouse:
    """A thin DuckDB wrapper over the price / document / feature tables."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        import duckdb

        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(self.path)
        self.init_schema()

    # -- lifecycle ---------------------------------------------------------- #

    def init_schema(self) -> None:
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS prices (
                asset VARCHAR NOT NULL,
                ts    TIMESTAMP NOT NULL,   -- UTC, tz-naive
                price DOUBLE  NOT NULL,
                PRIMARY KEY (asset, ts)
            );
            """
        )
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                asset       VARCHAR NOT NULL,
                ts          TIMESTAMP NOT NULL,
                source      VARCHAR,
                source_name VARCHAR,
                doc_id      VARCHAR
            );
            """
        )

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "DuckWarehouse":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- loading ------------------------------------------------------------ #

    def load_prices(self, asset: str, series: list[tuple[datetime, float]]) -> int:
        """Rebuild the price rows for ``asset`` from a ``[(ts, price)]`` series."""
        asset = asset.upper()
        rows = [(asset, _to_naive_utc(ts), float(p)) for ts, p in series]
        self.con.execute("DELETE FROM prices WHERE asset = ?", [asset])
        if rows:
            self.con.executemany(
                "INSERT OR REPLACE INTO prices VALUES (?, ?, ?)", rows
            )
        return len(rows)

    def load_documents(self, asset: str, documents) -> int:
        """Rebuild document-metadata rows for ``asset`` from ``Document`` objects.

        Only documents tagged with ``asset`` are stored (one row per doc).
        """
        asset = asset.upper()
        rows = []
        for doc in documents:
            if asset not in [a.upper() for a in getattr(doc, "assets", [])]:
                continue
            src = getattr(doc.source, "value", str(doc.source))
            rows.append(
                (asset, _to_naive_utc(doc.published_at), src, doc.source_name, doc.id)
            )
        self.con.execute("DELETE FROM documents WHERE asset = ?", [asset])
        if rows:
            self.con.executemany(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?)", rows
            )
        return len(rows)

    # -- SQL feature engineering ------------------------------------------- #

    def build_features(self, asset: str, rolling_hours: int = 24) -> int:
        """Materialize ``price_features`` for ``asset`` using SQL window functions.

        Grids to hourly buckets, then computes per-hour log return and rolling
        realized vol / mean return over the trailing ``rolling_hours`` window.
        Returns the number of feature rows.
        """
        asset = asset.upper()
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS price_features (
                asset VARCHAR, h TIMESTAMP, price DOUBLE,
                log_return DOUBLE, rolling_rv DOUBLE, rolling_mean_return DOUBLE,
                PRIMARY KEY (asset, h)
            );
            """
        )
        self.con.execute("DELETE FROM price_features WHERE asset = ?", [asset])
        self.con.execute(
            f"""
            INSERT INTO price_features
            WITH hourly AS (
                SELECT
                    asset,
                    time_bucket(INTERVAL '1 hour', ts) AS h,
                    last(price ORDER BY ts)            AS price
                FROM prices
                WHERE asset = ?
                GROUP BY asset, h
            ),
            ret AS (
                SELECT
                    asset, h, price,
                    ln(price / lag(price) OVER w) AS log_return
                FROM hourly
                WINDOW w AS (PARTITION BY asset ORDER BY h)
            )
            SELECT
                asset, h, price, log_return,
                sqrt(sum(log_return * log_return) OVER w) AS rolling_rv,
                avg(log_return)                     OVER w AS rolling_mean_return
            FROM ret
            WINDOW w AS (
                PARTITION BY asset ORDER BY h
                ROWS BETWEEN {int(rolling_hours)} PRECEDING AND CURRENT ROW
            )
            ORDER BY h;
            """,
            [asset],
        )
        return self.con.execute(
            "SELECT count(*) FROM price_features WHERE asset = ?", [asset]
        ).fetchone()[0]

    # -- reads (consumed by train) ----------------------------------------- #

    def read_price_grid(self, asset: str) -> list[tuple[datetime, float]]:
        """Return the SQL-gridded hourly ``[(ts, price)]`` series for ``asset``."""
        asset = asset.upper()
        rows = self.con.execute(
            """
            SELECT time_bucket(INTERVAL '1 hour', ts) AS h,
                   last(price ORDER BY ts)            AS price
            FROM prices WHERE asset = ?
            GROUP BY h ORDER BY h;
            """,
            [asset],
        ).fetchall()
        return [(_attach_utc(h), float(p)) for h, p in rows]

    def news_counts_by_hour(self, asset: str) -> dict[int, tuple[int, int]]:
        """Per-hour ``{epoch_hour: (count, regulator_count)}`` from SQL aggregation."""
        asset = asset.upper()
        rows = self.con.execute(
            """
            SELECT time_bucket(INTERVAL '1 hour', ts) AS h,
                   count(*) AS n,
                   sum(CASE WHEN source = 'regulator' THEN 1 ELSE 0 END) AS reg
            FROM documents WHERE asset = ?
            GROUP BY h ORDER BY h;
            """,
            [asset],
        ).fetchall()
        out: dict[int, tuple[int, int]] = {}
        for h, n, reg in rows:
            epoch_hour = int(_attach_utc(h).timestamp())
            out[epoch_hour] = (int(n), int(reg or 0))
        return out

    def stats(self, asset: str | None = None) -> dict:
        """Row counts + coverage window, for `warehouse stats`."""
        where = "WHERE asset = ?" if asset else ""
        params = [asset.upper()] if asset else []
        n_prices = self.con.execute(
            f"SELECT count(*) FROM prices {where}", params
        ).fetchone()[0]
        n_docs = self.con.execute(
            f"SELECT count(*) FROM documents {where}", params
        ).fetchone()[0]
        span = self.con.execute(
            f"SELECT min(ts), max(ts) FROM prices {where}", params
        ).fetchone()
        return {
            "asset": asset.upper() if asset else "ALL",
            "prices": n_prices,
            "documents": n_docs,
            "from": str(span[0]) if span and span[0] else None,
            "to": str(span[1]) if span and span[1] else None,
        }


def warehouse_news_fn(counts: dict[int, tuple[int, int]]):
    """Build a ``news_fn(start, end) -> dict`` from SQL hourly news counts.

    Mirrors :func:`crypto_intel.forecast.dataset.news_feature_fn` so a
    warehouse-sourced training run yields the same news-feature columns.
    """
    if not counts:
        return None
    items = sorted(counts.items())
    hours = [h for h, _ in items]

    def _fn(start: datetime, end: datetime) -> dict:
        import bisect

        s = int(start.timestamp())
        e = int(end.timestamp())
        lo = bisect.bisect_left(hours, s)
        hi = bisect.bisect_right(hours, e)
        recent_cut = e - int((e - s) * 0.25)
        n = reg = n_recent = 0
        last_h = None
        for h, (c, r) in items[lo:hi]:
            n += c
            reg += r
            if h >= recent_cut:
                n_recent += c
            last_h = h
        recency = (e - last_h) / 3600.0 if last_h is not None else 0.0
        return {
            "news_count": float(n),
            "news_count_recent": float(n_recent),
            "news_recency_hours": float(recency),
            "regulator_count": float(reg),
        }

    return _fn
