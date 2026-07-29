"""Tests for ingest orchestration: dedup, idempotency, fail-soft."""

from __future__ import annotations

from datetime import datetime, timezone

from crypto_intel.ingest.base import RawItem
from crypto_intel.models import SourceType
from crypto_intel.pipeline import ingest_all, load_existing_ids, read_documents_summary


class FakeConnector:
    """A connector that yields a fixed list of RawItems (no network)."""

    def __init__(self, name, source_type, items, raises=False):
        self.name = name
        self.source_type = source_type
        self._items = items
        self._raises = raises

    def fetch(self, lookback_hours):
        if self._raises:
            raise RuntimeError("boom")
        return list(self._items)


def _item(url, ts, text="Ethereum moved", source_name="CoinDesk"):
    return RawItem(
        source=SourceType.news,
        source_name=source_name,
        url=url,
        text=text,
        published_at=ts,
    )


def test_ingest_writes_deduplicated_documents(tmp_settings):
    ts = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
    # Two distinct items plus an in-batch duplicate of the first.
    items = [_item("https://x/1", ts), _item("https://x/2", ts), _item("https://x/1", ts)]
    conn = FakeConnector("fake", SourceType.news, items)

    result = ingest_all(connectors=[conn], lookback_hours=48, settings=tmp_settings, embed=False)

    assert result.fetched == 3
    assert result.added == 2
    assert result.duplicates == 1
    total, per_source, _ = read_documents_summary(tmp_settings.documents_file)
    assert total == 2
    assert per_source == {"news": 2}


def test_reingest_is_idempotent(tmp_settings):
    ts = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
    items = [_item("https://x/1", ts), _item("https://x/2", ts)]
    conn = FakeConnector("fake", SourceType.news, items)

    first = ingest_all(connectors=[conn], lookback_hours=48, settings=tmp_settings, embed=False)
    assert first.added == 2

    # Re-run with the identical connector: nothing new should be written.
    second = ingest_all(connectors=[conn], lookback_hours=48, settings=tmp_settings, embed=False)
    assert second.added == 0
    assert second.duplicates == 2

    total, _, _ = read_documents_summary(tmp_settings.documents_file)
    assert total == 2
    assert len(load_existing_ids(tmp_settings.documents_file)) == 2


def test_ingest_tags_assets_on_write(tmp_settings):
    ts = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
    conn = FakeConnector(
        "fake",
        SourceType.news,
        [_item("https://x/1", ts, text="Bitcoin and Ethereum both fell")],
    )
    ingest_all(connectors=[conn], lookback_hours=48, settings=tmp_settings, embed=False)

    from crypto_intel.pipeline import iter_documents

    docs = list(iter_documents(tmp_settings.documents_file))
    assert len(docs) == 1
    assert docs[0].assets == ["BTC", "ETH"]


class _CountingStore:
    """Minimal store stub: reports a chunk count without touching Chroma."""

    def __init__(self, count):
        self._count = count

    def count(self):
        return self._count


def test_ingest_skip_if_populated_skips_when_store_has_content(tmp_settings):
    ts = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
    conn = FakeConnector("fake", SourceType.news, [_item("https://x/1", ts)])

    result = ingest_all(
        connectors=[conn],
        settings=tmp_settings,
        store=_CountingStore(5),  # store already populated
        skip_if_populated=True,
    )
    assert result.skipped is True
    assert result.fetched == 0  # connector never ran
    # Nothing was written to the JSONL store.
    total, _, _ = read_documents_summary(tmp_settings.documents_file)
    assert total == 0


def test_ingest_skip_if_populated_runs_when_empty(tmp_settings):
    ts = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
    conn = FakeConnector("fake", SourceType.news, [_item("https://x/1", ts)])

    result = ingest_all(
        connectors=[conn],
        settings=tmp_settings,
        store=_CountingStore(0),  # empty store -> proceed
        embed=False,
        skip_if_populated=True,
    )
    assert result.skipped is False
    assert result.added == 1


def test_ingest_fails_soft_on_bad_connector(tmp_settings):
    ts = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
    good = FakeConnector("good", SourceType.news, [_item("https://x/1", ts)])
    bad = FakeConnector("bad", SourceType.news, [], raises=True)

    result = ingest_all(connectors=[good, bad], lookback_hours=48, settings=tmp_settings, embed=False)

    assert result.errors == 1
    assert result.added == 1  # the good connector still landed
