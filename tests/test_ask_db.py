"""Tests for the DB-backed ask_db flow (contract; no DB / no model load)."""

from __future__ import annotations

import crypto_intel.pipeline as pipeline
from crypto_intel.config import Settings
from crypto_intel.pipeline import ask_db


def test_ask_db_forces_injected_store_and_skips_prices(monkeypatch):
    captured: dict = {}

    def fake_retrieve_context(question, **kwargs):
        captured["question"] = question
        captured.update(kwargs)
        return "CTX"

    monkeypatch.setattr(pipeline, "retrieve_context", fake_retrieve_context)

    sentinel_store = object()
    out = ask_db(
        "What happened to ETH within the last 1 week?",
        settings=Settings(),
        asset_override="ETH",
        hours_override=168,
        store=sentinel_store,
    )

    assert out == "CTX"
    assert captured["store"] is sentinel_store  # forced the injected store
    assert captured["detect_prices"] is False  # GUI confirms the move separately
    assert captured["asset_override"] == "ETH"
    assert captured["hours_override"] == 168


def test_ask_db_defaults_to_pgvector_store(monkeypatch):
    from crypto_intel.pgstore import PgVectorStore

    captured: dict = {}
    monkeypatch.setattr(
        pipeline, "retrieve_context", lambda q, **kw: captured.update(kw) or "CTX"
    )

    ask_db("q", settings=Settings(database_url="postgresql://x/y"))

    # With no store injected, ask_db builds a PgVectorStore (construction is lazy,
    # so no connection is attempted here).
    assert isinstance(captured["store"], PgVectorStore)
