"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from crypto_intel.config import Settings


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    """Settings pointing the local store at an isolated temp directory.

    Pins the embedded Chroma backend so store/ingest/prune tests stay hermetic
    regardless of the developer's ``.env`` (e.g. a real ``STORE_BACKEND=pgvector``).
    """
    return Settings(
        documents_path=tmp_path / "documents.jsonl",
        chroma_path=tmp_path / "chroma",
        price_cache_path=tmp_path / "price_cache",
        store_backend="chroma",
        database_url=None,
    )
