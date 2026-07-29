"""Connector protocol and the shared raw-item dataclass.

A connector fetches recent content from one upstream source and yields
:class:`RawItem`s. Normalization (cleaning, asset tagging, dedup) happens later
in :mod:`crypto_intel.normalize`, so connectors stay thin and source-specific.

Contract: ``fetch`` must **fail soft** — a dead feed / missing credential logs a
warning and yields nothing, never raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Protocol, runtime_checkable

from ..models import SourceType


@dataclass
class RawItem:
    """A raw content item from a source, prior to normalization."""

    source: SourceType
    source_name: str  # "r/ethereum" | "CoinDesk" | "SEC" | ...
    url: str
    text: str
    published_at: datetime  # timezone-aware UTC
    title: str | None = None
    author: str | None = None
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class Connector(Protocol):
    """Anything that can fetch recent :class:`RawItem`s."""

    name: str
    source_type: SourceType

    def fetch(self, lookback_hours: int) -> Iterable[RawItem]:
        """Yield items published within the last ``lookback_hours``."""
        ...
