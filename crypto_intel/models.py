"""Core pydantic data models shared across the pipeline.

All timestamps are timezone-aware UTC. These models are the single normalized
representation every connector, the vector store, and the synthesizer agree on.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """The kind of upstream source a document came from."""

    reddit = "reddit"
    news = "news"
    exchange = "exchange"
    regulator = "regulator"
    cmc = "cmc"


class Document(BaseModel):
    """A normalized source document, the unit of ingestion + dedup."""

    id: str  # sha1(source_name + url + published_at), stable dedup key
    source: SourceType
    source_name: str  # "r/ethereum" | "CoinDesk" | "Binance" | "SEC" | ...
    url: str
    title: str | None = None
    text: str  # cleaned body used for chunking
    author: str | None = None
    published_at: datetime  # UTC
    assets: list[str] = Field(default_factory=list)  # tagged tickers, e.g. ["ETH"]
    metadata: dict = Field(default_factory=dict)  # source-specific extras


class Chunk(BaseModel):
    """A chunk of a :class:`Document`, the unit stored + retrieved in Chroma."""

    id: str  # f"{doc_id}:{chunk_index}"
    doc_id: str
    chunk_index: int
    text: str
    # Denormalized metadata mirrored into Chroma for filtering:
    source: SourceType
    source_name: str
    url: str
    published_at: datetime
    assets: list[str] = Field(default_factory=list)


class PriceEvent(BaseModel):
    """A confirmed price move over an analysed window."""

    asset: str  # "ETH"
    coingecko_id: str  # "ethereum"
    window_start: datetime  # UTC — start of the analysed window
    window_end: datetime  # UTC — "now" (or requested end)
    price_start: float
    price_end: float
    pct_change: float  # signed, over the window
    max_drawdown_pct: float  # largest adverse intra-window move
    move_start: datetime  # start of the steepest adverse leg (drives retrieval)
    move_end: datetime  # end of the steepest adverse leg
    direction: Literal["up", "down", "flat"]


class Citation(BaseModel):
    """A numbered source reference resolved from the answer's `[n]` markers."""

    n: int  # citation number used in answer_text ([1], [2], ...)
    doc_id: str
    source_name: str
    url: str
    published_at: datetime
    snippet: str


class Answer(BaseModel):
    """The final grounded, cited answer returned by `ask`."""

    question: str
    event: PriceEvent | None = None
    answer_text: str  # contains inline [n] markers
    citations: list[Citation] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)  # e.g. "thin evidence"


class VolForecast(BaseModel):
    """A next-window volatility / risk-regime forecast (phase S9).

    Deliberately **not** a price-direction or trade call — see
    ``docs/ML_ROADMAP.md`` § 1.1. ``notes`` always carries the
    not-investment-advice line so the guardrail travels with the object.
    """

    asset: str  # "ETH"
    as_of: datetime  # UTC — end of the lookback window the forecast is made from
    lookback_hours: int
    horizon_hours: int
    predicted_vol: float  # realized vol over the next horizon window
    predicted_vol_annualized: float
    regime: Literal["calm", "normal", "turbulent"]
    regime_thresholds: dict = Field(default_factory=dict)  # calm_max / turbulent_min
    model_name: str  # "baseline" | "sklearn" | "xgboost" | "lstm" | ...
    skill_vs_baseline: float | None = None  # from the bundle's test metrics
    trained_at: datetime | None = None
    drivers: dict = Field(default_factory=dict)  # top feature contributions
    notes: list[str] = Field(default_factory=list)
