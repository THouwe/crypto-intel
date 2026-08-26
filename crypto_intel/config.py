"""Typed application settings loaded from environment + `.env`.

All runtime configuration flows through :func:`get_settings`, which returns a
cached :class:`Settings` instance. Secrets (API keys) are optional at import
time so that offline commands (``stats``, ``price-event``) and the test suite
run without any credentials; commands that genuinely need a key validate its
presence at call time.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: two levels up from this file (crypto_intel/config.py -> repo/).
_REPO_ROOT = Path(__file__).resolve().parent.parent
# Directory holding shipped config data (feeds.yaml, assets.yaml).
_PACKAGE_DATA = Path(__file__).resolve().parent / "data"


class Settings(BaseSettings):
    """Application settings, read from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Secrets (optional at startup; required only by the flows that use them) ---
    anthropic_api_key: str | None = Field(default=None)
    reddit_client_id: str | None = Field(default=None)
    reddit_client_secret: str | None = Field(default=None)
    reddit_user_agent: str = Field(default="crypto-intel/0.1")
    cmc_api_key: str | None = Field(default=None)
    newsapi_key: str | None = Field(default=None)

    # --- Paths ---
    chroma_path: Path = Field(default=Path("data/store/chroma"))
    documents_path: Path = Field(default=Path("data/store/documents.jsonl"))
    price_cache_path: Path = Field(default=Path("data/store/price_cache"))
    eval_cases_path: Path = Field(default=Path("data/eval_cases.json"))

    # --- Models ---
    embed_model: str = Field(default="all-MiniLM-L6-v2")
    # Embedding backend: "onnx" (default — onnxruntime via chromadb, same MiniLM
    # model, no torch dependency, needs no extra install) or
    # "sentence-transformers" (requires the `st` extra: pip install -e .[st]).
    embed_backend: str = Field(default="onnx")
    synth_model: str = Field(default="claude-sonnet-5")

    # --- Chunking (word-count based) ---
    chunk_size: int = Field(default=220)
    chunk_overlap: int = Field(default=40)

    # --- Prices (CoinGecko) ---
    coingecko_base_url: str = Field(default="https://api.coingecko.com/api/v3")
    coingecko_api_key: str | None = Field(default=None)  # optional demo/pro key
    price_cache_ttl_seconds: int = Field(default=600)
    # Net move smaller than this (abs %) over the window is reported as "flat".
    flat_threshold_pct: float = Field(default=1.0)

    # --- Defaults ---
    default_lookback_hours: int = Field(default=48)
    default_k: int = Field(default=6)

    # --- Forecasting (phase S9) ---
    models_path: Path = Field(default=Path("data/models"))
    forecast_lookback_hours: int = Field(default=72)
    forecast_horizon_hours: int = Field(default=24)
    forecast_stride_hours: int = Field(default=6)
    forecast_history_days: int = Field(default=90)  # CoinGecko free-tier hourly ceiling
    regime_low_pct: float = Field(default=33.0)
    regime_high_pct: float = Field(default=66.0)

    # --- Warehouse (phase S10) ---
    warehouse_path: Path = Field(default=Path("data/warehouse.duckdb"))
    bq_project: str | None = Field(default=None)  # GCP project for the [bq] loader
    bq_dataset: str = Field(default="crypto_intel")

    # --- Chroma collection name ---
    chroma_collection: str = Field(default="crypto_intel_chunks")

    # --- Vector store backend ---
    # "chroma" (default, embedded/local) or "pgvector" (Postgres/Supabase; needs
    # the `pg` extra + database_url). Both use the same 384-dim MiniLM embeddings.
    store_backend: str = Field(default="chroma")
    database_url: str | None = Field(default=None)  # Postgres DSN for pgvector

    def resolve_path(self, path: Path) -> Path:
        """Resolve a (possibly relative) path against the repo root."""
        path = Path(path)
        return path if path.is_absolute() else (_REPO_ROOT / path)

    @property
    def chroma_dir(self) -> Path:
        return self.resolve_path(self.chroma_path)

    @property
    def documents_file(self) -> Path:
        return self.resolve_path(self.documents_path)

    @property
    def price_cache_dir(self) -> Path:
        return self.resolve_path(self.price_cache_path)

    @property
    def eval_cases_file(self) -> Path:
        return self.resolve_path(self.eval_cases_path)

    @property
    def warehouse_file(self) -> Path:
        return self.resolve_path(self.warehouse_path)

    @property
    def feeds_file(self) -> Path:
        return _PACKAGE_DATA / "feeds.yaml"

    @property
    def assets_file(self) -> Path:
        return _PACKAGE_DATA / "assets.yaml"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()
