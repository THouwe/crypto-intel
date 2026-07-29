"""Embedding backend (sentence-transformers, model ``all-MiniLM-L6-v2``).

Kept behind a small :class:`Embedder` protocol so the store/pipeline depend on
``encode(texts) -> list[list[float]]`` rather than on sentence-transformers
directly. This makes the embed step trivially injectable in tests (a
deterministic stub) and keeps the heavy ``torch`` import lazy — it only happens
the first time a real embedding is computed, never on ``--help`` or ``stats``.

The model weights download from HuggingFace on first use (network required).
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns texts into fixed-width float vectors."""

    model_name: str

    def encode(self, texts: list[str]) -> list[list[float]]:
        ...

    @property
    def dim(self) -> int:
        ...


class SentenceTransformerEmbedder:
    """Local sentence-transformers embedder; caches the loaded model."""

    def __init__(self, model_name: str | None = None, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self.model_name = model_name or settings.embed_model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            logger.info("Loading embedding model %s ...", self.model_name)
            from sentence_transformers import SentenceTransformer  # deferred

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,  # unit vectors -> clean cosine similarity
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()


class OnnxEmbedder:
    """MiniLM embedder via chromadb's bundled ONNX model (onnxruntime, no torch).

    Produces the same 384-dim ``all-MiniLM-L6-v2`` embeddings as the
    sentence-transformers backend, but without the heavy torch dependency. The
    ONNX weights download on first use (network required).
    """

    model_name = "all-MiniLM-L6-v2 (onnx)"

    def __init__(self, settings: Settings | None = None) -> None:
        self._fn = None

    @property
    def fn(self):
        if self._fn is None:
            logger.info("Loading ONNX MiniLM embedding model ...")
            from chromadb.utils import embedding_functions  # deferred

            self._fn = embedding_functions.ONNXMiniLM_L6_V2()
        return self._fn

    @property
    def dim(self) -> int:
        return 384

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.fn(texts)
        return [[float(x) for x in vec] for vec in vectors]


def get_embedder(settings: Settings | None = None) -> Embedder:
    """Return the embedder for the configured backend."""
    settings = settings or get_settings()
    backend = (settings.embed_backend or "sentence-transformers").lower()
    if backend == "onnx":
        return OnnxEmbedder(settings)
    return SentenceTransformerEmbedder(settings.embed_model, settings)
