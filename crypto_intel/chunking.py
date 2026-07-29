"""Split documents into overlapping, word-count-based chunks.

Word counts (not model tokens) keep chunking deterministic and dependency-free.
Defaults come from config (220 words / 40 overlap). Chunk ids are stable —
``f"{doc_id}:{chunk_index}"`` — so re-embedding a document upserts over its old
chunks rather than duplicating them.
"""

from __future__ import annotations

from .models import Chunk, Document


def chunk_text(text: str, chunk_size: int = 220, overlap: int = 40) -> list[str]:
    """Split ``text`` into overlapping word windows.

    Returns ``[]`` for empty text and a single chunk when the text is shorter
    than ``chunk_size``.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]

    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_document(
    doc: Document, chunk_size: int = 220, overlap: int = 40
) -> list[Chunk]:
    """Chunk a :class:`Document`, carrying its metadata onto each chunk."""
    pieces = chunk_text(doc.text, chunk_size=chunk_size, overlap=overlap)
    return [
        Chunk(
            id=f"{doc.id}:{i}",
            doc_id=doc.id,
            chunk_index=i,
            text=piece,
            source=doc.source,
            source_name=doc.source_name,
            url=doc.url,
            published_at=doc.published_at,
            assets=list(doc.assets),
        )
        for i, piece in enumerate(pieces)
    ]
