"""Grounded answer synthesis with inline citations (Claude API).

Builds a prompt from the confirmed :class:`PriceEvent` plus numbered evidence
chunks, asks Claude for a concise explanation that cites sources with ``[n]``
markers, then parses those markers back into :class:`Citation`s. The parsing
step is pure (no API call) so it's unit-testable from a fixture answer.

Guardrail: the prompt forbids buy/sell/hold advice and price targets — this is
decision-support, not financial advice.
"""

from __future__ import annotations

import logging
import re

from .config import Settings, get_settings
from .models import Answer, Citation, PriceEvent
from .retrieve import RetrievedChunk

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[(\d+)\]")
_SNIPPET_CHARS = 240


class SynthesisError(RuntimeError):
    """Raised when synthesis can't run (e.g. missing API key)."""


SYSTEM_PROMPT = (
    "You are a crypto market intelligence assistant. Your job is to explain WHY an "
    "asset moved, grounded ONLY in the numbered sources you are given.\n\n"
    "Rules:\n"
    "- Use only the numbered context below. Do not use outside knowledge and do not "
    "invent facts, numbers, or sources.\n"
    "- Attribute the move descriptively: 'reports link the move to X [1]', "
    "'an exchange notice cited Y [2]'. Every factual claim must carry an inline "
    "citation like [1] or [2] pointing to the specific source(s) that support it.\n"
    "- Only cite numbers that actually appear in the provided context.\n"
    "- Be concise: a short paragraph or a few bullet points.\n"
    "- If the evidence is thin, off-topic, or does not actually explain the move, say "
    "so plainly (e.g. 'the available sources do not clearly explain this move') rather "
    "than forcing an explanation.\n"
    "- This is decision-support, NOT financial advice. Do not give buy/sell/hold "
    "recommendations, price targets, or personalized advice."
)


def _event_facts(event: PriceEvent | None) -> str:
    if event is None:
        return "Price move could not be confirmed from market data."
    return (
        f"Confirmed price move for {event.asset} ({event.coingecko_id}): "
        f"{event.pct_change:+.2f}% over the window "
        f"{event.window_start.isoformat()} to {event.window_end.isoformat()} "
        f"(direction: {event.direction}). Max drawdown {event.max_drawdown_pct:.2f}%. "
        f"Steepest move {event.move_start.isoformat()} to {event.move_end.isoformat()}."
    )


def _numbered_context(chunks: list[RetrievedChunk]) -> str:
    lines: list[str] = []
    for i, c in enumerate(chunks, 1):
        text = " ".join(c.text.split())
        lines.append(
            f"[{i}] source: {c.source_name} | published: {c.published_at.isoformat()} | "
            f"url: {c.url}\n{text}"
        )
    return "\n\n".join(lines)


def build_user_prompt(
    question: str, event: PriceEvent | None, chunks: list[RetrievedChunk]
) -> str:
    return (
        f"Question: {question}\n\n"
        f"Market context:\n{_event_facts(event)}\n\n"
        f"Numbered sources (cite these with [n]):\n{_numbered_context(chunks)}\n\n"
        "Write the grounded explanation now, using [n] citations."
    )


def parse_citations(answer_text: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    """Resolve the ``[n]`` markers in ``answer_text`` to :class:`Citation`s.

    Only markers that reference a real numbered chunk (1..len) are kept, each once,
    in ascending order.
    """
    numbers = sorted({int(m) for m in _CITATION_RE.findall(answer_text)})
    citations: list[Citation] = []
    for n in numbers:
        if 1 <= n <= len(chunks):
            c = chunks[n - 1]
            snippet = " ".join(c.text.split())[:_SNIPPET_CHARS]
            citations.append(
                Citation(
                    n=n,
                    doc_id=c.doc_id,
                    source_name=c.source_name,
                    url=c.url,
                    published_at=c.published_at,
                    snippet=snippet,
                )
            )
    return citations


def _extract_text(response) -> str:
    parts: list[str] = []
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


def synthesize(
    question: str,
    event: PriceEvent | None,
    chunks: list[RetrievedChunk],
    settings: Settings | None = None,
    model: str | None = None,
    client=None,
) -> Answer:
    """Synthesize a grounded, cited :class:`Answer` from retrieved evidence.

    ``client`` (anything exposing ``messages.create``) is injectable for tests.
    Short-circuits without an API call when there is no evidence.
    """
    settings = settings or get_settings()
    model = model or settings.synth_model
    retrieved_ids = [c.chunk_id for c in chunks]

    if not chunks:
        return Answer(
            question=question,
            event=event,
            answer_text=(
                "No sources were found in the analysed window, so this move can't be "
                "explained from the ingested evidence. Try a wider --hours or ingest "
                "more sources."
            ),
            citations=[],
            retrieved_chunk_ids=[],
            notes=["no evidence in window"],
        )

    if client is None:
        if not settings.anthropic_api_key:
            raise SynthesisError(
                "ANTHROPIC_API_KEY is not set; cannot synthesize. Use --no-synth to see "
                "retrieved evidence only."
            )
        import anthropic  # deferred so offline commands/tests don't need the SDK

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    response = client.messages.create(
        model=model,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        thinking={"type": "disabled"},  # concise, grounded output; no reasoning spend
        messages=[{"role": "user", "content": build_user_prompt(question, event, chunks)}],
    )

    answer_text = _extract_text(response).strip()
    citations = parse_citations(answer_text, chunks)

    notes: list[str] = []
    if len(chunks) < 3:
        notes.append("thin evidence: few sources in the window")
    if not citations:
        notes.append("no inline citations were produced")

    return Answer(
        question=question,
        event=event,
        answer_text=answer_text,
        citations=citations,
        retrieved_chunk_ids=retrieved_ids,
        notes=notes,
    )
