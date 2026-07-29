"""Tests for citation parsing (fixture text) and answer assembly (fake client)."""

from __future__ import annotations

from datetime import datetime, timezone

from crypto_intel.config import Settings
from crypto_intel.models import Answer
from crypto_intel.retrieve import RetrievedChunk
from crypto_intel.synthesize import parse_citations, synthesize

BASE = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)


def _chunk(doc_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{doc_id}:0",
        doc_id=doc_id,
        text=text,
        source="news",
        source_name=f"Src-{doc_id}",
        url=f"https://x/{doc_id}",
        published_at=BASE,
        assets=["ETH"],
    )


CHUNKS = [
    _chunk("a", "Ethereum ETF outflows accelerated after the SEC filing."),
    _chunk("b", "Broad risk-off sentiment hit crypto markets."),
    _chunk("c", "A large validator exit was reported."),
]


# --- citation parsing (no API) ---------------------------------------------- #

def test_parse_citations_from_fixture_text():
    answer = (
        "ETH fell on ETF outflows [1] amid broad risk-off sentiment [2]. "
        "Some attribute it to a validator exit [1]."
    )
    cites = parse_citations(answer, CHUNKS)
    # [1] appears twice but resolves once; sorted ascending; [3] absent
    assert [c.n for c in cites] == [1, 2]
    c1 = next(c for c in cites if c.n == 1)
    assert c1.doc_id == "a"
    assert c1.url == "https://x/a"
    assert c1.published_at == BASE
    assert "Ethereum ETF outflows" in c1.snippet


def test_parse_citations_ignores_out_of_range_markers():
    # [4] and [9] don't map to any chunk -> dropped.
    cites = parse_citations("claim [4] other [9] real [2]", CHUNKS)
    assert [c.n for c in cites] == [2]


def test_parse_citations_none_present():
    assert parse_citations("no markers here at all", CHUNKS) == []


# --- synthesize() with a fake client (no API) ------------------------------- #

class _FakeBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.content = [_FakeBlock(text)]


class _FakeClient:
    """Records the call and returns a canned cited answer."""

    def __init__(self, text):
        self._text = text
        self.messages = self
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeResponse(self._text)


def test_synthesize_assembles_answer_with_citations():
    client = _FakeClient("ETH dropped on ETF outflows [1] and risk-off [2].")
    answer = synthesize(
        "Why did ETH drop today?",
        None,
        CHUNKS,
        settings=Settings(synth_model="claude-sonnet-5"),
        client=client,
    )
    assert isinstance(answer, Answer)
    assert "[1]" in answer.answer_text
    assert [c.n for c in answer.citations] == [1, 2]
    assert answer.retrieved_chunk_ids == [c.chunk_id for c in CHUNKS]
    # model + prompt were passed through
    assert client.last_kwargs["model"] == "claude-sonnet-5"
    assert "Numbered sources" in client.last_kwargs["messages"][0]["content"]


def test_synthesize_flags_thin_evidence():
    client = _FakeClient("Only one source says X [1].")
    answer = synthesize("Why?", None, CHUNKS[:1], settings=Settings(), client=client)
    assert any("thin evidence" in n for n in answer.notes)


def test_synthesize_no_chunks_short_circuits_without_client():
    # No client passed and no chunks -> must not attempt an API call.
    answer = synthesize("Why?", None, [], settings=Settings())
    assert answer.citations == []
    assert "no evidence in window" in answer.notes


def test_synthesize_flags_missing_citations():
    client = _FakeClient("A vague answer with no markers.")
    answer = synthesize("Why?", None, CHUNKS, settings=Settings(), client=client)
    assert any("no inline citations" in n for n in answer.notes)
