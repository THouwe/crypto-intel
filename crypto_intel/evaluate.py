"""Lightweight evaluation over a small committed set of cases.

A **quality signal, not a benchmark.** For each case in ``data/eval_cases.json``
it checks the retrieval side (did question parsing recover the expected asset and
time window? did retrieval return anything? did an expected source appear?) and,
optionally (with a key), the citation-coverage of the synthesized answer (fraction
of answer sentences carrying a ``[n]`` marker).

The scoring functions (:func:`score_retrieval`, :func:`citation_coverage`) are
pure and unit-tested offline; :func:`run_eval` wires them over the live store.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings, get_settings
from .retrieve import RetrievedChunk

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CITATION_RE = re.compile(r"\[\d+\]")


@dataclass
class EvalCase:
    question: str
    expected_asset: str | None = None
    expected_window_hours: int | None = None
    expected_source_contains: str | None = None


@dataclass
class CaseResult:
    question: str
    parsed_asset: str | None
    parsed_hours: int
    asset_ok: bool | None
    window_ok: bool | None
    n_retrieved: int
    source_hit: bool | None
    citation_coverage: float | None = None


@dataclass
class EvalReport:
    results: list[CaseResult] = field(default_factory=list)

    def _rate(self, values: list[bool]) -> float | None:
        return (sum(values) / len(values)) if values else None

    @property
    def asset_accuracy(self) -> float | None:
        return self._rate([r.asset_ok for r in self.results if r.asset_ok is not None])

    @property
    def window_accuracy(self) -> float | None:
        return self._rate([r.window_ok for r in self.results if r.window_ok is not None])

    @property
    def retrieval_nonempty_rate(self) -> float | None:
        return self._rate([r.n_retrieved > 0 for r in self.results]) if self.results else None

    @property
    def source_hit_rate(self) -> float | None:
        return self._rate([r.source_hit for r in self.results if r.source_hit is not None])

    @property
    def mean_citation_coverage(self) -> float | None:
        covs = [r.citation_coverage for r in self.results if r.citation_coverage is not None]
        return (sum(covs) / len(covs)) if covs else None


def load_eval_cases(path: Path) -> list[EvalCase]:
    """Load eval cases from a JSON list file."""
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return [
        EvalCase(
            question=item["question"],
            expected_asset=item.get("expected_asset"),
            expected_window_hours=item.get("expected_window_hours"),
            expected_source_contains=item.get("expected_source_contains"),
        )
        for item in raw
    ]


def citation_coverage(answer_text: str) -> float:
    """Fraction of answer sentences that carry a ``[n]`` citation marker."""
    sentences = [s for s in _SENTENCE_SPLIT.split(answer_text.strip()) if s.strip()]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if _CITATION_RE.search(s))
    return cited / len(sentences)


def score_retrieval(
    case: EvalCase,
    parsed_asset: str | None,
    parsed_hours: int,
    chunks: list[RetrievedChunk],
) -> dict:
    """Pure per-case retrieval scoring (no network/store)."""
    asset_ok = (
        parsed_asset == case.expected_asset if case.expected_asset is not None else None
    )
    window_ok = (
        parsed_hours == case.expected_window_hours
        if case.expected_window_hours is not None
        else None
    )
    source_hit: bool | None = None
    if case.expected_source_contains:
        token = case.expected_source_contains.lower()
        source_hit = any(
            token in f"{c.url} {c.source_name}".lower() for c in chunks
        )
    return {
        "asset_ok": asset_ok,
        "window_ok": window_ok,
        "n_retrieved": len(chunks),
        "source_hit": source_hit,
    }


def run_eval(
    cases: list[EvalCase],
    settings: Settings | None = None,
    k: int | None = None,
    synth: bool = False,
) -> EvalReport:
    """Evaluate each case against the live store (retrieval; optional synthesis).

    Retrieval uses the parsed time window directly (``detect_prices=False``) so a
    multi-case sweep doesn't hammer CoinGecko. Synthesis runs only when ``synth``
    is set and an API key is available.
    """
    settings = settings or get_settings()
    from .pipeline import retrieve_context  # deferred to keep --help fast

    report = EvalReport()
    for case in cases:
        ctx = retrieve_context(
            case.question, settings=settings, k=k, detect_prices=False
        )
        scores = score_retrieval(
            case, ctx.parsed.asset, ctx.parsed.hours, ctx.chunks
        )
        coverage: float | None = None
        if synth and ctx.chunks and settings.anthropic_api_key:
            from .synthesize import synthesize

            answer = synthesize(
                case.question, ctx.event, ctx.chunks, settings=settings
            )
            coverage = citation_coverage(answer.answer_text)

        report.results.append(
            CaseResult(
                question=case.question,
                parsed_asset=ctx.parsed.asset,
                parsed_hours=ctx.parsed.hours,
                asset_ok=scores["asset_ok"],
                window_ok=scores["window_ok"],
                n_retrieved=scores["n_retrieved"],
                source_hit=scores["source_hit"],
                citation_coverage=coverage,
            )
        )
    return report
