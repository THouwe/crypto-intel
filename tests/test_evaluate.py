"""Tests for the eval metrics (pure functions) and case loading."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from crypto_intel.evaluate import (
    EvalCase,
    EvalReport,
    CaseResult,
    citation_coverage,
    load_eval_cases,
    score_retrieval,
)
from crypto_intel.retrieve import RetrievedChunk

BASE = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)


def _chunk(doc_id: str, url: str, source_name: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{doc_id}:0",
        doc_id=doc_id,
        text="body",
        source="news",
        source_name=source_name,
        url=url,
        published_at=BASE,
        assets=["ETH"],
    )


# --- citation_coverage ------------------------------------------------------ #

def test_citation_coverage_all_sentences_cited():
    text = "ETH fell on outflows [1]. Sentiment was bearish [2]."
    assert citation_coverage(text) == pytest.approx(1.0)


def test_citation_coverage_partial():
    text = "ETH fell on outflows [1]. This sentence has no citation. Also this one."
    # 1 of 3 sentences carries a marker
    assert citation_coverage(text) == pytest.approx(1 / 3)


def test_citation_coverage_none():
    assert citation_coverage("No citations at all here. Nope.") == 0.0


def test_citation_coverage_empty():
    assert citation_coverage("   ") == 0.0


# --- score_retrieval -------------------------------------------------------- #

def test_score_retrieval_asset_and_window_ok():
    case = EvalCase("Why did ETH drop today?", expected_asset="ETH", expected_window_hours=24)
    scores = score_retrieval(case, "ETH", 24, [_chunk("a", "https://x/a", "Src")])
    assert scores["asset_ok"] is True
    assert scores["window_ok"] is True
    assert scores["n_retrieved"] == 1
    assert scores["source_hit"] is None  # no expected_source_contains


def test_score_retrieval_mismatch():
    case = EvalCase("q", expected_asset="BTC", expected_window_hours=168)
    scores = score_retrieval(case, "ETH", 24, [])
    assert scores["asset_ok"] is False
    assert scores["window_ok"] is False
    assert scores["n_retrieved"] == 0


def test_score_retrieval_source_hit_matches_url_or_name():
    case = EvalCase("q", expected_source_contains="beincrypto")
    chunks = [
        _chunk("a", "https://coindesk.com/x", "CoinDesk"),
        _chunk("b", "https://beincrypto.com/eth", "BeInCrypto"),
    ]
    assert score_retrieval(case, "ETH", 24, chunks)["source_hit"] is True


def test_score_retrieval_source_miss():
    case = EvalCase("q", expected_source_contains="nonexistent-outlet")
    chunks = [_chunk("a", "https://coindesk.com/x", "CoinDesk")]
    assert score_retrieval(case, "ETH", 24, chunks)["source_hit"] is False


def test_score_retrieval_no_expectations_are_none():
    case = EvalCase("q")  # nothing expected
    scores = score_retrieval(case, "ETH", 24, [])
    assert scores["asset_ok"] is None and scores["window_ok"] is None


# --- load_eval_cases -------------------------------------------------------- #

def test_load_eval_cases(tmp_path):
    p = tmp_path / "cases.json"
    p.write_text(
        json.dumps(
            [
                {"question": "Why did ETH drop?", "expected_asset": "ETH", "expected_window_hours": 24},
                {"question": "BTC?"},
            ]
        ),
        encoding="utf-8",
    )
    cases = load_eval_cases(p)
    assert len(cases) == 2
    assert cases[0].expected_asset == "ETH"
    assert cases[1].expected_asset is None


def test_load_shipped_eval_cases():
    from crypto_intel.config import Settings

    cases = load_eval_cases(Settings().eval_cases_file)
    assert len(cases) >= 3
    assert all(c.question for c in cases)


# --- EvalReport aggregates -------------------------------------------------- #

def test_eval_report_aggregates():
    report = EvalReport(
        results=[
            CaseResult("q1", "ETH", 24, True, True, 3, True, 1.0),
            CaseResult("q2", "BTC", 24, False, True, 0, None, None),
            CaseResult("q3", "SOL", 24, True, False, 2, None, 0.5),
        ]
    )
    assert report.asset_accuracy == pytest.approx(2 / 3)
    assert report.window_accuracy == pytest.approx(2 / 3)
    assert report.retrieval_nonempty_rate == pytest.approx(2 / 3)
    assert report.source_hit_rate == pytest.approx(1.0)  # only q1 had an expectation
    assert report.mean_citation_coverage == pytest.approx(0.75)  # (1.0 + 0.5)/2
