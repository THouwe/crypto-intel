"""Tests for the S12 RAG↔forecast stitch (regime woven into `ask`)."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from crypto_intel.config import Settings
from crypto_intel.models import Answer, VolForecast
from crypto_intel import synthesize as S
from crypto_intel import pipeline

UTC = timezone.utc


def _fake_forecast(regime="turbulent"):
    return VolForecast(
        asset="ETH", as_of=datetime(2026, 8, 20, tzinfo=UTC),
        lookback_hours=72, horizon_hours=24,
        predicted_vol=0.05, predicted_vol_annualized=0.68,
        regime=regime, model_name="xgboost", skill_vs_baseline=0.9,
        notes=["Not investment advice."],
    )


class _RecordingClient:
    """Stub Anthropic client that records the prompt and returns fixed text."""

    def __init__(self):
        self.captured = None
        self.messages = self

    def create(self, **kwargs):
        self.captured = kwargs
        block = SimpleNamespace(type="text", text="Reports link the move to an upgrade [1].")
        return SimpleNamespace(content=[block])


# --- prompt weaving --------------------------------------------------------- #

def test_forecast_context_string_is_background_not_advice():
    ctx = pipeline._forecast_context(_fake_forecast())
    assert "ETH" in ctx and "TURBULENT" in ctx
    assert "not a price prediction" in ctx.lower()


def test_build_user_prompt_includes_forecast_context():
    prompt = S.build_user_prompt("Why did ETH move?", None, [], forecast_context="REGIME LINE HERE")
    assert "REGIME LINE HERE" in prompt


def test_system_prompt_has_regime_guardrail():
    assert "volatility regime" in S.SYSTEM_PROMPT.lower()
    assert "not a source" in S.SYSTEM_PROMPT.lower() or "do not cite" in S.SYSTEM_PROMPT.lower()


def test_synthesize_passes_forecast_context_to_model():
    chunk = SimpleNamespace(
        text="ETH upgrade shipped.", source_name="CoinDesk",
        url="https://x/1", published_at=datetime(2026, 8, 20, tzinfo=UTC),
        chunk_id="c1", doc_id="d1",
    )
    client = _RecordingClient()
    ans = S.synthesize(
        "Why did ETH move?", None, [chunk], settings=Settings(),
        client=client, forecast_context="Current volatility regime for ETH: TURBULENT ...",
    )
    sent = client.captured["messages"][0]["content"]
    assert "Current volatility regime for ETH: TURBULENT" in sent
    # The regime is context, not a citable source: only [1] resolves to a chunk.
    assert [c.n for c in ans.citations] == [1]


# --- pipeline.ask attaches market_state ------------------------------------- #

def _patch_ask(monkeypatch, forecast):
    fake_ctx = SimpleNamespace(
        event=SimpleNamespace(asset="ETH"),
        chunks=[SimpleNamespace(chunk_id="c1")],
        notes=["retrieval-note"],
    )
    monkeypatch.setattr(pipeline, "retrieve_context", lambda *a, **k: fake_ctx)
    monkeypatch.setattr(pipeline, "_maybe_forecast", lambda event, settings, client=None: forecast)

    captured = {}

    def fake_synth(q, event, chunks, settings=None, model=None, forecast_context=None):
        captured["forecast_context"] = forecast_context
        return Answer(question=q, answer_text="ok", notes=["synth-note"])

    monkeypatch.setattr("crypto_intel.synthesize.synthesize", fake_synth)
    return captured


def test_ask_attaches_market_state_when_model_present(monkeypatch):
    fc = _fake_forecast("calm")
    captured = _patch_ask(monkeypatch, fc)

    ans = pipeline.ask("why did ETH move?", settings=Settings())
    assert ans.market_state == {
        "regime": "calm",
        "model_name": "xgboost",
        "skill_vs_baseline": 0.9,
        "predicted_vol_annualized": 0.68,
    }
    assert captured["forecast_context"] and "ETH" in captured["forecast_context"]
    # retrieval + synth notes both merged.
    assert "retrieval-note" in ans.notes and "synth-note" in ans.notes


def test_ask_without_regime_skips_forecast(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("_maybe_forecast should not be called with with_regime=False")

    captured = _patch_ask(monkeypatch, None)
    monkeypatch.setattr(pipeline, "_maybe_forecast", _boom)

    ans = pipeline.ask("why?", settings=Settings(), with_regime=False)
    assert ans.market_state is None
    assert captured["forecast_context"] is None


def test_maybe_forecast_none_when_no_event():
    assert pipeline._maybe_forecast(None, Settings()) is None
