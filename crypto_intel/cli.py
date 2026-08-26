"""`crypto-intel` command-line interface.

Wires the four subcommands (``ingest``, ``stats``, ``price-event``, ``ask``) to
the pipeline. In S1 only ``stats`` is fully implemented; the others are stubs
that announce they are not yet available so ``--help`` lists the full surface.
"""

from __future__ import annotations

import logging
from datetime import timezone

import typer

from .config import get_settings
from .pipeline import ingest_all, prune_all, read_documents_summary
from .ingest.registry import VALID_SOURCES, parse_sources
from .store import get_store

app = typer.Typer(
    name="crypto-intel",
    help="Explain crypto price moves with citation-grounded evidence from public sources.",
    no_args_is_help=True,
    add_completion=False,
)

_NOT_YET = "This command is not implemented yet (planned in a later build phase)."


def _configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # Keep chatty HTTP libraries out of our logs even at -v.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _fmt_ts(dt) -> str:
    """Compact UTC timestamp for display."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


@app.command()
def stats() -> None:
    """Show per-source document/chunk counts and the ingest window."""
    settings = get_settings()

    doc_total, doc_per_source, (min_ts, max_ts) = read_documents_summary(
        settings.documents_file
    )

    store = get_store(settings)
    store_stats = store.stats()

    backend = (settings.store_backend or "chroma").lower()
    store_loc = settings.chroma_dir if backend == "chroma" else "(pgvector: DATABASE_URL)"

    typer.echo("Crypto Market Intelligence — store stats")
    typer.echo("=" * 44)
    typer.echo(f"Store       : {backend}  {store_loc}")
    typer.echo(f"Documents   : {settings.documents_file}")
    typer.echo("")
    typer.echo(f"Documents (total): {doc_total}")
    typer.echo(f"Chunks    (total): {store_stats.chunk_count}")
    typer.echo("")

    all_sources = sorted(set(doc_per_source) | set(store_stats.per_source_chunks))
    if all_sources:
        typer.echo(f"{'source':<12} {'docs':>8} {'chunks':>8}")
        typer.echo(f"{'-' * 12} {'-' * 8} {'-' * 8}")
        for src in all_sources:
            typer.echo(
                f"{src:<12} {doc_per_source.get(src, 0):>8} "
                f"{store_stats.per_source_chunks.get(src, 0):>8}"
            )
    else:
        typer.echo("(store is empty — run `crypto-intel ingest` to populate it)")

    typer.echo("")
    if min_ts and max_ts:
        typer.echo(f"Ingest window: {min_ts}  ->  {max_ts}")
    else:
        typer.echo("Ingest window: (none)")


@app.command()
def ingest(
    sources: str = typer.Option(
        None, "--sources", help="Comma-separated subset: news,reddit,exchange,regulator,cmc."
    ),
    all_sources: bool = typer.Option(
        False, "--all", help="Run every configured connector."
    ),
    lookback_hours: int = typer.Option(
        None, "--lookback-hours", help="How far back to fetch (defaults to config)."
    ),
    if_empty: bool = typer.Option(
        False, "--if-empty", help="Skip if the store already has content (for one-time initial populate)."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging."),
) -> None:
    """Ingest recent public content into the local store."""
    _configure_logging(verbose)
    settings = get_settings()

    if all_sources and sources:
        raise typer.BadParameter("Use either --all or --sources, not both.")
    # --all runs every configured connector (incl. opt-in Reddit); the default
    # set (no flag) excludes Reddit. Both fail soft on missing credentials.
    selected = set(VALID_SOURCES) if all_sources else parse_sources(sources)

    hours = lookback_hours or settings.default_lookback_hours
    scope = "all" if selected == set(VALID_SOURCES) else (
        "default" if selected is None else ", ".join(sorted(selected))
    )
    typer.echo(f"Ingesting sources=[{scope}] lookback={hours}h ...")

    try:
        result = ingest_all(
            sources=selected,
            lookback_hours=hours,
            settings=settings,
            skip_if_populated=if_empty,
        )
    except ImportError as exc:
        typer.echo("")
        typer.echo(f"Embedding backend unavailable: {exc}")
        typer.echo("Install it with:  pip install sentence-transformers")
        raise typer.Exit(code=1)

    if result.skipped:
        typer.echo("Store already populated — skipped (--if-empty).")
        return

    typer.echo("")
    typer.echo(
        f"Connectors run : {result.connectors}"
        + (f"  (errors: {result.errors})" if result.errors else "")
    )
    typer.echo(f"Fetched        : {result.fetched}")
    typer.echo(f"Added (new)    : {result.added}")
    typer.echo(f"Chunks added   : {result.chunks_added}")
    typer.echo(f"Duplicates     : {result.duplicates}")
    if result.per_source_added:
        typer.echo("Added by source:")
        for src, n in sorted(result.per_source_added.items()):
            typer.echo(f"  {src:<10} {n}")
    typer.echo("")
    typer.echo(f"Store: {settings.documents_file}")


@app.command("price-event")
def price_event(
    asset: str = typer.Option(..., "--asset", help="Ticker, e.g. ETH."),
    hours: int = typer.Option(24, "--hours", help="Analysis window in hours."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging."),
) -> None:
    """Confirm a price move directly (no retrieval)."""
    _configure_logging(verbose)
    settings = get_settings()

    from .prices import PriceError, detect_event

    try:
        event = detect_event(asset, hours, settings=settings)
    except PriceError as exc:
        typer.echo(f"Could not detect price event: {exc}")
        raise typer.Exit(code=1)

    arrow = {"up": "▲", "down": "▼", "flat": "▬"}[event.direction]
    dur = event.move_end - event.move_start
    dur_hours = dur.total_seconds() / 3600.0

    typer.echo(f"Price event — {event.asset} ({event.coingecko_id})")
    typer.echo("=" * 44)
    typer.echo(f"Window       : {_fmt_ts(event.window_start)}  ->  {_fmt_ts(event.window_end)}  ({hours}h)")
    typer.echo(f"Price        : ${event.price_start:,.2f}  ->  ${event.price_end:,.2f}")
    typer.echo(f"Net change   : {arrow} {event.pct_change:+.2f}%   ({event.direction})")
    typer.echo(f"Max drawdown : -{event.max_drawdown_pct:.2f}%")
    typer.echo("")
    typer.echo("Steepest move (drives retrieval window):")
    typer.echo(f"  {_fmt_ts(event.move_start)}  ->  {_fmt_ts(event.move_end)}  ({dur_hours:.1f}h)")


def _print_evidence(chunks) -> None:
    """Print ranked retrieved chunks with source + timestamp + snippet."""
    typer.echo(f"Retrieved evidence ({len(chunks)} chunk(s), best first):")
    typer.echo("-" * 60)
    for i, c in enumerate(chunks, 1):
        snippet = " ".join(c.text.split())
        if len(snippet) > 220:
            snippet = snippet[:220].rstrip() + "…"
        typer.echo(
            f"[{i}] {c.source_name}  ·  {_fmt_ts(c.published_at)}  ·  "
            f"score={c.score:.3f} (sem={c.semantic_score:.3f}, bm25={c.bm25_score:.2f})"
        )
        typer.echo(f"    {c.url}")
        typer.echo(f"    {snippet}")
        typer.echo("")


@app.command()
def ask(
    question: str = typer.Argument(..., help='e.g. "Why did ETH drop 6%% today?"'),
    asset: str = typer.Option(None, "--asset", help="Override the parsed ticker."),
    hours: int = typer.Option(None, "--hours", help="Retrieval window override."),
    k: int = typer.Option(None, "--k", help="Top-k chunks to retrieve."),
    sources: str = typer.Option(None, "--sources", help="Subset filter."),
    no_synth: bool = typer.Option(
        False, "--no-synth", help="Show retrieved evidence only; skip synthesis."
    ),
    no_regime: bool = typer.Option(
        False, "--no-regime", help="Skip the S12 volatility-regime context/banner."
    ),
    model: str = typer.Option(None, "--model", help="Override the synthesis model."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging."),
) -> None:
    """Ask why a price moved and get a grounded, cited answer."""
    _configure_logging(verbose)
    settings = get_settings()

    from .pipeline import retrieve_context, _maybe_forecast, _forecast_context

    ctx = retrieve_context(
        question,
        settings=settings,
        asset_override=asset,
        hours_override=hours,
        k=k,
        sources=parse_sources(sources),
    )

    typer.echo(f'Question : "{ctx.parsed.question}"')
    if ctx.parsed.asset:
        claimed = f", claimed {ctx.parsed.claimed_pct:g}%" if ctx.parsed.claimed_pct else ""
        typer.echo(f"Parsed   : asset={ctx.parsed.asset}, window={ctx.parsed.hours}h{claimed}")

    if ctx.event:
        e = ctx.event
        arrow = {"up": "▲", "down": "▼", "flat": "▬"}[e.direction]
        typer.echo(
            f"Event    : {arrow} {e.pct_change:+.2f}% ({e.direction}), "
            f"drawdown -{e.max_drawdown_pct:.2f}%, move "
            f"{_fmt_ts(e.move_start)} -> {_fmt_ts(e.move_end)}"
        )

    # S12: weave the current volatility regime in as market-state context.
    forecast = None if no_regime else _maybe_forecast(ctx.event, settings)
    if forecast is not None:
        glyph = {"calm": "○", "normal": "◐", "turbulent": "●"}[forecast.regime]
        skill = (
            f", skill {forecast.skill_vs_baseline:+.2f}"
            if forecast.skill_vs_baseline is not None else ""
        )
        typer.echo(
            f"Regime   : {glyph} {forecast.regime.upper()} "
            f"({forecast.predicted_vol_annualized:.0%} ann. vol, "
            f"model {forecast.model_name}{skill})"
        )
    typer.echo("")

    for note in ctx.notes:
        typer.echo(f"note: {note}")

    if not ctx.chunks:
        raise typer.Exit(code=1)

    if no_synth:
        typer.echo("")
        _print_evidence(ctx.chunks)
        typer.echo("Not investment advice.")
        return

    # Synthesis path.
    from .synthesize import SynthesisError, synthesize

    if not settings.anthropic_api_key:
        typer.echo("")
        typer.echo(
            "ANTHROPIC_API_KEY is not set — cannot synthesize an answer. "
            "Re-run with --no-synth to see the retrieved evidence."
        )
        raise typer.Exit(code=1)

    try:
        answer = synthesize(
            ctx.parsed.question, ctx.event, ctx.chunks, settings=settings, model=model,
            forecast_context=_forecast_context(forecast) if forecast else None,
        )
    except SynthesisError as exc:
        typer.echo(f"\nSynthesis failed: {exc}")
        raise typer.Exit(code=1)

    typer.echo("")
    typer.echo(answer.answer_text)
    typer.echo("")
    typer.echo("Sources")
    typer.echo("-" * 60)
    if answer.citations:
        for c in answer.citations:
            typer.echo(f"[{c.n}] {c.source_name}  ·  {_fmt_ts(c.published_at)}")
            typer.echo(f"    {c.url}")
    else:
        typer.echo("(no inline citations were produced)")
    for note in answer.notes:
        typer.echo(f"note: {note}")
    typer.echo("\nNot investment advice.")


@app.command("ask-db")
def ask_db_cmd(
    question: str = typer.Argument(..., help='e.g. "What happened to ETH within the last 1 week?"'),
    asset: str = typer.Option(None, "--asset", help="Override the parsed ticker."),
    hours: int = typer.Option(None, "--hours", help="Retrieval window override."),
    k: int = typer.Option(None, "--k", help="Top-k chunks to retrieve."),
    sources: str = typer.Option(None, "--sources", help="Subset filter."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging."),
) -> None:
    """Retrieve time-and-asset-windowed evidence from the pgvector DB (no LLM).

    The DB-backed counterpart of `ask --no-synth`; needs DATABASE_URL + the `pg`
    extra. This is what the web GUI uses against Supabase.
    """
    _configure_logging(verbose)
    settings = get_settings()

    from .pipeline import ask_db

    try:
        ctx = ask_db(
            question,
            settings=settings,
            asset_override=asset,
            hours_override=hours,
            k=k,
            sources=parse_sources(sources),
        )
    except RuntimeError as exc:  # missing DATABASE_URL / pg extra
        typer.echo(f"ask-db failed: {exc}")
        raise typer.Exit(code=1)

    typer.echo(f'Question : "{ctx.parsed.question}"')
    if ctx.parsed.asset:
        typer.echo(f"Parsed   : asset={ctx.parsed.asset}, window={ctx.parsed.hours}h (pgvector DB)")
    typer.echo("")
    for note in ctx.notes:
        typer.echo(f"note: {note}")
    if not ctx.chunks:
        raise typer.Exit(code=1)

    typer.echo("")
    _print_evidence(ctx.chunks)
    typer.echo("Not investment advice.")


def _fmt_pct(v: float | None) -> str:
    return f"{v * 100:.0f}%" if v is not None else "n/a"


@app.command("eval")
def eval_cmd(
    cases: str = typer.Option(
        None, "--cases", help="Path to eval cases JSON (defaults to data/eval_cases.json)."
    ),
    synth: bool = typer.Option(
        False, "--synth", help="Also synthesize answers and measure citation coverage (needs a key)."
    ),
    k: int = typer.Option(None, "--k", help="Top-k chunks per question."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging."),
) -> None:
    """Score retrieval (and optionally citations) over a small committed case set.

    A quality signal, not a benchmark — results depend on what was ingested.
    """
    _configure_logging(verbose)
    settings = get_settings()

    from pathlib import Path
    from .evaluate import load_eval_cases, run_eval

    cases_path = Path(cases) if cases else settings.eval_cases_file
    if not cases_path.exists():
        typer.echo(f"Eval cases file not found: {cases_path}")
        raise typer.Exit(code=1)

    case_list = load_eval_cases(cases_path)
    typer.echo(f"Evaluating {len(case_list)} case(s) from {cases_path.name} ...")
    report = run_eval(case_list, settings=settings, k=k, synth=synth)

    typer.echo("")
    typer.echo(f"{'question':<44} {'asset':>6} {'win':>4} {'ret':>4} {'src':>4} {'cite':>5}")
    typer.echo("-" * 72)
    for r in report.results:
        def mark(v):
            return "-" if v is None else ("ok" if v else "x")

        q = r.question if len(r.question) <= 44 else r.question[:41] + "..."
        cov = f"{r.citation_coverage*100:.0f}%" if r.citation_coverage is not None else "-"
        typer.echo(
            f"{q:<44} {mark(r.asset_ok):>6} {mark(r.window_ok):>4} "
            f"{r.n_retrieved:>4} {mark(r.source_hit):>4} {cov:>5}"
        )

    typer.echo("")
    typer.echo("Scorecard (quality signal, not a benchmark)")
    typer.echo("-" * 44)
    typer.echo(f"Asset parse accuracy    : {_fmt_pct(report.asset_accuracy)}")
    typer.echo(f"Window parse accuracy   : {_fmt_pct(report.window_accuracy)}")
    typer.echo(f"Retrieval non-empty rate: {_fmt_pct(report.retrieval_nonempty_rate)}")
    typer.echo(f"Expected-source hit rate: {_fmt_pct(report.source_hit_rate)}")
    typer.echo(f"Mean citation coverage  : {_fmt_pct(report.mean_citation_coverage)}")
    if not synth:
        typer.echo("(run with --synth + ANTHROPIC_API_KEY for citation coverage)")


@app.command()
def prune(
    keep_days: float = typer.Option(
        7.0, "--keep-days", help="Retention window in days; older content is deleted."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging."),
) -> None:
    """Delete stored content older than the retention window (rolling window)."""
    _configure_logging(verbose)
    settings = get_settings()

    result = prune_all(keep_days, settings)
    typer.echo(
        f"Retention: keeping last {keep_days:g} day(s) (cutoff {_fmt_ts(result.cutoff)})"
    )
    typer.echo(f"Chunks removed    : {result.chunks_removed}")
    typer.echo(f"Documents removed : {result.documents_removed}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(8000, "--port", help="Port."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload (dev)."),
) -> None:
    """Launch the web GUI for `price-event` (needs the `web` extra)."""
    try:
        import uvicorn
    except ImportError:
        typer.echo("Web dependencies not installed. Run:  pip install -e .[web]")
        raise typer.Exit(code=1)

    typer.echo(f"Serving the price-event GUI at http://{host}:{port}  (Ctrl+C to stop)")
    uvicorn.run("crypto_intel.web.app:app", host=host, port=port, reload=reload)


def _require_forecast():
    """Import the forecast subsystem, or exit with an install hint."""
    try:
        import numpy  # noqa: F401  — proxy for the `forecast` extra
    except ImportError:
        typer.echo("Forecast dependencies not installed. Run:  pip install -e .[forecast]")
        typer.echo("For boosted trees / the LSTM add:  pip install -e .[gbm,dl]")
        raise typer.Exit(code=1)


@app.command()
def train(
    asset: str = typer.Option(..., "--asset", help="Ticker, e.g. ETH."),
    history_days: int = typer.Option(None, "--history-days", help="Days of history to fetch."),
    lookback: int = typer.Option(None, "--lookback", help="Lookback window (hours)."),
    horizon: int = typer.Option(None, "--horizon", help="Forecast horizon (hours)."),
    stride: int = typer.Option(None, "--stride", help="Window stride (hours)."),
    models: str = typer.Option(
        "baseline,sklearn,xgboost,lstm", "--models",
        help="Comma list: baseline,sklearn,xgboost,lightgbm,lstm.",
    ),
    source: str = typer.Option(
        "coingecko", "--source",
        help="Where price data comes from: coingecko | warehouse | offline.",
    ),
    offline: str = typer.Option(
        None, "--offline", help="Train from a timestamp,price CSV (implies --source offline)."
    ),
    news: bool = typer.Option(
        False, "--news", help="Add exogenous news features from the document store."
    ),
    track: bool = typer.Option(
        False, "--track", help="Log the run to MLflow + register the best model (phase S11)."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging."),
) -> None:
    """Train + compare volatility-forecast models; persist the best (phase S9)."""
    _configure_logging(verbose)
    _require_forecast()
    settings = get_settings()

    from pathlib import Path
    from .forecast.dataset import load_series_csv, fetch_price_history, prepare_dataset, news_feature_fn
    from .forecast.train import train_models

    L = lookback or settings.forecast_lookback_hours
    H = horizon or settings.forecast_horizon_hours
    S = stride or settings.forecast_stride_hours
    days = history_days or settings.forecast_history_days
    model_list = [m.strip() for m in models.split(",") if m.strip()]

    news_fn = news_feature_fn(asset, settings) if news else None

    if offline or source == "offline":
        if not offline:
            typer.echo("--source offline requires --offline <csv>.")
            raise typer.Exit(code=1)
        typer.echo(f"Loading offline series from {offline} ...")
        series = load_series_csv(Path(offline))
    elif source == "warehouse":
        from .warehouse.duck import DuckWarehouse, warehouse_news_fn
        wh_path = settings.warehouse_file
        if not wh_path.exists():
            typer.echo(f"No warehouse at {wh_path}. Run:  crypto-intel warehouse build --asset {asset.upper()}")
            raise typer.Exit(code=1)
        typer.echo(f"Reading SQL-gridded {asset.upper()} series from warehouse {wh_path} ...")
        wh = DuckWarehouse(wh_path)
        series = wh.read_price_grid(asset)
        if news:
            news_fn = warehouse_news_fn(wh.news_counts_by_hour(asset))
        wh.close()
        if not series:
            typer.echo(f"No prices for {asset.upper()} in the warehouse.")
            raise typer.Exit(code=1)
    else:
        from .prices import PriceError
        typer.echo(f"Fetching {days}d of {asset.upper()} history from CoinGecko ...")
        try:
            series = fetch_price_history(asset, days, settings)
        except PriceError as exc:
            typer.echo(f"Could not fetch price history: {exc}")
            raise typer.Exit(code=1)
    X, y, seqs, names, idx, _, _ = prepare_dataset(series, L, H, S, news_fn=news_fn)
    typer.echo(f"Built {X.shape[0]} windows × {X.shape[1]} features (L={L}, H={H}, S={S}).")

    try:
        report = train_models(
            X, y, seqs, names, idx,
            asset=asset, lookback=L, horizon=H, stride=S,
            models=model_list,
            low_pct=settings.regime_low_pct, high_pct=settings.regime_high_pct,
            models_dir=settings.resolve_path(settings.models_path),
        )
    except (ValueError, RuntimeError) as exc:
        typer.echo(f"Training failed: {exc}")
        raise typer.Exit(code=1)

    _print_train_report(report, settings)

    if track:
        if "bundle_dir" not in report:
            typer.echo("Nothing persisted to track.")
        else:
            from .mlops.tracking import track_training
            info = track_training(report, report["bundle_dir"], settings)
            if info is None:
                typer.echo("MLflow tracking unavailable. Install:  pip install -e .[mlops]")
            else:
                typer.echo("")
                typer.echo(f"MLflow run   : {info['run_id']}  (exp: {info['experiment']})")
                typer.echo(f"Tracking URI : {info['tracking_uri']}")
                if info["registered"]:
                    typer.echo(f"Registered   : {info['model_name']}")
                else:
                    typer.echo("Registered   : (skipped — non-tabular best model)")


def _print_train_report(report: dict, settings) -> None:
    typer.echo("")
    typer.echo(f"Model comparison — {report['asset']}  (test split, skill vs. persistence baseline)")
    typer.echo("-" * 68)
    typer.echo(f"{'model':<12} {'MAE':>9} {'RMSE':>9} {'R2':>7} {'skill':>8} {'regimeF1':>9}")
    for name, res in report["models"].items():
        t = res["test"]
        mark = "  <- best" if name == report["best"] else ""
        typer.echo(
            f"{name:<12} {t['mae']:>9.5f} {t['rmse']:>9.5f} {t['r2']:>7.3f} "
            f"{t['skill']:>8.3f} {t['regime_f1']:>9.3f}{mark}"
        )
    for name, why in report.get("skipped", {}).items():
        typer.echo(f"{name:<12} skipped — {why}")
    typer.echo("")
    typer.echo(f"Best model : {report['best']}")
    if "bundle_dir" in report:
        typer.echo(f"Saved to   : {report['bundle_dir']}")
        # Write the full comparison next to the bundle for the record.
        import json
        from pathlib import Path
        metrics_path = Path(report["bundle_dir"]).parent / "metrics.json"
        metrics_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        typer.echo(f"Metrics    : {metrics_path}")


@app.command()
def predict(
    asset: str = typer.Option(..., "--asset", help="Ticker, e.g. ETH."),
    model: str = typer.Option(None, "--model", help="Pick a model if several are trained."),
    offline: str = typer.Option(
        None, "--offline", help="Predict from a timestamp,price CSV (no network)."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging."),
) -> None:
    """Forecast next-window volatility / risk regime for an asset (phase S9)."""
    _configure_logging(verbose)
    _require_forecast()
    settings = get_settings()

    from pathlib import Path
    from .forecast.predict import predict as run_predict
    from .forecast.dataset import load_series_csv

    series = load_series_csv(Path(offline)) if offline else None
    try:
        fc = run_predict(asset, model=model, settings=settings, series=series)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Could not forecast: {exc}")
        raise typer.Exit(code=1)

    regime_glyph = {"calm": "○", "normal": "◐", "turbulent": "●"}[fc.regime]
    typer.echo(f"Volatility forecast — {fc.asset}  (model: {fc.model_name})")
    typer.echo("=" * 52)
    typer.echo(f"As of         : {_fmt_ts(fc.as_of)}")
    typer.echo(f"Horizon       : next {fc.horizon_hours}h  (lookback {fc.lookback_hours}h)")
    typer.echo(f"Predicted vol : {fc.predicted_vol:.5f}  (annualized {fc.predicted_vol_annualized:.2%})")
    typer.echo(f"Risk regime   : {regime_glyph} {fc.regime.upper()}")
    if fc.skill_vs_baseline is not None:
        typer.echo(f"Model skill   : {fc.skill_vs_baseline:+.3f} vs. persistence baseline")
    if fc.drivers:
        top = ", ".join(f"{k} ({v:.2f})" for k, v in fc.drivers.items())
        typer.echo(f"Top drivers   : {top}")
    typer.echo("")
    for note in fc.notes:
        typer.echo(note)


@app.command()
def monitor(
    asset: str = typer.Option(..., "--asset", help="Ticker, e.g. ETH."),
    model: str = typer.Option(None, "--model", help="Pick a model if several are trained."),
    history_days: int = typer.Option(None, "--history-days", help="Days of recent data to score."),
    offline: str = typer.Option(None, "--offline", help="Score drift against a timestamp,price CSV."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging."),
) -> None:
    """Build an Evidently feature-drift report vs. the training reference (S11)."""
    _configure_logging(verbose)
    _require_forecast()
    settings = get_settings()

    from pathlib import Path
    from .forecast.predict import bundle_dir_for
    from .forecast.models import load_bundle
    from .forecast.dataset import load_series_csv, fetch_price_history
    from .mlops.monitor import load_reference, compute_current_features, build_drift_report

    try:
        bundle = bundle_dir_for(asset, model, settings)
        ref_X, ref_names = load_reference(bundle)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Could not load model/reference: {exc}")
        raise typer.Exit(code=1)
    _, meta = load_bundle(bundle)

    if offline:
        series = load_series_csv(Path(offline))
    else:
        from .prices import PriceError
        days = history_days or settings.forecast_history_days
        typer.echo(f"Fetching {days}d of recent {asset.upper()} data from CoinGecko ...")
        try:
            series = fetch_price_history(asset, days, settings)
        except PriceError as exc:
            typer.echo(f"Could not fetch price history: {exc}")
            raise typer.Exit(code=1)

    cur_X, cur_names = compute_current_features(series, meta)

    # Align on the columns both matrices share (reference may carry news features).
    common = [n for n in ref_names if n in cur_names]
    ref_idx = [ref_names.index(n) for n in common]
    cur_idx = [cur_names.index(n) for n in common]
    ref_al = ref_X[:, ref_idx]
    cur_al = cur_X[:, cur_idx]
    if cur_al.shape[0] < 2:
        typer.echo("Not enough recent windows to assess drift; widen --history-days.")
        raise typer.Exit(code=1)

    out_html = settings.resolve_path(settings.monitoring_path) / f"{asset.upper()}.html"
    try:
        summary = build_drift_report(ref_al, cur_al, common, out_html)
    except ImportError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1)

    typer.echo(f"Drift report — {asset.upper()}")
    typer.echo("=" * 44)
    typer.echo(f"Reference windows : {ref_al.shape[0]}")
    typer.echo(f"Current windows   : {cur_al.shape[0]}")
    typer.echo(f"Features compared : {summary['n_features']}")
    if summary.get("drift_summary"):
        for k, v in summary["drift_summary"].items():
            typer.echo(f"  {k}: {v}")
    typer.echo(f"Report            : {summary['out_html']}")
    typer.echo("Serve it at /monitoring via:  crypto-intel serve")


# --------------------------------------------------------------------------- #
# Warehouse (phase S10)                                                       #
# --------------------------------------------------------------------------- #

warehouse_app = typer.Typer(help="Warehouse-backed feature pipeline (DuckDB / BigQuery).")
app.add_typer(warehouse_app, name="warehouse")


def _require_warehouse():
    try:
        import duckdb  # noqa: F401
    except ImportError:
        typer.echo("Warehouse dependencies not installed. Run:  pip install -e .[warehouse]")
        raise typer.Exit(code=1)


@warehouse_app.command("build")
def warehouse_build(
    asset: str = typer.Option(..., "--asset", help="Ticker, e.g. ETH."),
    history_days: int = typer.Option(None, "--history-days", help="Days of price history."),
    offline: str = typer.Option(None, "--offline", help="Load prices from a CSV instead of CoinGecko."),
    dest: str = typer.Option("duckdb", "--dest", help="Destination: duckdb | bigquery."),
    rolling_hours: int = typer.Option(24, "--rolling-hours", help="Rolling feature window (hours)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging."),
) -> None:
    """Land prices + document metadata and engineer features in SQL."""
    _configure_logging(verbose)
    _require_warehouse()
    settings = get_settings()

    from pathlib import Path
    from .forecast.dataset import load_series_csv, fetch_price_history
    from .pipeline import iter_documents

    days = history_days or settings.forecast_history_days
    if offline:
        typer.echo(f"Loading offline series from {offline} ...")
        series = load_series_csv(Path(offline))
    else:
        from .prices import PriceError
        typer.echo(f"Fetching {days}d of {asset.upper()} history from CoinGecko ...")
        try:
            series = fetch_price_history(asset, days, settings)
        except PriceError as exc:
            typer.echo(f"Could not fetch price history: {exc}")
            raise typer.Exit(code=1)

    documents = list(iter_documents(settings.documents_file))

    if dest == "bigquery":
        if not settings.bq_project:
            typer.echo("Set BQ_PROJECT (and GOOGLE_APPLICATION_CREDENTIALS) for --dest bigquery.")
            raise typer.Exit(code=1)
        try:
            from .warehouse.bq import load_to_bigquery
            summary = load_to_bigquery(
                settings.bq_project, settings.bq_dataset, asset, series, documents
            )
        except (ImportError, RuntimeError) as exc:
            typer.echo(f"BigQuery load failed: {exc}")
            raise typer.Exit(code=1)
        typer.echo(f"Loaded to BigQuery {summary['dataset']}: "
                   f"{summary['prices']} prices, {summary['documents']} documents.")
        return

    from .warehouse.duck import DuckWarehouse
    wh_path = settings.warehouse_file
    wh = DuckWarehouse(wh_path)
    n_prices = wh.load_prices(asset, series)
    n_docs = wh.load_documents(asset, documents)
    n_feats = wh.build_features(asset, rolling_hours=rolling_hours)
    wh.close()
    typer.echo(f"Warehouse {wh_path}")
    typer.echo(f"  prices loaded        : {n_prices}")
    typer.echo(f"  documents loaded     : {n_docs}")
    typer.echo(f"  SQL feature rows     : {n_feats}  (rolling {rolling_hours}h)")
    typer.echo(f"Train from it with:  crypto-intel train --asset {asset.upper()} --source warehouse")


@warehouse_app.command("stats")
def warehouse_stats(
    asset: str = typer.Option(None, "--asset", help="Restrict to one ticker."),
) -> None:
    """Show warehouse row counts + coverage window."""
    _require_warehouse()
    settings = get_settings()
    wh_path = settings.warehouse_file
    if not wh_path.exists():
        typer.echo(f"No warehouse at {wh_path}. Run:  crypto-intel warehouse build --asset ETH")
        raise typer.Exit(code=1)

    from .warehouse.duck import DuckWarehouse
    wh = DuckWarehouse(wh_path)
    s = wh.stats(asset)
    wh.close()
    typer.echo(f"Warehouse {wh_path}  ({s['asset']})")
    typer.echo(f"  prices    : {s['prices']}")
    typer.echo(f"  documents : {s['documents']}")
    typer.echo(f"  window    : {s['from']}  ->  {s['to']}")


def _force_utf8_output() -> None:
    """Ensure stdout/stderr use UTF-8 so glyphs survive piping on Windows.

    Redirected output on Windows defaults to the cp1252 'charmap' codec, which
    can't encode arrows/box-drawing chars and would crash the command.
    """
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main() -> None:
    _force_utf8_output()
    app()


if __name__ == "__main__":
    main()
