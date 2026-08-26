# ML expansion — design spec & roadmap (S9–S12)

The design reference for extending the Crypto Market Intelligence Assistant from a
citation-grounded RAG explainer into an **end-to-end ML system with a full MLOps
loop**. Companion to [ARCHITECTURE.md](ARCHITECTURE.md) (the RAG core, phases
S1–S8); this doc covers the new phases **S9–S12**. For usage see the
[README](../README.md); for deployment see [DEPLOYMENT.md](DEPLOYMENT.md).

> **Status:** **S9–S12 built** — the full ML expansion is complete (see § 4–7 and
> the [ARCHITECTURE build history](ARCHITECTURE.md#build-history-phases)).

---

## 1. Why extend it, and the framing constraint

The RAG core already demonstrates GenAI depth (embeddings, two vector-DB backends,
BM25 rerank, LLM synthesis, an eval scorecard, a live deployment). What it does not
yet show is the **modelling → training → serving → monitoring** loop that
industry data-science teams screen for: PyTorch / gradient-boosted trees,
experiment tracking, a model registry, CI/CD for models, drift monitoring, and a
warehouse-backed feature pipeline. S9–S12 add exactly that, **reusing the existing
crypto domain and deployment** so the result is one coherent system rather than two
half-projects.

### 1.1 The guardrail is load-bearing — predict volatility, not price

The RAG core's identity is *grounded explanation, explicitly **not** prediction or
investment advice*: `ask` never emits buy/sell/hold calls or price targets, and
every answer ends with a "not investment advice" line (see
[ARCHITECTURE.md § Guardrail](ARCHITECTURE.md#guardrail)).

The ML layer must not break that wall. Therefore the model's target is **not price
direction**. It forecasts **next-window realized volatility** and, derived from it,
a **risk regime** (`calm` / `normal` / `turbulent`). This is legitimate,
defensible quant work (volatility is forecastable in a way price direction is not),
and it *feeds* the explainer rather than competing with it: the S12 stitch lets a
cited answer say *how turbulent* the current market is when interpreting a move.

Every forecast output carries the same "not investment advice" line. No phase adds
order routing, portfolio logic, or a directional price call.

---

## 2. Scope (S9–S12)

**In scope**

- A **volatility / risk-regime forecasting** subsystem: feature engineering from the
  existing CoinGecko price series (and, optionally, from the news corpus), a
  supervised training pipeline, and a `predict` command returning a `VolForecast`.
- A **model zoo** with a unified interface and an honest baseline: a persistence
  baseline, a scikit-learn model, gradient-boosted trees (XGBoost / LightGBM), and a
  PyTorch sequence model (LSTM). Models are compared on a temporal split.
- A **warehouse-backed feature pipeline**: land price history + document metadata in
  **DuckDB** (embedded default) with an optional **BigQuery** free-tier loader;
  feature engineering expressible as SQL.
- A **full MLOps loop**: MLflow experiment tracking + model registry, model serving
  via the existing FastAPI app, GitHub Actions CI (test + lint) and a retrain
  workflow, and an Evidently drift-monitoring report.
- A **GenAI stitch**: inject the current regime/confidence into `ask` synthesis so
  the two subsystems form one product.
- Offline, deterministic **tests** for every new pure-logic unit, consistent with the
  existing suite.

**Out of scope**

- Directional price prediction, trading signals, order routing, portfolio/PnL logic,
  or any personalized financial advice (see § 1.1).
- Real-time / streaming inference. Training and prediction are on-demand commands
  (or a scheduled retrain); serving is request/response.
- A research-grade market dataset or tick data. Data granularity is bounded by the
  CoinGecko free tier (see § 8, Risk R1).
- GPU training or distributed training. Models are small and CPU-trainable.
- Automated model *promotion* to production without a human. CI retrains and logs;
  promotion to the "serving" alias is a manual, reviewed step.

---

## 3. Phase map

| Phase | Theme | Closes CV gap | Deliverable (one line) |
|---|---|---|---|
| **S9**  | **Modelling** | PyTorch · XGBoost/LightGBM · scikit-learn | `crypto-intel train` / `predict`: a `forecast/` subsystem that trains and compares models for next-window realized-vol / regime. **Built first.** |
| **S10** | **Scale**     | Warehouse · real feature eng | `crypto-intel warehouse build`: land prices + doc metadata in DuckDB (+ optional BigQuery); SQL feature pipeline feeding S9. |
| **S11** | **MLOps**     | Tracking · registry · CI/CD · monitoring | MLflow tracking+registry around `train`; FastAPI `/forecast`; GitHub Actions CI + retrain; Evidently drift report at `/monitoring`. |
| **S12** | **GenAI stitch** | Makes the RAG↔ML link explicit | Regime/confidence injected into `ask` synthesis; stack documented explicitly. |

Each phase is self-contained and leaves the repo green, mirroring the S1–S8 style.

---

## 4. S9 — Modelling subsystem (detailed spec)

The one phase specified to build-ready depth. S10–S12 are specified at roadmap
depth in § 5–7.

### 4.1 Problem definition

- **Grid.** Resample the raw CoinGecko series to a regular **hourly** grid
  (forward-fill / linear-interpolate small gaps). All windows are counted in hourly
  samples.
- **Log returns.** `r_t = ln(p_t / p_{t-1})`.
- **Realized volatility (RV) over a set of H hourly returns.**
  `RV = sqrt( Σ r_i² )` over the H returns in the window (H-hour realized vol).
  A human-readable **annualized** figure `RV_ann = RV · sqrt(24·365 / H)` is
  reported alongside but the model works in RV (and internally in `log RV` for
  positivity/stability).
- **Supervised construction.** Given lookback `L`, horizon `H`, stride `S`
  (defaults `L=72`, `H=24`, `S=6` hours): for each anchor `t` on the grid with a
  full `L`-window before and a full `H`-window after,
  - features `x_t` are computed **only** from prices/returns in `(t−L, t]`;
  - target `y_t = RV` over `(t, t+H]`.
  Strict temporal separation → **no label leakage** (enforced by a test, § 4.9).
- **Regime.** Tertiles (33rd / 66th percentiles) of `y` over the **training** split
  define thresholds `→ calm / normal / turbulent`. Thresholds are stored in the
  model bundle and reused at predict time. (Percentiles configurable.)

### 4.2 Features (tabular), from the lookback window

Autoregressive-vol and shape features (all derivable offline, deterministic):

| Feature | Definition |
|---|---|
| `rv_lookback` | RV over the full L-window — the dominant predictor |
| `rv_recent`   | RV over the last H hours of the lookback (near-term vol) |
| `vol_of_vol`  | std of rolling short-window RVs within L |
| `mean_return`, `std_return` | mean / std of hourly returns |
| `downside_vol` | std of negative returns only |
| `mean_abs_return` | mean \|r\| |
| `return_skew`, `return_kurt` | 3rd / 4th standardized moments of returns |
| `acf1` | lag-1 autocorrelation of returns |
| `max_drawdown_pct` | peak-to-trough decline over L (reuses `prices._max_drawdown_pct`) |
| `momentum` | `p_t / p_{t−L} − 1` |
| `range_pct` | `(max − min) / mean` price over L |
| `last_return` | most recent hourly return |

**Optional exogenous (news) features**, from the vector store's metadata, for
`(t−L, t]`: `news_count`, `news_count_recent` (last H), `news_recency_hours`,
`regulator_count`. Default to `0` when no corpus / not provided. *Honest note:* the
corpus is a rolling recent window (retention), so historical news features are
mostly unavailable for long training runs — these features matter most **at predict
time** and in the S12 stitch, not necessarily as a large training signal. The
architecture supports them; the spec does not oversell them.

The **sequence model** additionally consumes the raw standardized return sequence of
the L-window (and its squared-returns channel), not just the tabular vector.

### 4.3 Models (unified `Forecaster` interface)

All expose `fit(X, y)`, `predict(X) -> RV`, `name`, `save(dir)`, `load(dir)`, and
work in `log RV` internally.

| Model | Class | Backend | Role |
|---|---|---|---|
| Persistence baseline | `PersistenceBaseline` | none | `ŷ = rv_lookback`. The honest yardstick; skill is measured against it. |
| Linear/GBT (sklearn) | `SklearnForecaster` | `scikit-learn` | `HistGradientBoostingRegressor` (no extra dep) in a standardizing pipeline. |
| Boosted trees | `GBMForecaster` | `xgboost` / `lightgbm` | Tabular workhorse; logs feature importances. |
| Sequence | `SequenceForecaster` | `torch` | 1–2-layer LSTM (hidden 32–64) → MLP head; early stopping on val. |

Heavy backends stay optional (§ 4.7); the baseline + sklearn path runs on the
lighter `forecast` extra alone.

### 4.4 Training & evaluation protocol

- **Temporal split**, never shuffled: chronological train / val / test (default
  70 / 15 / 15). Val selects the model + early stopping; test is reported once.
- **Metrics** (per model, side by side): MAE, RMSE, R² on RV; `QLIKE` (a
  vol-appropriate loss); **skill score** `= 1 − MSE_model / MSE_baseline`; and
  regime **accuracy / macro-F1** from bucketed predictions.
- **Selection**: best val skill → persisted as the default bundle; the full
  comparison table is written to `metrics.json`.
- **Determinism**: fixed seeds; the whole path runs offline on a synthetic series
  (§ 4.9) with no network.

### 4.5 Persistence — model bundle

Each trained model is a self-describing bundle under `data/models/<asset>/<name>/`:

```
metadata.json   # model_name, asset, lookback/horizon/stride, feature_names,
                # regime_thresholds, standardizer stats, train window (from/to),
                # metrics{mae,rmse,r2,qlike,skill,regime_f1}, seed, git_sha, created_at
model.*         # sklearn/xgb/lgbm via joblib or native; torch via state_dict + config.json
```

`save_bundle(...)` / `load_bundle(...)` are the seam; `metadata.json` is the single
source of truth so `predict` reproduces the exact feature/regime setup used in
training. (This bundle is what S11 registers in MLflow.)

### 4.6 New data model (`models.py`)

```python
class VolForecast(BaseModel):
    asset: str                       # "ETH"
    as_of: datetime                  # UTC — end of the lookback window
    lookback_hours: int
    horizon_hours: int
    predicted_vol: float             # H-hour realized vol (RV)
    predicted_vol_annualized: float
    regime: Literal["calm", "normal", "turbulent"]
    regime_thresholds: dict          # {"calm_max": .., "turbulent_min": ..}
    model_name: str                  # "xgboost" | "lstm" | ...
    skill_vs_baseline: float | None  # from the trained bundle's test metrics
    trained_at: datetime
    drivers: dict = {}               # top feature contributions (interpretability)
    notes: list[str] = []            # always includes the not-investment-advice line
```

### 4.7 Config & packaging deltas

`config.py` (new settings, all with defaults so offline commands still need no keys):

```python
models_path: Path = Path("data/models")
forecast_lookback_hours: int = 72
forecast_horizon_hours: int = 24
forecast_stride_hours: int = 6
forecast_history_days: int = 90       # CoinGecko free-tier hourly ceiling (Risk R1)
regime_low_pct: float = 33.0
regime_high_pct: float = 66.0
```

`pyproject.toml` optional extras (**core stays torch-free**, per the repo ethos):

```toml
forecast = ["numpy", "scikit-learn", "joblib"]   # baseline + sklearn + features
gbm      = ["xgboost", "lightgbm"]               # boosted trees
dl       = ["torch"]                             # sequence model
```

> **Verified 2026-08-26:** numpy, pandas, scikit-learn, xgboost, lightgbm, duckdb,
> **and torch** all `pip install` and import cleanly on **Python 3.14** into the
> repo's `.venv`. This relaxes the ARCHITECTURE.md as-built note that torch would
> not install under this repo's path — re-checked and it now does. Torch is still
> kept in the optional `[dl]` extra so `pip install -e .` and CI stay torch-free.

### 4.8 CLI surface

```bash
# Train + compare models for an asset (fetches history, or reads an offline CSV)
crypto-intel train --asset ETH --history-days 90 \
    --lookback 72 --horizon 24 --stride 6 \
    --models baseline,sklearn,xgboost,lstm \
    [--offline path/to/series.csv]      # fully reproducible / test path
# → prints a comparison table (MAE/RMSE/skill/regime-F1), writes metrics.json,
#   persists the best bundle to data/models/ETH/<name>/

# Predict next-window volatility / regime from the latest data
crypto-intel predict --asset ETH [--as-of 2026-08-20T00:00Z] [--model xgboost]
# → prints a VolForecast: regime, predicted RV (+ annualized), top drivers,
#   skill-vs-baseline, and the not-investment-advice line
```

### 4.9 Module map & tests (S9)

New package `crypto_intel/forecast/`:

- `features.py` — **pure**: grid resample, log returns, `realized_vol`,
  `window_features`, `build_supervised(series, L, H, S, news_fn=None) ->
  (X, y, seqs, feature_names, index_ts)`. No network, no heavy deps beyond numpy.
- `dataset.py` — `fetch_price_history(asset, days)` (chunked CoinGecko range calls
  via the existing `CoinGeckoClient`), `news_feature_fn(store)`, `load_series_csv`.
- `models.py` — the four `Forecaster`s + `save_bundle` / `load_bundle`. Heavy
  imports are lazy/guarded.
- `train.py` — orchestration: build dataset → temporal split → fit/evaluate/compare
  → select → persist; returns a metrics dict.
- `predict.py` — load bundle → fetch/compute lookback features → `VolForecast`.

CLI: add `train` and `predict` to `cli.py` (lazy imports inside the commands, per
the existing pattern). `models.py` gains `VolForecast`.

Tests (offline, deterministic — matching the existing style):

- `test_forecast_features.py` — RV/return formulas on a known series; feature-vector
  shape & values on a synthetic sinusoid+noise; supervised window counts; **the
  no-leakage assertion** (targets depend only on `>t`, features only on `≤t`); gap
  handling.
- `test_forecast_models.py` — baseline predicts `rv_lookback`; sklearn beats the
  baseline on a constructed signal; `save_bundle`/`load_bundle` round-trip;
  regime thresholds monotonic. XGBoost/torch tests behind `pytest.importorskip`.
- `test_forecast_cli.py` — `train --offline synthetic.csv` runs and writes
  `metrics.json`; `predict` (stub client / prebuilt bundle) yields a `VolForecast`
  carrying a regime **and** the not-investment-advice note.

### 4.10 S9 acceptance criteria

1. `pip install -e ".[forecast]"` then `crypto-intel train --asset ETH
   --offline <csv> --models baseline,sklearn` trains and prints a comparison table
   with a **skill score vs. baseline**.
2. `crypto-intel predict --asset ETH` (with a trained bundle) prints a `VolForecast`
   with a regime and the not-investment-advice line.
3. New offline tests pass; the **existing suite stays green**; `pip install -e .`
   (no extras) still imports and runs the RAG CLI (forecast deps absent → the
   `train`/`predict` commands print a clear "install .[forecast]" notice).
4. XGBoost and LSTM paths run when their extras are installed and are skipped
   cleanly when they are not.

---

## 5. S10 — Scale: warehouse-backed features (**built**)

- **DuckDB (embedded default).** `crypto_intel/warehouse/duck.py` — `DuckWarehouse`
  lands price history + document metadata into `prices` / `documents` tables and
  engineers features in **SQL**: hourly gridding (`time_bucket` +
  `last(... ORDER BY ...)`), rolling window functions (log return, rolling realized
  vol, rolling mean return) materialized into a queryable `price_features` table,
  and per-hour news aggregation (`GROUP BY`). So "large-scale pipeline" is backed by
  real tooling, not a claim.
- **`train --source warehouse` genuinely consumes SQL output**: it reads the
  SQL-gridded price series and the SQL-aggregated hourly news counts back out
  (`warehouse_news_fn`) and feeds them into the proven S9 windower — no logic
  duplicated, no divergence. Verified to reproduce the direct-path skill numbers.
- **BigQuery (optional).** `warehouse/bq.py` behind a `[bq]` extra + a service
  account (`GOOGLE_APPLICATION_CREDENTIALS`); `warehouse build --dest bigquery`
  loads the same schema into a BigQuery free-tier dataset (`bq_project`/`bq_dataset`),
  making the "cloud warehouse" line literally true. Local/CI default stays DuckDB
  (no GCP auth). The pure row-mappers (`price_rows`/`document_rows`) are unit-tested;
  the network load path is guarded and exercised manually.
- **CLI**: `warehouse build --asset ETH [--offline csv] [--dest duckdb|bigquery]
  [--rolling-hours N]` and `warehouse stats [--asset ETH]`.
- **Tests** (8, offline): grid round-trip, the SQL window-function feature build,
  news aggregation, asset filtering, an end-to-end warehouse→train run, and the BQ
  row-mappers. **Config**: `warehouse_path`, `bq_project`, `bq_dataset`.

## 6. S11 — MLOps loop (**built**) — *highest-leverage half*

- **MLflow tracking + registry.** `mlops/tracking.py` — `train --track` logs params
  (L/H/S, models, best), per-model val/test metrics (MAE/RMSE/skill/regime-F1), and
  the model **bundle** as artifacts, then registers a servable `pyfunc` version
  `crypto-intel-vol-<ASSET>`. `MLFLOW_TRACKING_URI` defaults to a local **SQLite**
  backend (`sqlite:///data/mlflow.db`) — MLflow 3.x deprecated the file store and
  the registry needs a DB, so SQLite is the zero-infra default; a remote URI is
  configurable. Tracking is **best-effort** (a hiccup warns, never fails training).
  Promotion to a `serving` alias is a manual, reviewed step. *Boundary:* the pyfunc
  serves tabular forecasters (baseline/sklearn/xgboost/lightgbm); a sequence (LSTM)
  best model is logged + tracked but not registered as a tabular pyfunc.
- **Serving.** The existing FastAPI app gains `GET /api/forecast?asset=ETH[&model=]`
  → `VolForecast` JSON (404 untrained, 503 if forecast extras absent) and
  `GET /monitoring?asset=ETH` serving the drift HTML — reusing the current
  Docker/Railway deployment so it ships live and clickable.
- **CI/CD (GitHub Actions).** `.github/workflows/ci.yml`: on push/PR — install
  `.[dev,forecast,gbm,warehouse]`, `ruff` (real-error select) + full `pytest` on a
  **py3.11/3.12/3.13** matrix (torch-free; LSTM/MLflow/Evidently paths importorskip).
  `.github/workflows/train.yml`: `workflow_dispatch` (asset input) + weekly cron —
  generate a deterministic sample, `train --track`, upload `metrics.json` + `mlruns`
  as artifacts. No auto-promotion.
- **Monitoring.** `mlops/monitor.py` — training snapshots its feature distribution
  into the bundle (`reference_features.csv`); `crypto-intel monitor --asset ETH`
  builds an **Evidently** data-drift report (reference = training window vs. current)
  → static HTML served at `/monitoring`. Drifted-column count + share summarized to
  the terminal.
- **Config**: `mlflow_tracking_uri`, `mlflow_experiment`, `mlflow_registry_prefix`,
  `monitoring_path`. **Extras**: `[mlops]` (mlflow), `[monitor]` (evidently+pandas).
- **Tests** (7, offline): reference round-trip, MLflow run+registry against a temp
  SQLite store, an Evidently HTML report, current-feature geometry, and the
  `/api/forecast` + `/monitoring` endpoints via `TestClient`.

> **Verified 2026-08-26:** mlflow 3.15 and evidently 0.7 both install and run on
> **Python 3.14**. Default tracking is SQLite (not the deprecated file store).

## 7. S12 — GenAI stitch (**built**)

- **Regime woven into synthesis.** `pipeline.ask` computes a best-effort
  `VolForecast` for the event's asset (`_maybe_forecast`, never raises) and passes a
  one-line market-state string into `synthesize.py` via a new `forecast_context`
  parameter. `SYSTEM_PROMPT` gains a rule: the regime line is **background on how
  turbulent conditions are — a volatility estimate, not a source, not a price
  prediction, never cited or turned into advice.** Citations and the
  not-investment-advice line are unchanged.
- **Regime banner + structured field.** `ask` prints a one-line banner
  (`Regime : ◐ NORMAL (68% ann. vol, model xgboost, skill +0.90)`) above the answer,
  and the regime is attached to `Answer.market_state`
  `{regime, model_name, skill_vs_baseline, predicted_vol_annualized}`. A `--no-regime`
  flag (and `ask(..., with_regime=False)`) disables it; it also self-skips silently
  when no model is trained for the asset (no extra network call in that case).
- **Explicit stack table** naming the full RAG+ML surface is in the
  [README](../README.md#tech-stack-at-a-glance) so it is CV-legible in one glance.
- **Tests** (7, offline): the market-state string is background-not-advice, the
  prompt weaving, the `SYSTEM_PROMPT` guardrail, a recording-client synthesis check
  (regime present in the prompt, not citable), and `ask` attaching `market_state`
  (and skipping it under `--no-regime`).

**The ML expansion is complete.** All four CV gaps from the brief (§ 1) are closed:
PyTorch/XGBoost/sklearn (S9), warehouse + SQL feature engineering (S10), the
MLflow/CI/monitoring MLOps loop (S11), and the explicit GenAI↔ML product (S12).

---

## 8. Risks & open decisions

- **R1 — CoinGecko granularity.** The free `market_chart/range` endpoint returns
  hourly data only within ~90 days (5-minutely ≤1 day, daily >90 days). Default
  `history_days = 90` keeps hourly resolution; longer histories degrade to daily and
  reinterpret L/H/S in samples. *Decision:* default to 90d hourly; document the
  degradation; an offline CSV path (`--offline`) removes the ceiling for
  reproducible/large training and for tests.
- **R2 — Small-data overfitting.** ~2,000 hourly points at 90d. Mitigations: a
  strong persistence baseline (report **skill**, not raw error), temporal
  validation, regularization, LSTM early stopping. *We report skill honestly — a
  modest positive skill over persistence is a legitimate, defensible result and is
  not oversold.*
- **R3 — News features for training.** Retention keeps only a recent corpus, so
  historical news features are largely unavailable for long training. *Decision:*
  treat exogenous news features as optional (default 0), leaning on them at predict
  time and in the S12 stitch rather than claiming them as a training driver.
- **R4 — Torch on Windows / CI weight.** Torch now installs on 3.14 into the repo
  venv (§ 4.7), but stays optional (`[dl]`) so core installs and CI remain
  torch-free; the DL path is exercised locally and in an optional CI job.
- **R5 — Framing drift.** Any wording that implies a directional/price call is a
  spec violation (§ 1.1). Reviewer checklist: every forecast output must keep the
  not-investment-advice line and speak only in volatility/regime terms.

## 9. Guardrail (reaffirmed)

The ML layer forecasts **volatility and risk regime only** — never price direction,
trade signals, or personalized advice. No phase adds order routing or portfolio
logic. Every `predict`/`forecast`/`ask` output retains the not-investment-advice
line. This is a decision-support and demonstration system, not a trading system.
