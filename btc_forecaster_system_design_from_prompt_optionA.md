# State-of-the-Art BTC Forecaster — System Design Doc (Built from the “Principal ML Researcher + Quant Engineer” Spec)

This document **embeds the full build spec you provided** and then **implements it as a concrete system design**, selecting:

- **Data option:** **Option A — Binance Bulk Spot OHLCV (BTCUSDT, 1h)**  
- **Primary modeling path:** strong tabular baseline (LightGBM quantiles) + volatility baseline (GARCH) + one deep baseline (N-BEATS/LSTM) + one SOTA candidate (PatchTST/TFT-style) when ready  
- **Evaluation:** strict walk-forward, leakage-proof feature availability contract, probabilistic outputs, calibration

---

## 1) Embedded build spec (verbatim)

You are a **principal ML researcher + quant engineer** building a **state-of-the-art Bitcoin (BTC) price forecasting system** that is scientifically rigorous, leakage-proof, and production-ready.

### 0) Hard rules (must follow)
- **No lookahead/leakage**: every feature must be available at prediction time, with explicit timestamps + publication delays.
- **Walk-forward evaluation only** (anchored expanding or rolling). No random splits.
- **Probabilistic forecasts** required (quantiles or full distribution), not just point estimates.
- **Reproducible**: fixed seeds, deterministic pipelines where possible, config-driven runs.
- **Separation of concerns**: forecasting quality ≠ trading profitability. Provide both, but keep them distinct.
- If any requested data isn’t available, propose **drop-in substitutes** and keep the pipeline modular.

### 1) Objective
Build a BTC forecaster that predicts future price/returns over multiple horizons with uncertainty:
- Horizons: {1h, 4h, 1d, 3d, 7d} (adjustable)
- Targets (you choose best practice and justify):
  - log returns, future volatility, and/or price levels with appropriate transforms
- Output:
  - Point forecast + calibrated prediction intervals (e.g., P10/P50/P90)
  - Optional: density forecast (mixture/logistic, Student-t, or quantile set)

### 2) Inputs / Data (design for multiple tiers)
Assume we can start with **OHLCV** and optionally add:
- Market microstructure: funding rates, open interest, liquidation data, perp basis
- Order book / trades (if available): imbalance, spread, depth, VPIN proxies
- On-chain: active addresses, exchange inflow/outflow, realized cap metrics
- Macro/rates/risk: DXY, 2Y/10Y yields, S&P, Nasdaq, gold, oil, VIX
- Calendar: day-of-week, session effects, holidays
You must:
- Define **data schema** with timestamps, timezones, and “available_at” times for each feature.
- Implement a **feature availability contract** to guarantee point-in-time correctness.

### 3) Research plan (what you must deliver)
Provide a complete plan and then execute it in code:
A) Problem framing
- Choose target(s) and justify (returns vs price, stationarity, scaling).
- Define what “good” means: forecasting metrics + calibration + stability.

B) Baselines (must include)
- Naive: last value / random walk
- ARIMA/SARIMAX or ETS (where appropriate)
- GARCH family for volatility (if modeling vol)
- Tree model: LightGBM/XGBoost with carefully constructed lags
- A simple deep baseline: LSTM/GRU or N-BEATS

C) SOTA candidates (choose 1–2 primary, 1 secondary)
Pick models appropriate for time-series forecasting and justify:
- Temporal Fusion Transformer (TFT) / PatchTST / Informer-like
- N-HiTS / N-BEATSx
- DeepAR / diffusion-based time series (only if you can evaluate rigorously)
- Hybrid: tree model on engineered features + deep residual
Your choice must include:
- How you handle multi-horizon forecasting (direct, seq2seq, multi-output head).
- How you produce uncertainty (quantile regression, distributional head, conformal).

D) Feature engineering (strictly causal)
- Lags, rolling stats, realized volatility, momentum, drawdown, range measures
- Regime features: volatility regime, trend regime, volume regime
- Cross-asset features (if used) with correct alignments + delays
- Make a table listing each feature, its formula, and “available_at”.

E) Training & validation
- Walk-forward scheme details (window sizes, step sizes, purge/embargo if needed).
- Hyperparameter search strategy that avoids overfitting:
  - nested walk-forward or prequential tuning
  - limited trials + early stopping + Bayesian optimization if feasible
- Regularization and robustness:
  - dropout/weight decay, label smoothing if classification, noise injection
  - sample weighting (recent data emphasis) with justification

F) Metrics & diagnostics (must include)
Forecast metrics:
- MAE/RMSE on returns and/or price (after inverse transform)
- MASE or sMAPE where appropriate
- Directional accuracy + balanced accuracy (up/down)
- Quantile loss (pinball loss) for P10/P50/P90
- Calibration: coverage of prediction intervals, reliability curves
- Stability: performance by regime (high vol vs low vol), by year/quarter
Leakage checks:
- “Time travel” tests: intentionally shift features forward to confirm degradation
- Feature importance sanity checks

G) Trading evaluation (optional but separate)
- Define a simple strategy using forecasts (e.g., thresholded expected return / risk-adjusted signal)
- Include realistic assumptions: fees, slippage, spread proxy
- Evaluate with walk-forward only; report Sharpe, max drawdown, turnover
- Emphasize this is not financial advice; it’s an engineering eval.

H) Productionization
- Inference pipeline: data fetch → feature build → predict → store
- Model registry + versioning + configs
- Monitoring: drift (PSI), calibration drift, performance decay
- Retraining schedule trigger rules
- Clear API interface (Python function + optional FastAPI endpoint)

### 4) Deliverables (you must output all of these)
1) A concise architecture diagram description (text is fine).
2) A runnable Python project layout (folders, modules).
3) End-to-end code:
   - data ingestion (use CCXT or provide an offline CSV loader if internet unavailable)
   - feature pipeline (causal, tested)
   - training script with walk-forward validation
   - evaluation notebook/script producing metric tables + plots
   - inference script that outputs forecasts + intervals
4) A “readme” explaining how to run training and inference.
5) A section titled “Failure modes & how we mitigated them”.

### 5) Implementation constraints
- Use Python.
- Prefer: pandas, numpy, scikit-learn, lightgbm/xgboost, pytorch (or pytorch-forecasting).
- No hidden dependencies. Provide `requirements.txt`.
- Keep runtime reasonable on a single machine; offer “small/medium/large” configs.

### 6) Questions (only ask if truly necessary)
If you need assumptions, default them. Only ask if the choice is make-or-break:
- Data frequency (default: 1h bars)
- Exchange (default: Coinbase or Binance spot)
- Forecast horizon set (default above)

### 7) Output format
Return:
- A step-by-step plan (bulleted)
- Then the full code (multiple files separated with clear file headers)
- Then evaluation guidance + what “good” results look like
- Be explicit about any limitations and how to improve further.

Begin now.

---

## 2) Concrete design decisions for this build

### 2.1 Chosen data tier (Option A)
**Tier 0 (MVP):** Binance Bulk **Spot** OHLCV klines for **BTCUSDT**, frequency **1h**.  
Rationale:
- Free, very large history, no REST pagination rate-limit pain
- Clean reproducibility by pinning ZIP files + checksums
- Enough to build strong baselines and many “SOTA-ish” features (realized vol, regimes)

**Planned later tiers (still modular):**
- Tier 1: add Binance bulk derivatives signals (funding, mark/index price klines, etc.)
- Tier 2: add on-chain and macro (only if desired; requires extra providers)

---

## 3) Architecture diagram (text)

**Bulk ZIPs (raw)** → **Integrity verify (checksum)** → **Unzip + canonicalize (UTC + schema)** →  
**Feature builder (causal + `available_at` contract)** → **Walk-forward trainer (models + tuning)** →  
**Evaluator (metrics + calibration + regime slices + leakage tests)** →  
**Inference runner (predict + intervals)** → **Artifact store (parquet/json + reports)** →  
**Monitoring (drift + coverage + decay) + retraining triggers**

---

## 4) Data contract (schema + availability)

### 4.1 Canonical candle schema
`candles_btcusdt_1h`:
- `ts_utc` (datetime64[ns, UTC]): candle start time
- `open`, `high`, `low`, `close`, `volume` (float64)
- `symbol` (str) = `BTCUSDT`
- `freq` (str) = `1h`
- `source` (str) = `binance_bulk_spot`
- `ingested_at_utc` (datetime64[ns, UTC])
- `data_version` (str)
- `available_at_utc` (datetime64[ns, UTC]) = `close_time + latency_buffer` (default buffer: 2 minutes)

### 4.2 Feature availability contract
Every feature row includes:
- `feature_ts_utc`
- `available_at_utc` (the earliest time this feature is safe to use)
- `feature_name`
- `feature_value`
- `source_columns` (optional metadata)

**Rule enforced everywhere:** at prediction time `T`, the model may only use features with `available_at_utc <= T`.

---

## 5) Feature table (examples)

Assume candle index `t` is the candle starting at `ts_utc = t`. Candle closes at `t + 1h`.

| Feature | Formula (uses only history) | available_at_utc |
|---|---|---|
| `ret_1` | `log(close_t) - log(close_{t-1})` | `close_time_t + buffer` |
| `rv_24` | `sqrt(sum(ret_1^2) over last 24 hours)` | `close_time_t + buffer` |
| `mom_24` | `log(close_t / close_{t-24})` | `close_time_t + buffer` |
| `range_parkinson_24` | Parkinson estimator using OHLC in 24h window | `close_time_t + buffer` |
| `vol_z_72` | z-score of volume over last 72 hours | `close_time_t + buffer` |
| `trend_regime` | bucketed signal from EMA cross (e.g., EMA24-EMA168) | `close_time_t + buffer` |
| `vol_regime` | rolling volatility quantile bucket | `close_time_t + buffer` |

> NOTE: All of these are candle-derived, so they share the same availability timing convention.

---

## 6) Targets (recommended) + multi-horizon setup

### 6.1 Target choice
Primary target: **future log return** for each horizon `h`:
- `y_h(t) = log(close_{t+h} / close_t)`

Why:
- More stationary than raw price
- Easier for models to learn and calibrate
- Converts naturally to trading signals later (but evaluation remains separate)

Optional secondary target:
- Future realized volatility over horizon `h`

### 6.2 Multi-horizon training
Default approach:
- **Separate direct models** per horizon (`h ∈ {1h,4h,1d,3d,7d}`), because:
  - simpler debugging
  - avoids leakage mistakes from sequence-to-sequence alignment
  - strong baseline that often wins in practice

Later:
- a multi-output head (single model producing all horizons) if it improves consistency.

---

## 7) Models (baseline → SOTA)

### Must-have baselines
- Naive random walk (for returns, predict 0)
- ARIMA/ETS where appropriate
- GARCH family for volatility (if vol target is used)
- LightGBM quantile regression (P10/P50/P90)
- Simple deep baseline: LSTM/GRU or N-BEATS

### SOTA candidates (when ready)
- PatchTST or TFT-style model for multi-horizon forecasting
- Uncertainty via quantile head + conformal calibration

---

## 8) Training + validation

### Walk-forward (default)
- Training window: 18–36 months
- Step: 1 week (or 1 month for faster iteration)
- Holdout test: final 3–6 months

### Hyperparameter tuning without overfitting
- Prequential tuning on early folds only
- Limited trials + early stopping
- If using Bayesian optimization, keep search spaces narrow and document them

---

## 9) Metrics + diagnostics

### Forecast metrics
- MAE/RMSE on returns (and reconstructed price if needed)
- Directional accuracy + balanced accuracy
- Pinball loss for P10/P50/P90
- Interval coverage for P10–P90 (nominal 80%)
- Reliability curve (coverage vs nominal)
- Regime-sliced scores (high-vol vs low-vol; trend vs chop)

### Leakage checks
- Feature shift (“time travel”) test: shift features forward → performance must collapse
- Sanity: feature importance and correlation checks

---

## 10) Trading evaluation (optional, separate)

- Define a minimal strategy using forecasts (e.g., go long if P50 return > threshold and interval excludes 0)
- Include fees + slippage proxy
- Report Sharpe, drawdown, turnover on walk-forward only
- Clearly label as engineering evaluation, not advice

---

## 11) Productionization

### Inference pipeline
`download/refresh data → canonicalize → build features → predict (quantiles) → store outputs`

### Model registry + versioning
- `model_id`, `data_version`, `feature_version`, `config_hash`, `git_commit`

### Monitoring
- drift (PSI) on key features
- coverage drift (interval coverage over rolling windows)
- performance decay (rolling pinball loss)

### Retraining triggers
- scheduled (weekly/monthly)
- conditional (coverage drift or PSI thresholds)

---

## 12) Runnable project layout (target)

```
btc-forecaster/
  configs/
    small.yaml
    medium.yaml
    large.yaml
  data/
    raw/binance/...
    processed/btcusdt_1h.parquet
    features/features_btcusdt_1h.parquet
  src/
    data/
      download_binance_bulk.py
      verify_checksums.py
      build_canonical.py
    features/
      build_features.py
      feature_contract.py
    models/
      baselines.py
      lgbm_quantiles.py
      garch.py
      deep_nbeats.py
    eval/
      walk_forward.py
      metrics.py
      calibration.py
      leakage_tests.py
      regime_slices.py
    inference/
      predict.py
  outputs/
    forecasts/
    reports/
  requirements.txt
  README.md
```

---

## 13) Failure modes & mitigations

1) **Timestamp unit mismatch (ms vs µs)**  
   - Detect scale at load time; assert monotonic time; log anomalies; fail fast.

2) **Hidden lookahead in feature engineering**  
   - Enforce `available_at` contract everywhere; write tests; embargo if needed.

3) **Overfitting from tuning**  
   - Limit trials; nested/prequential tuning; keep validation strictly forward.

4) **Regime shifts**  
   - Monitor drift; recalibrate intervals; rolling retraining; regime-sliced reporting.

5) **Data gaps / duplicates**  
   - Continuity checks; dedupe; report and optionally exclude affected windows.

---

## 14) Step-by-step implementation plan

1) **Acquire data (Option A)**: download Binance bulk spot klines for BTCUSDT 1h + checksums  
2) Verify checksums; unzip; stage in `data/raw/binance/`  
3) Build canonical candles parquet with strict UTC timestamps and metadata  
4) Implement feature builder with `available_at` contract  
5) Implement walk-forward splitter + leakage checks  
6) Train baselines (naive, ARIMA, LightGBM quantiles)  
7) Add deep baseline (N-BEATS/LSTM) + one SOTA candidate (PatchTST/TFT)  
8) Evaluate: pinball loss, coverage, reliability, regime slices  
9) Add conformal calibration for robust intervals  
10) Build inference script + output artifacts  
11) Add monitoring hooks + retraining triggers

---

## 15) What “good” looks like (practical targets)

- **Beats random-walk** on pinball loss for most horizons, consistently across folds  
- **Interval coverage** close to nominal (P10–P90 ≈ 80% coverage) with stable drift  
- **Regime robustness:** doesn’t collapse completely in high-vol windows  
- **Reproducibility:** re-running with same `data_version + config_hash` yields same metrics

---

### Appendix A — The “LLM build prompt” as a reusable artifact
If you want to paste this into an LLM again, you can use **Section 1** above as the “instruction block”, and then attach this doc as the “system design + decisions” artifact.
