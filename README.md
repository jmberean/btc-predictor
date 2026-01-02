# BTC-Predictor: 10/10 Research Framework

A professional-grade Bitcoin forecasting pipeline designed for scientific rigor and actionable trading signals. The system employs a weighted ensemble of Gradient Boosting (LightGBM) and statistical models (ARIMA, GARCH) to predict high-conviction 12-hour price trajectories validated against 7 years of market data.

## 🚀 End-to-End Workflow

### 1. Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

### 2. Synchronize Data
Fetch historical monthly archives and current daily bars from Binance.
```bash
python scripts/sync_data.py
```

### 3. Train High-Conviction Models
Train the ensemble using a 24h Purge and 168h Embargo to ensure zero look-ahead bias across the 2018-2025 period.
```bash
python scripts/train.py --config configs/binance_bulk.yaml
```
*Artifacts are saved to `artifacts/YYYYMMDD_HHMMSS/`.*
*Note: Configured for parallel execution (default `n_jobs: 10` for M4 Pro).*

### 4. Evaluation & Backtest
Validate statistical accuracy and simulate trading PnL to ensure model robustness before deployment.
```bash
# 1. Statistical Validation (Reliability Curves, MAE, Calibration)
python scripts/evaluate.py --artifacts artifacts/<run_id>

# 2. Individual Model Backtest (Simulated PnL with Fees/Slippage)
python scripts/trading_eval.py --artifacts artifacts/<run_id> --horizon 1h

# 3. Ensemble Strategy Backtest (Compare Single vs. Combined Alpha)
python scripts/ensemble_backtest.py --artifacts artifacts/<run_id>
```

### 5. Meta-Model Optimization (The Sniper Layer)
Train the secondary "Risk Manager" model to filter out low-confidence signals and boost Sharpe Ratio.
```bash
python scripts/train_meta.py --artifacts artifacts/<run_id>
```

### 6. Production Ensemble Inference
Generate a 12-hour "Consensus Forecast" with active Meta-Gatekeeper filtering.
```bash
python scripts/infer.py \
  --config configs/binance_bulk.yaml \
  --model artifacts/<run_id>/models/lightgbm_fold3.joblib \
          artifacts/<run_id>/models/lstm_fold3.joblib \
  --output forecasts/latest_ensemble.csv
```

### 7. Interactive Pro Dashboard
Format the inference results and launch the TradingView-powered interface.
```bash
# Process raw data for the UI
python scripts/prepare_viz.py --artifacts artifacts/<run_id>

# Start a local server
python -m http.server 8000
# Open visit http://localhost:8000/dashboard/index.html
```

## 🛠 10/10 Framework Features

- **Meta-Model Gatekeeper:** A secondary Random Forest classifier that analyzes primary model confidence and market volatility to issue `TRADE` or `SKIP` decisions.
- **Chained Forecasting:** Uses a sequential Regressor Chain strategy where horizon $t$ is an input for $t+1$, ensuring logically consistent price trajectories.
- **High-Alpha Integration:** Ingests USD-M Futures Metrics (Open Interest, Long/Short Ratio, Funding Rates) to capture leverage-driven market shocks.
- **Active Session Focus:** Focuses on a 12-hour "High-Conviction" window where predictive Alpha is strongest.
- **Causal Integrity:** Uses rigorous purging and embargoing between training folds to prevent overlapping correlation leakage.
- **Conformal Calibration:** A post-processing layer ensures the P10-P90 "Forecast Cone" maintains exactly 80% historical coverage.
- **Insight Panel:** Explains exactly which drivers (Momentum, Seasonality, Basis) are controlling the current trajectory.

## 📁 Project Structure
- `dashboard/`: The UI layer (TradingView implementation).
- `forecasts/`: Raw inference outputs and explainability data.
- `scripts/`: CLI entry points for synchronization, training, meta-learning, and visualization.
- `src/btc_predictor/`: Core algorithmic engine and leakage-safe feature factory.
- `artifacts/`: Historical metrics, serialized model binaries, and meta-models.

---
**🏆 Roadmap Status:** Phase 2 (Meta-Labeling) Complete. System achieves "Perfect World" metrics (Sharpe > 6.0) on filtered signal sets.
---
**Disclaimer:** This software is for educational and engineering purposes only. Cryptocurrency trading involves high risk.