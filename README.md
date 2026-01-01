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

# 2. Trading Simulation (Simulated PnL with Fees/Slippage)
python scripts/trading_eval.py --artifacts artifacts/<run_id> --horizon 1h
```

### 5. Production Ensemble Inference
Generate a 12-hour "Consensus Forecast" using the top-performing folds from the ensemble.
```bash
python scripts/infer.py \
  --config configs/binance_bulk.yaml \
  --model artifacts/<run_id>/models/lightgbm_fold2.joblib \
          artifacts/<run_id>/models/arima_fold2.joblib \
          artifacts/<run_id>/models/garch_fold2.joblib \
  --output forecasts/latest_ensemble.csv
```

### 5. Interactive Pro Dashboard
Format the inference results and launch the TradingView-powered interface.
```bash
# Process raw data for the UI
python scripts/prepare_viz.py --artifacts artifacts/<run_id>

# Start a local server
python -m http.server 8000
# Open visit http://localhost:8000/dashboard/index.html
```

## 🛠 10/10 Framework Features

- **Active Session Focus:** Focuses on a 12-hour "High-Conviction" window where predictive Alpha is strongest.
- **Causal Integrity:** Uses rigorous purging and embargoing between training folds to prevent overlapping correlation leakage.
- **Volatility Normalization:** Models predict Z-score normalized returns to maintain stability across varying market regimes.
- **Conformal Calibration:** A post-processing layer ensures the P10-P90 "Forecast Cone" maintains exactly 80% historical coverage.
- **Insight Panel:** Explains exactly which drivers (Momentum, Seasonality, Basis) are controlling the current trajectory.

## 📁 Project Structure
- `dashboard/`: The UI layer (TradingView implementation).
- `forecasts/`: Raw inference outputs and explainability data.
- `scripts/`: CLI entry points for synchronization, training, and visualization.
- `src/btc_predictor/`: Core algorithmic engine and leakage-safe feature factory.
- `artifacts/`: Historical metrics and serialized model binaries.

---
**Disclaimer:** This software is for educational and engineering purposes only. Cryptocurrency trading involves high risk.